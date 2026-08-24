"""F2 -- multi-object tracking. ByteTrack and BoT-SORT, both wired.

BoT-SORT is the default. Published comparisons give it the best IDF1 and the
lowest ID-switch count, and identity stability through fence and tree occlusion
is what this application actually needs. ByteTrack remains the throughput pick
and is one config key away.

One tracker instance per camera, always. The reference prototype we assessed
kept tracker state inside a single global model object, which is safe only
while exactly one stream is processed at a time -- two concurrent cameras would
cross-contaminate track identities.
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import numpy as np

from ..config import TrackerCfg
from ..contracts import Detection, ObjectGroup, Track, TrackPoint
from .detector import ClassMapper, DetBatch


class ObjectTracker:
    """Wraps an ultralytics tracker and returns our own ``Track`` objects."""

    def __init__(self, cfg: TrackerCfg, camera_id: str, mapper: ClassMapper,
                 class_names: dict[int, str]) -> None:
        from ultralytics.trackers import BOTSORT, BYTETracker

        self.cfg = cfg
        self.camera_id = camera_id
        self.mapper = mapper
        self.class_names = class_names
        self.tracker_id = cfg.kind

        args = SimpleNamespace(
            tracker_type=cfg.kind,
            track_high_thresh=cfg.track_high_thresh,
            track_low_thresh=cfg.track_low_thresh,
            new_track_thresh=cfg.new_track_thresh,
            track_buffer=cfg.track_buffer,
            match_thresh=cfg.match_thresh,
            fuse_score=cfg.fuse_score,
            gmc_method=cfg.gmc_method,
            proximity_thresh=cfg.proximity_thresh,
            appearance_thresh=cfg.appearance_thresh,
            with_reid=cfg.with_reid,
            model="auto",
        )
        self._impl = (BOTSORT(args) if cfg.kind == "botsort" else BYTETracker(args))
        self._tracks: dict[int, Track] = {}

    def reset(self) -> None:
        self._impl.reset()
        self._tracks.clear()

    def update(
        self,
        batch: DetBatch,
        dets: list[Detection],
        frame: np.ndarray,
        frame_idx: int,
        at: datetime,
    ) -> list[Track]:
        """Associate detections to persistent tracks.

        Returns only the tracks confirmed in THIS frame. Tracks the tracker is
        still holding but did not match are aged internally and reappear when
        re-associated -- which is the point of a tracker.
        """
        out = self._impl.update(batch, frame)
        if out is None or len(out) == 0:
            for t in self._tracks.values():
                t.misses += 1
            return []

        out = np.asarray(out)
        live: list[Track] = []
        seen: set[int] = set()

        for row in out:
            # Ultralytics returns [x1, y1, x2, y2, track_id, score, cls, det_idx].
            x1, y1, x2, y2 = (float(v) for v in row[:4])
            tid = int(row[4])
            score = float(row[5])
            cid = int(row[6])
            det_idx = int(row[7]) if len(row) > 7 else -1

            # Prefer the original Detection so model_id and edge_clipped survive
            # the round trip; fall back to reconstructing from the tracker row.
            if 0 <= det_idx < len(dets):
                det = dets[det_idx]
            else:
                det = Detection(
                    bbox_px=(x1, y1, x2, y2), class_id=cid,
                    class_name=self.class_names.get(cid, str(cid)),
                    group=self.mapper.group(cid), confidence=score,
                    model_id=batch.model_id, tier="T1",
                    edge_clipped=_edge_clipped((x1, y1, x2, y2), frame.shape))

            track = self._tracks.get(tid)
            if track is None:
                track = Track(
                    track_id=tid, camera_id=self.camera_id, group=det.group,
                    class_name=det.class_name, latest=det,
                    tracker_id=self.tracker_id, first_seen=at)
                self._tracks[tid] = track
            else:
                track.latest = det
                # A track's group can flip frame to frame on degraded input
                # (this camera's parked car alternates car/boat/truck). Trust
                # the first stable assignment rather than the latest jitter.
                if track.group == ObjectGroup.OTHER and det.group != ObjectGroup.OTHER:
                    track.group = det.group
                    track.class_name = det.class_name

            track.last_seen = at
            track.age_frames += 1
            track.misses = 0
            track.history.append(TrackPoint(
                t=at, frame_idx=frame_idx, footpoint_px=det.footpoint_px,
                footpoint_m=None,                 # filled by the spatial stage
                bbox_px=det.bbox_px, confidence=det.confidence))
            live.append(track)
            seen.add(tid)

        for tid, t in self._tracks.items():
            if tid not in seen:
                t.misses += 1

        # Forget tracks the tracker itself has given up on, so the dict does
        # not grow without bound over a long run.
        dead = [tid for tid, t in self._tracks.items()
                if t.misses > self.cfg.track_buffer * 2]
        for tid in dead:
            self._tracks.pop(tid, None)

        return live

    @property
    def active(self) -> dict[int, Track]:
        return self._tracks


def _edge_clipped(box: tuple[float, float, float, float], shape, m: int = 2) -> bool:
    h, w = shape[:2]
    x1, y1, x2, y2 = box
    return bool(x1 <= m or y1 <= m or x2 >= w - m or y2 >= h - m)
