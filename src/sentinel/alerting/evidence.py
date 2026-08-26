"""F13 -- the evidence store.

Every alert carries its evidence. An alert that cannot be audited afterwards is
an assertion, not a detection -- and "how do you know it's right?" should be
answerable by clicking the alert, not by a slide.

What gets stored, per alert:

* the keyframe, written from the **raw** buffer, never the annotated one
* a crop of the subject
* a short clip spanning the moments before and after the trigger
* every supporting detection with its own confidence and originating model_id
* the triggering geometry -- which line, which direction, distance in metres
* the homography identity, so a metric claim can be re-derived
* a snapshot of the thresholds in force at fire time

That last one is the least obvious and the most useful. Without it, "why did
this fire?" is unanswerable three days later once the demo profile has been
swapped for the operational one.
"""

from __future__ import annotations

import shutil
import subprocess
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from ..config import StorageCfg, SystemCfg, resolve
from ..contracts import Alert, EvidenceRef, ObjectGroup, Track


class ClipBuffer:
    """Rolling frame buffer so an alert can include the run-up, not just the moment.

    Holds decoded frames, which is memory-hungry but simple; at 720p and a
    300-frame ceiling that is roughly 600 MB worst case, so the cap is a real
    limit rather than a formality. Frames are stored downscaled for exactly
    that reason.
    """

    def __init__(self, cfg: StorageCfg, fps: float = 25.0, max_width: int = 960) -> None:
        self.cfg = cfg
        self.fps = max(1.0, fps)
        self.max_width = max_width
        self.frames: deque[tuple[datetime, np.ndarray]] = deque(
            maxlen=max(10, cfg.keep_clip_frames))

    def push(self, at: datetime, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        if w > self.max_width:
            s = self.max_width / w
            frame = cv2.resize(frame, (self.max_width, int(h * s)),
                               interpolation=cv2.INTER_AREA)
        self.frames.append((at, frame))

    def write_clip(self, path: Path, centre: datetime) -> tuple[int, int]:
        """Write the buffered window around ``centre``. Returns (start_ms, end_ms)."""
        if not self.frames:
            return 0, 0
        pre = self.cfg.clip_pre_s
        sel = [(t, f) for t, f in self.frames
               if (centre - t).total_seconds() <= pre and t <= centre]
        if len(sel) < 2:
            sel = list(self.frames)[-int(self.fps * pre or 10):]
        if len(sel) < 2:
            return 0, 0

        h, w = sel[0][1].shape[:2]
        path.parent.mkdir(parents=True, exist_ok=True)

        # H.264 so the clip plays in a browser <video> element. An evidence
        # clip nobody can open is not evidence, and OpenCV's mp4v fallback
        # produces MPEG-4 Part 2, which browsers will not decode natively.
        # Prefer ffmpeg, whose H.264 encoder is reliable; fall back to OpenCV
        # only if ffmpeg is absent.
        if _write_h264_ffmpeg(path, [f for _t, f in sel], self.fps, w, h):
            span_ms = int((sel[-1][0] - sel[0][0]).total_seconds() * 1000)
            return 0, max(0, span_ms)

        writer = None
        for fourcc in ("avc1", "mp4v"):
            cand = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc),
                                   self.fps, (w, h))
            if cand.isOpened():
                writer = cand
                break
            cand.release()
        if writer is None:
            return 0, 0
        for _t, f in sel:
            writer.write(f)
        writer.release()

        span_ms = int((sel[-1][0] - sel[0][0]).total_seconds() * 1000)
        return 0, max(0, span_ms)


def _write_h264_ffmpeg(path: Path, frames: list[np.ndarray],
                       fps: float, w: int, h: int) -> bool:
    """Encode frames to browser-playable H.264 by piping raw BGR to ffmpeg.

    Returns False if ffmpeg is unavailable or fails, so the caller can fall
    back rather than silently producing no clip.
    """
    exe = shutil.which("ffmpeg")
    if not exe or not frames:
        return False
    cmd = [
        exe, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
        "-r", f"{max(1.0, fps):.3f}", "-i", "pipe:0",
        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
        # yuv420p is required for playback in browsers and most players.
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
    ]
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        for f in frames:
            proc.stdin.write(np.ascontiguousarray(f).tobytes())
        proc.stdin.close()
        proc.wait(timeout=30)
        return proc.returncode == 0 and path.exists() and path.stat().st_size > 0
    except (OSError, subprocess.SubprocessError, BrokenPipeError):
        return False


class EvidenceWriter:
    """Builds and persists the evidence record for one alert."""

    def __init__(self, cfg: SystemCfg) -> None:
        self.cfg = cfg
        self.scfg: StorageCfg = cfg.storage
        self.evidence_dir = resolve(cfg, self.scfg.evidence_dir)
        self.clip_dir = resolve(cfg, self.scfg.clip_dir)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.clip_dir.mkdir(parents=True, exist_ok=True)
        self.model_ids: dict[str, str] = {}

        # Face localisation is detection only -- it cuts a recognisable image
        # out of the subject crop so an operator can see WHO was seen. It never
        # decides identity; that stays with the open-set gallery.
        from ..config import resolve as _resolve
        from ..identity.face import FaceLocator
        self.faces = FaceLocator(_resolve(cfg, "weights/face_detection_yunet_2023mar.onnx"))

    def register_models(self, **ids: str) -> None:
        """Record which models are in play, for the provenance block."""
        self.model_ids.update({k: v for k, v in ids.items() if v})
        if self.faces.available:
            self.model_ids.setdefault("face", self.faces.model_id)

    def build(
        self,
        alert: Alert,
        raw_frame: np.ndarray,
        tracks: dict[int, Track],
        clip: ClipBuffer | None,
        homography_id: str | None,
    ) -> EvidenceRef:
        ref = alert.evidence
        ref.created_at = datetime.now(timezone.utc)
        stamp = ref.created_at.strftime("%Y%m%dT%H%M%S")
        base = f"{alert.camera_id}_{stamp}_{alert.alert_id[:8]}"

        # Keyframe from the RAW buffer. Annotations are a rendering of a
        # decision; evidence must be the input that decision was made on.
        kf = self.evidence_dir / f"{base}_keyframe.jpg"
        cv2.imwrite(str(kf), raw_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        ref.keyframe_path = str(kf.relative_to(Path(self.cfg.root_dir)))

        subject = next((tracks[t] for t in alert.track_ids if t in tracks), None)
        if subject is not None:
            x1, y1, x2, y2 = (int(v) for v in subject.latest.bbox_px)
            mx, my = int(0.2 * (x2 - x1)), int(0.2 * (y2 - y1))
            h, w = raw_frame.shape[:2]
            crop = raw_frame[max(0, y1 - my):min(h, y2 + my),
                             max(0, x1 - mx):min(w, x2 + mx)]
            if crop.size:
                cp = self.evidence_dir / f"{base}_crop.jpg"
                cv2.imwrite(str(cp), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
                ref.crop_path = str(cp.relative_to(Path(self.cfg.root_dir)))

                # A head-and-shoulders cut, when the subject is a person and a
                # face is actually visible. Absence is normal and is left
                # blank rather than filled with a guess -- a subject facing
                # away has no face to show, and inventing one would be worse
                # than an empty tile.
                if subject.group is ObjectGroup.PERSON:
                    face = self.faces.cut(crop)
                    if face is not None and face.size:
                        fp = self.evidence_dir / f"{base}_face.jpg"
                        cv2.imwrite(str(fp), face, [cv2.IMWRITE_JPEG_QUALITY, 94])
                        ref.face_path = str(fp.relative_to(Path(self.cfg.root_dir)))

        if clip is not None:
            cl = self.clip_dir / f"{base}.mp4"
            s_ms, e_ms = clip.write_clip(cl, alert.opened_at)
            if cl.exists():
                ref.clip_path = str(cl.relative_to(Path(self.cfg.root_dir)))
                ref.clip_start_ms, ref.clip_end_ms = s_ms, e_ms

        # Every supporting detection, with its own confidence and model.
        ref.detections = []
        for tid in alert.track_ids:
            t = tracks.get(tid)
            if t is None:
                continue
            d = t.latest
            ref.detections.append({
                "track_id": tid,
                "class_name": d.class_name,
                "group": d.group.value,
                "confidence": round(d.confidence, 4),
                "model_id": d.model_id,
                "bbox_px": [round(v, 1) for v in d.bbox_px],
                "footpoint_m": ([round(v, 3) for v in t.footpoint_m]
                                if t.footpoint_m else None),
                "edge_clipped": d.edge_clipped,
                "stance": t.stance.value,
                "identity": (
                    {"label": t.identity.label.value,
                     "similarity": t.identity.similarity,
                     "threshold": t.identity.threshold_used,
                     "quality": t.identity.quality_score,
                     "matched": t.identity.matched_enrollee}
                    if t.identity else None),
            })

        ref.model_ids = dict(self.model_ids)
        ref.geometry = (alert.supporting[-1].geometry if alert.supporting else {})
        ref.homography_id = homography_id
        ref.config_snapshot = self.cfg.snapshot()
        return ref
