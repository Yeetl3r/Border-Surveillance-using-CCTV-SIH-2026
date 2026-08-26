"""The per-camera pipeline: wires every tier together and runs it.

One ``CameraPipeline`` per camera, each in its own thread with its own tracker,
state machine and zone set. Nothing is shared between cameras except the
repository and the alert manager, which are thread-safe.

Frame flow:

    ingest -> [Q1 decode, drop-oldest] -> perception -> spatial -> temporal
           -> alert manager -> WebSocket / SQLite / relay

Q1 exists so that a slow perception step cannot stall the decoder or grow a
backlog: it drops the oldest frame instead, and says how many it dropped.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Callable

import numpy as np

from ..alerting import AlertManager, ClipBuffer, EvidenceWriter, RelayEncoder
from ..calibration.zones import ZoneSet
from ..config import CameraCfg, SystemCfg
from ..contracts import FrameBundle, ObjectGroup, ThreatLevel, Track
from ..identity import FriendlyGallery, IdentityResolver, build_embedder
from ..ingest import build_source
from ..perception import ObjectTracker, YoloDetector
from ..spatial import SpatialAnalyzer
from ..storage import Repository
from ..temporal import TemporalStateMachine
from .queues import BoundedQueue, DropPolicy, QueueRegistry


class Perf:
    """Rolling timing, so reported FPS is measured rather than asserted."""

    def __init__(self, n: int = 90) -> None:
        self.frame = deque(maxlen=n)
        self.detect = deque(maxlen=n)
        self.track = deque(maxlen=n)
        self.logic = deque(maxlen=n)

    @staticmethod
    def _avg(d) -> float:
        return round(float(np.mean(d)) * 1000, 2) if d else 0.0

    def as_dict(self) -> dict[str, Any]:
        fps = (1.0 / float(np.mean(self.frame))) if self.frame else 0.0
        return {"fps": round(fps, 1), "frame_ms": self._avg(self.frame),
                "detect_ms": self._avg(self.detect), "track_ms": self._avg(self.track),
                "logic_ms": self._avg(self.logic)}


class CameraPipeline:
    """Everything that happens to one camera's frames."""

    def __init__(
        self,
        cfg: SystemCfg,
        cam: CameraCfg,
        repo: Repository,
        registry: QueueRegistry,
        shared: dict[str, Any],
        on_event: Callable[[str, dict], None] | None = None,
    ) -> None:
        self.cfg = cfg
        self.cam = cam
        self.repo = repo
        self.on_event = on_event
        self.camera_id = cam.camera_id

        self.detector: YoloDetector = shared["detector"]
        self.pose = shared.get("pose")
        self.zones = ZoneSet.from_camera(cam)
        self.tracker = ObjectTracker(cfg.tracker, cam.camera_id,
                                     self.detector.mapper, self.detector.class_names)
        self.spatial = SpatialAnalyzer(self.zones, cfg.spatial)
        self.temporal = TemporalStateMachine(cfg, cam.camera_id)
        self.identity: IdentityResolver = shared["identity"]
        self.alerts: AlertManager = shared["alerts"]

        self.clip = ClipBuffer(cfg.storage, fps=cfg.runtime.target_fps)
        self.q_decode: BoundedQueue[FrameBundle] = registry.add(BoundedQueue(
            f"{self.camera_id}:decode", cfg.pipeline.q_decode, DropPolicy.DROP_OLDEST))

        self.perf = Perf()
        self.latest_frame: np.ndarray | None = None
        self.latest_raw: np.ndarray | None = None
        self.latest_overlay: dict[str, Any] = {}
        self.tracks: dict[int, Track] = {}
        self.health_state = "OK"
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self.frames_in = 0
        self.frames_processed = 0

    # -- lifecycle -----------------------------------------------------------
    def start(self) -> None:
        self._threads = [
            threading.Thread(target=self._ingest_loop, name=f"ingest:{self.camera_id}",
                             daemon=True),
            threading.Thread(target=self._process_loop, name=f"proc:{self.camera_id}",
                             daemon=True),
        ]
        for t in self._threads:
            t.start()

    def stop(self) -> None:
        self._stop.set()
        self.q_decode.close()

    # -- stage 1: decode -----------------------------------------------------
    def _ingest_loop(self) -> None:
        source = build_source(self.cam, self.cfg)
        try:
            for bundle in source.frames():
                if self._stop.is_set():
                    break
                self.frames_in += 1
                self.q_decode.push(bundle)
        except (RuntimeError, FileNotFoundError) as exc:
            self._emit("error", {"camera_id": self.camera_id, "error": str(exc)})
        finally:
            source.close()

    # -- stage 2: perception, geometry, logic, alerting ----------------------
    def _process_loop(self) -> None:
        while not self._stop.is_set():
            bundle = self.q_decode.pop(timeout=0.5)
            if bundle is None:
                continue
            try:
                self._process(bundle)
            except Exception as exc:                        # noqa: BLE001
                # One bad frame must not take the pipeline down mid-demo.
                self._emit("error", {"camera_id": self.camera_id,
                                     "frame": bundle.frame_idx, "error": repr(exc)})

    def _process(self, bundle: FrameBundle) -> None:
        t_frame = time.perf_counter()
        self.frames_processed += 1
        self.health_state = bundle.health.state.value
        self.latest_frame = bundle.operator_frame
        self.latest_raw = bundle.raw
        self.clip.push(bundle.captured_at, bundle.raw)

        # -- Tier 1 ----------------------------------------------------------
        t0 = time.perf_counter()
        # The detector sees raw. Never the enhanced operator view.
        batch, dets = self.detector.detect(bundle.raw)
        self.perf.detect.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        tracks = self.tracker.update(batch, dets, bundle.raw,
                                     bundle.frame_idx, bundle.captured_at)
        self.perf.track.append(time.perf_counter() - t0)

        if self.pose is not None:
            self.pose.annotate(tracks, bundle.raw)

        # -- Tier 2 / 5 ------------------------------------------------------
        t0 = time.perf_counter()
        spatial_obs = self.spatial.analyse(tracks, bundle.frame_idx, bundle.captured_at)
        self.identity.resolve(tracks, bundle.raw, bundle.frame_idx)

        # -- Tier 3 ----------------------------------------------------------
        observations, confirmed = self.temporal.process(bundle, tracks, spatial_obs)
        self.tracks = {t.track_id: t for t in tracks}

        # -- Tier 8 ----------------------------------------------------------
        opened = self.alerts.promote(confirmed, bundle, self.tracks,
                                     self.clip, self.zones.homography.homography_id)
        closed = self.alerts.tick(bundle.captured_at, self.temporal.confirmer)
        self.perf.logic.append(time.perf_counter() - t0)

        if bundle.health.state.value != "OK" and bundle.frame_idx % 30 == 0:
            self.repo.log_health(self.camera_id, bundle.health)

        self.latest_overlay = self._overlay(bundle, tracks)
        if opened or closed:
            self._emit("alerts", {"camera_id": self.camera_id,
                                  "opened": [a.alert_id for a in opened],
                                  "closed": [a.alert_id for a in closed]})

        self.perf.frame.append(time.perf_counter() - t_frame)

    # -- outbound ------------------------------------------------------------
    def _overlay(self, bundle: FrameBundle, tracks: list[Track]) -> dict[str, Any]:
        """Per-frame state for the client to draw. Threat and confidence separate."""
        live_by_track: dict[int, Any] = {}
        for a in self.alerts.live.values():
            if a.camera_id != self.camera_id:
                continue
            for tid in a.track_ids:
                cur = live_by_track.get(tid)
                if cur is None or a.threat.severity > cur.threat.severity:
                    live_by_track[tid] = a

        boxes = []
        for t in tracks:
            a = live_by_track.get(t.track_id)
            ident = None
            if t.identity is not None:
                ident = {"label": t.identity.label.value,
                         "matched": t.identity.matched_enrollee,
                         "similarity": t.identity.similarity}
            boxes.append({
                "track_id": t.track_id,
                "bbox": [round(v, 1) for v in t.latest.bbox_px],
                "group": t.group.value,
                "class_name": t.latest.class_name,
                # F12 axis 1: detector confidence -> box STYLE (solid/hatched)
                "confidence": round(t.latest.confidence, 3),
                # F12 axis 2: threat -> box COLOUR
                "threat": a.threat.value if a else self._resting_threat(t),
                "alert_id": a.alert_id if a else None,
                "footpoint_m": ([round(v, 2) for v in t.footpoint_m]
                                if t.footpoint_m else None),
                "stance": t.stance.value,
                "identity": ident,
                "edge_clipped": t.latest.edge_clipped,
                "owner_track_id": t.owner_track_id,
                "age": t.age_frames,
            })

        return {
            "camera_id": self.camera_id,
            "frame_idx": bundle.frame_idx,
            "at": bundle.captured_at.isoformat(),
            "boxes": boxes,
            "health": {"state": bundle.health.state.value,
                       "blur": bundle.health.blur_score,
                       "ssim": bundle.health.ssim_ref,
                       "detail": bundle.health.detail},
            "bags": self.temporal.baggage.snapshot(),
            "perf": self.perf.as_dict(),
            "metric": self.zones.metric,
        }

    @staticmethod
    def _resting_threat(t: Track) -> str:
        """Colour for a track with no active alert.

        A tracked object is not by itself an event. Friendlies read green,
        everything else reads INFO -- neutral, not amber. Painting every
        untriggered detection as a concern is how a display becomes noise.
        """
        if t.identity is not None and t.identity.label.value == "FRIENDLY":
            return ThreatLevel.FRIENDLY.value
        return ThreatLevel.INFO.value

    def _emit(self, kind: str, payload: dict) -> None:
        if self.on_event:
            self.on_event(kind, payload)

    def status(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "name": self.cam.name or self.camera_id,
            "health": self.health_state,
            "frames_in": self.frames_in,
            "frames_processed": self.frames_processed,
            "queue_depth": len(self.q_decode),
            "queue_dropped": self.q_decode.stats.dropped,
            "perf": self.perf.as_dict(),
            "tracks": len(self.tracks),
            "pending_candidates": self.temporal.pending_candidates,
            "metric": self.zones.metric,
            "homography": self.zones.homography.describe(),
        }


class SentinelSystem:
    """Owns the shared models, the repository and every camera pipeline."""

    def __init__(self, cfg: SystemCfg,
                 on_event: Callable[[str, dict], None] | None = None) -> None:
        self.cfg = cfg
        self.on_event = on_event
        self.registry = QueueRegistry()
        self.repo = Repository(cfg.root_dir + "/" + cfg.storage.db_path
                               if not cfg.storage.db_path.startswith(("/", "\\"))
                               else cfg.storage.db_path)

        # F15: refuse network access at runtime. Set before any model loads.
        if cfg.runtime.offline:
            import os
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            os.environ.setdefault("YOLO_OFFLINE", "1")

        self.detector = YoloDetector(cfg)
        self.detector.warmup()

        self.pose = None
        if cfg.pose.enabled:
            from ..perception import PoseEstimator
            self.pose = PoseEstimator(cfg)

        self.embedder = build_embedder(cfg)
        self.gallery = FriendlyGallery(cfg, self.embedder)
        self.identity = IdentityResolver(cfg, self.embedder, self.gallery)

        self.evidence = EvidenceWriter(cfg)
        self.evidence.register_models(
            detector=self.detector.model_id,
            tracker=cfg.tracker.kind,
            embedder=self.embedder.model_id,
            pose=(self.pose.model_id if self.pose else ""),
            vlm=(cfg.vlm.model_id if cfg.vlm.enabled else ""),
        )
        self.relay = RelayEncoder(cfg)
        self.alerts = AlertManager(cfg, self.repo, self.evidence, self.relay,
                                   on_alert=self._on_alert)

        self.run_id = self.repo.start_run(
            cfg.profile_name, cfg.snapshot(), self.evidence.model_ids)

        shared = {"detector": self.detector, "pose": self.pose,
                  "identity": self.identity, "alerts": self.alerts}
        self.pipelines: dict[str, CameraPipeline] = {}
        for cam in cfg.cameras:
            if cam.enabled:
                self.pipelines[cam.camera_id] = CameraPipeline(
                    cfg, cam, self.repo, self.registry, shared, on_event)

    def _on_alert(self, alert, phase: str) -> None:
        from ..alerting import as_payload
        if self.on_event:
            self.on_event("alert", {"phase": phase, "alert": as_payload(alert)})

    def start(self) -> None:
        for p in self.pipelines.values():
            p.start()

    def stop(self) -> None:
        for p in self.pipelines.values():
            p.stop()
        self.registry.close_all()

    def status(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "profile": self.cfg.profile_name,
            "offline": self.cfg.runtime.offline,
            "cameras": [p.status() for p in self.pipelines.values()],
            "queues": self.registry.stats(),
            "queue_drops_total": self.registry.total_dropped(),
            "threat": self.alerts.highest_live.value,
            "live_alerts": len(self.alerts.live),
            "counters": self.alerts.counters,
            "identity": self.gallery.describe(),
            "models": self.evidence.model_ids,
            "pose_enabled": self.cfg.pose.enabled,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
