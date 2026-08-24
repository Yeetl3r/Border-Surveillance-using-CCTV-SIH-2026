"""Video source interface and the OpenCV implementation.

The source is behind a Protocol so that swapping OpenCV for a GStreamer/NVDEC
pipeline later touches this package and nothing else. Start with OpenCV; add
hardware decode only if decode actually becomes the bottleneck, which at one
720p stream it will not.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Protocol, runtime_checkable

import cv2
import numpy as np

from ..config import CameraCfg, EnhanceCfg, HealthCfg, SystemCfg, resolve
from ..contracts import FrameBundle, HealthState, HealthStatus
from .enhance import enhance_for_operator
from .health import HealthMonitor


@runtime_checkable
class VideoSource(Protocol):
    """Anything that can yield FrameBundles for one camera."""

    camera_id: str

    def frames(self) -> Iterator[FrameBundle]: ...
    def close(self) -> None: ...


class OpenCVSource:
    """File or RTSP source via OpenCV.

    Handles the two things that break a live demo: a file source that replays
    a 12-second clip at 900 fps and floods every downstream queue, and an RTSP
    source that drops and never comes back.
    """

    def __init__(self, cam: CameraCfg, cfg: SystemCfg) -> None:
        self.camera_id = cam.camera_id
        self.cam = cam
        self.cfg = cfg
        self.health = HealthMonitor(cfg.health, cam.camera_id)
        self.enhance_cfg: EnhanceCfg = cfg.enhance
        self._cap: cv2.VideoCapture | None = None
        self._frame_idx = 0
        self._stop = False
        self._is_file = False
        self._src_fps = 0.0
        self.reconnects = 0

    # -- lifecycle -----------------------------------------------------------
    def _resolve_source(self) -> str | int:
        src = self.cam.source
        if isinstance(src, str) and src.isdigit():
            return int(src)                       # webcam index
        if isinstance(src, str) and "://" in src:
            return src                            # rtsp:// http:// etc.
        p = resolve(self.cfg, src)
        if not p.exists():
            raise FileNotFoundError(f"camera {self.camera_id}: source not found: {p}")
        self._is_file = True
        return str(p)

    def _open(self) -> cv2.VideoCapture:
        target = self._resolve_source()
        cap = cv2.VideoCapture(target)
        if not cap.isOpened():
            raise RuntimeError(f"camera {self.camera_id}: cannot open source {target!r}")
        # Keep the driver buffer shallow: on a live feed a deep buffer means we
        # process stale frames, which is worse than dropping them.
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        except cv2.error:
            pass
        self._src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        return cap

    def close(self) -> None:
        self._stop = True
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # -- iteration -----------------------------------------------------------
    def frames(self) -> Iterator[FrameBundle]:
        self._cap = self._open()
        # Pace a file source to its own frame rate so a short clip behaves like
        # a live camera rather than a firehose.
        interval = 0.0
        if self._is_file and self.cam.pace_to_source_fps and self._src_fps > 1:
            interval = 1.0 / min(self._src_fps, self.cfg.runtime.target_fps)
        next_due = time.perf_counter()

        while not self._stop:
            ok, frame = self._cap.read()

            if not ok:
                if self._is_file and self.cam.loop:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    # A loop wrap is a discontinuity, not a camera move -- do
                    # not let it register as structural drift.
                    self.health.reset_reference()
                    continue
                if not self._is_file:
                    yield self._signal_lost_bundle()
                    if not self._reconnect():
                        break
                    continue
                break

            self._frame_idx += 1
            if interval:
                now = time.perf_counter()
                if next_due > now:
                    time.sleep(next_due - now)
                next_due = max(next_due + interval, time.perf_counter() - interval)

            health = self.health.update(frame)
            view = enhance_for_operator(frame, self.enhance_cfg)

            yield FrameBundle(
                camera_id=self.camera_id,
                frame_idx=self._frame_idx,
                captured_at=datetime.now(timezone.utc),
                raw=frame,                 # detector input -- never enhanced
                view=view,                 # operator input -- enhanced if enabled
                channels=("B", "G", "R"),
                health=health,
                source_fps=self._src_fps,
            )

        self.close()

    def _signal_lost_bundle(self) -> FrameBundle:
        h, w = 720, 1280
        if self._cap is not None:
            h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or h
            w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or w
        return FrameBundle(
            camera_id=self.camera_id,
            frame_idx=self._frame_idx,
            captured_at=datetime.now(timezone.utc),
            raw=np.zeros((h, w, 3), dtype=np.uint8),
            health=HealthStatus(state=HealthState.SIGNAL_LOST,
                                detail="decode returned no frame"),
            source_fps=self._src_fps,
        )

    def _reconnect(self) -> bool:
        """Reopen a dropped live source. Degrade gracefully, do not collapse."""
        if self._cap is not None:
            self._cap.release()
        time.sleep(self.cam.reconnect_delay_s)
        try:
            self._cap = self._open()
            self.reconnects += 1
            self.health.reset_reference()
            return True
        except (RuntimeError, FileNotFoundError):
            return False


def build_source(cam: CameraCfg, cfg: SystemCfg) -> VideoSource:
    """Factory. The only place that decides which source implementation to use."""
    return OpenCVSource(cam, cfg)
