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

        # Transport state. Only meaningful for a recorded source: a live camera
        # has no position to seek to and no past to replay, so the controls are
        # withheld rather than shown doing nothing.
        self._paused = False
        self._rate = 1.0
        self._seek_to: int | None = None
        self.total_frames = 0

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
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        return cap

    # -- transport (recorded sources only) -----------------------------------
    @property
    def is_live(self) -> bool:
        """A live camera cannot be paused, scrubbed or replayed.

        The distinction is exposed rather than inferred in the interface: a
        transport bar over a live feed would be lying about what it can do.
        """
        return not self._is_file

    def pause(self) -> None:
        if self._is_file:
            self._paused = True

    def resume(self) -> None:
        self._paused = False

    def toggle(self) -> bool:
        if self._is_file:
            self._paused = not self._paused
        return self._paused

    def set_rate(self, rate: float) -> None:
        if self._is_file:
            self._rate = max(0.1, min(8.0, float(rate)))

    def seek_fraction(self, frac: float) -> None:
        if self._is_file and self.total_frames > 0:
            f = int(max(0.0, min(1.0, float(frac))) * (self.total_frames - 1))
            self._seek_to = f

    def transport(self) -> dict:
        return {
            "live": self.is_live,
            "paused": self._paused,
            "rate": self._rate,
            "frame": self._frame_idx,
            "position": int(self._cap.get(cv2.CAP_PROP_POS_FRAMES)) if self._cap else 0,
            "total_frames": self.total_frames,
            "fps": self._src_fps,
            "duration_s": round(self.total_frames / self._src_fps, 2)
                          if self._src_fps > 0 and self.total_frames else 0.0,
        }

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
            # A requested seek is applied on the decode thread rather than from
            # the request handler: VideoCapture is not safe to reposition while
            # another thread is mid-read.
            if self._seek_to is not None:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, self._seek_to)
                self._frame_idx = self._seek_to
                self._seek_to = None
                self.health.reset_reference()   # a jump is not a camera move
                next_due = time.perf_counter()

            if self._paused:
                # Hold position without spinning. The pipeline keeps serving the
                # last decoded frame, so the picture stays up while paused.
                time.sleep(0.05)
                continue

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
                step = interval / max(0.1, self._rate)   # playback speed
                now = time.perf_counter()
                if next_due > now:
                    time.sleep(next_due - now)
                next_due = max(next_due + step, time.perf_counter() - step)

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
