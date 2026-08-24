"""Tier 0 contracts: what ingestion emits.

The load-bearing decision in this module is that ``FrameBundle`` carries two
image buffers. ``raw`` is the detector's only input; ``view`` is what the
operator sees. Enhancement (CLAHE, dark-channel-prior defog) writes to ``view``
and never to ``raw``.

This is not a style preference. Enhancement optimises perceptual quality, not
the detection objective, and damages small-object semantics specifically --
which is the regime this system operates in. Keeping the buffers separate at
the type level makes the mistake unrepresentable rather than merely discouraged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np


class HealthState(str, Enum):
    """Camera health, in escalating order of operational concern."""

    OK = "OK"
    DEGRADED = "DEGRADED"          # soft-focus, rain, gradual contrast loss
    TAMPERED = "TAMPERED"          # lens covered, sprayed, repositioned
    SIGNAL_LOST = "SIGNAL_LOST"    # decode failure or frozen feed


@dataclass(slots=True)
class HealthStatus:
    """F9 -- anti-tamper and camera health for a single frame.

    ``blur_score`` is the variance of the Laplacian; low values mean the image
    has lost high-frequency content (defocus, lens obscuration, heavy fog).
    ``ssim_ref`` is structural similarity against a rolling reference frame;
    a sharp drop means the camera moved or the scene was occluded.
    """

    state: HealthState = HealthState.OK
    blur_score: float = 0.0
    ssim_ref: float = 1.0
    frozen_frames: int = 0
    since: datetime | None = None
    detail: str = ""

    @property
    def usable(self) -> bool:
        """Whether perception should trust this frame at all."""
        return self.state in (HealthState.OK, HealthState.DEGRADED)


@dataclass(slots=True)
class FrameBundle:
    """One decoded frame plus everything Tier 0 knows about it.

    Attributes:
        camera_id: Stable identifier; track IDs are only unique within one.
        frame_idx: Monotonic per stream, from the decoder not the wall clock.
        captured_at: Timezone-aware wall clock. Frame index over FPS is not a
            substitute -- it is meaningless for a live stream and useless in a
            forensic record.
        raw: Decoded frame, shape (H, W, C). The detector's ONLY input.
        view: Operator-viewable frame, possibly enhanced. None means "use raw".
        channels: Channel descriptor. ("B", "G", "R") today. This exists so a
            four-channel RGB-thermal frame is a new producer rather than a
            rewrite of the detector interface (Phase 2, RGB-T fusion).
        health: F9 status for this frame.
        source_fps: Nominal FPS reported by the source, for timing sanity checks.
    """

    camera_id: str
    frame_idx: int
    captured_at: datetime
    raw: np.ndarray
    view: np.ndarray | None = None
    channels: tuple[str, ...] = ("B", "G", "R")
    health: HealthStatus = field(default_factory=HealthStatus)
    source_fps: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def height(self) -> int:
        return int(self.raw.shape[0])

    @property
    def width(self) -> int:
        return int(self.raw.shape[1])

    @property
    def operator_frame(self) -> np.ndarray:
        """The frame a human should look at. Never fed to a detector."""
        return self.view if self.view is not None else self.raw
