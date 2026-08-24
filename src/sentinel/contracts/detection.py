"""Tier 1 contracts: detection and tracking.

A ``Detection`` is one object in one frame and carries no identity over time.
A ``Track`` is identity over time and owns the trajectory. Conflating the two
is how counting bugs get in -- if a detection can carry a track_id, something
eventually invents one when the tracker has not assigned it yet.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

# ---------------------------------------------------------------------------
# Class grouping
# ---------------------------------------------------------------------------
# F1 asks for person / vehicle / animal / bag. COCO has no such classes -- it
# has 80 fine-grained ones. The mapping lives in config (detector.class_groups)
# so the class list is data, not an enum. That is deliberate: Phase 2 drone
# detection adds classes, and a four-class assumption baked into a type here
# would have to be unpicked later.


class ObjectGroup(str, Enum):
    """Operational grouping of detector classes."""

    PERSON = "person"
    VEHICLE = "vehicle"
    ANIMAL = "animal"
    BAG = "bag"
    OTHER = "other"


class Stance(str, Enum):
    """F16 -- infiltration posture, derived from pose keypoints on a crop."""

    STANDING = "STANDING"
    CROUCHING = "CROUCHING"
    PRONE = "PRONE"           # crawling or lying
    UNKNOWN = "UNKNOWN"       # keypoints absent or too low-confidence to judge


class IdentityLabel(str, Enum):
    """F8 -- open-set identity verdict.

    There is no HOSTILE value and there will not be one. No legitimate dataset
    of adversary appearance exists at CCTV resolution, and a closed-set
    classifier answers "what about a civilian in dark clothing?" badly and
    confidently.

    INSUFFICIENT_QUALITY is distinct from UNIDENTIFIED on purpose. "The crop
    was too poor to judge" and "this is not someone we enrolled" are different
    claims, and collapsing them is how a system starts asserting more than it
    knows.
    """

    FRIENDLY = "FRIENDLY"
    UNIDENTIFIED = "UNIDENTIFIED"
    INSUFFICIENT_QUALITY = "INSUFFICIENT_QUALITY"


@dataclass(slots=True)
class IdentityVerdict:
    """Result of matching a person crop against the enrolled friendly gallery."""

    label: IdentityLabel
    matched_enrollee: str | None = None
    similarity: float | None = None
    threshold_used: float = 0.0
    quality_score: float = 0.0
    model_id: str = ""
    decided_at: datetime | None = None


@dataclass(slots=True)
class Detection:
    """One object, one frame, before tracking.

    ``model_id`` is required rather than optional because F13 says every
    detection must be able to name the model that produced it. An evidence
    record that cannot answer "which weights said this?" is not auditable.
    """

    bbox_px: tuple[float, float, float, float]      # x1, y1, x2, y2
    class_id: int
    class_name: str
    group: ObjectGroup
    confidence: float
    model_id: str
    tier: str = "T1"
    edge_clipped: bool = False   # touches a frame border; box geometry is unreliable

    @property
    def footpoint_px(self) -> tuple[float, float]:
        """Bottom-centre of the box -- the ground-contact estimate."""
        x1, _y1, x2, y2 = self.bbox_px
        return ((x1 + x2) / 2.0, y2)

    @property
    def centre_px(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox_px
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def width_px(self) -> float:
        return self.bbox_px[2] - self.bbox_px[0]

    @property
    def height_px(self) -> float:
        return self.bbox_px[3] - self.bbox_px[1]

    @property
    def aspect_ratio(self) -> float:
        return self.width_px / max(1e-6, self.height_px)


@dataclass(slots=True)
class TrackPoint:
    """One sample of a track's trajectory.

    Retained rather than discarded after the current frame. Phase 2 drone
    detection needs flight-path linearity and hover behaviour to separate
    quadcopters from birds, and that analysis reads this history. Phase 1 does
    not consume it, but must not throw it away.
    """

    t: datetime
    frame_idx: int
    footpoint_px: tuple[float, float]
    footpoint_m: tuple[float, float] | None
    bbox_px: tuple[float, float, float, float]
    confidence: float


@dataclass(slots=True)
class Track:
    """Identity over time for one object on one camera."""

    track_id: int
    camera_id: str
    group: ObjectGroup
    class_name: str
    latest: Detection
    tracker_id: str = ""

    # Ground-plane position in metres (F3). None when the camera is
    # uncalibrated or the footpoint falls outside the homography's valid
    # region. Downstream must handle None rather than assume a number -- a
    # fabricated metre value is worse than an absent one.
    footpoint_m: tuple[float, float] | None = None

    history: deque[TrackPoint] = field(default_factory=lambda: deque(maxlen=256))
    first_seen: datetime | None = None
    last_seen: datetime | None = None
    age_frames: int = 0
    misses: int = 0

    stance: Stance = Stance.UNKNOWN
    stance_confidence: float = 0.0
    identity: IdentityVerdict | None = None

    # F6 -- set on BAG tracks to the track_id of the associated person.
    owner_track_id: int | None = None

    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        """Globally unique within a run: track IDs are per-camera only."""
        return f"{self.camera_id}:{self.track_id}"

    def displacement_m(self, back: int = 10) -> float | None:
        """Straight-line ground distance moved over the last ``back`` samples.

        Returns None if the track lacks metric positions -- the honest answer
        when the camera is uncalibrated.
        """
        pts = [p.footpoint_m for p in list(self.history)[-back:] if p.footpoint_m]
        if len(pts) < 2:
            return None
        (x0, y0), (x1, y1) = pts[0], pts[-1]
        return ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
