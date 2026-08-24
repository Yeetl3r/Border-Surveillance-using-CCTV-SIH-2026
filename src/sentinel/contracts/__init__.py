"""Data contracts shared across every tier.

This package imports nothing from the rest of ``sentinel``. That constraint is
what makes the temporal and spatial layers unit-testable without a GPU, a
model, or a video file -- they depend only on these shapes.
"""

from .alert import (
    Alert,
    AlertState,
    ConfidenceBand,
    EvidenceRef,
    ThreatLevel,
)
from .cue import Bearing, CueKind, LinearPosition, SensorCue
from .detection import (
    Detection,
    IdentityLabel,
    IdentityVerdict,
    ObjectGroup,
    Stance,
    Track,
    TrackPoint,
)
from .frame import FrameBundle, HealthState, HealthStatus
from .observation import CrossDirection, Observation, ObservationKind

__all__ = [
    "Alert",
    "AlertState",
    "Bearing",
    "ConfidenceBand",
    "CrossDirection",
    "CueKind",
    "Detection",
    "EvidenceRef",
    "FrameBundle",
    "HealthState",
    "HealthStatus",
    "IdentityLabel",
    "IdentityVerdict",
    "LinearPosition",
    "ObjectGroup",
    "Observation",
    "ObservationKind",
    "SensorCue",
    "Stance",
    "ThreatLevel",
    "Track",
    "TrackPoint",
]
