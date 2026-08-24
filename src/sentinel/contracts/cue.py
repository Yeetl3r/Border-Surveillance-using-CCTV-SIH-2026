"""Phase 2 hook: external sensor cues.

NOTHING IN PHASE 1 PRODUCES A ``SensorCue``. This module defines the type only,
so that the cue path exists conceptually and a Phase 2 producer can be added
without touching Tier 1.

The intended Phase 2 producer is a *simulated* distributed acoustic sensing
feed. Real DAS requires a phi-OTDR interrogator -- specialist hardware costing
tens of thousands of dollars that this project does not have and will not get.
The classification and fusion layers are real, buildable work; the interrogator
is not.

Hence ``simulated`` is a required field with no default. A cue cannot be
constructed without declaring whether it is real. That turns the honesty
requirement into a property of the type system rather than something someone
has to remember at 2 a.m. before a demo.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class CueKind(str, Enum):
    FOOTSTEPS = "FOOTSTEPS"
    VEHICLE = "VEHICLE"
    DIGGING = "DIGGING"
    FENCE_TAMPER = "FENCE_TAMPER"
    UNKNOWN = "UNKNOWN"


@dataclass(slots=True)
class LinearPosition:
    """Position along a fibre run, in metres from the interrogator."""

    fibre_metre: float
    uncertainty_m: float = 0.0


@dataclass(slots=True)
class Bearing:
    """Direction from a known sensor origin, in degrees clockwise from north."""

    degrees: float
    uncertainty_deg: float = 0.0


@dataclass(slots=True)
class SensorCue:
    """A cue from a non-camera sensor.

    Args:
        simulated: REQUIRED. True means this cue was replayed or synthesised
            and must be labelled ``SIMULATED`` everywhere it is displayed.
    """

    source_id: str
    simulated: bool
    at: datetime
    kind: CueKind
    confidence: float
    position: LinearPosition | Bearing | None = None
    cue_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    detail: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.simulated, bool):
            raise TypeError(
                "SensorCue.simulated must be an explicit bool -- a cue may not "
                "be constructed without declaring whether it is real."
            )
