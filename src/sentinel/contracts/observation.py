"""Tier 2/3 contract: the Observation.

An Observation says "something happened in this frame". It is not an alert and
must never reach an operator on its own.

The distinction matters because of the documented failure mode in this domain:
SBInet produced apprehensions on roughly 2% of sensor alerts, and CIBMS
assessments name false-positive suppression as the hardest engineering problem.
A system that promotes every observation to an alert trains its operator to
ignore it, at which point the detector's accuracy is irrelevant.

So observations are cheap, frequent and disposable. Producing one commits to
nothing. Only the alert manager promotes them, and only after temporal
confirmation (F7).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class ObservationKind(str, Enum):
    """What kind of thing was noticed."""

    TRIPWIRE_CROSS = "TRIPWIRE_CROSS"      # F4 -- directional line crossing
    ZONE_PRESENCE = "ZONE_PRESENCE"        # F4 -- inside a restricted polygon
    PROXIMITY = "PROXIMITY"                # F5 -- within N metres of an asset
    BAG_UNATTENDED = "BAG_UNATTENDED"      # F6 -- dwell timer expired
    STANCE_ANOMALY = "STANCE_ANOMALY"      # F16 -- crawling / prone
    IDENTITY_UNKNOWN = "IDENTITY_UNKNOWN"  # F8 -- person not in friendly gallery
    WEAPON_VISIBLE = "WEAPON_VISIBLE"      # specialised detector on a person crop
    AERIAL_INTRUSION = "AERIAL_INTRUSION"  # specialised detector, full frame
    HEALTH_FAULT = "HEALTH_FAULT"          # F9 -- tamper or signal loss
    SENSOR_CUE = "SENSOR_CUE"              # Phase 2 -- external cue (DAS sim)


class CrossDirection(str, Enum):
    """Which way a track crossed a tripwire.

    Direction is the whole point of a tripwire as opposed to a zone test.
    Infiltration and exfiltration are operationally different events and a
    containment check cannot tell them apart.
    """

    INBOUND = "INBOUND"     # negative side -> positive side (into our territory)
    OUTBOUND = "OUTBOUND"
    UNKNOWN = "UNKNOWN"


@dataclass(slots=True)
class Observation:
    """A single noticed event, pre-confirmation.

    Attributes:
        confidence: Evidential strength of THIS observation, wholly independent
            of how serious it would be if true. Threat and confidence are two
            axes and are never multiplied into one score.
        geometry: The triggering geometry -- which line was crossed, in which
            direction, which polygon, distance in metres. This is what makes
            an alert auditable after the fact rather than an assertion.
    """

    kind: ObservationKind
    camera_id: str
    at: datetime
    frame_idx: int
    confidence: float
    obs_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    track_ids: tuple[int, ...] = ()
    geometry: dict[str, Any] = field(default_factory=dict)
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def subject_key(self) -> str:
        """Identity used to group observations into one candidate alert.

        Observations about the same kind of thing happening to the same tracks
        on the same camera are the same candidate event, and should accumulate
        toward one confirmation rather than N competing ones.
        """
        ids = ",".join(str(i) for i in sorted(self.track_ids))
        zone = self.geometry.get("zone_id", "")
        return f"{self.camera_id}|{self.kind.value}|{ids}|{zone}"
