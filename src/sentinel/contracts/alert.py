"""Tier 8 contracts: Alert and Evidence.

Two rules are enforced by the shapes in this module.

1. Threat and confidence are independent fields and are never combined into a
   single score. Threat drives colour, confidence drives style (solid vs
   hatched). Multiplying them would let a low-confidence critical event and a
   high-confidence trivial one render identically, which is exactly the
   ambiguity the two-axis model exists to remove. (F12)

2. An Alert cannot exist without a resolvable EvidenceRef. An alert that cannot
   be audited afterwards is an assertion, not a detection. (F13)
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from .observation import Observation, ObservationKind


class ThreatLevel(str, Enum):
    """The colour axis.

    Ordering matters -- ``severity`` is used to decide what may be dropped from
    a saturated queue and what may never be.
    """

    INFO = "INFO"          # blue   -- logged, not actionable
    FRIENDLY = "FRIENDLY"  # green  -- identified friendly personnel
    WATCH = "WATCH"        # amber  -- unidentified, no breach yet
    WARNING = "WARNING"    # orange -- breach by an unidentified subject
    CRITICAL = "CRITICAL"  # red    -- confirmed breach, or breach + aggravator

    @property
    def severity(self) -> int:
        return {
            ThreatLevel.INFO: 0,
            ThreatLevel.FRIENDLY: 1,
            ThreatLevel.WATCH: 2,
            ThreatLevel.WARNING: 3,
            ThreatLevel.CRITICAL: 4,
        }[self]


class ConfidenceBand(str, Enum):
    """The style axis. Rendered as solid / hatched / dashed, never as colour."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

    @staticmethod
    def from_score(score: float, high: float = 0.75, medium: float = 0.50) -> "ConfidenceBand":
        if score >= high:
            return ConfidenceBand.HIGH
        if score >= medium:
            return ConfidenceBand.MEDIUM
        return ConfidenceBand.LOW


class AlertState(str, Enum):
    """Alert lifecycle.

    CLEARED exists because the reference prototype we assessed had no way back:
    its breach set was append-only, so one false positive latched the threat
    gauge permanently. An alert that cannot clear is a bug, not a safety margin.
    """

    PENDING = "PENDING"    # observations accumulating, not yet confirmed
    ACTIVE = "ACTIVE"      # confirmed and live
    CLEARED = "CLEARED"    # condition resolved (owner returned, subject left)
    EXPIRED = "EXPIRED"    # aged out without resolution


@dataclass(slots=True)
class EvidenceRef:
    """F13 -- everything needed to re-litigate an alert after the fact.

    ``config_snapshot`` captures the thresholds actually in force when the
    alert fired. Without it, "why did this fire?" is unanswerable three days
    later when the demo profile has been swapped for the operational one.
    """

    evidence_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    keyframe_path: str = ""
    crop_path: str = ""
    clip_path: str = ""
    clip_start_ms: int = 0
    clip_end_ms: int = 0
    detections: list[dict[str, Any]] = field(default_factory=list)
    model_ids: dict[str, str] = field(default_factory=dict)
    geometry: dict[str, Any] = field(default_factory=dict)
    homography_id: str | None = None
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None

    @property
    def resolvable(self) -> bool:
        """An evidence record with no keyframe cannot support an alert."""
        return bool(self.keyframe_path)


@dataclass(slots=True)
class Alert:
    """A confirmed, operator-visible event.

    Only ``sentinel.alerting.manager.AlertManager`` may construct one. Detection
    produces observations; promotion is a single component's responsibility so
    that the suppression policy lives in exactly one place.
    """

    kind: ObservationKind
    camera_id: str
    threat: ThreatLevel
    confidence: float
    opened_at: datetime
    alert_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    state: AlertState = AlertState.PENDING
    updated_at: datetime | None = None
    cleared_at: datetime | None = None
    track_ids: tuple[int, ...] = ()

    # The observations that satisfied temporal confirmation. The alert
    # literally carries its own proof (F7 + F13).
    supporting: list[Observation] = field(default_factory=list)

    evidence: EvidenceRef = field(default_factory=EvidenceRef)
    rationale: str = ""
    subject_key: str = ""
    relay_sent: bool = False

    @property
    def confidence_band(self) -> ConfidenceBand:
        return ConfidenceBand.from_score(self.confidence)

    @property
    def is_live(self) -> bool:
        return self.state in (AlertState.PENDING, AlertState.ACTIVE)

    def to_row(self) -> dict[str, Any]:
        """Flat representation for the SQL alert log (F10)."""
        return {
            "alert_id": self.alert_id,
            "camera_id": self.camera_id,
            "kind": self.kind.value,
            "threat": self.threat.value,
            "threat_severity": self.threat.severity,
            "confidence": round(self.confidence, 4),
            "confidence_band": self.confidence_band.value,
            "state": self.state.value,
            "opened_at": self.opened_at.isoformat(),
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "cleared_at": self.cleared_at.isoformat() if self.cleared_at else None,
            "track_ids": ",".join(str(t) for t in self.track_ids),
            "rationale": self.rationale,
            "subject_key": self.subject_key,
            "evidence_id": self.evidence.evidence_id,
            "keyframe_path": self.evidence.keyframe_path,
            "crop_path": self.evidence.crop_path,
            "clip_path": self.evidence.clip_path,
        }
