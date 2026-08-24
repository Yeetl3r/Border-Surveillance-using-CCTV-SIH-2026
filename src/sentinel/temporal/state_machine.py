"""Tier 3 -- the temporal state machine.

Every timer, counter and dwell threshold in the system lives here. Nothing
downstream of Tier 1 keeps its own clock, because timing logic scattered
through a detection loop is untestable and cannot be reasoned about as a
policy.

This module has no GPU dependency and no video dependency. It takes tracks and
observations in and produces confirmed candidates out, which is what makes the
suppression policy testable with synthetic data.
"""

from __future__ import annotations

from datetime import datetime

from ..config import SystemCfg
from ..contracts import (
    FrameBundle,
    HealthState,
    IdentityLabel,
    Observation,
    ObservationKind,
    ObjectGroup,
    Stance,
    Track,
)
from .association import BaggageAssociator
from .confirm import Candidate, TemporalConfirmer


class TemporalStateMachine:
    """Owns all temporal logic for one camera."""

    def __init__(self, cfg: SystemCfg, camera_id: str) -> None:
        self.cfg = cfg
        self.camera_id = camera_id
        self.confirmer = TemporalConfirmer(cfg.confirm)
        self.baggage = BaggageAssociator(cfg.baggage)
        self._last_sweep: datetime | None = None

    def reset(self) -> None:
        self.baggage.reset()

    def process(
        self,
        bundle: FrameBundle,
        tracks: list[Track],
        spatial_obs: list[Observation],
    ) -> tuple[list[Observation], list[Candidate]]:
        """Advance the state machine by one frame.

        Returns every observation produced this frame (for the audit log) and
        the candidates that have just crossed the confirmation bar (which the
        alert manager may promote).
        """
        at = bundle.captured_at
        idx = bundle.frame_idx

        # A tick is one processed frame. The hit ratio in F7 is measured
        # against these, which is what makes confirmation frame-rate agnostic.
        self.confirmer.tick(self.camera_id, at)

        observations: list[Observation] = list(spatial_obs)
        observations.extend(self._health(bundle))
        observations.extend(self.baggage.update(tracks, at, self.camera_id))
        observations.extend(self._stance(tracks, at, idx))
        observations.extend(self._identity(tracks, at, idx))

        for o in observations:
            if o.frame_idx == 0:
                o.frame_idx = idx
            self.confirmer.submit(o)

        confirmed = self.confirmer.evaluate(at)

        # Periodic housekeeping: drop candidates nothing has fed for a while.
        if self._last_sweep is None or (at - self._last_sweep).total_seconds() > 10.0:
            self.confirmer.sweep(at, self.cfg.alerts.expire_after_s)
            self._last_sweep = at

        return observations, confirmed

    # -- F9 ------------------------------------------------------------------
    def _health(self, bundle: FrameBundle) -> list[Observation]:
        h = bundle.health
        if h.state == HealthState.OK:
            return []
        # DEGRADED is a condition, not an event -- rain and dusk produce it and
        # an operator does not need telling. TAMPERED and SIGNAL_LOST are events.
        if h.state == HealthState.DEGRADED:
            return []
        return [Observation(
            kind=ObservationKind.HEALTH_FAULT,
            camera_id=self.camera_id, at=bundle.captured_at,
            frame_idx=bundle.frame_idx,
            confidence=0.95 if h.state == HealthState.SIGNAL_LOST else 0.85,
            track_ids=(),
            geometry={"zone_id": f"health:{h.state.value}"},
            detail={"state": h.state.value, "blur_score": h.blur_score,
                    "ssim_ref": h.ssim_ref, "reason": h.detail,
                    "frozen_frames": h.frozen_frames},
        )]

    # -- F16 -----------------------------------------------------------------
    def _stance(self, tracks: list[Track], at: datetime, idx: int) -> list[Observation]:
        if not self.cfg.pose.enabled:
            return []
        out: list[Observation] = []
        for t in tracks:
            if t.group != ObjectGroup.PERSON:
                continue
            if t.stance not in (Stance.PRONE, Stance.CROUCHING):
                continue
            # Never promote stance straight to an alert. It goes through the
            # same confirmation as everything else, with a longer window --
            # a genuine crawl persists, a mis-detection does not.
            out.append(Observation(
                kind=ObservationKind.STANCE_ANOMALY,
                camera_id=self.camera_id, at=at, frame_idx=idx,
                confidence=min(t.stance_confidence, t.latest.confidence),
                track_ids=(t.track_id,),
                geometry={"zone_id": f"stance:{t.stance.value}"},
                detail={"stance": t.stance.value,
                        "stance_confidence": t.stance_confidence,
                        "edge_clipped": t.latest.edge_clipped},
            ))
        return out

    # -- F8 ------------------------------------------------------------------
    def _identity(self, tracks: list[Track], at: datetime, idx: int) -> list[Observation]:
        if not self.cfg.identity.enabled:
            return []
        out: list[Observation] = []
        for t in tracks:
            if t.group != ObjectGroup.PERSON or t.identity is None:
                continue
            # Only UNIDENTIFIED is an observation. INSUFFICIENT_QUALITY is the
            # system saying "I could not tell", which is information for the
            # operator panel but is not an event -- raising an alert because a
            # crop was blurry would be alarming on our own uncertainty.
            if t.identity.label != IdentityLabel.UNIDENTIFIED:
                continue
            out.append(Observation(
                kind=ObservationKind.IDENTITY_UNKNOWN,
                camera_id=self.camera_id, at=at, frame_idx=idx,
                confidence=t.latest.confidence,
                track_ids=(t.track_id,),
                geometry={"zone_id": f"identity:{t.track_id}"},
                detail={"similarity": t.identity.similarity,
                        "threshold": t.identity.threshold_used,
                        "quality": t.identity.quality_score},
            ))
        return out

    @property
    def pending_candidates(self) -> int:
        return self.confirmer.pending
