"""Tier 8 -- the alert manager.

**This is the only component permitted to construct an Alert.** Detection
produces observations; promotion is one component's responsibility so that the
suppression policy lives in exactly one auditable place rather than being
scattered through a detection loop.

Three rules it enforces:

1. *Nothing is promoted without temporal confirmation.* The confirmer decides;
   this class acts on its verdict.
2. *Threat and confidence are separate.* Threat comes from the configured
   matrix (observation kind x identity verdict). Confidence comes from the
   evidence. They are never multiplied into one number, because a
   low-confidence critical event and a high-confidence trivial one must not
   render the same.
3. *Alerts can clear.* An alert whose supporting observations stop arriving
   goes CLEARED; one that never resolves goes EXPIRED. A monotonic threat
   state that only ever escalates is a bug -- one false positive would latch
   the display permanently and the operator would learn to ignore it.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Callable

import numpy as np

from ..config import SystemCfg
from ..contracts import (
    Alert,
    AlertState,
    ConfidenceBand,
    FrameBundle,
    IdentityLabel,
    Observation,
    ObservationKind,
    ThreatLevel,
    Track,
)
from ..storage import Repository
from ..temporal.confirm import Candidate
from .evidence import ClipBuffer, EvidenceWriter
from .relay import RelayEncoder


class AlertManager:
    """Promotes confirmed candidates to alerts and owns their lifecycle."""

    def __init__(
        self,
        cfg: SystemCfg,
        repo: Repository,
        evidence: EvidenceWriter,
        relay: RelayEncoder,
        on_alert: Callable[[Alert, str], None] | None = None,
    ) -> None:
        self.cfg = cfg
        self.repo = repo
        self.evidence = evidence
        self.relay = relay
        self.on_alert = on_alert
        self.live: dict[str, Alert] = {}          # subject_key -> Alert
        self.recent: list[Alert] = []
        self.counters = {"promoted": 0, "cleared": 0, "expired": 0, "suppressed": 0}

    # -- threat resolution ---------------------------------------------------
    def _identity_key(self, tracks: dict[int, Track], track_ids) -> str:
        """Which identity column of the threat matrix applies."""
        for tid in track_ids:
            t = tracks.get(tid)
            if t is not None and t.identity is not None:
                return t.identity.label.value
        return "NONE"

    def threat_for(
        self, kind: ObservationKind, tracks: dict[int, Track], track_ids
    ) -> ThreatLevel:
        row = self.cfg.alerts.threat_matrix.get(kind.value, {})
        key = self._identity_key(tracks, track_ids)
        name = row.get(key) or row.get("NONE") or "WATCH"
        try:
            return ThreatLevel(name)
        except ValueError:
            return ThreatLevel.WATCH

    # -- confidence ----------------------------------------------------------
    def confidence_for(
        self, obs: list[Observation], tracks: dict[int, Track], track_ids
    ) -> tuple[float, list[str]]:
        """Evidential confidence, with the reasons it was discounted.

        Deliberately pessimistic. Under-stating confidence costs a hatched box;
        over-stating it costs operator trust, which does not come back.
        """
        if not obs:
            return 0.0, ["no supporting observations"]

        base = float(np.mean([o.confidence for o in obs[-12:]]))
        notes: list[str] = []

        # Uncalibrated geometry: the test ran in pixels, so any distance claim
        # attached to it is not metric. Cap what we will assert.
        if any(o.geometry.get("degraded") for o in obs):
            base = min(base, 0.55)
            notes.append("uncalibrated geometry (pixel-space fallback)")

        for tid in track_ids:
            t = tracks.get(tid)
            if t is None:
                continue
            if t.latest.edge_clipped:
                base *= 0.85
                notes.append("subject box clipped by frame edge")
            if t.age_frames < 5:
                base *= 0.9
                notes.append("track only just established")
            if t.identity is not None and \
                    t.identity.label is IdentityLabel.INSUFFICIENT_QUALITY:
                # Say WHY identity is undetermined. "Crop below quality gate"
                # and "nobody is enrolled" are different facts, and reporting
                # the wrong one sends whoever reads the alert to the wrong fix.
                why = t.meta.get("identity_note", "crop below quality gate")
                notes.append(f"identity undetermined ({why})")

        return float(np.clip(base, 0.0, 1.0)), notes

    # -- promotion -----------------------------------------------------------
    def promote(
        self,
        candidates: list[Candidate],
        bundle: FrameBundle,
        tracks: dict[int, Track],
        clip: ClipBuffer | None,
        homography_id: str | None,
    ) -> list[Alert]:
        out: list[Alert] = []
        for c in candidates:
            if c.subject_key in self.live:
                continue                          # already alerting on this subject

            if len(self.live) >= self.cfg.alerts.max_live:
                self.counters["suppressed"] += 1
                continue

            obs = [o for _t, o in c.hits]
            track_ids = obs[-1].track_ids if obs else ()
            threat = self.threat_for(c.kind, tracks, track_ids)
            conf, notes = self.confidence_for(obs, tracks, track_ids)

            alert = Alert(
                kind=c.kind,
                camera_id=c.camera_id,
                threat=threat,
                confidence=conf,
                opened_at=obs[0].at if obs else bundle.captured_at,
                state=AlertState.ACTIVE,
                updated_at=bundle.captured_at,
                track_ids=tuple(track_ids),
                supporting=obs[-12:],
                subject_key=c.subject_key,
            )
            alert.rationale = self._rationale(alert, obs, tracks, notes)

            alert.evidence = self.evidence.build(
                alert, bundle.raw, tracks, clip, homography_id)
            if not alert.evidence.resolvable:
                # An alert with no retrievable evidence is an assertion. Refuse
                # to emit it rather than show the operator something unauditable.
                self.counters["suppressed"] += 1
                continue

            self.live[c.subject_key] = alert
            self.recent.append(alert)
            del self.recent[:-200]
            self.counters["promoted"] += 1

            self.repo.upsert_alert(alert)
            self.repo.write_evidence(alert)
            self.repo.write_observations(obs, alert.alert_id)
            self._relay(alert, bundle, tracks)

            if self.on_alert:
                self.on_alert(alert, "opened")
            out.append(alert)
        return out

    def _relay(self, alert: Alert, bundle: FrameBundle, tracks: dict[int, Track]) -> None:
        if not self.cfg.relay.enabled:
            return
        crop = None
        subject = next((tracks[t] for t in alert.track_ids if t in tracks), None)
        if subject is not None:
            x1, y1, x2, y2 = (int(max(0, v)) for v in subject.latest.bbox_px)
            h, w = bundle.raw.shape[:2]
            c = bundle.raw[y1:min(h, y2), x1:min(w, x2)]
            crop = c if c.size else None
        try:
            res = self.relay.emit(alert, crop)
            alert.relay_sent = True
            self.repo.log_relay(alert.alert_id, res["bytes"],
                                res["jpeg_quality"], res["path"])
        except AssertionError:
            alert.relay_sent = False              # budget breach: recorded, not fatal

    # -- lifecycle -----------------------------------------------------------
    def tick(self, now: datetime, confirmer) -> list[Alert]:
        """Clear and expire live alerts. Call once per frame."""
        changed: list[Alert] = []
        clear_after = timedelta(seconds=self.cfg.alerts.clear_after_s)
        expire_after = timedelta(seconds=self.cfg.alerts.expire_after_s)

        for key, alert in list(self.live.items()):
            cand = confirmer.state(key)
            last = cand.last_hit if cand else None

            if last is not None and now - last < clear_after:
                alert.updated_at = now
                continue

            if now - alert.opened_at >= expire_after:
                alert.state = AlertState.EXPIRED
                self.counters["expired"] += 1
            else:
                alert.state = AlertState.CLEARED
                self.counters["cleared"] += 1

            alert.cleared_at = now
            alert.updated_at = now
            self.live.pop(key, None)
            confirmer.forget(key)
            self.repo.upsert_alert(alert)
            if self.on_alert:
                self.on_alert(alert, "closed")
            changed.append(alert)

        return changed

    # -- presentation --------------------------------------------------------
    def _rationale(self, alert, obs, tracks, notes) -> str:
        """One human-readable line saying why this fired."""
        g = obs[-1].geometry if obs else {}
        d = obs[-1].detail if obs else {}
        subj = ", ".join(f"#{t}" for t in alert.track_ids) or "scene"
        cls = d.get("class_name", "")
        metric = g.get("metric", False)

        if alert.kind is ObservationKind.TRIPWIRE_CROSS:
            head = (f"{cls} {subj} crossed {g.get('zone_label', 'tripwire')} "
                    f"{g.get('direction', '').lower()}")
        elif alert.kind is ObservationKind.ZONE_PRESENCE:
            head = f"{cls} {subj} present in {g.get('zone_label', 'restricted zone')}"
        elif alert.kind is ObservationKind.PROXIMITY:
            dist = g.get("distance_m")
            head = (f"{cls} {subj} within "
                    f"{dist:.1f} m of {g.get('zone_label', 'asset')}"
                    if dist is not None else
                    f"{cls} {subj} near {g.get('zone_label', 'asset')} (pixel-space)")
        elif alert.kind is ObservationKind.BAG_UNATTENDED:
            head = (f"bag #{d.get('bag_track')} unattended for "
                    f"{d.get('dwell_s', 0):.0f} s "
                    f"(threshold {d.get('dwell_threshold_s', 0):.0f} s)")
        elif alert.kind is ObservationKind.STANCE_ANOMALY:
            head = f"{subj} {d.get('stance', '').lower()} stance sustained"
        elif alert.kind is ObservationKind.IDENTITY_UNKNOWN:
            head = f"{subj} not matched to any enrolled friendly"
        elif alert.kind is ObservationKind.HEALTH_FAULT:
            head = f"camera health: {d.get('state', '')} -- {d.get('reason', '')}"
        else:
            head = alert.kind.value

        bits = [head, f"confirmed over {len(obs)} observations"]
        if not metric and alert.kind in (ObservationKind.PROXIMITY,
                                         ObservationKind.TRIPWIRE_CROSS):
            bits.append("PIXEL-SPACE (camera uncalibrated)")
        if notes:
            bits.append("; ".join(dict.fromkeys(notes)))
        return " | ".join(bits)

    def snapshot(self) -> list[dict]:
        return [as_payload(a) for a in sorted(
            self.live.values(), key=lambda a: -a.threat.severity)]

    @property
    def highest_live(self) -> ThreatLevel:
        """Threat level of the whole sector: the worst LIVE alert, not the worst ever.

        Reading the live set rather than a running maximum is what allows the
        sector to return to secure once subjects leave.
        """
        if not self.live:
            return ThreatLevel.INFO
        return max((a.threat for a in self.live.values()), key=lambda t: t.severity)


def as_payload(a: Alert) -> dict:
    """Wire form for the UI: threat and confidence stay separate fields (F12)."""
    return {
        "alert_id": a.alert_id,
        "camera_id": a.camera_id,
        "kind": a.kind.value,
        "threat": a.threat.value,
        "threat_severity": a.threat.severity,
        "confidence": round(a.confidence, 3),
        "confidence_band": a.confidence_band.value,
        "state": a.state.value,
        "opened_at": a.opened_at.isoformat(),
        "cleared_at": a.cleared_at.isoformat() if a.cleared_at else None,
        "track_ids": list(a.track_ids),
        "rationale": a.rationale,
        "keyframe": a.evidence.keyframe_path,
        "crop": a.evidence.crop_path,
        "clip": a.evidence.clip_path,
        "geometry": a.evidence.geometry,
        "relay_sent": a.relay_sent,
        "supporting_count": len(a.supporting),
    }
