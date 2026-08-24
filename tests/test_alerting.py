"""Alert promotion, the threat matrix, and the lifecycle that lets alerts clear.

The regression this file exists to prevent: the reference prototype we assessed
kept its breached-target set append-only, so a single false positive latched
the threat display to CRITICAL for the rest of the run. An alert that cannot
clear is a bug, not a safety margin.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from sentinel.alerting import AlertManager, EvidenceWriter, RelayEncoder
from sentinel.config import load_config
from sentinel.contracts import (
    AlertState,
    ConfidenceBand,
    Detection,
    FrameBundle,
    IdentityLabel,
    IdentityVerdict,
    ObjectGroup,
    Observation,
    ObservationKind,
    ThreatLevel,
    Track,
)
from sentinel.storage import Repository
from sentinel.temporal import TemporalConfirmer

T0 = datetime(2026, 8, 24, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture()
def kit(tmp_path):
    cfg = load_config(".")
    cfg.root_dir = str(tmp_path)
    cfg.storage.db_path = str(tmp_path / "t.db")
    cfg.storage.evidence_dir = str(tmp_path / "ev")
    cfg.storage.clip_dir = str(tmp_path / "cl")
    cfg.relay.out_dir = str(tmp_path / "re")
    repo = Repository(cfg.storage.db_path)
    evidence = EvidenceWriter(cfg)
    # SentinelSystem does this at startup; the provenance block in every
    # evidence record depends on it.
    evidence.register_models(detector="yolo26s@test", tracker=cfg.tracker.kind,
                             embedder="osnet_x0_25@test")
    mgr = AlertManager(cfg, repo, evidence, RelayEncoder(cfg))
    return cfg, repo, mgr


def person_track(tid=1, label: IdentityLabel | None = None) -> Track:
    det = Detection(bbox_px=(100, 100, 160, 300), class_id=0, class_name="person",
                    group=ObjectGroup.PERSON, confidence=0.9, model_id="test")
    t = Track(track_id=tid, camera_id="cam01", group=ObjectGroup.PERSON,
              class_name="person", latest=det, first_seen=T0, age_frames=30)
    t.footpoint_m = (0.5, 4.0)
    if label is not None:
        t.identity = IdentityVerdict(label=label, similarity=0.8,
                                     threshold_used=0.666, model_id="osnet")
    return t


def bundle(idx=1, at=T0) -> FrameBundle:
    return FrameBundle(camera_id="cam01", frame_idx=idx, captured_at=at,
                       raw=np.zeros((720, 1280, 3), np.uint8))


def candidate(kind=ObservationKind.TRIPWIRE_CROSS, track=1, n=6):
    from sentinel.temporal.confirm import Candidate
    c = Candidate(subject_key=f"cam01|{kind.value}|{track}|z1", kind=kind,
                  camera_id="cam01")
    for i in range(n):
        t = T0 + timedelta(seconds=i * 0.1)
        o = Observation(kind=kind, camera_id="cam01", at=t, frame_idx=i,
                        confidence=0.9, track_ids=(track,),
                        geometry={"zone_id": "z1", "zone_label": "PERIM",
                                  "direction": "INBOUND", "metric": True},
                        detail={"class_name": "person"})
        c.hits.append((t, o))
        c.last_hit = t
    c.confirmed = True
    return c


# ------------------------------------------------------------- threat matrix
def test_threat_matrix_uses_identity(kit):
    _cfg, _repo, mgr = kit
    unknown = {1: person_track(1, IdentityLabel.UNIDENTIFIED)}
    friendly = {1: person_track(1, IdentityLabel.FRIENDLY)}
    k = ObservationKind.TRIPWIRE_CROSS
    assert mgr.threat_for(k, unknown, (1,)) is ThreatLevel.CRITICAL
    assert mgr.threat_for(k, friendly, (1,)) is ThreatLevel.FRIENDLY


def test_identity_never_yields_a_hostile_verdict(kit):
    """There is no HOSTILE threat level reachable from an identity verdict."""
    _cfg, _repo, mgr = kit
    for lbl in IdentityLabel:
        tracks = {1: person_track(1, lbl)}
        for kind in ObservationKind:
            th = mgr.threat_for(kind, tracks, (1,))
            assert th in set(ThreatLevel)
    assert not hasattr(ThreatLevel, "HOSTILE")


# ------------------------------------------------------------------ promotion
def test_promotion_creates_an_alert_with_evidence(kit):
    _cfg, repo, mgr = kit
    tracks = {1: person_track(1, IdentityLabel.UNIDENTIFIED)}
    out = mgr.promote([candidate()], bundle(), tracks, None, "h1")
    assert len(out) == 1
    a = out[0]
    assert a.state is AlertState.ACTIVE
    assert a.threat is ThreatLevel.CRITICAL
    assert a.evidence.resolvable            # keyframe written
    assert a.evidence.config_snapshot       # thresholds frozen
    assert a.supporting                     # carries its own proof
    assert repo.recent_alerts(10)


def test_threat_and_confidence_are_independent(kit):
    """F12: a low-confidence CRITICAL must remain CRITICAL.

    Collapsing the two axes into one score is what makes a display ambiguous
    about whether it is unsure or unconcerned.
    """
    _cfg, _repo, mgr = kit
    t = person_track(1, IdentityLabel.UNIDENTIFIED)
    t.latest.edge_clipped = True
    t.age_frames = 2
    out = mgr.promote([candidate()], bundle(), {1: t}, None, None)
    a = out[0]
    assert a.threat is ThreatLevel.CRITICAL         # colour unchanged
    assert a.confidence < 0.9                       # style downgraded
    assert "clipped" in a.rationale


def test_uncalibrated_geometry_caps_confidence(kit):
    _cfg, _repo, mgr = kit
    c = candidate()
    for _t, o in c.hits:
        o.geometry["degraded"] = True
        o.geometry["metric"] = False
    out = mgr.promote([c], bundle(), {1: person_track(1)}, None, None)
    a = out[0]
    assert a.confidence <= 0.55
    assert "PIXEL-SPACE" in a.rationale


def test_same_subject_does_not_double_alert(kit):
    _cfg, _repo, mgr = kit
    tracks = {1: person_track(1)}
    c = candidate()
    assert len(mgr.promote([c], bundle(), tracks, None, None)) == 1
    assert mgr.promote([c], bundle(), tracks, None, None) == []


# ------------------------------------------------------------------ lifecycle
def test_alert_clears_when_observations_stop(kit):
    """The regression guard. A threat state that only escalates is a bug."""
    cfg, _repo, mgr = kit
    conf = TemporalConfirmer(cfg.confirm)
    c = candidate()
    conf._candidates[c.subject_key] = c
    mgr.promote([c], bundle(), {1: person_track(1)}, None, None)
    assert mgr.highest_live is not ThreatLevel.INFO

    later = T0 + timedelta(seconds=cfg.alerts.clear_after_s + 5)
    closed = mgr.tick(later, conf)
    assert len(closed) == 1
    assert closed[0].state is AlertState.CLEARED
    assert mgr.highest_live is ThreatLevel.INFO      # sector returns to secure


def test_alert_expires_if_never_resolved(kit):
    cfg, _repo, mgr = kit
    conf = TemporalConfirmer(cfg.confirm)
    c = candidate()
    conf._candidates[c.subject_key] = c
    mgr.promote([c], bundle(), {1: person_track(1)}, None, None)
    way_later = T0 + timedelta(seconds=cfg.alerts.expire_after_s + 10)
    closed = mgr.tick(way_later, conf)
    assert closed[0].state is AlertState.EXPIRED


def test_highest_live_reads_the_live_set_not_a_running_max(kit):
    cfg, _repo, mgr = kit
    conf = TemporalConfirmer(cfg.confirm)
    crit = candidate(track=1)
    conf._candidates[crit.subject_key] = crit
    mgr.promote([crit], bundle(), {1: person_track(1, IdentityLabel.UNIDENTIFIED)},
                None, None)
    assert mgr.highest_live is ThreatLevel.CRITICAL
    mgr.tick(T0 + timedelta(seconds=999), conf)
    assert mgr.highest_live is ThreatLevel.INFO


# ------------------------------------------------------------------- bands
def test_confidence_bands():
    assert ConfidenceBand.from_score(0.9) is ConfidenceBand.HIGH
    assert ConfidenceBand.from_score(0.6) is ConfidenceBand.MEDIUM
    assert ConfidenceBand.from_score(0.2) is ConfidenceBand.LOW


def test_evidence_is_required_for_an_alert(kit, monkeypatch):
    """An alert with no retrievable evidence is an assertion; refuse to emit it."""
    _cfg, _repo, mgr = kit

    def blank(*a, **k):
        from sentinel.contracts import EvidenceRef
        return EvidenceRef()                 # no keyframe -> not resolvable

    monkeypatch.setattr(mgr.evidence, "build", blank)
    out = mgr.promote([candidate()], bundle(), {1: person_track(1)}, None, None)
    assert out == []
    assert mgr.counters["suppressed"] == 1


def test_evidence_round_trips_through_sql(kit):
    """Regression: the config snapshot must survive the read path.

    F13 requires that "why did this fire?" is answerable later, which depends
    on the thresholds frozen at fire time coming back out of the database.
    A column-name mapping bug previously parsed this field onto itself and
    then deleted it, so the API reported an alert with no snapshot.
    """
    _cfg, repo, mgr = kit
    tracks = {1: person_track(1, IdentityLabel.UNIDENTIFIED)}
    alert = mgr.promote([candidate()], bundle(), tracks, None, "h1")[0]

    row = repo.alert_with_evidence(alert.alert_id)
    assert row is not None
    assert row["config_snapshot"], "thresholds in force at fire time were lost"
    assert row["config_snapshot"]["profile"]
    assert row["config_snapshot"]["confirm"]["window_s"] > 0
    assert row["detections"] and row["detections"][0]["model_id"]
    assert row["model_ids"]
    assert row["geometry"]["zone_label"] == "PERIM"
    assert row["keyframe_path"]
    # None of the raw column names should leak through the API surface.
    for col in ("detections_json", "model_ids_json", "geometry_json"):
        assert col not in row
