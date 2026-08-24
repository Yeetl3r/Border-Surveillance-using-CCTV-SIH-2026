"""F6 -- unattended-baggage association and dwell, and F14 relay budget.

Synthetic tracks only. No GPU, no video, no models.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from sentinel.config import BaggageCfg, RelayCfg, SystemCfg
from sentinel.contracts import (
    Alert,
    Detection,
    ObjectGroup,
    ObservationKind,
    ThreatLevel,
    Track,
)
from sentinel.temporal import BaggageAssociator, BagState

T0 = datetime(2026, 8, 24, 12, 0, 0, tzinfo=timezone.utc)


def track(tid: int, group: ObjectGroup, xy_m: tuple[float, float],
          conf: float = 0.9) -> Track:
    det = Detection(bbox_px=(100, 100, 160, 300), class_id=0,
                    class_name=group.value, group=group, confidence=conf,
                    model_id="test")
    t = Track(track_id=tid, camera_id="cam01", group=group,
              class_name=group.value, latest=det, first_seen=T0)
    t.footpoint_m = xy_m
    return t


def cfg(**kw) -> BaggageCfg:
    base = dict(enabled=True, assoc_radius_m=2.5, detach_distance_m=4.0,
                dwell_s=15.0, reclaim_grace_s=2.0, min_bag_conf=0.3)
    base.update(kw)
    return BaggageCfg(**base)


# ------------------------------------------------------------------- states
def test_bag_near_person_becomes_accompanied():
    a = BaggageAssociator(cfg())
    a.update([track(1, ObjectGroup.PERSON, (0.0, 5.0)),
              track(2, ObjectGroup.BAG, (0.5, 5.0))], T0, "cam01")
    rec = a.bags[2]
    assert rec.state is BagState.ACCOMPANIED
    assert rec.owner_track_id == 1


def test_bag_never_near_anyone_is_orphan_and_never_alerts():
    """We do not claim abandonment for an object with no owner.

    Without an owner there is no abandonment EVENT -- only an object we cannot
    explain. Alerting on it would be asserting more than we know.
    """
    a = BaggageAssociator(cfg())
    for i in range(400):
        t = T0 + timedelta(seconds=i)
        obs = a.update([track(2, ObjectGroup.BAG, (0.0, 5.0))], t, "cam01")
        assert obs == []
    assert a.bags[2].state is BagState.ORPHAN


def test_owner_walks_away_and_dwell_expires():
    a = BaggageAssociator(cfg(dwell_s=15.0))
    bag = (0.0, 5.0)
    a.update([track(1, ObjectGroup.PERSON, (0.0, 5.0)),
              track(2, ObjectGroup.BAG, bag)], T0, "cam01")

    fired = []
    for i in range(1, 40):
        t = T0 + timedelta(seconds=i)
        # Owner recedes: at 4 m the timer starts.
        person = (0.0, 5.0 + i * 1.0)
        obs = a.update([track(1, ObjectGroup.PERSON, person),
                        track(2, ObjectGroup.BAG, bag)], t, "cam01")
        fired.extend(obs)

    assert fired, "dwell should have expired"
    first = fired[0]
    assert first.kind is ObservationKind.BAG_UNATTENDED
    # Detach begins once separation exceeds 4 m, i.e. at t=+4 s; the alert
    # follows dwell_s later.
    assert 15.0 <= (first.at - T0).total_seconds() <= 24.0
    assert first.detail["dwell_threshold_s"] == 15.0


def test_owner_returns_before_dwell_clears_the_timer():
    a = BaggageAssociator(cfg(dwell_s=15.0))
    bag = (0.0, 5.0)
    a.update([track(1, ObjectGroup.PERSON, (0.0, 5.0)),
              track(2, ObjectGroup.BAG, bag)], T0, "cam01")

    fired = []
    for i in range(1, 30):
        t = T0 + timedelta(seconds=i)
        # Away for 8 s, then back within reach and held there.
        person = (0.0, 12.0) if i < 8 else (0.0, 5.2)
        fired.extend(a.update([track(1, ObjectGroup.PERSON, person),
                               track(2, ObjectGroup.BAG, bag)], t, "cam01"))
    assert fired == []
    assert a.bags[2].state is BagState.ACCOMPANIED


def test_demo_and_operational_dwell_differ():
    """The profile overlay is the whole reason dwell is config, not a constant."""
    assert cfg(dwell_s=15.0).dwell_s == 15.0
    assert cfg(dwell_s=120.0).dwell_s == 120.0


def test_uncalibrated_falls_back_to_pixels_and_flags_it():
    a = BaggageAssociator(cfg())
    person = track(1, ObjectGroup.PERSON, (0.0, 0.0))
    bag = track(2, ObjectGroup.BAG, (0.0, 0.0))
    person.footpoint_m = None                # simulate an uncalibrated camera
    bag.footpoint_m = None
    a.update([person, bag], T0, "cam01")
    assert a.bags[2].degraded is True


def test_low_confidence_bags_are_ignored():
    a = BaggageAssociator(cfg(min_bag_conf=0.5))
    a.update([track(1, ObjectGroup.PERSON, (0.0, 5.0)),
              track(2, ObjectGroup.BAG, (0.5, 5.0), conf=0.2)], T0, "cam01")
    assert 2 not in a.bags


# ----------------------------------------------------------------- F14 relay
def test_relay_payload_stays_under_budget(tmp_path):
    """The budget is enforced by degrading the image, never by truncating JSON.

    A clipped JSON document is unparseable and therefore worthless; a grainier
    thumbnail is still a thumbnail.
    """
    import json

    from sentinel.alerting import RelayEncoder

    cfg_ = SystemCfg()
    cfg_.root_dir = str(tmp_path)
    cfg_.relay = RelayCfg(enabled=True, max_bytes=3072, crop_max_px=160,
                          jpeg_quality_start=60, jpeg_quality_min=15,
                          out_dir=str(tmp_path / "relay"))
    enc = RelayEncoder(cfg_)

    alert = Alert(kind=ObservationKind.TRIPWIRE_CROSS, camera_id="cam01",
                  threat=ThreatLevel.CRITICAL, confidence=0.91, opened_at=T0,
                  track_ids=(7,))
    alert.evidence.geometry = {"at": [1.2, 4.3], "direction": "INBOUND",
                               "distance_m": 1.44}

    rng = np.random.default_rng(0)
    # Noise is close to the worst case for JPEG: it does not compress.
    crop = rng.integers(0, 255, (640, 320, 3), dtype=np.uint8)

    payload, quality = enc.encode(alert, crop)
    assert len(payload) <= 3072
    doc = json.loads(payload)              # must always parse
    assert doc["cam"] == "cam01" and doc["th"] == 4
    assert doc["xy"] == [1.2, 4.3]
    assert quality == -1 or "img" in doc


def test_relay_without_an_image_still_carries_the_fields():
    import json

    from sentinel.alerting import RelayEncoder

    cfg_ = SystemCfg()
    cfg_.relay = RelayCfg(enabled=False, max_bytes=3072)
    enc = RelayEncoder(cfg_)
    alert = Alert(kind=ObservationKind.BAG_UNATTENDED, camera_id="cam02",
                  threat=ThreatLevel.WARNING, confidence=0.7, opened_at=T0)
    payload, quality = enc.encode(alert, None)
    doc = json.loads(payload)
    assert quality == -1 and "img" not in doc
    assert doc["cam"] == "cam02" and doc["k"] == "BG"
    assert len(payload) < 3072
