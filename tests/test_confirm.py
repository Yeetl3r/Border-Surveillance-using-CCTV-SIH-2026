"""F7 -- temporal confirmation.

These tests are the reason the temporal layer has no GPU dependency. The
suppression policy is the thing that decides whether this system is
operationally useful or operationally ignored, so it must be testable without
a model, a camera, or a video file.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from sentinel.config import ConfirmCfg
from sentinel.contracts import Observation, ObservationKind
from sentinel.temporal import TemporalConfirmer

T0 = datetime(2026, 8, 24, 12, 0, 0, tzinfo=timezone.utc)


def obs(t: datetime, kind=ObservationKind.ZONE_PRESENCE, track=1, conf=0.9,
        cam="cam01") -> Observation:
    return Observation(kind=kind, camera_id=cam, at=t, frame_idx=0,
                       confidence=conf, track_ids=(track,),
                       geometry={"zone_id": "z1"})


def feed(c: TemporalConfirmer, n_ticks: int, hit_every: int, fps: float = 25.0,
         kind=ObservationKind.ZONE_PRESENCE, cam="cam01"):
    """Advance n_ticks frames, submitting an observation every hit_every frames."""
    newly = []
    for i in range(n_ticks):
        t = T0 + timedelta(seconds=i / fps)
        c.tick(cam, t)
        if hit_every and i % hit_every == 0:
            c.submit(obs(t, kind=kind, cam=cam))
        newly.extend(c.evaluate(t))
    return newly


def cfg(**kw) -> ConfirmCfg:
    base = dict(window_s=2.0, min_ratio=0.6, min_hits=3, overrides={})
    base.update(kw)
    return ConfirmCfg(**base)


def test_single_observation_never_confirms():
    """One frame of evidence is not an event. This is the whole point of F7."""
    c = TemporalConfirmer(cfg())
    c.tick("cam01", T0)
    c.submit(obs(T0))
    assert c.evaluate(T0) == []


def test_sustained_presence_confirms():
    c = TemporalConfirmer(cfg())
    newly = feed(c, n_ticks=50, hit_every=1)
    assert len(newly) == 1
    assert newly[0].confirmed


def test_confirms_once_not_every_frame():
    """A subject standing in a zone is one alert, not one alert per frame."""
    c = TemporalConfirmer(cfg())
    newly = feed(c, n_ticks=200, hit_every=1)
    assert len(newly) == 1


def test_sparse_hits_fail_the_ratio():
    """Enough hits to pass min_hits, but too few of the frames that arrived.

    This is the flickering-detector case: an object detected in 1 frame of
    every 5 is not a confident observation, and must not alert.
    """
    c = TemporalConfirmer(cfg(min_hits=3, min_ratio=0.6))
    newly = feed(c, n_ticks=60, hit_every=5)      # ratio ~0.2
    assert newly == []


def test_ratio_is_frame_rate_independent():
    """The core reason F7 is specified in seconds, not frames.

    The same real-world event -- a subject present for two seconds -- must
    confirm identically at 25 fps and at 8 fps. A raw frame-count threshold
    would silently change meaning between them, and 8 fps is exactly what
    degraded backhaul produces.
    """
    fast = TemporalConfirmer(cfg())
    slow = TemporalConfirmer(cfg())
    got_fast = feed(fast, n_ticks=50, hit_every=1, fps=25.0)
    got_slow = feed(slow, n_ticks=16, hit_every=1, fps=8.0)
    assert len(got_fast) == 1
    assert len(got_slow) == 1


def test_frame_count_rule_would_have_differed():
    """Demonstrates the bug the ratio design avoids.

    At 8 fps a 2-second window holds 16 frames; at 25 fps it holds 50. A
    fixed min_hits of 20 would confirm at 25 fps and never confirm at 8 fps
    for the identical real-world event.
    """
    c_fast = TemporalConfirmer(cfg(min_hits=20, min_ratio=0.0))
    c_slow = TemporalConfirmer(cfg(min_hits=20, min_ratio=0.0))
    assert len(feed(c_fast, 50, 1, fps=25.0)) == 1
    assert len(feed(c_slow, 16, 1, fps=8.0)) == 0


def test_window_expires_old_hits():
    c = TemporalConfirmer(cfg(window_s=1.0, min_hits=3, min_ratio=0.5))
    for i in range(3):
        t = T0 + timedelta(seconds=i * 0.05)
        c.tick("cam01", t)
        c.submit(obs(t))
    # Jump well past the window; the accumulated hits must age out.
    later = T0 + timedelta(seconds=5)
    c.tick("cam01", later)
    assert c.evaluate(later) == []


def test_per_kind_overrides_apply():
    """Health faults must trip faster than perimeter events."""
    c = TemporalConfirmer(ConfirmCfg(
        window_s=5.0, min_ratio=0.9, min_hits=50,
        overrides={"HEALTH_FAULT": {"window_s": 0.5, "min_ratio": 0.5, "min_hits": 2}}))
    newly = feed(c, n_ticks=10, hit_every=1, kind=ObservationKind.HEALTH_FAULT)
    assert len(newly) == 1
    rule = c.rule_for(ObservationKind.HEALTH_FAULT)
    assert rule.min_hits == 2
    assert c.rule_for(ObservationKind.ZONE_PRESENCE).min_hits == 50


def test_separate_subjects_confirm_separately():
    c = TemporalConfirmer(cfg())
    for i in range(50):
        t = T0 + timedelta(seconds=i / 25.0)
        c.tick("cam01", t)
        c.submit(obs(t, track=1))
        c.submit(obs(t, track=2))
        newly = c.evaluate(t)
        if newly:
            assert len({n.subject_key for n in newly}) == len(newly)
    keys = {c.state(k).subject_key for k in
            [obs(T0, track=1).subject_key, obs(T0, track=2).subject_key]}
    assert len(keys) == 2


def test_sweep_drops_stale_candidates():
    c = TemporalConfirmer(cfg())
    feed(c, n_ticks=50, hit_every=1)
    c.sweep(T0 + timedelta(seconds=3600), stale_after_s=60)
    assert c.pending == 0


def test_health_check_within_one_second():
    """The brief's only hard latency requirement.

    A camera-health fault must reach confirmation within a second of onset.
    """
    c = TemporalConfirmer(ConfirmCfg(
        window_s=2.0, min_ratio=0.6, min_hits=3,
        overrides={"HEALTH_FAULT": {"window_s": 0.8, "min_ratio": 0.7, "min_hits": 2}}))
    confirmed_at = None
    for i in range(30):
        t = T0 + timedelta(seconds=i / 25.0)
        c.tick("cam01", t)
        c.submit(obs(t, kind=ObservationKind.HEALTH_FAULT))
        if c.evaluate(t):
            confirmed_at = t
            break
    assert confirmed_at is not None
    assert (confirmed_at - T0).total_seconds() <= 1.0
