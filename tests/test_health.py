"""F9 -- camera health and anti-tamper.

Two regressions this file exists to prevent, both found by running the system
against real footage rather than by reading it.

1. **A static night scene read as a dead camera.** Freeze detection compared
   consecutive frames with a mean-difference threshold. On the project's own
   demo clip, 90% of *live* frames fall below that threshold, because a night
   scene at low bitrate barely changes. The system reported SIGNAL_LOST on a
   perfectly healthy feed.

2. **An absolute sharpness floor does not transfer between cameras.** A floor
   tuned against a dark IR scene (healthy Laplacian ~190) is meaningless on a
   bright indoor camera (healthy ~970), and a hand held over a well-lit lens
   stays sharp -- it never approaches an absolute floor at all.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from sentinel.config import HealthCfg
from sentinel.contracts import HealthState
from sentinel.ingest.health import HealthMonitor, laplacian_variance, ssim

W, H = 320, 180
_RNG = np.random.default_rng(1)

# A real camera returns the SAME scene every frame with small variation --
# sensor noise, a leaf moving, someone walking. Generating fresh noise per
# frame is not a healthy feed, it is a different scene every frame, and no
# structural-similarity check should call that stable.
_SCENE = _RNG.integers(0, 255, (H, W, 3), dtype=np.uint8)


def _palm() -> np.ndarray:
    """A surface like a hand held over the lens.

    Deliberately not blurred noise -- blurring noise destroys detail at every
    scale and produces something no camera ever sees. A palm has smooth
    gradients, a few coarse creases and fine grain, which leaves it genuinely
    sharp in absolute terms while carrying a fraction of a real scene's
    high-frequency energy. That combination is the whole difficulty.
    """
    g = np.tile(np.linspace(70, 190, W, dtype=np.uint8), (H, 1))
    surf = cv2.merge([g, g, g]).astype(np.uint8)
    for y in (H // 4, H // 2, 3 * H // 4):
        cv2.line(surf, (0, y + 10), (W - 1, y - 10), (60, 55, 50), 3, cv2.LINE_AA)
    return cv2.add(surf, _RNG.integers(0, 26, (H, W, 3), dtype=np.uint8))


_OTHER = _palm()


def _jitter(base: np.ndarray, i: int) -> np.ndarray:
    """The same view again, with a little sensor noise on top.

    Never bit-identical -- that is reserved for a genuinely frozen feed.
    """
    n = np.zeros_like(base)
    n.flat[(i * 977) % n.size] = 1 + (i % 3)
    return cv2.add(base, n)


def live(i: int) -> np.ndarray:
    """A healthy frame from a stable scene."""
    return _jitter(_SCENE, i)


def covered(i: int = 0) -> np.ndarray:
    """A covered lens: no structure, no detail, but still a live sensor."""
    return _jitter(cv2.GaussianBlur(np.full((H, W, 3), 9, np.uint8), (31, 31), 0), i)


def obstruction(i: int = 0) -> np.ndarray:
    """Something held over the lens: a different surface, still sharp."""
    return _jitter(_OTHER, i)


def cfg(**kw) -> HealthCfg:
    base = dict(enabled=True, consecutive_bad=3, blur_warmup_frames=10)
    base.update(kw)
    return HealthCfg(**base)


def settle(mon: HealthMonitor, frames: int = 60) -> None:
    """Feed healthy frames so the baselines and reference establish."""
    for i in range(frames):
        mon.update(live(i))


# ── primitives ────────────────────────────────────────────────────────────
def test_laplacian_falls_when_detail_is_lost():
    assert laplacian_variance(cv2.cvtColor(live(1), cv2.COLOR_BGR2GRAY)) > \
           laplacian_variance(cv2.cvtColor(covered(), cv2.COLOR_BGR2GRAY))


def test_ssim_identity_and_difference():
    a = cv2.cvtColor(live(2), cv2.COLOR_BGR2GRAY)
    assert ssim(a, a) == pytest.approx(1.0, abs=1e-6)
    b = cv2.cvtColor(covered(), cv2.COLOR_BGR2GRAY)
    assert ssim(a, b) < 0.5


# ── healthy feeds must stay healthy ───────────────────────────────────────
def test_healthy_feed_reports_ok():
    mon = HealthMonitor(cfg(), "cam")
    for i in range(120):
        st = mon.update(live(i))
    assert st.state is HealthState.OK
    assert st.detail == ""


def test_near_static_scene_is_not_a_frozen_feed():
    """The regression from finding #1.

    Consecutive frames that are *similar* but not identical are what a real
    static night scene produces. Only bit-identical frames indicate a dead feed.
    """
    mon = HealthMonitor(cfg(), "cam")
    st = None
    for i in range(160):
        st = mon.update(live(i))       # similar every frame, never identical
    assert st.state is HealthState.OK, st.detail


def test_truly_identical_frames_are_signal_lost():
    mon = HealthMonitor(cfg(freeze_max_frames=10), "cam")
    settle(mon)
    frozen = live(99)
    st = None
    for _ in range(40):
        st = mon.update(frozen)
    assert st.state is HealthState.SIGNAL_LOST
    assert "frozen" in st.detail


# ── tampering ─────────────────────────────────────────────────────────────
def test_covered_lens_is_tampered_within_the_latency_budget():
    """The brief's only hard latency requirement: trip within one second."""
    c = cfg()
    mon = HealthMonitor(c, "cam")
    settle(mon)

    fps, tripped_at = 15.0, None
    for i in range(1, 40):
        st = mon.update(covered(i))
        if st.state is not HealthState.OK:
            tripped_at = i
            break
    assert tripped_at is not None
    assert tripped_at / fps <= c.check_within_s
    assert st.state is HealthState.TAMPERED


def test_obstruction_that_stays_sharp_is_still_tampering():
    """The regression from finding #2.

    An object held over the lens -- a palm, a sticker, a leaf -- replaces the
    view with something that is still sharp and still structured. Neither the
    absolute blur floor nor SSIM alone separates it from a healthy frame; the
    conjunction of relative collapse in both does.
    """
    mon = HealthMonitor(cfg(), "cam")
    settle(mon)

    # A different, softer, but genuinely detailed surface -- nothing like a
    # black frame, and far above any absolute sharpness floor.
    assert laplacian_variance(cv2.cvtColor(obstruction(), cv2.COLOR_BGR2GRAY)) > cfg().blur_min

    st = None
    for i in range(12):
        st = mon.update(obstruction(i))
    assert st.state is HealthState.TAMPERED, st.detail
    assert "obstruct" in st.detail or "collapse" in st.detail


def test_baseline_does_not_learn_the_tamper():
    """A tamper must not teach the camera that the tamper is normal.

    Baselines and the reference frame update only on frames that raised no
    fault. Keying that on the *reported state* instead let the covered lens be
    absorbed during the confirmation delay, and the tamper erased its own
    evidence.
    """
    mon = HealthMonitor(cfg(), "cam")
    settle(mon)
    healthy_baseline = mon._blur_baseline

    for i in range(120):
        mon.update(covered(i))

    assert mon._blur_baseline == pytest.approx(healthy_baseline, rel=1e-6)
    assert mon.update(covered(999)).state is HealthState.TAMPERED


def test_recovery_returns_to_ok():
    mon = HealthMonitor(cfg(), "cam")
    settle(mon)
    for i in range(15):
        mon.update(covered(i))
    assert mon.update(covered(50)).state is HealthState.TAMPERED

    st = None
    for i in range(60):
        st = mon.update(live(3000 + i))
    assert st.state is HealthState.OK, st.detail


def test_disabled_monitor_is_inert():
    mon = HealthMonitor(cfg(enabled=False), "cam")
    for i in range(30):
        st = mon.update(covered(i))
    assert st.state is HealthState.OK
