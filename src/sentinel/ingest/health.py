"""F9 -- anti-tamper and camera health.

Adversaries disable cameras before crossing. A surveillance system that cannot
tell "nothing is happening" from "I have been blinded" is reporting a clear
sector when it has actually stopped seeing.

Three cheap signals, no model:

* **Laplacian variance** -- high-frequency energy. Collapses when the lens is
  covered, sprayed, defocused, or fogged.
* **SSIM against a rolling reference** -- structural drift. Collapses when the
  camera is repositioned or the view is occluded, even if the image stays sharp.
* **Frame-difference floor** -- a frozen feed (looped or stalled) produces
  identical frames, which no scene ever does.

SSIM is implemented here rather than pulled from scikit-image: it is twenty
lines, it removes a dependency from an air-gapped install, and it lets us run
it on a downscaled grey image, which is the only sensible way to afford it
per-frame.
"""

from __future__ import annotations

from datetime import datetime, timezone

import cv2
import numpy as np

from ..config import HealthCfg
from ..contracts import HealthState, HealthStatus


def laplacian_variance(grey: np.ndarray) -> float:
    """Focus / obscuration measure. Higher is sharper."""
    return float(cv2.Laplacian(grey, cv2.CV_64F).var())


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Structural similarity between two single-channel images.

    Standard Wang et al. formulation with an 11x11 Gaussian window,
    sigma = 1.5, on images scaled to [0, 1].
    """
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    a = a.astype(np.float64) / 255.0
    b = b.astype(np.float64) / 255.0

    c1, c2 = 0.01 ** 2, 0.03 ** 2
    win, sigma = (11, 11), 1.5

    mu_a = cv2.GaussianBlur(a, win, sigma)
    mu_b = cv2.GaussianBlur(b, win, sigma)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b

    sig_a2 = cv2.GaussianBlur(a * a, win, sigma) - mu_a2
    sig_b2 = cv2.GaussianBlur(b * b, win, sigma) - mu_b2
    sig_ab = cv2.GaussianBlur(a * b, win, sigma) - mu_ab

    num = (2 * mu_ab + c1) * (2 * sig_ab + c2)
    den = (mu_a2 + mu_b2 + c1) * (sig_a2 + sig_b2 + c2)
    return float(np.mean(num / np.maximum(den, 1e-12)))


class HealthMonitor:
    """Per-camera health tracker. One instance per stream; holds the reference."""

    def __init__(self, cfg: HealthCfg, camera_id: str) -> None:
        self.cfg = cfg
        self.camera_id = camera_id
        self._ref: np.ndarray | None = None
        self._last_small: np.ndarray | None = None
        self._bad_streak = 0
        self._good_streak = 0
        self._frozen = 0
        self._state = HealthState.OK
        self._since = datetime.now(timezone.utc)
        self._last_detail = ""

    def _small_grey(self, frame: np.ndarray) -> np.ndarray:
        s = self.cfg.ssim_size
        h, w = frame.shape[:2]
        scale = s / max(1, max(h, w))
        small = cv2.resize(frame, (max(8, int(w * scale)), max(8, int(h * scale))),
                           interpolation=cv2.INTER_AREA)
        if small.ndim == 3:
            small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        return small

    def update(self, frame: np.ndarray) -> HealthStatus:
        if not self.cfg.enabled:
            return HealthStatus(state=HealthState.OK, blur_score=0.0, ssim_ref=1.0)

        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        blur = laplacian_variance(grey)
        small = self._small_grey(frame)

        if self._ref is None:
            self._ref = small.astype(np.float32)

        s = ssim(small, self._ref.astype(np.uint8))

        # Frozen-feed detection requires BIT-IDENTICAL consecutive frames, not
        # merely similar ones. Measured on this project's own demo footage: 90%
        # of live frames differ by a mean of less than 0.35 grey levels, because
        # a night scene at low bitrate is mostly static. A mean-difference
        # threshold therefore reports a healthy camera as a dead one -- exactly
        # the false alarm this system exists to suppress.
        #
        # Exact equality separates cleanly: the longest run of identical frames
        # in live footage measured 1, while a genuinely frozen feed produces an
        # unbounded run.
        if self._last_small is not None:
            identical = int(cv2.absdiff(small, self._last_small).max()) == 0
            self._frozen = self._frozen + 1 if identical else 0
        self._last_small = small

        faults: list[str] = []
        if blur < self.cfg.blur_min:
            faults.append(f"low sharpness ({blur:.0f} < {self.cfg.blur_min:.0f})")
        if s < self.cfg.ssim_min:
            faults.append(f"structural drift (SSIM {s:.2f} < {self.cfg.ssim_min:.2f})")
        if self._frozen >= self.cfg.freeze_max_frames:
            faults.append(f"feed frozen ({self._frozen} identical frames)")

        if faults:
            self._bad_streak += 1
            self._good_streak = 0
        else:
            self._good_streak += 1
            self._bad_streak = 0

        new_state = self._state
        if self._bad_streak >= self.cfg.consecutive_bad:
            if self._frozen >= self.cfg.freeze_max_frames:
                new_state = HealthState.SIGNAL_LOST
            elif blur < self.cfg.blur_min * 0.45 or s < self.cfg.ssim_min * 0.6:
                new_state = HealthState.TAMPERED     # decisive loss, not weather
            else:
                new_state = HealthState.DEGRADED
            self._last_detail = "; ".join(faults)
        elif self._good_streak >= self.cfg.consecutive_bad:
            new_state = HealthState.OK
            self._last_detail = ""

        if new_state != self._state:
            self._state = new_state
            self._since = datetime.now(timezone.utc)

        # Only fold healthy frames into the reference, and slowly. A fast EMA
        # would absorb a lens cover within a second and then report OK -- the
        # tamper would erase its own evidence.
        if self._state == HealthState.OK:
            a = self.cfg.ref_update_alpha
            self._ref = (1 - a) * self._ref + a * small.astype(np.float32)

        return HealthStatus(
            state=self._state, blur_score=round(blur, 2), ssim_ref=round(s, 4),
            frozen_frames=self._frozen, since=self._since, detail=self._last_detail)

    def reset_reference(self) -> None:
        """Called after a deliberate repositioning, so drift is measured afresh."""
        self._ref = None
        self._bad_streak = self._good_streak = self._frozen = 0
