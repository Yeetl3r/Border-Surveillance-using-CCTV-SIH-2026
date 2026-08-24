"""Operator-stream image enhancement. NEVER in the detection path.

It is intuitive to brighten a dark frame before running the detector. The
research consensus is against it: enhancement optimises perceptual quality,
not the detection objective, and the artefacts it introduces damage
small-object detection specifically -- which is precisely the regime this
system works in (a person at 150 m is 15-25 px tall).

So the detector sees ``FrameBundle.raw`` and the human sees ``FrameBundle.view``.
This module only ever writes the latter. The separation is enforced by the
frame contract; this docstring records why it exists.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..config import EnhanceCfg


def clahe(frame: np.ndarray, clip: float = 2.5, grid: int = 8) -> np.ndarray:
    """Contrast-limited adaptive histogram equalisation on luminance only.

    Operating on the L channel of LAB leaves chroma alone, which avoids the
    colour shift that equalising BGR channels independently produces.
    """
    if frame.ndim == 2:
        return cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(frame)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def dark_channel_prior(
    frame: np.ndarray, omega: float = 0.85, patch: int = 11, t0: float = 0.1
) -> np.ndarray:
    """Single-image dehazing via the dark channel prior (He et al.).

    Fog and darkness are different problems and want different treatments: DCP
    recovers contrast lost to scattering, CLAHE recovers contrast lost to
    under-exposure. Neither helps the other case much, which is why ``mode``
    selects rather than always applying both.
    """
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    img = frame.astype(np.float32) / 255.0

    dark = cv2.erode(np.min(img, axis=2),
                     cv2.getStructuringElement(cv2.MORPH_RECT, (patch, patch)))

    # Atmospheric light: mean of the brightest 0.1% of dark-channel pixels.
    flat = dark.ravel()
    n = max(1, int(flat.size * 0.001))
    idx = np.argpartition(flat, -n)[-n:]
    A = np.maximum(img.reshape(-1, 3)[idx].mean(axis=0), 1e-3)

    t = 1.0 - omega * cv2.erode(
        np.min(img / A, axis=2),
        cv2.getStructuringElement(cv2.MORPH_RECT, (patch, patch)))
    # Edge-preserving refinement; a box-filtered t produces halos at depth
    # discontinuities, which look like artefacts an evaluator will ask about.
    t = cv2.ximgproc.guidedFilter(
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), t.astype(np.float32), 40, 1e-3
    ) if hasattr(cv2, "ximgproc") else cv2.GaussianBlur(t, (41, 41), 0)

    t = np.maximum(t, t0)[..., None]
    out = (img - A) / t + A
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def enhance_for_operator(frame: np.ndarray, cfg: EnhanceCfg) -> np.ndarray | None:
    """Produce the operator-viewable frame, or None if enhancement is off.

    Returning None rather than a copy means the pipeline can skip a whole
    buffer allocation per frame when enhancement is disabled.
    """
    if not cfg.enabled or cfg.mode == "none":
        return None

    out = frame
    if "dcp" in cfg.mode:
        out = dark_channel_prior(out, cfg.dcp_omega, cfg.dcp_patch, cfg.dcp_t0)
    if "clahe" in cfg.mode:
        out = clahe(out, cfg.clahe_clip, cfg.clahe_grid)
    return out if out is not frame else frame.copy()
