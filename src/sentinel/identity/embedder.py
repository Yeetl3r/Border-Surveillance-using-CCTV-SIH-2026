"""Appearance embedding with a quality gate.

The gate matters as much as the model. A 30-px-tall motion-blurred crop
produces an embedding that is confidently wrong, and matching on it manufactures
a FRIENDLY verdict out of noise. When the crop cannot support a claim, the
honest output is INSUFFICIENT_QUALITY -- which is a distinct answer from
UNIDENTIFIED, not a synonym for it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import cv2
import numpy as np

from ..config import IdentityCfg, SystemCfg, resolve
from ..contracts import Detection

# OSNet re-ID convention: 256x128 (h x w), ImageNet normalisation.
_INPUT_H, _INPUT_W = 256, 128
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@runtime_checkable
class Embedder(Protocol):
    """Anything that turns person crops into comparable vectors."""

    model_id: str
    dim: int
    available: bool

    def embed(self, crops: list[np.ndarray]) -> np.ndarray: ...


class QualityGate:
    """Decides whether a crop can support an identity claim at all."""

    def __init__(self, cfg: IdentityCfg) -> None:
        self.cfg = cfg

    def score(self, crop: np.ndarray, det: Detection) -> tuple[float, str | None]:
        """Return (quality 0-1, reason-for-rejection or None)."""
        h, w = crop.shape[:2]
        if h < self.cfg.min_crop_h or w < self.cfg.min_crop_w:
            return 0.0, f"crop too small ({w}x{h})"
        if det.confidence < self.cfg.min_det_conf:
            return 0.0, f"detection confidence {det.confidence:.2f} below gate"
        if det.edge_clipped:
            return 0.0, "box clipped by frame edge"

        grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        blur = float(cv2.Laplacian(grey, cv2.CV_64F).var())
        if blur < self.cfg.min_blur:
            return 0.0, f"crop too blurred (Laplacian var {blur:.1f})"

        # Blend the three signals into a single 0-1 usable quality figure,
        # recorded on the verdict so a match can be audited later.
        size_q = min(1.0, h / (self.cfg.min_crop_h * 2.0))
        blur_q = min(1.0, blur / (self.cfg.min_blur * 4.0))
        return round(float(0.5 * size_q + 0.3 * blur_q + 0.2 * det.confidence), 3), None


class OSNetEmbedder:
    """OSNet x0_25 appearance embedder."""

    def __init__(self, cfg: SystemCfg) -> None:
        self.cfg: IdentityCfg = cfg.identity
        self.dim = 512
        self.available = False
        self.model = None
        self.model_id = "osnet_x0_25:unavailable"
        self.unavailable_reason = ""
        self.gate = QualityGate(cfg.identity)

        path = resolve(cfg, self.cfg.weights)
        if not path.exists():
            path = resolve(cfg, str(Path("weights") / self.cfg.weights))
        if not path.exists():
            self.unavailable_reason = (
                f"OSNet weights not found ({self.cfg.weights}). "
                f"Run scripts/fetch_weights.py once with network access."
            )
            return

        try:
            import torch
            from .osnet import load_osnet
            self.device = "cuda" if (cfg.runtime.device != "cpu"
                                     and torch.cuda.is_available()) else "cpu"
            self.model = load_osnet(path, device=self.device)
            self.torch = torch
            self.available = True
            self.model_id = f"osnet_x0_25@{path.stem}"
        except Exception as exc:                      # noqa: BLE001 - report, don't crash
            self.unavailable_reason = f"OSNet failed to load: {exc}"

    @staticmethod
    def preprocess(crops: list[np.ndarray]) -> np.ndarray:
        batch = np.empty((len(crops), _INPUT_H, _INPUT_W, 3), dtype=np.float32)
        for i, c in enumerate(crops):
            if c.ndim == 2:
                c = cv2.cvtColor(c, cv2.COLOR_GRAY2BGR)
            r = cv2.resize(c, (_INPUT_W, _INPUT_H), interpolation=cv2.INTER_LINEAR)
            batch[i] = cv2.cvtColor(r, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        batch = (batch - _MEAN) / _STD
        return batch.transpose(0, 3, 1, 2)            # NHWC -> NCHW

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        """Return L2-normalised embeddings, shape (N, 512)."""
        if not self.available or not crops:
            return np.zeros((0, self.dim), dtype=np.float32)

        x = self.torch.from_numpy(self.preprocess(crops)).to(self.device)
        with self.torch.no_grad():
            # BatchNorm1d in the fc head needs batch > 1 in train mode; the
            # model is in eval mode so running stats are used and a batch of
            # one is fine.
            f = self.model(x)
        f = self.torch.nn.functional.normalize(f, dim=1)
        return f.cpu().numpy().astype(np.float32)

    def describe(self) -> dict:
        return {"model_id": self.model_id, "available": self.available,
                "dim": self.dim, "reason": self.unavailable_reason}


def build_embedder(cfg: SystemCfg) -> OSNetEmbedder:
    return OSNetEmbedder(cfg)
