"""YOLO26 detector adapter, with optional sliced inference.

YOLO26-s is the default: NMS-free end-to-end, DFL removed, and STAL label
assignment which guarantees positive coverage for small objects. Small-object
coverage is the defining difficulty here -- a person at 150-200 m on wide-angle
CCTV is 15-25 px tall.

Licence: Ultralytics models are AGPL-3.0. Fielding this in a product requires
open-sourcing the derivative work or an Ultralytics enterprise licence. The
permissive alternative is RF-DETR (Apache-2.0) behind the same Detector
Protocol. That is a real question for a government deployment and the answer
belongs in the README, not buried here.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ..config import DetectorCfg, SystemCfg, resolve
from ..contracts import Detection
from .detector import ClassMapper, DetBatch


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list[int]:
    """Greedy NMS used only to merge overlapping tiles in sliced inference."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / np.maximum(1e-9, areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou <= iou_thr]
    return keep


class YoloDetector:
    """Ultralytics YOLO behind the Detector Protocol."""

    def __init__(self, cfg: SystemCfg, overrides: dict | None = None) -> None:
        from ultralytics import YOLO           # imported late: heavy, and optional

        d: DetectorCfg = cfg.detector.model_copy(update=overrides or {})
        self.cfg = d
        self.sys = cfg

        weights = resolve(cfg, d.model)
        if not weights.exists():
            weights = resolve(cfg, str(Path("weights") / d.model))
        if not weights.exists():
            if cfg.runtime.offline:
                raise FileNotFoundError(
                    f"detector weights not found: {d.model}. Offline mode is on "
                    f"(F15), so nothing will be downloaded. Run "
                    f"`python scripts/fetch_weights.py` once with network access."
                )
            weights = Path(d.model)

        self.model = YOLO(str(weights))
        self.model_id = f"{Path(d.model).stem}@{_sha_short(weights)}"
        self.class_names: dict[int, str] = dict(self.model.names)
        self.mapper = ClassMapper(d.class_groups)
        self.device = _pick_device(cfg.runtime.device)
        self._kept = self.mapper.kept_ids

    # -- inference -----------------------------------------------------------
    def warmup(self) -> None:
        self.model.predict(np.zeros((640, 640, 3), np.uint8), imgsz=self.cfg.imgsz,
                           device=self.device, verbose=False)

    def _predict(self, frame: np.ndarray, conf: float | None = None,
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r = self.model.predict(
            frame, imgsz=self.cfg.imgsz, conf=conf if conf is not None else self.cfg.conf,
            iou=self.cfg.iou,
            device=self.device, classes=self._kept or None,
            max_det=self.cfg.max_det, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return (np.zeros((0, 4), np.float32), np.zeros((0,), np.float32),
                    np.zeros((0,), np.float32))
        return (r.boxes.xyxy.cpu().numpy().astype(np.float32),
                r.boxes.conf.cpu().numpy().astype(np.float32),
                r.boxes.cls.cpu().numpy().astype(np.float32))

    def _predict_tiled(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sliced inference for distant targets.

        Full-frame inference downsamples a 1080p frame to 960 px, at which
        point a 20 px target is 18 px and effectively gone. Tiling runs the
        detector at native scale over overlapping windows, then merges. It
        costs roughly one forward pass per tile, which is why it is opt-in.
        """
        s = self.cfg.sahi
        h, w = frame.shape[:2]
        step = int(s.tile * (1.0 - s.overlap))
        boxes, confs, clss = [], [], []

        for y0 in range(0, max(1, h - 1), step):
            for x0 in range(0, max(1, w - 1), step):
                x1, y1 = min(x0 + s.tile, w), min(y0 + s.tile, h)
                x0c, y0c = max(0, x1 - s.tile), max(0, y1 - s.tile)
                tile = frame[y0c:y1, x0c:x1]
                if tile.size == 0:
                    continue
                b, c, k = self._predict(tile)
                if len(b):
                    b[:, [0, 2]] += x0c
                    b[:, [1, 3]] += y0c
                    boxes.append(b); confs.append(c); clss.append(k)
                if x1 >= w:
                    break
            if y1 >= h:
                break

        # Also run the full frame: tiling is good at small targets and bad at
        # ones larger than a tile, so the union of both is what we want.
        b, c, k = self._predict(frame)
        if len(b):
            boxes.append(b); confs.append(c); clss.append(k)

        if not boxes:
            return (np.zeros((0, 4), np.float32), np.zeros((0,), np.float32),
                    np.zeros((0,), np.float32))

        B = np.concatenate(boxes); C = np.concatenate(confs); K = np.concatenate(clss)
        keep: list[int] = []
        for cid in np.unique(K):                      # class-wise merge
            idx = np.flatnonzero(K == cid)
            keep.extend(idx[i] for i in _nms(B[idx], C[idx], s.merge_iou))
        keep_arr = np.asarray(sorted(keep), dtype=int)
        return B[keep_arr], C[keep_arr], K[keep_arr]

    def detect(self, frame: np.ndarray, conf: float | None = None,
               ) -> tuple[DetBatch, list[Detection]]:
        """Detect on one frame, optionally at a camera-specific confidence.

        The per-camera override exists because sensitivity is a property of the
        scene, not of the system. A crawling subject in a wide shot is detected
        in 23% of frames at the global 0.25 and 41% at 0.10 -- and the same
        0.10 on a busy perimeter would flood the tracker with noise. One
        threshold cannot serve both, so the camera carries its own.
        """
        s = self.cfg.sahi
        use_tiles = s.enabled and max(frame.shape[:2]) >= s.min_frame_px
        xyxy, conf_a, cls = (self._predict_tiled(frame) if use_tiles
                             else self._predict(frame, conf))

        h, w = frame.shape[:2]
        m = 2  # a box within 2 px of a border is clipped by the frame, not the world
        dets: list[Detection] = []
        for (x1, y1, x2, y2), cf, ci in zip(xyxy, conf_a, cls):
            cid = int(ci)
            dets.append(Detection(
                bbox_px=(float(x1), float(y1), float(x2), float(y2)),
                class_id=cid,
                class_name=self.class_names.get(cid, str(cid)),
                group=self.mapper.group(cid),
                confidence=float(cf),
                model_id=self.model_id,
                tier="T1",
                edge_clipped=bool(x1 <= m or y1 <= m or x2 >= w - m or y2 >= h - m),
            ))
        return DetBatch.from_xyxy(xyxy, conf_a, cls, self.model_id), dets


def _pick_device(pref: str) -> str | int:
    if pref == "cpu":
        return "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            return 0
    except ImportError:
        pass
    return "cpu"


def _sha_short(path: Path) -> str:
    """Short content hash of the weights file.

    Recorded into every Detection's ``model_id`` and into the evidence record,
    so "which weights produced this alert?" has an answer that survives someone
    swapping a .pt file with the same name.
    """
    import hashlib
    try:
        h = hashlib.sha256()
        with path.open("rb") as fh:
            h.update(fh.read(1 << 20))         # first MB is plenty to distinguish
            h.update(str(path.stat().st_size).encode())
        return h.hexdigest()[:8]
    except OSError:
        return "unknown"
