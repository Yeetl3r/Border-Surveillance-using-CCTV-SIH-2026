"""Detector interface.

Detection sits behind a Protocol so that swapping YOLO26 for RF-DETR touches
this package and nothing downstream. The abstraction boundary matters more than
which detector is chosen first -- the licensing question (Ultralytics is
AGPL-3.0, RF-DETR is Apache-2.0) may force the swap later, and it must be a
one-file change when it comes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from ..contracts import Detection, ObjectGroup


@dataclass(slots=True)
class DetBatch:
    """Detections for one frame in the array form the trackers expect.

    Ultralytics' BYTETracker and BOTSORT consume a "results-like" object with
    ``.xywh``, ``.conf`` and ``.cls``, plus boolean-mask indexing. Providing
    that here -- rather than handing the tracker a live ultralytics Results
    object -- is what lets detection and tracking be genuinely separate stages
    with our own types crossing between them.
    """

    xywh: np.ndarray      # (N, 4) centre-x, centre-y, width, height
    conf: np.ndarray      # (N,)
    cls: np.ndarray       # (N,)
    model_id: str = ""

    def __len__(self) -> int:
        return int(self.xywh.shape[0])

    def __getitem__(self, mask) -> "DetBatch":
        return DetBatch(self.xywh[mask], self.conf[mask], self.cls[mask], self.model_id)

    @classmethod
    def empty(cls, model_id: str = "") -> "DetBatch":
        return cls(np.zeros((0, 4), np.float32), np.zeros((0,), np.float32),
                   np.zeros((0,), np.float32), model_id)

    @classmethod
    def from_xyxy(cls, xyxy: np.ndarray, conf: np.ndarray, cls_ids: np.ndarray,
                  model_id: str = "") -> "DetBatch":
        if len(xyxy) == 0:
            return cls.empty(model_id)
        xyxy = np.asarray(xyxy, dtype=np.float32)
        w = xyxy[:, 2] - xyxy[:, 0]
        h = xyxy[:, 3] - xyxy[:, 1]
        xywh = np.stack([xyxy[:, 0] + w / 2, xyxy[:, 1] + h / 2, w, h], axis=1)
        return cls(xywh, np.asarray(conf, np.float32),
                   np.asarray(cls_ids, np.float32), model_id)

    @property
    def xyxy(self) -> np.ndarray:
        if len(self) == 0:
            return np.zeros((0, 4), np.float32)
        cx, cy, w, h = self.xywh[:, 0], self.xywh[:, 1], self.xywh[:, 2], self.xywh[:, 3]
        return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)


@runtime_checkable
class Detector(Protocol):
    """Anything that turns a frame into detections."""

    model_id: str
    #: Class id -> name. DATA, not an enum -- Phase 2 adds drone classes by
    #: config, and a hardcoded four-class assumption would have to be unpicked.
    class_names: dict[int, str]

    def detect(self, frame: np.ndarray, conf: float | None = None,
               ) -> tuple[DetBatch, list[Detection]]: ...
    def warmup(self) -> None: ...


class ClassMapper:
    """Maps fine-grained detector classes into the four operational groups.

    COCO has 80 classes; F1 asks for person / vehicle / animal / bag. The
    mapping is config so that adding a class is a YAML edit.
    """

    def __init__(self, groups: dict[str, list[int]]) -> None:
        self._by_id: dict[int, ObjectGroup] = {}
        for name, ids in groups.items():
            try:
                grp = ObjectGroup(name)
            except ValueError:
                grp = ObjectGroup.OTHER
            for i in ids:
                self._by_id[int(i)] = grp

    def group(self, class_id: int) -> ObjectGroup:
        return self._by_id.get(int(class_id), ObjectGroup.OTHER)

    @property
    def kept_ids(self) -> list[int]:
        """Class ids worth running through the pipeline at all."""
        return sorted(self._by_id.keys())
