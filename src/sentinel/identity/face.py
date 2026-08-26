"""Face localisation on person crops — YuNet (OpenCV Zoo, Apache-2.0).

Used for one purpose: cutting a recognisable head-and-shoulders image out of a
person crop so an alert can show *who* was seen, not just a full-body box the
operator has to squint at.

Deliberately **detection only**. This does not recognise, match, or identify
anyone by face. Identity in this system is decided by open-set appearance
matching against an enrolled friendly gallery (`identity/gallery.py`), and that
remains the only component allowed to make a claim about who someone is.

Why YuNet and not InsightFace: the InsightFace pretrained packs (`buffalo_*`,
`antelopev2`, and the SCRFD detectors inside them) are licensed for
non-commercial research only. For a system whose deliverable is fielded by a
security agency, that is disqualifying. YuNet is Apache-2.0, ships in OpenCV's
own model zoo, runs on CPU in about a millisecond, and is entirely adequate for
locating a face inside an already-cropped person.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class FaceLocator:
    """Finds the largest face in a crop. Reports unavailability rather than guessing."""

    def __init__(self, weights: Path | str, score_threshold: float = 0.6,
                 nms_threshold: float = 0.3, min_side: int = 24) -> None:
        self.available = False
        self.reason = ""
        self.model_id = "yunet:unavailable"
        self.min_side = min_side
        self._det = None

        path = Path(weights)
        if not path.exists():
            self.reason = (f"YuNet weights not found ({path.name}). "
                           f"Run scripts/fetch_weights.py.")
            return
        if not hasattr(cv2, "FaceDetectorYN"):
            self.reason = f"opencv {cv2.__version__} has no FaceDetectorYN"
            return
        try:
            self._det = cv2.FaceDetectorYN.create(
                str(path), "", (320, 320), score_threshold, nms_threshold, 5000)
            self.available = True
            self.model_id = f"yunet@{path.stem}"
        except cv2.error as exc:
            self.reason = f"YuNet failed to load: {exc}"

    def locate(self, crop: np.ndarray) -> tuple[int, int, int, int] | None:
        """Return the largest face box (x, y, w, h) within ``crop``, or None."""
        if not self.available or crop is None or crop.size == 0:
            return None
        h, w = crop.shape[:2]
        if min(h, w) < self.min_side:
            return None

        # YuNet expects the input size to be declared before each detect call;
        # person crops vary in size frame to frame.
        try:
            self._det.setInputSize((w, h))
            _rc, faces = self._det.detect(crop)
        except cv2.error:
            return None
        if faces is None or len(faces) == 0:
            return None

        # Largest by area: in a person crop the subject dominates, and a second
        # face is a bystander clipped into the box.
        best = max(faces, key=lambda f: float(f[2]) * float(f[3]))
        x, y, fw, fh = (int(round(v)) for v in best[:4])
        return max(0, x), max(0, y), max(1, fw), max(1, fh)

    def cut(self, crop: np.ndarray, pad: float = 0.45) -> np.ndarray | None:
        """Return a padded head-and-shoulders cut, or None when no face is found.

        The padding is generous on purpose: a tight face box is hard to
        recognise on a small screen, and context around the head is what makes
        the image useful to a human.
        """
        box = self.locate(crop)
        if box is None:
            return None
        x, y, fw, fh = box
        h, w = crop.shape[:2]
        px, py = int(fw * pad), int(fh * pad)
        x0, y0 = max(0, x - px), max(0, y - py)
        x1, y1 = min(w, x + fw + px), min(h, y + fh + py)
        out = crop[y0:y1, x0:x1]
        return out if out.size else None

    def describe(self) -> dict:
        return {"model_id": self.model_id, "available": self.available,
                "reason": self.reason}
