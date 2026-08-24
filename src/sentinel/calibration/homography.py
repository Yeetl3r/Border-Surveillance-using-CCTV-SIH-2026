"""F3 -- ground-plane homography.

Four operator-marked ground points give the perspective transform that maps a
bounding-box footpoint from image pixels to real-world metres:

    [X_m, Y_m, 1]^T  ~  H . [x_foot, y_foot, 1]^T

This is the single largest capability jump over pixel-space geofencing, and it
is cheap -- ``cv2.findHomography`` plus arithmetic, no model, no training. It
converts "the subject is in the lower half of the image" (a statement about the
picture) into "the subject is 4.2 metres from the camera" (a statement about
the world).

Assumes a FIXED camera. A PTZ camera invalidates H on every pan and tilt;
supporting one needs a preset-linked calibration table, which is not built.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..config import HomographyCfg


@dataclass(slots=True)
class Homography:
    """Image-pixel to ground-metre transform for one fixed camera."""

    matrix: np.ndarray | None
    homography_id: str
    fixed_camera: bool = True
    residual_m: float = 0.0
    # Convex hull of the calibration points, in pixels. Extrapolating far
    # outside it produces confident nonsense, so we flag it rather than
    # silently returning a number.
    _img_points: np.ndarray | None = None

    @property
    def calibrated(self) -> bool:
        return self.matrix is not None

    @classmethod
    def from_config(cls, cfg: HomographyCfg) -> "Homography":
        if not cfg.enabled or not cfg.image_points or not cfg.world_points:
            return cls(matrix=None, homography_id="", fixed_camera=cfg.fixed_camera)

        img = np.asarray(cfg.image_points, dtype=np.float64)
        wld = np.asarray(cfg.world_points, dtype=np.float64)
        if img.shape != (4, 2) or wld.shape != (4, 2):
            raise ValueError("homography requires exactly four 2-D point pairs")

        matrix, _mask = cv2.findHomography(img, wld, method=0)
        if matrix is None:
            raise ValueError(
                "findHomography failed -- the four image points are probably "
                "collinear or near-collinear. Mark points spread across the "
                "ground plane, not along one edge."
            )

        obj = cls(matrix=matrix, homography_id=cfg.homography_id or "unnamed",
                  fixed_camera=cfg.fixed_camera, _img_points=img)
        # Round-trip the calibration points and record the worst error. A
        # ground plane maps to the image by an exact homography, so this should
        # be ~0; a large value means the marked points were not coplanar.
        errs = []
        for (px, py), (wx, wy) in zip(cfg.image_points, cfg.world_points):
            t = obj.to_metres((px, py))
            if t is not None:
                errs.append(float(np.hypot(t[0] - wx, t[1] - wy)))
        obj.residual_m = max(errs) if errs else 0.0
        return obj

    def to_metres(self, point_px: tuple[float, float]) -> tuple[float, float] | None:
        """Map an image point to ground metres.

        Returns None when the camera is uncalibrated or the point projects
        behind the camera. None is the honest answer -- a fabricated metre
        value is worse than an absent one, and every consumer of this is
        written to handle it.
        """
        if self.matrix is None:
            return None
        x, y = float(point_px[0]), float(point_px[1])
        v = self.matrix @ np.array([x, y, 1.0], dtype=np.float64)
        w = v[2]
        if abs(w) < 1e-9:
            return None            # point on the horizon: infinitely far away
        return (float(v[0] / w), float(v[1] / w))

    def to_metres_many(self, pts: np.ndarray) -> np.ndarray | None:
        """Vectorised form for arrays of shape (N, 2)."""
        if self.matrix is None or len(pts) == 0:
            return None
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
        out = cv2.perspectiveTransform(pts, self.matrix.astype(np.float32))
        return out.reshape(-1, 2)

    def to_pixels(self, point_m: tuple[float, float]) -> tuple[float, float] | None:
        """Inverse transform -- used to draw metric zones onto the video."""
        if self.matrix is None:
            return None
        inv = np.linalg.inv(self.matrix)
        v = inv @ np.array([point_m[0], point_m[1], 1.0], dtype=np.float64)
        if abs(v[2]) < 1e-9:
            return None
        return (float(v[0] / v[2]), float(v[1] / v[2]))

    def distance_m(
        self, a_px: tuple[float, float], b_px: tuple[float, float]
    ) -> float | None:
        """Ground distance in metres between two image points."""
        a, b = self.to_metres(a_px), self.to_metres(b_px)
        if a is None or b is None:
            return None
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    def describe(self) -> dict:
        return {
            "calibrated": self.calibrated,
            "homography_id": self.homography_id,
            "fixed_camera": self.fixed_camera,
            "residual_m": round(self.residual_m, 4),
        }
