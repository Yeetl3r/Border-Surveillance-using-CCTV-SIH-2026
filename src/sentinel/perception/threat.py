"""Specialised detectors -- weapons on person crops, drones on the full frame.

These are separate weights behind the same idea as the perimeter detector, not
extra classes bolted onto it. Two reasons, and they are different reasons.

*Weapons* are separate because of **scale**. A rifle at 80 m occupies a handful
of pixels in a 1920-wide frame; no amount of confidence tuning recovers an
object the backbone never resolved. The person carrying it, however, is already
detected and already tracked, and their box is a crop at which the weapon is
legible. So the weapon model runs on person crops -- which is also cheaper than
running a second model over every full frame.

*Drones* are separate because of **class**. COCO has no drone. There is no
parent object to crop to either, so this one runs full-frame, at a deliberately
large `imgsz` because the whole difficulty is that the target is small.

Bird rejection
--------------
A drone detector that fires on every bird is worthless above a border, and the
usual fix -- train a `bird` class alongside `drone` -- is worse than it looks
here: the mixed drone/bird sets available carry a few dozen bird boxes, which
would produce a weak bird detector competing against a strong one. The stock
COCO model already carries bird as class 14, trained on far more bird imagery
than any drone set contains.

So the discrimination is made *between* models. The drone model proposes; the
perimeter model, which already ran on this frame, is consulted for an
overlapping bird; and the more confident of the two wins.

A withheld candidate is deliberately *not* emitted as a zero-confidence
observation. Temporal confirmation counts hits, not confidence, so a flock of
rejected birds would confirm into exactly the alert the rejection existed to
prevent. Withheld candidates go to a display-only list instead, which is what
lets the panel say *why* nothing fired rather than simply not firing.

Neither detector raises an alert directly. Both emit observations that must
survive temporal confirmation like anything else -- a single frame of a rifle
is a flicker, not an event.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np

from ..config import AerialCfg, SystemCfg, WeaponCfg, resolve
from ..contracts import Detection, Observation, ObservationKind, ObjectGroup, Track


def _iou(a: tuple[float, float, float, float],
         b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return float(inter / ua) if ua > 0 else 0.0


def resolve_bird(box: tuple[float, float, float, float], conf: float,
                 birds: list[Detection], cfg: AerialCfg,
                 ) -> tuple[bool, Detection | None, float]:
    """Should this drone candidate be withheld as a bird?

    Returns ``(withhold, rival, iou)``. Separated from the detector class so
    the decision can be tested without loading weights -- it is the part of
    this module most likely to be wrong, and the part least likely to be
    exercised by a demo where no birds happen to fly past.
    """
    rival, iou = max(((d, _iou(box, d.bbox_px)) for d in birds),
                     key=lambda p: p[1], default=(None, 0.0))
    withhold = (rival is not None
                and iou >= cfg.bird_iou
                and rival.confidence >= conf * cfg.bird_margin)
    return withhold, rival, iou


def _load(cfg: SystemCfg, model: str):
    """Resolve weights, preferring an explicit path, then ``weights/``."""
    w = resolve(cfg, model)
    if not w.exists():
        w = resolve(cfg, str(Path("weights") / model))
    if not w.exists():
        raise FileNotFoundError(
            f"specialised weights not found: {model}. "
            "Train them with tools/train_detector.py, or disable this detector.")
    from ultralytics import YOLO
    return YOLO(str(w)), Path(model).stem


def _cuda() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except ImportError:
        return False


class WeaponDetector:
    """Weapon detection on person crops."""

    def __init__(self, cfg: SystemCfg) -> None:
        self.cfg: WeaponCfg = cfg.threat.weapons
        self.model, self.model_id = _load(cfg, self.cfg.model)
        self.device = 0 if _cuda() else "cpu"
        self.class_names: dict[int, str] = dict(self.model.names)

    def scan(self, tracks: list[Track], frame: np.ndarray, camera_id: str,
             frame_idx: int, at: datetime) -> list[Observation]:
        out: list[Observation] = []
        h, w = frame.shape[:2]

        people = [t for t in tracks if t.group is ObjectGroup.PERSON]
        # Clear last frame's label from EVERY person, not just the ones about
        # to be examined. `meta` persists for the life of a track, so a single
        # false detection would otherwise mark someone as armed permanently --
        # and the label would survive both the weapon leaving frame and the
        # threshold being raised, which is how a demo ends up showing a rifle
        # on an unarmed person for two minutes.
        for t in people:
            t.meta.pop("weapon", None)

        # Largest first: a 40 px crop cannot support a weapon claim, and the
        # per-frame crop budget is finite.
        people.sort(key=lambda t: -t.latest.height_px)

        for t in people[: self.cfg.max_crops_per_frame]:
            x1, y1, x2, y2 = (float(v) for v in t.latest.bbox_px)
            pad = self.cfg.crop_pad * max(1.0, x2 - x1)
            cx1, cy1 = int(max(0, x1 - pad)), int(max(0, y1 - pad))
            cx2, cy2 = int(min(w, x2 + pad)), int(min(h, y2 + pad))
            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size == 0 or crop.shape[0] < self.cfg.min_crop_h:
                continue

            r = self.model.predict(crop, imgsz=self.cfg.imgsz, conf=self.cfg.conf,
                                   device=self.device, verbose=False)[0]
            if r.boxes is None or len(r.boxes) == 0:
                continue

            confs = r.boxes.conf.cpu().numpy()
            clses = r.boxes.cls.cpu().numpy().astype(int)
            xyxy = r.boxes.xyxy.cpu().numpy()
            i = int(np.argmax(confs))
            name = self.class_names.get(int(clses[i]), str(clses[i]))

            # Back to frame coordinates: an evidence box in crop space is
            # useless to anything downstream that draws on the full frame.
            bx1, by1, bx2, by2 = (float(v) for v in xyxy[i])
            box = (cx1 + bx1, cy1 + by1, cx1 + bx2, cy1 + by2)

            t.meta["weapon"] = f"{name} ({confs[i]:.2f})"
            out.append(Observation(
                kind=ObservationKind.WEAPON_VISIBLE,
                camera_id=camera_id, at=at, frame_idx=frame_idx,
                confidence=round(float(confs[i]), 3),
                track_ids=(t.track_id,),
                geometry={"bbox_px": [round(v, 1) for v in box],
                          "carrier_bbox_px": [round(v, 1) for v in t.latest.bbox_px]},
                detail={"weapon_class": name, "model_id": self.model_id,
                        "carrier_track_id": t.track_id,
                        "reason": f"{name} visible on person #{t.track_id} "
                                  f"at {confs[i]:.2f}"},
            ))
        return out


class AerialDetector:
    """Drone detection on the full frame, with bird rejection."""

    def __init__(self, cfg: SystemCfg) -> None:
        self.cfg: AerialCfg = cfg.threat.aerial
        self.model, self.model_id = _load(cfg, self.cfg.model)
        self.device = 0 if _cuda() else "cpu"
        self.class_names: dict[int, str] = dict(self.model.names)
        #: Candidates rejected as birds on the most recent scanned frame.
        #: Display only -- never fed to temporal confirmation.
        self.withheld: list[dict] = []
        #: Accepted drone boxes from the most recent scanned frame. Drones are
        #: not tracked -- they have no ground footpoint and the tracker's
        #: motion model is built for objects on a ground plane -- so the
        #: overlay needs them from here rather than from the track list.
        self.detections: list[dict] = []

    def scan(self, dets: list[Detection], frame: np.ndarray, camera_id: str,
             frame_idx: int, at: datetime) -> list[Observation]:
        """``dets`` are this frame's perimeter detections -- the bird source."""
        if frame_idx % max(1, self.cfg.every_n_frames):
            # Deliberately leaves `detections` and `withheld` intact. They feed
            # the overlay, and clearing them on a skipped frame would make
            # every drone box blink at half the frame rate.
            return []
        self.withheld, self.detections = [], []

        r = self.model.predict(frame, imgsz=self.cfg.imgsz, conf=self.cfg.conf,
                               device=self.device, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []

        birds = [d for d in dets if d.class_id == self.cfg.bird_class_id]
        frame_area = float(frame.shape[0] * frame.shape[1])
        confs = r.boxes.conf.cpu().numpy()
        clses = r.boxes.cls.cpu().numpy().astype(int)
        xyxy = r.boxes.xyxy.cpu().numpy()

        out: list[Observation] = []
        for i in range(len(confs)):
            box = tuple(float(v) for v in xyxy[i])
            conf = float(confs[i])
            name = self.class_names.get(int(clses[i]), "drone")

            # Size sanity, before the bird check: a box covering a third of the
            # frame is the model calling the whole scene a drone, which is what
            # it actually does on dark wide shots. Withheld like a bird rather
            # than dropped, so the restraint is visible.
            frac = ((box[2] - box[0]) * (box[3] - box[1])) / max(1.0, frame_area)
            if frac > self.cfg.max_area_frac:
                self.withheld.append({
                    "bbox_px": [round(v, 1) for v in box],
                    "drone_conf": round(conf, 3), "area_frac": round(frac, 3),
                    "reason": f"drone candidate at {conf:.2f} withheld -- covers "
                              f"{frac * 100:.0f}% of the frame, above the "
                              f"{self.cfg.max_area_frac * 100:.0f}% ceiling for a "
                              f"perimeter contact",
                })
                continue

            # Consult the perimeter model. A confident overlapping bird beats a
            # drone claim -- and the loss is recorded, not silently discarded.
            withhold, bird, iou = resolve_bird(box, conf, birds, self.cfg)
            if withhold:
                # A withheld candidate is NOT emitted as a zero-confidence
                # observation. Temporal confirmation counts hits, not
                # confidence, so a stream of withheld birds would confirm into
                # exactly the alert the rejection was meant to prevent. It is
                # recorded here instead, and surfaced for display only.
                self.withheld.append({
                    "bbox_px": [round(v, 1) for v in box],
                    "drone_conf": round(conf, 3),
                    "bird_conf": round(bird.confidence, 3),
                    "iou": round(iou, 3),
                    "reason": f"drone candidate at {conf:.2f} withheld -- "
                              f"perimeter model calls it a bird at "
                              f"{bird.confidence:.2f} (IoU {iou:.2f})",
                })
                continue

            self.detections.append({"bbox_px": [round(v, 1) for v in box],
                                    "conf": round(conf, 3), "class": name})
            out.append(Observation(
                kind=ObservationKind.AERIAL_INTRUSION,
                camera_id=camera_id, at=at, frame_idx=frame_idx,
                confidence=round(conf, 3),
                geometry={"bbox_px": [round(v, 1) for v in box]},
                detail={"aerial_class": name, "model_id": self.model_id,
                        "reason": f"{name} detected at {conf:.2f}"
                                  + (f", nearest bird IoU {iou:.2f}" if bird else "")},
            ))
        return out


class ThreatDetectors:
    """Both specialised detectors, each optional, behind one call.

    Holding them together keeps the runner from growing two more nullable
    attributes and two more ``if is not None`` branches per frame.
    """

    def __init__(self, cfg: SystemCfg) -> None:
        self.weapons = WeaponDetector(cfg) if cfg.threat.weapons.enabled else None
        self.aerial = AerialDetector(cfg) if cfg.threat.aerial.enabled else None

    @property
    def enabled(self) -> bool:
        return self.weapons is not None or self.aerial is not None

    @property
    def withheld(self) -> list[dict]:
        """Drone candidates rejected as birds -- for display, never for alerts."""
        return self.aerial.withheld if self.aerial is not None else []

    @property
    def aerial_boxes(self) -> list[dict]:
        """Accepted drone boxes from the last scan, for the overlay."""
        return self.aerial.detections if self.aerial is not None else []

    @property
    def model_ids(self) -> str:
        ids = [d.model_id for d in (self.weapons, self.aerial) if d is not None]
        return "+".join(ids)

    def scan(self, tracks: list[Track], dets: list[Detection], frame: np.ndarray,
             camera_id: str, frame_idx: int, at: datetime,
             suppress: frozenset[str] = frozenset()) -> list[Observation]:
        """Run the enabled detectors, minus any this camera suppresses.

        Suppression is per camera rather than global because it answers a
        question about the scene, not about the system: a lens filled by a hand
        is not footage any weapon claim should be made about, and running the
        detector there produces a confident answer to a question nobody asked.
        """
        obs: list[Observation] = []
        if self.weapons is not None and "weapons" not in suppress:
            obs += self.weapons.scan(tracks, frame, camera_id, frame_idx, at)
        if self.aerial is not None and "aerial" not in suppress:
            obs += self.aerial.scan(dets, frame, camera_id, frame_idx, at)
        elif self.aerial is not None:
            # Clear the display lists, or a suppressed panel keeps drawing the
            # boxes the last unsuppressed camera produced.
            self.aerial.withheld, self.aerial.detections = [], []
        return obs
