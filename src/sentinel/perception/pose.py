"""F16 -- infiltration stance from pose keypoints.

Disabled by default. Read this before enabling it.

**Why it is off.** Stance is an inherently high-false-alarm signal. Crouching
versus standing is a continuum at CCTV resolution, and a spurious CRAWLING
alert is exactly the alert that teaches an operator to stop trusting the panel
-- the documented number-one failure mode in this domain. Enable it knowing
that, not incidentally.

**Two corrections over the naive implementation.**

1. *Pose runs on person crops, not on the full frame.* The obvious approach --
   run a pose model over the whole frame and then match its skeletons back to
   the detector's boxes -- requires an association step, and the natural
   shortcut (nearest by x-centre) silently mis-assigns whenever two people
   share a column: one near, one far, or one behind another. Cropping makes the
   association exact by construction: one crop, one subject, no matching.

2. *Edge-clipped boxes are suppressed.* A bounding box truncated by the frame
   border has an aspect ratio that reflects the crop, not the body. On this
   project's own demo footage a subject walking toward the camera goes from
   aspect 0.39 to 0.84 purely because the box hits the bottom edge -- an
   aspect-ratio rule would call that crouching, then crawling, and latch a
   critical alert on a person who is simply walking.

Stance never becomes an alert directly. It emits a ``STANCE_ANOMALY``
observation which must then survive temporal confirmation like anything else.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import PoseCfg, SystemCfg, resolve
from ..contracts import ObjectGroup, Stance, Track

# COCO-17 keypoint indices
NOSE = 0
SHOULDER_L, SHOULDER_R = 5, 6
HIP_L, HIP_R = 11, 12
KNEE_L, KNEE_R = 13, 14
ANKLE_L, ANKLE_R = 15, 16


class PoseEstimator:
    """Keypoint-based stance classification on person crops."""

    def __init__(self, cfg: SystemCfg) -> None:
        from ultralytics import YOLO

        p: PoseCfg = cfg.pose
        self.cfg = p
        weights = resolve(cfg, p.model)
        if not weights.exists():
            weights = resolve(cfg, str(Path("weights") / p.model))
        if not weights.exists():
            raise FileNotFoundError(
                f"pose weights not found: {p.model}. Run scripts/fetch_weights.py."
            )
        self.model = YOLO(str(weights))
        self.model_id = f"{Path(p.model).stem}"
        self.device = 0 if _cuda() else "cpu"

    def human_evidence(self, crop: np.ndarray) -> tuple[float, int, np.ndarray | None]:
        """How strongly does this crop contain a HUMAN skeleton?

        Returns ``(score, visible_keypoints, keypoints)``. The score is the mean
        confidence of the visible COCO-17 joints, which a pose model trained on
        people does not produce for a dog, a cow or a goat: it may fire weakly
        on an animal's head and legs, but it will not resolve a coherent
        shoulder-hip-knee chain.

        This is the discriminator for crawling-human-versus-quadruped. Box
        aspect ratio cannot do it -- a person on hands and knees and a dog are
        the same shape -- and the class label alone cannot either, because a
        detector trained on upright people frequently calls a crawling person
        an animal and occasionally calls a large dog a person.
        """
        if crop is None or crop.size == 0 or crop.shape[0] < 32:
            return 0.0, 0, None
        r = self.model.predict(crop, imgsz=self.cfg.imgsz, conf=0.05,
                               device=self.device, verbose=False)[0]
        if r.keypoints is None or len(r.keypoints.data) == 0 or r.boxes is None:
            return 0.0, 0, None

        # The score is the POSE MODEL'S OWN person confidence, not an aggregate
        # invented on top of the keypoints. That model is trained to detect
        # people and nothing else, so its confidence already encodes "is this a
        # human body" far better than any hand-rolled keypoint statistic --
        # an earlier version scored the mean joint confidence weighted by how
        # much of the torso chain resolved, and it did not separate: real people
        # scored a median of 0.36, straddling any threshold worth setting.
        confs = r.boxes.conf.cpu().numpy()
        i = int(np.argmax(confs))
        score = float(confs[i])

        kd = r.keypoints.data.cpu().numpy()
        k = kd[i] if i < len(kd) else kd[0]
        n = int((k[:, 2] >= self.cfg.keypoint_conf).sum())
        return round(score, 3), n, k

    def disambiguate(self, tracks: list[Track], frame: np.ndarray) -> None:
        """Separate a crawling human from a quadruped, on evidence.

        Applied to animal detections and to low-slung person detections -- the
        two directions the confusion runs. The verdict and the evidence behind
        it are written onto the track so an alert can explain itself rather
        than simply asserting a class.
        """
        if not self.cfg.enabled:
            return

        h, w = frame.shape[:2]
        candidates = [
            t for t in tracks
            if t.group in (ObjectGroup.ANIMAL, ObjectGroup.PERSON)
            and t.latest.aspect_ratio >= self.cfg.quadruped_aspect
            and not (self.cfg.suppress_edge_clipped and t.latest.edge_clipped)
        ]
        candidates.sort(key=lambda t: -(t.latest.height_px))

        # `meta` persists for the life of a track, so last frame's verdict has
        # to be cleared before this frame's is written. Without this, a subject
        # that looked low-slung for a single frame keeps `quadruped_note`
        # forever -- and since annotate() now skips anything carrying that key,
        # the track would never be given a stance again. A track that stops
        # being a candidate at all (it stood up, so its aspect ratio dropped)
        # is cleared here too, which is why every candidate is cleared rather
        # than only the ones about to be examined.
        for t in tracks:
            t.meta.pop("quadruped_note", None)
            t.meta.pop("reclassified", None)

        for t in candidates[: self.cfg.max_crops_per_frame]:
            x1, y1, x2, y2 = (int(max(0, v)) for v in t.latest.bbox_px)
            crop = frame[y1:min(h, y2), x1:min(w, x2)]
            score, n, _k = self.human_evidence(crop)
            t.meta["human_skeleton"] = score
            t.meta["keypoints_visible"] = n

            if score >= self.cfg.human_skeleton_min and n >= self.cfg.min_keypoints:
                if t.group is ObjectGroup.ANIMAL:
                    # A coherent human skeleton outranks the class label. This
                    # is the crawling-infiltrator case the detector gets wrong.
                    t.meta["reclassified"] = (
                        f"detected as {t.latest.class_name} but resolves a human "
                        f"skeleton ({n} joints, score {score:.2f}) -- treated as a person")
                    t.group = ObjectGroup.PERSON
            elif t.group is ObjectGroup.PERSON:
                # Low-slung, and no human skeleton resolves. Do not claim a
                # crawling infiltrator on that evidence.
                t.meta["quadruped_note"] = (
                    f"low-slung person detection with no resolvable human skeleton "
                    f"({n} joints, score {score:.2f}) -- stance withheld")
                t.stance = Stance.UNKNOWN
                t.stance_confidence = 0.0

    def annotate(self, tracks: list[Track], frame: np.ndarray) -> None:
        """Set ``stance`` and ``stance_confidence`` on person tracks, in place."""
        if not self.cfg.enabled:
            return

        # Skip anything disambiguate() already adjudicated this frame. It runs
        # first and may have WITHHELD the stance claim on a low-slung subject
        # with no resolvable human skeleton; recomputing stance here would
        # overwrite that decision with a confident STANDING, so the track would
        # carry both the note explaining the refusal and the claim it refused
        # to make.
        people = [t for t in tracks
                  if t.group == ObjectGroup.PERSON and "quadruped_note" not in t.meta]
        # Prefer larger subjects: a 40 px crop cannot support a stance claim,
        # and the per-frame crop budget is finite.
        people.sort(key=lambda t: -(t.latest.height_px))
        people = people[: self.cfg.max_crops_per_frame]

        h, w = frame.shape[:2]
        for t in people:
            det = t.latest

            if self.cfg.suppress_edge_clipped and det.edge_clipped:
                t.stance, t.stance_confidence = Stance.UNKNOWN, 0.0
                t.meta["stance_note"] = "suppressed: box clipped by frame edge"
                continue

            x1, y1, x2, y2 = (int(v) for v in det.bbox_px)
            pad = int(0.08 * max(1, x2 - x1))
            x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
            x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[0] < 48:
                t.stance, t.stance_confidence = Stance.UNKNOWN, 0.0
                continue

            r = self.model.predict(crop, imgsz=self.cfg.imgsz, conf=self.cfg.conf,
                                   device=self.device, verbose=False)[0]
            if r.keypoints is None or len(r.keypoints.data) == 0:
                t.stance, t.stance_confidence = Stance.UNKNOWN, 0.0
                continue

            # One crop, one subject: take the largest skeleton if the pose model
            # found more than one (a bystander clipped into the crop).
            kd = r.keypoints.data.cpu().numpy()
            k = kd[0] if len(kd) == 1 else kd[int(np.argmax([_spread(x) for x in kd]))]
            t.stance, t.stance_confidence = self.classify(k, crop.shape)

    def classify(self, kpts: np.ndarray, crop_shape) -> tuple[Stance, float]:
        """Classify stance from one skeleton.

        Uses the torso axis rather than the bounding box. The vertical extent
        of shoulders-to-hips relative to the shoulder width is a body-intrinsic
        ratio -- it does not change when the box is cropped, which is precisely
        the failure the box-aspect approach has.
        """
        c = self.cfg
        vis = kpts[:, 2] >= c.keypoint_conf
        if int(vis.sum()) < c.min_keypoints:
            return Stance.UNKNOWN, 0.0

        def pt(i: int) -> np.ndarray | None:
            return kpts[i, :2] if vis[i] else None

        sh = _midpoint(pt(SHOULDER_L), pt(SHOULDER_R))
        hp = _midpoint(pt(HIP_L), pt(HIP_R))
        kn = _midpoint(pt(KNEE_L), pt(KNEE_R))
        ank = _midpoint(pt(ANKLE_L), pt(ANKLE_R))

        if sh is None or hp is None:
            return Stance.UNKNOWN, 0.0

        torso = hp - sh
        torso_len = float(np.hypot(*torso))
        if torso_len < 1e-3:
            return Stance.UNKNOWN, 0.0

        # 0 = torso horizontal (prone), 1 = torso vertical (upright).
        verticality = abs(float(torso[1])) / torso_len

        shoulder_w = (float(np.hypot(*(pt(SHOULDER_L) - pt(SHOULDER_R))))
                      if vis[SHOULDER_L] and vis[SHOULDER_R] else 0.0)

        # A prone body is horizontal: the torso axis lies flat AND the torso is
        # not much longer vertically than the shoulders are wide.
        if verticality < c.prone_ratio:
            conf = float(np.clip((c.prone_ratio - verticality) / max(c.prone_ratio, 1e-6), 0, 1))
            return Stance.PRONE, round(0.5 + 0.5 * conf, 3)

        # Crouching: torso still upright, but the legs are folded -- hip-to-knee
        # vertical drop is small relative to torso length.
        if kn is not None:
            thigh_drop = abs(float(kn[1] - hp[1]))
            ratio = thigh_drop / torso_len
            if ratio < 0.55:
                return Stance.CROUCHING, round(float(np.clip(0.5 + (0.55 - ratio), 0, 1)), 3)
            if ank is not None:
                full = abs(float(ank[1] - sh[1]))
                if full > 0 and (torso_len / full) > c.crouch_ratio:
                    return Stance.CROUCHING, 0.6

        conf = float(np.clip(verticality, 0, 1))
        _ = shoulder_w  # retained for future width-based refinement
        return Stance.STANDING, round(conf, 3)


def _midpoint(a: np.ndarray | None, b: np.ndarray | None) -> np.ndarray | None:
    if a is not None and b is not None:
        return (a + b) / 2.0
    return a if a is not None else b


def _spread(k: np.ndarray) -> float:
    v = k[k[:, 2] > 0.2][:, :2]
    if len(v) < 2:
        return 0.0
    return float((v[:, 0].max() - v[:, 0].min()) * (v[:, 1].max() - v[:, 1].min()))


def _cuda() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except ImportError:
        return False
