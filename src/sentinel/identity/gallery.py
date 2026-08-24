"""F8 -- open-set identity against an enrolled friendly gallery.

This is the component where it is easiest to accidentally build something
indefensible, so the design constraint is stated plainly: **the system enrols
only friendly personnel and never classifies anyone as hostile.**

A closed-set friend/foe classifier is not implementable here. No legitimate
open dataset of adversary appearance exists at CCTV resolution, and such a
classifier answers "what about a civilian in dark clothing?" badly and
confidently. So the only claim made is membership of an enrolled set:

    MATCH      -> FRIENDLY      (with the similarity that produced it)
    NO MATCH   -> UNIDENTIFIED  (not "hostile" -- just not enrolled)
    BAD CROP   -> INSUFFICIENT_QUALITY

Threshold note: OSNet's fc head ends in ReLU, so embeddings live in the
positive orthant and cosine similarities between *unrelated* people sit far
above zero -- typically 0.5-0.9 rather than ~0.0. A threshold picked by
intuition from "cosine similarity" will therefore match everyone. Use
``tools/measure_threshold.py`` to fit it against a measured distribution on
real footage, and record the measured separation in the README.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from ..config import IdentityCfg, SystemCfg, resolve
from ..contracts import (
    Detection,
    IdentityLabel,
    IdentityVerdict,
    ObjectGroup,
    Track,
)
from .embedder import OSNetEmbedder


class FriendlyGallery:
    """Enrolled friendly personnel, searched by cosine similarity.

    Small galleries (tens of identities, a handful of images each) are searched
    exhaustively with a matrix multiply. That is faster than any index at this
    scale, adds no dependency, and works air-gapped. Milvus or ChromaDB become
    worthwhile at thousands of identities, not at Phase 1 scale.
    """

    def __init__(self, cfg: SystemCfg, embedder: OSNetEmbedder) -> None:
        self.cfg: IdentityCfg = cfg.identity
        self.embedder = embedder
        self.dir = resolve(cfg, self.cfg.gallery_dir)
        self.vectors = np.zeros((0, embedder.dim), dtype=np.float32)
        self.labels: list[str] = []
        self.enrolled: dict[str, int] = {}
        self.loaded_at: datetime | None = None
        self.load()

    # -- enrolment -----------------------------------------------------------
    def load(self) -> None:
        """Load the gallery from ``data/gallery/<person_name>/*.jpg``.

        Several images per identity across pose and lighting is not optional.
        Single-image enrolment false-matches at a rate that would be
        operationally useless, which is the documented risk for this feature.
        """
        self.vectors = np.zeros((0, self.embedder.dim), dtype=np.float32)
        self.labels, self.enrolled = [], {}
        if not self.dir.exists() or not self.embedder.available:
            return

        cache = self.dir / "_gallery.npz"
        if cache.exists():
            try:
                z = np.load(cache, allow_pickle=True)
                self.vectors = z["vectors"].astype(np.float32)
                self.labels = list(z["labels"])
                self.enrolled = {n: self.labels.count(n) for n in set(self.labels)}
                self.loaded_at = datetime.now(timezone.utc)
                return
            except (OSError, KeyError, ValueError):
                pass                                   # rebuild from images

        vecs, labs = [], []
        for person_dir in sorted(p for p in self.dir.iterdir() if p.is_dir()):
            imgs = [p for p in sorted(person_dir.iterdir())
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
            crops = [cv2.imread(str(p)) for p in imgs]
            crops = [c for c in crops if c is not None]
            if not crops:
                continue
            emb = self.embedder.embed(crops)
            for e in emb:
                vecs.append(e)
                labs.append(person_dir.name)

        if vecs:
            self.vectors = np.stack(vecs).astype(np.float32)
            self.labels = labs
            self.enrolled = {n: labs.count(n) for n in set(labs)}
            np.savez(cache, vectors=self.vectors, labels=np.array(labs, dtype=object))
        self.loaded_at = datetime.now(timezone.utc)

    @property
    def size(self) -> int:
        return int(self.vectors.shape[0])

    @property
    def active(self) -> bool:
        return self.embedder.available and self.size > 0

    # -- matching ------------------------------------------------------------
    def match(self, embedding: np.ndarray) -> tuple[str | None, float]:
        """Best enrolled identity and its cosine similarity."""
        if self.size == 0:
            return None, 0.0
        sims = self.vectors @ embedding.astype(np.float32)
        i = int(np.argmax(sims))
        return self.labels[i], float(sims[i])

    def verdict(self, embedding: np.ndarray, quality: float) -> IdentityVerdict:
        name, sim = self.match(embedding)
        thr = self.cfg.sim_threshold
        if name is not None and sim >= thr:
            label = IdentityLabel.FRIENDLY
        else:
            label = IdentityLabel.UNIDENTIFIED
        return IdentityVerdict(
            label=label,
            matched_enrollee=name if label is IdentityLabel.FRIENDLY else None,
            similarity=round(sim, 4),
            threshold_used=thr,
            quality_score=quality,
            model_id=self.embedder.model_id,
            decided_at=datetime.now(timezone.utc),
        )

    def describe(self) -> dict:
        return {
            "active": self.active,
            "identities": len(self.enrolled),
            "vectors": self.size,
            "enrolled": self.enrolled,
            "threshold": self.cfg.sim_threshold,
            "embedder": self.embedder.describe(),
        }


class IdentityResolver:
    """Assigns identity verdicts to person tracks, at a controlled rate."""

    def __init__(self, cfg: SystemCfg, embedder: OSNetEmbedder,
                 gallery: FriendlyGallery) -> None:
        self.cfg = cfg
        self.icfg: IdentityCfg = cfg.identity
        self.embedder = embedder
        self.gallery = gallery
        self._last_frame: dict[int, int] = {}

    def resolve(self, tracks: list[Track], frame: np.ndarray, frame_idx: int) -> None:
        """Annotate person tracks with an IdentityVerdict, in place."""
        if not self.icfg.enabled:
            return

        # No gallery means no basis for any identity claim. Say so once, on the
        # track, rather than silently marking everyone UNIDENTIFIED -- which
        # would look like a working system reporting a true negative.
        if not self.gallery.active:
            reason = (self.embedder.unavailable_reason
                      or "no friendly personnel enrolled")
            for t in tracks:
                if t.group == ObjectGroup.PERSON and t.identity is None:
                    t.identity = IdentityVerdict(
                        label=IdentityLabel.INSUFFICIENT_QUALITY,
                        threshold_used=self.icfg.sim_threshold,
                        model_id=self.embedder.model_id)
                    t.meta["identity_note"] = reason
            return

        def interval_for(t: Track) -> int:
            # A track that has never produced a usable crop retries on the next
            # frame rather than waiting a full interval. Degraded footage makes
            # detector confidence fluctuate frame to frame (0.27 to 0.92 on this
            # project's own demo clip), so the difference between "sample every
            # 4th frame" and "sample until one passes" is the difference between
            # a track that never gets identified and one that does.
            settled = t.identity is not None and \
                t.identity.label is not IdentityLabel.INSUFFICIENT_QUALITY
            return self.icfg.embed_every_n_frames if settled else 1

        due = [t for t in tracks
               if t.group == ObjectGroup.PERSON
               and frame_idx - self._last_frame.get(t.track_id, -10 ** 9)
               >= interval_for(t)]
        due.sort(key=lambda t: -(t.latest.height_px))
        due = due[: self.icfg.max_crops_per_frame]
        if not due:
            return

        h, w = frame.shape[:2]
        usable: list[Track] = []
        crops: list[np.ndarray] = []
        qualities: list[float] = []

        for t in due:
            det: Detection = t.latest
            x1, y1, x2, y2 = (int(max(0, v)) for v in det.bbox_px)
            x2, y2 = min(w, x2), min(h, y2)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            q, reject = self.gate_check(crop, det)
            self._last_frame[t.track_id] = frame_idx
            if reject is not None:
                # Do NOT overwrite a verdict already reached from a better crop.
                # Identity accrues over a track's life; one blurred frame is not
                # evidence that a previously confident match was wrong, and
                # letting it downgrade the verdict would make identity flicker.
                settled = t.identity is not None and \
                    t.identity.label is not IdentityLabel.INSUFFICIENT_QUALITY
                if not settled:
                    t.identity = IdentityVerdict(
                        label=IdentityLabel.INSUFFICIENT_QUALITY,
                        threshold_used=self.icfg.sim_threshold,
                        quality_score=q, model_id=self.embedder.model_id)
                    t.meta["identity_note"] = reject
                continue
            usable.append(t)
            crops.append(crop)
            qualities.append(q)

        if not crops:
            return
        for t, e, q in zip(usable, self.embedder.embed(crops), qualities):
            t.identity = self.gallery.verdict(e, q)
            t.meta.pop("identity_note", None)

        # Forget rate-limit state for tracks that have gone.
        live = {t.track_id for t in tracks}
        for tid in [k for k in self._last_frame if k not in live]:
            del self._last_frame[tid]

    def gate_check(self, crop: np.ndarray, det: Detection):
        return self.embedder.gate.score(crop, det)
