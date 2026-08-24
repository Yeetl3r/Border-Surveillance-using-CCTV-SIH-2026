"""Fit the F8 similarity threshold to measured data instead of intuition.

    python tools/measure_threshold.py --source assets/sample_border.mp4

Why this tool exists. OSNet's fc head ends in ReLU, so every embedding lies in
the positive orthant and cosine similarities between *unrelated* crops sit far
above zero -- typically 0.5-0.9, not ~0.0. Anyone picking a threshold from
intuition about "cosine similarity" will choose something near 0.5 and match
everybody. The configured default must be justified by a measured separation,
and this prints it.

It reports two distributions:

* **positive** -- the same tracked subject at different frames. A real number
  from our own footage.
* **negative proxy** -- person crops against non-person crops from the same
  scene. This is a WEAK proxy and is labelled as such: the distribution that
  actually matters is person-vs-*different*-person, and measuring it needs
  footage containing more than one subject. Where the demo clip has only one,
  that number is not available and is not invented.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sentinel.config import load_config, resolve            # noqa: E402
from sentinel.contracts import ObjectGroup                  # noqa: E402
from sentinel.identity import build_embedder                # noqa: E402
from sentinel.perception import ObjectTracker, YoloDetector # noqa: E402


def pct(a: np.ndarray, q: float) -> float:
    return float(np.percentile(a, q)) if len(a) else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser("measure_threshold")
    ap.add_argument("--source", default="assets/sample_border.mp4")
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--max-frames", type=int, default=400)
    args = ap.parse_args()

    from datetime import datetime, timezone

    cfg = load_config(args.root)
    det = YoloDetector(cfg)
    emb = build_embedder(cfg)
    if not emb.available:
        print(f"! OSNet unavailable: {emb.unavailable_reason}")
        return 1
    trk = ObjectTracker(cfg.tracker, "measure", det.mapper, det.class_names)

    cap = cv2.VideoCapture(str(resolve(cfg, args.source)))
    if not cap.isOpened():
        print("! cannot open source")
        return 1

    by_track: dict[int, list[np.ndarray]] = {}
    non_person: list[np.ndarray] = []
    n = 0
    while n < args.max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        batch, dets = det.detect(frame)
        tracks = trk.update(batch, dets, frame, n, datetime.now(timezone.utc))
        h, w = frame.shape[:2]
        for t in tracks:
            x1, y1, x2, y2 = (int(max(0, v)) for v in t.latest.bbox_px)
            crop = frame[y1:min(h, y2), x1:min(w, x2)]
            if crop.size == 0 or crop.shape[0] < 64:
                continue
            if t.group == ObjectGroup.PERSON:
                q, reject = emb.gate.score(crop, t.latest)
                if reject is None:
                    by_track.setdefault(t.track_id, []).append(crop)
            elif len(non_person) < 60:
                non_person.append(crop)
    cap.release()

    print(f"frames scanned: {n}")
    print(f"person tracks with gate-passing crops: "
          f"{ {k: len(v) for k, v in by_track.items()} }")
    print(f"non-person crops collected: {len(non_person)}\n")

    pos: list[float] = []
    embs_by_track: dict[int, np.ndarray] = {}
    for tid, crops in by_track.items():
        if len(crops) < 2:
            continue
        E = emb.embed(crops[:40])
        embs_by_track[tid] = E
        S = E @ E.T
        iu = np.triu_indices(len(E), k=1)
        pos.extend(S[iu].tolist())
    pos_a = np.asarray(pos)

    neg: list[float] = []
    if non_person and embs_by_track:
        N = emb.embed(non_person)
        for E in embs_by_track.values():
            neg.extend((E @ N.T).ravel().tolist())
    neg_a = np.asarray(neg)

    print("POSITIVE -- same subject, different frames (real):")
    if len(pos_a):
        print(f"  n={len(pos_a)}  min={pos_a.min():.4f}  p5={pct(pos_a,5):.4f}  "
              f"median={pct(pos_a,50):.4f}  max={pos_a.max():.4f}")
    else:
        print("  none -- no track had two gate-passing crops")

    print("\nNEGATIVE PROXY -- person vs non-person crops (WEAK proxy):")
    if len(neg_a):
        print(f"  n={len(neg_a)}  min={neg_a.min():.4f}  median={pct(neg_a,50):.4f}  "
              f"p95={pct(neg_a,95):.4f}  max={neg_a.max():.4f}")
    else:
        print("  none")

    if len(pos_a) and len(neg_a):
        lo, hi = pct(pos_a, 5), pct(neg_a, 95)
        print(f"\nseparation: positive p5 = {lo:.4f}   negative-proxy p95 = {hi:.4f}")
        if lo > hi:
            suggested = round((lo + hi) / 2, 3)
            print(f"  classes separate. A threshold near {suggested} sits between them.")
        else:
            suggested = round(lo, 3)
            print(f"  DISTRIBUTIONS OVERLAP. No threshold separates them cleanly on "
                  f"this data. Setting it at the positive p5 ({suggested}) accepts "
                  f"95% of true matches and will admit some false ones.")
        print(f"\n  config identity.sim_threshold is currently "
              f"{cfg.identity.sim_threshold}")
        print(f"  measured suggestion: {suggested}")

    print("""
CAVEAT -- read before quoting any of this.
  The negative distribution that matters operationally is person versus a
  DIFFERENT person. Measuring it requires footage with more than one subject.
  Where the source clip has only one, that number does not exist here and has
  not been estimated. Enrol a second subject and re-run before treating the
  threshold as validated.""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
