"""F8 -- enrol friendly personnel into the gallery.

    python tools/enroll.py --name patrol_01 --source assets/sample_border.mp4 \
                           --frames 1-40 --max 12

Extracts person crops that pass the quality gate and writes them to
``data/gallery/<name>/``. The gallery is rebuilt on next start.

Several images per identity across pose and lighting is not optional.
Single-image enrolment false-matches at a rate that makes the feature
operationally useless -- it is the documented risk for this capability, and
the mitigation is enrolment breadth plus a threshold fitted to measured data
(see tools/measure_threshold.py), not a threshold chosen by feel.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from sentinel.config import load_config, resolve            # noqa: E402
from sentinel.contracts import ObjectGroup                  # noqa: E402
from sentinel.identity import build_embedder                # noqa: E402
from sentinel.perception import YoloDetector                # noqa: E402


def parse_range(spec: str) -> tuple[int, int]:
    if "-" in spec:
        a, b = spec.split("-", 1)
        return int(a), int(b)
    return int(spec), int(spec)


def main() -> int:
    ap = argparse.ArgumentParser("enroll")
    ap.add_argument("--name", required=True, help="identity label, e.g. patrol_01")
    ap.add_argument("--source", required=True, help="video file or image directory")
    ap.add_argument("--frames", default="1-999999", help="frame range, e.g. 1-40")
    ap.add_argument("--max", type=int, default=12, help="max crops to keep")
    ap.add_argument("--every", type=int, default=3, help="sample every Nth frame")
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--replace", action="store_true", help="clear existing crops first")
    args = ap.parse_args()

    cfg = load_config(args.root)
    out_dir = resolve(cfg, cfg.identity.gallery_dir) / args.name
    if args.replace and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Invalidate the cached gallery so the new crops are picked up.
    cache = resolve(cfg, cfg.identity.gallery_dir) / "_gallery.npz"
    cache.unlink(missing_ok=True)

    detector = YoloDetector(cfg)
    embedder = build_embedder(cfg)
    if not embedder.available:
        print(f"! OSNet unavailable: {embedder.unavailable_reason}")
        print("  Crops will still be written; embeddings are built at startup.")

    lo, hi = parse_range(args.frames)
    src = resolve(cfg, args.source)
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        print(f"! cannot open {src}")
        return 1

    kept, idx, rejected = 0, 0, {}
    while kept < args.max:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx < lo or idx > hi or idx % args.every:
            continue

        _batch, dets = detector.detect(frame)
        people = sorted([d for d in dets if d.group == ObjectGroup.PERSON],
                        key=lambda d: -d.height_px)
        if not people:
            continue
        det = people[0]

        x1, y1, x2, y2 = (int(max(0, v)) for v in det.bbox_px)
        h, w = frame.shape[:2]
        crop = frame[y1:min(h, y2), x1:min(w, x2)]
        if crop.size == 0:
            continue

        q, reject = embedder.gate.score(crop, det)
        if reject is not None:
            rejected[reject] = rejected.get(reject, 0) + 1
            continue

        path = out_dir / f"{args.name}_{idx:05d}.jpg"
        cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        kept += 1
        print(f"  + frame {idx:5d}  {crop.shape[1]}x{crop.shape[0]}  "
              f"det={det.confidence:.2f}  quality={q:.3f}")

    cap.release()
    print(f"\nenrolled {kept} crops for '{args.name}' -> {out_dir}")
    if rejected:
        print("rejected by the quality gate:")
        for reason, n in sorted(rejected.items(), key=lambda kv: -kv[1]):
            print(f"  {n:4d}  {reason}")
    if kept < 4:
        print("\n! Fewer than 4 crops. Open-set matching from a thin gallery "
              "false-matches badly. Widen --frames or lower --every.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
