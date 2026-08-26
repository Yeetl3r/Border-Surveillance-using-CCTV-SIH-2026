"""Fine-tune a specialised detector — weapons or aerial objects.

    python tools/train_detector.py --task weapons --epochs 60
    python tools/train_detector.py --task drones  --epochs 60

These run on ROI crops or full frames as their own detector, behind the same
`Detector` protocol as the perimeter model. They are separate weights, not
extra classes on the main detector: a weapon is looked for on a person crop
that has already been detected and tracked, which is the only scale at which a
rifle at range is more than a handful of pixels.

Choosing the data
-----------------
The first weapon set tried here was a parquet export with 29 noisy labels. It
did not survive inspection, and the inspection is worth recording because the
failure was silent: training reached mAP@50 of 0.009 while the loss fell
normally.

    36% of boxes fell outside the image under the only plausible reading
    the knife class held 7 training instances against 276 in validation
    drawing the converted labels back onto their images put the box beside
        the weapon rather than on it

Those are broken annotations, not hard examples, and no amount of training
fixes them. The set in use now is a Roboflow COCO export: the coordinate
convention is fixed by specification rather than inferred, the class list is
already at the granularity this system wants, and every converted box is
range-checked here rather than trusted.

**Always draw converted labels back onto their images before training on
them.** It takes a minute and it is the only check that catches this class of
error.
"""

from __future__ import annotations

import argparse
import io
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Roboflow COCO export: unambiguous [x, y, w, h] top-left in absolute pixels,
# with the class list already at the operational granularity we want.
WEAPON_CLASSES = ["rifle", "pistol", "grenade", "knife"]
# One class. Bird discrimination is done against the stock COCO
# detector's class 14, not by a thin bird class trained here --
# see prepare_drones() for why.
DRONE_CLASSES = ["drone"]

def _write_yaml(root: Path, classes: list[str]) -> Path:
    y = root / "data.yaml"
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(classes))
    y.write_text(
        f"path: {root.as_posix()}\ntrain: images/train\nval: images/val\n"
        f"names:\n{names}\n", encoding="utf-8")
    return y


# ── weapons: Roboflow COCO zips -> YOLO ───────────────────────────────────
def prepare_weapons(src: Path, out: Path) -> Path:
    """Convert the Roboflow COCO export to YOLO layout.

    An earlier attempt used a parquet weapon set whose boxes did not survive
    inspection: 36% fell outside the image under the only plausible reading,
    the knife class had 7 training instances against 276 in validation, and
    drawing the converted labels back onto their images put boxes beside the
    weapon rather than on it. Training on that produced mAP@50 of 0.009 -- the
    honest reading of which is that the labels were wrong, not the model.

    A COCO export is unambiguous by specification, and every converted box is
    range-checked here rather than trusted.
    """
    import json

    if (out / "data.yaml").exists():
        print(f"  = already prepared at {out}")
        return out / "data.yaml"

    kept = skipped = 0
    for split, archive in (("train", "train.zip"), ("val", "valid.zip")):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(src / "data" / archive) as zf:
            names = zf.namelist()
            ann_name = next((m for m in names if m.endswith("_annotations.coco.json")), None)
            if ann_name is None:
                raise SystemExit(f"no COCO annotations inside {archive}")
            coco = json.loads(zf.read(ann_name))

            # COCO category ids are 1-based and may include a supercategory at 0.
            id_to_name = {c["id"]: c["name"].lower() for c in coco["categories"]}
            name_to_idx = {n: i for i, n in enumerate(WEAPON_CLASSES)}
            images = {im["id"]: im for im in coco["images"]}

            per_image: dict[int, list[str]] = {}
            for a in coco["annotations"]:
                im = images.get(a["image_id"])
                cls = name_to_idx.get(id_to_name.get(a["category_id"], ""))
                if im is None or cls is None:
                    skipped += 1
                    continue
                W, H = float(im["width"]), float(im["height"])
                x, y, bw, bh = (float(v) for v in a["bbox"])
                # Range-check in pixel space before normalising: a box that
                # leaves the image is a broken annotation, not a hard example.
                if bw <= 0 or bh <= 0 or x < -1 or y < -1 or x + bw > W + 1 or y + bh > H + 1:
                    skipped += 1
                    continue
                cx, cy = (x + bw / 2) / W, (y + bh / 2) / H
                per_image.setdefault(a["image_id"], []).append(
                    f"{cls} {cx:.6f} {cy:.6f} {bw / W:.6f} {bh / H:.6f}")
                kept += 1

            n = 0
            for image_id, lines in per_image.items():
                im = images[image_id]
                member = next((m for m in names if m.endswith(im["file_name"])), None)
                if member is None:
                    continue
                stem = f"{split}_{n:06d}"
                (out / "images" / split / f"{stem}.jpg").write_bytes(zf.read(member))
                (out / "labels" / split / f"{stem}.txt").write_text(
                    chr(10).join(lines), encoding="utf-8")
                n += 1
            print(f"  {split}: {n} images")

    print(f"  boxes kept {kept}, skipped {skipped} (unmapped class or out of frame)")
    return _write_yaml(out, WEAPON_CLASSES)


def _clean_yolo_label(raw: bytes) -> tuple[bytes, int]:
    """Keep only well-formed 5-field detection lines, in range.

    Real exports are not uniform: one file in 8,349 of the drone set carried an
    87-field segmentation polygon instead of a box. Ultralytics would parse that
    as a detection row and train on nonsense. Validate rather than trust --
    this is the same lesson the weapon set taught, applied before it costs a
    training run.
    """
    kept, dropped = [], 0
    for line in raw.decode("utf-8", "replace").splitlines():
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            dropped += 1
            continue
        try:
            cls = int(float(parts[0]))
            cx, cy, w, h = (float(v) for v in parts[1:])
        except ValueError:
            dropped += 1
            continue
        if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
            dropped += 1
            continue
        kept.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return (chr(10).join(kept)).encode("utf-8"), dropped


# ── drones: Seraphim YOLO archives -> YOLO tree ────────────────────────────
def prepare_drones(src: Path, out: Path) -> Path:
    """Convert the Seraphim drone export to YOLO layout.

    Why this set and not the synthetic one first tried here: that was 21 GB of
    simulator renders shipped as RGBA instance masks, with no boxes. Boxes are
    recoverable from masks, but the deeper problem is that a detector trained
    only on renders is asked at demo time to find a real drone in real footage,
    and nothing in its training distribution looks like that. Seraphim is real
    imagery curated from 23 open datasets, already boxed in YOLO format.

    **One class, deliberately.** The obvious second class is `bird` -- a drone
    detector that fires on every bird is worthless above a border. But the
    stock COCO detector already carries bird as class 14, trained on far more
    bird imagery than any drone set contains. Adding a thin bird class here
    (the mixed set available had 78 bird boxes) would produce a weaker bird
    detector competing with a stronger one. So the discrimination is made
    between models rather than inside one: this detector answers "is there a
    drone", the perimeter model answers "is there a bird", and disagreement is
    resolved on confidence. See `sentinel.perception.aerial`.
    """
    if (out / "data.yaml").exists():
        print(f"  = already prepared at {out}")
        return out / "data.yaml"

    img_zips = sorted(src.glob("**/images/*.zip"))
    lbl_zips = sorted(src.glob("**/labels/*.zip"))
    if not img_zips or not lbl_zips:
        raise SystemExit(
            f"expected images/*.zip and labels/*.zip under {src} -- "
            "is the download finished?")

    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Labels are small; hold them all in memory and index by stem so an image
    # with no label file is dropped rather than silently trained as background.
    labels: dict[str, bytes] = {}
    malformed = 0
    for z in lbl_zips:
        with zipfile.ZipFile(z) as zf:
            for m in zf.namelist():
                if m.endswith(".txt"):
                    clean, dropped = _clean_yolo_label(zf.read(m))
                    malformed += dropped
                    labels[Path(m).stem] = clean
    print(f"  {len(labels)} label files across {len(lbl_zips)} archive(s), "
          f"{malformed} malformed rows dropped")

    n = unlabelled = empty = 0
    for z in img_zips:
        with zipfile.ZipFile(z) as zf:
            for m in zf.namelist():
                if not m.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                lbl = labels.get(Path(m).stem)
                if lbl is None:
                    unlabelled += 1
                    continue
                if not lbl.strip():
                    empty += 1  # a real negative: keep it, it teaches restraint
                # Deterministic split: every 8th image to val, so a re-run
                # reproduces the same partition.
                split = "val" if (n % 8 == 0) else "train"
                stem = f"{n:06d}"
                (out / "images" / split / f"{stem}.jpg").write_bytes(zf.read(m))
                (out / "labels" / split / f"{stem}.txt").write_bytes(lbl)
                n += 1

    print(f"  {n} labelled images ({empty} intentional negatives), "
          f"{unlabelled} images dropped for having no label")
    if n == 0:
        raise SystemExit("no image/label pairs matched -- check the archive layout")
    return _write_yaml(out, DRONE_CLASSES)


def main() -> int:
    ap = argparse.ArgumentParser("train_detector")
    ap.add_argument("--task", choices=["weapons", "drones"], required=True)
    ap.add_argument("--model", default="yolo26n.pt",
                    help="base weights; the small model is the right size for a crop")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--prepare-only", action="store_true")
    args = ap.parse_args()

    src = ROOT / "datasets" / ("guns" if args.task == "weapons" else "drones_real")
    out = ROOT / "datasets" / f"{args.task}_yolo"
    if not src.exists():
        raise SystemExit(f"{src} not found -- run scripts/fetch_datasets.py first")

    print(f"preparing {args.task} …")
    data = prepare_weapons(src, out) if args.task == "weapons" else prepare_drones(src, out)
    print(f"  data.yaml -> {data}")
    if args.prepare_only:
        return 0

    from ultralytics import YOLO
    base = ROOT / "weights" / args.model
    model = YOLO(str(base) if base.exists() else args.model)

    print(f"\ntraining {args.model} on {args.task} for {args.epochs} epochs …")
    model.train(
        data=str(data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        project=str(ROOT / "runs"), name=args.task, exist_ok=True,
        device=0, patience=15, seed=0, verbose=True,
    )

    best = ROOT / "runs" / args.task / "weights" / "best.pt"
    if best.exists():
        dest = ROOT / "weights" / f"{args.task}_yolo26n.pt"
        shutil.copy2(best, dest)
        print(f"\nbest weights -> {dest}")
        metrics = model.val(data=str(data), device=0)
        print("\nMEASURED on the held-out split:")
        print(f"  mAP@50    {metrics.box.map50:.4f}")
        print(f"  mAP@50-95 {metrics.box.map:.4f}")
        for i, name in enumerate(metrics.names.values()):
            try:
                print(f"  {name:10} AP@50 {metrics.box.ap50[i]:.4f}")
            except (IndexError, TypeError):
                pass
        print("\nQuote these numbers, not the ones on the dataset card.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
