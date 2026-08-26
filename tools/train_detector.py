"""Fine-tune a specialised detector — weapons or aerial objects.

    python tools/train_detector.py --task weapons --epochs 60
    python tools/train_detector.py --task drones  --epochs 60

These run on ROI crops or full frames as their own detector, behind the same
`Detector` protocol as the perimeter model. They are separate weights, not
extra classes on the main detector: a weapon is looked for on a person crop
that has already been detected and tracked, which is the only scale at which a
rifle at range is more than a handful of pixels.

Class design
------------
The public weapon set carries 29 labels with heavy duplication (`Guns`/`guns`,
`Pistol`/`pistol`/`pistols`, `Rifle`/`rifle`), several non-object labels
(`violence`, `Stabbing`, `Blood`), and some that are simply junk (`al`). Those
labels cannot reliably separate a pistol from a rifle -- a box tagged `Guns`
could be either -- so they are collapsed to **firearm** and **knife**, which
the annotations do support. Splitting further would be inventing precision the
data does not have, and the split would fail on the first challenging example.

Non-object and ambiguous labels are dropped rather than folded in.
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

# Index -> name, in the order the dataset card lists them.
WEAPON_NAMES = [
    "weapons", "Aggressor", "Blood", "Guns", "Guns perspective", "Hand",
    "Heavy Gun", "Knife", "Knife_Deploy", "Knife_Weapon", "Long guns",
    "Person", "Pistol", "Rifle", "Shotgun", "Stabbing", "Victim", "al",
    "guns", "handgun", "heavyweapon", "larga", "person", "pistol",
    "pistols", "rifle", "shotgun", "violence", "weapon",
]

# Only labels that name a concrete object with an unambiguous type.
WEAPON_MAP = {
    3: 0, 4: 0, 6: 0, 10: 0, 12: 0, 13: 0, 14: 0,      # Guns, Heavy Gun, Long guns,
    18: 0, 19: 0, 20: 0, 21: 0, 23: 0, 24: 0, 25: 0,   # Pistol, Rifle, Shotgun,
    26: 0,                                              # handgun, larga, ...
    7: 1, 8: 1, 9: 1,                                   # Knife, Knife_Deploy, Knife_Weapon
}
WEAPON_CLASSES = ["firearm", "knife"]
DRONE_CLASSES = ["drone", "bird"]


def _write_yaml(root: Path, classes: list[str]) -> Path:
    y = root / "data.yaml"
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(classes))
    y.write_text(
        f"path: {root.as_posix()}\ntrain: images/train\nval: images/val\n"
        f"names:\n{names}\n", encoding="utf-8")
    return y


# ── weapons: parquet -> YOLO ──────────────────────────────────────────────
def prepare_weapons(src: Path, out: Path) -> Path:
    import pyarrow.parquet as pq

    if (out / "data.yaml").exists():
        print(f"  = already prepared at {out}")
        return out / "data.yaml"

    splits = {"train": "train-*.parquet", "val": "validation-*.parquet"}
    kept = dropped = 0
    for split, pattern in splits.items():
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        files = sorted(src.glob(f"data/{pattern}"))
        if not files:
            raise SystemExit(f"no parquet matching {pattern} under {src}")

        n = 0
        for f in files:
            table = pq.read_table(f)
            for row in table.to_pylist():
                w, h = float(row["width"]), float(row["height"])
                if w <= 0 or h <= 0:
                    continue
                lines = []
                for box, cat in zip(row["objects"]["bbox"], row["objects"]["category"]):
                    cls = WEAPON_MAP.get(int(cat))
                    if cls is None:
                        dropped += 1
                        continue
                    # COCO xywh (top-left) -> YOLO normalised centre form
                    x, y, bw, bh = (float(v) for v in box)
                    cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
                    nw, nh = bw / w, bh / h
                    if not (0 < nw <= 1 and 0 < nh <= 1) or not (0 <= cx <= 1 and 0 <= cy <= 1):
                        continue
                    lines.append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                    kept += 1
                if not lines:
                    continue                       # images with no kept object
                stem = f"{split}_{n:06d}"
                (out / "images" / split / f"{stem}.jpg").write_bytes(row["image"]["bytes"])
                (out / "labels" / split / f"{stem}.txt").write_text(
                    "\n".join(lines), encoding="utf-8")
                n += 1
        print(f"  {split}: {n} images")
    print(f"  boxes kept {kept}, dropped {dropped} (ambiguous or non-object labels)")
    return _write_yaml(out, WEAPON_CLASSES)


# ── drones: zipped YOLO archives -> merged YOLO tree ──────────────────────
def prepare_drones(src: Path, out: Path) -> Path:
    if (out / "data.yaml").exists():
        print(f"  = already prepared at {out}")
        return out / "data.yaml"

    zips = sorted(src.glob("*.zip"))
    if not zips:
        raise SystemExit(f"no zip archives under {src} -- is the download finished?")

    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    n = 0
    for z in zips:
        # Splits that contain birds are the ones that teach the discriminator
        # what a bird is; a drone-only archive teaches only "small thing in sky".
        with zipfile.ZipFile(z) as zf:
            members = zf.namelist()
            imgs = [m for m in members if m.lower().endswith((".jpg", ".jpeg", ".png"))]
            for m in imgs:
                lbl = str(Path(m).with_suffix(".txt"))
                if lbl not in members:
                    continue
                split = "val" if (n % 8 == 0) else "train"
                stem = f"{z.stem}_{n:06d}"
                (out / "images" / split / f"{stem}.jpg").write_bytes(zf.read(m))
                (out / "labels" / split / f"{stem}.txt").write_bytes(zf.read(lbl))
                n += 1
    print(f"  {n} labelled images across {len(zips)} archives")
    if n == 0:
        raise SystemExit("no image/label pairs found -- check the archive layout")
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

    src = ROOT / "datasets" / args.task
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
