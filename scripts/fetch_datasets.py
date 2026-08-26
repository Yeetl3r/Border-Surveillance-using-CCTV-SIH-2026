"""Fetch the training sets for the specialised detectors.

    python scripts/fetch_datasets.py --weapons
    python scripts/fetch_datasets.py --drones      # ~10 GB
    python scripts/fetch_datasets.py --all

LICENCE NOTE, and it matters.
----------------------------
The two sets named in the implementation plan are both **CC BY-NC-4.0**, not
CC BY-4.0 as the plan states:

    jsalazar/US-Real-time-gun-detection-in-CCTV      CC BY-NC-4.0
    Simuletic/CCTV_Weapon_Detection_...             CC BY-NC-4.0

NonCommercial blocks the deployment this system is written for, so neither is
used here. The sets below are CC BY-4.0 and carry no such restriction.

Verify every dataset licence before it enters a pipeline. At least one
commonly-cited Indian plate set is CC BY-NC-ND, whose NoDerivatives clause
prohibits even research fine-tuning.
"""

from __future__ import annotations

import argparse
import os
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent

SETS = {
    "weapons": ("Subh775/WeaponDetection", "cc-by-4.0",
                "5,871 / 1,491 / 2,295 images. Derived from a Roboflow Universe "
                "weapon-detection set; 29 noisy labels collapsed to firearm+knife "
                "at prepare time."),
    "drones":  ("mazqtpopx/cranfield-synthetic-drone-detection", "cc-by-4.0",
                "Cranfield synthetic drone set. Includes drones_with_birds splits, "
                "which are the ones that teach drone-versus-bird rather than "
                "merely 'small object in sky'."),
}


def fetch(key: str) -> None:
    repo, lic, note = SETS[key]
    out = ROOT / "datasets" / key
    print(f"\n{key}: {repo}")
    print(f"  licence: {lic}")
    print(f"  {note}")
    if out.exists() and any(out.rglob("*")):
        print(f"  = already present at {out}")
        return
    for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ.pop(var, None)
    from huggingface_hub import snapshot_download
    snapshot_download(repo, repo_type="dataset", local_dir=str(out), max_workers=8)
    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"  -> {out}  {total / 1e9:.2f} GB")


def main() -> int:
    ap = argparse.ArgumentParser("fetch_datasets")
    ap.add_argument("--weapons", action="store_true")
    ap.add_argument("--drones", action="store_true")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    picked = [k for k in SETS if getattr(a, k) or a.all]
    if not picked:
        ap.print_help()
        return 1
    for k in picked:
        fetch(k)
    print("\nNext: python tools/train_detector.py --task <weapons|drones>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
