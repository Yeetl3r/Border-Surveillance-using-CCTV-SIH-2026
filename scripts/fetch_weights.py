"""Fetch model weights once, so the system can then run air-gapped (F15).

    python scripts/fetch_weights.py              # detector + re-ID (small)
    python scripts/fetch_weights.py --pose       # + YOLO26-Pose for F16
    python scripts/fetch_weights.py --vlm        # + LocateAnything-3B (~7.8 GB)
    python scripts/fetch_weights.py --all

This is the ONLY component permitted to reach the network, and it is a build
step, not a runtime one. At runtime ``runtime.offline`` is true and the model
loaders refuse to download -- a system that quietly fetches weights on first
use is not air-gapped, whatever the README says.

Everything lands in ``weights/``, which is gitignored. Nothing is downloaded
implicitly by any other code path.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "weights"

# OSNet: the author's own MIT-licensed release. Filename on the hub is long and
# encodes the training recipe; we store it under a stable short name.
OSNET_REPO = "kaiyangzhou/osnet"
OSNET_FILE = ("osnet_x0_25_msmt17_combineall_256x128_amsgrad_ep150_stp60"
              "_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth")
OSNET_LOCAL = "osnet_x0_25_msmt17.pth"


def fetch_yolo(names: list[str]) -> None:
    from ultralytics.utils.downloads import attempt_download_asset
    for n in names:
        dst = WEIGHTS / n
        if dst.exists():
            print(f"  = {n} already present ({dst.stat().st_size/1e6:.1f} MB)")
            continue
        print(f"  > {n} ...")
        cwd = os.getcwd()
        try:
            os.chdir(WEIGHTS)
            attempt_download_asset(n)
        finally:
            os.chdir(cwd)
        if dst.exists():
            print(f"    {dst.stat().st_size/1e6:.1f} MB")


def fetch_osnet() -> None:
    dst = WEIGHTS / OSNET_LOCAL
    if dst.exists():
        print(f"  = {OSNET_LOCAL} already present ({dst.stat().st_size/1e6:.1f} MB)")
        return
    import urllib.request as u
    url = f"https://huggingface.co/{OSNET_REPO}/resolve/main/{OSNET_FILE}"
    print(f"  > {OSNET_LOCAL} (MIT, author's release, MSMT17) ...")
    rq = u.Request(url, headers={"User-Agent": "sentinel-fetch/1.0"})
    with u.urlopen(rq, timeout=180) as r, dst.open("wb") as f:
        f.write(r.read())
    print(f"    {dst.stat().st_size/1e6:.1f} MB")


def fetch_vlm() -> None:
    dst = WEIGHTS / "LocateAnything-3B"
    if dst.exists() and any(dst.glob("*.safetensors")):
        print("  = LocateAnything-3B already present")
        return
    print("  > nvidia/LocateAnything-3B (~7.8 GB) ...")
    print("    LICENCE: NVIDIA Open Model License -- non-commercial use, for")
    print("    academic and non-profit research only. Its base LM")
    print("    (Qwen2.5-3B-Instruct) carries the Qwen Research licence.")
    print("    This is the same restriction that rules out the InsightFace")
    print("    packs elsewhere in this project. Fielding F11 commercially")
    print("    requires replacing it -- Qwen3-VL (Apache-2.0) is the swap.")
    for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ.pop(var, None)
    from huggingface_hub import snapshot_download
    snapshot_download("nvidia/LocateAnything-3B", local_dir=str(dst),
                      max_workers=8, allow_patterns=None)
    print("    done")


def main() -> int:
    ap = argparse.ArgumentParser("fetch_weights")
    ap.add_argument("--pose", action="store_true", help="YOLO26-Pose for F16")
    ap.add_argument("--vlm", action="store_true", help="LocateAnything-3B for F11")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    WEIGHTS.mkdir(parents=True, exist_ok=True)
    print(f"weights -> {WEIGHTS}\n")

    print("detector (F1/F2):")
    fetch_yolo(["yolo26s.pt"])

    print("\nappearance re-ID (F8):")
    try:
        fetch_osnet()
    except Exception as exc:                          # noqa: BLE001
        print(f"  ! failed: {exc}")
        print("    F8 will report itself unavailable rather than guessing.")

    if args.pose or args.all:
        print("\npose (F16):")
        fetch_yolo(["yolo26s-pose.pt"])

    if args.vlm or args.all:
        print("\ngrounding VLM (F11):")
        try:
            fetch_vlm()
        except Exception as exc:                      # noqa: BLE001
            print(f"  ! failed: {exc}")
            print("    F11 will report itself unavailable; the SQL route is unaffected.")

    print("\nDone. The system can now run with runtime.offline = true.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
