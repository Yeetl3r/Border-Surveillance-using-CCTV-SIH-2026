"""One command from a fresh clone to a running demo.

    python scripts/bootstrap.py            # detector + re-ID, enrol, verify
    python scripts/bootstrap.py --all      # + pose (F16) + LocateAnything-3B (F11)

Three things are NOT in the repository and have to be recreated on each machine:

* ``weights/``      -- 20 MB to 7.8 GB of model files, gitignored
* ``data/gallery/`` -- the enrolled friendly gallery, derived from the committed clip
* ``data/db/``      -- the alert log, created on first run

This script produces all three, then runs the environment gate. It is the only
part of the project permitted to touch the network, and it is a build step: at
runtime ``runtime.offline`` is true and the model loaders refuse to download.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def step(n: int, total: int, title: str) -> None:
    print(f"\n[{n}/{total}] {title}\n{'-' * 66}", flush=True)


def run(cmd: list[str], why: str) -> bool:
    print(f"  $ {' '.join(cmd[1:]) if cmd[0] == sys.executable else ' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        print(f"  ! failed: {why}")
    return r.returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser("bootstrap")
    ap.add_argument("--all", action="store_true",
                    help="also fetch pose (F16) and LocateAnything-3B (F11, 7.8 GB)")
    ap.add_argument("--vlm", action="store_true", help="also fetch F11 only")
    ap.add_argument("--pose", action="store_true", help="also fetch F16 only")
    ap.add_argument("--skip-enroll", action="store_true")
    args = ap.parse_args()

    py = sys.executable
    total = 4
    print("=" * 66)
    print("SENTINEL bootstrap -- fresh clone to running demo")
    print("=" * 66)

    # -- 1. weights ---------------------------------------------------------
    step(1, total, "Model weights (the only networked step)")
    fetch = [py, "scripts/fetch_weights.py"]
    if args.all:
        fetch.append("--all")
    else:
        if args.pose:
            fetch.append("--pose")
        if args.vlm:
            fetch.append("--vlm")
    if not run(fetch, "could not fetch weights"):
        print("  Without detector weights nothing else will work. Stopping.")
        return 1

    # -- 2. friendly gallery ------------------------------------------------
    step(2, total, "Enrol the demo friendly (F8)")
    gallery = ROOT / "data" / "gallery" / "patrol_01"
    if args.skip_enroll:
        print("  skipped (--skip-enroll)")
    elif gallery.exists() and any(gallery.glob("*.jpg")):
        print(f"  = already enrolled: {len(list(gallery.glob('*.jpg')))} crops")
    else:
        print("  The gallery is derived from the committed clip, so it is")
        print("  regenerated rather than stored in git.")
        ok = run([py, "tools/enroll.py", "--name", "patrol_01",
                  "--source", "assets/sample_border.mp4",
                  "--frames", "1-70", "--max", "10", "--every", "4", "--replace"],
                 "enrolment failed")
        if not ok:
            print("  F8 will report itself unavailable rather than guessing.")

    # -- 3. tests -----------------------------------------------------------
    step(3, total, "Test suite (no GPU required)")
    run([py, "-m", "pytest", "-q"], "tests failed")

    # -- 4. environment gate ------------------------------------------------
    step(4, total, "Environment gate")
    gate_ok = run([py, "scripts/verify_env.py"], "environment gate reported failures")

    print("\n" + "=" * 66)
    if gate_ok:
        print("Ready.  Start the system with:   python run.py")
        print("Then open:                       http://127.0.0.1:8000")
    else:
        print("The environment gate reported failures. Resolve them before running.")
    print("=" * 66)
    return 0 if gate_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
