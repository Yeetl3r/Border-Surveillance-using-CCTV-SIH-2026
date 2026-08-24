"""Manufacture degraded footage from clean source video.

    python tools/degrade.py --input clean.mp4 --output assets/degraded.mp4 \
                            --preset legacy_cctv

The problem statement says "existing CCTV infrastructure". That word is
load-bearing: it means legacy analogue-to-IP conversions, 720p or worse
sensors, aggressive H.264 with visible blocking, smeared low-contrast night
footage, and lenses nobody has cleaned in years.

A system demonstrated on clean modern video has not addressed the stated
problem, and an experienced evaluator will say so. This tool exists so that
claim can be tested rather than asserted -- run the pipeline on both the clean
and degraded versions and report the delta honestly, including where it gets
worse.

Presets are cumulative transformations, applied in capture order: resolution
loss, then optics, then sensor, then codec. Applying them in a different order
produces artefacts real cameras do not have.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

PRESETS = {
    "legacy_cctv": dict(width=704, fps=12, bitrate="180k", blur=0.6,
                        noise=6, gamma=1.0, contrast=0.85),
    "night_ir": dict(width=960, fps=15, bitrate="140k", blur=0.9,
                     noise=11, gamma=0.7, contrast=0.7, grey=True),
    "harsh": dict(width=480, fps=8, bitrate="90k", blur=1.4,
                  noise=16, gamma=0.65, contrast=0.6, grey=True),
    "bitrate_only": dict(width=None, fps=None, bitrate="120k", blur=0,
                         noise=0, gamma=1.0, contrast=1.0),
}


def build_filters(p: dict) -> str:
    f = []
    if p.get("width"):
        f.append(f"scale={p['width']}:-2:flags=bilinear")
    if p.get("grey"):
        f.append("hue=s=0")
    if p.get("blur"):
        f.append(f"gblur=sigma={p['blur']}")
    if p.get("contrast", 1.0) != 1.0 or p.get("gamma", 1.0) != 1.0:
        f.append(f"eq=contrast={p.get('contrast',1.0)}:gamma={p.get('gamma',1.0)}")
    if p.get("noise"):
        f.append(f"noise=alls={p['noise']}:allf=t+u")
    if p.get("width"):
        # Upscale back so downstream resolution is unchanged but the detail is
        # genuinely gone -- which is what a legacy sensor actually gives you.
        f.append("scale=1280:-2:flags=bilinear")
    return ",".join(f) if f else "null"


def main() -> int:
    ap = argparse.ArgumentParser("degrade")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--preset", default="legacy_cctv", choices=sorted(PRESETS))
    ap.add_argument("--list", action="store_true", help="show presets and exit")
    args = ap.parse_args()

    if args.list:
        for k, v in PRESETS.items():
            print(f"  {k:14} {v}")
        return 0

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("! ffmpeg not found on PATH.")
        return 1

    src, dst = Path(args.input), Path(args.output)
    if not src.exists():
        print(f"! {src} not found")
        return 1
    dst.parent.mkdir(parents=True, exist_ok=True)

    p = PRESETS[args.preset]
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(src),
           "-vf", build_filters(p)]
    if p.get("fps"):
        cmd += ["-r", str(p["fps"])]
    cmd += ["-c:v", "libx264", "-b:v", p["bitrate"], "-maxrate", p["bitrate"],
            "-bufsize", p["bitrate"], "-preset", "veryfast",
            "-pix_fmt", "yuv420p", "-an", str(dst)]

    print(f"degrading {src.name} -> {dst.name}  [{args.preset}]")
    print(f"  filters: {build_filters(p)}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("! ffmpeg failed:\n" + r.stderr[-900:])
        return 1

    a, b = src.stat().st_size, dst.stat().st_size
    print(f"  {a/1e6:.2f} MB -> {b/1e6:.2f} MB  ({b/a*100:.0f}% of source)")
    print(f"\nNow run the pipeline against both and compare. Report the delta,")
    print(f"including where degradation makes it worse -- that is the honest result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
