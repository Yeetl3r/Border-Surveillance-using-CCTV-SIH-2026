"""SENTINEL entry point.

    python -m sentinel                      # demo profile, default config
    python -m sentinel --profile operational
    python -m sentinel --tracker bytetrack --pose
    python -m sentinel --check              # environment gate only, no server

One command starts the system: it ingests the configured sources, serves the
C2 interface, and runs entirely offline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("sentinel", description="SENTINEL border surveillance C2")
    p.add_argument("--root", default=".", help="project root containing configs/")
    p.add_argument("--profile", default=None, help="demo | operational")
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--tracker", choices=["botsort", "bytetrack"], default=None,
                   help="F2: both are wired; this selects which runs")
    p.add_argument("--detector", default=None, help="override detector weights")
    p.add_argument("--pose", action="store_true",
                   help="F16: enable infiltration-stance detection (off by default)")
    p.add_argument("--no-identity", action="store_true", help="disable F8 re-ID")
    p.add_argument("--no-vlm", action="store_true", help="disable F11 grounding")
    p.add_argument("--sahi", action="store_true", help="enable sliced inference")
    p.add_argument("--source", default=None, help="override camera 1 source (file or rtsp://)")
    p.add_argument("--online", action="store_true",
                   help="allow network access (F15 is on by default)")
    p.add_argument("--check", action="store_true", help="run the environment gate and exit")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    sys.path.insert(0, str(Path(args.root).resolve() / "src"))

    if args.check:
        from scripts.verify_env import main as verify   # type: ignore[import-not-found]
        return verify()

    from sentinel.config import load_config

    overrides: dict = {}
    if args.tracker:
        overrides.setdefault("tracker", {})["kind"] = args.tracker
    if args.detector:
        overrides.setdefault("detector", {})["model"] = args.detector
    if args.pose:
        overrides.setdefault("pose", {})["enabled"] = True
    if args.sahi:
        overrides.setdefault("detector", {}).setdefault("sahi", {})["enabled"] = True
    if args.no_identity:
        overrides.setdefault("identity", {})["enabled"] = False
    if args.no_vlm:
        overrides.setdefault("vlm", {})["enabled"] = False
    if args.online:
        overrides.setdefault("runtime", {})["offline"] = False
    if args.host:
        overrides.setdefault("api", {})["host"] = args.host
    if args.port:
        overrides.setdefault("api", {})["port"] = args.port

    cfg = load_config(args.root, profile=args.profile, overrides=overrides)

    if args.source:
        if not cfg.cameras:
            raise SystemExit("no cameras configured in configs/cameras/")
        cfg.cameras[0].source = args.source

    import uvicorn
    from sentinel.api import create_app

    banner(cfg)
    app = create_app(cfg)
    uvicorn.run(app, host=cfg.api.host, port=cfg.api.port, log_level="warning")
    return 0


def banner(cfg) -> None:
    cams = ", ".join(c.camera_id for c in cfg.cameras if c.enabled)
    print(f"""
  SENTINEL  --  border surveillance C2
  ------------------------------------------------------------------
  profile    {cfg.profile_name}   (baggage dwell {cfg.baggage.dwell_s:.0f}s,
             confirm {cfg.confirm.window_s}s / ratio {cfg.confirm.min_ratio} / min {cfg.confirm.min_hits})
  detector   {cfg.detector.model}  imgsz={cfg.detector.imgsz} sahi={cfg.detector.sahi.enabled}
  tracker    {cfg.tracker.kind}
  pose F16   {'enabled' if cfg.pose.enabled else 'disabled'}
  identity   {'enabled' if cfg.identity.enabled else 'disabled'}  (open-set, friendly gallery)
  grounding  {'enabled' if cfg.vlm.enabled else 'disabled'}  {cfg.vlm.model_id}
  offline    {cfg.runtime.offline}  (F15)
  cameras    {cams}
  ------------------------------------------------------------------
  C2 interface -> http://{cfg.api.host}:{cfg.api.port}
""", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
