"""Render annotated stills or video from a run, for slides and verification.

    python tools/render_demo.py --stills 3
    python tools/render_demo.py --video out/demo.mp4 --seconds 24

Runs the real pipeline headlessly and draws the same dual-axis scheme the live
client uses:

    THREAT     -> box COLOUR
    CONFIDENCE -> box STYLE   (solid / dashed / hatched)

Two things this is careful about. It draws onto the OPERATOR frame, never the
raw one -- the raw buffer is what the detector saw and what the evidence store
keeps, and annotating it would conflate the input with the decision. And it
burns in the "SCALE ESTIMATED" marker when the camera's homography is not
surveyed, so a still lifted into a slide deck carries its own caveat.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

THREAT_BGR = {
    "INFO": (230, 169, 90),
    "FRIENDLY": (132, 220, 61),
    "WATCH": (61, 194, 255),
    "WARNING": (26, 140, 255),
    "CRITICAL": (68, 51, 255),
}
ACCENT = (84, 255, 183)


def band_for(c: float) -> str:
    return "HIGH" if c >= 0.75 else "MEDIUM" if c >= 0.5 else "LOW"


def dashed_rect(img, p1, p2, colour, thick=2, dash=9, gap=6) -> None:
    x1, y1 = p1
    x2, y2 = p2
    for (ax, ay, bx, by) in ((x1, y1, x2, y1), (x2, y1, x2, y2),
                             (x2, y2, x1, y2), (x1, y2, x1, y1)):
        n = int(np.hypot(bx - ax, by - ay))
        if n == 0:
            continue
        for s in range(0, n, dash + gap):
            e = min(s + dash, n)
            t0, t1 = s / n, e / n
            cv2.line(img, (int(ax + (bx - ax) * t0), int(ay + (by - ay) * t0)),
                     (int(ax + (bx - ax) * t1), int(ay + (by - ay) * t1)),
                     colour, thick, cv2.LINE_AA)


def hatch_box(img, p1, p2, colour) -> None:
    x1, y1 = p1
    x2, y2 = p2
    roi = img[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
    if roi.size == 0:
        return
    layer = roi.copy()
    h, w = roi.shape[:2]
    for i in range(-h, w, 8):
        cv2.line(layer, (i, h), (i + h, 0), colour, 1, cv2.LINE_AA)
    cv2.addWeighted(layer, 0.20, roi, 0.80, 0, roi)


def draw_zones(frame, zones) -> None:
    for z in zones.get("zones", []):
        pts = z.get("points_px")
        if not pts:
            continue
        arr = np.array(pts, np.int32).reshape((-1, 1, 2))
        overlay = frame.copy()
        cv2.fillPoly(overlay, [arr], (40, 70, 20))
        cv2.addWeighted(overlay, 0.13, frame, 0.87, 0, frame)
        cv2.polylines(frame, [arr], True, (110, 200, 70), 1, cv2.LINE_AA)

    for w in zones.get("tripwires", []):
        pts = w.get("points_px")
        if not pts:
            continue
        (ax, ay), (bx, by) = (tuple(map(int, p)) for p in pts)
        cv2.line(frame, (ax, ay), (bx, by), (26, 140, 255), 3, cv2.LINE_AA)
        # Inbound direction comes from the geometry layer already projected to
        # pixels -- deriving it here would get the handedness wrong.
        arrow = w.get("inbound_arrow_px")
        if arrow:
            (sx, sy), (tx, ty) = (tuple(map(int, q)) for q in arrow)
            cv2.arrowedLine(frame, (sx, sy), (tx, ty), (26, 140, 255), 3,
                            cv2.LINE_AA, tipLength=0.32)
            cv2.putText(frame, "INBOUND", (tx + 8, ty + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (26, 140, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, w.get("label", w["id"]), (ax + 10, ay - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (26, 140, 255), 2, cv2.LINE_AA)

    for p in zones.get("proximity", []):
        pts = p.get("points_px")
        if not pts:
            continue
        px, py = (int(v) for v in pts[0])
        cv2.circle(frame, (px, py), 11, (68, 51, 255), 2, cv2.LINE_AA)
        cv2.drawMarker(frame, (px, py), (68, 51, 255), cv2.MARKER_CROSS, 16, 2)
        cv2.putText(frame, f"{p.get('label', p['id'])} <{p['threshold_m']}m",
                    (px + 16, py + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (68, 51, 255), 1, cv2.LINE_AA)


def draw_boxes(frame, boxes) -> None:
    for b in boxes:
        colour = THREAT_BGR.get(b["threat"], THREAT_BGR["INFO"])
        band = band_for(b["confidence"])
        x1, y1, x2, y2 = (int(v) for v in b["bbox"])

        if band == "LOW":
            hatch_box(frame, (x1, y1), (x2, y2), colour)
            dashed_rect(frame, (x1, y1), (x2, y2), colour, 2, 4, 5)
        elif band == "MEDIUM":
            dashed_rect(frame, (x1, y1), (x2, y2), colour, 2)
        else:
            cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2, cv2.LINE_AA)

        c = int(min(15, (x2 - x1) / 3, (y2 - y1) / 3))
        for (cx, cy, sx, sy) in ((x1, y1, 1, 1), (x2, y1, -1, 1),
                                 (x1, y2, 1, -1), (x2, y2, -1, -1)):
            cv2.line(frame, (cx, cy), (cx + sx * c, cy), colour, 3, cv2.LINE_AA)
            cv2.line(frame, (cx, cy), (cx, cy + sy * c), colour, 3, cv2.LINE_AA)

        bits = [f"#{b['track_id']}", b["group"].upper(),
                f"{b['confidence'] * 100:.0f}%"]
        if b.get("footpoint_m"):
            bits.append(f"{b['footpoint_m'][1]:.1f}m")
        ident = b.get("identity") or {}
        if ident.get("label") == "FRIENDLY":
            bits.append(ident.get("matched") or "FRIENDLY")
        elif ident.get("label") == "UNIDENTIFIED":
            bits.append("UNIDENT")
        if b.get("stance") not in (None, "UNKNOWN", "STANDING"):
            bits.append(b["stance"])
        label = "  ".join(bits)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ly = max(th + 6, y1 - 6)
        cv2.rectangle(frame, (x1, ly - th - 6), (x1 + tw + 10, ly + 3), colour, -1)
        cv2.putText(frame, label, (x1 + 5, ly - 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (12, 20, 14), 1, cv2.LINE_AA)

        cv2.circle(frame, ((x1 + x2) // 2, y2), 4, colour, -1, cv2.LINE_AA)


def draw_hud(frame, pipe, cfg, metric_est: bool) -> None:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 34), (14, 22, 18), -1)
    cv2.line(frame, (0, 34), (w, 34), (60, 90, 70), 1)
    threat = pipe.alerts.highest_live.value
    cv2.putText(frame, "SENTINEL // BORDER SURVEILLANCE C2", (14, 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (241, 247, 240), 1, cv2.LINE_AA)
    cv2.putText(frame, f"THREAT: {threat}", (w - 250, 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                THREAT_BGR.get(threat, (200, 200, 200)), 2, cv2.LINE_AA)
    perf = pipe.perf.as_dict()
    cv2.putText(frame, f"{perf['fps']:.0f} FPS  |  {len(pipe.tracks)} TRACKS"
                       f"  |  {pipe.health_state}",
                (int(w * 0.40), 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                ACCENT, 1, cv2.LINE_AA)

    # A still lifted into a deck must carry its own caveat about the scale.
    note = ("METRIC: HOMOGRAPHY (SCALE ESTIMATED, NOT SURVEYED)" if metric_est
            else "PIXEL SPACE -- CAMERA UNCALIBRATED")
    cv2.putText(frame, note, (14, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                (61, 194, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "OPERATOR STREAM (CLAHE) -- DETECTOR SEES RAW",
                (w - 430, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (140, 163, 150), 1, cv2.LINE_AA)


def main() -> int:
    ap = argparse.ArgumentParser("render_demo")
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--camera", default=None)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--video", default=None, help="write an annotated mp4 here")
    ap.add_argument("--stills", type=int, default=0, help="save N stills with alerts")
    ap.add_argument("--outdir", default="out")
    args = ap.parse_args()

    from sentinel.config import load_config
    from sentinel.pipeline import SentinelSystem

    cfg = load_config(args.root)
    system = SentinelSystem(cfg)
    system.start()

    cam_id = args.camera or next(iter(system.pipelines))
    pipe = system.pipelines[cam_id]
    zones = pipe.zones.overlay()
    metric_est = pipe.zones.metric

    outdir = Path(args.root) / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    writer = None
    frames_written, stills, seen = 0, 0, set()
    t0 = time.time()
    print(f"rendering {cam_id} for {args.seconds:.0f}s ...")

    try:
        while time.time() - t0 < args.seconds:
            frame = pipe.latest_frame
            ov = pipe.latest_overlay
            if frame is None or not ov:
                time.sleep(0.03)
                continue
            key = ov.get("frame_idx")
            if key in seen:
                time.sleep(0.01)
                continue
            seen.add(key)

            canvas = frame.copy()
            draw_zones(canvas, zones)
            draw_boxes(canvas, ov.get("boxes", []))
            draw_hud(canvas, pipe, cfg, metric_est)

            if args.video:
                if writer is None:
                    h, w = canvas.shape[:2]
                    vp = Path(args.video)
                    vp.parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(str(vp),
                                             cv2.VideoWriter_fourcc(*"mp4v"),
                                             20.0, (w, h))
                writer.write(canvas)
                frames_written += 1

            if stills < args.stills and ov.get("boxes"):
                interesting = any(b["threat"] not in ("INFO",) for b in ov["boxes"])
                if interesting or stills == 0:
                    p = outdir / f"still_{stills + 1:02d}.png"
                    cv2.imwrite(str(p), canvas)
                    print(f"  still -> {p}  "
                          f"({len(ov['boxes'])} boxes, threat "
                          f"{system.alerts.highest_live.value})")
                    stills += 1
                    time.sleep(1.5)
            time.sleep(0.02)
    finally:
        system.stop()
        if writer is not None:
            writer.release()

    if args.video and frames_written:
        # Re-encode to H.264 so the file plays in a browser and in slide decks.
        ff = shutil.which("ffmpeg")
        if ff:
            src = Path(args.video)
            tmp = src.with_suffix(".h264.mp4")
            r = subprocess.run(
                [ff, "-y", "-hide_banner", "-loglevel", "error", "-i", str(src),
                 "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(tmp)],
                capture_output=True)
            if r.returncode == 0:
                tmp.replace(src)
        print(f"video -> {args.video}  ({frames_written} frames)")

    print(f"done. alerts promoted={system.alerts.counters['promoted']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
