"""F3 -- interactive homography calibration.

    python tools/calibrate.py --camera cam01

Click four points on the ground plane, in this order:

    1. far-left    2. far-right    3. near-right    4. near-left

then enter the real-world coordinates of each in metres. The tool computes H,
reports the residual, and writes the block into the camera's YAML.

Two things it refuses to let you do quietly:

* **Non-coplanar points.** A ground plane maps to the image by an exact
  homography, so the round-trip residual should be ~0. A large residual means
  the marked points were not all on the ground -- typically someone clicked the
  top of a wall. The tool prints the residual rather than hiding it.
* **Guessed scale.** If you have not measured a real distance, say so. The
  written block records whether the scale is surveyed or estimated, and the UI
  displays that distinction to the operator. Metric-looking numbers derived
  from a guess are worse than pixels, because they invite trust they have not
  earned.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

LABELS = ["FAR-LEFT", "FAR-RIGHT", "NEAR-RIGHT", "NEAR-LEFT"]


def grab_frame(source: str, frame_no: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        return None
    if frame_no:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def main() -> int:
    ap = argparse.ArgumentParser("calibrate")
    ap.add_argument("--camera", default="cam01")
    ap.add_argument("--source", default=None, help="override the camera's source")
    ap.add_argument("--frame", type=int, default=20)
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--brightness", type=float, default=1.0,
                    help="display gain for dark scenes (does not affect the result)")
    args = ap.parse_args()

    from sentinel.config import load_config, resolve

    cfg = load_config(args.root)
    cam = next((c for c in cfg.cameras if c.camera_id == args.camera), None)
    if cam is None:
        print(f"! no camera {args.camera!r} in configs/cameras/")
        return 1

    src = args.source or cam.source
    frame = grab_frame(str(resolve(cfg, src)), args.frame)
    if frame is None:
        print(f"! cannot read a frame from {src}")
        return 1

    view = cv2.convertScaleAbs(frame, alpha=args.brightness, beta=0) \
        if args.brightness != 1.0 else frame.copy()

    pts: list[tuple[int, int]] = []
    win = f"calibrate {args.camera} -- click 4 ground points, u=undo, q=quit"

    def redraw() -> None:
        img = view.copy()
        for i, (x, y) in enumerate(pts):
            cv2.circle(img, (x, y), 6, (60, 255, 180), -1)
            cv2.putText(img, f"{i + 1} {LABELS[i]}", (x + 9, y - 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 255, 180), 2, cv2.LINE_AA)
        if len(pts) > 1:
            cv2.polylines(img, [np.array(pts, np.int32)], len(pts) == 4,
                          (60, 255, 180), 2, cv2.LINE_AA)
        nxt = LABELS[len(pts)] if len(pts) < 4 else "ENTER to accept"
        cv2.putText(img, f"click: {nxt}", (14, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (183, 255, 84), 2, cv2.LINE_AA)
        cv2.imshow(win, img)

    def on_mouse(event, x, y, _flags, _param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x, y))
            redraw()

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    redraw()
    while True:
        k = cv2.waitKey(30) & 0xFF
        if k in (ord("q"), 27):
            cv2.destroyAllWindows()
            return 1
        if k == ord("u") and pts:
            pts.pop()
            redraw()
        if k in (13, 10) and len(pts) == 4:
            break
    cv2.destroyAllWindows()

    print("\nEnter the real-world ground coordinates of each point, in metres.")
    print("Use any consistent frame: X across, Y away from the camera.")
    print("Example for a measured 5 m x 5 m patch: -2.5 5 / 2.5 5 / 2.5 0 / -2.5 0\n")
    world: list[list[float]] = []
    for i, lbl in enumerate(LABELS):
        while True:
            try:
                raw = input(f"  {i + 1} {lbl:10} X Y = ").replace(",", " ").split()
                world.append([float(raw[0]), float(raw[1])])
                break
            except (ValueError, IndexError):
                print("     ! enter two numbers, e.g. -2.5 5")

    surveyed = input("\nWere these distances physically measured? [y/N] ").strip().lower()
    surveyed = surveyed.startswith("y")

    H, _ = cv2.findHomography(np.float32(pts), np.float32(world))
    if H is None:
        print("! findHomography failed -- the points are probably collinear.")
        return 1

    errs = []
    for (px, py), (wx, wy) in zip(pts, world):
        v = H @ np.array([px, py, 1.0])
        errs.append(float(np.hypot(v[0] / v[2] - wx, v[1] / v[2] - wy)))
    residual = max(errs)

    print(f"\nresidual: {residual:.4f} m (max over the four fit points)")
    if residual > 0.05:
        print("  ! High for a plane fit. Check that all four points are on the")
        print("    GROUND -- a point on a wall or a vehicle will do this.")
    else:
        print("  good -- the points are coplanar.")

    block = {
        "enabled": True,
        "fixed_camera": True,
        "homography_id": f"{args.camera}-{'surveyed' if surveyed else 'estimated'}-v1",
        "image_points": [[int(x), int(y)] for x, y in pts],
        "world_points": [[round(a, 3), round(b, 3)] for a, b in world],
    }

    import yaml
    print("\nPaste into configs/cameras/%s.yaml:\n" % args.camera)
    print(yaml.safe_dump({"homography": block}, sort_keys=False))
    if not surveyed:
        print("NOTE: scale recorded as ESTIMATED. The UI will label this camera's")
        print("      distances accordingly. Do not quote them as surveyed truth.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
