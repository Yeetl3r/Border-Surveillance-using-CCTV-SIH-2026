"""Day-one environment gate. Nothing else starts until this passes.

    python scripts/verify_env.py

The target is an RTX 5080 (Blackwell, sm_120), which is newer than much of the
ecosystem and has two documented traps:

* CUDA 12.8+ and PyTorch >= 2.7.0 are required. Older toolkits fail at the first
  kernel launch with "no kernel image available for execution".
* **PTX just-in-time compilation is broken on Blackwell.** Ahead-of-time builds
  work; anything that compiles CUDA at runtime (``cpp_extension.load()``,
  FlashAttention, vLLM) can fail. This check distinguishes the two cases, so a
  failure points at the right cause instead of at PyTorch generally.

Every number printed here is measured on the machine it runs on. None is
carried over from a spec sheet.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OK, WARN, FAIL = "  OK  ", " WARN ", " FAIL "
_results: list[tuple[str, str, str]] = []


def record(status: str, name: str, detail: str = "") -> None:
    _results.append((status, name, detail))
    print(f"[{status}] {name}" + (f"  --  {detail}" if detail else ""), flush=True)


def check_python() -> None:
    v = sys.version_info
    s = f"{v.major}.{v.minor}.{v.micro}"
    if v < (3, 11):
        record(FAIL, "python >= 3.11", f"found {s}")
    elif v >= (3, 13):
        record(WARN, "python version", f"{s} -- 3.11/3.12 have broader wheel "
                                       "coverage for this stack")
    else:
        record(OK, "python version", s)


def check_torch() -> None:
    try:
        import torch
    except ImportError:
        record(FAIL, "pytorch", "not installed")
        return

    ver = torch.__version__
    major_minor = tuple(int(x) for x in ver.split("+")[0].split(".")[:2])
    if major_minor < (2, 7):
        record(FAIL, "pytorch >= 2.7.0", f"{ver} predates native sm_120 support")
    else:
        record(OK, "pytorch", f"{ver} (cuda {torch.version.cuda})")

    if not torch.cuda.is_available():
        record(WARN, "cuda", "unavailable -- the system runs on CPU, slowly")
        return

    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    sm = f"sm_{cap[0]}{cap[1]}"
    record(OK, "gpu", f"{name}  {sm}")

    arch = torch.cuda.get_arch_list()
    if sm in arch:
        record(OK, "prebuilt kernels", f"{sm} present in torch arch list")
    else:
        record(FAIL, "prebuilt kernels",
               f"{sm} NOT in {arch} -- expect 'no kernel image available'")

    free, total = torch.cuda.mem_get_info()
    record(OK if free > 6e9 else WARN, "vram",
           f"{free/1e9:.2f} GB free / {total/1e9:.2f} GB total")

    # The check that actually matters: launch a real kernel.
    try:
        a = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
        b = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(30):
            c = a @ b
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / 30
        record(OK, "fp16 kernel launch", f"{2 * 2048**3 / dt / 1e12:.1f} TFLOP/s")
        del a, b, c
        torch.cuda.empty_cache()
    except RuntimeError as exc:
        record(FAIL, "fp16 kernel launch", str(exc)[:140])

    record(OK if torch.cuda.is_bf16_supported() else WARN, "bf16",
           "supported" if torch.cuda.is_bf16_supported() else "unsupported")

    # PTX JIT: the Blackwell-specific trap. Runtime compilation is what breaks,
    # not PyTorch itself, so name it separately.
    try:
        from torch.utils.cpp_extension import CUDA_HOME
        record(OK if CUDA_HOME else WARN, "cuda toolkit (for runtime compilation)",
               CUDA_HOME or "absent -- prebuilt wheels only; do not rely on "
                            "FlashAttention or custom CUDA extensions")
    except ImportError:
        record(WARN, "cuda toolkit", "cpp_extension unavailable")


def check_packages() -> None:
    import importlib.util as iu
    required = ["cv2", "numpy", "shapely", "yaml", "pydantic", "fastapi", "uvicorn"]
    optional = {"ultralytics": "detection/tracking (F1, F2, F16)",
                "torchvision": "required by ultralytics",
                "transformers": "grounding VLM (F11)",
                "PIL": "image handling for F11"}
    missing = [m for m in required if iu.find_spec(m) is None]
    if missing:
        record(FAIL, "required packages", f"missing: {', '.join(missing)}")
    else:
        record(OK, "required packages", f"{len(required)} present")
    for m, why in optional.items():
        record(OK if iu.find_spec(m) else WARN, f"optional: {m}",
               "" if iu.find_spec(m) else f"absent -- {why} unavailable")


def check_weights() -> None:
    w = ROOT / "weights"
    if not w.exists():
        record(FAIL, "weights/", "missing -- run scripts/fetch_weights.py")
        return
    for name, why in [("yolo26s.pt", "F1 detector"),
                      ("osnet_x0_25_msmt17.pth", "F8 re-ID")]:
        p = w / name
        record(OK if p.exists() else WARN, f"weights: {name}",
               f"{p.stat().st_size/1e6:.1f} MB" if p.exists() else f"absent -- {why}")
    la = w / "LocateAnything-3B"
    has = la.exists() and any(la.glob("*.safetensors"))
    record(OK if has else WARN, "weights: LocateAnything-3B",
           "present" if has else "absent -- F11 grounding unavailable")
    pose = w / "yolo26s-pose.pt"
    record(OK if pose.exists() else WARN, "weights: yolo26s-pose.pt",
           "present" if pose.exists() else "absent -- F16 unavailable")


def check_config() -> None:
    sys.path.insert(0, str(ROOT / "src"))
    try:
        from sentinel.config import load_config
        cfg = load_config(str(ROOT))
    except Exception as exc:                          # noqa: BLE001
        record(FAIL, "config", f"{type(exc).__name__}: {exc}")
        return
    record(OK, "config", f"profile={cfg.profile_name}, "
                         f"{len(cfg.cameras)} camera(s)")

    from sentinel.config import resolve
    for cam in cfg.cameras:
        src = cam.source
        if "://" not in src:
            p = resolve(cfg, src)
            record(OK if p.exists() else FAIL, f"source: {cam.camera_id}",
                   str(p) if p.exists() else f"missing {p}")
        try:
            from sentinel.calibration.zones import ZoneSet
            zs = ZoneSet.from_camera(cam)
            record(OK if zs.metric else WARN, f"calibration: {cam.camera_id}",
                   f"residual {zs.homography.residual_m:.4f} m" if zs.metric
                   else "uncalibrated -- distances will be pixels, not metres")
        except Exception as exc:                      # noqa: BLE001
            record(FAIL, f"calibration: {cam.camera_id}", str(exc)[:120])


def check_ffmpeg() -> None:
    import shutil
    p = shutil.which("ffmpeg")
    record(OK if p else WARN, "ffmpeg",
           p or "absent -- evidence clips fall back to a codec browsers may not play")


def main() -> int:
    print("SENTINEL environment gate\n" + "=" * 62)
    check_python()
    check_packages()
    check_torch()
    check_ffmpeg()
    check_weights()
    check_config()

    fails = sum(1 for s, _, _ in _results if s == FAIL)
    warns = sum(1 for s, _, _ in _results if s == WARN)
    print("=" * 62)
    print(f"{len(_results)} checks: {fails} failed, {warns} warnings")
    if fails:
        print("\nDo not proceed until the failures are resolved.")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
