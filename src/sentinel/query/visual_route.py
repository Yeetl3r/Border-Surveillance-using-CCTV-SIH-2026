"""F11 -- forensic visual grounding with LocateAnything-3B.

Open-vocabulary grounding on alert keyframes, on demand. Never per-frame: this
is a 3-billion-parameter VLM and running it on the video stream would be
absurd. It answers "where in this frame is the person carrying a bag?" against
a keyframe an operator has already chosen to look at.

Three engineering decisions worth stating.

**Queue depth one, with rejection.** If a grounding job is already running, the
next request is refused with 503 rather than queued. Refusing is the feature.
A queue in front of a multi-second GPU job is how a live demo runs out of
memory, and the honest response to "the model is busy" is to say so.

**The decord shim is registered in ``sys.modules``, not written to disk.** The
model's remote code imports ``decord`` (a video reader we do not need for
single images), and ``decord==0.6.0`` has no wheels for current Python. The
tempting fix -- dropping a ``decord.py`` file in the project root -- shadows the
real package for *every* import in the project, which is a trap for whoever
next installs it properly. Registering a stub module object under an explicit
lock, scoped to this loader, does the same job without the landmine.

**Licence.** LocateAnything-3B is released under the NVIDIA Open Model License
for non-commercial use -- academic and non-profit research only -- and its base
LM (Qwen2.5-3B-Instruct) carries the Qwen Research licence. That is surfaced in
the API response and in the UI, because "can this be fielded?" is a fair
question from a government evaluator and it deserves an accurate answer rather
than silence.
"""

from __future__ import annotations

import re
import sys
import threading
import time
import types
from pathlib import Path
from typing import Any

import numpy as np

from ..config import SystemCfg, VlmCfg, resolve

_BOX_RE = re.compile(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>")
_POINT_RE = re.compile(r"<box><(\d+)><(\d+)></box>")

_shim_lock = threading.Lock()


def _install_decord_shim() -> bool:
    """Register a stub ``decord`` module so the model's import check passes.

    Returns True if a stub was installed (i.e. the real package was absent).
    """
    with _shim_lock:
        if "decord" in sys.modules:
            return False
        try:
            import decord  # noqa: F401
            return False
        except ImportError:
            pass
        stub = types.ModuleType("decord")
        stub.__doc__ = (
            "Stub registered by sentinel.query.visual_route. LocateAnything-3B's "
            "remote code imports decord for video input; SENTINEL only ever passes "
            "it single keyframes, so no video reader is required. Registered in "
            "sys.modules rather than written to disk so it cannot shadow a real "
            "decord installation elsewhere in the project."
        )

        class _Unavailable:
            def __init__(self, *a, **k):
                raise RuntimeError(
                    "decord is not installed. SENTINEL passes single images to "
                    "LocateAnything-3B and does not use decord's video path."
                )

        stub.VideoReader = _Unavailable
        stub.cpu = lambda *a, **k: None
        stub.gpu = lambda *a, **k: None
        stub.bridge = types.SimpleNamespace(set_bridge=lambda *a, **k: None)
        sys.modules["decord"] = stub
        return True


class VisualGroundingRoute:
    """On-demand grounding. Loads lazily, serves one request at a time."""

    def __init__(self, cfg: SystemCfg) -> None:
        self.cfg = cfg
        self.vcfg: VlmCfg = cfg.vlm
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.available = False
        self.load_error = ""
        self.loading = False
        self.shim_installed = False
        self._slot = threading.Semaphore(max(1, self.vcfg.queue_depth))
        self._load_lock = threading.Lock()
        self.stats = {"requests": 0, "rejected": 0, "errors": 0,
                      "last_latency_ms": 0.0}

    # -- status --------------------------------------------------------------
    @property
    def weights_path(self) -> Path:
        return resolve(self.cfg, self.vcfg.local_dir)

    @property
    def weights_present(self) -> bool:
        p = self.weights_path
        return p.exists() and any(p.glob("*.safetensors"))

    def describe(self) -> dict[str, Any]:
        return {
            "enabled": self.vcfg.enabled,
            "model_id": self.vcfg.model_id,
            "weights_present": self.weights_present,
            "loaded": self.available,
            "loading": self.loading,
            "error": self.load_error,
            "licence": self.vcfg.licence,
            "queue_depth": self.vcfg.queue_depth,
            "busy": self._slot._value == 0,           # noqa: SLF001 - status only
            "stats": dict(self.stats),
            "decord_shim": self.shim_installed,
        }

    # -- loading -------------------------------------------------------------
    def ensure_loaded(self) -> bool:
        if self.available:
            return True
        if not self.vcfg.enabled:
            self.load_error = "F11 disabled in config"
            return False
        if not self.weights_present:
            self.load_error = (
                f"LocateAnything-3B weights not found at {self.weights_path}. "
                f"Run: python scripts/fetch_weights.py --vlm"
            )
            return False

        with self._load_lock:
            if self.available:
                return True
            self.loading = True
            try:
                import torch
                from transformers import AutoModel, AutoProcessor, AutoTokenizer

                self.shim_installed = _install_decord_shim()
                dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                         "float32": torch.float32}[self.vcfg.dtype]
                dev = self.vcfg.device
                if dev == "auto":
                    dev = "cuda" if torch.cuda.is_available() else "cpu"
                if dev == "cpu":
                    dtype = torch.float32

                p = str(self.weights_path)
                self.tokenizer = AutoTokenizer.from_pretrained(p, trust_remote_code=True)
                self.processor = AutoProcessor.from_pretrained(p, trust_remote_code=True)
                self.model = AutoModel.from_pretrained(
                    p, torch_dtype=dtype, trust_remote_code=True).to(dev).eval()
                self.torch, self.device, self.dtype = torch, dev, dtype
                self.available = True
                self.load_error = ""
            except Exception as exc:                      # noqa: BLE001
                self.load_error = f"{type(exc).__name__}: {exc}"
                self.available = False
            finally:
                self.loading = False
        return self.available

    # -- inference -----------------------------------------------------------
    def ground(self, image_bgr: np.ndarray, phrase: str,
               mode: str = "multi") -> dict[str, Any]:
        """Ground a phrase in one keyframe. Never called per-frame."""
        self.stats["requests"] += 1

        if not self._slot.acquire(blocking=False):
            self.stats["rejected"] += 1
            return {
                "route": "grounding", "ok": False, "status": 503,
                "error": "grounding model busy -- one request at a time by design",
                "licence": self.vcfg.licence,
            }
        try:
            if not self.ensure_loaded():
                return {"route": "grounding", "ok": False, "status": 503,
                        "error": self.load_error, "licence": self.vcfg.licence,
                        "fallback": "structured SQL route is unaffected"}

            import cv2
            from PIL import Image

            h, w = image_bgr.shape[:2]
            m = self.vcfg.max_image_side
            if max(h, w) > m:
                s = m / max(h, w)
                image_bgr = cv2.resize(image_bgr, (int(w * s), int(h * s)),
                                       interpolation=cv2.INTER_AREA)
            img = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))

            verb = ("all the instances that match" if mode == "multi"
                    else "a single instance that matches")
            prompt = (f"Locate {verb} the following description: {phrase}.")

            t0 = time.perf_counter()
            answer = self._generate(img, prompt)
            ms = (time.perf_counter() - t0) * 1000
            self.stats["last_latency_ms"] = round(ms, 1)

            boxes = self.parse_boxes(answer, w, h)
            return {
                "route": "grounding", "ok": True, "status": 200,
                "model_id": self.vcfg.model_id, "phrase": phrase, "mode": mode,
                "answer": answer, "boxes": boxes, "count": len(boxes),
                "latency_ms": round(ms, 1), "licence": self.vcfg.licence,
            }
        except Exception as exc:                          # noqa: BLE001
            self.stats["errors"] += 1
            return {"route": "grounding", "ok": False, "status": 500,
                    "error": f"{type(exc).__name__}: {exc}",
                    "licence": self.vcfg.licence}
        finally:
            self._slot.release()

    def _generate(self, img, prompt: str) -> str:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": img}, {"type": "text", "text": prompt}]}]

        proc = self.processor
        apply_tpl = getattr(proc, "py_apply_chat_template", None) or \
            getattr(proc, "apply_chat_template")
        text = apply_tpl(messages, tokenize=False, add_generation_prompt=True)
        images, videos = proc.process_vision_info(messages)
        inputs = proc(text=[text], images=images, videos=videos,
                      return_tensors="pt").to(self.device)

        with self.torch.no_grad():
            resp = self.model.generate(
                pixel_values=inputs["pixel_values"].to(self.dtype),
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                image_grid_hws=inputs.get("image_grid_hws"),
                tokenizer=self.tokenizer,
                max_new_tokens=self.vcfg.max_new_tokens,
                use_cache=True,
                generation_mode="hybrid",
                temperature=0.7, do_sample=True, top_p=0.9,
                repetition_penalty=1.1, verbose=False,
            )
        out = resp[0] if isinstance(resp, tuple) else resp
        return out if isinstance(out, str) else str(out)

    # -- parsing -------------------------------------------------------------
    @staticmethod
    def parse_boxes(answer: str, width: int, height: int) -> list[dict]:
        """Model coordinates are normalised integers in [0, 1000]."""
        out = []
        for m in _BOX_RE.finditer(answer):
            x1, y1, x2, y2 = (int(g) for g in m.groups())
            out.append({
                "bbox": [round(x1 / 1000 * width, 1), round(y1 / 1000 * height, 1),
                         round(x2 / 1000 * width, 1), round(y2 / 1000 * height, 1)],
                "norm": [x1, y1, x2, y2],
            })
        return out

    @staticmethod
    def parse_points(answer: str, width: int, height: int) -> list[dict]:
        boxed = {m.span() for m in _BOX_RE.finditer(answer)}
        out = []
        for m in _POINT_RE.finditer(answer):
            if any(s <= m.start() < e for s, e in boxed):
                continue                       # part of a 4-tuple box, not a point
            x, y = int(m.group(1)), int(m.group(2))
            out.append({"point": [round(x / 1000 * width, 1),
                                  round(y / 1000 * height, 1)]})
        return out
