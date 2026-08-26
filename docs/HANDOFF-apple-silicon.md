# Handoff: Apple Silicon (MPS) support

This repository runs today on CUDA (developed on an RTX 5080) and falls back to
CPU. It has never been run on Apple Silicon. This document is the brief for
making it run on an M-series Mac using the Metal Performance Shaders backend.

**Nothing here requires retraining.** The two specialised detectors
(`weights/weapons_yolo26n.pt`, `weights/drones_yolo26n.pt`) are committed to the
repository precisely so that a new machine does not have to reproduce a GPU
training run. Everything below is device plumbing.

---

## Read these first, in this order

1. `SETUP.md` — how the project is installed and run.
2. `docs/DEMOS.md` — the six demonstrations and what each is meant to show.
3. `configs/system.yaml` — every threshold in the system, with the measurement
   that produced it recorded beside it. Do not change these values while
   porting; a number that was measured on CUDA is still the right number.
4. `docs/OPEN-DECISIONS.md` — assumptions taken without an answer.

---

## Environment first — do not skip this

```bash
python -c "import torch; print(torch.__version__, torch.backends.mps.is_available(), torch.backends.mps.is_built())"
```

Both must be `True`. If they are not, the PyTorch build is wrong and no amount
of code change will help.

Then:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

MPS does not implement every operator. Without this, an unimplemented op raises
instead of falling back to CPU, and it will be raised from inside Ultralytics
where the traceback is unhelpful. Set it before anything else.

---

## The one change that matters most

Device selection is currently duplicated in five places, each with its own
private `_cuda()` helper or inline `torch.cuda.is_available()`. **Do not add an
MPS branch to five copies.** Add one module and make every site call it:

```python
# src/sentinel/runtime/device.py  (new)

def pick_device(pref: str = "auto") -> str:
    """Return a device string Ultralytics and torch both accept.

    Order is deliberate: an explicit preference wins, then CUDA, then MPS, then
    CPU. MPS below CUDA because a machine with both is a machine where CUDA is
    the faster path.
    """
    ...

def supports_fp16(device: str) -> bool:
    """Half precision is reliable on CUDA and is not on MPS."""
    ...
```

Ultralytics accepts `device="mps"` as a string. `torch` accepts
`torch.device("mps")`. The existing code passes `0` (an integer CUDA ordinal) in
several places — that must become a string on the MPS path.

---

## Every site that needs changing

| File | Line (approx.) | What is there now |
|---|---|---|
| `src/sentinel/config/schema.py` | 26 | `RuntimeCfg.device: Literal["auto","cuda","cpu"]` — add `"mps"` |
| `src/sentinel/config/schema.py` | 342 | `VlmCfg.device` — same |
| `src/sentinel/config/schema.py` | 59 | `DetectorCfg.half: bool = True` — see fp16 below |
| `src/sentinel/perception/yolo_detector.py` | 76, 179 | `_pick_device()` — the perimeter detector |
| `src/sentinel/perception/pose.py` | 65, 280 | `_cuda()` and `device = 0 if ...` |
| `src/sentinel/perception/threat.py` | 97, 111, 180 | `_cuda()`, used by BOTH specialised detectors |
| `src/sentinel/identity/embedder.py` | 91 | OSNet device selection |
| `src/sentinel/query/visual_route.py` | 158 | LocateAnything-3B |
| `scripts/verify_env.py` | 59–65 | the environment gate reports CUDA only |
| `tools/train_detector.py` | 274, 282 | `device=0` hardcoded; only matters if you retrain |

---

## Half precision

`detector.half: true` in `configs/system.yaml` is fp16 inference. On MPS this is
at best unreliable and at worst silently wrong. **Force it off when the device
is MPS** — in code, not by editing the config, so that the same config file
still runs correctly on the CUDA machine:

```python
half = cfg.detector.half and supports_fp16(self.device)
```

This is the single most likely cause of "it runs but the boxes are nonsense".

---

## The device camera (Demo 6)

macOS needs two things Windows does not:

1. **Camera permission.** The first `cv2.VideoCapture(0)` triggers a TCC prompt
   against the *terminal application*, not against Python. If it is denied once
   it will not ask again — reset with
   `tccutil reset Camera` and restart the terminal.
2. **The AVFoundation backend.** `cv2.VideoCapture(0)` usually picks it, but
   being explicit avoids a slow probe through every backend:
   `cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)`. The relevant code is
   `src/sentinel/api/app.py` (`/api/source/device` and the demo runner's device
   probe) and `src/sentinel/ingest/source.py`.

---

## F11 (LocateAnything-3B) — the part that needs judgment

`configs/system.yaml` sets `vlm.dtype: bfloat16`. MPS support for bfloat16 is
weaker than for float16. Try, in order:

1. `float16` on MPS.
2. `float32` on MPS — roughly double the memory, and a 3B model in unified
   memory is a real constraint on a 16 GB machine.
3. Leave the VLM on CPU (`vlm.device: cpu`) and accept the latency.

**This feature is allowed to be slow or disabled.** It is licensed for
non-commercial research use only (see `LICENSING.md`), it already has a
`queue_depth: 1` reject-rather-than-queue policy, and every other part of the
system works without it. Do not spend the porting effort here first. If it does
not work, set `vlm.enabled: false` and move on — the query router falls back to
the structured SQL route, which is what answers most questions anyway.

---

## What NOT to change

- **Any threshold in `configs/system.yaml`.** Each one has its measurement
  recorded beside it. They were measured on specific footage, not on specific
  hardware.
- **The demo definitions in `configs/demos.yaml`.** Same reasoning.
- **`runtime.offline: true`.** F15 refuses network access at runtime and sets
  `HF_HUB_OFFLINE` before any model loads. Keep it.
- **The evidence, alerting, spatial and temporal layers.** None of them touch a
  GPU. If a change is needed there, the port has gone wrong somewhere else.

---

## Acceptance criteria

The port is done when all of these pass on the Mac:

```bash
python -m pytest                 # 75 tests, no GPU required — must pass unchanged
python scripts/verify_env.py     # must report MPS, not "cuda unavailable"
python run.py                    # banner must show the device it actually chose
```

and, in the browser at <http://127.0.0.1:8000/demos>:

1. **Demo 1** — the rifle is labelled on its carrier.
2. **Demo 2** — the birds panel draws boxes and no aerial reticle; the drone
   panel draws the reticle.
3. **Demo 3** — health escalates to `TAMPERED` and a `HEALTH_FAULT` alert
   appears in the list, within about ten seconds of the demo starting.
4. **Demo 4** — the herd reads as animal; the crawling subject is boxed
   continuously.
5. **Demo 5** — geometry is reported in metres, the query box answers "when was
   the last alert?" from the structured route, and clicking an alert loads its
   keyframe, subject crop and face.
6. **Demo 6** — the device camera opens after granting permission.

Switching between all six demos repeatedly must not hang. If it does, read the
comment above the stream endpoint in `src/sentinel/api/app.py` — that failure
has a specific cause and it is documented there.

---

## Report back

State the measured frame rate on MPS beside the CUDA figure (17.8 FPS with pose
and both specialised detectors enabled, 56 ms/frame). If it is much slower, say
so plainly rather than tuning thresholds to compensate — the thresholds are
measurements, and moving them to flatter a slower device makes the numbers
untrue rather than making the system faster.
