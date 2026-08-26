# Setting up SENTINEL on a new machine

From a fresh clone to a running demo. Tested end to end on the reference
machine (RTX 5080 / Windows 11); the Apple Silicon path is
[not yet implemented](docs/HANDOFF-apple-silicon.md).

---

## TL;DR

```bash
git clone https://github.com/varadharajanv0310/Border-Surveillance-using-CCTV-SIH-2026.git
cd Border-Surveillance-using-CCTV-SIH-2026

# 1. PyTorch first, matched to your GPU (see §2 -- this is the step that goes wrong)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 2. everything else
pip install -r requirements.txt

# 3. weights + demo gallery + tests + environment gate, one command
python scripts/bootstrap.py --all

# 4. run
python run.py            # -> http://127.0.0.1:8000
```

---

## 1. What is and is not in this repository

Three things are deliberately **not** committed, and `scripts/bootstrap.py`
recreates all three:

| Not in git | Size | How it comes back | Why not committed |
|---|---|---|---|
| `weights/` | 20 MB – 7.8 GB | `python scripts/fetch_weights.py --all` | Far past what git should carry; also lets F15's offline claim stay honest — one script fetches, nothing else ever does |
| `data/gallery/` | ~200 KB | `python tools/enroll.py …` | Derived from the committed clip, so regenerating is more correct than storing |
| `data/db/`, `data/evidence/`, `data/clips/` | grows | created on first run | Runtime output |

Everything else — source, config, tests, tools, and the 341 KB demo clip — is
in the repository. A clone plus `bootstrap.py` is a complete system.

---

## 2. PyTorch: install this first, and separately

**This is the step that goes wrong.** `requirements.txt` deliberately does not
pin torch, because the correct build depends on your hardware. Install it
*before* the rest, or pip will pull a CPU-only wheel and the detector will run
at about two frames per second.

**NVIDIA Blackwell (RTX 50-series, sm_120)** — the reference machine:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Requires CUDA 12.8+ and PyTorch ≥ 2.7.0 — the first release with native sm_120
support. Older toolkits fail at the first kernel launch with *"no kernel image
available for execution"*. Verified working: torch 2.11.0+cu128, torchvision
0.26.0+cu128.

**Older NVIDIA (30/40-series)** — use the cu121 or cu124 index instead.

**Apple Silicon** — see [`docs/HANDOFF-apple-silicon.md`](docs/HANDOFF-apple-silicon.md).
The code currently has no MPS path; it will fall back to CPU and run, slowly.

**CPU only** — works, at roughly 2–4 FPS. Set `runtime.device: cpu` in
`configs/system.yaml` and expect to demo from pre-rendered video.

### Python version

3.11 or 3.12 are recommended. 3.13 works (it is what the reference machine
runs) but has narrower wheel coverage for this stack; the environment gate
warns about it rather than failing.

---

## 3. Bootstrap

```bash
python scripts/bootstrap.py          # detector + re-ID (~30 MB)
python scripts/bootstrap.py --all    # + pose (F16) + LocateAnything-3B (F11, 7.8 GB)
```

It runs four steps and tells you which one failed:

1. **Weights** — YOLO26-s, OSNet, optionally YOLO26-Pose and LocateAnything-3B.
2. **Gallery** — enrols `patrol_01` from the committed clip so F8 has something
   to match against. Without it every subject reports *Unidentified*, which is
   correct behaviour but a less interesting demo.
3. **Tests** — 48 tests, no GPU needed. If these fail the checkout is broken.
4. **Environment gate** — `scripts/verify_env.py`. Do not skip it.

A healthy gate on the reference machine ends:

```
21 checks: 0 failed, 2 warnings
```

The two expected warnings are Python 3.13 and the absent CUDA toolkit (which
only matters for runtime CUDA compilation — see §6).

---

## 4. Run

```bash
python run.py                        # demo profile
python run.py --profile operational  # 120 s baggage dwell, stricter confirmation
python run.py --tracker bytetrack    # F2: swap the tracker
python run.py --pose                 # F16: enable stance detection
python run.py --sahi                 # sliced inference for distant targets
python run.py --source rtsp://user:pass@host/stream   # live camera
python run.py --check                # environment gate only, no server
```

Open <http://127.0.0.1:8000>.

### The four demonstrations

`docs/DEMOS.md` walks through each one — video tampering, weapon detection,
drone detection, and animal versus crawling human — with what to watch for and
what to say about it. The demo profile enables the pose model and both
specialised detectors; `configs/system.yaml` leaves all three off, because each
costs runtime and each is a claim the system should not make by default.

The two specialised detectors need weights that are **committed to this repo**
(`weights/weapons_yolo26n.pt`, `weights/drones_yolo26n.pt`) rather than fetched
by `scripts/fetch_weights.py` — they are the output of a training run against
specific datasets, not something upstream publishes. If either is missing the
system fails at startup with a message naming the file, which is deliberate: a
demo where the detector silently never fires is worse than one that refuses to
start. To retrain either:

```bash
python scripts/fetch_datasets.py --task weapons   # or drones
python tools/train_detector.py --task weapons --epochs 60
```

### Pointing it at your own camera

1. Copy `configs/cameras/cam01.yaml` to `cam02.yaml`, change `camera_id` and
   `source`.
2. Calibrate it: `python tools/calibrate.py --camera cam02` — click four ground
   points, enter their real-world coordinates, paste the emitted block back.
   Without this the camera runs in pixel space and every distance is flagged
   `degraded` rather than reported as metres.
3. Redraw the tripwire and zones in the same coordinates.

---

## 5. Verifying it actually works

```bash
python -m pytest                     # 70 tests, ~1.5 s, no GPU
python scripts/verify_env.py         # hardware + weights + config gate
python tools/measure_threshold.py    # re-fit the F8 threshold on your footage
python tools/render_demo.py --stills 3   # annotated stills into out/
```

`measure_threshold.py` is worth re-running on any new footage. The shipped
value (0.666) was fitted to the bundled clip; a different camera will have a
different separation, and the tool prints the distributions so you can see
whether the classes separate at all.

---

## 6. Known environment traps

**PTX just-in-time compilation is unavailable on Blackwell.** Ahead-of-time
builds work; anything compiling CUDA at runtime — FlashAttention, custom
`cpp_extension` builds — does not. LocateAnything-3B hits this path and falls
back from FlashAttention to SDPA on its own, which is why F11 still runs at
9.84 GB. The environment gate reports this as a warning, not a failure, and
names the distinction.

**OpenCV's H.264 writer may be broken** (`libopenh264: Incorrect library
version loaded`). Evidence clips are written by piping to `ffmpeg` for exactly
this reason, with an OpenCV fallback. Install ffmpeg and put it on `PATH`:
`winget install Gyan.FFmpeg` on Windows, `brew install ffmpeg` on macOS,
`apt install ffmpeg` on Debian.

**F11 needs three packages the model card names and pip does not resolve on its
own** — `transformers==4.57.1` (its remote code imports `qwen3` classes absent
before 4.57), plus `peft` and `lmdb`. All three are pinned in
`requirements.txt`. `decord` is deliberately *not* installed: it has no wheels
for current Python, and since SENTINEL only ever passes single images, a stub
is registered in `sys.modules` scoped to the loader (`query/visual_route.py`).
Do **not** solve this by creating a `decord.py` file in the project root — that
shadows the real package for every import in the project.

**First F11 call takes ~30 s.** The model loads lazily on first use, not at
startup, so the system is up and alerting immediately. Subsequent calls are
620 ms – 1.5 s.

---

## 7. Resuming work on this project

Read in this order:

1. `README.md` — what the system is, what is measured, and the honest limitations.
2. `configs/system.yaml` — every threshold in the system, with the reasoning in
   comments. Nothing in the code carries a magic number.
3. `src/sentinel/contracts/` — the data shapes everything else agrees on. This
   package imports nothing from the rest of the tree.
4. `tests/` — the suppression policy (F7) and geometry (F3/F4/F5) are specified
   here more precisely than in prose.

**Four decisions were made under assumption** because the questions were never
answered. Each is one config line to flip, and each is listed in
`docs/OPEN-DECISIONS.md`. The threat matrix in particular is a *proposal*, and
it is the core semantic of the whole system.

**Licensing is unresolved.** See `LICENSING.md` before adding a `LICENSE` file
— this repository is public and has an AGPL-3.0 dependency.
