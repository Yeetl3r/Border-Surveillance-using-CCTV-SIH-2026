# Handoff: Apple Silicon (MPS) support

The system currently has **no MPS path**. On a Mac it detects no CUDA device,
falls through to CPU, and runs at roughly 2–4 FPS — functional, but not
demonstrable.

The good news is that device selection is isolated to **six sites**, because
detection, tracking, embedding and grounding all sit behind interfaces. This is
a contained change, not a refactor.

Below is a ready-to-use prompt. Copy everything between the rules into a fresh
Claude Code session **on the Mac**, in the cloned repository.

---

````
You are adding Apple Silicon (MPS) support to SENTINEL, an AI border-surveillance
video analytics system built for SIH 2026 (problem statement SIH26187). The repo
is already complete and working on NVIDIA CUDA; your job is to make it run well on
Apple Silicon without breaking the CUDA path.

## Read these first, in this order

1. `README.md` — what the system is, what is measured, honest limitations
2. `SETUP.md` — how it installs and what is deliberately not in git
3. `docs/OPEN-DECISIONS.md` — four assumptions that are still open
4. `configs/system.yaml` — every threshold; nothing in code carries a magic number

Two rules this codebase is built around, which you must preserve:

- **Never fabricate a performance number.** If you did not measure it, say so.
  The README's numbers table is measured on an RTX 5080 and must stay labelled
  that way; add your Mac numbers as a separate, separately-labelled column or
  table. Do not overwrite the CUDA figures.
- **Never claim a capability that does not work.** If F11 will not fit in your
  Mac's memory, the correct outcome is that it reports itself unavailable with a
  clear reason — not that it silently thrashes.

## Environment first — do not skip this

```bash
python scripts/verify_env.py
```

It will fail or warn loudly on a Mac because it is written against CUDA. That is
expected and is part of what you are fixing.

Install PyTorch with MPS support (2.7+ recommended, matching what the project
already expects):

```bash
pip install torch torchvision            # the default macOS wheel has MPS
pip install -r requirements.txt
brew install ffmpeg                      # evidence clips are written via ffmpeg
python scripts/bootstrap.py              # weights + demo gallery + tests + gate
```

## The six sites that need changing

Device selection is deliberately isolated. Change these and nothing else needs to
know:

1. **`src/sentinel/config/schema.py`**
   - `RuntimeCfg.device` (~line 26): `Literal["auto", "cuda", "cpu"]` → add `"mps"`
   - `VlmCfg.device` (~line 304): same
   - `VlmCfg.dtype` (~line 305): default is `"bfloat16"`. **MPS support for
     bfloat16 is incomplete** — on MPS this must resolve to `float16`. Do not
     change the default for CUDA; resolve it at load time based on the device.
   - `DetectorCfg.half` (~line 59): fp16 is native on MPS, so this can stay true,
     but verify it does not produce NaNs on your chip.

2. **`src/sentinel/perception/yolo_detector.py`** — `_pick_device()` (~line 167).
   Currently returns `0` (a CUDA index) or `"cpu"`. Add an MPS branch returning
   the string `"mps"`. Ultralytics accepts `device="mps"` natively.

3. **`src/sentinel/perception/pose.py`** — `_cuda()` (~line 179) and its use at
   line 65 (`self.device = 0 if _cuda() else "cpu"`). Same treatment. Consider
   renaming the helper to `_accelerator()` since it is no longer CUDA-specific.

4. **`src/sentinel/identity/embedder.py`** (~lines 91–92). Hardcoded
   `"cuda" if torch.cuda.is_available() else "cpu"`. OSNet is plain PyTorch and
   should run on MPS, but **benchmark it** — depthwise separable convolutions
   (which OSNet uses heavily, four parallel streams per block) have historically
   been slow on MPS. If it is slower than CPU on your chip, that is a real
   finding: record it and make the device configurable rather than forcing MPS.

5. **`src/sentinel/query/visual_route.py`** (~lines 154–167). The dtype map and
   `dev = "cuda" if torch.cuda.is_available() else "cpu"`. See the F11 section
   below — this one needs judgment, not just a branch.

6. **`scripts/verify_env.py`** (~lines 55–100). The entire GPU block is CUDA-only:
   `get_device_name`, `get_device_capability`, `get_arch_list`, `mem_get_info`,
   `is_bf16_supported`, `empty_cache`, and a `device="cuda"` matmul. Write a
   parallel MPS branch. Useful MPS equivalents:
   - `torch.backends.mps.is_available()` / `torch.backends.mps.is_built()`
   - `torch.mps.current_allocated_memory()` / `torch.mps.driver_allocated_memory()`
   - `torch.mps.empty_cache()`
   - There is **no** MPS equivalent of `mem_get_info()`. Apple Silicon uses
     unified memory; report total system RAM (via `psutil` or
     `sysctl hw.memsize`) and say plainly that it is shared with the CPU.
   - There is no `is_bf16_supported()` for MPS. Report bfloat16 as unsupported.
   - Keep the real kernel-launch benchmark — it is the check that actually
     matters — just run it on `device="mps"`.

Suggested shape: add one helper, e.g. `sentinel/runtime/device.py`, exposing
`pick_device(pref)` and `describe_device()`, and have all six sites call it.
That is cleaner than six independent branches and makes a future ROCm or
DirectML path a single edit.

## Set this environment variable

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

Some operations are still unimplemented on MPS and will otherwise raise
`NotImplementedError` mid-run. With the fallback they silently execute on CPU.
Set it in code (`os.environ.setdefault`) near where the offline flags are set in
`SentinelSystem.__init__`, so it applies before any model loads — but **log
when a fallback actually happens** if you can detect it, because a silent CPU
fallback inside the detection loop is a performance cliff that will otherwise
look like an unexplained FPS drop.

## F11 (LocateAnything-3B) — the part that needs judgment

Measured on the reference machine: **9.84 GB peak, 7.8 GB of weights**, loads in
31.5 s, 620 ms – 1.5 s per query.

On Apple Silicon that memory is **unified** — shared with the OS, the browser,
and everything else. Realistically:

- **≥ 32 GB Mac**: should work. Verify and measure.
- **24 GB**: probably works with nothing else running. Measure before claiming it.
- **16 GB**: expect heavy swapping or an outright failure. **Do not force it.**
  The correct behaviour is that `VisualGroundingRoute.ensure_loaded()` fails
  cleanly with a reason, the interface shows F11 as unavailable, and the SQL
  route (F10) carries the query demonstration — which it already does, and which
  the code already handles as a first-class state.

Also note the model's remote code checks for `flash_attn` and `magi_attention`
and falls back to SDPA when absent. On CUDA that fallback is what makes it work
at all (PTX JIT is broken on Blackwell). On MPS the same fallback path should be
taken; confirm it is, and that SDPA is MPS-accelerated rather than falling back
to CPU.

Add a `dtype` resolution step: `bfloat16` → `float16` on MPS.

## What NOT to change

- **Do not touch the tier structure, contracts, or any threshold.** This is a
  device-support change.
- **Do not modify `src/sentinel/contracts/`.** It has zero internal dependencies
  by design and nothing about it is device-specific.
- **Do not "fix" the CLAHE/DCP enhancement being outside the detection path.**
  That is a deliberate research-backed decision documented in
  `ingest/enhance.py`. The detector sees raw frames; only the operator sees
  enhanced ones.
- **Do not create a `decord.py` in the project root** to satisfy F11's import
  check. There is a scoped `sys.modules` stub in `query/visual_route.py` that
  does this correctly without shadowing a real install for the whole project.
- **Do not lower the F8 similarity threshold to make matches happen.** It was
  fitted to measured distributions (`tools/measure_threshold.py`). If it behaves
  differently on your data, re-run that tool and report the new distributions.

## Acceptance criteria

Do not report success until all of these pass, and report each with a number:

1. `python scripts/verify_env.py` → **0 failures**, and it correctly identifies
   the Apple Silicon chip, unified memory, and MPS availability.
2. `python -m pytest` → **48 passed** (they are CPU-only and must be unaffected).
3. `python run.py` → interface live at <http://127.0.0.1:8000>, with a measured
   FPS on the status endpoint. **State the chip** (M1/M2/M3/M4, Pro/Max/Ultra).
4. A tripwire alert fires with `direction: INBOUND` and a distance in metres.
5. F8 produces a `FRIENDLY` match against the enrolled `patrol_01` gallery, and
   the same crossing renders `CRITICAL` when the gallery is empty.
6. F9 trips within 1.0 s of a covered lens (there is a test approach in the
   build report; the reference machine managed 0.20 s).
7. F14 relay payloads stay under 3072 bytes.
8. F11 either works — with measured latency and memory — or reports itself
   unavailable with an accurate reason. Both are acceptable outcomes. Guessing
   is not.

## Deliverables

1. The code changes, with the CUDA path still working (do not regress it — if
   you cannot test CUDA, keep the branches strictly additive and say so).
2. A new section in `README.md`: *"Measured on Apple Silicon"*, with your chip
   named, alongside — not replacing — the existing RTX 5080 table.
3. An update to `SETUP.md` §2 replacing the "Apple Silicon — not implemented"
   note with real install instructions.
4. A short note of anything that turned out to be slower or broken on MPS.
   Negative results are useful; a Mac-specific limitation recorded now is worth
   more than a surprise on stage.
````

---

## Why this is a contained change

The interfaces were built with exactly this in mind. `Detector`, `Tracker`,
`Embedder` and the grounding route are Protocols; the pipeline talks to those,
not to torch. The tiers below perception — spatial geometry, the temporal state
machine, the alert manager, storage — have **no GPU dependency at all**, which
is why the 48 tests run in 1.3 seconds without one. None of that changes.

What does change is the handful of places that answer *"which device?"*, and the
environment gate that reports what it found.
