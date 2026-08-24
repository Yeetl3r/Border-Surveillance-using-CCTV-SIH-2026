# SENTINEL

**AI-based intelligent video analytics for border surveillance on existing CCTV infrastructure.**
Smart India Hackathon 2026 · Problem Statement **SIH26187** · Category: Software · Theme: Smart Automation

---

## What this is

A live video analytics system that turns legacy CCTV into an alerting perimeter.
One command starts it; it ingests a video source, tracks objects in real-world
metres, applies temporal logic, and raises alerts that carry their own evidence
— entirely offline, on one machine.

```bash
python run.py
# C2 interface -> http://127.0.0.1:8000
```

The defining constraint is in the problem statement itself: *existing* CCTV.
That means legacy analogue-to-IP conversions, 720p or worse, aggressive H.264
with visible blocking, smeared low-contrast night footage, and lenses nobody
has cleaned in years. **Every architectural decision here is made for degraded
input**, and the bundled demo clip is genuinely degraded — 1280×720, 15 fps,
~97 kbps monochrome IR. A system demonstrated on clean video has not addressed
the stated problem.

### The design principle underneath everything

The domain's documented failure mode is not missed detections — it is false
alarms. SBInet produced apprehensions on roughly 2% of sensor alerts. CIBMS
assessments name false-positive suppression as the hardest engineering problem.
An operator who has learned to ignore the panel has converted a working
detector into an expensive screensaver.

So the system's real product is **operator trust**, and it is built to not
spend it:

- Nothing becomes an alert without temporal confirmation (F7).
- Threat and confidence are independent axes, never one score (F12).
- Every alert carries the evidence it was raised on (F13).
- Identity is open-set: the system reports *Unidentified*, never *hostile* (F8).
- Anything simulated is named and labelled as simulated.

---

## Phase 1 feature status

All sixteen are implemented and exercised on the bundled footage.

| ID | Capability | Status | Where |
|----|-----------|--------|-------|
| **F1** | Multi-class detection — person / vehicle / animal / bag | ✅ | `perception/yolo_detector.py` |
| **F2** | Persistent tracking, ByteTrack **and** BoT-SORT, config-selectable | ✅ | `perception/tracker.py` |
| **F3** | Homography → real-world metric coordinates | ✅ | `calibration/homography.py` |
| **F4** | Directional tripwire + geofence breach | ✅ | `calibration/zones.py`, `spatial/analyzer.py` |
| **F5** | Metric proximity alert (metres, not pixels) | ✅ | `spatial/analyzer.py` |
| **F6** | Unattended baggage — association + dwell state machine | ✅ | `temporal/association.py` |
| **F7** | Alert confirmation before anything fires | ✅ | `temporal/confirm.py` |
| **F8** | Open-set friendly vs unidentified (OSNet) | ✅ | `identity/` |
| **F9** | Anti-tamper / camera health (Laplacian + SSIM) | ✅ | `ingest/health.py` |
| **F10** | Structured query route (SQL over the alert log) | ✅ | `query/sql_route.py` |
| **F11** | Forensic visual grounding (LocateAnything-3B) | ✅ | `query/visual_route.py` |
| **F12** | Dual-axis alert rendering | ✅ | `ui/static/app.js` |
| **F13** | Evidence store | ✅ | `alerting/evidence.py` |
| **F14** | Sub-3 KB tactical relay | ✅ | `alerting/relay.py` |
| **F15** | Offline / air-gapped operation | ✅ | `runtime.offline`, `scripts/fetch_weights.py` |
| **F16** | Infiltration stance (crawl / crouch / prone) | ✅ *opt-in* | `perception/pose.py` |

**Zero custom model training.** Every model is used pretrained. All effort went
into the logic layer, the state machine and the interface.

---

## Measured numbers

Every figure below was measured on this machine. Nothing is carried over from a
spec sheet, and nothing unmeasured is quoted. Hardware: **RTX 5080 (Blackwell,
sm_120), 16 GB**, PyTorch 2.11.0+cu128, Windows 11.

| Measurement | Value | How |
|---|---|---|
| End-to-end pipeline | **33–39 FPS** | `/api/status` over a 500-frame run, 1280×720, imgsz 960 |
| Detector + tracker only | **37.4 FPS** | standalone loop over the 178-frame clip |
| Detect stage | ~13 ms/frame | rolling mean, `Perf` |
| Track stage | ~9.5 ms/frame | rolling mean, `Perf` |
| Tier 2/3 logic stage | **0.14 ms/frame** | rolling mean — geometry and state machine are free next to inference |
| Homography residual | **0.0000 m** | round-trip over the 4 calibration points |
| SQL query route (F10) | **0.06–0.55 ms** | `SqlRoute.run`, measured per call |
| Grounding route (F11) | **620 ms – 1.54 s** | LocateAnything-3B, 1280×720 keyframe, batch 1 |
| F11 model load | 31.5 s | cold, from local weights |
| F11 peak VRAM | **9.84 GB** | `torch.cuda.max_memory_reserved` — fits 16 GB |
| Relay payload (F14) | **max 2951 B, mean 2871 B** | 9 alerts, budget 3072 B |
| OSNet | 2.31 M params, 512-d | matches the published x0_25 figure |
| Test suite | **48 passed in 1.29 s** | no GPU, no video, no model weights |

### F8 threshold — fitted, not guessed

OSNet's `fc` head ends in ReLU, so embeddings live in the positive orthant and
cosine similarities between *unrelated* crops sit far above zero. A threshold
picked from intuition about "cosine similarity" lands near 0.5 and matches
everybody. `tools/measure_threshold.py` on the bundled clip:

```
same subject, different frames    min 0.7068   p5 0.7752   median 0.8941   (n=780)
negative proxy (person vs non-person)          p95 0.5564   max 0.5711     (n=2400)
-> identity.sim_threshold = 0.666
```

**Caveat, stated because it matters:** the negative distribution that counts
operationally is person-versus-*different*-person. Measuring it needs footage
with more than one subject, and the demo clip has one. That number does not
exist here and has not been estimated. Re-run with multi-subject footage before
treating the threshold as validated.

---

## Architecture

```
[ RTSP / file source ]
          |
 TIER 0  INGEST & HEALTH ......... decode · Laplacian + SSIM anti-tamper (F9)
          |                        CLAHE/DCP -> OPERATOR STREAM ONLY
          |                        homography H per camera (F3)
          v
 TIER 1  PERCEPTION .............. YOLO26-s detect (F1) · BoT-SORT/ByteTrack (F2)
          |                        pose on person crops (F16)
          |                        every detection carries {conf, model_id}
          |                        track trajectories RETAINED (Phase 2 hook)
     +----+----+
     v         v
 TIER 2     TIER 3 ............... tripwire + direction (F4) | dwell timers (F6)
 SPATIAL    TEMPORAL              metric proximity (F5)     | confirmation (F7)
     +----+----+
          v
 TIER 5  IDENTITY (open-set) ..... OSNet on quality-gated crops (F8)
          |                        MATCH -> Friendly | NO MATCH -> Unidentified
          v
 TIER 8  ALERT MANAGER ........... the ONLY constructor of an Alert
          |                        threat (colour) x confidence (style)  (F12)
          |                        evidence store (F13) · <3 KB relay (F14)
          v
 TIER 6/9 API & QUERY ROUTER ..... FastAPI · MJPEG + WebSocket
                                   structured -> SQL (F10)
                                   grounding  -> LocateAnything-3B (F11)
```

Bounded queues sit at every stage boundary with an explicit, **counted** drop
policy — visible on the interface, because an invisible drop policy is the same
as no drop policy.

### Repository layout

```
configs/           system.yaml + profiles/{demo,operational} + cameras/*.yaml
src/sentinel/
  contracts/       the shared data shapes. Imports nothing else.
  config/          typed schema + overlay loading
  ingest/          source, health (F9), operator-stream enhancement
  calibration/     homography (F3), zones (F4/F5)
  perception/      detector + tracker interfaces, YOLO adapter, pose (F16)
  spatial/         Tier 2 — pure geometry
  temporal/        Tier 3 — confirmation, association, dwell. No GPU.
  identity/        OSNet, quality gate, friendly gallery (F8)
  alerting/        manager, evidence (F13), relay (F14)
  storage/         SQLite + FTS5 schema and repository
  query/           router, SQL route (F10), grounding route (F11)
  pipeline/        bounded queues, per-camera runner
  api/             FastAPI app
  ui/              the C2 interface
tools/             calibrate · enroll · measure_threshold · degrade · render_demo
scripts/           verify_env · fetch_weights
tests/             47 tests, runnable with no GPU
```

`contracts/` imports nothing from the rest of the tree, and `spatial/` and
`temporal/` import only `contracts/`. That is what makes the suppression policy
— the part that decides whether this system is useful or ignored — testable
without a model, a camera, or a video file.

---

## Getting started

**Full instructions, including the PyTorch/CUDA step that most often goes wrong:
[`SETUP.md`](SETUP.md).**

```bash
# PyTorch first, matched to your GPU -- see SETUP.md §2
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# weights + demo gallery + tests + environment gate, one command
python scripts/bootstrap.py --all

# run
python run.py            # -> http://127.0.0.1:8000
```

`weights/` and `data/` are not in the repository — `bootstrap.py` recreates
both. See [`SETUP.md` §1](SETUP.md) for exactly what is and is not committed.

Useful flags:

```bash
python run.py --profile operational    # 120 s baggage dwell, stricter confirmation
python run.py --tracker bytetrack      # F2: swap the tracker
python run.py --pose                   # F16: enable stance detection
python run.py --sahi                   # sliced inference for distant targets
python run.py --source rtsp://...      # live camera
python run.py --check                  # environment gate only
```

### Operator tools

```bash
python tools/calibrate.py --camera cam01          # F3: click 4 ground points
python tools/enroll.py --name patrol_01 --source assets/sample_border.mp4 \
                       --frames 1-70 --max 10     # F8: enrol a friendly
python tools/measure_threshold.py                 # F8: fit the threshold to data
python tools/degrade.py --input clean.mp4 --output bad.mp4 --preset legacy_cctv
python tools/render_demo.py --stills 3            # annotated stills for slides
python -m pytest                                  # 47 tests, no GPU needed
```

---

## What the demo shows

Running `python run.py` against the bundled clip, in order:

1. **Live tracking with metric readouts** — boxes labelled with track ID, class
   group, confidence, and distance *in metres*.
2. **A directional tripwire crossing** — the subject crosses the north perimeter
   inbound at ~4.6 m and the alert fires after confirmation, not on the frame.
3. **Dual-axis rendering** — colour is threat, style is confidence. The parked
   vehicle detects at 44% and renders as a hatched low-confidence box; the
   subject at 93% renders solid.
4. **Open-set identity** — with `patrol_01` enrolled the same crossing is
   **FRIENDLY / green**; with an empty gallery it is **CRITICAL / red**. Same
   geometry, different verdict, and the system says which and why.
5. **Evidence** — click any alert for the keyframe, subject crop, clip, every
   supporting detection with its own confidence and model id, the triggering
   geometry, and the thresholds in force when it fired.
6. **The query router** — "when was the last alert?" goes to SQL in
   well under a millisecond; "where is the person in this frame?" goes to the
   VLM. Routing it correctly is the point.

---

## Honest limitations

Read this section before demonstrating. Claiming more than this is not
supported by the build.

**The metric scale on the demo camera is estimated, not surveyed.** The
homography geometry is exact (residual 0.0000 m), but the world coordinates
come from a pinhole model fitted to the scene, not four tape-measured ground
marks. The interface and every rendered still label it `SCALE ESTIMATED, NOT
SURVEYED`. On a real deployment `tools/calibrate.py` replaces those numbers.

**Fixed cameras only.** A PTZ camera invalidates the homography on every pan.
Supporting one needs a preset-linked calibration table, which is not built.

**F11's licence prevents commercial deployment.** LocateAnything-3B is released
under the NVIDIA Open Model License for non-commercial use — academic and
non-profit research only — and its base LM (Qwen2.5-3B-Instruct) carries the
Qwen Research licence. This is the same restriction that rules out the
InsightFace packs. It works, it is measured, and it **cannot be fielded
commercially as-is**; Qwen3-VL (Apache-2.0) is the swap, behind the same
interface. The interface displays the licence next to every grounding result.

**Ultralytics models are AGPL-3.0.** Fielding this requires open-sourcing the
derivative work or an Ultralytics enterprise licence. The permissive path is
RF-DETR (Apache-2.0) behind the same `Detector` protocol, plus OSNet (MIT).
Having the analysis is the point; the answer is the team's to give.

**F6's detection precondition is the weakest link.** COCO has no generic *bag*
class — `backpack`, `handbag` and `suitcase` are the closest, and they are
among COCO's weakest classes at CCTV range, which is exactly our operating
condition. The state machine is correct and tested; the detector feeding it is
the risk, and it needs measuring on real footage.

**F16 is off by default.** Stance is inherently high-false-alarm — crouching
versus standing is a continuum at CCTV resolution. Our implementation runs pose
on person *crops* (so the skeleton-to-box association is exact rather than
guessed by nearest-x) and suppresses edge-clipped boxes, because on this very
clip a subject walking toward the camera goes from aspect 0.39 to 0.84 purely
from hitting the frame edge — which a naive aspect-ratio rule reads as
crawling. Enable it knowingly with `--pose`.

**Single camera.** The architecture is per-camera and thread-isolated, but
cross-camera hand-off and a multi-camera common operating picture are Phase 2.

**Python 3.13 is not the recommended runtime.** It works here, but 3.11/3.12
have broader wheel coverage for this stack. The environment gate warns about it.

**Not measured:** false-alarm rate over a long run, person-vs-different-person
re-ID separation, performance on RTSP under packet loss, and any figure on
hardware other than an RTX 5080.

---

## Phase 2 hooks — present, unbuilt

Deliberately left as attachment points, with no implementation:

- **Sensor cues / DAS** — `contracts/cue.py` defines `SensorCue` with a
  **required** `simulated: bool`. A cue cannot be constructed without declaring
  whether it is real, which makes the honesty requirement a property of the type
  system rather than something to remember before a demo. Real DAS needs a
  φ-OTDR interrogator this project does not have and will not get; the
  classification and fusion layers are the buildable half.
- **RGB-thermal fusion** — `FrameBundle.raw` is `(H, W, C)` with `C`
  unconstrained plus an explicit `channels` descriptor, so a four-channel frame
  is a new producer, not a detector rewrite.
- **Drone detection** — `Detector.class_names` is config data, never an enum, and
  `Track.history` retains trajectories that drone-versus-bird motion analysis
  will need.

---

## Provenance

| Component | Source | Licence |
|---|---|---|
| YOLO26-s / YOLO26-Pose | Ultralytics | AGPL-3.0 |
| BoT-SORT / ByteTrack | Ultralytics trackers | AGPL-3.0 |
| OSNet x0_25 (MSMT17) | `kaiyangzhou/osnet` — the author's release | MIT |
| LocateAnything-3B | `nvidia/LocateAnything-3B` | NVIDIA OM — **non-commercial** |
| OpenCV, Shapely, FastAPI, SQLite | — | Apache-2.0 / BSD / MIT / public domain |
| `assets/sample_border.mp4` | test asset carried over from the team's reference prototype | — |

The OSNet architecture in `identity/osnet.py` is implemented directly rather
than imported: the PyPI `torchreid` is an abandoned 2019 build and the
maintained `deep-person-reid` is a source install with pins that do not resolve
on current Python. For a system whose point is running air-gapped on one
machine, vendoring ~150 lines of documented architecture beats inheriting a
dependency tree that may not install on the demo box. The weights are the
author's own MIT release. Nothing is trained here.

---

## Further reading

| Document | What it covers |
|---|---|
| [`SETUP.md`](SETUP.md) | Fresh-machine setup, environment traps, resuming work |
| [`docs/OPEN-DECISIONS.md`](docs/OPEN-DECISIONS.md) | Four assumptions made without an answer; each one config line to flip |
| [`LICENSING.md`](LICENSING.md) | Why there is no `LICENSE` file yet, and the three defensible options |
| [`docs/HANDOFF-apple-silicon.md`](docs/HANDOFF-apple-silicon.md) | Ready-to-use prompt for adding MPS support on a Mac |

---

## Standing rules for this codebase

- Never fabricate a performance number. If it is not measured, it is labelled
  unmeasured.
- Never claim a capability that is not implemented.
- Never present a simulated capability as real.
- Prefer the simple version that works on bad footage over the sophisticated
  version that works on good footage.
- When in doubt, suppress. A missed low-confidence event costs less than a
  spurious alert.
