# Open decisions

Four questions were flagged as blocking before implementation and were never
answered. Rather than stop with nothing delivered, each was resolved with a
stated assumption. **All four are reversible and each is roughly one config
line.** They are recorded here so they do not harden into unexamined defaults.

---

## 1. AGPL-3.0 accepted for the detector and trackers

**Assumed:** yes, AGPL is acceptable for this submission.

Ultralytics YOLO26-s and its BoT-SORT / ByteTrack implementations are
AGPL-3.0. Fielding this in a product requires open-sourcing the derivative work
or an Ultralytics enterprise licence.

**To reverse:** write one adapter implementing the `Detector` protocol in
`src/sentinel/perception/detector.py` — RF-DETR (Apache-2.0) is the intended
permissive target. Nothing downstream of the adapter changes. The trackers
would need hand-wiring, since `boxmot` is also AGPL-3.0.

**Why it matters:** for a system whose deliverable is deployed by a security
agency, *"can this actually be fielded?"* is a fair evaluator question. Having
the analysis is the differentiator; almost no competing team will have
considered it. See `LICENSING.md`.

---

## 2. F11 stays on LocateAnything-3B

**Assumed:** confirmed by explicit instruction, twice, after the licence
finding was raised.

LocateAnything-3B is released under the NVIDIA Open Model License for
**non-commercial use — academic and non-profit research only**. Its base
language model (Qwen2.5-3B-Instruct) additionally carries the Qwen Research
licence. This is the same class of restriction that rules out the InsightFace
face-recognition packs elsewhere in this project.

It works and is measured: 9.84 GB peak, 620 ms – 1.5 s per query.
**It cannot be deployed commercially as-is.**

**To reverse:** Qwen3-VL (Apache-2.0) behind the same `VisualGroundingRoute`
interface. The licence string is surfaced in every API response and shown in
the interface next to every grounding result, so it is disclosed rather than
buried.

---

## 3. Fixed cameras only

**Assumed:** every camera is fixed.

A PTZ camera invalidates its homography on every pan and tilt, so F3, F4 and F5
would silently report wrong metres. This is asserted in config as
`homography.fixed_camera: true` rather than assumed silently.

**To reverse:** PTZ support needs a preset-linked calibration table — one
homography per preset, selected by the camera's reported position. That is a
materially different design and is not built.

---

## 4. The threat matrix is a proposal

**Assumed:** the mapping in `configs/system.yaml` under `alerts.threat_matrix`.

This is **the core semantic of the entire system** — which observation, against
which identity verdict, produces which colour. It lives in config precisely so
that people who know the operational domain can argue with it.

Currently proposed:

| Observation | FRIENDLY | UNIDENTIFIED | INSUFFICIENT_QUALITY | no identity |
|---|---|---|---|---|
| `TRIPWIRE_CROSS` | FRIENDLY | **CRITICAL** | WARNING | WARNING |
| `ZONE_PRESENCE` | FRIENDLY | WARNING | WATCH | WATCH |
| `PROXIMITY` | INFO | WARNING | WATCH | WATCH |
| `BAG_UNATTENDED` | — | — | — | WARNING |
| `STANCE_ANOMALY` | WATCH | WARNING | WATCH | WATCH |
| `HEALTH_FAULT` | — | — | — | WARNING |

**This is the item most wanting review.** It is a domain judgment, not an
engineering one.

---

## Also unanswered

**The internal hackathon has no date.** Every cut line in the build order
depends on it. Phase 1 is complete, so this is no longer blocking, but it
determines how much Phase 2 is realistic.
