# Licensing — read before adding a LICENSE file

**This repository is public and has an AGPL-3.0 dependency. It deliberately
ships without a `LICENSE` file, because choosing one is a decision with legal
weight that belongs to the team, not to whoever wrote the code.**

## What is actually in here

| Component | Licence | Consequence |
|---|---|---|
| **Ultralytics** (YOLO26-s, YOLO26-Pose, BoT-SORT, ByteTrack) | **AGPL-3.0** | Strong copyleft. Distributing or network-serving a derivative work obliges you to offer the complete corresponding source under AGPL-3.0 — or to hold an Ultralytics enterprise licence. |
| **LocateAnything-3B** (F11) | NVIDIA Open Model License | **Non-commercial**: academic and non-profit research only. Weights are not redistributed here; `scripts/fetch_weights.py` downloads them and prints the licence. |
| Qwen2.5-3B-Instruct (F11's base LM) | Qwen Research License | Research use. |
| **OSNet x0_25** weights | MIT | Fine. Not redistributed; fetched from the author's release. |
| OpenCV, Shapely, FastAPI, NumPy, pydantic | Apache-2.0 / BSD / MIT | Permissive. Fine. |
| SQLite | Public domain | Fine. |
| Our own code (`src/`, `tools/`, `scripts/`, `tests/`) | **unset** | The decision below. |

## The actual problem

Publishing this repository publicly is, in itself, consistent with AGPL-3.0 —
the source is available. The trap is **labelling it MIT**. Our code imports
Ultralytics at runtime; an MIT header on a work that combines with AGPL-3.0
code misstates the terms downstream users actually receive. The reference
prototype this project superseded carried exactly that mismatch.

## Three defensible options

1. **License the project AGPL-3.0.** Simplest and internally consistent with
   the dependency already in use. Costs nothing for a hackathon submission or
   an academic deliverable. **This is the default recommendation.**

2. **Go fully permissive by removing the AGPL dependency.** Swap the detector
   for RF-DETR (Apache-2.0) behind the existing `Detector` protocol and
   hand-wire the trackers, then MIT or Apache-2.0 becomes honest. This is the
   path if the goal is something a government agency could actually field.
   Note that F11 would still need replacing for commercial use.

3. **Keep Ultralytics and state plainly that production deployment requires an
   Ultralytics enterprise licence.** Also defensible, and an honest answer to
   an evaluator who asks.

## What is safe to say on stage

> "Our detector is AGPL-3.0, so fielding this needs either an enterprise
> licence or an open-source release. The permissive swap is RF-DETR behind the
> same interface — it is one adapter. Our grounding model is NVIDIA
> non-commercial, so that one would be replaced with Qwen3-VL for deployment.
> We know which components carry which obligation."

Do not claim the system is deployable as-is. It is not, and the licences are
checkable in about thirty seconds.
