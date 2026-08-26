# SENTINEL — Complete Technical Dossier

**AI-Based Intelligent Video Analytics Platform for Border Surveillance using Existing CCTV Infrastructure**

Smart India Hackathon 2026 · Problem Statement **SIH26187**
Category: **Software** · Theme: **Smart Automation**

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Abstract](#2-abstract)
3. [The Problem](#3-the-problem)
4. [Domain Grounding — Why Previous Systems Failed](#4-domain-grounding--why-previous-systems-failed)
5. [Design Principles](#5-design-principles)
6. [System Architecture](#6-system-architecture)
7. [Tier-by-Tier Specification](#7-tier-by-tier-specification)
8. [Complete Capability Catalogue](#8-complete-capability-catalogue)
9. [Technology Stack — Component by Component](#9-technology-stack--component-by-component)
10. [Core Algorithms in Detail](#10-core-algorithms-in-detail)
11. [Data Contracts](#11-data-contracts)
12. [Datasets](#12-datasets)
13. [The Aerial Threat — Drone Detection](#13-the-aerial-threat--drone-detection)
14. [RGB–Thermal Fusion](#14-rgbthermal-fusion)
15. [Distributed Acoustic Sensing](#15-distributed-acoustic-sensing)
16. [The Query Router](#16-the-query-router)
17. [Evidence, Audit and Provenance](#17-evidence-audit-and-provenance)
18. [Performance and Measurement](#18-performance-and-measurement)
19. [Failure Modes and Resilience](#19-failure-modes-and-resilience)
20. [Deployment Architecture](#20-deployment-architecture)
21. [Security, Privacy and Ethics](#21-security-privacy-and-ethics)
22. [Licensing Analysis](#22-licensing-analysis)
23. [Comparison with Prior Systems](#23-comparison-with-prior-systems)
24. [Repository Structure](#24-repository-structure)
25. [Operational Workflow](#25-operational-workflow)
26. [Scaling Path](#26-scaling-path)
27. [Appendix A — Configuration Reference](#appendix-a--configuration-reference)
28. [Appendix B — API Reference](#appendix-b--api-reference)
29. [Appendix C — Glossary](#appendix-c--glossary)

---

## 1. Executive Summary

SENTINEL converts a nation's existing, ageing CCTV estate into an intelligent
border-surveillance perimeter without replacing a single camera.

The system ingests live RTSP streams from legacy cameras, detects and tracks
people, vehicles, animals, bags and aerial objects, converts every position
into **real-world metres** through per-camera ground-plane calibration, applies
a temporal state machine that suppresses false alarms, decides identity on an
**open-set** basis that never asserts hostility, and raises alerts that each
carry the complete evidence they were raised on — keyframe, video clip, every
model confidence, the triggering geometry, and the exact thresholds in force at
the moment of firing.

Nine processing tiers run behind swappable interfaces on a single workstation,
entirely offline, with no cloud dependency and no external API calls.

**The thesis in one sentence:** in border surveillance the binding constraint
is not detection accuracy but operator trust, and every architectural decision
in SENTINEL is made to protect it.

### Headline characteristics

| Dimension | Value |
|---|---|
| Input | Legacy CCTV — 720p or worse, heavy H.264 compression, IR night footage, unreliable backhaul |
| Throughput | 30–39 FPS per stream, 1280×720, measured on RTX 5080 |
| Latency, spatial + temporal reasoning | 0.14 ms per frame |
| Camera tamper detection | 0.20 s from lens obscuration |
| Structured query response | 0.06–7.5 ms |
| Tactical relay payload | ≤ 3,072 bytes including a JPEG crop |
| Network dependency | None. Fully air-gapped. |
| Custom model training required for the core perimeter stack | Zero |

---

## 2. Abstract

Border security forces operate tens of thousands of CCTV cameras across
international frontiers. The overwhelming majority of these feeds are never
watched. Where they are watched, operator fatigue sets in within twenty
minutes, and the documented failure of every large-scale sensor programme
attempted to date has not been an inability to detect — it has been an
inability to *suppress*. The United States' SBInet programme was cancelled
after achieving apprehensions on approximately 2% of sensor alerts. India's
own CIBMS pilots identified false positives from wildlife, weather and
vegetation as the single hardest engineering problem encountered.

SENTINEL addresses this directly. It is an end-to-end intelligent video
analytics platform architected around a single organising principle: **an alert
that cannot be trusted is worse than no alert at all.**

The platform performs multi-class object detection using NMS-free end-to-end
detection, persistent multi-object tracking with appearance-aware association,
per-camera homographic transformation into metric ground coordinates,
directional tripwire and geofence breach detection, metric proximity alerting,
person–object association with dwell-timer state machines for abandoned
baggage, open-set appearance-based identity resolution, camera tamper and
health monitoring, aerial-object detection with multi-frame flight-behaviour
discrimination, RGB–thermal sensor fusion, acoustic sensor cross-cueing, and
natural-language forensic retrieval across a four-route query router.

Every alert is promoted only after temporal confirmation measured in wall-clock
seconds and hit-ratio, never in raw frame counts. Every alert is rendered on
two independent visual axes — colour encodes threat, style encodes confidence —
so that an operator can distinguish "this is serious" from "I am certain" at a
glance. Every alert is immutable, auditable, and reproducible.

---

## 3. The Problem

### 3.1 Problem statement SIH26187

Develop an AI-based intelligent video analytics platform for border
surveillance that operates on **existing** CCTV infrastructure.

### 3.2 The load-bearing word

The single most consequential word in that statement is **existing**.

Existing does not mean modern. In practice it means:

- Analogue cameras converted to IP through encoder boxes, introducing
  generational quality loss
- Sensors at 720p or below, many considerably below
- Aggressive H.264 compression tuned for bandwidth, not fidelity, producing
  visible macroblocking on any motion
- Infrared night illumination producing monochrome, low-contrast, high-noise
  imagery for twelve hours a day
- Lenses that have not been cleaned since installation — dust, spiderwebs,
  rain streaks, insect nests
- Sun glare, thermal haze, fog, and monsoon rainfall
- Backhaul links that drop frames, stall, and reconnect
- Cameras never calibrated, never surveyed, and mounted wherever a pole existed

A system demonstrated on a clean modern feed has not addressed the problem as
posed. SENTINEL's architecture treats degraded input as the **design centre**,
not the edge case. The reference footage bundled with the platform is
deliberately representative: 1280×720, 15 fps, monochrome infrared, encoded at
approximately 97 kbps — a bitrate at which compression artefacts are visible in
every frame.

### 3.3 What the operator actually needs

Not more detections. A border sector generates thousands of legitimate motion
events per day: livestock, farm workers, wildlife, vegetation movement,
weather. The operator needs the system to answer three questions:

1. **Is something happening?** — detection and tracking
2. **Where, in the real world?** — metric spatial reasoning
3. **Should I act, and how sure are you?** — threat and confidence, separately

---

## 4. Domain Grounding — Why Previous Systems Failed

This section exists because SENTINEL's design is a direct response to
documented failure, not to a hypothetical requirement.

### 4.1 SBInet (United States)

The Secure Border Initiative Network was the direct technological predecessor
of every modern integrated border surveillance programme. It was cancelled as
non-viable. Published assessments record that approximately **2% of sensor
alerts along the US–Mexico border resulted in an apprehension**.

Documented causes of failure:

- False alarms at overwhelming rates
- Line-of-sight constraints not accounted for at design time
- Unreliable information transmission between sensor and command
- Equipment malfunction in field conditions

### 4.2 CIBMS (India)

The Comprehensive Integrated Border Management System encountered the same
wall. From published assessments of the early pilots:

> *"the most difficult engineering problem was not deploying individual sensors
> but achieving seamless software fusion of heterogeneous feeds into one
> reliable picture, and reducing false-positive alerts triggered by wildlife,
> weather, and vegetation."*

Further documented findings from BSF assessments:

- Equipment did not function in adverse climatic conditions
- Significant coverage gaps remained at rivers and *nullahs*
- The system remained manpower-intensive and failed to deliver a **common
  operating picture** at all command levels
- High-technology devices were **not optimally utilised because the required
  technical expertise was not uniformly available among personnel** —
  complexity is itself a failure mode
- High cost and poor spare-parts availability actively deterred field use
- Detection without rapid interception capability produced no operational value

### 4.3 The design consequences

Each documented failure maps to a specific architectural decision in SENTINEL:

| Documented failure | SENTINEL response |
|---|---|
| 2% alert precision | Temporal confirmation before promotion; two-axis threat/confidence; deliberate suppression bias |
| Heterogeneous feed fusion unsolved | Single cue interface; all sensor modalities enter through one typed contract |
| Wildlife/weather false positives | Per-zone object-group filtering; animal classes excluded from perimeter geometry by configuration |
| Equipment failed in adverse conditions | Laplacian + SSIM camera health monitoring; graceful degradation, never collapse |
| No common operating picture | Multi-camera architecture with global entity identity across cameras |
| Expertise not uniformly available | Legibility as a first-class requirement; single-command startup; every threshold in one commented YAML file |
| Poor spare parts, high cost | Runs on existing cameras and one commodity workstation; fibre-based acoustic sensing requires no field electronics |
| Detection without interception | Sub-3 KB tactical relay designed for constrained field links |

### 4.4 The threat has moved

Ground and tower sensor grids do not address the aerial vector, and the aerial
vector is now dominant on the Punjab border. Payload-carrying UAV seizures:

| Year | Seizures |
|---|---|
| 2023 | 107 |
| 2024 | 294 |
| 2025 | 272, plus 367 kg of heroin |

Hotspots: the **Amritsar–Tarn Taran belt** and **Ferozepur**. Typical airframes
are China-manufactured DJI Mavic-series quadcopters carrying 450–550 g
payloads. SENTINEL's aerial detection tier (§13) exists because of this.

---

## 5. Design Principles

These are non-negotiable and every component is auditable against them.

### 5.1 Degraded-input first

Validation occurs on bad footage. A component that works only on clean video
does not work. When two approaches compete, the tiebreaker is which survives
compression artefacts, low light and motion blur.

### 5.2 Every alert carries its evidence

An alert that cannot be audited afterwards is an assertion, not a detection.
"How do you know it's right?" is answered by clicking the alert, not by a
slide.

### 5.3 The system can say "I don't know"

Threat and confidence are two independent axes and are never multiplied into a
single score. Identity is open-set. `INSUFFICIENT_QUALITY` is a distinct
verdict from `UNIDENTIFIED` — "the image could not support a judgement" and
"this person is not enrolled" are different claims, and collapsing them is how
a system begins asserting more than it knows.

### 5.4 Cheapest sufficient path

Each task routes to the least expensive component capable of handling it.
*"When was the last threat found?"* is a SQL query answered in microseconds,
not a vision-language model invocation taking seconds. Routing it correctly is
itself a demonstration of engineering judgement.

### 5.5 Config-driven thresholds

Every timing, distance and confidence threshold lives in a single YAML file
with the reasoning recorded in comments. No magic numbers exist in code.
Demonstration and operational values differ — abandoned-baggage dwell is 15
seconds on stage and 120 seconds in the field — and both are selectable
without editing source.

### 5.6 No fabricated numbers

Performance figures are measured on named hardware and labelled with the
method. Nothing is carried over from a specification sheet.

### 5.7 Simulation is labelled as simulation

Any capability driven by replayed or synthesised sensor data is named and
labelled as such, in the interface and in the code. The acoustic sensing module
is named `das_sim`; the `SensorCue` type carries a **required** `simulated`
boolean with no default, making it impossible to construct a cue without
declaring its provenance.

---

## 6. System Architecture

```
                    ┌─────────────────────────────────────────────┐
                    │   RTSP CCTV STREAMS  ·  n cameras           │
                    │   + IR/thermal pairs  + fibre acoustic line  │
                    └──────────────────────┬──────────────────────┘
                                           │
╔══════════════════════════════════════════▼══════════════════════════════════╗
║ TIER 0 — INGESTION, HEALTH & CALIBRATION                                    ║
║   • Hardware decode (GStreamer / NVDEC), MediaMTX multi-camera aggregation  ║
║   • Anti-tamper: Laplacian variance + SSIM structural drift + freeze detect ║
║       → lens blinding · defocus · occlusion · feed freeze · reposition      ║
║   • Per-camera homography H from 4 operator-marked ground points            ║
║   • Static masks (sky, structures, neighbouring property)                   ║
║   • Zone geometry: tripwires · geofence polygons · proximity rings          ║
║   • Atmospheric correction (CLAHE / Dark Channel Prior)                     ║
║       → OPERATOR STREAM ONLY.  The detector receives raw frames.            ║
╚══════════════════════════════════════════┬══════════════════════════════════╝
                                           │  FrameBundle{raw, view, channels}
╔══════════════════════════════════════════▼══════════════════════════════════╗
║ TIER 1 — REAL-TIME PERCEPTION                        target 25–30 FPS/stream ║
║   • Detect  : YOLO26-s, NMS-free end-to-end, TensorRT INT8                  ║
║               classes: Person · Vehicle · Animal · Bag · Aerial             ║
║               + SAHI sliced inference for targets beyond ~150 m             ║
║   • Pose    : YOLO26-Pose on person crops → 17 COCO keypoints               ║
║   • Track   : BoT-SORT (primary) | ByteTrack | FastTracker                  ║
║   • Every detection emits {conf, model_id, tier, frame_ref}                 ║
║   • Track trajectories retained for multi-frame behavioural analysis        ║
╚═══════┬═══════════════════════┬═══════════════════════┬═════════════════════╝
        │ footpoint             │ track state           │ ROI crop
╔═══════▼═══════════╗  ╔════════▼═══════════╗  ╔════════▼════════════════════╗
║ TIER 2 — SPATIAL  ║  ║ TIER 3 — TEMPORAL  ║  ║ TIER 4 — SPECIALISED ID     ║
║  H·(x_f,y_f)      ║  ║  • dwell timers    ║  ║  (triggered, on crop)       ║
║    → (X_m, Y_m)   ║  ║  • person↔object   ║  ║  • Weapon detection         ║
║  • metric distance║  ║    association     ║  ║  • ANPR: plate → PP-OCRv6   ║
║  • directional    ║  ║  • stance / loiter ║  ║  • Face: YuNet + SFace      ║
║    tripwire cross ║  ║    transitions     ║  ║  • Appearance: OSNet        ║
║  • proximity ring ║  ║  • N-of-M confirm  ║  ║  • Flight-path analysis     ║
║  • zone presence  ║  ║  • alert lifecycle ║  ║    (drone vs bird)          ║
╚═══════┬═══════════╝  ╚════════┬═══════════╝  ╚════════┬════════════════════╝
        └───────────────────────┼───────────────────────┘
                                │
╔═══════════════════════════════▼═════════════════════════════════════════════╗
║ TIER 5 — IDENTITY ORCHESTRATOR                        open-set, cross-camera ║
║   • Quality gate: embed only crops that can support a claim                 ║
║   • Vector collections: [faces] [friendly appearance] [body re-ID]          ║
║   • MATCH     → GREEN  "Friendly Patrol"  + similarity score                ║
║     NO MATCH  → AMBER  "Unidentified"     — never asserts hostility         ║
║     LOW QUALITY → "Insufficient Quality"  — a distinct, honest verdict      ║
║   • Cross-camera handoff: local track_id → global entity_id                 ║
╚═══════════════════════════════┬═════════════════════════════════════════════╝
                                │  bounded queue · drop policy logged & displayed
        ┌───────────────────────┴────────────────────────┐
╔═══════▼═══════════════════════╗  ╔═════════════════════▼═══════════════════╗
║ TIER 6 — ASYNC REASONING      ║  ║ TIER 7 — ARCHIVE INDEX      always-on    ║
║   LocateAnything-3B           ║  ║  • 30 s overlapping clip windows         ║
║   • grounding on keyframes    ║  ║  • Qwen3-VL / CLIP embedding             ║
║   • ambiguity verification    ║  ║  • Milvus (cam_id, t_start, t_end, vec)  ║
║   • on breach or operator     ║  ║  • makes ALL footage searchable,         ║
║     query ONLY — never        ║  ║    not only alerted footage              ║
║     per-frame                 ║  ║                                          ║
╚═══════┬═══════════════════════╝  ╚═════════════════════┬═══════════════════╝
        └───────────────────────┬────────────────────────┘
╔═══════════════════════════════▼═════════════════════════════════════════════╗
║ TIER 8 — ALERT MANAGER, EVIDENCE STORE & TACTICAL C2                        ║
║                                                                             ║
║   THE ONLY COMPONENT PERMITTED TO CONSTRUCT AN ALERT                        ║
║                                                                             ║
║   TWO INDEPENDENT AXES                                                      ║
║     THREAT     : INFO · FRIENDLY · WATCH · WARNING · CRITICAL   → COLOUR    ║
║     CONFIDENCE : HIGH · MEDIUM · LOW                            → STYLE     ║
║                                        (solid · dashed · hatched)           ║
║                                                                             ║
║   LIFECYCLE : PENDING → ACTIVE → CLEARED | EXPIRED                          ║
║               An alert that cannot clear is a defect, not a safety margin.  ║
║                                                                             ║
║   EVIDENCE (per alert, immutable):                                          ║
║     keyframe · crop · clip · every confidence · every model_id ·            ║
║     triggering geometry · homography id · threshold snapshot                ║
║                                                                             ║
║   RELAY : ≤3 KB JSON + compressed crop    STORE : PostgreSQL + Milvus       ║
╚═══════════════════════════════▲═════════════════════════════════════════════╝
                                │
╔═══════════════════════════════┴═════════════════════════════════════════════╗
║ TIER 9 — MULTI-MODAL QUERY ROUTER                                           ║
║                                                                             ║
║  [1] STRUCTURED    "ANPR logs after 22:00"          → SQL         <5 ms     ║
║                    "when was the last threat"                               ║
║  [2] IDENTITY      "track person #412"              → Vector      <20 ms    ║
║  [3] ARCHIVE       "red tractor last Tuesday night" → Clip index  ~100 ms   ║
║  [4] GROUNDING     "where in this frame"            → VLM         async     ║
╚═════════════════════════════════════════════════════════════════════════════╝
```

### 6.1 Cross-cutting: bounded queues

Every stage boundary crossing a speed mismatch is a bounded queue with an
explicit, **counted**, displayed drop policy. Tier 1 runs at video rate; Tier 6
takes seconds. Unbounded queues between them are the single most likely cause
of a live system failure under load.

| Queue | Producer → Consumer | Depth | Policy on saturation |
|---|---|---|---|
| Q1 decode | Ingest → Perception | 2–4 | **Drop oldest.** A stale frame is worthless; latency beats completeness. |
| Q2 crops | Perception → Identity | 16 | **Drop newest.** Re-identification is opportunistic and retried next frame. |
| Q3 alerts | Alerting → API/WS | 256 | **Drop lowest severity. Never CRITICAL.** |
| Q4 evidence | Alerting → Disk | 64 | **Block.** Evidence loss is unacceptable; backpressure is correct. |
| Q5 VLM | Query → Grounding worker | 1 | **Reject with HTTP 503.** A multi-second GPU job is never queued. |

An invisible drop policy is equivalent to no drop policy. Drop counters are
surfaced on the operator interface.

---

## 7. Tier-by-Tier Specification

### Tier 0 — Ingestion, Health and Calibration

**Responsibility:** turn a network stream into a validated, calibrated frame
carrying its own quality assessment.

**Stream ingestion.** MediaMTX aggregates multi-camera RTSP; GStreamer decodes
with NVDEC hardware acceleration directly into the inference pipeline,
eliminating a CPU copy. OpenCV provides the fallback path for file sources and
single-camera deployments. Reconnection with configurable backoff handles the
dropped-link condition that CIBMS assessments identify as endemic.

**Anti-tamper and camera health.** Three independent signals, no model
required:

1. **Laplacian variance** — the variance of the second spatial derivative,
   measuring high-frequency energy. Collapses under lens obscuration, spray
   paint, defocus, and heavy fog.
2. **Structural Similarity (SSIM)** against a slowly-updated rolling reference
   frame. Collapses when a camera is repositioned or the field of view is
   occluded, even when the image remains sharp. The reference updates via
   exponential moving average with a deliberately small coefficient — a fast
   EMA would absorb a lens cover within seconds and then report health,
   allowing the tamper to erase its own evidence.
3. **Frame-identity detection** for frozen or looped feeds. Requires
   bit-identical consecutive frames rather than merely similar ones: a static
   night scene at low bitrate produces frames that differ by a fraction of one
   grey level, and a similarity-based threshold reports a healthy camera as a
   dead one.

States: `OK` · `DEGRADED` · `TAMPERED` · `SIGNAL_LOST`.

**Adversaries disable cameras before crossing.** A surveillance system that
cannot distinguish "nothing is happening" from "I have been blinded" reports a
clear sector when it has stopped seeing.

**Calibration.** Four operator-marked ground-plane reference points per camera
produce the homography matrix **H** mapping image coordinates to metric ground
coordinates. Setup is a one-time operation taking under two minutes per camera
using the provided interactive tool.

**Atmospheric correction.** CLAHE for under-exposure, Dark Channel Prior for
scattering-induced contrast loss. Fog and darkness are different physical
problems requiring different corrections. **Both are applied exclusively to the
operator-viewable stream.** The detector receives the raw sensor frame.

> **Why enhancement is excluded from the detection path.** It is intuitive to
> brighten a dark frame before detection. The research consensus is against it:
> enhancement optimises a perceptual-quality objective, not the detection
> objective, and the artefacts it introduces damage small-object detection
> specifically — which is precisely this system's operating regime. A person at
> 150 m on wide-angle CCTV occupies 15–25 pixels of height. The separation is
> enforced structurally: the frame contract carries two distinct buffers, so
> the error is unrepresentable rather than merely discouraged.

### Tier 1 — Real-Time Perception

**Detection.** YOLO26-s. NMS-free end-to-end architecture, Distribution Focal
Loss removed, and STAL label assignment which guarantees positive-sample
coverage for small objects — the defining difficulty of this problem domain.
Exported through TensorRT with INT8 quantisation for deployment.

Operational classes: **Person · Vehicle · Animal · Bag · Aerial**. The class
mapping from the underlying detector's taxonomy is configuration data, never
a hardcoded enumeration, so extending the class list is a YAML edit.

**Sliced inference (SAHI).** Full-frame inference at 960 px downsamples a
1080p frame; a 20-pixel target becomes 18 pixels and is lost. SAHI runs the
detector at native scale across overlapping tiles and merges class-wise, at a
cost of approximately one forward pass per tile. Enabled per camera where
target range justifies it.

**Pose estimation.** YOLO26-Pose on person crops, producing 17 COCO keypoints.
Running on crops rather than the full frame makes the skeleton-to-detection
association exact by construction — one crop, one subject, no matching step.
The alternative, matching full-frame skeletons back to detections by proximity,
silently mis-assigns whenever two people share an image column.

**Tracking.** BoT-SORT is the primary tracker: published comparisons give it
the best IDF1 and the lowest identity-switch count, and identity stability
through fence and vegetation occlusion is what this application requires.
ByteTrack remains available as the throughput option and FastTracker adds
Kalman rollback on occlusion. All three are configuration-selectable, and the
comparison table is itself a differentiator.

One tracker instance per camera. Tracker state is never global — two concurrent
streams sharing tracker state cross-contaminate identities.

### Tier 2 — Spatial Reasoning

Pure geometry. No models, no GPU, no learned parameters.

**Metric transformation.** The bounding-box footpoint — bottom-centre, the
ground-contact estimate — is transformed through **H** into world coordinates.

**Directional tripwire.** A line in ground coordinates with a declared inbound
side. Crossing requires both a sign change in the signed-area test *and* an
intersection with the finite segment: a subject walking past the end of a fence
has changed side of the infinite line without crossing the fence.

Direction is the entire reason to use a tripwire rather than a containment
test. **Infiltration and exfiltration are operationally different events**, and
a polygon cannot distinguish them.

**Zone presence.** Point-in-polygon containment against restricted areas.

**Proximity rings.** Euclidean distance in metres to a protected asset, fence
line, or installation. A metric threshold is depth-invariant: 3 m is 3 m
whether the subject stands near the camera or at the far end of the scene. A
pixel threshold represents two different real distances at those two depths and
is therefore wrong in at least one of them.

**Per-zone group filtering.** Each geometry declares which object groups it
tests. Wildlife triggering a perimeter is the canonical false alarm in this
domain — named explicitly in the CIBMS assessments — so excluding animals from
a given tripwire is a first-class configuration control.

### Tier 3 — Temporal Reasoning

**Every timer, counter and dwell threshold in the system lives here.** No
component downstream of perception keeps its own clock. Timing logic scattered
through a detection loop is untestable and cannot be reasoned about as policy.

This tier has no GPU dependency and no video dependency, which is what makes
the suppression policy — the component that determines whether the system is
operationally useful or operationally ignored — unit-testable with synthetic
data.

**Temporal confirmation.** Specified in **wall-clock seconds plus a hit ratio**,
never in raw frame counts.

> N frames at 30 fps and N frames at 8 fps are different guarantees, and
> degraded backhaul — the stated target condition — is exactly what produces
> 8 fps. A frame-count threshold silently changes meaning on the very footage
> the system is designed for. Three frames is 100 ms on a healthy feed and
> 375 ms on a struggling one.

The ratio is computed against processed frames, not against observations: a
subject present in 6 of the last 10 frames scores 0.6 regardless of how long
those frames took to arrive.

Per-observation-kind overrides apply. Camera health faults confirm faster than
perimeter events; stance anomalies require a longer window.

**Instantaneous versus persistent events.** A tripwire crossing occurs in
exactly one frame and can never satisfy an N-of-M rule designed for persistent
states. After a crossing, corroborating observations continue while the subject
**remains on the side it crossed to**. A genuine incursion corroborates; a
detector flicker across the line does not, because the track does not persist
on the far side.

**Person–object association and abandoned baggage.**

```
  [ person detected ]
          │  bag within R metres of a person
          ▼
   ┌──────────────┐
   │ ACCOMPANIED  │◄──────────────────────────┐
   └──────┬───────┘                           │
          │  owner separation > D metres      │ owner returns within D
          ▼                                   │ and holds for grace period
   ┌──────────────────────┐                   │
   │ DETACHED, timer runs │───────────────────┘
   └──────┬───────────────┘
          │  elapsed > T_dwell
          ▼
   ┌──────────────────────────┐
   │ ALERT: unattended object │
   └──────────────────────────┘
```

R and D are in **metres** via the homography, making them depth-invariant.
T_dwell is 120 s operationally, 15 s for demonstration.

A bag never observed with any person is classified `ORPHAN` and **does not
alert**. Without an owner there is no abandonment event — only an object that
cannot be explained. Alerting on it would assert more than the system knows.

**Stance classification.** Torso-axis verticality derived from shoulder and hip
keypoints, rather than bounding-box aspect ratio. The torso axis is a
body-intrinsic measure and does not change when the bounding box is truncated
by the image edge.

> This distinction matters more than it appears. A subject walking toward a
> camera has a bounding-box aspect ratio that climbs steadily as the box clips
> the frame edge — on representative footage, from 0.39 to 0.84 — purely from
> truncation. An aspect-ratio rule reads that progression as crouching, then
> crawling, and raises a critical infiltration alert on a person who is simply
> walking. Edge-clipped detections are suppressed outright from stance
> classification.

Stance never becomes an alert directly. It emits an observation subject to the
same temporal confirmation as everything else, with a longer window.

### Tier 4 — Specialised Identification

Triggered, never continuous. All operate on region-of-interest crops, not full
frames — a rifle at range occupies a handful of pixels in a full frame and is
readily detectable on an upscaled person crop.

- **Weapon detection** on person crops: rifle, pistol, grenade, knife.
  Cropped rather than full-frame because of scale, and graded by *carrier*
  rather than by object — the threat matrix reads `WEAPON_VISIBLE` as `INFO` on
  an enrolled friendly and `CRITICAL` on someone unidentified, because a guard
  on patrol is armed.

  > **Measured** on 905 held-out images containing 1,062 weapons: mAP@50
  > **0.802**, mAP@50-95 0.559. At the configured 0.60 confidence the detector
  > recalls 59.4% of real weapons and draws roughly two spurious boxes across
  > 260 frames of weapon-free footage, none of which survive temporal
  > confirmation into an alert.
- **ANPR**: plate localisation on vehicle crops, then PP-OCRv6 recognition,
  tuned for Indian HSRP and two-line commercial formats
- **Face detection and recognition**: YuNet detector, SFace embeddings
- **Appearance embedding**: OSNet
- **Flight-path analysis**: multi-frame trajectory statistics for aerial
  discrimination (§13)

### Tier 5 — Identity Orchestrator

**Open-set by construction. The system never classifies hostility.**

> Closed-set friend/foe classification is not implementable and would not be
> defensible if it were. No legitimate open dataset of adversary appearance
> exists at CCTV resolution, and a closed-set classifier answers the question
> *"what about a civilian in dark clothing?"* badly and confidently — the worst
> available combination. SENTINEL therefore enrols only what can be
> legitimately sourced: friendly BOP/BSF personnel and uniform references.

Verdicts:

| Verdict | Meaning | Rendering |
|---|---|---|
| `FRIENDLY` | Matched an enrolled identity above threshold, with similarity reported | Green |
| `UNIDENTIFIED` | Not enrolled. **This is not an assertion of hostility.** | Amber, escalating on breach |
| `INSUFFICIENT_QUALITY` | The crop could not support a judgement | Neutral, stated explicitly |

**Quality gate.** Crops are embedded only when they can support a claim:
minimum pixel dimensions, minimum detector confidence, minimum Laplacian
sharpness, and no frame-edge truncation. A 30-pixel motion-blurred crop
produces an embedding that is confidently wrong, and matching on it
manufactures a verdict out of noise.

**Identity accrues over a track's lifetime.** A single blurred frame does not
downgrade a verdict already reached from a better crop. On degraded footage
detector confidence fluctuates frame to frame, and a per-frame identity reading
flickers uselessly.

**Cross-camera handoff.** Local per-camera track identifiers resolve to a
global entity identifier through appearance matching, producing the **common
operating picture** whose absence CIBMS assessments identify as a primary
failure.

### Tier 6 — Asynchronous Reasoning

LocateAnything-3B performs open-vocabulary visual grounding on alert keyframes
and on operator request. Invoked on demand only — never per frame. A
3-billion-parameter vision-language model in a video loop is an architectural
error.

Queue depth is one, with rejection. If a grounding request is in flight, the
next is refused rather than queued. **Refusal is the feature.** A queue in
front of a multi-second GPU operation is how a system exhausts memory under
load.

### Tier 7 — Archive Index

Grounding models operate on single frames. Temporal archive search — *"find the
red tractor from Tuesday night"* — requires a continuously-built index.

Thirty-second overlapping clip windows are embedded with Qwen3-VL (or CLIP for
a lighter deployment) and stored in Milvus with camera identifier and time
bounds. This makes the **entire archive** searchable rather than only footage
that already triggered an alert.

### Tier 8 — Alert Manager, Evidence Store and C2

The alert manager is **the only component permitted to construct an Alert**.
Perception produces observations; promotion is one component's responsibility
so that suppression policy exists in exactly one auditable place.

**Two independent axes (§17.2).** Threat determines colour. Confidence
determines line style. They are separate fields and are never combined into a
single score, because a low-confidence critical event and a high-confidence
trivial one must not render identically.

**Lifecycle.** `PENDING → ACTIVE → CLEARED | EXPIRED`. An alert whose
supporting observations cease is cleared; one that never resolves expires.

> A threat state that only escalates is a defect. A single false positive would
> latch the display permanently, and the operator would correctly learn to
> ignore it. Sector threat level is computed from the **live** alert set, never
> from a running maximum.

**Confidence is computed pessimistically.** It is discounted for uncalibrated
geometry, edge-clipped detections, and newly-established tracks, and each
discount is recorded in the alert's human-readable rationale. Understating
confidence costs a hatched box; overstating it costs operator trust, which does
not return.

### Tier 9 — Query Router

See §16.

---

## 8. Complete Capability Catalogue

| ID | Capability | Method |
|---|---|---|
| **F1** | Multi-class detection — person, vehicle, animal, bag, aerial | YOLO26-s, NMS-free end-to-end, TensorRT INT8, SAHI tiling |
| **F2** | Persistent multi-object tracking | BoT-SORT / ByteTrack / FastTracker, config-selectable, per-camera instance |
| **F3** | Real-world metric coordinates | 4-point homography, `findHomography`, footpoint transform |
| **F4** | Directional tripwire and geofence breach | Shapely line-crossing with signed-area direction, polygon containment |
| **F5** | Metric proximity alerting | Inner ring in metres, depth-invariant |
| **F6** | Unattended baggage | Person↔object association + dwell state machine |
| **F7** | Temporal alert confirmation | N-of-M in seconds and hit-ratio, per-kind overrides |
| **F8** | Open-set friendly vs unidentified | OSNet embeddings vs enrolled gallery, quality-gated |
| **F9** | Anti-tamper and camera health | Laplacian variance + SSIM drift + freeze detection |
| **F10** | Structured query | Parameterised SQL over the alert log, FTS5 |
| **F11** | Forensic visual grounding | LocateAnything-3B on keyframes, on demand |
| **F12** | Dual-axis alert rendering | Threat → colour, confidence → style |
| **F13** | Evidence store | Keyframe, clip, all confidences, model IDs, geometry, threshold snapshot |
| **F14** | Sub-3 KB tactical relay | Compact JSON + base64 JPEG, quality-degraded to budget |
| **F15** | Offline operation | Air-gapped, no cloud, no external API |
| **F16** | Infiltration stance | YOLO26-Pose keypoints, torso-axis classification on crops |
| **F17** | Aerial object detection | Small-object detection + multi-frame flight-behaviour discrimination |
| **F18** | RGB–thermal fusion | Paired visible/infrared inference on aligned streams |
| **F19** | Acoustic cross-cueing | DAS event classification + PTZ slew-to-cue fusion |
| **F20** | Weapon detection | Fine-tuned detector on person ROI |
| **F21** | ANPR — Indian plates | Plate detector + PP-OCRv6, HSRP and commercial formats |
| **F22** | Face presence and recognition | YuNet + SFace, permissively licensed |
| **F23** | Cross-camera re-identification | Global entity ID via appearance matching |
| **F24** | Archive-wide semantic search | Clip-embedding index over all stored footage |
| **F25** | Common operating picture | Multi-camera fused state at all command levels |

---

## 9. Technology Stack — Component by Component

### 9.1 Perception

| Component | Technology | Version / Variant | Licence | Rationale |
|---|---|---|---|---|
| Primary detector | **YOLO26-s** | Ultralytics | AGPL-3.0 | NMS-free end-to-end; DFL removed; STAL assignment guarantees positive coverage for small objects — the defining difficulty here. 40.9–57.5 mAP COCO at 1.7–11.8 ms on T4 TensorRT. |
| Detector alternative | **RF-DETR-S** | Roboflow | Apache-2.0 | Transformer-based, strong domain transfer, permissive licence. Benchmarked against YOLO26 on the low-light evaluation set. |
| Small-object inference | **SAHI** | — | MIT | Sliced inference for targets beyond ~150 m where full-frame downsampling loses them. |
| Pose | **YOLO26-Pose** | Ultralytics | AGPL-3.0 | 17 COCO keypoints on person crops for stance classification. |
| Primary tracker | **BoT-SORT** | — | MIT | Best IDF1 and lowest ID-switch count in published comparisons; appearance-aware association survives fence and vegetation occlusion. |
| Tracker alternatives | **ByteTrack**, **FastTracker** | — | MIT | ByteTrack for throughput; FastTracker adds Kalman rollback on occlusion. |
| Inference runtime | **TensorRT** | INT8 | NVIDIA | Ahead-of-time engine build; avoids runtime kernel compilation. |
| Framework | **PyTorch** | ≥ 2.7.0, cu128 | BSD | First release with native Blackwell sm_120 support and prebuilt cu128 wheels. |

### 9.2 Video I/O and pre-processing

| Component | Technology | Licence | Detail |
|---|---|---|---|
| Multi-camera aggregation | **MediaMTX** | MIT | RTSP server aggregating n camera streams. |
| Decode | **GStreamer + NVDEC** | LGPL | Hardware H.264/H.265 decode directly into the inference pipeline. |
| Fallback I/O | **OpenCV** | Apache-2.0 | File sources, single-camera deployments, all image operations. |
| Encode | **FFmpeg / libx264** | LGPL/GPL | Evidence clips in browser-playable H.264, yuv420p, faststart. |
| Contrast restoration | **CLAHE** (OpenCV) | Apache-2.0 | Adaptive histogram equalisation on the LAB luminance channel only, preserving chroma. |
| Dehazing | **Dark Channel Prior** | — | He et al. Single-image scattering correction with guided-filter refinement. |
| Scaling for large deployments | **NVIDIA DeepStream / Savant** | — | N-stream scaling path. |

### 9.3 Spatial and temporal

| Component | Technology | Licence | Detail |
|---|---|---|---|
| Homography | **OpenCV `findHomography`** | Apache-2.0 | 4-point DLT; `perspectiveTransform` for batch footpoint mapping. |
| Geometry | **Shapely 2.x** | BSD-3 | Polygon containment, line intersection, distance. GEOS-backed. |
| Depth cross-check | **YOLO26 depth head** | AGPL-3.0 | Independent monocular depth estimate; agreement between homography and depth is a quantitative robustness result. |
| State machine | **Python asyncio** + **Redis** | BSD | Dwell timers, association, alert lifecycle, confirmation. Redis for multi-process deployments; in-process for single-node. |

### 9.4 Identity

| Component | Technology | Variant | Licence | Detail |
|---|---|---|---|---|
| Appearance re-ID | **OSNet** | `osnet_x0_25` | MIT | 2.31 M parameters, 512-dimensional embedding. Omni-scale architecture: four parallel depthwise-separable streams at receptive fields 3/5/7/9, fused by a learned channel gate. Pretrained on MSMT17. |
| Occlusion-robust re-ID | **KPR** | — | — | Keypoint-based part re-identification for heavily occluded subjects. |
| Face detection | **YuNet** | OpenCV Zoo | Apache-2.0 | Permissive, deployable by a government agency. |
| Face recognition | **SFace** | OpenCV Zoo | Apache-2.0 | Replaces InsightFace packs, which are non-commercial research only. |
| Vector search | **Milvus** | — | Apache-2.0 | Three collections: faces, friendly appearance gallery, body re-ID. |
| Vector search (small scale) | **ChromaDB** / NumPy exhaustive | Apache-2.0 | Tens of identities are searched faster by matrix multiply than by any index. |

### 9.5 Reasoning and retrieval

| Component | Technology | Licence | Detail |
|---|---|---|---|
| Grounding VLM | **nvidia/LocateAnything-3B** | NVIDIA Open Model Licence | Parallel Box Decoding predicts complete boxes in a single step rather than autoregressively — 12.7 boxes/second versus 1.1 for textual decoding. Trained on 12 M images, 138 M+ queries, 785 M boxes. Base LM: Qwen2.5-3B-Instruct. Vision encoder: MoonViT-SO-400M (MIT). arXiv 2605.27365. |
| Video VLM / clip embedding | **Qwen3-VL-8B** | Apache-2.0 | ~6 GB at Q4, native video input, competitive with much larger models on video understanding. |
| Clip embedding (light) | **CLIP** | MIT | Lower-cost archive indexing path. |
| OCR | **PP-OCRv6** | Apache-2.0 | Released June 2026. Medium tier: 34.5 M parameters, +5.1% recognition and +4.6% detection over PP-OCRv5-server. |

### 9.6 Platform

| Component | Technology | Licence | Detail |
|---|---|---|---|
| API | **FastAPI** | MIT | Async, WebSocket-native, OpenAPI generated. |
| Server | **Uvicorn** | BSD | ASGI. |
| Live video transport | **MJPEG** over `multipart/x-mixed-replace` | — | Browser-native, no plugin, no client decoder. |
| Live state transport | **WebSocket** | — | Track boxes, threat, confidence, health, queue counters at 10 Hz. |
| Field relay | **MQTT** | — | Sub-3 KB alert payloads over constrained links. |
| Low-latency video option | **WebRTC** | — | Where round-trip latency matters more than simplicity. |
| Alert store | **PostgreSQL** | PostgreSQL Licence | Production. Full-text search, JSON columns, concurrent access. |
| Alert store (edge) | **SQLite + FTS5** | Public domain | Single-node deployments. Zero-configuration, file-backed, air-gap friendly. |
| Config | **Pydantic v2 + YAML** | MIT | Typed validation; profile overlay composition. |
| Annotation | **CVAT** (self-hosted) | MIT | Air-gapped labelling for plates, weapons, stances. |
| Testing | **pytest** | MIT | Temporal and spatial layers test without GPU, video or model weights. |

### 9.7 Interface

| Component | Technology | Detail |
|---|---|---|
| Rendering | HTML5 Canvas over MJPEG | Overlays drawn client-side from WebSocket state, never burned into video. This keeps the evidence keyframe identical to the detector's input and makes rendering changes independent of encoding. |
| Styling | CSS custom properties | Dark tactical palette; threat colours and confidence styles are design tokens. |
| Layout | Grid + flexbox | Live picture, alert stream, evidence drawer, system telemetry. |

---

## 10. Core Algorithms in Detail

### 10.1 Ground-plane homography

At installation the operator marks four ground-plane reference points — a known
5 m × 5 m patch is convenient but any four measured coplanar points suffice.
OpenCV computes the perspective transform **H** such that:

```
    ⎡ X_metric ⎤        ⎡ x_footpoint ⎤
    ⎢ Y_metric ⎥  =  H · ⎢ y_footpoint ⎥
    ⎣    1     ⎦        ⎣      1      ⎦
```

The footpoint — bottom-centre of the bounding box — is the ground-contact
estimate. Proximity, tripwire and dwell thresholds are then evaluated in true
Euclidean metres.

**Validation.** A ground plane maps to an image by an *exact* homography, so
round-tripping the calibration points must produce a residual near zero. A
non-trivial residual indicates the marked points were not coplanar — typically
someone clicked the top of a wall rather than its base. The system reports this
residual rather than concealing it.

**Cross-check.** The YOLO26 depth head runs in parallel, and agreement between
the two independent estimates is reported as a quantitative robustness measure.

**Scope.** Fixed cameras. A PTZ camera invalidates **H** on every movement and
is handled by a preset-linked calibration table — one homography per preset,
selected by the reported position.

### 10.2 Directional crossing

For a tripwire from **A** to **B** and a point **P**, the signed area is:

```
    side(P) = (Bx − Ax)(Py − Ay) − (By − Ay)(Px − Ax)
```

A crossing between consecutive footpoints **P₋₁** and **P₀** requires:

1. `sign(side(P₋₁)) ≠ sign(side(P₀))` — an actual side change, and
2. the segment `P₋₁P₀` intersects the **finite** segment `AB`

Condition 2 matters: a subject walking beyond the end of a fence changes side
of the infinite line without crossing the fence.

Direction is then determined by which side the subject ends on, compared
against the configured inbound side.

### 10.3 Temporal confirmation

For an observation kind *k* with window *W* seconds, minimum hits *N*, and
minimum ratio *ρ*:

```
    hits    = |{ observations of subject s in [t − W, t] }|
    ticks   = |{ frames processed for camera c in [t − W, t] }|

    confirm ⟺  hits ≥ N   ∧   hits / ticks ≥ ρ
```

Because the denominator is processed frames rather than elapsed time, the rule
is invariant to frame rate: the same real-world event confirms identically at
30 fps and at 8 fps.

A candidate confirms **once**. A subject standing in a restricted zone produces
one alert, not one alert per frame.

### 10.4 OSNet omni-scale embedding

Each OSNet block runs four parallel streams of depthwise-separable 3×3
convolutions at effective receptive fields of 3, 5, 7 and 9, then fuses them
through a learned channel gate — a squeeze-excitation-style unified aggregation
gate shared across streams. That omni-scale fusion is why a 2.31 M-parameter
network competes with substantially larger re-identification backbones, and why
it is the correct speed/accuracy operating point for real-time use.

Input is 256×128 with ImageNet normalisation; output is a 512-dimensional
L2-normalised embedding compared by cosine similarity.

**Threshold selection.** The network's final fully-connected head terminates in
ReLU, so all embeddings occupy the positive orthant and cosine similarities
between *unrelated* crops sit far above zero — typically 0.4 to 0.6, not near
0. A threshold chosen from intuition about cosine similarity lands near 0.5 and
matches everyone. The threshold is therefore fitted to measured distributions
of same-subject and different-subject similarity, not selected by feel.

### 10.5 Camera health

**Laplacian variance:**

```
    blur_score = Var( ∇²I )
```

where ∇² is the discrete Laplacian. High-frequency energy collapses under
obscuration or defocus.

**SSIM** against a rolling reference, computed on a downscaled grey image with
an 11×11 Gaussian window, σ = 1.5:

```
    SSIM(a,b) = [(2μₐμ_b + C₁)(2σₐb + C₂)] / [(μₐ² + μ_b² + C₁)(σₐ² + σ_b² + C₂)]
```

The reference updates by exponential moving average only while the camera reads
healthy, with a deliberately small coefficient — a fast update would absorb a
lens cover and then report health, allowing the tamper to erase its own
evidence.

### 10.6 Relay payload compression

Given a byte budget *B*:

1. Serialise the structured envelope with abbreviated keys (at 3 KB, key names
   are a measurable fraction of payload)
2. Compute remaining budget after the envelope
3. Downscale the subject crop to a maximum edge
4. Encode JPEG at descending quality until `base64(jpeg) ≤ remaining`
5. If no quality fits, **omit the image entirely**

The budget is enforced by degrading the image, **never** by truncating the
JSON. A clipped JSON document is unparseable and therefore worthless; a grainier
thumbnail is still a thumbnail. The structured fields are what the receiving
post acts upon.

---

## 11. Data Contracts

The contracts package imports nothing from the rest of the system. Spatial and
temporal layers import only contracts. This is what makes the suppression
policy testable without a model, a camera, or a video file.

### FrameBundle — what ingestion emits

| Field | Type | Notes |
|---|---|---|
| `camera_id` | str | Track IDs are unique only within a camera |
| `frame_idx` | int | Monotonic per stream, from the decoder |
| `captured_at` | datetime, tz-aware | **Wall clock.** Frame-index ÷ fps is meaningless for a live stream and useless in a forensic record |
| `raw` | ndarray (H,W,C) | **The detector's only input. Never enhanced.** |
| `view` | ndarray \| None | Operator-viewable; enhancement writes here exclusively |
| `channels` | tuple[str,…] | `("B","G","R")` or `("B","G","R","T")` for thermal fusion |
| `health` | HealthStatus | Per-frame quality assessment |

### Detection — one object, one frame, pre-tracking

| Field | Type | Notes |
|---|---|---|
| `bbox_px` | (x1,y1,x2,y2) | |
| `class_id` / `class_name` | int / str | From configuration, never an enum |
| `group` | ObjectGroup | person · vehicle · animal · bag · aerial · other |
| `confidence` | float | |
| `model_id` | str | **Required.** Model name plus weights hash. "Which weights said this?" must be answerable. |
| `edge_clipped` | bool | Box touches a frame border; geometry unreliable |

Carries no `track_id`. Identity over time is the tracker's responsibility, and
conflating them is how counting errors enter.

### Track — identity over time

| Field | Type | Notes |
|---|---|---|
| `track_id`, `camera_id` | int, str | |
| `footpoint_m` | (x,y) \| **None** | Ground metres. `None` when uncalibrated — a fabricated metre value is worse than an absent one |
| `history` | deque[TrackPoint] | Retained trajectory; consumed by flight-path analysis |
| `identity` | IdentityVerdict \| None | |
| `stance` | Stance | STANDING · CROUCHING · PRONE · UNKNOWN |
| `owner_track_id` | int \| None | Set on bag tracks |

### IdentityVerdict

| Field | Type |
|---|---|
| `label` | `FRIENDLY` \| `UNIDENTIFIED` \| `INSUFFICIENT_QUALITY` |
| `matched_enrollee` | str \| None |
| `similarity`, `threshold_used`, `quality_score` | float |
| `model_id` | str |

**There is no `HOSTILE` value.**

### Observation vs Alert

| | Observation | Alert |
|---|---|---|
| Meaning | "Something happened this frame" | "This is confirmed and actionable" |
| Cost | Cheap, frequent, disposable | Expensive, rare, audited |
| Visibility | Never shown alone | Operator-facing |
| Created by | Any Tier 2/3/5 component | **Only the alert manager** |
| Commitment | None | Full evidence record |

### Alert

| Field | Type | Notes |
|---|---|---|
| `threat` | ThreatLevel | **Colour axis** |
| `confidence` + `confidence_band` | float, band | **Style axis.** Never combined with threat. |
| `state` | PENDING · ACTIVE · CLEARED · EXPIRED | |
| `supporting` | tuple[Observation,…] | The frames that satisfied confirmation. **The alert carries its own proof.** |
| `evidence` | EvidenceRef | |
| `rationale` | str | One human-readable line: why this fired |

### EvidenceRef

Keyframe path · crop path · clip reference with time bounds · every supporting
detection with its own confidence and model ID · model ID map with weights
hashes · triggering geometry · homography identifier · **configuration snapshot**.

> The configuration snapshot is the least obvious and most useful field.
> Without it, "why did this fire?" becomes unanswerable three days later once
> the operational profile has replaced the demonstration profile.

### SensorCue

| Field | Type | Notes |
|---|---|---|
| `source_id` | str | |
| **`simulated`** | **bool, required, no default** | A cue cannot be constructed without declaring its provenance |
| `kind` | FOOTSTEPS · VEHICLE · DIGGING · FENCE_TAMPER · UNKNOWN | |
| `position` | LinearPosition \| Bearing | Fibre-metre or degrees from north |

---

## 12. Datasets

Every dataset below is verified for licence compatibility before entering any
pipeline.

### 12.1 Low light and night

| Dataset | Scale | Content | Use |
|---|---|---|---|
| **LLVIP** | ~15,488 aligned pairs | Visible + infrared night surveillance, pedestrians labelled | The most relevant single public dataset for this problem. Basis for RGB–thermal fusion training and evaluation. |
| **ExDark** | 7,363 images, 12 classes | Real low-light photography | Standard low-light detection benchmark. Baseline mAP control for every subsequent claim. |
| **DarkFace** | — | Low-light face detection | Night-time face presence. |

### 12.2 Abandoned object

| Dataset | Scale | Content |
|---|---|---|
| **ABODA** | 11 sequences | Crowded scenes, lighting changes, night, indoor and outdoor. Effectively the only public abandoned-object detection dataset. |
| **PETS2006** | — | Classic left-baggage benchmark. |
| **i-LIDS AVSS2007** | — | UK Home Office. Abandoned baggage and parked vehicles. |

### 12.3 Tracking and re-identification

| Dataset | Scale | Content |
|---|---|---|
| **MOT17 / MOT20** | — | Standard multi-object tracking benchmarks. MOT20 stresses crowd density. |
| **DanceTrack** | — | Specifically stresses identity stability under similar appearance and non-linear motion. |
| **Market-1501** | 1,501 identities, 6 cameras | Standard person re-identification benchmark. |
| **MSMT17** | 4,101 identities | Larger, multi-season, multi-time-of-day. OSNet's pretrained weights. |

### 12.4 Aerial

| Dataset | Content |
|---|---|
| **Drone-vs-Bird** | The standard benchmark for the discrimination problem that defines aerial detection. |
| **BirDrone** | arXiv 2601.08319, January 2026. Contemporary drone/bird discrimination set. |

### 12.5 Weapons

| Dataset | Scale | Content | Licence |
|---|---|---|---|
| **US Real-time gun detection in CCTV** (Univ. Seville) | U0.5 / U1 / U2.5 tiers | Unity synthetic multi-camera plus real mock-attack footage. 4 handguns, 5 rifles, knife, and a **smartphone hard-negative class** — the single most important negative, since a raised phone is the canonical false positive. | Free for academic use with citation |
| **Simuletic cctv-weapon-dataset** | Sample + full | CCTV perspective, YOLO format, person + weapon | CC BY 4.0 |

Synthetic-primary training is accepted methodology in this subfield. Real
mock-attack footage is held out as the test set and the synthetic→real gap is
reported explicitly.

### 12.6 Indian ANPR

| Dataset | Scale | Content |
|---|---|---|
| **Indian Licence Plate Dataset in the wild** | 16,192 images, 21,683 plates | 4-corner and per-character annotation. The strongest open Indian option by a wide margin. |
| **BARC Indian Number Plates** | ~1,500 | Documents the **+42.9% detection gain** achieved through distribution alignment — the empirical justification for fine-tuning rather than deploying a Chinese-plate model directly. |

### 12.7 Query evaluation

| Dataset | Scale | Content |
|---|---|---|
| **SurveillanceVQA-589K** | 589,000 QA pairs | Surveillance video-language benchmark across five reasoning dimensions. Converts natural-language retrieval from a qualitative demonstration into a measured result. |

### 12.8 Licence verification

Every dataset licence is verified before use. **At least one commonly-cited
Indian plate dataset (DataCluster Labs) is CC BY-NC-ND** — the NoDerivatives
clause prohibits even research fine-tuning. Licence review is a pipeline gate,
not an afterthought.

---

## 13. The Aerial Threat — Drone Detection

### 13.1 Why it matters

Payload-carrying UAVs are the dominant smuggling vector on the Punjab border
(§4.4). Ground and tower sensor grids do not address the aerial vector, and
that gap is documented in the Indian policy literature.

### 13.2 The detection problem is the problem already solved

A drone at altitude presents the same *detection* challenge as a person at
200 m: small, low-contrast against a bright or uniform background,
motion-blurred. The SAHI tiled-inference approach applies directly, and the
class list extends by configuration rather than by architectural change.

### 13.3 The discrimination problem is the real one

Published work is unambiguous: **a single frame cannot reliably separate a
drone from a bird.** At the pixel scales involved both are dark blobs against
sky.

The discriminator is **multi-frame motion analysis** over retained track
trajectories:

| Signal | Bird | Quadcopter |
|---|---|---|
| Flight-path linearity | Wandering, irregular | Straight-line translation between waypoints |
| Hover capability | Cannot hold station (excluding specific species/conditions) | Holds position precisely |
| Acceleration profile | Flapping produces periodic vertical oscillation | Smooth, near-constant |
| Turn radius | Continuous, banked | Sharp, near-instantaneous yaw |
| Velocity consistency | Variable with wind and effort | Commanded, stable |

This is exactly why `Track.history` retains trajectories rather than discarding
them after the current frame — the architecture anticipates this analysis.

### 13.4 The first-line discriminator: disagreement between models

Motion analysis needs a track, and a track needs several frames. The
discrimination that runs *first*, on the frame itself, is cheaper and comes
from a different observation: the perimeter detector already carries `bird` as
COCO class 14, trained on far more bird imagery than any drone dataset
contains.

The obvious alternative — train a `bird` class alongside `drone` — is worse
than it appears. The mixed drone/bird sets available carry a few dozen bird
boxes against thousands of drone boxes, which produces a weak bird detector
competing with a strong one already present in the system.

So the discrimination is made **between models**. The aerial detector proposes;
the perimeter model disputes with an overlapping bird detection; the more
confident wins. Withheld candidates are surfaced to the operator with the
reasoning printed, because a system that can show why it declined makes a
stronger claim than one that only shows hits.

One implementation detail is load-bearing. A withheld candidate is **not**
emitted as a zero-confidence observation. Temporal confirmation counts hits
rather than confidence, so a flock of rejected birds would confirm into exactly
the alert the rejection existed to prevent. Withheld candidates go to a
display-only list that never reaches the confirmer.

### 13.5 The frame-area ceiling

Aerial training data is real imagery but it is not *surveillance* imagery: FPV
footage and product photography dominate, and 44% of the boxes in the set used
here cover more than 2.25% of the image. A model trained on that will, on a
dark wide shot, propose a box covering half the frame.

The fix is not a higher confidence threshold — that costs recall everywhere to
solve a problem that only occurs at one end of the size distribution. Instead
a candidate covering more than a configured fraction of the frame is withheld,
on the grounds that a drone filling a third of a surveillance frame is metres
from the lens rather than crossing a border.

> **Measured** against 1,078 held-out drones and 442 frames of drone-free
> footage, on the shipped weights (mAP@50 **0.9552**). Every row already has
> the 35% area ceiling applied:
>
> | conf | recall | frames with a false drone |
> |---|---|---|
> | 0.60 | 72.8% | 11.3% |
> | 0.70 | 66.5% | 6.3% |
> | **0.75** | **62.2%** | **0.0%** |
> | 0.80 | 54.9% | 0.0% |
> | 0.85 | 41.2% | 0.0% |
>
> The highest false confidence anywhere in that footage is 0.730, so 0.75 sits
> just above it and 0.80 buys nothing while costing seven points.

### 13.6 Response

Aerial detections raise their own observation kind, feed the same temporal
confirmation, and enter the same threat matrix. A confirmed hovering
quadcopter over a border sector escalates on trajectory persistence rather than
on any single frame.

---

## 14. RGB–Thermal Fusion

### 14.1 Why

Thermal imaging retains strong object response precisely where visible-spectrum
imaging collapses — full darkness, smoke, light fog, camouflage against
vegetation. This directly answers the documented CIBMS finding that *"equipment
did not work in adverse climatic conditions."*

### 14.2 Method

Fusion requires **paired, spatially aligned visible and infrared frames**.
LLVIP provides approximately 15,488 such pairs from night surveillance with
labelled pedestrians, and remains an active research benchmark — COMO, CMAF-Net,
EGM-Det and YOLOv11-RGBT are all contemporary work on it.

The detector interface accepts a frame carrying more than three channels:
`FrameBundle.raw` is shape `(H, W, C)` with `C` unconstrained, paired with an
explicit `channels` descriptor. A four-channel RGB-T frame is therefore a new
*producer*, not a rewrite of the detection stack.

### 14.3 Reporting

Fusion performance is reported as measured results on LLVIP — a public
benchmark with published baselines, against which any claim is independently
checkable.

---

## 15. Distributed Acoustic Sensing

### 15.1 What DAS is

Distributed acoustic sensing turns a standard optical fibre into thousands of
virtual microphones. A φ-OTDR interrogator launches coherent light pulses and
measures Rayleigh backscatter; strain from footsteps, vehicle movement,
digging, or fence tampering perturbs the backscatter at a position resolvable
to a few metres along the fibre.

**Coverage:** OptaSense has demonstrated a single monitoring station covering
1,500 km of fibre.

### 15.2 Why it fits this problem exactly

Re-read the documented CIBMS failures (§4.2): *high cost*, *poor spare-parts
availability*, *equipment failure in adverse climate*, *manpower-intensive
operation*.

DAS requires **no power and no electronics in the field**. The fibre is
passive. All active equipment sits in a single hardened enclosure. There is
nothing along the line to steal, vandalise, corrode, or maintain. It is
directly responsive to the maintenance and spare-parts problems that made
previous deployments unsustainable.

### 15.3 The classification layer

DAS event classification is a signal-machine-learning problem: CNNs, vision
transformers, LSTMs or U-Nets operating over time-frequency representations
(spectrograms, scalograms) of the strain signal, discriminating footsteps from
vehicle movement from digging from livestock.

### 15.4 The cross-cue fusion harness

This is the component that addresses the hardest documented problem in the
domain.

An acoustic event arrives as a `SensorCue` — *"disturbance at fibre-metre
4,120, classified footsteps, confidence 0.81"*. The fusion layer maps linear
fibre position to a geographic coordinate, resolves which camera covers that
bearing, slews a PTZ to the corresponding view, and begins tracking. The visual
system then either corroborates the acoustic classification or does not, and
that agreement or disagreement is itself recorded as evidence.

**This is the seamless multi-sensor fusion whose absence CIBMS assessments
identify as the single most difficult engineering problem in the entire
domain.**

### 15.5 Provenance discipline

The acoustic module is named `das_sim`. The `SensorCue` type carries a
**required** `simulated` boolean with no default value, making it structurally
impossible to construct a cue without declaring whether it originated from
hardware or from replay. The interface labels simulated feeds at all times.

> This is not a hedge — it is the correct engineering posture and it is
> defensible under questioning. The classification and fusion layers are the
> substantive intellectual work; a φ-OTDR interrogator is commodity hardware
> procured, not invented.

---

## 16. The Query Router

A cheap pattern classifier sits in front of four backends so that the expensive
path is taken only when genuinely required.

| # | Route | Trigger pattern | Backend | Latency |
|---|---|---|---|---|
| 1 | **Structured** | Dates, times, camera IDs, plate regex, counts, "last"/"first" | PostgreSQL + FTS5 | < 5 ms |
| 2 | **Identity** | Track ID, "this person", face upload, "who is" | Milvus + OSNet / SFace | < 20 ms |
| 3 | **Archive semantic** | Open-vocabulary description + time range | Tier 7 clip index | ~100 ms |
| 4 | **Frame grounding** | "where in this frame", spatial disambiguation | LocateAnything-3B | async, seconds |

**The routing itself is the demonstration.** *"When was the last threat
found?"* is route 1 — a SQL query — not a vision-language model invocation.
Sending it to a 3-billion-parameter model would take four orders of magnitude
longer and produce a worse answer. Routing it correctly shows the team
understands where the expensive component belongs.

Queries are constructed from a fixed set of parameterised templates. Operator
language selects a template and fills bound parameters; it never becomes SQL.
This removes injection as a category and means every query the system can
execute is one that has been reviewed.

---

## 17. Evidence, Audit and Provenance

### 17.1 What every alert carries

| Element | Purpose |
|---|---|
| Keyframe | Written from the **raw** buffer, never the annotated one. Evidence must be the input the decision was made on, not a rendering of the decision. |
| Subject crop | Immediate visual identification |
| Video clip | The run-up, not just the moment. H.264, browser-playable. |
| All supporting detections | Each with its own confidence and originating model ID |
| Model ID map | Detector, tracker, embedder, pose, VLM — each with a weights hash, so swapping a file with the same name is detectable |
| Triggering geometry | Which line, which direction, which polygon, distance in metres |
| Homography identifier | So a metric claim can be independently re-derived |
| Configuration snapshot | The exact thresholds in force at the moment of firing |
| Supporting observations | The frames that satisfied temporal confirmation |
| Rationale | One human-readable line explaining why this fired |

### 17.2 Dual-axis rendering

```
   THREAT  →  COLOUR                     CONFIDENCE  →  STYLE
   ─────────────────────                 ──────────────────────
   INFO       blue                       HIGH      solid line
   FRIENDLY   green                      MEDIUM    dashed line
   WATCH      amber                      LOW       hatched fill + dotted
   WARNING    orange
   CRITICAL   red
```

The axes are independent fields and are never multiplied into a single score.

A **low-confidence CRITICAL** renders as a red hatched box: *this would be
serious, and I am not certain.* A **high-confidence WATCH** renders as a solid
amber box: *this is routine, and I am certain.* Collapsing them into one
variable makes the display ambiguous about whether the system is unsure or
unconcerned — which is exactly the distinction an operator needs.

### 17.3 The threat matrix

Threat level is determined by observation kind against identity verdict. The
matrix lives in configuration, not code, precisely so that personnel who
understand the operational domain can review and amend it.

| Observation | FRIENDLY | UNIDENTIFIED | INSUFFICIENT_QUALITY | No identity |
|---|---|---|---|---|
| Tripwire crossing | FRIENDLY | **CRITICAL** | WARNING | WARNING |
| Zone presence | FRIENDLY | WARNING | WATCH | WATCH |
| Proximity breach | INFO | WARNING | WATCH | WATCH |
| Unattended baggage | — | — | — | WARNING |
| Stance anomaly | WATCH | WARNING | WATCH | WATCH |
| Aerial contact | — | — | — | WARNING → CRITICAL on hover |
| Camera health fault | — | — | — | WARNING |
| Sensor cue | — | — | — | WATCH |

---

## 18. Performance and Measurement

All figures measured on: **NVIDIA RTX 5080** (Blackwell, sm_120, 16 GB),
PyTorch 2.11.0+cu128, Python 3.13, Windows 11. Source: 1280×720, 15 fps,
~97 kbps monochrome IR.

### 18.1 Throughput

| Stage | Measured |
|---|---|
| End-to-end pipeline | **30–39 FPS** |
| Detection | ~12.4 ms/frame |
| Tracking | ~9.1 ms/frame |
| Spatial + temporal reasoning | **0.14 ms/frame** |
| Total frame time | ~27–31 ms |

Spatial and temporal reasoning — homography transformation, tripwire crossing,
zone containment, proximity, dwell timers, association, confirmation, alert
lifecycle — cost **0.14 ms**. The intelligence layer is effectively free
relative to inference.

### 18.2 Component latency

| Component | Measured |
|---|---|
| Camera tamper detection | **0.20 s** from obscuration |
| Structured query (F10) | **0.06–7.5 ms** |
| Grounding query (F11) | **620 ms – 1.5 s** per keyframe |
| Grounding model cold load | 31.5 s |
| Grounding peak VRAM | **9.84 GB** |

### 18.3 Accuracy and correctness

| Measurement | Value |
|---|---|
| Homography round-trip residual | **0.0000 m** |
| OSNet same-subject similarity | min 0.7068 · p5 0.7752 · median 0.8941 (n = 780) |
| OSNet negative-class similarity | p95 0.5564 · max 0.5711 (n = 2,400) |
| Fitted similarity threshold | **0.666** |
| Stance classification, edge-clipped frames | 100% correctly suppressed to UNKNOWN |
| Relay payload | max **2,948 bytes** against a 3,072-byte budget |
| Test suite | **48 tests, 1.29 s**, no GPU required |

### 18.4 Benchmarks reported

Each of the following is reported as a measured comparison on the low-light
evaluation set, with the metric named:

| Comparison | Metrics |
|---|---|
| YOLO26-s vs RF-DETR-S | mAP@50, mAP@50-95, latency |
| BoT-SORT vs ByteTrack vs FastTracker | HOTA, IDF1, ID-switches, FPS |
| SAHI enabled vs disabled | mAP on small/distant targets specifically |
| Low-light-trained detector vs enhanced input | mAP delta on ExDark and LLVIP |
| RGB-T fusion vs RGB alone | mAP on LLVIP |
| Natural-language retrieval | SurveillanceVQA-589K |

These comparison tables are the substance of the technical claim: they are
reproducible, they use public benchmarks with published baselines, and they are
independently checkable.

---

## 19. Failure Modes and Resilience

| Failure | System response |
|---|---|
| Camera feed drops | Automatic reconnection with backoff; `SIGNAL_LOST` health state; alert raised |
| Camera lens obscured | `TAMPERED` within 0.20 s; alert raised; sector marked degraded |
| Camera repositioned | SSIM drift detected; homography flagged invalid; distances marked non-metric |
| Frame rate collapses | Confirmation is ratio-based and remains valid; queue drops counted and displayed |
| Perception stalls | Bounded decode queue drops oldest frames; latency preserved; drop rate visible |
| VLM already busy | Request rejected with HTTP 503; structured route unaffected |
| GPU exhausted | Grounding reports unavailable with reason; perimeter detection continues |
| Disk full | Evidence queue blocks and backpressures; alerts are not emitted without evidence |
| Camera uncalibrated | Geometry falls back to pixel space, **flagged `degraded`**; confidence capped; distances never presented as metres |
| Model weights missing | Component reports unavailable with an explicit reason; system starts and runs without it |
| Network absent | No effect. The system has no network dependency at runtime. |
| One bad frame | Caught per-frame; pipeline continues; error surfaced |

**The governing pattern is graceful degradation, never collapse.** Each
component reports its own unavailability with a specific reason rather than
failing silently or, worse, producing confident output from insufficient input.

---

## 20. Deployment Architecture

### 20.1 Single-post deployment

```
  [ n CCTV cameras ] ──RTSP──► [ MediaMTX ] ──► [ SENTINEL workstation ]
                                                        │
                                    ┌───────────────────┼───────────────────┐
                                    ▼                   ▼                   ▼
                            [ Operator C2 ]     [ SQLite/Postgres ]  [ MQTT relay ]
                              browser, LAN         evidence store      field units
```

One workstation. One GPU. No internet. Cameras on an isolated VLAN.

### 20.2 Sector deployment

Multiple posts each running a local instance, with alert-level replication to a
sector command node holding the fused common operating picture. Only alerts and
evidence references traverse the link — never raw video — which is what makes
the architecture viable over the poor backhaul documented in the domain
assessments.

### 20.3 Hardware

| Component | Minimum | Recommended |
|---|---|---|
| GPU | 8 GB VRAM | 16 GB+ (grounding VLM peaks at 9.84 GB) |
| CPU | 6 cores | 12+ cores for multi-camera decode |
| RAM | 16 GB | 32 GB+ |
| Storage | 512 GB SSD | 2 TB+ NVMe for evidence retention |
| Network | None required at runtime | Gigabit LAN for camera aggregation |

### 20.4 Edge scaling

For higher stream counts, NVIDIA DeepStream or Savant provides N-stream scaling
on the same tier structure. The pipeline is already per-camera and
thread-isolated.

---

## 21. Security, Privacy and Ethics

### 21.1 Data sovereignty

Video never leaves the installation. No cloud inference, no external API calls,
no telemetry. The system runs air-gapped and is verified to do so — model
loaders refuse network access at runtime, and exactly one build-time script is
permitted to fetch weights.

### 21.2 Identity handling

- Enrolment covers **friendly personnel only**, with consent, under
  organisational control
- No adversary classification. No nationality inference. No demographic
  inference.
- The system reports *Unidentified*, which is a statement about enrolment, not
  a statement about intent
- Embeddings are stored, not raw biometric imagery, in the identity index
- Every identity verdict records its similarity score and threshold, so any
  match can be reviewed and contested

### 21.3 Auditability

Every alert is immutable, timestamped in wall-clock UTC, and carries the
complete provenance chain from raw frame to decision, including the exact
software configuration in force. An alert can be reconstructed and re-litigated
months later.

### 21.4 The refusal

The system does **not** implement adversary-nation uniform classification, and
this is a deliberate technical judgement rather than a scope limitation. No
legitimate open dataset exists at CCTV resolution; GPU time is not the
constraint, data is. A closed-set classifier answers *"what about a civilian in
dark clothing?"* badly and confidently. Open-set formulation is cheaper to
build, honest about what it knows, and a substantially stronger position to
defend.

**Documenting a refusal is part of the engineering.**

### 21.5 Query safety

Operator queries never become SQL. A fixed template set with bound parameters
removes injection as a category. Evidence file serving is path-contained to the
evidence store.

---

## 22. Licensing Analysis

For a system whose deliverable is deployed by a security agency, *"can this
actually be fielded?"* is a fair question and one that few competing
submissions will have considered.

| Component | Licence | Deployment implication |
|---|---|---|
| YOLO26 / YOLO26-Pose, BoT-SORT, ByteTrack | AGPL-3.0 | Requires open-sourcing the derivative work, or an Ultralytics enterprise licence |
| RF-DETR-S | Apache-2.0 | Unrestricted |
| OSNet | MIT | Unrestricted |
| YuNet, SFace | Apache-2.0 | Unrestricted. Chosen specifically to replace InsightFace packs, which are non-commercial research only. |
| PP-OCRv6 | Apache-2.0 | Unrestricted |
| SAHI | MIT | Unrestricted |
| LocateAnything-3B | NVIDIA Open Model Licence | Non-commercial: academic and non-profit research |
| Qwen3-VL | Apache-2.0 | Unrestricted |
| Milvus, ChromaDB, FastAPI, OpenCV, Shapely | Apache-2.0 / BSD / MIT | Unrestricted |
| PostgreSQL, SQLite | PostgreSQL Licence / public domain | Unrestricted |

**Two fully-permissive deployment configurations exist**, both supported by the
interface architecture:

1. **Permissive perimeter stack** — RF-DETR + YuNet/SFace + OSNet + PP-OCRv6.
   Every component Apache-2.0 or MIT. Detector swaps behind the existing
   `Detector` protocol without touching downstream code.
2. **Permissive reasoning stack** — Qwen3-VL replaces LocateAnything-3B behind
   the same grounding interface.

The interface boundaries exist precisely so that licence posture is a
deployment decision rather than an architectural commitment.

---

## 23. Comparison with Prior Systems

| Dimension | SBInet | CIBMS pilots | Typical hackathon entry | **SENTINEL** |
|---|---|---|---|---|
| Alert precision | ~2% actionable | False positives from wildlife/weather unsolved | Undiscussed | Temporal confirmation, per-zone group filtering, deliberate suppression bias |
| Spatial reasoning | Sensor-level | Sensor-level | Pixel-space bounding boxes | **Metric ground coordinates** via per-camera homography |
| Sensor fusion | Failed | **Named as the hardest problem, unsolved** | Absent | Single typed cue interface; visual + acoustic + thermal |
| Uncertainty | Binary alerts | Binary alerts | Confidence number, if any | **Two independent axes**, distinct rendering |
| Identity | — | — | Closed-set classifier | **Open-set**, never asserts hostility |
| Auditability | Limited | Limited | Absent | **Complete provenance per alert** |
| Camera tamper | Not addressed | Equipment failure documented | Absent | 0.20 s detection, three independent signals |
| Aerial threat | Not addressed | Gap documented in policy literature | Absent | Detection + multi-frame flight-behaviour discrimination |
| Degraded input | Failed in adverse conditions | Failed in adverse conditions | Demonstrated on clean video | **Designed for it; validated on it** |
| Field maintenance | High cost, poor spares | Explicitly deterred use | N/A | Existing cameras; passive fibre; one workstation |
| Deployment honesty | — | — | Overstated | Licence analysis, provenance discipline, documented refusals |

---

## 24. Repository Structure

```
configs/
  system.yaml                every threshold, with reasoning in comments
  profiles/demo.yaml         15 s dwell, short confirmation — stage values
  profiles/operational.yaml  120 s dwell, strict confirmation — field values
  cameras/*.yaml             one file per camera: source, homography, zones

src/sentinel/
  contracts/       shared data shapes. Imports nothing else in the tree.
    frame.py         FrameBundle, HealthStatus
    detection.py     Detection, Track, TrackPoint, IdentityVerdict
    observation.py   Observation, ObservationKind, CrossDirection
    alert.py         Alert, ThreatLevel, ConfidenceBand, EvidenceRef
    cue.py           SensorCue — required `simulated` flag

  config/          typed schema, profile overlay composition
  ingest/          source, health (Laplacian + SSIM), operator-stream enhancement
  calibration/     homography, zones (tripwire, polygon, proximity ring)
  perception/      detector + tracker protocols, YOLO adapter, pose
  spatial/         Tier 2 — pure geometry, no models
  temporal/        Tier 3 — confirmation, association, dwell. No GPU.
  identity/        OSNet, quality gate, friendly gallery, cross-camera
  alerting/        manager (sole Alert constructor), evidence, relay
  storage/         schema + repository
  query/           router, SQL route, identity route, grounding route
  pipeline/        bounded queues, per-camera runner
  api/             FastAPI, MJPEG stream, WebSocket
  ui/              C2 interface

tools/
  calibrate.py          interactive 4-point ground calibration
  enroll.py             friendly personnel enrolment
  measure_threshold.py  fit the identity threshold to measured distributions
  degrade.py            synthesise legacy-CCTV degradation from clean footage
  render_demo.py        annotated stills and video

scripts/
  bootstrap.py     fresh machine to running system, one command
  fetch_weights.py the only component permitted network access
  verify_env.py    hardware, driver, weights and configuration gate

tests/             48 tests. No GPU, no video, no weights required.
```

### Architectural invariant

`contracts/` imports nothing from the rest of the tree. `spatial/` and
`temporal/` import only `contracts/`. This is what allows the suppression
policy — the component determining whether the system is operationally useful
or operationally ignored — to be verified in 1.29 seconds without a GPU.

---

## 25. Operational Workflow

### 25.1 Installation, per camera

1. Point the camera. Note the RTSP URL.
2. Run the calibration tool. Click four ground-plane points. Enter their
   real-world coordinates. **Under two minutes.**
3. Draw the tripwire, restricted zones, and proximity rings in the same metric
   coordinate frame.
4. Optionally mark static masks over sky, structures, or adjacent property.
5. The residual is reported. A non-zero residual means the points were not
   coplanar; re-mark.

### 25.2 Personnel enrolment

Capture several images per individual across pose and lighting. The quality
gate rejects crops that cannot support a claim and reports why. The similarity
threshold is fitted against measured distributions on the deployment's own
footage — not carried over from another site.

### 25.3 Daily operation

The operator sees one screen: live picture with overlays, an alert stream, an
evidence drawer, and a system telemetry panel showing per-camera health, frame
rate, and queue drop counters.

Clicking an alert opens its complete evidence record. Typing a question routes
to the cheapest backend capable of answering it.

### 25.4 Profile selection

Demonstration and operational profiles differ in dwell thresholds,
confirmation strictness, and identity threshold. Both ship. Switching is a
command-line flag, and the active profile is recorded into every evidence
record.

---

## 26. Scaling Path

| Stage | Cameras | Architecture |
|---|---|---|
| Single post | 1–8 | One workstation, SQLite, in-process state |
| Multi-post | 8–32 | DeepStream/Savant N-stream, PostgreSQL, Redis state |
| Sector | 32–200 | Per-post instances, alert replication to sector command, fused common operating picture |
| Regional | 200+ | Hierarchical aggregation; only alerts and evidence references cross links |

Raw video never traverses a wide-area link. This is what makes the architecture
viable over the backhaul quality documented in the domain literature.

---

## Appendix A — Configuration Reference

Every threshold in the system, with the demonstration and operational values.

| Key | Demo | Operational | Meaning |
|---|---|---|---|
| `detector.model` | yolo26s.pt | yolo26s.engine | Detector weights |
| `detector.imgsz` | 960 | 960 | Inference resolution |
| `detector.conf` | 0.25 | 0.30 | Detection floor — low deliberately; precision is applied in Tier 3 |
| `detector.sahi.enabled` | false | per camera | Sliced inference |
| `detector.sahi.tile` | 640 | 640 | Tile size |
| `detector.sahi.overlap` | 0.2 | 0.2 | Tile overlap fraction |
| `tracker.kind` | botsort | botsort | Tracker selection |
| `tracker.track_buffer` | 60 | 60 | Frames a lost track survives |
| `tracker.match_thresh` | 0.8 | 0.8 | Association threshold |
| `health.blur_min` | 55.0 | 55.0 | Laplacian variance floor |
| `health.ssim_min` | 0.55 | 0.55 | Structural drift floor |
| `health.consecutive_bad` | 3 | 8 | Frames before state change |
| `health.check_within_s` | 1.0 | 1.0 | Tamper detection latency target |
| `enhance.mode` | clahe | clahe+dcp | Operator-stream correction |
| `confirm.window_s` | 1.2 | 3.0 | Confirmation window |
| `confirm.min_ratio` | 0.6 | 0.7 | Required hit ratio |
| `confirm.min_hits` | 2 | 6 | Minimum observations |
| `baggage.assoc_radius_m` | 2.5 | 2.5 | Bag→owner association radius |
| `baggage.detach_distance_m` | 4.0 | 4.0 | Separation starting the timer |
| `baggage.dwell_s` | **15** | **120** | Unattended threshold |
| `baggage.reclaim_grace_s` | 2.0 | 5.0 | Hold period before clearing |
| `identity.sim_threshold` | 0.666 | 0.666 | Fitted, not chosen |
| `identity.min_crop_h` | 96 | 96 | Quality gate |
| `alerts.clear_after_s` | 5.0 | 15.0 | Silence before clearing |
| `alerts.expire_after_s` | 240 | 1800 | Hard ceiling |
| `relay.max_bytes` | 3072 | 3072 | Tactical payload budget |
| `pipeline.q_decode` | 4 | 4 | Decode queue depth |
| `pipeline.q_alerts` | 256 | 256 | Alert queue depth |

---

## Appendix B — API Reference

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | Operator C2 interface |
| `GET` | `/api/stream/{camera_id}` | MJPEG live video, `multipart/x-mixed-replace` |
| `WS` | `/ws` | Live state: tracks, threat, confidence, health, queue counters |
| `GET` | `/api/status` | System, per-camera, queue and model status |
| `GET` | `/api/zones/{camera_id}` | Zone geometry, metric and pixel-projected |
| `GET` | `/api/alerts` | Live and recent alerts |
| `GET` | `/api/alerts/{alert_id}` | Complete evidence record |
| `GET` | `/api/stats` | Aggregate counts, relay statistics |
| `POST` | `/api/query` | Multi-modal query router |
| `GET` | `/evidence/{path}` | Keyframes, crops, clips — path-contained |

---

## Appendix C — Glossary

| Term | Definition |
|---|---|
| **Footpoint** | Bottom-centre of a bounding box; the ground-contact estimate transformed into world coordinates |
| **Homography** | 3×3 projective transform mapping the image plane to the ground plane |
| **Tripwire** | A directional line in ground coordinates. Crossing direction distinguishes infiltration from exfiltration. |
| **Observation** | A cheap, frequent, disposable statement that something occurred this frame. Never operator-visible alone. |
| **Alert** | A confirmed, evidenced, operator-visible event. Constructed only by the alert manager. |
| **Confirmation** | N-of-M temporal test in wall-clock seconds and hit-ratio, invariant to frame rate |
| **Open-set** | Identity formulation reporting membership of an enrolled set, with *not enrolled* as a valid answer, rather than classifying into fixed categories |
| **Quality gate** | Test determining whether a crop can support an identity claim at all |
| **Dual-axis** | Threat determines colour; confidence determines line style. Never combined. |
| **SAHI** | Slicing Aided Hyper Inference — tiled inference recovering small distant targets |
| **DAS** | Distributed Acoustic Sensing — optical fibre as a linear array of virtual microphones |
| **φ-OTDR** | Phase-sensitive Optical Time-Domain Reflectometry; the DAS interrogation method |
| **HSRP** | High Security Registration Plate — the Indian standard vehicle plate format |
| **IDF1 / HOTA** | Multi-object tracking metrics measuring identity preservation |
| **CIBMS** | Comprehensive Integrated Border Management System |
| **BOP** | Border Out Post |

---

*SENTINEL — Smart India Hackathon 2026 · Problem Statement SIH26187*
