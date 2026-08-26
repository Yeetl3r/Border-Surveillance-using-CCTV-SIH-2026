# SENTINEL — the four demonstrations

Each demo exercises a different part of the system, and each one is chosen
because it shows something a naive implementation gets wrong. The interesting
half of every demo is not that it fires — it is *what it refuses to claim*.

Run everything from the repository root with the demo profile:

```bash
python run.py
```

Open <http://127.0.0.1:8000> and press **PROCEED WITH DEMO**, or go straight to
<http://127.0.0.1:8000/demos>. Each demo has its footage already bound; pick one
from the rail and it starts. `/demos#3` deep-links to a specific one, which is
how you get back mid-presentation without clicking through the others.

Selecting a demo starts only the cameras that demo needs and stops the rest.
Three pipelines each running the detector would leave the one being watched at a
third of its frame rate, and the demo is the only thing anyone is looking at.

Which clip belongs to which demo lives in `configs/demos.yaml` — data, not code,
so it can be reworded or repointed an hour before a presentation.

The demo profile (`configs/profiles/demo.yaml`) is what turns these on. In
`configs/system.yaml` the pose model and both specialised detectors are **off**,
because each costs runtime and each is a claim the system should not make by
default.

---

## Demo 1 — Weapon detection


**What it shows.** A specialised detector answering a question the perimeter
model cannot: *is that person carrying something.*

**Footage.** `assets/demo/weapon_patrol.mp4` — military personnel walking with
a rifle. Measured live on the demo page: the rifle is labelled on the carrier's
box at **0.85** confidence.

**What to watch for.** The weapon label appears on the person's box, and the
alert is graded by **who is carrying it**, not merely by what it is.

**The part worth explaining.** Two design decisions.

*It runs on person crops, not full frames.* A rifle at 80 m is a handful of
pixels in a 1920-wide frame; no confidence tuning recovers an object the
backbone never resolved. The person carrying it is already detected and already
tracked, and their box is a crop at which the weapon is legible.

*A weapon on a friendly is not an alert.* The threat matrix in
`configs/system.yaml` grades `WEAPON_VISIBLE` as `INFO` on an enrolled
friendly and `CRITICAL` on someone unidentified. A guard on patrol is armed.
That is the whole point of grading by identity rather than by object class.

> **Measured** on the held-out validation split (905 images, 1,062 weapons):
> mAP@50 **0.802**. At the configured 0.60 confidence the detector recalls
> 59.4% of real weapons and draws roughly two spurious boxes across 260 frames
> of weapon-free footage — none of which survive temporal confirmation into an
> alert.
>
> End to end through the pipeline the number that matters is different, because
> the detector only ever sees a person crop: across 193 held-out weapon images
> where a person was detected, **52.3%** raised a `WEAPON_VISIBLE` observation,
> against **5.9%** of person crops from weapon-free footage. Widening the crop
> beyond the person box from 15% to 30% bought seven of those points at no
> cost in false positives — a rifle held out to the side leaves the box that
> contains its carrier.

**Honesty note.** The first weapons dataset tried here produced mAP@50 of
0.009. It was not a training failure: 36% of its boxes fell outside the image,
one class had 7 training instances against 276 in validation, and drawing the
converted labels back onto their images put boxes beside the weapons. The
lesson is written into `tools/train_detector.py` as a rule — always draw
converted labels back onto their images before training on them.

---

## Demo 2 — Drone versus bird


**What it shows.** A class COCO does not have, and the discrimination that
makes it useful rather than annoying.

**Footage.** `assets/demo/bird_flight.mp4` (two birds) on top,
`assets/demo/drone_flight.mp4` (a quadcopter) below. The device camera works
too — but hold the drone at arm's length rather than up against the lens.
A candidate covering more than 35% of the frame is withheld by design (see
below), so a drone filling the whole webcam view will be declined, with the
reason printed. At roughly a metre from a laptop camera a small quadcopter
occupies about 2% of the frame, well inside the ceiling.

**What to watch for.** The drone is drawn with a reticle rather than a plain
box, because an aerial contact is not a ground contact. And if a bird is in
frame, watch for the **grey dashed box** labelled `BIRD` — that is the system
declining to call something a drone, with its reasoning printed on screen.

**The part worth explaining.** A drone detector that fires on every bird is
worthless above a border. The obvious fix is to train a `bird` class alongside
`drone`, and it is worse than it looks: the mixed drone/bird sets available
carry a few dozen bird boxes, which would produce a weak bird detector
competing against a strong one. The stock COCO model **already** carries bird
as class 14, trained on far more bird imagery than any drone set contains.

So the discrimination is made *between models*. The drone model proposes, the
perimeter model disputes, and the more confident wins. A withheld candidate is
deliberately **not** emitted as a zero-confidence observation — temporal
confirmation counts hits rather than confidence, so a flock of rejected birds
would confirm into exactly the alert the rejection existed to prevent. Withheld
candidates go to a display-only list.

**What the footage actually showed, and it is not flattering.** The drone
detector *cannot tell a bird from a drone on its own*: measured on the two demo
clips, it fires on **28% of frames of bird-only footage** against 36% of drone
footage, with a median confidence of 0.654 on the birds and 0.684 on the drone.
Trained on drone positives with no bird negatives, it is a small-object-in-sky
detector wearing a drone's name.

Everything that makes Demo 2 work is therefore outside that model, and both
halves had to be measured rather than assumed:

- **The veto's parameters were both wrong at their obvious values.** Requiring
  IoU ≥ 0.45 rejected most true matches, because a drone-trained model and a
  bird-trained model box the same bird differently — median best-bird IoU is
  0.391. And requiring the bird to be *more* confident than the drone failed,
  because the drone model is the systematically over-confident one. At 0.45 and
  a 1.0 margin only **14%** of bird-clip candidates were withheld; at 0.30 and
  0.25 it is about **76%**.
- **The veto alone still was not enough.** `AERIAL_INTRUSION` confirmation was
  raised from three hits at a 0.5 ratio to **eight hits at 0.75 over three
  seconds**. What separates the two in the end is persistence: a drone crossing
  frame is detected in most consecutive frames, while bird misfires are
  scattered.

> **Measured end to end**, alerts raised on 440 frames of bird-only footage:
>
> | veto margin | confirmation | alerts on birds | alert on the drone |
> |---|---|---|---|
> | 0.50 | 3 hits / 0.5 / 2.5 s | 1 | yes |
> | 0.25 | 3 hits / 0.5 / 2.5 s | 1 | yes |
> | **0.25** | **8 hits / 0.75 / 3.0 s** | **0** | **yes** |

This is the part of the system least likely to be exercised by chance, so it is
also tested directly in `tests/test_threat.py` rather than left to luck.

**The size ceiling.** Aerial training data is real imagery but it is not
*surveillance* imagery — FPV footage and product photography dominate, and 44%
of the boxes in the set used here cover more than 2.25% of the image. A model
trained on that will, on a dark wide shot, propose a box covering half the
frame at 0.57 confidence. On this project's own demo clip that fired on **98.3%
of frames**.

The fix is not a higher confidence threshold, which costs recall everywhere to
solve a problem that only occurs at one end of the size distribution. A
candidate covering more than a configured fraction of the frame is withheld
instead, on the grounds that a drone filling a third of a surveillance frame is
metres from the lens rather than crossing a border.

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

**A result worth stating.** An earlier checkpoint at epoch 30 (mAP@50 0.933)
needed only 0.50 confidence for the same silence on this footage. Training the
remaining 30 epochs raised mAP on the drone validation set to 0.955 and made
the model *more* confident on night-IR driveway footage containing no drones at
all — false frames went from 0.2% to 17% at the old threshold. Higher mAP is
not the same thing as a better detector for a given deployment, and here it was
the threshold rather than the metric that absorbed the difference.

**Data note.** The first drone dataset tried here was 21 GB of simulator
renders shipped as RGBA instance masks with no boxes. Boxes are recoverable
from masks, but a detector trained only on renders is asked at demo time to
find a real drone in real footage, and nothing in its training distribution
looks like that. The set in use is real imagery curated from 23 open datasets.

---

## Demo 3 — Video tampering


**What it shows.** The camera itself is a thing that can be attacked. Covering
a lens, defocusing it, or cutting the feed and replaying a still are all attacks
that leave the detector reporting a perfectly calm, perfectly empty scene.

**Footage.** `assets/tamper_test.mp4` — a hand moving across the lens.

**What to watch for.** The health strip escalates
`OK → DEGRADED → TAMPERED`, and the alert names which test tripped.

**The part worth explaining.** There is no absolute blur threshold, because
there cannot be one — an IR night scene and a daylight scene differ in
sharpness by more than a covered lens differs from an open one. The system
learns a per-camera baseline and looks for a *relative* collapse, and it
requires a **conjunction**: sharpness must fall against that camera's own
history *and* the scene must stop matching its own reference. Either alone is a
lighting change.

> **Measured** on the reference clip: TAMPERED at frame 571 (38.1 s), sustained
> for 388 frames, with zero false positives on the healthy portion.

The first two attempts at this failed on real footage, and both failures are
recorded in `src/sentinel/ingest/health.py`: an absolute blur floor does not
transfer between cameras, and a reference frame updated by an exponential
moving average quietly absorbs the tamper it is supposed to detect.

---

## Demo 4 — Animal versus crawling human


**What it shows.** The single hardest discrimination in this problem domain,
and the one where box geometry is provably useless.

**Footage.** `assets/demo/animal_elephants.mp4` (a herd) on top,
`assets/demo/human_crawl.mp4` (a person crawling) below.

**What to watch for.** The herd reads `ANIMAL` and stays that way. The crawling
subject reads `PERSON` — and its stance claim is **withheld**, with the reason
on the box, rather than asserted.

**The part worth explaining.** A person on hands and knees and a quadruped are
the same shape. Aspect ratio cannot separate them, and neither can the class
label: a detector trained on upright people frequently calls a crawling person
an animal. So above a configured aspect ratio the label is not trusted and the
pose model is asked whether it sees a person — using its *own* person
confidence rather than a statistic computed from its keypoints, because it is
trained to detect people and nothing else.

> **Measured on the demo footage itself**, which is the honest place to measure:
>
> | | elephant herd | crawling human |
> |---|---|---|
> | frames with a correct-group detection | **97%** | **41%** |
> | pose-confidence median | 0.087 | 0.250 |
> | falsely reclassified as human | **0.6%** (12 of 2,117) | — |

Three things in that table are worth saying out loud.

**The classification works, and it is the demo.** The herd is called an animal
in 97% of frames and a crawling human is never called one.

**Crawling is genuinely hard to detect at all.** At the global 0.25 confidence
the crawling subject is found in only 23% of frames; this camera runs at 0.10,
where it is 41%. That is a per-panel override in `configs/demos.yaml`, and the
same 0.10 on a perimeter camera would flood the tracker — sensitivity is a
property of the footage, not of the system.

**The pose signal barely separates these two, and the system is configured
around that rather than against it.** Against cats and dogs the split was clean
(people median 0.826, animals 0.000). Against *elephants* and a *crawling*
person it is not: 0.087 against 0.250, overlapping heavily. Lowering the
threshold to catch more crawling humans would start calling elephants human —
at 0.10 it would reclassify 47% of them. So the threshold stays where it
protects against the expensive error, the reclassification path stays at 0.6%
false, and when the pose model cannot resolve a skeleton the system **withholds
the stance claim instead of guessing**. Saying "this is a person, and I will not
tell you what posture they are in" is the correct answer here, and it is the
one it gives.

---

## Demo 5 — Live device camera


**What it shows.** The same pipeline on a feed with no recording behind it.

Nothing downstream changes for a live source — the detector, the confirmation
windows and the evidence records all behave exactly as they do for a file. The
transport bar hides itself, because pause and seek have no meaning on a live
feed and controls that silently do nothing are worse than none.

If no camera is attached, or another application already holds it, the demo
says so in as many words rather than presenting a black panel.

---

## What to say if something does not fire

Say what the system measured. Every alert carries its evidence, every threshold
lives in `configs/system.yaml` rather than in code, and the panel shows the
withheld candidates alongside the confirmed ones. A system that can show you
*why it declined* is making a stronger claim than one that only shows hits.
