# SENTINEL — the four demonstrations

Each demo exercises a different part of the system, and each one is chosen
because it shows something a naive implementation gets wrong. The interesting
half of every demo is not that it fires — it is *what it refuses to claim*.

Run everything from the repository root with the demo profile:

```bash
python run.py
```

Then open <http://127.0.0.1:8000>, pick a source, and go to the analysis page.

The demo profile (`configs/profiles/demo.yaml`) is what turns these on. In
`configs/system.yaml` the pose model and both specialised detectors are **off**,
because each costs runtime and each is a claim the system should not make by
default.

---

## Demo 1 — Video tampering

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

## Demo 2 — Weapon detection

**What it shows.** A specialised detector answering a question the perimeter
model cannot: *is that person carrying something.*

**Footage.** Any clip with a visible firearm, or the device camera with a
prop. The device camera is the easier live demo — see below.

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
> mAP@50 **0.802**. Recall against false-positive cost is a tunable trade, and
> both halves are recorded above the `conf` setting in `configs/system.yaml`.
> At the configured 0.60 the detector recalls 59.4% of real weapons and draws
> roughly two spurious boxes across 260 frames of weapon-free footage — none of
> which survive temporal confirmation into an alert.

**Honesty note.** The first weapons dataset tried here produced mAP@50 of
0.009. It was not a training failure: 36% of its boxes fell outside the image,
one class had 7 training instances against 276 in validation, and drawing the
converted labels back onto their images put boxes beside the weapons. The
lesson is written into `tools/train_detector.py` as a rule — always draw
converted labels back onto their images before training on them.

---

## Demo 3 — Drone detection

**What it shows.** A class COCO does not have, and the discrimination that
makes it useful rather than annoying.

**Footage.** Any clip containing a small UAV against sky.

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

This is the part of the system least likely to be exercised by a demo, since it
only fires when a bird happens to fly through frame, so it is tested directly
in `tests/test_threat.py` rather than left to luck.

**Data note.** The first drone dataset tried here was 21 GB of simulator
renders shipped as RGBA instance masks with no boxes. Boxes are recoverable
from masks, but a detector trained only on renders is asked at demo time to
find a real drone in real footage, and nothing in its training distribution
looks like that. The set in use is real imagery curated from 23 open datasets.

---

## Demo 4 — Animal versus crawling human

**What it shows.** The single hardest discrimination in this problem domain,
and the one where box geometry is provably useless.

**Footage.** A person crawling, and any animal. The device camera works well
for the crawling half.

**What to watch for.** A subject the detector calls an animal, reclassified to
`HUMAN` with the evidence that justified it — or a low-slung person detection
where the system **withholds** the stance claim because no human skeleton
resolves.

**The part worth explaining.** A person on hands and knees and a dog are the
same shape. Aspect ratio cannot separate them, and neither can the class label:
a detector trained on upright people frequently calls a crawling person an
animal, and occasionally calls a large dog a person.

So above a configured aspect ratio the class label is not trusted, and the pose
model is asked whether it sees a person. The signal is the pose model's *own*
person confidence rather than any statistic computed from its keypoints — it is
trained to detect people and nothing else, so its confidence already encodes
"is this a human body" better than a hand-rolled aggregate. An earlier version
scored the mean joint confidence weighted by torso-chain completeness, and it
did not separate: real people scored a median of 0.36, straddling any threshold
worth setting.

> **Measured** on 121 person crops against 1,533 animal crops (COCO cats and
> dogs):
>
> | | p5 | median | p95 |
> |---|---|---|---|
> | people | 0.208 | 0.826 | 0.939 |
> | animals | 0.000 | 0.000 | 0.633 |
>
> The animal median of **zero** is the result that matters: on most animal
> crops the pose model returns no person at all. The configured threshold of
> 0.37 keeps 90% of people and admits 11% of animals at the crop level.
>
> End to end, the number is better: the check also requires five visible
> joints, and across 1,086 real animal images only **25 of 807** eligible
> animal tracks were falsely reclassified as human — **3.1%**, per frame,
> against a stance alert that needs five corroborating hits over three
> seconds.

---

## Live input — the device camera

Demos 2 and 4 are strongest live. Pick the device camera as the source; the
transport bar hides itself, because pause and seek are meaningless on a live
feed and offering controls that silently do nothing is worse than offering
none.

Everything else behaves identically — the same detector, the same confirmation
windows, the same evidence records.

---

## What to say if something does not fire

Say what the system measured. Every alert carries its evidence, every threshold
lives in `configs/system.yaml` rather than in code, and the panel shows the
withheld candidates alongside the confirmed ones. A system that can show you
*why it declined* is making a stronger claim than one that only shows hits.
