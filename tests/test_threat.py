"""Bird-versus-drone resolution.

This is the part of the specialised-detector path most likely to be wrong and
least likely to be exercised by a demo, because it only fires when a bird
happens to fly through frame. Test it directly rather than hoping.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from sentinel.config import AerialCfg, WeaponCfg
from sentinel.contracts import Detection, ObjectGroup, Track
from sentinel.perception import resolve_bird
from sentinel.perception.threat import WeaponDetector, _iou

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def bird(bbox, conf=0.8) -> Detection:
    return Detection(bbox_px=bbox, class_id=14, class_name="bird",
                     group=ObjectGroup.ANIMAL, confidence=conf, model_id="yolo26s")


@pytest.fixture
def cfg() -> AerialCfg:
    return AerialCfg()


def test_iou_identical_boxes_is_one():
    assert _iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)


def test_iou_disjoint_boxes_is_zero():
    assert _iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_half_overlap():
    # Two 10x10 boxes sharing a 5x10 strip: intersection 50, union 150.
    assert _iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)


def test_no_birds_means_no_withholding(cfg):
    withhold, rival, iou = resolve_bird((0, 0, 10, 10), 0.9, [], cfg)
    assert withhold is False
    assert rival is None
    assert iou == 0.0


def test_confident_overlapping_bird_withholds_the_drone(cfg):
    withhold, rival, iou = resolve_bird((0, 0, 10, 10), 0.5,
                                        [bird((0, 0, 10, 10), conf=0.8)], cfg)
    assert withhold is True
    assert rival is not None
    assert iou == pytest.approx(1.0)


def test_bird_elsewhere_in_frame_does_not_withhold(cfg):
    """A bird on the other side of the sky is not evidence about this box."""
    withhold, _rival, iou = resolve_bird((0, 0, 10, 10), 0.5,
                                         [bird((500, 500, 520, 520), conf=0.95)], cfg)
    assert withhold is False
    assert iou == 0.0


def test_more_confident_drone_beats_a_weaker_bird(cfg):
    """The point of the rule is that the more confident model wins, not that
    any bird at all vetoes a drone."""
    withhold, _rival, _iou = resolve_bird((0, 0, 10, 10), 0.90,
                                          [bird((0, 0, 10, 10), conf=0.40)], cfg)
    assert withhold is False


def test_margin_can_be_tuned_to_favour_the_drone(cfg):
    """bird_margin > 1 demands the bird beat the drone by that factor."""
    strict = AerialCfg(bird_margin=2.0)
    b = [bird((0, 0, 10, 10), conf=0.8)]
    assert resolve_bird((0, 0, 10, 10), 0.5, b, cfg)[0] is True     # 0.8 >= 0.5
    assert resolve_bird((0, 0, 10, 10), 0.5, b, strict)[0] is False  # 0.8 < 1.0


def test_the_strongest_overlapping_bird_is_the_rival(cfg):
    """With several birds in frame the decision must use the overlapping one,
    not merely the most confident one."""
    birds = [bird((500, 500, 520, 520), conf=0.99), bird((0, 0, 10, 10), conf=0.6)]
    withhold, rival, iou = resolve_bird((0, 0, 10, 10), 0.5, birds, cfg)
    assert withhold is True
    assert rival is not None and rival.confidence == pytest.approx(0.6)
    assert iou == pytest.approx(1.0)


def test_grazing_overlap_below_iou_threshold_does_not_withhold(cfg):
    """A bird clipping the corner of a drone box is not the same object."""
    # 10x10 boxes overlapping by 2x10: intersection 20, union 180 -> IoU 0.11.
    withhold, _rival, iou = resolve_bird((0, 0, 10, 10), 0.5,
                                         [bird((8, 0, 18, 10), conf=0.99)], cfg)
    assert iou < cfg.bird_iou
    assert withhold is False


# ---------------------------------------------------------------------------
# The sticky-label regression.
#
# `Track.meta` lives for the life of the track. The first version of the weapon
# detector wrote `meta["weapon"]` on a hit and never cleared it, so one frame's
# false positive marked a person as armed permanently -- surviving the weapon
# leaving frame, the person putting their hands down, and even the confidence
# threshold being raised afterwards. It inflated the observed false-positive
# count 33x (99 labelled boxes in 260 frames, against a true 3) and hid the fact
# that raising `conf` was having no effect at all.
# ---------------------------------------------------------------------------

class _NoBoxes:
    """An ultralytics-shaped result carrying no detections."""
    boxes = None
    keypoints = None


class _StubModel:
    names = {0: "rifle"}

    def __init__(self, results):
        self._results = results

    def predict(self, *_a, **_kw):
        return [self._results]


def _armed_person(tid: int = 1) -> Track:
    det = Detection(bbox_px=(100, 100, 200, 400), class_id=0, class_name="person",
                    group=ObjectGroup.PERSON, confidence=0.9, model_id="test")
    return Track(track_id=tid, camera_id="cam01", group=ObjectGroup.PERSON,
                 class_name="person", latest=det, first_seen=T0, age_frames=30)


def _detector(result) -> WeaponDetector:
    """A WeaponDetector with a stubbed model -- no weights on disk needed."""
    d = object.__new__(WeaponDetector)
    d.cfg = WeaponCfg()
    d.model = _StubModel(result)
    d.model_id = "stub"
    d.device = "cpu"
    d.class_names = {0: "rifle"}
    return d


def test_a_weapon_label_does_not_outlive_the_detection():
    d = _detector(_NoBoxes())
    t = _armed_person()
    t.meta["weapon"] = "rifle (0.91)"          # last frame said armed
    frame = np.zeros((480, 640, 3), np.uint8)

    d.scan([t], frame, "cam01", 1, T0)

    assert "weapon" not in t.meta, (
        "a weapon label must not survive a frame in which no weapon was seen")


def test_the_label_is_cleared_even_for_people_beyond_the_crop_budget():
    """Only `max_crops_per_frame` people are examined each frame. The rest must
    still lose a stale label, or the budget itself becomes a way to be
    permanently armed."""
    d = _detector(_NoBoxes())
    d.cfg = d.cfg.model_copy(update={"max_crops_per_frame": 1})
    tracks = [_armed_person(1), _armed_person(2), _armed_person(3)]
    # Make #1 the tallest so it is the one inside the budget.
    tracks[0].latest.bbox_px = (100, 0, 200, 470)
    for t in tracks:
        t.meta["weapon"] = "rifle (0.91)"

    d.scan(tracks, np.zeros((480, 640, 3), np.uint8), "cam01", 1, T0)

    assert all("weapon" not in t.meta for t in tracks)


# ---------------------------------------------------------------------------
# The frame-area ceiling.
#
# The drone training set is real imagery but it is not surveillance imagery:
# 44% of its boxes cover more than 2.25% of the image, because FPV and product
# shots dominate. The model learned from that, and on a dark wide shot it
# proposes a box covering half the frame at 0.57 confidence -- on the project's
# own demo clip that fired on 98.3% of frames at the original threshold.
#
# Raising confidence alone fixes it at a cost: 62.6% recall. Capping the box
# area instead keeps 68.9% and encodes something true -- a drone filling a
# third of a surveillance frame is metres from the lens, not a perimeter
# intrusion.
# ---------------------------------------------------------------------------

class _Vals:
    def __init__(self, a): self._a = np.asarray(a)
    def cpu(self): return self
    def numpy(self): return self._a


class _Boxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy, self.conf, self.cls = _Vals(xyxy), _Vals(conf), _Vals(cls)

    def __len__(self): return len(self.conf.numpy())


class _Result:
    def __init__(self, boxes): self.boxes = boxes


def _aerial(result, **cfg_overrides):
    from sentinel.perception.threat import AerialDetector
    d = object.__new__(AerialDetector)
    d.cfg = AerialCfg(**cfg_overrides)
    d.model = _StubModel(result)
    d.model_id = "stub"
    d.device = "cpu"
    d.class_names = {0: "drone"}
    d.withheld, d.detections = [], []
    return d


FRAME = np.zeros((720, 1280, 3), np.uint8)     # 921,600 px


def test_a_box_covering_half_the_frame_is_withheld():
    # 761 x 597 = 454,317 px = 49% of frame -- the real false positive observed.
    d = _aerial(_Result(_Boxes([[365, 105, 1126, 702]], [0.575], [0])))
    obs = d.scan([], FRAME, "cam01", 0, T0)

    assert obs == []
    assert len(d.withheld) == 1
    assert "49% of the frame" in d.withheld[0]["reason"]


def test_a_small_box_at_the_same_confidence_is_accepted():
    """The ceiling must reject on size, not quietly on confidence."""
    d = _aerial(_Result(_Boxes([[600, 200, 660, 240]], [0.575], [0])))
    obs = d.scan([], FRAME, "cam01", 0, T0)

    assert len(obs) == 1
    assert obs[0].confidence == pytest.approx(0.575)
    assert d.withheld == []
    assert len(d.detections) == 1


def test_the_ceiling_is_configurable_and_off_above_one():
    d = _aerial(_Result(_Boxes([[365, 105, 1126, 702]], [0.575], [0])),
                max_area_frac=1.01)
    obs = d.scan([], FRAME, "cam01", 0, T0)

    assert len(obs) == 1
    assert d.withheld == []


def test_an_oversized_box_never_reaches_the_bird_check():
    """Order matters: a frame-filling box is nonsense regardless of what the
    perimeter model thinks, and asking about birds first would let a scene with
    no birds in it through."""
    d = _aerial(_Result(_Boxes([[0, 0, 1280, 720]], [0.9], [0])))
    obs = d.scan([], FRAME, "cam01", 0, T0)      # no birds passed at all

    assert obs == []
    assert "ceiling" in d.withheld[0]["reason"]
