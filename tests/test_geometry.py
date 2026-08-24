"""F3/F4/F5 -- homography, tripwire direction and metric proximity.

Pure geometry, no models, no video. These encode the difference between
"the subject is in the lower half of the image" and "the subject is 4.2 metres
from the camera" -- which is the substantive capability gap over pixel-space
geofencing.
"""

from __future__ import annotations

import math

import pytest

from sentinel.calibration.homography import Homography
from sentinel.calibration.zones import ProximityRing, RestrictedZone, Tripwire, ZoneSet
from sentinel.config import (
    CameraCfg,
    HomographyCfg,
    ProximityCfg,
    SpatialCfg,
    TripwireCfg,
    ZonePolygonCfg,
)
from sentinel.contracts import CrossDirection
from shapely.geometry import Point, Polygon


# A synthetic pinhole ground plane: f=640, camera height 2.2 m, cy=360.
#   Y_m = f*H/(y-cy)  ;  X_m = (x-cx)*Y_m/f
def ground(x: float, y: float) -> tuple[float, float]:
    Y = 640.0 * 2.2 / (y - 360.0)
    return ((x - 640.0) * Y / 640.0, Y)


IMG = [[350, 470], [900, 460], [980, 715], [120, 715]]
WLD = [list(ground(*p)) for p in IMG]


def make_h() -> Homography:
    return Homography.from_config(HomographyCfg(
        enabled=True, image_points=IMG, world_points=WLD,
        homography_id="test"))


# --------------------------------------------------------------- homography
def test_homography_is_exact_on_a_plane():
    """A ground plane maps to the image by an exact homography.

    A non-trivial residual means the marked points were not coplanar -- which
    is the single most common calibration mistake (clicking the top of a wall).
    """
    h = make_h()
    assert h.calibrated
    assert h.residual_m < 1e-6


def test_transform_matches_the_ground_model():
    h = make_h()
    for px, py in [(700, 600), (500, 550), (900, 690)]:
        got = h.to_metres((px, py))
        want = ground(px, py)
        assert got is not None
        assert math.isclose(got[0], want[0], abs_tol=1e-3)
        assert math.isclose(got[1], want[1], abs_tol=1e-3)


def test_uncalibrated_returns_none_not_a_guess():
    """None is the honest answer. A fabricated metre value is worse than none."""
    h = Homography.from_config(HomographyCfg(enabled=False))
    assert not h.calibrated
    assert h.to_metres((100, 100)) is None
    assert h.distance_m((0, 0), (10, 10)) is None


def test_metric_distance_is_depth_invariant():
    """The reason thresholds are in metres, not pixels.

    Two subjects the same real distance apart must measure the same whether
    they are near the camera or far from it. In pixels they would not.
    """
    h = make_h()
    near = h.distance_m((600, 700), (700, 700))
    far = h.distance_m((600, 480), (700, 480))
    assert near is not None and far is not None

    # Both pairs are 100 px apart in the image. In the world they are not:
    # ground scale grows linearly with depth, so the ratio of the two metric
    # distances must equal the ratio of their depths.
    depth_near = ground(650, 700)[1]
    depth_far = ground(650, 480)[1]
    assert math.isclose(far / near, depth_far / depth_near, rel_tol=1e-3)
    assert far > near * 2.5          # ~2.83x here -- a pixel threshold would
                                     # mean two different real distances


def test_inverse_round_trip():
    h = make_h()
    for m in [(0.0, 5.0), (2.0, 4.0), (-3.0, 8.0)]:
        px = h.to_pixels(m)
        assert px is not None
        back = h.to_metres(px)
        assert math.isclose(back[0], m[0], abs_tol=1e-3)
        assert math.isclose(back[1], m[1], abs_tol=1e-3)


# ----------------------------------------------------------------- tripwire
def wire(inbound_sign: int = -1) -> Tripwire:
    return Tripwire(id="w", label="W", a=(-6.0, 4.6), b=(6.0, 4.6),
                    space="metric", inbound_sign=inbound_sign)


def test_tripwire_detects_inbound_crossing():
    w = wire()
    assert w.crossing((0.0, 5.4), (0.0, 4.2)) is CrossDirection.INBOUND


def test_tripwire_detects_outbound_crossing():
    """Direction is the whole reason to use a tripwire over a polygon.

    Infiltration and exfiltration are operationally different events, and a
    containment test cannot distinguish them.
    """
    w = wire()
    assert w.crossing((0.0, 4.2), (0.0, 5.4)) is CrossDirection.OUTBOUND


def test_no_crossing_when_staying_on_one_side():
    w = wire()
    assert w.crossing((0.0, 5.4), (0.0, 5.0)) is None
    assert w.crossing((0.0, 4.2), (0.0, 3.0)) is None


def test_crossing_beyond_the_wire_ends_is_not_a_crossing():
    """Changing side of the infinite line is not the same as crossing the wire.

    A subject walking past the end of a 12 m fence has not crossed it.
    """
    w = wire()
    assert w.crossing((20.0, 5.4), (20.0, 4.2)) is None


def test_inbound_sign_flips_the_semantics():
    assert wire(-1).crossing((0.0, 5.4), (0.0, 4.2)) is CrossDirection.INBOUND
    assert wire(+1).crossing((0.0, 5.4), (0.0, 4.2)) is CrossDirection.OUTBOUND


def test_point_exactly_on_the_line_is_undecidable():
    w = wire()
    assert w.crossing((0.0, 4.6), (0.0, 4.2)) is None


# ---------------------------------------------------------------- proximity
def test_proximity_threshold_in_metres():
    ring = ProximityRing(id="a", label="A", geometry=Point(1.9, 3.94),
                         space="metric", threshold_m=1.5)
    breached, d = ring.breached((0.6, 4.0))
    assert breached and d < 1.5
    breached, d = ring.breached((0.4, 5.4))
    assert not breached and d > 1.5


def test_zone_containment():
    z = RestrictedZone(id="z", label="Z",
                       polygon=Polygon([(-6, 4.6), (6, 4.6), (6, 2), (-6, 2)]),
                       space="metric")
    assert z.contains((0.0, 3.5))
    assert not z.contains((0.0, 5.5))


# ------------------------------------------------------------------ zoneset
def camera_cfg() -> CameraCfg:
    return CameraCfg(
        camera_id="t", source="none.mp4",
        homography=HomographyCfg(enabled=True, image_points=IMG,
                                 world_points=WLD, homography_id="t"),
        tripwires=[TripwireCfg(id="w", points=[[-6, 4.6], [6, 4.6]],
                               space="metric", inbound_sign=-1, groups=["person"])],
        zones=[ZonePolygonCfg(id="z", points=[[-6, 4.6], [6, 4.6], [6, 2], [-6, 2]],
                              space="metric", groups=["person"])],
        proximity=[ProximityCfg(id="a", geometry="point", points=[[1.9, 3.94]],
                                space="metric", threshold_m=1.5,
                                groups=["person", "animal"])],
    )


def test_zoneset_builds_and_reports_metric():
    zs = ZoneSet.from_camera(camera_cfg())
    assert zs.metric
    assert len(zs.tripwires) == 1 and len(zs.zones) == 1 and len(zs.proximity) == 1


def test_uncalibrated_zoneset_degrades_rather_than_failing():
    """Graceful degradation: the test still runs, but says it is not metric."""
    cam = camera_cfg()
    cam.homography = HomographyCfg(enabled=False)
    zs = ZoneSet.from_camera(cam)
    assert not zs.metric
    pt, degraded = zs.resolve((700.0, 600.0), "metric")
    assert degraded is True
    assert pt == (700.0, 600.0)          # pixel fallback, explicitly flagged


def test_overlay_projects_metric_zones_back_to_pixels():
    zs = ZoneSet.from_camera(camera_cfg())
    ov = zs.overlay()
    assert ov["metric"] is True
    assert ov["tripwires"][0]["points_px"] is not None
    assert ov["proximity"][0]["points_px"] is not None
