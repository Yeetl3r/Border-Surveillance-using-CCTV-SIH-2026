"""Zone geometry: tripwires, restricted polygons and proximity rings.

Everything here is pure geometry with no dependency on models or video. Zones
are defined in ground metres when the camera is calibrated, and fall back to
pixel space when it is not -- but a pixel-space result is marked ``degraded``
so the UI can say so rather than quietly presenting pixels as metres.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import LineString, Point, Polygon

from ..config import CameraCfg, ProximityCfg, TripwireCfg, ZonePolygonCfg
from ..contracts import CrossDirection
from .homography import Homography


@dataclass(slots=True)
class Tripwire:
    """F4 -- a directional line.

    Direction is the whole reason to use a tripwire rather than a containment
    test. Infiltration and exfiltration are operationally different events, and
    a polygon cannot tell them apart.
    """

    id: str
    label: str
    a: tuple[float, float]
    b: tuple[float, float]
    space: str                    # "metric" | "pixel"
    inbound_sign: int             # sign of the cross product on the inside
    groups: list[str] = field(default_factory=list)

    def side(self, p: tuple[float, float]) -> float:
        """Signed area of the triangle (a, b, p). Sign = which side of the line."""
        return ((self.b[0] - self.a[0]) * (p[1] - self.a[1])
                - (self.b[1] - self.a[1]) * (p[0] - self.a[0]))

    def crossing(
        self, prev: tuple[float, float], cur: tuple[float, float]
    ) -> CrossDirection | None:
        """Did the segment prev->cur cross this line, and which way?

        Requires an actual sign change AND an intersection with the finite
        segment -- a track passing beyond the end of the wire has changed side
        of the infinite line without crossing the wire.
        """
        s_prev, s_cur = self.side(prev), self.side(cur)
        if s_prev == 0.0 or s_cur == 0.0:
            return None                       # exactly on the line: undecidable
        if (s_prev > 0) == (s_cur > 0):
            return None                       # same side, no crossing

        wire = LineString([self.a, self.b])
        if not wire.intersects(LineString([prev, cur])):
            return None                       # crossed the extension, not the wire

        ends_inside = (s_cur > 0) == (self.inbound_sign > 0)
        return CrossDirection.INBOUND if ends_inside else CrossDirection.OUTBOUND

    def to_dict(self) -> dict:
        return {"id": self.id, "label": self.label, "space": self.space,
                "points": [list(self.a), list(self.b)],
                "inbound_sign": self.inbound_sign, "groups": self.groups}


@dataclass(slots=True)
class RestrictedZone:
    """F4 -- containment. Distinct from a tripwire: no direction, just presence."""

    id: str
    label: str
    polygon: Polygon
    space: str
    kind: str = "restricted"
    groups: list[str] = field(default_factory=list)

    def contains(self, p: tuple[float, float]) -> bool:
        return self.polygon.contains(Point(p))

    def to_dict(self) -> dict:
        return {"id": self.id, "label": self.label, "space": self.space,
                "kind": self.kind, "groups": self.groups,
                "points": [list(c) for c in self.polygon.exterior.coords[:-1]]}


@dataclass(slots=True)
class ProximityRing:
    """F5 -- distance to a protected asset, thresholded in METRES not pixels.

    A metric threshold is depth-invariant: 3 m is 3 m whether the subject is
    near the camera or at the far end of the scene. A pixel threshold is not,
    which is why the pixel fallback is reported as degraded.
    """

    id: str
    label: str
    geometry: LineString | Point | Polygon
    space: str
    threshold_m: float
    groups: list[str] = field(default_factory=list)

    def distance(self, p: tuple[float, float]) -> float:
        return float(self.geometry.distance(Point(p)))

    def breached(self, p: tuple[float, float]) -> tuple[bool, float]:
        d = self.distance(p)
        return d <= self.threshold_m, d

    def to_dict(self) -> dict:
        if isinstance(self.geometry, Point):
            pts = [[self.geometry.x, self.geometry.y]]
        elif isinstance(self.geometry, Polygon):
            pts = [list(c) for c in self.geometry.exterior.coords[:-1]]
        else:
            pts = [list(c) for c in self.geometry.coords]
        return {"id": self.id, "label": self.label, "space": self.space,
                "points": pts, "threshold_m": self.threshold_m, "groups": self.groups}


@dataclass(slots=True)
class ZoneSet:
    """All geometry for one camera, plus the homography it is expressed in."""

    camera_id: str
    homography: Homography
    tripwires: list[Tripwire] = field(default_factory=list)
    zones: list[RestrictedZone] = field(default_factory=list)
    proximity: list[ProximityRing] = field(default_factory=list)
    masks: list[RestrictedZone] = field(default_factory=list)

    @property
    def metric(self) -> bool:
        """Whether distances from this zone set are true metres."""
        return self.homography.calibrated

    @classmethod
    def from_camera(cls, cam: CameraCfg) -> "ZoneSet":
        h = Homography.from_config(cam.homography)
        zs = cls(camera_id=cam.camera_id, homography=h)

        for t in cam.tripwires:
            if len(t.points) != 2:
                raise ValueError(f"tripwire {t.id!r} needs exactly 2 points")
            zs.tripwires.append(Tripwire(
                id=t.id, label=t.label or t.id,
                a=(float(t.points[0][0]), float(t.points[0][1])),
                b=(float(t.points[1][0]), float(t.points[1][1])),
                space=t.space, inbound_sign=int(t.inbound_sign),
                groups=list(t.groups)))

        for z in cam.zones:
            if len(z.points) < 3:
                raise ValueError(f"zone {z.id!r} needs at least 3 points")
            rz = RestrictedZone(id=z.id, label=z.label or z.id,
                                polygon=Polygon([(float(a), float(b)) for a, b in z.points]),
                                space=z.space, kind=z.kind, groups=list(z.groups))
            (zs.masks if z.kind == "mask" else zs.zones).append(rz)

        for p in cam.proximity:
            pts = [(float(a), float(b)) for a, b in p.points]
            if p.geometry == "point":
                geom: LineString | Point | Polygon = Point(pts[0])
            elif p.geometry == "polygon":
                geom = Polygon(pts)
            else:
                geom = LineString(pts)
            zs.proximity.append(ProximityRing(
                id=p.id, label=p.label or p.id, geometry=geom,
                space=p.space, threshold_m=p.threshold_m, groups=list(p.groups)))

        return zs

    # -- coordinate resolution ----------------------------------------------
    def resolve(
        self, footpoint_px: tuple[float, float], space: str
    ) -> tuple[tuple[float, float] | None, bool]:
        """Express a footpoint in the coordinate space a zone is defined in.

        Returns ``(point, degraded)``. ``degraded`` is True when a metric zone
        had to fall back to pixels because the camera is uncalibrated -- the
        test still runs, but the distances it produces are not metres and every
        consumer is told so.
        """
        if space == "pixel":
            return footpoint_px, False
        m = self.homography.to_metres(footpoint_px)
        if m is None:
            return footpoint_px, True
        return m, False

    def masked_out(self, footpoint_px: tuple[float, float]) -> bool:
        """Static masks -- sky, structures, a neighbour's driveway."""
        for m in self.masks:
            p, _ = self.resolve(footpoint_px, m.space)
            if p is not None and m.contains(p):
                return True
        return False

    def overlay(self) -> dict:
        """Geometry for the UI, in both metric and pixel form where possible.

        The client draws in pixels, so metric zones are projected back through
        the inverse homography here rather than in JavaScript.
        """
        def project(pts: list[list[float]], space: str) -> list[list[float]] | None:
            if space == "pixel":
                return pts
            if not self.homography.calibrated:
                return None
            out = []
            for x, y in pts:
                px = self.homography.to_pixels((x, y))
                if px is None:
                    return None
                out.append([round(px[0], 1), round(px[1], 1)])
            return out

        def pack(items) -> list[dict]:
            packed = []
            for it in items:
                d = it.to_dict()
                d["points_px"] = project(d["points"], d["space"])
                packed.append(d)
            return packed

        def inbound_arrow(w: Tripwire) -> list[list[float]] | None:
            """The inbound direction, expressed in PIXELS.

            Computed here rather than in each renderer because the two
            coordinate frames have opposite handedness: metric Y increases
            away from the camera, image y increases toward it. Deriving the
            normal from projected pixel points while applying the metric sign
            convention silently draws the arrow backwards -- so the direction
            is resolved in the space the sign is defined in, then projected.
            """
            mid = ((w.a[0] + w.b[0]) / 2.0, (w.a[1] + w.b[1]) / 2.0)
            dx, dy = w.b[0] - w.a[0], w.b[1] - w.a[1]
            length = (dx * dx + dy * dy) ** 0.5
            if length == 0:
                return None
            # Unit normal on the inbound side, in the wire's own space.
            nx, ny = (-dy / length) * w.inbound_sign, (dx / length) * w.inbound_sign
            step = length * 0.06
            tip = (mid[0] + nx * step, mid[1] + ny * step)

            if w.space == "pixel":
                return [[mid[0], mid[1]], [tip[0], tip[1]]]
            if not self.homography.calibrated:
                return None
            a_px = self.homography.to_pixels(mid)
            b_px = self.homography.to_pixels(tip)
            if a_px is None or b_px is None:
                return None
            return [[round(a_px[0], 1), round(a_px[1], 1)],
                    [round(b_px[0], 1), round(b_px[1], 1)]]

        wires = pack(self.tripwires)
        for d, w in zip(wires, self.tripwires):
            d["inbound_arrow_px"] = inbound_arrow(w)

        return {
            "camera_id": self.camera_id,
            "metric": self.metric,
            "homography": self.homography.describe(),
            "tripwires": wires,
            "zones": pack(self.zones),
            "proximity": pack(self.proximity),
        }
