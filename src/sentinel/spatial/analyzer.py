"""Tier 2 -- spatial reasoning. Pure geometry, no models, no GPU.

Turns tracks into observations about where they are in the world:

* F4 -- directional tripwire crossings
* F4 -- restricted-zone presence
* F5 -- metric proximity to a protected asset

Everything here operates on ground metres when the camera is calibrated. When
it is not, the tests still run in pixel space but every observation produced is
flagged ``degraded`` so nothing downstream presents pixels as metres.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from ..config import SpatialCfg
from ..contracts import (
    CrossDirection,
    Observation,
    ObservationKind,
    ObjectGroup,
    Track,
)
from ..calibration.zones import ZoneSet


@dataclass(slots=True)
class _Crossing:
    """A crossing that happened, held open for corroboration."""

    at: datetime
    direction: CrossDirection
    from_pt: tuple[float, float]
    to_pt: tuple[float, float]
    side_sign: float


class SpatialAnalyzer:
    """Per-camera spatial tests. Holds per-track state between frames."""

    def __init__(self, zones: ZoneSet, cfg: SpatialCfg | None = None) -> None:
        self.zones = zones
        self.cfg = cfg or SpatialCfg()
        self._default_groups = {ObjectGroup(g) for g in self.cfg.default_groups}
        self._prev: dict[tuple[int, str], tuple[float, float]] = {}
        self._crossings: dict[tuple[int, str], _Crossing] = {}

    def reset(self) -> None:
        self._prev.clear()
        self._crossings.clear()

    def _applies(self, groups: list[str], group: ObjectGroup) -> bool:
        """Whether a zone tests this object group.

        Per-zone group lists exist because wildlife triggering a perimeter is
        the canonical false alarm in this domain -- named explicitly in the
        CIBMS assessments -- and because an asset should not register proximity
        to itself.
        """
        if not groups:
            return group in self._default_groups
        return group.value in groups

    def analyse(
        self, tracks: list[Track], frame_idx: int, at: datetime
    ) -> list[Observation]:
        """Produce spatial observations for this frame.

        Also writes ``footpoint_m`` onto each track and back-fills the metric
        coordinate on the newest history sample, so the temporal stage and any
        Phase 2 motion analysis can work in metres.
        """
        obs: list[Observation] = []
        cam = self.zones.camera_id
        metric = self.zones.metric

        for t in tracks:
            fp_px = t.latest.footpoint_px

            t.footpoint_m = self.zones.homography.to_metres(fp_px)
            if t.history:
                t.history[-1].footpoint_m = t.footpoint_m

            if self.zones.masked_out(fp_px):
                continue

            obs.extend(self._tripwires(t, fp_px, cam, frame_idx, at, metric))
            obs.extend(self._zones(t, fp_px, cam, frame_idx, at, metric))
            obs.extend(self._proximity(t, fp_px, cam, frame_idx, at, metric))

        self._prune(tracks)
        return obs

    # -- F4: directional crossing -------------------------------------------
    def _tripwires(self, t, fp_px, cam, frame_idx, at, metric) -> list[Observation]:
        out: list[Observation] = []
        for wire in self.zones.tripwires:
            if not self._applies(wire.groups, t.group):
                continue
            pt, degraded = self.zones.resolve(fp_px, wire.space)
            if pt is None:
                continue

            key = (t.track_id, wire.id)
            prev = self._prev.get(key)
            self._prev[key] = pt

            direction: CrossDirection | None = None
            corroborating = False

            if prev is not None:
                direction = wire.crossing(prev, pt)

            if direction is not None:
                self._crossings[key] = _Crossing(
                    at=at, direction=direction, from_pt=prev, to_pt=pt,
                    side_sign=1.0 if wire.side(pt) > 0 else -1.0)
            else:
                # No crossing this frame. If one happened recently AND the
                # subject is still on the side it crossed to, keep corroborating
                # -- an instantaneous event cannot otherwise satisfy an N-of-M
                # confirmation rule, and a detector flicker across the line will
                # not persist on the far side the way a real incursion does.
                c = self._crossings.get(key)
                if c is not None:
                    held = (at - c.at).total_seconds()
                    still_across = (1.0 if wire.side(pt) > 0 else -1.0) == c.side_sign
                    if held <= self.cfg.tripwire_corroborate_s and still_across:
                        direction = c.direction
                        corroborating = True
                    else:
                        self._crossings.pop(key, None)

            if direction is None:
                continue

            c = self._crossings.get(key)
            out.append(Observation(
                kind=ObservationKind.TRIPWIRE_CROSS,
                camera_id=cam, at=at, frame_idx=frame_idx,
                confidence=t.latest.confidence,
                track_ids=(t.track_id,),
                geometry={
                    "zone_id": wire.id, "zone_label": wire.label,
                    "direction": direction.value,
                    "space": wire.space, "degraded": degraded,
                    "from": [round(c.from_pt[0], 3), round(c.from_pt[1], 3)] if c else None,
                    "to": [round(c.to_pt[0], 3), round(c.to_pt[1], 3)] if c else None,
                    "at": [round(pt[0], 3), round(pt[1], 3)],
                    "metric": metric and not degraded,
                },
                detail={"group": t.group.value, "class_name": t.latest.class_name,
                        "inbound": direction == CrossDirection.INBOUND,
                        "corroborating": corroborating,
                        "held_s": round((at - c.at).total_seconds(), 2) if c else 0.0},
            ))
        return out

    # -- F4: containment -----------------------------------------------------
    def _zones(self, t, fp_px, cam, frame_idx, at, metric) -> list[Observation]:
        out: list[Observation] = []
        for z in self.zones.zones:
            if not self._applies(z.groups, t.group):
                continue
            pt, degraded = self.zones.resolve(fp_px, z.space)
            if pt is None or not z.contains(pt):
                continue
            out.append(Observation(
                kind=ObservationKind.ZONE_PRESENCE,
                camera_id=cam, at=at, frame_idx=frame_idx,
                confidence=t.latest.confidence,
                track_ids=(t.track_id,),
                geometry={"zone_id": z.id, "zone_label": z.label,
                          "space": z.space, "degraded": degraded,
                          "at": [round(pt[0], 3), round(pt[1], 3)],
                          "metric": metric and not degraded},
                detail={"group": t.group.value, "class_name": t.latest.class_name},
            ))
        return out

    # -- F5: metric proximity ------------------------------------------------
    def _proximity(self, t, fp_px, cam, frame_idx, at, metric) -> list[Observation]:
        out: list[Observation] = []
        for ring in self.zones.proximity:
            if not self._applies(ring.groups, t.group):
                continue
            pt, degraded = self.zones.resolve(fp_px, ring.space)
            if pt is None:
                continue
            breached, dist = ring.breached(pt)
            if not breached:
                continue
            out.append(Observation(
                kind=ObservationKind.PROXIMITY,
                camera_id=cam, at=at, frame_idx=frame_idx,
                confidence=t.latest.confidence,
                track_ids=(t.track_id,),
                geometry={"zone_id": ring.id, "zone_label": ring.label,
                          "distance_m": round(dist, 2) if not degraded else None,
                          "distance_px": round(dist, 1) if degraded else None,
                          "threshold_m": ring.threshold_m,
                          "space": ring.space, "degraded": degraded,
                          "at": [round(pt[0], 3), round(pt[1], 3)],
                          "metric": metric and not degraded},
                detail={"group": t.group.value, "class_name": t.latest.class_name},
            ))
        return out

    def _prune(self, tracks: list[Track]) -> None:
        """Drop remembered state for tracks that are gone.

        Without this the dicts grow for the lifetime of the process, and a
        recycled track id would be compared against a stale position from a
        different subject -- which would fabricate a crossing.
        """
        live = {t.track_id for t in tracks}
        for key in [k for k in self._prev if k[0] not in live]:
            del self._prev[key]
        for key in [k for k in self._crossings if k[0] not in live]:
            del self._crossings[key]
