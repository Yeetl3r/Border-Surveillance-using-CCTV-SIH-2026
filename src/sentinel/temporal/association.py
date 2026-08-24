"""F6 -- unattended baggage: person-object association and dwell.

State machine, exactly as specified:

    [person + bag within R]  ->  ACCOMPANIED
                                      |
                          (owner further than D)
                                      v
                             DETACHED, timer running
                                      |
                              (t > T_dwell)
                                      v
                          ALERT: unattended object
                                      |
                       (owner returns within D, held
                        for the reclaim grace period)
                                      v
                                ACCOMPANIED

R and D are in METRES via the homography, not pixels, so they are
depth-invariant: a bag 4 m from its owner is 4 m away whether it is near the
camera or at the far end of the scene. A pixel threshold would mean two
different real distances at those two depths, and would therefore be wrong
in at least one of them.

T_dwell is 15 s in the demo profile and 120 s operationally. Nobody watches a
two-minute timer on stage; the operational value is what gets stated aloud.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import numpy as np

from ..config import BaggageCfg
from ..contracts import Observation, ObservationKind, ObjectGroup, Track


class BagState(str, Enum):
    ACCOMPANIED = "ACCOMPANIED"
    DETACHED = "DETACHED"
    UNATTENDED = "UNATTENDED"
    ORPHAN = "ORPHAN"        # never seen with anyone -- cannot claim abandonment


@dataclass(slots=True)
class BagRecord:
    track_id: int
    state: BagState = BagState.ORPHAN
    owner_track_id: int | None = None
    detached_at: datetime | None = None
    reclaimed_at: datetime | None = None
    last_seen: datetime | None = None
    last_distance: float | None = None
    degraded: bool = False
    history: list[str] = field(default_factory=list)

    def note(self, msg: str) -> None:
        self.history.append(msg)
        if len(self.history) > 24:
            del self.history[:-24]


class BaggageAssociator:
    """Tracks bag ownership and dwell for one camera."""

    def __init__(self, cfg: BaggageCfg) -> None:
        self.cfg = cfg
        self.bags: dict[int, BagRecord] = {}

    def reset(self) -> None:
        self.bags.clear()

    @staticmethod
    def _pos(t: Track) -> tuple[tuple[float, float], bool]:
        """Ground position if calibrated, else pixel footpoint flagged degraded."""
        if t.footpoint_m is not None:
            return t.footpoint_m, False
        return t.latest.footpoint_px, True

    def update(
        self, tracks: list[Track], at: datetime, camera_id: str
    ) -> list[Observation]:
        if not self.cfg.enabled:
            return []

        people = [t for t in tracks if t.group == ObjectGroup.PERSON]
        bags = [t for t in tracks if t.group == ObjectGroup.BAG
                and t.latest.confidence >= self.cfg.min_bag_conf]

        obs: list[Observation] = []
        live_bag_ids = {b.track_id for b in bags}

        for bag in bags:
            rec = self.bags.get(bag.track_id)
            if rec is None:
                rec = BagRecord(track_id=bag.track_id)
                self.bags[bag.track_id] = rec
            rec.last_seen = at

            bag_pt, degraded = self._pos(bag)
            rec.degraded = degraded
            assoc_r = self.cfg.assoc_radius_px if degraded else self.cfg.assoc_radius_m
            detach_d = self.cfg.detach_distance_px if degraded else self.cfg.detach_distance_m

            nearest, nearest_d = self._nearest(bag_pt, people, degraded)
            rec.last_distance = nearest_d

            # -- ownership ---------------------------------------------------
            if rec.owner_track_id is None:
                if nearest is not None and nearest_d is not None and nearest_d <= assoc_r:
                    rec.owner_track_id = nearest.track_id
                    rec.state = BagState.ACCOMPANIED
                    rec.detached_at = None
                    bag.owner_track_id = nearest.track_id
                    rec.note(f"owner {nearest.track_id} at {nearest_d:.2f}")
                else:
                    # A bag that has never been seen with a person is an ORPHAN.
                    # We deliberately do NOT alert on it: without an owner there
                    # is no abandonment event, only an object we cannot explain.
                    # Claiming otherwise would be asserting more than we know.
                    rec.state = BagState.ORPHAN
                continue

            bag.owner_track_id = rec.owner_track_id
            owner = next((p for p in people if p.track_id == rec.owner_track_id), None)

            if owner is None:
                # Owner not visible this frame. Treat as detached, but the
                # tracker's buffer means a brief occlusion will re-associate
                # before the dwell expires.
                d = None
            else:
                owner_pt, _ = self._pos(owner)
                d = float(np.hypot(bag_pt[0] - owner_pt[0], bag_pt[1] - owner_pt[1]))
                rec.last_distance = d

            # -- state transitions -------------------------------------------
            if d is not None and d <= detach_d:
                if rec.state in (BagState.DETACHED, BagState.UNATTENDED):
                    if rec.reclaimed_at is None:
                        rec.reclaimed_at = at
                    elif (at - rec.reclaimed_at).total_seconds() >= self.cfg.reclaim_grace_s:
                        rec.state = BagState.ACCOMPANIED
                        rec.detached_at = None
                        rec.reclaimed_at = None
                        rec.note(f"reclaimed at {d:.2f}")
                else:
                    rec.state = BagState.ACCOMPANIED
                    rec.detached_at = None
                    rec.reclaimed_at = None
                continue

            # Owner is far, or gone.
            rec.reclaimed_at = None
            if rec.state == BagState.ACCOMPANIED or rec.detached_at is None:
                rec.state = BagState.DETACHED
                rec.detached_at = at
                rec.note("detached")

            elapsed = (at - rec.detached_at).total_seconds()
            if elapsed >= self.cfg.dwell_s:
                rec.state = BagState.UNATTENDED
                obs.append(Observation(
                    kind=ObservationKind.BAG_UNATTENDED,
                    camera_id=camera_id, at=at, frame_idx=0,
                    confidence=bag.latest.confidence,
                    track_ids=tuple(sorted(
                        {bag.track_id, rec.owner_track_id} - {None})),  # type: ignore[arg-type]
                    geometry={
                        "zone_id": f"bag{bag.track_id}",
                        "distance_m": round(d, 2) if (d is not None and not degraded) else None,
                        "distance_px": round(d, 1) if (d is not None and degraded) else None,
                        "threshold_m": self.cfg.detach_distance_m,
                        "degraded": degraded, "metric": not degraded,
                        "at": [round(bag_pt[0], 2), round(bag_pt[1], 2)],
                    },
                    detail={
                        "bag_track": bag.track_id,
                        "owner_track": rec.owner_track_id,
                        "dwell_s": round(elapsed, 1),
                        "dwell_threshold_s": self.cfg.dwell_s,
                        "owner_visible": owner is not None,
                        "class_name": bag.latest.class_name,
                    },
                ))

        # Forget bags that have left the scene entirely.
        for tid in [t for t in self.bags if t not in live_bag_ids]:
            rec = self.bags[tid]
            if rec.last_seen and (at - rec.last_seen).total_seconds() > 30.0:
                self.bags.pop(tid, None)

        return obs

    def _nearest(self, pt, people: list[Track], degraded: bool):
        best, best_d = None, None
        for p in people:
            ppt, pdeg = self._pos(p)
            if pdeg != degraded:
                continue                    # never mix metric and pixel distances
            d = float(np.hypot(pt[0] - ppt[0], pt[1] - ppt[1]))
            if best_d is None or d < best_d:
                best, best_d = p, d
        return best, best_d

    def snapshot(self) -> list[dict]:
        """Bag states for the UI, so the dwell timer is visible while it runs."""
        return [
            {"track_id": r.track_id, "state": r.state.value,
             "owner": r.owner_track_id,
             "distance": round(r.last_distance, 2) if r.last_distance is not None else None,
             "degraded": r.degraded}
            for r in self.bags.values()
        ]
