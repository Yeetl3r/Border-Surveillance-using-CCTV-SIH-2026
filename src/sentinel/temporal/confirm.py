"""F7 -- temporal confirmation before any alert fires.

Specified in **seconds and a hit ratio**, not in a raw frame count.

"N-frame confirmation" means different things at 30 fps and at 8 fps, and
degraded backhaul -- the stated target condition for this system -- is exactly
what produces 8 fps. A frame-count threshold silently changes meaning on the
footage we are designing for: three frames is 100 ms on a healthy feed and
375 ms on a struggling one. So the window is wall-clock and the bar is a
proportion of the frames that actually arrived.

The ratio is computed against ticks, not against observations. Every frame
processed for a camera is a tick; an observation is a hit. A subject seen in
6 of the last 10 frames scores 0.6 regardless of how long those frames took.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from ..config import ConfirmCfg
from ..contracts import Observation, ObservationKind


@dataclass(slots=True)
class ConfirmRule:
    window_s: float
    min_ratio: float
    min_hits: int


@dataclass(slots=True)
class Candidate:
    """One accumulating candidate event, identified by subject_key."""

    subject_key: str
    kind: ObservationKind
    camera_id: str
    hits: deque[tuple[datetime, Observation]] = field(default_factory=deque)
    confirmed: bool = False
    last_hit: datetime | None = None

    def prune(self, now: datetime, window_s: float) -> None:
        cutoff = now - timedelta(seconds=window_s)
        while self.hits and self.hits[0][0] < cutoff:
            self.hits.popleft()


class TemporalConfirmer:
    """Sliding-window N-of-M confirmation, per subject."""

    def __init__(self, cfg: ConfirmCfg) -> None:
        self.cfg = cfg
        self._candidates: dict[str, Candidate] = {}
        # Per-camera tick history, so the denominator of the hit ratio is the
        # number of frames that actually arrived in the window.
        self._ticks: dict[str, deque[datetime]] = {}
        self._max_window = max(
            [cfg.window_s] + [float(o.get("window_s", cfg.window_s))
                              for o in cfg.overrides.values()]
        )

    def rule_for(self, kind: ObservationKind) -> ConfirmRule:
        o = self.cfg.overrides.get(kind.value, {})
        return ConfirmRule(
            window_s=float(o.get("window_s", self.cfg.window_s)),
            min_ratio=float(o.get("min_ratio", self.cfg.min_ratio)),
            min_hits=int(o.get("min_hits", self.cfg.min_hits)),
        )

    def tick(self, camera_id: str, at: datetime) -> None:
        """Record that a frame was processed. Call once per frame per camera."""
        dq = self._ticks.setdefault(camera_id, deque())
        dq.append(at)
        cutoff = at - timedelta(seconds=self._max_window)
        while dq and dq[0] < cutoff:
            dq.popleft()

    def _ticks_in(self, camera_id: str, now: datetime, window_s: float) -> int:
        dq = self._ticks.get(camera_id)
        if not dq:
            return 0
        cutoff = now - timedelta(seconds=window_s)
        return sum(1 for t in dq if t >= cutoff)

    def submit(self, obs: Observation) -> None:
        key = obs.subject_key
        c = self._candidates.get(key)
        if c is None:
            c = Candidate(subject_key=key, kind=obs.kind, camera_id=obs.camera_id)
            self._candidates[key] = c
        c.hits.append((obs.at, obs))
        c.last_hit = obs.at

    def evaluate(self, now: datetime) -> list[Candidate]:
        """Return candidates that have just crossed the confirmation bar.

        A candidate confirms once. Continued observations keep it alive (the
        alert manager reads ``last_hit`` to decide when to clear) but do not
        re-confirm it, so a subject standing in a zone produces one alert, not
        one per frame.
        """
        newly: list[Candidate] = []
        for c in list(self._candidates.values()):
            rule = self.rule_for(c.kind)
            c.prune(now, rule.window_s)

            if not c.hits:
                # Nothing left in the window. Drop unconfirmed candidates so
                # the dict does not accumulate; confirmed ones are the alert
                # manager's to expire.
                if not c.confirmed:
                    self._candidates.pop(c.subject_key, None)
                continue

            if c.confirmed:
                continue

            hits = len(c.hits)
            ticks = max(1, self._ticks_in(c.camera_id, now, rule.window_s))
            ratio = hits / ticks

            if hits >= rule.min_hits and ratio >= rule.min_ratio:
                c.confirmed = True
                newly.append(c)

        return newly

    def state(self, subject_key: str) -> Candidate | None:
        return self._candidates.get(subject_key)

    def forget(self, subject_key: str) -> None:
        self._candidates.pop(subject_key, None)

    def sweep(self, now: datetime, stale_after_s: float) -> None:
        """Drop candidates with no hits for a long time, confirmed or not."""
        cutoff = now - timedelta(seconds=stale_after_s)
        for key, c in list(self._candidates.items()):
            if c.last_hit is None or c.last_hit < cutoff:
                self._candidates.pop(key, None)

    @property
    def pending(self) -> int:
        return sum(1 for c in self._candidates.values() if not c.confirmed)
