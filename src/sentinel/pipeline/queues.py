"""Bounded queues with explicit, counted drop policies.

Tier 1 runs at video rate; the VLM takes seconds. Unbounded queues between
stages with that speed mismatch are the single most likely cause of a live-demo
crash -- memory climbs until the process dies, usually at the worst moment.

So every queue here is bounded and every drop is counted, and the counters are
surfaced on the UI health strip. **An invisible drop policy is the same as no
drop policy**: if frames are being discarded, the operator has to be able to
see that, otherwise the system is quietly lying about its coverage.

Four policies, chosen per queue by what the stage is for:

* ``DROP_OLDEST`` -- live video. A stale frame is worthless; latency beats
  completeness.
* ``DROP_NEWEST`` -- opportunistic work like re-ID. The next frame will offer
  another chance at the same subject.
* ``DROP_LOWEST`` -- alerts. Shed the least severe first, and never a CRITICAL.
* ``BLOCK`` -- evidence. Losing evidence is unacceptable, so backpressure onto
  the producer is the correct behaviour.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Generic, TypeVar

T = TypeVar("T")


class DropPolicy(str, Enum):
    DROP_OLDEST = "drop_oldest"
    DROP_NEWEST = "drop_newest"
    DROP_LOWEST = "drop_lowest"
    BLOCK = "block"


@dataclass
class QueueStats:
    name: str
    capacity: int
    policy: str
    pushed: int = 0
    popped: int = 0
    dropped: int = 0
    blocked_ms: float = 0.0
    high_water: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "capacity": self.capacity, "policy": self.policy,
            "pushed": self.pushed, "popped": self.popped, "dropped": self.dropped,
            "high_water": self.high_water,
            "drop_rate": round(self.dropped / self.pushed, 4) if self.pushed else 0.0,
            "blocked_ms": round(self.blocked_ms, 1),
        }


class BoundedQueue(Generic[T]):
    """A bounded queue that reports what it threw away."""

    def __init__(
        self,
        name: str,
        capacity: int,
        policy: DropPolicy = DropPolicy.DROP_OLDEST,
        priority: Callable[[T], int] | None = None,
        never_drop: Callable[[T], bool] | None = None,
    ) -> None:
        self.capacity = max(1, capacity)
        self.policy = policy
        self.priority = priority
        self.never_drop = never_drop
        self._dq: deque[T] = deque()
        self._lock = threading.Condition()
        self._closed = False
        self.stats = QueueStats(name=name, capacity=self.capacity, policy=policy.value)

    def __len__(self) -> int:
        with self._lock:
            return len(self._dq)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._lock.notify_all()

    @property
    def closed(self) -> bool:
        return self._closed

    def push(self, item: T, timeout: float = 1.0) -> bool:
        """Enqueue. Returns False if the item was dropped."""
        with self._lock:
            if self._closed:
                return False

            if len(self._dq) >= self.capacity:
                if self.policy is DropPolicy.BLOCK:
                    import time
                    t0 = time.perf_counter()
                    ok = self._lock.wait_for(
                        lambda: len(self._dq) < self.capacity or self._closed,
                        timeout=timeout)
                    self.stats.blocked_ms += (time.perf_counter() - t0) * 1000
                    if not ok or self._closed:
                        self.stats.dropped += 1
                        return False

                elif self.policy is DropPolicy.DROP_NEWEST:
                    self.stats.dropped += 1
                    return False

                elif self.policy is DropPolicy.DROP_LOWEST:
                    victim = self._lowest_droppable(item)
                    if victim is None:
                        self.stats.dropped += 1
                        return False           # everything queued outranks this
                    self._dq.remove(victim)
                    self.stats.dropped += 1

                else:                          # DROP_OLDEST
                    self._dq.popleft()
                    self.stats.dropped += 1

            self._dq.append(item)
            self.stats.pushed += 1
            self.stats.high_water = max(self.stats.high_water, len(self._dq))
            self._lock.notify()
            return True

    def _lowest_droppable(self, incoming: T) -> T | None:
        """Pick the least important queued item, respecting never_drop."""
        if self.priority is None:
            return self._dq[0] if self._dq else None
        cands = [x for x in self._dq
                 if not (self.never_drop and self.never_drop(x))]
        if not cands:
            return None
        victim = min(cands, key=self.priority)
        if self.priority(victim) > self.priority(incoming):
            return None                        # incoming is the least important
        return victim

    def pop(self, timeout: float | None = 0.5) -> T | None:
        with self._lock:
            if not self._dq:
                self._lock.wait_for(lambda: bool(self._dq) or self._closed,
                                    timeout=timeout)
            if not self._dq:
                return None
            item = self._dq.popleft()
            self.stats.popped += 1
            self._lock.notify()
            return item

    def drain(self, limit: int = 64) -> list[T]:
        with self._lock:
            out = []
            while self._dq and len(out) < limit:
                out.append(self._dq.popleft())
                self.stats.popped += 1
            self._lock.notify_all()
            return out


@dataclass
class QueueRegistry:
    """All queues in one place, so the UI can show every drop counter."""

    queues: dict[str, BoundedQueue] = field(default_factory=dict)

    def add(self, q: BoundedQueue) -> BoundedQueue:
        self.queues[q.stats.name] = q
        return q

    def stats(self) -> list[dict[str, Any]]:
        return [q.stats.as_dict() for q in self.queues.values()]

    def total_dropped(self) -> int:
        return sum(q.stats.dropped for q in self.queues.values())

    def close_all(self) -> None:
        for q in self.queues.values():
            q.close()
