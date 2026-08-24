"""F10 -- the structured query route.

"When was the last threat found?" is a SQL query, not a VLM call. Routing it
correctly is itself worth demonstrating: it shows where the expensive component
belongs, and it answers in single-digit milliseconds instead of seconds.

Queries are built from a fixed set of parameterised templates rather than by
generating SQL from text. The operator's words select a template and fill its
bound parameters; they never become SQL. That removes injection as a category
and means every query the system can run is one we have read.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any

from ..config import QueryCfg
from ..storage import Repository

_THREAT_WORDS = {
    "critical": "CRITICAL", "warning": "WARNING", "watch": "WATCH",
    "friendly": "FRIENDLY", "info": "INFO",
}

_KIND_WORDS = {
    "tripwire": "TRIPWIRE_CROSS", "crossing": "TRIPWIRE_CROSS",
    "breach": "TRIPWIRE_CROSS", "perimeter": "TRIPWIRE_CROSS",
    "zone": "ZONE_PRESENCE", "intrusion": "ZONE_PRESENCE",
    "proximity": "PROXIMITY", "near": "PROXIMITY",
    "bag": "BAG_UNATTENDED", "baggage": "BAG_UNATTENDED",
    "luggage": "BAG_UNATTENDED", "abandoned": "BAG_UNATTENDED",
    "unattended": "BAG_UNATTENDED",
    "crawl": "STANCE_ANOMALY", "stance": "STANCE_ANOMALY",
    "prone": "STANCE_ANOMALY", "crouch": "STANCE_ANOMALY",
    "tamper": "HEALTH_FAULT", "health": "HEALTH_FAULT",
    "camera": "HEALTH_FAULT", "blind": "HEALTH_FAULT",
    "identity": "IDENTITY_UNKNOWN", "unidentified": "IDENTITY_UNKNOWN",
}


class SqlRoute:
    """Parameterised structured queries over the alert log."""

    def __init__(self, cfg: QueryCfg, repo: Repository) -> None:
        self.cfg = cfg
        self.repo = repo

    # -- text -> filters -----------------------------------------------------
    def parse(self, text: str) -> dict[str, Any]:
        q = text.lower().strip()
        f: dict[str, Any] = {}

        for w, k in _KIND_WORDS.items():
            if w in q:
                f["kind"] = k
                break
        for w, t in _THREAT_WORDS.items():
            if w in q:
                f["threat"] = t
                break

        cam = re.search(r"\b(cam\w*[-_ ]?\d+)\b", q)
        if cam:
            f["camera_id"] = cam.group(1).replace(" ", "").replace("-", "")

        # Relative time windows.
        m = re.search(r"last\s+(\d+)\s*(second|sec|minute|min|hour|hr|day)s?", q)
        if m:
            n, unit = int(m.group(1)), m.group(2)
            mult = {"second": 1, "sec": 1, "minute": 60, "min": 60,
                    "hour": 3600, "hr": 3600, "day": 86400}[unit]
            f["since"] = datetime.now(timezone.utc) - timedelta(seconds=n * mult)
        elif "today" in q:
            now = datetime.now(timezone.utc)
            f["since"] = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif "tonight" in q or "overnight" in q:
            f["since"] = datetime.now(timezone.utc) - timedelta(hours=12)

        # Absolute clock time, e.g. "after 22:00".
        m = re.search(r"after\s+(\d{1,2}):(\d{2})", q)
        if m:
            now = datetime.now(timezone.utc)
            f["since"] = now.replace(hour=int(m.group(1)) % 24,
                                     minute=int(m.group(2)),
                                     second=0, microsecond=0)

        if re.search(r"\b(last|latest|most recent|newest)\b", q):
            f["limit"] = 1
        if re.search(r"\b(count|how many|number of|total)\b", q):
            f["aggregate"] = "count"
        if "live" in q or "active" in q or "current" in q or "open" in q:
            f["state"] = ("PENDING", "ACTIVE")
        if re.search(r"\b(cleared|resolved|closed)\b", q):
            f["state"] = ("CLEARED",)
        return f

    # -- filters -> SQL ------------------------------------------------------
    def build(self, f: dict[str, Any]) -> tuple[str, tuple]:
        where: list[str] = []
        params: list[Any] = []

        if f.get("kind"):
            where.append("kind = ?"); params.append(f["kind"])
        if f.get("threat"):
            where.append("threat = ?"); params.append(f["threat"])
        if f.get("camera_id"):
            where.append("camera_id = ?"); params.append(f["camera_id"])
        if f.get("since"):
            where.append("opened_at >= ?"); params.append(f["since"].isoformat())
        if f.get("state"):
            marks = ",".join("?" for _ in f["state"])
            where.append(f"state IN ({marks})"); params.extend(f["state"])

        clause = (" WHERE " + " AND ".join(where)) if where else ""

        if f.get("aggregate") == "count":
            return f"SELECT COUNT(*) AS count FROM alerts{clause}", tuple(params)

        limit = min(int(f.get("limit", 25)), self.cfg.max_rows)
        return (
            "SELECT alert_id, camera_id, kind, threat, confidence, confidence_band,"
            " state, opened_at, cleared_at, track_ids, rationale, keyframe_path,"
            " crop_path, clip_path"
            f" FROM alerts{clause} ORDER BY opened_at DESC LIMIT ?",
            tuple(params) + (limit,),
        )

    def run(self, text: str) -> dict[str, Any]:
        import time
        t0 = time.perf_counter()
        filters = self.parse(text)
        sql, params = self.build(filters)
        rows = self.repo.query(sql, params)
        ms = (time.perf_counter() - t0) * 1000

        return {
            "route": "structured",
            "backend": "sqlite",
            "sql": sql,
            "params": [p.isoformat() if isinstance(p, datetime) else p for p in params],
            "filters": {k: (v.isoformat() if isinstance(v, datetime) else v)
                        for k, v in filters.items()},
            "rows": rows,
            "count": rows[0]["count"] if filters.get("aggregate") == "count" and rows
                     else len(rows),
            "latency_ms": round(ms, 2),
            "answer": self._phrase(text, filters, rows),
        }

    def _phrase(self, text: str, f: dict, rows: list[dict]) -> str:
        """A plain-language answer, so the operator does not read a table to get one."""
        if f.get("aggregate") == "count":
            return f"{rows[0]['count'] if rows else 0} matching alerts."
        if not rows:
            return "No alerts match that query."
        r = rows[0]
        when = r["opened_at"].replace("T", " ")[:19]
        if f.get("limit") == 1:
            return (f"Most recent: {r['threat']} {r['kind'].replace('_', ' ').lower()} "
                    f"on {r['camera_id']} at {when} UTC "
                    f"({r['confidence_band'].lower()} confidence). {r['rationale']}")
        return f"{len(rows)} alerts; most recent {r['kind']} on {r['camera_id']} at {when} UTC."
