"""SQLite persistence for alerts, evidence, observations and health.

Thread-safe by giving every thread its own connection. Phase 1 has one writer
(the alert manager) and several readers (the API), which SQLite in WAL mode
handles without ceremony.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from ..contracts import Alert, HealthStatus, Observation

_SCHEMA = Path(__file__).with_name("schema.sql")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Repository:
    """Alert log and evidence store."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._write_lock = threading.Lock()
        self._init_schema()

    # -- connection handling -------------------------------------------------
    @property
    def conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(self.db_path, timeout=15.0, check_same_thread=False)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA foreign_keys = ON")
            self._local.conn = c
        return c

    def _init_schema(self) -> None:
        sql = _SCHEMA.read_text(encoding="utf-8")
        with self._write_lock:
            self.conn.executescript(sql)
            self.conn.commit()

    def close(self) -> None:
        c = getattr(self._local, "conn", None)
        if c is not None:
            c.close()
            self._local.conn = None

    # -- runs ----------------------------------------------------------------
    def start_run(self, profile: str, config: dict, models: dict) -> str:
        run_id = uuid.uuid4().hex
        with self._write_lock:
            self.conn.execute(
                "INSERT INTO runs (run_id, started_at, profile, config_json, models_json)"
                " VALUES (?,?,?,?,?)",
                (run_id, _now(), profile, json.dumps(config, default=str),
                 json.dumps(models, default=str)),
            )
            self.conn.commit()
        return run_id

    # -- alerts --------------------------------------------------------------
    def upsert_alert(self, alert: Alert) -> None:
        row = alert.to_row()
        cols = ", ".join(row.keys())
        marks = ", ".join("?" for _ in row)
        updates = ", ".join(f"{k}=excluded.{k}" for k in row if k != "alert_id")
        with self._write_lock:
            self.conn.execute(
                f"INSERT INTO alerts ({cols}) VALUES ({marks}) "
                f"ON CONFLICT(alert_id) DO UPDATE SET {updates}",
                tuple(row.values()),
            )
            self.conn.execute("DELETE FROM alerts_fts WHERE alert_id = ?", (alert.alert_id,))
            self.conn.execute(
                "INSERT INTO alerts_fts (alert_id, camera_id, kind, threat, rationale)"
                " VALUES (?,?,?,?,?)",
                (alert.alert_id, alert.camera_id, alert.kind.value,
                 alert.threat.value, alert.rationale),
            )
            self.conn.commit()

    def write_evidence(self, alert: Alert) -> None:
        e = alert.evidence
        with self._write_lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO evidence (evidence_id, alert_id, created_at,"
                " keyframe_path, crop_path, clip_path, clip_start_ms, clip_end_ms,"
                " detections_json, model_ids_json, geometry_json, homography_id,"
                " config_snapshot) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    e.evidence_id, alert.alert_id,
                    (e.created_at or datetime.now(timezone.utc)).isoformat(),
                    e.keyframe_path, e.crop_path, e.clip_path,
                    e.clip_start_ms, e.clip_end_ms,
                    json.dumps(e.detections, default=str),
                    json.dumps(e.model_ids, default=str),
                    json.dumps(e.geometry, default=str),
                    e.homography_id,
                    json.dumps(e.config_snapshot, default=str),
                ),
            )
            self.conn.commit()

    def write_observations(self, obs: Iterable[Observation], alert_id: str | None = None) -> None:
        rows = [
            (o.obs_id, alert_id, o.camera_id, o.kind.value, o.at.isoformat(),
             o.frame_idx, o.confidence, ",".join(str(t) for t in o.track_ids),
             json.dumps(o.geometry, default=str), json.dumps(o.detail, default=str))
            for o in obs
        ]
        if not rows:
            return
        with self._write_lock:
            self.conn.executemany(
                "INSERT OR REPLACE INTO observations (obs_id, alert_id, camera_id, kind,"
                " at, frame_idx, confidence, track_ids, geometry_json, detail_json)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)",
                rows,
            )
            self.conn.commit()

    def log_health(self, camera_id: str, health: HealthStatus) -> None:
        with self._write_lock:
            self.conn.execute(
                "INSERT INTO health_events (camera_id, at, state, blur_score, ssim_ref, detail)"
                " VALUES (?,?,?,?,?,?)",
                (camera_id, _now(), health.state.value, health.blur_score,
                 health.ssim_ref, health.detail),
            )
            self.conn.commit()

    def log_relay(self, alert_id: str, byte_size: int, quality: int, path: str) -> None:
        with self._write_lock:
            self.conn.execute(
                "INSERT INTO relay_log (alert_id, at, byte_size, jpeg_quality, payload_path)"
                " VALUES (?,?,?,?,?)",
                (alert_id, _now(), byte_size, quality, path),
            )
            self.conn.commit()

    # -- reads ---------------------------------------------------------------
    def query(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        cur = self.conn.execute(sql, params)
        return [dict(r) for r in cur.fetchall()]

    def recent_alerts(self, limit: int = 50, camera_id: str | None = None) -> list[dict]:
        if camera_id:
            return self.query(
                "SELECT * FROM alerts WHERE camera_id=? ORDER BY opened_at DESC LIMIT ?",
                (camera_id, limit))
        return self.query("SELECT * FROM alerts ORDER BY opened_at DESC LIMIT ?", (limit,))

    def alert_with_evidence(self, alert_id: str) -> dict | None:
        rows = self.query(
            "SELECT a.*, e.detections_json, e.model_ids_json, e.geometry_json,"
            " e.homography_id, e.config_snapshot, e.clip_start_ms, e.clip_end_ms"
            " FROM alerts a LEFT JOIN evidence e ON e.alert_id = a.alert_id"
            " WHERE a.alert_id = ?", (alert_id,))
        if not rows:
            return None
        row = rows[0]
        # Column name -> the key the API exposes. config_snapshot is listed
        # explicitly because it has no `_json` suffix to strip: deriving the
        # target name by string-replacement mapped it onto itself and the
        # subsequent pop then deleted the parsed value.
        for column, key in (("detections_json", "detections"),
                            ("model_ids_json", "model_ids"),
                            ("geometry_json", "geometry"),
                            ("config_snapshot", "config_snapshot")):
            raw = row.pop(column, None)
            if raw:
                try:
                    row[key] = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    row[key] = None
            else:
                row.setdefault(key, None)
        row["observations"] = self.query(
            "SELECT * FROM observations WHERE alert_id=? ORDER BY at", (alert_id,))
        return row

    def stats(self) -> dict[str, Any]:
        total = self.query("SELECT COUNT(*) n FROM alerts")[0]["n"]
        by_threat = {r["threat"]: r["n"] for r in self.query(
            "SELECT threat, COUNT(*) n FROM alerts GROUP BY threat")}
        by_kind = {r["kind"]: r["n"] for r in self.query(
            "SELECT kind, COUNT(*) n FROM alerts GROUP BY kind")}
        live = self.query(
            "SELECT COUNT(*) n FROM alerts WHERE state IN ('PENDING','ACTIVE')")[0]["n"]
        relay = self.query(
            "SELECT COUNT(*) n, COALESCE(MAX(byte_size),0) mx,"
            " COALESCE(AVG(byte_size),0) av FROM relay_log")[0]
        return {
            "alerts_total": total,
            "alerts_live": live,
            "by_threat": by_threat,
            "by_kind": by_kind,
            "relay_count": relay["n"],
            "relay_max_bytes": relay["mx"],
            "relay_avg_bytes": round(relay["av"] or 0, 1),
        }
