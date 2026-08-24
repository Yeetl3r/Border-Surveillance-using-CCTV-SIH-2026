-- SENTINEL alert log and evidence store.
--
-- SQLite with FTS5. Postgres is a later migration; at Phase 1 scale a single
-- file that runs air-gapped with zero setup is the correct trade.
--
-- F10 queries this. F13 requires that every alert here can be traced back to
-- the keyframe, the confidences, the model weights and the thresholds that
-- produced it -- so evidence is a first-class table, not a JSON blob column.

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- ---------------------------------------------------------------------------
-- Alerts (F10 -- the structured query route reads this)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS alerts (
    alert_id          TEXT PRIMARY KEY,
    camera_id         TEXT NOT NULL,
    kind              TEXT NOT NULL,
    threat            TEXT NOT NULL,
    threat_severity   INTEGER NOT NULL,
    confidence        REAL NOT NULL,
    confidence_band   TEXT NOT NULL,
    state             TEXT NOT NULL,
    opened_at         TEXT NOT NULL,      -- ISO-8601, timezone-aware
    updated_at        TEXT,
    cleared_at        TEXT,
    track_ids         TEXT,
    rationale         TEXT,
    subject_key       TEXT,
    evidence_id       TEXT,
    keyframe_path     TEXT,
    crop_path         TEXT,
    clip_path         TEXT
);

CREATE INDEX IF NOT EXISTS ix_alerts_opened   ON alerts(opened_at DESC);
CREATE INDEX IF NOT EXISTS ix_alerts_camera   ON alerts(camera_id, opened_at DESC);
CREATE INDEX IF NOT EXISTS ix_alerts_kind     ON alerts(kind, opened_at DESC);
CREATE INDEX IF NOT EXISTS ix_alerts_threat   ON alerts(threat_severity DESC, opened_at DESC);
CREATE INDEX IF NOT EXISTS ix_alerts_state    ON alerts(state);

-- ---------------------------------------------------------------------------
-- Evidence (F13 -- one row per alert, immutable once written)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS evidence (
    evidence_id       TEXT PRIMARY KEY,
    alert_id          TEXT NOT NULL,
    created_at        TEXT NOT NULL,
    keyframe_path     TEXT,
    crop_path         TEXT,
    clip_path         TEXT,
    clip_start_ms     INTEGER DEFAULT 0,
    clip_end_ms       INTEGER DEFAULT 0,
    -- Every detection that supported the alert, with its own confidence and
    -- the model that produced it. "Which weights said this?" must be answerable.
    detections_json   TEXT,
    model_ids_json    TEXT,
    -- The triggering geometry: which line, which direction, which polygon,
    -- distance in metres. This is what makes an alert auditable rather than
    -- an assertion.
    geometry_json     TEXT,
    homography_id     TEXT,
    -- Thresholds in force at fire time. Without this, "why did this fire?" is
    -- unanswerable once the profile has been switched.
    config_snapshot   TEXT,
    FOREIGN KEY (alert_id) REFERENCES alerts(alert_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS ix_evidence_alert ON evidence(alert_id);

-- ---------------------------------------------------------------------------
-- Observations (the raw, unpromoted stream -- kept for audit and tuning)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS observations (
    obs_id        TEXT PRIMARY KEY,
    alert_id      TEXT,
    camera_id     TEXT NOT NULL,
    kind          TEXT NOT NULL,
    at            TEXT NOT NULL,
    frame_idx     INTEGER,
    confidence    REAL,
    track_ids     TEXT,
    geometry_json TEXT,
    detail_json   TEXT
);

CREATE INDEX IF NOT EXISTS ix_obs_at    ON observations(at DESC);
CREATE INDEX IF NOT EXISTS ix_obs_alert ON observations(alert_id);

-- ---------------------------------------------------------------------------
-- Camera health history (F9)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS health_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id   TEXT NOT NULL,
    at          TEXT NOT NULL,
    state       TEXT NOT NULL,
    blur_score  REAL,
    ssim_ref    REAL,
    detail      TEXT
);

CREATE INDEX IF NOT EXISTS ix_health_at ON health_events(camera_id, at DESC);

-- ---------------------------------------------------------------------------
-- Relay log (F14 -- proof the sub-3 KB budget was actually met)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS relay_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    alert_id     TEXT NOT NULL,
    at           TEXT NOT NULL,
    byte_size    INTEGER NOT NULL,
    jpeg_quality INTEGER,
    payload_path TEXT
);

-- ---------------------------------------------------------------------------
-- Run metadata -- which build, which profile, which weights produced this data
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    started_at  TEXT NOT NULL,
    profile     TEXT,
    config_json TEXT,
    models_json TEXT
);

-- ---------------------------------------------------------------------------
-- Full-text search over alert rationales, for the structured query route.
-- ---------------------------------------------------------------------------
CREATE VIRTUAL TABLE IF NOT EXISTS alerts_fts USING fts5(
    alert_id UNINDEXED,
    camera_id,
    kind,
    threat,
    rationale,
    tokenize = 'porter'
);
