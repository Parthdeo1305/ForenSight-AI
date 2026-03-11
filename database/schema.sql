-- ForenSight AI — Database Schema
-- Uses SQLite out of the box, auto-created via `database.py`.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS users (
    user_id     TEXT PRIMARY KEY,          -- Firebase UID
    name        TEXT NOT NULL,
    email       TEXT NOT NULL UNIQUE,
    photo_url   TEXT,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS analyses (
    analysis_id         TEXT PRIMARY KEY,
    user_id             TEXT NOT NULL,
    file_name           TEXT NOT NULL,
    file_type           TEXT NOT NULL,          -- 'image' | 'video'
    result              TEXT,                   -- 'REAL' | 'FAKE'
    confidence_score    REAL,
    deepfake_probability REAL,
    cnn_score           REAL,
    vit_score           REAL,
    temporal_score      REAL,
    model_agreement     REAL,
    face_detected       INTEGER,               -- 0 | 1
    frames_analyzed     INTEGER,
    processing_time_sec REAL,
    heatmap_path        TEXT,                  -- relative path or base64
    created_at          TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_analyses_user_id ON analyses (user_id);
