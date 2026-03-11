"""
ForenSight AI — Database Layer
SQLite via aiosqlite + databases for async access.

Tables:
  - users     : Google-authenticated users
  - analyses  : Detection history per user

Author: ForenSight AI Team
"""

import os
from datetime import datetime
from pathlib import Path

import aiosqlite
import databases

DATABASE_PATH = os.environ.get("DATABASE_PATH", "/data/forensight.db")

DATABASE_URL = f"sqlite+aiosqlite:///{DATABASE_PATH}"

database = databases.Database(DATABASE_URL)

# ─── Schema ────────────────────────────────────────────────────────────────────

CREATE_USERS_TABLE = """
CREATE TABLE IF NOT EXISTS users (
    user_id     TEXT PRIMARY KEY,          -- Firebase UID
    name        TEXT NOT NULL,
    email       TEXT NOT NULL UNIQUE,
    photo_url   TEXT,
    created_at  TEXT NOT NULL
);
"""

CREATE_ANALYSES_TABLE = """
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
"""

CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_analyses_user_id ON analyses (user_id);
"""

# ─── Lifecycle ─────────────────────────────────────────────────────────────────

async def init_db():
    """Create tables on startup."""
    async with aiosqlite.connect(DATABASE_PATH) as conn:
        await conn.execute("PRAGMA journal_mode=WAL;")   # better concurrency
        await conn.execute("PRAGMA foreign_keys=ON;")
        await conn.execute(CREATE_USERS_TABLE)
        await conn.execute(CREATE_ANALYSES_TABLE)
        await conn.execute(CREATE_INDEX)
        await conn.commit()
    await database.connect()


async def close_db():
    """Close DB connection on shutdown."""
    await database.disconnect()


# ─── Users CRUD ────────────────────────────────────────────────────────────────

async def get_or_create_user(
    user_id: str,
    name: str,
    email: str,
    photo_url: str = None,
) -> dict:
    """
    Upsert a user record by Firebase UID.
    Returns the user row dict.
    """
    now = datetime.utcnow().isoformat()

    # Try insert; if exists update name/photo
    query = """
        INSERT INTO users (user_id, name, email, photo_url, created_at)
        VALUES (:user_id, :name, :email, :photo_url, :created_at)
        ON CONFLICT(user_id) DO UPDATE SET
            name = excluded.name,
            photo_url = excluded.photo_url
        RETURNING *
    """
    # SQLite < 3.35 doesn't support RETURNING clause robustly — use two-step
    upsert_query = """
        INSERT INTO users (user_id, name, email, photo_url, created_at)
        VALUES (:user_id, :name, :email, :photo_url, :created_at)
        ON CONFLICT(user_id) DO UPDATE SET
            name = excluded.name,
            photo_url = excluded.photo_url
    """
    values = {
        "user_id": user_id,
        "name": name,
        "email": email,
        "photo_url": photo_url,
        "created_at": now,
    }
    await database.execute(query=upsert_query, values=values)

    row = await database.fetch_one(
        "SELECT * FROM users WHERE user_id = :uid", {"uid": user_id}
    )
    return dict(row)


async def get_user_by_id(user_id: str) -> dict | None:
    row = await database.fetch_one(
        "SELECT * FROM users WHERE user_id = :uid", {"uid": user_id}
    )
    return dict(row) if row else None


# ─── Analyses CRUD ─────────────────────────────────────────────────────────────

async def save_analysis(
    analysis_id: str,
    user_id: str,
    file_name: str,
    file_type: str,
    result: dict,
) -> dict:
    """
    Persist a detection result to the analyses table.
    `result` is the raw dict returned by the inference engine.
    Returns the saved analysis row.
    """
    import uuid
    now = datetime.utcnow().isoformat()

    query = """
        INSERT INTO analyses (
            analysis_id, user_id, file_name, file_type,
            result, confidence_score, deepfake_probability,
            cnn_score, vit_score, temporal_score, model_agreement,
            face_detected, frames_analyzed, processing_time_sec,
            heatmap_path, created_at
        ) VALUES (
            :analysis_id, :user_id, :file_name, :file_type,
            :result, :confidence_score, :deepfake_probability,
            :cnn_score, :vit_score, :temporal_score, :model_agreement,
            :face_detected, :frames_analyzed, :processing_time_sec,
            :heatmap_path, :created_at
        )
    """
    values = {
        "analysis_id": analysis_id,
        "user_id": user_id,
        "file_name": file_name,
        "file_type": file_type,
        "result": result.get("label"),
        "confidence_score": result.get("confidence"),
        "deepfake_probability": result.get("deepfake_probability"),
        "cnn_score": result.get("cnn_score"),
        "vit_score": result.get("vit_score"),
        "temporal_score": result.get("temporal_score"),
        "model_agreement": result.get("model_agreement"),
        "face_detected": 1 if result.get("face_detected") else 0,
        "frames_analyzed": result.get("frames_analyzed"),
        "processing_time_sec": result.get("processing_time_sec"),
        "heatmap_path": result.get("heatmap"),   # base64 string or None
        "created_at": now,
    }
    await database.execute(query=query, values=values)
    return {**values, "created_at": now}


async def get_user_history(user_id: str, limit: int = 50) -> list[dict]:
    """Fetch a user's analysis history, newest first."""
    rows = await database.fetch_all(
        """
        SELECT analysis_id, file_name, file_type, result,
               confidence_score, deepfake_probability, face_detected,
               processing_time_sec, created_at
        FROM analyses
        WHERE user_id = :uid
        ORDER BY created_at DESC
        LIMIT :limit
        """,
        {"uid": user_id, "limit": limit},
    )
    return [dict(r) for r in rows]


async def get_analysis_by_id(analysis_id: str, user_id: str) -> dict | None:
    """Fetch one full analysis, ensuring it belongs to the requesting user."""
    row = await database.fetch_one(
        "SELECT * FROM analyses WHERE analysis_id = :aid AND user_id = :uid",
        {"aid": analysis_id, "uid": user_id},
    )
    return dict(row) if row else None


async def delete_analysis(analysis_id: str, user_id: str) -> bool:
    """Delete an analysis. Returns True if a row was deleted."""
    result = await database.execute(
        "DELETE FROM analyses WHERE analysis_id = :aid AND user_id = :uid",
        {"aid": analysis_id, "uid": user_id},
    )
    return result > 0
