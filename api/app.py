"""
ForenSight AI — Detecting Digital Truth in the Age of Deepfakes
Main FastAPI Application

Endpoints:
  System:
    GET  /health              — Health check
    GET  /                    — API root info

  Auth:
    POST /auth/login          — Verify Firebase token, return user profile

  User:
    GET  /user/me             — Current user info

  Detection:
    POST /upload-image        — Upload image (auth required)
    POST /upload-video        — Upload video (auth required)
    POST /detect              — Run detection, save to DB (auth required)
    GET  /result/{task_id}    — Retrieve cached result (auth required)

  History:
    GET  /history             — User's analysis history (auth required)
    GET  /history/{id}        — Full result for one analysis (auth required)
    DELETE /history/{id}      — Delete an analysis (auth required)

API docs are hidden in production (set ENABLE_DOCS=true for local dev).

Author: ForenSight AI Team
"""

import os
import uuid
from pathlib import Path
from typing import Dict, Optional

import aiofiles
from dotenv import load_dotenv
from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Load .env if present
load_dotenv()

# ─── Configuration ─────────────────────────────────────────────────────────────

UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", "uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

WEIGHTS_DIR = Path(os.environ.get("WEIGHTS_DIR", "weights"))
MAX_FILE_SIZE_MB = int(os.environ.get("MAX_FILE_SIZE_MB", "500"))
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

ENABLE_DOCS = os.environ.get("ENABLE_DOCS", "false").lower() == "true"
DISABLE_AUTH = os.environ.get("DISABLE_AUTH", "false").lower() == "true"

ALLOWED_ORIGINS = os.environ.get(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173"
).split(",")

# In production, also allow all Vercel and HF origins
ALLOWED_ORIGINS += [
    "https://*.vercel.app",
    "https://*.hf.space",
]

ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
ALLOWED_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

# ─── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="ForenSight AI",
    description="Detecting Digital Truth in the Age of Deepfakes",
    version="2.0.0",
    # API docs visible only in dev via ENABLE_DOCS=true
    docs_url="/docs" if ENABLE_DOCS else None,
    redoc_url="/redoc" if ENABLE_DOCS else None,
    openapi_url="/openapi.json" if ENABLE_DOCS else None,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Include Routers ──────────────────────────────────────────────────────────

from api.routes.auth_routes import router as auth_router
from api.routes.user_routes import router as user_router
from api.routes.history_routes import router as history_router
from api.routes.report_routes import router as report_router

app.include_router(auth_router)
app.include_router(user_router)
app.include_router(history_router)
app.include_router(report_router)

# ─── Startup / Shutdown ───────────────────────────────────────────────────────

_inference_engine = None
_pretrained_detector = None

@app.on_event("startup")
async def startup():
    global _inference_engine, _pretrained_detector

    # Init database
    from api.database import init_db
    await init_db()
    print("[ForenSight AI] Database initialized.")

    # Try loading the full ensemble engine (requires trained weights)
    try:
        from api.inference import get_inference_engine
        _inference_engine = get_inference_engine(
            weights_dir=str(WEIGHTS_DIR),
            enable_gradcam=True,
        )
        print("[ForenSight AI] Full inference engine ready.")
    except Exception as e:
        print(f"[ForenSight AI] Full engine unavailable: {e}")

    # Always load the pre-trained detector as primary/fallback
    try:
        from api.pretrained_detector import get_pretrained_detector
        _pretrained_detector = get_pretrained_detector()
        print("[ForenSight AI] Pre-trained detector ready (EfficientNet-B0 / FaceForensics++).")
    except Exception as e:
        print(f"[ForenSight AI] WARNING: Pre-trained detector failed: {e}")
        print("[ForenSight AI] Falling back to demo mode.")


@app.on_event("shutdown")
async def shutdown():
    from api.database import close_db
    await close_db()


def get_engine():
    """Get the best available detection engine."""
    if _inference_engine is not None:
        return _inference_engine
    if _pretrained_detector is not None:
        return _pretrained_detector
    raise HTTPException(status_code=503, detail="No detection engine available.")


# ─── In-Memory Task Cache ─────────────────────────────────────────────────────
# Maps task_id → {status, file_path, file_type, filename, result, user_id}

_task_cache: Dict[str, dict] = {}

# ─── Dependencies ─────────────────────────────────────────────────────────────

from api.auth import get_current_user

# ─── Utility ──────────────────────────────────────────────────────────────────

def _validate_ext(filename: str, allowed: set) -> bool:
    return Path(filename).suffix.lower() in allowed


async def _save_upload(file: UploadFile, task_id: str) -> Path:
    ext = Path(file.filename).suffix.lower()
    save_path = UPLOAD_DIR / f"{task_id}{ext}"
    async with aiofiles.open(save_path, "wb") as f:
        content = await file.read()
        if len(content) > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Max: {MAX_FILE_SIZE_MB} MB",
            )
        await f.write(content)
    return save_path


def _cleanup_file(path: str):
    try:
        Path(path).unlink(missing_ok=True)
    except Exception:
        pass


# ─── System Endpoints ─────────────────────────────────────────────────────────

@app.get("/", tags=["System"])
async def root():
    return {
        "name": "ForenSight AI",
        "tagline": "Detecting Digital Truth in the Age of Deepfakes",
        "version": "2.0.0",
        "status": "running",
        "docs": "/docs" if ENABLE_DOCS else "Disabled in production",
    }


@app.get("/health", tags=["System"])
async def health():
    try:
        import torch
        device = str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        cuda = torch.cuda.is_available()
    except ImportError:
        device = "cpu (torch not installed — demo mode)"
        cuda = False
    return {
        "status": "ok",
        "version": "2.0.0",
        "models_loaded": _inference_engine is not None,
        "pretrained_detector": _pretrained_detector is not None,
        "auth_mode": "disabled (dev)" if DISABLE_AUTH else "Firebase",
        "device": device,
        "cuda_available": cuda,
    }


# ─── Detection Endpoints ──────────────────────────────────────────────────────

@app.post("/upload-image", tags=["Detection"])
async def upload_image(
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user),
):
    if not _validate_ext(file.filename, ALLOWED_IMAGE_EXTS):
        raise HTTPException(400, f"Invalid image format. Allowed: {ALLOWED_IMAGE_EXTS}")

    task_id = str(uuid.uuid4())
    save_path = await _save_upload(file, task_id)
    _task_cache[task_id] = {
        "status": "uploaded",
        "file_path": str(save_path),
        "file_type": "image",
        "filename": file.filename,
        "user_id": user["user_id"],
    }
    return {"task_id": task_id, "status": "uploaded", "filename": file.filename}


@app.post("/upload-video", tags=["Detection"])
async def upload_video(
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user),
):
    if not _validate_ext(file.filename, ALLOWED_VIDEO_EXTS):
        raise HTTPException(400, f"Invalid video format. Allowed: {ALLOWED_VIDEO_EXTS}")

    task_id = str(uuid.uuid4())
    save_path = await _save_upload(file, task_id)
    _task_cache[task_id] = {
        "status": "uploaded",
        "file_path": str(save_path),
        "file_type": "video",
        "filename": file.filename,
        "user_id": user["user_id"],
    }
    return {"task_id": task_id, "status": "uploaded", "filename": file.filename}


@app.post("/detect", tags=["Detection"])
async def detect(
    request_body: dict,
    background_tasks: BackgroundTasks,
    user: dict = Depends(get_current_user),
):
    """
    Run deepfake detection and persist result to database.
    Returns full result including heatmap.
    """
    task_id = request_body.get("task_id")
    if not task_id or task_id not in _task_cache:
        raise HTTPException(404, "Task not found. Upload a file first.")

    task = _task_cache[task_id]

    # Security: task must belong to the requesting user
    if task.get("user_id") != user["user_id"]:
        raise HTTPException(403, "Access denied.")

    if task.get("status") == "processing":
        return {"task_id": task_id, "status": "processing"}

    if task.get("status") == "complete":
        return {"task_id": task_id, "status": "complete", **task.get("result", {})}

    _task_cache[task_id]["status"] = "processing"
    file_path = task["file_path"]

    try:
        engine = get_engine()
        result = engine.predict(file_path)
    except HTTPException:
        result = _demo_result(task["file_type"])
    except Exception as e:
        _task_cache[task_id]["status"] = "error"
        return {"task_id": task_id, "status": "error", "error": str(e)}

    # Save to database
    analysis_id = str(uuid.uuid4())
    from api.database import save_analysis
    await save_analysis(
        analysis_id=analysis_id,
        user_id=user["user_id"],
        file_name=task["filename"],
        file_type=task["file_type"],
        result=result,
    )

    _task_cache[task_id]["status"] = "complete"
    _task_cache[task_id]["result"] = {**result, "analysis_id": analysis_id}
    background_tasks.add_task(_cleanup_file, file_path)

    return {
        "task_id": task_id,
        "analysis_id": analysis_id,
        "status": "complete",
        **result,
    }


@app.get("/result/{task_id}", tags=["Detection"])
async def get_result(
    task_id: str,
    user: dict = Depends(get_current_user),
):
    if task_id not in _task_cache:
        raise HTTPException(404, f"Task not found: {task_id}")

    task = _task_cache[task_id]
    if task.get("user_id") != user["user_id"]:
        raise HTTPException(403, "Access denied.")

    return {
        "task_id": task_id,
        "status": task.get("status"),
        **task.get("result", {}),
    }


# ─── Demo Helper ──────────────────────────────────────────────────────────────

def _demo_result(file_type: str) -> dict:
    import random
    import base64
    from io import BytesIO
    from PIL import Image, ImageDraw

    # Generate random scores but keep them somewhat consistent
    prob = round(random.uniform(0.3, 0.9), 4)
    is_fake = prob >= 0.5
    
    # Generate a dummy heatmap image instead of broken SVG
    img = Image.new('RGB', (224, 224), color=(30, 30, 40))
    d = ImageDraw.Draw(img)
    if is_fake:
        d.ellipse([50, 50, 174, 174], fill=(239, 68, 68, 128))
    else:
        d.ellipse([50, 50, 174, 174], fill=(6, 182, 212, 128))
    
    d.text((20, 100), "Trained Model Required", fill=(255, 255, 255))
    
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode()

    return {
        "label": "FAKE" if is_fake else "REAL",
        "deepfake_probability": prob,
        "confidence": round(abs(prob - 0.5) * 2, 4),
        "cnn_score": round(prob + random.uniform(-0.1, 0.1), 4),
        "vit_score": round(prob + random.uniform(-0.1, 0.1), 4),
        "temporal_score": round(prob + random.uniform(-0.1, 0.1), 4) if file_type == "video" else None,
        "model_agreement": round(random.uniform(0.7, 1.0), 4),
        "heatmap": img_str,
        "face_detected": True,
        "processing_time_sec": round(random.uniform(0.5, 2.5), 3),
        "input_type": file_type,
        "note": "Demo mode — Models are currently training",
    }


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)
