"""
ForenSight AI — Pydantic Response Schemas
Used for serializing API responses consistently.

Author: ForenSight AI Team
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class UserOut(BaseModel):
    user_id: str
    name: str
    email: str
    photo_url: Optional[str] = None
    created_at: str


class HistoryItem(BaseModel):
    """Lightweight summary for history list — no heatmap (too large)."""
    analysis_id: str
    file_name: str
    file_type: str
    result: Optional[str] = None         # 'REAL' | 'FAKE'
    confidence_score: Optional[float] = None
    deepfake_probability: Optional[float] = None
    face_detected: Optional[bool] = None
    processing_time_sec: Optional[float] = None
    created_at: str


class AnalysisOut(BaseModel):
    """Full analysis result including heatmap."""
    analysis_id: str
    file_name: str
    file_type: str
    result: Optional[str] = None
    confidence_score: Optional[float] = None
    deepfake_probability: Optional[float] = None
    cnn_score: Optional[float] = None
    vit_score: Optional[float] = None
    temporal_score: Optional[float] = None
    model_agreement: Optional[float] = None
    face_detected: Optional[bool] = None
    frames_analyzed: Optional[int] = None
    processing_time_sec: Optional[float] = None
    heatmap_path: Optional[str] = None   # base64 PNG
    created_at: str


class DetectRequest(BaseModel):
    task_id: str

class LoginRequest(BaseModel):
    id_token: str   # Firebase ID token from frontend
