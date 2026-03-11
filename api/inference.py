"""
Anti-Gravity Deepfake Detection System
Backend API: Inference Engine

Handles model loading, preprocessing, and inference for both images and videos.
Called by the FastAPI endpoints in app.py.

Author: Anti-Gravity Team
"""

import io
import os
import sys
import time
import uuid
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image

# Internal imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.ensemble_model import EnsembleDetector, build_ensemble_from_config
from utils.augmentation import get_val_transforms
from utils.face_detection import FaceDetector
from utils.video_utils import extract_frames, extract_uniform_frames
from utils.gradcam import generate_cnn_gradcam


class DeepfakeInferenceEngine:
    """
    Production inference engine for the Anti-Gravity deepfake detection system.
    
    Features:
    - Lazy model loading (loaded once at startup)
    - Face detection and alignment before inference
    - Grad-CAM heatmap generation for explainability
    - Structured JSON response generation
    - Error-resilient fallback behaviour
    """

    # Default ensemble weights if no config provided
    DEFAULT_CONFIG = {
        "model": {
            "cnn_weight": 0.40,
            "vit_weight": 0.30,
            "temporal_weight": 0.30,
            "image_size": 224,
            "sequence_length": 16,
        },
        "paths": {
            "weights_dir": "weights",
        }
    }

    def __init__(
        self,
        weights_dir: str = "weights",
        device: Optional[str] = None,
        enable_gradcam: bool = True,
    ):
        """
        Initialize inference engine.
        
        Args:
            weights_dir: Directory containing model checkpoint files
            device: Compute device ('cuda', 'cpu', or None for auto)
            enable_gradcam: Whether to compute Grad-CAM heatmaps
        """
        self.weights_dir = weights_dir
        self.enable_gradcam = enable_gradcam
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"[InferenceEngine] Device: {self.device}")

        # ── Preprocessing ──────────────────────────────────────────────
        self.face_detector = FaceDetector(target_size=224, device=str(self.device))
        self.val_transform = get_val_transforms(image_size=224)

        # ── Load ensemble model ────────────────────────────────────────
        self.ensemble = self._load_ensemble()

        print("[InferenceEngine] Ready for inference.")

    def _load_ensemble(self) -> EnsembleDetector:
        """Load ensemble model with available checkpoints."""
        cnn_ckpt = os.path.join(self.weights_dir, "cnn_best.pth")
        vit_ckpt = os.path.join(self.weights_dir, "vit_best.pth")
        tmp_ckpt = os.path.join(self.weights_dir, "temporal_best.pth")

        ensemble = EnsembleDetector(
            cnn_weight=0.40,
            vit_weight=0.30,
            temporal_weight=0.30,
            device=str(self.device),
        )

        load_status = ensemble.load_weights(
            cnn_checkpoint=cnn_ckpt if Path(cnn_ckpt).exists() else None,
            vit_checkpoint=vit_ckpt if Path(vit_ckpt).exists() else None,
            temporal_checkpoint=tmp_ckpt if Path(tmp_ckpt).exists() else None,
        )

        if not any(load_status.values()):
            warnings.warn(
                "[InferenceEngine] No trained checkpoints found in weights/. "
                "Running with random weights — train models first for real predictions. "
                "Demo mode: predictions will be random but the pipeline works correctly."
            )

        ensemble.to_device()
        ensemble.eval()
        return ensemble

    # ─── Image Inference ──────────────────────────────────────────────────────

    def predict_image(self, image_path: str) -> Dict:
        """
        Run deepfake detection on a single image.
        
        Args:
            image_path: Path to image file (JPG, PNG, WebP)
        
        Returns:
            Structured result dictionary
        """
        start_time = time.time()

        # Load image
        try:
            pil_img = Image.open(image_path).convert("RGB")
            img_np = np.array(pil_img)
        except Exception as e:
            return {"error": f"Failed to load image: {e}"}

        # Face detection
        face_np, landmarks = self.face_detector.detect(img_np)
        face_detected = face_np is not None

        if face_np is None:
            # Fall back to full image if no face detected
            face_np = cv2.resize(img_np, (224, 224))
            face_detected = False
            warnings.warn("[InferenceEngine] No face detected — using full image.")

        # Face image for display (original crop)
        face_display = face_np.copy()

        # Preprocess → tensor
        transformed = self.val_transform(image=face_np)
        face_tensor = transformed["image"].unsqueeze(0).to(self.device)  # (1, 3, 224, 224)

        # Ensemble inference
        with torch.no_grad():
            result = self.ensemble.predict_image(face_tensor)

        # Grad-CAM heatmap
        heatmap_b64 = None
        if self.enable_gradcam:
            try:
                _, heatmap_b64 = generate_cnn_gradcam(
                    self.ensemble.cnn_model,
                    face_tensor.clone(),
                    Image.fromarray(face_display),
                )
            except Exception as e:
                warnings.warn(f"[InferenceEngine] Grad-CAM failed: {e}")

        result.update({
            "face_detected": face_detected,
            "heatmap": heatmap_b64,
            "processing_time_sec": round(time.time() - start_time, 3),
            "input_type": "image",
        })

        return result

    # ─── Video Inference ──────────────────────────────────────────────────────

    def predict_video(
        self,
        video_path: str,
        sample_fps: float = 1.0,
        max_frames: int = 32,
    ) -> Dict:
        """
        Run deepfake detection on a video file.
        
        Args:
            video_path: Path to video file
            sample_fps: Frames per second to sample for CNN/ViT analysis
            max_frames: Maximum frames to process
        
        Returns:
            Structured result dictionary with per-frame scores
        """
        start_time = time.time()

        # ── Extract frames ─────────────────────────────────────────────
        try:
            frames_rgb = extract_frames(
                video_path,
                target_fps=sample_fps,
                max_frames=max_frames,
            )
        except Exception as e:
            return {"error": f"Failed to extract frames: {e}"}

        if not frames_rgb:
            return {"error": "No frames extracted from video."}

        # ── Extract clip for temporal model ────────────────────────────
        try:
            clip_frames = extract_uniform_frames(video_path, num_frames=16)
        except Exception:
            clip_frames = frames_rgb[:16] if len(frames_rgb) >= 16 else frames_rgb

        # ── Process each frame ─────────────────────────────────────────
        frame_tensors = []
        face_count = 0

        for frame_np in frames_rgb:
            face_np, _ = self.face_detector.detect(frame_np)
            if face_np is None:
                face_np = cv2.resize(frame_np, (224, 224))
            else:
                face_count += 1

            t = self.val_transform(image=face_np)["image"]
            frame_tensors.append(t)

        if not frame_tensors:
            return {"error": "No processable frames found."}

        # ── Aggregate frame predictions ────────────────────────────────
        batch = torch.stack(frame_tensors).to(self.device)  # (N, 3, H, W)
        per_frame_scores = []

        with torch.no_grad():
            # Process in mini-batches of 8 for memory efficiency
            for i in range(0, len(batch), 8):
                mini_batch = batch[i:i+8]
                cnn_logit = self.ensemble.cnn_model(mini_batch)
                cnn_prob = torch.sigmoid(cnn_logit).squeeze(1).cpu().numpy()
                per_frame_scores.extend(cnn_prob.tolist())

        # Representative frame for ensemble (use middle frame)
        mid = len(frame_tensors) // 2
        representative_frame = frame_tensors[mid].unsqueeze(0).to(self.device)

        # ── Temporal model on clip ─────────────────────────────────────
        clip_tensor = None
        if len(clip_frames) >= 8:
            clip_processed = []
            for cf in clip_frames[:16]:
                face_np, _ = self.face_detector.detect(cf)
                if face_np is None:
                    face_np = cv2.resize(cf, (224, 224))
                t = self.val_transform(image=face_np)["image"]
                clip_processed.append(t)

            while len(clip_processed) < 16:
                clip_processed.append(clip_processed[-1].clone())

            clip_tensor = torch.stack(clip_processed).unsqueeze(0).to(self.device)  # (1,16,3,H,W)

        # ── Full ensemble prediction ───────────────────────────────────
        with torch.no_grad():
            result = self.ensemble.predict_video(representative_frame, clip_tensor)

        # ── Grad-CAM on representative frame ───────────────────────────
        heatmap_b64 = None
        if self.enable_gradcam:
            try:
                face_np, _ = self.face_detector.detect(frames_rgb[mid])
                if face_np is None:
                    face_np = cv2.resize(frames_rgb[mid], (224, 224))
                _, heatmap_b64 = generate_cnn_gradcam(
                    self.ensemble.cnn_model,
                    representative_frame.clone(),
                    Image.fromarray(face_np),
                )
            except Exception as e:
                warnings.warn(f"[InferenceEngine] Grad-CAM failed: {e}")

        result.update({
            "face_detected": face_count > 0,
            "faces_detected_count": face_count,
            "frames_analyzed": len(frame_tensors),
            "per_frame_scores": [round(s, 4) for s in per_frame_scores],
            "heatmap": heatmap_b64,
            "processing_time_sec": round(time.time() - start_time, 3),
            "input_type": "video",
        })

        return result

    # ─── File Router ──────────────────────────────────────────────────────────

    def predict(self, file_path: str) -> Dict:
        """
        Auto-detect media type and run appropriate inference.
        
        Args:
            file_path: Path to image or video file
        
        Returns:
            Structured result dictionary
        """
        ext = Path(file_path).suffix.lower()
        video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".m4v"}
        image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

        if ext in image_exts:
            return self.predict_image(file_path)
        elif ext in video_exts:
            return self.predict_video(file_path)
        else:
            return {"error": f"Unsupported file format: {ext}"}


# ─── Singleton Instance ───────────────────────────────────────────────────────

_engine_instance: Optional[DeepfakeInferenceEngine] = None


def get_inference_engine(
    weights_dir: str = "weights",
    enable_gradcam: bool = True,
) -> DeepfakeInferenceEngine:
    """
    Get or create the global inference engine singleton.
    Ensures models are loaded only once at startup.
    
    Args:
        weights_dir: Directory with trained model checkpoints
        enable_gradcam: Enable Grad-CAM heatmap generation
    
    Returns:
        DeepfakeInferenceEngine instance
    """
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = DeepfakeInferenceEngine(
            weights_dir=weights_dir,
            enable_gradcam=enable_gradcam,
        )
    return _engine_instance


if __name__ == "__main__":
    # Smoke test
    engine = get_inference_engine()
    print("[Inference] Engine initialized successfully.")
    print("[Inference] Use engine.predict(file_path) to run detection.")
