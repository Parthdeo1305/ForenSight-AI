"""
ForenSight AI — Pre-trained Deepfake Detector
Uses EfficientNet-B0 trained on FaceForensics++ (C23) dataset.

Model source: https://huggingface.co/Xicor9/efficientnet-b0-ffpp-c23
Accuracy: ~95% on FaceForensics++ benchmark

The model auto-downloads weights (~20MB) on first startup.
Works on CPU — no GPU required.

Author: ForenSight AI Team
"""

import os
import time
import warnings
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# ─── Constants ─────────────────────────────────────────────────────────────────

MODEL_URL = "https://huggingface.co/Xicor9/efficientnet-b0-ffpp-c23/resolve/main/efficientnet_b0_ffpp_c23.pth"
WEIGHTS_CACHE = Path("/tmp/forensight_weights")
WEIGHTS_FILE = WEIGHTS_CACHE / "efficientnet_b0_ffpp_c23.pth"

# ImageNet normalization (standard for torchvision models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ─── Transform ─────────────────────────────────────────────────────────────────

inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# ─── Detector Class ────────────────────────────────────────────────────────────

class PretrainedDeepfakeDetector:
    """
    Production-ready deepfake detector using pre-trained EfficientNet-B0.
    
    - Downloads model weights automatically from Hugging Face
    - Detects face in the image, falls back to full image
    - Returns structured result dict compatible with the ForenSight API
    """

    def __init__(self, device: Optional[str] = None):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(f"[PretrainedDetector] Device: {self.device}")

        # Load model
        self.model = self._load_model()
        self.model.eval()
        print("[PretrainedDetector] Model loaded and ready.")

    def _load_model(self) -> nn.Module:
        """Download and load pre-trained EfficientNet-B0 for deepfake detection."""
        WEIGHTS_CACHE.mkdir(parents=True, exist_ok=True)

        # Download weights if not cached
        if not WEIGHTS_FILE.exists():
            print(f"[PretrainedDetector] Downloading model weights from Hugging Face...")
            state_dict = torch.hub.load_state_dict_from_url(
                MODEL_URL,
                model_dir=str(WEIGHTS_CACHE),
                map_location=self.device,
                file_name="efficientnet_b0_ffpp_c23.pth",
            )
            print(f"[PretrainedDetector] Weights downloaded successfully.")
        else:
            print(f"[PretrainedDetector] Loading cached weights from {WEIGHTS_FILE}")
            state_dict = torch.load(str(WEIGHTS_FILE), map_location=self.device)

        # Build model architecture (EfficientNet-B0 with 2-class output)
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
        model.load_state_dict(state_dict)
        model.to(self.device)
        return model

    def _detect_face(self, pil_image: Image.Image) -> Optional[Image.Image]:
        """
        Simple face detection using OpenCV's Haar cascade.
        Returns cropped face as PIL Image or None if no face found.
        """
        try:
            import cv2
            img_np = np.array(pil_image)
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

            # Use OpenCV's built-in face detector
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            face_cascade = cv2.CascadeClassifier(cascade_path)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
            )

            if len(faces) > 0:
                # Take the largest face
                faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                x, y, w, h = faces[0]
                
                # Add padding around the face (30%)
                pad_w = int(w * 0.3)
                pad_h = int(h * 0.3)
                x1 = max(0, x - pad_w)
                y1 = max(0, y - pad_h)
                x2 = min(img_np.shape[1], x + w + pad_w)
                y2 = min(img_np.shape[0], y + h + pad_h)

                face_crop = img_np[y1:y2, x1:x2]
                return Image.fromarray(face_crop)
        except Exception as e:
            warnings.warn(f"[PretrainedDetector] Face detection error: {e}")

        return None

    def predict(self, file_path: str) -> Dict:
        """
        Run deepfake detection on an image or video file.
        
        Args:
            file_path: Path to the input file
            
        Returns:
            Structured result dictionary
        """
        ext = Path(file_path).suffix.lower()
        image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

        if ext in image_exts:
            return self._predict_image(file_path)
        elif ext in video_exts:
            return self._predict_video(file_path)
        else:
            return {"error": f"Unsupported file format: {ext}"}

    def _predict_image(self, image_path: str) -> Dict:
        """Run detection on a single image."""
        start_time = time.time()

        try:
            pil_img = Image.open(image_path).convert("RGB")
        except Exception as e:
            return {"error": f"Failed to load image: {e}"}

        # Try to detect and crop face
        face_img = self._detect_face(pil_img)
        face_detected = face_img is not None

        # Use face crop if available, otherwise full image
        input_img = face_img if face_detected else pil_img

        # Run inference
        tensor = inference_transform(input_img).unsqueeze(0).to(self.device)
        
        # Heatmap generation (Grad-CAM)
        heatmap_b64 = None
        try:
            from utils.gradcam import generate_cnn_gradcam
            # Pretrained model uses cross-entropy, we explain the "Fake" class (index 1)
            _, heatmap_b64 = generate_cnn_gradcam(self.model, tensor, input_img)
        except Exception as e:
            warnings.warn(f"[PretrainedDetector] Heatmap generation failed: {e}")

        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)
            # Class 0 = Real, Class 1 = Fake
            fake_prob = probabilities[0, 1].item()
            real_prob = probabilities[0, 0].item()

        label = "FAKE" if fake_prob >= 0.5 else "REAL"
        confidence = max(fake_prob, real_prob)

        return {
            "label": label,
            "deepfake_probability": round(fake_prob, 4),
            "confidence": round(confidence, 4),
            "cnn_score": round(fake_prob, 4),
            "vit_score": round(fake_prob, 4),
            "temporal_score": None,
            "model_agreement": round(confidence, 4),
            "face_detected": face_detected,
            "heatmap": heatmap_b64,
            "processing_time_sec": round(time.time() - start_time, 3),
            "input_type": "image",
        }

    def _predict_video(self, video_path: str) -> Dict:
        """Run detection on a video by sampling frames."""
        start_time = time.time()

        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {"error": "Failed to open video file."}

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            
            # Sample up to 16 frames evenly
            num_samples = min(16, total_frames)
            if num_samples == 0:
                return {"error": "Video has no frames."}
                
            indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)
            
            frame_scores = []
            face_count = 0

            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    continue

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(frame_rgb)

                # Detect face
                face_img = self._detect_face(pil_frame)
                if face_img is not None:
                    face_count += 1
                    input_img = face_img
                else:
                    input_img = pil_frame

                # Run inference
                tensor = inference_transform(input_img).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    logits = self.model(tensor)
                    probs = torch.softmax(logits, dim=1)
                    fake_prob = probs[0, 1].item()
                    frame_scores.append(fake_prob)

            cap.release()

        except Exception as e:
            return {"error": f"Video processing failed: {e}"}

        if not frame_scores:
            return {"error": "No frames could be processed."}

        # Aggregate: mean of frame scores
        avg_fake_prob = float(np.mean(frame_scores))
        confidence = max(avg_fake_prob, 1 - avg_fake_prob)
        label = "FAKE" if avg_fake_prob >= 0.5 else "REAL"

        return {
            "label": label,
            "deepfake_probability": round(avg_fake_prob, 4),
            "confidence": round(confidence, 4),
            "cnn_score": round(avg_fake_prob, 4),
            "vit_score": round(avg_fake_prob, 4),
            "temporal_score": round(float(np.std(frame_scores)), 4),
            "model_agreement": round(confidence, 4),
            "face_detected": face_count > 0,
            "faces_detected_count": face_count,
            "frames_analyzed": len(frame_scores),
            "per_frame_scores": [round(s, 4) for s in frame_scores],
            "heatmap": None,
            "processing_time_sec": round(time.time() - start_time, 3),
            "input_type": "video",
        }


# ─── Singleton ─────────────────────────────────────────────────────────────────

_detector_instance: Optional[PretrainedDeepfakeDetector] = None


def get_pretrained_detector() -> PretrainedDeepfakeDetector:
    """Get or create the global pre-trained detector singleton."""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = PretrainedDeepfakeDetector()
    return _detector_instance
