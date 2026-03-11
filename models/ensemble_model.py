"""
Anti-Gravity Deepfake Detection System
Ensemble Model — Weighted Multi-Model Combiner

Combines three specialized detectors:
  - CNN (EfficientNet-B4): spatial artifact detection (weight=0.40)
  - ViT (ViT-B/16):        global inconsistency detection (weight=0.30)
  - Temporal (ResNet+LSTM): temporal inconsistency detection (weight=0.30)

For images: CNN + ViT weighted average (renormalized)
For videos: All three models — CNN+ViT on individual frames, Temporal on clip

Output: final probability, per-model scores, confidence level, label

Author: Anti-Gravity Team
"""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np

from models.cnn_model import EfficientNetDetector
from models.vit_model import ViTDetector
from models.temporal_model import TemporalDetector


class EnsembleDetector(nn.Module):
    """
    Weighted ensemble combining CNN, ViT, and Temporal deepfake detectors.
    
    Ensemble rationale:
    - Each model exploits different complementary signals
    - CNN catches local spatial artifacts
    - ViT catches global structural violations via attention
    - Temporal catches time-domain inconsistencies
    - Disagreement among models → higher uncertainty → lower confidence
    
    Weighting strategy:
    - CNN (0.40): performs best on single-frame artifact detection
    - ViT (0.30): strong global reasoning, slightly less sensitive on small crops
    - Temporal (0.30): critical for video but not applicable to images
    """

    def __init__(
        self,
        cnn_weight: float = 0.40,
        vit_weight: float = 0.30,
        temporal_weight: float = 0.30,
        temperature: float = 1.5,
        device: Optional[str] = None,
    ):
        """
        Args:
            cnn_weight: Ensemble weight for CNN model
            vit_weight: Ensemble weight for ViT model
            temporal_weight: Ensemble weight for Temporal model
            temperature: Calibration temperature for scaling logits (T > 1.0 softens probs)
            device: Target device ('cuda', 'cpu', or None for auto-detect)
        """
        super().__init__()

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Store weights
        self.cnn_weight = cnn_weight
        self.vit_weight = vit_weight
        self.temporal_weight = temporal_weight
        self.temperature = temperature

        # Instantiate models (without pretrained weights — loaded from checkpoints)
        self.cnn_model = EfficientNetDetector(pretrained=False)
        self.vit_model = ViTDetector(pretrained=False)
        self.temporal_model = TemporalDetector(pretrained=False)

        # Flag: which models are loaded
        self._cnn_loaded = False
        self._vit_loaded = False
        self._temporal_loaded = False

    def load_weights(
        self,
        cnn_checkpoint: Optional[str] = None,
        vit_checkpoint: Optional[str] = None,
        temporal_checkpoint: Optional[str] = None,
    ) -> Dict[str, bool]:
        """
        Load model weights from checkpoint files.
        
        Args:
            cnn_checkpoint: Path to CNN model checkpoint (.pth)
            vit_checkpoint: Path to ViT model checkpoint (.pth)
            temporal_checkpoint: Path to Temporal model checkpoint (.pth)
        
        Returns:
            Dict indicating which models were successfully loaded
        """
        status = {"cnn": False, "vit": False, "temporal": False}

        if cnn_checkpoint and Path(cnn_checkpoint).exists():
            state = torch.load(cnn_checkpoint, map_location=self.device)
            # Support checkpoints saved as {'model': state_dict, ...}
            if "model" in state:
                state = state["model"]
            self.cnn_model.load_state_dict(state, strict=False)
            self._cnn_loaded = True
            status["cnn"] = True
            print(f"[Ensemble] CNN model loaded from {cnn_checkpoint}")

        if vit_checkpoint and Path(vit_checkpoint).exists():
            state = torch.load(vit_checkpoint, map_location=self.device)
            if "model" in state:
                state = state["model"]
            self.vit_model.load_state_dict(state, strict=False)
            self._vit_loaded = True
            status["vit"] = True
            print(f"[Ensemble] ViT model loaded from {vit_checkpoint}")

        if temporal_checkpoint and Path(temporal_checkpoint).exists():
            state = torch.load(temporal_checkpoint, map_location=self.device)
            if "model" in state:
                state = state["model"]
            self.temporal_model.load_state_dict(state, strict=False)
            self._temporal_loaded = True
            status["temporal"] = True
            print(f"[Ensemble] Temporal model loaded from {temporal_checkpoint}")

        return status

    def to_device(self):
        """Move all models to the configured device."""
        self.cnn_model = self.cnn_model.to(self.device)
        self.vit_model = self.vit_model.to(self.device)
        self.temporal_model = self.temporal_model.to(self.device)
        return self

    def predict_image(
        self,
        image_tensor: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Run inference on a single face image.
        Uses CNN + ViT (Temporal not applicable to single images).
        
        Args:
            image_tensor: Preprocessed tensor of shape (1, 3, 224, 224)
        
        Returns:
            Dictionary with prediction results
        """
        self.eval()
        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            # Apply Temperature Scaling to Logits before Sigmoid
            cnn_logit = self.cnn_model(image_tensor) / self.temperature
            cnn_score = torch.sigmoid(cnn_logit).item()

            vit_logit = self.vit_model(image_tensor) / self.temperature
            vit_score = torch.sigmoid(vit_logit).item()

        # Renormalize weights for image (no temporal)
        total_weight = self.cnn_weight + self.vit_weight
        w_cnn = self.cnn_weight / total_weight
        w_vit = self.vit_weight / total_weight

        ensemble_score = w_cnn * cnn_score + w_vit * vit_score

        return self._build_result(
            ensemble_score=ensemble_score,
            cnn_score=cnn_score,
            vit_score=vit_score,
            temporal_score=None,
            media_type="image",
        )

    def predict_video(
        self,
        frame_tensor: torch.Tensor,
        clip_tensor: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Run inference on a video.
        
        Args:
            frame_tensor: Representative frame tensor (1, 3, 224, 224)
                         — used for CNN+ViT scores
            clip_tensor: Full clip tensor (1, T, 3, 224, 224)
                        — used for Temporal model
                        If None, Temporal is skipped
        
        Returns:
            Dictionary with prediction results
        """
        self.eval()
        frame_tensor = frame_tensor.to(self.device)

        with torch.no_grad():
            # Apply Temperature Scaling Logits
            cnn_logit = self.cnn_model(frame_tensor) / self.temperature
            cnn_score = torch.sigmoid(cnn_logit).item()

            vit_logit = self.vit_model(frame_tensor) / self.temperature
            vit_score = torch.sigmoid(vit_logit).item()

            temporal_score = None
            if clip_tensor is not None and self._temporal_loaded:
                clip_tensor = clip_tensor.to(self.device)
                tmp_logit = self.temporal_model(clip_tensor) / self.temperature
                temporal_score = torch.sigmoid(tmp_logit).item()

        # Compute weighted ensemble
        if temporal_score is not None:
            ensemble_score = (
                self.cnn_weight * cnn_score
                + self.vit_weight * vit_score
                + self.temporal_weight * temporal_score
            )
        else:
            total = self.cnn_weight + self.vit_weight
            ensemble_score = (
                (self.cnn_weight / total) * cnn_score
                + (self.vit_weight / total) * vit_score
            )

        return self._build_result(
            ensemble_score=ensemble_score,
            cnn_score=cnn_score,
            vit_score=vit_score,
            temporal_score=temporal_score,
            media_type="video",
        )

    @staticmethod
    def _build_result(
        ensemble_score: float,
        cnn_score: float,
        vit_score: float,
        temporal_score: Optional[float],
        media_type: str,
        threshold: float = 0.5,
    ) -> Dict:
        """
        Build structured result dictionary.
        
        Confidence is derived from how far the score is from the decision
        boundary (0.5). High confidence = score far from 0.5.
        """
        label = "FAKE" if ensemble_score >= threshold else "REAL"
        # Confidence: calibrated distance from boundary
        confidence = min(abs(ensemble_score - threshold) * (1 / threshold), 1.0) if ensemble_score < threshold else \
                     min(abs(ensemble_score - threshold) * (1 / (1 - threshold)), 1.0)

        scores = [cnn_score, vit_score]
        if temporal_score is not None:
            scores.append(temporal_score)
        model_agreement = 1.0 - float(np.std(scores))

        return {
            "label": label,
            "deepfake_probability": round(ensemble_score, 4),
            "confidence": round(confidence, 4),
            "model_agreement": round(model_agreement, 4),
            "cnn_score": round(cnn_score, 4),
            "vit_score": round(vit_score, 4),
            "temporal_score": round(temporal_score, 4) if temporal_score is not None else None,
            "media_type": media_type,
            "threshold": threshold,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard PyTorch forward — treated as image inference.
        Returns ensemble logit for training compatibility.
        """
        cnn_logit = self.cnn_model(x)
        vit_logit = self.vit_model(x)

        total = self.cnn_weight + self.vit_weight
        w_cnn = self.cnn_weight / total
        w_vit = self.vit_weight / total

        # Weighted average in logit space (approximately)
        ensemble_logit = w_cnn * cnn_logit + w_vit * vit_logit
        return ensemble_logit


def build_ensemble_from_config(config: dict) -> EnsembleDetector:
    """
    Build ensemble model from config dictionary.
    Automatically loads available checkpoints from weights directory.
    
    Args:
        config: Full config dictionary (from config.yaml)
    
    Returns:
        EnsembleDetector with loaded weights
    """
    weights_dir = config.get("paths", {}).get("weights_dir", "weights")
    ensemble = EnsembleDetector(
        cnn_weight=config["model"]["cnn_weight"],
        vit_weight=config["model"]["vit_weight"],
        temporal_weight=config["model"]["temporal_weight"],
    )

    cnn_path = os.path.join(weights_dir, "cnn_best.pth")
    vit_path = os.path.join(weights_dir, "vit_best.pth")
    tmp_path = os.path.join(weights_dir, "temporal_best.pth")

    ensemble.load_weights(
        cnn_checkpoint=cnn_path if Path(cnn_path).exists() else None,
        vit_checkpoint=vit_path if Path(vit_path).exists() else None,
        temporal_checkpoint=tmp_path if Path(tmp_path).exists() else None,
    )
    ensemble.to_device()
    return ensemble


if __name__ == "__main__":
    # Smoke test — using random weights (no checkpoints)
    ensemble = EnsembleDetector(device="cpu")
    ensemble.cnn_model.eval()
    ensemble.vit_model.eval()
    ensemble.temporal_model.eval()

    # Image inference
    img = torch.randn(1, 3, 224, 224)
    img_result = ensemble.predict_image(img)
    print("Image result:", img_result)

    # Video inference
    frame = torch.randn(1, 3, 224, 224)
    clip = torch.randn(1, 8, 3, 224, 224)
    vid_result = ensemble.predict_video(frame, clip)
    print("Video result:", vid_result)
    print("EnsembleDetector smoke test PASSED ✓")
