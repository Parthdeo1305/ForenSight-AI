"""
Anti-Gravity Deepfake Detection System
Model 1: Spatial CNN using EfficientNet-B4

Detects spatial artifacts introduced by deepfake generation:
  - Blending edge artifacts at face boundaries
  - Skin tone inconsistencies and unnatural textures
  - Compression artifacts from GAN generation
  - Frequency-domain anomalies invisible to humans

Architecture:
  EfficientNet-B4 backbone (pretrained on ImageNet)
  → Global Average Pooling
  → Dropout(0.4)
  → Linear(1792 → 512) + ReLU
  → Dropout(0.3)
  → Linear(512 → 1) + Sigmoid

Author: Anti-Gravity Team
"""

import torch
import torch.nn as nn
import timm
from typing import Optional, Tuple


class EfficientNetDetector(nn.Module):
    """
    EfficientNet-B4 based deepfake detector for spatial artifact analysis.
    
    EfficientNet-B4 was chosen because:
    1. Best accuracy/efficiency trade-off on ImageNet
    2. Compound scaling captures both fine-grained textures and global structure
    3. Pretrained features transfer well to forgery detection
    4. ~19M parameters — manageable on consumer GPUs
    """

    def __init__(
        self,
        backbone: str = "efficientnet_b4",
        pretrained: bool = True,
        dropout_rate: float = 0.4,
        num_classes: int = 1,
        freeze_backbone: bool = False,
    ):
        """
        Args:
            backbone: timm model name (efficientnet_b4 default)
            pretrained: Load ImageNet pretrained weights
            dropout_rate: Dropout probability in classifier head
            num_classes: 1 for binary classification (REAL/FAKE)
            freeze_backbone: If True, freeze backbone and only train head
        """
        super().__init__()

        # ── Load backbone ──────────────────────────────────────────────
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,         # Remove original classifier
            global_pool="avg",     # Apply global average pooling
        )
        feature_dim = self.backbone.num_features  # 1792 for B4

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # ── Custom classification head ─────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=dropout_rate * 0.75),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

        # ── Weight initialization for custom head ──────────────────────
        self._init_weights()

    def _init_weights(self):
        """Initialize classifier head with Kaiming normal initialization."""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, 3, H, W) — normalized face image
        
        Returns:
            Logits tensor of shape (B, 1) — raw score before sigmoid
        """
        features = self.backbone(x)      # (B, 1792)
        logits = self.classifier(features)  # (B, 1)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns deepfake probability (0=REAL, 1=FAKE) after sigmoid.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
        
        Returns:
            Probability tensor of shape (B,)
        """
        with torch.no_grad():
            logits = self.forward(x)
            probas = torch.sigmoid(logits).squeeze(1)
        return probas

    def get_feature_vector(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract feature vector from backbone (used by ensemble for late fusion).
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
        
        Returns:
            Feature tensor of shape (B, 1792)
        """
        return self.backbone(x)

    def freeze_backbone(self):
        """Freeze all backbone parameters (useful for fine-tuning head only)."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def unfreeze_last_n_blocks(self, n: int = 3):
        """
        Unfreeze the last N blocks of the backbone for gradual fine-tuning.
        
        Args:
            n: Number of late blocks to unfreeze
        """
        blocks = list(self.backbone.blocks)
        for block in blocks[-n:]:
            for param in block.parameters():
                param.requires_grad = True


def build_cnn_model(config: dict) -> EfficientNetDetector:
    """
    Factory function to build CNN model from config dict.
    
    Args:
        config: Dictionary with keys: cnn_backbone, cnn_pretrained, cnn_dropout
    
    Returns:
        EfficientNetDetector instance
    """
    return EfficientNetDetector(
        backbone=config.get("cnn_backbone", "efficientnet_b4"),
        pretrained=config.get("cnn_pretrained", True),
        dropout_rate=config.get("cnn_dropout", 0.4),
    )


if __name__ == "__main__":
    # Quick smoke test
    model = EfficientNetDetector(pretrained=False)
    model.eval()
    dummy = torch.randn(2, 3, 224, 224)
    logits = model(dummy)
    probas = model.predict_proba(dummy)
    feats = model.get_feature_vector(dummy)
    print(f"Logits shape:   {logits.shape}")    # (2, 1)
    print(f"Probas shape:   {probas.shape}")    # (2,)
    print(f"Features shape: {feats.shape}")     # (2, 1792)
    print("EfficientNetDetector smoke test PASSED ✓")
