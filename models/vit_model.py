"""
Anti-Gravity Deepfake Detection System
Model 2: Vision Transformer (ViT-B/16)

Detects global structural inconsistencies using self-attention:
  - Long-range dependency violations (ears don't match face orientation)
  - Subtle manipulation patterns invisible to local convolutions
  - Global context inconsistencies in pose, lighting, and geometry
  - Attention to multi-scale features via patch-based processing

Architecture:
  ViT-B/16 backbone (pretrained on ImageNet-21k + ImageNet-1k)
  → [CLS] token feature (768-dim)
  → Linear(768 → 256) + GELU
  → Dropout(0.3)
  → Linear(256 → 1) + Sigmoid

Author: Anti-Gravity Team
"""

import torch
import torch.nn as nn
import timm
from typing import Dict, Optional, Tuple


class ViTDetector(nn.Module):
    """
    Vision Transformer (ViT-B/16) based deepfake detector.
    
    ViT was chosen because:
    1. Self-attention captures global inconsistencies CNNs miss
    2. Patch-based processing can detect inter-region inconsistencies
    3. Pretrained on large-scale data gives rich semantic features
    4. Attention maps provide natural interpretability
    
    Key difference from CNN: 
    - CNN: local → global (might miss long-range inconsistencies)
    - ViT: all patches attend to all others simultaneously
    """

    def __init__(
        self,
        backbone: str = "vit_base_patch16_224",
        pretrained: bool = True,
        dropout_rate: float = 0.3,
        num_classes: int = 1,
    ):
        """
        Args:
            backbone: timm ViT model name
            pretrained: Load pretrained weights (ImageNet-21k + ImageNet-1k)
            dropout_rate: Dropout probability in classifier head
            num_classes: 1 for binary classification
        """
        super().__init__()

        # ── Load ViT backbone ──────────────────────────────────────────
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,       # Remove original head
        )
        feature_dim = self.backbone.embed_dim  # 768 for ViT-B

        # ── Custom classification head ─────────────────────────────────
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, 256),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(p=dropout_rate * 0.5),
            nn.Linear(64, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize classification head weights."""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ViT.
        
        Args:
            x: Input tensor of shape (B, 3, 224, 224)
        
        Returns:
            Logits tensor of shape (B, 1)
        """
        # ViT returns [CLS] token representation
        features = self.backbone(x)          # (B, 768)
        logits = self.classifier(features)   # (B, 1)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns deepfake probability (0=REAL, 1=FAKE).
        
        Args:
            x: Input tensor of shape (B, 3, 224, 224)
        
        Returns:
            Probability tensor of shape (B,)
        """
        with torch.no_grad():
            logits = self.forward(x)
            probas = torch.sigmoid(logits).squeeze(1)
        return probas

    def get_feature_vector(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract [CLS] feature vector from ViT backbone.
        Used by ensemble model for feature-level fusion.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
        
        Returns:
            Feature tensor of shape (B, 768)
        """
        return self.backbone(x)

    def get_attention_maps(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract attention maps from the last transformer block.
        Useful for visualization of which patches the model attends to.
        
        Args:
            x: Input tensor of shape (1, 3, 224, 224)  — single image
        
        Returns:
            Dictionary with 'attention_maps' key containing attention weights
        """
        attention_maps = {}
        hooks = []

        def hook_fn(module, input, output):
            # Capture attention weights from MultiheadAttention
            if hasattr(output, "clone"):
                attention_maps["last_attn"] = output.detach()

        # Hook the last attention block
        last_block = self.backbone.blocks[-1]
        if hasattr(last_block, "attn"):
            hook = last_block.attn.register_forward_hook(hook_fn)
            hooks.append(hook)

        with torch.no_grad():
            _ = self.forward(x)

        for hook in hooks:
            hook.remove()

        return attention_maps

    def freeze_backbone(self):
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_last_n_blocks(self, n: int = 4):
        """
        Unfreeze the last N transformer blocks for gradual fine-tuning.
        
        Args:
            n: Number of transformer blocks to unfreeze from the top
        """
        blocks = self.backbone.blocks
        for block in blocks[-n:]:
            for param in block.parameters():
                param.requires_grad = True

        # Always unfreeze the norm layer and head
        for param in self.backbone.norm.parameters():
            param.requires_grad = True


def build_vit_model(config: dict) -> ViTDetector:
    """
    Factory function to build ViT model from config dict.
    
    Args:
        config: Dictionary with keys: vit_backbone, vit_pretrained, vit_dropout
    
    Returns:
        ViTDetector instance
    """
    return ViTDetector(
        backbone=config.get("vit_backbone", "vit_base_patch16_224"),
        pretrained=config.get("vit_pretrained", True),
        dropout_rate=config.get("vit_dropout", 0.3),
    )


if __name__ == "__main__":
    # Quick smoke test
    model = ViTDetector(pretrained=False)
    model.eval()
    dummy = torch.randn(2, 3, 224, 224)
    logits = model(dummy)
    probas = model.predict_proba(dummy)
    feats = model.get_feature_vector(dummy)
    print(f"Logits shape:   {logits.shape}")   # (2, 1)
    print(f"Probas shape:   {probas.shape}")   # (2,)
    print(f"Features shape: {feats.shape}")    # (2, 768)
    print("ViTDetector smoke test PASSED ✓")
