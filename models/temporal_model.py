"""
Anti-Gravity Deepfake Detection System
Model 3: Temporal CNN+LSTM for Video Analysis

Detects temporal inconsistencies across video frames:
  - Blinking anomalies (too fast, too slow, or absent)
  - Lip movement vs. audio desynchronization artifacts
  - Temporal flickering and frame-level rendering artifacts
  - Identity drift: subtle pose/expression changes between frames

Architecture:
  ResNet-18 frame feature extractor (512-dim per frame)
  → Bidirectional LSTM (2 layers, hidden=256)
  → Last hidden state (512-dim due to bidirectional)
  → Linear(512 → 128) + ReLU
  → Dropout(0.3)
  → Linear(128 → 1)

Author: Anti-Gravity Team
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, Tuple


class FrameFeatureExtractor(nn.Module):
    """
    ResNet-18 based per-frame spatial feature extractor.
    Strips the final classification layer and returns 512-dim features.
    """

    def __init__(self, pretrained: bool = True, freeze: bool = False):
        """
        Args:
            pretrained: Load ImageNet pretrained ResNet-18 weights
            freeze: Freeze all backbone params (train LSTM only initially)
        """
        super().__init__()

        # Load ResNet-18 and remove classifier
        resnet = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        # Remove avgpool and fc — we keep everything up to layer4
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = 512  # ResNet-18 final channel count

        if freeze:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract spatial features for a single frame.
        
        Args:
            x: Image tensor of shape (B, 3, H, W)
        
        Returns:
            Feature tensor of shape (B, 512)
        """
        x = self.feature_extractor(x)  # (B, 512, 7, 7)
        x = self.adaptive_pool(x)      # (B, 512, 1, 1)
        x = x.flatten(1)              # (B, 512)
        return x


class TemporalDetector(nn.Module):
    """
    Temporal deepfake detector using CNN frame features + Bidirectional LSTM.
    
    Why Bidirectional LSTM?
    - Forward pass: captures causal temporal patterns
    - Backward pass: considers future frames for context
    - Bidirectionality doubles effective capacity without doubling depth
    
    Why ResNet-18 (not EfficientNet)?
    - Faster inference for per-frame extraction (16 frames × batch)
    - Avoids gradient issues when training CNN+RNN jointly
    - EfficientNet is reserved for the dedicated spatial model
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        hidden_size: int = 256,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout_rate: float = 0.3,
        sequence_length: int = 16,
        num_classes: int = 1,
        freeze_cnn: bool = False,
    ):
        """
        Args:
            backbone: CNN backbone model name (currently only resnet18 supported)
            pretrained: Load pretrained CNN weights
            hidden_size: LSTM hidden state dimension (per direction)
            num_layers: Number of LSTM layers
            bidirectional: Use bidirectional LSTM
            dropout_rate: Dropout probability in classifier head
            sequence_length: Number of frames per clip
            num_classes: 1 for binary classification
            freeze_cnn: Freeze CNN and train only LSTM
        """
        super().__init__()

        self.sequence_length = sequence_length
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # ── Frame feature extractor ────────────────────────────────────
        self.frame_extractor = FrameFeatureExtractor(
            pretrained=pretrained,
            freeze=freeze_cnn,
        )
        frame_feature_dim = self.frame_extractor.feature_dim  # 512

        # ── Temporal LSTM ──────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=frame_feature_dim,       # 512
            hidden_size=hidden_size,            # 256
            num_layers=num_layers,              # 2
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # ── Classification head ────────────────────────────────────────
        lstm_out_dim = hidden_size * self.num_directions  # 512 (bidirectional)
        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_out_dim),
            nn.Linear(lstm_out_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize LSTM and classifier weights."""
        # LSTM orthogonal initialization
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        # Classifier initialization
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        return_sequence: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass for a video clip.
        
        Args:
            x: Video tensor of shape (B, T, 3, H, W)
               B=batch, T=sequence_length, H=W=224
            return_sequence: If True, return per-frame logits instead of final
        
        Returns:
            Logits tensor of shape (B, 1) [or (B, T, 1) if return_sequence=True]
        """
        B, T, C, H, W = x.shape

        # ── Extract per-frame features ─────────────────────────────────
        # Reshape to process all frames through CNN simultaneously
        x_flat = x.view(B * T, C, H, W)          # (B*T, 3, H, W)
        frame_features = self.frame_extractor(x_flat)  # (B*T, 512)
        frame_features = frame_features.view(B, T, -1)  # (B, T, 512)

        # ── LSTM temporal reasoning ────────────────────────────────────
        lstm_out, (h_n, c_n) = self.lstm(frame_features)  # lstm_out: (B, T, 512)

        if return_sequence:
            # Per-frame classification
            logits = self.classifier(lstm_out)  # (B, T, 1)
            return logits

        # Use concatenated final hidden states from both directions
        if self.bidirectional:
            # h_n: (num_layers*2, B, hidden) → take last forward+backward
            h_forward = h_n[-2, :, :]   # (B, 256) — last layer, forward
            h_backward = h_n[-1, :, :]  # (B, 256) — last layer, backward
            h_final = torch.cat([h_forward, h_backward], dim=1)  # (B, 512)
        else:
            h_final = h_n[-1, :, :]  # (B, 256)

        logits = self.classifier(h_final)  # (B, 1)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns deepfake probability for a video clip.
        
        Args:
            x: Video tensor of shape (B, T, 3, H, W) or (T, 3, H, W)
        
        Returns:
            Probability tensor of shape (B,)
        """
        if x.dim() == 4:
            x = x.unsqueeze(0)  # Add batch dim

        with torch.no_grad():
            logits = self.forward(x)
            probas = torch.sigmoid(logits).squeeze(1)
        return probas

    def extract_frame_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract per-frame CNN features — useful for ensemble feature fusion.
        
        Args:
            x: Video tensor of shape (B, T, 3, H, W)
        
        Returns:
            Frame features of shape (B, T, 512)
        """
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)
        feats = self.frame_extractor(x_flat)
        return feats.view(B, T, -1)

    def freeze_cnn(self):
        """Freeze CNN parameters, train only LSTM + classifier."""
        for param in self.frame_extractor.parameters():
            param.requires_grad = False

    def unfreeze_cnn(self):
        """Unfreeze CNN parameters for joint training."""
        for param in self.frame_extractor.parameters():
            param.requires_grad = True


def build_temporal_model(config: dict) -> TemporalDetector:
    """
    Factory function to build temporal model from config dict.
    
    Args:
        config: Dictionary with keys: temporal_backbone, temporal_pretrained,
                temporal_hidden_size, temporal_num_layers, temporal_bidirectional,
                sequence_length
    
    Returns:
        TemporalDetector instance
    """
    return TemporalDetector(
        backbone=config.get("temporal_backbone", "resnet18"),
        pretrained=config.get("temporal_pretrained", True),
        hidden_size=config.get("temporal_hidden_size", 256),
        num_layers=config.get("temporal_num_layers", 2),
        bidirectional=config.get("temporal_bidirectional", True),
        sequence_length=config.get("sequence_length", 16),
    )


if __name__ == "__main__":
    # Quick smoke test
    model = TemporalDetector(pretrained=False, sequence_length=8)
    model.eval()
    dummy = torch.randn(2, 8, 3, 224, 224)  # 2 clips, 8 frames each
    logits = model(dummy)
    probas = model.predict_proba(dummy)
    seq_logits = model(dummy, return_sequence=True)
    print(f"Logits shape:     {logits.shape}")       # (2, 1)
    print(f"Probas shape:     {probas.shape}")       # (2,)
    print(f"Seq logits shape: {seq_logits.shape}")   # (2, 8, 1)
    print("TemporalDetector smoke test PASSED ✓")
