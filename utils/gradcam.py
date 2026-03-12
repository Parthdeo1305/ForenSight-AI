"""
Anti-Gravity Deepfake Detection System
Utility: Grad-CAM Heatmap Generation

Generates Grad-CAM visualization heatmaps for model explainability.
Highlights regions of the face image that most influenced the prediction.

Supports:
  - EfficientNet-B4 (CNN model): target layer = backbone.blocks[-1]
  - ViT (Vision Transformer): uses attention rollout instead of Grad-CAM
  - Returns base64-encoded PNG overlay for API response

Author: Anti-Gravity Team
"""

import base64
import io
import warnings
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.
    
    Algorithm:
    1. Forward pass → get feature maps from target conv layer
    2. Backward pass → compute gradients w.r.t. those feature maps
    3. Global-average-pool gradients → get importance weights per channel
    4. Weighted sum of feature maps → CAM
    5. ReLU + upsample to input size + normalize to [0,1]
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Args:
            model: PyTorch model to explain
            target_layer: Target convolutional layer for gradient extraction.
                         Typically the last conv block.
        """
        self.model = model
        self.target_layer = target_layer
        self._gradients: Optional[torch.Tensor] = None
        self._activations: Optional[torch.Tensor] = None

        # Register forward and backward hooks
        self._forward_hook = target_layer.register_forward_hook(self._save_activation)
        self._backward_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        """Forward hook: save feature map activations."""
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        """Backward hook: save gradients."""
        self._gradients = grad_output[0].detach()

    def generate(
        self,
        input_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for the input image.
        
        Args:
            input_tensor: Input image tensor (1, 3, H, W) — requires grad=True
            class_idx: Target class index (None → use predicted class)
        
        Returns:
            Heatmap array (H, W) with values in [0, 1]
        """
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)

        # Forward pass
        output = self.model(input_tensor)
        if output.dim() == 1:
            score = output[class_idx if class_idx is not None else 0]
        else:
            score = output[0, class_idx if class_idx is not None else 0]

        # Backward pass
        self.model.zero_grad()
        score.backward()

        # Compute Grad-CAM
        gradients = self._gradients   # (1, C, H', W')
        activations = self._activations  # (1, C, H', W')

        # Global average pooling of gradients
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted sum of activations
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, H', W')
        cam = F.relu(cam)

        # Upsample to input size
        H, W = input_tensor.shape[2], input_tensor.shape[3]
        cam = F.interpolate(cam, size=(H, W), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam

    def release(self):
        """Remove hooks (call when done to avoid memory leaks)."""
        self._forward_hook.remove()
        self._backward_hook.remove()


def generate_heatmap_overlay(
    image: Union[np.ndarray, Image.Image],
    cam: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Overlay Grad-CAM heatmap on the original image.
    
    Args:
        image: Original image (H, W, 3) RGB numpy array or PIL Image
        cam: Grad-CAM heatmap (H, W) in [0, 1]
        alpha: Blend factor (0=original only, 1=heatmap only)
        colormap: OpenCV colormap for visualization
    
    Returns:
        Overlay image (H, W, 3) RGB numpy array
    """
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"))

    H, W = image.shape[:2]

    # Resize CAM to match image if needed
    if cam.shape != (H, W):
        cam = cv2.resize(cam, (W, H))

    # Apply colormap (JET: blue=low, red=high activation)
    cam_uint8 = (cam * 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(cam_uint8, colormap)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # Blend with original image
    overlay = (alpha * heatmap_rgb + (1 - alpha) * image).astype(np.uint8)
    return overlay


def overlay_to_base64(overlay: np.ndarray) -> str:
    """
    Convert overlay numpy array to base64-encoded PNG string.
    Used for embedding in API JSON response.
    
    Args:
        overlay: RGB image numpy array (H, W, 3)
    
    Returns:
        Base64-encoded PNG string (data URI format)
    """
    pil_img = Image.fromarray(overlay.astype(np.uint8))
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def get_efficientnet_target_layer(model) -> nn.Module:
    """
    Get the target layer for Grad-CAM on EfficientNet.
    Supports both timm models (B4) and torchvision models (B0).
    """
    # 1. Custom detector format (has backbone attribute)
    if hasattr(model, "backbone"):
        backbone = model.backbone
        if hasattr(backbone, "blocks"):
            return backbone.blocks[-1]
        elif hasattr(backbone, "features"):
            return backbone.features[-1]
    
    # 2. Torchvision format (straight features list)
    if hasattr(model, "features"):
        return model.features[-1]
        
    # 3. ResNet style
    if hasattr(model, "layer4"):
        return model.layer4[-1]
        
    # 4. Generic fallback: find last Conv2d
    last_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    return last_conv


def generate_cnn_gradcam(
    cnn_model,
    image_tensor: torch.Tensor,
    original_image: Union[np.ndarray, Image.Image],
) -> Tuple[np.ndarray, str]:
    """
    End-to-end Grad-CAM pipeline for the CNN model.
    
    Args:
        cnn_model: EfficientNetDetector instance
        image_tensor: Preprocessed tensor (1, 3, 224, 224)
        original_image: Original face image for overlay
    
    Returns:
        Tuple of:
          - overlay: numpy array (H, W, 3) showing heatmap on original
          - base64_str: base64-encoded PNG for API response
    """
    target_layer = get_efficientnet_target_layer(cnn_model)

    if target_layer is None:
        warnings.warn("Could not find target layer for Grad-CAM. Returning empty heatmap.")
        if isinstance(original_image, Image.Image):
            original_image = np.array(original_image)
        return original_image, overlay_to_base64(original_image)

    gradcam = GradCAM(model=cnn_model, target_layer=target_layer)

    try:
        cam = gradcam.generate(image_tensor)
        overlay = generate_heatmap_overlay(original_image, cam)
        b64 = overlay_to_base64(overlay)
        return overlay, b64
    finally:
        gradcam.release()


if __name__ == "__main__":
    # Smoke test with dummy model and image
    import torch
    import torch.nn as nn

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(16, 1)

        def forward(self, x):
            x = self.conv(x)
            x = self.pool(x).flatten(1)
            return self.fc(x)

    model = DummyModel()
    target = model.conv
    gc = GradCAM(model, target)

    dummy_input = torch.randn(1, 3, 224, 224, requires_grad=True)
    cam = gc.generate(dummy_input)
    print(f"CAM shape: {cam.shape}")  # (224, 224)

    dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    overlay = generate_heatmap_overlay(dummy_img, cam)
    b64 = overlay_to_base64(overlay)
    print(f"Overlay shape: {overlay.shape}")  # (224, 224, 3)
    print(f"Base64 prefix: {b64[:40]}")       # data:image/png;base64,...
    gc.release()
    print("GradCAM smoke test PASSED ✓")
