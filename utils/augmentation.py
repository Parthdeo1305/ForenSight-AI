"""
Anti-Gravity Deepfake Detection System
Utility: Data Augmentation Pipeline

Albumentations-based augmentation specifically designed for deepfake detection:
  - Standard geometric augmentations (flip, rotate)
  - Color and lighting perturbations
  - Compression artifact simulation (JPEG quality)
  - Gaussian noise and blur
  - CoarseDropout (occlusion simulation)
  - ImageNet normalization

Author: Anti-Gravity Team
"""

import numpy as np
from typing import Callable, Dict, Optional, Tuple

import albumentations as A
from albumentations.pytorch import ToTensorV2


# ─── ImageNet normalization constants ──────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_train_transforms(
    image_size: int = 224,
    horizontal_flip_prob: float = 0.5,
    brightness_contrast_prob: float = 0.3,
    gaussian_blur_prob: float = 0.2,
    jpeg_quality_min: int = 70,
    jpeg_quality_max: int = 100,
    coarse_dropout_prob: float = 0.2,
    rotation_limit: int = 10,
) -> A.Compose:
    """
    Training augmentation pipeline.
    
    Why these augmentations for deepfake detection?
    - HorizontalFlip: Deepfakes often have handedness artifacts; flipping helps generalize
    - JPEG compression: Simulates post-processing encoding that hides deepfake traces
    - GaussianBlur: Handles varying video encoding quality
    - ColorJitter: Handles monitor/camera color variations
    - CoarseDropout: Forces model to use global context, not single patch
    - RandomRotate: Handles non-frontal face captures
    
    Args:
        image_size: Target image size (default 224)
        horizontal_flip_prob: Probability of horizontal flip
        brightness_contrast_prob: Probability of brightness-contrast jitter
        gaussian_blur_prob: Probability of Gaussian blur
        jpeg_quality_min: Min JPEG quality for compression simulation
        jpeg_quality_max: Max JPEG quality for compression simulation
        coarse_dropout_prob: Probability of coarse dropout
        rotation_limit: Max rotation angle in degrees
    
    Returns:
        albumentations Compose pipeline
    """
    return A.Compose([
        # Resize to target size (face should already be cropped)
        A.Resize(image_size, image_size),

        # Geometric augmentations
        A.HorizontalFlip(p=horizontal_flip_prob),
        A.Rotate(limit=rotation_limit, p=0.3, border_mode=0),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=5,
            p=0.2,
        ),

        # Color and lighting augmentations
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=brightness_contrast_prob,
        ),
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=10,
            p=0.2,
        ),
        A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.2),
        A.RandomGamma(gamma_limit=(80, 120), p=0.2),

        # Noise and blur — simulates camera/video encoding artifacts
        A.GaussianBlur(blur_limit=(3, 5), p=gaussian_blur_prob),
        A.GaussNoise(var_limit=(10, 40), p=0.2),
        A.MotionBlur(blur_limit=5, p=0.1),

        # JPEG compression simulation — critical for deepfake robustness
        A.ImageCompression(
            quality_lower=jpeg_quality_min,
            quality_upper=jpeg_quality_max,
            p=0.3,
        ),

        # Occlusion simulation (forces model to use global context)
        A.CoarseDropout(
            max_holes=4,
            max_height=20,
            max_width=20,
            min_holes=1,
            min_height=8,
            min_width=8,
            fill_value=0,
            p=coarse_dropout_prob,
        ),

        # Normalization and tensor conversion
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int = 224) -> A.Compose:
    """
    Validation/inference augmentation pipeline (no random augmentations).
    Only resize + normalize + ToTensor.
    
    Args:
        image_size: Target image size
    
    Returns:
        albumentations Compose pipeline
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_tta_transforms(image_size: int = 224) -> list:
    """
    Test-Time Augmentation (TTA) transforms.
    Returns a list of transforms to average predictions over at inference.
    
    TTA improves accuracy ~1-2% by averaging predictions over augmented copies.
    
    Args:
        image_size: Target image size
    
    Returns:
        List of albumentations Compose pipelines
    """
    base = [A.Resize(image_size, image_size), A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]
    return [
        A.Compose(base),
        A.Compose([A.Resize(image_size, image_size), A.HorizontalFlip(p=1.0), A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]),
        A.Compose([A.Resize(image_size, image_size), A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0), A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]),
    ]


def build_transforms_from_config(config: dict) -> Tuple[A.Compose, A.Compose]:
    """
    Build train and val transforms from config dictionary.
    
    Args:
        config: Full config dict (from config.yaml)
    
    Returns:
        Tuple of (train_transform, val_transform)
    """
    aug = config.get("augmentation", {})
    image_size = config.get("model", {}).get("image_size", 224)

    train_tf = get_train_transforms(
        image_size=image_size,
        horizontal_flip_prob=aug.get("horizontal_flip_prob", 0.5),
        brightness_contrast_prob=aug.get("brightness_contrast_prob", 0.3),
        gaussian_blur_prob=aug.get("gaussian_blur_prob", 0.2),
        jpeg_quality_min=aug.get("jpeg_quality_min", 70),
        jpeg_quality_max=aug.get("jpeg_quality_max", 100),
        coarse_dropout_prob=aug.get("coarse_dropout_prob", 0.2),
        rotation_limit=aug.get("rotation_limit", 10),
    )
    val_tf = get_val_transforms(image_size=image_size)

    return train_tf, val_tf


if __name__ == "__main__":
    # Smoke test
    import numpy as np
    dummy_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

    train_tf = get_train_transforms()
    val_tf = get_val_transforms()

    result_train = train_tf(image=dummy_img)
    result_val = val_tf(image=dummy_img)

    print(f"Train tensor shape: {result_train['image'].shape}")  # (3, 224, 224)
    print(f"Val tensor shape:   {result_val['image'].shape}")    # (3, 224, 224)
    print("Augmentation pipeline smoke test PASSED ✓")
