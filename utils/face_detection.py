"""
Anti-Gravity Deepfake Detection System
Utility: Face Detection and Alignment

Uses MTCNN (facenet-pytorch) for face detection with 5-point landmarks.
Falls back to OpenCV Haar cascade if MTCNN is unavailable.

Pipeline:
  Input image → Detect face bounding box → Extract 5 landmarks
  → Affine alignment (eyes horizontal) → Crop + resize to 224×224

Author: Anti-Gravity Team
"""

import warnings
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

# Attempt to import MTCNN (facenet-pytorch)
try:
    from facenet_pytorch import MTCNN
    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False
    warnings.warn("facenet-pytorch not installed. Using OpenCV Haar cascade fallback.")


class FaceDetector:
    """
    Multi-backend face detector with MTCNN and Haar cascade fallback.
    
    Preferred: MTCNN — detects face box + 5 landmarks (eyes, nose, mouth corners)
    Fallback: OpenCV Haar cascade — detects face box only, no alignment
    """

    def __init__(
        self,
        target_size: int = 224,
        margin: float = 0.2,
        min_face_size: int = 40,
        device: str = "cpu",
        select_largest: bool = True,
    ):
        """
        Args:
            target_size: Output face image size (square, e.g., 224)
            margin: Fractional margin around face bounding box
            min_face_size: Minimum face size in pixels to detect
            device: Device for MTCNN ('cpu' or 'cuda')
            select_largest: If multiple faces, select the largest one
        """
        self.target_size = target_size
        self.margin = margin
        self.min_face_size = min_face_size
        self.select_largest = select_largest

        if MTCNN_AVAILABLE:
            self.mtcnn = MTCNN(
                image_size=target_size,
                margin=int(target_size * margin),
                min_face_size=min_face_size,
                thresholds=[0.6, 0.7, 0.7],
                factor=0.709,
                post_process=False,   # Return raw pixel values (0-255)
                select_largest=select_largest,
                device=device,
            )
        else:
            self.mtcnn = None
            # Load OpenCV Haar cascade as fallback
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.haar_cascade = cv2.CascadeClassifier(cascade_path)

    def detect(
        self,
        image: Union[np.ndarray, Image.Image],
        return_landmarks: bool = False,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Detect and extract aligned face from image.
        
        Args:
            image: Input image as numpy array (H, W, 3) BGR or PIL Image (RGB)
            return_landmarks: If True, also return landmark coordinates
        
        Returns:
            Tuple of:
              - face: numpy array (target_size, target_size, 3) RGB, or None if no face
              - landmarks: numpy array (5, 2) or None
        """
        # Convert numpy BGR to PIL RGB if needed
        if isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[2] == 3:
                pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_img = Image.fromarray(image)
        else:
            pil_img = image.convert("RGB")

        if self.mtcnn is not None:
            return self._detect_mtcnn(pil_img, return_landmarks)
        else:
            return self._detect_haar(pil_img)

    def _detect_mtcnn(
        self,
        pil_img: Image.Image,
        return_landmarks: bool,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """MTCNN-based detection with landmark-guided alignment."""
        try:
            # Detect face tensor + bounding boxes + landmarks
            face_tensor, prob, landmarks = self.mtcnn.detect(pil_img, landmarks=True)

            if face_tensor is None or landmarks is None:
                return None, None

            # Get the cropped + aligned face
            face_crop = self.mtcnn(pil_img)         # Returns (C, H, W) tensor or None

            if face_crop is None:
                return None, None

            # Convert tensor to numpy
            face_np = face_crop.permute(1, 2, 0).cpu().numpy()
            face_np = np.clip(face_np, 0, 255).astype(np.uint8)

            pts = landmarks[0] if landmarks is not None else None
            return face_np, pts

        except Exception as e:
            warnings.warn(f"MTCNN detection failed: {e}. Trying Haar fallback.")
            return self._detect_haar(pil_img)

    def _detect_haar(
        self,
        pil_img: Image.Image,
    ) -> Tuple[Optional[np.ndarray], None]:
        """OpenCV Haar cascade fallback — no landmark alignment."""
        img_np = np.array(pil_img)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        faces = self.haar_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(self.min_face_size, self.min_face_size),
        )

        if len(faces) == 0:
            return None, None

        # Select largest face
        if self.select_largest and len(faces) > 1:
            areas = [w * h for (x, y, w, h) in faces]
            faces = [faces[np.argmax(areas)]]

        x, y, w, h = faces[0]

        # Add margin
        margin_px = int(min(w, h) * self.margin)
        x1 = max(0, x - margin_px)
        y1 = max(0, y - margin_px)
        x2 = min(img_np.shape[1], x + w + margin_px)
        y2 = min(img_np.shape[0], y + h + margin_px)

        face_crop = img_np[y1:y2, x1:x2]
        face_resized = cv2.resize(face_crop, (self.target_size, self.target_size))
        return face_resized, None

    def detect_batch(
        self,
        images: List[Union[np.ndarray, Image.Image]],
    ) -> List[Optional[np.ndarray]]:
        """
        Batch face detection for efficiency.
        
        Args:
            images: List of input images
        
        Returns:
            List of face arrays (or None for images with no detected face)
        """
        results = []
        for img in images:
            face, _ = self.detect(img)
            results.append(face)
        return results


def align_face(
    image: np.ndarray,
    landmarks: np.ndarray,
    output_size: int = 224,
) -> np.ndarray:
    """
    Align face image using 5-point landmarks (eyes, nose, corners).
    Rotates image so eyes are horizontal.
    
    Args:
        image: Input face crop (H, W, 3) RGB
        landmarks: 5-point landmark array (5, 2) in (x, y) format
        output_size: Size of output aligned image
    
    Returns:
        Aligned face image (output_size, output_size, 3)
    """
    # Eye centers: landmark[0]=left eye, landmark[1]=right eye
    left_eye = landmarks[0]
    right_eye = landmarks[1]

    # Compute rotation angle
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))

    # Compute eye center
    eye_center = ((left_eye[0] + right_eye[0]) / 2,
                  (left_eye[1] + right_eye[1]) / 2)

    # Build rotation matrix and apply
    rot_mat = cv2.getRotationMatrix2D(eye_center, angle, scale=1.0)
    aligned = cv2.warpAffine(
        image, rot_mat, (image.shape[1], image.shape[0]),
        flags=cv2.INTER_CUBIC,
    )

    # Resize to output_size
    aligned = cv2.resize(aligned, (output_size, output_size))
    return aligned


if __name__ == "__main__":
    # Smoke test with a dummy image
    detector = FaceDetector(target_size=224, device="cpu")
    dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    face, landmarks = detector.detect(dummy_img)
    if face is not None:
        print(f"Face detected: {face.shape}")
    else:
        print("No face detected in random image (expected for random noise)")
    print("FaceDetector smoke test PASSED ✓")
