"""
Anti-Gravity Deepfake Detection System
Utility: Video Frame Extraction

OpenCV-based frame extractor that samples video at configurable FPS.
Also supports uniform frame sampling for fixed-length clips.

Author: Anti-Gravity Team
"""

import os
from pathlib import Path
from typing import Generator, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


def extract_frames(
    video_path: str,
    target_fps: float = 1.0,
    max_frames: Optional[int] = None,
    start_sec: float = 0.0,
    end_sec: Optional[float] = None,
) -> List[np.ndarray]:
    """
    Extract frames from a video at the given FPS rate.
    
    Args:
        video_path: Path to video file
        target_fps: Number of frames to extract per second.
                   Use 1.0 for 1 frame/sec, 0.5 for 1 frame/2sec.
        max_frames: Maximum number of frames to extract (None = no limit)
        start_sec: Start time in seconds
        end_sec: End time in seconds (None = end of video)
    
    Returns:
        List of RGB numpy arrays (H, W, 3)
    
    Raises:
        FileNotFoundError: If video file does not exist
        RuntimeError: If video cannot be opened
    """
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    # Video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps

    # Clamp time range
    end_sec = min(end_sec or duration, duration)
    start_sec = max(0.0, start_sec)

    # Calculate which frame indices to extract
    frame_interval = max(1, int(video_fps / target_fps))
    start_frame = int(start_sec * video_fps)
    end_frame = int(end_sec * video_fps)

    frames: List[np.ndarray] = []
    frame_idx = 0
    extracted = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_idx = start_frame

    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if (frame_idx - start_frame) % frame_interval == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb_frame)
            extracted += 1

            if max_frames is not None and extracted >= max_frames:
                break

        frame_idx += 1

    cap.release()
    return frames


def extract_uniform_frames(
    video_path: str,
    num_frames: int = 16,
) -> List[np.ndarray]:
    """
    Extract exactly N uniformly-spaced frames from a video.
    Used for temporal model input (fixed-length clip).
    
    Args:
        video_path: Path to video file
        num_frames: Exact number of frames to extract
    
    Returns:
        List of exactly num_frames RGB numpy arrays (H, W, 3)
        (pads with last frame if video is shorter)
    """
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames = max(1, total_frames)

    # Uniform frame indices across the full video
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb)
        elif frames:
            # Pad with last valid frame
            frames.append(frames[-1].copy())

    cap.release()

    # Ensure exact num_frames (shouldn't happen but safety)
    while len(frames) < num_frames:
        frames.append(frames[-1].copy() if frames else np.zeros((224, 224, 3), dtype=np.uint8))

    return frames[:num_frames]


def get_video_info(video_path: str) -> dict:
    """
    Get video metadata without reading all frames.
    
    Args:
        video_path: Path to video file
    
    Returns:
        Dictionary with: fps, total_frames, duration_sec, width, height, codec
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": f"Cannot open {video_path}"}

    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "codec": int(cap.get(cv2.CAP_PROP_FOURCC)),
    }
    info["duration_sec"] = info["total_frames"] / max(info["fps"], 1.0)
    cap.release()
    return info


def frames_to_pil(frames: List[np.ndarray]) -> List[Image.Image]:
    """
    Convert list of numpy RGB frames to PIL Images.
    
    Args:
        frames: List of (H, W, 3) numpy arrays
    
    Returns:
        List of PIL Images (RGB)
    """
    return [Image.fromarray(f) for f in frames]


if __name__ == "__main__":
    print("Video utils module loaded. Use extract_frames() or extract_uniform_frames().")
    print("Functions: extract_frames, extract_uniform_frames, get_video_info, frames_to_pil")
