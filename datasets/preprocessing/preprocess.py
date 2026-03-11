"""
Anti-Gravity Deepfake Detection System
Dataset Preprocessing Pipeline

Full end-to-end pipeline:
  1. Read raw video files from dataset directories
  2. Extract frames at configurable FPS
  3. Detect and align faces using MTCNN
  4. Resize to 224×224
  5. Save PNG faces + generate CSV manifests

Usage:
    python datasets/preprocessing/preprocess.py \
        --dataset_root /path/to/datasets \
        --output_root /path/to/processed \
        --fps 1.0 \
        --split train

Author: Anti-Gravity Team
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.face_detection import FaceDetector
from utils.video_utils import extract_frames


# ─── Dataset Configuration ────────────────────────────────────────────────────

DATASET_CONFIG = {
    "faceforensics": {
        "description": "FaceForensics++ — face swap and expression manipulation",
        "real_subdir": "original_sequences/youtube/c23/videos",
        "fake_subdir": "manipulated_sequences/Deepfakes/c23/videos",
        "train_split": 0.35,  # proportion of total training data
    },
    "dfdc": {
        "description": "DeepFake Detection Challenge — diverse real-world deepfakes",
        "real_subdir": "real",
        "fake_subdir": "fake",
        "train_split": 0.35,
    },
    "celebdf": {
        "description": "Celeb-DF v2 — high-quality celebrity face swaps",
        "real_subdir": "Celeb-real",
        "fake_subdir": "Celeb-synthesis",
        "train_split": 0.20,
    },
    "forgerynet": {
        "description": "ForgeryNet — diverse forgery types including partial manipulation",
        "real_subdir": "real",
        "fake_subdir": "fake",
        "train_split": 0.10,
    },
}


def process_single_video(
    video_path: Path,
    output_dir: Path,
    label: int,
    source: str,
    detector: FaceDetector,
    target_fps: float = 1.0,
    max_frames: int = 30,
) -> List[Dict]:
    """
    Process one video: extract frames → detect face → save PNG.
    
    Returns list of records for CSV manifest.
    """
    records = []
    video_name = video_path.stem
    video_output = output_dir / source / ("real" if label == 0 else "fake") / video_name
    video_output.mkdir(parents=True, exist_ok=True)

    try:
        frames = extract_frames(str(video_path), target_fps=target_fps, max_frames=max_frames)
    except Exception as e:
        print(f"[Preprocess] Failed to extract frames from {video_path}: {e}")
        return records

    for frame_idx, frame_rgb in enumerate(frames):
        face_np, _ = detector.detect(frame_rgb)
        if face_np is None:
            continue

        frame_filename = f"frame_{frame_idx:04d}.png"
        save_path = video_output / frame_filename

        try:
            Image.fromarray(face_np).save(save_path)
            records.append({
                "path": str(save_path),
                "label": label,
                "source": source,
                "video": video_name,
                "frame": frame_idx,
            })
        except Exception:
            continue

    return records


def process_dataset(
    dataset_name: str,
    dataset_root: Path,
    output_root: Path,
    fps: float = 1.0,
    max_frames_per_video: int = 30,
    num_workers: int = 4,
) -> List[Dict]:
    """
    Process a complete dataset directory.
    
    Args:
        dataset_name: Dataset key from DATASET_CONFIG
        dataset_root: Root path to this dataset
        output_root: Root path for processed output
        fps: Frames per second to extract
        max_frames_per_video: Max frames from each video
        num_workers: Parallel processing workers
    
    Returns:
        List of manifest records
    """
    config = DATASET_CONFIG.get(dataset_name, {})
    real_dir = dataset_root / config.get("real_subdir", "real")
    fake_dir = dataset_root / config.get("fake_subdir", "fake")

    detector = FaceDetector(target_size=224)

    all_records = []
    tasks = []

    # Collect real and fake video paths
    for label, vid_dir in [(0, real_dir), (1, fake_dir)]:
        if not vid_dir.exists():
            print(f"[Preprocess] WARNING: {vid_dir} does not exist. Skipping.")
            continue
        video_exts = (".mp4", ".avi", ".mov", ".mkv")
        for vp in vid_dir.glob("**/*"):
            if vp.suffix.lower() in video_exts:
                tasks.append((vp, label))

    print(f"[Preprocess] {dataset_name}: {len(tasks)} videos to process")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                process_single_video,
                video_path=vp,
                output_dir=output_root,
                label=label,
                source=dataset_name,
                detector=detector,
                target_fps=fps,
                max_frames=max_frames_per_video,
            ): vp
            for vp, label in tasks
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {dataset_name}"):
            try:
                records = future.result()
                all_records.extend(records)
            except Exception as e:
                print(f"[Preprocess] Worker error: {e}")

    return all_records


def split_manifest(
    records: List[Dict],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    output_dir: Path = Path("datasets/manifests"),
    seed: int = 42,
):
    """
    Split records into train/val/test and save CSV manifests.
    
    Stratified by label to maintain class balance across splits.
    """
    import random
    random.seed(seed)

    real = [r for r in records if r["label"] == 0]
    fake = [r for r in records if r["label"] == 1]

    def split_list(items):
        random.shuffle(items)
        n = len(items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        return items[:n_train], items[n_train:n_train+n_val], items[n_train+n_val:]

    real_tr, real_va, real_te = split_list(real)
    fake_tr, fake_va, fake_te = split_list(fake)

    splits = {
        "train": real_tr + fake_tr,
        "val":   real_va + fake_va,
        "test":  real_te + fake_te,
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_records in splits.items():
        random.shuffle(split_records)
        out_path = output_dir / f"{split_name}.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "label", "source", "video", "frame"])
            writer.writeheader()
            writer.writerows(split_records)

        n_real = sum(1 for r in split_records if r["label"] == 0)
        n_fake = sum(1 for r in split_records if r["label"] == 1)
        print(f"[Preprocess] {split_name:5s}: {len(split_records):6,} samples (Real: {n_real:,} | Fake: {n_fake:,})")

    return splits


# ─── CLI Entry Point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Anti-Gravity Dataset Preprocessing Pipeline")
    parser.add_argument("--dataset_root", type=str, required=True, help="Root directory containing all datasets")
    parser.add_argument("--output_root", type=str, default="datasets/processed", help="Output directory for processed faces")
    parser.add_argument("--fps", type=float, default=1.0, help="Frames per second to extract from videos")
    parser.add_argument("--max_frames", type=int, default=30, help="Max frames per video")
    parser.add_argument("--num_workers", type=int, default=4, help="Parallel processing workers")
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_CONFIG.keys()),
                        choices=list(DATASET_CONFIG.keys()), help="Which datasets to process")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    all_records = []
    for ds_name in args.datasets:
        ds_path = dataset_root / ds_name
        if not ds_path.exists():
            print(f"[Preprocess] Dataset '{ds_name}' not found at {ds_path}. Skipping.")
            continue

        records = process_dataset(
            dataset_name=ds_name,
            dataset_root=ds_path,
            output_root=output_root,
            fps=args.fps,
            max_frames_per_video=args.max_frames,
            num_workers=args.num_workers,
        )
        all_records.extend(records)
        print(f"[Preprocess] {ds_name}: {len(records):,} frames extracted")

    print(f"\n[Preprocess] Total: {len(all_records):,} face images")

    split_manifest(
        all_records,
        output_dir=Path("datasets/manifests"),
    )

    print("\n[Preprocess] Done! CSV manifests saved to datasets/manifests/")
    print("Next step: python training/train.py --model cnn --config training/config.yaml")


if __name__ == "__main__":
    main()
