"""
Anti-Gravity Deepfake Detection System
Evaluation & Error Analysis Script

Evaluates the trained ensemble model on a test set.
Computes comprehensive metrics: Accuracy, Precision, Recall, F1, ROC-AUC, FPR, FNR.
Performs Error Analysis by saving False Positives and False Negatives to disk for visual review.

Usage:
    python evaluation/evaluate.py --config training/config.yaml --manifest datasets/manifests/test.csv
"""

import argparse
import os
import sys
from pathlib import Path
import shutil

import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

# Internal imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.ensemble_model import build_ensemble_from_config
from utils.augmentation import get_val_transforms

def evaluate(config_path: str, manifest_path: str, output_dir: str):
    """Run full evaluation and error analysis on the test set."""
    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Evaluate] Starting on device: {device}")

    # Build model
    model = build_ensemble_from_config(config)
    model.eval()

    # Build transforms
    image_size = config.get("model", {}).get("image_size", 224)
    val_tf = get_val_transforms(image_size)

    # Load dataset
    df = pd.read_csv(manifest_path)
    print(f"[Evaluate] Loaded {len(df)} samples from {manifest_path}")

    # Setup error analysis directories
    fp_dir = os.path.join(output_dir, "false_positives")  # Real predicted as Fake
    fn_dir = os.path.join(output_dir, "false_negatives") # Fake predicted as Real
    os.makedirs(fp_dir, exist_ok=True)
    os.makedirs(fn_dir, exist_ok=True)

    y_true = []
    y_pred = []
    y_prob = []

    print("[Evaluate] Running inference...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        true_label = int(row["label"]) # 0=REAL, 1=FAKE
        img_path = row["path"]
        
        try:
            img = Image.open(img_path).convert("RGB")
            img_np = np.array(img)
            tensor = val_tf(image=img_np)["image"].unsqueeze(0) # (1, 3, 224, 224)

            result = model.predict_image(tensor)
            prob = result["deepfake_probability"]
            pred_label = 1 if prob >= 0.5 else 0

            y_true.append(true_label)
            y_pred.append(pred_label)
            y_prob.append(prob)

            # --- Error Analysis Save ---
            if true_label == 0 and pred_label == 1:
                # False Positive
                save_path = os.path.join(fp_dir, f"score_{prob:.2f}_{os.path.basename(img_path)}")
                shutil.copy2(img_path, save_path)
            elif true_label == 1 and pred_label == 0:
                # False Negative
                save_path = os.path.join(fn_dir, f"score_{prob:.2f}_{os.path.basename(img_path)}")
                shutil.copy2(img_path, save_path)

        except Exception as e:
            print(f"Failed to process {row['path']}: {e}")

    # Convert to pure numpy arrays for sklearn
    y_true_np = np.array(y_true, dtype=int)
    y_pred_np = np.array(y_pred, dtype=int)
    y_prob_np = np.array(y_prob, dtype=float)

    # Compute Metrics
    acc = accuracy_score(y_true_np, y_pred_np)
    prec = precision_score(y_true_np, y_pred_np, zero_division=0)
    rec = recall_score(y_true_np, y_pred_np, zero_division=0)
    f1 = f1_score(y_true_np, y_pred_np, zero_division=0)
    auc = roc_auc_score(y_true_np, y_prob_np)

    tn, fp, fn, tp = confusion_matrix(y_true_np, y_pred_np).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    print("\n" + "="*50)
    print("ForenSight AI — Evaluation Report")
    print("="*50)
    print(f"Total Samples: {len(y_true)}")
    print(f"Accuracy:      {acc:.4f}  (Target: >= 0.92)")
    print(f"Precision:     {prec:.4f}  (Target: >= 0.90)")
    print(f"Recall:        {rec:.4f}  (Target: >= 0.90)")
    print(f"F1 Score:      {f1:.4f}  (Target: >= 0.90)")
    print(f"ROC-AUC:       {auc:.4f}")
    print("-" * 50)
    print(f"False Positive Rate (FPR): {fpr:.4f} ({fp} photos falsely flagged as deepfake)")
    print(f"False Negative Rate (FNR): {fnr:.4f} ({fn} deepfakes missed)")
    print("="*50)
    print(f"\n[Error Analysis] Saved visual failure examples to: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Deepfake Model")
    parser.add_argument("--config", default="training/config.yaml", help="Path to config.yaml")
    parser.add_argument("--manifest", required=True, help="Path to test.csv manifest")
    parser.add_argument("--output_dir", default="results/error_analysis", help="Output dir for error analysis images")
    
    args = parser.parse_args()
    evaluate(args.config, args.manifest, args.output_dir)
