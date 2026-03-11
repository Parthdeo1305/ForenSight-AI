"""
Anti-Gravity Deepfake Detection System
Evaluation: Comprehensive Metrics

Computes all standard deepfake detection metrics:
  - Accuracy, Precision, Recall, F1-Score
  - ROC-AUC score and ROC curve plot
  - Per-dataset breakdown
  - Ablation study comparison across all models

Author: Anti-Gravity Team
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


# ─── Metric Computation ───────────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: Ground truth labels (0=REAL, 1=FAKE)
        y_pred: Predicted binary labels
        y_prob: Predicted probabilities for positive class (FAKE)
        threshold: Classification threshold
    
    Returns:
        Dictionary with all metric values
    """
    # Ensure binary predictions at threshold
    y_pred_thresh = (y_prob >= threshold).astype(int)

    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc_auc = float("nan")

    try:
        avg_precision = average_precision_score(y_true, y_prob)
    except Exception:
        avg_precision = float("nan")

    return {
        "accuracy": round(accuracy_score(y_true, y_pred_thresh), 4),
        "precision": round(precision_score(y_true, y_pred_thresh, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred_thresh, zero_division=0), 4),
        "f1_score": round(f1_score(y_true, y_pred_thresh, zero_division=0), 4),
        "roc_auc": round(roc_auc, 4),
        "average_precision": round(avg_precision, 4),
        "threshold": threshold,
        "n_samples": len(y_true),
        "n_real": int((y_true == 0).sum()),
        "n_fake": int((y_true == 1).sum()),
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Tuple[float, Dict[str, float]]:
    """
    Find optimal classification threshold via Youden's J statistic.
    Maximizes (sensitivity + specificity - 1).
    
    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities
    
    Returns:
        Tuple of (optimal_threshold, metrics_at_threshold)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_stat = tpr - fpr
    optimal_idx = np.argmax(j_stat)
    optimal_threshold = float(thresholds[optimal_idx])
    y_pred = (y_prob >= optimal_threshold).astype(int)
    metrics = compute_metrics(y_true, y_pred, y_prob, threshold=optimal_threshold)
    return optimal_threshold, metrics


# ─── ROC Curve Plot ───────────────────────────────────────────────────────────

def plot_roc_curve(
    model_results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    save_path: Optional[str] = None,
    title: str = "ROC Curves — Anti-Gravity Deepfake Detection",
) -> plt.Figure:
    """
    Plot ROC curves for multiple models on the same axis.
    
    Args:
        model_results: Dict mapping model_name → (y_true, y_prob)
        save_path: Path to save the figure
        title: Plot title
    
    Returns:
        Matplotlib Figure
    """
    COLORS = {
        "CNN (EfficientNet-B4)": "#7C3AED",
        "ViT (Vision Transformer)": "#06B6D4",
        "Temporal (CNN+LSTM)": "#10B981",
        "Ensemble": "#F59E0B",
    }

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_facecolor("#0F172A")
    fig.patch.set_facecolor("#0F172A")

    for model_name, (y_true, y_prob) in model_results.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        color = COLORS.get(model_name, "#FFFFFF")
        ax.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.3f})", color=color, linewidth=2)

    # Diagonal baseline
    ax.plot([0, 1], [0, 1], "w--", alpha=0.4, linewidth=1, label="Random (AUC=0.500)")

    ax.set_xlabel("False Positive Rate", color="white", fontsize=12)
    ax.set_ylabel("True Positive Rate", color="white", fontsize=12)
    ax.set_title(title, color="white", fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="lower right", facecolor="#1E293B", edgecolor="#7C3AED", labelcolor="white")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#374151")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(alpha=0.2, color="#374151")

    plt.tight_layout()
    if save_path:
        os.makedirs(Path(save_path).parent, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"ROC curve saved: {save_path}")

    return fig


# ─── Ablation Study ───────────────────────────────────────────────────────────

def run_ablation_study(
    model_metrics: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
) -> str:
    """
    Generate formatted ablation study table comparing all models.
    
    Args:
        model_metrics: Dict mapping model_name → metrics_dict (from compute_metrics)
        save_path: Path to save results as JSON
    
    Returns:
        Formatted markdown table string
    """
    header = (
        "| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |\n"
        "|-------|----------|-----------|--------|----------|---------|\n"
    )
    rows = []
    for model_name, m in model_metrics.items():
        row = (
            f"| **{model_name}** "
            f"| {m['accuracy']*100:.1f}% "
            f"| {m['precision']*100:.1f}% "
            f"| {m['recall']*100:.1f}% "
            f"| {m['f1_score']*100:.1f}% "
            f"| {m['roc_auc']*100:.1f}% |"
        )
        rows.append(row)

    table = header + "\n".join(rows)

    if save_path:
        os.makedirs(Path(save_path).parent, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(model_metrics, f, indent=2)
        print(f"Ablation results saved: {save_path}")

    return table


def plot_ablation_bar(
    model_metrics: Dict[str, Dict[str, float]],
    metric: str = "accuracy",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Bar chart comparing models on a single metric.
    
    Args:
        model_metrics: Dict mapping model_name → metrics_dict
        metric: Metric to plot ('accuracy', 'f1_score', 'roc_auc')
        save_path: Save path
    
    Returns:
        Matplotlib Figure
    """
    COLORS = ["#7C3AED", "#06B6D4", "#10B981", "#F59E0B"]
    models = list(model_metrics.keys())
    values = [model_metrics[m][metric] * 100 for m in models]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_facecolor("#0F172A")
    fig.patch.set_facecolor("#0F172A")

    bars = ax.bar(models, values, color=COLORS[:len(models)], edgecolor="#374151", linewidth=1.2)

    # Value labels on bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{val:.1f}%",
            ha="center", va="bottom",
            color="white", fontsize=11, fontweight="bold",
        )

    ax.set_ylabel(f"{metric.replace('_', ' ').title()} (%)", color="white", fontsize=12)
    ax.set_title(f"Ablation Study — {metric.replace('_', ' ').title()}", color="white", fontsize=14, fontweight="bold")
    ax.tick_params(colors="white", axis="both")
    ax.spines[:].set_color("#374151")
    ax.set_ylim([max(0, min(values) - 5), 100])
    ax.yaxis.grid(alpha=0.3, color="#374151")
    ax.set_axisbelow(True)

    plt.tight_layout()

    if save_path:
        os.makedirs(Path(save_path).parent, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())

    return fig


def save_metrics_report(
    metrics: Dict[str, float],
    model_name: str,
    output_dir: str = "results",
) -> str:
    """
    Save metrics to JSON and return formatted summary string.
    
    Args:
        metrics: Dict from compute_metrics()
        model_name: Model identifier
        output_dir: Directory to save results
    
    Returns:
        Formatted summary string
    """
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, f"{model_name}_metrics.json")

    with open(report_path, "w") as f:
        json.dump({**metrics, "model": model_name}, f, indent=2)

    summary = (
        f"\n{'='*50}\n"
        f"Model: {model_name}\n"
        f"{'='*50}\n"
        f"  Accuracy:   {metrics['accuracy']*100:.2f}%\n"
        f"  Precision:  {metrics['precision']*100:.2f}%\n"
        f"  Recall:     {metrics['recall']*100:.2f}%\n"
        f"  F1-Score:   {metrics['f1_score']*100:.2f}%\n"
        f"  ROC-AUC:    {metrics['roc_auc']*100:.2f}%\n"
        f"  Samples:    {metrics['n_samples']} (Real: {metrics['n_real']} | Fake: {metrics['n_fake']})\n"
        f"{'='*50}\n"
    )
    print(summary)
    return summary


if __name__ == "__main__":
    # Demonstration with synthetic data
    np.random.seed(42)
    n = 1000
    y_true = np.random.randint(0, 2, n)

    # Simulate model outputs with different performance levels
    model_results = {}
    simulated = {
        "CNN (EfficientNet-B4)": 0.91,
        "ViT (Vision Transformer)": 0.92,
        "Temporal (CNN+LSTM)": 0.90,
        "Ensemble": 0.95,
    }

    ablation_data = {}
    for name, perf in simulated.items():
        noise = np.random.randn(n) * 0.15
        y_prob = np.clip(y_true * perf + (1 - y_true) * (1 - perf) + noise, 0, 1)
        y_pred = (y_prob >= 0.5).astype(int)
        metrics = compute_metrics(y_true, y_pred, y_prob)
        ablation_data[name] = metrics
        model_results[name] = (y_true, y_prob)

    # Print ablation table
    table = run_ablation_study(ablation_data, save_path="results/ablation_results.json")
    print(table)

    # Plot ROC curves
    fig_roc = plot_roc_curve(model_results, save_path="results/roc_curves.png")

    # Plot ablation bar
    fig_bar = plot_ablation_bar(ablation_data, save_path="results/ablation_accuracy.png")

    plt.show()
    print("\nMetrics module demonstration PASSED ✓")
