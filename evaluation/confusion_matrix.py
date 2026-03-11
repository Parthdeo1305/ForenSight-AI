"""
Anti-Gravity Deepfake Detection System
Evaluation: Confusion Matrix Visualization

Generates styled confusion matrix plots for model evaluation.
Supports single-model and multi-model comparison grids.

Author: Anti-Gravity Team
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    class_names: List[str] = ["REAL", "FAKE"],
    save_path: Optional[str] = None,
    normalize: bool = True,
) -> plt.Figure:
    """
    Plot a styled confusion matrix.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        model_name: Model name for title
        class_names: List of class labels
        save_path: Path to save figure
        normalize: If True, show percentages instead of counts
    
    Returns:
        Matplotlib Figure
    """
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm_display = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = ".1%"
    else:
        cm_display = cm
        fmt = "d"

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#0F172A")

    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Purples",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.5,
        linecolor="#374151",
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 14, "weight": "bold", "color": "white"},
    )

    ax.set_xlabel("Predicted Label", color="white", fontsize=12, labelpad=10)
    ax.set_ylabel("True Label", color="white", fontsize=12, labelpad=10)
    ax.set_title(f"Confusion Matrix — {model_name}", color="white", fontsize=13, fontweight="bold", pad=15)
    ax.tick_params(colors="white", labelsize=11)

    # Style colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.set_tick_params(colors="white")
    cbar.outline.set_edgecolor("#374151")

    # Add TN, FP, FN, TP annotations below matrix
    tn, fp, fn, tp = cm.ravel()
    summary = f"TN={tn} | FP={fp} | FN={fn} | TP={tp}"
    fig.text(0.5, 0.01, summary, ha="center", va="bottom",
             color="#94A3B8", fontsize=9)

    plt.tight_layout(rect=[0, 0.03, 1, 1])

    if save_path:
        os.makedirs(Path(save_path).parent, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Confusion matrix saved: {save_path}")

    return fig


def plot_multi_model_confusion_matrices(
    model_predictions: Dict[str, Tuple[np.ndarray, np.ndarray]],
    class_names: List[str] = ["REAL", "FAKE"],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot confusion matrices for multiple models in a grid layout.
    
    Args:
        model_predictions: Dict mapping model_name → (y_true, y_pred)
        class_names: List of class labels
        save_path: Path to save figure
    
    Returns:
        Matplotlib Figure with grid of confusion matrices
    """
    n_models = len(model_predictions)
    n_cols = min(2, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    fig.patch.set_facecolor("#0F172A")

    if n_models == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [axes]

    for idx, (model_name, (y_true, y_pred)) in enumerate(model_predictions.items()):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]
        ax.set_facecolor("#0F172A")

        cm = confusion_matrix(y_true, y_pred)
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        sns.heatmap(
            cm_pct,
            annot=True,
            fmt=".1%",
            cmap="Purples",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
            linewidths=0.5,
            linecolor="#374151",
            cbar=False,
            annot_kws={"size": 13, "weight": "bold", "color": "white"},
        )
        ax.set_title(model_name, color="white", fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted", color="#94A3B8", fontsize=10)
        ax.set_ylabel("True", color="#94A3B8", fontsize=10)
        ax.tick_params(colors="white")

    # Hide unused subplots
    for idx in range(n_models, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    fig.suptitle("Confusion Matrices — Ablation Study", color="white",
                 fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(Path(save_path).parent, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Multi-model confusion matrix saved: {save_path}")

    return fig


if __name__ == "__main__":
    # Demo with synthetic data
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 500)

    models = {
        "CNN (EfficientNet-B4)": 0.91,
        "ViT (Vision Transformer)": 0.92,
        "Temporal (CNN+LSTM)": 0.90,
        "Ensemble": 0.95,
    }

    preds = {}
    for name, acc in models.items():
        noise = np.random.rand(len(y_true)) > acc
        y_pred = y_true.copy()
        y_pred[noise] = 1 - y_pred[noise]
        preds[name] = (y_true, y_pred)

        # Single-model plot
        plot_confusion_matrix(
            y_true, y_pred,
            model_name=name,
            save_path=f"results/cm_{name.split()[0].lower()}.png"
        )

    # Multi-model grid
    plot_multi_model_confusion_matrices(preds, save_path="results/confusion_matrices_all.png")
    plt.show()
    print("Confusion matrix module PASSED ✓")
