from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_training_history(history: pd.DataFrame, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(12, 5))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(history["epoch"], history["train_rmse"], label="Train Dist-RMSE")
    ax1.plot(history["epoch"], history["test_rmse"], label="Test Dist-RMSE")
    ax1.set_title("Pairwise Distance RMSE per Epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Normalized RMSE")
    ax1.legend()

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(history["epoch"], history["train_mae"], label="Train Dist-MAE")
    ax2.plot(history["epoch"], history["test_mae"], label="Test Dist-MAE")
    ax2.set_title("Pairwise Distance MAE per Epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Normalized MAE")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_rmse_distribution(window_rmse: np.ndarray, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(window_rmse, bins=30, edgecolor="black", alpha=0.7)
    ax.set_title("Distribution of Test-Window Distance RMSE")
    ax.set_xlabel("Distance RMSE (normalized)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_true_vs_predicted(
    true_flat: np.ndarray,
    pred_flat: np.ndarray,
    output_path: str | Path,
    max_points: int = 10000,
    seed: int = 42,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    n = len(true_flat)
    idx = rng.choice(n, size=min(max_points, n), replace=False)

    x = true_flat[idx]
    y = pred_flat[idx]
    max_val = max(float(x.max()), float(y.max()))

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x, y, s=4, alpha=0.3)
    ax.plot([0, max_val], [0, max_val], "k--", linewidth=1)
    ax.set_title("Scatter: True vs Predicted Distances")
    ax.set_xlabel("True Distance (normalized)")
    ax.set_ylabel("Predicted Distance (normalized)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_nussinov_arc(sequence: str, pairs: set[tuple[int, int]], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(8, 3))
    ax = fig.add_subplot(1, 1, 1)
    ax.hlines(0, 0, len(sequence) - 1, linewidth=1)

    for i, nt in enumerate(sequence):
        ax.text(i, -0.1, nt, ha="center", va="top", fontsize=12)

    for i, j in pairs:
        xs = np.linspace(i, j, 200)
        heights = (j - i) / 2 * np.sin(np.pi * (xs - i) / (j - i))
        ax.plot(xs, heights)

    ax.set_title("Nussinov-Predicted Secondary Structure")
    ax.set_ylim(-0.5, max(1, len(sequence) * 0.3))
    ax.set_xlim(-0.5, len(sequence) - 0.5)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
