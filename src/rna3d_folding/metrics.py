from __future__ import annotations

import numpy as np
import torch


def pairwise_distances(coords: torch.Tensor) -> torch.Tensor:
    return torch.cdist(coords, coords, p=2)


def rmse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - target) ** 2)))


def mae(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - target)))


def pearson_corr(pred: np.ndarray, target: np.ndarray) -> float:
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    if pred_flat.std() == 0 or target_flat.std() == 0:
        return float("nan")
    return float(np.corrcoef(target_flat, pred_flat)[0, 1])


def summarize_predictions(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    window_rmse = np.sqrt(np.mean((pred - target) ** 2, axis=(1, 2)))
    window_mae = np.mean(np.abs(pred - target), axis=(1, 2))

    return {
        "rmse": rmse(pred, target),
        "mae": mae(pred, target),
        "window_rmse_mean": float(window_rmse.mean()),
        "window_rmse_std": float(window_rmse.std()),
        "window_mae_mean": float(window_mae.mean()),
        "window_mae_std": float(window_mae.std()),
        "pearson": pearson_corr(pred, target),
    }
