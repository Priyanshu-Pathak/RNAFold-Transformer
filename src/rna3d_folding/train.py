from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .config import ExperimentConfig
from .data import create_dataloaders, prepare_dataframe
from .metrics import pairwise_distances, summarize_predictions
from .model import build_model
from .plotting import plot_rmse_distribution, plot_training_history, plot_true_vs_predicted
from .utils import count_parameters, ensure_dir, get_device, save_json, set_seed


def run_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    is_train = optimizer is not None
    model.train(is_train)

    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    losses: list[float] = []

    with torch.set_grad_enabled(is_train):
        for features, coords in loader:
            features = features.to(device)
            coords = coords.to(device)

            target_dist = pairwise_distances(coords)
            pred_dist = model(features)

            if torch.isnan(pred_dist).any() or torch.isnan(target_dist).any():
                continue

            loss = loss_fn(pred_dist, target_dist)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            losses.append(float(loss.detach().cpu()))
            preds.append(pred_dist.detach().cpu().numpy())
            targets.append(target_dist.detach().cpu().numpy())

    if not preds:
        raise RuntimeError("No valid batches were processed.")

    pred_arr = np.concatenate(preds, axis=0)
    target_arr = np.concatenate(targets, axis=0)
    metrics = summarize_predictions(pred_arr, target_arr)
    return metrics["rmse"], metrics["mae"], pred_arr, target_arr


def train(config: ExperimentConfig) -> dict[str, float]:
    start = time.time()
    set_seed(config.seed)

    output_dir = ensure_dir(config.output_dir)
    figure_dir = ensure_dir(output_dir / "figures")
    checkpoint_dir = ensure_dir(output_dir / "checkpoints")

    device = get_device(config.device)
    df, _ = prepare_dataframe(
        data_dir=config.data_dir,
        labels_file=config.labels_file,
        sequences_file=config.sequences_file,
        msa_dir=config.msa_dir,
        window=config.window,
        sample_size=config.sample_size,
        seed=config.seed,
    )
    train_loader, test_loader = create_dataloaders(
        df,
        batch_size=config.batch_size,
        test_size=config.test_size,
        seed=config.seed,
        num_workers=config.num_workers,
    )

    model = build_model(
        window=config.window,
        d_model=config.d_model,
        nhead=config.nhead,
        dim_feedforward=config.dim_feedforward,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_fn = nn.MSELoss()

    history: list[dict[str, float]] = []
    best_rmse = float("inf")
    best_path = checkpoint_dir / "best_model.pt"

    print(f"Device: {device}")
    print(f"Valid windows: {len(df)}")
    print(f"Train batches: {len(train_loader)} | Test batches: {len(test_loader)}")
    print(f"Trainable parameters: {count_parameters(model):,}")

    for epoch in range(1, config.epochs + 1):
        train_rmse, train_mae, _, _ = run_epoch(model, train_loader, device, loss_fn, optimizer)
        test_rmse, test_mae, test_pred, test_target = run_epoch(model, test_loader, device, loss_fn)

        row = {
            "epoch": epoch,
            "train_rmse": train_rmse,
            "test_rmse": test_rmse,
            "train_mae": train_mae,
            "test_mae": test_mae,
        }
        history.append(row)

        if config.save_best and test_rmse < best_rmse:
            best_rmse = test_rmse
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(config),
                    "epoch": epoch,
                    "test_rmse": test_rmse,
                    "test_mae": test_mae,
                },
                best_path,
            )

        if epoch == 1 or epoch % config.log_every == 0 or epoch == config.epochs:
            print(
                f"Epoch {epoch:3d} | "
                f"Dist-RMSE = {train_rmse:.4f}/{test_rmse:.4f} | "
                f"Dist-MAE = {train_mae:.4f}/{test_mae:.4f}"
            )

    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / "training_history.csv", index=False)
    plot_training_history(history_df, figure_dir / "training_curves.png")

    final_metrics = summarize_predictions(test_pred, test_target)
    final_metrics.update(
        {
            "best_test_rmse": float(best_rmse),
            "epochs": config.epochs,
            "sample_size": int(len(df)),
            "runtime_seconds": float(time.time() - start),
        }
    )
    save_json(final_metrics, output_dir / "metrics.json")

    window_rmse = np.sqrt(np.mean((test_pred - test_target) ** 2, axis=(1, 2)))
    plot_rmse_distribution(window_rmse, figure_dir / "rmse_distribution.png")
    plot_true_vs_predicted(
        test_target.reshape(-1),
        test_pred.reshape(-1),
        figure_dir / "true_vs_predicted.png",
        seed=config.seed,
    )

    if not config.save_best:
        torch.save(
            {"model_state_dict": model.state_dict(), "config": asdict(config)},
            checkpoint_dir / "last_model.pt",
        )

    print(f"Saved outputs to {output_dir}")
    return final_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DistTransformer for RNA pairwise distance prediction.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--data-dir", default=None, help="Dataset directory.")
    parser.add_argument("--output-dir", default=None, help="Output directory.")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs.")
    parser.add_argument("--sample-size", type=int, default=None, help="Maximum number of residue windows.")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size.")
    parser.add_argument("--device", default=None, help="auto, cpu, cuda, or cuda:0.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig.from_yaml(args.config).update(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        sample_size=args.sample_size,
        batch_size=args.batch_size,
        device=args.device,
    )
    train(config)


if __name__ == "__main__":
    main()
