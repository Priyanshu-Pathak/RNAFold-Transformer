from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .config import ExperimentConfig
from .data import create_dataloaders, prepare_dataframe
from .metrics import pairwise_distances, summarize_predictions
from .model import build_model
from .plotting import plot_rmse_distribution, plot_true_vs_predicted
from .utils import ensure_dir, get_device, save_json, set_seed


def evaluate(config: ExperimentConfig, checkpoint_path: str | Path) -> dict[str, float]:
    set_seed(config.seed)
    output_dir = ensure_dir(config.output_dir)
    figure_dir = ensure_dir(output_dir / "figures")
    device = get_device(config.device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    ckpt_config = checkpoint.get("config", {})
    model_config = {**ckpt_config, **config.__dict__}

    df, _ = prepare_dataframe(
        data_dir=config.data_dir,
        labels_file=config.labels_file,
        sequences_file=config.sequences_file,
        msa_dir=config.msa_dir,
        window=int(model_config.get("window", config.window)),
        sample_size=config.sample_size,
        seed=config.seed,
    )
    _, test_loader = create_dataloaders(
        df,
        batch_size=config.batch_size,
        test_size=config.test_size,
        seed=config.seed,
        num_workers=config.num_workers,
    )

    model = build_model(
        window=int(model_config.get("window", config.window)),
        d_model=int(model_config.get("d_model", config.d_model)),
        nhead=int(model_config.get("nhead", config.nhead)),
        dim_feedforward=int(model_config.get("dim_feedforward", config.dim_feedforward)),
        num_layers=int(model_config.get("num_layers", config.num_layers)),
        dropout=float(model_config.get("dropout", config.dropout)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    loss_fn = nn.MSELoss()

    with torch.no_grad():
        for features, coords in test_loader:
            features = features.to(device)
            coords = coords.to(device)

            target = pairwise_distances(coords)
            pred = model(features)

            if torch.isnan(pred).any() or torch.isnan(target).any():
                continue

            _ = loss_fn(pred, target)
            preds.append(pred.cpu().numpy())
            targets.append(target.cpu().numpy())

    pred_arr = np.concatenate(preds, axis=0)
    target_arr = np.concatenate(targets, axis=0)

    metrics = summarize_predictions(pred_arr, target_arr)
    save_json(metrics, output_dir / "evaluation_metrics.json")

    window_rmse = np.sqrt(np.mean((pred_arr - target_arr) ** 2, axis=(1, 2)))
    plot_rmse_distribution(window_rmse, figure_dir / "eval_rmse_distribution.png")
    plot_true_vs_predicted(
        target_arr.reshape(-1),
        pred_arr.reshape(-1),
        figure_dir / "eval_true_vs_predicted.png",
        seed=config.seed,
    )

    print(metrics)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained DistTransformer checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--data-dir", default=None, help="Dataset directory.")
    parser.add_argument("--output-dir", default="outputs/evaluation", help="Output directory.")
    parser.add_argument("--sample-size", type=int, default=None, help="Maximum number of residue windows.")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size.")
    parser.add_argument("--device", default=None, help="auto, cpu, cuda, or cuda:0.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig.from_yaml(args.config).update(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        batch_size=args.batch_size,
        device=args.device,
    )
    evaluate(config, args.checkpoint)


if __name__ == "__main__":
    main()
