from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ExperimentConfig:
    data_dir: str = "data"
    output_dir: str = "outputs"
    labels_file: str | None = None
    sequences_file: str | None = None
    msa_dir: str | None = None
    window: int = 10
    sample_size: int | None = 8000
    test_size: float = 0.2
    seed: int = 42
    batch_size: int = 64
    num_workers: int = 0
    epochs: int = 500
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    d_model: int = 64
    nhead: int = 8
    dim_feedforward: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    log_every: int = 50
    save_best: bool = True
    device: str = "auto"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        with open(path, "r", encoding="utf-8") as f:
            data: dict[str, Any] = yaml.safe_load(f) or {}
        allowed = set(cls.__dataclass_fields__.keys())
        clean = {k: v for k, v in data.items() if k in allowed}
        return cls(**clean)

    def update(self, **kwargs: Any) -> "ExperimentConfig":
        data = asdict(self)
        for key, value in kwargs.items():
            if value is not None and key in data:
                data[key] = value
        return ExperimentConfig(**data)
