from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm.auto import tqdm

from .features import encode_window_features, extract_coordinate_window


def _find_file(data_dir: str | Path, candidates: list[str], explicit: str | None = None) -> Path:
    data_dir = Path(data_dir)
    if explicit:
        path = Path(explicit)
        if not path.is_absolute():
            path = data_dir / path
        if path.exists():
            return path
        raise FileNotFoundError(f"Could not find file: {path}")

    for name in candidates:
        path = data_dir / name
        if path.exists():
            return path

    raise FileNotFoundError(
        f"None of these files were found under {data_dir}: {', '.join(candidates)}"
    )


def load_sequences(data_dir: str | Path, sequences_file: str | None = None) -> dict[str, str]:
    path = _find_file(
        data_dir,
        ["train_sequences.xlsx", "train_sequences.v2.xlsx", "train_sequences.csv", "train_sequences.v2.csv"],
        sequences_file,
    )

    if path.suffix.lower() in {".xlsx", ".xls"}:
        seqs_df = pd.read_excel(path, engine="openpyxl")
    else:
        seqs_df = pd.read_csv(path)

    required = {"target_id", "sequence"}
    missing = required - set(seqs_df.columns)
    if missing:
        raise ValueError(f"Sequence file is missing required columns: {missing}")

    seqs_df["target_id"] = seqs_df["target_id"].astype(str).str.strip()
    seqs_df["sequence"] = seqs_df["sequence"].astype(str).str.strip().str.upper()
    return dict(zip(seqs_df["target_id"], seqs_df["sequence"]))


def infer_base_id(label_id: str, seq_map: dict[str, str]) -> str:
    text = str(label_id).strip()
    pieces = text.split("_")

    candidates = []
    if len(pieces) > 1:
        candidates.append("_".join(pieces[:-1]))
    if len(pieces) >= 2:
        candidates.append("_".join(pieces[:2]))
    candidates.append(pieces[0])

    for candidate in candidates:
        if candidate in seq_map:
            return candidate

    return candidates[0]


def prepare_dataframe(
    data_dir: str | Path,
    labels_file: str | None = None,
    sequences_file: str | None = None,
    msa_dir: str | None = None,
    window: int = 10,
    sample_size: int | None = 8000,
    seed: int = 42,
) -> tuple[pd.DataFrame, StandardScaler]:
    data_dir = Path(data_dir)
    labels_path = _find_file(data_dir, ["train_labels.v2.csv", "train_labels.csv"], labels_file)
    msa_path = Path(msa_dir) if msa_dir else data_dir / "MSA"

    if not msa_path.exists():
        raise FileNotFoundError(f"MSA directory was not found: {msa_path}")

    seq_map = load_sequences(data_dir, sequences_file)
    labels_df = pd.read_csv(labels_path)

    required_cols = {"ID", "resname", "resid", "x_1", "y_1", "z_1"}
    missing = required_cols - set(labels_df.columns)
    if missing:
        raise ValueError(f"Label file is missing required columns: {missing}")

    labels_df = labels_df[labels_df["resname"].isin(["A", "C", "G", "U"])].copy()
    labels_df = labels_df.dropna(subset=["x_1", "y_1", "z_1", "ID", "resid"]).reset_index(drop=True)
    labels_df["base_id"] = labels_df["ID"].apply(lambda x: infer_base_id(x, seq_map))
    labels_df["sequence"] = labels_df["base_id"].map(seq_map)
    labels_df = labels_df.dropna(subset=["sequence"]).reset_index(drop=True)

    coords_raw = labels_df[["x_1", "y_1", "z_1"]].values.astype(np.float32)
    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords_raw)
    labels_df[["x", "y", "z"]] = coords_scaled

    coord_map = {
        (row.base_id, int(row.resid)): (float(row.x), float(row.y), float(row.z))
        for _, row in labels_df.iterrows()
    }

    rows: list[dict[str, object]] = []
    iterator = tqdm(labels_df.itertuples(index=False), total=len(labels_df), desc="Building windows")
    for row in iterator:
        base_id = str(row.base_id)
        resid = int(row.resid)

        coords = extract_coordinate_window(base_id, resid, coord_map, window)
        if coords is None:
            continue

        features = encode_window_features(str(row.sequence), base_id, resid, msa_path, window)
        if features is None:
            continue

        rows.append(
            {
                "ID": row.ID,
                "base_id": base_id,
                "resid": resid,
                "resname": row.resname,
                "sequence": row.sequence,
                "features": features,
                "true_coords": coords,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("No valid windows were created. Check data paths and MSA filenames.")

    if sample_size is not None and len(out) > sample_size:
        out = out.sample(sample_size, random_state=seed).reset_index(drop=True)

    return out.reset_index(drop=True), scaler


class DistanceDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.features = np.asarray(df["features"].tolist(), dtype=np.float32)
        self.true_coords = np.asarray(df["true_coords"].tolist(), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.features[index], dtype=torch.float32),
            torch.tensor(self.true_coords[index], dtype=torch.float32),
        )


def create_dataloaders(
    df: pd.DataFrame,
    batch_size: int = 64,
    test_size: float = 0.2,
    seed: int = 42,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    dataset = DistanceDataset(df)
    test_n = max(1, int(round(len(dataset) * test_size)))
    train_n = len(dataset) - test_n
    if train_n <= 0:
        raise ValueError("Dataset is too small for the requested test split.")

    generator = torch.Generator().manual_seed(seed)
    train_set, test_set = random_split(dataset, [train_n, test_n], generator=generator)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader
