from __future__ import annotations

from pathlib import Path

import numpy as np

from .msa import compute_msa_profile_window

NUC_ONE_HOT = {
    "A": [1.0, 0.0, 0.0, 0.0],
    "C": [0.0, 1.0, 0.0, 0.0],
    "G": [0.0, 0.0, 1.0, 0.0],
    "U": [0.0, 0.0, 0.0, 1.0],
    "N": [0.0, 0.0, 0.0, 0.0],
}


def get_sequence_window(sequence: str, resid: int, window: int = 10) -> str:
    sequence = str(sequence).upper()
    pos = resid - 1
    start = max(0, pos - window)
    end = min(len(sequence), pos + window + 1)
    seq_window = sequence[start:end]
    target_len = 2 * window + 1
    missing = target_len - len(seq_window)

    if missing > 0:
        if start == 0:
            seq_window = "N" * missing + seq_window
        else:
            seq_window = seq_window + "N" * missing

    return seq_window


def encode_window_features(
    sequence: str,
    base_id: str,
    resid: int,
    msa_dir: str | Path,
    window: int = 10,
) -> np.ndarray | None:
    msa_path = Path(msa_dir) / f"{base_id}.MSA.fasta"
    if not msa_path.exists():
        return None

    seq_window = get_sequence_window(sequence, resid, window)
    profile_window = compute_msa_profile_window(msa_path, resid, window)
    if profile_window is None or len(profile_window) != 2 * window + 1:
        return None

    features = []
    for i, base in enumerate(seq_window):
        one_hot = NUC_ONE_HOT.get(base, NUC_ONE_HOT["N"])
        profile = profile_window[i]
        features.append(one_hot + profile)

    arr = np.asarray(features, dtype=np.float32)
    if np.isnan(arr).any() or np.isinf(arr).any():
        return None
    return arr


def extract_coordinate_window(
    base_id: str,
    resid: int,
    coord_map: dict[tuple[str, int], tuple[float, float, float]],
    window: int = 10,
) -> np.ndarray | None:
    coords = []
    for offset in range(-window, window + 1):
        key = (base_id, resid + offset)
        if key not in coord_map:
            return None
        coords.append(coord_map[key])

    arr = np.asarray(coords, dtype=np.float32)
    if np.isnan(arr).any() or np.isinf(arr).any():
        return None
    return arr
