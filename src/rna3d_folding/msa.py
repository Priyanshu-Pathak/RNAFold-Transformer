from __future__ import annotations

from collections import Counter
from pathlib import Path

NUCLEOTIDES = ("A", "C", "G", "U")


def read_fasta(path: str | Path) -> list[str]:
    records: list[str] = []
    current: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current:
                    records.append("".join(current).upper())
                current = []
            else:
                current.append(line)
    if current:
        records.append("".join(current).upper())
    return records


def residue_to_alignment_column(target_sequence: str, resid: int) -> int | None:
    observed_residues = 0
    for column_index, base in enumerate(target_sequence):
        if base != "-":
            observed_residues += 1
            if observed_residues == resid:
                return column_index
    return None


def compute_msa_profile_window(
    msa_path: str | Path,
    resid: int,
    window: int = 10,
) -> list[list[float]] | None:
    msa_sequences = read_fasta(msa_path)
    if not msa_sequences:
        return None

    target_sequence = msa_sequences[0]
    column_index = residue_to_alignment_column(target_sequence, resid)
    if column_index is None:
        column_index = len(target_sequence) - 1

    profiles: list[list[float]] = []
    length = len(target_sequence)

    for offset in range(-window, window + 1):
        j = min(max(column_index + offset, 0), length - 1)
        counts = Counter(seq[j] for seq in msa_sequences if j < len(seq))
        denom = float(sum(counts[nuc] for nuc in NUCLEOTIDES))
        if denom == 0:
            profiles.append([0.0, 0.0, 0.0, 0.0])
        else:
            profiles.append([counts[nuc] / denom for nuc in NUCLEOTIDES])

    return profiles
