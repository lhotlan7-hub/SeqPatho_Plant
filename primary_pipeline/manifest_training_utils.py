#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


def load_manifest_for_run(
    manifest_path: str | Path,
    protocol: str,
    fold_id: Optional[str | int] = None,
    require_train: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return train/val/test dataframes from a row-level manifest."""
    df = pd.read_csv(manifest_path, low_memory=False)

    if "protocol" in df.columns:
        df = df[df["protocol"].astype(str) == str(protocol)].copy()

    if fold_id is not None:
        if "fold_id" not in df.columns:
            raise ValueError("fold_id was provided but manifest has no fold_id column")
        df = df[df["fold_id"].astype(str) == str(fold_id)].copy()

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    if require_train:
        if len(train_df) == 0:
            raise ValueError("training split is empty")
        if len(val_df) == 0:
            raise ValueError("validation split is empty")
    if len(test_df) == 0:
        raise ValueError("test split is empty")

    return train_df, val_df, test_df


def random_nt_window(seq: str, max_nt: int, rng: Optional[np.random.Generator] = None) -> str:
    """Random nucleotide crop for training."""
    seq = str(seq)
    if len(seq) <= max_nt:
        return seq
    if rng is None:
        rng = np.random.default_rng()
    start = int(rng.integers(0, len(seq) - max_nt + 1))
    return seq[start : start + max_nt]


def sliding_nt_windows(seq: str, window_nt: int, stride_nt: int) -> List[str]:
    """Deterministic nucleotide windows for gene-level inference."""
    seq = str(seq)
    if len(seq) <= window_nt:
        return [seq]

    windows = [seq[start : start + window_nt] for start in range(0, len(seq) - window_nt + 1, stride_nt)]
    if windows[-1] != seq[-window_nt:]:
        windows.append(seq[-window_nt:])
    return windows


def aggregate_window_scores(scores: Iterable[float], aggregation: str = "mean") -> float:
    values = np.asarray(list(scores), dtype=float)
    if len(values) == 0:
        return float("nan")
    if aggregation == "mean":
        return float(values.mean())
    if aggregation == "max":
        return float(values.max())
    raise ValueError(f"Unknown aggregation: {aggregation}")


def prediction_output_frame(
    df: pd.DataFrame,
    scores: Iterable[float],
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Build the standardized gene-level prediction table."""
    keep_cols = [
        "record_id",
        "label",
        "source_tax_id",
        "source_species",
        "source_species_key",
        "fold_id",
        "heldout_species_key",
    ]
    out = df[[c for c in keep_cols if c in df.columns]].copy()
    out["score"] = np.asarray(list(scores), dtype=float)
    out["pred"] = (out["score"] >= threshold).astype(int)
    return out
