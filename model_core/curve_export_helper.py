import os
from typing import Tuple

import numpy as np
import pandas as pd


def _is_true(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def apply_runtime_overrides(default_epochs: int) -> Tuple[int, bool]:
    eval_only = _is_true(os.getenv("SEQPATHO_EVAL_ONLY", "0"))
    epochs = default_epochs

    epochs_raw = os.getenv("SEQPATHO_EPOCHS", "").strip()
    if epochs_raw:
        try:
            epochs = int(epochs_raw)
        except ValueError:
            print(f"[WARN] Invalid SEQPATHO_EPOCHS={epochs_raw}, keep default={default_epochs}")

    if eval_only:
        epochs = 0

    return epochs, eval_only


def apply_seed_override(default_seed: int) -> int:
    seed_raw = os.getenv("SEQPATHO_SEED", "").strip()
    if not seed_raw:
        return default_seed

    try:
        return int(seed_raw)
    except ValueError:
        print(f"[WARN] Invalid SEQPATHO_SEED={seed_raw}, keep default={default_seed}")
        return default_seed


def resolve_checkpoint_path(default_path: str) -> str:
    override = os.getenv("SEQPATHO_CKPT_PATH", "").strip().strip('"').strip("'")
    return override if override else default_path


def save_curve_points(output_dir: str, fpr, tpr, recall, precision) -> None:
    os.makedirs(output_dir, exist_ok=True)

    roc_df = pd.DataFrame({
        "fpr": np.asarray(fpr, dtype=float),
        "tpr": np.asarray(tpr, dtype=float),
    })
    pr_df = pd.DataFrame({
        "recall": np.asarray(recall, dtype=float),
        "precision": np.asarray(precision, dtype=float),
    })

    roc_df.to_csv(os.path.join(output_dir, "roc_curve_points.csv"), index=False)
    pr_df.to_csv(os.path.join(output_dir, "pr_curve_points.csv"), index=False)
