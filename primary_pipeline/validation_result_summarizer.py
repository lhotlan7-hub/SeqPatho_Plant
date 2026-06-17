#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


SCORE_CANDIDATES = [
    "score",
    "prob_pos",
    "prob_positive",
    "phi_positive_prob",
    "p_positive",
    "probability",
    "prob",
]


def compute_metrics(y_true, y_score, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= threshold).astype(int)

    out = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }

    if len(set(y_true)) == 2:
        out["AUROC"] = roc_auc_score(y_true, y_score)
        out["AUPRC"] = average_precision_score(y_true, y_score)
    else:
        out["AUROC"] = np.nan
        out["AUPRC"] = np.nan

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out.update(
        {
            "TP": int(tp),
            "FP": int(fp),
            "TN": int(tn),
            "FN": int(fn),
            "Support": int(len(y_true)),
            "Positive": int(y_true.sum()),
            "Negative": int(len(y_true) - y_true.sum()),
        }
    )
    return out


def load_prediction_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    if "label" not in df.columns and "y_true" in df.columns:
        df = df.rename(columns={"y_true": "label"})
    if "score" not in df.columns:
        for col in SCORE_CANDIDATES:
            if col in df.columns:
                df = df.rename(columns={col: "score"})
                break
    if "label" not in df.columns or "score" not in df.columns:
        raise ValueError(f"{path} must contain label/y_true and a positive-class score column.")
    df["label"] = df["label"].astype(int)
    df["score"] = df["score"].astype(float)
    return df


def summarize_single(path: Path, out_csv: Path, protocol_name: str) -> pd.DataFrame:
    df = load_prediction_csv(path)
    metrics = compute_metrics(df["label"], df["score"])
    metrics["Protocol"] = protocol_name
    out = pd.DataFrame([metrics])
    out.to_csv(out_csv, index=False)
    print(f"[write] {out_csv}")
    return out


def discover_loso_prediction_files(pred_dir: Path) -> List[Path]:
    patterns = [
        "fold_*/test_gene_level_predictions.csv",
        "fold_*/test_predictions.csv",
        "fold_*/predictions_test.csv",
    ]
    files: List[Path] = []
    for pattern in patterns:
        files.extend(sorted(pred_dir.glob(pattern)))
    return sorted(set(files))


def summarize_loso(pred_dir: Path, out_csv: Path) -> pd.DataFrame:
    rows = []
    files = discover_loso_prediction_files(pred_dir)
    if not files:
        raise RuntimeError(f"No LOSO fold prediction files found under {pred_dir}")

    for path in files:
        df = load_prediction_csv(path)
        fold_name = path.parent.name.replace("fold_", "")
        fold_id = int(fold_name) if fold_name.isdigit() else fold_name

        if "heldout_species_key" in df.columns and df["heldout_species_key"].notna().any():
            heldout = str(df["heldout_species_key"].dropna().iloc[0])
        elif "source_species_key" in df.columns and df["source_species_key"].notna().any():
            heldout = str(df["source_species_key"].dropna().iloc[0])
        else:
            heldout = str(fold_id)

        metrics = compute_metrics(df["label"], df["score"])
        metrics["fold_id"] = fold_id
        metrics["heldout_species_key"] = heldout
        rows.append(metrics)

    result = pd.DataFrame(rows)
    result.to_csv(out_csv, index=False)
    print(f"[write] {out_csv}")

    mean_std = {}
    for col in ["Accuracy", "Precision", "Recall", "F1", "MCC", "AUROC", "AUPRC"]:
        mean_std[col] = f"{result[col].mean():.4f} +/- {result[col].std(ddof=1):.4f}"
    print(json.dumps({"LOSO mean +/- std": mean_std}, indent=2, ensure_ascii=False))
    return result


def plot_loso_heatmap(loso_csv: Path, out_png: Path) -> None:
    df = pd.read_csv(loso_csv)
    metrics = ["Accuracy", "Precision", "Recall", "F1", "MCC", "AUROC", "AUPRC"]
    mat = df.set_index("heldout_species_key")[metrics].sort_values("AUPRC", ascending=False)

    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(mat))))
    im = ax.imshow(mat.values, aspect="auto")
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(metrics, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(mat.index)))
    ax.set_yticklabels(mat.index)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            value = mat.values[i, j]
            label = "NA" if pd.isna(value) else f"{value:.2f}"
            ax.text(j, i, label, ha="center", va="center", fontsize=7)

    ax.set_title("Source-pathogen LOSO metrics")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"[write] {out_png}")


def plot_support_vs_auprc(loso_csv: Path, out_png: Path) -> None:
    df = pd.read_csv(loso_csv)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(df["Support"], df["AUPRC"])

    for _, row in df.iterrows():
        ax.annotate(
            str(row["heldout_species_key"])[:25],
            (row["Support"], row["AUPRC"]),
            fontsize=7,
            alpha=0.8,
        )

    ax.set_xlabel("Held-out test support")
    ax.set_ylabel("AUPRC")
    ax.set_title("Source-pathogen LOSO: support vs AUPRC")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"[write] {out_png}")


def plot_hard_negative_pr_roc(pred_csv: Path, out_png: Path) -> None:
    df = load_prediction_csv(pred_csv)
    y_true = df["label"].astype(int).to_numpy()
    y_score = df["score"].astype(float).to_numpy()
    if len(set(y_true)) != 2:
        print("[skip] hard-negative PR/ROC plot requires both labels.")
        return

    fpr, tpr, _ = roc_curve(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    auroc = roc_auc_score(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].plot(fpr, tpr, label=f"AUROC={auroc:.3f}")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC")
    axes[0].legend(loc="lower right")

    axes[1].plot(recall, precision, label=f"AUPRC={auprc:.3f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-recall")
    axes[1].legend(loc="lower left")

    fig.suptitle("Hard-negative validation")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"[write] {out_png}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize validation prediction files.")
    parser.add_argument("--ood_pred", type=Path, default=None)
    parser.add_argument("--hard_pred", type=Path, default=None)
    parser.add_argument("--loso_pred_dir", type=Path, default=None)
    parser.add_argument("--outdir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    if args.ood_pred:
        summarize_single(
            args.ood_pred,
            args.outdir / "source_species_ood_metrics.csv",
            "Source-pathogen species-disjoint OOD split",
        )

    if args.hard_pred:
        summarize_single(
            args.hard_pred,
            args.outdir / "hard_negative_metrics.csv",
            "PHI-positive vs unaffected-pathogenicity hard negatives",
        )
        plot_hard_negative_pr_roc(args.hard_pred, args.outdir / "hard_negative_pr_roc.png")

    if args.loso_pred_dir:
        loso_csv = args.outdir / "source_species_loso_metrics.csv"
        summarize_loso(args.loso_pred_dir, loso_csv)
        plot_loso_heatmap(loso_csv, args.outdir / "source_species_loso_heatmap.png")
        plot_support_vs_auprc(loso_csv, args.outdir / "source_species_support_vs_auprc.png")


if __name__ == "__main__":
    main()
