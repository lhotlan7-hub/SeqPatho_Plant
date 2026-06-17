#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SPLIT_ORDER = ["train", "val", "test"]
POS_COLOR = "#3B82F6"
NEG_COLOR = "#F59E0B"
ACCENT_COLOR = "#10B981"
MUTED_COLOR = "#64748B"


def ensure_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def split_ordered(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["split"] = pd.Categorical(out["split"], categories=SPLIT_ORDER, ordered=True)
    return out.sort_values("split")


def short_label(value: object, max_len: int = 42) -> str:
    text = str(value)
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def plot_split_label_counts(summary: pd.DataFrame, title: str, out_png: Path) -> None:
    df = split_ordered(ensure_numeric(summary, ["pos", "neg", "n"]))
    x = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(x, df["neg"], label="Negative", color=NEG_COLOR)
    ax.bar(x, df["pos"], bottom=df["neg"], label="Positive", color=POS_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(df["split"].astype(str))
    ax.set_ylabel("Records")
    ax.set_title(title)
    ax.legend(frameon=False)

    for i, row in enumerate(df.itertuples(index=False)):
        ax.text(i, row.n, f"{int(row.n):,}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def plot_split_species_counts(summary: pd.DataFrame, title: str, out_png: Path) -> None:
    df = split_ordered(ensure_numeric(summary, ["species_n", "pos_rate"]))
    x = np.arange(len(df))

    fig, ax1 = plt.subplots(figsize=(7, 4.2))
    ax1.bar(x, df["species_n"], color=ACCENT_COLOR, label="Source species")
    ax1.set_ylabel("Source species count")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["split"].astype(str))

    ax2 = ax1.twinx()
    ax2.plot(x, df["pos_rate"], marker="o", color=MUTED_COLOR, label="Positive rate")
    ax2.set_ylabel("Positive rate")
    ax2.set_ylim(0, max(1.0, float(df["pos_rate"].max()) * 1.15))

    ax1.set_title(title)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def plot_protocol_split_comparison(main_summary: pd.DataFrame, ood_summary: pd.DataFrame, out_png: Path) -> None:
    main = split_ordered(ensure_numeric(main_summary, ["n"]))
    ood = split_ordered(ensure_numeric(ood_summary, ["n"]))
    x = np.arange(len(SPLIT_ORDER))
    width = 0.36

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.bar(x - width / 2, main["n"], width=width, label="Locked main", color=POS_COLOR)
    ax.bar(x + width / 2, ood["n"], width=width, label="Source OOD", color=ACCENT_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(SPLIT_ORDER)
    ax.set_ylabel("Records")
    ax.set_title("Split size comparison")
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def plot_loso_support_hist(support: pd.DataFrame, out_png: Path) -> None:
    df = ensure_numeric(support, ["n", "pos", "neg"])
    values = df["n"].dropna().clip(lower=1)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.hist(np.log10(values), bins=45, color=POS_COLOR, alpha=0.85)
    ax.set_xlabel("log10(held-out support)")
    ax.set_ylabel("LOSO folds")
    ax.set_title("Source-pathogen LOSO support distribution")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def plot_loso_top_support(support: pd.DataFrame, out_png: Path, top_n: int = 30) -> None:
    df = ensure_numeric(support, ["n", "pos", "neg"])
    df = df.sort_values("n", ascending=False).head(top_n).sort_values("n", ascending=True)
    labels = [short_label(x, 48) for x in df["source_species"]]

    fig, ax = plt.subplots(figsize=(9, max(5, 0.28 * len(df))))
    y = np.arange(len(df))
    ax.barh(y, df["neg"], color=NEG_COLOR, label="Negative")
    ax.barh(y, df["pos"], left=df["neg"], color=POS_COLOR, label="Positive")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Held-out records")
    ax.set_title(f"Top {top_n} LOSO held-out source species by support")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def plot_loso_pos_neg_scatter(support: pd.DataFrame, out_png: Path) -> None:
    df = ensure_numeric(support, ["n", "pos", "neg"])
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    sc = ax.scatter(df["neg"], df["pos"], c=df["n"], cmap="viridis", s=18, alpha=0.78)
    max_v = float(max(df["pos"].max(), df["neg"].max()))
    ax.plot([0, max_v], [0, max_v], color=MUTED_COLOR, linestyle="--", linewidth=1)
    ax.set_xlabel("Held-out negatives")
    ax.set_ylabel("Held-out positives")
    ax.set_title("LOSO fold label balance")
    fig.colorbar(sc, ax=ax, label="Total support")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def plot_loso_support_heatmap(support: pd.DataFrame, out_png: Path, top_n: int = 50) -> None:
    df = ensure_numeric(support, ["n", "pos", "neg"])
    df["pos_rate"] = df["pos"] / df["n"]
    df = df.sort_values("n", ascending=False).head(top_n).copy()
    raw = df[["n", "pos", "neg", "pos_rate"]].to_numpy(dtype=float)
    norm = raw.copy()
    for j in range(norm.shape[1]):
        col = norm[:, j]
        span = np.nanmax(col) - np.nanmin(col)
        norm[:, j] = 0.5 if span == 0 else (col - np.nanmin(col)) / span

    fig, ax = plt.subplots(figsize=(8.4, max(6, 0.22 * len(df))))
    im = ax.imshow(norm, aspect="auto", cmap="magma")
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(["n", "pos", "neg", "pos_rate"])
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels([short_label(x, 44) for x in df["source_species"]], fontsize=7)
    ax.set_title(f"Top {top_n} LOSO source species support heatmap")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="Column-normalized value")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


def write_visualization_index(rows: List[dict], out_csv: Path) -> None:
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize generated validation protocol manifests.")
    parser.add_argument("--outdir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    outdir = args.outdir
    main_summary_path = outdir / "locked_main_split_summary.csv"
    ood_summary_path = outdir / "source_species_ood_summary.csv"
    loso_support_path = outdir / "source_species_loso_support.csv"

    required = [main_summary_path, ood_summary_path, loso_support_path]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")

    main_summary = pd.read_csv(main_summary_path)
    ood_summary = pd.read_csv(ood_summary_path)
    loso_support = pd.read_csv(loso_support_path)

    rows: List[dict] = []

    outputs = [
        (
            outdir / "locked_main_split_label_counts.png",
            lambda p: plot_split_label_counts(main_summary, "Locked main split label counts", p),
            "Locked main train/val/test positive-negative counts.",
        ),
        (
            outdir / "locked_main_split_species_counts.png",
            lambda p: plot_split_species_counts(main_summary, "Locked main split source species counts", p),
            "Locked main split source species counts and positive rate.",
        ),
        (
            outdir / "source_species_ood_split_label_counts.png",
            lambda p: plot_split_label_counts(ood_summary, "Source-pathogen OOD split label counts", p),
            "Source-pathogen OOD train/val/test positive-negative counts.",
        ),
        (
            outdir / "source_species_ood_split_species_counts.png",
            lambda p: plot_split_species_counts(ood_summary, "Source-pathogen OOD source species counts", p),
            "Source-pathogen OOD split source species counts and positive rate.",
        ),
        (
            outdir / "protocol_split_size_comparison.png",
            lambda p: plot_protocol_split_comparison(main_summary, ood_summary, p),
            "Record-count comparison between locked main and source OOD splits.",
        ),
        (
            outdir / "source_species_loso_support_distribution.png",
            lambda p: plot_loso_support_hist(loso_support, p),
            "Distribution of held-out source species support across LOSO folds.",
        ),
        (
            outdir / "source_species_loso_top30_support.png",
            lambda p: plot_loso_top_support(loso_support, p, top_n=30),
            "Top 30 held-out source species by total support.",
        ),
        (
            outdir / "source_species_loso_label_balance.png",
            lambda p: plot_loso_pos_neg_scatter(loso_support, p),
            "Positive-negative balance for each LOSO held-out source species.",
        ),
        (
            outdir / "source_species_loso_support_heatmap.png",
            lambda p: plot_loso_support_heatmap(loso_support, p, top_n=50),
            "Support heatmap for top 50 LOSO source species.",
        ),
    ]

    for path, func, description in outputs:
        func(path)
        rows.append({"file": str(path), "description": description})
        print(f"[write] {path}")

    index_path = outdir / "visualization_index.csv"
    write_visualization_index(rows, index_path)
    print(f"[write] {index_path}")


if __name__ == "__main__":
    main()
