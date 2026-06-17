# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent


def latest(prefix: str) -> Path:
    candidates = sorted([p for p in ROOT.glob(f"results_{prefix}_*") if p.is_dir()])
    if not candidates:
        raise FileNotFoundError(f"no results folder matching results_{prefix}_*")
    return candidates[-1]


def standard_rows() -> pd.DataFrame:
    rows = []

    tab_dir = latest("tabular_kmer_baselines")
    tab = pd.read_csv(tab_dir / "aggregate_baseline_summary.csv")
    tab_std = tab[tab["benchmark"] == "standard_species_disjoint"].copy()
    for _, row in tab_std.iterrows():
        rows.append(
            {
                "family": "length_gc_kmer",
                "baseline": row["baseline"],
                "accuracy": row["accuracy_mean"],
                "mcc": row["mcc_mean"],
                "auroc": row["auroc_mean"],
                "auprc": row["auprc_mean"],
                "ap": row["ap_mean"],
                "n_runs": row["n_runs"],
                "source_dir": str(tab_dir),
            }
        )

    nn_dir = latest("nearest_neighbor_baseline")
    nn = pd.read_csv(nn_dir / "aggregate_nearest_neighbor_summary.csv")
    nn_std = nn[nn["benchmark"] == "standard_species_disjoint"].copy()
    for _, row in nn_std.iterrows():
        rows.append(
            {
                "family": "homology_nearest_neighbor",
                "baseline": row["baseline"],
                "accuracy": row["accuracy_mean"],
                "mcc": row["mcc_mean"],
                "auroc": row["auroc_mean"],
                "auprc": row["auprc_mean"],
                "ap": row["ap_mean"],
                "n_runs": row["n_runs"],
                "source_dir": str(nn_dir),
            }
        )

    dna_dir = latest("dnaberts_baselines")
    dna = pd.read_csv(dna_dir / "dnaberts_baseline_results.csv")
    dna_std = dna[dna["benchmark"] == "standard_species_disjoint"].copy()
    for _, row in dna_std.iterrows():
        rows.append(
            {
                "family": "dnaberts_ablation",
                "baseline": row["baseline"],
                "accuracy": row["accuracy"],
                "mcc": row["mcc"],
                "auroc": row["auroc"],
                "auprc": row["auprc"],
                "ap": row["ap"],
                "n_runs": 1,
                "source_dir": str(dna_dir),
            }
        )
    return pd.DataFrame(rows).sort_values("auprc", ascending=False)


def loso_rows() -> pd.DataFrame:
    rows = []
    tab_dir = latest("tabular_kmer_baselines")
    tab = pd.read_csv(tab_dir / "aggregate_baseline_summary.csv")
    tab_loso = tab[tab["benchmark"] == "loso_species"].copy()
    for _, row in tab_loso.iterrows():
        rows.append(
            {
                "family": "length_gc_kmer",
                "baseline": row["baseline"],
                "accuracy_mean": row["accuracy_mean"],
                "accuracy_std": row["accuracy_std"],
                "mcc_mean": row["mcc_mean"],
                "mcc_std": row["mcc_std"],
                "auroc_mean": row["auroc_mean"],
                "auroc_std": row["auroc_std"],
                "auprc_mean": row["auprc_mean"],
                "auprc_std": row["auprc_std"],
                "ap_mean": row["ap_mean"],
                "ap_std": row["ap_std"],
                "n_runs": row["n_runs"],
                "source_dir": str(tab_dir),
            }
        )

    nn_dir = latest("nearest_neighbor_baseline")
    nn = pd.read_csv(nn_dir / "aggregate_nearest_neighbor_summary.csv")
    nn_loso = nn[nn["benchmark"] == "loso_species"].copy()
    for _, row in nn_loso.iterrows():
        rows.append(
            {
                "family": "homology_nearest_neighbor",
                "baseline": row["baseline"],
                "accuracy_mean": row["accuracy_mean"],
                "accuracy_std": row["accuracy_std"],
                "mcc_mean": row["mcc_mean"],
                "mcc_std": row["mcc_std"],
                "auroc_mean": row["auroc_mean"],
                "auroc_std": row["auroc_std"],
                "auprc_mean": row["auprc_mean"],
                "auprc_std": row["auprc_std"],
                "ap_mean": row["ap_mean"],
                "ap_std": row["ap_std"],
                "n_runs": row["n_runs"],
                "source_dir": str(nn_dir),
            }
        )
    return pd.DataFrame(rows).sort_values("auprc_mean", ascending=False)


def save_with_md(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")
    try:
        with open(path.with_suffix(".md"), "w", encoding="utf-8") as f:
            f.write(df.to_markdown(index=False))
            f.write("\n")
    except Exception:
        pass


def main() -> None:
    out_dir = ROOT / "combined_latest_summary"
    out_dir.mkdir(exist_ok=True)
    std = standard_rows()
    loso = loso_rows()
    save_with_md(std, out_dir / "combined_standard_species_disjoint.csv")
    save_with_md(loso, out_dir / "combined_loso_species.csv")
    print(out_dir)


if __name__ == "__main__":
    main()

