#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_NAMES = [
    "clean_len",
    "log1p_clean_len",
    "a_frac",
    "c_frac",
    "g_frac",
    "t_frac",
    "gc_frac",
    "n_frac",
    "starts_atg",
    "stops_taa",
    "stops_tag",
    "stops_tga",
]

BASELINES = {
    "length_gc_logreg": ["clean_len", "log1p_clean_len", "gc_frac", "n_frac"],
    "nucleotide_composition_logreg": FEATURE_NAMES,
}


@dataclass(frozen=True)
class BaselineResult:
    baseline: str
    benchmark: str
    threshold: float
    runtime_sec: float
    metrics: Dict[str, float]


def build_features(df: pd.DataFrame) -> np.ndarray:
    n = len(df)
    out = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float32)
    seqs = df["clean_seq"].astype(str).to_numpy()
    lengths = df["clean_len"].astype(np.float32).to_numpy()
    out[:, 0] = lengths
    out[:, 1] = np.log1p(lengths)

    for i, seq in enumerate(seqs):
        if i and i % 100000 == 0:
            print(f"[features] {i:,}/{n:,}")
        a = seq.count("A")
        c = seq.count("C")
        g = seq.count("G")
        t = seq.count("T")
        nn = seq.count("N")
        denom = max(a + c + g + t, 1)
        total = max(denom + nn, 1)
        out[i, 2] = a / denom
        out[i, 3] = c / denom
        out[i, 4] = g / denom
        out[i, 5] = t / denom
        out[i, 6] = (g + c) / denom
        out[i, 7] = nn / total
        out[i, 8] = 1.0 if seq.startswith("ATG") else 0.0
        out[i, 9] = 1.0 if seq.endswith("TAA") else 0.0
        out[i, 10] = 1.0 if seq.endswith("TAG") else 0.0
        out[i, 11] = 1.0 if seq.endswith("TGA") else 0.0
    return out


def feature_indices(feature_names: Iterable[str]) -> List[int]:
    idx = {name: i for i, name in enumerate(FEATURE_NAMES)}
    return [idx[name] for name in feature_names]


def fit_logreg(x: np.ndarray, y: np.ndarray) -> Pipeline:
    model = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(x, y)
    return model


def positive_scores(model: Pipeline, x: np.ndarray) -> np.ndarray:
    prob = model.predict_proba(x)
    return prob[:, 1].astype(float)


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= threshold).astype(int)
    out: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "n": int(len(y_true)),
        "pos": int(y_true.sum()),
        "neg": int(len(y_true) - y_true.sum()),
    }
    if len(set(y_true.tolist())) == 2:
        out["auroc"] = float(roc_auc_score(y_true, y_score))
        out["auprc"] = float(average_precision_score(y_true, y_score))
    else:
        out["auroc"] = float("nan")
        out["auprc"] = float("nan")
    return out


def choose_threshold(y_true: np.ndarray, y_score: np.ndarray, metric: str = "mcc") -> float:
    candidates = np.linspace(0.05, 0.95, 181)
    best_t = 0.5
    best_v = -np.inf
    for t in candidates:
        y_pred = (y_score >= t).astype(int)
        if metric == "f1":
            value = f1_score(y_true, y_pred, zero_division=0)
        else:
            value = matthews_corrcoef(y_true, y_pred)
        if value > best_v:
            best_v = value
            best_t = float(t)
    return best_t


def result_row(result: BaselineResult, extra: Dict[str, object] | None = None) -> Dict[str, object]:
    row: Dict[str, object] = {
        "benchmark": result.benchmark,
        "baseline": result.baseline,
        "threshold": result.threshold,
        "runtime_sec": result.runtime_sec,
    }
    row.update(result.metrics)
    if extra:
        row.update(extra)
    return row


def save_df(df: pd.DataFrame, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    try:
        with open(csv_path.with_suffix(".md"), "w", encoding="utf-8") as f:
            f.write(df.to_markdown(index=False))
            f.write("\n")
    except Exception:
        pass
    print(f"[write] {csv_path}")


def summarize_loso(loso_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = ["accuracy", "precision", "recall", "f1", "mcc", "auroc", "auprc"]
    for baseline, sub in loso_df.groupby("baseline"):
        row: Dict[str, object] = {
            "benchmark": "source_pathogen_loso_high_support",
            "baseline": baseline,
            "n_folds": int(len(sub)),
            "support_min": int(sub["n"].min()),
            "support_median": float(sub["n"].median()),
            "support_max": int(sub["n"].max()),
        }
        for metric in metrics:
            values = sub[metric].astype(float)
            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1))
            row[f"{metric}_median"] = float(values.median())
            row[f"{metric}_iqr"] = float(q3 - q1)
            row[f"{metric}_q1"] = float(q1)
            row[f"{metric}_q3"] = float(q3)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("auprc_mean", ascending=False)


def top_bottom_species(loso_df: pd.DataFrame, metric: str = "auprc", k: int = 5) -> pd.DataFrame:
    rows = []
    for baseline, sub in loso_df.groupby("baseline"):
        sub = sub.sort_values(metric, ascending=False)
        for rank, (_, row) in enumerate(sub.head(k).iterrows(), start=1):
            out = row.to_dict()
            out["rank_group"] = f"top_{metric}"
            out["rank"] = rank
            rows.append(out)
        for rank, (_, row) in enumerate(sub.tail(k).sort_values(metric, ascending=True).iterrows(), start=1):
            out = row.to_dict()
            out["rank_group"] = f"bottom_{metric}"
            out["rank"] = rank
            rows.append(out)
    return pd.DataFrame(rows)


def latex_float(x: float, digits: int = 3) -> str:
    if pd.isna(x):
        return "NA"
    return f"{float(x):.{digits}f}"


def mean_std(row: pd.Series, metric: str) -> str:
    return f"{latex_float(row[f'{metric}_mean'])} $\\pm$ {latex_float(row[f'{metric}_std'])}"


def median_iqr(row: pd.Series, metric: str) -> str:
    return f"{latex_float(row[f'{metric}_median'])} [{latex_float(row[f'{metric}_q1'])}, {latex_float(row[f'{metric}_q3'])}]"


def write_latex_tables(ood_df: pd.DataFrame, loso_summary: pd.DataFrame, top_bottom: pd.DataFrame, out_path: Path) -> None:
    lines: List[str] = []
    lines.append("% Auto-generated by run_lightweight_protocol_baselines.py")
    lines.append("% Source-pathogen OOD and high-support LOSO performance results.")
    lines.append("")
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Source-pathogen species-disjoint OOD performance. Threshold-dependent metrics use an MCC-optimized validation threshold.}")
    lines.append("\\label{tab:source_pathogen_ood_perf}")
    lines.append("\\begin{tabular}{lrrrrrrr}")
    lines.append("\\toprule")
    lines.append("Model & Accuracy & Precision & Recall & F1 & MCC & AUROC & AUPRC \\\\")
    lines.append("\\midrule")
    for _, row in ood_df.sort_values("auprc", ascending=False).iterrows():
        lines.append(
            f"{row['baseline'].replace('_', '\\_')} & "
            f"{latex_float(row['accuracy'])} & {latex_float(row['precision'])} & "
            f"{latex_float(row['recall'])} & {latex_float(row['f1'])} & "
            f"{latex_float(row['mcc'])} & {latex_float(row['auroc'])} & {latex_float(row['auprc'])} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Source-pathogen LOSO performance on high-support held-out species. Values are mean $\\pm$ standard deviation across LOSO folds. Median and IQR are reported in the text/supplementary table.}")
    lines.append("\\label{tab:source_pathogen_loso_perf}")
    lines.append("\\begin{tabular}{lrrrrr}")
    lines.append("\\toprule")
    lines.append("Model & Folds & F1 & MCC & AUROC & AUPRC \\\\")
    lines.append("\\midrule")
    for _, row in loso_summary.sort_values("auprc_mean", ascending=False).iterrows():
        lines.append(
            f"{row['baseline'].replace('_', '\\_')} & {int(row['n_folds'])} & "
            f"{mean_std(row, 'f1')} & {mean_std(row, 'mcc')} & "
            f"{mean_std(row, 'auroc')} & {mean_std(row, 'auprc')} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Top and bottom source-pathogen LOSO species by AUPRC for the best lightweight baseline.}")
    lines.append("\\label{tab:source_pathogen_loso_species_extremes}")
    lines.append("\\begin{tabular}{llrrrr}")
    lines.append("\\toprule")
    lines.append("Group & Held-out species & Support & F1 & MCC & AUPRC \\\\")
    lines.append("\\midrule")
    if len(loso_summary) > 0:
        best_baseline = str(loso_summary.sort_values("auprc_mean", ascending=False).iloc[0]["baseline"])
        tb = top_bottom[top_bottom["baseline"] == best_baseline].copy()
        tb = tb[tb["rank_group"].isin(["top_auprc", "bottom_auprc"])]
        for _, row in tb.iterrows():
            species = str(row["source_species"]).replace("_", "\\_")
            species = species[:80]
            group = str(row["rank_group"]).replace("_", " ")
            lines.append(
                f"{group} & {species} & {int(row['n'])} & "
                f"{latex_float(row['f1'])} & {latex_float(row['mcc'])} & {latex_float(row['auprc'])} \\\\"
            )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[write] {out_path}")


def write_interpretation(ood_df: pd.DataFrame, loso_summary: pd.DataFrame, top_bottom: pd.DataFrame, out_path: Path) -> None:
    best_ood = ood_df.sort_values("auprc", ascending=False).iloc[0]
    best_loso = loso_summary.sort_values("auprc_mean", ascending=False).iloc[0]
    tb = top_bottom[
        (top_bottom["baseline"] == best_loso["baseline"]) & top_bottom["rank_group"].isin(["top_auprc", "bottom_auprc"])
    ].copy()

    lines = [
        "# Source-pathogen OOD/LOSO performance insert",
        "",
        "Use this text to replace a Results paragraph that currently only reports split statistics.",
        "",
        (
            "On the source-pathogen species-disjoint OOD split, the best lightweight baseline "
            f"({best_ood['baseline']}) achieved AUROC {best_ood['auroc']:.3f}, "
            f"AUPRC {best_ood['auprc']:.3f}, MCC {best_ood['mcc']:.3f}, and F1 {best_ood['f1']:.3f} "
            f"on {int(best_ood['n']):,} held-out test sequences."
        ),
        (
            "For the high-support LOSO analysis, each fold was retrained after removing the held-out "
            f"source pathogen species. Across {int(best_loso['n_folds'])} LOSO folds "
            f"(minimum held-out support {int(best_loso['support_min'])}), "
            f"{best_loso['baseline']} obtained AUROC {best_loso['auroc_mean']:.3f} ± {best_loso['auroc_std']:.3f}, "
            f"AUPRC {best_loso['auprc_mean']:.3f} ± {best_loso['auprc_std']:.3f}, "
            f"MCC {best_loso['mcc_mean']:.3f} ± {best_loso['mcc_std']:.3f}, and "
            f"F1 {best_loso['f1_mean']:.3f} ± {best_loso['f1_std']:.3f}."
        ),
        (
            f"The corresponding median [IQR] values were AUROC {median_iqr(best_loso, 'auroc')}, "
            f"AUPRC {median_iqr(best_loso, 'auprc')}, MCC {median_iqr(best_loso, 'mcc')}, and "
            f"F1 {median_iqr(best_loso, 'f1')}."
        ),
        "",
        "Top/bottom LOSO species by AUPRC for the best baseline:",
        "",
    ]
    if len(tb) > 0:
        lines.append(tb[["rank_group", "rank", "source_species", "n", "f1", "mcc", "auroc", "auprc"]].to_markdown(index=False))
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[write] {out_path}")


def run_ood(df: pd.DataFrame, x_all: np.ndarray, outdir: Path) -> pd.DataFrame:
    rows = []
    y = df["label"].astype(int).to_numpy()
    split = df["split"].astype(str).to_numpy()
    record_cols = ["record_id", "label", "split", "source_species_key", "source_species"]

    train_mask = split == "train"
    val_mask = split == "val"
    test_mask = split == "test"

    for baseline, names in BASELINES.items():
        t0 = time.time()
        idx = feature_indices(names)
        model = fit_logreg(x_all[train_mask][:, idx], y[train_mask])
        val_scores = positive_scores(model, x_all[val_mask][:, idx])
        threshold = choose_threshold(y[val_mask], val_scores, metric="mcc")
        test_scores = positive_scores(model, x_all[test_mask][:, idx])
        metrics = compute_metrics(y[test_mask], test_scores, threshold)
        result = BaselineResult(
            baseline=baseline,
            benchmark="source_pathogen_species_disjoint_ood",
            threshold=threshold,
            runtime_sec=time.time() - t0,
            metrics=metrics,
        )
        rows.append(result_row(result))

        pred = df.loc[test_mask, record_cols].copy()
        pred["score"] = test_scores
        pred["threshold"] = threshold
        pred["pred"] = (test_scores >= threshold).astype(int)
        pred.to_csv(outdir / f"source_species_ood_predictions_{baseline}.csv", index=False, encoding="utf-8-sig")
        print(f"[OOD] {baseline}: AUROC={metrics['auroc']:.4f} AUPRC={metrics['auprc']:.4f} MCC={metrics['mcc']:.4f}")

    out = pd.DataFrame(rows).sort_values("auprc", ascending=False)
    save_df(out, outdir / "source_species_ood_lightweight_metrics.csv")
    return out


def run_loso(df: pd.DataFrame, x_all: np.ndarray, loso_registry: pd.DataFrame, min_support: int, max_folds: int | None, outdir: Path) -> pd.DataFrame:
    registry = loso_registry[loso_registry["n"].astype(int) >= int(min_support)].copy()
    registry = registry.sort_values(["n", "heldout_species_key"], ascending=[False, True]).reset_index(drop=True)
    if max_folds is not None:
        registry = registry.head(max_folds).copy()
    if len(registry) == 0:
        raise RuntimeError(f"No LOSO folds meet min_support={min_support}.")

    rows = []
    pred_rows = []
    y = df["label"].astype(int).to_numpy()
    species_key = df["source_species_key"].astype(str).to_numpy()
    source_species_lookup = (
        df.groupby("source_species_key")["source_species"].agg(lambda s: next((str(x) for x in s if str(x)), "")).to_dict()
    )

    print(f"[LOSO] running {len(registry)} folds with min_support={min_support}")
    for _, fold in registry.iterrows():
        heldout_key = str(fold["heldout_species_key"])
        test_mask = species_key == heldout_key
        train_mask = ~test_mask
        if y[test_mask].min() == y[test_mask].max():
            continue
        for baseline, names in BASELINES.items():
            t0 = time.time()
            idx = feature_indices(names)
            model = fit_logreg(x_all[train_mask][:, idx], y[train_mask])
            test_scores = positive_scores(model, x_all[test_mask][:, idx])
            threshold = 0.5
            metrics = compute_metrics(y[test_mask], test_scores, threshold)
            extra = {
                "fold_id": int(fold["fold_id"]),
                "heldout_species_key": heldout_key,
                "source_species": source_species_lookup.get(heldout_key, str(fold.get("source_species", ""))),
                "support_pos": int(fold["pos"]),
                "support_neg": int(fold["neg"]),
            }
            result = BaselineResult(
                baseline=baseline,
                benchmark="source_pathogen_loso_high_support",
                threshold=threshold,
                runtime_sec=time.time() - t0,
                metrics=metrics,
            )
            row = result_row(result, extra=extra)
            rows.append(row)

            pred = pd.DataFrame(
                {
                    "record_id": df.loc[test_mask, "record_id"].to_numpy(),
                    "label": y[test_mask],
                    "score": test_scores,
                    "threshold": threshold,
                    "pred": (test_scores >= threshold).astype(int),
                    "fold_id": int(fold["fold_id"]),
                    "heldout_species_key": heldout_key,
                    "source_species": extra["source_species"],
                    "baseline": baseline,
                }
            )
            pred_rows.append(pred)
        print(f"[LOSO] fold {int(fold['fold_id'])}: {heldout_key} n={int(fold['n'])}")

    out = pd.DataFrame(rows).sort_values(["baseline", "auprc"], ascending=[True, False])
    save_df(out, outdir / "source_species_loso_lightweight_metrics.csv")
    if pred_rows:
        pred_df = pd.concat(pred_rows, ignore_index=True)
        pred_df.to_csv(outdir / "source_species_loso_lightweight_predictions.csv", index=False, encoding="utf-8-sig")
        print(f"[write] {outdir / 'source_species_loso_lightweight_predictions.csv'}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lightweight source-pathogen OOD/LOSO baselines.")
    parser.add_argument("--ood_manifest", type=Path, default=Path("outputs/source_species_ood_manifest.csv"))
    parser.add_argument("--loso_support", type=Path, default=Path("outputs/source_species_loso_support.csv"))
    parser.add_argument("--outdir", type=Path, default=Path("outputs/source_pathogen_lightweight_baselines"))
    parser.add_argument("--loso_min_support", type=int, default=1250)
    parser.add_argument("--max_loso_folds", type=int, default=None)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    print(f"[load] {args.ood_manifest}")
    df = pd.read_csv(
        args.ood_manifest,
        usecols=[
            "record_id",
            "label",
            "split",
            "source_species_key",
            "source_species",
            "clean_len",
            "clean_seq",
        ],
        dtype={"source_species_key": str},
        low_memory=False,
    )
    df["source_species"] = df["source_species"].fillna("").astype(str)
    df["clean_seq"] = df["clean_seq"].fillna("").astype(str)
    df["clean_len"] = df["clean_len"].astype(int)
    y_counts = df["label"].value_counts().to_dict()
    print(f"[data] rows={len(df):,} label_counts={y_counts}")

    t0 = time.time()
    x_all = build_features(df)
    print(f"[features] complete in {time.time() - t0:.1f}s")

    ood_df = run_ood(df, x_all, args.outdir)

    print(f"[load] {args.loso_support}")
    loso_registry = pd.read_csv(args.loso_support, dtype={"heldout_species_key": str})
    loso_df = run_loso(df, x_all, loso_registry, args.loso_min_support, args.max_loso_folds, args.outdir)
    loso_summary = summarize_loso(loso_df)
    save_df(loso_summary, args.outdir / "source_species_loso_lightweight_summary.csv")

    top_bottom = top_bottom_species(loso_df, metric="auprc", k=5)
    save_df(top_bottom, args.outdir / "source_species_loso_lightweight_top_bottom.csv")

    write_latex_tables(
        ood_df,
        loso_summary,
        top_bottom,
        args.outdir / "source_pathogen_ood_loso_results_tables.tex",
    )
    write_interpretation(
        ood_df,
        loso_summary,
        top_bottom,
        args.outdir / "source_pathogen_ood_loso_results_insert.md",
    )

    manifest = {
        "ood_manifest": str(args.ood_manifest),
        "loso_support": str(args.loso_support),
        "outdir": str(args.outdir),
        "loso_min_support": args.loso_min_support,
        "max_loso_folds": args.max_loso_folds,
        "baselines": BASELINES,
        "feature_names": FEATURE_NAMES,
        "ood_threshold_rule": "MCC-optimized threshold selected on source_species_ood_manifest validation split",
        "loso_threshold_rule": "fixed 0.5 threshold; each fold refits after excluding the held-out source species",
    }
    (args.outdir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[write] {args.outdir / 'run_manifest.json'}")


if __name__ == "__main__":
    main()
