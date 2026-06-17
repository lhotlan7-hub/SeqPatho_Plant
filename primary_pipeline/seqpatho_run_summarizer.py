#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


METRIC_ORDER = ["Accuracy", "Precision", "Recall", "F1", "MCC", "AUROC", "AUPRC"]


def read_metric_csv(path: Path) -> Dict[str, object]:
    df = pd.read_csv(path)
    if {"Metric", "Value"}.issubset(df.columns):
        out = dict(zip(df["Metric"].astype(str), df["Value"]))
    else:
        out = df.iloc[0].to_dict()
    for key, value in list(out.items()):
        try:
            out[key] = float(value)
        except (TypeError, ValueError):
            pass
    return out


def summarize_single(run_dir: Path, out_csv: Path) -> pd.DataFrame:
    metrics_path = run_dir / "test_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    row = read_metric_csv(metrics_path)
    row["run_dir"] = str(run_dir)
    result = pd.DataFrame([row])
    result.to_csv(out_csv, index=False)
    print(f"[write] {out_csv}")
    return result


def summarize_loso(run_root: Path, out_csv: Path) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for fold_dir in sorted(run_root.glob("fold_*"), key=lambda p: int(p.name.split("_")[-1]) if p.name.split("_")[-1].isdigit() else p.name):
        metrics_path = fold_dir / "test_metrics.csv"
        if not metrics_path.exists():
            continue
        row = read_metric_csv(metrics_path)
        fold_id = fold_dir.name.replace("fold_", "")
        row["fold_id"] = int(fold_id) if str(fold_id).isdigit() else fold_id
        row["run_dir"] = str(fold_dir)
        rows.append(row)
    if not rows:
        raise RuntimeError(f"No fold_*/test_metrics.csv files found under {run_root}")

    result = pd.DataFrame(rows)
    result.to_csv(out_csv, index=False)
    print(f"[write] {out_csv}")

    summary = {}
    for metric in METRIC_ORDER:
        if metric in result.columns:
            summary[metric] = {
                "mean": float(result[metric].mean()),
                "std": float(result[metric].std(ddof=1)) if len(result) > 1 else 0.0,
            }
    summary["folds"] = int(len(result))
    json_path = out_csv.with_suffix(".summary.json")
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[write] {json_path}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ood_run_dir", type=Path, default=None)
    parser.add_argument("--loso_run_root", type=Path, default=None)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.ood_run_dir:
        summarize_single(args.ood_run_dir, args.outdir / "seqpatho_source_pathogen_ood_metrics.csv")
    if args.loso_run_root:
        summarize_loso(args.loso_run_root, args.outdir / "seqpatho_source_pathogen_loso_metrics.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
