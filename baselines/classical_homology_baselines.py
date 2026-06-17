# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import math
import os
import re
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from baseline_common import (
    DATASET_TOPK_REQUIRE_BOTH_LABELS,
    DATASET_TOPK_SPECIES,
    NEG_FILE,
    OOD_SPECIES_TOPK,
    OUT_ROOT,
    POS_FILE,
    SEED,
    TAXONOMY_MAP_FILE,
    attach_result_context,
    clean_sequence,
    compute_classification_metrics,
    ensure_dir,
    filter_species_candidates,
    find_executable,
    gc_fraction,
    kmer_tokens,
    load_combined_data,
    maybe_plot_curves,
    now_stamp,
    probs_to_preds,
    restrict_dataset_to_top_species,
    save_df,
    save_json,
    save_prediction_table,
    save_run_readme,
    save_split_frames,
    set_seed,
    split_holdout_unit,
    split_three_way_by_unit,
    write_fasta,
)


class KmerAnalyzer:
    def __init__(self, k: int):
        self.k = k

    def __call__(self, seq: str):
        return kmer_tokens(seq, k=self.k)


def numeric_feature_matrix(df: pd.DataFrame, mode: str) -> np.ndarray:
    features = []
    if mode in {"length", "length_gc"}:
        features.append(np.log1p(df["seq_len"].values.astype(float)))
    if mode in {"gc", "length_gc"}:
        features.append(df["seq"].map(gc_fraction).values.astype(float))
    return np.vstack(features).T


def positive_scores(model, x) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(x)[:, 1], dtype=float)
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(x), dtype=float)
        scores = np.clip(scores, -60, 60)
        return 1.0 / (1.0 + np.exp(-scores))
    return np.asarray(model.predict(x), dtype=float)


def make_numeric_lr(mode: str) -> Tuple[str, Pipeline]:
    name_map = {
        "length": "length_only_logistic_regression",
        "gc": "gc_only_logistic_regression",
        "length_gc": "length_gc_logistic_regression",
    }
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    solver="liblinear",
                    class_weight="balanced",
                    random_state=SEED,
                    max_iter=5000,
                ),
            ),
        ]
    )
    return name_map[mode], model


def make_6mer_tfidf_lr(k: int, min_df: int) -> Tuple[str, Pipeline]:
    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer=KmerAnalyzer(k),
                    lowercase=False,
                    min_df=min_df,
                    norm="l2",
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    solver="liblinear",
                    class_weight="balanced",
                    random_state=SEED,
                    max_iter=5000,
                ),
            ),
        ]
    )
    return "%dmer_tfidf_logistic_regression" % k, model


def make_6mer_tfidf_svm(k: int, min_df: int) -> Tuple[str, Pipeline]:
    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer=KmerAnalyzer(k),
                    lowercase=False,
                    min_df=min_df,
                    norm="l2",
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LinearSVC(
                    class_weight="balanced",
                    random_state=SEED,
                    max_iter=10000,
                ),
            ),
        ]
    )
    return "%dmer_tfidf_linear_svm" % k, model


def make_6mer_xgboost(k: int, min_df: int, jobs: int):
    try:
        from xgboost import XGBClassifier
    except Exception as exc:
        raise RuntimeError("xgboost is not installed in the active Python environment: %s" % exc)

    model = Pipeline(
        [
            (
                "count",
                CountVectorizer(
                    analyzer=KmerAnalyzer(k),
                    lowercase=False,
                    min_df=min_df,
                    binary=False,
                ),
            ),
            (
                "clf",
                XGBClassifier(
                    n_estimators=500,
                    max_depth=4,
                    learning_rate=0.03,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=SEED,
                    n_jobs=jobs,
                    tree_method="hist",
                ),
            ),
        ]
    )
    return "%dmer_xgboost" % k, model


def run_sklearn_baseline(
    baseline_name: str,
    model,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    run_dir: Path,
    feature_mode: Optional[str] = None,
) -> Dict:
    ensure_dir(run_dir)
    if feature_mode is None:
        x_train = df_train["seq"].tolist()
        x_val = df_val["seq"].tolist()
        x_test = df_test["seq"].tolist()
    else:
        x_train = numeric_feature_matrix(df_train, feature_mode)
        x_val = numeric_feature_matrix(df_val, feature_mode)
        x_test = numeric_feature_matrix(df_test, feature_mode)

    y_train = df_train["label"].values.astype(int)
    y_val = df_val["label"].values.astype(int)
    y_test = df_test["label"].values.astype(int)

    model.fit(x_train, y_train)
    val_probs = positive_scores(model, x_val)
    test_probs = positive_scores(model, x_test)
    val_preds = probs_to_preds(val_probs)
    test_preds = probs_to_preds(test_probs)

    val_metrics = compute_classification_metrics(y_val, val_preds, val_probs)
    test_metrics = compute_classification_metrics(y_test, test_preds, test_probs)
    result = attach_result_context(test_metrics, baseline_name, run_dir, df_train, df_val, df_test)
    result["val_accuracy"] = val_metrics["accuracy"]
    result["val_f1"] = val_metrics["f1"]
    result["val_mcc"] = val_metrics["mcc"]
    result["val_auroc"] = val_metrics["auroc"]
    result["val_auprc"] = val_metrics["auprc"]

    save_json({"baseline": baseline_name, "split": "val", **val_metrics}, run_dir / "val_metrics.json")
    save_json(result, run_dir / "final_test_metrics.json")
    save_prediction_table(df_val, val_probs, val_preds, run_dir / "val_predictions.csv")
    save_prediction_table(df_test, test_probs, test_preds, run_dir / "test_predictions.csv")
    maybe_plot_curves(y_test, test_probs, test_preds, run_dir, prefix="test")
    try:
        joblib.dump(model, run_dir / "model.joblib")
    except Exception as exc:
        save_json({"warning": "model serialization failed", "error": str(exc)}, run_dir / "model_serialization_warning.json")
    return result


def parse_label_from_seq_id(seq_id: str) -> int:
    m = re.search(r"_y([01])$", seq_id)
    if not m:
        return 0
    return int(m.group(1))


def run_blast_nearest_neighbor(
    df_train: pd.DataFrame,
    df_query: pd.DataFrame,
    run_dir: Path,
    split_name: str,
    jobs: int,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    makeblastdb = find_executable(
        ["makeblastdb"],
        [
            r"D:\6Bioinformatics\blast-2.17.0+\bin\makeblastdb.exe",
            r"E:\2RTools\BLAST\blast-2.17.0+\bin\makeblastdb.exe",
        ],
    )
    blastn = find_executable(
        ["blastn"],
        [
            r"D:\6Bioinformatics\blast-2.17.0+\bin\blastn.exe",
            r"E:\2RTools\BLAST\blast-2.17.0+\bin\blastn.exe",
        ],
    )
    if not makeblastdb or not blastn:
        raise RuntimeError("BLAST executables were not found (makeblastdb=%s, blastn=%s)." % (makeblastdb, blastn))

    tmp_dir = ensure_dir(run_dir / ("blast_%s_tmp" % split_name))
    train_fasta = tmp_dir / "train.fasta"
    query_fasta = tmp_dir / ("%s.fasta" % split_name)
    train_ids = write_fasta(df_train, train_fasta, "train")
    query_ids = write_fasta(df_query, query_fasta, split_name)
    train_id_to_label = dict(zip(train_ids["seq_id"], train_ids["label"]))
    query_id_to_index = dict(zip(query_ids["seq_id"], query_ids["row_index"]))

    db_prefix = tmp_dir / "train_db"
    subprocess.run(
        [makeblastdb, "-in", str(train_fasta), "-dbtype", "nucl", "-out", str(db_prefix)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    out_tsv = tmp_dir / ("%s_vs_train.blast.tsv" % split_name)
    outfmt = "6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore qlen slen"
    subprocess.run(
        [
            blastn,
            "-query",
            str(query_fasta),
            "-db",
            str(db_prefix),
            "-task",
            "blastn",
            "-dust",
            "no",
            "-soft_masking",
            "false",
            "-evalue",
            "1e-5",
            "-max_target_seqs",
            "10",
            "-num_threads",
            str(max(1, jobs)),
            "-outfmt",
            outfmt,
            "-out",
            str(out_tsv),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    cols = [
        "qseqid",
        "sseqid",
        "pident",
        "length",
        "mismatch",
        "gapopen",
        "qstart",
        "qend",
        "sstart",
        "send",
        "evalue",
        "bitscore",
        "qlen",
        "slen",
    ]
    if out_tsv.exists() and out_tsv.stat().st_size > 0:
        hits = pd.read_csv(out_tsv, sep="\t", names=cols)
    else:
        hits = pd.DataFrame(columns=cols)
    if len(hits) > 0:
        hits["subject_label"] = hits["sseqid"].map(lambda x: train_id_to_label.get(x, parse_label_from_seq_id(str(x))))
        hits["query_index"] = hits["qseqid"].map(query_id_to_index)
        hits = hits.sort_values(["qseqid", "bitscore", "evalue", "pident", "length"], ascending=[True, False, True, False, False])

    train_rate = float(df_train["label"].mean())
    probs = np.full(len(df_query), train_rate, dtype=float)
    preds = np.full(len(df_query), int(train_rate >= 0.5), dtype=int)
    detail_rows = []
    for qseqid, qrow in query_ids[["seq_id", "row_index"]].itertuples(index=False):
        sub_hits = hits[hits["qseqid"] == qseqid] if len(hits) > 0 else pd.DataFrame()
        idx = int(qrow)
        if len(sub_hits) == 0:
            detail_rows.append(
                {
                    "homology_backend": "blastn",
                    "nearest_subject": "",
                    "nearest_label": np.nan,
                    "nearest_pident": np.nan,
                    "nearest_aln_len": np.nan,
                    "nearest_bitscore": np.nan,
                    "nearest_evalue": np.nan,
                    "n_hits": 0,
                    "no_hit": True,
                }
            )
            continue
        nearest = sub_hits.iloc[0]
        weights = sub_hits["bitscore"].astype(float).clip(lower=1e-9).values
        labels = sub_hits["subject_label"].astype(float).values
        weighted_prob = float(np.sum(weights * labels) / np.sum(weights))
        probs[idx] = weighted_prob
        preds[idx] = int(nearest["subject_label"])
        detail_rows.append(
            {
                "homology_backend": "blastn",
                "nearest_subject": nearest["sseqid"],
                "nearest_label": int(nearest["subject_label"]),
                "nearest_pident": float(nearest["pident"]),
                "nearest_aln_len": int(nearest["length"]),
                "nearest_bitscore": float(nearest["bitscore"]),
                "nearest_evalue": float(nearest["evalue"]),
                "n_hits": int(len(sub_hits)),
                "no_hit": False,
            }
        )

    detail = pd.DataFrame(detail_rows)
    save_df(hits, run_dir / ("%s_all_blast_hits.tsv" % split_name))
    return probs, preds, detail


def run_mmseqs_or_blast_nearest_neighbor(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    run_dir: Path,
    jobs: int,
    backend: str,
) -> Dict:
    ensure_dir(run_dir)
    mmseqs = find_executable(["mmseqs"])
    backend_used = backend
    if backend == "auto":
        backend_used = "mmseqs2" if mmseqs else "blastn"
    if backend_used == "mmseqs2":
        raise RuntimeError(
            "MMseqs2 executable was requested/found, but this script currently requires the BLASTN fallback on this machine. "
            "Set --homology-backend blastn or install a tested mmseqs nucleotide workflow."
        )
    if backend_used != "blastn":
        raise RuntimeError("Unsupported homology backend: %s" % backend_used)

    val_probs, val_preds, val_detail = run_blast_nearest_neighbor(df_train, df_val, run_dir, "val", jobs)
    test_probs, test_preds, test_detail = run_blast_nearest_neighbor(df_train, df_test, run_dir, "test", jobs)
    val_metrics = compute_classification_metrics(df_val["label"].values, val_preds, val_probs)
    test_metrics = compute_classification_metrics(df_test["label"].values, test_preds, test_probs)
    result = attach_result_context(
        test_metrics,
        "mmseqs2_or_blast_nearest_neighbor",
        run_dir,
        df_train,
        df_val,
        df_test,
    )
    result["homology_backend_used"] = backend_used
    result["val_accuracy"] = val_metrics["accuracy"]
    result["val_f1"] = val_metrics["f1"]
    result["val_mcc"] = val_metrics["mcc"]
    result["val_auroc"] = val_metrics["auroc"]
    result["val_auprc"] = val_metrics["auprc"]
    save_json({"baseline": "mmseqs2_or_blast_nearest_neighbor", "split": "val", **val_metrics}, run_dir / "val_metrics.json")
    save_json(result, run_dir / "final_test_metrics.json")
    save_prediction_table(df_val, val_probs, val_preds, run_dir / "val_predictions.csv", extra=val_detail)
    save_prediction_table(df_test, test_probs, test_preds, run_dir / "test_predictions.csv", extra=test_detail)
    maybe_plot_curves(df_test["label"].values, test_probs, test_preds, run_dir, prefix="test")
    return result


def run_all_classical_for_split(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    out_dir: Path,
    args,
) -> pd.DataFrame:
    ensure_dir(out_dir)
    rows: List[Dict] = []

    for mode in ["length", "gc", "length_gc"]:
        baseline_name, model = make_numeric_lr(mode)
        print("[Baseline] %s" % baseline_name)
        try:
            rows.append(run_sklearn_baseline(baseline_name, model, df_train, df_val, df_test, out_dir / baseline_name, feature_mode=mode))
        except Exception as exc:
            err_dir = ensure_dir(out_dir / baseline_name)
            rows.append(attach_result_context({}, baseline_name, err_dir, df_train, df_val, df_test, status="error", error=str(exc)))
            save_json({"error": str(exc), "traceback": traceback.format_exc()}, err_dir / "error.json")

    sequence_baselines = [
        make_6mer_tfidf_lr(args.kmer, args.min_df),
        make_6mer_tfidf_svm(args.kmer, args.min_df),
    ]
    for baseline_name, model in sequence_baselines:
        print("[Baseline] %s" % baseline_name)
        try:
            rows.append(run_sklearn_baseline(baseline_name, model, df_train, df_val, df_test, out_dir / baseline_name))
        except Exception as exc:
            err_dir = ensure_dir(out_dir / baseline_name)
            rows.append(attach_result_context({}, baseline_name, err_dir, df_train, df_val, df_test, status="error", error=str(exc)))
            save_json({"error": str(exc), "traceback": traceback.format_exc()}, err_dir / "error.json")

    baseline_name = "%dmer_xgboost" % args.kmer
    if not args.skip_xgboost:
        print("[Baseline] %s" % baseline_name)
        try:
            xgb_name, xgb_model = make_6mer_xgboost(args.kmer, args.min_df, args.jobs)
            rows.append(run_sklearn_baseline(xgb_name, xgb_model, df_train, df_val, df_test, out_dir / xgb_name))
        except Exception as exc:
            err_dir = ensure_dir(out_dir / baseline_name)
            rows.append(attach_result_context({}, baseline_name, err_dir, df_train, df_val, df_test, status="skipped", error=str(exc)))
            save_json({"error": str(exc), "traceback": traceback.format_exc()}, err_dir / "skipped.json")

    if not args.skip_homology:
        baseline_name = "mmseqs2_or_blast_nearest_neighbor"
        print("[Baseline] %s" % baseline_name)
        try:
            rows.append(
                run_mmseqs_or_blast_nearest_neighbor(
                    df_train,
                    df_val,
                    df_test,
                    out_dir / baseline_name,
                    args.jobs,
                    args.homology_backend,
                )
            )
        except Exception as exc:
            err_dir = ensure_dir(out_dir / baseline_name)
            rows.append(attach_result_context({}, baseline_name, err_dir, df_train, df_val, df_test, status="error", error=str(exc)))
            save_json({"error": str(exc), "traceback": traceback.format_exc()}, err_dir / "error.json")

    summary = pd.DataFrame(rows)
    save_df(summary, out_dir / "baseline_summary.csv", out_dir / "baseline_summary.md")
    return summary


def prepare_dataset(root_dir: Path, args) -> pd.DataFrame:
    df_all = load_combined_data(args.pos_file, args.neg_file, args.taxonomy_map_file)
    raw_manifest = df_all[["seq_len", "label", "tax_id", "species_name", "species_label", "clade_key", "group_id", "species_key"]].copy()
    save_df(raw_manifest, root_dir / "dataset_manifest_before_topk.csv")
    if args.topk_species > 0:
        df_all, manifest_all, selected = restrict_dataset_to_top_species(
            df_all,
            top_k=args.topk_species,
            require_both_labels=DATASET_TOPK_REQUIRE_BOTH_LABELS,
        )
        save_df(manifest_all, root_dir / "species_manifest_before_filter.csv", root_dir / "species_manifest_before_filter.md")
        save_df(selected, root_dir / "species_manifest_selected.csv", root_dir / "species_manifest_selected.md")
    save_df(
        df_all[["seq_len", "label", "tax_id", "species_name", "species_label", "clade_key", "group_id", "species_key"]],
        root_dir / "dataset_manifest_compact.csv",
        root_dir / "dataset_manifest_compact.md",
    )
    return df_all


def run_standard(df_all: pd.DataFrame, root_dir: Path, args) -> pd.DataFrame:
    out_dir = ensure_dir(root_dir / "standard_species_disjoint")
    df_train, df_val, df_test, split_stats = split_three_way_by_unit(df_all, unit_col="species_key", random_state=args.seed)
    save_json(split_stats, out_dir / "split_stats.json")
    save_split_frames(out_dir, df_train, df_val, df_test)
    summary = run_all_classical_for_split(df_train, df_val, df_test, out_dir, args)
    save_df(summary, root_dir / "standard_species_disjoint_summary.csv", root_dir / "standard_species_disjoint_summary.md")
    return summary


def run_loso(df_all: pd.DataFrame, root_dir: Path, args) -> pd.DataFrame:
    out_dir = ensure_dir(root_dir / "loso_species")
    manifest = df_all.groupby("species_key").size()
    species_manifest = pd.DataFrame()
    from baseline_common import build_unit_manifest

    species_manifest = build_unit_manifest(df_all, "species_key")
    candidates = filter_species_candidates(species_manifest)
    if args.max_loso > 0:
        candidates = candidates.head(args.max_loso).copy()
    save_df(species_manifest, out_dir / "species_manifest_all.csv", out_dir / "species_manifest_all.md")
    save_df(candidates, out_dir / "species_candidates.csv", out_dir / "species_candidates.md")

    all_rows = []
    for i, row in candidates.iterrows():
        species_key = row["species_key"]
        species_label = row.get("species_label", species_key)
        run_dir = ensure_dir(out_dir / ("holdout_species_%02d" % (i + 1)))
        print("[LOSO] %s" % species_key)
        try:
            df_train, df_val, df_test, split_stats = split_holdout_unit(
                df_all,
                holdout_unit_col="species_key",
                holdout_value=species_key,
                val_group_col="species_key",
                random_state=args.seed + i,
            )
            save_json(split_stats, run_dir / "split_stats.json")
            save_split_frames(run_dir, df_train, df_val, df_test)
            summary = run_all_classical_for_split(df_train, df_val, df_test, run_dir, args)
            summary.insert(0, "heldout_species", species_key)
            summary.insert(1, "heldout_species_label", species_label)
            summary["species_support_total"] = int(row["n_total"])
            summary["species_support_pos"] = int(row["n_pos"])
            summary["species_support_neg"] = int(row["n_neg"])
            all_rows.append(summary)
        except Exception as exc:
            all_rows.append(
                pd.DataFrame(
                    [
                        {
                            "heldout_species": species_key,
                            "heldout_species_label": species_label,
                            "baseline": "ALL",
                            "status": "error",
                            "error": str(exc),
                        }
                    ]
                )
            )
            save_json({"error": str(exc), "traceback": traceback.format_exc()}, run_dir / "error.json")
    summary_all = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    save_df(summary_all, out_dir / "loso_species_summary.csv", out_dir / "loso_species_summary.md")
    return summary_all


def build_master_summary(root_dir: Path, standard_df: Optional[pd.DataFrame], loso_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    rows = []
    if standard_df is not None and len(standard_df) > 0:
        for _, row in standard_df.iterrows():
            rows.append(
                {
                    "benchmark": "standard_species_disjoint",
                    "baseline": row.get("baseline"),
                    "status": row.get("status"),
                    "accuracy": row.get("accuracy"),
                    "precision": row.get("precision"),
                    "recall": row.get("recall"),
                    "f1": row.get("f1"),
                    "mcc": row.get("mcc"),
                    "auroc": row.get("auroc"),
                    "auprc": row.get("auprc"),
                    "ap": row.get("ap"),
                    "error": row.get("error"),
                }
            )
    if loso_df is not None and len(loso_df) > 0 and "baseline" in loso_df.columns:
        ok = loso_df[loso_df["status"].fillna("ok").eq("ok")].copy()
        for baseline, sub in ok.groupby("baseline"):
            rows.append(
                {
                    "benchmark": "loso_species_mean",
                    "baseline": baseline,
                    "status": "ok",
                    "accuracy": sub["accuracy"].mean(),
                    "precision": sub["precision"].mean(),
                    "recall": sub["recall"].mean(),
                    "f1": sub["f1"].mean(),
                    "mcc": sub["mcc"].mean(),
                    "auroc": sub["auroc"].mean(),
                    "auprc": sub["auprc"].mean(),
                    "ap": sub["ap"].mean(),
                    "n_runs": len(sub),
                }
            )
    master = pd.DataFrame(rows)
    save_df(master, root_dir / "master_baseline_summary.csv", root_dir / "master_baseline_summary.md")
    return master


def parse_args():
    parser = argparse.ArgumentParser(description="Classical composition and homology baselines for SeqPatho.")
    parser.add_argument("--pos-file", default=str(POS_FILE))
    parser.add_argument("--neg-file", default=str(NEG_FILE))
    parser.add_argument("--taxonomy-map-file", default=str(TAXONOMY_MAP_FILE))
    parser.add_argument("--output-root", default=str(OUT_ROOT))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--topk-species", type=int, default=DATASET_TOPK_SPECIES)
    parser.add_argument("--benchmarks", nargs="+", default=["standard"], choices=["standard", "loso"])
    parser.add_argument("--max-loso", type=int, default=OOD_SPECIES_TOPK)
    parser.add_argument("--kmer", type=int, default=6)
    parser.add_argument("--min-df", type=int, default=1)
    parser.add_argument("--jobs", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--skip-xgboost", action="store_true")
    parser.add_argument("--skip-homology", action="store_true")
    parser.add_argument("--homology-backend", default="auto", choices=["auto", "blastn", "mmseqs2"])
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    run_name = args.run_name or ("results_classical_homology_%s" % now_stamp())
    root_dir = ensure_dir(Path(args.output_root) / run_name)
    save_json(vars(args), root_dir / "config.json")
    save_run_readme(
        root_dir / "README.md",
        [
            "# Classical and homology baseline run",
            "",
            "This run uses the same data source, canonical reverse-complement deduplication, top-species filtering, and species-disjoint split style as Models_06_NPlants/Ot3.py.",
            "",
            "Baselines included: length-only LR, GC-only LR, length+GC LR, 6-mer TF-IDF LR, 6-mer TF-IDF linear SVM, optional 6-mer XGBoost, and MMseqs2/BLAST nearest-neighbor.",
        ],
    )
    print("[Output] %s" % root_dir)
    df_all = prepare_dataset(root_dir, args)

    standard_df = None
    loso_df = None
    if "standard" in args.benchmarks:
        standard_df = run_standard(df_all, root_dir, args)
    if "loso" in args.benchmarks:
        loso_df = run_loso(df_all, root_dir, args)
    build_master_summary(root_dir, standard_df, loso_df)
    print("[Done] %s" % root_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
