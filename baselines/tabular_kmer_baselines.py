# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import math
import pickle
import time
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

from baseline_common import (
    DEFAULT_NEG_FILE,
    DEFAULT_POS_FILE,
    DEFAULT_TAXONOMY_MAP_FILE,
    OOD_VAL_FRAC_FROM_REST,
    SEED,
    compute_classification_metrics,
    filter_species_candidates,
    make_result_dir,
    plot_confusion_matrix,
    plot_metric_bars,
    plot_roc_pr,
    prepare_dataset,
    prediction_frame,
    save_df,
    save_json,
    save_split_files,
    set_seed,
    split_holdout_unit,
    split_three_way_by_unit,
    build_unit_manifest,
    ensure_dir,
)


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def gc_content(seq: str) -> float:
    s = str(seq).upper()
    a = s.count("A")
    c = s.count("C")
    g = s.count("G")
    t = s.count("T")
    denom = a + c + g + t
    return float((g + c) / denom) if denom > 0 else 0.0


def basic_feature_matrix(df: pd.DataFrame, mode: str) -> np.ndarray:
    length = df["seq_len"].astype(float).to_numpy().reshape(-1, 1)
    gc = df["seq"].map(gc_content).astype(float).to_numpy().reshape(-1, 1)
    if mode == "length":
        return length
    if mode == "gc":
        return gc
    if mode == "length_gc":
        return np.hstack([length, gc])
    raise ValueError(f"unknown basic feature mode: {mode}")


def sequence_texts(df: pd.DataFrame) -> List[str]:
    return df["seq"].astype(str).tolist()


def positive_scores(model, x) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(x)
        if probs.ndim == 2 and probs.shape[1] > 1:
            return probs[:, 1].astype(float)
        return probs.reshape(-1).astype(float)
    if hasattr(model, "decision_function"):
        scores = model.decision_function(x).astype(float)
        scores = np.clip(scores, -50, 50)
        return 1.0 / (1.0 + np.exp(-scores))
    preds = model.predict(x)
    return np.asarray(preds, dtype=float)


def safe_predict(model, x) -> Tuple[np.ndarray, np.ndarray]:
    probs = positive_scores(model, x)
    preds = (probs >= 0.5).astype(int)
    return probs, preds


def make_tfidf(max_features: Optional[int], min_df: int) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char",
        ngram_range=(6, 6),
        lowercase=False,
        min_df=min_df,
        max_features=max_features,
        dtype=np.float32,
        sublinear_tf=True,
    )


def save_linear_coefficients(model, out_csv: Path, feature_names: Optional[np.ndarray] = None) -> None:
    rows = []
    clf = model
    if isinstance(model, Pipeline):
        clf = model.steps[-1][1]
        if feature_names is None and "tfidf" in model.named_steps:
            feature_names = model.named_steps["tfidf"].get_feature_names_out()
    if not hasattr(clf, "coef_"):
        return
    coefs = clf.coef_.reshape(-1)
    if feature_names is None:
        feature_names = np.array([f"x{i}" for i in range(len(coefs))])
    order = np.argsort(coefs)
    for idx in order[:50]:
        rows.append({"rank_type": "negative", "feature": feature_names[idx], "coefficient": float(coefs[idx])})
    for idx in order[-50:][::-1]:
        rows.append({"rank_type": "positive", "feature": feature_names[idx], "coefficient": float(coefs[idx])})
    save_df(pd.DataFrame(rows), out_csv)


def save_xgb_importance(model: XGBClassifier, feature_names: np.ndarray, out_csv: Path) -> None:
    try:
        score = model.get_booster().get_score(importance_type="gain")
    except Exception:
        return
    rows = []
    for key, value in score.items():
        if key.startswith("f"):
            idx = int(key[1:])
            feature = feature_names[idx] if idx < len(feature_names) else key
        else:
            feature = key
        rows.append({"feature": feature, "gain": float(value)})
    if rows:
        save_df(pd.DataFrame(rows).sort_values("gain", ascending=False).head(200), out_csv)


def fit_length_gc_logreg(df_train: pd.DataFrame, mode: str):
    x_train = basic_feature_matrix(df_train, mode)
    model = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    max_iter=5000,
                    class_weight="balanced",
                    random_state=SEED,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    model.fit(x_train, df_train["label"].to_numpy())
    return model


def fit_kmer_logreg(df_train: pd.DataFrame, max_features: Optional[int], min_df: int):
    model = Pipeline(
        steps=[
            ("tfidf", make_tfidf(max_features=max_features, min_df=min_df)),
            (
                "logreg",
                LogisticRegression(
                    max_iter=5000,
                    class_weight="balanced",
                    random_state=SEED,
                    solver="liblinear",
                ),
            ),
        ]
    )
    model.fit(sequence_texts(df_train), df_train["label"].to_numpy())
    return model


def fit_kmer_svm(df_train: pd.DataFrame, max_features: Optional[int], min_df: int):
    model = Pipeline(
        steps=[
            ("tfidf", make_tfidf(max_features=max_features, min_df=min_df)),
            (
                "svm",
                CalibratedClassifierCV(
                    base_estimator=LinearSVC(class_weight="balanced", random_state=SEED, max_iter=10000),
                    cv=3,
                    method="sigmoid",
                ),
            ),
        ]
    )
    model.fit(sequence_texts(df_train), df_train["label"].to_numpy())
    return model


def fit_kmer_xgboost(df_train: pd.DataFrame, max_features: Optional[int], min_df: int, n_estimators: int, n_jobs: int):
    if XGBClassifier is None:
        raise RuntimeError("xgboost is not installed in this Python environment")
    vectorizer = make_tfidf(max_features=max_features, min_df=min_df)
    x_train = vectorizer.fit_transform(sequence_texts(df_train))
    y_train = df_train["label"].to_numpy()
    n_pos = max(1, int(y_train.sum()))
    n_neg = max(1, int(len(y_train) - y_train.sum()))
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=SEED,
        n_jobs=n_jobs,
        scale_pos_weight=float(n_neg / n_pos),
        use_label_encoder=False,
    )
    model.fit(x_train, y_train, verbose=False)
    return vectorizer, model


def evaluate_model(
    baseline: str,
    model_obj,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    out_dir: Path,
    feature_mode: Optional[str] = None,
    is_kmer_text: bool = False,
    is_xgb: bool = False,
) -> Dict:
    ensure_dir(out_dir)
    start = time.time()
    y_val = df_val["label"].to_numpy()
    y_test = df_test["label"].to_numpy()

    if feature_mode is not None:
        x_val = basic_feature_matrix(df_val, feature_mode)
        x_test = basic_feature_matrix(df_test, feature_mode)
        probs_val, preds_val = safe_predict(model_obj, x_val)
        probs_test, preds_test = safe_predict(model_obj, x_test)
    elif is_xgb:
        vectorizer, model = model_obj
        x_val = vectorizer.transform(sequence_texts(df_val))
        x_test = vectorizer.transform(sequence_texts(df_test))
        probs_val, preds_val = safe_predict(model, x_val)
        probs_test, preds_test = safe_predict(model, x_test)
    elif is_kmer_text:
        probs_val, preds_val = safe_predict(model_obj, sequence_texts(df_val))
        probs_test, preds_test = safe_predict(model_obj, sequence_texts(df_test))
    else:
        raise ValueError("no evaluation input type was selected")

    val_metrics = compute_classification_metrics(y_val, preds_val, probs_val)
    test_metrics = compute_classification_metrics(y_test, preds_test, probs_test)

    save_df(prediction_frame(df_val, probs_val, preds_val), out_dir / "predictions_val.csv")
    save_df(prediction_frame(df_test, probs_test, preds_test), out_dir / "predictions_test.csv")
    save_json({"baseline": baseline, "val": val_metrics, "test": test_metrics}, out_dir / "metrics.json")
    plot_confusion_matrix(y_test, preds_test, out_dir / "confusion_matrix_test.png", title=f"{baseline} | Test")
    plot_roc_pr(y_test, probs_test, out_dir / "roc_curve_test.png", out_dir / "pr_curve_test.png", title_prefix=baseline)

    try:
        with open(out_dir / "model.pkl", "wb") as f:
            pickle.dump(model_obj, f)
    except Exception as exc:
        save_json({"warning": f"model pickle skipped: {exc}"}, out_dir / "model_pickle_warning.json")

    row = {
        "baseline": baseline,
        "train_n": int(len(df_train)),
        "val_n": int(len(df_val)),
        "test_n": int(len(df_test)),
        "runtime_sec": float(time.time() - start),
    }
    row.update(test_metrics)
    for key, value in val_metrics.items():
        row[f"val_{key}"] = value
    return row


def run_baseline_suite(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    suite_dir: Path,
    args: argparse.Namespace,
    benchmark_name: str,
) -> pd.DataFrame:
    rows = []
    baseline_specs = [
        ("length_only_logreg", "length"),
        ("gc_only_logreg", "gc"),
        ("length_gc_logreg", "length_gc"),
    ]

    for baseline, mode in baseline_specs:
        print(f"[Run] {benchmark_name} / {baseline}")
        out_dir = ensure_dir(suite_dir / baseline)
        model = fit_length_gc_logreg(df_train, mode)
        save_linear_coefficients(model, out_dir / "coefficients.csv")
        row = evaluate_model(
            baseline,
            model,
            df_train,
            df_val,
            df_test,
            out_dir,
            feature_mode=mode,
        )
        row["benchmark"] = benchmark_name
        rows.append(row)

    print(f"[Run] {benchmark_name} / 6mer_tfidf_logreg")
    out_dir = ensure_dir(suite_dir / "6mer_tfidf_logreg")
    model = fit_kmer_logreg(df_train, args.max_features, args.min_df)
    save_linear_coefficients(model, out_dir / "top_kmer_coefficients.csv")
    row = evaluate_model(
        "6mer_tfidf_logreg",
        model,
        df_train,
        df_val,
        df_test,
        out_dir,
        is_kmer_text=True,
    )
    row["benchmark"] = benchmark_name
    rows.append(row)

    if not args.skip_svm:
        print(f"[Run] {benchmark_name} / 6mer_tfidf_linear_svm")
        out_dir = ensure_dir(suite_dir / "6mer_tfidf_linear_svm")
        model = fit_kmer_svm(df_train, args.max_features, args.min_df)
        row = evaluate_model(
            "6mer_tfidf_linear_svm",
            model,
            df_train,
            df_val,
            df_test,
            out_dir,
            is_kmer_text=True,
        )
        row["benchmark"] = benchmark_name
        rows.append(row)

    print(f"[Run] {benchmark_name} / 6mer_tfidf_xgboost")
    out_dir = ensure_dir(suite_dir / "6mer_tfidf_xgboost")
    model_obj = fit_kmer_xgboost(df_train, args.max_features, args.min_df, args.xgb_estimators, args.n_jobs)
    vectorizer, xgb_model = model_obj
    save_xgb_importance(xgb_model, vectorizer.get_feature_names_out(), out_dir / "top_kmer_xgb_gain.csv")
    row = evaluate_model(
        "6mer_tfidf_xgboost",
        model_obj,
        df_train,
        df_val,
        df_test,
        out_dir,
        is_xgb=True,
    )
    row["benchmark"] = benchmark_name
    rows.append(row)

    summary = pd.DataFrame(rows)
    save_df(summary, suite_dir / "summary.csv", suite_dir / "summary.md")
    plot_metric_bars(summary, suite_dir / "summary_auprc.png", f"{benchmark_name}: AUPRC", metric="auprc")
    plot_metric_bars(summary, suite_dir / "summary_mcc.png", f"{benchmark_name}: MCC", metric="mcc")
    return summary


def aggregate_summary(all_rows: pd.DataFrame) -> pd.DataFrame:
    metrics = ["accuracy", "precision", "recall", "f1", "mcc", "auroc", "auprc", "ap"]
    rows = []
    if len(all_rows) == 0:
        return pd.DataFrame()
    for (benchmark, baseline), sub in all_rows.groupby(["benchmark", "baseline"]):
        row = {
            "benchmark": benchmark,
            "baseline": baseline,
            "n_runs": int(len(sub)),
        }
        for metric in metrics:
            row[f"{metric}_mean"] = float(sub[metric].mean())
            row[f"{metric}_std"] = float(sub[metric].std(ddof=0))
            row[f"{metric}_min"] = float(sub[metric].min())
            row[f"{metric}_max"] = float(sub[metric].max())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["benchmark", "auprc_mean"], ascending=[True, False])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run length/GC and 6-mer baselines.")
    parser.add_argument("--pos-file", default=str(DEFAULT_POS_FILE))
    parser.add_argument("--neg-file", default=str(DEFAULT_NEG_FILE))
    parser.add_argument("--taxonomy-map-file", default=str(DEFAULT_TAXONOMY_MAP_FILE))
    parser.add_argument("--output-root", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--top-k-species", type=int, default=10)
    parser.add_argument("--no-topk", action="store_true")
    parser.add_argument("--run-loso", action="store_true", default=True)
    parser.add_argument("--no-loso", action="store_false", dest="run_loso")
    parser.add_argument("--loso-top-k", type=int, default=10)
    parser.add_argument("--max-features", type=int, default=200000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--xgb-estimators", type=int, default=300)
    parser.add_argument("--skip-svm", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(SEED)
    root_dir = make_result_dir(args.output_root, "tabular_kmer_baselines")
    print(f"[Output] {root_dir}")

    config = vars(args).copy()
    config.update(
        {
            "seed": SEED,
            "baselines": [
                "length_only_logreg",
                "gc_only_logreg",
                "length_gc_logreg",
                "6mer_tfidf_logreg",
                "6mer_tfidf_linear_svm" if not args.skip_svm else "skipped",
                "6mer_tfidf_xgboost",
            ],
        }
    )
    save_json(config, root_dir / "config.json")

    df_all = prepare_dataset(
        root_dir,
        pos_file=args.pos_file,
        neg_file=args.neg_file,
        taxonomy_map_file=args.taxonomy_map_file,
        limit_to_topk_species=not args.no_topk,
        top_k_species=args.top_k_species,
        require_both_labels=True,
    )

    all_summaries = []
    split_result = split_three_way_by_unit(df_all, unit_col="species_key", random_state=SEED)
    df_train, df_val, df_test = split_result[:3]
    standard_dir = ensure_dir(root_dir / "standard_species_disjoint")
    save_split_files(standard_dir, df_train, df_val, df_test)
    standard_summary = run_baseline_suite(df_train, df_val, df_test, standard_dir, args, "standard_species_disjoint")
    all_summaries.append(standard_summary)

    if args.run_loso:
        loso_root = ensure_dir(root_dir / "loso_species")
        manifest = build_unit_manifest(df_all, "species_key")
        candidates = filter_species_candidates(manifest, top_k=args.loso_top_k)
        save_df(candidates, loso_root / "species_candidates.csv", loso_root / "species_candidates.md")
        loso_rows = []
        for i, row in candidates.iterrows():
            species_key = row["species_key"]
            species_label = row.get("species_label", species_key)
            print(f"[LOSO] {i + 1}/{len(candidates)} holdout={species_key}")
            run_dir = ensure_dir(loso_root / f"holdout_species_{i + 1:02d}")
            try:
                split_result = split_holdout_unit(
                    df_all,
                    holdout_unit_col="species_key",
                    holdout_value=species_key,
                    val_group_col="species_key",
                    val_frac_from_rest=OOD_VAL_FRAC_FROM_REST,
                    random_state=SEED + i,
                )
                tr, va, te = split_result[:3]
                save_split_files(run_dir, tr, va, te)
                summary = run_baseline_suite(tr, va, te, run_dir, args, "loso_species")
                summary["heldout_species"] = species_key
                summary["heldout_species_label"] = species_label
                summary["species_support_total"] = int(row["n_total"])
                summary["species_support_pos"] = int(row["n_pos"])
                summary["species_support_neg"] = int(row["n_neg"])
                save_df(summary, run_dir / "holdout_summary.csv", run_dir / "holdout_summary.md")
                loso_rows.append(summary)
            except Exception as exc:
                err = pd.DataFrame(
                    [
                        {
                            "benchmark": "loso_species",
                            "heldout_species": species_key,
                            "heldout_species_label": species_label,
                            "error": str(exc),
                        }
                    ]
                )
                save_df(err, run_dir / "error.csv", run_dir / "error.md")
                loso_rows.append(err)

        if loso_rows:
            loso_summary = pd.concat(loso_rows, ignore_index=True)
            save_df(loso_summary, loso_root / "loso_species_summary.csv", loso_root / "loso_species_summary.md")
            ok_loso = loso_summary[~loso_summary.get("accuracy", pd.Series(index=loso_summary.index)).isna()].copy()
            if len(ok_loso) > 0:
                all_summaries.append(ok_loso)

    combined = pd.concat(all_summaries, ignore_index=True) if all_summaries else pd.DataFrame()
    save_df(combined, root_dir / "all_baseline_results.csv", root_dir / "all_baseline_results.md")
    agg = aggregate_summary(combined)
    save_df(agg, root_dir / "aggregate_baseline_summary.csv", root_dir / "aggregate_baseline_summary.md")
    print("[Done] wrote:")
    print(f"  {root_dir / 'all_baseline_results.csv'}")
    print(f"  {root_dir / 'aggregate_baseline_summary.csv'}")


if __name__ == "__main__":
    main()
