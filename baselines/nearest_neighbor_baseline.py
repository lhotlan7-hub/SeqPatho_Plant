# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from baseline_common import (
    DEFAULT_NEG_FILE,
    DEFAULT_POS_FILE,
    DEFAULT_TAXONOMY_MAP_FILE,
    OOD_VAL_FRAC_FROM_REST,
    SEED,
    build_unit_manifest,
    compute_classification_metrics,
    ensure_dir,
    filter_species_candidates,
    make_result_dir,
    plot_confusion_matrix,
    plot_roc_pr,
    prepare_dataset,
    prediction_frame,
    save_df,
    save_json,
    save_split_files,
    set_seed,
    split_holdout_unit,
    split_three_way_by_unit,
    write_fasta,
)


def run_command(cmd, cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
    print("[CMD] " + " ".join(str(x) for x in cmd))
    return subprocess.run(
        [str(x) for x in cmd],
        cwd=str(cwd) if cwd else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )


def detect_backend(requested_backend: str) -> Tuple[str, Optional[str]]:
    requested_backend = requested_backend.lower()
    if requested_backend in {"mmseqs", "mmseqs2"}:
        exe = shutil.which("mmseqs")
        if exe is None:
            raise RuntimeError("requested MMseqs2 backend, but `mmseqs` was not found on PATH")
        return "mmseqs", exe
    if requested_backend == "blastn":
        exe = shutil.which("blastn")
        if exe is None:
            raise RuntimeError("requested BLASTN backend, but `blastn` was not found on PATH")
        return "blastn", exe
    if requested_backend != "auto":
        raise ValueError(f"unsupported backend: {requested_backend}")

    mmseqs = shutil.which("mmseqs")
    if mmseqs:
        return "mmseqs", mmseqs
    blastn = shutil.which("blastn")
    if blastn:
        return "blastn", blastn
    raise RuntimeError("neither `mmseqs` nor `blastn` was found on PATH")


def run_mmseqs_search(
    train_fasta: Path,
    query_fasta: Path,
    out_tsv: Path,
    tmp_dir: Path,
    threads: int,
) -> pd.DataFrame:
    ensure_dir(tmp_dir)
    cmd = [
        "mmseqs",
        "easy-search",
        query_fasta,
        train_fasta,
        out_tsv,
        tmp_dir,
        "--threads",
        str(threads),
        "--format-output",
        "query,target,pident,alnlen,evalue,bits",
    ]
    run_command(cmd)
    if not out_tsv.exists() or out_tsv.stat().st_size == 0:
        return pd.DataFrame(columns=["query_id", "target_id", "pident", "alnlen", "evalue", "bitscore"])
    hits = pd.read_csv(
        out_tsv,
        sep="\t",
        names=["query_id", "target_id", "pident", "alnlen", "evalue", "bitscore"],
    )
    return hits


def run_blastn_search(
    train_fasta: Path,
    query_fasta: Path,
    out_tsv: Path,
    db_dir: Path,
    threads: int,
    evalue: str,
    max_target_seqs: int,
) -> pd.DataFrame:
    ensure_dir(db_dir)
    db_prefix = db_dir / "train_db"
    run_command(["makeblastdb", "-in", train_fasta, "-dbtype", "nucl", "-parse_seqids", "-out", db_prefix])
    outfmt = "6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore qcovs"
    cmd = [
        "blastn",
        "-task",
        "blastn",
        "-query",
        query_fasta,
        "-db",
        db_prefix,
        "-out",
        out_tsv,
        "-outfmt",
        outfmt,
        "-evalue",
        evalue,
        "-max_target_seqs",
        str(max_target_seqs),
        "-num_threads",
        str(threads),
    ]
    run_command(cmd)
    if not out_tsv.exists() or out_tsv.stat().st_size == 0:
        return pd.DataFrame(
            columns=["query_id", "target_id", "pident", "alnlen", "evalue", "bitscore", "qcovs"]
        )
    hits = pd.read_csv(
        out_tsv,
        sep="\t",
        names=[
            "query_id",
            "target_id",
            "pident",
            "alnlen",
            "mismatch",
            "gapopen",
            "qstart",
            "qend",
            "sstart",
            "send",
            "evalue",
            "bitscore",
            "qcovs",
        ],
    )
    return hits


def best_hit_table(hits: pd.DataFrame) -> pd.DataFrame:
    if len(hits) == 0:
        return pd.DataFrame(columns=hits.columns)
    sort_cols = []
    ascending = []
    for col, asc in [("query_id", True), ("bitscore", False), ("evalue", True), ("pident", False), ("alnlen", False)]:
        if col in hits.columns:
            sort_cols.append(col)
            ascending.append(asc)
    hits = hits.sort_values(sort_cols, ascending=ascending)
    return hits.drop_duplicates(subset=["query_id"], keep="first").reset_index(drop=True)


def run_nearest_neighbor_split(
    df_train: pd.DataFrame,
    df_eval: pd.DataFrame,
    out_dir: Path,
    backend: str,
    threads: int,
    evalue: str,
    max_target_seqs: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ensure_dir(out_dir)
    train_fasta = out_dir / "train.fasta"
    query_fasta = out_dir / "query.fasta"
    write_fasta(df_train, train_fasta)
    write_fasta(df_eval, query_fasta)

    start = time.time()
    if backend == "mmseqs":
        hits = run_mmseqs_search(train_fasta, query_fasta, out_dir / "mmseqs_hits.tsv", out_dir / "mmseqs_tmp", threads)
    elif backend == "blastn":
        hits = run_blastn_search(
            train_fasta,
            query_fasta,
            out_dir / "blastn_hits.tsv",
            out_dir / "blastdb",
            threads,
            evalue,
            max_target_seqs,
        )
    else:
        raise ValueError(f"unsupported backend: {backend}")

    best = best_hit_table(hits)
    save_df(hits, out_dir / "all_hits.csv")
    save_df(best, out_dir / "best_hits.csv")

    train_label_map = dict(zip(df_train["sample_id"].astype(str), df_train["label"].astype(int)))
    prior = float(df_train["label"].mean())
    rows = []
    best_by_query = {str(row["query_id"]): row for _, row in best.iterrows()}
    for _, row in df_eval.iterrows():
        query_id = str(row["sample_id"])
        hit = best_by_query.get(query_id)
        if hit is None:
            prob = prior
            pred = int(prob >= 0.5)
            hit_info = {
                "hit_found": 0,
                "target_id": "",
                "target_label": np.nan,
                "pident": np.nan,
                "alnlen": np.nan,
                "evalue": np.nan,
                "bitscore": np.nan,
                "qcovs": np.nan,
            }
        else:
            target_id = str(hit["target_id"])
            target_label = train_label_map.get(target_id, int(prior >= 0.5))
            prob = float(target_label)
            pred = int(target_label)
            hit_info = {
                "hit_found": 1,
                "target_id": target_id,
                "target_label": int(target_label),
                "pident": float(hit.get("pident", np.nan)),
                "alnlen": float(hit.get("alnlen", np.nan)),
                "evalue": float(hit.get("evalue", np.nan)),
                "bitscore": float(hit.get("bitscore", np.nan)),
                "qcovs": float(hit.get("qcovs", np.nan)),
            }
        rows.append({"prob_pos": prob, "pred": pred, **hit_info})

    pred_extra = pd.DataFrame(rows)
    pred_df = prediction_frame(df_eval, pred_extra["prob_pos"], pred_extra["pred"], extra=pred_extra.drop(columns=["prob_pos", "pred"]))
    pred_df["backend"] = backend
    pred_df["runtime_sec_total_search"] = float(time.time() - start)
    save_df(pred_df, out_dir / "predictions.csv")
    return pred_df, best


def evaluate_nearest_neighbor(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    out_dir: Path,
    backend: str,
    args: argparse.Namespace,
    benchmark: str,
) -> Dict:
    ensure_dir(out_dir)
    val_pred, _ = run_nearest_neighbor_split(
        df_train,
        df_val,
        out_dir / "val_search",
        backend,
        args.threads,
        args.evalue,
        args.max_target_seqs,
    )
    test_pred, _ = run_nearest_neighbor_split(
        df_train,
        df_test,
        out_dir / "test_search",
        backend,
        args.threads,
        args.evalue,
        args.max_target_seqs,
    )

    val_metrics = compute_classification_metrics(val_pred["label"], val_pred["pred"], val_pred["prob_pos"])
    test_metrics = compute_classification_metrics(test_pred["label"], test_pred["pred"], test_pred["prob_pos"])
    save_json({"benchmark": benchmark, "backend": backend, "val": val_metrics, "test": test_metrics}, out_dir / "metrics.json")
    plot_confusion_matrix(
        test_pred["label"],
        test_pred["pred"],
        out_dir / "confusion_matrix_test.png",
        title=f"{backend} nearest-neighbor | Test",
    )
    plot_roc_pr(
        test_pred["label"],
        test_pred["prob_pos"],
        out_dir / "roc_curve_test.png",
        out_dir / "pr_curve_test.png",
        title_prefix=f"{backend} nearest-neighbor",
    )

    row = {
        "benchmark": benchmark,
        "baseline": f"{backend}_nearest_neighbor",
        "backend": backend,
        "train_n": int(len(df_train)),
        "val_n": int(len(df_val)),
        "test_n": int(len(df_test)),
        "test_hit_rate": float(test_pred["hit_found"].mean()),
        "val_hit_rate": float(val_pred["hit_found"].mean()),
    }
    row.update(test_metrics)
    for key, value in val_metrics.items():
        row[f"val_{key}"] = value
    return row


def aggregate_summary(df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["accuracy", "precision", "recall", "f1", "mcc", "auroc", "auprc", "ap", "test_hit_rate"]
    rows = []
    for (benchmark, baseline), sub in df.groupby(["benchmark", "baseline"]):
        row = {"benchmark": benchmark, "baseline": baseline, "n_runs": int(len(sub))}
        for metric in metrics:
            if metric in sub.columns:
                row[f"{metric}_mean"] = float(sub[metric].mean())
                row[f"{metric}_std"] = float(sub[metric].std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["benchmark", "auprc_mean"], ascending=[True, False])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MMseqs2/BLASTN nearest-neighbor baseline.")
    parser.add_argument("--pos-file", default=str(DEFAULT_POS_FILE))
    parser.add_argument("--neg-file", default=str(DEFAULT_NEG_FILE))
    parser.add_argument("--taxonomy-map-file", default=str(DEFAULT_TAXONOMY_MAP_FILE))
    parser.add_argument("--output-root", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--top-k-species", type=int, default=10)
    parser.add_argument("--no-topk", action="store_true")
    parser.add_argument("--backend", default="auto", choices=["auto", "mmseqs", "mmseqs2", "blastn"])
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--evalue", default="1e-5")
    parser.add_argument("--max-target-seqs", type=int, default=10)
    parser.add_argument("--run-loso", action="store_true", default=True)
    parser.add_argument("--no-loso", action="store_false", dest="run_loso")
    parser.add_argument("--loso-top-k", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(SEED)
    root_dir = make_result_dir(args.output_root, "nearest_neighbor_baseline")
    print(f"[Output] {root_dir}")

    backend, exe = detect_backend(args.backend)
    backend_info = {
        "requested_backend": args.backend,
        "actual_backend": backend,
        "executable": exe,
        "note": "MMseqs2 is used when available; BLASTN fallback is used otherwise.",
    }
    save_json({**vars(args), **backend_info, "seed": SEED}, root_dir / "config.json")
    save_json(backend_info, root_dir / "backend_info.json")
    print(f"[Backend] requested={args.backend} actual={backend} exe={exe}")

    df_all = prepare_dataset(
        root_dir,
        pos_file=args.pos_file,
        neg_file=args.neg_file,
        taxonomy_map_file=args.taxonomy_map_file,
        limit_to_topk_species=not args.no_topk,
        top_k_species=args.top_k_species,
        require_both_labels=True,
    )

    rows = []
    split_result = split_three_way_by_unit(df_all, unit_col="species_key", random_state=SEED)
    df_train, df_val, df_test = split_result[:3]
    standard_dir = ensure_dir(root_dir / "standard_species_disjoint")
    save_split_files(standard_dir, df_train, df_val, df_test)
    rows.append(
        evaluate_nearest_neighbor(
            df_train,
            df_val,
            df_test,
            standard_dir,
            backend,
            args,
            benchmark="standard_species_disjoint",
        )
    )

    if args.run_loso:
        loso_root = ensure_dir(root_dir / "loso_species")
        manifest = build_unit_manifest(df_all, "species_key")
        candidates = filter_species_candidates(manifest, top_k=args.loso_top_k)
        save_df(candidates, loso_root / "species_candidates.csv", loso_root / "species_candidates.md")
        for i, row in candidates.iterrows():
            species_key = row["species_key"]
            species_label = row.get("species_label", species_key)
            run_dir = ensure_dir(loso_root / f"holdout_species_{i + 1:02d}")
            print(f"[LOSO] {i + 1}/{len(candidates)} holdout={species_key}")
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
                result = evaluate_nearest_neighbor(tr, va, te, run_dir, backend, args, benchmark="loso_species")
                result.update(
                    {
                        "heldout_species": species_key,
                        "heldout_species_label": species_label,
                        "species_support_total": int(row["n_total"]),
                        "species_support_pos": int(row["n_pos"]),
                        "species_support_neg": int(row["n_neg"]),
                    }
                )
                rows.append(result)
            except Exception as exc:
                err = {
                    "benchmark": "loso_species",
                    "baseline": f"{backend}_nearest_neighbor",
                    "heldout_species": species_key,
                    "heldout_species_label": species_label,
                    "error": str(exc),
                }
                rows.append(err)
                save_json(err, run_dir / "error.json")

    result_df = pd.DataFrame(rows)
    save_df(result_df, root_dir / "nearest_neighbor_results.csv", root_dir / "nearest_neighbor_results.md")
    ok = result_df[~result_df.get("accuracy", pd.Series(index=result_df.index)).isna()].copy()
    if len(ok) > 0:
        agg = aggregate_summary(ok)
        save_df(agg, root_dir / "aggregate_nearest_neighbor_summary.csv", root_dir / "aggregate_nearest_neighbor_summary.md")
    print("[Done] wrote:")
    print(f"  {root_dir / 'nearest_neighbor_results.csv'}")


if __name__ == "__main__":
    main()
