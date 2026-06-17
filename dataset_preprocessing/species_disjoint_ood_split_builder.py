#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


DEFAULT_ASSIGNMENT_SEED = 4740
SPLIT_NAMES = ("train", "validation", "test")
SPLIT_FRACTIONS = np.array([0.70, 0.15, 0.15], dtype=float)


def sha1_text(value: str) -> str:
    return hashlib.sha1(str(value).encode("utf-8", errors="ignore")).hexdigest()


def read_taxid_set(path: Path) -> set[int]:
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    return set(pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna().astype(int))


def normalize_species_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["tax_id"] = pd.to_numeric(df["tax_id"], errors="coerce").astype("Int64")
    df["Species_Name"] = df["Species_Name"].astype(str).str.strip()
    df["gene_sequence"] = df["gene_sequence"].astype(str).str.strip().str.upper()
    df = df[df["tax_id"].notna()]
    df = df[df["Species_Name"].ne("")]
    df = df[df["gene_sequence"].ne("")]
    df["species_key"] = df["tax_id"].astype(str) + "|" + df["Species_Name"]
    return df


def load_benchmark(root: Path) -> pd.DataFrame:
    pos_path = root / "Dataset" / "genes_dedup.csv"
    neg_path = root / "Dataset" / "NONgenes_dedup2.csv"
    neg_only_taxids_path = root / "Dataset" / "taxid_nonpatho_only.csv"

    pos = pd.read_csv(pos_path, low_memory=False)
    neg = pd.read_csv(neg_path, low_memory=False)

    pos = normalize_species_frame(pos[["tax_id", "GeneID", "Species_Name", "gene_sequence"]])
    neg = normalize_species_frame(neg[["tax_id", "GeneID", "Species_Name", "gene_sequence"]])

    neg_only_taxids = read_taxid_set(neg_only_taxids_path)
    if neg_only_taxids:
        neg = neg[~neg["tax_id"].astype(int).isin(neg_only_taxids)].copy()

    pos["label"] = 1
    pos["source_table"] = "positive"
    pos["raw_row_id"] = np.arange(len(pos), dtype=np.int64)
    neg["label"] = 0
    neg["source_table"] = "unlabeled_control"
    neg["raw_row_id"] = np.arange(len(neg), dtype=np.int64)

    df = pd.concat([pos, neg], ignore_index=True)
    df["record_id"] = [
        f"{src}_{idx}_{sha1_text(seq)[:10]}"
        for src, idx, seq in zip(df["source_table"], df["raw_row_id"], df["gene_sequence"])
    ]
    df["seq_len"] = df["gene_sequence"].str.len().astype(int)
    df["seq_sha1"] = df["gene_sequence"].map(sha1_text)
    return df


def split_group_counts(n_groups: int) -> Tuple[int, int, int]:
    n_train = int(round(n_groups * SPLIT_FRACTIONS[0]))
    n_val = int(round(n_groups * SPLIT_FRACTIONS[1]))
    n_test = n_groups - n_train - n_val
    return n_train, n_val, n_test


def assign_species(df: pd.DataFrame, assignment_seed: int) -> Dict[str, str]:
    species = (
        df.groupby("species_key", sort=False)
        .agg(n=("label", "size"), pos=("label", "sum"))
        .reset_index()
    )

    n_train, n_val, _ = split_group_counts(len(species))
    rng = np.random.default_rng(assignment_seed)
    perm = rng.permutation(len(species))
    split_index = {
        "train": perm[:n_train],
        "validation": perm[n_train : n_train + n_val],
        "test": perm[n_train + n_val :],
    }

    assignment: Dict[str, str] = {}
    for split_name, idx in split_index.items():
        for species_key in species.iloc[idx]["species_key"]:
            assignment[str(species_key)] = split_name
    return assignment


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split_name in SPLIT_NAMES:
        sub = df[df["split"].eq(split_name)]
        n = int(len(sub))
        pos = int(sub["label"].sum())
        neg = int((sub["label"] == 0).sum())
        rows.append(
            {
                "split": split_name,
                "n": n,
                "positive": pos,
                "unlabeled_control": neg,
                "positive_rate": pos / n if n else 0.0,
                "source_species": int(sub["species_key"].nunique()),
            }
        )

    n = int(len(df))
    pos = int(df["label"].sum())
    neg = int((df["label"] == 0).sum())
    rows.append(
        {
            "split": "overall",
            "n": n,
            "positive": pos,
            "unlabeled_control": neg,
            "positive_rate": pos / n if n else 0.0,
            "source_species": int(df["species_key"].nunique()),
        }
    )
    return pd.DataFrame(rows)


def build_leakage_report(df: pd.DataFrame) -> dict:
    species_by_split = {
        split_name: set(df.loc[df["split"].eq(split_name), "species_key"])
        for split_name in SPLIT_NAMES
    }
    overlaps = {}
    for i, left in enumerate(SPLIT_NAMES):
        for right in SPLIT_NAMES[i + 1 :]:
            key = f"{left}_vs_{right}"
            overlaps[key] = sorted(species_by_split[left] & species_by_split[right])
    return {
        "species_disjoint": all(len(v) == 0 for v in overlaps.values()),
        "overlap_counts": {k: len(v) for k, v in overlaps.items()},
        "overlaps": overlaps,
    }


def write_outputs(df: pd.DataFrame, outdir: Path, write_sequence_splits: bool) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    columns = [
        "record_id",
        "source_table",
        "raw_row_id",
        "label",
        "split",
        "tax_id",
        "Species_Name",
        "species_key",
        "GeneID",
        "seq_len",
        "seq_sha1",
    ]
    df[columns].to_csv(outdir / "dataset_main_species_disjoint_ood_manifest.csv", index=False)

    summary = compute_summary(df)
    summary.to_csv(outdir / "dataset_main_species_disjoint_ood_summary.csv", index=False)

    species_assignment = (
        df.groupby(["split", "species_key"], sort=False)
        .agg(
            tax_id=("tax_id", "first"),
            Species_Name=("Species_Name", "first"),
            n=("label", "size"),
            positive=("label", "sum"),
        )
        .reset_index()
    )
    species_assignment["unlabeled_control"] = species_assignment["n"] - species_assignment["positive"]
    species_assignment["positive_rate"] = species_assignment["positive"] / species_assignment["n"]
    species_assignment.to_csv(outdir / "species_split_assignment.csv", index=False)

    leakage_report = build_leakage_report(df)
    (outdir / "leakage_check.json").write_text(
        json.dumps(leakage_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if write_sequence_splits:
        split_columns = [
            "record_id",
            "label",
            "tax_id",
            "Species_Name",
            "species_key",
            "GeneID",
            "seq_len",
            "seq_sha1",
            "gene_sequence",
        ]
        for split_name in SPLIT_NAMES:
            df.loc[df["split"].eq(split_name), split_columns].to_csv(
                outdir / f"{split_name}.csv", index=False
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a 70/15/15 species-disjoint OOD split for Dataset_Main."
    )
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument(
        "--outdir",
        default="Dataset/species_disjoint_ood_154023",
        help="Output directory relative to --root unless absolute.",
    )
    parser.add_argument("--assignment-seed", type=int, default=DEFAULT_ASSIGNMENT_SEED)
    parser.add_argument(
        "--write-sequence-splits",
        action="store_true",
        help="Also write train.csv, validation.csv, and test.csv with sequences.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = root / outdir

    df = load_benchmark(root)
    assignment = assign_species(df, args.assignment_seed)
    df["split"] = df["species_key"].map(assignment)
    if df["split"].isna().any():
        raise RuntimeError("Some records were not assigned to a split.")

    write_outputs(df, outdir, args.write_sequence_splits)

    config = {
        "benchmark_records": int(len(df)),
        "positive": int(df["label"].sum()),
        "unlabeled_control": int((df["label"] == 0).sum()),
        "source_species": int(df["species_key"].nunique()),
        "split_fractions": dict(zip(SPLIT_NAMES, SPLIT_FRACTIONS.tolist())),
        "split_unit": "species_key = tax_id|Species_Name",
        "assignment_seed": int(args.assignment_seed),
        "excluded_negative_taxids": sorted(read_taxid_set(root / "Dataset" / "taxid_nonpatho_only.csv")),
        "write_sequence_splits": bool(args.write_sequence_splits),
    }
    (outdir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    print(compute_summary(df).to_string(index=False))
    print(f"Output: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
