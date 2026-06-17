#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


DNA_ALPHABET = set("ACGTN")
RC_TABLE = str.maketrans("ACGTNacgtn", "TGCANtgcan")
PLACEHOLDER_VALUES = {"", "-", ".", "na", "n/a", "nan", "none", "null", "unknown"}


SEQ_COL_CANDIDATES = [
    "gene_sequence",
    "sequence",
    "nucleotide_sequence",
    "DNA_sequence",
    "NT sequence #no EMBL#",
]

TAX_ID_COL_CANDIDATES = [
    "tax_id",
    "#tax_id",
    "taxid",
    "TaxID",
    "NCBI_tax_id",
    "Pathogen_NCBI_species_Taxonomy ID",
    "Pathogen ID",
]

SPECIES_COL_CANDIDATES = [
    "Species_Name",
    "species_name",
    "species",
    "source_species",
    "organism",
    "Pathogen_species",
    "Pathogen species",
]

GENE_ID_COL_CANDIDATES = [
    "GeneID",
    "gene_id",
    "Gene_ID",
    "Gene ID",
    "GeneLocusID",
]

PROTEIN_ACC_COL_CANDIDATES = [
    "protein_accession.version",
    "protein_accession",
    "protein accession",
    "ProteinID",
    "Protein ID",
]

RNA_ACC_COL_CANDIDATES = [
    "RNA_nucleotide_accession.version",
    "RNA_accession.version",
    "rna_accession",
    "RNA accession",
]

GENOMIC_ACC_COL_CANDIDATES = [
    "genomic_nucleotide_accession.version",
    "genomic_accession.version",
    "genomic_accession",
    "genomic accession",
]

GENOMIC_START_COL_CANDIDATES = [
    "start_position_on_the_genomic_accession",
    "genomic_start",
    "start",
]

GENOMIC_END_COL_CANDIDATES = [
    "end_position_on_the_genomic_accession",
    "genomic_end",
    "end",
]

PHENOTYPE_COL_CANDIDATES = [
    "phenotype",
    "Phenotype",
    "pathogenicity_phenotype",
    "mutant_phenotype",
    "PHI phenotype",
    "phi_phenotype",
    "status",
    "Phenotype_of_mutant",
    "Mutant Phenotype",
]

MANIFEST_COLUMNS = [
    "record_id",
    "source_table",
    "raw_row_id",
    "label",
    "split",
    "protocol",
    "fold_id",
    "heldout_species_key",
    "source_tax_id",
    "source_species",
    "source_species_key",
    "main_group_id",
    "clean_len",
    "canon_hash",
    "clean_seq",
]


def clean_sequence(seq: object) -> str:
    """Uppercase sequence and replace non-ACGTN characters with N."""
    if pd.isna(seq):
        return ""
    s = re.sub(r"\s+", "", str(seq).upper().strip())
    if not s:
        return ""
    return "".join(ch if ch in DNA_ALPHABET else "N" for ch in s)


def rev_comp(seq: str) -> str:
    return seq.translate(RC_TABLE)[::-1].upper()


def canon_rc(seq: str) -> str:
    rc = rev_comp(seq)
    return min(seq, rc)


def sha1_text(x: str) -> str:
    return hashlib.sha1(x.encode("utf-8")).hexdigest()


def safe_str(x: object) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in PLACEHOLDER_VALUES:
        return ""
    return s


def safe_tax_id(x: object) -> str:
    s = safe_str(x)
    if not s:
        return ""
    if re.fullmatch(r"\d+(\.0+)?", s):
        return s.split(".", 1)[0]
    return s


def first_existing_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    stripped_lower = {str(c).strip().lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        key = c.strip().lower()
        if key in stripped_lower:
            return stripped_lower[key]
    return None


def require_col(df: pd.DataFrame, candidates: Sequence[str], role: str) -> str:
    col = first_existing_col(df, candidates)
    if col is None:
        raise ValueError(f"Cannot find required {role} column. Tried: {list(candidates)}")
    return col


def load_table(path: Path, label: int, source_table: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df = df.copy()
    df["label"] = int(label)
    df["source_table"] = source_table
    df["raw_row_id"] = np.arange(len(df), dtype=np.int64)
    return df


def standardize_records(df: pd.DataFrame, min_len: int = 50) -> pd.DataFrame:
    seq_col = require_col(df, SEQ_COL_CANDIDATES, "sequence")
    tax_col = first_existing_col(df, TAX_ID_COL_CANDIDATES)
    species_col = first_existing_col(df, SPECIES_COL_CANDIDATES)

    out = df.copy()
    out["clean_seq"] = out[seq_col].map(clean_sequence)
    out["clean_len"] = out["clean_seq"].str.len()
    out = out[out["clean_len"] >= int(min_len)].copy()

    out["canon_seq"] = out["clean_seq"].map(canon_rc)
    out["canon_hash"] = out["canon_seq"].map(sha1_text)

    out["source_tax_id"] = out[tax_col].map(safe_tax_id) if tax_col else ""
    out["source_species"] = out[species_col].map(safe_str) if species_col else ""

    out["source_species_key"] = np.where(
        out["source_tax_id"].astype(str).str.len() > 0,
        out["source_tax_id"].astype(str),
        out["source_species"].astype(str),
    )
    out["source_species_key"] = np.where(
        out["source_species_key"].astype(str).str.len() > 0,
        out["source_species_key"],
        "seqhash:" + out["canon_hash"].astype(str),
    )

    out["record_id"] = (
        out["source_table"].astype(str)
        + "_"
        + out["raw_row_id"].astype(str)
        + "_"
        + out["canon_hash"].str[:10]
    )
    return out


def make_identifier_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    gene_col = first_existing_col(out, GENE_ID_COL_CANDIDATES)
    prot_col = first_existing_col(out, PROTEIN_ACC_COL_CANDIDATES)
    rna_col = first_existing_col(out, RNA_ACC_COL_CANDIDATES)
    gen_col = first_existing_col(out, GENOMIC_ACC_COL_CANDIDATES)
    start_col = first_existing_col(out, GENOMIC_START_COL_CANDIDATES)
    end_col = first_existing_col(out, GENOMIC_END_COL_CANDIDATES)

    out["key_gene_id"] = out[gene_col].map(safe_str) if gene_col else ""
    out["key_protein_acc"] = out[prot_col].map(safe_str) if prot_col else ""
    out["key_rna_acc"] = out[rna_col].map(safe_str) if rna_col else ""
    out["key_genomic_acc"] = out[gen_col].map(safe_str) if gen_col else ""

    if gen_col and start_col and end_col:
        gen = out[gen_col].map(safe_str)
        start = out[start_col].map(safe_str)
        end = out[end_col].map(safe_str)
        coord = gen + ":" + start + "-" + end
        out["key_genomic_coord"] = np.where((gen != "") & (start != "") & (end != ""), coord, "")
    else:
        out["key_genomic_coord"] = ""

    return out


def remove_control_overlap_with_positives(positives: pd.DataFrame, controls: pd.DataFrame) -> pd.DataFrame:
    positives = make_identifier_columns(positives)
    controls = make_identifier_columns(controls)
    overlap_cols = ["canon_hash", "key_gene_id", "key_protein_acc", "key_rna_acc", "key_genomic_coord"]

    positive_sets: Dict[str, set] = {}
    for col in overlap_cols:
        positive_sets[col] = set(v for v in positives[col].astype(str).tolist() if v)

    keep = np.ones(len(controls), dtype=bool)
    for col, values in positive_sets.items():
        if values:
            keep &= ~controls[col].astype(str).isin(values).to_numpy()

    print(f"[overlap] removed controls overlapping positives: {int((~keep).sum())}")
    return controls.loc[keep].copy()


def collapse_duplicates_and_conflicts(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse exact/reverse-complement duplicates by canon_hash; drop conflicting labels."""
    rows = []
    conflict_hashes = 0
    for _, group in df.groupby("canon_hash", sort=False):
        if group["label"].astype(int).nunique() > 1:
            conflict_hashes += 1
            continue
        rows.append(group.iloc[0])
    out = pd.DataFrame(rows).reset_index(drop=True)
    print(f"[dedup] input={len(df)}, retained={len(out)}, conflict_hashes={conflict_hashes}")
    return out


def make_component_key(df: pd.DataFrame) -> pd.Series:
    gen_col = first_existing_col(df, GENOMIC_ACC_COL_CANDIDATES)
    if gen_col:
        comp = df[gen_col].map(safe_str)
    else:
        comp = pd.Series([""] * len(df), index=df.index)

    group = []
    for species_key, component in zip(df["source_species_key"].astype(str), comp.astype(str)):
        group.append(f"{species_key}|{component}" if component else species_key)
    return pd.Series(group, index=df.index)


def prepare_main_dataset(positive_csv: Path, control_csv: Path, min_len: int) -> pd.DataFrame:
    positives = standardize_records(load_table(positive_csv, 1, "phi_positive"), min_len=min_len)
    controls = standardize_records(load_table(control_csv, 0, "unlabeled_control"), min_len=min_len)
    controls = remove_control_overlap_with_positives(positives, controls)

    df = pd.concat([positives, controls], ignore_index=True)
    df = collapse_duplicates_and_conflicts(df)
    df["main_group_id"] = make_component_key(df)
    return df.reset_index(drop=True)


def validate_no_group_overlap(df: pd.DataFrame, group_col: str, split_col: str) -> None:
    split_groups = {
        split: set(group[group_col].astype(str).tolist())
        for split, group in df.groupby(split_col, dropna=False)
    }
    splits = list(split_groups)
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            inter = split_groups[splits[i]] & split_groups[splits[j]]
            if inter:
                raise RuntimeError(
                    f"Group leakage between {splits[i]} and {splits[j]} in {group_col}: {len(inter)} groups"
                )


def validate_no_canon_overlap(df: pd.DataFrame, split_col: str) -> None:
    split_hashes = {
        split: set(group["canon_hash"].astype(str).tolist())
        for split, group in df.groupby(split_col, dropna=False)
    }
    splits = list(split_hashes)
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            inter = split_hashes[splits[i]] & split_hashes[splits[j]]
            if inter:
                raise RuntimeError(
                    f"Canonical-sequence leakage between {splits[i]} and {splits[j]}: {len(inter)} sequences"
                )


def validate_split_label_support(
    df: pd.DataFrame,
    required_splits: Sequence[str],
    split_col: str = "split",
    require_both_labels: bool = True,
) -> None:
    for split in required_splits:
        sub = df[df[split_col] == split]
        if len(sub) == 0:
            raise RuntimeError(f"Split {split} is empty")
        if require_both_labels and sub["label"].astype(int).nunique() < 2:
            raise RuntimeError(f"Split {split} does not contain both labels")


def group_random_split_once(
    df: pd.DataFrame,
    group_col: str,
    seed: int,
    train_size: float,
    val_size: float,
    test_size: float,
) -> pd.DataFrame:
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6

    out = df.copy()
    groups = out[group_col]
    gss1 = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=seed)
    train_idx, tmp_idx = next(gss1.split(out, out["label"], groups))

    tmp = out.iloc[tmp_idx].copy()
    val_frac_of_tmp = val_size / (val_size + test_size)
    gss2 = GroupShuffleSplit(n_splits=1, train_size=val_frac_of_tmp, random_state=seed + 1)
    val_rel_idx, test_rel_idx = next(gss2.split(tmp, tmp["label"], tmp[group_col]))

    out["split"] = "unassigned"
    out.iloc[train_idx, out.columns.get_loc("split")] = "train"
    out.iloc[tmp_idx[val_rel_idx], out.columns.get_loc("split")] = "val"
    out.iloc[tmp_idx[test_rel_idx], out.columns.get_loc("split")] = "test"
    return out


def group_random_split(
    df: pd.DataFrame,
    group_col: str,
    seed: int = 42,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    max_attempts: int = 100,
) -> pd.DataFrame:
    last_error: Optional[Exception] = None
    for offset in range(max_attempts):
        try:
            out = group_random_split_once(df, group_col, seed + offset, train_size, val_size, test_size)
            validate_no_group_overlap(out, group_col, "split")
            validate_no_canon_overlap(out, "split")
            validate_split_label_support(out, ["train", "val", "test"])
            return out
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Could not create valid group random split after {max_attempts} attempts: {last_error}")


def greedy_group_split_once(
    df: pd.DataFrame,
    group_col: str,
    seed: int,
    ratios: Tuple[float, ...],
    split_names: Tuple[str, ...],
) -> pd.DataFrame:
    if len(ratios) != len(split_names):
        raise ValueError("ratios and split_names must have the same length")
    ratios_arr = np.asarray(ratios, dtype=float)
    if not np.isclose(ratios_arr.sum(), 1.0):
        ratios_arr = ratios_arr / ratios_arr.sum()

    rng = np.random.default_rng(seed)
    stats = df.groupby(group_col).agg(total=("label", "size"), pos=("label", "sum")).reset_index()
    stats["neg"] = stats["total"] - stats["pos"]
    stats["jitter"] = rng.random(len(stats))
    stats = stats.sort_values(["total", "jitter"], ascending=[False, True]).reset_index(drop=True)

    target_total = ratios_arr * len(df)
    target_pos = ratios_arr * df["label"].sum()
    target_neg = ratios_arr * (len(df) - df["label"].sum())
    cur_total = np.zeros(len(ratios_arr))
    cur_pos = np.zeros(len(ratios_arr))
    cur_neg = np.zeros(len(ratios_arr))
    assignment = {}

    for _, row in stats.iterrows():
        best_k = 0
        best_loss = np.inf
        for k in range(len(ratios_arr)):
            new_total = cur_total.copy()
            new_pos = cur_pos.copy()
            new_neg = cur_neg.copy()
            new_total[k] += row["total"]
            new_pos[k] += row["pos"]
            new_neg[k] += row["neg"]
            loss = (
                ((new_total - target_total) / np.maximum(target_total, 1.0)) ** 2
                + ((new_pos - target_pos) / np.maximum(target_pos, 1.0)) ** 2
                + ((new_neg - target_neg) / np.maximum(target_neg, 1.0)) ** 2
            ).sum()
            if loss < best_loss:
                best_loss = loss
                best_k = k
        assignment[row[group_col]] = split_names[best_k]
        cur_total[best_k] += row["total"]
        cur_pos[best_k] += row["pos"]
        cur_neg[best_k] += row["neg"]

    out = df.copy()
    out["split"] = out[group_col].map(assignment)
    return out


def greedy_group_split(
    df: pd.DataFrame,
    group_col: str,
    seed: int,
    ratios: Tuple[float, ...],
    split_names: Tuple[str, ...],
    required_splits: Sequence[str],
    max_attempts: int = 100,
) -> pd.DataFrame:
    last_error: Optional[Exception] = None
    for offset in range(max_attempts):
        try:
            out = greedy_group_split_once(df, group_col, seed + offset, ratios, split_names)
            validate_no_group_overlap(out, group_col, "split")
            validate_no_canon_overlap(out, "split")
            validate_split_label_support(out, required_splits)
            return out
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Could not create valid greedy group split after {max_attempts} attempts: {last_error}")


def make_loso_registry(
    df: pd.DataFrame,
    species_col: str = "source_species_key",
    min_total: int = 30,
    min_pos: int = 5,
    min_neg: int = 5,
) -> pd.DataFrame:
    stats = df.groupby(species_col).agg(total=("label", "size"), pos=("label", "sum")).reset_index()
    stats["neg"] = stats["total"] - stats["pos"]

    species_names = (
        df.groupby(species_col)["source_species"]
        .agg(lambda s: next((x for x in s.astype(str).tolist() if x), ""))
        .reset_index()
        .rename(columns={"source_species": "heldout_species_name"})
    )

    registry = stats.merge(species_names, on=species_col, how="left")
    registry = registry[
        (registry["total"] >= min_total)
        & (registry["pos"] >= min_pos)
        & (registry["neg"] >= min_neg)
    ].copy()
    registry = registry.sort_values(["total", species_col], ascending=[False, True]).reset_index(drop=True)
    registry["fold_id"] = np.arange(len(registry), dtype=int)
    registry["heldout_species_key"] = registry[species_col].astype(str)
    registry["protocol"] = "source_pathogen_loso"
    registry["manifest_type"] = "compact_loso_fold_registry"
    registry = registry[
        [
            "fold_id",
            "heldout_species_key",
            "heldout_species_name",
            "total",
            "pos",
            "neg",
            "protocol",
            "manifest_type",
        ]
    ]
    print(f"[LOSO] eligible held-out source pathogen species: {len(registry)}")
    return registry


def materialize_loso_fold_from_base(
    base_df: pd.DataFrame,
    heldout_species_key: str,
    fold_id: int,
    seed: int = 42,
) -> pd.DataFrame:
    fold_df = base_df.copy()
    fold_df["heldout_species_key"] = str(heldout_species_key)
    fold_df["fold_id"] = int(fold_id)
    is_test = fold_df["source_species_key"].astype(str) == str(heldout_species_key)

    rest = fold_df.loc[~is_test].copy()
    rest = greedy_group_split(
        rest,
        group_col="source_species_key",
        seed=seed + int(fold_id),
        ratios=(0.85, 0.15),
        split_names=("train", "val"),
        required_splits=("train", "val"),
    )

    fold_df["split"] = "unused"
    fold_df.loc[is_test, "split"] = "test"
    fold_df.loc[rest.index, "split"] = rest["split"].values
    fold_df = fold_df[fold_df["split"].isin(["train", "val", "test"])].copy()
    validate_split_label_support(fold_df, ["train", "val", "test"])
    assert not (
        fold_df.loc[fold_df["split"].isin(["train", "val"]), "source_species_key"].astype(str)
        == str(heldout_species_key)
    ).any()
    validate_no_canon_overlap(fold_df, "split")
    return fold_df


def filter_unaffected_pathogenicity(df: pd.DataFrame) -> pd.DataFrame:
    phenotype_col = first_existing_col(df, PHENOTYPE_COL_CANDIDATES)
    if phenotype_col is None:
        print("[hard-negative] no phenotype column found; assuming input is already filtered.")
        return df.copy()
    mask = df[phenotype_col].astype(str).str.lower().str.contains(
        r"unaffected\s+pathogenicity", regex=True, na=False
    )
    out = df.loc[mask].copy()
    print(f"[hard-negative] unaffected pathogenicity rows retained: {len(out)} / {len(df)}")
    return out


def make_empty_manifest(protocol: str = "") -> pd.DataFrame:
    out = pd.DataFrame(columns=MANIFEST_COLUMNS)
    if protocol:
        out["protocol"] = out.get("protocol", pd.Series(dtype=str))
    return out


def make_hard_negative_eval_manifest(
    main_split: pd.DataFrame,
    hard_negative_csv: Path,
    min_len: int = 50,
) -> pd.DataFrame:
    hn_raw = pd.read_csv(hard_negative_csv, low_memory=False)
    hn_raw = filter_unaffected_pathogenicity(hn_raw)
    if len(hn_raw) == 0:
        return make_empty_manifest("hard_negative_unaffected_pathogenicity")

    hn_raw = hn_raw.copy()
    hn_raw["label"] = 0
    hn_raw["source_table"] = "phi_unaffected_pathogenicity"
    hn_raw["raw_row_id"] = np.arange(len(hn_raw), dtype=np.int64)
    hn = standardize_records(hn_raw, min_len=min_len)
    if len(hn) == 0:
        print("[hard-negative] no unaffected records with usable sequence after length filtering.")
        return make_empty_manifest("hard_negative_unaffected_pathogenicity")

    hn = make_identifier_columns(hn)
    pos_all = make_identifier_columns(main_split[main_split["label"] == 1].copy())

    positive_sets = {
        col: set(v for v in pos_all[col].astype(str).tolist() if v)
        for col in ["canon_hash", "key_gene_id", "key_protein_acc", "key_rna_acc", "key_genomic_coord"]
    }
    keep = np.ones(len(hn), dtype=bool)
    for col, values in positive_sets.items():
        if values:
            keep &= ~hn[col].astype(str).isin(values).to_numpy()
    hn = hn.loc[keep].copy()
    hn = collapse_duplicates_and_conflicts(hn)

    test_pos = main_split[(main_split["split"] == "test") & (main_split["label"] == 1)].copy()
    test_pos["eval_source"] = "main_test_phi_positive"
    hn["eval_source"] = "phi_unaffected_pathogenicity_hard_negative"

    eval_df = pd.concat([test_pos, hn], ignore_index=True)
    eval_df["split"] = "test"
    eval_df["protocol"] = "hard_negative_unaffected_pathogenicity"
    eval_df = collapse_duplicates_and_conflicts(eval_df)
    return eval_df


def save_manifest(df: pd.DataFrame, path: Path) -> None:
    out = df.copy()
    for col in MANIFEST_COLUMNS:
        if col not in out.columns:
            out[col] = ""
    out = out[MANIFEST_COLUMNS]
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(f"[write] {path} rows={len(out)}")


def summarize_split(df: pd.DataFrame, split_col: str = "split") -> pd.DataFrame:
    if len(df) == 0:
        return pd.DataFrame(columns=[split_col, "n", "pos", "neg", "pos_rate", "species_n"])
    summary = (
        df.groupby(split_col)
        .agg(n=("label", "size"), pos=("label", "sum"), species_n=("source_species_key", "nunique"))
        .reset_index()
    )
    summary["neg"] = summary["n"] - summary["pos"]
    summary["pos_rate"] = summary["pos"] / summary["n"]
    return summary[[split_col, "n", "pos", "neg", "pos_rate", "species_n"]]


def write_status(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[write] {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate source-pathogen validation manifests.")
    parser.add_argument("--positive_csv", type=Path, required=True)
    parser.add_argument("--control_csv", type=Path, required=True)
    parser.add_argument("--hard_negative_csv", type=Path, default=None)
    parser.add_argument("--outdir", type=Path, default=Path("outputs"))
    parser.add_argument("--min_len", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--loso_min_total", type=int, default=30)
    parser.add_argument("--loso_min_pos", type=int, default=5)
    parser.add_argument("--loso_min_neg", type=int, default=5)
    parser.add_argument(
        "--loso_manifest_mode",
        choices=["compact", "full"],
        default="compact",
        help="compact writes only the fold registry; full expands all train/val/test rows for every fold.",
    )
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    df = prepare_main_dataset(args.positive_csv, args.control_csv, min_len=args.min_len)

    source_map = (
        df.groupby("source_species_key")
        .agg(
            source_tax_id=("source_tax_id", lambda s: next((x for x in s.astype(str).tolist() if x), "")),
            source_species=("source_species", lambda s: next((x for x in s.astype(str).tolist() if x), "")),
            n=("label", "size"),
            pos=("label", "sum"),
        )
        .reset_index()
    )
    source_map["neg"] = source_map["n"] - source_map["pos"]
    source_map.to_csv(args.outdir / "source_species_key_map.csv", index=False)

    main_split = group_random_split(df, group_col="main_group_id", seed=args.seed)
    main_split["protocol"] = "locked_group_aware_random_split"
    save_manifest(main_split, args.outdir / "locked_main_split_manifest.csv")
    summarize_split(main_split).to_csv(args.outdir / "locked_main_split_summary.csv", index=False)

    ood_split = greedy_group_split(
        df,
        group_col="source_species_key",
        seed=args.seed,
        ratios=(0.70, 0.15, 0.15),
        split_names=("train", "val", "test"),
        required_splits=("train", "val", "test"),
    )
    ood_split["protocol"] = "source_pathogen_species_disjoint_ood"
    save_manifest(ood_split, args.outdir / "source_species_ood_manifest.csv")
    summarize_split(ood_split).to_csv(args.outdir / "source_species_ood_summary.csv", index=False)

    loso_registry = make_loso_registry(
        df,
        species_col="source_species_key",
        min_total=args.loso_min_total,
        min_pos=args.loso_min_pos,
        min_neg=args.loso_min_neg,
    )
    if len(loso_registry) == 0:
        raise RuntimeError("No eligible species for LOSO. Lower min_total/min_pos/min_neg thresholds.")

    if args.loso_manifest_mode == "compact":
        loso_registry.to_csv(args.outdir / "source_species_loso_manifest.csv", index=False)
        print(f"[write] {args.outdir / 'source_species_loso_manifest.csv'} rows={len(loso_registry)}")
    else:
        rows: List[pd.DataFrame] = []
        for _, row in loso_registry.iterrows():
            rows.append(
                materialize_loso_fold_from_base(
                    df,
                    heldout_species_key=str(row["heldout_species_key"]),
                    fold_id=int(row["fold_id"]),
                    seed=args.seed,
                )
            )
        loso_full = pd.concat(rows, ignore_index=True)
        loso_full["protocol"] = "source_pathogen_loso"
        save_manifest(loso_full, args.outdir / "source_species_loso_manifest.csv")

    loso_registry.rename(
        columns={"total": "n", "heldout_species_name": "source_species"},
    ).to_csv(args.outdir / "source_species_loso_support.csv", index=False)

    hard_status: Dict[str, object] = {"status": "not_requested", "path": None}
    if args.hard_negative_csv is not None and args.hard_negative_csv.exists():
        hard_eval = make_hard_negative_eval_manifest(
            main_split=main_split,
            hard_negative_csv=args.hard_negative_csv,
            min_len=args.min_len,
        )
        save_manifest(hard_eval, args.outdir / "hard_negative_eval_manifest.csv")
        summarize_split(hard_eval).to_csv(args.outdir / "hard_negative_eval_summary.csv", index=False)
        hard_status = {
            "status": "generated" if len(hard_eval) > 0 else "empty_after_filtering",
            "path": str(args.hard_negative_csv),
            "rows": int(len(hard_eval)),
        }
    else:
        save_manifest(make_empty_manifest("hard_negative_unaffected_pathogenicity"), args.outdir / "hard_negative_eval_manifest.csv")
        pd.DataFrame(columns=["split", "n", "pos", "neg", "pos_rate", "species_n"]).to_csv(
            args.outdir / "hard_negative_eval_summary.csv", index=False
        )
        if args.hard_negative_csv is not None:
            hard_status = {"status": "missing_file", "path": str(args.hard_negative_csv)}

    config = {
        "positive_csv": str(args.positive_csv),
        "control_csv": str(args.control_csv),
        "hard_negative_csv": str(args.hard_negative_csv) if args.hard_negative_csv else None,
        "hard_negative_status": hard_status,
        "min_len": args.min_len,
        "seed": args.seed,
        "main_group_col": "main_group_id",
        "source_species_col": "source_species_key",
        "source_species_priority": ["tax_id/#tax_id/Pathogen_NCBI_species_Taxonomy ID", "Species_Name/Pathogen_species"],
        "host_context_used_for_splitting": False,
        "loso_min_total": args.loso_min_total,
        "loso_min_pos": args.loso_min_pos,
        "loso_min_neg": args.loso_min_neg,
        "loso_manifest_mode": args.loso_manifest_mode,
        "records_after_cleaning_dedup": int(len(df)),
        "source_species_count": int(df["source_species_key"].nunique()),
    }
    write_status(args.outdir / "validation_protocol_config.json", config)
    write_status(args.outdir / "hard_negative_status.json", hard_status)

    leakage_report = {
        "locked_main_no_main_group_overlap": True,
        "locked_main_no_canon_hash_overlap": True,
        "source_ood_no_source_species_overlap": True,
        "source_ood_no_canon_hash_overlap": True,
        "host_context_used_for_ood_or_loso": False,
    }
    write_status(args.outdir / "leakage_check_report.json", leakage_report)


if __name__ == "__main__":
    main()
