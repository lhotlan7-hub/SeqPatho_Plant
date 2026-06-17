# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import os
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import GroupShuffleSplit


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POS_FILE = PROJECT_ROOT / "Dataset" / "Patho_plantlen2000.csv"
DEFAULT_NEG_FILE = PROJECT_ROOT / "Dataset" / "Nonpatho_plants_only.csv"
DEFAULT_TAXONOMY_MAP_FILE = PROJECT_ROOT / "Dataset" / "taxonomy_map.csv"
DEFAULT_DNABERTS_DIR = Path(r"C:\Users\GQ\.cache\huggingface\modules\transformers_modules\DNABERT-S")

SEED = 42
VALID_DNA_CHARS = set("ACGTN")
RC_TABLE = str.maketrans("ACGTN", "TGCAN")

STANDARD_TRAIN_FRAC = 0.70
STANDARD_VAL_FRAC = 0.15
STANDARD_TEST_FRAC = 0.15
OOD_VAL_FRAC_FROM_REST = 0.15

OOD_SPECIES_TOPK = 10
OOD_SPECIES_MIN_TOTAL = 80
OOD_SPECIES_MIN_POS = 10
OOD_SPECIES_MIN_NEG = 10


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: os.PathLike) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def make_result_dir(output_root: os.PathLike, prefix: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ensure_dir(Path(output_root) / f"results_{prefix}_{timestamp}")


def save_json(obj: Dict, path: os.PathLike) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_df(df: pd.DataFrame, out_csv: os.PathLike, out_md: Optional[os.PathLike] = None) -> None:
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    if out_md is not None:
        try:
            with open(out_md, "w", encoding="utf-8") as f:
                f.write(df.to_markdown(index=False))
                f.write("\n")
        except Exception:
            pass


def safe_str(x, na: str = "NA") -> str:
    if pd.isna(x):
        return na
    s = str(x).strip()
    return s if s else na


def clean_sequence(seq: str) -> str:
    s = str(seq).strip().upper()
    if not s:
        return ""
    return "".join(ch if ch in VALID_DNA_CHARS else "N" for ch in s)


def rev_comp(seq: str) -> str:
    return str(seq).translate(RC_TABLE)[::-1]


def canon_rc(seq: str) -> str:
    s = clean_sequence(seq)
    if not s:
        return ""
    rc = rev_comp(s)
    return s if s <= rc else rc


def normalize_species_name(x: str) -> str:
    s = safe_str(x, na="Unknown")
    return re.sub(r"\s+", " ", s.strip()) or "Unknown"


def normalize_tax_id(x) -> str:
    if pd.isna(x):
        return "NA"
    try:
        if isinstance(x, (int, np.integer)):
            return str(int(x))
        if isinstance(x, float):
            if np.isnan(x):
                return "NA"
            return str(int(x))
        xs = str(x).strip()
        if not xs or xs == "-":
            return "NA"
        if re.fullmatch(r"\d+\.0", xs):
            return str(int(float(xs)))
        return xs
    except Exception:
        return safe_str(x, na="NA")


def species_label_from_key(x: str) -> str:
    s = safe_str(x, na="Unknown")
    if "|" in s:
        parts = s.split("|", 1)
        if len(parts) == 2 and parts[1].strip():
            return parts[1].strip()
    return s


def _find_first_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


OOMYCETE_KEYWORDS = [
    "phytophthora",
    "pythium",
    "peronospora",
    "plasmopara",
    "hyaloperonospora",
    "albugo",
    "downy mildew",
    "bremia",
]

FUNGI_KEYWORDS = [
    "fusarium",
    "magnaporthe",
    "botrytis",
    "colletotrichum",
    "verticillium",
    "ustilago",
    "puccinia",
    "rhizoctonia",
    "alternaria",
    "sclerotinia",
    "cladosporium",
    "aspergillus",
    "penicillium",
    "cercospora",
    "blumeria",
]

BACTERIA_KEYWORDS = [
    "xanthomonas",
    "pseudomonas",
    "ralstonia",
    "erwinia",
    "pectobacterium",
    "agrobacterium",
    "burkholderia",
    "clavibacter",
    "streptomyces",
    "dickeya",
    "xylella",
    "bacillus",
]


def infer_clade_from_species_name(species_name: str) -> str:
    s = normalize_species_name(species_name).lower()
    if s in {"unknown", "na", "-"}:
        return "Unknown"
    if any(k in s for k in OOMYCETE_KEYWORDS):
        return "Oomycete"
    if any(k in s for k in FUNGI_KEYWORDS):
        return "Fungi"
    if any(k in s for k in BACTERIA_KEYWORDS):
        return "Bacteria"
    return "Other/Unknown"


def apply_clade_mapping(df_all: pd.DataFrame, taxonomy_map_file: os.PathLike) -> pd.DataFrame:
    df_all = df_all.copy()
    df_all["clade_label"] = np.nan
    taxonomy_map_file = Path(taxonomy_map_file)
    if taxonomy_map_file.exists():
        taxmap = pd.read_csv(taxonomy_map_file, low_memory=False)
        if "clade_label" in taxmap.columns:
            if "tax_id" in taxmap.columns:
                tmp = taxmap[["tax_id", "clade_label"]].dropna().drop_duplicates().copy()
                tmp["tax_id"] = tmp["tax_id"].map(normalize_tax_id)
                df_all = df_all.merge(tmp, on="tax_id", how="left", suffixes=("", "_taxmap_tax"))
                if "clade_label_taxmap_tax" in df_all.columns:
                    df_all["clade_label"] = df_all["clade_label_taxmap_tax"].fillna(df_all["clade_label"])
                    df_all.drop(columns=["clade_label_taxmap_tax"], inplace=True)
            if "species_name" in taxmap.columns:
                tmp = taxmap[["species_name", "clade_label"]].dropna().drop_duplicates().copy()
                tmp["species_name"] = tmp["species_name"].map(normalize_species_name)
                df_all = df_all.merge(tmp, on="species_name", how="left", suffixes=("", "_taxmap_sp"))
                if "clade_label_taxmap_sp" in df_all.columns:
                    need = df_all["clade_label"].isna() | (df_all["clade_label"].astype(str).str.len() == 0)
                    df_all.loc[need, "clade_label"] = df_all.loc[need, "clade_label_taxmap_sp"]
                    df_all.drop(columns=["clade_label_taxmap_sp"], inplace=True)

    need_heuristic = df_all["clade_label"].isna() | (df_all["clade_label"].astype(str).str.len() == 0)
    df_all.loc[need_heuristic, "clade_label"] = df_all.loc[need_heuristic, "species_name"].map(
        infer_clade_from_species_name
    )
    df_all["clade_label"] = df_all["clade_label"].fillna("Unknown").astype(str)
    return df_all


def load_combined_data(
    pos_file: os.PathLike = DEFAULT_POS_FILE,
    neg_file: os.PathLike = DEFAULT_NEG_FILE,
    taxonomy_map_file: os.PathLike = DEFAULT_TAXONOMY_MAP_FILE,
) -> pd.DataFrame:
    df_a = pd.read_csv(pos_file, low_memory=False)
    df_b = pd.read_csv(neg_file, low_memory=False)

    seq_col_a = df_a.columns[14]
    seq_col_b = df_b.columns[13]

    tax_col_a = _find_first_col(df_a, ["tax_id", "#tax_id", "Tax_ID", "taxid"])
    tax_col_b = _find_first_col(df_b, ["tax_id", "#tax_id", "Tax_ID", "taxid"])
    species_col_a = _find_first_col(df_a, ["Species_Name", "species_name", "Species", "species"])
    species_col_b = _find_first_col(df_b, ["Species_Name", "species_name", "Species", "species"])
    comp_col_a = "comp_id" if "comp_id" in df_a.columns else None
    comp_col_b = "comp_id" if "comp_id" in df_b.columns else None

    pos = pd.DataFrame({"seq_raw": df_a[seq_col_a].astype(str), "label": 1})
    neg = pd.DataFrame({"seq_raw": df_b[seq_col_b].astype(str), "label": 0})

    if tax_col_a:
        pos["tax_id"] = df_a[tax_col_a].values
    if tax_col_b:
        neg["tax_id"] = df_b[tax_col_b].values
    if species_col_a:
        pos["species_name"] = df_a[species_col_a].values
    if species_col_b:
        neg["species_name"] = df_b[species_col_b].values
    if comp_col_a:
        pos["comp_id"] = df_a[comp_col_a].values
    if comp_col_b:
        neg["comp_id"] = df_b[comp_col_b].values

    df_all = pd.concat([pos, neg], ignore_index=True)
    print(f"[Load] raw sequences: pos={len(pos)} neg={len(neg)} total={len(df_all)}")

    df_all["seq"] = df_all["seq_raw"].map(clean_sequence)
    df_all["seq_len"] = df_all["seq"].str.len()
    df_all = df_all[df_all["seq_len"] > 0].copy()
    df_all["canon"] = df_all["seq"].map(canon_rc)
    df_all["tax_id"] = df_all.get("tax_id", pd.Series(["NA"] * len(df_all))).map(normalize_tax_id)
    df_all["species_name"] = df_all.get("species_name", pd.Series(["Unknown"] * len(df_all))).map(
        normalize_species_name
    )
    if "comp_id" in df_all.columns:
        df_all["comp_id"] = df_all["comp_id"].map(lambda x: safe_str(x, na="NA"))
    else:
        df_all["comp_id"] = "NA"

    n_before = len(df_all)
    conflict_groups = df_all.groupby("canon")["label"].nunique()
    n_conflict = int((conflict_groups > 1).sum())
    if n_conflict > 0:
        print(f"[Dedup] conflicting canonical groups={n_conflict}; keeping positive records first")
        df_all = df_all.sort_values(by="label", ascending=False)
    df_all = df_all.drop_duplicates(subset="canon", keep="first").reset_index(drop=True)
    print(f"[Dedup] canonical reverse-complement: {n_before} -> {len(df_all)}")

    def build_group_id(row: pd.Series) -> str:
        tax = row["tax_id"]
        species = row["species_name"]
        comp = row["comp_id"]
        if comp != "NA" and tax != "NA":
            return f"{tax}|c{comp}"
        if comp != "NA":
            return f"NA|c{comp}"
        if tax != "NA":
            return tax
        if species != "Unknown":
            return species
        return row["canon"]

    def build_species_key(row: pd.Series) -> str:
        tax = row["tax_id"]
        sp = row["species_name"]
        if tax != "NA" and sp != "Unknown":
            return f"{tax}|{sp}"
        if tax != "NA":
            return f"{tax}|Unknown"
        return f"NA|{sp}"

    df_all["group_id"] = df_all.apply(build_group_id, axis=1)
    df_all["species_key"] = df_all.apply(build_species_key, axis=1)
    df_all["species_label"] = df_all["species_key"].map(species_label_from_key)
    df_all = apply_clade_mapping(df_all, taxonomy_map_file)
    df_all["clade_key"] = df_all["clade_label"].fillna("Unknown").astype(str)
    df_all["label"] = df_all["label"].astype(int)
    df_all.insert(0, "sample_id", [f"seq_{i:06d}" for i in range(len(df_all))])

    print(f"[Meta] unique group_id={df_all['group_id'].nunique()}")
    print(f"[Meta] unique species_key={df_all['species_key'].nunique()}")
    return df_all


def log_split_stats(name: str, df: pd.DataFrame, unit_col: Optional[str] = None) -> None:
    pos = int(df["label"].sum())
    neg = int((1 - df["label"]).sum())
    msg = f"[{name}] n={len(df)} pos={pos} neg={neg} pos_rate={df['label'].mean():.4f}"
    if unit_col is not None and unit_col in df.columns:
        msg += f" unique_{unit_col}={df[unit_col].nunique()}"
    print(msg)


def ensure_split_has_both_labels(name: str, df: pd.DataFrame) -> None:
    labels = sorted(df["label"].dropna().astype(int).unique().tolist())
    if len(labels) < 2:
        raise ValueError(f"{name} only contains labels {labels}; adjust top-k species or split settings.")


def strict_overlap_check(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    split_unit_col: Optional[str] = None,
) -> None:
    for col in ["canon", "group_id"]:
        a, b, c = set(df_train[col]), set(df_val[col]), set(df_test[col])
        inter = {"train_val": len(a & b), "train_test": len(a & c), "val_test": len(b & c)}
        print(f"[LeakCheck] {col}: {inter}")
        if any(v != 0 for v in inter.values()):
            raise AssertionError(f"{col} overlap detected across splits")

    if split_unit_col is not None:
        a, b, c = set(df_train[split_unit_col]), set(df_val[split_unit_col]), set(df_test[split_unit_col])
        inter = {"train_val": len(a & b), "train_test": len(a & c), "val_test": len(b & c)}
        print(f"[LeakCheck] {split_unit_col}: {inter}")
        if any(v != 0 for v in inter.values()):
            raise AssertionError(f"{split_unit_col} overlap detected across splits")


def split_three_way_by_unit(
    df_all: pd.DataFrame,
    unit_col: str,
    train_frac: float = STANDARD_TRAIN_FRAC,
    val_frac: float = STANDARD_VAL_FRAC,
    test_frac: float = STANDARD_TEST_FRAC,
    random_state: int = SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if abs(train_frac + val_frac + test_frac - 1.0) >= 1e-8:
        raise ValueError("train/val/test fractions must sum to 1.0")
    units = df_all[unit_col].values
    y = df_all["label"].values
    gss1 = GroupShuffleSplit(n_splits=1, test_size=(1 - train_frac), random_state=random_state)
    train_idx, temp_idx = next(gss1.split(df_all, y, groups=units))
    df_train = df_all.iloc[train_idx].reset_index(drop=True)
    df_temp = df_all.iloc[temp_idx].reset_index(drop=True)

    relative_test = test_frac / (val_frac + test_frac)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=relative_test, random_state=random_state + 1)
    val_idx, test_idx = next(gss2.split(df_temp, df_temp["label"].values, groups=df_temp[unit_col].values))
    df_val = df_temp.iloc[val_idx].reset_index(drop=True)
    df_test = df_temp.iloc[test_idx].reset_index(drop=True)

    log_split_stats("train", df_train, unit_col)
    log_split_stats("val", df_val, unit_col)
    log_split_stats("test", df_test, unit_col)
    ensure_split_has_both_labels("train", df_train)
    ensure_split_has_both_labels("val", df_val)
    ensure_split_has_both_labels("test", df_test)
    strict_overlap_check(df_train, df_val, df_test, split_unit_col=unit_col)
    return df_train, df_val, df_test


def split_holdout_unit(
    df_all: pd.DataFrame,
    holdout_unit_col: str,
    holdout_value: str,
    val_group_col: str = "species_key",
    val_frac_from_rest: float = OOD_VAL_FRAC_FROM_REST,
    random_state: int = SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_test = df_all[df_all[holdout_unit_col] == holdout_value].reset_index(drop=True)
    df_rest = df_all[df_all[holdout_unit_col] != holdout_value].reset_index(drop=True)
    if len(df_test) == 0:
        raise ValueError(f"empty holdout: {holdout_unit_col}={holdout_value}")

    gss = GroupShuffleSplit(n_splits=1, test_size=val_frac_from_rest, random_state=random_state)
    train_idx, val_idx = next(gss.split(df_rest, df_rest["label"].values, groups=df_rest[val_group_col].values))
    df_train = df_rest.iloc[train_idx].reset_index(drop=True)
    df_val = df_rest.iloc[val_idx].reset_index(drop=True)

    log_split_stats("train", df_train, holdout_unit_col)
    log_split_stats("val", df_val, val_group_col)
    log_split_stats("test", df_test, holdout_unit_col)
    ensure_split_has_both_labels("train", df_train)
    ensure_split_has_both_labels("val", df_val)
    ensure_split_has_both_labels("test", df_test)
    strict_overlap_check(df_train, df_val, df_test, split_unit_col=holdout_unit_col)
    return df_train, df_val, df_test


def build_unit_manifest(df_all: pd.DataFrame, unit_col: str) -> pd.DataFrame:
    rows = []
    for unit, sub in df_all.groupby(unit_col):
        row = {
            unit_col: unit,
            "n_total": len(sub),
            "n_pos": int(sub["label"].sum()),
            "n_neg": int((1 - sub["label"]).sum()),
            "pos_rate": float(sub["label"].mean()),
            "n_groups": sub["group_id"].nunique(),
            "median_len": float(sub["seq_len"].median()),
            "mean_len": float(sub["seq_len"].mean()),
        }
        if unit_col == "species_key":
            row["species_label"] = species_label_from_key(unit)
        elif unit_col == "clade_key":
            row["clade_label_display"] = str(unit)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["n_total", "n_pos"], ascending=[False, False]).reset_index(drop=True)


def restrict_dataset_to_top_species(
    df_all: pd.DataFrame,
    top_k: int = 10,
    require_both_labels: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manifest = build_unit_manifest(df_all, "species_key")
    selected = manifest.copy()
    if require_both_labels:
        selected = selected[(selected["n_pos"] > 0) & (selected["n_neg"] > 0)].copy()
    selected = selected.sort_values(["n_total", "n_pos"], ascending=[False, False]).reset_index(drop=True)
    if top_k is not None:
        selected = selected.head(top_k).copy()
    if len(selected) == 0:
        raise ValueError("top-k species filtering produced an empty dataset")
    selected.insert(0, "rank", np.arange(1, len(selected) + 1))
    filtered = df_all[df_all["species_key"].isin(selected["species_key"])].reset_index(drop=True)
    ensure_split_has_both_labels("filtered_dataset", filtered)
    print(f"[TopSpecies] kept {len(selected)} species: {len(df_all)} -> {len(filtered)}")
    return filtered, manifest, selected


def filter_species_candidates(manifest: pd.DataFrame, top_k: int = OOD_SPECIES_TOPK) -> pd.DataFrame:
    out = manifest[
        (manifest["n_total"] >= OOD_SPECIES_MIN_TOTAL)
        & (manifest["n_pos"] >= OOD_SPECIES_MIN_POS)
        & (manifest["n_neg"] >= OOD_SPECIES_MIN_NEG)
    ].copy()
    out = out.sort_values(["n_total", "n_pos"], ascending=[False, False]).reset_index(drop=True)
    if top_k is not None:
        out = out.head(top_k).copy()
    return out


def save_split_files(root_dir: os.PathLike, df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame) -> None:
    split_dir = ensure_dir(Path(root_dir) / "splits")
    cols = [
        "sample_id",
        "label",
        "seq_len",
        "tax_id",
        "species_name",
        "species_label",
        "clade_key",
        "group_id",
        "species_key",
    ]
    save_df(df_train[cols], split_dir / "train.csv")
    save_df(df_val[cols], split_dir / "val.csv")
    save_df(df_test[cols], split_dir / "test.csv")


def prepare_dataset(
    root_dir: os.PathLike,
    pos_file: os.PathLike = DEFAULT_POS_FILE,
    neg_file: os.PathLike = DEFAULT_NEG_FILE,
    taxonomy_map_file: os.PathLike = DEFAULT_TAXONOMY_MAP_FILE,
    limit_to_topk_species: bool = True,
    top_k_species: int = 10,
    require_both_labels: bool = True,
) -> pd.DataFrame:
    df_all = load_combined_data(pos_file, neg_file, taxonomy_map_file)
    species_manifest_all = build_unit_manifest(df_all, "species_key")
    save_df(
        species_manifest_all,
        Path(root_dir) / "species_manifest_before_filter.csv",
        Path(root_dir) / "species_manifest_before_filter.md",
    )
    if limit_to_topk_species:
        df_all, _, selected = restrict_dataset_to_top_species(
            df_all, top_k=top_k_species, require_both_labels=require_both_labels
        )
        save_df(selected, Path(root_dir) / "species_manifest_selected.csv", Path(root_dir) / "species_manifest_selected.md")

    compact_cols = [
        "sample_id",
        "seq_len",
        "label",
        "tax_id",
        "species_name",
        "species_label",
        "clade_key",
        "group_id",
        "species_key",
    ]
    save_df(
        df_all[compact_cols],
        Path(root_dir) / "dataset_manifest_compact.csv",
        Path(root_dir) / "dataset_manifest_compact.md",
    )
    return df_all


def compute_classification_metrics(labels, preds, probs) -> Dict[str, float]:
    labels = np.asarray(labels).astype(int)
    preds = np.asarray(preds).astype(int)
    probs = np.asarray(probs).astype(float)
    if len(labels) == 0:
        return {
            "accuracy": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "mcc": np.nan,
            "auroc": np.nan,
            "auprc": np.nan,
            "ap": np.nan,
            "n": 0,
            "pos": 0,
            "neg": 0,
        }
    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, preds)),
        "n": int(len(labels)),
        "pos": int(np.sum(labels == 1)),
        "neg": int(np.sum(labels == 0)),
    }
    try:
        fpr, tpr, _ = roc_curve(labels, probs)
        metrics["auroc"] = float(auc(fpr, tpr))
    except Exception:
        metrics["auroc"] = np.nan
    try:
        precision_vals, recall_vals, _ = precision_recall_curve(labels, probs)
        metrics["auprc"] = float(auc(recall_vals, precision_vals))
        metrics["ap"] = float(average_precision_score(labels, probs))
    except Exception:
        metrics["auprc"] = np.nan
        metrics["ap"] = np.nan
    return metrics


def prediction_frame(df: pd.DataFrame, probs, preds, extra: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    out = df[
        [
            "sample_id",
            "label",
            "seq_len",
            "tax_id",
            "species_name",
            "species_label",
            "group_id",
            "species_key",
        ]
    ].copy()
    out["prob_pos"] = np.asarray(probs, dtype=float)
    out["pred"] = np.asarray(preds, dtype=int)
    if extra is not None:
        out = pd.concat([out.reset_index(drop=True), extra.reset_index(drop=True)], axis=1)
    return out


def plot_confusion_matrix(labels, preds, out_png: os.PathLike, title: str = "Confusion Matrix") -> None:
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def plot_roc_pr(labels, probs, roc_png: os.PathLike, pr_png: os.PathLike, title_prefix: str = "Test") -> None:
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs).astype(float)
    try:
        fpr, tpr, _ = roc_curve(labels, probs)
        auroc = auc(fpr, tpr)
        plt.figure(figsize=(5, 4))
        plt.plot(fpr, tpr, lw=2, label=f"ROC (AUROC = {auroc:.4f})")
        plt.plot([0, 1], [0, 1], lw=1.5, linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{title_prefix} ROC Curve")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(roc_png, dpi=300)
        plt.close()
    except Exception:
        pass

    try:
        precision_vals, recall_vals, _ = precision_recall_curve(labels, probs)
        ap = average_precision_score(labels, probs)
        auprc = auc(recall_vals, precision_vals)
        plt.figure(figsize=(5, 4))
        plt.step(recall_vals, precision_vals, where="post", lw=2, label=f"PR (AP = {ap:.4f}, AUPRC = {auprc:.4f})")
        plt.fill_between(recall_vals, precision_vals, step="post", alpha=0.2)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"{title_prefix} PR Curve")
        plt.legend(loc="upper right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(pr_png, dpi=300)
        plt.close()
    except Exception:
        pass


def plot_metric_bars(df: pd.DataFrame, out_png: os.PathLike, title: str, metric: str = "auprc") -> None:
    if len(df) == 0 or metric not in df.columns:
        return
    plot_df = df.sort_values(metric, ascending=False).copy()
    plt.figure(figsize=(10, max(4, 0.45 * len(plot_df))))
    sns.barplot(data=plot_df, x=metric, y="baseline")
    plt.title(title)
    plt.xlim(0, min(1.0, max(0.05, float(plot_df[metric].max()) * 1.08)))
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def write_fasta(df: pd.DataFrame, path: os.PathLike, id_col: str = "sample_id") -> None:
    with open(path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            seq_id = str(row[id_col]).replace("|", "_").replace(" ", "_")
            f.write(f">{seq_id}\n")
            seq = str(row["seq"])
            for i in range(0, len(seq), 80):
                f.write(seq[i : i + 80] + "\n")


def sample_train_window(seq: str, l_tokens: int = 500, k: int = 6) -> str:
    s = str(seq).strip().upper()
    window_nt = l_tokens + k - 1
    if len(s) <= window_nt:
        return s
    start = random.randint(0, len(s) - window_nt)
    return s[start : start + window_nt]


def make_windows(seq: str, l_tokens: int = 500, k: int = 6, stride_tokens: Optional[int] = None) -> List[str]:
    s = str(seq).strip().upper()
    if stride_tokens is None:
        stride_tokens = l_tokens
    window_nt = l_tokens + k - 1
    if len(s) <= window_nt:
        return [s]
    step_nt = stride_tokens + k - 1
    out = [s[i : i + window_nt] for i in range(0, len(s) - window_nt + 1, step_nt)]
    if (len(s) - window_nt) % step_nt != 0:
        out.append(s[-window_nt:])
    return out


# ---------------------------------------------------------------------------
# Compatibility names and helpers used by the supplemental baseline scripts.
# These aliases keep the public API stable even if this shared file was created
# from an earlier draft during an interrupted run.
# ---------------------------------------------------------------------------
OUT_ROOT = PROJECT_ROOT / "Models_10_buchong"
POS_FILE = DEFAULT_POS_FILE
NEG_FILE = DEFAULT_NEG_FILE
TAXONOMY_MAP_FILE = DEFAULT_TAXONOMY_MAP_FILE
DNABERT_DIR = PROJECT_ROOT / "DNABERT-S" if (PROJECT_ROOT / "DNABERT-S").exists() else DEFAULT_DNABERTS_DIR
DATASET_TOPK_SPECIES = 10
DATASET_TOPK_REQUIRE_BOTH_LABELS = True


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_float(x, default=np.nan) -> float:
    try:
        if x is None:
            return default
        if hasattr(x, "item"):
            return float(x.item())
        return float(x)
    except Exception:
        return default


def gc_fraction(seq: str) -> float:
    s = clean_sequence(seq)
    denom = sum(1 for ch in s if ch in "ACGT")
    if denom == 0:
        return 0.0
    return float(s.count("G") + s.count("C")) / float(denom)


def kmer_tokens(seq: str, k: int = 6, skip_n: bool = True) -> List[str]:
    s = clean_sequence(seq)
    if len(s) < k:
        return []
    out = []
    for i in range(0, len(s) - k + 1):
        tok = s[i : i + k]
        if skip_n and "N" in tok:
            continue
        out.append(tok)
    return out


def probs_to_preds(probs, threshold: float = 0.5) -> np.ndarray:
    return (np.asarray(probs, dtype=float) >= threshold).astype(int)


def attach_result_context(
    metrics: Dict,
    baseline_name: str,
    run_dir: os.PathLike,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    status: str = "ok",
    error: Optional[str] = None,
) -> Dict:
    out = {
        "baseline": baseline_name,
        "status": status,
        "error": error,
        **metrics,
        "n_train": int(len(df_train)),
        "n_val": int(len(df_val)),
        "n_test": int(len(df_test)),
        "train_pos": int(df_train["label"].sum()),
        "train_neg": int((1 - df_train["label"]).sum()),
        "val_pos": int(df_val["label"].sum()),
        "val_neg": int((1 - df_val["label"]).sum()),
        "test_pos": int(df_test["label"].sum()),
        "test_neg": int((1 - df_test["label"]).sum()),
        "run_dir": str(run_dir),
    }
    return out


def save_split_frames(root_dir: os.PathLike, df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame) -> None:
    cols = [
        "sample_id",
        "label",
        "seq_len",
        "tax_id",
        "species_name",
        "species_label",
        "clade_key",
        "group_id",
        "species_key",
        "canon",
        "seq",
    ]
    cols = [c for c in cols if c in df_train.columns]
    save_df(df_train[cols], Path(root_dir) / "train_split.csv")
    save_df(df_val[cols], Path(root_dir) / "val_split.csv")
    save_df(df_test[cols], Path(root_dir) / "test_split.csv")


def save_prediction_table(df: pd.DataFrame, probs, preds, path: os.PathLike, extra: Optional[pd.DataFrame] = None) -> None:
    out = prediction_frame(df, probs, preds, extra=extra)
    save_df(out, path)


def maybe_plot_curves(labels, probs, preds, out_dir: os.PathLike, prefix: str = "test") -> None:
    out_dir = Path(out_dir)
    try:
        plot_confusion_matrix(labels, preds, out_dir / f"{prefix}_confusion_matrix.png")
        plot_roc_pr(labels, probs, out_dir / f"{prefix}_roc_curve.png", out_dir / f"{prefix}_pr_curve.png")
    except Exception:
        pass


def find_executable(names: Sequence[str], extra_candidates: Sequence[os.PathLike] = ()) -> Optional[str]:
    import shutil

    for name in names:
        p = shutil.which(name)
        if p:
            return p
    for cand in extra_candidates:
        cand = Path(cand)
        if cand.exists():
            return str(cand)
    return None


def save_run_readme(path: os.PathLike, lines: Iterable[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(str(line).rstrip() + "\n")


_legacy_write_fasta = write_fasta


def write_fasta(df: pd.DataFrame, path: os.PathLike, id_prefix: str = "seq") -> pd.DataFrame:
    rows = []
    with open(path, "w", encoding="utf-8") as f:
        for i, row in df.reset_index(drop=True).iterrows():
            seq_id = f"{id_prefix}_{i:06d}_y{int(row['label'])}"
            seq = clean_sequence(row["seq"])
            f.write(f">{seq_id}\n")
            for j in range(0, len(seq), 80):
                f.write(seq[j : j + 80] + "\n")
            rows.append({"seq_id": seq_id, "row_index": i, "label": int(row["label"])})
    return pd.DataFrame(rows)


_legacy_split_three_way_by_unit = split_three_way_by_unit


def split_three_way_by_unit(
    df_all: pd.DataFrame,
    unit_col: str = "species_key",
    train_frac: float = STANDARD_TRAIN_FRAC,
    val_frac: float = STANDARD_VAL_FRAC,
    test_frac: float = STANDARD_TEST_FRAC,
    random_state: int = SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    df_train, df_val, df_test = _legacy_split_three_way_by_unit(
        df_all,
        unit_col=unit_col,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        random_state=random_state,
    )
    stats = {
        "split_unit_col": unit_col,
        "splits": [
            {"split": "train", "n": int(len(df_train)), "pos": int(df_train["label"].sum()), "neg": int((1 - df_train["label"]).sum())},
            {"split": "val", "n": int(len(df_val)), "pos": int(df_val["label"].sum()), "neg": int((1 - df_val["label"]).sum())},
            {"split": "test", "n": int(len(df_test)), "pos": int(df_test["label"].sum()), "neg": int((1 - df_test["label"]).sum())},
        ],
    }
    return df_train, df_val, df_test, stats


# EOF overrides for the current baseline scripts.
def write_fasta(df: pd.DataFrame, path: os.PathLike, id_prefix: str = "seq") -> pd.DataFrame:
    rows = []
    with open(path, "w", encoding="utf-8") as f:
        for i, row in df.reset_index(drop=True).iterrows():
            seq_id = f"{id_prefix}_{i:06d}_y{int(row['label'])}"
            seq = clean_sequence(row["seq"])
            f.write(f">{seq_id}\n")
            for j in range(0, len(seq), 80):
                f.write(seq[j : j + 80] + "\n")
            rows.append({"seq_id": seq_id, "row_index": i, "label": int(row["label"])})
    return pd.DataFrame(rows)


def _final_split_stats_dict(name: str, df: pd.DataFrame, unit_col: str) -> Dict:
    return {
        "split": name,
        "n": int(len(df)),
        "pos": int(df["label"].sum()),
        "neg": int((1 - df["label"]).sum()),
        "pos_rate": float(df["label"].mean()),
        f"unique_{unit_col}": int(df[unit_col].nunique()) if unit_col in df.columns else np.nan,
    }


def split_three_way_by_unit(
    df_all: pd.DataFrame,
    unit_col: str = "species_key",
    train_frac: float = STANDARD_TRAIN_FRAC,
    val_frac: float = STANDARD_VAL_FRAC,
    test_frac: float = STANDARD_TEST_FRAC,
    random_state: int = SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    if abs(train_frac + val_frac + test_frac - 1.0) >= 1e-8:
        raise ValueError("train/val/test fractions must sum to 1.0")
    units = df_all[unit_col].values
    y = df_all["label"].values
    gss1 = GroupShuffleSplit(n_splits=1, test_size=(1 - train_frac), random_state=random_state)
    train_idx, temp_idx = next(gss1.split(df_all, y, groups=units))
    df_train = df_all.iloc[train_idx].reset_index(drop=True)
    df_temp = df_all.iloc[temp_idx].reset_index(drop=True)
    relative_test = test_frac / (val_frac + test_frac)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=relative_test, random_state=random_state + 1)
    val_idx, test_idx = next(gss2.split(df_temp, df_temp["label"].values, groups=df_temp[unit_col].values))
    df_val = df_temp.iloc[val_idx].reset_index(drop=True)
    df_test = df_temp.iloc[test_idx].reset_index(drop=True)
    log_split_stats("train", df_train, unit_col)
    log_split_stats("val", df_val, unit_col)
    log_split_stats("test", df_test, unit_col)
    ensure_split_has_both_labels("train", df_train)
    ensure_split_has_both_labels("val", df_val)
    ensure_split_has_both_labels("test", df_test)
    strict_overlap_check(df_train, df_val, df_test, split_unit_col=unit_col)
    stats = {
        "split_unit_col": unit_col,
        "splits": [
            _final_split_stats_dict("train", df_train, unit_col),
            _final_split_stats_dict("val", df_val, unit_col),
            _final_split_stats_dict("test", df_test, unit_col),
        ],
    }
    return df_train, df_val, df_test, stats


def split_holdout_unit(
    df_all: pd.DataFrame,
    holdout_unit_col: str,
    holdout_value: str,
    val_group_col: str = "species_key",
    val_frac_from_rest: float = OOD_VAL_FRAC_FROM_REST,
    random_state: int = SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    df_test = df_all[df_all[holdout_unit_col] == holdout_value].reset_index(drop=True)
    df_rest = df_all[df_all[holdout_unit_col] != holdout_value].reset_index(drop=True)
    if len(df_test) == 0:
        raise ValueError(f"empty holdout: {holdout_unit_col}={holdout_value}")
    gss = GroupShuffleSplit(n_splits=1, test_size=val_frac_from_rest, random_state=random_state)
    train_idx, val_idx = next(gss.split(df_rest, df_rest["label"].values, groups=df_rest[val_group_col].values))
    df_train = df_rest.iloc[train_idx].reset_index(drop=True)
    df_val = df_rest.iloc[val_idx].reset_index(drop=True)
    log_split_stats("train", df_train, holdout_unit_col)
    log_split_stats("val", df_val, val_group_col)
    log_split_stats("test", df_test, holdout_unit_col)
    ensure_split_has_both_labels("train", df_train)
    ensure_split_has_both_labels("val", df_val)
    ensure_split_has_both_labels("test", df_test)
    strict_overlap_check(df_train, df_val, df_test, split_unit_col=holdout_unit_col)
    stats = {
        "holdout_unit_col": holdout_unit_col,
        "holdout_value": holdout_value,
        "val_group_col": val_group_col,
        "splits": [
            _final_split_stats_dict("train", df_train, holdout_unit_col),
            _final_split_stats_dict("val", df_val, val_group_col),
            _final_split_stats_dict("test", df_test, holdout_unit_col),
        ],
    }
    return df_train, df_val, df_test, stats


# Hard final overrides. Keep these at EOF so earlier interrupted-run definitions
# cannot shadow the API used by run_classical_and_homology_baselines.py.
def write_fasta(df: pd.DataFrame, path: os.PathLike, id_prefix: str = "seq") -> pd.DataFrame:
    rows = []
    with open(path, "w", encoding="utf-8") as f:
        for i, row in df.reset_index(drop=True).iterrows():
            seq_id = f"{id_prefix}_{i:06d}_y{int(row['label'])}"
            seq = clean_sequence(row["seq"])
            f.write(f">{seq_id}\n")
            for j in range(0, len(seq), 80):
                f.write(seq[j : j + 80] + "\n")
            rows.append({"seq_id": seq_id, "row_index": i, "label": int(row["label"])})
    return pd.DataFrame(rows)


def _split_stats_dict(name: str, df: pd.DataFrame, unit_col: str) -> Dict:
    return {
        "split": name,
        "n": int(len(df)),
        "pos": int(df["label"].sum()),
        "neg": int((1 - df["label"]).sum()),
        "pos_rate": float(df["label"].mean()),
        f"unique_{unit_col}": int(df[unit_col].nunique()) if unit_col in df.columns else np.nan,
    }


def split_three_way_by_unit(
    df_all: pd.DataFrame,
    unit_col: str = "species_key",
    train_frac: float = STANDARD_TRAIN_FRAC,
    val_frac: float = STANDARD_VAL_FRAC,
    test_frac: float = STANDARD_TEST_FRAC,
    random_state: int = SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    if abs(train_frac + val_frac + test_frac - 1.0) >= 1e-8:
        raise ValueError("train/val/test fractions must sum to 1.0")
    units = df_all[unit_col].values
    y = df_all["label"].values
    gss1 = GroupShuffleSplit(n_splits=1, test_size=(1 - train_frac), random_state=random_state)
    train_idx, temp_idx = next(gss1.split(df_all, y, groups=units))
    df_train = df_all.iloc[train_idx].reset_index(drop=True)
    df_temp = df_all.iloc[temp_idx].reset_index(drop=True)
    relative_test = test_frac / (val_frac + test_frac)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=relative_test, random_state=random_state + 1)
    val_idx, test_idx = next(gss2.split(df_temp, df_temp["label"].values, groups=df_temp[unit_col].values))
    df_val = df_temp.iloc[val_idx].reset_index(drop=True)
    df_test = df_temp.iloc[test_idx].reset_index(drop=True)
    log_split_stats("train", df_train, unit_col)
    log_split_stats("val", df_val, unit_col)
    log_split_stats("test", df_test, unit_col)
    ensure_split_has_both_labels("train", df_train)
    ensure_split_has_both_labels("val", df_val)
    ensure_split_has_both_labels("test", df_test)
    strict_overlap_check(df_train, df_val, df_test, split_unit_col=unit_col)
    stats = {
        "split_unit_col": unit_col,
        "splits": [
            _split_stats_dict("train", df_train, unit_col),
            _split_stats_dict("val", df_val, unit_col),
            _split_stats_dict("test", df_test, unit_col),
        ],
    }
    return df_train, df_val, df_test, stats


def split_holdout_unit(
    df_all: pd.DataFrame,
    holdout_unit_col: str,
    holdout_value: str,
    val_group_col: str = "species_key",
    val_frac_from_rest: float = OOD_VAL_FRAC_FROM_REST,
    random_state: int = SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    df_test = df_all[df_all[holdout_unit_col] == holdout_value].reset_index(drop=True)
    df_rest = df_all[df_all[holdout_unit_col] != holdout_value].reset_index(drop=True)
    if len(df_test) == 0:
        raise ValueError(f"empty holdout: {holdout_unit_col}={holdout_value}")
    gss = GroupShuffleSplit(n_splits=1, test_size=val_frac_from_rest, random_state=random_state)
    train_idx, val_idx = next(gss.split(df_rest, df_rest["label"].values, groups=df_rest[val_group_col].values))
    df_train = df_rest.iloc[train_idx].reset_index(drop=True)
    df_val = df_rest.iloc[val_idx].reset_index(drop=True)
    log_split_stats("train", df_train, holdout_unit_col)
    log_split_stats("val", df_val, val_group_col)
    log_split_stats("test", df_test, holdout_unit_col)
    ensure_split_has_both_labels("train", df_train)
    ensure_split_has_both_labels("val", df_val)
    ensure_split_has_both_labels("test", df_test)
    strict_overlap_check(df_train, df_val, df_test, split_unit_col=holdout_unit_col)
    stats = {
        "holdout_unit_col": holdout_unit_col,
        "holdout_value": holdout_value,
        "val_group_col": val_group_col,
        "splits": [
            _split_stats_dict("train", df_train, holdout_unit_col),
            _split_stats_dict("val", df_val, val_group_col),
            _split_stats_dict("test", df_test, holdout_unit_col),
        ],
    }
    return df_train, df_val, df_test, stats


def split_holdout_unit(
    df_all: pd.DataFrame,
    holdout_unit_col: str,
    holdout_value: str,
    val_group_col: str = "species_key",
    val_frac_from_rest: float = OOD_VAL_FRAC_FROM_REST,
    random_state: int = SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_test = df_all[df_all[holdout_unit_col] == holdout_value].reset_index(drop=True)
    df_rest = df_all[df_all[holdout_unit_col] != holdout_value].reset_index(drop=True)
    if len(df_test) == 0:
        raise ValueError(f"empty holdout: {holdout_unit_col}={holdout_value}")

    gss = GroupShuffleSplit(n_splits=1, test_size=val_frac_from_rest, random_state=random_state)
    train_idx, val_idx = next(gss.split(df_rest, df_rest["label"].values, groups=df_rest[val_group_col].values))
    df_train = df_rest.iloc[train_idx].reset_index(drop=True)
    df_val = df_rest.iloc[val_idx].reset_index(drop=True)

    log_split_stats("train", df_train, holdout_unit_col)
    log_split_stats("val", df_val, val_group_col)
    log_split_stats("test", df_test, holdout_unit_col)
    ensure_split_has_both_labels("train", df_train)
    ensure_split_has_both_labels("val", df_val)
    ensure_split_has_both_labels("test", df_test)
    strict_overlap_check(df_train, df_val, df_test, split_unit_col=holdout_unit_col)
    return df_train, df_val, df_test


def split_holdout_unit(
    df_all: pd.DataFrame,
    holdout_unit_col: str,
    holdout_value: str,
    val_group_col: str = "species_key",
    val_frac_from_rest: float = OOD_VAL_FRAC_FROM_REST,
    random_state: int = SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_test = df_all[df_all[holdout_unit_col] == holdout_value].reset_index(drop=True)
    df_rest = df_all[df_all[holdout_unit_col] != holdout_value].reset_index(drop=True)
    if len(df_test) == 0:
        raise ValueError(f"empty holdout: {holdout_unit_col}={holdout_value}")

    gss = GroupShuffleSplit(n_splits=1, test_size=val_frac_from_rest, random_state=random_state)
    train_idx, val_idx = next(gss.split(df_rest, df_rest["label"].values, groups=df_rest[val_group_col].values))
    df_train = df_rest.iloc[train_idx].reset_index(drop=True)
    df_val = df_rest.iloc[val_idx].reset_index(drop=True)

    log_split_stats("train", df_train, holdout_unit_col)
    log_split_stats("val", df_val, val_group_col)
    log_split_stats("test", df_test, holdout_unit_col)
    ensure_split_has_both_labels("train", df_train)
    ensure_split_has_both_labels("val", df_val)
    ensure_split_has_both_labels("test", df_test)
    strict_overlap_check(df_train, df_val, df_test, split_unit_col=holdout_unit_col)
    return df_train, df_val, df_test


# Final public API used by the current scripts. These definitions intentionally
# return exactly three split frames and write FASTA records keyed by sample_id.
def write_fasta(df: pd.DataFrame, path: os.PathLike, id_col: str = "sample_id") -> None:
    with open(path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            seq_id = str(row[id_col]).replace("|", "_").replace(" ", "_")
            f.write(f">{seq_id}\n")
            seq = str(row["seq"])
            for i in range(0, len(seq), 80):
                f.write(seq[i : i + 80] + "\n")


def split_three_way_by_unit(
    df_all: pd.DataFrame,
    unit_col: str,
    train_frac: float = STANDARD_TRAIN_FRAC,
    val_frac: float = STANDARD_VAL_FRAC,
    test_frac: float = STANDARD_TEST_FRAC,
    random_state: int = SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if abs(train_frac + val_frac + test_frac - 1.0) >= 1e-8:
        raise ValueError("train/val/test fractions must sum to 1.0")
    units = df_all[unit_col].values
    y = df_all["label"].values
    gss1 = GroupShuffleSplit(n_splits=1, test_size=(1 - train_frac), random_state=random_state)
    train_idx, temp_idx = next(gss1.split(df_all, y, groups=units))
    df_train = df_all.iloc[train_idx].reset_index(drop=True)
    df_temp = df_all.iloc[temp_idx].reset_index(drop=True)

    relative_test = test_frac / (val_frac + test_frac)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=relative_test, random_state=random_state + 1)
    val_idx, test_idx = next(gss2.split(df_temp, df_temp["label"].values, groups=df_temp[unit_col].values))
    df_val = df_temp.iloc[val_idx].reset_index(drop=True)
    df_test = df_temp.iloc[test_idx].reset_index(drop=True)

    log_split_stats("train", df_train, unit_col)
    log_split_stats("val", df_val, unit_col)
    log_split_stats("test", df_test, unit_col)
    ensure_split_has_both_labels("train", df_train)
    ensure_split_has_both_labels("val", df_val)
    ensure_split_has_both_labels("test", df_test)
    strict_overlap_check(df_train, df_val, df_test, split_unit_col=unit_col)
    return df_train, df_val, df_test


def split_holdout_unit(
    df_all: pd.DataFrame,
    holdout_unit_col: str,
    holdout_value: str,
    val_group_col: str = "species_key",
    val_frac_from_rest: float = OOD_VAL_FRAC_FROM_REST,
    random_state: int = SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_test = df_all[df_all[holdout_unit_col] == holdout_value].reset_index(drop=True)
    df_rest = df_all[df_all[holdout_unit_col] != holdout_value].reset_index(drop=True)
    if len(df_test) == 0:
        raise ValueError(f"empty holdout: {holdout_unit_col}={holdout_value}")

    gss = GroupShuffleSplit(n_splits=1, test_size=val_frac_from_rest, random_state=random_state)
    train_idx, val_idx = next(gss.split(df_rest, df_rest["label"].values, groups=df_rest[val_group_col].values))
    df_train = df_rest.iloc[train_idx].reset_index(drop=True)
    df_val = df_rest.iloc[val_idx].reset_index(drop=True)

    log_split_stats("train", df_train, holdout_unit_col)
    log_split_stats("val", df_val, val_group_col)
    log_split_stats("test", df_test, holdout_unit_col)
    ensure_split_has_both_labels("train", df_train)
    ensure_split_has_both_labels("val", df_val)
    ensure_split_has_both_labels("test", df_test)
    strict_overlap_check(df_train, df_val, df_test, split_unit_col=holdout_unit_col)
    return df_train, df_val, df_test


_legacy_split_holdout_unit = split_holdout_unit


def split_holdout_unit(
    df_all: pd.DataFrame,
    holdout_unit_col: str,
    holdout_value: str,
    val_group_col: str = "species_key",
    val_frac_from_rest: float = OOD_VAL_FRAC_FROM_REST,
    random_state: int = SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    df_train, df_val, df_test = _legacy_split_holdout_unit(
        df_all,
        holdout_unit_col=holdout_unit_col,
        holdout_value=holdout_value,
        val_group_col=val_group_col,
        val_frac_from_rest=val_frac_from_rest,
        random_state=random_state,
    )
    stats = {
        "holdout_unit_col": holdout_unit_col,
        "holdout_value": holdout_value,
        "val_group_col": val_group_col,
        "splits": [
            {"split": "train", "n": int(len(df_train)), "pos": int(df_train["label"].sum()), "neg": int((1 - df_train["label"]).sum())},
            {"split": "val", "n": int(len(df_val)), "pos": int(df_val["label"].sum()), "neg": int((1 - df_val["label"]).sum())},
            {"split": "test", "n": int(len(df_test)), "pos": int(df_test["label"].sum()), "neg": int((1 - df_test["label"]).sum())},
        ],
    }
    return df_train, df_val, df_test, stats


# Final API overrides used by the active baseline scripts. They are placed after
# the legacy holdout wrapper so old definitions cannot shadow them.
def write_fasta(df: pd.DataFrame, path: os.PathLike, id_prefix: str = "seq") -> pd.DataFrame:
    rows = []
    with open(path, "w", encoding="utf-8") as f:
        for i, row in df.reset_index(drop=True).iterrows():
            seq_id = f"{id_prefix}_{i:06d}_y{int(row['label'])}"
            seq = clean_sequence(row["seq"])
            f.write(f">{seq_id}\n")
            for j in range(0, len(seq), 80):
                f.write(seq[j : j + 80] + "\n")
            rows.append({"seq_id": seq_id, "row_index": i, "label": int(row["label"])})
    return pd.DataFrame(rows)


def _final_split_stats_dict(name: str, df: pd.DataFrame, unit_col: str) -> Dict:
    return {
        "split": name,
        "n": int(len(df)),
        "pos": int(df["label"].sum()),
        "neg": int((1 - df["label"]).sum()),
        "pos_rate": float(df["label"].mean()),
        f"unique_{unit_col}": int(df[unit_col].nunique()) if unit_col in df.columns else np.nan,
    }


def split_three_way_by_unit(
    df_all: pd.DataFrame,
    unit_col: str = "species_key",
    train_frac: float = STANDARD_TRAIN_FRAC,
    val_frac: float = STANDARD_VAL_FRAC,
    test_frac: float = STANDARD_TEST_FRAC,
    random_state: int = SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    if abs(train_frac + val_frac + test_frac - 1.0) >= 1e-8:
        raise ValueError("train/val/test fractions must sum to 1.0")
    units = df_all[unit_col].values
    y = df_all["label"].values
    gss1 = GroupShuffleSplit(n_splits=1, test_size=(1 - train_frac), random_state=random_state)
    train_idx, temp_idx = next(gss1.split(df_all, y, groups=units))
    df_train = df_all.iloc[train_idx].reset_index(drop=True)
    df_temp = df_all.iloc[temp_idx].reset_index(drop=True)
    relative_test = test_frac / (val_frac + test_frac)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=relative_test, random_state=random_state + 1)
    val_idx, test_idx = next(gss2.split(df_temp, df_temp["label"].values, groups=df_temp[unit_col].values))
    df_val = df_temp.iloc[val_idx].reset_index(drop=True)
    df_test = df_temp.iloc[test_idx].reset_index(drop=True)
    log_split_stats("train", df_train, unit_col)
    log_split_stats("val", df_val, unit_col)
    log_split_stats("test", df_test, unit_col)
    ensure_split_has_both_labels("train", df_train)
    ensure_split_has_both_labels("val", df_val)
    ensure_split_has_both_labels("test", df_test)
    strict_overlap_check(df_train, df_val, df_test, split_unit_col=unit_col)
    stats = {
        "split_unit_col": unit_col,
        "splits": [
            _final_split_stats_dict("train", df_train, unit_col),
            _final_split_stats_dict("val", df_val, unit_col),
            _final_split_stats_dict("test", df_test, unit_col),
        ],
    }
    return df_train, df_val, df_test, stats


def split_holdout_unit(
    df_all: pd.DataFrame,
    holdout_unit_col: str,
    holdout_value: str,
    val_group_col: str = "species_key",
    val_frac_from_rest: float = OOD_VAL_FRAC_FROM_REST,
    random_state: int = SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    df_test = df_all[df_all[holdout_unit_col] == holdout_value].reset_index(drop=True)
    df_rest = df_all[df_all[holdout_unit_col] != holdout_value].reset_index(drop=True)
    if len(df_test) == 0:
        raise ValueError(f"empty holdout: {holdout_unit_col}={holdout_value}")
    gss = GroupShuffleSplit(n_splits=1, test_size=val_frac_from_rest, random_state=random_state)
    train_idx, val_idx = next(gss.split(df_rest, df_rest["label"].values, groups=df_rest[val_group_col].values))
    df_train = df_rest.iloc[train_idx].reset_index(drop=True)
    df_val = df_rest.iloc[val_idx].reset_index(drop=True)
    log_split_stats("train", df_train, holdout_unit_col)
    log_split_stats("val", df_val, val_group_col)
    log_split_stats("test", df_test, holdout_unit_col)
    ensure_split_has_both_labels("train", df_train)
    ensure_split_has_both_labels("val", df_val)
    ensure_split_has_both_labels("test", df_test)
    strict_overlap_check(df_train, df_val, df_test, split_unit_col=holdout_unit_col)
    stats = {
        "holdout_unit_col": holdout_unit_col,
        "holdout_value": holdout_value,
        "val_group_col": val_group_col,
        "splits": [
            _final_split_stats_dict("train", df_train, holdout_unit_col),
            _final_split_stats_dict("val", df_val, val_group_col),
            _final_split_stats_dict("test", df_test, holdout_unit_col),
        ],
    }
    return df_train, df_val, df_test, stats
