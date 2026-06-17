# -*- coding: utf-8 -*-

import os
import gc
import re
import sys
import math
import json
import random
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_curve, auc, precision_recall_curve,
    average_precision_score, confusion_matrix
)
from sklearn.exceptions import UndefinedMetricWarning
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# ============================================================
# 0. 基础配置
# ============================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if torch.cuda.is_available():
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

# ===== 路径（按你当前代码默认值填写） =====
MODEL_DIR = r"C:\Users\GQ\.cache\huggingface\modules\transformers_modules\DNABERT-S"
POS_FILE = r"E:\4PythonProject\SeqPatho\Dataset\Patho_plantlen2000.csv"
NEG_FILE = r"E:\4PythonProject\SeqPatho\Dataset\Nonpatho_plants_only.csv"

# 可选：外部 taxonomy/clade 映射文件
# 推荐准备一个 CSV：tax_id,species_name,clade_label
# 若文件不存在，脚本会自动回退到 species_name 启发式规则
TAXONOMY_MAP_FILE = r"E:\4PythonProject\SeqPatho\Dataset\taxonomy_map.csv"

# ===== 序列与模型 =====
MAX_LEN_TRAIN = 500
MAX_LEN_EVAL = 500
K_MER = 6
HIDDEN_SIZE = 64
NUM_CLASSES = 2
FREEZE_DNABERT = False

# ===== 训练 =====
BATCH_SIZE = 64
VAL_BATCH_SIZE = 2
TEST_BATCH_SIZE = 2
EPOCHS_STANDARD = 150
EPOCHS_OOD = 60
USE_AMP = torch.cuda.is_available()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TRAIN_MAX_TOKENS_PER_STEP = 20_000
LEARNING_RATE_ENC = 5e-6
LEARNING_RATE_HEAD = 2e-4
WEIGHT_DECAY_ENC = 0.01
WEIGHT_DECAY_HEAD = 2e-4
EARLY_STOP_PATIENCE = 8
EARLY_STOP_DELTA = 0.001

# ===== 评估 =====
EVAL_STRIDE_TOKENS = MAX_LEN_EVAL
EVAL_MAX_TOKENS_PER_CHUNK = 256_000
EVAL_MAX_WINDOWS_PER_SEQ = 512
FLUSH_GPU_CACHE = False

# ===== OOD 开关 =====
RUN_STANDARD_SPECIES_DISJOINT = True
RUN_LOSO_SPECIES = True
RUN_LOCO_CLADE = True

# ===== 数据集加速筛选 =====
LIMIT_DATASET_TO_TOPK_SPECIES = True
DATASET_TOPK_SPECIES = 10
DATASET_TOPK_REQUIRE_BOTH_LABELS = True

# ===== OOD 候选筛选 =====
OOD_SPECIES_TOPK = 10
OOD_SPECIES_MIN_TOTAL = 80
OOD_SPECIES_MIN_POS = 10
OOD_SPECIES_MIN_NEG = 10

OOD_CLADE_MIN_TOTAL = 100
OOD_CLADE_MIN_POS = 20
OOD_CLADE_MIN_NEG = 20
OOD_CLADE_ALLOW = None

# ===== 标准基准切分 =====
STANDARD_TRAIN_FRAC = 0.70
STANDARD_VAL_FRAC = 0.15
STANDARD_TEST_FRAC = 0.15
assert abs(STANDARD_TRAIN_FRAC + STANDARD_VAL_FRAC + STANDARD_TEST_FRAC - 1.0) < 1e-8

# ===== 验证集切分（LOSO/LOCO） =====
OOD_VAL_FRAC_FROM_REST = 0.15

print(f"使用设备: {DEVICE} (AMP={'On' if USE_AMP else 'Off'})")

# ============================================================
# 1. 工具函数
# ============================================================
VALID_DNA_CHARS = set("ACGTN")
_RC_TABLE = str.maketrans("ACGTN", "TGCAN")


def clean_sequence(seq: str) -> str:
    s = str(seq).strip().upper()
    if not s:
        return ""
    return "".join(ch if ch in VALID_DNA_CHARS else "N" for ch in s)


def rev_comp(seq: str) -> str:
    return str(seq).translate(_RC_TABLE)[::-1]


def canon_rc(seq: str) -> str:
    s = clean_sequence(seq)
    if not s:
        return ""
    rc = rev_comp(s)
    return s if s <= rc else rc


def safe_str(x, na="NA"):
    if pd.isna(x):
        return na
    s = str(x).strip()
    return s if s else na


def normalize_species_name(x: str) -> str:
    s = safe_str(x, na="Unknown")
    s = re.sub(r"\s+", " ", s.strip())
    return s


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


def _find_first_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def make_windows(seq, L_tokens=MAX_LEN_EVAL, k=K_MER, stride_tokens=EVAL_STRIDE_TOKENS):
    s = str(seq).strip().upper()
    wn = L_tokens + k - 1
    if len(s) <= wn:
        return [s]
    step_nt = stride_tokens + k - 1
    out = [s[i:i + wn] for i in range(0, len(s) - wn + 1, step_nt)]
    if (len(s) - wn) % step_nt != 0:
        out.append(s[-wn:])
    return out


def sample_train_window(seq, L_tokens=MAX_LEN_TRAIN, k=K_MER):
    s = str(seq).strip().upper()
    wn = L_tokens + k - 1
    if len(s) <= wn:
        return s
    st = random.randint(0, len(s) - wn)
    return s[st:st + wn]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def save_df(df: pd.DataFrame, out_csv: str, out_md: str = None):
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    if out_md is not None:
        try:
            with open(out_md, "w", encoding="utf-8") as f:
                f.write(df.to_markdown(index=False))
                f.write("\n")
        except Exception:
            pass


def safe_float(x, default=np.nan):
    try:
        if x is None:
            return default
        if isinstance(x, (list, tuple)):
            return float(x[0]) if len(x) > 0 else default
        if hasattr(x, "item"):
            return float(x.item())
        return float(x)
    except Exception:
        return default


def fmt4(x):
    x = safe_float(x)
    return f"{x:.4f}" if np.isfinite(x) else "nan"


def fmt2e(x):
    x = safe_float(x)
    return f"{x:.2e}" if np.isfinite(x) else "nan"


# ============================================================
# 2. clade 映射
# ============================================================
OOMYCETE_KEYWORDS = [
    "phytophthora", "pythium", "peronospora", "plasmopara", "hyaloperonospora",
    "albugo", "downy mildew", "bremia", "phanerochaete"
]
FUNGI_KEYWORDS = [
    "fusarium", "magnaporthe", "botrytis", "colletotrichum", "verticillium",
    "ustilago", "puccinia", "rhizoctonia", "alternaria", "sclerotinia",
    "cladosporium", "epenicillium", "aspergillus", "penicillium", "candida",
    "cercospora", "blumeria", "leptosphaeria", "zymoseptoria", "mycosphaerella",
    "curvularia", "bipolaris", "setosphaeria", "septoria", "epichloe"
]
BACTERIA_KEYWORDS = [
    "xanthomonas", "pseudomonas", "ralstonia", "erwinia", "pectobacterium",
    "agrobacterium", "acidovorax", "burkholderia", "clavibacter", "streptomyces",
    "dickeya", "xylella", "rhizobium", "bacillus", "escherichia", "salmonella"
]


def infer_clade_from_species_name(species_name: str) -> str:
    s = normalize_species_name(species_name).lower()
    if s in {"unknown", "na", "-"}:
        return "Unknown"
    for kw in OOMYCETE_KEYWORDS:
        if kw in s:
            return "Oomycete"
    for kw in BACTERIA_KEYWORDS:
        if kw in s:
            return "Bacteria"
    for kw in FUNGI_KEYWORDS:
        if kw in s:
            return "Fungi"
    return "Other/Unknown"


def load_taxonomy_map(taxonomy_map_file: str):
    if not taxonomy_map_file or (not os.path.exists(taxonomy_map_file)):
        return None
    df = pd.read_csv(taxonomy_map_file, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    if "clade_label" not in df.columns:
        raise ValueError("taxonomy_map_file 必须包含 clade_label 列")
    if "tax_id" in df.columns:
        df["tax_id"] = df["tax_id"].map(normalize_tax_id)
    if "species_name" in df.columns:
        df["species_name"] = df["species_name"].map(normalize_species_name)
    return df


def apply_clade_mapping(df_all: pd.DataFrame, taxonomy_map_file: str = None) -> pd.DataFrame:
    df_all = df_all.copy()
    df_all["clade_label"] = ""
    taxmap = load_taxonomy_map(taxonomy_map_file)

    if taxmap is not None:
        if "tax_id" in taxmap.columns:
            tmp_tax = taxmap[["tax_id", "clade_label"]].dropna().drop_duplicates()
            df_all = df_all.merge(tmp_tax, on="tax_id", how="left", suffixes=("", "_taxmap_tax"))
            if "clade_label_taxmap_tax" in df_all.columns:
                df_all["clade_label"] = df_all["clade_label_taxmap_tax"].fillna(df_all["clade_label"])
                df_all.drop(columns=["clade_label_taxmap_tax"], inplace=True)
        if "species_name" in taxmap.columns:
            tmp_sp = taxmap[["species_name", "clade_label"]].dropna().drop_duplicates()
            df_all = df_all.merge(tmp_sp, on="species_name", how="left", suffixes=("", "_taxmap_sp"))
            if "clade_label_taxmap_sp" in df_all.columns:
                need_fill = df_all["clade_label"].isna() | (df_all["clade_label"].astype(str).str.len() == 0)
                df_all.loc[need_fill, "clade_label"] = df_all.loc[need_fill, "clade_label_taxmap_sp"]
                df_all.drop(columns=["clade_label_taxmap_sp"], inplace=True)

    need_heuristic = df_all["clade_label"].isna() | (df_all["clade_label"].astype(str).str.len() == 0)
    df_all.loc[need_heuristic, "clade_label"] = df_all.loc[need_heuristic, "species_name"].map(infer_clade_from_species_name)
    df_all["clade_label"] = df_all["clade_label"].fillna("Unknown").astype(str)
    return df_all


# ============================================================
# 3. 数据加载 + 去重 + 元数据整理
# ============================================================
def load_combined_data(file_a, file_b, taxonomy_map_file=None, random_state=SEED) -> pd.DataFrame:
    df_a = pd.read_csv(file_a, low_memory=False)
    df_b = pd.read_csv(file_b, low_memory=False)

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
    print(f"载入原始序列: 正类{len(pos)} / 负类{len(neg)}，总计{len(df_all)}；将进行清洗和 canonical RC 去重")

    df_all["seq"] = df_all["seq_raw"].map(clean_sequence)
    df_all["seq_len"] = df_all["seq"].str.len()
    df_all = df_all[df_all["seq_len"] > 0].copy()
    df_all["canon"] = df_all["seq"].map(canon_rc)
    df_all["tax_id"] = df_all.get("tax_id", pd.Series(["NA"] * len(df_all))).map(normalize_tax_id)
    df_all["species_name"] = df_all.get("species_name", pd.Series(["Unknown"] * len(df_all))).map(normalize_species_name)
    if "comp_id" in df_all.columns:
        df_all["comp_id"] = df_all["comp_id"].map(lambda x: safe_str(x, na="NA"))
    else:
        df_all["comp_id"] = "NA"

    n_before = len(df_all)
    conflict_groups = df_all.groupby("canon")["label"].nunique()
    n_conflict = int((conflict_groups > 1).sum())
    if n_conflict > 0:
        print(f"[Dedup] 发现 {n_conflict} 个 canonical 组同时包含正负标签，优先保留正类。")
        df_all = df_all.sort_values(by="label", ascending=False)
    df_all = df_all.drop_duplicates(subset="canon", keep="first").reset_index(drop=True)
    print(f"[Dedup] canonical RC 去重: {n_before} -> {len(df_all)}")

    def _build_group_id(row):
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

    df_all["group_id"] = df_all.apply(_build_group_id, axis=1)

    def _build_species_key(row):
        tax = row["tax_id"]
        sp = row["species_name"]
        if tax != "NA" and sp != "Unknown":
            return f"{tax}|{sp}"
        if tax != "NA":
            return f"{tax}|Unknown"
        return f"NA|{sp}"

    df_all["species_key"] = df_all.apply(_build_species_key, axis=1)
    df_all["species_label"] = df_all["species_key"].map(species_label_from_key)

    df_all = apply_clade_mapping(df_all, taxonomy_map_file=taxonomy_map_file)
    df_all["clade_key"] = df_all["clade_label"].fillna("Unknown").astype(str)
    df_all["label"] = df_all["label"].astype(int)

    print(f"[Meta] 唯一 group_id 数量: {df_all['group_id'].nunique()}")
    print(f"[Meta] 唯一 species_key 数量: {df_all['species_key'].nunique()}")
    print(f"[Meta] clade 分布:\n{df_all['clade_key'].value_counts(dropna=False).to_string()}\n")
    return df_all


# ============================================================
# 4. 泄漏检查与切分
# ============================================================
def log_split_stats(name: str, df: pd.DataFrame, unit_col: str = None):
    pos = int(df["label"].sum())
    neg = int((1 - df["label"]).sum())
    msg = f"[{name}] n={len(df)}, pos={pos}, neg={neg}, pos_rate={df['label'].mean():.4f}"
    if unit_col is not None and unit_col in df.columns:
        msg += f", unique_{unit_col}={df[unit_col].nunique()}"
    print(msg)


def ensure_split_has_both_labels(name: str, df: pd.DataFrame):
    labels = sorted(df["label"].dropna().astype(int).unique().tolist())
    if len(labels) < 2:
        raise ValueError(
            f"{name} 仅包含类别 {labels}。当前前 K 物种筛选过窄，请增大 DATASET_TOPK_SPECIES 或关闭 DATASET_TOPK_REQUIRE_BOTH_LABELS。"
        )


def strict_overlap_check(df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame, split_unit_col: str = None):
    c_tr, c_va, c_te = set(df_train["canon"]), set(df_val["canon"]), set(df_test["canon"])
    inter_c = {"train_val": len(c_tr & c_va), "train_test": len(c_tr & c_te), "val_test": len(c_va & c_te)}
    print(f"[LeakCheck] canonical 交叉: {inter_c}")
    assert all(v == 0 for v in inter_c.values()), "检测到 canonical 序列跨集重复"

    g_tr, g_va, g_te = set(df_train["group_id"]), set(df_val["group_id"]), set(df_test["group_id"])
    inter_g = {"train_val": len(g_tr & g_va), "train_test": len(g_tr & g_te), "val_test": len(g_va & g_te)}
    print(f"[LeakCheck] group_id 交叉: {inter_g}")
    assert all(v == 0 for v in inter_g.values()), "检测到 group_id 跨集重复"

    if split_unit_col is not None:
        u_tr, u_va, u_te = set(df_train[split_unit_col]), set(df_val[split_unit_col]), set(df_test[split_unit_col])
        inter_u = {"train_val": len(u_tr & u_va), "train_test": len(u_tr & u_te), "val_test": len(u_va & u_te)}
        print(f"[LeakCheck] {split_unit_col} 交叉: {inter_u}")
        assert all(v == 0 for v in inter_u.values()), f"检测到 {split_unit_col} 跨集重复"


def split_three_way_by_unit(df_all: pd.DataFrame, unit_col: str,
                            train_frac=0.70, val_frac=0.15, test_frac=0.15,
                            random_state=SEED):
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-8
    units = df_all[unit_col].values
    y = df_all["label"].values

    gss1 = GroupShuffleSplit(n_splits=1, test_size=(1 - train_frac), random_state=random_state)
    train_idx, temp_idx = next(gss1.split(df_all, y, groups=units))
    df_train = df_all.iloc[train_idx].reset_index(drop=True)
    df_temp = df_all.iloc[temp_idx].reset_index(drop=True)

    temp_units = df_temp[unit_col].values
    temp_y = df_temp["label"].values
    relative_test = test_frac / (val_frac + test_frac)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=relative_test, random_state=random_state + 1)
    val_idx, test_idx = next(gss2.split(df_temp, temp_y, groups=temp_units))
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


def split_holdout_unit(df_all: pd.DataFrame, holdout_unit_col: str, holdout_value: str,
                       val_group_col: str = "species_key", val_frac_from_rest: float = 0.15,
                       random_state: int = SEED):
    df_test = df_all[df_all[holdout_unit_col] == holdout_value].reset_index(drop=True)
    df_rest = df_all[df_all[holdout_unit_col] != holdout_value].reset_index(drop=True)

    if len(df_test) == 0:
        raise ValueError(f"holdout {holdout_unit_col}={holdout_value} 在数据中不存在")

    rest_groups = df_rest[val_group_col].values
    rest_y = df_rest["label"].values
    unique_rest_units = pd.Series(rest_groups).nunique()
    if unique_rest_units < 2:
        raise ValueError(f"剩余数据在 {val_group_col} 上不足以继续切 train/val：{unique_rest_units}")

    gss = GroupShuffleSplit(n_splits=1, test_size=val_frac_from_rest, random_state=random_state)
    train_idx, val_idx = next(gss.split(df_rest, rest_y, groups=rest_groups))
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


# ============================================================
# 5. Dataset
# ============================================================
class SeqDataset(Dataset):
    def __init__(self, df: pd.DataFrame, train=True, L_tokens_train=MAX_LEN_TRAIN):
        self.df = df.reset_index(drop=True).copy()
        self.train = train
        self.L_tokens_train = L_tokens_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = row["seq"]
        y = int(row["label"])
        if self.train:
            seq = sample_train_window(seq, L_tokens=self.L_tokens_train)
        return seq, y


def collate_fn(batch):
    seqs, ys = zip(*batch)
    return list(seqs), torch.tensor(ys, dtype=torch.long)


def make_loader(df: pd.DataFrame, train: bool, batch_size: int):
    return DataLoader(
        SeqDataset(df, train=train, L_tokens_train=MAX_LEN_TRAIN),
        batch_size=batch_size,
        shuffle=train,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        num_workers=0
    )


# ============================================================
# 6. 模型
# ============================================================
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super().__init__()
        hidden = max(1, in_channels // reduction_ratio)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_perm = x.permute(0, 2, 1)
        avg_out = self.fc(self.avg_pool(x_perm).squeeze(-1))
        max_out = self.fc(self.max_pool(x_perm).squeeze(-1))
        scale = self.sigmoid(avg_out + max_out).unsqueeze(-1)
        out = x_perm * scale
        return out.permute(0, 2, 1)


class StackedChannelAttention(nn.Module):
    def __init__(self, in_channels, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList([ChannelAttention(in_channels) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(in_channels) for _ in range(num_layers)])

    def forward(self, x):
        for attn, norm in zip(self.layers, self.norms):
            residual = x
            x = attn(x)
            x = norm(x + residual)
        return x


class DNABiGRUWithDNABERT(nn.Module):
    def __init__(self, dnabert_dir, num_classes=2, hidden_size=64, freeze_dnabert=False):
        super().__init__()
        self.dnabert_model = AutoModel.from_pretrained(dnabert_dir, trust_remote_code=True)
        self.bert_hidden_size = 768
        self.freeze_dnabert = freeze_dnabert

        for name, param in self.dnabert_model.named_parameters():
            if "encoder.layer" in name:
                layer_num = int(name.split("encoder.layer.")[1].split(".")[0])
                param.requires_grad = (layer_num >= 10) and (not freeze_dnabert)
            else:
                param.requires_grad = False

        if hasattr(self.dnabert_model, "gradient_checkpointing_enable"):
            try:
                self.dnabert_model.gradient_checkpointing_enable()
            except Exception:
                pass
        if hasattr(self.dnabert_model.config, "use_cache"):
            try:
                self.dnabert_model.config.use_cache = False
            except Exception:
                pass

        self.bigru = nn.GRU(
            input_size=self.bert_hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True
        )
        self.channel_attention = StackedChannelAttention(in_channels=hidden_size * 2, num_layers=3)
        self.layernorm = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(hidden_size * 2, 32)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, input_ids, attention_mask=None):
        if self.freeze_dnabert:
            with torch.no_grad():
                bert_output = self.dnabert_model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            bert_output = self.dnabert_model(input_ids=input_ids, attention_mask=attention_mask)

        embeddings = bert_output[0]
        seq_lengths = attention_mask.sum(dim=1) if attention_mask is not None else (input_ids != 0).sum(dim=1)
        packed = nn.utils.rnn.pack_padded_sequence(embeddings, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.bigru(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True, total_length=embeddings.size(1))

        output = self.channel_attention(output)
        output = self.layernorm(output)
        output = self.dropout(output)

        L = output.size(1)
        mask = attention_mask[:, :L].unsqueeze(-1).expand(-1, -1, output.size(2)).float()
        masked_output = output * mask
        sum_output = masked_output.sum(dim=1)
        avg_output = sum_output / (mask.sum(dim=1).clamp_min(1e-6))

        x = F.relu(self.fc1(avg_output))
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits


def build_model():
    model = DNABiGRUWithDNABERT(
        dnabert_dir=MODEL_DIR,
        num_classes=NUM_CLASSES,
        hidden_size=HIDDEN_SIZE,
        freeze_dnabert=FREEZE_DNABERT,
    ).to(DEVICE)

    max_pos = getattr(model.dnabert_model.config, "max_position_embeddings", MAX_LEN_EVAL)
    if MAX_LEN_TRAIN > max_pos or MAX_LEN_EVAL > max_pos:
        print(f"[WARN] max_position_embeddings={max_pos}，请自行将 MAX_LEN_TRAIN / MAX_LEN_EVAL 调小")
    return model


def build_param_groups(model, lr_enc=LEARNING_RATE_ENC, lr_head=LEARNING_RATE_HEAD,
                       wd_enc=WEIGHT_DECAY_ENC, wd_head=WEIGHT_DECAY_HEAD):
    enc_decay, enc_nodecay = [], []
    for n, p in model.dnabert_model.named_parameters():
        if not p.requires_grad:
            continue
        if any(nd in n for nd in ["bias", "LayerNorm.weight", "layer_norm.weight"]):
            enc_nodecay.append(p)
        else:
            enc_decay.append(p)

    head_params = []
    for m in [model.bigru, model.channel_attention, model.layernorm, model.fc1, model.fc2]:
        for p in m.parameters():
            if p.requires_grad:
                head_params.append(p)

    groups = []
    if enc_decay:
        groups.append({"params": enc_decay, "lr": lr_enc, "weight_decay": wd_enc, "group_name": "enc_decay"})
    if enc_nodecay:
        groups.append({"params": enc_nodecay, "lr": lr_enc, "weight_decay": 0.0, "group_name": "enc_nodecay"})
    if head_params:
        groups.append({"params": head_params, "lr": lr_head, "weight_decay": wd_head, "group_name": "head"})
    return groups


# ============================================================
# 7. 训练/评估
# ============================================================
class EarlyStopping:
    def __init__(self, save_path, patience=8, verbose=True, delta=0.001):
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False
        self.best_epoch = 0
        ensure_dir(save_path)

    def __call__(self, val_loss, model, epoch):
        old_best = self.best_loss
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            if self.verbose:
                print(f"验证损失改善: {old_best:.6f} -> {self.best_loss:.6f}. 保存模型...")
            torch.save(model.state_dict(), os.path.join(self.save_path, f"best_model_epoch{epoch + 1}.pth"))
            self.counter = 0
            self.best_epoch = epoch
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def get_group_lrs(optimizer):
    lr_enc, lr_head = None, None
    for g in optimizer.param_groups:
        name = g.get("group_name", "")
        if name.startswith("enc"):
            lr_enc = g["lr"]
        if name == "head":
            lr_head = g["lr"]
    return lr_enc, lr_head


CRITERION = nn.CrossEntropyLoss(label_smoothing=0.05)


@torch.inference_mode()
def evaluate_sequence_loader(model, data_loader, stride_tokens=EVAL_STRIDE_TOKENS,
                             use_amp=USE_AMP):
    model.eval()
    seq_loss_sum, n_seq_count = 0.0, 0
    all_preds, all_labels, all_probs = [], [], []
    windows_per_chunk = max(1, EVAL_MAX_TOKENS_PER_CHUNK // MAX_LEN_EVAL)
    device_cap_major = torch.cuda.get_device_capability(0)[0] if torch.cuda.is_available() else 0

    for batch_sequences, batch_labels in data_loader:
        for seq, y in zip(batch_sequences, batch_labels.tolist()):
            wins = make_windows(seq, L_tokens=MAX_LEN_EVAL, k=K_MER, stride_tokens=stride_tokens)
            if EVAL_MAX_WINDOWS_PER_SEQ and len(wins) > EVAL_MAX_WINDOWS_PER_SEQ:
                idx = np.linspace(0, len(wins) - 1, EVAL_MAX_WINDOWS_PER_SEQ, dtype=int)
                wins = [wins[i] for i in idx]
            if len(wins) == 0:
                continue

            logits_sum = torch.zeros(NUM_CLASSES, device=DEVICE, dtype=torch.float32)
            n_win = 0
            for off in range(0, len(wins), windows_per_chunk):
                chunk = wins[off:off + windows_per_chunk]
                tok = TOKENIZER(chunk, padding="longest", truncation=True,
                                max_length=MAX_LEN_EVAL, return_tensors="pt")
                input_ids = tok["input_ids"].to(DEVICE, non_blocking=True)
                attention_mask = tok["attention_mask"].to(DEVICE, non_blocking=True).long()

                vocab_size = model.dnabert_model.config.vocab_size
                valid_mask = (
                    (attention_mask.sum(dim=1) > 0) &
                    (input_ids.min(dim=1).values >= 0) &
                    (input_ids.max(dim=1).values < vocab_size)
                )
                if valid_mask.sum() == 0:
                    del input_ids, attention_mask
                    if FLUSH_GPU_CACHE and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue

                input_ids = input_ids[valid_mask]
                attention_mask = attention_mask[valid_mask]

                amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and device_cap_major >= 8) else torch.float16
                with autocast(enabled=use_amp, dtype=amp_dtype):
                    logits = model(input_ids, attention_mask)

                logits_sum += logits.to(torch.float32).sum(dim=0)
                n_win += logits.shape[0]

                del input_ids, attention_mask, logits
                if FLUSH_GPU_CACHE and torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if n_win == 0:
                continue

            agg_logits = logits_sum / n_win
            loss_seq = CRITERION(agg_logits.unsqueeze(0), torch.tensor([y], device=DEVICE))
            seq_loss_sum += loss_seq.item()
            n_seq_count += 1

            prob = F.softmax(agg_logits, dim=0)[1].item()
            pred = int(torch.argmax(agg_logits).item())
            all_probs.append(prob)
            all_preds.append(pred)
            all_labels.append(y)

    avg_loss = (seq_loss_sum / max(1, n_seq_count)) if n_seq_count > 0 else float("inf")
    return avg_loss, np.array(all_labels), np.array(all_preds), np.array(all_probs)


def compute_classification_metrics(labels, preds, probs):
    if len(labels) == 0:
        return dict(accuracy=np.nan, precision=np.nan, recall=np.nan, f1=np.nan,
                    mcc=np.nan, auroc=np.nan, auprc=np.nan, ap=np.nan,
                    n=0, pos=0, neg=0)
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    try:
        mcc = matthews_corrcoef(labels, preds)
    except Exception:
        mcc = np.nan
    try:
        fpr, tpr, _ = roc_curve(labels, probs)
        auroc = auc(fpr, tpr)
    except Exception:
        auroc = np.nan
    try:
        prec_vals, rec_vals, _ = precision_recall_curve(labels, probs)
        auprc = auc(rec_vals, prec_vals)
        ap = average_precision_score(labels, probs)
    except Exception:
        auprc, ap = np.nan, np.nan
    return dict(
        accuracy=accuracy, precision=precision, recall=recall, f1=f1,
        mcc=mcc, auroc=auroc, auprc=auprc, ap=ap,
        n=len(labels), pos=int(np.sum(labels == 1)), neg=int(np.sum(labels == 0))
    )


def plot_confusion_matrix(labels, preds, out_png, title="Confusion Matrix"):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def plot_roc_pr(labels, probs, roc_png, pr_png, title_prefix="Test"):
    try:
        fpr, tpr, _ = roc_curve(labels, probs)
        auroc = auc(fpr, tpr)
        plt.figure(figsize=(5, 4))
        plt.plot(fpr, tpr, lw=2, label=f'ROC (AUROC = {auroc:.4f})')
        plt.plot([0, 1], [0, 1], lw=1.5, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{title_prefix} ROC Curve')
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
        plt.step(recall_vals, precision_vals, where='post', lw=2, label=f'PR (AP = {ap:.4f}, AUPRC = {auprc:.4f})')
        plt.fill_between(recall_vals, precision_vals, step='post', alpha=0.2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'{title_prefix} PR Curve')
        plt.legend(loc="upper right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(pr_png, dpi=300)
        plt.close()
    except Exception:
        pass


def train_one_experiment(df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame,
                         run_dir: str, epochs: int, experiment_name: str):
    ensure_dir(run_dir)
    checkpoint_dir = ensure_dir(os.path.join(run_dir, "checkpoints"))

    train_loader = make_loader(df_train, train=True, batch_size=BATCH_SIZE)
    val_loader = make_loader(df_val, train=False, batch_size=VAL_BATCH_SIZE)
    test_loader = make_loader(df_test, train=False, batch_size=TEST_BATCH_SIZE)

    model = build_model()
    param_groups = build_param_groups(model)
    optimizer = optim.AdamW(param_groups, eps=1e-8)
    scaler = GradScaler(enabled=USE_AMP)

    num_update_steps_per_epoch = max(1, len(train_loader))
    num_train_steps = max(1, epochs * num_update_steps_per_epoch)
    num_warmup_steps = max(1, int(0.05 * num_train_steps))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_train_steps)

    early_stopping = EarlyStopping(
        save_path=checkpoint_dir,
        patience=EARLY_STOP_PATIENCE,
        verbose=True,
        delta=EARLY_STOP_DELTA
    )

    history_rows = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        seen_batches = 0
        progress_bar = tqdm(train_loader, desc=f"{experiment_name} | Epoch [{epoch + 1}/{epochs}]", leave=False)

        for batch_sequences, batch_labels in progress_bar:
            batch_labels = batch_labels.to(DEVICE)
            tokenized = TOKENIZER(batch_sequences, padding='longest', truncation=True,
                                  max_length=MAX_LEN_TRAIN, return_tensors='pt')
            input_ids = tokenized['input_ids'].to(DEVICE, non_blocking=True)
            attention_mask = tokenized['attention_mask'].to(DEVICE, non_blocking=True).long()

            vocab_size = model.dnabert_model.config.vocab_size
            valid_mask = (
                (attention_mask.sum(dim=1) > 0) &
                (input_ids.min(dim=1).values >= 0) &
                (input_ids.max(dim=1).values < vocab_size)
            )
            if valid_mask.sum() == 0:
                continue
            input_ids = input_ids[valid_mask]
            attention_mask = attention_mask[valid_mask]
            batch_labels = batch_labels[valid_mask]

            lengths = attention_mask.sum(dim=1)
            N = input_ids.size(0)
            Lmax = int(lengths.max().item())
            sub_bs = max(1, int(TRAIN_MAX_TOKENS_PER_STEP // max(Lmax, 1)))
            num_sub = math.ceil(N / sub_bs)

            optimizer.zero_grad(set_to_none=True)
            loss_accum = 0.0
            for i in range(0, N, sub_bs):
                j = min(i + sub_bs, N)
                ids_sub = input_ids[i:j]
                mask_sub = attention_mask[i:j]
                labels_sub = batch_labels[i:j]

                with autocast(enabled=USE_AMP):
                    logits = model(ids_sub, mask_sub)
                    loss_sub = CRITERION(logits, labels_sub)
                    loss = loss_sub / num_sub

                scaler.scale(loss).backward()
                loss_accum += loss_sub.item()

            scale_before = scaler.get_scale() if USE_AMP else None
            scaler.step(optimizer)
            scaler.update()
            scale_after = scaler.get_scale() if USE_AMP else None

            if (not USE_AMP) or (scale_after is None) or (scale_before is None) or (scale_after >= scale_before):
                scheduler.step()

            total_loss += (loss_accum / num_sub if num_sub > 0 else 0.0)
            seen_batches += 1
            progress_bar.set_postfix(loss=(total_loss / max(1, seen_batches)))

        avg_train_loss = total_loss / max(1, seen_batches)
        val_loss, val_labels, val_preds, val_probs = evaluate_sequence_loader(model, val_loader)
        val_metrics = compute_classification_metrics(val_labels, val_preds, val_probs)
        lr_enc, lr_head = get_group_lrs(optimizer)

        epoch_row = {
            "epoch": epoch + 1,
            "train_loss": safe_float(avg_train_loss),
            "val_loss": safe_float(val_loss),
            "accuracy": safe_float(val_metrics.get("accuracy")),
            "precision": safe_float(val_metrics.get("precision")),
            "recall": safe_float(val_metrics.get("recall")),
            "f1": safe_float(val_metrics.get("f1")),
            "mcc": safe_float(val_metrics.get("mcc")),
            "auroc": safe_float(val_metrics.get("auroc")),
            "auprc": safe_float(val_metrics.get("auprc")),
            "ap": safe_float(val_metrics.get("ap")),
            "n": int(val_metrics.get("n", 0)),
            "pos": int(val_metrics.get("pos", 0)),
            "neg": int(val_metrics.get("neg", 0)),
            "lr_enc": safe_float(lr_enc),
            "lr_head": safe_float(lr_head),
        }
        history_rows.append(epoch_row)

        lr_enc_str = fmt2e(lr_enc)
        lr_head_str = fmt2e(lr_head)

        print(
            f"{experiment_name} | Epoch [{epoch + 1}/{epochs}] - "
            f"Train Loss: {fmt4(avg_train_loss)}, Val Loss: {fmt4(val_loss)}, "
            f"Acc: {fmt4(val_metrics.get('accuracy'))}, "
            f"P: {fmt4(val_metrics.get('precision'))}, "
            f"R: {fmt4(val_metrics.get('recall'))}, "
            f"F1: {fmt4(val_metrics.get('f1'))}, "
            f"MCC: {fmt4(val_metrics.get('mcc'))}, "
            f"AUROC: {fmt4(val_metrics.get('auroc'))}, "
            f"AUPRC: {fmt4(val_metrics.get('auprc'))}, "
            f"AP: {fmt4(val_metrics.get('ap'))}, "
            f"LR_Enc: {lr_enc_str}, LR_Head: {lr_head_str}"
        )

        early_stopping(val_loss, model, epoch)
        if early_stopping.early_stop:
            print(f"{experiment_name} | 早停触发于第 {epoch + 1} 轮，最佳模型来自第 {early_stopping.best_epoch + 1} 轮")
            break

    history_df = pd.DataFrame(history_rows)
    save_df(history_df, os.path.join(run_dir, "history.csv"), os.path.join(run_dir, "history.md"))

    best_ckpt_path = os.path.join(checkpoint_dir, f"best_model_epoch{early_stopping.best_epoch + 1}.pth")
    if os.path.exists(best_ckpt_path):
        model.load_state_dict(torch.load(best_ckpt_path, map_location=DEVICE))
        print(f"[{experiment_name}] 加载最佳验证轮权重：{best_ckpt_path}")
    else:
        print(f"[{experiment_name}] 未找到最佳权重文件，使用当前模型参数进行测试。")

    test_loss, test_labels, test_preds, test_probs = evaluate_sequence_loader(model, test_loader)
    test_metrics = compute_classification_metrics(test_labels, test_preds, test_probs)

    with open(os.path.join(run_dir, "final_test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({
            "experiment_name": experiment_name,
            "test_loss": test_loss,
            **test_metrics,
            "best_epoch": int(early_stopping.best_epoch + 1),
            "early_stop": bool(early_stopping.early_stop),
        }, f, ensure_ascii=False, indent=2)

    plot_confusion_matrix(test_labels, test_preds, os.path.join(run_dir, "confusion_matrix_test.png"),
                          title=f"{experiment_name} | Confusion Matrix")
    plot_roc_pr(test_labels, test_probs,
                os.path.join(run_dir, "roc_curve_test.png"),
                os.path.join(run_dir, "pr_curve_test.png"),
                title_prefix=experiment_name)

    if len(history_df) > 0:
        plt.figure(figsize=(6, 4))
        plt.plot(history_df["epoch"], history_df["train_loss"], label="Train Loss")
        plt.plot(history_df["epoch"], history_df["val_loss"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{experiment_name} | Train/Val Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "loss_curve.png"), dpi=300)
        plt.close()

        for metric in ["accuracy", "precision", "recall", "f1", "mcc", "auroc", "auprc"]:
            plt.figure(figsize=(6, 4))
            plt.plot(history_df["epoch"], history_df[metric])
            plt.xlabel("Epoch")
            plt.ylabel(metric.upper())
            plt.title(f"{experiment_name} | Val {metric.upper()} over Epochs")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(run_dir, f"val_{metric}_curve.png"), dpi=300)
            plt.close()

    result = {
        "experiment_name": experiment_name,
        "best_epoch": int(early_stopping.best_epoch + 1),
        "test_loss": test_loss,
        **test_metrics,
        "n_train": len(df_train),
        "n_val": len(df_val),
        "n_test": len(df_test),
        "train_pos": int(df_train["label"].sum()),
        "train_neg": int((1 - df_train["label"]).sum()),
        "val_pos": int(df_val["label"].sum()),
        "val_neg": int((1 - df_val["label"]).sum()),
        "test_pos": int(df_test["label"].sum()),
        "test_neg": int((1 - df_test["label"]).sum()),
        "run_dir": run_dir,
    }

    del train_loader, val_loader, test_loader, optimizer, scheduler, scaler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


# ============================================================
# 8. OOD 候选清单与可视化
# ============================================================
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


def restrict_dataset_to_top_species(df_all: pd.DataFrame, top_k: int,
                                    require_both_labels: bool = True):
    manifest = build_unit_manifest(df_all, "species_key")
    selected = manifest.copy()
    if require_both_labels:
        selected = selected[(selected["n_pos"] > 0) & (selected["n_neg"] > 0)].copy()
    selected = selected.sort_values(["n_total", "n_pos"], ascending=[False, False]).reset_index(drop=True)

    if top_k is not None:
        selected = selected.head(top_k).copy()
    if len(selected) == 0:
        raise ValueError("前 K 物种筛选后没有可用数据，请检查 DATASET_TOPK_* 配置。")

    selected.insert(0, "rank", np.arange(1, len(selected) + 1))
    keep_species = selected["species_key"].tolist()
    filtered = df_all[df_all["species_key"].isin(keep_species)].reset_index(drop=True)

    print(
        f"[TopSpecies] 保留前 {len(keep_species)} 个物种，总样本 {len(df_all)} -> {len(filtered)}；require_both_labels={require_both_labels}"
    )
    print("[TopSpecies] 选中的物种:")
    cols_to_show = ["rank", "species_label", "n_total", "n_pos", "n_neg"] if "species_label" in selected.columns else ["rank", "species_key", "n_total", "n_pos", "n_neg"]
    print(selected[cols_to_show].to_string(index=False))

    ensure_split_has_both_labels("filtered_dataset", filtered)
    return filtered, manifest, selected


def filter_species_candidates(manifest: pd.DataFrame) -> pd.DataFrame:
    out = manifest[(manifest["n_total"] >= OOD_SPECIES_MIN_TOTAL) &
                   (manifest["n_pos"] >= OOD_SPECIES_MIN_POS) &
                   (manifest["n_neg"] >= OOD_SPECIES_MIN_NEG)].copy()
    out = out.sort_values(["n_total", "n_pos"], ascending=[False, False]).reset_index(drop=True)
    if OOD_SPECIES_TOPK is not None:
        out = out.head(OOD_SPECIES_TOPK).copy()
    return out


def filter_clade_candidates(manifest: pd.DataFrame) -> pd.DataFrame:
    out = manifest[(manifest["n_total"] >= OOD_CLADE_MIN_TOTAL) &
                   (manifest["n_pos"] >= OOD_CLADE_MIN_POS) &
                   (manifest["n_neg"] >= OOD_CLADE_MIN_NEG)].copy()
    if OOD_CLADE_ALLOW is not None:
        out = out[out["clade_key"].isin(OOD_CLADE_ALLOW)].copy()
    out = out.sort_values(["n_total", "n_pos"], ascending=[False, False]).reset_index(drop=True)
    return out


def plot_unit_manifest(manifest_df: pd.DataFrame, unit_col: str, out_png: str, title: str, top_n: int = 20):
    show_df = manifest_df.head(top_n).copy()
    if len(show_df) == 0:
        return

    label_col = unit_col
    ylabel = unit_col
    if unit_col == "species_key" and "species_label" in show_df.columns:
        label_col = "species_label"
        ylabel = "Species"
    elif unit_col == "clade_key" and "clade_label_display" in show_df.columns:
        label_col = "clade_label_display"
        ylabel = "Clade"

    plt.figure(figsize=(10, max(4, 0.4 * len(show_df))))
    ax = sns.barplot(data=show_df, y=label_col, x="n_total")
    for i, v in enumerate(show_df["n_total"].tolist()):
        ax.text(v + max(1, 0.005 * show_df["n_total"].max()), i, str(int(v)), va='center', fontsize=8)
    plt.title(title)
    plt.xlabel("Number of samples")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def plot_summary_bar(summary_df: pd.DataFrame, unit_col: str, metric: str, out_png: str, title: str, label_col: str = None):
    if len(summary_df) == 0 or metric not in summary_df.columns:
        return
    show_df = summary_df.sort_values(metric, ascending=False).copy()
    display_col = label_col if (label_col is not None and label_col in show_df.columns) else unit_col

    plt.figure(figsize=(10, max(4, 0.45 * len(show_df))))
    sns.barplot(data=show_df, y=display_col, x=metric)
    plt.title(title)
    plt.xlabel(metric.upper())
    plt.ylabel("Species" if "species" in display_col.lower() else unit_col)
    plt.xlim(0, min(1.0, max(0.05, float(show_df[metric].max()) * 1.08)))
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def plot_summary_heatmap(summary_df: pd.DataFrame, unit_col: str, out_png: str, title: str,
                         metrics=("accuracy", "auprc", "auroc", "f1", "mcc", "precision", "recall"),
                         label_col: str = None):
    if len(summary_df) == 0:
        return

    index_col = label_col if (label_col is not None and label_col in summary_df.columns) else unit_col
    keep_cols = [index_col] + [m for m in metrics if m in summary_df.columns]
    plot_df = summary_df[keep_cols].copy().set_index(index_col)

    plt.figure(figsize=(8, max(4, 0.45 * len(plot_df))))
    sns.heatmap(plot_df, annot=True, fmt=".3f", cmap="YlGnBu")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def plot_support_vs_metric(summary_df: pd.DataFrame, unit_col: str, metric: str, out_png: str, title: str, label_col: str = None):
    if len(summary_df) == 0 or metric not in summary_df.columns:
        return

    display_col = label_col if (label_col is not None and label_col in summary_df.columns) else unit_col

    plt.figure(figsize=(6, 5))
    sns.scatterplot(data=summary_df, x="n_test", y=metric, s=80)
    for _, row in summary_df.iterrows():
        plt.text(row["n_test"], row[metric], str(row[display_col]), fontsize=8)
    plt.xlabel("Test samples")
    plt.ylabel(metric.upper())
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def summarize_runs(summary_df: pd.DataFrame, label: str) -> pd.DataFrame:
    metrics = ["accuracy", "precision", "recall", "f1", "mcc", "auroc", "auprc", "ap"]
    out = []
    for m in metrics:
        if m not in summary_df.columns or len(summary_df) == 0:
            continue
        out.append({
            "benchmark": label,
            "metric": m,
            "mean": float(summary_df[m].mean()),
            "std": float(summary_df[m].std(ddof=0)),
            "min": float(summary_df[m].min()),
            "max": float(summary_df[m].max()),
            "median": float(summary_df[m].median()),
            "n_runs": int(summary_df[m].notna().sum())
        })
    return pd.DataFrame(out)


# ============================================================
# 9. 基准实验
# ============================================================
def run_standard_species_disjoint(df_all: pd.DataFrame, root_dir: str):
    out_dir = ensure_dir(os.path.join(root_dir, "standard_species_disjoint"))
    df_train, df_val, df_test = split_three_way_by_unit(
        df_all,
        unit_col="species_key",
        train_frac=STANDARD_TRAIN_FRAC,
        val_frac=STANDARD_VAL_FRAC,
        test_frac=STANDARD_TEST_FRAC,
        random_state=SEED
    )

    result = train_one_experiment(
        df_train, df_val, df_test,
        run_dir=out_dir,
        epochs=EPOCHS_STANDARD,
        experiment_name="standard_species_disjoint"
    )
    return pd.DataFrame([result])


# ============================================================
# 10. LOSO / LOCO
# ============================================================
def run_loso_species(df_all: pd.DataFrame, root_dir: str):
    out_dir = ensure_dir(os.path.join(root_dir, "loso_species"))
    manifest = build_unit_manifest(df_all, "species_key")
    save_df(manifest, os.path.join(out_dir, "species_manifest_all.csv"), os.path.join(out_dir, "species_manifest_all.md"))
    plot_unit_manifest(manifest, "species_key", os.path.join(out_dir, "species_manifest_top20.png"),
                       title="Species support (top 20)", top_n=20)

    candidates = filter_species_candidates(manifest)
    save_df(candidates, os.path.join(out_dir, "species_candidates.csv"), os.path.join(out_dir, "species_candidates.md"))
    plot_unit_manifest(candidates, "species_key", os.path.join(out_dir, "species_candidates.png"),
                       title="LOSO candidate species", top_n=len(candidates))

    summary_rows = []
    for i, row in candidates.iterrows():
        species_key = row["species_key"]
        species_label = row["species_label"] if "species_label" in row else species_label_from_key(species_key)
        run_dir = ensure_dir(os.path.join(out_dir, f"holdout_species_{i+1:02d}"))
        print("\n" + "=" * 100)
        print(f"[LOSO] ({i+1}/{len(candidates)}) Hold-out species: {species_key}")
        print("=" * 100)
        try:
            df_train, df_val, df_test = split_holdout_unit(
                df_all,
                holdout_unit_col="species_key",
                holdout_value=species_key,
                val_group_col="species_key",
                val_frac_from_rest=OOD_VAL_FRAC_FROM_REST,
                random_state=SEED + i
            )
            result = train_one_experiment(
                df_train, df_val, df_test,
                run_dir=run_dir,
                epochs=EPOCHS_OOD,
                experiment_name=f"LOSO::{species_key}"
            )
            result.update({
                "heldout_species": species_key,
                "heldout_species_label": species_label,
                "species_support_total": int(row["n_total"]),
                "species_support_pos": int(row["n_pos"]),
                "species_support_neg": int(row["n_neg"]),
            })
            summary_rows.append(result)
        except Exception as e:
            print(f"[LOSO][ERROR] {species_key}: {e}")
            summary_rows.append({
                "heldout_species": species_key,
                "heldout_species_label": species_label,
                "error": str(e),
                "species_support_total": int(row["n_total"]),
                "species_support_pos": int(row["n_pos"]),
                "species_support_neg": int(row["n_neg"]),
            })

    summary_df = pd.DataFrame(summary_rows)
    save_df(summary_df, os.path.join(out_dir, "loso_species_summary.csv"), os.path.join(out_dir, "loso_species_summary.md"))

    ok_df = summary_df[summary_df.get("error", pd.Series([np.nan] * len(summary_df))).isna()].copy() if len(summary_df) > 0 else pd.DataFrame()
    if len(ok_df) > 0:
        plot_summary_bar(
            ok_df, "heldout_species", "auprc",
            os.path.join(out_dir, "loso_species_auprc_bar.png"),
            title="LOSO species OOD: AUPRC by held-out species",
            label_col="heldout_species_label"
        )
        plot_summary_bar(
            ok_df, "heldout_species", "mcc",
            os.path.join(out_dir, "loso_species_mcc_bar.png"),
            title="LOSO species OOD: MCC by held-out species",
            label_col="heldout_species_label"
        )
        plot_summary_heatmap(
            ok_df, "heldout_species",
            os.path.join(out_dir, "loso_species_metric_heatmap.png"),
            title="LOSO species OOD metric heatmap",
            metrics=("accuracy", "auprc", "auroc", "f1", "mcc", "precision", "recall"),
            label_col="heldout_species_label"
        )
        plot_support_vs_metric(
            ok_df, "heldout_species", "auprc",
            os.path.join(out_dir, "loso_species_support_vs_auprc.png"),
            title="LOSO species OOD: test support vs AUPRC",
            label_col="heldout_species_label"
        )

        stats_df = summarize_runs(ok_df, label="LOSO_species")
        save_df(stats_df, os.path.join(out_dir, "loso_species_overall_stats.csv"), os.path.join(out_dir, "loso_species_overall_stats.md"))

    return summary_df


def run_loco_clade(df_all: pd.DataFrame, root_dir: str):
    out_dir = ensure_dir(os.path.join(root_dir, "loco_clade"))
    manifest = build_unit_manifest(df_all, "clade_key")
    save_df(manifest, os.path.join(out_dir, "clade_manifest_all.csv"), os.path.join(out_dir, "clade_manifest_all.md"))
    plot_unit_manifest(manifest, "clade_key", os.path.join(out_dir, "clade_manifest.png"),
                       title="Clade support", top_n=len(manifest))

    candidates = filter_clade_candidates(manifest)
    save_df(candidates, os.path.join(out_dir, "clade_candidates.csv"), os.path.join(out_dir, "clade_candidates.md"))

    summary_rows = []
    for i, row in candidates.iterrows():
        clade_key = row["clade_key"]
        run_dir = ensure_dir(os.path.join(out_dir, f"holdout_clade_{i+1:02d}_{re.sub(r'[^A-Za-z0-9._-]+', '_', clade_key)}"))
        print("\n" + "=" * 100)
        print(f"[LOCO] ({i+1}/{len(candidates)}) Hold-out clade: {clade_key}")
        print("=" * 100)
        try:
            df_train, df_val, df_test = split_holdout_unit(
                df_all,
                holdout_unit_col="clade_key",
                holdout_value=clade_key,
                val_group_col="species_key",
                val_frac_from_rest=OOD_VAL_FRAC_FROM_REST,
                random_state=SEED + 100 + i
            )
            result = train_one_experiment(
                df_train, df_val, df_test,
                run_dir=run_dir,
                epochs=EPOCHS_OOD,
                experiment_name=f"LOCO::{clade_key}"
            )
            result.update({
                "heldout_clade": clade_key,
                "clade_support_total": int(row["n_total"]),
                "clade_support_pos": int(row["n_pos"]),
                "clade_support_neg": int(row["n_neg"]),
            })
            summary_rows.append(result)
        except Exception as e:
            print(f"[LOCO][ERROR] {clade_key}: {e}")
            summary_rows.append({
                "heldout_clade": clade_key,
                "error": str(e),
                "clade_support_total": int(row["n_total"]),
                "clade_support_pos": int(row["n_pos"]),
                "clade_support_neg": int(row["n_neg"]),
            })

    summary_df = pd.DataFrame(summary_rows)
    save_df(summary_df, os.path.join(out_dir, "loco_clade_summary.csv"), os.path.join(out_dir, "loco_clade_summary.md"))

    ok_df = summary_df[summary_df.get("error", pd.Series([np.nan] * len(summary_df))).isna()].copy() if len(summary_df) > 0 else pd.DataFrame()
    if len(ok_df) > 0:
        plot_summary_bar(ok_df, "heldout_clade", "auprc", os.path.join(out_dir, "loco_clade_auprc_bar.png"),
                         title="LOCO clade OOD: AUPRC by held-out clade")
        plot_summary_bar(ok_df, "heldout_clade", "mcc", os.path.join(out_dir, "loco_clade_mcc_bar.png"),
                         title="LOCO clade OOD: MCC by held-out clade")
        plot_summary_heatmap(ok_df, "heldout_clade", os.path.join(out_dir, "loco_clade_metric_heatmap.png"),
                             title="LOCO clade OOD metric heatmap",
                             metrics=("accuracy", "auprc", "auroc", "f1", "mcc", "precision", "recall"))
        plot_support_vs_metric(ok_df, "heldout_clade", "auprc", os.path.join(out_dir, "loco_clade_support_vs_auprc.png"),
                               title="LOCO clade OOD: test support vs AUPRC")

        stats_df = summarize_runs(ok_df, label="LOCO_clade")
        save_df(stats_df, os.path.join(out_dir, "loco_clade_overall_stats.csv"), os.path.join(out_dir, "loco_clade_overall_stats.md"))

    return summary_df


# ============================================================
# 11. 总汇总表与总图
# ============================================================
def build_master_benchmark_table(standard_df: pd.DataFrame = None,
                                 loso_df: pd.DataFrame = None,
                                 loco_df: pd.DataFrame = None) -> pd.DataFrame:
    rows = []
    metrics = ["accuracy", "precision", "recall", "f1", "mcc", "auroc", "auprc", "ap"]

    if standard_df is not None and len(standard_df) > 0:
        row = {"benchmark": "Standard species-disjoint", "aggregation": "single_run"}
        for m in metrics:
            if m in standard_df.columns:
                row[m] = float(standard_df.iloc[0][m])
                row[f"{m}_mean"] = float(standard_df.iloc[0][m])
                row[f"{m}_std"] = 0.0
        row["n_runs"] = 1
        rows.append(row)

    if loso_df is not None and len(loso_df) > 0:
        ok = loso_df[loso_df.get("error", pd.Series([np.nan] * len(loso_df))).isna()].copy()
        if len(ok) > 0:
            row = {"benchmark": "LOSO species OOD", "aggregation": "mean±std"}
            for m in metrics:
                if m in ok.columns:
                    row[m] = f"{ok[m].mean():.4f} ± {ok[m].std(ddof=0):.4f}"
                    row[f"{m}_mean"] = float(ok[m].mean())
                    row[f"{m}_std"] = float(ok[m].std(ddof=0))
            row["n_runs"] = int(len(ok))
            rows.append(row)

    if loco_df is not None and len(loco_df) > 0:
        ok = loco_df[loco_df.get("error", pd.Series([np.nan] * len(loco_df))).isna()].copy()
        if len(ok) > 0:
            row = {"benchmark": "LOCO clade OOD", "aggregation": "mean±std"}
            for m in metrics:
                if m in ok.columns:
                    row[m] = f"{ok[m].mean():.4f} ± {ok[m].std(ddof=0):.4f}"
                    row[f"{m}_mean"] = float(ok[m].mean())
                    row[f"{m}_std"] = float(ok[m].std(ddof=0))
            row["n_runs"] = int(len(ok))
            rows.append(row)

    return pd.DataFrame(rows)


def plot_master_summary(master_df: pd.DataFrame, out_png: str):
    if len(master_df) == 0:
        return
    plot_rows = []
    for _, row in master_df.iterrows():
        for metric in ["accuracy_mean", "auprc_mean", "auroc_mean", "f1_mean", "mcc_mean"]:
            if metric in row and pd.notna(row[metric]):
                plot_rows.append({
                    "benchmark": row["benchmark"],
                    "metric": metric.replace("_mean", "").upper(),
                    "value": row[metric],
                })
    if len(plot_rows) == 0:
        return
    plot_df = pd.DataFrame(plot_rows)
    plt.figure(figsize=(8, 5))
    sns.barplot(data=plot_df, x="benchmark", y="value", hue="metric")
    plt.ylim(0, 1.0)
    plt.ylabel("Score")
    plt.xlabel("")
    plt.title("OOD benchmark summary")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


# ============================================================
# 12. main
# ============================================================
def main():
    try:
        script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0] or "interactive"
    except Exception:
        script_name = "interactive"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_dir = ensure_dir(f"results_{script_name}_{timestamp}")
    print(f"[Output] 所有结果将保存到: {root_dir}")

    config_dump = {
        "MODEL_DIR": MODEL_DIR,
        "POS_FILE": POS_FILE,
        "NEG_FILE": NEG_FILE,
        "TAXONOMY_MAP_FILE": TAXONOMY_MAP_FILE,
        "MAX_LEN_TRAIN": MAX_LEN_TRAIN,
        "MAX_LEN_EVAL": MAX_LEN_EVAL,
        "BATCH_SIZE": BATCH_SIZE,
        "EPOCHS_STANDARD": EPOCHS_STANDARD,
        "EPOCHS_OOD": EPOCHS_OOD,
        "RUN_STANDARD_SPECIES_DISJOINT": RUN_STANDARD_SPECIES_DISJOINT,
        "RUN_LOSO_SPECIES": RUN_LOSO_SPECIES,
        "RUN_LOCO_CLADE": RUN_LOCO_CLADE,
        "LIMIT_DATASET_TO_TOPK_SPECIES": LIMIT_DATASET_TO_TOPK_SPECIES,
        "DATASET_TOPK_SPECIES": DATASET_TOPK_SPECIES,
        "DATASET_TOPK_REQUIRE_BOTH_LABELS": DATASET_TOPK_REQUIRE_BOTH_LABELS,
        "OOD_SPECIES_TOPK": OOD_SPECIES_TOPK,
        "OOD_SPECIES_MIN_TOTAL": OOD_SPECIES_MIN_TOTAL,
        "OOD_SPECIES_MIN_POS": OOD_SPECIES_MIN_POS,
        "OOD_SPECIES_MIN_NEG": OOD_SPECIES_MIN_NEG,
        "OOD_CLADE_MIN_TOTAL": OOD_CLADE_MIN_TOTAL,
        "OOD_CLADE_MIN_POS": OOD_CLADE_MIN_POS,
        "OOD_CLADE_MIN_NEG": OOD_CLADE_MIN_NEG,
        "OOD_CLADE_ALLOW": OOD_CLADE_ALLOW,
        "STANDARD_TRAIN_FRAC": STANDARD_TRAIN_FRAC,
        "STANDARD_VAL_FRAC": STANDARD_VAL_FRAC,
        "STANDARD_TEST_FRAC": STANDARD_TEST_FRAC,
        "OOD_VAL_FRAC_FROM_REST": OOD_VAL_FRAC_FROM_REST,
        "SEED": SEED,
        "DEVICE": str(DEVICE),
    }
    with open(os.path.join(root_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config_dump, f, ensure_ascii=False, indent=2)

    df_all = load_combined_data(POS_FILE, NEG_FILE, taxonomy_map_file=TAXONOMY_MAP_FILE, random_state=SEED)
    species_manifest_all = build_unit_manifest(df_all, "species_key")
    save_df(species_manifest_all,
            os.path.join(root_dir, "species_manifest_before_filter.csv"),
            os.path.join(root_dir, "species_manifest_before_filter.md"))

    if LIMIT_DATASET_TO_TOPK_SPECIES:
        df_all, _, selected_species_manifest = restrict_dataset_to_top_species(
            df_all,
            top_k=DATASET_TOPK_SPECIES,
            require_both_labels=DATASET_TOPK_REQUIRE_BOTH_LABELS
        )
        save_df(selected_species_manifest,
                os.path.join(root_dir, "species_manifest_selected.csv"),
                os.path.join(root_dir, "species_manifest_selected.md"))

    save_df(df_all[["seq_len", "label", "tax_id", "species_name", "species_label", "clade_key", "group_id", "species_key"]],
            os.path.join(root_dir, "dataset_manifest_compact.csv"),
            os.path.join(root_dir, "dataset_manifest_compact.md"))

    standard_df = None
    loso_df = None
    loco_df = None

    if RUN_STANDARD_SPECIES_DISJOINT:
        print("\n" + "#" * 120)
        print("开始运行：标准 species-disjoint 基准")
        print("#" * 120)
        standard_df = run_standard_species_disjoint(df_all, root_dir)
        save_df(standard_df, os.path.join(root_dir, "standard_species_disjoint_summary.csv"),
                os.path.join(root_dir, "standard_species_disjoint_summary.md"))

    if RUN_LOSO_SPECIES:
        print("\n" + "#" * 120)
        print("开始运行：LOSO 跨物种 OOD")
        print("#" * 120)
        loso_df = run_loso_species(df_all, root_dir)

    if RUN_LOCO_CLADE:
        print("\n" + "#" * 120)
        print("开始运行：LOCO 跨类群 OOD")
        print("#" * 120)
        loco_df = run_loco_clade(df_all, root_dir)

    master_df = build_master_benchmark_table(standard_df=standard_df, loso_df=loso_df, loco_df=loco_df)
    save_df(master_df, os.path.join(root_dir, "master_benchmark_table.csv"),
            os.path.join(root_dir, "master_benchmark_table.md"))
    plot_master_summary(master_df, os.path.join(root_dir, "master_benchmark_summary.png"))

    print("\n训练与 OOD 评估结束。")
    print(f"结果目录: {root_dir}")
    print("建议首先查看：")
    print("1) master_benchmark_table.csv / .md")
    print("2) loso_species/loso_species_summary.csv")
    print("3) loco_clade/loco_clade_summary.csv")
    print("4) 各种 *heatmap.png / *bar.png")


if __name__ == "__main__":
    main()
