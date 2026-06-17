# -*- coding: utf-8 -*-

import os
import sys
from datetime import datetime
import random
import warnings
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_curve, auc, precision_recall_curve,
    average_precision_score, confusion_matrix
)
from sklearn.exceptions import UndefinedMetricWarning
from tqdm import tqdm
import matplotlib.pyplot as plt
from curve_export_helper import (
    apply_runtime_overrides,
    apply_seed_override,
    resolve_checkpoint_path,
    save_curve_points,
)
import seaborn as sns
import csv

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# ========= 可复现性 =========
SEED = apply_seed_override(42)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# 如更看重速度可改 False
DETERMINISTIC = True
if DETERMINISTIC:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
else:
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# ========== 全局超参数 ==========
MAX_SEQ_LEN = 128          # 每个窗口的核苷酸长度（固定）
SLIDE_STEP = 500           # sliding-window 步长（核苷酸）
batch_size = 16
epochs = 50
epochs, EVAL_ONLY = apply_runtime_overrides(epochs)
learning_rate = 1e-3
weight_decay = 1e-4

# ========== 可选：长度匹配 / GC 匹配 ==========
ENABLE_LENGTH_MATCH = False
LENGTH_MATCH_BINS = 10

ENABLE_GC_MATCH = False
GC_MATCH_BINS = 10

# ========== 可选：偏差诊断 ==========
ENABLE_BIAS_DIAGNOSTIC = True
DIAG_TOPK = 15

# ========== 可选：训练集困难负样本采样 ==========
ENABLE_HARD_NEG_TRAIN = True
HARD_NEG_MIN_POS = 200
HARD_NEG_MIN_POS_RATIO = 0.05
HARD_NEG_RANDOM_STATE = SEED

# ========== DNA 工具函数 ==========
VALID_DNA_CHARS = set("ACGTN")
_RC_TABLE = str.maketrans("ACGTN", "TGCAN")


def clean_sequence(seq: str) -> str:
    """将序列转为大写，并把非 A/C/G/T/N 字符全部置为 N。"""
    s = str(seq).strip().upper()
    if not s:
        return ""
    return "".join(ch if ch in VALID_DNA_CHARS else "N" for ch in s)


def rev_comp(seq: str) -> str:
    """反向互补序列（仅用于 canonical RC key）。"""
    s = str(seq)
    return s.translate(_RC_TABLE)[::-1]


def canon_rc(seq: str) -> str:
    """canonical RC：seq 与其反向互补中，取字典序较小者。"""
    s = clean_sequence(seq)
    if not s:
        return ""
    rc = rev_comp(s)
    return s if s <= rc else rc


def gc_content(seq: str) -> float:
    s = clean_sequence(seq)
    L = len(s)
    if L == 0:
        return 0.0
    g = s.count("G")
    c = s.count("C")
    return (g + c) / L


# ========== 结果保存根目录 ==========
try:
    script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0] or "cnn_multi"
except Exception:
    script_name = "cnn_multi"

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
ROOT_RESULTS_DIR = f"results_{script_name}_{timestamp}"
os.makedirs(ROOT_RESULTS_DIR, exist_ok=True)


# ========== 数据加载 + canonical RC 去重 + GroupSplit ==========
def _find_first_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def length_match_after_dedup(df_all: pd.DataFrame,
                             n_bins: int = 10,
                             random_state: int = SEED) -> pd.DataFrame:
    """
    长度匹配（canonical 去重后）：
    - 用 qcut 做分位数分箱
    - 每个 bin 内对 pos/neg 采样 min(n_pos, n_neg)
    """
    df = df_all.copy().reset_index(drop=True)
    df["seq_len"] = df["seq"].str.len()
    df["len_bin"] = pd.qcut(df["seq_len"], q=n_bins, duplicates="drop")

    parts = []
    kept_bins = 0
    dropped_bins = 0

    for b, g in df.groupby("len_bin", observed=True):
        pos_g = g[g["label"] == 1]
        neg_g = g[g["label"] == 0]
        n = min(len(pos_g), len(neg_g))
        if n <= 0:
            dropped_bins += 1
            continue
        kept_bins += 1
        parts.append(pos_g.sample(n=n, random_state=random_state))
        parts.append(neg_g.sample(n=n, random_state=random_state))

    if len(parts) == 0:
        print("[LenMatch] 警告：长度匹配后没有可用 bin，返回原始去重数据。")
        return df_all

    df_matched = pd.concat(parts, ignore_index=True)
    df_matched = df_matched.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    n_pos = int((df_matched["label"] == 1).sum())
    n_neg = int((df_matched["label"] == 0).sum())
    print(f"[LenMatch] 目标 {n_bins} bins（duplicates='drop' 可能减少实际 bin）。")
    print(f"[LenMatch] 有效 bin={kept_bins}, 空/单边 bin={dropped_bins}。")
    print(f"[LenMatch] 匹配后样本数: pos={n_pos}, neg={n_neg}, total={len(df_matched)}")

    return df_matched


def gc_match_after_dedup(df_all: pd.DataFrame,
                         n_bins: int = 10,
                         random_state: int = SEED) -> pd.DataFrame:
    """
    GC 匹配（canonical 去重后）：
    - 用 qcut 做 GC 分位数分箱
    - 每个 bin 内对 pos/neg 采样 min(n_pos, n_neg)
    """
    df = df_all.copy().reset_index(drop=True)
    df["gc"] = df["seq"].map(gc_content)
    df["gc_bin"] = pd.qcut(df["gc"], q=n_bins, duplicates="drop")

    parts = []
    kept_bins = 0
    dropped_bins = 0

    for b, g in df.groupby("gc_bin", observed=True):
        pos_g = g[g["label"] == 1]
        neg_g = g[g["label"] == 0]
        n = min(len(pos_g), len(neg_g))
        if n <= 0:
            dropped_bins += 1
            continue
        kept_bins += 1
        parts.append(pos_g.sample(n=n, random_state=random_state))
        parts.append(neg_g.sample(n=n, random_state=random_state))

    if len(parts) == 0:
        print("[GCMatch] 警告：GC 匹配后没有可用 bin，返回原始去重数据。")
        return df_all

    df_matched = pd.concat(parts, ignore_index=True)
    df_matched = df_matched.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    n_pos = int((df_matched["label"] == 1).sum())
    n_neg = int((df_matched["label"] == 0).sum())
    print(f"[GCMatch] 目标 {n_bins} bins（duplicates='drop' 可能减少实际 bin）。")
    print(f"[GCMatch] 有效 bin={kept_bins}, 空/单边 bin={dropped_bins}。")
    print(f"[GCMatch] 匹配后样本数: pos={n_pos}, neg={n_neg}, total={len(df_matched)}")

    return df_matched


def _print_bias_diagnostic(df_all: pd.DataFrame, topk=15):
    """
    偏差诊断（不影响训练）：
    - GC 统计（按 label）
    - 若存在 tax_id，输出 label purity 极端 tax_id
    """
    try:
        df = df_all.copy()
        if "gc" not in df.columns:
            df["gc"] = df["seq"].map(gc_content)

        print("[Diag] GC content by label:")
        print(df.groupby("label")["gc"].describe())

        if "tax_id" in df.columns:
            purity = df.groupby("tax_id")["label"].mean().sort_values()
            print(f"[Diag] tax_id label purity (lowest {topk}):")
            print(purity.head(topk))
            print(f"[Diag] tax_id label purity (highest {topk}):")
            print(purity.tail(topk))
    except Exception as e:
        print(f"[Diag] 诊断输出失败（不影响训练）：{e}")


def load_combined_data(file_a, file_b, random_state=SEED) -> pd.DataFrame:
    """
    加载 patho / nonpatho，采样为类平衡，做 canonical RC 去重，并构建 group_id：
      - 优先使用 tax_id + comp_id
      - 否则 tax_id
      - 再否则 Species_Name
      - 再否则 fallback 到 canonical 序列

    关键改动：
      - canonical 冲突序列（同一 canon 同时含正负标签）：
        正负双方全部删除（更保守）
    """
    df_a = pd.read_csv(file_a, low_memory=False)
    df_b = pd.read_csv(file_b, low_memory=False)

    # 类平衡采样（保留你原逻辑）
    min_len = min(len(df_a), len(df_b))
    df_a = df_a.sample(n=min_len, random_state=random_state).reset_index(drop=True)
    df_b = df_b.sample(n=min_len, random_state=random_state).reset_index(drop=True)

    # 序列列：正类第 15 列（index 14），负类第 14 列（index 13）
    seq_col_a = df_a.columns[14]
    seq_col_b = df_b.columns[13]

    tax_col_a = _find_first_col(df_a, ["tax_id", "#tax_id", "Tax_ID", "taxid"])
    tax_col_b = _find_first_col(df_b, ["tax_id", "#tax_id", "Tax_ID", "taxid"])
    species_col_a = _find_first_col(df_a, ["Species_Name", "species_name", "Species", "species"])
    species_col_b = _find_first_col(df_b, ["Species_Name", "species_name", "Species", "species"])
    comp_col_a = "comp_id" if "comp_id" in df_a.columns else None
    comp_col_b = "comp_id" if "comp_id" in df_b.columns else None

    pos = pd.DataFrame({
        "seq_raw": df_a[seq_col_a].astype(str),
        "label": 1,
    })
    neg = pd.DataFrame({
        "seq_raw": df_b[seq_col_b].astype(str),
        "label": 0,
    })

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
    print(
        f"载入原始序列(采样平衡后): 正类{len(pos)} / 负类{len(neg)}，总计{len(df_all)}；"
        "将进行清洗和 canonical RC 去重"
    )

    # 清洗 + canonical RC
    df_all["seq"] = df_all["seq_raw"].map(clean_sequence)
    df_all["seq_len"] = df_all["seq"].str.len()
    df_all = df_all[df_all["seq_len"] > 0].copy()
    df_all["canon"] = df_all["seq"].map(canon_rc)

    n_before = len(df_all)

    # canonical 标签冲突：正负双方全部删除
    conflict_groups = df_all.groupby("canon")["label"].nunique()
    conflict_canons = conflict_groups[conflict_groups > 1].index
    n_conflict = int(len(conflict_canons))
    if n_conflict > 0:
        print(f"[Dedup] 发现 {n_conflict} 个 canonical 组同时包含正负标签：将正负双方全部删除。")
        df_all = df_all[~df_all["canon"].isin(conflict_canons)].copy()

    # canonical 去重
    df_all = df_all.drop_duplicates(subset="canon", keep="first").reset_index(drop=True)
    n_after = len(df_all)
    print(f"[Dedup] canonical RC 去重: {n_before} -> {n_after} 条序列")

    # 构建 group_id
    def _build_group_id(row):
        tax = None
        if "tax_id" in row and pd.notna(row.get("tax_id", None)):
            v = row["tax_id"]
            if isinstance(v, (int, np.integer)):
                tax = str(int(v))
            elif isinstance(v, float) and not np.isnan(v):
                tax = str(int(v))
            else:
                tax = str(v)

        species = str(row["species_name"]) if "species_name" in row and pd.notna(row.get("species_name", None)) else None
        comp = str(row["comp_id"]) if "comp_id" in row and pd.notna(row.get("comp_id", None)) else None

        if comp is not None and tax is not None:
            return f"{tax}|c{comp}"
        if comp is not None:
            return f"NA|c{comp}"
        if tax is not None:
            return tax
        if species is not None:
            return species
        return row["canon"]

    df_all["group_id"] = df_all.apply(_build_group_id, axis=1)
    print(f"[GroupKey] 唯一 group_id 数量: {df_all['group_id'].nunique()}")

    df_all["label"] = df_all["label"].astype(int)
    df_all["gc"] = df_all["seq"].map(gc_content)

    return df_all


def group_split_70_15_15_frames(df_all: pd.DataFrame, random_state=SEED):
    """
    使用 GroupShuffleSplit 按 group_id 分为 train / val / test:
      - 先 70% train / 30% temp
      - 再对 temp 50/50 划分 val / test
    严格避免 group_id 和 canonical 的跨集泄漏。

    返回 DataFrame，便于训练集做困难负样本采样。
    """
    groups = df_all["group_id"].values
    labels = df_all["label"].values

    gss1 = GroupShuffleSplit(
        n_splits=1,
        test_size=0.30,
        random_state=random_state
    )
    train_idx, temp_idx = next(gss1.split(df_all, labels, groups))

    df_train = df_all.iloc[train_idx].reset_index(drop=True)
    df_temp = df_all.iloc[temp_idx].reset_index(drop=True)

    groups_temp = df_temp["group_id"].values
    labels_temp = df_temp["label"].values
    gss2 = GroupShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=random_state + 1
    )
    val_idx, test_idx = next(gss2.split(df_temp, labels_temp, groups_temp))
    df_val = df_temp.iloc[val_idx].reset_index(drop=True)
    df_test = df_temp.iloc[test_idx].reset_index(drop=True)

    def _log_split(name, frame):
        pos_rate = frame["label"].mean()
        n_samples = len(frame)
        n_groups = frame["group_id"].nunique()
        print(
            f"[GroupSplit] {name}: 样本={n_samples}, 组数={n_groups}, "
            f"pos_rate={pos_rate:.3f}"
        )

    _log_split("train", df_train)
    _log_split("val", df_val)
    _log_split("test", df_test)

    # 检查 group_id 是否跨集重复
    g_tr = set(df_train["group_id"])
    g_va = set(df_val["group_id"])
    g_te = set(df_test["group_id"])
    inter_tr_va = g_tr & g_va
    inter_tr_te = g_tr & g_te
    inter_va_te = g_va & g_te
    print(
        f"[GroupSplit] 交叉 group_id：train∩val={len(inter_tr_va)}, "
        f"train∩test={len(inter_tr_te)}, val∩test={len(inter_va_te)}"
    )
    assert len(inter_tr_va) == 0 and len(inter_tr_te) == 0 and len(inter_va_te) == 0, \
        "GroupSplit 出现跨集共享 group_id，存在潜在泄漏！"

    # 再检查 canonical 是否跨集重复
    c_tr = set(df_train["canon"])
    c_va = set(df_val["canon"])
    c_te = set(df_test["canon"])
    inter_tr_va_c = c_tr & c_va
    inter_tr_te_c = c_tr & c_te
    inter_va_te_c = c_va & c_te
    print(
        f"[LeakCheck] canonical RC 交叉：train∩val={len(inter_tr_va_c)}, "
        f"train∩test={len(inter_tr_te_c)}, val∩test={len(inter_va_te_c)}"
    )
    assert len(inter_tr_va_c) == 0 and len(inter_tr_te_c) == 0 and len(inter_va_te_c) == 0, \
        "存在跨集完全相同的 canonical 序列！"

    print(
        f"[Split] train={len(df_train)} ({df_train['label'].mean():.3f} pos rate), "
        f"val={len(df_val)} ({df_val['label'].mean():.3f}), "
        f"test={len(df_test)} ({df_test['label'].mean():.3f})"
    )

    return df_train, df_val, df_test


def build_hard_negative_trainset(df_train: pd.DataFrame,
                                 random_state: int = SEED,
                                 min_pos: int = 200,
                                 min_pos_ratio: float = 0.05) -> pd.DataFrame:
    """
    训练集困难负样本采样：
    目标：尽量在同背景 key 内做到 pos/neg 平衡抽样。
    优先级：
      1) group_id
      2) tax_id
      3) species_name
      4) canon
    若有效样本太少则回退原训练集。
    """
    df = df_train.copy().reset_index(drop=True)
    total_pos = int((df["label"] == 1).sum())
    total_neg = int((df["label"] == 0).sum())
    print(f"[HardNeg] 原训练集: pos={total_pos}, neg={total_neg}, total={len(df)}")

    def _match_by_key(key_col: str):
        if key_col not in df.columns:
            return None

        counts = df.groupby([key_col, "label"]).size().unstack(fill_value=0)
        if 0 not in counts.columns:
            counts[0] = 0
        if 1 not in counts.columns:
            counts[1] = 0

        valid_keys = counts[(counts[0] > 0) & (counts[1] > 0)].index.tolist()
        if len(valid_keys) == 0:
            return None

        parts = []
        kept = 0
        dropped = 0

        for k in valid_keys:
            g = df[df[key_col] == k]
            pos_g = g[g["label"] == 1]
            neg_g = g[g["label"] == 0]
            n = min(len(pos_g), len(neg_g))
            if n <= 0:
                dropped += 1
                continue
            kept += 1
            parts.append(pos_g.sample(n=n, random_state=random_state))
            parts.append(neg_g.sample(n=n, random_state=random_state))

        if len(parts) == 0:
            return None

        out = pd.concat(parts, ignore_index=True)
        out = out.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

        n_pos = int((out["label"] == 1).sum())
        n_neg = int((out["label"] == 0).sum())
        print(f"[HardNeg] 按 {key_col} 匹配: 有效key={kept}, 空/单边key={dropped}")
        print(f"[HardNeg] 匹配后: pos={n_pos}, neg={n_neg}, total={len(out)}")
        return out

    def _enough(df_candidate):
        if df_candidate is None:
            return False
        n_pos = int((df_candidate["label"] == 1).sum())
        threshold = max(min_pos, int(min_pos_ratio * max(1, total_pos)))
        return n_pos >= threshold

    # 1) group_id
    df_hard = _match_by_key("group_id")

    # 2) tax_id
    if not _enough(df_hard) and "tax_id" in df.columns:
        df["hn_tax"] = df["tax_id"].astype(str)
        df_hard = _match_by_key("hn_tax")

    # 3) species
    if not _enough(df_hard) and "species_name" in df.columns:
        df["hn_sp"] = df["species_name"].astype(str)
        df_hard = _match_by_key("hn_sp")

    # 4) canon
    if not _enough(df_hard):
        df["hn_canon"] = df["canon"].astype(str)
        df_hard = _match_by_key("hn_canon")

    # 回退
    if not _enough(df_hard):
        print("[HardNeg] 警告：困难负样本匹配不足，回退使用原训练集。")
        return df_train

    # 全局再平衡（保险）
    n_pos = int((df_hard["label"] == 1).sum())
    n_neg = int((df_hard["label"] == 0).sum())
    if n_pos != n_neg:
        n = min(n_pos, n_neg)
        df_pos = df_hard[df_hard["label"] == 1].sample(n=n, random_state=random_state)
        df_neg = df_hard[df_hard["label"] == 0].sample(n=n, random_state=random_state)
        df_hard = pd.concat([df_pos, df_neg], ignore_index=True)\
                    .sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        print(f"[HardNeg] 全局再平衡: pos={n}, neg={n}, total={len(df_hard)}")

    return df_hard


# ========== 序列裁剪、滑窗与编码 ==========
BASE2ID = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}


def center_crop_or_pad(seq, max_len=MAX_SEQ_LEN):
    """中心截取或右侧补 N 到 max_len，保证可重复性。"""
    s = clean_sequence(seq)
    if len(s) == 0:
        return "N" * max_len
    if len(s) == max_len:
        return s
    if len(s) < max_len:
        return s + "N" * (max_len - len(s))
    start = (len(s) - max_len) // 2
    return s[start:start + max_len]


def sliding_windows(seq, window_len=MAX_SEQ_LEN, step=SLIDE_STEP):
    """
    sliding-window 生成窗口：
      - 序列长度 <= window_len 时：右侧补 N，单窗口
      - 否则：从头以 step 滑动，确保最后一段覆盖末端
    返回窗口字符串列表，每个长度均为 window_len。
    """
    s = clean_sequence(seq)
    if len(s) == 0:
        return ["N" * window_len]
    if len(s) <= window_len:
        if len(s) < window_len:
            s = s + "N" * (window_len - len(s))
        return [s]

    windows = []
    pos = 0
    while pos + window_len <= len(s):
        windows.append(s[pos:pos + window_len])
        pos += step
    if pos < len(s):
        windows.append(s[-window_len:])
    return windows


def encode_sequence(seq):
    s = str(seq).strip().upper()
    if len(s) == 0:
        s = "N" * MAX_SEQ_LEN
    return torch.tensor([BASE2ID.get(ch, 4) for ch in s], dtype=torch.long)


# ========== Dataset（固定窗口 / 滑窗 + pooling）==========
class SeqDatasetFixed(Dataset):
    """固定单窗口：每条序列中心截取/补齐到 MAX_SEQ_LEN。"""
    def __init__(self, sequences, labels):
        self.seqs = list(sequences)
        self.labels = np.array(labels).astype(int)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        s = self.seqs[idx]
        s_proc = center_crop_or_pad(s, MAX_SEQ_LEN)
        x = encode_sequence(s_proc)
        y = int(self.labels[idx])
        return x, y


class SeqDatasetSliding(Dataset):
    """sliding-window：一个样本对应多个窗口 + 一个序列标签。"""
    def __init__(self, sequences, labels):
        self.seqs = list(sequences)
        self.labels = np.array(labels).astype(int)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        s = self.seqs[idx]
        win_strs = sliding_windows(s, MAX_SEQ_LEN, SLIDE_STEP)
        win_tensors = [encode_sequence(ws) for ws in win_strs]
        x = torch.stack(win_tensors, dim=0)   # [W, L]
        y = int(self.labels[idx])
        return x, y


def collate_fixed(batch):
    xs, ys = zip(*batch)
    xs = torch.stack(xs, dim=0)
    ys = torch.tensor(ys, dtype=torch.long)
    window_counts = torch.ones(len(ys), dtype=torch.long)
    return xs, ys, window_counts


def collate_sliding(batch):
    xs, ys = zip(*batch)
    window_counts = [x.shape[0] for x in xs]
    all_windows = torch.cat(xs, dim=0)
    labels_seq = torch.tensor(ys, dtype=torch.long)
    window_counts = torch.tensor(window_counts, dtype=torch.long)
    return all_windows, labels_seq, window_counts


# ========== 小窗口池化修正工具 ==========
def _safe_maxpool1d(x, kernel=4, stride=4):
    """
    避免小窗口第二次池化把长度压到 1。
    规则：
      - 若当前长度 < kernel：不池化
      - 计算标准 out_len，若 out_len <= 1 且长度 >= 2：
          尝试用 kernel=2,stride=2
      - 否则按默认 kernel/stride
    """
    L = x.size(-1)
    if L < kernel:
        return x

    # 标准 out_len（无 padding）
    out_len = (L - kernel) // stride + 1
    if out_len <= 1 and L >= 2:
        k2, s2 = 2, 2
        if L >= k2:
            return F.max_pool1d(x, kernel_size=k2, stride=s2)
        return x

    return F.max_pool1d(x, kernel_size=kernel, stride=stride)


# ========== CNN 模型（Embedding / one-hot）==========
class CNNEmbedding(nn.Module):
    """
    字符级 Embedding + 1D-CNN
    输入： [N_windows, L] 的 token id
    输出： [N_windows, num_classes] 的窗口级 logits
    """
    def __init__(self, vocab_size=5, embed_dim=4, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv1 = nn.Conv1d(embed_dim, 16, kernel_size=8, padding=3)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 8, kernel_size=8, padding=3)
        self.bn2 = nn.BatchNorm1d(8)
        self.fc1 = nn.Linear(8, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x_ids):
        # x_ids: [N, L]
        x = self.embedding(x_ids)      # [N, L, E]
        x = x.permute(0, 2, 1)         # [N, E, L]

        x = F.relu(self.bn1(self.conv1(x)))
        x = _safe_maxpool1d(x, kernel=4, stride=4)

        x = F.relu(self.bn2(self.conv2(x)))
        x = _safe_maxpool1d(x, kernel=4, stride=4)

        x = x.mean(dim=2)              # [N, 8]
        x = F.relu(self.fc1(x))        # [N, 64]
        x = self.fc2(x)                # [N, num_classes]
        return x


class CNNOneHot(nn.Module):
    """
    one-hot + 1D-CNN
    输入： [N_windows, L] 的 token id
    输出： [N_windows, num_classes] 的窗口级 logits
    """
    def __init__(self, num_classes=2, num_channels=5):
        super().__init__()
        self.num_channels = num_channels
        self.conv1 = nn.Conv1d(num_channels, 16, kernel_size=8, padding=3)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 8, kernel_size=8, padding=3)
        self.bn2 = nn.BatchNorm1d(8)
        self.fc1 = nn.Linear(8, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x_ids):
        # x_ids: [N, L]
        x = F.one_hot(x_ids, num_classes=self.num_channels).float()  # [N, L, C]
        x = x.permute(0, 2, 1)                                       # [N, C, L]

        x = F.relu(self.bn1(self.conv1(x)))
        x = _safe_maxpool1d(x, kernel=4, stride=4)

        x = F.relu(self.bn2(self.conv2(x)))
        x = _safe_maxpool1d(x, kernel=4, stride=4)

        x = x.mean(dim=2)                                            # [N, 8]
        x = F.relu(self.fc1(x))                                      # [N, 64]
        x = self.fc2(x)                                              # [N, num_classes]
        return x


# ========== 序列级 pooling：窗口 logits -> 序列 logits ==========
def aggregate_logits_by_sequence(logits_windows, window_counts, mode="mean"):
    """
    logits_windows: [N_windows, num_classes]
    window_counts:  [B_seq]，每条序列的窗口数
    返回：
      seq_logits: [B_seq, num_classes]
    """
    assert logits_windows.dim() == 2
    assert window_counts.dim() == 1
    total_windows = logits_windows.size(0)
    assert window_counts.sum().item() == total_windows, \
        f"窗口数量不匹配: sum(window_counts)={window_counts.sum().item()}, N_windows={total_windows}"

    seq_logits = []
    start = 0
    for count in window_counts.tolist():
        end = start + count
        lw = logits_windows[start:end]
        if mode == "max":
            agg = lw.max(dim=0, keepdim=True)[0]
        else:
            agg = lw.mean(dim=0, keepdim=True)
        seq_logits.append(agg)
        start = end
    return torch.cat(seq_logits, dim=0)


# ========== 指标计算 ==========
def compute_classification_metrics(labels, preds, probs):
    if len(labels) == 0:
        return dict(
            accuracy=np.nan, precision=np.nan, recall=np.nan, f1=np.nan,
            mcc=np.nan, auroc=np.nan, auprc=np.nan, ap=np.nan
        )
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    mcc = matthews_corrcoef(labels, preds)
    try:
        fpr, tpr, _ = roc_curve(labels, probs)
        auroc = auc(fpr, tpr)
    except Exception:
        auroc = float("nan")
    try:
        prec_vals, rec_vals, _ = precision_recall_curve(labels, probs)
        auprc = auc(rec_vals, prec_vals)
        ap = average_precision_score(labels, probs)
    except Exception:
        auprc, ap = float("nan"), float("nan")
    return dict(
        accuracy=accuracy, precision=precision, recall=recall, f1=f1,
        mcc=mcc, auroc=auroc, auprc=auprc, ap=ap
    )


# ========== 早停 ==========
class EarlyStopping:
    def __init__(self, save_path, patience=8, verbose=True, delta=0.0):
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_loss = np.Inf
        self.early_stop = False
        self.best_epoch = 0
        os.makedirs(save_path, exist_ok=True)

    def __call__(self, val_loss, model, epoch):
        if val_loss < self.best_loss - self.delta:
            if self.verbose:
                print(f"验证损失改善: {self.best_loss:.6f} -> {val_loss:.6f}. 保存模型...")
            self.best_loss = val_loss
            torch.save(
                model.state_dict(),
                os.path.join(self.save_path, f'best_model_epoch{epoch + 1}.pth')
            )
            self.counter = 0
            self.best_epoch = epoch
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


criterion = nn.CrossEntropyLoss()


@torch.inference_mode()
def evaluate(model, data_loader, device, pool_mode="mean"):
    model.eval()
    loss_sum = 0.0
    n_batches = 0
    all_labels, all_preds, all_probs = [], [], []

    for windows, labels_seq, window_counts in data_loader:
        windows = windows.to(device)
        labels_seq = labels_seq.to(device)
        window_counts = window_counts.to(device)

        logits_windows = model(windows)
        seq_logits = aggregate_logits_by_sequence(logits_windows, window_counts, mode=pool_mode)
        loss = criterion(seq_logits, labels_seq)
        loss_sum += loss.item()
        n_batches += 1

        probs = F.softmax(seq_logits, dim=1)[:, 1].detach().cpu().numpy()
        preds = torch.argmax(seq_logits, dim=1).detach().cpu().numpy()
        all_labels.extend(labels_seq.cpu().numpy().tolist())
        all_preds.extend(preds.tolist())
        all_probs.extend(probs.tolist())

    avg_loss = loss_sum / max(1, n_batches)
    metrics = compute_classification_metrics(
        np.array(all_labels), np.array(all_preds), np.array(all_probs)
    )
    return avg_loss, metrics, np.array(all_labels), np.array(all_preds), np.array(all_probs)


def run_experiment(exp_name,
                   model_type,
                   sliding,
                   train_sequences, y_train,
                   val_sequences, y_val,
                   test_sequences, y_test):
    """
    运行单个实验：
      model_type: "emb" 或 "onehot"
      sliding   : True 使用 sliding-window + pooling，False 使用单窗口
    """
    print("=" * 80)
    print(f"[Experiment] {exp_name} | model_type={model_type}, sliding={sliding}")
    print("=" * 80)

    output_dir = os.path.join(ROOT_RESULTS_DIR, f"exp_{exp_name}")
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    if sliding:
        TrainDataset = SeqDatasetSliding
        collate_fn = collate_sliding
    else:
        TrainDataset = SeqDatasetFixed
        collate_fn = collate_fixed

    train_loader = DataLoader(
        TrainDataset(train_sequences, y_train),
        batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(), num_workers=0
    )
    val_loader = DataLoader(
        TrainDataset(val_sequences, y_val),
        batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(), num_workers=0
    )
    test_loader = DataLoader(
        TrainDataset(test_sequences, y_test),
        batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(), num_workers=0
    )

    if model_type == "emb":
        model = CNNEmbedding(vocab_size=5, embed_dim=4, num_classes=2).to(device)
    elif model_type == "onehot":
        model = CNNOneHot(num_classes=2, num_channels=5).to(device)
    else:
        raise ValueError(f"未知 model_type: {model_type}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    early_stopping = EarlyStopping(save_path=checkpoint_dir, patience=8, verbose=True, delta=0.0)

    train_losses, val_losses = [], []
    accuracy_list, precision_list, recall_list, f1_list, mcc_list = [], [], [], [], []
    auroc_list, auprc_list, ap_list = [], [], []

    metrics_file_path = os.path.join(output_dir, "metrics_per_epoch.txt")
    with open(metrics_file_path, "w", encoding="utf-8") as f:
        f.write(
            "Epoch\tTrain_Loss\tVal_Loss\tAccuracy\tPrecision\tRecall\tF1\tMCC\t"
            "AUROC\tAUPRC\tAP\n"
        )

    pool_mode = "mean"

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        progress_bar = tqdm(train_loader, desc=f"{exp_name} Epoch [{epoch + 1}/{epochs}]", leave=False)

        for windows, labels_seq, window_counts in progress_bar:
            windows = windows.to(device)
            labels_seq = labels_seq.to(device)
            window_counts = window_counts.to(device)

            optimizer.zero_grad()
            logits_windows = model(windows)
            seq_logits = aggregate_logits_by_sequence(logits_windows, window_counts, mode=pool_mode)
            loss = criterion(seq_logits, labels_seq)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            progress_bar.set_postfix(loss=(total_loss / max(1, n_batches)))

        avg_train_loss = total_loss / max(1, n_batches)
        train_losses.append(avg_train_loss)

        val_loss, val_metrics, val_labels, val_preds, val_probs = evaluate(
            model, val_loader, device, pool_mode=pool_mode
        )
        val_losses.append(val_loss)

        accuracy_list.append(val_metrics['accuracy'])
        precision_list.append(val_metrics['precision'])
        recall_list.append(val_metrics['recall'])
        f1_list.append(val_metrics['f1'])
        mcc_list.append(val_metrics['mcc'])
        auroc_list.append(val_metrics['auroc'])
        auprc_list.append(val_metrics['auprc'])
        ap_list.append(val_metrics['ap'])

        print(
            f"[{exp_name}] Epoch [{epoch + 1}/{epochs}] - "
            f"Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, "
            f"Acc: {val_metrics['accuracy']:.4f}, P: {val_metrics['precision']:.4f}, "
            f"R: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}, "
            f"MCC: {val_metrics['mcc']:.4f}, AUROC: {val_metrics['auroc']:.4f}, "
            f"AUPRC: {val_metrics['auprc']:.4f}, AP: {val_metrics['ap']:.4f}"
        )
        with open(metrics_file_path, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch + 1}\t{avg_train_loss:.4f}\t{val_loss:.4f}\t"
                f"{val_metrics['accuracy']:.4f}\t{val_metrics['precision']:.4f}\t"
                f"{val_metrics['recall']:.4f}\t{val_metrics['f1']:.4f}\t{val_metrics['mcc']:.4f}\t"
                f"{val_metrics['auroc']:.4f}\t{val_metrics['auprc']:.4f}\t{val_metrics['ap']:.4f}\n"
            )

        early_stopping(val_loss, model, epoch)
        if early_stopping.early_stop:
            print(f"[{exp_name}] 早停触发于第 {epoch + 1} 轮，最佳模型来自第 {early_stopping.best_epoch + 1} 轮")
            break

    best_epoch_idx = early_stopping.best_epoch
    best_ckpt_path = resolve_checkpoint_path(os.path.join(checkpoint_dir, f'best_model_epoch{best_epoch_idx + 1}.pth'))
    if os.path.exists(best_ckpt_path):
        model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
        print(f"[{exp_name}] [Load Best] 加载最佳验证轮权重：{best_ckpt_path}")
    else:
        print(f"[{exp_name}] [Load Best] 未找到最佳权重文件，使用当前模型参数进行测试。")

    test_loss, test_metrics, test_labels, test_preds, test_probs = evaluate(
        model, test_loader, device, pool_mode=pool_mode
    )

    fpr, tpr, _ = roc_curve(test_labels, test_probs)
    auroc_final = auc(fpr, tpr)
    precision_vals, recall_vals, _ = precision_recall_curve(test_labels, test_probs)
    ap_final = average_precision_score(test_labels, test_probs)
    auprc_final = auc(recall_vals, precision_vals)

    
    save_curve_points(output_dir, fpr=fpr, tpr=tpr, recall=recall_vals, precision=precision_vals)
    cm = confusion_matrix(test_labels, test_preds)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix (Test) - {exp_name}')
    plt.savefig(os.path.join(output_dir, f"confusion_matrix_test_{exp_name}.png"))
    plt.close()

    plt.figure()
    plt.plot(fpr, tpr, lw=2, label=f'ROC (AUROC = {auroc_final:.2f})')
    plt.plot([0, 1], [0, 1], lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve (Test) - {exp_name}')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"roc_curve_test_{exp_name}.png"))
    plt.close()

    plt.figure()
    plt.step(
        recall_vals, precision_vals, where='post', lw=2,
        label=f'PR (AP = {ap_final:.2f}, AUPRC = {auprc_final:.2f})'
    )
    plt.fill_between(recall_vals, precision_vals, step='post', alpha=0.2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'PR Curve (Test) - {exp_name}')
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"pr_curve_test_{exp_name}.png"))
    plt.close()

    final_accuracy = test_metrics["accuracy"]
    final_precision_metric = test_metrics["precision"]
    final_recall_metric = test_metrics["recall"]
    final_f1 = test_metrics["f1"]
    final_mcc = test_metrics["mcc"]
    avg_train_loss = train_losses[-1] if len(train_losses) > 0 else float('nan')
    avg_val_loss = val_losses[-1] if len(val_losses) > 0 else float('inf')

    metrics_text = f"""Final Test Metrics ({exp_name}, evaluated on TEST set only):
Train Loss (last epoch): {avg_train_loss:.4f}
Validation Loss (last epoch): {avg_val_loss:.4f}
Test Loss: {test_loss:.4f}
Accuracy: {final_accuracy:.4f}
Precision: {final_precision_metric:.4f}
Recall: {final_recall_metric:.4f}
F1-score: {final_f1:.4f}
MCC: {final_mcc:.4f}
AUROC: {auroc_final:.4f}
AUPRC: {auprc_final:.4f}
Average Precision (AP): {ap_final:.4f}
Early Stopping Triggered: {'Yes' if early_stopping.early_stop else 'No'}
Best Epoch (by val loss): {early_stopping.best_epoch + 1}

MAX_SEQ_LEN: {MAX_SEQ_LEN}
Length Match Enabled: {ENABLE_LENGTH_MATCH}
Length Match Bins: {LENGTH_MATCH_BINS}
GC Match Enabled: {ENABLE_GC_MATCH}
GC Match Bins: {GC_MATCH_BINS}
HardNeg Train Enabled: {ENABLE_HARD_NEG_TRAIN}
"""
    with open(os.path.join(output_dir, f"final_metrics_test_{exp_name}.txt"), "w", encoding="utf-8") as f:
        f.write(metrics_text)

    with open(os.path.join(output_dir, f"final_metrics_test_{exp_name}.csv"), mode='w', newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Train Loss (last)", avg_train_loss])
        writer.writerow(["Validation Loss (last)", avg_val_loss])
        writer.writerow(["Test Loss", test_loss])
        writer.writerow(["Accuracy", final_accuracy])
        writer.writerow(["Precision", final_precision_metric])
        writer.writerow(["Recall", final_recall_metric])
        writer.writerow(["F1-score", final_f1])
        writer.writerow(["MCC", final_mcc])
        writer.writerow(["AUROC", auroc_final])
        writer.writerow(["AUPRC", auprc_final])
        writer.writerow(["Average Precision (AP)", ap_final])
        writer.writerow(["Early Stopping Triggered", 'Yes' if early_stopping.early_stop else 'No'])
        writer.writerow(["Best Epoch (val)", early_stopping.best_epoch + 1])
        writer.writerow(["MAX_SEQ_LEN", MAX_SEQ_LEN])
        writer.writerow(["Length Match Enabled", ENABLE_LENGTH_MATCH])
        writer.writerow(["Length Match Bins", LENGTH_MATCH_BINS])
        writer.writerow(["GC Match Enabled", ENABLE_GC_MATCH])
        writer.writerow(["GC Match Bins", GC_MATCH_BINS])
        writer.writerow(["HardNeg Train Enabled", ENABLE_HARD_NEG_TRAIN])

    loss_data = pd.DataFrame({
        'Epoch': range(1, len(train_losses) + 1),
        'Train_Loss': train_losses,
        'Validation_Loss': val_losses
    })
    loss_data.to_csv(os.path.join(output_dir, f"loss_data_{exp_name}.csv"), index=False)

    metrics_df = pd.DataFrame({
        'Epoch': list(range(1, len(f1_list) + 1)),
        'Accuracy': accuracy_list,
        'Precision': precision_list,
        'Recall': recall_list,
        'F1': f1_list,
        'MCC': mcc_list,
        'AUROC': auroc_list,
        'AUPRC': auprc_list,
        'AP': ap_list,
        'Train_Loss': train_losses[:len(f1_list)],
        'Val_Loss': val_losses[:len(f1_list)],
    })
    metrics_df.to_csv(os.path.join(output_dir, f"val_metrics_{exp_name}.csv"), index=False)

    plt.figure()
    plt.plot(loss_data['Epoch'], loss_data['Train_Loss'], label='Train Loss')
    plt.plot(loss_data['Epoch'], loss_data['Validation_Loss'], label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training vs Validation Loss ({exp_name})")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"loss_curve_{exp_name}.png"))
    plt.close()

    metric_series = {
        "F1-score": f1_list,
        "Accuracy": accuracy_list,
        "Precision": precision_list,
        "Recall": recall_list,
        "MCC": mcc_list,
        "AUROC": auroc_list,
        "AUPRC": auprc_list
    }
    for metric_name, values in metric_series.items():
        if len(values) == 0:
            continue
        plt.figure()
        plt.plot(range(1, len(values) + 1), values, label=metric_name)
        plt.xlabel("Epoch")
        plt.ylabel(metric_name)
        plt.title(f"{metric_name} over Epochs (Validation, {exp_name})")
        plt.grid(True)
        plt.legend()
        filename = f"{metric_name.lower().replace('-', '_')}_curve_val_{exp_name}.png"
        plt.savefig(os.path.join(output_dir, filename))
        plt.close()

    print(f"[{exp_name}] Training completed! Results saved to: {output_dir}")
    print(f"[{exp_name}] Best model saved in: {checkpoint_dir}")


def resolve_requested_experiments():
    requested_raw = os.getenv("SEQPATHO_BENCHMARK_EXPERIMENTS", "").strip()
    all_experiments = [
        {"name": "cnn_emb_fixed", "model_type": "emb", "sliding": False},
        {"name": "cnn_emb_sliding", "model_type": "emb", "sliding": True},
        {"name": "cnn_onehot_fixed", "model_type": "onehot", "sliding": False},
        {"name": "cnn_onehot_sliding", "model_type": "onehot", "sliding": True},
    ]

    if not requested_raw:
        return all_experiments

    wanted = {name.strip() for name in requested_raw.split(",") if name.strip()}
    selected = [cfg for cfg in all_experiments if cfg["name"] in wanted]
    missing = sorted(wanted.difference({cfg["name"] for cfg in selected}))

    if missing:
        print(f"[WARN] Unknown benchmark experiment(s): {', '.join(missing)}")

    if not selected:
        raise ValueError("No valid benchmark experiments selected.")

    return selected


def main():
    # ========== 路径 ==========
    POS_PATH = r'E:\4PythonProject\SeqPatho\Dataset\Patho_plantlen2000.csv'
    NEG_PATH = r'E:\4PythonProject\SeqPatho\Data_processing\Nonpatho_all.csv'

    # ========== 加载 + canonical 冲突双边删除 + canonical 去重 ==========
    df_all = load_combined_data(POS_PATH, NEG_PATH, random_state=SEED)

    # ========== 偏差诊断（可选） ==========
    if ENABLE_BIAS_DIAGNOSTIC:
        _print_bias_diagnostic(df_all, topk=DIAG_TOPK)

    # ========== 长度匹配（可选，canonical 去重后） ==========
    if ENABLE_LENGTH_MATCH:
        df_all = length_match_after_dedup(df_all, n_bins=LENGTH_MATCH_BINS, random_state=SEED)
        try:
            stat = (
                df_all.assign(seq_len=df_all["seq"].str.len())
                      .groupby(["len_bin", "label"], observed=True)
                      .size()
                      .reset_index(name="count")
            )
            stat.to_csv(os.path.join(ROOT_RESULTS_DIR, "length_match_bin_counts.csv"), index=False)
        except Exception as e:
            print(f"[LenMatch] 长度分箱统计保存失败（不影响训练）：{e}")
    else:
        print("[LenMatch] 已关闭长度匹配。")

    # ========== GC 匹配（可选，canonical 去重后） ==========
    if ENABLE_GC_MATCH:
        df_all = gc_match_after_dedup(df_all, n_bins=GC_MATCH_BINS, random_state=SEED)
        try:
            stat = (
                df_all.assign(gc=df_all["seq"].map(gc_content))
                      .groupby(["gc_bin", "label"], observed=True)
                      .size()
                      .reset_index(name="count")
            )
            stat.to_csv(os.path.join(ROOT_RESULTS_DIR, "gc_match_bin_counts.csv"), index=False)
        except Exception as e:
            print(f"[GCMatch] GC 分箱统计保存失败（不影响训练）：{e}")
    else:
        print("[GCMatch] 已关闭 GC 匹配。")

    # ========== GroupSplit 70/15/15（保留 leak check） ==========
    df_train, df_val, df_test = group_split_70_15_15_frames(df_all, random_state=SEED)

    # ========== 训练集困难负样本采样（可选） ==========
    if ENABLE_HARD_NEG_TRAIN:
        df_train_used = build_hard_negative_trainset(
            df_train,
            random_state=HARD_NEG_RANDOM_STATE,
            min_pos=HARD_NEG_MIN_POS,
            min_pos_ratio=HARD_NEG_MIN_POS_RATIO
        )
    else:
        df_train_used = df_train

    train_sequences = df_train_used["seq"].tolist()
    y_train = df_train_used["label"].values

    val_sequences = df_val["seq"].tolist()
    y_val = df_val["label"].values

    test_sequences = df_test["seq"].tolist()
    y_test = df_test["label"].values

    print(
        f"[TrainUsed] train_used={len(train_sequences)} ({np.mean(y_train):.3f} pos rate), "
        f"val={len(val_sequences)} ({np.mean(y_val):.3f}), "
        f"test={len(test_sequences)} ({np.mean(y_test):.3f})"
    )

    # ========== 实验列表 ==========
    experiments = resolve_requested_experiments()

    for cfg in experiments:
        run_experiment(
            exp_name=cfg["name"],
            model_type=cfg["model_type"],
            sliding=cfg["sliding"],
            train_sequences=train_sequences, y_train=y_train,
            val_sequences=val_sequences, y_val=y_val,
            test_sequences=test_sequences, y_test=y_test
        )


if __name__ == "__main__":
    main()
