# -*- coding: utf-8 -*-
import os
import sys
from datetime import datetime
import random
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch import optim

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

from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler
import csv

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# =============================================================================
# USER CONFIG（与 DanQ/DeepSEA 修改版对齐）
# =============================================================================
SEED = apply_seed_override(42)

# 1) 模型（本地路径）
MODEL_NAME_OR_PATH = r"E:\4PythonProject\SeqPatho\CData\plant-dnagpt-6mer-promoter"

# 2) 数据路径
POS_FILE = r"E:\4PythonProject\SeqPatho\Dataset\Patho_plantlen2000.csv"
NEG_FILE = r"E:\4PythonProject\SeqPatho\Data_processing\Nonpatho_all.csv"

# 3) 训练超参
EPOCHS = 20
EPOCHS, EVAL_ONLY = apply_runtime_overrides(EPOCHS)
BATCH_SIZE_TRAIN = 16
BATCH_SIZE_EVAL = 16
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-2
WARMUP_RATIO = 0.05

# token 级最大长度
MAX_LENGTH_TOKENS = 500

# -------- 可选：长度匹配 / GC 匹配（冲突删除 + canonical 去重后、split 前） --------
ENABLE_LENGTH_MATCH = False
LENGTH_MATCH_BINS = 10

ENABLE_GC_MATCH = False
GC_MATCH_BINS = 10

# -------- 可选：偏差诊断输出 --------
ENABLE_BIAS_DIAGNOSTIC = True
DIAG_TOPK = 15

# -------- 可选：训练集困难负样本匹配 --------
ENABLE_HARD_NEG_TRAIN = True
HARD_NEG_RANDOM_STATE = SEED
HARD_NEG_MIN_POS = 200
HARD_NEG_MIN_POS_RATIO = 0.05

# =============================================================================
# 可复现性
# =============================================================================
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# =============================================================================
# 设备 & AMP
# =============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()
print(f"使用设备: {device} (AMP={'On' if USE_AMP else 'Off'})")
scaler = GradScaler(enabled=USE_AMP)

# =============================================================================
# 结果保存目录
# =============================================================================
try:
    script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0] or "interactive"
except Exception:
    script_name = "interactive"

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"results_PDLLM_{script_name}_{timestamp}"
os.makedirs(output_dir, exist_ok=True)
checkpoint_dir = os.path.join(output_dir, "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)

print(f"[Init] 输出目录: {output_dir}")

# =============================================================================
# DNA 工具函数
# =============================================================================
VALID_DNA_CHARS = set("ACGTN")
_RC_TABLE = str.maketrans("ACGTN", "TGCAN")


def clean_sequence(seq: str) -> str:
    """转大写，非 A/C/G/T/N 统一替换为 N。"""
    s = str(seq).strip().upper()
    if not s:
        return ""
    return "".join(ch if ch in VALID_DNA_CHARS else "N" for ch in s)


def rev_comp(seq: str) -> str:
    """反向互补，仅用于 canonical RC。"""
    s = str(seq)
    return s.translate(_RC_TABLE)[::-1]


def canon_rc(seq: str) -> str:
    """canonical RC：原序列与反向互补中字典序较小者。"""
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
    return (s.count("G") + s.count("C")) / L


# =============================================================================
# 早停
# =============================================================================
class EarlyStopping:
    def __init__(self, save_path, patience=8, verbose=True, delta=0.001):
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
        old_best = self.best_loss
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            if self.verbose:
                print(f"验证损失改善: {old_best:.6f} -> {self.best_loss:.6f}，保存模型...")
            torch.save(
                model.state_dict(),
                os.path.join(self.save_path, f"best_model_epoch{epoch + 1}.pth"),
            )
            self.counter = 0
            self.best_epoch = epoch
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# =============================================================================
# 加载 tokenizer & 模型
# =============================================================================
print(f"[Init] 加载 PDLLM 模型: {MODEL_NAME_OR_PATH}")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME_OR_PATH,
    trust_remote_code=True
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME_OR_PATH,
    num_labels=2,
    trust_remote_code=True
).to(device)

# 自动根据模型 config 调整 MAX_LENGTH_TOKENS（保守）
conf_max_pos = None
for attr in ["max_position_embeddings", "n_positions"]:
    if hasattr(model.config, attr):
        conf_max_pos = getattr(model.config, attr)
        break
if conf_max_pos is not None and conf_max_pos > 0:
    old_max = MAX_LENGTH_TOKENS
    MAX_LENGTH_TOKENS = min(MAX_LENGTH_TOKENS, int(conf_max_pos))
    print(f"[WARN] 模型位置上限={conf_max_pos}，已将 MAX_LENGTH_TOKENS {old_max}->{MAX_LENGTH_TOKENS}")
else:
    print(f"[INFO] 模型未提供 max_position_embeddings / n_positions，沿用 MAX_LENGTH_TOKENS={MAX_LENGTH_TOKENS}")

# =============================================================================
# 数据加载 + canonical 冲突删除 + 去重 + group_id
# =============================================================================
def _find_first_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_combined_data(file_a, file_b, random_state=SEED) -> pd.DataFrame:
    """
    与 DanQ/DeepSEA 修改版一致：
      - 不做类平衡下采样
      - 清洗 + 计算 gc + canonical
      - canonical 冲突（正负共存） -> 正负双方全部删除
      - canonical 去重
      - 构建 group_id
    """
    df_a = pd.read_csv(file_a, low_memory=False)
    df_b = pd.read_csv(file_b, low_memory=False)

    # 序列列：正类第 15 列(index=14)，负类第 14 列(index=13)
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
        f"载入原始序列(未下采样): 正类{len(pos)} / 负类{len(neg)}，总计{len(df_all)}；"
        "将进行清洗和 canonical RC 去重"
    )

    df_all["seq"] = df_all["seq_raw"].map(clean_sequence)
    df_all["seq_len"] = df_all["seq"].str.len()
    df_all = df_all[df_all["seq_len"] > 0].copy()

    df_all["gc"] = df_all["seq"].map(gc_content)
    df_all["canon"] = df_all["seq"].map(canon_rc)

    n_before = len(df_all)

    # canonical 冲突：正负双方全部删除
    conflict_groups = df_all.groupby("canon")["label"].nunique()
    conflict_canons = conflict_groups[conflict_groups > 1].index
    n_conflict = int(len(conflict_canons))
    if n_conflict > 0:
        print(f"[Dedup] 发现 {n_conflict} 个 canonical 组同时包含正负标签：正负双方全部删除。")
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
    df_all["label"] = df_all["label"].astype(int)

    print(f"[GroupKey] 唯一 group_id 数量: {df_all['group_id'].nunique()}")
    return df_all

# =============================================================================
# 可选：长度匹配 / GC 匹配（去重后、split 前）
# =============================================================================
def length_match_after_dedup(df_all: pd.DataFrame, n_bins: int = 10, random_state: int = SEED) -> pd.DataFrame:
    df = df_all.copy().reset_index(drop=True)
    try:
        df["len_bin"] = pd.qcut(df["seq_len"], q=n_bins, duplicates="drop")
    except Exception as e:
        print(f"[LenMatch] 分箱失败，跳过长度匹配。原因: {e}")
        return df_all

    parts = []
    kept_bins = 0
    dropped_bins = 0

    for _, g in df.groupby("len_bin", observed=True):
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
        print("[LenMatch] 警告：长度匹配后无可用 bin，返回原数据。")
        return df_all

    out = pd.concat(parts, ignore_index=True)\
            .sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    print(f"[LenMatch] 有效 bin={kept_bins}, 空/单边 bin={dropped_bins}")
    print(f"[LenMatch] 匹配后: pos={(out['label']==1).sum()}, neg={(out['label']==0).sum()}, total={len(out)}")
    return out


def gc_match_after_dedup(df_all: pd.DataFrame, n_bins: int = 10, random_state: int = SEED) -> pd.DataFrame:
    df = df_all.copy().reset_index(drop=True)
    try:
        df["gc_bin"] = pd.qcut(df["gc"], q=n_bins, duplicates="drop")
    except Exception as e:
        print(f"[GCMatch] 分箱失败，跳过 GC 匹配。原因: {e}")
        return df_all

    parts = []
    kept_bins = 0
    dropped_bins = 0

    for _, g in df.groupby("gc_bin", observed=True):
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
        print("[GCMatch] 警告：GC 匹配后无可用 bin，返回原数据。")
        return df_all

    out = pd.concat(parts, ignore_index=True)\
            .sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    print(f"[GCMatch] 有效 bin={kept_bins}, 空/单边 bin={dropped_bins}")
    print(f"[GCMatch] 匹配后: pos={(out['label']==1).sum()}, neg={(out['label']==0).sum()}, total={len(out)}")
    return out

# =============================================================================
# 可选：偏差诊断
# =============================================================================
def print_bias_diagnostic(df_all: pd.DataFrame, topk: int = 15):
    try:
        df = df_all.copy()
        print("[Diag] seq_len by label:")
        print(df.groupby("label")["seq_len"].describe())

        if "gc" in df.columns:
            print("[Diag] GC by label:")
            print(df.groupby("label")["gc"].describe())

        if "tax_id" in df.columns:
            tmp = df.dropna(subset=["tax_id"]).copy()
            if len(tmp) > 0:
                purity = tmp.groupby("tax_id")["label"].mean().sort_values()
                print(f"[Diag] tax_id label purity (lowest {topk}):")
                print(purity.head(topk))
                print(f"[Diag] tax_id label purity (highest {topk}):")
                print(purity.tail(topk))
    except Exception as e:
        print(f"[Diag] 偏差诊断失败（不影响训练）：{e}")

# =============================================================================
# GroupSplit 70/15/15（frame 版本，支持 HardNeg）
# =============================================================================
def group_split_70_15_15_frames(df_all: pd.DataFrame, random_state=SEED):
    groups = df_all["group_id"].values
    labels = df_all["label"].values

    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=random_state)
    train_idx, temp_idx = next(gss1.split(df_all, labels, groups))

    df_train = df_all.iloc[train_idx].reset_index(drop=True)
    df_temp = df_all.iloc[temp_idx].reset_index(drop=True)

    groups_temp = df_temp["group_id"].values
    labels_temp = df_temp["label"].values
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=random_state + 1)
    val_idx, test_idx = next(gss2.split(df_temp, labels_temp, groups_temp))

    df_val = df_temp.iloc[val_idx].reset_index(drop=True)
    df_test = df_temp.iloc[test_idx].reset_index(drop=True)

    def _log_split(name, frame):
        pos_rate = frame["label"].mean() if len(frame) else 0.0
        n_samples = len(frame)
        n_groups = frame["group_id"].nunique()
        print(f"[GroupSplit] {name}: 样本={n_samples}, 组数={n_groups}, pos_rate={pos_rate:.3f}")

    _log_split("train", df_train)
    _log_split("val", df_val)
    _log_split("test", df_test)

    # group leakage check
    g_tr = set(df_train["group_id"])
    g_va = set(df_val["group_id"])
    g_te = set(df_test["group_id"])
    print(
        f"[GroupSplit] 交叉 group_id：train∩val={len(g_tr & g_va)}, "
        f"train∩test={len(g_tr & g_te)}, val∩test={len(g_va & g_te)}"
    )
    assert len(g_tr & g_va) == 0 and len(g_tr & g_te) == 0 and len(g_va & g_te) == 0, \
        "GroupSplit 出现跨集共享 group_id！"

    # canonical leakage check
    c_tr = set(df_train["canon"])
    c_va = set(df_val["canon"])
    c_te = set(df_test["canon"])
    print(
        f"[LeakCheck] canonical RC 交叉：train∩val={len(c_tr & c_va)}, "
        f"train∩test={len(c_tr & c_te)}, val∩test={len(c_va & c_te)}"
    )
    assert len(c_tr & c_va) == 0 and len(c_tr & c_te) == 0 and len(c_va & c_te) == 0, \
        "存在跨集完全相同的 canonical 序列！"

    return df_train, df_val, df_test

# =============================================================================
# 可选：训练集困难负样本匹配（仅重建 train）
# =============================================================================
def build_hard_negative_trainset(df_train: pd.DataFrame,
                                 random_state: int = SEED,
                                 min_pos: int = 200,
                                 min_pos_ratio: float = 0.05) -> pd.DataFrame:
    df = df_train.copy().reset_index(drop=True)
    total_pos = int((df["label"] == 1).sum())
    total_neg = int((df["label"] == 0).sum())
    print(f"[HardNeg] 原训练集: pos={total_pos}, neg={total_neg}, total={len(df)}")

    def _match_by_key(key_col: str):
        if key_col not in df.columns:
            return None

        tmp = df.copy()
        tmp[key_col] = tmp[key_col].astype(str).fillna("NA")

        counts = tmp.groupby([key_col, "label"]).size().unstack(fill_value=0)
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
            g = tmp[tmp[key_col] == k]
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

        out = pd.concat(parts, ignore_index=True)\
                .sample(frac=1.0, random_state=random_state).reset_index(drop=True)
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

    df_hard = _match_by_key("group_id")

    if not _enough(df_hard) and "tax_id" in df.columns:
        df_hard = _match_by_key("tax_id")

    if not _enough(df_hard) and "species_name" in df.columns:
        df_hard = _match_by_key("species_name")

    if not _enough(df_hard):
        df_hard = _match_by_key("canon")

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

# =============================================================================
# Dataset & collate
# =============================================================================
class SeqDataset(Dataset):
    def __init__(self, sequences, labels):
        self.seqs = list(sequences)
        self.labels = np.array(labels).astype(int)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx], int(self.labels[idx])


def collate_fn(batch):
    seqs, ys = zip(*batch)
    enc = tokenizer(
        list(seqs),
        padding="longest",
        truncation=True,
        max_length=MAX_LENGTH_TOKENS,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    labels = torch.tensor(ys, dtype=torch.long)
    return input_ids, attention_mask, labels

# =============================================================================
# 评估工具（加单一类别保护）
# =============================================================================
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

    auroc = float("nan")
    auprc = float("nan")
    ap = float("nan")

    try:
        if len(np.unique(labels)) > 1:
            fpr, tpr, _ = roc_curve(labels, probs)
            auroc = auc(fpr, tpr)
    except Exception:
        pass

    try:
        if len(np.unique(labels)) > 1:
            prec_vals, rec_vals, _ = precision_recall_curve(labels, probs)
            auprc = auc(rec_vals, prec_vals)
            ap = average_precision_score(labels, probs)
    except Exception:
        pass

    return dict(
        accuracy=accuracy, precision=precision, recall=recall, f1=f1,
        mcc=mcc, auroc=auroc, auprc=auprc, ap=ap
    )


@torch.inference_mode()
def evaluate_loader(model, data_loader):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_labels, all_preds, all_probs = [], [], []

    for input_ids, attention_mask, labels in data_loader:
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(enabled=USE_AMP):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

        total_loss += loss.item()
        n_batches += 1

        probs = F.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
        preds = torch.argmax(logits, dim=-1).detach().cpu().numpy()
        ys = labels.detach().cpu().numpy()

        all_labels.extend(ys.tolist())
        all_preds.extend(preds.tolist())
        all_probs.extend(probs.tolist())

    avg_loss = total_loss / max(1, n_batches)
    return avg_loss, np.array(all_labels), np.array(all_preds), np.array(all_probs)

# =============================================================================
# Pipeline: load -> diag -> (len/gc match) -> split -> hard-neg(train only)
# =============================================================================
df_all = load_combined_data(POS_FILE, NEG_FILE, random_state=SEED)

if ENABLE_BIAS_DIAGNOSTIC:
    print_bias_diagnostic(df_all, topk=DIAG_TOPK)

if ENABLE_LENGTH_MATCH:
    df_all = length_match_after_dedup(df_all, n_bins=LENGTH_MATCH_BINS, random_state=SEED)
else:
    print("[LenMatch] 已关闭长度匹配。")

if ENABLE_GC_MATCH:
    df_all = gc_match_after_dedup(df_all, n_bins=GC_MATCH_BINS, random_state=SEED)
else:
    print("[GCMatch] 已关闭 GC 匹配。")

df_train, df_val, df_test = group_split_70_15_15_frames(df_all, random_state=SEED)

if ENABLE_HARD_NEG_TRAIN:
    df_train_used = build_hard_negative_trainset(
        df_train,
        random_state=HARD_NEG_RANDOM_STATE,
        min_pos=HARD_NEG_MIN_POS,
        min_pos_ratio=HARD_NEG_MIN_POS_RATIO
    )
else:
    df_train_used = df_train
    print("[HardNeg] 已关闭训练集困难负样本匹配。")

train_sequences = df_train_used["seq"].tolist()
val_sequences = df_val["seq"].tolist()
test_sequences = df_test["seq"].tolist()

y_train = df_train_used["label"].values
y_val = df_val["label"].values
y_test = df_test["label"].values

print(
    f"[TrainUsed] train_used={len(train_sequences)} ({np.mean(y_train):.3f} pos rate), "
    f"val={len(val_sequences)} ({np.mean(y_val):.3f}), "
    f"test={len(test_sequences)} ({np.mean(y_test):.3f})"
)

# =============================================================================
# DataLoaders
# =============================================================================
train_loader = DataLoader(
    SeqDataset(train_sequences, y_train),
    batch_size=BATCH_SIZE_TRAIN,
    shuffle=True,
    collate_fn=collate_fn,
    pin_memory=torch.cuda.is_available(),
    num_workers=0,
)
val_loader = DataLoader(
    SeqDataset(val_sequences, y_val),
    batch_size=BATCH_SIZE_EVAL,
    shuffle=False,
    collate_fn=collate_fn,
    pin_memory=torch.cuda.is_available(),
    num_workers=0,
)
test_loader = DataLoader(
    SeqDataset(test_sequences, y_test),
    batch_size=BATCH_SIZE_EVAL,
    shuffle=False,
    collate_fn=collate_fn,
    pin_memory=torch.cuda.is_available(),
    num_workers=0,
)

# =============================================================================
# 优化器 & 调度器
# =============================================================================
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

num_update_steps_per_epoch = len(train_loader)
num_train_steps = EPOCHS * num_update_steps_per_epoch
num_warmup_steps = max(1, int(WARMUP_RATIO * num_train_steps))

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_train_steps,
)

# =============================================================================
# 训练监控
# =============================================================================
train_losses, val_losses = [], []
accuracy_list, precision_list, recall_list, f1_list, mcc_list = [], [], [], [], []
auroc_list, auprc_list, ap_list = [], [], []
epoch_lrs = []

early_stopping = EarlyStopping(save_path=checkpoint_dir, patience=8, verbose=True, delta=0.001)

metrics_file_path = os.path.join(output_dir, "metrics_per_epoch.txt")
with open(metrics_file_path, "w", encoding="utf-8") as f:
    f.write(
        "Epoch\tTrain_Loss\tVal_Loss\tAccuracy\tPrecision\tRecall\tF1\tMCC\t"
        "AUROC\tAUPRC\tAP\tLR\n"
    )

# =============================================================================
# 训练循环
# =============================================================================
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    seen_batches = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{EPOCHS}]", leave=False)

    for input_ids, attention_mask, labels in progress_bar:
        input_ids = input_ids.to(device, non_blocking=True)
        attention_mask = attention_mask.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=USE_AMP):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        seen_batches += 1
        progress_bar.set_postfix(loss=(total_loss / max(1, seen_batches)))

    avg_train_loss = total_loss / max(1, seen_batches)
    train_losses.append(avg_train_loss)

    # 验证
    val_loss, val_labels, val_preds, val_probs = evaluate_loader(model, val_loader)
    val_losses.append(val_loss)
    val_metrics = compute_classification_metrics(val_labels, val_preds, val_probs)

    lr_now = optimizer.param_groups[0]["lr"]
    epoch_lrs.append(lr_now)

    accuracy_list.append(val_metrics["accuracy"])
    precision_list.append(val_metrics["precision"])
    recall_list.append(val_metrics["recall"])
    f1_list.append(val_metrics["f1"])
    mcc_list.append(val_metrics["mcc"])
    auroc_list.append(val_metrics["auroc"])
    auprc_list.append(val_metrics["auprc"])
    ap_list.append(val_metrics["ap"])

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] - "
        f"Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, "
        f"Acc: {val_metrics['accuracy']:.4f}, P: {val_metrics['precision']:.4f}, "
        f"R: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}, "
        f"MCC: {val_metrics['mcc']:.4f}, AUROC: {val_metrics['auroc']:.4f}, "
        f"AUPRC: {val_metrics['auprc']:.4f}, AP: {val_metrics['ap']:.4f}, "
        f"LR: {lr_now:.2e}"
    )

    with open(metrics_file_path, "a", encoding="utf-8") as f:
        f.write(
            f"{epoch + 1}\t{avg_train_loss:.4f}\t{val_loss:.4f}\t"
            f"{val_metrics['accuracy']:.4f}\t{val_metrics['precision']:.4f}\t"
            f"{val_metrics['recall']:.4f}\t{val_metrics['f1']:.4f}\t{val_metrics['mcc']:.4f}\t"
            f"{val_metrics['auroc']:.4f}\t{val_metrics['auprc']:.4f}\t{val_metrics['ap']:.4f}\t"
            f"{lr_now:.6e}\n"
        )

    # 早停
    early_stopping(val_loss, model, epoch)
    if early_stopping.early_stop:
        print(f"早停触发于第 {epoch + 1} 轮，最佳模型来自第 {early_stopping.best_epoch + 1} 轮")
        break

# =============================================================================
# 加载最佳模型并在 TEST 上评估
# =============================================================================
best_epoch_idx = early_stopping.best_epoch
best_ckpt_path = resolve_checkpoint_path(os.path.join(checkpoint_dir, f"best_model_epoch{best_epoch_idx + 1}.pth"))

if os.path.exists(best_ckpt_path):
    model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    print(f"[Load Best] 加载最佳验证轮权重：{best_ckpt_path}")
else:
    print("[Load Best] 未找到最佳权重文件，使用当前模型参数进行测试。")

test_loss, test_labels, test_preds, test_probs = evaluate_loader(model, test_loader)
test_metrics = compute_classification_metrics(test_labels, test_preds, test_probs)

# 画 ROC/PR 前的保护
if len(test_labels) > 0 and len(np.unique(test_labels)) > 1:
    fpr, tpr, _ = roc_curve(test_labels, test_probs)
    auroc_final = auc(fpr, tpr)
    precision_vals, recall_vals, _ = precision_recall_curve(test_labels, test_probs)
    ap_final = average_precision_score(test_labels, test_probs)
    auprc_final = auc(recall_vals, precision_vals)

    save_curve_points(output_dir, fpr=fpr, tpr=tpr, recall=recall_vals, precision=precision_vals)
else:
    auroc_final = float("nan")
    ap_final = float("nan")
    auprc_final = float("nan")
    fpr, tpr = np.array([0, 1]), np.array([0, 1])
    precision_vals, recall_vals = np.array([1.0]), np.array([0.0])

# =============================================================================
# Plots
# =============================================================================
cm = confusion_matrix(test_labels, test_preds)
plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (Test)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "confusion_matrix_test.png"))
plt.close()

plt.figure()
plt.plot(fpr, tpr, lw=2, label=f"ROC (AUROC = {auroc_final:.2f})")
plt.plot([0, 1], [0, 1], lw=2, linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Test)")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "roc_curve_test.png"))
plt.close()

plt.figure()
plt.step(
    recall_vals, precision_vals, where="post", lw=2,
    label=f"PR (AP = {ap_final:.2f}, AUPRC = {auprc_final:.2f})"
)
plt.fill_between(recall_vals, precision_vals, step="post", alpha=0.2)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("PR Curve (Test)")
plt.legend(loc="upper right")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "pr_curve_test.png"))
plt.close()

# =============================================================================
# Save metrics
# =============================================================================
final_accuracy = test_metrics["accuracy"]
final_precision = test_metrics["precision"]
final_recall = test_metrics["recall"]
final_f1 = test_metrics["f1"]
final_mcc = test_metrics["mcc"]
avg_train_loss_last = train_losses[-1] if len(train_losses) > 0 else float("nan")
avg_val_loss_last = val_losses[-1] if len(val_losses) > 0 else float("inf")

metrics_text = f"""Final Test Metrics (evaluated on TEST set only):
Train Loss (last epoch): {avg_train_loss_last:.4f}
Validation Loss (last epoch): {avg_val_loss_last:.4f}
Test Loss: {test_loss:.4f}
Accuracy: {final_accuracy:.4f}
Precision: {final_precision:.4f}
Recall: {final_recall:.4f}
F1-score: {final_f1:.4f}
MCC: {final_mcc:.4f}
AUROC: {auroc_final:.4f}
AUPRC: {auprc_final:.4f}
Average Precision (AP): {ap_final:.4f}
Early Stopping Triggered: {'Yes' if early_stopping.early_stop else 'No'}
Best Epoch (by val loss): {early_stopping.best_epoch + 1}

MAX_LENGTH_TOKENS: {MAX_LENGTH_TOKENS}
Length Match Enabled: {ENABLE_LENGTH_MATCH}
Length Match Bins: {LENGTH_MATCH_BINS}
GC Match Enabled: {ENABLE_GC_MATCH}
GC Match Bins: {GC_MATCH_BINS}
HardNeg Train Enabled: {ENABLE_HARD_NEG_TRAIN}
"""
with open(os.path.join(output_dir, "final_metrics_test.txt"), "w", encoding="utf-8") as f:
    f.write(metrics_text)

with open(os.path.join(output_dir, "final_metrics_test.csv"), mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Metric", "Value"])
    writer.writerow(["Train Loss (last)", avg_train_loss_last])
    writer.writerow(["Validation Loss (last)", avg_val_loss_last])
    writer.writerow(["Test Loss", test_loss])
    writer.writerow(["Accuracy", final_accuracy])
    writer.writerow(["Precision", final_precision])
    writer.writerow(["Recall", final_recall])
    writer.writerow(["F1-score", final_f1])
    writer.writerow(["MCC", final_mcc])
    writer.writerow(["AUROC", auroc_final])
    writer.writerow(["AUPRC", auprc_final])
    writer.writerow(["Average Precision (AP)", ap_final])
    writer.writerow(["Early Stopping Triggered", "Yes" if early_stopping.early_stop else "No"])
    writer.writerow(["Best Epoch (val)", early_stopping.best_epoch + 1])
    writer.writerow(["MAX_LENGTH_TOKENS", MAX_LENGTH_TOKENS])
    writer.writerow(["Length Match Enabled", ENABLE_LENGTH_MATCH])
    writer.writerow(["Length Match Bins", LENGTH_MATCH_BINS])
    writer.writerow(["GC Match Enabled", ENABLE_GC_MATCH])
    writer.writerow(["GC Match Bins", GC_MATCH_BINS])
    writer.writerow(["HardNeg Train Enabled", ENABLE_HARD_NEG_TRAIN])

loss_data = pd.DataFrame({
    "Epoch": range(1, len(train_losses) + 1),
    "Train_Loss": train_losses,
    "Validation_Loss": val_losses[:len(train_losses)],
    "LR": epoch_lrs[:len(train_losses)],
})
loss_data.to_csv(os.path.join(output_dir, "loss_data.csv"), index=False)

metrics_df = pd.DataFrame({
    "Epoch": list(range(1, len(f1_list) + 1)),
    "Accuracy": accuracy_list,
    "Precision": precision_list,
    "Recall": recall_list,
    "F1": f1_list,
    "MCC": mcc_list,
    "AUROC": auroc_list,
    "AUPRC": auprc_list,
    "AP": ap_list,
    "Train_Loss": train_losses[:len(f1_list)],
    "Val_Loss": val_losses[:len(f1_list)],
    "LR": epoch_lrs[:len(f1_list)],
})
metrics_df.to_csv(os.path.join(output_dir, "val_metrics.csv"), index=False)

for metric_name, values in {
    "F1-score": f1_list,
    "Accuracy": accuracy_list,
    "Precision": precision_list,
    "Recall": recall_list,
    "MCC": mcc_list,
    "AUROC": auroc_list,
    "AUPRC": auprc_list,
}.items():
    if len(values) == 0:
        continue
    plt.figure()
    plt.plot(range(1, len(values) + 1), values, label=metric_name)
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.title(f"{metric_name} over Epochs (Validation)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    filename = f"{metric_name.lower().replace('-', '_')}_curve_val.png"
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

if len(epoch_lrs) > 0:
    plt.figure()
    plt.plot(range(1, len(epoch_lrs) + 1), epoch_lrs, label="LR")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate over Epochs")
    plt.yscale("log")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "lr_curve.png"))
    plt.close()

plt.figure()
plt.plot(loss_data["Epoch"], loss_data["Train_Loss"], label="Train Loss")
plt.plot(loss_data["Epoch"], loss_data["Validation_Loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "loss_curve.png"))
plt.close()

print(f"Training completed! Results saved to: {output_dir}")
print(f"Best model saved in: {checkpoint_dir}")
print("注意：训练期指标基于 Validation；最终报告指标基于 Test。")
