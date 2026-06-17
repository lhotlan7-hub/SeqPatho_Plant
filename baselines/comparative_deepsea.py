# -*- coding: utf-8 -*-

# =========================================================
#  Windows 超强稳定性保护（必须放在所有科学计算库 import 之前）
# =========================================================
import os

SAFE_MODE = True

# ---- 统一限制并行线程（Windows 下止血核心）----
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

# ---- OpenMP / KMP 进一步“止血级”设置 ----
# 这些组合在 Windows + Anaconda 混装生态下对 0xC0000005 很关键
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["KMP_BLOCKTIME"] = "0"
os.environ["KMP_AFFINITY"] = "disabled"
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"

# ---- MKL 线程层尽量走最保守路径 ----
os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"
# 不建议默认强制此项；某些机器/驱动组合可能更不稳
# os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"

# ---- HuggingFace tokenizer 并行关闭（即便当前脚本不用也无害） ----
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---- 可选：降低 CUDA 内存碎片风险（不一定需要，SAFE_MODE 保守开启）----
# 若你发现显存分配相关异常，可尝试打开
# os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

# ---- 彻底避免交互式图形后端 ----
import matplotlib
matplotlib.use("Agg")

# =========================================================
#  常规 import
# =========================================================
import sys
from datetime import datetime
import random
import warnings
import gc
import csv as pycsv

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

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

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# =========================================================
#  CSV 超长字段上限保护（必须放在任何 csv.reader 之前）
# =========================================================
def _set_csv_field_size_limit():
    """
    解决 _csv.Error: field larger than field limit (131072)
    使用“固定大值优先 + 逐级回退”的方式，最大化兼容性。
    """
    try:
        pycsv.field_size_limit(1024 * 1024 * 1024)  # 1GB
        return
    except Exception:
        pass

    max_int = sys.maxsize
    while True:
        try:
            pycsv.field_size_limit(max_int)
            break
        except OverflowError:
            max_int = int(max_int / 10)
        except Exception:
            try:
                pycsv.field_size_limit(10 * 1024 * 1024)  # 10MB
            except Exception:
                pass
            break

_set_csv_field_size_limit()

# =========================================================
#  可复现性
# =========================================================
SEED = apply_seed_override(42)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ---- SAFE_MODE 下进一步限制 PyTorch 线程 ----
if SAFE_MODE:
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

# =========================================================
#  设备
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = False
print(f"使用设备: {device} (AMP={'On' if USE_AMP else 'Off'})")
print(f"[SAFE_MODE] {'ON' if SAFE_MODE else 'OFF'}")

# =========================================================
#  配置
# =========================================================
POS_FILE = r"E:\4PythonProject\SeqPatho\Dataset\Patho_plantlen2000.csv"
NEG_FILE = r"E:\4PythonProject\SeqPatho\Data_processing\Nonpatho_all.csv"

SEQ_LEN = 50

EPOCHS = 80
EPOCHS, EVAL_ONLY = apply_runtime_overrides(EPOCHS)
BATCH_SIZE_TRAIN = 512
BATCH_SIZE_EVAL = 512
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
WARMUP_RATIO = 0.05

# ---- 功能开关 ----
ENABLE_CANON_CONFLICT_DROP = True

# 长度/GC 匹配
# 更严格评估建议：只对训练集做（main 中已实现 train-only）
ENABLE_LENGTH_MATCH = True
LENGTH_MATCH_BINS = 10

ENABLE_GC_MATCH = True
GC_MATCH_BINS = 10

# 偏差诊断输出
ENABLE_BIAS_DIAG = False
DIAG_TOPK = 15

# 训练集困难负样本采样（成对匹配式）
ENABLE_HARD_NEG_TRAIN = False if SAFE_MODE else False
HARD_NEG_RANDOM_STATE = SEED
HARD_NEG_MIN_POS = 200
HARD_NEG_MIN_POS_RATIO = 0.05

# DataLoader 的 pin_memory：SAFE_MODE 下关闭
PIN_MEMORY = False if SAFE_MODE else torch.cuda.is_available()

# ---- 负类保留上限（SAFE_MODE 更保守，降低大 set/大 DF 峰值风险）----
MAX_NEG_KEEP = 50_000 if SAFE_MODE else None
NEG_POS_RATIO_KEEP = None  # 例如 3.0

# ---- Pass2 内部分块缓存行数（SAFE_MODE 更保守）----
PART_ROWS = 20_000 if SAFE_MODE else 50_000

# ---- 可选：超长序列硬过滤（更偏“数据卫生”）----
MAX_SEQ_LEN_HARD = 100_000 if SAFE_MODE else None

# =========================================================
#  更严格评估协议（为了让指标更真实且通常更低）
# =========================================================
# 严格单层级 group_id，避免混用 tax/species/canon 的分组强度不一致
# 可选: "tax_id" | "species_name" | "comp_tax" | "canon"
GROUP_LEVEL = "tax_id"
STRICT_GROUP_FALLBACK_TO_CANON = True

# 外层 holdout：先按 tax/species 留出一块“超严格测试”
ENABLE_OUTER_TEST_HOLDOUT = True
OUTER_TEST_RATIO = 0.15
OUTER_GROUP_LEVEL = "tax_id"  # 或 "species_name"

# =========================================================
#  DNA 工具函数
# =========================================================
VALID_DNA_CHARS = set("ACGTN")
_RC_TABLE = str.maketrans("ACGTN", "TGCAN")


def clean_sequence(seq: str) -> str:
    s = str(seq).strip().upper()
    if not s:
        return ""
    return "".join(ch if ch in VALID_DNA_CHARS else "N" for ch in s)


def rev_comp(seq: str) -> str:
    s = str(seq)
    return s.translate(_RC_TABLE)[::-1]


def canon_rc(seq: str) -> str:
    s = clean_sequence(seq)
    if not s:
        return ""
    rc = rev_comp(s)
    return s if s <= rc else rc


def pad_or_crop_center(seq: str, L: int) -> str:
    s = clean_sequence(seq)
    if len(s) == L:
        return s
    if len(s) < L:
        return s + "N" * (L - len(s))
    start = (len(s) - L) // 2
    return s[start:start + L]


def random_crop_or_pad(seq: str, L: int) -> str:
    s = clean_sequence(seq)
    if len(s) == L:
        return s
    if len(s) < L:
        return s + "N" * (L - len(s))
    start = random.randint(0, len(s) - L)
    return s[start:start + L]


def gc_content(seq: str) -> float:
    s = clean_sequence(seq)
    L = len(s)
    if L == 0:
        return 0.0
    return float(s.count("G") + s.count("C")) / float(L)


# =========================================================
#  One-hot 编码
# =========================================================
NUC_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}


def one_hot_encode(seq: str, L: int = SEQ_LEN) -> torch.Tensor:
    assert len(seq) == L
    arr = torch.zeros(4, L, dtype=torch.float32)
    for i, ch in enumerate(seq):
        idx = NUC_TO_IDX.get(ch, None)
        if idx is not None:
            arr[idx, i] = 1.0
    return arr


# =========================================================
#  早停
# =========================================================
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


# =========================================================
#  输出目录
# =========================================================
def make_output_dirs():
    try:
        script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0] or "interactive"
    except Exception:
        script_name = "interactive"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results_DeepSEA_{script_name}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"[Init] 输出目录: {output_dir}")
    return output_dir, checkpoint_dir


def safe_savefig(path):
    try:
        plt.tight_layout()
        plt.savefig(path)
    except Exception as e:
        print(f"[PlotWarn] 保存图失败: {os.path.basename(path)} | {repr(e)}")
    finally:
        plt.close()


# =========================================================
#  CSV（纯 Python）辅助
# =========================================================
def _find_first_col_in_header(header, candidates):
    lower = {h.lower(): i for i, h in enumerate(header)}
    for c in candidates:
        idx = lower.get(c.lower(), None)
        if idx is not None:
            return idx
    return None


def _safe_str_val(x):
    if x is None:
        return pd.NA
    s = str(x).strip()
    return s if s else pd.NA


def read_csv_header(path):
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = pycsv.reader(f)
        header = next(reader)
    return header


def iter_csv_rows(path):
    """
    最稳读取：utf-8 + replace，避免异常编码导致崩溃。
    """
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = pycsv.reader(f)
        _ = next(reader, None)  # skip header
        for row in reader:
            yield row


# =========================================================
#  数据加载 + canonical 处理（最稳：纯 Python 两遍流式）
# =========================================================
def load_combined_data(file_a, file_b,
                       max_neg_keep: int = None,
                       neg_pos_ratio_keep: float = None) -> pd.DataFrame:
    """
    最稳版本：
      - 不使用 pandas.read_csv 扫描大文件
      - Pass-1:
          * 构建 pos_canons
          * 扫描 neg 仅检测是否与 pos_canons 重叠 -> conflict_canons
      - Pass-2:
          * 冲突删除（可选）
          * 全局 canonical 去重
          * 逐步累积为 DataFrame parts，降低峰值
    """

    header_a = read_csv_header(file_a)
    header_b = read_csv_header(file_b)

    try:
        seq_col_a_name = header_a[14]
        seq_col_b_name = header_b[13]
        seq_idx_a = 14
        seq_idx_b = 13
    except Exception as e:
        raise RuntimeError(f"无法按固定列号定位序列列。请检查 CSV 列数。原始错误: {e}")

    tax_idx_a = _find_first_col_in_header(header_a, ["tax_id", "#tax_id", "Tax_ID", "taxid"])
    tax_idx_b = _find_first_col_in_header(header_b, ["tax_id", "#tax_id", "Tax_ID", "taxid"])
    species_idx_a = _find_first_col_in_header(header_a, ["Species_Name", "species_name", "Species", "species"])
    species_idx_b = _find_first_col_in_header(header_b, ["Species_Name", "species_name", "Species", "species"])
    comp_idx_a = _find_first_col_in_header(header_a, ["comp_id"])
    comp_idx_b = _find_first_col_in_header(header_b, ["comp_id"])

    print("载入原始序列(未下采样): 正负规模将进行纯 Python 流式扫描；将进行清洗和 canonical 处理")
    print(f"[Header] POS seq col = #{seq_idx_a} ({seq_col_a_name}) | NEG seq col = #{seq_idx_b} ({seq_col_b_name})")

    # =====================================================
    # Pass-1: 只存 pos_canons，再检测 neg 重叠 -> conflict
    # =====================================================
    print("[Load/Pass1] 扫描正类以构建 pos_canons...")
    pos_canons = set()

    pos_rows_seen = 0
    for row in iter_csv_rows(file_a):
        pos_rows_seen += 1
        if seq_idx_a >= len(row):
            continue
        c = canon_rc(row[seq_idx_a])
        if c:
            pos_canons.add(c)

        if SAFE_MODE and (pos_rows_seen % 200000 == 0):
            gc.collect()

    print(f"[Load/Pass1] 正类扫描完成：rows={pos_rows_seen}, unique_pos_canons={len(pos_canons)}")

    print("[Load/Pass1] 扫描负类以定位 canonical 正负冲突...")
    conflict_canons = set()

    neg_rows_seen = 0
    for row in iter_csv_rows(file_b):
        neg_rows_seen += 1
        if seq_idx_b >= len(row):
            continue
        c = canon_rc(row[seq_idx_b])
        if c and c in pos_canons:
            conflict_canons.add(c)

        if SAFE_MODE and (neg_rows_seen % 200000 == 0):
            gc.collect()

    print(f"[CanonConflict/Pass1] 发现 canonical 正负冲突组数量: {len(conflict_canons)}")

    del pos_canons
    gc.collect()

    # =====================================================
    # Pass-2: 逐行构建最终数据（冲突删除 + 全局去重）
    # =====================================================
    print("[Load/Pass2] 构建最终数据（冲突删除 + 全局 canonical 去重）...")
    seen_canons = set()
    parts = []
    rows_buf = []

    pos_kept = 0
    neg_kept = 0

    def _flush_buf():
        nonlocal rows_buf, parts
        if rows_buf:
            parts.append(pd.DataFrame(rows_buf))
            rows_buf = []

    def _process_file(path, seq_idx, label, tax_idx, species_idx, comp_idx):
        nonlocal pos_kept, neg_kept, rows_buf

        rows_seen = 0

        for row in iter_csv_rows(path):
            rows_seen += 1
            if seq_idx >= len(row):
                continue

            raw = row[seq_idx]
            c = canon_rc(raw)
            if not c:
                continue

            if ENABLE_CANON_CONFLICT_DROP and (c in conflict_canons):
                continue

            if c in seen_canons:
                continue
            seen_canons.add(c)

            seq_clean = clean_sequence(raw)
            L = len(seq_clean)
            if L <= 0:
                continue

            if MAX_SEQ_LEN_HARD is not None and L > MAX_SEQ_LEN_HARD:
                continue

            if label == 0:
                if max_neg_keep is not None and neg_kept >= max_neg_keep:
                    continue
                if neg_pos_ratio_keep is not None and pos_kept > 0:
                    if neg_kept >= int(neg_pos_ratio_keep * pos_kept):
                        continue

            rec = {
                "seq": seq_clean,
                "seq_len": L,
                "canon": c,
                "gc": gc_content(seq_clean),
                "label": int(label),
            }

            if tax_idx is not None and tax_idx < len(row):
                rec["tax_id"] = row[tax_idx]
            if species_idx is not None and species_idx < len(row):
                rec["species_name"] = row[species_idx]
            if comp_idx is not None and comp_idx < len(row):
                rec["comp_id"] = row[comp_idx]

            rows_buf.append(rec)

            if label == 1:
                pos_kept += 1
            else:
                neg_kept += 1

            if len(rows_buf) >= PART_ROWS:
                _flush_buf()

            if SAFE_MODE and (rows_seen % 200000 == 0):
                gc.collect()

        return rows_seen

    print("[Load/Pass2] 处理正类...")
    pos_seen = _process_file(file_a, seq_idx_a, 1, tax_idx_a, species_idx_a, comp_idx_a)
    _flush_buf()
    gc.collect()

    print("[Load/Pass2] 处理负类...")
    neg_seen = _process_file(file_b, seq_idx_b, 0, tax_idx_b, species_idx_b, comp_idx_b)
    _flush_buf()
    gc.collect()

    if not parts:
        raise RuntimeError("Pass-2 未产生任何有效数据，请检查输入文件与序列列。")

    df_all = pd.concat(parts, ignore_index=True).reset_index(drop=True)
    del parts, rows_buf
    gc.collect()

    if "tax_id" in df_all.columns:
        df_all["tax_id"] = df_all["tax_id"].map(_safe_str_val).astype("string")
    if "species_name" in df_all.columns:
        df_all["species_name"] = df_all["species_name"].map(_safe_str_val).astype("string")
    if "comp_id" in df_all.columns:
        df_all["comp_id"] = df_all["comp_id"].map(_safe_str_val).astype("string")

    df_all["group_id"] = df_all["canon"].astype("string")

    print(f"[Load/Done] POS rows seen={pos_seen}, NEG rows seen={neg_seen}")
    print(f"[Load/Done] 去冲突+去重后：pos={int(df_all['label'].sum())}, "
          f"neg={int((1-df_all['label']).sum())}, total={len(df_all)}")
    print(f"[GroupKey/Temp] 当前 group_id=canon，唯一数: {df_all['group_id'].nunique()}")

    return df_all


# =========================================================
#  严格单层级 group_id 重建
# =========================================================
def rebuild_group_id_strict(df: pd.DataFrame,
                            level: str = "tax_id",
                            fallback_to_canon: bool = True) -> pd.DataFrame:
    df = df.copy()
    df["canon"] = df["canon"].astype("string")

    has_tax = "tax_id" in df.columns
    has_sp = "species_name" in df.columns
    has_comp = "comp_id" in df.columns

    if level == "canon":
        df["group_id"] = df["canon"]
        return df

    if level == "tax_id":
        if has_tax:
            tax = df["tax_id"].astype("string")
            if fallback_to_canon:
                df["group_id"] = tax.where(tax.notna(), "NA_TAX|" + df["canon"])
            else:
                df["group_id"] = tax.fillna("NA_TAX")
        else:
            df["group_id"] = ("NA_TAX|" + df["canon"]) if fallback_to_canon else "NA_TAX"
        return df

    if level == "species_name":
        if has_sp:
            sp = df["species_name"].astype("string")
            if fallback_to_canon:
                df["group_id"] = sp.where(sp.notna(), "NA_SP|" + df["canon"])
            else:
                df["group_id"] = sp.fillna("NA_SP")
        else:
            df["group_id"] = ("NA_SP|" + df["canon"]) if fallback_to_canon else "NA_SP"
        return df

    if level == "comp_tax":
        comp = df["comp_id"].astype("string") if has_comp else pd.Series(pd.NA, index=df.index, dtype="string")
        tax = df["tax_id"].astype("string") if has_tax else pd.Series(pd.NA, index=df.index, dtype="string")

        both = comp.notna() & tax.notna()
        only_tax = tax.notna() & comp.isna()
        only_comp = comp.notna() & tax.isna()

        gid = pd.Series(pd.NA, index=df.index, dtype="string")
        gid.loc[both] = tax.loc[both].astype(str) + "|c" + comp.loc[both].astype(str)
        gid.loc[only_tax] = "NA|cNA|" + tax.loc[only_tax].astype(str)
        gid.loc[only_comp] = "NA_TAX|c" + comp.loc[only_comp].astype(str)

        if fallback_to_canon:
            gid = gid.where(gid.notna(), "NA_CT|" + df["canon"])
        else:
            gid = gid.fillna("NA_CT")

        df["group_id"] = gid
        return df

    raise ValueError(f"Unknown GROUP_LEVEL: {level}")


# =========================================================
#  外层 holdout（超严格测试）
# =========================================================
def outer_holdout_split(df: pd.DataFrame,
                        group_col: str,
                        test_ratio: float = 0.15,
                        random_state: int = 42):
    if group_col not in df.columns:
        return df, None

    groups = df[group_col].astype("string").fillna("NA")
    labels = df["label"].values

    gss = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=random_state)
    rest_idx, outer_idx = next(gss.split(df, labels, groups.values))

    df_rest = df.iloc[rest_idx].reset_index(drop=True)
    df_outer = df.iloc[outer_idx].reset_index(drop=True)

    print(
        f"[OuterHoldout] level={group_col}, ratio={test_ratio:.2f} | "
        f"rest={len(df_rest)}, outer_test={len(df_outer)}, "
        f"outer_unique_groups={df_outer[group_col].nunique()}"
    )
    return df_rest, df_outer


# =========================================================
#  可选：长度/GC 匹配（qcut 分箱）
# =========================================================
def length_match_after_dedup(df_all: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
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
        parts.append(pos_g.sample(n=n, random_state=SEED))
        parts.append(neg_g.sample(n=n, random_state=SEED))

    if not parts:
        print("[LenMatch] 警告：长度匹配后无可用 bin，返回原数据。")
        return df_all

    out = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    print(f"[LenMatch] 有效 bin={kept_bins}, 空/单边 bin={dropped_bins}")
    print(f"[LenMatch] 匹配后: pos={(out['label']==1).sum()}, neg={(out['label']==0).sum()}, total={len(out)}")
    return out


def gc_match_after_dedup(df_all: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
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
        parts.append(pos_g.sample(n=n, random_state=SEED))
        parts.append(neg_g.sample(n=n, random_state=SEED))

    if not parts:
        print("[GCMatch] 警告：GC 匹配后无可用 bin，返回原数据。")
        return df_all

    out = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    print(f"[GCMatch] 有效 bin={kept_bins}, 空/单边 bin={dropped_bins}")
    print(f"[GCMatch] 匹配后: pos={(out['label']==1).sum()}, neg={(out['label']==0).sum()}, total={len(out)}")
    return out


# =========================================================
#  可选：偏差诊断（SAFE_MODE 下自动降级）
# =========================================================
def print_bias_diagnostic(df_all: pd.DataFrame, topk: int = 15):
    try:
        df = df_all.copy()
        print("[Diag] seq_len by label:")
        print(df.groupby("label")["seq_len"].describe())

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


def bias_diagnosis_files(df_all: pd.DataFrame, out_dir: str, tag: str):
    """
    SAFE_MODE 下：
      - 只输出 summary CSV
      - 不导入 seaborn
      - 避免 Windows 绘图库链路触发访问冲突
    """
    os.makedirs(out_dir, exist_ok=True)

    summary = {
        "tag": tag,
        "n_total": len(df_all),
        "n_pos": int(df_all["label"].sum()),
        "n_neg": int((1 - df_all["label"]).sum()),
        "pos_rate": float(df_all["label"].mean()) if len(df_all) else np.nan,
        "len_pos_mean": float(df_all.loc[df_all["label"] == 1, "seq_len"].mean()) if (df_all["label"] == 1).any() else np.nan,
        "len_neg_mean": float(df_all.loc[df_all["label"] == 0, "seq_len"].mean()) if (df_all["label"] == 0).any() else np.nan,
        "gc_pos_mean": float(df_all.loc[df_all["label"] == 1, "gc"].mean()) if (df_all["label"] == 1).any() else np.nan,
        "gc_neg_mean": float(df_all.loc[df_all["label"] == 0, "gc"].mean()) if (df_all["label"] == 0).any() else np.nan,
        "n_groups": int(df_all["group_id"].nunique()) if "group_id" in df_all.columns else np.nan,
        "n_canons": int(df_all["canon"].nunique()) if "canon" in df_all.columns else np.nan,
    }
    pd.DataFrame([summary]).to_csv(os.path.join(out_dir, f"bias_summary_{tag}.csv"), index=False)

    if SAFE_MODE:
        print(f"[BiasDiag] SAFE_MODE=ON，跳过 seaborn 诊断绘图，仅输出 summary CSV：{tag}")
        return

    # 非 SAFE_MODE 才尝试 seaborn 绘图
    try:
        import seaborn as sns

        plt.figure()
        sns.histplot(data=df_all, x="seq_len", hue="label", bins=50, stat="count", element="step")
        plt.title(f"Length Distribution ({tag})")
        plt.xlabel("Sequence length")
        plt.ylabel("Count")
        safe_savefig(os.path.join(out_dir, f"bias_length_hist_{tag}.png"))

        plt.figure()
        sns.histplot(data=df_all, x="gc", hue="label", bins=50, stat="count", element="step")
        plt.title(f"GC Distribution ({tag})")
        plt.xlabel("GC content")
        plt.ylabel("Count")
        safe_savefig(os.path.join(out_dir, f"bias_gc_hist_{tag}.png"))
    except Exception as e:
        print(f"[BiasDiagWarn] 诊断绘图失败 | {repr(e)}")

    print(f"[BiasDiag] 已输出 {tag} 偏差诊断到: {out_dir}")


# =========================================================
#  GroupSplit 70/15/15 + leak check
# =========================================================
def group_split_70_15_15(df_all: pd.DataFrame, random_state=SEED):
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
        n_groups = frame["group_id"].nunique() if len(frame) else 0
        print(f"[GroupSplit] {name}: 样本={n_samples}, 组数={n_groups}, pos_rate={pos_rate:.3f}")

    _log_split("train", df_train)
    _log_split("val", df_val)
    _log_split("test", df_test)

    g_tr = set(df_train["group_id"])
    g_va = set(df_val["group_id"])
    g_te = set(df_test["group_id"])
    print(
        f"[GroupSplit] 交叉 group_id："
        f"train∩val={len(g_tr & g_va)}, "
        f"train∩test={len(g_tr & g_te)}, "
        f"val∩test={len(g_va & g_te)}"
    )
    assert len(g_tr & g_va) == 0 and len(g_tr & g_te) == 0 and len(g_va & g_te) == 0, \
        "GroupSplit 出现跨集共享 group_id！"

    c_tr = set(df_train["canon"])
    c_va = set(df_val["canon"])
    c_te = set(df_test["canon"])
    print(
        f"[LeakCheck] canonical RC 交叉："
        f"train∩val={len(c_tr & c_va)}, "
        f"train∩test={len(c_tr & c_te)}, "
        f"val∩test={len(c_va & c_te)}"
    )
    assert len(c_tr & c_va) == 0 and len(c_tr & c_te) == 0 and len(c_va & c_te) == 0, \
        "存在跨集完全相同的 canonical 序列！"

    return df_train, df_val, df_test


# =========================================================
#  训练集困难负样本采样（成对匹配式）
# =========================================================
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
        tmp[key_col] = tmp[key_col].astype("string").fillna("NA")

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

        if not parts:
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


# =========================================================
#  Dataset
# =========================================================
class DeepSEATrainDataset(Dataset):
    def __init__(self, sequences, labels, seq_len=SEQ_LEN):
        self.seqs = list(sequences)
        self.labels = np.array(labels).astype(int)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        raw = self.seqs[idx]
        y = int(self.labels[idx])
        s = random_crop_or_pad(raw, self.seq_len)
        x = one_hot_encode(s, self.seq_len)
        return x, y


class DeepSEAValDataset(Dataset):
    def __init__(self, sequences, labels, seq_len=SEQ_LEN):
        self.seqs = list(sequences)
        self.labels = np.array(labels).astype(int)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        raw = self.seqs[idx]
        y = int(self.labels[idx])
        s = pad_or_crop_center(raw, self.seq_len)
        x = one_hot_encode(s, self.seq_len)
        return x, y


def collate_fn(batch):
    xs, ys = zip(*batch)
    xs = torch.stack(xs, dim=0)
    ys = torch.tensor(ys, dtype=torch.long)
    return xs, ys


# =========================================================
#  DeepSEA 模型（小窗口池化修正）
# =========================================================
def choose_pool_k(seq_len: int) -> int:
    if seq_len < 64:
        return 1
    if seq_len < 128:
        return 2
    return 4


class DeepSEA(nn.Module):
    def __init__(self, seq_len=SEQ_LEN, num_classes=2):
        super(DeepSEA, self).__init__()
        self.seq_len = seq_len
        pool_k = choose_pool_k(seq_len)

        self.conv1 = nn.Conv1d(4, 320, kernel_size=8)
        self.bn1 = nn.BatchNorm1d(320)
        self.pool1 = nn.MaxPool1d(kernel_size=pool_k, stride=pool_k) if pool_k > 1 else nn.Identity()

        self.conv2 = nn.Conv1d(320, 480, kernel_size=8)
        self.bn2 = nn.BatchNorm1d(480)
        self.pool2 = nn.MaxPool1d(kernel_size=pool_k, stride=pool_k) if pool_k > 1 else nn.Identity()

        self.conv3 = nn.Conv1d(480, 960, kernel_size=8)
        self.bn3 = nn.BatchNorm1d(960)

        self.dropout_conv = nn.Dropout(0.2)
        self.dropout_fc = nn.Dropout(0.5)

        with torch.no_grad():
            dummy = torch.zeros(1, 4, seq_len)
            x = self.pool1(F.relu(self.bn1(self.conv1(dummy))))
            x = self.pool2(F.relu(self.bn2(self.conv2(x))))
            x = F.relu(self.bn3(self.conv3(x)))
            fc_input_dim = x.view(1, -1).size(1)

        self.fc1 = nn.Linear(fc_input_dim, 925)
        self.fc2 = nn.Linear(925, num_classes)

        print(f"[Model] SEQ_LEN={seq_len}, pool_k={pool_k}, fc_input_dim={fc_input_dim}")

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout_conv(x)
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout_conv(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout_fc(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        return self.fc2(x)


# =========================================================
#  评估工具
# =========================================================
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


@torch.inference_mode()
def evaluate_loader(model, data_loader, criterion, use_amp):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_labels, all_preds, all_probs = [], [], []

    for xs, ys in data_loader:
        xs = xs.to(device, non_blocking=True)
        ys = ys.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(xs)
            loss = criterion(logits, ys)

        total_loss += loss.item()
        n_batches += 1

        probs = F.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
        preds = torch.argmax(logits, dim=-1).detach().cpu().numpy()
        ys_np = ys.detach().cpu().numpy()

        all_labels.extend(ys_np.tolist())
        all_preds.extend(preds.tolist())
        all_probs.extend(probs.tolist())

    avg_loss = total_loss / max(1, n_batches)
    return avg_loss, np.array(all_labels), np.array(all_preds), np.array(all_probs)


# =========================================================
#  学习率计划
# =========================================================
def get_lr(step, total_steps, warmup_steps, lr_peak):
    if step < warmup_steps:
        return lr_peak * float(step) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return lr_peak * max(0.0, 1.0 - progress)


# =========================================================
#  主流程
# =========================================================
def main():
    output_dir, checkpoint_dir = make_output_dirs()

    # -------------------------
    # 数据准备（最稳 + 更严格评估协议）
    # -------------------------
    df_all = load_combined_data(
        POS_FILE, NEG_FILE,
        max_neg_keep=MAX_NEG_KEEP,
        neg_pos_ratio_keep=NEG_POS_RATIO_KEEP
    )
    gc.collect()

    if ENABLE_BIAS_DIAG:
        print_bias_diagnostic(df_all, topk=DIAG_TOPK)

    # 统一为严格单层级 group_id
    df_all = rebuild_group_id_strict(
        df_all,
        level=GROUP_LEVEL,
        fallback_to_canon=STRICT_GROUP_FALLBACK_TO_CANON
    )
    print(
        f"[StrictGroup] GROUP_LEVEL={GROUP_LEVEL}, "
        f"fallback_to_canon={STRICT_GROUP_FALLBACK_TO_CANON}, "
        f"unique group_id={df_all['group_id'].nunique()}"
    )

    # 外层 holdout（超严格 outer_test）
    if ENABLE_OUTER_TEST_HOLDOUT:
        if OUTER_GROUP_LEVEL not in df_all.columns:
            print(f"[OuterHoldout] {OUTER_GROUP_LEVEL} 不存在，跳过外层 holdout。")
            df_rest = df_all
            df_outer_test = None
        else:
            df_rest, df_outer_test = outer_holdout_split(
                df_all,
                group_col=OUTER_GROUP_LEVEL,
                test_ratio=OUTER_TEST_RATIO,
                random_state=SEED
            )
    else:
        df_rest = df_all
        df_outer_test = None

    # 内层 GroupSplit 70/15/15
    df_train, df_val, df_test = group_split_70_15_15(df_rest, random_state=SEED)

    # 长度/GC 匹配：只对训练集生效（更保守、更真实）
    if ENABLE_LENGTH_MATCH:
        df_train = length_match_after_dedup(df_train, n_bins=LENGTH_MATCH_BINS)
    else:
        print("[LenMatch] 已关闭长度匹配（train-only）。")

    if ENABLE_GC_MATCH:
        df_train = gc_match_after_dedup(df_train, n_bins=GC_MATCH_BINS)
    else:
        print("[GCMatch] 已关闭 GC 匹配（train-only）。")

    # 可选：诊断落盘（SAFE_MODE 内部自动降级）
    if ENABLE_BIAS_DIAG:
        bias_dir = os.path.join(output_dir, "bias_diagnosis")
        try:
            bias_diagnosis_files(df_rest, bias_dir, tag="rest_before_inner_split")
            bias_diagnosis_files(df_train, bias_dir, tag="train_after_train_only_match")
            bias_diagnosis_files(df_val, bias_dir, tag="val_inner")
            bias_diagnosis_files(df_test, bias_dir, tag="test_inner")
            if df_outer_test is not None:
                bias_diagnosis_files(df_outer_test, bias_dir, tag="outer_test_strict")
        except Exception as e:
            print(f"[BiasDiagWarn] 诊断文件输出失败 | {repr(e)}")

    # 释放全量/中间集合
    del df_all, df_rest
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # -------------------------
    # 可选：困难负样本训练集构造
    # -------------------------
    if ENABLE_HARD_NEG_TRAIN:
        df_train_used = build_hard_negative_trainset(
            df_train,
            random_state=HARD_NEG_RANDOM_STATE,
            min_pos=HARD_NEG_MIN_POS,
            min_pos_ratio=HARD_NEG_MIN_POS_RATIO
        )
    else:
        df_train_used = df_train
        print("[HardNeg] 已关闭训练集困难负样本采样。")

    # -------------------------
    # 取序列与标签
    # -------------------------
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

    # outer_test 序列（若存在）
    if df_outer_test is not None and len(df_outer_test) > 0:
        outer_sequences = df_outer_test["seq"].tolist()
        y_outer = df_outer_test["label"].values
        print(
            f"[OuterTest] outer_test={len(outer_sequences)} ({np.mean(y_outer):.3f} pos rate) "
            f"| level={OUTER_GROUP_LEVEL}, ratio={OUTER_TEST_RATIO}"
        )
    else:
        outer_sequences, y_outer = None, None

    # -------------------------
    # DataLoader
    # -------------------------
    train_loader = DataLoader(
        DeepSEATrainDataset(train_sequences, y_train, seq_len=SEQ_LEN),
        batch_size=BATCH_SIZE_TRAIN,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=PIN_MEMORY,
        num_workers=0,
    )

    val_loader = DataLoader(
        DeepSEAValDataset(val_sequences, y_val, seq_len=SEQ_LEN),
        batch_size=BATCH_SIZE_EVAL,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=PIN_MEMORY,
        num_workers=0,
    )

    test_loader = DataLoader(
        DeepSEAValDataset(test_sequences, y_test, seq_len=SEQ_LEN),
        batch_size=BATCH_SIZE_EVAL,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=PIN_MEMORY,
        num_workers=0,
    )

    outer_loader = None
    if outer_sequences is not None:
        outer_loader = DataLoader(
            DeepSEAValDataset(outer_sequences, y_outer, seq_len=SEQ_LEN),
            batch_size=BATCH_SIZE_EVAL,
            shuffle=False,
            collate_fn=collate_fn,
            pin_memory=PIN_MEMORY,
            num_workers=0,
        )

    # -------------------------
    # 模型与优化器
    # -------------------------
    model = DeepSEA(seq_len=SEQ_LEN, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    num_update_steps_per_epoch = len(train_loader)
    num_train_steps = EPOCHS * num_update_steps_per_epoch
    num_warmup_steps = max(1, int(WARMUP_RATIO * num_train_steps))

    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    # -------------------------
    # 训练监控
    # -------------------------
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

    global_step = 0

    # -------------------------
    # 训练循环
    # -------------------------
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        seen_batches = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{EPOCHS}]", leave=False)

        for xs, ys in progress_bar:
            xs = xs.to(device, non_blocking=True)
            ys = ys.to(device, non_blocking=True)

            lr_now = get_lr(global_step, num_train_steps, num_warmup_steps, LEARNING_RATE)
            for g in optimizer.param_groups:
                g["lr"] = lr_now

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=USE_AMP):
                logits = model(xs)
                loss = criterion(logits, ys)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            seen_batches += 1
            global_step += 1

            progress_bar.set_postfix(loss=(total_loss / max(1, seen_batches)), lr=lr_now)

        avg_train_loss = total_loss / max(1, seen_batches)
        train_losses.append(avg_train_loss)

        val_loss, val_labels, val_preds, val_probs = evaluate_loader(
            model, val_loader, criterion, USE_AMP
        )
        val_losses.append(val_loss)
        val_metrics = compute_classification_metrics(val_labels, val_preds, val_probs)

        lr_epoch = get_lr(global_step, num_train_steps, num_warmup_steps, LEARNING_RATE)
        epoch_lrs.append(lr_epoch)

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
            f"LR: {lr_epoch:.2e}"
        )

        with open(metrics_file_path, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch + 1}\t{avg_train_loss:.4f}\t{val_loss:.4f}\t"
                f"{val_metrics['accuracy']:.4f}\t{val_metrics['precision']:.4f}\t"
                f"{val_metrics['recall']:.4f}\t{val_metrics['f1']:.4f}\t{val_metrics['mcc']:.4f}\t"
                f"{val_metrics['auroc']:.4f}\t{val_metrics['auprc']:.4f}\t{val_metrics['ap']:.4f}\t"
                f"{lr_epoch:.6e}\n"
            )

        early_stopping(val_loss, model, epoch)
        if early_stopping.early_stop:
            print(f"早停触发于第 {epoch + 1} 轮，最佳模型来自第 {early_stopping.best_epoch + 1} 轮")
            break

        if SAFE_MODE and (epoch + 1) % 5 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # -------------------------
    # 加载最佳模型并测试
    # -------------------------
    best_epoch_idx = early_stopping.best_epoch
    best_ckpt_path = resolve_checkpoint_path(os.path.join(checkpoint_dir, f"best_model_epoch{best_epoch_idx + 1}.pth"))

    if os.path.exists(best_ckpt_path):
        model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
        print(f"[Load Best] 加载最佳验证轮权重：{best_ckpt_path}")
    else:
        print("[Load Best] 未找到最佳权重文件，使用当前模型参数进行测试。")

    # inner test
    test_loss, test_labels, test_preds, test_probs = evaluate_loader(
        model, test_loader, criterion, USE_AMP
    )
    test_metrics = compute_classification_metrics(test_labels, test_preds, test_probs)

    # outer test（更严格）
    outer_loss = None
    outer_metrics = None
    outer_labels = outer_preds = outer_probs = None
    if outer_loader is not None:
        outer_loss, outer_labels, outer_preds, outer_probs = evaluate_loader(
            model, outer_loader, criterion, USE_AMP
        )
        outer_metrics = compute_classification_metrics(outer_labels, outer_preds, outer_probs)

        print(
            f"[OuterTest/Strict] Loss={outer_loss:.4f}, "
            f"Acc={outer_metrics['accuracy']:.4f}, "
            f"F1={outer_metrics['f1']:.4f}, "
            f"AUROC={outer_metrics['auroc']:.4f}, "
            f"AUPRC={outer_metrics['auprc']:.4f}"
        )

    # -------------------------
    # 可视化与最终输出（SAFE_MODE 下避免 seaborn）
    # -------------------------
    if len(test_labels) > 0:
        fpr, tpr, _ = roc_curve(test_labels, test_probs)
        auroc_final = auc(fpr, tpr)
        precision_vals, recall_vals, _ = precision_recall_curve(test_labels, test_probs)
        ap_final = average_precision_score(test_labels, test_probs)
        auprc_final = auc(recall_vals, precision_vals)
    
        save_curve_points(output_dir, fpr=fpr, tpr=tpr, recall=recall_vals, precision=precision_vals)
    else:
        fpr, tpr = np.array([0, 1]), np.array([0, 1])
        precision_vals, recall_vals = np.array([1]), np.array([0])
        auroc_final = float("nan")
        ap_final = float("nan")
        auprc_final = float("nan")

    cm = confusion_matrix(test_labels, test_preds)

    # ---- 混淆矩阵：SAFE_MODE 用纯 matplotlib ----
    try:
        plt.figure()
        if SAFE_MODE:
            plt.imshow(cm, interpolation="nearest")
            plt.title("Confusion Matrix (Inner Test)")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.xticks([0, 1], [0, 1])
            plt.yticks([0, 1], [0, 1])
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, str(cm[i, j]), ha="center", va="center")
            safe_savefig(os.path.join(output_dir, "confusion_matrix_test.png"))
        else:
            import seaborn as sns
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.title("Confusion Matrix (Inner Test)")
            safe_savefig(os.path.join(output_dir, "confusion_matrix_test.png"))
    except Exception as e:
        print(f"[PlotWarn] confusion matrix 绘图失败 | {repr(e)}")

    # ROC / PR 使用 matplotlib，通常更稳
    try:
        plt.figure()
        plt.plot(fpr, tpr, lw=2, label=f"ROC (AUROC = {auroc_final:.2f})")
        plt.plot([0, 1], [0, 1], lw=2, linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve (Inner Test)")
        plt.legend(loc="lower right")
        plt.grid(True)
        safe_savefig(os.path.join(output_dir, "roc_curve_test.png"))
    except Exception as e:
        print(f"[PlotWarn] ROC 绘图失败 | {repr(e)}")

    try:
        plt.figure()
        plt.step(recall_vals, precision_vals, where="post", lw=2,
                 label=f"PR (AP = {ap_final:.2f}, AUPRC = {auprc_final:.2f})")
        plt.fill_between(recall_vals, precision_vals, step="post", alpha=0.2)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("PR Curve (Inner Test)")
        plt.legend(loc="upper right")
        plt.grid(True)
        safe_savefig(os.path.join(output_dir, "pr_curve_test.png"))
    except Exception as e:
        print(f"[PlotWarn] PR 绘图失败 | {repr(e)}")

    # outer test 可选图（SAFE_MODE 下仍尽量走 matplotlib）
    outer_auroc = outer_auprc = outer_ap = float("nan")
    if outer_metrics is not None and outer_labels is not None and len(outer_labels) > 0:
        try:
            fpr_o, tpr_o, _ = roc_curve(outer_labels, outer_probs)
            outer_auroc = auc(fpr_o, tpr_o)
            prec_o, rec_o, _ = precision_recall_curve(outer_labels, outer_probs)
            outer_ap = average_precision_score(outer_labels, outer_probs)
            outer_auprc = auc(rec_o, prec_o)

            cm_o = confusion_matrix(outer_labels, outer_preds)

            plt.figure()
            if SAFE_MODE:
                plt.imshow(cm_o, interpolation="nearest")
                plt.title("Confusion Matrix (Outer Strict Test)")
                plt.xlabel("Predicted")
                plt.ylabel("True")
                plt.xticks([0, 1], [0, 1])
                plt.yticks([0, 1], [0, 1])
                for i in range(cm_o.shape[0]):
                    for j in range(cm_o.shape[1]):
                        plt.text(j, i, str(cm_o[i, j]), ha="center", va="center")
                safe_savefig(os.path.join(output_dir, "confusion_matrix_outer.png"))
            else:
                import seaborn as sns
                sns.heatmap(cm_o, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
                plt.xlabel("Predicted")
                plt.ylabel("True")
                plt.title("Confusion Matrix (Outer Strict Test)")
                safe_savefig(os.path.join(output_dir, "confusion_matrix_outer.png"))

            plt.figure()
            plt.plot(fpr_o, tpr_o, lw=2, label=f"ROC (AUROC = {outer_auroc:.2f})")
            plt.plot([0, 1], [0, 1], lw=2, linestyle="--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve (Outer Strict Test)")
            plt.legend(loc="lower right")
            plt.grid(True)
            safe_savefig(os.path.join(output_dir, "roc_curve_outer.png"))

            plt.figure()
            plt.step(rec_o, prec_o, where="post", lw=2,
                     label=f"PR (AP = {outer_ap:.2f}, AUPRC = {outer_auprc:.2f})")
            plt.fill_between(rec_o, prec_o, step="post", alpha=0.2)
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("PR Curve (Outer Strict Test)")
            plt.legend(loc="upper right")
            plt.grid(True)
            safe_savefig(os.path.join(output_dir, "pr_curve_outer.png"))
        except Exception as e:
            print(f"[OuterPlotWarn] outer 可视化失败 | {repr(e)}")

    # 训练/验证损失与指标落盘
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

    # 指标曲线（Validation）
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
        try:
            plt.figure()
            plt.plot(range(1, len(values) + 1), values, label=metric_name)
            plt.xlabel("Epoch")
            plt.ylabel(metric_name)
            plt.title(f"{metric_name} over Epochs (Validation)")
            plt.grid(True)
            plt.legend()
            filename = f"{metric_name.lower().replace('-', '_')}_curve_val.png"
            safe_savefig(os.path.join(output_dir, filename))
        except Exception as e:
            print(f"[PlotWarn] {metric_name} 曲线绘图失败 | {repr(e)}")

    if len(epoch_lrs) > 0:
        try:
            plt.figure()
            plt.plot(range(1, len(epoch_lrs) + 1), epoch_lrs, label="LR")
            plt.xlabel("Epoch")
            plt.ylabel("Learning Rate")
            plt.title("Learning Rate over Epochs")
            plt.yscale("log")
            plt.legend()
            plt.grid(True)
            safe_savefig(os.path.join(output_dir, "lr_curve.png"))
        except Exception as e:
            print(f"[PlotWarn] LR 曲线绘图失败 | {repr(e)}")

    try:
        plt.figure()
        plt.plot(loss_data["Epoch"], loss_data["Train_Loss"], label="Train Loss")
        plt.plot(loss_data["Epoch"], loss_data["Validation_Loss"], label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training vs Validation Loss")
        plt.legend()
        plt.grid(True)
        safe_savefig(os.path.join(output_dir, "loss_curve.png"))
    except Exception as e:
        print(f"[PlotWarn] loss 曲线绘图失败 | {repr(e)}")

    avg_train_loss_last = train_losses[-1] if len(train_losses) > 0 else float("nan")
    avg_val_loss_last = val_losses[-1] if len(val_losses) > 0 else float("inf")

    # -------------------------
    # 最终报告文本
    # -------------------------
    metrics_text = f"""Final Metrics Report:

[Validation-guided training]
Train Loss (last epoch): {avg_train_loss_last:.4f}
Validation Loss (last epoch): {avg_val_loss_last:.4f}
Early Stopping Triggered: {'Yes' if early_stopping.early_stop else 'No'}
Best Epoch (by val loss): {early_stopping.best_epoch + 1}

[Inner Test (GroupSplit 70/15/15 on rest)]
Test Loss: {test_loss:.4f}
Accuracy: {test_metrics['accuracy']:.4f}
Precision: {test_metrics['precision']:.4f}
Recall: {test_metrics['recall']:.4f}
F1-score: {test_metrics['f1']:.4f}
MCC: {test_metrics['mcc']:.4f}
AUROC: {auroc_final:.4f}
AUPRC: {auprc_final:.4f}
Average Precision (AP): {ap_final:.4f}

[Outer Strict Test]
Enabled: {ENABLE_OUTER_TEST_HOLDOUT}
Outer Group Level: {OUTER_GROUP_LEVEL}
Outer Ratio: {OUTER_TEST_RATIO}
Outer Test Loss: {outer_loss if outer_loss is not None else 'NA'}
Outer Accuracy: {outer_metrics['accuracy'] if outer_metrics is not None else 'NA'}
Outer Precision: {outer_metrics['precision'] if outer_metrics is not None else 'NA'}
Outer Recall: {outer_metrics['recall'] if outer_metrics is not None else 'NA'}
Outer F1-score: {outer_metrics['f1'] if outer_metrics is not None else 'NA'}
Outer MCC: {outer_metrics['mcc'] if outer_metrics is not None else 'NA'}
Outer AUROC: {outer_auroc}
Outer AUPRC: {outer_auprc}
Outer AP: {outer_ap}

[Data Strictness Switches]
SEQ_LEN: {SEQ_LEN}
ENABLE_CANON_CONFLICT_DROP: {ENABLE_CANON_CONFLICT_DROP}

GROUP_LEVEL: {GROUP_LEVEL}
STRICT_GROUP_FALLBACK_TO_CANON: {STRICT_GROUP_FALLBACK_TO_CANON}

ENABLE_LENGTH_MATCH (train-only): {ENABLE_LENGTH_MATCH}
LENGTH_MATCH_BINS: {LENGTH_MATCH_BINS}
ENABLE_GC_MATCH (train-only): {ENABLE_GC_MATCH}
GC_MATCH_BINS: {GC_MATCH_BINS}

ENABLE_HARD_NEG_TRAIN: {ENABLE_HARD_NEG_TRAIN}
SAFE_MODE: {'ON' if SAFE_MODE else 'OFF'}

MAX_NEG_KEEP: {MAX_NEG_KEEP}
NEG_POS_RATIO_KEEP: {NEG_POS_RATIO_KEEP}
PART_ROWS: {PART_ROWS}
MAX_SEQ_LEN_HARD: {MAX_SEQ_LEN_HARD}
"""
    with open(os.path.join(output_dir, "final_metrics_report.txt"), "w", encoding="utf-8") as f:
        f.write(metrics_text)

    # -------------------------
    # 最终报告 CSV
    # -------------------------
    with open(os.path.join(output_dir, "final_metrics_report.csv"), mode="w", newline="", encoding="utf-8") as file:
        writer = pycsv.writer(file)
        writer.writerow(["Section", "Metric", "Value"])

        writer.writerow(["Train/Val", "Train Loss (last)", avg_train_loss_last])
        writer.writerow(["Train/Val", "Validation Loss (last)", avg_val_loss_last])
        writer.writerow(["Train/Val", "Early Stopping", "Yes" if early_stopping.early_stop else "No"])
        writer.writerow(["Train/Val", "Best Epoch (val)", early_stopping.best_epoch + 1])

        writer.writerow(["Inner Test", "Test Loss", test_loss])
        writer.writerow(["Inner Test", "Accuracy", test_metrics["accuracy"]])
        writer.writerow(["Inner Test", "Precision", test_metrics["precision"]])
        writer.writerow(["Inner Test", "Recall", test_metrics["recall"]])
        writer.writerow(["Inner Test", "F1-score", test_metrics["f1"]])
        writer.writerow(["Inner Test", "MCC", test_metrics["mcc"]])
        writer.writerow(["Inner Test", "AUROC", auroc_final])
        writer.writerow(["Inner Test", "AUPRC", auprc_final])
        writer.writerow(["Inner Test", "AP", ap_final])

        writer.writerow(["Outer Strict Test", "Enabled", ENABLE_OUTER_TEST_HOLDOUT])
        writer.writerow(["Outer Strict Test", "Group Level", OUTER_GROUP_LEVEL])
        writer.writerow(["Outer Strict Test", "Ratio", OUTER_TEST_RATIO])
        writer.writerow(["Outer Strict Test", "Loss", outer_loss if outer_loss is not None else "NA"])
        writer.writerow(["Outer Strict Test", "Accuracy", outer_metrics["accuracy"] if outer_metrics is not None else "NA"])
        writer.writerow(["Outer Strict Test", "Precision", outer_metrics["precision"] if outer_metrics is not None else "NA"])
        writer.writerow(["Outer Strict Test", "Recall", outer_metrics["recall"] if outer_metrics is not None else "NA"])
        writer.writerow(["Outer Strict Test", "F1-score", outer_metrics["f1"] if outer_metrics is not None else "NA"])
        writer.writerow(["Outer Strict Test", "MCC", outer_metrics["mcc"] if outer_metrics is not None else "NA"])
        writer.writerow(["Outer Strict Test", "AUROC", outer_auroc])
        writer.writerow(["Outer Strict Test", "AUPRC", outer_auprc])
        writer.writerow(["Outer Strict Test", "AP", outer_ap])

        writer.writerow(["Config", "SEQ_LEN", SEQ_LEN])
        writer.writerow(["Config", "GROUP_LEVEL", GROUP_LEVEL])
        writer.writerow(["Config", "STRICT_GROUP_FALLBACK_TO_CANON", STRICT_GROUP_FALLBACK_TO_CANON])
        writer.writerow(["Config", "ENABLE_LENGTH_MATCH (train-only)", ENABLE_LENGTH_MATCH])
        writer.writerow(["Config", "LENGTH_MATCH_BINS", LENGTH_MATCH_BINS])
        writer.writerow(["Config", "ENABLE_GC_MATCH (train-only)", ENABLE_GC_MATCH])
        writer.writerow(["Config", "GC_MATCH_BINS", GC_MATCH_BINS])
        writer.writerow(["Config", "ENABLE_HARD_NEG_TRAIN", ENABLE_HARD_NEG_TRAIN])
        writer.writerow(["Config", "SAFE_MODE", "ON" if SAFE_MODE else "OFF"])
        writer.writerow(["Config", "MAX_NEG_KEEP", MAX_NEG_KEEP])
        writer.writerow(["Config", "NEG_POS_RATIO_KEEP", NEG_POS_RATIO_KEEP])
        writer.writerow(["Config", "PART_ROWS", PART_ROWS])
        writer.writerow(["Config", "MAX_SEQ_LEN_HARD", MAX_SEQ_LEN_HARD])

    print(f"Training completed! Results saved to: {output_dir}")
    print(f"Best model saved in: {checkpoint_dir}")
    print("注意：训练期指标基于 Validation；最终报告含 Inner Test 与 Outer Strict Test（若启用）。")


# =========================================================
#  Windows 必备入口保护
# =========================================================
if __name__ == "__main__":
    try:
        import torch.multiprocessing as mp
        mp.set_start_method("spawn", force=True)
    except Exception:
        pass

    main()
