# -*- coding: utf-8 -*-
import os
import sys
from datetime import datetime
import random
import warnings

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
import seaborn as sns
import csv

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# ========= 鍙鐜版€?=========
SEED = apply_seed_override(42)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# 濡傛灉鏇寸湅閲嶉€熷害鑰屼笉鏄弗鏍煎鐜帮紝灏辫涓?False
DETERMINISTIC = False
if DETERMINISTIC:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
else:
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

# ========= 璁惧 =========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()
print(f"浣跨敤璁惧: {device} (AMP={'On' if USE_AMP else 'Off'})")

# ========= 閰嶇疆 =========
POS_FILE = r"/Dataset/Patho_plantlen2000.csv"
NEG_FILE = r"/Data_processing/Nonpatho_all.csv"

def resolve_data_path(path_value: str) -> str:
    """
    Resolve dataset paths robustly on Windows/Linux:
    - keep valid absolute paths as-is
    - map "/Dataset/xxx" style paths to project root
    """
    raw = str(path_value).strip().strip('"').strip("'")
    if not raw:
        return raw

    if os.path.exists(raw):
        return raw

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

    normalized = raw.replace("/", os.sep).replace("\\", os.sep)
    if raw.startswith(("/", "\\")):
        normalized = normalized.lstrip("/\\")

    return os.path.join(project_root, normalized)

POS_FILE = resolve_data_path(POS_FILE)
NEG_FILE = resolve_data_path(NEG_FILE)
SEQ_LEN = 150  # 杈撳叆闀垮害锛堟寜闇€鏀规垚 150/300/500/1000 绛夛級

EPOCHS = 80
EPOCHS, EVAL_ONLY = apply_runtime_overrides(EPOCHS)
BATCH_SIZE_TRAIN = 1024
BATCH_SIZE_EVAL = 1024
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
WARMUP_RATIO = 0.05   # 绾挎€?warmup + 绾挎€ц“鍑?

NUM_WORKERS = 4

# ========= 闀垮害鍖归厤寮€鍏?=========
ENABLE_LENGTH_MATCH = False
LENGTH_MATCH_BINS = 10

# ========= GC 鍖归厤寮€鍏筹紙寤鸿鎬€鐤戣櫄楂樻椂鍏堝紑锛?=========
ENABLE_GC_MATCH = True
GC_MATCH_BINS = 10

# ========= 鍋忓樊璇婃柇 =========
ENABLE_BIAS_DIAGNOSTIC = True
DIAG_SHOW_TOPK = 15

# ========= 鍥伴毦璐熸牱鏈噰鏍凤紙浠呰缁冮泦锛?=========
ENABLE_HARD_NEG_TRAIN = True
HARD_NEG_MIN_POS = 200
HARD_NEG_MIN_POS_RATIO = 0.05
HARD_NEG_RANDOM_STATE = SEED

# ========= DNA 宸ュ叿鍑芥暟 =========
VALID_DNA_CHARS = set("ACGTN")
_RC_TABLE = str.maketrans("ACGTN", "TGCAN")


def clean_sequence(seq: str) -> str:
    """Uppercase sequence and map non-ACGTN bases to N."""
    s = str(seq).strip().upper()
    if not s:
        return ""
    return "".join(ch if ch in VALID_DNA_CHARS else "N" for ch in s)


def rev_comp(seq: str) -> str:
    """Return reverse complement of a DNA sequence."""
    s = str(seq)
    return s.translate(_RC_TABLE)[::-1]


def canon_rc(seq: str) -> str:
    """Choose lexicographically smaller one between sequence and reverse-complement."""
    s = clean_sequence(seq)
    if not s:
        return ""
    rc = rev_comp(s)
    return s if s <= rc else rc


def pad_or_crop_center(seq: str, L: int) -> str:
    """Center crop or pad sequence to fixed length L (for eval/test)."""
    s = clean_sequence(seq)
    if len(s) == L:
        return s
    if len(s) < L:
        return s + "N" * (L - len(s))
    start = (len(s) - L) // 2
    return s[start:start + L]


def random_crop_or_pad(seq: str, L: int) -> str:
    """Random crop or pad sequence to fixed length L (for training)."""
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
    g = s.count("G")
    c = s.count("C")
    return (g + c) / L


# ========= One-hot 缂栫爜锛堝悜閲忓寲鐗堟湰锛?=========
NUC_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}

_LUT = np.full(256, 4, dtype=np.int64)
_LUT[ord("A")] = 0
_LUT[ord("C")] = 1
_LUT[ord("G")] = 2
_LUT[ord("T")] = 3


def one_hot_encode(seq: str, L: int = SEQ_LEN) -> torch.Tensor:
    """
    灏嗛暱搴︿负 L 鐨?DNA 搴忓垪缂栫爜涓?[4, L] 鐨?one-hot銆?
    N/鍏朵粬纰卞熀涓哄叏 0銆?
    """
    assert len(seq) == L
    b = seq.encode("ascii")
    arr = np.frombuffer(b, dtype=np.uint8)
    idxs = _LUT[arr]

    x = np.zeros((4, L), dtype=np.float32)
    pos = np.arange(L, dtype=np.int64)
    mask = idxs < 4
    x[idxs[mask], pos[mask]] = 1.0
    return torch.from_numpy(x)


# ========= 鏃╁仠 =========
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
                print(f"楠岃瘉鎹熷け鏀瑰杽: {old_best:.6f} -> {self.best_loss:.6f}锛屼繚瀛樻ā鍨?..")
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


# ========= 鏁版嵁鍔犺浇 + canonical RC 鍘婚噸 + GroupSplit =========
def _find_first_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def length_match_after_dedup(df_all: pd.DataFrame,
                             n_bins: int = 10,
                             random_state: int = SEED) -> pd.DataFrame:
    """
    闀垮害鍖归厤锛坈anonical 鍘婚噸鍚庯級锛?
    鍒嗕綅鏁板垎绠憋紝姣忎釜 bin 鍐?pos/neg 鍙?min 閲囨牱銆?
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
        print("[LenMatch] Warning: no usable bins after matching; return original deduplicated data.")
        return df_all

    df_matched = pd.concat(parts, ignore_index=True)
    df_matched = df_matched.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    n_pos = int((df_matched["label"] == 1).sum())
    n_neg = int((df_matched["label"] == 0).sum())

    print(f"[LenMatch] Target bins={n_bins} (qcut duplicates='drop' may reduce actual bins).")
    print(f"[LenMatch] Valid bins={kept_bins}, dropped bins={dropped_bins}.")
    print(f"[LenMatch] 鍖归厤鍚庢牱鏈暟: pos={n_pos}, neg={n_neg}, total={len(df_matched)}")

    return df_matched


def gc_match_after_dedup(df_all: pd.DataFrame,
                         n_bins: int = 10,
                         random_state: int = SEED) -> pd.DataFrame:
    """
    GC 鍖归厤锛坈anonical 鍘婚噸鍚庯級锛?
    鍒嗕綅鏁板垎绠憋紝姣忎釜 bin 鍐?pos/neg 鍙?min 閲囨牱銆?
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
        print("[GCMatch] Warning: no usable bins after matching; return original deduplicated data.")
        return df_all

    df_matched = pd.concat(parts, ignore_index=True)
    df_matched = df_matched.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    n_pos = int((df_matched["label"] == 1).sum())
    n_neg = int((df_matched["label"] == 0).sum())

    print(f"[GCMatch] Target bins={n_bins} (qcut duplicates='drop' may reduce actual bins).")
    print(f"[GCMatch] Valid bins={kept_bins}, dropped bins={dropped_bins}.")
    print(f"[GCMatch] 鍖归厤鍚庢牱鏈暟: pos={n_pos}, neg={n_neg}, total={len(df_matched)}")

    return df_matched


def _print_bias_diagnostic(df_all: pd.DataFrame, topk=15):
    """
    蹇€熻瘖鏂細
    - label 绾у埆 GC 鍒嗗竷
    - 鑻ュ瓨鍦?tax_id锛岃緭鍑烘爣绛剧函搴︽瀬绔?tax_id
    """
    try:
        df_tmp = df_all.copy()
        if "gc" not in df_tmp.columns:
            df_tmp["gc"] = df_tmp["seq"].map(gc_content)

        print("[Diag] GC content by label:")
        print(df_tmp.groupby("label")["gc"].describe())

        if "tax_id" in df_tmp.columns:
            purity = df_tmp.groupby("tax_id")["label"].mean().sort_values()
            print(f"[Diag] tax_id label purity (lowest {topk}):")
            print(purity.head(topk))
            print(f"[Diag] tax_id label purity (highest {topk}):")
            print(purity.tail(topk))
    except Exception as e:
        print(f"[Diag] 璇婃柇杈撳嚭澶辫触锛堜笉褰卞搷璁粌锛夛細{e}")


def load_combined_data(file_a, file_b, random_state=SEED) -> pd.DataFrame:
    """
    鍔犺浇 patho / nonpatho锛?
    - 娓呮礂
    - canonical RC
    - 鍏抽敭鏀瑰姩锛氬悓涓€ canonical 鍚屾椂鍚/璐熸爣绛?-> 姝ｈ礋鍙屾柟鍏ㄩ儴鍒犻櫎
    - canonical 鍘婚噸
    - 鏋勫缓 group_id锛堜紭鍏?tax|comp锛?
    """
    df_a = pd.read_csv(file_a, low_memory=False)
    df_b = pd.read_csv(file_b, low_memory=False)

    # 搴忓垪鍒楋細姝ｇ被绗?15 鍒?index=14)锛岃礋绫荤 14 鍒?index=13)
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
    print(f"杞藉叆鍘熷搴忓垪: 姝ｇ被{len(pos)} / 璐熺被{len(neg)}锛屾€昏{len(df_all)}锛涘皢杩涜娓呮礂鍜?canonical RC 鍘婚噸")

    # 娓呮礂 + canonical RC
    df_all["seq"] = df_all["seq_raw"].map(clean_sequence)
    df_all["seq_len"] = df_all["seq"].str.len()
    df_all = df_all[df_all["seq_len"] > 0].copy()
    df_all["canon"] = df_all["seq"].map(canon_rc)

    n_before = len(df_all)

    # 鍐茬獊 canonical锛氭璐熷弻鏂瑰叏閮ㄥ垹闄?
    conflict_groups = df_all.groupby("canon")["label"].nunique()
    conflict_canons = conflict_groups[conflict_groups > 1].index
    n_conflict = int(len(conflict_canons))
    if n_conflict > 0:
        print(f"[Dedup] Found {n_conflict} canonical groups containing both labels; removing both sides.")
        df_all = df_all[~df_all["canon"].isin(conflict_canons)].copy()

    # canonical 鍘婚噸
    df_all = df_all.drop_duplicates(subset="canon", keep="first").reset_index(drop=True)
    n_after = len(df_all)
    print(f"[Dedup] Canonical-RC dedup: {n_before} -> {n_after} sequences")

    df_all["seq_len"] = df_all["seq"].str.len()

    # 鏋勫缓 group_id
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
    print(f"[GroupKey] 鍞竴 group_id 鏁伴噺: {df_all['group_id'].nunique()}")

    df_all["label"] = df_all["label"].astype(int)
    df_all["gc"] = df_all["seq"].map(gc_content)

    return df_all


def group_split_70_15_15_frames(df_all: pd.DataFrame, random_state=SEED):
    """
    杩斿洖 DataFrame 绾у埆 train/val/test锛屼究浜庤缁冮泦鍋氬洶闅捐礋鏍锋湰閲囨牱銆?
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
            f"[GroupSplit] {name}: 鏍锋湰={n_samples}, 缁勬暟={n_groups}, pos_rate={pos_rate:.3f}"
        )

    _log_split("train", df_train)
    _log_split("val", df_val)
    _log_split("test", df_test)

    # 妫€鏌?group_id 浜ゅ弶
    g_tr = set(df_train["group_id"])
    g_va = set(df_val["group_id"])
    g_te = set(df_test["group_id"])
    print(
        f"[GroupSplit] 浜ゅ弶 group_id锛歵rain鈭﹙al={len(g_tr & g_va)}, "
        f"train鈭﹖est={len(g_tr & g_te)}, val鈭﹖est={len(g_va & g_te)}"
    )
    assert len(g_tr & g_va) == 0 and len(g_tr & g_te) == 0 and len(g_va & g_te) == 0, \
        "GroupSplit found overlapping group_id across splits."

    # 鍐嶆鏌?canonical
    c_tr = set(df_train["canon"])
    c_va = set(df_val["canon"])
    c_te = set(df_test["canon"])
    print(
        f"[LeakCheck] canonical RC 浜ゅ弶锛歵rain鈭﹙al={len(c_tr & c_va)}, "
        f"train鈭﹖est={len(c_tr & c_te)}, val鈭﹖est={len(c_va & c_te)}"
    )
    assert len(c_tr & c_va) == 0 and len(c_tr & c_te) == 0 and len(c_va & c_te) == 0, \
        "Found identical canonical sequences across splits."

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
    璁粌闆嗗洶闅捐礋鏍锋湰閲囨牱锛?
    鐩爣锛氬敖閲忓湪鍚岃儗鏅紙group_id 浼樺厛锛夊唴鍋氬埌 pos/neg 骞宠　鎶芥牱銆?
    鑻ヤ笉瓒冲垯闄嶇骇鍒?tax_id -> species_name -> canon銆?
    鑻ヤ粛涓嶈冻鍒欏洖閫€鍘熻缁冮泦銆?
    """
    df = df_train.copy().reset_index(drop=True)
    total_pos = int((df["label"] == 1).sum())
    total_neg = int((df["label"] == 0).sum())

    print(f"[HardNeg] 鍘熻缁冮泦: pos={total_pos}, neg={total_neg}, total={len(df)}")

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

        print(f"[HardNeg] 鎸?{key_col} 鍖归厤: 鏈夋晥key={kept}, 绌?鍗曡竟key={dropped}")
        print(f"[HardNeg] 鍖归厤鍚? pos={n_pos}, neg={n_neg}, total={len(out)}")

        return out

    def _enough(df_candidate):
        if df_candidate is None:
            return False
        n_pos = int((df_candidate["label"] == 1).sum())
        threshold = max(min_pos, int(min_pos_ratio * max(1, total_pos)))
        return n_pos >= threshold

    # Stage 1: group_id
    df_hard = _match_by_key("group_id")

    # Stage 2: 闄嶇骇
    if not _enough(df_hard):
        print("[HardNeg] group_id 鍖归厤鏍锋湰杩囧皯锛屽皾璇曢檷绾т负 tax_id/species/canon")
        if "tax_id" in df.columns:
            df["hn_tax"] = df["tax_id"].astype(str)
            df_hard = _match_by_key("hn_tax")
        elif "species_name" in df.columns:
            df["hn_sp"] = df["species_name"].astype(str)
            df_hard = _match_by_key("hn_sp")
        else:
            df["hn_canon"] = df["canon"].astype(str)
            df_hard = _match_by_key("hn_canon")

    # Final: 鍥為€€
    if not _enough(df_hard):
        print("[HardNeg] Warning: insufficient matched hard negatives; fallback to original train set.")
        return df_train

    # 鍏ㄥ眬鍐嶅钩琛★紙淇濋櫓锛?
    n_pos = int((df_hard["label"] == 1).sum())
    n_neg = int((df_hard["label"] == 0).sum())
    if n_pos != n_neg:
        n = min(n_pos, n_neg)
        df_pos = df_hard[df_hard["label"] == 1].sample(n=n, random_state=random_state)
        df_neg = df_hard[df_hard["label"] == 0].sample(n=n, random_state=random_state)
        df_hard = pd.concat([df_pos, df_neg], ignore_index=True)\
                    .sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        print(f"[HardNeg] 鍏ㄥ眬鍐嶅钩琛? pos={n}, neg={n}, total={len(df_hard)}")

    return df_hard


# ========= Dataset =========
class BasenjiTrainDataset(Dataset):
    """Training dataset: random crop/pad and one-hot encoding."""
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


class BasenjiValDataset(Dataset):
    """Validation/test dataset: center crop/pad and one-hot encoding."""
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


# ========= Basenji 妯″瀷 =========
class Basenji(nn.Module):
    """
    Basenji 椋庢牸 1D CNN锛?
    鍏抽敭鏀瑰姩锛氶拡瀵瑰皬绐楀彛锛堝 64锛変慨姝ｇ涓夋姹犲寲锛岄伩鍏嶉暱搴﹀缂╁埌 1銆?
    """
    def __init__(self, seq_len=SEQ_LEN, num_classes=2,
                 base_filters=64, n_dilated_blocks=4, dilations=(1, 2, 4, 8)):
        super(Basenji, self).__init__()
        self.seq_len = seq_len

        self.conv1 = nn.Conv1d(4, base_filters, kernel_size=15, padding=7)
        self.bn1 = nn.BatchNorm1d(base_filters)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)

        self.conv2 = nn.Conv1d(base_filters, base_filters * 2, kernel_size=15, padding=7)
        self.bn2 = nn.BatchNorm1d(base_filters * 2)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)

        self.conv3 = nn.Conv1d(base_filters * 2, base_filters * 4, kernel_size=15, padding=7)
        self.bn3 = nn.BatchNorm1d(base_filters * 4)

        # 灏忕獥鍙ｈ嚜閫傚簲绗笁娆℃睜鍖?
        if seq_len <= 128:
            self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        else:
            self.pool3 = nn.MaxPool1d(kernel_size=4, stride=4)

        channels = base_filters * 4
        self.dilated_blocks = nn.ModuleList()
        self.dilated_bns = nn.ModuleList()
        self.dilated_dropouts = nn.ModuleList()

        if n_dilated_blocks != len(dilations):
            if n_dilated_blocks < len(dilations):
                dilations = dilations[:n_dilated_blocks]
            else:
                dilations = tuple(list(dilations) + [dilations[-1]] * (n_dilated_blocks - len(dilations)))

        for d in dilations:
            conv = nn.Conv1d(channels, channels, kernel_size=3, padding=d, dilation=d)
            bn = nn.BatchNorm1d(channels)
            self.dilated_blocks.append(conv)
            self.dilated_bns.append(bn)
            self.dilated_dropouts.append(nn.Dropout(0.1))

        self.dropout_conv = nn.Dropout(0.2)
        self.dropout_fc = nn.Dropout(0.5)

        self.fc1 = nn.Linear(channels, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout_conv(x)

        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout_conv(x)

        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout_conv(x)

        for conv, bn, drop in zip(self.dilated_blocks, self.dilated_bns, self.dilated_dropouts):
            y = F.relu(bn(conv(x)))
            y = drop(y)
            x = x + y

        x = x.mean(dim=2)

        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        logits = self.fc2(x)
        return logits


# ========= 璇勪及宸ュ叿 =========
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
def evaluate_loader(model, data_loader, criterion):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_labels, all_preds, all_probs = [], [], []

    for xs, ys in data_loader:
        xs = xs.to(device, non_blocking=True)
        ys = ys.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=USE_AMP):
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


# ========= 涓昏缁冩祦绋?=========
def main():
    # ---- 杈撳嚭鐩綍 ----
    try:
        script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0] or "interactive"
    except Exception:
        script_name = "interactive"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results_Basenji_{script_name}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"[Init] 杈撳嚭鐩綍: {output_dir}")

    # ---- 鏁版嵁鍑嗗 ----
    df_all = load_combined_data(POS_FILE, NEG_FILE, random_state=SEED)

    # ---- 璇婃柇 ----
    if ENABLE_BIAS_DIAGNOSTIC:
        _print_bias_diagnostic(df_all, topk=DIAG_SHOW_TOPK)

    # ---- 闀垮害鍖归厤 ----
    if ENABLE_LENGTH_MATCH:
        print("[LenMatch] 寮€鍚暱搴﹀尮閰嶏紙canonical 鍘婚噸鍚庯級")
        df_all = length_match_after_dedup(df_all, n_bins=LENGTH_MATCH_BINS, random_state=SEED)
        try:
            stat = (
                df_all.assign(seq_len=df_all["seq"].str.len())
                      .groupby(["len_bin", "label"], observed=True)
                      .size()
                      .reset_index(name="count")
            )
            stat.to_csv(os.path.join(output_dir, "length_match_bin_counts.csv"), index=False)
        except Exception as e:
            print(f"[LenMatch] 闀垮害鍒嗙缁熻淇濆瓨澶辫触锛堜笉褰卞搷璁粌锛夛細{e}")
    else:
        print("[LenMatch] Disabled.")

    # ---- GC 鍖归厤 ----
    if ENABLE_GC_MATCH:
        print("[GCMatch] 寮€鍚?GC 鍖归厤锛坈anonical 鍘婚噸鍚庯級")
        df_all = gc_match_after_dedup(df_all, n_bins=GC_MATCH_BINS, random_state=SEED)
        try:
            gc_stat = (
                df_all.assign(gc=df_all["seq"].map(gc_content))
                      .groupby(["gc_bin", "label"], observed=True)
                      .size()
                      .reset_index(name="count")
            )
            gc_stat.to_csv(os.path.join(output_dir, "gc_match_bin_counts.csv"), index=False)
        except Exception as e:
            print(f"[GCMatch] GC 鍒嗙缁熻淇濆瓨澶辫触锛堜笉褰卞搷璁粌锛夛細{e}")
    else:
        print("[GCMatch] Disabled.")

    # ---- GroupSplit锛坒rames鐗堬級----
    df_train, df_val, df_test = group_split_70_15_15_frames(df_all, random_state=SEED)

    # ---- 鍥伴毦璐熸牱鏈缁冮泦鏋勯€狅紙浠呰缁冮泦锛?---
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

    # ---- DataLoader ----
    pin_mem = torch.cuda.is_available()

    train_loader = DataLoader(
        BasenjiTrainDataset(train_sequences, y_train, seq_len=SEQ_LEN),
        batch_size=BATCH_SIZE_TRAIN,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=pin_mem,
        num_workers=NUM_WORKERS,
        persistent_workers=True if NUM_WORKERS > 0 else False,
    )

    val_loader = DataLoader(
        BasenjiValDataset(val_sequences, y_val, seq_len=SEQ_LEN),
        batch_size=BATCH_SIZE_EVAL,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=pin_mem,
        num_workers=NUM_WORKERS,
    )

    test_loader = DataLoader(
        BasenjiValDataset(test_sequences, y_test, seq_len=SEQ_LEN),
        batch_size=BATCH_SIZE_EVAL,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=pin_mem,
        num_workers=NUM_WORKERS,
    )

    # ---- 妯″瀷 & 浼樺寲鍣?----
    model = Basenji(seq_len=SEQ_LEN, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    num_update_steps_per_epoch = len(train_loader)
    num_train_steps = EPOCHS * num_update_steps_per_epoch
    num_warmup_steps = max(1, int(WARMUP_RATIO * num_train_steps))

    def get_lr(step, total_steps, warmup_steps, lr_peak):
        if step < warmup_steps:
            return lr_peak * float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return lr_peak * max(0.0, 1.0 - progress)

    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    # ---- 璁粌鐩戞帶 ----
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

    # ---- 璁粌寰幆 ----
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

        # 楠岃瘉
        val_loss, val_labels, val_preds, val_probs = evaluate_loader(model, val_loader, criterion)
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
            print(f"Early stopping triggered at epoch {epoch + 1}; best epoch is {early_stopping.best_epoch + 1}.")
            break

    # ---- 鍔犺浇鏈€浣虫ā鍨嬪苟鍦?TEST 涓婅瘎浼?----
    best_epoch_idx = early_stopping.best_epoch
    best_ckpt_path = resolve_checkpoint_path(os.path.join(checkpoint_dir, f"best_model_epoch{best_epoch_idx + 1}.pth"))

    if os.path.exists(best_ckpt_path):
        model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
        print(f"[Load Best] Loaded best checkpoint: {best_ckpt_path}")
    else:
        print("[Load Best] Best checkpoint not found, using current model weights for testing.")

    test_loss, test_labels, test_preds, test_probs = evaluate_loader(model, test_loader, criterion)
    test_metrics = compute_classification_metrics(test_labels, test_preds, test_probs)

    fpr, tpr, _ = roc_curve(test_labels, test_probs)
    auroc_final = auc(fpr, tpr)
    precision_vals, recall_vals, _ = precision_recall_curve(test_labels, test_probs)
    ap_final = average_precision_score(test_labels, test_probs)
    auprc_final = auc(recall_vals, precision_vals)

    
    save_curve_points(output_dir, fpr=fpr, tpr=tpr, recall=recall_vals, precision=precision_vals)
# ---- 缁樺浘 ----
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
    plt.step(recall_vals, precision_vals, where="post", lw=2,
             label=f"PR (AP = {ap_final:.2f}, AUPRC = {auprc_final:.2f})")
    plt.fill_between(recall_vals, precision_vals, step="post", alpha=0.2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR Curve (Test)")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pr_curve_test.png"))
    plt.close()

    # ---- 姹囨€?----
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

SEQ_LEN: {SEQ_LEN}
Length Match Enabled: {ENABLE_LENGTH_MATCH}
Length Match Bins: {LENGTH_MATCH_BINS}
GC Match Enabled: {ENABLE_GC_MATCH}
GC Match Bins: {GC_MATCH_BINS}
HardNeg Train Enabled: {ENABLE_HARD_NEG_TRAIN}
HardNeg Min Pos: {HARD_NEG_MIN_POS}
HardNeg Min Pos Ratio: {HARD_NEG_MIN_POS_RATIO}
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
        writer.writerow(["SEQ_LEN", SEQ_LEN])
        writer.writerow(["Length Match Enabled", ENABLE_LENGTH_MATCH])
        writer.writerow(["Length Match Bins", LENGTH_MATCH_BINS])
        writer.writerow(["GC Match Enabled", ENABLE_GC_MATCH])
        writer.writerow(["GC Match Bins", GC_MATCH_BINS])
        writer.writerow(["HardNeg Train Enabled", ENABLE_HARD_NEG_TRAIN])
        writer.writerow(["HardNeg Min Pos", HARD_NEG_MIN_POS])
        writer.writerow(["HardNeg Min Pos Ratio", HARD_NEG_MIN_POS_RATIO])

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
        plt.grid()
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
    print("Note: training-period metrics are based on validation; final report metrics are based on test set.")
    print("Note: this version includes canonical dedup + pooling fix + length/GC matching + hard negative training.")
    print("If metrics remain unusually high, inspect data source/species distribution mismatch and redefine negatives.")


if __name__ == "__main__":
    main()


