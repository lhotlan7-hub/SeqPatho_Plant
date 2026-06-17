# -*- coding: utf-8 -*-
import os
import re
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle, PathPatch
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties
from matplotlib.transforms import Affine2D

from transformers import AutoModel, AutoTokenizer


# ============================================================
# USER CONFIG（在这里直接填写你的路径与参数）
# ============================================================

MODEL_DIR = r"C:\Users\GQ\.cache\huggingface\modules\transformers_modules\DNABERT-S"
PTH_PATH  = r"E:\4PythonProject\SeqPatho\Models_06_NPlants\results_PGPred_11_500_20251112_180132\checkpoints\best_model_epoch18.pth"
OUTDIR    = r"E:\4PythonProject\SeqPatho\Sample\attn_viz_out"

INPUTS: List[Dict] = [
    {"path": r"E:\4PythonProject\SeqPatho\Dataset\patho_plants_only.csv",    "csv_col": None, "csv_col_idx": 14, "tag": "patho"},
    {"path": r"E:\4PythonProject\SeqPatho\Dataset\Nonpatho_plants_only.csv", "csv_col": None, "csv_col_idx": 13, "tag": "nonpatho"},
]

MAX_LEN = 500
STRIDE_TOKENS = MAX_LEN // 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HIDDEN_SIZE = 64
FREEZE_DNABERT = False
FINETUNE_FROM_LAYER = 10
WINDOW_PICK_METRIC = "prob1"   # "prob1" or "margin"

MAX_SEQS_PER_FILE = 50
STRICT_LOAD = False

# 图1：flow
TOPK_EDGES = 4
MAX_TOKENS_TO_PLOT_FLOW = 140
EDGE_ALPHA_SCALE = 0.25

# 图2：heatmap
HEATMAP_LAYER = -1
MAX_TOKENS_TO_PLOT_HEATMAP = 120

# 图3：importance —— 输出更像“示例2”的 logo + heat strip
IMPORTANCE_STYLE = "saliency_logo"   # "saliency_logo" or "bar"
ROLLOUT_ADD_IDENTITY = True
MAX_TOKENS_FOR_IMPORTANCE = 260      # 参与 importance 的 token 上限（越大越慢）

# === base importance 映射策略（解决“连续几个高度一样”的主要开关）===
# token->base 聚合： "sum" / "mean" / "max"
# - mean 容易出现大片平台（k-mer覆盖导致）
# - max 更容易突出局部峰（推荐）
BASE_AGG_MODE = "max"

# importance 归一化/压缩（防止个别位置极大导致其余都挤扁）
SAL_CLIP_PCT = 99.5          # 99.0 ~ 99.9 之间试
SAL_LOGO_GAMMA = 0.85        # <1 放大差异；=1 线性；>1 压缩差异
SAL_MIN_HEIGHT = 0.0         # 最小高度（一般 0）

# === 图形布局（解决：字母不覆盖色块；关键区域居中显示）===
SAL_FIG_W = 24
SAL_FIG_H = 4.2
SAL_CMAP = "YlGnBu"          # 背景色块 colormap
SAL_CENTER_ON_PEAK = True    # 强制把最高点放在可视化中心
SAL_SHOW_BASES = 240         # 最终展示多少个碱基（越大越密）
SAL_FLANK = 120              # peak 左右各保留多少碱基（一般 = SAL_SHOW_BASES//2）
SAL_HIGHLIGHT_WIDTH_BASES = 40
SAL_SEQ_FONT_SIZE = 9        # 序列行字体
SAL_SEQ_MAX_LETTERS = 280    # 超过这个长度就自动隔几个字母标一次（避免糊）

# DNA logo 配色（接近常见 sequence logo）
BASE_COLORS = {
    "A": "#2ca02c",  # green
    "C": "#1f77b4",  # blue
    "G": "#ff7f0e",  # orange
    "T": "#d62728",  # red
    "N": "#7f7f7f"
}

# 可选：同时输出条形 TopK（你若不需要可以关）
ALSO_SAVE_BAR = False
BAR_TOPK = 40


# ============================================================
# Utils
# ============================================================
def canon_seq(s: str) -> str:
    return "".join(str(s).split()).upper()

def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def read_sequences_any(
    path: str,
    csv_col: Optional[str] = None,
    csv_col_idx: Optional[int] = None,
    max_n: Optional[int] = None
) -> List[str]:
    p = Path(path)
    suf = p.suffix.lower()

    seqs: List[str] = []
    if suf in [".fa", ".fasta", ".fna", ".ffn"]:
        cur: List[str] = []
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if cur:
                        seqs.append(canon_seq("".join(cur)))
                        cur = []
                        if max_n and len(seqs) >= max_n:
                            break
                else:
                    cur.append(line)
            if cur and (not max_n or len(seqs) < max_n):
                seqs.append(canon_seq("".join(cur)))

    elif suf in [".txt", ".tsv"]:
        with p.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                seqs.append(canon_seq(line))
                if max_n and len(seqs) >= max_n:
                    break

    elif suf in [".csv"]:
        import pandas as pd
        df = pd.read_csv(p, low_memory=False)
        if csv_col is not None:
            if csv_col not in df.columns:
                raise ValueError(f"CSV列名不存在: {csv_col}. 可选列示例: {list(df.columns)[:30]}")
            series = df[csv_col]
        elif csv_col_idx is not None:
            series = df.iloc[:, int(csv_col_idx)]
        else:
            series = df.iloc[:, 0]

        for x in series.dropna().astype(str).tolist():
            seqs.append(canon_seq(x))
            if max_n and len(seqs) >= max_n:
                break
    else:
        raise ValueError(f"不支持的输入格式: {suf}（支持 csv/fa/fasta/txt）")

    return [s for s in seqs if len(s) > 0]

def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    new_sd = {}
    for k, v in state_dict.items():
        new_sd[k.replace("module.", "", 1)] = v
    return new_sd

def clean_token_for_display(tok: str) -> str:
    return tok.replace("##", "").replace("Ġ", "").replace("▁", "")

def is_special_token(tokenizer: Any, tok: str) -> bool:
    try:
        return tok in set(tokenizer.all_special_tokens)
    except Exception:
        return tok.startswith("[") and tok.endswith("]")


# ============================================================
# Model (match training)
# ============================================================
class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, reduction_ratio: int = 8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_perm = x.permute(0, 2, 1)  # [B,C,L]
        avg_out = self.fc(self.avg_pool(x_perm).squeeze(-1))
        max_out = self.fc(self.max_pool(x_perm).squeeze(-1))
        scale = self.sigmoid(avg_out + max_out).unsqueeze(-1)
        out = x_perm * scale
        return out.permute(0, 2, 1)

class StackedChannelAttention(nn.Module):
    def __init__(self, in_channels: int, num_layers: int = 3):
        super().__init__()
        self.layers = nn.ModuleList([ChannelAttention(in_channels) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(in_channels) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attn, norm in zip(self.layers, self.norms):
            residual = x
            x = attn(x)
            x = norm(x + residual)
        return x


# ============================================================
# Attention capture patch (不依赖 out.attentions)
# ============================================================
def _infer_heads_and_dim(attn_mod: nn.Module, hidden_size: int = 768) -> Tuple[int, int]:
    for hn_attr in ["num_attention_heads", "n_heads", "num_heads"]:
        if hasattr(attn_mod, hn_attr):
            n_heads = int(getattr(attn_mod, hn_attr))
            break
    else:
        n_heads = 12

    for hd_attr in ["attention_head_size", "head_dim", "dim_head"]:
        if hasattr(attn_mod, hd_attr):
            head_dim = int(getattr(attn_mod, hd_attr))
            break
    else:
        head_dim = hidden_size // max(n_heads, 1)

    if n_heads * head_dim != hidden_size:
        if hidden_size % n_heads == 0:
            head_dim = hidden_size // n_heads
        else:
            n_heads = 12
            head_dim = hidden_size // 12
    return n_heads, head_dim

def _compute_attn_probs_from_hidden(attn_mod: nn.Module, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if hidden_states is None or (not torch.is_tensor(hidden_states)):
        return None

    x = hidden_states
    if x.dim() == 2:
        x3 = x.unsqueeze(0)
    elif x.dim() == 3:
        x3 = x
    else:
        return None

    B, L, C = x3.shape

    q = k = v = None
    if hasattr(attn_mod, "Wqkv") and isinstance(attn_mod.Wqkv, nn.Linear):
        in_feat = attn_mod.Wqkv.in_features
        if C == in_feat:
            qkv = attn_mod.Wqkv(x3)
            q, k, v = qkv.chunk(3, dim=-1)
            H = q.shape[-1]
        elif C == 3 * in_feat:
            qkv = x3
            q, k, v = qkv.chunk(3, dim=-1)
            H = q.shape[-1]
        else:
            q = None

    if q is None:
        if hasattr(attn_mod, "query") and hasattr(attn_mod, "key") and hasattr(attn_mod, "value"):
            q = attn_mod.query(x3)
            k = attn_mod.key(x3)
            v = attn_mod.value(x3)
            H = q.shape[-1]
        else:
            return None

    n_heads, head_dim = _infer_heads_and_dim(attn_mod, hidden_size=H)
    if q.shape[-1] != n_heads * head_dim:
        if q.shape[-1] % n_heads == 0:
            head_dim = q.shape[-1] // n_heads
        else:
            return None

    q = q.view(B, L, n_heads, head_dim).transpose(1, 2)
    k = k.view(B, L, n_heads, head_dim).transpose(1, 2)
    v = v.view(B, L, n_heads, head_dim).transpose(1, 2)

    scores = torch.matmul(q, k.transpose(-1, -2)) / (head_dim ** 0.5)

    if attention_mask is not None and torch.is_tensor(attention_mask):
        if attention_mask.dim() == 2:
            pad = (attention_mask == 0)
            if pad.shape[1] == L:
                scores = scores.masked_fill(pad[:, None, None, :], -1e4)
        elif attention_mask.dim() == 4:
            scores = scores + attention_mask

    probs = torch.softmax(scores, dim=-1)
    if hasattr(attn_mod, "dropout") and isinstance(attn_mod.dropout, nn.Dropout):
        probs = attn_mod.dropout(probs)

    return probs

def patch_bert_attention_capture(bert_model: nn.Module) -> int:
    if hasattr(bert_model, "_attn_capture_modules"):
        return len(getattr(bert_model, "_attn_capture_modules"))

    candidates = []
    for name, m in bert_model.named_modules():
        cls = m.__class__.__name__.lower()
        if ("selfattention" in cls) or ("unpadselfattention" in cls) or ("self_att" in cls) or ("attention" in cls and "self" in cls):
            if hasattr(m, "Wqkv") or (hasattr(m, "query") and hasattr(m, "key") and hasattr(m, "value")):
                layer_id = 10**9
                mm = re.search(r"encoder\.layer\.(\d+)\.", name)
                if mm:
                    layer_id = int(mm.group(1))
                candidates.append((layer_id, name, m))

    candidates.sort(key=lambda x: x[0])
    attn_mods = [x[2] for x in candidates]

    bert_model._capture_attn = False
    bert_model._attn_capture_modules = attn_mods

    for m in attn_mods:
        m._last_attn = None
        orig_forward = m.forward

        def make_wrapped_forward(mod, orig_fwd):
            def wrapped_forward(*args, **kwargs):
                out = orig_fwd(*args, **kwargs)

                if not getattr(bert_model, "_capture_attn", False):
                    return out

                hidden_states = None
                attention_mask = None

                if len(args) >= 1 and torch.is_tensor(args[0]):
                    hidden_states = args[0]
                if "hidden_states" in kwargs and torch.is_tensor(kwargs["hidden_states"]):
                    hidden_states = kwargs["hidden_states"]

                if len(args) >= 2 and torch.is_tensor(args[1]) and args[1].dim() in (2, 4):
                    attention_mask = args[1]
                if "attention_mask" in kwargs and torch.is_tensor(kwargs["attention_mask"]):
                    attention_mask = kwargs["attention_mask"]

                try:
                    probs = _compute_attn_probs_from_hidden(mod, hidden_states, attention_mask)
                    mod._last_attn = probs.detach().float().cpu() if probs is not None else None
                except Exception:
                    mod._last_attn = None

                return out
            return wrapped_forward

        m.forward = make_wrapped_forward(m, orig_forward)

    return len(attn_mods)


# ============================================================
# Main classification model
# ============================================================
class DNABiGRUWithDNABERT(nn.Module):
    def __init__(
        self,
        dnabert_dir: str,
        num_classes: int = 2,
        hidden_size: int = 64,
        freeze_dnabert: bool = False,
        finetune_from_layer: int = 10
    ):
        super().__init__()
        self.channel_attention = StackedChannelAttention(in_channels=hidden_size * 2, num_layers=3)
        self.dropout = nn.Dropout(p=0.5)
        self.layernorm = nn.LayerNorm(hidden_size * 2)

        self.dnabert_model = AutoModel.from_pretrained(dnabert_dir, trust_remote_code=True)
        self.bert_hidden_size = 768
        self.freeze_dnabert = freeze_dnabert

        for name, param in self.dnabert_model.named_parameters():
            if "encoder.layer" in name:
                m = re.search(r"encoder\.layer\.(\d+)\.", name)
                if m:
                    layer_num = int(m.group(1))
                    param.requires_grad = (layer_num >= finetune_from_layer)
                else:
                    param.requires_grad = False
            else:
                param.requires_grad = False

        self.bigru = nn.GRU(
            input_size=self.bert_hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True
        )
        self.fc1 = nn.Linear(hidden_size * 2, 32)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, return_bert_attn: bool = False):
        if return_bert_attn and hasattr(self.dnabert_model, "_attn_capture_modules"):
            self.dnabert_model._capture_attn = True
            for m in self.dnabert_model._attn_capture_modules:
                m._last_attn = None

        if self.freeze_dnabert:
            with torch.no_grad():
                out = self.dnabert_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        else:
            out = self.dnabert_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)

        embeddings = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]

        bert_attns = None
        if return_bert_attn and hasattr(self.dnabert_model, "_attn_capture_modules"):
            bert_attns = []
            for m in self.dnabert_model._attn_capture_modules:
                bert_attns.append(m._last_attn)
            self.dnabert_model._capture_attn = False

        if attention_mask is not None:
            seq_lengths = attention_mask.sum(dim=1)
        else:
            seq_lengths = (input_ids != 0).sum(dim=1)

        packed = nn.utils.rnn.pack_padded_sequence(
            embeddings, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.bigru(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, total_length=embeddings.size(1)
        )

        output = self.channel_attention(output)
        output = self.layernorm(output)
        output = self.dropout(output)

        L = output.size(1)
        if attention_mask is None:
            mask = (input_ids != 0)[:, :L].unsqueeze(-1).float()
        else:
            mask = attention_mask[:, :L].unsqueeze(-1).float()

        masked_output = output * mask
        sum_output = masked_output.sum(dim=1)
        avg_output = sum_output / (mask.sum(dim=1).clamp_min(1e-6))

        x = F.relu(self.fc1(avg_output))
        x = self.dropout(x)
        logits = self.fc2(x)

        if return_bert_attn:
            return logits, bert_attns
        return logits


# ============================================================
# Tokenization helper: try to get offsets_mapping
# ============================================================
def tokenize_with_optional_offsets(tokenizer, seq: str, max_len: int, stride_tokens: int):
    try:
        enc = tokenizer(
            seq,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_overflowing_tokens=True,
            stride=stride_tokens,
            return_offsets_mapping=True,
            return_tensors="pt"
        )
        return enc, True
    except Exception:
        enc = tokenizer(
            seq,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_overflowing_tokens=True,
            stride=stride_tokens,
            return_tensors="pt"
        )
        return enc, False


# ============================================================
# Attn extraction (choose best window then get attn)
# ============================================================
@torch.no_grad()
def pick_best_window_and_get_attn(
    seq: str,
    model: nn.Module,
    tokenizer,
    device: torch.device,
    max_len: int,
    stride_tokens: int,
    metric: str
):
    model.eval()
    enc, has_offsets = tokenize_with_optional_offsets(tokenizer, seq, max_len, stride_tokens)

    ids_all = enc["input_ids"]
    mask_all = enc["attention_mask"]
    if ids_all.ndim == 1:
        ids_all = ids_all.unsqueeze(0)
        mask_all = mask_all.unsqueeze(0)

    W = ids_all.size(0)
    if W == 0:
        return None, None, None, -1, float("nan"), float("nan")

    best_idx = 0
    best_score = -1e18
    best_prob1 = float("nan")

    for i in range(W):
        ids = ids_all[i:i+1].to(device)
        msk = mask_all[i:i+1].to(device).long()
        logits = model(ids, msk).float().squeeze(0)

        p1 = torch.softmax(logits, dim=0)[1].item()
        score = p1 if metric == "prob1" else (logits[1] - logits[0]).item()
        if score > best_score:
            best_score = score
            best_idx = i
            best_prob1 = p1

    ids = ids_all[best_idx:best_idx+1].to(device)
    msk = mask_all[best_idx:best_idx+1].to(device).long()
    logits, bert_attns = model(ids, msk, return_bert_attn=True)

    if bert_attns is None or all(x is None for x in bert_attns):
        return None, None, None, best_idx, float(best_score), float(best_prob1)

    V = int(msk.sum().item())
    ids_v = ids[0, :V].detach().cpu().tolist()
    tokens = tokenizer.convert_ids_to_tokens(ids_v)

    offsets = None
    if has_offsets:
        off_all = enc.get("offset_mapping", None)
        if off_all is not None:
            try:
                off_win = off_all[best_idx]
                if torch.is_tensor(off_win):
                    off_win = off_win[:V].detach().cpu().tolist()
                else:
                    off_win = off_win[:V]
                offsets = [(int(a), int(b)) for a, b in off_win]
            except Exception:
                offsets = None

    attn_mats = []
    for A in bert_attns:
        if A is None:
            continue
        A2 = A[0].mean(dim=0)[:V, :V].float().numpy()
        A2 = A2 / (A2.sum(axis=1, keepdims=True) + 1e-12)
        attn_mats.append(A2)

    if len(attn_mats) == 0:
        return None, None, None, best_idx, float(best_score), float(best_prob1)

    return tokens, offsets, attn_mats, best_idx, float(best_score), float(best_prob1)


# ============================================================
# Plot 1: flow
# ============================================================
def plot_attention_flow(attn_mats: List[np.ndarray], out_png: str,
                        topk: int, max_tokens_to_plot: int, edge_alpha_scale: float):
    n_layers = len(attn_mats)
    V = attn_mats[0].shape[0]
    Vp = min(V, max_tokens_to_plot)

    xs = np.arange(n_layers + 1)
    ys = np.linspace(1.0, 0.0, Vp)

    segments = []
    colors = []

    for l in range(1, n_layers + 1):
        A = attn_mats[l - 1][:Vp, :Vp]
        for i in range(Vp):
            row = A[i]
            idx = np.argsort(row)[-topk:][::-1]
            for j in idx:
                w = float(row[j])
                segments.append([(xs[l - 1], ys[j]), (xs[l], ys[i])])
                a = min(1.0, edge_alpha_scale * (w * topk * 2.5))
                colors.append((0.25, 0.70, 1.00, a))

    fig = plt.figure(figsize=(13, 4), facecolor="black")
    ax = fig.add_subplot(111)
    ax.set_facecolor("black")

    lc = LineCollection(segments, colors=colors, linewidths=0.7)
    ax.add_collection(lc)

    for x in xs:
        ax.axvline(x, color=(1, 1, 1, 0.06), linewidth=1.0)

    ax.set_xlim(xs[0] - 0.5, xs[-1] + 0.5)
    ax.set_ylim(-0.05, 1.05)
    ax.axis("off")

    fig.tight_layout(pad=0)
    fig.savefig(out_png, dpi=220, facecolor=fig.get_facecolor())
    plt.close(fig)


# ============================================================
# Plot 2: heatmap
# ============================================================
def plot_attention_heatmap(attn_mats: List[np.ndarray], out_png: str, layer: int, max_tokens_to_plot: int, title: str):
    L = len(attn_mats)
    li = layer if layer >= 0 else (L + layer)
    li = max(0, min(L - 1, li))

    A = attn_mats[li]
    V = A.shape[0]
    Vp = min(V, max_tokens_to_plot)
    A = A[:Vp, :Vp]

    fig = plt.figure(figsize=(7.5, 6.8), facecolor="white")
    ax = fig.add_subplot(111)
    im = ax.imshow(A, aspect="auto", interpolation="nearest")
    ax.set_xlabel("Key token index")
    ax.set_ylabel("Query token index")
    ax.set_title(title if title else f"Attention heatmap (layer {li+1}/{L})")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Attention weight")
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


# ============================================================
# Rollout importance
# ============================================================
def attention_rollout(attn_mats: List[np.ndarray], add_identity: bool = True) -> np.ndarray:
    V = attn_mats[0].shape[0]
    R = np.eye(V, dtype=np.float32)
    for A in attn_mats:
        A2 = A.astype(np.float32)
        if add_identity:
            A2 = A2 + np.eye(V, dtype=np.float32)
        A2 = A2 / (A2.sum(axis=1, keepdims=True) + 1e-12)
        R = A2 @ R
    return R

def clip_and_normalize(x: np.ndarray, clip_pct: float = 99.5) -> np.ndarray:
    if x.size == 0:
        return x.astype(np.float32)
    x = x.astype(np.float32)
    x = x - float(np.min(x))
    hi = float(np.percentile(x, clip_pct))
    if hi <= 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    x = np.clip(x, 0.0, hi) / hi
    return x

def plot_token_importance_topk(tokens: List[str], importance: np.ndarray, out_png: str, topk: int, title: str):
    imp = importance.astype(np.float32)
    s = imp.sum()
    if s > 0:
        imp = imp / (s + 1e-12)
    idx = np.argsort(imp)[::-1][:min(topk, len(imp))]

    sel_imp = imp[idx][::-1]
    sel_tok = [clean_token_for_display(tokens[i]) for i in idx][::-1]

    fig = plt.figure(figsize=(10.5, 7.0), facecolor="white")
    ax = fig.add_subplot(111)
    ax.barh(range(len(sel_tok)), sel_imp)
    ax.set_yticks(range(len(sel_tok)))
    ax.set_yticklabels(sel_tok, fontsize=8)
    ax.set_xlabel("Normalized importance")
    ax.set_title(title if title else f"Token importance (Top-{len(sel_tok)})")
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)

def save_token_importance_csv(tokens: List[str], importance: np.ndarray, out_csv: str):
    imp = importance.astype(np.float32)
    s = imp.sum()
    if s > 0:
        imp = imp / (s + 1e-12)
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("idx,token,importance\n")
        for i, (t, v) in enumerate(zip(tokens, imp.tolist())):
            f.write(f"{i},{t},{v:.10f}\n")


# ============================================================
# token importance -> base importance（关键：聚合方式可选，影响“高度是否平台化”）
# ============================================================
def token_importance_to_base_importance(
    seq: str,
    offsets: List[Tuple[int, int]],
    token_imp: np.ndarray,
    agg_mode: str = "max"
) -> Tuple[str, np.ndarray, int, int]:
    valid = [(i, s, e) for i, (s, e) in enumerate(offsets) if (s is not None and e is not None and e > s)]
    if not valid:
        return "", np.zeros((0,), dtype=np.float32), 0, 0

    win_start = min(s for _, s, _ in valid)
    win_end   = max(e for _, _, e in valid)
    win_start = max(0, win_start)
    win_end   = min(len(seq), win_end)
    win_seq = seq[win_start:win_end]
    n = len(win_seq)

    base_sum = np.zeros(n, dtype=np.float32)
    base_cnt = np.zeros(n, dtype=np.float32)
    base_max = np.zeros(n, dtype=np.float32)

    for i, s, e in valid:
        s2 = max(s, win_start) - win_start
        e2 = min(e, win_end) - win_start
        if e2 > s2:
            v = float(token_imp[i])
            base_sum[s2:e2] += v
            base_cnt[s2:e2] += 1.0
            base_max[s2:e2] = np.maximum(base_max[s2:e2], v)

    mode = (agg_mode or "max").lower()
    if mode == "sum":
        base_imp = base_sum
    elif mode == "mean":
        base_imp = base_sum / np.maximum(base_cnt, 1.0)
    else:  # "max"
        base_imp = base_max

    return win_seq, base_imp.astype(np.float32), win_start, win_end


# ============================================================
# New: true DNA logo (letter height = score), plus heat strip and sequence row
# ============================================================
_DNA_FONT = FontProperties(family="DejaVu Sans", weight="bold")

def draw_scaled_letter(ax, letter: str, x_center: float, height: float, color: str):
    """
    画一个可缩放字形：高度为 height（0~1）
    x_center: 用“碱基列”为单位的中心位置
    """
    letter = letter.upper()
    if height <= 1e-6:
        return
    # TextPath 默认以字体大小为单位，先做一个 unit glyph，再缩放
    tp = TextPath((0, 0), letter, size=1.0, prop=_DNA_FONT)
    bb = tp.get_extents()
    if bb.height <= 1e-12 or bb.width <= 1e-12:
        return

    # 目标：列宽约 1.0；高度=height
    sx = 0.85 / bb.width
    sy = float(height) / bb.height

    # 平移到列中心，并让基线贴近 y=0
    trans = (Affine2D()
             .scale(sx, sy)
             .translate(x_center - (bb.width * sx) / 2.0, 0.0))

    patch = PathPatch(tp, transform=trans + ax.transData, fc=color, ec="none", alpha=1.0)
    ax.add_patch(patch)

def _pick_display_slice(n: int, peak: int, show_bases: int, center_on_peak: bool = True) -> Tuple[int, int]:
    show_bases = min(show_bases, n)
    if show_bases <= 0:
        return 0, n
    if not center_on_peak:
        return 0, show_bases
    left = peak - show_bases // 2
    left = max(0, min(left, n - show_bases))
    right = left + show_bases
    return left, right

def plot_saliency_logo_strip(
    win_seq: str,
    base_imp: np.ndarray,
    out_png: str,
    title: str,
    highlight_width: int,
    cmap: str,
    fig_w: float,
    fig_h: float,
    clip_pct: float,
    gamma: float,
    min_height: float,
    show_bases: int,
    center_on_peak: bool,
    seq_font_size: int,
    seq_max_letters: int
):
    """
    输出 3 行：
      (1) DNA logo：单序列，字母高度=importance（连续）
      (2) sequence row：字母在色块上方（不覆盖）
      (3) heat strip：底部色块
    """
    if base_imp.size == 0 or len(win_seq) == 0:
        return

    n = min(len(win_seq), base_imp.size)
    win_seq = win_seq[:n]
    base_imp = base_imp[:n]

    # robust normalize
    base_imp = clip_and_normalize(base_imp, clip_pct=clip_pct)

    # gamma 变换：增强差异
    base_imp = np.power(np.clip(base_imp, 0.0, 1.0), float(gamma)).astype(np.float32)
    if min_height > 0:
        base_imp = np.maximum(base_imp, float(min_height))

    peak = int(np.argmax(base_imp)) if n > 0 else 0
    ss, ee = _pick_display_slice(n, peak, show_bases, center_on_peak=center_on_peak)

    seq_plot = win_seq[ss:ee]
    imp_plot = base_imp[ss:ee]
    m = len(seq_plot)
    if m <= 0:
        return

    # highlight range around peak in displayed coords
    hw = max(5, int(highlight_width))
    hs = max(ss, peak - hw // 2)
    he = min(ee, hs + hw)
    hs2 = hs - ss
    he2 = he - ss

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    gs = fig.add_gridspec(3, 1, height_ratios=[1.35, 0.45, 0.85], hspace=0.05)

    # --- (1) logo
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.set_xlim(-0.5, m - 0.5)
    ax0.set_ylim(0.0, 1.05)
    ax0.axis("off")

    for i, (ch, v) in enumerate(zip(seq_plot, imp_plot)):
        col = BASE_COLORS.get(ch.upper(), BASE_COLORS["N"])
        draw_scaled_letter(ax0, ch, x_center=float(i), height=float(v), color=col)

    # highlight overlay
    ax0.add_patch(Rectangle((hs2 - 0.5, 0.0), max(0.0, he2 - hs2), 1.05,
                            facecolor="red", alpha=0.08, edgecolor="red", linewidth=1.2))

    # --- (2) sequence row (letters ABOVE heat strip)
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.set_xlim(-0.5, m - 0.5)
    ax1.set_ylim(0.0, 1.0)
    ax1.axis("off")

    # 太长时隔 N 个写一个，避免糊
    step = max(1, int(np.ceil(m / max(1, seq_max_letters))))
    for i in range(0, m, step):
        ax1.text(i, 0.5, seq_plot[i], ha="center", va="center",
                 fontsize=seq_font_size, family="DejaVu Sans Mono", color="black")

    ax1.add_patch(Rectangle((hs2 - 0.5, 0.0), max(0.0, he2 - hs2), 1.0,
                            facecolor="red", alpha=0.08, edgecolor="red", linewidth=1.2))

    # --- (3) heat strip
    ax2 = fig.add_subplot(gs[2, 0])
    ax2.set_xlim(-0.5, m - 0.5)
    ax2.set_ylim(-0.5, 0.5)
    ax2.axis("off")

    heat = imp_plot.reshape(1, -1)
    ax2.imshow(heat, aspect="auto", interpolation="nearest", cmap=cmap,
               extent=[-0.5, m - 0.5, -0.5, 0.5])

    ax2.add_patch(Rectangle((hs2 - 0.5, -0.5), max(0.0, he2 - hs2), 1.0,
                            facecolor="red", alpha=0.12, edgecolor="red", linewidth=1.2))

    if title:
        fig.suptitle(title, y=0.98, fontsize=12)

    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


# ============================================================
# Run
# ============================================================
def run_one_input(model: nn.Module, tokenizer, device: torch.device, in_cfg: Dict, out_root: Path):
    in_path = in_cfg["path"]
    csv_col = in_cfg.get("csv_col", None)
    csv_col_idx = in_cfg.get("csv_col_idx", None)
    tag = in_cfg.get("tag", Path(in_path).stem)

    outdir = out_root / tag
    safe_mkdir(outdir)

    seqs = read_sequences_any(in_path, csv_col=csv_col, csv_col_idx=csv_col_idx, max_n=MAX_SEQS_PER_FILE)
    print(f"\n[INFO] Input: {in_path}")
    print(f"[INFO] Tag  : {tag}")
    print(f"[INFO] Loaded sequences: {len(seqs)}")

    ok = 0
    skipped = 0
    total_png = 0

    for idx, seq in enumerate(seqs):
        tokens, offsets, attn_mats, widx, wscore, p1 = pick_best_window_and_get_attn(
            seq, model, tokenizer, device, MAX_LEN, STRIDE_TOKENS, WINDOW_PICK_METRIC
        )

        if tokens is None or attn_mats is None:
            skipped += 1
            print(f"[SKIP] {tag} seq#{idx} attentions not returned.")
            continue

        ok += 1
        n_layers = len(attn_mats)
        file_tag = f"{tag}_seq{idx:04d}_p1_{p1:.4f}_win{widx}_score_{wscore:.4f}_L{n_layers}"

        # 1) flow
        png_flow = outdir / f"attn_flow_{file_tag}.png"
        plot_attention_flow(attn_mats, str(png_flow), TOPK_EDGES, MAX_TOKENS_TO_PLOT_FLOW, EDGE_ALPHA_SCALE)

        # 2) heatmap
        png_heat = outdir / f"attn_heatmap_{file_tag}_layer{HEATMAP_LAYER}.png"
        plot_attention_heatmap(attn_mats, str(png_heat), HEATMAP_LAYER, MAX_TOKENS_TO_PLOT_HEATMAP,
                               title=f"{tag} seq#{idx}  p1={p1:.4f}  win={widx}  layers={n_layers}")

        # 3) importance (rollout from CLS)
        V = attn_mats[0].shape[0]
        Vp = min(V, MAX_TOKENS_FOR_IMPORTANCE)

        attn_cut = [A[:Vp, :Vp] for A in attn_mats]
        tokens_cut = tokens[:Vp]
        offsets_cut = offsets[:Vp] if (offsets is not None and len(offsets) >= Vp) else None

        R = attention_rollout(attn_cut, add_identity=ROLLOUT_ADD_IDENTITY)
        imp = R[0, :].copy()  # [CLS] -> all

        # 过滤 special tokens（避免 [CLS] 导致显示异常）
        keep_idx = [i for i, t in enumerate(tokens_cut) if not is_special_token(tokenizer, t)]
        if len(keep_idx) >= 5:
            tokens_vis = [tokens_cut[i] for i in keep_idx]
            imp_vis = imp[keep_idx]
            offsets_vis = [offsets_cut[i] for i in keep_idx] if offsets_cut is not None else None
        else:
            tokens_vis = tokens_cut
            imp_vis = imp
            offsets_vis = offsets_cut

        # 输出 saliency/logo
        png_imp = None
        if IMPORTANCE_STYLE == "bar":
            png_imp = outdir / f"attn_importance_bar_{file_tag}_top{BAR_TOPK}.png"
            plot_token_importance_topk(tokens_vis, imp_vis, str(png_imp), BAR_TOPK,
                                       title=f"Rollout importance BAR (tag={tag}, seq#{idx}, p1={p1:.4f})")
        else:
            if offsets_vis is not None:
                win_seq, base_imp, ws, we = token_importance_to_base_importance(
                    seq=seq,
                    offsets=offsets_vis,
                    token_imp=imp_vis,
                    agg_mode=BASE_AGG_MODE
                )
                png_imp = outdir / f"attn_importance_logo_{file_tag}.png"
                plot_saliency_logo_strip(
                    win_seq=win_seq,
                    base_imp=base_imp,
                    out_png=str(png_imp),
                    title=f"Saliency+Logo (tag={tag}, seq#{idx}, p1={p1:.4f}, win={widx}, layers={n_layers})",
                    highlight_width=SAL_HIGHLIGHT_WIDTH_BASES,
                    cmap=SAL_CMAP,
                    fig_w=SAL_FIG_W,
                    fig_h=SAL_FIG_H,
                    clip_pct=SAL_CLIP_PCT,
                    gamma=SAL_LOGO_GAMMA,
                    min_height=SAL_MIN_HEIGHT,
                    show_bases=SAL_SHOW_BASES,
                    center_on_peak=SAL_CENTER_ON_PEAK,
                    seq_font_size=SAL_SEQ_FONT_SIZE,
                    seq_max_letters=SAL_SEQ_MAX_LETTERS
                )
            else:
                # offsets 不支持：只能退化到 token-strip 或 bar
                png_imp = outdir / f"attn_importance_tokenstrip_{file_tag}.png"
                vals = clip_and_normalize(imp_vis, clip_pct=SAL_CLIP_PCT)
                fig = plt.figure(figsize=(SAL_FIG_W, 2.2), facecolor="white")
                ax = fig.add_subplot(111)
                heat = vals.reshape(1, -1)
                ax.imshow(heat, aspect="auto", interpolation="nearest", cmap=SAL_CMAP)
                ax.set_yticks([])
                ax.set_xticks([])
                ax.set_title(f"Token saliency (offset_mapping unsupported)  tag={tag} seq#{idx} p1={p1:.4f}", fontsize=10)
                fig.tight_layout()
                fig.savefig(str(png_imp), dpi=220)
                plt.close(fig)

        # 附加输出：tokens + attn npz + importance csv
        tok_path = outdir / f"tokens_{file_tag}.txt"
        npz_path = outdir / f"attn_{file_tag}.npz"
        csv_imp  = outdir / f"token_importance_{file_tag}.csv"

        with tok_path.open("w", encoding="utf-8") as f:
            for i, t in enumerate(tokens):
                f.write(f"{i}\t{t}\n")

        np.savez_compressed(npz_path, **{f"layer_{i}": attn_mats[i] for i in range(len(attn_mats))})
        save_token_importance_csv(tokens_vis, imp_vis, str(csv_imp))

        # 可选再存一张 bar（方便快速看 top）
        if ALSO_SAVE_BAR:
            png_bar = outdir / f"attn_importance_bar_{file_tag}_top{BAR_TOPK}.png"
            plot_token_importance_topk(tokens_vis, clip_and_normalize(imp_vis, SAL_CLIP_PCT), str(png_bar), BAR_TOPK,
                                       title=f"Rollout importance BAR (tag={tag}, seq#{idx}, p1={p1:.4f})")
            total_png += 1

        total_png += 3
        print(f"[OK]  {file_tag}")
        print(f"      flow : {png_flow}")
        print(f"      heat : {png_heat}")
        print(f"      imp  : {png_imp}")

    print(f"[SUMMARY] {tag}: OK={ok}, SKIP={skipped}, PNG_generated={total_png} (3 per OK seq)")
    print(f"[SUMMARY] {tag}: outputs dir = {outdir}")


def main():
    out_root = Path(OUTDIR)
    safe_mkdir(out_root)

    device = torch.device(DEVICE)
    print(f"[INFO] device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

    model = DNABiGRUWithDNABERT(
        dnabert_dir=MODEL_DIR,
        num_classes=2,
        hidden_size=HIDDEN_SIZE,
        freeze_dnabert=FREEZE_DNABERT,
        finetune_from_layer=FINETUNE_FROM_LAYER
    ).to(device)

    sd = torch.load(PTH_PATH, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    sd = strip_module_prefix(sd)

    missing, unexpected = model.load_state_dict(sd, strict=STRICT_LOAD)
    if missing or unexpected:
        print("[WARN] load_state_dict mismatch:")
        if missing:
            print("  missing keys   :", missing[:20], "..." if len(missing) > 20 else "")
        if unexpected:
            print("  unexpected keys:", unexpected[:20], "..." if len(unexpected) > 20 else "")
        if STRICT_LOAD:
            raise RuntimeError("STRICT_LOAD=True 且出现 mismatch，已中断。请确认结构/hidden_size/保存方式一致。")

    n_patched = patch_bert_attention_capture(model.dnabert_model)
    print(f"[INFO] Patched attention capture modules: {n_patched}")
    if n_patched == 0:
        print("[WARN] 没找到可patch的attention模块（DNABERT-S实现差异）。将无法生成注意力图。")

    model.eval()

    for in_cfg in INPUTS:
        run_one_input(model, tokenizer, device, in_cfg, out_root)

    print(f"\n[DONE] outputs saved to: {out_root}")


if __name__ == "__main__":
    main()
