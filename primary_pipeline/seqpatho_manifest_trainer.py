#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


SCORE_COL = "score"
SEQ_COL_CANDIDATES = ["clean_seq", "gene_sequence", "sequence", "nucleotide_sequence"]
VAL_SPLIT_NAMES = {"val", "validation", "valid", "dev"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def find_first_col(df: pd.DataFrame, candidates: Sequence[str]) -> str:
    lower = {str(c).lower(): c for c in df.columns}
    for name in candidates:
        if name in df.columns:
            return name
        if name.lower() in lower:
            return lower[name.lower()]
    raise ValueError(f"None of these columns were found: {list(candidates)}")


def normalize_split_name(value: object) -> str:
    s = str(value).strip().lower()
    if s in VAL_SPLIT_NAMES:
        return "val"
    if s in {"train", "training"}:
        return "train"
    if s in {"test", "testing"}:
        return "test"
    return s


def normalize_records(df: pd.DataFrame) -> pd.DataFrame:
    seq_col = find_first_col(df, SEQ_COL_CANDIDATES)
    if "label" not in df.columns:
        raise ValueError("Input records must contain a label column.")

    out = df.copy()
    out["sequence"] = out[seq_col].astype(str).str.upper().str.replace(r"\s+", "", regex=True)
    out["sequence"] = out["sequence"].str.replace(r"[^ACGTN]", "N", regex=True)
    out = out[out["sequence"].str.len() > 0].copy()
    out["label"] = out["label"].astype(int)

    if "record_id" not in out.columns:
        out["record_id"] = [f"record_{i}" for i in range(len(out))]
    out["record_id"] = out["record_id"].astype(str)
    return out


def maybe_sample(df: pd.DataFrame, max_records: Optional[int], seed: int) -> pd.DataFrame:
    if max_records is None or max_records <= 0 or len(df) <= max_records:
        return df.reset_index(drop=True)
    return (
        df.sample(n=int(max_records), random_state=seed)
        .sort_index()
        .reset_index(drop=True)
    )


def load_from_split_dir(split_dir: Path, seed: int, max_train: Optional[int], max_val: Optional[int], max_test: Optional[int]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = normalize_records(pd.read_csv(split_dir / "train.csv", low_memory=False))
    val_name = "validation.csv" if (split_dir / "validation.csv").exists() else "val.csv"
    val = normalize_records(pd.read_csv(split_dir / val_name, low_memory=False))
    test = normalize_records(pd.read_csv(split_dir / "test.csv", low_memory=False))
    return (
        maybe_sample(train, max_train, seed),
        maybe_sample(val, max_val, seed),
        maybe_sample(test, max_test, seed),
    )


def load_from_manifest(
    manifest: Path,
    protocol: Optional[str],
    fold_id: Optional[str],
    seed: int,
    max_train: Optional[int],
    max_val: Optional[int],
    max_test: Optional[int],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(manifest, low_memory=False)
    if protocol and "protocol" in df.columns:
        df = df[df["protocol"].astype(str).eq(str(protocol))].copy()
    if fold_id is not None and "fold_id" in df.columns:
        df = df[df["fold_id"].astype(str).eq(str(fold_id))].copy()
    if "split" not in df.columns:
        raise ValueError("Manifest must contain a split column.")
    df["split_norm"] = df["split"].map(normalize_split_name)

    train = normalize_records(df[df["split_norm"].eq("train")].copy())
    val = normalize_records(df[df["split_norm"].eq("val")].copy())
    test = normalize_records(df[df["split_norm"].eq("test")].copy())
    if len(train) == 0 or len(val) == 0 or len(test) == 0:
        raise ValueError(
            f"Empty split after filtering: train={len(train)}, val={len(val)}, test={len(test)}"
        )
    return (
        maybe_sample(train, max_train, seed),
        maybe_sample(val, max_val, seed),
        maybe_sample(test, max_test, seed),
    )


def random_nt_window(seq: str, nt_window: int) -> str:
    if nt_window <= 0 or len(seq) <= nt_window:
        return seq
    start = random.randint(0, len(seq) - nt_window)
    return seq[start : start + nt_window]


def deterministic_windows(seq: str, nt_window: int, nt_stride: int, max_windows: int) -> List[str]:
    if nt_window <= 0 or len(seq) <= nt_window:
        return [seq]
    stride = max(1, nt_stride)
    windows = [seq[i : i + nt_window] for i in range(0, len(seq) - nt_window + 1, stride)]
    if windows[-1] != seq[-nt_window:]:
        windows.append(seq[-nt_window:])
    if max_windows > 0 and len(windows) > max_windows:
        keep = np.linspace(0, len(windows) - 1, max_windows, dtype=int)
        windows = [windows[i] for i in keep]
    return windows


class SequenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, train: bool, train_nt_window: int):
        self.df = df.reset_index(drop=True)
        self.train = bool(train)
        self.train_nt_window = int(train_nt_window)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        row = self.df.iloc[idx]
        seq = str(row["sequence"])
        if self.train:
            seq = random_nt_window(seq, self.train_nt_window)
        return {
            "record_id": str(row["record_id"]),
            "sequence": seq,
            "label": int(row["label"]),
        }


def collate_records(batch: List[Dict[str, object]]) -> Dict[str, object]:
    return {
        "record_id": [str(x["record_id"]) for x in batch],
        "sequence": [str(x["sequence"]) for x in batch],
        "label": torch.tensor([int(x["label"]) for x in batch], dtype=torch.long),
    }


class MaskedChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(1, channels // int(reduction))
        self.fc1 = nn.Linear(channels, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, channels, bias=False)

    def gate(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.unsqueeze(-1).to(dtype=x.dtype)
        length = m.sum(dim=1).clamp_min(1.0)
        avg = (x * m).sum(dim=1) / length
        x_for_max = x.masked_fill(m.eq(0), -1e4)
        maxv = x_for_max.max(dim=1).values
        alpha = torch.sigmoid(self.gate(avg) + self.gate(maxv)).unsqueeze(1)
        return x * alpha


class StackedMaskedChannelAttention(nn.Module):
    def __init__(self, channels: int, layers: int = 3, reduction: int = 8):
        super().__init__()
        self.blocks = nn.ModuleList([MaskedChannelAttention(channels, reduction) for _ in range(int(layers))])
        self.norms = nn.ModuleList([nn.LayerNorm(channels) for _ in range(int(layers))])

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for block, norm in zip(self.blocks, self.norms):
            x = norm(x + block(x, mask))
        return x


class SeqPathoPlant(nn.Module):
    def __init__(
        self,
        model_dir: str,
        hidden_size: int = 64,
        attention_layers: int = 3,
        reduction: int = 8,
        dropout: float = 0.5,
        top_unfrozen_layers: int = 2,
        freeze_encoder: bool = False,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_dir, trust_remote_code=True)
        enc_hidden = int(getattr(self.encoder.config, "hidden_size", 768))

        n_layers = int(getattr(self.encoder.config, "num_hidden_layers", 12))
        if freeze_encoder:
            top_unfrozen_layers = 0
        first_trainable = max(0, n_layers - int(top_unfrozen_layers))
        for name, param in self.encoder.named_parameters():
            trainable = False
            if "encoder.layer." in name and int(name.split("encoder.layer.")[1].split(".")[0]) >= first_trainable:
                trainable = top_unfrozen_layers > 0
            param.requires_grad = trainable

        if hasattr(self.encoder, "gradient_checkpointing_enable") and top_unfrozen_layers > 0:
            try:
                self.encoder.gradient_checkpointing_enable()
            except Exception:
                pass
        if hasattr(self.encoder.config, "use_cache"):
            self.encoder.config.use_cache = False

        self.bigru = nn.GRU(enc_hidden, hidden_size, batch_first=True, bidirectional=True)
        channels = hidden_size * 2
        self.attn = StackedMaskedChannelAttention(channels, layers=attention_layers, reduction=reduction)
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(channels, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        enc_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        enc = enc_out.last_hidden_state if hasattr(enc_out, "last_hidden_state") else enc_out[0]
        lengths = attention_mask.sum(dim=1).clamp_min(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(enc, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.bigru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=enc.size(1))
        mask = attention_mask[:, : out.size(1)].bool()
        out = self.attn(out, mask)
        out = self.norm(out)
        out = self.dropout(out)
        m = mask.unsqueeze(-1).to(dtype=out.dtype)
        pooled = (out * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        logits = self.fc2(self.dropout(F.relu(self.fc1(pooled))))
        return logits


def build_param_groups(model: SeqPathoPlant, lr_encoder: float, lr_head: float, wd_encoder: float, wd_head: float) -> List[Dict[str, object]]:
    enc_decay, enc_nodecay, head = [], [], []
    for name, param in model.encoder.named_parameters():
        if not param.requires_grad:
            continue
        if any(x in name for x in ["bias", "LayerNorm.weight", "layer_norm.weight"]):
            enc_nodecay.append(param)
        else:
            enc_decay.append(param)
    for module in [model.bigru, model.attn, model.norm, model.fc1, model.fc2]:
        head.extend([p for p in module.parameters() if p.requires_grad])

    groups: List[Dict[str, object]] = []
    if enc_decay:
        groups.append({"params": enc_decay, "lr": lr_encoder, "weight_decay": wd_encoder})
    if enc_nodecay:
        groups.append({"params": enc_nodecay, "lr": lr_encoder, "weight_decay": 0.0})
    if head:
        groups.append({"params": head, "lr": lr_head, "weight_decay": wd_head})
    return groups


def tokenize_batch(tokenizer, sequences: Sequence[str], max_tokens: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    tok = tokenizer(
        list(sequences),
        padding="longest",
        truncation=True,
        max_length=int(max_tokens),
        return_tensors="pt",
    )
    return tok["input_ids"].to(device), tok["attention_mask"].to(device).long()


def compute_metrics(y_true: np.ndarray, score: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (score >= threshold).astype(int)
    out: Dict[str, float] = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
        "Threshold": float(threshold),
        "Support": int(len(y_true)),
        "Positive": int(np.sum(y_true == 1)),
        "Negative": int(np.sum(y_true == 0)),
    }
    if len(set(y_true.tolist())) == 2:
        out["AUROC"] = float(roc_auc_score(y_true, score))
        out["AUPRC"] = float(average_precision_score(y_true, score))
    else:
        out["AUROC"] = float("nan")
        out["AUPRC"] = float("nan")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out.update({"TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)})
    return out


def best_mcc_threshold(y_true: np.ndarray, score: np.ndarray) -> float:
    thresholds = np.unique(np.concatenate(([0.0, 0.5, 1.0], score.astype(float))))
    if len(thresholds) > 5000:
        thresholds = np.linspace(0.0, 1.0, 5001)
    best_t, best_mcc = 0.5, -2.0
    for t in thresholds:
        mcc = matthews_corrcoef(y_true, (score >= t).astype(int))
        if mcc > best_mcc:
            best_mcc = float(mcc)
            best_t = float(t)
    return best_t


@torch.inference_mode()
def evaluate(
    model: SeqPathoPlant,
    tokenizer,
    df: pd.DataFrame,
    device: torch.device,
    max_tokens: int,
    nt_window: int,
    nt_stride: int,
    max_windows: int,
    criterion: nn.Module,
    use_amp: bool,
    window_batch_size: int,
    record_batch_size: int,
) -> Tuple[float, pd.DataFrame]:
    model.eval()
    rows: List[Dict[str, object]] = []
    losses: List[float] = []
    amp_dtype = torch.float16
    record_batch_size = max(1, int(record_batch_size))
    for start_row in tqdm(range(0, len(df), record_batch_size), desc="eval", leave=False):
        sub = df.iloc[start_row : start_row + record_batch_size].reset_index(drop=True)
        all_windows: List[str] = []
        owners: List[int] = []
        n_windows_by_owner = np.zeros(len(sub), dtype=np.int64)

        for owner, row in sub.iterrows():
            windows = deterministic_windows(str(row["sequence"]), nt_window, nt_stride, max_windows)
            n_windows_by_owner[owner] = len(windows)
            all_windows.extend(windows)
            owners.extend([owner] * len(windows))

        if not all_windows:
            continue

        owner_arr = np.asarray(owners, dtype=np.int64)
        score_sums = np.zeros(len(sub), dtype=np.float64)
        logit_sums = np.zeros((len(sub), 2), dtype=np.float64)
        counts = np.zeros(len(sub), dtype=np.float64)

        for off in range(0, len(all_windows), window_batch_size):
            chunk = all_windows[off : off + window_batch_size]
            owner_chunk = owner_arr[off : off + len(chunk)]
            input_ids, attention_mask = tokenize_batch(tokenizer, chunk, max_tokens, device)
            with autocast(enabled=use_amp, dtype=amp_dtype):
                logits = model(input_ids, attention_mask)
            logits_np = logits.float().detach().cpu().numpy()
            scores_np = F.softmax(logits.float(), dim=1)[:, 1].detach().cpu().numpy()
            for local_idx, owner in enumerate(owner_chunk):
                score_sums[owner] += float(scores_np[local_idx])
                logit_sums[owner] += logits_np[local_idx]
                counts[owner] += 1.0

        counts = np.maximum(counts, 1.0)
        mean_scores = score_sums / counts
        mean_logits_np = logit_sums / counts[:, None]
        labels_np = sub["label"].to_numpy(dtype=np.int64)
        with torch.no_grad():
            mean_logits = torch.tensor(mean_logits_np, dtype=torch.float32, device=device)
            labels_t = torch.tensor(labels_np, dtype=torch.long, device=device)
            loss_vec = F.cross_entropy(mean_logits, labels_t, reduction="none", label_smoothing=getattr(criterion, "label_smoothing", 0.0))
            losses.extend([float(x) for x in loss_vec.detach().cpu().numpy().tolist()])

        for owner, row in sub.iterrows():
            rows.append(
                {
                    "record_id": str(row["record_id"]),
                    "label": int(row["label"]),
                    SCORE_COL: float(mean_scores[owner]),
                    "n_windows": int(n_windows_by_owner[owner]),
                }
            )
    return float(np.mean(losses)) if losses else float("nan"), pd.DataFrame(rows)


def train_one_epoch(
    model: SeqPathoPlant,
    tokenizer,
    loader: DataLoader,
    device: torch.device,
    max_tokens: int,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    use_amp: bool,
    max_tokens_per_step: int,
) -> float:
    model.train()
    losses: List[float] = []
    amp_dtype = torch.float16
    for batch in tqdm(loader, desc="train", leave=False):
        labels = batch["label"].to(device)
        input_ids, attention_mask = tokenize_batch(tokenizer, batch["sequence"], max_tokens, device)
        lengths = attention_mask.sum(dim=1)
        sub_bs = max(1, int(max_tokens_per_step // max(1, int(lengths.max().item()))))
        n = input_ids.size(0)
        n_sub = int(math.ceil(n / sub_bs))

        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        for start in range(0, n, sub_bs):
            end = min(start + sub_bs, n)
            with autocast(enabled=use_amp, dtype=amp_dtype):
                logits = model(input_ids[start:end], attention_mask[start:end])
                loss_raw = criterion(logits, labels[start:end])
                loss = loss_raw / n_sub
            scaler.scale(loss).backward()
            running += float(loss_raw.item())
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        losses.append(running / max(1, n_sub))
    return float(np.mean(losses)) if losses else float("nan")


def write_metrics(path: Path, metrics: Dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        for key, value in metrics.items():
            writer.writerow([key, value])


@dataclass
class RunConfig:
    model_dir: str
    epochs: int
    batch_size: int
    max_tokens_train: int
    max_tokens_eval: int
    train_nt_window: int
    eval_nt_window: int
    eval_nt_stride: int
    max_eval_windows: int
    hidden_size: int
    attention_layers: int
    reduction: int
    dropout: float
    top_unfrozen_layers: int
    freeze_encoder: bool
    lr_encoder: float
    lr_head: float
    wd_encoder: float
    wd_head: float
    label_smoothing: float
    patience: int
    seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--split_dir", type=Path, help="Directory containing train.csv, validation.csv, test.csv.")
    source.add_argument("--manifest", type=Path, help="Row-level manifest containing split and sequence columns.")
    parser.add_argument("--protocol", default=None)
    parser.add_argument("--fold_id", default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model_dir", default=str(Path(__file__).resolve().parents[1] / "DNABERT-S"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--test_only", action="store_true")

    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_window_batch_size", type=int, default=32)
    parser.add_argument("--eval_record_batch_size", type=int, default=64)
    parser.add_argument("--max_tokens_train", type=int, default=1000)
    parser.add_argument("--max_tokens_eval", type=int, default=1000)
    parser.add_argument("--train_nt_window", type=int, default=1000)
    parser.add_argument("--eval_nt_window", type=int, default=1000)
    parser.add_argument("--eval_nt_stride", type=int, default=1000)
    parser.add_argument("--max_eval_windows", type=int, default=512)
    parser.add_argument("--max_tokens_per_step", type=int, default=20000)

    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--attention_layers", type=int, default=3)
    parser.add_argument("--reduction", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--top_unfrozen_layers", type=int, default=2)
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--lr_encoder", type=float, default=5e-6)
    parser.add_argument("--lr_head", type=float, default=2e-4)
    parser.add_argument("--wd_encoder", type=float, default=0.01)
    parser.add_argument("--wd_head", type=float, default=2e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max_train_records", type=int, default=0)
    parser.add_argument("--max_val_records", type=int, default=0)
    parser.add_argument("--max_test_records", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    max_train = args.max_train_records or None
    max_val = args.max_val_records or None
    max_test = args.max_test_records or None
    if args.split_dir:
        train_df, val_df, test_df = load_from_split_dir(args.split_dir, args.seed, max_train, max_val, max_test)
    else:
        train_df, val_df, test_df = load_from_manifest(
            args.manifest, args.protocol, args.fold_id, args.seed, max_train, max_val, max_test
        )
    print(f"[data] train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    model = SeqPathoPlant(
        model_dir=args.model_dir,
        hidden_size=args.hidden_size,
        attention_layers=args.attention_layers,
        reduction=args.reduction,
        dropout=args.dropout,
        top_unfrozen_layers=args.top_unfrozen_layers,
        freeze_encoder=args.freeze_encoder,
    ).to(device)

    max_pos = int(getattr(model.encoder.config, "max_position_embeddings", args.max_tokens_train))
    args.max_tokens_train = min(args.max_tokens_train, max_pos)
    args.max_tokens_eval = min(args.max_tokens_eval, max_pos)

    cfg = RunConfig(
        model_dir=str(args.model_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_tokens_train=args.max_tokens_train,
        max_tokens_eval=args.max_tokens_eval,
        train_nt_window=args.train_nt_window,
        eval_nt_window=args.eval_nt_window,
        eval_nt_stride=args.eval_nt_stride,
        max_eval_windows=args.max_eval_windows,
        hidden_size=args.hidden_size,
        attention_layers=args.attention_layers,
        reduction=args.reduction,
        dropout=args.dropout,
        top_unfrozen_layers=args.top_unfrozen_layers,
        freeze_encoder=args.freeze_encoder,
        lr_encoder=args.lr_encoder,
        lr_head=args.lr_head,
        wd_encoder=args.wd_encoder,
        wd_head=args.wd_head,
        label_smoothing=args.label_smoothing,
        patience=args.patience,
        seed=args.seed,
    )
    (args.output_dir / "run_config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"[load] {args.checkpoint}")

    train_loader = DataLoader(
        SequenceDataset(train_df, train=True, train_nt_window=args.train_nt_window),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_records,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    best_path = args.output_dir / "best_model.pt"
    history: List[Dict[str, float]] = []
    if not args.test_only:
        optimizer = torch.optim.AdamW(
            build_param_groups(model, args.lr_encoder, args.lr_head, args.wd_encoder, args.wd_head),
            eps=1e-8,
        )
        total_steps = max(1, args.epochs * len(train_loader))
        scheduler = get_linear_schedule_with_warmup(optimizer, max(1, int(0.05 * total_steps)), total_steps)
        scaler = GradScaler(enabled=use_amp)

        best_val_loss = float("inf")
        bad_epochs = 0
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(
                model,
                tokenizer,
                train_loader,
                device,
                args.max_tokens_train,
                criterion,
                optimizer,
                scheduler,
                scaler,
                use_amp,
                args.max_tokens_per_step,
            )
            val_loss, val_pred = evaluate(
                model,
                tokenizer,
                val_df,
                device,
                args.max_tokens_eval,
                args.eval_nt_window,
                args.eval_nt_stride,
                args.max_eval_windows,
                criterion,
                use_amp,
                args.eval_window_batch_size,
                args.eval_record_batch_size,
            )
            y_val = val_pred["label"].to_numpy(dtype=int)
            s_val = val_pred[SCORE_COL].to_numpy(dtype=float)
            val_t = best_mcc_threshold(y_val, s_val)
            val_metrics = compute_metrics(y_val, s_val, val_t)
            row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **val_metrics}
            history.append(row)
            pd.DataFrame(history).to_csv(args.output_dir / "training_log.csv", index=False)
            val_pred.to_csv(args.output_dir / "validation_predictions_latest.csv", index=False)
            print(
                f"[epoch {epoch}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                f"val_mcc={val_metrics['MCC']:.4f} val_auprc={val_metrics['AUPRC']:.4f} "
                f"threshold={val_t:.4f}"
            )

            if val_loss < best_val_loss - 1e-3:
                best_val_loss = val_loss
                bad_epochs = 0
                torch.save(model.state_dict(), best_path)
                val_pred.to_csv(args.output_dir / "validation_predictions.csv", index=False)
            else:
                bad_epochs += 1
                if bad_epochs >= args.patience:
                    print(f"[early_stop] epoch={epoch}")
                    break

    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))
        print(f"[best] loaded {best_path}")

    val_loss, val_pred = evaluate(
        model,
        tokenizer,
        val_df,
        device,
        args.max_tokens_eval,
        args.eval_nt_window,
        args.eval_nt_stride,
        args.max_eval_windows,
        criterion,
        use_amp,
        args.eval_window_batch_size,
        args.eval_record_batch_size,
    )
    y_val = val_pred["label"].to_numpy(dtype=int)
    s_val = val_pred[SCORE_COL].to_numpy(dtype=float)
    threshold = best_mcc_threshold(y_val, s_val)
    val_metrics = compute_metrics(y_val, s_val, threshold)
    val_metrics["Loss"] = val_loss
    val_pred.to_csv(args.output_dir / "validation_predictions.csv", index=False)
    write_metrics(args.output_dir / "validation_metrics.csv", val_metrics)

    test_loss, test_pred = evaluate(
        model,
        tokenizer,
        test_df,
        device,
        args.max_tokens_eval,
        args.eval_nt_window,
        args.eval_nt_stride,
        args.max_eval_windows,
        criterion,
        use_amp,
        args.eval_window_batch_size,
        args.eval_record_batch_size,
    )
    y_test = test_pred["label"].to_numpy(dtype=int)
    s_test = test_pred[SCORE_COL].to_numpy(dtype=float)
    test_metrics = compute_metrics(y_test, s_test, threshold)
    test_metrics["Loss"] = test_loss
    test_metrics["Validation_MCC_threshold"] = threshold
    test_pred.to_csv(args.output_dir / "test_gene_level_predictions.csv", index=False)
    test_pred.to_csv(args.output_dir / "test_predictions.csv", index=False)
    write_metrics(args.output_dir / "test_metrics.csv", test_metrics)
    (args.output_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    print(json.dumps(test_metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
