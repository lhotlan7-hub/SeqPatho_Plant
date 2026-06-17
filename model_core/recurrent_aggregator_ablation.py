# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    auc,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
    f1_score,
)
from sklearn.model_selection import train_test_split
from torch import optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
import warnings

try:
    from curve_export_helper import apply_runtime_overrides, apply_seed_override
except Exception:
    def apply_runtime_overrides(default_epochs: int) -> Tuple[int, bool]:
        raw_epochs = os.getenv("SEQPATHO_EPOCHS", "").strip()
        eval_only = os.getenv("SEQPATHO_EVAL_ONLY", "0").strip().lower() in {"1", "true", "yes"}
        if raw_epochs:
            try:
                default_epochs = int(raw_epochs)
            except ValueError:
                pass
        return default_epochs, eval_only

    def apply_seed_override(default_seed: int) -> int:
        raw_seed = os.getenv("SEQPATHO_SEED", "").strip()
        if raw_seed:
            try:
                return int(raw_seed)
            except ValueError:
                pass
        return default_seed


warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

VALID_AGGREGATORS = ("lstm", "gru", "bilstm", "bigru")
DEFAULT_MODEL_DIR = r"C:\Users\GQ\.cache\huggingface\modules\transformers_modules\DNABERT-S"
DEFAULT_POS_FILE = r"E:\4PythonProject\SeqPatho\Dataset\patho_plants_only.csv"
DEFAULT_NEG_FILE = r"E:\4PythonProject\SeqPatho\Dataset\Nonpatho_plants_only.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a controlled LSTM/GRU/BiLSTM/BiGRU ablation under one split and evaluation protocol."
    )
    parser.add_argument("--aggregators", nargs="+", default=list(VALID_AGGREGATORS),
                        choices=VALID_AGGREGATORS)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--pos-file", default=DEFAULT_POS_FILE)
    parser.add_argument("--neg-file", default=DEFAULT_NEG_FILE)
    parser.add_argument("--output-root", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-len-train", type=int, default=1000)
    parser.add_argument("--max-len-eval", type=int, default=1000)
    parser.add_argument("--k-mer", type=int, default=6)
    parser.add_argument("--rnn-output-dim", type=int, default=128)
    parser.add_argument("--head-hidden", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr-enc", type=float, default=5e-6)
    parser.add_argument("--lr-head", type=float, default=2e-4)
    parser.add_argument("--wd-enc", type=float, default=0.01)
    parser.add_argument("--wd-head", type=float, default=2e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--delta", type=float, default=0.001)
    parser.add_argument("--max-samples-per-class", type=int, default=0,
                        help="Debug option. Use 0 for the full dataset.")
    parser.add_argument("--freeze-dnabert", action="store_true",
                        help="Freeze all DNABERT parameters to isolate the recurrent aggregator.")
    parser.add_argument("--train-max-tokens-per-step", type=int, default=20000)
    parser.add_argument("--eval-max-tokens-per-chunk", type=int, default=256000)
    parser.add_argument("--eval-max-windows-per-seq", type=int, default=512)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def configure_device() -> torch.device:
    if torch.cuda.is_available():
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_combined_data(
    pos_file: str,
    neg_file: str,
    seed: int,
    max_samples_per_class: int = 0,
) -> Tuple[List[str], np.ndarray]:
    df_pos = pd.read_csv(pos_file, low_memory=False)
    df_neg = pd.read_csv(neg_file, low_memory=False)

    pos_seqs = df_pos.iloc[:, 14].dropna().astype(str)
    neg_seqs = df_neg.iloc[:, 13].dropna().astype(str)

    n = min(len(pos_seqs), len(neg_seqs))
    if max_samples_per_class and max_samples_per_class > 0:
        n = min(n, max_samples_per_class)

    pos_seqs = pos_seqs.sample(n=n, random_state=seed).reset_index(drop=True)
    neg_seqs = neg_seqs.sample(n=n, random_state=seed).reset_index(drop=True)

    sequences = list(pos_seqs) + list(neg_seqs)
    labels = np.array([1] * n + [0] * n, dtype=np.int64)
    return sequences, labels


def split_70_15_15(
    sequences: Sequence[str],
    labels: np.ndarray,
    seed: int,
) -> Tuple[Tuple[List[str], np.ndarray], Tuple[List[str], np.ndarray], Tuple[List[str], np.ndarray]]:
    seq_train, seq_temp, y_train, y_temp = train_test_split(
        list(sequences),
        labels,
        test_size=0.30,
        stratify=labels,
        random_state=seed,
    )
    seq_val, seq_test, y_val, y_test = train_test_split(
        seq_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=seed,
    )
    return (seq_train, y_train), (seq_val, y_val), (seq_test, y_test)


def sequence_hash(seq: str) -> str:
    return hashlib.sha1(str(seq).encode("utf-8", errors="ignore")).hexdigest()


def save_split_manifest(
    output_root: Path,
    train_data: Tuple[List[str], np.ndarray],
    val_data: Tuple[List[str], np.ndarray],
    test_data: Tuple[List[str], np.ndarray],
) -> None:
    rows = []
    for split_name, (seqs, labels) in [
        ("train", train_data),
        ("val", val_data),
        ("test", test_data),
    ]:
        for i, (seq, label) in enumerate(zip(seqs, labels)):
            rows.append({
                "split": split_name,
                "index": i,
                "label": int(label),
                "length": len(str(seq)),
                "sha1": sequence_hash(seq),
            })
    pd.DataFrame(rows).to_csv(output_root / "split_manifest.csv", index=False)


def window_nt(length_tokens: int, k_mer: int) -> int:
    return length_tokens + k_mer - 1


def make_windows(seq: str, length_tokens: int, k_mer: int, stride_tokens: int) -> List[str]:
    seq = str(seq).strip().upper()
    width_nt = window_nt(length_tokens, k_mer)
    if len(seq) <= width_nt:
        return [seq]
    step_nt = stride_tokens + k_mer - 1
    windows = [seq[i:i + width_nt] for i in range(0, len(seq) - width_nt + 1, step_nt)]
    if (len(seq) - width_nt) % step_nt != 0:
        windows.append(seq[-width_nt:])
    return windows


def sample_train_window(seq: str, length_tokens: int, k_mer: int) -> str:
    seq = str(seq).strip().upper()
    width_nt = window_nt(length_tokens, k_mer)
    if len(seq) <= width_nt:
        return seq
    start = random.randint(0, len(seq) - width_nt)
    return seq[start:start + width_nt]


class SeqDataset(Dataset):
    def __init__(self, sequences: Sequence[str], labels: np.ndarray, train: bool,
                 max_len_train: int, k_mer: int):
        self.sequences = list(sequences)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.train = train
        self.max_len_train = max_len_train
        self.k_mer = k_mer

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> Tuple[str, int]:
        seq = self.sequences[index]
        if self.train:
            seq = sample_train_window(seq, self.max_len_train, self.k_mer)
        return seq, int(self.labels[index])


def collate_fn(batch: Sequence[Tuple[str, int]]) -> Tuple[List[str], torch.Tensor]:
    seqs, labels = zip(*batch)
    return list(seqs), torch.tensor(labels, dtype=torch.long)


class EarlyStopping:
    def __init__(self, checkpoint_dir: Path, patience: int, delta: float):
        self.checkpoint_dir = checkpoint_dir
        self.patience = patience
        self.delta = delta
        self.best_loss = float("inf")
        self.best_epoch = 0
        self.counter = 0
        self.early_stop = False
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def step(self, val_loss: float, model: nn.Module, epoch_index: int) -> None:
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.best_epoch = epoch_index
            self.counter = 0
            torch.save(model.state_dict(), self.checkpoint_dir / f"best_model_epoch{epoch_index + 1}.pth")
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction_ratio: int = 8):
        super().__init__()
        hidden = max(1, channels // reduction_ratio)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_perm = x.permute(0, 2, 1)
        avg_out = self.fc(self.avg_pool(x_perm).squeeze(-1))
        max_out = self.fc(self.max_pool(x_perm).squeeze(-1))
        scale = self.sigmoid(avg_out + max_out).unsqueeze(-1)
        return (x_perm * scale).permute(0, 2, 1)


class StackedChannelAttention(nn.Module):
    def __init__(self, channels: int, num_layers: int = 3):
        super().__init__()
        self.layers = nn.ModuleList([ChannelAttention(channels) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(channels) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for attn, norm in zip(self.layers, self.norms):
            residual = x
            x = attn(x)
            x = norm(x + residual)
        return x


class DNABERTRecurrentClassifier(nn.Module):
    def __init__(
        self,
        model_dir: str,
        aggregator: str,
        output_dim: int,
        head_hidden: int,
        dropout: float,
        freeze_dnabert: bool,
        num_classes: int = 2,
    ):
        super().__init__()
        aggregator = aggregator.lower()
        if aggregator not in VALID_AGGREGATORS:
            raise ValueError(f"Unsupported aggregator: {aggregator}")

        self.aggregator = aggregator
        self.dnabert_model = AutoModel.from_pretrained(model_dir, trust_remote_code=True)
        self.bert_hidden_size = int(getattr(self.dnabert_model.config, "hidden_size", 768))
        self.freeze_dnabert = freeze_dnabert

        for name, param in self.dnabert_model.named_parameters():
            if freeze_dnabert:
                param.requires_grad = False
            elif "encoder.layer" in name:
                layer_num = int(name.split("encoder.layer.")[1].split(".")[0])
                param.requires_grad = layer_num >= 10
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

        bidirectional = aggregator.startswith("bi")
        directions = 2 if bidirectional else 1
        if output_dim % directions != 0:
            raise ValueError("--rnn-output-dim must be divisible by 2 for bidirectional aggregators.")
        hidden_size = output_dim // directions
        rnn_cls = nn.LSTM if "lstm" in aggregator else nn.GRU
        self.recurrent = rnn_cls(
            input_size=self.bert_hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.channel_attention = StackedChannelAttention(output_dim, num_layers=3)
        self.layernorm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(output_dim, head_hidden)
        self.fc2 = nn.Linear(head_hidden, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.freeze_dnabert:
            with torch.no_grad():
                bert_output = self.dnabert_model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            bert_output = self.dnabert_model(input_ids=input_ids, attention_mask=attention_mask)

        embeddings = bert_output[0]
        lengths = attention_mask.sum(dim=1).clamp_min(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            embeddings,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.recurrent(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=embeddings.size(1),
        )

        output = self.channel_attention(output)
        output = self.layernorm(output)
        output = self.dropout(output)

        mask = attention_mask[:, :output.size(1)].unsqueeze(-1).float()
        pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-6)
        x = F.relu(self.fc1(pooled))
        x = self.dropout(x)
        return self.fc2(x)


def build_param_groups(model: DNABERTRecurrentClassifier, args: argparse.Namespace) -> List[Dict[str, object]]:
    enc_decay: List[nn.Parameter] = []
    enc_nodecay: List[nn.Parameter] = []
    for name, param in model.dnabert_model.named_parameters():
        if not param.requires_grad:
            continue
        if any(nd in name for nd in ["bias", "LayerNorm.weight", "layer_norm.weight"]):
            enc_nodecay.append(param)
        else:
            enc_decay.append(param)

    head_params: List[nn.Parameter] = []
    for module in [model.recurrent, model.channel_attention, model.layernorm, model.fc1, model.fc2]:
        head_params.extend([p for p in module.parameters() if p.requires_grad])

    groups: List[Dict[str, object]] = []
    if enc_decay:
        groups.append({"params": enc_decay, "lr": args.lr_enc, "weight_decay": args.wd_enc, "group_name": "enc_decay"})
    if enc_nodecay:
        groups.append({"params": enc_nodecay, "lr": args.lr_enc, "weight_decay": 0.0, "group_name": "enc_nodecay"})
    groups.append({"params": head_params, "lr": args.lr_head, "weight_decay": args.wd_head, "group_name": "head"})
    return groups


def valid_token_mask(input_ids: torch.Tensor, attention_mask: torch.Tensor, vocab_size: int) -> torch.Tensor:
    return (
        (attention_mask.sum(dim=1) > 0)
        & (input_ids.min(dim=1).values >= 0)
        & (input_ids.max(dim=1).values < vocab_size)
    )


def compute_metrics(labels: np.ndarray, preds: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    if len(labels) == 0:
        return {k: float("nan") for k in ["accuracy", "precision", "recall", "f1", "mcc", "auroc", "auprc", "ap"]}

    out = {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "mcc": matthews_corrcoef(labels, preds),
    }
    try:
        fpr, tpr, _ = roc_curve(labels, probs)
        out["auroc"] = auc(fpr, tpr)
    except Exception:
        out["auroc"] = float("nan")
    try:
        precision_vals, recall_vals, _ = precision_recall_curve(labels, probs)
        out["auprc"] = auc(recall_vals, precision_vals)
        out["ap"] = average_precision_score(labels, probs)
    except Exception:
        out["auprc"] = float("nan")
        out["ap"] = float("nan")
    return out


@torch.inference_mode()
def evaluate_sequence_loader(
    model: DNABERTRecurrentClassifier,
    tokenizer: AutoTokenizer,
    data_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    use_amp: bool,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    labels_all: List[int] = []
    preds_all: List[int] = []
    probs_all: List[float] = []
    loss_sum = 0.0
    n_sequences = 0
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    windows_per_chunk = max(1, args.eval_max_tokens_per_chunk // args.max_len_eval)
    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
    vocab_size = int(model.dnabert_model.config.vocab_size)

    for batch_sequences, batch_labels in data_loader:
        for seq, label in zip(batch_sequences, batch_labels.tolist()):
            windows = make_windows(seq, args.max_len_eval, args.k_mer, args.max_len_eval)
            if args.eval_max_windows_per_seq and len(windows) > args.eval_max_windows_per_seq:
                idx = np.linspace(0, len(windows) - 1, args.eval_max_windows_per_seq, dtype=int)
                windows = [windows[i] for i in idx]
            if not windows:
                continue

            logits_sum = torch.zeros(2, device=device, dtype=torch.float32)
            n_windows = 0
            for start in range(0, len(windows), windows_per_chunk):
                chunk = windows[start:start + windows_per_chunk]
                encoded = tokenizer(
                    chunk,
                    padding="longest",
                    truncation=True,
                    max_length=args.max_len_eval,
                    return_tensors="pt",
                )
                input_ids = encoded["input_ids"].to(device, non_blocking=True)
                attention_mask = encoded["attention_mask"].to(device, non_blocking=True).long()
                mask = valid_token_mask(input_ids, attention_mask, vocab_size)
                if mask.sum() == 0:
                    continue

                input_ids = input_ids[mask]
                attention_mask = attention_mask[mask]
                with autocast(enabled=use_amp, dtype=amp_dtype):
                    logits = model(input_ids, attention_mask)
                logits_sum += logits.float().sum(dim=0)
                n_windows += logits.shape[0]

            if n_windows == 0:
                continue

            agg_logits = logits_sum / n_windows
            y = torch.tensor([label], device=device)
            loss_sum += criterion(agg_logits.unsqueeze(0), y).item()
            n_sequences += 1

            prob = F.softmax(agg_logits, dim=0)[1].item()
            pred = int(torch.argmax(agg_logits).item())
            labels_all.append(int(label))
            probs_all.append(float(prob))
            preds_all.append(pred)

    return (
        loss_sum / max(1, n_sequences),
        np.asarray(labels_all, dtype=np.int64),
        np.asarray(preds_all, dtype=np.int64),
        np.asarray(probs_all, dtype=np.float32),
    )


def run_one_aggregator(
    aggregator: str,
    train_data: Tuple[List[str], np.ndarray],
    val_data: Tuple[List[str], np.ndarray],
    test_data: Tuple[List[str], np.ndarray],
    tokenizer: AutoTokenizer,
    args: argparse.Namespace,
    output_root: Path,
    device: torch.device,
) -> Dict[str, object]:
    seed_everything(args.seed)
    run_dir = output_root / aggregator
    checkpoint_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)

    model = DNABERTRecurrentClassifier(
        model_dir=args.model_dir,
        aggregator=aggregator,
        output_dim=args.rnn_output_dim,
        head_hidden=args.head_hidden,
        dropout=args.dropout,
        freeze_dnabert=args.freeze_dnabert,
    ).to(device)

    max_pos = int(getattr(model.dnabert_model.config, "max_position_embeddings", args.max_len_eval))
    if args.max_len_train > max_pos or args.max_len_eval > max_pos:
        old_train, old_eval = args.max_len_train, args.max_len_eval
        args.max_len_train = min(args.max_len_train, max_pos)
        args.max_len_eval = min(args.max_len_eval, max_pos)
        print(
            f"[WARN] max_position_embeddings={max_pos}; "
            f"max_len_train {old_train}->{args.max_len_train}, "
            f"max_len_eval {old_eval}->{args.max_len_eval}"
        )

    train_sequences, y_train = train_data
    val_sequences, y_val = val_data
    test_sequences, y_test = test_data

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        SeqDataset(train_sequences, y_train, train=True, max_len_train=args.max_len_train, k_mer=args.k_mer),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        num_workers=0,
    )
    val_loader = DataLoader(
        SeqDataset(val_sequences, y_val, train=False, max_len_train=args.max_len_train, k_mer=args.k_mer),
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        num_workers=0,
    )
    test_loader = DataLoader(
        SeqDataset(test_sequences, y_test, train=False, max_len_train=args.max_len_train, k_mer=args.k_mer),
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        num_workers=0,
    )

    optimizer = optim.AdamW(build_param_groups(model, args), eps=1e-8)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = GradScaler(enabled=torch.cuda.is_available())
    use_amp = torch.cuda.is_available()
    num_train_steps = max(1, args.epochs * len(train_loader))
    num_warmup_steps = max(1, int(0.05 * num_train_steps))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_train_steps)
    stopper = EarlyStopping(checkpoint_dir, patience=args.patience, delta=args.delta)

    history_rows: List[Dict[str, float]] = []
    vocab_size = int(model.dnabert_model.config.vocab_size)
    amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16

    for epoch in range(args.epochs):
        model.train()
        train_loss_sum = 0.0
        seen_batches = 0
        progress = tqdm(train_loader, desc=f"{aggregator} epoch {epoch + 1}/{args.epochs}", leave=False)

        for batch_sequences, batch_labels in progress:
            encoded = tokenizer(
                batch_sequences,
                padding="longest",
                truncation=True,
                max_length=args.max_len_train,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device, non_blocking=True)
            attention_mask = encoded["attention_mask"].to(device, non_blocking=True).long()
            batch_labels = batch_labels.to(device, non_blocking=True)

            mask = valid_token_mask(input_ids, attention_mask, vocab_size)
            if mask.sum() == 0:
                continue
            input_ids = input_ids[mask]
            attention_mask = attention_mask[mask]
            batch_labels = batch_labels[mask]

            lengths = attention_mask.sum(dim=1)
            n_items = input_ids.size(0)
            max_len = int(lengths.max().item())
            sub_batch = max(1, int(args.train_max_tokens_per_step // max(max_len, 1)))
            num_sub_batches = math.ceil(n_items / sub_batch)

            optimizer.zero_grad(set_to_none=True)
            loss_accum = 0.0
            for start in range(0, n_items, sub_batch):
                end = min(start + sub_batch, n_items)
                with autocast(enabled=use_amp, dtype=amp_dtype):
                    logits = model(input_ids[start:end], attention_mask[start:end])
                    loss_sub = criterion(logits, batch_labels[start:end])
                    loss = loss_sub / num_sub_batches
                scaler.scale(loss).backward()
                loss_accum += loss_sub.item()

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss_sum += loss_accum / max(1, num_sub_batches)
            seen_batches += 1
            progress.set_postfix(loss=train_loss_sum / max(1, seen_batches))

        train_loss = train_loss_sum / max(1, seen_batches)
        val_loss, val_labels, val_preds, val_probs = evaluate_sequence_loader(
            model, tokenizer, val_loader, args, device, use_amp
        )
        val_metrics = compute_metrics(val_labels, val_preds, val_probs)
        history_row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history_rows.append(history_row)
        pd.DataFrame(history_rows).to_csv(run_dir / "history.csv", index=False)

        print(
            f"[{aggregator}] epoch={epoch + 1} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"acc={val_metrics['accuracy']:.4f} f1={val_metrics['f1']:.4f} "
            f"mcc={val_metrics['mcc']:.4f} auroc={val_metrics['auroc']:.4f} auprc={val_metrics['auprc']:.4f}"
        )

        stopper.step(val_loss, model, epoch)
        if stopper.early_stop:
            break

    best_path = checkpoint_dir / f"best_model_epoch{stopper.best_epoch + 1}.pth"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))

    test_loss, test_labels, test_preds, test_probs = evaluate_sequence_loader(
        model, tokenizer, test_loader, args, device, use_amp
    )
    test_metrics = compute_metrics(test_labels, test_preds, test_probs)
    final_row: Dict[str, object] = {
        "aggregator": aggregator,
        "best_epoch": stopper.best_epoch + 1,
        "test_loss": test_loss,
        **test_metrics,
    }

    with (run_dir / "final_metrics_test.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Test Loss", test_loss])
        writer.writerow(["Accuracy", test_metrics["accuracy"]])
        writer.writerow(["Precision", test_metrics["precision"]])
        writer.writerow(["Recall", test_metrics["recall"]])
        writer.writerow(["F1-score", test_metrics["f1"]])
        writer.writerow(["MCC", test_metrics["mcc"]])
        writer.writerow(["AUROC", test_metrics["auroc"]])
        writer.writerow(["AUPRC", test_metrics["auprc"]])
        writer.writerow(["Average Precision (AP)", test_metrics["ap"]])
        writer.writerow(["Best Epoch (val)", stopper.best_epoch + 1])

    with (run_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return final_row


def write_latex_table(summary_df: pd.DataFrame, output_path: Path) -> None:
    label_map = {"lstm": "LSTM", "gru": "GRU", "bilstm": "BiLSTM", "bigru": "BiGRU"}
    lines = [
        r"\begin{table}[h]",
        r"	\centering",
        r"	\caption{Performance of recurrent aggregators in long-distance feature capture.}\label{tab:long_distance_recurrent_ablation}%",
        r"	\begin{tabular}{@{}llllllll@{}}",
        r"		\toprule",
        r"		Models & Accuracy & Precision & Recall & F1 & MCC & AUROC & AUPRC \\",
        r"		\midrule",
    ]
    for aggregator in VALID_AGGREGATORS:
        row = summary_df[summary_df["aggregator"] == aggregator]
        if row.empty:
            continue
        r0 = row.iloc[0]
        lines.append(
            "		"
            + f"{label_map[aggregator]} &"
            + f"{float(r0['accuracy']):.4f} &"
            + f"{float(r0['precision']):.4f} &"
            + f"{float(r0['recall']):.4f} &"
            + f"{float(r0['f1']):.4f} &"
            + f"{float(r0['mcc']):.4f} &"
            + f"{float(r0['auroc']):.4f} &"
            + f"{float(r0['auprc']):.4f} \\\\"
        )
    lines.extend([
        r"		\bottomrule",
        r"	\end{tabular}",
        r"\end{table}",
        "",
    ])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.seed = apply_seed_override(args.seed)
    if args.epochs is None:
        args.epochs, eval_only = apply_runtime_overrides(150)
        if eval_only:
            raise ValueError("SEQPATHO_EVAL_ONLY is not supported for this ablation script without per-aggregator checkpoints.")

    seed_everything(args.seed)
    device = configure_device()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root) if args.output_root else Path(f"results_recurrent_aggregator_ablation_{timestamp}")
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Output root: {output_root}")
    print(f"Aggregators: {', '.join(args.aggregators)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    sequences, labels = load_combined_data(args.pos_file, args.neg_file, args.seed, args.max_samples_per_class)
    train_data, val_data, test_data = split_70_15_15(sequences, labels, args.seed)
    save_split_manifest(output_root, train_data, val_data, test_data)

    print(
        f"Split: train={len(train_data[0])} val={len(val_data[0])} test={len(test_data[0])}; "
        f"positive rates={np.mean(train_data[1]):.3f}/{np.mean(val_data[1]):.3f}/{np.mean(test_data[1]):.3f}"
    )

    results: List[Dict[str, object]] = []
    for aggregator in args.aggregators:
        results.append(run_one_aggregator(aggregator, train_data, val_data, test_data, tokenizer, args, output_root, device))
        summary_df = pd.DataFrame(results)
        summary_df.to_csv(output_root / "recurrent_ablation_summary.csv", index=False)
        write_latex_table(summary_df, output_root / "recurrent_ablation_table.tex")

    print(f"Summary: {output_root / 'recurrent_ablation_summary.csv'}")
    print(f"LaTeX: {output_root / 'recurrent_ablation_table.tex'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
