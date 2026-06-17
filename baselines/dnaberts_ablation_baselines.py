# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import gc
import math
import os
import time
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.exceptions import UndefinedMetricWarning
from torch import optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from baseline_common import (
    DEFAULT_DNABERTS_DIR,
    DEFAULT_NEG_FILE,
    DEFAULT_POS_FILE,
    DEFAULT_TAXONOMY_MAP_FILE,
    OOD_VAL_FRAC_FROM_REST,
    SEED,
    build_unit_manifest,
    compute_classification_metrics,
    ensure_dir,
    filter_species_candidates,
    make_result_dir,
    make_windows,
    plot_confusion_matrix,
    plot_roc_pr,
    prepare_dataset,
    prediction_frame,
    sample_train_window,
    save_df,
    save_json,
    save_split_files,
    set_seed,
    split_holdout_unit,
    split_three_way_by_unit,
)


warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=UserWarning)


NUM_CLASSES = 2


def set_torch_seed(seed: int = SEED) -> None:
    set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def safe_float(x, default=np.nan):
    try:
        if x is None:
            return default
        if hasattr(x, "item"):
            return float(x.item())
        return float(x)
    except Exception:
        return default


class SeqDataset(Dataset):
    def __init__(self, df: pd.DataFrame, max_len_tokens: int, k_mer: int):
        self.df = df.reset_index(drop=True).copy()
        self.max_len_tokens = max_len_tokens
        self.k_mer = k_mer

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        seq = sample_train_window(row["seq"], l_tokens=self.max_len_tokens, k=self.k_mer)
        return seq, int(row["label"])


def collate_batch(batch):
    seqs, labels = zip(*batch)
    return list(seqs), torch.tensor(labels, dtype=torch.long)


class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, in_channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, in_channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_perm = x.transpose(1, 2)
        avg = self.fc(self.avg_pool(x_perm))
        maxv = self.fc(self.max_pool(x_perm))
        scale = self.sigmoid(avg + maxv).transpose(1, 2)
        return x * scale


class StackedChannelAttention(nn.Module):
    def __init__(self, in_channels: int, num_layers: int = 3):
        super().__init__()
        self.layers = nn.ModuleList([ChannelAttention(in_channels) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class FrozenEncoderMLP(nn.Module):
    def __init__(self, model_dir: str, hidden_dropout: float = 0.3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_dir, trust_remote_code=True)
        self.hidden_size = int(getattr(self.encoder.config, "hidden_size", 768))
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.mlp = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(hidden_dropout),
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(hidden_dropout),
            nn.Linear(256, NUM_CLASSES),
        )

    def forward(self, input_ids, attention_mask=None):
        with torch.no_grad():
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = out[0]
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-6)
        return self.mlp(pooled)


class DNABERTBiGRU(nn.Module):
    def __init__(
        self,
        model_dir: str,
        random_init: bool = False,
        hidden_size: int = 64,
        unfreeze_last_n: int = 2,
        random_encoder_trainable: str = "all",
        dropout: float = 0.5,
    ):
        super().__init__()
        if random_init:
            config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
            self.encoder = AutoModel.from_config(config, trust_remote_code=True)
        else:
            self.encoder = AutoModel.from_pretrained(model_dir, trust_remote_code=True)
        self.hidden_size = int(getattr(self.encoder.config, "hidden_size", 768))
        self.random_init = bool(random_init)

        if hasattr(self.encoder, "gradient_checkpointing_enable"):
            try:
                self.encoder.gradient_checkpointing_enable()
            except Exception:
                pass
        if hasattr(self.encoder.config, "use_cache"):
            try:
                self.encoder.config.use_cache = False
            except Exception:
                pass

        self._set_encoder_trainability(unfreeze_last_n, random_encoder_trainable)
        self.bigru = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.channel_attention = StackedChannelAttention(in_channels=hidden_size * 2, num_layers=3)
        self.layernorm = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(hidden_size * 2, 32)
        self.fc2 = nn.Linear(32, NUM_CLASSES)

    def _set_encoder_trainability(self, unfreeze_last_n: int, random_encoder_trainable: str) -> None:
        num_layers = int(getattr(self.encoder.config, "num_hidden_layers", 12))
        cutoff = max(0, num_layers - int(unfreeze_last_n))
        if self.random_init and random_encoder_trainable == "all":
            for param in self.encoder.parameters():
                param.requires_grad = True
            return
        for name, param in self.encoder.named_parameters():
            if "encoder.layer" in name:
                try:
                    layer_num = int(name.split("encoder.layer.")[1].split(".")[0])
                    param.requires_grad = layer_num >= cutoff
                except Exception:
                    param.requires_grad = False
            else:
                param.requires_grad = False

    def forward(self, input_ids, attention_mask=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = out[0]
        lengths = attention_mask.sum(dim=1) if attention_mask is not None else (input_ids != 0).sum(dim=1)
        lengths = lengths.clamp_min(1)
        packed = nn.utils.rnn.pack_padded_sequence(embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.bigru(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True, total_length=embeddings.size(1))
        output = self.channel_attention(output)
        output = self.layernorm(output)
        output = self.dropout(output)

        seq_len = output.size(1)
        mask = attention_mask[:, :seq_len].unsqueeze(-1).expand(-1, -1, output.size(2)).float()
        pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-6)
        x = F.relu(self.fc1(pooled))
        x = self.dropout(x)
        return self.fc2(x)


class EarlyStopping:
    def __init__(self, checkpoint_path: Path, patience: int = 6, delta: float = 0.001):
        self.checkpoint_path = checkpoint_path
        self.patience = patience
        self.delta = delta
        self.best_loss = float("inf")
        self.counter = 0
        self.best_epoch = 0
        self.early_stop = False

    def step(self, val_loss: float, model: nn.Module, epoch: int) -> None:
        if val_loss < self.best_loss - self.delta:
            old = self.best_loss
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch
            torch.save(model.state_dict(), self.checkpoint_path)
            print(f"[EarlyStopping] val_loss improved {old:.6f} -> {val_loss:.6f}")
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def build_model(baseline: str, args: argparse.Namespace, device: torch.device) -> nn.Module:
    model_dir = str(args.model_dir)
    if baseline == "frozen_mlp":
        model = FrozenEncoderMLP(model_dir=model_dir, hidden_dropout=args.dropout)
    elif baseline == "partial_bigru":
        model = DNABERTBiGRU(
            model_dir=model_dir,
            random_init=False,
            hidden_size=args.hidden_size,
            unfreeze_last_n=args.unfreeze_last_n,
            random_encoder_trainable=args.random_encoder_trainable,
            dropout=args.dropout,
        )
    elif baseline == "random_init_bigru":
        model = DNABERTBiGRU(
            model_dir=model_dir,
            random_init=True,
            hidden_size=args.hidden_size,
            unfreeze_last_n=args.unfreeze_last_n,
            random_encoder_trainable=args.random_encoder_trainable,
            dropout=args.dropout,
        )
    else:
        raise ValueError(f"unknown DNABERT-S baseline: {baseline}")
    return model.to(device)


def build_param_groups(model: nn.Module, args: argparse.Namespace):
    enc_decay, enc_nodecay, head = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("encoder."):
            if any(nd in name for nd in ["bias", "LayerNorm.weight", "layer_norm.weight"]):
                enc_nodecay.append(param)
            else:
                enc_decay.append(param)
        else:
            head.append(param)
    groups = []
    if enc_decay:
        groups.append({"params": enc_decay, "lr": args.lr_encoder, "weight_decay": args.weight_decay_encoder})
    if enc_nodecay:
        groups.append({"params": enc_nodecay, "lr": args.lr_encoder, "weight_decay": 0.0})
    if head:
        groups.append({"params": head, "lr": args.lr_head, "weight_decay": args.weight_decay_head})
    return groups


def tokenize(tokenizer, seqs: List[str], max_len: int, device: torch.device):
    tok = tokenizer(seqs, padding="longest", truncation=True, max_length=max_len, return_tensors="pt")
    return tok["input_ids"].to(device, non_blocking=True), tok["attention_mask"].to(device, non_blocking=True).long()


def valid_token_mask(input_ids: torch.Tensor, attention_mask: torch.Tensor, vocab_size: int) -> torch.Tensor:
    return (
        (attention_mask.sum(dim=1) > 0)
        & (input_ids.min(dim=1).values >= 0)
        & (input_ids.max(dim=1).values < vocab_size)
    )


def evaluate_dataframe(
    model: nn.Module,
    tokenizer,
    df: pd.DataFrame,
    args: argparse.Namespace,
    device: torch.device,
    criterion: nn.Module,
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    labels, preds, probs = [], [], []
    loss_sum = 0.0
    n_count = 0
    with torch.no_grad():
        for _, row in tqdm(df.iterrows(), total=len(df), desc="eval", leave=False):
            y = int(row["label"])
            windows = make_windows(
                row["seq"],
                l_tokens=args.max_len_eval,
                k=args.k_mer,
                stride_tokens=args.eval_stride_tokens,
            )[: args.eval_max_windows_per_seq]
            if not windows:
                continue
            logits_sum = torch.zeros(NUM_CLASSES, device=device, dtype=torch.float32)
            n_valid = 0
            max_chunk = max(1, args.eval_max_tokens_per_chunk // max(1, args.max_len_eval))
            for start in range(0, len(windows), max_chunk):
                chunk = windows[start : start + max_chunk]
                input_ids, attention_mask = tokenize(tokenizer, chunk, args.max_len_eval, device)
                vocab_size = int(getattr(model.encoder.config, "vocab_size", 4096))
                keep = valid_token_mask(input_ids, attention_mask, vocab_size)
                if keep.sum() == 0:
                    continue
                input_ids = input_ids[keep]
                attention_mask = attention_mask[keep]
                with autocast(enabled=args.use_amp):
                    logits = model(input_ids, attention_mask)
                logits_sum += logits.float().sum(dim=0)
                n_valid += int(logits.size(0))
            if n_valid == 0:
                continue
            avg_logits = logits_sum / max(1, n_valid)
            loss = criterion(avg_logits.unsqueeze(0), torch.tensor([y], device=device))
            prob = torch.softmax(avg_logits, dim=-1)[1].item()
            pred = int(prob >= 0.5)
            labels.append(y)
            preds.append(pred)
            probs.append(prob)
            loss_sum += float(loss.item())
            n_count += 1
    avg_loss = loss_sum / max(1, n_count)
    return avg_loss, np.asarray(labels), np.asarray(preds), np.asarray(probs)


def plot_history(history: pd.DataFrame, out_dir: Path, prefix: str) -> None:
    if len(history) == 0:
        return
    for metric in ["loss", "accuracy", "auprc", "auroc", "f1", "mcc"]:
        plt.figure(figsize=(6, 4))
        if metric == "loss":
            plt.plot(history["epoch"], history["train_loss"], label="Train Loss")
            plt.plot(history["epoch"], history["val_loss"], label="Val Loss")
            plt.ylabel("Loss")
        else:
            if metric not in history.columns:
                plt.close()
                continue
            plt.plot(history["epoch"], history[metric], label=f"Val {metric.upper()}")
            plt.ylabel(metric.upper())
        plt.xlabel("Epoch")
        plt.title(f"{prefix}: {metric}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{metric}_curve.png", dpi=300)
        plt.close()


def train_one_experiment(
    baseline: str,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    out_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
    tokenizer,
) -> Dict:
    ensure_dir(out_dir)
    checkpoint_path = out_dir / "best_model.pt"
    train_loader = DataLoader(
        SeqDataset(df_train, args.max_len_train, args.k_mer),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_batch,
        pin_memory=torch.cuda.is_available(),
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    model = build_model(baseline, args, device)
    trainable_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    total_params = int(sum(p.numel() for p in model.parameters()))
    save_json(
        {
            "baseline": baseline,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "trainable_fraction": trainable_params / max(1, total_params),
        },
        out_dir / "model_size.json",
    )

    optimizer = optim.AdamW(build_param_groups(model, args), eps=1e-8)
    scaler = GradScaler(enabled=args.use_amp)
    steps_per_epoch = max(1, len(train_loader))
    total_steps = max(1, steps_per_epoch * args.epochs)
    warmup_steps = max(1, int(args.warmup_ratio * total_steps))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    stopper = EarlyStopping(checkpoint_path, patience=args.patience, delta=args.early_stop_delta)
    history_rows = []
    start_time = time.time()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        seen = 0
        pbar = tqdm(train_loader, desc=f"{baseline} epoch {epoch + 1}/{args.epochs}", leave=False)
        for seqs, labels in pbar:
            labels = labels.to(device)
            input_ids, attention_mask = tokenize(tokenizer, seqs, args.max_len_train, device)
            vocab_size = int(getattr(model.encoder.config, "vocab_size", 4096))
            keep = valid_token_mask(input_ids, attention_mask, vocab_size)
            if keep.sum() == 0:
                continue
            input_ids = input_ids[keep]
            attention_mask = attention_mask[keep]
            labels = labels[keep]

            lengths = attention_mask.sum(dim=1)
            n = input_ids.size(0)
            lmax = int(lengths.max().item())
            sub_bs = max(1, int(args.train_max_tokens_per_step // max(lmax, 1)))
            num_sub = math.ceil(n / sub_bs)

            optimizer.zero_grad(set_to_none=True)
            loss_accum = 0.0
            for start in range(0, n, sub_bs):
                end = min(start + sub_bs, n)
                with autocast(enabled=args.use_amp):
                    logits = model(input_ids[start:end], attention_mask[start:end])
                    loss_sub = criterion(logits, labels[start:end])
                    loss = loss_sub / max(1, num_sub)
                scaler.scale(loss).backward()
                loss_accum += float(loss_sub.item())

            scale_before = scaler.get_scale() if args.use_amp else None
            scaler.step(optimizer)
            scaler.update()
            scale_after = scaler.get_scale() if args.use_amp else None
            if (not args.use_amp) or scale_after is None or scale_before is None or scale_after >= scale_before:
                scheduler.step()

            total_loss += loss_accum / max(1, num_sub)
            seen += 1
            pbar.set_postfix(loss=total_loss / max(1, seen))

        train_loss = total_loss / max(1, seen)
        val_loss, val_labels, val_preds, val_probs = evaluate_dataframe(
            model, tokenizer, df_val, args, device, criterion
        )
        val_metrics = compute_classification_metrics(val_labels, val_preds, val_probs)
        row = {
            "epoch": epoch + 1,
            "train_loss": safe_float(train_loss),
            "val_loss": safe_float(val_loss),
            **{k: safe_float(v) for k, v in val_metrics.items()},
        }
        history_rows.append(row)
        print(
            f"[{baseline}] epoch={epoch + 1} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_acc={val_metrics.get('accuracy', np.nan):.4f} val_auprc={val_metrics.get('auprc', np.nan):.4f} "
            f"val_mcc={val_metrics.get('mcc', np.nan):.4f}"
        )
        stopper.step(val_loss, model, epoch + 1)
        if stopper.early_stop:
            print(f"[{baseline}] early stop at epoch {epoch + 1}; best epoch={stopper.best_epoch}")
            break

    history = pd.DataFrame(history_rows)
    save_df(history, out_dir / "history.csv", out_dir / "history.md")
    plot_history(history, out_dir, baseline)

    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    test_loss, test_labels, test_preds, test_probs = evaluate_dataframe(model, tokenizer, df_test, args, device, criterion)
    test_metrics = compute_classification_metrics(test_labels, test_preds, test_probs)
    save_json(
        {
            "baseline": baseline,
            "test_loss": test_loss,
            "best_epoch": int(stopper.best_epoch),
            "early_stop": bool(stopper.early_stop),
            **test_metrics,
        },
        out_dir / "final_test_metrics.json",
    )
    save_df(prediction_frame(df_test, test_probs, test_preds), out_dir / "predictions_test.csv")
    plot_confusion_matrix(test_labels, test_preds, out_dir / "confusion_matrix_test.png", title=f"{baseline} | Test")
    plot_roc_pr(test_labels, test_probs, out_dir / "roc_curve_test.png", out_dir / "pr_curve_test.png", title_prefix=baseline)

    row = {
        "baseline": baseline,
        "best_epoch": int(stopper.best_epoch),
        "early_stop": bool(stopper.early_stop),
        "runtime_sec": float(time.time() - start_time),
        "train_n": int(len(df_train)),
        "val_n": int(len(df_val)),
        "test_n": int(len(df_test)),
        "total_params": total_params,
        "trainable_params": trainable_params,
    }
    row.update(test_metrics)
    if len(history) > 0:
        last_val = history.iloc[-1].to_dict()
        for key, value in last_val.items():
            if key != "epoch":
                row[f"last_val_{key}"] = value

    del model, optimizer, scheduler, scaler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row


def aggregate_summary(df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["accuracy", "precision", "recall", "f1", "mcc", "auroc", "auprc", "ap"]
    rows = []
    if len(df) == 0:
        return pd.DataFrame()
    for (benchmark, baseline), sub in df.groupby(["benchmark", "baseline"]):
        row = {"benchmark": benchmark, "baseline": baseline, "n_runs": int(len(sub))}
        for metric in metrics:
            row[f"{metric}_mean"] = float(sub[metric].mean())
            row[f"{metric}_std"] = float(sub[metric].std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["benchmark", "auprc_mean"], ascending=[True, False])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DNABERT-S ablation baselines.")
    parser.add_argument("--pos-file", default=str(DEFAULT_POS_FILE))
    parser.add_argument("--neg-file", default=str(DEFAULT_NEG_FILE))
    parser.add_argument("--taxonomy-map-file", default=str(DEFAULT_TAXONOMY_MAP_FILE))
    parser.add_argument("--model-dir", default=str(DEFAULT_DNABERTS_DIR))
    parser.add_argument("--output-root", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--top-k-species", type=int, default=10)
    parser.add_argument("--no-topk", action="store_true")
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=["frozen_mlp", "partial_bigru", "random_init_bigru"],
        choices=["frozen_mlp", "partial_bigru", "random_init_bigru"],
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--early-stop-delta", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-len-train", type=int, default=500)
    parser.add_argument("--max-len-eval", type=int, default=500)
    parser.add_argument("--eval-stride-tokens", type=int, default=500)
    parser.add_argument("--eval-max-tokens-per-chunk", type=int, default=128000)
    parser.add_argument("--eval-max-windows-per-seq", type=int, default=512)
    parser.add_argument("--train-max-tokens-per-step", type=int, default=20000)
    parser.add_argument("--k-mer", type=int, default=6)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--unfreeze-last-n", type=int, default=2)
    parser.add_argument("--random-encoder-trainable", choices=["all", "partial"], default="all")
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--lr-encoder", type=float, default=5e-6)
    parser.add_argument("--lr-head", type=float, default=2e-4)
    parser.add_argument("--weight-decay-encoder", type=float, default=0.01)
    parser.add_argument("--weight-decay-head", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--run-loso", action="store_true", default=False)
    parser.add_argument("--loso-top-k", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_torch_seed(SEED)
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    device = torch.device(args.device)
    args.use_amp = (not args.no_amp) and device.type == "cuda"
    if device.type == "cuda":
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    root_dir = make_result_dir(args.output_root, "dnaberts_baselines")
    print(f"[Output] {root_dir}")
    print(f"[Device] {device} amp={args.use_amp}")
    config = vars(args).copy()
    config["seed"] = SEED
    config["device"] = str(device)
    save_json(config, root_dir / "config.json")

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), trust_remote_code=True)
    df_all = prepare_dataset(
        root_dir,
        pos_file=args.pos_file,
        neg_file=args.neg_file,
        taxonomy_map_file=args.taxonomy_map_file,
        limit_to_topk_species=not args.no_topk,
        top_k_species=args.top_k_species,
        require_both_labels=True,
    )

    rows = []
    split_result = split_three_way_by_unit(df_all, unit_col="species_key", random_state=SEED)
    df_train, df_val, df_test = split_result[:3]
    standard_root = ensure_dir(root_dir / "standard_species_disjoint")
    save_split_files(standard_root, df_train, df_val, df_test)
    for baseline in args.baselines:
        print(f"[Run] standard_species_disjoint / {baseline}")
        out_dir = ensure_dir(standard_root / baseline)
        row = train_one_experiment(baseline, df_train, df_val, df_test, out_dir, args, device, tokenizer)
        row["benchmark"] = "standard_species_disjoint"
        rows.append(row)

    if args.run_loso:
        loso_root = ensure_dir(root_dir / "loso_species")
        manifest = build_unit_manifest(df_all, "species_key")
        candidates = filter_species_candidates(manifest, top_k=args.loso_top_k)
        save_df(candidates, loso_root / "species_candidates.csv", loso_root / "species_candidates.md")
        for i, cand in candidates.iterrows():
            species_key = cand["species_key"]
            species_label = cand.get("species_label", species_key)
            run_root = ensure_dir(loso_root / f"holdout_species_{i + 1:02d}")
            print(f"[LOSO] {i + 1}/{len(candidates)} holdout={species_key}")
            try:
                split_result = split_holdout_unit(
                    df_all,
                    holdout_unit_col="species_key",
                    holdout_value=species_key,
                    val_group_col="species_key",
                    val_frac_from_rest=OOD_VAL_FRAC_FROM_REST,
                    random_state=SEED + i,
                )
                tr, va, te = split_result[:3]
                save_split_files(run_root, tr, va, te)
                for baseline in args.baselines:
                    out_dir = ensure_dir(run_root / baseline)
                    row = train_one_experiment(baseline, tr, va, te, out_dir, args, device, tokenizer)
                    row.update(
                        {
                            "benchmark": "loso_species",
                            "heldout_species": species_key,
                            "heldout_species_label": species_label,
                            "species_support_total": int(cand["n_total"]),
                            "species_support_pos": int(cand["n_pos"]),
                            "species_support_neg": int(cand["n_neg"]),
                        }
                    )
                    rows.append(row)
            except Exception as exc:
                save_json({"heldout_species": species_key, "error": str(exc)}, run_root / "error.json")

    result_df = pd.DataFrame(rows)
    save_df(result_df, root_dir / "dnaberts_baseline_results.csv", root_dir / "dnaberts_baseline_results.md")
    agg = aggregate_summary(result_df)
    save_df(agg, root_dir / "aggregate_dnaberts_summary.csv", root_dir / "aggregate_dnaberts_summary.md")
    print("[Done] wrote:")
    print(f"  {root_dir / 'dnaberts_baseline_results.csv'}")
    print(f"  {root_dir / 'aggregate_dnaberts_summary.csv'}")


if __name__ == "__main__":
    main()
