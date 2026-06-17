# -*- coding: utf-8 -*-
import os
import sys
from datetime import datetime
import random
import warnings
from collections import defaultdict
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
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
import csv
from curve_export_helper import (
    apply_runtime_overrides,
    apply_seed_override,
    resolve_checkpoint_path,
    save_curve_points,
)

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# ========= 可复现性 =========
SEED = apply_seed_override(42)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# =========（可选）TF32 =========
if torch.cuda.is_available():
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

# ========== 配置（显存友好）==========
model_dir = r"C:\Users\GQ\.cache\huggingface\modules\transformers_modules\DNABERT-S"

# 训练与评估分开控制长度；超过模型上限会自动裁剪
MAX_LEN_TRAIN = 1000
MAX_LEN_EVAL  = 1000
K_MER = 6

batch_size = 32           # DataLoader 的大 batch（内部还有微批，见下）
learning_rate = 2e-6      # 保留字段，未实际使用
epochs = 150
epochs, EVAL_ONLY = apply_runtime_overrides(epochs)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_AMP = torch.cuda.is_available()
print(f"使用设备: {device} (AMP={'On' if USE_AMP else 'Off'})")

# 步长：评估无重叠，窗口更少
TRAIN_STRIDE_TOKENS = MAX_LEN_TRAIN // 2
EVAL_STRIDE_TOKENS  = MAX_LEN_EVAL
EVAL_CHUNK_WINDOWS  = 2048  # 兼容占位

# ====== 评估资源控制======
VAL_BATCH_SIZE = 1
TEST_BATCH_SIZE = 1
EVAL_MAX_TOKENS_PER_CHUNK = 256_000
EVAL_MAX_WINDOWS_PER_SEQ = 512
FLUSH_GPU_CACHE = False

# ====== 训练资源控制（微批 + 梯度累积关键阈）======
TRAIN_MAX_TOKENS_PER_STEP = 20_000

def window_nt(L_tokens, k=K_MER):
    return L_tokens + k - 1

# ========== 早停 ==========
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
                print(f"验证损失改善: {old_best:.6f} -> {self.best_loss:.6f}. 保存模型...")
            torch.save(model.state_dict(), os.path.join(self.save_path, f'best_model_epoch{epoch + 1}.pth'))
            self.counter = 0
            self.best_epoch = epoch
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# ========== 结果保存 ==========
try:
    script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0] or "interactive"
except Exception:
    script_name = "interactive"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"results_{script_name}_{timestamp}"
os.makedirs(output_dir, exist_ok=True)
checkpoint_dir = os.path.join(output_dir, "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)

# ========== DNABERT / Tokenizer ==========
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

# ========== 数据加载 ==========
def load_combined_data(file_a, file_b, random_state=SEED):
    df_a = pd.read_csv(file_a, low_memory=False)
    df_b = pd.read_csv(file_b, low_memory=False)
    seqs_a = df_a.iloc[:, 14].dropna().astype(str)  # 正类
    seqs_b = df_b.iloc[:, 13].dropna().astype(str)  # 负类
    min_len = min(len(seqs_a), len(seqs_b))
    seqs_a = seqs_a.sample(n=min_len, random_state=random_state)
    seqs_b = seqs_b.sample(n=min_len, random_state=random_state)
    sequences = list(seqs_a) + list(seqs_b)
    labels = np.array([1]*len(seqs_a) + [0]*len(seqs_b))
    print(f"载入序列: 正类{len(seqs_a)} / 负类{len(seqs_b)}，总计{len(sequences)}；未做长度丢弃")
    return sequences, labels

def split_70_15_15(sequences, labels, random_state=SEED):
    seq_train, seq_temp, y_train, y_temp = train_test_split(
        sequences, labels, test_size=0.30, stratify=labels, random_state=random_state
    )
    seq_val, seq_test, y_val, y_test = train_test_split(
        seq_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=random_state
    )
    return (seq_train, y_train), (seq_val, y_val), (seq_test, y_test)

# ========== 滑窗 ==========
def make_windows(seq, L_tokens=MAX_LEN_EVAL, k=K_MER, stride_tokens=EVAL_STRIDE_TOKENS):
    s = str(seq).strip().upper()
    wn = L_tokens + k - 1
    if len(s) <= wn:
        return [s]
    step_nt = stride_tokens + k - 1
    out = [s[i:i+wn] for i in range(0, len(s)-wn+1, step_nt)]
    if (len(s)-wn) % step_nt != 0:
        out.append(s[-wn:])
    return out

def sample_train_window(seq, L_tokens=MAX_LEN_TRAIN, k=K_MER):
    s = str(seq).strip().upper()
    wn = L_tokens + k - 1
    if len(s) <= wn:
        return s
    st = random.randint(0, len(s) - wn)
    return s[st:st+wn]

# ========== Dataset ==========
class SeqDataset(Dataset):
    def __init__(self, sequences, labels, train=True, L_tokens_train=MAX_LEN_TRAIN):
        self.seqs = list(sequences)
        self.labels = np.array(labels).astype(int)
        self.train = train
        self.L_tokens_train = L_tokens_train
    def __len__(self):
        return len(self.seqs)
    def __getitem__(self, idx):
        s = self.seqs[idx]
        y = int(self.labels[idx])
        if self.train:
            s = sample_train_window(s, L_tokens=self.L_tokens_train)
        return s, y

def collate_fn(batch):
    seqs, ys = zip(*batch)
    return list(seqs), torch.tensor(ys, dtype=torch.long)

# ========== 路径 ==========
sequences, labels = load_combined_data(
    r'E:\4PythonProject\SeqPatho\Dataset\patho_plants_only.csv',
    r'E:\4PythonProject\SeqPatho\Dataset\Nonpatho_plants_only.csv'
)
(train_sequences, y_train), (val_sequences, y_val), (test_sequences, y_test) = split_70_15_15(sequences, labels, SEED)
print(f"[Split] train={len(train_sequences)} ({np.mean(y_train):.3f} pos rate), "
      f"val={len(val_sequences)} ({np.mean(y_val):.3f}), test={len(test_sequences)} ({np.mean(y_test):.3f})")

train_loader = DataLoader(
    SeqDataset(train_sequences, y_train, train=True, L_tokens_train=MAX_LEN_TRAIN),
    batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
    pin_memory=torch.cuda.is_available(), num_workers=0
)
val_loader = DataLoader(
    SeqDataset(val_sequences, y_val, train=False),
    batch_size=VAL_BATCH_SIZE, shuffle=False, collate_fn=collate_fn,
    pin_memory=torch.cuda.is_available(), num_workers=0
)
test_loader = DataLoader(
    SeqDataset(test_sequences, y_test, train=False),
    batch_size=TEST_BATCH_SIZE, shuffle=False, collate_fn=collate_fn,
    pin_memory=torch.cuda.is_available(), num_workers=0
)

# ========== 模型 ==========
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)
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
        super(StackedChannelAttention, self).__init__()
        self.layers = nn.ModuleList([ChannelAttention(in_channels) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(in_channels) for _ in range(num_layers)])
    def forward(self, x):
        for attn, norm in zip(self.layers, self.norms):
            residual = x
            x = attn(x)
            x = norm(x + residual)
        return x

class DNABiGRUWithDNABERT(nn.Module):
    def __init__(self, dnabert_dir, num_classes, hidden_size=64, freeze_dnabert=False):
        super(DNABiGRUWithDNABERT, self).__init__()
        self.channel_attention = StackedChannelAttention(in_channels=hidden_size * 2, num_layers=3)
        self.dropout = nn.Dropout(p=0.5)
        self.layernorm = nn.LayerNorm(hidden_size * 2)
        self.dnabert_model = AutoModel.from_pretrained(dnabert_dir, trust_remote_code=True)
        self.bert_hidden_size = 768
        self.freeze_dnabert = freeze_dnabert

        # 仅微调后2-3层
        for name, param in self.dnabert_model.named_parameters():
            if "encoder.layer" in name:
                layer_num = int(name.split("encoder.layer.")[1].split(".")[0])
                param.requires_grad = (layer_num >= 10)
            else:
                param.requires_grad = False

        # Gradient checkpointing
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
        self.fc1 = nn.Linear(hidden_size * 2, 32)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, input_ids, attention_mask=None):
        if self.freeze_dnabert:
            with torch.no_grad():
                bert_output = self.dnabert_model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            bert_output = self.dnabert_model(input_ids=input_ids, attention_mask=attention_mask)

        embeddings = bert_output[0]  # [B, L, 768]
        seq_lengths = attention_mask.sum(dim=1) if attention_mask is not None else (input_ids != 0).sum(dim=1)

        packed = nn.utils.rnn.pack_padded_sequence(
            embeddings, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.bigru(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, total_length=embeddings.size(1)
        )  # [B, L_fixed, 2H]

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

model = DNABiGRUWithDNABERT(
    dnabert_dir=model_dir,
    num_classes=2,
    hidden_size=64,
    freeze_dnabert=False
).to(device)

# 自动裁剪到模型位置上限
MAX_POS = getattr(model.dnabert_model.config, "max_position_embeddings", MAX_LEN_EVAL)
if MAX_LEN_TRAIN > MAX_POS or MAX_LEN_EVAL > MAX_POS:
    old_train, old_eval = MAX_LEN_TRAIN, MAX_LEN_EVAL
    MAX_LEN_TRAIN = min(MAX_LEN_TRAIN, MAX_POS)
    MAX_LEN_EVAL  = min(MAX_LEN_EVAL,  MAX_POS)
    print(f"[WARN] max_position_embeddings={MAX_POS}，已将 MAX_LEN_TRAIN {old_train}->{MAX_LEN_TRAIN}，MAX_LEN_EVAL {old_eval}->{MAX_LEN_EVAL}")

criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

# ====== 分组学习率（编码器小、头部大）======
def build_param_groups(model, lr_enc=5e-6, lr_head=2e-4, wd_enc=0.01, wd_head=2e-4):
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
    param_groups = []
    if enc_decay:
        param_groups.append({"params": enc_decay,   "lr": lr_enc,  "weight_decay": wd_enc, "group_name": "enc_decay"})
    if enc_nodecay:
        param_groups.append({"params": enc_nodecay, "lr": lr_enc,  "weight_decay": 0.0,    "group_name": "enc_nodecay"})
    if head_params:
        param_groups.append({"params": head_params, "lr": lr_head, "weight_decay": wd_head,"group_name": "head"})
    return param_groups

param_groups = build_param_groups(
    model,
    lr_enc=5e-6,   # ← 推荐档：可改为 3e-6(稳) / 1e-5(激进)
    lr_head=2e-4,  # ← 头部更大，加速召回
    wd_enc=0.01,
    wd_head=2e-4
)
optimizer = optim.AdamW(param_groups, eps=1e-8)
scaler = GradScaler(enabled=USE_AMP)

# 调度器：5% warmup + 线性衰减（按“大 batch”计步）
num_update_steps_per_epoch = len(train_loader)
num_train_steps = epochs * num_update_steps_per_epoch
num_warmup_steps = max(1, int(0.05 * num_train_steps))
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_train_steps)
USE_WARMUP_SCHED = True

# ========== 评估（逐序列流式 + token预算 + longest padding）==========
@torch.inference_mode()
def evaluate_sequence_loader(model, data_loader, stride_tokens=EVAL_STRIDE_TOKENS,
                             chunk_windows=EVAL_CHUNK_WINDOWS, use_amp=USE_AMP):
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

            logits_sum = torch.zeros(2, device=device, dtype=torch.float32)
            n_win = 0

            for off in range(0, len(wins), windows_per_chunk):
                chunk = wins[off:off + windows_per_chunk]
                tok = tokenizer(chunk, padding="longest", truncation=True,
                                max_length=MAX_LEN_EVAL, return_tensors='pt')
                input_ids = tok['input_ids'].to(device, non_blocking=True)
                attention_mask = tok['attention_mask'].to(device, non_blocking=True).long()

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
            loss_seq = criterion(agg_logits.unsqueeze(0), torch.tensor([y], device=device))
            seq_loss_sum += loss_seq.item()
            n_seq_count += 1

            prob = F.softmax(agg_logits, dim=0)[1].item()
            pred = int(torch.argmax(agg_logits).item())
            all_probs.append(prob); all_preds.append(pred); all_labels.append(y)

    avg_val_loss = (seq_loss_sum / max(1, n_seq_count)) if n_seq_count > 0 else float('inf')
    return avg_val_loss, np.array(all_labels), np.array(all_preds), np.array(all_probs)

def compute_classification_metrics(labels, preds, probs):
    if len(labels) == 0:
        return dict(accuracy=np.nan, precision=np.nan, recall=np.nan, f1=np.nan,
                    mcc=np.nan, auroc=np.nan, auprc=np.nan, ap=np.nan)
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    mcc = matthews_corrcoef(labels, preds)
    try:
        fpr, tpr, _ = roc_curve(labels, probs); auroc = auc(fpr, tpr)
    except Exception:
        auroc = float("nan")
    try:
        prec_vals, rec_vals, _ = precision_recall_curve(labels, probs)
        auprc = auc(rec_vals, prec_vals); ap = average_precision_score(labels, probs)
    except Exception:
        auprc, ap = float("nan"), float("nan")
    return dict(accuracy=accuracy, precision=precision, recall=recall, f1=f1,
                mcc=mcc, auroc=auroc, auprc=auprc, ap=ap)

# ========== 训练监控 ==========
train_losses, val_losses = [], []
accuracy_list, precision_list, recall_list, f1_list, mcc_list = [], [], [], [], []
auroc_list, auprc_list, ap_list = [], [], []
epoch_lrs_enc, epoch_lrs_head = [], []
early_stopping = EarlyStopping(save_path=checkpoint_dir, patience=8, verbose=True, delta=0.001)
metrics_file_path = os.path.join(output_dir, "metrics_per_epoch.txt")
with open(metrics_file_path, "w", encoding="utf-8") as f:
    f.write("Epoch\tTrain_Loss\tVal_Loss\tAccuracy\tPrecision\tRecall\tF1\tMCC\tAUROC\tAUPRC\tAP\tLR_Enc\tLR_Head\n")

def get_group_lrs(optimizer):
    lr_enc = None
    lr_head = None
    for g in optimizer.param_groups:
        name = g.get("group_name", "")
        if name.startswith("enc"):
            lr_enc = g["lr"]
        if name == "head":
            lr_head = g["lr"]
    # 如果 enc 分了两个组，lr_enc 会被最后一次覆盖，但两组lr相同，不影响
    return lr_enc, lr_head

# ========== 训练（微批 + 梯度累积 + 每步调度）==========
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    seen_batches = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{epochs}]", leave=False)

    for batch_sequences, batch_labels in progress_bar:
        batch_labels = batch_labels.to(device)

        # 动态 padding
        tokenized = tokenizer(batch_sequences, padding='longest', truncation=True,
                              max_length=MAX_LEN_TRAIN, return_tensors='pt')
        input_ids = tokenized['input_ids'].to(device, non_blocking=True)
        attention_mask = tokenized['attention_mask'].to(device, non_blocking=True).long()

        # 合法性过滤
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

        # 微批切分：保证每次前向 tokens <= 预算
        lengths = attention_mask.sum(dim=1)               # [N]
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
                loss_sub = criterion(logits, labels_sub)
                loss = loss_sub / num_sub

            scaler.scale(loss).backward()
            loss_accum += loss_sub.item()

        scaler.step(optimizer)
        scaler.update()
        if USE_WARMUP_SCHED:
            scheduler.step()

        total_loss += (loss_accum / num_sub if num_sub > 0 else 0.0)
        seen_batches += 1
        progress_bar.set_postfix(loss=(total_loss / max(1, seen_batches)))

    avg_train_loss = total_loss / max(1, seen_batches)
    train_losses.append(avg_train_loss)

    # ===== 验证 =====
    val_loss, val_labels, val_preds, val_probs = evaluate_sequence_loader(
        model, val_loader, stride_tokens=EVAL_STRIDE_TOKENS,
        chunk_windows=EVAL_CHUNK_WINDOWS, use_amp=USE_AMP
    )
    val_losses.append(val_loss)
    val_metrics = compute_classification_metrics(val_labels, val_preds, val_probs)

    # 记录本 epoch 结束时的两个 LR
    lr_enc, lr_head = get_group_lrs(optimizer)
    epoch_lrs_enc.append(lr_enc if lr_enc is not None else float('nan'))
    epoch_lrs_head.append(lr_head if lr_head is not None else float('nan'))

    print(
        f"Epoch [{epoch + 1}/{epochs}] - "
        f"Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, "
        f"Acc: {val_metrics['accuracy']:.4f}, P: {val_metrics['precision']:.4f}, "
        f"R: {val_metrics['recall']:.4f}, F1: {val_metrics['f1']:.4f}, "
        f"MCC: {val_metrics['mcc']:.4f}, AUROC: {val_metrics['auroc']:.4f}, "
        f"AUPRC: {val_metrics['auprc']:.4f}, AP: {val_metrics['ap']:.4f}, "
        f"LR_Enc: {epoch_lrs_enc[-1]:.2e}, LR_Head: {epoch_lrs_head[-1]:.2e}"
    )
    with open(metrics_file_path, "a", encoding="utf-8") as f:
        f.write(
            f"{epoch + 1}\t{avg_train_loss:.4f}\t{val_loss:.4f}\t"
            f"{val_metrics['accuracy']:.4f}\t{val_metrics['precision']:.4f}\t"
            f"{val_metrics['recall']:.4f}\t{val_metrics['f1']:.4f}\t{val_metrics['mcc']:.4f}\t"
            f"{val_metrics['auroc']:.4f}\t{val_metrics['auprc']:.4f}\t{val_metrics['ap']:.4f}\t"
            f"{epoch_lrs_enc[-1]:.6e}\t{epoch_lrs_head[-1]:.6e}\n"
        )

    # 早停（基于 val loss）
    early_stopping(val_loss, model, epoch)
    if early_stopping.early_stop:
        print(f"早停触发于第 {epoch + 1} 轮，最佳模型来自第 {early_stopping.best_epoch + 1} 轮")
        break

# ========== 使用最佳验证轮权重做最终测试 ==========
best_epoch_idx = early_stopping.best_epoch
best_ckpt_path = resolve_checkpoint_path(os.path.join(checkpoint_dir, f'best_model_epoch{best_epoch_idx + 1}.pth'))
if os.path.exists(best_ckpt_path):
    model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    print(f"[Load Best] 加载最佳验证轮权重：{best_ckpt_path}")
else:
    print("[Load Best] 未找到最佳权重文件，使用当前模型参数进行测试。")

# ========== 最终评估 ==========
test_loss, test_labels, test_preds, test_probs = evaluate_sequence_loader(
    model, test_loader, stride_tokens=EVAL_STRIDE_TOKENS,
    chunk_windows=EVAL_CHUNK_WINDOWS, use_amp=USE_AMP
)
test_metrics = compute_classification_metrics(test_labels, test_preds, test_probs)

# 曲线与报告
fpr, tpr, _ = roc_curve(test_labels, test_probs)
auroc_final = auc(fpr, tpr)
precision_vals, recall_vals, _ = precision_recall_curve(test_labels, test_probs)
ap_final = average_precision_score(test_labels, test_probs)
auprc_final = auc(recall_vals, precision_vals)
save_curve_points(output_dir, fpr=fpr, tpr=tpr, recall=recall_vals, precision=precision_vals)

cm = confusion_matrix(test_labels, test_preds)
plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Confusion Matrix (Test)')
plt.savefig(os.path.join(output_dir, "confusion_matrix_test.png")); plt.close()

plt.figure()
plt.plot(fpr, tpr, lw=2, label=f'ROC (AUROC = {auroc_final:.2f})')
plt.plot([0, 1], [0, 1], lw=2, linestyle='--')
plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('ROC Curve (Test)')
plt.legend(loc="lower right"); plt.grid(True)
plt.savefig(os.path.join(output_dir, "roc_curve_test.png")); plt.close()

plt.figure()
plt.step(recall_vals, precision_vals, where='post', lw=2, label=f'PR (AP = {ap_final:.2f}, AUPRC = {auprc_final:.2f})')
plt.fill_between(recall_vals, precision_vals, step='post', alpha=0.2)
plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('PR Curve (Test)')
plt.legend(loc="upper right"); plt.grid(True)
plt.savefig(os.path.join(output_dir, "pr_curve_test.png")); plt.close()

final_accuracy = test_metrics["accuracy"]
final_precision = test_metrics["precision"]
final_recall = test_metrics["recall"]
final_f1 = test_metrics["f1"]
final_mcc = test_metrics["mcc"]
avg_train_loss = train_losses[-1] if len(train_losses)>0 else float('nan')
avg_val_loss = val_losses[-1] if len(val_losses)>0 else float('inf')

metrics_text = f"""Final Test Metrics (evaluated on TEST set only):
Train Loss (last epoch): {avg_train_loss:.4f}
Validation Loss (last epoch): {avg_val_loss:.4f}
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
"""
with open(os.path.join(output_dir, "final_metrics_test.txt"), "w", encoding="utf-8") as f:
    f.write(metrics_text)

with open(os.path.join(output_dir, "final_metrics_test.csv"), mode='w', newline='', encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Metric", "Value"])
    writer.writerow(["Train Loss (last)", avg_train_loss])
    writer.writerow(["Validation Loss (last)", avg_val_loss])
    writer.writerow(["Test Loss", test_loss])
    writer.writerow(["Accuracy", final_accuracy])
    writer.writerow(["Precision", final_precision])
    writer.writerow(["Recall", final_recall])
    writer.writerow(["F1-score", final_f1])
    writer.writerow(["MCC", final_mcc])
    writer.writerow(["AUROC", auroc_final])
    writer.writerow(["AUPRC", auprc_final])
    writer.writerow(["Average Precision (AP)", ap_final])
    writer.writerow(["Early Stopping Triggered", 'Yes' if early_stopping.early_stop else 'No'])
    writer.writerow(["Best Epoch (val)", early_stopping.best_epoch + 1])

loss_data = pd.DataFrame({
    'Epoch': range(1, len(train_losses) + 1),
    'Train_Loss': train_losses,
    'Validation_Loss': val_losses
})
loss_data.to_csv(os.path.join(output_dir, "loss_data.csv"), index=False)

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
    'LR_Enc': epoch_lrs_enc[:len(f1_list)],
    'LR_Head': epoch_lrs_head[:len(f1_list)]
})
metrics_df.to_csv(os.path.join(output_dir, "val_metrics.csv"), index=False)

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
    plt.figure()
    plt.plot(range(1, len(values)+1), values, label=metric_name)
    plt.xlabel("Epoch"); plt.ylabel(metric_name)
    plt.title(f"{metric_name} over Epochs (Validation)")
    plt.grid(True); plt.legend()
    filename = f"{metric_name.lower().replace('-', '_')}_curve_val.png"
    plt.savefig(os.path.join(output_dir, filename)); plt.close()

# 学习率曲线（编码器 vs 头部）
if len(epoch_lrs_enc) > 0:
    plt.figure()
    plt.plot(range(1, len(epoch_lrs_enc)+1), epoch_lrs_enc, label='LR_Enc')
    plt.plot(range(1, len(epoch_lrs_head)+1), epoch_lrs_head, label='LR_Head')
    plt.xlabel("Epoch"); plt.ylabel("LR"); plt.title("Learning Rate over Epochs")
    plt.yscale('log')
    plt.legend(); plt.grid(True)
    plt.savefig(os.path.join(output_dir, "lr_curve.png")); plt.close()

plt.figure()
plt.plot(loss_data['Epoch'], loss_data['Train_Loss'], label='Train Loss')
plt.plot(loss_data['Epoch'], loss_data['Validation_Loss'], label='Validation Loss')
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training vs Validation Loss")
plt.legend(); plt.grid(True)
plt.savefig(os.path.join(output_dir, "loss_curve.png")); plt.close()

print(f"Training completed! Results saved to: {output_dir}")
print(f"Best model saved in: {checkpoint_dir}")
print("注意：训练期所有指标均基于 Validation；最终图与报告仅基于 Test。")
