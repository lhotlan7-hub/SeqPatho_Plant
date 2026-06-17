# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = ROOT / "Sample"
if str(SAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(SAMPLE_DIR))

import attention_rollout_and_saliency as attn03  # noqa: E402


DEFAULT_FASTA = ROOT / "Instantiation" / "downloads" / "Cmu1" / "dna" / "XM_011393174.1.fasta"
DEFAULT_OUTDIR = SAMPLE_DIR / "per_base_predictions"
DEFAULT_MODEL_DIR = Path(r"C:\Users\GQ\.cache\huggingface\modules\transformers_modules\DNABERT-S")
FALLBACK_MODEL_DIR = ROOT / "DNABERT-S"
DEFAULT_CHECKPOINT = (
    ROOT
    / "Models_06_NPlants"
    / "results_PGPred_12_500_LR_20251122_230359"
    / "checkpoints"
    / "best_model_epoch24.pth"
)

MAX_LEN = 500
K_MER = 6
WINDOW_NT = MAX_LEN + K_MER - 1
STRIDE_NT = WINDOW_NT
THRESHOLD = 0.5

RC_TABLE = str.maketrans("ACGTN", "TGCAN")
VALID_BASES = set("ACGTN")


def read_fasta(path: Path) -> Tuple[str, str]:
    header = ""
    parts: List[str] = []
    with path.open("r", encoding="utf-8-sig", errors="ignore") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if not header:
                    header = line[1:].strip()
                continue
            parts.append(line)
    seq = "".join(parts).upper()
    seq = "".join(ch if ch in VALID_BASES else "N" for ch in seq)
    if not seq:
        raise ValueError(f"No DNA sequence found in {path}")
    return header, seq


def revcomp(seq: str) -> str:
    return seq.translate(RC_TABLE)[::-1]


def sanitize_tag(raw: str) -> str:
    tag = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in raw.strip())
    tag = tag.strip("._-")
    return tag.replace(".", "_") or "query"


def make_windows(seq: str) -> List[Tuple[int, str]]:
    if len(seq) <= WINDOW_NT:
        return [(0, seq)]

    starts = list(range(0, len(seq) - WINDOW_NT + 1, STRIDE_NT))
    tail_start = len(seq) - WINDOW_NT
    if starts[-1] != tail_start:
        starts.append(tail_start)
    return [(start, seq[start : start + WINDOW_NT]) for start in starts]


def resolve_model_dir(raw: Path) -> Path:
    if raw.exists():
        return raw
    if FALLBACK_MODEL_DIR.exists():
        return FALLBACK_MODEL_DIR
    raise FileNotFoundError(f"DNABERT-S model dir not found: {raw} or {FALLBACK_MODEL_DIR}")


def load_model(model_dir: Path, checkpoint: Path, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    model = attn03.DNABiGRUWithDNABERT(
        dnabert_dir=str(model_dir),
        num_classes=2,
        hidden_size=64,
        freeze_dnabert=False,
        finetune_from_layer=10,
    ).to(device)

    state = torch.load(str(checkpoint), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = attn03.strip_module_prefix(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    model.eval()
    patched = attn03.patch_bert_attention_capture(model.dnabert_model)
    return tokenizer, model, missing, unexpected, patched


def normalize_sum1(values: np.ndarray) -> np.ndarray:
    arr = values.astype(np.float64)
    arr = np.maximum(arr, 0.0)
    total = float(arr.sum())
    if total <= 0:
        return np.zeros_like(arr, dtype=np.float64)
    return arr / total


def normalize_minmax(values: np.ndarray) -> np.ndarray:
    arr = values.astype(np.float64)
    lo = float(arr.min()) if arr.size else 0.0
    hi = float(arr.max()) if arr.size else 0.0
    if hi - lo <= 1e-15:
        return np.zeros_like(arr, dtype=np.float64)
    return (arr - lo) / (hi - lo)


@torch.no_grad()
def predict_window_with_attention(window_seq: str, tokenizer, model, device: torch.device) -> Dict:
    enc = tokenizer(
        window_seq,
        padding="longest",
        truncation=True,
        max_length=MAX_LEN,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device).long()

    logits, bert_attns = model(input_ids, attention_mask, return_bert_attn=True)
    logits = logits.float().squeeze(0)
    prob_pathogenic = float(F.softmax(logits, dim=0)[1].item())

    visible_len = int(attention_mask[0].sum().item())
    ids_visible = input_ids[0, :visible_len].detach().cpu().tolist()
    tokens = tokenizer.convert_ids_to_tokens(ids_visible)

    offsets_raw = enc["offset_mapping"][0, :visible_len].detach().cpu().tolist()
    offsets = [(int(a), int(b)) for a, b in offsets_raw]

    attn_mats = []
    if bert_attns is not None:
        for layer_attn in bert_attns:
            if layer_attn is None:
                continue
            mat = layer_attn[0].mean(dim=0)[:visible_len, :visible_len].float().cpu().numpy()
            mat = mat / (mat.sum(axis=1, keepdims=True) + 1e-12)
            attn_mats.append(mat)
    if not attn_mats:
        raise RuntimeError("No attention matrices were captured.")

    limit = min(attn_mats[0].shape[0], attn03.MAX_TOKENS_FOR_IMPORTANCE)
    mats_cut = [m[:limit, :limit] for m in attn_mats]
    tokens_cut = tokens[:limit]
    offsets_cut = offsets[:limit]

    rollout = attn03.attention_rollout(mats_cut, add_identity=attn03.ROLLOUT_ADD_IDENTITY)
    token_importance = rollout[0, :].astype(np.float64)

    keep = [i for i, tok in enumerate(tokens_cut) if not attn03.is_special_token(tokenizer, tok)]
    tokens_keep = [tokens_cut[i] for i in keep]
    offsets_keep = [offsets_cut[i] for i in keep]
    token_imp_keep = normalize_sum1(token_importance[keep])

    win_base_raw = np.zeros(len(window_seq), dtype=np.float64)
    win_base_count = np.zeros(len(window_seq), dtype=np.float64)
    for tok, (start, end), weight in zip(tokens_keep, offsets_keep, token_imp_keep):
        if end <= start:
            continue
        start = max(0, min(start, len(window_seq)))
        end = max(0, min(end, len(window_seq)))
        if end <= start:
            continue
        # Match the existing BASE_AGG_MODE=max behavior for token->base mapping.
        win_base_raw[start:end] = np.maximum(win_base_raw[start:end], float(weight))
        win_base_count[start:end] += 1.0

    return {
        "logits": logits.detach().cpu().numpy().astype(np.float64),
        "prob_pathogenic": prob_pathogenic,
        "tokens": tokens_keep,
        "offsets": offsets_keep,
        "token_importance": token_imp_keep,
        "base_raw": win_base_raw,
        "base_token_count": win_base_count,
        "num_layers": len(attn_mats),
        "visible_tokens": visible_len,
    }


def predict_sequence(seq: str, tokenizer, model, device: torch.device, collect_attention: bool) -> Dict:
    windows = make_windows(seq)
    logits_list = []
    window_rows = []
    token_rows = []
    base_sum = np.zeros(len(seq), dtype=np.float64)
    base_cov = np.zeros(len(seq), dtype=np.float64)
    base_token_count = np.zeros(len(seq), dtype=np.float64)

    for widx, (start, window_seq) in enumerate(windows):
        result = predict_window_with_attention(window_seq, tokenizer, model, device)
        logits_list.append(result["logits"])
        window_rows.append(
            {
                "window_index": widx,
                "start_1based": start + 1,
                "end_1based": start + len(window_seq),
                "seq_len_nt": len(window_seq),
                "prob_pathogenic": result["prob_pathogenic"],
                "visible_tokens": result["visible_tokens"],
                "num_attention_layers": result["num_layers"],
            }
        )

        if collect_attention:
            end = start + len(window_seq)
            base_sum[start:end] += result["base_raw"]
            base_cov[start:end] += 1.0
            base_token_count[start:end] += result["base_token_count"]

            for tidx, (tok, (tok_start, tok_end), weight) in enumerate(
                zip(result["tokens"], result["offsets"], result["token_importance"])
            ):
                if tok_end <= tok_start:
                    continue
                token_rows.append(
                    {
                        "window_index": widx,
                        "window_start_1based": start + 1,
                        "token_index": tidx,
                        "token": tok,
                        "token_start_1based": start + tok_start + 1,
                        "token_end_1based_inclusive": start + tok_end,
                        "token_weight_sum1_within_window": float(weight),
                    }
                )

    logits_mean = np.mean(np.stack(logits_list, axis=0), axis=0)
    probs = torch.softmax(torch.tensor(logits_mean, dtype=torch.float32), dim=0).numpy()
    prob_pathogenic = float(probs[1])

    base_raw = np.divide(base_sum, np.maximum(base_cov, 1.0))
    base_sum1 = normalize_sum1(base_raw)
    base_minmax = normalize_minmax(base_raw)

    return {
        "prob_pathogenic": prob_pathogenic,
        "label": "pathogenic" if prob_pathogenic >= THRESHOLD else "non_pathogenic",
        "logits_mean": logits_mean,
        "windows": window_rows,
        "token_rows": token_rows,
        "base_raw": base_raw,
        "base_sum1": base_sum1,
        "base_minmax": base_minmax,
        "base_cov": base_cov,
        "base_token_count": base_token_count,
    }


def write_csv(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta", type=Path, default=DEFAULT_FASTA)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tag", default=None, help="Output file prefix. Defaults to the first FASTA header token.")
    args = parser.parse_args()

    model_dir = resolve_model_dir(args.model_dir)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    header, seq = read_fasta(args.fasta)
    tag_source = args.tag or (header.split()[0] if header else args.fasta.stem)
    tag = sanitize_tag(tag_source)
    outdir = args.outdir / tag
    outdir.mkdir(parents=True, exist_ok=True)
    tokenizer, model, missing, unexpected, patched = load_model(model_dir, args.checkpoint, device)

    forward = predict_sequence(seq, tokenizer, model, device, collect_attention=True)
    reverse = predict_sequence(revcomp(seq), tokenizer, model, device, collect_attention=False)

    if forward["label"] == reverse["label"]:
        robust_call = forward["label"]
    else:
        robust_call = "uncertain"

    base_rows = []
    for i, base in enumerate(seq):
        base_rows.append(
            {
                "position_1based": i + 1,
                "base": base,
                "weight": f"{forward['base_sum1'][i]:.12g}",
                "weight_raw_window_mean": f"{forward['base_raw'][i]:.12g}",
                "weight_norm_0_1": f"{forward['base_minmax'][i]:.12g}",
                "windows_covering": int(forward["base_cov"][i]),
                "token_cover_count": int(forward["base_token_count"][i]),
            }
        )

    base_csv = outdir / f"{tag}_per_nucleotide_weights.csv"
    write_csv(
        base_csv,
        base_rows,
        [
            "position_1based",
            "base",
            "weight",
            "weight_raw_window_mean",
            "weight_norm_0_1",
            "windows_covering",
            "token_cover_count",
        ],
    )

    token_csv = outdir / f"{tag}_token_weights.csv"
    write_csv(
        token_csv,
        forward["token_rows"],
        [
            "window_index",
            "window_start_1based",
            "token_index",
            "token",
            "token_start_1based",
            "token_end_1based_inclusive",
            "token_weight_sum1_within_window",
        ],
    )

    window_csv = outdir / f"{tag}_window_predictions.csv"
    write_csv(
        window_csv,
        forward["windows"],
        [
            "window_index",
            "start_1based",
            "end_1based",
            "seq_len_nt",
            "prob_pathogenic",
            "visible_tokens",
            "num_attention_layers",
        ],
    )

    top_idx = np.argsort(forward["base_sum1"])[::-1][:20]
    top_rows = [
        {
            "rank": rank,
            "position_1based": int(i + 1),
            "base": seq[int(i)],
            "weight": f"{forward['base_sum1'][i]:.12g}",
            "weight_raw_window_mean": f"{forward['base_raw'][i]:.12g}",
            "weight_norm_0_1": f"{forward['base_minmax'][i]:.12g}",
        }
        for rank, i in enumerate(top_idx, start=1)
    ]
    top_csv = outdir / f"{tag}_top20_nucleotide_weights.csv"
    write_csv(
        top_csv,
        top_rows,
        ["rank", "position_1based", "base", "weight", "weight_raw_window_mean", "weight_norm_0_1"],
    )

    summary = {
        "fasta": str(args.fasta),
        "header": header,
        "record_tag": tag,
        "sequence_length_nt": len(seq),
        "model_dir": str(model_dir),
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "attention_modules_patched": patched,
        "load_missing_keys_count": len(missing),
        "load_unexpected_keys_count": len(unexpected),
        "threshold": THRESHOLD,
        "window_nt": WINDOW_NT,
        "stride_nt": STRIDE_NT,
        "forward_prob_pathogenic": forward["prob_pathogenic"],
        "forward_label": forward["label"],
        "revcomp_prob_pathogenic": reverse["prob_pathogenic"],
        "revcomp_label": reverse["label"],
        "prob_mean_forward_revcomp": float((forward["prob_pathogenic"] + reverse["prob_pathogenic"]) / 2.0),
        "prob_abs_diff_forward_revcomp": float(abs(forward["prob_pathogenic"] - reverse["prob_pathogenic"])),
        "robust_call": robust_call,
        "outputs": {
            "per_nucleotide_weights": str(base_csv),
            "token_weights": str(token_csv),
            "window_predictions": str(window_csv),
            "top20_nucleotide_weights": str(top_csv),
        },
    }
    summary_json = outdir / f"{tag}_prediction_summary.json"
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
