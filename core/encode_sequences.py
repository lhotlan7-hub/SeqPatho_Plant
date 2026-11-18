"""Build ordered frozen sequence representations. Inputs/models are local and explicit.

The input sequences are restricted third-party data and are not distributed here.
This reproduces the published view and CLS-query attention pooling operations;
agreement with historical cache must be evaluated, not assumed.
"""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer


def sha(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def dna_views(seq):
    seq = re.sub(r'[^ACGTN]', 'N', str(seq).upper())
    if len(seq) <= 1000:
        return [seq, '', '', ''], [True, False, False, False]
    indices = np.linspace(0, len(seq) - 1, 1000).round().astype(int)
    pos = max(0, (len(seq) - 1000) // 2)
    return [''.join(seq[i] for i in indices), seq[:1000], seq[pos:pos+1000], seq[-1000:]], [True] * 4


def protein_views(seq):
    seq = re.sub(r'[^ACDEFGHIKLMNPQRSTVWYBXZOU]', 'X', str(seq).upper())
    if len(seq) <= 1022:
        return [seq, '', ''], [bool(seq), False, False]
    starts = [0, max(0, (len(seq) - 1022) // 2), len(seq) - 1022]
    return [seq[i:i+1022] for i in starts], [True] * 3


def reverse(seq):
    return seq.translate(str.maketrans('ACGTNacgtn', 'TGCANtgcan'))[::-1].upper()


@torch.no_grad()
def encode(model, tokenizer, strings, max_len, batch, device, amp):
    special = [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id,
               getattr(tokenizer, 'eos_token_id', None), getattr(tokenizer, 'bos_token_id', None)]
    output = []
    for start in range(0, len(strings), batch):
        tokens = tokenizer(strings[start:start+batch], padding=True, truncation=True,
                           max_length=max_len, return_tensors='pt', add_special_tokens=True)
        tokens = {k: v.to(device) for k, v in tokens.items()}
        with torch.cuda.amp.autocast(enabled=amp and device.type == 'cuda'):
            x = model(**tokens, output_attentions=False, output_hidden_states=False)
            hidden = x[0] if isinstance(x, (tuple, list)) else x.last_hidden_state
        mask = tokens['attention_mask'].bool().clone()
        for token in special:
            if token is not None:
                mask &= tokens['input_ids'].ne(int(token))
        empty = ~mask.any(dim=1)
        if empty.any():
            mask[empty] = tokens['attention_mask'][empty].bool()
        query = hidden[:, 0, :].float()
        scores = torch.einsum('bld,bd->bl', hidden.float(), query) / math.sqrt(hidden.shape[-1])
        weights = torch.softmax(scores.masked_fill(~mask, -1e4), dim=1)
        output.append(torch.einsum('bl,bld->bd', weights, hidden.float()).cpu().numpy().astype(np.float16))
    return np.concatenate(output, axis=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--sequences', required=True)
    p.add_argument('--dna-model', required=True, help='Pinned local snapshot; uses trusted model code')
    p.add_argument('--protein-model', required=True, help='Pinned local snapshot')
    p.add_argument('--out', required=True)
    p.add_argument('--limit', type=int, default=0, help='Optional smoke subset, 0 means all')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    p.add_argument('--no-amp', action='store_true')
    p.add_argument('--compare-dna')
    p.add_argument('--compare-protein')
    p.add_argument('--compare-dna-mask')
    p.add_argument('--compare-protein-mask')
    a = p.parse_args()
    dest = Path(a.out)
    if dest.exists() and any(dest.iterdir()):
        raise SystemExit('Output directory must be empty; refusing to overwrite existing cache')
    dest.mkdir(parents=True, exist_ok=True)
    device = torch.device(a.device)
    m = pd.read_csv(a.manifest, dtype={'record_id': str}, low_memory=False)
    m = m[m.include_evidence_linked == True].reset_index(drop=True)
    seq = pd.read_csv(a.sequences, dtype={'record_id': str}).set_index('record_id', verify_integrity=True)
    assert m.record_id.is_unique and len(m) == 1116
    assert m.cache_row.tolist() == list(range(1116)), 'unexpected cache order'
    if a.limit:
        m = m.iloc[:a.limit].copy()
    seq = seq.loc[m.record_id].reset_index()
    assert seq.record_id.tolist() == m.record_id.tolist()
    for i, r in seq.iterrows():
        assert sha(str(r.dna_sequence_norm)) == m.iloc[i].dna_sha256_norm
        assert sha(str(r.protein_sequence_used)) == m.iloc[i].protein_sha256_used
    dna_items, protein_items = [], []
    dm, pm = np.zeros((len(m), 4), bool), np.zeros((len(m), 3), bool)
    for i, row in seq.iterrows():
        views, dm[i] = dna_views(row.dna_sequence_norm)
        for j, v in enumerate(views):
            if dm[i, j]:
                dna_items.extend([(i, 0, j, v), (i, 1, j, reverse(v))])
        views, pm[i] = protein_views(row.protein_sequence_used)
        for j, v in enumerate(views):
            if pm[i, j]:
                protein_items.append((i, j, v))
    assert dm.any(axis=1).all() and pm.any(axis=1).all()
    print('loading DNA', flush=True)
    tk = AutoTokenizer.from_pretrained(a.dna_model, trust_remote_code=True, local_files_only=True)
    model = AutoModel.from_pretrained(a.dna_model, trust_remote_code=True, local_files_only=True).to(device).eval()
    arr = np.zeros((len(m), 2, 4, 768), np.float16)
    for st in range(0, len(dna_items), 16):
        part = dna_items[st:st+16]
        vec = encode(model, tk, [x[-1] for x in part], 512, 16, device, not a.no_amp)
        for (i, strand, j, _), v in zip(part, vec): arr[i, strand, j] = v
        if st % 256 == 0: print('DNA views', st, '/', len(dna_items), flush=True)
    np.save(dest/'dna_pooled.float16.npy', arr)
    np.save(dest/'dna_view_mask.bool.npy', dm)
    del model, tk
    if device.type == 'cuda': torch.cuda.empty_cache()
    print('loading protein', flush=True)
    tk = AutoTokenizer.from_pretrained(a.protein_model, local_files_only=True)
    model = AutoModel.from_pretrained(a.protein_model, local_files_only=True, add_pooling_layer=False).to(device).eval()
    prot = np.zeros((len(m), 3, 640), np.float16)
    for st in range(len(protein_items)):
        i, j, v = protein_items[st]
        prot[i, j] = encode(model, tk, [v], 1024, 1, device, not a.no_amp)[0]
        if st % 100 == 0: print('protein windows', st, '/', len(protein_items), flush=True)
    np.save(dest/'protein_pooled.float16.npy', prot)
    np.save(dest/'protein_view_mask.bool.npy', pm)
    m[['record_id', 'cache_row', 'label', 'dna_sha256_norm', 'protein_sha256_used']].to_csv(dest/'cache_index.csv', index=False)
    result = {'n_records': len(m), 'dna_shape': list(arr.shape), 'protein_shape': list(prot.shape),
              'device': str(device), 'amp': not a.no_amp, 'torch': torch.__version__, 'models': [a.dna_model, a.protein_model], 'comparison': {}}
    for name, generated, path in [('dna',arr,a.compare_dna),('protein',prot,a.compare_protein),('dna_mask',dm,a.compare_dna_mask),('protein_mask',pm,a.compare_protein_mask)]:
        if path:
            old=np.load(path, mmap_mode='r')[:len(m)]
            assert old.shape == generated.shape
            if generated.dtype == bool:
                result['comparison'][name] = {'exact_match': bool(np.array_equal(generated,old))}
            else:
                delta=np.abs(generated.astype('float32')-np.asarray(old,dtype='float32'))
                result['comparison'][name]={'max_abs':float(delta.max()),'mean_abs':float(delta.mean()),'exact_fraction':float(np.mean(generated==old))}
    (dest/'encoding_metadata.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
