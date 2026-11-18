# -*- coding: utf-8 -*-
"""Pairing control for the DNA representation (supplementary Table S17).

Question addressed: does the DNA representation contribute information that is
specific to the gene it is paired with, or is whatever it carries recoverable
from species- and length-level statistics?

Control design.  The DNA feature of every record is permuted among
records drawn from the same pathogen TaxID and the same DNA-length stratum, so
the shuffled features keep the species and length composition of the cohort but
randomize gene-specific pairing; permutations may retain fixed points and singletons.  The protein features are never moved.  A
class-balanced logistic probe is then fitted on the same inner-training split
and evaluated on the same outer test folds as the main baselines.  If the DNA
representation carried gene-specific complementarity, the true pairing should
rank above the shuffled pairings.

Reads from ``--archive`` and writes ``pairing_shuffle.csv`` under ``--out-dir``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from _paths import ARCHIVE as RELEASE, OUTPUT
SPLIT = "taxid_evidence_1116"
C_GRID = [0.003, 0.01, 0.03, 0.1]
N_PERMUTATIONS = 10


def load_arrays() -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    idx = pd.read_csv(RELEASE / "embeddings" / "cache_index.csv")
    prot = np.load(RELEASE / "embeddings" / "protein_pooled.float16.npy").astype(np.float32)
    pmask = np.load(RELEASE / "embeddings" / "protein_view_mask.bool.npy").astype(np.float32)
    dna = np.load(RELEASE / "embeddings" / "dna_pooled.float16.npy").astype(np.float32)
    dmask = np.load(RELEASE / "embeddings" / "dna_view_mask.bool.npy").astype(np.float32)

    P = (prot * pmask[..., None]).sum(1) / np.maximum(pmask.sum(1), 1e-6)[:, None]
    d = dna.mean(1)                                        # mean over the two strands
    D = (d * dmask[..., None]).sum(1) / np.maximum(dmask.sum(1), 1e-6)[:, None]
    return P, D, idx


def build_strata(idx: pd.DataFrame) -> np.ndarray:
    """Species x DNA-length-tertile strata used for the permutation."""
    man = pd.read_csv(RELEASE / "manifest" / "cohort_manifest_audited.csv", low_memory=False)
    tax = man.set_index("record_id").pathogen_taxid.reindex(idx.record_id).values
    dlen = idx.dna_len.values.astype(float)
    strata = np.empty(len(idx), dtype=object)
    for t in pd.unique(tax):
        ii = np.where(tax == t)[0]
        if len(ii) < 2:
            strata[ii] = [f"{t}_single"] * len(ii)
            continue
        q = pd.qcut(dlen[ii], min(3, len(ii)), labels=False, duplicates="drop")
        strata[ii] = [f"{t}_{'na' if pd.isna(x) else int(x)}" for x in q]
    return strata


def permutation(strata: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    perm = np.arange(len(strata))
    for g in pd.unique(strata):
        ii = np.where(strata == g)[0]
        if len(ii) > 1:
            perm[ii] = rng.permutation(ii)
    return perm


def evaluate(X_blocks: list[np.ndarray], idx: pd.DataFrame, roles: pd.DataFrame, y: np.ndarray,
             row: dict) -> tuple[float, float]:
    aucs = []
    for fold in sorted(roles.fold.unique()):
        s = roles[roles.fold == fold]
        tr = s[s.outer_role == "outer_train"]
        fit = np.array([row[r] for r in tr.loc[tr.inner_role == "inner_train", "record_id"]])
        val = np.array([row[r] for r in tr.loc[tr.inner_role == "inner_validation", "record_id"]])
        te = np.array([row[r] for r in s.loc[s.outer_role == "outer_test", "record_id"]])

        def stack(ids):
            return np.hstack([X[ids] for X in X_blocks])

        scaler = StandardScaler().fit(stack(fit))
        best = None
        for C in C_GRID:
            clf = LogisticRegression(C=C, max_iter=4000, class_weight="balanced").fit(
                scaler.transform(stack(fit)), y[fit])
            a = roc_auc_score(y[val], clf.predict_proba(scaler.transform(stack(val)))[:, 1])
            if best is None or a > best[1]:
                best = (C, a)
        clf = LogisticRegression(C=best[0], max_iter=4000, class_weight="balanced").fit(
            scaler.transform(stack(fit)), y[fit])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(scaler.transform(stack(te)))[:, 1]))
    aucs = np.array(aucs)
    return float(aucs.mean()), float(aucs.std(ddof=1))


def main() -> None:
    P, D, idx = load_arrays()
    roles = pd.read_csv(RELEASE / "manifest" / "split_manifest.csv")
    roles = roles[roles.split_name == SPLIT]
    strata = build_strata(idx)
    row = {r: i for i, r in enumerate(idx.record_id)}
    y = idx.label.values.astype(int)

    rows = []
    m, s = evaluate([P], idx, roles, y, row)
    rows.append(dict(condition="protein only", permutation=None, auroc_mean=m, auroc_sd=s))
    m, s = evaluate([P, D], idx, roles, y, row)
    rows.append(dict(condition="protein + DNA, true pairing", permutation=None, auroc_mean=m, auroc_sd=s))
    for seed in range(1, N_PERMUTATIONS + 1):
        m, s = evaluate([P, D[permutation(strata, seed)]], idx, roles, y, row)
        rows.append(dict(condition="protein + DNA, shuffled pairing", permutation=seed,
                         auroc_mean=m, auroc_sd=s))

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT / "pairing_shuffle.csv", index=False)

    shuffled = out[out.permutation.notna()].auroc_mean.values
    true_val = float(out.loc[out.condition.str.contains("true"), "auroc_mean"].iloc[0])
    prot_val = float(out.loc[out.condition == "protein only", "auroc_mean"].iloc[0])
    rank = int((shuffled < true_val).sum()) + 1
    print(out.to_string(index=False))
    print(f"\nprotein only            : {prot_val:.4f}")
    print(f"true pairing            : {true_val:.4f}")
    print(f"shuffled pairing        : mean {shuffled.mean():.4f}, sd {shuffled.std(ddof=1):.4f}, "
          f"range {shuffled.min():.4f}-{shuffled.max():.4f}")
    print(f"true minus shuffled mean: {true_val - shuffled.mean():+.4f}")
    print(f"rank of true value among {len(shuffled) + 1} evaluated values: {rank}/{len(shuffled) + 1}")
    print(f"\nwritten: {OUTPUT / 'pairing_shuffle.csv'}")


if __name__ == "__main__":
    main()
