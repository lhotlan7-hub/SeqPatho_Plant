# -*- coding: utf-8 -*-
"""Reproduce Table 3 of the manuscript from the released archive alone.

Every input is read from the external ``--archive`` directory.
The recipe follows Section 2.4 of the manuscript:

* protein feature  : mean of the valid cached ESM-2 window vectors (640-d);
* DNA feature      : mean of the cached DNABERT-2 vectors over strands and valid
                     views (768-d);
* concatenation    : [protein | DNA];
* composition      : protein length, DNA length, amino-acid frequencies (20),
                     GC content and dinucleotide frequencies (16) -> 39-d;
* length only      : protein length;
* classifier       : StandardScaler + class-balanced LogisticRegression, with C
                     selected from {0.003,0.01,0.03,0.1,0.3,1,3} on the single
                     inner validation split of each outer fold, then fit on the
                     inner-training partition (not the full outer-training set);
* homology         : identity x bidirectional coverage of the best training
                     match, from the audit search (not the temporal search).

Metrics are computed on each outer test fold and summarised as mean and sample
SD over the five folds. P@10% uses K = round(0.1 x n_test); enrichment is P@10%
divided by the fold's positive rate.

Usage
-----
    python reproduce_table3.py --archive /path/to/release
    python reproduce_table3.py --archive /path/to/release --compare

Writes ``table3_recomputed.csv`` under ``--out-dir`` (see ``--help``).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from _paths import ARCHIVE as RELEASE, OUTPUT
SPLIT = "taxid_evidence_1116"
AA = "ACDEFGHIKLMNPQRSTVWY"
DINUC = [a + b for a in "ACGT" for b in "ACGT"]
C_GRID = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]

# Pre-6.2 values, retained only for historical comparison.
HISTORICAL_TABLE3 = {
    "protein":     (0.634, 0.082, 0.835, 0.864, 1.14),
    "dna":         (0.488, 0.074, 0.746, 0.661, 0.88),
    "concat":      (0.571, 0.071, 0.785, 0.744, 0.98),
    "composition": (0.567, 0.100, 0.810, 0.839, 1.11),
    "homology":    (0.566, 0.030, 0.819, 0.878, 1.16),
    "length_only": (0.484, 0.069, 0.765, 0.793, 1.05),
}
MODELS = [("protein", "ESM-2 pooled + linear head"),
          ("dna", "DNA pooled + linear head"),
          ("concat", "DNA + protein pooled, concatenated"),
          ("composition", "Composition + linear head"),
          ("homology", "Homology retrieval"),
          ("length_only", "Protein length only")]


def load_features() -> pd.DataFrame:
    """Feature blocks for the 1,116 evidence-linked records, indexed by record_id."""
    idx = pd.read_csv(RELEASE / "embeddings" / "cache_index.csv")
    prot = np.load(RELEASE / "embeddings" / "protein_pooled.float16.npy")
    pmask = np.load(RELEASE / "embeddings" / "protein_view_mask.bool.npy")
    dna = np.load(RELEASE / "embeddings" / "dna_pooled.float16.npy")
    dmask = np.load(RELEASE / "embeddings" / "dna_view_mask.bool.npy")

    pm = pmask.astype(np.float32)
    f_prot = (prot.astype(np.float32) * pm[..., None]).sum(1) / np.maximum(pm.sum(1), 1e-6)[:, None]

    dm = dmask.astype(np.float32)
    d_st = dna.astype(np.float32).mean(1)                  # mean over the two strands
    f_dna = (d_st * dm[..., None]).sum(1) / np.maximum(dm.sum(1), 1e-6)[:, None]

    seq = pd.read_csv(RELEASE / "sequences" / "sequences_original.csv.gz")
    seq = idx[["record_id"]].merge(seq, on="record_id", how="left")
    dna_seq = seq["dna_sequence_norm"].fillna("").astype(str)
    prt_seq = seq["protein_sequence_used"].fillna("").astype(str)

    comp = np.zeros((len(seq), 2 + 20 + 1 + 16), dtype=np.float32)
    for i, (ds, ps) in enumerate(zip(dna_seq, prt_seq)):
        ds_clean = "".join(c for c in ds.upper() if c in "ACGT")
        nd = max(len(ds_clean), 1)
        comp[i, 0] = nd
        comp[i, 1] = len(ps)
        comp[i, 2:22] = [ps.upper().count(a) / max(len(ps), 1) for a in AA]
        comp[i, 22] = (ds_clean.count("G") + ds_clean.count("C")) / nd
        comp[i, 23:39] = [ds_clean.count(dn) / max(nd - 1, 1) for dn in DINUC]

    return pd.DataFrame({
        "record_id": idx["record_id"],
        "label": idx["label"].astype(int),
        "protein": list(f_prot),
        "dna": list(f_dna),
        "concat": list(np.hstack([f_prot, f_dna])),
        "composition": list(comp),
        "length_only": list(np.asarray(idx["protein_len"], dtype=np.float32)[:, None]),
    }).set_index("record_id")


def load_homology() -> pd.Series:
    h = pd.read_csv(RELEASE / "dependency_audit" / f"homology_best_train_match_{SPLIT}.csv")
    score = h["best_identity"].fillna(0.0) * h["best_bidir_cov"].fillna(0.0)
    return pd.Series(score.values, index=h["test_record_id"].values, name="homology")


def load_roles() -> pd.DataFrame:
    """(fold, record) roles for the split. A record is outer_test in one fold and
    outer_train in the other four, so the manifest is kept in long format."""
    sp = pd.read_csv(RELEASE / "manifest" / "split_manifest.csv")
    sp = sp[sp.split_name == SPLIT][["record_id", "fold", "outer_role", "inner_role"]]
    assert sp.groupby("record_id").size().eq(5).all(), "each record needs five fold rows"
    return sp


def fit_predict(Xtr, ytr, Xva, yva, Xte) -> tuple[np.ndarray, float]:
    sc = StandardScaler().fit(Xtr)
    Ztr, Zva = sc.transform(Xtr), sc.transform(Xva)
    best_c, best_auc = C_GRID[0], -np.inf
    for C in C_GRID:
        clf = LogisticRegression(C=C, max_iter=5000, class_weight="balanced").fit(Ztr, ytr)
        auc = roc_auc_score(yva, clf.predict_proba(Zva)[:, 1])
        if auc > best_auc:
            best_c, best_auc = C, auc
    clf = LogisticRegression(C=best_c, max_iter=5000, class_weight="balanced").fit(Ztr, ytr)
    return clf.predict_proba(sc.transform(Xte))[:, 1], best_c


def evaluate(feat: pd.DataFrame, roles: pd.DataFrame, hom: pd.Series, col: str) -> pd.DataFrame:
    rows = []
    for fold in sorted(roles.fold.unique()):
        s = roles[roles.fold == fold]
        te_ids = s.loc[s.outer_role == "outer_test", "record_id"].values
        tr = s[s.outer_role == "outer_train"]
        fit_ids = tr.loc[tr.inner_role == "inner_train", "record_id"].values
        va_ids = tr.loc[tr.inner_role == "inner_validation", "record_id"].values

        y = feat.loc[te_ids, "label"].values
        if col == "homology":
            prob, chosen_c = hom.loc[te_ids].to_numpy(float), np.nan
        elif col == "length_only":
            # A single-feature baseline is ranked directly by the feature: fitting a
            # balanced logistic regression on one dimension lets the coefficient sign
            # flip on folds where the classes overlap, which would report the model's
            # instability rather than the feature's ranking power.
            prob, chosen_c = feat.loc[te_ids, col].to_numpy(float).ravel(), np.nan
        else:
            prob, chosen_c = fit_predict(
                np.vstack(feat.loc[fit_ids, col].values), feat.loc[fit_ids, "label"].values,
                np.vstack(feat.loc[va_ids, col].values), feat.loc[va_ids, "label"].values,
                np.vstack(feat.loc[te_ids, col].values))

        k = max(1, int(round(0.1 * len(te_ids))))
        p_at = float(y[np.argsort(-prob)[:k]].sum() / k)
        rows.append({"fold": fold, "n_test": len(te_ids), "n_pos": int(y.sum()),
                     "C_selected": chosen_c, "auroc": roc_auc_score(y, prob),
                     "ap": average_precision_score(y, prob),
                     "p_at_10pct": p_at, "enrichment": p_at / float(y.mean())})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", action="store_true", help="print the historical Table 3 beside the recomputation")
    args = ap.parse_args()

    feat, roles, hom = load_features(), load_roles(), load_homology()
    out = []
    all_folds = []
    for col, label in MODELS:
        pf = evaluate(feat, roles, hom, col)
        all_folds.append(pf.assign(model=col))
        out.append({"model": col, "label": label, "n_records": len(feat),
                    "auroc": pf.auroc.mean(), "auroc_sd": pf.auroc.std(ddof=1),
                    "ap": pf.ap.mean(), "ap_sd": pf.ap.std(ddof=1),
                    "p_at_10pct": pf.p_at_10pct.mean(),
                    "enrichment": pf.enrichment.mean(),
                    "C_selected": ",".join(f"{c:g}" for c in pf.C_selected.dropna().unique())})
        r = out[-1]
        print(f"{label:42s} AUROC {r['auroc']:.4f} ± {r['auroc_sd']:.4f}  "
              f"AP {r['ap']:.4f}  P@10% {r['p_at_10pct']:.4f}  Enr {r['enrichment']:.3f}")

    result = pd.DataFrame(out)
    result.to_csv(OUTPUT / "table3_recomputed.csv", index=False)
    pd.concat(all_folds, ignore_index=True).to_csv(OUTPUT / "table3_fold_metrics.csv", index=False)
    print(f"\nwritten: {OUTPUT / 'table3_recomputed.csv'}")

    if args.compare:
        print("\n=== historical Table 3 vs recomputation ===")
        rows = []
        for r in out:
            m = HISTORICAL_TABLE3[r["model"]]
            rows.append({"model": r["label"],
                         "AUROC (historical)": f"{m[0]:.3f} ± {m[1]:.3f}",
                         "AUROC (recomputed)": f"{r['auroc']:.3f} ± {r['auroc_sd']:.3f}",
                         "Δ AUROC": f"{r['auroc'] - m[0]:+.3f}",
                         "AP": f"{m[2]:.3f} → {r['ap']:.3f}",
                         "P@10%": f"{m[3]:.3f} → {r['p_at_10pct']:.3f}",
                         "Enrich": f"{m[4]:.2f} → {r['enrichment']:.2f}"})
        print(pd.DataFrame(rows).to_string(index=False))
        pd.DataFrame(rows).to_csv(OUTPUT / "table3_before_after.csv", index=False)


if __name__ == "__main__":
    main()
