#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from protocol_manifest_builder import (
    materialize_loso_fold_from_base,
    save_manifest,
    summarize_split,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize one source-pathogen LOSO fold.")
    parser.add_argument("--base_manifest", type=Path, required=True)
    parser.add_argument("--loso_manifest", type=Path, required=True)
    parser.add_argument("--fold_id", required=True)
    parser.add_argument("--out_manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base = pd.read_csv(args.base_manifest, low_memory=False)
    loso = pd.read_csv(args.loso_manifest, low_memory=False)

    fold_rows = loso[loso["fold_id"].astype(str) == str(args.fold_id)].copy()
    if len(fold_rows) == 0:
        raise RuntimeError(f"fold_id={args.fold_id} not found in {args.loso_manifest}")

    # If the LOSO manifest was generated in full row-level mode, just filter it.
    if {"record_id", "split", "clean_seq", "source_species_key"}.issubset(set(loso.columns)) and len(fold_rows) > 1:
        fold = fold_rows.copy()
    else:
        heldout = str(fold_rows.iloc[0]["heldout_species_key"])
        fold = materialize_loso_fold_from_base(
            base,
            heldout_species_key=heldout,
            fold_id=int(args.fold_id),
            seed=args.seed,
        )
        fold["protocol"] = "source_pathogen_loso"

    save_manifest(fold, args.out_manifest)
    summary_path = args.out_manifest.with_name(args.out_manifest.stem + "_summary.csv")
    summarize_split(fold).to_csv(summary_path, index=False)
    print(f"[write] {summary_path}")


if __name__ == "__main__":
    main()
