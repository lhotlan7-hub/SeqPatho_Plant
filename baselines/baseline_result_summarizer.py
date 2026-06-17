# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from baseline_common import OUT_ROOT, save_df


def read_if_exists(path: Path):
    if path.exists():
        return pd.read_csv(path)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Combine SeqPatho supplemental baseline summaries.")
    parser.add_argument("--output-root", default=str(OUT_ROOT))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    root = Path(args.output_root)
    frames = []
    for path in root.glob("results_*"):
        candidates = [
            path / "master_baseline_summary.csv",
            path / "master_dnabert_baseline_summary.csv",
            path / "standard_species_disjoint_summary.csv",
            path / "standard_species_disjoint_dnabert_summary.csv",
        ]
        for cand in candidates:
            df = read_if_exists(cand)
            if df is None:
                continue
            df.insert(0, "source_run", path.name)
            df.insert(1, "source_file", str(cand.relative_to(root)))
            frames.append(df)
    out_df = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    out_path = Path(args.out) if args.out else root / "combined_baseline_summary.csv"
    save_df(out_df, out_path, out_path.with_suffix(".md"))
    print("Wrote %s with %d rows" % (out_path, len(out_df)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
