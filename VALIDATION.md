# Release checks — 18 November 2026

Checks use the local manuscript 6.2 archive as external input. Data and generated results are not committed.

## Passed

- All Python files compile.
- Three synthetic behavioral tests pass: masked-softmax handling, protein-only invariance to DNA and masked protein views, and training/checkpoint save-and-load for both heads.
- All five primary folds pass file hashes, record alignment, label alignment, representation dimensions and train/validation/test TaxID-disjointness checks through the packaged CLI.
- A real-data CPU CLI smoke run on fold 0, limited to one epoch, completes all four model/seed combinations and writes checkpoints, metrics and 2,036 predictions. This is an execution check, not a full retraining result.
- Model and encoder sources are text-identical to the source files apart from line endings. The non-main computational functions in Table 3, pairing and subgroup scripts have identical Python ASTs to the manuscript archive scripts.
- Table 3's six rows execute successfully. Mean AUROC values agree with the saved recomputation within 0.0015; individual metrics need not be bit-identical.
- Table 4's four supported methods pass all 20 per-fold AUROC checks at the original tolerance of 0.0015.
- The subgroup analysis, including all five subgroup definitions and 5,000 bootstrap draws, executes successfully and reproduces the saved numeric table.

## Pairing-table discrepancy retained for investigation

The packaged script and the **unmodified original** `run_pairing_shuffle.py` produce exactly the same output when executed in the same checked environment. That output differs from the archived `pairing_shuffle.csv`:

| Quantity | Archived table | Current execution |
|---|---:|---:|
| Protein only AUROC | 0.627946 | 0.627816 |
| True pairing AUROC | 0.569757 | 0.569799 |
| Mean over shuffled pairings | approximately 0.6073 | approximately 0.6086 |
| Shuffled-pairing range | approximately 0.5825–0.6240 | approximately 0.5701–0.6354 |

The maximum per-condition AUROC difference is approximately 0.03637. The cause has not been established. This release preserves the original calculation and does not claim exact reproduction of Table S17. No manuscript or archived results were changed to match the rerun.

## Environments and scope

- Head-training checks: Python 3.10, PyTorch 2.11.0+cu128, NumPy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2; tests used CPU.
- Analysis-only checks: Python 3.12.4, NumPy 1.26.4, pandas 2.2.2, SciPy 1.13.1, scikit-learn 1.4.2. BLAS/OpenMP thread counts were set to one.
- No full neural five-fold retraining, model download, raw-sequence re-encoding or clean-environment dependency installation was performed as part of this code publication.
