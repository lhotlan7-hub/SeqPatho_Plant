# Source provenance

This document describes the sources of the core implementation. `PROVENANCE.json` records upstream file hashes and relative source labels for the copied code.

| Repository component | Source |
|---|---|
| Models and shared training | `20261102_primary_retraining/code/models.py` |
| Single-fold and multi-fold training | `20261102_primary_retraining/code/run_cached_fold.py`, `run_experiment.py` |
| Frozen sequence encoding | `20261102_primary_retraining/code/encode_sequences.py` |
| Training settings | Model, improvement and frozen-training sections of `20261102_primary_retraining/inputs/experiment_config.json` |
| Baselines and supplementary analyses | the manuscript evaluation archive (`release/`) |

The model source matches `20261029_135100_PrimaryCohortAudit_ProteinOnly/code/models.py` except for the optional checkpoint-saving parameter and save operation. Its mathematical computation is unchanged. The source experiment's primary OOF prediction file and the evaluation archive's corresponding file have the same SHA-256.

Packaging changes:

1. Keep the model and encoder source unchanged apart from line endings.
2. Adapt training script locations, add explicit device selection and a validation-only option; preserve optimization, seeds, thresholds and metrics.
3. Add `prepare_config.py` to bind external inputs and compute their hashes.
4. Add analysis `--archive`/`--out-dir` paths; preserve statistical calculations and send writes outside the source archive.
5. Label Table 3's old comparison constants as historical; clarify that the pairing permutations can retain fixed points. Neither change affects computation.
6. Include configuration, dependency files, input documentation and synthetic tests. Exclude datasets, feature caches, model weights, predictions and manuscript assets.

The evaluation archive is an external input. Code publication does not constitute a complete rerun of all manuscript experiments.
