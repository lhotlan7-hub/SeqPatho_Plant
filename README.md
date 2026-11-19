# SeqPatho-Plant: core implementation

Core implementation for **SeqPatho-Plant: A DNA-centered multi-view framework for plant pathogen gene prioritization**.

SeqPatho-Plant starts from nucleotide view construction and strand-aware aggregation, with an additional protein branch for paired input modeling and explicit candidate-budget evaluation. The repository contains the paired and protein-only heads, frozen sequence encoding, cached training, and the principal baseline and supplementary analyses. The code corresponds to the 1,116-record evidence-linked cohort with the frozen `taxid_evidence_1116` partition. The matched controls assess representation contributions; DNA-centered describes the design starting point, not an established advantage over the protein-only control.

This is a **code-only release**. Sequence data, manifests, cached embeddings, checkpoints, prediction archives, figures and manuscript files are external inputs or generated outputs. They are not included in this repository. The commands below require the separately held evaluation archive accompanying the manuscript; a public download of that archive is not provided by this release. See [DATA_FORMAT.md](DATA_FORMAT.md) for the input contract.

## Core files

| File | Purpose |
|---|---|
| `core/models.py` | Dual-branch head, protein-only head, shared training and validation-threshold selection |
| `core/encode_sequences.py` | Frozen DNABERT-2 and ESM-2 view encoding and CLS-query attention pooling |
| `core/prepare_config.py` | Bind an external archive/cache to a local configuration and record SHA-256 hashes |
| `core/run_experiment.py` | Matched training across selected frozen folds and seeds; saves checkpoints and predictions |
| `core/run_cached_fold.py` | Internal single-fold runner with hashes, record alignment and TaxID-disjointness checks |
| `analysis/reproduce_table3.py` | Protein/DNA/concatenation/composition, homology and length baselines |
| `analysis/reproduce_table4.py` | Four cached-feature adaptations: pLM4VF-style, FunVFPred AAC/AAC+DDE, NT-v2 |
| `analysis/run_pairing_shuffle.py` | Species/length-stratified pairing permutation control (Table S17) |
| `analysis/run_subgroup_analysis.py` | Paired subgroup and budget analyses (Table S16) |
| `analysis/check_subgroup_inputs.py` | Input alignment checks required by the subgroup analysis |

The architecture and training computation are preserved from the matched experiment. Configuration in `config/training.json` retains its 40-epoch maximum, patience 7, batch size 32 and learning rate 0.0002. The two seeds are 42 and 43; the actual initialization seed is `seed * 100 + fold`.

## Cached training

Use Python 3.10 and install the training/analysis dependencies:

```shell
python -m pip install -r requirements.txt
python core/prepare_config.py --archive /path/to/evaluation-archive
python core/run_experiment.py --config runtime_config.json --name check_inputs --folds 0,1,2,3,4 --check
python core/run_experiment.py --config runtime_config.json --name matched_run --folds 0,1,2,3,4 --device cpu
```

For a CUDA installation, select the appropriate PyTorch build for the hardware and use `--device cuda`. Commands also work in PowerShell; quote paths containing spaces. Each run name must be new. `--check` validates the inputs without training and records its per-fold configurations under `runs/`.

Training writes `runs/<name>/metrics.csv`, `oof.csv`, `summary.csv`, and per-fold checkpoints and provenance. `summary.csv` summarizes **fold–seed metrics**. Manuscript comparisons with single-fit baselines instead average each record's two seed scores before calculating fold metrics; these two summaries are different estimands.

Input hashes detect changes after configuration creation. They do not independently establish the provenance of a newly supplied archive. Preserve the archive's original manifests and row order; run without Python's `-O` flag because the historical runner uses assertions for invariants.

## Baselines and supplementary analyses

For analysis without model training, use a separate Python 3.12 environment with
`python -m pip install -r requirements-analysis.txt`. This records the analysis
environment used during release verification. See `VALIDATION.md` for numerical
agreement and the outstanding pairing-table discrepancy.

```shell
python analysis/reproduce_table3.py --archive /path/to/release
python analysis/reproduce_table4.py --archive /path/to/release
python analysis/run_pairing_shuffle.py --archive /path/to/release
python analysis/run_subgroup_analysis.py --archive /path/to/release
```

Outputs go to `outputs/<script-name>/`. Override this with `--out-dir`; input archives are never used as output directories. `SEQPATHO_ARCHIVE` can supply the archive path instead of `--archive`.

Table 3 uses inner-validation selection of logistic-regression C and fits on inner training only. Its length row ranks directly by protein length. `--compare` is an optional comparison with the **historical reference values**, retained from the original script. The computed results are the current analysis. Table S17 intentionally uses its own narrower C grid; it is not numerically interchangeable with the Table 3 concatenation row. Its permutations can retain fixed points and singletons.

The Table 4 script covers four cached-feature methods. Original DTVF/VirulentHunter training, effector-tool integrations, historical experiments and figure-generation utilities are outside this core-code selection.

## Optional sequence re-encoding

The verified historical encoder environment differs from the head-training environment: Python 3.9, PyTorch 2.3.1+cu121 and Transformers 4.49.0. Use a separate environment with `requirements-encoder.txt`. Local model snapshots and recorded revisions/weight hashes are described in `config/model_sources.json`.

```shell
python core/encode_sequences.py --manifest /path/to/release/manifest/cohort_manifest_audited.csv --sequences /path/to/release/sequences/sequences_original.csv.gz --dna-model /path/to/dnabert2-snapshot --protein-model /path/to/esm2-snapshot --out /path/to/new-cache --device cuda
python core/prepare_config.py --archive /path/to/release --embeddings /path/to/new-cache --output runtime_config_reencoded.json
```

The encoder requires local snapshots and loads DNABERT-2 custom model code. It verifies sequence hashes, creates ordered embeddings and refuses to overwrite a nonempty output directory. Re-encoding can change embeddings and training results across software/device environments; full retraining is distinct from recomputing archived predictions. Dependency lists capture observed environments, not a fully resolved cross-platform lockfile.

## Verification and provenance

```shell
python -m unittest discover -s tests -v
```

Tests cover masked-view behavior, protein-only independence from DNA, and both heads' training/checkpoint paths using synthetic inputs. See [VALIDATION.md](VALIDATION.md) for checks performed on this release, and [PROVENANCE.md](PROVENANCE.md) for source mapping and packaging changes.
