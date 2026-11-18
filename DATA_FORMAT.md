# External inputs

The required archive is the `release/` directory accompanying the manuscript. This code release does not include or download that archive. A fresh clone can run the synthetic tests; scientific analyses require the external inputs below.

## Matched training

- `manifest/cohort_manifest_audited.csv`: unique `record_id`, `label` (0/1), `pathogen_taxid`, `include_evidence_linked`, and `cache_row`. Filtering `include_evidence_linked == True` must yield the ordered 1,116 records with contiguous cache rows 0–1115.
- `manifest/split_manifest.csv`: `split_name`, `fold` (0–4), `record_id`, `row_index`, `label`, `outer_role`, `inner_role`. The selected split is `taxid_evidence_1116`. Each fold assigns each included record once to `inner_train`, `inner_validation` or `outer_test`. All three roles contain both classes and have disjoint TaxIDs.
- `embeddings/cache_index.csv`: `record_id`, `label`, `cache_row`, `dna_len`, `protein_len`; order must match the manifest.
- `embeddings/dna_pooled.float16.npy`: (1116, 2, 4, 768).
- `embeddings/dna_view_mask.bool.npy`: (1116, 4).
- `embeddings/protein_pooled.float16.npy`: (1116, 3, 640).
- `embeddings/protein_view_mask.bool.npy`: (1116, 3).

`config/training.json` is supplied by the repository. `prepare_config.py` records absolute external paths in an ignored local configuration so that no author's machine-specific paths are embedded in the source.

## Additional analysis inputs

Table 3 also reads `sequences/sequences_original.csv.gz` (`record_id`, `dna_sequence_norm`, `protein_sequence_used`) and `dependency_audit/homology_best_train_match_taxid_evidence_1116.csv`. Preserve the original homology table columns and record identifiers used by `load_homology()`.

Table 4 reads `table4_assets/source_index.csv`, `per_fold_seedavg.csv`, `esm2_mean_1022.npy`, `feat_aac.npy`, `feat_aac_dde.npy`, and `ntv2_mean.npy`, alongside the frozen split manifest. The original source index supplies `record_id`, `label` and contiguous `manifest_row`.

The pairing control uses embeddings, `dna_len`, manifest TaxIDs and the frozen split manifest.

The subgroup analysis additionally reads:

- `matched_comparison/by_split/taxid_evidence_1116_oof_predictions.csv`: 4,464 rows, one per record × model (`dual`, `protein`) × seed (42, 43); columns include `record_id`, `fold`, `seed`, `label`, `taxid`, `model`, `score`.
- `dependency_audit/homology_best_train_match_taxid_evidence_1116.csv`: including `test_record_id`, `fold`, `test_label`, `n_train_30_80`.
- `sequences/sequence_pair_audit.csv`: `record_id`, `dna_is_exact_cds`, matching that flag in the cohort manifest.

For raw-sequence encoding, the manifest also requires `dna_sha256_norm` and `protein_sha256_used`. The encoder matches the sequence CSV by `record_id`, checks the hashes and writes its own ordered cache index.
