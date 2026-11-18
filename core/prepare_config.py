"""Bind the code-only release to an external 6.2 evaluation archive."""
import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, required=True, help='External 6.2 release directory')
    p.add_argument('--embeddings', type=Path, help='Optional freshly encoded cache directory')
    p.add_argument('--output', type=Path, default=root / 'runtime_config.json')
    args = p.parse_args()
    archive = args.archive.resolve()
    cache = (args.embeddings or archive / 'embeddings').resolve()
    inputs = {'manifest': archive / 'manifest/cohort_manifest_audited.csv',
              'splits': archive / 'manifest/split_manifest.csv',
              'training_config': root / 'config/training.json'}
    arrays = {'dna': cache / 'dna_pooled.float16.npy',
              'dna_mask': cache / 'dna_view_mask.bool.npy',
              'protein': cache / 'protein_pooled.float16.npy',
              'protein_mask': cache / 'protein_view_mask.bool.npy',
              'protein_index': cache / 'cache_index.csv'}
    missing = [str(f) for f in [*inputs.values(), *arrays.values()] if not f.is_file()]
    if missing:
        p.error('Required external inputs are missing:\n' + '\n'.join(missing))
    if args.output.exists():
        p.error('Output configuration already exists; choose a new --output path')
    config = {k: str(v) for k, v in inputs.items()}
    config.update(cache={k: str(v) for k, v in arrays.items()},
                  split='taxid_evidence_1116', fold=0, seeds=[42, 43],
                  expected_input_sha256={k: sha256(v) for k, v in inputs.items()},
                  expected_cache_sha256={k: sha256(v) for k, v in arrays.items()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    print('Created:', args.output.resolve())
    print('Hashes record the selected files; use run_experiment.py --check to validate alignment and splits.')


if __name__ == '__main__':
    main()
