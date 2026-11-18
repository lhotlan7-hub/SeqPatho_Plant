"""Shared CLI paths; scripts read external inputs and write separate outputs."""
import argparse
import os
from pathlib import Path
import sys

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--archive', type=Path, default=os.environ.get('SEQPATHO_ARCHIVE'),
                    help='Path to the external 6.2 release archive (or SEQPATHO_ARCHIVE)')
parser.add_argument('--out-dir', type=Path,
                    help='Output directory; default: outputs/<script name>')
args, remaining = parser.parse_known_args()
if '-h' in remaining or '--help' in remaining:
    parser.print_help()
    if Path(sys.argv[0]).name == 'reproduce_table3.py':
        print('  --compare   Also compare against the historical, pre-6.2 table values.')
    raise SystemExit(0)
if args.archive is None:
    parser.error('--archive is required; this repository contains code only')
ARCHIVE = args.archive.resolve()
if not ARCHIVE.is_dir():
    parser.error('Archive directory does not exist: ' + str(ARCHIVE))
OUTPUT = (args.out_dir or Path(__file__).resolve().parent.parent / 'outputs' / Path(sys.argv[0]).stem).resolve()
if OUTPUT == ARCHIVE or OUTPUT.is_relative_to(ARCHIVE):
    parser.error('--out-dir must be outside the input archive')
OUTPUT.mkdir(parents=True, exist_ok=True)
sys.argv = [sys.argv[0], *remaining]
