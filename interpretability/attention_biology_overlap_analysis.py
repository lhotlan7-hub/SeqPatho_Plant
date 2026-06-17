# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_CSV = ROOT / "paper_assets" / "attr_bio_overlap.csv"
DEFAULT_OUT_TEX = ROOT / "paper_assets" / "attr_bio_overlap.tex"

FEATURE_ORDER = [
    "Signal peptide",
    "RxLR motif",
    "CRN motif",
    "Pfam/InterPro domain",
    "Surface-exposed residues",
    "Transmembrane region",
    "Low-complexity region",
]

AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "SEC": "U",
    "PYL": "O",
}

MAX_ASA_TIEN = {
    "ALA": 129.0,
    "ARG": 274.0,
    "ASN": 195.0,
    "ASP": 193.0,
    "CYS": 167.0,
    "GLN": 225.0,
    "GLU": 223.0,
    "GLY": 104.0,
    "HIS": 224.0,
    "ILE": 197.0,
    "LEU": 201.0,
    "LYS": 236.0,
    "MET": 224.0,
    "PHE": 240.0,
    "PRO": 159.0,
    "SER": 155.0,
    "THR": 172.0,
    "TRP": 285.0,
    "TYR": 263.0,
    "VAL": 174.0,
}

VDW_RADII = {
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "S": 1.80,
    "P": 1.80,
    "H": 1.10,
}

FEATURE_ALIASES = {
    "signal": "Signal peptide",
    "signal peptide": "Signal peptide",
    "signal_peptide": "Signal peptide",
    "signalp": "Signal peptide",
    "sp": "Signal peptide",
    "rxlr": "RxLR motif",
    "rxlr motif": "RxLR motif",
    "rxlr-like motif": "RxLR motif",
    "crn": "CRN motif",
    "crn motif": "CRN motif",
    "lflak": "CRN motif",
    "hvlv": "CRN motif",
    "domain": "Pfam/InterPro domain",
    "domains": "Pfam/InterPro domain",
    "pfam": "Pfam/InterPro domain",
    "interpro": "Pfam/InterPro domain",
    "pfam/interpro": "Pfam/InterPro domain",
    "pfam/interpro domain": "Pfam/InterPro domain",
    "tm": "Transmembrane region",
    "tmh": "Transmembrane region",
    "transmembrane": "Transmembrane region",
    "transmembrane region": "Transmembrane region",
    "low complexity": "Low-complexity region",
    "low-complexity": "Low-complexity region",
    "low_complexity": "Low-complexity region",
    "low-complexity region": "Low-complexity region",
    "surface": "Surface-exposed residues",
    "surface exposed": "Surface-exposed residues",
    "surface-exposed": "Surface-exposed residues",
    "surface-exposed residues": "Surface-exposed residues",
    "sasa": "Surface-exposed residues",
}


@dataclass(frozen=True)
class ProteinRecord:
    gene: str
    sequence: str
    fasta_path: Path


@dataclass(frozen=True)
class Interval:
    gene: str
    feature_type: str
    start: int
    end: int
    source: str


@dataclass
class FeatureSet:
    intervals: List[Interval]
    tested_genes: Set[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate attr_bio_overlap.csv and a LaTeX table by comparing "
            "top-attribution residues against biological feature intervals."
        )
    )
    parser.add_argument(
        "--attributions",
        type=Path,
        default=None,
        help=(
            "CSV/TSV file with per-residue attribution scores. Required columns "
            "are gene, residue_index, and attribution_score/importance/score. "
            "Use --token-attributions instead when only DNA token importance "
            "CSV files are available."
        ),
    )
    parser.add_argument(
        "--token-attributions",
        nargs="*",
        default=None,
        metavar="GENE=CSV",
        help=(
            "Optional DNA token-importance mappings with columns idx, token, "
            "importance. Tokens are treated as non-overlapping CDS tokens and "
            "are averaged into protein residue scores by codon."
        ),
    )
    parser.add_argument(
        "--cds-fastas",
        nargs="*",
        default=None,
        metavar="GENE=FASTA",
        help=(
            "CDS FASTA mappings used with --token-attributions. Defaults to the "
            "repository Cmu1 and AVR3a DNA FASTA files when present."
        ),
    )
    parser.add_argument(
        "--cds-frame",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="1-based CDS frame used to map nucleotide tokens to residues. Default: 1.",
    )
    parser.add_argument(
        "--feature-annotations",
        type=Path,
        default=None,
        help=(
            "Optional CSV/TSV file with columns gene, feature_type, start, end. "
            "Use this for SignalP, Pfam/InterPro, TMHMM/DeepTMHMM, or manual "
            "curated features."
        ),
    )
    parser.add_argument(
        "--interpro-tsv",
        type=Path,
        default=None,
        help=(
            "Optional raw InterProScan TSV output. Rows with coordinates are "
            "treated as Pfam/InterPro domain intervals."
        ),
    )
    parser.add_argument(
        "--proteins",
        nargs="*",
        default=None,
        metavar="GENE=FASTA",
        help=(
            "Protein FASTA mappings. Defaults to the repository Cmu1 and AVR3a "
            "FASTA files when present."
        ),
    )
    parser.add_argument(
        "--pdbs",
        nargs="*",
        default=None,
        metavar="GENE=PDB",
        help=(
            "PDB mappings for surface-exposed residue analysis. Defaults to "
            "Cmu1 6FPF and AVR3a 2NAR when present."
        ),
    )
    parser.add_argument(
        "--genes",
        nargs="*",
        default=None,
        help="Restrict analysis to these genes. Defaults to genes in the attribution file.",
    )
    parser.add_argument(
        "--top-fraction",
        type=float,
        default=0.10,
        help="Top attribution fraction per gene. Ignored when --top-k is set. Default: 0.10.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Use the top K residues per gene instead of --top-fraction.",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=10000,
        help="Number of random residue-set permutations. Default: 10000.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed used for permutation testing. Default: 7.",
    )
    parser.add_argument(
        "--surface-rsa-cutoff",
        type=float,
        default=0.25,
        help="Residues with RSA >= this value are treated as surface-exposed. Default: 0.25.",
    )
    parser.add_argument(
        "--surface-points",
        type=int,
        default=96,
        help="Sphere points per atom for approximate Shrake-Rupley SASA. Default: 96.",
    )
    parser.add_argument(
        "--no-auto-signal",
        action="store_true",
        help=(
            "Disable the built-in N-terminal signal peptide heuristic. For final "
            "manuscript numbers, external SignalP annotations are recommended."
        ),
    )
    parser.add_argument(
        "--no-auto-motifs",
        action="store_true",
        help="Disable built-in RxLR and CRN motif scanning.",
    )
    parser.add_argument(
        "--no-auto-low-complexity",
        action="store_true",
        help="Disable built-in low-complexity scanning.",
    )
    parser.add_argument(
        "--no-surface",
        action="store_true",
        help="Disable surface-exposed residue analysis from PDB files.",
    )
    parser.add_argument(
        "--table-features",
        nargs="*",
        default=None,
        help=(
            "Feature rows to include in the LaTeX table. Defaults to all rows "
            "generated by the script, in a manuscript-friendly order."
        ),
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=DEFAULT_OUT_CSV,
        help=f"Output CSV path. Default: {DEFAULT_OUT_CSV}",
    )
    parser.add_argument(
        "--out-tex",
        type=Path,
        default=DEFAULT_OUT_TEX,
        help=f"Output LaTeX table path. Default: {DEFAULT_OUT_TEX}",
    )
    return parser.parse_args()


def sniff_sep(path: Path) -> str:
    if path.suffix.lower() in {".tsv", ".tab"}:
        return "\t"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        sample = handle.read(4096)
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t;").delimiter
    except csv.Error:
        return ","


def normalize_col(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def first_existing(candidates: Sequence[Path]) -> Optional[Path]:
    for path in candidates:
        if path.exists():
            return path
    return None


def parse_key_path_pairs(items: Optional[Sequence[str]]) -> Dict[str, Path]:
    pairs: Dict[str, Path] = {}
    if not items:
        return pairs
    for item in items:
        if "=" not in item:
            raise ValueError(f"expected GENE=PATH mapping, got: {item}")
        gene, raw_path = item.split("=", 1)
        gene = gene.strip()
        path = Path(raw_path.strip())
        if not path.is_absolute():
            path = ROOT / path
        pairs[gene] = path
    return pairs


def default_protein_paths() -> Dict[str, Path]:
    candidates = {
        "Cmu1": [
            ROOT / "Instantiation" / "downloads" / "Cmu1" / "protein" / "XP_011391476.1.fasta",
        ],
        "AVR3a": [
            ROOT / "Instantiation" / "downloads" / "AVR3a" / "protein" / "E2DWQ7.fasta",
        ],
    }
    return {gene: path for gene, paths in candidates.items() if (path := first_existing(paths))}


def default_pdb_paths() -> Dict[str, Path]:
    candidates = {
        "Cmu1": [
            ROOT / "Instantiation" / "downloads" / "Cmu1" / "pdb" / "6FPF" / "6FPF.pdb",
            ROOT / "Instantiation" / "downloads" / "Cmu1" / "pdb" / "6TI2" / "6TI2.pdb",
        ],
        "AVR3a": [
            ROOT / "Instantiation" / "downloads" / "AVR3a" / "pdb" / "2NAR" / "2NAR.pdb",
            ROOT
            / "Instantiation"
            / "downloads"
            / "AVR3a"
            / "pdb"
            / "alphafold"
            / "AF-E2DWQ7-F1-model_v6.pdb",
        ],
    }
    return {gene: path for gene, paths in candidates.items() if (path := first_existing(paths))}


def default_cds_paths() -> Dict[str, Path]:
    candidates = {
        "Cmu1": [
            ROOT / "Instantiation" / "downloads" / "Cmu1" / "dna" / "XM_011393174.1.fasta",
            ROOT / "Sample" / "single_query_XM_011393174_1.fasta",
        ],
        "AVR3a": [
            ROOT / "Instantiation" / "downloads" / "AVR3a" / "dna" / "GU258052.fasta",
            ROOT / "Instantiation" / "downloads" / "AVR3a" / "dna" / "GU258053.fasta",
            ROOT / "Instantiation" / "downloads" / "AVR3a" / "dna" / "GU258054.fasta",
        ],
    }
    return {gene: path for gene, paths in candidates.items() if (path := first_existing(paths))}


def read_first_fasta(path: Path) -> Tuple[str, str]:
    header = ""
    seq_parts: List[str] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seq_parts:
                    break
                header = line[1:].strip()
            else:
                seq_parts.append(line)
    if not seq_parts:
        raise ValueError(f"no sequence found in FASTA: {path}")
    return header, re.sub(r"[^A-Za-z]", "", "".join(seq_parts)).upper()


def load_proteins(paths: Mapping[str, Path], genes: Optional[Set[str]]) -> Dict[str, ProteinRecord]:
    proteins: Dict[str, ProteinRecord] = {}
    for gene, path in paths.items():
        if genes and gene not in genes:
            continue
        if not path.exists():
            raise FileNotFoundError(path)
        _, sequence = read_first_fasta(path)
        proteins[gene] = ProteinRecord(gene=gene, sequence=sequence, fasta_path=path)
    return proteins


def find_column(columns: Sequence[str], candidates: Sequence[str], required: bool = True) -> Optional[str]:
    normalized = {normalize_col(col): col for col in columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    if required:
        raise ValueError(
            "missing required column. Expected one of "
            f"{', '.join(candidates)}; observed columns: {', '.join(map(str, columns))}"
        )
    return None


def load_attributions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=sniff_sep(path))
    if df.empty:
        raise ValueError(f"attribution file is empty: {path}")

    gene_col = find_column(df.columns, ["gene", "gene_id", "protein", "protein_id", "id"])
    residue_col = find_column(
        df.columns,
        ["residue_index", "residue", "position", "pos", "aa_index", "resid", "idx"],
    )
    score_col = find_column(
        df.columns,
        [
            "attribution_score",
            "attribution",
            "importance",
            "score",
            "saliency",
            "ig",
            "integrated_gradients",
        ],
    )

    out = df[[gene_col, residue_col, score_col]].copy()
    out.columns = ["gene", "residue_index", "attribution_score"]
    out["gene"] = out["gene"].astype(str)
    out["residue_index"] = pd.to_numeric(out["residue_index"], errors="coerce")
    out["attribution_score"] = pd.to_numeric(out["attribution_score"], errors="coerce")
    out = out.dropna(subset=["gene", "residue_index", "attribution_score"])
    out["residue_index"] = out["residue_index"].astype(int)
    out = out[out["residue_index"] >= 1].copy()
    if out.empty:
        raise ValueError(f"no valid residue attribution rows found in {path}")

    return out


def load_token_attribution(
    gene: str,
    token_csv: Path,
    cds_fasta: Path,
    protein_len: int,
    frame: int,
) -> pd.DataFrame:
    """Convert non-overlapping DNA token importance to per-residue scores."""

    if not token_csv.exists():
        raise FileNotFoundError(token_csv)
    if not cds_fasta.exists():
        raise FileNotFoundError(cds_fasta)

    _, cds_seq = read_first_fasta(cds_fasta)
    cds_seq = cds_seq.replace("U", "T")
    df = pd.read_csv(token_csv, sep=sniff_sep(token_csv))
    if df.empty:
        raise ValueError(f"token attribution file is empty: {token_csv}")
    token_col = find_column(df.columns, ["token", "kmer", "word"])
    score_col = find_column(
        df.columns,
        ["importance", "attribution_score", "attribution", "score", "saliency"],
    )

    residue_score_sum: Dict[int, float] = defaultdict(float)
    residue_nt_count: Dict[int, int] = defaultdict(int)
    nt_pos = 1
    reconstructed: List[str] = []

    for _, row in df.iterrows():
        raw_token = str(row[token_col]).strip().upper().replace("U", "T")
        score = pd.to_numeric(row[score_col], errors="coerce")
        if pd.isna(score):
            continue
        # Skip tokenizer special symbols without moving the nucleotide cursor.
        if not raw_token or re.search(r"[^ACGTN]", raw_token):
            continue
        token_len = len(raw_token)
        reconstructed.append(raw_token)
        token_start = nt_pos
        token_end = nt_pos + token_len - 1
        for pos in range(token_start, token_end + 1):
            if pos < frame:
                continue
            residue_index = ((pos - frame) // 3) + 1
            if 1 <= residue_index <= protein_len:
                residue_score_sum[residue_index] += float(score)
                residue_nt_count[residue_index] += 1
        nt_pos += token_len

    token_seq = "".join(reconstructed)
    if token_seq and not cds_seq.startswith(token_seq[: min(len(token_seq), len(cds_seq))]):
        print(
            f"Warning: token sequence for {gene} does not match the CDS FASTA prefix; "
            "check orientation/frame if results look unexpected.",
            file=sys.stderr,
        )
    if token_seq and abs(len(token_seq) - len(cds_seq)) > 6:
        print(
            f"Warning: token coverage length for {gene} is {len(token_seq)} nt, "
            f"but CDS FASTA length is {len(cds_seq)} nt. This may be a windowed attribution file.",
            file=sys.stderr,
        )

    rows = []
    for residue_index in sorted(residue_score_sum):
        count = residue_nt_count[residue_index]
        if count:
            rows.append(
                {
                    "gene": gene,
                    "residue_index": residue_index,
                    "attribution_score": residue_score_sum[residue_index] / count,
                }
            )
    if not rows:
        raise ValueError(f"no residue scores could be derived from token file: {token_csv}")
    return pd.DataFrame(rows)


def canonical_feature(raw_feature: str) -> str:
    key = re.sub(r"[_\-]+", " ", str(raw_feature).strip().lower())
    key = re.sub(r"\s+", " ", key)
    return FEATURE_ALIASES.get(key, str(raw_feature).strip())


def add_interval(
    buckets: Dict[str, FeatureSet],
    gene: str,
    feature_type: str,
    start: int,
    end: int,
    source: str,
    seq_len: Optional[int] = None,
) -> None:
    if start > end:
        start, end = end, start
    start = max(1, int(start))
    end = int(end)
    if seq_len is not None:
        end = min(end, seq_len)
    if end < start:
        return
    feature_type = canonical_feature(feature_type)
    buckets[feature_type].intervals.append(Interval(gene, feature_type, start, end, source))
    buckets[feature_type].tested_genes.add(gene)


def mark_tested(buckets: Dict[str, FeatureSet], feature_type: str, genes: Iterable[str]) -> None:
    feature_type = canonical_feature(feature_type)
    buckets[feature_type].tested_genes.update(genes)


def load_generic_annotations(
    path: Optional[Path],
    proteins: Mapping[str, ProteinRecord],
    buckets: Dict[str, FeatureSet],
) -> None:
    if path is None:
        return
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, sep=sniff_sep(path))
    if df.empty:
        return
    gene_col = find_column(df.columns, ["gene", "gene_id", "protein", "protein_id", "id", "accession"])
    type_col = find_column(df.columns, ["feature_type", "feature", "type", "annotation"])
    start_col = find_column(df.columns, ["start", "begin", "from"])
    end_col = find_column(df.columns, ["end", "stop", "to"])

    for _, row in df.iterrows():
        gene = str(row[gene_col])
        if gene not in proteins:
            continue
        start = pd.to_numeric(row[start_col], errors="coerce")
        end = pd.to_numeric(row[end_col], errors="coerce")
        if pd.isna(start) or pd.isna(end):
            continue
        add_interval(
            buckets,
            gene,
            str(row[type_col]),
            int(start),
            int(end),
            source=path.name,
            seq_len=len(proteins[gene].sequence),
        )


def load_interpro_tsv(
    path: Optional[Path],
    proteins: Mapping[str, ProteinRecord],
    buckets: Dict[str, FeatureSet],
) -> None:
    if path is None:
        return
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 8:
                continue
            protein_id = parts[0]
            if protein_id not in proteins:
                continue
            try:
                start = int(parts[6])
                end = int(parts[7])
            except ValueError:
                continue
            source = "InterProScan"
            if len(parts) > 11 and parts[11]:
                source = f"InterProScan:{parts[11]}"
            elif len(parts) > 4 and parts[4]:
                source = f"InterProScan:{parts[4]}"
            add_interval(
                buckets,
                protein_id,
                "Pfam/InterPro domain",
                start,
                end,
                source=source,
                seq_len=len(proteins[protein_id].sequence),
            )


def hydrophobic_fraction(seq: str) -> float:
    if not seq:
        return 0.0
    hydrophobic = set("AILMFWV")
    return sum(1 for aa in seq if aa in hydrophobic) / len(seq)


def infer_signal_peptide(sequence: str) -> Optional[Tuple[int, int]]:
    """Small heuristic for secretory N termini.

    This is not a replacement for SignalP. It only provides a reproducible
    fallback so that preliminary tables can be generated before SignalP output
    is available.
    """

    n_term = sequence[:35]
    if len(n_term) < 18:
        return None
    best_end = None
    for start in range(2, 9):
        for end in range(start + 8, min(len(n_term), 24) + 1):
            segment = n_term[start - 1 : end]
            if hydrophobic_fraction(segment) >= 0.55:
                best_end = max(best_end or 0, end)
    if best_end is None:
        return None
    cleavage_end = min(len(sequence), min(max(best_end + 3, 18), 30))
    return 1, cleavage_end


def infer_motifs(
    proteins: Mapping[str, ProteinRecord],
    buckets: Dict[str, FeatureSet],
    enabled: bool,
) -> None:
    if not enabled:
        return
    genes = set(proteins)
    mark_tested(buckets, "RxLR motif", genes)
    mark_tested(buckets, "CRN motif", genes)

    for gene, record in proteins.items():
        seq = record.sequence
        # Canonical oomycete RxLR-like motif. The N-terminal limit avoids
        # unrelated internal R..LR matches dominating non-effector proteins.
        for match in re.finditer(r"R.LR", seq[:120]):
            add_interval(
                buckets,
                gene,
                "RxLR motif",
                match.start() + 1,
                match.end(),
                source="regex:R.LR",
                seq_len=len(seq),
            )
        # CRN effectors are commonly represented by LFLAK and/or HVLVxxP motifs.
        for pattern in [r"LFLAK", r"HVLV..P"]:
            for match in re.finditer(pattern, seq[:120]):
                add_interval(
                    buckets,
                    gene,
                    "CRN motif",
                    match.start() + 1,
                    match.end(),
                    source=f"regex:{pattern}",
                    seq_len=len(seq),
                )


def infer_signal_peptides(
    proteins: Mapping[str, ProteinRecord],
    buckets: Dict[str, FeatureSet],
    enabled: bool,
) -> None:
    if not enabled:
        return
    mark_tested(buckets, "Signal peptide", proteins.keys())
    for gene, record in proteins.items():
        interval = infer_signal_peptide(record.sequence)
        if interval:
            add_interval(
                buckets,
                gene,
                "Signal peptide",
                interval[0],
                interval[1],
                source="n_terminal_hydrophobic_heuristic",
                seq_len=len(record.sequence),
            )


def shannon_entropy(seq: str) -> float:
    if not seq:
        return 0.0
    counts = defaultdict(int)
    for aa in seq:
        counts[aa] += 1
    total = len(seq)
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


def merge_ranges(ranges: Sequence[Tuple[int, int]], max_gap: int = 1) -> List[Tuple[int, int]]:
    if not ranges:
        return []
    ordered = sorted((min(a, b), max(a, b)) for a, b in ranges)
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + max_gap + 1:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def infer_low_complexity(
    proteins: Mapping[str, ProteinRecord],
    buckets: Dict[str, FeatureSet],
    enabled: bool,
    window: int = 12,
    entropy_cutoff: float = 1.55,
    max_unique: int = 3,
) -> None:
    if not enabled:
        return
    mark_tested(buckets, "Low-complexity region", proteins.keys())
    for gene, record in proteins.items():
        seq = record.sequence
        ranges: List[Tuple[int, int]] = []
        if len(seq) < window:
            continue
        for i in range(0, len(seq) - window + 1):
            segment = seq[i : i + window]
            if len(set(segment)) <= max_unique or shannon_entropy(segment) <= entropy_cutoff:
                ranges.append((i + 1, i + window))
        for start, end in merge_ranges(ranges):
            add_interval(
                buckets,
                gene,
                "Low-complexity region",
                start,
                end,
                source="entropy_window",
                seq_len=len(seq),
            )


def fibonacci_sphere(n_points: int) -> np.ndarray:
    points = []
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(max(8, n_points)):
        y = 1.0 - (i / float(n_points - 1)) * 2.0 if n_points > 1 else 0.0
        radius = math.sqrt(max(0.0, 1.0 - y * y))
        theta = golden_angle * i
        x = math.cos(theta) * radius
        z = math.sin(theta) * radius
        points.append((x, y, z))
    return np.asarray(points, dtype=float)


def parse_pdb_atoms(path: Path) -> Dict[str, List[dict]]:
    chains: Dict[str, List[dict]] = defaultdict(list)
    seen_model = False
    with path.open("r", encoding="utf-8-sig", errors="ignore") as handle:
        for line in handle:
            record = line[:6].strip()
            if record == "MODEL":
                if seen_model:
                    break
                seen_model = True
            if record not in {"ATOM", "HETATM"}:
                continue
            if line[16] not in {" ", "A"}:
                continue
            resname = line[17:20].strip().upper()
            if resname not in AA3_TO_1:
                continue
            atom_name = line[12:16].strip()
            element = line[76:78].strip().upper() or re.sub(r"[^A-Za-z]", "", atom_name)[:1].upper()
            if element == "H":
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                resseq = int(line[22:26])
            except ValueError:
                continue
            chain = line[21].strip() or "_"
            insertion_code = line[26].strip()
            chains[chain].append(
                {
                    "coord": (x, y, z),
                    "element": element,
                    "residue_key": (chain, resseq, insertion_code, resname),
                    "resname": resname,
                    "resseq": resseq,
                }
            )
    return chains


def chain_residues(atoms: Sequence[dict]) -> List[Tuple[Tuple[str, int, str, str], str]]:
    residues: List[Tuple[Tuple[str, int, str, str], str]] = []
    seen: Set[Tuple[str, int, str, str]] = set()
    for atom in atoms:
        key = atom["residue_key"]
        if key in seen:
            continue
        seen.add(key)
        residues.append((key, AA3_TO_1[key[3]]))
    return residues


def map_chain_to_sequence(chain_seq: str, full_seq: str) -> Dict[int, int]:
    """Map 0-based chain residue indices to 1-based full-sequence positions."""

    start = full_seq.find(chain_seq)
    if start >= 0:
        return {i: start + i + 1 for i in range(len(chain_seq))}

    # Fallback dynamic programming alignment. The scoring favors contiguous
    # matches while allowing missing residues in crystallographic fragments.
    match_score = 2
    mismatch_score = -1
    gap_score = -2
    n = len(chain_seq)
    m = len(full_seq)
    score = np.zeros((n + 1, m + 1), dtype=int)
    trace = np.zeros((n + 1, m + 1), dtype=np.int8)
    for i in range(1, n + 1):
        score[i, 0] = score[i - 1, 0] + gap_score
        trace[i, 0] = 1
    for j in range(1, m + 1):
        score[0, j] = score[0, j - 1] + gap_score
        trace[0, j] = 2
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag = score[i - 1, j - 1] + (
                match_score if chain_seq[i - 1] == full_seq[j - 1] else mismatch_score
            )
            up = score[i - 1, j] + gap_score
            left = score[i, j - 1] + gap_score
            best = max(diag, up, left)
            score[i, j] = best
            trace[i, j] = 0 if best == diag else (1 if best == up else 2)

    mapping: Dict[int, int] = {}
    i, j = n, m
    while i > 0 or j > 0:
        direction = trace[i, j]
        if direction == 0:
            if i > 0 and j > 0 and chain_seq[i - 1] == full_seq[j - 1]:
                mapping[i - 1] = j
            i -= 1
            j -= 1
        elif direction == 1:
            i -= 1
        else:
            j -= 1
    return mapping


def compute_residue_sasa(atoms: Sequence[dict], n_points: int, probe_radius: float = 1.4) -> Dict:
    try:
        from scipy.spatial import cKDTree
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("surface analysis requires scipy.spatial.cKDTree") from exc

    if not atoms:
        return {}
    coords = np.asarray([atom["coord"] for atom in atoms], dtype=float)
    elements = [atom["element"] for atom in atoms]
    radii = np.asarray([VDW_RADII.get(element, 1.70) + probe_radius for element in elements])
    sphere = fibonacci_sphere(n_points)
    tree = cKDTree(coords)
    max_radius = float(radii.max())

    residue_sasa: Dict[Tuple[str, int, str, str], float] = defaultdict(float)
    for atom_index, atom in enumerate(atoms):
        center = coords[atom_index]
        radius = radii[atom_index]
        test_points = center + sphere * radius
        accessible = 0
        for point in test_points:
            buried = False
            for neighbor in tree.query_ball_point(point, max_radius):
                if neighbor == atom_index:
                    continue
                diff = point - coords[neighbor]
                if float(np.dot(diff, diff)) < float(radii[neighbor] ** 2):
                    buried = True
                    break
            if not buried:
                accessible += 1
        atom_sasa = 4.0 * math.pi * (radius**2) * accessible / len(test_points)
        residue_sasa[atom["residue_key"]] += atom_sasa
    return residue_sasa


def infer_surface_residues(
    proteins: Mapping[str, ProteinRecord],
    pdb_paths: Mapping[str, Path],
    buckets: Dict[str, FeatureSet],
    enabled: bool,
    rsa_cutoff: float,
    n_points: int,
) -> None:
    if not enabled:
        return
    for gene, record in proteins.items():
        pdb_path = pdb_paths.get(gene)
        if pdb_path is None or not pdb_path.exists():
            continue
        chains = parse_pdb_atoms(pdb_path)
        if not chains:
            continue
        mark_tested(buckets, "Surface-exposed residues", [gene])
        exposed_positions: Set[int] = set()
        for atoms in chains.values():
            residues = chain_residues(atoms)
            chain_seq = "".join(aa for _, aa in residues)
            if not chain_seq:
                continue
            chain_to_full = map_chain_to_sequence(chain_seq, record.sequence)
            if not chain_to_full:
                continue
            residue_key_to_local = {key: i for i, (key, _) in enumerate(residues)}
            sasa_by_key = compute_residue_sasa(atoms, n_points=n_points)
            for key, sasa in sasa_by_key.items():
                local_index = residue_key_to_local.get(key)
                if local_index is None or local_index not in chain_to_full:
                    continue
                max_asa = MAX_ASA_TIEN.get(key[3])
                if not max_asa:
                    continue
                rsa = sasa / max_asa
                if rsa >= rsa_cutoff:
                    exposed_positions.add(chain_to_full[local_index])
        for start, end in merge_ranges([(pos, pos) for pos in sorted(exposed_positions)], max_gap=0):
            add_interval(
                buckets,
                gene,
                "Surface-exposed residues",
                start,
                end,
                source=f"{pdb_path.name}:rsa>={rsa_cutoff:g}",
                seq_len=len(record.sequence),
            )


def intervals_to_positions(intervals: Sequence[Interval], gene: str) -> Set[int]:
    positions: Set[int] = set()
    for interval in intervals:
        if interval.gene != gene:
            continue
        positions.update(range(interval.start, interval.end + 1))
    return positions


def select_top_residues(
    attr: pd.DataFrame,
    genes: Iterable[str],
    top_fraction: float,
    top_k: Optional[int],
) -> Tuple[Dict[str, Set[int]], Dict[str, Set[int]]]:
    top_by_gene: Dict[str, Set[int]] = {}
    universe_by_gene: Dict[str, Set[int]] = {}
    for gene in genes:
        rows = attr[attr["gene"] == gene].copy()
        if rows.empty:
            continue
        grouped = (
            rows.groupby("residue_index", as_index=False)["attribution_score"]
            .max()
            .sort_values("attribution_score", ascending=False)
        )
        universe = set(int(x) for x in grouped["residue_index"])
        k = top_k if top_k is not None else int(math.ceil(len(grouped) * top_fraction))
        k = max(1, min(int(k), len(grouped)))
        top = set(int(x) for x in grouped.head(k)["residue_index"])
        top_by_gene[gene] = top
        universe_by_gene[gene] = universe
    return top_by_gene, universe_by_gene


def format_overlap(count: Optional[int], total: Optional[int]) -> str:
    if count is None or total in {None, 0}:
        return "--"
    rate = count / total
    return f"{count}/{total} ({rate:.1%})"


def format_random(mean_rate: Optional[float], sd_rate: Optional[float]) -> str:
    if mean_rate is None or sd_rate is None:
        return "--"
    return f"{mean_rate:.1%} $\\pm$ {sd_rate:.1%}"


def format_p_value(p_value: Optional[float]) -> str:
    if p_value is None or math.isnan(p_value):
        return "--"
    if p_value < 0.001:
        return "$<0.001$"
    return f"{p_value:.3f}"


def run_enrichment(
    feature_type: str,
    feature_set: FeatureSet,
    top_by_gene: Mapping[str, Set[int]],
    universe_by_gene: Mapping[str, Set[int]],
    n_permutations: int,
    rng: np.random.Generator,
) -> dict:
    genes = sorted(set(feature_set.tested_genes) & set(top_by_gene) & set(universe_by_gene))
    genes = [gene for gene in genes if top_by_gene[gene] and universe_by_gene[gene]]
    if not genes:
        return {
            "feature_type": feature_type,
            "genes_tested": 0,
            "top_overlap_count": None,
            "top_total": None,
            "top_overlap_rate": None,
            "random_overlap_mean": None,
            "random_overlap_sd": None,
            "p_value": None,
            "enrichment_ratio": None,
        }

    feature_positions = {
        gene: intervals_to_positions(feature_set.intervals, gene) & universe_by_gene[gene]
        for gene in genes
    }
    observed_count = sum(len(top_by_gene[gene] & feature_positions[gene]) for gene in genes)
    top_total = sum(len(top_by_gene[gene]) for gene in genes)
    observed_rate = observed_count / top_total if top_total else None

    random_counts = np.zeros(n_permutations, dtype=float)
    universe_arrays = {gene: np.asarray(sorted(universe_by_gene[gene]), dtype=int) for gene in genes}
    feature_lookup = {gene: feature_positions[gene] for gene in genes}

    for i in range(n_permutations):
        total = 0
        for gene in genes:
            universe = universe_arrays[gene]
            k = len(top_by_gene[gene])
            if k >= len(universe):
                sample = universe
            else:
                sample = rng.choice(universe, size=k, replace=False)
            total += sum(int(pos) in feature_lookup[gene] for pos in sample)
        random_counts[i] = total

    random_rates = random_counts / top_total if top_total else random_counts
    random_mean = float(random_rates.mean()) if top_total else None
    random_sd = float(random_rates.std(ddof=1)) if n_permutations > 1 and top_total else 0.0
    p_value = float((np.sum(random_counts >= observed_count) + 1) / (n_permutations + 1))
    enrichment_ratio = (
        float(observed_rate / random_mean)
        if observed_rate is not None and random_mean not in {None, 0.0}
        else None
    )
    return {
        "feature_type": feature_type,
        "genes_tested": len(genes),
        "top_overlap_count": observed_count,
        "top_total": top_total,
        "top_overlap_rate": observed_rate,
        "random_overlap_mean": random_mean,
        "random_overlap_sd": random_sd,
        "p_value": p_value,
        "enrichment_ratio": enrichment_ratio,
    }


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(ch, ch) for ch in str(text))


def write_latex_table(df: pd.DataFrame, path: Path, features: Optional[Sequence[str]]) -> None:
    if features:
        selected = [canonical_feature(feature) for feature in features]
        table_df = df[df["feature_type"].isin(selected)].copy()
        order = {feature: i for i, feature in enumerate(selected)}
        table_df["_order"] = table_df["feature_type"].map(order)
        table_df = table_df.sort_values("_order").drop(columns=["_order"])
    else:
        order = {feature: i for i, feature in enumerate(FEATURE_ORDER)}
        table_df = df.copy()
        table_df["_order"] = table_df["feature_type"].map(order).fillna(len(order))
        table_df = table_df.sort_values(["_order", "feature_type"]).drop(columns=["_order"])

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Biological plausibility analysis of high-attribution regions.}",
        r"\label{tab:attr_bio_overlap}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Feature type & Genes tested & Top-attribution overlap & Random overlap & $P$ value \\",
        r"\midrule",
    ]
    for _, row in table_df.iterrows():
        feature = latex_escape(row["feature_type"])
        genes_tested = int(row["genes_tested"]) if pd.notna(row["genes_tested"]) else "--"
        top_overlap = row["top_attribution_overlap"]
        random_overlap = row["random_overlap"]
        p_value = row["p_value_formatted"]
        lines.append(
            f"{feature} & {genes_tested} & {top_overlap} & {random_overlap} & {p_value} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_feature_sets(args: argparse.Namespace, proteins: Mapping[str, ProteinRecord]) -> Dict[str, FeatureSet]:
    buckets: Dict[str, FeatureSet] = defaultdict(lambda: FeatureSet(intervals=[], tested_genes=set()))
    load_generic_annotations(args.feature_annotations, proteins, buckets)
    load_interpro_tsv(args.interpro_tsv, proteins, buckets)
    infer_signal_peptides(proteins, buckets, enabled=not args.no_auto_signal)
    infer_motifs(proteins, buckets, enabled=not args.no_auto_motifs)
    infer_low_complexity(proteins, buckets, enabled=not args.no_auto_low_complexity)

    pdb_paths = default_pdb_paths()
    pdb_paths.update(parse_key_path_pairs(args.pdbs))
    infer_surface_residues(
        proteins,
        pdb_paths,
        buckets,
        enabled=not args.no_surface,
        rsa_cutoff=args.surface_rsa_cutoff,
        n_points=args.surface_points,
    )

    for feature in FEATURE_ORDER:
        buckets.setdefault(feature, FeatureSet(intervals=[], tested_genes=set()))
    return buckets


def build_table(args: argparse.Namespace) -> pd.DataFrame:
    residue_attr = load_attributions(args.attributions) if args.attributions else None
    token_attr_paths = parse_key_path_pairs(args.token_attributions)
    if residue_attr is None and not token_attr_paths:
        raise ValueError("provide either --attributions or --token-attributions")

    input_genes: Set[str] = set()
    if residue_attr is not None:
        input_genes.update(str(gene) for gene in residue_attr["gene"].unique())
    input_genes.update(token_attr_paths)
    requested_genes = set(args.genes) if args.genes else input_genes

    protein_paths = default_protein_paths()
    protein_paths.update(parse_key_path_pairs(args.proteins))
    proteins = load_proteins(protein_paths, requested_genes)
    if not proteins:
        raise ValueError(
            "no protein FASTA files were available for the requested genes. "
            "Use --proteins GENE=path/to/protein.fasta."
        )

    attr_parts: List[pd.DataFrame] = []
    if residue_attr is not None:
        attr_parts.append(residue_attr[residue_attr["gene"].isin(requested_genes)].copy())
    if token_attr_paths:
        cds_paths = default_cds_paths()
        cds_paths.update(parse_key_path_pairs(args.cds_fastas))
        for gene, token_csv in token_attr_paths.items():
            if gene not in requested_genes:
                continue
            if gene not in proteins:
                raise ValueError(f"cannot map token attribution for {gene}: missing protein FASTA")
            if gene not in cds_paths:
                raise ValueError(
                    f"cannot map token attribution for {gene}: missing CDS FASTA. "
                    "Use --cds-fastas GENE=path/to/cds.fasta."
                )
            attr_parts.append(
                load_token_attribution(
                    gene=gene,
                    token_csv=token_csv,
                    cds_fasta=cds_paths[gene],
                    protein_len=len(proteins[gene].sequence),
                    frame=args.cds_frame,
                )
            )

    attr = pd.concat(attr_parts, ignore_index=True) if attr_parts else pd.DataFrame()
    if attr.empty:
        raise ValueError("no attribution rows are available after filtering requested genes")

    genes = sorted(set(proteins) & set(attr["gene"].unique()))
    missing = sorted(requested_genes - set(proteins))
    if missing:
        print(
            "Warning: missing protein FASTA for genes: " + ", ".join(missing),
            file=sys.stderr,
        )
    if not genes:
        raise ValueError("no overlap between attribution genes and loaded protein FASTA genes")

    feature_sets = build_feature_sets(args, {gene: proteins[gene] for gene in genes})
    top_by_gene, universe_by_gene = select_top_residues(
        attr,
        genes=genes,
        top_fraction=args.top_fraction,
        top_k=args.top_k,
    )

    rng = np.random.default_rng(args.seed)
    rows = []
    for feature_type in FEATURE_ORDER:
        result = run_enrichment(
            feature_type,
            feature_sets[feature_type],
            top_by_gene,
            universe_by_gene,
            n_permutations=args.n_permutations,
            rng=rng,
        )
        result["top_attribution_overlap"] = format_overlap(
            result["top_overlap_count"], result["top_total"]
        )
        result["random_overlap"] = format_random(
            result["random_overlap_mean"], result["random_overlap_sd"]
        )
        result["p_value_formatted"] = format_p_value(result["p_value"])
        rows.append(result)

    out = pd.DataFrame(rows)
    return out[
        [
            "feature_type",
            "genes_tested",
            "top_overlap_count",
            "top_total",
            "top_overlap_rate",
            "random_overlap_mean",
            "random_overlap_sd",
            "p_value",
            "enrichment_ratio",
            "top_attribution_overlap",
            "random_overlap",
            "p_value_formatted",
        ]
    ]


def main() -> None:
    args = parse_args()
    table = build_table(args)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    write_latex_table(table, args.out_tex, args.table_features)
    print(f"Wrote {args.out_csv}")
    print(f"Wrote {args.out_tex}")


if __name__ == "__main__":
    main()
