# -*- coding: utf-8 -*-

import json
import time
from pathlib import Path
from typing import Dict, List, Set, Optional
import pandas as pd
import requests
import xml.etree.ElementTree as ET
from tqdm.auto import tqdm

# ========== 你只需改这里 ==========
IN_CSV   = r"E:\4PythonProject\SeqPatho\Data_processing\Nonpatho_all.csv"          # 输入
OUT_CSV  = r"E:\4PythonProject\SeqPatho\Data_processing\Nonpatho_plants_only.csv"  # 输出
CACHE    = r"taxonomy_cache.json"     # 缓存: tax_id -> 是否植物
EMAIL    = "d240201012@stu.cqupt.edu.cn"
API_KEY  = "c823e786be422dc3732df412d54e8e9e5a08"  # 有则填，没有留空
CHUNKSIZE = 20000
BATCH     = 150
SLEEP_SEC = 0.35
PLANT_CLADES = ["Viridiplantae", "Embryophyta", "Streptophyta", "Chlorophyta", "Charophyta"]
# ==================================

NCBI_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# 允许的核心列（Disease 为可选；首列可能叫 tax_id 或 #tax_id）
ALLOWED_COLS_BASE = {
    "tax_id","GeneID","status","RNA_nucleotide_accession.version",
    "protein_accession.version","genomic_nucleotide_accession.version",
    "start_position_on_the_genomic_accession","end_position_on_the_genomic_accession",
    "orientation","assembly","mature_peptide_accession.version",
    "Symbol","Species_Name","gene_sequence"
}
OPTIONAL_COLS = {"Disease"}
TAXID_ALIASES = ["tax_id", "#tax_id"]

def load_cache(path: Path) -> Dict[str, bool]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def save_cache(path: Path, cache: Dict[str, bool]):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)

def normalize_tax_id(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.isdigit():
        return s if int(s) > 0 else ""
    try:
        v = int(float(s))
        return str(v) if v > 0 else ""
    except Exception:
        return ""

def fetch_taxonomy_is_plant(
    ids: List[str],
    plant_terms_lower: Set[str],
    email: str = "",
    api_key: str = "",
) -> Dict[str, bool]:
    params = {
        "db": "taxonomy",
        "id": ",".join(ids),
        "retmode": "xml",
        "tool": "pgpred_filter",
    }
    if email:
        params["email"] = email
    if api_key:
        params["api_key"] = api_key

    r = requests.get(NCBI_EFETCH_URL, params=params, timeout=60)
    r.raise_for_status()

    out: Dict[str, bool] = {tid: False for tid in ids}
    root = ET.fromstring(r.text)

    for taxon in root.findall(".//Taxon"):
        taxid = (taxon.findtext("TaxId") or "").strip()
        lineage = (taxon.findtext("Lineage") or "").lower()
        lineage_ex_names = []
        for t in taxon.findall("LineageEx/Taxon"):
            nm = t.findtext("ScientificName")
            if nm:
                lineage_ex_names.append(nm.lower())
        text = " ; ".join([lineage] + lineage_ex_names)
        out[taxid] = any(term in text for term in plant_terms_lower)
    return out

def ensure_plant_flags(
    needed_ids: Set[str],
    cache: Dict[str, bool],
    plant_terms_lower: Set[str],
    email: str,
    api_key: str,
    batch: int,
    sleep_sec: float,
    pbar_tax: Optional[tqdm] = None,
):
    to_query = [tid for tid in needed_ids if tid and tid not in cache]
    if not to_query:
        return

    for i in range(0, len(to_query), batch):
        chunk = to_query[i:i+batch]
        for attempt in range(3):
            try:
                res = fetch_taxonomy_is_plant(chunk, plant_terms_lower, email, api_key)
                cache.update(res)
                if pbar_tax is not None:
                    pbar_tax.update(len(chunk))
                    pbar_tax.set_postfix_str(f"cached={len(cache)}")
                break
            except requests.HTTPError as e:
                wait = min(60, (attempt + 1) * 5)
                if pbar_tax is not None:
                    pbar_tax.write(f"[HTTPError] {e}. Retrying in {wait}s...")
                time.sleep(wait)
            except Exception as e:
                if pbar_tax is not None:
                    pbar_tax.write(f"[Warn] {e}. Retrying in 3s...")
                time.sleep(3)
        if not api_key and i + batch < len(to_query):
            time.sleep(sleep_sec)

def decide_columns(in_path: Path):
    """读取表头，决定用哪些列 & tax_id 列名"""
    header_cols = pd.read_csv(in_path, nrows=0).columns.tolist()
    header_cols = [str(c) for c in header_cols]

    # 选择 tax 列名
    tax_col = None
    for alias in TAXID_ALIASES:
        if alias in header_cols:
            tax_col = alias
            break
    if tax_col is None:
        raise ValueError(f"未找到 tax_id 列，支持的列名有: {TAXID_ALIASES}")

    # 允许的列 = 基础列 + (如果存在则包含 Disease)
    allowed = set(ALLOWED_COLS_BASE)
    if "Disease" in header_cols:
        allowed |= OPTIONAL_COLS

    # usecols = 文件中存在 ∩ 允许集合
    usecols = [c for c in header_cols if c in allowed or c == tax_col]
    if not usecols:
        raise ValueError("未匹配到任何需要的列，请检查CSV表头。")

    return tax_col, usecols

def main():
    in_path = Path(IN_CSV)
    out_path = Path(OUT_CSV)
    cache_path = Path(CACHE)

    if not in_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {in_path}")

    # 动态决定 tax 列名与 usecols（兼容 #tax_id / tax_id、是否含 Disease）
    tax_col, usecols = decide_columns(in_path)

    plant_terms_lower = set(t.strip().lower() for t in PLANT_CLADES if t.strip())
    cache: Dict[str, bool] = load_cache(cache_path)

    wrote_header = False
    total_in = 0
    total_out = 0
    save_every_n_rows = CHUNKSIZE * 5

    pbar_rows = tqdm(total=None, unit="rows", desc="Rows processed", mininterval=0.5)
    pbar_out  = tqdm(total=None, unit="rows", desc="Plant rows written", mininterval=0.5)
    pbar_tax  = tqdm(total=None, unit="taxid", desc="Taxonomy queried", mininterval=0.5)

    try:
        for chunk in pd.read_csv(
            in_path,
            chunksize=CHUNKSIZE,
            dtype=str,
            usecols=usecols,
            keep_default_na=False
        ):
            n_chunk = len(chunk)
            total_in += n_chunk
            pbar_rows.update(n_chunk)

            # 将 tax 列标准化为 tax_id_norm
            chunk["tax_id_norm"] = chunk[tax_col].map(normalize_tax_id)
            ids_in_chunk = set(chunk["tax_id_norm"].unique().tolist())
            ids_in_chunk.discard("")

            ensure_plant_flags(
                needed_ids=ids_in_chunk,
                cache=cache,
                plant_terms_lower=plant_terms_lower,
                email=EMAIL,
                api_key=API_KEY,
                batch=BATCH,
                sleep_sec=SLEEP_SEC,
                pbar_tax=pbar_tax
            )

            chunk["is_plant"] = chunk["tax_id_norm"].map(lambda x: cache.get(x, False))
            keep = chunk[chunk["is_plant"]].drop(columns=["tax_id_norm", "is_plant"])

            if not wrote_header:
                keep.to_csv(out_path, index=False, mode="w", encoding="utf-8")
                wrote_header = True
            else:
                keep.to_csv(out_path, index=False, mode="a", header=False, encoding="utf-8")

            total_out += len(keep)
            pbar_out.update(len(keep))
            pbar_rows.set_postfix_str(f"cached={len(cache)} out={total_out}")

            if total_in % save_every_n_rows < CHUNKSIZE:
                save_cache(cache_path, cache)
                pbar_rows.write(f"[保存缓存] {len(cache)} 个 tax_id -> {cache_path}")

    finally:
        save_cache(cache_path, cache)
        pbar_rows.close()
        pbar_out.close()
        pbar_tax.close()

    print(f"[完成] 输入总行数: {total_in}; 输出(植物)行数: {total_out}")
    print(f"[缓存] 已写入: {cache_path.resolve()}")
    print(f"[输出] 植物 CSV: {out_path.resolve()}")

if __name__ == "__main__":
    main()
