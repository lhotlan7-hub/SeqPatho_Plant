#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import os
import json
import time
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")  # 非交互式后端
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import requests

# =====================
# USER CONFIG（请在此处填写）
# =====================
POS_CSV: str = r"E:\4PythonProject\SeqPatho\Dataset\genes_dedup.csv"
NEG_CSV: str = r"E:\4PythonProject\SeqPatho\Dataset\NONgenes_dedup2.csv"
OUTDIR: str = r""

NCBI_EMAIL: Optional[str] = None
NCBI_API_KEY: Optional[str] = None
PREFER_NCBI: bool = False   # True=全部以 NCBI 学名覆盖；False=仅在缺失时补齐

# 绘图参数
TOP_N: int = 20
YBREAK: bool = True
YBREAK_FACTOR: float = 2.5   # 断轴触发条件：max >= factor * second
YBREAK_MIN_TOP: int = 100    # 断轴触发的最小绝对阈值
XTICK_LABELSIZE: int = 13     # 竖排标签字号
BOTTOM_MARGIN: float = 0.34  # 竖排标签底部边距（0~1）

# 出图仅使用英文（物种名、坐标轴、标题、标注）
PLOT_ENGLISH_ONLY: bool = True

# =====================
# 字体设置（英文优先；如需中文可切 english_only=False）
# =====================
def _setup_chinese_font(english_only: bool = True) -> None:
    """
    english_only=True 时：使用通用英文字体（DejaVu Sans 优先），不做中文字体探测。
    english_only=False 时：自动探测可用中文字体。
    """
    if english_only:
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Liberation Sans"]
        plt.rcParams["axes.unicode_minus"] = False  # 负号正常显示
        return

    candidates = [
        "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC",
        "PingFang SC", "Hiragino Sans GB", "WenQuanYi Zen Hei", "Arial Unicode MS",
        "Songti SC", "STHeiti", "DejaVu Sans"
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    chosen = next((n for n in candidates if n in available), "DejaVu Sans")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [chosen]
    plt.rcParams["axes.unicode_minus"] = False  # 负号正常显示

_setup_chinese_font(english_only=PLOT_ENGLISH_ONLY)

# ---------------------------
# 工具函数
# ---------------------------
def _ensure_outdir(path: str) -> None:
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

def _read_minimal_csv(csv_path: str) -> pd.DataFrame:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"未找到文件: {csv_path}")
    with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
        header_raw = f.readline().rstrip('\n').rstrip('\r')
    header = header_raw.split(',')
    header_fixed = [c.replace('#tax_id', 'tax_id') for c in header]
    if 'tax_id' not in header_fixed:
        raise ValueError(f"文件 {csv_path} 缺少 'tax_id' 列")
    has_species = 'Species_Name' in header_fixed
    usecols_names = ['tax_id'] + (['Species_Name'] if has_species else [])
    usecols_idx = [header_fixed.index(c) for c in usecols_names]
    df = pd.read_csv(csv_path, usecols=usecols_idx, header=0, encoding='utf-8', low_memory=False)
    df.columns = usecols_names
    df['tax_id'] = pd.to_numeric(df['tax_id'], errors='coerce').astype('Int64')
    if has_species:
        df['Species_Name'] = df['Species_Name'].astype(str).fillna('').str.strip()
    else:
        df['Species_Name'] = ''
    n_before = len(df)
    df = df.dropna(subset=['tax_id'])
    n_drop = n_before - len(df)
    if n_drop > 0:
        print(f"[WARN] {csv_path}: 丢弃 {n_drop} 行缺失 tax_id 的记录")
    return df[['tax_id', 'Species_Name']].reset_index(drop=True)

# ---------------------------
# NCBI Tax Resolver（带缓存）
# ---------------------------
class NCBITaxResolver:
    def __init__(
        self,
        cache_path: str,
        email: Optional[str] = None,
        api_key: Optional[str] = None,
        batch_size: int = 400,
        sleep_no_key: float = 0.34,
        sleep_with_key: float = 0.15
    ):
        self.cache_path = cache_path
        self.email = email
        self.api_key = api_key
        self.batch_size = batch_size
        self.sleep_no_key = sleep_no_key
        self.sleep_with_key = sleep_with_key
        self.cache: Dict[str, Optional[str]] = self._load_cache()

    def _load_cache(self) -> Dict[str, Optional[str]]:
        if os.path.isfile(self.cache_path):
            try:
                with open(self.cache_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[WARN] 无法读取缓存 {self.cache_path}: {e}")
        return {}

    def _save_cache(self) -> None:
        tmp = self.cache_path + ".tmp"
        with open(tmp, 'w', encoding='utf-8', newline='\n') as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.cache_path)

    def resolve_many(self, tax_ids: List[int]) -> Dict[int, Optional[str]]:
        unresolved = [str(t) for t in tax_ids if (t is not None and str(t) not in self.cache)]
        if not unresolved:
            return {int(k): v for k, v in self.cache.items() if k.isdigit()}
        print(f"[NCBI] 需要解析 {len(unresolved)} 个未缓存 tax_id，批量查询...")
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        params_common = {"db": "taxonomy", "retmode": "json"}
        if self.email:
            params_common["email"] = self.email
        if self.api_key:
            params_common["api_key"] = self.api_key
            sleep = self.sleep_with_key
        else:
            sleep = self.sleep_no_key
        for i in tqdm(range(0, len(unresolved), self.batch_size), desc="NCBI esummary"):
            batch = unresolved[i:i+self.batch_size]
            params = {**params_common, "id": ",".join(batch)}
            try:
                r = requests.get(base_url, params=params, timeout=30)
                r.raise_for_status()
                data = r.json()
                result = data.get('result', {})
                uids = result.get('uids', [])
                for uid in uids:
                    rec = result.get(uid, {})
                    sci = rec.get('scientificname') or rec.get('scientificname_in_ref') or rec.get('title')
                    if sci:
                        self.cache[str(uid)] = sci
                # 标记未命中
                for mid in set(batch) - set(uids):
                    self.cache[mid] = None
            except Exception as e:
                print(f"[WARN] 请求失败（批次起始 {i}）：{e}")
                for mid in batch:
                    self.cache.setdefault(mid, None)
            finally:
                time.sleep(sleep)
        self._save_cache()
        return {int(k): v for k, v in self.cache.items() if k.isdigit()}

# ---------------------------
# 统计与绘图
# ---------------------------
def _normalize_species_name(name: str) -> str:
    return ' '.join(str(name).strip().split()) if isinstance(name, str) else ''

def fill_species_names(df: pd.DataFrame, resolver: NCBITaxResolver, prefer_ncbi: bool = False) -> pd.DataFrame:
    df = df.copy()
    df['Species_Name'] = df['Species_Name'].astype(str).apply(_normalize_species_name)
    need_ncbi_mask = df['Species_Name'].eq('') | df['Species_Name'].str.lower().isin({'nan','none','null','na'})
    need_ids = sorted(df.loc[need_ncbi_mask, 'tax_id'].dropna().astype(int).unique().tolist())
    if prefer_ncbi:
        need_ids = sorted(df['tax_id'].dropna().astype(int).unique().tolist())
    if need_ids:
        mapping = resolver.resolve_many(need_ids)
        def map_name(row):
            if prefer_ncbi or row['Species_Name'] == '':
                sci = mapping.get(int(row['tax_id']))
                if isinstance(sci, str) and sci.strip():
                    return sci.strip()
            return row['Species_Name']
        df['Species_Name'] = df.apply(map_name, axis=1)
    df['Species_Name'] = df['Species_Name'].apply(lambda s: s if s.strip() else "Unknown")
    return df

def compute_stats(species_series: pd.Series) -> Tuple[pd.DataFrame, int]:
    counts = species_series.value_counts(dropna=False)
    total = int(counts.sum())
    df = counts.rename_axis('Species_Name').reset_index(name='count')
    df['proportion'] = df['count'] / total if total > 0 else 0.0
    return df, total

def write_stats_txt(df_stats: pd.DataFrame, total: int, path: str, title: str) -> None:
    with open(path, 'w', encoding='utf-8-sig', newline='\n') as f:
        f.write(f"# {title}\n")
        f.write(f"Total Records: {total}\n")
        f.write("Rank\tSpecies_Name\tCount\tProportion(%)\n")
        for i, row in enumerate(df_stats.itertuples(index=False), start=1):
            f.write(f"{i}\t{row.Species_Name}\t{row.count}\t{row.proportion*100:.2f}\n")

def write_stats_csv(df_stats: pd.DataFrame, path: str) -> None:
    df_stats.assign(proportion_percent=(df_stats['proportion']*100).round(4)) \
            .to_csv(path, index=False, encoding='utf-8-sig', lineterminator='\n')

def save_topN_bar(
    df_stats: pd.DataFrame,
    path_png: str,
    title: str,
    N: int = TOP_N,
    ybreak: bool = YBREAK,
    factor: float = YBREAK_FACTOR,
    min_top: int = YBREAK_MIN_TOP,
    english_only: bool = True
) -> None:
    """
    出图仅英文：english_only=True 时直接使用 Species_Name（学名）；
    english_only=False 时可在 name_map 中自定义英文->中文映射（此处留空占位）。
    """
    name_map: Dict[str, str] = {
        # "Arabidopsis thaliana": "拟南芥",
        # （如需中文再填充）
    }

    df_stats = df_stats.copy()
    topN = df_stats.head(N)

    # 物种名：仅英文时直接使用原始学名；否则尝试映射为中文
    if english_only:
        names = list(topN['Species_Name'])
    else:
        names = [name_map.get(n, n) for n in topN['Species_Name']]

    counts = topN['count'].astype(float).to_numpy()
    props = topN['proportion'].to_numpy()

    fig_w = max(12, 0.45 * len(topN) + 8)
    max_c = float(np.max(counts)) if len(counts) > 0 else 0.0
    second_c = float(np.partition(counts, -2)[-2]) if len(counts) >= 2 else 0.0
    use_break = (ybreak and len(topN) >= 2 and max_c >= min_top and second_c > 0 and max_c >= factor * second_c)

    if not use_break:
        fig, ax = plt.subplots(figsize=(fig_w, 7))
        bars = ax.bar(names, counts)
        ax.set_title(title)
        ax.set_xlabel(f"Top {N} species")
        ax.set_ylabel("Count")
        ax.tick_params(axis='x', rotation=90, labelsize=XTICK_LABELSIZE)
        for rect, p in zip(bars, props):
            h = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2, h, f"{p*100:.2f}%", ha='center', va='bottom', fontsize=9)
        fig.subplots_adjust(bottom=BOTTOM_MARGIN)
        fig.savefig(path_png, dpi=500)
        plt.close(fig)
        return

    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(fig_w, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 3], hspace=0.05)
    ax_high = fig.add_subplot(gs[0])
    ax_low = fig.add_subplot(gs[1], sharex=ax_high)

    lower_max = max(1.0, second_c * 1.1)
    upper_min = max(lower_max * 1.02, max_c * 0.9)
    ax_low.set_ylim(0, lower_max)
    ax_high.set_ylim(upper_min, max_c * 1.05)

    ax_low.bar(names, counts)
    ax_high.bar(names, counts)

    ax_high.tick_params(labelbottom=False)
    ax_low.tick_params(axis='x', rotation=90, labelsize=XTICK_LABELSIZE)
    ax_high.set_ylabel('Count')
    ax_low.set_ylabel('Count')
    ax_low.set_xlabel(f"Top {N} species")
    fig.suptitle(title)

    # 断裂标记
    d = .008
    kwargs = dict(transform=ax_high.transAxes, color='k', clip_on=False)
    ax_high.plot((-d, +d), (-d, +d), **kwargs)
    ax_high.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    kwargs.update(transform=ax_low.transAxes)
    ax_low.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_low.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    for i, (c, p) in enumerate(zip(counts, props)):
        (ax_low if c <= lower_max else ax_high).text(i, c, f"{p*100:.2f}%", ha='center', va='bottom', fontsize=9)

    fig.subplots_adjust(bottom=BOTTOM_MARGIN)
    fig.savefig(path_png, dpi=500)
    plt.close(fig)

def export_taxid_mapping(dfs: List[pd.DataFrame], outdir: str) -> None:
    mapping: Dict[int, set] = {}
    for df in dfs:
        for t, s in zip(df['tax_id'].astype(int), df['Species_Name']):
            mapping.setdefault(int(t), set()).add(s)
    pd.DataFrame(
        [{"tax_id": t, "Species_Names_found": "; ".join(sorted(v))} for t, v in mapping.items()]
    ).sort_values('tax_id').to_csv(
        os.path.join(outdir, 'taxid_to_species.csv'),
        index=False, encoding='utf-8-sig', lineterminator='\n'
    )

# ---------------------------
# 主流程
# ---------------------------
def main() -> None:
    if not POS_CSV or not os.path.isfile(POS_CSV):
        raise FileNotFoundError("请正确设置 POS_CSV")
    if not NEG_CSV or not os.path.isfile(NEG_CSV):
        raise FileNotFoundError("请正确设置 NEG_CSV")

    outdir = OUTDIR.strip() or os.path.join(os.path.dirname(POS_CSV), 'species_stats_output')
    _ensure_outdir(outdir)

    resolver = NCBITaxResolver(os.path.join(outdir, 'ncbi_tax_cache.json'), NCBI_EMAIL, NCBI_API_KEY)

    print("[LOAD] 读取 CSV...")
    df_pos_raw = _read_minimal_csv(POS_CSV)
    df_neg_raw = _read_minimal_csv(NEG_CSV)

    print("[NCBI] 填充/校正物种学名...")
    df_pos = fill_species_names(df_pos_raw, resolver, PREFER_NCBI)
    df_neg = fill_species_names(df_neg_raw, resolver, PREFER_NCBI)
    df_all = pd.concat([df_pos, df_neg], ignore_index=True)

    print("[STATS] 统计阳性数据集 ...")
    pos_stats, pos_total = compute_stats(df_pos['Species_Name'])
    print("[STATS] 统计阴性数据集 ...")
    neg_stats, neg_total = compute_stats(df_neg['Species_Name'])
    print("[STATS] 统计合并数据集 ...")
    all_stats, all_total = compute_stats(df_all['Species_Name'])

    write_stats_txt(pos_stats, pos_total, os.path.join(outdir, 'species_stats_positive.txt'), 'Positive Dataset Species Stats')
    write_stats_csv(pos_stats, os.path.join(outdir, 'species_stats_positive.csv'))

    write_stats_txt(neg_stats, neg_total, os.path.join(outdir, 'species_stats_negative.txt'), 'Negative Dataset Species Stats')
    write_stats_csv(neg_stats, os.path.join(outdir, 'species_stats_negative.csv'))

    write_stats_txt(all_stats, all_total, os.path.join(outdir, 'species_stats_combined.txt'), 'Combined Dataset Species Stats')
    write_stats_csv(all_stats, os.path.join(outdir, 'species_stats_combined.csv'))

    export_taxid_mapping([df_pos, df_neg], outdir)

    print(f"[PLOT] 绘制 Top-{TOP_N} 柱状图（英文） ...")
    save_topN_bar(
        pos_stats,
        os.path.join(outdir, f'top{TOP_N}_positive.png'),
        f'Top-{TOP_N} species (positive)',
        english_only=PLOT_ENGLISH_ONLY
    )
    save_topN_bar(
        neg_stats,
        os.path.join(outdir, f'top{TOP_N}_negative.png'),
        f'Top-{TOP_N} species (negative)',
        english_only=PLOT_ENGLISH_ONLY
    )
    save_topN_bar(
        all_stats,
        os.path.join(outdir, f'top{TOP_N}_combined.png'),
        f'Top-{TOP_N} species (combined)',
        english_only=PLOT_ENGLISH_ONLY
    )

    print("[DONE] 输出目录:", os.path.abspath(outdir))
    print("       - species_stats_positive.txt / .csv")
    print("       - species_stats_negative.txt / .csv")
    print("       - species_stats_combined.txt  / .csv")
    print(f"       - top{TOP_N}_positive.png / top{TOP_N}_negative.png / top{TOP_N}_combined.png")
    print("       - taxid_to_species.csv / ncbi_tax_cache.json")

if __name__ == '__main__':
    main()
