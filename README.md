# SeqPatho BaseModel Core Code Summary

本目录是从 `E:\4PythonProject\SeqPatho` 全项目代码中提取出的核心研究代码快照，并按功能重命名/归类。原始项目文件未被修改。

## Project Code Analysis

- 全项目共有约 `220` 个 Python 文件；排除外部包 `DNABERT-S`、`CData/Plant_DNA_LLMs-main` 和本次新建的 `BaseModel` 后，研究相关 Python 文件约 `196` 个。
- `Models` 到 `Models_08_len` 主要是早期 PGPredict/PGPred 训练、长度窗口、下采样、物种划分、植物子集、聚类和调参脚本，存在大量同一模型的历史变体。
- `Models_09_Pcontrast` 是较完整的模型对照阶段，包含 `Ours` 以及 Basenji/Basset/DanQ/DeepSEA/PDLLMs/Genomic Benchmarks 等比较模型。
- `Models_10_buchong` 是补充基线阶段，重点是 length/GC、6-mer TF-IDF、SVM/XGBoost、BLAST/MMseqs nearest-neighbor、DNABERT-S ablation 等基线。
- `SourcePathogen_Validation_Protocols_20260518` 是当前最清晰、最可复现的验证协议代码：manifest 驱动，支持 locked split、source-pathogen OOD、LOSO 和 hard-negative evaluation。
- `Sample` 与 `paper_assets` 主要承担模型解释、注意力可视化、per-base attribution 和生物学特征重叠分析。
- `Dataset`、`Data_statistics`、`CData`、`Instantiation` 多为数据下载、清洗、统计、案例实例化脚本；只提取了和核心切分/统计直接相关的脚本。

## Core Scientific Pipeline

核心任务是植物病原相关核酸序列二分类。主模型流程为：

1. 清洗 DNA 序列，只保留/映射到 `ACGTN`。
2. 基于 canonical reverse-complement hash 去重并检查泄漏。
3. 用 pathogen/source species tax id 或 species name 构建 source-pathogen 分组。
4. 生成 locked group-aware split、source-pathogen species-disjoint OOD split 和 LOSO fold registry。
5. 训练 SeqPatho 模型：DNABERT-S encoder + BiGRU + stacked masked channel attention + masked pooling + binary classifier。
6. 对长序列做 nucleotide sliding windows，窗口分数聚合为 gene-level score。
7. 输出 `record_id,label,score` 级别预测文件，再汇总 AUROC/AUPRC/MCC/F1 等指标。

## Directory Layout

### `primary_pipeline`

当前建议优先使用的复现实验主线。

- `protocol_manifest_builder.py`: 生成 locked split、source-pathogen OOD、LOSO registry 和 hard-negative manifest。
- `loso_fold_materializer.py`: 将 compact LOSO registry 中的单个 fold 展开成 row-level train/val/test manifest。
- `manifest_training_utils.py`: manifest 加载、窗口切分、窗口分数聚合、标准预测表生成工具。
- `seqpatho_manifest_trainer.py`: manifest-driven SeqPatho 训练/评估入口，避免训练脚本内部重新随机切分。
- `validation_result_summarizer.py`: 汇总 OOD/LOSO/hard-negative 预测 CSV 为论文指标和图表。
- `protocol_manifest_visualizer.py`: 可视化 protocol manifest。
- `seqpatho_run_summarizer.py`: 汇总 SeqPatho 训练运行结果。

### `model_core`

SeqPatho 主模型及其历史主线。

- `seqpatho_original_train_eval.py`: 从 `Models_09_Pcontrast/Ours/Ours.py` 提取，代表最终对照实验中的 Ours 主模型训练脚本。
- `recurrent_aggregator_ablation.py`: recurrent aggregator 消融实验。
- `legacy_full_ood_experiment_ot3.py`: 从 `Models_06_NPlants/Ot3.py` 提取，包含旧版完整 standard species-disjoint、LOSO、LOCO clade 实验流程。
- `curve_export_helper.py`: 曲线导出和运行时覆盖辅助函数。

### `baselines`

对照模型和补充基线。

- `baseline_common.py`: 基线共享的数据清洗、split、指标、绘图、FASTA 输出工具。
- `classical_homology_baselines.py`: length/GC、6-mer、SVM/XGBoost、BLAST/MMseqs nearest-neighbor 综合基线。
- `tabular_kmer_baselines.py`: 表格式 length/GC 和 k-mer 基线。
- `dnaberts_ablation_baselines.py`: DNABERT-S frozen、partial fine-tune、random-init ablation。
- `nearest_neighbor_baseline.py`: 最近邻同源性基线。
- `protocol_lightweight_baselines.py`: source-pathogen protocol 绑定的 lightweight baselines。
- `comparative_basenji.py`, `comparative_basset.py`, `comparative_danq.py`, `comparative_deepsea.py`, `comparative_genomic_benchmarks.py`, `comparative_pdllms.py`: 深度学习比较模型。
- `curve_export_helper.py`: 比较模型运行需要的曲线导出辅助函数。
- `baseline_result_summarizer.py`, `latest_baseline_summary_builder.py`: 基线结果汇总。

### `interpretability`

预测解释和注意力/生物学重叠分析。

- `attention_rollout_and_saliency.py`: 注意力捕获、attention rollout、token/base importance、sequence-logo 式可视化。
- `per_base_prediction.py`: 单序列 per-base prediction/attribution 入口。
- `attention_biology_overlap_analysis.py`: attribution hotspot 与蛋白/结构/功能特征重叠富集分析。

### `dataset_preprocessing`

数据准备和统计中保留的核心脚本。

- `species_disjoint_ood_split_builder.py`: 物种不重叠 OOD split 构建。
- `species_proportion_statistics.py`: 物种组成和比例统计。
- `plant_taxid_filter.py`: 基于 tax id 的植物记录筛选。

## Notes

- 这些文件是从原工程提取的可追溯快照；部分脚本仍保留原始绝对路径、环境路径或数据路径，运行前需要按当前机器环境调整。
- 外部模型目录 `DNABERT-S` 和外部项目 `CData/Plant_DNA_LLMs-main` 没有复制到这里；它们作为依赖保留在原仓库。
- 如果只复现实验主线，优先从 `primary_pipeline/protocol_manifest_builder.py` 和 `primary_pipeline/seqpatho_manifest_trainer.py` 开始。
