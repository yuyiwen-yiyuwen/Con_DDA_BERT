# organized_attantion — 代码架构重组目录

> 本目录将 `/home/yiwen/AIPC/scripts/attantion/` 下的代码按功能模块重新组织。所有代码文件为独立拷贝，数据目录通过软链接指向原位置。

---

## 目录总览

```
organized_attantion/
├── data/                  # 📊 所有数据（软链接指向 ../attantion/）
├── model/                 # 🔧 模型定义（3个变体，独立拷贝）
├── pipeline/              # 🔄 数据处理流水线（4步）
│   ├── 1_gen_parquet/     #   步骤1：原始数据 → Parquet
│   ├── 2_convert_pkl/    #   步骤2：Parquet → PKL
│   ├── 3_split_data/     #   步骤3：划分训练/验证集
│   └── 4_rearrange/      #   步骤4：质量锚定重排
├── scoring/               # 🎯 模型打分与 Target/Decoy 筛选
├── training/              # 🏋️ 训练脚本
├── validation/            # ✅ 验证/评测脚本
├── analysis/              # 📈 结果对比与分析
├── config/                # ⚙️ 配置文件
└── experiments/           # 🧪 实验变体
    └── with_RT/           #   保留时间 (RT) 实验（完整独立变体）
```

---

## data/ — 数据目录（软链接）

所有训练、验证、推理过程中产生的数据均通过软链接集中于此，指向 `../attantion/` 下的原始数据目录。

| 软链接 | 目标 | 说明 |
|--------|------|------|
| `pkl_dataset/` | `../attantion/pkl_dataset/` | 主训练数据集（train/val 划分），pkl 格式 |
| `pkl_dataset_tims/` | `../attantion/pkl_dataset_tims/` | TIMS 数据集，包含 qvalue_select / train_select 等子集 |
| `processed_parquet_tims/` | `../attantion/processed_parquet_tims/` | TIMS 预处理后的 parquet 文件 |
| `mzml_results/` | `../attantion/mzml_results/` | mzML 数据验证结果（results_8 ~ results_11） |
| `mzml_cpmpare_results/` | `../attantion/mzml_cpmpare_results/` | mzML 对比实验结果 |
| `tims_cpmpare_results/` | `../attantion/tims_cpmpare_results/` | TIMS 对比实验结果 |
| `val_train_results/` | `../attantion/val_train_results/` | 训练过程验证曲线（含 loss 图） |
| `val_train_results_mlm/` | `../attantion/val_train_results_mlm/` | MLM 训练验证结果 |
| `val_train_results_tims_valselect_check/` | `../attantion/val_train_results_tims_valselect_check/` | TIMS 验证集选择检查结果 |
| `checkpoints/` | `../attantion/checkpoints/` | 基础模型 checkpoint |
| `checkpoints_compare/` | `../attantion/checkpoints_compare/` | 对比模型 checkpoint（含断点续训） |
| `checkpoints_tims/` | `../attantion/checkpoints_tims/` | TIMS 模型 checkpoint |
| `mzml_pt/` | `../attantion/mzml_pt/` | mzML 预训练 checkpoint |
| `train_mzml_compare_score/` | `../attantion/train_mzml_compare_score/` | mzML 对比模型逐 batch 打分结果 (TSV) |
| `train_tims_all_psm_score/` | `../attantion/train_tims_all_psm_score/` | TIMS 全量 PSM 打分结果 (train/val) |
| `split_dataset_tims/` | `../attantion/split_dataset_tims/` | TIMS 数据集划分输出目录（当前为空） |

---

## model/ — 模型定义（代码拷贝）

三套模型架构，核心均为基于 Transformer 的谱图-肽段匹配打分模型（MSGPT）。代码从原目录完整拷贝，独立维护。

### transformer/
基础模型库，与 `training/train.py` 配合使用。
- `model.py` — MSGPT 模型主体（Multi-Scale Peak Embedding + Peptide Decoder）
- `dataset.py` — SpectrumDataset 数据加载与 collate
- `iterable_dataset_online_parquet.py` — 在线 parquet 流式数据集
- `decoder.py` — 肽段序列编解码器
- `layers.py` — Transformer 层实现
- `train_deepspeed.py` — DeepSpeed 分布式训练支持

### transformer_compare/
对比模型变体，与 `training/train_compare.py` 配合。结构与 transformer/ 类似，但在损失函数和训练策略上有差异。

### transformer_compare_rt/
RT（保留时间）感知的对比模型变体。来源于 `with_RT/` 实验。precursors 张量额外包含 `deltaRT` 和 `predictedRT` 两个特征维度，用于融合色谱保留时间信息。

---

## pipeline/ — 数据处理流水线（代码拷贝）

完整的数据处理流程，按执行顺序分为 4 步。所有脚本从原目录独立拷贝。

### 1_gen_parquet/ — 步骤1：原始数据 → 训练用 Parquet
将原始质谱搜索结果（Sage、FP）与原始谱图合并，生成统一的训练用 parquet 文件。

| 文件 | 功能 |
|------|------|
| `1_gen_parquet.py` | **核心脚本**：读取 raw / sage / fp 三份 parquet，构造统一 PSM 标识，做 target/decoy 交集与平衡，合并谱图，过滤保存 |
| `1_gen_parquet_call.py` | **并行调度器**：使用 ProcessPoolExecutor 并行调用 `1_gen_parquet.py` 处理多个原始文件 |
| `1_gen_parquet_tims.py` | **TIMS 版本**：将 TIMS 单文件 parquet 转为训练用逐 PSM parquet，支持 q-value 阈值过滤 |

**数据流**：`raw.parquet + sage.parquet + fp.parquet → 合并后的 mzml.parquet`

### 2_convert_pkl/ — 步骤2：Parquet → PKL 缓存
将 parquet 格式转换为模型可直接快速加载的 pkl 缓存。

| 文件 | 功能 |
|------|------|
| `3_convert_parquet2pkl.py` | mzML 版本：parquet → pkl 转换 |
| `3_convert_parquet2pkl_tims.py` | TIMS 版本：parquet → pkl 转换 |

**数据流**：`*.parquet → *.pkl`

### 3_split_data/ — 步骤3：划分训练/验证集
将全量数据按比例划分为训练集和验证集。

| 文件 | 功能 |
|------|------|
| `split_data.py` | **mzML 数据划分**：读取步骤1输出，并行拆分为 train/val parquet，支持按比例或按文件数划分 |
| `split_data_tims.py` | **TIMS 数据划分**：TIMS 版本的 train/val 拆分 |

**数据流**：`全量 parquet → train/val parquet`

### 4_rearrange/ — 步骤4：质量锚定重排
对训练数据按母离子质量进行重排，使同一 batch 内的谱图在质量空间上接近。

| 文件 | 功能 |
|------|------|
| `rearrange_train_all_mass_anchored.py` | 加载全部 pkl 建立全局索引，按质量锚定策略（动态窗口 + 扩展因子）重排，输出为分块小 pkl 文件。支持批内肽段去重、随机挑选 batch、多线程写出 |

**数据流**：`train pkl → 质量排序的分块 pkl`

---

## scoring/ — 打分与筛选（代码拷贝）

用已训练的模型对全量 PSM 打分，并根据分数筛选高质量 Target 和 Decoy 构建最终训练集。

| 文件 | 功能 |
|------|------|
| `score_tims_train_all_psm.py` | 加载训练好的 MSGPT 模型，对 train/val 所有 PSM 逐条打分，输出分数文件 |
| `build_qvalue_dataset_from_all_psm_scores.py` | 基于模型分数计算 q-value，构建带 q-value 标签的数据集 |
| `select_tims_top_target_balanced_decoy_to_pklgz.py` | **筛选脚本**：Target 按分数从高到低取 top；Decoy 采用"一半高分 + 一半随机"策略；切分 train/val 后调用 `rearrange_train_all_mass_anchored.py` 做重排，输出 `.pkl.gz` |

**数据流**：`模型 checkpoint + pkl 数据 → PSM 分数 → 筛选后的高质量 pkl.gz`

---

## training/ — 训练脚本（代码拷贝）

| 文件 | 功能 |
|------|------|
| `train.py` | **基础训练**：使用 `model/transformer/`，标准的谱图-肽段匹配训练。支持 warmup、amp、torch.compile、DeepSpeed |
| `train_compare.py` | **对比训练**：使用 `model/transformer_compare/`，包含对比学习策略的变体训练 |
| `train_compare_resume_epoch8.py` | **断点续训**：从 epoch 8 的 checkpoint 恢复 `train_compare` 训练，用于在已有模型基础上继续微调 |

---

## validation/ — 验证与评测（代码拷贝）

| 文件 | 使用模型 | 数据格式 | 用途 |
|------|----------|----------|------|
| `validate_parquet.py` | transformer_compare | Parquet | 验证 parquet 格式测试集的 ID 准确率与 AUC |
| `validate_parquet_original_dda_bert.py` | transformer (原始) | Parquet | 使用原始 DDA BERT 模型验证 parquet 数据 |
| `validate_mzml_all_pt.py` | transformer | mzML (pkl) | 在所有 mzML 预训练数据上做全量验证 |
| `validate_mzml_compare.py` | transformer_compare | mzML (pkl) | 对比模型在 mzML 数据上的验证，含 FDR 曲线与统计 |
| `validate_tims_compare.py` | transformer_compare | TIMS (pkl) | 对比模型在 TIMS 数据上的验证 |
| `validate_tims_single_pt.py` | transformer | TIMS (单个 pkl) | 对单个 TIMS pkl 文件做快速验证 |

---

## analysis/ — 结果对比与分析（代码拷贝）

| 文件 | 功能 |
|------|------|
| `compare_and_merge.py` | 比较两个结果目录中的 tsv，按 FDR ≤ 1% 统计肽段/PSM 数量，合并最佳结果到输出目录 |
| `compare_counts.py` | 批量对比两个结果目录中共有文件的 ID 数量差异，输出统计表 |

---

## config/ — 配置文件（独立拷贝）

| 文件 | 说明 |
|------|------|
| `model.yaml` | 模型与训练全局配置：谱图预处理参数（n_peaks=150, min_mz=50, max_mz=2500）、模型架构（dim=768, 9层, 16头）、氨基酸残基质量表、训练超参数（lr=5e-5, batch_size=256, 25 epochs）、数据路径等 |

---

## experiments/with_RT/ — RT 实验变体（代码拷贝）

`with_RT` 是一个**完整且独立的实验变体**，在原始 pipeline 的基础上针对**保留时间 (Retention Time)** 进行了适配。代码来源于 `../attantion/with_RT/`，完整拷贝，独立维护。其内部结构与主流程镜像对应：

```
with_RT/
├── model/transformer_compare/   # RT 感知的 transformer_compare（precursors 含 deltaRT + predictedRT）
├── pipeline/                    # 1~4 步数据处理（RT 适配版）
│   ├── 1_gen_parquet/
│   ├── 2_convert_pkl/
│   ├── 3_split_data/
│   └── 4_rearrange/
├── scoring/                     # 打分与筛选（RT 版）
├── training/                    # train.py / train_compare.py / train_compare_resume_epoch8.py
└── validation/                  # 全部 6 个验证脚本
```

---

## 数据处理全流程速览

```
原始 mzML/TIMS 数据
       │
       ▼
[1_gen_parquet]     原始谱图 + Sage结果 + FP结果 → 合并 parquet
       │
       ▼
[2_convert_pkl]     parquet → pkl 缓存（模型快速加载）
       │
       ▼
[3_split_data]      全量数据 → train / val 划分
       │
       ▼
[4_rearrange]       train pkl → 质量锚定重排（batch 内质量接近）
       │
       ▼
[scoring]           模型对全量 PSM 打分 → q-value 计算 → 筛选高质量 Target/Decoy
       │
       ▼
[training]          重排后的 pkl → MSGPT 训练
       │
       ▼
[validation]        模型在 test set 上验证 → FDR 曲线 / AUC / 准确率
       │
       ▼
[analysis]          多组结果对比、合并最佳结果
```

---

## 与原目录的关系

```
/home/yiwen/AIPC/scripts/
├── attantion/              ← 原始目录（完全未动）
│   ├── train.py, ...
│   ├── transformer/
│   ├── with_RT/
│   └── pkl_dataset/ ...
│
└── organized_attantion/    ← 本重组目录
    ├── data/               → 软链接 → ../attantion/xxx
    ├── model/              → 代码拷贝
    ├── pipeline/           → 代码拷贝
    ├── ...
    └── README.md
```
