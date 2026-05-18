# MSGPT — 基于 DDA-BERT 改进的肽段-谱图匹配 (PSM) 模型

本项目基于 DDA-BERT 架构改进，构建了一个用于蛋白质组学中**肽段-谱图匹配 (Peptide-Spectrum Match, PSM)** 二分类任务的深度学习模型。核心思路是将质谱数据与肽段序列作为多模态输入，利用 Transformer 进行联合编码与交叉注意力解码，辅以对比学习分支提升判别能力。

## 项目结构

```
├── config/                  # YAML 配置文件（数据路径、模型参数、训练超参等）
├── model/
│   ├── transformer/         # 基础 Transformer 模型（多尺度正弦峰嵌入 + Encoder-Decoder）
│   ├── transformer_compare/ # 加入对比学习分支（CoCa 风格 batch 内损失）
│   └── transformer_compare_rt/ # 在 compare 基础上增加保留时间 (RT) 特征
├── pipeline/                # 数据生成管线：原始数据 → Parquet → PKL → 按质量重排序
├── training/                # 训练脚本（基础、对比、含验证、断点恢复）
├── scoring/                 # 使用训练好的模型对所有 PSM 候选进行推理打分
├── selecting/               # 基于 q 值方法筛选高置信度 PSM
├── validation/              # 测试集验证与 FDR (False Discovery Rate) 计算
├── analysis/                # 结果比较、合并与 FDR 分析脚本
├── picture_code/            # 评分分布可视化（半小提琴图）
└── experiments/             # 实验目录（含 RT 特征的实验镜像配置）
```

## 核心方法

### 模型架构

- **多尺度正弦峰嵌入 (Multi-scale Sinusoidal Peak Embedding)**：将质谱中的 m/z 值编码为连续嵌入向量，替代传统的离散化分箱方式
- **TransformerEncoder**：对质谱图进行自注意力编码，捕获峰之间的全局依赖关系
- **PeptideDecoder**：以编码后的谱图特征为 memory，对肽段序列 token 进行交叉注意力解码
- **双任务训练**：同时进行 PSM 预测（BCE 二分类）和 MLM（掩码语言模型）辅助任务
- **对比学习分支 (Compare 版本)**：引入批次内对比损失，拉近匹配的谱图-肽段对，推远不匹配的

### 关键特征

- 直接使用连续 m/z 值，无需离散化分箱
- 多任务学习提升泛化能力
- 对比学习增强正负样本判别边界
- 支持保留时间 (Retention Time) 作为辅助特征
- FDR (q-value) 控制的高置信度 PSM 筛选

## 依赖环境

- Python 3.8+
- PyTorch >= 1.10
- Transformers (HuggingFace)
- NumPy / Pandas
- PyYAML
- scikit-learn
- Matplotlib / Seaborn（可视化）

## 使用流程

### 1. 数据准备

```bash
cd pipeline
# 将原始数据转换为 Parquet / PKL 格式，并做质量重排序
```

### 2. 模型训练

```bash
cd training
python train.py --config ../config/base.yaml
```

### 3. PSM 打分

```bash
cd scoring
python score.py --model ../checkpoints/model.pt --data ../data/candidates.pkl
```

### 4. 高置信度 PSM 筛选

```bash
cd selecting
python select.py --scores ../output/scores.csv --fdr 0.01
```

### 5. 验证与 FDR 计算

```bash
cd validation
python validate.py --predictions ../output/selected_psms.csv --ground_truth ../data/ground_truth.csv
```

## 参考文献

本项目基于 DDA-BERT 方法改进：

- DDA-BERT: 将 BERT 架构应用于 Data-Dependent Acquisition 质谱数据的肽段鉴定任务

---

*项目持续迭代中，各模块具体参数说明请参阅 `config/` 目录下的配置文件注释。*
