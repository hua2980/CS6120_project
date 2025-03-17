# 增强社交媒体短文本检索系统

**最后更新**: 2025年3月17日

## 项目概述

本项目旨在构建一个高效的社交媒体短文本检索系统，采用混合架构结合SBERT和BM25，并利用GPU加速的FAISS HNSW索引实现极低延迟(0.02ms)和高吞吐量(1,000 QPS)的检索性能。

### 核心技术策略

- **混合检索架构**：结合SBERT(在STS-B上达到85.4%性能)和BM25，与SOTA方法如ColBERTv2保持一致
- **GPU加速向量检索**：使用FAISS HNSW索引，性能远超传统Elasticsearch解决方案
- **动态权重策略**：实现查询分类器(Logistic Regression)动态调整权重
- **优化的FAISS索引**：实现IVF_PQ量化(4x压缩)，将向量存储从fp32转换为fp16(减少50%内存占用)
- **备用策略**：准备RoBERTa-base作为SBERT的降级替代方案(2x速度提升，5%精度损失)

### 性能目标

- MRR@10 ≥ 0.72 (位于MS MARCO文档排名榜单前20%)
- 1,000 QPS吞吐量 (利用A100 GPU能力)
- 支持高达5,000 QPS的峰值流量

## 技术栈

- **主要语言**：Python
- **核心技术**：
  - SBERT (语义编码)
  - BM25 (关键词检索)
  - FAISS (向量检索)
  - GPU加速
- **支持工具**：
  - Bayesian优化 (参数调优)
  - 容器化部署

## 项目结构

```
CS6120_project/
├── configs/               # 配置文件
│   ├── eval_config.json   # 评估配置
│   ├── index_config.json  # 索引配置
│   └── model_config.json  # 模型配置
├── data/                  # 数据目录 (包含__init__.py)
│   ├── embeddings/        # 预计算的嵌入向量
│   ├── processed/         # 预处理后的数据
│   └── raw/               # 原始数据
├── models/                # 模型目录 (包含__init__.py)
│   ├── fallback/          # RoBERTa备用模型
│   ├── ranker/            # 动态权重分类器模型
│   └── sbert_model/       # 微调后的SBERT模型
├── notebooks/             # Jupyter笔记本
│   ├── 1_data_exploration.ipynb    # 数据探索
│   ├── 2_sbert_finetuning.ipynb    # SBERT微调
│   ├── 3_index_construction.ipynb  # 索引构建
│   ├── 4_hybrid_retrieval.ipynb    # 混合检索实现
│   └── 5_evaluation.ipynb          # 系统评估
└── src/                   # 源代码 (包含__init__.py)
    ├── bm25_retriever.py          # BM25检索实现
    ├── data_preparation.py        # 数据清洗与预处理
    ├── dynamic_weighting.py       # 动态权重策略实现
    ├── evaluation.py              # 评估指标计算
    ├── index_builder.py           # FAISS索引构建
    ├── model_training.py          # SBERT微调和训练
    └── utils.py                   # 公共工具函数
```
# 增强社交媒体短文本检索系统

**最后更新**: 2025年3月17日

## 项目概述

本项目旨在构建一个高效的社交媒体短文本检索系统，采用混合架构结合SBERT和BM25，并利用GPU加速的FAISS HNSW索引实现极低延迟(0.02ms)和高吞吐量(1,000 QPS)的检索性能。

### 核心技术策略

- **混合检索架构**：结合SBERT(在STS-B上达到85.4%性能)和BM25，与SOTA方法如ColBERTv2保持一致
- **GPU加速向量检索**：使用FAISS HNSW索引，性能远超传统Elasticsearch解决方案
- **动态权重策略**：实现查询分类器(Logistic Regression)动态调整权重
- **优化的FAISS索引**：实现IVF_PQ量化(4x压缩)，将向量存储从fp32转换为fp16(减少50%内存占用)
- **备用策略**：准备RoBERTa-base作为SBERT的降级替代方案(2x速度提升，5%精度损失)

### 性能目标

- MRR@10 ≥ 0.72 (位于MS MARCO文档排名榜单前20%)
- 1,000 QPS吞吐量 (利用A100 GPU能力)
- 支持高达5,000 QPS的峰值流量

## 技术栈

- **主要语言**：Python
- **核心技术**：
  - SBERT (语义编码)
  - BM25 (关键词检索)
  - FAISS (向量检索)
  - GPU加速
- **支持工具**：
  - Bayesian优化 (参数调优)
  - 容器化部署

## 项目结构

```
CS6120_project/
├── configs/               # 配置文件
│   ├── eval_config.json   # 评估配置
│   ├── index_config.json  # 索引配置
│   └── model_config.json  # 模型配置
├── data/                  # 数据目录
│   ├── embeddings/        # 预计算的嵌入向量
│   ├── processed/         # 预处理后的数据
│   └── raw/               # 原始数据
├── models/                # 模型目录
│   ├── fallback/          # RoBERTa备用模型
│   ├── ranker/            # 动态权重分类器模型
│   └── sbert_model/       # 微调后的SBERT模型
├── notebooks/             # Jupyter笔记本
│   ├── 1_data_exploration.ipynb    # 数据探索
│   ├── 2_sbert_finetuning.ipynb    # SBERT微调
│   ├── 3_index_construction.ipynb  # 索引构建
│   ├── 4_hybrid_retrieval.ipynb    # 混合检索实现
│   └── 5_evaluation.ipynb          # 系统评估
└── src/                   # 源代码
    ├── bm25_retriever.py          # BM25检索实现
    ├── data_preparation.py        # 数据清洗与预处理
    ├── dynamic_weighting.py       # 动态权重策略实现
    ├── evaluation.py              # 评估指标计算
    ├── index_builder.py           # FAISS索引构建
    ├── model_training.py          # SBERT微调和训练
    └── utils.py                   # 公共工具函数
```

## 当前进度

| 阶段 | 状态 | 完成度 | 备注 |
|------|------|--------|------|
| 项目规划 | ✅ 已完成 | 100% | 完成项目提案和技术路线规划 |
| 环境搭建 | ✅ 已完成 | 100% | 创建项目结构和配置文件 |
| 数据准备 | 🔄 进行中 | 0% | 计划下载和处理MSMARCO和STS数据集 |
| 模型开发 | 📅 未开始 | 0% | 计划在数据准备完成后开始 |
| 系统集成 | 📅 未开始 | 0% | 计划在模型开发完成后开始 |
| 评估优化 | 📅 未开始 | 0% | 计划在系统集成完成后开始 |

### 近期任务

- [ ] 下载MSMARCO和STS Benchmark数据集
- [ ] 实现社交媒体文本清理功能
- [ ] 处理并准备训练数据
- [ ] 开始SBERT模型微调

## 实施路线图

| 阶段 | 持续时间 | 关键任务 |
|------|----------|----------|
| 1. 数据准备 | 2周 | 使用EnCBP技术处理社交媒体噪声 |
| 2. 模型开发 | 2周 | SBERT使用Twitter数据集微调；使用贝叶斯优化进行参数调优 |
| 3. 系统集成 | 1周 | 实现动态权重；进行并行AB测试 |
| 4. 评估与优化 | 1周 | 压力测试(5,000 QPS峰值)；在BEIR的18个任务上进行领域特定性能评估 |

## 使用指南

### 环境设置

本项目计划在Google Colab上进行训练，利用其GPU资源。设置步骤如下：

1. 克隆项目仓库
2. 在Google Drive中创建项目目录结构
3. 挂载Google Drive并设置项目路径

```python
# 在Colab笔记本中运行
from google.colab import drive
drive.mount('/content/drive')

# 设置项目路径
import os
PROJECT_PATH = "/content/drive/MyDrive/CS6120_project"
```

### 数据准备

```python
# 下载并处理MSMARCO数据集
!wget -q https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz -O {PROJECT_PATH}/data/raw/collection.tar.gz
!wget -q https://msmarco.blob.core.windows.net/msmarcoranking/queries.tar.gz -O {PROJECT_PATH}/data/raw/queries.tar.gz
!wget -q https://msmarco.blob.core.windows.net/msmarcoranking/qrels.dev.small.tsv -O {PROJECT_PATH}/data/raw/qrels.dev.small.tsv

# 下载STS Benchmark数据集
!wget -q http://ixa2.si.ehu.es/stswiki/images/4/48/Stsbenchmark.tar.gz -O {PROJECT_PATH}/data/raw/stsbenchmark.tar.gz
```

## 风险管理

| 模块 | 风险 | 缓解策略 |
|------|------|----------|
| 语义编码 | 领域漂移(如网络俚语) | 使用Twitter数据集微调SBERT |
| 混合排序 | 参数调优效率低 | 用贝叶斯优化替代网格搜索(节省40%时间) |
| 部署 | 冷启动延迟 | 在GPU实例上实现容器预热 |

## 参考资源

- [ColBERTv2论文](https://arxiv.org/abs/2112.01488)
- [FAISS优化指南](https://github.com/facebookresearch/faiss/wiki/Faster-search)
- [NVIDIA向量搜索优化](https://developer.nvidia.com/blog/accelerating-vector-search-fine-tuning-gpu-index-algorithms/)
- [Sentence-BERT论文](https://paperswithcode.com/paper/sentence-bert-sentence-embeddings-using)
