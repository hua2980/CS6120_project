# Enhanced Social Media Short Text Retrieval System

**Last Updated**: March 25, 2025

## Project Overview

This project aims to build an efficient social media short text retrieval system using a hybrid architecture combining SBERT and BM25, and leveraging GPU-accelerated FAISS HNSW indexing to achieve ultra-low latency (0.02ms) and high throughput (1,000 QPS).

### Core Technical Strategies

- **Hybrid Retrieval Architecture**: Combining SBERT (achieving 85.4% performance on STS-B) and BM25, consistent with SOTA methods like ColBERTv2
- **GPU-Accelerated Vector Retrieval**: Using FAISS HNSW indexing, outperforming traditional Elasticsearch solutions
- **Dynamic Weighting Strategy**: Implementing a query classifier (Logistic Regression) to dynamically adjust weights
- **Optimized FAISS Indexing**: Implementing IVF_PQ quantization (4x compression), converting vector storage from fp32 to fp16 (reducing memory usage by 50%)
- **Fallback Strategy**: Preparing RoBERTa-base as a fallback for SBERT (2x speed improvement, 5% accuracy loss)

### Performance Goals

- MRR@10 ≥ 0.72 (Top 20% on MS MARCO document ranking leaderboard)
- 1,000 QPS throughput (leveraging A100 GPU capabilities)
- Support up to 5,000 QPS peak traffic

## Tech Stack

- **Primary Language**: Python
- **Core Technologies**:
  - SBERT (Semantic Encoding)
  - BM25 (Keyword Retrieval)
  - FAISS (Vector Retrieval)
  - GPU Acceleration
- **Supporting Tools**:
  - Bayesian Optimization (Parameter Tuning)
  - Containerized Deployment

## Project Structure

```
CS6120_project/
├── configs/               # Configuration files
│   ├── eval_config.json   # Evaluation configuration
│   ├── index_config.json  # Index configuration
│   └── model_config.json  # Model configuration
├── data/                  # Data directory
│   ├── embeddings/        # Precomputed embeddings
│   ├── processed/         # Processed data
│   └── raw/               # Raw data
├── models/                # Models directory
│   ├── fallback/          # RoBERTa fallback model
│   ├── ranker/            # Dynamic weighting classifier model
│   └── sbert_model/       # Fine-tuned SBERT model
├── notebooks/             # Jupyter notebooks
│   ├── 1_data_exploration.ipynb    # Data exploration
│   ├── 2_sbert_finetuning.ipynb    # SBERT fine-tuning
│   ├── 3_index_construction.ipynb  # Index construction
│   ├── 4_hybrid_retrieval.ipynb    # Hybrid retrieval implementation
│   └── 5_evaluation.ipynb          # System evaluation
└── src/                   # Source code
    ├── bm25_retriever.py          # BM25 retrieval implementation
    ├── data_preparation.py        # Data cleaning and preprocessing
    ├── dynamic_weighting.py       # Dynamic weighting strategy implementation
    ├── evaluation.py              # Evaluation metrics calculation
    ├── index_builder.py           # FAISS index construction
    ├── model_training.py          # SBERT fine-tuning and training
    └── utils.py                   # Utility functions
```

## Current Progress

| Phase | Status | Completion | Notes |
|-------|--------|------------|-------|
| Project Planning | ✅ Completed | 100% | Completed project proposal and technical route planning |
| Environment Setup | ✅ Completed | 100% | Created project structure and configuration files |
| Data Preparation | ✅ Completed | 100% | Successfully downloaded and processed MSMARCO and Twitter datasets |
| Model Development | 🔄 In Progress | 0% | Starting SBERT fine-tuning with processed data |
| System Integration | 📅 Not Started | 0% | Plan to start after model development |
| Evaluation & Optimization | 📅 Not Started | 0% | Plan to start after system integration |

### Upcoming Tasks

- [x] Download MSMARCO and Twitter datasets
- [x] Implement social media text cleaning functionality
- [x] Process and prepare training data
- [ ] Start SBERT model fine-tuning

## Implementation Roadmap

| Phase | Duration | Key Tasks |
|-------|----------|-----------|
| 1. Data Preparation | 2 weeks | Use EnCBP technology to handle social media noise |
| 2. Model Development | 2 weeks | Fine-tune SBERT using Twitter dataset; use Bayesian optimization for parameter tuning |
| 3. System Integration | 1 week | Implement dynamic weighting; conduct parallel AB testing |
| 4. Evaluation & Optimization | 1 week | Stress test (5,000 QPS peak); domain-specific performance evaluation on 18 BEIR tasks |

## User Guide

### Environment Setup

This project is planned to be trained on Google Colab, utilizing its GPU resources. Setup steps are as follows:

1. Clone the project repository
2. Create the project directory structure in Google Drive
3. Mount Google Drive and set the project path

```python
# Run in Colab notebook
from google.colab import drive
drive.mount('/content/drive')

# Set project path
import os
PROJECT_PATH = "/content/drive/MyDrive/CS6120_project"
```

### Data Preparation

```python
# Download and process MSMARCO dataset
!wget -q https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz -O {PROJECT_PATH}/data/raw/collection.tar.gz
!wget -q https://msmarco.blob.core.windows.net/msmarcoranking/queries.tar.gz -O {PROJECT_PATH}/data/raw/queries.tar.gz
!wget -q https://msmarco.blob.core.windows.net/msmarcoranking/qrels.dev.small.tsv -O {PROJECT_PATH}/data/raw/qrels.dev.small.tsv

# Download STS Benchmark dataset
!wget -q http://ixa2.si.ehu.es/stswiki/images/4/48/Stsbenchmark.tar.gz -O {PROJECT_PATH}/data/raw/stsbenchmark.tar.gz
```

## Risk Management

| Module | Risk | Mitigation Strategy |
|--------|------|---------------------|
| Semantic Encoding | Domain drift (e.g., internet slang) | Fine-tune SBERT using Twitter dataset |
| Hybrid Ranking | Low parameter tuning efficiency | Use Bayesian optimization instead of grid search (saves 40% time) |
| Deployment | Cold start latency | Implement container pre-warming on GPU instances |

## References

- [ColBERTv2 Paper](https://arxiv.org/abs/2112.01488)
- [FAISS Optimization Guide](https://github.com/facebookresearch/faiss/wiki/Faster-search)
- [NVIDIA Vector Search Optimization](https://developer.nvidia.com/blog/accelerating-vector-search-fine-tuning-gpu-index-algorithms/)
- [Sentence-BERT Paper](https://paperswithcode.com/paper/sentence-bert-sentence-embeddings-using)
