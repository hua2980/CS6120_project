# Enhanced Social Media Short Text Retrieval System
## Project Overview

This project aims to design and evaluate a robust social media short text retrieval system utilizing a hybrid retrieval architecture. By combining lexical retrieval (BM25) and semantic retrieval (SBERT-based dense embeddings), the system enhances retrieval accuracy and coverage.

### Core Techniques

- **Hybrid Retrieval Architecture**:
    - Combines lexical-based BM25 scoring and semantic SBERT embeddings.
    - Integrates fine-tuned SBERT embeddings optimized on STS-B and MS MARCO datasets.
- **SBERT Fine-tuning**:
    - Phase 1: Fine-tuned using the STS-B dataset with CosineSimilarityLoss to capture detailed semantic similarity.
    - Phase 2: Further fine-tuned on MS MARCO dataset with MultipleNegativesRankingLoss to optimize for passage retrieval tasks.
- **Efficient Dense Retrieval**:
    - Employs FAISS HNSW indexing for rapid approximate nearest-neighbor search, balancing accuracy with retrieval speed.

### Performance Goals

The system targets improvements across key retrieval metrics, particularly aiming for:

- High Precision@k
- High Recall@100 (>0.98)
- High nDCG scores (>0.57)
- Mean Reciprocal Rank (MRR@100) approaching 0.46

## Tech Stack

- **Languages & Libraries**: Python, PyTorch, Transformers, SentenceTransformers, FAISS, Rank-BM25, NLTK
- **Datasets**: MS MARCO v1.1, STS-B (Semantic Textual Similarity Benchmark)
- **Indexing & Retrieval**: FAISS HNSW (dense vector retrieval), BM25 (lexical retrieval)
- **Evaluation**: SentEval toolkit, evaluation metrics (Precision@k, Recall@100, nDCG, MRR@100)

## Project Structure

```
CS6120_project/
├── data/                  # Data directory
│   ├── embeddings/        # Precomputed embeddings
│   ├── processed/         # Processed data
│   └── raw/               # Raw data
├── models/                # Models directory
│   └── sbert_model/       # Fine-tuned SBERT model
├── notebooks/             # Jupyter notebooks
│   ├── 1_data_exploration.ipynb    # Data exploration and preprocessing
│   ├── 2_sbert_finetuning.ipynb    # SBERT fine-tuning on STS-B and MSMARCO
│   └── 3_Hybrid_Retrieval_&_Evaluation.ipynb  # Hybrid retrieval implementation and evaluation
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
| Project Planning | ✅ Completed | 100% | Project proposal and technical roadmap finalized |
| Environment Setup | ✅ Completed | 100% | Project scaffolding and colab setup |
| Data Preparation | ✅ Completed | 100% | MSMARCO & STS-B datasets processed |
| Model Development | ✅ Completed | 100% | SBERT fine-tuning achieved Spearman 0.85 |
| System Integration | ✅ Completed | 100% | FAISS HNSW index implemented |
| Evaluation & Optimization | ✅ Completed | 100% | MRR@10=0.7264 achieved (exceeds 0.72 target) |

### Upcoming Tasks

- [x] SBERT fine-tuning on STS-B (Spearman 0.8542)
- [x] Hybrid retrieval system implementation (BM25 + SBERT-HNSW)
- [x] Dynamic weighting strategy (alpha=0.5) - Basic implementation complete
- [ ] Logistic regression classifier for dynamic alpha adjustment
- [ ] BEIR multi-domain evaluation
- [ ] RoBERTa-base fallback implementation

## Implementation Roadmap

| Phase | Duration | Key Tasks |
|-------|----------|-----------|
| 1. Data Preparation | 2 weeks | / |
| 2. Model Development | 2 weeks | Fine-tune SBERT using MSMARCO and STSb datasets |
| 3. System Integration | 1 week | Implement the hybrid retrieval pipeline |
| 4. Evaluation & Optimization | 1 week | domain-specific performance evaluation on MSMARCO v1.1 Validation set |

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

