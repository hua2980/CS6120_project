## 1. Project Background & Motivation

In social media applications, users post large amounts of short texts or topic discussions on a daily basis, requiring an efficient and accurate way to retrieve relevant content. Traditional keyword-based search methods (e.g., BM25) may not adequately capture the diverse expressions and underlying semantics of these short texts. In contrast, **BERT-based** pre-trained models have proven highly feasible for semantic representation. By leveraging **SBERT (Sentence-BERT)** for sentence-level encoding and **FAISS** for vector retrieval, we can capture richer semantic associations among “short texts” or “short passages.”

Nonetheless, purely semantic retrieval may overlook some results that are directly matched by critical keywords. Hence, this project will retain or extend keyword-based retrieval and blend it with vector-based retrieval (Hybrid Search) to balance colloquial expressions with precise keyword matching. For this purpose, the project will use two well-known public datasets, **MSMARCO-Question-Answering** and **STS Benchmark**, for training and evaluating “query–short passage” retrieval performance and “sentence-level semantic similarity,” respectively. The ultimate goal is to build a prototype system for social media short-text retrieval.

------

## 2. Project Objectives

1. **Build a Short-Text Retrieval System Prototype**
    - Use **SBERT** (sentence-level vector representation) and **FAISS** (vector retrieval) to implement semantic retrieval for short texts.
    - Integrate keyword search (BM25) with vector retrieval to form a **Hybrid Search** mechanism.
2. **Train and Evaluate Using Public English Datasets**
    - Employ the **MSMARCO-Question-Answering** dataset for training and testing in a “query–short passage” retrieval scenario.
    - Use the **STS Benchmark** to assess the model’s performance on sentence-level semantic similarity tasks.

------

## 3. Data Collection & Processing

### 3.1 Datasets

1. **MSMARCO-Question-Answering**
    - Public dataset from Microsoft Bing search queries and associated short passages.
    - Large-scale **query–passage** annotations, serving as the primary dataset for training and evaluating SBERT’s retrieval capacity.
2. **STS Benchmark**
    - A standard “sentence similarity” evaluation dataset, containing paired sentences with similarity scores (0–5).
    - Used for deeper fine-tuning or calibration of SBERT for better sentence-level representations.
3. **BEIR (Benchmarking IR)**
    1. 

### 3.2 Data Preprocessing

1. **Data Cleaning**
    - Remove HTML tags, special characters, and other noisy elements.
    - Filter out very short or semantically empty texts (e.g., whitespace or purely links).
2. **Tokenization & Normalization**
    - Perform necessary tokenization and normalization (e.g., lowercasing, lemmatization/stemming if needed for keyword search).
3. **Data Splits**
    - MSMARCO and STS Benchmark typically come with recommended splits for training, validation, and testing. We will follow the official splits.

------

## 4. Technical Approach

### 4.1 SBERT Model Fine-tuning

- **Model Loading**: Utilize the Hugging Face Transformers library to load a pre-trained SBERT model.
- Fine-tuning Strategy:
    - Adopt a Siamese or triplet network structure for STS Benchmark to bring embeddings of similar sentences closer and push dissimilar sentences further apart.
    - Employ contrastive losses (e.g., triplet loss) or other suitable losses for similarity tasks.

### 4.2 Vector Retrieval & Index Construction (FAISS)

- **Text Vectorization**: Encode each social media short text or passage using the fine-tuned SBERT model.
- **Build FAISS Index**: Given that the test data scale is relatively small, **IndexFlatL2** may suffice. Alternatively, **HNSW** can be used for approximate nearest neighbor search.

### 4.3 Keyword Retrieval (BM25)

- Use a lightweight Python library such as **Whoosh**, which provides a BM25-based retrieval model.

### 4.4 Ranking & Returning Results

- **Ranking Strategy**: Apply a weighted fusion. Normalize both `BM25_score` and `vector_score` to the same 0–1 scale and combine with defined weights:

    `final_score=α×BM25_score+β×semantic_scorefinal_score=α×BM25_score+β×semantic_score`

    Conduct experiments to adjust the hyperparameters `α` and `β` for the best performance.

### 4.5 Evaluation Metrics

- **Retrieval Metrics**: Assess performance using Precision, Recall, F1 scores, `MRR@10`, `nDCG@10`, etc., to track the quality of retrieval.

------

## 5. Project Implementation & Timeline

1. **Phase 1 (Week 1)**: Data preparation and initial model training.
2. **Phase 2 (Weeks 2–3)**: SBERT fine-tuning, BM25/FAISS index construction, and initial retrieval experiments.
3. **Phase 3 (Weeks 4–5)**: Hybrid retrieval strategy, parameter tuning, and model evaluation.
4. **Phase 4 (Week 6)**: System integration and final report.