# System Architecture & Scalability Design

## 1. Executive Overview

This document details the production-style architecture of the **Intelligent Resume-to-Job Matching and Explainable Recommendation System**. The system implements a decoupled **Two-Stage Retrieval and Ranking Architecture**, separating expensive offline corpus indexing from low-latency online request handling.

---

## 2. High-Level Architecture Diagram

```
                                OFFLINE PIPELINE
                                ────────────────
                        Job Postings Dataset (23,425 jobs)
                                        │
                    ┌───────────────────┴───────────────────┐
                    ▼                                       ▼
        [Lexical Precomputation]               [Semantic Dense Embeddings]
        Text Normalization                     sentence-transformers/all-MiniLM-L6-v2
        TfidfVectorizer(max=10k, sublinear)     L2 Vector Normalization (d = 384)
                    │                                       │
                    ▼                                       ▼
        Sparse Compressed Matrix               FAISS Vector Index (IndexFlatIP / IVF)
        (data/indices/tfidf_matrix.npz)         (data/indices/jobs_index.faiss)
                    │                                       │
                    └───────────────────┬───────────────────┘
                                        ▼
                               Persisted Artifacts
                         (Vectors, Indices, Metadata)
                                        │
════════════════════════════════════════╪════════════════════════════════════════
                                        │
                                ONLINE PIPELINE
                                ────────────────
                              Uploaded Resume PDF
                                        │
                                        ▼
                              [pdfplumber Extractor]
                                        │
                                        ▼
                            [Structured Resume Parser]
                    (Skills Normalization, Experience, Degrees)
                                        │
                     ┌──────────────────┴──────────────────┐
                     ▼                                     ▼
             [Resume TF-IDF Vector]              [Resume Dense Embedding]
             Transform via saved vocab           all-MiniLM-L6-v2 (1x inference)
                     │                                     │
                     └──────────────────┬──────────────────┘
                                        ▼
                         [Stage 1: Hybrid Candidate Retrieval]
                       Score = α · S_lexical + (1 - α) · S_semantic
                       Filters 23,425 jobs ──► Top 100 Candidates
                                        │
                                        ▼
                         [Stage 2: Feature Engineering]
                   • lexical_score          • semantic_score
                   • skill_match_ratio      • matched_skills_count
                   • missing_skills_count   • experience_compatibility
                   • education_compatibility
                                        │
                                        ▼
                           [Interpretable Multi-Factor Ranker]
                         Weighted Scoring / Constrained Proxy Model
                                        │
                                        ▼
                               Top 10 Ranked Jobs
                                        │
                     ┌──────────────────┴──────────────────┐
                     ▼                                     ▼
        [Explainability Engine]                 [Grounded LLM Layer]
        • Feature score breakdown cards         • Structured context injection
        • Rule-derived "Why this job?"          • Candidate advising & Q&A
                     │                                     │
                     └──────────────────┬──────────────────┘
                                        ▼
                           Streamlit Web Application
```

---

## 3. Offline Pipeline Specifications

The offline pipeline executes as a standalone batch job (`scripts/build_index.py`), decoupling dataset growth from user-facing latency.

### 3.1 Lexical Indexing
1. **Cleaning**: Job text is stripped of markup, normalized, and tokenized.
2. **TF-IDF Vocabulary**: Fitted using sublinear term-frequency scaling ($1 + \log(\text{tf})$) and an n-gram range of $(1, 2)$ with a 10,000 max feature cap.
3. **Artifact Serialization**:
   - `tfidf_vectorizer.joblib`
   - `tfidf_matrix.npz` (Compressed Scipy CSR sparse matrix)

### 3.2 Semantic Dense Indexing
1. **Model Selection**: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, ~80MB, fast CPU inference).
2. **Vector Normalization**: Each vector $v$ is $L_2$-normalized such that $\|v\|_2 = 1.0$.
   $$\hat{v} = \frac{v}{\|v\|_2}$$
   *Property*: The inner product between two unit vectors equals their cosine similarity:
   $$\langle \hat{u}, \hat{v} \rangle = \|\hat{u}\|_2 \|\hat{v}\|_2 \cos(\theta) = \cos(\theta)$$
3. **FAISS Indexing**:
   - For working set ($N \approx 500$ to $25,000$): `faiss.IndexFlatIP(384)` delivers exact inner product search in under $5\text{ ms}$.
   - For scale ($N > 100,000$): `faiss.IndexIVFFlat` partitions the vector space into Voronoi cells using k-means centroids, delivering approximate sublinear search.
4. **Fallback Mechanism**: If FAISS is not installed, the system persists `.npz` vector arrays and utilizes vectorized NumPy dot-products.

---

## 4. Online Pipeline Specifications

1. **Resume Ingestion ($< 50\text{ ms}$)**: `pdfplumber` streams text; regex and canonical dictionary lookups extract raw skills, normalize them to canonical representations, and extract tenure/degree indicators.
2. **Single-Pass Encoding ($\approx 15\text{--}25\text{ ms}$)**: The resume is encoded once by `all-MiniLM-L6-v2` on CPU.
3. **Candidate Retrieval ($\approx 5\text{--}10\text{ ms}$)**:
   - Evaluates hybrid similarity across precomputed matrices.
   - Truncates to $K_{\text{cand}} = 100$ candidate postings.
4. **Fine-Grained Feature Extraction ($\approx 15\text{ ms}$)**: Computes skill overlaps, missing skill penalties, tenure alignment, and education satisfaction across the 100 candidates.
5. **Ranking ($\approx 2\text{ ms}$)**: Applies linear weighted scoring to sort candidates descending.
6. **Explanation & UI Rendering ($< 10\text{ ms}$)**: Dynamically synthesizes transparent attribution cards and renders Streamlit components.

**Total End-to-End Online Latency**: Under $150\text{ ms}$ on standard laptop CPU.

---

## 5. Computational Complexity Analysis

| Pipeline Stage | Brute-Force Approach | Two-Stage Indexed Architecture (Implemented) |
| :--- | :--- | :--- |
| **Resume Preprocessing** | $O(L_{\text{resume}})$ | $O(L_{\text{resume}})$ |
| **Job Text Vectorization** | $O(N \cdot L_{\text{job}})$ on every request | **$O(1)$** at runtime (Precomputed offline) |
| **Semantic Encoding** | $O(N \cdot d)$ forward passes on every request | **$1$ forward pass** for resume; jobs encoded offline |
| **Candidate Search** | $O(N \cdot d)$ linear scan | Exact Flat: $O(N \cdot d)$ optimized BLAS / FAISS SIMD<br>IVF / HNSW: **$O(\log N \cdot d)$** sublinear |
| **Fine-Grained Feature Extraction** | $O(N \cdot S)$ over all 23,425 jobs | **$O(K_{\text{cand}} \cdot S)$** where $K_{\text{cand}} = 100 \ll N$ |
| **Final Sorting** | $O(N \log N)$ | **$O(K_{\text{cand}} \log K_{\text{cand}})$** |

### Mathematical Breakdown:
- **Brute-Force Complexity**: Running full text tokenization, TF-IDF fitting, and neural embedding for $N = 23,425$ jobs on every user request requires $O(N \cdot L)$ text processing and $23,425$ transformer forward passes, taking minutes and freezing the server.
- **Two-Stage Complexity**:
  1. Offline preprocessing is amortized over infinite online queries.
  2. Online retrieval searches precomputed unit vectors in $O(N d)$ using SIMD inner products or $O(\sqrt{N} d)$ using an inverted file (IVF) index.
  3. Expensive skill matching and feature engineering only evaluate $K_{\text{cand}} = 100$ items ($0.4\%$ of the corpus).

---

## 6. Scaling to 1,000,000+ Jobs

To transition from $23,425$ jobs to $1,000,000+$ jobs in an enterprise production environment:

1. **Approximate Nearest Neighbor (ANN) Indexing**:
   - Replace `IndexFlatIP` with `IndexIVFPQ` (Inverted File with Product Quantization) or `HNSW` (Hierarchical Navigable Small World).
   - *Memory Footprint*: Product Quantization compresses 384-dimensional 32-bit float vectors ($1.5\text{ KB}$ per vector) down to 32 or 48 bytes per vector, reducing 1M vectors from $1.5\text{ GB}$ to under $50\text{ MB}$ of RAM.
2. **Distributed Storage & Retrieval**:
   - Deploy managed vector databases (e.g., Milvus, Qdrant, or OpenSearch with k-NN plugin) with sharded indices.
3. **Lexical Indexing with Elasticsearch / BM25**:
   - Replace in-memory Scipy CSR sparse matrix with Elasticsearch or Lucene inverted indices utilizing BM25 scoring for scalable keyword filtering.
4. **Asynchronous Ingestion Queue**:
   - New job postings stream into Kafka or AWS SQS, undergo automated worker embedding, and incrementally insert into the vector store without downtime.
