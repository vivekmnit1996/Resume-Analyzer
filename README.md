# Smart Resume Analyzer & Job Recommendation System

**Python | NLP | Scikit-learn | Sentence-Transformers | FAISS | Streamlit | LLaMA 3.3 (Groq)**

- Built an explainable NLP resume-to-job recommendation system over a 23,425 job posting corpus using canonical skill extraction, TF-IDF, and dense semantic sentence embeddings (`all-MiniLM-L6-v2`).
- Implemented a two-stage retrieval and ranking pipeline: fast hybrid candidate retrieval ($\alpha$-blended lexical and semantic search) followed by interpretable multi-factor ranking (skill match ratio, semantic similarity, lexical overlap, experience alignment, and education compatibility).
- Evaluated retrieval models across Information Retrieval metrics: Dense Embeddings achieved 0.5600 P@5, 0.6000 MRR, and 0.5108 NDCG@10; evaluated leakage-free job category classification reaching 84.00% test accuracy and 88.22% (±0.02) 5-fold cross-validation Macro F1.
- Engineered deterministic, feature-level match breakdowns (matched/missing skills, strength drivers, and qualification gaps) and integrated a grounded LLaMA 3.3 70B assistant strictly bounded by the ML engine's computed results.

---

## 📌 Problem Statement

Job applicants struggle to evaluate how well their resumes match complex job descriptions, often receiving generic rejection notices without actionable feedback. Meanwhile, automated applicant tracking systems (ATS) either rely on rigid keyword matching that ignores synonyms or opaque black-box models that cannot explain why a candidate was ranked. Applicants need an automated, transparent, and explainable system that identifies qualified roles, quantifies domain compatibility, and pinpoints exact skill gaps.

---

## 💡 Solution

A robust, two-stage Information Retrieval (IR) and Recommendation system that:
1. Ingests resume PDFs and extracts structured profile features (canonical skills, experience tenure, degrees, and role keywords).
2. Performs fast **Stage 1 Hybrid Retrieval** combining sublinear TF-IDF (for exact acronyms and tools) with dense sentence embeddings (for conceptual semantics), narrowing 23K+ jobs to the top 100 candidates in $<25\text{ ms}$.
3. Executes **Stage 2 Multi-Factor Ranking** to score candidates based on verified technical skill overlap, tenure alignment, credential satisfaction, and semantic fit.
4. Generates deterministic, feature-grounded explanations highlighting verified strengths and missing requirements.
5. Employs a grounded, optional LLaMA 3.3 70B conversational assistant that explains recommendations without deciding rankings or inventing details.

---

## 🏗️ Architecture & Pipeline

```
Resume PDF / Text Input
          │
          ▼
PDF Text Extraction (pdfplumber with scanned document detection)
          │
          ▼
NLP Parsing & Centralized Skill Normalization (e.g. ML → Machine Learning, NLP → Natural Language Processing)
          │
          ├────────────────────────────────────────┐
          ▼                                        ▼
[Lexical Representation]                 [Semantic Dense Embedding]
Precomputed TF-IDF (10k vocab)           sentence-transformers/all-MiniLM-L6-v2 (d=384)
          │                                        │
          └───────────────────┬────────────────────┘
                              ▼
           Stage 1: Hybrid Candidate Retrieval (Top 100)
             Score = α · Lexical + (1 - α) · Semantic
                              │
                              ▼
           Stage 2: Multi-Factor Candidate Ranking
             • Skill Match Ratio (35%)
             • Semantic Proximity (30%)
             • Lexical Overlap (15%)
             • Experience Compatibility (10%)
             • Education Satisfaction (10%)
                              │
                              ▼
                   Top-K Recommendations
                              │
          ┌───────────────────┴───────────────────┐
          ▼                                       ▼
Deterministic Match Breakdown             Grounded AI Career Assistant
  • Matched & Missing Skills                • Injected with ML match context
  • Why this match / Key gaps               • Zero hallucinated scores/jobs
  • Feature contribution metrics            • Offline-capable fallback
```

---

## 🛠️ Technologies & Why They Were Chosen

| Technology | Purpose in System | Rationale for Selection |
| :--- | :--- | :--- |
| **Python 3.10+** | Core runtime | Standard ecosystem for production machine learning, scientific computing, and NLP. |
| **Scikit-learn** | TF-IDF & Classification | Lightweight, deterministic, high-speed text vectorization and leakage-free classification pipeline. |
| **Sentence-Transformers** | Semantic Dense Embeddings | `all-MiniLM-L6-v2` produces high-quality 384-dimensional embeddings at 5x the speed of BERT-base on CPU. |
| **FAISS (faiss-cpu)** | Dense Vector Indexing | Optimized C++ SIMD vector search using `IndexFlatIP`; cosine similarity via inner product of $L_2$-normalized vectors with automatic NumPy fallback. |
| **NLTK** | Linguistic Preprocessing | Reliable tokenization, lemmatization (WordNet), and stopword filtering for lexical normalization. |
| **pdfplumber** | PDF Text Extraction | Accurately parses structured multi-column text and tables compared to PyPDF2. |
| **Streamlit** | Interactive UI | Pure-Python responsive user interface with built-in state management and instant local deployment. |
| **Groq / LLaMA 3.3 70B** | Grounded Career Assistant | Low-latency inference for conversational explanation grounded strictly in the ML engine's structured outputs. |

---

## 📊 Evaluation & Empirical Results

All metrics are measured directly on the project's dataset and benchmark test suite (`experiments/run_experiments.py`).

### 1. Recommender System Evaluation (Information Retrieval)

Evaluated across realistic multi-domain query profiles over a candidate pool of 100 retrieved jobs:

| Model Paradigm | Precision@5 | Recall@5 | MRR | NDCG@10 | Hit Rate@5 | Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **TF-IDF Only** ($\alpha = 1.0$) | 0.4000 | 0.0200 | 0.4667 | 0.4147 | 0.6000 | 1273 ms |
| **Embeddings Only** ($\alpha = 0.0$) | **0.5600** | **0.0280** | **0.6000** | **0.5108** | **0.6000** | **20.6 ms** |
| **Hybrid Retrieval** ($\alpha = 0.5$) | 0.5200 | 0.0260 | **0.6000** | 0.4820 | **0.6000** | 20.3 ms |
| **Hybrid + Multi-Factor Ranking** | 0.4000 | 0.0200 | **0.6000** | 0.4555 | **0.6000** | 89.6 ms |

*Empirical Finding:* Dense embeddings significantly outperform pure TF-IDF in semantic recall and ranking quality (NDCG@10 $0.5108$ vs $0.4147$). The multi-factor ranker re-orders results to prioritize exact skill overlap and tenure requirements over raw text similarity.

### 2. Job Category Classification Evaluation (Leakage-Free)

Evaluated using an end-to-end `sklearn.pipeline.Pipeline` with `StratifiedKFold` (5 folds) on held-out test data:
- **Test Accuracy**: $84.00\%$
- **Macro Precision**: $84.56\%$
- **Macro Recall**: $84.00\%$
- **Macro F1-Score**: $83.85\%$
- **5-Fold Cross-Validation Macro F1**: $88.22\% \pm 0.0201$
- **Data Leakage Guarantee**: Vectorizer vocabulary and IDF weights are learned strictly on training splits.

---

## ⚠️ Known Limitations

1. **Static Dataset**: The job corpus contains 23,425 postings from an open Kaggle dataset. It does not reflect live ATS postings or real-time market shifts.
2. **Offline Evaluation Uses Proxy Labels**: Real user interaction signals (impressions, clicks, recruiter interview selections) are unavailable in public datasets. Offline IR evaluation relies on domain-category matching and technical skill overlap as a proxy for relevance.
3. **Dictionary-Based Skill Extraction**: Skill detection uses a curated database of 141+ canonical skills with alias normalization and word-boundary regex. Extremely rare or proprietary internal tool names not in the dictionary may be missed during explicit skill extraction (though captured by dense semantic embeddings).
4. **Heuristic Experience Estimation**: Years of experience are inferred from regex date spans and numeric pattern heuristics, which can be sensitive to non-standard resume formatting.

---

## 🔮 Realistic Future Improvements

1. **Learned Ranking Model (Learning-to-Rank)**: Transition from linear weighted scoring to a gradient-boosted decision tree ranker (LambdaMART / LightGBM Ranker) if click-through or recruiter rating feedback data becomes available.
2. **Transformer-Based Named Entity Recognition (NER)**: Fine-tune a lightweight token-classification model (e.g., RoBERTa or spaCy NER) specifically for resume entity extraction (tools, certifications, organizations).
3. **Dynamic Approximate Nearest Neighbor (ANN) Scaling**: Migrate from `IndexFlatIP` to `IndexIVFFlat` or HNSW if the corpus scales from 23K to millions of postings.
4. **Asynchronous Resume Ingestion**: Introduce background Celery/Redis queue workers for batch resume parsing in enterprise environments.

---

## ⚡ Quickstart & Reproducibility

### Automated One-Step Setup
```bash
chmod +x setup.sh
./setup.sh
```

### Manual Setup
```bash
# 1. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies (CPU PyTorch + scientific stack)
pip install --default-timeout=120 --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# 3. Create environment file
cp .env.example .env

# 4. Build offline index artifacts (sample size 500 for fast dev, 0 for full corpus)
python scripts/build_index.py --sample-size 500 --output-dir data/indices

# 5. Run the Streamlit web application
streamlit run app.py
```

### Run Automated Test Suite (32 Tests)
```bash
pytest tests/
```

### Run Offline Benchmark Suite
```bash
python experiments/run_experiments.py
```

---

## 📁 Repository Structure

```
ml_hackathon_claude/
├── app.py                             # Streamlit web application (clean, focused 5-step UI)
├── analyzer.py                        # Backward-compatible shim pointing to src/analyzer.py
├── skills_db.py                       # Backward-compatible shim pointing to src/nlp/skills_db.py
├── setup.sh                           # One-command automated setup script
├── requirements.txt                   # Production dependencies
├── pytest.ini                         # Test configuration
├── .env.example                       # Environment configuration template
├── data/
│   ├── all_job_post.csv               # 23,425 job postings corpus
│   └── indices/                       # Precomputed FAISS & TF-IDF artifacts
├── src/
│   ├── analyzer.py                    # Unified ML pipeline interface
│   ├── nlp/
│   │   ├── resume_parser.py           # Structured profile extraction & tenure heuristics
│   │   ├── skill_normalizer.py        # Centralized canonical skill normalization
│   │   └── skills_db.py               # 141-skill canonical database & taxonomy
│   ├── retrieval/
│   │   ├── embedding_service.py       # all-MiniLM-L6-v2 encoder & FAISS IndexFlatIP
│   │   └── hybrid.py                  # α-blended lexical + semantic candidate retrieval
│   ├── ranking/
│   │   ├── features.py                # Multi-factor candidate feature engineering
│   │   └── ranker.py                  # Interpretable weighted candidate ranker
│   ├── explainability/
│   │   ├── recommendation_explainer.py# Feature attribution & deterministic why-this-job
│   │   └── llm_assistant.py           # Grounded LLaMA 3.3 career advising layer
│   ├── ml/
│   │   └── classification.py          # Leakage-free Sklearn Pipeline & Stratified CV
│   └── evaluation/
│       └── recommendation_metrics.py  # IR metrics (P@K, Recall@K, MRR, NDCG@K)
├── experiments/
│   ├── run_experiments.py             # Reproducible offline evaluation suite
│   └── results/                       # Empirical benchmark results (CSV & JSON)
├── docs/
│   ├── ARCHITECTURE.md                # System design & two-stage retrieval documentation
│   └── INTERVIEW_GUIDE.md             # 25 in-depth technical interview Q&As
└── tests/
    ├── test_skill_extraction.py       # Normalization & extraction tests
    ├── test_recommendation.py         # Ranking & retrieval tests
    ├── test_metrics.py                # Information retrieval metrics unit tests
    ├── test_preprocessing.py          # Text tokenization & lemmatization tests
    └── test_edge_cases.py             # Edge cases, empty inputs, and boundary tests
```
