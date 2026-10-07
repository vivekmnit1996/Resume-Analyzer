# Architecture Assessment & Upgrade Plan

## 1. Executive Summary

This document provides the architectural evaluation of the **Smart Resume Analyzer and Job Recommendation System** and details the phased roadmap to upgrade it into an **Intelligent Resume-to-Job Matching and Explainable Recommendation System** for Senior ML Engineer interview readiness.

---

## 2. Current Architecture Assessment

### 2.1 Current Workflow & Data Flow
1. **Resume Ingestion**: The user uploads a PDF. `extract_text_from_pdf` uses `pdfplumber` to extract raw text.
2. **Preprocessing**: `preprocess_text` applies lowercase conversion, regex special-character removal, NLTK tokenization, stopword removal, and WordNet lemmatization.
3. **Skill Extraction**: `extract_skills` performs regex word-boundary pattern matching (`\b<skill>\b`) against a hardcoded 141-skill dictionary in `skills_db.py`.
4. **Dataset Loading**: `load_jobs(sample_size=500)` loads `all_job_post.csv` (23,425 total rows across 5 categories) and downsamples to 500 rows (100 per category).
5. **Job Recommendation**: `get_job_recommendations` preprocesses the resume and all 500 jobs on-the-fly, fits a `TfidfVectorizer` on `[processed_resume] + processed_jobs`, and calculates `cosine_similarity`.
6. **Classification**: `classify_resume` trains a `LogisticRegression` model from scratch on the 500 sample jobs during request execution and predicts category probabilities.
7. **Skill Gap Analysis**: Compares extracted resume skills with the top job's `job_skill_set` string (parsed via `ast.literal_eval`).
8. **Clustering & Visualization**: Fits `KMeans(n_clusters=5)` and `PCA(2)` on the runtime TF-IDF matrix.
9. **Evaluation**: `evaluate_classifier` splits the transformed matrix into 80/20 train/test to compute accuracy and confusion matrix.
10. **LLM Assistant**: Calls Groq API (LLaMA 3.3 70B) using a hardcoded API key in `analyzer.py`.

---

## 3. Critical Flaws & Technical Debt Identified

### 3.1 Data Leakage & Evaluation Methodology
* **Vocabulary & IDF Leakage**: The TF-IDF vectorizer fits on `[processed_resume] + processed_jobs`. In `evaluate_classifier`, `train_test_split` is performed **after** the vectorizer is already fitted on the entire dataset. Test features, term frequencies, and document frequencies bleed into training data.
* **Transient Model Training**: Logistic Regression is fitted repeatedly inside `classify_resume` and `evaluate_classifier` on arbitrary downsampled batches without caching or an isolated evaluation split.
* **Incomplete Classification Metrics**: Only raw accuracy and a confusion matrix are reported; missing class-balanced Precision, Recall, Macro F1, and Weighted F1.
* **Flawed Recommendation Evaluation**: Recommendation "Precision@K" currently tests whether the recommended job matches the classifier's predicted category (`category == predicted_category`). This conflates recommendation accuracy with category classification and lacks standard Information Retrieval metrics (P@K, Recall@K, Hit Rate@K, MRR, NDCG@K).

### 3.2 Performance & Scalability Bottlenecks
* **Brute-Force O(N) Processing**: The full corpus contains 23,425 postings, but the system must artificially sample only 500 jobs because it runs full NLTK preprocessing and TF-IDF fitting synchronously during every request.
* **Redundant Preprocessing**: `preprocess_text` is called redundantly across `get_job_recommendations`, `classify_resume`, and `evaluate_classifier`.
* **Zero Offline Indexing**: No vector store or precomputed indexing exists.

### 3.3 Skill Representation & Semantic Limitations
* **Pure Lexical TF-IDF**: Fails to capture semantic equivalence (e.g., "distributed systems" vs "scalable backend services").
* **No Skill Normalization**: Variations like "ML" vs "machine learning", "scikit-learn" vs "scikit learn", or "AWS" vs "amazon web services" are treated as distinct or missed entirely.
* **No Structured Resume Profile**: The parser extracts a flat bag-of-words list of skills without inferring experience, education, or role keywords.

### 3.4 Recommendation Ranking & Explainability
* **Single-Score Cosine Similarity**: Recommendations rely solely on text cosine similarity. No multi-factor ranking layer accounts for verified skill matches, experience alignment, or missing skills.
* **Opaque Explanations**: Recommendations provide only a percentage without component breakdown (semantic vs lexical vs skills) or feature importance.

### 3.5 Security & Configuration
* **Exposed API Secret**: `analyzer.py` contains a hardcoded Groq API key in plaintext.
* **Hardcoded Parameters**: Hyperparameters, paths, and sample sizes are scattered across procedural functions.

---

## 4. Target Production-Style Architecture

```
                                [PDF Resume]
                                      │
                                      ▼
                      [nlp/resume_parser.py]
                   (pdfplumber + structured extraction)
                                      │
                                      ▼
                        [Resume Profile Schema]
                   • skills (raw & normalized)
                   • years_of_experience (inferred)
                   • education level
                   • domain keywords
                                      │
                     ┌────────────────┴────────────────┐
                     ▼                                 ▼
         [Lexical Retrieval]               [Semantic Retrieval]
         Precomputed TF-IDF                Precomputed MiniLM-L6
         Matrix & Vectorizer               Embeddings (FAISS / Fallback)
                     │                                 │
                     └────────────────┬────────────────┘
                                      ▼
                         [retrieval/hybrid.py]
                    Hybrid Candidate Retrieval (Top 100)
                    Score = α * Lexical + (1 - α) * Semantic
                                      │
                                      ▼
                          [ranking/features.py]
                       Multi-Factor Feature Matrix:
                  • lexical_score       • semantic_score
                  • skill_overlap_ratio • matched_skills_count
                  • missing_skills_count• experience_alignment
                                      │
                                      ▼
                           [ranking/ranker.py]
                        Interpretable Ranker /
                       Configurable Scored Rank
                                      │
                                      ▼
                                [Top 10 Jobs]
                                      │
                     ┌────────────────┴────────────────┐
                     ▼                                 ▼
    [explainability/explainer.py]           [ml/classification.py]
    • Feature attribution score cards       • Leakage-free Sklearn Pipeline
    • Natural language gap reasoning        • Top contributing TF-IDF terms
                     │                                 │
                     └────────────────┬────────────────┘
                                      ▼
                             [Streamlit UI & LLM]
                         • Grounded LLM Context
                         • Transparent Metrics & Diagnostics
```

---

## 5. File Modifications & New Modules

### 5.1 Existing Files to Refactor
* `analyzer.py`: Preserve backwards-compatible API surface for `app.py`, but delegate internal logic to modular components; remove hardcoded API secrets.
* `app.py`: Enhance existing 8 tabs with Explainable Match Breakdowns, Model Comparison (TF-IDF vs Semantic vs Hybrid), Classification feature attributions, and grounded LLM assistance.
* `requirements.txt`: Add `sentence-transformers`, `faiss-cpu` (with graceful fallback), and `python-dotenv`.
* `setup.sh`: Include offline indexing run and environment template setup.
* `.gitignore`: Ensure `.env`, indexes, and cached embeddings are not committed.

### 5.2 New Files to Create
1. `docs/UPGRADE_PLAN.md`: (This document) Architecture assessment.
2. `docs/ARCHITECTURE.md`: Detailed offline vs online scalability and complexity analysis.
3. `docs/INTERVIEW_GUIDE.md`: 25 core ML/DS interview questions and rigorous answers based on this system.
4. `.env.example`: Secure environment configuration template.
5. `nlp/skill_normalizer.py`: Canonical skill dictionary mapping, synonym normalization, and normalization rules.
6. `nlp/resume_parser.py`: Structured resume extractor (skills, experience, education, role keywords).
7. `ml/classification.py`: Leak-free sklearn Pipeline (`TfidfVectorizer` + `LogisticRegression`), stratified train/test split, comprehensive evaluation metrics, and feature log-odds explainability.
8. `retrieval/embedding_service.py`: Lightweight Sentence-Transformer (`all-MiniLM-L6-v2`) embedder with vector normalization and FAISS/numpy cosine fallback.
9. `retrieval/hybrid.py`: Configurable hybrid search engine blending lexical and semantic candidate scores.
10. `ranking/features.py`: Feature engineering pipeline for candidate job ranking.
11. `ranking/ranker.py`: Interpretable scoring and ranking engine with documented proxy relevance strategy.
12. `explainability/recommendation_explainer.py`: Multi-component match score breakdown and template/heuristic explanation generator.
13. `evaluation/recommendation_metrics.py`: Ranking evaluation metrics: Precision@K, Recall@K, Hit Rate@K, MRR, NDCG@K.
14. `scripts/build_index.py`: Offline indexing CLI to preprocess jobs, fit vectorizers, compute embeddings, and build FAISS index.
15. `experiments/run_experiments.py`: Automated reproducible benchmarking suite comparing TF-IDF, Semantic, Hybrid, and Ranking.
16. `experiments/README.md`: Log of benchmark metrics, dataset parameters, and latency results.
17. `tests/`: Automated unit and integration test suite (`test_preprocessing.py`, `test_skill_extraction.py`, `test_recommendation.py`, `test_metrics.py`).

---

## 6. Rationale for Each Improvement

| Component | Why It Is Necessary | Measurable Benefit |
| :--- | :--- | :--- |
| **Leakage-Free Sklearn Pipeline** | Prevents data leakage where test vocabulary/IDF influenced training. | True generalization metrics; mathematically sound evaluation. |
| **Skill Normalization** | Resumes contain synonyms ("ML", "machine-learning") that missed exact regex. | Increases recall in skill extraction without false positives. |
| **Structured Resume Profile** | Resumes contain more than unigram skills; experience and education contextualize fit. | Enables multi-attribute ranking beyond pure text similarity. |
| **Offline Embedding & Indexing** | Avoids on-the-fly embedding of 23,425 jobs on every user request. | Cuts search latency from seconds/minutes to milliseconds; enables scaling to full corpus. |
| **Hybrid Retrieval (Lexical + Semantic)** | Embeddings capture semantics; TF-IDF captures exact acronyms and tools. | Superior candidate retrieval combining high recall with high precision. |
| **Skill-Aware Ranking Layer** | A 70% text similarity job might miss 100% of required technical skills. | Ensures top recommendations meet core functional job requirements. |
| **Standard RecSys Evaluation** | `category == predicted_category` was an invalid proxy. | Evaluates ranking quality with standard IR metrics (MRR, NDCG, P@K). |
| **Explainable Recommendations** | "Why this job?" builds user trust and makes the ML decisions auditable. | Shows actionable skill gaps and transparent score attributions. |
| **Grounded LLM Layer** | Pure LLMs hallucinate jobs and requirements. | Constrains LLM reasoning to actual computed match vectors and features. |
| **Security Hardening** | Plaintext API keys in version control pose security risks. | Protects credentials using environment variables and `.env.example`. |
