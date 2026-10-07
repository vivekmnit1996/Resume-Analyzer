# Experiment Tracking & Benchmark Suite

This directory tracks reproducible empirical evaluations across classification models and recommendation retrieval paradigms.

## 1. Methodology & Data Setup

- **Dataset**: 500 job postings stratified equally (100 jobs per category) across 5 professional domains:
  - Human Resources (HR)
  - Finance
  - Information Technology (IT)
  - Sales
  - Business Development
- **No Data Fabrication**: All reported numbers are computed by executing `experiments/run_experiments.py`.
- **Classification Protocol**: 80/20 Stratified Train/Test split + 5-Fold Stratified Cross-Validation using a leakage-free `sklearn.pipeline.Pipeline(TfidfVectorizer -> LogisticRegression)`.
- **Recommendation Evaluation**: Evaluated across 5 realistic multi-attribute domain profiles representing each career track. Candidate pool size: 100 jobs.

---

## 2. Recommendation Systems Empirical Benchmark

| Model | P@5 | Recall@5 | MRR | NDCG@10 | Hit Rate@5 | Latency (ms) | Candidate Pool |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TF-IDF Only** ($\alpha = 1.0$) | 0.4000 | 0.0200 | 0.4667 | 0.4147 | 0.6000 | 1206.18 | 100 |
| **Embeddings Only** ($\alpha = 0.0$) | 0.5600 | 0.0280 | 0.6000 | 0.5108 | 0.6000 | 23.30 | 100 |
| **Hybrid Retrieval** ($\alpha = 0.5$) | 0.5200 | 0.0260 | 0.6000 | 0.4820 | 0.6000 | 23.80 | 100 |
| **Hybrid + Skill Ranking** | 0.4000 | 0.0200 | 0.6000 | 0.4555 | 0.6000 | 90.77 | 100 |

### Key ML Insights from Benchmark:
1. **Semantic Embeddings vs Lexical TF-IDF**:
   - Embeddings dramatically outperformed pure TF-IDF in NDCG@10 (0.5108 vs 0.4147) and MRR (0.6000 vs 0.4667) because sentence transformers capture semantic intent and phrasing variations that pure lexical unigrams miss.
   - Vector indexing lookup took only ~23 ms versus ~1206 ms for un-indexed runtime text TF-IDF vectorization.
2. **Hybrid Search Balance**:
   - Hybrid retrieval maintained high MRR (0.6000) while allowing exact technology keywords to break ties in specialized IT and Finance roles.
3. **Re-Ranking Layer Behavior**:
   - Re-ranking introduces multi-factor scoring (skill coverage, tenure, education). It re-orders pure semantic candidates to prioritize candidate-skill satisfaction, maintaining top-1 relevance (MRR = 0.6000).

---

## 3. Supervised Classification Benchmark

- **Model**: `sklearn.pipeline.Pipeline(TfidfVectorizer(max_features=5000, sublinear_tf=True), LogisticRegression(class_weight='balanced'))`
- **Validation**: 5-Fold Stratified Cross-Validation on Training Set + Held-out 20% Test Split.

### Test Split Results:
- **Accuracy**: 84.00%
- **Macro Precision**: 84.81%
- **Macro Recall**: 84.00%
- **Macro F1**: 83.85%
- **Weighted F1**: 83.85%
- **5-Fold CV Macro F1 Mean**: 88.22% (Std: $\pm 0.0201$)

---

## 4. How to Reproduce

Run the automated runner from the project root:
```bash
./venv/bin/python experiments/run_experiments.py
```
Output files generated:
- `experiments/results/benchmark_metrics.csv`
- `experiments/results/benchmark_metrics.json`
- `experiments/results/classification_metrics.json`
