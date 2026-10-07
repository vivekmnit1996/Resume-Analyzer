"""
retrieval/hybrid.py
Hybrid candidate retrieval combining lexical (TF-IDF) and semantic (dense embedding) signals.

Key ML Decisions:
1. Why Lexical + Semantic?
   - Semantic embeddings excel at conceptual understanding and paraphrase detection
     (e.g., "distributed systems architect" ~ "cloud infrastructure lead").
   - Lexical TF-IDF excels at exact acronym matching, rare libraries, and domain jargon
     (e.g., "PyTorch", "Kubernetes", "FMLA", "Kafka").
   - Combining both captures the strengths of both representations (Hybrid Search).
2. Score Normalization:
   - Both TF-IDF cosine similarity and L2-normalized dense embedding dot-products
     naturally live in [0, 1] for non-negative texts.
   - We clamp scores to [0.0, 1.0] before interpolation.
3. Configurable Blending Parameter (alpha):
   - hybrid_score = alpha * lexical_score + (1 - alpha) * semantic_score
   - alpha = 1.0 -> Pure Lexical (TF-IDF)
   - alpha = 0.0 -> Pure Semantic (Embeddings)
   - alpha in (0, 1) -> Tunable Hybrid
"""

from typing import Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
try:
    from retrieval.embedding_service import VectorIndex
except ImportError:
    try:
        from src.retrieval.embedding_service import VectorIndex
    except ImportError:
        from embedding_service import VectorIndex


def compute_lexical_similarities(
    resume_text: str,
    tfidf_vectorizer: TfidfVectorizer,
    jobs_tfidf_matrix
) -> np.ndarray:
    """
    Computes cosine similarity between resume text and precomputed job TF-IDF matrix.
    Vectorizes resume using the fitted vectorizer (transform only).
    """
    resume_vec = tfidf_vectorizer.transform([resume_text])
    sims = cosine_similarity(resume_vec, jobs_tfidf_matrix).flatten()
    return np.clip(sims, 0.0, 1.0)


def compute_semantic_similarities(
    resume_embedding: np.ndarray,
    vector_index: VectorIndex
) -> np.ndarray:
    """
    Computes dense cosine similarity across all jobs in the vector index.
    """
    if resume_embedding.ndim == 1:
        q = resume_embedding.reshape(1, -1)
    else:
        q = resume_embedding

    norm = np.linalg.norm(q, axis=1, keepdims=True)
    if norm[0, 0] > 0:
        q = q / norm

    # Matrix product with normalized job vectors
    sims = np.dot(q, vector_index.vectors.T)[0]
    return np.clip(sims, 0.0, 1.0)


def hybrid_retrieve_candidates(
    resume_text: str,
    resume_embedding: np.ndarray,
    vector_index: VectorIndex,
    tfidf_vectorizer: TfidfVectorizer,
    jobs_tfidf_matrix,
    jobs_df: pd.DataFrame,
    alpha: float = 0.5,
    candidate_k: int = 100
) -> pd.DataFrame:
    """
    Performs hybrid retrieval against the corpus.

    Args:
        resume_text: Cleaned or raw resume text.
        resume_embedding: Dense embedding vector for resume.
        vector_index: Precomputed VectorIndex containing dense vectors.
        tfidf_vectorizer: Fitted TfidfVectorizer.
        jobs_tfidf_matrix: Sparse matrix of precomputed job TF-IDF vectors.
        jobs_df: DataFrame containing job postings.
        alpha: Weight for lexical similarity in [0, 1].
        candidate_k: Number of candidate jobs to retrieve for downstream ranking.

    Returns:
        DataFrame of candidate jobs with lexical_score, semantic_score, and hybrid_score.
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"Alpha must be between 0.0 and 1.0, got {alpha}")

    # 1. Lexical Similarity
    lexical_scores = compute_lexical_similarities(resume_text, tfidf_vectorizer, jobs_tfidf_matrix)

    # 2. Semantic Similarity
    semantic_scores = compute_semantic_similarities(resume_embedding, vector_index)

    # 3. Hybrid Combination
    hybrid_scores = alpha * lexical_scores + (1.0 - alpha) * semantic_scores

    # 4. Form Candidate DataFrame
    candidates = jobs_df.copy().reset_index(drop=True)
    candidates['lexical_score'] = np.round(lexical_scores, 4)
    candidates['semantic_score'] = np.round(semantic_scores, 4)
    candidates['hybrid_score'] = np.round(hybrid_scores, 4)

    # Sort descending by hybrid score
    candidates = candidates.sort_values('hybrid_score', ascending=False)

    return candidates.head(candidate_k).copy()
