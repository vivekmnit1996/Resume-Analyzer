"""
analyzer.py
Unified ML pipeline interface for Resume Analysis, Hybrid Retrieval, and Explainable Matching.

Maintains 100% backwards-compatibility for existing UI hooks while routing core
operations through production-grade, modular subpackages:
- nlp: structured parsing, canonical skill normalization
- retrieval: dense sentence embeddings, FAISS indexing, hybrid lexical/semantic search
- ranking: multi-factor candidate scoring and feature engineering
- explainability: transparent attribution and grounded LLM assistance
- ml: leakage-free classification pipeline
"""

import sys
import os
import re
import ast
import time
import logging
from typing import Tuple, List, Dict, Any, Optional

# Ensure repository root and src directory are in sys.path
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_CURRENT_DIR) if os.path.basename(_CURRENT_DIR) == "src" else _CURRENT_DIR
for _p in [_CURRENT_DIR, _ROOT_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd
import pdfplumber
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from dotenv import load_dotenv

# Load local environment configuration
load_dotenv()

try:
    from nlp.skills_db import ALL_SKILLS, SKILLS_BY_CATEGORY
except ImportError:
    from skills_db import ALL_SKILLS, SKILLS_BY_CATEGORY

from nlp.skill_normalizer import normalize_skill, normalize_skill_list, parse_job_skills_string
from nlp.resume_parser import parse_resume, extract_skills_robust, ResumeProfile
from retrieval.embedding_service import EmbeddingService, VectorIndex, FAISS_AVAILABLE
from retrieval.hybrid import hybrid_retrieve_candidates
from ranking.ranker import CandidateRanker
from ranking.features import build_candidate_ranking_features
from explainability.recommendation_explainer import generate_match_breakdown, explain_recommendation
from explainability.llm_assistant import chat_grounded_assistant, get_groq_api_key
from ml.classification import (
    train_and_evaluate_classifier,
    predict_category,
    explain_prediction,
    build_classification_pipeline
)
from evaluation.recommendation_metrics import (
    precision_at_k as ir_precision_at_k,
    recall_at_k as ir_recall_at_k,
    hit_rate_at_k as ir_hit_rate_at_k,
    reciprocal_rank as ir_reciprocal_rank,
    ndcg_at_k as ir_ndcg_at_k
)

logger = logging.getLogger(__name__)


# ─── PDF TEXT EXTRACTION ───────────────────────────────────────────

def extract_text_from_pdf(uploaded_file) -> str:
    """Extract all text from an uploaded PDF file with robust error handling."""
    text = ""
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + " "
    except Exception as e:
        logger.error(f"Error parsing PDF file: {e}")
        raise ValueError(f"Failed to read PDF file: {e}")
    return text.strip()


# ─── TEXT PREPROCESSING ────────────────────────────────────────────

def preprocess_text(text: str) -> str:
    """
    Clean and preprocess text:
    lowercase -> remove special chars -> tokenize -> remove stopwords -> lemmatize
    """
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)


# ─── SKILL EXTRACTION ─────────────────────────────────────────────

def extract_skills(raw_text: str) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Extracts skills by matching against the predefined database and canonical synonyms.
    Delegates to canonical extract_skills_robust to eliminate duplicated regex logic.
    Returns:
        found_skills: Flat list of detected raw skills.
        categorized: Dictionary mapping category name to detected skills.
    """
    found_raw, _, categorized = extract_skills_robust(raw_text)
    return found_raw, categorized


# ─── DATASET AND ARTIFACT PATH RESOLVERS ─────────────────────────

def get_dataset_path(filename: str = 'all_job_post.csv') -> str:
    """
    Resolves dataset path checking data/<filename> first, then root <filename>.
    """
    candidates = [
        os.path.join(_ROOT_DIR, 'data', filename),
        os.path.join(_ROOT_DIR, filename),
        os.path.join(os.getcwd(), 'data', filename),
        os.path.join(os.getcwd(), filename),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return os.path.join(_ROOT_DIR, 'data', filename)


def get_indices_dir() -> str:
    """
    Resolves directory containing offline precomputed index artifacts.
    """
    candidates = [
        os.path.join(_ROOT_DIR, 'data', 'indices'),
        os.path.join(os.getcwd(), 'data', 'indices'),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return os.path.join(_ROOT_DIR, 'data', 'indices')


# ─── LOAD DATASET ─────────────────────────────────────────────────

def load_jobs(sample_size: int = 500, csv_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load job postings from CSV with stratified sampling across categories.
    Checks data/all_job_post.csv first with fallback to all_job_post.csv.
    """
    data_path = csv_path or get_dataset_path('all_job_post.csv')
    df = pd.read_csv(data_path)
    df = df.dropna(subset=['job_description', 'category']).copy()
    df['job_id'] = df.get('job_id', range(len(df))).astype(str)

    if sample_size > 0 and sample_size < len(df):
        samples = []
        samples_per_cat = max(1, sample_size // df['category'].nunique())
        for cat in df['category'].unique():
            cat_df = df[df['category'] == cat]
            samples.append(cat_df.sample(n=min(len(cat_df), samples_per_cat), random_state=42))
        return pd.concat(samples).reset_index(drop=True)

    return df.reset_index(drop=True)


# ─── ADVANCED HYBRID & RANKED JOB RECOMMENDATION ──────────────────

def get_job_recommendations(
    resume_text: str,
    jobs_df: pd.DataFrame,
    top_k: int = 10,
    alpha: float = 0.5,
    use_hybrid: bool = True,
    resume_profile: Optional[ResumeProfile] = None
) -> Tuple[pd.DataFrame, Any, Any]:
    """
    Intelligent two-stage recommendation pipeline:
    Stage 1: Hybrid candidate retrieval (Lexical TF-IDF + Semantic Sentence-Transformers)
    Stage 2: Multi-factor ranking (Skills, Experience, Education, Semantics, Lexical)

    Maintains backwards compatibility with previous return signature:
    (top_jobs_df, tfidf_vectorizer, tfidf_matrix)
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from scipy import sparse
    import joblib

    # 1. Parse structured resume profile (or reuse provided profile)
    profile = resume_profile if resume_profile is not None else parse_resume(resume_text)

    # 2. Check if precomputed index artifacts exist
    indices_dir = get_indices_dir()
    meta_path = os.path.join(indices_dir, "jobs_index_meta.parquet")
    tfidf_path = os.path.join(indices_dir, "tfidf_vectorizer.joblib")
    matrix_path = os.path.join(indices_dir, "tfidf_matrix.npz")

    use_precomputed = (
        os.path.exists(meta_path) and
        os.path.exists(tfidf_path) and
        os.path.exists(matrix_path)
    )

    if use_precomputed and use_hybrid:
        try:
            vec_index = VectorIndex.load(indices_dir, base_name="jobs_index")
            vectorizer = joblib.load(tfidf_path)
            tfidf_matrix = sparse.load_npz(matrix_path)
            target_df = vec_index.metadata_df
        except Exception as e:
            logger.warning(f"Could not load precomputed index ({e}), falling back to runtime indexing.")
            use_precomputed = False

    if not use_precomputed:
        # Runtime indexing fallback
        target_df = jobs_df.copy()
        vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', sublinear_tf=True)
        tfidf_matrix = vectorizer.fit_transform(target_df['job_description'].astype(str))

        embedder = EmbeddingService()
        job_embeddings = embedder.encode(target_df['job_description'].tolist(), batch_size=64)
        vec_index = VectorIndex(dimension=job_embeddings.shape[1], use_faiss=FAISS_AVAILABLE)
        vec_index.add(job_embeddings, target_df)

    # Encode resume embedding
    embedder = EmbeddingService()
    resume_emb = embedder.encode([resume_text], normalize=True)

    # Retrieval alpha configuration
    effective_alpha = 1.0 if not use_hybrid else float(alpha)

    # Candidate Retrieval (Top 100)
    candidates_df = hybrid_retrieve_candidates(
        resume_text=resume_text,
        resume_embedding=resume_emb,
        vector_index=vec_index,
        tfidf_vectorizer=vectorizer,
        jobs_tfidf_matrix=tfidf_matrix,
        jobs_df=target_df,
        alpha=effective_alpha,
        candidate_k=min(100, len(target_df))
    )

    # Multi-Factor Re-Ranking
    ranker = CandidateRanker()
    ranked_jobs = ranker.rank(candidates_df, profile, top_k=top_k)

    # Add legacy compatibility columns
    ranked_jobs['similarity_score'] = ranked_jobs['ranking_score']
    ranked_jobs['similarity_pct'] = ranked_jobs['ranking_score_pct']

    # Synthesize explainability text and structured breakdown for each recommended job
    ranked_jobs['match_breakdown'] = [
        generate_match_breakdown(row)
        for _, row in ranked_jobs.iterrows()
    ]
    ranked_jobs['explanation'] = [
        explain_recommendation(row, candidate_years=profile.years_of_experience)
        for _, row in ranked_jobs.iterrows()
    ]

    return ranked_jobs, vectorizer, tfidf_matrix


# ─── JOB ROLE CLASSIFICATION ──────────────────────────────────────

def classify_resume(
    resume_text: str,
    jobs_df: pd.DataFrame,
    vectorizer: Any = None
) -> Tuple[str, Dict[str, float]]:
    """
    Predicts which job category the resume belongs to using a leakage-free
    sklearn Pipeline and outputs calibrated class probabilities.
    """
    pipeline = build_classification_pipeline(random_state=42)
    pipeline.fit(jobs_df['job_description'].astype(str), jobs_df['category'].astype(str))
    
    pred_category, probabilities = predict_category(pipeline, resume_text)
    return pred_category, probabilities


def explain_category_decision(
    pipeline: Any,
    resume_text: str,
    predicted_category: str
) -> List[Tuple[str, float]]:
    """
    Computes top contributing TF-IDF terms driving the category prediction.
    """
    return explain_prediction(pipeline, resume_text, target_class=predicted_category, top_n=6)


# ─── SKILL GAP ANALYSIS ───────────────────────────────────────────

def get_skill_gap(my_skills: List[str], job_skill_set_str: str) -> Tuple[List[str], List[str]]:
    """
    Compares resume skills with job skills using canonical normalization.
    """
    job_skills_norm = parse_job_skills_string(job_skill_set_str)
    my_skills_norm, _ = normalize_skill_list(my_skills)
    my_skills_set = set(my_skills_norm)

    matching = [s for s in job_skills_norm if s in my_skills_set]
    missing = [s for s in job_skills_norm if s not in my_skills_set]

    return matching, missing


# ─── CLUSTERING ────────────────────────────────────────────────────

def cluster_jobs(tfidf_matrix, labels, n_clusters: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Clusters jobs using K-Means and reduces dimensions to 2D using PCA.
    """
    dense_matrix = tfidf_matrix.toarray() if hasattr(tfidf_matrix, 'toarray') else np.array(tfidf_matrix)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(dense_matrix)

    pca = PCA(n_components=2, random_state=42)
    coords_2d = pca.fit_transform(dense_matrix)

    return coords_2d, cluster_labels


# ─── EVALUATION ────────────────────────────────────────────────────

def evaluate_classifier(jobs_df: pd.DataFrame, vectorizer: Any = None) -> Tuple[float, np.ndarray, List[str]]:
    """
    Evaluates the classifier using a strict stratified train/test split.
    Guarantees no data leakage.
    """
    pipeline, metrics, class_names = train_and_evaluate_classifier(
        jobs_df,
        text_col='job_description',
        label_col='category',
        test_size=0.2,
        random_state=42,
        cv_folds=5
    )
    conf_matrix = np.array(metrics['confusion_matrix'])
    return metrics['accuracy'], conf_matrix, class_names


def precision_at_k(recommended_jobs_df: pd.DataFrame, predicted_category: str, k: int) -> float:
    """
    Legacy Precision@K proxy: Checks proportion of top K matches that share the predicted category.
    """
    top_k = recommended_jobs_df.head(k)
    relevant = top_k[top_k['category'].astype(str).str.lower() == str(predicted_category).lower()]
    return float(len(relevant) / max(1, k))


# ─── JOB SEARCH CHATBOT (Grounded Groq LLaMA 3.3) ──────────────────

def chat_job_search(messages: List[Dict[str, str]], resume_profile: Any = None, top_jobs: Any = None) -> str:
    """
    Sends chat messages to the grounded LLaMA 3.3 70B assistant via Groq.
    Grounds responses in structured recommendation context.
    """
    return chat_grounded_assistant(messages, resume_profile=resume_profile, top_jobs=top_jobs)
