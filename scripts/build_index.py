"""
scripts/build_index.py
Offline Indexing CLI for Intelligent Resume-to-Job Matching.

Responsibilities:
1. Load job dataset (supports full 23,425 rows or sampled development subsets).
2. Clean/preprocess job texts.
3. Fit and persist TF-IDF vectorizer & sparse lexical matrix.
4. Compute dense embeddings with sentence-transformers/all-MiniLM-L6-v2.
5. Build and serialize FAISS vector index (with numpy fallback support) and metadata mapping.
"""

import os
import sys
import argparse
import time
import logging
import joblib
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

# Add parent directory and src to sys.path so modules import seamlessly
_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC_DIR = os.path.join(_ROOT_DIR, "src")
for _p in [_SRC_DIR, _ROOT_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from retrieval.embedding_service import EmbeddingService, VectorIndex, FAISS_AVAILABLE

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_index")


def resolve_data_path(path: str = "all_job_post.csv") -> str:
    """
    Checks data/all_job_post.csv first, then root all_job_post.csv.
    """
    if os.path.exists(path):
        return path
    candidates = [
        os.path.join("data", os.path.basename(path)),
        os.path.join(_ROOT_DIR, "data", os.path.basename(path)),
        os.path.join(_ROOT_DIR, os.path.basename(path)),
        os.path.join(_ROOT_DIR, "data", "all_job_post.csv"),
        os.path.join(_ROOT_DIR, "all_job_post.csv"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return path


def parse_args():
    parser = argparse.ArgumentParser(description="Build offline lexical and semantic indices.")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/all_job_post.csv",
        help="Path to job postings CSV."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/indices",
        help="Directory to save index artifacts."
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="Number of rows to sample (0 for full dataset, e.g. 500 for fast test mode)."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for sentence embedding encoding."
    )
    return parser.parse_args()


def build_offline_indices(
    data_path: str = "data/all_job_post.csv",
    output_dir: str = "data/indices",
    sample_size: int = 0,
    batch_size: int = 64
):
    start_total = time.time()
    resolved_path = resolve_data_path(data_path)
    logger.info(f"Loading job data from: {resolved_path}")
    if not os.path.exists(resolved_path):
        raise FileNotFoundError(f"Job postings file not found at {data_path} or {resolved_path}")

    df = pd.read_csv(resolved_path)
    df = df.dropna(subset=['job_description', 'category']).copy()

    if sample_size > 0 and sample_size < len(df):
        logger.info(f"Sampling {sample_size} jobs stratified across categories...")
        samples = []
        samples_per_cat = max(1, sample_size // df['category'].nunique())
        for cat in df['category'].unique():
            cat_df = df[df['category'] == cat]
            samples.append(cat_df.sample(n=min(len(cat_df), samples_per_cat), random_state=42))
        df = pd.concat(samples).reset_index(drop=True)

    df['job_id'] = df.get('job_id', range(len(df))).astype(str)
    n_jobs = len(df)
    logger.info(f"Indexing {n_jobs} jobs across {df['category'].nunique()} categories.")

    os.makedirs(output_dir, exist_ok=True)

    # ─────────────────────────────────────────────────────────────
    # 1. Lexical TF-IDF Indexing
    # ─────────────────────────────────────────────────────────────
    logger.info("Building Lexical TF-IDF index (sublinear_tf, max_features=10000)...")
    t0 = time.time()
    tfidf = TfidfVectorizer(
        max_features=10000,
        stop_words='english',
        ngram_range=(1, 2),
        sublinear_tf=True
    )
    descriptions = df['job_description'].astype(str).tolist()
    tfidf_matrix = tfidf.fit_transform(descriptions)
    logger.info(f"Lexical TF-IDF fitted in {time.time() - t0:.2f}s with shape {tfidf_matrix.shape}")

    tfidf_model_path = os.path.join(output_dir, "tfidf_vectorizer.joblib")
    tfidf_matrix_path = os.path.join(output_dir, "tfidf_matrix.npz")
    joblib.dump(tfidf, tfidf_model_path)
    sparse.save_npz(tfidf_matrix_path, tfidf_matrix)

    # ─────────────────────────────────────────────────────────────
    # 2. Dense Semantic Embeddings & FAISS Indexing
    # ─────────────────────────────────────────────────────────────
    logger.info("Initializing SentenceTransformer embedding service (all-MiniLM-L6-v2)...")
    embedder = EmbeddingService()

    logger.info(f"Encoding {n_jobs} descriptions (batch_size={batch_size})...")
    t0 = time.time()
    embeddings = embedder.encode(
        descriptions,
        batch_size=batch_size,
        normalize=True,
        show_progress_bar=True
    )
    logger.info(f"Dense embeddings encoded in {time.time() - t0:.2f}s with shape {embeddings.shape}")

    # Build Vector Index
    logger.info(f"Building VectorIndex (FAISS available: {FAISS_AVAILABLE})...")
    vec_index = VectorIndex(dimension=embeddings.shape[1], use_faiss=FAISS_AVAILABLE)
    vec_index.add(embeddings, df)
    vec_index.save(output_dir, base_name="jobs_index")

    # Save lightweight ID-to-metadata mapping
    id_map = df[['job_id', 'category', 'job_title']].copy()
    id_map.to_csv(os.path.join(output_dir, "job_id_mapping.csv"), index=False)

    total_duration = time.time() - start_total
    logger.info(f"Offline indexing complete in {total_duration:.2f}s! Artifacts stored at: {output_dir}")


if __name__ == "__main__":
    args = parse_args()
    build_offline_indices(
        data_path=args.data_path,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        batch_size=args.batch_size
    )
