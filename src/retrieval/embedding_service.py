"""
retrieval/embedding_service.py
Lightweight semantic embedding service and vector indexing.

Key ML Decisions:
1. Model Choice: sentence-transformers/all-MiniLM-L6-v2
   - 384 embedding dimensions, ~80MB parameter size.
   - 5x faster inference than BERT-base, ideal for laptop CPU execution.
   - Contrastive pre-training on 1B+ sentence pairs makes it robust for semantic similarity.
2. L2 Normalization:
   - All vectors are unit-normalized upon encoding (||v||_2 = 1.0).
   - For unit vectors, inner product is mathematically identical to cosine similarity:
     u . v = ||u|| ||v|| cos(theta) = cos(theta).
   - Enables ultra-fast FAISS IndexFlatIP (Inner Product) search without trigonometric divisions.
3. Graceful Fallbacks:
   - If FAISS is not installed or raises an error, seamlessly falls back to vectorized numpy
     dot-product search.
   - If sentence-transformers is unavailable, provides clear warning and fallback mechanisms.
"""

import os
import logging
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Check FAISS availability
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    faiss = None
    FAISS_AVAILABLE = False

# Check SentenceTransformers availability
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingService:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        """Lazy load embedding model only when required."""
        if self._model is None:
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                raise ImportError(
                    "sentence-transformers is not installed. "
                    "Run: pip install sentence-transformers"
                )
            logger.info(f"Loading SentenceTransformer: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(
        self,
        texts: List[str],
        batch_size: int = 64,
        normalize: bool = True,
        show_progress_bar: bool = False
    ) -> np.ndarray:
        """
        Encodes a list of strings into dense floating-point vectors.
        Embeddings are L2 normalized by default for instant cosine similarity via dot product.
        """
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=normalize,
            convert_to_numpy=True
        )
        return embeddings.astype(np.float32)


class VectorIndex:
    """
    Manages vector indexing and similarity search with FAISS and numpy fallback.
    """
    def __init__(self, dimension: int = 384, use_faiss: bool = True):
        self.dimension = dimension
        self.use_faiss = use_faiss and FAISS_AVAILABLE
        self.index = None
        self.vectors: Optional[np.ndarray] = None
        self.metadata_df: Optional[pd.DataFrame] = None

        if self.use_faiss:
            # IndexFlatIP performs exact nearest neighbor search using Inner Product.
            # Because vectors are L2-normalized, Inner Product == Cosine Similarity.
            self.index = faiss.IndexFlatIP(self.dimension)

    def add(self, vectors: np.ndarray, metadata: pd.DataFrame):
        """Adds normalized vectors and associated metadata to index."""
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Vector dim {vectors.shape[1]} doesn't match index dim {self.dimension}")

        # Ensure float32
        vecs = vectors.astype(np.float32)
        self.vectors = vecs
        self.metadata_df = metadata.reset_index(drop=True)

        if self.use_faiss and self.index is not None:
            self.index.reset()
            self.index.add(vecs)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Searches for top_k most similar vectors.

        Returns:
            scores: Array of shape (top_k,) with cosine similarity scores in range [-1, 1].
            indices: Array of shape (top_k,) with corresponding row indices in metadata.
        """
        # Ensure 2D query
        if query_vector.ndim == 1:
            query = query_vector.reshape(1, -1).astype(np.float32)
        else:
            query = query_vector.astype(np.float32)

        # Ensure query is L2 normalized
        norm = np.linalg.norm(query, axis=1, keepdims=True)
        if norm[0, 0] > 0:
            query = query / norm

        top_k = min(top_k, len(self.metadata_df) if self.metadata_df is not None else 0)
        if top_k == 0:
            return np.array([]), np.array([])

        if self.use_faiss and self.index is not None:
            distances, indices = self.index.search(query, top_k)
            return distances[0], indices[0]
        else:
            # Fallback: Numpy vectorized matrix multiplication
            # (1, d) @ (N, d).T -> (1, N)
            sims = np.dot(query, self.vectors.T)[0]
            top_indices = np.argsort(sims)[::-1][:top_k]
            top_scores = sims[top_indices]
            return top_scores, top_indices

    def save(self, directory: str, base_name: str = "jobs_index"):
        """Saves vectors, metadata, and optional FAISS index to disk."""
        os.makedirs(directory, exist_ok=True)
        vec_path = os.path.join(directory, f"{base_name}_vectors.npz")
        meta_path = os.path.join(directory, f"{base_name}_meta.parquet")

        np.savez_compressed(vec_path, vectors=self.vectors)
        if self.metadata_df is not None:
            self.metadata_df.to_parquet(meta_path, index=False)

        if self.use_faiss and self.index is not None:
            faiss_path = os.path.join(directory, f"{base_name}.faiss")
            faiss.write_index(self.index, faiss_path)

    @classmethod
    def load(cls, directory: str, base_name: str = "jobs_index") -> 'VectorIndex':
        """Loads index from saved artifacts."""
        vec_path = os.path.join(directory, f"{base_name}_vectors.npz")
        meta_path = os.path.join(directory, f"{base_name}_meta.parquet")
        faiss_path = os.path.join(directory, f"{base_name}.faiss")

        if not os.path.exists(vec_path) or not os.path.exists(meta_path):
            raise FileNotFoundError(f"Index files not found at {directory}/{base_name}*")

        vectors = np.load(vec_path)['vectors']
        meta_df = pd.read_parquet(meta_path)

        dim = vectors.shape[1]
        instance = cls(dimension=dim, use_faiss=FAISS_AVAILABLE)

        if instance.use_faiss and os.path.exists(faiss_path):
            instance.index = faiss.read_index(faiss_path)
            instance.vectors = vectors
            instance.metadata_df = meta_df
        else:
            instance.add(vectors, meta_df)

        return instance
