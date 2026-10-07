"""
evaluation/recommendation_metrics.py
Standard Information Retrieval (IR) and Recommender System evaluation metrics.

Metrics Implemented:
1. Precision@K: Fraction of top-K recommended jobs that are relevant.
2. Recall@K: Fraction of all relevant jobs that appear in the top-K.
3. Hit Rate@K: Binary indicator of whether at least one relevant job is in top-K.
4. MRR (Mean Reciprocal Rank): 1 / rank of first relevant item. Measures how high
   the top relevant recommendation is placed.
5. NDCG@K (Normalized Discounted Cumulative Gain): Measures ranking quality with
   logarithmic position discount, penalizing relevant items placed lower in the list.
"""

from typing import List, Set, Dict, Any, Union
import numpy as np


def precision_at_k(recommended_ids: List[Any], relevant_ids: Set[Any], k: int) -> float:
    """
    Computes Precision@K.
    Precision@K = |Top-K intersected with Relevant| / K
    """
    if k <= 0:
        return 0.0
    top_k = recommended_ids[:k]
    hits = sum(1 for item in top_k if item in relevant_ids)
    return float(hits / k)


def recall_at_k(recommended_ids: List[Any], relevant_ids: Set[Any], k: int) -> float:
    """
    Computes Recall@K.
    Recall@K = |Top-K intersected with Relevant| / |Relevant|
    """
    if not relevant_ids or k <= 0:
        return 0.0
    top_k = recommended_ids[:k]
    hits = sum(1 for item in top_k if item in relevant_ids)
    return float(hits / len(relevant_ids))


def hit_rate_at_k(recommended_ids: List[Any], relevant_ids: Set[Any], k: int) -> float:
    """
    Computes Hit Rate@K.
    1.0 if at least one relevant item appears in top-K, else 0.0.
    """
    top_k = recommended_ids[:k]
    return 1.0 if any(item in relevant_ids for item in top_k) else 0.0


def reciprocal_rank(recommended_ids: List[Any], relevant_ids: Set[Any]) -> float:
    """
    Computes Reciprocal Rank (RR).
    RR = 1 / rank of the first relevant item (1-indexed).
    Returns 0.0 if no relevant items are found in recommendations.
    """
    for rank, item in enumerate(recommended_ids, start=1):
        if item in relevant_ids:
            return float(1.0 / rank)
    return 0.0


def dcg_at_k(recommended_ids: List[Any], relevant_ids: Set[Any], k: int) -> float:
    """
    Computes Discounted Cumulative Gain at K (DCG@K).
    Binary relevance: rel_i = 1 if item in relevant_ids else 0.
    Formula: sum_{i=1}^k (rel_i / log2(i + 1))
    """
    dcg = 0.0
    for i, item in enumerate(recommended_ids[:k], start=1):
        if item in relevant_ids:
            dcg += 1.0 / np.log2(i + 1)
    return float(dcg)


def ndcg_at_k(recommended_ids: List[Any], relevant_ids: Set[Any], k: int) -> float:
    """
    Computes Normalized Discounted Cumulative Gain at K (NDCG@K).
    NDCG@K = DCG@K / IDCG@K
    IDCG is the DCG of an ideal ranking where all relevant items come first.
    """
    if not relevant_ids or k <= 0:
        return 0.0

    actual_dcg = dcg_at_k(recommended_ids, relevant_ids, k)

    # Ideal DCG: top min(k, |relevant|) items are all 1
    ideal_hits = min(k, len(relevant_ids))
    idcg = sum(1.0 / np.log2(i + 1) for i in range(1, ideal_hits + 1))

    if idcg <= 0.0:
        return 0.0

    return float(actual_dcg / idcg)


def compute_ranking_metrics_single_query(
    recommended_ids: List[Any],
    relevant_ids: Set[Any],
    k_values: List[int] = (1, 3, 5, 10)
) -> Dict[str, float]:
    """
    Computes a complete suite of recommendation metrics for a single query/resume.
    """
    results: Dict[str, float] = {}
    rel_set = set(relevant_ids)

    results['mrr'] = reciprocal_rank(recommended_ids, rel_set)

    for k in k_values:
        results[f'p@{k}'] = precision_at_k(recommended_ids, rel_set, k)
        results[f'recall@{k}'] = recall_at_k(recommended_ids, rel_set, k)
        results[f'hit_rate@{k}'] = hit_rate_at_k(recommended_ids, rel_set, k)
        results[f'ndcg@{k}'] = ndcg_at_k(recommended_ids, rel_set, k)

    return results


def aggregate_ranking_metrics(
    query_results: List[Dict[str, float]]
) -> Dict[str, float]:
    """
    Averages metric scores across multiple evaluation queries.
    """
    if not query_results:
        return {}

    metric_keys = query_results[0].keys()
    aggregated = {}

    for k in metric_keys:
        vals = [q[k] for q in query_results if k in q]
        aggregated[k] = float(np.mean(vals)) if vals else 0.0

    return aggregated
