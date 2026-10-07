"""
tests/test_metrics.py
Unit tests for Information Retrieval and Classification evaluation metrics.
"""

import pytest
import numpy as np
from evaluation.recommendation_metrics import (
    precision_at_k,
    recall_at_k,
    hit_rate_at_k,
    reciprocal_rank,
    ndcg_at_k,
    compute_ranking_metrics_single_query
)


def test_precision_at_k():
    recs = ["j1", "j2", "j3", "j4", "j5"]
    relevant = {"j1", "j3", "j10"}
    # Top 3: j1, j2, j3 -> 2 hits out of 3
    assert precision_at_k(recs, relevant, k=3) == pytest.approx(2.0 / 3.0)
    # Top 5: j1, j2, j3, j4, j5 -> 2 hits out of 5
    assert precision_at_k(recs, relevant, k=5) == pytest.approx(2.0 / 5.0)
    # No relevant items
    assert precision_at_k(recs, set(), k=5) == 0.0


def test_recall_at_k():
    recs = ["j1", "j2", "j3", "j4", "j5"]
    relevant = {"j1", "j3", "j10", "j11"}  # 4 total relevant
    # Top 3 has j1, j3 -> 2 / 4 = 0.5
    assert recall_at_k(recs, relevant, k=3) == pytest.approx(0.5)
    # Empty relevant
    assert recall_at_k(recs, set(), k=3) == 0.0


def test_hit_rate_at_k():
    recs = ["j1", "j2", "j3"]
    relevant = {"j3", "j9"}
    assert hit_rate_at_k(recs, relevant, k=2) == 0.0
    assert hit_rate_at_k(recs, relevant, k=3) == 1.0


def test_reciprocal_rank():
    # First relevant item is at rank 2
    recs = ["j1", "j2", "j3"]
    relevant = {"j2", "j3"}
    assert reciprocal_rank(recs, relevant) == pytest.approx(0.5)

    # First relevant item is at rank 1
    assert reciprocal_rank(recs, {"j1"}) == pytest.approx(1.0)

    # No relevant item
    assert reciprocal_rank(recs, {"j9"}) == 0.0


def test_ndcg_at_k():
    # Ideal ranking: relevant item at rank 1
    ideal_recs = ["j1", "j2", "j3"]
    relevant = {"j1"}
    assert ndcg_at_k(ideal_recs, relevant, k=3) == pytest.approx(1.0)

    # Sub-optimal ranking: relevant item at rank 2
    sub_recs = ["j2", "j1", "j3"]
    ndcg_sub = ndcg_at_k(sub_recs, relevant, k=3)
    # DCG = 1 / log2(2 + 1) = 1 / log2(3) ~ 0.6309
    # IDCG = 1 / log2(1 + 1) = 1.0
    assert ndcg_sub < 1.0
    assert ndcg_sub > 0.0


def test_metrics_suite_empty_inputs():
    res = compute_ranking_metrics_single_query([], set(), k_values=[1, 5])
    assert res['p@1'] == 0.0
    assert res['recall@5'] == 0.0
    assert res['mrr'] == 0.0
    assert res['ndcg@5'] == 0.0
