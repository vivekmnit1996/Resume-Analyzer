"""
tests/test_recommendation.py
Unit tests for hybrid candidate retrieval and multi-factor ranking.
"""

import pytest
import pandas as pd
import numpy as np
from nlp.resume_parser import parse_resume
from ranking.ranker import CandidateRanker
from ranking.features import build_candidate_ranking_features
from explainability.recommendation_explainer import generate_match_breakdown, explain_recommendation


@pytest.fixture
def sample_candidates_df():
    return pd.DataFrame([
        {
            "job_id": "1",
            "category": "IT",
            "job_title": "ML Engineer",
            "job_description": "Requires 5 years experience in Python and PyTorch. Master's degree required.",
            "job_skill_set": "['python', 'pytorch', 'docker', 'sql']",
            "lexical_score": 0.85,
            "semantic_score": 0.90,
            "hybrid_score": 0.875
        },
        {
            "job_id": "2",
            "category": "IT",
            "job_title": "Junior Developer",
            "job_description": "Entry level position requiring Java and Git.",
            "job_skill_set": "['java', 'git']",
            "lexical_score": 0.20,
            "semantic_score": 0.35,
            "hybrid_score": 0.275
        },
        {
            "job_id": "3",
            "category": "Finance",
            "job_title": "Accountant",
            "job_description": "Accounting and tax compliance role.",
            "job_skill_set": "['accounting', 'excel', 'tax']",
            "lexical_score": 0.10,
            "semantic_score": 0.15,
            "hybrid_score": 0.125
        }
    ])


def test_recommendation_ranking_order(sample_candidates_df):
    resume_text = "Senior ML Engineer with 6 years experience in Python, PyTorch, Docker. Master of Science."
    profile = parse_resume(resume_text)
    ranker = CandidateRanker()

    ranked = ranker.rank(sample_candidates_df, profile, top_k=3)

    assert len(ranked) == 3
    # Scores must be sorted descending
    scores = ranked['ranking_score'].tolist()
    assert scores == sorted(scores, reverse=True)
    # The ML Engineer role should strictly outrank Junior Developer and Accountant
    assert ranked.iloc[0]['job_id'] == "1"


def test_empty_candidates():
    empty_df = pd.DataFrame(columns=['job_id', 'job_description', 'job_skill_set'])
    profile = parse_resume("Python developer")
    ranker = CandidateRanker()
    ranked = ranker.rank(empty_df, profile)
    assert len(ranked) == 0


def test_missing_job_description_handling():
    df = pd.DataFrame([{
        "job_id": "99",
        "category": "IT",
        "job_title": "Developer",
        "job_description": None,
        "job_skill_set": "['python']",
        "lexical_score": 0.5,
        "semantic_score": 0.5,
        "hybrid_score": 0.5
    }])
    profile = parse_resume("Python developer")
    ranker = CandidateRanker()
    ranked = ranker.rank(df, profile)
    assert len(ranked) == 1
    assert ranked.iloc[0]['ranking_score'] > 0


def test_score_breakdown_and_explanation(sample_candidates_df):
    resume_text = "Python ML Engineer with PyTorch experience"
    profile = parse_resume(resume_text)
    ranker = CandidateRanker()
    ranked = ranker.rank(sample_candidates_df, profile, top_k=1)
    top_row = ranked.iloc[0]

    breakdown = generate_match_breakdown(top_row)
    assert "overall_match_pct" in breakdown
    assert "semantic_similarity_pct" in breakdown
    assert "lexical_similarity_pct" in breakdown
    assert "skill_match_pct" in breakdown
    assert isinstance(breakdown["matched_skills"], list)

    exp_text = explain_recommendation(top_row)
    assert isinstance(exp_text, str)
    assert len(exp_text) > 20
