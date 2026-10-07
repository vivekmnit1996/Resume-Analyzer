"""
tests/test_edge_cases.py
Edge case and robustness validation suite.
"""

import pytest
import numpy as np
import pandas as pd
from nlp.resume_parser import parse_resume, ResumeProfile
from ranking.ranker import CandidateRanker, compute_weighted_ranking_score
from ranking.features import (
    extract_job_required_experience,
    compute_experience_compatibility,
    compute_education_compatibility,
    build_candidate_ranking_features
)
from explainability.recommendation_explainer import generate_match_breakdown, explain_recommendation
from retrieval.hybrid import compute_lexical_similarities, compute_semantic_similarities
from ml.classification import build_classification_pipeline, predict_category


def test_empty_resume_parsing():
    profile = parse_resume("")
    assert profile.skills_raw == []
    assert profile.skills_normalized == []
    assert profile.years_of_experience is None
    assert profile.education == []
    assert profile.certifications == []
    assert profile.projects == []


def test_whitespace_resume_parsing():
    profile = parse_resume("   \n\t  \n  ")
    assert profile.skills_raw == []
    assert profile.skills_normalized == []


def test_extremely_long_resume():
    # 5,000 words repetition
    long_text = "Experienced Senior Python Engineer with Docker and AWS experience. " * 500
    profile = parse_resume(long_text)
    assert "python" in profile.skills_normalized
    assert "docker" in profile.skills_normalized
    assert "aws" in profile.skills_normalized


def test_experience_compatibility_edge_cases():
    # None years
    assert compute_experience_compatibility(None, 5.0) == pytest.approx(0.2)
    # 0 required years
    assert compute_experience_compatibility(3.0, 0.0) == pytest.approx(0.9)
    # Candidate exceeds required
    assert compute_experience_compatibility(8.0, 5.0) == pytest.approx(1.0)
    # Candidate below required
    assert compute_experience_compatibility(2.0, 4.0) == pytest.approx(0.5)


def test_education_compatibility_edge_cases():
    # Candidate has PhD
    assert compute_education_compatibility(["Ph.D."], "Requires PhD in Computer Science") == 1.0
    # Candidate has Bachelor applying to PhD job
    assert compute_education_compatibility(["Bachelor"], "Requires PhD in Computer Science") == 0.4
    # Candidate has Master applying to Bachelor job
    assert compute_education_compatibility(["Master"], "Requires Bachelor degree") == 1.0
    # Job has no explicit degree requirement
    assert compute_education_compatibility([], "General software engineering role") == 0.85


def test_weighted_scoring_bounds():
    # All zeros
    zero_row = pd.Series({
        'skill_match_ratio': 0.0,
        'semantic_score': 0.0,
        'lexical_score': 0.0,
        'experience_compatibility': 0.0,
        'education_compatibility': 0.0
    })
    assert compute_weighted_ranking_score(zero_row) == 0.0

    # All ones
    one_row = pd.Series({
        'skill_match_ratio': 1.0,
        'semantic_score': 1.0,
        'lexical_score': 1.0,
        'experience_compatibility': 1.0,
        'education_compatibility': 1.0
    })
    assert compute_weighted_ranking_score(one_row) == pytest.approx(1.0)


def test_explainability_structure_integrity():
    row = pd.Series({
        'ranking_score': 0.85,
        'semantic_score': 0.80,
        'lexical_score': 0.75,
        'skill_match_ratio': 0.90,
        'experience_compatibility': 1.0,
        'education_compatibility': 1.0,
        'matched_skills': ['python', 'pytorch'],
        'missing_skills': ['docker']
    })
    breakdown = generate_match_breakdown(row)
    assert 'overall_match_pct' in breakdown
    assert breakdown['overall_match_pct'] == 85.0
    assert 'strong_factors' in breakdown
    assert 'gap_factors' in breakdown
    assert len(breakdown['strong_factors']) > 0
    assert len(breakdown['gap_factors']) > 0

    explanation = explain_recommendation(row)
    assert isinstance(explanation, str)
    assert len(explanation) > 10


def test_classification_pipeline_edge_inputs():
    pipeline = build_classification_pipeline()
    train_texts = [
        "Python data scientist building machine learning models",
        "Financial analyst conducting valuation and accounting",
        "HR recruiter managing talent acquisition and payroll"
    ]
    train_labels = ["IT", "Finance", "HR"]
    pipeline.fit(train_texts, train_labels)

    pred, probs = predict_category(pipeline, "Python machine learning")
    assert pred == "IT"
    assert "IT" in probs
    assert sum(probs.values()) == pytest.approx(100.0, abs=0.5)
