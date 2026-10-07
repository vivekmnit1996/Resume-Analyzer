"""
ranking/ranker.py
Interpretable candidate re-ranking engine.

Key ML Decisions & Interview Talking Points:
1. Two-Stage Retrieval vs Single-Stage:
   - Stage 1 (Retrieval): Fast hybrid vector/lexical search narrows 23,425 jobs down to top 100.
   - Stage 2 (Ranking): Rich multi-factor scoring combines lexical, semantic, skill coverage,
     experience, and education compatibility.
2. Transparent Weighted Scoring (Default Ranker):
   - Linear combination of normalized feature scores.
   - Fully interpretable: each component has a direct percentage attribution.
   - Default weights:
     * skill_match_ratio: 0.35 (Direct technical skill overlap)
     * semantic_similarity: 0.30 (Deep contextual fit)
     * lexical_similarity: 0.15 (Exact keyword/acronym presence)
     * experience_compatibility: 0.10 (Tenure requirement fulfillment)
     * education_compatibility: 0.10 (Academic credential satisfaction)
3. Defensible Proxy Relevance Strategy (for offline supervised experimentation):
   - Public job datasets lack user interaction feedback (clicks, applies, recruiter screenings).
   - Limitation explicitly documented: We NEVER claim proxy labels represent true human relevance.
   - Proxy label formulation for offline training/metrics:
     proxy_relevant = 1 if (same_category AND skill_match_ratio >= 0.20 AND semantic_sim >= 0.40) else 0.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
try:
    from nlp.resume_parser import ResumeProfile
    from ranking.features import build_candidate_ranking_features
except ImportError:
    try:
        from src.nlp.resume_parser import ResumeProfile
        from src.ranking.features import build_candidate_ranking_features
    except ImportError:
        from resume_parser import ResumeProfile
        from features import build_candidate_ranking_features

DEFAULT_RANKING_WEIGHTS: Dict[str, float] = {
    'skill_match_ratio': 0.35,
    'semantic_similarity': 0.30,
    'lexical_similarity': 0.15,
    'experience_compatibility': 0.10,
    'education_compatibility': 0.10
}


def compute_weighted_ranking_score(
    row: pd.Series,
    weights: Dict[str, float] = DEFAULT_RANKING_WEIGHTS
) -> float:
    """
    Computes a linear weighted score in range [0.0, 1.0].
    Weights automatically sum-normalize to 1.0.
    """
    w_sum = sum(weights.values())
    normalized_weights = {k: v / w_sum for k, v in weights.items()}

    score = (
        normalized_weights.get('skill_match_ratio', 0.0) * float(row.get('skill_match_ratio', 0.0)) +
        normalized_weights.get('semantic_similarity', 0.0) * float(row.get('semantic_score', 0.0)) +
        normalized_weights.get('lexical_similarity', 0.0) * float(row.get('lexical_score', 0.0)) +
        normalized_weights.get('experience_compatibility', 0.0) * float(row.get('experience_compatibility', 0.0)) +
        normalized_weights.get('education_compatibility', 0.0) * float(row.get('education_compatibility', 0.0))
    )
    return float(np.clip(score, 0.0, 1.0))


class CandidateRanker:
    """
    Candidate ranking system supporting both interpretable weighted scoring
    and supervised proxy models.
    """
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        supervised_model: Optional[LogisticRegression] = None
    ):
        self.weights = weights or DEFAULT_RANKING_WEIGHTS
        self.supervised_model = supervised_model

    def rank(
        self,
        candidates_df: pd.DataFrame,
        resume_profile: ResumeProfile,
        top_k: int = 10
    ) -> pd.DataFrame:
        """
        Extracts ranking features and sorts candidate jobs descending by rank score.
        """
        if candidates_df.empty:
            return candidates_df

        # Step 1: Feature engineering
        features_df = build_candidate_ranking_features(candidates_df, resume_profile)

        # Step 2: Scoring
        if self.supervised_model is not None:
            # Supervised prediction using feature matrix
            X = features_df[[
                'lexical_score', 'semantic_score', 'skill_match_ratio',
                'matched_skills_count', 'missing_skills_count',
                'experience_compatibility', 'education_compatibility'
            ]].values
            scores = self.supervised_model.predict_proba(X)[:, 1]
        else:
            # Interpretable weighted score
            scores = [
                compute_weighted_ranking_score(row, self.weights)
                for _, row in features_df.iterrows()
            ]

        features_df['ranking_score'] = np.round(scores, 4)
        features_df['ranking_score_pct'] = (features_df['ranking_score'] * 100.0).round(1)

        # Step 3: Sort and return top K
        ranked_df = features_df.sort_values('ranking_score', ascending=False).reset_index(drop=True)
        return ranked_df.head(top_k)


def create_proxy_relevance_labels(
    candidates_df: pd.DataFrame,
    target_category: str,
    min_skill_ratio: float = 0.15,
    min_semantic: float = 0.35
) -> np.ndarray:
    """
    Constructs an explicit, defensible proxy relevance indicator for offline IR benchmarking.

    Documentation Note:
    Because real ATS user click/apply/hire logs do not exist in the open Kaggle dataset,
    this proxy marks a candidate as relevant (1) iff:
    1. The job belongs to the target domain category, AND
    2. The job shares at least `min_skill_ratio` overlapping skills with candidate, AND
    3. The semantic content exceeds `min_semantic` threshold.

    This is strictly documented as an evaluation proxy.
    """
    relevant = (
        (candidates_df['category'].astype(str).str.lower() == target_category.lower()) &
        (candidates_df['skill_match_ratio'] >= min_skill_ratio) &
        (candidates_df['semantic_score'] >= min_semantic)
    ).astype(int).values

    return relevant
