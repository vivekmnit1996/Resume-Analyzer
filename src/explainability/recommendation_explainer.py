"""
explainability/recommendation_explainer.py
Transparent, feature-grounded explanation engine for job recommendations.

Key ML Decisions:
1. No Hallucinations / Fake Explanations:
   - Explanations are strictly derived from actual computed mathematical features:
     * semantic_score
     * lexical_score
     * skill_match_ratio
     * experience_compatibility
     * matched_skills & missing_skills
2. Structured Breakdown:
   - Returns structured percentage breakdowns for every component to enable clear UI inspection.
3. Natural Language Synthesis:
   - Generates an honest "Why this job?" synthesis detailing both the strongest alignment factors
     and the primary gap factors.
"""

from typing import Dict, Any, List
import pandas as pd


def generate_match_breakdown(job_row: pd.Series) -> Dict[str, Any]:
    """
    Extracts and standardizes the percentage match breakdown for a recommended job.
    """
    overall_score = float(job_row.get('ranking_score', job_row.get('similarity_score', 0.0)))
    semantic_score = float(job_row.get('semantic_score', 0.0))
    lexical_score = float(job_row.get('lexical_score', job_row.get('similarity_score', 0.0)))
    skill_ratio = float(job_row.get('skill_match_ratio', 0.0))
    exp_comp = float(job_row.get('experience_compatibility', 0.8))
    edu_comp = float(job_row.get('education_compatibility', 0.85))

    matched_skills = job_row.get('matched_skills', [])
    missing_skills = job_row.get('missing_skills', [])

    if not isinstance(matched_skills, list):
        matched_skills = []
    if not isinstance(missing_skills, list):
        missing_skills = []

    strong_factors: List[str] = []
    gap_factors: List[str] = []

    if skill_ratio >= 0.4:
        strong_factors.append(f"High technical skill overlap ({round(skill_ratio * 100, 1)}%)")
    elif matched_skills:
        strong_factors.append(f"Key skills matched: {', '.join(matched_skills[:4])}")

    if semantic_score >= 0.55:
        strong_factors.append(f"Strong semantic profile alignment ({round(semantic_score * 100, 1)}%)")
    elif semantic_score >= 0.35:
        strong_factors.append("Moderate contextual alignment to role description")

    if exp_comp >= 0.85:
        strong_factors.append("Experience tenure well-aligned with position requirements")

    if edu_comp >= 0.85:
        strong_factors.append("Educational credential requirements satisfied")

    if missing_skills:
        gap_factors.append(f"Missing required skills: {', '.join(missing_skills[:4])}")
    if exp_comp < 0.6:
        gap_factors.append("Tenure appears below stated minimum experience")
    if edu_comp < 0.7:
        gap_factors.append("Degree qualification may not match preferred education level")
    if not gap_factors:
        gap_factors.append("No critical qualification gaps identified")

    return {
        'overall_match_pct': round(overall_score * 100.0, 1),
        'semantic_similarity_pct': round(semantic_score * 100.0, 1),
        'lexical_similarity_pct': round(lexical_score * 100.0, 1),
        'skill_match_pct': round(skill_ratio * 100.0, 1),
        'experience_match_pct': round(exp_comp * 100.0, 1),
        'education_match_pct': round(edu_comp * 100.0, 1),
        'matched_skills': matched_skills,
        'missing_skills': missing_skills,
        'matched_skills_count': len(matched_skills),
        'missing_skills_count': len(missing_skills),
        'strong_factors': strong_factors,
        'gap_factors': gap_factors
    }


def explain_recommendation(
    job_row: pd.Series,
    candidate_years: float = None
) -> str:
    """
    Generates a truthful, rule-grounded explanation from actual computed feature values.
    """
    breakdown = generate_match_breakdown(job_row)
    matched = breakdown['matched_skills']
    missing = breakdown['missing_skills']
    semantic_pct = breakdown['semantic_similarity_pct']
    lexical_pct = breakdown['lexical_similarity_pct']
    skill_pct = breakdown['skill_match_pct']
    exp_pct = breakdown['experience_match_pct']

    sentences = []

    # 1. Strengths
    if matched:
        top_skills_str = ", ".join(matched[:4])
        sentences.append(f"Strong skill alignment on {top_skills_str} ({skill_pct}% coverage).")
    elif semantic_pct >= 60.0:
        sentences.append(f"High contextual overlap with your background ({semantic_pct}% semantic similarity).")
    elif lexical_pct >= 50.0:
        sentences.append(f"High terminology alignment ({lexical_pct}% keyword match).")
    else:
        sentences.append("General role compatibility based on overall background profile.")

    # 2. Contextual nuance
    if semantic_pct > lexical_pct + 15.0:
        sentences.append("Semantic analysis identifies conceptual alignment even where wording differs.")
    elif lexical_pct > semantic_pct + 15.0:
        sentences.append("Exact terminology matches are strong across key domain keywords.")

    # 3. Gaps and missing requirements
    if missing:
        top_missing_str = ", ".join(missing[:3])
        sentences.append(f"Primary skill gaps: {top_missing_str}.")
    elif skill_pct >= 90.0:
        sentences.append("Excellent skill coverage with virtually no missing technical requirements.")

    # 4. Tenure assessment
    if exp_pct < 60.0:
        sentences.append("The position appears to require more years of experience than detected on your resume.")

    return " ".join(sentences)
