"""
ranking/features.py
Feature engineering for second-stage job candidate re-ranking.

Key ML Decisions:
1. Two-Stage Retrieval Architecture:
   - First stage (Retrieval): Hybrid fast search filters 23,425 jobs down to top 100 candidates.
   - Second stage (Ranking): Heavy, granular multi-factor feature extraction over candidate set.
2. Feature Vectors:
   - tfidf_similarity (lexical signal)
   - semantic_similarity (dense embedding signal)
   - skill_match_ratio (Jaccard-like coverage of job's required skills)
   - matched_skills_count & missing_skills_count (raw counts)
   - experience_compatibility (tenure alignment based on regex heuristic)
   - education_compatibility (degree requirement satisfaction)
"""

import re
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
try:
    from nlp.resume_parser import ResumeProfile
    from nlp.skill_normalizer import parse_job_skills_string
except ImportError:
    try:
        from src.nlp.resume_parser import ResumeProfile
        from src.nlp.skill_normalizer import parse_job_skills_string
    except ImportError:
        from resume_parser import ResumeProfile
        from skill_normalizer import parse_job_skills_string


def extract_job_required_experience(job_description: str) -> float:
    """
    Extracts required years of experience from job text using regex heuristics.
    Returns 0.0 if not specified.
    """
    desc_lower = job_description.lower()
    pattern = r'(?:minimum|at least|requires?|\+)?\s*(\d{1,2})\s*\+?\s*(?:years|yrs)(?:\s+of)?\s*(?:experience|exp)'
    matches = re.findall(pattern, desc_lower)
    if matches:
        try:
            valid = [float(m) for m in matches if 0 < float(m) <= 25]
            if valid:
                return float(min(valid))  # usually minimum required
        except ValueError:
            pass
    return 0.0


def compute_experience_compatibility(candidate_years: float, required_years: float) -> float:
    """
    Evaluates tenure compatibility score in [0.0, 1.0].
    If required_years is 0 (entry or unstated), score is 1.0 if candidate has experience, else 0.8.
    If required_years > 0:
      - candidate >= required -> 1.0
      - candidate < required  -> ratio (penalized proportionally)
    """
    if candidate_years is None:
        candidate_years = 0.0

    if required_years <= 0.0:
        return 0.9  # No explicit tenure hurdle

    if candidate_years >= required_years:
        return 1.0

    return float(max(0.2, candidate_years / required_years))


def compute_education_compatibility(candidate_degrees: List[str], job_description: str) -> float:
    """
    Checks if degrees in candidate resume satisfy degrees mentioned in job posting.
    """
    desc_lower = job_description.lower()
    requires_phd = bool(re.search(r'\b(ph\.?d|doctorate)\b', desc_lower))
    requires_master = bool(re.search(r'\b(master|m\.?s|mba|m\.?tech)\b', desc_lower))
    requires_bachelor = bool(re.search(r'\b(bachelor|b\.?s|b\.?tech|b\.?e)\b', desc_lower))

    cand_str = " ".join(candidate_degrees).lower()
    cand_clean = cand_str.replace(".", "")
    has_phd = bool(re.search(r'\b(ph\.?d|doctorate)\b', cand_str)) or "phd" in cand_clean or "doctorate" in cand_clean
    has_master = bool(re.search(r'\b(master|m\.?s|mba|m\.?tech)\b', cand_str)) or "master" in cand_clean or "ms" in cand_clean or "mba" in cand_clean or "mtech" in cand_clean
    has_bachelor = bool(re.search(r'\b(bachelor|b\.?s|b\.?tech|b\.?e)\b', cand_str)) or "bachelor" in cand_clean or "bs" in cand_clean or "btech" in cand_clean or "be" in cand_clean

    if requires_phd:
        return 1.0 if has_phd else 0.4
    if requires_master:
        return 1.0 if (has_master or has_phd) else 0.6
    if requires_bachelor:
        return 1.0 if (has_bachelor or has_master or has_phd) else 0.7

    # If job has no explicit degree requirement stated
    return 0.85


def build_candidate_ranking_features(
    candidates_df: pd.DataFrame,
    resume_profile: ResumeProfile
) -> pd.DataFrame:
    """
    Computes fine-grained ranking features for all candidate jobs.
    Returns enriched DataFrame with feature columns and matched/missing skill lists.
    """
    df = candidates_df.copy()
    candidate_skills_set = set(resume_profile.skills_normalized)

    matched_skills_list = []
    missing_skills_list = []
    skill_ratios = []
    matched_counts = []
    missing_counts = []
    exp_compatibilities = []
    edu_compatibilities = []

    for _, row in df.iterrows():
        desc = str(row.get('job_description', ''))
        raw_skill_str = str(row.get('job_skill_set', '[]'))
        job_skills = parse_job_skills_string(raw_skill_str)

        if not job_skills:
            # Fallback: check if any resume skills are explicitly in job description text (with word boundaries)
            desc_lower = desc.lower()
            matched = [
                s for s in resume_profile.skills_raw
                if re.search(r'\b' + re.escape(s) + r'\b', desc_lower)
            ]
            missing = []
            ratio = min(1.0, len(matched) / 10.0)
        else:
            matched = [s for s in job_skills if s in candidate_skills_set]
            missing = [s for s in job_skills if s not in candidate_skills_set]
            ratio = len(matched) / len(job_skills)

        # Experience & Education
        req_exp = extract_job_required_experience(desc)
        exp_comp = compute_experience_compatibility(resume_profile.years_of_experience, req_exp)
        edu_comp = compute_education_compatibility(resume_profile.education, desc)

        matched_skills_list.append(matched)
        missing_skills_list.append(missing)
        skill_ratios.append(round(ratio, 4))
        matched_counts.append(len(matched))
        missing_counts.append(len(missing))
        exp_compatibilities.append(round(exp_comp, 4))
        edu_compatibilities.append(round(edu_comp, 4))

    df['matched_skills'] = matched_skills_list
    df['missing_skills'] = missing_skills_list
    df['skill_match_ratio'] = skill_ratios
    df['matched_skills_count'] = matched_counts
    df['missing_skills_count'] = missing_counts
    df['experience_compatibility'] = exp_compatibilities
    df['education_compatibility'] = edu_compatibilities

    return df
