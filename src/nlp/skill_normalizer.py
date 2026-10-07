"""
nlp/skill_normalizer.py
Standardized canonical skill normalizer for intelligent resume matching.

Responsibilities:
- Map variations, acronyms, and punctuation styles to canonical skill representations.
- Provide bidirectional mapping: normalized skill for retrieval/matching,
  and original casing/presentation for UI rendering.
- Normalize both resume skills and job requirement skill strings.
"""

import re
from typing import List, Dict, Set, Tuple

# Canonical dictionary mapping various surface forms, abbreviations, and misspellings
# to their standard canonical representation.
SKILL_SYNONYMS: Dict[str, str] = {
    # Data Science & Machine Learning
    "ml": "machine learning",
    "machine-learning": "machine learning",
    "machinelearning": "machine learning",
    "dl": "deep learning",
    "deep-learning": "deep learning",
    "nlp": "natural language processing",
    "natural language processing": "natural language processing",
    "cv": "computer vision",
    "computer-vision": "computer vision",
    "sklearn": "scikit-learn",
    "scikit learn": "scikit-learn",
    "scikit-learn": "scikit-learn",
    "tf": "tensorflow",
    "tensor-flow": "tensorflow",
    "tensorflow": "tensorflow",
    "pytorch": "pytorch",
    "torch": "pytorch",
    "ai": "artificial intelligence",
    "artificial intelligence": "artificial intelligence",
    "generative ai": "generative ai",
    "genai": "generative ai",
    "llm": "large language models",
    "large language models": "large language models",

    # Programming Languages & Runtimes
    "py": "python",
    "python3": "python",
    "python": "python",
    "js": "javascript",
    "javascript": "javascript",
    "ts": "typescript",
    "typescript": "typescript",
    "c++": "c++",
    "cpp": "c++",
    "c#": "c#",
    "csharp": "c#",
    "golang": "go",
    "go": "go",
    "node": "node.js",
    "nodejs": "node.js",
    "node.js": "node.js",

    # Web & Frameworks
    "react": "react",
    "reactjs": "react",
    "react.js": "react",
    "react native": "react native",
    "vue": "vue.js",
    "vuejs": "vue.js",
    "vue.js": "vue.js",
    "next": "next.js",
    "nextjs": "next.js",
    "next.js": "next.js",
    "express": "express.js",
    "expressjs": "express.js",
    "express.js": "express.js",
    "django": "django",
    "flask": "flask",
    "fastapi": "fastapi",
    "spring": "spring boot",
    "springboot": "spring boot",
    "spring boot": "spring boot",

    # Cloud & DevOps
    "amazon web services": "aws",
    "aws": "aws",
    "google cloud": "gcp",
    "google cloud platform": "gcp",
    "gcp": "gcp",
    "microsoft azure": "azure",
    "azure": "azure",
    "k8s": "kubernetes",
    "kubernetes": "kubernetes",
    "docker": "docker",
    "containerization": "docker",
    "ci/cd": "ci/cd",
    "ci cd": "ci/cd",
    "cicd": "ci/cd",
    "continuous integration": "ci/cd",
    "github actions": "github actions",
    "jenkins": "jenkins",
    "terraform": "terraform",

    # Databases
    "postgres": "postgresql",
    "postgresql": "postgresql",
    "mongo": "mongodb",
    "mongodb": "mongodb",
    "mysql": "mysql",
    "redis": "redis",
    "elasticsearch": "elasticsearch",
    "elastic search": "elasticsearch",

    # Business Intelligence & Analytics
    "powerbi": "power bi",
    "power bi": "power bi",
    "tableau": "tableau",
    "bi": "business intelligence",
    "business intelligence": "business intelligence",
    "ms excel": "excel",
    "excel": "excel",
    "crm": "crm",
    "salesforce": "salesforce",
    "sfdc": "salesforce"
}


def clean_skill_string(s: str) -> str:
    """Pre-cleans a skill string by lowercasing, stripping, and normalizing whitespace/dashes."""
    if not s:
        return ""
    cleaned = s.strip().lower()
    # Normalize multiple whitespace into single space
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned


def normalize_skill(skill: str) -> str:
    """
    Normalizes a single skill string to its canonical representation.
    Falls back to cleaned input if no synonym is registered.
    """
    cleaned = clean_skill_string(skill)
    if not cleaned:
        return ""
    return SKILL_SYNONYMS.get(cleaned, cleaned)


def normalize_skill_list(skills: List[str]) -> Tuple[List[str], Dict[str, str]]:
    """
    Normalizes a list of skills.

    Returns:
        canonical_skills: Deduplicated list of normalized skill names.
        raw_to_canonical: Mapping from original observed string to canonical representation.
    """
    canonical_set: Set[str] = set()
    raw_to_canonical: Dict[str, str] = {}

    for raw in skills:
        cleaned_raw = raw.strip()
        if not cleaned_raw:
            continue
        canon = normalize_skill(cleaned_raw)
        if canon:
            canonical_set.add(canon)
            raw_to_canonical[cleaned_raw] = canon

    return sorted(list(canonical_set)), raw_to_canonical


def parse_job_skills_string(skill_set_str: str) -> List[str]:
    """
    Robustly parses and normalizes the job_skill_set field from the CSV.
    Handles python list representations, comma-delimited strings, or brackets.
    """
    if not skill_set_str or not isinstance(skill_set_str, str):
        return []

    s = skill_set_str.strip()
    # Strip brackets if present
    if s.startswith('[') and s.endswith(']'):
        s = s[1:-1]

    # Split by comma or quotes
    raw_items = re.split(r",(?=(?:[^\']*\'[^\']*\')*[^\']*$)|,(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)", s)
    parsed: List[str] = []

    for item in raw_items:
        clean_item = item.strip().strip("'\"").strip()
        if clean_item:
            norm = normalize_skill(clean_item)
            if norm and norm not in parsed:
                parsed.append(norm)

    return parsed
