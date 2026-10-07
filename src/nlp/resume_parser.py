"""
nlp/resume_parser.py
Structured resume extraction engine.

Key Engineering Decisions:
1. Transparent Heuristics: We explicitly document that regex and rule-based
   heuristics provide an approximation, avoiding false claims of perfect parsing.
2. Structured Schema: Extracts a typed ResumeProfile (skills, education,
   experience indicators, projects, estimated tenure, and role keywords).
3. Preserves Baseline: Uses the proven dictionary-based skill matcher while
   augmenting it with normalization and synonym expansion.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
try:
    from nlp.skills_db import ALL_SKILLS, SKILLS_BY_CATEGORY
    from nlp.skill_normalizer import normalize_skill, normalize_skill_list
except ImportError:
    try:
        from src.nlp.skills_db import ALL_SKILLS, SKILLS_BY_CATEGORY
        from src.nlp.skill_normalizer import normalize_skill, normalize_skill_list
    except ImportError:
        from skills_db import ALL_SKILLS, SKILLS_BY_CATEGORY
        from skill_normalizer import normalize_skill, normalize_skill_list


@dataclass
class ResumeProfile:
    raw_text: str
    skills_raw: List[str] = field(default_factory=list)
    skills_normalized: List[str] = field(default_factory=list)
    skills_by_category: Dict[str, List[str]] = field(default_factory=dict)
    education: List[str] = field(default_factory=list)
    experience_snippets: List[str] = field(default_factory=list)
    projects: List[str] = field(default_factory=list)
    certifications: List[str] = field(default_factory=list)
    years_of_experience: Optional[float] = None
    technologies: List[str] = field(default_factory=list)
    role_keywords: List[str] = field(default_factory=list)


EDUCATION_DEGREES = [
    r"\bph\.?d\b",
    r"\bdoctorate\b",
    r"\bmaster(?:'s)?(?:\s+degree)?\b",
    r"\bm\.?s\.?\b",
    r"\bm\.?tech\b",
    r"\bmba\b",
    r"\bbachelor(?:'s)?(?:\s+degree)?\b",
    r"\bb\.?s\.?\b",
    r"\bb\.?tech\b",
    r"\bb\.?e\.?\b",
    r"\bassociate(?:'s)?(?:\s+degree)?\b",
    r"\bdiploma\b"
]

ROLE_KEYWORDS = [
    "data scientist", "machine learning engineer", "ml engineer", "software engineer",
    "backend developer", "frontend developer", "full stack developer", "data analyst",
    "business analyst", "devops engineer", "cloud architect", "product manager",
    "project manager", "hr manager", "recruiter", "financial analyst", "accountant",
    "sales executive", "business development manager", "team lead", "intern"
]

TECH_CATEGORIES = {
    "Programming Languages", "Web & Frameworks", "Data Science & ML",
    "Databases", "Cloud & DevOps"
}


def extract_skills_robust(raw_text: str) -> Tuple[List[str], List[str], Dict[str, List[str]]]:
    """
    Extracts skills by matching against predefined dictionary and canonical synonyms.
    Returns:
        skills_raw: List of detected skill names as found.
        skills_normalized: Normalized canonical skill forms.
        skills_by_category: Categorized dictionary with raw display skill names.
    """
    text_lower = raw_text.lower()
    found_raw: List[str] = []

    # 1. Match against known baseline dictionary
    for skill in ALL_SKILLS:
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            found_raw.append(skill)

    # 2. Match targeted acronyms that might otherwise be missed
    acronym_checks = ["ml", "dl", "nlp", "cv", "aws", "gcp", "k8s", "ci/cd", "ts", "bi", "crm"]
    for acr in acronym_checks:
        pattern = r'\b' + re.escape(acr) + r'\b'
        if re.search(pattern, text_lower) and acr not in found_raw:
            found_raw.append(acr)

    found_raw = list(set(found_raw))

    # 3. Categorize raw skills
    categorized: Dict[str, List[str]] = {}
    for category, skills_list in SKILLS_BY_CATEGORY.items():
        matched = [s for s in found_raw if s in skills_list]
        if matched:
            categorized[category] = matched

    # 4. Normalize to canonical form for ML ranking and retrieval
    skills_normalized, _ = normalize_skill_list(found_raw)

    return found_raw, skills_normalized, categorized


def infer_years_of_experience(text: str) -> Optional[float]:
    """
    Infers approximate years of professional experience using explicit mentions
    and date ranges. Returns None if no confident pattern matches.
    """
    text_lower = text.lower()

    # Pattern 1: Explicit statements like "5+ years of experience" or "over 3 years"
    explicit_pattern = r'(?:over|more than|\+)?\s*(\d{1,2}(?:\.\d)?)\s*\+?\s*(?:years|yrs)(?:\s+of)?\s*(?:experience|exp)'
    matches = re.findall(explicit_pattern, text_lower)
    if matches:
        try:
            years = [float(m) for m in matches if 0 < float(m) <= 40]
            if years:
                return max(years)
        except ValueError:
            pass

    # Pattern 2: Look for year spans e.g. 2018 - 2023 or 2019 - Present
    date_spans = re.findall(r'\b(20\d\d)\s*[-–—to]+\s*(20\d\d|present|current)\b', text_lower)
    total_span_years = 0.0
    current_year = 2026.0

    for start, end in date_spans:
        try:
            s_yr = float(start)
            e_yr = current_year if end in ('present', 'current') else float(end)
            if 1980 <= s_yr <= current_year and s_yr <= e_yr:
                diff = e_yr - s_yr
                if diff > total_span_years:
                    total_span_years = diff
        except ValueError:
            continue

    if total_span_years > 0.0:
        return min(total_span_years, 35.0)

    return None


def extract_education(text: str) -> List[str]:
    """Extracts degree levels and academic qualifications."""
    text_lower = text.lower()
    detected = []

    for deg_pattern in EDUCATION_DEGREES:
        if re.search(deg_pattern, text_lower):
            # Clean human readable label
            label = deg_pattern.replace(r'\b', '').replace(r'\.', '').replace(r'(?:\s+degree)?', '')
            label = label.replace(r'(?:', '').replace(r')?', '').strip()
            if label not in detected:
                detected.append(label.title())

    return detected


def extract_role_keywords(text: str) -> List[str]:
    """Identifies functional role keywords present in the text."""
    text_lower = text.lower()
    matched = []
    for role in ROLE_KEYWORDS:
        if re.search(r'\b' + re.escape(role) + r'\b', text_lower):
            matched.append(role.title())
    return list(set(matched))


def extract_certifications(text: str) -> List[str]:
    """Identifies professional certifications."""
    text_lower = text.lower()
    cert_patterns = [
        r"aws certified(?: [a-z\s]+)?",
        r"azure certified(?: [a-z\s]+)?",
        r"google cloud certified(?: [a-z\s]+)?|gcp certified",
        r"certified kubernetes (?:administrator|application developer)|cka|ckad",
        r"pmp|project management professional",
        r"certified scrum master|csm|psm",
        r"cissp|cism",
        r"comptia (?:security\+|network\+|a\+)",
        r"deep learning specialization|machine learning specialization",
        r"six sigma(?: [a-z]+ belt)?"
    ]
    detected = []
    for pat in cert_patterns:
        matches = re.findall(pat, text_lower)
        for m in matches:
            clean = re.sub(r'\s+', ' ', m).strip().title()
            if clean not in detected:
                detected.append(clean)
    return detected


def extract_projects(text: str) -> List[str]:
    """Extracts project titles or project bullet points where clearly marked."""
    projects = []
    proj_section = re.search(
        r'(?:projects|key projects|academic projects|technical projects)[\s\:\-]+(.*?)(?:education|experience|skills|certifications|$)',
        text,
        re.IGNORECASE | re.DOTALL
    )
    if proj_section:
        sec_text = proj_section.group(1).strip()
        lines = [line.strip().strip('•-* ') for line in sec_text.split('\n') if line.strip()]
        for line in lines[:5]:
            if 5 < len(line) < 90:
                projects.append(line)
    return projects


def parse_resume(raw_text: str) -> ResumeProfile:
    """
    Parses raw resume text into a structured, validated ResumeProfile object.
    """
    skills_raw, skills_normalized, skills_by_category = extract_skills_robust(raw_text)
    years_exp = infer_years_of_experience(raw_text)
    education = extract_education(raw_text)
    roles = extract_role_keywords(raw_text)
    certifications = extract_certifications(raw_text)
    projects = extract_projects(raw_text)

    # Technology subset
    tech_skills = []
    for cat in TECH_CATEGORIES:
        if cat in skills_by_category:
            tech_skills.extend(skills_by_category[cat])

    return ResumeProfile(
        raw_text=raw_text,
        skills_raw=skills_raw,
        skills_normalized=skills_normalized,
        skills_by_category=skills_by_category,
        education=education,
        projects=projects,
        certifications=certifications,
        years_of_experience=years_exp,
        technologies=tech_skills,
        role_keywords=roles
    )


# Backward-compatible alias
parse_resume_profile = parse_resume
