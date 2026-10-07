"""
tests/test_skill_extraction.py
Unit tests for skill extraction, normalization, and deduplication.
"""

import pytest
from analyzer import extract_skills, get_skill_gap
from nlp.skill_normalizer import normalize_skill, normalize_skill_list, parse_job_skills_string
from nlp.resume_parser import parse_resume


def test_extract_skills_standard():
    text = "Proficient in Python, SQL, Docker, and Machine Learning algorithms."
    found_skills, categorized = extract_skills(text)
    assert "python" in found_skills
    assert "sql" in found_skills
    assert "docker" in found_skills
    assert "machine learning" in found_skills
    assert "Programming Languages" in categorized
    assert "python" in categorized["Programming Languages"]


def test_extract_skills_empty_input():
    found_skills, categorized = extract_skills("")
    assert found_skills == []
    assert categorized == {}


def test_extract_skills_no_known_skills():
    text = "The quick brown fox jumps over the lazy dog."
    found_skills, categorized = extract_skills(text)
    assert found_skills == []
    assert categorized == {}


def test_skill_normalization_synonyms():
    assert normalize_skill("ml") == "machine learning"
    assert normalize_skill("ML") == "machine learning"
    assert normalize_skill("machine-learning") == "machine learning"
    assert normalize_skill("sklearn") == "scikit-learn"
    assert normalize_skill("scikit learn") == "scikit-learn"
    assert normalize_skill("nlp") == "natural language processing"
    assert normalize_skill("NLP") == "natural language processing"
    assert normalize_skill("natural language processing") == "natural language processing"
    assert normalize_skill("k8s") == "kubernetes"
    assert normalize_skill("aws") == "aws"
    assert normalize_skill("amazon web services") == "aws"


def test_certifications_and_projects_extraction():
    resume_text = """
    Certifications: AWS Certified Solutions Architect, Certified Kubernetes Administrator.
    Projects:
    - Real-Time RecSys: Built hybrid recommendation system using FAISS and PyTorch.
    - Distributed Pipeline: Scalable microservices with Kafka and Docker.
    """
    profile = parse_resume(resume_text)
    assert len(profile.certifications) >= 1
    assert any("Aws Certified" in c for c in profile.certifications)
    assert len(profile.projects) >= 1


def test_skill_deduplication():
    skills = ["Python", "python", "PYTHON", "ml", "machine learning"]
    canonical, mapping = normalize_skill_list(skills)
    # python and machine learning should each appear exactly once
    assert canonical.count("python") == 1
    assert canonical.count("machine learning") == 1


def test_parse_job_skills_string():
    raw_str = "['Python', 'Machine Learning', 'AWS', 'Docker']"
    parsed = parse_job_skills_string(raw_str)
    assert "python" in parsed
    assert "machine learning" in parsed
    assert "aws" in parsed
    assert "docker" in parsed


def test_skill_gap_analysis():
    my_skills = ["Python", "SQL"]
    job_skills_str = "['Python', 'SQL', 'Docker', 'Kubernetes']"
    matching, missing = get_skill_gap(my_skills, job_skills_str)
    assert "python" in matching
    assert "sql" in matching
    assert "docker" in missing
    assert "kubernetes" in missing


def test_resume_parser_profile():
    resume_text = """
    Jane Doe - Senior Data Scientist
    Experience: 6+ years of experience in AI.
    Education: Master of Science in Computer Science.
    Skills: Python, TensorFlow, PyTorch, SQL, AWS, Docker.
    """
    profile = parse_resume(resume_text)
    assert profile.years_of_experience == 6.0
    assert "Master" in " ".join(profile.education)
    assert "python" in profile.skills_normalized
    assert "Data Scientist" in " ".join(profile.role_keywords)
