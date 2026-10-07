"""
tests/test_preprocessing.py
Unit tests for text preprocessing and edge cases.
"""

import pytest
from analyzer import preprocess_text


def test_preprocess_standard_text():
    raw = "Senior Software Engineer developing scalable microservices in Python!"
    processed = preprocess_text(raw)
    assert "software" in processed
    assert "engineer" in processed
    assert "python" in processed
    # Stopwords should be removed
    assert "in" not in processed.split()


def test_preprocess_empty_text():
    assert preprocess_text("") == ""
    assert preprocess_text("   ") == ""


def test_preprocess_special_characters():
    raw = "!@#$%^&*()_+=-~`{}[]|;:',.<>?/"
    processed = preprocess_text(raw)
    assert processed == ""


def test_preprocess_numeric_input():
    # Should safely convert non-string or numeric representation without crash
    processed = preprocess_text(12345)
    assert processed == "12345"


def test_preprocess_lemmatization():
    raw = "Engineers were running models and testing algorithms"
    processed = preprocess_text(raw)
    tokens = processed.split()
    assert "engineer" in tokens or "engineers" in tokens
    assert "model" in tokens
    assert "algorithm" in tokens
