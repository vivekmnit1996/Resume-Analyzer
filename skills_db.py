"""
Backward-compatible shim for skills_db.
Canonical definition lives at src/nlp/skills_db.py
"""

import sys
import os

_SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

try:
    from nlp.skills_db import ALL_SKILLS, SKILLS_BY_CATEGORY
except ImportError:
    from src.nlp.skills_db import ALL_SKILLS, SKILLS_BY_CATEGORY

__all__ = ["ALL_SKILLS", "SKILLS_BY_CATEGORY"]
