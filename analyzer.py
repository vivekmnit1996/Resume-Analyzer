"""
analyzer.py
Backward-compatible root shim.
Canonical implementation is located at src/analyzer.py.
"""

import sys
import os

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_CURRENT_DIR, "src")
for _p in [_SRC_DIR, _CURRENT_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Import all symbols from src.analyzer
from src.analyzer import *  # noqa: F401, F403
import src.analyzer as _src_analyzer

for _attr in dir(_src_analyzer):
    if not _attr.startswith("__"):
        globals()[_attr] = getattr(_src_analyzer, _attr)
