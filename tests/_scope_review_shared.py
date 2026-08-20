"""Module loader shared by the scope-review suites.

Split out of ``tests/test_scope_review.py`` when that module was divided by theme;
``REPO`` and ``_get_module`` are verbatim, so every sibling suite imports the
production modules through the same path-injecting loader it was written against.
"""

import importlib
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _get_module(name):
    sys.path.insert(0, REPO)
    return importlib.import_module(name)
