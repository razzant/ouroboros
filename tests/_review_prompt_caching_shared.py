"""The shipped prompt-cache TTL pin shared by the review-economics suites.

Split out of ``tests/test_review_prompt_caching.py`` when that module was
divided by theme; the default-TTL golden constant and the autouse pin are
verbatim, so every sibling suite runs on the shipped default unless a test
sets the global itself.
"""

from __future__ import annotations

import pytest

# The shipped global default (config.SETTINGS_DEFAULTS["OUROBOROS_PROMPT_CACHE_TTL"]):
# the review lanes' former REVIEW_CACHE_TTL constant collapsed into that setting, so
# these goldens pin the DEFAULT projection ('1h') plus the explicit-value lanes below.
_DEFAULT_GLOBAL_TTL = "1h"


@pytest.fixture(autouse=True)
def _pin_shipped_global_ttl(monkeypatch):
    """Every golden in this file runs on the SHIPPED default unless it sets the
    global itself — an ambient OUROBOROS_PROMPT_CACHE_TTL must not flip pins."""
    monkeypatch.delenv("OUROBOROS_PROMPT_CACHE_TTL", raising=False)
