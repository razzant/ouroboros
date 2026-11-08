"""v6.46.0 GAIA-forensic hardening: regression tests for the core fixes.

Covers the behaviors the review-convergence loop flagged as untested: the answer
latch precedence, budget terminalization, OR_PROVIDER routing, the narrowed
observability redaction, and the generative context-probe gating.
"""

import json


# --- OR_PROVIDER routing resolver (llm._resolve_or_provider) -----------------

def test_resolve_or_provider_presets_and_raw(monkeypatch):
    from ouroboros.llm import _resolve_or_provider

    monkeypatch.delenv("OUROBOROS_OR_PROVIDER", raising=False)
    assert _resolve_or_provider() == {}

    monkeypatch.setenv("OUROBOROS_OR_PROVIDER", "resilience")
    assert _resolve_or_provider() == {"allow_fallbacks": True}

    monkeypatch.setenv("OUROBOROS_OR_PROVIDER", "repro")
    assert _resolve_or_provider() == {"allow_fallbacks": False}

    monkeypatch.setenv("OUROBOROS_OR_PROVIDER", json.dumps({"order": ["openai"], "allow_fallbacks": False}))
    assert _resolve_or_provider() == {"order": ["openai"], "allow_fallbacks": False}

    monkeypatch.setenv("OUROBOROS_OR_PROVIDER", "not-json-not-a-preset")
    assert _resolve_or_provider() == {}  # invalid -> no routing (fail-safe)


# --- Generative context-probe gating (fail-closed, no network) ---------------
