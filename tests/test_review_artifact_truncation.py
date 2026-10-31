"""Review failures retain their complete safe cause before any display bound.

Short unparseable bodies also stay readable; a marker is not a substitute for
the only retained copy of an engine's actionable refusal.
"""

from __future__ import annotations

import asyncio
import json


class _ExplodingClient:
    """Minimal async-chat-compatible stub that raises a long exception."""

    def __init__(self, error_body: str) -> None:
        self._error_body = error_body

    async def chat_async(self, *args, **kwargs):  # noqa: ARG002 — stub signature
        raise RuntimeError(self._error_body)


def _run_query_model(client, model, messages):
    """Synchronously execute _query_model so tests don't need pytest-asyncio."""
    from ouroboros.tools.review import _query_model

    async def _go():
        semaphore = asyncio.Semaphore(1)
        return await _query_model(client, model, messages, semaphore)

    return asyncio.run(_go())


def test_query_model_preserves_full_error_body_under_4000_chars():
    # 500-char error body — well above the old 200 cap, well under the 4K limit
    body = "Error code: 404 - " + "X" * 500
    client = _ExplodingClient(body)
    model, result, _headers = _run_query_model(client, "anthropic/claude-opus-4.6", [])

    assert model == "anthropic/claude-opus-4.6"
    assert isinstance(result, dict)
    # Full body must survive
    assert "X" * 500 in result["error"]
    # No omission note when under the limit
    assert "OMISSION NOTE" not in result["error"]


def test_query_model_over_limit_keeps_the_whole_safe_cause():
    secret = "sk-ant-api03-" + "A1b2C3d4E5f6G7h8" * 4
    body = "A" * 5000 + secret + "\nTAIL: measured 940000 characters; limit 900000 characters"
    client = _ExplodingClient(body)
    _, result, _ = _run_query_model(client, "anthropic/claude-opus-4.6", [])

    assert isinstance(result, dict)
    assert result["error"] == "Error: " + body.replace(secret, "***REDACTED***")
    assert secret not in result["error"] and "OMISSION NOTE" not in result["error"]


def test_parse_model_response_unparseable_preserves_full_body():
    """_parse_model_response wraps unparseable upstream responses via truncate_review_artifact."""
    from ouroboros.tools.review import _parse_model_response

    # Build a "no choices" payload whose JSON serialisation is > 200 chars but < 4K
    payload = {
        "choices": [],
        "some_field": "Y" * 400,
        "another": "Z" * 400,
    }
    parsed = _parse_model_response("anthropic/claude-opus-4.6", payload, None)

    assert parsed["verdict"] == "ERROR"
    # Full serialised body preserved (no silent clipping)
    text = parsed.get("text") or ""
    assert "Y" * 400 in text
    assert "Z" * 400 in text
    assert "OMISSION NOTE" not in text


def test_parse_model_response_malformed_choices_preserves_body():
    """_parse_model_response's (KeyError, IndexError, TypeError) branch also uses the helper."""
    from ouroboros.tools.review import _parse_model_response

    # An unexpected shape that will trigger the except clause
    bad = {"choices": [{"message": None}], "big_field": "Q" * 300}
    parsed = _parse_model_response("anthropic/claude-opus-4.6", bad, None)

    assert parsed["verdict"] == "ERROR"
    # Ensure error text preserves context (it will include `choices` and nested None)
    text = parsed.get("text") or ""
    assert "Q" * 300 in text or '"Q"' in json.dumps(bad)
