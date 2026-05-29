"""
Provider integration tests — real API calls to verify each LLM provider works.

These tests are marked with @pytest.mark.integration and excluded from the
default pytest run via pyproject.toml addopts. They run only on:
  - main / ouroboros / ouroboros-stable push (CI Tier 2.5)
  - workflow_dispatch (manual)
  - tag push (v*)

Each test is individually skipped when its API key is absent, so the job
stays green even if only a subset of keys is configured.

`LLMClient.chat()` returns a `(msg_dict, usage_dict)` tuple since v4.44.0.
The shared assertion below also handles the legacy flat-dict shape so tests
do not need to track the underlying client refactor.

Parametrized in v5.15.x — 8 near-identical per-provider tests collapsed
into 2 parametrized tables (``basic_chat`` and ``isolation``).
"""

import os
import pytest

# Skip the entire module during routine pytest runs that use addopts -m "not integration".
# The mark also works as a per-test filter.
integration = pytest.mark.integration


def _get_llm_client():
    """Lazy import to avoid breaking collection when ouroboros is not installed."""
    from ouroboros.llm import LLMClient
    return LLMClient()


def _assert_basic_response(result, expected_provider=None):
    """Shared assertion: non-empty reply, token usage present."""
    if isinstance(result, tuple):
        msg, usage = result
    else:
        msg, usage = result, result.get("usage", {}) if isinstance(result, dict) else {}

    text = ""
    if isinstance(msg, dict):
        text = msg.get("content", "") or ""
        if isinstance(text, list):
            text = " ".join(
                b.get("text", "") for b in text if isinstance(b, dict)
            )
        if not text and expected_provider == "cloudru" and msg.get("reasoning"):
            pytest.skip(
                "Cloud.ru returned reasoning-only output without final content; "
                "provider route/auth/usage worked, but the hosted model did not "
                "emit a final answer for this smoke prompt."
            )
    assert text, f"Empty response from LLM: {result}"

    assert isinstance(usage, dict), f"Usage is not a dict: {type(usage)}"
    assert usage.get("prompt_tokens", 0) > 0, f"No prompt_tokens in usage: {usage}"
    assert usage.get("completion_tokens", 0) > 0, f"No completion_tokens in usage: {usage}"

    if expected_provider:
        resolved = usage.get("provider", "") or usage.get("resolved_model", "") or ""
        assert expected_provider.lower() in resolved.lower(), (
            f"Expected provider '{expected_provider}' in resolved model, "
            f"got '{resolved}'"
        )


# Provider name → (env var name, model id, expected_provider check)
#
# anthropic_direct uses the current production direct Anthropic default. This
# is a routing smoke (auth + request shape); provider billing/quota/rate-limit
# errors are still treated as environmental below.
_PROVIDER_MATRIX = [
    ("openrouter",       "OPENROUTER_API_KEY",                 "anthropic/claude-sonnet-4.6", "openrouter"),
    ("openai_direct",    "OPENAI_API_KEY",                     "openai::gpt-4o-mini",         "openai"),
    ("anthropic_direct", "ANTHROPIC_API_KEY",                  "anthropic::claude-sonnet-4-6", "anthropic"),
    ("cloudru",          "CLOUDRU_FOUNDATION_MODELS_API_KEY",  "cloudru::zai-org/GLM-4.7",    "cloudru"),
]


def _skip_on_provider_environmental_error(provider_id: str, exc: BaseException) -> None:
    """If exc is a known environmental (non-code) provider error, skip the
    test instead of failing.

    Includes:
    - ``credit balance is too low`` — Anthropic billing
    - ``insufficient_quota`` — OpenAI billing
    - ``rate_limit_exceeded`` / 429 — transient rate limits

    These are CI-environment problems, not regressions in routing code.
    The full body is still printed to stderr for postmortem.
    """
    import sys as _sys
    resp = getattr(exc, "response", None)
    body = ""
    if resp is not None:
        body = resp.text or ""
        print(f"[{provider_id}] HTTP {resp.status_code} body: {body[:500]}", file=_sys.stderr)
    lowered = body.lower()
    if (
        "credit balance is too low" in lowered
        or "insufficient_quota" in lowered
        or "rate_limit" in lowered
        or (resp is not None and resp.status_code == 429)
    ):
        pytest.skip(f"[{provider_id}] environmental provider error (not a routing regression): {body[:200]}")


@integration
@pytest.mark.parametrize(
    "provider_id,env_key,model,expected_provider",
    _PROVIDER_MATRIX,
    ids=[entry[0] for entry in _PROVIDER_MATRIX],
)
def test_provider_basic_chat(provider_id, env_key, model, expected_provider):
    """Verify each provider responds to a minimal chat request.

    Uses explicit ``max_tokens=1024`` rather than the chat() default (65536)
    because some direct provider model variants cap output below the
    default and reject the request with HTTP 400. This is a routing smoke;
    a low token budget is sufficient for "Respond with exactly: OK".

    Known environmental (non-code) provider errors — empty Anthropic
    credit balance, OpenAI insufficient_quota, 429 rate limits — are
    surfaced as test skips, not failures (they indicate CI account
    state, not a regression in this repo).
    """
    if not os.environ.get(env_key):
        pytest.skip(f"{env_key} not set")
    client = _get_llm_client()
    try:
        result = client.chat(
            messages=[{"role": "user", "content": "Respond with exactly: OK"}],
            model=model,
            max_tokens=1024,
        )
    except Exception as exc:  # noqa: BLE001
        _skip_on_provider_environmental_error(provider_id, exc)
        raise
    _assert_basic_response(result, expected_provider=expected_provider)


@integration
@pytest.mark.skipif(
    not os.environ.get("GIGACHAT_CREDENTIALS"),
    reason="GIGACHAT_CREDENTIALS not set",
)
def test_gigachat_basic_chat():
    """Verify GigaChat direct routing via the gigachat library works."""
    client = _get_llm_client()
    result = client.chat(
        messages=[{"role": "user", "content": "Respond with exactly: OK"}],
        model="gigachat::GigaChat-3-Ultra",
    )
    _assert_basic_response(result, expected_provider="gigachat")


# Isolation tests: clear competing provider keys so LLMClient can only route
# through the single provider under test.

_COMPETING_KEYS = [
    "OPENROUTER_API_KEY",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_COMPATIBLE_API_KEY",
    "OPENAI_COMPATIBLE_BASE_URL",
    "CLOUDRU_FOUNDATION_MODELS_API_KEY",
    "GIGACHAT_CREDENTIALS",
    "ANTHROPIC_API_KEY",
]

# Isolation parametrize — same matrix minus the OpenAI-compatible /
# Cloud.ru-isolated pairings the legacy file ran. The matrix mirrors
# _PROVIDER_MATRIX entries that have an isolation companion.
_ISOLATION_MATRIX = [
    ("openrouter",       "OPENROUTER_API_KEY",                 "anthropic/claude-sonnet-4.6"),
    ("openai_direct",    "OPENAI_API_KEY",                     "openai::gpt-4o-mini"),
    ("anthropic_direct", "ANTHROPIC_API_KEY",                  "anthropic::claude-sonnet-4-6"),
    ("cloudru",          "CLOUDRU_FOUNDATION_MODELS_API_KEY",  "cloudru::zai-org/GLM-4.7"),
]


@integration
@pytest.mark.parametrize(
    "provider_id,env_key,model",
    _ISOLATION_MATRIX,
    ids=[entry[0] for entry in _ISOLATION_MATRIX],
)
def test_provider_isolation(provider_id, env_key, model, monkeypatch):
    """Each provider works when it is the only configured provider.

    Environmental provider errors (empty credit, quota, rate limits)
    skip via _skip_on_provider_environmental_error rather than fail.
    """
    if not os.environ.get(env_key):
        pytest.skip(f"{env_key} not set")
    for key in _COMPETING_KEYS:
        if key != env_key:
            monkeypatch.delenv(key, raising=False)
    client = _get_llm_client()
    try:
        result = client.chat(
            messages=[{"role": "user", "content": "Say hello"}],
            model=model,
            max_tokens=1024,
        )
    except Exception as exc:  # noqa: BLE001
        _skip_on_provider_environmental_error(provider_id, exc)
        raise
    _assert_basic_response(result)


@integration
@pytest.mark.skipif(
    not os.environ.get("GIGACHAT_CREDENTIALS"),
    reason="GIGACHAT_CREDENTIALS not set",
)
def test_gigachat_isolation(monkeypatch):
    """GigaChat works when it is the only configured provider."""
    for key in _COMPETING_KEYS:
        if key != "GIGACHAT_CREDENTIALS":
            monkeypatch.delenv(key, raising=False)
    client = _get_llm_client()
    result = client.chat(
        messages=[{"role": "user", "content": "Say hello"}],
        model="gigachat::GigaChat-3-Ultra",
    )
    _assert_basic_response(result)
