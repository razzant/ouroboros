"""v6.73.2 — reasoning-effort FLOORS for endpoints where reasoning is MANDATORY.

Live incident 2026-07-19/20: google/gemini-3.5-flash on OpenRouter answered
``400 Reasoning is mandatory for this endpoint and cannot be disabled.`` to the
safety supervisor's ``reasoning_effort="none"`` → the safety check failed CLOSED
and blocked benign tools. These tests pin the exact-route floor recovery through
the real retry ladder (no model-global learning), the clamp-note discard on a
terminal or accounting failure, and that floor/ceiling rows older releases
stored still load but never move the dispatched effort.
"""

import pytest

LIVE_GEMINI_ERROR = (
    "Error code: 400 - {'error': {'message': 'Reasoning is mandatory for this "
    "endpoint and cannot be disabled.', 'code': 400}}"
)


class _Observed400(RuntimeError):
    status_code = 400
    body = {"error": {
        "message": "Reasoning is mandatory for this endpoint and cannot be disabled.",
        "code": 400,
    }}


def _wire_target():
    return {
        "provider": "openrouter",
        "usage_model": "google/gemini-3.5-flash",
        "resolved_model": "google/gemini-3.5-flash",
        "base_url": "https://openrouter.example/v1",
    }


@pytest.fixture
def _clean_effort_clamp_note():
    """A tier note an earlier test on this worker staged and never consumed must not reach these calls."""
    from ouroboros.llm_capability_policy import _EFFORT_CLAMP_CVAR

    _EFFORT_CLAMP_CVAR.set(None)
    yield
    _EFFORT_CLAMP_CVAR.set(None)


@pytest.fixture
def _isolated_data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path)
    return tmp_path


# --- exact-route floor recovery through the real retry ladder -----------------


def test_terminal_retry_death_discards_clamp_note(
    _clean_effort_clamp_note, _isolated_data_dir, monkeypatch
):
    """A failed exact-route floor retry leaves no clamp note behind."""
    import pytest as _pytest

    import ouroboros.llm_attempt as llm_attempt_mod
    from ouroboros.llm import LLMClient

    monkeypatch.setattr(llm_attempt_mod, "execute_physical_attempt", lambda req, fn: fn())
    client = LLMClient(api_key="test")
    sends = []

    def create_fn(**kwargs):
        sends.append(kwargs)
        raise _Observed400(LIVE_GEMINI_ERROR)

    kwargs = {
        "model": "google/gemini-3.5-flash",
        "messages": [],
        "extra_body": {"reasoning": {"effort": "none", "exclude": False}},
    }
    target = _wire_target()
    with _pytest.raises(RuntimeError):
        client._create_chat_completion_with_retries(create_fn, kwargs, target)
    assert len(sends) == 2  # one floored retry, no loop
    assert client._pop_effort_clamp_disclosure() is None


# --- floor/ceiling rows older releases stored: loadable, never dispatch authority --
