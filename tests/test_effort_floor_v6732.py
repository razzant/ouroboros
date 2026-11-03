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


def test_mandatory_reasoning_400_through_retry_ladder(
    _clean_effort_clamp_note, _isolated_data_dir, monkeypatch
):
    """End-to-end through the REAL OpenRouter retry seam
    (_create_chat_completion_with_retries with a fake create_fn): first send
    400s with the live Gemini text, the floored retry succeeds, and the second
    physical payload carries effort low with the carrier present."""
    import ouroboros.llm_attempt as llm_attempt_mod
    from ouroboros.llm import LLMClient

    monkeypatch.setattr(llm_attempt_mod, "execute_physical_attempt", lambda req, fn: fn())
    client = LLMClient(api_key="test")
    sends = []

    def create_fn(**kwargs):
        sends.append(kwargs)
        if len(sends) == 1:
            raise _Observed400(LIVE_GEMINI_ERROR)
        return {"ok": True}

    kwargs = {
        "model": "google/gemini-3.5-flash",
        "messages": [],
        "extra_body": {"reasoning": {"effort": "none", "exclude": False}},
    }
    target = _wire_target()
    resp = client._create_chat_completion_with_retries(create_fn, kwargs, target)
    assert resp == {"ok": True}
    assert len(sends) == 2
    assert sends[1]["extra_body"]["reasoning"]["effort"] == "low"
    _, usage = client._normalize_remote_response(
        {"choices": [{"message": {"role": "assistant", "content": "ok"}}],
         "usage": {"prompt_tokens": 2, "completion_tokens": 1}},
        {"provider": "openrouter", "resolved_model": "gemini-3.5-flash",
         "usage_model": "google/gemini-3.5-flash"},
        skip_cost_fetch=True,
    )
    assert "reasoning_effort_clamped" not in usage


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


def test_mandatory_reasoning_body_400_in_http200_recovers(
    _clean_effort_clamp_note, _isolated_data_dir, monkeypatch
):
    """HTTP-200 body 400 uses the same exact-route floor recovery seam."""

    class _FakeResp:
        def __init__(self, body):
            self._body = body

        def model_dump(self):
            return self._body

    import ouroboros.llm_attempt as llm_attempt_mod
    from ouroboros.llm import LLMClient

    monkeypatch.setattr(llm_attempt_mod, "execute_physical_attempt", lambda req, fn: fn())
    client = LLMClient(api_key="test")
    sends = []
    body_err = _FakeResp({
        "error": {"code": 400, "message": "Reasoning is mandatory for this endpoint and cannot be disabled."},
        "choices": None,
    })
    good = _FakeResp({"choices": [{"message": {"role": "assistant", "content": "ok"}}]})

    def create_fn(**kwargs):
        sends.append(kwargs)
        return body_err if len(sends) == 1 else good

    kwargs = {
        "model": "google/gemini-3.5-flash",
        "messages": [],
        "extra_body": {"reasoning": {"effort": "none", "exclude": False}},
    }
    target = _wire_target()
    resp = client._create_chat_completion_with_retries(create_fn, kwargs, target)
    assert resp is good
    assert len(sends) == 2
    assert sends[1]["extra_body"]["reasoning"]["effort"] == "low"
    assert client._pop_effort_clamp_disclosure() is None

    # Unmatched body-400: returned unchanged, single send, nothing learned.
    sends.clear()
    other = _FakeResp({"error": {"code": 400, "message": "malformed request body"}, "choices": None})
    resp2 = client._create_chat_completion_with_retries(
        lambda **kw: (sends.append(kw) or other), dict(kwargs), target,
    )
    assert resp2 is other and len(sends) == 1


def test_usage_accounting_abort_discards_clamp_note(
    _clean_effort_clamp_note, _isolated_data_dir, monkeypatch
):
    """Triad r2 regression (sync + async drivers): an admission failure
    (UsageAccountingError) before dispatch leaves no response to consume a
    build-time clamp note — the note must be discarded on the re-raise so it
    cannot misattach to a later non-clamping call on the same thread. The
    Anthropic driver carries the identical structural discard."""
    import asyncio

    import pytest as _pytest

    import ouroboros.llm_attempt as llm_attempt_mod
    from ouroboros.llm import LLMClient
    from ouroboros.llm_capability_policy import _EFFORT_CLAMP_CVAR
    from ouroboros.usage_accounting import UsageAccountingError

    def _abort(req, fn):
        raise UsageAccountingError("budget rail")

    monkeypatch.setattr(llm_attempt_mod, "execute_physical_attempt", _abort)

    async def _abort_async(req, fn):
        raise UsageAccountingError("budget rail")

    monkeypatch.setattr(llm_attempt_mod, "execute_physical_attempt_async", _abort_async)
    client = LLMClient(api_key="test")
    target = {"usage_model": "vendor/uae-test", "resolved_model": "vendor/uae-test"}

    # A build-time tier note, staged the way the DeepSeek/Z.ai payload builders stage it.
    note = {"requested": "none", "applied": "low", "reason": "provider_wire_mapping", "model": "vendor/uae-test"}
    _EFFORT_CLAMP_CVAR.set(dict(note))
    with _pytest.raises(UsageAccountingError):
        client._create_chat_completion_with_retries(lambda **kw: {"ok": True}, {"model": "m"}, target)
    assert client._pop_effort_clamp_disclosure() is None

    # The note lives in a ContextVar (isolated per thread AND per asyncio
    # task, like the reasoning pin), so stage it where chat_async does: inside
    # the coroutine that runs the driver.
    async def _stage_then_abort():
        _EFFORT_CLAMP_CVAR.set(dict(note))
        with _pytest.raises(UsageAccountingError):
            await client._create_chat_completion_with_retries_async(
                lambda **kw: {"ok": True}, {"model": "m"}, target,
            )
        assert client._pop_effort_clamp_disclosure() is None

    asyncio.run(_stage_then_abort())


def test_uae_on_floored_resend_discards_clamp_note(
    _clean_effort_clamp_note, _isolated_data_dir, monkeypatch
):
    """Triad r4 regression: the CENTRAL _send discard — when the exact-route
    FLOORED body-400 resend is aborted by the accounting rail, no clamp note
    survives the abort."""

    class _FakeResp:
        def __init__(self, body):
            self._body = body

        def model_dump(self):
            return self._body

    import pytest as _pytest

    import ouroboros.llm_attempt as llm_attempt_mod
    from ouroboros.llm import LLMClient
    from ouroboros.usage_accounting import UsageAccountingError

    calls = {"n": 0}

    def _gate(req, fn):
        calls["n"] += 1
        if calls["n"] == 2:
            raise UsageAccountingError("budget rail")
        return fn()

    monkeypatch.setattr(llm_attempt_mod, "execute_physical_attempt", _gate)
    client = LLMClient(api_key="test")
    body_err = _FakeResp({
        "error": {"code": 400, "message": "Reasoning is mandatory for this endpoint and cannot be disabled."},
        "choices": None,
    })
    kwargs = {
        "model": "google/gemini-3.5-flash",
        "messages": [],
        "extra_body": {"reasoning": {"effort": "none", "exclude": False}},
    }
    target = _wire_target()
    with _pytest.raises(UsageAccountingError):
        client._create_chat_completion_with_retries(lambda **kw: body_err, kwargs, target)
    assert calls["n"] == 2  # initial send + aborted floored resend
    assert client._pop_effort_clamp_disclosure() is None


# --- floor/ceiling rows older releases stored: loadable, never dispatch authority --


def test_stored_floor_and_ceiling_rows_never_move_the_dispatched_effort(_isolated_data_dir):
    from ouroboros import capability_evidence as ce
    from ouroboros.llm import LLMClient
    from tests._shared import write_legacy_capability_rows

    write_legacy_capability_rows(_isolated_data_dir, effort_floors={"vendor/banded": "low"},
                                 effort_ceilings={"vendor/banded": "high"})
    assert ce._load(_isolated_data_dir)["effort_floors"]["vendor/banded"]["floor"] == "low"
    client = LLMClient(api_key="test")
    target = client._resolve_remote_target("vendor/banded")
    for requested in ("none", "medium", "xhigh"):  # below the floor, inside, above the ceiling
        kwargs = client._build_remote_kwargs(
            target=target,
            messages=[{"role": "user", "content": "hi"}],
            reasoning_effort=requested,
            max_tokens=256,
            tool_choice="auto",
            temperature=None,
            tools=None,
            skip_capability_fetch=True,
        )
        assert kwargs["extra_body"]["reasoning"]["effort"] == requested


def test_load_seeds_all_namespaces(tmp_path):
    from ouroboros import capability_evidence as ce

    data = ce._load(tmp_path)
    for ns in ("probes", "owner_acks", "effort_ceilings", "effort_floors", "rejected_params"):
        assert ns in data
