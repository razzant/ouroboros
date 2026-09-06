"""v6.73.2 — learned reasoning-effort FLOORS + value-rejection classifier markers.

The value-too-low mirror of the v6.57.0 effort ceilings: endpoints where
reasoning is MANDATORY (live incident 2026-07-19/20: google/gemini-3.5-flash on
OpenRouter answered ``400 Reasoning is mandatory for this endpoint and cannot
be disabled.`` to the safety supervisor's ``reasoning_effort="none"``, which the
old classifier did not recognize → the safety check failed CLOSED and blocked
benign tools). These tests pin: the two new marker families, the floor branch
(learn + raise + disclose + carrier preserved), the negative boundaries against
the pre-existing drop/ceiling machinery, and the durable store semantics.
"""

import time

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
def _clean_effort_caches():
    from ouroboros.llm import LLMClient

    for store in (
        LLMClient._EFFORT_CEILING_CACHE,
        LLMClient._EFFORT_FLOOR_CACHE,
        LLMClient._EFFORT_FLOOR_LOADED,
        LLMClient._REJECTED_PARAMS_CACHE,
        LLMClient._REJECTED_PARAMS_LOADED,
    ):
        store.clear()
    LLMClient._EFFORT_CEILING_LOADED.clear()
    yield
    for store in (
        LLMClient._EFFORT_CEILING_CACHE,
        LLMClient._EFFORT_FLOOR_CACHE,
        LLMClient._EFFORT_FLOOR_LOADED,
        LLMClient._REJECTED_PARAMS_CACHE,
        LLMClient._REJECTED_PARAMS_LOADED,
    ):
        store.clear()
    LLMClient._EFFORT_CEILING_LOADED.clear()


@pytest.fixture
def _isolated_data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path)
    return tmp_path


# --- classifier: the two value-rejection marker families ----------------------


def test_classifier_matches_mandatory_value_rejections():
    from ouroboros.llm import LLMClient

    # The exact live incident text.
    assert LLMClient._parameter_rejection_error(RuntimeError(LIVE_GEMINI_ERROR)) is True
    assert LLMClient._mandatory_value_rejection(RuntimeError(LIVE_GEMINI_ERROR)) is True
    # Family variants.
    assert LLMClient._parameter_rejection_error(
        RuntimeError("400: reasoning must be enabled for this model")
    ) is True
    # Range/value family (drop path, not floor).
    assert LLMClient._parameter_rejection_error(
        RuntimeError("400: temperature must be between 0 and 2")
    ) is True
    assert LLMClient._parameter_rejection_error(
        RuntimeError("400: top_p out of range")
    ) is True
    assert LLMClient._parameter_rejection_error(
        RuntimeError("400: invalid value for reasoning.effort")
    ) is True


def test_classifier_negative_boundaries():
    from ouroboros.llm import LLMClient

    # Reasoning-signature / encrypted 400s carry param names but no marker —
    # their dedicated strip handlers keep owning them.
    assert LLMClient._parameter_rejection_error(
        RuntimeError("400: Invalid signature in thinking block")
    ) is False
    assert LLMClient._parameter_rejection_error(
        RuntimeError("400: The encrypted content for item rs_abc could not be verified")
    ) is False
    # A marker with NO droppable-param name in the text never matches.
    assert LLMClient._parameter_rejection_error(
        RuntimeError("400: this endpoint is mandatory-tls only")
    ) is False
    # Bare "required" (pydantic missing-field style) is deliberately NOT a
    # marker: it may not classify as a param rejection at all.
    assert LLMClient._parameter_rejection_error(
        RuntimeError("400: reasoning.effort: field required")
    ) is False
    assert LLMClient._mandatory_value_rejection(RuntimeError("400: field required")) is False


# --- floor branch: learn + raise + disclose, carrier preserved ----------------


def test_mandatory_reasoning_400_learns_floor_and_retries(
    _clean_effort_caches, _isolated_data_dir
):
    from ouroboros import capability_evidence as ce
    from ouroboros.llm import LLMClient

    client = LLMClient(api_key="test")
    payload = {
        "model": "m",
        "messages": [],
        "extra_body": {
            "reasoning": {"effort": "none", "exclude": False},
            "provider": {"require_parameters": True},
        },
    }
    retry = client._retry_without_optional_sampling(
        payload, "google/gemini-3.5-flash", RuntimeError(LIVE_GEMINI_ERROR)
    )
    assert retry is not None
    # Carrier PRESERVED and raised — never dropped.
    assert retry["extra_body"]["reasoning"]["effort"] == "low"
    assert retry["extra_body"]["provider"] == {"require_parameters": True}
    # Floor learned in-process and durably; nothing poisoned.
    assert LLMClient._EFFORT_FLOOR_CACHE["google/gemini-3.5-flash"] == "low"
    assert ce.get_effort_floor(_isolated_data_dir, "google/gemini-3.5-flash") == "low"
    assert "google/gemini-3.5-flash" not in LLMClient._EFFORT_CEILING_CACHE
    assert not ce.get_rejected_params(_isolated_data_dir, "google/gemini-3.5-flash")
    # The learning retry itself is DISCLOSED (BIBLE P1): the pending note the
    # response normalizer merges into usage reads learned_floor.
    note = client._pop_effort_clamp_disclosure()
    assert note == {
        "requested": "none",
        "applied": "low",
        "reason": "learned_floor",
        "model": "google/gemini-3.5-flash",
    }
    # The original payload is untouched (deep copy).
    assert payload["extra_body"]["reasoning"]["effort"] == "none"


def test_mandatory_reasoning_400_through_retry_ladder(
    _clean_effort_caches, _isolated_data_dir, monkeypatch
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
    assert "google/gemini-3.5-flash" not in LLMClient._EFFORT_FLOOR_CACHE


def test_terminal_retry_death_discards_clamp_note(
    _clean_effort_caches, _isolated_data_dir, monkeypatch
):
    """A failed exact-route floor retry commits no model-global authority."""
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
    assert "google/gemini-3.5-flash" not in LLMClient._EFFORT_FLOOR_CACHE


def test_mandatory_reasoning_body_400_in_http200_recovers(
    _clean_effort_caches, _isolated_data_dir, monkeypatch
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
    assert "google/gemini-3.5-flash" not in LLMClient._EFFORT_FLOOR_CACHE
    assert client._pop_effort_clamp_disclosure() is None

    # Unmatched body-400: returned unchanged, single send, nothing learned.
    sends.clear()
    other = _FakeResp({"error": {"code": 400, "message": "malformed request body"}, "choices": None})
    resp2 = client._create_chat_completion_with_retries(
        lambda **kw: (sends.append(kw) or other), dict(kwargs), target,
    )
    assert resp2 is other and len(sends) == 1


def test_unsupported_carrier_at_none_still_drops(_clean_effort_caches, _isolated_data_dir):
    """Capability-absent vs value-forbidden need OPPOSITE remedies: a genuine
    'not supported' rejection at effort none must DROP the carrier (existing
    behavior), never raise it or learn a floor."""
    from ouroboros.llm import LLMClient

    client = LLMClient(api_key="test")
    payload = {"model": "m", "extra_body": {"reasoning": {"effort": "none"}}}
    retry = client._retry_without_optional_sampling(
        payload, "vendor/no-reasoning-model",
        RuntimeError("400: reasoning is not supported on this endpoint"),
    )
    assert retry is not None
    assert "reasoning" not in retry["extra_body"]
    assert "vendor/no-reasoning-model" not in LLMClient._EFFORT_FLOOR_CACHE
    # No phantom ceiling either (none floors below "low" — v6.61.1 rule).
    assert "vendor/no-reasoning-model" not in LLMClient._EFFORT_CEILING_CACHE


def test_mandatory_rejection_at_higher_effort_propagates(
    _clean_effort_caches, _isolated_data_dir
):
    """A mandatory-value rejection while effort is NOT bottom-tier (e.g. the
    endpoint objects to `exclude`) must neither raise nor drop-and-poison the
    carrier: it propagates unrecovered (status quo), leaving review lanes
    untouched."""
    from ouroboros import capability_evidence as ce
    from ouroboros.llm import LLMClient

    client = LLMClient(api_key="test")
    payload = {"model": "m", "extra_body": {"reasoning": {"effort": "high", "exclude": True}}}
    retry = client._retry_without_optional_sampling(
        payload, "vendor/model-h",
        RuntimeError("400: reasoning is mandatory and cannot be disabled"),
    )
    assert retry is None
    assert "vendor/model-h" not in LLMClient._EFFORT_FLOOR_CACHE
    assert not ce.get_rejected_params(_isolated_data_dir, "vendor/model-h")


def test_range_value_rejection_drops_param(_clean_effort_caches, _isolated_data_dir):
    """Owner-approved range family: a value-out-of-range 400 for an optional
    sampling param takes the EXISTING drop path and never touches floors."""
    from ouroboros.llm import LLMClient

    client = LLMClient(api_key="test")
    payload = {"model": "m", "temperature": 7.0, "messages": []}
    retry = client._retry_without_optional_sampling(
        payload, "vendor/model-t",
        RuntimeError("400: temperature must be between 0 and 2"),
    )
    assert retry is not None and "temperature" not in retry
    assert "vendor/model-t" not in LLMClient._EFFORT_FLOOR_CACHE


def test_named_param_binding_spares_coexisting_params(
    _clean_effort_caches, _isolated_data_dir
):
    """Triad r1 CRITICAL regression: on a direct-OpenAI-shaped payload carrying
    BOTH temperature and reasoning_effort, a temperature range rejection drops
    and durably remembers ONLY temperature — reasoning-effort control survives
    for later task/review calls."""
    from ouroboros import capability_evidence as ce
    from ouroboros.llm import LLMClient

    client = LLMClient(api_key="test")
    payload = {
        "model": "gpt-test",
        "messages": [],
        "temperature": 7.0,
        "reasoning_effort": "medium",
        "response_format": {"type": "json_object"},
    }
    retry = client._retry_without_optional_sampling(
        payload, "openai/gpt-named-test",
        RuntimeError("400: temperature must be between 0 and 2"),
    )
    assert retry is not None
    assert "temperature" not in retry
    assert retry["reasoning_effort"] == "medium"
    assert retry["response_format"] == {"type": "json_object"}
    assert ce.get_rejected_params(_isolated_data_dir, "openai/gpt-named-test") == {"temperature"}


def test_dotted_alias_binds_to_effort_carrier(_clean_effort_caches, _isolated_data_dir):
    """Triad r2 regression: 'invalid value for reasoning.effort' names the
    reasoning_effort carrier in dotted alias form — the named binding must
    canonicalize it, never fall back to drop-all and collaterally strip the
    coexisting optional params."""
    from ouroboros import capability_evidence as ce
    from ouroboros.llm import LLMClient

    client = LLMClient(api_key="test")
    payload = {
        "model": "gpt-test",
        "messages": [],
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
        "reasoning_effort": "high",
    }
    retry = client._retry_without_optional_sampling(
        payload, "openai/gpt-alias-test",
        RuntimeError("400: invalid value for reasoning.effort"),
    )
    assert retry is not None
    assert "reasoning_effort" not in retry  # the NAMED carrier drops...
    assert retry["temperature"] == 0.2       # ...its neighbors survive
    assert retry["response_format"] == {"type": "json_object"}
    assert ce.get_rejected_params(_isolated_data_dir, "openai/gpt-alias-test") == {"reasoning_effort"}


def test_nested_carrier_named_binding_spares_neighbors(
    _clean_effort_caches, _isolated_data_dir
):
    """Codex final-review CRITICAL regression: on a production-shaped OpenRouter
    payload (temperature + response_format + nested extra_body.reasoning), an
    'invalid value for reasoning.effort' error resolves the NESTED carrier as
    the named param — only it is dropped and remembered; the neighbors
    survive (no drop-all fallback)."""
    from ouroboros import capability_evidence as ce
    from ouroboros.llm import LLMClient

    client = LLMClient(api_key="test")
    payload = {
        "model": "m",
        "messages": [],
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
        "extra_body": {"reasoning": {"effort": "high", "exclude": False}},
    }
    retry = client._retry_without_optional_sampling(
        payload, "vendor/nested-named",
        RuntimeError("400: invalid value for reasoning.effort"),
    )
    assert retry is not None
    assert "reasoning" not in retry["extra_body"]     # the named nested carrier drops
    assert retry["temperature"] == 0.2                 # ...neighbors survive
    assert retry["response_format"] == {"type": "json_object"}
    assert ce.get_rejected_params(_isolated_data_dir, "vendor/nested-named") == {
        LLMClient._NESTED_REASONING_PARAM
    }


def test_usage_accounting_abort_discards_clamp_note(
    _clean_effort_caches, _isolated_data_dir, monkeypatch
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
    from ouroboros.usage_accounting import UsageAccountingError

    def _abort(req, fn):
        raise UsageAccountingError("budget rail")

    monkeypatch.setattr(llm_attempt_mod, "execute_physical_attempt", _abort)

    async def _abort_async(req, fn):
        raise UsageAccountingError("budget rail")

    monkeypatch.setattr(llm_attempt_mod, "execute_physical_attempt_async", _abort_async)
    client = LLMClient(api_key="test")
    target = {"usage_model": "vendor/uae-test", "resolved_model": "vendor/uae-test"}

    # Simulate a build-time clamp note (e.g. a learned floor raised none->low).
    LLMClient._EFFORT_FLOOR_CACHE["vendor/uae-test"] = "low"
    import time as _time
    LLMClient._EFFORT_FLOOR_LOADED["vendor/uae-test"] = _time.monotonic()
    assert client._clamp_effort_for_model("vendor/uae-test", "none") == "low"
    with _pytest.raises(UsageAccountingError):
        client._create_chat_completion_with_retries(lambda **kw: {"ok": True}, {"model": "m"}, target)
    assert client._pop_effort_clamp_disclosure() is None

    # The note lives in a ContextVar (isolated per thread AND per asyncio
    # task, like the reasoning pin), so stage it where chat_async does: inside
    # the coroutine that runs the driver.
    async def _stage_then_abort():
        assert client._clamp_effort_for_model("vendor/uae-test", "none") == "low"
        with _pytest.raises(UsageAccountingError):
            await client._create_chat_completion_with_retries_async(
                lambda **kw: {"ok": True}, {"model": "m"}, target,
            )
        assert client._pop_effort_clamp_disclosure() is None

    asyncio.run(_stage_then_abort())


def test_uae_on_floored_resend_discards_clamp_note(
    _clean_effort_caches, _isolated_data_dir, monkeypatch
):
    """Triad r4 regression: the CENTRAL _send discard — when the FLOORED
    body-400 resend (which just recorded a learned_floor note) is aborted by
    the accounting rail, the note is discarded before the abort propagates."""

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


def test_non_effort_mandatory_error_never_learns_floor(
    _clean_effort_caches, _isolated_data_dir
):
    """Triad r1 CRITICAL regression: a NON-effort mandatory error while the
    payload happens to carry bottom-tier reasoning must not learn a floor (and
    must not drop — exclusivity): it propagates."""
    from ouroboros import capability_evidence as ce
    from ouroboros.llm import LLMClient

    client = LLMClient(api_key="test")
    payload = {
        "model": "m",
        "messages": [],
        "response_format": {"type": "json_object"},
        "extra_body": {"reasoning": {"effort": "none"}},
    }
    retry = client._retry_without_optional_sampling(
        payload, "vendor/model-rf",
        RuntimeError("400: response_format is mandatory for this endpoint"),
    )
    assert retry is None
    assert "vendor/model-rf" not in LLMClient._EFFORT_FLOOR_CACHE
    assert not ce.get_rejected_params(_isolated_data_dir, "vendor/model-rf")


# --- clamp: [floor, ceiling] band with direction-derived disclosure -----------


def test_floor_clamps_up_with_disclosure(_clean_effort_caches, _isolated_data_dir):
    from ouroboros.llm import LLMClient

    client = LLMClient(api_key="test")
    LLMClient._EFFORT_FLOOR_CACHE["vendor/mandatory-model"] = "low"
    LLMClient._EFFORT_FLOOR_LOADED["vendor/mandatory-model"] = time.monotonic()
    assert client._clamp_effort_for_model("vendor/mandatory-model", "none") == "low"
    note = client._pop_effort_clamp_disclosure()
    assert note["requested"] == "none" and note["applied"] == "low"
    assert note["reason"] == "learned_floor"
    # In-band values are untouched and produce no disclosure.
    assert client._clamp_effort_for_model("vendor/mandatory-model", "medium") == "medium"
    assert client._pop_effort_clamp_disclosure() is None


def test_floor_ceiling_interaction(_clean_effort_caches, _isolated_data_dir):
    from ouroboros.llm import LLMClient

    client = LLMClient(api_key="test")
    LLMClient._EFFORT_FLOOR_CACHE["vendor/banded"] = "low"
    LLMClient._EFFORT_FLOOR_LOADED["vendor/banded"] = time.monotonic()
    LLMClient._EFFORT_CEILING_CACHE["vendor/banded"] = "high"
    assert client._clamp_effort_for_model("vendor/banded", "none") == "low"
    assert client._pop_effort_clamp_disclosure()["reason"] == "learned_floor"
    assert client._clamp_effort_for_model("vendor/banded", "max") == "high"
    assert client._pop_effort_clamp_disclosure()["reason"] == "learned_ceiling"
    assert client._clamp_effort_for_model("vendor/banded", "medium") == "medium"
    assert client._pop_effort_clamp_disclosure() is None


def test_floor_does_not_poison_review_lane(_clean_effort_caches, _isolated_data_dir):
    """After the floor is learned, the SAME model still honors high effort:
    floors are keyed by normalized model identity (deliberately cross-lane,
    like ceilings/rejected_params), and a floor of low never lowers or strips a
    reviewer's high-effort carrier."""
    from ouroboros.llm import LLMClient

    client = LLMClient(api_key="test")
    payload = {"model": "m", "extra_body": {"reasoning": {"effort": "none"}}}
    client._retry_without_optional_sampling(
        payload, "google/gemini-3.5-flash", RuntimeError(LIVE_GEMINI_ERROR)
    )
    client._pop_effort_clamp_disclosure()  # consume the learning-retry note
    assert client._clamp_effort_for_model("google/gemini-3.5-flash", "high") == "high"
    assert client._pop_effort_clamp_disclosure() is None
    fresh = {"extra_body": {"reasoning": {"effort": "high"}}}
    LLMClient._apply_rejected_param_cache(fresh, "google/gemini-3.5-flash")
    assert fresh["extra_body"]["reasoning"]["effort"] == "high"


def test_floor_cache_hourly_resync_evicts_expired(
    _clean_effort_caches, _isolated_data_dir, monkeypatch
):
    """Long-running processes heal like restarts: the hourly re-sync REPLACES
    the in-process floor from the durable store, so a 14-day-expired durable
    floor evicts without a restart (mirror of the rejected-params pattern)."""
    from ouroboros import capability_evidence as ce
    from ouroboros.llm import LLMClient

    ce.record_effort_floor(_isolated_data_dir, "vendor/heals", "low")
    assert LLMClient._effort_floor_for("vendor/heals") == "low"
    # Age the durable record past the TTL, then age the process cache past the
    # reload interval.
    with ce._STORE_LOCK:
        data = ce._load(_isolated_data_dir)
        data["effort_floors"]["vendor/heals"]["observed_at"] = "2020-01-01T00:00:00+00:00"
        ce._save(_isolated_data_dir, data)
    LLMClient._EFFORT_FLOOR_LOADED["vendor/heals"] = (
        time.monotonic() - LLMClient._EFFORT_FLOOR_RELOAD_SEC - 1
    )
    assert LLMClient._effort_floor_for("vendor/heals") == ""


# --- capability_evidence durable layer (first direct effort-namespace tests) --


def test_effort_floor_roundtrip_and_higher_wins(tmp_path):
    from ouroboros import capability_evidence as ce

    ce.record_effort_floor(tmp_path, "vendor/m", "low")
    assert ce.get_effort_floor(tmp_path, "vendor/m") == "low"
    ce.record_effort_floor(tmp_path, "vendor/m", "medium")
    assert ce.get_effort_floor(tmp_path, "vendor/m") == "medium"
    # A LOWER floor never silently wins within the window.
    ce.record_effort_floor(tmp_path, "vendor/m", "low")
    assert ce.get_effort_floor(tmp_path, "vendor/m") == "medium"


def test_effort_floor_expiry_and_fail_open(tmp_path):
    from ouroboros import capability_evidence as ce

    ce.record_effort_floor(tmp_path, "vendor/exp", "low")
    with ce._STORE_LOCK:
        data = ce._load(tmp_path)
        data["effort_floors"]["vendor/exp"]["observed_at"] = "2020-01-01T00:00:00+00:00"
        ce._save(tmp_path, data)
    assert ce.get_effort_floor(tmp_path, "vendor/exp") == ""
    # Fail-open on absence and on junk input.
    assert ce.get_effort_floor(tmp_path, "vendor/never-seen") == ""
    assert ce.get_effort_floor(tmp_path, "") == ""
    ce.record_effort_floor(tmp_path, "", "low")  # no-op, never raises


def test_effort_ceiling_roundtrip_parity(tmp_path):
    """First direct unit coverage of the (pre-existing) ceiling store."""
    from ouroboros import capability_evidence as ce

    ce.record_effort_ceiling(tmp_path, "vendor/c", "high")
    assert ce.get_effort_ceiling(tmp_path, "vendor/c") == "high"
    assert ce.get_effort_ceiling(tmp_path, "vendor/none") == ""


def test_load_seeds_all_namespaces(tmp_path):
    from ouroboros import capability_evidence as ce

    data = ce._load(tmp_path)
    for ns in ("probes", "owner_acks", "effort_ceilings", "effort_floors", "rejected_params"):
        assert ns in data
