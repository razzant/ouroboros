"""Context overflow classification and recovery disclosure."""

import json
import time

import pytest

from ouroboros.llm import LocalContextTooLargeError
from ouroboros.loop_transport import provider_recovery_hint as _provider_recovery_hint
from ouroboros.loop_llm_call import (
    _LlmErrorContext,
    _is_context_overflow_error,
    _record_llm_call_error,
    classify_llm_exception,
)


class _TypedOverflow(RuntimeError):
    status_code = 400
    body = {"error": {"code": "context_length_exceeded", "type": "invalid_request_error"}}


def test_untyped_context_scan_excludes_rate_and_output_size_errors():
    assert _is_context_overflow_error(LocalContextTooLargeError("too big"), "")
    assert _is_context_overflow_error(Exception(), "prompt is too long for context window")
    assert not _is_context_overflow_error(Exception(), "429 rate limit exceeded")
    assert not _is_context_overflow_error(Exception(), "Rate limit: too many tokens per minute")
    # Output-size precedence is applied by the SHARED helper, so the untyped
    # scan itself rejects output-limit wording that mentions the context window.
    assert not _is_context_overflow_error(
        Exception(), "max_tokens 65536 exceeds maximum context length 32768")

    for text in (
        "max_tokens 65536 exceeds maximum context length 32768",
        "maximum output tokens exceed the context window",
        "request body too large",
    ):
        assert classify_llm_exception(RuntimeError(text), text).kind == "request_too_large"


def test_structured_context_code_wins_over_output_wording():
    result = classify_llm_exception(
        _TypedOverflow("max_tokens exceeds maximum context length"),
        "max_tokens exceeds maximum context length",
    )
    assert result.kind == "context_overflow"
    assert result.retry_same_request is False


def test_recovery_hint_uses_typed_kind_without_suggesting_owner_mode_change():
    hint = _provider_recovery_hint({"_last_llm_error_kind": "context_overflow"})
    assert "context overflowed" in hint.lower()
    assert "low context mode" not in hint.lower()


def test_remote_context_overflow_is_not_logged_as_local_or_global_mode_hint(tmp_path):
    usage = {
        "_context_profile": "owner_low",
        "_context_target_miss": True,
        "_context_automatic_pass_used": True,
    }
    ctx = _LlmErrorContext(
        task_id="task-ctx",
        task_type="task",
        execution_id="exec-1",
        round_id="round-1",
        llm_call_id="call-1",
        round_idx=1,
        attempt=0,
        model="provider/model",
        request_ref=None,
        drive_logs=tmp_path,
        event_queue=None,
        accumulated_usage=usage,
        context_fit_event_fields={
            "context_profile": "owner_low",
            "context_target_miss": True,
            "context_automatic_pass_used": True,
        },
    )

    stop_retry = _record_llm_call_error(_TypedOverflow("provider rejected request"), ctx)
    rows = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]

    assert stop_retry is True
    assert any(row["type"] == "remote_context_overflow" for row in rows)
    assert not any(row["type"] == "local_context_overflow" for row in rows)
    assert usage["_last_llm_error_kind"] == "context_overflow"
    assert "context_overflow_suggest_low" not in usage
    api_error = next(row for row in rows if row["type"] == "llm_api_error")
    assert api_error["context_profile"] == "owner_low"
    assert api_error["context_target_miss"] is True
    assert api_error["context_automatic_pass_used"] is True


class _StructuredLocalOverflow(RuntimeError):
    body = {"error": {"code": "context_window_exceeded", "type": "invalid_request_error"}}


@pytest.mark.parametrize("overflow_exc", [
    _StructuredLocalOverflow("provider rejected request"),
    RuntimeError("Error code: 400 - prompt is too long: 250000 tokens > 200000 maximum"),
])
def test_local_transport_stops_unchanged_retry_on_any_overflow_shape(monkeypatch, overflow_exc):
    """The local path raises LocalContextTooLargeError on the FIRST attempt for
    every structured overflow code and message marker — an identical over-window
    payload is never resent (the old path matched one literal code only)."""
    from ouroboros import llm as llm_mod
    from ouroboros import llm_local as llm_local_mod

    calls = {"n": 0}

    def _fake_execute(request, send, before):
        calls["n"] += 1
        raise overflow_exc

    monkeypatch.setattr(llm_local_mod, "_execute_candidate", _fake_execute)
    monkeypatch.setattr(llm_local_mod, "_attempt_request", lambda *a, **k: None)
    client = llm_mod.LLMClient.__new__(llm_mod.LLMClient)
    monkeypatch.setattr(client, "_get_local_client", lambda: object(), raising=False)
    monkeypatch.setattr(
        client, "_normalize_system_message_placement", lambda m: list(m), raising=False)
    monkeypatch.setattr(
        client, "_strip_openrouter_roundtrip_metadata", lambda m: list(m), raising=False)
    monkeypatch.setattr(
        client, "_copy_messages_with_cache_policy",
        lambda m, **k: [dict(x) for x in m], raising=False)

    with pytest.raises(llm_mod.LocalContextTooLargeError):
        client._chat_local([{"role": "user", "content": "hi"}], None, 512, "auto")
    assert calls["n"] == 1


def test_local_output_limit_error_takes_normal_retry_path_not_overflow(monkeypatch):
    """An OUTPUT-limit rejection ("max_tokens ... exceeds maximum context
    length ...") must NOT be classified as a context overflow by the local
    transport: no LocalContextTooLargeError, and the original provider error
    surfaces UNCHANGED to the caller's retry policy (S1 N-1 misclassification
    probe). The lane makes exactly one physical attempt either way."""
    from ouroboros import llm as llm_mod
    from ouroboros import llm_local as llm_local_mod

    output_limit_exc = RuntimeError("max_tokens 65536 exceeds maximum context length 32768")
    calls = {"n": 0}

    def _fake_execute(request, send, before):
        calls["n"] += 1
        raise output_limit_exc

    monkeypatch.setattr(llm_local_mod, "_execute_candidate", _fake_execute)
    monkeypatch.setattr(llm_local_mod, "_attempt_request", lambda *a, **k: None)
    client = llm_mod.LLMClient.__new__(llm_mod.LLMClient)
    monkeypatch.setattr(client, "_get_local_client", lambda: object(), raising=False)
    monkeypatch.setattr(
        client, "_normalize_system_message_placement", lambda m: list(m), raising=False)
    monkeypatch.setattr(
        client, "_strip_openrouter_roundtrip_metadata", lambda m: list(m), raising=False)
    monkeypatch.setattr(
        client, "_copy_messages_with_cache_policy",
        lambda m, **k: [dict(x) for x in m], raising=False)

    with pytest.raises(RuntimeError) as excinfo:
        client._chat_local([{"role": "user", "content": "hi"}], None, 512, "auto")
    assert excinfo.value is output_limit_exc
    assert not isinstance(excinfo.value, llm_mod.LocalContextTooLargeError)
    assert calls["n"] == 1


def test_local_retry_does_not_inherit_unrelated_physical_capture(monkeypatch, tmp_path):
    """A missing exception-owned capture must not borrow a prior operation's custody."""
    from ouroboros import llm as llm_mod
    from ouroboros import llm_local as llm_local_mod
    from ouroboros import usage_accounting as ua

    # Leave an unresolved capture in the current ContextVar, as a previous
    # provider call would.  The compatibility executor below then raises an
    # ordinary local error without entering the physical-attempt seam.
    with pytest.raises(RuntimeError):
        ua.execute_physical_attempt(
            ua.AttemptRequest(
                model="seed-model", provider="local", reservation_usd=0.0,
                drive_root=tmp_path, task_id="seed-task",
            ),
            lambda: (_ for _ in ()).throw(RuntimeError("seed transport failure")),
        )
    assert ua.last_physical_attempt_capture().state == "unresolved"

    output_limit_exc = RuntimeError(
        "max_tokens 65536 exceeds maximum context length 32768"
    )
    calls = {"n": 0}

    def _fake_execute(request, send, before):
        calls["n"] += 1
        raise output_limit_exc

    monkeypatch.setattr(llm_local_mod, "_execute_candidate", _fake_execute)
    monkeypatch.setattr(llm_local_mod, "_attempt_request", lambda *a, **k: None)
    client = llm_mod.LLMClient.__new__(llm_mod.LLMClient)
    monkeypatch.setattr(client, "_get_local_client", lambda: object(), raising=False)
    monkeypatch.setattr(
        client, "_normalize_system_message_placement", lambda m: list(m), raising=False)
    monkeypatch.setattr(
        client, "_strip_openrouter_roundtrip_metadata", lambda m: list(m), raising=False)
    monkeypatch.setattr(
        client, "_copy_messages_with_cache_policy",
        lambda m, **k: [dict(x) for x in m], raising=False)

    with pytest.raises(RuntimeError) as excinfo:
        client._chat_local([{"role": "user", "content": "hi"}], None, 512, "auto")
    assert excinfo.value is output_limit_exc
    assert calls["n"] == 1


def test_local_transport_makes_exactly_one_physical_attempt(monkeypatch):
    """spec 4.3.2: the local lane dispatches ONE candidate per call. Retrying
    inside the lane spent the caller's physical-attempt budget without the
    caller authorising it; the single retry policy that owns that decision is
    `loop_llm_call.call_llm_with_retry`, which counts the attempts it makes."""
    from ouroboros import llm as llm_mod
    from ouroboros import llm_attempt as llm_attempt_mod
    from ouroboros import llm_local as llm_local_mod  # noqa: F401 - the lane under test

    transient = RuntimeError("connection reset by peer")
    calls = {"n": 0}
    slept: list[float] = []

    # Count at the TRUE physical boundary (wave-3 conformance review): the
    # real _execute_candidate runs, so an internal retry anywhere above
    # execute_physical_attempt would be seen here as calls["n"] > 1.
    def _fake_physical(request, send, before_dispatch=None):
        calls["n"] += 1
        raise transient

    monkeypatch.setattr(llm_attempt_mod, "execute_physical_attempt", _fake_physical)
    # The request builder needs an initialised client; the physical boundary
    # above is what this pin counts, so a None request is fine.
    monkeypatch.setattr(llm_local_mod, "_attempt_request", lambda *a, **k: None)
    monkeypatch.setattr(time, "sleep", slept.append)
    client = llm_mod.LLMClient.__new__(llm_mod.LLMClient)
    monkeypatch.setattr(client, "_get_local_client", lambda: object(), raising=False)
    monkeypatch.setattr(
        client, "_normalize_system_message_placement", lambda m: list(m), raising=False)
    monkeypatch.setattr(
        client, "_strip_openrouter_roundtrip_metadata", lambda m: list(m), raising=False)
    monkeypatch.setattr(
        client, "_copy_messages_with_cache_policy",
        lambda m, **k: [dict(x) for x in m], raising=False)

    with pytest.raises(RuntimeError) as excinfo:
        client._chat_local([{"role": "user", "content": "hi"}], None, 512, "auto")

    assert excinfo.value is transient
    assert calls["n"] == 1
    assert slept == []
