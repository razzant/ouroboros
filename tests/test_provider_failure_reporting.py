from unittest.mock import patch

from ouroboros.loop_round_limits import _provider_failure_hint
from ouroboros.loop_llm_call import call_llm_with_retry, classify_llm_exception
from ouroboros.usage_accounting import PhysicalAttemptContext


class _FailingLLM:
    def chat(self, **kwargs):
        raise RuntimeError("AuthenticationError('401 invalid_api_key')")


class _QuotaFailingLLM:
    calls = 0

    def chat(self, **kwargs):
        self.calls += 1
        raise RuntimeError("Provider returned 402 insufficient credits")


class _SuccessfulLLM:
    def chat(self, **kwargs):
        return {"content": "ok"}, {"provider": "anthropic", "resolved_model": "anthropic/claude-sonnet-4-6"}


class _ProviderError(Exception):
    def __init__(self, message, *, status_code=None, code=None):
        super().__init__(message)
        self.status_code = status_code
        if code is not None:
            self.code = code


def test_call_llm_with_retry_records_last_error(tmp_path):
    usage = {}

    msg, cost = call_llm_with_retry(
        _FailingLLM(),
        [{"role": "user", "content": "hi"}],
        "openai::gpt-5.5",
        None,
        "medium",
        1,
        tmp_path,
        "task-1",
        1,
        None,
        usage,
        "task",
        False,
    )

    assert msg is None
    assert cost == 0.0
    assert "invalid_api_key" in usage["_last_llm_error"]
    assert usage["_last_llm_error_kind"] == "auth_error"
    assert usage["_last_llm_retry_same_request"] is False


class _OverflowStoppedLLM:
    """Non-empty output whose finish_reason reports a context-window overflow —
    the successful truncated-generation shape (acceptance row ERR-3)."""

    def __init__(self):
        self.calls = 0

    def chat(self, **kwargs):
        self.calls += 1
        return (
            {"content": "partial but useful answer",
             "finish_reason": "model_context_window_exceeded"},
            {"provider": "openai", "resolved_model": "openai::gpt-5.5"},
        )


def test_non_empty_overflow_stopped_output_is_kept_without_retry(tmp_path):
    """ERR-3: a non-empty overflow-stopped response is retained as the delivery
    candidate — exactly one provider call, no semantic or transient-empty retry."""
    usage = {}
    llm = _OverflowStoppedLLM()
    msg, _cost = call_llm_with_retry(
        llm,
        [{"role": "user", "content": "hi"}],
        "openai::gpt-5.5", None, "medium", 3, tmp_path, "task-1", 1, None, usage, "task",
        False,
    )
    assert llm.calls == 1
    assert msg is not None
    assert msg["content"] == "partial but useful answer"
    assert "_last_llm_error" not in usage
    assert "_last_llm_error_kind" not in usage


class _RateLimitBodyLLM:
    """HTTP-200 response whose BODY carries a 429 (provider_error kind=rate_limit) with a
    present finish_reason — the canonical cloud.ru/OpenRouter rate-limit shape."""

    def chat(self, **kwargs):
        return (
            {"content": "", "finish_reason": "stop"},
            {"provider": "openai", "resolved_model": "openai::gpt-5.5",
             "provider_error": {"kind": "rate_limit", "code": 429}},
        )


def test_body_error_429_marks_rate_limit_kind_for_cooldown(tmp_path):
    from ouroboros.loop_llm_call import _COOLDOWN_ERROR_KINDS
    usage = {}
    msg, _cost = call_llm_with_retry(
        _RateLimitBodyLLM(),
        [{"role": "user", "content": "hi"}],
        "openai::gpt-5.5", None, "medium", 1, tmp_path, "task-1", 1, None, usage, "task",
        False,
        attempt_cap=1,
    )
    assert msg is None
    # The body-error 429 kind must be exposed for the F1 cooldown gate even though the
    # finish_reason is present (the generic event_type would be the non-cooling
    # "llm_empty_response"); preferring the body kind keeps a rate-limited model coolable.
    assert usage["_last_llm_error_kind"] == "rate_limit"
    assert usage["_last_llm_error_kind"] in _COOLDOWN_ERROR_KINDS


def test_call_llm_with_retry_clears_stale_last_error_on_success(tmp_path):
    usage = {
        "_last_llm_error": "old error",
        "_last_llm_error_kind": "auth_error",
    }

    msg, _cost = call_llm_with_retry(
        _SuccessfulLLM(),
        [{"role": "user", "content": "hi"}],
        "anthropic::claude-sonnet-4-6",
        None,
        "medium",
        1,
        tmp_path,
        "task-2",
        1,
        None,
        usage,
        "task",
        False,
    )

    assert msg == {"content": "ok"}
    assert "_last_llm_error" not in usage
    assert "_last_llm_error_kind" not in usage


def test_non_main_call_does_not_project_stale_main_fit_fields(tmp_path):
    import json

    usage = {
        "_context_profile": "owner_low",
        "_context_fit_mode": "low",
        "_context_target_miss": True,
    }
    msg, _cost = call_llm_with_retry(
        _SuccessfulLLM(), [{"role": "user", "content": "forced"}],
        "anthropic::claude-sonnet-4-6", None, "medium", 1, tmp_path,
        "task-forced", 1, None, usage, "task", False,
        physical_context=None,
    )
    assert msg == {"content": "ok"}
    event = next(
        json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
        if '"type": "llm_round"' in line
    )
    assert "context_profile" not in event
    assert "context_target_miss" not in event


class _EmptyBodyErrorLLM:
    def __init__(self, provider_error):
        self.provider_error = provider_error
        self.calls = 0

    def chat(self, **kwargs):
        self.calls += 1
        return (
            {"content": "", "tool_calls": [], "finish_reason": "stop"},
            {"provider_error": self.provider_error},
        )


def test_empty_body_output_or_body_size_is_not_context_recovery(tmp_path):
    for provider_error in (
        {"kind": "provider_error", "code": 400,
         "message": "max_tokens 65536 exceeds maximum context length 32768"},
        {"kind": "provider_error", "code": 413, "message": "request body too large"},
    ):
        usage = {}
        llm = _EmptyBodyErrorLLM(provider_error)
        msg, _cost = call_llm_with_retry(
            llm, [{"role": "user", "content": "hi"}], "openai/gpt-5.5",
            None, "medium", 1, tmp_path, "task-body-size", 1, None, usage,
            "task", False, attempt_cap=1,
        )
        assert msg is None
        assert llm.calls == 1
        assert usage["_last_llm_error_kind"] == "request_too_large"


def test_empty_structured_context_overflow_event_carries_fit_projection(tmp_path):
    usage = {
        "_context_profile": "owner_low",
        "_context_fit_mode": "low",
        "_context_target_miss": True,
        "_context_automatic_pass_used": True,
    }
    llm = _EmptyBodyErrorLLM({
        "kind": "provider_error",
        "code": "context_length_exceeded",
        "type": "invalid_request_error",
        "message": "input rejected",
    })
    msg, _cost = call_llm_with_retry(
        llm, [{"role": "user", "content": "hi"}], "openai/gpt-5.5",
        None, "medium", 1, tmp_path, "task-empty-overflow", 1, None, usage,
        "task", False, attempt_cap=1,
        physical_context=PhysicalAttemptContext(
            profile="owner_low",
            rendered_mode="low",
            measurement_basis="cold_estimate",
            route_fp="route",
            round_id="round-1",
            target_total_tokens=200_000,
            capacity_total_tokens=500_000,
            context_target_miss=True,
            automatic_pass_used=True,
        ),
    )
    assert msg is None and llm.calls == 1
    assert usage["_last_llm_error_kind"] == "context_overflow"
    event = next(
        __import__("json").loads(line)
        for line in (tmp_path / "events.jsonl").read_text().splitlines()
        if '"type": "remote_context_overflow"' in line
    )
    assert event["context_profile"] == "owner_low"
    assert event["context_fit_mode"] == "low"
    assert event["context_target_miss"] is True
    assert event["context_automatic_pass_used"] is True


def test_call_llm_with_retry_stops_non_retryable_same_request(tmp_path):
    usage = {}
    llm = _QuotaFailingLLM()

    msg, cost = call_llm_with_retry(
        llm,
        [{"role": "user", "content": "hi"}],
        "google/gemini-3.5-flash",
        None,
        "medium",
        3,
        tmp_path,
        "task-quota",
        1,
        None,
        usage,
        "task",
        False,
    )

    assert msg is None
    assert cost == 0.0
    assert llm.calls == 1
    assert usage["_last_llm_error_kind"] == "quota_exhausted"
    assert usage["_last_llm_retry_same_request"] is False


class _GlitchThenOkLLM:
    """finish_reason=null provider glitch for N calls, then a real response."""

    def __init__(self, glitches: int):
        self.glitches = glitches
        self.calls = 0

    def chat(self, **kwargs):
        self.calls += 1
        if self.calls <= self.glitches:
            return {"content": "", "tool_calls": [], "finish_reason": None}, {}
        return {"content": "recovered"}, {"provider": "openrouter", "resolved_model": "openai/gpt-5.5"}


class _TransientFailingLLM:
    def __init__(self):
        self.calls = 0

    def chat(self, **kwargs):
        self.calls += 1
        raise _ProviderError("503 service unavailable, please retry", status_code=503)


def test_transient_finish_reason_null_recovers_on_same_model(tmp_path, monkeypatch):
    """finish_reason=null glitches retry the SAME model beyond the permanent
    3-attempt budget (terminal-bench death class) and recover without any
    cross-model fallback."""
    import time as _time
    monkeypatch.setattr(_time, "sleep", lambda _s: None)
    usage = {}
    llm = _GlitchThenOkLLM(glitches=4)  # would die under the old max_retries=3

    msg, _cost = call_llm_with_retry(
        llm, [{"role": "user", "content": "hi"}], "openai/gpt-5.5", None,
        "medium", 3, tmp_path, "task-transient", 1, None, usage, "task", False,
    )

    assert msg == {"content": "recovered"}
    assert llm.calls == 5  # 4 glitches + 1 success, all same model
    assert "_last_llm_error" not in usage


def test_transient_retry_respects_remaining_deadline(tmp_path, monkeypatch):
    """Transient retries must not sleep past the task deadline."""
    import time as _time
    sleeps = []
    monkeypatch.setattr(_time, "sleep", lambda s: sleeps.append(s))
    usage = {}
    llm = _TransientFailingLLM()

    msg, _cost = call_llm_with_retry(
        llm, [{"role": "user", "content": "hi"}], "openai/gpt-5.5", None,
        "medium", 3, tmp_path, "task-deadline", 1, None, usage, "task", False,
        deadline_ts=_time.time() + 1.0,  # no room for any backoff sleep
    )

    assert msg is None
    assert llm.calls == 1  # stopped before burning the deadline on sleeps
    assert sleeps == []
    assert usage["_last_llm_error_kind"] == "provider_transient"


def test_finish_reason_null_deadline_stop_emits_durable_event(tmp_path, monkeypatch):
    """The finish_reason=null path must record llm_retry_deadline_exhausted
    when the deadline refuses the next backoff — same observability as the
    exception path."""
    import json as _json
    import time as _time
    monkeypatch.setattr(_time, "sleep", lambda _s: None)
    usage = {}
    llm = _GlitchThenOkLLM(glitches=10)

    msg, _cost = call_llm_with_retry(
        llm, [{"role": "user", "content": "hi"}], "openai/gpt-5.5", None,
        "medium", 3, tmp_path, "task-null-deadline", 1, None, usage, "task", False,
        deadline_ts=_time.time() + 1.0,
    )

    assert msg is None
    assert llm.calls == 1
    events = [
        _json.loads(line)
        for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    exhausted = [e for e in events if e.get("type") == "llm_retry_deadline_exhausted"]
    assert len(exhausted) == 1
    assert exhausted[0]["error_kind"] == "provider_incomplete_response"


def test_transient_retry_budget_env_override(tmp_path, monkeypatch):
    """OUROBOROS_TRANSIENT_RETRY_MAX tunes the transient budget but never
    drops below the caller's default budget."""
    from ouroboros.loop_llm_call import transient_retry_max

    monkeypatch.delenv("OUROBOROS_TRANSIENT_RETRY_MAX", raising=False)
    assert transient_retry_max(3) == 6  # default
    monkeypatch.setenv("OUROBOROS_TRANSIENT_RETRY_MAX", "8")
    assert transient_retry_max(3) == 8
    monkeypatch.setenv("OUROBOROS_TRANSIENT_RETRY_MAX", "1")
    assert transient_retry_max(3) == 3  # floored at caller default
    monkeypatch.setenv("OUROBOROS_TRANSIENT_RETRY_MAX", "junk")
    assert transient_retry_max(3) == 6


def test_transient_retry_max_propagates_from_settings():
    """A settings.json value must reach os.environ via apply_settings_to_env
    (the only hot-reload path) — otherwise the knob is silently inert.

    apply_settings_to_env pops every registered env key missing from the dict,
    so the WHOLE environ is snapshot-restored to avoid wiping provider
    credentials/model settings for later tests in this process.
    """
    import os

    from ouroboros.config import apply_settings_to_env
    from ouroboros.loop_llm_call import transient_retry_max

    snapshot = dict(os.environ)
    try:
        apply_settings_to_env({"OUROBOROS_TRANSIENT_RETRY_MAX": 9})
        assert transient_retry_max(3) == 9
    finally:
        os.environ.clear()
        os.environ.update(snapshot)


def test_permanent_classes_still_fail_fast(tmp_path, monkeypatch):
    """Permanent classes (auth) must not consume the transient budget."""
    import time as _time
    monkeypatch.setattr(_time, "sleep", lambda _s: None)
    usage = {}

    class _CountingAuthLLM:
        calls = 0

        def chat(self, **kwargs):
            _CountingAuthLLM.calls += 1
            raise RuntimeError("AuthenticationError('401 invalid_api_key')")

    msg, _cost = call_llm_with_retry(
        _CountingAuthLLM(), [{"role": "user", "content": "hi"}], "openai/gpt-5.5",
        None, "medium", 3, tmp_path, "task-auth", 1, None, usage, "task", False,
    )

    assert msg is None
    assert _CountingAuthLLM.calls == 1
    assert usage["_last_llm_error_kind"] == "auth_error"


def test_classify_llm_exception_distinguishes_retryable_rate_limit():
    rate = classify_llm_exception(RuntimeError("429 rate limit exceeded"))
    quota = classify_llm_exception(RuntimeError("402 insufficient credits"))

    assert rate.kind == "provider_transient"
    assert rate.retry_same_request is True
    assert quota.kind == "quota_exhausted"
    assert quota.retry_same_request is False


def test_classify_llm_exception_uses_provider_code_before_429_status():
    quota = classify_llm_exception(
        _ProviderError("rate limit transport status", status_code=429, code="insufficient_quota")
    )

    assert quota.kind == "quota_exhausted"
    assert quota.retry_same_request is False
    assert quota.status_code == 429
    assert quota.provider_code == "insufficient_quota"


def test_classify_llm_exception_keeps_429_token_rate_retryable():
    rate = classify_llm_exception(
        _ProviderError("429 too many tokens per minute", status_code=429)
    )

    assert rate.kind == "provider_transient"
    assert rate.retry_same_request is True


def test_classify_llm_exception_keeps_text_only_token_rate_retryable():
    rate = classify_llm_exception(RuntimeError("Rate limit reached: too many tokens per minute"))
    plain_429 = classify_llm_exception(RuntimeError("429 too many tokens per minute"))

    assert rate.kind == "provider_transient"
    assert rate.retry_same_request is True
    assert plain_429.kind == "provider_transient"
    assert plain_429.retry_same_request is True


def test_provider_failure_hint_formats_detail():
    hint = _provider_failure_hint({"_last_llm_error": "  AuthenticationError('401 invalid_api_key')  "})

    assert hint == " Last provider error: AuthenticationError('401 invalid_api_key')"


def test_provider_failure_hint_empty_without_error():
    assert _provider_failure_hint({}) == ""


def test_call_llm_with_retry_accumulates_live_catalog_estimated_cost(tmp_path):
    import queue

    class _EstimatedCostLLM:
        def chat(self, **kwargs):
            return (
                {"content": "ok"},
                {
                    "provider": "openrouter",
                    "resolved_model": "openai/gpt-new",
                    "prompt_tokens": 1000,
                    "completion_tokens": 100,
                    "cached_tokens": 0,
                    "cache_write_tokens": 0,
                },
            )

    usage = {}
    event_queue = queue.Queue()
    with patch("ouroboros.loop_llm_call.estimate_cost_optional", return_value=0.123456):
        _msg, _cost = call_llm_with_retry(
            _EstimatedCostLLM(),
            [{"role": "user", "content": "hi"}],
            "openai/gpt-new",
            None,
            "medium",
            1,
            tmp_path,
            "task-3",
            1,
            event_queue,
            usage,
            "task",
            False,
        )

    assert usage["cost"] == 0.123456
    events = [event_queue.get_nowait() for _ in range(event_queue.qsize())]
    usage_event = next(evt for evt in events if evt.get("type") == "llm_usage")
    assert usage_event["cost_estimated"] is True
