"""Rate-limit lane of the safety supervisor (PR: stop reporting safety-model rate
limits as SAFETY_VIOLATION) plus the bounded conversation-transcript budget.

Split from tests/test_safety_policy.py at its 1600-line module gate; shares that
module's stub-client idiom (the tiny ``_patch_llm_client`` shim is repeated here
rather than imported across test modules).
"""

from __future__ import annotations

import json
import pathlib
import time

import pytest


@pytest.fixture(autouse=True)
def _ensure_remote_key(monkeypatch):
    """Same ambient routing as test_safety_policy.py: a fake remote key keeps the LLM
    path active so ``_resolve_safety_routing`` doesn't take the misconfigured-fail-open
    branch; lane-specific tests override with their own env calls."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-key-for-routing")
    monkeypatch.delenv("USE_LOCAL_LIGHT", raising=False)
    yield


def _patch_llm_client(monkeypatch, stub) -> None:
    import ouroboros.safety as safety

    monkeypatch.setattr(safety, "LLMClient", lambda: stub)

# ---------------------------------------------------------------------------
# Rate-limit fail-open + bounded transcript (OB-02)
# ---------------------------------------------------------------------------


class _ScriptedLLMClient:
    """``_StubLLMClient`` with a per-call script: each entry is an Exception to raise or a
    ``(content, usage)`` tuple to return. The last entry repeats, so a one-element script
    models a provider that keeps failing the same way."""

    def __init__(self, script):
        self.script = list(script)
        self.calls: list[dict] = []

    def chat(self, *, messages, model, use_local, **kwargs):
        self.calls.append({"messages": messages, "model": model, "use_local": use_local, **kwargs})
        step = self.script[min(len(self.calls) - 1, len(self.script) - 1)]
        if isinstance(step, Exception):
            raise step
        content, usage = step
        return {"content": content}, usage


class _RateLimitError(Exception):
    """Exception-shaped 429 (the shape `classify_llm_exception` reads a status from)."""

    status_code = 429


class _QuotaError(Exception):
    """Structured insufficient-quota that ALSO carries HTTP 429 — the case whose
    PERMANENT classification must win over the status code and keep blocking."""

    status_code = 429
    code = "insufficient_quota"


# The production shape: HTTP 200 whose BODY carried the rate limit, surfaced by
# ``llm._normalize_remote_response`` as a typed marker on ``usage``. Nothing raises.
_BODY_RATE_LIMIT_USAGE = {
    "provider_error": {
        "code": "429",
        "type": "rate_limit_error",
        "message": "Rate limit exceeded",
        "kind": "rate_limit",
    },
    "prompt_tokens": 12,
    "completion_tokens": 0,
    "cost": 0.0,
}


class _DriveCtx:
    """ToolContext-shaped stub whose durable safety events land in a REAL ``events.jsonl``
    under tmp_path — the audit surface the fix must write, read back through the same
    file an owner would open."""

    def __init__(self, root):
        self.task_id = "t-safety"
        self.drive_root = str(root)
        self._logs = pathlib.Path(root) / "logs"
        self._logs.mkdir(parents=True, exist_ok=True)

    def drive_logs(self):
        return self._logs


def _read_events(ctx, event_type: str | None = None) -> list[dict]:
    path = ctx.drive_logs() / "events.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [r for r in rows if event_type is None or r.get("type") == event_type]


@pytest.fixture(autouse=True)
def _cold_storm_latch(monkeypatch):
    """Every test in this module starts with the process-local storm latch disarmed: a
    prior test's terminal rate-limit outcome must not short-circuit an unrelated one
    (monkeypatch restores the original value afterwards)."""
    import ouroboros.safety as safety

    monkeypatch.setattr(safety, "_SAFETY_STORM_UNTIL", 0.0)


@pytest.fixture
def _no_backoff(monkeypatch):
    """Skip the real 2s sleep. The backoff's PLACEMENT is asserted separately by
    ``test_rate_limit_retry_takes_one_slot_per_attempt``."""
    import ouroboros.safety as safety

    monkeypatch.setattr(safety, "_safety_rate_limit_backoff", lambda ctx: None)


def test_exception_shaped_rate_limit_blocks_unchecked_after_one_retry(monkeypatch, tmp_path, _no_backoff):
    """Two 429s: BLOCK this one call with the typed non-verdict SAFETY_UNAVAILABLE
    outcome (never a SAFETY_VIOLATION accusation, never an unchecked execution), and
    leave a durable audit row. `full` keeps its owner contract: an unchecked guarded
    call does not run."""
    from ouroboros.safety import check_safety

    stub = _ScriptedLLMClient([_RateLimitError("Rate limit exceeded")])
    _patch_llm_client(monkeypatch, stub)
    ctx = _DriveCtx(tmp_path)

    ok, msg = check_safety("create_github_issue", {"title": "x"}, ctx=ctx)

    assert ok is False, "an unchecked guarded call must not execute in full mode"
    assert msg.startswith("⚠️ SAFETY_UNAVAILABLE:"), "typed non-verdict outcome, not a verdict"
    assert "NOT a verdict" in msg and "Retry the same call" in msg
    assert "SAFETY_VIOLATION" not in msg and "_BLOCKED" not in msg.splitlines()[0]
    assert len(stub.calls) == 2, "exactly one retry, then the typed blocked outcome"

    rows = _read_events(ctx, "safety_check_rate_limited")
    assert len(rows) == 1
    assert rows[0]["tool"] == "create_github_issue"
    assert rows[0]["action"] == "blocked_unchecked_after_retry"
    assert rows[0]["error"], "the audit row carries the sanitized bounded error"
    assert rows[0]["task_id"] == "t-safety"


def test_http200_body_rate_limit_blocks_unchecked_after_one_retry(monkeypatch, tmp_path, _no_backoff):
    """THE production shape: HTTP 200, empty content, ``usage['provider_error']``. Nothing
    raises, so an exception-only check would still walk the unparseable-response repair
    path into SAFETY_VIOLATION — this fails with the bug alive even when (a) passes."""
    from ouroboros.safety import check_safety

    monkeypatch.setattr("supervisor.state.update_budget_from_usage", lambda usage: None)
    stub = _ScriptedLLMClient([("", dict(_BODY_RATE_LIMIT_USAGE))])
    _patch_llm_client(monkeypatch, stub)
    ctx = _DriveCtx(tmp_path)

    ok, msg = check_safety("create_github_issue", {"title": "x"}, ctx=ctx)

    assert ok is False
    assert msg.startswith("⚠️ SAFETY_UNAVAILABLE:") and "rate-limited" in msg
    assert len(stub.calls) == 2
    assert len(_read_events(ctx, "safety_check_rate_limited")) == 1
    # A rate limit is not an unparseable verdict: the parse-repair lane must not run.
    assert _read_events(ctx, "safety_parse_retry") == []
    assert _read_events(ctx, "safety_parse_failed") == []


def test_safety_unavailable_classifies_as_plain_error_never_safety_violation():
    """The downstream status contract the typed outcome rides on: a first line carrying
    `_UNAVAILABLE` (and no `_VIOLATION`/`_BLOCKED`) classifies as a plain tool `error`,
    so telemetry and the agent never see a rate-limited check as `safety_violation`."""
    from ouroboros.safety import _safety_unavailable_blocked

    ok, msg = _safety_unavailable_blocked(None, "run_command", "provider_transient: 429")
    first = msg.splitlines()[0]
    assert ok is False
    assert "_UNAVAILABLE" in first
    assert "_VIOLATION" not in first and "_BLOCKED" not in first


def test_storm_latch_short_circuits_without_new_provider_calls(monkeypatch, tmp_path, _no_backoff):
    """After a terminal rate-limited outcome, the process-local storm latch answers the
    NEXT check with the same typed blocked outcome and ZERO provider calls inside the
    window — the highest-frequency LIGHT consumer must not re-probe a storming route on
    every guarded call."""
    from ouroboros.safety import check_safety

    stub = _ScriptedLLMClient([_RateLimitError("Rate limit exceeded")])
    _patch_llm_client(monkeypatch, stub)
    ctx = _DriveCtx(tmp_path)

    ok1, msg1 = check_safety("create_github_issue", {"title": "x"}, ctx=ctx)
    calls_after_first = len(stub.calls)
    ok2, msg2 = check_safety("create_github_issue", {"title": "y"}, ctx=ctx)

    assert ok1 is False and ok2 is False
    assert msg2.startswith("⚠️ SAFETY_UNAVAILABLE:")
    assert "cooling down" in msg2
    assert len(stub.calls) == calls_after_first, "the latched window makes no provider calls"


def test_expired_deadline_permits_no_second_rate_limit_attempt(monkeypatch, tmp_path, _no_backoff):
    """A spent task deadline bounds the RETRY, not only the sleep: one physical attempt,
    then the typed blocked outcome — never a second paid call past the deadline."""
    from ouroboros.safety import check_safety

    stub = _ScriptedLLMClient([_RateLimitError("Rate limit exceeded")])
    _patch_llm_client(monkeypatch, stub)
    ctx = _DriveCtx(tmp_path)
    ctx.task_metadata = {"deadline_at": "2020-01-01T00:00:00Z"}

    ok, msg = check_safety("create_github_issue", {"title": "x"}, ctx=ctx)

    assert ok is False
    assert msg.startswith("⚠️ SAFETY_UNAVAILABLE:")
    assert len(stub.calls) == 1, "no second physical attempt after the deadline"
    rows = _read_events(ctx, "safety_check_rate_limited")
    assert rows[-1]["action"] == "blocked_unchecked_deadline_expired", \
        "the audit row must not claim a retry that never happened"
    import ouroboros.safety as safety
    assert safety._SAFETY_STORM_UNTIL == 0.0, \
        "one 429 cut by the deadline is not a confirmed storm; the latch stays cold"


def test_body_quota_refusal_reads_structured_fields_never_free_text(monkeypatch):
    """The quota-over-429 precedence scans only structured code/type and skips numeric
    markers: `402`/`billing` occur incidentally inside request ids and hostnames in
    free-form messages, and a false quota match would re-label a plain throttle as a
    verdict — the exact bug this lane removes."""
    from ouroboros.safety import _body_is_quota_refusal

    assert _body_is_quota_refusal({"code": "insufficient_quota", "message": "x"}) is True
    assert _body_is_quota_refusal({"type": "billing_hard_limit", "message": "x"}) is True
    assert _body_is_quota_refusal(
        {"code": "429", "type": "rate_limit_error",
         "message": "Too many requests; retry after 402 seconds (id req_402ab19)"}
    ) is False
    assert _body_is_quota_refusal(
        {"code": "429", "type": "rate_limit_error",
         "message": "upstream billing-tier throughput cap reached"}
    ) is False


def test_single_rate_limit_then_success_returns_the_normal_verdict(monkeypatch, tmp_path, _no_backoff):
    """One 429 then a real verdict: ordinary result, no fail-open, no audit row."""
    from ouroboros.safety import check_safety

    stub = _ScriptedLLMClient([
        _RateLimitError("Rate limit exceeded"),
        ('{"status":"SAFE","reason":"ok"}', None),
    ])
    _patch_llm_client(monkeypatch, stub)
    ctx = _DriveCtx(tmp_path)

    ok, msg = check_safety("create_github_issue", {"title": "x"}, ctx=ctx)

    assert ok is True
    assert msg == "", "a recovered check returns the ordinary SAFE verdict"
    assert len(stub.calls) == 2
    assert _read_events(ctx) == [], "no audit row when the retry actually succeeded"


def test_structured_insufficient_quota_with_429_still_blocks(monkeypatch, tmp_path, _no_backoff):
    """PERMANENT precedence: an insufficient-quota carried on a 429 keeps TODAY'S
    blocking path — one attempt, the exact existing message, no fail-open."""
    from ouroboros.safety import check_safety

    stub = _ScriptedLLMClient([_QuotaError("Rate limit exceeded (insufficient_quota)")])
    _patch_llm_client(monkeypatch, stub)
    ctx = _DriveCtx(tmp_path)

    ok, msg = check_safety("create_github_issue", {"title": "x"}, ctx=ctx)

    assert ok is False
    assert msg == (
        "⚠️ SAFETY_VIOLATION: Safety check failed with error: "
        "_QuotaError: Rate limit exceeded (insufficient_quota)"
    )
    assert len(stub.calls) == 1, "a permanent class must not buy a retry"
    assert _read_events(ctx, "safety_check_rate_limited") == []


def test_non_rate_limit_exception_keeps_todays_violation_path(monkeypatch, tmp_path, _no_backoff):
    """Every other exception class is byte-identical to today: block, one attempt."""
    from ouroboros.safety import check_safety

    stub = _ScriptedLLMClient([RuntimeError("network down")])
    _patch_llm_client(monkeypatch, stub)
    ctx = _DriveCtx(tmp_path)

    ok, msg = check_safety("create_github_issue", {"title": "x"}, ctx=ctx)

    assert ok is False
    assert msg == (
        "⚠️ SAFETY_VIOLATION: Safety check failed with error: RuntimeError: network down"
    )
    assert len(stub.calls) == 1
    assert _read_events(ctx) == []


def test_conversation_section_is_bounded_newest_first_with_counted_marker():
    """The CONVERSATION section is bounded, keeps the NEWEST rounds, and discloses
    the exact number it dropped. The tool-arguments section is untouched."""
    import re

    from ouroboros.safety import _SAFETY_CONTEXT_CHAR_BUDGET, _build_check_prompt

    messages = [{"role": "user", "content": f"round{i} " + "x" * 400} for i in range(40)]
    messages[-1] = {"role": "user", "content": "NEWEST_ROUND_MARKER"}

    prompt = _build_check_prompt("run_command", {"cmd": ["echo", "hello"]}, messages)

    head, context = prompt.split("Conversation context:\n", 1)
    context = context.rsplit("\nIs this safe?", 1)[0]

    assert len(context) <= _SAFETY_CONTEXT_CHAR_BUDGET
    assert "NEWEST_ROUND_MARKER" in context, "the newest round must survive"
    assert "round0 " not in context, "the oldest rounds must be dropped"

    lines = context.splitlines()
    matched = re.match(r"^\[… (\d+) older messages omitted\]$", lines[0])
    assert matched, f"missing counted omission marker: {lines[0]!r}"
    assert int(matched.group(1)) == len(messages) - (len(lines) - 1)

    # The proposed call is the SUBJECT of the check and stays outside the budget.
    assert '"cmd"' in head and "hello" in head


def test_short_conversation_keeps_every_message_and_no_marker():
    """Under budget: no marker, nothing dropped (the marker is not decoration)."""
    from ouroboros.safety import _format_messages_for_safety

    rendered = _format_messages_for_safety([
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "second"},
    ])

    assert rendered == "[user] first\n[assistant] second"
    assert "omitted" not in rendered


def test_rate_limit_retry_takes_one_slot_per_attempt(monkeypatch, tmp_path):
    """model_concurrency caps CONCURRENT calls, so the retry must take a FRESH slot
    and the backoff must sleep BETWEEN slot contexts — never inside a held one."""
    import contextlib

    import ouroboros.safety as safety
    from ouroboros import model_concurrency
    from ouroboros.safety import check_safety

    state = {"acquired": 0, "depth": 0, "slept_at_depth": []}

    @contextlib.contextmanager
    def _counting_slot(model, use_local=False, deadline_ts=None):
        state["acquired"] += 1
        state["depth"] += 1
        try:
            yield
        finally:
            state["depth"] -= 1

    monkeypatch.setattr(model_concurrency, "model_call_slot", _counting_slot)
    monkeypatch.setattr(
        safety, "_safety_rate_limit_backoff",
        lambda ctx: state["slept_at_depth"].append(state["depth"]),
    )

    stub = _ScriptedLLMClient([_RateLimitError("Rate limit exceeded")])
    _patch_llm_client(monkeypatch, stub)
    ctx = _DriveCtx(tmp_path)

    ok, _ = check_safety("create_github_issue", {"title": "x"}, ctx=ctx)

    assert ok is False, "double rate limit ends in the typed blocked outcome"
    assert state["acquired"] == 2, "one slot per attempt, not one slot around both"
    assert state["slept_at_depth"] == [0], "the backoff must not run inside a held slot"
    assert state["depth"] == 0, "every acquired slot is released"


def test_rate_limit_backoff_is_capped_by_the_task_deadline(monkeypatch):
    """The one sleep is bounded by the REAL task deadline; an expired deadline
    skips it entirely rather than sleeping past the task."""
    import ouroboros.safety as safety

    slept: list[float] = []
    monkeypatch.setattr(safety.time, "sleep", lambda s: slept.append(s))
    monkeypatch.setattr(safety, "_safety_deadline_epoch", lambda ctx: safety.time.time() + 0.25)

    safety._safety_rate_limit_backoff(None)
    assert slept and slept[0] <= 0.25 < safety._SAFETY_RATE_LIMIT_BACKOFF_SEC

    slept.clear()
    monkeypatch.setattr(safety, "_safety_deadline_epoch", lambda ctx: safety.time.time() - 5)
    safety._safety_rate_limit_backoff(None)
    assert slept == [], "an expired deadline must not sleep at all"


class _ServerError(Exception):
    """A 503 outage: `classify_llm_exception` calls it `provider_transient`, but it is
    NOT throttling, so the safety lane must keep blocking on it."""

    status_code = 503


def test_server_outage_is_not_a_rate_limit_and_still_blocks(monkeypatch, tmp_path, _no_backoff):
    """5xx is an outage, not throughput: today's blocking path, one attempt, no audit."""
    from ouroboros.safety import check_safety

    stub = _ScriptedLLMClient([_ServerError("Service Unavailable")])
    _patch_llm_client(monkeypatch, stub)
    ctx = _DriveCtx(tmp_path)

    ok, msg = check_safety("create_github_issue", {"title": "x"}, ctx=ctx)

    assert ok is False
    assert msg == (
        "⚠️ SAFETY_VIOLATION: Safety check failed with error: "
        "_ServerError: Service Unavailable"
    )
    assert len(stub.calls) == 1, "an outage must not buy the rate-limit retry"
    assert _read_events(ctx) == []


def test_read_timeout_is_not_a_rate_limit_and_still_blocks(monkeypatch, tmp_path, _no_backoff):
    """A transport timeout also classifies `provider_transient`; it must still block."""
    from ouroboros.safety import check_safety

    class _ReadTimeout(Exception):
        pass

    stub = _ScriptedLLMClient([_ReadTimeout("The read operation timed out")])
    _patch_llm_client(monkeypatch, stub)
    ctx = _DriveCtx(tmp_path)

    ok, msg = check_safety("create_github_issue", {"title": "x"}, ctx=ctx)

    assert ok is False
    assert "SAFETY_VIOLATION" in msg
    assert len(stub.calls) == 1
    assert _read_events(ctx) == []


def test_http200_body_transient_that_is_not_429_still_blocks(monkeypatch, tmp_path, _no_backoff):
    """Body lane: only `kind == "rate_limit"` (llm.py assigns it solely to a transient
    body error whose code IS 429) waves through; a body `provider_transient` keeps the
    existing unparseable-response outcome."""
    from ouroboros.safety import check_safety

    monkeypatch.setattr("supervisor.state.update_budget_from_usage", lambda usage: None)
    usage = {"provider_error": {"code": "502", "type": "server_error",
                                "message": "Bad gateway", "kind": "provider_transient"},
             "prompt_tokens": 5, "completion_tokens": 0, "cost": 0.0}
    stub = _ScriptedLLMClient([("", usage)])
    _patch_llm_client(monkeypatch, stub)
    ctx = _DriveCtx(tmp_path)

    ok, msg = check_safety("create_github_issue", {"title": "x"}, ctx=ctx)

    assert ok is False
    assert "SAFETY_VIOLATION" in msg
    assert _read_events(ctx, "safety_check_rate_limited") == []


def test_local_fallback_lane_rate_limit_takes_the_audited_fail_open(monkeypatch, tmp_path, _no_backoff):
    """Disclosed nuance: the local-FALLBACK lane keeps its documented fail-open contract
    (ARCHITECTURE "Safety and runtime mode" case (c)) — a genuine 429 there takes the two-attempt fail-open WITH the
    audit row (a 429 must not be stricter than the RuntimeError beside it), while every
    other error keeps its unchanged one-attempt 'Local safety runtime unreachable'
    warning. Both allow; the remote lanes are the ones that block typed."""
    from ouroboros.safety import check_safety

    for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "MINIMAX_API_KEY",
              "OPENAI_COMPATIBLE_API_KEY", "CLOUDRU_FOUNDATION_MODELS_API_KEY",
              "GIGACHAT_CREDENTIALS", "USE_LOCAL_LIGHT", "USE_LOCAL_HEAVY",
              "USE_LOCAL_FALLBACK"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-fake")
    monkeypatch.setenv("OUROBOROS_MODEL_LIGHT", "anthropic::claude-sonnet-4.6")
    monkeypatch.setenv("USE_LOCAL_MAIN", "true")

    stub = _ScriptedLLMClient([_RateLimitError("Rate limit exceeded")])
    _patch_llm_client(monkeypatch, stub)
    ctx = _DriveCtx(tmp_path)

    ok, msg = check_safety("totally_new_tool_mixed_config", {"arg": 1}, ctx=ctx)
    assert ok is True
    assert "rate-limited" in msg
    assert len(stub.calls) == 2
    assert len(_read_events(ctx, "safety_check_rate_limited")) == 1

    # Everything else on this lane is unchanged: one attempt, original wording.
    other = _ScriptedLLMClient([RuntimeError("local server down")])
    _patch_llm_client(monkeypatch, other)
    ctx2 = _DriveCtx(tmp_path / "second")
    ok2, msg2 = check_safety("totally_new_tool_mixed_config", {"arg": 1}, ctx=ctx2)
    assert ok2 is True
    assert "Local safety runtime unreachable" in msg2
    assert len(other.calls) == 1
    assert _read_events(ctx2) == []


def test_omission_marker_space_is_reserved_inside_the_budget():
    """Boundary case: the messages tile the budget so tightly that the MARKER'S OWN
    length decides the final cut. Without the reservation the last line still fits, the
    marker is appended after the check, and the section overflows the budget."""
    import re

    from ouroboros.safety import _SAFETY_CONTEXT_CHAR_BUDGET, _format_messages_for_safety

    # Each message renders to exactly 40 chars ("[user] " is 7).
    messages = [{"role": "user", "content": "y" * 33} for _ in range(130)]
    context = _format_messages_for_safety(messages)

    assert len(context) <= _SAFETY_CONTEXT_CHAR_BUDGET, (
        "the marker must be paid for INSIDE the budget, not appended after the check"
    )
    # Prove the case is genuinely tight: one more 40-char line would have fit if the
    # marker had cost nothing, so this test fails the moment the reserve is dropped.
    assert len(context) > _SAFETY_CONTEXT_CHAR_BUDGET - 80

    lines = context.splitlines()
    matched = re.match(r"^\[… (\d+) older messages omitted\]$", lines[0])
    assert matched, f"missing counted omission marker: {lines[0]!r}"
    assert int(matched.group(1)) == len(messages) - (len(lines) - 1)


def test_marker_bearing_5xx_message_is_not_a_rate_limit_and_still_blocks(monkeypatch, tmp_path, _no_backoff):
    """The markers are SUBSTRINGS, so `429`/`rpm`/`tpm` occur inside the request ids and
    hostnames real outages carry. A KNOWN non-429 status must never reach the text
    branch: this 503 carries `429` in its request id and must still block."""
    from ouroboros.safety import check_safety

    stub = _ScriptedLLMClient([
        _ServerError("503 Service Unavailable (request id: req_1f429ab0)"),
    ])
    _patch_llm_client(monkeypatch, stub)
    ctx = _DriveCtx(tmp_path)

    ok, msg = check_safety("create_github_issue", {"title": "x"}, ctx=ctx)

    assert ok is False, "a marker inside a request id must not wave a 5xx outage through"
    assert "SAFETY_VIOLATION" in msg
    assert "req_1f429ab0" in msg
    assert len(stub.calls) == 1
    assert _read_events(ctx) == []


def test_statusless_rate_limit_text_takes_the_typed_blocked_outcome(monkeypatch, tmp_path, _no_backoff):
    """The text branch keeps its own case: a bare throttling message, no status code."""
    from ouroboros.safety import check_safety

    class _BareRateLimit(Exception):
        pass

    stub = _ScriptedLLMClient([_BareRateLimit("rate limit exceeded")])
    _patch_llm_client(monkeypatch, stub)
    ctx = _DriveCtx(tmp_path)

    ok, msg = check_safety("create_github_issue", {"title": "x"}, ctx=ctx)

    assert ok is False
    assert msg.startswith("⚠️ SAFETY_UNAVAILABLE:") and "rate-limited" in msg
    assert len(stub.calls) == 2
    assert len(_read_events(ctx, "safety_check_rate_limited")) == 1


def test_body_shaped_quota_refusal_on_a_429_still_blocks(monkeypatch, tmp_path, _no_backoff):
    """Quota precedence holds in the HTTP-200 body shape too: permanent, one attempt,
    no fail-open, no audit row (mirrors the exception lane's quota-over-429 rule)."""
    from ouroboros.safety import check_safety

    usage = dict(_BODY_RATE_LIMIT_USAGE)
    usage["provider_error"] = dict(usage["provider_error"],
                                   type="insufficient_quota",
                                   message="insufficient_quota: billing hard limit reached")
    stub = _ScriptedLLMClient([("", usage), ("", dict(usage))])
    _patch_llm_client(monkeypatch, stub)
    ctx = _DriveCtx(tmp_path)
    ok, msg = check_safety("create_github_issue", {"title": "x"}, ctx=ctx)
    assert ok is False and "SAFETY_VIOLATION" in msg
    # today's blocking path for an empty body IS the parse lane incl. its one repair
    # retry; the point here is the rate-limit fail-open lane must NOT engage.
    assert len(stub.calls) == 2
    assert _read_events(ctx, "safety_check_rate_limited") == []


def test_safe_verdict_with_invalid_reported_cost_stays_safe(monkeypatch, tmp_path, caplog):
    """A garbage usage cost must never flip a SAFE verdict into a violation: the
    cost degrades to unknown and the verdict survives (no-fabricated-costs rule)."""
    import logging
    from ouroboros.safety import check_safety

    stub = _ScriptedLLMClient([('{"status":"SAFE","reason":"ok"}', {"cost": "abc", "prompt_tokens": 5})])
    _patch_llm_client(monkeypatch, stub)
    with caplog.at_level(logging.WARNING):
        ok, msg = check_safety("create_github_issue", {"title": "x"}, ctx=_DriveCtx(tmp_path))
    assert ok is True and msg == ""
    assert "invalid reported cost" in caplog.text



def test_latched_short_circuit_does_not_extend_the_storm_window(monkeypatch, tmp_path, _no_backoff):
    """A latched answer carries no NEW storm evidence, so it must not re-arm the
    window — else the advised retry would keep the latch alive forever. Its audit row
    also names the zero-attempt shape instead of claiming a retry happened."""
    import ouroboros.safety as safety
    from ouroboros.safety import check_safety

    stub = _ScriptedLLMClient([_RateLimitError("Rate limit exceeded")])
    _patch_llm_client(monkeypatch, stub)
    ctx = _DriveCtx(tmp_path)

    check_safety("create_github_issue", {"title": "x"}, ctx=ctx)  # arms the latch
    window = safety._SAFETY_STORM_UNTIL
    assert window > time.time()
    check_safety("create_github_issue", {"title": "y"}, ctx=ctx)  # latched short-circuit

    assert safety._SAFETY_STORM_UNTIL == window, "a latched outcome must not extend the window"
    rows = _read_events(ctx, "safety_check_rate_limited")
    assert [r["action"] for r in rows] == [
        "blocked_unchecked_after_retry", "blocked_unchecked_storm_latched",
    ]


def test_unparseable_verdict_then_rate_limited_repair_takes_the_typed_outcome(
    monkeypatch, tmp_path, _no_backoff,
):
    """The parse-REPAIR lane is a provider call too: an unparseable verdict whose repair
    attempts both 429 must end in the same typed SAFETY_UNAVAILABLE outcome (latch
    armed), never in a SAFETY_VIOLATION accusation born of the transport."""
    import ouroboros.safety as safety
    from ouroboros.safety import check_safety

    stub = _ScriptedLLMClient([
        ("this is not json at all", {}),
        _RateLimitError("Rate limit exceeded"),
        _RateLimitError("Rate limit exceeded"),
    ])
    _patch_llm_client(monkeypatch, stub)
    ctx = _DriveCtx(tmp_path)

    ok, msg = check_safety("create_github_issue", {"title": "x"}, ctx=ctx)

    assert ok is False
    assert msg.startswith("⚠️ SAFETY_UNAVAILABLE:")
    assert "SAFETY_VIOLATION" not in msg.splitlines()[0]
    assert len(stub.calls) == 3, "verdict + one repair retry wall, nothing more"
    assert safety._SAFETY_STORM_UNTIL > time.time(), "the repair-lane 429 arms the latch"


def test_backoff_that_consumes_the_deadline_permits_no_second_attempt(monkeypatch, tmp_path):
    """The deadline is re-checked AFTER the backoff sleep: a deadline that expires
    DURING the sleep must not be followed by a second paid provider call."""
    import ouroboros.safety as safety
    from ouroboros.safety import check_safety

    from ouroboros.deadline_utils import parse_deadline_ts

    stub = _ScriptedLLMClient([_RateLimitError("Rate limit exceeded")])
    _patch_llm_client(monkeypatch, stub)
    ctx = _DriveCtx(tmp_path)
    iso = "2030-01-01T00:00:10Z"
    deadline = parse_deadline_ts(iso).timestamp()
    ctx.task_metadata = {"deadline_at": iso}
    clock = {"now": deadline - 5.0}
    monkeypatch.setattr(safety.time, "time", lambda: clock["now"])
    monkeypatch.setattr(
        safety.time, "sleep", lambda _d: clock.__setitem__("now", deadline + 1.0),
    )

    ok, msg = check_safety("create_github_issue", {"title": "x"}, ctx=ctx)

    assert ok is False and msg.startswith("⚠️ SAFETY_UNAVAILABLE:")
    assert len(stub.calls) == 1, "the sleep spent the deadline; no second paid call"
    assert _read_events(ctx, "safety_check_rate_limited")[-1]["action"] == \
        "blocked_unchecked_deadline_expired"


# ---------------------------------------------------------------------------
# Bounded SUBJECT budget: an unreviewable call is refused, never truncated
# ---------------------------------------------------------------------------


class _MustNotBeCalled:
    """LLMClient stand-in for the over-budget path: any provider call is a failure."""

    def chat(self, **kwargs):  # pragma: no cover - reaching this IS the failure
        raise AssertionError("the safety model must not be called for an over-budget subject")


def test_oversized_subject_blocks_without_a_model_call(monkeypatch, tmp_path):
    """A subject above ``_SAFETY_SUBJECT_CHAR_BUDGET`` is refused fail-closed BEFORE
    any provider call: never truncated (the reviewer would miss the tail), never
    executed unchecked, and classified as a policy denial (first-line ``_BLOCKED``
    marker), not as SAFETY_VIOLATION — the command was not judged, only its size."""
    import ouroboros.safety as safety
    from ouroboros.safety import check_safety

    _patch_llm_client(monkeypatch, _MustNotBeCalled())
    ctx = _DriveCtx(tmp_path)
    huge = "x" * (safety._SAFETY_SUBJECT_CHAR_BUDGET + 1)

    ok, msg = check_safety("create_github_issue", {"title": huge}, ctx=ctx)

    assert ok is False, "an unreviewable guarded call must not execute"
    assert msg.splitlines()[0].startswith("⚠️ SAFETY_SUBJECT_TOO_LARGE_BLOCKED:")
    assert "SAFETY_VIOLATION" not in msg
    assert "split" in msg, "the refusal carries the working alternative"
    # The first version of this refusal told the agent to stage the body with
    # write_file (a POLICY_SKIP tool) and run the staged file. That is a route
    # around the supervisor: it reviews the call it is handed, and a script run
    # from a path hands it a path, not the bytes. The remediation must not name
    # it, and the earlier test asserting "write_file" in msg pinned the defect.
    assert "write_file" not in msg
    assert "staged" not in msg

    rows = _read_events(ctx, "safety_subject_too_large")
    assert len(rows) == 1
    assert rows[0]["tool"] == "create_github_issue"
    assert rows[0]["subject_chars"] > safety._SAFETY_SUBJECT_CHAR_BUDGET
    assert rows[0]["budget_chars"] == safety._SAFETY_SUBJECT_CHAR_BUDGET


def test_non_ascii_subject_near_cap_is_not_inflated(monkeypatch, tmp_path):
    """``ensure_ascii=False`` keeps serialized length ≈ input length: a near-cap
    Cyrillic payload must reach the model instead of being refused through the
    6x ``\\uXXXX`` inflation the default serialization would apply — otherwise
    a script the budget admits in ASCII would be refused for its Cyrillic twin."""
    from ouroboros.safety import check_safety

    stub = _ScriptedLLMClient([
        (json.dumps({"status": "SAFE"}),
         {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.0}),
    ])
    _patch_llm_client(monkeypatch, stub)
    ctx = _DriveCtx(tmp_path)
    big_cyrillic = "ы" * 200_000  # inflated 6x this would blow the 250k budget

    ok, _msg = check_safety("create_github_issue", {"title": big_cyrillic}, ctx=ctx)

    assert ok is True
    assert len(stub.calls) == 1, "the near-cap non-ASCII subject reached the model"
    assert not _read_events(ctx, "safety_subject_too_large")


def test_lone_surrogate_subject_serializes_without_exploding():
    """A validly parsed ``\\ud800`` escape in tool args must not crash the
    UTF-8 provider send downstream: the rendered subject substitutes it."""
    from ouroboros.safety import _render_subject_json

    rendered = _render_subject_json({"title": "bad \ud800 surrogate"})

    rendered.encode("utf-8")  # must not raise
    assert "bad" in rendered


def test_safe_subject_whitelist_stays_llm_free_for_oversized_calls(monkeypatch, tmp_path):
    """INTENTIONAL pass-through pin: a deterministic safe-subject call
    (``run_command`` with a whitelisted argv head) never builds a check prompt,
    so the subject budget does not apply — there is nothing to inflate. The
    budget governs only subjects that would ENTER the LLM check."""
    from ouroboros.safety import check_safety

    _patch_llm_client(monkeypatch, _MustNotBeCalled())
    ctx = _DriveCtx(tmp_path)

    ok, msg = check_safety(
        "run_command", {"cmd": ["pytest", "-k", "x" * 300_000]}, ctx=ctx)

    assert ok is True and msg == ""
    assert not _read_events(ctx, "safety_subject_too_large")
