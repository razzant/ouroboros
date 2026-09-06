"""Loop miscellaneous regressions: the message stream and pacing seams.

Consolidated from former ``test_loop_incoming_messages.py`` (image payload
preservation) and ``test_loop_skill_finalization.py``, then divided by theme
(the v7 L-B test split): the task-acceptance gate lives in
``test_loop_acceptance_gate.py``, the self-authored skill finalization gate in
``test_loop_skill_finalization.py``, the ``run_llm_loop`` round/finalization
suite in ``test_run_llm_loop.py`` and the image auto-attach seam in
``test_loop_image_attach.py``.

Kept as the home for loop micro-regressions that do not justify a standalone
file: message draining, owner directives, self-check and pacing injections,
the final-answer latch, the provider-death retry wall and the per-task
web gate.
"""
from __future__ import annotations

import json
import queue
from types import SimpleNamespace

import pytest

import ouroboros.loop as loop_mod
from ouroboros.loop_llm_call import RETRY_WALL_EXHAUSTED_KEY
from ouroboros.loop import (
    run_llm_loop,
)
from ouroboros.loop_acceptance import _latch_final_answer_marker, _server_web_allowed_by_task
from ouroboros.loop_messages import _initialize_owner_directives
from ouroboros.loop_nudges import _maybe_inject_self_check, _maybe_inject_time_budget_milestone
from ouroboros.loop_round_limits import _drain_incoming_messages



# ---------------------------------------------------------------------------
# _drain_incoming_messages — telegram image payload preservation
# ---------------------------------------------------------------------------


def test_drain_incoming_messages_preserves_image_payload():
    messages: list = []
    incoming_messages: queue.Queue = queue.Queue()
    incoming_messages.put({
        "text": "photo from telegram",
        "image_base64": "aW1hZ2U=",
        "image_mime": "image/png",
        "image_caption": "photo from telegram",
    })

    _drain_incoming_messages(
        messages=messages,
        incoming_messages=incoming_messages,
        drive_root=None,
        task_id="",
        event_queue=None,
        _owner_msg_seen=set(),
    )

    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    content = messages[0]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert content[0]["text"] == "[Message from my human]: photo from telegram"
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"] == "data:image/png;base64,aW1hZ2U="


def test_owner_directives_survive_compaction_without_control_prose(tmp_path):
    from ouroboros import task_pacing
    from ouroboros.deadline_utils import parse_deadline_ts
    from ouroboros.owner_mailbox import (
        KIND_FINALIZE_NOW,
        drain_owner_entries,
        write_owner_message,
    )

    ctx = SimpleNamespace()
    messages = [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "Initial requirement verbatim"},
    ]
    _initialize_owner_directives(ctx, messages)
    incoming: queue.Queue = queue.Queue()
    incoming.put({"text": "direct follow-up", "client_message_id": "direct-1"})
    write_owner_message(tmp_path, "mailbox follow-up", task_id="root", msg_id="mail-1")
    write_owner_message(
        tmp_path, "deadline control", task_id="root", msg_id="control-1",
        kind=KIND_FINALIZE_NOW,
    )
    control_entry = next(
        row for row in drain_owner_entries(tmp_path, "root")
        if row["msg_id"] == "control-1"
    )
    expected_deadline = (
        parse_deadline_ts(control_entry["ts"]).timestamp()
        + task_pacing.effective_finalization_reserve_sec(ctx)
    )

    controls = _drain_incoming_messages(
        messages,
        incoming,
        tmp_path,
        "root",
        None,
        set(),
        owner_ctx=ctx,
    )

    assert set(controls) == {"finalize_now", "finalize_deadline_ts"}
    assert controls["finalize_now"] == "deadline control"
    assert controls["finalize_deadline_ts"] == expected_deadline
    assert [row["source"] for row in ctx._owner_directives] == [
        "initial_user", "direct_incoming", "owner_mailbox",
    ]
    assert ctx._owner_directives[0]["content"] == "Initial requirement verbatim"
    assert ctx._owner_directives[1]["msg_id"] == "direct-1"
    assert ctx._owner_directives[2] == {
        "source": "owner_mailbox",
        "content": "mailbox follow-up",
        "msg_id": "mail-1",
    }
    assert "deadline control" not in json.dumps(ctx._owner_directives)


def test_maybe_inject_self_check_handles_assistant_none_content():
    messages = [
        {"role": "user", "content": "inspect"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call-1",
                "type": "function",
                "function": {"name": "read_file", "arguments": "{}"},
            }],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "done"},
    ]
    progress = []

    injected = _maybe_inject_self_check(
        15,
        30,
        messages,
        {"cost": 0.0},
        progress.append,
    )

    assert injected is True
    assert messages[-1]["role"] == "user"
    assert "[CHECKPOINT 1" in messages[-1]["content"]
    assert progress


def test_time_budget_milestone_injects_once_per_threshold(monkeypatch):
    messages = [{"role": "user", "content": "solve"}]
    ctx = SimpleNamespace(
        task_metadata={
            "created_at": "2026-06-10T00:00:00Z",
            "deadline_at": "2026-06-10T10:00:00Z",
        },
    )

    from datetime import datetime, timezone

    monkeypatch.setattr("ouroboros.task_pacing.utc_now", lambda: datetime(2026, 6, 10, 5, 1, tzinfo=timezone.utc))

    injected = _maybe_inject_time_budget_milestone(
        messages,
        SimpleNamespace(_ctx=ctx),
        event_queue=None,
        task_id="task-time",
        drive_logs=None,
    )
    injected_again = _maybe_inject_time_budget_milestone(messages, SimpleNamespace(_ctx=ctx))

    assert injected is True
    assert injected_again is False
    assert "[TIME BUDGET" in messages[-1]["content"]
    assert "50% remaining" in messages[-1]["content"]
    assert ctx._time_budget_milestones_seen == {"50%"}


def test_intrinsic_pacing_injects_without_deadline(monkeypatch):
    """No deadline_at: surface elapsed/rounds/cost once per interval bucket.
    v6.60.0: the FINAL ANSWER phrase appears ONLY when the task contract declares
    answer_protocol="final_answer_line" (marker phrases are protocol-gated)."""
    messages = [{"role": "user", "content": "solve"}]
    ctx = SimpleNamespace(task_metadata={"created_at": "2026-06-10T00:00:00Z"})  # no deadline_at
    from datetime import datetime, timezone

    monkeypatch.delenv("OUROBOROS_PACING_INTERVAL_SEC", raising=False)
    # 20 min elapsed, default interval 600s -> bucket 2.
    monkeypatch.setattr("ouroboros.task_pacing.utc_now", lambda: datetime(2026, 6, 10, 0, 20, tzinfo=timezone.utc))

    injected = _maybe_inject_time_budget_milestone(
        messages, SimpleNamespace(_ctx=ctx), round_idx=7,
        accumulated_usage={"cost": 1.25}, task_id="t",
    )
    injected_again = _maybe_inject_time_budget_milestone(
        messages, SimpleNamespace(_ctx=ctx), round_idx=8, accumulated_usage={"cost": 1.4},
    )

    assert injected is True
    assert injected_again is False  # same bucket -> not repeated
    assert "[PACING" in messages[-1]["content"]
    assert "Rounds so far: 7" in messages[-1]["content"]
    assert "FINAL ANSWER:" not in messages[-1]["content"]  # no protocol declared

    # With the protocol declared, the salvage phrase rides the SAME milestone.
    proto_ctx = SimpleNamespace(
        task_metadata={"created_at": "2026-06-10T00:00:00Z"},
        task_contract={"answer_protocol": "final_answer_line"},
    )
    proto_messages = [{"role": "user", "content": "solve"}]
    assert _maybe_inject_time_budget_milestone(
        proto_messages, SimpleNamespace(_ctx=proto_ctx), round_idx=7,
        accumulated_usage={"cost": 1.25}, task_id="t2",
    ) is True
    assert "FINAL ANSWER:" in proto_messages[-1]["content"]


def test_latch_final_answer_marker_captures_explicit_marker_only():
    trace = {"tool_calls": [{"tool": "read_file"}]}
    _latch_final_answer_marker(trace, "analysis\nFINAL ANSWER: 123")
    assert trace["best_valid_final_answer"] == "123"
    assert trace["best_valid_final_answer_tools"] == 1
    _latch_final_answer_marker(trace, "answer-ish prose without marker")
    assert trace["best_valid_final_answer"] == "123"


def test_latch_final_answer_marker_counts_same_turn_tool_calls():
    trace = {"tool_calls": [{"tool": "read_file"}]}
    current = [{"function": {"name": "run_command"}}, {"function": {"name": "verify_and_record"}}]
    _latch_final_answer_marker(trace, "FINAL ANSWER: draft", current_tool_calls=current)
    assert trace["best_valid_final_answer"] == "draft"
    # Same-turn tool calls are newer grounding and must invalidate this latch unless
    # the model re-emits the marker after those tools complete.
    assert trace["best_valid_final_answer_tools"] == 1


def test_server_web_allowed_respects_task_resource_contract():
    assert _server_web_allowed_by_task(SimpleNamespace(task_contract={})) is True
    assert _server_web_allowed_by_task(SimpleNamespace(task_contract={"allowed_resources": {"web": False}})) is False
    assert _server_web_allowed_by_task(SimpleNamespace(task_contract={"allowed_resources": {"network": False}})) is False
    assert _server_web_allowed_by_task(SimpleNamespace(task_contract={"disabled_tools": ["web_search"]})) is True


def test_intrinsic_pacing_disabled_when_interval_zero(monkeypatch):
    messages = [{"role": "user", "content": "solve"}]
    ctx = SimpleNamespace(task_metadata={"created_at": "2026-06-10T00:00:00Z"})
    from datetime import datetime, timezone

    monkeypatch.setenv("OUROBOROS_PACING_INTERVAL_SEC", "0")
    monkeypatch.setattr("ouroboros.task_pacing.utc_now", lambda: datetime(2026, 6, 10, 1, 0, tzinfo=timezone.utc))

    assert _maybe_inject_time_budget_milestone(messages, SimpleNamespace(_ctx=ctx), round_idx=3) is False


def test_deadline_local_finalize_gate(monkeypatch):
    """Self-finalize only when a REAL deadline is within the grace window."""
    from datetime import datetime, timezone

    captured = {}

    def _fake_final(ctx, *, prompt, fallback_text, reason_code):
        captured["reason_code"] = reason_code
        return ("BEST EFFORT", {"reason_code": reason_code}, {})

    monkeypatch.setattr(loop_mod, "_forced_final_answer", _fake_final)
    # v6.54.4: the gate consults the task_pacing effective reserve SSOT.
    monkeypatch.setattr("ouroboros.task_pacing.effective_finalization_reserve_sec", lambda ctx: 120.0)
    monkeypatch.setattr(loop_mod, "utc_now", lambda: datetime(2026, 6, 10, 9, 59, 0, tzinfo=timezone.utc))

    # Far from deadline (10:30 vs now 09:59 -> ~31 min left > 120s) -> no finalize.
    far = SimpleNamespace(_ctx=SimpleNamespace(task_metadata={"deadline_at": "2026-06-10T10:30:00Z"}))
    assert loop_mod._maybe_deadline_local_finalize(SimpleNamespace(), far) is None
    # Within grace (10:00 vs now 09:59 -> 60s < 120s) -> finalize best-effort.
    near = SimpleNamespace(_ctx=SimpleNamespace(task_metadata={"deadline_at": "2026-06-10T10:00:00Z"}))
    result = loop_mod._maybe_deadline_local_finalize(SimpleNamespace(), near)
    assert result is not None and result[0] == "BEST EFFORT"
    assert captured["reason_code"] == "deadline_local"
    # No deadline_at at all -> never fires (no synthesized deadline).
    none_ctx = SimpleNamespace(_ctx=SimpleNamespace(task_metadata={}))
    assert loop_mod._maybe_deadline_local_finalize(SimpleNamespace(), none_ctx) is None


# ---------------------------------------------------------------------------
# Skill finalization gate (self-authored skills must reach ready+enabled
# before the loop accepts a final text response)
# ---------------------------------------------------------------------------


def test_unknown_dispatched_outcome_skips_cross_model_fallback(tmp_path, monkeypatch):
    from ouroboros.tools.registry import ToolRegistry

    class FakeLLM:
        def default_model(self):
            return "test-model"

    calls = {"primary": 0, "fallback": 0}

    def ambiguous(*args, **_kwargs):
        calls["primary"] += 1
        usage = args[10]
        usage["_last_llm_error"] = "provider outcome unknown"
        usage["_last_llm_error_kind"] = "provider_outcome_unknown"
        usage["_last_llm_retry_same_request"] = False
        return None, 0.0

    def forbidden_fallback(**_kwargs):
        calls["fallback"] += 1
        raise AssertionError("unknown physical work must stop the paid chain")

    monkeypatch.setattr(loop_mod, "call_llm_with_retry", ambiguous)
    monkeypatch.setattr(loop_mod, "_run_cross_model_fallback_chain", forbidden_fallback)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)

    result, usage, _trace = run_llm_loop(
        messages=[{"role": "user", "content": "go"}],
        tools=registry,
        llm=FakeLLM(),
        drive_logs=tmp_path,
        emit_progress=lambda _text, *, incident=None: None,
        incoming_messages=queue.Queue(),
        task_id="unknown-provider-task",
        drive_root=tmp_path,
    )

    assert calls == {"primary": 1, "fallback": 0}
    assert usage["_last_llm_error_kind"] == "provider_outcome_unknown"
    assert "no retry or paid fallback" in result


# ---------------------------------------------------------------------------
# OB-01 — provider death does not re-burn the budget on a second forced call
# ---------------------------------------------------------------------------


def _provider_death_ctx(tmp_path, accumulated):
    """A minimal forced-rail context. ``tools=None`` on purpose: every forced
    helper then takes its documented tool-less path, so the test observes the
    loop's own decisions instead of a registry stub's."""
    return loop_mod._RoundLimitContext(
        messages=[
            {"role": "user", "content": "do the thing"},
            {"role": "assistant", "content": "PARTIAL RESULT: step one is done."},
        ],
        llm=SimpleNamespace(),
        active_model="openai/gpt-5.5",
        active_effort="medium",
        max_retries=3,
        drive_logs=tmp_path,
        task_id="task-provider-death",
        round_idx=7,
        event_queue=None,
        accumulated_usage=accumulated,
        task_type="task",
        active_use_local=False,
        max_rounds=200,
        drive_root=tmp_path,
    )


def _count_forced_helpers(monkeypatch):
    """Wrap the non-model work of the forced rail with call counters, keeping the
    REAL implementations so "it still runs" is observed, not simulated."""
    seen = {"model": [], "services": [], "drain": []}
    real_services = loop_mod._finalize_forced_services
    real_drain = loop_mod._drain_forced_owner_directives

    def _model(ctx):
        seen["model"].append(1)
        return "FRESH FORCED ANSWER"

    def _services(ctx, trace):
        seen["services"].append(1)
        return real_services(ctx, trace)

    def _drain(ctx, trace):
        seen["drain"].append(1)
        return real_drain(ctx, trace)

    monkeypatch.setattr(loop_mod, "_call_forced_model_once", _model)
    monkeypatch.setattr(loop_mod, "_finalize_forced_services", _services)
    monkeypatch.setattr(loop_mod, "_drain_forced_owner_directives", _drain)
    return seen


def test_provider_death_skips_the_forced_call_when_the_retry_wall_is_spent(tmp_path, monkeypatch):
    """The whole point of OB-01: the transport already spent the same-model retry
    wall, so the forced finalization must NOT re-burn the budget on a request that
    cannot land. Everything the forced rail owns besides the call still runs."""
    seen = _count_forced_helpers(monkeypatch)
    accumulated = {  # realistic transport stamps of an exhausted transient wall
        RETRY_WALL_EXHAUSTED_KEY: True,
        "execution_status": "infra_failed",
        "reason_code": "llm_api_error",
        "_last_llm_error_kind": "provider_transient",
    }
    ctx = _provider_death_ctx(tmp_path, accumulated)
    messages_before = [dict(m) for m in ctx.messages]

    text, usage, trace = loop_mod._handle_provider_unavailable(ctx)

    assert seen["model"] == []                    # ZERO llm calls from this rail
    assert len(seen["services"]) == 1             # services still finalized
    assert len(seen["drain"]) == 1                # exactly ONE directive drain
    # Salvage still delivered, and the delivery candidate still packaged.
    assert "PARTIAL RESULT: step one is done." in text
    assert trace["forced_finalization"]["reason_code"] == "provider_unavailable"
    # Replay durability: an unsent prompt must never reach the transcript, or a
    # resume would read it as a request the model ignored.
    assert [dict(m) for m in ctx.messages] == messages_before
    assert not any(
        "[PROVIDER_UNAVAILABLE]" in str(m.get("content") or "") for m in ctx.messages
    )
    # False completion: an outage is an INFRA FAILURE, and no model answer was
    # extracted, so the best_effort gate must not be handed a typed success fact.
    assert usage["reason_code"] == "provider_unavailable"
    assert usage["execution_status"] == "infra_failed"
    assert "_best_effort_extracted" not in usage


def test_provider_death_still_makes_the_forced_call_when_the_wall_is_unspent(tmp_path, monkeypatch):
    """The skip is not the new default. A PERMANENT refusal (auth/quota/bad
    request) fails fast and leaves the wall unspent, so the forced rail keeps the
    one chance its class is entitled to."""
    seen = _count_forced_helpers(monkeypatch)
    accumulated = {}  # no marker: the transport never exhausted its retries
    ctx = _provider_death_ctx(tmp_path, accumulated)

    text, usage, _trace = loop_mod._handle_provider_unavailable(ctx)

    assert seen["model"] == [1]
    assert "FRESH FORCED ANSWER" in text
    assert usage["execution_status"] == "infra_failed"
    assert any(
        "[PROVIDER_UNAVAILABLE]" in str(m.get("content") or "") for m in ctx.messages
    )


@pytest.mark.parametrize(
    "marker, expect_call",
    [
        ("unexpected-string", False),
        (1, False),
        ({"nested": "garbage"}, False),
        (0, True),
        ("", True),
        (None, True),
    ],
    ids=["truthy_str", "truthy_int", "truthy_dict", "zero", "empty_str", "none"],
)
def test_malformed_retry_wall_marker_reads_as_a_plain_truth_value(
    tmp_path, monkeypatch, marker, expect_call,
):
    """A shared usage dict can hold anything, so the read is `bool(...)` and
    nothing else: any truthy value means the wall is spent, any falsy value means
    it is not. No shape assumption, no crash, no silent third behaviour."""
    seen = _count_forced_helpers(monkeypatch)
    ctx = _provider_death_ctx(tmp_path, {
        RETRY_WALL_EXHAUSTED_KEY: marker,
        "execution_status": "infra_failed", "reason_code": "llm_api_error",
    })

    _text, usage, _trace = loop_mod._handle_provider_unavailable(ctx)

    assert seen["model"] == ([1] if expect_call else [])
    assert usage["execution_status"] == "infra_failed"


@pytest.mark.parametrize(
    "reason_code, marker",
    [
        ("round_limit", "[ROUND_LIMIT] wrap up"),
        ("finalization_grace", "[FINALIZE_NOW] wrap up"),
        ("deadline_local", "[DEADLINE] wrap up"),
        ("owner_requested_finalization", "[OWNER_STOP] wrap up"),
        ("budget_exhausted", "[BUDGET] wrap up"),
        ("children_unabsorbed", "[CHILDREN] wrap up"),
    ],
)
def test_other_forced_rails_never_skip_even_with_the_wall_marker_set(
    tmp_path, monkeypatch, reason_code, marker,
):
    """The skip is gated on the provider-death rail SPECIFICALLY. Every OTHER
    reason code `loop.py` passes to `_forced_final_answer` — round limit,
    finalization grace, deadline, owner stop, budget exhaustion and unabsorbed
    children — still makes its one forced call even when a transient wall was
    spent earlier in the same task: those rails end for their own reasons, not
    because the provider is unreachable. The list is exhaustive against the
    `reason_code=` literals in loop.py, so a new rail cannot silently inherit
    the skip."""
    seen = _count_forced_helpers(monkeypatch)
    ctx = _provider_death_ctx(tmp_path, {RETRY_WALL_EXHAUSTED_KEY: True})

    text, _usage, _trace = loop_mod._forced_final_answer(
        ctx, prompt=marker, fallback_text="fallback", reason_code=reason_code,
    )

    assert seen["model"] == [1]
    assert "FRESH FORCED ANSWER" in text
    assert any(marker in str(m.get("content") or "") for m in ctx.messages)
