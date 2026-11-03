"""Typed host-driven model handover and fallback-finalization regressions."""

from __future__ import annotations

import queue
from types import SimpleNamespace

from ouroboros.loop_delivery import _no_tool_final_answer
from ouroboros.loop_model_call import _adopt_fallback_route
from ouroboros.loop_nudges import _maybe_inject_finalization_nudges
from ouroboros.loop_round_limits import _RoundLimitContext
from ouroboros.outcomes import EXECUTION_DEGRADED, derive_loop_outcome
from ouroboros.tools.registry import ToolRegistry


def _ctx(tmp_path):
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    trace = {"tool_calls": [{"tool": "vcs_read", "status": "ok"}], "reasoning_notes": []}
    registry._ctx._execution_trace = trace
    registry._ctx._accumulated_usage = {}
    return registry, trace


def test_successful_fallback_adoption_records_typed_handover(tmp_path):
    registry, trace = _ctx(tmp_path)
    registry._ctx.active_model = "codex=gpt-6-astra"
    registry._ctx.active_use_local = False
    usage = registry._ctx._accumulated_usage
    messages = [{"role": "user", "content": "work"}]
    fallback_messages = list(messages)

    _adopt_fallback_route(
        registry._ctx,
        registry,
        "claude-fable-5-1",
        False,
        messages,
        fallback_messages,
        None,
        "max",
        [],
        usage,
        handover_from_model="codex=gpt-6-astra",
        handover_reason="invalid_continuation",
        tool_calls_at_handover=1,
    )

    assert registry._ctx._authoring_handover["from_model"] == "codex=gpt-6-astra"
    assert registry._ctx._authoring_handover["to_model"] == "claude-fable-5-1"
    assert usage["authoring_handovers"][0]["reason"] == "invalid_continuation"
    assert trace["route_handovers"][0]["tool_calls_at_handover"] == 1


def test_first_toolless_response_after_handover_gets_one_recovery_round(tmp_path):
    registry, trace = _ctx(tmp_path)
    registry._ctx._authoring_handover = {
        "from_model": "codex=gpt-6-astra",
        "to_model": "claude-fable-5-1",
        "reason": "invalid_continuation",
        "tool_calls_at_handover": 1,
        "recovery_prompted": False,
    }
    messages = []
    progress = []

    injected = _maybe_inject_finalization_nudges(
        registry, tmp_path, "handover-1", trace, "the worktree is clean", messages,
        progress.append,
    )

    assert injected is True
    assert registry._ctx._authoring_handover["recovery_prompted"] is True
    assert registry._ctx._accumulated_usage == {}
    assert any("model handover" in str(row.get("content") or "") for row in messages)
    assert any("handover recovery" in str(item).lower() for item in progress)


def test_second_toolless_response_after_handover_is_degraded(tmp_path):
    registry, trace = _ctx(tmp_path)
    registry._ctx._authoring_handover = {
        "from_model": "codex=gpt-6-astra",
        "to_model": "claude-fable-5-1",
        "reason": "invalid_continuation",
        "tool_calls_at_handover": 1,
        "recovery_prompted": True,
    }

    injected = _maybe_inject_finalization_nudges(
        registry, tmp_path, "handover-2", trace, "the worktree is still clean", [],
        lambda _text: None,
    )

    assert injected is False
    assert registry._ctx._authoring_handover is None
    assert trace["authoring_handover_incomplete"]["status"] == "incomplete"
    assert registry._ctx._accumulated_usage["execution_status"] == EXECUTION_DEGRADED
    assert registry._ctx._accumulated_usage["reason_code"] == "authoring_handover_incomplete"
    outcome = derive_loop_outcome(
        "the worktree is still clean",
        registry._ctx._accumulated_usage,
        trace,
    )
    assert outcome["outcome_axes"]["execution"]["status"] == EXECUTION_DEGRADED
    assert outcome["outcome_axes"]["execution"]["reason_code"] == "authoring_handover_incomplete"


def test_handover_is_fulfilled_when_successor_uses_a_tool(tmp_path):
    registry, trace = _ctx(tmp_path)
    registry._ctx._authoring_handover = {
        "from_model": "codex=gpt-6-astra",
        "to_model": "claude-fable-5-1",
        "reason": "fallback",
        "tool_calls_at_handover": 1,
        "recovery_prompted": False,
    }
    trace["tool_calls"].append({"tool": "read_file", "status": "ok"})

    injected = _maybe_inject_finalization_nudges(
        registry, tmp_path, "handover-3", trace, "done", [], lambda _text: None,
    )

    assert injected is False
    assert registry._ctx._authoring_handover is None
    assert getattr(registry._ctx, "_authoring_handover_incomplete", False) is False


def test_ordinary_toolless_final_without_handover_is_unchanged(tmp_path):
    registry, trace = _ctx(tmp_path)
    injected = _maybe_inject_finalization_nudges(
        registry, tmp_path, "handover-4", trace, "ordinary answer", [], lambda _text: None,
    )
    assert injected is False
    assert registry._ctx._accumulated_usage == {}


def test_production_finalizer_requires_selection_during_handover_recovery(tmp_path):
    from copy import deepcopy
    from ouroboros.loop_delivery import completion_observation, consume_completion_request
    from ouroboros.tools.control_runtime import _finish_task
    registry, trace = _ctx(tmp_path)
    registry._ctx._authoring_handover = {
        "from_model": "codex=gpt-6-astra",
        "to_model": "claude-fable-5-1",
        "reason": "invalid_continuation",
        "tool_calls_at_handover": 1,
        "recovery_prompted": False,
    }
    messages = []
    usage = registry._ctx._accumulated_usage
    limit = _RoundLimitContext(
        messages=messages,
        llm=SimpleNamespace(),
        active_model="claude-fable-5-1",
        active_effort="medium",
        max_retries=3,
        drive_logs=tmp_path,
        task_id="handover-finalizer",
        round_idx=1,
        event_queue=None,
        accumulated_usage=usage,
        task_type="task",
        active_use_local=False,
        max_rounds=100,
        drive_root=tmp_path,
        root_task_id="handover-finalizer",
        tools=registry,
        llm_trace=trace,
        incoming_messages=queue.Queue(),
        owner_msg_seen=set(),
        tool_schemas=[],
    )
    # Match run_llm_loop's binding before calling its finalizer in isolation.
    registry._ctx.task_id, registry._ctx.messages = limit.task_id, messages

    first = _no_tool_final_answer(
        "the worktree is clean", limit, trace, registry, queue.Queue(), set(), lambda _text: None,
    )

    assert first is None
    candidate = registry._ctx._delivery_candidate
    assert candidate.finalization_control == "authoring_handover_recovery_required"
    assert registry._ctx._delivery_control_required is True
    assert not any("DELIVERY_FINALIZATION_CONTROL" in str(row.get("content") or "") for row in messages)

    before_second = deepcopy(messages)
    second = _no_tool_final_answer(
        "I will continue the requested work", limit, trace, registry, queue.Queue(), set(), lambda _text: None,
    )

    assert second is None
    assert candidate.full_text == "the worktree is clean"
    assert messages[:len(before_second)] == before_second
    assert messages[len(before_second)] == {"role": "assistant", "content": "I will continue the requested work"}
    assert all(row["role"] == "user" for row in messages[len(before_second) + 1:])
    assert messages[-1]["role"] == "user" and "No completion selection" in messages[-1]["content"]
    registry._ctx._completion_observation = completion_observation(registry._ctx, trace)
    _finish_task(registry._ctx, action="finish", answer="No further work was performed.")
    assert consume_completion_request(registry, limit, trace)
    third = _no_tool_final_answer("No further work was performed.", limit, trace, registry,
                                 queue.Queue(), set(), lambda _text: None, explicit_candidate=True)
    assert third is not None and third[0] == "No further work was performed."
    assert usage["reason_code"] == "authoring_handover_incomplete"
    assert third[1]["execution_status"] == "degraded"


def test_handover_incomplete_trace_survives_usage_round_cleanup(tmp_path):
    registry, trace = _ctx(tmp_path)
    trace["authoring_handover_incomplete"] = {
        "from_model": "codex=gpt-6-astra",
        "to_model": "claude-fable-5-1",
    }
    outcome = derive_loop_outcome(
        "still incomplete",
        {"rounds": 2},
        trace,
    )
    assert outcome["outcome_axes"]["execution"]["status"] == EXECUTION_DEGRADED
    assert outcome["outcome_axes"]["execution"]["reason_code"] == "authoring_handover_incomplete"


def test_same_model_wait_reprepare_does_not_overwrite_pending_handover():
    from ouroboros.loop_model_call import _pending_model_wait_handover

    ctx = SimpleNamespace(_pending_model_wait_handover=("model-a", "model-b", 1))
    _pending_model_wait_handover(ctx, from_model="model-b", to_model="model-b", tool_calls=1)
    assert ctx._pending_model_wait_handover == ("model-a", "model-b", 1)
    _pending_model_wait_handover(ctx, from_model="model-b", to_model="model-c", tool_calls=2)
    assert ctx._pending_model_wait_handover == ("model-a", "model-c", 1)


def test_later_tools_heal_only_handover_warning_and_keep_history(tmp_path):
    registry, trace = _ctx(tmp_path)
    from ouroboros.loop_model_call import _record_authoring_handover
    _record_authoring_handover(registry._ctx, from_model="a", to_model="b",
                              reason="fallback", tool_calls_at_handover=1)
    row = registry._ctx._authoring_handover
    for _ in range(2):
        _maybe_inject_finalization_nudges(registry, tmp_path, "healing", trace,
                                         "status", [], lambda _: None)
    usage = registry._ctx._accumulated_usage
    assert derive_loop_outcome("status", usage, trace)["reason_code"] == "authoring_handover_incomplete"
    assert row["incomplete_observed"] and row["recovery_prompted"]
    trace["tool_calls"].append({"tool": "read_file", "status": "ok"})
    _maybe_inject_finalization_nudges(registry, tmp_path, "healing", trace,
                                     "finished", [], lambda _: None)
    assert "authoring_handover_incomplete" not in trace
    assert "execution_status" not in usage
    assert derive_loop_outcome("finished", usage, trace)["outcome_axes"]["execution"]["status"] == "ok"
    assert row["status"] == "recovered" and row["incomplete_observed"]
    from ouroboros.task_finalization import terminal_result_fields
    assert terminal_result_fields(usage)["authoring_handovers"][-1] == row


def test_repeated_status_does_not_skip_existing_skill_nudge(tmp_path, monkeypatch):
    from ouroboros import loop
    registry, trace = _ctx(tmp_path)
    registry._ctx._authoring_handover = {
        "from_model": "a", "to_model": "b", "tool_calls_at_handover": 1,
        "recovery_prompted": True,
    }
    monkeypatch.setattr(loop, "_skill_finalization_message", lambda *_: "skill still needs work")
    messages = []
    assert _maybe_inject_finalization_nudges(registry, tmp_path, "skill", trace,
                                           "status", messages, lambda _: None)
    assert "skill still needs work" in messages[-1]["content"]
    assert trace["authoring_handover_incomplete"]["status"] == "incomplete"
