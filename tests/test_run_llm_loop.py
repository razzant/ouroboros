"""``run_llm_loop`` round mechanics and finalization rails.

Split out of ``tests/test_loop_misc.py`` when that module was divided by
theme; every moved block is verbatim. Covers assistant metadata round-trips,
the direct-final admission fence, the budget rail, display-only reasoning,
finalize_now, per-task model overrides, the swarm force-plan gate and the
subagent handoff/absorption rails.
"""
from __future__ import annotations

import json
import queue
from types import SimpleNamespace

import pytest

from tests.test_completion_selection import finish

import ouroboros.loop as loop_mod
from ouroboros.loop import run_llm_loop


def test_run_llm_loop_preserves_assistant_tool_call_metadata(tmp_path, monkeypatch):
    from ouroboros.tools.registry import ToolRegistry

    messages = [{"role": "user", "content": "inspect"}]
    assistant_metadata = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "call-1",
            "type": "function",
            "function": {"name": "read_file", "arguments": "{}"},
        }],
        "reasoning": "I need the file first.",
        "reasoning_details": [{"type": "reasoning.text", "text": "I need the file first."}],
        "response_id": "gen-123",
    }
    seen_second_request = {}
    calls = {"count": 0}

    class FakeLLM:
        def default_model(self):
            return "test-model"

    def fake_call_llm_with_retry(_llm, request_messages, *_args, **_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return dict(assistant_metadata), 0.0
        seen_second_request["messages"] = [dict(item) for item in request_messages]
        return {"role": "assistant", "content": "done"}, 0.0

    def fake_handle_tool_calls(tool_calls, _tools, _drive_logs, _task_id, _executor, request_messages, _trace, _progress,
                               *, fit_candidate=None, tool_schemas=None):
        request_messages.append({"role": "tool", "tool_call_id": tool_calls[0]["id"], "content": "file"})
        return 0

    monkeypatch.setattr(loop_mod, "call_llm_with_retry", fake_call_llm_with_retry)
    monkeypatch.setattr(loop_mod, "handle_tool_calls", fake_handle_tool_calls)

    result, _usage, _trace = run_llm_loop(
        messages=messages,
        tools=ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path),
        llm=FakeLLM(),
        drive_logs=tmp_path,
        emit_progress=lambda _text, **_meta: None,
        incoming_messages=queue.Queue(),
        task_id="roundtrip",
        drive_root=tmp_path,
    )

    assert result == "done"
    assistant_msg = next(item for item in seen_second_request["messages"] if item.get("response_id") == "gen-123")
    assert assistant_msg["tool_calls"] == assistant_metadata["tool_calls"]
    assert assistant_msg["reasoning"] == assistant_metadata["reasoning"]
    assert assistant_msg["reasoning_details"] == assistant_metadata["reasoning_details"]
    assert assistant_msg["response_id"] == "gen-123"

def test_direct_final_admission_fence_consumes_followup_before_return(tmp_path, monkeypatch):
    import threading

    from ouroboros.owner_mailbox import write_owner_message
    from ouroboros.tools.registry import ToolRegistry

    class FakeLLM:
        def default_model(self):
            return "test-model"

    direct_agent = SimpleNamespace(
        _owner_message_admission_lock=threading.Lock(),
        _accepting_owner_messages=True,
        _busy=True,
        _current_task_id="direct-fence",
    )
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.owner_message_admission_lock = direct_agent._owner_message_admission_lock
    registry._ctx.owner_message_admission_agent = direct_agent
    calls = []

    def fake_call(_llm, request_messages, *_args, **_kwargs):
        calls.append([dict(row) for row in request_messages])
        if len(calls) == 1:
            write_owner_message(
                tmp_path,
                "Use FusionBrain images too",
                "direct-fence",
                msg_id="followup-1",
            )
            return {"role": "assistant", "content": "Initial draft"}, 0.0
        return {"role": "assistant", "content": "Revised with FusionBrain"}, 0.0

    monkeypatch.setattr(loop_mod, "call_llm_with_retry", fake_call)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")

    result, _usage, _trace = run_llm_loop(
        messages=[{"role": "user", "content": "Build the AIRI report"}],
        tools=registry,
        llm=FakeLLM(),
        drive_logs=tmp_path,
        emit_progress=lambda _text, **_meta: None,
        incoming_messages=queue.Queue(),
        task_id="direct-fence",
        drive_root=tmp_path,
    )

    assert result == "Revised with FusionBrain"
    assert len(calls) == 2
    assert any(
        row.get("role") == "user" and "FusionBrain" in str(row.get("content") or "")
        for row in calls[1]
    )
    assert direct_agent._accepting_owner_messages is False

def test_budget_rail_after_dispatch_is_terminal_without_provider_fallback(tmp_path, monkeypatch):
    from ouroboros.tools.registry import ToolRegistry
    from ouroboros.usage_accounting import AttemptRequest, BudgetExceeded, execute_physical_attempt

    class FakeLLM:
        def default_model(self):
            return "test-model"

    calls = {"primary": 0, "fallback": 0}

    def blocked(*_args, **_kwargs):
        calls["primary"] += 1
        raise BudgetExceeded(
            "root limit closed",
            limit_scope="root",
            root_task_id="budget-root",
        )

    def forbidden_fallback(**_kwargs):
        calls["fallback"] += 1
        raise AssertionError("budget rails must never enter model fallback")

    monkeypatch.setattr(loop_mod, "call_llm_with_retry", blocked)
    monkeypatch.setattr(loop_mod, "_run_cross_model_fallback_chain", forbidden_fallback)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    execute_physical_attempt(
        AttemptRequest(
            model="local/test",
            provider="local",
            drive_root=tmp_path,
            task_id="budget-task",
            root_task_id="budget-root",
        ),
        lambda: {"usage": {"prompt_tokens": 1, "completion_tokens": 1}},
    )
    events = queue.Queue()
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.root_task_id = "budget-root"
    registry._ctx.budget_drive_root = tmp_path

    result, usage, trace = run_llm_loop(
        messages=[{"role": "user", "content": "go"}],
        tools=registry,
        llm=FakeLLM(),
        drive_logs=tmp_path,
        emit_progress=lambda _text, **_meta: None,
        incoming_messages=queue.Queue(),
        event_queue=events,
        task_id="budget-task",
        drive_root=tmp_path,
    )

    assert calls == {"primary": 1, "fallback": 0}
    assert result.startswith("🚫 Resource limit reached")
    assert usage["reason_code"] == "budget_exhausted"
    assert usage["resource_limit"] == trace["resource_limit"]
    assert usage["resource_limit"]["status"] == "resource_limited"
    assert usage["resource_limit"]["resume_policy"] == "cancel_or_new_run"
    checkpoint = events.get_nowait()["data"]
    assert checkpoint["checkpoint_kind"] == "budget_scope_paused"
    assert checkpoint["scope"] == "root"
    root_fence = events.get_nowait()
    assert root_fence["type"] == "budget_root_fence"
    assert root_fence["root_task_id"] == "budget-root"

def test_run_llm_loop_narrates_reasoning_to_bubble_not_trace(tmp_path, monkeypatch):
    """Display-only contract: a pure tool-call round with no visible content narrates the
    provider's readable reasoning to the progress BUBBLE, but never records it in the durable
    trace (``reasoning_notes`` feeds build_trace_summary / task summaries) — so display-only
    reasoning cannot leak out of the display path."""
    from ouroboros.tools.registry import ToolRegistry

    messages = [{"role": "user", "content": "go"}]
    tool_round = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}],
        "reasoning": "Let me read the file before answering.",
    }
    calls = {"count": 0}
    emitted: list = []

    class FakeLLM:
        def default_model(self):
            return "test-model"

    def fake_call_llm_with_retry(_llm, request_messages, *_a, **_k):
        calls["count"] += 1
        if calls["count"] == 1:
            return dict(tool_round), 0.0
        return {"role": "assistant", "content": "final answer"}, 0.0

    def fake_handle_tool_calls(tool_calls, _tools, _dl, _tid, _ex, request_messages, _tr, _pg,
               *, fit_candidate=None, tool_schemas=None):
        request_messages.append({"role": "tool", "tool_call_id": tool_calls[0]["id"], "content": "file body"})
        return 0

    monkeypatch.setattr(loop_mod, "call_llm_with_retry", fake_call_llm_with_retry)
    monkeypatch.setattr(loop_mod, "handle_tool_calls", fake_handle_tool_calls)
    monkeypatch.setenv("OUROBOROS_REASONING_SUMMARY", "auto")

    result, _usage, trace = run_llm_loop(
        messages=messages,
        tools=ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path),
        llm=FakeLLM(),
        drive_logs=tmp_path,
        emit_progress=lambda text, **_meta: emitted.append(text),
        incoming_messages=queue.Queue(),
        task_id="narrate",
        drive_root=tmp_path,
    )

    assert result == "final answer"
    # the readable reasoning reached the display bubble...
    assert any("read the file before answering" in str(e) for e in emitted)
    # ...but did NOT leak into the durable trace (display-only).
    assert all("read the file before answering" not in str(n) for n in trace["reasoning_notes"])

def test_run_llm_loop_finalize_now_control_forces_best_effort_answer(tmp_path, monkeypatch):
    """A supervisor finalize_now control makes the loop extract one tool-less
    final answer and stamp the finalization_grace reason (typed best_effort
    gate downstream) — a deadline never returns emptiness."""
    from ouroboros.owner_mailbox import KIND_FINALIZE_NOW, write_owner_message
    from ouroboros.tool_policy import initial_tool_schemas
    from ouroboros.tools.registry import ToolRegistry

    write_owner_message(tmp_path, "deadline", task_id="graceful1", kind=KIND_FINALIZE_NOW)
    seen = {}

    class FakeLLM:
        def default_model(self):
            return "test-model"

    def fake_call_llm_with_retry(_llm, request_messages, _model, tools_arg, *_args, **_kwargs):
        seen["tools"] = tools_arg
        seen["messages"] = [dict(item) for item in request_messages]
        return {"role": "assistant", "content": "best effort summary"}, 0.0

    monkeypatch.setattr(loop_mod, "call_llm_with_retry", fake_call_llm_with_retry)

    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    # The round-1 envelope, taken where the loop takes it (`initial_tool_schemas`
    # is run_llm_loop's first step on the untouched registry): the same call
    # AFTER the run answers differently once the loop has bound its context
    # (credential-gated tools drop out), so the snapshot must precede the run.
    expected_envelope = initial_tool_schemas(registry)
    assert expected_envelope  # a real envelope: equality below is not empty == empty
    result, usage, _trace = run_llm_loop(
        messages=[{"role": "user", "content": "long job"}],
        tools=registry,
        llm=FakeLLM(),
        drive_logs=tmp_path,
        emit_progress=lambda _text, **_meta: None,
        incoming_messages=queue.Queue(),
        task_id="graceful1",
        drive_root=tmp_path,
    )

    assert result == "best effort summary"
    assert usage["reason_code"] == "finalization_grace"
    assert usage["execution_status"] == "failed"  # lifted to best_effort by the outcome gate
    assert usage["_best_effort_extracted"] is True  # typed fact: real model answer
    # The forced turn keeps the round's tool envelope so the provider prefix
    # stays a cache hit; "tool-less" is an instruction in text, not an empty
    # envelope. Pinned to the EXACT round-1 envelope snapshotted above — not
    # merely "something was passed".
    assert seen["tools"] == expected_envelope
    joined = json.dumps(seen["messages"], ensure_ascii=False)
    assert "[FINALIZE_NOW]" in joined

    # End-to-end: the derived outcome lands on the typed best_effort shelf.
    from ouroboros.outcomes import EXECUTION_BEST_EFFORT, derive_loop_outcome
    outcome = derive_loop_outcome(result, usage, {"tool_calls": [], "reasoning_notes": []})
    assert outcome["outcome_axes"]["execution"]["status"] == EXECUTION_BEST_EFFORT

def test_run_llm_loop_keeps_task_model_override_across_tool_rounds(tmp_path, monkeypatch):
    from ouroboros.tools.registry import ToolRegistry

    messages = [{"role": "user", "content": "inspect"}]
    seen_models: list[str] = []
    seen_use_local: list[bool] = []
    calls = {"count": 0}

    class FakeLLM:
        def default_model(self):
            return "default-model"

    def fake_call_llm_with_retry(_llm, request_messages, model, *_args, **kwargs):
        seen_models.append(model)
        seen_use_local.append(bool(kwargs.get("use_local")))
        calls["count"] += 1
        if calls["count"] == 1:
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }],
            }, 0.0
        return {"role": "assistant", "content": "done"}, 0.0

    def fake_handle_tool_calls(tool_calls, _tools, _drive_logs, _task_id, _executor, request_messages, _trace, _progress,
                               *, fit_candidate=None, tool_schemas=None):
        request_messages.append({"role": "tool", "tool_call_id": tool_calls[0]["id"], "content": "file"})
        return 0

    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.task_model_override = "subagent-light"
    registry._ctx.task_use_local_override = True
    monkeypatch.setattr(loop_mod, "call_llm_with_retry", fake_call_llm_with_retry)
    monkeypatch.setattr(loop_mod, "handle_tool_calls", fake_handle_tool_calls)

    result, _usage, _trace = run_llm_loop(
        messages=messages,
        tools=registry,
        llm=FakeLLM(),
        drive_logs=tmp_path,
        emit_progress=lambda _text, **_meta: None,
        incoming_messages=queue.Queue(),
        task_id="subagent1",
        drive_root=tmp_path,
    )

    assert result == "done"
    assert seen_models == ["subagent-light", "subagent-light"]
    assert seen_use_local == [True, True]

def test_run_llm_loop_enforces_swarm_force_plan_before_final(tmp_path, monkeypatch):
    from ouroboros.tools.registry import ToolRegistry

    messages = [{"role": "user", "content": "ship"}]
    calls = {"count": 0}
    seen_second_request = {}

    class FakeLLM:
        def default_model(self):
            return "test-model"

    def fake_call_llm_with_retry(_llm, request_messages, *_args, **_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return {"role": "assistant", "content": "premature final"}, 0.0
        if calls["count"] == 2:
            seen_second_request["messages"] = [dict(item) for item in request_messages]
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call-plan",
                    "type": "function",
                    "function": {"name": "plan_task", "arguments": "{}"},
                }],
            }, 0.0
        assert calls["count"] == 3, "the observed plan result needs only one completion selection"
        return finish("done after plan"), 0.0

    real_handle_tool_calls = loop_mod.handle_tool_calls

    def fake_handle_tool_calls(tool_calls, _tools, _drive_logs, _task_id, _executor, request_messages, trace, _progress,
                               *, fit_candidate=None, tool_schemas=None):
        if tool_calls[0]["function"]["name"] != "plan_task":
            return real_handle_tool_calls(tool_calls, _tools, _drive_logs, _task_id, _executor, request_messages, trace, _progress,
                                          fit_candidate=fit_candidate, tool_schemas=tool_schemas)
        from ouroboros.task_results import STATUS_RUNNING, record_plan_review_wave, write_task_result

        fingerprint = "a" * 64
        write_task_result(tmp_path, "task1", STATUS_RUNNING, result="running")
        record_plan_review_wave(tmp_path, "task1", {
            "schema_version": 2, "cycle_index": 1, "request_fingerprint": fingerprint,
            "spec": {"goal": "g"}, "spec_hash": "b" * 64, "findings": [], "aggregate": "GREEN",
            "closed": True, "dispositions": [], "paid": True,
        })
        trace["tool_calls"].append({
            "tool": tool_calls[0]["function"]["name"],
            "args": {},
            "result": "## Plan Review Results\n\nAGGREGATE: GREEN",
            "is_error": False,
            "plan_review_outcome": "GREEN",
            "plan_review_closed": True,
        })
        request_messages.append({"role": "tool", "tool_call_id": tool_calls[0]["id"], "content": "## Plan Review Results\n\nAGGREGATE: GREEN"})
        return 0

    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.task_metadata = {"force_plan": True, "force_plan_source": "swarm"}
    monkeypatch.setattr(loop_mod, "call_llm_with_retry", fake_call_llm_with_retry)
    monkeypatch.setattr(loop_mod, "handle_tool_calls", fake_handle_tool_calls)

    result, _usage, trace = run_llm_loop(
        messages=messages,
        tools=registry,
        llm=FakeLLM(),
        drive_logs=tmp_path,
        emit_progress=lambda _text, **_meta: None,
        incoming_messages=queue.Queue(),
        task_id="task1",
        drive_root=tmp_path,
    )

    assert result == "done after plan"
    assert calls["count"] == 3
    assert sum(row.get("role") == "assistant" and row.get("content") == "premature final"
               for row in seen_second_request["messages"]) == 1
    assert any("Call plan_task" in str(item.get("content") or "") for item in seen_second_request["messages"])
    assert trace["tool_calls"][0]["tool"] == "plan_task"

def test_force_plan_decision_does_not_treat_trace_marker_as_authority(tmp_path, monkeypatch):
    ctx = SimpleNamespace(
        task_metadata={"force_plan": True},
        task_id="root1",
        drive_root=tmp_path,
        budget_drive_root=str(tmp_path),
    )
    monkeypatch.setattr(loop_mod, "get_review_enforcement", lambda: "blocking")

    decision = loop_mod._force_plan_decision(ctx, {
        "tool_calls": [{
            "tool": "plan_task",
            "is_error": False,
            "plan_review_outcome": "GREEN",
            "plan_review_closed": True,
        }],
    })

    assert decision["allow"] is False
    assert decision["status"] == "absent"

def test_run_llm_loop_does_not_accept_failed_plan_task_for_swarm_force_plan(tmp_path, monkeypatch):
    from ouroboros.tools.registry import ToolRegistry

    messages = [{"role": "user", "content": "ship"}]
    calls = {"count": 0}

    class FakeLLM:
        def default_model(self):
            return "test-model"

    def fake_call_llm_with_retry(_llm, _request_messages, *_args, **_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return {"role": "assistant", "content": "premature final"}, 0.0
        if calls["count"] == 2:
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call-plan",
                    "type": "function",
                    "function": {"name": "plan_task", "arguments": "{}"},
                }],
            }, 0.0
        assert calls["count"] == 3, "the advisory plan gap is already observed"
        return finish("done despite unavailable plan"), 0.0

    real_handle_tool_calls = loop_mod.handle_tool_calls

    def fake_handle_tool_calls(tool_calls, _tools, _drive_logs, _task_id, _executor, request_messages, trace, _progress,
                               *, fit_candidate=None, tool_schemas=None):
        if tool_calls[0]["function"]["name"] != "plan_task":
            return real_handle_tool_calls(tool_calls, _tools, _drive_logs, _task_id, _executor, request_messages, trace, _progress,
                                          fit_candidate=fit_candidate, tool_schemas=tool_schemas)
        from ouroboros.task_results import STATUS_RUNNING, record_plan_review_attempt, write_task_result

        write_task_result(tmp_path, "task1", STATUS_RUNNING, result="running")
        record_plan_review_attempt(tmp_path, "task1", fingerprint="d" * 64)
        trace["tool_calls"].append({
            "tool": tool_calls[0]["function"]["name"],
            "args": {},
            "result": "ERROR: plan_task planning swarm failed closed: no planning subagent completed.",
            "is_error": False,
        })
        request_messages.append({"role": "tool", "tool_call_id": tool_calls[0]["id"], "content": "ERROR: plan_task planning swarm failed closed."})
        return 0

    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.task_metadata = {"force_plan": True, "force_plan_source": "swarm"}
    monkeypatch.setattr(loop_mod, "call_llm_with_retry", fake_call_llm_with_retry)
    monkeypatch.setattr(loop_mod, "handle_tool_calls", fake_handle_tool_calls)

    result, usage, trace = run_llm_loop(
        messages=messages,
        tools=registry,
        llm=FakeLLM(),
        drive_logs=tmp_path,
        emit_progress=lambda _text, **_meta: None,
        incoming_messages=queue.Queue(),
        task_id="task1",
        drive_root=tmp_path,
    )

    assert result.startswith("done despite unavailable plan")
    assert "advisory enforcement" in usage["terminal_host_notice"]
    assert calls["count"] == 3
    assert usage.get("reason_code") != "swarm_force_plan_not_called"
    assert trace["tool_calls"][0]["tool"] == "plan_task"

@pytest.mark.parametrize("explicit", [False, True])
def test_run_llm_loop_injects_subagent_handoff_before_final_text(tmp_path, monkeypatch, explicit):
    from ouroboros.task_results import STATUS_COMPLETED, write_task_result
    from ouroboros.tools.registry import ToolRegistry
    from tests._delivery_candidate_shared import write_confirmed_disposition_fixture

    write_task_result(
        tmp_path,
        "child1",
        STATUS_COMPLETED,
        parent_task_id="parent1",
        root_task_id="parent1",
        delegation_role="subagent",
        role="reviewer",
        result="child handoff",
    )
    messages = [{"role": "user", "content": "inspect"}]
    calls = {"count": 0}
    seen_second_request = {}
    progress = []

    class FakeLLM:
        def default_model(self):
            return "test-model"

    def fake_call_llm_with_retry(_llm, request_messages, *_args, **_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return (finish("premature final") if explicit else
                    {"role": "assistant", "content": "premature final"}), 0.0
        if calls["count"] == 2:
            write_confirmed_disposition_fixture(
                tmp_path,
                disposition="integrated",
                rationale="consumed in the final synthesis",
            )
        seen_second_request["messages"] = [dict(item) for item in request_messages]
        assert calls["count"] == 2
        return finish("final after handoff"), 0.0

    monkeypatch.setattr(loop_mod, "call_llm_with_retry", fake_call_llm_with_retry)

    result, _usage, trace = run_llm_loop(
        messages=messages,
        tools=ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path),
        llm=FakeLLM(),
        drive_logs=tmp_path,
        emit_progress=lambda text, **_meta: progress.append(text),
        incoming_messages=queue.Queue(),
        task_id="parent1",
        drive_root=tmp_path,
    )

    assert result == "final after handoff"
    assert calls["count"] == 2
    assert any("Subagent handoff status refreshed" in item for item in progress)
    assert any("Subagent handoff status refreshed" in item for item in trace["reasoning_notes"])
    second_text = "\n".join(str(item.get("content") or "") for item in seen_second_request["messages"])
    # C3.4: the parent now ABSORBS the child's FULL authored result before
    # finalizing (not just a 240-char preview), with a durable get_task_result pointer.
    assert "[SUBAGENT_RESULTS" in second_text
    assert "child child1" in second_text
    assert "child handoff" in second_text
    assert "get_task_result" in second_text
    assert sum(row.get("role") == "assistant" and row.get("content") == "premature final"
               for row in seen_second_request["messages"]) == 1
    assert ("Selected completion was held" if explicit else "No completion selection was made") in second_text

def test_run_llm_loop_appends_orphan_note_when_finalizing_with_unhandled_child(tmp_path, monkeypatch):
    """D#7 / P5: the subagent handoff reminder fires once per CHANGE (not every round, not
    suppressed by parsing the final prose). When the agent finalizes with a child still
    unhandled (not absorbed, not discarded/cancelled), the answer carries a LOUD orphan note
    instead of silently dropping the child."""
    from ouroboros.task_results import STATUS_RUNNING, write_task_result
    from ouroboros.tools.registry import ToolRegistry

    # This regression isolates the bounded handoff/orphan-note path; acceptance
    # quiescence has its own tests and would correctly wait for the running child.
    monkeypatch.setattr(loop_mod, "get_task_review_mode", lambda: "off")

    write_task_result(
        tmp_path,
        "child1",
        STATUS_RUNNING,
        parent_task_id="parent1",
        root_task_id="parent1",
        delegation_role="subagent",
        role="reviewer",
        result="still collecting evidence",
    )
    messages = [{"role": "user", "content": "inspect"}]
    calls = {"count": 0}
    progress = []

    class FakeLLM:
        def default_model(self):
            return "test-model"

    def fake_call_llm_with_retry(_llm, _request_messages, *_args, **_kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return {"role": "assistant", "content": "child1 is still running; I will finalize now."}, 0.0
        if calls["count"] < 4:
            return finish("Best effort: child1 is still running."), 0.0
        assert calls["count"] == 4
        return finish("Best effort: child1 is still running.", action="stop",
                      rationale="The child is still collecting evidence; this task is unfinished."), 0.0

    monkeypatch.setattr(loop_mod, "call_llm_with_retry", fake_call_llm_with_retry)

    result, _usage, trace = run_llm_loop(
        messages=messages,
        tools=ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path),
        llm=FakeLLM(),
        drive_logs=tmp_path,
        emit_progress=lambda text, **_meta: progress.append(text),
        incoming_messages=queue.Queue(),
        task_id="parent1",
        drive_root=tmp_path,
    )

    # Handoff and repeated finish remain held; only the explicit unfinished stop exits.
    assert calls["count"] == 4
    assert sum(1 for item in progress if "Subagent handoff status refreshed" in item) == 1
    assert trace["task_completion"]["action"] == "stop"
    assert not _usage.get("_best_effort_extracted")
    # The selected partial answer is preserved beside the host-authored child notice.
    assert result.startswith("Best effort: child1 is still running.")
    assert "child1" in result and "NOTE: finalized" in _usage["terminal_host_notice"]

def test_run_llm_loop_keeps_child_absorption_open_until_explicit_stop(tmp_path, monkeypatch):
    from ouroboros.task_results import STATUS_RUNNING, write_task_result
    from ouroboros.tools.registry import ToolRegistry

    write_task_result(
        tmp_path,
        "child1",
        STATUS_RUNNING,
        parent_task_id="parent1",
        root_task_id="parent1",
        delegation_role="subagent",
        role="reviewer",
        result="still collecting evidence",
    )
    messages = [{"role": "user", "content": "inspect"}]
    calls = {"count": 0}
    seen_requests = []
    progress = []
    tools = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    tools._ctx.task_contract = {"delegation_budget": {"may_delegate": True, "may_fan_out": True}}

    class FakeLLM:
        def default_model(self):
            return "test-model"

    def fake_call_llm_with_retry(_llm, request_messages, *_args, **_kwargs):
        calls["count"] += 1
        seen_requests.append([dict(m) for m in request_messages])
        if calls["count"] == 1:
            return {"role": "assistant", "content": "answer 1"}, 0.0
        if calls["count"] < 4:
            return finish(f"answer {calls['count']}"), 0.0
        assert calls["count"] == 4
        return finish("Stopped with child1 still running.", action="stop",
                      rationale="The child's evidence is still missing."), 0.0

    monkeypatch.setattr(loop_mod, "call_llm_with_retry", fake_call_llm_with_retry)
    monkeypatch.setattr(loop_mod, "_forced_final_answer",
                        lambda *_a, **_kw: pytest.fail("reminder count cannot buy a forced final"))

    result, usage, trace = run_llm_loop(
        messages=messages,
        tools=tools,
        llm=FakeLLM(),
        drive_logs=tmp_path,
        emit_progress=lambda text, **_meta: progress.append(text),
        incoming_messages=queue.Queue(),
        task_id="parent1",
        drive_root=tmp_path,
    )

    from ouroboros.outcomes import derive_loop_outcome
    from ouroboros.task_results import load_task_result

    assert result == "Stopped with child1 still running."
    assert trace["task_completion"]["action"] == "stop"
    assert derive_loop_outcome(result, usage, trace)["outcome_axes"]["objective"]["reason"] == "author_stop"
    assert not usage.get("_best_effort_extracted")
    assert "child task(s) not explicitly absorbed" in usage["terminal_host_notice"]
    assert calls["count"] == 4
    assert load_task_result(tmp_path, "child1")["status"] == "running"
    absorption_round_delta = seen_requests[2][len(seen_requests[1]):]
    delta_text = "\n".join(str(m.get("content") or "") for m in absorption_round_delta)
    assert "[CHILD_ABSORPTION_REQUIRED]" in delta_text and "child1 [running]" in delta_text
    assert "[DELIVERY_FINALIZATION_CONTROL]" not in delta_text
    # The next attempted finish receives the same actual obligation, never a forced terminal.
    assert "[CHILD_ABSORPTION_REQUIRED]" in str(seen_requests[3])
    assert "[FINALIZE_WITH_UNABSORBED_CHILDREN]" not in str(seen_requests)

def test_run_llm_loop_does_not_include_current_subagent_in_own_handoff(tmp_path, monkeypatch):
    from ouroboros.task_results import STATUS_RUNNING, write_task_result
    from ouroboros.tools.registry import ToolRegistry

    write_task_result(
        tmp_path,
        "child1",
        STATUS_RUNNING,
        parent_task_id="parent1",
        root_task_id="parent1",
        delegation_role="subagent",
        role="reviewer",
        result="my own running mirror",
    )
    messages = [{"role": "user", "content": "inspect"}]
    calls = {"count": 0}
    progress = []
    tools = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    tools._ctx.task_metadata = {
        "parent_task_id": "parent1",
        "root_task_id": "parent1",
        "delegation_role": "subagent",
    }

    class FakeLLM:
        def default_model(self):
            return "test-model"

    def fake_call_llm_with_retry(_llm, _request_messages, *_args, **_kwargs):
        calls["count"] += 1
        return {"role": "assistant", "content": "subagent final"}, 0.0

    monkeypatch.setattr(loop_mod, "call_llm_with_retry", fake_call_llm_with_retry)

    result, _usage, trace = run_llm_loop(
        messages=messages,
        tools=tools,
        llm=FakeLLM(),
        drive_logs=tmp_path,
        emit_progress=lambda text, **_meta: progress.append(text),
        incoming_messages=queue.Queue(),
        task_id="child1",
        drive_root=tmp_path,
    )

    assert result == "subagent final"
    assert calls["count"] == 1
    assert not any("Subagent handoff status refreshed" in item for item in progress)
    assert not any("Subagent handoff status refreshed" in item for item in trace["reasoning_notes"])
