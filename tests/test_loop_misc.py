"""Loop miscellaneous regressions.

Consolidated from former ``test_loop_incoming_messages.py`` (image
payload preservation) and ``test_loop_skill_finalization.py``
(self-authored skill finalization gate). Both modules exercise
narrow corners of ``ouroboros.loop`` that did not justify standalone
files after Phase 5.

Kept here as one module so future loop micro-regressions have a
natural home instead of producing yet another single-test file.
"""
from __future__ import annotations

import json
import queue
from types import SimpleNamespace

import ouroboros.loop as loop_mod
from ouroboros.loop import (
    _drain_incoming_messages,
    _maybe_inject_self_check,
    _maybe_inject_time_budget_milestone,
    _run_task_acceptance_review_once,
    _skill_finalization_message,
    _skill_names_touched_by_trace,
    _task_acceptance_eligible,
    run_llm_loop,
)
from ouroboros.skill_loader import (
    SkillReviewState,
    compute_content_hash,
    save_enabled,
    save_review_state,
)


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

    monkeypatch.setattr(loop_mod, "utc_now", lambda: datetime(2026, 6, 10, 5, 1, tzinfo=timezone.utc))

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


def test_task_acceptance_auto_is_llm_first_not_host_enforced(monkeypatch):
    trace = {
        "tool_calls": [
            {"tool": "write_file", "args": {"path": "x.py"}},
            {"tool": "run_command", "args": {"cmd": ["pytest"]}},
        ]
    }

    # auto stays LLM-first (host never enforces), regardless of effects.
    assert _task_acceptance_eligible("auto", trace, True)[0] is False
    # required is effect-gated: this trace has a workspace write -> eligible.
    assert _task_acceptance_eligible("required", trace, True)[0] is True
    assert _task_acceptance_eligible("off", trace, True)[0] is False

    monkeypatch.setattr(loop_mod, "get_task_review_mode", lambda: "required")
    ctx = SimpleNamespace(_task_acceptance_reviewed=False, is_direct_chat=False, drive_root="/tmp")
    reviewed_trace = {
        "tool_calls": [{"tool": "task_acceptance_review", "args": {}}],
        "review_runs": [{"request": {"surface": "task_acceptance"}, "aggregate_signal": "PASS"}],
    }
    assert _run_task_acceptance_review_once(
        tools=SimpleNamespace(_ctx=ctx),
        content="done",
        task_id="task1",
        task_type="task",
        llm_trace=reviewed_trace,
        drive_root=None,
        messages=[{"role": "system", "content": ""}, {"role": "user", "content": "goal"}],
        emit_progress=lambda _msg: None,
    ) is False
    assert ctx._task_acceptance_reviewed is True
    assert reviewed_trace["review_decision"]["trigger"] == "agent_called_tool_result"


# ---------------------------------------------------------------------------
# Skill finalization gate (self-authored skills must reach ready+enabled
# before the loop accepts a final text response)
# ---------------------------------------------------------------------------


def _write_self_authored_skill(drive_root, name: str = "alpha"):
    skill_dir = drive_root / "skills" / "external" / name
    state_dir = drive_root / "state" / "skills" / name
    skill_dir.mkdir(parents=True)
    state_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: alpha\ntype: instruction\nversion: 0.1.0\n---\nbody\n",
        encoding="utf-8",
    )
    marker = {
        "schema_version": 1,
        "origin": "self_authored",
        "task_id": "task-1",
        "created_at": "2026-05-07T00:00:00+00:00",
    }
    (skill_dir / ".self_authored.json").write_text(json.dumps(marker), encoding="utf-8")
    (state_dir / "self_authored.json").write_text(json.dumps(marker), encoding="utf-8")
    return skill_dir


def test_skill_names_touched_by_trace_detects_data_skill_edits():
    trace = {
        "tool_calls": [
            {"tool": "write_file", "args": {"path": "skills/external/alpha/plugin.py"}},
            {"tool": "edit_text", "args": {"path": "data/skills/external/beta/SKILL.md"}},
            {"tool": "claude_code_edit", "args": {"cwd": "skills/external/gamma"}},
            {"tool": "write_file", "args": {"path": "SKILL.md", "bucket": "external", "skill_name": "delta"}},
        ]
    }

    assert _skill_names_touched_by_trace(trace) == ["alpha", "beta", "gamma", "delta"]


def test_skill_finalization_message_blocks_unreviewed_self_authored_skill(tmp_path):
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    _write_self_authored_skill(drive_root)
    trace = {"tool_calls": [{"tool": "write_file", "args": {"path": "skills/external/alpha/SKILL.md"}}]}

    message = _skill_finalization_message(drive_root, trace)

    assert "SKILL_NOT_FINALIZED" in message
    assert "alpha" in message


def test_skill_finalization_message_allows_ready_self_authored_skill(tmp_path):
    drive_root = tmp_path / "drive"
    drive_root.mkdir()
    skill_dir = _write_self_authored_skill(drive_root)
    content_hash = compute_content_hash(skill_dir)
    save_review_state(drive_root, "alpha", SkillReviewState(status="pass", content_hash=content_hash))
    save_enabled(drive_root, "alpha", True)
    trace = {"tool_calls": [{"tool": "write_file", "args": {"path": "skills/external/alpha/SKILL.md"}}]}

    assert _skill_finalization_message(drive_root, trace) == ""


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

    def fake_handle_tool_calls(tool_calls, _tools, _drive_logs, _task_id, _executor, request_messages, _trace, _progress):
        request_messages.append({"role": "tool", "tool_call_id": tool_calls[0]["id"], "content": "file"})
        return 0

    monkeypatch.setattr(loop_mod, "call_llm_with_retry", fake_call_llm_with_retry)
    monkeypatch.setattr(loop_mod, "handle_tool_calls", fake_handle_tool_calls)

    result, _usage, _trace = run_llm_loop(
        messages=messages,
        tools=ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path),
        llm=FakeLLM(),
        drive_logs=tmp_path,
        emit_progress=lambda _text: None,
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


def test_run_llm_loop_finalize_now_control_forces_best_effort_answer(tmp_path, monkeypatch):
    """A supervisor finalize_now control makes the loop extract one tool-less
    final answer and stamp the finalization_grace reason (typed best_effort
    gate downstream) — a deadline never returns emptiness."""
    from ouroboros.owner_mailbox import KIND_FINALIZE_NOW, write_owner_message
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

    result, usage, _trace = run_llm_loop(
        messages=[{"role": "user", "content": "long job"}],
        tools=ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path),
        llm=FakeLLM(),
        drive_logs=tmp_path,
        emit_progress=lambda _text: None,
        incoming_messages=queue.Queue(),
        task_id="graceful1",
        drive_root=tmp_path,
    )

    assert result == "best effort summary"
    assert usage["reason_code"] == "finalization_grace"
    assert usage["execution_status"] == "failed"  # lifted to best_effort by the outcome gate
    assert usage["_best_effort_extracted"] is True  # typed fact: real model answer
    assert seen["tools"] is None  # tool-less final extraction
    joined = json.dumps(seen["messages"], ensure_ascii=False)
    assert "[FINALIZE_NOW]" in joined

    # End-to-end: the derived outcome lands on the typed best_effort shelf.
    from ouroboros.outcomes import EXECUTION_BEST_EFFORT, derive_loop_outcome
    outcome = derive_loop_outcome(result, usage, {"tool_calls": [], "reasoning_notes": []})
    assert outcome["outcome_axes"]["execution"]["status"] == EXECUTION_BEST_EFFORT


def test_run_llm_loop_injects_busy_written_owner_message(tmp_path, monkeypatch):
    """End-to-end coupling for the busy-branch fix: a plain owner message
    persisted to the live task's mailbox (server.py) is drained into the LLM
    request at the next round boundary — using the same (drive_root, task_id)
    contract the busy-branch writes under."""
    from ouroboros.owner_mailbox import write_owner_message
    from ouroboros.tools.registry import ToolRegistry

    assert write_owner_message(tmp_path, "ты тут?", task_id="busytask") is True
    seen = {}

    class FakeLLM:
        def default_model(self):
            return "test-model"

    def fake_call_llm_with_retry(_llm, request_messages, _model, tools_arg, *_args, **_kwargs):
        seen["messages"] = [dict(item) for item in request_messages]
        return {"role": "assistant", "content": "yes, here"}, 0.0

    monkeypatch.setattr(loop_mod, "call_llm_with_retry", fake_call_llm_with_retry)

    run_llm_loop(
        messages=[{"role": "user", "content": "long job"}],
        tools=ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path),
        llm=FakeLLM(),
        drive_logs=tmp_path,
        emit_progress=lambda _text: None,
        incoming_messages=queue.Queue(),
        task_id="busytask",
        drive_root=tmp_path,
    )

    joined = json.dumps(seen["messages"], ensure_ascii=False)
    assert "ты тут?" in joined  # the busy-written message reached the model


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

    def fake_handle_tool_calls(tool_calls, _tools, _drive_logs, _task_id, _executor, request_messages, _trace, _progress):
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
        emit_progress=lambda _text: None,
        incoming_messages=queue.Queue(),
        task_id="subagent1",
        drive_root=tmp_path,
    )

    assert result == "done"
    assert seen_models == ["subagent-light", "subagent-light"]
    assert seen_use_local == [True, True]


def test_run_llm_loop_enforces_consilium_force_plan_before_final(tmp_path, monkeypatch):
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
        return {"role": "assistant", "content": "done after plan"}, 0.0

    def fake_handle_tool_calls(tool_calls, _tools, _drive_logs, _task_id, _executor, request_messages, trace, _progress):
        trace["tool_calls"].append({
            "tool": tool_calls[0]["function"]["name"],
            "args": {},
            "result": "## Plan Review Results\n\nAGGREGATE: GREEN",
            "is_error": False,
            # v6.26.0: the force-plan gate reads this structured flag (captured
            # from the FULL tool result), not a substring of the 700-char preview.
            "plan_review_aggregate": True,
        })
        request_messages.append({"role": "tool", "tool_call_id": tool_calls[0]["id"], "content": "## Plan Review Results\n\nAGGREGATE: GREEN"})
        return 0

    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.task_metadata = {"force_plan": True, "force_plan_source": "consilium"}
    monkeypatch.setattr(loop_mod, "call_llm_with_retry", fake_call_llm_with_retry)
    monkeypatch.setattr(loop_mod, "handle_tool_calls", fake_handle_tool_calls)

    result, _usage, trace = run_llm_loop(
        messages=messages,
        tools=registry,
        llm=FakeLLM(),
        drive_logs=tmp_path,
        emit_progress=lambda _text: None,
        incoming_messages=queue.Queue(),
        task_id="task1",
        drive_root=tmp_path,
    )

    assert result == "done after plan"
    assert calls["count"] == 3
    assert any("plan_task is required" in str(item.get("content") or "") for item in seen_second_request["messages"])
    assert trace["tool_calls"][0]["tool"] == "plan_task"


def test_run_llm_loop_does_not_accept_failed_plan_task_for_consilium_force_plan(tmp_path, monkeypatch):
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
        return {"role": "assistant", "content": "still finalizing without a valid plan"}, 0.0

    def fake_handle_tool_calls(tool_calls, _tools, _drive_logs, _task_id, _executor, request_messages, trace, _progress):
        trace["tool_calls"].append({
            "tool": tool_calls[0]["function"]["name"],
            "args": {},
            "result": "ERROR: plan_task planning swarm failed closed: no planning subagent completed.",
            "is_error": False,
        })
        request_messages.append({"role": "tool", "tool_call_id": tool_calls[0]["id"], "content": "ERROR: plan_task planning swarm failed closed."})
        return 0

    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.task_metadata = {"force_plan": True, "force_plan_source": "consilium"}
    monkeypatch.setattr(loop_mod, "call_llm_with_retry", fake_call_llm_with_retry)
    monkeypatch.setattr(loop_mod, "handle_tool_calls", fake_handle_tool_calls)

    result, usage, trace = run_llm_loop(
        messages=messages,
        tools=registry,
        llm=FakeLLM(),
        drive_logs=tmp_path,
        emit_progress=lambda _text: None,
        incoming_messages=queue.Queue(),
        task_id="task1",
        drive_root=tmp_path,
    )

    assert result.startswith("⚠️ CONSILIUM_FORCE_PLAN_BLOCKED")
    assert calls["count"] == 4
    assert usage["reason_code"] == "consilium_force_plan_not_called"
    assert trace["tool_calls"][0]["tool"] == "plan_task"


def test_run_llm_loop_injects_subagent_handoff_before_final_text(tmp_path, monkeypatch):
    from ouroboros.task_results import STATUS_COMPLETED, write_task_result
    from ouroboros.tools.registry import ToolRegistry

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
            return {"role": "assistant", "content": "premature final"}, 0.0
        seen_second_request["messages"] = [dict(item) for item in request_messages]
        return {"role": "assistant", "content": "final after handoff"}, 0.0

    monkeypatch.setattr(loop_mod, "call_llm_with_retry", fake_call_llm_with_retry)

    result, _usage, trace = run_llm_loop(
        messages=messages,
        tools=ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path),
        llm=FakeLLM(),
        drive_logs=tmp_path,
        emit_progress=progress.append,
        incoming_messages=queue.Queue(),
        task_id="parent1",
        drive_root=tmp_path,
    )

    assert result == "final after handoff"
    assert calls["count"] == 2
    assert any("Subagent handoff status refreshed" in item for item in progress)
    assert any("Subagent handoff status refreshed" in item for item in trace["reasoning_notes"])
    second_text = "\n".join(str(item.get("content") or "") for item in seen_second_request["messages"])
    assert "[SUBAGENT_HANDOFF_STATUS]" in second_text
    assert "result_available" in second_text
    assert "child handoff" in second_text
    assert "Use get_task_result" in second_text


def test_run_llm_loop_reinjects_incomplete_subagent_handoff_until_final_acknowledges_status(tmp_path, monkeypatch):
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
    progress = []

    class FakeLLM:
        def default_model(self):
            return "test-model"

    def fake_call_llm_with_retry(_llm, _request_messages, *_args, **_kwargs):
        calls["count"] += 1
        if calls["count"] <= 2:
            return {"role": "assistant", "content": "done"}, 0.0
        return {"role": "assistant", "content": "child1 is still running and incomplete; final answer will wait."}, 0.0

    monkeypatch.setattr(loop_mod, "call_llm_with_retry", fake_call_llm_with_retry)

    result, _usage, trace = run_llm_loop(
        messages=messages,
        tools=ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path),
        llm=FakeLLM(),
        drive_logs=tmp_path,
        emit_progress=progress.append,
        incoming_messages=queue.Queue(),
        task_id="parent1",
        drive_root=tmp_path,
    )

    assert result == "child1 is still running and incomplete; final answer will wait."
    assert calls["count"] == 3
    assert sum(1 for item in progress if "Subagent handoff status refreshed" in item) == 2
    assert sum(1 for item in trace["reasoning_notes"] if "Subagent handoff status refreshed" in item) == 2


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
        emit_progress=progress.append,
        incoming_messages=queue.Queue(),
        task_id="child1",
        drive_root=tmp_path,
    )

    assert result == "subagent final"
    assert calls["count"] == 1
    assert not any("Subagent handoff status refreshed" in item for item in progress)
    assert not any("Subagent handoff status refreshed" in item for item in trace["reasoning_notes"])
