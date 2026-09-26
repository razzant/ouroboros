"""Presence completion traverses the real tool batch and shared finalization gates."""

import hashlib
import json
import queue
import threading
from types import SimpleNamespace

import pytest

import ouroboros.loop as loop
from ouroboros.outcomes import derive_loop_outcome
from ouroboros.presence_authority import presence_ceiling_payload
from ouroboros.presence_runner import build_presence_result_event
from ouroboros.tools.registry import ToolRegistry
from tests.test_presence_runner import _admission


def _call(outcome="message", message="Ready"):
    return {"role": "assistant", "content": None, "tool_calls": [{
        "id": "finish", "type": "function", "function": {
            "name": "presence_finish", "arguments": json.dumps({"outcome": outcome, "message": message}),
        },
    }]}


@pytest.fixture
def turn(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.task_contract = {"capability_ceiling": presence_ceiling_payload(_admission().capability_ceiling)}
    registry._ctx.is_direct_chat = True
    calls = []

    def run(responses):
        iterator = iter(responses)

        def respond(_llm, messages, *_a, **_k):
            calls.append([dict(row) for row in messages])
            value = next(iterator)
            return (value() if callable(value) else value), 0.0

        monkeypatch.setattr(loop, "call_llm_with_retry", respond)
        return loop.run_llm_loop(
            [{"role": "user", "content": "Please help"}], registry,
            SimpleNamespace(default_model=lambda: "test-model"), tmp_path / "logs",
            lambda *_a, **_k: None, queue.Queue(), task_id="parent1", drive_root=tmp_path,
        )

    return registry, calls, run


@pytest.mark.parametrize("outcome,message", [
    ("message", "Ready"), ("deferred", "Working on it"),
    ("silent", ""), ("tool_delivered", ""),
    ("message", '{"delivery_control":"keep"}'),
])
def test_explicit_finish_uses_one_model_round_and_real_tool_batch(turn, outcome, message):
    registry, calls, run = turn
    registry._ctx._swarm_handoff_attempt = {"status": "scheduled", "task_id": "later-work"}
    completed = []
    registry.override_handler("chat_history", lambda _ctx, **_kw: completed.append("sibling") or "History read")
    response = _call(outcome, message)
    response["tool_calls"].append({"id": "sibling", "type": "function", "function": {
        "name": "chat_history", "arguments": "{}",
    }})
    text, usage, trace = run([response])
    assert len(calls) == 1
    assert completed == ["sibling"]
    assert text == message
    assert registry._ctx._presence_completion_accepted is True
    assert usage["presence_completion_outcome"] == outcome
    assert usage["terminal_origin"] == "model_final"
    assert len(trace["tool_calls"]) == 2
    result = build_presence_result_event({"id": "parent1"}, text, registry._ctx, terminal_origin=usage.get("terminal_origin", ""))
    assert result["outcome"] == outcome
    assert result["text"] == (message if outcome in {"message", "deferred"} else "")
    assert derive_loop_outcome(text, usage, trace)["outcome_axes"]["execution"]["status"] == "ok"


@pytest.mark.parametrize("outcome", ["message", "deferred"])
def test_omitted_message_keeps_normal_final_round(turn, outcome):
    registry, calls, run = turn
    registry._ctx._swarm_handoff_attempt = {"status": "scheduled", "task_id": "later-work"}
    text, usage, _trace = run([_call(outcome, ""), {"content": "Authored final"}])
    assert len(calls) == 2
    assert text == "Authored final"
    assert usage["presence_completion_outcome"] == outcome


def test_review_hold_drops_old_outcome_and_uses_replacement(turn, monkeypatch):
    registry, calls, run = turn
    reviews = []

    def review(**kwargs):
        reviews.append(kwargs["content"])
        return len(reviews) == 1

    monkeypatch.setattr(loop, "_run_task_acceptance_review_once", review)
    text, usage, _trace = run([_call("tool_delivered", "Old answer"), {"content": "Revised answer"}])
    assert len(calls) == 2 and reviews == ["Old answer", "Revised answer"]
    assert registry._ctx._presence_completion is None
    assert "presence_completion_outcome" not in usage
    result = build_presence_result_event({"id": "parent1"}, text, registry._ctx, terminal_origin=usage.get("terminal_origin", ""))
    assert (result["outcome"], result["text"]) == ("message", "Revised answer")


def test_new_explicit_finish_after_hold_replaces_old_nonempty_candidate(turn, monkeypatch):
    registry, calls, run = turn
    reviews = []

    def review(**kwargs):
        reviews.append(kwargs["content"])
        return len(reviews) == 1

    monkeypatch.setattr(loop, "_run_task_acceptance_review_once", review)
    text, usage, trace = run([_call("message", "Old answer"), _call("silent", "")])
    assert len(calls) == 2 and reviews == ["Old answer", ""]
    assert text == "" and usage["presence_completion_outcome"] == "silent"
    assert trace["delivery_candidate"]["content_sha256"] == hashlib.sha256(b"").hexdigest()


def test_ordinary_task_still_requires_its_normal_model_final(turn):
    registry, calls, run = turn
    registry._ctx.task_contract = {}
    text, usage, _trace = run([_call("silent", ""), {"content": "Ordinary final"}])
    assert len(calls) == 2 and text == "Ordinary final"
    assert "presence_completion_outcome" not in usage


@pytest.mark.parametrize("late", [False, True])
def test_owner_followup_invalidates_finish_before_or_during_final_gate(turn, tmp_path, monkeypatch, late):
    from ouroboros.owner_mailbox import write_owner_message

    registry, calls, run = turn
    registry._ctx.owner_message_admission_lock = threading.Lock()
    registry._ctx.owner_message_admission_agent = SimpleNamespace(
        _busy=True, _current_task_id="parent1", _accepting_owner_messages=True,
    )

    def followup():
        write_owner_message(tmp_path, "Include the new detail", "parent1", msg_id="revision")

    if late:
        reviews = []

        def review(**_kw):
            if not reviews:
                followup()
            reviews.append(1)
            return False

        monkeypatch.setattr(loop, "_run_task_acceptance_review_once", review)
    else:
        from ouroboros.tools.presence import _finish_presence

        def finish(ctx, **kwargs):
            result = _finish_presence(ctx, **kwargs)
            followup()
            return result

        registry.override_handler("presence_finish", finish)
    text, usage, _trace = run([_call("silent", "Old"), {"content": "With the new detail"}])
    assert len(calls) == 2 and text == "With the new detail"
    assert any("new detail" in str(row.get("content")) for row in calls[-1])
    assert "presence_completion_outcome" not in usage
    assert build_presence_result_event({"id": "parent1"}, text, registry._ctx, terminal_origin=usage.get("terminal_origin", ""))["outcome"] == "message"


@pytest.mark.parametrize("reason", ["cancel", "budget"])
def test_control_or_budget_tail_precedes_pending_finish(turn, tmp_path, monkeypatch, reason):
    from ouroboros.tools.presence import _finish_presence
    from ouroboros.usage_accounting import BudgetExceeded

    registry, calls, run = turn
    tail = []
    if reason == "cancel":
        from ouroboros.owner_mailbox import KIND_FINALIZE_NOW, write_owner_message
        from supervisor.owner_stop import REASON_OWNER_STOPPED_DIRECT_TURN

        def finish(ctx, **kwargs):
            result = _finish_presence(ctx, **kwargs)
            write_owner_message(tmp_path, REASON_OWNER_STOPPED_DIRECT_TURN, "parent1", kind=KIND_FINALIZE_NOW)
            return result

        registry.override_handler("presence_finish", finish)
    else:
        def budget(*_a, **_kw):
            tail.append("budget")
            raise BudgetExceeded("test limit", limit_scope="root", root_task_id="parent1")

        monkeypatch.setattr(loop, "_finish_tool_round_budget", budget)
    text, usage, _trace = run([_call("silent", "Old answer"), {"content": "Controlled wrap-up"}])
    assert "presence_completion_outcome" not in usage
    assert registry._ctx._presence_completion_accepted is False
    assert usage["execution_status"] == "failed"
    result = build_presence_result_event({"id": "parent1"}, text, registry._ctx, terminal_origin=usage.get("terminal_origin", ""))
    assert result["outcome"] == "silent" and result["text"] == ""
    assert usage["terminal_origin"] == "host_notice"
    if reason == "budget":
        assert tail == ["budget"] and len(calls) == 1
    else:
        assert len(calls) == 1 and registry._ctx._skip_post_task_synthesis is True


def test_ordinary_empty_and_failed_silent_outcomes_remain_failed():
    for usage in ({}, {"presence_completion_outcome": "silent", "execution_status": "failed"},
                  {"presence_completion_outcome": "tool_delivered", "execution_status": "infra_failed"}):
        outcome = derive_loop_outcome("", usage, {"tool_calls": []})
        assert outcome["outcome_axes"]["execution"]["status"] in {"failed", "infra_failed"}


_DECLARED = json.dumps({"delivery_control": "replace", "full_answer": "Best available; child1 still running",
                        "presence_finish": {"outcome": "message", "message": "Here is what I have so far."}})


@pytest.mark.parametrize("forced,outcome,spoken", [
    # Owner Q4: the forced answer is the internal record; undeclared prose never becomes speech.
    ("Best available; child1 still running", "silent", ""),
    (_DECLARED, "message", "Here is what I have so far."),
])
def test_pending_children_still_require_absorption(turn, tmp_path, forced, outcome, spoken):
    from ouroboros.task_results import write_task_result, STATUS_RUNNING

    _registry, calls, run = turn
    # A real turn carries its Presence metadata; the ceiling alone does not arm the protocol.
    _registry._ctx.task_metadata = {"presence": {"binding_id": "a" * 32, "event": {"conversation_key": "k"}}}
    write_task_result(tmp_path, "child1", STATUS_RUNNING, parent_task_id="parent1",
                      root_task_id="parent1", delegation_role="subagent", role="reviewer", result="Still running")
    text, usage, _trace = run([
        _call("silent", "Premature"),
        {"content": '{"delivery_control":"keep"}'},
        {"content": "Best available"}, {"content": forced},
    ])
    assert len(calls) > 1
    assert usage["reason_code"] == "children_unabsorbed"
    assert "presence_completion_outcome" not in usage
    assert text == "Best available; child1 still running"  # the internal record keeps the child facts
    assert "[PRESENCE_DELIVERY]" in str(calls[-1][-1]["content"])
    assert "name the unabsorbed" not in str(calls[-1][-1]["content"])
    task = {"id": "parent1"}
    result = build_presence_result_event(task, text, _registry._ctx, terminal_origin=usage.get("terminal_origin", ""))
    assert (result["outcome"], result["text"]) == (outcome, spoken)
    assert task["metadata"]["presence_declaration"]["status"] == ("declared" if spoken else "missing")
