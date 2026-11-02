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
from tests.presence_legacy_helpers import legacy_registry as ToolRegistry
from tests.test_presence_runner import _admission
from tests._store_shared import read_result


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
    text, usage, trace = run([response, _call(outcome, message)])
    assert len(calls) == 2
    assert completed == ["sibling"]
    assert text == message
    assert registry._ctx._presence_completion_accepted is True
    assert usage["presence_completion_outcome"] == outcome
    assert usage["terminal_origin"] == "model_final"
    assert len(trace["tool_calls"]) == 3
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


@pytest.mark.parametrize("outcome,reply_later", [
    ("message", False), ("message", True), ("deferred", False), ("deferred", True),
    ("silent", False), ("tool_delivered", False),
])
def test_stop_records_selected_bytes_without_an_extra_final_generation(turn, monkeypatch, outcome, reply_later):
    registry, calls, run = turn
    registry._ctx._swarm_handoff_attempt = {"status": "scheduled", "task_id": "later-work"}
    reserves_reply = reply_later and outcome in {"message", "deferred"}
    text = "The requested work remains unfinished." if outcome in {"message", "deferred"} else ""
    response = _call(outcome, "" if reserves_reply else text)
    arguments = json.loads(response["tool_calls"][0]["function"]["arguments"])
    arguments.update(action="stop", rationale="The requested work remains unfinished.")
    response["tool_calls"][0]["function"]["arguments"] = json.dumps(arguments)
    # A reserved future reply is not selected text yet; it retains the ordinary
    # budget/control tail. Once text exists, no third generation or review is bought.
    tails = []
    real_tail = loop._finish_tool_round_budget
    def tail(*args, **kwargs):
        tails.append(1)
        return real_tail(*args, **kwargs)
    monkeypatch.setattr(loop, "_finish_tool_round_budget", tail)
    monkeypatch.setattr("ouroboros.tools.review_result.result_review_at_delivery",
                        lambda *_a, **_k: pytest.fail("stop bought review"))
    result, usage, trace = run([response, {"content": text}] if reserves_reply else [response])
    assert result == text and len(calls) == (2 if reserves_reply else 1)
    assert len(tails) == int(reserves_reply)
    assert trace["task_completion"]["action"] == "stop"
    assert derive_loop_outcome(result, usage, trace)["outcome_axes"]["objective"]["reason"] == "author_stop"
    assert usage["presence_completion_outcome"] == outcome


def _hold_first_delivery(monkeypatch):
    """Hold the first delivery once at the host's delivery step, then let it pass.

    ``review_result.result_review_at_delivery`` is the delivery-time step whose ``True`` holds the
    answer for another turn; it records the content it was asked about. The real result review
    never holds Presence speech: the hold here stands for any delivery hold of a selected finish."""
    reviews = []

    def review(_ctx, content, _trace, _hold):
        reviews.append(content)
        return len(reviews) == 1

    monkeypatch.setattr("ouroboros.tools.review_result.result_review_at_delivery", review)
    return reviews


def test_a_delivery_hold_drops_old_outcome_and_uses_replacement(turn, monkeypatch):
    registry, calls, run = turn
    reviews = _hold_first_delivery(monkeypatch)
    text, usage, _trace = run([_call("tool_delivered", "Old answer"), _call("message", "Revised answer")])
    assert len(calls) == 2 and reviews == ["Old answer", "Revised answer"]
    assert registry._ctx._presence_completion["message"] == "Revised answer"
    assert usage["presence_completion_outcome"] == "message"
    result = build_presence_result_event({"id": "parent1"}, text, registry._ctx, terminal_origin=usage.get("terminal_origin", ""))
    assert (result["outcome"], result["text"]) == ("message", "Revised answer")


def test_new_explicit_finish_after_hold_replaces_old_nonempty_candidate(turn, monkeypatch):
    registry, calls, run = turn
    reviews = _hold_first_delivery(monkeypatch)
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

        def review(*_a, **_kw):
            if not reviews:
                followup()
            reviews.append(1)
            return False

        monkeypatch.setattr("ouroboros.tools.review_result.result_review_at_delivery", review)
    else:
        from tests.presence_legacy_helpers import _finish_presence

        def finish(ctx, **kwargs):
            result = _finish_presence(ctx, **kwargs)
            followup()
            return result

        registry.override_handler("presence_finish", finish)
    text, usage, _trace = run([_call("silent", "Old"), _call("message", "With the new detail")])
    assert len(calls) == 2 and text == "With the new detail"
    assert any("new detail" in str(row.get("content")) for row in calls[-1])
    assert usage["presence_completion_outcome"] == "message"
    assert build_presence_result_event({"id": "parent1"}, text, registry._ctx, terminal_origin=usage.get("terminal_origin", ""))["outcome"] == "message"


@pytest.mark.parametrize("reason", ["cancel", "budget"])
def test_control_or_budget_tail_precedes_pending_finish(turn, tmp_path, monkeypatch, reason):
    from tests.presence_legacy_helpers import _finish_presence
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
    assert text == "Old answer"  # retained authored bytes survive the real hard rail
    assert usage["terminal_origin"] == "model_final"
    if reason == "budget":
        assert tail == ["budget"] and len(calls) == 1
    else:
        assert len(calls) == 1 and registry._ctx._skip_post_task_synthesis is True


def test_ordinary_empty_and_failed_silent_outcomes_remain_failed():
    for usage in ({}, {"presence_completion_outcome": "silent", "execution_status": "failed"},
                  {"presence_completion_outcome": "tool_delivered", "execution_status": "infra_failed"}):
        outcome = derive_loop_outcome("", usage, {"tool_calls": []})
        assert outcome["outcome_axes"]["execution"]["status"] in {"failed", "infra_failed"}


@pytest.mark.parametrize("outcome,spoken", [("silent", ""), ("message", "Here is what I have so far.")])
def test_pending_children_hold_until_an_explicit_unfinished_stop(turn, tmp_path, outcome, spoken):
    from ouroboros.task_results import write_task_result, STATUS_RUNNING
    registry, calls, run = turn
    registry._ctx.task_metadata = {"presence": {"binding_id": "a" * 32, "event": {"conversation_key": "k"}}}
    write_task_result(tmp_path, "child1", STATUS_RUNNING, parent_task_id="parent1",
                      root_task_id="parent1", delegation_role="subagent", role="reviewer", result="Still running")
    stop = _call(outcome, spoken)
    arguments = json.loads(stop["tool_calls"][0]["function"]["arguments"])
    arguments.update(action="stop", rationale="Child1 is still running and remains unfinished.")
    stop["tool_calls"][0]["function"]["arguments"] = json.dumps(arguments)
    text, usage, trace = run([_call("silent", "Premature"), _call("silent", "Still premature"), stop])
    assert len(calls) == 3 and trace["task_completion"]["action"] == "stop"
    assert usage.get("reason_code") != "children_unabsorbed"
    assert trace["child_result_dispositions"]["current"] == []
    saved = read_result(tmp_path, "child1")
    assert saved["status"] == STATUS_RUNNING
    result = build_presence_result_event({"id": "parent1"}, text, registry._ctx, terminal_origin=usage.get("terminal_origin", ""))
    assert (result["outcome"], result["text"]) == (outcome, spoken)


@pytest.mark.parametrize('receipt_state,expected_calls', [('delivered', 1), ('uncertain', 2), ('failed', 2)])
def test_same_batch_presence_send_needs_its_actual_host_receipt(turn, tmp_path, receipt_state, expected_calls):
    from ouroboros.utils import append_jsonl
    registry, calls, run = turn
    registry._ctx.task_metadata = {'presence': {
        'binding_id': 'a' * 32, 'delivery_reporting_version': 1,
        'event': {'conversation_key': 'room'},
    }}
    def send(ctx, **kw):
        append_jsonl(tmp_path / 'logs' / 'chat.jsonl', {
            'task_id': 'parent1', 'type': 'presence_delivery', 'text': 'Actual send',
            'transport': {'conversation_key': 'room', 'origin': {'kind': 'tool'},
                'delivery': {'state': receipt_state, 'delivery_id': 'send-1', 'part_id': '0'}},
        })
        return 'Send returned'
    registry.override_handler('chat_history', send)
    response = _call('tool_delivered', '')
    response['tool_calls'].insert(0, {'id': 'send', 'type': 'function', 'function': {
        'name': 'chat_history', 'arguments': '{}'}})
    text, usage, trace = run([response, _call('silent', '')])
    assert len(calls) == expected_calls
    assert bool(trace['tool_calls'][0].get('presence_delivery_confirmed')) == (receipt_state == 'delivered')
    assert usage['presence_completion_outcome'] == ('tool_delivered' if expected_calls == 1 else 'silent')
