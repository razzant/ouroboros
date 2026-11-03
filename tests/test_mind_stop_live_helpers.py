"""The mind's Stop reaches what the owner's Stop reaches: a settled root's live helpers.

``cancel_task`` runs in the worker, where the supervisor's queue maps are out of
reach. A settled root whose helpers still run answered "Nothing to cancel" there,
while the owner's door treated the same root as still owned and cascaded over it.
The queue snapshot twin (``task_status.task_subtree_has_live_queue_rows``) proves
that subtree live, the intent is minted for the settled root and widened to the
cascade, and the supervisor's existing custody (the ``cancel_task`` control event,
or the intent sweep when that event is lost) stops the live members while the
root's own result stays. Observe and delegated narrowing, and "completion wins"
for a tree with nothing live, are unchanged.
"""
from __future__ import annotations

import queue as std_queue
import time
from types import SimpleNamespace

import pytest

from ouroboros import cancel_intents as ci
from ouroboros.task_results import STATUS_CANCELLED, STATUS_COMPLETED, load_task_result, write_task_result
from ouroboros.tools.join_ledger import _cancel_task
from ouroboros.utils import utc_now_iso
from tests._cancel_intents_shared import qenv as _qenv
from tests._shared import write_queue_document

# Requested by name as a test parameter: re-bound through a module attribute (F811 otherwise).
qenv = _qenv

ROOT, HELPER = "root-done", "helper-live"


def _snapshot(drive, *, pending=(), running=(), stale=False):
    write_queue_document(drive / "state" / "queue_snapshot.json", {
        "ts": "2000-01-01T00:00:00Z" if stale else utc_now_iso(),
        "running": list(running), "pending": list(pending)})


def _row(task_id, root_id):
    return {"id": task_id, "task": {"id": task_id, "root_task_id": root_id, "parent_task_id": root_id}}


def _caller(drive, **metadata):
    return SimpleNamespace(task_id="mind-task", drive_root=drive, pending_events=[], event_queue=std_queue.Queue(),
                           task_metadata={"root_task_id": "mind-task", "delegation_role": "root", **metadata})


def _settled_root_with_a_live_helper(drive):
    write_task_result(drive, ROOT, STATUS_COMPLETED, result="the release is out", root_task_id=ROOT, chat_id=0)
    write_task_result(drive, HELPER, "scheduled", root_task_id=ROOT, parent_task_id=ROOT, chat_id=0)
    _snapshot(drive, pending=[_row(HELPER, ROOT)])


def _emitted(monkeypatch):
    events = []
    monkeypatch.setattr("ouroboros.tools.control._emit_control_event", lambda _ctx, evt: events.append(evt) or "live")
    return events


@pytest.mark.parametrize("driver", ["control_event", "intent_sweep"])
def test_a_settled_roots_live_helper_is_stopped_by_the_supervisors_custody(qenv, monkeypatch, driver):
    _settled_root_with_a_live_helper(qenv.drive)
    events = _emitted(monkeypatch)
    # Worker-shaped: the supervisor maps hold nothing; only the durable snapshot is read.
    assert qenv.q.PENDING == [] and qenv.q.RUNNING == {}
    reply = _cancel_task(_caller(qenv.drive), ROOT, reason="the release is out")
    assert reply.startswith(f"Cancel requested: {ROOT}") and "helpers and external runs stop with it" in reply, reply
    intent = ci.active_intent(qenv.drive, ROOT)
    assert (intent["task_id"], intent["scope"], intent["source"]) == (ROOT, ci.SCOPE_CASCADE, "agent_tool")
    assert [(event["type"], event["task_id"]) for event in events] == [("cancel_task", ROOT)]

    # Supervisor side: the helper is live in its own maps; the root left them when it settled.
    qenv.q.PENDING[:] = [{"id": HELPER, "chat_id": 0, "root_task_id": ROOT, "parent_task_id": ROOT, "depth": 1}]
    monkeypatch.setattr(qenv.q, "_emit_cancel_task_done", lambda *_a, **_k: None)
    if driver == "control_event":
        from supervisor.events_runtime_controls import _drive_cancel_task_event

        _drive_cancel_task_event(events[0], SimpleNamespace())
    else:
        # The event was lost: the watchdog re-feeds the durable intent as the cascade it records.
        qenv.tl.sweep_cancel_intents(now=time.time() + 3600)
    assert qenv.q.PENDING == []
    assert load_task_result(qenv.drive, HELPER)["status"] == STATUS_CANCELLED
    root = load_task_result(qenv.drive, ROOT)
    assert (root["status"], root["result"]) == (STATUS_COMPLETED, "the release is out"), "completion wins"
    assert ci.active_intent(qenv.drive, ROOT) is None, "the cascade's postcondition settled it"


def test_a_settled_root_with_nothing_live_still_has_nothing_to_cancel(tmp_path, monkeypatch):
    write_task_result(tmp_path, ROOT, STATUS_COMPLETED, result="done", root_task_id=ROOT)
    # Its own past row and another tree's helper are not a live subtree of this root.
    _snapshot(tmp_path, pending=[_row("other-helper", "other-root")])
    events = _emitted(monkeypatch)
    reply = _cancel_task(_caller(tmp_path), ROOT)
    assert "Nothing to cancel" in reply and STATUS_COMPLETED in reply
    assert ci.active_intent(tmp_path, ROOT) is None and events == []


@pytest.mark.parametrize("metadata", [
    pytest.param({"tool_profile": "local_readonly_subagent"}, id="delegated"),
    pytest.param({"initiator": "consciousness", "consciousness_autonomy": "observe"}, id="observe"),
])
def test_restricted_callers_still_cannot_stop_another_roots_helpers(tmp_path, monkeypatch, metadata):
    _settled_root_with_a_live_helper(tmp_path)
    events = _emitted(monkeypatch)
    reply = _cancel_task(_caller(tmp_path, **metadata), ROOT)
    assert "is not a child of this task" in reply
    assert ci.active_intent(tmp_path, ROOT) is None and events == []


def test_the_snapshot_twin_reads_only_the_roots_other_members_and_fails_open(tmp_path):
    from ouroboros.task_status import task_subtree_has_live_queue_rows as live

    assert live(tmp_path, ROOT) is True, "a missing snapshot proves nothing"
    _snapshot(tmp_path, running=[_row(ROOT, ROOT)], pending=[_row("other-helper", "other-root")])
    assert live(tmp_path, ROOT) is False
    _snapshot(tmp_path, running=[_row(HELPER, ROOT)])
    assert live(tmp_path, ROOT) is True
    _snapshot(tmp_path, pending=[_row(HELPER, ROOT)])
    assert live(tmp_path, ROOT) is True
    _snapshot(tmp_path, stale=True)
    assert live(tmp_path, ROOT) is True, "a stale snapshot proves nothing"


def test_an_exact_retry_replays_the_stop_after_its_helpers_ended(tmp_path, monkeypatch):
    _settled_root_with_a_live_helper(tmp_path)
    _emitted(monkeypatch)
    first = _cancel_task(_caller(tmp_path), ROOT, stop_action_id="stop-release-helpers")
    assert first.startswith(f"Cancel requested: {ROOT}") and "helpers and external runs stop with it" in first
    _snapshot(tmp_path)  # the helpers ended before the uncertain answer was retried
    again = _cancel_task(_caller(tmp_path), ROOT, stop_action_id="stop-release-helpers")
    assert "STOP_ACTION_CONFLICT" not in again and "(same Stop action replayed)" in again, again
    assert ci.active_intent(tmp_path, ROOT)["scope"] == ci.SCOPE_CASCADE
