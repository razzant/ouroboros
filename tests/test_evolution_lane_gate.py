"""The evolution idle gate waits for project work, never a conversation turn (review R4).

Base ran owner turns, wakes and Presence turns in the server: they were never RUNNING rows and never held the
gate. Every one of them is a RUNNING conversation-lane row now, so a chat turn (or a turn parked on an owner question,
or a busy place) held the next cycle and the status read ``waiting_for_idle`` with it counted. Only project work
holds it, in both reads (``enqueue_evolution_task_if_needed`` and the status snapshot).
"""
import pytest


def _row(task_id, kind):
    from ouroboros.task_carrier import EXECUTOR_WORKER, LANE_CONVERSATION, LANE_PROJECT
    from supervisor.conversation_lane import _awaiting_row

    if kind == "project":
        return {"task": {"id": task_id, "type": "task", "chat_id": 1, "text": "work", "_execution_owner": {
            "kind": "pooled", "lane": LANE_PROJECT, "executor": EXECUTOR_WORKER}}, "worker_id": 1, "attempt": 1}
    task = {"id": task_id, "type": "task", "chat_id": 1, "text": "hello", "_is_direct_chat": True,
            **({"_presence_turn": True} if kind == "presence" else {}),
            "_execution_owner": {"kind": kind, "lane": LANE_CONVERSATION, "executor": EXECUTOR_WORKER}}
    return _awaiting_row(task, attempt=1, client_message_id="", reply_expected=True, fresh_identity=True)


@pytest.mark.parametrize("kind, holds", [("direct", False), ("presence", False), ("project", True)])
def test_only_project_work_holds_the_evolution_idle_gate(tmp_path, monkeypatch, kind, holds):
    from supervisor import queue
    from tests.test_evolution_redesign import _owner_campaign

    pending, _notices, _ctx = _owner_campaign(tmp_path, monkeypatch)
    queue.RUNNING["turn-1"] = _row("turn-1", kind)
    status = queue.get_evolution_status_snapshot()
    queue.enqueue_evolution_task_if_needed()
    assert (status["status"] == "waiting_for_idle") is holds and status["running_count"] == (1 if holds else 0)
    assert [row.get("type") for row in pending] == ([] if holds else ["evolution"])
