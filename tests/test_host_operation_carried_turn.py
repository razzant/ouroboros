"""#2622's operation view of a message whose owner turn a conversation-lane carrier runs (review N3).

The target's turn was a registry actor and never a RUNNING row, so the managed root it promoted answered the
operation. A carried turn IS a RUNNING lane row: it answers as that actor did (``direct_chat``, running, its
own cancel), never as the managed root, and a project task it promoted answers (and is what ``/chat/cancel``
targets) while the turn's row is still in its post-task tail. A settled carried turn's own record is the
turn's (its origin), so its text is the origin-indexed final, as the registry turn's was.
"""

from __future__ import annotations

import time

from ouroboros.task_carrier import EXECUTOR_WORKER, LANE_CONVERSATION
from tests.test_host_service_operations import (
    CHAT, MSG, _client, _final, _headers, _inbound, _isolate_queue, _origin_ref, _view, operation_ref,
)

CARRIED = {"kind": "direct", "lane": LANE_CONVERSATION, "executor": EXECUTOR_WORKER}


def _carried_turn(running, ref, task_id="turn-1"):
    """The lane row the admission and dispatch leave for an owner turn (``conversation_lane._awaiting_row``)."""
    running[task_id] = {"task": {"id": task_id, "chat_id": CHAT, "root_task_id": task_id, "_is_direct_chat": True,
                                 "origin_message_ref": ref, "_execution_owner": dict(CARRIED)},
                        "worker_id": 91, "lane": LANE_CONVERSATION, "attempt": 1, "started_at": time.time()}


def test_a_carried_turn_answers_as_the_turn_and_its_promoted_root_answers_while_the_turn_finishes(tmp_path, monkeypatch):
    from ouroboros.gateway import host_service
    from ouroboros.task_results import STATUS_RUNNING, write_task_result

    queue, pending = _isolate_queue(monkeypatch, tmp_path, [])
    client = _client(tmp_path)
    _inbound(tmp_path, "build it")
    ref = _origin_ref(tmp_path)
    write_task_result(tmp_path, "turn-1", STATUS_RUNNING, origin_message_ref=ref, _is_direct_chat=True,
                      execution_owner=dict(CARRIED))
    _carried_turn(queue.RUNNING, ref)
    state = _view(client)
    assert (state["task_id"], state["phase"], state["status"], state["cancel_supported"]) == (
        "turn-1", "direct_chat", STATUS_RUNNING, True)

    # The turn promoted work and answered; its lane row is still in its post-task tail.
    pending.append({"id": "proj-task", "chat_id": 1001, "root_task_id": "proj-task", "origin_message_ref": ref})
    write_task_result(tmp_path, "turn-1", "completed", result="Started a project task.", origin_message_ref=ref,
                      _is_direct_chat=True, execution_owner=dict(CARRIED))
    _final(tmp_path, ref, "Started a project task.")
    assert (_view(client)["task_id"], _view(client)["status"]) == ("proj-task", "pending")
    pending.clear()  # a project worker takes it
    queue.RUNNING["proj-task"] = {"task": {"id": "proj-task", "chat_id": 1001, "root_task_id": "proj-task",
                                           "origin_message_ref": ref}, "worker_id": 1, "attempt": 1}
    write_task_result(tmp_path, "proj-task", STATUS_RUNNING, origin_message_ref=ref, project_id="room", chat_id=1001)
    targets = []
    monkeypatch.setattr(host_service, "_cancel_task_through_owner",
                        lambda _skill, task_id, _reason, base, _root: targets.append(task_id) or (200, base))
    for _phase in ("tail", "turn gone"):
        state = _view(client)
        assert (state["task_id"], state["phase"], state["status"], state["cancel_supported"]) == (
            "proj-task", "managed_task", STATUS_RUNNING, True)
        assert client.post("/chat/cancel", headers=_headers(), json={"operation_ref": operation_ref(CHAT, MSG)})
        queue.RUNNING.pop("turn-1", None)  # the turn's row leaves: the operation still reads the promoted root
    assert targets == ["proj-task", "proj-task"]


def test_a_settled_carried_turn_answers_with_its_origin_indexed_final(tmp_path, monkeypatch):
    from ouroboros.task_results import write_task_result

    _isolate_queue(monkeypatch, tmp_path, [])
    client = _client(tmp_path)
    _inbound(tmp_path, "hello")
    ref = _origin_ref(tmp_path)
    write_task_result(tmp_path, "turn-1", "completed", result="the stored result", origin_message_ref=ref,
                      _is_direct_chat=True, execution_owner=dict(CARRIED))
    _final(tmp_path, ref, "the turn's answer")
    state = _view(client)
    assert (state["status"], state["text"], state["task_id"]) == ("completed", "the turn's answer", "turn-1")
    assert state.get("phase") != "managed_task"
