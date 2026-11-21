"""What a conversation turn carried by a lane worker keeps of its own execution (review M4, Astra m1).

The round facts read the turn's own wait (``context_runtime_facts._owner_wait_fact``) in whichever process
runs the turn: an inline executor's registry entry, or the carrier's own execution. And a finished turn
leaves in its carrier only the small immutable late-event address its endpoint stamps late rows with, never
its hooks (the namer's closure holds the whole task, the frame observer its execution).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from supervisor import workers
from tests.conversation_carrier_support import carry, isolate_lane_pool, lane_rows


ORIGIN = {"chat_id": 1, "client_message_id": "owner-msg-1", "ts": "2026-11-02T10:00:00+00:00"}


@pytest.fixture
def lane(monkeypatch, tmp_path):
    import queue

    from supervisor import message_bus, state

    bus = queue.Queue()
    bridge = SimpleNamespace(send_chat_action=lambda *a, **k: None, broadcast=lambda *a, **k: None,
                             send_message=lambda *a, **k: (True, ""))
    [spare] = isolate_lane_pool(monkeypatch, tmp_path, event_q=bus)
    monkeypatch.setattr(workers, "REPO_DIR", tmp_path / "repo")
    monkeypatch.setattr(workers, "send_with_budget", lambda *a, **k: (True, ""))
    monkeypatch.setattr(message_bus, "get_bridge", lambda: bridge)
    monkeypatch.setattr(state, "load_state", lambda: {"owner_chat_id": 7})
    monkeypatch.setattr(state, "budget_remaining", lambda *a, **kw: 100)
    workers.open_repo_writer_admission()
    return SimpleNamespace(spare=spare, bus=bus)


class _Actor:
    def __init__(self, behave):
        self.behave, self._busy, self._last_activity_ts = behave, False, None

    def handle_task(self, task):
        return self.behave(self, task)


def _owner_turn(lane, text="hello"):
    workers.handle_chat_direct(1, text, task_metadata={"client_message_id": ORIGIN["client_message_id"],
                                                       "origin_message_ref": dict(ORIGIN)})
    [row] = lane_rows(workers).values()
    assert row["worker_id"] == lane.spare.wid
    return row["task"]["id"]


def test_a_carried_owner_turn_reads_its_own_wait_in_its_carrier(lane):
    from ouroboros.context_runtime_facts import _owner_wait_fact

    seen = {}

    def behave(_actor, task):
        seen["during"] = _owner_wait_fact(SimpleNamespace(task_id=task["id"]))
        return []

    task_id = _owner_turn(lane)
    carry(workers, lane.spare, _Actor(behave))
    assert seen["during"].startswith(f"my human's message accepted at {ORIGIN['ts']}; ")
    assert _owner_wait_fact(SimpleNamespace(task_id=task_id)) == ""  # released with the turn


def test_a_resumed_turn_waits_from_its_own_entry_and_keeps_its_messages_acceptance_apart(lane):
    """A turn that resumes saved work (here its warm wait, returned after a restart) answers no fresh message: its
    wait runs from where this carrier took it, never from the turn's first admission a day ago, and the message's
    acceptance is its own fact; the fresh turn above still measures from that acceptance."""
    from ouroboros.context_runtime_facts import _owner_wait_fact
    from supervisor.active_activity import turn_execution
    from tests.conversation_carrier_support import dispatched_tasks

    seen = {}

    def behave(_actor, task):
        seen.update(fact=_owner_wait_fact(SimpleNamespace(task_id=task["id"])), receipt=turn_execution(task["id"]).receipt())
        return []

    _owner_turn(lane)
    for task in dispatched_tasks(lane.spare):
        task["_owner_wait_resume"] = {"wait_id": "warm-1"}  # a governing continuation (owner_wait.governing_continuation)
        task["_conversation_prepare"]["admitted_at"] -= 86_400
        (lane.spare.in_q.append if isinstance(lane.spare.in_q, list) else lane.spare.in_q.put)(task)
    carry(workers, lane.spare, _Actor(behave))
    receipt = seen["receipt"]
    assert (receipt["resumed"], receipt["origin_accepted_at"]) == ("owner_wait", ORIGIN["ts"])
    assert "accepted_at" not in receipt and "owner_wait_sec" not in receipt and receipt["handler_wait_sec"] < 3600
    assert seen["fact"].startswith("my resumed turn (owner_wait) began at ") and f"accepted at {ORIGIN['ts']}" in seen["fact"]


def test_a_finished_carried_turn_keeps_only_its_late_event_address(lane):
    from supervisor.log_addressing import _TASK_ENDPOINTS, stamp_recorded_row

    task_id = _owner_turn(lane, "x" * 1_000_000)  # a large input
    carry(workers, lane.spare, _Actor(lambda _actor, _task: []))
    endpoint = _TASK_ENDPOINTS[task_id]
    assert endpoint._on_first_work is None and endpoint._on_frame is None  # no namer closure, no observer
    late = stamp_recorded_row({"type": "tool_call", "task_id": task_id})
    assert late["chat_id"] == 1 and late.get("_is_direct_chat") is True  # the address still stamps a late row
