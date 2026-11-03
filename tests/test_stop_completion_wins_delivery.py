"""Running Stop custody owes its salvage before the terminal write; completion can still win.

When the worker's own terminal result lands inside the custody window, the monotonic
guard keeps it, and the provisional ``cancel:<tid>:<request_id>`` salvage owed before
the write is withdrawn with its undelivered receipt block: the outbox owes only the
kept answer, so the Done card is not held behind a false row and no "was stopped"
message is replayed after the real answer. A Stop that wins keeps its salvage owed.
"""
from __future__ import annotations

import types

import pytest

from ouroboros import cancel_intents as ci
from ouroboros.task_results import STATUS_COMPLETED, STATUS_RUNNING, load_task_result, write_task_result
from tests._cancel_intents_shared import DeliveryBus, _LiveProc, _reap_spawned_live_procs  # noqa: F401
from tests._cancel_intents_shared import qenv as _qenv

qenv = _qenv


def _running_root(qenv, task_id):
    task = {"id": task_id, "chat_id": 5, "delegation_role": "root", "root_task_id": task_id, "text": "do it"}
    proc = _LiveProc()
    qenv.workers.WORKERS[0] = types.SimpleNamespace(wid=0, proc=proc, busy_task_id=task_id, reaping=False)
    qenv.q.RUNNING[task_id] = {"task": task, "worker_id": 0}
    write_task_result(qenv.drive, task_id, STATUS_RUNNING, result="working")
    ci.request_cancel(qenv.drive, task_id)
    return proc, ci.active_intent(qenv.drive, task_id)["request_id"]


def _replayed(qenv, monkeypatch):
    from supervisor import terminal_delivery as td

    monkeypatch.setattr(td, "_REPLAY_MIN_AGE_SEC", 0.0)
    bus = DeliveryBus()
    td.replay_pending_deliveries(qenv.drive, event_queue=bus)
    sent = []
    while not bus.empty():
        sent.append(bus.get_nowait())
    return sent


@pytest.mark.serial
def test_completion_that_wins_the_custody_window_withdraws_the_provisional_salvage(qenv, monkeypatch):
    from supervisor import terminal_delivery as td

    task_id = "root-race"
    proc, rid = _running_root(qenv, task_id)
    real_claim = ci.claim_still_owned

    def worker_stores_completed_in_the_window(root, tid, intent):
        write_task_result(qenv.drive, task_id, STATUS_COMPLETED, result="Hello, the real answer.")
        return real_claim(root, tid, intent)

    monkeypatch.setattr(ci, "claim_still_owned", worker_stores_completed_in_the_window)
    try:
        qenv.tl.cancel_task_custody(task_id)
    finally:
        proc.terminate()

    stored = load_task_result(qenv.drive, task_id)
    assert stored["status"] == STATUS_COMPLETED
    assert "cancel_receipt" not in stored, "an undelivered Stop receipt does not stay on a completed task"
    owed = td.terminal_answer_receipts(qenv.drive, task_id)["owed_delivery_ids"]
    assert f"cancel:{task_id}:{rid}" not in owed and owed and all(did.startswith("final:") for did in owed)
    texts = [str(event.get("text") or "") for event in _replayed(qenv, monkeypatch)]
    assert texts == ["Hello, the real answer."], "only the kept answer is ever sent, never a 'was stopped' row"


@pytest.mark.serial
def test_a_stop_that_wins_keeps_its_salvage_owed(qenv, monkeypatch):
    """Control: no completion in the window, so the cancelled salvage is the owed answer."""
    from supervisor import terminal_delivery as td

    task_id = "root-stopped"
    proc, rid = _running_root(qenv, task_id)
    try:
        qenv.tl.cancel_task_custody(task_id)
    finally:
        proc.terminate()

    assert load_task_result(qenv.drive, task_id)["status"] == "cancelled"
    assert td.terminal_answer_receipts(qenv.drive, task_id)["owed_delivery_ids"] == [f"cancel:{task_id}:{rid}"]
