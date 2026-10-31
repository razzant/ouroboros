"""A task stopped before it ever ran keeps the room it was queued for.

The cancel writes such a task's first result row, and it used to write it without the queue
row's address. The terminal publication sweep can run before the ``task_done`` event that
carries the address (the event queue lags after a restart), and the "Stopped" row then went to
the hidden partition: Main lost it from its own history and the Telegram task notice skipped
it, while the same stop a minute later, with ``task_done`` first, reached its room.
"""

from __future__ import annotations

import pytest

from ouroboros import cancel_intents, terminal_projection
from ouroboros.task_results import load_task_result
from supervisor.cancel_publication import _cancel_result_fields
from tests._cancel_intents_shared import qenv as _qenv

qenv = _qenv


def _terminal_rows(root, task_id):
    from tests._dialogue import chat_rows

    return [row for row in chat_rows(root) if row.get("task_id") == task_id
            and row.get("summary_kind") == "terminal_root_projection"]


@pytest.mark.parametrize("chat", [1, 9346735, 0], ids=["main", "project-room", "hidden"])
def test_a_publication_before_task_done_keeps_the_queued_room(qenv, monkeypatch, chat):
    emitted = []
    monkeypatch.setattr(qenv.q, "_emit_cancel_task_done",
                        lambda task, task_id, **_kw: emitted.append((task_id, task.get("chat_id"))))
    qenv.q.PENDING.append({"id": "turn-1", "type": "task", "chat_id": chat, "root_task_id": "turn-1"})
    cancel_intents.request_cancel(qenv.drive, "turn-1", source="http_single")

    assert qenv.tl.cancel_task_custody("turn-1") == qenv.tl.CANCEL_CANCELLED
    # The sweep publishes first: the task_done event (it carries the same address) is still queued.
    terminal_projection.reconcile_terminal_projections(qenv.drive)
    assert [row["chat_id"] for row in _terminal_rows(qenv.drive, "turn-1")] == [chat]
    assert load_task_result(qenv.drive, "turn-1").get("chat_id") == chat and emitted == [("turn-1", chat)]


def test_the_assignment_pass_drop_keeps_the_queued_room_too(qenv, monkeypatch):
    from supervisor import workers

    emitted = []
    monkeypatch.setattr(workers, "_emit_task_done_terminal",
                        lambda task, tid, status, **_kw: emitted.append((tid, task.get("chat_id"))) or True)
    monkeypatch.setattr(workers, "PENDING", qenv.q.PENDING, raising=False)
    monkeypatch.setattr(workers, "DRIVE_ROOT", qenv.drive, raising=False)
    qenv.q.PENDING[:] = [{"id": "turn-2", "type": "task", "chat_id": 9346735, "root_task_id": "turn-2"}]
    cancel_intents.request_cancel(qenv.drive, "turn-2", source="http_single")

    workers._drop_cancelled_pending()
    terminal_projection.reconcile_terminal_projections(qenv.drive)
    assert [row["chat_id"] for row in _terminal_rows(qenv.drive, "turn-2")] == [9346735]
    assert load_task_result(qenv.drive, "turn-2")["chat_id"] == 9346735 and emitted == [("turn-2", 9346735)]


def test_an_existing_address_wins_and_a_missing_one_is_never_invented():
    assert "chat_id" not in _cancel_result_fields({"chat_id": 1}, existing={"chat_id": 0}, result="x")
    assert _cancel_result_fields({"chat_id": 1}, existing={"status": "scheduled"}, result="x")["chat_id"] == 1
    assert _cancel_result_fields({"chat_id": -1001}, existing=None, result="x")["chat_id"] == -1001
    assert "chat_id" not in _cancel_result_fields({"id": "t"}, existing={}, result="x")
