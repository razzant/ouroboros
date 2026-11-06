"""Focused durability coverage for abnormal terminal and replay-safe paths."""

from __future__ import annotations

import pathlib
import tempfile
from types import SimpleNamespace

import pytest


def _available_cost_fields(*, calls: int = 0, degraded: bool = False) -> dict:
    """The REAL projection over an empty ledger, not a hand-copied mirror of it.

    This used to be a ten-key dict literal. It was the reason a projection key
    the terminal emitter could not bind stayed invisible in a green suite: every
    test that reached the emitter substituted this stale copy for the real
    `reconstruct_task_cost`, so the shape under test was the one the test author
    remembered rather than the one production builds."""
    from supervisor.state import reconstruct_task_cost

    fields = reconstruct_task_cost(
        "no-such-task", fields=True, drive_root=pathlib.Path(tempfile.mkdtemp()),
    )
    assert fields["cost_accounting_status"] == "available"
    fields["total_rounds"] = calls
    fields["ledger_integrity_degraded"] = degraded
    return fields


def test_assign_keeps_unsafe_pending_when_terminal_write_is_not_durable(tmp_path, monkeypatch):
    from supervisor import queue, state, workers

    task = {
        "id": "unsafe-write-failure",
        "type": "task",
        "chat_id": 0,
        "_attempt": 2,
        "original_task_id": "first-attempt",
    }
    events = []
    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(workers, "PENDING", [task])
    monkeypatch.setattr(workers, "RUNNING", {})
    monkeypatch.setattr(workers, "WORKERS", {})
    monkeypatch.setattr(workers, "load_state", lambda: {"owner_chat_id": 0})
    monkeypatch.setattr(workers, "reconstruct_task_cost", lambda *_a, **_k: _available_cost_fields(calls=1))
    monkeypatch.setattr(workers, "get_event_q", lambda: SimpleNamespace(put=events.append))
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 0.0)
    monkeypatch.setattr(queue, "persist_queue_snapshot", lambda reason="": None)
    monkeypatch.setattr(
        "ouroboros.task_results.write_task_result",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("durable write failed")),
    )

    workers.assign_tasks()

    assert [item["id"] for item in workers.PENDING] == [task["id"]]
    assert "_budget_pause" not in workers.PENDING[0]
    assert not any(event.get("type") == "task_done" for event in events)


@pytest.mark.parametrize(
    ("corruption", "expected_error"),
    (("quarantined_tail", "replay_unsafe"), ("midstream", "accounting_unavailable")),
)
def test_corrupt_or_integrity_degraded_ledger_never_permits_budget_resume(
    tmp_path, monkeypatch, corruption, expected_error,
):
    from supervisor import queue, state, workers

    state.init(tmp_path, total_budget_limit=10.0)
    queue.init(tmp_path)
    monkeypatch.setattr(queue, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(queue, "PENDING", [{
        "id": "replay-risk",
        "type": "task",
        "chat_id": 0,
        "_budget_pause": {
            "status": "paused_before_dispatch",
            "physical_calls": 0,
            "replay_safe": True,
            "auto_resume": False,
        },
    }])
    monkeypatch.setattr(queue, "RUNNING", {})
    monkeypatch.setattr(workers, "WORKERS", {})
    monkeypatch.setattr(queue, "persist_queue_snapshot", lambda reason="": None)

    from tests._usage_store_testing import write_journal

    # A journal the store has not imported yet (an install upgrading from a
    # journal whose writer crashed): the import quarantines a torn final row
    # (integrity degraded) and refuses damage before it (unavailable).
    row = {"attempt_id": "a1", "kind": "attempt", "model": "test/model", "provider": "test",
           "task_id": "replay-risk", "root_task_id": "replay-risk",
           "reservation_upper_bound_usd": 0.01, "pricing_known": True}
    ledger = write_journal(tmp_path, [{**row, "state": "reserved"},
                                      {**row, "state": "released", "reason": "test_setup"}])
    if corruption == "quarantined_tail":
        with ledger.open("ab") as handle:
            handle.write(b'{"seq":')
    else:
        lines = ledger.read_text(encoding="utf-8").splitlines()
        ledger.write_text(lines[0] + "\nnot-json\n" + lines[1] + "\n", encoding="utf-8")

    result = queue.resume_budget_paused_task("replay-risk")

    assert result == {
        "ok": False,
        "error": expected_error,
        "action": "cancel_or_new_run",
        "message": {"replay_unsafe": "Resume refused: replay unsafe",
                    "accounting_unavailable": "Resume refused: accounting unavailable"}[expected_error],
    }
    assert "_budget_pause" in queue.PENDING[0]


def test_terminal_emitter_still_withholds_an_unavailable_projection(tmp_path, monkeypatch):
    """The opaque hand-off must not turn `cost_accounting_status: unavailable`
    into published numbers: the projection's `None` placeholders are absence,
    and a card that shows them as $0.00 is a false accounting claim (P1).

    The unavailable projection is produced by breaking the real ledger rather
    than hand-written here -- a fixture is one more copy of the field set, and
    copies of this field set are what the pass above went in to delete."""
    from types import SimpleNamespace

    import ouroboros.usage_accounting as usage_accounting
    from supervisor import workers
    from supervisor.state import reconstruct_task_cost

    def _ledger_down(*_a, **_k):
        raise RuntimeError("ledger down")

    monkeypatch.setattr(usage_accounting, "usage_breakdown", _ledger_down)
    unavailable = reconstruct_task_cost("t1", fields=True, drive_root=tmp_path)
    assert unavailable["cost_accounting_status"] == "unavailable"
    assert unavailable["accounted_upper_bound_usd"] is None

    events = []
    monkeypatch.setattr(workers, "get_event_q", lambda: SimpleNamespace(put=events.append))
    assert workers._emit_task_done_terminal(None, "t1", "failed", cost_fields=unavailable) is True
    emitted = events[-1]
    assert emitted["cost_accounting_status"] == "unavailable"
    assert emitted["cost_accounting_error"] == "ledger_unavailable"
    assert emitted["ledger_integrity_degraded"] is True
    assert [k for k, v in emitted.items() if v is None] == []
    assert "accounted_upper_bound_usd" not in emitted and "non_final_rows" not in emitted


def _install_supervisor(tmp_path, monkeypatch):
    from supervisor import queue, state, workers

    state.init(tmp_path, total_budget_limit=10.0)
    queue.init(tmp_path)
    monkeypatch.setattr(workers, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(queue, "DRIVE_ROOT", tmp_path)
    workers.PENDING[:] = []
    workers.RUNNING.clear()
    workers.WORKERS.clear()
    queue.BUDGET_ROOT_FENCES.clear()
    queue.init_queue_refs(workers.PENDING, workers.RUNNING, workers.QUEUE_SEQ_COUNTER_REF)
    events = []
    monkeypatch.setattr(workers, "get_event_q", lambda: SimpleNamespace(put=events.append))
    monkeypatch.setattr(workers, "send_with_budget", lambda *_a, **_k: None)
    monkeypatch.setattr(queue, "persist_queue_snapshot", lambda reason="": None)
    return queue, state, workers, events


class _DeadProc:
    exitcode = 0

    @staticmethod
    def is_alive():
        return False

    @staticmethod
    def join(timeout=None):
        del timeout


def _crashed_worker(monkeypatch, workers, task):
    workers.WORKERS[0] = SimpleNamespace(
        wid=0, busy_task_id=task["id"], proc=_DeadProc(), reaping=False,
    )
    workers.RUNNING[task["id"]] = {
        "task": task, "started_at": 1.0, "last_heartbeat_at": 1.0, "attempt": 1,
    }
    monkeypatch.setattr(workers, "QUEUE_MAX_RETRIES", 3)
    monkeypatch.setattr(workers, "_LAST_SPAWN_TIME", 0)
    monkeypatch.setattr(workers, "respawn_worker", lambda _wid: None)
    monkeypatch.setattr("ouroboros.tools.services.archive_task_service_logs", lambda *_a, **_k: None)
