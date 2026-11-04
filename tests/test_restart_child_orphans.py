"""An interrupted root leaves no orphan behind, whichever door the stop came through.

A PLANNED shutdown already cancels the PENDING children of an interrupted root
(``kill_workers(preserve_pending=True)``, trigger ``pending_parent_interrupted``).
These tests pin the same rule for the two doors that missed it: an UNPLANNED stop,
where the pre-restart RUNNING rows only exist inside ``state/queue_snapshot.json``,
and a direct-chat root, which lives in the server process and is listed only by the
off-lock fragment ``state/direct_roots.json``.
"""

from __future__ import annotations

import json
import time
import types

import pytest
from ouroboros import cancel_intents as ci
from ouroboros.task_results import (
    STATUS_CANCELLED,
    STATUS_FAILED,
    STATUS_RUNNING,
    load_task_result,
    write_task_result,
)
from ouroboros.utils import utc_now_iso
from tests._shared import write_queue_document

from tests._cancel_intents_shared import qenv as _qenv

# Re-bound through a module attribute: a direct import of a name that reappears
# as a test parameter is an F811 redefinition under the CI ruff gate.
qenv = _qenv

SERVER_STOPPED_CANCEL = "Task cancelled: the server stopped while this task was still running."
PARENT_INTERRUPTED = "Parent task was interrupted before this child started."


def _restart_snapshot(qenv, monkeypatch, *, running: list, pending: list = ()) -> None:
    """Write a queue snapshot the way an unplanned stop left it, and aim restore at it."""
    path = qenv.drive / "state" / "queue_snapshot.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_queue_document(path, {
        "ts": utc_now_iso(),
        "pending": [{"task": task} for task in pending],
        "running": running,
        "acceptance_fences": [],
        "budget_root_fences": [],
    })
    monkeypatch.setattr(qenv.q, "QUEUE_SNAPSHOT_PATH", path, raising=False)


def _direct_roots_fragment(drive, rows: list, *, incomplete: bool = False) -> None:
    path = drive / "state" / "direct_roots.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_queue_document(path, {"ts": utc_now_iso(), "roots": rows, "incomplete": incomplete})



def _boot_pool(qenv, monkeypatch) -> list:
    """Bind the worker pool to the test drive and capture its published events."""
    events: list = []
    monkeypatch.setattr(qenv.workers, "DRIVE_ROOT", qenv.drive, raising=False)
    monkeypatch.setattr(qenv.workers, "PENDING", qenv.q.PENDING, raising=False)
    monkeypatch.setattr(qenv.workers, "RUNNING", qenv.q.RUNNING, raising=False)
    monkeypatch.setattr(qenv.workers, "_EVENT_Q_SHUTDOWN", False, raising=False)
    monkeypatch.setattr(qenv.workers, "get_event_q",
                        lambda: types.SimpleNamespace(put=events.append), raising=False)
    return events


def _marker(pending: list, task_id: str) -> dict:
    row = next((row for row in pending if str(row.get("id") or "") == task_id), None)
    assert row is not None, f"{task_id} is not in PENDING"
    return row.get("_terminalization_retry") or {}


def test_restore_cancels_the_pending_child_of_an_interrupted_root(qenv, monkeypatch):
    """#1104: after an unplanned stop the fenced RUNNING root's PENDING child is
    cancelled with the parent's cause, instead of starting as an orphan on the
    first tick. Restore only marks it; the boot's own ``kill_workers`` settles the
    marker through the existing shutdown-custody path."""
    events = _boot_pool(qenv, monkeypatch)
    write_task_result(qenv.drive, "root-live", STATUS_RUNNING, chat_id=1)
    child = {
        "id": "child-waiting", "type": "task", "text": "sub", "chat_id": 1,
        "parent_task_id": "root-live", "root_task_id": "root-live",
        "delegation_role": "subagent",
    }
    write_task_result(qenv.drive, "child-waiting", "scheduled", chat_id=1)
    _restart_snapshot(
        qenv, monkeypatch,
        running=[{"id": "root-live", "task": {"id": "root-live", "chat_id": 1}}],
        pending=[child],
    )

    fenced: list = []
    assert qenv.q.restore_pending_from_snapshot(terminalized=fenced) == 0
    assert fenced == ["root-live"]
    assert _marker(qenv.q.PENDING, "child-waiting") == {
        "reason": PARENT_INTERRUPTED,
        "status": STATUS_CANCELLED,
        "trigger": "pending_parent_interrupted",
        "reconcile_delegate_custody": True,
    }
    rows = [json.loads(line) for line in
            (qenv.drive / "logs" / "supervisor.jsonl").read_text(encoding="utf-8").splitlines()]
    restore_rows = [row for row in rows if row["type"] == "queue_restored_from_snapshot"]
    assert restore_rows[-1]["pending_parent_interrupted"] == ["child-waiting"]

    # The existing boot step settles the marker: durable cancel, reconstructed
    # cost (never a fabricated $0) and a published task_done.
    qenv.workers.kill_workers(preserve_pending=True)
    assert [row["id"] for row in qenv.q.PENDING] == []
    settled = load_task_result(qenv.drive, "child-waiting")
    assert settled["status"] == STATUS_CANCELLED
    assert settled["result"] == PARENT_INTERRUPTED
    assert "cost_accounting_status" in settled  # reconstructed, not omitted
    done = [event for event in events if event["type"] == "task_done"]
    assert [event["task_id"] for event in done] == ["child-waiting"]
    assert done[0]["status"] == STATUS_CANCELLED


def test_the_interruption_guard_touches_neither_roots_nor_owned_rows(qenv, monkeypatch):
    """The three things the guard must leave exactly where it found them: a row
    with no parent (a root, a schedule, an evolution cycle), a child that
    cancellation custody already owns, and a row that already carries its own
    shutdown-custody marker with a different trigger."""
    _boot_pool(qenv, monkeypatch)
    write_task_result(qenv.drive, "root-live", STATUS_RUNNING, chat_id=1)
    schedule = {"id": "nightly", "type": "schedule", "text": "sweep", "chat_id": 1,
                "root_task_id": "root-live"}
    owned = {"id": "child-owned", "type": "task", "text": "sub", "chat_id": 1,
             "parent_task_id": "root-live", "root_task_id": "root-live"}
    marked = {"id": "child-marked", "type": "task", "text": "sub", "chat_id": 1,
              "parent_task_id": "root-live", "root_task_id": "root-live",
              "_terminalization_retry": {
                  "reason": "Worker process crashed.", "status": STATUS_FAILED,
                  "trigger": "worker_pool_kill", "reconcile_delegate_custody": False}}
    for task_id in ("nightly", "child-owned", "child-marked"):
        write_task_result(qenv.drive, task_id, "scheduled", chat_id=1)
    owned_intent = ci.request_cancel(qenv.drive, "child-owned", reason="owner pressed Stop")
    _restart_snapshot(
        qenv, monkeypatch,
        running=[{"id": "root-live", "task": {"id": "root-live", "chat_id": 1}}],
        pending=[schedule, owned, marked],
    )

    # The parentless schedule plus the row that arrived carrying its own marker.
    assert qenv.q.restore_pending_from_snapshot() == 2
    assert _marker(qenv.q.PENDING, "nightly") == {}
    assert not any(row["id"] == "child-owned" for row in qenv.q.PENDING)
    assert ci.active_intent(qenv.drive, "child-owned")["request_id"] == owned_intent["request_id"]
    assert _marker(qenv.q.PENDING, "child-marked")["trigger"] == "worker_pool_kill"


def test_restore_keeps_the_pending_children_of_a_root_it_revives(qenv, monkeypatch):
    """The guard stays quiet on the working case: an owner-wait handoff restores
    the root itself to PENDING, so nothing about it was interrupted and its child
    keeps its place in the queue."""
    _boot_pool(qenv, monkeypatch)
    root = {"id": "root-parked", "type": "task", "text": "root", "chat_id": 1,
            "root_task_id": "root-parked"}
    child = {"id": "child-parked", "type": "task", "text": "sub", "chat_id": 1,
             "parent_task_id": "root-parked", "root_task_id": "root-parked",
             "delegation_role": "subagent"}
    write_task_result(qenv.drive, "root-parked", "scheduled", chat_id=1)
    write_task_result(qenv.drive, "child-parked", "scheduled", chat_id=1)
    _restart_snapshot(qenv, monkeypatch, running=[], pending=[root, child])

    fenced: list = []
    assert qenv.q.restore_pending_from_snapshot(terminalized=fenced) == 2
    assert fenced == []
    assert sorted(row["id"] for row in qenv.q.PENDING) == ["child-parked", "root-parked"]
    assert all("_terminalization_retry" not in row for row in qenv.q.PENDING)
    assert load_task_result(qenv.drive, "child-parked")["status"] == "scheduled"


def test_restore_cancels_a_pending_child_whose_root_already_settled_as_shutdown(
    qenv, monkeypatch,
):
    """Multi-boot: the first boot fenced the root and custody already wrote its
    ``cancelled/server_shutdown`` result, so the root is no longer a RUNNING row.
    The child it never started is still in the snapshot and still cannot run."""
    _boot_pool(qenv, monkeypatch)
    write_task_result(
        qenv.drive, "root-gone", STATUS_CANCELLED, chat_id=1,
        result=SERVER_STOPPED_CANCEL,
        cancel_origin={"reason": "server_shutdown", "source": "snapshot_restore"},
    )
    child = {"id": "child-after-boot", "type": "task", "text": "sub", "chat_id": 1,
             "parent_task_id": "root-gone", "root_task_id": "root-gone",
             "delegation_role": "subagent"}
    write_task_result(qenv.drive, "child-after-boot", "scheduled", chat_id=1)
    _restart_snapshot(qenv, monkeypatch, running=[], pending=[child])

    fenced: list = []
    assert qenv.q.restore_pending_from_snapshot(terminalized=fenced) == 0
    assert fenced == []
    assert _marker(qenv.q.PENDING, "child-after-boot")["trigger"] == "pending_parent_interrupted"

    qenv.workers.kill_workers(preserve_pending=True)
    assert load_task_result(qenv.drive, "child-after-boot")["status"] == STATUS_CANCELLED


def test_a_completed_root_never_cancels_its_restored_child(qenv, monkeypatch):
    """The other direction of the multi-boot guard: a root that finished normally
    is not an interruption, so its queued child is revived as ordinary work."""
    _boot_pool(qenv, monkeypatch)
    write_task_result(qenv.drive, "root-done", STATUS_CANCELLED, chat_id=1,
                      cancel_origin={"reason": "owner pressed Stop", "source": "owner"})
    child = {"id": "child-of-owner-stop", "type": "task", "text": "sub", "chat_id": 1,
             "parent_task_id": "root-done", "root_task_id": "root-done",
             "delegation_role": "subagent"}
    write_task_result(qenv.drive, "child-of-owner-stop", "scheduled", chat_id=1)
    _restart_snapshot(qenv, monkeypatch, running=[], pending=[child])

    assert qenv.q.restore_pending_from_snapshot() == 1
    assert _marker(qenv.q.PENDING, "child-of-owner-stop") == {}


def test_a_direct_root_killed_by_the_window_is_cancelled_not_orphaned(qenv, monkeypatch):
    """Owner Q11=A: a direct-chat root the window closed on ends as an honest
    Cancelled, not ``failed / orphaned_running_after_worker_restart`` — and the
    PENDING child it never started is cancelled with it."""
    from ouroboros.task_status import load_effective_task_result

    _boot_pool(qenv, monkeypatch)
    write_task_result(qenv.drive, "direct-turn", STATUS_RUNNING, chat_id=1)
    child = {"id": "direct-child", "type": "task", "text": "sub", "chat_id": 1,
             "parent_task_id": "direct-turn", "root_task_id": "direct-turn",
             "delegation_role": "subagent"}
    write_task_result(qenv.drive, "direct-child", "scheduled", chat_id=1)
    _restart_snapshot(qenv, monkeypatch, running=[], pending=[child])
    _direct_roots_fragment(qenv.drive, [
        {"task_id": "direct-turn", "title": "a question", "chat_id": 1, "project_id": ""},
    ])

    qenv.q.init(qenv.drive)
    fenced: list = []
    assert qenv.q.restore_pending_from_snapshot(terminalized=fenced) == 0
    assert fenced == ["direct-turn"]
    intent = ci.active_intent(qenv.drive, "direct-turn")
    assert intent and (intent["reason"], intent["source"]) == ("server_shutdown", "snapshot_restore")
    assert _marker(qenv.q.PENDING, "direct-child")["trigger"] == "pending_parent_interrupted"

    assert qenv.tl.sweep_cancel_intents(now=time.time() + 60)["direct-turn"] == "cancelled"
    effective = load_effective_task_result(qenv.drive, "direct-turn")
    assert effective["status"] == STATUS_CANCELLED
    assert effective["cancel_origin"]["reason"] == "server_shutdown"
    assert effective.get("reason_code") != "orphaned_running_after_worker_restart"
    assert "TASK_ORPHAN_RECONCILED" not in str(effective.get("result") or "")

    qenv.workers.kill_workers(preserve_pending=True)
    assert load_task_result(qenv.drive, "direct-child")["status"] == STATUS_CANCELLED


def test_a_supervisor_revival_never_fences_a_direct_turn_alive_in_this_process(qenv, monkeypatch):
    """An in-process supervisor revival re-runs queue init while direct turns of THIS process
    are still running: the roster then names live work, not what a stop caught. The quiet
    direction is the window-close test above — a turn nobody runs any more IS fenced."""
    from supervisor.active_activity import get_direct_activity_registry

    _boot_pool(qenv, monkeypatch)
    write_task_result(qenv.drive, "live-direct-turn", STATUS_RUNNING, chat_id=1)
    _restart_snapshot(qenv, monkeypatch, running=[])
    _direct_roots_fragment(qenv.drive, [{"task_id": "live-direct-turn", "chat_id": 1}])
    registry = get_direct_activity_registry()
    registry.register("live-direct-turn", 1)
    try:
        qenv.q.init(qenv.drive)
        fenced: list = []
        assert qenv.q.restore_pending_from_snapshot(terminalized=fenced) == 0
    finally:
        registry.unregister("live-direct-turn")
    assert fenced == []
    assert ci.active_intent(qenv.drive, "live-direct-turn") is None


@pytest.mark.parametrize("fragment", ["incomplete", "missing"])
def test_a_direct_root_the_roster_does_not_name_is_never_fabricated(qenv, monkeypatch, fragment):
    """A turn the roster does not name gets no invented row: an ``incomplete``
    roster still hands over the turns it DOES list (they are real), while the
    skipped turn keeps exactly the projection it has today."""
    from ouroboros.task_status import load_effective_task_result

    _boot_pool(qenv, monkeypatch)
    write_task_result(qenv.drive, "unseen-turn", STATUS_RUNNING, chat_id=1,
                      ts="2026-05-28T00:00:00+00:00")
    _restart_snapshot(qenv, monkeypatch, running=[])
    if fragment == "incomplete":
        write_task_result(qenv.drive, "listed-turn", STATUS_RUNNING, chat_id=1,
                          ts="2026-05-28T00:00:00+00:00")
        _direct_roots_fragment(
            qenv.drive, [{"task_id": "listed-turn", "chat_id": 1}], incomplete=True,
        )

    qenv.q.init(qenv.drive)
    fenced: list = []
    assert qenv.q.restore_pending_from_snapshot(terminalized=fenced) == 0
    assert fenced == (["listed-turn"] if fragment == "incomplete" else [])
    assert ci.active_intent(qenv.drive, "unseen-turn") is None

    qenv.q.append_jsonl(qenv.drive / "logs" / "events.jsonl",
                        {"ts": "2026-05-28T00:00:01+00:00", "type": "llm_round",
                         "task_id": "unseen-turn"})
    qenv.q.append_jsonl(qenv.drive / "logs" / "events.jsonl",
                        {"ts": "2026-05-28T00:00:02+00:00", "type": "worker_boot"})
    effective = load_effective_task_result(qenv.drive, "unseen-turn")
    assert effective["status"] == STATUS_FAILED
    assert effective["reason_code"] == "orphaned_running_after_worker_restart"


def test_queue_init_hands_the_direct_root_roster_over_and_clears_it(qenv):
    """A stale process's turns never outlive it: the fragment is read at queue
    init — the one place that clears it — and the ids reach restore from there,
    so the file on disk is empty however the boot continues."""
    _direct_roots_fragment(qenv.drive, [
        {"task_id": "handed-over", "chat_id": 1},
        {"task_id": "", "chat_id": 1},
    ])

    qenv.q.init(qenv.drive)

    assert qenv.q.PRIOR_DIRECT_ROOTS["task_ids"] == ["handed-over"]
    assert qenv.q.PRIOR_DIRECT_ROOTS["incomplete"] is False
    assert not (qenv.drive / "state" / "direct_roots.json").exists(), "taken over and removed: nothing rewrites it"


def test_startup_takes_the_roster_over_before_restore_reads_it():
    """The handover only works in one order: the pool init that takes and clears
    the roster (`supervisor.queue.init`) runs before snapshot restore, which is
    the single reader of it — and restore still runs before the worker reset."""
    import inspect

    import server

    source = inspect.getsource(server._run_supervisor)
    init = source.index("workers_init(")
    restore = source.index("restored_pending = restore_pending_from_snapshot(")
    reset = source.index("kill_workers(preserve_pending=True)")
    assert init < restore < reset

    from supervisor import queue as queue_mod

    queue_source = inspect.getsource(queue_mod.init)
    assert "take_direct_roots" in queue_source and "PRIOR_DIRECT_ROOTS" in queue_source
