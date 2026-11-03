"""A direct-chat task the dead process had already QUEUED is reconciled at boot.

The contributor's incident (task ``aaf0dcca``, 2026-11-01 18:29Z): a direct turn
was admitted and its row queued; the server restarted; the restored PENDING copy
was held for an explicit Resume that no actor could ever answer — its direct
owner named the previous server's generation — so it stayed ``scheduled`` forever,
counted as active by every idle gate until the owner cancelled it by hand.

Snapshot restore excludes pending ids from the caught-direct handover (the queued
copy is the one that decides), which is why the caught-direct semantics never
reached it. These tests pin the repair: the queued copy whose stored owner names
a replaced generation is fenced like any direct root the stop caught, handing it
to the one cancel-intent custody that settles it — while rows somebody else can
still finish keep their existing branches and are never fenced: a live-owner row,
a schedule-born row (DIRECT here, so an overbroad predicate would fail it), a
previous-generation exact pause and a previous-generation acknowledged owner wait
(both still retained for their explicit Resume, with no cancel intent minted),
and a second restore over the settled orphan changes nothing.
"""

from __future__ import annotations

import time

from ouroboros import cancel_intents as ci
from ouroboros.task_results import (
    STATUS_CANCELLED,

    STATUS_SCHEDULED,
    load_task_result,
    write_task_result,
)
from ouroboros.utils import utc_now_iso
from tests._shared import write_queue_document
from tests._cancel_intents_shared import qenv as _qenv

# Re-bound through a module attribute: a fixture imported by name that reappears
# as a test parameter is an F811 redefinition under the CI ruff gate.
qenv = _qenv

ORPHAN = "orphan-scheduled"
LIVE = "live-direct"
FUTURE = "future-schedule"
PAUSED = "paused-direct"
WAITING = "owner-wait-direct"


def _snapshot_with_pending(qenv, monkeypatch, pending: list) -> None:
    path = qenv.drive / "state" / "queue_snapshot.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_queue_document(path, {
        "ts": utc_now_iso(),
        "pending": [{"task": task} for task in pending],
        "running": [],
        "acceptance_fences": [],
        "budget_root_fences": [],
    })
    monkeypatch.setattr(qenv.q, "QUEUE_SNAPSHOT_PATH", path, raising=False)


def _prior_generation(monkeypatch, drive) -> None:
    """The booting process replaced the dead one: a new custody session, this root's admission held."""
    from ouroboros import process_custody
    from ouroboros.server_entrypoint import admit_data_root

    monkeypatch.setattr(process_custody, "_SESSION_ID", "next-server-generation")
    admit_data_root(drive, wait_sec=0)  # released by conftest's autouse admission fixture


def _restore(qenv) -> list:
    terminalized: list = []
    qenv.q.restore_pending_from_snapshot(terminalized=terminalized)
    return terminalized


def _queued_scheduled_row(drive, task_id, *, owner_kind="direct", stamp="dead-server-generation",
                          started: bool = True, **stored_extra):
    """The durable row the dead process's admission lane left: queued, non-terminal, owner stamped."""
    from ouroboros.task_status import execution_owner_record

    owner = execution_owner_record(drive, {"id": task_id, "_attempt": 0}, owner_kind)
    owner["process"] = stamp
    write_task_result(drive, task_id, STATUS_SCHEDULED, chat_id=1, task_attempt=0,
                      execution_owner=owner, _is_direct_chat=True,
                      started_at=utc_now_iso() if started else None, **stored_extra)


def _direct_pending_row(task_id: str, **extra) -> dict:
    row = {"id": task_id, "type": "task", "chat_id": 1,
           "_is_direct_chat": True, "root_task_id": task_id,
           "admitted_dispatch": "none"}
    row.update(extra)
    return row


def _pending_rows(qenv, task_id: str) -> list:
    return [task for task in qenv.q.PENDING if str(task.get("id") or "") == task_id]


def test_queued_direct_row_of_dead_generation_is_fenced_at_boot(qenv, monkeypatch):
    _prior_generation(monkeypatch, qenv.drive)
    _queued_scheduled_row(qenv.drive, ORPHAN)
    _snapshot_with_pending(qenv, monkeypatch, [_direct_pending_row(ORPHAN)])

    terminalized = _restore(qenv)

    assert ORPHAN in terminalized
    # The fence is the durable intent; the boot's cancel sweep settles it with the
    # typed shutdown reason (pinned fully by the settle test below).
    assert ci.has_active_intent(qenv.drive, ORPHAN, strict=True)
    assert not qenv.q.PENDING or ORPHAN not in {str(task.get("id") or "") for task in qenv.q.PENDING}


def test_settle_sweep_completes_the_fenced_orphan(qenv, monkeypatch):
    """The fence is an intent; the sweep the boot already runs settles it, releasing the idle gate."""
    _prior_generation(monkeypatch, qenv.drive)
    _queued_scheduled_row(qenv.drive, ORPHAN)
    _snapshot_with_pending(qenv, monkeypatch, [_direct_pending_row(ORPHAN)])
    _restore(qenv)

    qenv.tl.sweep_cancel_intents(now=time.time() + 60)

    stored = load_task_result(qenv.drive, ORPHAN, strict=True) or {}
    assert stored.get("status") == STATUS_CANCELLED
    assert stored.get("cancel_origin", {}).get("reason") == "server_shutdown"
    # The idle gate's inputs are the queue maps: nothing of the orphan remains.
    assert ORPHAN not in {str(task.get("id") or "") for task in qenv.q.PENDING}
    assert ORPHAN not in set(qenv.q.RUNNING)


def test_live_direct_owner_row_is_untouched(qenv, monkeypatch):
    """This generation's own stamp is a live owner: the row keeps its ordinary restore branch."""
    _prior_generation(monkeypatch, qenv.drive)
    from ouroboros import process_custody

    owner_stamp = process_custody.current_custody_session_id()
    _queued_scheduled_row(qenv.drive, LIVE, stamp=owner_stamp)
    _snapshot_with_pending(qenv, monkeypatch, [_direct_pending_row(LIVE)])

    terminalized = _restore(qenv)

    assert LIVE not in terminalized
    assert not ci.has_active_intent(qenv.drive, LIVE, strict=True)
    # Retained, not silently dropped: the restored queue still holds the row.
    assert _pending_rows(qenv, LIVE), "the live-owner row must remain in the restored queue"
    stored = load_task_result(qenv.drive, LIVE, strict=True) or {}
    assert stored.get("status") == STATUS_SCHEDULED  # not cancelled, not fenced


def test_future_schedule_direct_row_is_untouched(qenv, monkeypatch):
    """A DIRECT schedule-born row (a real future occurrence) belongs to the scheduler's
    reconcile, never to this fence — a dead-generation owner stamp alone must not select it."""
    from supervisor.schedule_occurrence import _occurrence_of

    _prior_generation(monkeypatch, qenv.drive)
    occurrence = {"token": "future-token", "schedule_id": "sched-1", "dispatch": "none"}
    _queued_scheduled_row(qenv.drive, FUTURE, started=False,
                          schedule_admission={"token": "future-token", "schedule_id": "sched-1",
                                              "dispatch": "none", "task": {"id": FUTURE}})
    _snapshot_with_pending(qenv, monkeypatch, [
        _direct_pending_row(FUTURE, metadata={"schedule_occurrence": dict(occurrence)})])

    terminalized = _restore(qenv)

    assert FUTURE not in terminalized
    assert not ci.has_active_intent(qenv.drive, FUTURE, strict=True)
    # Retained, not silently dropped: the occurrence's own reconcile owns the row.
    assert _pending_rows(qenv, FUTURE), "the future-schedule row must remain in the restored queue"
    assert _occurrence_of(_pending_rows(qenv, FUTURE)[0]) is not None
    stored = load_task_result(qenv.drive, FUTURE, strict=True) or {}
    assert stored.get("status") == STATUS_SCHEDULED


def test_previous_generation_exact_pause_stays_retained_for_resume(qenv, monkeypatch):
    """An exact pause whose owner belonged to the dead generation is a saved form: its
    explicit Resume survives the restart, so the fence must not cancel it."""
    from ouroboros import budget_pause as bp
    from supervisor.restart_retention import exact_pause_row

    _prior_generation(monkeypatch, qenv.drive)
    pause_id = "pause-prev-generation"
    _queued_scheduled_row(qenv.drive, PAUSED)
    bp.set_budget_pause(qenv.drive, PAUSED, {
        "pause_id": pause_id, "task_attempt": 0, "state": bp.STATE_PAUSED,
        "reason": "budget", "scope": "root", "root_task_id": PAUSED,
        "source_ref": "ctx-pause-prev-generation",
    })
    _snapshot_with_pending(qenv, monkeypatch, [
        _direct_pending_row(PAUSED, _budget_pause={
            "exact_continuation": True, "pause_id": pause_id,
            "checkpoint": {"pause_id": pause_id}, "scope": "root",
            "root_task_id": PAUSED, "reason": "budget"})])

    terminalized = _restore(qenv)

    assert PAUSED not in terminalized
    assert not ci.has_active_intent(qenv.drive, PAUSED, strict=True)
    # Held for Resume, not merely still `scheduled`: the restored queue keeps the
    # row under its exact-pause locator.
    rows = _pending_rows(qenv, PAUSED)
    assert rows, "the exact-pause row must remain held in the restored queue"
    assert exact_pause_row(rows[0])
    stored = load_task_result(qenv.drive, PAUSED, strict=True) or {}
    assert stored.get("status") == STATUS_SCHEDULED


def test_previous_generation_owner_wait_stays_retained(qenv, monkeypatch):
    """An acknowledged owner wait whose owner belonged to the dead generation is a saved
    form: the owner can still answer it, so the fence must not cancel it."""
    _prior_generation(monkeypatch, qenv.drive)
    _queued_scheduled_row(qenv.drive, WAITING, owner_wait={
        "state": "waiting", "reason": "owner", "source_ref": "ctx-wait-prev-generation",
        "wait_id": "wait-prev-generation", "task_attempt": 0})
    _snapshot_with_pending(qenv, monkeypatch, [_direct_pending_row(WAITING)])

    terminalized = _restore(qenv)

    assert WAITING not in terminalized
    assert not ci.has_active_intent(qenv.drive, WAITING, strict=True)
    # Retained for its owner, not silently dropped and not cancelled.
    assert _pending_rows(qenv, WAITING), "the owner-wait row must remain held in the restored queue"
    stored = load_task_result(qenv.drive, WAITING, strict=True) or {}
    assert stored.get("status") == STATUS_SCHEDULED
    assert (stored.get("owner_wait") or {}).get("state") == "waiting"


def test_reconciliation_is_idempotent(qenv, monkeypatch):
    _prior_generation(monkeypatch, qenv.drive)
    _queued_scheduled_row(qenv.drive, ORPHAN)
    _snapshot_with_pending(qenv, monkeypatch, [_direct_pending_row(ORPHAN)])
    _restore(qenv)
    qenv.tl.sweep_cancel_intents(now=time.time() + 60)
    first = dict(load_task_result(qenv.drive, ORPHAN, strict=True) or {})

    # A second restore over the same settled row must not mint a second intent,
    # resurrect the pending copy or rewrite the terminal result.
    _prior_generation(monkeypatch, qenv.drive)
    terminalized = _restore(qenv)
    qenv.tl.sweep_cancel_intents(now=time.time() + 120)

    assert terminalized == [] or all(tid != ORPHAN for tid in terminalized)
    second = load_task_result(qenv.drive, ORPHAN, strict=True) or {}
    assert second.get("status") == STATUS_CANCELLED
    assert second.get("cancel_origin") == first.get("cancel_origin")
    assert ORPHAN not in {str(task.get("id") or "") for task in qenv.q.PENDING}
