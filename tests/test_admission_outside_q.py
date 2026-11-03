"""Admission persists outside Q: reserve in memory, persist on the admission lane, revalidate, put.

Real queue maps, real reservations, real SQLite contention (a pinned reader on the rollback journal,
a competing writer on WAL), the real preparation generation and its lanes. Every moved path is
checked in both directions: it admits once under contention without a storage acquisition while Q is
owned, and the same path admits at once when nothing contends.
"""
from __future__ import annotations

import contextlib
import os
import queue as stdqueue
import threading
import time
import traceback

import pytest

from ouroboros import store
from ouroboros.platform_layer import process_start_time
from ouroboros.task_results import STATUS_REQUESTED, STATUS_SCHEDULED, write_task_result
from supervisor import queue, queue_snapshot, task_admission, task_lifecycle, workers
from supervisor import workspace_preparation as prep
from tests._store_journal_shared import use_journal
from tests.test_admission_contention import database_reader
from tests.test_scheduled_receipt_contention import install_worker, request
from tests.test_workspace_preparation import _WAIT_SEC, spy_provisions
from tests.test_workspace_preparation import env as env

pytestmark = pytest.mark.serial


def wait_for(predicate, what):
    deadline = time.monotonic() + _WAIT_SEC
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError(f"timed out waiting for {what}")


def store_opens_under_q(monkeypatch):
    """Every store connection opened while Q is owned, named by the function that opened it.
    The moved paths must never appear here; an entry is the old defect come back. ``seen.stop()``
    restores the real seam before paths outside this file's subject run."""
    class Seen(list):
        def stop(self) -> None:
            monkeypatch.setattr(store, "connection", real)

    seen = Seen()
    real = store.connection

    @contextlib.contextmanager
    def connection(*args, **kwargs):
        if queue._queue_lock._is_owned():
            seen.append([frame.name for frame in traceback.extract_stack()][-12:])
        with real(*args, **kwargs) as conn:
            yield conn

    monkeypatch.setattr(store, "connection", connection)
    return seen


def hold_at_receipt(env, monkeypatch, stack, *, write):
    """Begin the hold at the first receipt attempt (the lane's earlier marker writes are done), as a
    reader (``write=False``: SHARED, which holds a rollback-journal COMMIT) or a writer (WAL)."""
    original = task_admission.write_task_result
    held, transitions, failures = [], [], []

    def receipt(root, task_id, status, **fields):
        if task_id != "receipt-child" or status != STATUS_SCHEDULED:
            return original(root, task_id, status, **fields)
        transitions.append(fields["delegation_admission"]["transition_id"])
        if not held:
            held.append(stack.enter_context(database_reader(env.data, write=write)))
        try:
            return original(root, task_id, status, **fields)
        except store.StoreBusy as exc:
            failures.append((exc.reason, getattr(exc, "rollback_confirmed", None)))
            raise

    monkeypatch.setattr(task_admission, "write_task_result", receipt)
    return held, transitions, failures


@pytest.mark.parametrize("journal", ["PERSIST", "WAL"])
@pytest.mark.parametrize("acting", [False, True])
def test_pinned_reader_admits_once_at_the_lane_without_storage_under_q(env, monkeypatch, journal, acting):
    use_journal(monkeypatch, journal)
    event = request(env, acting)
    bus = env.ctx.event_queue = stdqueue.Queue()
    copies = spy_provisions(monkeypatch)
    worker, delivered = install_worker(env, monkeypatch)
    monkeypatch.setattr(workers, "MAX_WORKERS", 1)  # one slot: the pool spawns no process for the second child
    under_q = store_opens_under_q(monkeypatch)
    with contextlib.ExitStack() as stack:
        held, transitions, failures = hold_at_receipt(env, monkeypatch, stack, write=journal == "WAL")
        assert env.dispatch_returns(event)
        wait_for(lambda: len(failures) >= 2, "the receipt retried at its lane")
        assert all(row == ("contention", True) for row in failures)
        assert env.pending_ids() == [] and delivered == [] and bus.empty()
        assert [claim.task_id for claim in prep._LIVE.values()] == ["receipt-child"]
        assert not env.gen.wait_idle(0)  # still owned by its claim, retrying at the head of its lane
        held[0]()
        assert env.gen.wait_idle(_WAIT_SEC)
    assert env.pending_ids() == ["receipt-child"] and len(set(transitions)) == 1 and bus.empty()
    row = env.row("receipt-child")
    assert row["status"] == STATUS_SCHEDULED and row["delegation_admission"]["transition_id"] == transitions[0]
    assert row[prep.MARKER]["phase"] == prep.ADMITTED
    assert copies == (["receipt-child"] if acting else [])
    assert under_q == [], under_q
    assert queue.ADMISSION_RESERVATIONS == {} and not prep._LIVE
    # The other direction: with nothing contending the same path admits at once.
    second = env.request("receipt-second", root="root", parent="parent") if acting else env.readonly("receipt-second", root="root")
    assert env.dispatch_returns(second) and env.gen.wait_idle(_WAIT_SEC)
    assert sorted(env.pending_ids()) == ["receipt-child", "receipt-second"]
    assert env.row("receipt-second")["status"] == STATUS_SCHEDULED and under_q == [], under_q
    seen = under_q
    seen.stop()  # assignment's own untouched reads are the launch suite's subject
    workers.assign_tasks()  # reserves under Q; the admission lane puts
    assert env.gen.wait_idle(_WAIT_SEC)
    assert [task["id"] for task in delivered] == ["receipt-child"] and worker.busy_task_id == "receipt-child"


@pytest.mark.parametrize("acting", [False, True])
@pytest.mark.parametrize("control", ["stop", "pause"])
def test_control_between_receipt_and_put_wins_and_settles_the_receipt_typed(env, monkeypatch, acting, control):
    from supervisor.events_budget import _set_root_budget_pause_locked

    event = request(env, acting)
    _worker, delivered = install_worker(env, monkeypatch)
    landed, proceed = threading.Event(), threading.Event()
    real = task_admission.write_task_result

    def write(root, tid, status, **fields):
        result = real(root, tid, status, **fields)
        if tid == "receipt-child" and status == STATUS_SCHEDULED:
            landed.set()  # the receipt is durable; the put has not been entered
            assert proceed.wait(_WAIT_SEC)
        return result

    monkeypatch.setattr(task_admission, "write_task_result", write)
    assert env.dispatch_returns(event) and landed.wait(_WAIT_SEC)
    assert env.row("receipt-child")["status"] == STATUS_SCHEDULED and env.pending_ids() == []
    with queue._queue_lock:
        if control == "stop":
            task_lifecycle.CANCELLED_ROOT_FENCES["root"] = "2026-10-20T00:00:00Z"
        else:
            _set_root_budget_pause_locked("root", {"cause": "owner_pause", "fence_id": "between-receipt-and-put"})
    proceed.set()
    assert env.gen.wait_idle(_WAIT_SEC)
    workers.assign_tasks()
    assert delivered == [] and env.pending_ids() == [] and queue.ADMISSION_RESERVATIONS == {}
    row = env.row("receipt-child")
    assert row["status"] == "failed" and row["admission_outcome"] == "never_admitted"
    assert row["reason_code"] == ("root_cancelled" if control == "stop" else "root_owner_paused")
    assert row[prep.MARKER]["phase"] == prep.REFUSED
    assert any("not scheduled" in text for _, text, _ in env.ctx.sent)  # the parent is informed as today


def test_receipt_without_put_is_refused_by_recovery_once_the_saved_queue_is_owned(env, monkeypatch):
    """W4: the receipt landed and the process died before the put. Absence from the restored queue
    proves the put never happened only once this process owns the saved queue; before that, unknown."""
    event = env.request("crash-put")
    copies = spy_provisions(monkeypatch)

    def died(task):
        raise RuntimeError("the process died before the put")

    monkeypatch.setattr(env.ctx, "enqueue_task", died)
    assert env.dispatch_returns(event) and env.gen.wait_idle(_WAIT_SEC)
    row = env.row("crash-put")
    assert row["status"] == STATUS_SCHEDULED and row[prep.MARKER]["phase"] == prep.ADMITTING
    assert env.pending_ids() == [] and queue.ADMISSION_RESERVATIONS == {} and not prep._LIVE
    resource = row[prep.MARKER]["resource"]["path"]
    monkeypatch.setattr(queue_snapshot, "saved_queue_unowned", lambda: True)
    report = prep.recover_interrupted_preparations(env.data)
    assert report["uncertain"] == ["crash-put"] and env.row("crash-put")["status"] == STATUS_SCHEDULED
    monkeypatch.setattr(queue_snapshot, "saved_queue_unowned", lambda: False)
    report = prep.recover_interrupted_preparations(env.data)
    assert report["settled"] == ["crash-put"], report
    row = env.row("crash-put")
    assert row["status"] == "failed" and row["admission_outcome"] == "never_admitted"
    assert row[prep.MARKER]["phase"] == prep.REFUSED and row[prep.MARKER]["resource_disposition"] == "after_refusal_receipt"
    assert not os.path.exists(resource) and copies == ["crash-put"]  # nothing re-copied, retried or dispatched
    assert env.pending_ids() == [] and prep.recover_interrupted_preparations(env.data)["settled"] == []


def test_put_row_whose_admitted_marker_was_never_written_is_retired_by_recovery(env, monkeypatch):
    """W5: the put and its snapshot happened; the closing marker write did not. The queue holds the
    row, so recovery retires the marker instead of refusing anything."""
    monkeypatch.setattr(prep.PreparationClaim, "close_admitted", lambda self: None)
    assert env.dispatch_returns(env.request("late-admit")) and env.gen.wait_idle(_WAIT_SEC)
    assert env.pending_ids() == ["late-admit"] and env.phase("late-admit") == prep.ADMITTING
    assert env.row("late-admit")["status"] == STATUS_SCHEDULED
    report = prep.recover_interrupted_preparations(env.data)
    assert report["admitted"] == ["late-admit"] and report["settled"] == []
    row = env.row("late-admit")
    assert row["status"] == STATUS_SCHEDULED and row[prep.MARKER]["phase"] == prep.ADMITTED
    assert env.pending_ids() == ["late-admit"] and "admission_outcome" not in row


def test_marker_less_requested_rows_resolve_only_once_their_request_cannot_arrive(env, monkeypatch):
    from supervisor import message_bus

    sent = []
    monkeypatch.setattr(message_bus, "send_with_budget", lambda chat_id, text, **kwargs: sent.append((chat_id, text, kwargs)))
    gone = {"pid": 2147483200, "birth": "ended"}
    me = {"pid": os.getpid(), "birth": process_start_time(os.getpid())}
    write_task_result(env.data, "ended-parent", "cancelled", result="stopped")
    write_task_result(env.data, "live-parent", "running", result="working")
    rows = {
        "lost": ("live-parent", {**gone, "host_pid": gone["pid"], "host_birth": "ended"}),  # producer and bus ended
        "bus-live": ("live-parent", {**gone, "host_pid": me["pid"], "host_birth": me["birth"]}),  # this bus may carry it
        "legacy-orphan": ("ended-parent", None),  # predates the stamp: its parent ended
        "legacy-held": ("live-parent", None),  # predates the stamp: its parent lives
        "queued-elsewhere": ("live-parent", {**gone, "host_pid": gone["pid"], "host_birth": "ended"}),
    }
    for tid, (parent, stamp) in rows.items():
        write_task_result(env.data, tid, STATUS_REQUESTED, parent_task_id=parent, root_task_id=parent,
                          delegation_role="subagent", chat_id=7, result="Awaiting supervisor acceptance.",
                          **({"requested_by": stamp} if stamp else {}))
    with queue._queue_lock:
        queue.PENDING.append({"id": "queued-elsewhere", "root_task_id": "live-parent"})
    report = prep.recover_interrupted_preparations(env.data)
    assert report["settled"] == ["legacy-orphan", "lost"], report
    assert sorted(report["live"]) == ["bus-live", "legacy-held", "queued-elsewhere"]
    for tid, reason in (("lost", prep.REQUEST_LOST), ("legacy-orphan", prep.REQUEST_ORPHANED)):
        row = env.row(tid)
        assert row["status"] == "failed" and row["admission_outcome"] == "never_admitted"
        assert row["reason_code"] == reason
    for tid in ("bus-live", "legacy-held", "queued-elsewhere"):
        assert env.row(tid)["status"] == STATUS_REQUESTED and "admission_outcome" not in env.row(tid)
    notices = [kw.get("progress_meta") or {} for _chat, _text, kw in sent]
    assert {meta.get("subagent_task_id") for meta in notices if meta.get("subagent_event") == "rejected"} == {"lost", "legacy-orphan"}
    assert prep.recover_interrupted_preparations(env.data)["settled"] == []
    with queue._queue_lock:
        queue.PENDING.clear()
