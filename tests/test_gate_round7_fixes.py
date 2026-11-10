"""GATE ROUND-7 (v6.98.0 phase A) — closure-verification regression tests.

GR7-1  ingress gaps of the GR6-1 live-ownership predicate: (a) the worker-side
       queue-snapshot twin fails OPEN toward liveness — a missing, unreadable,
       or STALE snapshot cannot prove a dead worker, so the agent cancel tool
       mints the intent instead of answering "Nothing to cancel" over a
       burning worker; (b) the HTTP cascade Stop asks the same live-ownership
       predicate once its intent request found the root settled — a
       settled-but-LIVE root mints + runs custody, a settled-AND-dead tree
       keeps the 404 envelope;
GR7-2  verbatim preservation order in the settled-capture kill lane: the
       settled short-circuit runs ABOVE every mutating step (child copy-back,
       artifact finalize, memory export) — the stored terminal row survives
       BYTE-IDENTICAL, and a split-drive child answer never replaces the
       canonical settled answer;
GR7-3  project-deletion wind-down only RE-CHECKS quiescence — it never re-runs
       the cancel pass over a purely settled-lingering set (each re-mint
       delivered a duplicate owner summary); a stuck/new non-settled task
       still gets re-cancelled;
GR7-4  stable delivery identity for disclosure-bearing single-task messages:
       the mutable unreconciled-runs note rides the TEXT but never the id, so
       a replay whose rebuilt note shrank dedups to one delivery;
GR7-5  a quarantined malformed intent row emits its typed disclosure EVENT
       once per row content (in-process memo); the log.error stays per sweep.
"""

from __future__ import annotations

import time
import types

import pytest


from ouroboros import cancel_intents as ci
from ouroboros.task_results import (
    STATUS_CANCELLED,
    STATUS_COMPLETED,
    load_task_result,
    write_task_result,
)


class _CaptureQueue:
    def __init__(self):
        self.events = []

    def put(self, evt):
        self.events.append(evt)


@pytest.fixture()
def qenv(tmp_path, monkeypatch):
    import supervisor.queue as q
    from supervisor import task_lifecycle, workers

    monkeypatch.setattr(q, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(q, "PENDING", [])
    monkeypatch.setattr(q, "RUNNING", {}, raising=False)
    monkeypatch.setattr(workers, "WORKERS", {}, raising=False)
    monkeypatch.setattr(workers, "respawn_worker", lambda wid: None, raising=False)
    monkeypatch.setattr(q, "persist_queue_snapshot", lambda reason="": None)
    monkeypatch.setattr(task_lifecycle, "CANCELLED_ROOT_FENCES", {}, raising=False)
    monkeypatch.setattr(task_lifecycle, "_ACTIVE_CASCADE_FENCES", {}, raising=False)
    return types.SimpleNamespace(q=q, tl=task_lifecycle, workers=workers, drive=tmp_path)


# --------------------------------------------------------------------------
# GR7-1a — the worker-side twin fails OPEN toward liveness
# --------------------------------------------------------------------------
# GR7-1b — a settled HTTP cascade target consults live ownership
# --------------------------------------------------------------------------
# GR7-2 — verbatim preservation on the settled-capture kill lane
# --------------------------------------------------------------------------
# GR7-3 — wind-down re-checks quiescence without re-minting
# --------------------------------------------------------------------------


def _patch_project_registry(monkeypatch):
    import ouroboros.projects_registry as pr

    done: dict = {"complete": [], "fail": []}
    monkeypatch.setattr(pr, "project_task_bindings", lambda root: {})
    monkeypatch.setattr(
        pr, "complete_project_deletion",
        lambda root, pid: done["complete"].append(str(pid)),
    )
    monkeypatch.setattr(
        pr, "fail_project_deletion",
        lambda root, pid, detail: done["fail"].append((str(pid), str(detail))),
    )
    return done


def _record_mints(monkeypatch):
    mints: list = []
    real_mint = ci.request_cancel

    def _recording_mint(drive, task_id, **kw):
        mints.append(str(task_id))
        return real_mint(drive, task_id, **kw)

    monkeypatch.setattr(ci, "request_cancel", _recording_mint)
    return mints


def test_gr7_3_stuck_nonsettled_task_still_gets_recancelled(qenv, monkeypatch):
    """A non-settled task appearing in the remaining set re-enters the cancel
    pass (targeted at the roots covering it), while the settled-lingering
    root is never re-minted."""
    from supervisor import queue_transitions as qt
    from supervisor import workers

    done = _patch_project_registry(monkeypatch)
    queue = _CaptureQueue()
    monkeypatch.setattr(workers, "get_event_q", lambda *, create=True: queue, raising=False)
    monkeypatch.setattr(qenv.q, "_emit_cancel_task_done", lambda *a, **kw: None)

    root_id = "w7b-root"
    qenv.q.RUNNING[root_id] = {
        "task": {"id": root_id, "chat_id": 8, "project_id": "p7b",
                 "root_task_id": root_id},
        "worker_id": 0,
    }
    write_task_result(qenv.drive, root_id, STATUS_COMPLETED, chat_id=8,
                      result="done; winding down")
    mints = _record_mints(monkeypatch)

    sleeps: list = []

    def _inject_then_release(seconds):
        sleeps.append(seconds)
        if len(sleeps) == 1:
            # A NEW non-settled task appears mid-wind-down.
            qenv.q.PENDING.append(
                {"id": "w7b-new", "chat_id": 8, "project_id": "p7b"},
            )
        if len(sleeps) >= 2:
            qenv.q.RUNNING.pop(root_id, None)

    monkeypatch.setattr(time, "sleep", _inject_then_release)

    qt.run_project_deletion(qenv.drive, "p7b", 8)

    assert done["complete"] == ["p7b"] and not done["fail"]
    assert load_task_result(qenv.drive, "w7b-new")["status"] == STATUS_CANCELLED, (
        "GR7-3: the stuck/new non-settled task is still re-cancelled"
    )
    assert mints.count(root_id) == 1, (
        "the settled-lingering root is never re-minted by the re-entered pass"
    )
    assert mints.count("w7b-new") >= 1


# --------------------------------------------------------------------------
# GR7-4 — stable delivery identity: the note rides the text, never the id
# --------------------------------------------------------------------------


def test_gr7_4_completed_delivery_dedups_across_a_changed_note(tmp_path):
    """The probe: ["run-a"] → id …X; after reconcile, [] → id …Y — both owed.
    The id must digest the CORE answer only."""
    from supervisor import terminal_delivery as td

    stored = {"status": "completed", "result": "the answer"}
    task = {"id": "sid7", "chat_id": 4}
    first = td.build_completed_result_event(
        tmp_path, task, "sid7", stored, unreconciled_runs=["run-a"],
    )
    replay = td.build_completed_result_event(
        tmp_path, task, "sid7", stored, unreconciled_runs=[],
    )
    assert first["text"] == replay["text"] == "the answer", "the answer is the model's words"
    assert "run-a" in first["terminal_custody_notice"] and "terminal_custody_notice" not in replay, (
        "the disclosure rides the custody row (#1006)"
    )
    assert first["delivery_id"] == replay["delivery_id"], (
        "GR7-4: identity comes from the stable core, never the mutable note"
    )
    # End to end: owed → sent → replay with the shrunk note dedups to nothing.
    assert td.register_pending_delivery(tmp_path, first) is True
    assert td.register_delivery(tmp_path, first["delivery_id"]) is True
    queue = _CaptureQueue()
    assert td.deliver_completed_result(
        tmp_path, task, "sid7", stored, event_queue=queue, unreconciled_runs=[],
    ) is False
    assert queue.events == []
    assert td.pending_deliveries(tmp_path) == [], "one delivery, nothing re-owed"


# --------------------------------------------------------------------------
# GR7-5 — quarantine disclosure: typed event once per row, log every sweep
# --------------------------------------------------------------------------
