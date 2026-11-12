"""An explicit Resume of a paused or saved Continue returns it to its writer wait (owner, 1 Nov).

The real Continue, Pause, Resume and Stop handlers, the one Resume door, the held-row selection, the Continue's
reconciliation (``recheck_held_continuations``), the restart's kill and restore, the dispatcher and the server's
work projection run here. A Resume over earlier writers that all observably run returns the work to the Continue's
own writer hold: it says what it waits for, offers no Resume, and starts by itself exactly once when they settle,
selecting no child. A newer Pause, a Stop or a restart replaces the wait with its own stop; a fact no wait settles
(an unknown outcome) still refuses; a money stop met after the release is said as such, never as the old wait.

The running server's queue marks reach the disk only on its tick, so the faults below run with that publisher alive
(``_live_tick``) and "the next process" reads the disk as the last publication left it (``_boot_after_crash``). A
refused Resume leaves nothing a crash can restore and dispatch. The host's own start stays the host's wherever a door's
call reaches it: never that door's author, its named decision or the owner's allowance decision.
"""
from __future__ import annotations

import pathlib
import sys
import threading
from types import SimpleNamespace

import pytest

from tests.test_budget_pause_holds import _idle_worker
from tests.test_continue_blocker_diagnostics import RUNNING, _held, _http
from tests.test_continue_independent_review import PRED, _current_pool  # noqa: F401 - pytest fixture
from tests.test_owner_continue import NONCE
from tests.test_presence_moved_lifecycle import root as root  # pytest fixture
from tests.test_resume_attempt_history import (
    _events, _paused_held_continue, _restart_held, _settle_leaf, _stored, _work,
)

LEAF = [{**RUNNING, "task_id": PRED, "review_owned": False}]


def _row(workers, task_id):
    return next((row for row in workers.PENDING if row["id"] == task_id), None)


def _actions(work):
    return {control["action"] for control in work["controls"]}


def _waiting(tmp_path, monkeypatch):
    """The owner's Resume of the paused Continue, its predecessor's leaf still running, returned it to its wait."""
    from ouroboros.gateway.tasks import api_task_resume
    from supervisor import state

    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_k: 5.0)
    queue, workers, successor = _paused_held_continue(tmp_path, monkeypatch, "running_delegated_leaf")
    paused = _work(tmp_path, successor)
    assert paused["activity"] == "paused" and "resume" in _actions(paused), "a paused Continue offers Resume"
    status, ack = _http(api_task_resume, successor)
    assert status == 200 and ack["ok"] and ack["held"] and ack["blockers"] == LEAF, ack
    return queue, workers, successor


def test_a_paused_continue_waits_for_its_earlier_writers_then_starts_exactly_once(tmp_path, monkeypatch):
    from ouroboros import owner_pause
    from ouroboros.task_results import load_task_result
    from ouroboros.work_stop import writer_wait
    from supervisor.continuation_admission import recheck_held_continuations

    queue, workers, successor = _waiting(tmp_path, monkeypatch)
    row = _row(workers, successor)
    assert writer_wait(row) and row["_budget_pause_hold"]["blockers"] == LEAF and "_resume_selection" not in row
    assert not owner_pause.fence_closed(owner_pause.read_fence(tmp_path, successor))
    assert successor not in queue.BUDGET_ROOT_FENCES, "the Pause is released; the wait holds it"
    assert load_task_result(tmp_path, successor)["reason_code"] == "continuation_writer_unsettled"
    assert _stored(tmp_path, successor)["ok"] is True and _stored(tmp_path, successor)["by"] == {"actor": "owner"}
    [resumed] = _events(tmp_path, "owner_pause_resumed")
    assert resumed["held"] is True and resumed["blockers"] == LEAF
    # Waiting, saying what it waits for (the interrupted attempt's run, which nobody can confirm ended), with no
    # Resume or Pause: Stop, Stop of the earlier attempt, and a primary choice before starting.
    work = _work(tmp_path, successor)
    assert (work["activity"], work["because"][0]["code"]) == ("waiting", "earlier_work_unconfirmed")
    assert [(tail["code"], tail["ids"]) for tail in work["tails"]] == [("external_runs_unconfirmed", ["run-leaf"])]
    assert _actions(work) == {"stop", "stop_earlier", "model"} and work["turn"] == [] and "last_resume" not in work
    from ouroboros.gateway.task_model import api_task_model
    status, chosen = _http(api_task_model, successor, {"request_id": "while-held", "action": "set", "model": "owner/chosen"})
    assert status == 200 and chosen["applied"] == "at_start"
    assert row["metadata"]["lead"]["model"] == load_task_result(tmp_path, successor)["metadata"]["lead"]["model"] == "owner/chosen"
    sent = []
    _idle_worker(workers, sent)
    workers.assign_tasks()
    assert sent == [] and recheck_held_continuations() == [], "nothing starts while its earlier writer runs"

    _settle_leaf(tmp_path)
    assert recheck_held_continuations() == [successor]
    assert recheck_held_continuations() == [], "released once"
    assert not writer_wait(row) and row["_resume_selection"]["selected_by"] == "reconciliation"
    workers.assign_tasks()
    workers.assign_tasks()
    assert [task["id"] for task in sent] == [successor], "it starts exactly once"
    assert sent[0]["metadata"]["lead"]["model"] == "owner/chosen"


def test_a_newer_pause_while_it_waits_replaces_the_wait(tmp_path, monkeypatch):
    from ouroboros import owner_pause
    from ouroboros.gateway.task_pause import api_task_pause
    from ouroboros.gateway.tasks import api_task_resume
    from ouroboros.work_stop import writer_wait
    from supervisor.continuation_admission import recheck_held_continuations

    _queue, workers, successor = _waiting(tmp_path, monkeypatch)
    status, pause = _http(api_task_pause, successor, {"request_id": "pause-while-waiting", "actor": "owner"})
    assert status in {200, 202} and pause["ok"], pause
    row = _row(workers, successor)
    assert not writer_wait(row) and owner_pause.fence_closed(owner_pause.read_fence(tmp_path, successor))
    _settle_leaf(tmp_path)
    assert recheck_held_continuations() == [], "its writers settling releases nothing the newer Pause holds"
    sent = []
    _idle_worker(workers, sent)
    workers.assign_tasks()
    assert sent == []
    work = _work(tmp_path, successor)
    assert (work["activity"], work["because"][0]["code"]) == ("paused", "paused_by_owner") and "resume" in _actions(work)
    status, resumed = _http(api_task_resume, successor)
    assert status == 200 and resumed["ok"] and not resumed.get("held"), "its writers settled: Resume releases it"


def test_stop_while_it_waits_ends_it_and_nothing_starts(tmp_path, monkeypatch):
    from ouroboros.gateway.tasks import api_task_cancel
    from ouroboros.task_results import load_task_result
    from supervisor import task_lifecycle
    from supervisor.continuation_admission import recheck_held_continuations

    # The cancelled root's process-wide fence stays this test's: every test here reuses the same successor id.
    monkeypatch.setattr(task_lifecycle, "CANCELLED_ROOT_FENCES", {}, raising=False)
    _queue, workers, successor = _waiting(tmp_path, monkeypatch)
    status, stop = _http(api_task_cancel, successor, {"cascade": True})
    assert status in {200, 202}, stop
    _settle_leaf(tmp_path)
    assert recheck_held_continuations() == []
    sent = []
    _idle_worker(workers, sent)
    workers.assign_tasks()
    assert sent == [] and _row(workers, successor) is None
    assert load_task_result(tmp_path, successor)["status"] == "cancelled"


@pytest.mark.parametrize("door", ["owner_restart", "panic", "app_stop"])
def test_a_restart_while_it_waits_keeps_its_ordinary_return_and_selects_no_child(tmp_path, monkeypatch, door):
    """The restart applies its own rule to the wait, as to any Continue's writer wait: the owner's Restart holds it for
    an explicit Resume and a Panic for the owner's own (``retained_pending``); a plain stop of the application keeps
    the wait, which starts it once its writers settle. That Resume of the SAVED Continue, its writer still running,
    returns it to its wait through the held-row selection; its release starts it and never its held child."""
    from ouroboros import delegate_recovery
    from ouroboros.gateway.tasks import api_task_resume
    from ouroboros.work_stop import hold_fact, stop_record, writer_wait
    from supervisor import active_activity, queue_snapshot
    from supervisor.continuation_admission import recheck_held_continuations

    queue, workers, successor = _waiting(tmp_path, monkeypatch)
    registry = active_activity.DirectActivityRegistry()
    monkeypatch.setattr(active_activity, "get_direct_activity_registry", lambda: registry)
    monkeypatch.setattr(queue_snapshot, "_RESTORED", set())
    monkeypatch.setattr(queue_snapshot, "_RESTORING", {})
    if door == "owner_restart":  # the owner's Restart, published by the generation it stops
        delegate_recovery.prepare_owner_restart(tmp_path, legacy=True)
    elif door == "panic":  # the launcher's Panic input, which the next boot translates into work stops
        (tmp_path / "state").mkdir(exist_ok=True)
        (tmp_path / "state" / "panic_stop.flag").write_text("panic", encoding="utf-8")
    workers.kill_workers(force=True, terminal_status="cancelled", result_reason="Server shutdown.",
                         stop_source="server_shutdown", retain_saved_work=True, preserve_pending=True)
    workers.PENDING[:] = []
    queue.init(tmp_path)
    assert queue.restore_pending_from_snapshot() >= 1
    row = _row(workers, successor)
    restored = hold_fact(row)
    print(door, "restored hold:", restored)
    assert recheck_held_continuations() == [], "nothing starts while its earlier writer runs"
    if door == "panic":
        assert restored["reason"] == "panic_hold" and restored["release"] == "owner" and not writer_wait(row), restored
        _settle_leaf(tmp_path)
        assert recheck_held_continuations() == [], "the Panic replaced the wait: its writers settling starts nothing"
        return
    child = {"id": "succ-child", "type": "task", "chat_id": 7, "root_task_id": successor, "parent_task_id": successor,
             "_attempt": 1, "admitted_dispatch": "none",
             "_budget_pause_hold": stop_record("saved_work_hold", held_at=restored["held_at"],
                                               binding={"root_task_id": successor})}
    workers.PENDING.append(child)
    if door == "owner_restart":
        assert restored["reason"] == "owner_restart_hold" and not writer_wait(row), "the Restart holds it, not the wait"
        work = _work(tmp_path, successor)
        assert work["activity"] == "paused" and "resume" in _actions(work)
        status, ack = _http(api_task_resume, successor)
        assert status == 200 and ack["held"] and ack["blockers"] == LEAF, ack
    assert writer_wait(row) and hold_fact(child)["reason"] == "saved_work_hold"
    _settle_leaf(tmp_path)
    assert recheck_held_continuations() == [successor]
    sent = []
    _idle_worker(workers, sent)
    workers.assign_tasks()
    workers.assign_tasks()
    assert [task["id"] for task in sent] == [successor], "the Continue starts once; its child is not selected"
    assert hold_fact(child) is not None and "_resume_selection" not in child


@pytest.mark.parametrize("blocker", ["running_and_unknown_start", "unbound_unknown_start"])
def test_an_unknown_outcome_is_not_a_writer_wait(tmp_path, monkeypatch, blocker):
    from ouroboros import owner_pause
    from ouroboros.gateway.tasks import api_task_resume
    from ouroboros.work_stop import hold_fact

    _queue, workers, successor = _paused_held_continue(tmp_path, monkeypatch, blocker)
    fence = owner_pause.read_fence(tmp_path, successor)
    status, refusal = _http(api_task_resume, successor)
    assert status == 409 and refusal["error"] == "predecessor_writers_unsettled", refusal
    assert "action" not in refusal and "acceptable_unknown" not in refusal and not refusal.get("held")
    row = _row(workers, successor)
    assert hold_fact(row) is None and owner_pause.read_fence(tmp_path, successor) == fence, "still paused, no wait"
    work = _work(tmp_path, successor)
    assert work["activity"] == "paused" and "resume" in _actions(work) and work["last_resume"]["error"] == refusal["error"]


def test_a_money_stop_after_the_release_is_said_not_the_old_wait(tmp_path, monkeypatch):
    from supervisor import state
    from supervisor.continuation_admission import recheck_held_continuations

    _queue, workers, successor = _waiting(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_k: 0.0)
    sent = []
    _idle_worker(workers, sent)
    workers.assign_tasks()
    row = _row(workers, successor)
    assert "_budget_pause" not in row, "no second pause over the wait"
    assert _work(tmp_path, successor)["because"][0]["code"] == "earlier_work_unconfirmed"
    _settle_leaf(tmp_path)
    assert recheck_held_continuations() == [successor]
    workers.assign_tasks()
    assert sent == [] and row["_budget_pause"]["status"] == "paused_before_dispatch"
    work = _work(tmp_path, successor)
    assert (work["activity"], work["because"][0]["code"]) == ("paused", "spending_limit"), work["because"]
    assert not [reason for reason in work["because"] if reason["code"].startswith("earlier")]
    assert "resume" in _actions(work)


def test_a_minds_resume_returns_it_to_the_wait_under_its_existing_authority(tmp_path, monkeypatch):
    from supervisor.events_mind_controls import handle_mind_task_control
    from ouroboros.work_stop import writer_wait

    _queue, workers, successor = _paused_held_continue(tmp_path, monkeypatch, "running_delegated_leaf")
    pushed = []
    ctx = SimpleNamespace(DRIVE_ROOT=tmp_path, bridge=SimpleNamespace(push_log=pushed.append))
    handle_mind_task_control({"task_id": successor, "action": "resume_task", "actor_task_id": "coordinator-1",
                              "action_nonce": "resume-0002", "reason": "Restore the saved work."}, ctx)
    [outcome] = _events(tmp_path, "mind_task_control_outcome")
    assert outcome["outcome"]["ok"] and outcome["outcome"]["held"] and outcome["outcome"]["blockers"] == LEAF
    assert writer_wait(_row(workers, successor))
    assert _stored(tmp_path, successor)["by"] == {"actor": "ouroboros", "task_id": "coordinator-1"}
    assert _events(tmp_path, "task_resume_outcome") == []


def test_a_start_refused_after_its_writers_settle_is_said_in_place_of_the_wait_and_retried(tmp_path, monkeypatch):
    """Owner 1 Nov: what later keeps it from starting is said, never frozen under the old wait. Its earlier attempt's run
    settles, but the usage ledger cannot be read, so its money cannot be accounted: the host's own start is refused.
    The work says that refusal, dated, in place of the wait's sentence, still waits and offers no Resume; the re-check
    keeps one record of one unchanged refusal; once the ledger reads again it starts by itself, exactly once."""
    from ouroboros import usage_accounting
    from ouroboros.task_store import task_versions
    from supervisor.continuation_admission import recheck_held_continuations

    _queue, workers, successor = _waiting(tmp_path, monkeypatch)
    real, unreadable = usage_accounting.usage_breakdown, {"on": True}

    def ledger(root, *args, task_id=None, **kwargs):
        if unreadable["on"] and task_id == successor:
            raise OSError("usage ledger unreadable")
        return real(root, *args, task_id=task_id, **kwargs)

    monkeypatch.setattr(usage_accounting, "usage_breakdown", ledger)
    _settle_leaf(tmp_path)
    assert recheck_held_continuations() == [], "its writers settled, its start was refused"
    stored = _stored(tmp_path, successor)
    assert (stored["ok"], stored["by"], stored["selected_by"], stored["answer"]["error"]) == (
        False, {"actor": "host"}, "reconciliation", "accounting_unavailable"), stored
    work = _work(tmp_path, successor)
    assert (work["activity"], [reason["code"] for reason in work["because"]]) == ("waiting", ["start_refused"])
    assert work["last_resume"]["by"] == {"actor": "host"} and work["last_resume"]["error"] == "accounting_unavailable"
    assert _actions(work) == {"stop", "model"} and work["turn"] == [], "it tries again by itself: no Resume, not the owner's turn"
    from ouroboros.gateway.task_model import api_task_model
    from ouroboros.task_results import load_task_result
    status, chosen = _http(api_task_model, successor, {"request_id": "before-start", "action": "set", "model": "owner/chosen"})
    assert status == 200 and chosen["applied"] == "at_start"
    assert load_task_result(tmp_path, successor)["metadata"]["lead"]["model"] == "owner/chosen"
    # Without that kept answer the work would still say the wait its settled writers no longer explain.
    assert _work(tmp_path, successor, attempts=False)["because"][0]["code"] == "earlier_work_unconfirmed"
    version, history = task_versions(tmp_path, [successor]), len(_events(tmp_path, "task_resume_outcome"))
    assert recheck_held_continuations() == []
    assert task_versions(tmp_path, [successor]) == version, "an unchanged refusal is kept once"
    assert len(_events(tmp_path, "task_resume_outcome")) == history
    unreadable["on"] = False
    assert recheck_held_continuations() == [successor]
    kept = _stored(tmp_path, successor)
    assert kept["ok"] is True and kept["by"] == {"actor": "host"}, "the start ends the refusal it said"
    [*_owner, refused, started] = _events(tmp_path, "task_resume_outcome")
    assert (refused["actor"], refused["outcome"]["error"], started["outcome"]["ok"]) == (
        "host", "accounting_unavailable", True), "the refusal stays in the history"
    sent = []
    _idle_worker(workers, sent)
    workers.assign_tasks()
    workers.assign_tasks()
    assert [task["id"] for task in sent] == [successor], "it starts exactly once"


def _live_tick(monkeypatch):
    """The running server: a queue mark only schedules its publication, which the tick (or a barrier) writes."""
    from supervisor import queue_publication

    monkeypatch.setattr(queue_publication, "_PUBLISHER", threading.current_thread())
    assert queue_publication.tick_publisher_alive()


def _boot_after_crash(tmp_path, monkeypatch, workers, queue):
    """The next process: nothing in memory, the queue as the last publication left it on disk."""
    from supervisor import active_activity, queue_snapshot

    registry = active_activity.DirectActivityRegistry()
    monkeypatch.setattr(active_activity, "get_direct_activity_registry", lambda: registry)
    monkeypatch.setattr(queue_snapshot, "_RESTORED", set())
    monkeypatch.setattr(queue_snapshot, "_RESTORING", {})
    workers.PENDING[:] = []
    queue.BUDGET_ROOT_FENCES.clear()
    queue.init(tmp_path)
    assert queue.restore_pending_from_snapshot() >= 1


def _on_disk(tmp_path, task_id):
    """The hold the published queue keeps for this row, as a restart would read it."""
    from ouroboros.queue_observation import load_queue_document

    [row] = [row for row in load_queue_document(tmp_path, fresh=True).get("pending") or () if row["id"] == task_id]
    return row["task"].get("_budget_pause_hold") or {}


def _saved_continue(tmp_path, monkeypatch, *, settled=False):
    """The owner's Restart holds a Continue whose predecessor's leaf runs (or has settled), under a live tick."""
    from ouroboros import delegate_recovery
    from ouroboros.gateway.task_continue import api_task_continue
    from supervisor import state

    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_k: 5.0)
    queue, workers = _held(tmp_path, monkeypatch, "running_delegated_leaf")
    status, ack = _http(api_task_continue, PRED, {"action_nonce": NONCE})
    assert status == 200 and ack["held"]
    successor = ack["successor_task_id"]
    _restart_held(tmp_path, monkeypatch, workers, queue)
    # That boot took the Restart's door up (``worker_chat_lane``): a later crash is an application stop of its own.
    delegate_recovery.consume_owner_restart(tmp_path, delegate_recovery.pending_owner_restart(tmp_path))
    if settled:
        _settle_leaf(tmp_path)
    _live_tick(monkeypatch)
    hold = dict(_row(workers, successor)["_budget_pause_hold"])
    assert hold["reason"] == "owner_restart_hold" and _on_disk(tmp_path, successor) == hold
    return queue, workers, successor, hold


def _decision_unreadable(monkeypatch):
    """The owner's decision cannot read the task's record: the one read that door makes, exactly, fails."""
    from ouroboros import task_results

    real = task_results.load_task_result

    def read(root, task_id, *args, **kwargs):
        if sys._getframe(1).f_code.co_name == "owner_resume_decision":
            raise OSError("record unreadable")
        return real(root, task_id, *args, **kwargs)

    monkeypatch.setattr(task_results, "load_task_result", read)


def _nothing_starts(tmp_path, workers):
    """Its writers settle, the re-check runs and a worker stands idle: what the restored row does then."""
    from supervisor.continuation_admission import recheck_held_continuations

    _settle_leaf(tmp_path)
    released, sent = recheck_held_continuations(), []
    _idle_worker(workers, sent)
    workers.assign_tasks()
    workers.assign_tasks()
    assert (released, [task["id"] for task in sent]) == ([], []), "a refused Resume released or started it"


@pytest.mark.parametrize("settled", [False, True], ids=["into_its_wait", "selected_to_start"])
def test_a_refused_resume_leaves_nothing_a_crash_can_restore_and_start(tmp_path, monkeypatch, settled):
    """Both routes out of a saved Continue's stop: the owner's decision refuses while only the stop is on disk, so the
    refusal has no withdrawal to lose. The next process restores that stop; its writers settling starts nothing."""
    from ouroboros.gateway.tasks import api_task_resume
    from supervisor import queue as queue_module

    queue, workers, successor, hold = _saved_continue(tmp_path, monkeypatch, settled=settled)
    published = []
    real_publish = queue_module.publish_queue_snapshot
    with monkeypatch.context() as patched:
        patched.setattr(queue_module, "publish_queue_snapshot", lambda **k: published.append(k) or real_publish(**k))
        _decision_unreadable(patched)
        status, refusal = _http(api_task_resume, successor)
    assert status == 409 and refusal["error"] == "grant_not_recorded", refusal
    assert _row(workers, successor)["_budget_pause_hold"] == hold
    landed = _on_disk(tmp_path, successor)

    _boot_after_crash(tmp_path, monkeypatch, workers, queue)
    _nothing_starts(tmp_path, workers)
    assert published == [] and landed == hold, "nothing the refusal would have to withdraw was published"
    assert _row(workers, successor)["_budget_pause_hold"] == hold
    work = _work(tmp_path, successor)
    assert (work["activity"], work["because"][0]["code"]) == ("paused", "after_restart") and "resume" in _actions(work)
    assert work["last_resume"]["error"] == "grant_not_recorded", "the same stop keeps its dated refusal"
    # The repeated Resume, its record readable again, is an ordinary one, and starts it once.
    status, resumed = _http(api_task_resume, successor)
    assert status == 200 and resumed["ok"] and not resumed.get("held"), resumed
    sent = []
    _idle_worker(workers, sent)
    workers.assign_tasks()
    workers.assign_tasks()
    assert [task["id"] for task in sent] == [successor]


def test_a_resume_into_its_wait_is_on_disk_before_it_answers_and_a_crash_keeps_the_wait(tmp_path, monkeypatch):
    from ouroboros.gateway.tasks import api_task_resume
    from ouroboros.work_stop import writer_wait
    from supervisor.continuation_admission import recheck_held_continuations

    queue, workers, successor, _hold = _saved_continue(tmp_path, monkeypatch)
    status, ack = _http(api_task_resume, successor)
    assert status == 200 and ack["held"] and ack["blockers"] == LEAF, ack
    assert _on_disk(tmp_path, successor) == _row(workers, successor)["_budget_pause_hold"]
    assert writer_wait(_row(workers, successor))
    _boot_after_crash(tmp_path, monkeypatch, workers, queue)
    assert writer_wait(_row(workers, successor))
    assert recheck_held_continuations() == [], "its writer still runs"
    _settle_leaf(tmp_path)
    assert recheck_held_continuations() == [successor]
    sent = []
    _idle_worker(workers, sent)
    workers.assign_tasks()
    workers.assign_tasks()
    assert [task["id"] for task in sent] == [successor], "it starts exactly once"


@pytest.mark.parametrize(("settled", "fault"), [
    pytest.param(False, "nothing_landed", id='into_its_wait-nothing_landed'),
    pytest.param(False, "part_landed_and_no_write_after", id='into_its_wait-part_landed_and_no_write_after'),
    pytest.param(True, "part_landed_then_retried", id='selected_to_start-part_landed_then_retried'),
])
def test_a_publication_that_fails_returns_the_stop_and_the_disk_follows(tmp_path, monkeypatch, fault, settled):
    """A failed write can land part of its bundle (``queue_publication._publish``: here the snapshot lands and its
    certificate does not; a document whose bytes did not change is not written at all), so the door puts the stop
    back over it at once. When that write fails too, the row in memory is still the stop and the tick's owed
    rewrite lands it."""
    from ouroboros.gateway.tasks import api_task_resume
    from supervisor import queue_publication
    from supervisor.queue_publication import publish_queue_changes

    queue, workers, successor, hold = _saved_continue(tmp_path, monkeypatch, settled=settled)
    real_write, real_alive, writes = queue.atomic_write_text, queue_publication._write_alive, []

    def write(target, content, *args, **kwargs):
        writes.append(pathlib.Path(target).name)
        refused = fault == "nothing_landed" or (fault == "part_landed_and_no_write_after" and len(writes) > 1)
        return False if refused else real_write(target, content, *args, **kwargs)

    def certify(target, content):
        writes.append("queue.alive")
        refused = (fault == "part_landed_and_no_write_after"
                   or (fault == "part_landed_then_retried" and writes.count("queue.alive") == 1))
        return False if refused else real_alive(target, content)

    with monkeypatch.context() as patched:
        patched.setattr(queue, "atomic_write_text", write)
        patched.setattr(queue_publication, "_write_alive", certify)
        status, refusal = _http(api_task_resume, successor)
    assert status == 409 and refusal["error"] == "snapshot_not_persisted", refusal
    assert _row(workers, successor)["_budget_pause_hold"] == hold
    if fault == "nothing_landed":
        assert _on_disk(tmp_path, successor) == hold, "the old document stands"
    elif fault == "part_landed_then_retried":
        assert _on_disk(tmp_path, successor) == hold, "the withdrawal went straight back over the part that landed"
    else:
        assert _on_disk(tmp_path, successor) != hold, "the part that landed is what the disk says until a write lands"
        assert publish_queue_changes() is True
        assert _on_disk(tmp_path, successor) == hold, "the tick's owed rewrite recovers the original stop"
    _boot_after_crash(tmp_path, monkeypatch, workers, queue)
    _nothing_starts(tmp_path, workers)
    assert _row(workers, successor)["_budget_pause_hold"] == hold


@pytest.mark.parametrize("settled", [False, True], ids=["into_its_wait", "selected_to_start"])
def test_a_crash_before_any_write_lands_after_a_torn_publication_still_starts_nothing(tmp_path, monkeypatch, settled):
    """The worst of it: the row a refused door published landed alone, every write after it (its certificate first)
    failed, and the process dies before its tick. No certificate names that bundle, so the restart keeps no wait
    from it: only a Resume starts it."""
    from ouroboros.gateway.tasks import api_task_resume
    from supervisor import queue_publication

    queue, workers, successor, hold = _saved_continue(tmp_path, monkeypatch, settled=settled)
    real_write, writes = queue.atomic_write_text, []

    def write(target, content, *args, **kwargs):
        writes.append(pathlib.Path(target).name)
        return False if len(writes) > 1 else real_write(target, content, *args, **kwargs)

    with monkeypatch.context() as patched:
        patched.setattr(queue, "atomic_write_text", write)
        patched.setattr(queue_publication, "_write_alive", lambda *_a, **_k: False)
        status, refusal = _http(api_task_resume, successor)
    assert status == 409 and refusal["error"] == "snapshot_not_persisted", refusal
    assert _on_disk(tmp_path, successor) != hold, "the part that landed is what the disk says"
    _boot_after_crash(tmp_path, monkeypatch, workers, queue)
    _nothing_starts(tmp_path, workers)
    assert _row(workers, successor)["_budget_pause_hold"]["reason"] == "saved_work_hold"
    assert "resume" in _actions(_work(tmp_path, successor))
    status, resumed = _http(api_task_resume, successor)
    assert status == 200 and resumed["ok"], resumed
    sent = []
    _idle_worker(workers, sent)
    workers.assign_tasks()
    assert [task["id"] for task in sent] == [successor], "the owner's Resume still starts it"


@pytest.mark.parametrize("withdrawal", ["landed", "refused"])
def test_a_refused_release_of_an_unstarted_root_stays_behind_its_closed_fence(tmp_path, monkeypatch, withdrawal):
    """The other route into the wait: its publication comes first while the durable fence stays closed, so whichever
    of them a crash finds on disk, the Continue it refused cannot launch, and its fence still offers Resume."""
    from ouroboros import owner_pause
    from ouroboros.gateway.tasks import api_task_resume
    from supervisor import queue as queue_module
    from supervisor import state

    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_k: 5.0)
    queue, workers, successor = _paused_held_continue(tmp_path, monkeypatch, "running_delegated_leaf")
    _live_tick(monkeypatch)
    real_publish = queue_module.publish_queue_snapshot

    def publish(*, reason=""):
        return False if withdrawal == "refused" and reason == "owner_pause_root_resume_held" else real_publish(reason=reason)

    def refused_fence(*_a, **_k):
        raise OSError("fence unwritable")

    with monkeypatch.context() as patched:
        patched.setattr(queue_module, "publish_queue_snapshot", publish)
        patched.setattr(owner_pause, "set_fence_state", refused_fence)
        status, refusal = _http(api_task_resume, successor)
    assert status != 200 and refusal["error"] == "owner_pause_fence_unwritable", refusal
    _boot_after_crash(tmp_path, monkeypatch, workers, queue)
    assert owner_pause.fence_closed(owner_pause.read_fence(tmp_path, successor))
    _nothing_starts(tmp_path, workers)
    assert "resume" in _actions(_work(tmp_path, successor))


def _ledger_unreadable(monkeypatch, successor):
    from ouroboros import usage_accounting

    real = usage_accounting.usage_breakdown

    def ledger(root, *args, task_id=None, **kwargs):
        if task_id == successor:
            raise OSError("usage ledger unreadable")
        return real(root, *args, task_id=task_id, **kwargs)

    monkeypatch.setattr(usage_accounting, "usage_breakdown", ledger)


def _says_the_hosts_refusal(tmp_path, successor):
    stored = _stored(tmp_path, successor)
    assert (stored["by"], stored["selected_by"], stored["answer"]["error"]) == (
        {"actor": "host"}, "reconciliation", "accounting_unavailable"), stored
    [event] = [row for row in _events(tmp_path, "task_resume_outcome") if row["outcome"].get("error")]
    assert event["actor"] == "host" and event["task_id"] == successor
    work = _work(tmp_path, successor)
    assert [reason["code"] for reason in work["because"]] == ["start_refused"], work["because"]
    assert work["last_resume"]["by"] == {"actor": "host"} and "resume" not in _actions(work)


@pytest.mark.parametrize("nested", ["owner_door", "mind_control"])
def test_the_hosts_start_is_the_hosts_inside_a_door_it_did_not_answer(tmp_path, monkeypatch, nested):
    """A door's call can reach this release: it asked about another task, so what the release answers is the host's."""
    from ouroboros.control_actor import control_actor, mind_control, owner_door, owner_door_bound
    from supervisor.continuation_admission import release_settled_continuations

    _queue, _workers, successor = _waiting(tmp_path, monkeypatch)
    _ledger_unreadable(monkeypatch, successor)
    _settle_leaf(tmp_path)
    door = owner_door("api_task_resume") if nested == "owner_door" else mind_control("coordinator-1", "resume the earlier one")
    with door:
        bound = control_actor(), owner_door_bound()
        assert release_settled_continuations(PRED) == []
        assert (control_actor(), owner_door_bound()) == bound, "the door keeps its own binding afterwards"
    _says_the_hosts_refusal(tmp_path, successor)


@pytest.mark.parametrize("door", ["owner", "mind"])
def test_a_resume_of_the_answered_predecessor_leaves_the_continues_start_the_hosts(tmp_path, monkeypatch, door):
    """Through the real doors: the Resume of an answered root's Pause (``resume_late_phase``) re-checks its Continue."""
    from ouroboros import owner_pause
    from ouroboros.gateway.tasks import api_task_resume
    from ouroboros.task_results import STATUS_CANCELLED, write_task_result
    from ouroboros.work_stop import writer_wait
    from supervisor.events_mind_controls import handle_mind_task_control

    _queue, workers, successor = _waiting(tmp_path, monkeypatch)
    _ledger_unreadable(monkeypatch, successor)
    _settle_leaf(tmp_path)
    write_task_result(tmp_path, PRED, STATUS_CANCELLED, result="interrupted")
    fence, created = owner_pause.install_fence(tmp_path, PRED, request_id="pause-predecessor", late_work=True)
    assert created and owner_pause.fence_closed(fence)
    if door == "owner":
        status, outcome = _http(api_task_resume, PRED)
        assert status == 200 and outcome["ok"] and "late_phase" in outcome, outcome
        assert _stored(tmp_path, PRED)["by"] == {"actor": "owner"}
    else:
        ctx = SimpleNamespace(DRIVE_ROOT=tmp_path, bridge=SimpleNamespace(push_log=lambda _event: None))
        handle_mind_task_control({"task_id": PRED, "action": "resume_task", "actor_task_id": "coordinator-1",
                                  "action_nonce": "resume-0003", "reason": "Restore the saved work."}, ctx)
        [event] = _events(tmp_path, "mind_task_control_outcome")
        assert event["outcome"]["ok"] and "late_phase" in event["outcome"], event
        assert _stored(tmp_path, PRED)["by"] == {"actor": "ouroboros", "task_id": "coordinator-1"}
    _says_the_hosts_refusal(tmp_path, successor)
    assert writer_wait(_row(workers, successor)), "the wait and its re-check go on"


def test_the_callers_named_decision_stays_with_the_door_it_was_named_at(tmp_path, monkeypatch):
    """The named decision to go on past held calls belongs to the call's own census (``action_writers``): the host's
    release inside that call neither applies it nor records it under anyone's name."""
    from ouroboros import local_custody_repair
    from ouroboros.control_actor import accepting_unknown, owner_door
    from supervisor.continuation_admission import action_writers, release_settled_continuations

    _queue, _workers, _successor = _waiting(tmp_path, monkeypatch)
    applied = []
    monkeypatch.setattr(local_custody_repair, "accept_unknown_custody",
                        lambda _root, _members, ids, **_k: applied.append(list(ids)) or {"accepted": [], "refused": []})
    with owner_door("api_task_resume"), accepting_unknown(["op-1"], "the owner read it"):
        assert release_settled_continuations(PRED) == []
        assert applied == [], "the host's census goes on past nothing"
        action_writers(_queue, PRED)
    assert applied == [["op-1"]], "the door's own census still applies it"


def _moved_continue(root, monkeypatch, *, settled=False):
    """A moved autonomous Presence turn with its producer allowance, a Continue under an application stop."""
    from tests.test_continue_independent_review import _rows
    from tests.test_owner_continue import _interrupted
    from tests.test_presence_moved_allowance import _behind_grantless_door, _focus_built
    from supervisor import state

    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 5.0)
    _interrupted(root, PRED, origin={"source": "owner_restart", "reason": "owner_restart", "scope": "single"})
    _rows(root, leaf=True, review=False)
    if settled:
        _settle_leaf(root)
    built = _focus_built(root, monkeypatch)
    row = _behind_grantless_door(root, built, built["id"], "held")
    row["metadata"]["continuation"] = {"predecessor_task_id": PRED}
    return row, built["id"]


def _allowance(root, row, task_id):
    from ouroboros.task_pacing import recorded_producer_ceiling
    from ouroboros.task_results import load_task_result

    return recorded_producer_ceiling(row), recorded_producer_ceiling(load_task_result(root, task_id))


@pytest.mark.parametrize("fault", ["none", "store_refuses", "publication_fails", "publication_and_store_fail"])
def test_the_owners_decision_stands_only_with_the_wait_it_decided_for(root, monkeypatch, fault):
    """#2566 on the wait's route: the owner's own Resume decides first. A store that refuses moves nothing and nothing is
    published; a publication that fails takes the decision back; only when its record cannot be written either does the
    decision stay there, no release standing, and the queue row keeps its number."""
    from ouroboros import task_store
    from ouroboros.work_stop import hold_fact, writer_wait
    from supervisor import queue as queue_module
    from tests.test_presence_moved_allowance import CEILING, _resume_as

    row, task_id = _moved_continue(root, monkeypatch)
    hold = dict(hold_fact(row))
    with monkeypatch.context() as patched:
        real_patch = task_store.patch_document
        written = []

        def patch(drive, tid, mutator):
            written.append(tid)
            if fault == "store_refuses" or (fault == "publication_and_store_fail" and len(written) > 1):
                raise OSError("store refused")
            return real_patch(drive, tid, mutator)

        patched.setattr(task_store, "patch_document", patch)
        if fault.startswith("publication"):
            patched.setattr(queue_module, "publish_queue_snapshot", lambda **_k: False)
        answer = _resume_as("owner_in_process", task_id)
    if fault == "none":
        assert answer["ok"] and answer["held"] and writer_wait(row), answer
        assert _allowance(root, row, task_id) == (None, None)
        assert row["metadata"]["producer_allowance_lifted"]["by"]["door"] == "api_task_resume"
        return
    assert answer["ok"] is False and answer["error"] == (
        "grant_not_recorded" if fault == "store_refuses" else "snapshot_not_persisted"), answer
    assert hold_fact(row) == hold and not writer_wait(row)
    assert "producer_allowance_lifted" not in row["metadata"]
    durable_ceiling = None if fault == "publication_and_store_fail" else CEILING
    assert _allowance(root, row, task_id) == (CEILING, durable_ceiling)


def test_a_crash_between_the_decision_and_the_wait_restores_the_stop_with_the_decision(root, monkeypatch):
    """The process dies past the decision, before its publication: the disk says the stop, the record the owner's
    decision (a snapshot older than it: ``adopt_saved_records``). The stop stands, so nothing starts without a Resume."""
    from ouroboros.work_stop import hold_fact
    from supervisor import queue as queue_module
    from supervisor import workers
    from tests.test_presence_moved_allowance import _Crash, _resume_as

    row, task_id = _moved_continue(root, monkeypatch)
    hold = dict(hold_fact(row))
    assert queue_module.publish_queue_snapshot(reason="test_saved") is True
    _live_tick(monkeypatch)

    def dies(**_k):
        raise _Crash()

    with monkeypatch.context() as patched:
        patched.setattr(queue_module, "publish_queue_snapshot", dies)
        with pytest.raises(_Crash):
            _resume_as("owner_in_process", task_id)
    assert _on_disk(root, task_id) == hold
    _boot_after_crash(root, monkeypatch, workers, queue_module)
    restored = _row(workers, task_id)
    assert hold_fact(restored) == hold
    assert _allowance(root, restored, task_id) == (None, None), "the decision it recorded is the record's"
    _nothing_starts(root, workers)


@pytest.mark.parametrize("settled", [False, True], ids=["into_its_wait", "selected_to_start"])
def test_the_owners_own_resume_of_the_successor_still_lifts_and_the_hosts_release_inside_a_door_never_does(
        root, monkeypatch, settled):
    """The allowance decision needs the owner's own Resume OF THIS WORK. The host's release of its wait, even inside the
    owner's Resume of another task, keeps the allowance; the owner's Resume of the Continue itself lifts it on either
    route out of its stop."""
    from ouroboros.control_actor import owner_door
    from ouroboros.work_stop import writer_wait
    from supervisor import workers
    from supervisor.continuation_admission import release_settled_continuations
    from supervisor.events_budget import hold_continuation_writers
    from tests.test_presence_moved_allowance import CEILING, _focus_built, _resume_as, _saved_record_row, _stored_turn

    row, task_id = _moved_continue(root, monkeypatch, settled=settled)
    answer = _resume_as("owner_in_process", task_id)
    assert answer["ok"] and bool(answer.get("held")) is not settled, answer
    assert writer_wait(row) is not settled
    assert _allowance(root, row, task_id) == (None, None), "the owner's own Resume of the work lifts it"

    built = _focus_built(root, monkeypatch)
    automatic_id = built["id"] + "-automatic"
    automatic = _saved_record_row(built, automatic_id, _stored_turn(root, built, automatic_id), admitted_dispatch="none")
    automatic["metadata"]["continuation"] = {"predecessor_task_id": PRED}
    workers.PENDING.append(automatic)
    hold_continuation_writers(automatic, PRED, [])
    _settle_leaf(root)
    with owner_door("api_task_resume"):
        assert automatic_id in release_settled_continuations(PRED)
    assert not writer_wait(automatic) and automatic["_resume_selection"]["selected_by"] == "reconciliation"
    assert _allowance(root, automatic, automatic_id) == (CEILING, CEILING), "the host's release lifts nothing"
    assert "producer_allowance_lifted" not in automatic["metadata"]
