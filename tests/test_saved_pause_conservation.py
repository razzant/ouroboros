"""An unavailable result cannot turn accepted saved work into a failed pending task."""
from __future__ import annotations

import copy
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

import pytest

from ouroboros import budget_pause, store, task_results
from ouroboros.artifacts import read_actor_source_bytes
from supervisor import queue_snapshot
from tests._budget_pause_exact_helpers import _install_queue, _pause
from tests._store_journal_shared import exclusive_hold
from tests._store_shared import unreadable_result
from tests.test_owner_wait_restart import restart_case  # noqa: F401


def _saved_pause(root, monkeypatch, workers, *, project=False):
    ctx, _limit, pause = _pause(root, monkeypatch, task_id="saved", scope="task",
                               rail=budget_pause.RAIL_BILLING_AUTHORITY)
    budget_pause.end_dispatch_fence(ctx.task_id)
    row = budget_pause.set_budget_pause(root, ctx.task_id, {**pause, "state": budget_pause.STATE_PAUSED})
    assert row["reason"] == "authority"
    task = {"id": ctx.task_id, "type": "task", "chat_id": 0, "root_task_id": ctx.task_id,
            "_attempt": 1, "_budget_pause": budget_pause.exact_pause_marker(row)}
    if project:
        task["project_id"] = "project"
    workers.PENDING.append(task)
    return task, row, read_actor_source_bytes(root, ctx.task_id, row["source_ref"])


@contextmanager
def _result_gap(root, monkeypatch, kind, *, task_id="saved"):
    with monkeypatch.context() as patch:
        if kind == "unreadable":
            unreadable_result(patch, task_id, root=root)
        else:
            # Only this result observation is contended. Cancellation authority
            # and the snapshot remain readable, as in the real pending failure.
            # The observation is the row's document and its typed control columns alike
            # (``store_hot_facts.read`` serves the pause row without decoding the document).
            from ouroboros import store_hot_facts

            calls = []

            def contended(original):
                def busy_read(drive, tid, **kwargs):
                    if str(tid) != task_id:
                        return original(drive, tid, **kwargs)
                    temporary_root = Path(os.environ.get("OUROBOROS_TEST_TEMP_ROOT") or tempfile.gettempdir())
                    assert root.resolve().is_relative_to(temporary_root.resolve())
                    conn = exclusive_hold(store.store_path(root))  # under WAL too, the read is busy
                    try:
                        with store.acquisition_budget(0):
                            return original(drive, tid, **kwargs)
                    except store.StoreBusy as exc:
                        assert exc.reason == "contention" and exc.rollback_confirmed
                        calls.append(exc)
                        raise
                    finally:
                        conn.execute("ROLLBACK")
                        conn.close()
                return busy_read

            patch.setattr(task_results, "load_task_result", contended(task_results.load_task_result))
            patch.setattr(store_hot_facts, "read", contended(store_hot_facts.read))
        yield
        if kind == "busy":
            assert calls, "the consumer must actually encounter SQLite contention"


@pytest.mark.parametrize("entry", ["live", "restore"])
@pytest.mark.parametrize("gap", ["busy", "unreadable"])
@pytest.mark.parametrize("project", [False, True], ids=["unscoped", "project"])
def test_saved_technical_pause_survives_result_gap_then_explicit_resume(tmp_path, monkeypatch, entry, gap, project):
    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 5.0)
    task, pause, source = _saved_pause(tmp_path, monkeypatch, workers, project=project)
    original_marker = copy.deepcopy(task["_budget_pause"])
    if entry == "restore":
        assert queue.persist_queue_snapshot(reason="saved-pause-test")
        workers.PENDING.clear()
    with _result_gap(tmp_path, monkeypatch, gap):
        if entry == "restore":
            assert queue.restore_pending_from_snapshot() == 1
        else:
            with queue._queue_lock:
                assert workers._drop_cancelled_pending()
        [kept] = workers.PENDING
        assert kept["id"] == "saved" and kept["_attempt"] == 1
        assert {k: v for k, v in kept["_budget_pause"].items() if k != "refusal"} == original_marker
        assert not kept.get("_terminalization_retry")
        assert not kept.get("_project_admission_restore_hold")
        assert not kept.get("_budget_pause_resume")
    with queue._queue_lock:
        assert workers._drop_cancelled_pending()
    assert workers.PENDING == [kept] and not kept.get("_budget_pause_resume")
    assert read_actor_source_bytes(tmp_path, "saved", pause["source_ref"]) == source
    assert task_results.load_task_result(tmp_path, "saved")["status"] != "failed"
    resumed = queue.resume_budget_paused_task("saved")
    assert resumed["ok"], resumed
    assert kept["_budget_pause_resume"]["source_ref"] == pause["source_ref"]
    assert kept["_budget_pause_resume"]["pause_id"] == pause["pause_id"]


@pytest.mark.parametrize("marker", [None, {}, {"exact_continuation": False}, {"exact_continuation": "true"},
                                    {"exact_continuation": 1}])
def test_unsaved_unreadable_pending_keeps_its_existing_terminal_custody(tmp_path, monkeypatch, marker):
    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    task_results.write_task_result(tmp_path, "saved", "scheduled")
    task = {"id": "saved", "type": "task", "chat_id": 0, "_attempt": 1}
    if marker is not None:
        task["_budget_pause"] = marker
    workers.PENDING.append(task)
    with _result_gap(tmp_path, monkeypatch, "unreadable"), queue._queue_lock:
        assert workers._drop_cancelled_pending()
    [kept] = workers.PENDING
    assert kept["_terminalization_retry"]["trigger"] == "pending_result_authority"
    assert kept["_terminalization_retry"]["status"] == "failed"
    assert not queue_snapshot._exact_pause_row(kept)


def test_accepted_project_without_exact_pause_keeps_project_verification_hold(tmp_path, monkeypatch):
    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    task_results.write_task_result(tmp_path, "saved", "scheduled")
    task = {"id": "saved", "type": "task", "chat_id": 0, "project_id": "project", "_attempt": 1}
    workers.PENDING.append(task)
    with _result_gap(tmp_path, monkeypatch, "unreadable"), queue._queue_lock:
        assert workers._drop_cancelled_pending()
    assert workers.PENDING == [task]
    assert task["_project_admission_restore_hold"]["reason"] == "project_routing_fence_lookup_failed"
    assert not task.get("_terminalization_retry") and not task.get("_budget_pause")


@pytest.mark.parametrize("entry", ["live", "restore"])
@pytest.mark.parametrize("outcome", ["completed", "stop"])
def test_saved_pause_does_not_override_known_terminal_or_stop(tmp_path, monkeypatch, entry, outcome):
    from ouroboros.cancel_intents import request_cancel

    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    _saved_pause(tmp_path, monkeypatch, workers)
    if entry == "restore":
        assert queue.persist_queue_snapshot(reason="saved-pause-control")
        workers.PENDING.clear()
    if outcome == "completed":
        task_results.write_task_result(tmp_path, "saved", "completed", result="retained answer")
    else:
        request_cancel(tmp_path, "saved", reason="owner Stop")
    if entry == "restore":
        assert queue.restore_pending_from_snapshot() == 0
    else:
        with queue._queue_lock:
            assert workers._drop_cancelled_pending()
    assert workers.PENDING == []
    if outcome == "completed":
        row = task_results.load_task_result(tmp_path, "saved")
        assert row["status"] == "completed" and row["result"] == "retained answer"


@pytest.mark.parametrize("entry", ["live", "restore"])
@pytest.mark.parametrize("gap", ["busy", "unreadable"])
def test_native_warm_handoff_survives_result_gap_until_explicit_resume(tmp_path, monkeypatch, entry, gap):
    """#1917's native carrier and the shared unreadable-result rule conserve the same work."""
    from ouroboros import owner_wait
    from tests.test_warm_restart_retention import _handoff, _park, _pool_events

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    events = _pool_events(workers, monkeypatch)
    ctx, limit, wait, _task = _park(tmp_path, workers, "sleep")
    original = read_actor_source_bytes(tmp_path, "saved-warm", wait["source_ref"])
    # The real current selector writes the native handoff. An unacknowledged
    # stop does not authorize its return; later explicit Resume owns recovery.
    selected = _handoff(tmp_path, workers, wait, ack=False)
    assert selected == {"saved-warm"}
    assert workers.kill_workers(force=True, terminal_status="cancelled", result_reason="Server shutdown.",
                                stop_source="server_shutdown", retain_saved_work=True, preserve_pending=True,
                                preserve_running_task_ids=selected)
    [pending] = workers.PENDING
    carrier = copy.deepcopy(pending["_owner_wait_resume"])
    assert carrier["wait_id"] == wait["wait_id"] and not pending.get("_budget_pause")
    if entry == "restore":
        workers.PENDING.clear()
    with _result_gap(tmp_path, monkeypatch, gap, task_id="saved-warm"):
        if entry == "restore":
            assert queue.restore_pending_from_snapshot() == 1
        else:
            with queue._queue_lock:
                assert workers._drop_cancelled_pending()
        [kept] = workers.PENDING
        assert {k: v for k, v in kept["_owner_wait_resume"].items() if k != "refusal"} == carrier
        assert kept["_attempt"] == 1
        assert not kept.get("_terminalization_retry") and not kept.get("_budget_pause_resume")
        assert not kept.get("_project_admission_restore_hold")
    with queue._queue_lock:
        assert workers._drop_cancelled_pending()
    assert not owner_wait.restore_owner_wait_allowed(tmp_path, kept)
    assert not kept.get("_budget_pause_resume") and workers.RUNNING == {}
    assert not [event for event in events if event.get("type") == "task_done"]
    assert read_actor_source_bytes(tmp_path, "saved-warm", wait["source_ref"]) == original
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 5.0)
    resumed = queue.resume_budget_paused_task("saved-warm")
    assert resumed["ok"], resumed
    assert not kept.get("_owner_wait_resume") and kept["_budget_pause_resume"]["grant_id"]
    ctx.budget_pause_resume = kept["_budget_pause_resume"]
    saved = budget_pause.load_budget_pause(ctx)
    assert saved["messages"] == limit.messages and saved["task_attempt"] == 1
    assert read_actor_source_bytes(tmp_path, "saved-warm", wait["source_ref"]) == original


@pytest.mark.parametrize("entry", ["live", "restore"])
def test_native_project_handoff_keeps_its_verification_owner(restart_case, monkeypatch, entry):  # noqa: F811
    from types import SimpleNamespace
    from ouroboros.work_stop import continuation_refusal
    from supervisor import queue, workers
    from tests.test_owner_wait_restart import restore_project_wait
    from tests.test_project_hold_recovery import worker

    case = restart_case
    restored = restore_project_wait(case, monkeypatch, registry_outage=False)
    carrier = copy.deepcopy(restored["_owner_wait_resume"])
    source = read_actor_source_bytes(case.root, case.task_id, carrier["source_ref"])
    if entry == "restore":
        assert queue.persist_queue_snapshot(reason="native-project-result-gap")
        workers.PENDING.clear()
    with _result_gap(case.root, monkeypatch, "unreadable", task_id=case.task_id):
        if entry == "restore":
            queue.restore_pending_from_snapshot()
        else:
            with queue._queue_lock:
                assert workers._drop_cancelled_pending()
        [kept] = workers.PENDING
        assert {k: v for k, v in kept["_owner_wait_resume"].items() if k != "refusal"} == carrier
        assert kept["_project_admission_restore_hold"]["reason"] == "project_routing_fence_lookup_failed"
        assert not kept.get("_terminalization_retry")
    sent = worker(SimpleNamespace(root=case.root), monkeypatch)
    workers.assign_tasks()
    workers.assign_tasks()
    if entry == "restore":
        # Project verification does not erase boot's separate saved-source hold.
        # This test proves conservation and that each existing owner still runs.
        assert not sent and continuation_refusal(kept)
        assert not kept.get("_project_admission_restore_hold")
        assert {k: v for k, v in kept["_owner_wait_resume"].items() if k != "refusal"} == carrier
        assert not kept.get("_terminalization_retry")
        assert read_actor_source_bytes(case.root, case.task_id, carrier["source_ref"]) == source
        return
    assert [row["id"] for row in sent] == [case.task_id]
    assert sent[0]["_owner_wait_resume"] == carrier and sent[0]["_attempt"] == case.attempt
    assert not sent[0].get("_project_admission_restore_hold")
    assert read_actor_source_bytes(case.root, case.task_id, carrier["source_ref"]) == source
