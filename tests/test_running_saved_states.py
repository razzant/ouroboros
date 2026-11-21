"""A work that goes on keeps its current points; its earlier saved states leave from recorded facts (#3797).

Owner decision 2026-11-19: the follow-up of #3789 also retires the earlier saved states of works that are still
running, the owner's days after a write superseded them or a pass first proved from recorded facts that nothing can
load them again, never by a file's time. Each rule is witnessed here in both directions: it fires on the dead or
blocked file and stays quiet on the live one.
"""
from __future__ import annotations

import json
import os
import time

import pytest

from ouroboros.startup_migrations import prepare_startup_state
from ouroboros.task_results import write_task_result
from tests.test_worked_copy_retirement import DAYS, SHA, _file, _iso, _ledger

TASK = "run"


@pytest.fixture
def root(tmp_path, monkeypatch):
    data = tmp_path / "data"
    prepare_startup_state(data)
    monkeypatch.setenv("OUROBOROS_GC_RETENTION_DAYS", str(DAYS))
    # A live publisher renews its stamp every few seconds; on a loaded runner one pass can outlast the 10-second
    # freshness, so the test's queue stays fresh as long as a stale-queue test does not age it on purpose.
    monkeypatch.setattr("ouroboros.task_status._SNAPSHOT_OWNERSHIP_FRESH_SEC", 600.0)
    _publish(data)
    return data


def _pass(root, *, now=None):
    """One pass, the test's published queue stamped again first (a slow runner must not see it go stale)."""
    from ouroboros.saved_state_retirement import prune_worked_copies
    from ouroboros.utils import atomic_write_json

    try:
        document = json.loads((root / "state/queue_snapshot.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        document = None
    if isinstance(document, dict):
        stamp = {"generation": document.get("generation") or "copies-test", "ts": _iso(time.time())}
        atomic_write_json(root / "state/queue_snapshot.json", {**document, **stamp})
        atomic_write_json(root / "state/queue.alive", stamp)
    return prune_worked_copies(root, now=now)


def _later(root):
    """A pass once the owner's days have passed since the passes before it."""
    return _pass(root, now=time.time() + (DAYS + 1) * 86400)


def _publish(root, *entries):
    """A fresh published queue running these rows (``{"id": ..., "task": {...}}``)."""
    from ouroboros.utils import atomic_write_json

    stamp = {"generation": "copies-test", "ts": _iso(time.time())}
    atomic_write_json(root / "state/queue_snapshot.json", {**stamp, "pending": [], "running": list(entries)})
    atomic_write_json(root / "state/queue.alive", stamp)


def _running(root, task_id=TASK, *, attempt):
    write_task_result(root, task_id, "running", task_attempt=attempt)
    _file(root / "observability" / "calls" / task_id / "llm-1.json", "{}")
    return root / "task_results" / "artifacts" / task_id / "source_handles"


def _park(root, task_id=TASK, *, attempt, state=None):
    """A real park: its file through ``owner_wait.checkpoint_owner_wait`` (whose head names the attempt) and, with
    ``state``, the row's wait through ``set_owner_wait`` (the store seam sees that write)."""
    from ouroboros.owner_wait import checkpoint_owner_wait, set_owner_wait
    from tests._budget_pause_exact_helpers import _loop_ctx

    ctx, limit_ctx = _loop_ctx(root, task_id, attempt=attempt)
    wait = checkpoint_owner_wait(ctx, limit_ctx.messages, {}, {}, 4, [], set())
    if state:
        wait = {**wait, "state": state}
        set_owner_wait(root, task_id, wait)
    return wait, root / "task_results" / "artifacts" / task_id / wait["source_ref"]["path"]


def _set_wait(root, wait, task_id=TASK, **changes):
    from ouroboros.owner_wait import set_owner_wait

    set_owner_wait(root, task_id, {**wait, **changes})


def _ref(name):
    return {"kind": "task_source", "root": "artifact_store", "path": f"source_handles/context_checkpoints/{name}",
            "size": 5, "sha256": SHA}


def _pause(root, pause_id, *, attempt, state="paused", task_id=TASK, on_row=True, **facts):
    """A pause file and, with ``on_row``, the row's pause through ``budget_pause.set_budget_pause``."""
    from ouroboros import budget_pause

    name = f"budget-pause-{pause_id}-{SHA}.json"
    path = _file(root / "task_results" / "artifacts" / task_id / "source_handles" / "context_checkpoints" / name, "pause")
    if on_row:
        budget_pause.set_budget_pause(root, task_id, {
            "pause_id": pause_id, "pause_generation": 1, "task_attempt": attempt, "reason": "owner", "rail": "owner_pause",
            "scope": "root", "state": state, "source_ref": _ref(name), **facts})
    return path


def _basis(handles, path):
    return next(line["basis"] for line in _ledger(handles) if line["name"] == path.name)


def test_the_current_wait_stays_even_past_its_attempt_and_an_earlier_one_goes(root):
    """A cold return loads attempt 1's wait in attempt 2 and marks it ``resumed``: the row still names it, and a relay
    can load it again (map §1.3), so the file the row names is never a candidate, whatever its state."""
    handles = _running(root, attempt=1)
    _orphan, orphan = _park(root, attempt=1)  # a park of attempt 1 whose row write was refused
    _current, current = _park(root, attempt=1, state="resumed")
    write_task_result(root, TASK, "running", task_attempt=2)
    _pass(root)
    report = _later(root)
    assert not orphan.exists() and current.exists(), report
    assert [line["name"] for line in _ledger(handles)] == [orphan.name]


def test_the_wait_a_live_pause_retains_stays_until_the_pause_is_consumed(root):
    handles = _running(root, attempt=2)
    retained, kept = _park(root, attempt=1)  # here only the pause names it
    _park(root, attempt=2, state="resumed")
    _pause(root, "a" * 32, attempt=2, retained_owner_wait_id=retained["wait_id"])
    _pass(root)
    assert _later(root)["removed"] == 0 and kept.exists(), "live_park(verify_source=True) still opens it"
    _pause(root, "a" * 32, attempt=2, state="resumed", retained_owner_wait_id=retained["wait_id"])  # consumed
    _pass(root)
    assert _later(root)["removed"] == 1 and not kept.exists()
    assert _basis(handles, kept) == "earlier_attempt"


def test_this_tasks_carrier_holds_by_path_and_by_wait_until_the_receiving_attempt_saves(root):
    """``_working_recovery`` is the one carrier read by its own ``source_ref``, whatever its admissibility, and
    ``_carry_recovery`` relays it while the receiving attempt has no rolling file (EXP1)."""
    handles = _running(root, attempt=1)
    by_path, path_file = _park(root, attempt=1)
    by_wait, wait_file = _park(root, attempt=1)
    _park(root, attempt=1, state="resumed")
    write_task_result(root, TASK, "running", task_attempt=2)  # attempt 2 started from the carriers below
    carriers = {"_working_recovery": {"source_kind": "owner_wait", "source_task_id": TASK, "from_attempt": 1,
                                       "wait_id": "f" * 32, "source_ref": by_path["source_ref"]},
                "_terminalization_retry": {"unsubmitted_launch": {"source": {
                    "source_kind": "owner_wait", "source_task_id": TASK, "from_attempt": 1, "wait_id": by_wait["wait_id"]}}}}
    _publish(root, {"id": TASK, "task": {"id": TASK, **carriers}})
    _pass(root)
    report = _later(root)
    assert report["held"] == {"lent_to_receiver": 2} and path_file.exists() and wait_file.exists(), report
    _file(handles / "working" / "working_checkpoint-a2.json", "attempt 2 saved")  # the relay can no longer fire
    report = _later(root)
    assert report["removed"] == 2 and not path_file.exists() and not wait_file.exists(), report


def test_another_tasks_carrier_or_retry_line_holds_all_and_this_tasks_own_row_holds_nothing(root):
    handles = _running(root, attempt=2)
    stale = _file(handles / "working" / "working_checkpoint-a1.json", "attempt 1")
    own = {"id": TASK, "task": {"id": TASK}}  # the queue runs this task: not a receiver of its own earlier states
    _publish(root, own, {"id": "other", "task": {"id": "other", "_working_recovery": {"source_task_id": TASK}}})
    _pass(root)
    assert _later(root)["held"] == {"lent_to_receiver": 1} and stale.exists()
    _publish(root, own, {"id": "succ", "task": {"id": "succ", "timeout_retry_from": TASK, "original_task_id": TASK}})
    assert _later(root)["held"] == {"lent_to_receiver": 1} and stale.exists()
    _publish(root, own)
    assert _later(root)["removed"] == 1 and not stale.exists()


def test_a_stale_or_missing_queue_holds_and_a_stopped_publisher_is_seen_at_once(root):
    from ouroboros.owner_keep import set_owner_keep
    from ouroboros.saved_state_retirement import prune_worked_copies
    from ouroboros.utils import atomic_write_json

    handles = _running(root, attempt=2)
    stale = _file(handles / "working" / "working_checkpoint-a1.json", "attempt 1")
    _pass(root)
    later = time.time() + (DAYS + 1) * 86400
    set_owner_keep(root, TASK, keep=True, source={"basis": "test"})
    _publish(root, {"id": TASK, "task": {"id": TASK}})
    assert prune_worked_copies(root, now=later)["held"] == {"owner_keep": 1}, "this pass read the fresh queue"
    set_owner_keep(root, TASK, keep=False, source={"basis": "test"})
    atomic_write_json(root / "state/queue.alive", {"generation": "copies-test", "ts": _iso(time.time() - 3600)})
    assert prune_worked_copies(root, now=later)["held"] == {"lent_to_receiver": 1} and stale.exists(), \
        "only the certificate aged: its publisher stopped, so the parse it gave proves nothing now"
    (root / "state/queue_snapshot.json").unlink()
    assert prune_worked_copies(root, now=later)["held"] == {"lent_to_receiver": 1} and stale.exists()
    _publish(root, {"id": TASK, "task": {"id": TASK}})
    assert prune_worked_copies(root, now=later)["removed"] == 1 and not stale.exists()


def test_a_wait_of_an_earlier_attempt_goes_and_one_of_this_attempt_stays_while_the_row_is_resumed(root):
    """A wait of the row's attempt that the row does not name may be a park whose row write has not landed yet
    (``worker_owner_wait`` writes the file first): with the row's wait ``resumed`` nothing proves it dead."""
    handles = _running(root, attempt=1)
    _old, old = _park(root, attempt=1)
    write_task_result(root, TASK, "running", task_attempt=2)
    _park(root, attempt=2, state="resumed")
    _in_flight, in_flight = _park(root, attempt=2)
    _pass(root)
    report = _later(root)
    assert not old.exists() and in_flight.exists(), report
    assert [(line["name"], line["basis"]) for line in _ledger(handles)] == [(old.name, "earlier_attempt")]


def test_a_wait_of_this_attempt_goes_once_this_attempt_parks_on_another(root):
    handles = _running(root, attempt=2)
    _park(root, attempt=1, state="waiting")  # attempt 2 runs while attempt 1's wait still waits (a failed consume)
    _x, x = _park(root, attempt=2)
    _pass(root)
    assert _later(root)["removed"] == 0 and x.exists(), "the waiting wait is another attempt's: attempt 2 may be parking"
    _park(root, attempt=2, state="waiting")  # attempt 2 parked: none of its parks is in flight
    _pass(root)
    _later(root)
    assert not x.exists() and _basis(handles, x) == "parked_on_another_wait"


def test_a_settled_live_pause_proves_the_attempts_other_waits_dead_but_pausing_does_not(root):
    handles = _running(root, attempt=2)
    _park(root, attempt=2, state="resumed")
    _x, x = _park(root, attempt=2)
    _pause(root, "c" * 32, attempt=2, state="pausing")  # the seed comes first; the loop may still be parking
    older = _running(root, "older", attempt=2)  # a live pause of an earlier attempt says nothing of attempt 2's parks
    _park(root, "older", attempt=2, state="resumed")
    _y, y = _park(root, "older", attempt=2)
    _pause(root, "d" * 32, attempt=1, task_id="older")
    visited = _file(older / "working" / "working_checkpoint-a1.json", "attempt 1")  # goes: the pass reads this task
    _pass(root)  # the history walk owes both works
    report = _later(root)
    assert x.exists() and y.exists() and not visited.exists(), report
    _pause(root, "c" * 32, attempt=2, state="paused")  # the stack unwound: only the work's own loop seeds a pause
    _pass(root)
    _later(root)
    assert not x.exists() and _basis(handles, x) == "paused" and y.exists()


def test_a_pause_the_row_no_longer_names_goes_and_the_current_one_stays(root):
    handles = _running(root, attempt=1)
    old = _pause(root, "e" * 32, attempt=1, on_row=False)  # the row moved past it before this body ran
    current = _pause(root, "b" * 32, attempt=1, state="resumed")
    _pass(root)
    _later(root)
    assert not old.exists() and current.exists() and _basis(handles, old) == "pause_replaced"


def test_a_restart_conversion_in_flight_keeps_its_pause_copy(root):
    """``retain_sleep_checkpoint`` and ``retain_owner_pause_park`` write the copy before the row names it."""
    handles = _running(root, attempt=1)
    checkpoints = handles / "context_checkpoints"
    sleep_id, owner_id = "1" * 32, "3" * 32
    for wait_id in (sleep_id, owner_id):
        _file(checkpoints / f"owner-wait-{wait_id}-{SHA}.json")
    sleeping = {"wait_id": sleep_id, "state": "waiting", "reason": "sleep", "task_attempt": 1,
                "source_ref": _ref(f"owner-wait-{sleep_id}-{SHA}.json")}
    _set_wait(root, sleeping)
    sleep_copy = _file(checkpoints / f"budget-pause-sleep-{sleep_id}-{SHA}.json")
    other_copy = _file(checkpoints / f"budget-pause-owner-pause-{'2' * 32}-{SHA}.json")
    _pass(root)
    _later(root)
    assert sleep_copy.exists() and not other_copy.exists(), "only the sleep's own conversion may still land"
    _set_wait(root, {**sleeping, "wait_id": owner_id, "reason": "owner_pause",
                     "source_ref": _ref(f"owner-wait-{owner_id}-{SHA}.json")})  # a warm owner-pause park
    owner_copy = _file(checkpoints / f"budget-pause-owner-pause-{'4' * 32}-{SHA}.json")
    _pass(root)
    _later(root)
    assert owner_copy.exists() and not sleep_copy.exists()


def test_a_frozen_point_of_an_earlier_attempt_goes_once_the_current_attempt_has_its_own_rolling_file(root):
    handles = _running(root, attempt=2)
    frozen = _file(handles / "context_checkpoints" / f"working-a1-s3-{SHA}.json")
    _pass(root)
    assert _later(root)["removed"] == 0 and frozen.exists(), "attempt 2 may still die before its first save and relay it"
    _file(handles / "working" / "working_checkpoint-a2.json", "attempt 2 saved")
    _pass(root)
    _later(root)
    assert not frozen.exists() and _basis(handles, frozen) == "earlier_attempt_continued"


def test_a_rolling_file_of_an_earlier_attempt_goes_and_the_current_attempts_stays(root):
    handles = _running(root, attempt=2)
    earlier = _file(handles / "working" / "working_checkpoint-a1.json", "attempt 1")
    current = _file(handles / "working" / "working_checkpoint-a2.json", "attempt 2")
    _pass(root)
    _later(root)
    assert not earlier.exists() and current.exists()


def test_a_replaced_wait_or_pause_is_recorded_with_its_time_and_a_change_of_state_is_not(root):
    from ouroboros import maintenance_jobs, store
    from ouroboros.saved_state_retirement import WORKED_COPIES

    def facts():
        with store.open_store(root, write=False) as txn:
            return txn.get_row(WORKED_COPIES, TASK) or {}

    def owed_now(due=None):
        with store.open_store(root, write=True) as txn:
            maintenance_jobs.record_in(txn, "runtime_retirement", next_due=due)

    _running(root, attempt=1)
    first, first_file = _park(root, attempt=1, state="waiting")
    _set_wait(root, first, state="resumed")
    _set_wait(root, first, state="waiting")  # a warm grant rolled back: the same wait, never superseded (EXP3)
    pause_file = _pause(root, "5" * 32, attempt=1)
    _pause(root, "5" * 32, attempt=1, state="resumed")
    assert not facts().get("superseded"), "a change of state of the same wait or pause moves past nothing"
    owed_now()
    before = time.time()
    _park(root, attempt=1, state="waiting")
    _pause(root, "6" * 32, attempt=1)
    superseded = facts()["superseded"]
    assert set(superseded) == {first_file.name, pause_file.name} and min(superseded.values()) >= before
    assert abs(maintenance_jobs.job(root, "runtime_retirement")["next_due"] - (before + DAYS * 86400)) < 60
    owed_now()
    write_task_result(root, TASK, "running", task_attempt=2)  # a new attempt: the pass dates the earlier one's files now
    assert maintenance_jobs.job(root, "runtime_retirement")["next_due"] <= time.time()


def test_a_write_that_moves_nothing_reads_and_writes_no_row(root, monkeypatch):
    from ouroboros import store

    _running(root, attempt=1)
    wait, _wait_file = _park(root, attempt=1, state="waiting")
    touched = []
    real_get, real_put = store.Txn.get_row, store.Txn.put_row
    monkeypatch.setattr(store.Txn, "get_row", lambda txn, name, identity: touched.append(name) or real_get(txn, name, identity))
    monkeypatch.setattr(store.Txn, "put_row", lambda txn, name, identity, facts: touched.append(name) or real_put(txn, name, identity, facts))
    write_task_result(root, TASK, "running", title="a title only")
    _set_wait(root, wait, resume_reason="answer")  # the same wait, one more fact on it
    assert "worked_copies" not in touched
    _park(root, attempt=1, state="waiting")
    assert "worked_copies" in touched


def test_the_window_runs_from_the_seams_time_or_the_first_proof_never_the_files_time(root):
    handles = _running(root, attempt=1)
    _first, first = _park(root, attempt=1, state="waiting")
    replaced = time.time()
    _park(root, attempt=1, state="waiting")
    stale = _file(handles / "working" / "working_checkpoint-a1.json", "attempt 1")
    write_task_result(root, TASK, "running", task_attempt=2)
    for path in (first, stale):  # saved long ago: a file's time proves nothing
        os.utime(path, (time.time() - 30 * 86400,) * 2)
    proved = replaced + DAYS * 86400 - 60
    report = _pass(root, now=proved)
    assert first.exists() and stale.exists() and report["waiting"] == 2, "a minute before the seam's days end"
    assert _pass(root, now=replaced + DAYS * 86400 + 60)["removed"] == 1 and not first.exists() and stale.exists()
    assert _pass(root, now=proved + DAYS * 86400 + 60)["removed"] == 1 and not stale.exists()
    lines = {line["name"]: line for line in _ledger(handles)}
    assert "superseded_at" in lines[first.name] and "proven_at" in lines[stale.name]


def test_the_row_and_the_queue_are_read_again_under_the_lock(root, monkeypatch):
    from ouroboros import saved_state_retirement, store

    handles = _running(root, attempt=1)
    revived, revived_file = _park(root, attempt=1)
    carried, carried_file = _park(root, attempt=1)
    _free, free_file = _park(root, attempt=1)
    _park(root, attempt=1, state="resumed")
    write_task_result(root, TASK, "running", task_attempt=2)
    _pass(root)
    real = saved_state_retirement._call_records

    def meanwhile(root_, task_id):  # after the first read, before the lock: the row names one, a carrier names another
        _set_wait(root, revived, state="waiting")
        _publish(root, {"id": TASK, "task": {"id": TASK, "_working_recovery": {
            "source_kind": "owner_wait", "source_task_id": TASK, "from_attempt": 1, "wait_id": carried["wait_id"]}}})
        return real(root_, task_id)

    monkeypatch.setattr(saved_state_retirement, "_call_records", meanwhile)
    report = _later(root)
    assert report["held"] == {"changed": 1, "lent_to_receiver": 1} and revived_file.exists() and carried_file.exists()
    assert report["removed"] == 1 and not free_file.exists(), "an unchanged proof still holds under the lock"
    assert [line["name"] for line in _ledger(handles)] == [free_file.name]
    monkeypatch.setattr(saved_state_retirement, "_call_records", real)
    _pass(root)
    with store.open_store(root, write=False) as txn:
        proven = (txn.get_row(saved_state_retirement.WORKED_COPIES, TASK) or {}).get("proven") or {}
    assert revived_file.name not in proven and carried_file.name in proven, "a current point's proof is lifted"


def test_a_running_works_first_settled_write_is_a_new_obligation(root):
    from ouroboros import maintenance_jobs, store
    from ouroboros.saved_state_retirement import WORKED_COPIES

    _running(root, attempt=1)
    _park(root, attempt=1, state="waiting")
    _park(root, attempt=1, state="waiting")  # its row holds a running work's facts, never a settled date
    with store.open_store(root, write=True) as txn:
        assert "superseded" in txn.get_row(WORKED_COPIES, TASK)
        maintenance_jobs.record_in(txn, "runtime_retirement", next_due=None)
    before = time.time()
    write_task_result(root, TASK, "completed", completed_at=_iso(time.time() - 10 * 86400))  # its window already ended
    assert maintenance_jobs.job(root, "runtime_retirement")["next_due"] <= before + 5, "a new obligation is owed now"
    with store.open_store(root, write=False) as txn:
        assert txn.get_row(WORKED_COPIES, TASK)["settled_at"] >= before


def test_a_running_works_record_says_what_it_moved_on_to_and_a_read_says_so(root):
    from ouroboros.saved_state_retirement import retired_message
    from ouroboros.tools.core_file_tools import _read_file
    from ouroboros.tools.tool_context import ToolContext

    handles = _running(root, attempt=1)
    _first, first = _park(root, attempt=1, state="waiting")
    _current, current = _park(root, attempt=1, state="waiting")
    _pass(root)
    _later(root)
    [line] = _ledger(handles)
    assert line["name"] == first.name and line["status"] == "running" and line["basis"] == "superseded"
    assert line["continues_from"] == current.name and "superseded_at" in line and "finished_at" not in line
    assert line["rebuild"].startswith("An earlier saved state of a work that went on from a newer one")
    said = retired_message(root, TASK, first.name)
    assert f"after the work moved on to {current.name}" in said and "after its work ended" not in said
    ctx = ToolContext(repo_dir=root, drive_root=root, task_id=TASK, budget_drive_root=str(root))
    answer = _read_file(ctx, f"source_handles/context_checkpoints/{first.name}", root="artifact_store")
    assert "was retired at" in answer and f"moved on to {current.name}" in answer and "work ended" not in answer


def test_the_attempt_comes_from_the_head_the_wait_writer_gives_its_file(root):
    handles = _running(root, attempt=1)
    _written, written = _park(root, attempt=1)  # bytes of the one writer: ``{"task_id": ..., "task_attempt": 1, ...``
    checkpoints = handles / "context_checkpoints"
    reordered = _file(checkpoints / f"owner-wait-{'7' * 32}-{SHA}.json", json.dumps({"task_attempt": 1, "task_id": TASK}))
    foreign = _file(checkpoints / f"owner-wait-{'8' * 32}-{SHA}.json", json.dumps({"task_id": "other", "task_attempt": 1}))
    write_task_result(root, TASK, "running", task_attempt=2)
    _park(root, attempt=2, state="resumed")
    _pass(root)
    _later(root)
    assert not written.exists() and reordered.exists() and foreign.exists(), "an attempt not read proves nothing"


def test_keep_a_promotion_and_missing_call_records_hold_a_running_works_earlier_states(root):
    from ouroboros import store
    from ouroboros.owner_keep import set_owner_keep

    handles = _running(root, attempt=2)
    stale = _file(handles / "working" / "working_checkpoint-a1.json", "attempt 1")
    _pass(root)
    set_owner_keep(root, TASK, keep=True, source={"basis": "test"})
    assert _later(root)["held"] == {"owner_keep": 1} and stale.exists()
    set_owner_keep(root, TASK, keep=False, source={"basis": "test"})
    with store.open_store(root, write=True) as txn:
        txn.conn.execute("UPDATE tasks SET has_pending_promotion=1 WHERE task_id=?", (TASK,))
    assert _later(root)["held"] == {"child_refs_pending": 1} and stale.exists()
    with store.open_store(root, write=True) as txn:
        txn.conn.execute("UPDATE tasks SET has_pending_promotion=0 WHERE task_id=?", (TASK,))
    calls = root / "observability" / "calls" / TASK / "llm-1.json"
    calls.unlink()
    assert _later(root)["held"] == {"no_call_records": 1} and stale.exists()
    _file(calls, "{}")
    assert _later(root)["removed"] == 1 and not stale.exists()


def test_the_history_walk_owes_running_holders_and_keeps_their_dates(root):
    from ouroboros import store

    _pass(root)  # the walk of an empty history under this body
    write_task_result(root, TASK, "running", task_attempt=2)  # a running work from before this body: no row for it
    _file(root / "observability" / "calls" / TASK / "llm-1.json", "{}")
    stale = _file(root / "task_results" / "artifacts" / TASK / "source_handles" / "working" / "working_checkpoint-a1.json")
    _pass(root)
    assert _later(root)["removed"] == 0 and stale.exists(), "no row: the walk under this body ran before it held anything"
    _running(root, "dated", attempt=1)
    _first, superseded = _park(root, "dated", attempt=1, state="waiting")
    current, _current_file = _park(root, "dated", attempt=1, state="waiting")
    _set_wait(root, current, "dated", state="resumed")  # now only the recorded replacement proves the first file dead
    with store.open_store(root, write=True) as txn:  # another body booted on the store: the walk runs again
        txn.set_meta("last_store_body", {"sha": "0" * 40, "at": _iso(time.time())})
    _pass(root)
    assert _later(root)["removed"] == 2 and not stale.exists() and not superseded.exists()


def test_a_partial_pack_listing_holds_a_running_works_states_and_a_whole_one_names_the_generation(root, monkeypatch):
    from ouroboros import evidence_packs, saved_state_retirement, store

    handles = _running(root, attempt=1)
    _first, first = _park(root, attempt=1)
    with monkeypatch.context() as older_body:  # a body before #3789 packed every handle, one generation a pass
        older_body.setattr(saved_state_retirement, "worked_copy", lambda folder, name: None)
        older_body.setattr(evidence_packs, "task_packable", lambda txn, task_id: True)  # and unlinked a live row's too
        evidence_packs.pack_task(root, TASK)
        _park(root, attempt=1)
        evidence_packs.pack_task(root, TASK)
    rel = f"source_handles/context_checkpoints/{first.name}"
    assert evidence_packs.materialize_handle(root, TASK, rel) and first.exists()
    write_task_result(root, TASK, "running", task_attempt=2)
    (evidence_packs.packs_dir(root) / evidence_packs.sidecar_name(TASK, 1)).write_text("{", encoding="utf-8")
    _pass(root)
    with store.name_lock(root, evidence_packs.INDEX_LOCK_NAME, timeout_sec=0), store.acquisition_budget(0):
        report = _later(root)
    assert report["held"] == {"packs_unknown": 1} and first.exists() and not _ledger(handles), report
    assert _later(root)["removed"] == 1 and not first.exists()
    [line] = _ledger(handles)
    assert line["packed"] == [f"observability/packs/{TASK}.xz"] and "went on from a newer one" in line["rebuild"]


def test_a_tasks_only_unreadable_pack_sidecar_holds_its_states_and_the_pass_keeps_its_dates(root, monkeypatch):
    """One pack generation, its sidecar unreadable and no row of it in the index: unknown, never an error that ends
    the pass before it writes the dates it proved."""
    from ouroboros import evidence_packs, saved_state_retirement, store

    handles = _running(root, attempt=1)
    _first, first = _park(root, attempt=1)
    with monkeypatch.context() as older_body:  # a body before #3789 packed every handle
        older_body.setattr(saved_state_retirement, "worked_copy", lambda folder, name: None)
        older_body.setattr(evidence_packs, "task_packable", lambda txn, task_id: True)  # and unlinked a live row's too
        evidence_packs.pack_task(root, TASK)
    assert evidence_packs.materialize_handle(root, TASK, f"source_handles/context_checkpoints/{first.name}")
    write_task_result(root, TASK, "running", task_attempt=2)
    sidecar = evidence_packs.packs_dir(root) / evidence_packs.sidecar_name(TASK, 1)
    intact, index = sidecar.read_bytes(), evidence_packs.packs_dir(root) / evidence_packs.INDEX_NAME
    sidecar.write_text("{", encoding="utf-8")
    index.unlink()
    evidence_packs.ensure_index(root)  # rebuilt while that sidecar cannot be read: no row of this task
    _pass(root)
    stale = _file(handles / "working" / "working_checkpoint-a1.json", "attempt 1")  # first proven by the next pass
    report = _later(root)
    assert report["held"] == {"packs_unknown": 1} and report["errors"] == 0 and first.exists(), report
    with store.open_store(root, write=False) as txn:
        assert stale.name in txn.get_row(saved_state_retirement.WORKED_COPIES, TASK)["proven"], "the pass kept its dates"
    with evidence_packs.open_index(root, write=True) as conn:  # an index that never indexed this task, and says no fault
        for table in ("packs", "members", "handles"):
            conn.execute(f"DELETE FROM {table} WHERE task_id=?", (TASK,))
        conn.execute("DELETE FROM meta WHERE key='sidecar_errors'")
    assert _later(root)["held"] == {"packs_unknown": 1} and first.exists(), "an index without its rows proves no absence"
    sidecar.write_bytes(intact)
    index.unlink()
    assert _later(root)["removed"] == 1 and not first.exists()
    assert _ledger(handles)[0]["packed"] == [f"observability/packs/{TASK}.xz"]


def test_the_passs_own_read_of_a_wait_head_is_no_change_and_a_new_write_is(root, monkeypatch):
    from ouroboros import saved_state_retirement

    _running(root, attempt=1)
    _read, read = _park(root, attempt=1)
    _written, written = _park(root, attempt=1)
    write_task_result(root, TASK, "running", task_attempt=2)
    _park(root, attempt=2, state="resumed")
    _pass(root)
    real_head, real_calls = saved_state_retirement._wait_attempt, saved_state_retirement._call_records

    def relatime(path, task_id):  # Linux relatime: this read moves the access time of a file older than a day
        attempt = real_head(path, task_id)
        info = os.stat(path)
        os.utime(path, ns=(info.st_atime_ns + 10**12, info.st_mtime_ns))
        return attempt

    def meanwhile(root_, task_id):  # after the listing, before the lock: a write lands on the other file
        info = os.stat(written)
        os.utime(written, ns=(info.st_atime_ns, info.st_mtime_ns + 10**9))
        return real_calls(root_, task_id)

    monkeypatch.setattr(saved_state_retirement, "_wait_attempt", relatime)
    monkeypatch.setattr(saved_state_retirement, "_call_records", meanwhile)
    report = _later(root)
    assert not read.exists() and report["removed"] == 1, report
    assert report["held"] == {"changed": 1} and written.exists()


def test_a_saved_state_written_back_while_the_work_runs_is_not_dated_as_settled(root):
    """Carbon m1: a reader wrote a packed saved state back while the work ran; its later settled write without a
    recorded end (a recovered crash) must start the owner's days then, not at the write-back."""
    from ouroboros import store
    from ouroboros.saved_state_retirement import WORKED_COPIES, owe_written_back

    def row(task_id):
        with store.open_store(root, write=False) as txn:
            return txn.get_row(WORKED_COPIES, task_id) or {}

    _running(root, attempt=1)
    _park(root, attempt=1, state="waiting")
    owe_written_back(root, TASK)
    assert row(TASK).get("written_back") and "settled_at" not in row(TASK)
    settled = time.time()
    write_task_result(root, TASK, "failed", reason_code="worker_crash_owner_wait")  # recovered: no recorded end
    assert row(TASK)["settled_at"] >= settled, "its days run from the settled write"
    write_task_result(root, "done", "completed")
    owe_written_back(root, "done")
    assert row("done").get("settled_at"), "a finished work's write-back is dated, as before"
