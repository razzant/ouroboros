"""The ``review_threads`` step inside the maintenance child's pass (``maintenance_retirement``).

The tasks the step left waiting (a release whose outcome is unknown, an export that failed for now) run it whenever a
child runs for another job, alone, and never start one; it yields between operations to a delegate capture that child's
own capture pass never had; an unchanged pass writes nothing; and an archive item the child's compression rewrote is
synced before any release relies on it. Host producers, readers, the job order and the due rules are the real ones; the
engine is the in-memory MOCK of ``tests/test_review_thread_retention.py``."""
from __future__ import annotations

import io
import json
import os
import pathlib
import stat
import sys

import pytest

from ouroboros import (
    artifact_storage,
    artifacts,
    delegate_capture_job,
    delegate_custody_current,
    maintenance,
    maintenance_jobs,
    owner_keep,
    store,
)
from ouroboros import maintenance_retirement as retirement
from ouroboros import review_thread_retention as rt
from ouroboros.gateways.claudexor import ClaudexorUnavailable
from ouroboros.task_store import read_document
from tests.test_review_thread_retention import (
    KEEP,
    TASK,
    THREAD,
    C,
    R,
    _held_run,
    _own_thread,
    _publish_queue,
    _row,
    _second_seat,
    _two_owners,
)
from tests.test_review_thread_retention import real as real  # noqa: F401 (fixture)
from tests.test_review_thread_retention import world as world  # noqa: F401 (fixture)

B = "thr-2"  # the second seat's thread (``_second_seat``)
FACTS = ("state", "inputs", "errors", "next_due", "arrival")  # the history-sized steps' own facts on the job row


def _jobs(monkeypatch, *names, **stand_ins):
    """The child's pass over these jobs in the real registry order; a stand-in takes a real job's slot."""
    monkeypatch.setattr(maintenance, "JOBS", {name: stand_ins.get(name, job) for name, job in maintenance.JOBS.items()
                                              if name in names})


def _attach(real, monkeypatch):
    from ouroboros import claudexor_daemon

    monkeypatch.setattr(claudexor_daemon, "read_owned_gateway", lambda **_kw: real.engine)  # the step's own attach


def _capture_request(real, monkeypatch, events=None):
    """The server addresses a capture of another tree: a real reason for the next child. Its capture work is stubbed;
    the job's selection, order and records are real."""
    run_id = _held_run(real.root)
    assert delegate_capture_job.request_or_result(
        real.root, delegate_custody_current.run(real.root, run_id), generation="server")["status"] == "pending"

    def capture(root, entry, should_stop):
        if events is not None:
            events.append(("capture", entry.run_id))
        return delegate_capture_job._record(root, entry, {**entry.capture_job, "state": "ready", "child_generation": "t"})

    monkeypatch.setattr(delegate_capture_job, "_capture", capture)
    return run_id


def _facts(root):
    row = maintenance_jobs.job(root, retirement.JOB) or {}
    return {key: row.get(key) for key in FACTS}


def _started_and_at_rest(real):
    """A server start's full pass, then the server's look: nothing is due, so no child would start."""
    _publish_queue(real.root)
    retirement.record_startup(real.root, gaps=False)
    maintenance.run(real.root)
    assert maintenance.due(real.root) == []


# ---- work the step owes runs in a child another job started, alone -----------------------------------------------


DEBTS = {"unknown_release": "accepted", "kept_unknown_release": "abandoned", "export_unavailable": "accepted"}


@pytest.mark.parametrize("debt", sorted(DEBTS))
def test_waiting_work_is_taken_up_by_the_next_child_another_job_starts(real, monkeypatch, debt):
    _own_thread(real.root, R, R, real.record_id)
    _row(real.root, R, "completed")
    _attach(real, monkeypatch)
    _jobs(monkeypatch, "delegate_capture", "runtime_retirement")
    failing, export = {"on": debt == "export_unavailable"}, real.engine.thread_export

    def thread_export(thread_id, **kwargs):  # the engine's export read failing for now
        if failing["on"]:
            raise ClaudexorUnavailable("daemon_unreachable", "transient", status_code=0)
        return export(thread_id, **kwargs)

    real.engine.thread_export = thread_export
    if debt != "export_unavailable":
        real.engine.faults.add("lost_before")  # the retire request never reaches the engine
    _started_and_at_rest(real)  # the debt alone starts no child
    first, facts = real.release(), _facts(real.root)
    assert maintenance_jobs.job(real.root, retirement.JOB)[rt.JOB_FIELD]["waiting"] == [R]
    assert first.get("state") == (None if debt == "export_unavailable" else "bound"), first
    if debt == "kept_unknown_release":
        assert owner_keep.set_owner_keep(real.root, R, keep=True, source=KEEP)["status"] == "release_unconfirmed"
    failing["on"] = False
    _capture_request(real, monkeypatch)
    swept = []
    monkeypatch.setattr(retirement, "prune_stale_bytecode", lambda root, **_kw: swept.append(root) or {})
    real.engine.calls.clear()
    _publish_queue(real.root)
    assert set(maintenance.run(real.root)["jobs"]) == {"delegate_capture", "runtime_retirement"}
    assert real.release()["state"] == DEBTS[debt]
    if first:  # the same operation, read back first
        operation = first["operation_id"]
        assert real.engine.calls == [("read", operation), ("abandon" if "kept" in debt else "retire", operation)]
    else:
        assert real.engine.calls[0] == ("export", THREAD), real.engine.calls
    assert maintenance_jobs.job(real.root, retirement.JOB)[rt.JOB_FIELD]["waiting"] == []
    assert swept == [] and _facts(real.root) == facts  # the history-sized steps neither ran nor moved


def _events(root):
    log = pathlib.Path(root) / "logs" / "events.jsonl"
    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()] if log.exists() else []
    return [row for row in rows if row.get("type") == "runtime_retirement"]


def _revision(root):
    with store.open_store(root, write=False) as txn:
        return txn.revision()


def _step(root):
    return maintenance_jobs.job(root, retirement.JOB)[rt.JOB_FIELD]


# ---- what the engine answered survives the real caller, once, until it changes -----------------------------------


@pytest.mark.parametrize("down", [False, True], ids=["route_absent", "engine_unavailable"])
def test_the_engines_hold_survives_the_maintenance_caller_once_until_it_changes(real, monkeypatch, down):
    from ouroboros import claudexor_daemon

    reason = "engine_unavailable" if down else "engine_export_unsupported"
    _own_thread(real.root, R, R, real.record_id)
    _row(real.root, R, "completed")
    _jobs(monkeypatch, "delegate_capture", "runtime_retirement")
    if down:
        monkeypatch.setattr(claudexor_daemon, "read_owned_gateway",
                            lambda **_kw: (_ for _ in ()).throw(OSError("down")))
    else:
        _attach(real, monkeypatch)
        real.engine.supported = False
    _started_and_at_rest(real)  # the hold alone starts no child
    assert _step(real.root)["engine_hold"] == reason and _step(real.root)["waiting"] == [R]
    event = _events(real.root)[-1]["reports"]["review_threads"]
    assert event["holds"] == {reason: 1} and event["observed"] == [{"engine_hold": reason}], event
    count, revision = len(_events(real.root)), _revision(real.root)
    _publish_queue(real.root)
    assert "runtime_retirement" in maintenance.run(real.root)["jobs"]  # the next child takes the waiting task up
    assert (len(_events(real.root)), _revision(real.root)) == (count, revision)  # unchanged: saves and logs nothing
    _attach(real, monkeypatch)
    real.engine.supported = True
    _publish_queue(real.root)
    maintenance.run(real.root)
    assert real.release()["state"] == "accepted"
    assert "engine_hold" not in _step(real.root) and _step(real.root)["waiting"] == []
    assert _events(real.root)[-1]["reports"]["review_threads"]["observed"] == [{"engine_hold": ""}]


@pytest.mark.parametrize("answer, state", [("purged", "purged"), ("absent", "not_found")])
def test_a_typed_export_answer_survives_the_maintenance_caller_and_reaches_keep(real, monkeypatch, answer, state):
    _own_thread(real.root, R, R, real.record_id)
    _row(real.root, R, "completed")
    _attach(real, monkeypatch)
    _jobs(monkeypatch, "delegate_capture", "runtime_retirement")
    setattr(real.engine, answer, True)
    _started_and_at_rest(real)
    assert real.release()["state"] == state and _step(real.root)["waiting"] == []
    assert "engine_hold" not in _step(real.root)
    assert _events(real.root)[-1]["reports"]["review_threads"]["observed"] == [
        {"task_id": R, "thread_id": THREAD, "state": state}]
    count, calls = len(_events(real.root)), len(real.engine.calls)
    retirement.run(real.root, lambda: False)  # its own write names the task once more; then nothing is left to say
    assert len(_events(real.root)) == count and real.release()["state"] == state
    assert len(real.engine.calls) == calls + (0 if state == "purged" else 1)  # a purge is final; absence is asked again
    revision = _revision(real.root)
    retirement.run(real.root, lambda: False)
    assert _revision(real.root) == revision and len(_events(real.root)) == count
    answered = owner_keep.set_owner_keep(real.root, R, keep=True, source=KEEP)
    assert answered["releases"] == {f"review_thread:{THREAD}": state}
    assert answered["retired" if state == "purged" else "unknown"] == ([f"review_thread:{THREAD}"] if state == "purged"
                                                                     else ["review_threads"])
    assert f"review_thread:{THREAD}" not in answered["retained"]


def test_an_ordinary_pass_and_its_retry_log_nothing_new_and_keep_no_hold(real, monkeypatch):
    _own_thread(real.root, R, R, real.record_id)
    _row(real.root, R, "completed")
    _attach(real, monkeypatch)
    _jobs(monkeypatch, "delegate_capture", "runtime_retirement")
    export, failing = real.engine.thread_export, {"on": True}

    def thread_export(thread_id, **kwargs):
        if failing["on"]:
            raise ClaudexorUnavailable("daemon_unreachable", "transient", status_code=0)
        return export(thread_id, **kwargs)

    real.engine.thread_export = thread_export
    _started_and_at_rest(real)
    assert _step(real.root)["waiting"] == [R] and "engine_hold" not in _step(real.root) and _events(real.root) == []
    failing["on"] = False
    _publish_queue(real.root)
    maintenance.run(real.root)
    assert real.release()["state"] == "accepted" and _step(real.root)["waiting"] == [] and _events(real.root) == []


def test_a_pass_whose_task_read_failed_keeps_it_waiting_and_is_logged(real, monkeypatch):
    _own_thread(real.root, R, R, real.record_id)
    _row(real.root, R, "completed")
    _attach(real, monkeypatch)
    _jobs(monkeypatch, "delegate_capture", "runtime_retirement")
    monkeypatch.setattr(rt, "seat_threads", lambda *_a, **_kw: (_ for _ in ()).throw(store.StoreBusy("held")))
    _publish_queue(real.root)
    retirement.record_startup(real.root, gaps=False)
    maintenance.run(real.root)
    assert _step(real.root)["waiting"] == [R] and _events(real.root)[-1]["errors"] == ["review_threads"]
    assert "StoreBusy" in _events(real.root)[-1]["reports"]["review_threads"]["errors"][0]
    assert not real.release() and real.engine.calls == []
    monkeypatch.undo()
    _attach(real, monkeypatch)
    _jobs(monkeypatch, "delegate_capture", "runtime_retirement")
    _publish_queue(real.root)
    maintenance.run(real.root)
    assert real.release()["state"] == "accepted" and _step(real.root)["waiting"] == []


def test_a_pass_that_changes_nothing_writes_nothing(real):
    """A waiting task asked again over the same facts commits nothing: the maintenance child's next look (the evidence
    packer re-arms on any store write) sees no change from it, so the step cannot keep children coming."""
    _own_thread(real.root, R, R, real.record_id)
    _row(real.root, C, "scheduled", parent=R)  # a native child of the ended root: the tree is not over
    _row(real.root, R, "completed")
    assert real.run()["waiting"] == 1
    with store.open_store(real.root, write=False) as txn:
        before = txn.revision(), txn.meta(maintenance_jobs.PREFIX + retirement.JOB)
    assert real.run()["waiting"] == 1 and not real.engine.calls
    with store.open_store(real.root, write=False) as txn:
        assert (txn.revision(), txn.meta(maintenance_jobs.PREFIX + retirement.JOB)) == before


# ---- yielding to a capture this child's capture pass never had --------------------------------------------------


def test_a_capture_published_after_the_childs_capture_pass_takes_the_next_child_before_any_export(real, monkeypatch):
    events, published = [], []
    _two_owners(real, monkeypatch, events)
    run_id = _held_run(real.root)

    def server_publishes(root, should_stop):  # the concurrent server, after the child's capture pass looked
        assert delegate_capture_job.request_or_result(
            root, delegate_custody_current.run(root, run_id), generation="server")["status"] == "pending"
        published.append(run_id)
        events.append(("requested", run_id))
        return {"state": "done", "items": 1}

    def capture(root, entry, should_stop):
        events.append(("capture", entry.run_id))
        return delegate_capture_job._record(root, entry, {**entry.capture_job, "state": "ready", "child_generation": "t"})

    names = list(maintenance.JOBS)
    assert names.index("delegate_capture") < names.index("subagent_metrics") < names.index("runtime_retirement")
    _jobs(monkeypatch, "delegate_capture", "subagent_metrics", "runtime_retirement",
          subagent_metrics=(lambda *_args: not published, server_publishes))  # its slot between the two jobs
    monkeypatch.setattr(delegate_capture_job, "_capture", capture)
    _publish_queue(real.root)
    retirement.record_startup(real.root, gaps=False)
    maintenance.run(real.root)
    assert events == [("requested", run_id)], events  # nothing exported ahead of it
    assert maintenance_jobs.job(real.root, retirement.JOB)[rt.JOB_FIELD]["waiting"] == sorted([R, C])
    events.clear()
    _publish_queue(real.root)
    maintenance.run(real.root)  # the next child: the capture first, then every export
    assert events[0] == ("capture", run_id) and sorted(thread for _kind, thread in events[1:]) == [THREAD, B], events
    assert maintenance_jobs.job(real.root, retirement.JOB)[rt.JOB_FIELD]["waiting"] == []


# ---- one boundary across a task's operations: bound releases, then fresh threads ------------------------------------


def _lost(thread_id, body):
    raise ClaudexorUnavailable("daemon_unreachable", "lost before arrival", status_code=0)


def _stop_on_first_read(world, monkeypatch):
    stop, reads, read = {"on": False}, [], world.engine.thread_retirement

    def reading(thread_id, **kwargs):  # the stop arrives while the first operation is read back
        reads.append(kwargs["operation_id"])
        stop["on"] = True
        return read(thread_id, **kwargs)

    monkeypatch.setattr(world.engine, "thread_retirement", reading)
    report = rt.retire_finished_review_threads(world.root, engine=world.engine, live=lambda _task: False, census=set(),
                                               should_stop=lambda: stop["on"])
    monkeypatch.setattr(world.engine, "thread_retirement", read)
    return report, reads


def _release(world, thread):
    return (read_document(world.root, TASK).get(rt.RELEASES) or {}).get(thread) or {}


def test_a_stop_between_two_bound_releases_leaves_the_second_bound_for_the_next_pass(world, monkeypatch):
    _second_seat(world, monkeypatch)["on"] = False
    monkeypatch.setattr(world.engine, "retire_thread", _lost)
    assert world.run()["unknown"] == 2  # both operations bound, their outcomes unknown
    first, second = sorted(_release(world, thread)["operation_id"] for thread in (THREAD, B))
    world.engine.calls.clear()
    report, reads = _stop_on_first_read(world, monkeypatch)
    assert len(reads) == 1 and report.get("stopped") and report["waiting"] == 1, (reads, report)
    untouched = ({first, second} - set(reads)).pop()
    assert untouched not in [operation for _kind, operation in world.engine.calls]
    world.engine.calls.clear()
    world.run()
    assert ("read", untouched) in world.engine.calls  # the same operation, read back first


def test_a_stop_after_a_bound_release_leaves_a_fresh_thread_unexported(world, monkeypatch):
    failing = _second_seat(world, monkeypatch)  # B's export fails on the first pass: it stays fresh
    world.engine.faults.add("lost_before")
    first = world.run()
    assert (first["unknown"], first["waiting"]) == (1, 1) and not _release(world, B), first
    failing["on"] = False
    world.engine.calls.clear()
    report, reads = _stop_on_first_read(world, monkeypatch)
    assert ("export", B) not in world.engine.calls and not _release(world, B), report  # the boundary held
    assert reads == [_release(world, THREAD)["operation_id"]] and _release(world, THREAD)["state"] == "accepted"
    assert report.get("stopped") and report["waiting"] == 1, report
    world.engine.purged, world.engine.accepted = False, None  # B is another thread of the mock's one store
    assert world.run()["accepted"] == 1 and _release(world, B)["state"] == "accepted"


# ---- an archive item the child's compression rewrote ------------------------------------------------------------


@pytest.mark.skipif(sys.platform != "darwin" or not artifact_storage.compression_supported(),
                    reason="the native macOS compression writer; elsewhere nothing replaces the item")
@pytest.mark.parametrize("left", ["held", "bound"])
def test_an_item_the_childs_compression_rewrote_is_synced_before_a_release_relies_on_it(real, monkeypatch, left):
    """The same child's ``record_compression`` (after ``runtime_retirement``) replaces a retained item's inode; the
    next child's release reuses a held archive's item or resends a bound operation, never fetching it again."""
    plane, path, _data = real.engine.files["run-1"]
    real.engine.files["run-1"] = (plane, path, b'{"seq":1}\n' * 20000)  # past the compression floor
    _own_thread(real.root, R, R, real.record_id)
    _row(real.root, R, "completed")
    _attach(real, monkeypatch)
    _jobs(monkeypatch, "delegate_capture", "runtime_retirement", "record_compression")
    if left == "held":
        real.engine.corrupt.add("nat-1")  # one item arrives short: the verified rest is kept
    else:
        real.engine.faults.add("lost_before")  # bound, its retire request lost
    synced, fsync = [], os.fsync
    monkeypatch.setattr(os, "fsync", lambda fd: synced.append((os.fstat(fd).st_dev, os.fstat(fd).st_ino)) or fsync(fd))
    _publish_queue(real.root)
    retirement.record_startup(real.root, gaps=False)
    maintenance.run(real.root)
    assert real.release()["state"] == left
    receipt = rt.read_thread_export(real.root, R, real.release()["receipt"])
    ref = next(row["ref"] for row in receipt["items"] if row["item_id"] == "run-1")
    item = artifacts.task_artifact_dir_path(real.root, R).joinpath(*pathlib.PurePosixPath(ref["path"]).parts)
    info = item.stat()
    assert info.st_flags & stat.UF_COMPRESSED and (info.st_dev, info.st_ino) in synced  # replaced, synced first
    real.engine.corrupt.clear()
    real.engine.calls.clear()
    _capture_request(real, monkeypatch)
    _publish_queue(real.root)
    maintenance.run(real.root)
    assert real.release()["state"] == "accepted" and ("item", "run-1") not in real.engine.calls
    sink = io.BytesIO()
    rt.stream_retained_item(real.root, R, rt.read_thread_export(real.root, R, real.release()["receipt"]), "run-1", sink)
    assert sink.getvalue() == real.engine.files["run-1"][2]
