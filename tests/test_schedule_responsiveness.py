"""Automatic schedule work never holds Q while it waits (Audit05 A).

Real consumers over an isolated root: the automatic pass (``automatic_acquisitions``
around ``check_scheduled_tasks``), a real SQLite EXCLUSIVE holder or a held table,
cancellation or launch lock, and competing Q consumers that never read the store.
Contention defers the same identity with no hold, audit or claim; corrupt and foreign
records keep their typed holds; the off-loop duty, its generation and the returning
stop doors bound what the pass may still admit.
"""
from __future__ import annotations

import contextlib
import copy
import json
import pathlib
import threading
import time

import pytest

from ouroboros import store
from ouroboros.task_results import load_task_result, write_task_result
from tests._store_journal_shared import exclusive_hold
from tests.test_g1_followup_policy import world as _world
from tests.test_schedule_occurrence import _row, _rows, q as _q

pytestmark = pytest.mark.serial
RealThread = threading.Thread
# Fixtures requested by name, re-bound as module attributes (a direct import of a name that
# reappears as a parameter is an F811 redefinition under the CI ruff gate).
q, world = _q, _world


@contextlib.contextmanager
def held_store(root):
    """A real independent SQLite connection holding the task store EXCLUSIVE, its reads busy too
    under WAL (``exclusive_hold``; it must exist: a connection to an absent path would create an
    empty, foreign file)."""
    assert store.store_path(root).exists()
    holder = exclusive_hold(store.store_path(root))
    try:
        yield
    finally:
        holder.execute("ROLLBACK")
        holder.close()


@contextlib.contextmanager
def held_lock(path):
    from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock

    fd = acquire_exclusive_file_lock(path, owner_aware_stale=True)
    assert fd is not None
    try:
        yield
    finally:
        release_exclusive_file_lock(path, fd)


def automatic_pass(stop=None):
    from supervisor.queue_schedules import automatic_acquisitions, check_scheduled_tasks

    with automatic_acquisitions():
        check_scheduled_tasks(stop=stop)


def run_off_thread(target, name):
    errors, done = [], threading.Event()

    def body():
        try:
            target()
        except BaseException as exc:  # surfaced by the asserting test
            errors.append(exc)
        finally:
            done.set()

    thread = RealThread(target=body, name=name, daemon=True)
    thread.start()
    return thread, done, errors


def events(root, kind="schedule_mutation"):
    path = root / "logs" / "events.jsonl"
    if not path.exists():
        return []
    return [row for row in map(json.loads, path.read_text(encoding="utf-8").splitlines()) if row.get("type") == kind]


def table_bytes(q):
    return (q.root / "state" / "scheduled_tasks.json").read_bytes()


def seed_history(q, count=7):
    """Disabled, settled rows whose last task each reads a receipt (the I09b shape)."""
    from supervisor import queue_schedules

    for index in range(count):
        write_task_result(q.root, f"old-{index}", "completed", result="ended")
    data = q.queue.load_schedule_store(q.root)
    data["tasks"].extend({"id": f"cron-{index}", "last_task_id": f"old-{index}", "enabled": False,
                          "followup_relation": {"kind": "independent"},
                          "trigger": {"type": "cron", "expr": "* * * * *"}} for index in range(count))
    queue_schedules._write_scheduled_tasks(data, q.root)


def add_last_task(q, schedule_id, task_id):
    from supervisor import queue_schedules

    data = q.queue.load_schedule_store(q.root)
    next(row for row in data["tasks"] if row["id"] == schedule_id)["last_task_id"] = task_id
    queue_schedules._write_scheduled_tasks(data, q.root)


def test_held_store_defers_the_pass_while_unrelated_queue_consumers_finish(q, monkeypatch):
    """Probe01 reversed: the held store no longer holds Q; nothing durable changes until release."""
    from ouroboros import task_results

    seed_history(q)
    _row(q, "admitted", intent={"kind": "system_repo"})
    automatic_pass()
    [admitted] = q.pending
    q.pending.clear()  # an accepted occurrence no longer in flight: its receipt must be read
    _row(q, "due", intent={"kind": "system_repo"})
    add_last_task(q, "due", "old-0")  # an enabled legacy row: its binding repair reads the store
    before, audits = table_bytes(q), len(events(q.root))
    entered, real_read, consumers = threading.Event(), task_results.load_task_result, []

    def read(root, task_id, **kwargs):
        if kwargs.get("signed"):  # the pass reached its off-Q receipt preparation
            assert not q.queue._queue_lock._is_owned()
            entered.set()
        return real_read(root, task_id, **kwargs)

    def unrelated():
        consumers.append(q.queue.enqueue_task({"id": "unrelated", "type": "task", "chat_id": 0,
                                               "text": "fixture", "_project_scope_none": True},
                                              restoring_snapshot=True))
        consumers.append(q.queue.persist_queue_snapshot(reason="audit05_consumer"))

    monkeypatch.setattr(task_results, "load_task_result", read)
    with held_store(q.root):
        scheduler, scheduled, errors = run_off_thread(automatic_pass, "audit05-schedule")
        assert entered.wait(5), "the pass never prepared its receipt reads"
        competitor, competed, competitor_errors = run_off_thread(unrelated, "audit05-enqueue")
        assert scheduled.wait(5) and competed.wait(5), "a held store must delay neither the pass nor Q readers"
        assert not errors and not competitor_errors, (errors, competitor_errors)
        assert consumers[1] is True and [t["id"] for t in q.pending] == ["unrelated"]
        assert table_bytes(q) == before and len(events(q.root)) == audits
        rows = _rows(q)
        assert not rows["due"].get("occurrence") and not rows["admitted"].get("hold")
    monkeypatch.setattr(task_results, "load_task_result", real_read)
    automatic_pass()  # released: the same occurrence republishes, the due row claims and admits once
    ids = [task["id"] for task in q.pending]
    assert ids.count(admitted["id"]) == 1 and len(ids) == 3
    rows = _rows(q)
    assert rows["due"]["occurrence"]["phase"] == "admitted" and not rows["due"].get("hold")
    assert rows["due"]["last_outcome"]["task_id"] == "old-0"
    assert not any(row.get("hold") for row in rows.values())


@pytest.mark.parametrize("which", ["table", "cancellation", "launch"])
def test_a_held_table_or_control_lock_defers_at_once(q, which, monkeypatch):
    from supervisor import followup_policy, queue_schedules

    _row(q, "independent", intent={"kind": "system_repo"})
    write_task_result(q.root, "origin", "running", root_task_id="origin")
    related = {"id": "related", "enabled": True, "source": "task_followup",
               "trigger": {"type": "once", "run_at": "2000-01-01T00:00:00+00:00"},
               "followup_relation": {"kind": "unknown-but-recorded"},
               "followup_origin": {"task_id": "origin", "root_task_id": "origin"},
               "task": {"type": "task", "text": "related", "chat_id": 42}}
    data = q.queue.load_schedule_store(q.root)
    data["tasks"].append(related)
    queue_schedules._write_scheduled_tasks(data, q.root)
    assert followup_policy.relation_kind(related) == "unknown"
    from ouroboros import owner_pause

    path = {"table": q.root / "state" / "scheduled_tasks.json.lock",
            "cancellation": q.root / "state" / "cancel_intents.json.lock",
            "launch": owner_pause._lock_path(q.root, "origin")}[which]
    before = _rows(q)["related"]
    from supervisor.queue_schedules import ScheduleLockTimeout
    from supervisor.schedule_occurrence import contention
    from ouroboros import platform_layer

    acquire, waits = platform_layer.acquire_exclusive_file_lock, []

    def observed_acquire(lock_path, **kwargs):
        started = time.monotonic()
        try:
            return acquire(lock_path, **kwargs)
        finally:
            if pathlib.Path(lock_path) == path:
                waits.append(time.monotonic() - started)

    with held_lock(path):
        monkeypatch.setattr(platform_layer, "acquire_exclusive_file_lock", observed_acquire)
        monkeypatch.setattr(queue_schedules, "acquire_exclusive_file_lock", observed_acquire)
        if which == "table":  # the whole pass defers; the duty treats this refusal as contention
            with pytest.raises(ScheduleLockTimeout) as refused:
                automatic_pass()
            assert contention(refused.value)
        else:
            automatic_pass()
    # Measure the real contended acquisition, not unrelated admission/database I/O.
    assert waits and max(waits) < 2.0, f"a held {which} lock must not wait under Q ({waits})"
    rows = _rows(q)
    if which == "table":
        assert q.pending == [] and "occurrence" not in rows["independent"]  # the whole pass waits
    else:
        assert [t["metadata"]["schedule_id"] for t in q.pending] == ["independent"]
        assert rows["related"] == before  # deferred: no hold, wait or audit written for contention
    queue_schedules.check_scheduled_tasks()  # the owner-path default still waits and then proceeds


def test_a_busy_followup_policy_read_persists_no_wait_and_no_audit(q, monkeypatch):
    """A related row's policy read under contention is not a restriction to persist."""
    from supervisor import followup_policy, queue_schedules

    write_task_result(q.root, "origin", "running", root_task_id="origin")
    row = {"id": "legacy", "enabled": True, "source": "task_followup",
           "trigger": {"type": "once", "run_at": "2000-01-01T00:00:00+00:00"},
           "followup_origin": {"task_id": "origin", "root_task_id": "origin"},
           "task": {"type": "task", "text": "legacy follow-up", "chat_id": 42,
                    "metadata": {"origin_task_id": "origin", "resource_intent": {"kind": "system_repo"}}}}
    data = q.queue.load_schedule_store(q.root)
    data["tasks"].append(row)
    queue_schedules._write_scheduled_tasks(data, q.root)
    audits, before = len(events(q.root)), _rows(q)["legacy"]
    with held_store(q.root):
        automatic_pass()
        view = followup_policy.policy_view(q.root, q.queue.load_schedule_store(q.root), before)
    assert view.get("busy") is True and view["wait"] == "followup_authority_unavailable"
    assert _rows(q)["legacy"] == before and len(events(q.root)) == audits and q.pending == []
    automatic_pass()
    assert [t["metadata"]["schedule_id"] for t in q.pending] == ["legacy"]


def test_corrupt_and_foreign_receipts_keep_their_typed_holds(q, monkeypatch):
    from supervisor import schedule_occurrence as occurrences

    _row(q, "a", intent={"kind": "system_repo"}, cron=True)
    _row(q, "b", intent={"kind": "system_repo"}, cron=True)
    automatic_pass()
    tasks = {t["metadata"]["schedule_id"]: t for t in q.pending}
    q.pending.clear()
    write_task_result(q.root, tasks["b"]["id"], "scheduled", schedule_admission={"token": "foreign", "schedule_id": "b"})
    real = occurrences._read_back

    def read(task_id, **kwargs):
        if task_id == tasks["a"]["id"]:
            raise ValueError("corrupt receipt")
        return real(task_id, **kwargs)

    monkeypatch.setattr(occurrences, "_read_back", read)
    automatic_pass()
    rows = _rows(q)
    assert rows["a"]["hold"]["reason"] == "occurrence_result_unreadable"
    assert rows["b"]["hold"]["reason"] == "occurrence_task_conflict"
    assert q.pending == []


def count_forks(monkeypatch):
    """The real memory fork (``headless.prepare_task_drive``), counted by task id."""
    from ouroboros import headless

    forks, real = [], headless.prepare_task_drive
    monkeypatch.setattr(headless, "prepare_task_drive",
                        lambda root, task_id, *a, **k: forks.append(task_id) or real(root, task_id, *a, **k))
    return forks


@pytest.mark.parametrize("arm", ["rolled_back", "committed_unread", "table_held"])
def test_contention_after_preparation_forks_once_and_admits_once(q, tmp_path, monkeypatch, arm):
    """1F/1G and a held schedule sidecar at the final admit: no hold; the same token and task
    id; one PENDING row, one receipt and one fork of the occurrence's drive after release."""
    from ouroboros import task_results
    from tests.test_schedule_occurrence import _project
    from supervisor import queue_schedules, schedule_occurrence as occurrences

    folder = tmp_path / "work"
    folder.mkdir()
    write_task_result(q.root, "seed", "completed", result="the store exists before it is held")
    _project(q, folder=folder)
    _row(q, "s1", intent={"kind": "explicit_resource", "root": str(folder)}, project_id="proj")
    forks = count_forks(monkeypatch)
    real_prepare, real_write = occurrences.prepare, task_results.write_task_result
    holder = contextlib.ExitStack()
    armed = [True]

    def prepare(claimed):
        prepared = real_prepare(claimed)
        if armed[0] and arm == "table_held":
            armed[0] = False
            holder.enter_context(held_lock(q.root / "state" / "scheduled_tasks.json.lock"))
        return prepared

    def write(root, task_id, status, **fields):
        if armed[0] and arm == "rolled_back" and fields.get("schedule_admission"):
            armed[0] = False
            holder.enter_context(held_store(q.root))
        stored = real_write(root, task_id, status, **fields)
        if armed[0] and arm == "committed_unread" and fields.get("schedule_admission"):
            armed[0] = False
            # The commit stands, but its acknowledgement was lost. No readback is
            # needed on success; this unknown outcome still reconciles next pass.
            raise store.StoreBusy("committed receipt reply unavailable", reason="contention")
        return stored

    monkeypatch.setattr(occurrences, "prepare", prepare)
    monkeypatch.setattr(task_results, "write_task_result", write)
    with holder:
        with pytest.raises(queue_schedules.ScheduleLockTimeout) if arm == "table_held" else contextlib.nullcontext():
            automatic_pass()
    assert q.pending == [] and not _rows(q)["s1"].get("hold") and len(occurrences._DEFERRED) == 1
    token = _rows(q)["s1"]["occurrence"]["token"]
    automatic_pass()
    [task] = q.pending
    receipt = load_task_result(q.root, task["id"])["schedule_admission"]
    assert receipt["token"] == token == task["metadata"]["schedule_occurrence"]["token"]
    assert forks == [task["id"]], "contention after preparation never forks the task drive again"
    assert task["workspace_root"] == str(folder.resolve()) and task["drive_root"]
    assert _rows(q)["s1"]["occurrence"]["phase"] == "admitted" and not occurrences._DEFERRED


@pytest.mark.parametrize("change", ["unchanged", "folder_removed", "room_moved"])
def test_a_deferred_preparation_resolves_its_resource_again(q, tmp_path, monkeypatch, change):
    """A kept preparation lends only its fork; the folder, its usability, the room's current
    default and the Project basis are read again, exactly as a fresh preparation reads them."""
    from ouroboros.projects_registry import update_project
    from tests.test_schedule_occurrence import _project
    from supervisor import schedule_occurrence as occurrences

    first, moved = tmp_path / "first", tmp_path / "moved"
    first.mkdir()
    moved.mkdir()
    _project(q, folder=first)
    intent = ({"kind": "room_default", "project_id": "proj"} if change == "room_moved"
              else {"kind": "explicit_resource", "root": str(first)})
    _row(q, "s1", intent=intent, project_id="proj")
    forks = count_forks(monkeypatch)
    real_billing, armed = q.queue.prepare_root_billing, [True]

    def billing(task):
        if armed[0]:
            armed[0] = False
            raise store.StoreBusy("money ledger held", reason="contention")
        return real_billing(task)

    monkeypatch.setattr(q.queue, "prepare_root_billing", billing)
    automatic_pass()
    assert q.pending == [] and len(occurrences._DEFERRED) == 1 and len(forks) == 1
    if change == "folder_removed":
        first.rmdir()
    elif change == "room_moved":
        update_project(q.root, "proj", working_dir=str(moved))
    automatic_pass()
    assert not occurrences._DEFERRED
    if change == "folder_removed":
        # Read again and now absent: the kept address starts with its availability fact (#1799),
        # and the same address keeps the fork its first preparation made.
        [task] = q.pending
        assert task["workspace_root"] == str(first.resolve())
        assert task["metadata"]["workspace_availability"]["state"] == "missing" and len(forks) == 1
        return
    [task] = q.pending
    assert task["workspace_root"] == str((moved if change == "room_moved" else first).resolve())
    assert len(forks) == (2 if change == "room_moved" else 1), "only a changed target forks again"
    assert task["project_id"] == "proj" and pathlib.Path(task["drive_root"]).is_dir()


@pytest.mark.parametrize("door", ["planned_restart", "owner_restart", "managed_update", "generation_stop"])
def test_a_closed_door_admits_nothing_and_the_next_generation_admits_the_same_claim(q, monkeypatch, door):
    """A returning stop that named its queue (or a stopping generation) leaves the claim unsubmitted."""
    from supervisor import schedule_occurrence as occurrences, workers

    _row(q, "s1", intent={"kind": "system_repo"})
    stop = [False]
    real_prepare = occurrences.prepare

    def prepare(claimed):
        result = real_prepare(claimed)
        if door == "generation_stop":
            stop[0] = True
        elif door == "managed_update":
            workers.close_repo_writer_admission("managed_update:audit05")
        else:
            q.queue.name_queue_returns(q.queue.PENDING, door)
        return result

    monkeypatch.setattr(occurrences, "prepare", prepare)
    automatic_pass(stop=lambda: stop[0])
    claim = _rows(q)["s1"]["occurrence"]
    assert q.pending == [] and claim["phase"] == "claimed" and not _rows(q)["s1"].get("hold")
    assert load_task_result(q.root, claim["task_id"]) is None
    monkeypatch.setattr(occurrences, "prepare", real_prepare)
    workers.open_repo_writer_admission()
    q.queue.init_queue_refs(q.pending, {}, {"value": 0})  # the next generation starts open
    automatic_pass()
    [task] = q.pending
    assert task["id"] == claim["task_id"]
    assert task["metadata"]["schedule_occurrence"]["token"] == claim["token"]


def test_restart_returns_close_admission_before_naming_the_queue(q, monkeypatch):
    """The owner's Restart door: an off-loop pass blocked in preparation cannot slip a row in
    after the restart transaction copied the queue it returns."""
    from ouroboros import server_restart
    from supervisor import restart_retention, schedule_occurrence as occurrences
    from types import SimpleNamespace

    _row(q, "s1", intent={"kind": "system_repo"})
    entered, release = threading.Event(), threading.Event()
    real_prepare = occurrences.prepare

    def prepare(claimed):
        entered.set()
        assert release.wait(5)
        return real_prepare(claimed)

    monkeypatch.setattr(occurrences, "prepare", prepare)
    named = []
    real_returns = restart_retention.prepare_restart_returns
    monkeypatch.setattr(restart_retention, "prepare_restart_returns",
                        lambda root, running, pending, **kw: named.append([t["id"] for t in pending]) or set())
    monkeypatch.setattr("ouroboros.owner_wait.prepare_owner_wait_handoffs", lambda *_a, **_k: set())
    thread, done, errors = run_off_thread(automatic_pass, "audit05-door")
    assert entered.wait(5)
    returning, owner_waits = server_restart._prepare_owner_restart_returns(
        SimpleNamespace(RUNNING={}, PENDING=q.queue.PENDING))
    release.set()
    assert done.wait(5) and not errors, errors
    assert named == [[]] and q.pending == [] and q.queue.RETURNS_NAMED == {"door": "owner_restart"}
    assert _rows(q)["s1"]["occurrence"]["phase"] == "claimed"
    assert real_returns is not None


def test_the_offloop_duty_never_holds_the_tick_and_runs_single_flight(q, monkeypatch):
    from ouroboros import server_maintenance
    from supervisor import queue_schedules

    entered, release, passes = threading.Event(), threading.Event(), []

    def slow_pass(*, stop=None):
        passes.append(queue_schedules.automatic())
        entered.set()
        assert release.wait(5)

    monkeypatch.setattr(queue_schedules, "check_scheduled_tasks", slow_pass)
    # Other cadences are not due in this tick: only the schedule duty starts. The maintenance child is owed on a
    # process's first look, so without this its thread would outlive the test whenever this test runs first.
    monkeypatch.setattr(server_maintenance, "_LAST_CANCEL_INTENT_SWEEP", [time.time()])
    monkeypatch.setattr(server_maintenance, "_LAST_COLD_CUSTODY_SWEEP", [time.time()])
    monkeypatch.setattr(server_maintenance, "_tick_maintenance_child", lambda: None)
    stop_event = threading.Event()
    stamps = [time.time(), time.time()]
    try:
        started = time.monotonic()
        server_maintenance._periodic_supervisor_maintenance(stamps, list(stamps), stop_event=stop_event)
        assert entered.wait(5)
        server_maintenance._periodic_supervisor_maintenance(stamps, list(stamps), stop_event=stop_event)
        assert time.monotonic() - started < 2.0, "the loop tick never waits for a schedule pass"
        assert passes == [True], "a running pass makes the next tick skip, never queue"
        assert q.queue.enqueue_task({"id": "owner-intake", "type": "task", "chat_id": 0, "text": "x",
                                     "_project_scope_none": True}, restoring_snapshot=True)
    finally:
        release.set()
    deadline = time.monotonic() + 5
    while server_maintenance._SCHEDULE_SWEEP_LOCK.locked() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not server_maintenance._SCHEDULE_SWEEP_LOCK.locked()
    stop_event.set()
    server_maintenance._periodic_supervisor_maintenance(stamps, list(stamps), stop_event=stop_event)
    assert passes == [True], "a closed generation starts no pass"


def test_billing_reads_never_hold_q(q, monkeypatch):
    from supervisor import queue as queue_module, queue_schedules

    observed = []
    real_billing = queue_module.prepare_root_billing

    def billing(task):
        observed.append(("billing", queue_module._queue_lock._is_owned()))
        return real_billing(task)

    monkeypatch.setattr(queue_module, "prepare_root_billing", billing)
    _row(q, "s1", intent={"kind": "system_repo"})
    data = q.queue.load_schedule_store(q.root)
    data["tasks"].append({"id": "skill-row", "enabled": True, "source": "skill_manifest", "skill": "demo",
                          "trigger": {"type": "once", "run_at": "2000-01-01T00:00:00+00:00"},
                          "task": {"type": "task", "text": "skill work", "chat_id": 42,
                                   "metadata": {"resource_intent": {"kind": "system_repo"}}}})
    queue_schedules._write_scheduled_tasks(data, q.root)
    automatic_pass()
    kinds = {kind for kind, _ in observed}
    assert {"billing"} == kinds and not any(held for _, held in observed), observed


@pytest.mark.parametrize("change", ["disable", "edit", "delete"])
def test_owner_changes_during_preparation_govern_the_unstarted_claim(q, monkeypatch, change):
    from supervisor import schedule_occurrence as occurrences

    _row(q, "s1", intent={"kind": "system_repo"})
    entered, release = threading.Event(), threading.Event()
    real_prepare = occurrences.prepare

    def prepare(claimed):
        entered.set()
        assert release.wait(5)
        return real_prepare(claimed)

    monkeypatch.setattr(occurrences, "prepare", prepare)
    thread, done, errors = run_off_thread(automatic_pass, "audit05-owner")
    assert entered.wait(5)
    if change == "edit":
        row = _rows(q)["s1"]
        q.queue.upsert_scheduled_task({**row, "task": {**row["task"], "text": "edited objective"}})
    else:
        q.queue.mutate_scheduled_task(change, "s1", reason="owner change", actor="owner:test", drive_root=q.root)
    release.set()
    assert done.wait(5) and not errors, errors
    assert q.pending == []
    rows = _rows(q)
    assert "s1" not in rows or "occurrence" not in rows["s1"]


def test_mixed_load_under_toggling_contention_admits_each_occurrence_once(world, monkeypatch):
    """Finite workload (50 independent + 20 related follow-ups), owner intake, holder toggling:
    each occurrence admits once, related ones with their origin's wallet and deadline."""
    from tests.test_g1_followup_policy import BINDING, DEADLINE, register

    from supervisor import queue_schedules

    _, root, pending = world
    shapes = {"independent": register(world, "independent"), "related": register(world, "related")}
    data = queue_schedules.load_schedule_store(root)  # the tool's per-task cap: clone its two real rows
    data["tasks"] = [{**copy.deepcopy(shapes[kind]), "id": f"{kind}-{index}", "name": f"{kind} {index}"}
                     for kind, count in (("independent", 50), ("related", 20)) for index in range(count)]
    queue_schedules._write_scheduled_tasks(data, root)
    audits = len(events(root))
    for round_ in range(6):
        if round_ % 2 == 0:
            # Off-Q reads (preparation, billing) may wait; bound them so the finite run stays short.
            with held_store(root), store.acquisition_budget(0.2):
                automatic_pass()
                assert q_enqueue(f"intake-{round_}")
        else:
            automatic_pass()
    for _ in range(3):
        automatic_pass()
    scheduled = [t for t in pending if t.get("metadata", {}).get("schedule_id")]
    assert len(scheduled) == 70 and len({t["metadata"]["schedule_id"] for t in scheduled}) == 70
    related = [t for t in scheduled if t["metadata"]["followup_relation"]["kind"] == "related"]
    assert len(related) == 20
    assert all(t["metadata"]["billing_group"] == BINDING and t["deadline_at"] == DEADLINE for t in related)
    rows = schedules_rows(root)
    assert not any(row.get("hold") or row.get("followup_wait") or row.get("followup_hold") for row in rows)
    assert len(events(root)) == audits, "contention wrote no audit rows"


def q_enqueue(task_id):
    from supervisor import queue

    return queue.enqueue_task({"id": task_id, "type": "task", "chat_id": 0, "text": "owner intake",
                               "_project_scope_none": True}, restoring_snapshot=True)


def schedules_rows(root):
    from supervisor import queue_schedules

    return queue_schedules.load_schedule_store(root)["tasks"]


def test_the_planned_restart_names_its_queue_after_closing_automatic_admission(q, monkeypatch):
    """The agent's planned restart (``server._perform_supervisor_restart``) closes the door in
    the same Q step that copies the queue it returns; a later automatic admission waits."""
    import server
    from supervisor import restart_retention
    from types import SimpleNamespace

    _row(q, "s1", intent={"kind": "system_repo"})
    named = []
    monkeypatch.setattr(restart_retention, "prepare_restart_returns",
                        lambda root, running, pending, **kw: named.append(
                            (dict(q.queue.RETURNS_NAMED), [t["id"] for t in pending])) or set())
    monkeypatch.setattr("ouroboros.owner_wait.prepare_owner_wait_handoffs", lambda *_a, **_k: set())
    monkeypatch.setattr("ouroboros.delegate_recovery.prepare_planned_restart_handoffs", lambda *_a, **_k: set())
    monkeypatch.setattr(server, "_request_restart_exit", lambda: None)
    monkeypatch.setattr(server, "_managed_update_pending_kwargs", lambda: {})
    ctx = SimpleNamespace(load_state=lambda: {}, safe_restart=lambda **_kw: (True, "ok"),
                          kill_workers=lambda **_kw: None, persist_queue_snapshot=lambda **_kw: True,
                          update_state=lambda _fn: None, DRIVE_ROOT=q.root, REPO_DIR=q.root,
                          RUNNING={}, PENDING=q.queue.PENDING)
    server._perform_supervisor_restart(ctx)
    assert named == [({"door": "planned_restart"}, [])]
    automatic_pass()
    claim = _rows(q)["s1"]["occurrence"]
    assert q.pending == [] and claim["phase"] == "claimed"


def test_a_planned_restart_that_raises_before_its_exit_reopens_admission(q, monkeypatch):
    """No exit follows a raised stop: the same generation keeps serving, so its schedules admit."""
    import server
    from supervisor import restart_retention
    from types import SimpleNamespace

    _row(q, "s1", intent={"kind": "system_repo"})
    monkeypatch.setattr(restart_retention, "prepare_restart_returns", lambda *_a, **_k: set())
    monkeypatch.setattr("ouroboros.owner_wait.prepare_owner_wait_handoffs", lambda *_a, **_k: set())
    monkeypatch.setattr("ouroboros.delegate_recovery.prepare_planned_restart_handoffs", lambda *_a, **_k: set())
    exits = []
    monkeypatch.setattr(server, "_request_restart_exit", lambda: exits.append(True))
    monkeypatch.setattr(server, "_managed_update_pending_kwargs", lambda: {})

    def kill_workers(**_kw):
        assert q.queue.RETURNS_NAMED == {"door": "planned_restart"}
        raise RuntimeError("worker shutdown raised")

    ctx = SimpleNamespace(load_state=lambda: {}, safe_restart=lambda **_kw: (True, "ok"),
                          kill_workers=kill_workers, persist_queue_snapshot=lambda **_kw: True,
                          update_state=lambda _fn: None, DRIVE_ROOT=q.root, REPO_DIR=q.root,
                          RUNNING={}, PENDING=q.queue.PENDING)
    with pytest.raises(RuntimeError):
        server._perform_supervisor_restart(ctx)
    assert exits == [] and q.queue.RETURNS_NAMED == {}
    automatic_pass()
    assert [t["metadata"]["schedule_id"] for t in q.pending] == ["s1"]
