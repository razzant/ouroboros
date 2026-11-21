"""The machine is the limit (pool-capacity D, owner Q1 = D): physical admission, one count, the optional owner cap.

Counts, never a clock. The machine is a ``MemorySnapshot`` stub (as ``tests/test_test_parallelism.py`` stubs it);
the pool is the real one (``tests/test_owner_wait_pool.pool``: real ``Worker`` slots, the real ``assign_tasks``).
Every guard is shown both ways in its test.
"""

from __future__ import annotations

import json
import pathlib
import time
import threading

_REAL_THREAD = threading.Thread

import pytest

from ouroboros.host_memory import MemorySnapshot
from supervisor import conversation_lane, pool_admission, queue, worker_pool_lifecycle as lifecycle, workers
from tests.test_owner_wait_pool import InputQueue, Process, park, pool as pool  # noqa: F401 -- fixture
from tests.worker_pool_support import affirm_ready

GIB = 2 ** 30
_REAL_LANE_RECONCILE = conversation_lane.reconcile_conversation_lane  # the pool fixture stubs the lane's maintenance


@pytest.fixture
def assigning(pool, monkeypatch):  # noqa: F811 -- the imported fixture is this one's input
    from supervisor import evolution_lifecycle, state

    monkeypatch.setattr(state, "budget_remaining", lambda *a, **kw: 100)
    monkeypatch.setattr(evolution_lifecycle, "evolution_block_reason", lambda: "")
    monkeypatch.setenv("OUROBOROS_MAX_WORKERS", "")  # no number: the machine is the limit
    monkeypatch.setattr(pool_admission, "_READY", {"seq": 0, "generation": None, "rss_bytes": None})
    monkeypatch.setattr(pool_admission, "_FACTS", {})
    return pool


def task(tid, **fields):
    return {"id": tid, "type": "task", "chat_id": 0, "text": "work", **fields}


def child(tid, parent="owner", depth=1):
    return task(tid, delegation_role="subagent", root_task_id="owner", parent_task_id=parent, depth=depth)


def machine(monkeypatch, available_gib, *, total_gib=10, pressure=False, calls=None):
    """The host reading: ``total_gib`` with ``available_gib`` reclaimable; a reserve of 10% (1 GiB of 10)."""
    def physical_memory():
        if calls is not None:
            calls.append((queue._queue_lock._is_owned(), lifecycle._WORKER_LIFECYCLE_LOCK._is_owned()))
        return MemorySnapshot(int(total_gib * GIB), int(available_gib * GIB), pressure)

    monkeypatch.setattr(pool_admission, "physical_memory", physical_memory)


def starts(reading=None):
    with queue._queue_lock:
        return lifecycle.admissible_starts(pool_admission.read() if reading is None else reading)


def booting_slot(wid, lane="project"):
    slot = workers.Worker(wid, Process(500 + wid), InputQueue(), reaping=True, booting=True, lane=lane)
    workers.WORKERS[wid] = slot
    return slot


def readiness_park(wid):
    slot = booting_slot(wid)
    slot.proc.alive = False
    lifecycle._replace_or_park_unready_slot(wid, slot, 0, 0.0, lifecycle.WORKER_READY_MAX_ATTEMPTS)
    assert slot.readiness_exhausted
    return slot


@pytest.mark.parametrize("reading, admitted", [(None, 0), (pool_admission.PROBE, 1)])
def test_one_readiness_park_leaves_a_waiting_parent_its_floor_and_two_stop_growth(
        assigning, monkeypatch, reading, admitted):
    from ouroboros.context_runtime_facts import _queue_context_fact
    from ouroboros.engine_facts import engine_pool_line

    park(assigning)
    machine(monkeypatch, 0.1)  # no room: the floor is the waiting parent's one start (critical pressure: below)
    assert starts() == 1
    memory_fact = pool_admission.facts()
    with queue._queue_lock:
        assert lifecycle.admissible_starts(reading) == admitted
    assert pool_admission.facts() == memory_fact, "passes without a reading retain ordinary memory facts"
    readiness_park(1)
    workers.PENDING.append(child("child"))
    workers.assign_tasks()
    assert len(assigning.created) == 1, "one park cannot deadlock a healthy parent waiting for its queued child"
    failed = workers.WORKERS[2]
    failed.proc.alive = False
    lifecycle._replace_or_park_unready_slot(2, failed, 0, 0.0, lifecycle.WORKER_READY_MAX_ATTEMPTS)
    assert pool_admission.parked(lifecycle.project_slots()) == 2
    for _ in range(2):
        workers.assign_tasks()
    assert len(assigning.created) == 1 and workers.worker_pool_admission_state()["available"]
    queue.persist_queue_snapshot("readiness_finiteness")
    fact = _queue_context_fact({"budget_drive_root": str(assigning.root)})["memory_admission"]
    assert fact["read_at"] and fact["starts_admitted"] == 0
    assert fact["reason"] == "worker_readiness_finiteness", "publish the refusal, not the older memory reason"
    assert "worker_readiness_finiteness" in engine_pool_line(assigning.root)
    # An already-booting worker's success resets the count, even though both exhausted slots remain.
    fresh = booting_slot(3)
    assert lifecycle._open_ready_slot(3, fresh, {"pid": fresh.proc.pid}, "", 0, 0.0, 1)
    assert pool_admission.parked(lifecycle.project_slots()) == 0
    with queue._queue_lock:
        assert lifecycle.admissible_starts(reading) == admitted
    assert pool_admission.facts() == {}, "a ready retracts the finiteness refusal even without a memory reading"
    queue.persist_queue_snapshot("readiness_recovered")
    assert "worker_readiness_finiteness" not in engine_pool_line(assigning.root)
    workers.assign_tasks()
    assert workers.RUNNING["child"]["worker_id"] == 3


@pytest.mark.parametrize("state", ["acting", "waiting", "idle"])
def test_thirty_ready_processes_keep_growing_through_a_readiness_blip(assigning, monkeypatch, state):
    machine(monkeypatch, 90, total_gib=100)
    for wid in range(30):
        tid = f"live-{wid}"
        slot = affirm_ready(workers.Worker(wid, Process(500 + wid), InputQueue(), busy_task_id=tid))
        slot.rss = {"rss_bytes": GIB}
        workers.WORKERS[wid] = slot
        workers.RUNNING[tid] = {"task": task(tid), "worker_id": wid}
        if state == "waiting":
            slot.active_capacity = False
            workers.RUNNING[tid]["owner_wait"] = {"state": "waiting"}
        elif state == "idle":
            slot.busy_task_id = None
            del workers.RUNNING[tid]
    workers.RUNNING.pop("owner")
    readiness_park(30)
    assert starts() > 0, "acting, waiting and idle ready processes all sustain growth through a blip"
    workers.PENDING.extend(task(f"new-{i}") for i in range(31 if state == "idle" else 1))
    workers.assign_tasks()
    workers.assign_tasks()
    assert len(assigning.created) == 1


@pytest.mark.parametrize("state", ["dead", "unready", "reaping", "conversation", "inline"])
def test_only_live_ready_project_processes_allow_growth_past_a_park(assigning, state):
    slot = assigning.original
    readiness_park(1)
    if state == "dead":
        slot.proc.alive = False
    elif state == "unready":
        slot.ready_identity = None
    elif state == "reaping":
        slot.reaping = True
    elif state == "conversation":
        slot.lane = "conversation"
    else:
        slot.executor = "inline_degraded"
    assert starts(pool_admission.PROBE) == 0
    assert pool_admission.facts()["reason"] == "worker_readiness_finiteness"
    slot.proc.alive, slot.reaping, slot.lane, slot.executor = True, False, "project", "worker"
    affirm_ready(slot)
    assert starts() > 0


def test_a_broken_checkout_stops_after_one_park_and_is_exhausted(assigning):
    workers.WORKERS.clear()
    # The dead parent's unsettled row keeps teardown pending while admission already sees exhaustion.
    readiness_park(1)
    workers.PENDING.append(task("unstarted"))
    for _ in range(2):
        assert starts(pool_admission.PROBE) == 0
        workers.assign_tasks()
    assert not assigning.created
    assert workers._worker_pool_execution_state()["disabled_reason"] == "worker_readiness_exhausted"
    workers.RUNNING.clear()
    assert lifecycle.disable_exhausted_worker_pool()
    assert not workers._worker_pool_execution_state()["available"]


# ---- D-a/D-b: available - R >= T * (B + 1) per start; the floor; pressure; an unknown reading --------------------

def test_room_admits_k_starts_and_starting_processes_of_both_lanes_take_their_share(assigning, monkeypatch):
    """T = 1 GiB (the parent's own heartbeat size), R = 1 GiB of 10: room for k admits exactly k; each process still
    starting, project or conversation, takes one T of it."""
    assigning.original.rss = {"rss_bytes": GIB, "rss_measure": "current"}
    for k in (0, 1, 3):
        machine(monkeypatch, 1 + k)
        assert starts() == k, k
    machine(monkeypatch, 1 + 3)
    booting_slot(5)
    assert starts() == 2, "a starting project process reduces the room"
    booting_slot(6, lane="conversation")
    assert starts() == 1, "a starting conversation carrier is inside B too"
    facts = pool_admission.facts()
    assert (facts["available_bytes"], facts["reserve_bytes"], facts["typical_worker_bytes"],
            facts["starting_processes"], facts["starts_admitted"], facts["reason"]) == (
        4 * GIB, GIB, GIB, 2, 1, "memory_room")


def test_confirmed_critical_pressure_admits_no_start_and_an_unread_verdict_admits_as_before(assigning, monkeypatch):
    """Pressure refuses only a new process. A confirmed critical verdict admits none, the floor included (owner
    choice, 2026-11-14); without it the floor (nothing acting, nothing starting) still admits exactly one, with room
    or without, and one starting process takes it back. An unread verdict (None) is published as unknown and admits
    exactly what a reading without critical pressure admits: unknown is neither critical nor normal."""
    assigning.original.rss = {"rss_bytes": GIB, "rss_measure": "current"}
    machine(monkeypatch, 9, pressure=True)
    assert starts() == 0 and pool_admission.facts()["reason"] == "memory_pressure_critical"
    machine(monkeypatch, 0.5)  # no room at all
    assert starts() == 0 and pool_admission.facts()["reason"] == "memory_full"
    for pressure, published in ((False, "not_critical"), (None, "unknown")):
        machine(monkeypatch, 9, pressure=pressure)
        facts = pool_admission.facts() if starts() == 8 else pytest.fail(f"room decides beside {published}")
        assert (facts["pressure"], facts["reason"]) == (published, "memory_room")
    workers.RUNNING.clear()
    assigning.original.busy_task_id = None
    assigning.original.ready_identity = None  # not idle either: nothing acts, nothing starts
    machine(monkeypatch, 9, pressure=True)
    assert starts() == 0, "nothing acts or starts, and still a confirmed critical verdict admits no start"
    facts = pool_admission.facts()
    assert facts["read_at"] and (facts["pressure"], facts["starts_admitted"], facts["reason"]) == (
        "critical", 0, "memory_pressure_critical")
    for pressure, available in ((False, 0.5), (None, 0.5), (None, 9)):
        machine(monkeypatch, available, pressure=pressure)
        assert starts() == (1 if available < 1 else 8), "the floor admits one start; an unread verdict, as before"
    machine(monkeypatch, 0.5)
    booting_slot(5)
    assert starts() == 0, "one starting process: the floor is spent"


def test_a_waiting_parents_child_waits_out_critical_pressure_visibly_stoppably_and_starts_when_it_clears(
        assigning, monkeypatch):
    """The floor once started a new process for a parent that waits on its own process (holding its memory) for
    its queued child, even under critical pressure. Now that child waits while the kernel confirms critical: no
    process starts, the published facts date the wait and name its reason for the mind (runtime facts, the pool
    line its schedule_subagent result carries) and the owner (/status), Stop settles a waiting row before it starts
    as any queued row, the
    conversation lane still gets its carrier, and the first pass whose reading is not critical starts the child
    with no other actor's help. Reuse and a warm return never ask (``test_an_idle_process_and_a_warm_return_go_at_
    zero_room``)."""
    from ouroboros import cancel_intents
    from ouroboros.context_runtime_facts import _queue_context_fact
    from ouroboros.engine_facts import engine_pool_line
    from ouroboros.task_results import load_task_result, write_task_result
    from supervisor.state import status_text

    monkeypatch.setattr(workers, "_emit_task_done_terminal", lambda *args, **kwargs: True)
    park(assigning)  # the parent waits on its own process for what it queued
    calls = []
    machine(monkeypatch, 9, pressure=True, calls=calls)
    for tid in ("child", "stopped"):
        write_task_result(assigning.root, tid, "scheduled")
        workers.PENDING.append(child(tid))
    for _tick in range(3):
        workers.assign_tasks()
    assert not assigning.created and [row["id"] for row in workers.PENDING] == ["child", "stopped"]
    assert calls == [(False, False)] * 3, "one reading a pass, outside Q and the serializer: no busy loop"
    queue.persist_queue_snapshot("memory_pressure_wait")
    fact = _queue_context_fact({"budget_drive_root": str(assigning.root)})["memory_admission"]
    assert fact["read_at"] and (fact["pressure"], fact["starts_admitted"], fact["reason"]) == (
        "critical", 0, "memory_pressure_critical")
    line = engine_pool_line(assigning.root)
    assert "kernel pressure critical" in line and "new processes admitted 0 (memory_pressure_critical)" in line
    owner = status_text(workers.WORKERS, workers.PENDING, workers.RUNNING).splitlines()  # the owner's /status
    assert f"memory_admission: no new project process starts: the kernel reports critical memory pressure (read " \
           f"{fact['read_at']}); queued work that needs one waits and starts by itself once a reading is not " \
           "critical; Stop ends a queued task" in owner and owner.index("pending: 2") + 1 == owner.index(
               next(row for row in owner if row.startswith("memory_admission: ")))
    cancel_intents.request_cancel(assigning.root, "stopped", reason="owner stop while it waited")
    workers.assign_tasks()
    assert [row["id"] for row in workers.PENDING] == ["child"] and not assigning.created
    assert load_task_result(assigning.root, "stopped")["status"] == "cancelled", "Stop ended a waiting row"
    monkeypatch.setattr(conversation_lane, "reconcile_conversation_lane", _REAL_LANE_RECONCILE)
    monkeypatch.setattr(conversation_lane, "_ARMED_TABLE", id(workers.WORKERS))  # a started generation
    lifecycle.reconcile_worker_capacity(lambda: 0, pool_admission.read())
    lane = [slot for slot in workers.WORKERS.values() if slot.lane == "conversation"]
    assert len(lane) == 1 and assigning.created == [lane[0].proc], "the conversation lane stays outside admission"
    machine(monkeypatch, 9)  # the verdict clears
    workers.assign_tasks()
    assert len(assigning.created) == 2 and [row["id"] for row in workers.PENDING] == ["child"]
    assert pool_admission.facts()["reason"] == "memory_room", "the next pass admitted the start on its own"
    owner = status_text(workers.WORKERS, workers.PENDING, workers.RUNNING)
    assert "no new project process starts" not in owner and "new project process(es) admitted (memory_room), " \
           "kernel pressure not_critical (read " in owner


def _held_rows(root) -> list:
    path = root / "logs" / "supervisor.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
    return [row for row in rows if row.get("type") == "worker_replacement_held"]


def _dies_idle(slot):
    """The slot's process died with no task left on it (a crash, a reap, jetsam): the reaper's respawn comes next."""
    slot.proc.alive = False
    slot.busy_task_id = None
    workers.RUNNING.pop("owner", None)


def _lane_spare():
    workers.WORKERS[7] = affirm_ready(workers.Worker(7, Process(507), InputQueue(), lane="conversation"))
    return workers.WORKERS[7]


def test_a_dead_project_slot_is_not_replaced_under_confirmed_critical_and_the_next_pass_admits_a_process(
        assigning, monkeypatch):
    """A replacement is bound to no task (once ready it takes the first queued row, new work included), so a confirmed
    critical verdict holds it like any start. Through the reaper's own respawn the dead slot is forgotten on a reading
    taken outside Q and the lifecycle serializer, and is never told as a refused respawn; the queued
    retry and new work wait; the next ordinary pass whose reading is not critical starts a process by the old policy."""
    from supervisor import task_reaper

    monkeypatch.setattr(task_reaper, "_RESPAWN_REFUSALS", {})
    spare, calls, original = _lane_spare(), [], assigning.original
    machine(monkeypatch, 9, pressure=True, calls=calls)
    _dies_idle(original)
    task_reaper._respawn_after_reap(queue, workers, 0, expected_worker=original)
    assert calls == [(False, False)], "one reading, outside both locks"
    assert not assigning.created and 0 not in workers.WORKERS and workers.WORKERS == {7: spare}
    assert task_reaper._RESPAWN_REFUSALS == {}, "a held replacement is no refused respawn"
    assert [(row["outcome"], row["reason"]) for row in _held_rows(assigning.root)] == [
        ("retired", "memory_pressure_critical")]
    assert workers.worker_pool_admission_state()["available"], "the table keeps its member: the pool stays enabled"
    workers.PENDING.extend([task("owner-retry", _attempt=2), task("new-work")])
    for _tick in range(2):
        workers.assign_tasks()
    assert not assigning.created and pool_admission.facts()["reason"] == "memory_pressure_critical"
    machine(monkeypatch, 9)  # the verdict clears: the ordinary pass admits, by the old policy
    workers.assign_tasks()
    (fresh,) = [slot for slot in workers.WORKERS.values() if slot.booting]
    assert len(assigning.created) == 1 and fresh.lane == "project"
    fresh.reaping = False
    affirm_ready(fresh)
    workers.assign_tasks()
    assert workers.RUNNING["owner-retry"]["worker_id"] == fresh.wid


@pytest.mark.parametrize("pressure, lane", [(None, "project"), (False, "project"), (True, "conversation")],
                         ids=["unread-verdict", "not-critical", "conversation-lane"])
def test_unknown_and_not_critical_replace_as_before_and_the_conversation_lane_is_never_held(
        assigning, monkeypatch, pressure, lane):
    from supervisor import task_reaper

    calls, original = [], assigning.original
    machine(monkeypatch, 9, pressure=pressure, calls=calls)
    original.lane = lane
    _dies_idle(original)
    task_reaper._respawn_after_reap(queue, workers, 0, expected_worker=original)
    fresh = workers.WORKERS[0]
    assert fresh is not original and fresh.booting and len(assigning.created) == 1 and not _held_rows(assigning.root)
    assert calls == ([] if lane == "conversation" else [(False, False)]), "the lane never reads the machine"


def test_the_tables_last_member_stays_dead_through_critical_and_is_replaced_in_place_after_it(assigning, monkeypatch):
    """Growth needs a member, so the table's dead last member is never forgotten: under a confirmed critical verdict
    it stays, handed back to the crash detector (reaping cleared) and decided again each tick, one row for the whole
    hold, the pool enabled; ``retire_or_replace_worker`` (an idle retirement, a dead sleeper) holds it alike; the first
    reading that is not critical replaces it in place, as before."""
    from supervisor import task_reaper

    machine(monkeypatch, 9, pressure=True)
    original = assigning.original
    _dies_idle(original)
    for _tick in range(2):
        original.reaping = True  # the crash detector's hand-back
        task_reaper._respawn_after_reap(queue, workers, 0, expected_worker=original)
        assert workers.WORKERS == {0: original} and not original.reaping and not original.booting
    assert lifecycle.retire_or_replace_worker(0, original) == "" and workers.WORKERS == {0: original}
    assert [row["outcome"] for row in _held_rows(assigning.root)] == ["kept_last_member"]
    assert not assigning.created and workers.worker_pool_admission_state()["available"]
    machine(monkeypatch, 9)
    task_reaper._respawn_after_reap(queue, workers, 0, expected_worker=original)
    assert workers.WORKERS[0] is not original and workers.WORKERS[0].booting and len(assigning.created) == 1


def test_a_failed_readiness_respawn_is_held_too_never_parked_or_counted_as_exhaustion(assigning, monkeypatch):
    """Not the table's last member, it stays too: its spent attempt waits on it for the crash detector."""
    _lane_spare()
    machine(monkeypatch, 9, pressure=True)
    slot = booting_slot(5)
    slot.proc.alive = False  # its stop is confirmed; one attempt is spent
    lifecycle._replace_or_park_unready_slot(5, slot, 0, 0.0, 1)
    assert workers.WORKERS[5] is slot and not slot.readiness_exhausted and not assigning.created
    assert not slot.booting and not slot.reaping and slot.replacement_ready_attempt == 2
    assert workers._worker_pool_execution_state()["disabled_reason"] == ""
    assert [row["outcome"] for row in _held_rows(assigning.root)] == ["kept_readiness_attempt"]


def test_readiness_attempts_beside_a_lane_spare_count_through_critical_holds_and_park(assigning, monkeypatch):
    """Astra r1: a non-last slot past its first readiness attempt was retired under critical and the next start
    counted from 1, so a never-ready checkout restarted forever. Through the crash detector's own respawn its count
    now reaches each replacement and the bound parks it as on a calm machine; one row per hold."""
    from types import SimpleNamespace

    from supervisor import task_reaper

    workers.WORKERS.clear()
    workers.RUNNING.clear()
    lane, attempts = _lane_spare(), []
    monkeypatch.setattr(lifecycle.threading, "Thread",
                        lambda **kw: attempts.append(kw["args"][2]) or SimpleNamespace(start=lambda: None))
    machine(monkeypatch, 9)
    lifecycle.reconcile_worker_capacity(lambda: 1, pool_admission.read())
    while True:
        (slot,) = lifecycle.project_slots()
        slot.proc.alive = False  # confirmed teardown after this readiness attempt timed out
        machine(monkeypatch, 9, pressure=True)
        lifecycle._replace_or_park_unready_slot(slot.wid, slot, 0, 0.0, attempts[-1])
        if slot.readiness_exhausted:
            break
        assert workers.WORKERS == {lane.wid: lane, slot.wid: slot} and not slot.reaping
        machine(monkeypatch, 9)
        slot.reaping = True  # the crash detector's hand-back
        task_reaper._respawn_after_reap(queue, workers, slot.wid, expected_worker=slot)
    assert attempts == list(range(1, lifecycle.WORKER_READY_MAX_ATTEMPTS + 1))
    assert workers._worker_pool_execution_state()["disabled_reason"]
    assert [row["outcome"] for row in _held_rows(assigning.root)] == (
        ["kept_readiness_attempt"] * (lifecycle.WORKER_READY_MAX_ATTEMPTS - 1))


def test_an_unknown_reading_admits_one_starting_process_at_a_time_or_the_owners_cap(assigning, monkeypatch):
    """No reading (an unsupported platform, a failed probe, an unreadable cgroup limit): one starting process at a
    time, said as memory_unmeasured; with the owner's cap set, the cap decides. Nothing is kept: the next pass reads."""
    monkeypatch.setattr(pool_admission, "physical_memory", lambda: None)
    reading = pool_admission.read()
    assert reading.memory is None and reading.unmeasured.startswith("memory_unmeasured: ")
    assert starts(reading) == 1 and pool_admission.facts()["reason"].startswith("memory_unmeasured: ")
    booting_slot(5)
    assert starts(reading) == 0, "one at a time"
    monkeypatch.setenv("OUROBOROS_MAX_WORKERS", "4")
    assert starts(reading) == 2, "the owner's cap: 4 less one acting and one starting"
    machine(monkeypatch, 9)
    assigning.original.rss = {"rss_bytes": GIB, "rss_measure": "current"}
    assert starts() == 2 and pool_admission.facts()["status"] == "measured", "the next pass reads again"


def test_the_typical_size_is_the_heartbeat_mean_else_this_generations_ready_size_else_only_the_floor(
        assigning, monkeypatch):
    machine(monkeypatch, 9)
    assigning.original.rss = None  # no report yet
    assert pool_admission.typical_size(workers.WORKERS.values()) == (None, "unknown")
    assert starts() == 0 and pool_admission.facts()["reason"] == "worker_size_unknown", "only the floor admits"
    pool_admission.note_ready({"rss_bytes": GIB // 4})
    assert pool_admission.typical_size(workers.WORKERS.values()) == (GIB // 4, "worker_ready")
    carrier = affirm_ready(workers.Worker(3, Process(503), InputQueue(), lane="conversation"))
    inline = affirm_ready(workers.Worker(4, Process(504), InputQueue(), executor="inline_degraded"))
    workers.WORKERS.update({3: carrier, 4: inline})
    assigning.original.rss = {"rss_bytes": GIB, "rss_measure": "current"}
    carrier.rss = {"rss_bytes": 3 * GIB, "rss_measure": "footprint"}
    inline.rss = {"rss_bytes": 40 * GIB, "rss_measure": "current"}  # the server's own size: never a worker's
    assert pool_admission.typical_size(workers.WORKERS.values()) == (2 * GIB, "heartbeat_mean")
    carrier.proc.alive = False  # a dead process reports nothing
    assert pool_admission.typical_size(workers.WORKERS.values()) == (GIB, "heartbeat_mean")


def test_an_idle_process_and_a_warm_return_go_at_zero_room(assigning, monkeypatch):
    """Reuse and return never ask the machine: with no room at all, the idle worker takes queued work and a parked
    author's wake continues on its own process."""
    from supervisor import worker_owner_wait

    calls = []
    machine(monkeypatch, 0.1, pressure=True, calls=calls)
    park(assigning)
    idle = affirm_ready(workers.Worker(1, Process(43), InputQueue()))
    workers.WORKERS[1] = idle
    workers.PENDING.append(task("queued"))
    worker_owner_wait.handle_owner_wait({**assigning.event, "phase": "resume"}, workers)
    workers.assign_tasks()
    assert assigning.original.in_q.get_nowait()["phase"] == "resume_granted"
    assert workers.RUNNING["queued"]["worker_id"] == 1 and calls == [], "nothing was read: nothing new started"
    workers.PENDING.append(task("another"))  # the other direction: work no process carries asks the machine
    workers.assign_tasks()
    assert len(calls) == 1 and not assigning.created and "another" not in workers.RUNNING


# ---- D-f: once a pass, outside Q and the serializer, only when chosen work needs a new process -----------------

def test_the_machine_is_read_once_a_pass_outside_q_and_the_serializer_and_never_beside_an_idle_process(
        assigning, monkeypatch):
    calls = []
    machine(monkeypatch, 9, calls=calls)
    assigning.original.rss = {"rss_bytes": GIB, "rss_measure": "current"}
    workers.assign_tasks()
    assert calls == [], "nothing queued: nothing read"
    workers.PENDING.extend([task("held", _owner_hold={"reason": "owner"})])
    workers.assign_tasks()
    assert calls == [], "a held row is no work: nothing read"
    workers.PENDING.extend(task(f"t{i}") for i in range(5))
    workers.assign_tasks()
    assert calls == [(False, False)], "one reading, outside Q and outside the lifecycle serializer"
    assert len(assigning.created) == 5
    print(f"MEMORY_READINGS_PER_PASS {{'idle': 0, 'nothing_queued': 0, 'held_only': 0, 'growth': {len(calls)}}}")
    for slot in workers.WORKERS.values():
        slot.reaping = False
        affirm_ready(slot)
    del calls[:]
    workers.assign_tasks()  # five idle processes take the five rows
    assert calls == [] and len(workers.RUNNING) == 6
    from supervisor import worker_assignment

    def contended(_queue, row, _worker, deferred=None):  # the launch defers it: the same work retries, still demand
        deferred.add(row["id"])
        return False

    workers.WORKERS[99] = affirm_ready(workers.Worker(99, Process(2147483299), InputQueue()))
    workers.PENDING.append(task("deferred"))
    monkeypatch.setattr(worker_assignment, "_claim_worker_launch", contended)
    workers.assign_tasks()
    assert calls == [] and "deferred" not in workers.RUNNING, "an idle process remains: a deferred row reads nothing"


def test_byte_noise_keeps_the_census_and_a_changed_verdict_moves_it(assigning, monkeypatch):
    from supervisor.queue_publication import _census, _pool_counts

    assigning.original.rss = {"rss_bytes": GIB, "rss_measure": "current"}
    machine(monkeypatch, 5)
    starts()
    with queue._queue_lock:
        census = _census()
    machine(monkeypatch, 5.003)  # a few megabytes moved
    starts()
    with queue._queue_lock:
        assert _census() == census, "byte noise moves no revision"
        assert _pool_counts()["memory_admission"]["available_bytes"] == int(5.003 * GIB), "the facts are fresh"
    machine(monkeypatch, 5, pressure=True)  # the verdict changes
    starts()
    with queue._queue_lock:
        assert _census() != census


def test_the_published_facts_reach_the_mind_dated_and_the_worker_never_reads_the_machine(assigning, monkeypatch,
                                                                                       tmp_path):
    from ouroboros.context_runtime_facts import _queue_context_fact
    from ouroboros.engine_facts import engine_pool_line

    assigning.original.rss = {"rss_bytes": GIB, "rss_measure": "current"}
    machine(monkeypatch, 4)
    workers.PENDING.append(task("work"))
    workers.assign_tasks()
    queue.persist_queue_snapshot("facts")
    document = json.loads((tmp_path / "state" / "queue_snapshot.json").read_text(encoding="utf-8"))
    monkeypatch.setattr(pool_admission, "physical_memory", lambda: pytest.fail("a reader measured the machine"))
    fact = _queue_context_fact({"budget_drive_root": str(tmp_path)})
    line = engine_pool_line(tmp_path)
    memory = fact["memory_admission"]
    assert memory == document["memory_admission"] and memory["read_at"]
    assert (memory["starts_admitted"], memory["starting_processes"], memory["reason"]) == (3, 0, "memory_room")
    assert fact["recorded_max_workers"] is None and fact["acting_worker_count"] == 1
    # The shipped per-root subagent number is none: the fact says so as null, and the note says what null means.
    assert fact["max_active_subagents_per_root"] is None and "max_active_subagents_per_root" in fact["note"]
    assert "owner cap none (the machine's memory is the limit)" in line
    assert "available 4.0 GiB of 10.0 GiB, reserve 1.0 GiB" in line and "new processes admitted 3 (memory_room)" in line


# ---- D8: one count; the owner's cap counts acting tasks; the root's limit counts acting children -----------------

def test_the_one_count_on_each_kind_of_slot(assigning):
    from supervisor.queue_publication import _pool_counts

    now = time.time()
    meta = workers.RUNNING["owner"]  # slot 0 acts
    slots = {1: ("warm", {"owner_wait": {"state": "waiting"}}), 2: ("in_slot", {"active_operation_leases": {
        "in_slot_wait:tasks:x": {"kind": "in_slot_wait", "until_ts": now + 60}}}),
        3: ("lapsed_lease", {"active_operation_leases": {"in_slot_wait:mail:y": {"kind": "in_slot_wait",
                                                                                  "until_ts": now - 1}}}),
        4: ("claim", None)}
    for wid, (tid, facts) in slots.items():
        slot = affirm_ready(workers.Worker(wid, Process(40 + wid), InputQueue(), busy_task_id=tid))
        workers.WORKERS[wid] = slot
        if facts is not None:
            workers.RUNNING[tid] = {**meta, "task": task(tid), "worker_id": wid, **facts}
    parked = affirm_ready(workers.Worker(5, Process(45), InputQueue(), busy_task_id="stopped"))
    parked.active_capacity = False  # its park flag stands while a Stop's custody took the row
    workers.WORKERS[5] = parked
    workers.WORKERS[6] = affirm_ready(workers.Worker(6, Process(46), InputQueue()))  # idle
    booting_slot(7)
    workers.WORKERS[8] = affirm_ready(workers.Worker(8, Process(48), InputQueue(), busy_task_id="turn",
                                                     lane="conversation"))
    counts = _pool_counts()
    assert {key: counts[key] for key in ("worker_total", "busy_worker_count", "acting_worker_count",
                                         "parked_worker_count", "assignable_idle_workers", "booting_count",
                                         "lane_carrier_total")} == {
        "worker_total": 8, "busy_worker_count": 6, "acting_worker_count": 3, "parked_worker_count": 3,
        "assignable_idle_workers": 1, "booting_count": 1, "lane_carrier_total": 1}


@pytest.mark.parametrize("wait", ["warm", "in_slot", "cold"])
def test_cap_one_a_waiting_parent_lets_its_children_start_wide_and_deep(assigning, monkeypatch, wait):
    """C = 1: a parent that waits (warm on its process, in its slot, or cold without one) acts no more, so its child
    starts; of a wide fan-out one child acts at a time; a child that waits in turn lets its own child start."""
    monkeypatch.setenv("OUROBOROS_MAX_WORKERS", "1")
    assigning.original.rss = {"rss_bytes": GIB, "rss_measure": "current"}
    machine(monkeypatch, 9)
    meta = workers.RUNNING["owner"]
    if wait == "warm":
        park(assigning)
    elif wait == "in_slot":
        meta["active_operation_leases"] = {"in_slot_wait:tasks:c": {"kind": "in_slot_wait",
                                                                    "until_ts": time.time() + 600}}
    else:  # the root sleeps without a process: its row left RUNNING and its worker is free
        workers.RUNNING.clear()
        assigning.original.busy_task_id = None
    workers.PENDING.extend([child("c1"), child("c2"), child("c3")])
    for _tick in range(3):
        workers.assign_tasks()
        for slot in workers.WORKERS.values():
            if slot.booting:
                slot.reaping = False
                affirm_ready(slot)
    children = sorted(tid for tid in workers.RUNNING if tid.startswith("c"))
    assert children == ["c1"], "one child acts under the cap of one; its parent's wait counts nothing"
    # Deep: the running child waits in its slot for its own child, which then starts.
    workers.RUNNING["c1"]["active_operation_leases"] = {"in_slot_wait:tasks:g": {
        "kind": "in_slot_wait", "until_ts": time.time() + 600}}
    workers.PENDING.insert(0, child("g1", parent="c1", depth=2))
    for _tick in range(3):
        workers.assign_tasks()
        for slot in workers.WORKERS.values():
            if slot.booting:
                slot.reaping = False
                affirm_ready(slot)
    assert "g1" in workers.RUNNING and "c2" not in workers.RUNNING, "the cap's one acting task is the grandchild"


def test_the_root_child_limit_counts_acting_children(assigning, monkeypatch):
    from ouroboros import config

    monkeypatch.setattr(config, "get_max_active_subagents_per_root", lambda: 1)
    running = [{"task": child("a")}, {"task": child("b"), "owner_wait": {"state": "waiting"}},
               {"task": child("c"), "active_operation_leases": {"in_slot_wait:mail:z": {
                   "kind": "in_slot_wait", "until_ts": time.time() + 60}}},
               {"task": {**child("d"), "root_task_id": "other"}}, {"task": task("root-row")}]
    assert pool_admission.acting_children("owner", running) == 1, "a waiting child holds no share of the limit"
    running[0]["owner_wait"] = {"state": "waiting"}
    assert pool_admission.acting_children("owner", running) == 0
    machine(monkeypatch, 9)
    assigning.original.rss = {"rss_bytes": GIB, "rss_measure": "current"}
    workers.RUNNING["kid"] = {**workers.RUNNING["owner"], "task": child("kid"), "worker_id": 9,
                              "owner_wait": {"state": "waiting"}}
    workers.PENDING.append(child("next"))
    workers.assign_tasks()
    assert len(assigning.created) == 1, "the waiting child leaves the limit's one place to the next child"
    del workers.RUNNING["kid"]["owner_wait"]  # the other direction: it acts again, the next child waits
    workers.PENDING.append(child("third"))
    workers.assign_tasks()
    assert len(assigning.created) == 1


@pytest.mark.parametrize("number", [None, "6"])
def test_the_shipped_default_admits_a_roots_seventh_child_through_scheduling_and_assignment(assigning, monkeypatch,
                                                                                         number):
    """No monkeypatched number: the getter reads the shipped default (no number) or the owner's 6, and both the
    scheduling decision and worker selection take it as it is; no number holds nothing behind six acting children."""
    from supervisor.events_schedule_task import _active_cap_state
    from ouroboros.config import get_max_active_subagents_per_root

    from ouroboros import config

    monkeypatch.delenv("OUROBOROS_MAX_ACTIVE_SUBAGENTS_PER_ROOT", raising=False)
    if number is None:  # a fresh install, as the server loads and projects it: the shipped default reaches the reader
        monkeypatch.setattr(config, "SETTINGS_PATH", assigning.root / "absent-settings.json")
        config.apply_settings_to_env(config.load_settings(), environ=(environ := {}))
        if "OUROBOROS_MAX_ACTIVE_SUBAGENTS_PER_ROOT" in environ:
            monkeypatch.setenv("OUROBOROS_MAX_ACTIVE_SUBAGENTS_PER_ROOT", environ["OUROBOROS_MAX_ACTIVE_SUBAGENTS_PER_ROOT"])
    else:
        monkeypatch.setenv("OUROBOROS_MAX_ACTIVE_SUBAGENTS_PER_ROOT", number)
    machine(monkeypatch, 9)
    assigning.original.rss = {"rss_bytes": GIB, "rss_measure": "current"}
    for index in range(6):
        workers.RUNNING[f"kid{index}"] = {**workers.RUNNING["owner"], "task": child(f"kid{index}"),
                                          "worker_id": 20 + index}
    seventh = child("seventh")
    workers.PENDING.append(seventh)
    blocked, refusal = _active_cap_state("owner", "owner", [], workers.RUNNING, get_max_active_subagents_per_root())
    assert (blocked, refusal) == (number is not None, ""), "scheduling: queued only behind the owner's number"
    workers.assign_tasks()
    assert len(assigning.created) == (0 if number else 1), "assignment: the seventh starts while the pool admits it"
    if number:  # the owner removes the number (No limit: the server's projection drops the variable)
        monkeypatch.delenv("OUROBOROS_MAX_ACTIVE_SUBAGENTS_PER_ROOT")
        workers.assign_tasks()
        assert len(assigning.created) == 1, "the held seventh starts at the next pass"


def test_a_slots_wait_flag_counts_the_same_for_the_owner_and_child_caps(assigning, monkeypatch):
    from ouroboros import config

    monkeypatch.setattr(config, "get_max_active_subagents_per_root", lambda: 1)
    slot = assigning.original
    workers.RUNNING["owner"]["task"] = child("owner")
    workers.PENDING.append(child("next"))
    assert lifecycle.pool_slot_counts()["acting_worker_count"] == 1
    assert pool_admission.acting_children("owner", workers.RUNNING.values()) == 1
    workers.assign_tasks()
    assert not assigning.created
    slot.active_capacity = False  # Stop custody can hold the row after the wait metadata has left it.
    assert lifecycle.pool_slot_counts()["acting_worker_count"] == 0
    assert pool_admission.acting_children("owner", workers.RUNNING.values()) == 0
    assert pool_admission.acting_children("owner", [{"task": child("selected")}]) == 1, "a projected launch acts"
    workers.assign_tasks()
    assert len(assigning.created) == 1


# ---- the owner's number: optional, read live, migration A ----------------------------------------------------------

_WORKERS, _PER_ROOT = "OUROBOROS_MAX_WORKERS", "OUROBOROS_MAX_ACTIVE_SUBAGENTS_PER_ROOT"


@pytest.mark.parametrize(("key", "saved", "marked", "read"), [
    (_WORKERS, 10, False, None),  # the form's 10, saved on every save: no choice
    (_WORKERS, "10", False, None),
    (_WORKERS, 40, False, 40),  # the owner's own number stays his cap
    (_WORKERS, 1, False, 1),  # off Colab a 1 is a choice (Colab's first-launch 1: tests/test_colab_bootstrap.py)
    (_WORKERS, "", False, None),
    (_WORKERS, "abc", False, 10),  # a typo is finite, never "no cap"
    (_WORKERS, 0, False, 10),
    (_WORKERS, 10, True, 10),  # after the format marker, 10 is the owner's again
    # The per-root subagent number: every save wrote the shipped 6 before it became optional.
    (_PER_ROOT, 6, False, None),
    (_PER_ROOT, "6", False, None),
    (_PER_ROOT, 64, False, 64),  # the owner's own number stays
    (_PER_ROOT, "", False, None),
    (_PER_ROOT, "abc", False, 6),  # a typo is finite, never "no number"
    (_PER_ROOT, 0, False, 6),
    (_PER_ROOT, 6, True, 6),  # a 6 the new Settings saved carries the marker and is the owner's
])
def test_a_saved_worker_number_reads_as_migration_a_says_and_never_repeats(tmp_path, monkeypatch, key, saved, marked,
                                                                         read):
    from ouroboros import config
    from ouroboros.runtime_limits import get_max_active_subagents_per_root, get_max_workers
    from ouroboros.settings_scales import OWNER_CAP_FORMAT_KEY, SUBAGENT_CAP_FORMAT_KEY

    marker, getter = {_WORKERS: (OWNER_CAP_FORMAT_KEY, get_max_workers),
                      _PER_ROOT: (SUBAGENT_CAP_FORMAT_KEY, get_max_active_subagents_per_root)}[key]
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({key: saved, **({marker: 1} if marked else {})}), encoding="utf-8")
    monkeypatch.setattr(config, "SETTINGS_PATH", path)
    monkeypatch.delenv(key, raising=False)
    loaded = config.load_settings()
    assert loaded[marker] == 1, "the view carries the marker into the next save"
    config.apply_settings_to_env(loaded, environ=(environ := {}))
    if key in environ:  # the server's projection, which the supervisor reads on every pass
        monkeypatch.setenv(key, environ[key])
    assert getter() == read
    config.save_settings(loaded)  # a read stays a read; the first save writes the marker: never again
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored[marker] == 1 and config.load_settings()[key] == ("" if read is None else read)


@pytest.mark.parametrize("fixture", ["settings_v6.113.4.json", "settings_v6.87.5.json"])
def test_an_earlier_release_document_reads_its_saved_six_as_no_number(tmp_path, monkeypatch, fixture):
    from ouroboros import config
    from ouroboros.settings_scales import SUBAGENT_CAP_FORMAT_KEY

    document = json.loads((pathlib.Path(__file__).parent / "fixtures" / "nminus1" / fixture).read_text(encoding="utf-8"))
    assert document[_PER_ROOT] == 6 and SUBAGENT_CAP_FORMAT_KEY not in document, "the release saved its default"
    path = tmp_path / "settings.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    monkeypatch.setattr(config, "SETTINGS_PATH", path)
    monkeypatch.delenv(_PER_ROOT, raising=False)
    loaded = config.load_settings()
    assert loaded[_PER_ROOT] == "" and loaded[SUBAGENT_CAP_FORMAT_KEY] == 1
    config.save_settings(loaded)
    assert config.load_settings()[_PER_ROOT] == "", "the next save keeps no number"


@pytest.mark.parametrize("provenance", [False, True])
@pytest.mark.parametrize("key, legacy", [(_WORKERS, 10), (_PER_ROOT, 6)])
def test_legacy_worker_cap_projection_does_not_undo_migration(tmp_path, monkeypatch, provenance, key, legacy):
    from ouroboros import config
    from ouroboros.runtime_limits import get_max_active_subagents_per_root, get_max_workers
    from ouroboros.settings_integrity import PROJECTED_SETTINGS_ENV
    from ouroboros.settings_scales import OWNER_CAP_FORMAT_KEY, SUBAGENT_CAP_FORMAT_KEY

    marker, getter = {_WORKERS: (OWNER_CAP_FORMAT_KEY, get_max_workers),
                      _PER_ROOT: (SUBAGENT_CAP_FORMAT_KEY, get_max_active_subagents_per_root)}[key]
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({key: legacy}), encoding="utf-8")
    monkeypatch.setattr(config, "SETTINGS_PATH", path)
    inherited = {}
    config.apply_settings_to_env({**config.SETTINGS_DEFAULTS, key: legacy}, environ=inherited)
    monkeypatch.setenv(key, inherited[key])
    monkeypatch.delenv(PROJECTED_SETTINGS_ENV, raising=False)
    if provenance:
        monkeypatch.setenv(PROJECTED_SETTINGS_ENV, inherited[PROJECTED_SETTINGS_ENV])
    loaded = config.load_settings()
    assert loaded[marker] == 1 and loaded[key] == ""
    config.apply_settings_to_env(loaded, environ=inherited)
    assert key not in inherited
    monkeypatch.delenv(key)
    config.save_settings(loaded)
    assert getter() is None
    assert json.loads(path.read_text(encoding="utf-8"))[key] == ""
    config.save_settings({**loaded, key: legacy})
    assert config.load_settings()[key] == legacy, "after migration, a chosen legacy number stays chosen"


def test_the_environment_configures_only_an_unauthored_worker_cap(tmp_path, monkeypatch):
    from ouroboros import config
    from ouroboros.gateway.owner_settings import _owner_update_settings, fields_on_saved_document

    path = tmp_path / "settings.json"
    path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(config, "SETTINGS_PATH", path)
    monkeypatch.setenv("OUROBOROS_MAX_WORKERS", "40")
    loaded = config.load_settings()
    assert loaded["OUROBOROS_MAX_WORKERS"] == 40
    config.save_settings(loaded)
    assert "OUROBOROS_MAX_WORKERS" not in json.loads(path.read_text(encoding="utf-8"))
    monkeypatch.setenv("OUROBOROS_MAX_WORKERS", "20")
    assert config.load_settings()["OUROBOROS_MAX_WORKERS"] == 20
    _owner_update_settings(fields_on_saved_document({"OUROBOROS_MAX_WORKERS": ""}),
                           authored_keys=["OUROBOROS_MAX_WORKERS"])
    assert config.load_settings()["OUROBOROS_MAX_WORKERS"] == "", "a saved blank wins over the environment too"


@pytest.mark.parametrize("door", ["reaper", "readiness", "deferred_readiness", "idle"])
@pytest.mark.parametrize("change", ["fence", "replacement"])
def test_slow_replacement_probe_holds_neither_lock_and_revalidates_identity(assigning, monkeypatch, door, change):
    """A native read stalls; another thread can still stop/change the generation.
    Returning a healthy reading never revives the captured old membership.
    """
    from supervisor import task_reaper

    monkeypatch.setattr(threading, "Thread", _REAL_THREAD)
    entered, release = threading.Event(), threading.Event()
    old = assigning.original
    _dies_idle(old)
    old.booting = door == "readiness"
    old.retiring = door == "idle"
    old.quiescence_token = "owned"
    job = {"worker": old, "worker_id": old.wid, "drive_root": str(assigning.root),
           "owner_chat_id": 0, "started": 0, "attempt": 1, "token": "owned"}
    def slow_read():
        assert not queue._queue_lock._is_owned()
        assert not lifecycle._WORKER_LIFECYCLE_LOCK._is_owned()
        entered.set()
        assert release.wait(10)
        return pool_admission.Reading(MemorySnapshot(10 * GIB, 9 * GIB, False), "", "now")
    monkeypatch.setattr(pool_admission, "read", slow_read)
    starts = []
    monkeypatch.setattr(lifecycle, "_spawn_worker_slot", lambda *a, **kw: starts.append(a))
    errors = []
    def run():
        try:
            if door == "reaper":
                task_reaper._respawn_after_reap(queue, workers, old.wid, expected_worker=old)
            elif door == "readiness":
                lifecycle._replace_unready_slot(old.wid, old, 0, time.time(), 1)
            elif door == "deferred_readiness":
                lifecycle.finish_unready_teardown(job)
            else:
                lifecycle.reap_idle_worker(job)
        except BaseException as exc:
            errors.append(exc)
    thread = _REAL_THREAD(target=run)
    thread.start()
    try:
        assert entered.wait(5), errors
        assert lifecycle._WORKER_LIFECYCLE_LOCK.acquire(timeout=1), "native probe blocks control"
        try:
            with queue._queue_lock:
                if change == "fence":
                    workers._WORKER_POOL_DISABLED_REASON = "managed_update"
                else:
                    workers.WORKERS[old.wid] = affirm_ready(workers.Worker(old.wid, Process(9876), InputQueue()))
        finally:
            lifecycle._WORKER_LIFECYCLE_LOCK.release()
    finally:
        release.set()
        thread.join(10)
    assert not thread.is_alive() and not errors and not starts
    assert workers.WORKERS[old.wid] is old if change == "fence" else workers.WORKERS[old.wid] is not old


def test_dead_last_parked_member_never_probes_on_warm_return_tick(assigning, monkeypatch):
    from supervisor import worker_owner_wait, task_reaper

    old = assigning.original
    _dies_idle(old)
    old.active_capacity = False
    calls = []
    machine(monkeypatch, 9, pressure=True, calls=calls)
    worker_owner_wait.grant_warm_returns()
    assert calls == [] and workers.WORKERS == {old.wid: old}
    assert not old.reaping and not old.retiring
    task_reaper._respawn_after_reap(queue, workers, old.wid, expected_worker=old)
    assert calls == [(False, False)] and not assigning.created


def test_readiness_attempt_survives_a_held_last_member(assigning, monkeypatch):
    from supervisor import task_reaper
    old = assigning.original
    _dies_idle(old)
    machine(monkeypatch, 9, pressure=True)
    lifecycle._replace_or_park_unready_slot(old.wid, old, 0, 0.0, 1)
    assert old.replacement_ready_attempt == 2
    attempts = []
    monkeypatch.setattr(lifecycle, "_spawn_worker_slot", lambda *a, **kw: attempts.append(kw["ready_attempt"]) or True)
    machine(monkeypatch, 9)
    task_reaper._respawn_after_reap(queue, workers, old.wid, expected_worker=old)
    assert attempts == [2]


def test_reaper_without_a_captured_slot_never_claims_a_later_replacement(assigning, monkeypatch):
    from supervisor import task_reaper

    workers.WORKERS.clear()
    monkeypatch.setattr(workers, "respawn_worker", lambda *a, **k: pytest.fail("no captured slot to replace"))
    task_reaper._respawn_after_reap(queue, workers, assigning.original.wid)


@pytest.mark.parametrize("door", ["reaper", "stranded_cancel"])
def test_failed_replacement_never_clears_a_new_slots_reaping_marker(assigning, monkeypatch, door):
    from supervisor import task_reaper, task_lifecycle

    old = assigning.original
    _dies_idle(old)
    old.busy_task_id, old.reaping = "settled", True
    newer = workers.Worker(old.wid, Process(9876), InputQueue())
    newer.reaping = True
    def failed(wid, *, expected_worker):
        assert expected_worker is old
        with queue._queue_lock:
            workers.WORKERS[wid] = newer
        raise RuntimeError("controlled late replacement failure")
    monkeypatch.setattr(workers, "respawn_worker", failed)
    monkeypatch.setattr(task_lifecycle, "_reconcile_dead_review_owner", lambda *args: None)
    monkeypatch.setattr(task_lifecycle, "_reaping_owner_abandoned", lambda intent: True)
    if door == "reaper":
        task_reaper._respawn_after_reap(queue, workers, old.wid)
    else:
        task_lifecycle._recover_stranded_reaping_slot(queue, "settled", {"claim_owner": "dead"})
    assert workers.WORKERS[old.wid] is newer and newer.reaping


def _held_after_its_crash(assigning, monkeypatch):
    """The table's last member dies with its task; its crash retry moved the task back to PENDING (as
    ``worker_health._readmit_crash_retry`` pops RUNNING and enqueues) and the reaper's respawn reads critical."""
    from supervisor import task_reaper, workspace_preparation

    monkeypatch.setattr(pool_admission, "_READY", {"seq": 1, "generation": workspace_preparation.current_generation(),
                                                   "rss_bytes": GIB})  # this generation's worker size is known
    held = assigning.original
    machine(monkeypatch, 9, pressure=True)
    held.proc.alive = False
    workers.RUNNING.pop("owner")
    workers.PENDING.append(task("owner", _attempt=2))
    task_reaper._respawn_after_reap(queue, workers, 0, expected_worker=held)
    return held


@pytest.mark.parametrize("later", ["not_critical", "critical"])
def test_a_held_slot_drops_its_settled_task_so_the_retry_elsewhere_never_strands_it(assigning, monkeypatch, later):
    """Opus review: a held dead slot kept its crashed task's ``busy_task_id``. When the retry ran on a process grown on
    a non-critical reading, the crash detector's next job for the held slot carried the retry's RUNNING row, was never
    current, and the slot stayed ``reaping`` and dead in the table, never replaced or retired. Held, it starts as its
    replacement would, with no task: the next job is current; beside the grown process a calm reading replaces it and
    a critical one retires it, and the retry keeps its own process."""
    from supervisor import worker_health

    held = _held_after_its_crash(assigning, monkeypatch)
    assert workers.WORKERS == {0: held} and not held.reaping
    task_at_hold = held.busy_task_id  # the mechanism, checked after the outcome: a tree without it reaches the strand
    machine(monkeypatch, 9)  # the dispatch pass reads calm and grows a process for the retry
    workers.assign_tasks()
    (fresh,) = [slot for slot in workers.WORKERS.values() if slot is not held]
    fresh.reaping = False
    affirm_ready(fresh)
    workers.assign_tasks()
    retry = workers.RUNNING["owner"]
    assert retry["worker_id"] == fresh.wid and fresh.busy_task_id == "owner"
    machine(monkeypatch, 9, pressure=later == "critical")
    with queue._queue_lock:
        workers._ensure_workers_healthy_locked(queue)
    job = queue._reap_queue.get_nowait()
    assert job["worker"] is held
    worker_health.recover_confirmed_dead_worker(job)
    with queue._queue_lock:
        workers._ensure_workers_healthy_locked(queue)  # a stranded slot stays reaping: the detector skips it
    stranded = workers.WORKERS.get(0) is held and held.reaping and not held.proc.alive
    assert not stranded and workers.WORKERS.get(0) is not held, {  # replaced when calm, retired when critical
        "stranded": stranded, "reaping": held.reaping, "busy_task_id": held.busy_task_id, "job_task_id": job["task_id"],
        "detector_requeued": not queue._reap_queue.empty()}
    assert (0 in workers.WORKERS) is (later == "not_critical")
    assert task_at_hold is None and job["task_id"] == ""  # held as its replacement starts, so its job was current
    assert workers.RUNNING["owner"] is retry and fresh.busy_task_id == "owner" and workers.WORKERS[fresh.wid] is fresh
    assert [row["outcome"] for row in _held_rows(assigning.root)] == (
        ["kept_last_member"] + (["retired"] if later == "critical" else []))
    assert queue._reap_queue.empty() or queue._reap_queue.get_nowait()["worker"] is not fresh


def test_a_held_slot_keeps_a_task_whose_running_row_still_names_it(assigning, monkeypatch):
    """Ownership is never dropped while the task's RUNNING row still names the dead slot: held under critical, it
    keeps the task, and the crash detector's next job is current, so its death is still recovered."""
    from supervisor import worker_health

    held = assigning.original
    machine(monkeypatch, 9, pressure=True)
    held.proc.alive = False
    assert workers.RUNNING["owner"]["worker_id"] == 0
    reading = pool_admission.read()
    with lifecycle._WORKER_LIFECYCLE_LOCK:
        assert lifecycle._replacement_waits(0, held, reading) is True
    assert workers.WORKERS == {0: held} and held.busy_task_id == "owner" and not held.reaping
    with queue._queue_lock:
        workers._ensure_workers_healthy_locked(queue)
        job = queue._reap_queue.get_nowait()
        assert job["task_id"] == "owner" and worker_health._dead_job_is_current(job)
