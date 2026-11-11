"""The machine is the limit (pool-capacity D, owner Q1 = D): physical admission, one count, the optional owner cap.

Counts, never a clock. The machine is a ``MemorySnapshot`` stub (as ``tests/test_test_parallelism.py`` stubs it);
the pool is the real one (``tests/test_owner_wait_pool.pool``: real ``Worker`` slots, the real ``assign_tasks``).
Every guard is shown both ways in its test.
"""

from __future__ import annotations

import json
import time

import pytest

from ouroboros.host_memory import MemorySnapshot
from supervisor import pool_admission, queue, worker_pool_lifecycle as lifecycle, workers
from tests.test_owner_wait_pool import InputQueue, Process, park, pool as pool  # noqa: F401 -- fixture
from tests.worker_pool_support import affirm_ready

GIB = 2 ** 30


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
    machine(monkeypatch, 0.1, pressure=True)
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


def test_critical_pressure_admits_nothing_beside_acting_work_and_the_floor_one_start(assigning, monkeypatch):
    """Pressure refuses only a new process; the floor (nothing acting, nothing starting) still admits exactly one,
    with room or without, and one starting process takes it back."""
    assigning.original.rss = {"rss_bytes": GIB, "rss_measure": "current"}
    machine(monkeypatch, 9, pressure=True)
    assert starts() == 0 and pool_admission.facts()["reason"] == "memory_pressure_critical"
    machine(monkeypatch, 0.5)  # no room at all
    assert starts() == 0 and pool_admission.facts()["reason"] == "memory_full"
    workers.RUNNING.clear()
    assigning.original.busy_task_id = None
    assigning.original.ready_identity = None  # not idle either: nothing acts, nothing starts
    for pressure, available in ((True, 9), (False, 0.5)):
        machine(monkeypatch, available, pressure=pressure)
        assert starts() == 1, "the floor admits one start"
    booting_slot(5)
    assert starts() == 0, "one starting process: the floor is spent"


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

@pytest.mark.parametrize(("saved", "marked", "read"), [
    (10, False, None),  # the form's 10, saved on every save: no choice
    ("10", False, None),
    (40, False, 40),  # the owner's own number stays his cap
    (1, False, 1),  # off Colab a 1 is a choice (Colab's first-launch 1: tests/test_colab_bootstrap.py)
    ("", False, None),
    ("abc", False, 10),  # a typo is finite, never "no cap"
    (0, False, 10),
    (10, True, 10),  # after the format marker, 10 is the owner's again
])
def test_a_saved_worker_number_reads_as_migration_a_says_and_never_repeats(tmp_path, monkeypatch, saved, marked,
                                                                         read):
    from ouroboros import config
    from ouroboros.runtime_limits import get_max_workers
    from ouroboros.settings_scales import OWNER_CAP_FORMAT_KEY

    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"OUROBOROS_MAX_WORKERS": saved, **({OWNER_CAP_FORMAT_KEY: 1} if marked else {})}),
                    encoding="utf-8")
    monkeypatch.setattr(config, "SETTINGS_PATH", path)
    monkeypatch.delenv("OUROBOROS_MAX_WORKERS", raising=False)
    loaded = config.load_settings()
    assert loaded[OWNER_CAP_FORMAT_KEY] == 1, "the view carries the marker into the next save"
    config.apply_settings_to_env(loaded, environ=(environ := {}))
    if "OUROBOROS_MAX_WORKERS" in environ:  # the server's projection, which the supervisor reads on every pass
        monkeypatch.setenv("OUROBOROS_MAX_WORKERS", environ["OUROBOROS_MAX_WORKERS"])
    assert get_max_workers() == read
    config.save_settings(loaded)  # a read stays a read; the first save writes the marker: never again
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored[OWNER_CAP_FORMAT_KEY] == 1 and config.load_settings()["OUROBOROS_MAX_WORKERS"] == (
        "" if read is None else read)


@pytest.mark.parametrize("provenance", [False, True])
def test_legacy_worker_cap_projection_does_not_undo_migration(tmp_path, monkeypatch, provenance):
    from ouroboros import config
    from ouroboros.runtime_limits import get_max_workers
    from ouroboros.settings_integrity import PROJECTED_SETTINGS_ENV
    from ouroboros.settings_scales import OWNER_CAP_FORMAT_KEY

    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"OUROBOROS_MAX_WORKERS": 10}), encoding="utf-8")
    monkeypatch.setattr(config, "SETTINGS_PATH", path)
    inherited = {}
    config.apply_settings_to_env({**config.SETTINGS_DEFAULTS, "OUROBOROS_MAX_WORKERS": 10}, environ=inherited)
    monkeypatch.setenv("OUROBOROS_MAX_WORKERS", inherited["OUROBOROS_MAX_WORKERS"])
    monkeypatch.delenv(PROJECTED_SETTINGS_ENV, raising=False)
    if provenance:
        monkeypatch.setenv(PROJECTED_SETTINGS_ENV, inherited[PROJECTED_SETTINGS_ENV])
    loaded = config.load_settings()
    assert loaded[OWNER_CAP_FORMAT_KEY] == 1 and loaded["OUROBOROS_MAX_WORKERS"] == ""
    config.apply_settings_to_env(loaded, environ=inherited)
    assert "OUROBOROS_MAX_WORKERS" not in inherited
    monkeypatch.delenv("OUROBOROS_MAX_WORKERS")
    config.save_settings(loaded)
    assert get_max_workers() is None
    assert json.loads(path.read_text(encoding="utf-8"))["OUROBOROS_MAX_WORKERS"] == ""
    config.save_settings({**loaded, "OUROBOROS_MAX_WORKERS": 10})
    assert config.load_settings()["OUROBOROS_MAX_WORKERS"] == 10, "after migration, a chosen 10 stays chosen"


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
