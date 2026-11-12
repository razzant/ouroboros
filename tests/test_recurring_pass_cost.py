"""The server's recurring passes cost the same whatever the live set holds (cost law L1, L2, L3, L14).

``tests/test_cost_contract.py`` B7 holds every duty flat against HISTORY. These hold the same passes flat
against the LIVE SET, which is what grew on 01.11.2026: a second pass over an unchanged queue reads no task
body and no usage row, writes nothing, and costs the same at 1 and at many running, pending or paused rows. A
companion beside each law shows the pass still does its work when a real input changes, so a law cannot be
satisfied by a pass that stopped looking. Counting is not timing: no clock decides anything here.
"""
from __future__ import annotations

import time

import pytest

from tests._budget_pause_exact_helpers import _install_queue
from tests._cost_meter import CostMeter, tagged

# The work a pass may not repeat: connections, transactions, statements, decoded JSON (a decoded task
# body is one) and every write. The tag's count of each must be equal at both live-set sizes.
WORK = ("connect", "begins", "statements", "commits", "writes", "json_loads", "json_chars")


def work(counts: dict) -> dict:
    return {key: counts.get(key, 0) for key in WORK}


@pytest.fixture
def meter(tmp_path):
    with CostMeter(tmp_path) as installed:
        yield installed


def counted(meter: CostMeter, tag: str, call) -> dict:
    result = tagged(tag, call)
    meter.counts[tag]["_result"] = 0  # the tag exists even for a pass that did no counted work
    return {"counts": work(meter.snapshot(tag)["counts"]), "result": result,
            "statements": meter.snapshot(tag)["statements"]}


# ---- L3: the pre-assignment pass over PENDING ------------------------------------------------------

def _pending(tmp_path, monkeypatch, name: str, rows: int):
    from ouroboros.task_results import write_task_result

    root = tmp_path / name
    root.mkdir()
    queue, _state, workers = _install_queue(root, monkeypatch)
    for index in range(rows):
        task_id = f"{name}-{index}"
        write_task_result(root, task_id, "scheduled", chat_id=0, text="queued work " * 2000)
        workers.PENDING.append({"id": task_id, "type": "task", "chat_id": 0, "_attempt": 1})
    return root, queue, workers


def test_l3_the_pending_pass_reads_once_at_any_queue_size_and_never_a_body(tmp_path, monkeypatch, meter):
    from ouroboros import task_results

    def one_pass(name, rows):
        _root, queue, workers = _pending(tmp_path, monkeypatch, name, rows)
        monkeypatch.setattr(task_results, "load_task_result",
                            lambda *a, **k: pytest.fail("the pending pass read a task body"))
        with queue._queue_lock:
            outcome = counted(meter, f"L3:{name}", workers._drop_cancelled_pending)
        monkeypatch.undo()
        assert outcome["result"] is True
        return outcome

    small, large = one_pass("one", 1), one_pass("fifty", 50)
    assert small["counts"] == large["counts"], (small["counts"], large["counts"])
    assert small["counts"]["begins"] == 1 and small["counts"]["writes"] == 0, small


def test_l3_a_new_intent_still_settles_its_row_and_the_next_row_reads_the_moved_projection(
        tmp_path, monkeypatch, meter):
    """The companion: the one read per pass still acts, and the row AFTER a settlement sees what it changed."""
    from ouroboros import cancel_intents
    from ouroboros.task_results import STATUS_CANCELLED, load_task_result
    from supervisor import workers as workers_module

    root, queue, workers = _pending(tmp_path, monkeypatch, "live", 3)
    monkeypatch.setattr(workers_module, "_emit_task_done_terminal", lambda *a, **k: True)
    cancel_intents.request_cancel(root, "live-1", reason="owner Stop")
    reads = []
    real = cancel_intents.active_intents
    monkeypatch.setattr(cancel_intents, "active_intents",
                        lambda *a, **k: reads.append(1) or real(*a, **k))
    with queue._queue_lock:
        assert workers._drop_cancelled_pending() is True
    assert [row["id"] for row in workers.PENDING] == ["live-0", "live-2"]
    assert load_task_result(root, "live-1")["status"] == STATUS_CANCELLED
    assert not real(root), "the settled intent left the projection"
    reads.clear()
    with queue._queue_lock:
        assert workers._drop_cancelled_pending() is True
    assert len(reads) == 1, "a pass over rows nothing cancels reads the projection once"


@pytest.mark.parametrize("gap", ["unknown_status", "unreadable_projection"])
def test_l3_unknown_authority_never_dispatches(tmp_path, monkeypatch, meter, gap):
    from ouroboros import cancel_intents, store_hot_facts

    _root, queue, workers = _pending(tmp_path, monkeypatch, "gap", 1)
    if gap == "unknown_status":
        boom = store_hot_facts.statuses

        def failing(root, ids):
            answers = boom(root, ids)
            return {key: RuntimeError("store cannot answer") for key in answers}

        monkeypatch.setattr(store_hot_facts, "statuses", failing)
        with queue._queue_lock:
            assert workers._drop_cancelled_pending() is True
        assert workers.PENDING[0]["_terminalization_retry"]["trigger"] == "pending_result_authority"
        return

    def corrupt(*_a, **_k):
        raise cancel_intents.CancelIntentProjectionCorrupt("unreadable")

    monkeypatch.setattr(cancel_intents, "active_intents", corrupt)
    with queue._queue_lock:
        assert workers._drop_cancelled_pending() is False
    assert [row["id"] for row in workers.PENDING] == ["gap-0"]


# ---- L2: the cold-custody pass, every 3 s while any actor lives -----------------------------------

def _running(tmp_path, monkeypatch, name: str, rows: int):
    from ouroboros.task_results import write_task_result

    root = tmp_path / name
    root.mkdir()
    queue, _state, workers = _install_queue(root, monkeypatch)
    for index in range(rows):
        task_id = f"{name}-{index}"
        write_task_result(root, task_id, "running", chat_id=0, text="running work " * 2000)
        workers.RUNNING[task_id] = {"task": {"id": task_id, "type": "task", "chat_id": 0, "root_task_id": task_id},
                                    "worker_id": index, "attempt": 1, "started_at": 1000.0}
    return root, queue, workers


def test_l2_the_cold_custody_pass_reads_once_at_any_running_count(tmp_path, monkeypatch, meter):
    from ouroboros import budget_pause, sleep_custody

    def one_pass(name, rows):
        _root, queue, _workers = _running(tmp_path, monkeypatch, name, rows)
        monkeypatch.setattr(budget_pause, "budget_pause_row", lambda *_a: pytest.fail("one read per running actor"))
        outcome = counted(meter, f"L2:{name}", lambda: sleep_custody.reconcile(queue))
        monkeypatch.undo()
        assert outcome["result"] == set()
        return outcome

    small, large = one_pass("one", 1), one_pass("fifty", 50)
    assert small["counts"] == large["counts"], (small["counts"], large["counts"])
    assert small["counts"]["begins"] == 1 and small["counts"]["writes"] == 0, small


def test_l2_a_saved_request_among_many_running_actors_is_still_accepted_and_kept(tmp_path, monkeypatch, meter):
    """The companion: the real ``request, accept, retain`` protocol with 50 unrelated actors running beside it."""
    from ouroboros import sleep_custody
    from ouroboros.task_results import write_task_result
    from tests.test_cold_external_custody import _park, _setup

    env = _setup(tmp_path, monkeypatch)
    for index in range(50):
        task_id = f"beside-{index}"
        write_task_result(tmp_path, task_id, "running", chat_id=0)
        env.workers.RUNNING[task_id] = {"task": {"id": task_id, "type": "task", "chat_id": 0, "root_task_id": task_id},
                                        "worker_id": index + 1, "attempt": 1, "started_at": 1000.0}
    row = _park(env, tmp_path, monkeypatch)
    assert sleep_custody.accepted_receipt("sleeper", row), row
    calls = len(env.runtime.calls)
    assert sleep_custody.reconcile(env.q, runtime=env.runtime) == {"sleeper"}
    assert len(env.runtime.calls) == calls + 1, "the accepted receipt is rebound, the unrelated actors are not read"


def test_l2_an_unanswering_store_infers_nothing_and_keeps_every_accepted_carrier(tmp_path, monkeypatch, meter):
    from ouroboros import sleep_custody, store_hot_facts
    from tests.test_cold_external_custody import _park, _setup

    env = _setup(tmp_path, monkeypatch)
    _park(env, tmp_path, monkeypatch)

    def busy(*_a, **_k):
        raise store_hot_facts.json.JSONDecodeError("store cannot answer", "", 0)

    monkeypatch.setattr(store_hot_facts, "cold_custody_rows", busy)
    assert sleep_custody.reconcile(env.q, runtime=env.runtime) == {"sleeper"}, "unknown is not an orphan"


# ---- L14: the running rails on every supervisor tick ------------------------------------------------

def _live_tree(tmp_path, monkeypatch, name: str, children: int):
    """A root with ``children`` running descendants and as many queued ones: every shape the rails ask about."""
    root = tmp_path / name
    root.mkdir()
    queue, _state, workers = _install_queue(root, monkeypatch)
    monkeypatch.setattr(queue, "get_task_idle_timeout_sec", lambda: 900.0)
    monkeypatch.setattr(queue, "get_task_abs_ceiling_sec", lambda: None)
    workers.RUNNING["root"] = {"task": {"id": "root", "root_task_id": "root"}, "worker_id": 0, "attempt": 1,
                               "started_at": 1000.0, "last_progress_at": 1000.0}
    for index in range(children):
        child = f"child-{index}"
        workers.RUNNING[child] = {"task": {"id": child, "root_task_id": "root", "parent_task_id": "root"},
                                  "worker_id": index + 1, "attempt": 1, "started_at": 1000.0,
                                  "last_progress_at": 1000.0}
        workers.PENDING.append({"id": f"queued-{index}", "root_task_id": "root", "parent_task_id": child})
    return queue, workers


def test_l14_a_pass_walks_each_live_row_once_not_once_per_running_row(tmp_path, monkeypatch):
    """O(R+P), never O(R·(R+P)): the walk is the only per-row cost, so its count is the law. Every row is
    silent: a row with its own recent progress short-circuits the terms a per-row rescan would hide in."""
    from supervisor import queue_timeouts, silence_attention

    walks = []
    real = queue_timeouts._ancestors
    monkeypatch.setattr(queue_timeouts, "_ancestors", lambda task: walks.append(1) or real(task))
    monkeypatch.setattr(silence_attention, "observe", lambda *_a: None)

    def one_pass(name, children):
        queue, workers = _live_tree(tmp_path, monkeypatch, name, children)
        workers.PENDING[:] = [row for row in workers.PENDING if row["id"] != "queued-0"]  # a row with none queued below
        walks.clear()
        queue._enforce_task_timeouts_locked(workers, 5000.0, 0, {})
        return len(walks), len(workers.RUNNING) + len(workers.PENDING)

    for name, children in (("small", 1), ("large", 50)):
        counted_walks, live = one_pass(name, children)
        assert counted_walks == live, (name, counted_walks, live)


@pytest.mark.parametrize("progressing, expect_observed", [(True, False), (False, True)])
def test_l14_the_index_keeps_every_decision_a_scan_made(tmp_path, monkeypatch, progressing, expect_observed):
    """The companion: a fresh grandchild still spares its root, and a truly
    silent tree still reaches attention. Walked through the chain and through ``root_task_id`` alike."""
    from supervisor import queue_timeouts, silence_attention

    queue, workers = _live_tree(tmp_path, monkeypatch, "tree", 2)
    workers.PENDING.clear()
    now = 5000.0
    for meta in workers.RUNNING.values():
        meta["last_progress_at"] = now - 2000.0
    workers.RUNNING["child-1"]["task"]["root_task_id"] = ""  # only the parent chain ties it to the root
    workers.RUNNING["child-1"]["task"]["parent_task_id"] = "child-0"
    workers.RUNNING["child-0"]["task"]["root_task_id"] = ""
    if progressing:
        workers.RUNNING["child-1"]["last_progress_at"] = now - 10.0
    observed = []
    monkeypatch.setattr(silence_attention, "observe", lambda _q, tid, *_a: observed.append(tid))
    monkeypatch.setattr(silence_attention, "recovered", lambda *_a: None)
    assert queue._subtree_progressing("root", now, 900.0) is progressing
    assert queue_timeouts._tree_activity().progress.get("child-0", 0.0) == (now - 10.0 if progressing else now - 2000.0)
    queue._enforce_task_timeouts_locked(workers, now, 0, {})
    assert ("root" in observed) is expect_observed, observed


def test_l14_a_descendants_unread_observation_still_defers_its_ancestors_idle_judgment(tmp_path, monkeypatch):
    from supervisor import silence_attention

    queue, workers = _live_tree(tmp_path, monkeypatch, "tree", 1)
    workers.PENDING.clear()
    now = 5000.0
    for meta in workers.RUNNING.values():
        meta["last_progress_at"] = now - 2000.0
    observed = []
    monkeypatch.setattr(silence_attention, "observe", lambda _q, tid, *_a: observed.append(tid))
    queue._enforce_task_timeouts_locked(workers, now, 0, {}, pending_events={"child-0"})
    assert "root" not in observed and "child-0" not in observed
    queue._enforce_task_timeouts_locked(workers, now, 0, {}, pending_events=None)
    assert not observed, "an unreadable bus defers idle for everyone"
    queue._enforce_task_timeouts_locked(workers, now, 0, {}, pending_events=set())
    assert set(observed) == {"root", "child-0"}


class _Tick:
    """The tick's entry under a clock this test owns, counting the passes it let through."""

    def __init__(self, monkeypatch, queue):
        from supervisor import queue_timeouts

        self.now, self.passes, self.reads = 10_000.0, 0, 0
        self.queue_timeouts, self.queue = queue_timeouts, queue
        real = queue_timeouts.enforce_task_timeouts

        def counted_pass(**kwargs):
            self.passes += 1
            return real(**kwargs)

        monkeypatch.setattr(queue_timeouts, "enforce_task_timeouts", counted_pass)
        monkeypatch.setattr(queue_timeouts.time, "time", lambda: self.now)
        monkeypatch.setitem(queue_timeouts._clocks, "members", None)
        monkeypatch.setitem(queue_timeouts._clocks, "due", 0.0)

    def __call__(self, advance: float = 0.0) -> int:
        self.now += advance
        before = self.passes

        def pending():
            self.reads += 1
            return set()

        self.queue_timeouts.enforce_due_task_timeouts(pending)
        return self.passes - before


def _clocked(tmp_path, monkeypatch, name="clocks", **task):
    root = tmp_path / name
    root.mkdir()
    queue, _state, workers = _install_queue(root, monkeypatch)
    monkeypatch.setattr(queue, "get_task_idle_timeout_sec", lambda: 900.0)
    monkeypatch.setattr(queue, "get_task_abs_ceiling_sec", lambda: None)
    monkeypatch.setattr(queue, "FINALIZATION_GRACE_SEC", 60.0)
    monkeypatch.setattr(queue, "_request_finalization_grace", lambda *_a, **_k: "control")
    workers.RUNNING["task"] = {"task": {"id": "task", "root_task_id": "task", **task}, "worker_id": 0, "attempt": 1,
                               "started_at": 10_000.0, "last_progress_at": 10_000.0}
    return queue, workers


def test_l14_a_tick_with_no_clock_due_and_no_membership_change_does_nothing(tmp_path, monkeypatch):
    queue, _workers = _clocked(tmp_path, monkeypatch)
    tick = _Tick(monkeypatch, queue)
    assert tick() == 1, "the first tick of a process judges every row once"
    reads = tick.reads
    for _second in range(5):
        assert tick(1.0) == 0
    assert tick.reads == reads, "a tick that runs no pass asks the bus nothing"


def test_l14_a_deadline_is_met_on_its_own_tick_not_on_the_cadence(tmp_path, monkeypatch):
    import datetime

    queue, workers = _clocked(tmp_path, monkeypatch, deadline_at=datetime.datetime.fromtimestamp(
        10_005.0, datetime.timezone.utc).isoformat())
    tick = _Tick(monkeypatch, queue)
    assert tick() == 1
    assert tick(4.0) == 0
    assert "finalization_requested_at" not in workers.RUNNING["task"]
    assert tick(1.0) == 1, "the hard clock itself wakes the pass"
    assert workers.RUNNING["task"]["finalization_reason"] == "deadline"
    assert tick(1.0) == 1, "an open grace stays per tick: progress may withdraw it, its end reaps"


def test_l14_a_ceiling_is_met_on_its_own_tick(tmp_path, monkeypatch):
    queue, workers = _clocked(tmp_path, monkeypatch, execution_timeout_sec=5)
    tick = _Tick(monkeypatch, queue)
    assert tick() == 1
    assert tick(4.0) == 0
    assert tick(1.0) == 1
    assert workers.RUNNING["task"]["finalization_reason"] == "absolute_ceiling"


def test_l14_a_pending_deadline_wakes_the_pass_and_settles_without_the_cadence(tmp_path, monkeypatch):
    import datetime

    queue, workers = _clocked(tmp_path, monkeypatch)
    workers.PENDING.append({"id": "queued", "deadline_at": datetime.datetime.fromtimestamp(
        10_003.0, datetime.timezone.utc).isoformat()})
    from supervisor import queue_timeouts

    submitted = []
    monkeypatch.setattr(queue_timeouts, "_queue_pending_deadline", lambda task, **_k: submitted.append(task["id"]))
    monkeypatch.setattr(queue_timeouts, "pending_deadline_expirable", lambda _task: True)
    tick = _Tick(monkeypatch, queue)
    assert tick() == 1 and not submitted
    assert tick(2.0) == 0 and not submitted
    assert tick(1.0) == 1 and submitted == ["queued"]


def test_l14_membership_and_the_cadence_each_bring_the_pass_back(tmp_path, monkeypatch):
    from ouroboros.config import BODY_MAINTENANCE_INTERVAL_SEC

    queue, workers = _clocked(tmp_path, monkeypatch)
    tick = _Tick(monkeypatch, queue)
    assert tick() == 1 and tick(1.0) == 0
    workers.RUNNING["new"] = {"task": {"id": "new", "root_task_id": "new"}, "worker_id": 1, "attempt": 1,
                              "started_at": tick.now, "last_progress_at": tick.now}
    assert tick() == 1, "a row nobody judged yet"
    assert tick(1.0) == 0
    workers.PENDING.append({"id": "queued"})
    assert tick() == 1
    assert tick(BODY_MAINTENANCE_INTERVAL_SEC) == 1, "what no clock names (a setting, silence) is seen within the cadence"


class _Visited(dict):
    """RUNNING that counts the walks over it (a membership read that visits every key)."""

    visits = 0

    def __iter__(self):
        self.visits += 1
        return super().__iter__()


class _VisitedList(list):
    visits = 0

    def __iter__(self):
        self.visits += 1
        return super().__iter__()


def test_l14_a_quiet_tick_visits_no_row_at_any_live_set_size(tmp_path, monkeypatch):
    """Astra m2: the skip itself is O(1). Membership and clocks unchanged, the gate compares the queue's
    change mark and sizes and walks neither queue, at 1+1 rows and at 50+61; a mark brings the pass back."""
    queue, workers = _clocked(tmp_path, monkeypatch)
    rows = {"running": dict(workers.RUNNING), "pending": [{"id": "queued"}]}
    for size in ((1, 1), (50, 61)):
        running, pending = _Visited(rows["running"]), _VisitedList(rows["pending"])
        for index in range(len(running), size[0]):
            running[f"task-{index}"] = {**running["task"], "task": {"id": f"task-{index}", "root_task_id": f"task-{index}"}}
        pending.extend({"id": f"queued-{index}"} for index in range(len(pending), size[1]))
        for module in (workers, queue):
            monkeypatch.setattr(module, "RUNNING", running)
            monkeypatch.setattr(module, "PENDING", pending)
        tick = _Tick(monkeypatch, queue)
        assert tick() == 1, "the first tick judges every row once"
        running.visits = pending.visits = 0
        assert tick(1.0) == 0 and tick(1.0) == 0
        assert (running.visits, pending.visits) == (0, 0), size
        queue.persist_queue_snapshot(reason="hold_changed")  # any marked change of the queue
        assert tick() == 1 and tick(1.0) == 0


def test_l14_a_clock_set_back_cannot_park_the_pass(tmp_path, monkeypatch):
    queue, _workers = _clocked(tmp_path, monkeypatch)
    tick = _Tick(monkeypatch, queue)
    assert tick() == 1
    assert tick(-3600.0) == 1, "a due instant beyond the cadence is a clock that moved, never a reason to wait"


def test_l14_an_empty_queue_keeps_the_tick_free(tmp_path, monkeypatch):
    root = tmp_path / "empty"
    root.mkdir()
    queue, _state, _workers = _install_queue(root, monkeypatch)
    tick = _Tick(monkeypatch, queue)
    assert tick() == 0 and tick.reads == 0


# ---- L1: the owner-pause census ------------------------------------------------------------------

def _paused_trees(tmp_path, monkeypatch, name: str, roots: int, *, children: int = 1):
    """``roots`` trees the owner's real Pause settled: never-started rows under their accepted latch."""
    from ouroboros.owner_pause import FENCE_PAUSED, read_fence
    from ouroboros.task_results import write_task_result
    from supervisor.owner_pause_control import request_owner_pause

    root = tmp_path / name
    root.mkdir()
    queue, _state, workers = _install_queue(root, monkeypatch)
    ids = []
    for index in range(roots):
        root_id = f"{name}-{index}"
        ids.append(root_id)
        write_task_result(root, root_id, "scheduled", chat_id=0, root_task_id=root_id, text="paused work " * 2000)
        workers.PENDING.append({"id": root_id, "type": "task", "chat_id": 0, "root_task_id": root_id, "_attempt": 1})
        for child in range(children):
            child_id = f"{root_id}-c{child}"
            write_task_result(root, child_id, "scheduled", chat_id=0, root_task_id=root_id, text="child " * 2000)
            workers.PENDING.append({"id": child_id, "type": "task", "chat_id": 0, "root_task_id": root_id,
                                    "parent_task_id": root_id, "_attempt": 1})
        assert request_owner_pause(root_id, request_id=f"pause-{root_id}")["ok"]
        assert read_fence(root, root_id)["state"] == FENCE_PAUSED
    return root, queue, workers, ids


def test_l1_settled_trees_cost_nothing_per_pass_at_one_root_or_twenty(tmp_path, monkeypatch, meter):
    """The owner's Pause must not ENLARGE the census (live 01.11: 4.4-5.5k fence reads/min at 24 fences)."""
    from supervisor import owner_pause_control

    def one_pass(name, roots):
        _root, queue, _workers, ids = _paused_trees(tmp_path, monkeypatch, name, roots)
        assert all(owner_pause_control._AT_REST.get(root_id) for root_id in ids)
        outcome = counted(meter, f"L1:{name}", lambda: owner_pause_control.settle_requested_owner_pauses(queue))
        monkeypatch.undo()
        assert outcome["result"] == [], "a settled tree is not examined again"
        return outcome

    small, large = one_pass("one", 1), one_pass("twenty", 20)
    assert small["counts"] == large["counts"], (small["counts"], large["counts"])
    assert set(small["counts"].values()) == {0}, ("no store read, no decode, no write at all", small)


def test_l1_a_new_process_reads_each_fenced_tree_once_then_rests(tmp_path, monkeypatch, meter):
    """The dirty-set's safety: nothing in memory survives a restart, so the first look is the full census."""
    from supervisor import owner_pause_control

    _root, queue, _workers, ids = _paused_trees(tmp_path, monkeypatch, "boot", 3)
    owner_pause_control._AT_REST.clear()
    census = []
    real = owner_pause_control.refresh_owner_pause_tree
    monkeypatch.setattr(owner_pause_control, "refresh_owner_pause_tree",
                        lambda root_id, **kw: census.append(root_id) or real(root_id, **kw))
    assert sorted(owner_pause_control.settle_requested_owner_pauses(queue)) == sorted(ids)
    assert sorted(census) == sorted(ids)
    census.clear()
    assert owner_pause_control.settle_requested_owner_pauses(queue) == [] and census == []


@pytest.mark.parametrize("change", ["member_left", "attempt_changed", "new_generation"])
def test_l1_a_member_transition_brings_exactly_its_own_tree_back(tmp_path, monkeypatch, meter, change):
    """The companion: a transition wakes the tree it touched. Its neighbours keep costing nothing."""
    from supervisor import owner_pause_control

    _root, queue, workers, ids = _paused_trees(tmp_path, monkeypatch, "moves", 5)
    target = ids[2]
    census = []
    real = owner_pause_control.refresh_owner_pause_tree
    monkeypatch.setattr(owner_pause_control, "refresh_owner_pause_tree",
                        lambda root_id, **kw: census.append(root_id) or real(root_id, **kw))
    assert owner_pause_control.settle_requested_owner_pauses(queue) == [] and not census
    if change == "member_left":
        workers.PENDING[:] = [row for row in workers.PENDING if row["id"] != f"{target}-c0"]
    elif change == "member_arrived":
        workers.PENDING.append({"id": f"{target}-late", "root_task_id": target, "parent_task_id": target, "_attempt": 1})
    elif change == "attempt_changed":
        workers.RUNNING[f"{target}-c0"] = {"task": {"id": f"{target}-c0", "root_task_id": target}, "attempt": 2,
                                           "worker_id": 0, "started_at": 1000.0}
    elif change == "marked":
        owner_pause_control.mark_tree_changed(target)
    else:
        queue.BUDGET_ROOT_FENCES[target] = {**queue.BUDGET_ROOT_FENCES[target], "fence_id": "a-newer-fence"}
    owner_pause_control.settle_requested_owner_pauses(queue)
    assert census == [target]


def test_l1_the_look_is_owed_by_the_cadence_and_a_mark_and_never_without_a_pause(tmp_path, monkeypatch):
    from supervisor import owner_pause_control as control

    queue = type("Q", (), {"BUDGET_ROOT_FENCES": {}})()
    control.request_settlement_look()
    assert not control.settlement_look_owed(queue), "no owner Pause stands: nothing is owed"
    queue.BUDGET_ROOT_FENCES = {"budget-root": {"cause": "budget"}}
    control.request_settlement_look()
    assert not control.settlement_look_owed(queue), "a budget latch is not this census's"
    queue.BUDGET_ROOT_FENCES = {"root": {"cause": "owner_pause"}}
    assert not control.settlement_look_owed(queue), "nothing asked for a look"
    control.mark_tree_changed("root")
    assert control.settlement_look_owed(queue)


def test_l1_the_pass_consumes_the_look_and_a_stopped_pass_keeps_what_it_never_reached(tmp_path, monkeypatch, meter):
    from supervisor import owner_pause_control as control

    _root, queue, _workers, ids = _paused_trees(tmp_path, monkeypatch, "stop", 2)
    control.request_settlement_look()
    control.mark_tree_changed(ids[0])
    control.mark_tree_changed(ids[1])
    assert control.settle_requested_owner_pauses(queue, stop_requested=lambda: True) == []
    assert control.settlement_look_owed(queue), "the marked trees were not looked at: still owed"
    assert sorted(control.settle_requested_owner_pauses(queue)) == sorted(ids)
    assert not control.settlement_look_owed(queue)


def test_l1_a_pass_longer_than_the_cadence_is_never_followed_at_once_by_another(monkeypatch):
    """The closed loop this replaced: a pass longer than its spacing ran back to back. The cadence that
    finds a pass running owes nothing and journals the stall; only a transition may start the next at once."""
    from ouroboros import maintenance
    from ouroboros import server_maintenance as sm
    from supervisor import owner_pause_control as control
    from supervisor import queue

    started = []

    def start(latch, name, _target, _args):
        started.append(name)
        latch.release()
        return True

    stalled = []
    monkeypatch.setattr(sm, "_start_maintenance_thread", start)
    monkeypatch.setattr(sm, "_duty_busy", lambda latch, _sec: stalled.append(latch))
    monkeypatch.setattr(sm, "_tick_maintenance_child", lambda: None)
    monkeypatch.setattr(maintenance, "child", maintenance.MaintenanceChild())
    monkeypatch.setattr(queue, "RUNNING", {})
    monkeypatch.setattr(queue, "BUDGET_ROOT_FENCES", {"root": {"cause": "owner_pause", "fence_id": "f"}})
    with control._BOOK_LOCK:
        control._LOOK_OWED[0] = False
        control._CHANGED.clear()

    def tick():
        monkeypatch.setattr(sm, "_LAST_CANCEL_INTENT_SWEEP", [0.0])  # the 20 s block is due
        started.clear()
        sm._periodic_supervisor_maintenance([time.time()], [time.time()])
        return "owner_pause_settlement" in started

    assert sm._OWNER_PAUSE_SWEEP_LOCK.acquire(blocking=False)
    try:
        assert not tick(), "its latch is held: the pass already running is this look"
    finally:
        sm._OWNER_PAUSE_SWEEP_LOCK.release()
    assert sm._OWNER_PAUSE_SWEEP_LOCK in stalled
    assert not control.settlement_look_owed(queue), "the pass ended: nothing is owed until the next cadence"
    assert tick(), "the next cadence looks again"


def test_l1_an_unconfirmed_stop_keeps_its_tree_due_without_rereading_a_body(tmp_path, monkeypatch, meter):
    """The real re-check stays: an unsettled stop is read back each look, and that is all it costs."""
    from ouroboros import budget_pause, task_results
    from supervisor import owner_pause_control
    from tests.test_pause_observation_cas import _owner_paused_solo

    queue, _workers = _owner_paused_solo(tmp_path, monkeypatch)
    observed = []

    def custody(root, task_id, **kw):
        observed.append(kw.get("reason"))
        return {"custody_read": "ok", "runs": [{"run_id": "run-1", "state": "stop_requested", "stop_outcome": "requested"}]}

    monkeypatch.setattr(budget_pause, "observe_task_runs", custody)
    assert owner_pause_control.settle_requested_owner_pauses(queue) == []
    assert owner_pause_control._AT_REST.get("solo") is None
    monkeypatch.setattr(task_results, "load_task_result", lambda *a, **k: pytest.fail("a look at an unsettled stop read a body"))
    observed.clear()
    assert owner_pause_control.settle_requested_owner_pauses(queue) == []
    assert observed == ["owner_pause_settlement_check"], "read back once per look, never a second control"
    monkeypatch.undo()


def test_l1_a_tree_held_by_an_open_writer_reaches_its_census_each_look_and_reads_no_member(tmp_path, monkeypatch, meter):
    """The look that DOES reach the census: nothing moved, so the memo answers it, and the writer's end is seen."""
    from ouroboros import budget_pause, owner_pause, task_results
    from ouroboros import delegate_custody as custody
    from ouroboros.task_results import write_task_result
    from supervisor import owner_pause_control

    root = tmp_path / "held"
    root.mkdir()
    queue, _state, workers = _install_queue(root, monkeypatch)
    write_task_result(root, "held-root", "scheduled", chat_id=0, root_task_id="held-root", text="work " * 2000)
    write_task_result(root, "held-child", "scheduled", chat_id=0, root_task_id="held-root", text="work " * 2000)
    workers.PENDING.extend([{"id": "held-root", "type": "task", "chat_id": 0, "root_task_id": "held-root", "_attempt": 1},
                            {"id": "held-child", "type": "task", "chat_id": 0, "root_task_id": "held-root", "_attempt": 1}])
    assert custody.emit(root, custody.STARTED, {"run_id": "run-open", "invocation_id": "invoke-open", "task_id": "held-child",
                                                "root_task_id": "held-root", "route": "route", "source": "delegate_start"})
    assert owner_pause_control.request_owner_pause("held-root", request_id="p")["state"] == owner_pause.FENCE_REQUESTED
    census = []
    real_census = owner_pause_control.refresh_owner_pause_tree
    monkeypatch.setattr(owner_pause_control, "refresh_owner_pause_tree",
                        lambda root_id, **kw: census.append(root_id) or real_census(root_id, **kw))
    with monkeypatch.context() as patch:
        patch.setattr(task_results, "load_task_result", lambda *a, **k: pytest.fail("a settled member's body was re-read"))
        patch.setattr(budget_pause, "observe_task_runs", lambda *a, **k: pytest.fail("unchanged custody was re-read"))
        assert owner_pause_control.settle_requested_owner_pauses(queue) == []
    assert census == ["held-root"], "the tree stays due: its census ran"
    assert owner_pause.read_fence(root, "held-root")["state"] == owner_pause.FENCE_REQUESTED
    assert custody.emit(root, custody.SETTLED, {"run_id": "run-open", "invocation_id": "invoke-open", "task_id": "held-child",
                                                "root_task_id": "held-root", "terminal_state": "succeeded"})
    assert owner_pause_control.settle_requested_owner_pauses(queue) == ["held-root"], "the custody write is the census's input"
    assert owner_pause.read_fence(root, "held-root")["state"] == owner_pause.FENCE_PAUSED


def test_l1_the_stop_confirmed_later_settles_the_tree_on_the_next_look(tmp_path, monkeypatch, meter):
    from ouroboros import budget_pause, owner_pause
    from supervisor import owner_pause_control
    from tests.test_pause_observation_cas import _owner_paused_solo, _quiet_custody

    queue, _workers = _owner_paused_solo(tmp_path, monkeypatch)
    monkeypatch.setattr(budget_pause, "observe_task_runs", _quiet_custody)
    assert owner_pause_control.settle_requested_owner_pauses(queue) == ["solo"]
    assert owner_pause.read_fence(tmp_path, "solo")["state"] == owner_pause.FENCE_PAUSED
    assert owner_pause_control._AT_REST.get("solo"), "and from here it costs nothing"
    assert owner_pause_control.settle_requested_owner_pauses(queue) == []


def test_l1_a_paused_tree_whose_late_work_settles_with_no_queue_row_is_still_released(tmp_path, monkeypatch):
    """Why a state string is no proof of rest (k3): no member is left to announce this, so it stays looked at."""
    from ouroboros import owner_pause
    from ouroboros.task_results import write_task_result
    from supervisor import owner_pause_control

    root = tmp_path / "late"
    root.mkdir()
    queue, _state, _workers = _install_queue(root, monkeypatch)
    write_task_result(root, "answered", "running", chat_id=0, root_task_id="answered")
    fence, _created = owner_pause.install_fence(root, "answered", request_id="p", late_work=True)
    owner_pause.set_fence_state(root, "answered", fence_id=fence["fence_id"], state=owner_pause.FENCE_PAUSED)
    queue.BUDGET_ROOT_FENCES["answered"] = {"cause": "owner_pause", "fence_id": fence["fence_id"],
                                            "root_task_id": "answered", "status": "paused"}
    assert owner_pause_control.settle_requested_owner_pauses(queue) == ["answered"], "not settled: its row still runs"
    assert owner_pause_control._AT_REST.get("answered") is None, "a tree with no queue row is never at rest"
    write_task_result(root, "answered", "completed", result="answered")  # an orphan repair: no hook names it
    assert owner_pause_control.settle_requested_owner_pauses(queue) == []
    assert owner_pause.read_fence(root, "answered")["state"] == owner_pause.FENCE_RELEASED
    assert "answered" not in queue.BUDGET_ROOT_FENCES


def test_l1_a_finishing_review_keeps_its_tree_due_until_it_ends(tmp_path, monkeypatch, meter):
    from ouroboros import owner_pause, review_operation
    from supervisor import owner_pause_control

    root, queue, _workers, ids = _paused_trees(tmp_path, monkeypatch, "review", 1)
    target = ids[0]
    live = {"on": True}
    monkeypatch.setattr(review_operation, "task_has_live_review_operation",
                        lambda _r, task_id, **_k: live["on"] and task_id == target)
    owner_pause_control.mark_tree_changed(target)
    assert owner_pause_control.settle_requested_owner_pauses(queue) == [target]
    fence = owner_pause.read_fence(root, target)
    assert fence["finishing_reviews"] == [target] and owner_pause_control._AT_REST.get(target) is None
    live["on"] = False
    assert owner_pause_control.settle_requested_owner_pauses(queue) == [target], "still due, with no transition"
    assert owner_pause.read_fence(root, target)["finishing_reviews"] == []
    assert owner_pause_control._AT_REST.get(target), "and now at rest"


# ---- L1: one census memo for the settle pass, Pause, Resume and the display -----------------------

def _census_tree(tmp_path, monkeypatch, name="census"):
    root, queue, workers, ids = _paused_trees(tmp_path, monkeypatch, name, 1, children=2)
    return root, queue, workers, ids[0]


def _write_seq_reads(monkeypatch):
    """Fail the moment a member's body or a member's custody is read: what a memo hit must not do."""
    from ouroboros import budget_pause, task_results

    monkeypatch.setattr(task_results, "load_task_result", lambda *a, **k: pytest.fail("a census hit read a body"))
    monkeypatch.setattr(budget_pause, "observe_task_runs", lambda *a, **k: pytest.fail("a census hit read custody"))


def test_l1_memo_an_unchanged_tree_answers_every_census_consumer_without_a_member_read(tmp_path, monkeypatch, meter):
    from supervisor.continuation_admission import action_writers, conflicting_writers, model_handoffs

    root, queue, _workers, target = _census_tree(tmp_path, monkeypatch)
    first = conflicting_writers(queue, target, drive_root=root)
    with monkeypatch.context() as patch:
        _write_seq_reads(patch)
        assert conflicting_writers(queue, target, drive_root=root) == first
        assert model_handoffs(root, target) == []
    # An addressed action first repairs its own tree (its one body read, per press); its census is the same memo.
    assert action_writers(queue, target, drive_root=root) == first


@pytest.mark.parametrize("change", ["task_row", "custody_run", "open_request"])
def test_l1_memo_any_write_of_the_tree_is_a_new_census(tmp_path, monkeypatch, meter, change):
    from ouroboros import delegate_custody as custody
    from ouroboros.task_results import write_task_result
    from supervisor.continuation_admission import conflicting_writers

    root, queue, _workers, target = _census_tree(tmp_path, monkeypatch)
    child = f"{target}-c0"
    conflicting_writers(queue, target, drive_root=root)
    if change == "task_row":
        write_task_result(root, child, "scheduled", launch_handoffs={"op": {"tool": "opaque", "state": "claimed"}})
        kind = "tool_handoff"
    elif change == "custody_run":
        assert custody.emit(root, custody.STARTED, {"run_id": "run-new", "invocation_id": "invoke-new",
                                                    "task_id": child, "root_task_id": target,
                                                    "route": "route", "source": "delegate_start"})
        kind = "delegated_run"
    else:
        assert custody.record_start_requested(root, run_id="", task_id=child, invocation_id="invoke-open",
                                               idempotency_key="invoke-open", request={"prompt": "work"},
                                               route="codex", source="delegate_start")
        kind = "delegated_run"
    blockers = conflicting_writers(queue, target, drive_root=root)
    assert any(row["kind"] == kind and row.get("task_id") == child for row in blockers), blockers


def test_l1_memo_a_fact_no_revision_names_is_derived_fresh_on_a_hit(tmp_path, monkeypatch, meter):
    """Inputs are kept, never verdicts: a claim's owner that died, a queue member, a process."""
    from ouroboros import tool_custody
    from ouroboros.task_results import write_task_result
    from supervisor.continuation_admission import conflicting_writers

    root, queue, workers, target = _census_tree(tmp_path, monkeypatch)
    child = f"{target}-c0"
    write_task_result(root, child, "scheduled", launch_handoffs={"op": {"tool": "opaque", "state": "claimed"}})
    assert any(row["kind"] == "tool_handoff" for row in conflicting_writers(queue, target, drive_root=root))
    holds = [False]
    monkeypatch.setattr(tool_custody, "retained_tool_custody",
                        lambda *_a, **_k: [{"kind": "tool_handoff", "task_id": child}] if holds[0] else [])
    with monkeypatch.context() as patch:
        _write_seq_reads(patch)
        assert not any(row["kind"] == "tool_handoff" for row in conflicting_writers(queue, target, drive_root=root))
        holds[0] = True
        assert any(row["kind"] == "tool_handoff" for row in conflicting_writers(queue, target, drive_root=root))
    workers.RUNNING["new-member"] = {"task": {"id": "new-member", "root_task_id": target}, "attempt": 1,
                                     "worker_id": 0, "started_at": 1000.0}
    assert any(row == {"kind": "running_member", "task_id": "new-member"}
               for row in conflicting_writers(queue, target, drive_root=root))


def test_l1_memo_unknown_is_never_kept(tmp_path, monkeypatch, meter):
    from ouroboros import owner_pause
    from supervisor import tree_census
    from supervisor.continuation_admission import conflicting_writers

    root, queue, _workers, target = _census_tree(tmp_path, monkeypatch)
    tree_census.forget(root)
    with monkeypatch.context() as patch:
        def unreadable(*_a, **_k):
            raise ValueError("tree_membership_unreadable")

        patch.setattr(owner_pause, "tree_member_results", unreadable)
        assert any(row["kind"] == "tree_census_unreadable"
                   for row in conflicting_writers(queue, target, drive_root=root))
    assert not any(row["kind"] == "tree_census_unreadable"
                   for row in conflicting_writers(queue, target, drive_root=root)), "the next reader reads again"


def test_l1_memo_holds_no_task_body(tmp_path, monkeypatch, meter):
    from supervisor import tree_census

    root, _queue, _workers, target = _census_tree(tmp_path, monkeypatch)
    facts = tree_census.writer_facts(root, target)
    assert set(facts.rows) == {target, f"{target}-c0", f"{target}-c1"}
    for row in facts.rows.values():
        assert set(row) <= set(tree_census.MEMBER_FIELDS), row


def test_l1_seam_many_trees_one_attempt_read_and_no_member_read_without_an_attempt(tmp_path, monkeypatch, meter):
    """What ``/api/state`` calls over its paused roots (noted for LE): flat in the number of trees."""
    from supervisor.continuation_admission import trees_model_handoffs

    def many(name, roots):
        root, _queue, _workers, ids = _paused_trees(tmp_path, monkeypatch, name, roots, children=0)
        with monkeypatch.context() as patch:
            _write_seq_reads(patch)
            outcome = counted(meter, f"seam:{name}", lambda: trees_model_handoffs(root, ids))
        monkeypatch.undo()
        assert outcome["result"] == {root_id: [] for root_id in ids}
        return outcome

    small, large = many("one", 1), many("twenty", 20)
    assert small["counts"] == large["counts"], (small["counts"], large["counts"])
    assert small["counts"]["writes"] == 0 and trees_model_handoffs(tmp_path, []) == {}


def test_l1_seam_a_tree_with_an_open_attempt_still_answers_it(tmp_path, monkeypatch, meter):
    from ouroboros import owner_pause
    from supervisor.continuation_admission import model_handoffs, trees_model_handoffs

    root, _queue, _workers, ids = _paused_trees(tmp_path, monkeypatch, "open", 2, children=0)
    open_attempt = {"attempt_id": "a1", "task_id": ids[0], "root_task_id": ids[0], "state": "dispatched",
                    "review_slot_id": "slot"}
    monkeypatch.setattr(owner_pause, "trees_model_requests",
                        lambda _r, roots: {root_id: [open_attempt] if root_id == ids[0] else [] for root_id in roots})
    answered = trees_model_handoffs(root, ids)
    assert [row["kind"] for row in answered[ids[0]]] == ["model_handoff"] and answered[ids[0]][0]["review_owned"]
    assert answered[ids[1]] == []
    assert answered[ids[0]] == model_handoffs(root, ids[0], attempts=[open_attempt])
