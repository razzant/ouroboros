"""The server's recurring passes cost the same whatever the live set holds (cost law L1, L2, L3, L14).

``tests/test_cost_contract.py`` B7 holds every duty flat against HISTORY. These hold the same passes flat
against the LIVE SET, which is what grew on 01.11.2026: a second pass over an unchanged queue reads no task
body and no usage row, writes nothing, and costs the same at 1 and at many running, pending or paused rows. A
companion beside each law shows the pass still does its work when a real input changes, so a law cannot be
satisfied by a pass that stopped looking. Counting is not timing: no clock decides anything here.
"""
from __future__ import annotations

import pathlib

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
    """The companion: a fresh grandchild still spares its root, a parked diagnostic never does, and a truly
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
    workers.PENDING.append({"id": "diag", "root_task_id": "root", "metadata": {"work_diagnostic": {
        "kind": "silence", "task_id": "root", "root_task_id": "root", "episode_id": "episode", "task_attempt": 1}}})
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
