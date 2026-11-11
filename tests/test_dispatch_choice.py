"""The dispatcher pays only for free capacity and for what changed (pool-capacity D9, WP7), counted on the real pass.

Counts, never a clock, on the real ``assign_tasks`` with real worker slots (``tests/test_owner_wait_pool.pool``: the
pool's own ``Worker`` rows with live process doubles) over a real store: store transactions are counted at the store's
one connection door (``store.connection``) with whether the queue lock was owned and whether the chooser opened it;
memory-gate calls are spied at the shared predicate, the chooser's walks at ``_collect``. Every guard is shown both ways
in its test.
"""

from __future__ import annotations

import contextlib
import copy
import pathlib
import re
import traceback
from unittest.mock import Mock

import pytest

from supervisor import dispatch_choice, queue, workers
from tests.test_owner_wait_pool import pool as pool  # noqa: F401 -- fixture
from tests.test_workspace_preparation import env as env  # noqa: F401 -- fixture

REPO = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture
def assigning(pool, monkeypatch):  # noqa: F811 -- the imported fixture is this one's input
    from supervisor import evolution_lifecycle, state

    monkeypatch.setattr(state, "budget_remaining", lambda *a, **kw: 100)
    monkeypatch.setattr(evolution_lifecycle, "evolution_block_reason", lambda: "")
    return pool


@pytest.fixture
def memory_gate(monkeypatch):
    from supervisor import worker_assignment

    gate = Mock(wraps=worker_assignment._candidate_ineligible)
    monkeypatch.setattr(worker_assignment, "_candidate_ineligible", gate)
    return gate


def task(tid, **fields):
    return {"id": tid, "type": "task", "chat_id": 0, "text": "work", **fields}


def store_transactions(monkeypatch):
    """Every store transaction from here on: whether Q was owned, and whether the chooser opened it."""
    from ouroboros import store

    seen, real = [], store.connection

    @contextlib.contextmanager
    def connection(*args, **kwargs):
        files = [frame.filename for frame in traceback.extract_stack()]
        seen.append({"under_q": queue._queue_lock._is_owned(),
                     "chooser": any(pathlib.PurePath(name).parts[-2:] == ("supervisor", "dispatch_choice.py") for name in files)})
        with real(*args, **kwargs) as conn:
            yield conn

    monkeypatch.setattr(store, "connection", connection)
    return seen


def held_prefix(count):
    """Rows the memory gates hold, of three kinds: an owner's hold, a saved pause, and children of one root
    carrying a selection while the root's child limit is reached (their C2 inputs never need a read)."""
    rows = []
    for index in range(count):
        kind = index % 3
        if kind == 0:
            rows.append(task(f"held-{index}", _owner_hold={"reason": "owner"}))
        elif kind == 1:
            rows.append(task(f"paused-{index}", _budget_pause={"reason": "owner", "exact_continuation": True,
                                                               "status": "paused_before_dispatch"}))
        else:
            rows.append(task(f"child-{index}", delegation_role="subagent", root_task_id="owner",
                             parent_task_id="owner", depth=1,
                             _resume_selection={"selected_by": "owner", "grant_id": "grant-1"}))
    return rows


def test_a_full_pool_chooses_nothing_and_reads_nothing_until_a_slot_frees(assigning, monkeypatch, memory_gate):
    """Full pool: no pass, no row looked at, no store transaction by the chooser and none under Q. The tick's
    cancel hygiene keeps its one statuses() SELECT (the disclosed residual), read before Q. Then one slot frees:
    the next tick passes once and launches the first row every gate admits."""
    from ouroboros import config

    monkeypatch.setattr(config, "get_max_active_subagents_per_root", lambda: 0)  # every carrier child is held
    workers.PENDING.extend([*held_prefix(30), task("runnable")])
    collect = Mock(wraps=dispatch_choice._collect)
    monkeypatch.setattr(dispatch_choice, "_collect", collect)
    seen = store_transactions(monkeypatch)
    for _tick in range(3):
        workers.assign_tasks()
    assert collect.call_count == memory_gate.call_count == 0  # no walk, no row looked at
    assert [row for row in seen if row["chooser"] or row["under_q"]] == []
    assert len(seen) == 3, seen  # one hygiene statuses() SELECT a tick, outside Q
    assert "runnable" not in workers.RUNNING
    workers.RUNNING.clear()  # the other direction: the parent's slot frees
    assigning.original.busy_task_id = None
    workers.assign_tasks()
    assert collect.call_count >= 1
    assert memory_gate.call_count == 32  # 31 rows walked, then the chosen row revalidated
    assert workers.RUNNING["runnable"]["worker_id"] == 0


@pytest.mark.parametrize("prefix", [100, 1000, 10000])
def test_one_free_slot_walks_a_held_prefix_in_memory_and_reads_only_the_row_it_takes(
        assigning, monkeypatch, prefix, memory_gate):
    """k = 1 behind a long held prefix: each pass costs one memory look per row. The held
    children carrying a selection have C2 decided once, at first sight, with one lineage read for their one root and
    one read-ahead; while that root is unchanged no pass reads for them again. Choosing the eligible row behind them
    costs one read-ahead, the same at every prefix length, and nothing opens the store under Q."""
    from ouroboros import config

    monkeypatch.setattr(config, "get_max_active_subagents_per_root", lambda: 0)  # every carrier child is held
    workers.RUNNING.clear()
    assigning.original.busy_task_id = None  # one idle worker, MAX_WORKERS 1: no growth, k = 1
    workers.PENDING.extend(held_prefix(prefix))
    seen = store_transactions(monkeypatch)
    workers.assign_tasks()  # first sight: C2 of the carriers, read once for their one root
    cold = [row for row in seen if row["chooser"]]
    assert len(cold) == 2 and not any(row["under_q"] for row in seen), seen  # the root's lineage and one read-ahead
    assert len(dispatch_choice._ADMITTED) == prefix // 3, "each carrier's admission is kept by its identity"
    del seen[:]
    memory_gate.reset_mock()
    workers.assign_tasks()  # unchanged: the prefix walked once in memory, nothing read for it
    assert memory_gate.call_count == prefix
    assert [row for row in seen if row["chooser"] or row["under_q"]] == []
    memory_gate.reset_mock()
    workers.PENDING.append(task("eligible"))
    chosen = dispatch_choice.choose(queue, 100)
    assert [row["id"] for row in chosen] == ["eligible"]
    assert memory_gate.call_count == prefix + 2  # one look per row and the chosen row's revalidation
    assert [row for row in seen if row["chooser"]] == [{"under_q": False, "chooser": True}]
    assert not any(row["under_q"] for row in seen)
    print(f"DISPATCH_PREFIX {{'prefix': {prefix}, 'rows_examined_per_pass': {prefix + 1}, "
          f"'first_sight_store_transactions': {len(cold)}, 'unchanged_pass_store_transactions': 0, "
          f"'choice_store_transactions': 1}}")
    workers.assign_tasks()  # the real pass takes it on the idle worker
    assert workers.RUNNING["eligible"]["worker_id"] == 0


@pytest.mark.parametrize("control", ["none", "pause_before", "pause_between", "latch_between", "resume_between",
                                     "resume_elsewhere"])
def test_a_pause_or_fence_between_the_chooser_read_and_the_handoff_refuses_the_launch(env, monkeypatch, control):
    """The chooser's read-ahead is a pre-selection, never the authority: an owner Pause installed after it read the
    fence, a root latch after it chose, or the root's money moved by a Resume or Pause after C2 admitted a selection
    made under it, is refused at the launch, which reads again under its own lock and before its put; a Pause the
    chooser already saw is never claimed at all; with nothing between, the same pass launches. A Pause another
    process wrote moves no count this process keeps, so the put's refusal forgets the kept admission and the
    next pass decides C2 from a full read and repairs the selection instead of choosing it again."""
    from ouroboros import owner_pause
    from ouroboros.budget_pause import set_budget_pause
    from supervisor.events_budget import _root_resume_state, _set_root_budget_pause_locked
    from tests.test_launch_outside_q import occupy_lane, scheduled_child
    from tests.test_scheduled_receipt_contention import install_worker
    from tests.test_workspace_preparation import _WAIT_SEC

    worker, delivered = install_worker(env, monkeypatch)
    row = scheduled_child(env)
    if control.startswith("resume_"):  # the owner's choice of this child, made under the root's current money basis
        with queue._queue_lock:
            basis = _root_resume_state(queue, "root", env.data, actor_id="root")["basis"]
            row["_resume_selection"] = {"selected_by": "owner", "root_resume_basis": basis}
    if control == "pause_before":
        owner_pause.install_fence(env.data, "root", request_id="seen-by-the-chooser")
    release = occupy_lane(env.gen)
    workers.assign_tasks()  # chooses (or refuses) from its read-ahead; the launch waits behind the lane
    if control == "pause_before":
        assert worker.busy_task_id is None and "_launch_claim" not in row, "the chooser saw the Pause"
    else:
        assert worker.busy_task_id == "launch-child" and row.get("_launch_claim")
    if control == "pause_between":
        owner_pause.install_fence(env.data, "root", request_id="after-the-read-ahead")
    elif control == "latch_between":
        with queue._queue_lock:
            _set_root_budget_pause_locked("root", {"cause": "owner_pause", "fence_id": "after-the-choice"})
    elif control.startswith("resume_"):  # the root's money pause moves the basis the choice was admitted under
        assert dispatch_choice._ADMITTED.get("launch-child"), "C2 admitted the selection at the choice"
        with monkeypatch.context() as written:
            if control == "resume_elsewhere":  # a worker process's write: this process's control count stands
                from ouroboros import store_hot_facts

                written.setattr(store_hot_facts, "note_written", lambda columns: None)
            set_budget_pause(env.data, "root", {"pause_id": "later", "state": "paused", "reason": "budget",
                                                "pause_generation": 1})
    release.set()
    assert env.gen.wait_idle(_WAIT_SEC)
    if control == "none":
        assert [task["id"] for task in delivered] == ["launch-child"] and "launch-child" in queue.RUNNING
    else:
        assert delivered == [] and "launch-child" not in queue.RUNNING and worker.busy_task_id is None
        assert "_launch_claim" not in row and env.pending_ids() == ["launch-child"]
    if control == "resume_elsewhere":
        assert "launch-child" not in dispatch_choice._ADMITTED, "the put's refusal forgets the kept admission"
        workers.assign_tasks()
        assert env.gen.wait_idle(_WAIT_SEC)
        assert "_resume_selection" not in row and delivered == [], "the next pass repaired the stale selection"


# ---- C2: the in-memory identity decides only whether to read again ---------------------------------------------

def test_every_writer_of_what_the_c2_identity_reads_moves_it(assigning, monkeypatch):
    """R3 item 17: C2's admission is kept while the identity it rested on stands, and every writer of what that
    identity stands for moves it, so the next pass reads again: the root's latch, the owner's hold and a budget hold
    on the root's row, the root's saved pause, its money base and resume grants (``set_budget_pause``), the owner's
    fence and its warm grant, a restored row and a new process. An ordinary write moves nothing (the other way)."""
    from ouroboros import owner_pause, store_hot_facts
    from ouroboros.budget_pause import set_budget_pause
    from ouroboros.task_results import write_task_result
    from supervisor import events_budget
    from supervisor.events_budget import HOLD_ROOT_FENCE_LIFTED, _set_root_budget_pause_locked, hold_budget_row

    root_row = assigning.meta["task"]  # the running root ("owner"), its own physical actor
    child = task("child", delegation_role="subagent", root_task_id="owner", parent_task_id="owner", depth=1,
                 _resume_selection={"selected_by": "owner", "grant_id": "grant-1"})
    workers.PENDING.append(child)
    allowed = Mock(return_value=True)
    monkeypatch.setattr(events_budget, "budget_resume_dispatch_allowed", allowed)

    def identity():
        with queue._queue_lock:
            return dispatch_choice._identity(queue, child, dispatch_choice._index(queue), "owner")

    def chosen():
        return [row["id"] for row in dispatch_choice.choose(queue, 100)]

    assert chosen() == ["child"] and allowed.call_count == 1  # a new process: nothing kept, one full read
    assert chosen() == ["child"] and allowed.call_count == 1, "unchanged: the kept admission stands, nothing is read"
    kept = identity()
    write_task_result(assigning.root, "unrelated", "running", result="progress")  # an ordinary write
    assert identity() == kept and chosen() == ["child"] and allowed.call_count == 1

    writers = {
        "latch": lambda: _set_root_budget_pause_locked("owner", {"fence_id": "money"}),
        "owner_hold": lambda: root_row.__setitem__("_owner_hold", {"reason": "owner"}),
        "budget_hold": lambda: hold_budget_row(root_row, reason=HOLD_ROOT_FENCE_LIFTED, detail="test"),
        "saved_pause": lambda: root_row.__setitem__("_budget_pause", {"reason": "budget"}),
        "money_base_and_grant": lambda: set_budget_pause(assigning.root, "owner", {
            "pause_id": "pause-1", "state": "paused", "reason": "budget", "pause_generation": 1}),
        "owner_fence": lambda: owner_pause.install_fence(assigning.root, "owner", request_id="pause-req"),
        "own_selection": lambda: child.__setitem__("_resume_selection", {"selected_by": "owner", "grant_id": "g2"}),
    }
    for name, write in writers.items():
        before, control_writes = identity(), store_hot_facts.CONTROL_WRITES[0]
        with queue._queue_lock:
            write()
        assert identity() != before, name
        if name in ("money_base_and_grant", "owner_fence"):
            assert store_hot_facts.CONTROL_WRITES[0] > control_writes, name  # a durable writer: counted at the store

    restored = copy.deepcopy(child)  # a snapshot restore replaces the row object
    with queue._queue_lock:
        workers.PENDING[:] = [restored if row is child else row for row in workers.PENDING]
        index = dispatch_choice._index(queue)
        assert dispatch_choice._ADMITTED.get("child") and not dispatch_choice._kept(queue, restored, index, {})
    monkeypatch.setattr(dispatch_choice, "_ADMITTED", {})  # boot: a new process keeps nothing
    with queue._queue_lock:
        assert not dispatch_choice._kept(queue, restored, dispatch_choice._index(queue), {})


def test_the_c2_identity_covers_every_queue_fact_c2_reads():
    """Every task key ``budget_resume_dispatch_allowed`` and the root state it reads take from a queue row is in the
    identity (by value), and every store write of a task document passes the one hook that counts control writes
    (``Txn.store_task`` -> ``obligations.task_written`` -> ``store_hot_facts.note_written``): a new key or a new
    task writer fails here until the identity says how it sees it."""
    import inspect

    from ouroboros import work_stop
    from supervisor import events_budget

    sources = "".join(inspect.getsource(function) for function in (
        events_budget.budget_resume_dispatch_allowed, events_budget._root_resume_state,
        events_budget._root_resume_grants, work_stop.continuation_refusal, work_stop.hold_fact))
    keys = set(re.findall(r"""['"](_[a-z][a-z_]+)['"]""", sources)) | set(work_stop._POINTS)
    assert keys <= set(dispatch_choice._CONTROL_KEYS), sorted(keys - set(dispatch_choice._CONTROL_KEYS))
    assert "BUDGET_ROOT_FENCES" in inspect.getsource(dispatch_choice._identity)  # the latch, by value
    store_source = (REPO / "ouroboros" / "store.py").read_text(encoding="utf-8")
    writers = re.findall(r"(INSERT OR REPLACE INTO tasks|INSERT INTO tasks|UPDATE tasks SET [a-z_]+)", store_source)
    assert writers == ["INSERT OR REPLACE INTO tasks", "UPDATE tasks SET write_seq"], writers
    assert "task_written(self, task_id, columns)" in store_source
    obligations = (REPO / "ouroboros" / "obligations.py").read_text(encoding="utf-8")
    assert "note_written(columns)" in obligations


def test_note_written_counts_control_facts_and_only_them():
    from ouroboros import store_hot_facts

    before = store_hot_facts.CONTROL_WRITES[0]
    store_hot_facts.note_written({"owner_pause_facts": "{}", "budget_pause_facts": "{}", "superseded_by": None})
    assert store_hot_facts.CONTROL_WRITES[0] == before
    store_hot_facts.note_written({"budget_pause_facts": '{"state":"paused"}'})
    store_hot_facts.note_written({"retry_task_id": "retry-1"})
    assert store_hot_facts.CONTROL_WRITES[0] == before + 2


def test_a_strict_read_ahead_answers_what_it_read_and_never_opens_the_store(tmp_path):
    from ouroboros import store_hot_facts
    from ouroboros.task_results import write_task_result

    write_task_result(tmp_path, "read", "running", result="x")
    facts = store_hot_facts.read_ahead([(tmp_path, "read")])
    with store_hot_facts.serving(facts, strict=True):
        assert store_hot_facts.read(tmp_path, "read")["status"] == "running"
        with pytest.raises(store_hot_facts.NotRead):
            store_hot_facts.read(tmp_path, "unread")
    with store_hot_facts.serving(facts):  # a lenient serving reads the store for what it lacks
        assert store_hot_facts.read(tmp_path, "unread") is None


def test_statuses_read_ahead_reads_only_rows_queued_since(tmp_path, monkeypatch):
    from ouroboros import store, store_hot_facts
    from ouroboros.task_results import write_task_result

    for tid in ("a", "b"):
        write_task_result(tmp_path, tid, "scheduled", result="x")
    opened, real = [], store.connection

    @contextlib.contextmanager
    def connection(*args, **kwargs):
        opened.append(1)
        with real(*args, **kwargs) as conn:
            yield conn

    monkeypatch.setattr(store, "connection", connection)
    with store_hot_facts.statuses_ahead(tmp_path, ["a"]):
        assert len(opened) == 1
        assert store_hot_facts.statuses(tmp_path, ["a"]) == {"a": "scheduled"} and len(opened) == 1
        assert store_hot_facts.statuses(tmp_path, ["a", "b"]) == {"a": "scheduled", "b": "scheduled"}
        assert len(opened) == 2  # only "b", queued since the read-ahead, was read again
