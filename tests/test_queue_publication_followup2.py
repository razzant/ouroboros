"""Queue publication, review follow-up 2 (RV_MAIN M5, M6, M7, W9 and the format stamp of the export).

M5: a mark is never read as "persisted" — it answers None while the tick publishes, every
runtime caller that acts on the publication reaches the barrier, and a guard finds a caller
that reads the mark's value. M6: a fact the publication carries that changed under Q without
a mark is published by the next tick (the census of volatile row facts). M7: the live root
cost figure is marked by a call-state frame, a wait lease and a server-side custody
transition, so the flush reads an in-flight call or an open run. W9: the Runtime queue fact
counts project slots; lane carriers are reported apart. The rollback export carries the
format an older body treats as current.
"""

import ast
import json
import pathlib
import threading
import types
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from ouroboros import queue_observation, task_status
from supervisor import queue, queue_publication, queue_snapshot
from tests._source_scan import parse

REPO = pathlib.Path(__file__).resolve().parents[1]
MARK = "persist_queue_snapshot"


@pytest.fixture
def published(tmp_path, monkeypatch):
    now = [1_800_000_000.0]
    monkeypatch.setattr(queue, "DRIVE_ROOT", tmp_path)
    monkeypatch.setattr(queue, "QUEUE_SNAPSHOT_PATH", tmp_path / "state/queue_snapshot.json")
    monkeypatch.setattr(queue, "QUEUE_PUBLICATION", None)
    monkeypatch.setattr(queue, "INITIALIZED", True)
    for name, value in (("PENDING", []), ("RUNNING", {}), ("BUDGET_ROOT_FENCES", {}), ("ACCEPTANCE_FENCES", {})):
        monkeypatch.setattr(queue, name, value)
    monkeypatch.setattr(queue_snapshot, "_RESTORED", set())
    monkeypatch.setattr(queue_snapshot, "_RESTORING", {})
    monkeypatch.setattr(queue_publication, "utc_now_iso", lambda: datetime.fromtimestamp(now[0], timezone.utc).isoformat())
    monkeypatch.setattr(queue_publication, "time", SimpleNamespace(monotonic=lambda: now[0], sleep=lambda _s: None))
    for module in (queue_observation, task_status):
        monkeypatch.setattr(module, "time", SimpleNamespace(time=lambda: now[0]))
    monkeypatch.setattr(queue_observation, "_sleep", lambda _sec: None)
    queue_publication.reset()
    yield tmp_path, now
    queue_publication.reset()


def _read(root, name):
    return json.loads((root / "state" / name).read_text(encoding="utf-8"))


def _task(task_id, **extra):
    return {"id": task_id, "type": "task", "chat_id": 1, "root_task_id": task_id, "text": f"task {task_id}", **extra}


# ---- M5: a mark is never a persisted fact -------------------------------------------------------

def test_a_mark_answers_none_while_the_tick_publishes_and_true_only_when_on_disk(published):
    root, _now = published
    queue.PENDING.append(_task("a"))
    assert queue.persist_queue_snapshot("admit") is True, "no tick publisher: the caller published"
    assert [row["id"] for row in _read(root, "queue_snapshot.json")["pending"]] == ["a"]
    queue_publication.tick_publisher_started()
    queue.PENDING.append(_task("b"))
    assert queue.persist_queue_snapshot("admit") is None, "a mark: nothing is on disk yet"
    assert [row["id"] for row in _read(root, "queue_snapshot.json")["pending"]] == ["a"]
    assert queue.publish_queue_snapshot("admit") is True, "the barrier: on disk before it returns"
    assert [row["id"] for row in _read(root, "queue_snapshot.json")["pending"]] == ["a", "b"]
    gone = threading.Thread(target=lambda: None)
    gone.start(); gone.join()
    queue_publication._PUBLISHER = gone
    queue.PENDING.append(_task("c"))
    assert queue.persist_queue_snapshot("admit") is True, "the loop is gone: the caller publishes again"


def test_barrier_of_prefers_the_contexts_barrier_then_its_stand_in_then_nothing():
    calls = []
    both = types.SimpleNamespace(publish_queue_snapshot=lambda **kw: calls.append(("barrier", kw)) or True,
                                 persist_queue_snapshot=lambda **kw: calls.append(("mark", kw)) or None)
    assert queue_publication.barrier_of(both)(reason="x") is True and calls == [("barrier", {"reason": "x"})]
    stub_only = types.SimpleNamespace(persist_queue_snapshot=lambda **kw: calls.append(("stub", kw)) or True)
    assert queue_publication.barrier_of(stub_only)(reason="y") is True and calls[-1] == ("stub", {"reason": "y"})
    assert queue_publication.barrier_of(types.SimpleNamespace()) is None, "no persister: the caller decides"


def test_the_servers_event_context_carries_the_barrier_beside_the_mark():
    source = (REPO / "server.py").read_text(encoding="utf-8")
    assert "publish_queue_snapshot=publish_queue_snapshot" in source and "persist_queue_snapshot=persist_queue_snapshot" in source


def _runtime_files():
    for folder in ("ouroboros", "supervisor"):
        yield from sorted((REPO / folder).rglob("*.py"))
    yield REPO / "server.py"


def _names_the_mark(func: ast.AST) -> bool:
    return (isinstance(func, ast.Attribute) and func.attr == MARK) or (isinstance(func, ast.Name) and func.id == MARK)


def test_no_runtime_caller_reads_the_marks_value_or_reaches_it_through_getattr():
    """The mark answers None while the tick publishes: a caller that acts on the publication
    (a reply, a confirmation, a custody fact, a rollback decision) must call the barrier. A call
    whose value is used, and a ``getattr(ctx, "persist_queue_snapshot")`` outside the one barrier
    seam (``queue_publication.barrier_of``) and the one statement-only rollback, is an offender."""
    allowed_value_users = {
        "supervisor/queue_publication.py",  # the barrier's own delegation to the mark without a tick publisher
    }
    allowed_getattr = {
        "supervisor/queue_publication.py",  # barrier_of
        "supervisor/events_project_routing.py",  # the promote rollback: called as a statement, value unused
    }
    offenders = []
    for path in _runtime_files():
        rel = path.relative_to(REPO).as_posix()
        tree = parse(path.read_text(encoding="utf-8"))
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if _names_the_mark(node.func) and not isinstance(parents.get(node), ast.Expr) and rel not in allowed_value_users:
                offenders.append(f"{rel}:{node.lineno} reads the mark's value")
            if (isinstance(node.func, ast.Name) and node.func.id == "getattr" and len(node.args) >= 2
                    and isinstance(node.args[1], ast.Constant) and node.args[1].value == MARK and rel not in allowed_getattr):
                offenders.append(f"{rel}:{node.lineno} reaches the mark through getattr")
    assert offenders == [], offenders


# ---- M6: a fact changed under Q without a mark is published by the next tick ----------------------

def _running(task_id, now):
    return {"task": _task(task_id, model_waits={}), "attempt": 1, "worker_id": 3, "started_at": now - 100,
            "last_heartbeat_at": now - 1, "last_progress_at": now - 50, "budget_paused_sec": 0.0, "soft_sent": False}


def test_an_unmarked_change_of_a_published_row_fact_is_published_by_the_tick(published):
    root, now = published
    queue.RUNNING["r"] = _running("r", now[0])
    assert queue.persist_queue_snapshot("start") is True
    queue_publication.tick_publisher_started()
    generation = _read(root, "queue.alive")["generation"]
    assert queue_publication.publish_queue_changes() is True
    assert _read(root, "queue.alive")["generation"] == generation, "an unchanged live set publishes nothing"
    # silence_attention.observe: a worker report mutates RUNNING in place under Q with no mark (#1376's source).
    with queue._queue_lock:
        queue.RUNNING["r"]["silence_attention"] = {"episode_id": "e1", "state": "pending_attention", "last_progress_at": now[0] - 50}
    assert queue_publication.publish_queue_changes() is True
    row = _read(root, "queue_snapshot.json")["running"][0]
    assert row["silence_attention"]["episode_id"] == "e1" and row["last_progress_at"] == now[0] - 50
    second = _read(root, "queue.alive")["generation"]
    assert second != generation, "the census saw the unmarked change"
    # The progress that revokes the episode (work_diagnostics reads it from the snapshot) is published too.
    with queue._queue_lock:
        queue.RUNNING["r"]["last_progress_at"] = now[0] - 10
    assert queue_publication.publish_queue_changes() is True
    assert _read(root, "queue_snapshot.json")["running"][0]["last_progress_at"] == now[0] - 10
    # The quota clock (task_model_wait) and the model waits a worker reports are watched the same way.
    with queue._queue_lock:
        queue.RUNNING["r"]["model_wait_quota_clock"] = {"revision": 2, "active": True, "elapsed_sec": 3, "observed_at": now[0]}
        queue.RUNNING["r"]["task"]["model_waits"] = {"w1": {"state": "waiting", "revision": 1, "task_attempt": 1}}
    third = _read(root, "queue.alive")["generation"]
    assert queue_publication.publish_queue_changes() is True
    row = _read(root, "queue_snapshot.json")["running"][0]
    assert row["model_wait_quota_clock"]["revision"] == 2 and row["task"]["model_waits"]["w1"]["state"] == "waiting"
    assert _read(root, "queue.alive")["generation"] != third
    # Rest law: another tick over the unchanged rows writes no document.
    fourth = _read(root, "queue.alive")["generation"]
    assert queue_publication.publish_queue_changes() is True
    assert _read(root, "queue.alive")["generation"] == fourth


def test_the_census_watches_exactly_the_row_facts_the_capture_publishes():
    """One list for both: a fact the capture writes is a fact the census watches, by construction."""
    now = 1_800_000_000.0
    meta = _running("r", now)
    meta["silence_attention"] = {"state": "pending_attention"}
    facts = queue_publication._row_facts(meta)
    assert set(facts) == set(queue_publication._ROW_FACTS) | {"silence_attention", "last_progress_at"}
    meta["silence_attention"] = {"state": "progress_observed"}
    assert "last_progress_at" not in queue_publication._row_facts(meta), "ordinary progress rotates nothing"


# ---- W9: project slots counted, lane carriers apart -----------------------------------------------

def test_pool_counts_count_project_slots_and_report_lane_carriers_apart(monkeypatch):
    from supervisor import workers
    from supervisor.worker_pool_lifecycle import LANE_CONVERSATION, LANE_PROJECT

    def slot(wid, lane, busy=None):
        return types.SimpleNamespace(wid=wid, lane=lane, busy_task_id=busy, active_capacity=True, resume_hold="",
                                     reaping=False, readiness_exhausted=False, process_birth="b",
                                     ready_identity=(100 + wid, "b"), proc=types.SimpleNamespace(pid=100 + wid, is_alive=lambda: True))

    monkeypatch.setattr(workers, "WORKERS", {0: slot(0, LANE_CONVERSATION), 1: slot(1, LANE_PROJECT), 2: slot(2, LANE_PROJECT, busy="t")})
    monkeypatch.setattr(workers, "_worker_pool_execution_state", lambda: {"disabled_reason": ""})
    from supervisor import worker_pool_lifecycle
    monkeypatch.setattr(worker_pool_lifecycle, "_slot_pid", lambda s: s.proc.pid)
    counts = queue_publication._pool_counts()
    assert counts["worker_total"] == 2 and counts["assignable_idle_workers"] == 1, "the lane's spare is not a project slot"
    assert counts["active_worker_count"] == 2 and counts["parked_worker_count"] == 0
    assert counts["lane_carrier_total"] == 1 and counts["lane_carrier_idle"] == 1


# ---- the rollback export carries the format an older body treats as current -----------------------

def test_the_rollback_export_is_stamped_format_one(published):
    from supervisor.work_stop_migration import FORMAT_KEY, FORMAT_VERSION

    root, _now = published
    queue.PENDING.append(_task("a", origin_message_text="x" * 2000, metadata={"owner_corpus": "x" * 2000}))
    assert queue.persist_queue_snapshot("admit") is True
    assert _read(root, "queue_snapshot.json")[FORMAT_KEY] == FORMAT_VERSION
    assert queue.publish_queue_snapshot("server_shutdown") is True
    exported = _read(root, "queue_snapshot.json")
    assert exported[FORMAT_KEY] == 1 and queue_observation.SHARED_KEY not in exported
    assert exported["pending"][0]["task"]["origin_message_text"] == "x" * 2000
    assert queue_observation.load_queue_document(root, fresh=True)["pending"][0]["task"]["origin_message_text"] == "x" * 2000


# ---- M7: the live root cost figure sees a call in flight and an open run -------------------------

def _memo_reads(monkeypatch):
    from ouroboros import usage_accounting
    reads = []
    monkeypatch.setattr(usage_accounting, "usage_breakdown", lambda *a, **k: reads.append(k) or {
        "accounted_usd": 1.0, "attempt_counts": {"settled": 1}, "priced_rows": 1, "tracked_nonfinal_rows": 0,
        "accounting_open_rows": 1, "reserved_usd": 0.5})
    return reads


def test_a_call_state_frame_marks_the_root_so_the_flush_reads_the_in_flight_call(tmp_path, monkeypatch):
    from ouroboros.cost_projection import LIVE_ROOT_COSTS
    from ouroboros.server_liveness import flush_budget_projection
    from supervisor.events import _handle_main_llm_call_state

    reads = _memo_reads(monkeypatch)
    task = {"id": "root", "type": "task", "root_task_id": "root", "_attempt": 1}
    running = {"root": {"task": task, "attempt": 1, "started_at": 1.0}}
    ctx = types.SimpleNamespace(DRIVE_ROOT=tmp_path, RUNNING=running, PENDING=[], update_budget_from_usage=lambda u: True)
    assert LIVE_ROOT_COSTS.fields(tmp_path, "root", task) == {}
    flush_budget_projection(ctx)
    assert len(reads) == 1, "first sight: one read"
    flush_budget_projection(ctx)
    assert len(reads) == 1, "unchanged money: no read"
    frame = {"task_id": "root", "phase": "started", "llm_call_id": "c1", "execution_id": "e1", "round_id": "r1",
             "task_attempt": 1, "call_attempt": 1}
    _handle_main_llm_call_state(frame, ctx)
    assert "active_llm_call" in running["root"]
    flush_budget_projection(ctx)
    assert len(reads) == 2, "a call opened: the figure is read again (the reservation is an open row)"
    assert LIVE_ROOT_COSTS.fields(tmp_path, "root", task)["cost_presentation"]["accounting_open"] is True
    _handle_main_llm_call_state({**frame, "phase": "observed", "model_activity": {"kind": "reasoning"}}, ctx)
    flush_budget_projection(ctx)
    assert len(reads) == 2, "an observation inside the call changes no money"
    _handle_main_llm_call_state({**frame, "phase": "finished"}, ctx)
    flush_budget_projection(ctx)
    assert len(reads) == 3, "a call closed: read again"


def test_a_wait_lease_and_a_custody_transition_mark_the_root(tmp_path, monkeypatch):
    from ouroboros.cost_projection import LIVE_ROOT_COSTS
    from ouroboros.delegate_custody_reconcile import _mark_root_cost
    from ouroboros.server_liveness import flush_budget_projection
    from supervisor.events_worker_reports import _handle_external_wait_lease

    reads = _memo_reads(monkeypatch)
    task = {"id": "root", "type": "task", "root_task_id": "root"}
    child = {"id": "child", "type": "task", "root_task_id": "root", "parent_task_id": "root", "delegation_role": "subagent"}
    running = {"root": {"task": task, "attempt": 1}, "child": {"task": child, "attempt": 1}}
    ctx = types.SimpleNamespace(DRIVE_ROOT=tmp_path, RUNNING=running, update_budget_from_usage=lambda u: True)
    LIVE_ROOT_COSTS.fields(tmp_path, "root", task)
    flush_budget_projection(ctx)
    flush_budget_projection(ctx)
    assert len(reads) == 1
    _handle_external_wait_lease({"task_id": "child", "run_id": "run-1", "until_ts": 9e12, "lease_id": "l1"}, ctx)
    assert running["child"]["external_wait_lease_run_id"] == "run-1"
    flush_budget_projection(ctx)
    assert len(reads) == 2, "a hold over an open run marks the ROOT, not the child"
    _handle_external_wait_lease({"task_id": "child", "run_id": "run-1", "until_ts": 0, "lease_id": "l1"}, ctx)
    flush_budget_projection(ctx)
    assert len(reads) == 3, "the hold's release marks it again"
    _mark_root_cost(tmp_path, types.SimpleNamespace(root_task_id="root", task_id="child", run_id="run-1"))
    flush_budget_projection(ctx)
    assert len(reads) == 4, "a settlement this process made is a custody transition"
    flush_budget_projection(ctx)
    assert len(reads) == 4
