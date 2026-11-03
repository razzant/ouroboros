"""One unsent child survives a rolled-back receipt write at its own lane: its COMMIT a reader holds
(rollback journal), or its acquisition another writer holds (WAL). Nothing is re-put on the bus."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import datetime
import queue as stdqueue
import threading
from types import SimpleNamespace

import pytest

from ouroboros import owner_pause, store, usage_accounting
from ouroboros.task_results import write_task_result
from supervisor import queue, task_admission, task_lifecycle, workers
from tests.test_admission_contention import database_reader
from tests._store_journal_shared import use_journal
from tests.test_workspace_preparation import env as env, spy_provisions, _WAIT_SEC
from tests.worker_pool_support import affirm_ready

pytestmark = pytest.mark.serial


def request(env, acting, tid="receipt-child"):
    for task_id in ("root", "parent"):
        write_task_result(env.data, task_id, "running", root_task_id="root",
                          task_contract={"delegation_budget": {"may_delegate": True, "may_fan_out": True}})
    env.ctx.event_queue = stdqueue.Queue()
    return env.request(tid) if acting else env.readonly(tid, root="root")


def install_worker(env, monkeypatch):
    from supervisor import state
    state.init(env.data, total_budget_limit=1000)
    state.init_state()
    delivered = []
    # Assignment opens only a ready slot of a live process birth (on-demand pool).
    worker = affirm_ready(SimpleNamespace(wid=0, busy_task_id=None,
                                          in_q=SimpleNamespace(put=lambda task: delivered.append(dict(task)))))
    monkeypatch.setattr(workers, "DRIVE_ROOT", env.data)
    monkeypatch.setattr(workers, "REPO_DIR", env.system)
    monkeypatch.setattr(workers, "PENDING", queue.PENDING)
    monkeypatch.setattr(workers, "RUNNING", queue.RUNNING)
    monkeypatch.setattr(workers, "WORKERS", {0: worker})
    monkeypatch.setattr(workers, "_WORKER_POOL_DISABLED_REASON", "")
    return worker, delivered


def hold_receipt(env, monkeypatch, stack, tid="receipt-child"):
    original = task_admission.write_task_result
    release, transitions, failures = [], [], []
    busy = threading.Event()

    def write(root, task_id, status, **fields):
        if task_id != tid or status != "scheduled" or (fields.get("delegation_admission") or {}).get("status") != "accepted":
            return original(root, task_id, status, **fields)
        transitions.append(fields["delegation_admission"]["transition_id"])
        if not release:
            release.append(stack.enter_context(database_reader(env.data, write=True)))
        try:
            return original(root, task_id, status, **fields)
        except store.StoreBusy as exc:
            failures.append((exc.reason, getattr(exc, "rollback_confirmed", None)))
            busy.set()
            raise

    monkeypatch.setattr(task_admission, "write_task_result", write)
    return busy, release, transitions, failures


def outside_locks(env):
    def observe():
        with queue._queue_lock, owner_pause.launch_lock(env.data, "root", wait=False):
            with usage_accounting._locked(env.data, timeout_sec=0):
                return [row["id"] for row in queue.PENDING], dict(queue.ADMISSION_RESERVATIONS)
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(observe).result(timeout=3)


@pytest.mark.parametrize("acting", [False, True], ids=["readonly", "prepared"])
def test_receipt_commit_busy_keeps_same_event_copy_and_one_worker_handoff(env, monkeypatch, acting):
    use_journal(monkeypatch, "PERSIST")  # a reader holds only a rollback-journal COMMIT
    event = request(env, acting)
    copies = spy_provisions(monkeypatch)
    worker, delivered = install_worker(env, monkeypatch)
    with ExitStack() as stack:
        busy, release, transitions, failures = hold_receipt(env, monkeypatch, stack)
        assert env.dispatch_returns(event) and busy.wait(_WAIT_SEC)
        pending, reservations = outside_locks(env)
        assert pending == [] and delivered == []
        assert failures and all(row == ("contention", True) for row in failures)
        assert env.row("receipt-child")["status"] == "requested"
        assert "receipt-child" in reservations and env.ctx.event_queue.empty()  # the claim keeps it, not the bus
        release[0]()
        assert env.gen.wait_idle(_WAIT_SEC)
    assert env.pending_ids() == ["receipt-child"] and env.ctx.event_queue.empty()
    row = env.row("receipt-child")
    assert row["status"] == "scheduled" and len(set(transitions)) == 1
    assert row["delegation_admission"]["transition_id"] == transitions[0]
    assert copies == (["receipt-child"] if acting else [])
    assert env.dispatch_returns(event) and env.gen.wait_idle(_WAIT_SEC)
    workers.assign_tasks()  # reserves under Q; the admission lane persists the launch and puts
    assert env.gen.wait_idle(_WAIT_SEC)
    workers.assign_tasks()
    assert env.gen.wait_idle(_WAIT_SEC)
    assert [task["id"] for task in delivered] == ["receipt-child"]
    assert worker.busy_task_id == "receipt-child" and "receipt-child" in queue.RUNNING
    assert env.pending_ids() == [] and queue.ADMISSION_RESERVATIONS == {}
    assert not any("not scheduled" in text for _, text, _ in env.ctx.sent)


@pytest.mark.parametrize("journal", ["PERSIST", "WAL"])
@pytest.mark.parametrize("acting", [False, True])
@pytest.mark.parametrize("control", ["stop", "pause", "deadline"])
def test_control_during_receipt_wait_prevents_worker_handoff(env, monkeypatch, acting, control, journal):
    from supervisor import worker_assignment
    from supervisor.events_budget import _set_root_budget_pause_locked
    use_journal(monkeypatch, journal)  # a held write fails its COMMIT (rollback journal), its BEGIN (WAL)
    event = request(env, acting)
    deadline = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=1)
    if control == "deadline":
        event["task_contract"] = {"deadline_at": deadline.isoformat()}
    _worker, delivered = install_worker(env, monkeypatch)
    with ExitStack() as stack:
        busy, release, transitions, _failures = hold_receipt(env, monkeypatch, stack)
        assert env.dispatch_returns(event) and busy.wait(_WAIT_SEC)
        assert outside_locks(env)[0] == [] and env.ctx.event_queue.empty()
        with queue._queue_lock:
            if control == "stop":
                task_lifecycle.CANCELLED_ROOT_FENCES["root"] = "2026-10-14T00:00:00Z"
            elif control == "pause":
                _set_root_budget_pause_locked("root", {"cause": "owner_pause", "fence_id": "during-receipt"})
        if control == "deadline":
            # The real handoff sees that the wait consumed the declared calendar horizon.
            monkeypatch.setattr(worker_assignment, "time", SimpleNamespace(time=lambda: deadline.timestamp() + 1))
            monkeypatch.setattr(queue, "_ensure_reaper_started", lambda: None)
            monkeypatch.setattr(queue, "_reap_queue", stdqueue.Queue())
        release[0]()
        assert env.gen.wait_idle(_WAIT_SEC)
    workers.assign_tasks()
    assert env.gen.wait_idle(_WAIT_SEC)
    assert delivered == [] and "receipt-child" not in queue.RUNNING
    assert len(set(transitions)) == 1
    if control == "deadline":
        assert queue.PENDING[0]["_terminalization_retry"]["trigger"] == "deadline"
    else:
        assert env.row("receipt-child")["reason_code"] == ("root_cancelled" if control == "stop" else "root_owner_paused")


@pytest.mark.parametrize("acting", [False, True])
@pytest.mark.parametrize("readback_unknown", [False, True])
def test_committed_receipt_lost_reply_preserves_custody_and_one_start(env, monkeypatch, acting, readback_unknown):
    event = request(env, acting)
    _worker, delivered = install_worker(env, monkeypatch)
    copies = spy_provisions(monkeypatch)
    write, read = task_admission.write_task_result, task_admission.load_task_result
    transitions, pending_read = [], []

    def lose_reply(root, task_id, status, **fields):
        result = write(root, task_id, status, **fields)
        if task_id == "receipt-child" and status == "scheduled" and fields.get("delegation_admission"):
            transitions.append(fields["delegation_admission"]["transition_id"])
            pending_read.append(readback_unknown)
            error = store.StoreBusy("reply lost after successful COMMIT", reason="contention")
            error.rollback_confirmed = False
            raise error
        return result

    def unknown_once(root, task_id, **kwargs):
        if task_id == "receipt-child" and pending_read and pending_read.pop():
            raise store.StoreBusy("readback unavailable", reason="uninspectable")
        return read(root, task_id, **kwargs)

    monkeypatch.setattr(task_admission, "write_task_result", lose_reply)
    monkeypatch.setattr(task_admission, "load_task_result", unknown_once)
    assert env.dispatch_returns(event) and env.gen.wait_idle(_WAIT_SEC)
    assert env.pending_ids() == ["receipt-child"] and len(transitions) == 1
    assert env.row("receipt-child")["delegation_admission"]["transition_id"] == transitions[0]
    assert env.dispatch_returns(event) and env.gen.wait_idle(_WAIT_SEC)
    workers.assign_tasks()
    assert env.gen.wait_idle(_WAIT_SEC)
    workers.assign_tasks()
    assert env.gen.wait_idle(_WAIT_SEC)
    assert [row["id"] for row in delivered] == ["receipt-child"]
    assert copies == (["receipt-child"] if acting else [])


@pytest.mark.parametrize("acting", [False, True])
def test_hard_receipt_write_failure_keeps_existing_refusal(env, monkeypatch, acting):
    event = request(env, acting)
    write = task_admission.write_task_result
    attempts = []

    def fail(root, task_id, status, **fields):
        if task_id == "receipt-child" and status == "scheduled" and fields.get("delegation_admission"):
            attempts.append(True)
            raise OSError("receipt disk unavailable")
        return write(root, task_id, status, **fields)

    monkeypatch.setattr(task_admission, "write_task_result", fail)
    assert env.dispatch_returns(event) and env.gen.wait_idle(_WAIT_SEC)
    assert env.row("receipt-child")["reason_code"] == "scheduled_result_persist_failed"
    assert attempts == [True] and env.pending_ids() == []



@pytest.mark.parametrize("journal", ["PERSIST", "WAL"])
def test_registered_schedule_subagent_reaches_one_worker_after_real_receipt_contention(env, monkeypatch, journal):
    from ouroboros.tools.registry import ToolRegistry
    from tests._shared import configure_test_subagent

    use_journal(monkeypatch, journal)  # a held write fails its COMMIT (rollback journal), its BEGIN (WAL)
    worker, delivered = install_worker(env, monkeypatch)
    write_task_result(env.data, "root", "running", root_task_id="root",
                      task_contract={"delegation_budget": {"may_delegate": True, "may_fan_out": True}})
    subagent_id = configure_test_subagent(monkeypatch, kind="api_model", target="openai/fixture-model")
    registry = ToolRegistry(repo_dir=env.source, drive_root=env.data)
    registry._ctx.task_id = registry._ctx.root_task_id = "root"
    registry._ctx.current_chat_id = 7
    registry._ctx.task_depth = 0
    bus = env.ctx.event_queue = registry._ctx.event_queue = stdqueue.Queue()
    result = registry.execute_result("schedule_subagent", {
        "subagent_id": subagent_id, "objective": "Inspect the source", "expected_output": "One finding",
        "memory_mode": "empty", "write_surface": "read_only"})
    assert result.status == "ok", result.text
    event = bus.get(timeout=2)
    assert event["type"] == "schedule_subagent" and bus.empty()
    tid = event["task_id"]
    assert env.row(tid)["status"] == "requested"
    with ExitStack() as stack:
        busy, release, transitions, failures = hold_receipt(env, monkeypatch, stack, tid)
        assert env.dispatch_returns(event) and busy.wait(_WAIT_SEC)
        assert bus.empty() and env.pending_ids() == [] and delivered == []
        assert outside_locks(env)[0] == [] and failures[0] == ("contention", True)
        release[0]()
        assert env.gen.wait_idle(_WAIT_SEC)
    assert bus.empty() and env.row(tid)["status"] == "scheduled" and len(set(transitions)) == 1
    workers.assign_tasks()
    assert env.gen.wait_idle(_WAIT_SEC)
    workers.assign_tasks()
    assert env.gen.wait_idle(_WAIT_SEC)
    assert [task["id"] for task in delivered] == [tid]
    assert worker.busy_task_id == tid and env.pending_ids() == []


def test_a_child_refused_after_its_receipt_is_no_direct_child(env, monkeypatch):
    """An owner Pause between the SCHEDULED receipt and the put settles the child never admitted,
    and its receipt's accepted admission settles with it: the parent's durable direct-child count
    stays where it was, so a single-child parent can still spawn its one child."""
    from ouroboros.task_results import STATUS_SCHEDULED
    from ouroboros.tools.control_delegation import check_delegation_admission, durable_direct_child_count
    from supervisor.events_budget import _set_root_budget_pause_locked

    event = request(env, False)
    install_worker(env, monkeypatch)
    landed, proceed = threading.Event(), threading.Event()
    real = task_admission.write_task_result

    def write(root, tid, status, **fields):
        result = real(root, tid, status, **fields)
        if tid == "receipt-child" and status == STATUS_SCHEDULED:
            landed.set()  # the receipt is durable; the put has not been entered
            assert proceed.wait(_WAIT_SEC)
        return result

    monkeypatch.setattr(task_admission, "write_task_result", write)
    before = durable_direct_child_count(env.data, "root", exclude_task_id="next-child")
    assert env.dispatch_returns(event) and landed.wait(_WAIT_SEC)
    with queue._queue_lock:
        _set_root_budget_pause_locked("root", {"cause": "owner_pause", "fence_id": "between-receipt-and-put"})
    proceed.set()
    assert env.gen.wait_idle(_WAIT_SEC)
    row = env.row("receipt-child")
    assert row["status"] == "failed" and row["admission_outcome"] == "never_admitted"
    assert row["delegation_admission"]["status"] == "never_admitted"
    after = durable_direct_child_count(env.data, "root", exclude_task_id="next-child")
    assert before == after == 0
    assert check_delegation_admission({"may_fan_out": False}, direct_child_count=after).ok
