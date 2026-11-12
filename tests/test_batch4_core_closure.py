"""Bounded recovery: exact consumer death, retained unknowns and tree membership."""

import asyncio
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tests._budget_pause_exact_helpers import _install_queue
from tests._usage_store_testing import ledger_rows
from tests.test_owner_continue import NONCE, _interrupted
from tests._store_shared import seed_result



@pytest.mark.parametrize("asynchronous", [False, True])
def test_retired_local_answer_owner_releases_same_continue_without_settling_money(tmp_path, monkeypatch, asynchronous):
    from ouroboros import platform_layer, usage_accounting as ua
    from supervisor.continuation_admission import admit_continuation
    from supervisor.queue_transitions import resume_budget_paused_task

    queue, _, workers = _install_queue(tmp_path, monkeypatch)
    _interrupted(tmp_path)
    monkeypatch.setenv("TOTAL_BUDGET", "100")
    request = ua.AttemptRequest(model="m", provider="test", drive_root=tmp_path,
                                task_id="pred-1", root_task_id="pred-1", reservation_usd=0.5)

    def failed_send():
        raise RuntimeError("transport result unknown")

    async def failed_async_send():
        return failed_send()

    from ouroboros.model_wait import TaskModelWait, operation_wait_scope
    consumer = TaskModelWait(task={"id": "pred-1"}, drive_root=tmp_path,
                             event_queue=None, worker_slot_held=True)
    with operation_wait_scope(consumer), pytest.raises(RuntimeError, match="transport result unknown"):
        if asynchronous:
            asyncio.run(ua.execute_physical_attempt_async(request, failed_async_send))
        else:
            ua.execute_physical_attempt(request, failed_send)
    before = ledger_rows(tmp_path)
    final = before[-1]
    assert final["state"] == "unresolved"
    assert final["local_answer_owner_pid"] == os.getpid()
    held = admit_continuation("pred-1", action_nonce=NONCE)
    assert held["ok"] and held["held"], "a terminal task is not a dead execution"
    successor = held["successor_task_id"]
    assert not resume_budget_paused_task(successor)["ok"]
    with patch.object(platform_layer, "pid_is_alive", side_effect=OSError("probe unavailable")):
        assert not resume_budget_paused_task(successor)["ok"]
    monkeypatch.setattr(platform_layer, "pid_is_alive", lambda pid: pid != os.getpid())
    assert not resume_budget_paused_task(successor)["ok"], "PID absence alone is not exact consumer retirement"
    consumer.close()
    assert resume_budget_paused_task(successor)["ok"]
    replay = admit_continuation("pred-1", action_nonce=NONCE)
    assert replay["successor_task_id"] == successor
    assert len(workers.PENDING) == 1
    assert ledger_rows(tmp_path) == before, "writer classification must not mutate money"
    projection = ua.usage_projection(tmp_path, billing_group_id="pred-1")
    # Releasing a local-writer hold neither settles nor forgets its liability: the
    # bound stays disclosed exposure, and (#1487) it is never counted as spending.
    assert projection["unresolved_upper_bound_usd"] == 0.5
    assert projection["settled_usd"] == 0.0 and projection["accounted_usd"] == 0.5
    assert ledger_rows(tmp_path) == before
    # An actual late receipt still settles the same liability.
    reservation = ua.AttemptReservation(final["attempt_id"], tmp_path, "m", "test", 0.5)
    ua.settle_attempt(reservation, {}, cost_usd=0.7, cost_final=True)
    assert ua.usage_projection(tmp_path, billing_group_id="pred-1")["accounted_usd"] == 0.7


def test_restart_retains_legacy_unknown_and_saved_zero_dispatch_without_inference(tmp_path, monkeypatch):
    from ouroboros.work_stop import continuation_refusal
    from ouroboros.task_results import load_task_result, write_task_result
    from supervisor.queue_transitions import resume_budget_paused_task
    from supervisor.restart_retention import never_started
    from tests.test_restart_retention import _pool_events

    queue, _, workers = _install_queue(tmp_path, monkeypatch)
    events = _pool_events(workers, monkeypatch)
    unknown = {"id": "legacy", "type": "task", "chat_id": 0, "_attempt": 1}
    zero = {"id": "zero", "type": "task", "chat_id": 0, "_attempt": 1,
            "_budget_pause": {"status": "paused_before_dispatch", "scope": "global",
                              "replay_safe": True, "physical_calls": 0}}
    for row in (unknown, zero):
        write_task_result(tmp_path, row["id"], "scheduled")
    workers.PENDING.extend([unknown, zero])
    assert not never_started(unknown) and not never_started(zero)
    assert workers.kill_workers(force=True, hold_never_started=True,
                                archive_service_logs=False, reconcile_delegate_custody=False)
    assert {t["id"] for t in workers.PENDING} == {"legacy", "zero"}
    assert not any(e.get("type") == "task_done" for e in events)
    held = next(t for t in workers.PENDING if t["id"] == "legacy")
    assert continuation_refusal(held)["reason"] == "dispatch_outcome_unknown"
    assert held.get("admitted_dispatch") != "none"
    assert resume_budget_paused_task("legacy")["error"] == "dispatch_outcome_unknown"
    # The historical positive pause keeps its original same-generation Resume.
    assert resume_budget_paused_task("zero")["ok"]
    assert load_task_result(tmp_path, "legacy")["status"] == "scheduled"
    assert queue.persist_queue_snapshot(reason="closure_test")
    workers.PENDING.clear()
    queue.restore_pending_from_snapshot()
    held = next(t for t in workers.PENDING if t["id"] == "legacy")
    assert continuation_refusal(held)["reason"] == "dispatch_outcome_unknown"
    assert held.get("admitted_dispatch") != "none"


def test_indexed_census_preserves_legacy_members_and_reads_no_foreign_body(tmp_path, monkeypatch):
    """The census selects the tree by the typed ``root_task_id`` column: a legacy member whose
    root lives only in its metadata is still read (its column is NULL), the 1000 foreign roots
    are never decoded, and an unavailable store refuses the census instead of narrowing it."""
    from ouroboros import task_store
    from ouroboros.model_sleep import cold_blockers
    from ouroboros.store import StoreBusy
    from ouroboros.task_results import stamp_task_result_schema, write_task_result
    from supervisor.continuation_admission import conflicting_writers

    queue, _, _ = _install_queue(tmp_path, monkeypatch)
    seed_result(tmp_path, "legacy-child", stamp_task_result_schema({
        "task_id": "legacy-child", "status": "completed", "metadata": {"root_task_id": "root"},
        "launch_handoffs": {"op": {"tool": "external_write"}}}))
    for number in range(1000):
        task_id = f"foreign-{number}"
        seed_result(tmp_path, task_id, stamp_task_result_schema({
            "task_id": task_id, "status": "completed", "root_task_id": task_id}))
    write_task_result(tmp_path, "root", "running", root_task_id="root")  # first touch imports the files above
    from ouroboros import task_results
    decoded = []
    real = task_store.read_document
    for module in (task_store, task_results):
        monkeypatch.setattr(module, "read_document", lambda root, task_id: (decoded.append(task_id), real(root, task_id))[1])
    assert any(b.get("task_id") == "legacy-child" for b in conflicting_writers(queue, "root"))
    assert "legacy-child" in decoded and not any(task_id.startswith("foreign-") for task_id in decoded)
    decoded.clear()
    # A top-level project sleeper: its cold sleep is the one that releases its tree's project lane.
    context = SimpleNamespace(drive_root=tmp_path, task_id="root", root_task_id="root", project_id="room")
    assert any(b.get("detail") == "legacy-child" for b in cold_blockers(context))
    assert not any(task_id.startswith("foreign-") for task_id in decoded)
    with monkeypatch.context() as busy:
        busy.setattr(task_store, "open_store", lambda *a, **kw: (_ for _ in ()).throw(StoreBusy("busy")))
        assert any(b["kind"] == "tree_census_unreadable" for b in conflicting_writers(queue, "root"))
    assert any(b.get("task_id") == "legacy-child" for b in conflicting_writers(queue, "root"))


@pytest.mark.parametrize("race", ["none", "grant", "stop", "pause", "hold", "same_bytes", "money_unreadable"])
def test_manual_synthetic_hold_gets_fresh_observation_after_mutation(tmp_path, monkeypatch, race):
    from ouroboros.cancel_intents import request_cancel
    from ouroboros.owner_pause import install_fence
    from supervisor.events_budget import _set_root_budget_pause_locked, budget_hold_fact, hold_budget_row
    from supervisor.queue_transitions import resume_budget_paused_task

    queue, _, workers = _install_queue(tmp_path, monkeypatch)
    _set_root_budget_pause_locked("root", {"scope": "root", "root_task_id": "root"})
    root = {"id": "root", "admitted_dispatch": "none"}
    child = {"id": "child", "root_task_id": "root", "admitted_dispatch": "none"}
    workers.PENDING.extend([root, child])
    assert resume_budget_paused_task("root")["ok"]
    reads = []
    entered, finish = threading.Event(), threading.Event()
    real_cost = queue.reconstruct_task_cost

    def cost(*args, **kwargs):
        assert not queue._queue_lock._is_owned()
        reads.append(bool(budget_hold_fact(child)))
        if len(reads) == 2:
            entered.set()
            assert finish.wait(4)
            if race == "money_unreadable":
                return {"cost_accounting_status": "unavailable"}
        return real_cost(*args, **kwargs)

    monkeypatch.setattr(queue, "reconstruct_task_cost", cost)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(resume_budget_paused_task, "child", selected_by="root")
        try:
            assert entered.wait(4)
            assert queue._queue_lock.acquire(timeout=0.5)
            try:
                if race == "grant":
                    root["_resume_selection"]["root_grant_id"] = "replacement-grant"
                elif race == "stop":
                    request_cancel(tmp_path, "child", reason="Stop", source="owner")
                elif race == "pause":
                    install_fence(tmp_path, "root", request_id="new-pause")
                elif race == "hold":
                    hold_budget_row(child, reason="dispatch_outcome_unknown")
                elif race == "same_bytes":
                    workers.PENDING[1] = dict(child)
            finally:
                queue._queue_lock.release()
        finally:
            finish.set()
        outcome = future.result(timeout=4)
    assert reads == [False, True], "synthetic hold invalidates pre-mutation evidence"
    assert outcome["ok"] is (race == "none")
    assert bool(budget_hold_fact(workers.PENDING[1])) is (race != "none")


@pytest.mark.parametrize("first", ["usage_admission", "usage_accounting"])
def test_review_admission_public_alias_imports_in_both_orders_and_prices_group(tmp_path, first):
    import subprocess
    import sys

    program = f'''
import importlib
importlib.import_module("ouroboros.{first}")
from ouroboros import usage_admission as admission, usage_accounting as ua
from ouroboros.tools.review_helpers import review_wave_budget_gate
from pathlib import Path
from types import SimpleNamespace
assert ua.review_wave_admission is admission.review_wave_admission
root = Path({str(tmp_path)!r})
ua._reservation_cost = lambda request: 0.25
scope = ua.UsageScope(drive_root=root, task_id="successor", root_task_id="successor",
    root_limit_usd=10, global_limit_usd=100, billing_group_id="original",
    billing_group_limit_usd=0.0, billing_group_limit_source="initial_task_admission")
with ua.usage_scope(scope):
    verdict = review_wave_budget_gate(SimpleNamespace(task_id="successor", pending_events=[]),
        surface="test", models=["test/model"], prompt_chars=100)
    assert verdict["fits"] is False and verdict["binding_axis"] == "group", verdict
'''
    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
