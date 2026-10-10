"""Bounded recovery: exact consumer death, retained unknowns and tree membership."""

import asyncio
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tests._budget_pause_exact_helpers import _install_queue
from tests._usage_store_testing import ledger_rows
from tests.test_owner_continue import NONCE, _interrupted

pytestmark = pytest.mark.serial


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


def test_unbound_model_and_generic_tool_effects_survive_terminal_owner(tmp_path, monkeypatch):
    from ouroboros import platform_layer, usage_accounting as ua
    from ouroboros.task_results import write_task_result
    from supervisor.continuation_admission import conflicting_writers

    queue, _, _ = _install_queue(tmp_path, monkeypatch)
    _interrupted(tmp_path, launch_handoffs={"op": {"tool": "external_write", "state": "claimed"}})
    monkeypatch.setenv("TOTAL_BUDGET", "100")
    reservation = ua.reserve_attempt(ua.AttemptRequest(
        model="m", provider="test", drive_root=tmp_path, task_id="pred-1",
        root_task_id="pred-1", reservation_usd=0.5))
    ua.mark_dispatched(reservation)  # legacy/tool dispatch: no sole answer consumer binding
    monkeypatch.setattr(platform_layer, "pid_is_alive", lambda pid: False)
    blockers = conflicting_writers(queue, "pred-1")
    assert {"tool_handoff", "model_handoff"} <= {b["kind"] for b in blockers}
    # Positive model receipt changes only the model operation; remote tool remains unknown.
    ua.settle_attempt(reservation, {}, cost_usd=0.5, cost_final=True)
    assert [b["kind"] for b in conflicting_writers(queue, "pred-1")] == ["tool_handoff"]
    write_task_result(tmp_path, "pred-1", "cancelled", launch_handoffs={})
    assert conflicting_writers(queue, "pred-1") == []


def test_unreadable_pause_preserves_cause_and_prevents_physical_send(tmp_path, monkeypatch):
    from ouroboros import owner_pause, usage_accounting as ua
    from ouroboros.llm_attempt import PhysicalDispatchInterrupted
    from ouroboros.task_results import task_result_path

    monkeypatch.setenv("TOTAL_BUDGET", "100")
    path = task_result_path(tmp_path, "root")
    path.write_text("{torn")
    source = SimpleNamespace(task_id="root", root_task_id="root", drive_root=tmp_path)
    with pytest.raises(owner_pause.OwnerPauseRefused, match="owner_pause_authority_unreadable"):
        with owner_pause.launch_admission(source):
            pytest.fail("unknown authority admitted a tool")
    sent = []
    with pytest.raises(PhysicalDispatchInterrupted) as exc:
        ua.execute_physical_attempt(ua.AttemptRequest(
            model="m", provider="test", drive_root=tmp_path, task_id="root",
            root_task_id="root", reservation_usd=0.5), lambda: sent.append(True))
    assert exc.value.control_reason == "owner_pause_authority_unreadable"
    assert sent == []
    assert ledger_rows(tmp_path)[-1]["state"] == "released"


def test_restart_retains_legacy_unknown_and_saved_zero_dispatch_without_inference(tmp_path, monkeypatch):
    from ouroboros.task_results import load_task_result, write_task_result
    from supervisor.events_budget import budget_hold_fact
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
    assert budget_hold_fact(held)["reason"] == "dispatch_outcome_unknown"
    assert held.get("admitted_dispatch") != "none"
    assert resume_budget_paused_task("legacy")["error"] == "dispatch_outcome_unknown"
    # The historical positive pause keeps its original same-generation Resume.
    assert resume_budget_paused_task("zero")["ok"]
    assert load_task_result(tmp_path, "legacy")["status"] == "scheduled"
    assert queue.persist_queue_snapshot(reason="closure_test")
    workers.PENDING.clear()
    queue.restore_pending_from_snapshot()
    held = next(t for t in workers.PENDING if t["id"] == "legacy")
    assert budget_hold_fact(held)["reason"] == "dispatch_outcome_unknown"
    assert held.get("admitted_dispatch") != "none"


def test_owner_pause_cause_survives_root_selection_but_not_new_budget_pause(tmp_path, monkeypatch):
    from ouroboros.task_results import load_task_result, write_task_result
    from supervisor.events_budget import (
        BUDGET_HOLD_KEY, _set_root_budget_pause_locked, hold_root_resume_descendants,
    )
    from supervisor.queue_transitions import budget_pause_fact

    queue, _, workers = _install_queue(tmp_path, monkeypatch)
    child = {"id": "child", "root_task_id": "root", "admitted_dispatch": "none"}
    workers.PENDING.append(child)
    write_task_result(tmp_path, "child", "scheduled", root_task_id="root")
    fence = _set_root_budget_pause_locked("root", {"cause": "owner_pause", "fence_id": "owner-f"})
    queue.BUDGET_ROOT_FENCES.pop("root")
    hold_root_resume_descendants(queue, "root", fence, {"grant_id": "g1", "generation": 1})
    assert budget_pause_fact(child)["cause"] == "owner_pause"
    assert load_task_result(tmp_path, "child")["reason_code"] == "owner_paused"
    # A new monetary pause must not inherit the old owner's cause.
    fence = _set_root_budget_pause_locked("root", {"reason": "budget", "fence_id": "money-f"})
    assert "cause" not in fence
    queue.BUDGET_ROOT_FENCES.pop("root")
    hold_root_resume_descendants(queue, "root", fence, {"grant_id": "g2", "generation": 2})
    assert child[BUDGET_HOLD_KEY]["cause"] == ""
    assert child[BUDGET_HOLD_KEY]["root_grant_id"] == "g2"


def test_indexed_census_preserves_legacy_members_and_reloads_only_changed_foreign_rows(tmp_path, monkeypatch):
    from ouroboros import task_result_facts as scan
    from ouroboros.model_sleep import cold_blockers
    from ouroboros.task_results import stamp_task_result_schema, task_result_path, write_task_result
    from supervisor.continuation_admission import conflicting_writers

    queue, _, _ = _install_queue(tmp_path, monkeypatch)
    write_task_result(tmp_path, "root", "running", root_task_id="root")
    path = task_result_path(tmp_path, "legacy-child")
    path.write_text(json.dumps(stamp_task_result_schema({
        "task_id": "legacy-child", "status": "completed", "metadata": {"root_task_id": "root"},
        "launch_handoffs": {"op": {"tool": "external_write"}}})))
    for number in range(1000):
        task_id = f"foreign-{number}"
        task_result_path(tmp_path, task_id).write_text(json.dumps(stamp_task_result_schema({
            "task_id": task_id, "status": "completed", "root_task_id": task_id})))
    parsed = []
    reader = scan.read_json_dict
    monkeypatch.setattr(scan, "read_json_dict", lambda path: (parsed.append(path.name), reader(path))[1])
    assert any(b.get("task_id") == "legacy-child" for b in conflicting_writers(queue, "root"))
    assert len(parsed) == 1002
    parsed.clear()
    context = SimpleNamespace(drive_root=tmp_path, task_id="root", root_task_id="root")
    assert any(b.get("detail") == "legacy-child" for b in cold_blockers(context))
    assert parsed == []
    changed = task_result_path(tmp_path, "foreign-0")
    changed.write_text("{torn")
    assert any(b["kind"] == "tree_census_unreadable" for b in conflicting_writers(queue, "root"))
    assert parsed == ["foreign-0.json"]
    changed.unlink()
    assert any(b.get("task_id") == "legacy-child" for b in conflicting_writers(queue, "root"))


@pytest.mark.parametrize("manual", [False, True])
@pytest.mark.parametrize("race", ["none", "restart", "pause", "stop", "panic", "replacement", "same_bytes", "unreadable"])
def test_settled_continue_observes_off_lock_and_revalidates_owner_authority(tmp_path, monkeypatch, race, manual):
    from ouroboros import budget_pause
    from ouroboros.cancel_intents import request_cancel
    from ouroboros.owner_pause import install_fence
    from ouroboros.task_results import write_task_result
    from supervisor.continuation_admission import admit_continuation, release_settled_continuations
    from supervisor.events_budget import budget_hold_fact
    from supervisor.restart_retention import hold_for_owner_restart
    from supervisor.queue_transitions import resume_budget_paused_task

    queue, _, workers = _install_queue(tmp_path, monkeypatch)
    _interrupted(tmp_path, launch_handoffs={"op": {"tool": "external_write"}})
    accepted = admit_continuation("pred-1", action_nonce=NONCE)
    assert accepted["held"]
    successor = accepted["successor_task_id"]
    write_task_result(tmp_path, "pred-1", "cancelled", launch_handoffs={})
    entered, finish = threading.Event(), threading.Event()
    real_observe = budget_pause.observe_task_runs

    def observe(*args, **kwargs):
        entered.set()
        assert finish.wait(4)
        return {"custody_read": "unknown"} if race == "unreadable" else real_observe(*args, **kwargs)

    real_cost = queue.reconstruct_task_cost
    money_reads = []

    def cost(*args, **kwargs):
        assert not queue._queue_lock._is_owned(), "ledger observation occupied queue lock"
        money_reads.append(True)
        return real_cost(*args, **kwargs)

    monkeypatch.setattr(budget_pause, "observe_task_runs", observe)
    monkeypatch.setattr(queue, "reconstruct_task_cost", cost)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = (pool.submit(resume_budget_paused_task, successor) if manual else
                  pool.submit(release_settled_continuations, "pred-1"))
        try:
            assert entered.wait(4)
            assert queue._queue_lock.acquire(timeout=0.5), "daemon observation occupied queue lock"
            try:
                task = workers.PENDING[0]
                if race == "restart":
                    hold_for_owner_restart(task, tmp_path)
                elif race == "pause":
                    install_fence(tmp_path, successor, request_id="new-owner-pause")
                elif race == "stop":
                    request_cancel(tmp_path, successor, reason="new owner Stop", source="owner")
                elif race == "panic":
                    (tmp_path / "state" / "panic_stop.flag").write_text("panic")
                elif race == "same_bytes":
                    workers.PENDING[:] = [dict(task)]
                elif race == "replacement":
                    workers.PENDING[:] = [{**task, "_attempt": 2}]
            finally:
                queue._queue_lock.release()
        finally:
            finish.set()
        released = future.result(timeout=4)
    assert money_reads
    assert (released["ok"] if manual else released == [successor]) is (race == "none")
    assert bool(budget_hold_fact(workers.PENDING[0])) is (race != "none")


def test_export_and_import_keep_local_consumer_binding_and_late_receipt(tmp_path, monkeypatch):
    from ouroboros import usage_accounting as ua, usage_store

    monkeypatch.setenv("TOTAL_BUDGET", "100")
    for index in range(3):
        settled = ua.reserve_attempt(ua.AttemptRequest(model="m", provider="test", drive_root=tmp_path,
            task_id=f"done-{index}", root_task_id=f"done-{index}", reservation_usd=0.1))
        ua.mark_dispatched(settled)
        ua.settle_attempt(settled, {}, cost_usd=0.1, cost_final=True)
    reservation = ua.reserve_attempt(ua.AttemptRequest(model="m", provider="test", drive_root=tmp_path,
        task_id="local", root_task_id="local", reservation_usd=0.5))
    ua.mark_dispatched(reservation, local_answer_owner_pid=os.getpid())
    ua.mark_unresolved(reservation, "unknown response")

    def current():
        row = {r["attempt_id"]: r for r in ledger_rows(tmp_path)}[reservation.attempt_id]
        return {key: value for key, value in row.items() if key != "seq"}

    before = current()
    # The downgrade export, then the next access's one-time import of it.
    usage_store.export_journal(tmp_path)
    assert current() == before
    ua.settle_attempt(reservation, {}, cost_usd=0.7, cost_final=True)
    row = current()
    assert row["local_answer_owner_pid"] == os.getpid() and row["cost_usd"] == 0.7


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
                    root["_budget_pause_hold"]["root_grant_id"] = "replacement-grant"
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


def test_owner_pause_wake_ack_is_control_receipt_not_transcript_or_settlement(tmp_path):
    import queue
    from ouroboros.loop_round_limits import _drain_incoming_messages
    from ouroboros.owner_mailbox import (
        KIND_OWNER_PAUSE, acknowledged_task_message_ids, drain_owner_entries, write_owner_message,
    )
    from ouroboros.owner_pause import install_fence, read_fence

    from ouroboros.task_results import write_task_result
    write_task_result(tmp_path, "root", "running", root_task_id="root")
    install_fence(tmp_path, "root", request_id="p")
    assert write_owner_message(tmp_path, "wake", "root", msg_id="pause-p", kind=KIND_OWNER_PAUSE)
    before = read_fence(tmp_path, "root")
    messages = []
    controls = _drain_incoming_messages(messages, queue.Queue(), tmp_path, "root", None, set())
    assert controls == {} and messages == []
    assert acknowledged_task_message_ids(tmp_path, "root") == {"pause-p"}
    assert drain_owner_entries(tmp_path, "root", set()) == []
    assert read_fence(tmp_path, "root") == before
    acks = (tmp_path / "memory" / "owner_mailbox" / "root.acks.jsonl").read_text()
    assert json.loads(acks)["wake_id"] == "owner_pause_wake"


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


@pytest.mark.parametrize("dispatch", [None, "possible", "none"])
def test_fresh_owner_pause_and_zero_ledger_do_not_prove_legacy_dispatch(tmp_path, monkeypatch, dispatch):
    from ouroboros.owner_pause import read_fence
    from ouroboros.task_results import write_task_result
    from supervisor.owner_pause_control import request_owner_pause
    from supervisor.queue_transitions import resume_budget_paused_task

    queue, _, workers = _install_queue(tmp_path, monkeypatch)
    task = {"id": "legacy", "root_task_id": "legacy", "type": "task", "chat_id": 0, "_attempt": 1}
    if dispatch is not None:
        task["admitted_dispatch"] = dispatch
    write_task_result(tmp_path, "legacy", "scheduled", root_task_id="legacy")
    workers.PENDING.append(task)
    assert request_owner_pause("legacy", request_id="p")["ok"]
    before = read_fence(tmp_path, "legacy")
    result = resume_budget_paused_task("legacy")
    if dispatch is not None:
        # Recorded evidence decides the path; a Resume never rewrites it.
        assert result["ok"] and result["owner_pause_released"]
        assert result["never_started"] is (dispatch == "none")
        assert read_fence(tmp_path, "legacy")["state"] == "released"
    else:
        assert result["error"] == "dispatch_outcome_unknown"
        assert read_fence(tmp_path, "legacy") == before
    assert task.get("admitted_dispatch") == dispatch


def test_escaping_tool_exception_keeps_unknown_effects_after_owner_terminal(tmp_path, monkeypatch):
    from ouroboros.owner_pause import operation_start, tool_handoff
    from ouroboros.task_results import load_task_result, write_task_result
    from supervisor.continuation_admission import conflicting_writers

    queue, _, _ = _install_queue(tmp_path, monkeypatch)
    write_task_result(tmp_path, "root", "running", root_task_id="root")
    source = SimpleNamespace(task_id="root", root_task_id="root", drive_root=tmp_path)
    with pytest.raises(TimeoutError):
        with tool_handoff(source, "external_write"):
            with operation_start(source):
                raise TimeoutError("request sent, response unknown")
    claim = load_task_result(tmp_path, "root")["launch_handoffs"]
    assert claim
    write_task_result(tmp_path, "root", "cancelled")
    assert load_task_result(tmp_path, "root")["launch_handoffs"] == claim
    assert any(b["kind"] == "tool_handoff" for b in conflicting_writers(queue, "root"))


def test_pause_corpus_matches_actual_registry_producer(tmp_path):
    from ouroboros.owner_pause import install_fence
    from ouroboros.task_results import write_task_result
    from ouroboros.tools.registry import ToolRegistry
    from tests.tool_classification_corpus import _PRODUCER_SHAPES

    write_task_result(tmp_path, "root", "running", root_task_id="root")
    install_fence(tmp_path, "root", request_id="p")
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.task_id = registry._ctx.root_task_id = "root"
    registry._ctx.budget_drive_root = tmp_path
    result = registry.execute_result("knowledge_read", {"topic": "x"})
    row = next(row for row in _PRODUCER_SHAPES if row[0] == "owner_pause_not_started")
    assert result.text == row[2] and result.code == row[3]
    assert result.meta["owner_pause_not_started"] is True


def test_unstarted_resume_never_reopens_a_replaced_owner_fence(tmp_path, monkeypatch):
    from contextlib import contextmanager

    from ouroboros import owner_pause
    from ouroboros.task_results import write_task_result
    from supervisor.owner_pause_control import request_owner_pause
    from supervisor.queue_transitions import resume_budget_paused_task

    queue, _, workers = _install_queue(tmp_path, monkeypatch)
    task = {"id": "root", "root_task_id": "root", "admitted_dispatch": "none", "chat_id": 0}
    write_task_result(tmp_path, "root", "scheduled", root_task_id="root")
    workers.PENDING.append(task)
    assert request_owner_pause("root", request_id="first")["ok"]
    real_lock = owner_pause.launch_lock
    installed = []
    replacing = False

    @contextmanager
    def replace_before_claim(*args, **kwargs):
        nonlocal replacing
        if not replacing:
            replacing = True
            # A real replacement can win before the final lock, never inside
            # it: install_fence also takes the process lock (non-reentrant).
            with real_lock(*args, **kwargs):
                owner_pause.release_fence(tmp_path, "root", reason="earlier owner Resume")
            newer, _ = owner_pause.install_fence(tmp_path, "root", request_id="newer")
            installed.append(newer)
        with real_lock(*args, **kwargs):
            yield

    monkeypatch.setattr(owner_pause, "launch_lock", replace_before_claim)
    assert resume_budget_paused_task("root")["error"] == "selection_authority_changed"
    assert owner_pause.read_fence(tmp_path, "root") == installed[0]
    assert "root" in queue.BUDGET_ROOT_FENCES
