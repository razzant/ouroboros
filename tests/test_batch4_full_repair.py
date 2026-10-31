"""Full-qualification regressions through real admission and Pause consumers."""
from contextlib import ExitStack
from types import SimpleNamespace
import os
import subprocess
import sys

import pytest

from tests._budget_pause_exact_helpers import _install_queue
from tests.test_batch4_repair_compositions import _running
from tests.test_owner_pause import _owner_park
from tests.worker_pool_support import affirm_ready

pytestmark = pytest.mark.serial


def _park(root, monkeypatch, queue, workers):
    from ouroboros import budget_pause
    from supervisor.events_budget import install_exact_budget_pause
    from supervisor.owner_pause_control import request_owner_pause

    assert request_owner_pause("root", request_id="qualified-pause")["ok"]
    row = _owner_park(root, monkeypatch, "root", [], root="root")
    ctx = SimpleNamespace(DRIVE_ROOT=root, RUNNING=workers.RUNNING, PENDING=workers.PENDING,
                          WORKERS=workers.WORKERS, sort_pending=lambda: None,
                          persist_queue_snapshot=queue.persist_queue_snapshot, bridge=None)
    install_exact_budget_pause(ctx, "root", budget_pause.exact_pause_marker(row)["checkpoint"])
    return row


@pytest.mark.parametrize("member", ["root", "child"])
@pytest.mark.parametrize("split_root", [False, True])
def test_resume_waits_for_exact_root_or_child_effect_retirement(tmp_path, monkeypatch, member, split_root):
    from ouroboros import owner_pause
    from ouroboros.gateway.state import _chat_activities_snapshot_safe
    from ouroboros.task_results import load_task_result, write_task_result
    from supervisor.owner_pause_control import refresh_owner_pause_tree

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_kw: 5.0)
    root = tmp_path / "canonical" if split_root else tmp_path
    root.mkdir(exist_ok=True)
    task = _running(root, workers)
    task["budget_drive_root"] = str(root)
    if member != "root":
        write_task_result(root, member, "running", root_task_id="root", parent_task_id="root")
    source = SimpleNamespace(drive_root=root, task_id=member, root_task_id="root")
    with ExitStack() as stack:
        claim = stack.enter_context(owner_pause.tool_handoff(source, "local-operation"))
        assert owner_pause.run_operation(source, lambda: "returned") == "returned"
        assert claim["not_started"] is False
        _park(root, monkeypatch, queue, workers)
        phase = next(row["phase"] for row in _chat_activities_snapshot_safe(tmp_path)
                     if row["activity_id"] == "root")
        assert phase == "budget_paused"  # its stack parked; the handed tool is a separate blocker
        before = load_task_result(root, "root")["budget_pause"]
        refused = queue.resume_budget_paused_task("root")
        assert refused["error"] == "owner_pause_effects_unsettled", refused
        assert any(row["kind"] == "tool_handoff" and row["task_id"] == member
                   for row in refused["blockers"])
        assert load_task_result(root, "root")["budget_pause"] == before
        assert "root" in queue.BUDGET_ROOT_FENCES
        assert owner_pause.read_fence(root, "root")["state"] == "requested"
        claim["settled"] = True  # this exact local invocation has returned; consumer now retires it
    assert not load_task_result(root, member).get("launch_handoffs", {})
    refresh_owner_pause_tree("root")
    assert owner_pause.read_fence(root, "root")["state"] == "paused"
    assert queue.resume_budget_paused_task("root")["ok"]
    assert owner_pause.read_fence(root, "root")["state"] == "paused", "only consumption reopens it"


@pytest.mark.parametrize("kind", ["missing_binary", "nonzero"])
def test_real_local_process_completion_permits_saved_resume(tmp_path, monkeypatch, kind):
    from tests.test_batch4_producer_custody import _registry
    from ouroboros.task_results import load_task_result
    from supervisor import state

    registry, queue, workers = _registry(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_kw: 5.0)
    cmd = (["batch4-no-such-program-153ad9"] if kind == "missing_binary"
           else [sys.executable, "-c", "raise SystemExit(7)"])
    result = registry.execute_result("run_command", {"cmd": cmd})
    if kind == "missing_binary":
        assert result.text.startswith("⚠️ SHELL_ARG_ERROR"), result
        assert "exit_code" not in result.meta
        assert result.meta["operation_outcome"] == "completed_no_effect"
    else:
        assert result.meta["exit_code"] == 7
        assert result.meta["operation_outcome"] == "completed"
    assert not load_task_result(tmp_path, "root").get("launch_handoffs")
    _park(tmp_path, monkeypatch, queue, workers)
    assert queue.resume_budget_paused_task("root")["ok"]


def test_unmarked_missing_file_after_executor_handoff_is_not_no_effect(tmp_path, monkeypatch):
    """An unwound local body keeps its error, without claiming an executor is still alive."""
    from tests.test_batch4_producer_custody import _registry
    from ouroboros.task_results import load_task_result

    registry, queue, workers = _registry(tmp_path, monkeypatch)
    def unknown(*_a, **_kw):
        raise FileNotFoundError("missing receipt after an opaque launch")
    monkeypatch.setattr("ouroboros.tools.shell._tracked_subprocess_run", unknown)
    result = registry.execute_result("run_command", {"cmd": ["opaque"]})
    assert result.text.startswith("⚠️ SHELL_ERROR")
    assert result.meta.get("operation_outcome") != "completed_no_effect"
    assert not load_task_result(tmp_path, "root").get("launch_handoffs", {})
    _park(tmp_path, monkeypatch, queue, workers)
    assert queue.resume_budget_paused_task("root")["ok"]


@pytest.mark.parametrize("task_id", ["system:provider_test", "system:capability_probe"])
def test_host_operation_never_bypasses_global_money(tmp_path, task_id):
    """A host operation pays through the same ledger and global cap. (The task-amendment overlay it
    used to skip, the deleted task acceptance's root-cap amendments, left with the review fold.)"""
    from ouroboros import usage_accounting as ua

    scope = ua.UsageScope(drive_root=tmp_path, task_id=task_id, root_task_id=task_id,
                         non_task_operation=True, global_limit_usd=0.0)
    sent = []
    with ua.usage_scope(scope), pytest.raises(ua.BudgetExceeded):
        ua.execute_physical_attempt(ua.AttemptRequest(model="offline", provider="offline", reservation_usd=0.1),
                                    lambda: sent.append(True))
    assert not sent


def test_unretired_model_consumer_blocks_resume_but_retirement_keeps_unknown_money(tmp_path, monkeypatch):
    from ouroboros import model_wait, usage_accounting as ua
    from ouroboros.task_results import load_task_result

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_kw: 5.0)
    _running(tmp_path, workers)
    owner = model_wait.TaskModelWait(task={"id": "root", "_attempt": 1}, drive_root=tmp_path,
                                    event_queue=None, worker_slot_held=True)
    with ua.usage_scope(ua.UsageScope(drive_root=tmp_path, task_id="root", root_task_id="root")):
        with model_wait.operation_wait_scope(owner):
            reservation = ua.reserve_attempt(ua.AttemptRequest(model="m", provider="p", reservation_usd=0.1))
            ua.mark_dispatched(reservation, local_answer_owner_pid=os.getpid())
            ua.mark_unresolved(reservation, "receipt_unknown")
    _park(tmp_path, monkeypatch, queue, workers)
    assert queue.resume_budget_paused_task("root")["error"] == "owner_pause_effects_unsettled"
    owner.close()
    assert queue.resume_budget_paused_task("root")["ok"]
    assert ua.read_usage_records(tmp_path)[-1]["state"] == "unresolved"
    assert load_task_result(tmp_path, "root")["retired_model_consumers"][owner.answer_consumer_id]["task_attempt"] == 1


def test_host_observed_local_extension_return_releases_only_its_invocation(tmp_path, monkeypatch):
    """Local return is observable without claiming completion of arbitrary effects."""
    from ouroboros import extension_loader
    from ouroboros.extension_surface_names import extension_surface_name
    from ouroboros.tools.registry import ToolRegistry
    from ouroboros.task_results import load_task_result
    from tests.test_extension_registration_atomicity import _load_dispatch_extension

    name = "batch4local"
    _, root = _load_dispatch_extension(tmp_path, name)
    try:
        queue, _, workers = _install_queue(root, monkeypatch)
        _running(root, workers)
        monkeypatch.setattr("ouroboros.safety.check_safety", lambda *_a, **_kw: (True, ""))
        monkeypatch.setattr(extension_loader, "is_extension_live", lambda *_a, **_kw: True)
        registry = ToolRegistry(repo_dir=tmp_path, drive_root=root)
        registry._ctx.task_id = registry._ctx.root_task_id = "root"
        result = registry.execute_result(extension_surface_name(name, "t1"), {})
        assert result.status == "ok" and result.text == "ok"
        assert result.meta["physical_dispatch"] and result.meta["dynamic_provider"]
        claims = load_task_result(root, "root").get("launch_handoffs", {})
        assert claims == {}
        _park(root, monkeypatch, queue, workers)
        assert queue.resume_budget_paused_task("root")["ok"]
        assert load_task_result(root, "root").get("launch_handoffs", {}) == claims
    finally:
        extension_loader.unload_extension(name)


def test_settled_tool_does_not_release_independent_owned_process(tmp_path, monkeypatch):
    from ouroboros import process_custody
    from tests.test_batch4_producer_custody import _registry
    from ouroboros.task_results import load_task_result
    from supervisor import state

    registry, queue, workers = _registry(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_kw: 5.0)
    cmd = [sys.executable, "-c", "import time; time.sleep(30)"]
    process = subprocess.Popen(cmd)
    try:
        process_custody.record_process(tmp_path, pid=process.pid, cmd=cmd, purpose="repair-test",
                                      scope="task", owner_task_id="root", reap_process_group=False)
        result = registry.execute_result("read_file", {"path": "absent"})
        assert result.meta["operation_outcome"] == "completed_no_effect"
        assert not load_task_result(tmp_path, "root").get("launch_handoffs")
        _park(tmp_path, monkeypatch, queue, workers)
        refused = queue.resume_budget_paused_task("root")
        assert refused["error"] == "owner_pause_effects_unsettled"
        assert any(blocker["kind"] == "owned_process" for blocker in refused["blockers"])
    finally:
        process.terminate()
        process.wait(timeout=5)
    assert queue.resume_budget_paused_task("root")["ok"]


@pytest.mark.parametrize("scope", ["global", "root"])
def test_owner_pause_over_saved_budget_checkpoint_keeps_effect_gate_and_reopens_on_consumption(tmp_path, monkeypatch, scope):
    from ouroboros import budget_pause, owner_pause
    from tests._budget_pause_exact_helpers import _loop_ctx, _parked
    from supervisor.owner_pause_control import request_owner_pause, refresh_owner_pause_tree

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_kw: 5.0)
    _running(tmp_path, workers)
    source = SimpleNamespace(drive_root=tmp_path, task_id="root", root_task_id="root")
    with owner_pause.tool_handoff(source, "prior-local") as claim:
        owner_pause.run_operation(source, lambda: None)
        workers.RUNNING.clear()
        task, row = _parked(tmp_path, monkeypatch, task_id="root", scope=scope)
        assert row["reason"] != "owner"
        assert request_owner_pause("root", request_id="overlay")["ok"]
        refused = queue.resume_budget_paused_task("root")
        assert refused["error"] == "owner_pause_effects_unsettled", refused
        claim["settled"] = True
    refresh_owner_pause_tree("root")
    assert queue.resume_budget_paused_task("root")["ok"]
    sent = []
    workers.WORKERS[0] = affirm_ready(SimpleNamespace(wid=0, busy_task_id=None, reaping=False,
                                        in_q=SimpleNamespace(put=lambda task: sent.append(dict(task)))))
    workers.assign_tasks()
    assert [task["id"] for task in sent] == ["root"]
    loop_ctx, limit_ctx = _loop_ctx(tmp_path, "root")
    loop_ctx.budget_pause_resume = sent[0]["_budget_pause_resume"]
    saved = budget_pause.load_budget_pause(loop_ctx)
    assert saved["_pause_row"]["reason"] == row["reason"], "budget provenance is retained"
    assert saved["_pause_row"]["grant"]["owner_pause_fence_id"] == owner_pause.read_fence(tmp_path, "root")["fence_id"]
    monkeypatch.setattr("ouroboros.owner_wait.rebind_restored_route", lambda *_a, **_kw: (None, "max"))
    monkeypatch.setattr("ouroboros.owner_wait.restore_continuation_state", lambda *_a, **_kw: None)
    budget_pause.resume_paused_loop(limit_ctx.tools, saved, list(limit_ctx.messages), {}, {}, set(),
                                    budget_remaining_usd=5.0)
    assert owner_pause.read_fence(tmp_path, "root")["state"] == "released"


def test_custody_becoming_unreadable_after_paused_state_refuses_resume(tmp_path, monkeypatch):
    from ouroboros import process_custody, owner_pause

    queue, _, workers = _install_queue(tmp_path, monkeypatch)
    _running(tmp_path, workers)
    _park(tmp_path, monkeypatch, queue, workers)
    assert owner_pause.read_fence(tmp_path, "root")["state"] == "paused"
    path = process_custody.ledger_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"corrupt":\n', encoding="utf-8")
    refused = queue.resume_budget_paused_task("root")
    assert refused["error"] == "owner_pause_effects_unsettled"
    assert any(row["kind"] == "process_custody_unreadable" for row in refused["blockers"])
    assert "root" in queue.BUDGET_ROOT_FENCES


def test_owner_pause_over_sleep_retains_sleep_source_and_explicit_resume(tmp_path, monkeypatch):
    from ouroboros import budget_pause, owner_pause
    from tests._budget_pause_exact_helpers import _loop_ctx
    from tests.test_model_sleep import _cold_park, _mail
    from supervisor.owner_pause_control import request_owner_pause
    from supervisor.sleep_wake import wake_ready_sleepers

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_kw: 5.0)
    row = _cold_park(tmp_path, monkeypatch, workers, task_id="root", wake_after_sec=60)
    assert request_owner_pause("root", request_id="sleep-overlay")["ok"]
    _mail(tmp_path, "root", "wake", kind="owner_text")
    wake = wake_ready_sleepers(queue)
    assert wake[0]["error"] == "sleep_wake_vetoed" and wake[0]["veto"] == "pause"
    assert queue.resume_budget_paused_task("root")["ok"]
    sent = []
    workers.WORKERS[0] = affirm_ready(SimpleNamespace(wid=0, busy_task_id=None, reaping=False,
                                        in_q=SimpleNamespace(put=lambda task: sent.append(dict(task)))))
    workers.assign_tasks()
    assert [task["id"] for task in sent] == ["root"]
    ctx, limit = _loop_ctx(tmp_path, "root")
    ctx.budget_pause_resume = sent[0]["_budget_pause_resume"]
    saved = budget_pause.load_budget_pause(ctx)
    assert saved["_pause_row"]["reason"] == "sleep"
    assert saved["_pause_row"]["source_ref"] == row["source_ref"]
    assert saved["_pause_row"]["sleep"] == row["sleep"]
    monkeypatch.setattr("ouroboros.owner_wait.rebind_restored_route", lambda *_a, **_kw: (None, "max"))
    monkeypatch.setattr("ouroboros.owner_wait.restore_continuation_state", lambda *_a, **_kw: None)
    budget_pause.resume_paused_loop(limit.tools, saved, list(limit.messages), {}, {}, set(),
                                    budget_remaining_usd=5.0)
    assert owner_pause.read_fence(tmp_path, "root")["state"] == "released"


@pytest.mark.parametrize("stage", ["observation", "assignment", "consumption"])
def test_exact_resume_never_opens_a_replacement_owner_fence(tmp_path, monkeypatch, stage):
    from ouroboros import budget_pause, owner_pause
    from ouroboros.model_wait import ModelWaitInterrupted
    from tests._budget_pause_exact_helpers import _loop_ctx
    from supervisor import continuation_admission

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_kw: 5.0)
    _running(tmp_path, workers)
    _park(tmp_path, monkeypatch, queue, workers)
    replacement = []
    def replace_fence():
        with owner_pause.launch_lock(tmp_path, "root"):
            owner_pause.release_fence(tmp_path, "root", reason="prior generation released")
        fence, created = owner_pause.install_fence(tmp_path, "root", request_id="replacement")
        assert created
        replacement.append(fence)
    if stage == "observation":
        observe = continuation_admission.conflicting_writers
        def observe_then_replace(*args, **kwargs):
            result = observe(*args, **kwargs)
            replace_fence()
            return result
        monkeypatch.setattr(continuation_admission, "conflicting_writers", observe_then_replace)
        assert queue.resume_budget_paused_task("root")["error"] == "selection_authority_changed"
    else:
        assert queue.resume_budget_paused_task("root")["ok"]
        sent = []
        workers.WORKERS[0] = affirm_ready(SimpleNamespace(wid=0, busy_task_id=None, reaping=False,
                                            in_q=SimpleNamespace(put=lambda task: sent.append(dict(task)))))
        if stage == "assignment":
            replace_fence()
        workers.assign_tasks()
        if stage == "assignment":
            assert not sent
        else:
            assert [task["id"] for task in sent] == ["root"]
            ctx, limit = _loop_ctx(tmp_path, "root")
            ctx.budget_pause_resume = sent[0]["_budget_pause_resume"]
            saved = budget_pause.load_budget_pause(ctx)
            replace_fence()
            monkeypatch.setattr("ouroboros.owner_wait.restore_continuation_state", lambda *_a, **_kw: None)
            monkeypatch.setattr("ouroboros.owner_wait.rebind_restored_route", lambda *_a, **_kw: (None, "max"))
            monkeypatch.setattr(budget_pause, "_hold_control_reason", lambda _ctx: "stop")
            usage = {}
            with pytest.raises(ModelWaitInterrupted):
                budget_pause.resume_paused_loop(limit.tools, saved, list(limit.messages), {}, usage, set(),
                                                budget_remaining_usd=5.0)
            assert usage["budget_pause_hold"]["error"] == "ValueError: owner_pause_fence_changed"
    assert owner_pause.read_fence(tmp_path, "root") == replacement[0]
