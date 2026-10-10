"""Adversarial consumers at the Batch4 launch and sleep boundaries."""
import asyncio
import copy
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from tests._budget_pause_exact_helpers import _install_queue, _loop_ctx
from tests.test_batch4_repair_compositions import _running
from tests.test_restart_retention import _pool_events, _restart_door

pytestmark = pytest.mark.serial


def test_worker_preparation_cannot_launch_after_pause(tmp_path, monkeypatch):
    from tests.test_budget_pause_holds import _idle_worker
    from ouroboros.task_results import write_task_result
    from supervisor.owner_pause_control import request_owner_pause
    _, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_k: 5)
    write_task_result(tmp_path, "root", "scheduled", root_task_id="root")
    task = {"id": "root", "root_task_id": "root", "type": "task", "chat_id": 0,
            "admitted_dispatch": "none", "_attempt": 1}
    workers.PENDING.append(task)
    sent = []
    _idle_worker(workers, sent)
    # Real Pause needs queue membership while assignment prepares this row.
    def preparation(candidate):
        assert request_owner_pause("root", request_id="worker-handoff")["ok"]
        return ""
    monkeypatch.setattr(workers, "_evolution_assignment_error", preparation)
    workers.assign_tasks()
    assert sent == [] and workers.PENDING == [task]
    assert task["admitted_dispatch"] == "none"


def test_mcp_initialization_pause_fences_call_across_async_runner(tmp_path, monkeypatch):
    from ouroboros import mcp_client
    from ouroboros.tools.registry import ToolRegistry
    from ouroboros.task_results import load_task_result
    from supervisor.owner_pause_control import request_owner_pause
    _, _, workers = _install_queue(tmp_path, monkeypatch)
    _running(tmp_path, workers)
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.task_id = registry._ctx.root_task_id = "root"
    sent = []
    initialized = []
    monkeypatch.setattr(registry, "_mcp_name_miss", lambda _name: None)
    @asynccontextmanager
    async def transport(_cfg):
        yield (None, None)
    class Session:
        def __init__(self, *_a):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *_a):
            pass
        async def initialize(self):
            initialized.append(True)
            assert request_owner_pause("root", request_id="mcp-final")["ok"]
        async def call_tool(self, *_a):
            sent.append("effect")
            return SimpleNamespace(content=[], isError=False)
    monkeypatch.setattr(mcp_client, "_MCP_SDK_AVAILABLE", True)
    monkeypatch.setattr(mcp_client, "_transport_factory", transport)
    monkeypatch.setattr(mcp_client, "ClientSession", Session)
    monkeypatch.setattr(mcp_client, "_call_mcp_tool_result", lambda *_a: mcp_client._run_async(
        lambda: mcp_client._call_tool_async(None, "effect", {}, timeout_sec=10)))
    async def in_event_loop():
        return registry.execute_result("mcp_demo__effect", {})
    result = asyncio.run(in_event_loop())
    assert initialized == [True] and not sent, result
    assert not load_task_result(tmp_path, "root").get("launch_handoffs")


def test_business_metadata_does_not_outlive_a_joined_builtin(tmp_path, monkeypatch):
    from ouroboros.tools.registry import ToolRegistry
    from ouroboros.tools.tool_result import ToolResult
    from ouroboros.task_results import load_task_result
    _, _, workers = _install_queue(tmp_path, monkeypatch)
    _running(tmp_path, workers)
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.task_id = registry._ctx.root_task_id = "root"
    for dynamic in (False, True):
        registry.override_handler("list_available_tools", lambda *_a, **_kw: ToolResult(
            status="ok", code="LEGACY_WARNING", text="empty read",
            meta={"operation_outcome": "completed_no_effect", "dynamic_provider": dynamic}))
        result = registry.execute_result("list_available_tools", {})
        assert result.meta['dynamic_provider'] is dynamic
        assert result.meta['operation_outcome'] == 'completed_no_effect'
        assert not load_task_result(tmp_path, "root").get("launch_handoffs")


@pytest.mark.parametrize("veto", ["unknown_dispatch", "stop", "foreign_hold", "snapshot", "fence_write"])
@pytest.mark.parametrize("app_stopped", [False, True])
def test_unstarted_combined_resume_retains_other_vetoes(tmp_path, monkeypatch, veto, app_stopped):
    from ouroboros import owner_pause
    from ouroboros.task_results import write_task_result
    from supervisor.owner_pause_control import request_owner_pause
    from supervisor.events_budget import HOLD_SAVED_WORK, budget_hold_fact, hold_budget_row
    q, _, workers = _install_queue(tmp_path, monkeypatch)
    _pool_events(workers, monkeypatch)
    write_task_result(tmp_path, "root", "scheduled", root_task_id="root")
    task = {"id": "root", "root_task_id": "root", "type": "task", "chat_id": 0,
            "admitted_dispatch": "none", "_attempt": 1, "_admission_owner_token": "original"}
    workers.PENDING.append(task)
    assert request_owner_pause("root", request_id="pause")["ok"]
    _restart_door(tmp_path, monkeypatch, workers)
    if app_stopped:
        hold_budget_row(task, reason=HOLD_SAVED_WORK)
    if veto == "unknown_dispatch":
        task.pop("admitted_dispatch")
    elif veto == "stop":
        from ouroboros.cancel_intents import request_cancel
        request_cancel(tmp_path, "root", reason="stop", source="owner", requested_by="owner")
    elif veto == "foreign_hold":
        hold_budget_row(task, reason="opaque_control_hold")
    elif veto == "snapshot":
        monkeypatch.setattr(q, "persist_queue_snapshot", lambda **_k: False)
    elif veto == "fence_write":
        monkeypatch.setattr(owner_pause, "set_fence_state", lambda *_a, **_k: (_ for _ in ()).throw(OSError("disk")))
    before = copy.deepcopy(task)
    outcome = q.resume_budget_paused_task("root")
    assert not outcome["ok"], outcome
    assert task == before and task["_admission_owner_token"] == "original"
    assert owner_pause.read_fence(tmp_path, "root")["state"] != "released"
    # Owner 2026-10-08 (quiz d2f7532b): the owner's Restart names its never-started row
    # for return instead of holding it; the Pause's own fence and queue latch still veto.
    assert q.BUDGET_ROOT_FENCES["root"] and bool(budget_hold_fact(task)) is (app_stopped or veto == "foreign_hold")


def test_cold_request_exempts_only_its_own_invocation(tmp_path, monkeypatch):
    from ouroboros.tools.registry import ToolRegistry
    from ouroboros.task_results import load_task_result, write_task_result
    _, _, workers = _install_queue(tmp_path, monkeypatch)
    _running(tmp_path, workers)
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.task_id = registry._ctx.root_task_id = "root"
    registry._ctx.task_attempt = 1
    registry._ctx.owner_wait_callback = lambda *_a: "unknown"
    result = registry.execute_result("await_messages", {"mode": "cold", "wake_after_sec": 3600})
    assert "sleep_armed" in result.text, result
    assert not load_task_result(tmp_path, "root").get("launch_handoffs")
    write_task_result(tmp_path, "root", "running", launch_handoffs={"different-op": {
        "tool": "await_messages", "state": "claimed"}})
    second = registry.execute_result("await_messages", {"mode": "cold", "wake_after_sec": 3600})
    assert "member_custody" in second.text and "sleep_armed" not in second.text


def test_cold_checkpoint_closes_child_start_before_lane_release(tmp_path, monkeypatch):
    from ouroboros import budget_pause, model_sleep, owner_wait
    from ouroboros.task_results import write_task_result
    from ouroboros.tools.registry import ToolRegistry
    from supervisor.events_budget import install_exact_budget_pause
    from ouroboros.project_lease import running_project_ids
    q, _, workers = _install_queue(tmp_path, monkeypatch)
    task = _running(tmp_path, workers)
    task["project_id"] = "project"
    write_task_result(tmp_path, "child", "scheduled", root_task_id="root", parent_task_id="root")
    ctx, limit = _loop_ctx(tmp_path, "root")
    monkeypatch.setattr(budget_pause, "_HOLD_POLL_SEC", 0.001)
    monkeypatch.setattr(budget_pause, "_hold_control_reason", lambda _ctx: "cancelled" if
        limit.accumulated_usage.get("budget_pause_hold") else "")
    ctx._model_sleep = {"sleep_id": "sleep", "mode": "cold", **model_sleep.selectors(ctx, wake_after_sec=3600)}
    child = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    child._ctx.task_id, child._ctx.root_task_id = "child", "root"
    effects = []
    child.override_handler("knowledge_read", lambda *_a, **_kw: effects.append("write") or "ok")
    store = owner_wait.store_continuation_source
    def at_checkpoint(*args, **kwargs):
        assert "project" in running_project_ids(workers.RUNNING.values())
        outcome = child.execute_result("knowledge_read", {"topic": "x"})
        assert outcome.code == "OWNER_PAUSE_NOT_STARTED", outcome
        return store(*args, **kwargs)
    monkeypatch.setattr(owner_wait, "store_continuation_source", at_checkpoint)
    with pytest.raises(budget_pause.BudgetPauseRequested) as caught:
        budget_pause.enter_cold_sleep(limit)
    budget_pause.end_dispatch_fence("root")
    install_exact_budget_pause(workers, "root", budget_pause.exact_pause_marker(caught.value.pause)["checkpoint"])
    assert effects == [] and "root" not in workers.RUNNING
    assert "project" not in running_project_ids(workers.RUNNING.values())
    assert child.execute_result("knowledge_read", {"topic": "later"}).code == "OWNER_PAUSE_NOT_STARTED"


@pytest.mark.parametrize("direct", [False, True])
def test_warm_restart_restores_after_failed_conversion(tmp_path, monkeypatch, direct):
    from ouroboros import owner_wait, model_sleep, budget_pause, server_restart
    from ouroboros.task_results import load_task_result
    from supervisor import active_activity
    from supervisor.events_budget import budget_hold_fact
    q, _, workers = _install_queue(tmp_path, monkeypatch)
    _pool_events(workers, monkeypatch)
    _running(tmp_path, workers)
    ctx, limit = _loop_ctx(tmp_path, "root", direct=direct)
    model_sleep.request_sleep(ctx, model_sleep.selectors(ctx, wake_after_sec=3600), "warm")
    checkpoint = owner_wait.checkpoint_owner_wait(ctx, limit.messages, {}, {}, 1, [], set())
    owner_wait.set_owner_wait(tmp_path, "root", {**checkpoint, "state": "waiting"})
    if direct:
        workers.RUNNING.clear()
        monkeypatch.setattr(active_activity, "get_direct_activity_registry", lambda: SimpleNamespace(
            snapshot=lambda: [{"activity_id": "root"}]))
        monkeypatch.setattr(q, "PRIOR_DIRECT_ROOTS", {"task_ids": ["root"]})
    monkeypatch.setattr(server_restart, "DATA_DIR", tmp_path)
    assert "root" not in server_restart._owned_live_task_ids(SimpleNamespace(RUNNING=workers.RUNNING))
    with monkeypatch.context() as failing:
        failing.setattr(budget_pause, "set_budget_pause", lambda *_a, **_kw: (_ for _ in ()).throw(OSError("disk")))
        _restart_door(tmp_path, monkeypatch, workers)
    if direct:
        monkeypatch.setattr(active_activity, "get_direct_activity_registry", lambda: SimpleNamespace(snapshot=lambda: []))
    workers.PENDING.clear()
    assert q.restore_pending_from_snapshot() == 1
    task = workers.PENDING[0]
    assert task["id"] == "root" and budget_hold_fact(task)
    assert load_task_result(tmp_path, "root")["status"] != "cancelled"
    assert q.resume_budget_paused_task("root")["ok"]
    ctx.budget_pause_resume = task["_budget_pause_resume"]
    saved = budget_pause.load_budget_pause(ctx)
    assert saved["messages"] == limit.messages and saved["sleep"] == checkpoint["sleep"]
    monkeypatch.setattr(owner_wait, "rebind_restored_route", lambda *_a, **_kw: (None, "max"))
    messages = []
    budget_pause.resume_paused_loop(SimpleNamespace(_ctx=ctx), saved, messages, {}, {}, set(),
                                    budget_remaining_usd=5)
    assert budget_pause.budget_pause_row(tmp_path, "root")["state"] == budget_pause.STATE_RESUMED
    assert load_task_result(tmp_path, "root")["owner_wait"]["state"] == "retained"
    assert messages[:len(limit.messages)] == limit.messages


@pytest.mark.parametrize("invalid", ["source", "attempt", "quiz", "review", "stop"])
def test_warm_retention_requires_current_sleep_proof(tmp_path, monkeypatch, invalid):
    from ouroboros import owner_wait, model_sleep
    from supervisor.restart_retention import pause_retention
    _, _, workers = _install_queue(tmp_path, monkeypatch)
    _running(tmp_path, workers)
    ctx, limit = _loop_ctx(tmp_path, "root")
    model_sleep.request_sleep(ctx, model_sleep.selectors(ctx, wake_after_sec=3600), "warm")
    row = owner_wait.checkpoint_owner_wait(ctx, limit.messages, {}, {}, 1, [], set())
    if invalid == "source":
        row["source_ref"] = {**row["source_ref"], "sha256": "0" * 64}
    elif invalid == "attempt":
        row["task_attempt"] = 2
    elif invalid in {"quiz", "review"}:
        row["reason"] = "owner" if invalid == "quiz" else "review"
    elif invalid == "stop":
        from ouroboros.cancel_intents import request_cancel
        request_cancel(tmp_path, "root", reason="stop", source="owner", requested_by="owner")
    owner_wait.set_owner_wait(tmp_path, "root", {**row, "state": "waiting"})
    assert not pause_retention(tmp_path, "root", 1)


def test_resume_crash_before_durable_commit_restores_usable_fence(tmp_path, monkeypatch):
    from ouroboros import owner_pause
    from ouroboros.task_results import write_task_result
    from supervisor.owner_pause_control import request_owner_pause
    q, _, workers = _install_queue(tmp_path, monkeypatch)
    _pool_events(workers, monkeypatch)
    write_task_result(tmp_path, "root", "scheduled", root_task_id="root")
    workers.PENDING.append({"id": "root", "root_task_id": "root", "type": "task", "chat_id": 0,
                            "admitted_dispatch": "none", "_attempt": 1, "_admission_owner_token": "original"})
    assert request_owner_pause("root", request_id="pause")["ok"]
    _restart_door(tmp_path, monkeypatch, workers)
    with monkeypatch.context() as crash:
        crash.setattr(owner_pause, "set_fence_state", lambda *_a, **_kw: (_ for _ in ()).throw(SystemExit("crash")))
        with pytest.raises(SystemExit):
            q.resume_budget_paused_task("root")
    workers.PENDING.clear()
    q.BUDGET_ROOT_FENCES.clear()
    assert q.restore_pending_from_snapshot() == 1
    assert "root" in q.BUDGET_ROOT_FENCES
    from supervisor.events_budget import HOLD_SAVED_WORK, budget_hold_fact

    assert budget_hold_fact(workers.PENDING[0])["reason"] == HOLD_SAVED_WORK
    resumed = q.resume_budget_paused_task("root")
    assert resumed["ok"], resumed
    assert workers.PENDING[0]["_admission_owner_token"] == "original"


def test_cold_rechecks_a_child_claim_that_lands_after_initial_census(tmp_path, monkeypatch):
    from ouroboros import budget_pause, model_sleep
    from ouroboros.task_results import write_task_result
    from ouroboros.model_wait import ModelWaitInterrupted
    _, _, workers = _install_queue(tmp_path, monkeypatch)
    _running(tmp_path, workers)
    write_task_result(tmp_path, "child", "scheduled", root_task_id="root")
    ctx, limit = _loop_ctx(tmp_path, "root")
    ctx._model_sleep = {"sleep_id": "s", "mode": "cold", **model_sleep.selectors(ctx, wake_after_sec=3600)}
    original = model_sleep.cold_blockers
    observations = []
    def initial_census(source):
        blockers = original(source)
        observations.append(blockers)
        if len(observations) == 1:
            write_task_result(tmp_path, "child", "scheduled", launch_handoffs={"external": {"tool": "opaque"}})
        return blockers
    monkeypatch.setattr(model_sleep, "cold_blockers", initial_census)
    monkeypatch.setattr(budget_pause, "_HOLD_POLL_SEC", .001)
    monkeypatch.setattr(budget_pause, "_hold_control_reason", lambda _ctx: "cancelled" if
        limit.accumulated_usage.get("budget_pause_hold") else "")
    with pytest.raises(ModelWaitInterrupted):
        budget_pause.enter_cold_sleep(limit)
    budget_pause.end_dispatch_fence("root")
    assert len(observations) == 2 and observations[0] == []
    assert observations[1] == [{"kind": "member_custody", "detail": "child"}]
    assert "root" in workers.RUNNING
    assert not budget_pause.budget_pause_row(tmp_path, "root").get("source_ref")


from tests.test_llm_claudexor import setup as model_setup  # noqa: F401 - pytest fixture


@pytest.mark.parametrize("prior_unknown", [False, True])
def test_handed_model_session_finishes_after_pause_with_same_operation(model_setup, monkeypatch, prior_unknown):  # noqa: F811 - pytest fixture
    from tests.test_llm_claudexor import MODEL, ledger
    from supervisor.owner_pause_control import request_owner_pause
    root, gateway, client = model_setup
    _, _, workers = _install_queue(root, monkeypatch)
    _running(root, workers, "task-one")
    gateway.lose_create = prior_unknown
    observations = []
    def checkpoint(value):
        observations.append(value)
        if len(observations) == (2 if prior_unknown else 1):
            assert request_owner_pause("task-one", request_id="model-final")["ok"]
    answer, _usage = client.chat([], MODEL, model_operation_observer=checkpoint)
    assert answer["content"] == "Ответ 🐍"
    assert len(gateway.creates) == (2 if prior_unknown else 1)
    assert len(gateway.accepted_operations) == 1, "lost create reply rejoins the exact original operation"
    assert len(gateway.uploads) == 1 and not gateway.cancels
    assert ledger(root)[-1]["state"] == "settled"


def test_delegated_start_pause_after_claim_prevents_post(tmp_path, monkeypatch):
    from ouroboros import claudexor_daemon, delegate_custody
    from ouroboros.tools import delegate
    from tests._delegated_transport_shared import _nanny_ctx, _LiveRunStub, _transport_snapshot
    from ouroboros import subagent_runtime, subagents
    from supervisor.owner_pause_control import request_owner_pause
    _, _, workers = _install_queue(tmp_path, monkeypatch)
    _running(tmp_path, workers)
    gateway = _LiveRunStub()
    starts = []
    monkeypatch.setattr(gateway, "start_run", lambda *_a, **_kw: starts.append(True) or {"runId": "unwanted"})
    monkeypatch.setattr(claudexor_daemon, "ensure_owned_gateway", lambda: gateway)
    monkeypatch.setenv("OUROBOROS_SUBAGENT_HARNESS", "some-route=weak-model:low")
    original = delegate.claimed_start_request
    def claim(*args, **kwargs):
        value = original(*args, **kwargs)
        assert value[0]
        assert request_owner_pause("root", request_id="delegate-final")["ok"]
        return value
    monkeypatch.setattr(delegate, "claimed_start_request", claim)
    token = subagent_runtime._EXACT_START_SELECTION.set({"snapshot": _transport_snapshot(subagents.get_subagent_harness())})
    try:
        outcome = delegate._delegate_start(_nanny_ctx(tmp_path, "root"), "work")
    finally:
        subagent_runtime._EXACT_START_SELECTION.reset(token)
    assert not starts and '"reason": "owner_pause"' in outcome.text
    assert not delegate_custody.pending_invocations(tmp_path)


def test_review_session_pause_after_checkpoint_prevents_post(tmp_path, monkeypatch):
    from ouroboros import claudexor_daemon, review_execution, delegate_custody
    from tests._review_session_route_shared import FakeGateway, _run_session_directly
    from supervisor.owner_pause_control import request_owner_pause
    _, _, workers = _install_queue(tmp_path, monkeypatch)
    _running(tmp_path, workers, "t-b")
    FakeGateway.reset()
    gateway = FakeGateway()
    monkeypatch.setattr(claudexor_daemon, "ensure_owned_gateway", lambda: gateway)
    monkeypatch.setenv(review_execution.REVIEW_SESSION_ROUTE_ENV, "fake-review=fake-small:low")
    def checkpoint(*_a, **_kw):
        assert request_owner_pause("t-b", request_id="review-final")["ok"]
    monkeypatch.setattr(review_execution, "checkpoint_pending_invocation", checkpoint)
    with pytest.raises(review_execution.ReviewRouteUnavailable, match="fenced"):
        _run_session_directly(tmp_path, root=str(tmp_path))
    assert not gateway.start_requests
    assert not delegate_custody.pending_invocations(tmp_path)


def test_timed_out_mcp_runner_cannot_start_after_caller_return(tmp_path, monkeypatch):
    import threading
    from ouroboros import mcp_client
    from ouroboros.tools.registry import ToolRegistry
    from ouroboros.tools.tool_result import ToolResult
    from ouroboros.task_results import load_task_result
    _, _, workers = _install_queue(tmp_path, monkeypatch)
    _running(tmp_path, workers)
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.task_id = registry._ctx.root_task_id = "root"
    monkeypatch.setattr(registry, "_mcp_name_miss", lambda _name: None)
    release, entered = threading.Event(), threading.Event()
    threads, sent = [], []
    @asynccontextmanager
    async def transport(_cfg):
        yield (None, None)
    class Session:
        def __init__(self, *_a):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *_a):
            pass
        async def initialize(self):
            threads.append(threading.current_thread())
            entered.set()
            assert release.wait(5)
        async def call_tool(self, *_a):
            sent.append("effect")
            return SimpleNamespace(content=[], isError=False)
    monkeypatch.setattr(mcp_client, "_MCP_SDK_AVAILABLE", True)
    monkeypatch.setattr(mcp_client, "_transport_factory", transport)
    monkeypatch.setattr(mcp_client, "ClientSession", Session)
    def timed_call(*_a):
        try:
            return mcp_client._run_async(
                lambda: mcp_client._call_tool_async(None, "effect", {}, timeout_sec=10), join_timeout=0)
        except TimeoutError:
            return ToolResult(status="timeout", code="MCP_TIMEOUT", text="host wait elapsed",
                              meta={"dynamic_provider": True})
    monkeypatch.setattr(mcp_client, "_call_mcp_tool_result", timed_call)
    async def execute():
        return registry.execute_result("mcp_demo__effect", {})
    try:
        result = asyncio.run(execute())
        assert result.status == "timeout" and entered.wait(5)
        assert not load_task_result(tmp_path, "root").get("launch_handoffs")
    finally:
        release.set()
        assert entered.wait(5)
        for thread in threads:
            thread.join(5)
            assert not thread.is_alive()
    assert sent == []


def test_extension_staging_pause_prevents_real_spawn_boundary(tmp_path, monkeypatch):
    from ouroboros import extension_process_runner as runner, owner_pause
    from ouroboros.task_results import load_task_result
    from supervisor.owner_pause_control import request_owner_pause
    _, _, workers = _install_queue(tmp_path, monkeypatch)
    _running(tmp_path, workers)
    source = SimpleNamespace(drive_root=tmp_path, task_id="root", root_task_id="root")
    staged, spawned = [], []
    original = runner._write_private_json
    def stage(*args, **kwargs):
        original(*args, **kwargs)
        staged.append(args[0])
        assert request_owner_pause("root", request_id="extension-final")["ok"]
    monkeypatch.setattr(runner, "_write_private_json", stage)
    def spawn(*_a, **_kw):
        spawned.append(True)
        raise AssertionError("Popen reached after Pause")
    monkeypatch.setattr(runner.subprocess, "Popen", spawn)
    with pytest.raises(owner_pause.OwnerPauseRefused):
        with owner_pause.tool_handoff(source, "extension_effect"):
            with runner._child_process({"skill_name": "test"}, skill_dir=tmp_path,
                    drive_root=tmp_path, repo_dir=tmp_path, env={}):
                pytest.fail("child body reached")
    assert staged and not spawned and not staged[0].exists()
    assert not load_task_result(tmp_path, "root").get("launch_handoffs")
