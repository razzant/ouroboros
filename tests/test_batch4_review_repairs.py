"""Batch4 exact-range review F1-F4, driven through their real consumers.

F1: a fresh owner Pause supersedes a Resume grant already handed to the worker.
F2: a definite pre-effect refusal settles its own registry launch claim.
F3: a stale settlement census never re-pauses a root that Resume released.
F4: MCP transport entry (a stdio server start) waits for launch admission.
"""
import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from tests._budget_pause_exact_helpers import _install_queue, _loop_ctx
from tests.test_batch4_full_repair import _park
from tests.test_batch4_producer_custody import _registry
from tests.test_batch4_repair_compositions import _running
from tests.test_owner_pause import _owner_park
from tests.worker_pool_support import affirm_ready

pytestmark = pytest.mark.serial


def _granted_and_handed(tmp_path, monkeypatch):
    """Pause A parked the root; the owner's Resume grant was handed to a worker."""
    from ouroboros import budget_pause
    from ouroboros.task_results import STATUS_RUNNING, write_task_result
    from supervisor.events_budget import install_exact_budget_pause
    from supervisor.owner_pause_control import request_owner_pause

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_k: 5.0)
    write_task_result(tmp_path, "solo", STATUS_RUNNING, chat_id=0)
    workers.RUNNING["solo"] = {"task": {"id": "solo", "type": "task", "chat_id": 0, "root_task_id": "solo"},
                               "worker_id": 0, "attempt": 1}
    first = request_owner_pause("solo", request_id="pause-a")
    row = _owner_park(tmp_path, monkeypatch, "solo", [], root="solo")
    ctx = SimpleNamespace(DRIVE_ROOT=tmp_path, RUNNING=workers.RUNNING, PENDING=workers.PENDING,
                          WORKERS=workers.WORKERS, sort_pending=lambda: None,
                          persist_queue_snapshot=queue.persist_queue_snapshot, bridge=None)
    install_exact_budget_pause(ctx, "solo", budget_pause.exact_pause_marker(row)["checkpoint"])
    assert queue.resume_budget_paused_task("solo")["ok"] is True
    sent = []
    workers.WORKERS[0] = affirm_ready(SimpleNamespace(wid=0, busy_task_id=None, reaping=False,
                                         in_q=SimpleNamespace(put=lambda task: sent.append(dict(task)))))
    workers.assign_tasks()
    assert [task["id"] for task in sent] == ["solo"]
    loop_ctx, limit_ctx = _loop_ctx(tmp_path, "solo")
    loop_ctx.budget_pause_resume = sent[0]["_budget_pause_resume"]
    monkeypatch.setattr("ouroboros.owner_wait.rebind_restored_route", lambda *_a, **_k: (None, "max"))
    monkeypatch.setattr("ouroboros.owner_wait.restore_continuation_state", lambda *_a, **_k: None)
    return queue, first, loop_ctx, limit_ctx


@pytest.mark.parametrize("window", ["handed", "consumed"])
def test_a_fresh_pause_supersedes_a_handed_resume_grant(tmp_path, monkeypatch, window):
    """Pause B lands after grant G reached the worker (before, or just after, the
    worker consumed G): G must not release B. Same-request replay stays B."""
    from ouroboros import budget_pause, owner_pause
    from supervisor.owner_pause_control import request_owner_pause

    queue, first, loop_ctx, limit_ctx = _granted_and_handed(tmp_path, monkeypatch)
    acks = []
    if window == "handed":
        acks.append(request_owner_pause("solo", request_id="pause-b"))
    else:
        real = budget_pause._reopen_owner_fence

        def pause_between_consumption_and_release(*args, **kwargs):
            assert budget_pause.budget_pause_row(tmp_path, "solo")["state"] == budget_pause.STATE_RESUMED
            acks.append(request_owner_pause("solo", request_id="pause-b"))
            return real(*args, **kwargs)

        monkeypatch.setattr(budget_pause, "_reopen_owner_fence", pause_between_consumption_and_release)
    blob = budget_pause.load_budget_pause(loop_ctx)
    budget_pause.resume_paused_loop(limit_ctx.tools, blob, list(limit_ctx.messages), {}, {}, set(),
                                    budget_remaining_usd=5.0)

    fence = owner_pause.read_fence(tmp_path, "solo")
    assert fence["state"] != owner_pause.FENCE_RELEASED, "an earlier Resume released the newer Pause"
    assert owner_pause.fence_closed(fence) and owner_pause.member_fence(loop_ctx), "new handoffs stay fenced"
    ack = acks[0]
    assert ack["ok"] is True and ack["duplicate"] is False
    assert ack["fence_id"] == fence["fence_id"] != first["fence_id"]
    assert queue.BUDGET_ROOT_FENCES["solo"]["fence_id"] == fence["fence_id"]
    again = request_owner_pause("solo", request_id="pause-b")
    assert again["duplicate"] is True and again["fence_id"] == fence["fence_id"]
    # The worker meets Pause B at its next safe boundary and parks under it.
    parked = _owner_park(tmp_path, monkeypatch, "solo", [], root="solo")
    assert parked["owner_fence_id"] == fence["fence_id"]


def test_a_pause_before_dispatch_returns_the_grant_and_a_later_resume_releases_only_its_own_fence(
        tmp_path, monkeypatch):
    """Pause B lands after the Resume grant but before assignment; B's own Resume
    then runs the ordinary grant/consume path and reopens B's fence."""
    from ouroboros import budget_pause, owner_pause
    from ouroboros.task_results import STATUS_RUNNING, write_task_result
    from supervisor.events_budget import install_exact_budget_pause
    from supervisor.owner_pause_control import request_owner_pause

    queue, state, workers = _install_queue(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda _st, **_k: 5.0)
    write_task_result(tmp_path, "solo", STATUS_RUNNING, chat_id=0)
    workers.RUNNING["solo"] = {"task": {"id": "solo", "type": "task", "chat_id": 0, "root_task_id": "solo"},
                               "worker_id": 0, "attempt": 1}
    ctx = SimpleNamespace(DRIVE_ROOT=tmp_path, RUNNING=workers.RUNNING, PENDING=workers.PENDING,
                          WORKERS=workers.WORKERS, sort_pending=lambda: None,
                          persist_queue_snapshot=queue.persist_queue_snapshot, bridge=None)
    first = request_owner_pause("solo", request_id="pause-a")
    row = _owner_park(tmp_path, monkeypatch, "solo", [], root="solo")
    install_exact_budget_pause(ctx, "solo", budget_pause.exact_pause_marker(row)["checkpoint"])
    assert queue.resume_budget_paused_task("solo")["ok"] is True
    second = request_owner_pause("solo", request_id="pause-b")
    assert second["ok"] is True and second["fence_id"] != first["fence_id"]
    sent = []
    workers.WORKERS[0] = affirm_ready(SimpleNamespace(wid=0, busy_task_id=None, reaping=False,
                                         in_q=SimpleNamespace(put=lambda task: sent.append(dict(task)))))
    workers.assign_tasks()
    assert sent == [], "the superseded grant is never dispatched"
    assert owner_pause.read_fence(tmp_path, "solo")["fence_id"] == second["fence_id"]
    assert budget_pause.budget_pause_row(tmp_path, "solo")["grant"]["revoked_at"]

    assert queue.resume_budget_paused_task("solo")["ok"] is True
    workers.assign_tasks()
    assert [task["id"] for task in sent] == ["solo"]
    handoff = sent[0]["_budget_pause_resume"]
    assert budget_pause.budget_pause_row(tmp_path, "solo")["grant"]["owner_pause_fence_id"] == second["fence_id"]
    loop_ctx, limit_ctx = _loop_ctx(tmp_path, "solo")
    loop_ctx.budget_pause_resume = handoff
    monkeypatch.setattr("ouroboros.owner_wait.rebind_restored_route", lambda *_a, **_k: (None, "max"))
    monkeypatch.setattr("ouroboros.owner_wait.restore_continuation_state", lambda *_a, **_k: None)
    budget_pause.resume_paused_loop(limit_ctx.tools, budget_pause.load_budget_pause(loop_ctx),
                                    list(limit_ctx.messages), {}, {}, set(), budget_remaining_usd=5.0)
    fence = owner_pause.read_fence(tmp_path, "solo")
    assert fence["fence_id"] == second["fence_id"] and fence["state"] == owner_pause.FENCE_RELEASED
    assert owner_pause.member_fence(loop_ctx) == {}


def test_a_stale_settlement_census_cannot_repause_a_resumed_queued_root(tmp_path, monkeypatch):
    """Resume of a queued root lands between refresh's census and its publication."""
    from ouroboros import owner_pause
    from ouroboros.task_results import STATUS_SCHEDULED, write_task_result
    from supervisor import continuation_admission
    from supervisor.owner_pause_control import refresh_owner_pause_tree, request_owner_pause
    from supervisor.queue_transitions import budget_pause_fact

    queue, _state, workers = _install_queue(tmp_path, monkeypatch)
    write_task_result(tmp_path, "queued-root", STATUS_SCHEDULED, chat_id=0)
    workers.PENDING.append({"id": "queued-root", "type": "task", "chat_id": 0, "_attempt": 1,
                            "root_task_id": "queued-root", "admitted_dispatch": "none"})
    real = continuation_admission.conflicting_writers
    armed, resumed = [True], []

    def census_then_resume(*args, **kwargs):
        writers = real(*args, **kwargs)
        if armed and armed.pop():
            resumed.append(queue.resume_budget_paused_task("queued-root"))
        return writers

    monkeypatch.setattr(continuation_admission, "conflicting_writers", census_then_resume)
    ack = request_owner_pause("queued-root", request_id="p")

    assert ack["ok"] is True and resumed and resumed[0]["owner_pause_released"] is True, resumed
    fence = owner_pause.read_fence(tmp_path, "queued-root")
    assert fence["state"] == owner_pause.FENCE_RELEASED, "a stale census revived released authority"
    assert fence["release_reason"] == "owner_resume_unstarted_root"
    assert ack["state"] == owner_pause.FENCE_RELEASED
    assert "queued-root" not in queue.BUDGET_ROOT_FENCES and budget_pause_fact(workers.PENDING[0]) is None
    assert refresh_owner_pause_tree("queued-root") == owner_pause.FENCE_RELEASED
    events = (tmp_path / "logs" / "events.jsonl").read_text()
    assert '"owner_pause_settled"' not in events


@pytest.mark.parametrize("tool,args", [
    ("github_read", {"method": "get", "kind": "issue", "target": 0}),
    ("github_write", {"method": "comment", "target": 3, "body": " "}),
    ("github_write", {"method": "update", "target": -1, "state": "closed"}),
    ("github_read", {"method": "get", "kind": "pr", "target": 0}),
    ("github_write", {"method": "review", "target": 0, "body": "x"}),
    ("github_write", {"method": "create_issue", "title": " "}),
    ("start_service", {"cmd": ["sleep", "1"], "name": "bad name!"}),
    ("start_service", {"cmd": ["sleep", "1"], "readiness": {"timeout_sec": -1}}),
])
def test_definite_pre_effect_refusal_releases_its_launch_claim(tmp_path, monkeypatch, tool, args):
    from ouroboros.task_results import load_task_result
    from ouroboros.tools import github
    from supervisor import state

    registry, queue, workers = _registry(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_kw: 5.0)
    monkeypatch.setenv("GITHUB_TOKEN", "test-token")  # the credential gate precedes the handler
    launched = []
    monkeypatch.setattr(github, "_gh_run", lambda *a, **kw: launched.append(a))
    result = registry.execute_result(tool, args)
    assert not launched and result.status != "ok", result
    assert result.meta.get("operation_outcome") == "completed_no_effect", result
    assert not load_task_result(tmp_path, "root").get("launch_handoffs")
    _park(tmp_path, monkeypatch, queue, workers)
    assert queue.resume_budget_paused_task("root")["ok"] is True


def test_returned_github_parse_error_does_not_keep_local_execution_alive(tmp_path, monkeypatch):
    from ouroboros.task_results import load_task_result
    from ouroboros.tools import github
    from supervisor import state

    registry, queue, workers = _registry(tmp_path, monkeypatch)
    monkeypatch.setattr(state, "budget_remaining", lambda *_a, **_kw: 5.0)
    monkeypatch.setenv("GITHUB_TOKEN", "test-token")
    monkeypatch.setattr(github, "_gh_run", lambda *_a, **_kw: github.GhResult(True, "{truncated", 0, None, ""))
    result = registry.execute_result("github_read", {"method": "get", "kind": "issue", "target": 5})
    assert "failed to parse issue JSON" in result.text
    assert result.meta.get("operation_outcome") != "completed_no_effect"
    assert not load_task_result(tmp_path, "root").get("launch_handoffs", {})
    _park(tmp_path, monkeypatch, queue, workers)
    assert queue.resume_budget_paused_task("root")["ok"]


def test_pause_during_mcp_safety_preparation_starts_no_transport(tmp_path, monkeypatch):
    """The stdio transport's entry starts the server process: it is gated before."""
    from ouroboros import mcp_client, safety
    from ouroboros.task_results import load_task_result
    from ouroboros.tools.registry import ToolRegistry
    from supervisor.owner_pause_control import request_owner_pause

    _, _, workers = _install_queue(tmp_path, monkeypatch)
    _running(tmp_path, workers)
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.task_id = registry._ctx.root_task_id = "root"
    events, pause_during_safety = [], []
    monkeypatch.setattr(registry, "_mcp_name_miss", lambda _name: None)

    def safety_check(*_a, **_kw):
        if pause_during_safety:
            assert request_owner_pause("root", request_id="mcp-safety")["ok"]
        return True, ""

    @asynccontextmanager
    async def transport(_cfg):
        events.append("transport_started")
        yield (None, None)

    class Session:
        def __init__(self, *_a):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_a):
            pass

        async def initialize(self):
            events.append("initialize")

        async def call_tool(self, *_a):
            events.append("call_tool")
            return SimpleNamespace(content=[], isError=False)

    monkeypatch.setattr(safety, "check_safety", safety_check)
    monkeypatch.setattr(mcp_client, "_MCP_SDK_AVAILABLE", True)
    monkeypatch.setattr(mcp_client, "_transport_factory", transport)
    monkeypatch.setattr(mcp_client, "ClientSession", Session)
    monkeypatch.setattr(mcp_client, "_call_mcp_tool_result", lambda *_a: mcp_client._run_async(
        lambda: mcp_client._call_tool_async(None, "effect", {}, timeout_sec=10)))

    async def in_event_loop():
        return registry.execute_result("mcp_demo__effect", {})

    admitted = asyncio.run(in_event_loop())
    assert events == ["transport_started", "initialize", "call_tool"], admitted
    claims = load_task_result(tmp_path, "root").get("launch_handoffs")
    assert claims, "an opaque MCP result keeps its custody"
    events.clear()
    pause_during_safety.append(True)
    refused = asyncio.run(in_event_loop())
    assert events == [], refused
    assert load_task_result(tmp_path, "root").get("launch_handoffs") == claims
