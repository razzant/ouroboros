"""Physical Emergency Stop requests bypass real owner locks and durable I/O.

All children belong to this isolated test process. Receipts attest requests;
separate bounded waits below establish each real child's death.
"""
from __future__ import annotations

import concurrent.futures
import json
import sys
import threading
import time

import pytest

from ouroboros import extension_companion as companions
from ouroboros import workspace_executor as executor
from ouroboros.platform_layer import request_process_tree_kill
from ouroboros.tools import services, shell_process
from ouroboros.tools.registry import ToolContext, ToolRegistry

pytestmark = pytest.mark.serial


def _wait_for(read):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        value = read()
        if value:
            return value
        time.sleep(0.01)
    raise AssertionError("owned process was not published")


@pytest.fixture
def owners(monkeypatch, tmp_path):
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "advanced")
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))
    for owner in (services, executor, shell_process):
        monkeypatch.setattr(owner, "_panic_requested", False)
    monkeypatch.setattr(services, "_SERVICES", {})
    monkeypatch.setattr(executor, "_SERVICES", {})
    monkeypatch.setattr(executor, "_FOREGROUND", {})
    monkeypatch.setattr(shell_process, "_active_subprocesses", set())
    supervisor = companions.CompanionSupervisor(tmp_path / "data")
    monkeypatch.setattr(companions, "_SERVER_PROCESS_PID", __import__("os").getpid())
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    context = ToolContext(repo_dir=tmp_path / "system", drive_root=tmp_path / "data",
                          workspace_root=workspace, workspace_mode="external", task_id="panic-child")
    registry = ToolRegistry(repo_dir=context.repo_dir, drive_root=context.drive_root)
    registry.set_context(context)
    return supervisor, context, registry, workspace


@pytest.mark.parametrize("owner", ["shell", "executor_foreground", "service", "executor_service", "companion"])
def test_real_owned_requests_do_not_wait_for_owner_lock(owners, owner, monkeypatch):
    supervisor, ctx, registry, workspace = owners
    cmd = [sys.executable, "-c", "import time; time.sleep(30)"]
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    future = None
    proc = None
    lock = None
    try:
        if owner in {"executor_foreground", "executor_service"}:
            ctx.executor_ref = {"type": "local", "workspace_host_path": str(workspace),
                                "workspace_backend_path": "/workspace"}
        if owner == "shell":
            future = pool.submit(registry.execute, "run_command", {"cmd": cmd})
            proc = _wait_for(lambda: next(iter(shell_process._active_subprocesses.copy()), None))
            lock, request = shell_process._subprocess_lock, shell_process.kill_all_tracked_subprocesses
        elif owner == "executor_foreground":
            future = pool.submit(registry.execute, "run_command", {"cmd": cmd})
            proc = _wait_for(lambda: next(iter(executor._FOREGROUND.copy()), None))
            lock, request = executor._STATE_LOCK, executor.kill_all_foreground
        elif owner in {"service", "executor_service"}:
            result = registry.execute("start_service", {"name": "sleepy", "cmd": cmd,
                                       "readiness": {"timeout_sec": 0}})
            assert json.loads(result)["state"] == "running"
            if owner == "service":
                proc = services._SERVICES["panic-child:sleepy"].proc
                lock = services._LOCK
            else:
                proc = executor._SERVICES["panic-child:sleepy"].local_proc
                lock = executor._STATE_LOCK
            request = services.kill_all_services
        else:
            assert supervisor.start(companions.CompanionDescriptor(
                "panic-test", "sleepy", cmd, workspace, {}, restart_policy="on_failure"))
            proc = supervisor._runtimes["panic-test:sleepy"].process
            lock, request = supervisor._lock, supervisor.panic_kill_all
            monkeypatch.setattr(companions, "kill_process_on_port", lambda *_: pytest.fail("request must not scan ports"))
        lock.acquire()
        try:
            # A different thread is essential for the two RLocks: this tests
            # actual contention, not the current thread's reentrant acquire.
            receipts = pool.submit(request, request_only=True).result(timeout=2)
            assert any(row["pid"] == proc.pid and row["requested"] for row in receipts)
            assert all(row.get("state") != "stopped" for row in receipts)
            assert proc.wait(timeout=2) is not None
        finally:
            lock.release()
        if future is not None:
            future.result(timeout=5)
        if owner in {"service", "executor_service"}:
            stopped = json.loads(registry.execute("stop_service", {"name": "sleepy"}))
            assert stopped["state"] in {"exited", "stopped"}
        if owner == "companion":
            assert not supervisor.start(companions.CompanionDescriptor(
                "panic-test", "late", cmd, workspace, {}))
    finally:
        if proc is not None:
            request_process_tree_kill(proc)
            proc.wait(timeout=5)
        if future is not None:
            future.result(timeout=5)
        pool.shutdown(wait=True)
        # The owner's ordinary settlement remains callable after the physical
        # request and does the same record/log cleanup as before.
        services.kill_all_services(ctx.drive_root)
        supervisor.stop_all(timeout_sec=0.1)


def test_backend_only_request_keeps_unconfirmed_custody(owners, monkeypatch):
    _supervisor, ctx, _registry, workspace = owners
    record = executor._ExecutorService(
        service_id="panic-child:backend", task_id="panic-child", name="backend",
        executor=executor.ExecutorRef("docker_exec", "declared", "none", (), "owned-container"),
        cmd=["sleep", "30"], host_cwd=workspace, backend_cwd="/workspace",
        cwd_root="active_workspace", outputs=[], before_outputs={}, backend_pid="123",
    )
    executor._SERVICES[record.service_id] = record
    monkeypatch.setattr(executor, "_owned_process_records", lambda *_: pytest.fail("request must not read durable records"))
    receipts = services.kill_all_services(ctx.drive_root, request_only=True)
    assert receipts == [{"service_id": record.service_id, "requested": False, "scope": "backend",
                         "error": "backend-only process requires executor settlement"}]
    assert executor._SERVICES[record.service_id] is record


@pytest.mark.parametrize("owner", ["service", "executor_service", "companion"])
def test_spawn_identity_is_published_before_custody_write(owners, owner, monkeypatch):
    supervisor, ctx, registry, workspace = owners
    entered, release = threading.Event(), threading.Event()

    def blocked_record(*args, **kwargs):
        entered.set()
        assert release.wait(timeout=5)

    monkeypatch.setattr("ouroboros.process_custody.record_process", blocked_record)
    cmd = [sys.executable, "-c", "import time; time.sleep(30)"]
    if owner == "executor_service":
        ctx.executor_ref = {"type": "local", "workspace_host_path": str(workspace),
                            "workspace_backend_path": "/workspace"}
    proc = None
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    try:
        if owner == "companion":
            future = pool.submit(supervisor.start, companions.CompanionDescriptor(
                "panic-test", "blocked", cmd, workspace, {}))
        else:
            future = pool.submit(registry.execute, "start_service", {"name": "blocked", "cmd": cmd,
                                 "readiness": {"timeout_sec": 0}})
        assert entered.wait(timeout=5)
        if owner == "companion":
            proc = supervisor._runtimes["panic-test:blocked"].process
            request = supervisor.panic_kill_all
        elif owner == "service":
            proc = services._SERVICES["panic-child:blocked"].proc
            request = services.kill_all_services
        else:
            proc = executor._SERVICES["panic-child:blocked"].local_proc
            request = services.kill_all_services
        receipts = pool.submit(request, request_only=True).result(timeout=2)
        assert any(row["pid"] == proc.pid and row["requested"] for row in receipts)
        assert not future.done(), "custody I/O should still be blocked"
        assert proc.wait(timeout=2) is not None
    finally:
        release.set()
        pool.shutdown(wait=True)
        if proc is not None:
            request_process_tree_kill(proc)
            proc.wait(timeout=5)
        services.kill_all_services(ctx.drive_root)
        supervisor.stop_all(timeout_sec=0.1)
