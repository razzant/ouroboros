"""A tool registry whose root task is running (a queue worker row and a running task result), the invocation
owner scope its calls run under, and one tool call through the ordinary execution path; shared by the
invocation, custody and MCP task-session tests."""

from __future__ import annotations

from contextlib import contextmanager
import json

from ouroboros.loop_tool_execution import _execute_with_timeout
from ouroboros.model_wait import task_model_wait_scope
from tests._budget_pause_exact_helpers import _install_queue


def _running(root, workers, task_id="root"):
    from ouroboros.task_results import write_task_result
    write_task_result(root, task_id, "running", root_task_id=task_id, chat_id=0)
    task = {"id": task_id, "root_task_id": task_id, "type": "task", "chat_id": 0, "_attempt": 1}
    workers.RUNNING[task_id] = {"task": task, "attempt": 1, "worker_id": 0}
    return task


def _registry(tmp_path, monkeypatch, backend=None):
    from ouroboros import workspace_executor as executor
    from ouroboros import safety
    from ouroboros.tools.registry import ToolRegistry

    queue, _, workers = _install_queue(tmp_path, monkeypatch)
    _running(tmp_path, workers)
    workspace = tmp_path.parent / (tmp_path.name + "-workspace")
    workspace.mkdir()
    registry = ToolRegistry(repo_dir=tmp_path.parent / (tmp_path.name + "-system"), drive_root=tmp_path)
    ctx = registry._ctx
    ctx.task_id = ctx.root_task_id = "root"
    ctx.task_attempt = 1
    ctx.workspace_mode, ctx.workspace_root = "external", workspace
    if backend:
        ctx.executor_ref = {"type": backend, "container_name": "test-container", "network": "host",
                            "workspace_host_path": str(workspace), "workspace_backend_path": "/workspace"}
    monkeypatch.setattr(executor, "_panic_requested", False)
    monkeypatch.setattr(executor, "_FOREGROUND", {})
    monkeypatch.setattr(executor, "_SERVICES", {})
    monkeypatch.setattr(safety, "check_safety", lambda *_a, **_kw: (True, ""))
    return registry, queue, workers


@contextmanager
def owner(registry, *, attempt=1, control=None):
    ctx = registry._ctx
    with task_model_wait_scope(task={"id": ctx.task_id, "_attempt": attempt},
            drive_root=ctx.drive_root, event_queue=None, worker_slot_held=False,
            **({"owner_control": lambda: control[0]} if control is not None else {})) as held:
        held.tool_context = ctx
        yield held


def run(registry, *, timeout=3, name="knowledge_read", args=None):
    return _execute_with_timeout(registry, {"id": "call", "function": {
        "name": name, "arguments": json.dumps(args or {"topic": "fixture"})}},
        registry._ctx.drive_root / "logs", timeout, task_id=registry._ctx.task_id)
