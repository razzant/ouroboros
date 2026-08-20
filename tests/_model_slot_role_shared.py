"""Contexts and dispatch helpers shared by the model-slot role suites.

Split out of ``tests/test_model_slot_role_model.py`` when that module was divided by theme;
the helpers are verbatim, so every sibling suite schedules against the same context, the same
supervisor enqueue path and the same transport it was written against.
"""

from __future__ import annotations


import pytest



@pytest.fixture(autouse=True)
def _owned_gateway_uses_each_test_transport(monkeypatch):
    from ouroboros import claudexor_daemon
    from ouroboros.gateways import claudexor as gateway_module

    monkeypatch.setattr(
        claudexor_daemon,
        "ensure_owned_gateway",
        lambda: gateway_module.ClaudexorGateway(),
    )

def _scheduling_ctx(tmp_path, *, parent_deadline: str = "", parent_lane: str = ""):
    import queue

    from ouroboros.tools.registry import ToolContext

    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path)
    ctx.task_id = "parent1"
    ctx.task_depth = 0
    ctx.current_chat_id = 1
    ctx.event_queue = queue.Queue()
    ctx.task_metadata = {"root_task_id": "root1", "session_id": "sess1"}
    if parent_lane:
        ctx.task_metadata["effective_model_lane"] = parent_lane
    if parent_deadline:
        ctx.task_metadata["task_contract"] = {"deadline_at": parent_deadline}
    return ctx

def _enqueue_through_supervisor(tmp_path, monkeypatch, *, parent_lane: str = "", **schedule_kwargs):
    """Drive the REAL path: tool call -> event -> supervisor -> the task a worker is handed."""
    from types import SimpleNamespace

    from supervisor import events as ev_module
    from supervisor import events_schedule_task as schedule_module
    from ouroboros.tools.control import _schedule_task

    ctx = _scheduling_ctx(tmp_path, parent_lane=parent_lane)
    out = _schedule_task(ctx, objective="o", expected_output="e", **schedule_kwargs)
    assert "TOOL_ARG_ERROR" not in out, out
    event = ctx.event_queue.get_nowait()
    event["type"] = "schedule_subagent"
    event["depth"] = 0
    event["delegation_role"] = ""

    monkeypatch.setattr(schedule_module, "_find_duplicate_task", lambda *a, **k: None)
    enqueued = []

    class FakeCtx:
        DRIVE_ROOT = tmp_path
        PENDING = []
        RUNNING = {}
        WORKERS = {0: SimpleNamespace(busy_task_id=None)}

        def load_state(self):
            return {"owner_chat_id": 1}

        def send_with_budget(self, chat_id, text, **kwargs):
            pass

        def enqueue_task(self, task):
            enqueued.append(task)

        def persist_queue_snapshot(self, reason=""):
            pass

    ev_module._handle_schedule_task(event, FakeCtx())
    assert enqueued, "supervisor did not enqueue the task"
    return enqueued[0]
