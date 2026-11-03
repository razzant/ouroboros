"""The real worker start seam for #939 clock tests (no loop, no model call)."""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest


class _Started(BaseException):
    """Raised right after the real agent restored this start's clocks (no loop runs)."""

    def __init__(self, ctx):
        super().__init__("started")
        self.ctx = ctx


def agent_start(row, root, monkeypatch):
    """The REAL worker seam: ``OuroborosAgent.handle_task`` builds the task's wait owner,
    ``_handle_task_scoped`` takes the origin decision and ``_prepare_task_context`` restores the
    saved clocks (``_restore_saved_clocks``). Only side-effect sinks are stubbed."""
    import queue as stdqueue
    import threading

    from ouroboros import agent as agent_module
    from ouroboros.agent import Env, OuroborosAgent
    from ouroboros.settings_integrity import task_settings_snapshot

    real_restore = agent_module._restore_saved_clocks

    def restore_then_stop(ctx, task, *args, **kwargs):
        real_restore(ctx, task, *args, **kwargs)
        raise _Started(ctx)

    monkeypatch.setattr(agent_module, "_restore_saved_clocks", restore_then_stop)
    monkeypatch.setattr(agent_module.subagent_runtime, "apply_task_start_settings_or_disclose",
                        lambda *_a: task_settings_snapshot({}, os.environ))
    monkeypatch.setattr(agent_module, "attach_task_contract", lambda task: task)
    monkeypatch.setattr(agent_module, "apply_consciousness_authority", lambda task: task)
    monkeypatch.setattr(agent_module, "resolve_dispatch_axes", lambda task: {})
    monkeypatch.setattr(agent_module, "_record_executor_resolution", lambda *_a: None)
    monkeypatch.setattr(agent_module, "sanitize_task_for_event", lambda task, _logs: task)
    repo = root / "agent-repo"
    repo.mkdir(exist_ok=True)
    agent = object.__new__(OuroborosAgent)
    agent.env = Env(repo_dir=repo, drive_root=root)
    agent._event_queue = None
    agent._owner_message_admission_lock = threading.Lock()
    agent._incoming_messages = stdqueue.Queue()
    agent.tools = SimpleNamespace(_ctx=None)
    agent._start_task_heartbeat_loop = lambda _task_id: None
    agent._persist_running_record = lambda _task: None
    with pytest.raises(_Started) as started:
        agent.handle_task(row)
    return started.value.ctx.model_wait_context
