"""Review regressions at the existing restart, wait and schedule owners."""

import json
import os
import threading
from types import SimpleNamespace

import pytest

from ouroboros import delegate_recovery


@pytest.mark.parametrize("lifecycle", ["tombstoned", "deleting"])
def test_project_followup_consumes_only_a_permanent_refusal(tmp_path, monkeypatch, lifecycle):
    from ouroboros.projects_registry import create_project, begin_project_deletion, complete_project_deletion
    from ouroboros.tools.followup import _handle_schedule_followup
    from ouroboros.tools.registry import ToolContext
    from supervisor import queue

    root = tmp_path / "data"
    monkeypatch.setattr(queue, "DRIVE_ROOT", root)
    for name, value in {"PENDING": [], "RUNNING": {}, "ADMISSION_RESERVATIONS": {},
                        "BUDGET_ROOT_FENCES": {}, "ACCEPTANCE_FENCES": {},
                        "QUEUE_SEQ_COUNTER_REF": {"value": 0}}.items():
        monkeypatch.setattr(queue, name, value)
    monkeypatch.setattr(queue, "persist_queue_snapshot", lambda **_: None)
    project = create_project(root, "project-a", name="Project A")
    ctx = ToolContext(repo_dir=tmp_path, drive_root=root, task_id="source-task",
                      project_id=project["id"], current_chat_id=project["chat_id"])
    ctx.task_metadata = {"resource_intent": {"kind": "room_default", "project_id": project["id"]}}
    assert _handle_schedule_followup(ctx, run_at="2000-01-01T00:00:00Z",
                                     objective="Continue in the same project", relation="independent").startswith("FOLLOWUP_SCHEDULED")
    begin_project_deletion(root, project["id"])
    if lifecycle == "tombstoned":
        complete_project_deletion(root, project["id"])
    monkeypatch.setattr("ouroboros.config.get_bg_wakeup_min_sec", lambda: 0)  # a wait ends at once here
    queue.check_scheduled_tasks()
    first = queue.list_scheduled_tasks(root)["tasks"][0]
    queue.check_scheduled_tasks()
    second = queue.list_scheduled_tasks(root)["tasks"][0]
    assert queue.PENDING == []
    # #1315: neither a permanent nor a transient refusal mints a failed root.
    assert not list((root / "task_results").glob("*.json"))
    assert not second.get("failure_count")
    if lifecycle == "tombstoned":
        assert second["enabled"] is False and second["completed_at"]  # the obligation is consumed, loudly
        assert "deleted" in second["last_error"] and "occurrence" not in second
    else:
        assert second["enabled"] is True and not second.get("completed_at")
        assert second["hold"]["reason"] == "project_routing_fence"
        assert second["occurrence"]["task_id"] == first["occurrence"]["task_id"]  # one occurrence, retried


class ReachedExec(BaseException):
    """Stop before OS effects while retaining the real exec environment."""


def direct_exec_environment(monkeypatch, root):
    import server
    import ouroboros.server_control as control

    captured = {}
    monkeypatch.setattr(control, "IS_WINDOWS", False)  # this transport is the POSIX same-PID exec
    monkeypatch.setattr(server, "DATA_DIR", root)
    monkeypatch.setattr(server, "_owner_restart_requested", threading.Event())
    monkeypatch.setenv("OUROBOROS_SERVER_HOST", "127.0.0.1")
    def execvpe(_file, _argv, env):
        captured.update(env)
        raise ReachedExec()
    monkeypatch.setattr(control.os, "execvpe", execvpe)
    monkeypatch.setattr("ouroboros.process_custody.spawn_supervised",
                        lambda argv, **kw: execvpe(argv[0], argv, kw["env"]))
    with pytest.raises(ReachedExec):
        server._restart_current_process("127.0.0.1", 8765)
    return captured


@pytest.mark.parametrize("intent", ["manual_restart", "manual_rollback"])
def test_aborted_update_does_not_lend_old_handoff_authority_to_manual_restart(tmp_path, monkeypatch, intent):
    from supervisor import update_merge

    transaction = {"transaction_id": "aborted", "status": "prepared", "supervisor_pid": os.getpid()}
    delegate_recovery._write_restart_transaction(tmp_path, transaction)
    delegate_recovery._active_restart_transaction_path(tmp_path).write_text(json.dumps({"transaction_id": "aborted"}))
    monkeypatch.delenv(delegate_recovery.PLANNED_RESTART_TRANSACTION_ENV, raising=False)
    monkeypatch.setattr(update_merge, "active_update_tx", lambda: {})
    monkeypatch.setattr(update_merge, "read_update_tx_strict", lambda: ("absent", {}))
    if intent == "manual_restart":
        from tests._work_stop_helpers import live_restart
        live_restart(tmp_path, monkeypatch)
    env = direct_exec_environment(monkeypatch, tmp_path)
    assert delegate_recovery.PLANNED_RESTART_TRANSACTION_ENV not in env
    assert delegate_recovery._read_restart_transaction(tmp_path, "aborted")["status"] == "prepared"


def test_failed_exception_attachment_keeps_the_original_error_and_unknown_projection(tmp_path):
    from ouroboros.agent import _task_exception_terminal
    from ouroboros.loop_budget import _LoopExitContext

    class FixedError(RuntimeError):
        def __setattr__(self, _key, _value):
            raise TypeError("attributes unavailable")

    error = FixedError("original failure")
    _LoopExitContext(None, None, "t", None, tmp_path, {"rounds": 7}, {"tool_calls": [1]}).attach_exception_evidence(error)
    text, usage, trace = _task_exception_terminal(SimpleNamespace(drive_root=tmp_path), {"id": "t"}, error, tmp_path)
    assert "FixedError: original failure" in text
    assert usage["loop_evidence_unavailable"] is trace["loop_evidence_unavailable"] is True
