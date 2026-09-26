"""Review regressions at the existing restart, wait and schedule owners."""

import json
import os
import threading
from types import SimpleNamespace

import pytest

from ouroboros import delegate_recovery, owner_quiz, owner_wait
from ouroboros.task_results import load_task_result, write_task_result
from tests.test_owner_wait_restart import restart_case as _restart_case, restore_stale_snapshot

restart_case = _restart_case


@pytest.mark.parametrize("terminal", ["cancelled", "failed"])
def test_answered_quiz_closes_wait_before_resume_capacity_is_granted(tmp_path, terminal):
    write_task_result(tmp_path, "t", "running")
    owner_quiz.record_asked(tmp_path, "t", quiz_id="q", question="Proceed?", options=["Yes"])
    wait = {"quiz_id": "q", "wait_id": "w", "state": "waiting", "source_ref": {"path": "preserved"}}
    owner_wait.set_owner_wait(tmp_path, "t", wait)
    answer = owner_quiz.record_answered(tmp_path, "t", quiz_id="q", option_index=0,
                                       request_id="answer-1", comment="Keep this exact answer")
    write_task_result(tmp_path, "t", terminal, result="terminal work")

    assert owner_quiz.reconcile_terminal(tmp_path, "t") == []
    row = load_task_result(tmp_path, "t")
    assert row["owner_quiz"]["q"] == answer["block"]
    assert row["owner_wait"]["state"] == "expired_terminal"
    assert row["owner_wait"]["source_ref"] == wait["source_ref"]
    assert row["result"] == "terminal work"
    assert owner_quiz.reconcile_terminal(tmp_path, "t") == []
    assert load_task_result(tmp_path, "t") == row


@pytest.mark.parametrize("lifecycle", ["tombstoned", "deleting"])
def test_project_followup_consumes_only_a_permanent_refusal(tmp_path, monkeypatch, lifecycle):
    from ouroboros.projects_registry import create_project, begin_project_deletion, complete_project_deletion
    from ouroboros.tools.followup import _handle_schedule_followup
    from ouroboros.tools.registry import ToolContext
    from supervisor import queue, queue_schedules

    root = tmp_path / "data"
    monkeypatch.setattr(queue, "DRIVE_ROOT", root)
    for name, value in {"PENDING": [], "RUNNING": {}, "ADMISSION_RESERVATIONS": {},
                        "BUDGET_ROOT_FENCES": {}, "ACCEPTANCE_FENCES": {},
                        "QUEUE_SEQ_COUNTER_REF": {"value": 0}}.items():
        monkeypatch.setattr(queue, name, value)
    monkeypatch.setattr(queue, "persist_queue_snapshot", lambda **_: None)
    monkeypatch.setattr(queue_schedules, "resync_skill_schedules", lambda *_: {})
    project = create_project(root, "project-a", name="Project A")
    ctx = ToolContext(repo_dir=tmp_path, drive_root=root, task_id="source-task",
                      project_id=project["id"], current_chat_id=project["chat_id"])
    assert _handle_schedule_followup(ctx, run_at="2000-01-01T00:00:00Z",
                                     objective="Continue in the same project").startswith("FOLLOWUP_SCHEDULED")
    begin_project_deletion(root, project["id"])
    if lifecycle == "tombstoned":
        complete_project_deletion(root, project["id"])
    queue.check_scheduled_tasks()
    first = queue.list_scheduled_tasks(root)["tasks"][0]
    failed = load_task_result(root, first["last_task_id"])
    assert failed["status"] == "failed" and failed["reason_code"] == "project_routing_fence"
    queue.check_scheduled_tasks()
    second = queue.list_scheduled_tasks(root)["tasks"][0]
    assert queue.PENDING == []
    if lifecycle == "tombstoned":
        assert second["last_task_id"] == first["last_task_id"]
        assert second["enabled"] is False and second["completed_at"]
        assert len(list((root / "task_results").glob("*.json"))) == 1
        assert second["failure_count"] == 1 and "project_routing_fence" in second["last_error"]
    else:
        assert second["last_task_id"] != first["last_task_id"]
        assert second["enabled"] is True and not second.get("completed_at")


class ReachedExec(BaseException):
    """Stop before OS effects while retaining the real exec environment."""


def direct_exec_environment(monkeypatch, root):
    import server
    import ouroboros.server_control as control

    captured = {}
    monkeypatch.setattr(server, "DATA_DIR", root)
    monkeypatch.setattr(server, "_owner_restart_requested", threading.Event())
    monkeypatch.setenv("OUROBOROS_SERVER_HOST", "127.0.0.1")
    def execvpe(_file, _argv, env):
        captured.update(env)
        raise ReachedExec()
    monkeypatch.setattr(control.os, "execvpe", execvpe)
    with pytest.raises(ReachedExec):
        server._restart_current_process("127.0.0.1", 8765)
    return captured


def test_assisted_update_carries_the_already_parked_wait_to_direct_successor(restart_case, monkeypatch):
    import server
    from ouroboros.gateway import control
    from supervisor import git_ops, update_merge, workers, queue

    case = restart_case
    monkeypatch.setattr(git_ops, "DRIVE_ROOT", case.root)
    monkeypatch.setattr(workers, "close_repo_writer_admission", lambda *_: None)
    monkeypatch.setattr(workers, "drain_repo_writers", lambda: [])
    real_kill = workers.kill_workers
    def kill(**kwargs):
        return real_kill(**{**kwargs, "reconcile_delegate_custody": False, "archive_service_logs": False})
    monkeypatch.setattr(workers, "kill_workers", kill)
    assert control._quiesce_repo_writers("assisted") == []
    assert not workers.RUNNING and not case.process.is_alive()
    successor = workers.PENDING[0]
    transaction_id = successor["_owner_wait_resume"]["restart_transaction_id"]
    monkeypatch.setattr(update_merge, "active_update_tx", lambda: {"phase": "pending_boot_smoke"})
    monkeypatch.setattr(update_merge, "read_update_tx_strict", lambda: ("valid", {"phase": "pending_boot_smoke"}))
    monkeypatch.setattr(server, "_safe_restart_serialized", lambda *_a, **_kw: (True, "ok"))
    monkeypatch.setattr(server, "_request_restart_exit", lambda: None)
    monkeypatch.setattr(server, "_planned_delegate_restart_transaction_id", "")
    state = {}
    ctx = SimpleNamespace(DRIVE_ROOT=case.root, REPO_DIR=case.root, RUNNING=workers.RUNNING,
        load_state=lambda: state.copy(), save_state=lambda s: state.update(s), safe_restart=lambda **_: (True, "ok"),
        kill_workers=kill, persist_queue_snapshot=queue.persist_queue_snapshot)
    server._perform_supervisor_restart(ctx)
    assert server._planned_delegate_restart_transaction_id == ""
    env = direct_exec_environment(monkeypatch, case.root)
    assert env[delegate_recovery.PLANNED_RESTART_TRANSACTION_ENV] == transaction_id
    monkeypatch.setenv(delegate_recovery.PLANNED_RESTART_TRANSACTION_ENV, env[delegate_recovery.PLANNED_RESTART_TRANSACTION_ENV])
    assert restore_stale_snapshot(case) == 1
    assert workers.PENDING[0]["_owner_wait_resume"]["wait_id"] == case.wait["wait_id"]
    assert delegate_recovery._read_restart_transaction(case.root, transaction_id)["status"] == "normal_exit_acknowledged"


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
        (tmp_path / "state" / "owner_restart_no_resume.flag").write_text("owner restart")
    env = direct_exec_environment(monkeypatch, tmp_path)
    assert delegate_recovery.PLANNED_RESTART_TRANSACTION_ENV not in env
    assert delegate_recovery._read_restart_transaction(tmp_path, "aborted")["status"] == "prepared"


def test_owner_no_resume_flag_dominates_a_restartable_update(tmp_path, monkeypatch):
    from supervisor import update_merge

    delegate_recovery._write_restart_transaction(tmp_path,
        {"transaction_id": "prior", "status": "prepared", "supervisor_pid": os.getpid()})
    delegate_recovery._active_restart_transaction_path(tmp_path).write_text(json.dumps({"transaction_id": "prior"}))
    (tmp_path / "state" / "owner_restart_no_resume.flag").write_text("owner restart")
    monkeypatch.setenv(delegate_recovery.PLANNED_RESTART_TRANSACTION_ENV, "prior")
    monkeypatch.setattr(update_merge, "read_update_tx_strict", lambda: ("valid", {"phase": "pending_boot_smoke"}))
    assert delegate_recovery.PLANNED_RESTART_TRANSACTION_ENV not in direct_exec_environment(monkeypatch, tmp_path)


def test_pre_loop_checkpoint_failure_reports_unknown_counts_without_reading_corrupt_source(restart_case, monkeypatch):
    from ouroboros import agent as agent_module, agent_task_pipeline, loop
    from ouroboros.agent import Env, OuroborosAgent

    case = restart_case
    case.task["_owner_wait_resume"] = {**case.wait, "restart_transaction_id": "prepared-test"}
    monkeypatch.setattr(OuroborosAgent, "_log_worker_boot_once", lambda _: None)
    monkeypatch.setattr(agent_module, "build_llm_messages", lambda **_: ([], {}))
    monkeypatch.setattr(agent_task_pipeline, "_run_post_task_processing_async", lambda *_a, **_kw: None)
    def unreadable(_ctx):
        raise ValueError("owner-wait source checksum mismatch")
    monkeypatch.setattr(loop, "load_owner_wait", unreadable)
    monkeypatch.setattr("ouroboros.llm.LLMClient.chat", lambda *_a, **_kw: pytest.fail("no generation during failed recovery"))
    agent = OuroborosAgent(Env(repo_dir=case.root, drive_root=case.root))
    events = agent._handle_task_scoped(case.task)
    stored = load_task_result(case.root, case.task_id)
    assert stored["status"] == "failed" and stored["reason_code"] == "task_exception"
    assert "unknown" in stored["trace_summary"] and "0 calls" not in stored["trace_summary"]
    assert stored["loop_outcome"]["usage"]["total_rounds"] is None
    assert stored["loop_outcome"]["usage"]["prompt_tokens"] is None
    metrics = next(event for event in events if event["type"] == "task_metrics")
    assert metrics["tool_calls"] is None and metrics["tool_errors"] is None
    from supervisor.events_worker_reports import _handle_task_metrics
    published = []
    metrics_ctx = SimpleNamespace(DRIVE_ROOT=case.root, RUNNING={},
        append_jsonl=lambda _path, row: published.append(row),
        bridge=SimpleNamespace(push_log=lambda row: None))
    _handle_task_metrics(metrics, metrics_ctx)
    assert published[0]["tool_calls"] is None and published[0]["tool_errors"] is None
    assert stored["owner_wait"]["source_ref"] == case.wait["source_ref"]


def test_unknown_exception_facts_row_does_not_invent_zero_rounds_or_buy_a_model_call(tmp_path, monkeypatch):
    from ouroboros.post_task_synthesis import _record_task_facts

    rows = []
    monkeypatch.setattr("ouroboros.project_dialogue.append_canonical_task_summary",
                        lambda _root, row: rows.append(row))
    monkeypatch.setattr("ouroboros.llm_observability.chat_observed", lambda *_a, **_kw: pytest.fail("no new paid summary"))
    _record_task_facts(SimpleNamespace(drive_root=tmp_path), {"id": "unknown", "text": "Recover work"},
                       {"loop_evidence_unavailable": True},
                       {"loop_evidence_unavailable": True, "tool_calls": []}, tmp_path / "logs")
    assert rows[0]["tool_calls"] is None and rows[0]["rounds"] is None
    assert rows[0]["summary_kind"] == "host_task_facts" and rows[0]["text"] == ""


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


def test_unknown_exception_evidence_stays_unknown_in_actual_reflection(tmp_path, monkeypatch):
    from ouroboros import post_task_synthesis, reflection, llm_observability
    captured = []
    monkeypatch.setattr(reflection, 'append_reflection_routed', lambda env, task, row: captured.append(row))
    monkeypatch.setattr(llm_observability, 'chat_observed', lambda *a, **kw: ({'content': 'Reflection over the disclosed unknown trace.'}, {}))
    task = {'id': 'cold-source-failure', 'type': 'task', 'text': 'Continue saved workspace work', 'workspace_root': str(tmp_path / 'workspace'), 'drive_root': str(tmp_path)}
    usage = {'loop_evidence_unavailable': True, 'execution_status': 'infra_failed', 'reason_code': 'task_exception'}
    trace = {'loop_evidence_unavailable': True, 'tool_calls': [], 'reasoning_notes': []}
    post_task_synthesis._run_reflection(SimpleNamespace(drive_root=tmp_path), None, task, usage, trace, {})
    assert len(captured) == 1, 'The real reflection path must execute and publish one row.'
    assert captured[0]['rounds'] is None and captured[0]['error_count'] is None, captured[0]
