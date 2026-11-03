"""Real scheduling, queue/worker mirrors and native continuation for supervisor cognition."""
from __future__ import annotations

import copy
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from ouroboros.model_slots import task_model_binding, task_processing_preference
from ouroboros.task_results import load_task_result
from ouroboros.tools import control
from ouroboros.tools.control_scheduling import _schedule_task
from ouroboros.tools.registry import ToolContext
from tests.test_available_subagents_runtime import _api_row, _session_row, _settings
from tests.test_supervisor_route import cognition_row, selected_task
from tests.test_swarm_host_admission import host as host

pytestmark = pytest.mark.serial


def root_context(tmp_path, settings, monkeypatch):
    from tests._shared import saved_settings

    monkeypatch.setattr(control, "load_settings", lambda: settings)
    saved_settings(monkeypatch, tmp_path, **settings)  # route keys (a row's key) read the saved document
    monkeypatch.setenv("OUROBOROS_MAX_SUBAGENT_DEPTH", "4")
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id="root")
    ctx.active_model, ctx.active_effort, ctx.active_use_local = "openai::parent", "xhigh", False
    ctx.task_metadata = {"root_task_id": "root"}
    return ctx


def schedule(ctx, **extra):
    return _schedule_task(ctx, subagent_id="session-builder", objective="Inspect the assigned files",
                          expected_output="Findings", memory_mode="empty", **extra)


@pytest.mark.parametrize("installation", ["api", "subscription", "mixed"])
@pytest.mark.parametrize("mode", ["advisory", "blocking", "cyber_pro"])
def test_schema_and_selection_are_unconditional_across_installs_and_modes(tmp_path, monkeypatch, installation, mode):
    from ouroboros.tools.control_subagent_spec import schedule_subagent_properties

    before = schedule_subagent_properties()
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", "blocking" if mode == "blocking" else "advisory")
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", "cyber_pro" if mode == "cyber_pro" else "advanced")
    settings = _settings(_api_row()) if installation == "api" else _settings(_session_row(), cognition_row())
    if installation == "mixed":
        row = _api_row("cognition", "openai::gpt-5.6-sol")
        row["processing_preference"] = "standard"
        settings = {**_settings(_session_row(), row), "OPENAI_API_KEY": "test-key"}
    ctx = root_context(tmp_path, settings, monkeypatch)
    if installation == "api":
        answer = _schedule_task(ctx, subagent_id="api-builder", objective="Inspect", expected_output="Findings", memory_mode="empty")
    else:
        answer = schedule(ctx, supervisor="cognition")
    assert answer.startswith("Subagent request queued"), answer
    event = next(e for e in ctx.pending_events if e["type"] == "schedule_subagent")
    if installation != "api":
        assert event["parent_cognitive_route"]["chosen_by"] == "delegator"
        assert "chosen by the delegator (cognition)" in answer
    assert schedule_subagent_properties() == before
    assert before["supervisor"]["type"] == "string"


@pytest.mark.parametrize("bad", ["session", "disabled", "credentials"])
def test_tool_refuses_invalid_supervisor_before_drive_or_event_and_accepts_valid(tmp_path, monkeypatch, bad):
    row = {"session": _session_row("cognition"), "disabled": {**cognition_row(), "enabled": False},
           "credentials": _api_row("cognition", "openai::gpt-5.6-sol")}[bad]
    ctx = root_context(tmp_path, _settings(_session_row(), row), monkeypatch)
    answer = schedule(ctx, supervisor="cognition")
    assert {"session": "supervisor_requires_api_model", "disabled": "supervisor_disabled",
            "credentials": "supervisor_credentials_unavailable"}[bad] in answer
    assert not any(e["type"] == "schedule_subagent" for e in ctx.pending_events)
    assert not (tmp_path / "task_drives").exists()
    monkeypatch.setattr(control, "load_settings", lambda: _settings(_session_row(), cognition_row()))
    assert schedule(ctx, supervisor="cognition").startswith("Subagent request queued")


def test_explicit_route_crosses_event_queue_restart_worker_and_agent_mirrors(host, tmp_path, monkeypatch):
    from ouroboros.agent import OuroborosAgent
    from ouroboros.subagent_messages import subagent_message_meta
    from ouroboros.task_results import write_task_result
    from supervisor import events, queue, workers
    from tests.test_project_hold_recovery import worker

    ctx = root_context(host.root, _settings(_session_row(), cognition_row()), monkeypatch)
    # Restore requires the real parent's durable lineage, as a running agent publishes it.
    write_task_result(host.root, "root", "running", root_task_id="root", delegation_role="root")
    assert schedule(ctx, supervisor="cognition").startswith("Subagent request queued")
    event = next(e for e in ctx.pending_events if e["type"] == "schedule_subagent")
    original = copy.deepcopy(event["parent_cognitive_route"])
    tid = event["task_id"]
    assert load_task_result(host.root, tid)["parent_cognitive_route"] == original
    events._handle_schedule_task(event, host.ctx)
    assert len(host.pending) == 1
    assert host.pending[0]["metadata"]["parent_cognitive_route"] == original
    # Queue persistence retains the full JSON snapshot, including its selected row.
    assert queue.persist_queue_snapshot()
    host.pending.clear()
    monkeypatch.setattr(control, "load_settings", lambda: _settings(_api_row("cognition", "different-model")))
    assert queue.restore_pending_from_snapshot() == 1
    assert host.pending[0]["parent_cognitive_route"] == original
    assert queue.resume_budget_paused_task(tid)["ok"]
    sent = worker(host, monkeypatch)
    workers.assign_tasks()
    assert len(sent) == 1
    task = host.running[tid]["task"]
    assert task["parent_cognitive_route"] == original
    assert load_task_result(host.root, tid)["parent_cognitive_route"] == original
    OuroborosAgent._persist_running_record(SimpleNamespace(env=SimpleNamespace(drive_root=host.root)), task)
    assert load_task_result(host.root, tid)["parent_cognitive_route"] == original
    assert task_model_binding(task) == ("subagent:cognition", "account-a")
    assert task_processing_preference(task) == "economy"
    # This shared projection feeds running, heartbeat and replayed card metadata.
    meta = subagent_message_meta(task, task_id=tid, event="running")
    assert meta["parent_cognitive_route"] == original
    assert not host.attempts


@pytest.mark.parametrize("valid", [False, True])
def test_admission_refuses_invalid_v2_without_enqueuing(host, monkeypatch, valid):
    from supervisor import events

    ctx = root_context(host.root, _settings(_session_row(), cognition_row()), monkeypatch)
    assert schedule(ctx, supervisor="cognition").startswith("Subagent request queued")
    event = next(e for e in ctx.pending_events if e["type"] == "schedule_subagent")
    if not valid:
        event["parent_cognitive_route"].pop("credential_profile_id")
    events._handle_schedule_task(event, host.ctx)
    result = load_task_result(host.root, event["task_id"])
    assert bool(host.pending) is valid
    assert result["status"] == ("scheduled" if valid else "failed")
    if not valid:
        assert result["reason_code"] == "invalid_parent_cognitive_route"


def test_dispatch_keeps_leaf_separate_and_discloses_recorded_choice(monkeypatch):
    from ouroboros import subagents
    from ouroboros.agent_dispatch import dispatch_executor_note
    from ouroboros import claudexor_daemon

    task = selected_task()
    task.update(id="child", delegation_role="subagent", task_constraint={})
    seen = []
    monkeypatch.setattr(claudexor_daemon, "ensure_owned_gateway", lambda: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(subagents, "route_health", lambda *a, **kw: seen.append(kw) or ("", ""))
    dispatch = subagents.resolve_subagent_dispatch(task, task_type="task")
    assert dispatch.executor == "harness"
    assert seen[0]["pinned_profile"] == "profile-1"
    assert dispatch.lane.model == task["parent_cognitive_route"]["model"]
    assert dispatch.effort == "high"
    note = dispatch_executor_note(dispatch.executor_resolution, dispatch.lane, cognitive_route=task["parent_cognitive_route"])
    assert "chosen by the delegator (cognition)" in note and "switch_model" in note
    assert "metered rounds" not in note
    # Validation happens before any fresh leaf health request.
    task["parent_cognitive_route"].pop("processing_preference")
    from ouroboros.subagent_runtime import SubagentSelectionError
    with pytest.raises(SubagentSelectionError, match="invalid_parent_cognitive_route"):
        subagents.resolve_subagent_dispatch(task, task_type="task")
    assert len(seen) == 1


def test_worker_payload_validates_and_copies_the_whole_binding():
    from ouroboros.subagent_runtime import SubagentSelectionError
    from supervisor.task_dispatch import build_scheduled_task_payload

    fields = {**selected_task(), "tid": "child", "delegation_role": "subagent"}
    expected = copy.deepcopy(fields["parent_cognitive_route"])
    payload = build_scheduled_task_payload(fields)
    assert payload["parent_cognitive_route"] == payload["metadata"]["parent_cognitive_route"] == expected
    fields["parent_cognitive_route"]["selected_supervisor_snapshot"]["route"]["target_id"] = "edited"
    assert payload["parent_cognitive_route"] == expected
    fields["parent_cognitive_route"].pop("credential_profile_id")
    with pytest.raises(SubagentSelectionError, match="invalid_parent_cognitive_route"):
        build_scheduled_task_payload(fields)


@pytest.mark.parametrize("switch", ["none", "owner", "role"])
def test_restart_restores_frozen_binding_and_later_switch(tmp_path, monkeypatch, switch):
    from ouroboros import loop, owner_wait
    from ouroboros.model_wait import TaskModelWait
    from tests.test_working_checkpoint import _registry

    registry = _registry(tmp_path, monkeypatch, "child", 1)
    ctx = registry._ctx
    task = selected_task()
    ctx.task_metadata = copy.deepcopy(task)
    role = "light" if switch == "role" else "subagent:cognition"
    ctx.context_fit_plan = replace(ctx.context_fit_plan, model_role=role)
    ctx.active_model = "claudexor::codex=owner-switch" if switch != "none" else task["parent_cognitive_route"]["model"]
    ctx.active_effort, ctx.active_use_local = "xhigh", False
    # A real continuation stores the wait-card override independently of frozen intent.
    ctx.model_wait_context = TaskModelWait(task={"id": "child"}, drive_root=tmp_path, event_queue=None, worker_slot_held=True)
    if switch == "owner":
        ctx.model_wait_context.overrides[role] = {"model": ctx.active_model, "use_local": False,
                                                 "model_account_override": "owner-account"}
    monkeypatch.setenv("OUROBOROS_MODEL_ACCOUNTS", '{"main":"wrong-main","light":"light-account"}')
    saved = owner_wait.continuation_state(ctx, [], {}, {}, 3, [], set())
    saved = json.loads(json.dumps(saved, ensure_ascii=False))
    restored = _registry(tmp_path, monkeypatch, "child", 2)
    restored._ctx.task_metadata = copy.deepcopy(task)
    restored._ctx.model_wait_context = TaskModelWait(task={"id": "child"}, drive_root=tmp_path, event_queue=None, worker_slot_held=True)
    restored._ctx.model_wait_context.restore_continuation(saved["model_wait"], started_at=None)
    owner_wait.restore_continuation_state(restored, saved, [], {}, {}, set())
    calls = []
    monkeypatch.setattr(loop, "_rebind_context_fit_plan", lambda *a, **kw: calls.append(kw) or ("plan", "max"))
    assert owner_wait.rebind_restored_route(restored, saved, []) == ("plan", "max")
    call = calls[0]
    assert call["model"] == ctx.active_model and call["model_role"] == role
    assert call["credential_profile_id"] == ("owner-account" if switch == "owner" else None if switch == "role" else "account-a")
    assert task_processing_preference({"task_metadata": restored._ctx.task_metadata}) == "economy"
    assert restored._ctx.task_metadata == task


@pytest.mark.parametrize("inherited", [False, True])
def test_real_loop_send_uses_supervisors_own_role_account_and_processing(tmp_path, monkeypatch, inherited):
    from ouroboros import loop
    from ouroboros.supervisor_route import capture_cognitive_route
    from tests.test_loop_transport_wait import _loop_kwargs
    from tests.test_working_checkpoint import _registry
    from tests.test_supervisor_route import parent

    class SendReached(BaseException):
        pass

    task = selected_task()
    if inherited:
        task["parent_cognitive_route"] = capture_cognitive_route(parent(task_metadata=task), task["configured_subagent"], {})
    role, pin = task_model_binding(task)
    registry = _registry(tmp_path, monkeypatch, "child", 1)
    ctx = registry._ctx
    ctx.task_metadata = task
    ctx.task_model_override = task["parent_cognitive_route"]["model"]
    ctx.task_use_local_override = False
    ctx.context_fit_plan = replace(ctx.context_fit_plan, model_role=role, model=ctx.task_model_override)
    calls = []

    def send(*args, **kwargs):
        calls.append(kwargs)
        assert args[2] == ctx.task_model_override
        assert kwargs["model_role"] == role != "main"
        assert kwargs["model_account_override"] == pin == "account-a"
        assert kwargs["processing_preference"] == "economy"
        raise SendReached()

    monkeypatch.setattr(loop, "call_llm_with_retry", send)
    with pytest.raises(SendReached):
        loop.run_llm_loop(**_loop_kwargs(tmp_path, registry, []))
    assert len(calls) == 1
