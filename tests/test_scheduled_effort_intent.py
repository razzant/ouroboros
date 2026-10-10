"""Effort intent through the real child event, admission, queue and first start.

Only the external model/session boundaries are synthetic. In particular, do not
turn the child into a root or seed a resolved effort before dispatch: both hid
the lost request in the original scheduling regression.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest


pytestmark = pytest.mark.serial  # Production state/queue init rebinds module globals.
_OMITTED = object()
_TIERS = ("none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra")
_DERIVED = ("reasoning_effort", "effective_model_lane", "effective_executor",
            "model", "capability_delta", "effort_level", "effort_requested", "effort_source")


@pytest.fixture
def runtime(tmp_path, monkeypatch, _rebind_runtime_roots_between_tests):
    from ouroboros import claudexor_daemon, config, provider_models
    from ouroboros.llm import LLMClient
    from supervisor import git_ops, queue, state, workers
    from tests.test_subagent_role_texts import _env

    repo, drive, _ = _env(tmp_path)
    for key, value in {
        "OUROBOROS_DATA_DIR": drive, "OUROBOROS_SETTINGS_PATH": drive / "settings.json",
        "OUROBOROS_REPO_DIR": repo, "OUROBOROS_MAX_SUBAGENT_DEPTH": "3",
        "OUROBOROS_EFFORT_MIN": "none", "OUROBOROS_EFFORT_TASK": "xhigh",
        "OUROBOROS_EFFORT_MAX": "ultra",
    }.items():
        monkeypatch.setenv(key, str(value))
    monkeypatch.setattr(config, "DATA_DIR", drive)
    monkeypatch.setattr(config, "SETTINGS_PATH", drive / "settings.json")
    monkeypatch.setattr(config, "_BOOT_RUNTIME_MODE", "pro")
    for name, value in {
        "PENDING": [], "RUNNING": {}, "QUEUE_SEQ_COUNTER_REF": {"value": 0},
        "ADMISSION_RESERVATIONS": {}, "ACCEPTANCE_FENCES": {},
        "BUDGET_ROOT_FENCES": {}, "PRIOR_DIRECT_ROOTS": {},
    }.items():
        monkeypatch.setattr(queue, name, value)
    state.init(drive)
    queue.init(drive)
    for module in (workers, git_ops):
        monkeypatch.setattr(module, "DRIVE_ROOT", drive)
        monkeypatch.setattr(module, "REPO_DIR", repo)
    worker_pool = {0: SimpleNamespace(busy_task_id=None)}
    monkeypatch.setattr(workers, "WORKERS", worker_pool)
    for root in (config.DATA_DIR, state.DRIVE_ROOT, queue.DRIVE_ROOT,
                 workers.DRIVE_ROOT, git_ops.DRIVE_ROOT):
        assert root.is_relative_to(tmp_path)
    monkeypatch.setattr(provider_models, "model_has_credentials", lambda _model: True)

    def no_external_call(*_args, **_kwargs):
        pytest.fail("Scheduling regression attempted a model call or live session connection")

    monkeypatch.setattr(LLMClient, "chat", no_external_call)
    monkeypatch.setattr(claudexor_daemon, "ensure_owned_gateway", no_external_call)
    sent = []
    supervisor = SimpleNamespace(
        DRIVE_ROOT=drive, REPO_DIR=repo, PENDING=queue.PENDING, RUNNING=queue.RUNNING,
        WORKERS=worker_pool, load_state=lambda: {"owner_chat_id": 7},
        enqueue_task=queue.enqueue_task, persist_queue_snapshot=queue.persist_queue_snapshot,
        sort_pending=queue.sort_pending,
        send_with_budget=lambda chat, text, **kw: sent.append((chat, text, kw)),
    )
    return SimpleNamespace(repo=repo, drive=drive, queue=queue, supervisor=supervisor, sent=sent)


def _event(runtime, monkeypatch, *, effort=_OMITTED, pin="", kind="api_model",
           target="openai/effort-fixture-model"):
    from ouroboros.tools.control_scheduling import _schedule_task
    from ouroboros.tools.registry import ToolContext
    from tests._shared import configure_test_subagent

    sid = configure_test_subagent(monkeypatch, kind=kind, target=target, effort=pin)
    parent = ToolContext(repo_dir=runtime.repo, drive_root=runtime.drive, task_id="parent",
                         task_depth=0, current_chat_id=7, task_metadata={"root_task_id": "parent"})
    parent.active_model, parent.active_effort, parent.active_use_local = "openai/parent", "high", False
    kwargs = {} if effort is _OMITTED else {"effort": effort}
    reply = _schedule_task(parent, subagent_id=sid, objective="Inspect this source",
                           expected_output="One finding", memory_mode="empty", **kwargs)
    assert len(parent.pending_events) == 1, reply
    event = parent.pending_events[0]
    assert event["type"] == "schedule_subagent"
    assert event["delegation_role"] == "subagent" and event["depth"] == 1
    assert event["configured_subagent"]["effort"] == pin
    return parent, event


def _admit(runtime, event):
    from ouroboros.task_results import load_task_result
    from supervisor import events

    before = json.loads(json.dumps(event))
    events._handle_schedule_task(event, runtime.supervisor)
    assert event == before, "Admission mutated the producer's event"
    assert len(runtime.queue.PENDING) == 1, runtime.sent
    task = runtime.queue.PENDING[0]
    assert task["id"] == event["task_id"]
    assert task["depth"] == 1 and task["delegation_role"] == "subagent"
    assert task["metadata"]["delegation_role"] == "subagent"
    assert task["task_constraint"]["mode"] == "local_readonly_subagent"
    assert task["admitted_dispatch"] == "none"
    row = load_task_result(runtime.drive, task["id"])
    assert row["status"] == "scheduled"
    assert row["delegation_admission"]["status"] == "accepted"
    assert row["delegation_admission"]["transition_id"]
    for carrier in (event, task, task["metadata"], row):
        assert not set(_DERIVED).intersection(carrier)
    return task


def _restore(runtime):
    from ouroboros.task_results import write_task_result

    # A durable parent with no interrupted running row makes this a genuinely
    # restorable pending child, not an orphan whose lifecycle forbids dispatch.
    write_task_result(runtime.drive, "parent", "running", result="Waiting for child")
    assert runtime.queue.persist_queue_snapshot(reason="effort-intent-test")
    runtime.queue.PENDING.clear()
    assert runtime.queue.restore_pending_from_snapshot() == 1
    task = runtime.queue.PENDING[0]
    assert not task.get("_terminalization_retry")
    assert not task.get("_cancel_intent_authority_hold")
    return json.loads(json.dumps(task))  # The worker receives its own IPC copy.


@pytest.mark.parametrize("effort", _TIERS)
def test_every_raw_tier_reaches_dispatch_from_the_real_child_event(runtime, monkeypatch, effort):
    from ouroboros.agent_dispatch import resolve_dispatch_axes
    from ouroboros.task_results import load_task_result

    _, event = _event(runtime, monkeypatch, effort=effort)
    assert event["requested_effort"] == effort
    assert load_task_result(runtime.drive, event["task_id"])["requested_effort"] == effort
    task = _admit(runtime, event)
    for carrier in (task, task["metadata"], load_task_result(runtime.drive, task["id"])):
        assert carrier["requested_effort"] == effort
    dispatch = resolve_dispatch_axes(task)
    assert dispatch.executor == "native"
    assert dispatch.effort_fact == {"requested": effort, "applied": effort, "source": "auto"}


@pytest.mark.parametrize("effort", [_OMITTED, "auto", "", None], ids=["omitted", "auto", "blank", "null"])
def test_no_public_request_keeps_the_recommended_default(runtime, monkeypatch, effort):
    from ouroboros.agent_dispatch import resolve_dispatch_axes

    _, event = _event(runtime, monkeypatch, effort=effort)
    task = _admit(runtime, event)
    assert task["requested_effort"] == task["metadata"]["requested_effort"] == ""
    assert resolve_dispatch_axes(task).effort_fact == {"requested": "", "applied": "xhigh", "source": "auto"}


def test_old_event_without_request_survives_restore_without_inventing_one(runtime, monkeypatch):
    from ouroboros.agent_dispatch import resolve_dispatch_axes

    _, event = _event(runtime, monkeypatch)
    del event["requested_effort"]  # Shape of a pre-request scheduling event.
    task = _admit(runtime, event)
    assert "requested_effort" not in task and "requested_effort" not in task["metadata"]
    task = _restore(runtime)
    assert "requested_effort" not in task and "requested_effort" not in task["metadata"]
    # A genuinely old stored child can still carry the retired field. It never
    # becomes today's request or replaces the current default.
    task["reasoning_effort"] = "max"
    dispatch = resolve_dispatch_axes(task)
    assert dispatch.effort_fact == {"requested": "", "applied": "xhigh", "source": "auto"}
    assert task["reasoning_effort"] == "xhigh"


@pytest.mark.parametrize("pin,mode,target,requested,expected,source", [
    ("", "pro", "openai/effort-fixture-model", "ultra", "high", "auto"),
    ("max", "pro", "openai/effort-fixture-model", "low", "max", "pin"),
    ("high", "cyber_pro", "openai/effort-fixture-model", "ultra", "ultra", "cyber"),
    ("low", "cyber_pro", "claudexor::cursor=grok-4.7-max-fast", "none", "max", "model_name"),
], ids=["range-clamp", "owner-pin", "cyber-request", "model-name"])
def test_preserved_request_meets_the_existing_dispatch_policy(
    runtime, monkeypatch, pin, mode, target, requested, expected, source,
):
    from ouroboros.agent_dispatch import _initial_effort_for, resolve_dispatch_axes

    monkeypatch.setattr("ouroboros.config._BOOT_RUNTIME_MODE", mode)
    monkeypatch.setenv("OUROBOROS_EFFORT_MIN", "low")
    monkeypatch.setenv("OUROBOROS_EFFORT_TASK", "medium")
    monkeypatch.setenv("OUROBOROS_EFFORT_MAX", "high")
    _, event = _event(runtime, monkeypatch, effort=requested, pin=pin, target=target)
    task = _admit(runtime, event)
    dispatch = resolve_dispatch_axes(task)
    assert dispatch.executor == "native"
    assert dispatch.effort_fact == {"requested": requested, "applied": expected, "source": source}
    assert task["requested_effort"] == task["metadata"]["requested_effort"] == requested
    assert task["reasoning_effort"] == _initial_effort_for(task, "task") == expected


def test_pending_restore_keeps_intent_but_decides_against_the_current_range(runtime, monkeypatch):
    from ouroboros.agent_dispatch import resolve_dispatch_axes

    _, event = _event(runtime, monkeypatch, effort="ultra")
    task = _admit(runtime, event)
    selected = json.loads(json.dumps(task["configured_subagent"]))
    task = _restore(runtime)
    assert task["requested_effort"] == task["metadata"]["requested_effort"] == "ultra"
    assert task["configured_subagent"] == selected
    assert all(task.get(key) is None for key in _DERIVED)
    monkeypatch.setenv("OUROBOROS_EFFORT_TASK", "medium")
    monkeypatch.setenv("OUROBOROS_EFFORT_MAX", "high")
    assert resolve_dispatch_axes(task).effort_fact == {"requested": "ultra", "applied": "high", "source": "auto"}


def test_first_session_body_uses_the_leaf_choice_and_keeps_nanny_effort(runtime, monkeypatch):
    from ouroboros import claudexor_daemon, subagent_runtime, subagents
    from ouroboros.agent_dispatch import _initial_effort_for, resolve_dispatch_axes
    from ouroboros.delegate_shared import delegate_result
    from ouroboros.subagent_bootstrap import bootstrap_before_context
    from ouroboros.tools import delegate
    from ouroboros.tools.registry import ToolContext

    monkeypatch.setattr(claudexor_daemon, "ensure_owned_gateway", lambda: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(subagents, "route_health", lambda *_a, **_kw: ("", ""))
    _, event = _event(runtime, monkeypatch, effort="ultra", kind="agent_session", target="claude=route-a")
    task = _admit(runtime, event)
    dispatch = resolve_dispatch_axes(task)
    assert dispatch.executor == "harness"
    assert dispatch.effort_fact == {"requested": "ultra", "applied": "ultra", "source": "auto"}
    assert _initial_effort_for(task, "task") == dispatch.effort == "high"
    seen = []

    def start(ctx, prompt, *_args, **_kwargs):
        actor, refusal = subagent_runtime.prepare_delegate_start_actor(
            ctx, runtime.repo, recovering=False, invocation_id="",
            work_order_fingerprint="fixture", authority_fingerprint="fixture")
        assert refusal is None
        body = delegate._start_request(ctx, actor["route"], subagents.delegated_run_shape(False),
                                       str(runtime.repo), prompt, 60, "")
        seen.append((actor, body))
        return delegate_result({"status": "started", "run_id": "synthetic-first-start"})

    monkeypatch.setattr(delegate, "_delegate_start", start)
    ctx = ToolContext(repo_dir=runtime.repo, drive_root=Path(task["child_drive_root"]),
                      task_id=task["id"], task_metadata=task["metadata"])
    result = json.loads(bootstrap_before_context(ctx, task, dispatch))
    assert result["status"] == "configured_session_started" and len(seen) == 1
    actor, body = seen[0]
    assert body["effort"] == actor["route"].effort == "ultra"
    assert actor["row_effort"] == ""
    assert actor["effort_fact"] == dispatch.effort_fact


def test_real_result_writers_and_canonical_copyback_keep_the_scheduled_fact(runtime, monkeypatch):
    from ouroboros import agent_task_pipeline as pipeline
    from ouroboros.agent import OuroborosAgent
    from ouroboros.agent_dispatch import resolve_dispatch_axes
    from ouroboros.headless import copy_child_task_result
    from ouroboros.task_results import load_task_result
    from ouroboros.tools.control import _get_task_result

    monkeypatch.setenv("OUROBOROS_EFFORT_TASK", "medium")
    monkeypatch.setenv("OUROBOROS_EFFORT_MAX", "high")
    parent, event = _event(runtime, monkeypatch, effort="ultra")
    task = _admit(runtime, event)
    dispatch = resolve_dispatch_axes(task)
    child = Path(task["child_drive_root"])
    assert child != runtime.drive and child.is_relative_to(runtime.drive)
    env = SimpleNamespace(repo_dir=runtime.repo, drive_root=child, budget_drive_root=runtime.drive)
    OuroborosAgent._persist_running_record(SimpleNamespace(env=env), task)
    running = load_task_result(child, task["id"])
    monkeypatch.setattr(pipeline, "_run_post_task_processing_async", lambda *_a, **_kw: None)
    logs = child / "logs"
    logs.mkdir(exist_ok=True)
    pipeline.emit_task_results(env, None, None, [], task, "One finding.", {"rounds": 1},
                               {"tool_calls": [], "reasoning_notes": []}, 0.0, logs)
    completed = load_task_result(child, task["id"])
    canonical = copy_child_task_result(runtime.drive, task)
    for row in (running, completed, canonical):
        assert (row["effort_requested"], row["effort_level"], row["effort_source"]) == ("ultra", "high", "auto")
        assert row["metadata"]["requested_effort"] == "ultra"
    assert running["status"] == "running" and completed["status"] == canonical["status"] == "completed"
    assert canonical["requested_effort"] == dispatch.effort_fact["requested"] == "ultra"
    readback = _get_task_result(parent, task["id"])
    outcome = json.loads(readback.split("[SUBTASK_OUTCOME]\n", 1)[1].split("\n[/SUBTASK_OUTCOME]", 1)[0])
    assert outcome["effort"] == {
        "level": "high", "requested": "ultra", "source": "auto"}


def test_one_declared_intent_field_crosses_admission_and_snapshot(runtime, monkeypatch):
    from ouroboros import subagents
    from ouroboros.task_results import load_task_result

    # A producer's newly declared scalar must need no second membership list in
    # the supervisor or snapshot. An unrelated event key still stays out.
    monkeypatch.setattr(subagents, "SUBAGENT_INTENT_FIELDS", (*subagents.SUBAGENT_INTENT_FIELDS, "future_intent"))
    _, event = _event(runtime, monkeypatch, effort="none")
    event.update(future_intent="carried", undeclared_event_field="not intent")
    task = _admit(runtime, event)
    for carrier in (task, task["metadata"], load_task_result(runtime.drive, task["id"]), _restore(runtime)):
        assert carrier["future_intent"] == "carried"
        assert "undeclared_event_field" not in carrier


def test_refused_child_keeps_declared_intent_without_claiming_dispatch(runtime, monkeypatch):
    from ouroboros import subagents
    from ouroboros.task_results import load_task_result
    from supervisor import events

    monkeypatch.setattr(subagents, "SUBAGENT_INTENT_FIELDS", (*subagents.SUBAGENT_INTENT_FIELDS, "future_intent"))
    _, event = _event(runtime, monkeypatch, effort="minimal")
    event.update(future_intent="carried", required_model_lane="heavy", undeclared_event_field="not intent")
    runtime.supervisor.WORKERS = {}
    events._handle_schedule_task(event, runtime.supervisor)
    assert not runtime.queue.PENDING
    row = load_task_result(runtime.drive, event["task_id"])
    assert row["status"] == "failed" and row["reason_code"] == "workers_unavailable"
    assert row["requested_effort"] == "minimal" and row["future_intent"] == "carried"
    assert row["required_model_lane"] == ""
    assert "undeclared_event_field" not in row and not set(_DERIVED).intersection(row)


@pytest.mark.parametrize("required", ["", "main"], ids=["no-requirement", "admitted-requirement"])
def test_required_lane_comes_from_admission_not_raw_intent(runtime, monkeypatch, required):
    from ouroboros import task_tree_ledger
    from ouroboros.task_results import load_task_result

    _, event = _event(runtime, monkeypatch, effort="ultra")
    event["required_model_lane"] = "heavy"
    constraints = ([{"payload": {"directive": "require_lane", "scope": {"lane": required}}}]
                   if required else [])
    monkeypatch.setattr(task_tree_ledger, "open_delegation_constraints", lambda _root: constraints)
    task = _admit(runtime, event)
    for carrier in (task, task["metadata"], load_task_result(runtime.drive, task["id"]), _restore(runtime)):
        assert carrier["required_model_lane"] == required
        assert carrier["requested_effort"] == "ultra"
