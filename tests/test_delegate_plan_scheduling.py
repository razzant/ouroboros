"""Configured plan intent survives scheduling, dispatch and the initial leaf start."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from ouroboros.delegate_shared import delegate_result
from ouroboros.tools import control
from ouroboros.tools.control_subagent_spec import _validated_schedule_fields, schedule_subagent_properties
from ouroboros.tools.registry import ToolContext
from tests._shared import configure_test_subagent
from tests._store_shared import read_result
from tests import _delegate_plan_shared

plan_setup = _delegate_plan_shared.plan_setup


@pytest.mark.parametrize("kind", ["api_model", "agent_session"])
@pytest.mark.parametrize("mode", [None, "ask", "plan", "agent"])
def test_schedule_mode_is_explicit_session_intent_and_api_omission_still_runs(tmp_path, monkeypatch, kind, mode):
    selected = configure_test_subagent(monkeypatch, kind=kind,
        target="some-route=model" if kind == "agent_session" else "openai/test-model")
    ctx = ToolContext(repo_dir=tmp_path, drive_root=tmp_path / "data")
    ctx.task_id, ctx.pending_events = "parent", []
    result = control._schedule_task(ctx, subagent_id=selected, objective="Examine the change",
        expected_output="An explanation", **({"mode": mode} if mode else {}))
    events = [row for row in ctx.pending_events if row["type"] == "schedule_subagent"]
    if kind == "api_model" and mode:
        assert "delegated_mode_unsupported" in result and not events
        return
    assert len(events) == 1, result
    event = events[0]
    assert event.get("delegated_mode") == mode
    saved = read_result(ctx.drive_root, event["task_id"])
    assert saved.get("delegated_mode") == mode
    from ouroboros.subagents import subagent_intent_fields
    assert subagent_intent_fields(event).get("delegated_mode") == mode


@pytest.mark.parametrize("mode", ["plan", None])
def test_bootstrap_passes_saved_mode_to_exact_start(plan_setup, monkeypatch, mode):
    from ouroboros import subagent_runtime, subagent_bootstrap
    p, calls = plan_setup, []
    monkeypatch.setattr(subagent_runtime, "exact_start", lambda ctx, prompt, spec:
        calls.append(spec) or delegate_result({"status": "started", "run_id": "initial", "custody_durable": True}))
    task = {"id": p.ctx.task_id, "objective": "Plan the change", "configured_subagent": p.snapshot,
            "task_contract": {"objective": "Plan the change"}, **({"delegated_mode": mode} if mode else {})}
    result = json.loads(subagent_bootstrap.bootstrap_before_context(
        p.ctx, task, SimpleNamespace(executor="harness", blocked=False)))
    assert result["status"] == "configured_session_started", result
    assert len(calls) == 1 and calls[0].get("mode") == mode
    assert p.ctx._configured_actor_bootstrap.get("delegated_mode") == mode


def test_mode_schema_is_unconditional_and_bad_values_refuse():
    from ouroboros.tools.delegate import get_tools
    schema = next(tool.schema for tool in get_tools() if tool.name == "delegate_start")
    assert schema["parameters"]["properties"]["mode"]["enum"] == ["ask", "plan", "agent"]
    assert schedule_subagent_properties()["mode"]["enum"] == ["ask", "plan", "agent"]
    fields, refused = _validated_schedule_fields({"mode": "invented", "objective": "Plan", "expected_output": "Plan"})
    assert not fields and "mode must be" in refused
    fields, refused = _validated_schedule_fields({"mode": "plan", "objective": "Plan", "expected_output": "Plan"})
    assert not refused and fields["delegated_mode"] == "plan"
