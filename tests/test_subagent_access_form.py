"""All-fields scheduling forms preserve task authority and report real refusals."""

import json

import pytest

from ouroboros.subagent_runtime import SubagentSelectionError, select_subagent_snapshot
from ouroboros.tools import control
from ouroboros.tools.registry import ToolRegistry


def _settings():
    return {"OUROBOROS_SUBAGENTS": {"enabled": True, "items": [{
        "subagent_id": "api-scout", "recommended_use": "Inspect the assigned source.",
        "route": {"kind": "api_model", "target_id": "openai/test-model"},
    }]}}


@pytest.fixture
def registry(tmp_path, monkeypatch):
    import ouroboros.safety as safety

    monkeypatch.setattr(control, "load_settings", _settings)
    monkeypatch.setattr(safety, "check_safety", lambda *_args, **_kwargs: (True, ""))
    monkeypatch.setenv("OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS", "true")
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path / "data")
    registry._ctx.task_id = "parent"
    registry._ctx.pending_events = []
    return registry


@pytest.mark.parametrize("access", ["inherit", "readonly", "workspace_write"])
@pytest.mark.parametrize("surface", ["read_only", "self_worktree"])
def test_api_all_fields_form_queues_without_changing_write_surface(registry, access, surface):
    """The incident's complete optional shape must reach the actual registered handler."""
    args = {
        "subagent_id": "api-scout", "access": access,
        "objective": "Inspect the assigned source.", "expected_output": "Findings.",
        "role": "", "context": "", "constraints": "", "memory_mode": "forked",
        "input_sources": "shared",
        "write_surface": surface, "write_root": "", "workspace_root": "", "directory_strategy": "direct",
        "scope_paths": [], "protected_paths_grant": False, "external_tool_grants": [],
        "allowed_origins": [], "delegation_intent": "", "may_mutate": False,
        "may_fan_out": True, "max_children": 0, "requested_depth": 0,
        "required_capabilities": [], "deadline_at": "", "acceptance_claims": [], "effort": "auto",
    }
    schema = registry.get_schema_by_name("schedule_subagent")["function"]["parameters"]
    assert set(args) == set(schema["properties"])
    access_schema = schema["properties"]["access"]
    assert access_schema["enum"] == ["inherit", "readonly", "workspace_write"]
    assert access_schema["default"] == "inherit"

    result = registry.execute_result("schedule_subagent", args)

    assert (result.status, result.code) == ("ok", "OK"), result.text
    # The engine is named by its handle; the stored key never reaches the model.
    assert f"access={access!r} ignored for subagent_id='openai/test-model' (api_model)" in result.text
    assert "api-scout" not in result.text
    assert "write_surface controls read/write authority" in result.text
    events = [row for row in registry._ctx.pending_events if row["type"] == "schedule_subagent"]
    assert len(events) == 1
    event = events[0]
    assert event["write_surface"] == ("" if surface == "read_only" else surface)
    assert event["task_constraint"]["mode"] == (
        "local_readonly_subagent" if surface == "read_only" else "acting_subagent")
    assert "access" not in event["configured_subagent"]
    saved = json.loads((registry._ctx.drive_root / "task_results" / f"{event['task_id']}.json")
                       .read_text(encoding="utf-8"))
    assert saved["configured_subagent"] == event["configured_subagent"]
    assert saved["task_constraint"] == event["task_constraint"]


def test_api_omitted_access_keeps_result_free_of_an_ignored_field_notice(registry):
    result = registry.execute_result("schedule_subagent", {
        "subagent_id": "api-scout", "objective": "Inspect source.", "expected_output": "Findings.",
    })
    assert (result.status, result.code) == ("ok", "OK"), result.text
    assert "ignored" not in result.text


@pytest.mark.parametrize("access", ["full", "", "invalid", [], {}])
def test_invalid_access_names_field_value_and_selected_route(access):
    with pytest.raises(SubagentSelectionError) as raised:
        select_subagent_snapshot(_settings(), subagent_id="api-scout", access=access)
    assert raised.value.code == "subagent_access_invalid"
    assert f"access={access!r}" in raised.value.detail
    assert "subagent_id='openai/test-model' (api_model)" in raised.value.detail
    assert "api-scout" not in raised.value.detail
    assert "inherit, readonly or workspace_write" in raised.value.detail


@pytest.mark.parametrize("args,reason", [
    ({"access": "full"}, "subagent_access_invalid"),
    ({"subagent_id": "missing"}, "unknown_subagent_id"),
    ({"executor": "native"}, "subagent_selector_conflict"),
])
def test_selection_refusal_reaches_registered_pipeline_trace_and_outcome(registry, monkeypatch, args, reason):
    from ouroboros._outcome_tool_errors import _classify_tool_errors
    from ouroboros.loop_tool_execution import _execute_single_tool
    from ouroboros.tools.tool_result import LegacyTextResultAdapter

    monkeypatch.setattr(LegacyTextResultAdapter, "from_text", classmethod(
        lambda *_args, **_kwargs: pytest.fail("selection refusal fell back to text classification")))
    row = _execute_single_tool(registry, {"id": "call", "function": {
        "name": "schedule_subagent", "arguments": json.dumps({
            "subagent_id": "api-scout", "objective": "Inspect source.", "expected_output": "Findings.",
            **args,
        }),
    }}, registry._ctx.drive_logs())

    assert reason in row["result"]
    assert row["is_error"] is True
    assert (row["tool_result"].status, row["tool_result"].code) == ("error", "TOOL_ARG_ERROR")
    assert row["result_meta"]["status"] == "argument_error"
    assert row["result_meta"]["tool_result_code"] == "TOOL_ARG_ERROR"
    logged = json.loads((registry._ctx.drive_logs() / "tools.jsonl").read_text(encoding="utf-8"))
    assert logged["is_error"] is True
    assert logged["status"] == "argument_error"
    buckets = _classify_tool_errors({"tool_calls": [logged]})
    assert len(buckets["unresolved"]) == 1
    assert registry._ctx.pending_events == []
