"""Selected inputs survive the real scheduling consumers without changing ordinary children."""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from ouroboros.contracts.task_contract import build_task_contract, task_input_sources
from ouroboros.tools import control
from ouroboros.tools.control_scheduling import _build_child_subagent_contract
from ouroboros.tools.registry import ToolRegistry
from supervisor.task_dispatch import build_scheduled_task_payload


def _settings(kind="api_model"):
    target = "openai/test-model" if kind == "api_model" else "codex=gpt-5.6-sol"
    return {"OUROBOROS_SUBAGENTS": {"enabled": True, "items": [{
        "subagent_id": "scout", "recommended_use": "Inspect declared evidence.",
        "route": {"kind": kind, "target_id": target}, "effort": "high",
    }]}}


@pytest.fixture
def registry(tmp_path, monkeypatch):
    import ouroboros.safety as safety

    monkeypatch.setattr(control, "load_settings", _settings)
    monkeypatch.setattr(safety, "check_safety", lambda *_args, **_kwargs: (True, ""))
    monkeypatch.setenv("OUROBOROS_MAX_SUBAGENT_DEPTH", "4")
    registry = ToolRegistry(repo_dir=tmp_path / "repo", drive_root=tmp_path / "data")
    registry._ctx.task_id = "parent"
    registry._ctx.pending_events = []
    registry._ctx.task_contract = build_task_contract({"task_contract": {
        "context": "PREVIOUS_CASE_CONTEXT", "notes": "PREVIOUS_CASE_NOTES",
        "review_notes": "PREVIOUS_CASE_REVIEW", "disabled_tools": ["web_search"],
        "predecessor_authority": {
            "task_id": "previous", "result": "PREVIOUS_CASE_RESULT", "authority_sha256": "a" * 64,
            "source": {"kind": "task_result", "task_id": "previous", "projection": "authority",
                       "read": {"tool": "get_task_result", "arguments": {
                           "task_id": "previous", "include_authority": True}}},
        },
        "resource_policy": {"allowed_origins": ["https://example.test:443"]},
        "deadline_at": "2099-01-01T00:00:00Z",
        "delegation_budget": {"intent_note": "May delegate within the assigned authority."},
    }})
    return registry


def _schedule(registry, **options):
    return registry.execute_result("schedule_subagent", {
        "subagent_id": "scout", "objective": "Assess the declared evidence.",
        "expected_output": "First position and sources.", **options,
    })


def test_public_selector_is_optional_and_closed(registry):
    schema = registry.get_schema_by_name("schedule_subagent")["function"]["parameters"]
    assert schema["properties"]["input_sources"]["enum"] == ["shared", "declared"]
    assert "input_sources" not in schema.get("required", [])


@pytest.mark.parametrize("context", ["COMMON_FACTS: specimen A has two marks.", ""])
def test_declared_child_excludes_prior_carriers_keeps_authority_and_restarts(
        registry, monkeypatch, context):
    import ouroboros.tools.control_scheduling as scheduling

    monkeypatch.setattr(scheduling, "_materialize_child_attachment_manifest",
                        lambda *_args, **_kwargs: pytest.fail("inherited inputs were materialized"))
    registry._ctx.task_contract["attachment_manifest"] = [{
        "status": "staged", "label": "PREVIOUS_CASE_ATTACHMENT", "ordinal": 0,
    }]
    registry._ctx.task_contract["attachment_manifest_ref"] = {
        "root": "artifact_store", "path": "PREVIOUS_CASE_ATTACHMENT_MANIFEST",
    }
    before = copy.deepcopy(registry._ctx.task_contract)
    result = _schedule(registry, input_sources="declared", context=context)
    assert (result.status, result.code) == ("ok", "OK"), result.text
    event = registry._ctx.pending_events[-1]
    selected = event["task_contract"]
    assert selected["input_sources"] == "declared"
    assert selected["context"] == context
    assert selected["attachment_manifest"] == []
    assert "attachment_manifest_ref" not in selected
    assert not {"notes", "review_notes"} & selected.keys()
    assert "PREVIOUS_CASE" not in json.dumps(selected)
    predecessor = selected["predecessor_authority"]
    assert predecessor["source"] == before["predecessor_authority"]["source"]
    assert predecessor["task_id"] == "previous"
    assert predecessor["authority_sha256"] == "a" * 64
    assert predecessor["omitted_fields"] == ["result"]
    from ouroboros.tool_access import lineage_task_ids

    lineage = {"parent_task_id": "parent", "root_task_id": "root"}
    before_access = SimpleNamespace(task_id="child", task_contract=before, task_metadata=lineage)
    after_access = SimpleNamespace(task_id="child", task_contract=selected, task_metadata=lineage)
    assert lineage_task_ids(before_access) == lineage_task_ids(after_access) == (
        "child", "parent", "root", "previous")
    for key in ("disabled_tools", "resource_policy", "deadline_at", "allowed_resources", "budget_profile"):
        assert selected[key] == before[key]
    assert selected["delegation_budget"]["intent_note"] == before["delegation_budget"]["intent_note"]
    assert registry._ctx.task_contract == before
    assert event["configured_subagent"]["effort"] == "high"
    assert event["requested_executor"] == "native"
    assert "effective_model_lane" not in event and "model" not in event

    saved = json.loads((registry._ctx.drive_root / "task_results" / f"{event['task_id']}.json")
                       .read_text(encoding="utf-8"))
    assert saved["task_contract"] == selected
    task = build_scheduled_task_payload({
        **event, "tid": event["task_id"], "parent_id": "parent",
        "task_context": context, "desc": event["objective"],
    })
    assert task_input_sources(task) == "declared"
    assert task["metadata"]["task_contract"] == selected
    assert build_task_contract(json.loads(json.dumps(task)))["input_sources"] == "declared"
    assert build_task_contract({"metadata": task["metadata"]})["input_sources"] == "declared"


@pytest.mark.parametrize("options", [{}, {"input_sources": "shared"}])
def test_ordinary_child_keeps_parent_context_notes_and_attachment_route(registry, monkeypatch, options):
    import ouroboros.tools.control_scheduling as scheduling

    calls = []
    def inherit(parent, *_args, **_kwargs):
        calls.append(parent)
        return {"attachment_manifest": []}, ""
    monkeypatch.setattr(scheduling, "_materialize_child_attachment_manifest", inherit)
    result = _schedule(registry, context="CHILD_REFERENCE", **options)
    assert result.status == "ok", result.text
    assert len(calls) == 1 and calls[0] == registry._ctx.task_contract
    event = registry._ctx.pending_events[-1]
    contract = event["task_contract"]
    assert contract["context"] == "PREVIOUS_CASE_CONTEXT"
    assert contract["notes"] == "PREVIOUS_CASE_NOTES"
    assert contract["review_notes"] == "PREVIOUS_CASE_REVIEW"
    assert "PREVIOUS_CASE_RESULT" in json.dumps(contract["predecessor_authority"])
    assert event["context"] == "CHILD_REFERENCE"
    assert ("input_sources" in contract) == bool(options)


def test_declared_selection_inherits_to_nested_children(registry):
    registry._ctx.task_contract["input_sources"] = "declared"
    registry._ctx.task_metadata = {"task_contract": registry._ctx.task_contract}
    registry._ctx.task_depth = 1
    result = _schedule(registry, context="EXPLICIT_NESTED_FACTS")
    assert result.status == "ok", result.text
    contract = registry._ctx.pending_events[-1]["task_contract"]
    assert contract["input_sources"] == "declared"
    assert contract["context"] == "EXPLICIT_NESTED_FACTS"
    assert "PREVIOUS_CASE" not in json.dumps(contract)


def test_selected_contract_survives_existing_queue_snapshot(registry, tmp_path):
    from supervisor import queue as task_queue

    result = _schedule(registry, input_sources="declared", context="COMMON_FACTS")
    assert result.status == "ok", result.text
    event = registry._ctx.pending_events[-1]
    task = build_scheduled_task_payload({
        **event, "tid": event["task_id"], "parent_id": "parent",
        "task_context": event["context"], "desc": event["objective"],
    })
    pending, running = [task], {}
    task_queue.init(tmp_path / "queue-data")
    task_queue.init_queue_refs(pending, running, {"value": 0})
    assert task_queue.persist_queue_snapshot(reason="first-input-regression") is True
    pending.clear()
    assert task_queue.restore_pending_from_snapshot() == 1
    assert pending[0]["task_contract"] == task["task_contract"]
    assert task_input_sources(pending[0]) == "declared"
    assert build_task_contract(pending[0])["context"] == "COMMON_FACTS"


@pytest.mark.parametrize("selection", [None, "", "unknown", "DECLARED", True, [], {}])
def test_malformed_selection_has_typed_reason_without_creating_child(registry, selection):
    result = _schedule(registry, input_sources=selection)
    assert (result.status, result.code) == ("error", "TOOL_ARG_ERROR")
    assert result.meta["reason"] == "INPUT_SOURCE_SELECTION_INVALID"
    assert registry._ctx.pending_events == []
    assert not (registry._ctx.drive_root / "task_results").exists()


def test_declared_parent_cannot_request_shared_descendant(registry):
    registry._ctx.task_contract["input_sources"] = "declared"
    result = _schedule(registry, input_sources="shared")
    assert (result.status, result.code) == ("error", "TOOL_ARG_ERROR")
    assert result.meta["reason"] == "INPUT_SOURCE_SELECTION_WIDENING"
    assert registry._ctx.pending_events == []


@pytest.mark.parametrize("selection", ["shared", "declared"])
def test_session_selected_memory_survives_canonical_queue_and_source(registry, monkeypatch, selection):
    from ouroboros.subagent_work_order import compile_external_work_order, work_order_source_projection
    from ouroboros import chronicle_view
    from supervisor import queue as task_queue

    monkeypatch.setattr(control, "load_settings", lambda: _settings("agent_session"))
    reference = {"text": "SHARED_LIFE_AND_EXACT_PARENT", "facts": {"external_fit": "unobserved"}, "snapshot_sha256": "a" * 64}
    calls = []
    def capture(ctx, task=None):
        calls.append(task)
        return reference
    monkeypatch.setattr(chronicle_view, "helper_memory_reference", capture)
    result = _schedule(registry, input_sources=selection, context="EXPLICIT_EVIDENCE")
    assert result.status == "ok", result.text
    event = registry._ctx.pending_events[-1]
    assert event["requested_executor"] == "harness"
    saved = json.loads((registry._ctx.drive_root / "task_results" / f"{event['task_id']}.json").read_text(encoding="utf-8"))
    task = build_scheduled_task_payload({**event, "tid": event["task_id"], "parent_id": "parent",
                                        "task_context": event["context"], "desc": event["objective"]})
    if selection == "shared":
        assert len(calls) == 1
        assert task["memory_reference"] == saved["memory_reference"] == reference
    else:
        assert calls == []
        assert not saved.get("memory_reference") and not task.get("memory_reference")
    rendered = compile_external_work_order(task)
    assert rendered.count(reference["text"]) == (1 if selection == "shared" else 0)
    assert "EXPLICIT_EVIDENCE" in rendered
    if selection == "declared":
        assert "PREVIOUS_CASE" not in rendered
        assert "disabled_tools" in rendered
    projection, reason = work_order_source_projection(task, 0, len(rendered))
    assert not reason and projection["text"] == rendered
    pending, running = [task], {}
    task_queue.init(registry._ctx.drive_root)
    task_queue.init_queue_refs(pending, running, {"value": 0})
    assert task_queue.persist_queue_snapshot(reason="memory-reference")
    pending.clear()
    assert task_queue.restore_pending_from_snapshot() == 1
    assert compile_external_work_order(pending[0]) == rendered


def test_contract_selection_is_additive_strict_and_has_consistent_precedence():
    ordinary = build_task_contract({"objective": "Work"})
    assert "input_sources" not in ordinary
    assert task_input_sources({"task_contract": ordinary}) == "shared"
    for carrier in (
        {"input_sources": "declared"},
        {"task_contract": {"input_sources": "declared"}},
        {"metadata": {"task_contract": {"input_sources": "declared"}}},
    ):
        assert task_input_sources(carrier) == "declared"
        assert build_task_contract(carrier)["input_sources"] == "declared"
    task = {"task_contract": {"input_sources": "declared"},
            "metadata": {"task_contract": {"input_sources": "shared"}}}
    assert task_input_sources(task) == build_task_contract(task)["input_sources"] == "declared"
    for invalid in (None, "", {}, "unknown"):
        with pytest.raises(ValueError, match="input_sources"):
            build_task_contract({"task_contract": {"input_sources": invalid}})
        with pytest.raises(ValueError, match="input_sources"):
            task_input_sources({"task_contract": {"input_sources": invalid}})


def test_child_builder_cannot_drop_inherited_selection():
    selected = _build_child_subagent_contract({
        "parent_contract": {"input_sources": "declared", "context": "PRIOR_CASE"},
        "context": "AUTHORED_COMMON_FACTS",
    })
    assert selected["input_sources"] == "declared"
    assert selected["context"] == "AUTHORED_COMMON_FACTS"
    with pytest.raises(ValueError, match="cannot widen"):
        _build_child_subagent_contract({
            "parent_contract": selected, "input_sources": "shared",
        })


@pytest.mark.parametrize("selection", ["shared", "declared"])
def test_scheduled_session_bootstrap_sends_canonical_selected_inputs_once(registry, monkeypatch, selection):
    from ouroboros import chronicle_view, claudexor_daemon
    from ouroboros.contracts.task_constraint import TaskConstraint
    from ouroboros.subagent_bootstrap import bootstrap_before_context
    from ouroboros.subagent_work_order import compile_external_work_order
    from ouroboros.tools.registry import ToolContext
    from tests._delegated_transport_shared import _LiveRunStub
    settings = _settings("agent_session")
    settings["OUROBOROS_SUBAGENTS"]["items"][0]["route"]["target_id"] = "some-route=ordinary-model"
    monkeypatch.setattr(control, "load_settings", lambda: settings)
    reference = {"text": "ONE_PARENT_MEMORY_COPY", "snapshot_sha256": "fixed", "facts": {}}
    monkeypatch.setattr(chronicle_view, "helper_memory_reference", lambda *_a, **_kw: reference)
    result = _schedule(registry, input_sources=selection, context="CHOSEN_CASE")
    assert result.status == "ok", result.text
    event = registry._ctx.pending_events[-1]
    task = build_scheduled_task_payload({**event, "tid": event["task_id"], "parent_id": "parent",
                                        "task_context": event["context"], "desc": event["objective"]})
    requests = []
    class Stub(_LiveRunStub):
        def start_run(self, request, *, idempotency_key=""):
            requests.append(request)
            return super().start_run(request, idempotency_key=idempotency_key)
    monkeypatch.setattr(claudexor_daemon, "ensure_owned_gateway", lambda: Stub())
    monkeypatch.setattr(chronicle_view, "helper_memory_reference", lambda *_a, **_kw: pytest.fail("compiled work recaptured memory"))
    child = ToolContext(repo_dir=registry._ctx.repo_dir, drive_root=registry._ctx.drive_root,
        task_id=task["id"], task_metadata=task["metadata"], task_contract=task["task_contract"],
        task_constraint=TaskConstraint(mode="local_readonly_subagent"))
    wake = json.loads(bootstrap_before_context(child, task, SimpleNamespace(blocked=False)))
    assert wake["status"] == "configured_session_started", wake
    assert len(requests) == 1
    assert requests[0]["prompt"] == compile_external_work_order(task)
    assert json.dumps(requests[0]).count("ONE_PARENT_MEMORY_COPY") == (1 if selection == "shared" else 0)
    assert "CHOSEN_CASE" in requests[0]["prompt"]
    assert "disabled_tools" in requests[0]["prompt"]
