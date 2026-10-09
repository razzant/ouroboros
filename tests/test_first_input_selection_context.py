"""Positive/negative source composition and unchanged ordinary child continuity."""
from __future__ import annotations

import pytest

from tests.test_doc_context import _make_env_and_memory


@pytest.mark.parametrize("selection", [None, "shared", "declared"])
def test_capture_selects_channels_without_removing_governance_or_own_process(tmp_path, monkeypatch, selection):
    from ouroboros import context

    env, memory = _make_env_and_memory(tmp_path)
    child = {
        "id": "child", "type": "task", "delegation_role": "subagent",
        "text": "DECLARED_QUESTION", "context": "DECLARED_COMMON_FACTS",
        "configured_subagent": {"route": {"kind": "api_model"}},
        "task_contract": {"constraints": "DECLARED_CONSTRAINT"},
    }
    if selection is not None:
        child["task_contract"]["input_sources"] = selection
    calls = []

    def source(name, result):
        def read(*_args, **_kwargs):
            calls.append(name)
            return result
        return read

    omitted = ["SHARED_BIOGRAPHY", "PROJECT_KNOWLEDGE_AND_WORKPAD", "SHARED_HEALTH",
               "MEMORY_REGISTRY", "INSTALLED_SKILLS", "DRIVE_STATE", "SHARED_REVIEW"]
    monkeypatch.setattr(context, "build_memory_sections", source(omitted[0], [omitted[0]]))
    monkeypatch.setattr(context, "build_knowledge_sections", source(omitted[1], [omitted[1]]))
    for name, marker in zip(("build_health_invariants", "_build_registry_digest",
                             "_build_installed_skills_section", "_drive_state_section"), omitted[2:6]):
        monkeypatch.setattr(context, name, source(marker, marker))
    monkeypatch.setattr(memory, "recent_activity_sections", source("OWN_PROCESS", ["OWN_PROCESS"]))
    core = context._capture_context_core(env, memory, child, source(omitted[6], omitted[6]), None)
    text = (core.base_prompt + core.bible_md + core.semi_stable_text + core.dynamic_head_text + core.dynamic_text
            + core.user_content_json)
    assert "You are Ouroboros." in text and "Principle 0: Agency" in text
    assert "DECLARED_QUESTION" in text and "DECLARED_COMMON_FACTS" in text
    assert "DECLARED_CONSTRAINT" in text and "OWN_PROCESS" in text
    # A child's memory view holds no knowledge (it reads it with knowledge_read), whatever the selection.
    knowledge = omitted[1]
    assert knowledge not in text and knowledge not in calls
    for marker in omitted:
        if marker != knowledge:
            assert (marker in text) == (selection != "declared")
            assert (marker in calls) == (selection != "declared")
    assert bool(core.memory_view_json) == (selection != "declared")  # a declared child carries no view
    assert ("Input source selection" in text) == (selection == "declared")


@pytest.mark.parametrize("explicit", [None, {"knowledge": True}])
def test_a_child_loads_knowledge_only_when_its_view_holds_it(tmp_path, monkeypatch, explicit):
    from ouroboros import context

    env, memory = _make_env_and_memory(tmp_path)
    child = {"id": "child", "type": "task", "delegation_role": "subagent", "text": "Q",
             "configured_subagent": {"route": {"kind": "api_model"}}}
    if explicit is not None:
        child["memory_view"] = explicit
    monkeypatch.setattr(context, "build_knowledge_sections", lambda *_a, **_kw: ["KNOWLEDGE_MARKER"])
    core = context._capture_context_core(env, memory, child, None, None)
    assert ("KNOWLEDGE_MARKER" in core.dynamic_head_text) == (explicit is not None)
    assert "KNOWLEDGE_MARKER" not in core.semi_stable_text + core.dynamic_text


def test_selected_runtime_omits_other_task_narratives_but_keeps_authority(tmp_path, monkeypatch):
    from ouroboros import context, task_tree_ledger

    env, _memory = _make_env_and_memory(tmp_path)
    monkeypatch.setattr(context, "_scheduled_tasks_digest", lambda *_: {"summary": "OTHER_SCHEDULE"})
    monkeypatch.setattr(context, "_project_room_fact", lambda *_: {"summary": "OTHER_ROOM"})
    monkeypatch.setattr(context, "_delegation_capability_fact", lambda *_: {"summary": "OTHER_DELEGATION_HISTORY"})
    monkeypatch.setattr(context, "official_update_projection", lambda *_: {"letter": {"text": "OTHER_UPDATE_LETTER"}})
    monkeypatch.setattr(task_tree_ledger, "tree_ledger_tail_digest", lambda *_a, **_kw: "OTHER_TREE_CONCLUSION")
    task = {
        "id": "child", "root_task_id": "root", "delegation_role": "subagent",
        "context": "COMMON_EVIDENCE", "task_constraint": {"mode": "local_readonly_subagent"},
        "task_contract": {"input_sources": "declared", "disabled_tools": ["write_file"],
                          "deadline_at": "2099-01-01T00:00:00Z"},
        "metadata": {"current_chat": {"running_tasks": ["OTHER_CHAT"]},
                     "project_routing_manifest": {"text": "OTHER_ROUTING"},
                     "routing_contract": {"text": "OTHER_ROUTING_CONTRACT"}},
    }
    selected = context.build_runtime_section(env, task)
    assert "OTHER_" not in selected
    for value in ("COMMON_EVIDENCE", "local_readonly_subagent", "write_file", "2099-01-01"):
        assert value in selected
    task["task_contract"].pop("input_sources")
    shared = context.build_runtime_section(env, task)
    for value in ("OTHER_SCHEDULE", "OTHER_ROOM", "OTHER_TREE_CONCLUSION", "OTHER_CHAT", "OTHER_ROUTING", "OTHER_UPDATE_LETTER"):
        assert value in shared


@pytest.mark.parametrize("overrides", [
    {"delegation_role": "root"},
    {"configured_subagent": {"route": {"kind": "weird"}}},
    {"configured_subagent": {}},
    {"id": ""},
])
def test_unsupported_builder_task_class_is_not_silently_shared(tmp_path, overrides):
    from ouroboros import context

    env, memory = _make_env_and_memory(tmp_path)
    task = {"id": "child", "delegation_role": "subagent",
            "configured_subagent": {"route": {"kind": "api_model"}},
            "task_contract": {"input_sources": "declared"}, **overrides}
    with pytest.raises(ValueError, match="INPUT_SOURCE_SELECTION_UNSUPPORTED"):
        context._capture_context_core(env, memory, task, None, None)


@pytest.mark.parametrize("kind", ["api_model", "agent_session"])
def test_both_scheduled_child_routes_compose_declared_inputs(tmp_path, kind):
    """A configured-session child's nanny composes the declared core exactly as an API
    child does; the receipt rides the capture for either route."""
    from ouroboros import context

    env, memory = _make_env_and_memory(tmp_path)
    task = {"id": "child", "type": "task", "delegation_role": "subagent", "text": "DECLARED_QUESTION",
            "context": "DECLARED_COMMON_FACTS", "configured_subagent": {"route": {"kind": kind}},
            "task_contract": {"input_sources": "declared"}}
    core = context._capture_context_core(env, memory, task, None, None)
    text = core.base_prompt + core.bible_md + core.semi_stable_text + core.dynamic_text + core.user_content_json
    assert "DECLARED_QUESTION" in text and "DECLARED_COMMON_FACTS" in text
    assert "Input source selection" in text


def test_workorder_projection_and_digest_retain_selection_receipt():
    from hashlib import sha256
    from ouroboros.subagent_work_order import (
        compile_external_work_order, work_order_fingerprint, work_order_source_projection,
    )

    task = {"id": "child", "objective": "DECLARED_QUESTION", "context": "COMMON_EVIDENCE",
            "task_contract": {"input_sources": "declared", "context": "COMMON_EVIDENCE"}}
    rendered = compile_external_work_order(task)
    assert "INPUT SOURCE SELECTION" in rendered and "omitted_automatic" in rendered
    assert "COMMON_EVIDENCE" in rendered and "DECLARED_QUESTION" in rendered
    assert work_order_fingerprint(task) == sha256(rendered.encode()).hexdigest()
    projected, reason = work_order_source_projection(task, 0, len(rendered))
    assert reason == "" and projected is not None
    assert projected["text"] == rendered
    assert projected["complete_sha256"] == work_order_fingerprint(task)
    # A configured-session leaf's compiled work order carries the same receipt, naming
    # the session's two recipients.
    session_task = {**task, "configured_subagent": {"route": {"kind": "agent_session"}}}
    assert "two recipients" in compile_external_work_order(session_task)
    # Existing callers and work-order hashes gain no empty/default receipt.
    task["task_contract"].pop("input_sources")
    assert "INPUT SOURCE SELECTION" not in compile_external_work_order(task)


def test_declared_receipt_records_sources_without_prescribing_collaboration_order():
    from ouroboros.subagent_work_order import input_source_selection_receipt

    receipt = input_source_selection_receipt({"task_contract": {"input_sources": "declared"}})
    assert receipt["included"] and receipt["omitted_automatic"]
    assert "Entire task" in receipt["lifetime"]
    assert "no collaboration order" in receipt["collaboration"]
    for key in ("collaboration", "later_inputs"):
        assert "forward_to_worker" not in receipt[key]
        assert "first-position handoff" not in receipt[key]
        assert "before asking" not in receipt[key]
    assert "model-send/tool-source" in receipt["later_inputs"]
    # The receipt names what selection keeps and whom it reaches: the inherited intent
    # note is parent-authored advice that survives, and a configured-session child has
    # two recipients (nanny core, compiled leaf work order) with harness inputs unobserved.
    assert any("delegation_budget.intent_note" in item for item in receipt["included"])
    assert "two recipients" in receipt["limitations"] and "compiled work order" in receipt["limitations"]
    assert "unsupported" not in receipt["limitations"].lower()


def test_declared_tool_descriptions_leave_exchange_strategy_to_assignment():
    from ouroboros.tools.control import get_tools

    entry = next(tool for tool in get_tools() if tool.name == "schedule_subagent")
    schema = entry.schema
    selection = schema["parameters"]["properties"]["input_sources"]["description"]
    assert "assignment defines" in selection
    assert "prescribes no exchange sequence or transport" in selection
    assert "forward_to_worker" not in selection
    # The field carries the one cue for choosing the selector; the tool description no
    # longer repeats the selector paragraph (one SSOT, fewer cached-prefix bytes).
    assert "for an independent first position" in selection
    assert "input_sources=declared" not in schema["description"]
    assert "Retain the first position via forward_to_worker before" not in schema["description"]
