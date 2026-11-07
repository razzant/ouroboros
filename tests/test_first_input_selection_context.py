"""Positive/negative source composition and unchanged ordinary child continuity."""
from __future__ import annotations

import pytest

from tests.test_doc_context import _make_env_and_memory


@pytest.mark.parametrize("words", [False, True], ids=["scheduled-before", "words-fact"])
@pytest.mark.parametrize("selection", [None, "shared", "declared"])
def test_capture_selects_channels_without_removing_governance_or_own_process(tmp_path, monkeypatch, selection, words):
    from ouroboros import context
    from ouroboros.subagent_work_order import DECLARED_WORDS_FACT

    env, memory = _make_env_and_memory(tmp_path)
    child = {
        "id": "child", "type": "task", "delegation_role": "subagent",
        "text": "DECLARED_QUESTION", "context": "DECLARED_COMMON_FACTS",
        "configured_subagent": {"route": {"kind": "api_model"}},
        "task_contract": {"constraints": "DECLARED_CONSTRAINT"},
        "metadata": {"governing_owner_words": [{"text": "OWNER_ASKED", "source": "initial_user", "task_id": "root"}],
                     **({DECLARED_WORDS_FACT: True} if words else {})},
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
    for name, marker in zip(("build_health_invariants", "_build_registry_section",
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
    # One builder, one view: a declared child's draws the words, the marks and no story or room page.
    import json

    spec = json.loads(core.memory_view_json)["spec"]
    assert spec["marks"] == "room_and_global" and spec["story"] == (selection != "declared")
    assert spec["room_page"] == (selection != "declared") and not spec["room_lanes"] and not spec["knowledge"]
    assert ("Input source selection" in core.dynamic_text) == (selection == "declared")
    # The owner's words ride the view's block for a shared child and for a declared child whose schedule recorded
    # that its selection carries them; never the dynamic text. Every memory channel above stays shut either way.
    assert ("OWNER_ASKED" in core.memory_view_json) == (selection != "declared" or words)
    assert spec["owner_words"] == (selection != "declared" or words) and "OWNER_ASKED" not in core.dynamic_text


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


def test_a_parents_memory_view_field_rides_the_schedule_to_the_view_its_child_builds(tmp_path, monkeypatch):
    """The producer end to end: ``schedule_subagent(memory_view=…)`` is carried by the schedule's record and
    the supervisor's payload to the start the worker builds. A field the view knows widens the child (the
    whole knowledge base at its birth, no outline); a value it does not know is carried too, refused at the start
    with the role default and one event, and the child is born as any child. A view that turns my human's words or
    the marks off, or names a field the description does not offer, is refused where it enters and nothing is
    queued; a plain call still births a child holding the words."""
    import dataclasses
    import json
    import queue

    from ouroboros import memory_view as mv
    from ouroboros.context import build_llm_messages
    from ouroboros.knowledge import resolve_knowledge_address, write_knowledge_note
    from ouroboros.tools import control
    from ouroboros.tools.control_scheduling import _schedule_task
    from ouroboros.tools.registry import ToolContext
    from supervisor.task_dispatch import build_scheduled_task_payload
    from tests.test_available_subagents_runtime import _api_row, _settings

    monkeypatch.setattr(control, "load_settings", lambda: _settings(_api_row()))
    monkeypatch.setenv("OUROBOROS_MAX_SUBAGENT_DEPTH", "4")
    env, memory = _make_env_and_memory(tmp_path)
    root = memory.drive_root
    assert write_knowledge_note(resolve_knowledge_address(root, "people/owner", "global"),
                                "---\ntitle: people/owner\nsummary: WHOLE_INDEX_SUMMARY\n---\n# Owner\n").ok
    parent = ToolContext(repo_dir=env.repo_dir, drive_root=root, current_task_type="task", current_chat_id=1)
    parent.task_id, parent.event_queue = "root0001", queue.Queue()
    parent.task_metadata = {"root_task_id": "root0001",
                            "governing_owner_words": [{"text": "OWNER_ASKED", "source": "initial_user", "task_id": "root0001"}]}

    def call(**extra):
        return _schedule_task(parent, subagent_id="api-builder", objective="Count shelf one",
                              expected_output="a list", memory_mode="forked", **extra)

    def scheduled(**extra):
        result = call(**extra)
        assert not result.startswith("⚠️"), result
        event = parent.event_queue.get_nowait()
        return build_scheduled_task_payload({**event, "tid": event["task_id"], "parent_id": parent.task_id,
                                             "text": event["objective"], "desc": event["objective"]})

    def born(payload):
        messages, cap = build_llm_messages(env=env, memory=memory, task=payload)
        return messages[0]["content"][-1]["text"], cap["memory_view"]["spec"]

    def refusals():
        rows = [json.loads(line) for line in (root / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()]
        return [row for row in rows if row.get("type") == "context_memory_view_spec_refused"]

    plain = scheduled()
    assert "memory_view" not in plain and "memory_view" not in plain["metadata"]  # omitted: nothing carried
    widened = scheduled(memory_view={"knowledge": True})
    assert widened["memory_view"] == {"knowledge": True} == widened["metadata"]["memory_view"]
    assert mv.view_spec_for_task(widened, root) == dataclasses.replace(mv.view_spec_for_task(plain, root), knowledge=True)
    changing, spec = born(widened)
    assert spec["knowledge"] is True and "WHOLE_INDEX_SUMMARY" in changing and "## Knowledge base" in changing
    assert "## Knowledge outline" not in changing and "knowledge (overview, index, project journal and workpad)" in changing
    assert refusals() == []
    # The other direction: a value the view does not know rides the same way and is refused where it is read.
    malformed = scheduled(memory_view={"knowledge": "yes"})
    assert malformed["memory_view"] == {"knowledge": "yes"}  # carried as given: the start, not the schedule, judges it
    assert mv.view_spec_for_task(malformed, root) == mv.view_spec_for_task(plain, root)
    assert [(row["task_id"], row["reason"]) for row in refusals()] == [(malformed["id"], "knowledge is true or false")]
    changing, spec = born(malformed)
    assert spec == born(plain)[1] and "## Knowledge outline" in changing and "## Knowledge base" not in changing
    assert len(refusals()) == 2 and refusals()[-1]["task_id"] == malformed["id"]  # the start: one more event, no more
    # Refused where the field enters: a view that turns my human's words or the marks off (they are never turned off), or
    # names a field the description does not offer (a role, a room, a work room level); nothing is queued by any of them.
    for view, reason in (({"owner_words": False}, "only widens a child's view; it keeps my human's words and the marks"),
                         ({"marks": "none"}, "only widens"), ({"work_room_levels": "two"}, "fields are story, room_page")):
        refused = call(memory_view=view)
        assert refused.startswith("⚠️ TOOL_ARG_ERROR (schedule_subagent): memory_view") and reason in refused, view
        assert parent.event_queue.empty(), view
    changing, spec = born(plain)  # a plain call births a child holding the words and the marks in force
    assert spec["owner_words"] is True and spec["marks"] == "room_and_global" and "OWNER_ASKED" in changing, spec
    assert len(refusals()) == 2


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
    assert any("own-body DESIGN navigation or source-gap disclosure" in item for item in receipt["included"])
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
