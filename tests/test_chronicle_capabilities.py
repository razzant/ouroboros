"""Who may use the chronicle tools, and the focus signature of memory writes.

The host signs every memory write with the writer's focus (only a configured agent
session child is a nanny; an API child is a child). The sets: the integrating mind writes chronicle
pages, a presence reads and marks but writes no page, a delegated child reads,
writes knowledge and marks and drafts pages and parts in its own name (its sets
are pinned in test_chronicle_tools.py, its drafts in
test_child_chronicle_drafts.py), a native reviewer sees none of the memory writers or
readers, and only the short mark receipt is exempt from truncation.
"""
from __future__ import annotations

import json

from ouroboros import knowledge as knowledge_store
from ouroboros.consciousness_authority import consciousness_origin_metadata
from ouroboros.consciousness_wake import wake_task_metadata
from ouroboros.knowledge import UNKNOWN_STAMP, focus_signature
from ouroboros.tools import knowledge as tools
from ouroboros.tools.registry import ToolContext

CHILD = {"delegation_role": "subagent", "parent_task_id": "root0001", "root_task_id": "root0001",
         "configured_subagent": {}}


def _ctx(tmp_path, task_id, **fields):
    return ToolContext(repo_dir=tmp_path, drive_root=tmp_path, task_id=task_id, **fields)


def _history(tmp_path):
    shelf = knowledge_store.resolve_knowledge_address(tmp_path, "topic", "global").shelf
    path = shelf.parent / "knowledge_history.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _role(tmp_path, **fields):
    return focus_signature(_ctx(tmp_path, "task0001", **fields))["focus"]["role"]


def test_child_knowledge_write_carries_the_child_focus_in_history(tmp_path):
    child = _ctx(tmp_path, "kid00001", task_metadata=dict(CHILD), current_chat_id=1)
    assert "saved" in tools._knowledge_write(child, "people/rowan", "Rowan prefers short reports.", mode="append")
    root = _ctx(tmp_path, "root0001", current_chat_id=1)
    assert "saved" in tools._knowledge_write(root, "people/ada", "Ada reviews on Fridays.", mode="append")
    rows = {row["task_id"]: row for row in _history(tmp_path)}
    assert rows["kid00001"]["focus"] == {"role": "child", "task_id": "kid00001", "parent_task_id": "root0001",
                                         "root_task_id": "root0001", "chat_id": 1}
    assert rows["root0001"]["focus"]["role"] == "root" and rows["root0001"]["focus"]["parent_task_id"] == ""
    # A seam that names no focus keeps the honest unknown, like the other history stamps.
    address = knowledge_store.resolve_knowledge_address(tmp_path, "people/lee", "global")
    assert knowledge_store.write_knowledge_note(address, "Lee.", mode="append").ok
    assert {row["topic"]: row["focus"] for row in _history(tmp_path)}["people/lee"] == UNKNOWN_STAMP


def test_configured_agent_session_child_is_a_nanny_and_an_api_child_is_a_child(tmp_path):
    session = {**CHILD, "configured_subagent": {"route": {"kind": "agent_session"}}}
    assert _role(tmp_path, task_metadata=session) == "nanny"
    api = {**CHILD, "configured_subagent": {"route": {"kind": "api_model"}}}
    assert _role(tmp_path, task_metadata=api) == "child"
    # The default empty snapshot every API child carries is not a nanny.
    assert _role(tmp_path, task_metadata=dict(CHILD)) == "child"


def test_wake_is_consciousness_and_the_root_it_starts_is_a_root(tmp_path):
    wake = wake_task_metadata("observe", "heartbeat")
    assert _role(tmp_path, task_metadata=wake) == "consciousness"
    started = consciousness_origin_metadata(wake)
    assert started["initiator"] == "consciousness"
    assert _role(tmp_path, task_metadata=started) == "root"
    # A child of a wake is a child: the delegated role is decided before the ledger category.
    assert _role(tmp_path, task_metadata={**wake, **CHILD}) == "child"


def test_presence_main_and_root_roles(tmp_path):
    assert _role(tmp_path, task_metadata={"presence": {"binding_id": "b1"}}) == "presence"
    assert _role(tmp_path, task_metadata={"presence": {"binding_id": "b1"}}, is_direct_chat=True) == "presence"
    assert _role(tmp_path, is_direct_chat=True) == "main"
    assert _role(tmp_path) == "root"


def test_signature_carries_task_lineage_chat_and_observed_route(tmp_path):
    ctx = _ctx(tmp_path, "kid00001", task_metadata={**CHILD, "chat_id": 7})
    ctx._accumulated_usage = {"provider": "openrouter", "resolved_model": "openai/gpt-5.6"}
    signature = focus_signature(ctx)
    assert signature["kind"] == "mind" and signature["task_id"] == "kid00001"
    assert signature["focus"]["chat_id"] == 7 and signature["focus"]["root_task_id"] == "root0001"
    assert signature["route"] == {"provider": "openrouter", "model": "openai/gpt-5.6"}
    ctx.current_chat_id = 3
    assert focus_signature(ctx)["focus"]["chat_id"] == 3
    assert focus_signature(_ctx(tmp_path, "solo0001"))["focus"]["root_task_id"] == "solo0001"


# --- the chronicle tools in the capability sets ---------------------------------------------------

MEMORY_TOOLS = ("chronicle_write", "memory_read", "memory_mark")


def test_catalog_exports_the_three_tools_once_without_an_author_field():
    names = [entry.name for entry in tools.get_tools()]
    assert all(names.count(name) == 1 for name in MEMORY_TOOLS)
    for entry in tools.get_tools():
        if entry.name in MEMORY_TOOLS:
            parameters = entry.schema["parameters"]
            assert parameters["additionalProperties"] is False and "author" not in parameters["properties"]


def test_integrating_mind_writes_pages_and_presence_reads_and_marks_only():
    from ouroboros.tool_capabilities import (
        COGNITIVE_MEMORY_TOOL_NAMES, CORE_TOOL_NAMES, READ_ONLY_PARALLEL_TOOLS, UNTRUNCATED_TOOL_RESULTS,
        tool_result_limit,
    )

    assert set(MEMORY_TOOLS) <= CORE_TOOL_NAMES
    assert {"memory_read", "memory_mark"} <= COGNITIVE_MEMORY_TOOL_NAMES
    assert "chronicle_write" not in COGNITIVE_MEMORY_TOOL_NAMES
    # Only the short mark receipt is never cut; a read pages itself under its own cap.
    assert "memory_mark" in UNTRUNCATED_TOOL_RESULTS
    assert not {"chronicle_write", "memory_read"} & UNTRUNCATED_TOOL_RESULTS
    assert tool_result_limit("memory_read") == 80_000 > tool_result_limit("chronicle_write")
    assert "memory_read" in READ_ONLY_PARALLEL_TOOLS
    assert not {"chronicle_write", "memory_mark"} & READ_ONLY_PARALLEL_TOOLS


def test_presence_ceiling_reads_and_marks_memory_but_writes_no_page():
    from ouroboros.presence_authority import build_presence_capability_ceiling, presence_ceiling_allows_tool
    from ouroboros.presence_capabilities import PresenceProfileResolution
    from ouroboros.presence_runtime import ResolvedPresenceRuntime

    resolution = PresenceProfileResolution(
        active=(), missing_required=(), missing_optional=(), orphaned=(),
        runtime=ResolvedPresenceRuntime("main", 10, 10, False), profile_fingerprint="a" * 64,
        selection_fingerprint="b" * 64, required_selections_present=True)
    ceiling = build_presence_capability_ceiling(skill_name="community-helper", skill_content_hash="c" * 64,
                                                state_fingerprint="d" * 64, resolution=resolution)
    assert presence_ceiling_allows_tool(ceiling, "memory_read")
    assert presence_ceiling_allows_tool(ceiling, "memory_mark")
    assert not presence_ceiling_allows_tool(ceiling, "chronicle_write")


def test_native_reviewer_sees_none_of_the_memory_tools_but_keeps_its_inspection_tools(tmp_path):
    from ouroboros.review_native_episode import inspection_registry
    from ouroboros.tool_capabilities import LOCAL_READONLY_SUBAGENT_TOOL_NAMES

    # The child set the reviewer is cut from now carries knowledge_write, memory_read, memory_mark and
    # chronicle_write (a child's drafts only); the reviewer's disabled_tools takes them away with every
    # other non-inspection name.
    hidden = set(MEMORY_TOOLS) | {"knowledge_write"}
    assert hidden <= LOCAL_READONLY_SUBAGENT_TOOL_NAMES
    data = tmp_path / "data"
    data.mkdir()
    registry, _ctx, schemas = inspection_registry(str(tmp_path), data)
    names = {schema["function"]["name"] for schema in schemas}
    assert "read_file" in names and not hidden & names
    for name in hidden:
        assert registry.get_schema_by_name(name) is None
    assert registry.get_schema_by_name("read_file") is not None
