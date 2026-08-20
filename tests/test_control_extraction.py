"""Structural contracts for the semantic-no-op control tool extraction."""

from __future__ import annotations

import ast
import hashlib
import json
import pathlib

from ouroboros.tool_module_inventory import (
    build_frozen_tool_manifest,
    discover_tool_module_inventory,
    load_frozen_tool_modules,
)
from ouroboros.tools import (
    control,
    control_events,
    control_routing,
    control_runtime,
    control_scheduling,
    control_subagent_spec,
    control_task_results,
)
from supervisor.update_merge_policy import HOT_CODE_PATHS

REPO = pathlib.Path(__file__).parents[1]
TOOLS = REPO / "ouroboros" / "tools"

_LEAVES = (
    control_events,
    control_routing,
    control_subagent_spec,
    control_scheduling,
    control_runtime,
    control_task_results,
)

_MOVED_OWNERS = {
    "_PROMOTE_CONFIRM_POLL_SEC": control_events,
    "_PROMOTE_CONFIRM_TIMEOUT_SEC": control_events,
    "_SCHEDULE_EMIT_LOCK": control_events,
    "_emit_and_wait_for_routing": control_events,
    "_emit_control_event": control_events,
    "_promotion_pool_disabled_from_snapshot": control_events,
    "_routing_status_root": control_events,
    "_wait_for_promotion_admission": control_events,
    "_wait_for_routing_annotation": control_events,
    "_attach_origin_from_metadata": control_routing,
    "_attach_swarm_intent": control_routing,
    "_cached_swarm_handoff": control_routing,
    "_finish_swarm_handoff": control_routing,
    "_list_projects": control_routing,
    "_promote_chat_to_task": control_routing,
    "_route_to_project": control_routing,
    "_steer_task": control_routing,
    "RETIRED_SCHEDULE_PARAMS": control_subagent_spec,
    "VALID_SUBTASK_MEMORY_MODES": control_subagent_spec,
    "_INTERNAL_SCHEDULE_OPTIONS": control_subagent_spec,
    "_validated_schedule_fields": control_subagent_spec,
    "schedule_subagent_param_names": control_subagent_spec,
    "schedule_subagent_properties": control_subagent_spec,
    "_build_acting_constraint": control_scheduling,
    "_build_child_subagent_contract": control_scheduling,
    "_capability_mismatch_message": control_scheduling,
    "_earliest_deadline_at": control_scheduling,
    "_emit_swarm_fanout": control_scheduling,
    "_finalize_schedule_emission": control_scheduling,
    "_inherited_workspace_from_active_repo": control_scheduling,
    "_populate_subagent_event_extras": control_scheduling,
    "_prepare_child_drive": control_scheduling,
    "_record_scheduled_subagent": control_scheduling,
    "_resolve_executor_ref": control_scheduling,
    "_schedule_task": control_scheduling,
    "_select_subagent_constraint": control_scheduling,
    "_subagent_slot_note": control_scheduling,
    "_chat_history": control_runtime,
    "_evolution_restart_block_reason": control_runtime,
    "_promote_to_stable": control_runtime,
    "_request_deep_self_review": control_runtime,
    "_request_restart": control_runtime,
    "_send_user_message": control_runtime,
    "_set_tool_timeout": control_runtime,
    "_switch_model": control_runtime,
    "_toggle_consciousness": control_runtime,
    "_toggle_evolution": control_runtime,
    "_update_identity": control_runtime,
    "_update_scratchpad": control_runtime,
    "_UNMINTED_WAIT_GRACE_SEC": control_task_results,
    "_children_roster_projection": control_task_results,
    "_count_live_sibling_children": control_task_results,
    "_get_task_result": control_task_results,
    "_subtask_outcome_summary": control_task_results,
    "_unminted_wait_ids": control_task_results,
    "_wait_attention_poll": control_task_results,
    "_wait_for_task": control_task_results,
    "_wait_for_tasks": control_task_results,
    "cache_horizon_note": control_task_results,
    "disclosable_capability_delta": control_task_results,
}

# The catalog owner keeps the catalog and the one description hoisted out of it.
_STAYED = ("_PROMOTE_CHAT_DESCRIPTION", "get_tools", "log")


def test_control_leaves_are_non_catalog_owners_without_control_backedges(tmp_path):
    for module in _LEAVES:
        source_path = pathlib.Path(module.__file__)
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        assert not any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "get_tools"
            for node in tree.body
        )
        assert not any(
            isinstance(node, ast.ImportFrom)
            and node.module == "ouroboros.tools.control"
            for node in ast.walk(tree)
        )
        assert not any(
            isinstance(node, ast.Import)
            and any(alias.name == "ouroboros.tools.control" for alias in node.names)
            for node in ast.walk(tree)
        )

    source_inventory = discover_tool_module_inventory(TOOLS)
    assert "control" in source_inventory.tool_modules
    for module in _LEAVES:
        assert module.__name__.rsplit(".", 1)[-1] not in source_inventory.tool_modules
    manifest = tmp_path / "_frozen_tool_modules.v1.json"
    build_frozen_tool_manifest(TOOLS, manifest)
    assert load_frozen_tool_modules(manifest) == source_inventory.tool_modules


def test_control_catalog_schema_bytes_and_handler_owners_are_stable():
    entries = control.get_tools()
    assert tuple(entry.name for entry in entries) == (
        "set_tool_timeout", "request_restart", "promote_to_stable", "promote_chat_to_task",
        "ensure_project_scope", "list_projects", "route_to_project", "steer_task",
        "schedule_subagent", "request_deep_self_review", "chat_history", "update_scratchpad",
        "send_user_message", "update_identity", "toggle_evolution", "toggle_consciousness",
        "switch_model", "get_task_result", "wait_task", "wait_tasks",
    )
    schema_bytes = json.dumps(
        [entry.schema for entry in entries],
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    # Re-pinned at the v6.105.0 adoption: upstream rewrote three schedule_subagent
    # descriptions (objective states an OUTCOME, context is the delegated run's WORK
    # ORDER, an explicit model lane OVERRIDES dispatch policy). The schema bytes are
    # a prompt contract, so a deliberate upstream wording change moves this hash —
    # the pin exists to make that visible, not to forbid it.
    assert hashlib.sha256(schema_bytes).hexdigest() == (
        "a4acd52899af73a0ba0d9a609652febff3df93ed25306507ab76c72cdea7fc4d"
    )
    assert {
        entry.name: (entry.handler.__module__, entry.handler.__name__)
        for entry in entries
    } == {
        "set_tool_timeout": ("ouroboros.tools.control_runtime", "_set_tool_timeout"),
        "request_restart": ("ouroboros.tools.control_runtime", "_request_restart"),
        "promote_to_stable": ("ouroboros.tools.control_runtime", "_promote_to_stable"),
        "promote_chat_to_task": ("ouroboros.tools.control_routing", "_promote_chat_to_task"),
        "ensure_project_scope": ("ouroboros.tools.control_delegation", "_ensure_project_scope"),
        "list_projects": ("ouroboros.tools.control_routing", "_list_projects"),
        "route_to_project": ("ouroboros.tools.control_routing", "_route_to_project"),
        "steer_task": ("ouroboros.tools.control_routing", "_steer_task"),
        "schedule_subagent": ("ouroboros.tools.control_scheduling", "_schedule_task"),
        "request_deep_self_review": ("ouroboros.tools.control_runtime", "_request_deep_self_review"),
        "chat_history": ("ouroboros.tools.control_runtime", "_chat_history"),
        "update_scratchpad": ("ouroboros.tools.control_runtime", "_update_scratchpad"),
        "send_user_message": ("ouroboros.tools.control_runtime", "_send_user_message"),
        "update_identity": ("ouroboros.tools.control_runtime", "_update_identity"),
        "toggle_evolution": ("ouroboros.tools.control_runtime", "_toggle_evolution"),
        "toggle_consciousness": ("ouroboros.tools.control_runtime", "_toggle_consciousness"),
        "switch_model": ("ouroboros.tools.control_runtime", "_switch_model"),
        "get_task_result": ("ouroboros.tools.control_task_results", "_get_task_result"),
        "wait_task": ("ouroboros.tools.control_task_results", "_wait_for_task"),
        "wait_tasks": ("ouroboros.tools.control_task_results", "_wait_for_tasks"),
    }
    assert {entry.name: entry.timeout_sec for entry in entries if entry.timeout_sec != 360} == {
        "wait_task": 7200, "wait_tasks": 7200,
    }


def test_control_facade_reexports_every_moved_identity():
    """``tools/control.py`` keeps the exact objects, so plan review, the join
    ledger, delegation, review evidence and the tests that reach for a private
    helper see no identity change."""
    for name, owner in _MOVED_OWNERS.items():
        assert hasattr(control, name), name
        assert getattr(control, name) is getattr(owner, name), name
    owned = {name for module in _LEAVES for name in vars(module)}
    assert set(_MOVED_OWNERS) <= owned


def test_the_catalog_owner_kept_only_the_catalog():
    tree = ast.parse(pathlib.Path(control.__file__).read_text(encoding="utf-8"))
    defined = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.append(node.name)
        elif isinstance(node, ast.Assign):
            defined.extend(t.id for t in node.targets if isinstance(t, ast.Name))
    assert sorted(defined) == sorted(_STAYED)


def test_the_schedule_schema_and_the_handler_keyword_set_stay_derived_from_one_object():
    """The published parameter surface and the handler's closed keyword set read
    the same mapping, so a parameter can never be advertised and then refused."""
    properties = control_subagent_spec.schedule_subagent_properties()
    assert control_subagent_spec.schedule_subagent_param_names() == frozenset(properties)
    entry = next(item for item in control.get_tools() if item.name == "schedule_subagent")
    assert entry.schema["parameters"]["properties"] == properties
    assert entry.schema["parameters"]["additionalProperties"] is False
    # A fresh mapping per call: a caller that mutates one schema cannot corrupt the next.
    assert properties is not control_subagent_spec.schedule_subagent_properties()


def test_every_control_leaf_is_a_hot_code_path_like_its_catalog_owner():
    """Managed-update conflict labelling followed ``ouroboros/tools/control.py``;
    the leaves carry the same label so a split cannot silently downgrade it."""
    assert "ouroboros/tools/control.py" in HOT_CODE_PATHS
    for module in _LEAVES:
        rel = pathlib.Path(module.__file__).relative_to(REPO).as_posix()
        assert rel in HOT_CODE_PATHS, rel


def test_control_extraction_size_bounds_have_meaningful_headroom():
    counts = {
        module.__name__: len(
            pathlib.Path(module.__file__).read_text(encoding="utf-8").splitlines()
        )
        for module in (control, *_LEAVES)
    }
    assert counts["ouroboros.tools.control"] <= 600
    assert all(count <= 1000 for count in counts.values())
    assert 600 <= counts["ouroboros.tools.control_scheduling"] <= 1000
