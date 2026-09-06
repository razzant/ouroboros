"""Structural contracts for the semantic-no-op control tool extraction.

v7next D08 lane carried the D08 half of the reference split — ``control_events``
(the emission/confirmation seam), ``control_routing`` (the chat/project routing
verbs) and ``control_runtime`` (runtime self-control); the D07 lane carried the
rest — ``control_subagent_spec`` (the published schedule surface),
``control_scheduling`` (scheduling one live subagent) and
``control_task_results`` (absorbing a child). ``tools/control.py`` is now the
thin catalog facade; this suite pins the completed split shape.
"""

from __future__ import annotations

import ast
import pathlib

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

_LEAVES = (
    control_events,
    control_routing,
    control_runtime,
    control_scheduling,
    control_subagent_spec,
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
    "_attach_client_surface": control_routing,
    "_attach_origin_from_metadata": control_routing,
    "_attach_swarm_intent": control_routing,
    "_cached_swarm_handoff": control_routing,
    "_finish_swarm_handoff": control_routing,
    "_list_projects": control_routing,
    "_promote_chat_to_task": control_routing,
    "_route_to_project": control_routing,
    "_steer_task": control_routing,
    # unrowed post-cutoff upstream family riding with its only readers
    # (promote/route): the predecessor-authority selector and its two helpers.
    "_MISSING_PREDECESSOR_SELECTOR": control_routing,
    "_predecessor_selector_error": control_routing,
    "_attach_predecessor_authority_from_metadata": control_routing,
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
    # D07 half (oracle rows 2537-2556, 2569-2579 re-cut on tip bytes).
    "VALID_SUBTASK_MEMORY_MODES": control_subagent_spec,
    "schedule_subagent_properties": control_subagent_spec,
    "schedule_subagent_param_names": control_subagent_spec,
    "_INTERNAL_SCHEDULE_OPTIONS": control_subagent_spec,
    "_validated_schedule_fields": control_subagent_spec,
    "RETIRED_SCHEDULE_PARAMS": control_subagent_spec,
    "_record_scheduled_subagent": control_scheduling,
    "_emit_swarm_fanout": control_scheduling,
    "_subagent_slot_note": control_scheduling,
    "_capability_mismatch_message": control_scheduling,
    "_finalize_schedule_emission": control_scheduling,
    "_build_acting_constraint": control_scheduling,
    "_select_subagent_constraint": control_scheduling,
    "_populate_subagent_event_extras": control_scheduling,
    "_prepare_child_drive": control_scheduling,
    "_earliest_deadline_at": control_scheduling,
    "_build_child_subagent_contract": control_scheduling,
    "_resolve_executor_ref": control_scheduling,
    "_inherited_workspace_from_active_repo": control_scheduling,
    "_schedule_task": control_scheduling,
    # unrowed post-cutoff upstream neighbours riding with their only readers:
    # the depth probe and the attachment manifest with _schedule_task, the
    # fanout emitter with its external reader (tools/delegate.py keeps
    # importing it from the facade), the hidden-params set with the validator
    # and the handler attribute stamp.
    "_context_task_depth": control_scheduling,
    "_materialize_child_attachment_manifest": control_scheduling,
    "maybe_emit_delegated_run_fanout": control_scheduling,
    "HIDDEN_LEGACY_SCHEDULE_PARAMS": control_scheduling,
    "disclosable_capability_delta": control_task_results,
    "_subtask_outcome_summary": control_task_results,
    "_get_task_result": control_task_results,
    "_wait_attention_poll": control_task_results,
    "cache_horizon_note": control_task_results,
    "_wait_for_task": control_task_results,
    "_count_live_sibling_children": control_task_results,
    "_UNMINTED_WAIT_GRACE_SEC": control_task_results,
    "_unminted_wait_ids": control_task_results,
    "_children_roster_projection": control_task_results,
    "_wait_for_tasks": control_task_results,
}


def test_control_leaves_are_non_catalog_owners_without_control_backedges():
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


def test_control_catalog_handler_owners_point_at_the_carried_leaves():
    entries = {entry.name: entry for entry in control.get_tools()}
    owned = {
        "promote_chat_to_task": (control_routing, "_promote_chat_to_task"),
        "list_projects": (control_routing, "_list_projects"),
        "route_to_project": (control_routing, "_route_to_project"),
        "steer_task": (control_routing, "_steer_task"),
        "set_tool_timeout": (control_runtime, "_set_tool_timeout"),
        "request_restart": (control_runtime, "_request_restart"),
        "promote_to_stable": (control_runtime, "_promote_to_stable"),
        "request_deep_self_review": (control_runtime, "_request_deep_self_review"),
        "chat_history": (control_runtime, "_chat_history"),
        "update_scratchpad": (control_runtime, "_update_scratchpad"),
        "send_user_message": (control_runtime, "_send_user_message"),
        "update_identity": (control_runtime, "_update_identity"),
        "toggle_evolution": (control_runtime, "_toggle_evolution"),
        "toggle_consciousness": (control_runtime, "_toggle_consciousness"),
        "switch_model": (control_runtime, "_switch_model"),
        "schedule_subagent": (control_scheduling, "_schedule_task"),
        "get_task_result": (control_task_results, "_get_task_result"),
        "wait_task": (control_task_results, "_wait_for_task"),
        "wait_tasks": (control_task_results, "_wait_for_tasks"),
    }
    for name, (module, attr) in owned.items():
        assert name in entries, name
        handler = entries[name].handler
        assert handler.__module__ == module.__name__, name
        assert handler is getattr(module, attr), name


def test_control_facade_reexports_every_moved_identity():
    """``tools/control.py`` keeps the exact objects, so plan review, the join
    ledger, delegation, review evidence and the tests that reach for a private
    helper see no identity change."""
    for name, owner in _MOVED_OWNERS.items():
        assert hasattr(control, name), name
        assert getattr(control, name) is getattr(owner, name), name
    owned = {name for module in _LEAVES for name in vars(module)}
    assert set(_MOVED_OWNERS) <= owned


def test_the_facade_no_longer_defines_any_moved_name():
    tree = ast.parse(pathlib.Path(control.__file__).read_text(encoding="utf-8"))
    defined = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, ast.Assign):
            defined.update(t.id for t in node.targets if isinstance(t, ast.Name))
    assert defined & set(_MOVED_OWNERS) == set()
    # the catalog stays where it was
    assert "get_tools" in defined


def test_every_control_leaf_is_a_hot_code_path_like_its_catalog_owner():
    """Managed-update conflict labelling followed ``ouroboros/tools/control.py``;
    the leaves carry the same label so a split cannot silently downgrade it."""
    assert "ouroboros/tools/control.py" in HOT_CODE_PATHS
    for module in _LEAVES:
        rel = pathlib.Path(module.__file__).relative_to(REPO).as_posix()
        assert rel in HOT_CODE_PATHS, rel
