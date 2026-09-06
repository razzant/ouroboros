"""Facade-identity contract for the extracted tool descriptor/context owners.

Carried from the v7 reference (ouroboros_v7_wip @ 9f691656) with one identity
continuation to THIS tree's bytes: upstream added the ``alias_for`` field to
``ToolEntry`` after the reference cutoff, so the pinned descriptor contract
carries that row here (tip bytes are the truth of the transplant).
"""

from __future__ import annotations

import dataclasses
import inspect


def test_tool_descriptor_owner_facades_preserve_identity():
    """Extracted owners preserve facade identity and the characterized ABI."""
    import ouroboros.tools as tools_package
    from ouroboros.tools import registry, tool_catalog, tool_context

    assert registry.BrowserState is tool_context.BrowserState
    assert registry.ToolContext is tool_context.ToolContext
    assert tools_package.ToolContext is tool_context.ToolContext
    assert registry.ToolEntry is tool_catalog.ToolEntry
    assert tools_package.ToolEntry is tool_catalog.ToolEntry

    def field_contract(cls):
        contract = []
        for item in dataclasses.fields(cls):
            if item.default_factory is not dataclasses.MISSING:
                default = f"factory:{item.default_factory.__name__}"
            elif item.default is dataclasses.MISSING:
                default = "required"
            elif callable(item.default):
                default = f"callable:{item.default.__name__}"
            else:
                default = item.default
            contract.append((item.name, default))
        return tuple(contract)

    assert field_contract(tool_context.BrowserState) == (
        ("pw_instance", None),
        ("browser", None),
        ("page", None),
        ("last_screenshot_b64", None),
    )
    assert field_contract(tool_catalog.ToolEntry) == (
        ("name", "required"),
        ("schema", "required"),
        ("handler", "required"),
        ("is_code_tool", False),
        ("timeout_sec", 360),
        ("mutates_worktree", False),
        ("alias_for", ""),
    )
    assert field_contract(tool_context.ToolContext) == (
        ("repo_dir", "required"),
        ("drive_root", "required"),
        ("branch_dev", "ouroboros"),
        ("system_repo_dir", None),
        ("workspace_root", None),
        ("workspace_mode", ""),
        ("memory_mode", ""),
        ("budget_drive_root", ""),
        ("project_id", ""),
        ("task_metadata", "factory:dict"),
        ("executor_ref", "factory:dict"),
        ("pending_events", "factory:list"),
        ("current_chat_id", None),
        ("current_task_type", None),
        ("pending_restart_reason", None),
        ("last_push_succeeded", False),
        ("last_reviewed_commit_sha", ""),
        ("emit_progress_fn", "callable:<lambda>"),
        ("active_model_override", None),
        ("active_effort_override", None),
        ("active_use_local_override", None),
        ("task_model_override", None),
        ("task_use_local_override", None),
        ("active_context_mode", ""),
        ("browser_state", "factory:BrowserState"),
        ("event_queue", None),
        ("task_id", None),
        ("messages", None),
        ("task_constraint", None),
        ("task_contract", "factory:dict"),
        ("task_depth", 0),
        ("is_direct_chat", False),
        ("is_ephemeral_turn", False),
        ("_review_advisory", "factory:list"),
        ("_review_iteration_count", 0),
        ("_review_history", "factory:list"),
    )
    assert {
        name: str(inspect.signature(getattr(tool_context.ToolContext, name)))
        for name in (
            "active_repo_dir",
            "is_workspace_mode",
            "repo_path",
            "drive_path",
            "drive_logs",
            "task_drive_root",
            "workspace_executor_ref",
        )
    } == {
        "active_repo_dir": "(self) -> 'pathlib.Path'",
        "is_workspace_mode": "(self) -> 'bool'",
        "repo_path": "(self, rel: 'str') -> 'pathlib.Path'",
        "drive_path": "(self, rel: 'str') -> 'pathlib.Path'",
        "drive_logs": "(self) -> 'pathlib.Path'",
        "task_drive_root": "(self) -> 'pathlib.Path'",
        "workspace_executor_ref": "(self) -> 'Dict[str, Any]'",
    }


def test_registry_split_leaves_keep_protected_label_parity():
    """The D04 split moved guard/resolution bodies out of the protected,
    hot-code registry without moving any of the risk: every leaf carries the
    SAME safety-critical and hot-code membership as the parent (the inverse
    of the L-C2 parity rule pinned in test_lc2_owner_facades.py)."""
    from ouroboros.runtime_mode_policy import SAFETY_CRITICAL_PATHS
    from supervisor.update_merge_policy import HOT_CODE_PATHS

    parent = "ouroboros/tools/registry.py"
    leaves = (
        # extension_dispatch received the extension/MCP dispatch bodies that
        # lived on the hot ToolRegistry class, so it holds both labels too.
        "ouroboros/tools/extension_dispatch.py",
        "ouroboros/tools/registry_core.py",
        "ouroboros/tools/registry_guard_process.py",
        "ouroboros/tools/registry_guards.py",
        "ouroboros/tools/tool_catalog.py",
        "ouroboros/tools/tool_context.py",
        "ouroboros/tools/tool_resolution.py",
        "ouroboros/tools/tool_result.py",
    )
    for inventory in (SAFETY_CRITICAL_PATHS, HOT_CODE_PATHS):
        assert parent in inventory
        for leaf in leaves:
            assert leaf in inventory, leaf

    # The tool_access split's parent carries NEITHER label; its leaves must
    # not silently acquire one (same parity, other direction).
    ta_parent = "ouroboros/tool_access.py"
    ta_leaves = (
        "ouroboros/tool_access_types.py",
        "ouroboros/tool_access_paths.py",
        "ouroboros/tool_access_roots.py",
        "ouroboros/tool_access_user_files.py",
    )
    for inventory in (SAFETY_CRITICAL_PATHS, HOT_CODE_PATHS):
        assert ta_parent not in inventory
        for leaf in ta_leaves:
            assert leaf not in inventory, leaf
