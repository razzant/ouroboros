"""Facade-identity contract for the extracted tool descriptor/context owners.

Split out of tests/test_contracts.py so that module stays inside the v7 1500-line ratchet;
the assertions are unchanged.
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
