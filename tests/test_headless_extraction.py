"""Structural contracts for the semantic-no-op headless extraction."""

from __future__ import annotations

import ast
import pathlib

from ouroboros import headless, headless_status, workspace_patch_capture
from ouroboros.tool_module_inventory import (
    build_frozen_tool_manifest,
    discover_tool_module_inventory,
    load_frozen_tool_modules,
)


REPO = pathlib.Path(__file__).parents[1]
TOOLS = REPO / "ouroboros" / "tools"

_LEAVES = (headless_status, workspace_patch_capture)

_MOVED_OWNERS = {
    "ARTIFACT_STATUS_FAILED": headless_status,
    "ARTIFACT_STATUS_FINALIZING": headless_status,
    "ARTIFACT_STATUS_MISSING": headless_status,
    "ARTIFACT_STATUS_PENDING": headless_status,
    "ARTIFACT_STATUS_READY": headless_status,
    "ARTIFACT_STATUS_READY_NO_CHANGES": headless_status,
    "ARTIFACT_STATUS_READY_WITH_CHANGES": headless_status,
    "ARTIFACT_TERMINAL_STATUSES": headless_status,
    "_ARTIFACT_LIFECYCLE_FIELDS": headless_status,
    "_FINAL_STATUSES": headless_status,
    "_LOCAL_READONLY_SUBAGENT_MODE": headless_status,
    "SCRATCH_MANIFEST_NAME": workspace_patch_capture,
    "_GIT_UNBORN_HEAD": workspace_patch_capture,
    "_acting_constraint_from_task": workspace_patch_capture,
    "_append_git_output": workspace_patch_capture,
    "_empty_patch_manifest": workspace_patch_capture,
    "_git_bytes": workspace_patch_capture,
    "_git_empty_tree_oid": workspace_patch_capture,
    "_git_path_list": workspace_patch_capture,
    "_git_stdout": workspace_patch_capture,
    "_head_reflog_exists": workspace_patch_capture,
    "_looks_like_git_oid": workspace_patch_capture,
    "_preflight_head_from_task": workspace_patch_capture,
    "_preflight_head_present": workspace_patch_capture,
    "_untracked_blob_exclude_reason": workspace_patch_capture,
    "_workspace_patch_base": workspace_patch_capture,
    "_write_patch_separator": workspace_patch_capture,
    "build_workspace_patch": workspace_patch_capture,
    "untracked_capture_veto_reason": workspace_patch_capture,
    "write_workspace_patch_artifacts": workspace_patch_capture,
}


def test_headless_leaves_are_non_catalog_owners_without_headless_backedges(tmp_path):
    for module in (headless, *_LEAVES):
        source_path = pathlib.Path(module.__file__)
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        assert not any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "get_tools"
            for node in tree.body
        )
    for module in _LEAVES:
        tree = ast.parse(pathlib.Path(module.__file__).read_text(encoding="utf-8"))
        assert not any(
            isinstance(node, ast.ImportFrom) and node.module == "ouroboros.headless"
            for node in ast.walk(tree)
        )
        assert not any(
            isinstance(node, ast.Import)
            and any(alias.name == "ouroboros.headless" for alias in node.names)
            for node in ast.walk(tree)
        )

    source_inventory = discover_tool_module_inventory(TOOLS)
    for module in (headless, *_LEAVES):
        assert module.__name__.rsplit(".", 1)[-1] not in source_inventory.tool_modules
    manifest = tmp_path / "_frozen_tool_modules.v1.json"
    build_frozen_tool_manifest(TOOLS, manifest)
    assert load_frozen_tool_modules(manifest) == source_inventory.tool_modules


def test_headless_public_export_list_is_unchanged():
    """``__all__`` is the module's declared contract; the extraction moved owners,
    never the surface, so every published name still resolves on ``headless``."""
    assert headless.__all__ == [
        "ARTIFACT_STATUS_FAILED",
        "ARTIFACT_STATUS_FINALIZING",
        "ARTIFACT_STATUS_PENDING",
        "ARTIFACT_STATUS_READY",
        "build_memory_export",
        "build_workspace_patch",
        "copy_child_task_result",
        "finalize_task_artifacts",
        "task_is_readonly_subagent",
        "prepare_task_drive",
        "prune_headless_task_drives",
        "prune_task_drives",
        "task_artifacts_dir",
        "task_state_dir",
        "write_workspace_patch_artifacts",
        "write_workspace_preflight_artifact",
    ]
    for name in headless.__all__:
        assert hasattr(headless, name), name


def test_headless_facade_reexports_every_moved_identity():
    """``headless`` keeps the exact objects, so the supervisor, the gateway,
    outcomes, task_status, artifacts and the delegation owners see no identity
    change."""
    for name, owner in _MOVED_OWNERS.items():
        assert hasattr(headless, name), name
        assert getattr(headless, name) is getattr(owner, name), name
    owned = {name for module in _LEAVES for name in vars(module)}
    assert set(_MOVED_OWNERS) <= owned


def test_headless_extraction_size_bounds_have_meaningful_headroom():
    counts = {
        module.__name__: len(
            pathlib.Path(module.__file__).read_text(encoding="utf-8").splitlines()
        )
        for module in (headless, *_LEAVES)
    }
    assert counts["ouroboros.headless"] <= 1000
    assert all(count <= 1000 for count in counts.values())
    assert 400 <= counts["ouroboros.workspace_patch_capture"] <= 1000
