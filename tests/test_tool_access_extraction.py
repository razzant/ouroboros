"""Structural contracts for the semantic-no-op tool_access extraction."""

from __future__ import annotations

import ast
import pathlib

from ouroboros import (
    tool_access,
    tool_access_paths,
    tool_access_roots,
    tool_access_types,
    tool_access_user_files,
)
from ouroboros.tool_module_inventory import (
    build_frozen_tool_manifest,
    discover_tool_module_inventory,
    load_frozen_tool_modules,
)


REPO = pathlib.Path(__file__).parents[1]
TOOLS = REPO / "ouroboros" / "tools"

_LEAVES = (
    tool_access_types,
    tool_access_paths,
    tool_access_roots,
    tool_access_user_files,
)

_MOVED_OWNERS = {
    "Operation": tool_access_types,
    "ResolvedResourceBinding": tool_access_types,
    "ResourceRoot": tool_access_types,
    "SUBAGENT_CAPABILITIES": tool_access_types,
    "SubagentCapability": tool_access_types,
    "ToolAccessDecision": tool_access_types,
    "ToolProfile": tool_access_types,
    "_ALL_ROOTS": tool_access_types,
    "_POLICY": tool_access_types,
    "_READONLY_RESOURCE_ROOTS": tool_access_types,
    "_READ_OPS": tool_access_types,
    "_SUBAGENT_CAPABILITY_TO_OPERATION": tool_access_types,
    "_TOP_LEVEL_PRINCIPAL_POLICY": tool_access_types,
    "_TOP_LEVEL_PRINCIPAL_PROFILES": tool_access_types,
    "_deliverables_root": tool_access_paths,
    "_path_is_relative_to_casefold": tool_access_paths,
    "_user_files_root": tool_access_paths,
    "canonical_data_root": tool_access_paths,
    "normalize_root": tool_access_paths,
    "normalize_root_relative": tool_access_paths,
    "normalize_runtime_data_path": tool_access_paths,
    "path_is_relative_to": tool_access_paths,
    "paths_overlap_casefold": tool_access_paths,
    "workspace_mode_block_reason": tool_access_paths,
    "_is_subagent_ctx": tool_access_roots,
    "_skill_payload_base": tool_access_roots,
    "active_tool_profile": tool_access_roots,
    "binding_targets_system_repo": tool_access_roots,
    "is_external_workspace": tool_access_roots,
    "load_bound_skill": tool_access_roots,
    "predicted_subagent_profile": tool_access_roots,
    "project_room_lens_dir": tool_access_roots,
    "resource_root_path": tool_access_roots,
    "UserFilesPathBlockedError": tool_access_user_files,
    "_USER_FILES_ALLOWED_DOTNAMES": tool_access_user_files,
    "_USER_FILES_SECRET_COMPONENTS": tool_access_user_files,
    "_USER_FILES_SECRET_NAMES": tool_access_user_files,
    "_USER_FILES_SECRET_RE": tool_access_user_files,
    "_subagent_projects_read_hint": tool_access_user_files,
    "resolve_user_file_path": tool_access_user_files,
    "user_files_path_block_reason": tool_access_user_files,
}


def test_tool_access_leaves_are_non_catalog_owners_without_backedges(tmp_path):
    for module in (tool_access, *_LEAVES):
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
            isinstance(node, ast.ImportFrom) and node.module == "ouroboros.tool_access"
            for node in ast.walk(tree)
        )
        assert not any(
            isinstance(node, ast.Import)
            and any(alias.name == "ouroboros.tool_access" for alias in node.names)
            for node in ast.walk(tree)
        )

    source_inventory = discover_tool_module_inventory(TOOLS)
    for module in (tool_access, *_LEAVES):
        assert module.__name__.rsplit(".", 1)[-1] not in source_inventory.tool_modules
    manifest = tmp_path / "_frozen_tool_modules.v1.json"
    build_frozen_tool_manifest(TOOLS, manifest)
    assert load_frozen_tool_modules(manifest) == source_inventory.tool_modules


def test_tool_access_decision_surface_stays_with_the_matrix_owner():
    """The access decision, its affordance projections, and the binding builder
    remain authored by ``tool_access`` itself; the leaves own vocabulary,
    physical paths, root resolution, and one root's path policy."""
    for name in (
        "decide_tool_access",
        "subagent_profile_satisfies",
        "summarize_subagent_profile",
        "filesystem_affordance_map",
        "profile_readable_root_paths",
        "shell_cwd_block_message",
        "resolve_shell_cwd",
        "build_resolved_resource_binding",
        "resolve_resource_path",
    ):
        assert getattr(tool_access, name).__module__ == "ouroboros.tool_access", name


def test_tool_access_facade_reexports_every_moved_identity():
    """``tool_access`` keeps the exact objects, so the registry, the tool
    handlers, the supervisor and every guard that imports these names see no
    identity change."""
    for name, owner in _MOVED_OWNERS.items():
        assert hasattr(tool_access, name), name
        assert getattr(tool_access, name) is getattr(owner, name), name
    owned = {name for module in _LEAVES for name in vars(module)}
    assert set(_MOVED_OWNERS) <= owned


def test_tool_access_policy_matrix_is_one_object_across_owners():
    """Every reader of the matrix — the decision, the projections, the
    user_files hint — must observe the same mapping object, not a copy."""
    assert tool_access._POLICY is tool_access_types._POLICY
    assert tool_access_user_files._POLICY is tool_access_types._POLICY
    assert set(tool_access._ALL_ROOTS) == set(tool_access_types._ALL_ROOTS)


def test_tool_access_extraction_size_bounds_have_meaningful_headroom():
    counts = {
        module.__name__: len(
            pathlib.Path(module.__file__).read_text(encoding="utf-8").splitlines()
        )
        for module in (tool_access, *_LEAVES)
    }
    assert counts["ouroboros.tool_access"] <= 900
    assert all(count <= 1000 for count in counts.values())
    assert 200 <= counts["ouroboros.tool_access_user_files"] <= 1000
