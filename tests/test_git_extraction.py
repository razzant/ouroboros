"""Structural contracts for the semantic-no-op git tool extraction."""

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
    git,
    git_evolution,
    git_plumbing,
    git_repo_edit,
    git_review_cycle,
    git_vcs_ops,
)


REPO = pathlib.Path(__file__).parents[1]
TOOLS = REPO / "ouroboros" / "tools"

_LEAVES = (git_plumbing, git_review_cycle, git_evolution, git_repo_edit, git_vcs_ops)

_MOVED_OWNERS = {
    "_BINARY_EXTENSIONS": git_plumbing,
    "_acquire_git_lock": git_plumbing,
    "_binding_repo_rel": git_plumbing,
    "_binding_targets_system_repo": git_plumbing,
    "_current_runtime_mode": git_plumbing,
    "_ensure_gitignore": git_plumbing,
    "_protected_paths_block_message": git_plumbing,
    "_publish_git_error": git_plumbing,
    "_publish_review_blocked": git_plumbing,
    "_release_git_lock": git_plumbing,
    "_sanitize_git_error": git_plumbing,
    "_unstage_binaries": git_plumbing,
    "_DOC_ONLY_EXTENSIONS": git_review_cycle,
    "_diff_is_doc_only": git_review_cycle,
    "_finalize_blocked_review": git_review_cycle,
    "_fingerprint_staged_diff": git_review_cycle,
    "_handle_revalidation_failure": git_review_cycle,
    "_mark_failed_bypass_advisory_stale": git_review_cycle,
    "_refuse_capped_attempt": git_review_cycle,
    "_review_binding_precondition_error": git_review_cycle,
    "_review_cycle_infra_failure": git_review_cycle,
    "_run_non_committing_review_cycle": git_review_cycle,
    "_run_reviewed_stage_cycle": git_review_cycle,
    "_stage_candidate_for_review": git_review_cycle,
    "_verify_reviewed_commit_binding": git_review_cycle,
    "_check_evolution_commit_stage": git_evolution,
    "_evolution_commit_authority": git_evolution,
    "_evolution_publication_stopped_result": git_evolution,
    "_preserve_evolution_orphan": git_evolution,
    "_record_evolution_commit_receipt": git_evolution,
    "_CONTENT_OMITTED_PREFIX": git_repo_edit,
    "_check_shrink_guard": git_repo_edit,
    "_repo_write": git_repo_edit,
    "_str_replace_editor": git_repo_edit,
    "_binding_relative_path": git_vcs_ops,
    "_ff_pull": git_vcs_ops,
    "_git_diff": git_vcs_ops,
    "_git_status": git_vcs_ops,
    "_limit_git_output": git_vcs_ops,
    "_pull_from_remote": git_vcs_ops,
    "_restore_to_head": git_vcs_ops,
    "_revert_commit": git_vcs_ops,
    "_vcs_binding": git_vcs_ops,
    "_vcs_result": git_vcs_ops,
}


def test_git_leaves_are_non_catalog_owners_without_git_backedges(tmp_path):
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
            and node.module == "ouroboros.tools.git"
            for node in ast.walk(tree)
        )
        assert not any(
            isinstance(node, ast.Import)
            and any(alias.name == "ouroboros.tools.git" for alias in node.names)
            for node in ast.walk(tree)
        )

    source_inventory = discover_tool_module_inventory(TOOLS)
    assert "git" in source_inventory.tool_modules
    for module in _LEAVES:
        assert module.__name__.rsplit(".", 1)[-1] not in source_inventory.tool_modules
    manifest = tmp_path / "_frozen_tool_modules.v1.json"
    build_frozen_tool_manifest(TOOLS, manifest)
    assert load_frozen_tool_modules(manifest) == source_inventory.tool_modules


def test_git_catalog_schema_bytes_and_handler_owners_are_stable():
    entries = git.get_tools()
    assert tuple(entry.name for entry in entries) == (
        "commit_reviewed",
        "vcs_commit_reviewed",
        "vcs_status",
        "vcs_diff",
        "vcs_pull_ff",
        "vcs_restore",
        "vcs_revert",
    )
    schema_bytes = json.dumps(
        [entry.schema for entry in entries],
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    assert hashlib.sha256(schema_bytes).hexdigest() == (
        "2ae3faed303a4434fc671f394ee6b466f65e10e6f2ebcd18f2307b6dc00b9560"
    )
    assert {
        entry.name: (entry.handler.__module__, entry.handler.__name__)
        for entry in entries
    } == {
        "commit_reviewed": ("ouroboros.tools.git", "_repo_commit_push"),
        "vcs_commit_reviewed": ("ouroboros.tools.git", "_repo_commit_push"),
        "vcs_status": ("ouroboros.tools.git_vcs_ops", "_git_status"),
        "vcs_diff": ("ouroboros.tools.git_vcs_ops", "_git_diff"),
        "vcs_pull_ff": ("ouroboros.tools.git_vcs_ops", "_pull_from_remote"),
        "vcs_restore": ("ouroboros.tools.git_vcs_ops", "_restore_to_head"),
        "vcs_revert": ("ouroboros.tools.git_vcs_ops", "_revert_commit"),
    }


def test_git_facade_reexports_every_moved_identity():
    """``tools/git.py`` keeps the exact objects, so existing importers and
    ``inspect.getsource`` consumers see no identity change."""
    for name, owner in _MOVED_OWNERS.items():
        assert hasattr(git, name), name
        assert getattr(git, name) is getattr(owner, name), name
    owned = {name for module in _LEAVES for name in vars(module)}
    assert set(_MOVED_OWNERS) <= owned


def test_git_extraction_size_bounds_have_meaningful_headroom():
    counts = {
        module.__name__: len(
            pathlib.Path(module.__file__).read_text(encoding="utf-8").splitlines()
        )
        for module in (git, *_LEAVES)
    }
    assert counts["ouroboros.tools.git"] <= 1200
    assert all(count <= 1000 for count in counts.values())
    assert 700 <= counts["ouroboros.tools.git_review_cycle"] <= 1000
