"""Structural contracts for the semantic-no-op git tool extraction.

The facade retains the paid-cycle gate family, deferred update entry points
and tool catalog. Its leaves are non-catalog owners without import-time
backedges, and the facade re-exports every moved identity.
"""

from __future__ import annotations

import ast
import hashlib
import json
import pathlib

from ouroboros.tools import (
    git,
    git_evolution,
    git_managed_postcommit,
    git_plumbing,
    git_repo_edit,
    git_review_cycle,
    git_vcs_ops,
)


REPO = pathlib.Path(__file__).parents[1]
TOOLS = REPO / "ouroboros" / "tools"

_LEAVES = (git_plumbing, git_review_cycle, git_managed_postcommit, git_evolution, git_repo_edit, git_vcs_ops)

_MOVED_OWNERS = {
    "_BINARY_EXTENSIONS": git_plumbing,
    "_acquire_git_lock": git_plumbing,
    "_binding_repo_rel": git_plumbing,
    "_binding_targets_system_repo": git_plumbing,
    "_current_runtime_mode": git_plumbing,
    "_ensure_gitignore": git_plumbing,
    "_protected_paths_block_message": git_plumbing,
    "_release_git_lock": git_plumbing,
    "_sanitize_git_error": git_plumbing,
    "_unstage_binaries": git_plumbing,
    "_DOC_ONLY_EXTENSIONS": git_review_cycle,
    "_diff_is_doc_only": git_review_cycle,
    "_finalize_blocked_review": git_review_cycle,
    "_fingerprint_staged_diff": git_review_cycle,
    "_handle_revalidation_failure": git_review_cycle,
    "_review_binding_precondition_error": git_review_cycle,
    "_review_cycle_infra_failure": git_review_cycle,
    "_run_non_committing_review_cycle": git_review_cycle,
    "_run_reviewed_stage_cycle": git_review_cycle,
    "_stage_candidate_for_review": git_review_cycle,
    "_verify_reviewed_commit_binding": git_review_cycle,
    "_finish_managed_commit": git_managed_postcommit,
    "_managed_commit_paused": git_managed_postcommit,
    "_managed_post_commit_tests_gate": git_managed_postcommit,
    "_resume_managed_commit": git_managed_postcommit,
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


def test_git_leaves_are_non_catalog_owners_without_git_backedges():
    for module in _LEAVES:
        source_path = pathlib.Path(module.__file__)
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        assert not any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "get_tools"
            for node in tree.body
        )
        # No import-time backedge onto the facade: the leaves reach parent-owned
        # helpers only at call time through their _git() handle.
        for node in tree.body:
            if isinstance(node, ast.ImportFrom):
                assert node.module != "ouroboros.tools.git", module.__name__
            if isinstance(node, ast.Import):
                assert all(a.name != "ouroboros.tools.git" for a in node.names), module.__name__


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
    # Informed Advisory commit aliases plus explicit local vcs_diff base/head.
    assert hashlib.sha256(schema_bytes).hexdigest() == (
        "e20b653778ee12e2e41d0f6baee598e28165260483038540a4143686abdfe7dd"  # FIX6a 1: `reviewers`/`reason` compose the commit panel by review_change's rule (decision 1A); FIX3 T1 before it: `scope` described by the panel's coupling questions
    )
    assert {
        entry.name: (entry.handler.__module__, entry.handler.__name__)
        for entry in entries
    } == {
        # The public handler names the call's review record in every outcome (DEVELOPMENT 05).
        "commit_reviewed": ("ouroboros.tools.git", "_commit_reviewed"),
        "vcs_commit_reviewed": ("ouroboros.tools.git", "_commit_reviewed"),
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
    assert counts["ouroboros.tools.git"] <= 1800
    assert all(count <= 1000 for name, count in counts.items()
               if name != "ouroboros.tools.git")
    assert counts["ouroboros.tools.git_review_cycle"] <= 1000
