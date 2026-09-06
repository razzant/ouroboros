"""Structural contracts for the semantic-no-op review-helpers extraction.

``review_helpers`` keeps the review plumbing every surface shares — the prompt
token budget and its density calibration, drive-root resolution, event/usage
emission, the wave budget gate, cached prompt blocks, governance-document
loading, the scope actor record, the checklist section, the intent sections,
and the pre-advisory worktree checks. Two owners sit beside it:
``review_prompt_text`` (the fixed reviewer vocabulary and the sections
rendered from prior rounds) and ``review_file_pack`` (what counts as
sensitive/binary/oversized, the porcelain parsers, and the packs read from the
working tree). Cross-references run through the parent's call-time handle
(the D18/D33 mechanical exception), and the parent re-exports every moved
identity, so existing importers and monkeypatching tests see no change.
"""

from __future__ import annotations

import ast
import pathlib

from ouroboros.tools import (
    review_file_pack,
    review_helpers,
    review_prompt_text,
)


REPO = pathlib.Path(__file__).parents[1]

_LEAVES = (review_prompt_text, review_file_pack)

_MOVED_OWNERS = {
    "_JSON_SECRET_RE": review_prompt_text,
    "_SECRET_LINE_RE": review_prompt_text,
    "CRITICAL_FINDING_CALIBRATION": review_prompt_text,
    "REVIEW_PREAMBLE": review_prompt_text,
    "REVIEW_THOROUGHNESS_BLOCK": review_prompt_text,
    "REVIEW_SEVERITY_THRESHOLDS": review_prompt_text,
    "REPO_ANTI_PATTERN_LOCK_GUARD": review_prompt_text,
    "_ANTI_THRASHING_RULE_VERDICT": review_prompt_text,
    "_ANTI_THRASHING_RULE_ITEM_NAME": review_prompt_text,
    "_CONVERGENCE_RULE_TEXT": review_prompt_text,
    "_HISTORY_VERIFICATION_ONLY_RULE": review_prompt_text,
    "_OBLIGATION_SUFFIX_RE": review_prompt_text,
    "_make_fence": review_prompt_text,
    "build_anti_thrashing_rules_section": review_prompt_text,
    "build_obligations_block": review_prompt_text,
    "build_rebuttal_section": review_prompt_text,
    "build_review_history_section": review_prompt_text,
    "build_self_verification_template": review_prompt_text,
    "format_obligation_excerpt": review_prompt_text,
    "format_prompt_code_block": review_prompt_text,
    "format_review_history_entry": review_prompt_text,
    "normalize_reviewer_item": review_prompt_text,
    "normalize_reviewer_items": review_prompt_text,
    "normalize_reviewer_obligation_id": review_prompt_text,
    "redact_prompt_secrets": review_prompt_text,
    "single_line": review_prompt_text,
    "strip_obligation_suffix": review_prompt_text,
    "BINARY_EXTENSIONS": review_file_pack,
    "CARRIER_CUT_REASON": review_file_pack,
    "_BINARY_SNIFF_BYTES": review_file_pack,
    "_FILE_SIZE_LIMIT": review_file_pack,
    "_FULL_REPO_BINARY_EXTENSIONS": review_file_pack,
    "_FULL_REPO_SKIP_DIR_PREFIXES": review_file_pack,
    "_MAX_FULL_REPO_FILE_BYTES": review_file_pack,
    "_SENSITIVE_EXTENSIONS": review_file_pack,
    "_SENSITIVE_NAMES": review_file_pack,
    "_VENDORED_NAMES": review_file_pack,
    "_VENDORED_SUFFIXES": review_file_pack,
    "_is_probably_binary": review_file_pack,
    "_raw_bytes_binary": review_file_pack,
    "build_advisory_changed_context": review_file_pack,
    "build_full_repo_pack": review_file_pack,
    "build_head_snapshot_section": review_file_pack,
    "build_touched_file_pack": review_file_pack,
    "format_name_status_for_preflight": review_file_pack,
    "iter_repo_pack_entries": review_file_pack,
    "list_changed_paths_from_git_status": review_file_pack,
    "list_git_tracked_paths": review_file_pack,
    "parse_changed_paths_from_porcelain": review_file_pack,
    "parse_changed_paths_from_porcelain_z": review_file_pack,
    "parse_git_name_status": review_file_pack,
    "paths_from_name_status": review_file_pack,
    "paths_from_porcelain_line": review_file_pack,
    "pack_exclusion_note": review_file_pack,
    "span_only_release_carriers": review_file_pack,
}

_PARENT_OWNED = (
    "REPO_ROOT",
    "REVIEW_PROMPT_TOKEN_BUDGET",
    "SKILL_HOST_CONTEXT_FILES",
    "_COMMIT_SUBJECT_MAX_CHARS",
    "_commit_subject",
    "_run_review_preflight_tests",
    "build_blocking_findings_json_section",
    "build_goal_section",
    "build_scope_actor_record",
    "build_scope_section",
    "build_skill_host_context",
    "cached_prompt_blocks",
    "calibrated_input_token_limit",
    "check_worktree_readiness",
    "emit_review_event",
    "emit_review_usage",
    "format_advisory_error",
    "get_advisory_runtime_diagnostics",
    "load_checklist_section",
    "load_governance_doc",
    "resolve_intent",
    "review_drive_root",
    "review_wave_budget_gate",
)


def test_review_helper_leaves_never_import_their_parent_at_module_scope():
    for module in _LEAVES:
        tree = ast.parse(pathlib.Path(module.__file__).read_text(encoding="utf-8"))
        assert not any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "get_tools"
            for node in tree.body
        )
        for node in tree.body:
            if isinstance(node, ast.ImportFrom):
                assert node.module != "ouroboros.tools.review_helpers", module.__name__
            if isinstance(node, ast.Import):
                assert all(
                    a.name != "ouroboros.tools.review_helpers" for a in node.names
                ), module.__name__


def test_review_helpers_facade_reexports_every_moved_identity():
    for name, owner in _MOVED_OWNERS.items():
        assert hasattr(review_helpers, name), name
        assert getattr(review_helpers, name) is getattr(owner, name), name
    owned = {name for module in _LEAVES for name in vars(module)}
    assert set(_MOVED_OWNERS) <= owned


def test_review_helpers_keeps_the_shared_review_plumbing():
    defined = set()
    for node in ast.parse(
        pathlib.Path(review_helpers.__file__).read_text(encoding="utf-8")
    ).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, ast.Assign):
            defined.update(t.id for t in node.targets if isinstance(t, ast.Name))
    assert set(_PARENT_OWNED) <= defined
    assert defined.isdisjoint(_MOVED_OWNERS)


def test_review_prompt_text_reads_nothing_from_the_repository():
    """The vocabulary owner formats records; only the pack owner touches disk/git."""
    tree = ast.parse(pathlib.Path(review_prompt_text.__file__).read_text(encoding="utf-8"))
    imported = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    } | {
        alias.name for node in ast.walk(tree)
        if isinstance(node, ast.Import) for alias in node.names
    }
    assert "subprocess" not in imported and "pathlib" not in imported, imported
    assert not any(
        isinstance(node, ast.ImportFrom)
        and node.module == "ouroboros.tools.review_file_pack"
        for node in ast.walk(tree)
    )


def test_review_helper_leaves_are_review_stack_members():
    from ouroboros.tools.review_context_atlas import _REVIEW_STACK_PATHS, _is_force_include

    for module in _LEAVES:
        rel = pathlib.Path(module.__file__).relative_to(REPO).as_posix()
        assert rel in _REVIEW_STACK_PATHS, rel
        assert _is_force_include(rel), rel


def test_review_helpers_extraction_size_bounds_have_meaningful_headroom():
    counts = {
        module.__name__: len(
            pathlib.Path(module.__file__).read_text(encoding="utf-8").splitlines()
        )
        for module in (review_helpers, *_LEAVES)
    }
    assert all(count <= 1000 for count in counts.values()), counts
    assert counts["ouroboros.tools.review_helpers"] <= 850
    assert 300 <= counts["ouroboros.tools.review_prompt_text"] <= 1000
    assert 400 <= counts["ouroboros.tools.review_file_pack"] <= 1000
