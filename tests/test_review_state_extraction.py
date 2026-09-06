"""Structural contracts for the semantic-no-op review-state extraction.

``review_state`` keeps the STORE: deserialization and its authority-shape
validation, load/save under the advisory lock, repo identity, the snapshot
hash, staleness invalidation, and the status section rendered for the agent.
Three owners sit below it — ``review_state_records`` (the record types and the
pure rules that shape them), ``review_state_model`` (``AdvisoryReviewState``
and every transition it permits) and ``review_state_custody`` (the pending
review-invocation checkpoint and attempt-history hygiene, post-cutoff upstream
growth cut out as a NEW owner by the v7next D06 lane). Cross-references run
through the parent's call-time handle (the D18/D33 mechanical exception), so
every historical monkeypatch target stays live on ``ouroboros.review_state``.
"""

from __future__ import annotations

import ast
import pathlib

from ouroboros import (
    review_state,
    review_state_custody,
    review_state_model,
    review_state_records,
)


REPO = pathlib.Path(__file__).parents[1]

_LEAVES = (review_state_records, review_state_model, review_state_custody)

_MOVED_OWNERS = {
    "AdvisoryReviewState": review_state_model,
    "AdvisoryRunRecord": review_state_records,
    "CommitAttemptRecord": review_state_records,
    "CommitReadinessDebtItem": review_state_records,
    "ObligationItem": review_state_records,
    "_ATTEMPT_MERGE_INCOMING_FIRST": review_state_records,
    "_ATTEMPT_MERGE_INCOMING_LISTS": review_state_records,
    "_ATTEMPT_STR_DEFAULTS": review_state_records,
    "_CANONICAL_OBLIGATION_ITEM_RE": review_state_records,
    "_DEBT_STR_DEFAULTS": review_state_records,
    "_DEFAULT_ADVISORY_TOOL_NAME": review_state_records,
    "_DEFAULT_TOOL_NAME": review_state_records,
    "_LEGACY_CURRENT_REPO_KEY": review_state_records,
    "_MAX_ATTEMPT_HISTORY": review_state_records,
    "_MAX_COMMIT_READINESS_DEBTS": review_state_records,
    "_MAX_RUN_HISTORY": review_state_records,
    "_OBLIGATION_STR_DEFAULTS": review_state_records,
    "_OPEN_COMMIT_READINESS_DEBT_STATUSES": review_state_records,
    "_REVIEW_ATTEMPT_GRACE_SEC": review_state_records,
    "_REVIEW_ATTEMPT_TTL_SEC": review_state_records,
    "_RUN_STATUS_ICONS": review_state_records,
    "_RUN_STR_DEFAULTS": review_state_records,
    "_STATE_SCHEMA_VERSION": review_state_records,
    "_allocate_prefixed_id": review_state_records,
    "_append_finding_lines": review_state_records,
    "_attempt_identity_tuple": review_state_records,
    "_attempt_order_key": review_state_records,
    "_coerce_int": review_state_records,
    "_commit_readiness_debts_view": review_state_records,
    "_dedupe_strings": review_state_records,
    "_filter_lifecycle_records": review_state_records,
    "_filter_repo_scope": review_state_records,
    "_infer_next_prefixed_sequence": review_state_records,
    "_looks_like_public_obligation_id": review_state_records,
    "_make_obligation_fingerprint": review_state_records,
    "_max_iso_ts": review_state_records,
    "_merge_attempt": review_state_records,
    "_min_iso_ts": review_state_records,
    "_normalize_findings": review_state_records,
    "_normalize_fingerprint_text": review_state_records,
    "_normalize_obligation_item_key": review_state_records,
    "_parse_iso_ts": review_state_records,
    "_stable_digest": review_state_records,
    "_utc_now": review_state_records,
    "infer_review_phase": review_state_records,
    "_ACTIVE_REVIEW_OPERATION_STATES": review_state_custody,
    "_STRIPPED_DETAILS_LIMIT": review_state_custody,
    "_STRIPPED_MESSAGE_LIMIT": review_state_custody,
    "_attempt_has_active_review_custody": review_state_custody,
    "_attempt_history_evictable": review_state_custody,
    "_attempt_review_roster_rows": review_state_custody,
    "_review_roster_row_is_pending": review_state_custody,
    "_strip_attempt_heavy_payload": review_state_custody,
    "checkpoint_pending_review_invocation": review_state_custody,
}

_PARENT_OWNED = (
    "_ATTEMPT_AUTHORITY_BOOL_FIELDS",
    "_ATTEMPT_AUTHORITY_STRING_FIELDS",
    "_LOCK_RELPATH",
    "_SNAPSHOT_EXCLUDE_PATHS",
    "_STATE_RELPATH",
    "_build_invalidation_reason",
    "_commit_attempt_from_dict",
    "_load_state_unlocked",
    "_malformed_roster_row",
    "_prepare_state_for_persistence",
    "_resolve_mutation_repo_keys",
    "_save_state_unlocked",
    "_validate_attempt_authority_shape",
    "acquire_review_state_lock",
    "compute_obligation_semantic_redirects",
    "compute_snapshot_hash",
    "discover_repo_root",
    "format_status_section",
    "invalidate_advisory_after_mutation",
    "load_state",
    "make_repo_key",
    "mark_advisory_stale_after_edit",
    "release_review_state_lock",
    "save_state",
    "update_state",
)


def _module_imports(module) -> set[str]:
    tree = ast.parse(pathlib.Path(module.__file__).read_text(encoding="utf-8"))
    modules = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    modules |= {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    return {name for name in modules if name}


def test_review_state_leaves_never_import_their_parent_at_module_scope():
    """The only sanctioned parent reach is the call-time handle inside `_rs()`
    (pinned by test_module_handle_extraction); nothing at module scope."""
    for module in _LEAVES:
        tree = ast.parse(pathlib.Path(module.__file__).read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.ImportFrom):
                assert node.module != "ouroboros.review_state", module.__name__
            if isinstance(node, ast.Import):
                assert all(a.name != "ouroboros.review_state" for a in node.names), module.__name__


def test_review_state_owner_layering_runs_one_way():
    """records knows no sibling; the model imports records; nobody imports the
    custody leaf (its readers reach it through the parent facade)."""
    assert not any(
        name.startswith("ouroboros.review_state_")
        for name in _module_imports(review_state_records)
    )
    assert "ouroboros.review_state_records" in _module_imports(review_state_model)
    for module in _LEAVES:
        assert "ouroboros.review_state_custody" not in (
            _module_imports(module) - {module.__name__}
        ), module.__name__


def test_review_state_facade_reexports_every_moved_identity():
    for name, owner in _MOVED_OWNERS.items():
        assert hasattr(review_state, name), name
        assert getattr(review_state, name) is getattr(owner, name), name
    owned = {name for module in _LEAVES for name in vars(module)}
    assert set(_MOVED_OWNERS) <= owned


def test_review_state_keeps_the_durable_store():
    defined = set()
    for node in ast.parse(
        pathlib.Path(review_state.__file__).read_text(encoding="utf-8")
    ).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, ast.Assign):
            defined.update(t.id for t in node.targets if isinstance(t, ast.Name))
    assert set(_PARENT_OWNED) <= defined
    assert defined.isdisjoint(_MOVED_OWNERS)


def test_review_state_leaves_are_review_stack_members():
    from ouroboros.tools.review_context_atlas import _REVIEW_STACK_PATHS, _is_force_include

    for module in _LEAVES:
        rel = pathlib.Path(module.__file__).relative_to(REPO).as_posix()
        assert rel in _REVIEW_STACK_PATHS, rel
        assert _is_force_include(rel), rel


def test_review_state_extraction_size_bounds_have_meaningful_headroom():
    counts = {
        module.__name__: len(
            pathlib.Path(module.__file__).read_text(encoding="utf-8").splitlines()
        )
        for module in (review_state, *_LEAVES)
    }
    assert all(count <= 1000 for count in counts.values()), counts
    assert counts["ouroboros.review_state"] <= 850
    assert 300 <= counts["ouroboros.review_state_records"] <= 1000
    assert 600 <= counts["ouroboros.review_state_model"] <= 1000
    assert 150 <= counts["ouroboros.review_state_custody"] <= 600
