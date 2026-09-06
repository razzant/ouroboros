"""Structural contracts for the semantic-no-op review_substrate extraction.

Re-cut on the v7next tip. Two upstream re-homes are honored, not dragged back:
``reviewer_slots`` lives in ``reviewer_slot_config`` and ``_render_prompt`` in
``review_execution`` (the substrate re-imports both). ``slot_id_for_row``
likewise lives in ``review_dispatch`` on the tip.
"""

from __future__ import annotations

import ast
import pathlib

from ouroboros import (
    review_projection,
    review_records,
    review_substrate,
    review_verdict,
)

REPO = pathlib.Path(__file__).parents[1]
_LEAVES = (review_records, review_verdict, review_projection)

_MOVED_OWNERS = {
    "HARDNESS_ADVISORY_VISIBLE": review_records,
    "HARDNESS_HARD_GATE": review_records,
    "HARDNESS_LABEL_ONLY": review_records,
    "ReviewActorRecord": review_records,
    "ReviewRequest": review_records,
    "ReviewRunResult": review_records,
    "ReviewSlot": review_records,
    "TYPED_FAILURE_FACT_KEYS": review_records,
    "DIALOGUE_CONTINUE": review_verdict,
    "DIALOGUE_INCONCLUSIVE": review_verdict,
    "DIALOGUE_STABLE_DISAGREEMENT": review_verdict,
    "DIALOGUE_STATUS_VALUES": review_verdict,
    "DIALOGUE_TERMINAL_STATUSES": review_verdict,
    "DIALOGUE_UNREACHABLE": review_verdict,
    "DIALOGUE_VOTE_ABSTAIN_INVALID": review_verdict,
    "DIALOGUE_VOTE_CONTINUE_WITHOUT_FINDINGS": review_verdict,
    "_CRITERION_STATUSES": review_verdict,
    "_TIER_ORDER": review_verdict,
    "_contributing_actors": review_verdict,
    "_criteria_have_supported_evidence": review_verdict,
    "_criteria_shape_valid": review_verdict,
    "_unresolved_evidence_ref_labels": review_verdict,
    "aggregate_dialogue_status": review_verdict,
    "aggregate_outcome_tier": review_verdict,
    "build_improvement_capsule": review_verdict,
    "dissent_findings": review_verdict,
    "panel_reason": review_verdict,
    "task_acceptance_is_clean": review_verdict,
    "_public_review_reason": review_projection,
    "_response_ref_projection": review_projection,
    "_review_actor_projection": review_projection,
    "_review_enforcement_impact": review_projection,
    "_review_panel_id": review_projection,
    "_transport_error_status": review_projection,
    "build_review_binding": review_projection,
    "compact_review_projection": review_projection,
}


def test_review_substrate_leaves_are_non_catalog_owners_without_backedges():
    for module in (review_substrate, *_LEAVES):
        tree = ast.parse(pathlib.Path(module.__file__).read_text(encoding="utf-8"))
        assert not any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "get_tools"
            for node in tree.body
        )
    for module in _LEAVES:
        tree = ast.parse(pathlib.Path(module.__file__).read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.ImportFrom):
                assert node.module != "ouroboros.review_substrate", module.__name__
            if isinstance(node, ast.Import):
                assert all(
                    a.name != "ouroboros.review_substrate" for a in node.names
                ), module.__name__


def test_review_substrate_keeps_the_coordinator():
    """Running a panel and resolving the governance/subject roots stay authored
    by ``review_substrate`` itself; the leaves own the records, the reducers and
    the projection. The tip's own re-homes (``reviewer_slots`` ->
    reviewer_slot_config, ``slot_id_for_row`` -> review_dispatch,
    ``_render_prompt`` -> review_execution) are pinned as re-exports."""
    for name in (
        "ReviewCoordinator",
        "run_review_request",
        "scope_reviewer_slots",
        "review_repo_dirs_for",
    ):
        assert getattr(review_substrate, name).__module__ == "ouroboros.review_substrate", name
    assert review_substrate.reviewer_slots.__module__ == "ouroboros.reviewer_slot_config"
    assert review_substrate.slot_id_for_row.__module__ == "ouroboros.review_dispatch"
    assert review_substrate._render_prompt.__module__ == "ouroboros.review_execution"


def test_review_substrate_facade_reexports_every_moved_identity():
    """``review_substrate`` keeps the exact objects, so the loop, the scope and
    plan surfaces, the reviewer-slot config and every task-result consumer that
    imports these names see no identity change."""
    for name, owner in _MOVED_OWNERS.items():
        assert hasattr(review_substrate, name), name
        assert getattr(review_substrate, name) is getattr(owner, name), name
    owned = {name for module in _LEAVES for name in vars(module)}
    assert set(_MOVED_OWNERS) <= owned


def test_review_panel_records_are_one_class_across_owners():
    """Every producer and reader of a panel record must observe the same
    classes, so an ``isinstance`` check or an ``asdict`` round-trip cannot
    depend on the import site. The verdict and projection leaves reach the
    records through the parent's call-time handle, so the parent binding IS
    the leaf's class."""
    assert review_substrate.ReviewRunResult is review_records.ReviewRunResult
    assert review_substrate.ReviewRequest is review_records.ReviewRequest
    assert review_substrate.ReviewActorRecord is review_records.ReviewActorRecord
    assert review_substrate.panel_reason is review_verdict.panel_reason
    assert (
        review_substrate.DIALOGUE_STATUS_VALUES is review_verdict.DIALOGUE_STATUS_VALUES
    )


def test_review_substrate_extraction_size_bounds_have_meaningful_headroom():
    counts = {
        module.__name__: len(
            pathlib.Path(module.__file__).read_text(encoding="utf-8").splitlines()
        )
        for module in (review_substrate, *_LEAVES)
    }
    assert counts["ouroboros.review_substrate"] <= 900
    assert all(count <= 1000 for count in counts.values())
    assert 300 <= counts["ouroboros.review_verdict"] <= 1000
