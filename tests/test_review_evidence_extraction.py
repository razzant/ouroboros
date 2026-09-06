"""Structural contracts for the semantic-no-op review_evidence extraction.

Re-cut on the v7next tip: the reference's ``_ACCEPT_DELTA_CHILD_CAP`` and
``_accept_capability_deltas`` rows are superseded — upstream re-homed the
capability-delta aggregate into ``delegate_evidence`` (the facade reads it
back at call time), so those two names are neither moved nor pinned here.
"""

from __future__ import annotations

import ast
import pathlib

from ouroboros import review_evidence, review_evidence_sections
from ouroboros.tools import review_context_atlas

REPO = pathlib.Path(__file__).parents[1]
_LEAVES = (review_evidence_sections,)

_MOVED_NAMES = (
    "collect_turn_diff",
    "_ACCEPT_RESULT_CAP",
    "_ACCEPT_ARGS_CAP",
    "_ACCEPT_NOTES_CAP",
    "_ACCEPT_TRAJECTORY_MAX_CALLS",
    "_ACCEPT_ARTIFACT_PREVIEW_CAP",
    "_ACCEPT_ARTIFACT_PREVIEW_MAX_BYTES",
    "_ACCEPT_TOTAL_BUDGET",
    "_ACCEPT_OBLIGATIONS_MAX",
    "_ACCEPT_RETRIEVAL_URLS_MAX",
    "ACCEPTANCE_PROMPT_OVERHEAD_CHARS",
    "_ACCEPT_DENSE_CHARS_PER_TOKEN",
    "AcceptancePacketBudget",
    "acceptance_packet_budget_chars",
    "obligation_is_pending",
    "_accept_obligation_row",
    "task_acceptance_evidence_revision",
    "_accept_redact_cap",
    "_accept_task_contract",
    "_accept_protected_set",
    "_accept_verification_summary",
    "_accept_receipt_exhibits",
    "_accept_effective_claims",
    "_accept_claim_support_refs",
    "_accept_trajectory",
    "_accept_artifact_manifest",
    "_accept_enforce_budget",
    "_owner_content_projection",
    "_accept_owner_directives",
)


def test_review_evidence_leaf_is_a_non_catalog_owner_without_backedges():
    for module in (review_evidence, *_LEAVES):
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
                assert node.module != "ouroboros.review_evidence", module.__name__
            if isinstance(node, ast.Import):
                assert all(
                    a.name != "ouroboros.review_evidence" for a in node.names
                ), module.__name__


def test_review_evidence_keeps_the_packet_assembler_and_its_patchable_seams():
    """Host-owned seams are documented to be readable — and patchable — at
    ``review_evidence.<name>``, so the packet assembler and the annotator stay
    here. The commit-review status renderers no longer do: upstream 852ce967 cut
    them into the ``review_status_projection`` leaf, so the pin follows their real
    owner while the historical ``review_evidence.<name>`` read still resolves."""
    for name in (
        "build_task_acceptance_evidence",
        "annotate_criteria_evidence_resolution",
        "collect_review_evidence",
        "format_review_evidence_for_prompt",
    ):
        assert getattr(review_evidence, name).__module__ == "ouroboros.review_evidence", name
    for name in ("build_review_projection", "build_review_status_payload"):
        assert name in vars(review_evidence), name
        assert getattr(review_evidence, name).__module__ == (
            "ouroboros.review_status_projection"
        ), name
    for name in (
        "collect_turn_diff",
        "acceptance_evidence_ref_vocabulary",
        "resolve_criteria_evidence_refs",
    ):
        assert name in vars(review_evidence), name


def test_review_evidence_facade_reexports_every_moved_identity():
    """``review_evidence`` keeps the exact objects, so the loop, the review tool,
    the advisory surface, reflection and the acceptance tests see no identity
    change at their historical import site."""
    for name in _MOVED_NAMES:
        assert hasattr(review_evidence, name), name
        assert getattr(review_evidence, name) is getattr(review_evidence_sections, name), name
    assert set(_MOVED_NAMES) <= set(vars(review_evidence_sections))


def test_review_evidence_section_owner_is_forced_into_every_review_pack():
    """The acceptance packet's section author is part of the immune system's
    review surface exactly as its parent is: a review pack owes it in full
    instead of treating it as a budget-selected dependency."""
    for rel in (
        "ouroboros/review_evidence.py",
        "ouroboros/review_evidence_sections.py",
    ):
        assert rel in review_context_atlas._REVIEW_STACK_PATHS, rel


def test_review_evidence_extraction_size_bounds_have_meaningful_headroom():
    counts = {
        module.__name__: len(
            pathlib.Path(module.__file__).read_text(encoding="utf-8").splitlines()
        )
        for module in (review_evidence, *_LEAVES)
    }
    assert all(count <= 1000 for count in counts.values())
    assert 600 <= counts["ouroboros.review_evidence_sections"] <= 1000
    assert 600 <= counts["ouroboros.review_evidence"] <= 1000
