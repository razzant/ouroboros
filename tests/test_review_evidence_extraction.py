"""Structural contracts for the semantic-no-op review_evidence extraction."""

from __future__ import annotations

import ast
import pathlib

from ouroboros import review_evidence, review_evidence_sections
from ouroboros.tools import review_context_atlas
from ouroboros.tool_module_inventory import (
    build_frozen_tool_manifest,
    discover_tool_module_inventory,
    load_frozen_tool_modules,
)


REPO = pathlib.Path(__file__).parents[1]
TOOLS = REPO / "ouroboros" / "tools"

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
    "_ACCEPT_DELTA_CHILD_CAP",
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
    "_accept_capability_deltas",
)


def test_review_evidence_leaf_is_a_non_catalog_owner_without_backedges(tmp_path):
    for module in (review_evidence, *_LEAVES):
        tree = ast.parse(pathlib.Path(module.__file__).read_text(encoding="utf-8"))
        assert not any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "get_tools"
            for node in tree.body
        )
    for module in _LEAVES:
        tree = ast.parse(pathlib.Path(module.__file__).read_text(encoding="utf-8"))
        assert not any(
            isinstance(node, ast.ImportFrom)
            and node.module == "ouroboros.review_evidence"
            for node in ast.walk(tree)
        )
        assert not any(
            isinstance(node, ast.Import)
            and any(alias.name == "ouroboros.review_evidence" for alias in node.names)
            for node in ast.walk(tree)
        )

    source_inventory = discover_tool_module_inventory(TOOLS)
    for module in (review_evidence, *_LEAVES):
        assert module.__name__.rsplit(".", 1)[-1] not in source_inventory.tool_modules
    manifest = tmp_path / "_frozen_tool_modules.v1.json"
    build_frozen_tool_manifest(TOOLS, manifest)
    assert load_frozen_tool_modules(manifest) == source_inventory.tool_modules


def test_review_evidence_keeps_the_packet_assembler_and_its_patchable_seams():
    """Two host-owned seams are documented to be readable — and patchable — at
    ``review_evidence.<name>``: the host-collected working-tree diff that must
    override any agent-supplied ``repo_diff``, and the D-Q5 evidence-ref
    vocabulary/resolver the fail-closed annotator reads. Both are read through
    THIS module's globals, so the packet assembler and the annotator stay here
    with the review status/summary projections this module has always owned."""
    for name in (
        "build_task_acceptance_evidence",
        "annotate_criteria_evidence_resolution",
        "collect_review_evidence",
        "format_review_evidence_for_prompt",
        "build_review_projection",
        "build_review_status_payload",
    ):
        assert getattr(review_evidence, name).__module__ == "ouroboros.review_evidence", name
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
