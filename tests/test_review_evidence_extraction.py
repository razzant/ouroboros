"""Structural contracts for the semantic-no-op review_evidence extraction.

Re-cut on the v7next tip: the reference's ``_ACCEPT_DELTA_CHILD_CAP`` and
``_accept_capability_deltas`` rows are superseded — upstream re-homed the
capability-delta aggregate into ``delegate_evidence`` (the facade reads it
back at call time), so those two names are neither moved nor pinned here.
Module size follows the shared size-ratchet tests and manifest; this historical
extraction must not impose a second cap or a minimum amount of retained code.
"""

from __future__ import annotations

import ast
import pathlib

from ouroboros import review_evidence, review_evidence_sections

REPO = pathlib.Path(__file__).parents[1]
_LEAVES = (review_evidence_sections,)

_MOVED_NAMES = (
    # The acceptance packet's assembler, budget, trajectory, obligations and claim-support helpers
    # left with the task-acceptance machine; these section helpers outlived it (the result review
    # and the verification summary still read them through the historical facade).
    "_ACCEPT_NOTES_CAP",
    "_accept_cap",
    "_accept_task_contract",
    "_accept_verification_summary",
    "_accept_receipt_exhibits",
    "_accept_effective_claims",
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


def test_review_evidence_keeps_its_patchable_seams():
    """Host-owned seams are documented to be readable — and patchable — at
    ``review_evidence.<name>``, so the commit-review evidence collector stays here
    (the acceptance packet assembler and its criteria annotator left with the
    task-acceptance machine). The commit-review status renderers no longer do:
    upstream 852ce967 cut them into the ``review_status_projection`` leaf, so the
    pin follows their real owner while the historical read still resolves."""
    assert review_evidence.collect_review_evidence.__module__ == "ouroboros.review_evidence"
    for name in ("build_review_projection", "build_review_status_payload"):
        assert name in vars(review_evidence), name
        assert getattr(review_evidence, name).__module__ == (
            "ouroboros.review_status_projection"
        ), name


def test_review_evidence_facade_reexports_every_moved_identity():
    """``review_evidence`` keeps the exact objects, so the loop, the review tool
    and reflection see no identity change at their historical import site."""
    for name in _MOVED_NAMES:
        assert hasattr(review_evidence, name), name
        assert getattr(review_evidence, name) is getattr(review_evidence_sections, name), name
    assert set(_MOVED_NAMES) <= set(vars(review_evidence_sections))
