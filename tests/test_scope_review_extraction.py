"""Structural contracts for the semantic-no-op scope review extraction.

``scope_review`` keeps the run: dispatch, the typed result vocabulary, the
pack-status and oversize translations, and the P3 authority decision. Two
owners sit below it — ``scope_review_pack`` (assembling the reviewer's pack)
and ``scope_review_budget`` (the PR #383 token arithmetic, re-derived from tip
bytes; the reference-era timeout constant stays retired to ``None``).
Cross-references run through the parent's call-time handle, and the parent
re-exports every moved identity.

``_load_canonical_context_docs`` stayed a facade def on purpose: it reads
``load_governance_doc`` inside an f-string (the byte gate refuses f-string
reads of rebindable globals) and tests rebind that name on the parent. The
budget leaf's private owner aliases (``_SCOPE_MODEL_DEFAULT``,
``_SCOPE_BUDGET_TOKEN_LIMIT``, ``_SCOPE_FAILCLOSED_WINDOW``,
``_SCOPE_MODEL_CONTEXT_WINDOW``, ``_shared_window_scaled_reserves``,
``_calibrated_input_token_limit``, ``_is_provider_oversize_error``) moved with
their readers and stay import-frozen exactly as before the split.
"""

from __future__ import annotations

import ast
import pathlib

from ouroboros.tools import (
    scope_review,
    scope_review_budget,
    scope_review_pack,
)


REPO = pathlib.Path(__file__).parents[1]

_LEAVES = (scope_review_pack, scope_review_budget)

_MOVED_OWNERS = {
    "_SCOPE_INPUT_TOKEN_LIMIT": scope_review_budget,
    "_SCOPE_MAX_TOKENS": scope_review_budget,
    "_SCOPE_OUTPUT_MARGIN_TOKENS": scope_review_budget,
    "_SCOPE_REVIEW_SLOT_TIMEOUT_SEC": scope_review_budget,
    "_effective_scope_input_limit": scope_review_budget,
    "_get_scope_model": scope_review_budget,
    "_provider_error_is_oversize": scope_review_budget,
    "_window_scaled_reserves": scope_review_budget,
    "_CANONICAL_CONTEXT_DOCS": scope_review_pack,
    "_CURRENT_TOUCHED_CONTEXT_SKIP_PREFIXES": scope_review_pack,
    "_DELETED_INLINE_MAX_BYTES": scope_review_pack,
    "_SCOPE_CONTEXT_MANIFEST": scope_review_pack,
    "_SCOPE_STABLE_PREFIX_LEN": scope_review_pack,
    "_ScopeAtlasNotAssembled": scope_review_pack,
    "_ScopePromptContext": scope_review_pack,
    "_build_review_history_section": scope_review_pack,
    "_build_scope_history_section": scope_review_pack,
    "_build_scope_prompt": scope_review_pack,
    "_carrier_span_only_paths": scope_review_pack,
    "_classify_deleted_for_inline": scope_review_pack,
    "_current_scope_context_manifest": scope_review_pack,
    "_degradable_diff_only_paths": scope_review_pack,
    "_gather_scope_packs": scope_review_pack,
    "_inline_deleted_file_pack": scope_review_pack,
    "_parse_staged_name_status": scope_review_pack,
    "_record_ladder_steps": scope_review_pack,
    "_render_touched_section": scope_review_pack,
    "_should_skip_current_touched_context": scope_review_pack,
}

# The parent keeps these: the result type, dispatch, the pack-status/oversize
# translations, the P3 authority decision, the entry point — and the one
# f-string-gated context-doc loader named above.
_PARENT_OWNED = (
    "ScopeReviewResult",
    "_SCOPE_REQUIRED_ITEMS",
    "_apply_scope_authority",
    "_call_scope_llm",
    "_handle_prompt_signals",
    "_load_canonical_context_docs",
    "_log_scope_result",
    "_low_context_skip_result",
    "_scope_oversize_result",
    "_scope_review_skipped_in_low_context",
    "_scope_sub_floor_finding",
    "run_scope_review",
)


def test_scope_review_leaves_never_import_their_parent_at_module_scope():
    for module in _LEAVES:
        tree = ast.parse(pathlib.Path(module.__file__).read_text(encoding="utf-8"))
        assert not any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "get_tools"
            for node in tree.body
        )
        for node in tree.body:
            if isinstance(node, ast.ImportFrom):
                assert node.module != "ouroboros.tools.scope_review", module.__name__
            if isinstance(node, ast.Import):
                assert all(
                    a.name != "ouroboros.tools.scope_review" for a in node.names
                ), module.__name__


def test_scope_review_facade_reexports_every_moved_identity():
    for name, owner in _MOVED_OWNERS.items():
        assert hasattr(scope_review, name), name
        assert getattr(scope_review, name) is getattr(owner, name), name
    owned = {name for module in _LEAVES for name in vars(module)}
    assert set(_MOVED_OWNERS) <= owned


def test_scope_review_keeps_the_run_and_the_result_vocabulary():
    module_source = pathlib.Path(scope_review.__file__).read_text(encoding="utf-8")
    defined = set()
    for node in ast.parse(module_source).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, ast.Assign):
            defined.update(t.id for t in node.targets if isinstance(t, ast.Name))
    assert set(_PARENT_OWNED) <= defined
    assert defined.isdisjoint(_MOVED_OWNERS)


def test_scope_review_leaves_are_review_stack_members():
    from ouroboros.tools.review_context_atlas import _REVIEW_STACK_PATHS, _is_force_include

    for module in _LEAVES:
        rel = pathlib.Path(module.__file__).relative_to(REPO).as_posix()
        assert rel in _REVIEW_STACK_PATHS, rel
        assert _is_force_include(rel), rel


def test_scope_review_extraction_size_bounds_have_meaningful_headroom():
    counts = {
        module.__name__: len(
            pathlib.Path(module.__file__).read_text(encoding="utf-8").splitlines()
        )
        for module in (scope_review, *_LEAVES)
    }
    assert all(count <= 1000 for count in counts.values()), counts
    assert counts["ouroboros.tools.scope_review"] <= 1000
    assert 500 <= counts["ouroboros.tools.scope_review_pack"] <= 1000
