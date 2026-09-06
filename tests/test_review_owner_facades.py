"""Facade-identity contract for the v7 L-C review-stack leaf owners.

Every member the D06 split moved out of ``ouroboros/tools/review.py`` keeps a
parent re-export under its historical name, so existing callers and
monkeypatching tests keep working unchanged while the split lands.

Re-derived on the v7next tip from the reference suite: the reference's
``review_session_verdict`` block is superseded (upstream made the same
extraction as ``review_verdict_extraction`` and pins it with its own tests);
the ``claude_advisory_review`` blocks are re-derived on the native-episode
form under the organ's public rename (``preflight_review_prompt`` /
``preflight_review_run``) — the SDK-era rows
(``advisory_route_requires_api_key``, ``_advisory_session_deltas``,
``_advisory_sdk_budget``, ``_changed_paths``) died with the transport, and the
three deterministic preflights live with their upstream owner
``commit_admission`` (the parent keeps the alias seams);
``DEFAULT_REVIEW_MODEL_TIMEOUT_SEC`` / ``_review_model_timeout_sec`` are
retired with the adaptive-timeout contract while ``_parse_model_response``
lives with its upstream owner ``tools/review_response`` (the parent re-imports
it).
"""

from __future__ import annotations

import importlib

# parent module -> {leaf module -> every member the leaf owns
# (the parent re-exports each name)}.
REVIEW_LEAF_OWNERS: dict[str, dict[str, str]] = {
    "ouroboros.tools.review": {
        "ouroboros.tools.review_multi_model": (
            "MAX_MODELS CONCURRENCY_LIMIT _CONSTITUTIONAL_PREAMBLE "
            "_handle_multi_model_review _review_output_budget _query_model "
            "_multi_model_review_async"
        ),
    },
    "ouroboros.tools.claude_advisory_review": {
        "ouroboros.tools.preflight_review_prompt": (
            "_MAX_DIFF_CHARS_ERROR _get_staged_diff _get_changed_file_list "
            "_build_blocking_history_section _build_advisory_prompt"
        ),
        "ouroboros.tools.preflight_review_run": (
            "_ADVISORY_PROMPT_MAX_CHARS "
            "_ADVISORY_EXTRACT_CONTRACT _ADVISORY_SESSION_MAX_SECONDS "
            "_resolve_fallback_model _llm_extract_advisory_items "
            "_check_expected_items advisory_review_route advisory_slot_enabled "
            "advisory_gate_unavailability_reason advisory_gate_unavailable "
            "_run_advisory_delegated _note_meta_error _run_claude_advisory "
            "_is_clean_verdict _needs_fallback_extraction _parse_advisory_output "
            "_is_checklist_array"
        ),
    },
}


def test_the_deterministic_preflights_live_with_commit_admission():
    """The reference rows moved the three deterministic preflights into the
    advisory prompt leaf; upstream had already extracted them into
    commit_admission (Q3=A SSOT) — the upstream home wins and the parent's
    alias imports stay the gate's monkeypatch seams."""
    import ouroboros.commit_admission as owner
    import ouroboros.tools.claude_advisory_review as parent

    assert parent._release_metadata_preflight is owner.release_metadata_preflight
    assert (
        parent._auto_sync_release_metadata_if_needed
        is owner.auto_sync_release_metadata_if_needed
    )
    assert (
        parent._syntax_preflight_staged_py_files
        is owner.syntax_preflight_staged_py_files
    )


def test_review_owner_facades_preserve_identity():
    for parent_name, leaves in REVIEW_LEAF_OWNERS.items():
        parent = importlib.import_module(parent_name)
        for leaf_name, names in leaves.items():
            leaf = importlib.import_module(leaf_name)
            for name in names.split():
                assert getattr(parent, name) is getattr(leaf, name), f"{leaf_name}.{name}"


def test_parse_model_response_lives_with_its_upstream_owner():
    """The reference row moved `_parse_model_response` into the multi-model
    leaf; upstream had already extracted it into tools/review_response — the
    upstream home wins and the parent re-import stays the single alias."""
    import ouroboros.tools.review as parent
    import ouroboros.tools.review_response as owner

    assert parent._parse_model_response is owner.parse_model_response


def test_review_leaves_inherit_the_unlabeled_merge_class():
    """Managed-update conflict labelling names neither review parent, so the
    leaves inherit that — parity, not blanket labelling (the queue split pins
    the same rule in the labeled direction)."""
    from supervisor.update_merge_policy import HOT_CODE_PATHS

    for parent_name, leaves in REVIEW_LEAF_OWNERS.items():
        assert parent_name.replace(".", "/") + ".py" not in HOT_CODE_PATHS, parent_name
        for leaf_name in leaves:
            assert leaf_name.replace(".", "/") + ".py" not in HOT_CODE_PATHS, leaf_name


def test_advisory_leaves_are_mandatory_review_stack_context():
    """The advisory preflight's prompt/run leaves decide what the advisory
    gate sees, so they are force-include review-stack artifacts exactly like
    the scope leaves (F2 close-out conformance, item 2)."""
    from ouroboros.tools.review_context_atlas import _REVIEW_STACK_PATHS
    assert "ouroboros/tools/preflight_review_prompt.py" in _REVIEW_STACK_PATHS
    assert "ouroboros/tools/preflight_review_run.py" in _REVIEW_STACK_PATHS
