"""Facade-identity contract for the v7 L-C review-stack leaf owners.

Every member the D06 split moved out of ``ouroboros/tools/review.py`` keeps a
parent re-export under its historical name, so existing callers and
monkeypatching tests keep working unchanged while the split lands.

Re-derived on the v7next tip from the reference suite: the reference's
``review_session_verdict`` block is superseded (upstream made the same
extraction as ``review_verdict_extraction`` and pins it with its own tests);
the ``claude_advisory_review`` blocks left with the advisory pipeline itself
(decision 3A: the preflight is ``review_change(surface=preflight)``, and the
three deterministic preflights it ran stay with ``commit_admission``, called
by ``commit_gate.deterministic_preflight``);
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
}


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
