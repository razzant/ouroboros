"""Facade-identity contract for the supervisor/update_merge.py leaf owners.

Every member that lives in a leaf of the update engine's two-module-plus split
(the redesign's own ``update_candidate`` boundary and the re-cut
``update_merge_plan`` planner leaf) keeps an update_merge re-export under its
historical name, so existing callers and monkeypatching tests keep working
unchanged — the update_merge binding IS the leaf's object, the same way the
queue and loop splits pin their leaves. The hot-code label parity clause pins
the OTHER direction of the loop-split rule: ``supervisor/update_merge.py`` is
not a HOT_CODE_PATHS member, so a leaf that merely moved code out of it must
not silently acquire the label either.
"""

from __future__ import annotations

import importlib

# leaf module -> every member the leaf owns (update_merge re-exports each name).
UPDATE_MERGE_LEAF_OWNERS: dict[str, str] = {
    "update_merge_plan": (
        "_build_clean_merge_commit plan_managed_update_merge "
        "materialize_assisted_merge_live"
    ),
    "update_candidate": (
        "_MERGE_NEUTRAL_FLAGS _git_run _merge_head_sha _preserve_failed_update_attempt "
        "_rev_parse existing_failed_update_ref live_unmerged_paths "
        "UpdateTxCorrupt managed_assisted_marker_check managed_tests_evidence_covers "
        "record_managed_tests_evidence record_managed_tests_proof "
        "update_tx_phase update_tx_phase_or_keep worktree_snapshot_tree "
        "find_update_stash_sha restore_stash_with_marker restore_update_stash "
        "stash_local_changes_for_update lookup_update_stash "
        "destructive_apply_guard project_version_carriers"
    ),
}


def test_update_merge_owner_facade_preserves_identity():
    import supervisor.update_merge as update_merge

    for leaf, names in UPDATE_MERGE_LEAF_OWNERS.items():
        module = importlib.import_module(f"supervisor.{leaf}")
        for name in names.split():
            assert getattr(update_merge, name) is getattr(module, name), f"{leaf}.{name}"


def test_update_merge_leaves_keep_hot_code_label_parity():
    """Managed-update conflict labelling does not name ``supervisor/update_merge.py``;
    the split must not silently upgrade or downgrade the label for code that merely
    moved — parent and leaves carry the SAME membership."""
    from supervisor.update_merge_policy import HOT_CODE_PATHS

    parent_is_hot = "supervisor/update_merge.py" in HOT_CODE_PATHS
    for leaf in UPDATE_MERGE_LEAF_OWNERS:
        assert (f"supervisor/{leaf}.py" in HOT_CODE_PATHS) == parent_is_hot, leaf
    # The shared span resolver is new update machinery, not moved hot code:
    # same parity rule.
    assert ("supervisor/update_carriers.py" in HOT_CODE_PATHS) == parent_is_hot


def test_update_engine_split_keeps_release_invariant_protection():
    """``supervisor/update_merge.py`` is a release-invariant path; the re-split
    moved the planner/materializer bodies (and added the span resolver the
    engine executes under the update lock) without moving any of the risk, so
    the protecting inventory must cover them too — the same closure the G1
    git_ops split pinned. ``supervisor/update_candidate.py`` rides the same
    parity pin: upstream's own redesign split it out without listing it (the
    gap the F2.4 lane disclosed), closed additively in F3 (owner FYI,
    additive-literal precedent D10/#419)."""
    from ouroboros.runtime_mode_policy import RELEASE_INVARIANT_PATHS, protected_path_category

    for path in (
        "supervisor/update_merge.py",
        "supervisor/update_merge_policy.py",
        "supervisor/update_merge_plan.py",
        "supervisor/update_carriers.py",
        "supervisor/update_candidate.py",
    ):
        assert path in RELEASE_INVARIANT_PATHS, path
        assert protected_path_category(path) == "release-invariant", path
