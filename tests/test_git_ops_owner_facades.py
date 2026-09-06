"""Facade-identity contract for the v7 G1 supervisor/git_ops.py leaf owners.

Every member the G1 split moved out of ``supervisor/git_ops.py`` keeps a git_ops
re-export under its historical name, so existing callers and monkeypatching
tests keep working unchanged — the git_ops binding IS the leaf's object, the
same way the queue and events splits pin their leaves. The protection parity
clause pins the inventories: ``supervisor/git_ops.py`` is a release-invariant
path and release machinery, and the split moved the destructive remote/update/
reset/rescue bodies out of it without moving any of the risk, so an inventory
naming only the parent would leave the code that actually resets and rescues
the repository unguarded.

v7next transplant note: the reference (ouroboros_v7_wip @ 9f691656) derives
both inventories from one ``GIT_OPS_FAMILY_PATHS`` list inside
runtime_mode_policy.py. On this tree the closure is strictly ADDITIVE literal
entries in the protected file (the D04 registry-leaf precedent); the derived
re-cut stays with the oracle delta. D35 extends the transplant proof through
f-string expressions, so ``prepare_managed_update`` and ``safe_restart`` now
sit with their semantic leaf owners too.
"""

from __future__ import annotations

import importlib

# leaf module -> every member the leaf owns (git_ops re-exports each name).
GIT_OPS_LEAF_OWNERS: dict[str, str] = {
    "git_ops_remotes": (
        "configure_remote configure_personal_remote _configure_credential_helper "
        "push_to_remote"
    ),
    "git_ops_updates": (
        "list_versions list_commits ensure_official_update_remote "
        "managed_update_remote_url "
        "list_official_update_tags compute_managed_update_status prepare_managed_update"
    ),
    "git_ops_reset": (
        "_compute_ref_ahead_count _ref_points_at_ref preserve_local_ref_branch "
        "_preserve_branch_for_official_reset _run_git_resilient "
        "_admission_gate_for_unsynced_tree checkout_and_reset "
        "sync_runtime_dependencies import_test safe_restart"
    ),
    "git_ops_rescue": (
        "_collect_repo_sync_state _copy_untracked_for_rescue _atomic_write_bytes "
        "_create_rescue_snapshot _link_rescue_to_evolution_transaction "
        "_rescue_untracked_incomplete rescue_before_destructive_rollback rescue_into_tx"
    ),
}


def test_git_ops_owner_facade_preserves_identity():
    import supervisor.git_ops as git_ops

    for leaf, names in GIT_OPS_LEAF_OWNERS.items():
        module = importlib.import_module(f"supervisor.{leaf}")
        for name in names.split():
            assert getattr(git_ops, name) is getattr(module, name), f"{leaf}.{name}"


def test_the_facade_reexports_the_fstring_update_and_restart_entry_points():
    """D35 moves the last two G1 spans without changing facade identity."""
    import supervisor.git_ops as git_ops

    assert git_ops.prepare_managed_update.__module__ == "supervisor.git_ops_updates"
    assert git_ops.safe_restart.__module__ == "supervisor.git_ops_reset"


def test_every_git_ops_leaf_is_protected_exactly_like_the_parent():
    """The leaves hold the destructive machinery; the inventories must say so.

    ``supervisor/git_ops.py`` is a release-invariant path (the agent may not
    rewrite it outside pro mode) and release machinery (a contributor proposal
    touching it is release-sensitive). The G1 split moved the remote, managed-
    update, checkout/reset and rescue bodies out of it and moved none of that
    risk, so an inventory naming only the parent would leave the code that
    actually resets and rescues the repository unguarded."""
    import pathlib

    from ouroboros.runtime_mode_policy import RELEASE_INVARIANT_PATHS, protected_path_category
    from scripts.run_external_review import _RELEASE_MACHINERY_PATHS

    repo = pathlib.Path(__file__).resolve().parents[1]
    family = {"supervisor/git_ops.py"} | {f"supervisor/{leaf}.py" for leaf in GIT_OPS_LEAF_OWNERS}

    for path in sorted(family):
        assert (repo / path).is_file(), path
        assert path in RELEASE_INVARIANT_PATHS, path
        assert protected_path_category(path) == "release-invariant", path
        assert path in _RELEASE_MACHINERY_PATHS, path


def test_git_ops_leaves_keep_hot_code_label_parity():
    """Managed-update conflict labelling does not name ``supervisor/git_ops.py``;
    the split must not silently upgrade or downgrade the label for code that
    merely moved — parent and leaves carry the SAME membership."""
    from supervisor.update_merge_policy import HOT_CODE_PATHS

    parent_is_hot = "supervisor/git_ops.py" in HOT_CODE_PATHS
    for leaf in GIT_OPS_LEAF_OWNERS:
        assert (f"supervisor/{leaf}.py" in HOT_CODE_PATHS) == parent_is_hot, leaf


def test_git_ops_family_constant_matches_the_tree_and_every_inventory():
    """Batch-5 item 15: the derived family constant is the rot-killer - every
    supervisor/git_ops*.py that EXISTS must be in the constant, and the
    constant must be fully absorbed by each protection inventory that covers
    the parent (a future leaf added to the tree but not the family, or a
    family member dropped from an inventory, fails here loudly)."""
    import pathlib
    from ouroboros.runtime_mode_policy import (
        GIT_OPS_FAMILY_PATHS,
        RELEASE_INVARIANT_PATHS,
    )
    repo = pathlib.Path(__file__).resolve().parents[1]
    on_disk = {p.relative_to(repo).as_posix()
               for p in (repo / "supervisor").glob("git_ops*.py")}
    assert on_disk == GIT_OPS_FAMILY_PATHS
    assert GIT_OPS_FAMILY_PATHS <= RELEASE_INVARIANT_PATHS


def test_git_ops_family_is_absorbed_by_the_contributor_release_inventory():
    """F2 close-out conformance (item 7a): the hermetic contributor script
    hand-lists the family on purpose (it must stay stdlib-only), so THIS test
    is the coupling - a leaf added to the derived family without the script's
    inventory fails here."""
    import importlib.util
    import pathlib
    from ouroboros.runtime_mode_policy import GIT_OPS_FAMILY_PATHS
    repo = pathlib.Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "rer_probe", repo / "scripts" / "run_external_review.py")
    mod = importlib.util.module_from_spec(spec)
    import sys
    sys.modules["rer_probe"] = mod
    spec.loader.exec_module(mod)
    assert GIT_OPS_FAMILY_PATHS <= mod._RELEASE_MACHINERY_PATHS
