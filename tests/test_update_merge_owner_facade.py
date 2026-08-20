"""Facade-identity contract for the supervisor/update_merge.py leaf owners.

Every member the 1A split moved out of ``supervisor/update_merge.py`` keeps an
update_merge re-export under its historical name, so existing callers and
monkeypatching tests keep working unchanged — the update_merge binding IS the
leaf's object, the same way the queue and loop splits pin their leaves. The
hot-code label parity clause pins the OTHER direction of the loop-split rule:
``supervisor/update_merge.py`` is not a HOT_CODE_PATHS member, so a leaf that
merely moved code out of it must not silently acquire the label either.
"""

from __future__ import annotations

import importlib

# leaf module -> every member the leaf owns (update_merge re-exports each name).
UPDATE_MERGE_LEAF_OWNERS: dict[str, str] = {
    "update_merge_plan": (
        "_git_run _build_clean_merge_commit plan_managed_update_merge "
        "materialize_assisted_merge_live"
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
