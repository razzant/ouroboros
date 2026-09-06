"""Facade-identity contract for the v7 DEL1 delegate-family leaf owners.

Every member the DEL1 split moved out of the delegate family —
``ouroboros/delegate_custody.py``, ``ouroboros/tools/delegate_integration.py``,
``ouroboros/tools/subagent_integration.py`` — and the route-health family moved
out of ``ouroboros/subagents.py`` keeps a parent re-export under its historical
name for the DURATION of the v7 stream, so existing callers and monkeypatching
tests keep working unchanged while the split lands. This pins the facade
identity — the parent binding IS the leaf's object — and the hot-code label
parity for the leaves, the same way the queue and loop splits pin both for
theirs.

One reference row is deliberately absent: ``_capture_stranded_patch`` was
re-homed by upstream itself (public ``capture_stranded_patch`` in
``tools/delegate_integration.py``). The ``tools/delegate.py`` terminal leaf
landed as ``delegate_terminal_evidence.py`` — the ledger's
``tools/delegate_terminal.py`` name collided with upstream's own
``ouroboros/delegate_terminal.py`` (owner fork F-2=A rename).
"""

from __future__ import annotations

import importlib

# parent module -> {leaf module -> every member the leaf owns (parent re-exports each name)}.
DELEGATE_LEAF_OWNERS: dict[str, dict[str, str]] = {
    "ouroboros.delegate_custody": {
        "ouroboros.delegate_custody_reconcile": (
            "open_runs pending_invocations release_task_runs reconcile_task_runs "
            "reconcile_orphaned_runs _reconcile_each _recover_pending_invocation "
            "_retire_recovered_registration _reconcile_one"
        ),
    },
    "ouroboros.tools.delegate": {
        "ouroboros.tools.delegate_terminal_evidence": (
            "_containment_breach _NESTED_HOME_NOTE _NO_BOUNDARY_NOTE _containment_evidence "
            "_terminal_payload _access_evidence _record_containment _reported_cost "
            "_delivered_terminal_payload"
        ),
    },
    "ouroboros.tools.delegate_integration": {
        "ouroboros.tools.delegate_payload_patch": (
            "_reserved_payload_rel_path _snapshot_head_textual _write_payload_patch_artifacts "
            "_payload_reserved_paths _candidate_symlink_escapes _finalize_payload_apply "
            "integrate_payload_patch"
        ),
    },
    "ouroboros.tools.subagent_integration": {
        "ouroboros.tools.subagent_integration_delegated": (
            "_READY_CAPTURE_STATUSES _drift_refusal _locked_apply _manifest_capture_status "
            "_capture_failed_refusal _capture_at_disposition _delegated_disposition_refusal "
            "_unwritten_disposition_text _dispose_delegated _resolve_acknowledged_intent "
            "_integrate_delegated_patch"
        ),
    },
    "ouroboros.subagents": {
        "ouroboros.subagent_route_health": (
            "route_health _exhausted_window _model_scope_matches _cooldown_active"
        ),
    },
}


def test_delegate_owner_facades_preserve_identity() -> None:
    for parent_name, leaves in DELEGATE_LEAF_OWNERS.items():
        parent = importlib.import_module(parent_name)
        for leaf_name, members in leaves.items():
            leaf = importlib.import_module(leaf_name)
            for member in members.split():
                assert getattr(parent, member) is getattr(leaf, member), (
                    f"{parent_name}.{member} is not the {leaf_name} object"
                )


def test_delegate_leaves_share_their_parents_hot_code_label_parity() -> None:
    """The delegate family is UNLABELED in the managed-update conflict policy at
    the DEL1 base, so its leaves inherit that — parity, not blanket labelling
    (the queue split pins the same property from the labeled side, and the
    D07 control leaves are pinned hot by tests/test_control_extraction.py)."""
    from supervisor.update_merge_policy import HOT_CODE_PATHS

    parents = (
        "ouroboros/delegate_custody.py",
        "ouroboros/tools/delegate.py",
        "ouroboros/tools/delegate_integration.py",
        "ouroboros/tools/subagent_integration.py",
        "ouroboros/subagents.py",
    )
    leaves = tuple(
        leaf.replace(".", "/") + ".py"
        for owners in DELEGATE_LEAF_OWNERS.values()
        for leaf in owners
    )
    for path in parents:
        assert path not in HOT_CODE_PATHS, f"{path} gained a hot-code label; relabel its leaves too"
    for path in leaves:
        assert path not in HOT_CODE_PATHS, f"{path} must keep parity with its unlabeled parent"
