"""Registration persistence policy and the STARTED-row field tables.

The leaf module the delegate custody core leans on at the module line gate
(both ``delegate_custody.py`` and ``tools/delegate.py`` sit exactly on the
1600-line ceiling): the #362 persistence decision lives here, and the
STARTED-row shape tables move here with it so the core pays for the marker
by extraction, not compression.

#362 (the f9356572 A3 remediation, ported): on engines that support the
delegated ``workspaceRoot`` field the run registers the user's STABLE target
root as its project — an identity that outlives any individual run. The old
retire class deleted that registration at settlement (custody rows,
settle_run, pending invocations, retry binding, the orphan sweep), so a
user's own project vanished when a delegated run finished. A registration
marked persistent survives every retirement path; only ownership of the
one-shot snapshot registrations is discharged.
"""

from __future__ import annotations

from typing import Tuple

from ouroboros._usage_rows import REVIEW_ATTRIBUTION_KEYS


def persistent_registration(execution_root: str, access: str) -> bool:
    """Is this run's project registration a durable user identity?

    True exactly when the engine bound a stable execution workspace
    (``workspaceRoot`` supported => non-empty execution root) and the run
    writes for the user's own tree (``workspace_write`` or ``full``): that registration
    names the user's project, not a disposable snapshot, and must outlive
    the run (#362).
    """
    from ouroboros.subagents import is_mutating_delegated_access

    return bool(str(execution_root or "").strip()) and is_mutating_delegated_access(str(access or ""))


def record_persistent(record) -> bool:
    """Persistence of a DURABLE record, with a pre-marker upgrade fallback.

    Rows written before the marker existed carry no ``project_persistent``
    key; deriving from the immutable stored REQUEST (the delegated
    ``execution.workspaceRoot`` plus ``access``) keeps an old pending retry
    from deleting the user's stable project. Rows that carry the key are
    authoritative (first-wins doctrine: never recompute from a live engine).
    """
    if not isinstance(record, dict):
        return False
    if "project_persistent" in record:
        return bool(record.get("project_persistent"))
    request = record.get("request")
    if not isinstance(request, dict):
        return False
    execution = request.get("execution")
    workspace_root = execution.get("workspaceRoot") if isinstance(execution, dict) else ""
    return persistent_registration(str(workspace_root or ""), str(request.get("access") or ""))


def resolve_registration(gateway, scope_root: str, execution_root: str, access: str):
    """Register (or adopt) the scope root and decide the marker in one step.

    Returns ``(project_id, owned_project_id, project_persistent)``: an
    existing registration is adopted unowned; a fresh one is owned by this
    start; #362 marks the stable-target case persistent so no retire path
    deletes the user's own project.
    """
    existing_project = gateway.find_project_id(scope_root)
    project_id = existing_project or gateway.register_project(scope_root)
    owned_project_id = "" if existing_project else project_id
    return project_id, owned_project_id, persistent_registration(execution_root, access)


# The STARTED row's string facts as ``(RunCustody attribute, row key)`` pairs —
# one table shared by the replay and the ``record_started`` emit.
STARTED_STR_FIELDS: Tuple[Tuple[str, str], ...] = tuple(
    (attr, "route" if attr == "route_id" else attr) for attr in (
        "task_id", "route_id", "model", "profile_id", "project_id", "root_task_id",
        "parent_task_id", "category", "source", *REVIEW_ATTRIBUTION_KEYS,
        "ledger_root", "idempotency_key", "invocation_id",
        "snapshot_id", "execution_root", "baseline_sha", "target_root",
        "authority_source", "access", "mode", "isolation",
        "selected_subagent_id", "config_fingerprint", "work_order_fingerprint",
        "work_order_coverage", "authority_fingerprint",
    )
)
# Progress carried forward from a previous row: an idempotent re-start writes a
# SECOND started row; replacing wholesale would forget a settlement and put a
# finished run back into the orphan sweep (which would cancel it).
STARTED_PROGRESS_FLAGS: Tuple[str, ...] = (
    "ledger_recorded", "settled", "containment_disclosed", "unread_disclosed",
    "output_artifact", "output_complete", "output_sha", "output_consumed",
    "patch_captured", "patch_disposed", "patch_apply_pending")
# Binding/authority facts are FIRST-WINS (R1-2): a later idempotent STARTED row
# may be minted by a context that no longer knows the original binding; the
# first recorded fact is authoritative and is never erased or retargeted.
STARTED_FIRST_WINS_FACTS: Tuple[str, ...] = (
    "snapshot_id", "execution_root", "baseline_sha", "target_root",
    "authority_source", "resource_ref", "selected_subagent_id",
    "config_fingerprint", "work_order_fingerprint", "work_order_coverage",
    "authority_fingerprint", "work_order_source_request", "category", "source",
    *REVIEW_ATTRIBUTION_KEYS)
