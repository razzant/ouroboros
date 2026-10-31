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

A root Ouroboros mints for one task, run or review (a registry checkout, a
folderless read-only scratch, a review checkout or input view without a sticky
thread) is not an identity at all: it declares ``scope.ephemeral`` and registers
nothing, so it never becomes a permanent engine project or a ghost later.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from ouroboros._usage_rows import REVIEW_ATTRIBUTION_KEYS

# Does this engine's ``POST /v2/projects`` answer carry ``created``? Learned per engine
# identity (version, build) from the answers themselves: the field's presence decides,
# never a version. Unobserved or absent keeps the previous GET-then-POST flow.
_CREATED_REPORTED: Dict[Tuple[str, str], bool] = {}


def review_owned_source(source: Any) -> bool:
    """Does this custody row's durable ``source`` name a REVIEW surface?

    True for every ``review_substrate*`` spelling. The reading lives beside the
    STARTED-row tables because ``source`` is one of their first-wins binding
    facts, and every delegation-domain consumer of a custody row shares this
    ONE reading of it (issue #1006).
    """
    return str(source or "").startswith("review_substrate")


# The evidence-only purpose of a private checking copy (#1609): a researcher or an
# external reviewer may run configured checks there, and its copy may name the tree
# it was made from, but nothing it produces is ever integrated into that tree.
INVESTIGATION_AUTHORITY = "investigation"
_NO_APPLY_AUTHORITY_SOURCES = frozenset({INVESTIGATION_AUTHORITY, "readonly"})
INVESTIGATION_NOTE = "Evidence-only investigation: the captured changes are retained as evidence and never applied."
NO_CHANGES_NOTE = ("No changes: the run's result holds exactly the files its copy started from (files Git ignores "
                   "are never part of a result), so it closed itself; there is nothing to apply or reject.")


def no_write_authority(root: str, *, investigation: bool) -> dict:
    """The mutation-authority facts of a run that writes nothing back: read-only, or an
    evidence-only investigation whose root is provenance, never an apply target."""
    return {"target_root": root, "source": INVESTIGATION_AUTHORITY if investigation else "readonly",
            "capture_mode": "evidence" if investigation else "none"}


def _fact(record: Any, key: str) -> Any:
    return record.get(key) if isinstance(record, dict) else getattr(record, key, None)


def evidence_only(record: Any) -> bool:
    """The positive evidence-only purpose (#1609): no read of its capture asks for
    apply/reject. Keyed on the recorded investigation source, not on
    ``candidate_apply_possible`` being false, which also covers degenerate rows
    with no source and no capture binding."""
    return str(_fact(record, "authority_source") or "") == INVESTIGATION_AUTHORITY


def candidate_apply_possible(record: Any) -> bool:
    """Could this run's captured work be integrated into the tree it was made from?

    ONE reading of the first-wins authority facts, shared by every integration
    branch and every observer: a ``RunCustody``, a pending-invocation row or an
    ``invocation_record`` all carry ``authority_source``, ``snapshot_id`` and
    ``resource_ref``. False on the evidence-only (or readonly) authority source,
    or on no recorded source over no capture binding at all: nothing was captured
    that could be applied. That absence does not say the run never wrote, which
    ``independent_review_without_apply`` must prove separately. Every other shape,
    an unknown source included, stays possible: a conservative answer, not
    proof of permission.
    """
    if _fact(record, "execution_mode") == "plan" or _fact(record, "mode") == "plan":
        return False
    source = str(_fact(record, "authority_source") or "")
    if source in _NO_APPLY_AUTHORITY_SOURCES:
        return False
    if source:
        return True
    ref = _fact(record, "resource_ref")
    directory_copy = (isinstance(ref, dict) and ref.get("workspace_kind") == "directory"
                      and ref.get("strategy") == "copy")
    return bool(str(_fact(record, "snapshot_id") or "")) or directory_copy


def _recorded_readonly_ask(record: Any) -> bool:
    """The read-only reviewer's own recorded shape: a STARTED row's granted
    ``access``/``mode``, or a pending row's wire shape, is readonly Ask
    with no private execution binding. Absent shape facts prove nothing."""
    request = _fact(record, "request_shape") or _fact(record, "request")
    request = request if isinstance(request, dict) else {}
    return (str(_fact(record, "access") or request.get("access") or "") == "readonly"
            and str(_fact(record, "mode") or request.get("mode") or "") == "ask"
            and not request.get("execution") and not str(_fact(record, "execution_root") or ""))


def independent_review_without_apply(record: Any) -> bool:
    """A run its REVIEW panel owns that cannot change the reviewed candidate.

    True only for a recognized review-owned ``source`` together with positive
    no-write evidence: the evidence-only (or readonly) authority a #1609 start
    records, or, on a row without that fact (an explicitly read-only reviewer
    seat, or one from before it), its recorded readonly Ask shape with no
    capture binding. An unknown or missing source, or missing shape, is never
    independent.
    """
    if not review_owned_source(_fact(record, "source")):
        return False
    authority = str(_fact(record, "authority_source") or "")
    if authority:
        return authority in _NO_APPLY_AUTHORITY_SOURCES
    return not candidate_apply_possible(record) and _recorded_readonly_ask(record)


def persistent_registration(execution_root: str, access: str) -> bool:
    """Is this run's project registration a durable user identity?

    True exactly when the engine bound a stable execution workspace
    (``workspaceRoot`` supported => non-empty execution root) and the run
    writes for the user's own tree (``workspace_write`` or ``full``): that registration
    names the user's project, not a disposable snapshot, and must outlive
    the run (#362).
    """
    from ouroboros.configured_subagents import SESSION_ACCESS_PROFILES

    return bool(str(execution_root or "").strip()) and access in SESSION_ACCESS_PROFILES


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


def minted_review_root(root: str, drive: Any) -> bool:
    """A tree minted for one review: an isolated review checkout (``<data>/state/review_checkouts``),
    a retained review-input snapshot (``source_handles/review_inputs`` of a task under the data
    root) or a checkout the subagent worktree registry holds (a child's copy, a body candidate).
    Any other root, an owner's project or the system repository, keeps its registration
    wherever it lives, even under the data root."""
    import pathlib

    from ouroboros.subagent_worktrees import registered_checkout
    from ouroboros.tools.review_subject import CHECKOUT_SUBDIR

    path, data = pathlib.Path(root).resolve(), pathlib.Path(drive).resolve()
    parts = path.relative_to(data).parts if path.is_relative_to(data) else ()
    return (parts[:2] == ("state", CHECKOUT_SUBDIR)
            or any(parts[i:i + 2] == ("source_handles", "review_inputs") for i in range(len(parts) - 1))
            or registered_checkout(root))


def resolve_registration(gateway, scope_root: str, execution_root: str, access: str, *, minted: bool = False):
    """Register (or adopt) the scope root and decide the markers in one step.

    Returns ``(project_id, owned_project_id, project_persistent, ephemeral)``. A
    ``minted`` root on an engine that accepts ``scope.ephemeral``
    (``config.CLAUDEXOR_EPHEMERAL_SCOPE_MIN_VERSION``) registers nothing: ``("",
    "", False, True)``. Otherwise an existing registration is adopted unowned; a
    fresh one is owned by this start; #362 marks the stable-target case
    persistent so no retire path deletes the user's own project. Ownership comes
    from the engine's ``created`` once its answers carry it (one POST, no
    ``GET /v2/projects`` over every project); until then the previous flow runs,
    and on an unobserved engine one optional POST beside a found root learns it.
    A gateway without ``project_registration`` (an older client or a test double)
    keeps the previous flow exactly.
    """
    from ouroboros.config import CLAUDEXOR_EPHEMERAL_SCOPE_MIN_VERSION
    from ouroboros.gateways.claudexor import ClaudexorUnavailable, engine_at_least

    version = str(getattr(gateway, "engine_version", "") or "")
    if minted and engine_at_least(version, CLAUDEXOR_EPHEMERAL_SCOPE_MIN_VERSION):
        return "", "", False, True
    persistent = persistent_registration(execution_root, access)
    register = getattr(gateway, "project_registration", None)
    identity = (version, str(getattr(gateway, "engine_build_sha", "") or ""))
    reported = _CREATED_REPORTED.get(identity) if register is not None else False
    existing = "" if reported else gateway.find_project_id(scope_root)
    if register is None or (existing and reported is False):
        project_id = existing or gateway.register_project(scope_root)
        return project_id, ("" if existing else project_id), persistent, False
    try:
        project_id, created = register(scope_root)
    except ClaudexorUnavailable:
        if not existing:
            raise
        return existing, "", persistent, False  # the previous flow needed no POST here
    _CREATED_REPORTED[identity] = created is not None
    owned = created if created is not None else not (existing or reported)
    return project_id, (project_id if owned else ""), persistent, False


# How a run's ``maxSeconds`` was decided (#1196), recorded on the START_REQUESTED
# and STARTED rows as ``max_seconds_basis``. Only a cap the nanny ASKED for — an
# explicit finite leaf cap, at most narrowed by the engine's schema bound — makes
# a later ``wall_clock_exceeded`` a finite-leaf expiry a continuation may follow;
# a cap the nanny's own deadline or lifetime derived (or narrowed) expiring IS
# that deadline, and a row that predates the field is unknown, never assumed.
CAP_BASIS_REQUESTED = "requested"
CAP_BASIS_REQUESTED_CLAMPED_SCHEMA = "requested_clamped_by_schema"
CAP_BASIS_REQUESTED_CLAMPED_DEADLINE = "requested_clamped_by_deadline"
CAP_BASIS_REQUESTED_CLAMPED_LIFETIME = "requested_clamped_by_lifetime"
CAP_BASIS_DEADLINE_DERIVED = "deadline_derived"
CAP_BASIS_LIFETIME_DERIVED = "lifetime_derived"
CAP_BASIS_OPERATION_WINDOW = "operation_window"
FINITE_LEAF_CAP_BASES = frozenset({CAP_BASIS_REQUESTED, CAP_BASIS_REQUESTED_CLAMPED_SCHEMA})

# The STARTED row's string facts as ``(RunCustody attribute, row key)`` pairs —
# one table shared by the replay and the ``record_started`` emit.
STARTED_STR_FIELDS: Tuple[Tuple[str, str], ...] = tuple(
    (attr, "route" if attr == "route_id" else attr) for attr in (
        "task_id", "route_id", "model", "profile_id", "project_id", "root_task_id",
        "parent_task_id", "category", "source", *REVIEW_ATTRIBUTION_KEYS,
        "ledger_root", "idempotency_key", "invocation_id",
        "snapshot_id", "execution_root", "execution_binding_fingerprint", "baseline_sha", "target_root",
        "authority_source", "access", "mode", "isolation",
        "execution_mode", "effective_access", "thread_id",
        "selected_subagent_id", "config_fingerprint", "work_order_fingerprint",
        "work_order_coverage", "authority_fingerprint",
        # The prior run this start explicitly continues ("" for every ordinary
        # start) and, when it continues IN that run's snapshot, its own capture
        # identity and the task that keeps the snapshot's baseline and lock.
        "continuation_of", "capture_id", "snapshot_task_id",
    )
)
# None means an old row omitted the choice; '' is a captured default choice. ``row_effort`` is
# the configured row's pin ('' = Auto) beside ``effort``, the LEAF level the engine ran.
STARTED_MAPPING_FIELDS = ("authority_ceiling", "promotion")
STARTED_OPTION_FIELDS = ("effort", "processing_preference", "row_effort")
# Progress carried forward from a previous row: an idempotent re-start writes a
# SECOND started row; replacing wholesale would forget a settlement and put a
# finished run back into the orphan sweep (which would cancel it).
STARTED_PROGRESS_FLAGS: Tuple[str, ...] = (
    "ledger_recorded", "settled", "containment_disclosed", "unread_disclosed",
    "output_artifact", "output_complete", "output_sha", "output_consumed", "output_source",
    "patch_captured", "patch_disposed", "patch_disposed_by", "patch_result_lost", "patch_apply_pending",
    "superseded_by")
# The closing facts SETTLED wrote (and the fault CONTAINMENT_FAULT recorded): a later
# STARTED row never carries them, so replacing wholesale would forget a run's cost,
# its spend disclosure, the engine's model/profile/locators and an unresolved fault,
# or the owner's recorded terminal-wake acknowledgement.
STARTED_SETTLEMENT_FACTS: Tuple[str, ...] = (
    "terminal_state", "terminal_reason", "engine_model", "access_profile",
    "spend_disclosed", "spend_estimated", "cost_usd", "cost_partial",
    "settled_at", "engine_locator", "terminal_ack_attempt")
# Binding/authority facts are FIRST-WINS (R1-2): a later idempotent STARTED row
# may be minted by a context that no longer knows the original binding; the
# first recorded fact is authoritative and is never erased or retargeted.
STARTED_FIRST_WINS_FACTS: Tuple[str, ...] = (
    "snapshot_id", "execution_root", "execution_binding_fingerprint", "baseline_sha", "target_root",
    "capture_id", "snapshot_task_id", "authority_source", "resource_ref", "selected_subagent_id",
    "config_fingerprint", "work_order_fingerprint", "work_order_coverage",
    "authority_fingerprint", "work_order_source_request", "category", "source",
    "execution_mode", "effective_access", "thread_id", *STARTED_MAPPING_FIELDS,
    *REVIEW_ATTRIBUTION_KEYS)
