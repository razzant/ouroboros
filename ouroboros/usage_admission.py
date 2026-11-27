"""Read-side admission projections over the usage store.

Two readers that decide whether money may be committed, kept beside each other
because both must see the same axes the reservation itself checks
(``usage_accounting.reserve_attempt``): the review wave's whole-wave fit and the
whole-work BILLING GROUP (owner Batch4 2A).

The group is the one piece of monetary continuity across an owner's Continue:
a new root admitted by Continue keeps its own genuine task/root/parent ids,
but its attempts are stamped with the ORIGINAL root's ``billing_group_id`` and
checked against that root's ORIGINAL effective cap (the configured cap it
started under, carried, never re-issued). A row belongs to group ``G`` when it
carries ``billing_group_id == G``, or — for rows written before the field
existed — when it has no group and its root is ``G``, so legacy history joins
by the original root without being rewritten. The check is symmetric: the
original root's own later work (its reviews, its post-work) is a member of the
same group and sees every successor's spend. Nothing here is a second ledger:
the group axis is the store's ``group`` summary (``billing_group_key``), kept
in the same transaction as every attempt row. Every axis admits while its
KNOWN spend (the settled bucket: confirmed prices and disclosed estimates) is
below its limit (owner Q4-A, #1487); reservations and unresolved bounds are
disclosed exposure, never counted as spending, and an unknown price is never a
zero. The projection never promises a bound on an eventual bill: concurrent,
in-flight and late charges can exceed the cap. Such a charge is recorded and
bars the next admission; it was not prevented. A started work's cap is raised only
through its WORK CAP (``set_work_cap``), read by every admission in its own store
transaction; the original binding and the rows stay as they were written.
"""

from __future__ import annotations

from ouroboros.store import StoreBusy, bounded_wait
from ouroboros._usage_money import decimal_of, exact_money, known_room
from ouroboros._usage_rows import (
    ORDINARY_GROUP_CAP_SOURCES, ROOT_CAP_SCOPE, WORK_CAP_SCOPE, group_binds as _group_binds,
    prefer_group_tie as _prefer_group_tie, raised_cap, valid_work_cap, valid_work_cap_basis, work_cap_amount,
    work_cap_block, work_cap_change as _work_cap_change,
)
from ouroboros.usage_ledger import (
    USAGE_LOCK_TIMEOUT_SEC, BillingAuthorityBusy, BillingAuthorityUnavailable, UsageAccountingError,
    UsageLedgerCorrupt, UsageLockUnavailable,
)

import logging
import math
import pathlib
from dataclasses import replace
from typing import Any, Callable, Dict, Optional, Sequence
from ouroboros.task_store import patch_document

log = logging.getLogger(__name__)

GROUP_KEY_PREFIX = "group:"


def scope_group(scope: Any) -> tuple:
    """``(group_id, limit_usd)`` of a usage scope; a root is its own group by default."""
    group = str(getattr(scope, "billing_group_id", "") or getattr(scope, "root_task_id", "") or "")
    limit = getattr(scope, "billing_group_limit_usd", None)
    if (limit is None and not getattr(scope, "billing_group_limit_source", "")
            and group == str(getattr(scope, "root_task_id", "") or "")):
        limit = getattr(scope, "root_limit_usd", None)
    return group, (None if limit is None else max(0.0, float(limit)))


# Older records may carry an ``unavailable:<id>`` stand-in group with a zero cap. Nothing
# produces one any more and it names no bucket: an authority that cannot be established
# raises ``BillingAuthorityUnavailable`` (kind and cause), never a group or a zero.
UNAVAILABLE_GROUP_PREFIX = "unavailable:"
_GROUP_FIELDS = ("billing_group_id", "billing_group_limit_usd", "billing_group_limit_source",
                 "billing_group_limit_revision")


def authority_unavailable(kind: str, root_task_id: str, cause: str) -> BillingAuthorityUnavailable:
    """The typed non-money refusal: ``unreadable`` (an operational fault) or ``refused`` (positive policy)."""
    return BillingAuthorityUnavailable(f"billing group authority unavailable for root {root_task_id} ({kind}: {cause})",
                                       kind=kind, cause=cause, root_task_id=root_task_id)


def _group_fields(binding: Any, root_id: str) -> Dict[str, Any]:
    if (not isinstance(binding, dict) or not isinstance(binding.get("billing_group_id"), str)
            or not binding["billing_group_id"] or "billing_group_limit_usd" not in binding):
        raise authority_unavailable("unreadable", root_id, "malformed_binding")
    if binding["billing_group_id"].startswith(UNAVAILABLE_GROUP_PREFIX):
        raise authority_unavailable("refused", root_id, "stand_in_binding")
    limit = binding["billing_group_limit_usd"]
    try:
        if limit is None or not isinstance(limit, bool) and math.isfinite(float(limit)) and float(limit) >= 0:
            return {key: binding.get(key) for key in _GROUP_FIELDS}
    except (TypeError, ValueError, OverflowError):
        pass
    raise authority_unavailable("unreadable", root_id, "malformed_binding")


def _ledger_group_fields(row: Dict[str, Any], identity: str) -> Dict[str, Any]:
    return _group_fields({
        **row, "billing_group_id": row.get("billing_group_id") or identity,
        "billing_group_limit_usd": row.get("billing_group_limit_usd", row.get("root_limit_usd")),
        "billing_group_limit_source": row.get("billing_group_limit_source") or "ledger_first_row",
    }, identity)


def _published_group(row: dict) -> Optional[dict]:
    """Direct starts and schedule receipts pin a top-level group; pooled host admissions freeze stamped metadata."""
    pinned = row.get("billing_group")
    receipt, metadata = row.get("host_admission"), row.get("metadata")
    if (pinned is None and isinstance(receipt, dict) and receipt.get("status") == "accepted"
            and receipt.get("token") and isinstance(metadata, dict)):
        pinned = metadata.get("billing_group")
    return pinned


def ledger_billing_binding(budget_root: Any, root_task_id: str, *, allow_stale: bool = False) -> Dict[str, Any]:
    """The group binding the root's own earliest recorded row carries, never current settings.

    The store's ``bindings`` row (the import carried a compacted block's
    binding, or an older block's own cap literal: ``legacy_live``). ``{}`` when
    nothing recorded one: the caller then binds the root as its own group under
    the configured cap and discloses it (``legacy_default``).
    """
    from ouroboros import usage_accounting as ua
    from ouroboros import usage_store

    root = ua._drive_root(budget_root)
    # The money read probes without waiting inside a caller's sliced acquisition (``_usage_wait``).
    ctx = (usage_store.read(root, allow_stale=True) if allow_stale
           else usage_store.hold(root, timeout_sec=bounded_wait(USAGE_LOCK_TIMEOUT_SEC), write=False))
    with ctx as txn:
        row = txn.binding("root", root_task_id)
    if row is None:
        return {}
    binding = _ledger_group_fields(row, root_task_id)
    return {**binding, "billing_group_limit_usd": ua._number(binding["billing_group_limit_usd"])}


def task_billing_fields(task: Dict[str, Any], root_task_id: str, root_limit: Optional[float],
                        budget_root: Any = None, *, pin_initial: bool = False,
                        persist_initial: bool = True, allow_stale: bool = False) -> Dict[str, Any]:
    """Resolve the root's durable whole-work binding; only initial admission may pin it.

    Root caps remain independent of the original group cap. Missing descendant or
    continuation authority raises ``BillingAuthorityUnavailable``, never an independent
    wallet: ``refused`` when a clean read positively shows missing or forbidden lineage
    (a quarantined identity raises on the strict read instead: ``unreadable``). This is
    also used outside an execution context for late custody settlement. Scheduled
    occurrences derive the initial binding here but persist it atomically with
    their canonical frozen receipt (persist_initial=False).
    """
    from ouroboros.task_results import load_task_result, stamp_task_result_schema
    from ouroboros.utils import utc_now_iso

    refused = lambda cause: authority_unavailable("refused", root_task_id, cause)  # noqa: E731
    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    carried = metadata.get("continuation") or {}
    task_id = str(task.get("id") or task.get("task_id") or "")
    try:
        row = load_task_result(pathlib.Path(budget_root), root_task_id, strict=True) if budget_root else None
        # A root without a result row may still have spent (its live rows name its
        # group), so the ledger is consulted: a warm view off every other lock.
        historical = ledger_billing_binding(budget_root, root_task_id, allow_stale=allow_stale) if budget_root and not row else {}
        if not row and root_task_id != task_id and not historical:
            raise refused("root_lineage_missing")
        row = row or {}
        root_meta = row.get("metadata") or {}
        schedule_id = str(row.get("schedule_id") or root_meta.get("schedule_id") or metadata.get("schedule_id") or "")
        canonical = _published_group(row)
        # The scheduler's republish seam may repair an accepted occurrence it has not dispatched.
        # A worker, a late receipt and a money read take the occurrence that already actually ran.
        repairing = pin_initial and not persist_initial and _republishable_occurrence(row)
        if canonical is not None and not repairing:
            return {"root_limit_usd": root_limit, **_group_fields(canonical, root_task_id)}
        if schedule_id:
            if not repairing:
                historical = historical or (ledger_billing_binding(budget_root, root_task_id, allow_stale=allow_stale)
                                            if budget_root and row else {})
                if historical:
                    return {"root_limit_usd": root_limit, **historical}
            from supervisor.followup_policy import task_binding
            from supervisor.queue_schedules import load_schedule_store
            schedule = next((r for r in load_schedule_store(budget_root)["tasks"]
                             if r.get("id") == schedule_id), None)
            if schedule is None:
                # Absent both the frozen receipt and its template, never invent a related wallet.
                recorded = root_meta.get("followup_relation") if "followup_relation" in root_meta else {"kind": ""}
                if (recorded or {}).get("kind") not in {"", "independent", "unknown"}:
                    raise refused("related_followup_unbound")
            else:
                scheduled_binding = task_binding(budget_root, schedule)
                if scheduled_binding is not None:
                    return {"root_limit_usd": root_limit, **_group_fields(scheduled_binding, root_task_id)}
            # Historical templates cannot supply money or a continuation. The
            # independent root uses only its own canonical initial binding.
            own = row.get("billing_group") or {}
            if own and own.get("billing_group_id") != root_task_id:
                raise refused("foreign_initial_binding")
            metadata = {k: v for k, v in metadata.items() if k not in {"billing_group", "continuation"}}
            root_meta = {k: v for k, v in root_meta.items() if k not in {"billing_group", "continuation"}}
            carried = {}
        elif metadata.get("source") == "task_followup" or root_meta.get("source") == "task_followup":
            raise refused("followup_without_schedule")
        carried = root_meta.get("continuation") or carried
        binding = row.get("billing_group") or root_meta.get("billing_group") or metadata.get("billing_group") or carried or historical
        if not binding and row and budget_root:
            binding = ledger_billing_binding(budget_root, root_task_id, allow_stale=allow_stale)
        if pin_initial and task_id == root_task_id and budget_root:
            if not binding:
                # No pinned binding or usable ledger literal. A root that already ran
                # keeps its own group under a disclosed legacy default.
                legacy = bool(row.get("started_at")) or row.get("status") in {"completed", "cancelled", "failed"}
                binding = {"billing_group_id": root_task_id, "billing_group_limit_usd": root_limit,
                           "billing_group_limit_source": "legacy_default" if legacy else "initial_task_admission",
                           "billing_group_limit_revision": utc_now_iso()}
            if persist_initial:
                def pin(current):
                    nonlocal binding
                    if (prior := _published_group(current)) is not None:
                        binding = prior
                        return None
                    return stamp_task_result_schema({"task_id": root_task_id, "status": "requested", **current,
                                                     "billing_group": _group_fields(binding, root_task_id)})

                patch_document(pathlib.Path(budget_root), root_task_id, pin)
        if binding:
            return {"root_limit_usd": root_limit, **_group_fields(binding, root_task_id)}
        return {"root_limit_usd": root_limit, "billing_group_id": root_task_id,
                "billing_group_limit_usd": root_limit}
    except (StoreBusy, BillingAuthorityUnavailable):
        raise
    except Exception as exc:
        if isinstance(exc, UsageLockUnavailable) and exc.reason == "contention":
            # Positive money contention: still unreadable to every reader; a preparation's own wait retries it.
            raise BillingAuthorityBusy(root_task_id, rollback_confirmed=getattr(exc, "rollback_confirmed", False)) from exc
        log.warning("Billing authority unavailable for %s", root_task_id, exc_info=True)
        raise authority_unavailable("unreadable", root_task_id, f"read_failed:{type(exc).__name__}") from exc


def settlement_billing_fields(root: Any, task_id: str, root_task_id: str) -> Dict[str, Any]:
    """Late receipts keep the bucket their admission bound, even without an execution scope.

    Durable lineage, else the tree's delegated START custody. When neither can say,
    ``BillingAuthorityUnavailable`` leaves the receipt in its existing custody for a
    later pass: no row is written under a stand-in, missing or caller's own group.
    """
    if not root_task_id:  # no task identity at all: an unattributed receipt (global axis), no group to guess
        return {}
    failure = None
    try:
        binding = task_billing_fields({"id": task_id}, root_task_id, None, root)
    except BillingAuthorityUnavailable as exc:
        binding, failure = {}, exc
    if binding.get("billing_group_limit_source"):
        return binding
    from ouroboros.delegate_custody_current import tree_billing_group
    from ouroboros.store import StoreError

    try:
        carried = tree_billing_group(pathlib.Path(root), root_task_id)
    except StoreBusy:
        raise
    except StoreError as exc:
        raise authority_unavailable("unreadable", root_task_id, "start_custody_unreadable") from exc
    if carried:
        return carried
    if failure is not None:
        raise failure
    return binding


def _republishable_occurrence(row: dict) -> bool:
    return row.get("status") == "scheduled" and not row.get("started_at") and _schedule_dispatch(row) == "none"


def _schedule_dispatch(row: dict) -> Any:
    receipt = row.get("schedule_admission")
    return receipt.get("dispatch") if isinstance(receipt, dict) else None


def work_cap(view: Any, group_id: str) -> Optional[Dict[str, Any]]:
    """The cap raised for work ``group_id`` (``set_work_cap``), read in the caller's own store transaction; None when
    none was raised. A damaged record is an unreadable authority, never a wider cap."""
    record = view.binding(WORK_CAP_SCOPE, str(group_id)) if group_id else None
    if isinstance(record, dict) and "unverified_export" in record:
        retained = record["unverified_export"]
        if valid_work_cap(retained) and retained.get("billing_group_id") == str(group_id):
            bound = _ledger_group_fields(view.binding("group", str(group_id)) or {}, str(group_id))
            original = work_cap_amount(bound["billing_group_limit_usd"])
            if (bound["billing_group_id"] == str(group_id) and original is not None
                    and decimal_of(original) <= decimal_of(work_cap_amount(retained["original_limit_usd"]))):
                return {**record, "verified_original": bound}
            raise authority_unavailable("unreadable", str(group_id), "work_cap_original_unavailable")
    if record is not None and (not valid_work_cap(record) or record.get("billing_group_id") != str(group_id)):
        raise authority_unavailable("unreadable", str(group_id), "malformed_work_cap")
    return record


def _work_cap_of(view: Any, group: str) -> Optional[Dict[str, Any]]:
    return work_cap(view, group) if group and not group.startswith(UNAVAILABLE_GROUP_PREFIX) else None


def _attempt_root_cap(cap, limit, source, request_limit) -> Optional[float]:
    """Intersect the amended root cap and the attempt's independently supplied maximum."""
    limit = raised_cap(cap, limit, source, group=False)[0]
    if request_limit is None:
        return limit
    request_limit = max(0.0, float(request_limit))
    return request_limit if limit is None else min(limit, request_limit)


def _observed_root_limit(limit, source, revision, view, root, initial):
    """Keep a recorded independent root fence from addressed bindings only; never widen un-upgraded or ambiguous roots."""
    from ouroboros import usage_accounting as ua
    from ouroboros._usage_rows import root_cap_error
    from ouroboros.usage_origin import root_caps_backfill_pending

    if source:
        return limit, source, revision
    recorded = initial
    if recorded.get("root_limit_usd") is None and not recorded.get("root_limit_source"):
        try:
            pending = root_caps_backfill_pending(view.meta("import"))
        except UsageLedgerCorrupt as exc:
            raise authority_unavailable("unreadable", root, "root_cap_meta_invalid") from exc
        if pending:
            if view.bucket("root", root).min_cap() is not None:
                raise authority_unavailable("unreadable", root, "root_cap_backfill_pending")
        elif (materialized := view.binding(ROOT_CAP_SCOPE, root)) is not None:
            if (cause := root_cap_error(materialized)) is not None:
                raise authority_unavailable("unreadable", root, cause)
            recorded = materialized
    if (original := ua._number(recorded.get("root_limit_usd"))) is not None:
        stored_source = recorded.get("root_limit_source")
        if stored_source != "attempt_request" and (
            limit is None or (stored_source and decimal_of(original) <= decimal_of(limit))
        ):
            return original, stored_source, recorded.get("root_limit_revision")
    elif limit is None and recorded.get("root_limit_source") not in (None, "", "attempt_request"):
        return None, recorded.get("root_limit_source"), recorded.get("root_limit_revision")
    return limit, source, revision


def raise_work_refusal(view: Any, scope: Any, *, dispatch: bool = False, stash: bool = True,
                       request_root_limit: Optional[float] = None) -> tuple:
    """The root and whole-work axes of ONE reservation, dispatch or live check, inside the caller's store
    transaction; returns the ``(root, group)`` caps it applied (``raised_cap``). A reservation first refreshes the
    root's telemetry with the pre-append subtree sum, refused or not; a dispatch reads both axes again."""
    from ouroboros import usage_accounting as ua

    group, original_group_cap = scope_group(scope)
    root_id, cap = str(getattr(scope, "root_task_id", "") or ""), _work_cap_of(view, group)
    root_source, group_source = (getattr(scope, key, "") for key in ("root_limit_source", "billing_group_limit_source"))
    # scope_group mirrors a lone recorded root fence into the group axis when the root is its own
    # group; without a work cap that group limit is the same recorded fact, never an independent
    # group remedy, so on a tie it never carries the group message past the root fence it mirrors.
    mirrored_group_cap = group == root_id and cap is None and original_group_cap is not None and not group_source
    root_limit = _attempt_root_cap(cap, getattr(scope, "root_limit_usd", None), root_source, request_root_limit)
    group_limit = raised_cap(cap, original_group_cap, group_source, group=True)[0]
    if root_id and stash and not dispatch:
        ua._stash_root_accounting(root_id, view.summary(root_id), root_limit)
    if group.startswith(UNAVAILABLE_GROUP_PREFIX):  # an older record's stand-in is no bucket, never a zero cap
        raise authority_unavailable("refused", root_id or group, "stand_in_binding")
    root_spent = root_limit is not None and (root_id or dispatch) and view.exceeds_limit(root_limit, root_task_id=root_id)
    group_spent = group and group_limit is not None and view.exceeds_limit(group_limit, billing_group_id=group)
    if group_spent and (not root_spent or (not mirrored_group_cap and decimal_of(group_limit) <= decimal_of(root_limit)
            and _prefer_group_tie(root_limit, root_source, original_group_cap, group_source)
            and (request_root_limit is None or decimal_of(request_root_limit) > decimal_of(root_limit)))):
        raise ua.BudgetExceeded(
            ("whole-work budget changed before dispatch; " if dispatch else "")
            + f"whole-work budget exhausted for group {group}: "
            f"{ua._known_spend_text(view.summary(billing_group_id=group))}, limit=${group_limit:.6f}",
            limit_scope="root", root_task_id=root_id or group)
    if root_spent:
        raise ua.BudgetExceeded("root model budget changed before dispatch" if dispatch else (
            f"root model budget exhausted for {root_id}: {ua._known_spend_text(view.summary(root_id))}, "
            f"limit=${root_limit:.6f}"), limit_scope="root", root_task_id=root_id)
    return root_limit, group_limit


def stash_row_accounting(view: Any, row: Dict[str, Any]) -> None:
    """Refresh the process-local root and group snapshots under the row's caps. On a damaged work cap use the
    recorded inputs instead of raising after a write."""
    from ouroboros import usage_accounting as ua
    from ouroboros._usage_money import billing_group_key

    group, root_id = billing_group_key(row), str(row.get("root_task_id") or "")
    cap = view.binding(WORK_CAP_SCOPE, group) if group and not group.startswith(UNAVAILABLE_GROUP_PREFIX) else None
    cap = cap if valid_work_cap(cap) else None
    if group:
        ua._stash_root_accounting(f"{GROUP_KEY_PREFIX}{group}", view.summary(billing_group_id=group), raised_cap(
            cap, ua._number(row.get("billing_group_limit_usd", row.get("root_limit_usd"))),
            row.get("billing_group_limit_source"), group=True)[0])
    if root_id:
        ua._stash_root_accounting(root_id, view.summary(root_id), _attempt_root_cap(
            cap, ua._number(row.get("root_limit_usd")), row.get("root_limit_source"), row.get("request_root_limit_usd")))


def accounting_key(scope: Any) -> str:
    """The tree-accounting cache key a money reader of this scope uses."""
    group, limit = scope_group(scope)
    root = str(getattr(scope, "root_task_id", "") or "")
    return f"{GROUP_KEY_PREFIX}{group}" if group and group != root and limit is not None else root


def original_group_limit(drive_root: Any, group_id: str) -> Dict[str, Any]:
    """The cap group ``group_id`` started under, from its earliest recorded binding.

    ``{"limit_usd": float|None, "source": str}``; ``source`` is
    ``ledger_first_row``, ``legacy_live`` (an older block's own cap literal) or
    ``no_attempt_recorded`` (nothing recorded one: the caller then decides, and
    discloses, what applies — the configured cap, ``legacy_default``).
    """
    from ouroboros import usage_accounting as ua
    from ouroboros import usage_store
    from ouroboros._usage_rows import LEGACY_LIVE_SOURCE

    root = ua._drive_root(drive_root)
    with usage_store.read(root) as txn:
        row = txn.binding("group", group_id)
    if row is None:
        return {"limit_usd": None, "source": "no_attempt_recorded"}
    carried = row.get("billing_group_limit_usd", row.get("root_limit_usd"))
    source = LEGACY_LIVE_SOURCE if row.get("billing_group_limit_source") == LEGACY_LIVE_SOURCE else "ledger_first_row"
    return {"limit_usd": None if carried is None else ua._number(carried), "source": source}


def _reconcile_pending(txn: Any, root_task_id: str, group: str, raised: Any = None) -> Optional[str]:
    from ouroboros import usage_store

    try:
        pending = usage_store.work_caps_reconcile_pending(txn.meta("import"))
    except UsageLedgerCorrupt as exc:
        raise authority_unavailable("unreadable", root_task_id, "work_cap_export_invalid") from exc
    return pending if pending and raised is None and (not group or txn.binding(WORK_CAP_SCOPE, group) is None) else None


def _work_cap_target(txn, recorded: dict, root_task_id: str) -> dict:
    """The same finite, ordinary group is eligible before the check and under the write transaction."""
    pinned = _published_group(recorded)
    pinned = _group_fields(pinned, root_task_id) if pinned is not None else None
    ledger = txn.binding("root", root_task_id)
    indexed = _ledger_group_fields(ledger, root_task_id) if ledger is not None else None
    # An undispatched occurrence can still be republished under another relation.
    scheduled = (recorded.get("schedule_id") or "schedule_admission" in recorded
                 or (recorded.get("metadata") or {}).get("schedule_id"))
    if (_republishable_occurrence(recorded) or (scheduled and indexed is None
            and not recorded.get("started_at") and _schedule_dispatch(recorded) != "possible")):
        return {"error": "work_cap_unbound", "billing_group_id": root_task_id}
    if pinned and indexed and pinned["billing_group_id"] != indexed["billing_group_id"]:
        raise authority_unavailable("unreadable", root_task_id, "billing_group_mismatch")
    target = indexed or pinned
    if not target:
        return {"error": "work_cap_unbound", "billing_group_id": root_task_id}
    group = target["billing_group_id"]
    ledger_group = txn.binding("group", group)
    if ledger_group is not None:
        bound = _ledger_group_fields(ledger_group, group)
        if bound["billing_group_id"] != group:
            raise authority_unavailable("unreadable", root_task_id, "billing_group_mismatch")
    elif pinned and pinned.get("billing_group_limit_source") and pinned.get("billing_group_limit_revision"):
        bound = pinned  # exact admission before the first usage row
    else:
        return {"error": "work_cap_unbound", "billing_group_id": group}
    original = bound["billing_group_limit_usd"]
    if original is None:
        return {"error": "work_cap_unlimited", "billing_group_id": group}
    if (original := work_cap_amount(original)) is None:
        return {"error": "work_cap_unbound", "billing_group_id": group}
    if bound["billing_group_limit_source"] not in ORDINARY_GROUP_CAP_SOURCES:
        return {"error": "independent_cap_binds", "billing_group_id": group}
    if pending := _reconcile_pending(txn, root_task_id, group):
        return {"error": "work_cap_reconcile_pending", "billing_group_id": group, "reason": pending}
    return {**bound, "billing_group_limit_usd": original}


def work_cap_target(drive_root: Any, recorded: dict, root_task_id: str) -> dict:
    """Bind a checked cost-cap call without pinning an admission or consulting a schedule template."""
    from ouroboros import usage_accounting as ua
    from ouroboros import usage_store

    with usage_store.hold(ua._drive_root(drive_root), timeout_sec=bounded_wait(USAGE_LOCK_TIMEOUT_SEC), write=False) as txn:
        return _work_cap_target(txn, recorded, root_task_id)


def set_work_cap(drive_root: Any, task: Dict[str, Any], root_task_id: str, *, limit_usd: Any,
                 by: Dict[str, Any], basis: Dict[str, Any], expected_group: Optional[str] = None) -> Dict[str, Any]:
    """Raise the whole cap of ONE started work: its billing group's work cap, in one store transaction.

    The owner's cited amount supersedes older decisions. Ouroboros's own attributed Cyber Pro judgment can raise
    an ordinary work cap, not an explicit owner maximum. Choosing a number entrusted to Ouroboros is its judgment,
    not a number fixed by the owner. Nothing resets spending or resumes work."""
    from ouroboros import usage_accounting as ua
    from ouroboros import usage_store
    from ouroboros.store import StoreError
    from ouroboros.task_results import load_task_result
    from ouroboros.utils import utc_now_iso

    limit = work_cap_amount(limit_usd)
    if limit is None:
        return {"ok": False, "error": "invalid_limit"}
    if not valid_work_cap_basis(basis, limit):
        return {"ok": False, "error": "cost_cap_basis_unverified"}
    store_root = ua._drive_root(drive_root)
    try:
        # Read the task store off the money lock. This field is pinned by admission, never a live schedule lookup.
        recorded = load_task_result(store_root, root_task_id, strict=True) or {}
        with usage_store.hold(store_root, timeout_sec=bounded_wait(USAGE_LOCK_TIMEOUT_SEC)) as txn:
            bound = _work_cap_target(txn, recorded, root_task_id)
            if bound.get("error"):
                return {"ok": False, **bound}
            group = bound["billing_group_id"]
            if expected_group is not None and group != expected_group:
                return {"ok": False, "error": "cost_cap_target_changed"}
            current = work_cap(txn, group)
            if current and (fallback := current.get("verified_original")):
                return {"ok": False, "error": "work_cap_export_unverified", "billing_group_id": group,
                        "basis": fallback["billing_group_limit_source"], "limit_usd": fallback["billing_group_limit_usd"],
                        "revision": fallback["billing_group_limit_revision"]}
            original = bound["billing_group_limit_usd"]
            original_source = bound["billing_group_limit_source"]
            if current and work_cap_amount(current["original_limit_usd"]) != original:
                raise authority_unavailable("unreadable", root_task_id, "work_cap_original_limit_mismatch")
            before = work_cap_amount(current["limit_usd"]) if current else original
            error, changed, owner = _work_cap_change(current, before, limit, basis)
            if error:
                return {"ok": False, "error": error, "billing_group_id": group, "limit_usd": before}
            if changed:
                current = {"billing_group_id": group, "task_id": root_task_id, "limit_usd": limit,
                           "original_limit_usd": original, "previous_limit_usd": before,
                           "revision": utc_now_iso(), "by": dict(by),
                           "basis": {"kind": "own_judgement"} if basis.get("choose_limit") else dict(basis),
                           **({"owner_decision": owner} if owner else {})}
                txn.set_work_cap(group, current)
            spending = txn.bucket("group", group).render_summary()
            known = ua._number(spending["settled_usd"])
    except BillingAuthorityUnavailable as exc:
        return {"ok": False, "error": "billing_authority_unavailable", "billing_authority": exc.kind,
                "cause": getattr(exc, "cause", "")}
    except (StoreError, UsageAccountingError, OSError, ValueError) as exc:
        return {"ok": False, "error": "cost_cap_unwritten", "detail": type(exc).__name__}
    return {"ok": True, "task_id": root_task_id, "billing_group_id": group, "limit_usd": limit,
            "previous_limit_usd": before, "basis": current["basis"]["kind"] if current else original_source or "original_binding",
            "revision": current["revision"] if current else bound.get("billing_group_limit_revision"), "known_usd": known,
            "unknown_unmetered": spending["unknown_unmetered"],
            "remaining_known_usd": None if known is None else max(0.0, float(known_room(limit, known))),
            **(({"confirmed": True} if changed else {"unchanged": True}) if before == limit else {}),
            "note": "The work's whole cap, not money added: its spend stays as recorded, and a paused work stays "
                    "paused until its Resume."}


def group_money_spent(drive_root: Any, binding: Dict[str, Any]) -> bool:
    """Continue's decision: compare the amended group ceiling and known spend from ONE read."""
    from ouroboros import usage_accounting as ua
    from ouroboros import usage_store

    binding = _group_fields(binding, str(binding.get("billing_group_id") or ""))
    group = binding["billing_group_id"]
    with usage_store.read(ua._drive_root(drive_root)) as view:
        limit, _ = raised_cap(work_cap(view, group), binding.get("billing_group_limit_usd"),
                              binding.get("billing_group_limit_source"), group=True)
        return limit is not None and view.exceeds_limit(limit, billing_group_id=group)


def _per_slot(value: Any, count: int) -> list:
    """Broadcast one scalar, or align one per-slot sequence, over ``count`` slots."""
    if isinstance(value, (list, tuple)):
        values = list(value)
        return values[:count] + [values[-1] if values else 0] * max(0, count - len(values))
    return [value] * count


def projected_pricing_request(model, preference, resolve_target_for_model, *, review=False, use_local=None, **sizes):
    """Price the adapter's projected carrier on its resolved connection, before a physical request exists.

    ``use_local`` is the seat's own locality: True prices the local lane's own request (``llm_local``), False the
    remote route; None (no seat fact) leaves a review route the resolver reads as local unpriced. The projected
    carrier bounds the adapter's routing without a transcript; a review projection reads only cached tariffs."""
    from ouroboros.llm_attempt import ConnectionUnavailable
    from ouroboros.llm_routing import (
        _resolver_route, apply_processing_preference, attempt_pricing_fields, openrouter_chat_routing,
        processing_transport_modes, submitted_processing_mode,
    )
    from ouroboros.provider_models import provider_for_model, review_model_uses_local
    from ouroboros.usage_accounting import AttemptRequest

    if use_local:
        return AttemptRequest(model="local-model", provider="local", processing_preference=preference,
                              allow_live_fetch=not review, **sizes)
    unbound_local = review and use_local is None and review_model_uses_local(model)
    provider = provider_for_model(model) if unbound_local else _resolver_route(model).kind
    request = AttemptRequest(model=model, provider=provider, processing_preference=preference,
                             allow_live_fetch=not review, **sizes)
    try:
        target = resolve_target_for_model(model) if callable(resolve_target_for_model) and not unbound_local else None
    except ConnectionUnavailable:
        target = None
    if not target or target.get("provider") != provider or not target.get("base_url") or not target.get("usage_model"):
        return replace(request, force_unknown_reservation=True)
    tier = preference.strip().lower()
    target = {**target, "processing_preference": tier}
    payload = {"model": target.get("resolved_model")}
    if provider == "openrouter":
        routing = openrouter_chat_routing(target, messages=None)
        if routing:
            payload["extra_body"] = {"provider": routing}
    apply_processing_preference(target, payload)
    submitted = submitted_processing_mode(target, payload)
    if review and tier and not submitted:
        # An ordinary-only adapter sends Standard by omitting the field.
        if tier != "standard" or processing_transport_modes(target) != ("standard",):
            return replace(request, model=target["usage_model"], force_unknown_reservation=True)
    return replace(request, model=target["usage_model"], submitted_processing_mode=submitted,
                   **attempt_pricing_fields(target, payload))


def review_seat_envelope(slot_id: str, model: str, *, use_local: Optional[bool],
                         processing_preference: Optional[str]) -> Dict[str, Any]:
    """The locality and tier ONE api review seat's send carries, for its wave admission.

    The same reads its send makes: an unbound locality follows ``review_model_uses_local``
    (``review_multi_model._query_model``) and the tier resolves its ``reviewer:<slot>`` role as every send does
    (``model_wait.model_waitable``), an explicit empty value kept. A tier the send would refuse stays as written:
    the projection then prices it unknown, never as another tier."""
    from ouroboros.model_slots import resolve_processing_preference
    from ouroboros.provider_models import review_model_uses_local

    try:
        tier = resolve_processing_preference(f"reviewer:{slot_id}", override=processing_preference)
    except ValueError:
        tier = str(processing_preference or "")
    return {"use_local": review_model_uses_local(model) if use_local is None else bool(use_local),
            "processing_preference": tier}


def review_slot_wave_fields(slots: Sequence) -> dict:
    """Aligned API send choices and the separate subscription count from one review panel."""
    from ouroboros.review_execution import ReviewRouteKind

    api = [slot for slot in slots if slot.route != ReviewRouteKind.AGENT_SESSION]
    sends = [review_seat_envelope(slot.slot_id, slot.model, use_local=slot.use_local,
                                  processing_preference=slot.processing_preference) for slot in api]
    return {"models": [slot.model for slot in api], "slot_ids": [slot.slot_id for slot in api],
            "use_local": [send["use_local"] for send in sends],
            "processing_preferences": [send["processing_preference"] for send in sends],
            "session_slots": len(slots) - len(api)}


def _review_wave_money(drive_root, scope, *, root_limit, root_source, global_limit, needs_paid, unpriced_seats):
    """One bounded money observation of every wave fence; pricing and lineage lookup run before it."""
    from ouroboros import usage_accounting as ua
    from ouroboros import usage_store
    from ouroboros.settings_setup_contract import resolve_total_budget_usd

    def observed(cash, limit, source, prefix=""):
        known = cash["settled_usd"]
        return ({f"{prefix}limit_usd": limit, f"{prefix}known_usd": known,
                 **{f"{prefix}{key}": cash[key]
                    for key in ("accounted_usd", "reserved_usd", "unresolved_upper_bound_usd")},
                 **({"binding_source": source} if not prefix else {})},
                None if limit is None else known_room(limit, known))

    root = scope.root_task_id
    group, group_limit = scope_group(scope)
    group_source = scope.billing_group_limit_source
    if global_limit is None:
        global_limit = resolve_total_budget_usd()
    global_limit = None if global_limit is None else max(0.0, float(global_limit))
    with usage_store.read(ua._drive_root(drive_root)) as view:
        initial = view.binding("root", root) or {}
        # An addressed wave outside its original worker can still have an indexed group binding.
        if not scope.billing_group_id and initial and (
            initial.get("billing_group_id") not in {None, "", root}
            or initial.get("billing_group_limit_source") or root_limit is None
        ):
            carried = _ledger_group_fields(initial, root)
            group, group_limit, group_source, _ = (carried[key] for key in _GROUP_FIELDS)
        cap = work_cap(view, group)
        pending = _reconcile_pending(view, root, group, cap)
        root_limit, root_source, _ = _observed_root_limit(root_limit, root_source, None, view, root, initial)
        prefer_group = _prefer_group_tie(root_limit, root_source, group_limit, group_source)
        original_group_limit = (cap or {}).get("original_limit_usd", group_limit)
        original = work_cap_amount(original_group_limit)
        ordinary_group = original is not None and str(group_source or "") in ORDINARY_GROUP_CAP_SOURCES
        group_can_move = group_limit is not None and ordinary_group and decimal_of(group_limit) >= decimal_of(original)
        root_can_move = group_can_move and _prefer_group_tie(root_limit, root_source, original, group_source)
        root_limit, root_lifted = raised_cap(cap, root_limit, root_source, group=False)
        group_limit, group_lifted = raised_cap(cap, group_limit, group_source, group=True)
        fallback = (cap or {}).get("verified_original")
        if fallback and group_limit is not None and decimal_of(group_limit) == decimal_of(fallback["billing_group_limit_usd"]):
            group_source = fallback["billing_group_limit_source"]
        caps = {"root": (root_limit, WORK_CAP_SCOPE if root_lifted else root_source),
                "group": (group_limit, WORK_CAP_SCOPE if group_lifted else group_source)}
        choices = {"root": observed(view.summary(root), *caps["root"])}
        if group_limit is not None:
            choices["group"] = observed(view.summary(billing_group_id=group), *caps["group"])
        rooms = {axis: choice[1] for axis, choice in choices.items()}
        work_axis = "group" if _group_binds(rooms["root"], rooms.get("group"), prefer_tie=prefer_group) else "root"
        pending_group = bool(pending and ordinary_group)
        block = work_cap_block(caps, rooms, work_axis, original_group_limit,
                               unverified=bool(fallback), reconcile_pending=pending_group)
        if block:
            work_axis = block[1]
        detail, work_room = choices[work_axis]
        work_source = detail.get("binding_source")
        work_reached = (root_limit is not None and view.exceeds_limit(root_limit, root_task_id=root)
                        or group_limit is not None and view.exceeds_limit(group_limit, billing_group_id=group))
        global_detail, global_room = observed(view.summary(), global_limit, None, "global_")

        # Whether THIS wave can draw the global wallet — evaluated once and only when the wallet is reached.
        paid_wave: Optional[bool] = None

        def wave_needs_paid_admission() -> bool:
            nonlocal paid_wave
            if paid_wave is None:
                paid_wave = bool(needs_paid or any(ua._paid_at_limit(view, provider, None, model)
                                                    for provider, model in unpriced_seats))
            return paid_wave

        global_reached = global_limit is not None and view.exceeds_limit(global_limit)
        global_binds = bool(global_reached and wave_needs_paid_admission())
        fits = not work_reached and not global_binds
        binding_axes = []
        if not fits:
            if global_binds and work_reached:
                binding_axes = ["global", work_axis] if global_room < work_room else [work_axis, "global"]
            else:
                binding_axes = ["global"] if global_binds else [work_axis]
            binding_axis = binding_axes[0]
            room = global_room if binding_axis == "global" else work_room
        elif (global_room is not None and (work_room is None or global_room < work_room)
              and (not global_reached or work_room is None)):
            binding_axis, room = "global", global_room
        else:
            binding_axis, room = (work_axis if work_room is not None else None), work_room
        if binding_axis == "global":
            detail = {**detail, "binding_source": "settings_budget_resolver"}
        work_moves = group_can_move if work_axis == "group" else root_can_move
        work_state = (None if block and block[0] == "independent_cap_exhausted"
                      else "unverified_export" if fallback
                      else "reconcile_pending" if pending_group and work_moves else None)
        work_in_scope = work_axis in binding_axes or (fits and binding_axis in {"root", "group"})
    return {**detail, **global_detail, "global_remaining_usd": None if global_room is None else max(0., float(global_room)),
            "remaining_usd": None if room is None else max(0., float(room)),
            "binding_axis": binding_axis, "binding_axes": binding_axes, "billing_group_id": group, "fits": fits,
            "binding_can_use_work_cap": bool(work_in_scope and work_moves and not (fallback or pending_group or block)),
            **({"work_binding_axis": work_axis, "work_binding_source": work_source} if work_room is not None else {}),
            **({"amendment_state": work_state} if work_state else {})}


def review_wave_admission(
    drive_root: pathlib.Path | str | None = None,
    *,
    root_task_id: str,
    models: Sequence[str],
    prompt_chars: int | Sequence[int],
    max_completion_tokens: int | Sequence[int] = 65536,
    remaining_usd_override: float | None = None,
    task_id: str = "",
    root_limit_usd: float | None = None,
    root_limit_source: str = "",
    global_limit_usd: float | None = None,
    categories: str | Sequence[str] = "",
    slot_ids: str | Sequence[str] = "",
    processing_preferences: str | Sequence[str] = "",
    allow_live_fetch: bool = False,
    resolve_target_for_model: Optional[Callable[[str], Optional[dict]]] = None,
    use_local: Optional[bool] | Sequence[Optional[bool]] = None,
    session_slots: int = 0,
) -> Dict[str, Any]:
    """Read-only wave admission on the reservation's own known-spend rule.

    The wave is admitted while KNOWN spend is below every applicable limit
    (global, root, whole-work group), exactly what each seat's reservation will
    check (#1487). The summed seat bounds are information, never an earlier
    refusal: a wave that crosses the limit mid-dispatch is refused per seat by
    the reservation and dispatch fences, with truthful custody of what was sent.
    Standalone callers may supply remaining_usd_override (known room). An
    explicit root_limit_usd is the caller's current fence, otherwise the
    ledger's historical minimum governs. Global None resolves settings; a
    non-positive configured limit is unbounded. Open holds are disclosed beside
    the known spend; unknown prices stay unknown.

    Input sizes, outputs, categories, slots, processing and locality (``use_local``:
    None leaves the legacy model-only read) can be scalar or aligned per-slot
    values. Price each seat under its own sending scope, so the
    caller's warm cache split cannot stand in for a reviewer's cold prefix.
    Returned per-slot bounds and both remainders disclose the binding axis.
    ``session_slots`` counts additional session seats separately from the API price vectors. A session-only wave
    has no API estimate but still checks the root and original-group limits; an empty wave checks nothing.
    Review API seats use the default LLMClient's resolver. A caller using a different client
    supplies its bound resolver. Preflight reads only cached tariffs; the older
    ``allow_live_fetch`` argument remains accepted for existing callers.
    """
    from ouroboros import usage_accounting as ua

    result: Dict[str, Any] = {
        "fits": True,
        "estimated_wave_usd": None,
        "remaining_usd": None,
        "limit_usd": None,
        "slots": len(list(models or [])),
        **({"session_slots": session_slots} if session_slots else {}),
        "unpriced_slots": 0,
        "known_usd": None,
        "accounted_usd": None,
        "reserved_usd": None,
        "unresolved_upper_bound_usd": None,
        "slot_bounds": [],
        **{key: None for key in ("global_limit_usd", "global_known_usd", "global_accounted_usd", "global_remaining_usd",
                                 "global_reserved_usd", "global_unresolved_upper_bound_usd", "binding_axis",
                                 "binding_source", "billing_group_id")},
    }
    root_task_id = str(root_task_id or "").strip()
    if not root_task_id or not (models or session_slots):
        return result
    try:
        from ouroboros.llm_routing import resolve_target

        if remaining_usd_override is None:
            _, bound_scope = ua._merge_scope(ua.AttemptRequest(
                model="", provider="", drive_root=drive_root,
                task_id=task_id, root_task_id=root_task_id,
            ))
        chars = _per_slot(prompt_chars, len(models))
        outputs = _per_slot(max_completion_tokens, len(models))
        seat_categories = _per_slot(categories, len(models))
        seat_slot_ids = _per_slot(slot_ids, len(models))
        seat_processing = _per_slot(processing_preferences, len(models))
        seat_local = _per_slot(use_local, len(models))
        base_scope = ua.current_usage_scope() or ua.UsageScope()
        total = 0.0

        # Disclosed values only (the decision reads the unrounded bound): six decimals as before, except that a
        # positive estimate under 1e-6 stays exact, so the formatter can show it as `<$0.000001` instead of 0.
        def disclosed_usd(amount: float) -> float:
            return amount if 0.0 < amount < 1e-6 else round(amount, 6)
        needs_paid = False
        slot_bounds, unpriced_seats = [], []
        for index, model in enumerate(models):
            seat_scope = base_scope
            if str(seat_categories[index] or ""):
                seat_scope = replace(
                    base_scope, category=str(seat_categories[index]),
                    review_slot_id=str(seat_slot_ids[index] or ""),
                )
            with ua.usage_scope(seat_scope):
                request = projected_pricing_request(
                    str(model or ""), str(seat_processing[index] or ""), resolve_target_for_model or resolve_target,
                    review=True, use_local=seat_local[index],
                    prompt_tokens_estimate=max(0, int(chars[index] or 0)) // 4,
                    max_completion_tokens=max(0, int(outputs[index] or 0)), task_id=str(task_id or ""),
                )
                bound = ua._reservation_cost(request)
            needs_paid = needs_paid or ua.paid_admission(request.provider, bound)
            slot_bounds.append(None if bound is None else disclosed_usd(float(bound)))
            if bound is None:
                # Unknown contributes no invented price and remains explicitly counted.
                unpriced_seats.append((request.provider, request.model))
                continue
            total += float(bound)
        if remaining_usd_override is not None:
            remaining = float(remaining_usd_override)
            result.update(remaining_usd=remaining, fits=remaining > 1e-9)
        else:
            result.update(_review_wave_money(drive_root, bound_scope, root_limit=root_limit_usd,
                                             root_source=root_limit_source, global_limit=global_limit_usd,
                                             needs_paid=needs_paid, unpriced_seats=unpriced_seats))
            if result["binding_axis"] is None:
                return result
        result.update(slot_bounds=slot_bounds, unpriced_slots=len(unpriced_seats))
        if len(unpriced_seats) < len(models):
            result["estimated_wave_usd"] = disclosed_usd(total)
        return result
    except BillingAuthorityUnavailable as exc:
        from ouroboros._usage_rows import ROOT_CAP_AUTHORITY_CAUSES
        # Not money: no seat could reserve. The kind says whether rereading can help.
        axis = "root" if exc.cause in ROOT_CAP_AUTHORITY_CAUSES else "group"
        return {**result, "fits": False, "binding_axis": axis, "reason": "billing_authority_unavailable",
                "billing_authority": exc.kind, **({"cause": exc.cause} if exc.cause else {})}
    except Exception as exc:
        # Advisory: every seat's own reservation still decides. A failed read is
        # disclosed as unknown, never shown as room and never a new refusal.
        log.debug("review_wave_admission failed open", exc_info=True)
        return {**result, "fits": None, "admission_unknown": type(exc).__name__}


def task_accounting_key(result_root: Any, task: Dict[str, Any], root_task_id: str) -> str:
    """The strict money reader's key for a queued task: its group, or its own root.

    An unestablished authority yields no key: ``refresh_root_accounting`` then reads
    nothing it could call room.
    """
    try:
        fields = task_billing_fields(task, root_task_id, None, result_root)
    except BillingAuthorityUnavailable:
        return ""  # no key: the grant refuses ``root_accounting_unavailable``
    group = str(fields["billing_group_id"])
    return f"{GROUP_KEY_PREFIX}{group}" if group and group != root_task_id else root_task_id


def current_usage_projection(
    drive_root: pathlib.Path | str | None = None,
    *,
    root_task_id: str = "",
    global_limit_usd: Optional[float] = None,
    include_roots: bool = False, allow_stale: bool = False, billing_group_id: str = "",
) -> Dict[str, Any]:
    """The global projection, or one root's or one whole-work group's, from the
    store's summary rows. ``include_roots`` adds the per-root map (every root
    summary: an explicit request, never the default). ``allow_stale``: DISPLAY
    readers only (the short wait, then unavailable), never money."""
    from ouroboros import usage_accounting as ua
    from ouroboros import usage_store
    from ouroboros._usage_rows import projection_view

    root = ua._drive_root(drive_root)
    if billing_group_id or root_task_id:
        with usage_store.read(root, allow_stale=allow_stale) as txn:
            projection = projection_view(txn, root_task_id=root_task_id, billing_group_id=billing_group_id,
                                         degraded=usage_store.integrity_degraded(root))
        return projection
    if global_limit_usd is None:
        from ouroboros.settings_setup_contract import resolve_total_budget_usd
        global_limit_usd = resolve_total_budget_usd()  # None: no limit; an unreadable run cap is 0.0
    limit = None if global_limit_usd is None else max(0.0, float(global_limit_usd))
    with usage_store.read(root, allow_stale=allow_stale) as txn:
        return projection_view(txn, limit=limit, include_roots=include_roots,
                               degraded=usage_store.integrity_degraded(root))


def _task_money_snapshot_in_txn(txn: Any, fields: Dict[str, Any], root_id: str) -> Dict[str, Any]:
    """Project both independent ceilings from ``fields`` inside an open ``usage_store.read``; an axis its work's cap
    raised (``raised_cap``) names that cap as its source and revision."""
    group = str(fields.get("billing_group_id") or root_id)
    own, shared, initial = txn.summary(root_id), txn.summary(billing_group_id=group), txn.binding("root", root_id) or {}
    cap, root_source, root_revision = (fields.get(key) for key in ("root_limit_usd", "root_limit_source", "root_limit_revision"))
    cap, root_source, root_revision = _observed_root_limit(cap, root_source, root_revision, txn, root_id, initial)
    raised = work_cap(txn, group)
    pending = _reconcile_pending(txn, root_id, group, raised)
    sources = {"root": (root_source, root_revision),
               "group": (fields.get("billing_group_limit_source"), fields.get("billing_group_limit_revision"))}
    applied = {"root": raised_cap(raised, cap, root_source, group=False),
               "group": raised_cap(raised, fields.get("billing_group_limit_usd"), sources["group"][0], group=True)}
    limits = {axis: value for axis, (value, _) in applied.items()}
    sources.update({axis: (WORK_CAP_SCOPE, raised["revision"]) for axis, (_, lifted) in applied.items() if lifted})
    fallback = (raised or {}).get("verified_original")
    if fallback and limits["group"] is not None and decimal_of(limits["group"]) == decimal_of(fallback["billing_group_limit_usd"]):
        sources["group"] = fallback["billing_group_limit_source"], fallback["billing_group_limit_revision"]
    original_group_limit = (raised or {}).get("original_limit_usd", fields.get("billing_group_limit_usd"))
    pending_group = bool(pending and work_cap_amount(original_group_limit) is not None
                         and str(sources["group"][0] or "") in ORDINARY_GROUP_CAP_SOURCES)
    group_state = "unverified_export" if fallback else ("reconcile_pending" if pending_group else "")
    axes = {name: {"settled_usd": spent["settled_usd"], "accounted_usd": spent["accounted_usd"],
                   "limit_usd": limits[name], "source": sources[name][0], "revision": sources[name][1],
                   **({"basis_kind": raised["basis"]["kind"], "original_limit_usd": raised["original_limit_usd"]}
                      if applied[name][1] else {}),
                   **({"amendment_state": group_state} if group_state and name == "group" else {})}
            for name, spent in (("root", own), ("group", shared))}
    rooms = {key: known_room(axis["limit_usd"], axis["settled_usd"])
             for key, axis in axes.items() if axis["limit_usd"] is not None}
    prefer_group = _prefer_group_tie(cap, root_source, fields.get("billing_group_limit_usd"),
                                     fields.get("billing_group_limit_source"))
    binding = ("group" if _group_binds(rooms.get("root"), rooms.get("group"), prefer_tie=prefer_group)
               else "root" if "root" in rooms else None)
    room = rooms.get(binding)
    block = work_cap_block({name: (axis["limit_usd"], axis["source"]) for name, axis in axes.items()},
                           rooms, binding, original_group_limit, unverified=bool(fallback), reconcile_pending=pending_group)
    # A root ceiling is expressed on the outgoing group spend basis, without display rounding.
    with exact_money():
        effective = float(decimal_of(shared["settled_usd"]) + room) if room is not None else None
    return {"settled_usd": shared["settled_usd"], "accounted_usd": shared["accounted_usd"],
            "root_limit_usd": effective,
            "remaining_known_usd": None if room is None else float(room), "root_axis": axes["root"], "group_axis": axes["group"],
            "accounting_basis": "billing_group", "binding_axis": binding,
            **({"work_cap_block": {"error": block[0], "axis": block[1]}} if block else {}),
            "integrity_degraded": False, "age_sec": 0.0}


def task_money_snapshot(root, task, root_id, *, root_limit=None, strict_authority: bool = False):
    """One ledger observation of both independent ceilings, on group spend basis.

    Rooms are limit minus KNOWN (settled) spend, the reservation's own rule;
    ``accounted_usd`` is the group's exposure including open holds, disclosed.
    root_limit_usd is the effective allowance on the known basis; actual authored
    caps and attribution are exposed in root_axis/group_axis. Never cached
    under a shared group key, and never an amendment to a saved cost ceiling.
    """
    from ouroboros import usage_accounting as ua
    from ouroboros import usage_store

    try:
        fields = task_billing_fields(task, root_id, root_limit, root)
    except BillingAuthorityUnavailable:
        if strict_authority:
            raise
        return None
    store_root = ua._drive_root(root)
    with usage_store.read(store_root) as txn:
        snapshot = _task_money_snapshot_in_txn(txn, fields, root_id)
    snapshot["integrity_degraded"] = usage_store.integrity_degraded(store_root)
    return snapshot
