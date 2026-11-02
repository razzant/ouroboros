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
bars the next admission; it was not prevented.
"""

from __future__ import annotations

from ouroboros.store import StoreBusy, bounded_wait
from ouroboros.usage_ledger import (
    USAGE_LOCK_TIMEOUT_SEC, BillingAuthorityBusy, BillingAuthorityUnavailable, UsageLockUnavailable,
)

import logging
import math
import pathlib
from dataclasses import replace
from typing import Any, Dict, Optional, Sequence
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


def authority_unavailable(kind: str, root_task_id: str, cause: str) -> BillingAuthorityUnavailable:
    """The typed non-money refusal: ``unreadable`` (an operational fault) or ``refused`` (positive policy)."""
    return BillingAuthorityUnavailable(f"billing group authority unavailable for root {root_task_id} ({kind}: {cause})",
                                       kind=kind, cause=cause, root_task_id=root_task_id)


def ledger_billing_binding(budget_root: Any, root_task_id: str) -> Dict[str, Any]:
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
    with usage_store.hold(root, timeout_sec=bounded_wait(USAGE_LOCK_TIMEOUT_SEC), write=False) as txn:
        row = txn.binding("root", root_task_id)
    if row is None:
        return {}
    limit = row.get("billing_group_limit_usd", row.get("root_limit_usd"))
    return {"billing_group_id": str(row.get("billing_group_id") or root_task_id),
            "billing_group_limit_usd": None if limit is None else ua._number(limit),
            "billing_group_limit_source": row.get("billing_group_limit_source") or "ledger_first_row",
            "billing_group_limit_revision": row.get("billing_group_limit_revision")}


def task_billing_fields(task: Dict[str, Any], root_task_id: str, root_limit: Optional[float],
                        budget_root: Any = None, *, pin_initial: bool = False,
                        persist_initial: bool = True) -> Dict[str, Any]:
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
        historical = ledger_billing_binding(budget_root, root_task_id) if budget_root and not row else {}
        if not row and root_task_id != task_id and not historical:
            raise refused("root_lineage_missing")
        row = row or {}
        root_meta = row.get("metadata") or {}
        schedule_id = str(row.get("schedule_id") or root_meta.get("schedule_id") or metadata.get("schedule_id") or "")
        scheduled_binding = None
        if schedule_id:
            from supervisor.queue_schedules import load_schedule_store
            from supervisor.followup_policy import task_binding
            schedule = next((r for r in load_schedule_store(budget_root)["tasks"]
                             if r.get("id") == schedule_id), None)
            if schedule is None:
                # GC may remove completed schedule history. Only the producer's
                # canonical top-level binding can preserve related late costs;
                # old template metadata alone cannot establish that authority.
                relation = root_meta.get("followup_relation") or {}
                canonical = row.get("billing_group") or {}
                if (relation.get("kind") == "related" and canonical
                        and canonical == relation.get("billing_group")
                        and canonical.get("billing_group_limit_source")):
                    return {"root_limit_usd": root_limit, **canonical}
                # Independent work, or a task whose published row recorded no
                # relationship, keeps its own root; nothing else invents one.
                recorded = root_meta.get("followup_relation") if "followup_relation" in root_meta else {"kind": ""}
                if (recorded or {}).get("kind") not in {"", "independent", "unknown"}:
                    raise refused("related_followup_unbound")
            else:
                scheduled_binding = task_binding(budget_root, schedule)
                if scheduled_binding is not None:
                    return {"root_limit_usd": root_limit, **scheduled_binding}
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
            binding = ledger_billing_binding(budget_root, root_task_id)
        if not binding and pin_initial and task_id == root_task_id and budget_root:
            # A root that already ran without recording its cap (no pinned binding, no
            # usable ledger literal) is its own group under the configured cap, disclosed.
            legacy = bool(row.get("started_at")) or row.get("status") in {"completed", "cancelled", "failed"}
            binding = {"billing_group_id": root_task_id, "billing_group_limit_usd": root_limit,
                       "billing_group_limit_source": "legacy_default" if legacy else "initial_task_admission",
                       "billing_group_limit_revision": utc_now_iso()}
            def pin(current):
                nonlocal binding
                if current.get("billing_group"):
                    binding = current["billing_group"]
                    return None
                return stamp_task_result_schema({"task_id": root_task_id, "status": "requested", **current,
                                                 "billing_group": binding})
            if persist_initial:
                patch_document(pathlib.Path(budget_root), root_task_id, pin)
        if binding:
            if not isinstance(binding, dict) or not binding.get("billing_group_id") or "billing_group_limit_usd" not in binding:
                raise authority_unavailable("unreadable", root_task_id, "malformed_binding")
            if str(binding["billing_group_id"]).startswith(UNAVAILABLE_GROUP_PREFIX):
                raise refused("stand_in_binding")  # an older record's stand-in authorizes nothing
            limit = binding["billing_group_limit_usd"]
            if limit is not None and (isinstance(limit, bool) or not math.isfinite(float(limit))):
                raise authority_unavailable("unreadable", root_task_id, "malformed_binding")
            return {"root_limit_usd": root_limit, **{key: binding.get(key) for key in (
                "billing_group_id", "billing_group_limit_usd", "billing_group_limit_source",
                "billing_group_limit_revision")}}
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


def raise_group_refusal(view: Any, scope: Any) -> None:
    """The group axis of ONE reservation or dispatch, inside the caller's store transaction."""
    from ouroboros import usage_accounting as ua

    group, limit = scope_group(scope)
    if group.startswith(UNAVAILABLE_GROUP_PREFIX):  # an older record's stand-in is no bucket, never a zero cap
        raise authority_unavailable("refused", str(getattr(scope, "root_task_id", "") or group), "stand_in_binding")
    if not group or limit is None:
        return
    if view.exceeds_limit(limit, billing_group_id=group):
        raise ua.BudgetExceeded(
            f"whole-work budget exhausted for group {group}: "
            f"{ua._known_spend_text(view.summary(billing_group_id=group))}, limit=${limit:.6f}",
            limit_scope="root", root_task_id=str(getattr(scope, "root_task_id", "") or group))


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


def _per_slot(value: Any, count: int) -> list:
    """Broadcast one scalar, or align one per-slot sequence, over ``count`` slots."""
    if isinstance(value, (list, tuple)):
        values = list(value)
        return values[:count] + [values[-1] if values else 0] * max(0, count - len(values))
    return [value] * count


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
    allow_live_fetch: bool = True,
) -> Dict[str, Any]:
    """Read-only wave admission on the reservation's own known-spend rule.

    The wave is admitted while KNOWN spend is below every applicable limit
    (global, root, original group), exactly what each seat's reservation will
    check (#1487). The summed seat bounds are information, never an earlier
    refusal: a wave that crosses the limit mid-dispatch is refused per seat by
    the reservation and dispatch fences, with truthful custody of what was sent.
    Standalone callers may supply remaining_usd_override (known room). An
    explicit root_limit_usd is the caller's current fence, otherwise the
    ledger's historical minimum governs. Global None resolves settings; a
    non-positive configured limit is unbounded. Open holds are disclosed beside
    the known spend; unknown prices stay unknown.

    Input sizes, outputs, categories, slots and processing can be scalar or
    aligned per-slot values. Price each seat under its own sending scope, so the
    caller's warm cache split cannot stand in for a reviewer's cold prefix.
    Returned per-slot bounds and both remainders disclose the binding axis.
    ``allow_live_fetch=False`` prices from tariffs already cached in this process.
    """
    from ouroboros import usage_accounting as ua

    result: Dict[str, Any] = {
        "fits": True,
        "estimated_wave_usd": None,
        "remaining_usd": None,
        "limit_usd": None,
        "slots": len(list(models or [])),
        "unpriced_slots": 0,
        "known_usd": None,
        "accounted_usd": None,
        "reserved_usd": None,
        "slot_bounds": [],
        **{key: None for key in ("global_limit_usd", "global_known_usd", "global_accounted_usd",
                                 "global_remaining_usd", "global_reserved_usd", "binding_axis")},
    }
    root_task_id = str(root_task_id or "").strip()
    if not root_task_id or not models:
        return result
    try:
        from ouroboros.pricing import infer_provider_from_model

        if remaining_usd_override is not None:
            remaining = float(remaining_usd_override)
        else:
            # Open holds (reserved and in-flight/unresolved bounds) are disclosed
            # exposure beside the known spend; they do not shrink the room.
            holds = lambda p: round(float(ua._number(p.get("reserved_usd")) or 0.0)  # noqa: E731
                                    + float(ua._number(p.get("unresolved_upper_bound_usd")) or 0.0), 6)
            projection = ua.usage_projection(drive_root, root_task_id=root_task_id)
            limit = (
                max(0.0, float(root_limit_usd)) if root_limit_usd is not None
                else None if root_limit_source else ua._number(projection.get("limit_usd"))
            )
            known = ua._number(projection.get("settled_usd"))
            remaining = None
            if limit is not None and known is not None:
                remaining = round(max(0.0, limit - known), 6)
                result.update(limit_usd=limit, known_usd=known, accounted_usd=ua._number(projection.get("accounted_usd")),
                              reserved_usd=holds(projection), binding_axis="root")
            # Resolve the same task-bound durable group as a seat reservation,
            # without reserving money or pinning a binding.
            # A raised in-memory root fence alone cannot raise the original group.
            _, bound_scope = ua._merge_scope(ua.AttemptRequest(
                model="", provider="", drive_root=drive_root,
                task_id=task_id, root_task_id=root_task_id,
                root_limit_usd=root_limit_usd,
            ))
            group, group_limit = scope_group(bound_scope)
            if group and group_limit is not None:
                group_projection = ua.usage_projection(drive_root, billing_group_id=group)
                group_known = ua._number(group_projection.get("settled_usd"))
                if group_known is not None:
                    group_remaining = round(max(0.0, group_limit - group_known), 6)
                    if remaining is None or group_remaining < remaining:
                        remaining = group_remaining
                        result.update(limit_usd=group_limit, known_usd=group_known,
                                      accounted_usd=ua._number(group_projection.get("accounted_usd")),
                                      reserved_usd=holds(group_projection),
                                      binding_axis="group", billing_group_id=group)
            # The global axis reserve_attempt checks FIRST (all roots' rows).
            gp = ua.usage_projection(drive_root, global_limit_usd=global_limit_usd, include_roots=False)
            result.update(global_limit_usd=ua._number(gp.get("limit_usd")),
                          global_known_usd=ua._number(gp.get("settled_usd")),
                          global_accounted_usd=ua._number(gp.get("accounted_usd")),
                          global_remaining_usd=ua._number(gp.get("remaining_known_usd")), global_reserved_usd=holds(gp))
            global_remaining = result["global_remaining_usd"]
            if global_remaining is not None and (result["binding_axis"] is None or global_remaining < remaining):
                remaining, result["binding_axis"] = global_remaining, "global"
            if result["binding_axis"] is None:
                return result
        result["remaining_usd"] = remaining
        chars = _per_slot(prompt_chars, len(models))
        outputs = _per_slot(max_completion_tokens, len(models))
        seat_categories = _per_slot(categories, len(models))
        seat_slot_ids = _per_slot(slot_ids, len(models))
        seat_processing = _per_slot(processing_preferences, len(models))
        base_scope = ua.current_usage_scope() or ua.UsageScope()
        total = 0.0
        for index, model in enumerate(models):
            seat_scope = base_scope
            if str(seat_categories[index] or ""):
                seat_scope = replace(
                    base_scope, category=str(seat_categories[index]),
                    review_slot_id=str(seat_slot_ids[index] or ""),
                )
            with ua.usage_scope(seat_scope):
                bound = ua._reservation_cost(
                    ua.AttemptRequest(
                        model=str(model or ""),
                        provider=infer_provider_from_model(str(model or "")),
                        prompt_tokens_estimate=max(0, int(chars[index] or 0)) // 4,
                        max_completion_tokens=max(0, int(outputs[index] or 0)),
                        task_id=str(task_id or ""),
                        processing_preference=str(seat_processing[index] or ""),
                        # The captured preference projected onto the provider-neutral reservation mode.
                        submitted_processing_mode={"standard": "default", "fast": "priority", "economy": "flex"}.get(
                            str(seat_processing[index] or "").strip().lower(), ""),
                        allow_live_fetch=allow_live_fetch,
                    )
                )
            result["slot_bounds"].append(None if bound is None else round(float(bound), 6))
            if bound is None:
                # Unknown contributes no invented price and remains explicitly counted.
                result["unpriced_slots"] = int(result.get("unpriced_slots") or 0) + 1
                continue
            total += float(bound)
        result["estimated_wave_usd"] = round(total, 6)
        # The reservation's equality: no room once known spend reached the limit.
        result["fits"] = remaining > 1e-9
        return result
    except BillingAuthorityUnavailable as exc:
        # Not money: no seat could reserve. The kind says whether rereading can help.
        return {**result, "fits": False, "binding_axis": "group", "reason": "billing_authority_unavailable",
                "billing_authority": exc.kind}
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


def task_money_snapshot(root, task, root_id, *, root_limit=None):
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
        return None
    group = str(fields.get("billing_group_id") or root_id)
    with usage_store.read(ua._drive_root(root)) as txn:
        own, shared = txn.summary(root_id), txn.summary(billing_group_id=group)
        initial = txn.binding("root", root_id) or {}
        cap = fields.get("root_limit_usd")
        if cap is None and not fields.get("root_limit_source") and initial.get("root_limit_usd") is not None:
            cap = ua._number(initial["root_limit_usd"])
        axes = {"root": {"settled_usd": own["settled_usd"], "accounted_usd": own["accounted_usd"], "limit_usd": cap,
                         "source": fields.get("root_limit_source") or initial.get("root_limit_source"),
                         "revision": fields.get("root_limit_revision")},
                "group": {"settled_usd": shared["settled_usd"], "accounted_usd": shared["accounted_usd"],
                          "limit_usd": fields.get("billing_group_limit_usd"),
                          "source": fields.get("billing_group_limit_source"),
                          "revision": fields.get("billing_group_limit_revision")}}
        rooms = {key: axis["limit_usd"] - axis["settled_usd"] for key, axis in axes.items() if axis["limit_usd"] is not None}
        binding = min(rooms, key=rooms.get) if rooms else None
        room = rooms[binding] if binding else None
        return {"settled_usd": shared["settled_usd"], "accounted_usd": shared["accounted_usd"],
                "root_limit_usd": None if room is None else shared["settled_usd"] + room,
                "remaining_known_usd": room, "root_axis": axes["root"], "group_axis": axes["group"],
                "accounting_basis": "billing_group", "binding_axis": binding,
                "integrity_degraded": usage_store.integrity_degraded(ua._drive_root(root)), "age_sec": 0.0}
