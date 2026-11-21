"""Ledger-derived cost projections for the dashboard and root-task detail.

The physical-attempt ledger is the one cost authority; this module projects it
into the compatibility tables the UI reads (by model / api key / model category /
task category), the accounting envelope and the root-task breakdown. The
``gateway.history`` endpoint and ``gateway.tasks`` detail import seams remain.
"""

from __future__ import annotations

import asyncio
import logging
import pathlib
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional, TypedDict

from ouroboros.cost_projection import CostPresentation
from ouroboros.delegate_custody_usage import CostCustody

if TYPE_CHECKING:  # type-only: ``gateway.contracts`` re-exports this module's TypedDicts and stays transport-free
    from starlette.requests import Request
    from starlette.responses import JSONResponse

log = logging.getLogger(__name__)


class TaskCostSlice(TypedDict):
    """One recorded value of an axis inside a root's tree (a usage category, or the route that served
    the rows): its money as the card's own carrier (``build_cost_presentation``, so every surface words it
    alike) and its volume. ``tokens`` is None when no row of the slice reported tokens."""

    key: str
    cost_presentation: Optional[CostPresentation]
    model_calls: int
    sessions: int
    tokens: Optional[int]


class TaskCostBreakdown(TypedDict):
    """Root-only task-detail ledger projection, computed on read, never persisted.
    All keys appear together; ``own + children + unattributed == subtree`` (a None own counts as 0 there).
    Omit the object for non-roots, unavailable accounting or no attributable subtree rows; never invent $0.
    ``own_usd`` is None when the root's own rows exist but none has a price; ``children_usd`` and
    ``accounted_upper_bound_usd`` are None when the subtree's rows state no amount (subscription work).
    ``by_purpose`` (usage categories) and ``by_route`` (the routes that served the rows) each partition
    the same subtree."""

    own_usd: Optional[float]
    children_usd: Optional[float]
    unattributed_usd: float
    delegated_disclosed_usd: float
    # C2: own + children + unattributed is an accounted UPPER BOUND, not a receipt.
    accounted_upper_bound_usd: Optional[float]
    subscription_sessions: int
    unknown_unmetered: int
    non_final_rows: int
    cost_final: bool
    cost_custody: Optional[CostCustody]
    by_purpose: List[TaskCostSlice]
    by_route: List[TaskCostSlice]
    authority: Literal["physical_attempt_ledger"]

_ACCOUNTING_SUMMARY_FIELDS = (
    "settled_usd",
    "confirmed_usd",
    "estimated_usd",
    "reserved_usd",
    "unresolved_upper_bound_usd",
    "accounted_usd",
    "unknown_unmetered",
    "cost_final",
    # `cost_final`'s DISCLOSED CAUSE travels with the flag it explains — without it
    # the client's "Pending (N open)" text could never render (costs.js reads
    # `accounting.non_final_rows`), so the reason for a non-final cost never
    # reached the owner at all.
    "non_final_rows",
    "accounting_open_rows",  # the rows still open, the cause the Costs page names beside a non-final cost
    "attempt_counts",
)

def _compat_cost_bucket(bucket: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "cost": round(float(bucket.get("settled_usd") or 0.0), 6),
        "calls": int(bucket.get("physical_calls") or 0),
        # Keep the compatibility tables honest about rows whose settled dollar
        # amount is zero but whose accounting is still open or undisclosed.
        "unknown_unmetered": int(bucket.get("unknown_unmetered") or 0),
        "non_final_rows": int(bucket.get("non_final_rows") or 0),
        "accounting_open_rows": int(bucket.get("accounting_open_rows") or 0),  # rows that can still settle
        "cost_final": bool(bucket.get("cost_final")),
        "prompt_tokens": int(bucket.get("prompt_tokens") or 0),
        "completion_tokens": int(bucket.get("completion_tokens") or 0),
        "cached_tokens": int(bucket.get("cached_tokens") or 0),
        "cache_write_tokens": int(bucket.get("cache_write_tokens") or 0),
        "prompt_cache_ttls": dict(bucket.get("prompt_cache_ttls") or {}),
    }

def _recorded(bucket: Dict[str, Any]) -> bool:
    """The ledger recorded something for this value: a call, an unpriced row, a priced row (a disclosed
    $0 subscription session has no physical call) or an amount. A value with no row stays out, never $0."""
    return bool(
        int(bucket.get("physical_calls") or 0)
        or int(bucket.get("unknown_unmetered") or 0)
        or int(bucket.get("priced_rows") or 0)
        or int(bucket.get("subscription_sessions") or 0)
        or float(bucket.get("accounted_usd") or 0.0)
    )


def _compat_cost_groups(
    groups: Dict[str, Dict[str, Any]],
    unattributed: Dict[str, Any],
    *,
    group_key: Optional[Callable[[str], str]] = None,
) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for name, raw_bucket in groups.items():
        if not _recorded(raw_bucket):
            continue
        key = group_key(str(name)) if group_key else str(name)
        source = _compat_cost_bucket(raw_bucket)
        if key not in result:
            result[key] = source
            continue
        target = result[key]
        for field in (
            "cost", "calls", "unknown_unmetered", "non_final_rows", "accounting_open_rows",
            "prompt_tokens", "completion_tokens",
            "cached_tokens", "cache_write_tokens",
        ):
            target[field] += source[field]
        target["cost_final"] = target["cost_final"] and source["cost_final"]
        for ttl, count in source["prompt_cache_ttls"].items():
            target["prompt_cache_ttls"][ttl] = int(target["prompt_cache_ttls"].get(ttl, 0)) + int(count)
    if _recorded(unattributed):
        result["unattributed"] = _compat_cost_bucket(unattributed)
    for bucket in result.values():
        bucket["cost"] = round(float(bucket["cost"]), 6)
    return dict(sorted(result.items(), key=lambda item: item[1]["cost"], reverse=True))

def make_cost_breakdown_endpoint(data_dir: pathlib.Path):
    from starlette.responses import JSONResponse

    def _period_view(period: str) -> Dict[str, Any]:
        """The optional period view (``cost_period``); its own failure never takes the page's totals with it."""
        from ouroboros.gateway.cost_period import PERIODS, cost_period

        if period not in PERIODS:
            return {"available": False, "period": period, "error_code": "unknown_period", "periods": list(PERIODS)}
        try:
            return cost_period(data_dir, period)
        except Exception:
            log.exception("Cost period view unavailable")
            return {"available": False, "period": period, "error_code": "ledger_unavailable"}

    def _cost_breakdown_response(period: str = "") -> JSONResponse:
        try:
            from ouroboros.pricing import infer_model_category
            from ouroboros.usage_accounting import usage_breakdown

            # Display read: the store's summary rows; contention past the short display
            # wait reports accounting unavailable (503), never a zero.
            breakdown = usage_breakdown(data_dir, allow_stale=True)
            unattributed = dict(breakdown.get("unattributed") or {})
            by_model_raw = dict(breakdown.get("by_model") or {})
            try:
                from supervisor.state import TOTAL_BUDGET_LIMIT

                live_limit = float(TOTAL_BUDGET_LIMIT or 0.0)
            except (ImportError, TypeError, ValueError):
                live_limit = 0.0
            from ouroboros.settings_setup_contract import resolve_total_budget_usd
            resolved_limit = resolve_total_budget_usd()
            limit = live_limit if 0 < live_limit < float("inf") else float(resolved_limit or 0.0)
            accounting = {field: breakdown.get(field) for field in _ACCOUNTING_SUMMARY_FIELDS}
            accounting.update({
                "available": True,
                "authority": "physical_attempt_ledger",
                "limit_usd": round(limit, 6),
                # Room above KNOWN (settled) spend, the admission rule's own number.
                "remaining_known_usd": (
                    round(max(0.0, limit - float(breakdown.get("settled_usd") or 0.0)), 6)
                    if limit > 0
                    else None
                ),
            })
            return JSONResponse({
                # Compatibility fields now project the physical-attempt ledger;
                # events.jsonl is import evidence, never a second cost authority.
                "total_cost": round(float(breakdown.get("settled_usd") or 0.0), 6),
                "total_calls": int(breakdown.get("physical_calls") or 0),
                "total_prompt_tokens": int(breakdown.get("prompt_tokens") or 0),
                "total_completion_tokens": int(breakdown.get("completion_tokens") or 0),
                "total_cached_tokens": int(breakdown.get("cached_tokens") or 0),
                "total_cache_write_tokens": int(breakdown.get("cache_write_tokens") or 0),
                "prompt_cache_ttls": dict(breakdown.get("prompt_cache_ttls") or {}),
                "by_model": _compat_cost_groups(by_model_raw, dict(unattributed.get("model") or {})),
                "by_api_key": _compat_cost_groups(
                    dict(breakdown.get("by_provider") or {}),
                    dict(unattributed.get("provider") or {}),
                ),
                "by_model_category": _compat_cost_groups(
                    by_model_raw,
                    dict(unattributed.get("model") or {}),
                    group_key=infer_model_category,
                ),
                "by_task_category": _compat_cost_groups(
                    dict(breakdown.get("by_category") or {}),
                    dict(unattributed.get("category") or {}),
                ),
                "accounting": accounting,
                "unattributed": unattributed,
                **({"period": _period_view(period)} if period else {}),
            })
        except Exception:
            log.exception("Physical-attempt accounting unavailable")
            return JSONResponse({
                "error": "Physical-attempt accounting unavailable",
                "accounting": {
                    "available": False,
                    "authority": "physical_attempt_ledger",
                    "cost_final": False,
                    "error_code": "ledger_unavailable",
                },
            }, status_code=503)

    async def api_cost_breakdown(request: Request) -> JSONResponse:
        """Return ledger-derived cost and physical-attempt breakdowns; ``?period=24h|7d|30d`` adds that period's view."""
        # Off the event loop (as ``gateway/state.py`` does): a ledger read, stale
        # path or a cold first replay, must never hold every HTTP client behind it.
        params = getattr(request, "query_params", None) or {}
        return await asyncio.to_thread(_cost_breakdown_response, str(params.get("period") or ""))

    return api_cost_breakdown


def _cost_slices(groups: Any, unattributed: Any) -> List[TaskCostSlice]:
    """One slice per recorded value of an axis (the rows naming none as key ""), ordered by tracked money,
    then volume. A slice of disclosed $0 sessions stays; a value with no row at all is no slice."""
    from ouroboros.cost_projection import COST_SCOPE_ROOT_TREE, build_cost_presentation

    pairs = list(groups.items()) if isinstance(groups, dict) else []
    slices: List[TaskCostSlice] = []
    for key, bucket in [*pairs, ("", unattributed)]:
        counts = ("priced_rows", "unknown_unmetered", "physical_calls", "subscription_sessions")
        if not isinstance(bucket, dict) or not any(int(bucket.get(name) or 0) for name in counts):
            continue
        tokens = bucket.get("total_tokens")
        slices.append({"key": str(key), "cost_presentation": build_cost_presentation(bucket, scope=COST_SCOPE_ROOT_TREE),
                       "model_calls": int(bucket.get("physical_calls") or 0),
                       "sessions": int(bucket.get("subscription_sessions") or 0),
                       "tokens": int(tokens) if isinstance(tokens, (int, float)) and not isinstance(tokens, bool) else None})
    return sorted(slices, key=lambda item: (-float((item["cost_presentation"] or {}).get("tracked_amount") or 0.0),
                                            -(item["tokens"] or 0), item["key"]))


def _task_cost_breakdown_view(drive_root: pathlib.Path, result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Read-side "where did the money go" projection for a ROOT task's detail.

    Computed from the physical-attempt ledger AT READ TIME and never persisted
    into the task result — the ledger stays the single monetary authority (P7);
    the stored envelope keeps only its existing own/subtree projections.
    ``children_usd`` is subtree − own − unattributed (the subtraction every
    reader had to do by hand); ``delegated`` is a filter over the execution
    axis (subscription sessions), not a third sum. Unavailable accounting
    returns None — the field is simply absent, never a confident $0. That
    covers BOTH an unreadable ledger and a readable one that holds no
    attributable row for this subtree (empty or legacy-only): ``_summary()``
    always returns a float for ``accounted_usd``, so "no accounting happened"
    is decided on the ROW COUNTS, never on the dollar sum being 0.0. Rows that
    state no amount (subscription work) keep the view and its slices, with the
    amounts they cannot state as None."""
    task_id = str(result.get("task_id") or "")
    root_id = str(result.get("root_task_id") or "") or task_id
    # Subtree math is ledger-attributable only at the root (child rows carry
    # the ROOT's id, not every ancestor's); non-root details omit the view.
    if not task_id or root_id != task_id:
        return None
    try:
        from ouroboros.cost_projection import honest_accounted_amount
        from ouroboros.usage_accounting import usage_breakdown

        # Read at display time only: the root's summary rows (unavailable under contention).
        breakdown = usage_breakdown(pathlib.Path(result.get("budget_drive_root") or drive_root),
                                    root_task_id=root_id, custody_root=drive_root, allow_stale=True, include_custody=True)
    except Exception:
        log.debug("cost breakdown view unavailable for %s", task_id, exc_info=True)
        return None
    subtree = honest_accounted_amount(breakdown)
    counts = breakdown.get("attempt_counts")
    counts = counts if isinstance(counts, dict) else {}
    # `metadata_only` is a count of AMBIGUOUS legacy calls carrying no money, so
    # it can never make a $0 measured; only priced attempt rows or subscription
    # sessions can. With neither, nothing was accounted for this subtree and the
    # view is ABSENT — the empty/legacy-ledger case that a `0.0 == measured zero`
    # reading would have published as `own 0 / children 0 / cost_final true`.
    priced_rows = sum(int(value or 0) for key, value in counts.items() if key != "metadata_only")
    sessions = int(breakdown.get("subscription_sessions") or 0)
    if priced_rows <= 0 and sessions <= 0:
        return None
    own_bucket = (breakdown.get("by_task") or {}).get(task_id)
    # No rows attributed to the root itself is a MEASURED zero (all spend was
    # children's), not an unknown — unknowns ride `unknown_unmetered` below. Own
    # rows that exist with no price are unknown, never a confident $0 (11.08).
    own = float(own_bucket.get("accounted_usd") or 0.0) if isinstance(own_bucket, dict) else 0.0
    own_shown = honest_accounted_amount(own_bucket) if isinstance(own_bucket, dict) else 0.0
    # Money inside this subtree that no task id claims (legacy/blank-task rows)
    # is DISCLOSED on its own axis instead of being silently folded into the
    # children's share: own + children + unattributed == subtree.
    unattributed_bucket = (breakdown.get("unattributed") or {}).get("task")
    unattributed = (
        float(unattributed_bucket.get("accounted_usd") or 0.0)
        if isinstance(unattributed_bucket, dict) else 0.0
    )
    delegated = breakdown.get("delegated") if isinstance(breakdown.get("delegated"), dict) else {}
    axis_unattributed = breakdown.get("unattributed") if isinstance(breakdown.get("unattributed"), dict) else {}
    return {
        "own_usd": round(own_shown, 6) if own_shown is not None else None,
        "children_usd": round(max(0.0, float(subtree) - own - unattributed), 6) if subtree is not None else None,
        "unattributed_usd": round(unattributed, 6),
        "delegated_disclosed_usd": round(float(delegated.get("settled_usd") or 0.0), 6),
        # C2: the explicit subtree total under its honest name — an accounted
        # UPPER BOUND (own + children + unattributed), not a settled receipt.
        "accounted_upper_bound_usd": round(float(subtree), 6) if subtree is not None else None,
        "subscription_sessions": sessions,
        "unknown_unmetered": breakdown.get("unknown_unmetered"),
        "non_final_rows": breakdown.get("non_final_rows"),
        "cost_final": bool(breakdown.get("cost_final")),
        "cost_custody": breakdown.get("cost_custody"),
        # The same read's two partitions of the subtree: what it was for, and which route served it.
        "by_purpose": _cost_slices(breakdown.get("by_category"), axis_unattributed.get("category")),
        "by_route": _cost_slices(breakdown.get("by_provider"), axis_unattributed.get("provider")),
        "authority": "physical_attempt_ledger",
    }
