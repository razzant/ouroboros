"""Ledger-derived cost breakdown endpoint (``/api/cost-breakdown``).

The physical-attempt ledger is the one cost authority; this module projects it
into the compatibility tables the UI reads (by model / api key / model category /
task category) plus the accounting envelope. Split out of ``gateway/history.py``,
which keeps the chat-history window and stays the historical import path.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any, Callable, Dict, Optional

from starlette.requests import Request
from starlette.responses import JSONResponse

log = logging.getLogger(__name__)

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
        "cost_final": bool(bucket.get("cost_final")),
        "prompt_tokens": int(bucket.get("prompt_tokens") or 0),
        "completion_tokens": int(bucket.get("completion_tokens") or 0),
        "cached_tokens": int(bucket.get("cached_tokens") or 0),
        "cache_write_tokens": int(bucket.get("cache_write_tokens") or 0),
        "prompt_cache_ttls": dict(bucket.get("prompt_cache_ttls") or {}),
    }

def _compat_cost_groups(
    groups: Dict[str, Dict[str, Any]],
    unattributed: Dict[str, Any],
    *,
    group_key: Optional[Callable[[str], str]] = None,
) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for name, raw_bucket in groups.items():
        if not (
            int(raw_bucket.get("physical_calls") or 0)
            or int(raw_bucket.get("unknown_unmetered") or 0)
            or float(raw_bucket.get("accounted_usd") or 0.0)
        ):
            continue
        key = group_key(str(name)) if group_key else str(name)
        source = _compat_cost_bucket(raw_bucket)
        if key not in result:
            result[key] = source
            continue
        target = result[key]
        for field in (
            "cost", "calls", "unknown_unmetered", "non_final_rows",
            "prompt_tokens", "completion_tokens",
            "cached_tokens", "cache_write_tokens",
        ):
            target[field] += source[field]
        target["cost_final"] = target["cost_final"] and source["cost_final"]
        for ttl, count in source["prompt_cache_ttls"].items():
            target["prompt_cache_ttls"][ttl] = int(target["prompt_cache_ttls"].get(ttl, 0)) + int(count)
    if (
        int(unattributed.get("physical_calls") or 0)
        or int(unattributed.get("unknown_unmetered") or 0)
        or float(unattributed.get("accounted_usd") or 0.0)
    ):
        result["unattributed"] = _compat_cost_bucket(unattributed)
    for bucket in result.values():
        bucket["cost"] = round(float(bucket["cost"]), 6)
    return dict(sorted(result.items(), key=lambda item: item[1]["cost"], reverse=True))

def make_cost_breakdown_endpoint(data_dir: pathlib.Path):
    async def api_cost_breakdown(_request: Request) -> JSONResponse:
        """Return ledger-derived cost and physical-attempt breakdowns."""
        try:
            from ouroboros.pricing import infer_model_category
            from ouroboros.usage_accounting import ensure_legacy_imported, usage_breakdown

            ensure_legacy_imported(data_dir)
            breakdown = usage_breakdown(data_dir)
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
                "remaining_known_usd": (
                    round(max(0.0, limit - float(breakdown.get("accounted_usd") or 0.0)), 6)
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

    return api_cost_breakdown
