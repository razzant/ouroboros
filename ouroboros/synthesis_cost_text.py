"""Render and project the pre-synthesis cost/outcome snapshot.

Extracted from ``ouroboros/agent_task_pipeline.py`` at its module-size ceiling:
one coherent concern — wording the shared root usage snapshot for synthesis
prompts and projecting it into the summary row. The pipeline re-exports every
name here (same objects), so sibling code and tests keep the historical surface.
"""

from __future__ import annotations

import json
from typing import Any, Dict

from ouroboros.cost_projection import cost_display
from ouroboros.task_results import TASK_COST_META_FIELDS


def _synthesis_cost_usd(usage: Dict[str, Any]) -> float | None:
    """Prefer the subtree snapshot; preserve legacy callers without one."""
    key = "accounted_upper_bound_usd_with_children" if "accounted_upper_bound_usd_with_children" in usage else "cost"
    value = usage.get(key)
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 and parsed == parsed else None


def _synthesis_cost_text(usage: Dict[str, Any]) -> str:
    cost = _synthesis_cost_usd(usage)
    if cost is None:
        return "cost unavailable (non-final)" if "accounted_upper_bound_usd_with_children" in usage else "cost unknown"
    if bool(usage.get("cost_with_children_partial")):
        return f"${cost:.2f} subtree cost snapshot (non-final)"
    # Rendered by the cost SSOT so a known amount is spelled the same way here as
    # on every other surface (and a null never reaches this line as $0.00).
    return cost_display({"accounted_upper_bound_usd": cost, "cost_final": True})


def _summary_row_cost_fields(usage: Dict[str, Any]) -> Dict[str, Any]:
    """Flat task-scope cost fields for the task_summary chat row (v6.82 P1).

    Mapped explicitly from the pre-synthesis usage snapshot: only that snapshot's
    honest keys are copied. Its schema deliberately differs from the full browser
    set and a non-root snapshot without accounting keys yields nothing. Never
    fabricates values; the terminal task_results checkpoint stays authoritative.
    """
    return {key: usage[key] for key in TASK_COST_META_FIELDS if key in usage}


_SYNTHESIS_USAGE_PROMPT_FIELDS = (
    "accounted_upper_bound_usd_with_children",
    "reserved_usd",
    "unresolved_upper_bound_usd",
    "unknown_unmetered",
    "ledger_integrity",
    "cost_snapshot_at",
    "cost_final",
    "cost_with_children_partial",
    "cost_accounting_status",
    "reason_code",
    "outcome_axes",
)


def _synthesis_usage_snapshot_text(usage: Dict[str, Any]) -> str:
    """Render the bounded root snapshot section shared by synthesis prompts."""
    if not (
        "accounted_upper_bound_usd_with_children" in usage
        and str(usage.get("cost_snapshot_at") or "").strip()
        and usage.get("cost_final") is False
        and usage.get("cost_with_children_partial") is True
    ):
        return ""
    projection = {
        field: usage.get(field)
        for field in _SYNTHESIS_USAGE_PROMPT_FIELDS
    }
    payload = json.dumps(projection, ensure_ascii=False, indent=2, default=str)
    return (
        "## Shared pre-synthesis cost and outcome snapshot\n"
        "`accounted_upper_bound_usd_with_children` is accounted subtree cost only. `reserved_usd` and\n"
        "`unresolved_upper_bound_usd` are separate non-final exposure fields; do not add\n"
        "them to or describe them as already included in the accounted total. This snapshot is non-final:\n"
        "summary/reflection calls happen after it. Never turn null/unavailable values into zero.\n"
        "`outcome_axes` is canonical task truth: never describe objective best_effort,\n"
        "degraded, or fail — or a non-pass review axis — as clean success.\n"
        f"{payload}\n\n"
    )


__all__ = [
    "_SYNTHESIS_USAGE_PROMPT_FIELDS",
    "_summary_row_cost_fields",
    "_synthesis_cost_text",
    "_synthesis_cost_usd",
    "_synthesis_usage_snapshot_text",
]
