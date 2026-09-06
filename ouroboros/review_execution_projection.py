"""Tiny public read-side projections from review actor records.

This leaf owns the cross-surface presentation wire: execution receipts and the
bounded finding rows. It deliberately imports no review engine: callers pass
returned actor facts, and only actual receipt/finding content is projected.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from ouroboros.observability import redact_projection
from ouroboros.utils import truncate_review_artifact

# Bounded structured findings on the public actor projection: the bound is the
# ROW COUNT, never a second aggressive cut of the finding bodies — capping the
# only owner-reachable copy of reviewer text is the class v6.70.0 removed. The
# per-string bound mirrors plan review's MAX_FINDING_TEXT_CHARS (2000), not the
# 200-char path-list default; the durable remainder stays addressable through
# the actor's existing response_ref, so no full-set hash is needed.
MAX_PROJECTED_ACTOR_FINDINGS = 8
PROJECTED_FINDING_TEXT_CHARS = 2000
_PROJECTED_FINDING_KEYS = (
    "id", "severity", "verdict", "item", "summary", "evidence", "reason",
    "recommendation",
)


def projected_finding_row(item: Any) -> Dict[str, str]:
    """One bounded, redacted finding row for the public actor projection."""
    row: Dict[str, str] = {}
    if not isinstance(item, dict):
        return row
    for key in _PROJECTED_FINDING_KEYS:
        value = item.get(key)
        if value is None or value == "":
            continue
        if isinstance(value, str):
            rendered = str(redact_projection(value).value)
        else:
            # A non-string value keeps structural key-based masking: str()
            # first would flatten a nested secret past the key-name redactor.
            rendered = json.dumps(
                redact_projection(value).value, ensure_ascii=False, default=str,
            )
        row[key] = truncate_review_artifact(rendered, PROJECTED_FINDING_TEXT_CHARS)
    if not row:
        # An unknown finding shape still carries evidence; a silently empty row
        # would destroy it without a trace. Redact the OBJECT before
        # serializing: structural key-based secret masking does not survive a
        # pre-serialized string.
        row["item"] = truncate_review_artifact(
            json.dumps(redact_projection(item).value, ensure_ascii=False, default=str),
            PROJECTED_FINDING_TEXT_CHARS,
        )
    return row


_API_EXECUTION_RECEIPT_KEYS = frozenset({
    "resolved_model", "provider", "prompt_tokens", "completion_tokens",
    "cached_tokens", "cache_write_tokens", "cost", "total_cost",
    "ledger_attempt_ids",
})


def _has_receipt_value(value: Any) -> bool:
    """Return whether a receipt field carries a non-placeholder value."""
    if value is None or value == "":
        return False
    if isinstance(value, (list, tuple, set, dict)):
        return any(_has_receipt_value(item) for item in value)
    return True


def _has_api_execution_receipt(usage: Dict[str, Any]) -> bool:
    """Require at least one substantive allowlisted API receipt fact."""
    return any(
        key in usage and _has_receipt_value(usage.get(key))
        for key in _API_EXECUTION_RECEIPT_KEYS
    )


def normalize_review_executions(value: Any) -> List[Dict[str, str]]:
    """Allowlist the tiny public execution wire and deduplicate it stably."""
    out: List[Dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for item in value if isinstance(value, list) else []:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind") or "").strip().lower()
        if kind not in {"api", "harness", "native"}:
            continue
        harness_id = str(item.get("harness_id") or "").strip() if kind == "harness" else ""
        model = str(item.get("model") or "").strip()
        identity = (kind, harness_id, model)
        if identity in seen:
            continue
        seen.add(identity)
        row = {"kind": kind}
        if harness_id:
            row["harness_id"] = harness_id
        if model:
            row["model"] = model
        out.append(row)
    return out


def review_executions_from_actor_usage(actors: Any) -> List[Dict[str, str]]:
    """Project only executions proved by returned per-actor usage receipts."""
    executions: List[Dict[str, str]] = []
    for actor in actors if isinstance(actors, list) else []:
        if not isinstance(actor, dict):
            continue
        usage = actor.get("usage") if isinstance(actor.get("usage"), dict) else {}
        delegated_route = str(usage.get("delegated_route") or "").strip()
        model = str(usage.get("resolved_model") or "").strip()
        if delegated_route or (usage.get("provider") == "claudexor" and usage.get("delegated_run_id")):
            executions.append({
                "kind": "harness", **({"harness_id": delegated_route} if delegated_route else {}),
                **({"model": model} if model else {}),
            })
        elif _has_api_execution_receipt(usage):
            # The native tool-round episode is an API execution with a
            # different DELIVERY (the reviewer retrieved the subject itself);
            # the wire says so, or the owner reads a retrieving review as a
            # packet review.
            native = str(usage.get("delivery") or "") == "native_tool_rounds"
            executions.append({
                "kind": "native" if native else "api", **({"model": model} if model else {}),
            })
    return normalize_review_executions(executions)


__all__ = [
    "MAX_PROJECTED_ACTOR_FINDINGS",
    "PROJECTED_FINDING_TEXT_CHARS",
    "normalize_review_executions",
    "projected_finding_row",
    "review_executions_from_actor_usage",
]
