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
# per-string bound is 2000 characters (not the 200-char path-list default); the
# durable remainder stays addressable through the actor's existing
# response_ref, so no full-set hash is needed.
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


# Host-owned owner labels per review surface (a closed table beside the renderer,
# never a client list); an unknown surface renders raw, disclosed.
_SURFACE_LABELS = {
    "plan_review": "Plan reviewer", "task_acceptance": "Acceptance reviewer",
    "multi_model_review": "Commit reviewer", "triad": "Commit reviewer",
    "commit_gate": "Commit reviewer", "scope_review": "Scope reviewer",
    "skill_review": "Skill reviewer", "advisory_review": "Preflight reviewer",
    "deep_self_review": "Deep-review reviewer",
}


def _reviewer_verb(surface: str, phase: str, actor: Any) -> str:
    """The verb comes from the ACTOR's typed facts, never from the phase name."""
    states = {str(getattr(actor, "operation_state", "") or ""), str(getattr(actor, "status", "") or "")}
    if "not_dispatched" in states:  # a $0 refusal is never "didn't answer" (06 invariant 25)
        return "wasn't sent " + ("the plan" if surface == "plan_review" else "its request")
    if phase == "answer_received":
        return "answer received, verdict being read"
    if getattr(actor, "transport_status", "") == "interrupted":
        return "was interrupted; its answer may be lost"
    if actor is None or phase == "started":
        return "started"
    if getattr(actor, "status", "") == "empty":
        return "gave an empty answer"
    if getattr(actor, "status", "") == "error" or getattr(actor, "ok", None) is False:
        return "didn't answer"
    if getattr(actor, "parse_status", "") == "valid":
        return "answered"
    return "returned an unreadable verdict" if getattr(actor, "parse_status", "") == "malformed" else "answer received, verdict being read"


def review_actor_progress_text(surface: str, phase: str, slot: Any, actor: Any = None) -> str:
    """One owner sentence per reviewer row: who it is (the frozen route display name),
    what it was asked for at start, and at settlement whether it answered, how it ran
    (observed harness/model/account, or that this was not reported) or why it did not
    (the engine's reported sentence, quoted; a spent window's reset instant). No code,
    id or custody state reaches the row; the record keeps them."""
    label = _SURFACE_LABELS.get(surface) or f"{surface.replace('_', ' ')} reviewer"
    verb = _reviewer_verb(surface, phase, actor)
    text = f"{label} {getattr(slot, 'model', '') or 'not specified'} {verb}"
    usage = getattr(actor, "usage", {}) or {} if actor is not None else {}
    reset_at = str(getattr(actor, "reset_at", "") or "") if actor is not None else ""
    if verb == "started":
        route = getattr(slot, "route", "")
        asked = [f"effort {getattr(slot, 'effort', '')}"] if str(getattr(route, "value", route)) == "agent_session" and getattr(slot, "effort", "") else []
        asked += [f"account {slot.session_profile}"] if getattr(slot, "session_profile", "") else []
        text += f" — {', '.join(asked)}." if asked else "."
    elif verb == "answered":
        row = (review_executions_from_actor_usage([{"usage": usage}]) or [{}])[0]
        if row.get("kind") == "harness":
            ran = row.get("model") or row.get("harness_id", "")
            text += (f" — ran as {ran}" if ran else " — how it ran was not reported") + (
                "" if row.get("model") else ("; which model served it was not reported" if ran else ""))
        elif row:
            text += f" — sent as {row.get('model') or 'not reported'}; the provider did not report which model served it"
        else:
            text += " — how it ran was not reported"
        text += (f" (account {usage['applied_profile']})" if usage.get("applied_profile") else "") + "."
    else:
        cause = str(getattr(actor, "reported_cause", "") or "").replace("\n", " ")
        if verb == "didn't answer" and cause:
            text += f' — "{cause}"'
        else:
            text += f" — its window resets at {reset_at}." if reset_at else "."
    return str(redact_projection(text).value)


__all__ = [
    "MAX_PROJECTED_ACTOR_FINDINGS",
    "PROJECTED_FINDING_TEXT_CHARS",
    "normalize_review_executions",
    "projected_finding_row",
    "review_executions_from_actor_usage",
    "review_actor_progress_text",
]
