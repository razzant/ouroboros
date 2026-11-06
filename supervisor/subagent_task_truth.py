"""Delegation-truth enrichment for the subagent ``task_done`` transport frame.

Split from ``supervisor/events.py`` at its byte ceiling: the PUSHED (log
channel) ``task_done`` event is stamped with the same delegation truth the
chat frame carries — additive keys, applied only AFTER the durable
``events.jsonl`` append (transport enrichment, never a log-shape change) — so
a card whose chat frame never arrives still upgrades its executor chip from
the task_done log event instead of staying at the dispatch-only label.
"""

from __future__ import annotations

from typing import Any, Dict


def terminal_executor_route(effective_result: Dict[str, Any]) -> str:
    """The route a finished child's card names: the recorded one, or for a child refused before it started (its route
    failed the dispatch check, or its delegate tools were hidden or uncheckable) the session route its configured row
    named; the record keeps none because nothing ran on it, and the card says ``· blocked``. A blocked API-model row
    names no route, as live."""
    route = str(effective_result.get("executor_route") or "")
    if route or str(effective_result.get("effective_executor") or "") != "blocked":
        return route
    row = effective_result.get("configured_subagent")
    spec = row.get("route") if isinstance(row, dict) and isinstance(row.get("route"), dict) else {}
    if str(spec.get("kind") or "") != "agent_session":
        return ""
    from ouroboros.subagents import parse_subagent_harness

    parsed = parse_subagent_harness(spec.get("target_id"))
    return parsed.route_id if parsed is not None else ""


def enrich_task_done_event(
    task_done_event: Dict[str, Any], effective_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Stamp executor_route/execution_evidence/actual_substrate; return the envelope."""
    envelope = effective_result.get("subagent_envelope")
    envelope = envelope if isinstance(envelope, dict) else {}
    if route := terminal_executor_route(effective_result):
        task_done_event.setdefault("executor_route", route)
    if isinstance(envelope.get("execution_evidence"), dict):
        task_done_event.setdefault(
            "execution_evidence", envelope["execution_evidence"])
    if envelope.get("actual_substrate"):
        task_done_event.setdefault(
            "actual_substrate", str(envelope["actual_substrate"]))
    return envelope
