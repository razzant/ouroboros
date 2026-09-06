"""Durable handlers for rare, typed telemetry-only worker events.

The supervisor drops an event whose type is not in EVENT_HANDLERS (it decays
into a truncated ``unknown_worker_event`` repr in supervisor.jsonl), so every
literal-typed emitter needs a registered sink. The types here share one
type-agnostic durable passthrough — each row is appended verbatim to
events.jsonl for typed consumers (budget audits, advisory forensics). They
are all LOW-RATE by contract: every dispatch is a durable append, so
high-rate narration (live progress, chat frames) must never join this
registry — it rides the addressed ``send_message``/``log_event`` paths.

Split out of ``supervisor/events.py`` (which sits at the 200,000-byte module
ratchet ceiling) the same way ``cognitive_operations`` contributes ``_CEH``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from ouroboros.utils import append_jsonl, utc_now_iso

log = logging.getLogger(__name__)


def _handle_typed_telemetry(evt: Dict[str, Any], ctx: Any) -> None:
    """Persist one typed telemetry event durably (v6.69.0 lineage).

    Type-agnostic ({ts, **evt}): the registry decides which types route
    here. Without it the worker event would land in supervisor.jsonl as an
    unknown_worker_event repr instead of a typed events.jsonl row.
    """
    try:
        append_jsonl(
            ctx.DRIVE_ROOT / "logs" / "events.jsonl",
            {"ts": evt.get("ts", utc_now_iso()), **{k: v for k, v in evt.items() if k != "ts"}},
        )
    except Exception:
        log.debug("Failed to log telemetry event %s", evt.get("type"), exc_info=True)


def _handle_task_message_injected(evt: Dict[str, Any], ctx: Any) -> None:
    """Log A2A task-message injections so health checks can detect duplicate
    processing (sibling of owner_message_injected for the task channel)."""
    try:
        ctx.append_jsonl(ctx.DRIVE_ROOT / "logs" / "events.jsonl", {
            "ts": evt.get("ts", utc_now_iso()),
            "type": "task_message_injected",
            "task_id": evt.get("task_id", ""),
            "source_task_id": evt.get("source_task_id", ""),
            "provenance": evt.get("provenance", ""),
        })
    except Exception:
        log.warning("Failed to log task_message_injected event", exc_info=True)


# Merged into supervisor.events.EVENT_HANDLERS (the `**_CEH` pattern). The
# review_wave_budget_partial_unknown twin was the v6.69.0-class omission: the
# fix registered one branch of review_helpers' if/else and missed the other.
TELEMETRY_EVENT_HANDLERS = {
    "review_density_probe": _handle_typed_telemetry,
    "review_scope_lead_unobserved": _handle_typed_telemetry,
    "review_wave_admission_unavailable": _handle_typed_telemetry,
    "review_wave_budget_insufficient": _handle_typed_telemetry,
    "review_wave_budget_partial_unknown": _handle_typed_telemetry,
    "advisory_suspect_result": _handle_typed_telemetry,
    "advisory_contract_warning": _handle_typed_telemetry,
    "plan_task_deadline_skip": _handle_typed_telemetry,
    "task_message_injected": _handle_task_message_injected,
    # #Q-2b: the owner's quiz answer landed in a worker round — same
    # duplicate-detection telemetry as its task-message sibling.
    "quiz_answer_injected": _handle_typed_telemetry,
}
