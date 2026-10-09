"""Main handoff receipts over existing bindings and the terminal-delivery outbox.

A receipt describes a transfer, never liveness. Its identity is an ingress
message plus destination, or the exact task when no message identity exists.
The outbox owns persistence/replay and duplicate suppression; no second store.

The receipt answer is one typed word (``RECEIPT_STATES``), never a boolean,
because a conversion has three separable facts: the binding is committed
(the caller's, before this module runs), the Main history row is owed or was
already delivered, and the delivery transport was unavailable. A boolean read
an idempotent repeat as "unavailable" and a lost owed row as "durable".
"""
from __future__ import annotations

import hashlib
import json
import logging

log = logging.getLogger(__name__)

RECEIPT_DURABLE = "durable"                    # owed row written, live send queued
RECEIPT_ALREADY_DELIVERED = "already_delivered"  # the same receipt already reached Main
RECEIPT_UNREGISTERED = "unregistered"          # sent live, but not protected against a restart
RECEIPT_UNAVAILABLE = "unavailable"            # transport refused; nothing was sent
RECEIPT_UNBOUND = "unbound"                    # no committed project binding for this task
RECEIPT_ORIGIN_UNPROVEN = "origin_unproven"    # no recorded Main origin: no Main default is inferred
RECEIPT_NOT_ROUTABLE = "not_routable"          # the bound project is not an active room
RECEIPT_STATES = (
    RECEIPT_DURABLE, RECEIPT_ALREADY_DELIVERED, RECEIPT_UNREGISTERED, RECEIPT_UNAVAILABLE,
    RECEIPT_UNBOUND, RECEIPT_ORIGIN_UNPROVEN, RECEIPT_NOT_ROUTABLE,
)
DURABLE_RECEIPTS = frozenset({RECEIPT_DURABLE, RECEIPT_ALREADY_DELIVERED})


def handoff_identity(project_id: str, task_id: str, source_ref: object) -> str:
    """Retries sharing a captured origin share a receipt; missing origins never merge."""
    from ouroboros.projects_registry import origin_key

    origin = origin_key(source_ref)
    identity = ["origin", *origin, project_id] if origin else ["task", task_id, project_id]
    digest = hashlib.sha256(json.dumps(identity, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    return f"project-handoff:{digest}"


def enqueue_project_handoff(drive_root, task_id: str, *, source_ref=None) -> str:
    """Publish a receipt only for a proven binding with a recorded Main origin.

    Returns one of ``RECEIPT_STATES``. Callers never undo or misreport a
    successful bind on a non-durable answer; repeating the same conversion
    retries the same outbox identity, and the outbox answers
    ``already_delivered`` once the row went out. A ref-less legacy binding has
    no Main origin on record, so no receipt is written (``origin_unproven``):
    the origin is never inferred from an absent source or guessed from text.
    Every non-durable word is logged here, so a producer that discards the
    answer (the agent-side promote/scope handlers) still leaves a trace.
    """
    state = _receipt_state(drive_root, task_id, source_ref=source_ref)
    if state not in DURABLE_RECEIPTS:
        log.info("project handoff receipt for %s: %s", task_id, state)
    return state


def _receipt_state(drive_root, task_id: str, *, source_ref=None) -> str:
    from ouroboros.projects_registry import project_binding_for_task, task_presentation_snapshot
    from supervisor import terminal_delivery as outbox

    try:
        binding = project_binding_for_task(drive_root, task_id) or {}
        pid = str(binding.get("project_id") or "")
        if not pid:
            return RECEIPT_UNBOUND
        source = source_ref if source_ref is not None else binding.get("source_ref")
        if not isinstance(source, dict) or source.get("chat_id") != 1:
            return RECEIPT_ORIGIN_UNPROVEN
        snapshot = task_presentation_snapshot(drive_root, task_id, project_id=pid)
        if not snapshot["project_routable"]:
            return RECEIPT_NOT_ROUTABLE
        identity = handoff_identity(pid, task_id, source)
        outcome = outbox.enqueue_terminal_delivery_outcome(drive_root, {
            "type": "send_message", "chat_id": 1, "task_id": task_id,
            "role": "system", "system_type": "project_handoff",
            "text": snapshot["target_label"],
            "delivery_id": identity,
            "progress_meta": {
                "project_id": pid, "project_name": snapshot["project_name"],
                "task_name": snapshot["task_name"],
                "handoff_id": identity, "target_label": snapshot["target_label"],
            },
        })
        return {
            outbox.ENQUEUE_QUEUED: RECEIPT_DURABLE,
            outbox.ENQUEUE_ALREADY_DELIVERED: RECEIPT_ALREADY_DELIVERED,
            outbox.ENQUEUE_QUEUED_UNREGISTERED: RECEIPT_UNREGISTERED,
        }.get(outcome, RECEIPT_UNAVAILABLE)
    except Exception:
        log.warning("Project binding committed but handoff delivery unavailable for %s", task_id, exc_info=True)
        return RECEIPT_UNAVAILABLE
