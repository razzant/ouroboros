"""Durable routing-receipt waits (SSOT for the tool layer AND the gateway).

Extracted verbatim from ``ouroboros/tools/control.py`` (#198): the routing
picker's HTTP dispatcher must poll the SAME durable receipts the LLM routing
tools poll — the task-result ``promotion_admission`` record and the exact
chat-annotation receipt — without importing the whole tool registry into the
gateway process. Root-parameterized; ``tools/control.py`` keeps thin wrappers
that resolve the root from the tool context.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

PROMOTE_CONFIRM_TIMEOUT_SEC = 15.0
PROMOTE_CONFIRM_POLL_SEC = 0.05


def wait_for_promotion_admission(
    root: Path,
    task_id: str,
    routing_token: str,
    *,
    client_message_id: str = "",
    timeout_sec: float = PROMOTE_CONFIRM_TIMEOUT_SEC,
    poll_sec: float = PROMOTE_CONFIRM_POLL_SEC,
) -> Dict[str, Any]:
    """Wait for matching-token admission in the canonical task-result SSOT."""
    from ouroboros.task_results import load_task_result

    deadline = time.monotonic() + max(0.0, float(timeout_sec))
    while True:
        result = load_task_result(root, task_id) or {}
        admission = result.get("promotion_admission")
        if (
            isinstance(admission, dict)
            and str(admission.get("routing_token") or "") == routing_token
        ):
            status = str(admission.get("status") or "")
            if status in {"scheduled", "rejected", "unconfirmed"}:
                # The EFFECTIVE destination rides back with the receipt. The
                # admission handler re-resolves an implicit promote's project
                # under the claim lock, so the project the caller ASKED for and
                # the one the task landed in can differ; a caller that reports
                # its own argument names a room the work is not in.
                return {
                    **admission,
                    "task_status": str(result.get("status") or ""),
                    "effective_project_id": str(result.get("project_id") or ""),
                }
        # A duplicate id must never overwrite the existing task_result merely
        # to report the loser.  The exact-token chat annotation is therefore a
        # negative-only fallback; positive scheduling authority stays solely in
        # the task-result admission record.
        if str(client_message_id or "").strip():
            from ouroboros.project_dialogue import chat_annotation_receipt

            receipt = chat_annotation_receipt(
                root, str(client_message_id), routing_token
            )
            if str(receipt.get("status") or "") in {
                "needs_manual_target",
                "rejected",
                "unconfirmed",
            }:
                return receipt
        if time.monotonic() >= deadline:
            return {"status": "unconfirmed", "reason": "confirmation_timeout"}
        time.sleep(poll_sec)


def wait_for_routing_annotation(
    root: Path,
    client_message_id: str,
    routing_token: str,
    *,
    timeout_sec: float = PROMOTE_CONFIRM_TIMEOUT_SEC,
    poll_sec: float = PROMOTE_CONFIRM_POLL_SEC,
) -> Dict[str, Any]:
    """Wait for an exact existing chat-annotation receipt (manual/steer).

    ``rejected`` is a SETTLED outcome exactly like ``needs_manual_target``, and is
    accepted as one — the cancel-pending refusal (``supervisor/steering.py``) is
    the producer that writes it. While the accept-set omitted it, a refusal the
    supervisor had already written durably was polled until the deadline and then
    reported as ``confirmation_timeout``: the caller was told delivery COULD NOT BE
    CONFIRMED when the host had confirmed the opposite, and the model was invited
    to retry a steer the host will refuse again. The timeout keeps its one
    meaning: no matching receipt exists yet.
    """
    from ouroboros.project_dialogue import chat_annotation_receipt

    if not str(client_message_id or "").strip():
        return {"status": "unconfirmed", "reason": "client_message_id_missing"}
    deadline = time.monotonic() + max(0.0, float(timeout_sec))
    while True:
        receipt = chat_annotation_receipt(root, client_message_id, routing_token)
        status = str(receipt.get("status") or "")
        if status in {"delivered", "needs_manual_target", "rejected", "unconfirmed"}:
            return receipt
        if time.monotonic() >= deadline:
            return {"status": "unconfirmed", "reason": "confirmation_timeout"}
        time.sleep(poll_sec)
