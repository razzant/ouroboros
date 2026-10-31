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

PROMOTE_CONFIRM_POLL_SEC = 0.05


def is_own_admission_stub(result: Any, routing_token: str) -> bool:
    """The admission gates' form: a stub is read around ONLY by the token that wrote
    it. No token is no claim, so a tokenless admission meets any stub as a row that
    owns the id (``is_emitted_admission_stub`` without a token is for readers)."""
    token = str(routing_token or "").strip()
    return bool(token) and is_emitted_admission_stub(result, token)

def _confirm_wait_sec(timeout_sec: float | None) -> float:
    """The one bound of both confirmation waits (``runtime_limits``), read at the
    wait itself so this routing leaf keeps no import-time edge into the limits."""
    if timeout_sec is not None:
        return max(0.0, float(timeout_sec))
    from ouroboros.runtime_limits import get_promote_confirm_wait_sec

    return get_promote_confirm_wait_sec()

# The promote's OWN pre-receipt, written at emit into the task-result record it
# reserves: the request exists and durably says so, while the supervisor alone
# still grants the positive `scheduled` authority. `emitted` is therefore never a
# settled admission for the waiter, and never an owner of the id for the three
# admission gates (`is_emitted_admission_stub`).
PROMOTION_ADMISSION_EMITTED = "emitted"


def is_emitted_admission_stub(result: Any, routing_token: str = "") -> bool:
    """Whether ``result`` is ONLY a promote's emitted stub - optionally of this
    exact routing token, which is how an admission recognises its own pre-receipt
    instead of reading it as another task already owning the id."""
    if not isinstance(result, dict):
        return False
    admission = result.get("promotion_admission")
    token = str(routing_token or "").strip()
    return bool(
        isinstance(admission, dict)
        and str(admission.get("status") or "") == PROMOTION_ADMISSION_EMITTED
        and str(result.get("status") or "") == "requested"
        and (not token or str(admission.get("routing_token") or "") == token)
    )


def wait_for_promotion_admission(
    root: Path,
    task_id: str,
    routing_token: str,
    *,
    client_message_id: str = "",
    timeout_sec: float | None = None,
    poll_sec: float = PROMOTE_CONFIRM_POLL_SEC,
) -> Dict[str, Any]:
    """Wait for matching-token admission in the canonical task-result SSOT."""
    from ouroboros.task_results import load_task_result

    deadline = time.monotonic() + _confirm_wait_sec(timeout_sec)
    while True:
        result = load_task_result(root, task_id) or {}
        admission = result.get("promotion_admission")
        if (
            isinstance(admission, dict)
            and str(admission.get("routing_token") or "") == routing_token
        ):
            status = str(admission.get("status") or "")
            if status in {"scheduled", "rejected", "unconfirmed", "not_started"}:
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
    timeout_sec: float | None = None,
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
    deadline = time.monotonic() + _confirm_wait_sec(timeout_sec)
    while True:
        receipt = chat_annotation_receipt(root, client_message_id, routing_token)
        status = str(receipt.get("status") or "")
        if status in {"delivered", "read", "needs_manual_target", "rejected", "unconfirmed"}:
            return receipt
        if time.monotonic() >= deadline:
            return {"status": "unconfirmed", "reason": "confirmation_timeout"}
        time.sleep(poll_sec)


# A promote and a route mint a root: their positive outcome is the canonical admission.
ADMISSION_ROUTING_ACTIONS = frozenset({"promote_chat_to_task", "route_to_project"})


def routing_act_fields(root: Path, client_message_id: str, row: Dict[str, Any]) -> Dict[str, Any]:
    """One recorded routing act as every reader shows it: the decision turn's startup
    receipts and the live observer (``loop_model_call._append_routing_receipts``).

    A promote or route reads ``scheduled`` only from its admission, read exactly as
    its own waiter reads it (one pass, no wait; the annotation is the negative-only
    fallback), so no reader states an admission the waiter cannot confirm. Any other
    act's receipt is its annotation. A refusal keeps its code; a manifest says how
    many files were not attached.
    """
    action, target = str(row.get("action") or ""), str(row.get("target") or "")
    settled = wait_for_promotion_admission(
        root, target, str(row.get("routing_token") or ""),
        client_message_id=client_message_id, timeout_sec=0,
    ) if action in ADMISSION_ROUTING_ACTIONS else row
    status, reason = str(settled.get("status") or "unconfirmed"), str(settled.get("reason") or "")
    manifest = row.get("attachment_manifest") if isinstance(row.get("attachment_manifest"), list) else []
    rejected = sum(1 for item in manifest if isinstance(item, dict) and item.get("status") == "rejected")
    return {
        "action": action, "target": target, "target_label": str(row.get("target_label") or ""),
        "status": status, "ts": str(row.get("ts") or settled.get("confirmed_at") or ""),
        "project_id": str(row.get("project_id") or ""),
        **({"retry_available": True} if settled.get("retry_available") is True else {}),
        **({"reason": reason} if reason and status in {"rejected", "needs_manual_target"} else {}),
        **({"files_not_attached": rejected} if rejected else {}),
    }


def routing_act_lines(acts: Any) -> list:
    """One line per act (``routing_act_fields``), oldest first: the observer's row and the
    facts it compares against render the same way."""
    def where(act: Dict[str, Any]) -> str:
        label, target = str(act.get("target_label") or ""), str(act.get("target") or "")
        return f"{label} ({target})" if label and target and label != target else (label or target or "no target")

    def note(act: Dict[str, Any]) -> str:
        return "".join((f" ({act['reason']})" if act.get("reason") else "",
                        ", explicit retry via start_task available" if act.get("retry_available") else "",
                        f", {act['files_not_attached']} file(s) not attached" if act.get("files_not_attached") else ""))

    return [f"- {act.get('action') or '?'} → {where(act)}: {act.get('status') or 'unknown'}{note(act)}, "
            f"recorded {act.get('ts') or '?'}" for act in sorted(acts, key=lambda row: str(row.get("ts") or ""))]
