"""Durable routing-receipt waits (SSOT for the tool layer AND the gateway).

Extracted verbatim from ``ouroboros/tools/control.py`` (#198): the routing
picker's HTTP dispatcher must poll the SAME durable receipts the LLM routing
tools poll — the task-result ``promotion_admission`` record and the exact
chat-annotation receipt — without importing the whole tool registry into the
gateway process. Root-parameterized; ``tools/control.py`` keeps thin wrappers
that resolve the root from the tool context.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict

log = logging.getLogger(__name__)

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


_FILE_ROW_FIELDS = ("ordinal", "status", "reason", "label")


def admitted_files(result: Dict[str, Any]) -> Dict[str, Any]:
    """The files an admitted start recorded, from the task result already read: the inline rows its admission wrote
    (a long list is cut to a preview), how many files it declared, and the descriptor of the complete list when it
    saved one. Only each row's outcome (ordinal, status, reason, label) is kept, never a path."""
    contract = result.get("task_contract") if isinstance(result.get("task_contract"), dict) else {}
    listed = result.get("attachment_manifest") or contract.get("attachment_manifest")
    rows = [{key: row[key] for key in _FILE_ROW_FIELDS if key in row}
            for row in (listed if isinstance(listed, list) else []) if isinstance(row, dict)]
    ref = contract.get("attachment_manifest_ref")
    ref = ref if isinstance(ref, dict) else {}
    count = ref.get("count")
    return {"attachment_manifest": rows, "attachment_declared": max(len(rows), count if type(count) is int else 0),
            **({"attachment_manifest_ref": ref} if ref else {})}


def complete_admitted_files(root: Path, task_id: str, receipt: Dict[str, Any]) -> list:
    """The rows of an admitted start that declared more files than its inline view holds, read once from the
    descriptor its admission saved (verified by size and digest; ``resolve_attachment_manifest`` reaches a forked
    drive's retained source from the canonical root); the inline rows when it cannot be read. Nothing is read for a
    start whose inline rows are all it declared."""
    inline = list(receipt.get("attachment_manifest") or [])
    ref, declared = receipt.get("attachment_manifest_ref"), receipt.get("attachment_declared")
    if not isinstance(ref, dict) or type(declared) is not int or declared <= len(inline):
        return inline
    from ouroboros.artifacts import resolve_attachment_manifest

    try:
        return [{key: row[key] for key in _FILE_ROW_FIELDS if key in row}
                for row in resolve_attachment_manifest(root, task_id, {"attachment_manifest_ref": ref})]
    except Exception:
        log.debug("complete attachment manifest of %s not readable", task_id, exc_info=True)
        return inline


def wait_for_promotion_admission(
    root: Path,
    task_id: str,
    routing_token: str,
    *,
    client_message_id: str = "",
    timeout_sec: float | None = None,
    poll_sec: float = PROMOTE_CONFIRM_POLL_SEC,
) -> Dict[str, Any]:
    """Wait for matching-token admission in the canonical task-result SSOT. A timeout carries
    the wait it measured (``waited_sec``, monotonic): reads can overrun the configured bound.
    A ``scheduled`` outcome also carries the files that admission recorded (``admitted_files``)."""
    from ouroboros.task_results import load_task_result

    started = time.monotonic()
    deadline = started + _confirm_wait_sec(timeout_sec)
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
                    **(admitted_files(result) if status == "scheduled" else {}),
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
            return {"status": "unconfirmed", "reason": "confirmation_timeout",
                    "waited_sec": round(time.monotonic() - started, 1)}
        time.sleep(poll_sec)


def relay_receipt_scope(event: Dict[str, Any]) -> dict:
    """The primary quote's own room and physical address, to read the same act after a timeout or restart."""
    if (str(event.get("type") or "") != "steer_task" or not event.get("relay_id") or event.get("owner_source")
            or event.get("chat_id") is None):
        return {}
    room = int(event["chat_id"])
    scope = {"source_chat_id": room}
    from ouroboros.owner_relay import quote_receipt_address
    for quote in event.get("quotes") or []:
        source = quote.get("source") or {}
        if str(source.get("chat_id")) == str(room) and source.get("client_message_id") == event.get("client_message_id"):
            if address := quote_receipt_address(source, quote.get("owner_row") or {}):
                scope["source_row_address"] = address
            break
    return scope


def recorded_act_receipt(root: Path, act: Dict[str, Any]) -> Dict[str, Any]:
    """One recorded act's latest receipt row, read by its exact token: in the room its record names
    (``source_chat_id``: a note's relay, ``relay_receipt_scope``), else by its id alone, as a record made before
    records named a room reads it; between two rooms that read names neither."""
    from ouroboros.project_dialogue import chat_annotation_receipt

    room = act.get("source_chat_id")
    scope = {"source_chat_id": room} if type(room) is int else {}
    if scope and act.get("source_row_address"):
        scope["source_row_address"] = act["source_row_address"]
    return chat_annotation_receipt(root, str(act.get("client_message_id") or ""), str(act.get("routing_token") or ""),
                                   **scope)


def wait_for_routing_annotation(
    root: Path,
    client_message_id: str,
    routing_token: str,
    *,
    timeout_sec: float | None = None,
    poll_sec: float = PROMOTE_CONFIRM_POLL_SEC,
    source_chat_id: int | None = None,
    source_row_address: str = "",
) -> Dict[str, Any]:
    """Wait for an exact existing chat-annotation receipt (manual/steer). ``source_chat_id``: the room of the
    message the act keys its receipt on, when the waiter knows it (an act quoting two rooms' same-id messages).

    ``rejected`` is a SETTLED outcome exactly like ``needs_manual_target``, and is
    accepted as one — the cancel-pending refusal (``supervisor/steering.py``) is
    the producer that writes it. While the accept-set omitted it, a refusal the
    supervisor had already written durably was polled until the deadline and then
    reported as ``confirmation_timeout``: the caller was told delivery COULD NOT BE
    CONFIRMED when the host had confirmed the opposite, and the model was invited
    to retry a steer the host will refuse again. The timeout keeps its one
    meaning: no matching receipt exists yet.
    """
    if not str(client_message_id or "").strip():
        return {"status": "unconfirmed", "reason": "client_message_id_missing"}
    act = {"client_message_id": client_message_id, "routing_token": routing_token,
           "source_chat_id": source_chat_id, "source_row_address": source_row_address}
    deadline = time.monotonic() + _confirm_wait_sec(timeout_sec)
    while True:
        receipt = recorded_act_receipt(root, act)
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


def message_routing_acts(root: Path, sources: list) -> list:
    """Confirmed message identities -> their latest receipt per act, in journal order.
    A legacy delivery with no room can name a row only when the chat history identifies it uniquely.
    """
    from ouroboros.chat_annotations import message_receipts
    return [routing_act_fields(root, message, row) for message, row in message_receipts(root, sources)]


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
