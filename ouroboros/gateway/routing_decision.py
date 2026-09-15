"""The routing family of ``POST /api/decisions`` — the #198 picker dispatcher.

An owner's click on a routing picker card executes the chosen action WITHOUT
a new LLM turn, by reusing the exact supervisor handlers the LLM routing
tools already use: the event goes into the same worker-event queue
(``supervisor.workers.get_event_q``) that ``ouroboros/tools/control.py``
feeds, and the outcome is read from the same durable receipts
(``routing_wait`` — the task-result ``promotion_admission`` record and the
chat-annotation receipt). No parallel steering/promotion machinery exists
here; this module only validates the click against the durable refusal row
and translates it into the established event shapes.

Idempotency without a new registry (owner decision 1=A): the dispatched
event's ``routing_token`` and (for a new task) the ``task_id`` are DERIVED
deterministically from ``(client_message_id, refusal routing_token,
option_index)`` — a replayed request re-derives the same identities, so
the supervisor's admission reservation and the steer mailbox ``msg_id``
dedupe it instead of double-dispatching. After a confirmed dispatch the
gateway appends one closing annotation row under the ORIGINAL refusal token
carrying the ``request_id``, so replays read their own confirmation and a
different click honestly loses with the card's true state.
"""

from __future__ import annotations

import hashlib
import logging
import pathlib
from typing import Any, Dict, Tuple

from ouroboros.utils import utc_now_iso

log = logging.getLogger(__name__)

_DISPATCH_STATUS_BY_ACTION = {"steer_task": "delivered", "new_task_in_project": "scheduled"}


def parse_routing_decision_id(decision_id: str) -> Tuple[str, str]:
    """``routing:{client_message_id}:{routing_token}`` → (cmid, token)."""
    parts = str(decision_id or "").split(":")
    if len(parts) < 3 or parts[0] != "routing" or not parts[1] or not parts[-1]:
        return "", ""
    return ":".join(parts[1:-1]), parts[-1]


def _derived_identity(client_message_id: str, token: str, option_index: int) -> Tuple[str, str]:
    """Deterministic (dispatch routing_token, task_id) for idempotent replays."""
    seed = f"{client_message_id}:{token}:{int(option_index)}".encode("utf-8")
    dispatch_token = hashlib.sha256(seed + b":token").hexdigest()
    task_id = hashlib.sha256(seed + b":task").hexdigest()[:16]
    return dispatch_token, task_id


def _origin_message(drive_root: pathlib.Path, client_message_id: str) -> Dict[str, Any]:
    """Recover the owner's ORIGINAL message row by client_message_id.

    The picker transports the owner's exact ingress bytes to the chosen
    destination (the same rule the steer tool follows: a paraphrase must
    never replace the owner's text). Bounded reverse scan over the live
    chat log plus its rotation archives."""
    from ouroboros.gateway._helpers import read_rotated_jsonl_entries

    target = str(client_message_id or "")

    def _is_origin(entry: Dict[str, Any]) -> bool:
        return (
            str(entry.get("direction") or "") == "in"
            and str(entry.get("client_message_id") or "") == target
        )

    try:
        entries = read_rotated_jsonl_entries(
            drive_root / "logs" / "chat.jsonl",
            drive_root / "archive",
            "chat",
            want=1,
            counts_toward_quota=_is_origin,
        )
    except Exception:
        log.debug("origin-message scan failed for %s", client_message_id, exc_info=True)
        return {}
    for entry in reversed(list(entries or [])):
        if isinstance(entry, dict) and _is_origin(entry):
            return entry
    return {}


def handle_routing_decision(
    drive_root: pathlib.Path, *, request_id: str, decision_id: str,
    option_index: int, comment: str = "",
) -> Tuple[int, Dict[str, Any]]:
    """Execute one picker choice; returns (http_status, payload)."""
    client_message_id, token = parse_routing_decision_id(decision_id)
    if not client_message_id or not token:
        return 400, {"ok": False, "error": "malformed_decision_id",
                     "decision_id": decision_id}
    from ouroboros.project_dialogue import append_chat_annotation, latest_chat_annotations

    # The card is live only while its token is the message's LATEST act:
    # receipts are kept per token, but a newer routing attempt on the same
    # message supersedes the picker, so the click reads the latest row and
    # settles instead of retrying when the token no longer matches.
    latest = latest_chat_annotations(drive_root).get(client_message_id, {})
    receipt = latest if str(latest.get("routing_token") or "") == token else {}
    if not receipt:
        return 409, {"ok": False, "error": "decision_superseded",
                     "decision_id": decision_id,
                     "state": "superseded",
                     "latest_status": str(latest.get("status") or "")}
    status = str(receipt.get("status") or "")
    same_request = str(receipt.get("detail") or "") == f"request:{request_id}"
    if status == "dispatch_pending" and not same_request:
        # First-wins BEFORE the side effect (the quiz-card discipline): a
        # competing click while another request's dispatch is in flight is
        # refused; the winner's replay (same request_id) re-enters below.
        return 409, {"ok": False, "error": "dispatch_in_flight",
                     "decision_id": decision_id, "state": "pending"}
    if status == "dispatch_pending" and same_request:
        # The replay must name the SAME option the claim bound: the dispatch
        # identity derives from the option, so a same-id different-option
        # request is a competing click wearing the winner's key.
        claimed_option = str(receipt.get("reason") or "")
        current_option = (
            f"claimed_option:{int(option_index)}"
            if isinstance(option_index, int) and not isinstance(option_index, bool)
            else ""
        )
        if claimed_option != current_option:
            return 409, {"ok": False, "error": "request_option_mismatch",
                         "decision_id": decision_id, "state": "pending"}
    if status not in {"needs_manual_target", "dispatch_pending"}:
        if same_request:
            payload = {"ok": True, "decision_id": decision_id,
                       "state": "answered", "duplicate": True,
                       "dispatched": status}
            recorded = str(receipt.get("reason") or "")
            if recorded.startswith("answered_option:"):
                try:
                    payload["answered_index"] = int(recorded.split(":", 1)[1])
                except ValueError:
                    pass
            return 200, payload
        return 409, {"ok": False, "error": "decision_closed",
                     "decision_id": decision_id, "state": "answered",
                     "dispatched": status}
    options = receipt.get("options") if isinstance(receipt.get("options"), list) else []
    if not isinstance(option_index, int) or not (0 <= option_index < len(options)):
        return 400, {"ok": False, "error": "option_out_of_range",
                     "decision_id": decision_id}
    chosen = options[option_index] if isinstance(options[option_index], dict) else {}
    action = str(chosen.get("action") or "")
    if action not in _DISPATCH_STATUS_BY_ACTION:
        return 400, {"ok": False, "error": "option_not_dispatchable",
                     "decision_id": decision_id, "action": action}

    origin = _origin_message(drive_root, client_message_id)
    # VERBATIM contract: the dispatched message is the owner's exact ingress
    # bytes; strip() is only the emptiness CHECK, never a rewrite.
    raw_origin_text = str(origin.get("text") or "")
    origin_text = raw_origin_text
    if not raw_origin_text.strip():
        # The card stays OPEN and truthful: nothing was dispatched, and the
        # durable refusal row still says needs_manual_target.
        return 409, {"ok": False, "error": "origin_text_unavailable",
                     "decision_id": decision_id, "state": "open",
                     "reason": "the original message could not be recovered"}
    if comment:
        origin_text = f"{origin_text}\n\n[Owner picker comment] {comment}"
    chat_id = int(origin.get("chat_id") or 0)

    attachment_uploads = (
        [dict(row) for row in receipt.get("attachment_manifest") if isinstance(row, dict)]
        if isinstance(receipt.get("attachment_manifest"), list) else []
    )
    dispatch_token, derived_task_id = _derived_identity(client_message_id, token, option_index)
    # Origin provenance BY VALUE from the canonical row itself (the same rail
    # the LLM promote path rides): identity ref + full text + surface fact.
    from ouroboros.project_dialogue import build_owner_message_ref

    provenance: Dict[str, Any] = {
        "source_ref": build_owner_message_ref(
            chat_id=chat_id, client_message_id=client_message_id,
            ts=str(origin.get("ts") or ""), text=raw_origin_text,
        ),
        "source_text": raw_origin_text,
    }
    if isinstance(origin.get("client_surface"), dict) and origin.get("client_surface"):
        provenance["client_surface"] = dict(origin["client_surface"])
    if action == "steer_task":
        evt: Dict[str, Any] = {
            "type": "steer_task",
            "routing_token": dispatch_token,
            "target_task_id": str(chosen.get("task_id") or ""),
            "message": origin_text,
            "chat_id": chat_id,
            "client_message_id": client_message_id,
            # The owner clicked: an owner turn by construction. The option list
            # was host-built for this exact message's lane, and the room veto
            # reads that lane from the origin chat (Main sees every root).
            "issuer": {"kind": "owner_turn"},
            "attachment_uploads": attachment_uploads,
            **provenance,
            "ts": utc_now_iso(),
        }
    else:
        evt = {
            "type": "promote_chat_to_task",
            "task_id": derived_task_id,
            "routing_token": dispatch_token,
            "objective": origin_text,
            "project_id": str(chosen.get("project_id") or ""),
            "chat_id": chat_id,
            # Label-only provenance for the receipt: the picker click IS a
            # routing decision on a refused message, so it wears the
            # route_to_project receipt label regardless of source chat.
            "routed_from_main": True,
            "client_message_id": client_message_id,
            "attachment_uploads": attachment_uploads,
            **provenance,
            "ts": utc_now_iso(),
        }

    def _reopen_refusal() -> None:
        """Restore the actionable refusal as the LATEST row (same token)."""
        try:
            # Token-bound: if a NEWER routing attempt re-minted the card while
            # our dispatch settled, its authority wins and this reopen is a
            # no-op (the allowed set covers only our claim and our dispatch
            # receipt, never a foreign token).
            append_chat_annotation(
                drive_root, client_message_id, action="route_decision",
                status="needs_manual_target", routing_token=token,
                options=options,
                attachment_manifest=receipt.get("attachment_manifest"),
                require_latest_token={token, dispatch_token},
            )
        except Exception:
            log.debug("refusal reopen append failed", exc_info=True)

    # Claim BEFORE the side effect (first-wins): a compare-and-append under
    # the annotations lock — two concurrent clicks cannot both claim. The
    # claim row carries the options so a crash/reload before settlement still
    # renders honestly and the winner's replay re-enters with full validation
    # authority (its own claim is already latest, so it skips this step).
    if status == "needs_manual_target":
        claimed = append_chat_annotation(
            drive_root, client_message_id, action="route_decision",
            status="dispatch_pending", routing_token=token,
            detail=f"request:{request_id}", options=options,
            reason=f"claimed_option:{int(option_index)}",
            attachment_manifest=receipt.get("attachment_manifest"),
            require_latest_status={"needs_manual_target"},
            require_latest_token={token},
        )
        if not claimed:
            # Lost the claim race (or a transient lock failure) — the card
            # reads pending; a later click sees the settled truth.
            return 409, {"ok": False, "error": "dispatch_in_flight",
                         "decision_id": decision_id, "state": "pending"}
    try:
        from multiprocessing.reduction import ForkingPickler

        ForkingPickler.dumps(dict(evt))
        from supervisor.workers import get_event_q

        get_event_q().put_nowait(dict(evt))
    except Exception as exc:
        log.warning("routing decision dispatch failed", exc_info=True)
        _reopen_refusal()  # nothing is in flight — the card stays clickable
        return 503, {"ok": False, "error": "dispatch_unavailable",
                     "decision_id": decision_id,
                     "detail": f"{type(exc).__name__}: {exc}"}

    from ouroboros.routing_wait import (
        wait_for_promotion_admission,
        wait_for_routing_annotation,
    )

    if action == "steer_task":
        outcome = wait_for_routing_annotation(drive_root, client_message_id, dispatch_token)
    else:
        outcome = wait_for_promotion_admission(
            drive_root, derived_task_id, dispatch_token,
            client_message_id=client_message_id,
        )
    outcome_status = str(outcome.get("status") or "unconfirmed")
    expected = _DISPATCH_STATUS_BY_ACTION[action]
    if outcome_status == expected:
        try:
            # The closing row IS the presentation receipt after hydration, so
            # it wears the real routing action + label ("Steered task · X"),
            # plus the request id the replay path reads back.
            append_chat_annotation(
                drive_root, client_message_id,
                action="steer_task" if action == "steer_task" else "promote_chat_to_task",
                target=str(chosen.get("task_id") or chosen.get("project_id") or ""),
                target_label=str(chosen.get("label") or chosen.get("title")
                                 or chosen.get("project_name") or ""),
                status=expected, routing_token=token,
                detail=f"request:{request_id}",
                reason=f"answered_option:{int(option_index)}",
            )
        except Exception:
            log.debug("closing annotation append failed", exc_info=True)
        payload: Dict[str, Any] = {"ok": True, "decision_id": decision_id,
                                   "state": "answered", "dispatched": expected,
                                   "answered_index": int(option_index)}
        if action != "steer_task":
            payload["task_id"] = derived_task_id
        return 200, payload
    if outcome_status in {"rejected", "needs_manual_target"}:
        # The handler's rejection receipt (under the DISPATCH token) is now
        # the latest row — re-assert the refusal under the ORIGINAL token so
        # the card the UI re-opens still validates and replays cleanly.
        _reopen_refusal()
        return 409, {"ok": False, "error": "dispatch_rejected",
                     "decision_id": decision_id, "state": "open",
                     "reason": str(outcome.get("reason") or outcome_status)}
    # Unconfirmed: honestly retriable — the derived identities make a replay
    # of the SAME request byte-identical, so the supervisor dedupes it.
    return 503, {"ok": False, "error": "dispatch_unconfirmed",
                 "decision_id": decision_id,
                 "reason": str(outcome.get("reason") or "confirmation_timeout")}


__all__ = ["handle_routing_decision", "parse_routing_decision_id"]
