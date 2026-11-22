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


# The card's original is not among the readable rows of a history holding an unread line.
_ORIGIN_GAP = "_origin_gap"


def _origin_message(drive_root: pathlib.Path, client_message_id: str, room=None) -> Dict[str, Any]:
    """Recover the owner's ORIGINAL message row by client_message_id.

    The picker transports the owner's exact ingress bytes to the chosen
    destination (the same rule the steer tool follows: a paraphrase must
    never replace the owner's text). Only rows carrying this id are read.
    A card written before receipts named their room (``room`` None) is placed by the one readable owner row
    with its id. Two distinct readable rows with one id are told apart by nothing the card saved, and an unread
    line hides only itself: while no single readable row answers, the card stays open."""
    from ouroboros import dialogue_store as ds
    from ouroboros.project_dialogue import unique_inbound_message

    target = str(client_message_id or "")
    if not target:
        return {}
    try:
        if room is None:
            room = ds.annotation_rooms(drive_root, [target]).get(target)
        found = unique_inbound_message(drive_root, {room}, target) if room is not None else None
        if found:
            return found
        ds.require_imported(drive_root, "chat")
        if ds.range_coverage(drive_root, "chat")["gaps"]:
            return {_ORIGIN_GAP: "the original message is not among its readable rows"}
    except ds.StoreError:
        raise
    except Exception:
        log.debug("origin-message lookup failed for %s", client_message_id, exc_info=True)
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
    from ouroboros.project_dialogue import (
        ROUTING_DESTINATION_FIELDS,
        append_chat_annotation,
        chat_annotation_receipt,
        routing_refusal_cause,
    )

    # The card is live only while its token is the message's LATEST act:
    # receipts are kept per token, but a newer routing attempt on the same
    # message supersedes the picker, so the click reads the latest row and
    # settles instead of retrying when the token no longer matches.
    from ouroboros.chat_annotations import latest_picker_annotation
    room, latest = latest_picker_annotation(drive_root, client_message_id, token)
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

    from ouroboros.store import StoreError, StoreMigrating

    try:
        address = receipt.get("source_row_address")
        if address:
            from ouroboros.chat_chain import _row_chat_id, resolve_row
            row, _ = resolve_row(drive_root, address)
            origin = row if (row and _row_chat_id(row) == room and row.get("direction") == "in"
                             and row.get("client_message_id") == client_message_id) else {}
        else:
            origin = _origin_message(drive_root, client_message_id, room)
    except ValueError:
        origin = {}  # a malformed recorded address identifies no owner row
    except StoreMigrating as exc:
        return 503, {"ok": False, "error": "history_importing", "decision_id": decision_id,
                     "state": "open", "reason": str(exc)}
    except StoreError as exc:
        return 503, {"ok": False, "error": "history_unavailable", "decision_id": decision_id,
                     "state": "open", "reason": str(exc)}
    # VERBATIM contract: the dispatched message is the owner's exact ingress
    # bytes; strip() is only the emptiness CHECK, never a rewrite.
    raw_origin_text = str(origin.get("text") or "")
    if not raw_origin_text.strip():
        # The card stays OPEN and truthful: nothing was dispatched, and the
        # durable refusal row still says needs_manual_target.
        return 409, {"ok": False, "error": "origin_text_unavailable",
                     "decision_id": decision_id, "state": "open",
                     "reason": (f"The history holds an unread line, so {origin[_ORIGIN_GAP]}. "
                                "You can send a new request naming the intended work."
                                if _ORIGIN_GAP in origin else "the original message could not be recovered")}
    picker_note = f"[Owner picker comment] {comment}" if comment else ""
    from ouroboros.chat_chain import format_address, row_address
    physical = row_address(origin)
    chat_id, address = physical["chat_id"], format_address(physical)

    dispatch_token, derived_task_id = _derived_identity(client_message_id, token, option_index)
    from ouroboros.settings_scales import EFFORT_SCALE

    # #1539: the routing turn's explicit start for a NEW task picked here, kept on
    # every card row; a hand-edited unknown tier stays the default. A steer starts
    # no root, so it carries none and nothing below claims one applied.
    card_effort = str(receipt.get("reasoning_effort") or "")
    card_effort = card_effort if card_effort in EFFORT_SCALE else ""
    # Origin provenance BY VALUE from the canonical row itself (the same rail
    # the LLM promote path rides): identity ref + full text + surface fact.
    from ouroboros.owner_relay import origin_snapshot, owner_row_source, upload_specs

    manifest = receipt.get("attachment_manifest")
    evt: Dict[str, Any] = {
        "routing_token": dispatch_token, "chat_id": chat_id, "client_message_id": client_message_id,
        "attachment_uploads": upload_specs(origin, drive_root) if origin.get("attachments") else
                              [dict(row) for row in (manifest or []) if isinstance(row, dict)],
        "ts": utc_now_iso(),
        "source_ref": owner_row_source(origin)[0],
        "origin_message_snapshot": origin_snapshot(origin, drive_root),
        "source_text": raw_origin_text,
    }
    if isinstance(origin.get("client_surface"), dict) and origin.get("client_surface"):
        evt["client_surface"] = dict(origin["client_surface"])
    if action == "steer_task":
        from ouroboros.owner_relay import owner_quote

        evt.update({
            "type": "steer_task",
            "target_task_id": str(chosen.get("task_id") or ""),
            # His click names this live work for his own retained message: the words
            # travel as a verified quote with his force (D7); the note is only his comment.
            "message": picker_note,
            "quotes": [owner_quote(drive_root, origin)],
            # The owner clicked: an owner turn by construction. The option list
            # was host-built for this exact message's lane, and the room veto
            # reads that lane from the origin chat (Main sees every root).
            "issuer": {"kind": "owner_turn"},
        })
    else:
        evt.update({
            "type": "promote_chat_to_task",
            "task_id": derived_task_id,
            "objective": f"{raw_origin_text}\n\n{picker_note}" if picker_note else raw_origin_text,
            "project_id": str(chosen.get("project_id") or ""),
            # Label-only provenance for the receipt: the picker click IS a
            # routing decision on a refused message, so it wears the
            # route_to_project receipt label regardless of source chat.
            "routed_from_main": True,
            # The owner's click, not a model turn, issued this promote: a
            # refusal is told by the handler's typed System row (no narrator).
            "host_initiated": True,
            **({"reasoning_effort": card_effort} if card_effort else {}),
        })

    def _reopen_refusal() -> None:
        """Restore the actionable refusal as the LATEST row (same token)."""
        try:
            # Token-bound: if a NEWER routing attempt re-minted the card while
            # our dispatch settled, its authority wins and this reopen is a
            # no-op (the allowed set covers only our claim and our dispatch
            # receipt, never a foreign token).
            append_chat_annotation(
                drive_root, client_message_id, source_chat_id=chat_id, source_row_address=address, action="route_decision",
                status="needs_manual_target", routing_token=token,
                options=options,
                attachment_manifest=receipt.get("attachment_manifest"),
                reasoning_effort=card_effort,
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
            drive_root, client_message_id, source_chat_id=chat_id, source_row_address=address, action="route_decision",
            status="dispatch_pending", routing_token=token,
            detail=f"request:{request_id}", options=options,
            reason=f"claimed_option:{int(option_index)}",
            attachment_manifest=receipt.get("attachment_manifest"),
            reasoning_effort=card_effort,
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

        if action == "new_task_in_project":
            from ouroboros.promotion_intent import prepare_promotion_intent

            prepare_promotion_intent(drive_root, evt)
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
        # What the admitted root's own row says it starts on, never the card's echo;
        # an unreadable row reports nothing rather than the request.
        admitted_effort = ""
        if action != "steer_task" and card_effort:
            from ouroboros.task_results import load_task_result

            try:
                admitted_effort = str((load_task_result(drive_root, derived_task_id) or {}).get("reasoning_effort") or "")
            except Exception:
                log.debug("admitted effort read failed for %s", derived_task_id, exc_info=True)
        try:
            # The closing row IS the presentation receipt after hydration, so it wears the real routing action + label
            # ("Steered task · X"), plus the request id the replay path reads back, and where the dispatched act's own
            # receipt says the words landed (#968): a reload names the room its live frame did.
            try:
                landed = chat_annotation_receipt(drive_root, client_message_id, dispatch_token)
            except OSError:  # the closing row is still written, without the destination, as before
                landed = {}
            append_chat_annotation(
                drive_root, client_message_id, source_chat_id=chat_id, source_row_address=address,
                action="steer_task" if action == "steer_task" else "promote_chat_to_task",
                target=str(chosen.get("task_id") or chosen.get("project_id") or ""),
                target_label=str(chosen.get("label") or chosen.get("title") or chosen.get("project_name") or ""),
                status=expected, routing_token=token,
                detail=f"request:{request_id}",
                reason=f"answered_option:{int(option_index)}",
                reasoning_effort=admitted_effort,
                **{key: landed[key] for key in ROUTING_DESTINATION_FIELDS if key in landed},
            )
        except Exception:
            log.debug("closing annotation append failed", exc_info=True)
        payload: Dict[str, Any] = {"ok": True, "decision_id": decision_id,
                                   "state": "answered", "dispatched": expected,
                                   "answered_index": int(option_index)}
        if action != "steer_task":
            payload["task_id"] = derived_task_id
        if admitted_effort:
            payload["reasoning_effort"] = admitted_effort
        return 200, payload
    if outcome_status in {"rejected", "needs_manual_target"}:
        # The handler's rejection receipt (under the DISPATCH token) is now
        # the latest row — re-assert the refusal under the ORIGINAL token so
        # the card the UI re-opens still validates and replays cleanly.
        _reopen_refusal()
        reason = str(outcome.get("reason") or outcome_status)
        # The same act the receipt wore, so the sentence carries its prefix.
        receipt_action = (
            "steer_task" if action == "steer_task"
            else ("route_to_project" if evt.get("routed_from_main") else "promote_chat_to_task")
        )
        return 409, {"ok": False, "error": "dispatch_rejected",
                     "decision_id": decision_id, "state": "open",
                     "reason": reason,
                     # The owner-facing sentence from the host's own table (the
                     # toast shows it instead of the raw code).
                     "cause": routing_refusal_cause(receipt_action, "needs_manual_target", reason, None)}
    # Unconfirmed: honestly retriable — the derived identities make a replay
    # of the SAME request byte-identical, so the supervisor dedupes it.
    return 503, {"ok": False, "error": "dispatch_unconfirmed",
                 "decision_id": decision_id,
                 "reason": str(outcome.get("reason") or "confirmation_timeout")}


__all__ = ["handle_routing_decision", "parse_routing_decision_id"]
