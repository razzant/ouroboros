"""Letters from a linked Ouroboros (``/api/letters*``) and my human's introduction doors (``/api/peer-links/*``).

``NetworkAuthGate.classify_caller`` admits a valid link key, over a direct connection from this computer, to the
letter routes only and stamps the caller (``scope["ouroboros_caller"]``); these handlers read that stamp, never a
header of their own, so a request without it is ``link_unknown``. Every letter route first checks the link as it
stands now (``_link_refusal``: the protocol, both install ids, the link's place); only a new letter or answer then
meets my human's choices (``_admission``: the confirmed code, a pause, ``mine``, ``may_start``), so a paused link
still reads, repeats and cancels what it already sent. A letter id names one outcome on its link, a new work (its
root, ``letter_task_id``) or one answer (its receipt, ``peer_links.reply_receipt``), checked and recorded under
that id's lock before anything of it is published. A record that cannot be read now is a 503, never an absent
one. Every refused letter or introduction leaves a row in ``logs/events.jsonl`` before any model call and
creates no work; a refused read leaves none. An accepted letter is admitted through the one task admission core
(``gateway.tasks._create_task_from_body``) with host facts the caller cannot set: the room of the link's place,
the agent author record, the letter's identity and the lesser of its ceiling and this installation's per-work
limit. A cancel asks the Stop's own owner (``gateway.tasks._record_stop``) as the link, never as my human. An
ordinary message (``kind: message``) takes the Host's one Presence intake instead, a reply to my own message as
context only, never a wake; this host is that conversation's connector (the message section below).
"""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import pathlib
import threading
from typing import Any, Iterator

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from ouroboros.gateway._helpers import request_drive_root, request_json_or, run_sync_to_completion
from ouroboros.gateway.peer_contracts import (LETTER_ID_PATTERN, REFUSAL_STATUS, LetterRefused, parse_letter,
                                              refusal_body)
# The cancel source of a linked Ouroboros (``supervisor.cancel_publication`` names it; never my human's Stop).
from supervisor.cancel_publication import PEER_CANCEL_SOURCE

log = logging.getLogger(__name__)

_NO_STORE = {"Cache-Control": "no-store"}
_LETTER_LOCKS = tuple(threading.Lock() for _ in range(64))  # a fixed set: nothing grows with the letters


def _lock_index(key: str) -> int:
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % len(_LETTER_LOCKS)


def _letter_lock(key: str) -> threading.Lock:
    """The lock of one letter id or one question, chosen by the key's hash: keys sharing it only wait for each other."""
    return _LETTER_LOCKS[_lock_index(key)]


@contextlib.contextmanager
def _holding(*keys: str) -> Iterator[None]:
    """The locks of ``keys``, each once, in the pool's order: an answer holds its letter id's and its question's, a
    new letter its id's, and no two requests can wait for each other in a circle."""
    with contextlib.ExitStack() as stack:
        for index in sorted({_lock_index(key) for key in keys}):
            stack.enter_context(_LETTER_LOCKS[index])
        yield


def _id_key(place_id: str, letter_id: str) -> str:
    return f"letter:{place_id}:{letter_id}"


def _not_found(letter_id: str) -> JSONResponse:
    return JSONResponse({**refusal_body("letter_not_found"), "letter_id": letter_id}, status_code=404,
                        headers=_NO_STORE)


def _refuse(root: Any, code: str, message: str = "", *, audited: bool = True, **facts: Any) -> JSONResponse:
    if audited:
        from ouroboros.peer_letters import audit

        audit(root, "peer_letter_refused", refusal_code=code, **facts)
    return JSONResponse(refusal_body(code, message), status_code=REFUSAL_STATUS.get(code, 400), headers=_NO_STORE)


def _peer(request: Request) -> dict | None:
    caller = request.scope.get("ouroboros_caller")
    return caller if isinstance(caller, dict) and caller.get("kind") == "peer" else None


def _known_budget_exhausted() -> bool:
    """A known exhausted global budget refuses; an unreadable ledger is unknown, left to the queue's own admission."""
    from supervisor.state import budget_remaining

    try:
        return budget_remaining({}, strict=True) <= 0
    except Exception:
        return False


def _places(root: Any) -> Any:
    from ouroboros.places import PlaceError, place_records

    try:
        return place_records(root)
    except PlaceError as exc:
        raise LetterRefused("store_unavailable", f"the place store cannot be read now ({exc.code})") from exc


def _link_refusal(request: Request, caller: dict) -> tuple[str, str] | None:
    """What every letter route checks first, the same for a letter, an answer, a read and a cancel: the protocol,
    both installations' ids and the link's place, as they stand now."""
    from ouroboros.install_identity import recorded_id

    root = request_drive_root(request)
    if request.headers.get("x-ouroboros-protocol", "").strip() != "1":
        return "protocol_unsupported", "X-Ouroboros-Protocol must be 1"
    if request.headers.get("x-ouroboros-install", "").strip() != caller["peer_install_id"]:
        return "install_mismatch", "the sender's install id differs from the one this link was made with"
    if recorded_id(root) != caller["own_install_id"]:
        return "install_mismatch", "this installation's data directory moved since the link was made"
    if _places(root).places.get(caller["place_id"]) is None:
        return "link_unknown", "this link's place is no longer connected"
    return None


def _admission(root: Any, caller: dict, *, new_work: bool) -> tuple[tuple[str, str] | None, Any]:
    """What a new letter (``new_work``) or a new answer meets after the link checks, and the link's place: the one
    reading the place's card states (``peer_links.effective_permissions``, a Panic read for new work only). A repeat
    of what this link already sent never meets it again: it reads, as GET does."""
    from ouroboros.peer_links import DOOR_REFUSALS, effective_permissions
    from ouroboros.place_projection import switch_state

    records = _places(root)
    place = records.places.get(caller["place_id"])
    if place is None:
        return ("link_unknown", "this link's place is no longer connected"), None
    effect = effective_permissions(records.peer(caller["place_id"]), state=place.state,
                                   link={"code_check": "confirmed" if caller.get("code_confirmed") else "pending"},
                                   switch=switch_state(root) if new_work else "")
    refused = not effect["messages"] or (new_work and not effect["assignments"])
    return (DOOR_REFUSALS[effect["reason"]] if refused else None), place


def _room(place: Any) -> int:
    """The room of the link's place: the chat id every reader of a place conversation derives from it
    (``presence_bindings.conversation_key`` without a thread), so the letter's work lives where its place does."""
    from ouroboros.presence_bindings import conversation_key
    from ouroboros.presence_runner import presence_chat_id

    return presence_chat_id(conversation_key(place.provider, place.account_id, place.conversation_id, ""))


def _stored(root: Any, task_id: str) -> dict | None:
    """The stored row of ``task_id`` or None when there is none; a busy or damaged store, or a row its admission
    refuses, is ``store_unavailable``, never an absent letter."""
    from ouroboros.task_results import load_task_result

    try:
        return load_task_result(root, task_id, strict=True)
    except Exception as exc:  # noqa: BLE001 - every failed read is "not known now"
        raise LetterRefused("store_unavailable",
                            f"the letter's record cannot be read now ({type(exc).__name__})") from exc


def _effective(root: Any, row: dict) -> dict:
    from ouroboros.task_status import effective_task_result

    try:
        return effective_task_result(pathlib.Path(root), row, materialize_artifacts=False)
    except Exception as exc:  # noqa: BLE001
        raise LetterRefused("store_unavailable",
                            f"the letter's state cannot be read now ({type(exc).__name__})") from exc


def _ours(row: Any, caller: dict) -> bool:
    letter = (row.get("metadata") or {}).get("peer_letter") if isinstance(row, dict) else None
    return isinstance(letter, dict) and letter.get("link_id") == caller["link_id"]


def _letter_root(root: Any, caller: dict, letter_id: str) -> tuple[str, dict] | None:
    """This link's own letter ``letter_id`` (the root it became, never an answer) and its effective row."""
    from ouroboros.peer_letters import letter_task_id

    task_id = letter_task_id(caller["place_id"], letter_id)
    row = _stored(root, task_id)
    return (task_id, _effective(root, row)) if row is not None and _ours(row, caller) else None


def _letter_row(root: Any, caller: dict, letter_id: str) -> tuple[str, dict] | None:
    """What ``letter_id`` names on this link, for a read, a repeat and a cancel alike: its own root, or the root
    its delivered answer of that id went to (a reservation names none); None when this link sent nothing of it."""
    from ouroboros.peer_links import reply_receipt

    found = _letter_root(root, caller, letter_id)
    if found is not None:
        return found
    receipt = reply_receipt(root, caller["link_id"], letter_id)
    if not receipt or receipt.get("delivered") is not True:
        return None
    target = str(receipt.get("target_task_id") or "")
    row = _stored(root, target) if target else None
    return (target, _effective(root, row)) if row is not None and _ours(row, caller) else None


def _status_view(letter_id: str, row: dict, *, root: Any, link_id: str) -> dict:
    from ouroboros.peer_letters import disclosed_letter_state

    return {"letter_id": letter_id, **disclosed_letter_state(root, row, link_id=link_id)}


def _reply_write_admission(root: Any, caller: dict) -> tuple[str, str] | None:
    """Current link and permission check on the boundary of a NEW answer mailbox write."""
    from ouroboros.peer_links import link_of_place

    found = link_of_place(root, caller["place_id"])
    if found is None or found[0] != caller["link_id"] or found[1].get("revoked_at"):
        return ("link_unknown", "this link is no longer active")
    live_caller = {**caller, "code_confirmed": bool(found[1].get("code_confirmed_at"))}
    refusal, _place = _admission(root, live_caller, new_work=False)
    return refusal


def _answer_receipt(letter: dict, msg_id: str) -> dict:
    """``peer_contracts.LetterAnswerReceipt``: POST's own answer for an answer letter, never GET's ``LetterStatus``."""
    return {"letter_id": letter["letter_id"], "in_reply_to": dict(letter["in_reply_to"]), "delivered": True,
            "msg_id": msg_id}


def _unconfirmed(why: str, *, root: Any = None, **facts: Any) -> JSONResponse:
    """Not known now (503), naming the letter id its facts carry, audited with them when ``root`` is given: the
    sender keeps its letter and may send the same bytes again."""
    if root is not None:
        from ouroboros.peer_letters import audit

        audit(root, "peer_letter_unconfirmed", **facts)
    letter_id = str(facts.get("letter_id") or "")
    return JSONResponse({"error": "unconfirmed", **({"letter_id": letter_id} if letter_id else {}),
                         "detail": str(why)[:300]}, status_code=503, headers=_NO_STORE)


def _accept_new(request: Request, caller: dict, letter: dict, sha: str, facts: dict) -> JSONResponse:
    from ouroboros.dialogue_provenance import LINK_KEY, recorded_author
    from ouroboros.gateway.tasks import _create_task_from_body
    from ouroboros.peer_letters import audit, letter_ceiling, letter_task_id
    from ouroboros.peer_links import message_receipt, reply_receipt

    root, letter_id, place_id = request_drive_root(request), letter["letter_id"], caller["place_id"]
    task_id = letter_task_id(place_id, letter_id)
    with _holding(_id_key(place_id, letter_id)):
        existing = _stored(root, task_id)
        if existing is not None:
            recorded = ((existing.get("metadata") or {}).get("peer_letter") or {}).get("content_sha256")
            if recorded != sha:
                return _refuse(root, "letter_id_reused", "this letter id already carried another letter", **facts)
            if not _ours(existing, caller):  # its record names no link or another one: never this letter's work
                return _refuse(root, "letter_id_reused", "this letter id names work this link did not start", **facts)
            return JSONResponse(_status_view(letter_id, _effective(root, existing), root=root,
                                             link_id=caller["link_id"]), headers=_NO_STORE)
        if reply_receipt(root, caller["link_id"], letter_id) is not None:
            return _refuse(root, "letter_id_reused", "this letter id already carried an answer", **facts)
        if message_receipt(root, caller["link_id"], letter_id) is not None:
            return _refuse(root, "letter_id_reused", "this letter id already carried a message", **facts)
        refusal, place = _admission(root, caller, new_work=True)
        if refusal is not None:
            return _refuse(root, *refusal, **facts)
        if _known_budget_exhausted():
            return _refuse(root, "no_budget", "this installation's budget is exhausted", **facts)
        ceiling = letter_ceiling(letter.get("cost_ceiling_usd"))
        metadata = {
            "peer_letter": {"place_id": place_id, "link_id": caller["link_id"], "letter_id": letter_id,
                            "content_sha256": sha, "sender_label": caller["peer_label"],
                            "sender_install_id": caller["peer_install_id"], "chain": letter.get("chain") or [],
                            "protocol": 1},
            "letter_author": recorded_author(caller["peer_label"], basis=LINK_KEY, key_id=caller["link_id"]),
            **({"root_cost_ceiling_usd": ceiling} if ceiling is not None else {}),
        }
        body = {"description": letter["text"], "task_id": task_id, "source": "peer_letter",
                "actor_id": f"peer:{caller['link_id']}", **({"title": letter["title"]} if letter.get("title") else {})}
        response = _create_task_from_body(request, body, host_facts={"chat_id": _room(place), "metadata": metadata})
    try:
        answer = json.loads(bytes(response.body).decode("utf-8"))
    except (ValueError, UnicodeError):
        answer = {}
    if response.status_code == 200 and answer.get("task_id") == task_id:
        audit(root, "peer_letter_accepted", task_id=task_id, root_cost_ceiling_usd=ceiling, **facts)
        return JSONResponse({"letter_id": letter_id, "state": "accepted_running", "phase": "queued"},
                            headers=_NO_STORE)
    why = str(answer.get("error") or "")[:300]
    return _unconfirmed(why, root=root, task_id=task_id, status=response.status_code, detail=why, **facts)


def _deliver_reply(request: Request, caller: dict, letter: dict, sha: str, task_id: str, row: dict, msg_id: str,
                   facts: dict, *, reserved: bool = False) -> JSONResponse:
    """Put an answer's words in the asking root's mailbox as one place event (``msg_id``); mark the question answered,
    settle its receipt and answer ``LetterAnswerReceipt`` only once that row is proven (appended now, or there from a
    lost settlement). A root that ended first takes nothing: these bytes' reservation without its row is not known now,
    any other answer is closed. Completing an undelivered reservation re-checks current link admission and budget."""
    from ouroboros.dialogue_provenance import LINK_KEY, recorded_author
    from ouroboros.owner_mailbox import place_event_retained, write_task_message
    from ouroboros.peer_letters import SETTLED_LETTER_STATUSES, answer, audit, confirm_answer, quoted_letter
    from ouroboros.peer_links import PeerLinkError, record_reply
    from ouroboros.task_pacing import recorded_producer_ceiling

    root, letter_id = request_drive_root(request), letter["letter_id"]
    question_id = letter["in_reply_to"]["question_id"]
    task_drive = pathlib.Path(str(row.get("drive_root") or root))
    if not place_event_retained(task_drive, task_id, msg_id):
        if not _ours(row, caller):
            return _refuse(root, "letter_id_reused", "this letter id names work this link did not start", **facts)
        if reserved:
            refusal = _reply_write_admission(root, caller)
            if refusal is not None:
                return _refuse(root, *refusal, **facts)
            ceiling = recorded_producer_ceiling(row)
            spent = row.get("accounted_upper_bound_usd_with_children")
            if (ceiling is not None and isinstance(spent, (int, float))
                    and not isinstance(spent, bool) and float(spent) >= ceiling):
                return _refuse(root, "no_budget", "that letter's cost ceiling is spent", **facts)
        if str(row.get("status") or "") in SETTLED_LETTER_STATUSES:
            questions = row.get("peer_questions") if isinstance(row.get("peer_questions"), dict) else {}
            if reserved or (questions.get(question_id) or {}).get("reply_sha256") == sha:
                return _unconfirmed("the asking letter ended before this answer's delivery was confirmed", root=root,
                                    task_id=task_id, **facts)
            return _refuse(root, "question_closed", "that letter ended before the answer was delivered", **facts)
        try:
            marked = answer(root, task_id, question_id, reply_letter_id=letter_id, reply_sha256=sha, confirmed=False)
        except Exception as exc:  # noqa: BLE001 - a busy or unreadable store is not known now, never a closed question
            raise LetterRefused("store_unavailable",
                                f"the letter's record cannot be updated now ({type(exc).__name__})") from exc
        if marked == "unknown":
            return _refuse(root, "question_not_found", "that letter asked no such question", **facts)
        if marked == "settled_unconfirmed":
            return _unconfirmed("the asking letter ended before this answer's delivery was confirmed", root=root,
                                task_id=task_id, **facts)
        if marked not in {"reserved", "answered", "repeat"}:
            return _refuse(root, "question_closed", "that question was answered or closed meanwhile", **facts)
        author = recorded_author(caller["peer_label"], basis=LINK_KEY, key_id=caller["link_id"])
        text = (f"Answer from {caller['peer_label']} (my human's Ouroboros, link {caller['link_id']}) to the question "
                f"{question_id} I asked it; every line quoted with \"> \" is its text, not my human's and not the "
                f"host's:\n\n{quoted_letter(letter['text'])}")
        admit_verdicts: list[bool] = []
        admit_refusal: list[tuple[str, str]] = []
        admit_unconfirmed: list[str] = []

        def admit_new_write() -> bool:
            """Re-verify current link admission, task identity, settlement and budget right inside task_mail_lock
            immediately before a NEW mailbox append (never called if the row is already retained)."""
            try:
                live_refusal = _reply_write_admission(root, caller)
                if live_refusal is not None:
                    admit_refusal.append(live_refusal)
                    admit_verdicts.append(False)
                    return False
                live_row = _stored(root, task_id)
                if live_row is None or not _ours(live_row, caller):
                    admit_refusal.append(("letter_id_reused", "this letter id names work this link did not start"))
                    admit_verdicts.append(False)
                    return False
                if str(live_row.get("status") or "") in SETTLED_LETTER_STATUSES:
                    admit_unconfirmed.append("the asking letter ended before this answer's delivery was confirmed")
                    admit_verdicts.append(False)
                    return False
                live_ceiling = recorded_producer_ceiling(live_row)
                live_spent = live_row.get("accounted_upper_bound_usd_with_children")
                if (live_ceiling is not None and isinstance(live_spent, (int, float))
                        and not isinstance(live_spent, bool) and float(live_spent) >= live_ceiling):
                    admit_refusal.append(("no_budget", "that letter's cost ceiling is spent"))
                    admit_verdicts.append(False)
                    return False
                admit_verdicts.append(True)
                return True
            except (LetterRefused, PeerLinkError) as exc:
                admit_unconfirmed.append(str(exc))
                admit_verdicts.append(False)
                return False
            except Exception as exc:  # noqa: BLE001 - unreadable check cannot prove absence of effect
                admit_unconfirmed.append(f"the answer's admission could not be verified now ({type(exc).__name__})")
                admit_verdicts.append(False)
                return False

        if not write_task_message(  # a place event of one msg_id: delivering it again changes nothing
                task_drive, text, task_id, source_task_id="", provenance="place_event", msg_id=msg_id,
                sender_origin={"place_id": caller["place_id"], "link_id": caller["link_id"]},
                place_arrival={"wake": True, "author": author, "peer_reply": {"letter_id": letter_id,
                                                                             "question_id": question_id}},
                admit=admit_new_write):
            if admit_verdicts == [False] and admit_refusal:
                return _refuse(root, *admit_refusal[0], **facts)
            return _unconfirmed(
                admit_unconfirmed[0] if admit_unconfirmed else "the answer's delivery is not confirmed",
                root=root, task_id=task_id, **facts,
            )
    try:
        confirmed = confirm_answer(root, task_id, question_id, reply_letter_id=letter_id, reply_sha256=sha,
                                   msg_id=msg_id)
    except Exception as exc:  # noqa: BLE001 - mailbox row landed; retry settles question state and receipt
        return _unconfirmed(f"the answer's confirmation could not be recorded now ({type(exc).__name__})",
                            root=root, task_id=task_id, **facts)
    if confirmed not in {"answered", "repeat"}:
        return _unconfirmed("the answer's question state could not be confirmed now", root=root, task_id=task_id, **facts)
    try:
        record_reply(root, caller["link_id"], letter_id, target_task_id=task_id, question_id=question_id,
                     content_sha256=sha, delivered=True, msg_id=msg_id)
    except PeerLinkError:  # delivered, its receipt not settled: the same bytes again settle it from the mailbox row
        return _unconfirmed("the answer's receipt could not be settled now", root=root, task_id=task_id, **facts)
    audit(root, "peer_letter_reply_accepted", task_id=task_id, question_id=question_id, **facts)
    return JSONResponse(_answer_receipt(letter, msg_id), headers=_NO_STORE)


def _accept_reply(request: Request, caller: dict, letter: dict, sha: str, facts: dict | None = None) -> JSONResponse:
    """An answer to this side's question: the same root wakes with what is left of its ceiling, no new work. Under
    its letter id's lock and its question's it reserves the id (``peer_links.record_reply``) before anything lands; a
    repeat of the id and bytes gets the same receipt with nothing written again, or finishes an unproven delivery."""
    from ouroboros.peer_letters import SETTLED_LETTER_STATUSES, letter_task_id
    from ouroboros.peer_links import message_receipt, record_reply, reply_receipt
    from ouroboros.task_pacing import recorded_producer_ceiling

    root, letter_id, place_id = request_drive_root(request), letter["letter_id"], caller["place_id"]
    target, question_id = letter["in_reply_to"]["letter_id"], letter["in_reply_to"]["question_id"]
    msg_id = f"peer-reply-{letter_task_id(place_id, letter_id)}"
    facts = facts or {"link_id": caller["link_id"], "place_id": place_id, "letter_id": letter_id, "in_reply_to": target}
    with _holding(_id_key(place_id, letter_id), f"question:{place_id}:{target}:{question_id}"):
        receipt = reply_receipt(root, caller["link_id"], letter_id)
        if receipt is not None:
            if receipt.get("content_sha256") != sha:
                return _refuse(root, "letter_id_reused", "this letter id already carried another answer", **facts)
            if receipt.get("delivered") is True:  # proven already: its receipt again, nothing written or spent
                return JSONResponse(_answer_receipt(letter, str(receipt.get("msg_id") or msg_id)), headers=_NO_STORE)
            found = _letter_root(root, caller, target)  # the asking root may be gone (retention): no state to give
            if found is None:
                return _not_found(target)
            if (str(receipt.get("target_task_id") or "") != found[0]
                    or str(receipt.get("question_id") or "") != question_id):
                return _refuse(root, "letter_id_reused", "this letter id already carried another answer", **facts)
            reply_msg_id = str(receipt.get("msg_id") or msg_id)
            return _deliver_reply(request, caller, letter, sha, found[0], found[1], reply_msg_id, facts, reserved=True)
        if _stored(root, letter_task_id(place_id, letter_id)) is not None:
            return _refuse(root, "letter_id_reused", "this letter id already carried a letter", **facts)
        if message_receipt(root, caller["link_id"], letter_id) is not None:
            return _refuse(root, "letter_id_reused", "this letter id already carried a message", **facts)
        refusal = _reply_write_admission(root, caller)
        if refusal is not None:
            return _refuse(root, *refusal, **facts)
        found = _letter_root(root, caller, target)
        if found is None:
            return _refuse(root, "question_not_found", "no letter of this link has that id", **facts)
        task_id, row = found
        questions = row.get("peer_questions") if isinstance(row.get("peer_questions"), dict) else {}
        question = questions.get(question_id)
        if not isinstance(question, dict):
            return _refuse(root, "question_not_found", "that letter asked no such question", **facts)
        if question.get("state") != "open" or str(row.get("status") or "") in SETTLED_LETTER_STATUSES:
            return _refuse(root, "question_closed", "that question is no longer waiting for an answer", **facts)
        ceiling = recorded_producer_ceiling(row)
        spent = row.get("accounted_upper_bound_usd_with_children")
        if ceiling is not None and isinstance(spent, (int, float)) and float(spent) >= ceiling:
            return _refuse(root, "no_budget", "that letter's cost ceiling is spent", **facts)
        record_reply(root, caller["link_id"], letter_id, target_task_id=task_id, question_id=question_id,
                     content_sha256=sha, msg_id=msg_id)
        return _deliver_reply(request, caller, letter, sha, task_id, row, msg_id, facts)


# --- Ordinary messages: a linked Ouroboros's words in its place's conversation, never a work assignment ------------
# One logical message is one letter id on its link (``peer_replies.message_letter_id``): the same words of the same
# producer travel again under it and ``peer_links.message_receipt`` answers that repeat without a second row. An
# arrival takes the Host's one intake (``host_service_places.presence_intake``, version 2) and its answer waits for
# the same typed start ``/presence/turn`` returns. Outgoing sends and automatic replies live in
# ``ouroboros.peer_replies`` (decoupled from this inbound HTTP gateway).

_START_WAIT_SEC = 90.0  # how long an arrival's answer waits for its turn's typed start; the turn never depends on it
_RETRY_LATER = frozenset({"presence_capacity_full", "presence_rate_limited", "consciousness_allowance_exhausted",
                          "wake_import_pending"})  # the Host's 429 retries; its other retries are 503


def _peer_message_event_payload(place: Any, caller: dict, letter: dict) -> dict[str, Any]:
    """One live Presence event shape for a standalone message from a verified linked Ouroboros."""
    from ouroboros.dialogue_provenance import MACHINE_ORIGIN
    from ouroboros.presence_bindings import conversation_key

    key = conversation_key(place.provider, place.account_id, place.conversation_id, "")
    return {
        "source_event_id": f"peer-msg:{letter['letter_id']}",
        "provider": place.provider,
        "account_id": place.account_id,
        "conversation_id": place.conversation_id,
        "thread_id": "",
        "conversation_key": key,
        "actor": {
            "platform_actor_id": f"peer:{caller['link_id']}",
            "display_name": caller["peer_label"],
            "authenticated": True,
            "is_bot": True,
            "origin": MACHINE_ORIGIN,
            "kind": "ouroboros",
            "install_id": caller["peer_install_id"],
            "link_id": caller["link_id"],
        },
        "conversation": {
            "title": caller["peer_label"],
            "type": "direct",
            "link_id": caller["link_id"],
            "peer_install_id": caller["peer_install_id"],
        },
        "message": {
            "message_id": letter["letter_id"],
            "peer_message": True,
            **({"reply_to": letter["reply_to"]} if letter.get("reply_to") else {}),
        },
        "text": letter["text"],
        "facts": ["direct_message"],
    }


def _received(letter_id: str) -> JSONResponse:
    return JSONResponse({"letter_id": letter_id, "kind": "message", "state": "received"}, headers=_NO_STORE)


def _hear_reply(root: pathlib.Path, admission: Any, captured: dict, event: Any) -> None:
    """A reply to a message of mine is context: heard, or background to the conversation's live author, never a
    wake or a new focus, so two Ouroboroses never answer each other's automatic replies in a loop."""
    from ouroboros.place_intake import ROUTE_TO_MIND, decide_attention, offer_arrival, settle_heard
    from ouroboros.presence_runner import presence_event_identity, presence_turn_replay, presence_turn_task_id

    turn_id = presence_turn_task_id(admission.binding_id, event.source_event_id)
    identity = presence_event_identity(admission.binding_id, event)
    if presence_turn_replay(root, turn_id, event.conversation_key, identity, event.continuation_version) is not None:
        return
    decision, allowance = decide_attention(root, admission, event)
    if decision == "silent":
        settle_heard(root, captured, event, allowance["reason"])
        return
    handed, recipient = offer_arrival(root, captured, event, wake=False)
    settle_heard(root, captured, event, handed if handed and handed != ROUTE_TO_MIND else "peer_reply",
                 recipient=recipient)


def _typed_answer(root: pathlib.Path, exc: Exception, facts: dict) -> JSONResponse | None:
    """The intake's typed outcome as ``/presence/turn`` gives it a connector (``presence_disposition``): None when the
    queue keeps the turn (saved work: received); a retry, 429 for capacity, rate and allowance and 503 otherwise, asks
    for the same letter again; a rejection is the sender's definite refusal. A failure naming no code is unconfirmed."""
    from ouroboros.gateway.host_service_places import presence_disposition
    from ouroboros.peer_letters import audit

    code = str(getattr(exc, "code", "") or "")
    if not code:
        return _unconfirmed(str(exc), root=root, **facts)
    disposition = "retry" if code in _RETRY_LATER else presence_disposition(exc)
    if disposition == "retained":
        return None
    if code == "presence_event_identity_conflict":
        return _refuse(root, "letter_id_reused", "this letter id already carried another message", **facts)
    retry_after = (getattr(exc, "facts", None) or {}).get("retry_after", getattr(exc, "retry_after", None))
    known = {key: value for key, value in (("retry_after", retry_after), ("turn_ref", getattr(exc, "turn_ref", "")),
                                           ("custody", getattr(exc, "custody", ""))) if value not in (None, "")}
    retry = disposition == "retry"
    audit(root, "peer_message_retry" if retry else "peer_letter_refused", refusal_code=code, **facts)
    return JSONResponse({"error": code, "refusal_code": code, "detail": str(exc)[:300], "disposition": disposition,
                         **known}, status_code=(429 if code in _RETRY_LATER else 503) if retry else 409,
                        headers=_NO_STORE)


class _Unstarted(Exception):
    """A turn whose start nobody confirmed: the same letter again joins or starts it."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code, self.disposition = code, "retry"


_UNWRITTEN = object()  # no owed record of this dispatch is known to the waiter
_OWED_SEEN: dict[str, Any] = {}  # a live dispatch's owed record as it wrote it, until that turn settles


def _owed_key(root: pathlib.Path, link_id: str, turn: str) -> str:
    return f"{pathlib.Path(root).resolve()}\0{link_id}\0{turn}"


class _PendingIntake:
    """A started or joined Presence turn whose exact ``initial`` Future is awaited on the async loop; ``owed_seen`` the
    owed record this intake's own dispatch wrote, as it wrote it (``peer_links.owed_version``)."""

    __slots__ = ("letter_id", "root", "link_id", "conversation_key", "facts", "initial", "turn", "identity",
                 "owed_seen")

    def __init__(self, *, letter_id: str, root: pathlib.Path, link_id: str, conversation_key: str, facts: dict,
                 initial: Any, turn: str, identity: str, owed_seen: Any = _UNWRITTEN) -> None:
        self.letter_id, self.root, self.link_id = letter_id, root, link_id
        self.conversation_key, self.facts, self.initial = conversation_key, facts, initial
        self.turn, self.identity, self.owed_seen = turn, identity, owed_seen


def _is_typed_prestart_exc(exc: BaseException) -> bool:
    from ouroboros.gateway.host_service_places import PresenceIntakeRefused, presence_disposition
    from ouroboros.places import PlaceError
    from ouroboros.presence_admission import PresenceAdmissionError, WakeImportPending
    from ouroboros.presence_events import PresenceTurnError
    from ouroboros.presence_executions import PresenceTurnNotStarted

    if isinstance(exc, (PresenceTurnNotStarted, PresenceIntakeRefused, WakeImportPending)):
        return True
    return isinstance(exc, (PresenceTurnError, PresenceAdmissionError, PlaceError)) and (
        presence_disposition(exc) != "retained"
    )


def _is_typed_prestart_outcome(done: Any) -> bool:
    """Whether a settled execution Future ended with a typed pre-start refusal or pre-start heard decision
    (never an untyped failure, carrier interruption, or retained turn)."""
    if done.cancelled():
        return False
    exc = done.exception()
    if exc is None:
        res = done.result()
        att = res.get("attention") if isinstance(res, dict) else getattr(res, "attention", None)
        return isinstance(att, dict) and att.get("decision") == "heard"
    return _is_typed_prestart_exc(exc)


def _prestart_unowed(root: pathlib.Path, link_id: str, turn: str, conversation_key: str, identity: str) -> bool:
    """True only when ``turn`` (and any retry successor) is neither live nor held in queue custody, carries no
    approved speech, and either wrote no task row at all or ended with a canonical ``presence_retry_proof``."""
    from ouroboros.conversation_lease import live_author
    from ouroboros.gateway.host_service_places import host_context
    from ouroboros.peer_links import owed_record
    from ouroboros.presence_recovery import CUSTODY_HOLDS, attempt_ended, queue_custody
    from ouroboros.presence_runner import (
        _retry_reset, _retry_target, _stored_turn, _terminal_refusal, presence_turn_is_live,
    )

    ctx = host_context(root)
    if (ctx.presence_turns.find(turn) is not None or presence_turn_is_live(turn, drive_root=root)
            or live_author(conversation_key, drive_root=root) == turn):
        return False
    try:
        if (owed_record(root, link_id, turn) or {}).get("approved"):
            return False
        target = _retry_target(root, turn, identity) if identity else turn
        stored = _stored_turn(root, target, identity)
        if target != turn and (ctx.presence_turns.find(target) is not None
                               or presence_turn_is_live(target, drive_root=root)
                               or live_author(conversation_key, drive_root=root) == target):
            return False
        if queue_custody(root, target, stored or None) in CUSTODY_HOLDS:
            return False
    except Exception:  # noqa: BLE001 - unreadable custody or task state keeps any owed record
        return False
    if not stored:
        return True
    return (attempt_ended(stored, root) and _terminal_refusal(stored) == "presence_resources_unavailable"
            and _retry_reset(stored, target, identity) is not None)


def _settle_prestart_owed(root: pathlib.Path, link_id: str, turn: str, conversation_key: str,
                          identity: str, seen: tuple | None, done: Any) -> None:
    """Bound to ``PresenceTurnExecution.result`` on newly dispatched peer turns: clears ``owed`` when the turn settles
    as a clean pre-start refusal/heard outcome, including after an HTTP waiter already timed out, and only the record
    its own dispatch wrote (``seen``): another attempt's rewrite stays."""
    from ouroboros.install_identity import acting_as
    from ouroboros.peer_links import settle_owed

    if _OWED_SEEN.get(key := _owed_key(root, link_id, turn)) == seen:
        _OWED_SEEN.pop(key, None)
    try:
        if not _is_typed_prestart_outcome(done):
            return
        with acting_as(root):  # read and removed in one step: a newer attempt's record stays (``settle_owed``)
            settle_owed(root, link_id, turn, seen=seen,
                        unowed=lambda: _prestart_unowed(root, link_id, turn, conversation_key, identity))
    except Exception:  # noqa: BLE001 - settlement callbacks must never break Future resolution
        log.warning("pre-start owed cleanup for %s failed", turn, exc_info=True)


def _repeat_message_gate(root: pathlib.Path, place: Any, letter_id: str, wire: Any,
                         facts: dict) -> tuple[str, JSONResponse | _PendingIntake | Exception | None]:
    """Classify a repeated ``letter_id`` with matching bytes before admission:
    - ``("live", _PendingIntake)`` when an execution is currently live (bound to its exact ``initial`` Future);
    - ``("done", JSONResponse)`` when replay or queue custody already settles/retains the outcome;
    - ``("reconcile", None)`` when a non-retryable durable row already exists: answered ``received`` from that row,
      nothing admitted, captured or dispatched for it;
    - ``("new", pending_exc | None)`` when no started work exists yet, so current link/place permissions must pass."""
    from ouroboros.gateway.host_service_places import _queue_custody_refusal, host_context, presence_disposition
    from ouroboros.presence_events import PresenceTurnError
    from ouroboros.presence_runner import (
        _retry_reset, _retry_target, _stored_turn, _terminal_refusal,
        presence_event_identity, presence_turn_replay, presence_turn_task_id,
    )

    turn_id = presence_turn_task_id(place.place_id, wire.source_event_id)
    identity = presence_event_identity(place.place_id, wire)
    if (live := host_context(root).presence_turns.find(turn_id)) is not None:
        if identity and live.identity and identity != live.identity:
            return "done", _refuse(root, "letter_id_reused", "this letter id already carried another message", **facts)
        return "live", _PendingIntake(letter_id=letter_id, root=root, link_id=place.conversation_id,
                                      conversation_key=wire.conversation_key, facts=facts,
                                      initial=live.initial, turn=turn_id, identity=identity,
                                      owed_seen=_OWED_SEEN.get(_owed_key(root, place.conversation_id, turn_id),
                                                               _UNWRITTEN))
    try:
        if presence_turn_replay(root, turn_id, wire.conversation_key, identity, wire.continuation_version) is not None:
            return "done", _received(letter_id)
    except PresenceTurnError as exc:
        if exc.code == "presence_resources_unavailable" and presence_disposition(exc) == "retry":
            return "new", exc
        return "done", _typed_answer(root, exc, facts) or _received(letter_id)
    except Exception as exc:  # noqa: BLE001
        return "done", _typed_answer(root, exc, facts) or _received(letter_id)
    if "reply_to" in wire.message:
        return "new", None
    try:
        _queue_custody_refusal(root, turn_id, identity)
        target = _retry_target(root, turn_id, identity)
        stored = _stored_turn(root, target, identity)
    except Exception as exc:  # noqa: BLE001
        return "done", _typed_answer(root, exc, facts) or _received(letter_id)
    if not stored or (_terminal_refusal(stored) == "presence_resources_unavailable"
                      and _retry_reset(stored, target, identity) is not None):
        return "new", None
    return "reconcile", None


def _start_intake(letter_id: str, root: pathlib.Path, place: Any, admission: Any, event: Any, payload: dict,
                  captured: dict, facts: dict) -> JSONResponse | _PendingIntake | None:
    """Synchronous Presence intake step off the event loop: delegates attention to ``presence_intake``, records owed
    speech only on actual new turn dispatch, binds pre-start owed cleanup to the execution's settlement, and hands the
    exact ``initial`` Future to the async route handler so waiting never holds a worker thread."""
    from ouroboros.gateway.host_service_places import host_context, presence_intake
    from ouroboros.peer_links import PeerLinkError, owed_version, record_owed
    from ouroboros.presence_executions import PresenceTurnInterrupted, PresenceTurnNotStarted
    from ouroboros.presence_runner import presence_event_identity, presence_turn_task_id

    if "reply_to" in event.message:
        try:
            _hear_reply(root, admission, captured, event)
        except Exception as exc:  # noqa: BLE001
            return _typed_answer(root, exc, facts)
        return None
    ctx = host_context(root)
    turn = presence_turn_task_id(admission.binding_id, event.source_event_id)
    identity = presence_event_identity(admission.binding_id, event)

    written: list = []

    def bind_owed(execution: Any) -> None:
        try:
            record_owed(root, place.conversation_id, turn, place_id=place.place_id)
            seen = owed_version(root, place.conversation_id, turn)  # the record this dispatch wrote, as it wrote it
        except PeerLinkError as exc:  # not recorded now: no turn was dispatched, the same letter again starts it
            raise _Unstarted(exc.code, str(exc)) from exc
        written.append(seen)
        _OWED_SEEN[_owed_key(root, place.conversation_id, turn)] = seen  # what a joining waiter may clear, exactly
        execution.result.add_done_callback(
            lambda done: _settle_prestart_owed(root, place.conversation_id, turn,
                                               event.conversation_key, identity, seen, done)
        )

    try:
        _result, execution = presence_intake(
            ctx, place.connector, admission, event, payload, captured,
            continuing=event.continuation_version, on_dispatch=bind_owed,
        )
        if execution is None:
            return None
        return _PendingIntake(letter_id=letter_id, root=root, link_id=place.conversation_id,
                              conversation_key=event.conversation_key, facts=facts, initial=execution.initial,
                              turn=turn, identity=identity, owed_seen=written[0] if written else _UNWRITTEN)
    except (PresenceTurnNotStarted, PresenceTurnInterrupted) as exc:
        return _typed_answer(root, _Unstarted("presence_turn_not_started" if isinstance(exc, PresenceTurnNotStarted)
                                              else "presence_turn_interrupted", str(exc)), facts)
    except Exception as exc:  # noqa: BLE001 - every typed outcome is the Host's own; an untyped one is unconfirmed
        return _typed_answer(root, exc, facts)


def _settle_failed_intake(pending: _PendingIntake, exc: Exception, *, timed_out: bool = False) -> JSONResponse | None:
    """A waited turn's typed failure as its answer. A clean pre-start one first clears the owed record that turn's
    dispatch wrote, only as it wrote it (its settlement callback does the same, whichever comes first); a waiter that
    knows no such record, or whose wait ran out, touches none."""
    from ouroboros.peer_links import PeerLinkError, settle_owed
    from ouroboros.presence_executions import PresenceTurnInterrupted, PresenceTurnNotStarted

    if not timed_out and pending.owed_seen is not _UNWRITTEN and _is_typed_prestart_exc(exc):
        with contextlib.suppress(OSError, PeerLinkError):  # a record not removed now stays for its callback
            settle_owed(pending.root, pending.link_id, pending.turn, seen=pending.owed_seen,
                        unowed=lambda: _prestart_unowed(pending.root, pending.link_id, pending.turn,
                                                        pending.conversation_key, pending.identity))
    if isinstance(exc, (PresenceTurnNotStarted, PresenceTurnInterrupted)):
        exc = _Unstarted("presence_turn_not_started" if isinstance(exc, PresenceTurnNotStarted)
                         else "presence_turn_interrupted", str(exc))
    return _typed_answer(pending.root, exc, pending.facts)


async def _await_intake(pending: _PendingIntake) -> JSONResponse:
    """Await ``pending.initial`` on the event loop via ``asyncio.wrap_future`` shielded from one reader's timeout,
    so neither a worker thread is held nor the shared Future is cancelled when one HTTP caller times out."""
    import concurrent.futures

    wrapped = asyncio.wrap_future(pending.initial)
    wrapped.add_done_callback(lambda done: None if done.cancelled() else done.exception())
    try:
        await asyncio.wait_for(asyncio.shield(wrapped), timeout=_START_WAIT_SEC)
    except (asyncio.TimeoutError, concurrent.futures.TimeoutError):
        unconfirmed = _Unstarted("presence_start_unconfirmed",
                                 "the turn's start is not confirmed yet; send the same letter again")
        answer = await run_sync_to_completion(_settle_failed_intake, pending, unconfirmed, timed_out=True)
        return answer or _received(pending.letter_id)
    except Exception as exc:  # noqa: BLE001 - typed carrier or turn outcomes map through ``_settle_failed_intake``
        answer = await run_sync_to_completion(_settle_failed_intake, pending, exc)
        return answer or _received(pending.letter_id)
    return _received(pending.letter_id)


def _accept_message(request: Request, caller: dict, letter: dict, sha: str,
                    facts: dict) -> JSONResponse | _PendingIntake:
    """An ordinary message of the link: saved in the link's place conversation, then the Host's one intake (Off hears
    it, On follows the place's attention), never a work assignment; ``may_start`` is not asked. Live executions bind
    their exact ``initial`` Future and already-settled/retained outcomes replay directly, while any path capable of
    starting new work checks current link/place permissions once before admission."""
    from ouroboros.peer_letters import audit, letter_task_id
    from ouroboros.peer_links import message_receipt, record_message, reply_receipt
    from ouroboros.presence_admission import PresenceAdmissionError, admit_presence_turn
    from ouroboros.presence_events import PresenceEventInvalid, PresenceTurnError, parse_event_wire, presence_turn_event
    from ouroboros.presence_runner import capture_presence_event
    from ouroboros.store import StoreError, StoreMigrating

    root, letter_id, place_id = pathlib.Path(request_drive_root(request)), letter["letter_id"], caller["place_id"]
    with _holding(_id_key(place_id, letter_id)):
        receipt = message_receipt(root, caller["link_id"], letter_id)
        if receipt is not None and receipt.get("content_sha256") != sha:
            return _refuse(root, "letter_id_reused", "this letter id already carried another message", **facts)
        if receipt is None:
            if _stored(root, letter_task_id(place_id, letter_id)) is not None:
                return _refuse(root, "letter_id_reused", "this letter id already carried a letter", **facts)
            if reply_receipt(root, caller["link_id"], letter_id) is not None:
                return _refuse(root, "letter_id_reused", "this letter id already carried an answer", **facts)
            refusal, place = _admission(root, caller, new_work=False)
            if refusal is not None:
                return _refuse(root, *refusal, **facts)
        else:
            place = _places(root).places.get(place_id)
            if place is None:
                return _refuse(root, "link_unknown", "this link's place is no longer connected", **facts)
            try:
                wire = parse_event_wire(_peer_message_event_payload(place, caller, letter),
                                        reporting_version=1, continuing=2)
            except PresenceEventInvalid as exc:
                return _typed_answer(root, exc, facts) or _unconfirmed(str(exc), root=root, **facts)
            state, detail = _repeat_message_gate(root, place, letter_id, wire, facts)
            if state in {"live", "done"}:
                return detail  # type: ignore[return-value]
            if state == "new":
                refusal, place = _admission(root, caller, new_work=False)
                if refusal is not None:
                    return _refuse(root, *refusal, **facts)
                if isinstance(detail, Exception):
                    return _typed_answer(root, detail, facts) or _received(letter_id)
            else:  # its turn left a durable row no retry may follow: received, nothing admitted or dispatched
                return _received(letter_id)
        payload = _peer_message_event_payload(place, caller, letter)
        try:
            admission = admit_presence_turn(
                drive_root=root, authenticated_transport_skill=place.connector, binding_id=place.place_id,
                provider=place.provider, account_id=place.account_id, conversation_id=place.conversation_id,
                thread_id="", live_event=True)
            # This host is the link's connector: it reports actual delivery (1) and takes the early answer (2).
            event = presence_turn_event(payload, admission, reporting_version=1, continuing=2)
            captured = capture_presence_event(admission, event, drive_root=root)
        except PresenceTurnError as exc:
            if exc.code == "presence_event_identity_conflict":
                return _refuse(root, "letter_id_reused", "this letter id already carried another message", **facts)
            return _unconfirmed(str(exc), root=root, **facts)
        except (PresenceAdmissionError, StoreMigrating, StoreError) as exc:
            return _unconfirmed(str(exc), root=root, **facts)
        if receipt is None:
            record_message(root, caller["link_id"], letter_id, task_id=str(captured["id"]),
                           chat_id=int(captured["chat_id"]), content_sha256=sha)
            audit(root, "peer_message_received", task_id=str(captured["id"]), chat_id=int(captured["chat_id"]), **facts)
    answer = _start_intake(letter_id, root, place, admission, event, payload, captured, facts)
    return answer if answer is not None else _received(letter_id)


def _letter(request: Request, caller: dict, body: Any) -> JSONResponse | _PendingIntake:
    from ouroboros.peer_letters import content_sha256
    from ouroboros.peer_links import PeerLinkError

    root = request_drive_root(request)
    facts: dict = {"link_id": caller["link_id"], "place_id": caller["place_id"]}
    try:
        letter = parse_letter(body)
        facts["letter_id"] = letter["letter_id"]
        refusal = _link_refusal(request, caller)
        if refusal is not None:
            return _refuse(root, *refusal, **facts)
        if letter.get("kind") == "message":
            if letter.get("reply_to"):
                facts["reply_to"] = letter["reply_to"]
            return _accept_message(request, caller, letter, content_sha256(letter), facts)
        if "in_reply_to" in letter:
            facts["in_reply_to"] = letter["in_reply_to"]["letter_id"]
            return _accept_reply(request, caller, letter, content_sha256(letter), facts)
        return _accept_new(request, caller, letter, content_sha256(letter), facts)
    except (LetterRefused, PeerLinkError) as exc:
        return _refuse(root, exc.code, str(exc), **facts)


async def api_letters_create(request: Request) -> JSONResponse:
    caller = _peer(request)
    if caller is None:
        return await asyncio.to_thread(_refuse, request_drive_root(request), "link_unknown", "a link key is required")
    raw = await request.body()
    try:
        body = json.loads(raw.decode("utf-8")) if raw.strip() else None
    except (ValueError, UnicodeError):
        body = None
    outcome = await run_sync_to_completion(_letter, request, caller, body)
    if isinstance(outcome, _PendingIntake):
        return await _await_intake(outcome)
    return outcome


def _read_letter(request: Request, caller: dict, letter_id: str, view: str) -> JSONResponse:
    from ouroboros import peer_letters
    from ouroboros.peer_links import PeerLinkError

    root = request_drive_root(request)
    try:
        refusal = _link_refusal(request, caller)
        if refusal is not None:
            return _refuse(root, *refusal, audited=False)
        found = _letter_row(root, caller, letter_id) if LETTER_ID_PATTERN.fullmatch(letter_id) else None
    except (LetterRefused, PeerLinkError) as exc:
        return _refuse(root, exc.code, str(exc), audited=False)
    if found is None:
        return _not_found(letter_id)
    if view == "state":  # a Stop's read-back: the state alone, so no words of the work cross for it
        return JSONResponse({"letter_id": letter_id, "view": "state", **peer_letters.letter_state_only(found[1])},
                            headers=_NO_STORE)
    link_id = caller["link_id"]
    status = _status_view(letter_id, found[1], root=root, link_id=link_id)
    if view == "status":
        return JSONResponse(status, headers=_NO_STORE)
    if status["state"] != "done":
        return JSONResponse({"error": "not_done", "state": status["state"]}, status_code=409, headers=_NO_STORE)
    return JSONResponse(
        {"letter_id": letter_id, "state": "done",
         **peer_letters.disclosed_letter_result(root, found[1], link_id=link_id)},
        headers=_NO_STORE,
    )


async def _letter_view(request: Request, view: str) -> JSONResponse:
    caller = _peer(request)
    if caller is None:  # a refused read leaves no row (a sender polling without its key would add one each time)
        return _refuse(request_drive_root(request), "link_unknown", "a link key is required", audited=False)
    return await run_sync_to_completion(_read_letter, request, caller, str(request.path_params["letter_id"]), view)


async def api_letter_status(request: Request) -> JSONResponse:
    """``GET /api/letters/{letter_id}``: the sender's view of its letter, whatever the id spells."""
    return await _letter_view(request, "status")


async def api_letter_state(request: Request) -> JSONResponse:
    """``GET /api/letters/{letter_id}/state``: the letter's state alone (``view: "state"``), what its sender's Stop
    reads back, also across a Pause of the sender's place: never a question, failure text or result."""
    return await _letter_view(request, "state")


async def api_letter_result(request: Request) -> JSONResponse:
    """``GET /api/letters/{letter_id}/result``: the addressed answer, only when done."""
    return await _letter_view(request, "result")


def _cancel_letter(request: Request, caller: dict, letter_id: str) -> JSONResponse:
    """The sender's cancel of its own letter: the existing Stop of that root and its helpers, asked of the Stop's
    owner (``_record_stop``), which alone decides whether anything of a settled root still runs. It is recorded as
    the link's (``PEER_CANCEL_SOURCE``, the link in ``request_origin``), never as my human's."""
    from ouroboros.gateway.tasks import _drive_stop, _record_stop
    from ouroboros.peer_letters import audit
    from ouroboros.peer_links import PeerLinkError

    root = request_drive_root(request)
    facts = {"link_id": caller["link_id"], "place_id": caller["place_id"], "letter_id": letter_id}
    try:
        refusal = _link_refusal(request, caller)
        if refusal is not None:
            return _refuse(root, *refusal, **facts)
        found = _letter_row(root, caller, letter_id) if LETTER_ID_PATTERN.fullmatch(letter_id) else None
        if found is None:
            return _not_found(letter_id)
        task_id = found[0]
        origin = {"kind": "linked_ouroboros", "link_id": caller["link_id"], "place_id": caller["place_id"]}
        try:
            intent = _record_stop(task_id, cascade=True, stop_action_id=f"peer-cancel-{task_id}",
                                  source=PEER_CANCEL_SOURCE, request_origin=origin)
        except Exception as exc:  # noqa: BLE001 - no intent was recorded: nothing changed, ask again
            raise LetterRefused("store_unavailable",
                                f"the cancel could not be recorded now ({type(exc).__name__})") from exc
        if intent is not None:
            audit(root, "peer_letter_cancel", task_id=task_id, **facts)
            _drive_stop(task_id, intent, cascade=True)
        found = _letter_row(root, caller, letter_id) or found
    except (LetterRefused, PeerLinkError) as exc:
        return _refuse(root, exc.code, str(exc), **facts)
    return JSONResponse({"letter_id": letter_id,
                         "state": _status_view(letter_id, found[1], root=root, link_id=caller["link_id"])["state"]},
                        headers=_NO_STORE)


async def api_letter_cancel(request: Request) -> JSONResponse:
    caller = _peer(request)
    if caller is None:
        return await asyncio.to_thread(_refuse, request_drive_root(request), "link_unknown", "a link key is required")
    return await run_sync_to_completion(_cancel_letter, request, caller, str(request.path_params["letter_id"]))


def _owner_door(request: Request) -> JSONResponse | None:
    if request.scope.get("ouroboros_caller"):
        return JSONResponse(refusal_body("peer_route_forbidden"), status_code=403, headers=_NO_STORE)
    return None


async def _owner_call(request: Request, operation: Any) -> JSONResponse:
    from ouroboros.peer_links import PeerLinkError
    from ouroboros.places import PlaceError

    if (refused := _owner_door(request)) is not None:
        return refused
    raw = await request.body()
    try:
        body = json.loads(raw.decode("utf-8")) if raw.strip() else {}
    except (ValueError, UnicodeError):
        body = None
    if not isinstance(body, dict):
        return JSONResponse(refusal_body("invalid_request", "the body must be a JSON object"), status_code=400)
    try:
        return JSONResponse(await run_sync_to_completion(operation, request, body), headers=_NO_STORE)
    except PeerLinkError as exc:
        return JSONResponse(refusal_body(exc.code, str(exc)), status_code=exc.status, headers=_NO_STORE)
    except PlaceError as exc:
        return JSONResponse(refusal_body(exc.code or "place_invalid", str(exc)),
                            status_code=503 if exc.code == "places_unreadable" else 400, headers=_NO_STORE)


def _own_url(request: Request, body: dict) -> str:
    return str(body.get("own_url") or str(request.base_url)).rstrip("/")


def _invite(request: Request, body: dict) -> dict:
    from ouroboros.peer_links import PeerLinkError, create_invite

    if set(body) - {"own_url", "label", "own_label", "note", "may_start"}:
        raise PeerLinkError("unexpected_fields", "invite takes own_url, label, own_label, note and may_start")
    return create_invite(request_drive_root(request), own_url=_own_url(request, body), label=body.get("label") or "",
                         own_label=body.get("own_label") or "", note=body.get("note") or "",
                         may_start=body.get("may_start"))


def _accept(request: Request, body: dict) -> dict:
    from ouroboros.peer_links import PeerLinkError, accept_invite, preview_invite

    if set(body) - {"invite", "preview", "own_url", "label", "own_label", "note", "may_start"}:
        raise PeerLinkError("unexpected_fields",
                            "accept takes invite, preview, own_url, label, own_label, note and may_start")
    if body.get("preview") is True:
        return preview_invite(body.get("invite"))
    return accept_invite(request_drive_root(request), body.get("invite"), own_url=_own_url(request, body),
                         label=body.get("label") or "", own_label=body.get("own_label") or "",
                         note=body.get("note") or "", may_start=body.get("may_start"))


async def api_peer_link_invite(request: Request) -> JSONResponse:
    return await _owner_call(request, _invite)


async def api_peer_link_accept(request: Request) -> JSONResponse:
    return await _owner_call(request, _accept)


async def api_peer_link_revoke(request: Request) -> JSONResponse:
    from ouroboros.peer_links import revoke

    link_id = str(request.path_params["link_id"])
    return await _owner_call(request, lambda req, _body: revoke(request_drive_root(req), link_id))


def _confirm(request: Request, body: dict) -> dict:
    from ouroboros.peer_links import PeerLinkError, confirm

    if body:
        raise PeerLinkError("unexpected_fields", "confirm takes no fields: my human compared the codes")
    return confirm(request_drive_root(request), str(request.path_params["link_id"]))


async def api_peer_link_confirm(request: Request) -> JSONResponse:
    """My human's check of the link's six digits on this side: only then do its letters flow (peer_links.confirm)."""
    return await _owner_call(request, _confirm)


async def api_peer_link_introduce(request: Request) -> JSONResponse:
    """The accepting side's call under a one-time invitation: the token in ``X-Ouroboros-Invite`` is the whole
    authority, checked and claimed by ``peer_links.introduce``."""
    from ouroboros.peer_links import PeerLinkError, introduce
    from ouroboros.places import PlaceError

    root = request_drive_root(request)
    body = await request_json_or(request, None)
    try:
        answer = await run_sync_to_completion(introduce, root, request.headers.get("x-ouroboros-invite", ""), body)
    except PeerLinkError as exc:
        return await asyncio.to_thread(_refuse, root, exc.code, str(exc))
    except PlaceError as exc:  # place store failure during introduction
        code = "store_unavailable" if exc.code == "places_unreadable" else "invalid_request"
        return await asyncio.to_thread(_refuse, root, code, str(exc))
    return JSONResponse(answer, headers=_NO_STORE)


def peer_letter_routes() -> list[Route]:
    return [Route("/api/letters", api_letters_create, methods=["POST"]),
            Route("/api/letters/{letter_id}", api_letter_status, methods=["GET"]),
            Route("/api/letters/{letter_id}/state", api_letter_state, methods=["GET"]),
            Route("/api/letters/{letter_id}/result", api_letter_result, methods=["GET"]),
            Route("/api/letters/{letter_id}/cancel", api_letter_cancel, methods=["POST"]),
            Route("/api/peer-links/invite", api_peer_link_invite, methods=["POST"]),
            Route("/api/peer-links/accept", api_peer_link_accept, methods=["POST"]),
            Route("/api/peer-links/introduce", api_peer_link_introduce, methods=["POST"]),
            Route("/api/peer-links/{link_id}/confirm", api_peer_link_confirm, methods=["POST"]),
            Route("/api/peer-links/{link_id}/revoke", api_peer_link_revoke, methods=["POST"])]
