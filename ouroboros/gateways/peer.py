"""The sending side's wire to a linked Ouroboros: the engine kind ``peer`` of a delegate custody record.

Its gateway speaks the letter verbs of the contract (``POST /api/letters``, ``GET /api/letters/{id}``, a Stop's
state-only ``GET /api/letters/{id}/state``, ``GET /api/letters/{id}/result``, ``POST /api/letters/{id}/cancel``) and
translates a letter's state into the run detail the shared custody readers already understand, so ``delegate_wait``,
``delegate_cancel``, ``delegate_answer``, supervision and collection reach the linked server through
``execution_domain.bound_gateway`` and never the owned Claudexor engine.

The link (its address, both install ids and its secret's file) comes from ``peer_links.outgoing_link``; the three
headers are read at each request and never kept, printed or returned. A refusal the receiver answers (4xx) is
definite and carries its ``refusal_code``. Anything else on a send (503 ``unconfirmed``, another 5xx, an answer
that does not name the letter, a dead transport) leaves acceptance unknown: only the same letter id with the same
bytes (``letter_bytes``) may be sent again. An answer to a question is itself a letter (``in_reply_to``) whose id
derives from the question and the answer, so the same answer sent again is the same letter. Same-computer URL links
stay on loopback addresses (other URLs are refused before the secret leaves); cross-machine channel links send only to
the derived POSIX Unix-domain socket for the link.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import threading
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

import httpx

log = logging.getLogger(__name__)

UNCONFIRMED = "unconfirmed"
NOTHING_SENT = frozenset({
    "peer_link_unavailable", "peer_url_refused",
    "peer_channel_refused", "peer_door_unsupported", "peer_door_dir_invalid",
    "invalid_session", "channel_session_invalid", "channel_end_missing",
    "place_silent", "place_paused", "places_unreadable",
    "peer_answer_conflict", "peer_answer_pin_unavailable",
    "answer_held",  # the sending task's own launch gate (``delegate_interactions._peer_answer_post``)
})  # raised before any request leaves
_CONNECT_TIMEOUT_SEC = 5.0
_READ_TIMEOUT_SEC = 30.0
# A letter's state -> the run state the custody readers know (``delegate_custody.TERMINAL_STATES``).
_RUN_STATE = {"done": "succeeded", "failed": "failed", "cancelled": "cancelled",
              "accepted_running": "running", "needs_input": "running", "waiting_owner": "running"}


def peer_outgoing_speech_refusal(drive_root: Any, place_id: str) -> Optional[Tuple[str, str]]:
    """Narrow read-only preflight for outgoing peer speech (``POST /api/letters``).

    ``peer_links.outgoing_link`` keeps local Silent links readable for safe ``GET``/lookup paths;
    any actual outgoing POST still honours the place's current Silent/Paused prohibition without
    taking any write lock or mutating ``places.json``.
    """
    from ouroboros import peer_links, places

    pid = str(place_id or "").strip()
    if not drive_root or not pid:
        return None
    if not places.places_path(drive_root).exists():
        if peer_links.registry_path(drive_root).exists():
            return ("peer_link_unavailable", "this link's place record is missing here; nothing was sent")
        return None
    try:
        doc = places._read(drive_root)
    except (places.PlaceError, OSError, ValueError) as exc:
        return ("places_unreadable", f"place records could not be read here ({exc}); nothing was sent")
    row = (doc.get("places") or {}).get(pid)
    if not isinstance(row, dict):
        return ("peer_link_unavailable", "this link's place is no longer connected here; nothing was sent")
    state = str(row.get("state") or "")
    label = str(row.get("label") or pid)
    if state == places.STATE_SILENT:
        return ("place_silent", f"{label} is Silent here; outgoing letters are blocked while Silent; nothing was sent")
    if state == places.STATE_PAUSED:
        return ("place_paused", f"{label} is Paused here; outgoing letters are blocked while Paused; nothing was sent")
    if state != places.STATE_ACTIVE:
        return (f"place_{state or 'unavailable'}",
                f"{label} is {state or 'unavailable'} here; outgoing letters are blocked; nothing was sent")
    return None


def letter_bytes(body: Dict[str, Any]) -> bytes:
    """The one serialization of a letter body: a repeat of a recorded letter sends exactly these bytes."""
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _unavailable(code: str, message: str, status: int = 0, refusal: str = "") -> Exception:
    from ouroboros.gateways.claudexor import ClaudexorUnavailable

    exc = ClaudexorUnavailable(code, message, status_code=status)
    exc.refusal_code = refusal  # the receiver's own typed refusal; empty for everything else
    return exc


def peer_cancel_proven_unsent(engine_ref: Any, exc: Any) -> bool:
    """An offline peer's proved pre-send refusal leaves a recovered Stop owed, not ambiguous."""
    return (isinstance(engine_ref, dict) and engine_ref.get("kind") == "peer"
            and (getattr(exc, "channel_connect_failed", False) or exc.code in NOTHING_SENT))


def _this_letter(body: Any, run_id: Any) -> Dict[str, Any]:
    """The receiver's answer about letter ``run_id``, only when it names that letter (its ``letter_id``): an answer
    naming none or another is a typed read refusal, never that letter's state or result."""
    if not isinstance(body, dict) or str(body.get("letter_id") or "") != str(run_id):
        raise _unavailable("malformed_response", f"the linked Ouroboros's answer does not name letter {run_id}; "
                                                  "nothing of it is read")
    return body


def _normalized_peer_cost(raw: Any) -> Dict[str, Any]:
    """Receiver-reported cost as attributed info only: never sender cash, never turning unknown or non-final into final."""
    cost = raw if isinstance(raw, dict) else {}
    known = cost.get("known_usd")
    known_usd = (float(known) if isinstance(known, (int, float)) and not isinstance(known, bool)
                 and math.isfinite(float(known)) and float(known) >= 0.0 else None)
    return {"known_usd": known_usd, "final": bool(cost.get("final") is True and known_usd is not None)}


def _hold_code(value: Any) -> bool:
    """A receiver's hold code (``peer_contracts.LETTER_HOLDS``, or one a later version adds): a typed token."""
    return isinstance(value, str) and re.fullmatch(r"[a-z][a-z0-9_]{0,63}", value) is not None


def _held(what: str, code: str, question_id: str = "", *, reserved: bool = False) -> Dict[str, Any]:
    """What the receiving Ouroboros holds now, and the next step, as ``summary.peerHold``: its question's words
    (``what="question"``) or its done letter's answer (``"result"``), each cleared for this link. A question with a
    reserved answer takes only that same answer again, whatever holds its words."""
    if what == "question" and reserved:
        note = (f"The linked Ouroboros asked question {question_id}; an answer to it is reserved there, its delivery "
                f"not proven, and that installation holds the question's words now ({code}). Only that same answer "
                "may go: send it again with the same rows once that installation hears this link, never another; or "
                "keep waiting.")
    elif what == "question" and code == "rules_changed":
        note = (f"The linked Ouroboros asked question {question_id}; its words were cleared under that installation's "
                "earlier rules for this link and are not shown under its current ones. Its work asks again only when "
                f"it wakes: delegate_answer with interaction_id={question_id} and a short answer saying you cannot see "
                "the question wakes it; else keep waiting or cancel.")
    elif what == "question":
        note = (f"The linked Ouroboros asked question {question_id}; that installation holds its words now "
                f"({code}) and shows them once it may speak to this link again, when it arrives here as a question. "
                "Keep waiting or cancel.")
    elif code == "rules_changed":
        note = ("The letter is done there with an answer cleared for this link under that installation's earlier "
                "rules; its rules for this link changed since, so the answer is not given under the current ones and "
                "does not arrive by itself. This is not an empty answer.")
    else:
        note = (f"The letter is done there; that installation holds its answer now ({code}). This run stays open and "
                "reads the answer, with no new work there, once it may speak to this link again: keep waiting, or "
                "read it again with delegate_wait.")
    return {"what": what, "reason": code, **({"question_id": question_id} if question_id else {}),
            **({"delivery_state": "reserved"} if reserved else {}), "note": note}


def _validated_answer_text(interaction_id: str, answers: Any) -> str:
    """Validate that ``answers`` carries exactly one row addressed to ``interaction_id`` within envelope text limits."""
    from ouroboros.gateway.peer_contracts import TEXT_MAX_CHARS

    expected = str(interaction_id or "").strip()
    if not expected:
        raise _unavailable("answer_row_mismatch", "interaction_id is required to answer a peer question",
                           400, "answer_row_mismatch")
    if len(expected) > 64:
        raise _unavailable("invalid_letter", "interaction_id must be at most 64 characters", 400, "invalid_letter")
    if not isinstance(answers, list) or len(answers) != 1 or not isinstance(answers[0], dict):
        count = len(answers) if isinstance(answers, list) else "invalid"
        raise _unavailable("answer_row_mismatch",
                           f"peer question {expected} requires exactly one answer row (got {count})",
                           400, "answer_row_mismatch")
    row = answers[0]
    qid = str(row.get("questionId") or row.get("question_id") or "").strip()
    if qid != expected:
        raise _unavailable("answer_row_mismatch",
                           f"answer row question_id {qid!r} does not match interaction_id {expected!r}",
                           400, "answer_row_mismatch")
    parts: List[str] = [str(label) for label in (row.get("selectedLabels") or [])]
    parts.append(str(row.get("freeText") or ""))
    text = "\n".join(part for part in parts if part.strip())
    if not text.strip():
        raise _unavailable("empty_answer", f"answer text is required for peer question {expected}",
                           400, "empty_answer")
    if len(text) > TEXT_MAX_CHARS:
        raise _unavailable("invalid_letter",
                           f"answer text must be at most {TEXT_MAX_CHARS} characters (got {len(text)})",
                           400, "invalid_letter")
    return text


def peer_answer_letter(run_id: str, interaction_id: str, answers: Any) -> Tuple[str, Dict[str, Any], bytes]:
    """Deterministic reply letter id, validated body and serialized bytes for one peer clarification answer."""
    from ouroboros.gateway.peer_contracts import LetterRefused, parse_letter
    from ouroboros.peer_links import PROTOCOL

    text = _validated_answer_text(interaction_id, answers)
    digest = hashlib.sha256("\0".join((str(run_id), str(interaction_id), text)).encode("utf-8")).hexdigest()
    body = {"letter_id": "reply-" + digest[:32], "text": text, "protocol": PROTOCOL,
            "in_reply_to": {"letter_id": str(run_id), "question_id": str(interaction_id)}}
    try:
        parse_letter(body)
    except LetterRefused as exc:
        raise _unavailable(exc.code, str(exc), exc.status, exc.code) from exc
    return body["letter_id"], body, letter_bytes(body)


class PeerGateway:
    """One linked server: the four letter verbs. ``engine_ref`` lets ``execution_domain.same_engine`` prove that an
    answer (a 404 included) speaks for the record's link; the peer protocol has no store id, so
    ``store_identity`` stays empty."""

    engine_version = "peer-1"  # a peer server has no handshake route
    _endpoint = None

    def __init__(self, link: Any, *, timeout_sec: Optional[float] = None) -> None:
        from ouroboros import peer_channel

        self._link = link
        self.engine_ref = {"kind": "peer", "place_id": link.place_id, "link_id": link.link_id}
        self.store_identity = ""
        self._timeout_sec = timeout_sec if timeout_sec is not None else _READ_TIMEOUT_SEC
        self._channel_tls = threading.local()
        if peer_channel.is_channel_link(link):
            self._client = peer_channel.peer_http_client(
                channel_resolver=self._resolve_active_channel_socket,
                timeout_sec=self._timeout_sec,
                connect_timeout_sec=_CONNECT_TIMEOUT_SEC,
            )
        else:
            self._client = peer_channel.peer_http_client(
                peer_url=str(link.peer_url or ""),
                timeout_sec=self._timeout_sec,
                connect_timeout_sec=_CONNECT_TIMEOUT_SEC,
            )

    def _resolve_active_channel_socket(self) -> str:
        from ouroboros import peer_channel, peer_links

        bound = getattr(self._channel_tls, "socket_path", "")
        if bound:
            return bound
        link = self._link
        now = peer_links.outgoing_link(link.place_id, drive_root=link.drive_root) or peer_links.stopping_link(
            link.place_id, drive_root=link.drive_root)
        return str(peer_channel.channel_socket_for_link(now or link))

    def handshake(self, **_kw: Any) -> Dict[str, Any]:
        return {}

    def operations(self, *, timeout_sec: Optional[float] = None) -> List[Dict[str, Any]]:
        """A linked Ouroboros serves only its four letter routes (no live-message or event-stream routes)."""
        return []

    def agent_capabilities(self, *, timeout_sec: Optional[float] = None) -> Dict[str, Any]:
        """A peer letter has no live-input channel."""
        return {"harnesses": [{"id": "peer", "liveInput": "none"}]}

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:  # noqa: BLE001 - closing an idle pool never decides anything
            log.debug("peer client close failed", exc_info=True)

    def __enter__(self) -> "PeerGateway":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    def _request(self, method: str, path: str, *, content: Optional[bytes] = None,
                 timeout_sec: Optional[float] = None, stop: str = "") -> Any:
        return self._exchange(method, path, content=content, timeout_sec=timeout_sec, stop=stop)[0]

    def _exchange(self, method: str, path: str, *, content: Optional[bytes] = None,
                  timeout_sec: Optional[float] = None, stop: str = "") -> Tuple[Any, bool]:
        """One request and whether it went through a Pause of my place (only my Stop's own requests do)."""
        from ouroboros import peer_channel, peer_links
        from ouroboros.peer_links import PeerLinkError

        # The link as it stands at this request, like its key: a client kept across a Pause, an unmarking or a
        # revocation sends nothing then. A Pause still lets my side's own Stop of a letter it started (``stop``)
        # and that Stop's read-back through (``peer_links.stopping_link``).
        link = self._link
        now = peer_links.outgoing_link(link.place_id, drive_root=link.drive_root)
        paused = now is None and bool(stop)
        if paused:
            now = peer_links.stopping_link(link.place_id, drive_root=link.drive_root)
        if now is None or peer_channel.link_identity(now) != peer_channel.link_identity(link):
            raise _unavailable("peer_link_unavailable", "no usable link to that place here now (unknown, revoked, "
                                                        "paused, its code unconfirmed or not marked mine); nothing "
                                                        "was sent")
        ok, err_code, err_msg = peer_channel.link_target_valid(now)
        if not ok:
            raise _unavailable(err_code, err_msg)
        channel = peer_channel.is_channel_link(now)
        target_str = ""
        if channel:
            target_sock = peer_channel.channel_socket_for_link(now)
            if not target_sock.exists():
                exc = _unavailable("channel_unavailable",
                                   f"the channel socket to the linked Ouroboros is not present ({target_sock.name})")
                exc.channel_connect_failed = True
                exc.channel_transport_error = True
                raise exc
            target_str = str(target_sock)
        try:  # read at this request: the secret stays in its file between sends
            headers = {**now.auth_headers(), "Content-Type": "application/json"}
        except PeerLinkError as exc:
            raise _unavailable("peer_link_unavailable", f"this link has no key here; nothing was sent ({exc.code})")
        bound = {"timeout": httpx.Timeout(timeout_sec, connect=_CONNECT_TIMEOUT_SEC)} if timeout_sec is not None else {}
        if channel:
            self._channel_tls.socket_path = target_str
        try:
            response = self._client.request(method, path, content=content, headers=headers, **bound)
        except (httpx.HTTPError, OSError) as exc:
            if channel and peer_channel.is_channel_connect_failure(exc):
                err = _unavailable("channel_unavailable",
                                   f"the channel to the linked Ouroboros refused connection on {method} {path}: "
                                   f"{type(exc).__name__}")
                err.channel_connect_failed = True
                err.channel_transport_error = True
                raise err from exc
            err = _unavailable("peer_unreachable", f"the linked Ouroboros did not answer {method} {path}: "
                                                   f"{type(exc).__name__}")
            if channel:
                err.channel_transport_error = True
            raise err from exc
        finally:
            if channel:
                self._channel_tls.socket_path = ""
        if response.status_code >= 400:
            self._raise_refusal(response)
        try:
            return (response.json() if response.content else None), paused
        except ValueError as exc:
            raise _unavailable("malformed_response", f"the linked Ouroboros answered {method} {path} without JSON",
                               response.status_code) from exc

    def _mark_remote_unknown(self) -> None:
        drive = str(getattr(self._link, "drive_root", "") or "")
        link_id = str(getattr(self._link, "link_id", "") or "")
        if not drive or not link_id:
            return
        try:
            from ouroboros.peer_links import mark_remote_unknown

            mark_remote_unknown(drive, link_id)
        except Exception:  # noqa: BLE001 - marking link state must never mask the 401 refusal
            log.debug("mark_remote_unknown failed for link %s", link_id, exc_info=True)

    def _check_outgoing_speech(self) -> None:
        refusal = peer_outgoing_speech_refusal(getattr(self._link, "drive_root", ""),
                                                getattr(self._link, "place_id", ""))
        if refusal is not None:
            raise _unavailable(refusal[0], refusal[1])

    def _raise_refusal(self, response: httpx.Response) -> None:
        try:
            body = response.json()
        except ValueError:
            body = None
        body, status = (body if isinstance(body, dict) else {}), response.status_code
        error = str(body.get("error") or "")
        raw_refusal = body.get("refusal_code")
        refusal = (raw_refusal if isinstance(raw_refusal, str) and re.fullmatch(r"[a-z][a-z0-9_]*", raw_refusal)
                   else "")
        detail = str(body.get("detail") or error or response.text[:300])
        if status == 503 or error == UNCONFIRMED:
            raise _unavailable(UNCONFIRMED, detail or "acceptance unconfirmed", status)
        if status == 401 and refusal == "link_unknown":
            self._mark_remote_unknown()
        if refusal:
            exc = _unavailable(refusal, detail, status, refusal)
            exc.retry_after = body.get("retry_after")  # a typed retry's own delay (429), else None
            raise exc
        # An error token without a refusal code (404 letter_not_found, 409 not_done) is itself the typed code.
        raise _unavailable(error if re.fullmatch(r"[a-z][a-z0-9_]*", error) else f"http_{status}", detail, status)

    def start_run(self, request: Dict[str, Any], *, idempotency_key: str = "") -> Dict[str, Any]:
        """``POST /api/letters`` with the recorded letter's bytes. Accepted is ``{runId = letter_id, state, phase}``;
        a receiver 4xx carrying a typed ``refusal_code`` raises with that code; any unnamed 4xx, 5xx, transport fault
        or foreign answer raises ``unconfirmed``."""
        letter_id = str((request or {}).get("letter_id") or "")
        self._check_outgoing_speech()
        try:
            body = self._request("POST", "/api/letters", content=letter_bytes(dict(request or {})))
        except Exception as exc:
            status = int(getattr(exc, "status_code", 0) or 0)
            refusal = str(getattr(exc, "refusal_code", "") or "")
            if (getattr(exc, "code", "") in NOTHING_SENT
                    or getattr(exc, "channel_connect_failed", False)
                    or (refusal and 400 <= status < 500)):
                raise
            raise _unavailable(UNCONFIRMED, f"whether the letter was accepted is unknown: {exc}", status) from exc
        if not isinstance(body, dict) or str(body.get("letter_id") or "") != letter_id:
            raise _unavailable(UNCONFIRMED, "the answer does not name this letter; whether it was accepted is unknown")
        return {"runId": letter_id, "state": body.get("state"), "phase": body.get("phase")}

    def send_message(self, text: str, *, letter_id: str, reply_to: str = "",
                     timeout_sec: Optional[float] = None) -> Dict[str, Any]:
        """``POST /api/letters`` with ``kind="message"``: one ordinary message into the link's place conversation,
        never a work assignment, under the caller's stable ``letter_id`` (the same words again travel as the same
        letter). ``state="received"`` naming it means saved in that room, never read; a typed retry (429) keeps its
        ``retry_after``."""
        from ouroboros.peer_links import PROTOCOL

        lid = str(letter_id or "").strip()
        if not lid:
            raise _unavailable("invalid_letter", "a message needs its stable letter id; nothing was sent")
        self._check_outgoing_speech()
        payload: Dict[str, Any] = {"letter_id": lid, "kind": "message", "text": str(text), "protocol": PROTOCOL,
                                   **({"reply_to": str(reply_to).strip()} if str(reply_to or "").strip() else {})}
        try:
            body = self._request("POST", "/api/letters", content=letter_bytes(payload), timeout_sec=timeout_sec)
        except Exception as exc:
            status = int(getattr(exc, "status_code", 0) or 0)
            refusal = str(getattr(exc, "refusal_code", "") or "")
            if (getattr(exc, "code", "") in NOTHING_SENT or (refusal and 400 <= status < 500)
                    or getattr(exc, "channel_connect_failed", False)
                    or getattr(exc, "channel_transport_error", False)):
                raise
            raise _unavailable(UNCONFIRMED, f"whether the message was received is unknown: {exc}", status) from exc
        if (not isinstance(body, dict) or str(body.get("letter_id") or "") != lid
                or str(body.get("state") or "") != "received"):
            raise _unavailable(UNCONFIRMED, "the answer does not confirm receipt of this message; "
                                            "whether it was received is unknown")
        return {"letter_id": lid, "kind": "message", "state": "received"}

    def get_run(self, run_id: str, *, timeout_sec: Optional[float] = None, for_stop: bool = False) -> Dict[str, Any]:
        """``GET /api/letters/{id}`` as the run detail custody readers expect: ``summary.state`` in their vocabulary,
        an addressed question as ``pendingInteractions`` and, at ``done``, the result as the primary output. The
        detail is named by the receiver's own ``letter_id`` (``_this_letter``); unknown states or malformed
        ``needs_input`` questions raise ``malformed_response``. A question or answer the receiver holds now (its
        ``held`` / ``result_held`` code) is ``summary.peerHold`` with its next step (``_held``), never running work or
        an empty answer. An explicit operational Stop read-back (``for_stop=True``) observes state alone through
        ``_stop_state``. An ordinary ``get_run`` (``for_stop=False``) never inherits Pause permission from an earlier
        ``cancel_run``."""
        if for_stop:
            return self._stop_state(run_id, timeout_sec)
        body, _paused = self._exchange("GET", f"/api/letters/{quote(str(run_id), safe='')}", timeout_sec=timeout_sec)
        letter = _this_letter(body, run_id)
        name, state = str(letter["letter_id"]), str(letter.get("state") or "")
        if state not in _RUN_STATE:
            raise _unavailable("malformed_response",
                               f"the linked Ouroboros returned unknown letter state {state!r} for {run_id}")
        summary: Dict[str, Any] = {"runId": name, "state": _RUN_STATE[state], "peerState": state,
                                   **({"phase": letter["phase"]} if letter.get("phase") else {})}
        detail: Dict[str, Any] = {"runId": name, "lastSeq": 1, "summary": summary, "pendingInteractions": []}
        if state == "needs_input":
            question = letter.get("question")
            qid = question.get("question_id") if isinstance(question, dict) else None
            qtext = question.get("text") if isinstance(question, dict) else None
            held = question.get("held") if isinstance(question, dict) else None
            delivery_state = question.get("delivery_state") if isinstance(question, dict) else None
            if (not isinstance(qid, str) or not qid.strip() or len(qid) > 64
                    or (held is None and (not isinstance(qtext, str) or not qtext.strip()))
                    or (held is not None and (qtext is not None or not _hold_code(held)))
                    or delivery_state not in (None, "reserved")):
                raise _unavailable("malformed_response",
                                   f"the linked Ouroboros reported needs_input for {run_id} without a valid question")
            if held is not None:  # asked, its words held there now: nothing here to read or answer yet
                summary["peerHold"] = _held("question", held, qid.strip(), reserved=bool(delivery_state))
                return detail
            summary["waitingOnUser"] = True
            detail["pendingInteractions"] = [{
                "interactionId": qid.strip(), "runId": str(run_id), "delivery": "blocking",
                **({"deliveryState": delivery_state} if delivery_state else {}),
                "questions": [{"id": qid.strip(), "question": qtext,
                               "options": [], "multi_select": False}]}]
        elif state == "waiting_owner":  # the receiver's human was asked or work is held there; sender never reads owner text
            summary["waitingOnUser"] = True
        if letter.get("error") or letter.get("refusal_code"):
            summary.update(error=str(letter.get("error") or ""), refusalCode=str(letter.get("refusal_code") or ""))
        if state == "done":
            self._done(detail, summary, timeout_sec)
        elif state in {"failed", "cancelled"}:
            # Non-done letters have no /result endpoint; keep receiver spend explicitly unknown, never sender cash.
            detail["peerCost"] = {"known_usd": None, "final": False}
        return detail

    def _done(self, detail: Dict[str, Any], summary: Dict[str, Any], timeout_sec: Optional[float]) -> None:
        """The result of a done letter, read once: its addressed text only when proven available, when it ended and
        what is known of the receiver's cost (attributed info only — never copied into sender ``spendUsd``).

        Wire ``result_available: bool`` rides unchanged in ``finalSummary.result_available``:
        - ``True`` with a string ``result`` (empty included): populates ``primaryOutput`` and ``result_available=True``.
        - ``False``: confirmed terminal without an addressed answer; ``primaryOutput`` stays absent (``None``).
        - ``False`` with ``result_held``: an answer cleared for this link is held there now (``summary.peerHold``). A
          passing hold leaves the run live (``summary.state`` running): it ends with the answer once the receiver may
          give it, read by a later read or wait with no new work there, never settled as an empty one;
          ``rules_changed`` ends it with that fact (``finalSummary.result_held``), since it does not come by itself.
        - Absent or invalid (non-bool ``result_available``, or ``True`` without a string ``result``): ``primaryOutput``
          stays absent (``None``) and ``result_available`` stays absent while ``succeeded`` state and cost remain.
        """
        result = self.get_run_result(summary["runId"], timeout_sec=timeout_sec)
        final: Dict[str, Any] = {"finished_at": result["finished_at"]} if result.get("finished_at") else {}
        avail = result.get("result_available")
        raw_text = result.get("result")
        held = result.get("result_held")
        if avail is False and _hold_code(held):
            summary["peerHold"] = _held("result", held)
            if held != "rules_changed":
                summary["state"] = _RUN_STATE["accepted_running"]
                return
            final["result_held"] = held
        if type(avail) is bool:
            if avail is True:
                if isinstance(raw_text, str):
                    detail["primaryOutput"] = {
                        "text": raw_text, "bytes": len(raw_text.encode("utf-8")), "truncated": False,
                    }
                    final["result_available"] = True
            else:
                final["result_available"] = False
        peer_cost = _normalized_peer_cost(result.get("cost"))
        detail["peerCost"] = peer_cost
        if isinstance(result.get("cost"), dict):
            final["cost"] = dict(peer_cost)
        if final:
            detail["finalSummary"] = final

    def _stop_state(self, run_id: str, timeout_sec: Optional[float]) -> Dict[str, Any]:
        """A Stop's read-back: ``GET /api/letters/{id}/state``, which the receiver answers with the state alone
        (``view: "state"``), also across a Pause of my place (``summary.contentWithheld``): never a question, failure
        text or ``/result``, even while the link is Active. A receiver that does not serve that read (an earlier
        version refuses the route) is ``peer_state_read_unsupported``: its full status, which can carry a question's
        words, is never read in its place."""
        try:
            body, paused = self._exchange("GET", f"/api/letters/{quote(str(run_id), safe='')}/state",
                                          timeout_sec=timeout_sec, stop=str(run_id))
        except Exception as exc:
            status, code = int(getattr(exc, "status_code", 0) or 0), str(getattr(exc, "code", "") or "")
            if code == "peer_route_forbidden" or (status in (404, 405) and code != "letter_not_found"):
                raise _unavailable("peer_state_read_unsupported", "the linked Ouroboros does not serve the state-only "
                                   "read a Stop uses; its full status is not read in its place", status) from exc
            raise
        letter = _this_letter(body, run_id)
        state = str(letter.get("state") or "")
        if letter.get("view") != "state":
            raise _unavailable("peer_state_read_unsupported", "the linked Ouroboros's answer to the state-only read "
                                                              "does not confirm that view; nothing of it is read")
        if state not in _RUN_STATE:
            raise _unavailable("malformed_response",
                               f"the linked Ouroboros returned unknown letter state {state!r} for {run_id}")
        phase = {"phase": letter["phase"]} if letter.get("phase") in ("queued", "running") else {}
        summary: Dict[str, Any] = {"runId": str(letter["letter_id"]), "state": _RUN_STATE[state], "peerState": state,
                                   **phase, "contentWithheld": "paused" if paused else "stop"}
        return {"runId": summary["runId"], "lastSeq": 1, "summary": summary, "pendingInteractions": [],
                "peerCost": {"known_usd": None, "final": False}}

    def get_run_result(self, run_id: str, *, timeout_sec: Optional[float] = None) -> Dict[str, Any]:
        """``GET /api/letters/{id}/result``: only a done letter has one (else ``409 not_done`` raises typed)."""
        body = self._request("GET", f"/api/letters/{quote(str(run_id), safe='')}/result", timeout_sec=timeout_sec)
        return _this_letter(body, run_id)  # content: never through a Pause, a Stop's included

    def cancel_run(self, run_id: str, *, reason: str = "") -> Dict[str, Any]:
        """``POST /api/letters/{id}/cancel``: the receiver's Stop of that work; ``cancel_and_verify`` reads it back
        via ``get_run(..., for_stop=True)``. My side's own Stop of a letter it started goes during a Pause of its
        place too, without granting Pause permission to ordinary ``get_run`` calls."""
        body = self._request("POST", f"/api/letters/{quote(str(run_id), safe='')}/cancel", stop=str(run_id))
        letter = _this_letter(body, run_id)
        state = str(letter.get("state") or "")
        if state not in _RUN_STATE:
            raise _unavailable("malformed_response",
                               f"the linked Ouroboros returned unknown cancel state {state!r} for {run_id}")
        return {"accepted": True, "status": "accepted", "state": state}

    def answer_interaction(self, run_id: str, interaction_id: str, answers: List[Dict[str, Any]],
                           *, timeout_sec: Optional[float] = None,
                           before_send: Optional[Any] = None) -> Dict[str, Any]:
        """The answer letter to a ``needs_input`` question (``in_reply_to``; no title, ceiling or chain). Requires
        exactly one answer row matching ``interaction_id`` and an active local link (not Silent/Paused). When
        ``before_send(letter_id, raw_bytes)`` is given, it runs after local validation/link guards and before any
        network POST (outside any lock). Only the receiver's ``LetterAnswerReceipt`` for this reply letter (its id,
        its exact ``in_reply_to``, ``delivered`` and the mailbox record ``msg_id`` its words landed as) confirms
        delivery, the same for a repeat of the same bytes; any other response or unnamed 4xx raises ``unconfirmed``."""
        from ouroboros import peer_channel, peer_links
        from ouroboros.peer_links import PeerLinkError

        letter_id, body, raw = peer_answer_letter(run_id, interaction_id, answers)
        self._check_outgoing_speech()
        now = peer_links.outgoing_link(self._link.place_id, drive_root=self._link.drive_root)
        if now is None or peer_channel.link_identity(now) != peer_channel.link_identity(self._link):
            raise _unavailable("peer_link_unavailable", "no usable link to that place here now (unknown, revoked, "
                                                        "paused, its code unconfirmed or not marked mine); nothing "
                                                        "was sent")
        ok, err_code, err_msg = peer_channel.link_target_valid(now)
        if not ok:
            raise _unavailable(err_code, err_msg)
        if peer_channel.is_channel_link(now):
            target_sock = peer_channel.channel_socket_for_link(now)
            if not target_sock.exists():
                exc = _unavailable("channel_unavailable",
                                   f"the channel socket to the linked Ouroboros is not present ({target_sock.name})")
                exc.channel_connect_failed = True
                exc.channel_transport_error = True
                raise exc
        try:
            now.auth_headers()
        except PeerLinkError as exc:
            raise _unavailable("peer_link_unavailable", f"this link has no key here; nothing was sent ({exc.code})")
        if before_send is not None:
            before_send(letter_id, raw)
        try:
            resp = self._request("POST", "/api/letters", content=raw, timeout_sec=timeout_sec)
        except Exception as exc:
            if str(getattr(exc, "code", "") or "") in NOTHING_SENT or getattr(exc, "channel_connect_failed", False):
                raise
            refusal = str(getattr(exc, "refusal_code", "") or "")
            status = int(getattr(exc, "status_code", 0) or 0)
            if refusal in {"question_not_found", "question_closed"}:
                mapped = "not_found" if refusal == "question_not_found" else "already_resolved"
                return {"status": mapped, "accepted": False, "message": refusal}
            if refusal and 400 <= status < 500:
                raise
            raise _unavailable(UNCONFIRMED, f"whether the answer was delivered is unknown: {exc}", 0) from exc
        receipt = resp if isinstance(resp, dict) else {}
        msg_id = receipt.get("msg_id")
        if (receipt.get("letter_id") != letter_id or receipt.get("delivered") is not True
                or receipt.get("in_reply_to") != body["in_reply_to"] or not isinstance(msg_id, str) or not msg_id.strip()):
            raise _unavailable(UNCONFIRMED, "the answer's receipt does not prove this reply letter in the asking "
                                            "work's mailbox; whether it was delivered is unknown")
        return {"status": "delivered", "accepted": True, "message": "", "letter_id": letter_id,
                "msg_id": msg_id.strip()}

    def _unsupported(self, what: str) -> None:
        raise _unavailable("unsupported", f"a letter has no {what}: create, status, result and cancel only", 501)

    def send_run_message(self, run_id: str, text: str, *, idempotency_key: str, expected_attempt_id: str = "",
                         timeout_sec: Optional[float] = None) -> Dict[str, Any]:
        """A linked Ouroboros has no live-message channel: return typed ``unsupported`` without sending anything."""
        return {"outcome": "unsupported", "reason": "peer_live_input_unsupported",
                "message": "a linked Ouroboros has no live-message channel; nothing was sent"}

    def get_run_artifact(self, run_id: str, path: str) -> bytes:
        self._unsupported("artifact")

    def open_run_events(self, run_id: str, after_seq: int, timeout_sec: float) -> Any:
        self._unsupported("event stream")

    def take_run_result(self, run_id: str, *, idempotency_key: str) -> Dict[str, Any]:
        self._unsupported("result-taken decision")


def peer_gateway(ref: Any, *, timeout_sec: Optional[float] = None) -> PeerGateway:
    """The gateway of a ``{"kind": "peer", "place_id", "link_id"}`` record through its current link; a link that is
    gone, revoked, unconfirmed or not marked mine, or an invalid transport target, raises typed: a peer record
    never falls back to another engine. A Paused place still yields one whose requests carry only my side's Stop of
    its own letter and that Stop's read-back (``PeerGateway._request``)."""
    from ouroboros import peer_channel, peer_links

    ref = ref if isinstance(ref, dict) else {}
    place_id = str(ref.get("place_id") or "")
    link = peer_links.outgoing_link(place_id) or peer_links.stopping_link(place_id)
    if link is None or link.link_id != str(ref.get("link_id") or ""):
        raise _unavailable("peer_link_unavailable", "no usable link to that place here (unknown, revoked, paused, "
                                                    "its code unconfirmed or not marked mine); nothing was sent")
    ok, err_code, err_msg = peer_channel.link_target_valid(link)
    if not ok:
        raise _unavailable(err_code, err_msg)
    return PeerGateway(link, timeout_sec=timeout_sec)
