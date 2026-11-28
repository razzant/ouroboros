"""Outgoing ordinary peer messages and automatic replies to a linked Ouroboros."""
from __future__ import annotations

import contextlib
import hashlib
import logging
import pathlib
import threading
import time
from typing import Any

log = logging.getLogger(__name__)

MESSAGE_PART = "0"  # one letter is one receipt part
_SEND_ATTEMPTS = 3  # the same bytes again while the receiver's confirmation is lost
_ATTEMPT_SEC = 20.0  # one attempt's wait: three of them stay inside a tool call's 120 s
_RETRY_PAUSE_SEC = 0.5  # before the next attempt, growing by one step each time
_MAX_PASSES = 4  # calls that send one letter without a confirmation, across calls and starts; then it stops
_FOLLOW_UP_SEC = (30.0, 120.0, 480.0)  # an automatic reply's next passes in this process after a retry or no answer
_FOLLOW_UP_MAX_SEC = 900.0  # the longest wait of one follow-up, whatever the receiver's own retry_after
_RECORDERS_LOCK = threading.Lock()
_RECORDERS: dict[str, Any] = {}
_SENDING_LOCK = threading.Lock()
_SENDING: set[str] = set()  # letters a call of this process is sending now (``send_peer_message``'s claim)
_NOTHING_SENT = "; nothing was sent"

# Holds after which an unsent automatic reply stays owed and unsaid for a later start or my human's return:
# the switch (Panic or unreadable), my place's lawful Pause or Silence, and transient unreadable storage/rules.
STILL_OWED = _STILL_OWED = frozenset({
    "panic_hold", "switch_unreadable", "place_paused", "place_silent", "peer_links_unreadable", "places_unreadable",
    "rules_unreadable", "sending_unrecorded", "receipts_unreadable", "earlier_attempts_unknown", "peer_link_transient",
    "channel_unavailable",
})


def follow_up_schedule() -> tuple[tuple[float, ...], float]:
    """Return the active bounded follow-up delays and maximum wait (honors test monkeypatches)."""
    return tuple(_FOLLOW_UP_SEC), float(_FOLLOW_UP_MAX_SEC)


class AttemptHeld(Exception):
    """The gate of one physical attempt closed before it (``send_peer_message(gate=...)``): nothing of it left."""

    def __init__(self, code: str, message: str = "") -> None:
        super().__init__(message or code)
        self.code = code


def message_letter_id(link_id: str, *identity: str) -> str:
    """The one letter id of one logical message on ``link_id``, from what makes it that message."""
    return "msg-" + hashlib.sha256("\0".join((link_id, *identity)).encode("utf-8")).hexdigest()[:32]


def _receipts(root: Any) -> Any:
    """This process's writer of peer message receipts (``PresenceDeliveryRecorder``): their identities are the
    ``ouroboros`` connector's alone, and the store keeps one row per identity across processes."""
    from ouroboros.presence_delivery import PresenceDeliveryRecorder

    key = str(pathlib.Path(root).resolve())
    with _RECORDERS_LOCK:
        if key not in _RECORDERS:
            _RECORDERS[key] = PresenceDeliveryRecorder(pathlib.Path(root))
        return _RECORDERS[key]


def _held(letter_id: str, code: str, message: str) -> dict:
    return {"state": "held", "code": code, "message": message, "letter_id": letter_id}


def prewire_hold(letter_id: str, earlier: str, code: str, message: str) -> dict:
    if earlier == "uncertain":
        return {**kept_outcome(earlier, message), "held_code": code}
    return _held(letter_id, code, message)


def _unknown(letter_id: str, code: str, message: str) -> dict:
    return {"state": "unknown", "code": code, "message": message, "letter_id": letter_id, "recorded": False}


def earlier_state(root: Any, link_id: str, account_id: str, letter_id: str) -> tuple[str, set, dict, str]:
    """What earlier, finished calls left for ``letter_id``, read before this call sends anything.

    Reads canonical receipts and ``sent/`` independently: a confirmed outcome (``delivered`` or ``failed`` in either
    store) settles the letter even when the other store is temporarily unreadable; without a confirmed outcome, an
    unreadable store yields ``unknown`` (never proof of none, and never downgraded to a fabricated ``uncertain``
    receipt). Returns ``(state, recorded_states_set, sending_record_dict, read_error_code)``.
    """
    from ouroboros import peer_links
    from ouroboros.presence_delivery import recorded_states
    states, states_err, record, record_err = set(), "", {}, ""
    # Writers publish the receipt before dropping sent/. Read in the opposite order.
    try:
        record = peer_links.sending_record(root, link_id, letter_id) or {}
    except peer_links.PeerLinkError:
        record_err = "peer_links_unreadable"
    try:
        states = recorded_states(root, peer_links.CONNECTOR, account_id, letter_id, MESSAGE_PART)
    except Exception:  # noqa: BLE001 - an unreadable receipt store is unknown, never absent
        states_err = "receipts_unreadable"
    confirmed = str(record.get("confirmed_state") or "")
    confirmed = confirmed if confirmed in {"delivered", "failed"} else ""
    if "delivered" in states or confirmed == "delivered":
        return "delivered", states, record, ""
    if "uncertain" not in states and ("failed" in states or confirmed == "failed"):
        return "failed", states, record, ""
    if states_err or record_err:
        return "unknown", states, record, states_err or record_err
    pre_tx = bool(record.get("pre_transmission")) and int(record.get("passes") or 0) == 0
    return ("uncertain" if ("uncertain" in states or (record and not pre_tx)) else "none"), states, record, ""


def _recorded_receipts(root: Any, link_id: str, account_id: str, letter_id: str) -> list[dict]:
    """Indexed lookup of canonical receipt payloads for ``(link_id, account_id, letter_id)`` (no history scan)."""
    from ouroboros import dialogue_store as ds, peer_links
    from ouroboros.presence_delivery import _STATES, _client_id, _key, _payload_from_row

    ds.require_imported(pathlib.Path(root), "chat")
    rows: list[dict] = []
    for state in ("delivered", "failed", "uncertain", "accepted"):
        if state not in _STATES:
            continue
        key = (peer_links.CONNECTOR, account_id, letter_id, MESSAGE_PART, state)
        for entry in ds.iter_probes(pathlib.Path(root), "chat", [("client_message_id", _client_id(key))]):
            parsed = _payload_from_row(entry.row)
            if parsed is not None and _key(*parsed) == key and str(parsed[1].get("conversation_id") or "") == link_id:
                rows.append(parsed[1])
    return rows


def resolve_message_retry(root: Any, link_id: str, account_id: str, letter_id: str) -> tuple[dict | None, str]:
    """Read the exact original task, place, recipient, words and reply address from `sent/` or canonical receipts.
    This lookup cannot send, change address or designate a different task as the creator.
    """
    from ouroboros import peer_links

    record_err = ""
    try:
        record = peer_links.sending_record(root, link_id, letter_id)
    except peer_links.PeerLinkError:
        record, record_err = None, "peer_links_unreadable"
    if isinstance(record, dict):
        return record, ""
    receipts_err = ""
    try:
        receipts = _recorded_receipts(root, link_id, account_id, letter_id)
    except Exception:  # noqa: BLE001
        receipts, receipts_err = [], "receipts_unreadable"
    if receipts:
        payload = receipts[0]
        msg = payload.get("message") if isinstance(payload.get("message"), dict) else {}
        return {"letter_id": letter_id, "account_id": str(payload.get("account_id") or ""), "place_id": "",
                "text": str(payload.get("text") or ""), "reply_to": str(msg.get("reply_to") or ""),
                "origin": dict(payload.get("origin") or {}) if isinstance(payload.get("origin"), dict) else {}}, ""
    return None, record_err or receipts_err


def _prior_envelope(root: Any, link_id: str, account_id: str, letter_id: str,
                    record: dict, states: set) -> tuple[dict | None, str]:
    """Return recorded envelope (``account_id``, ``text``, ``reply_to``) of an earlier unsettled attempt."""
    if isinstance(record, dict) and any(k in record for k in ("account_id", "text", "reply_to")):
        return record, ""
    if "uncertain" in states:
        try:
            receipts = _recorded_receipts(root, link_id, account_id, letter_id)
        except Exception:  # noqa: BLE001
            return None, "receipts_unreadable"
        if receipts:
            payload = receipts[0]
            msg = payload.get("message") if isinstance(payload.get("message"), dict) else {}
            return {"letter_id": letter_id, "account_id": str(payload.get("account_id") or ""),
                    "text": str(payload.get("text") or ""), "reply_to": str(msg.get("reply_to") or ""),
                    "origin": payload.get("origin")}, ""
    return None, ""


def _pin_prior_envelope(prior: dict | None, *, account_id: str, text: str,
                        reply_to: str) -> tuple[str, tuple[str, str] | None]:
    """Validate recipient/text against a recorded prior attempt and return the frozen ``reply_to``."""
    if not isinstance(prior, dict):
        return reply_to, None
    if prior.get("account_id") and str(prior["account_id"]) != account_id:
        return reply_to, ("retry_recipient_mismatch",
                          "this letter id already belongs to another recipient; nothing was sent")
    if "text" in prior and prior["text"] is not None and str(prior["text"]) != text:
        return reply_to, ("retry_text_mismatch",
                          "this letter id already carries different text; nothing was sent")
    pinned = str(prior["reply_to"]) if ("reply_to" in prior and prior["reply_to"] is not None) else reply_to
    return pinned, None



def _check_link_and_place(root: Any, place_id: str, link_id: str, *,
                          expected_account_id: str = "") -> tuple[str, str] | None:
    """Validate registry row (link_id, recipient, revocation, confirmation, remote_unknown) and place speech state."""
    from ouroboros import peer_links
    from ouroboros.gateways.peer import peer_outgoing_speech_refusal

    try:
        found = peer_links.link_of_place(root, place_id)
    except peer_links.PeerLinkError:
        return "peer_links_unreadable", "the link registry cannot be read now; nothing was sent"
    if found is None or found[0] != link_id:
        return "peer_link_unavailable", ("no usable link to that place here now (revoked, paused, its code "
                                         "unconfirmed or not marked mine); nothing was sent")
    row = found[1] if isinstance(found[1], dict) else {}
    if expected_account_id and str(row.get("peer_install_id") or "") != expected_account_id:
        return "retry_recipient_mismatch", "the link at this place now names another recipient; nothing was sent"
    if row.get("revoked_at"):
        return "peer_link_unavailable", "this link was revoked here; nothing was sent"
    if row.get("created_at") and not row.get("code_confirmed_at"):
        return "peer_link_unavailable", "this link's code is not confirmed here yet; nothing was sent"
    if row.get("remote_unknown_at"):
        return "link_unknown", (f"the linked Ouroboros has not known this link since {row['remote_unknown_at']}; "
                                "it needs a new introduction; nothing was sent")
    return peer_outgoing_speech_refusal(root, place_id)


def _explain_missing_outgoing(root: Any, place_id: str, link_id: str, *,
                              expected_account_id: str = "") -> tuple[str, str]:
    """Classify why ``outgoing_link`` returned None without turning transient/unproven causes into terminal."""
    from ouroboros import places

    if (refusal := _check_link_and_place(root, place_id, link_id, expected_account_id=expected_account_id)) is not None:
        return refusal
    try:
        records = places.place_records(root)
    except (places.PlaceError, OSError, ValueError):
        return "places_unreadable", "place records could not be read here; nothing was sent"
    place, peer = records.places.get(str(place_id or "")), records.peer(str(place_id or ""))
    if place is None or not peer or not peer.get("mine"):
        return "peer_link_unavailable", ("no usable link to that place here now (revoked, paused, its code "
                                         "unconfirmed or not marked mine); nothing was sent")
    if place.state == places.STATE_PAUSED:
        return "place_paused", f"{place.label or place_id} is Paused here; nothing was sent"
    if place.state == places.STATE_SILENT:
        return "place_silent", f"{place.label or place_id} is Silent here; nothing was sent"
    # Do not re-read peer_links._read_secret here: that helper collapses OSError/unavailability/corruption and absence
    # into "", so on a proven Active/mine/confirmed link an unreadable key must stay owed as peer_link_transient.
    return "peer_link_transient", "the link was temporarily unavailable before sending; nothing was sent"


def resolve_outgoing(root: Any, place_id: str, link_id: str, *,
                     expected_account_id: str = "") -> tuple[Any, tuple[str, str] | None]:
    """Resolve current outgoing send permission without collapsing Pause, Silence or unreadable stores."""
    from ouroboros import peer_channel, peer_channel_queue, peer_links

    if hold := peer_channel_queue.outgoing_switch_hold(root):
        return None, (hold[0], hold[1] + _NOTHING_SENT)
    if (refusal := _check_link_and_place(root, place_id, link_id, expected_account_id=expected_account_id)) is not None:
        return None, refusal
    link = peer_links.outgoing_link(place_id, drive_root=root)
    if link is None or link.link_id != link_id:
        return None, _explain_missing_outgoing(root, place_id, link_id, expected_account_id=expected_account_id)
    if expected_account_id and link.peer_install_id != expected_account_id:
        return None, ("retry_recipient_mismatch", "the link at this place now names another recipient; nothing was sent")
    if link.remote_unknown_at:
        return None, ("link_unknown", (f"the linked Ouroboros has not known this link since {link.remote_unknown_at}; "
                                       "it needs a new introduction; nothing was sent"))
    ok, target_code, target_msg = peer_channel.link_target_valid(link)
    return (link, None) if ok else (None, (target_code, target_msg))


def _classify_prewire_unavailable(root: Any, place_id: str, link_id: str, *,
                                  expected_account_id: str = "") -> tuple[str, str]:
    """Re-classify prewire ``peer_link_unavailable`` against fresh proven state."""
    _link, hold = resolve_outgoing(root, place_id, link_id, expected_account_id=expected_account_id)
    return hold if hold is not None else ("peer_link_transient",
                                          "the link was temporarily unavailable before sending; nothing was sent")


def check_speech_basis(root: Any, place_id: str, link_id: str, *, kind: str, approved_basis: str,
                       expected_account_id: str = "") -> tuple[str, str] | None:
    """Verify conversation rules against ``approved_basis`` without treating lawful Silence or transient holds as
    terminal ``rules_changed``."""
    from ouroboros.peer_letters import speech_basis

    current = speech_basis(root, place_id, kind=kind)
    if current and current == approved_basis:
        return None
    _link, hold = resolve_outgoing(root, place_id, link_id, expected_account_id=expected_account_id)
    if hold is not None:
        return hold
    if current:
        return "rules_changed", "the rules for this conversation changed after these words were approved; nothing was sent"
    return "rules_unreadable", "the rules for this conversation could not be read now; nothing was sent"


_check_speech_basis = check_speech_basis


def kept_outcome(earlier: str, why: str = "") -> dict:
    """The outcome of a letter an earlier attempt of which may have arrived, as this call leaves it (``why`` nothing
    new left now): unconfirmed, never "nothing was sent"."""
    tail = "an earlier attempt of this letter is unconfirmed: it may have arrived"
    if why:
        clean = why[:-len(_NOTHING_SENT)] if why.endswith(_NOTHING_SENT) else why
        tail = f"{clean}; nothing new was sent now, and {tail}"
    return {"state": "uncertain", "code": "earlier_attempt_unconfirmed", "message": tail}


def _record_receipt(root: Any, link_id: str, account_id: str, letter_id: str, state: str, *, text: str, reply_to: str,
                    origin: dict, refusal: dict | None = None) -> bool:
    """One receipt of my message in this installation's history (``PresenceDeliveryRecorder``): whether it landed."""
    from ouroboros import peer_links

    receipt = {"schema_version": 1, "delivery_id": letter_id, "part_id": MESSAGE_PART, "state": state,
               "provider": peer_links.CONNECTOR, "account_id": account_id, "conversation_id": link_id,
               "thread_id": "", "text": text, "format": "markdown",
               "message": {"message_id": letter_id, "peer_message": True, **({"reply_to": reply_to} if reply_to else {}),
                           **({"peer_refusal": refusal} if refusal else {})},
               "origin": dict(origin)}
    try:
        _receipts(root).record(peer_links.CONNECTOR, receipt)
        return True
    except Exception:  # noqa: BLE001 - the receipt is evidence; without it nothing reads as confirmed
        log.warning("peer message %s receipt not recorded", letter_id, exc_info=True)
        return False


def record_receipt(root: Any, link_id: str, account_id: str, letter_id: str, state: str, *, text: str, reply_to: str,
                   origin: dict) -> bool:
    """Public receipt writer delegating to ``_record_receipt``."""
    return _record_receipt(root, link_id, account_id, letter_id, state, text=text, reply_to=reply_to, origin=origin)


def _refusal_fields(record: dict) -> dict:
    return {"code": str(record.get("confirmed_code") or "refused_earlier"),
            "message": str(record.get("confirmed_message") or "the linked Ouroboros refused this letter")}


def reconcile_confirmed(root: Any, link_id: str, account_id: str, letter_id: str, *, earlier: str, states: set,
                        record: dict, text: str, reply_to: str, origin: dict,
                        facts: dict | None = None) -> dict:
    """Finalize an already-confirmed ``delivered`` or ``failed`` outcome without any wire POST or current-permission
    check, recording its canonical receipt from ``sent/`` if the original receipt write had failed."""
    from ouroboros import peer_links
    from ouroboros.peer_letters import audit

    if earlier in states:
        if record:
            with contextlib.suppress(OSError):
                peer_links.settle_sending(root, link_id, letter_id)
        return {"state": earlier, "letter_id": letter_id, "recorded": True, "earlier": True,
                **({"code": "refused_earlier", "message": "the linked Ouroboros refused this letter earlier (its failed "
                    "receipt is in this room); it is not sent again"} if earlier == "failed" else {})}

    rec_account = str(record["account_id"]) if record.get("account_id") else account_id
    rec_text = str(record["text"]) if ("text" in record and record["text"] is not None) else text
    rec_reply_to = str(record["reply_to"]) if ("reply_to" in record and record["reply_to"] is not None) else reply_to
    rec_origin = record.get("origin") if isinstance(record.get("origin"), dict) and record.get("origin") else origin
    failed_fields = _refusal_fields(record) if earlier == "failed" else {}
    recorded = _record_receipt(root, link_id, rec_account, letter_id, earlier,
                               text=rec_text, reply_to=rec_reply_to, origin=rec_origin, refusal=failed_fields)
    if recorded:
        with contextlib.suppress(OSError):
            peer_links.settle_sending(root, link_id, letter_id)
    if facts:
        audit(root, f"peer_message_{earlier}", receipt_recorded=recorded,
              **({"code": str(record.get("confirmed_code") or "refused_earlier")} if earlier == "failed" else {}),
              **facts)
    return {"state": earlier, "letter_id": letter_id, "recorded": recorded, "earlier": True, **failed_fields}


def _directly(attempt: Any) -> Any:
    return attempt()


def send_peer_message(root: Any, place_id: str, *, letter_id: str, text: str, reply_to: str = "", origin: dict,
                      ready: Any = None, gate: Any = None, link_id: str = "", account_id: str = "",
                      max_attempts: int = _SEND_ATTEMPTS) -> dict:
    """Physical server-only send: delivered/failed with receipt evidence, retry (receiver asked to wait),
    uncertain (earlier attempt may have arrived), waiting (channel absent), or a typed hold/unknown/busy.
    `link_id` and `account_id` bind the original link and recipient across every recheck and attempt.
    """
    from ouroboros import peer_links
    from ouroboros.utils import in_worker_process

    if in_worker_process():
        return _held(letter_id, "delivery_server_only", "only the server sends peer messages")
    bound_link, bound_account = str(link_id or "").strip(), str(account_id or "").strip()
    if not bound_link or not bound_account:
        try:
            found = peer_links.link_of_place(root, place_id)
        except peer_links.PeerLinkError:
            return _unknown(letter_id, "peer_links_unreadable", "the link registry cannot be read now; nothing was sent")
        if found is None or (bound_link and found[0] != bound_link):
            return _held(letter_id, "peer_link_unavailable", "no link to that place here; nothing was sent")
        bound_link, bound_account = bound_link or found[0], bound_account or str(found[1].get("peer_install_id") or "")
    claim = f"{pathlib.Path(root).resolve()}\0{bound_link}\0{letter_id}"
    with _SENDING_LOCK:
        if claim in _SENDING:
            return {"state": "busy", "code": "letter_in_flight", "letter_id": letter_id,
                    "message": "this letter is being sent now by another call; nothing more was sent"}
        _SENDING.add(claim)
    try:
        return _send_claimed(root, place_id, bound_link, bound_account, letter_id=letter_id, text=text,
                             reply_to=reply_to, origin=origin, ready=ready, gate=gate, max_attempts=max_attempts)
    finally:
        with _SENDING_LOCK:
            _SENDING.discard(claim)


def _send_claimed(root: Any, place_id: str, link_id: str, account_id: str, *, letter_id: str, text: str,
                  reply_to: str, origin: dict, ready: Any, gate: Any, max_attempts: int) -> dict:
    from ouroboros import peer_channel_queue, peer_links
    from ouroboros.gateways.peer import NOTHING_SENT, PeerGateway
    from ouroboros.peer_letters import audit

    earlier, states, record, read_error = earlier_state(root, link_id, account_id, letter_id)
    if earlier in {"delivered", "failed"}:
        return reconcile_confirmed(root, link_id, account_id, letter_id, earlier=earlier, states=states,
                                   record=record, text=text, reply_to=reply_to, origin=origin)
    if earlier == "unknown":
        return _unknown(letter_id, read_error or "earlier_attempts_unknown", "earlier attempts of this letter cannot "
                        "be read now; nothing was sent now, and whether an earlier attempt arrived is not known here")
    unconfirmed, passes = earlier == "uncertain", int(record.get("passes") or 0)
    retained_pre_tx = bool(record.get("pre_transmission")) and not passes
    if unconfirmed or retained_pre_tx:
        prior, prior_err = _prior_envelope(root, link_id, account_id, letter_id, record, states)
        if prior_err:
            return _unknown(letter_id, prior_err, "earlier attempts of this letter cannot be read now; nothing was sent")
        reply_to, mismatch = _pin_prior_envelope(prior, account_id=account_id, text=text, reply_to=reply_to)
        if mismatch is not None:
            return _held(letter_id, *mismatch)
        if prior and isinstance(prior.get("origin"), dict) and prior["origin"]:
            origin = prior["origin"]  # an approved automatic output may adopt the tool letter, not its authorship

    facts = {"place_id": str(place_id), "link_id": link_id, "letter_id": letter_id,
             "task_id": str(origin.get("task_id") or ""), **({"reply_to": reply_to} if reply_to else {})}
    outcome = kept_outcome(earlier) if unconfirmed else {}
    learned, transmitted, started = False, False, 0

    def held(code: str, message: str) -> None:
        nonlocal outcome
        if not outcome or not transmitted:
            outcome = prewire_hold(letter_id, earlier, code, message)
        elif code in STILL_OWED:
            outcome = {**outcome, "held_code": code}

    def rollback() -> None:
        nonlocal started
        if started and not transmitted:
            with contextlib.suppress(OSError, peer_links.PeerLinkError):
                peer_links.rollback_sending(root, link_id, letter_id, expected_passes=started,
                                            keep_record=unconfirmed or retained_pre_tx,
                                            pre_transmission=retained_pre_tx)
            started = 0

    def final_ready() -> Any:
        link, refusal = resolve_outgoing(root, place_id, link_id, expected_account_id=account_id)
        if refusal is None and ready is not None:
            refusal = ready()
        if refusal is not None:
            raise AttemptHeld(*refusal)
        return link

    def attempt() -> None:
        nonlocal started
        # Re-read after launch admission (which may have waited), then again after the durable pre-POST write.
        final_ready()
        if not started:
            try:
                started = peer_links.record_sending(root, link_id, letter_id, account_id=account_id,
                                                    place_id=str(place_id), text=text, reply_to=reply_to, origin=origin)
            except peer_links.PeerLinkError as exc:
                code = exc.code if exc.code == "sending_identity_mismatch" else "sending_unrecorded"
                raise AttemptHeld(code, str(exc)) from exc
        gateway = PeerGateway(final_ready())
        try:
            gateway.send_message(text, letter_id=letter_id, reply_to=reply_to, timeout_sec=_ATTEMPT_SEC)
        finally:
            gateway.close()

    if passes >= _MAX_PASSES:
        held("attempts_spent", f"this letter went out in {passes} calls without the linked Ouroboros confirming it")
        outcome["code"] = "attempts_spent"
    else:
        for number in range(max_attempts):
            try:
                (gate or _directly)(attempt)
                outcome, learned = {"state": "delivered"}, True
                break
            except AttemptHeld as exc:
                rollback()
                held(exc.code, str(exc))
                break
            except Exception as exc:  # every untyped wire outcome remains uncertain
                code, status = str(getattr(exc, "code", "") or ""), int(getattr(exc, "status_code", 0) or 0)
                refused = str(getattr(exc, "refusal_code", "") or "")
                if getattr(exc, "channel_connect_failed", False):
                    rollback()
                    if not outcome:
                        return {"state": "waiting", "code": "channel_unavailable", "message": str(exc),
                                "letter_id": letter_id, "recorded": False, "channel_wait": True}
                    outcome = {**outcome, "channel_wait": True}
                    break
                if code in NOTHING_SENT or code == "invalid_letter" or (status == 0 and code.startswith("place_")):
                    rollback()
                    msg = str(exc)
                    if code == "peer_link_unavailable":
                        code, msg = _classify_prewire_unavailable(root, place_id, link_id, expected_account_id=account_id)
                    held(code, msg)
                    break
                broken_channel = bool(getattr(exc, "channel_transport_error", False))
                if broken_channel and started and not transmitted:
                    # Transport loss may have reached the receiver, but it does not spend the
                    # retry allowance for a receiver that is offline. Keep the original envelope.
                    with contextlib.suppress(OSError, peer_links.PeerLinkError):
                        if peer_links.rollback_sending(root, link_id, letter_id, expected_passes=started,
                                                       keep_record=True):
                            started -= 1
                transmitted = True
                if status == 429:
                    retry_after = getattr(exc, "retry_after", None)
                    outcome = ({**outcome, "retry_code": refused or code, "retry_after": retry_after} if outcome else
                               {"state": "retry", "code": refused or code, "message": str(exc), "retry_after": retry_after})
                    break
                if refused and 400 <= status < 500:
                    if not outcome:
                        outcome, learned = {"state": "failed", "code": refused, "message": str(exc)}, True
                    break
                outcome, learned = {"state": "uncertain", "code": code or "unconfirmed", "message": str(exc)}, True
                if broken_channel:
                    outcome["channel_wait"] = True
                    break
                if number + 1 < max_attempts:
                    time.sleep(_RETRY_PAUSE_SEC * (number + 1))

    if outcome["state"] == "held":
        return outcome
    spent = started >= _MAX_PASSES and outcome["state"] in {"uncertain", "retry"}
    if outcome["state"] == "retry":
        if not spent:
            return {**outcome, "letter_id": letter_id}
        outcome, learned = {"state": "uncertain", "code": "attempts_spent", "message": outcome["message"]}, True
    if spent:
        outcome["spent"] = True
    refusal = {k: outcome[k] for k in ("code", "message")} if outcome["state"] == "failed" else None
    recorded = (not learned and "uncertain" in states) or _record_receipt(
        root, link_id, account_id, letter_id, outcome["state"], text=text, reply_to=reply_to, origin=origin,
        refusal=refusal)
    if outcome["state"] in {"delivered", "failed"}:
        if recorded:
            with contextlib.suppress(OSError):
                peer_links.settle_sending(root, link_id, letter_id)
        else:
            with contextlib.suppress(OSError, peer_links.PeerLinkError):
                peer_links.confirm_sending(root, link_id, letter_id, state=outcome["state"], account_id=account_id,
                                           place_id=str(place_id), text=text, reply_to=reply_to, origin=origin,
                                           code=str(outcome.get("code") or ""), message=str(outcome.get("message") or ""))
        if outcome["state"] == "delivered":
            peer_channel_queue.note_verified_exchange(root, link_id)
    audit(root, f"peer_message_{outcome['state']}", receipt_recorded=recorded,
          **({"code": outcome["code"]} if outcome.get("code") else {}), **facts)
    return {**outcome, "letter_id": letter_id, "recorded": recorded}


def deliver_owed(root: Any, task_id: str, record: Any = None) -> None:
    """Hand one producer's owed speech to its server coordinator; a parked output keeps the same stable ID."""
    from ouroboros import peer_channel_queue
    from ouroboros.install_identity import acting_as
    from ouroboros.task_results import load_task_result

    root, task_id = pathlib.Path(root), str(task_id or "")
    try:
        with acting_as(root):
            stored = record if isinstance(record, dict) else (load_task_result(root, task_id) or {}) if task_id else {}
            if task_id and stored.get("status") in {"completed", "failed", "cancelled"}:
                peer_channel_queue.retire_parked_messages(root, task_id)
            successor = str(stored.get("presence_retry_next") or "")
            place = _peer_speaker(root, stored) or (_peer_speaker(root, load_task_result(root, successor) or {})
                                                    if successor else None)
            if not task_id or place is None:
                return
        peer_channel_queue.trigger_link_flush(root, place.conversation_id, external=True)
    except Exception:  # noqa: BLE001 - a producer's end never fails over its delivery
        log.warning("peer reply of %s was not handed on", task_id, exc_info=True)


def _one_pass(root: pathlib.Path, link_id: str, ref: str, gate: Any = None) -> tuple[bool, float | None]:
    from ouroboros import peer_links
    from ouroboros.install_identity import acting_as

    guard = gate if gate is not None else contextlib.nullcontext()
    with guard, acting_as(root):  # read and write as the install whose link it answers; never held across sleeps
        seen = peer_links.owed_version(root, link_id, ref)
        return _deliver_producer(root, ref, link_id=link_id, seen=seen)


def _deliver_batch(root: pathlib.Path, refs: list[tuple[str, str]], *, gate: Any = None) -> bool:
    """Pass every eligible ref before waiting for the earliest remaining per-letter deadline."""
    from ouroboros.peer_channel_queue import deliver_until_idle

    outcomes = {}

    def delivery_pass() -> float | None:
        wait = None
        for link_id, ref in refs:
            if outcomes.get((link_id, ref)):
                continue
            settled, after = _one_pass(root, link_id, ref, gate)
            outcomes[(link_id, ref)] = settled
            wait = _sooner(wait, _deadline_at(after))
        return wait

    deliver_until_idle(delivery_pass, threading.Event())
    return all(outcomes.values())


def _deliver_until_settled(root: pathlib.Path, link_id: str, ref: str) -> bool:
    return _deliver_batch(root, [(link_id, ref)])


def resume_owed(root: Any, *, link_id: str = "", at_start: bool = True) -> None:
    """Reconcile addressed ``sending_records`` (at start) and wake each link's single coordinator."""
    from ouroboros import peer_channel_queue, peer_links
    from ouroboros.place_projection import switch_state

    root = pathlib.Path(root)
    try:
        owed, sending = peer_links.owed_refs(root, link_id), peer_links.sending_records(root) if at_start else []
        queued = list(dict.fromkeys(lid for lid, _row in peer_channel_queue.queued_messages(root, link_id)))
    except Exception:  # noqa: BLE001 - an unreadable registry is unknown: its records wait for a later start
        log.warning("owed peer replies of %s could not be read now", root, exc_info=True)
        return
    open_sw = switch_state(root) in {"on", "off"}
    owed, queued = (owed if open_sw else []), (queued if open_sw else [])
    if owed or sending or queued:
        resume_args = (root, owed, sending, queued) if queued else (root, owed, sending)
        threading.Thread(target=_resume_quietly, args=resume_args,
                         name="peer-replies-owed", daemon=True).start()


def resume_returned(root: Any, *, link_id: str = "") -> None:
    """Resume owed replies after my human's confirmed return (place Active again or end of Panic)."""
    try:
        resume_owed(root, link_id=link_id, at_start=False)
    except Exception:  # noqa: BLE001 - my human's act never fails over its follow-up
        log.warning("owed peer replies of %s were not resumed after a return", root, exc_info=True)


def _reconcile_sending(root: pathlib.Path, sending: list) -> None:
    """Reconcile unsettled ``sent/`` records without POSTing; skip live owners and provably unsent shells."""
    from ouroboros import peer_links
    from ouroboros.presence_delivery import recorded_states

    for link_id, record, revoked in sending:
        letter_id, account_id = str(record.get("letter_id") or ""), str(record.get("account_id") or "")
        confirmed = str(record.get("confirmed_state") or "")
        confirmed = confirmed if confirmed in {"delivered", "failed"} else ""
        try:
            states = recorded_states(root, peer_links.CONNECTOR, account_id, letter_id, MESSAGE_PART)
            origin = record.get("origin") if isinstance(record.get("origin"), dict) else {}
            text, reply_to = str(record.get("text") or ""), str(record.get("reply_to") or "")
            if confirmed:
                if (confirmed in states or (confirmed == "failed" and "delivered" in states)
                        or _record_receipt(root, link_id, account_id, letter_id, confirmed,
                                           text=text, reply_to=reply_to, origin=origin,
                                           refusal=_refusal_fields(record) if confirmed == "failed" else None)):
                    peer_links.settle_sending(root, link_id, letter_id)
                continue
            said = bool(states & {"delivered", "failed"})
            if bool(record.get("pre_transmission")) and int(record.get("passes") or 0) == 0 and "uncertain" not in states:
                if said or revoked:
                    peer_links.settle_sending(root, link_id, letter_id)
                continue
            if not said and "uncertain" not in states:
                said = _record_receipt(root, link_id, account_id, letter_id, "uncertain",
                                                    text=text, reply_to=reply_to, origin=origin) and revoked
            if said or (revoked and "uncertain" in states):
                peer_links.settle_sending(root, link_id, letter_id)
        except Exception:  # noqa: BLE001 - one record's failure leaves the others and keeps it for a later start
            log.warning("sending record of %s was not reconciled", letter_id, exc_info=True)


def _resume_quietly(root: pathlib.Path, owed: list, sending: list = (), queued: list = ()) -> None:
    from ouroboros.install_identity import acting_as

    if sending:
        with acting_as(root):
            _reconcile_sending(root, list(sending))
    from ouroboros import peer_channel_queue

    for link_id in dict.fromkeys([*queued, *(lid for lid, _ref in owed)]):
        peer_channel_queue.trigger_link_flush(root, link_id)


def _peer_speaker(root: pathlib.Path, stored: Any) -> Any:
    """The peer place a stored producer speaks to with v1 receipts, else None: most tasks, from their record alone."""
    from ouroboros import peer_links
    from ouroboros.places import place_records

    metadata = stored.get("metadata") if isinstance(stored, dict) else None
    presence = metadata.get("presence") if isinstance(metadata, dict) else None
    if (not isinstance(presence, dict) or presence.get("delivery_reporting_version") != 1
            or not str(presence.get("place_id") or "")):
        return None
    place = place_records(root).places.get(str(presence["place_id"]))
    return place if place is not None and place.connector == peer_links.CONNECTOR else None


def _no_live_turn(root: pathlib.Path, ref: str) -> bool:
    """No attempt of ``ref`` runs here now (the Host's execution of it or its live turn), so a newer attempt's owed
    record is never the one a finished pass removes (``peer_links.settle_owed``)."""
    from ouroboros.gateway.host_service_places import host_context
    from ouroboros.presence_runner import presence_turn_is_live

    return host_context(root).presence_turns.find(ref) is None and not presence_turn_is_live(ref, drive_root=root)


def _deadline_at(after: float | None) -> float | None:
    return time.monotonic() + after if after is not None else None


def _sooner(wait: float | None, after: float | None) -> float | None:
    return after if wait is None else wait if after is None else min(wait, after)


def _deliver_producer(root: pathlib.Path, task_id: str, visited: set | None = None, *, link_id: str,
                      seen: tuple | None) -> tuple[bool, float | None]:
    """Send what one producer owes its link now, from its own retained record (``_producer_view``: its words as
    recorded, never the outward view a place's state gives them for the while it lasts): the speech it released (a
    parked author's ``outputs``, pending or not) and, once it ended, its terminal output, each once under its stable
    output identity, addressed to the message it answers (``_automatic_reply_target``)."""
    from ouroboros import peer_links
    from ouroboros.presence_bindings import conversation_key
    from ouroboros.safety_config import coverage_kind
    from ouroboros.task_results import load_task_result

    visited = set() if visited is None else visited
    ref, stored = task_id, {}
    while task_id and task_id not in visited:  # a certified retry's successor answers the event (``presence_retry_next``)
        visited.add(task_id)
        stored = load_task_result(root, task_id, strict=True) or {}  # unreadable raises: unknown, the record stays
        if not (successor := str(stored.get("presence_retry_next") or "")):
            break
        task_id = successor
    if not stored:
        return False, None  # no row under this reference yet: unknown, never "nothing owed"
    place = _peer_speaker(root, stored)
    if place is None or place.conversation_id != link_id:  # its own row speaks to no place of this link
        _settle(root, link_id, ref, seen)
        return True, None
    handoff = _handoff(root, stored, place)
    if handoff is not None and not handoff[1]:
        return False, None  # its turn has not ended: that end answers it, under the input its exit answered
    if handoff is not None and not _named_exit(handoff[0], stored):  # the turn's end named other work, or none
        _settle(root, link_id, ref, seen)
        return True, None
    status, view = _producer_view(root, stored, task_id)
    metadata = stored.get("metadata") or {}
    event = (metadata.get("presence") or {}).get("event")
    event = event if isinstance(event, dict) else {}
    reply_to = _automatic_reply_target(stored, event, handoff)
    key, settled, wait = conversation_key(place.provider, place.account_id, place.conversation_id, ""), True, None
    for output in [*(view.get("outputs") or ()), *([view] if status == 200 else [])]:
        text = str(output.get("text") or "")
        if text and output.get("outcome", "message") in {"message", "deferred"}:
            done, after = _deliver_output(root, place, key, producer=str(view.get("task_id") or task_id), text=text,
                                          output_ref=str(output.get("output_ref") or f"{task_id}:terminal"),
                                          reply_to=reply_to, source_event_id=str(event.get("source_event_id") or ""),
                                          kind=coverage_kind(metadata))
            settled, wait = settled and done, _sooner(wait, _deadline_at(after))
    if status != 200:
        return False, _remaining(wait)
    if (child := str(view.get("child_work_ref") or "")) and child not in visited:  # its deferred exit, named by itself
        peer_links.record_owed(root, link_id, child, place_id=place.place_id)  # before its end can miss it
        child_seen = peer_links.owed_version(root, link_id, child)
        if str((load_task_result(root, child, strict=True) or {}).get("status") or "") in {"completed", "failed",
                                                                                         "cancelled"}:
            wait = _sooner(wait, _deadline_at(_deliver_producer(root, child, visited, link_id=link_id,
                                                             seen=child_seen)[1]))
    if settled:
        _settle(root, link_id, ref, seen)
    return settled, _remaining(wait)


def _remaining(deadline: float | None) -> float | None:
    return max(0.0, deadline - time.monotonic()) if deadline is not None else None


def _settle(root: pathlib.Path, link_id: str, ref: str, seen: tuple | None) -> None:
    """``ref``'s owed record leaves only as this pass read it and while no attempt of it runs here."""
    from ouroboros import peer_links

    peer_links.settle_owed(root, link_id, ref, seen=seen, unowed=lambda: _no_live_turn(root, ref))


def _producer_view(root: pathlib.Path, stored: dict, task_id: str) -> tuple[int, dict]:
    """``presence_recovery.retained_work_view`` of a producer for its link: an ended record's own words as recorded
    (read without the place's present outward view, ``place_reply.project_reply``, which a Pause or Silence of the
    place would empty for its while and say there), a live one's status as this installation observes it."""
    from ouroboros.presence_recovery import retained_work_view
    from ouroboros.presence_runner import _terminal_refusal

    if str(stored.get("status") or "") in {"completed", "failed", "cancelled"} and not _terminal_refusal(stored):
        return retained_work_view(stored, task_id, None)
    return retained_work_view(stored, task_id, root)


def _handoff(root: pathlib.Path, stored: dict, place: Any) -> tuple[dict, bool] | None:
    """The turn that handed this work off to answer the same place (the task that drafted it, ``objective_author``,
    speaking there itself), with whether its end is recorded; None when this work is no such handoff."""
    from ouroboros.presence_recovery import attempt_ended
    from ouroboros.task_results import load_task_result

    metadata = stored.get("metadata") if isinstance(stored.get("metadata"), dict) else {}
    author = metadata.get("objective_author") if isinstance(metadata.get("objective_author"), dict) else {}
    turn_id = str(author.get("task_id") or "") if author.get("kind") == "task" else ""
    turn = ((load_task_result(root, turn_id, strict=True) or {})  # unreadable raises: unknown, nothing goes now
            if turn_id and turn_id != str(stored.get("task_id") or "") else {})
    turn_place = ((turn.get("metadata") or {}).get("presence") or {}).get("place_id") if turn else None
    if not turn or turn_place != place.place_id:
        return None
    return turn, str(turn.get("status") or "") in {"completed", "failed", "cancelled"} or attempt_ended(turn, root)


def _named_exit(turn: dict, stored: dict) -> bool:
    """The turn's own record names this work as its deferred exit (``presence_work_ref``)."""
    handed = turn.get("metadata") if isinstance(turn.get("metadata"), dict) else {}
    return str(handed.get("presence_work_ref") or "") == str(stored.get("task_id") or "\0")


def _automatic_reply_target(stored: dict, event: dict, handoff: tuple[dict, bool] | None) -> str:
    """The message an automatic output answers: its producer's own selected input (``presence_reply_anchor``); for
    work a turn handed off, the input that turn's deferred exit answered once its terminal names this work; else the
    producer's waking message."""
    metadata = stored.get("metadata") if isinstance(stored.get("metadata"), dict) else {}
    anchor = metadata.get("presence_reply_anchor")
    if not (isinstance(anchor, dict) and isinstance(anchor.get("message"), dict)) and handoff is not None \
            and _named_exit(handoff[0], stored):
        anchor = (handoff[0].get("metadata") or {}).get("presence_reply_anchor")
    message = anchor.get("message") if isinstance(anchor, dict) and isinstance(anchor.get("message"), dict) else {}
    return str(message.get("message_id") or (event.get("message") or {}).get("message_id") or "")


def _tool_sent(root: pathlib.Path, producer: str, key: str, text: str, reply_to: str) -> bool:
    """These exact words already left through the producer's own tool send into this conversation as its answer to
    the same message (its delivered or accepted receipt there, naming that ``reply_to``)."""
    from ouroboros.presence_runner import _live_task_rows

    for row in _live_task_rows(root, producer, key):
        transport = row.get("transport") if isinstance(row.get("transport"), dict) else {}
        message = transport.get("message") if isinstance(transport.get("message"), dict) else {}
        if (row.get("type") == "presence_delivery" and (transport.get("origin") or {}).get("kind") == "tool"
                and (transport.get("delivery") or {}).get("state") in {"delivered", "accepted"}
                and row.get("text") == text and str(message.get("reply_to") or "") == reply_to):
            return True
    return False


def _said_held(root: pathlib.Path, key: str, letter_id: str) -> bool:
    """An automatic output's one hold is already said in its room: it stays its end."""
    from ouroboros import dialogue_store
    from ouroboros.places import conversation_chat_id
    return bool(dialogue_store.rows_by_client_id(root, f"place-notice:{letter_id}:held",
                                                 chat_id=conversation_chat_id(key), limit=1))


def _deliver_output(root: pathlib.Path, place: Any, key: str, *, producer: str, output_ref: str, text: str,
                    reply_to: str, source_event_id: str, kind: str) -> tuple[bool, float | None]:
    """One automatic output, one end: a confirmed delivery or refusal (in canonical receipts or preserved on ``sent/``)
    reconciles its receipt before any current switch/pause/rules check and sends nothing. Unreadable stores, Panic,
    unreadable switch, and lawful Pause or Silence keep an unsent reply owed without forgetting it; genuine rule
    changes, unrecorded approvals, or unusable/revoked links hold it honestly once."""
    from ouroboros import peer_channel_queue, peer_links
    from ouroboros.place_reply import record_notice

    letter_id = message_letter_id(place.conversation_id, output_ref)
    origin = {"kind": "automatic", "task_id": producer, "source_event_id": source_event_id}
    earlier, states, record, _read_error = earlier_state(root, place.conversation_id, place.account_id, letter_id)
    if earlier in {"delivered", "failed"}:
        done = reconcile_confirmed(root, place.conversation_id, place.account_id, letter_id, earlier=earlier,
                                   states=states, record=record, text=text, reply_to=reply_to, origin=origin)
        return (True, None) if done.get("recorded") else (False, None)
    if earlier == "unknown":
        return False, None
    if earlier == "uncertain":
        prior, prior_err = _prior_envelope(root, place.conversation_id, place.account_id, letter_id, record, states)
        if prior_err:
            return False, None
        reply_to, mismatch = _pin_prior_envelope(prior, account_id=place.account_id, text=text, reply_to=reply_to)
        if mismatch is not None:
            return _hold_output(root, place, key, letter_id, states, text=text, reply_to=reply_to, origin=origin,
                                why=mismatch[1])
    if _said_held(root, key, letter_id):
        return True, None
    if earlier == "none":
        if peer_channel_queue.matching_queued(root, place.conversation_id, producer=producer, text=text,
                                              reply_to=reply_to):
            return False, None  # the same link pass already owns that submission and its deadline
        tool_id = message_letter_id(place.conversation_id, producer, reply_to, text)
        t_earlier, t_states, t_rec, _t_err = earlier_state(root, place.conversation_id, place.account_id, tool_id)
        if t_earlier == "delivered":
            done = reconcile_confirmed(root, place.conversation_id, place.account_id, tool_id, earlier="delivered",
                                       states=t_states, record=t_rec, text=text, reply_to=reply_to, origin=origin)
            return (True, None) if done.get("recorded") else (False, None)
        if t_earlier == "unknown":
            return False, None
        if t_earlier == "uncertain" or t_rec.get("pre_transmission"):
            prior, prior_err = _prior_envelope(root, place.conversation_id, place.account_id, tool_id, t_rec, t_states)
            if prior_err:
                return False, None
            reply_to, mismatch = _pin_prior_envelope(prior, account_id=place.account_id, text=text, reply_to=reply_to)
            if mismatch is not None:
                return _hold_output(root, place, key, tool_id, t_states, text=text, reply_to=reply_to, origin=origin,
                                    why=mismatch[1])
            if _said_held(root, key, tool_id):
                return True, None
            letter_id, earlier, states, record = tool_id, t_earlier, t_states, t_rec
        if _tool_sent(root, producer, key, text, reply_to):
            return True, None
    _link, pre_hold = resolve_outgoing(root, place.place_id, place.conversation_id, expected_account_id=place.account_id)
    if pre_hold is not None:
        if pre_hold[0] in STILL_OWED:
            return False, None
        return _hold_output(root, place, key, letter_id, states, text=text, reply_to=reply_to, origin=origin,
                            why=pre_hold[1])
    try:
        owed_row = peer_links.owed_record(root, place.conversation_id, producer)
    except peer_links.PeerLinkError:
        return False, None
    approved = (owed_row or {}).get("approved")
    basis = approved.get(hashlib.sha256(text.encode("utf-8")).hexdigest()) if isinstance(approved, dict) else None
    if not basis:
        return _hold_output(root, place, key, letter_id, states, text=text, reply_to=reply_to, origin=origin,
                            why="the rules these words were approved under are not recorded; nothing was sent")
    basis_hold = check_speech_basis(root, place.place_id, place.conversation_id, kind=kind, approved_basis=basis,
                                    expected_account_id=place.account_id)
    if basis_hold is not None:
        if basis_hold[0] in STILL_OWED:
            return False, None
        return _hold_output(root, place, key, letter_id, states, text=text, reply_to=reply_to, origin=origin,
                            why=basis_hold[1])

    eligible, wait, _at = peer_channel_queue.deadline(root, place.conversation_id, letter_id, record)
    if not eligible:
        return False, wait

    def ready() -> tuple[str, str] | None:
        return check_speech_basis(root, place.place_id, place.conversation_id, kind=kind, approved_basis=basis,
                                  expected_account_id=place.account_id)

    sent = send_peer_message(root, place.place_id, letter_id=letter_id, text=text, reply_to=reply_to, ready=ready,
                             origin=origin, link_id=place.conversation_id, account_id=place.account_id)
    state, code, held = sent["state"], str(sent.get("code") or ""), str(sent.get("held_code") or "")
    if state in {"delivered", "failed"}:
        return (True, None) if sent.get("recorded") else (False, None)
    if state in {"busy", "unknown", "waiting"} or sent.get("channel_wait") or code in STILL_OWED or held in STILL_OWED:
        return False, None
    last = bool(sent.get("spent")) or code == "attempts_spent"
    if state in {"retry", "uncertain"} and not last and not held:
        receiver_wait = float(sent.get("retry_after") or 0.0)
        wait, _at = peer_channel_queue.delay_letter(root, place.conversation_id, letter_id, record,
                                                  receiver_wait=receiver_wait, place_id=place.place_id,
                                                  account_id=place.account_id, task_id=producer, text=text)
        return False, wait
    why = str(sent.get("message") or code)
    if last:  # its last pass: said once, and nothing more is sent automatically
        record_notice(root, place_id=place.place_id, conversation_key=key, task_id=producer,
                      text=f"Reply not confirmed: {why}; nothing more is sent automatically",
                      identity=f"{letter_id}:held", held_text=text)
        return True, None
    record_notice(root, place_id=place.place_id, conversation_key=key, task_id=producer, text=f"Reply held: {why}",
                  identity=f"{letter_id}:held", held_text=text)
    return True, None


def _hold_output(root: pathlib.Path, place: Any, key: str, letter_id: str, states: set, *, text: str, reply_to: str,
                 origin: dict, why: str) -> tuple[bool, float | None]:
    """An automatic output held before any POST of this pass: its one hold, said in its room with its words beside it.
    A confirmed earlier outcome reconciles its receipt instead of holding; an unreadable store keeps the reply owed;
    an earlier unconfirmed attempt is said as ``uncertain`` first and never called unsent."""
    from ouroboros.place_reply import record_notice

    earlier, cur_states, record, _read_error = earlier_state(root, place.conversation_id, place.account_id, letter_id)
    if earlier in {"delivered", "failed"}:
        done = reconcile_confirmed(root, place.conversation_id, place.account_id, letter_id, earlier=earlier,
                                   states=cur_states, record=record, text=text, reply_to=reply_to, origin=origin)
        return (True, None) if done.get("recorded") else (False, None)
    if earlier == "unknown":
        return False, None
    if earlier == "uncertain" and "uncertain" not in (states | cur_states):
        sent_as = record.get("origin") if isinstance(record.get("origin"), dict) else {}
        rec_text = str(record["text"]) if ("text" in record and record["text"] is not None) else text
        rec_reply_to = str(record["reply_to"]) if ("reply_to" in record and record["reply_to"] is not None) else reply_to
        if not _record_receipt(root, place.conversation_id, place.account_id, letter_id, "uncertain",
                               text=rec_text, reply_to=rec_reply_to, origin=sent_as or origin):
            return False, None
    if earlier == "uncertain":
        why += " now, and an earlier attempt of these words is unconfirmed: it may have arrived"
    record_notice(root, place_id=place.place_id, conversation_key=key, task_id=str(origin.get("task_id") or ""),
                  text=f"Reply held: {why}", identity=f"{letter_id}:held", held_text=text)
    return True, None
