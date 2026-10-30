"""Named ingress acceptance: one canonical inbound row per message id, its dispatch an obligation.

The owner of a named owner/skill message's acceptance (ARCHITECTURE 04 "WebSocket chat", 12
"Operation correlation"). The accepted row, its receipt and the dispatch it owes are ONE
transaction in ``state/dialog.sqlite`` (``dialogue_store.append``): the row lands with the
obligation ``pending``, ``claim_dispatch`` is the durable "entered" mark taken by exactly one
caller -- the acceptor, a same-id retry, or ``recover_pending_dispatches`` after a crash -- and
``complete_dispatch`` records the handoff's return. A same-id redelivery rejoins the accepted row
when its words and ordered attachment content match and is refused otherwise; it dispatches only
by taking a dispatch still owed (a crash between the commit and the handoff), never by guessing.
``_INGRESS_LOCK`` serializes check -> row -> dispatch inside one process; the store's UNIQUE index
settles the race between processes. Which row an id names is the store's answer
(``dialogue_store.accepted_row``): while the chat archive is still importing, a miss is checked
in the segments not yet imported, so unknown is never "new". ``supervisor.message_bus``
re-exports the public names; its ``DATA_DIR`` and ``accept_chat`` are read at call time.

Dispatch facts are durable now, so an ended process's row keeps its state: ``pending`` means
the handoff never happened (the recovery or a retry performs it once), ``dispatched`` means it
began and its outcome is ambiguous (never replayed), ``completed`` that it returned. The old
"undispatched" proof (a write that raised after landing) cannot exist: the write is atomic.
"""

from __future__ import annotations

import functools
import logging
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from ouroboros import dialogue_store
from ouroboros.chat_uploads import attachment_views, same_message, stored_refs
from ouroboros.store import StoreError
from ouroboros.subagent_messages import CARD_ROW_PLACEMENTS, SUBAGENT_MESSAGE_FIELDS
from ouroboros.utils import utc_now_iso

log = logging.getLogger(__name__)

_INGRESS_LOCK = threading.Lock()
# The dispatchers an obligation can name: replayable from the recorded input and the stored row.
_DISPATCHERS: Dict[str, Callable[..., Any]] = {}


def _bus_dispatch(bridge: Any, text: str, **message: Any) -> None:
    bridge.enqueue_local_message(text, **message)


def _dispatchers() -> Dict[str, Callable[..., Any]]:
    if not _DISPATCHERS:
        from ouroboros.server_control import dispatch_accepted_restart

        _DISPATCHERS.update({"bus": _bus_dispatch, "accepted_restart": dispatch_accepted_restart})
    return _DISPATCHERS


def dispatcher_name(dispatch: Any) -> Optional[str]:
    """The name an obligation records for ``dispatch``: the bus (``None``), the host's accepted
    Restart dispatcher, or ``None`` for a caller's own callback (dispatched now, never replayed)."""
    if dispatch is None:
        return "bus"
    target = dispatch.func if isinstance(dispatch, functools.partial) else dispatch
    return next((name for name, known in _dispatchers().items() if known is target and name != "bus"), None)


def obligation(name: Optional[str], text: str, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The recorded input of a replayable dispatch (bytes ride as the base64 text they already are)."""
    if name is None:
        return None
    return {"dispatcher": name, "text": text,
            "message": {key: value for key, value in message.items() if key not in ("accepted_source_ref", "accepted_source_row")}}


def _root(drive_root: Any = None) -> Optional[Path]:
    from supervisor import message_bus

    root = drive_root if drive_root is not None else message_bus.DATA_DIR
    return None if root is None else Path(root)


def _named(row: Dict[str, Any]) -> Tuple[Optional[int], str]:
    try:
        chat_id = int(row.get("chat_id")) if row.get("chat_id") not in (None, "") else None
    except (TypeError, ValueError):
        chat_id = None
    return chat_id, str(row.get("client_message_id") or "")


def _process() -> str:
    from ouroboros.process_custody import current_custody_session_id

    return current_custody_session_id()


def dispatch_facts(row: Dict[str, Any], drive_root: Any = None) -> Optional[Dict[str, Any]]:
    """The row's durable dispatch facts (``dialogue_store.dispatch_facts``): ``state``, the
    ``process`` that took it, its ``dispatcher``; ``None`` for a row that never owed one."""
    root = _root(drive_root)
    if root is None or not isinstance(row, dict):
        return None
    chat_id, client_message_id = _named(row)
    return dialogue_store.dispatch_facts(root, chat_id, client_message_id)


def _entered_here(facts: Optional[Dict[str, Any]]) -> bool:
    return bool(facts) and facts["state"] in ("dispatched", "completed") and facts["process"] == _process()


def dispatch_entered(row: dict, drive_root: Any = None) -> bool:
    """The handoff began in THIS process (whose queue holds it); another process's entry is
    unknown here, as it was before the store."""
    return _entered_here(dispatch_facts(row, drive_root))


def dispatch_completed(row: dict, drive_root: Any = None) -> bool:
    facts = dispatch_facts(row, drive_root)
    return _entered_here(facts) and facts["state"] == "completed"


def dispatch_taken_over_here(row: dict, drive_root: Any = None) -> bool:
    """THIS process took the row's dispatch after another process accepted it (a retry of an owed
    dispatch, or the recovery): the work is live here whatever session accepted the row."""
    facts = dispatch_facts(row, drive_root)
    return _entered_here(facts) and facts["process"] != str((row or {}).get("ingress_process") or "")


def dispatch_lost(row: dict, drive_root: Any = None) -> bool:
    """The row's dispatch began in a process that ended: its queue died with it. A row still
    ``pending`` is not lost (the recovery or a retry performs it); a legacy row without a dispatch
    fact is lost when the process that accepted it ended."""
    facts = dispatch_facts(row, drive_root)
    if facts is None:
        return bool((row or {}).get("ingress_process")) and not accepted_here(row)
    return facts["state"] in ("dispatched", "completed") and facts["process"] != _process()


def accepted_here(row: dict) -> bool:
    """True when THIS host process wrote that named acceptance row (``accept_chat`` stamps the process
    generation as ``ingress_process``). This proves acceptance only; the dispatch fact is the row's."""
    stamp = str((row or {}).get("ingress_process") or "")
    return bool(stamp) and stamp == _process()


def delivery_facts(row: dict, chat_id: int) -> dict:
    """What the store knows of an accepted row's dispatch, as its echo and history row say it.

    ``ingress_pending``: the dispatch is still owed (the acceptor is between the commit and the
    handoff, or a crash left it: the recovery or a same-id retry performs it once), or this
    process accepted a row for its own dispatcher and has not entered it yet;
    ``ingress_dispatched``: the handoff began in THIS process. Nothing: a row that owes no bus
    dispatch, an imported one, an abandoned obligation, or a dispatch an ended process took
    (its outcome is unknown here, as before the store)."""
    facts = dispatch_facts({**(row or {}), "chat_id": chat_id})
    if facts is not None and (facts["state"] == "pending" or (facts["state"] == "none" and accepted_here(row))):
        return {"ingress_pending": True}
    if _entered_here(facts):
        return {"ingress_dispatched": True}
    return {}


# --- the canonical chat writers ----------------------------------------------------------------

def _chat_record(
    direction: str,
    chat_id: int,
    user_id: int,
    text: str,
    ts: Optional[str] = None,
    fmt: str = "",
    source: str = "",
    sender_label: str = "",
    sender_session_id: str = "",
    client_message_id: str = "",
    transport: Optional[Dict[str, Any]] = None,
    task_id: str = "",
    record_type: str = "",
    filename: str = "",
    mime: str = "",
    download_url: str = "",
    download_url_compat: str = "",
    caption: str = "",
    actions: Optional[List[Dict[str, str]]] = None,
    title: str = "",
    quiz: Optional[Dict[str, Any]] = None,
    card: Optional[Dict[str, Any]] = None,
    size_bytes: Optional[int] = None,
    client_surface: Optional[Dict[str, Any]] = None,
    message_meta: Optional[Dict[str, Any]] = None,
    drive_root=None,
    require_write: bool = False,
) -> Tuple[Any, Optional[dict]]:
    """``(root, the chat row as every reader expects it)``; ``(None, None)`` without a data root."""
    from supervisor import message_bus

    root = drive_root if drive_root is not None else message_bus.DATA_DIR
    if root:
        from pathlib import Path
        from ouroboros.utils import read_json_dict

        root = Path(root)
        record = {
            "ts": ts or utc_now_iso(),
            "session_id": ((read_json_dict(root / "state" / "state.json") or {})
                           if drive_root is not None else message_bus.load_state()).get("session_id"),
            "direction": direction,
            "chat_id": chat_id,
            "user_id": user_id,
            "text": text,
            "format": fmt,
            "source": source,
            "sender_label": sender_label,
            "sender_session_id": sender_session_id,
            "client_message_id": client_message_id,
            "transport": dict(transport or {}),
            "task_id": str(task_id or ""),
        }
        if direction == "in":
            from ouroboros.settings_integrity import effective_subagents_mode

            record["message_accepted_at"] = utc_now_iso()
            try:
                if effective_subagents_mode(task={"chat_id": chat_id}, drive_root=root) == "swarm":
                    record["subagents_mode"] = "swarm"
            except Exception:
                log.debug("Subagents label unavailable for the inbound row", exc_info=True)
        if direction == "in" and source == "web" and client_message_id and require_write:
            # The canonical append is the proof used by the live echo. Keep it
            # on that SAME row so history can replay the fact after a reload.
            record["ingress_accepted"] = True
        if direction == "in" and client_message_id and require_write:  # a named acceptance: which process took it
            from ouroboros.process_custody import current_custody_session_id

            record["ingress_process"] = current_custody_session_id()  # message_ingress.accepted_here
        # Media rows (e.g. delivered documents) carry a variable ``type`` plus
        # lightweight metadata so /api/chat/history can rebuild the bubble on
        # reload WITHOUT persisting base64. ``type`` is set from a variable (not a
        # literal) so the frozen-contract AST parity scan does not treat this
        # persisted row as a DocumentOutbound WS envelope.
        if record_type:
            record["type"] = record_type
        if client_surface:
            # Per-message sending-surface provenance (Owner Surface Fact);
            # optional column, never a "type":"chat" literal (AST-scan hygiene).
            record["client_surface"] = dict(client_surface)
        # Speech rows persist only the compact identity required to route a
        # child final after progress retention/rotation has removed its cards.
        meta = dict(message_meta or {})
        if meta.get("role") == "system":
            record["role"] = "system"
            record["system_type"] = str(meta.get("system_type") or "")
        if isinstance(meta.get("host_fact"), dict) and meta["host_fact"].get("kind") == "call_open":
            record["host_fact"] = dict(meta["host_fact"])
        if isinstance(meta.get("execution_transport"), dict):
            record["execution_transport"] = dict(meta["execution_transport"])
        if direction == "out" and task_id and record_type != "relay_entry" and "reply_to" not in meta:
            from ouroboros.chat_references import origin_reply
            meta["reply_to"] = origin_reply(root, task_id)
        for key in ("reply_to", "relay"):
            if isinstance(meta.get(key), dict) and meta[key]:
                record[key] = dict(meta[key])
        if meta.get("reply_to_relay"):
            record["reply_to_relay"] = str(meta["reply_to_relay"])
        if isinstance(meta.get("presence_contact"), dict):
            record["presence_contact"] = dict(meta["presence_contact"])
        if meta.get("late_answer"):
            from ouroboros.owner_quiz import late_answer_ref
            accepted_answer = late_answer_ref(meta["late_answer"])
            if accepted_answer:
                record["late_answer"] = accepted_answer
        if meta.get("attachments"):  # owner attachments: server-measured refs, never paths or bytes
            record["attachments"] = stored_refs(meta["attachments"])
        if meta.get("text_placeholder") is True and text:  # the host wrote TEXT; the sender sent no words
            record["text_placeholder"] = True
        for key in SUBAGENT_MESSAGE_FIELDS:
            if key in meta:
                record[key] = meta[key]
        if "narration" in meta:
            record["narration"] = bool(meta["narration"])
        if record_type in ("project_started", "project_handoff", "project_completion_summary"):
            for key in ("project_id", "project_name", "task_name", "target_label", "status", "completion_answer", "completion_reason", "completion_notice", "handoff_id", "terminal_time", "outcome_axes"):
                if key in meta:
                    record[key] = meta[key]
        if meta.get("delivery_id"):
            record["delivery_id"] = str(meta["delivery_id"])
        if "task_terminal_status" in meta:
            record["task_terminal_status"] = str(meta.get("task_terminal_status") or "")
        # The turn's origin label (a consciousness wake-up) survives the row
        # like the terminal status: a final bubble is labelled on reload too.
        if meta.get("initiator"):
            record["initiator"] = str(meta.get("initiator") or "")
        if isinstance(meta.get("origin_message_ref"), dict):
            record["origin_message_ref"] = dict(meta["origin_message_ref"])
        # The host's placement fact for a task-keyed System row: the row belongs
        # to the task's card, not beside it. Only the two named placements are
        # persisted, and the row's stable identity rides with one of them or not
        # at all — a bare id without a placement names nothing on reload.
        if meta.get("card_row") in CARD_ROW_PLACEMENTS:
            record["card_row"] = str(meta["card_row"])
            card_row_id = str(meta.get("card_row_id") or "")
            if card_row_id and len(card_row_id) <= 200:
                record["card_row_id"] = card_row_id
                revision = meta.get("card_row_revision")
                if type(revision) is int and revision >= 0:
                    record["card_row_revision"] = revision
        if filename:
            record["filename"] = filename
        if mime:
            record["mime"] = mime
        if download_url:
            record["download_url"] = download_url
        if download_url_compat:
            # Second address for the same bytes, for desktop launchers whose
            # file bridge predates the task-artifact route.
            record["download_url_compat"] = download_url_compat
        if caption:
            record["caption"] = caption
        if actions:
            record["actions"] = [dict(action) for action in actions]
            record["title"] = str(title or "")
        if quiz:
            # Quiz rows persist the full card (no base64 anywhere) so history
            # replay rebuilds it with its lifecycle state.
            record["quiz"] = dict(quiz)
        if card:  # the identity only; the body stays in the card store (supervisor/card_delivery.py)
            record["card"] = dict(card)
        if size_bytes is not None:
            record["size_bytes"] = int(size_bytes)
        return root, record
    elif require_write:
        raise RuntimeError("canonical message acceptance requires a data root")
    return None, None


def reopen_for_owner_message(drive_root, row: dict) -> None:
    """A fresh local owner's row reopens its room; replay never undoes a later closure.

    The web door fixes user_id=1. The numeric guard excludes non-project addresses cheaply;
    registry membership remains authority. Failure preserves the already accepted message.
    """
    from ouroboros.contracts.chat_id_policy import is_project_chat_id

    if (row.get("direction") != "in" or row.get("source") != "web" or row.get("user_id") != 1
            or not is_project_chat_id(row.get("chat_id"))):
        return
    from ouroboros.projects_registry import PROJECT_ACTIVE, list_reserved_projects, update_project

    try:
        address = row.get("chat_id")
        project = next((p for p in list_reserved_projects(drive_root, strict=True)
                        if p.get("lifecycle") == PROJECT_ACTIVE and p.get("chat_id") == address), None)
        if project and "closed" in project:
            update_project(drive_root, project["id"], closed=None)
    except Exception:
        log.warning("Accepted owner message could not reopen project room %s", row.get("chat_id"), exc_info=True)


def log_chat(direction: str, chat_id: int, user_id: int, text: str, *, feed: bool = False,
             require_write: bool = False, ensure_record_boundary: bool = False, **fields: Any) -> Optional[dict]:
    """Store one chat row (``dialogue_store.append``): the stored row, or None on a best-effort
    failure; ``require_write`` raises instead. ``feed`` moves the room's revision in the same
    transaction (a standalone message the owner has not read). A named inbound row is a named
    acceptance: an existing row of that id answers (no second row). ``ensure_record_boundary`` is
    the retired chain writer's parameter, accepted for its callers."""
    root, record = _chat_record(direction, chat_id, user_id, text, require_write=require_write, **fields)
    if record is None:
        return None
    try:
        shown = _shown_elsewhere(root, record, feed)
        receipt = (dialogue_store.append(root, "chat", record, feed=feed, idempotent=direction != "in", **shown)
                   if require_write else dialogue_store.append_quietly(root, "chat", record, feed=feed, **shown))
    except (StoreError, OSError, ValueError) as exc:
        if require_write:
            raise RuntimeError("canonical message acceptance could not be persisted") from exc
        log.warning("chat row not stored in %s", root, exc_info=True)
        return None
    if receipt is None:
        return None
    from ouroboros.notice_receipts import record as record_notice_receipt
    record_notice_receipt(root, record)  # idempotent: a duplicate also repairs a receipt its first write lost
    if not receipt.duplicate:
        reopen_for_owner_message(root, receipt.row)
    return receipt.row


def _shown_elsewhere(root: Any, record: dict, feed: bool) -> Dict[str, Any]:
    """The append facts of a row some Project also shows (``room_appearances.write_facts``): its
    derived index keys and, for work speech, the revision stamp and the other rooms it feeds, so
    one message is unread where it is shown."""
    from ouroboros.project_dialogue import WORK_SPEECH_ELSEWHERE_TYPES
    from ouroboros.room_appearances import write_facts

    keys, shown = write_facts(root, record)
    speech = record.get("type") in WORK_SPEECH_ELSEWHERE_TYPES
    return {"appearances": keys, **({"room_revisions": True, "also_feed": shown} if speech and feed else {})}


def accept_chat(chat_id: int, user_id: int, text: str, *, dispatch: Optional[Dict[str, Any]] = None,
                **fields: Any) -> "dialogue_store.Receipt":
    """A named acceptance's canonical write: the inbound row, its dedup, the room and the dispatch
    obligation in ONE transaction, answered with the store's receipt (``duplicate`` when an
    earlier delivery's row answered). A store failure is ``RuntimeError`` (nothing landed)."""
    root, record = _chat_record("in", chat_id, user_id, text, require_write=True, **fields)
    try:
        receipt = dialogue_store.append(root, "chat", record, dispatch=dispatch, **_shown_elsewhere(root, record, False))
    except (StoreError, OSError, ValueError) as exc:
        raise RuntimeError("canonical message acceptance could not be persisted") from exc
    if not receipt.duplicate:
        from ouroboros.notice_receipts import record as record_notice_receipt
        record_notice_receipt(root, record)
        reopen_for_owner_message(root, receipt.row)
    return receipt


# --- which accepted row an id already names --------------------------------------------------

def accepted_chat_message(drive_root, chat_id: int, client_message_id: str) -> Optional[dict]:
    """The named inbound row ``(chat_id, client_message_id)`` already names (the newest; the answer
    a full scan of the retained chain gave). A store that cannot answer raises typed
    (``store.StoreError``), never None."""
    found = dialogue_store.accepted_row(Path(drive_root), chat_id, client_message_id)
    return None if found is None else dict(found.row)


def rejoin_or_refuse(row: dict, source: str, logged: str, refs: Any, reply: Optional[dict] = None,
                    *, execution_transport: Optional[dict] = None) -> None:
    """A redelivered frame may rejoin the accepted row only as the same message of the same source
    (text, ordered attachment content and the Reply it points at)."""
    if row.get("source") != source:
        raise ValueError("client_message_id is already bound to another source")
    if row.get("execution_transport") != execution_transport:
        from ouroboros.contracts.realtime import RealtimeSetupError

        raise RealtimeSetupError("transport_identity", "client_message_id already names another execution transport")
    if execution_transport is not None and not accepted_here(row):
        from ouroboros.contracts.realtime import RealtimeSetupError

        raise RealtimeSetupError("realtime_restart_required", "this call's accepting process ended; start a new call")
    if (not same_message(row, logged, refs)
            or (row.get("reply_to") or {}).get("address", "") != (reply or {}).get("address", "")):
        raise ValueError("client_message_id was already used for a different message")


def hand_over(root: Path, chat_id: int, message_id: str, name: Optional[str]) -> bool:
    """The durable "entered" mark before the handoff, stamped with this process: the owed
    dispatch (one winner) or the caller's own dispatcher's entry."""
    if name is None:
        return dialogue_store.note_dispatched(root, chat_id, message_id, process=_process())
    return dialogue_store.claim_dispatch(root, chat_id, message_id, process=_process())


def accept_local_message(bridge, drive_root, text: str, *, retain_inputs=None, dispatch=None, **message) -> tuple[dict, bool]:
    """Accept a named skill delivery once, then schedule/queue its exact source: ``(row, rejoined)``.
    The ingress lock serializes receipt-before-dispatch. A same-id retry rejoins without dispatch
    unless the row's dispatch is still owed (``pending``: the acceptor died before the handoff):
    then it dispatches, not ``rejoined``. ``retain_inputs`` takes custody of the request's copies
    once a write is attempted or a row is handed over; a refusal before that adopts nothing.
    """
    from ouroboros.project_dialogue import build_owner_message_ref
    from ouroboros.chat_uploads import attachment_placeholder
    from ouroboros.owner_quiz import late_answer_ref
    from supervisor import message_bus

    root = Path(drive_root)
    chat_id = int(message["chat_id"])
    message_id = str(message["client_message_id"])
    source = str(message["source"])
    logged = text.strip() or str(message.get("image_caption") or "").strip() or attachment_placeholder(
        message.get("image_base64"), message.get("task_metadata")
    )
    if not logged:
        raise ValueError("message is empty")
    from ouroboros.chat_references import ingress_reply
    reply = ingress_reply(drive_root, message.get("task_metadata") or {})
    if reply:  # a replayed obligation dispatches the same reference
        message["task_metadata"] = {**(message.get("task_metadata") or {}), "reply_reference": reply}
    refs = stored_refs((message.get("task_metadata") or {}).get("chat_attachments"))
    late_answer = late_answer_ref((message.get("task_metadata") or {}).get("late_answer"))
    placeholder = not text.strip() and not str(message.get("image_caption") or "").strip()
    name = dispatcher_name(dispatch)
    with _INGRESS_LOCK:
        row = accepted_chat_message(root, chat_id, message_id)
        fresh = attempted = False
        if row is None:
            attempted = True
            try:
                receipt = message_bus.accept_chat(
                    chat_id, int(message.get("user_id") or 0), logged, ts=utc_now_iso(), source=source,
                    client_message_id=message_id, sender_label=str(message.get("sender_label") or ""),
                    transport=message.get("transport"), drive_root=root,
                    message_meta={"attachments": refs, "text_placeholder": placeholder, "reply_to": reply,
                                  **({"late_answer": late_answer} if late_answer else {}),
                                  "execution_transport": (message.get("task_metadata") or {}).get("execution_transport")},
                    dispatch=obligation(name, text, message))
            finally:
                # Once the write is attempted, failure may have left the canonical row: transfer input
                # custody without claiming acceptance or queue success.
                if retain_inputs is not None:
                    retain_inputs()
            row, fresh = receipt.row, not receipt.duplicate
        if fresh:
            hand_over(root, chat_id, message_id, name)
        else:
            # The row an earlier delivery landed (or the race lost by a hair): rejoin it, or take its owed dispatch.
            rejoin_or_refuse(row, source, logged, refs, reply,
                            execution_transport=(message.get("task_metadata") or {}).get("execution_transport"))
            saved_answer = late_answer_ref(row.get("late_answer"))
            if saved_answer:
                message["task_metadata"] = {**(message.get("task_metadata") or {}), "late_answer": saved_answer}
            if not dialogue_store.claim_dispatch(root, chat_id, message_id, process=_process()):
                return row, True
            # Its dispatch was still owed: this retry's identical copies feed it; the echo shows the row's refs.
            if retain_inputs is not None and not attempted:
                retain_inputs()
            if (message.get("task_metadata") or {}).get("chat_attachments"):
                message["task_metadata"] = {**message["task_metadata"], "chat_attachments": row.get("attachments")}
        ts = str(row.get("ts") or "")
        if row.get("reply_to"):  # the reference the host resolved when it accepted the row
            message["task_metadata"] = {**(message.get("task_metadata") or {}), "reply_reference": row["reply_to"]}
        ref = build_owner_message_ref(chat_id=chat_id, client_message_id=message_id, ts=ts, text=logged)
        # record_inbound_message gets the row witness and receipt time.
        # Dispatch only schedules/queues; slow work must not hold the ingress lock.
        (dispatch or bridge.enqueue_local_message)(text, **message, accepted_source_ref=ref, accepted_source_row=row, received_at=ts)
        dialogue_store.complete_dispatch(root, chat_id, message_id)
        return row, False


def recover_pending_dispatches(bridge, drive_root) -> int:
    """Perform every dispatch the store still owes (a crash between the commit and the handoff),
    exactly once each: the partial index names them, no chain is read. An obligation naming a
    dispatcher this body lacks is abandoned with that reason. Returns the number dispatched."""
    from ouroboros.project_dialogue import build_owner_message_ref

    root = Path(drive_root)
    dispatched = 0
    for stored, payload in dialogue_store.pending_dispatches(root):
        row = stored.row
        chat_id, message_id = _named(row)
        name = str(payload.get("dispatcher") or "")
        dispatcher = _dispatchers().get(name)
        if dispatcher is None or chat_id is None:
            dialogue_store.abandon_dispatch(root, chat_id, message_id, f"dispatcher_unknown:{name or 'none'}")
            log.warning("Owed dispatch of %s:%s abandoned: dispatcher %r unknown", chat_id, message_id, name)
            continue
        with _INGRESS_LOCK:
            if not dialogue_store.claim_dispatch(root, chat_id, message_id, process=_process()):
                continue  # a retry took it meanwhile
            ts = str(row.get("ts") or "")
            text = str(payload.get("text") or "")
            message = dict(payload.get("message") or {})
            ref = build_owner_message_ref(chat_id=chat_id, client_message_id=message_id, ts=ts, text=str(row.get("text") or text))
            try:
                dispatcher(bridge, text, **message, accepted_source_ref=ref, accepted_source_row=row, received_at=ts)
            except Exception:
                log.error("Owed dispatch of %s:%s raised after entry; its outcome is unknown", chat_id, message_id, exc_info=True)
                continue
            dialogue_store.complete_dispatch(root, chat_id, message_id)
            dispatched += 1
    return dispatched


def record_inbound_message(bridge, message: dict, *, chat_id: int, user_id: int,
                           client_message_id: str, text: str, ts: str) -> Optional[dict]:
    """Keep one canonical ingress writer for dequeued and preaccepted messages."""
    from ouroboros.project_dialogue import (
        _text_sha256, build_owner_message_ref, entry_matches_source_ref, owner_message_ref_is_valid,
    )
    from supervisor import message_bus

    source = str(message.get("source") or "web")
    ref = message.get("accepted_source_ref")
    if ref:
        # A web acceptance (socket or late quiz answer) wrote this row in-process
        # before enqueue; its returned row is the item's exact witness. A skill
        # source, or a web item without a witness (the queue's empty default is
        # absent, not a row to match), re-reads the retained row.
        accepted_row = message.get("accepted_source_row")
        row = (accepted_row if source == "web" and isinstance(accepted_row, dict) and accepted_row
               else accepted_chat_message(message_bus.DATA_DIR, chat_id, client_message_id)) if owner_message_ref_is_valid(ref) else None
        if (not row or row.get("source") != source or row.get("chat_id") != chat_id
                or row.get("client_message_id") != client_message_id or not entry_matches_source_ref(row, [ref])
                or ref["text_sha256"] != _text_sha256(text)):
            raise ValueError("accepted source does not match the queued message")
        ref = dict(ref)
        ts = ref["ts"]
        placeholder = row.get("text_placeholder") is True
    elif message.get("suppress_chat_log"):
        return None
    else:
        metadata = message.get("task_metadata") or {}
        # no words or caption from the sender: TEXT is the host's placeholder
        placeholder = not str(message.get("text") or "").strip() and not str(message.get("image_caption") or "").strip()
        from ouroboros.chat_references import ingress_reply
        reply = ingress_reply(message_bus.DATA_DIR, metadata)
        if reply:
            metadata["reply_reference"] = reply
            message["task_metadata"] = metadata
        row = message_bus.log_chat(
            "in", chat_id, user_id, text, ts=ts, source=source,
            sender_label=str(message.get("sender_label") or ""),
            sender_session_id=str(message.get("sender_session_id") or ""),
            client_message_id=client_message_id, transport=message.get("transport"),
            client_surface=(metadata.get("client_surface") if isinstance(metadata, dict)
                            and isinstance(metadata.get("client_surface"), dict) else None),
            message_meta={"attachments": metadata.get("chat_attachments") if isinstance(metadata, dict) else None,
                          "text_placeholder": placeholder, "reply_to": reply},
        )
        ref = build_owner_message_ref(chat_id=chat_id, client_message_id=client_message_id, ts=ts, text=text)
    from ouroboros.chat_references import projection_fields
    from ouroboros.room_appearances import frame_rooms
    if source != "web":
        # A parked inline photo is one of the views: one bubble, never a second photo frame;
        # its text is the row's, placeholder mark included, as history replays it.
        views = attachment_views((message.get("task_metadata") or {}).get("chat_attachments"))
        bridge.broadcast({
            "type": "photo" if message.get("image_base64") and not views else "chat", "role": "user",
            "content": text if views and placeholder else str(message.get("text") or ""),
            "caption": str(message.get("image_caption") or ""), **({"text_placeholder": True} if views and placeholder else {}),
            "image_base64": "" if views else str(message.get("image_base64") or ""), **({"attachments": views} if views else {}),
            "mime": str(message.get("image_mime") or "image/jpeg"), "ts": ts, "source": source,
            "sender_label": str(message.get("sender_label") or ""),
            "sender_session_id": str(message.get("sender_session_id") or ""),
            "client_message_id": client_message_id, "transport": message.get("transport") or {},
            "chat_id": chat_id, **(projection_fields(row) if row else {}), **frame_rooms(message_bus.DATA_DIR, row),
            **({"subagents_mode": "swarm"} if row and row.get("subagents_mode") == "swarm" else {}),
        })
    return ref


def handle_web_message(
    self,
    text: str,
    *,
    sender_session_id: str = "",
    client_message_id: str = "",
    image_base64: str = "",
    image_mime: str = "",
    image_caption: str = "",
    task_metadata: Optional[Dict[str, Any]] = None,
    chat_id: int = 1,
    project_id: str = "",
    dispatch=None,
) -> None:
    from supervisor import message_bus as bus
    from ouroboros.chat_uploads import claim_refs
    from ouroboros.chat_references import ingress_reply, projection_fields
    from ouroboros.room_appearances import frame_rooms

    # Web ownership stays user_id=1; positive chat_id selects Main/Project.
    # Negative A2A ids are not web destinations and fall back to Main.
    try:
        thread_id = int(chat_id or 1)
    except (TypeError, ValueError):
        thread_id = 1
    if thread_id < 1:
        thread_id = 1
    clean_text = str(text or "").strip()
    if self.panic.request(clean_text):
        return
    if not clean_text and not image_base64 and not (task_metadata or {}).get("chat_attachment_uploads"):
        return
    import uuid

    from ouroboros.process_custody import current_custody_session_id
    from ouroboros.project_dialogue import build_owner_message_ref

    ts = utc_now_iso()
    message_id = client_message_id or f"web-{uuid.uuid4().hex}"
    metadata = dict(task_metadata or {})
    host_fact = metadata.get("host_fact") or {}  # host call admission; browser ingress constructs its own metadata
    call_open = isinstance(host_fact, dict) and host_fact.get("kind") == "call_open"
    reply = ingress_reply(bus.DATA_DIR, metadata)
    if reply:  # the obligation of a fresh acceptance replays the same reference
        metadata["reply_reference"] = reply
    if str(project_id or "").strip():
        metadata.setdefault("project_id", str(project_id).strip())
    from ouroboros.chat_uploads import attachment_placeholder

    log_text = clean_text or str(image_caption or "").strip() or attachment_placeholder(image_base64, metadata)
    placeholder = call_open or not clean_text  # host facts and attachment labels are not owner words
    # The canonical row precedes echo/dispatch: neither WS nor Queue survives restart.
    # The row, its receipt and the dispatch obligation are ONE store transaction; the
    # returned witness lets dequeue validation avoid any chat read.
    name = dispatcher_name(dispatch)
    with _INGRESS_LOCK:
        row = accepted_chat_message(bus.DATA_DIR, thread_id, client_message_id) if client_message_id else None
        fresh = False
        if row is None:
            attachments = claim_refs(metadata.get("chat_attachments"), bus.DATA_DIR)  # DELETE refuses them now
            message = dict(
                chat_id=thread_id, user_id=1, source="web",
                sender_session_id=sender_session_id, client_message_id=message_id,
                image_base64=image_base64, image_mime=image_mime,
                image_caption=image_caption, task_metadata=metadata,
            )
            receipt = bus.accept_chat(
                thread_id, 1, log_text, ts=ts, source="web",
                sender_session_id=sender_session_id, client_message_id=message_id,
                client_surface=(metadata.get("client_surface") if isinstance(metadata.get("client_surface"), dict) else None),
                message_meta={"attachments": attachments, "text_placeholder": placeholder, "reply_to": reply,
                              "execution_transport": metadata.get("execution_transport"),
                              **({"role": "system", "system_type": "call_open", "host_fact": dict(host_fact)}
                                 if call_open else {})},
                dispatch=obligation(name, clean_text, message))
            row, fresh = receipt.row, not receipt.duplicate
        if fresh:
            handed = True
        else:
            # A redelivered frame names the accepted message: the same words, the same ordered
            # attachment content and the same Reply rejoin it (no second row); anything else under
            # that id is refused, claiming none of its uploads. A rejoin dispatches only by taking
            # the row's still-owed dispatch (once); a dispatch that began is never replayed.
            rejoin_or_refuse(row, "web", log_text, stored_refs(metadata.get("chat_attachments")), reply,
                            execution_transport=metadata.get("execution_transport"))
            ts, attachments = str(row.get("ts") or ts), row.get("attachments")
            log_text, placeholder = str(row.get("text") or log_text), row.get("text_placeholder") is True
            if handed := dialogue_store.claim_dispatch(bus.DATA_DIR, thread_id, message_id, process=current_custody_session_id()):
                claim_refs(metadata.get("chat_attachments"), bus.DATA_DIR)  # the copies it stages are kept
        if row.get("reply_to"):  # the reference the host resolved when it accepted the row
            metadata["reply_reference"] = row["reply_to"]
        if handed:
            ref = build_owner_message_ref(chat_id=thread_id, client_message_id=message_id, ts=ts, text=log_text)
            message = dict(
                chat_id=thread_id, user_id=1, source="web",
                sender_session_id=sender_session_id, client_message_id=message_id,
                image_base64=image_base64, image_mime=image_mime,
                image_caption=image_caption, task_metadata=metadata,
                accepted_source_ref=ref, accepted_source_row=row,
            )
            if dispatch is None:
                if fresh:
                    hand_over(bus.DATA_DIR, thread_id, message_id, name)  # the durable "entered" mark
                self.enqueue_local_message(clean_text, **message)
                dialogue_store.complete_dispatch(bus.DATA_DIR, thread_id, message_id)
    try:
        if self._broadcast_fn:
            # The echo is the accepted row as history replays it (text, time, views, placeholder
            # mark); a rejoin re-echoes it, so every tab keys it as the same saved bubble.
            echo = {
                "type": "chat", "role": row.get("role") or "user", "content": log_text,
                "ts": ts, "source": "web", "chat_id": thread_id, "attachments": attachment_views(attachments),
                "sender_session_id": sender_session_id, "client_message_id": message_id,
                "ingress_accepted": True,  # Canonical row exists; execution is not promised.
            }
            if row.get("subagents_mode") == "swarm":
                echo["subagents_mode"] = "swarm"
            if placeholder:
                echo["text_placeholder"] = True
            if row.get("system_type"):
                echo["system_type"] = row["system_type"]
            # Positive call-entry evidence, never the acceptance stamp alone; a deferred dispatch is still pending.
            echo.update(delivery_facts(row, thread_id))
            echo.update(projection_fields(row), **frame_rooms(bus.DATA_DIR, row))  # a Reply a Project also shows
            bus.stamp_project_thread(bus.DATA_DIR, echo)
            self._broadcast_fn(echo)
    finally:
        if dispatch is not None and handed:
            if fresh:
                hand_over(bus.DATA_DIR, thread_id, message_id, name)  # entered now, not at the echo
            dispatch(clean_text, **message)  # Accepted work survives an echo/disconnect failure.
            dialogue_store.complete_dispatch(bus.DATA_DIR, thread_id, message_id)
