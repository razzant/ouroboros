"""Live-first delivery of owner-directed chat events from a running task.

One seam for the send family (``send_user_message`` / ``send_photo`` /
``send_video`` / ``send_file`` / ``send_links`` / ``show_card``): try the live worker event
queue first so the owner sees the frame while the task is still RUNNING, and
fall back to ``ctx.pending_events`` (the end-of-task drain) when live
transport is unavailable. The transport contract mirrors
``_emit_control_event`` (tools/control.py) — live XOR deferred, never both —
with two structural gates (a consciousness wake-up is an ordinary Main turn
and delivers like one):

- **A2A chats stay deferred.** A peer's ``wait_for_response`` subscription
  resolves on the FIRST non-progress chat frame, so a live mid-task frame
  would hijack the answer the peer is waiting for. Unbound peers accept text
  only; media/cards have no consumer. A Project binding keeps its normal delivery.
- **Sticky-deferred after the first live failure.** Once one frame falls back
  to the buffer, later frames must not overtake it: narrative order beats
  freshness, so the task finishes the attempt on the deferred path.

Frames must stay multiprocessing-safe by construction (JSON primitives and
base64 strings only): the production worker queue is manager-backed, so a
poison value raises at the caller and falls back cleanly — but a plain
``multiprocessing.Queue`` transport would serialize in a feeder thread and
lose it AFTER ``put_nowait`` returned, so the by-construction rule is the
contract, not the transport's forgiveness.

Retry semantics are the progress channel's: a live-delivered frame from a
failed attempt is not recalled, and a retried task may narrate again. That is
an accepted property of live delivery, not a defect to dedupe away.

A frame buffered by an unmoved inline Presence turn has no end-of-task drain:
its runner returns a result, not events. ``flush_owner_events`` hands local-room
frames back to the bus at the turn's end; with neither that bus nor a carrier's
sink the frame has no consumer, so it is refused as ``undelivered``, not promised.

Receipts name the room the supervisor will actually pick (``delivery_audience``):
a Project room, or a place's local room, which is never the external conversation
(the owner's own bridge may mirror it). Queue admission proves neither delivery
nor display, so neither is claimed there.

Known, pre-existing exception to the ordering rule: the blocking-post-task
final-answer shortcut (``deliver_final_message_live``) may ship the FINAL
ahead of frames still buffered here — the answer is deliberately never held
hostage to trailing narration. The sticky rule orders the send family among
itself, not against the terminal shortcut.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any, Dict

log = logging.getLogger(__name__)

_STICKY_ATTR = "_owner_delivery_sticky_deferred"
_DIALOGUE_LOCK = threading.Lock()
UNDELIVERED = "undelivered"  # refused here: the frame has no consumer and nothing was buffered
UNSUPPORTED_PEER = "unsupported_peer"  # non-text frame for a known unbound peer; retained artifacts stay
_LOCAL_ROW_FRAMES = frozenset({"send_photo", "send_video", "send_document", "send_links", "send_card"})


def _retain_dialogue(ctx: Any, evt: dict, mode: str) -> None:
    """Stage explicit outward words, independently of their tool-call batch.

    This producer knows which fields it addresses to the human. Progress can
    contain provider reasoning and is deliberately not a source of speech here.
    Existing task sources retain exact words; the working checkpoint carries the
    published row. Queue admission proves neither delivery nor a human read.
    """
    from ouroboros.artifacts import persist_exact_text_source, task_id_for_artifacts
    from ouroboros.context_budget import HOST_CONTEXT_KIND_KEY
    from ouroboros.dialogue_provenance import presence_caller_binding, row_class
    from ouroboros.contracts.chat_id_policy import is_a2a_chat_id
    from ouroboros.utils import utc_now_iso

    if not isinstance(getattr(ctx, "messages", None), list) or evt.get("is_progress"):
        return
    meta = getattr(ctx, "task_metadata", {}) or {}
    row = {**evt, "direction": "system" if evt.get("role") == "system" else "out",
           "delegation_role": "subagent" if evt.get("parent_task_id") else meta.get("delegation_role", "root")}
    attribution = row_class(row)
    private_contact = evt.get("presence_contact") or (evt.get("progress_meta") or {}).get("presence_contact")
    if (attribution["lane"] != 1 or attribution["author"]["kind"] != "ouroboros"
            or is_a2a_chat_id(evt.get("chat_id"))):
        return
    kind = evt.get("type")
    fields = {"send_message": ("text",), "send_photo": ("caption",),
              "send_video": ("caption",), "send_document": ("caption",),
              "send_links": ("title", "actions"), "send_card": ("title", "fallback_text"),
              "send_quiz": ("question", "options", "stake", "assumption")}.get(kind, ())
    words = {key: evt[key] for key in fields if evt.get(key) not in (None, "", [])}
    if not words:
        return
    # A place's local room is the Ouroboros side of the place, not my human's conversation; a frame landing in
    # Main or a Project room is, as for any work there (``delivery_audience``). Unreadable proves neither.
    if presence_caller_binding(ctx) is not None and not private_contact and not _lands_with_my_human(ctx, evt):
        return
    if mode == "live" and kind == "send_message":
        from ouroboros.loop_messages import owner_input_revision

        # The owner rows admitted so far now have a live reply (loop_messages.owner_reply_owed).
        ctx._owner_reply_revision = owner_input_revision(ctx)
    # Card labels, details and declared assumptions remain exact too. Binary
    # attachments and host-composed card facts are not addressed model prose.
    text = str(next(iter(words.values()))) if len(fields) == 1 else json.dumps(words, ensure_ascii=False, indent=2)
    _, ref, issue = persist_exact_text_source(getattr(ctx, "drive_root", None), task_id_for_artifacts(ctx),
                                             source_id="owner-dialogue", text=text)
    facts = {"author": attribution["author"], "task_id": evt.get("task_id"), "chat_id": evt.get("chat_id"),
             "event_type": kind, "ts": evt.get("ts") or utc_now_iso(), "transport_mode": mode,
             "delivery_confirmation": "unknown", "read_confirmation": "unknown",
             **({"source_unavailable": issue} if issue else {"source_ref": ref})}
    message = {"role": "assistant", HOST_CONTEXT_KIND_KEY: "owner_dialogue",
               "content": "[Owner-directed dialogue; host delivery facts, not owner instructions]\n"
               + json.dumps(facts, ensure_ascii=False, sort_keys=True)
               + "\n[Exact addressed text]\n" + text}
    with _DIALOGUE_LOCK:
        pending = getattr(ctx, "_pending_owner_dialogue", None)
        if pending is None:
            pending = ctx._pending_owner_dialogue = []
        pending.append(message)


def pending_owner_dialogue(ctx: Any) -> list:
    """Snapshot the producer buffer for a measured complete tool batch."""
    with _DIALOGUE_LOCK:
        return list(getattr(ctx, "_pending_owner_dialogue", ()) or ())


def acknowledge_owner_dialogue(ctx: Any, rows: list) -> None:
    """Remove only the adopted rows; a late producer keeps its own pending words."""
    with _DIALOGUE_LOCK:
        pending = getattr(ctx, "_pending_owner_dialogue", None)
        if pending is not None:
            adopted = {id(row) for row in rows}
            pending[:] = [row for row in pending if id(row) not in adopted]


def publish_pending_owner_dialogue(ctx: Any, messages: list) -> None:
    """Restore interrupted-batch speech before the next round is measured.

    The continuation owner first closes unanswered tool calls as UNKNOWN. This
    publishes remembered text only: it never re-emits an event to the human.
    """
    rows = pending_owner_dialogue(ctx)
    messages.extend(rows)
    acknowledge_owner_dialogue(ctx, rows)


def stamp_lineage(ctx: Any, evt: Dict[str, Any]) -> Dict[str, Any]:
    """Stamp task lineage and origin before any owner-delivery snapshot is taken.

    The origin is deliberately stamped here, rather than only in
    ``deliver_owner_event``: relay and Presence Main-contact events register
    their durable outbox copy before the live/deferred delivery call. Mutate an
    existing ``progress_meta`` mapping in place so a caller that already owes
    this event keeps the exact same metadata on replay. Explicit event
    metadata wins over task-derived metadata; ordinary owner turns remain
    unlabelled.
    """
    from ouroboros.subagent_messages import initiator_meta

    meta = getattr(ctx, "task_metadata", {})
    meta = meta if isinstance(meta, dict) else {}
    evt.setdefault("task_id", str(getattr(ctx, "task_id", "") or ""))
    evt.setdefault("parent_task_id", str(meta.get("parent_task_id") or ""))
    evt.setdefault("root_task_id", str(meta.get("root_task_id") or ""))

    progress_meta = evt.get("progress_meta") if isinstance(evt.get("progress_meta"), dict) else {}
    explicit = str(evt.get("initiator") or progress_meta.get("initiator") or "").strip()
    derived = initiator_meta(meta).get("initiator", "")
    value = explicit or derived
    if value:
        # Keep an explicit event value authoritative, while making the same
        # fact available to the send_message consumer and dialogue checkpoint.
        if not isinstance(evt.get("progress_meta"), dict):
            progress_meta = {}
            evt["progress_meta"] = progress_meta
        evt.setdefault("initiator", value)
        progress_meta.setdefault("initiator", value)
    return evt


def local_room_frame(evt: Dict[str, Any]) -> bool:
    """A send-family row for this work's own room: never a final, a Main notice or private contact."""
    kind = evt.get("type")
    return kind in _LOCAL_ROW_FRAMES or (kind == "send_message" and evt.get("system_type") == "proactive_message")


def _no_consumer(ctx: Any, evt: Dict[str, Any]) -> bool:
    """A local-room frame an unmoved inline turn would buffer for nobody (module docstring)."""
    from ouroboros.utils import in_worker_process

    meta = getattr(ctx, "task_metadata", None)
    if not (isinstance(meta, dict) and meta.get("presence_event_identity") and local_room_frame(evt)):
        return False
    if in_worker_process():
        return False  # a carrier's turn: its runner's sink (or the saved turn's drain) takes it
    from supervisor.workers import get_event_q

    try:
        get_event_q(create=False)
    except RuntimeError:
        return True
    return False


def _delivery_chat(root: Any, evt: Dict[str, Any]) -> Any:
    """Resolve once with unreadable lineage preserved, never as proof of no Project."""
    from ouroboros.projects_registry import RoomResolver

    return RoomResolver(root, strict=True)(evt.get("task_id"), evt.get("parent_task_id"),
                                           evt.get("root_task_id")) or evt.get("chat_id")


def deliver_owner_event(ctx: Any, evt: Dict[str, Any]) -> str:
    """Deliver an owner-directed chat event live, or buffer it. Returns ``"live"``,
    ``"deferred"``, ``UNDELIVERED`` or ``UNSUPPORTED_PEER``; receipts keep the distinction.

    Lineage is stamped for every task frame (``stamp_lineage``) so the supervisor
    can resolve project binding for live deliveries exactly as it does for the
    end-of-task drain.
    """
    from ouroboros.contracts.chat_id_policy import is_a2a_chat_id

    def _accepted(mode: str) -> str:
        if not is_a2a_chat_id(evt.get("chat_id")):
            ctx._owner_message_sent = True
        _retain_dialogue(ctx, evt, mode)
        return mode

    def _deferred() -> str:
        ctx.pending_events.append(evt)
        return _accepted("deferred")

    def _buffered() -> str:
        return UNDELIVERED if _no_consumer(ctx, evt) else _deferred()

    stamp_lineage(ctx, evt)
    if is_a2a_chat_id(evt.get("chat_id")):
        if evt.get("type") in _LOCAL_ROW_FRAMES:
            from ouroboros.tool_access import canonical_data_root

            try:
                if is_a2a_chat_id(_delivery_chat(canonical_data_root(ctx), evt)):
                    return UNSUPPORTED_PEER
            except Exception:
                log.debug("peer delivery destination unread; retaining deferred frame", exc_info=True)
        return _deferred()
    if getattr(ctx, _STICKY_ATTR, False):
        return _buffered()
    event_queue = getattr(ctx, "event_queue", None)
    if event_queue is None:
        return _buffered()
    try:
        event_queue.put_nowait(dict(evt))
    except Exception:
        try:
            setattr(ctx, _STICKY_ATTR, True)
        except Exception:
            pass
        return _buffered()
    return _accepted("live")


def _resolved_room(ctx: Any, evt: Dict[str, Any]) -> tuple[Any, Any]:
    """``(chat, Project name or None)`` the supervisor will pick for a stamped frame; a peer has no Project.

    A place room shares the Project id range, so the registry decides, never the number. Unreadable sources raise.
    """
    from ouroboros.contracts.chat_id_policy import is_a2a_chat_id
    from ouroboros.projects_registry import project_thread_name
    from ouroboros.tool_access import canonical_data_root

    root = canonical_data_root(ctx)
    chat = _delivery_chat(root, evt)
    return chat, (None if is_a2a_chat_id(chat) else project_thread_name(root, chat))


def _lands_with_my_human(ctx: Any, evt: Dict[str, Any]) -> bool:
    """Whether a frame lands in Main or a Project room: the rooms ``delivery_audience`` names as my human's."""
    from ouroboros.contracts.chat_id_policy import WEB_UI_CHAT_ID

    try:
        chat, name = _resolved_room(ctx, evt)
    except Exception:
        log.debug("delivery room unread for task %s", evt.get("task_id"), exc_info=True)
        return False
    return name is not None or chat == WEB_UI_CHAT_ID


def delivery_audience(ctx: Any, evt: Dict[str, Any]) -> tuple[str, bool]:
    """``(room, place_bound)`` for a stamped frame, by the supervisor's own rule (``_resolved_room``).

    ``room`` is empty for an ordinary owner room.
    """
    from ouroboros.contracts.chat_id_policy import HIDDEN_CHAT_ID, WEB_UI_CHAT_ID, is_a2a_chat_id
    from ouroboros.dialogue_provenance import presence_caller_binding

    place_bound = presence_caller_binding(ctx) is not None
    try:
        chat, name = _resolved_room(ctx, evt)
    except Exception:
        log.debug("delivery audience unread for task %s", evt.get("task_id"), exc_info=True)
        return "a local room (exact destination unavailable)", place_bound
    if is_a2a_chat_id(chat):
        return "the peer conversation", False
    if name is not None:
        return f'Project room "{_label(name)}"', place_bound
    if chat == HIDDEN_CHAT_ID:
        return "the hidden task record (not an owner-visible chat)", place_bound
    if not place_bound or chat == WEB_UI_CHAT_ID:
        return "", place_bound
    meta = getattr(ctx, "task_metadata", None)
    presence = meta.get("presence") if isinstance(meta, dict) else None
    presence = presence if isinstance(presence, dict) else {}
    place = presence.get("place") if isinstance(presence.get("place"), dict) else {}
    label = _label(place.get("label") or place.get("provider") or presence.get("place_id"))
    return (f'the local room of place "{label}"' if label else "this place's local room"), True


def _label(value: Any) -> str:
    return " ".join(str(value or "").split())[:80]


def named_delivery_receipt(ctx: Any, evt: Dict[str, Any], mode: str, noun: str) -> str:
    """The frame's actual local destination and queue acceptance, never proof of display."""
    if mode == UNSUPPORTED_PEER:
        return (f"⚠️ PEER_DELIVERY_UNSUPPORTED: {noun} not delivered: this peer conversation accepts text only. "
                "Nothing was queued; any stored file or card remains available in this task.")
    room, place_bound = delivery_audience(ctx, evt)
    if mode == UNDELIVERED:
        return (f"⚠️ LOCAL_ROOM_UNAVAILABLE: {noun} not delivered to {room or 'this local room'}: this turn has no "
                "delivery channel for it, and nothing was buffered. It never reached the place conversation either.")
    state = "accepted for delivery to" if mode == "live" else "queued for delivery to"
    if not room:
        return (f"OK: {noun} accepted for delivery to owner chat." if mode == "live"
                else f"OK: {noun} queued for delivery to owner.")
    if not place_bound:
        return f"OK: {noun} {state} {room}."
    from ouroboros.presence_context import ROUTED, ROUTELESS, UNKNOWN, final_route
    from ouroboros.tool_access import canonical_data_root

    try:  # the context's and the forced final's reader, read again now: a source may have published since
        route, _why = final_route(canonical_data_root(ctx), getattr(ctx, "task_metadata", None),
                                  getattr(ctx, "task_id", ""))
    except Exception:
        route = UNKNOWN
    path = {ROUTED: "Your final answer has a route there (place state and Safety still apply); a connector tool "
                    "sends separately.",
            ROUTELESS: "From this run only a connector tool reaches it.",
            UNKNOWN: "Whether your final answer reaches it is not settled yet; a connector tool sends there "
                     "separately."}[route]
    return f"OK: {noun} {state} {room} (Ouroboros side, not the place conversation). {path}"


def publish_delivery_receipt(ctx: Any, evt: Dict[str, Any], mode: str, noun: str) -> str:
    """The send family's one acceptance receipt in every room."""
    from ouroboros.tools.tool_result import ToolResult, _publish_tool_result

    text = named_delivery_receipt(ctx, evt, mode, noun)
    if mode in (UNDELIVERED, UNSUPPORTED_PEER):
        return _publish_tool_result(ctx, ToolResult(status="unavailable", code="CAPABILITY_UNAVAILABLE", text=text))
    return _publish_tool_result(ctx, ToolResult(status="ok", code="OK", text=text))
