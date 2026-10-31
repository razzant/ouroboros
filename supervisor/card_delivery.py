"""Card rows in the owner's chat: the bridge half of ``show_card`` (``ouroboros/tools/cards.py``).

The tool stores the card (``ouroboros/card_store.py``) and emits ``send_card``. Here the
supervisor writes the chat row (``type: "card"`` with a typed ``card`` identity; the body
stays in the store), broadcasts the live ``card`` frame, publishes ``chat.card`` for
skills and hands transports that cannot draw a card its ``fallback_text`` on
``chat.outbound``. A replaced body or a recorded press is a ``card_state`` signal: the
row already exists, and a client reads the current card again.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from ouroboros.card_store import record_publication, submissions_summary
from ouroboros.contracts.chat_id_policy import is_a2a_chat_id
from ouroboros.event_bus import CHAT_CARD, CHAT_OUTBOUND, publish_event
from ouroboros.utils import utc_now_iso

# What a transport without card rendering adds under an interactive card's plain text.
INTERACTIVE_NOTE = "(Open Ouroboros to use this card.)"


def card_identity(record: Dict[str, Any]) -> Dict[str, Any]:
    """The ``card`` payload of the row and of both frames: who the card is, never its body."""
    return {"card_id": str(record["card_id"]), "kind": str(record.get("kind") or ""),
            "title": str(record.get("title") or ""), "revision": int(record.get("revision") or 1)}


def send_card(bridge: Any, record: Dict[str, Any], chat_id: int, *, task_id: str,
              interactive: bool) -> Tuple[bool, str]:
    """Write the card's chat row, then tell the page, the skills and plain transports."""
    from ouroboros.chat_references import projection_fields
    from ouroboros.projects_registry import stamp_project_thread
    from supervisor import message_bus as bus
    from supervisor.state import load_state

    if is_a2a_chat_id(chat_id):
        return True, "ok"
    ts = utc_now_iso()
    identity = card_identity(record)
    fallback = str(record.get("fallback_text") or "")
    try:
        owner_id = int(load_state().get("owner_id") or 0)
    except Exception:
        owner_id = 0
    # ``feed``: a card is a standalone message, so the room's unread moves with its row (one transaction).
    stored = bus.log_chat("out", int(chat_id), owner_id, fallback, ts=ts, task_id=task_id,
                          record_type="card", feed=True, card=identity)
    # The same ``card`` identity the history row carries (CardOutbound; literal keys for the parity scan).
    frame = {"type": "card", "role": "assistant", "card": identity, "fallback_text": fallback,
             "ts": ts, "chat_id": int(chat_id), "task_id": task_id}
    if stored:
        frame.update(projection_fields(stored))
        try:  # the row a late press answers (the owner's message replies to the card); never blocks delivery
            record_publication(bus.DATA_DIR, identity["card_id"], int(chat_id), address=str(frame.get("row_address") or ""))
        except (OSError, TimeoutError, ValueError):
            pass
    stamp_project_thread(bus.DATA_DIR, frame)
    bridge.broadcast(frame)
    transport = bridge.chat_transport(chat_id)
    publish_event(CHAT_CARD, {**identity, "chat_id": int(chat_id), "transport": transport,
                              "fallback_text": fallback, "interactive": bool(interactive),
                              "task_id": task_id, "ts": ts})
    publish_event(CHAT_OUTBOUND, {
        "chat_id": int(chat_id), "text": fallback + (f"\n\n{INTERACTIVE_NOTE}" if interactive else ""),
        "markdown": False, "is_progress": False, "ts": ts, "task_id": task_id, "transport": transport,
        "card_id": identity["card_id"],
    })
    return True, "ok"


def send_card_state(bridge: Any, record: Dict[str, Any]) -> None:
    """The live half of a card change (new body or a recorded press); the store is the truth."""
    frame: Dict[str, Any] = {
        "type": "card_state", "card": card_identity(record), "submissions_summary": submissions_summary(record),
        "submissions_dropped": int(record.get("submissions_dropped") or 0), "ts": utc_now_iso(),
    }
    chat_id = record.get("publication_chat_id", record.get("chat_id"))
    if type(chat_id) is int:
        frame["chat_id"] = chat_id
    bridge.broadcast(frame)


__all__ = ["INTERACTIVE_NOTE", "card_identity", "send_card", "send_card_state"]
