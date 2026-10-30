"""Additive stored-row identity shared by live chat and attachment frames."""
from typing import Any, Literal, TypedDict


class ChatReplyReference(TypedDict, total=False):
    """A ``reply_to`` reference as the host resolved it (``chat_references.reference_from_row``); an
    unresolved one is ``{address, unresolved: true}``."""
    address: str
    chat_id: int
    role: Literal["user", "assistant", "system"]
    author_label: str
    kind: str
    snippet: str
    ts: str
    task_id: str
    target_home: Literal["work", "room", "none"]  # what decides the quoted row's room; an older reference lacks it
    unresolved: bool


class ChatReferenceFields(TypedDict, total=False):
    row_address: str
    row_client_message_id: str  # a row's own saved name (outgoing or host-fact ingress); never the owner's submission id
    reply_to: ChatReplyReference
    relay: dict[str, Any]
    reply_to_relay: str
    kind: str
    room_revisions: dict[str, int]  # a row shown in several rooms: the revision it took in each room it counted in


class SharedRowFrame(TypedDict, total=False):
    also_rooms: list[int]  # a live chat frame of a row other Projects show too: those rooms (``room_appearances.frame_rooms``)
