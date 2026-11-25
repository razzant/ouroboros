"""``POST /api/command``'s agent-note variant (#3873): a program's words for Main, under the author it declares.

A note is exactly ``text`` and ``author: {kind: "agent", label}``, optionally with the Reply citation and boot
precondition. Other fields (including ordinary commands and owner controls) are refused before any note effect.
The existing Panic command retains priority in ``api_command``, outside the note parser. An older host reads only
``cmd`` and cannot record a note as its owner's words. The host builds the author through
its one constructor (``dialogue_provenance.recorded_author``, basis ``self_declared``), accepts the row durably
(``message_ingress.accept_local_message``, its owed dispatch recovered after a crash) and queues it as a Main
message. The note body is literal text and my human's words nowhere: never parsed as a command,
never owner activity, delivered to a running Main reader as an agent's note or opening an ordinary Main turn without
the owner door's stamp (``server_owner_routing``). The answer names the recorded author, the row's address and
``delivery``: ``queued`` once the row is saved and handed to the queue (CLIENT_CONTRACT §15).
"""
from __future__ import annotations

import asyncio
import uuid
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse

from ouroboros.gateway._helpers import boot_precondition, json_error, request_drive_root

NOTE_FIELDS = frozenset({"text", "author", "reply_to", "reply_excerpt", "if_boot"})


def note_request(body: Any) -> tuple[str, dict]:
    """``(text, recorded author)`` of a well-formed note, else ValueError saying what is wrong."""
    from ouroboros.dialogue_provenance import AGENT_AUTHOR, recorded_author
    from ouroboros.owner_mailbox import TASK_MESSAGE_MAX_CHARS

    extra = sorted(set(body) - NOTE_FIELDS) if isinstance(body, dict) else ["(not an object)"]
    if extra:
        raise ValueError(f"an agent note carries text and author only, not {', '.join(extra)}")
    author, text = body.get("author"), body.get("text")
    if not isinstance(author, dict) or set(author) != {"kind", "label"} or author["kind"] != AGENT_AUTHOR:
        raise ValueError('author must be exactly {"kind": "agent", "label": "..."}')
    if not isinstance(text, str) or len(text) > TASK_MESSAGE_MAX_CHARS or not text.strip():
        raise ValueError(f"an agent note needs 1 to {TASK_MESSAGE_MAX_CHARS} characters of text")
    if not isinstance(body.get("reply_to", ""), str):
        raise ValueError("reply_to must be an address string")
    return text, recorded_author(author["label"])


def _accept(bridge: Any, root: Any, text: str, author: dict, body: dict) -> dict:
    """The durable row, written before it is queued; a malformed Reply citation raises ValueError before any write."""
    from supervisor.message_ingress import accept_local_message

    # This door has no owner client. Only the supplied citation rides the task metadata.
    metadata = {key: body[key] for key in ("reply_to", "reply_excerpt") if key in body}
    row, _rejoined = accept_local_message(bridge, root, text, author=author, chat_id=1, user_id=0, source="web",
                                          client_message_id=f"agent-note-{uuid.uuid4().hex}", task_metadata=metadata)
    return row


async def api_agent_note(request: Request, body: Any) -> JSONResponse:
    try:
        text, author = note_request(body)
    except ValueError as exc:
        return json_error(str(exc), 400, accepted=False)
    if (refused := boot_precondition(body)) is not None:
        return refused
    from supervisor.message_bus import try_get_bridge

    bridge = try_get_bridge()
    if bridge is None:
        return json_error("Complete provider setup before leaving a note.", 409, accepted=False)
    from ouroboros.store import StoreError

    try:
        row = await asyncio.to_thread(_accept, bridge, request_drive_root(request), text, author, body)
    except ValueError as exc:  # the Reply it cites cannot be read or quoted
        return json_error(str(exc), 400, accepted=False)
    except (RuntimeError, StoreError):  # storage or queue custody may already have changed
        return json_error("Agent note acceptance is unconfirmed.", 503)
    from ouroboros.chat_chain import format_address, row_address

    return JSONResponse({"status": "ok", "accepted": True, "author": author,
                         "row_address": format_address(row_address(row)), "delivery": "queued"})
