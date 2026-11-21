"""The presentation-only source label of an outbound transport message.

A Project's message names its Project; named work of its own in Main names
that work; a direct reply to the owner, and any source the host cannot name,
carries no label (never a guessed Main, room or objective). The delivered room
is already chosen: never route by this label or cache it across a rename/move.
"""
from __future__ import annotations

import json
import logging
import pathlib

from ouroboros.contracts.chat_id_policy import WEB_UI_CHAT_ID
from ouroboros.project_facts import project_display_name
from ouroboros.projects_registry import RoomResolver, project_of_room, task_presentation_name
from ouroboros.store import acquisition_budget, open_store

log = logging.getLogger(__name__)
_MAX_ANCESTORS = 8
_HINTS = ("parent_task_id", "root_task_id", "delegation_role")


def _project_label(root, chat_id) -> str:
    project = project_of_room(root, chat_id)
    return project_display_name(project.get("name"), project.get("id")) if project else ""


def _work_root(resolver, task_id: str, hints: dict) -> str:
    """The top work of ``task_id``'s tree; "" when a recorded link is missing or cyclic."""
    current, facts, seen = task_id, {**hints, **resolver.lineage(task_id)}, set()
    while current not in seen and len(seen) < _MAX_ANCESTORS:
        seen.add(current)
        root_id, parent = str(facts.get("root_task_id") or ""), str(facts.get("parent_task_id") or "")
        if root_id and (root_id != current or not parent):  # a worker is never its own root
            return root_id
        if not parent:
            # Unknown, or a worker whose parent is not recorded, has no nameable source.
            return current if facts and facts.get("delegation_role") != "subagent" else ""
        current, facts = parent, resolver.lineage(parent)
    return ""


def _owner_chat(root, chat_id) -> bool:
    """Whether ``chat_id`` is the owner's own bound conversation.

    A transport chat shares the numeric range with Project rooms and other
    people's rooms. Only the owner bindings in ``state.json`` (the external
    slot a transport's own owner command fills, or the owner's primary chat)
    prove it; an unknown or unreadable binding proves nothing.
    """
    if type(chat_id) is not int or chat_id <= 0 or chat_id == WEB_UI_CHAT_ID:
        return False
    from supervisor import state

    path = pathlib.Path(root) / "state" / "state.json"
    for key in ("owner_external_chat_id", "owner_chat_id"):
        known, bound = state.control_in_copy(path, key)
        if known and type(bound) is int and bound == chat_id:
            return True
    return False


def _owner_chat_from_provenance(root, chat_id: int, metadata: dict, origin_ref) -> bool:
    """Accept a confirmed owner door when no external slot was written.

    Telegram-first and strict/safe-command installs can admit owner work without
    forwarding a host slash command.  In that case state proves the owner while
    the task row carries transport-neutral admission facts.  This remains a
    presentation-only fallback; project/direct/Presence exclusions happen before
    it is called.
    """
    from supervisor import state

    path = pathlib.Path(root) / "state" / "state.json"
    owner_known, owner_id = state.control_in_copy(path, "owner_id")
    external_known, external_chat = state.control_in_copy(path, "owner_external_chat_id")
    if not owner_known or owner_id is None or not external_known or external_chat is not None:
        return False
    if isinstance(origin_ref, str):
        try:
            origin_ref = json.loads(origin_ref)
        except (TypeError, ValueError):
            origin_ref = None
    ref = origin_ref if isinstance(origin_ref, dict) else metadata.get("origin_message_ref")
    try:
        ref_chat = int(ref.get("chat_id")) if isinstance(ref, dict) and ref.get("chat_id") is not None else None
    except (TypeError, ValueError):
        ref_chat = None
    if ref_chat is not None and ref_chat != chat_id:
        return False
    work_kind = str(metadata.get("work_kind") or "").strip()
    if work_kind == "owner":
        return True
    if ref_chat == chat_id and (
        bool(metadata.get("continuation"))
        or bool(metadata.get("continuation_of"))
        or bool(metadata.get("client_surface"))
    ):
        return True
    # Scheduled roots are addressed to the owner's chat by schedule admission;
    # the scheduler deliberately strips transport-specific metadata.
    return bool(metadata.get("schedule_id")) and str(metadata.get("source") or "").strip() == "scheduled_task"


def _owner_work(root, chat_id: int, body_chat, presence_turn, source, metadata, origin_ref) -> bool:
    """Named owner work in the owner's chat, from its already-read root row.

    The work must belong to this very chat and not be carried by Presence;
    how it started (a transport message, Continue, a schedule, /review, the
    API) does not matter.
    """
    if chat_id == WEB_UI_CHAT_ID:
        return True
    try:
        if body_chat is None or int(body_chat) != chat_id:
            return False
    except (TypeError, ValueError):
        return False
    if presence_turn or str(source or "").strip() == "presence":
        return False
    try:
        metadata = json.loads(metadata) if isinstance(metadata, str) else {}
    except (TypeError, ValueError):
        metadata = {}
    from ouroboros.dialogue_provenance import presence_record_binding
    if isinstance(metadata, dict) and presence_record_binding({"metadata": metadata}):
        return False
    return _owner_chat(root, chat_id) or _owner_chat_from_provenance(root, chat_id, metadata, origin_ref)


def message_origin(root, chat_id, task_id="", *, event=None):
    """``{"label": name}`` for a nameable source, else ``{}``."""
    event = event if isinstance(event, dict) else {}
    task_id = str(event.get("subagent_task_id") or task_id or "").strip()
    if root is None:
        return {}
    try:
        with acquisition_budget(0.05):
            project = project_of_room(root, chat_id)
            if project:
                label = project_display_name(project.get("name"), project.get("id"))
                return {"label": label} if label else {}
            if not task_id:
                return {}
            # A Project's work speaks for its Project wherever it is delivered (its notes in Main too).
            resolver = RoomResolver(root)
            hints = {key: str(event[key]) for key in _HINTS if event.get(key)}
            if project_chat := resolver(task_id, hints.get("parent_task_id", ""), hints.get("root_task_id", "")):
                label = _project_label(root, project_chat)
                return {"label": label} if label else {}
            work = _work_root(resolver, task_id, hints) if type(chat_id) is int and (
                chat_id == WEB_UI_CHAT_ID or chat_id > 0
            ) else ""
            if not work or (work == task_id and event.get("_is_direct_chat") is True):
                return {}
            with open_store(root, write=False) as txn:
                row = txn.conn.execute(
                    "SELECT substr(json_extract(body, '$.title'),1,160), "
                    "substr(json_extract(body, '$.suggested_name'),1,160), "
                    "json_extract(body, '$._is_direct_chat'), "
                    "json_extract(body, '$.chat_id'), "
                    "json_extract(body, '$._presence_turn'), "
                    "json_extract(body, '$.source'), "
                    "json_extract(body, '$.metadata'), "
                    "json_extract(body, '$.origin_message_ref') "
                    "FROM task_bodies WHERE task_id=?", (work,)).fetchone()
            if not row or row[2]:
                return {}
            if not _owner_work(root, chat_id, *row[3:8]):
                return {}
            # Only the work's own title or coined name: a bare request is no name here.
            label = task_presentation_name({"title": row[0], "suggested_name": row[1]}, fallback="")
            return {"label": label} if label else {}
    except Exception:
        # Failed presentation lookup never refuses the message, nor guesses its source.
        log.debug("Outbound source label unavailable", exc_info=True)
        return {}


def stamp_message_origin(root, event, *, chat_id, task_id=""):
    """Last writer after producer metadata; identity is the sender's explicit selection."""
    event["message_origin"] = message_origin(root, chat_id, task_id, event=event)
