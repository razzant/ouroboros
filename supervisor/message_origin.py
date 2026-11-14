"""Small, presentation-only origin facts for outbound transport messages.

The delivered room is already chosen. Never route by these labels, inspect a full
queue/result, or cache a heading across a rename or a move.
"""
from __future__ import annotations

import logging

from ouroboros.projects_registry import RoomResolver, project_of_room, task_presentation_name
from ouroboros.store import acquisition_budget, open_store

log = logging.getLogger(__name__)
_MAX_ANCESTORS = 8


def message_origin(root, chat_id, task_id="", *, event=None):
    """Project membership and bounded emitting-task ancestry; missing facts stay absent."""
    event = event if isinstance(event, dict) else {}
    room = {"kind": "unknown"}
    if type(chat_id) is int and chat_id == 1:
        room = {"kind": "main"}
    elif root is not None:
        try:
            project = project_of_room(root, chat_id)
            if project:
                room = {"kind": "project", "name": str(project.get("name") or project["id"])}
        except Exception:
            log.debug("Outbound room label unavailable", exc_info=True)
    task_id = str(event.get("subagent_task_id") or task_id or "")
    worker = bool(event.get("subagent_task_id") or event.get("parent_task_id")
                  or event.get("delegation_role") == "subagent")
    direct = event.get("_is_direct_chat") is True and not worker
    origin = {"room": room, "tasks": ([{"id": task_id, "name": "", "worker": worker}] if task_id and not direct else [])}
    if not root or not task_id:
        return origin
    try:
        with acquisition_budget(0.05):
            resolver = RoomResolver(root)
            current, seen, chain = task_id, set(), []
            while current and current not in seen and len(chain) < _MAX_ANCESTORS:
                seen.add(current)
                facts = resolver.lineage(current)
                # Scheduled events may precede the durable task row.
                if current == task_id:
                    facts = {**{key: event[key] for key in ("parent_task_id", "root_task_id", "delegation_role")
                                if event.get(key)}, **facts}
                chain.append((current, facts))
                current = str(facts.get("parent_task_id") or "")
            # A deep tree can hide its middle without losing the explicitly recorded root work.
            root_id = str(chain[0][1].get("root_task_id") or "")
            if current and root_id and root_id not in seen:
                chain.append((root_id, resolver.lineage(root_id)))
            titles = {}
            with open_store(root, write=False) as txn:
                for tid, _facts in chain:
                    row = txn.conn.execute(
                        "SELECT substr(json_extract(body, '$.title'),1,160), "
                        "substr(json_extract(body, '$.suggested_name'),1,160), "
                        "json_extract(body, '$._is_direct_chat') FROM task_bodies WHERE task_id=?", (tid,)).fetchone()
                    if row:
                        titles[tid] = row
            origin["tasks"] = []
            for tid, facts in reversed(chain):
                title, suggested, direct = titles.get(tid, ("", "", False))
                worker = bool(facts.get("parent_task_id") or facts.get("delegation_role") == "subagent"
                              or tid == task_id and event.get("subagent_task_id"))
                if (direct or event.get("_is_direct_chat") is True) and not worker and tid == task_id:
                    continue
                origin["tasks"].append({"id": tid, "name": task_presentation_name(
                    {"title": title, "suggested_name": suggested}, fallback=""), "worker": worker})
                if worker and not facts.get("parent_task_id"):
                    origin["incomplete"] = True
            if current:
                origin["incomplete"] = True
    except Exception:
        # Failed presentation lookup never refuses the message. No guessed Main/root.
        log.debug("Outbound task label unavailable", exc_info=True)
    return origin


def stamp_message_origin(root, event, *, chat_id, task_id=""):
    """Last writer after producer metadata; identity is the sender's explicit selection."""
    event["message_origin"] = message_origin(root, chat_id, task_id, event=event)
