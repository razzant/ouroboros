"""Shared chat id policy for human-visible and synthetic conversations.

The negative chat id range is reserved for synthetic agent-to-agent traffic
that must not appear in human dialogue history or browser chat streams. Source
attribution belongs in message payloads (``source``), not in the numeric id.

Positive ids above ``PROJECT_CHAT_ID_MIN`` are per-project chats (v6.32.0,
multi-project): the SAME web owner (user_id stays 1 — owner binding is
security-load-bearing and never re-binds) talking to the same single agent in
a project-scoped thread. The main chat stays ``WEB_UI_CHAT_ID``, and
``HIDDEN_CHAT_ID`` is the hidden partition below it.
"""

from __future__ import annotations

import hashlib

WEB_UI_CHAT_ID = 1

# The hidden partition: the Skill Review panel plus every headless task admitted
# without a registered project. 0 is a REAL, addressable destination that no
# browser surface reads (a chat_id=0 history query coerces to Main, and the Main
# filter drops chat-0 rows), never "no chat" — the ``if chat_id:`` truthiness
# that read it as missing is the class this constant exists to end.
HIDDEN_CHAT_ID = 0

# Reserved for A2A-like synthetic conversations. Legacy A2A generated
# unbounded negative ids (-1001, -1002, ...), so every negative id remains
# internal for history/memory isolation.
A2A_CHAT_ID_MIN = -3999
A2A_CHAT_ID_MAX = -1

# Project chats occupy a high positive range so they can never collide with
# the main chat or small hand-assigned ids.
PROJECT_CHAT_ID_MIN = 1000


def is_a2a_chat_id(chat_id: object) -> bool:
    """Return True when ``chat_id`` belongs to synthetic A2A traffic."""
    try:
        value = int(chat_id)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return value < 0


def is_internal_chat_id(chat_id: object) -> bool:
    """Return True for synthetic chat ids hidden from human-facing history."""
    return is_a2a_chat_id(chat_id)


def is_project_chat_id(chat_id: object) -> bool:
    """Return True for per-project owner chats (positive high range)."""
    try:
        value = int(chat_id)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return value >= PROJECT_CHAT_ID_MIN


def project_chat_id(project_id: str) -> int:
    """Deterministic chat id for a project (stable across restarts/devices).

    Derived from the sanitized project id so the registry, history rows, and
    WS frames all agree without a separate allocator file. 28 bits of hash
    above the floor keeps collision odds negligible for realistic project
    counts while staying inside JS safe-integer range.
    """
    pid = str(project_id or "").strip()
    if not pid:
        return WEB_UI_CHAT_ID
    digest = hashlib.sha256(pid.encode("utf-8")).digest()
    return PROJECT_CHAT_ID_MIN + (int.from_bytes(digest[:4], "big") & 0x0FFFFFFF)


__all__ = [
    "A2A_CHAT_ID_MAX",
    "A2A_CHAT_ID_MIN",
    "HIDDEN_CHAT_ID",
    "PROJECT_CHAT_ID_MIN",
    "WEB_UI_CHAT_ID",
    "is_a2a_chat_id",
    "is_internal_chat_id",
    "is_project_chat_id",
    "project_chat_id",
]
