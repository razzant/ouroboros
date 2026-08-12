"""Shared chat id policy for human-visible and synthetic conversations.

The negative chat id range is reserved for synthetic agent-to-agent traffic
that must not appear in human dialogue history or browser chat streams. Source
attribution belongs in message payloads (``source``), not in the numeric id.

Positive ids above ``PROJECT_CHAT_ID_MIN`` are per-project chats (v6.32.0,
multi-project): the SAME web owner (user_id stays 1 — owner binding is
security-load-bearing and never re-binds) talking to the same single agent in
a project-scoped thread. The main chat stays ``WEB_UI_CHAT_ID``.
"""

from __future__ import annotations

import hashlib

WEB_UI_CHAT_ID = 1

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


# The digest slice minted above PROJECT_CHAT_ID_MIN. Deliberately NOT widened
# past 28 bits: ``GET /api/chat/history`` clamps its ``chat_id`` parameter to
# 2**31 - 1, so a wider space would silently become unreachable over the wire,
# and re-slicing would CHANGE every existing project's chat id — breaking the
# "thread #0 == today's project chat" identity below. Uniqueness is held by the
# registry-wide reservation invariant (projects_registry), not by hash width.
_CHAT_ID_HASH_MASK = 0x0FFFFFFF

# Thread #0 IS the project's own chat: every existing history row, task binding
# and test keeps its numbers, so threads need no migration.
MAIN_THREAD_ID = 0


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
    return PROJECT_CHAT_ID_MIN + (int.from_bytes(digest[:4], "big") & _CHAT_ID_HASH_MASK)


def thread_chat_id(project_id: str, thread_id: object = MAIN_THREAD_ID) -> int:
    """Deterministic chat id for ONE THREAD of a project.

    ``thread_chat_id(pid, 0) == project_chat_id(pid)`` EXACTLY — that identity
    is what makes threads a zero-migration addition. Every other thread derives
    from a distinct ``<pid>#<n>`` pre-image in the same 28-bit space.

    Determinism means a collision is possible in principle; this function stays
    allocator-free and stateless (the documented SSOT shape of this module) and
    the registry holds the uniqueness invariant — it simply RETRIES with the
    next thread id, which is free because thread ids are opaque integers.
    """
    pid = str(project_id or "").strip()
    if not pid:
        return WEB_UI_CHAT_ID
    try:
        index = int(thread_id)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        index = MAIN_THREAD_ID
    if index == MAIN_THREAD_ID:
        return project_chat_id(pid)
    digest = hashlib.sha256(f"{pid}#{index}".encode("utf-8")).digest()
    return PROJECT_CHAT_ID_MIN + (int.from_bytes(digest[:4], "big") & _CHAT_ID_HASH_MASK)


__all__ = [
    "A2A_CHAT_ID_MAX",
    "A2A_CHAT_ID_MIN",
    "MAIN_THREAD_ID",
    "PROJECT_CHAT_ID_MIN",
    "WEB_UI_CHAT_ID",
    "is_a2a_chat_id",
    "is_internal_chat_id",
    "is_project_chat_id",
    "project_chat_id",
    "thread_chat_id",
]
