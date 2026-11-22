"""Owner chat routing mode and outbound mirror policy for the unified Telegram skill.

One authority decides both halves of the owner DM experience:

1. ``TELEGRAM_OWNER_CHAT_MODE`` (``direct`` | ``presence``, default ``direct``):
   whether ordinary owner DM messages go straight to Main owner ingress
   (``/chat/inject``) or into Presence custody (Circles / Focus). Owner slash
   commands, ``/menu``, ``/mode``, ``/main <text>``, ``/panic``, quiz cards,
   and secret-request guards always stay on the immediate owner control lane.

2. ``TELEGRAM_MIRROR_MODE`` (``all`` | ``main`` | ``telegram_only`` | ``control_only``,
   default ``all``): which host events are mirrored out to the owner's pinned
   Telegram DM. Presence conversation rooms (``chat_id >= 2**40``) are owned by
   the Presence outbox and are never mirrored through ``chat.outbound``, preventing
   double delivery in every mode.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from ouroboros.contracts.chat_id_policy import WEB_UI_CHAT_ID, is_project_chat_id

OWNER_CHAT_MODE_DIRECT = "direct"
OWNER_CHAT_MODE_PRESENCE = "presence"
VALID_OWNER_CHAT_MODES = frozenset({OWNER_CHAT_MODE_DIRECT, OWNER_CHAT_MODE_PRESENCE})

MIRROR_MODE_ALL = "all"
MIRROR_MODE_MAIN = "main"
MIRROR_MODE_TELEGRAM_ONLY = "telegram_only"
MIRROR_MODE_CONTROL_ONLY = "control_only"
VALID_MIRROR_MODES = frozenset({
    MIRROR_MODE_ALL,
    MIRROR_MODE_MAIN,
    MIRROR_MODE_TELEGRAM_ONLY,
    MIRROR_MODE_CONTROL_ONLY,
})

# Matches ``ouroboros.places.conversation_chat_id`` and ``dialogue_store._OWNER_ROOMS``.
PRESENCE_CHAT_ID_MIN = 1 << 40

_TRUE_VALUES = frozenset({"on", "true", "1", "yes"})
# Owner-targeted System notices delivered even in narrowed mirror modes.
_OWNER_NOTICE_TYPES = frozenset({"reminder", "skill_notice", "safety_notice"})
# Main-pinned Project chatter filtered out in ``main`` mode (unless Telegram-originated).
_PROJECT_MAIN_NOISE_TYPES = frozenset({
    "project_started",
    "project_handoff",
    "project_completion_summary",
    "relay_entry",
})


def owner_chat_mode(settings: Dict[str, Any]) -> str:
    """Return normalized owner DM routing mode (``direct`` or ``presence``)."""
    raw = str(settings.get("TELEGRAM_OWNER_CHAT_MODE") or OWNER_CHAT_MODE_DIRECT).strip().lower()
    return raw if raw in VALID_OWNER_CHAT_MODES else OWNER_CHAT_MODE_DIRECT


def mirror_mode(settings: Dict[str, Any]) -> str:
    """Return normalized outbound mirror mode."""
    raw = str(settings.get("TELEGRAM_MIRROR_MODE") or MIRROR_MODE_ALL).strip().lower()
    return raw if raw in VALID_MIRROR_MODES else MIRROR_MODE_ALL


def mirror_quizzes_enabled(settings: Dict[str, Any]) -> bool:
    """Whether non-Telegram-originated question cards mirror in ``main`` / ``control_only``."""
    raw = str(settings.get("TELEGRAM_MIRROR_QUIZZES") or "on").strip().lower()
    return raw in _TRUE_VALUES


def _event_room_chat_id(event: Dict[str, Any]) -> Optional[int]:
    raw = event.get("chat_id")
    if raw is None or isinstance(raw, bool):
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def is_presence_room_event(event: Dict[str, Any]) -> bool:
    """Whether ``event`` belongs to a Presence room owned by the Presence outbox.

    Presence conversation rooms occupy ``chat_id >= 2**40`` (``places.conversation_chat_id``)
    or carry ``presence_provenance``. Private contact rows sent by a Presence focus to Main
    (``presence_contact`` / ``system_type == 'main_notice'``) are Main owner notices,
    never Presence room echoes, and return False here.
    """
    room_id = _event_room_chat_id(event)
    if room_id is not None and room_id >= PRESENCE_CHAT_ID_MIN:
        return True
    if _is_main_contact_notice(event):
        return False
    if isinstance(event.get("presence_provenance"), dict) and event.get("presence_provenance"):
        return True
    return str(event.get("task_type") or "").strip().lower() == "presence" and room_id != WEB_UI_CHAT_ID


def _is_main_contact_notice(event: Dict[str, Any]) -> bool:
    """Explicit private message/notice addressed to Main (including Presence -> Main contact)."""
    if str(event.get("system_type") or "") == "main_notice":
        return True
    if isinstance(event.get("presence_contact"), dict):
        return True
    progress_meta = event.get("progress_meta")
    return isinstance(progress_meta, dict) and isinstance(progress_meta.get("presence_contact"), dict)


def _is_owner_system_notice(event: Dict[str, Any]) -> bool:
    return event.get("role") == "system" and str(event.get("system_type") or "") in _OWNER_NOTICE_TYPES


def target_chat(settings: Dict[str, Any], event: Dict[str, Any]) -> int:
    """Return the Telegram owner chat ID that should receive ``event``, or 0 to skip."""
    if not isinstance(event, dict) or is_presence_room_event(event):
        return 0
    mode = mirror_mode(settings)
    configured = str(settings.get("TELEGRAM_CHAT_ID") or "").strip()
    transport = event.get("transport") if isinstance(event.get("transport"), dict) else {}
    from_telegram = transport.get("kind") == "telegram"
    if not configured:
        # Unbound owner chat: never mirror to a previously seen or stale conversation_id.
        return 0
    try:
        chat_id = int(configured)
    except ValueError:
        return 0
    if chat_id <= 0:
        return 0
    if mode == MIRROR_MODE_ALL:
        return chat_id
    if mode == MIRROR_MODE_TELEGRAM_ONLY:
        if from_telegram or _is_owner_system_notice(event):
            return chat_id
        return 0
    is_quiz = bool(event.get("quiz_id"))
    if mode == MIRROR_MODE_CONTROL_ONLY:
        if from_telegram or _is_owner_system_notice(event) or _is_main_contact_notice(event):
            return chat_id
        if is_quiz and mirror_quizzes_enabled(settings):
            return chat_id
        return 0
    # mode == MIRROR_MODE_MAIN: Main replies, explicit main_notice, owner notices,
    # quizzes (unless disabled), and Telegram-originated conversations; no raw
    # Project room chatter, subagent cards, or Main-pinned Project lifecycle noise.
    if from_telegram or _is_owner_system_notice(event) or _is_main_contact_notice(event):
        return chat_id
    if is_quiz:
        return chat_id if mirror_quizzes_enabled(settings) else 0
    if event.get("subagent_event"):
        return 0
    if str(event.get("role") or "").strip().lower() == "user":
        return 0
    sys_type = str(event.get("system_type") or "")
    if sys_type in _PROJECT_MAIN_NOISE_TYPES:
        return 0
    room_id = _event_room_chat_id(event)
    if room_id is not None and is_project_chat_id(room_id):
        return 0
    return chat_id


__all__ = [
    "MIRROR_MODE_ALL",
    "MIRROR_MODE_CONTROL_ONLY",
    "MIRROR_MODE_MAIN",
    "MIRROR_MODE_TELEGRAM_ONLY",
    "OWNER_CHAT_MODE_DIRECT",
    "OWNER_CHAT_MODE_PRESENCE",
    "PRESENCE_CHAT_ID_MIN",
    "VALID_MIRROR_MODES",
    "VALID_OWNER_CHAT_MODES",
    "is_presence_room_event",
    "mirror_mode",
    "mirror_quizzes_enabled",
    "owner_chat_mode",
    "target_chat",
]
