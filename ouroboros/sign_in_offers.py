"""«Connected» in the conversation that offered a skill's sign-in (owner's vision of 2026-11-27).

When the mind gives the owner a button to a skill's sign-in (a ``send_links`` place
``skill-sign-in/<skill>/<service>``, ``ouroboros/link_actions.py``), the conversation it was given in is noted per
skill and service (``note``). When a sign-in for that skill and service then completes as signed in, from any
surface, the host says so once in that conversation as a System row (``announce``): by then the mind's turn has
usually ended and no mind speaks (DESIGN "Chat authorship and System rows"). Nothing else is kept: no attempt, no
account, no address. One small file, ``state/sign_in/offers.json``, at most one entry per skill and service; an entry
leaves when it is announced, and a newer offer replaces it.
"""
from __future__ import annotations

import logging
import pathlib
import threading
from typing import Any, Iterable, Mapping

from ouroboros.link_actions import sign_in_place
from ouroboros.utils import atomic_write_json, read_json_dict, utc_now_iso

log = logging.getLogger(__name__)

_LOCK = threading.Lock()
CONNECTED_TEMPLATE = "{skill} is signed in to {service} now."


def _path(data_dir: Any) -> pathlib.Path:
    return pathlib.Path(data_dir) / "state" / "sign_in" / "offers.json"


def note(data_dir: Any, chat_id: int, actions: Iterable[Mapping[str, Any]]) -> None:
    """Remember the conversation each sign-in place of a delivered links row was offered in. Never raises."""
    offered = [found for found in (sign_in_place(action.get("place")) for action in actions or ()) if found]
    if not offered or not chat_id or not data_dir:
        return
    try:
        with _LOCK:
            saved = read_json_dict(_path(data_dir)) or {}
            for skill, service in offered:
                saved[f"{skill}/{service}"] = {"chat_id": int(chat_id), "at": utc_now_iso()}
            atomic_write_json(_path(data_dir), saved)
    except Exception as exc:  # noqa: BLE001 — the links row is delivered either way
        log.warning("A sign-in offer could not be noted (%s)", type(exc).__name__)


def announce(data_dir: Any, skill: str, service: str, label: str) -> bool:
    """One System row «<skill> is signed in to <service> now.» in the conversation that offered this sign-in, if one
    did; the offer leaves first, so the row is said at most once. Never raises; True when a row was sent."""
    key = f"{skill}/{service}"
    try:
        with _LOCK:
            saved = read_json_dict(_path(data_dir)) or {}
            offer = saved.pop(key, None)
            if not isinstance(offer, dict) or not offer.get("chat_id"):
                return False
            atomic_write_json(_path(data_dir), saved)
        from ouroboros import host_notice

        text, fields = host_notice.phrase("sign_in_connected", CONNECTED_TEMPLATE, skill=skill, service=label or service)
        from supervisor.message_bus import send_with_budget

        send_with_budget(int(offer["chat_id"]), text, role="system", system_type="sign_in_connected", progress_meta=fields)
        return True
    except Exception as exc:  # noqa: BLE001 — the sign-in itself already succeeded
        log.warning("A sign-in's «connected» line could not be said (%s)", type(exc).__name__)
        return False
