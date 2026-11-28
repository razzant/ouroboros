"""A links row's actions in Telegram: an address is a button; a place inside Ouroboros is a line saying where to go.

A place (``ouroboros/link_actions.py``) opens the owner's own interface on the computer running Ouroboros, such as a
skill's card where it signs in to a service. A phone usually cannot finish that sign-in: the provider returns it to
that computer. So the bot says, before anything starts, where to finish it, instead of drawing a button that leads
nowhere (owner's vision of 2026-11-27).
"""
from __future__ import annotations

import re
from typing import Any, List, Tuple

_MAX = 12  # the shared link-action cap: ouroboros.link_actions._MAX_LINK_ACTIONS
# The core's place: a canonical skill name (letters, digits, "-", "_", "."; not starting or ending with "." or "_") and
# a declared service id (ouroboros.link_actions.sign_in_place).
_PLACE = re.compile(r"skill-sign-in/((?![._])[\w.-]{1,64}(?<![._]))/([a-z0-9_]{1,40})")


def link_rows(raw_actions: Any) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """``(buttons, places)``: ``(label, url)`` for each address and ``(label, skill)`` for each skill's sign-in place."""
    buttons: List[Tuple[str, str]] = []
    places: List[Tuple[str, str]] = []
    for action in raw_actions if isinstance(raw_actions, list) else []:
        if not isinstance(action, dict):
            continue
        label, url = str(action.get("label") or "").strip(), str(action.get("url") or "").strip()
        place = _PLACE.fullmatch(str(action.get("place") or "").strip())
        if label and url:
            buttons.append((label, url))
        elif label and place:
            places.append((label, place.group(1)))
    return buttons[:_MAX], places[:_MAX]


def finish_lines(places: List[Tuple[str, str]], texts: Any) -> str:
    """One line per place, in the bridge's language table (``texts``: ``_LOCALIZED_TEXTS[lang]``)."""
    return "\n".join(texts.format("finish_on_computer", label=label, skill=skill) for label, skill in places)
