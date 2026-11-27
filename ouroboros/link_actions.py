"""The actions of a links row (``send_links``): a button per HTTP(S) address, or per place inside Ouroboros.

One validator serves the tool and the message bus: a batch is cleaned whole or refused whole. A place opens the
owner's own interface where the owner acts; today one kind exists: ``skill-sign-in/<skill>/<service>``, a skill's
card where its sign-in to that service is chosen and made (DESIGN "Sign-ins on a skill's card"). A place carries no
address, account or secret, and pressing it changes nothing by itself.
"""
from __future__ import annotations

import ipaddress
import re
from typing import Any, Dict, List, Optional, Tuple

_MAX_LINK_ACTIONS = 12


class LinkActionsValidationError(ValueError):
    """Typed atomic refusal from the shared link-action validator."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def validate_link_actions(actions: Any) -> List[Dict[str, str]]:
    """Return one cleaned batch of HTTP(S) and place actions, or refuse the entire batch."""
    from urllib.parse import urlparse

    if not isinstance(actions, list) or not actions:
        raise LinkActionsValidationError(
            "SEND_LINKS_ARG_ERROR", "provide a non-empty links array."
        )
    if len(actions) > _MAX_LINK_ACTIONS:
        raise LinkActionsValidationError(
            "SEND_LINKS_TOO_MANY", f"maximum {_MAX_LINK_ACTIONS} links."
        )
    cleaned: List[Dict[str, str]] = []
    for item in actions:
        if not isinstance(item, dict):
            raise LinkActionsValidationError(
                "SEND_LINKS_ARG_ERROR", "each link must contain label and url."
            )
        if "place" in item:
            cleaned.append(_place_action(item))
            continue
        label = str(item.get("label") or "")
        url = str(item.get("url") or "")
        invalid_url_char = any(
            ord(char) < 0x20 or ord(char) == 0x7F or char.isspace() for char in url
        )
        invalid_label_char = any(
            ord(char) < 0x20 or ord(char) == 0x7F
            or (char.isspace() and char != " ") for char in label
        )
        label = label.strip()
        if len(url) > 2048 or invalid_url_char:
            raise LinkActionsValidationError(
                "SEND_LINKS_URL_BLOCKED",
                "URL contains disallowed characters or exceeds 2048 characters.",
            )
        try:
            parsed = urlparse(url)
            hostname = parsed.hostname
            port = parsed.port
            if "[" in parsed.netloc:
                ipaddress.ip_address(hostname or "")
        except ValueError as exc:
            raise LinkActionsValidationError(
                "SEND_LINKS_URL_BLOCKED", "URL has an invalid authority."
            ) from exc
        if parsed.scheme not in {"http", "https"}:
            raise LinkActionsValidationError(
                "SEND_LINKS_URL_BLOCKED", "only http:// and https:// URLs are allowed."
            )
        if not hostname or (port is not None and not 0 <= port <= 65535):
            raise LinkActionsValidationError(
                "SEND_LINKS_URL_BLOCKED", "URL has an invalid authority."
            )
        if "\\" in parsed.netloc:
            raise LinkActionsValidationError("SEND_LINKS_URL_BLOCKED", "URL has an invalid authority.")
        invalid_reg_name = "[" not in parsed.netloc and any(
            not ((char.isascii() and (char.isalnum() or char in "._~-")) or
                 (not char.isascii() and char.isalpha())) for char in hostname
        )
        if "%" in hostname or invalid_reg_name:
            raise LinkActionsValidationError(
                "SEND_LINKS_URL_BLOCKED", "URL has an invalid authority."
            )
        if not label or invalid_label_char:
            raise LinkActionsValidationError(
                "SEND_LINKS_ARG_ERROR", "each link requires a label and absolute URL."
            )
        cleaned.append({"label": label[:120], "url": url})
    return cleaned


_SIGN_IN_PLACE = re.compile(r"skill-sign-in/([a-z0-9][a-z0-9_-]{0,63})/([A-Za-z0-9][A-Za-z0-9_.-]{0,63})")


def sign_in_place(place: Any) -> Optional[Tuple[str, str]]:
    """``(skill, service)`` of a sign-in place, else None."""
    found = _SIGN_IN_PLACE.fullmatch(str(place or ""))
    return (found.group(1), found.group(2)) if found else None


def _place_action(item: Dict[str, Any]) -> Dict[str, str]:
    label = str(item.get("label") or "")
    if item.get("url") or sign_in_place(item.get("place")) is None:
        raise LinkActionsValidationError(
            "SEND_LINKS_ARG_ERROR", "a place is skill-sign-in/<skill>/<service>, given instead of a url."
        )
    if not label.strip() or any(ord(char) < 0x20 or ord(char) == 0x7F or (char.isspace() and char != " ")
                                for char in label):
        raise LinkActionsValidationError("SEND_LINKS_ARG_ERROR", "each link requires a label.")
    return {"label": label.strip()[:120], "place": str(item["place"])}
