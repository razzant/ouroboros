"""The actions of a links row (``send_links``): a button per HTTP(S) address, or per place inside Ouroboros.

One validator serves the tool and the message bus: a batch is cleaned whole or refused whole. A place opens the
owner's own interface where the owner acts; today one kind exists: ``skill-sign-in/<skill>/<service>``, a skill's
card where its sign-in to that service is chosen and made (DESIGN "Sign-ins on a skill's card"). A place carries no
address, account or secret, and pressing it changes nothing by itself.
"""
from __future__ import annotations

import ipaddress
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
        if str(item.get("place") or "").strip():  # an empty place beside a url is no place
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


def sign_in_place(place: Any) -> Optional[Tuple[str, str]]:
    """``(skill, service)`` of a sign-in place, else None: a canonical skill name and a declared service id, exactly as
    the host's sign-in answers name them (``canonical_skill_name``, ``service_sign_in_declaration``)."""
    from ouroboros.contracts.skill_manifest import canonical_skill_name
    from ouroboros.service_sign_in_declaration import _ID as service_id

    parts = str(place or "").split("/")
    if len(parts) != 3 or parts[0] != "skill-sign-in" or not parts[1] or canonical_skill_name(parts[1]) != parts[1]:
        return None
    return (parts[1], parts[2]) if service_id.fullmatch(parts[2]) else None


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
