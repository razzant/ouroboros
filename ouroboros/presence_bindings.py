"""Read frozen transport bindings for migration and retained-work authentication."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ouroboros.contracts.skill_manifest import canonical_skill_name

PRESENCE_BINDINGS_FILENAME = "presence_bindings.json"
PRESENCE_BINDINGS_SCHEMA_VERSION = 1
_BINDING_ID_RE = re.compile(r"^[0-9a-f]{32}$")


class PresenceBindingError(ValueError):
    def __init__(self, code: str, field: str) -> None:
        self.code = str(code or "presence_binding_invalid")
        self.field = str(field or "binding")
        super().__init__(f"{self.code}: {self.field}")


def _text(value: Any, field: str, *, empty: bool = False) -> str:
    if not isinstance(value, str) or value != value.strip() or (not empty and not value):
        raise PresenceBindingError("presence_binding_invalid_string", field)
    return value


@dataclass(frozen=True)
class PresenceEndpoint:
    """Provider account plus an exact room or explicit account-wide ``*`` scope."""

    transport: str
    account_id: str
    conversation_id: str
    thread_id: str = ""

    def __post_init__(self) -> None:
        _text(self.transport, "endpoint.transport")
        _text(self.account_id, "endpoint.account_id")
        _text(self.conversation_id, "endpoint.conversation_id")
        _text(self.thread_id, "endpoint.thread_id", empty=True)
        if self.conversation_id == "*" and self.thread_id:
            raise PresenceBindingError(
                "presence_binding_wildcard_thread_invalid",
                "endpoint.thread_id",
            )


@dataclass(frozen=True)
class PresenceBinding:
    """One revocable owner decision joining a transport scope to one profile."""

    binding_id: str
    transport_skill: str
    behavior_skill: str
    origin: PresenceEndpoint
    destination: PresenceEndpoint
    enabled: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.binding_id, str) or not _BINDING_ID_RE.fullmatch(self.binding_id):
            raise PresenceBindingError("presence_binding_invalid_id", "binding.binding_id")
        for field in ("transport_skill", "behavior_skill"):
            value = _text(getattr(self, field), f"binding.{field}")
            if canonical_skill_name(value) != value:
                raise PresenceBindingError("presence_binding_invalid_skill", f"binding.{field}")
        if not isinstance(self.origin, PresenceEndpoint) or not isinstance(self.destination, PresenceEndpoint):
            raise PresenceBindingError("presence_binding_invalid_endpoint", "binding")
        if self.destination.conversation_id == "*":
            raise PresenceBindingError(
                "presence_binding_destination_must_be_exact",
                "binding.destination.conversation_id",
            )
        if not isinstance(self.enabled, bool):
            raise PresenceBindingError("presence_binding_invalid_enabled", "binding.enabled")


def conversation_key(provider: str, account_id: str, conversation_id: str, thread_id: str = "") -> str:
    """The one concurrency/history identity of a provider conversation (no thread = ``0``)."""
    parts = (provider, account_id, conversation_id, thread_id or "0")
    return ":".join(str(part or "") for part in parts)


def _state_path(data_root: Path) -> Path:
    return Path(data_root) / "state" / PRESENCE_BINDINGS_FILENAME


def _endpoint_from_payload(value: Any, field: str) -> PresenceEndpoint:
    if not isinstance(value, Mapping) or set(value) != {
        "transport", "account_id", "conversation_id", "thread_id"
    }:
        raise PresenceBindingError("presence_binding_invalid_endpoint", field)
    return PresenceEndpoint(**value)


def _binding_from_payload(value: Any, field: str) -> PresenceBinding:
    if not isinstance(value, Mapping) or set(value) != {
        "binding_id", "transport_skill", "behavior_skill", "origin", "destination", "enabled"
    }:
        raise PresenceBindingError("presence_binding_invalid_payload", field)
    return PresenceBinding(
        binding_id=value["binding_id"],
        transport_skill=value["transport_skill"],
        behavior_skill=value["behavior_skill"],
        origin=_endpoint_from_payload(value["origin"], f"{field}.origin"),
        destination=_endpoint_from_payload(value["destination"], f"{field}.destination"),
        enabled=value["enabled"],
    )


def _read_state(data_root: Path) -> dict[str, Any]:
    path = _state_path(data_root)
    if not path.exists():
        return {"schema_version": PRESENCE_BINDINGS_SCHEMA_VERSION, "bindings": {}}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PresenceBindingError("presence_bindings_unreadable", "presence_bindings") from exc
    if not isinstance(value, dict) or set(value) != {"schema_version", "bindings"}:
        raise PresenceBindingError("presence_bindings_invalid_state", "presence_bindings")
    if value["schema_version"] != PRESENCE_BINDINGS_SCHEMA_VERSION or not isinstance(value["bindings"], dict):
        raise PresenceBindingError("presence_bindings_invalid_state", "presence_bindings")
    return value


def list_presence_bindings(data_root: Path) -> tuple[PresenceBinding, ...]:
    state = _read_state(data_root)
    return tuple(
        _binding_from_payload(value, f"bindings.{binding_id}")
        for binding_id, value in sorted(state["bindings"].items())
    )


__all__ = [
    "PRESENCE_BINDINGS_FILENAME",
    "PRESENCE_BINDINGS_SCHEMA_VERSION",
    "PresenceBinding",
    "PresenceBindingError",
    "PresenceEndpoint",
    "conversation_key",
    "list_presence_bindings",
]
