"""Write historical transport bindings as migration/custody test inputs."""
import uuid
from pathlib import Path
from typing import Any
from dataclasses import asdict
from ouroboros.utils import update_json_locked
from ouroboros.presence_bindings import (
    PresenceBinding, PresenceBindingError, PRESENCE_BINDINGS_SCHEMA_VERSION,
    _state_path, _BINDING_ID_RE, _read_state, _binding_from_payload,
)

def new_presence_binding_id() -> str:
    return uuid.uuid4().hex



def load_presence_binding(
    data_root: Path,
    transport_skill: str,
    binding_id: str,
) -> PresenceBinding:
    """Load one exact binding and prove it belongs to the authenticated skill."""

    if not _BINDING_ID_RE.fullmatch(str(binding_id or "")):
        raise PresenceBindingError("presence_binding_invalid_id", "binding_id")
    value = _read_state(data_root)["bindings"].get(binding_id)
    if value is None:
        raise PresenceBindingError("presence_binding_not_found", "binding_id")
    binding = _binding_from_payload(value, f"bindings.{binding_id}")
    if binding.transport_skill != transport_skill:
        raise PresenceBindingError("presence_binding_wrong_transport", "transport_skill")
    if not binding.enabled:
        raise PresenceBindingError("presence_binding_disabled", "binding.enabled")
    return binding



def save_presence_binding(data_root: Path, binding: PresenceBinding) -> PresenceBinding:
    """Owner-side upsert; transport callers receive only the opaque id."""

    if not isinstance(binding, PresenceBinding):
        raise PresenceBindingError("presence_binding_invalid_payload", "binding")
    path = _state_path(data_root)

    def _update(current: dict[str, Any]) -> dict[str, Any]:
        if not current:
            current = {"schema_version": PRESENCE_BINDINGS_SCHEMA_VERSION, "bindings": {}}
        if current.get("schema_version") != PRESENCE_BINDINGS_SCHEMA_VERSION or not isinstance(current.get("bindings"), dict):
            raise PresenceBindingError("presence_bindings_invalid_state", "presence_bindings")
        bindings = dict(current["bindings"])
        bindings[binding.binding_id] = asdict(binding)
        return {"schema_version": PRESENCE_BINDINGS_SCHEMA_VERSION, "bindings": bindings}

    try:
        update_json_locked(path, _update, strict_existing_dict=True)
    except PresenceBindingError:
        raise
    except (OSError, TimeoutError, ValueError) as exc:
        raise PresenceBindingError("presence_bindings_write_failed", "presence_bindings") from exc
    return binding
