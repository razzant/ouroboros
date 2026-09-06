"""Small process-qualified projections for extension lifecycle receipts."""

from __future__ import annotations

from typing import Any, Mapping


def extension_process_receipt(state: Mapping[str, Any] | None) -> dict[str, str]:
    """Keep process custody attached when runtime state crosses the gateway."""
    source = state or {}
    return {
        "process": str(source.get("process") or ""),
        "server_reconcile": str(source.get("server_reconcile") or ""),
    }


def extension_reconcile_receipt(
    skill_name: str, state: Mapping[str, Any]
) -> dict[str, Any]:
    """Project the public reconcile response without dropping process custody."""
    return {
        "skill": skill_name,
        "extension_action": state.get("action"),
        "extension_reason": state.get("reason"),
        "live_loaded": bool(state.get("live_loaded")),
        "load_error": state.get("load_error"),
        **extension_process_receipt(state),
    }


__all__ = ["extension_process_receipt", "extension_reconcile_receipt"]
