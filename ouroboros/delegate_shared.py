"""Shared nanny-verb helpers: the typed refusal, the custody-rooted emit, and
run-ownership resolution.

Extracted from ``ouroboros/tools/delegate.py`` to break the import cycle the
delegate split left behind: ``delegate_interactions`` imported these three
helpers back from the facade (``tools/delegate`` → ``delegate_interactions`` →
``tools/delegate``), and the house seam pattern is one-way — an extracted
module never imports the facade back. ``tools.delegate`` re-exports all three,
so every existing reference and monkeypatch target keeps the same objects.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional, Tuple

from ouroboros import delegate_custody as custody
from ouroboros.delegate_custody import RunCustody as _RunCustody
from ouroboros.tools.registry import ToolContext

log = logging.getLogger(__name__)


def _fail(tool: str, code: str, detail: str, **extra: Any) -> str:
    payload = {"status": "refused", "tool": tool, "reason": code, "detail": detail, **extra}
    return json.dumps(payload, ensure_ascii=False, indent=2)


def engine_supports_workspace_root(gateway: Any) -> bool:
    from ouroboros.config import CLAUDEXOR_DELEGATED_WORKSPACE_ROOT_MIN_VERSION
    from ouroboros.gateways.claudexor import engine_at_least

    return engine_at_least(
        str(getattr(gateway, "engine_version", "") or ""),
        CLAUDEXOR_DELEGATED_WORKSPACE_ROOT_MIN_VERSION,
    )


def is_legacy_workspace_root_request(request: Any) -> bool:
    execution = request.get("execution") if isinstance(request, dict) else None
    return bool(
        isinstance(execution, dict)
        and request.get("mode", "agent") == "agent"
        and request.get("access") != "readonly"
        and execution.get("delegated") is True
        and execution.get("isolation") == "live"
        and "workspaceRoot" not in execution
    )


def workspace_root_recovery_block(gateway: Any, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    request = record.get("request") if isinstance(record, dict) else None
    execution = request.get("execution") if isinstance(request, dict) else None
    if not isinstance(execution, dict) or "workspaceRoot" not in execution:
        return None
    if engine_supports_workspace_root(gateway):
        return None
    return {
        "invocation_id": str(record.get("invocation_id") or ""),
        "task_id": str(record.get("task_id") or ""),
        "action": "recovery_blocked",
        "reason": "engine_rejects_delegated_workspace_root",
        "engine_version": str(getattr(gateway, "engine_version", "") or ""),
    }


def _emit(ctx: ToolContext, kind: str, payload: Dict[str, Any]) -> None:
    custody.emit(custody.custody_root(ctx), kind, {
        "task_id": str(getattr(ctx, "task_id", "") or ""), **payload,
    })


def _owned_run(ctx: ToolContext, tool: str, run_id: str) -> Tuple[Optional[str], Optional[_RunCustody]]:
    """Resolve custody for a run, or return a typed refusal payload.

    The daemon bearer token grants the ENTIRE Claudexor API, so a run id is not a
    capability the way a file descriptor is — anything that can name a run can reach it,
    read it, or CANCEL it, and cancelling a reviewer destroys the verdict that was the
    point of running it. Ownership is therefore replayed from the durable start row:
    a restarted worker keeps its runs, and an id with NO durable record is UNKNOWN
    (refused as unresolvable), which is a different fact from a run that demonstrably
    belongs to someone else.
    """
    status, entry = custody.lookup(custody.custody_root(ctx), str(getattr(ctx, "task_id", "") or ""), run_id)
    if status == custody.UNKNOWN:
        return _fail(tool, "run_ownership_unknown",
                     "No durable record of that run id exists on this drive, so ownership "
                     "cannot be established. Unknown ownership is refused, not waved through.",
                     run_id=run_id), None
    if status == custody.FOREIGN:
        return _fail(tool, "run_not_owned",
                     "That run belongs to another task. A delegated run may only be "
                     "waited on or cancelled by the task that started it.", run_id=run_id), None
    return None, entry


__all__ = [
    "_emit", "_fail", "_owned_run", "engine_supports_workspace_root",
    "is_legacy_workspace_root_request", "workspace_root_recovery_block",
]
