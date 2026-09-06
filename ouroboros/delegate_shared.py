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
                     run_id=run_id,
                     hint="The run may belong to a different drive or the id may be "
                          "mistyped; get_task_result(<task_id>) is the ownership-free "
                          "way to read another task's delegated-run outcome."), None
    if status == custody.FOREIGN:
        # The refusal stays a refusal; the additive facts give the caller the
        # two things it needs to stop being stuck — the run is over, and whom
        # to ask (get_task_result(owner_task_id) is the legitimate cross-task
        # read that already carries the delegated_runs_* counters).
        return _fail(tool, "run_not_owned",
                     "That run belongs to another task. A delegated run may only be "
                     "waited on or cancelled by the task that started it.",
                     run_id=run_id,
                     owner_task_id=str(getattr(entry, "task_id", "") or ""),
                     run_settled=bool(getattr(entry, "settled", False)),
                     run_terminal_state=str(getattr(entry, "terminal_state", "") or "")), None
    return None, entry


def orphan_disposition_status(
    ctx: ToolContext, drive: Any, run_id: str,
) -> Tuple[str, Optional[_RunCustody], str]:
    """Custody for a DISPOSITION, with the orphan rule applied.

    An obligation is held by the run's durable rows, not by the task that
    created them. While the owning task is LIVE, only that identity may decide
    its captured patch. Once that task is terminal, a live TOP-LEVEL task may
    apply or reject the orphan; every apply-path guard still runs unchanged
    (recorded-target match, protected paths, proven drift, the whole-payload
    CAS), and the PATCH_DISPOSED row records who wrote it.

    This is a DISPOSITION-ONLY upgrade. ``_owned_run`` governs wait/cancel/
    answer and is deliberately NOT widened: cancelling or answering a foreign
    run destroys work instead of closing an obligation.

    Returns ``(status, entry, orphan_of)``, where ``orphan_of`` is the terminal
    owner's task id when the upgrade applied and "" otherwise.
    """
    from ouroboros.delegate_terminal import _task_is_terminal
    from ouroboros.tool_access import _TOP_LEVEL_PRINCIPAL_PROFILES, active_tool_profile

    status, entry = custody.lookup(drive, str(getattr(ctx, "task_id", "") or ""), run_id)
    if (status == custody.FOREIGN and entry is not None
            and entry.settled and not entry.patch_disposed
            and str(active_tool_profile(ctx)) in _TOP_LEVEL_PRINCIPAL_PROFILES
            and _task_is_terminal(drive, entry.task_id)):
        return custody.OWNED, entry, str(entry.task_id or "")
    return status, entry, ""


__all__ = ["_emit", "_fail", "_owned_run", "orphan_disposition_status"]
