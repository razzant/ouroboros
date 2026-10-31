"""The agent's rollback: a ``rollback_request`` the supervisor answers through the UI door (P4-S).

The tool never resets the checkout inside a worker. A confirmed call emits one
``rollback_request`` event; the supervisor runs the Rollback button's own procedure
(``gateway.control.rollback_through_door``: lock, stop every writer, the authority switch
intent or the restore, restart), so the other tasks are interrupted at once and the facts
the store holds are exported before an older tree runs. The tool answers «requested».
"""

from __future__ import annotations

import logging
import pathlib
import subprocess
from typing import List, Tuple

from ouroboros.tools.registry import ToolContext, ToolEntry
from ouroboros.utils import utc_now_iso

log = logging.getLogger(__name__)


def _git(repo_dir: pathlib.Path, *args: str) -> Tuple[int, str]:
    result = subprocess.run(["git", *args], cwd=repo_dir, capture_output=True, text=True, timeout=10)
    return result.returncode, result.stdout.strip()


def _rollback_to_target(ctx: ToolContext, target: str, confirm: bool = False) -> str:
    """Preview the reset, or request it from the supervisor (the UI door's procedure)."""
    target = (target or "").strip()
    if not target:
        return "⚠️ ROLLBACK_ERROR: target parameter is required (tag name or commit SHA)."
    repo_dir = pathlib.Path(getattr(ctx, "serving_repo_dir", None) or ctx.repo_dir)  # rollback is of the RUNNING body
    try:
        rc, resolved = _git(repo_dir, "rev-parse", "--verify", target)
    except Exception as e:
        return f"⚠️ ROLLBACK_ERROR: git rev-parse failed: {e}"
    if rc != 0:
        return f"⚠️ ROLLBACK_ERROR: Cannot resolve '{target}' — not a valid tag or SHA."
    if not confirm:
        try:
            msg = _git(repo_dir, "log", "-1", "--format=%s", resolved)[1]
            head_sha = _git(repo_dir, "rev-parse", "HEAD")[1]
        except Exception:
            msg, head_sha = "(unknown)", "(unknown)"
        return (
            f"Will reset current branch to {target} ({resolved[:8]}):\n"
            f"  Target commit: {msg}\n"
            f"  Current HEAD:  {head_sha[:8]}\n\n"
            "A rescue snapshot of any uncommitted changes will be created first; every running task is "
            "interrupted and the server restarts.\n"
            "Call again with confirm=true to proceed."
        )
    queue = getattr(ctx, "event_queue", None)
    if queue is None:
        return "⚠️ ROLLBACK_ERROR: no supervisor is reachable from this context; use the UI Rollback button."
    try:
        queue.put_nowait({"type": "rollback_request", "target": target, "target_sha": resolved,
                          "reason": "agent_rollback_tool", "task_id": ctx.task_id, "ts": utc_now_iso()})
    except Exception as e:
        return f"⚠️ ROLLBACK_ERROR: the rollback request could not be queued: {e}"
    log.info("rollback_request emitted: %s (%s)", target, resolved[:8])
    return (f"✅ Rollback to {target} ({resolved[:8]}) requested. The supervisor stops every running task "
            "(this one included), restores the checkout — or records the authority switch an older tree needs "
            "and performs it at the restart exit — and restarts the server.")


def get_tools() -> List[ToolEntry]:
    return [
        ToolEntry("vcs_rollback", {
            "name": "vcs_rollback",
            "description": (
                "Roll Ouroboros's own running checkout (its current branch) back to a tag or commit SHA through "
                "the supervisor's Rollback procedure (the UI Restore button): every running task is interrupted, a rescue snapshot of "
                "uncommitted changes is created, the server restarts. "
                "Review-exempt mechanical recovery; the target's review status and compatibility are not guaranteed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "Tag name or commit SHA to reset to",
                    },
                    "confirm": {
                        "type": "boolean",
                        "description": "Must be true to execute. Without it, shows a preview.",
                    },
                },
                "required": ["target", "confirm"],
            },
        }, _rollback_to_target, is_code_tool=True),
    ]
