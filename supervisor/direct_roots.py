"""The supervisor's off-lock fragment of live direct-chat roots.

Pooled roots reach a worker through ``state/queue_snapshot.json``; direct-chat
turns live only in the server process's actor registry, so the main loop
writes this small projection beside the snapshot (owner decision 6C).  Actor
locks are only ever tried (``acquire(blocking=False)``): a turn mid-admission
is skipped and the fragment says so through ONE aggregate ``incomplete`` fact
rather than blocking the loop or fabricating a row.  Cleared on queue
init/restore so a stale process's turns never outlive it.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any, Dict

from ouroboros.utils import atomic_write_json, utc_now_iso

log = logging.getLogger(__name__)

FRAGMENT_NAME = pathlib.Path("state") / "direct_roots.json"


def _fragment_path(drive_root: Any) -> pathlib.Path:
    return pathlib.Path(drive_root) / FRAGMENT_NAME


def publish_direct_roots(drive_root: Any) -> Dict[str, Any]:
    """Write the current direct-root rows; never raises, never blocks on an actor."""
    rows = []
    incomplete = False
    try:
        from supervisor.active_activity import get_direct_activity_registry
        from supervisor.workers import direct_chat_turn

        for entry in get_direct_activity_registry().actors():
            lock = getattr(entry.actor, "_owner_message_admission_lock", None)
            if lock is None:
                continue
            if not lock.acquire(blocking=False):
                incomplete = True
                continue
            try:
                turn = direct_chat_turn(entry.activity_id)
            finally:
                lock.release()
            if turn is not None:
                rows.append({
                    "task_id": str(turn.get("id") or ""),
                    "title": str(turn.get("title") or "").strip(),
                    "chat_id": turn.get("chat_id"),
                    "project_id": str(turn.get("project_id") or ""),
                })
    except Exception:
        log.debug("direct roots projection failed", exc_info=True)
        incomplete = True
    payload = {"ts": utc_now_iso(), "roots": rows, "incomplete": incomplete}
    try:
        atomic_write_json(_fragment_path(drive_root), payload)
    except Exception:
        log.debug("direct roots fragment write failed", exc_info=True)
    return payload


def clear_direct_roots(drive_root: Any) -> None:
    """An empty fragment: no live direct roots are known for this process yet."""
    try:
        atomic_write_json(_fragment_path(drive_root), {"ts": utc_now_iso(), "roots": [], "incomplete": False})
    except Exception:
        log.debug("direct roots fragment clear failed", exc_info=True)
