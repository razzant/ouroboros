"""The facts of the recovery turn a Fault boot gives the mind (owner 2A), and the turn's words.

The previous generation's stop leaves interrupted roots without a saved point in two shapes. Its teardown
cancels them at once (``cancel_origin.source=server_shutdown``). A root the teardown did not reach is fenced
by THIS start's queue restore (``supervisor/queue_snapshot.py``): a cancel INTENT with
``source=snapshot_restore`` and ``reason=server_shutdown``, which the intent watchdog settles into
``cancelled`` only on a later tick, after the recovery turn is composed. So the turn reads both: the settled
rows and the open intents of this stop, each bound to it by the time of the cancel itself (``requested_at``;
``updated_at`` only for an older row without it). Only reads: the start order and the watchdog stay as they
are. Whether a settled root is open to Continue is ``owner_continue.continuation_eligibility``'s answer, and a
root already continued is not offered again. An unreadable store or intent projection is unknown, never "none".
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


def shutdown_cancel(origin: Dict[str, Any]) -> bool:
    """A cancel the stop itself made: the teardown's ``server_shutdown`` source, or the start's restore fence
    (``source=snapshot_restore`` with ``reason=server_shutdown``). Not the whole
    ``owner_continue.TECHNICAL_CANCEL_SOURCES`` set: it also holds ``owner_restart``, the owner's own stop."""
    source = str(origin.get("source") or "")
    return source == "server_shutdown" or (source == "snapshot_restore"
                                           and str(origin.get("reason") or "") == "server_shutdown")


def _of_this_stop(event_at: str, since: str) -> bool:
    return not since or event_at >= since


def interrupted_roots(root: pathlib.Path, since: str) -> Optional[Dict[str, List[Any]]]:
    """``{"cancelled": [ids open to Continue], "refused": [(id, refusal)], "cancelling": [ids], "unknown": [ids]}``
    for the roots the stop that began at ``since`` cut without a saved point; ``None`` when the rows or the
    intents cannot be read at all. An open intent whose own row cannot be read is that id's unknown, never a
    reason to drop the lists already read. Children are their roots' business."""
    from ouroboros.cancel_intents import read_active_intents
    from ouroboros.owner_continue import continuation_eligibility
    from ouroboros.task_results import list_task_results, load_task_result

    try:
        rows = list_task_results(root, statuses=["cancelled"])
        intents, unseen = read_active_intents(root)
        if unseen is None:
            raise ValueError("the cancel-intent projection is unreadable")
    except Exception:
        log.warning("Cancelled rows or cancel intents are unreadable; the recovery turn says so", exc_info=True)
        return None
    facts: Dict[str, List[Any]] = {"cancelled": [], "refused": [], "cancelling": [], "unknown": []}
    settled = set()
    for row in rows:
        origin = row.get("cancel_origin") if isinstance(row.get("cancel_origin"), dict) else {}
        task_id = str(row.get("task_id") or "")
        if not (task_id and shutdown_cancel(origin) and str(row.get("root_task_id") or task_id) == task_id
                and _of_this_stop(str(origin.get("requested_at") or row.get("updated_at") or ""), since)):
            continue
        settled.add(task_id)
        verdict = continuation_eligibility(row, task_id)
        refusal = "already_continued" if isinstance(row.get("continued_by"), dict) else verdict["refusal"]
        facts["refused" if refusal else "cancelled"].append((task_id, refusal) if refusal else task_id)
    for task_id, intent in intents.items():
        if (task_id in settled or not shutdown_cancel(intent)
                or not _of_this_stop(str(intent.get("requested_at") or ""), since)):
            continue
        try:
            row = load_task_result(root, task_id, strict=True) or {}
        except Exception:
            log.warning("The row of cancelling %s is unreadable; the recovery turn names it unknown", task_id,
                        exc_info=True)
            facts["unknown"].append(task_id)
            continue
        if str(row.get("root_task_id") or task_id) == task_id:
            facts["cancelling"].append(task_id)
    return {key: sorted(value) for key, value in facts.items()}


def recovery_turn_text(cause_words: str, ended_at: str, held: List[str], interrupted: Optional[Dict[str, List[Any]]],
                       root: pathlib.Path) -> str:
    """The recovery turn's message: facts, and the owner's one decision (2A: the mind decides and tells once)."""
    from ouroboros.places import paused_places_sentence

    if interrupted is None:
        cut = ("Interrupted roots with no saved point: unknown, the cancelled rows or cancel intents could not be "
               "read.",)
    else:
        refused = ", ".join(f"{task_id} ({refusal})" for task_id, refusal in interrupted["refused"])
        cut = ("Interrupted roots with no saved point, cancelled by the stop and open to Continue: "
               f"{', '.join(interrupted['cancelled']) or 'none'}.",
               f"Cancelled by the stop but not open to Continue: {refused}." if refused else "",
               "Being cancelled now by this start's restore (Continue is decided per root once its cancel "
               f"settles): {', '.join(interrupted['cancelling']) or 'none'}.",
               "Being cancelled now, but their rows could not be read (root and Continue unknown): "
               f"{', '.join(interrupted['unknown'])}." if interrupted.get("unknown") else "")
    return "\n".join(filter(None, (
        f"[recovery after a server fault] The previous server generation ended at {ended_at or 'an unknown time'}: "
        f"{cause_words}. This generation restarted itself with the same code.",
        "Tasks held with their saved state (Resume continues each from that state): "
        f"{', '.join(held) if held else 'none'}.",
        *cut,
        "You decide what to resume or continue, with your Resume and Continue tools, and tell the owner once.",
        paused_places_sentence(root),
    )))
