"""The supervisor's off-lock fragment of live direct-chat roots.

Pooled roots reach a worker through ``state/queue_snapshot.json``; direct-chat
turns live only in the server process's actor registry, so the main loop
collects this projection for generation publication beside the snapshot. Actor
locks are only ever tried (``acquire(blocking=False)``): a turn mid-admission
is skipped and the fragment says so through ONE aggregate ``incomplete`` fact
rather than blocking the loop or fabricating a row.  Queue init takes the
roster over and clears it in the same step, so a stale process's turns never
outlive it, and snapshot restore learns which direct roots the stop caught,
including the durable ones of an ended process that the roster never listed.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any, Dict

from ouroboros.utils import append_jsonl, atomic_write_json, read_json_dict, utc_now_iso
from ouroboros.focus import compact_focus as _compact_focus

log = logging.getLogger(__name__)

FRAGMENT_NAME = pathlib.Path("state") / "direct_roots.json"


def _fragment_path(drive_root: Any) -> pathlib.Path:
    return pathlib.Path(drive_root) / FRAGMENT_NAME


def direct_turn_facts(turn: Dict[str, Any]) -> Dict[str, Any]:
    """The live turn's suggested name, start and typed origin; absent stays absent."""
    from ouroboros.peer_roster import iso_from_epoch, typed_origin

    facts: Dict[str, Any] = {}
    suggested = str(turn.get("suggested_name") or "").strip()
    if suggested:
        facts["suggested_name"] = suggested
    started = iso_from_epoch(turn.get("_started_at"))
    if started:
        facts["started_at"] = started
    origin = typed_origin(turn)
    if origin:
        facts["origin"] = origin
    return facts


def collect_direct_roots() -> Dict[str, Any]:
    """Collect the current direct-root rows; never raises, never blocks on an actor."""
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
                row = {
                    "task_id": str(turn.get("id") or ""),
                    "title": str(turn.get("title") or "").strip(),
                    "chat_id": turn.get("chat_id"),
                    "project_id": str(turn.get("project_id") or ""),
                }
                # Host facts the live actor already holds, carried until the
                # turn's durable result exists (the roster prefers that result).
                row.update(direct_turn_facts(turn))
                focus = _compact_focus(turn.get("focus"))
                if focus is not None:
                    row["focus"] = focus
                rows.append(row)
    except Exception:
        log.debug("direct roots projection failed", exc_info=True)
        incomplete = True
    return {"roots": rows, "incomplete": incomplete}


def clear_direct_roots(drive_root: Any) -> None:
    """An empty fragment: no live direct roots are known for this process yet."""
    try:
        atomic_write_json(_fragment_path(drive_root), {"ts": utc_now_iso(), "roots": [], "incomplete": False})
    except Exception:
        log.debug("direct roots fragment clear failed", exc_info=True)


def take_direct_roots(drive_root: Any) -> Dict[str, Any]:
    """Hand the PREVIOUS process's rows over and clear the fragment in one step.

    Queue init is the moment the roster stops describing anything live, so it is
    also the last moment those ids exist: reading here keeps the clear exactly
    where it was — a stale process's turns never outlive it however the boot
    continues — while snapshot restore still learns which direct roots the stop
    caught.  An ``incomplete`` roster skipped a turn that was mid-admission: the
    rows it DOES list are real and are handed over, the skipped turn keeps the
    projection it has today, and the flag rides along so the restore record
    discloses that gap.  Never raises.
    """
    payload = read_json_dict(_fragment_path(drive_root))
    clear_direct_roots(drive_root)
    if not isinstance(payload, dict):
        return {"task_ids": [], "incomplete": False}
    incomplete = bool(payload.get("incomplete"))
    rows = payload.get("roots")
    task_ids = [] if not isinstance(rows, list) else [
        str(row.get("task_id") or "")
        for row in rows
        if isinstance(row, dict) and str(row.get("task_id") or "")
    ]
    return {"task_ids": task_ids, "incomplete": incomplete}


def owner_process_ended(drive_root: Any, result: Dict[str, Any]) -> bool:
    """Whether a durable direct row's own server process has positively ended.

    A direct turn runs only inside the server process that holds the data root's
    admission until it exits (``server_entrypoint.admit_data_root``), and its
    execution owner names that process generation (``execution_owner_record``).
    While THIS process holds the admission, a bound direct owner stamped by
    another generation names a turn whose process is gone. Nothing weaker is
    this fact: absence from the roster or registry, row age, a worker boot, or
    an in-process supervisor revival (the same generation, so the same stamp).
    An unstamped legacy row, an owner bound elsewhere and any positive live
    observation stay unknown.
    """
    from ouroboros.process_custody import current_custody_session_id
    from ouroboros.server_entrypoint import admission_held
    from ouroboros.task_status import task_execution_observation

    owner = result.get("execution_owner") if isinstance(result.get("execution_owner"), dict) else {}
    stamp = str(owner.get("process") or "")
    if not stamp or stamp == current_custody_session_id() or not admission_held(pathlib.Path(drive_root)):
        return False
    observed = task_execution_observation(drive_root, result)
    return (observed["kind"], observed["state"], observed["reason"]) == ("direct", "unknown", "live_execution_unobserved")


def adopt_orphaned_direct_results(drive_root: Any, taken: Dict[str, Any]) -> Dict[str, Any]:
    """Add the direct turns a crash left durable but unlisted to the roster's handover.

    The roster is a projection: a machine that died mid-turn may never have
    rewritten it, so it can miss a turn whose durable result still says
    ``running``. Queue init adds every such row whose own process positively
    ended (``owner_process_ended``) to the ids the roster handed over, and
    snapshot restore treats it as any direct root the stop caught: saved work
    and pauses wait for Resume, the rest is fenced for the one cancellation
    custody, which keeps an answer the host already holds. The roster's ids stay
    first and untouched; a failed read adds nothing, leaving every row exactly
    as unknown as it was. Never raises.
    """
    from ouroboros.task_results import STATUS_RUNNING, list_task_results

    handed_over = [str(task_id) for task_id in taken.get("task_ids") or [] if str(task_id)]
    known, adopted = set(handed_over), []
    try:
        for row in list_task_results(drive_root, statuses=[STATUS_RUNNING]):
            task_id = str(row.get("task_id") or "")
            if task_id and task_id not in known and owner_process_ended(drive_root, row):
                known.add(task_id)
                adopted.append(task_id)
    except Exception:
        log.debug("direct-result adoption sweep failed", exc_info=True)
        adopted = []
    if adopted:
        try:
            append_jsonl(
                pathlib.Path(drive_root) / "logs" / "supervisor.jsonl",
                {"ts": utc_now_iso(), "type": "direct_roots_adopted_orphans",
                 "task_ids": adopted},
            )
        except Exception:
            log.debug("direct-result adoption event append failed", exc_info=True)
    return {"task_ids": handed_over + adopted, "incomplete": bool(taken.get("incomplete"))}
