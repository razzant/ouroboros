"""The retired direct-roots fragment: the boot fold of a previous generation's file, nothing else.

Direct turns once lived only in the server process's actor registry, so the main
loop published ``state/direct_roots.json`` beside the queue snapshot by trying each
actor's lock. Every focus now owns a RUNNING row (the conversation lane's carriers
and the two inline executors alike), so the queue snapshot is the one roster and
nothing writes or reads this fragment any more. Queue init still takes a RESIDUAL
file over — a previous generation of the body may have left live rows in it — and
removes it in the same step, so a stale process's turns never outlive it, and
snapshot restore learns which direct roots the stop caught, including the durable
ones of an ended process that the roster never listed. An older body's publisher
recreates the file on its first tick after a rollback.
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any, Dict

from ouroboros.utils import append_jsonl, read_json_dict, utc_now_iso

log = logging.getLogger(__name__)

FRAGMENT_NAME = pathlib.Path("state") / "direct_roots.json"


def _fragment_path(drive_root: Any) -> pathlib.Path:
    return pathlib.Path(drive_root) / FRAGMENT_NAME


def take_direct_roots(drive_root: Any) -> Dict[str, Any]:
    """Hand the PREVIOUS process's rows over and remove the fragment in one step.

    Queue init is the moment the roster stops describing anything live, so it is
    also the last moment those ids exist: reading here keeps the removal exactly
    where the clear was — a stale process's turns never outlive it however the
    boot continues — while snapshot restore still learns which direct roots the
    stop caught.  An ``incomplete`` roster skipped a turn that was mid-admission:
    the rows it DOES list are real and are handed over, the skipped turn keeps the
    projection it has today, and the flag rides along so the restore record
    discloses that gap.  Never raises.
    """
    path = _fragment_path(drive_root)
    payload = read_json_dict(path)
    try:
        path.unlink(missing_ok=True)
    except OSError:
        log.debug("direct roots fragment removal failed", exc_info=True)
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

    An inline turn runs only inside the server process that holds the data root's
    admission until it exits (``server_entrypoint.admit_data_root``), and its
    execution owner names that process generation (``execution_owner_record``); a
    lane carrier adopts the same generation id, so its row reads alike. While
    THIS process holds the admission, a bound direct owner stamped by another
    generation names a turn whose process is gone. Nothing weaker is this fact:
    absence from the roster or registry, row age, a worker boot, or an in-process
    supervisor revival (the same generation, so the same stamp). An unstamped
    legacy row, an owner bound elsewhere and any positive live observation stay unknown.
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
