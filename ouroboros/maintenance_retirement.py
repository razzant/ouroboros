"""History-sized filesystem retirement, owned by the maintenance child.

Prunes report their earliest retained item's age eligibility. Producers schedule
new artifacts; startup and explicit repair discover legacy ones. Directory times
and unrelated completions never wake a sweep. Age grants no deletion authority.
Source removers (a retired copy, snapshot or drive) schedule the ``bytecode`` step
at once; a step that stops at its unlink budget owes the rest through ``next_due``.
The ``review_threads`` step lets the engine retire a finished reviewer thread only
after its export is retained here (``review_thread_retention``), and over the same tasks releases the
managed document imports a finished work can no longer name (``delegate_documents``); an import that
opens its obligation before the step's first pass sets where that pass starts reading
(``release_opened``). A managed import owner's terminal write schedules this step even without local files.
Otherwise waiting tasks, and the one-time walks while open (``delegate_result_taken``, ``historical_retirement``), run it
only in a child running for another job; it yields between operations
to a delegate capture that child's capture pass never had, the rest waiting for the next run.
What the step newly saved of the engine's answers (``observed``) reaches ``logs/events.jsonl`` once.
Task trees and service logs retire only behind the recovery gate. A pass that finds it closed (no
startup recorded, a startup gap, unreadable recovery links) says so on its job row as the typed hold
``source_recovery_pending``, its example naming the gap's first error or the first link that could not be
followed, which Storage and the mind's ``storage`` fact read; an open gate leaves no hold.
"""
from __future__ import annotations

import errno
import json
import logging
import os
import pathlib
import re
import stat
import sys
import uuid
from typing import Any

from ouroboros import maintenance_jobs, store
from ouroboros.record_compression import WIRE_JOURNAL
from ouroboros.retention import get_gc_retention_days, time
from ouroboros.unindexed import BYTECODE, LEGACY_BYTECODE
from ouroboros.utils import fsync_folder

log = logging.getLogger(__name__)
JOB = "runtime_retirement"
STARTUP = "startup_recovery"
RECOVERY_HOLD = "source_recovery_pending"  # the class ``headless.prune_task_trees`` gives a tree kept for recovery
BYTECODE_UNLINKS_PER_RUN = 20_000  # removals of one run; the rest is owed to the next (FSEvents pacing)


def record_startup(root, *, gaps, report=None):
    """Publish the recovery gate before a child may retire task sources. Missing means unknown.
    A gap also publishes how many records the recovery ``report`` left unresolved (its error lines
    when it names none, one without a report), a trace for the operator that no pass reads, and its
    first error, cut to a hold example's bound: the example of the pass's hold (``_startup_hold``),
    which carries no count. Every start also publishes
    ``workers_unconfirmed``, which ``record_compression``'s log plane reads beside the generation:
    true unless the recovery report says the previous generation's workers ended
    (``prior_workers_ended``, set only once its PID check passed; one of them may still hold a log
    by inode). No report, a recovery skipped or one that returned before that check confirms
    nothing. A start reads no history for it: the record costs the same on any install."""
    from ouroboros.plan_collection_job import recover_startup

    errors = [str(error) for error in (report or {}).get("errors") or ()]
    fact = {"generation": uuid.uuid4().hex, "gaps": bool(gaps),
            "workers_unconfirmed": (report or {}).get("prior_workers_ended") is not True}
    if gaps:
        unresolved = {item for item in (report or {}).get("unresolved") or () if item != "*"}
        example = errors[0] if errors else "startup recovery report missing" if report is None else ""
        fact.update(count=len(unresolved) or len(errors) or 1, example=example[:maintenance_jobs.HOLD_EXAMPLE_CHARS])
    with store.open_store(root, write=True) as txn:
        txn.set_meta(STARTUP, fact)
        recover_startup(txn)


def _startup_hold(startup):
    """The hold a pass reports while the recorded startup keeps the recovery gate closed, ``None``
    once a startup recorded no gap. No record is unknown, held like a gap. Its ``count`` is ``None``:
    the gate keeps every task tree and service log without listing them, so how many it holds is
    not known, and the recovery's own count is of records, not of what is held."""
    if not isinstance(startup, dict):
        return {"class": RECOVERY_HOLD, "count": None, "example": "no startup recovery record"}
    if startup.get("gaps") is False:
        return None
    return {"class": RECOVERY_HOLD, "count": None, "example": str(startup.get("example") or "")}


def _inputs(root):
    with store.open_store(root, write=False) as txn:
        return {"startup": txn.meta(STARTUP), "retention": get_gc_retention_days()}


def schedule_in(txn, eligible_at):
    """A producer's own artifact, never a polling interval or an install-wide counter."""
    row = txn.meta(maintenance_jobs.PREFIX + JOB) or {}
    due = min(row.get("next_due") or eligible_at, eligible_at)
    maintenance_jobs.record_in(txn, JOB, next_due=due, arrival=uuid.uuid4().hex)


def artifact_changed(root, *, timestamp=None, immediate=False):
    """Call after publishing disposable bytes, or (``immediate``) after removing a source tree whose
    bytecode mirror is owed now. A missed notification waits for startup/repair."""
    try:
        stamp = time.time() if timestamp is None else float(timestamp)
        with store.acquisition_budget(0), store.open_store(root, write=True) as txn:
            schedule_in(txn, stamp + (0 if immediate else get_gc_retention_days() * 86400))
    except Exception:
        log.debug("Retirement publication awaits startup discovery", exc_info=True)


def release_opened(txn, *, restored_tasks=()):
    """A managed document import opened its release obligation in this custody transaction
    (``delegate_documents.opens_release``). Until the ``review_threads`` step's first pass, keep the
    terminal-write position this transaction stands at as that pass's ``floor`` (the lowest wins), so
    its first pin goes no later and the owner's terminal write, always after this point, is read. An
    active step's cursor already reads every later write: nothing is written then. A floor written
    after the first pass read the job row is at or above its pin, so that race loses nothing.
    Explicit store import/fold instead supplies the restored owners: their terminal positions
    were rebuilt, so keep the owners in this step's waiting set and schedule its existing job."""
    from ouroboros.review_thread_retention import JOB_FIELD

    row = txn.meta(maintenance_jobs.PREFIX + JOB) or {}
    state = row.get(JOB_FIELD) if isinstance(row.get(JOB_FIELD), dict) else {}
    if restored_tasks:
        # A store rebuild writes task rows before it restores custody. Their new terminal
        # positions cannot be reconstructed from an old cursor: re-offer the actual owners.
        waiting = sorted(set(state.get("waiting") or ()) | set(restored_tasks))
        maintenance_jobs.record_in(txn, JOB, **{JOB_FIELD: {**state, "waiting": waiting}})
        schedule_in(txn, time.time())
        return
    position = int(txn.meta(store.TERMINAL_WRITES) or 0)
    if not state.get("activated_at") and not (type(state.get("floor")) is int and state["floor"] <= position):
        maintenance_jobs.record_in(txn, JOB, **{JOB_FIELD: {**state, "floor": position}})


def task_written(txn, task_id, columns):
    """A task can wake retirement only for its own physical inputs or managed import obligations."""
    from ouroboros.task_status import SETTLED_STATUSES
    if txn.root is None or columns.get("status") not in SETTLED_STATUSES:
        return
    if (txn.get_row("custody_marks", task_id) or {}).get("managed_imports"):
        # An import can end before START and leave no filesystem retirement input. Its terminal
        # write still owes this pass, including one racing the first activation's cursor read.
        schedule_in(txn, time.time())
    names = (f"task_trees/{task_id}", f"services/{task_id}",
             f"state/delegate_recovery/{task_id}.json", f"state/delegate_supervision/{task_id}.json",
             f"task_results/artifacts/{task_id}/{WIRE_JOURNAL}")
    deadlines = []
    for name in names:
        try:
            stamp = (txn.root / name).stat().st_mtime
            if name.startswith("task_trees/") and columns.get("drive_retention_at"):
                stamp = float(columns["drive_retention_at"])
        except (OSError, TypeError, ValueError):
            continue
        deadlines.append(stamp + get_gc_retention_days() * 86400)
    if holds_worked_copy(txn.root / "task_results" / "artifacts" / task_id):  # owed to ``prune_worked_copies``
        row = txn.get_row(WORKED_COPIES, task_id)  # ``settled_at``: when the host first wrote it settled, kept after
        txn.put_row(WORKED_COPIES, task_id, {"write_seq": columns.get("write_seq"), "settled_at": (row or {}).get("settled_at") or time.time()})
        try:  # a new obligation is owed at least now; a later write of an old work only by an end still ahead
            deadline = float(columns["drive_retention_at"]) + get_gc_retention_days() * 86400
        except (KeyError, TypeError, ValueError):
            deadline = 0.0
        deadlines += [max(deadline, time.time())] if row is None or deadline > time.time() else []
    if deadlines:
        schedule_in(txn, min(deadlines))


def _owed(row, inputs):
    """The history-sized steps' own facts: a stopped pass, other inputs (``inputs()``, read only then), or a deadline
    reached. Reached includes its own instant: a step that stops at its budget owes the rest at ``time.time()``, and on a
    coarse clock (Windows, Python 3.10: 15.6 ms) the next read can return that same value."""
    return (row.get("state") == "pending" or row.get("inputs") != inputs()
            or row.get("next_due") is not None and time.time() >= row["next_due"])


def due(root, row, facts, child):
    """The history-sized steps are owed; or, in a child already running for another job, the ``review_threads``
    step left tasks waiting (``review_thread_retention.owed``), its one-time ``taken`` walk is open
    (``delegate_result_taken.walk_open``) or its historical walk is open or waits (``historical_retirement.owed``),
    never by itself a reason to spawn one."""
    history_sized = _owed(row, lambda: _inputs(root))
    if history_sized or not child:
        return history_sized
    from ouroboros import historical_retirement
    from ouroboros.delegate_result_taken import walk_open
    from ouroboros.review_thread_retention import owed

    return owed(row) or walk_open(row) or historical_retirement.owed(row)


def _capture_arrived(root):
    """``arrived()``: is a delegate capture ready that this child's own capture pass was never offered
    (``delegate_capture_job.offered``)? An unread fact is no arrival: the capture job reads it itself."""
    from ouroboros.delegate_capture_job import offered, ready_requests

    had = offered(root)

    def arrived():
        try:
            return bool(ready_requests(root) - had)
        except Exception:
            return False

    return arrived


def run(root, should_stop):
    from ouroboros.code_intelligence import prune_stale_code_intel_roots
    from ouroboros.delegate_state_sweep import sweep_settled_delegate_state
    from ouroboros.extension_reconcile_queue import prune_failed_reconcile_markers
    from ouroboros.headless import prune_task_trees
    from ouroboros.review_thread_retention import retire_finished_review_threads
    from ouroboros.skill_uninstall_state import sweep_uninstalled_skill_state
    from ouroboros.startup_task_files import startup_tree_exclusions
    from ouroboros.tools.services import prune_service_logs
    from ouroboros.utils import append_jsonl, utc_now_iso

    inputs, reports, errors = _inputs(root), {}, []
    before = maintenance_jobs.job(root, JOB) or {}
    owed = _owed(before, lambda: inputs)
    arrived = _capture_arrived(root)  # the child's capture pass ran first: only a request it never had yields
    judged, hold = False, None  # the recovery gate as this pass read it: its hold while closed, None once open
    # Finished reviewer threads: exported and verified here first, then released by the engine. Between operations a
    # capture that arrived ends the step, its rest waiting for the child's next run. Alone when only its own work is
    # owed (``due``): the history-sized steps neither run nor change their facts.
    steps = [("review_threads", lambda root: retire_finished_review_threads(
        root, should_stop=lambda: should_stop() or arrived()))]
    if owed:
        steps = [("delegate_state", sweep_settled_delegate_state),
                 ("skill_state", sweep_uninstalled_skill_state), ("code_intel", prune_stale_code_intel_roots),
                 ("failed_reconcile", prune_failed_reconcile_markers), ("agent_media", prune_agent_media_uploads),
                 ("call_wire", prune_call_wire_journals), ("worked_copies", prune_worked_copies), *steps,
                 ("bytecode", prune_stale_bytecode)]
        # Read the durable gate again immediately before source retirement. A fresh child
        # must never substitute its empty memory for the server's recovery result.
        with store.open_store(root, write=False) as txn:
            startup = txn.meta(STARTUP)
        if not should_stop():
            judged, hold = True, _startup_hold(startup)
            exclusions, link = (None, "") if hold else startup_tree_exclusions(root)
            if exclusions is not None:
                steps[:0] = [("task_trees", lambda root: prune_task_trees(root, exclude_root_ids=exclusions)),
                             ("services", prune_service_logs)]
            elif hold is None:
                hold = {"class": RECOVERY_HOLD, "count": None,
                        "example": f"recovery links unreadable: {link}"[:maintenance_jobs.HOLD_EXAMPLE_CHARS]}
    for name, retire in steps:
        if should_stop():
            break
        try:
            report = retire(root) or {}
            reports[name] = report
            if report.get("errors"):
                errors.append(name)
        except Exception:
            errors.append(name)
            log.warning("%s retirement failed; sources stay owed", name, exc_info=True)
    state = "pending" if should_stop() else "failed" if errors else "done"
    if owed:
        deadlines = [report["next_due"] for report in reports.values() if report.get("next_due") is not None]
        with store.open_store(root, write=True) as txn:
            current = txn.meta(maintenance_jobs.PREFIX + JOB) or {}
            if current.get("arrival") != before.get("arrival") and current.get("next_due") is not None:
                deadlines.append(current["next_due"])
            held = {"holds": maintenance_jobs._with_since([hold], current.get("holds"), utc_now_iso())} if hold else {}
            maintenance_jobs.record_in(txn, JOB, state=state, inputs=inputs, errors=errors,
                                       next_due=min(deadlines) if deadlines else None, **held)
            if judged and not hold and "holds" in current:
                # An open gate holds nothing. The field goes: Storage reads any ``holds``, an empty
                # list too, as a retirer pass's report.
                maintenance_jobs.forget_in(txn, JOB, "holds")
    if errors or any(report.get(key) for report in reports.values() for key in (
            "removed", "pruned", "unlinked", "swept", "restored", "deleted_dirs", "deleted_files", "observed")):
        append_jsonl(pathlib.Path(root) / "logs/events.jsonl",
                     {"ts": utc_now_iso(), "type": "runtime_retirement", "reports": reports, "errors": errors})
    return {"state": state, "items": len(reports)}


def prune_agent_media_uploads(
    drive_root: pathlib.Path,
    retention_days: "int | None" = None,
    *,
    now: "float | None" = None,
) -> dict:
    """Age-prune AGENT-generated media under uploads/ (CPL4-C21, owner 6A).

    Only ``uploads/screenshots/`` (browser tool) and ``uploads/views/``
    (view_image durable copies) follow GC retention — owner attachments live
    in the uploads/ ROOT and are owner-explicit-delete only, untouched here.
    Readers already skip missing files (the eviction placeholder only formats
    a re-view hint), so a pruned screenshot degrades to a stale hint, never
    an error. Fail-soft per file.

    CONTAINED (audit #15-13): the sweep deletes only REGULAR FILES that live
    inside the real family directory. ``is_file()``/``stat()`` follow symlinks,
    so a symlinked ``uploads/screenshots`` — or a single symlink inside it —
    made an age sweep of the drive unlink old files anywhere on the host. Both
    shapes are now skipped by ``lstat`` and counted, never followed.
    """
    import stat as stat_module

    from ouroboros.retention import age_cutoff, get_gc_retention_days, keep_until

    if retention_days is None:
        retention_days = get_gc_retention_days()
    cutoff = age_cutoff(retention_days, now)
    report: dict = {"removed": 0, "kept": 0, "skipped": 0, "errors": 0}
    for family in ("screenshots", "views"):
        family_dir = pathlib.Path(drive_root) / "uploads" / family
        try:
            if family_dir.is_symlink():
                report["skipped"] += 1  # the whole family points out of the drive
                continue
            entries = sorted(family_dir.iterdir())
        except OSError:
            continue
        for path in entries:
            try:
                info = path.lstat()
                if not stat_module.S_ISREG(info.st_mode):
                    report["skipped"] += 1  # symlink, directory or special file
                    continue
                if info.st_mtime >= cutoff:
                    keep_until(report, info.st_mtime, retention_days)
                    report["kept"] += 1
                    continue
                path.unlink()
                report["removed"] += 1
            except OSError:
                report["errors"] += 1
    return report


def prune_call_wire_journals(root, retention_days=None, *, now=None) -> dict:
    """Keep a call's whole wire until shared GC age AND proof that its task has no live owner.

    Terminal writes schedule the journal's mtime deadline; startup/repair discover older files.
    Compression preserves that mtime. An aged terminal task with uncertain ownership stays owed;
    running/unrecorded tasks await their terminal write. No other task artifact is removed here.
    """
    from ouroboros.retention import age_cutoff, keep_until
    from ouroboros.subagent_worktrees import _task_proven_over
    from ouroboros.task_status import SETTLED_STATUSES, task_has_live_queue_ownership

    root = pathlib.Path(root)
    days = get_gc_retention_days() if retention_days is None else retention_days
    cutoff = age_cutoff(days, now)
    report = {"removed": 0, "kept": 0, "skipped": 0, "errors": 0}
    base = root / "task_results" / "artifacts"
    try:
        if any(not stat.S_ISDIR(path.lstat().st_mode) for path in (base.parent, base)):
            return report  # never follow a linked artifact family
        with os.scandir(base) as listing:
            folders = sorted((entry for entry in listing if _plain_directory(entry)), key=lambda entry: entry.name)
    except FileNotFoundError:
        return report
    except OSError:
        report["errors"] += 1
        return report
    for folder in folders:
        path = pathlib.Path(folder.path) / WIRE_JOURNAL
        try:
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode):
                report["skipped"] += 1
                continue
            with store.open_store(root, write=False) as txn:
                settled = txn.row(folder.name, "status") in SETTLED_STATUSES
            if not settled or info.st_mtime >= cutoff or not _task_proven_over(
                    folder.name, root, lambda task: task_has_live_queue_ownership(root, task)):
                report["kept"] += 1
                if settled:
                    keep_until(report, info.st_mtime, days)
                continue
            # A write/replacement while ownership was read invalidates this candidate.
            if path.lstat() != info:
                report["kept"] += 1
                keep_until(report, path.lstat().st_mtime, days)
                continue
            from ouroboros.owner_keep import gate

            with gate(root, folder.name, timeout_sec=0) as why:
                if why:
                    report["kept"] += 1
                    continue
                path.unlink()
            report["removed"] += 1
        except FileNotFoundError:
            continue
        except OSError:
            report["errors"] += 1
    return report


RETIRED_COPIES = "retired_copies.jsonl"  # in a task's source_handles/: one line per retired saved state
SAVED_STATES_JOB = "saved_states"  # its pass row: what it released and what it holds, on Storage → Task results
WORKED_COPIES = "worked_copies"  # obligation rows: finished tasks whose folder may still hold saved states
_SEEDED = "worked_copies_seeded"  # this job row's fact: the walk that put the history's rows ran, under which body
_OWNER_WAIT = re.compile(r"owner-wait-[0-9a-f]{32}-[0-9a-f]{64}\.json")
_BUDGET_PAUSE = re.compile(r"budget-pause-(?:sleep-|owner-pause-)?[0-9a-f]{32}-[0-9a-f]{64}\.json")
_FROZEN_WORKING = re.compile(r"working-a\d+-s\d+-[0-9a-f]{64}\.json")
_ROLLING_WORKING = re.compile(r"working_checkpoint-a\d+\.json")
_SHA_TAIL = re.compile(r"-([0-9a-f]{64})\.json")
_PARTS_MANIFEST = re.compile(r"working-a\d+\.json")  # ``working_checkpoint``'s parts manifest beside the call records
_FINISH_FIELDS = ("artifact_finalized_at", "completed_at", "finished_at")  # explicit ends; a creation ``ts`` is none
_REBUILD = ("A saved state for an exact return of this work. The work is over, so nothing resumes from it. "
            "This work's model calls and tool results are recorded at call_records and people's words in the dialogue; "
            "these exact bytes and the loop's own bookkeeping (cursors, delivery flags) are not kept.")


_REBUILD_PACKED = ("A loose copy of a saved state its evidence pack still holds (packed): the exact bytes stay there. "
                   "The work is over, so nothing resumes from it.")
_lent_cache: dict = {}  # data root -> (identity of the published queue file, the lent set it gave)


class SavedStateRetired(FileNotFoundError):
    """A read that missed a saved state this task's record says was retired; the message is that record."""

    retired = True


def worked_copy(folder: str, name: str):
    """The kind of a saved state only an exact return reads, else None: the whole names its writers mint, never a
    prefix (``owner-wait-<wait>-<sha256>.json``: ``owner_wait.store_continuation_source``;
    ``budget-pause-[sleep-|owner-pause-]<id>-<sha256>.json``: ``budget_pause._exact_continuation_row`` and the restart
    and rolling conversions; ``working-a<N>-s<seq>-<sha256>.json``: ``working_checkpoint.prepare_recovery``; the rolling
    ``working_checkpoint-a<N>.json``). Every other file in these folders is a source (cuts, room views, working notes,
    reviews, plans, focus and presence sources) and never matches."""
    if folder == "context_checkpoints":
        if _OWNER_WAIT.fullmatch(name):
            return "owner_wait"
        if _BUDGET_PAUSE.fullmatch(name):
            return "budget_pause"
        if _FROZEN_WORKING.fullmatch(name):
            return "working_frozen"
    elif folder == "working" and _ROLLING_WORKING.fullmatch(name):
        return "working_rolling"
    return None


def holds_worked_copy(folder: pathlib.Path) -> bool:
    """Whether one task's folders hold a name ``worked_copy`` knows; the first such name answers. A folder this
    process cannot read may hold one (True, so it stays owed); never raises: a terminal write asks."""
    for name in ("context_checkpoints", "working"):
        try:
            with os.scandir(folder / "source_handles" / name) as listing:
                if any(worked_copy(name, entry.name) for entry in listing):
                    return True
        except (FileNotFoundError, NotADirectoryError):
            continue
        except OSError:
            return True
    return False


def worked_copies_in(folder: pathlib.Path) -> list:
    """``[(path, kind, lstat)]`` of the regular files ``worked_copy`` names in one task's two folders."""
    found = []
    for name in ("context_checkpoints", "working"):
        try:
            listing = list(os.scandir(folder / "source_handles" / name))
        except (FileNotFoundError, NotADirectoryError):
            continue  # any other error (a folder the process may not read) is the caller's
        for entry in listing:
            kind = worked_copy(name, entry.name)
            try:
                info = os.lstat(entry.path) if kind else None
            except FileNotFoundError:  # gone since the listing: only this name is skipped
                continue
            if info is not None and stat.S_ISREG(info.st_mode):
                found.append((pathlib.Path(entry.path), kind, info))
    return found


def _call_records(root: pathlib.Path, task_id: str) -> list:
    """The addresses that hold the task's model calls, as checked now: its call folder with at least one record, and
    its first evidence pack (``evidence_packs`` names generation 1 ``<task>.xz``; later generations sit beside it)."""
    found = []
    try:  # a rolling checkpoint's own parts manifest (``working-a<N>.json``) is no call record
        with os.scandir(root / "observability" / "calls" / task_id) as listing:
            if any(not _PARTS_MANIFEST.fullmatch(entry.name) for entry in listing):
                found.append(f"observability/calls/{task_id}")
    except OSError:
        pass
    try:
        if (root / "observability" / "packs" / f"{task_id}.xz").stat().st_size > 0:
            found.append(f"observability/packs/{task_id}.xz")
    except OSError:
        pass
    return found


def _lent_sources(root: pathlib.Path):
    """The tasks whose saved states a queued or running task may still load: ``working_checkpoint.load_recovery``
    reads the file of the task its handoff names (``source_task_id``, a relay's ``continued_from_task_id``), and a
    task reads its whole retry line's sources (``source_retention._source_locations``, the reader's own walk; a walk
    that cannot finish proves nothing). ``None`` without a readable, fresh published queue: then every task holds. A
    pass asks before each task: one parse per publication, and a held answer is judged again on a renewed certificate
    (``queue.alive``), so a receiver published or a stamp renewed meanwhile is seen (liveness stays ``_task_proven_over``'s)."""
    identity = []
    for name in ("queue_snapshot.json", "queue.alive"):
        try:
            info = (root / "state" / name).stat()
            identity.append((info.st_ino, info.st_mtime_ns, info.st_size))
        except OSError:
            identity.append(None)  # a missing file is an identity too: it is read again once it appears
    cached = _lent_cache.get(str(root))  # a fresh answer stands while the snapshot does; a held one, until either moves
    if cached is not None and (cached[0] == identity or cached[1] is not None and cached[0][0] == identity[0]):
        return cached[1]
    lent = _published_lent(root)
    _lent_cache[str(root)] = (identity, lent)
    return lent


def _published_lent(root: pathlib.Path):
    from ouroboros.queue_observation import load_queue_document
    from ouroboros.source_retention import _source_locations
    from ouroboros.task_status import _snapshot_is_stale

    try:
        snapshot = load_queue_document(root)
    except (OSError, ValueError):  # a malformed document or a hole in its shared table
        return None
    if snapshot.get("_snapshot_missing") or snapshot.get("_snapshot_invalid") or _snapshot_is_stale(snapshot):
        return None
    lent = set()
    for entry in [*(snapshot.get("pending") or ()), *(snapshot.get("running") or ())]:
        task = entry.get("task") if isinstance(entry, dict) else None
        task = task if isinstance(task, dict) else {}
        lent.update(str(task[key]) for key in ("timeout_retry_from", "original_task_id") if task.get(key))
        for field in ("_working_recovery", "_owner_wait_resume", "_budget_pause", "_budget_pause_resume"):
            carrier = task.get(field)
            if isinstance(carrier, dict):
                lent.update(str(carrier[key]) for key in ("source_task_id", "continued_from_task_id") if carrier.get(key))
        receiver = str((entry.get("id") if isinstance(entry, dict) else "") or task.get("id") or "")
        try:  # a busy store or a torn link mid-walk: no predecessor is proven free
            lent.update([receiver, *(owner for _root, owner in _source_locations(root, receiver))] if receiver else [])
        except Exception:
            return None
    return lent


def _work_end(document: dict):
    """When the work ended: the latest of the host's attempt-bound ``terminal_time`` and the result's explicit end
    fields, never a creation ``ts``; ``None`` when none is known (its saved states then stay)."""
    from datetime import datetime, timezone

    from ouroboros.terminal_time import terminal_time_fact

    stamps = []
    for raw in (terminal_time_fact(document).get("occurred_at"), *(document.get(key) for key in _FINISH_FIELDS)):
        try:
            parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        except ValueError:
            continue
        stamps.append((parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)).timestamp())
    return max(stamps) if stamps else None


def owe_written_back(root: Any, task_id: str) -> None:
    """A saved state written back from its pack (``evidence_packs.materialize_handle``) is owed again: best effort,
    never a wait (a missed one waits for the next walk after another body boots)."""
    try:
        with store.acquisition_budget(0), store.open_store(pathlib.Path(root), write=True) as txn:
            txn.put_row(WORKED_COPIES, str(task_id), {**(txn.get_row(WORKED_COPIES, str(task_id)) or {}), "written_back": time.time()})
            schedule_in(txn, time.time())
    except Exception:
        log.debug("A written-back saved state of %s waits for the next walk", task_id, exc_info=True)


def _ledger(root: pathlib.Path, task_id: str) -> pathlib.Path:
    return root / "task_results" / "artifacts" / task_id / "source_handles" / RETIRED_COPIES


def retired_lines(root: pathlib.Path, task_id: str) -> list:
    """The task's record lines, oldest first (a torn or unreadable line is skipped; none when there is no file)."""
    lines = []
    try:
        text = _ledger(pathlib.Path(root), task_id).read_text(encoding="utf-8")
    except (OSError, ValueError):
        return lines
    for raw in text.splitlines():
        try:
            line = json.loads(raw)
        except ValueError:
            continue
        if isinstance(line, dict) and line.get("name"):
            lines.append(line)
    return lines


def retired_message(root: Any, task_id: str, name: str) -> str:
    """What a reader that missed ``name`` of this task's saved states can say: when it was retired, what it was and
    where what it held is recorded; ``""`` when no line names it."""
    line = next((line for line in reversed(retired_lines(pathlib.Path(root), task_id)) if line.get("name") == name), None)
    if line is None:
        return ""
    kept = f"its exact bytes stay in {', '.join(line['packed'])}" if line.get("packed") else "its exact bytes are not kept"
    return (f"{name} was retired at {line.get('retired_at')} ({line.get('kind')}, {line.get('bytes')} bytes) after its "
            f"work ended; the work's model calls are in {', '.join(line.get('call_records') or []) or 'no recorded place'}; "
            f"{kept}")


def retired_note_for(path: Any) -> str:
    """``retired_message`` for a path under ``task_results/artifacts/<task>/source_handles/{context_checkpoints,working}``;
    ``""`` for any other path or name no line records."""
    parts = pathlib.Path(path).parts
    for index in range(len(parts) - 5, -1, -1):
        if parts[index:index + 2] == ("task_results", "artifacts") and parts[index + 3] == "source_handles" \
                and len(parts) == index + 6 and parts[index + 4] in ("context_checkpoints", "working"):
            return retired_message(pathlib.Path(*parts[:index]), parts[index + 2], parts[-1])
    return ""


def _append_line(ledger: pathlib.Path, line: dict) -> None:
    """One durable line: a torn tail left by a crash gets its newline first, the file and its folder entry are synced
    before the caller removes anything (an earlier pass's folder sync may have failed after it made the file)."""
    with open(ledger, "ab+") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell():
            handle.seek(-1, os.SEEK_END)
            if handle.read(1) != b"\n":
                handle.write(b"\n")
        handle.write((json.dumps(line, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())
    fsync_folder(ledger.parent)


def _retire(root: pathlib.Path, path: pathlib.Path, kind: str, info, *, task_id: str, status: str, finished: float,
            days: int, sources: list, recorded: set, packed: list) -> None:
    """Write the line that says what this was, then remove it. The line of a pass that crashed before the removal is
    not written again: one generation is its name, size and modification time (the rolling file reuses its name)."""
    from datetime import datetime, timezone

    def iso(stamp):
        return datetime.fromtimestamp(float(stamp), timezone.utc).isoformat()

    saved_at = iso(info.st_mtime)
    key = (path.name, info.st_size, info.st_mtime_ns)
    if key in recorded:  # a crashed pass's line: make it and its folder entry durable before the file goes
        with open(_ledger(root, task_id), "ab") as handle:
            os.fsync(handle.fileno())
        fsync_folder(_ledger(root, task_id).parent)
    else:
        sha = _SHA_TAIL.search(path.name)
        _append_line(_ledger(root, task_id), {
            "name": path.name, "folder": path.parent.name, "kind": kind, "bytes": info.st_size,
            "sha256": sha.group(1) if sha else None,  # the digest the name carries; never re-hashed
            "saved_at": saved_at, "saved_ns": info.st_mtime_ns, "status": status, "finished_at": iso(finished),
            "retired_at": iso(time.time()), "window_days": days, "call_records": sources,
            **({"packed": packed, "rebuild": _REBUILD_PACKED} if packed else {"rebuild": _REBUILD})})
        recorded.add(key)
    path.unlink()


def _seed(root: pathlib.Path, body) -> None:
    """Put a row for every finished task whose folder already holds saved states, then record the walk under the
    store's body fact (``meta.last_store_body``). It runs again only when another HEAD booted on the store since:
    an older body puts no rows for the works it finishes, and a switch to it writes packed handles back loose
    (``evidence_packs.restore_checkpoint_handles``)."""
    from ouroboros.task_status import SETTLED_STATUSES

    base = root / "task_results" / "artifacts"
    holding = []
    try:
        if all(stat.S_ISDIR(path.lstat().st_mode) for path in (base.parent, base)):
            with os.scandir(base) as listing:  # an unreadable folder is owed too: each pass counts its error
                holding = sorted(entry.name for entry in listing
                                 if _plain_directory(entry) and holds_worked_copy(pathlib.Path(entry.path)))
    except FileNotFoundError:
        pass
    with store.open_store(root, write=True) as txn:
        for start in range(0, len(holding), 400):
            chunk = holding[start:start + 400]
            for task_id, status, write_seq in txn.conn.execute(
                    "SELECT task_id, status, write_seq FROM tasks WHERE task_id IN (SELECT value FROM json_each(?))",
                    (json.dumps(chunk),)):
                if status in SETTLED_STATUSES:
                    txn.put_row(WORKED_COPIES, task_id, {"write_seq": write_seq, "settled_at": time.time()})
        from ouroboros.utils import utc_now_iso

        maintenance_jobs.record_in(txn, JOB, **{_SEEDED: {"at": utc_now_iso(), "body": body}})


def prune_worked_copies(root, retention_days=None, *, now=None) -> dict:
    """Retire a finished work's saved states for exact return once the owner's window has passed (#2712).

    A wait, a pause or a dead attempt saves its loop's whole state so the same work can return exactly. Every loader
    of these files refuses a terminal row, so once the work is over nothing of it reads them; what they held is in the
    work's model-call records. The work list is the ``worked_copies`` obligation rows (a terminal write puts one,
    ``_seed`` once the history's), so a pass costs the owed tasks, never a walk of every folder. A file goes when its
    task's row is settled, no child result is still being promoted into it, the owner's retention days have passed
    since the work ended (``_work_end``; the row's ``drive_retention_at`` only lets a pass skip a task still inside
    them), no queued or running task may still load it (``_lent_sources``), the task is proven over, its call records
    exist (else the record would name an address that is not there) and the owner did not keep it
    (``owner_keep.gate``). A line in ``source_handles/retired_copies.jsonl`` says what it was first. A row goes once its
    folder holds no saved state, or once its task is no longer settled (its next terminal write puts it back)."""
    from ouroboros.owner_keep import gate
    from ouroboros.retention import age_cutoff, keep_until
    from ouroboros.subagent_worktrees import _task_proven_over
    from ouroboros.task_results import load_task_result
    from ouroboros.task_status import SETTLED_STATUSES, task_has_live_queue_ownership

    root = pathlib.Path(root)
    days = get_gc_retention_days() if retention_days is None else retention_days
    cutoff = age_cutoff(days, now)
    report = {"removed": 0, "removed_bytes": 0, "kept": 0, "waiting": 0, "errors": 0, "held": {}}
    examples: dict = {}
    task_id = ""

    def hold(reason, count):
        report["kept"] += count
        report["held"][reason] = report["held"].get(reason, 0) + count
        examples.setdefault(reason, task_id)

    def wait(stamp, count):  # inside the owner's days: the rule, not a keep (``maintenance_jobs``: age waits are not holds)
        keep_until(report, stamp, days)
        report["waiting"] += count

    with store.open_store(root, write=False) as txn:
        seeded = (txn.meta(maintenance_jobs.PREFIX + JOB) or {}).get(_SEEDED)
        body = txn.meta("last_store_body")
    body = dict(body) if isinstance(body, dict) else {}  # its ``at`` moves when another body booted and this one returned
    if not isinstance(seeded, dict) or seeded.get("body") != body:
        _seed(root, body)
    with store.open_store(root, write=False) as txn:
        observed = txn.set_rows(WORKED_COPIES)  # each row as read: only an unchanged row is dropped at the end
        owed = list(observed)
        rows = {}
        for start in range(0, len(owed), 400):
            rows.update({record[0]: tuple(record[1:]) for record in txn.conn.execute(
                "SELECT task_id, status, drive_retention_at, has_pending_promotion, write_seq FROM tasks "
                "WHERE task_id IN (SELECT value FROM json_each(?))", (json.dumps(owed[start:start + 400]),))})
    base = root / "task_results" / "artifacts"
    done = []
    for task_id in owed:
        try:
            folder = base / task_id
            found = worked_copies_in(folder) if folder.is_dir() and not folder.is_symlink() else []
            status, recorded, promoting, seq = rows.get(task_id, (None, None, None, None))
            if not found or status not in SETTLED_STATUSES:
                done.append(task_id)
                continue
            if promoting == 1:
                hold("child_refs_pending", len(found))
                continue
            try:  # a projection of the body's date, maybe its creation: it can only prove the window still open
                if float(recorded) >= cutoff:
                    wait(float(recorded), len(found))
                    continue
            except (TypeError, ValueError):
                pass
            try:
                finished = _work_end(load_task_result(root, task_id, strict=True) or {})
            except Exception:
                hold("task_result_unreadable", len(found))
                continue
            if finished is None:  # no recorded end (a recovered crash, a legacy record): from when the host wrote it settled
                finished = (observed.get(task_id) or {}).get("settled_at")
            if finished is None:
                hold("finish_time_unknown", len(found))
                continue
            if finished >= cutoff:
                wait(finished, len(found))
                continue
            lent = _lent_sources(root)  # this task's own look: a publication since the last task is read again
            if lent is None or task_id in lent:
                hold("lent_to_receiver", len(found))
                continue
            if not _task_proven_over(task_id, root, lambda task: task_has_live_queue_ownership(root, task)):
                hold("not_proven_over", len(found))
                continue
            sources = _call_records(root, task_id)
            if not sources:
                hold("no_call_records", len(found))
                continue
            from ouroboros.evidence_packs import packed_handle_packs

            packed = packed_handle_packs(root, task_id)  # a loose copy of packed bytes: its line names that generation
            recorded_lines = {(line.get("name"), line.get("bytes"), line.get("saved_ns")) for line in retired_lines(root, task_id)}
            left = 0
            with gate(root, task_id, timeout_sec=0) as why:
                if not why:  # under the custody lock a promotion or reopening published meanwhile is seen
                    with store.open_store(root, write=False) as txn:
                        now_row = txn.row(task_id, "status", "has_pending_promotion", "write_seq")
                    why = ("child_refs_pending" if now_row and now_row[1] == 1  # written since it was read: the next pass's
                           else "" if now_row and now_row[0] in SETTLED_STATUSES and now_row[2] == seq else "changed")
                if why:
                    hold(why, len(found))
                    continue
                for path, kind, info in found:
                    try:
                        if os.lstat(path) != info:  # written or replaced while the facts were read
                            hold("changed", 1)
                            left += 1
                            continue
                        in_pack = [packed[path.name]] if path.parent.name == "context_checkpoints" and path.name in packed else []
                        _retire(root, path, kind, info, task_id=task_id, status=str(status), finished=finished,
                                days=days, sources=sources, recorded=recorded_lines, packed=in_pack)
                    except FileNotFoundError:
                        continue
                    except OSError:  # a Windows sharing violation, a refused unlink: the next pass tries again
                        report["errors"] += 1
                        hold("removal_failed", 1)
                        left += 1
                        continue
                    report["removed"] += 1
                    report["removed_bytes"] += getattr(info, "st_blocks", 0) * 512 or info.st_size
            if not left:
                done.append(task_id)
        except OSError:
            report["errors"] += 1
    if done:
        with store.open_store(root, write=True) as txn:
            for task_id in done:  # a terminal write that put the row again during the pass keeps it owed
                if txn.get_row(WORKED_COPIES, task_id) == observed.get(task_id):
                    txn.delete_row(WORKED_COPIES, task_id)
    maintenance_jobs.record_pass(root, SAVED_STATES_JOB, {
        "released": report["removed"], "errors": ["OSError"] * report["errors"],
        "holds": [{"class": reason, "count": count, "example": examples.get(reason, "")}
                  for reason, count in sorted(report["held"].items())]})
    return report


def prune_stale_bytecode(root, *, budget=None, sources=None) -> dict:
    """Retire bytecode mirrors whose source directory is gone (``state/pycache.noindex``, the
    ``PYTHONPYCACHEPREFIX`` pool workers write; ``state/python-userbase`` holds real installs and
    is never read here). Under a prefix CPython writes ``<prefix>/<source directory without its
    root>/<name>.<tag>.pyc`` (``importlib`` ``cache_from_source``), so a mirror directory names its
    source: ``"/" + relative`` on POSIX; on Windows the drive is stripped, so a mirror is stale only
    when that path is missing on every drive present now (``_source_roots``). Top-down: a stale
    mirror goes whole and is never descended; an unreadable source (access denied, a drive not
    answering) keeps its mirror; links are neither followed nor removed. The legacy ``state/pycache``
    an older body or host writes again goes whole first, unless it is this process's own prefix.
    At most ``budget`` unlinks a run, the rest owed through ``next_due``. A pure cache: a wrong
    removal costs a recompile."""
    prefix, legacy = pathlib.Path(root) / "state" / BYTECODE, pathlib.Path(root) / "state" / LEGACY_BYTECODE
    report = {"removed": 0, "unlinked": 0, "unknown": 0, "errors": 0}
    left = [BYTECODE_UNLINKS_PER_RUN if budget is None else int(budget)]
    own = getattr(sys, "pycache_prefix", None) or ""
    own = os.path.normcase(os.path.abspath(own)) if own else ""
    if _plain_tree(legacy) and os.path.normcase(os.path.abspath(legacy)) != own:
        removed = _remove_mirror(legacy, left, report)
        if removed is None:
            report["next_due"] = time.time()
            return report
        report["removed"] += removed
    if not _plain_tree(prefix):
        return report
    roots = _source_roots() if sources is None else [os.fspath(path) for path in sources]

    def source(parts) -> str:
        """``present``, ``gone`` or ``unknown`` over every root; any root that has it keeps it."""
        unknown = not roots
        for base in roots:
            try:
                if stat.S_ISDIR(os.stat(os.path.join(base, *parts)).st_mode):
                    return "present"
            except (FileNotFoundError, NotADirectoryError):
                continue
            except OSError:
                unknown = True
        return "unknown" if unknown else "gone"

    def walk(directory, parts) -> bool:
        """False once the budget is spent."""
        try:
            with os.scandir(directory) as listing:
                entries = sorted((entry for entry in listing if _plain_directory(entry)), key=lambda entry: entry.name)
        except OSError:
            report["unknown"] += 1
            return True
        for entry in entries:
            here = (*parts, entry.name)
            verdict = source(here)
            if verdict == "present":
                if not walk(entry.path, here):
                    return False
            elif verdict == "unknown":
                report["unknown"] += 1
            else:
                removed = _remove_mirror(entry.path, left, report)
                if removed is None:
                    return False
                report["removed"] += removed
        return True

    if not walk(prefix, ()):
        report["next_due"] = time.time()  # owed: the next run of the job continues the walk
    return report


def _plain_tree(path) -> bool:
    """A real directory: a link is not our prefix."""
    try:
        return stat.S_ISDIR(os.lstat(path).st_mode)
    except OSError:
        return False


def _link(entry) -> bool:
    """A symlink or a Windows junction (a reparse point): never followed, never removed."""
    return entry.is_symlink() or (os.name == "nt" and bool(entry.stat(follow_symlinks=False).st_file_attributes & 0x400))


def _plain_directory(entry) -> bool:
    try:
        return entry.is_dir(follow_symlinks=False) and not _link(entry)
    except OSError:
        return False


def _remove_mirror(path, left: list, report: dict):
    """Unlink PATH bottom-up within the budget LEFT: True once it is gone, False when something stays
    (a link inside, a refusal, a recompile landing meanwhile), None when the budget ran out."""
    whole = True
    try:
        with os.scandir(path) as listing:
            entries = list(listing)
    except FileNotFoundError:
        return True
    except OSError:
        report["errors"] += 1
        return False
    for entry in entries:
        if left[0] <= 0:
            return None
        if _plain_directory(entry):
            outcome = _remove_mirror(entry.path, left, report)
            if outcome is None:
                return None
            whole = whole and outcome
            continue
        try:
            if _link(entry):
                whole = False
                continue
            os.unlink(entry.path)
            left[0] -= 1
            report["unlinked"] += 1
        except FileNotFoundError:
            continue
        except OSError:
            whole = False
            report["errors"] += 1
    if not whole:
        return False
    if left[0] <= 0:
        return None
    try:
        os.rmdir(path)
    except FileNotFoundError:
        return True
    except OSError as exc:
        if exc.errno not in (errno.ENOTEMPTY, errno.EEXIST):  # not a recompile that landed meanwhile
            report["errors"] += 1
        return False
    left[0] -= 1
    report["unlinked"] += 1
    return True


def _source_roots() -> list:
    """Where a mirror's source can be: ``/`` on POSIX; on Windows every drive present now
    (``os.listdrives`` on 3.12+, else the ``GetLogicalDrives`` bitmask, never a probe of each
    letter), each root asked once a run so a drive without media or a lost network share is no
    place a source can be."""
    if sys.platform != "win32":
        return [os.sep]
    if hasattr(os, "listdrives"):
        listed = list(os.listdrives())
    else:
        import ctypes

        mask = int(ctypes.windll.kernel32.GetLogicalDrives())  # type: ignore[attr-defined]
        listed = [f"{chr(ord('A') + bit)}:\\" for bit in range(26) if mask >> bit & 1]
    return [drive for drive in listed if _answers(drive)]


def _answers(path) -> bool:
    """A listed drive is a place a source can be unless its root is absent (no media, a lost share);
    one that refuses access is there, and the per-mirror access rule then keeps what it might hold."""
    try:
        os.stat(path)
    except PermissionError:
        return True
    except OSError:
        return False
    return True


