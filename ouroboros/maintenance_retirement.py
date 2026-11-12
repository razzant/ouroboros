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
``source_recovery_pending``, which Storage and the mind's ``storage`` fact read; an open gate leaves no hold.
"""
from __future__ import annotations

import errno
import logging
import os
import pathlib
import stat
import sys
import uuid

from ouroboros import maintenance_jobs, store
from ouroboros.record_compression import WIRE_JOURNAL
from ouroboros.retention import get_gc_retention_days, time
from ouroboros.unindexed import BYTECODE, LEGACY_BYTECODE

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
                 ("call_wire", prune_call_wire_journals), *steps, ("bytecode", prune_stale_bytecode)]
        # Read the durable gate again immediately before source retirement. A fresh child
        # must never substitute its empty memory for the server's recovery result.
        with store.open_store(root, write=False) as txn:
            startup = txn.meta(STARTUP)
        if not should_stop():
            judged, hold = True, _startup_hold(startup)
            exclusions = None if hold else startup_tree_exclusions(root)
            if exclusions is not None:
                steps[:0] = [("task_trees", lambda root: prune_task_trees(root, exclude_root_ids=exclusions)),
                             ("services", prune_service_logs)]
            elif hold is None:
                hold = {"class": RECOVERY_HOLD, "count": None, "example": "recovery links unreadable"}
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
    if errors or any(report.get(key) for report in reports.values()
                     for key in ("removed", "swept", "restored", "deleted_dirs", "deleted_files", "observed")):
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


