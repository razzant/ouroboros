"""Background maintenance: named jobs in separate lanes, never on a server thread.

Work whose cost grows with accumulated history (evidence discovery and packing, and the other
named jobs in ``JOBS``) runs in a child process at background priority: under one GIL a server
thread doing it turns every request's I/O into waiting. Each job keeps its facts in its
``maintenance_jobs`` row of ``state/current.sqlite``, so "once" is a fact of the installation,
a restart of either process resumes from the recorded facts, and a second run over unchanged
inputs decodes no history body, copies nothing and commits nothing.

Server half: ``child`` (``MaintenanceChildren``, one dispatch slot per lane). The supervisor calls
``tick`` after reconciliation and when a look is owed (the first tick or a producer's ``request``).
The dispatcher reserves the slot; the lane thread reads the due facts and spawns only for due work.
The trash lane retries unchanged holds on the reconcile cadence (``request(retry=True)``), looks
only at its own facts, and requests an ordinary look for retirement after its child runs.
The spawn is the existing custodied one (``startup_historical_audit.
spawn_maintenance_job``: session custody, its own process group, a parent lifeline; ``stop`` and
``live_children`` serve the Panic and exit census). Queue-owned work (orphan recovery, drive
settlement under the queue lock and the supervisor's liveness probe) stays in the server: a
fresh child owns no queue. After an ordinary child run not ended by Stop or Panic the server tells the owner
once what each one-time walk did, when it is over (``delegate_result_taken``/``historical_retirement.deliver_receipt``).

Lanes: ``custody_trash`` runs in a child of its own (``--lane trash``), every other job in the ordinary one, so a
moved drive's deletion never waits behind packing; a child started without ``--lane`` runs every job.

Child half: ``main`` (``python -m ouroboros.maintenance [--lane trash]``) lowers its priority and runs its lane's ``JOBS``
in order, each asking its own facts again, and prints one bounded JSON report.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import logging
import os
import pathlib
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from ouroboros import (delegate_capture_job, delegate_pending, delegate_result_taken, historical_retirement,
                       maintenance_jobs, maintenance_retirement, plan_collection_job, record_compression,
                       room_appearances, room_messages, store, subagent_metrics)
from ouroboros.startup_historical_audit import _REPORT_BYTES, finish_maintenance_job, spawn_maintenance_job

log = logging.getLogger(__name__)
PURPOSE = "maintenance"  # the custody purpose and the module the child runs


def _plain(value: Any) -> Any:
    """The JSON form a job row stores, so a fresh reading compares equal to a recorded one."""
    return json.loads(json.dumps(value, default=str))


def _store_facts(root: pathlib.Path) -> Dict[str, Any]:
    """The dispatcher's store reading, one read transaction: the revision and each pending
    promotion member's row signature (indexed flag column; no body)."""
    from ouroboros.obligations import members_in

    with store.open_store(root, write=False) as txn:
        return {"revision": txn.revision(),
                "promotions": {task_id: _plain(txn.row(task_id, "write_seq", "updated_at"))
                               for task_id in members_in(txn, "promotions")}}


def _now() -> str:
    from ouroboros.utils import utc_now_iso

    return utc_now_iso()


def _revision(root: pathlib.Path) -> int:
    """The store revision now (one meta read): what a pass beginning here has seen."""
    with store.open_store(root, write=False) as txn:
        return txn.revision()


# ---- jobs ------------------------------------------------------------------------------------

def _promotion_due(root: pathlib.Path, row: Dict[str, Any], facts: Dict[str, Any], child: bool) -> bool:
    """A pending promotion member with no remembered refusal at its current row and body: a new
    member, a transient failure, or a row that changed since (``source_retention.RetentionMemo``)."""
    from ouroboros.source_retention import memo_epoch

    observed, epoch = row.get("observed") or {}, memo_epoch()
    return any((observed.get(task_id) or {}).get("signature") != signature
               or (observed.get(task_id) or {}).get("epoch") != epoch
               for task_id, signature in facts["promotions"].items())


def _promotion_run(root: pathlib.Path, should_stop: Callable[[], bool]) -> Dict[str, Any]:
    """The pending child-ref promotion retry (copies a settled child's call history and sources
    into the canonical store under the task's custody file lock); its durable observations are
    this job's row, its state is written only when it changes."""
    from ouroboros.observability import retry_pending_child_ref_promotions

    report = retry_pending_child_ref_promotions(root, stop=should_stop)
    row = maintenance_jobs.job(root, "child_ref_promotion") or {}
    state = "pending" if _promotion_due(root, row, _store_facts(root), True) else "done"
    if row.get("state") != state:
        maintenance_jobs.record(root, "child_ref_promotion", state=state)
    return {"state": state, "items": len(report["retried"])}


def _any_entry(directory: pathlib.Path) -> bool:
    """Whether DIRECTORY holds anything (one scandir that stops at the first entry)."""
    try:
        with os.scandir(directory) as entries:
            return next(entries, None) is not None
    except OSError:
        return False


def _legacy_due(root: pathlib.Path, row: Dict[str, Any], facts: Dict[str, Any], child: bool) -> bool:
    """Scheduled (``pending``) by the boot detector or the explicit command (``store_import``);
    an interrupted run (``running``) resumes at its cursor; a failed one waits to be scheduled."""
    return row.get("state") in {"pending", "running"}


def _legacy_run(root: pathlib.Path, should_stop: Callable[[], bool]) -> Dict[str, Any]:
    """``store_import.run_legacy_reconcile``: the reconcile of the files a store-unaware body wrote,
    root and every descendant drive, with its own durable cursor in this job's row."""
    from ouroboros import store_import

    reconcile = getattr(store_import, "run_legacy_reconcile", None)
    if reconcile is None:  # a body without the job: the row waits for one that has it
        return {"state": "pending", "items": 0}
    row = reconcile(root, should_stop=should_stop) or {}
    return {"state": str(row.get("state") or "unknown"), "items": int(row.get("done") or 0)}


def _trash_due(root: pathlib.Path, row: Dict[str, Any], facts: Dict[str, Any], child: bool) -> bool:
    """Drives a settlement moved aside (``state/custody_trash``) wait for their deletion. An entry
    the last run could not remove is no reason to spawn the child until its identity changes, and
    the child retries it whenever it runs for another job: a released handle (Windows, a scanner)
    changes neither its inode nor its time. The trash lane's retry look (``child``: the reconcile
    cadence) tries unchanged entries again. An unreadable identity does not prove absence. A recorded
    ``retire_owed`` or a name missing from a whole listing may still owe its retirement wake.
    An unreadable listing cannot settle that debt and does not itself spawn a retry."""
    from ouroboros.task_custody import custody_trash_scan

    observed = {**(row.get("stuck") or {}), **(row.get("held") or {})}
    try:
        entries, complete = custody_trash_scan(root)
    except OSError:
        return False  # a denied or partial listing is neither work nor provable absence
    if row.get("retire_owed"):
        return complete
    listed = dict(entries)
    gone = complete and any(name not in listed for name in observed)
    if child:
        return bool(entries) or gone
    return gone or any(name not in observed or (identity is not None and observed[name] != identity)
                       for name, identity in listed.items())


def _trash_run(root: pathlib.Path, should_stop: Callable[[], bool]) -> Dict[str, Any]:
    """Delete the moved drives (each was custodied under the queue lock before its move; nothing
    serves or references it), so a child drive's tree is removed off the serving process. The row is
    written only when something changed: a retry that leaves the same entries stuck commits nothing.
    Stop leaves the old status and aggregate holds; it does not invalidate the earlier enumeration
    or confirmed removals. Drop only those confirmed-gone names when scheduling their retirement.
    ``retire_owed`` is set before the first unlink attempt (except an unchanged stuck entry's retry)
    and cleared in the transaction that schedules retirement. An unreadable enumeration cannot settle
    that debt. The same nonblocking lock excludes the compatible manual CLI; run owns contention."""
    from ouroboros.task_custody import remove_custody_trash

    with store.name_lock(root, "custody_trash.lock", timeout_sec=0):
        row = maintenance_jobs.job(root, "custody_trash") or {}
        observed = {**(row.get("stuck") or {}), **(row.get("held") or {})}
        owed, stopped = [bool(row.get("retire_owed"))], []

        def before_unlink(entry: pathlib.Path) -> None:
            if owed[0]:
                return
            with contextlib.suppress(OSError):
                info = entry.lstat()
                if observed.get(entry.name) == [info.st_ino, info.st_mtime_ns]:
                    return  # an unchanged stuck entry's retry commits nothing before its attempt
            maintenance_jobs.record(root, "custody_trash", retire_owed=True)
            owed[0] = True

        def stop() -> bool:
            if should_stop():
                stopped.append(True)
            return bool(stopped)

        listing: Dict[str, List[str]] = {}  # populated only after the initial enumeration succeeds
        report = remove_custody_trash(root, stop=stop, before_unlink=before_unlink,
                                      notify_retirement=False, listing=listing)
        complete = not stopped and not report.get("incomplete")
        removed, held = report["removed"], report.get("held") or {}
        state = "failed" if report["stuck"] else "done"
        gone = set(listing.get("gone", ()))
        enumerated = "initial" in listing
        if enumerated:
            gone |= set(observed) - set(listing["initial"])
        book = bool(gone) or (enumerated and owed[0])
        facts: Dict[str, Any] = {"removed": removed}
        if complete:
            facts.update(state=state, stuck=report["stuck"], held=held, holds=report["holds"])
        elif gone:
            for key in ("stuck", "held"):
                facts[key] = {name: ident for name, ident in (row.get(key) or {}).items() if name not in gone}
        changed = complete and (row.get("state"), row.get("stuck") or {}, row.get("held") or {}, row.get("holds") or []) != (
            state, report["stuck"], held, report["holds"])  # a changed hold set is evidence; an unchanged one is silent
        if book or changed:
            with store.open_store(root, write=True) as txn:
                maintenance_jobs.record_in(txn, "custody_trash", **facts)
                if book:
                    maintenance_retirement.schedule_in(txn, time.time())
                    if enumerated and owed[0]:
                        maintenance_jobs.forget_in(txn, "custody_trash", "retire_owed")
        return {"state": state if complete else "pending", "items": removed}


def _release() -> str:
    from ouroboros import __version__

    return str(__version__)


def _temp_due(root: pathlib.Path, row: Dict[str, Any], facts: Dict[str, Any], child: bool) -> bool:
    """Once per installed release: a hard kill's orphaned atomic temps are reaped by the next
    release's first child run. Never by itself a reason to spawn the child."""
    return child and row.get("release") != _release()


def _temp_run(root: pathlib.Path, should_stop: Callable[[], bool]) -> Dict[str, Any]:
    """The atomic writers' directories only (``temp_sweep.ATOMIC_TEMP_DIRS``), never the whole root or
    ``observability/``; the age guard keeps any in-flight write's temp. The startup sweep of the
    ``tmp_scripts`` fallback (no script can be live then) stays the server's."""
    from ouroboros.temp_sweep import sweep_stale_temp_files

    removed = sweep_stale_temp_files(root, scripts=False)
    maintenance_jobs.record(root, "temp_sweep", state="done", release=_release(), removed=removed)
    return {"state": "done", "items": removed}


def _task_rows(root: pathlib.Path) -> int:
    with store.open_store(root, write=False) as txn:
        return int(txn.conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0])


# The planner statistics every store keeps, on every SQLite build: row counts (``sqlite_stat1``), as the app's own
# SQLite keeps. A build with ENABLE_STAT4 (Homebrew's Python, for one) also samples each index into ``sqlite_stat4``,
# and every new connection loads those samples with the schema, a cost that grows with the history
# (tests/test_cost_contract.py B7). The job's row names the policy, so a store analyzed under another moves once.
STATISTICS = "stat1"


def _stale(row: Dict[str, Any], tasks: int) -> bool:
    """Planner statistics once per store file (this row lives in it), and again once the task
    table grew tenfold since: statistics describe the data's magnitude, so a fresh install's
    would mislead the planner as it grows."""
    return row.get("state") != "done" or tasks >= 10 * max(1, int(row.get("tasks") or 0))


def _sampled(root: pathlib.Path) -> bool:
    """Whether ``sqlite_stat4`` holds a sample: its schema row, then at most one of its rows."""
    with store.open_store(root, write=False) as txn:
        table = txn.conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='sqlite_stat4'").fetchone()
        return bool(table and txn.conn.execute("SELECT 1 FROM sqlite_stat4 LIMIT 1").fetchone())


def _analyze_due(root: pathlib.Path, row: Dict[str, Any], facts: Dict[str, Any], child: bool) -> bool:
    """Stale statistics, a row that does not name today's ``STATISTICS``, or samples beside one that does:
    an older body's job (a plain ``ANALYZE``, then a merge into this row) rebuilds them on a STAT4 build and
    keeps the row's word. Never by itself a reason to spawn the child: a store analyzed before moves when
    the child next runs."""
    if not child:
        return False
    return row.get("statistics") != STATISTICS or _sampled(root) or _stale(row, _task_rows(root))


def _analyze_run(root: pathlib.Path, should_stop: Callable[[], bool]) -> Dict[str, Any]:
    """``ANALYZE`` when the statistics are stale; then the samples go and this connection's planner
    reloads what stays (``ANALYZE sqlite_master``), in the transaction that writes the row: a store
    analyzed before keeps its ``sqlite_stat1`` and its tenfold base, an interrupted run changes nothing."""
    row = maintenance_jobs.job(root, "sqlite_analyze") or {}
    with store.open_store(root, write=True) as txn:
        tasks = int(txn.conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0])
        stale = _stale(row, tasks)
        if stale:
            txn.conn.execute("ANALYZE")
        if txn.conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='sqlite_stat4'").fetchone():
            txn.conn.execute("DELETE FROM sqlite_stat4")
            txn.conn.execute("ANALYZE sqlite_master")
        maintenance_jobs.record_in(txn, "sqlite_analyze", state="done", statistics=STATISTICS,
                                   **({"tasks": tasks} if stale else {}))
    return {"state": "done", "items": tasks if stale else 0}


def _packing_inputs(root: pathlib.Path) -> Dict[str, Any]:
    """The facts the packing rule reads, cheaply: the store's terminal-write counter (a task closing
    or a closed task written again moves it; a live task's writes do not; one meta row, never a read
    over the rows), whether archived usage segments wait, whether the evidence index awaits its rebuild
    (``evidence_packs.index_owed``: stats, nothing of the packs opened) and whether any loose evidence exists."""
    from ouroboros import evidence_packer, evidence_packs

    return {"terminal": _plain(evidence_packer.terminal_revision(root)),
            "archive_pending": evidence_packer.archive_pending(root), "index_owed": evidence_packs.index_owed(root),
            "loose": any(_any_entry(root / rel) for rel in ("observability/calls", "task_results/artifacts"))}


def _packing_due(root: pathlib.Path, row: Dict[str, Any], facts: Dict[str, Any], child: bool) -> bool:
    """Evidence may wait when archived usage segments wait or the evidence index awaits its rebuild
    (only the pass rebuilds it, so even a quiet install is due once its file is gone or a reader marked
    it unusable), or when loose evidence exists and the
    terminal rows moved since the last pass, or the last pass unlinked (one more look is owed: it
    records the moved directories' stamps), or it left candidates unfinished (``pending``: deferred
    by a held custody lock, busy files, faults, a manifest a candidate wrote after the census) and the
    store has been written since that pass began
    (the contender that held the lock or the file writes its receipt, during the pass or after it;
    nothing but a write re-arms it, so a persistent fault costs one pass per store change, never a
    tight loop); in the child the unfinished work runs whenever the child runs for another reason."""
    inputs = _packing_inputs(root)
    unfinished = bool(row.get("pending")) and (child or facts["revision"] != row.get("revision"))
    return inputs["archive_pending"] or inputs["index_owed"] or (inputs["loose"] and (
        row.get("terminal") != inputs["terminal"] or not row.get("quiet") or unfinished))


def _packing_run(root: pathlib.Path, should_stop: Callable[[], bool]) -> Dict[str, Any]:
    """Discovery (archives and an owed index rebuild first, then the first changed task stamp) and, when
    it finds work, one packing pass, both over the census the job row addresses: the rows written terminal since the
    position the last pass recorded (``census``, the counter it read before its pass, under the store's
    body fact ``body``) and the candidates it left (``pending``); every loose folder only without a
    readable position (``evidence_packer.census_since``: the first run, an older body's pass, another
    HEAD booted on the store since, a counter that went back). The inputs are read before the pass and
    recorded after it, so a task closing during the
    pass makes the next look differ. A pass that unlinked records ``quiet: False``; one that left
    candidates deferred, busy, faulted or written a manifest after the census records their ids as
    ``pending`` with the store revision
    it has seen: the one its own record leaves when nothing else wrote during the pass, else the
    one the pass began at, since a write during the pass (the contender's receipt, committed after
    the deferral and before this record) is one the pass has not seen, and recording the later
    revision would swallow it (``_packing_due`` re-arms on the next write; a quiet install then
    retries once and rests until the store moves again). A raised discovery or pass records
    nothing: the job stays due for the next look."""
    from ouroboros import evidence_packer, evidence_packs as packs

    row = maintenance_jobs.job(root, "evidence_packing") or {}
    inputs = _packing_inputs(root)
    began = _revision(root)  # the census reads the store after this point
    body = evidence_packer.store_body(root)
    since, owed = evidence_packer.census_since(row, inputs["terminal"], body), evidence_packer.owed_ids(row)
    report: Dict[str, Any] = {"status": "completed", **dict.fromkeys(evidence_packer.COUNTERS, 0), "unfinished": []}
    if evidence_packer.packing_pending(root, should_stop, since=since, owed=owed):
        report = evidence_packer.run_pass(root, since=since, owed=owed)
        packs.record_storage_pack(root, phase=report["status"], **{key: report[key] for key in evidence_packer.COUNTERS})
    if should_stop():
        return {"state": "pending", "items": report["examined"]}
    with store.open_store(root, write=True) as txn:  # one write: the row and the revision it has seen
        seen = txn.revision() + 1 if txn.revision() == began else began
        maintenance_jobs.record_in(txn, "evidence_packing", state="done", terminal=inputs["terminal"],
                                   census=inputs["terminal"], body=body, quiet=not report["unlinked"],
                                   pending=list(report["unfinished"]), revision=seen,
                                   packed=report["packed"], examined=report["examined"])
    return {"state": "done", "items": report["examined"]}


def _dialogue_due(root: pathlib.Path, row: Dict[str, Any], facts: Dict[str, Any], child: bool) -> bool:
    """Archive segments the cutover planned and the import has not read (``import_ledger`` pending);
    a failed segment waits on its own input, which the runner compares, so it is no reason to spawn."""
    from ouroboros import dialogue_import, dialogue_store

    if row.get("state") == "failed":
        return child
    return dialogue_store.present(root) and dialogue_import.import_owed(root)


def _dialogue_run(root: pathlib.Path, should_stop: Callable[[], bool]) -> Dict[str, Any]:
    from ouroboros.dialogue_import import run_dialogue_import

    row = run_dialogue_import(root, should_stop=should_stop) or {}
    return {"state": str(row.get("state") or "unknown"), "items": int(row.get("done") or 0)}


def _tail_due(root: pathlib.Path, row: Dict[str, Any], facts: Dict[str, Any], child: bool) -> bool:
    """The chronicle index's stored tail facts are absent, unfinished or of another dialogue store or frontier
    (``memory_inventory.tail_owed``: an update, a rebuilt index, a replaced store). A failed fill waits for the
    child to run for another reason; an unreadable fact defers to the next look, never blocking the other jobs."""
    from ouroboros.memory_inventory import tail_owed

    if row.get("state") == "failed":
        return child
    try:
        return tail_owed(root)
    except Exception:
        log.debug("Memory tail facts unreadable; the next look asks again", exc_info=True)
        return False


def _resumable(name: str, module: str, function: str = "run") -> Callable[..., Dict[str, Any]]:
    """``module.function(root, should_stop)``, resumable at its own stored facts; the job's row keeps its state,
    ``failed`` on a fault other than contention (its ``due`` then waits for the child to run for another reason)."""
    def run(root: pathlib.Path, should_stop: Callable[[], bool]) -> Dict[str, Any]:
        try:
            report = getattr(importlib.import_module(module), function)(root, should_stop)
        except Exception as exc:
            if not _contention(exc):
                maintenance_jobs.record(root, name, state="failed")
            raise
        if (maintenance_jobs.job(root, name) or {}).get("state") != report["state"]:
            maintenance_jobs.record(root, name, state=report["state"])
        return report
    return run


def _hot_facts_due(root: pathlib.Path, row: Dict[str, Any], facts: Dict[str, Any], child: bool) -> bool:
    """Rows written before the hot-fact columns (or by an older writer since) wait for a classification
    pass (``store_hot_facts.run_hot_facts``); a root without such rows has nothing to do, and neither has
    one whose only such rows are the damaged input its last pass named, unchanged (``hot_facts_due``)."""
    from ouroboros.store_hot_facts import hot_facts_due

    return hot_facts_due(root, row, child)


def _hot_facts_run(root: pathlib.Path, should_stop: Callable[[], bool]) -> Dict[str, Any]:
    from ouroboros.store_hot_facts import run_hot_facts

    row = run_hot_facts(root, should_stop=should_stop) or {}
    return {"state": str(row.get("state") or "unknown"), "items": int(row.get("done") or 0)}


# name -> (due(root, row, store facts, child), run(root, should_stop)); the child runs them in this
# order. ``child`` is False for the server's look: a job may then decline to be a reason to spawn
# (it still runs whenever the child runs for another reason).
JOBS: Dict[str, tuple] = {
    # Captures already addressed by the server must not queue behind a history import.
    "delegate_capture": (delegate_capture_job.due, delegate_capture_job.run),
    "subagent_metrics": (subagent_metrics.job_due, subagent_metrics.run_job),
    "pending_invocations": (delegate_pending.backfill_due, delegate_pending.backfill),
    # Before the archive import: the drives it reconciles can leave dialogue work, which the import
    # then finishes in the same pass (the next pass rests; tests/test_cost_contract.py B6).
    "legacy_reconcile": (_legacy_due, _legacy_run),
    "dialogue_import": (_dialogue_due, _dialogue_run),  # the owner's history: newest segments first
    "memory_tail": (_tail_due, _resumable("memory_tail", "ouroboros.memory_inventory", "fill_tail")),  # after the import
    "room_appearances": (room_appearances.due, _resumable("room_appearances", "ouroboros.room_appearances")),  # likewise
    "room_messages": (room_messages.due, _resumable("room_messages", "ouroboros.room_messages")),  # dates the history
    "hot_facts": (_hot_facts_due, _hot_facts_run),
    "ended_plan_asks": (plan_collection_job.due, plan_collection_job.run),
    "custody_trash": (_trash_due, _trash_run),  # before retirement: a removed drive's bytecode goes in the same pass
    "runtime_retirement": (maintenance_retirement.due, maintenance_retirement.run),
    "child_ref_promotion": (_promotion_due, _promotion_run),
    "temp_sweep": (_temp_due, _temp_run),
    "sqlite_analyze": (_analyze_due, _analyze_run),
    "record_compression": (record_compression.due, record_compression.run),
    "evidence_packing": (_packing_due, _packing_run),  # last: its quiet fact follows every other write
}
JOB_NAMES = tuple(JOBS)
TRASH_JOBS = ("custody_trash",)  # the trash lane's job; every other job is the ordinary lane's
LANES = ("ordinary", "trash")


def lane_jobs(lane: Optional[str] = None) -> tuple:
    """The jobs of LANE in ``JOBS`` order; every job for ``None`` (a child started without ``--lane``)."""
    if lane not in (None, *LANES):
        raise ValueError(f"unknown maintenance lane: {lane}")
    return tuple(name for name in JOBS if lane is None or (name in TRASH_JOBS) == (lane == "trash"))


def due(root: Any, *, lane: Optional[str] = None, child: bool = False) -> List[str]:
    """Read job facts, replaying owed producer wakes before dispatch; no history walk. LANE limits the look to
    its jobs; CHILD looks as the child does (only the trash lane's cadence retry asks it); the trash lane reads
    no store facts, so its look replays no ordinary producer's owed wake."""
    root, names = pathlib.Path(root), lane_jobs(lane)
    rows = maintenance_jobs.jobs(root, names)
    facts = {} if lane == "trash" else _store_facts(root)
    return [name for name in names if JOBS[name][0](root, rows.get(name) or {}, facts, child)]


def _contention(exc: BaseException) -> bool:
    """A held store or lock (current facts, packs index, money): the job stays due, nothing failed."""
    return isinstance(exc, store.StoreBusy) or getattr(exc, "reason", None) == "contention"


def run_job(root: Any, name: str, *, should_stop: Optional[Callable[[], bool]] = None) -> Optional[Dict[str, Any]]:
    """Run one named job in THIS process when its own facts say work may wait (the child's step;
    tests and the cost contract call it); ``None`` when the job is at rest: only its facts were read. A trash job
    reads no store facts, so the trash child replays no ordinary producer's owed wake."""
    root = pathlib.Path(root)
    is_due, run_one = JOBS[name]
    facts = {} if name in TRASH_JOBS else _store_facts(root)
    if not is_due(root, maintenance_jobs.job(root, name) or {}, facts, True):
        return None
    return run_one(root, should_stop or (lambda: False))


def _last_error(root: pathlib.Path, name: str, exc: Optional[BaseException]) -> None:
    """A job that threw keeps its ``state`` (several due rules read it: a transient exception must not
    park a resumable job) and gains ``last_error`` ``{class, at}``; its next normal return clears it.
    The facts stay readable beside the place the job serves (``storage_removers``)."""
    try:
        if exc is not None:
            maintenance_jobs.record(root, name, last_error={"class": type(exc).__name__, "at": _now()})
        elif "last_error" in (maintenance_jobs.job(root, name) or {}):
            with store.open_store(root, write=True) as txn:
                maintenance_jobs.forget_in(txn, name, "last_error")
    except Exception:
        log.debug("Maintenance job %s outcome not recorded", name, exc_info=True)


def run(root: Any, *, should_stop: Optional[Callable[[], bool]] = None, lane: Optional[str] = None) -> Dict[str, Any]:
    """The child's pass: every due job of LANE (every job without one) in order; one job's failure never skips the next."""
    root, stop = pathlib.Path(root), should_stop or (lambda: False)
    report: Dict[str, Any] = {"status": "completed", "jobs": {}}
    for name in lane_jobs(lane):
        if stop():
            break
        try:
            outcome = run_job(root, name, should_stop=stop)
        except Exception as exc:
            if not _contention(exc):
                log.warning("Maintenance job %s failed", name, exc_info=True)
                report["status"] = "failed"
                _last_error(root, name, exc)
            outcome = {"state": "pending" if _contention(exc) else "failed", "items": 0}
        else:
            if outcome is not None:
                _last_error(root, name, None)
        if outcome is not None:
            report["jobs"][name] = {"state": str(outcome.get("state") or "unknown"),
                                    "items": int(outcome.get("items") or 0)}
    return report


# ---- the server half -------------------------------------------------------------------------

def _record(data_root: Any, phase: str, **facts: Any) -> None:
    from ouroboros.utils import append_jsonl

    append_jsonl(pathlib.Path(data_root) / "logs" / "supervisor.jsonl",
                 {"ts": _now(), "type": "maintenance", "phase": phase, **facts})


def _report_fields(raw: bytes) -> Dict[str, Any]:
    """The child's report under a closed schema; anything else is ``{}`` (phase unknown)."""
    try:
        value = json.loads(raw) if len(raw) <= _REPORT_BYTES else None
        if not isinstance(value, dict) or value.get("status") not in {"completed", "unknown", "failed"}:
            return {}
        jobs = value.get("jobs")
        if not isinstance(jobs, dict) or set(jobs) - set(JOB_NAMES):
            return {}
        fields: Dict[str, Any] = {"status": value["status"], "jobs": {}}
        for name, outcome in jobs.items():
            state, items = (outcome or {}).get("state"), (outcome or {}).get("items")
            if state not in {*maintenance_jobs.STATES, "unknown"} or type(items) is not int or items < 0:
                return {}
            fields["jobs"][name] = {"state": state, "items": items}
        return fields
    except (ValueError, UnicodeError, TypeError, AttributeError):
        return {}


class MaintenanceChild:
    """One lane's maintenance child of this server process (module docstring); no lane runs every job."""

    def __init__(self, lane: Optional[str] = None) -> None:
        lane_jobs(lane)  # a known lane
        self._lane = lane
        self.sibling: Optional["MaintenanceChild"] = None  # the other lane, told when the trash lane ran
        self._retry = False  # the trash lane's next look is the reconcile cadence's retry window
        self._lock = threading.Lock()
        self._stopped = threading.Event()
        self._owed = True  # the first maintenance tick of a process looks at the job facts once
        self._process = None
        self._thread: Optional[threading.Thread] = None

    def request(self, *, retry: bool = False) -> None:
        """Ask the next supervisor maintenance tick to look at the job facts (a job was scheduled);
        RETRY also asks the trash lane to retry its unchanged entries in the same publication."""
        with self._lock:  # the same lane lock tick reserves the flags under
            self._retry = self._retry or (retry and self._lane == "trash")
            self._owed = True

    def retry_look(self) -> None:
        """Keep the trash lane's unspent retry window without owing a new look; other lanes ignore it."""
        if self._lane == "trash":
            with self._lock:
                self._retry = True

    def owed(self) -> bool:
        return self._owed and not self._stopped.is_set()

    def tick(self, data_root: Any, repo_dir: Any, *, owed_only: bool = False) -> bool:
        """Reserve the one slot; True means the dispatch thread started (it spawns the child only
        when a job is due). No filesystem or store access on the caller's thread. OWED_ONLY (the
        supervisor tick's event-driven look) reserves nothing for a lane no producer asked for, so a
        long child of one lane never starts a fresh fact-reading thread of the other on every tick."""
        with self._lock:
            if self._stopped.is_set() or (owed_only and not self._owed) or (
                    self._thread is not None and self._thread.is_alive()):
                return False
            self._owed, (retry, self._retry) = False, (self._retry, False)
            name = "maintenance-child" if self._lane in (None, "ordinary") else f"maintenance-{self._lane}"
            self._thread = threading.Thread(target=self._run, args=(pathlib.Path(data_root), pathlib.Path(repo_dir), retry),
                                            name=name, daemon=True)
            self._thread.start()
        return True

    def stop(self) -> None:
        self._stopped.set()
        process = self._process
        if process is not None:
            self._signal(process)

    def live_children(self) -> list:
        """Exit census after Stop: include an unfinished spawn until its owner returns."""
        process, thread = self._process, self._thread
        if process is not None and process.poll() is None:
            return [process]
        return [thread] if thread is not None and thread.is_alive() else []

    @staticmethod
    def _signal(process) -> None:
        with contextlib.suppress(OSError):
            process.kill()  # its own process group: the signal reaches only our child

    def _run(self, data_root: pathlib.Path, repo_dir: pathlib.Path, retry: bool = False) -> None:
        started = time.monotonic()
        lane = self._lane
        select = {} if lane is None else {"lane": lane, "child": retry and lane == "trash"}  # no lane: the whole catalog
        record = _record if lane in (None, "ordinary") else (
            lambda root, phase, **facts: _record(root, phase, lane=lane, **facts))
        try:
            try:
                with store.acquisition_budget(0):  # contention defers to the next look, never waits
                    names = due(data_root, **select)
            except Exception:
                if retry:  # a look the store refused keeps its retry window
                    self.retry_look()
                log.debug("Maintenance job facts unavailable; the next look retries", exc_info=True)
                return
            if not names or self._stopped.is_set():
                return
            args = {} if lane is None else {"args": ("--lane", lane)}
            process = spawn_maintenance_job(data_root, repo_dir, PURPOSE, **args)
            finish_maintenance_job(self, process, data_root, started, _report_fields, record)
            if self._stopped.is_set():  # nothing more after Stop or Panic
                return
            if lane == "trash":  # a removal owes the ordinary lane a look (retirement of the removed tree's bytecode)
                if self.sibling is not None:
                    self.sibling.request()
            else:  # the one-time walks' receipts
                delegate_result_taken.deliver_receipt(data_root)
                historical_retirement.deliver_receipt(data_root)
        except Exception as exc:
            with contextlib.suppress(Exception):
                record(data_root, phase="failed", exception_class=type(exc).__name__,
                       duration_seconds=time.monotonic() - started)
        finally:
            with self._lock:
                self._process = None


class MaintenanceChildren:
    """The server's one maintenance door over its two lanes (module docstring): each call reaches both."""

    def __init__(self) -> None:
        self.ordinary, self.trash = MaintenanceChild("ordinary"), MaintenanceChild("trash")
        self.ordinary.sibling, self.trash.sibling = self.trash, self.ordinary
        self._lanes = (self.ordinary, self.trash)

    def request(self, *, retry: bool = False) -> None:
        """Ask both lanes to look; RETRY applies to the trash lane only."""
        self.ordinary.request()
        self.trash.request(retry=retry)

    def owed(self) -> bool:
        return any(lane.owed() for lane in self._lanes)

    def tick(self, data_root: Any, repo_dir: Any, *, owed_only: bool = False) -> bool:
        return any([lane.tick(data_root, repo_dir, owed_only=owed_only) for lane in self._lanes])  # a list: every lane is asked

    def stop(self) -> None:
        for lane in self._lanes:
            lane.stop()

    def live_children(self) -> list:
        return [unit for lane in self._lanes for unit in lane.live_children()]


child = MaintenanceChildren()


# ---- the child half --------------------------------------------------------------------------

def _lower_priority() -> None:
    """Background priority: POSIX nice, the Windows below-normal class; best effort. The disk tier stays
    each job's own (``record_compression`` lowers its thread's around its run)."""
    try:
        if os.name == "nt":
            import ctypes

            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), 0x00004000)  # BELOW_NORMAL_PRIORITY_CLASS
        else:
            os.nice(10)
    except (OSError, AttributeError):
        log.debug("Maintenance priority unchanged", exc_info=True)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--custody-session", default="")
    parser.add_argument("--lane", choices=LANES, default=None)  # no lane: every job (a manual run)
    args = parser.parse_args(argv)
    os.environ["OUROBOROS_DATA_DIR"] = str(pathlib.Path(args.data_root).resolve())
    # Roots are bound before importing readers that capture config globals.
    from ouroboros.process_custody import adopt_session_id, start_parent_lifeline

    adopt_session_id(args.custody_session)
    start_parent_lifeline(label=PURPOSE)
    _lower_priority()
    try:
        report = run(pathlib.Path(args.data_root), lane=args.lane)
    except Exception as exc:
        report = {"status": "failed", "jobs": {}, "exception_class": type(exc).__name__}
    print(json.dumps({key: report[key] for key in ("status", "jobs")}, separators=(",", ":")), flush=True)
    return 1 if report["status"] == "failed" else 0


if __name__ == "__main__":
    sys.exit(main())
