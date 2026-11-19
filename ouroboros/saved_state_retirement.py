"""Saved states for exact return, retired from the maintenance child (#2712).

A wait, a pause or a dead attempt saves its loop's whole state so the same work can return exactly: owner waits,
exact pauses, frozen working points and the rolling working file (``worked_copy`` names them). Once nothing can load
such a file again, it goes after the owner's retention days, each with a line in its task's
``source_handles/retired_copies.jsonl`` saying what it was and where what it held is recorded (``retired_message``).
The work list is the ``worked_copies`` obligation rows: a task write that owes the folder puts one
(``task_written``), and a walk of the history puts the rest once per store body (``_seed``). The step runs inside
``maintenance_retirement``'s ``runtime_retirement`` job (``prune_worked_copies``).
"""
from __future__ import annotations

import json
import logging
import os
import pathlib
import re
import stat
from typing import Any

from ouroboros import maintenance_jobs, store
from ouroboros.maintenance_retirement import JOB, _plain_directory, schedule_in
from ouroboros.retention import get_gc_retention_days, time
from ouroboros.utils import fsync_folder

log = logging.getLogger(__name__)


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
            row = txn.get_row(WORKED_COPIES, str(task_id)) or {}
            txn.put_row(WORKED_COPIES, str(task_id), {**row, "written_back": time.time(), "settled_at": row.get("settled_at") or time.time()})
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
                    settled = (txn.get_row(WORKED_COPIES, task_id) or {}).get("settled_at")  # a new walk keeps the first date
                    txn.put_row(WORKED_COPIES, task_id, {"write_seq": write_seq, "settled_at": settled or time.time()})
        from ouroboros.utils import utc_now_iso

        maintenance_jobs.record_in(txn, JOB, **{_SEEDED: {"at": utc_now_iso(), "body": body}})


def prune_worked_copies(root, retention_days=None, *, now=None) -> dict:
    """Retire a finished work's saved states for exact return once the owner's window has passed (#2712).

    A wait, a pause or a dead attempt saves its loop's whole state so the same work can return exactly. Every loader
    of these files refuses a terminal row, so once the work is over nothing of it reads them; what they held is in the
    work's model-call records. The work list is the ``worked_copies`` obligation rows (a terminal write puts one,
    ``_seed`` once the history's), so a pass costs the owed tasks, never a walk of every folder. A file goes when its
    task's row is settled, no child result is still being promoted into it, the owner's retention days have passed
    since the work ended (``_work_end``, else the obligation row's ``settled_at``: when the host first wrote it settled;
    the row's ``drive_retention_at`` only lets a pass skip a task still inside them), no queued or running task may still load it (``_lent_sources``), the task is proven over, its call records
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


def task_written(txn, task_id, columns, current=None, document=None):
    """The saved states' half of a task write, in the writer's transaction (``maintenance_retirement.task_written``).
    A finished task whose folder holds a saved state is owed to ``prune_worked_copies``: its row (``settled_at``: when
    the host first wrote it settled, kept after; a running work's row has none, so its first settled write is a new
    obligation) and the pass at its window's end. An unfinished write is read for what it moved past (``_moved_on``)."""
    from ouroboros.task_status import SETTLED_STATUSES
    if txn.root is None:
        return
    if columns.get("status") not in SETTLED_STATUSES:
        _moved_on(txn, task_id, current, document)
        return
    if not holds_worked_copy(txn.root / "task_results" / "artifacts" / task_id):
        return
    settled = (txn.get_row(WORKED_COPIES, task_id) or {}).get("settled_at")
    txn.put_row(WORKED_COPIES, task_id, {"write_seq": columns.get("write_seq"), "settled_at": settled or time.time()})
    try:  # a new obligation is owed at least now; a later write of an old work only by an end still ahead
        deadline = float(columns["drive_retention_at"]) + get_gc_retention_days() * 86400
    except (KeyError, TypeError, ValueError):
        deadline = 0.0
    if not settled or deadline > time.time():
        schedule_in(txn, max(deadline, time.time()))


def _moved_on(txn, task_id, current, document) -> None:
    """An unfinished work's write that moved past a saved state, seen with the store's pristine ``current`` beside the
    new ``document``: a new wait or pause replaced the one whose file the row named (``owner_wait.wait_id`` or
    ``budget_pause.pause_id`` changed), so that file's name and the time go into the task's row (``superseded``, the
    first time kept) and the pass is owed when the owner's days end; a new attempt began, so the row is put and the pass
    owed now to date the earlier attempt's files. A state change of the same wait or pause (resumed, retained, a warm
    grant rolled back to waiting) moves past nothing, and a write that moved nothing reads and writes nothing."""
    if not isinstance(current, dict) or not isinstance(document, dict):
        return  # a direct upsert (an import) has no pristine document: nothing was replaced by a writer
    names = [_ref_name(old.get("source_ref")) for old, new, key in (
        (_facts(current, "owner_wait"), _facts(document, "owner_wait"), "wait_id"),
        (_facts(current, "budget_pause"), _facts(document, "budget_pause"), "pause_id")) if old.get(key) != new.get(key)]
    names = [name for name in names if name]
    visit = _number(document.get("task_attempt")) > _number(current.get("task_attempt"))
    if not names and not (visit and holds_worked_copy(txn.root / "task_results" / "artifacts" / task_id)):
        return
    now, row = time.time(), txn.get_row(WORKED_COPIES, task_id) or {}
    superseded = {**{name: now for name in names}, **_facts(row, "superseded")}
    txn.put_row(WORKED_COPIES, task_id, {**row, **({"superseded": superseded} if superseded else {})})
    schedule_in(txn, now if visit else now + get_gc_retention_days() * 86400)


def _facts(document: Any, key: str) -> dict:
    value = document.get(key) if isinstance(document, dict) else None
    return value if isinstance(value, dict) else {}


def _number(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _ref_name(ref: Any) -> str:
    """The file name a saved state's ``source_ref`` names (``artifacts.actor_source_reference``), ``""`` for none."""
    path = ref.get("path") if isinstance(ref, dict) else None
    return pathlib.PurePosixPath(str(path)).name if path else ""
