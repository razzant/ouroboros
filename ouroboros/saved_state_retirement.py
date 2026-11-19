"""Saved states for exact return, retired from the maintenance child (#2712).

A wait, a pause or a dead attempt saves its loop's whole state so the same work can return exactly: owner waits,
exact pauses, frozen working points and the rolling working file (``worked_copy`` names them). Once nothing can load
such a file again, it goes after the owner's retention days, each with a line in its task's
``source_handles/retired_copies.jsonl`` saying what it was and where what it held is recorded (``retired_message``):
a finished work's files their days after it ended, a running work's earlier ones their days after a write superseded
them or recorded facts proved them dead (never a file's time), its current points never (#3797).
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
from ouroboros.retention import get_gc_retention_days, time
from ouroboros.utils import fsync_folder

log = logging.getLogger(__name__)


RETIRED_COPIES = "retired_copies.jsonl"  # in a task's source_handles/: one line per retired saved state
SAVED_STATES_JOB = "saved_states"  # its pass row: what it released and what it holds, on Storage → Task results
WORKED_COPIES = "worked_copies"  # obligation rows: tasks whose folder may still hold saved states (finished or not)
_SEEDED = "worked_copies_seeded"  # this job row's fact: the walk that put the history's rows ran, under which body
_OWNER_WAIT = re.compile(r"owner-wait-([0-9a-f]{32})-[0-9a-f]{64}\.json")  # the wait it saved
_BUDGET_PAUSE = re.compile(r"budget-pause-((?:sleep-|owner-pause-)?[0-9a-f]{32})-[0-9a-f]{64}\.json")  # its pause_id
_FROZEN_WORKING = re.compile(r"working-a(\d+)-s\d+-[0-9a-f]{64}\.json")  # the attempt it froze
_ROLLING_WORKING = re.compile(r"working_checkpoint-a(\d+)\.json")  # its attempt
# The head ``owner_wait.continuation_state`` gives an owner-wait file (``json.dumps`` keeps its first two keys first); a
# task id is at most 128 characters (``validate_task_id``), so the first bytes always hold it.
_WAIT_HEAD = re.compile(rb'\{\s*"task_id"\s*:\s*"([A-Za-z0-9][A-Za-z0-9_.-]{0,127})"\s*,\s*"task_attempt"\s*:\s*(\d+)\s*[,}]')
_HEAD_BYTES = 512
_SHA_TAIL = re.compile(r"-([0-9a-f]{64})\.json")
_PARTS_MANIFEST = re.compile(r"working-a\d+\.json")  # ``working_checkpoint``'s parts manifest beside the call records
_FINISH_FIELDS = ("artifact_finalized_at", "completed_at", "finished_at")  # explicit ends; a creation ``ts`` is none
_REBUILD = ("A saved state for an exact return of this work. The work is over, so nothing resumes from it. "
            "This work's model calls and tool results are recorded at call_records and people's words in the dialogue; "
            "these exact bytes and the loop's own bookkeeping (cursors, delivery flags) are not kept.")


_REBUILD_PACKED = ("A loose copy of a saved state its evidence pack still holds (packed): the exact bytes stay there. "
                   "The work is over, so nothing resumes from it.")
_REBUILD_RUNNING = ("An earlier saved state of a work that went on from a newer one; nothing resumes from it. "
                    "This work's model calls and tool results are recorded at call_records and people's words in the "
                    "dialogue; these exact bytes and the loop's own bookkeeping (cursors, delivery flags) are not kept.")
_REBUILD_RUNNING_PACKED = ("A loose copy of a saved state its evidence pack still holds (packed): the exact bytes stay "
                           "there. The work went on from a newer one, so nothing resumes from it.")
_lent_cache: dict = {}  # data root -> (identity of the published queue file, what it said: ``_published_lent``)


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


def _plain_directory(entry) -> bool:
    """A task's own folder: a real directory, never followed through a link or a Windows junction (a reparse point).
    The test ``maintenance_retirement`` applies to its folders, kept here so this module imports nothing of it at load."""
    try:
        return entry.is_dir(follow_symlinks=False) and not (
            entry.is_symlink() or os.name == "nt" and bool(entry.stat(follow_symlinks=False).st_file_attributes & 0x400))
    except OSError:
        return False


def _same_file(path: pathlib.Path, info) -> bool:
    """Whether ``path`` is still the file ``worked_copies_in`` listed: a rewrite or a replacement moves its identity,
    size or modification time. Its access time is no part of it: a read moves it on Linux (relatime, strictatime),
    the pass's own read of an owner-wait head included."""
    now = os.lstat(path)
    return all(getattr(now, key) == getattr(info, key) for key in ("st_mode", "st_ino", "st_dev", "st_size", "st_mtime_ns"))


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
    view = _published(root)
    return None if view is None else view["lent"]


def _published(root: pathlib.Path):
    """What the published queue says (``_published_lent``), parsed once per publication."""
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
    view = _published_lent(root)
    _lent_cache[str(root)] = (identity, view)
    return view


def _queue_view(root: pathlib.Path):
    """``_published`` for an unfinished work's files, its freshness read again at every ask: a publisher that stopped
    leaves a stale queue even while its last parse stands. ``None`` (missing, stale, unreadable): every file holds."""
    from ouroboros.queue_observation import load_queue_document
    from ouroboros.task_status import _snapshot_is_stale

    try:
        if _snapshot_is_stale(load_queue_document(root)):
            return None
    except (OSError, ValueError):
        return None
    return _published(root)


def _published_lent(root: pathlib.Path):
    """``None`` without a readable, fresh published queue, else ``lent`` (every task a queued or running task may load,
    its own id among them), ``others`` (the tasks another queued or running task's carrier or retry line names) and
    ``carried`` (each task's own carriers that name a source by path: ``_working_recovery`` and the copy a deadline's
    unsubmitted launch keeps, ``queue_timeouts``)."""
    from ouroboros.queue_observation import load_queue_document
    from ouroboros.source_retention import _source_locations
    from ouroboros.task_status import _snapshot_is_stale

    try:
        snapshot = load_queue_document(root)
    except (OSError, ValueError):  # a malformed document or a hole in its shared table
        return None
    if snapshot.get("_snapshot_missing") or snapshot.get("_snapshot_invalid") or _snapshot_is_stale(snapshot):
        return None
    lent, others, carried = set(), set(), {}
    for entry in [*(snapshot.get("pending") or ()), *(snapshot.get("running") or ())]:
        task = entry.get("task") if isinstance(entry, dict) else None
        task = task if isinstance(task, dict) else {}
        named = {str(task[key]) for key in ("timeout_retry_from", "original_task_id") if task.get(key)}
        for field in ("_working_recovery", "_owner_wait_resume", "_budget_pause", "_budget_pause_resume"):
            carrier = task.get(field)
            if isinstance(carrier, dict):
                named.update(str(carrier[key]) for key in ("source_task_id", "continued_from_task_id") if carrier.get(key))
        receiver = str((entry.get("id") if isinstance(entry, dict) else "") or task.get("id") or "")
        try:  # a busy store or a torn link mid-walk: no predecessor is proven free
            line = [owner for _root, owner in _source_locations(root, receiver)] if receiver else []
        except Exception:
            return None
        lent.update(named, line, [receiver] if receiver else [])
        others.update(name for name in (*named, *line) if name != receiver)
        launch = task.get("_terminalization_retry")
        launch = launch.get("unsubmitted_launch") if isinstance(launch, dict) else None
        sources = (task.get("_working_recovery"), launch.get("source") if isinstance(launch, dict) else None)
        carried.setdefault(receiver, []).extend(source for source in sources if isinstance(source, dict))
    return {"lent": lent, "others": others, "carried": carried}


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
    never a wait (a missed one waits for the next walk after another body boots). Only a settled task's row is dated
    as settled: a running work's first settled write dates its own."""
    from ouroboros.maintenance_retirement import schedule_in
    from ouroboros.task_status import SETTLED_STATUSES

    try:
        with store.acquisition_budget(0), store.open_store(pathlib.Path(root), write=True) as txn:
            row = txn.get_row(WORKED_COPIES, str(task_id)) or {}
            settled = {"settled_at": row.get("settled_at") or time.time()} if txn.row(str(task_id), "status") in SETTLED_STATUSES else {}
            txn.put_row(WORKED_COPIES, str(task_id), {**row, "written_back": time.time(), **settled})
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
    after = ("after its work ended" if "finished_at" in line  # a running work's line has no end, only what it went on to
             else f"after the work moved on to {line.get('continues_from') or 'a newer saved state'}")
    return (f"{name} was retired at {line.get('retired_at')} ({line.get('kind')}, {line.get('bytes')} bytes) {after}; "
            f"the work's model calls are in {', '.join(line.get('call_records') or []) or 'no recorded place'}; {kept}")


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


def _iso(stamp) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(float(stamp), timezone.utc).isoformat()


def _retire(root: pathlib.Path, path: pathlib.Path, kind: str, info, *, task_id: str, status: str, when: dict,
            days: int, sources: list, recorded: set, packed: list) -> None:
    """Write the line that says what this was, then remove it. ``when``: a finished work's ``finished_at``, or a running
    work's ``superseded_at`` or ``proven_at`` with the rule (``basis``) and what it went on to (``continues_from``). The
    line of a pass that crashed before the removal is not written again: one generation is its name, size and
    modification time (the rolling file reuses its name)."""
    key = (path.name, info.st_size, info.st_mtime_ns)
    if key in recorded:  # a crashed pass's line: make it and its folder entry durable before the file goes
        with open(_ledger(root, task_id), "ab") as handle:
            os.fsync(handle.fileno())
        fsync_folder(_ledger(root, task_id).parent)
    else:
        sha, ended = _SHA_TAIL.search(path.name), "finished_at" in when
        rebuild = (_REBUILD_PACKED if packed else _REBUILD) if ended else (_REBUILD_RUNNING_PACKED if packed else _REBUILD_RUNNING)
        _append_line(_ledger(root, task_id), {
            "name": path.name, "folder": path.parent.name, "kind": kind, "bytes": info.st_size,
            "sha256": sha.group(1) if sha else None,  # the digest the name carries; never re-hashed
            "saved_at": _iso(info.st_mtime), "saved_ns": info.st_mtime_ns, "status": status, **when,
            "retired_at": _iso(time.time()), "window_days": days, "call_records": sources,
            **({"packed": packed} if packed else {}), "rebuild": rebuild})
        recorded.add(key)
    path.unlink()


def _seed(root: pathlib.Path, body) -> None:
    """Put a row for every task whose folder already holds saved states (a running work's row keeps its facts), then
    record the walk under the store's body fact (``meta.last_store_body``). It runs again only when another HEAD booted
    on the store since: an older body puts no rows for the works it finishes or moves on, and a switch to it writes
    packed handles back loose (``evidence_packs.restore_checkpoint_handles``)."""
    from ouroboros.maintenance_retirement import JOB
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
                row = txn.get_row(WORKED_COPIES, task_id)
                if status in SETTLED_STATUSES:  # a new walk keeps the first date
                    txn.put_row(WORKED_COPIES, task_id, {"write_seq": write_seq, "settled_at": (row or {}).get("settled_at") or time.time()})
                elif row is None:
                    txn.put_row(WORKED_COPIES, task_id, {})
        from ouroboros.utils import utc_now_iso

        maintenance_jobs.record_in(txn, JOB, **{_SEEDED: {"at": utc_now_iso(), "body": body}})


def prune_worked_copies(root, retention_days=None, *, now=None) -> dict:
    """Retire saved states for exact return once nothing can load them and the owner's window has passed (#2712, #3797).

    A wait, a pause or a dead attempt saves its loop's whole state so the same work can return exactly. Every loader
    of these files refuses a terminal row, so once the work is over nothing of it reads them; what they held is in the
    work's model-call records. The work list is the ``worked_copies`` obligation rows (a task write puts one,
    ``_seed`` once the history's), so a pass costs the owed tasks, never a walk of every folder. A finished work's file
    goes when its task's row is settled, no child result is still being promoted into it, the owner's retention days
    have passed since the work ended (``_work_end``, else the obligation row's ``settled_at``: when the host first wrote
    it settled; the row's ``drive_retention_at`` only lets a pass skip a task still inside them), no queued or running
    task may still load it (``_lent_sources``), the task is proven over, its call records exist (else the record would
    name an address that is not there) and the owner did not keep it (``owner_keep.gate``). A work that goes on keeps
    its current points; an earlier saved state of it goes the owner's days after recorded facts proved it dead
    (``_prune_running``). A line in ``source_handles/retired_copies.jsonl`` says what it was first. A row goes, if
    unchanged since the pass read it, once its folder holds no saved state."""
    from ouroboros.maintenance_retirement import JOB
    from ouroboros.owner_keep import gate
    from ouroboros.retention import age_cutoff, keep_until
    from ouroboros.subagent_worktrees import _task_proven_over
    from ouroboros.task_results import load_task_result
    from ouroboros.task_status import SETTLED_STATUSES, task_has_live_queue_ownership

    root = pathlib.Path(root)
    days = get_gc_retention_days() if retention_days is None else retention_days
    cutoff, stamp = age_cutoff(days, now), time.time() if now is None else float(now)
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
    done, rewrite = [], {}
    for task_id in owed:
        try:
            folder = base / task_id
            found = worked_copies_in(folder) if folder.is_dir() and not folder.is_symlink() else []  # listed before facts
            status, recorded, promoting, seq = rows.get(task_id, (None, None, None, None))
            if not found or status is None:
                done.append(task_id)
                continue
            if status not in SETTLED_STATUSES:
                left, facts = _prune_running(root, task_id, found, observed.get(task_id) or {}, promoting, days=days,
                                             stamp=stamp, cutoff=cutoff, hold=hold, wait=wait, report=report)
                if not left:
                    done.append(task_id)
                elif facts != observed.get(task_id):
                    rewrite[task_id] = facts
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

            packed = packed_handle_packs(root, task_id, complete=True)  # a loose copy of packed bytes: its line names that generation
            if packed is None:  # a listing an unreadable sidecar left partial: "not kept" would be unproven
                hold("packs_unknown", len(found))
                continue
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
                        if not _same_file(path, info):  # written or replaced while the facts were read
                            hold("changed", 1)
                            left += 1
                            continue
                        in_pack = [packed[path.name]] if path.parent.name == "context_checkpoints" and path.name in packed else []
                        _retire(root, path, kind, info, task_id=task_id, status=str(status), when={"finished_at": _iso(finished)},
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
    if done or rewrite:
        with store.open_store(root, write=True) as txn:
            for task_id in done:  # a task write that put the row again during the pass keeps it owed
                if txn.get_row(WORKED_COPIES, task_id) == observed.get(task_id):
                    txn.delete_row(WORKED_COPIES, task_id)
            for task_id, facts in rewrite.items():  # a running work's dates; a write meanwhile keeps its own row
                if txn.get_row(WORKED_COPIES, task_id) == observed.get(task_id):
                    txn.put_row(WORKED_COPIES, task_id, facts)
    maintenance_jobs.record_pass(root, SAVED_STATES_JOB, {
        "released": report["removed"], "errors": ["OSError"] * report["errors"],
        "holds": [{"class": reason, "count": count, "example": examples.get(reason, "")}
                  for reason, count in sorted(report["held"].items())]})
    return report


def _prune_running(root, task_id, found, row, promoting, *, days, stamp, cutoff, hold, wait, report) -> tuple:
    """One unfinished work's saved states (#3797): ``(names still there, its row's facts)``. Its current points are no
    candidates (``_held_by_row``), nor is a file no recorded fact proves dead (``_dead``; never a file's time). A proven
    one waits the owner's days from when a write superseded it (``superseded``) or a pass first proved it (``proven``,
    kept in the row), then goes unless the published queue is not fresh or may still relay it (``lent_to_receiver``),
    a child result is being promoted, the call records are missing or the owner keeps the work. Under the custody lock
    the row and the queue are read again and every rule applied again before the line and the unlink; no ``write_seq``
    compare, as a running work's row is written all the time."""
    from ouroboros.evidence_packs import packed_handle_packs
    from ouroboros.owner_keep import gate
    from ouroboros.task_results import load_task_result
    from ouroboros.task_status import SETTLED_STATUSES

    folder = root / "task_results" / "artifacts" / task_id
    names = {path.name for path, _kind, _info in found}
    superseded = {name: at for name, at in _facts(row, "superseded").items() if name in names}
    proven = {name: at for name, at in _facts(row, "proven").items() if name in names}

    def facts():  # a file's dates go with it
        kept = {key: {name: at for name, at in value.items() if name in names}
                for key, value in (("superseded", superseded), ("proven", proven))}
        return names, {key: value for key, value in kept.items() if value}

    try:
        document = load_task_result(root, task_id, strict=True) or {}
    except Exception:
        hold("task_result_unreadable", len(found))
        return facts()
    candidates = []
    for path, kind, info in found:
        if _held_by_row(path.name, document):  # a current point owes no window
            superseded.pop(path.name, None)
            proven.pop(path.name, None)
        elif _dead(path, kind, document, superseded, task_id, folder):
            anchor = superseded.get(path.name) or proven.setdefault(path.name, stamp)
            if anchor > cutoff:
                wait(anchor, 1)
            else:
                candidates.append((path, kind, info))
    if not candidates:
        return facts()
    if promoting == 1:
        hold("child_refs_pending", len(candidates))
        return facts()
    view, attempt = _queue_view(root), _number(document.get("task_attempt"))
    if view is None or task_id in view["others"]:  # a queue that proves nothing, or another task may load any of them
        hold("lent_to_receiver", len(candidates))
        return facts()
    free = [item for item in candidates if not _carried(item[0].name, view, task_id, attempt, folder)]
    if len(free) < len(candidates):
        hold("lent_to_receiver", len(candidates) - len(free))
    sources = _call_records(root, task_id) if free else []
    if free and not sources:
        hold("no_call_records", len(free))
    if not sources:
        return facts()
    packed = packed_handle_packs(root, task_id, complete=True)
    if packed is None:  # a pack listing an unreadable sidecar left partial: "not kept" would be unproven
        hold("packs_unknown", len(free))
        return facts()
    recorded = {(line.get("name"), line.get("bytes"), line.get("saved_ns")) for line in retired_lines(root, task_id)}
    with gate(root, task_id, timeout_sec=0) as why:
        if not why:  # the row, its promotion and the queue as of now: every rule again
            try:
                document = load_task_result(root, task_id, strict=True) or {}
                with store.open_store(root, write=False) as txn:
                    promotion = txn.row(task_id, "has_pending_promotion")
            except store.StoreBusy:
                why = "store_busy"
            except Exception:
                why = "task_result_unreadable"
            else:
                why = "child_refs_pending" if promotion == 1 else "changed" if document.get("status") in SETTLED_STATUSES else ""
        if why:
            hold(why, len(free))
            return facts()
        view, attempt, continues = _queue_view(root), _number(document.get("task_attempt")), _continues_from(document, folder)
        for path, kind, info in free:
            if view is None or task_id in view["others"] or _carried(path.name, view, task_id, attempt, folder):
                hold("lent_to_receiver", 1)
                continue
            basis = "" if _held_by_row(path.name, document) else _dead(path, kind, document, superseded, task_id, folder)
            try:
                if not basis or not _same_file(path, info):  # a pointer came to it, its proof lapsed, or it was rewritten
                    hold("changed", 1)
                    continue
                anchor = ("superseded_at", superseded[path.name]) if path.name in superseded else ("proven_at", proven[path.name])
                in_pack = [packed[path.name]] if path.parent.name == "context_checkpoints" and path.name in packed else []
                _retire(root, path, kind, info, task_id=task_id, status=str(document.get("status")), days=days,
                        when={anchor[0]: _iso(anchor[1]), "basis": basis, "continues_from": continues},
                        sources=sources, recorded=recorded, packed=in_pack)
            except FileNotFoundError:
                names.discard(path.name)
                continue
            except OSError:  # a Windows sharing violation, a refused unlink: the next pass tries again
                report["errors"] += 1
                hold("removal_failed", 1)
                continue
            names.discard(path.name)
            report["removed"] += 1
            report["removed_bytes"] += getattr(info, "st_blocks", 0) * 512 or info.st_size
    return facts()


def _rolling(folder: pathlib.Path, attempt: int) -> pathlib.Path:
    """The rolling working file an attempt writes (``working_checkpoint.checkpoint_path``)."""
    return folder / "source_handles" / "working" / f"working_checkpoint-a{attempt}.json"


def _held_by_row(name: str, document: dict) -> bool:
    """A current point, never a candidate: the file the row's wait or pause names, in any state (the row's loaders and
    ``working_checkpoint.live_park`` read only these), and the wait a live pause retains (``retained_owner_wait_id``)."""
    from ouroboros.budget_pause import LIVE_PAUSE_STATES

    wait, pause = _facts(document, "owner_wait"), _facts(document, "budget_pause")
    retained = pause.get("retained_owner_wait_id") if pause.get("state") in LIVE_PAUSE_STATES else None
    return (name in (_ref_name(wait.get("source_ref")), _ref_name(pause.get("source_ref")))
            or bool(retained) and name.startswith(f"owner-wait-{retained}-"))


def _wait_attempt(path: pathlib.Path, task_id: str):
    """The attempt an owner-wait file was saved by, read from the head its one writer gives it (``_WAIT_HEAD``), never
    from its time; ``None`` when that head is not there or names another task."""
    try:
        with open(path, "rb") as handle:
            head = _WAIT_HEAD.match(handle.read(_HEAD_BYTES))
    except OSError:
        return None
    return int(head.group(2)) if head and head.group(1).decode("ascii") == task_id else None


def _dead(path: pathlib.Path, kind: str, document: dict, superseded: dict, task_id: str, folder: pathlib.Path) -> str:
    """The recorded fact that proves an unfinished work can never load this saved state again (the record's ``basis``),
    ``""`` while none does: a write superseded the pointer that named it; an owner wait saved by an attempt older than
    the row's, or by the row's attempt while that attempt is parked on another wait or held by a settled live pause
    (only the work's own loop parks or pauses it, so no park of it is in flight); a pause the row no longer names,
    unless a restart conversion to that name may still land (``budget-pause-sleep-<wait>`` while that sleep waits, an
    ``owner-pause`` copy while an owner-pause park waits); a rolling file of an earlier attempt; a frozen point of an
    earlier attempt once the row's attempt has its own rolling file (``working_checkpoint._carry_recovery`` relays only
    while that file is absent)."""
    from ouroboros.budget_pause import LIVE_PAUSE_STATES, STATE_PAUSED, STATE_RESUME_GRANTED

    if path.name in superseded:
        return "superseded"
    attempt = _number(document.get("task_attempt"))
    wait, pause = _facts(document, "owner_wait"), _facts(document, "budget_pause")
    if kind == "owner_wait":
        saved = _wait_attempt(path, task_id)
        if saved is None or saved > attempt:
            return ""
        if saved < attempt:
            return "earlier_attempt"
        parked = wait.get("state") == "waiting" or (wait.get("state") == "retained" and pause.get("state") in LIVE_PAUSE_STATES
                                                    and pause.get("retained_owner_wait_id") == wait.get("wait_id"))
        if parked and _number(wait.get("task_attempt")) == saved and wait.get("wait_id") != _OWNER_WAIT.fullmatch(path.name).group(1):
            return "parked_on_another_wait"
        live = pause.get("state") in (STATE_PAUSED, STATE_RESUME_GRANTED) and _number(pause.get("task_attempt")) >= saved
        return "paused" if live else ""
    if kind == "budget_pause":
        pause_id = _BUDGET_PAUSE.fullmatch(path.name).group(1)
        converting = wait.get("state") == "waiting" and (
            pause_id == f"sleep-{wait.get('wait_id')}" and wait.get("reason") == "sleep"
            or pause_id.startswith("owner-pause-") and wait.get("reason") == "owner_pause")
        return "" if pause.get("pause_id") == pause_id or converting else "pause_replaced"
    if kind == "working_rolling":
        return "earlier_attempt" if int(_ROLLING_WORKING.fullmatch(path.name).group(1)) < attempt else ""
    frozen = int(_FROZEN_WORKING.fullmatch(path.name).group(1))
    return "earlier_attempt_continued" if frozen < attempt and _rolling(folder, attempt).exists() else ""


def _carried(name: str, view: dict, task_id: str, attempt: int, folder: pathlib.Path) -> bool:
    """Whether this task's own published carrier names the file, by path or (an owner wait) by its wait, whatever the
    carrier's admissibility, while its relay can still fire: until the row's attempt is past the carrier's
    ``from_attempt`` and has its own rolling file (``working_checkpoint._carry_recovery`` relays only without one)."""
    wait = _OWNER_WAIT.fullmatch(name)
    for carrier in view["carried"].get(task_id, ()):
        if name == _ref_name(carrier.get("source_ref")) or wait and carrier.get("wait_id") == wait.group(1):
            source = carrier.get("from_attempt")
            if not (type(source) is int and attempt > source and _rolling(folder, attempt).exists()):
                return True
    return False


def _continues_from(document: dict, folder: pathlib.Path):
    """What the work goes on from, for its record: the wait it parks on, its live pause, else its attempt's own rolling
    file; ``None`` when none of them is there."""
    from ouroboros.budget_pause import LIVE_PAUSE_STATES

    wait, pause = _facts(document, "owner_wait"), _facts(document, "budget_pause")
    if wait.get("state") == "waiting" and _ref_name(wait.get("source_ref")):
        return _ref_name(wait.get("source_ref"))
    if pause.get("state") in LIVE_PAUSE_STATES and _ref_name(pause.get("source_ref")):
        return _ref_name(pause.get("source_ref"))
    rolling = _rolling(folder, _number(document.get("task_attempt")))
    return rolling.name if rolling.exists() else None


def task_written(txn, task_id, columns, current=None, document=None):
    """The saved states' half of a task write, in the writer's transaction (``maintenance_retirement.task_written``).
    A finished task whose folder holds a saved state is owed to ``prune_worked_copies``: its row (``settled_at``: when
    the host first wrote it settled, kept after; a running work's row has none, so its first settled write is a new
    obligation) and the pass at its window's end. An unfinished write is read for what it moved past (``_moved_on``)."""
    from ouroboros.maintenance_retirement import schedule_in
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
    from ouroboros.maintenance_retirement import schedule_in

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
