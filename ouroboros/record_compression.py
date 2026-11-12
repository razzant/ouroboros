"""Transparent compression of finished records: the maintenance child's ``record_compression`` job.

Finished records stay whole while retained; where the filesystem compresses transparently their bytes
can occupy less with no byte a reader sees changed. Task evidence remains PERSISTENCE ``[record]``;
the call wire journal alone follows shared GC retention (``maintenance_retirement.prune_call_wire_journals``),
also requiring a task proven over. Compression preserves its age. Three planes (``PLANES``):

- ``evidence``: the published evidence of a task proven over (``subagent_worktrees._task_proven_over``: a
  terminal record, a negative live probe, no lineage process; this child holds no queue, so its probe is
  the queue snapshot's, ``task_has_live_queue_ownership``), ``task_results/artifacts/<task>/delegated_runs/
  */evidence*/**``, ``.../source_handles/{test-evidence,tool_results,delegated_activity}/**`` (write-once by
  digest: a reader's ref keeps its path, size and sha256) and a live call's ``conversation-wire.jsonl``,
  its row read again right before each replace; a terminal task not proven over yet waits and is asked
  again on later passes;
- ``seats``: reviewer seat homes ``review_checking_prune.finished_seat_homes`` offers (copy gone, task
  proven over the same way): their ``evidence*`` and ``.test-evidence`` folders, under the seat lock held
  through each replace;
- ``logs``: the closed segments of the rotated logs, ``archive/<log>_<UTC>[_<n>].jsonl`` (the rotator's
  names alone: never a live ``logs/*.jsonl``, a subfolder of ``archive/`` or another file there). A reader
  finds the generation that was live when it began by inode (``utils.jsonl_chain_handles``, the worker
  start cursor), macOS compression gives a file a new inode, and such a pointer lives in the memory of
  one server generation's processes. So a segment is offered only when a walk under an earlier server
  generation already listed it closed: no process of the serving generation ever held it live. A walk
  ends by recording the closed set it listed on the job row (``witness``: the ``startup_recovery``
  generation it ended under and, per chain, the newest name and how many names are no newer), and a
  later generation's walk offers a chain up to that name while as many names are no newer now; a server
  start lists nothing. A segment closed after that record waits for one more start. What the plane
  keeps for another reason is a typed hold on the job row (``holds``, the retirer passes' shape, read
  beside Archives), counting the segments not compressed yet: ``closed_set_unrecorded`` (no witness of
  an earlier generation on the row: every chain), ``prior_workers_unconfirmed`` (the start could not
  confirm that the previous generation's workers ended: every chain) and ``closed_set_changed`` (another
  number of names no newer than the witness, as after a clock step back: that chain). They wait for the
  next start too.

Never ``area``/``area.noindex`` or a ``.git`` path; every file passes the one floor
(``artifact_storage.compression_outcome``). The job reaps only its own interrupted siblings
(``artifact_storage.is_compression_sibling``) in the folders it walks (under the seat lock in a home;
``temp_sweep`` never enters ``state/review_checkouts`` or ``archive/``); any other atomic-signature name
is neither a candidate nor reaped. A folder it cannot list is counted (``unreadable``), never taken for empty.
``in_scope`` names the evidence and seat places to read-only lineage staging, which streams uncompressed
records where compression is supported; that staging copies task drives and artifact folders, never ``archive/``.

Pacing: at most ``FILES_PER_RUN`` files past the stat floor a run, a stop check between files, the cursor
durable in the job row; a run reads and rewrites at the utility disk tier (its own thread, its ``ditto``).
The first pass walks every task folder and seat home (the backlog); later passes look at the rows written
terminal since the counter the last one covered, the seat homes changed since it looked and what still
waits for its task (a task written again, or a home changed, is walked from its start). ``COVERAGE`` names
the evidence scope a position was earned under: a row from another scope, done or stopped partway, walks
every task folder again from its first, since no terminal write will name an old task; what it already
compressed costs one stat. The log plane walks once a server generation, after the other two: the row's
``generation`` is the ``startup_recovery`` generation its walk ended under, ``witness`` what it listed
then. Due while a pass is unfinished, new terminal writes arrived, the server generation is another, the
coverage differs or another platform's ``unsupported`` is on the row, never by elapsed time or a rotation;
Linux, Docker and Colab record ``unsupported`` once. Totals are kept per plane and are the job's own
record: the Storage count stays the ground truth.
"""
from __future__ import annotations

import bisect
import os
import pathlib
import re
import stat
import sys
import time
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from ouroboros import artifact_storage, maintenance_jobs, store
from ouroboros.temp_sweep import is_atomic_temp

JOB = "record_compression"
COVERAGE = 2  # the evidence plane's scope; a row without it was covered by the scope before retained sources
# Task evidence (in the Task results place), reviewer seat homes (in App state), closed log segments (in Archives).
PLANES = ("evidence", "seats", "logs")
_WAITING = PLANES[:2]  # the planes whose records wait for their task by name; a log segment waits for no task
FILES_PER_RUN = 2_000  # files past the stat floor one run reads (and mostly rewrites): the rest waits a run
_SEATS = pathlib.Path("state") / "review_checkouts" / "seats"
_SEGMENT = re.compile(r"(.+)_\d{8}T\d{6}(?:_\d+)?\.jsonl")  # the rotator's name; its chain is the part before the stamp


def _counter(root: pathlib.Path) -> int:
    with store.open_store(root, write=False) as txn:
        return int(txn.meta(store.TERMINAL_WRITES) or 0)


def _startup(root: pathlib.Path) -> Any:
    """The serving generation's ``startup_recovery`` record; None before any start recorded one."""
    from ouroboros.maintenance_retirement import STARTUP

    with store.open_store(root, write=False) as txn:
        return txn.meta(STARTUP)


def _generation(startup: Any) -> Optional[str]:
    return startup.get("generation") if isinstance(startup, dict) else None


def due(root: Any, row: Dict[str, Any], facts: Dict[str, Any], child: bool) -> bool:
    """A pass unfinished (the backlog, or a run its budget or a stop ended), a position earned under
    another ``COVERAGE``, another platform's ``unsupported`` still on the row, rows written terminal
    since the counter the last pass covered, or a server generation other than the one the log plane
    last finished under."""
    if not artifact_storage.compression_supported():
        return row.get("unsupported") != sys.platform
    root = pathlib.Path(root)
    return (row.get("state") != "done" or row.get("coverage") != COVERAGE or "unsupported" in row
            or row.get("counter") != _counter(root) or row.get("generation") != _generation(_startup(root)))


def _closed(root: pathlib.Path) -> Dict[str, List[os.DirEntry]]:
    """Each rotated log's closed segments in ``archive/`` by chain, in the readers' order (by name:
    ``utils.jsonl_archive_segments``): one listing of the folder's own files under the rotator's names. An
    absent folder holds none; one that cannot be listed raises."""
    chains: Dict[str, List[os.DirEntry]] = {}
    try:
        with os.scandir(root / "archive") as listing:
            for entry in listing:
                named = _SEGMENT.fullmatch(entry.name)
                if named and entry.is_file(follow_symlinks=False):
                    chains.setdefault(named[1], []).append(entry)
    except FileNotFoundError:
        return {}
    return {chain: sorted(entries, key=lambda entry: entry.name) for chain, entries in chains.items()}


def _mark(value: Any) -> Tuple[str, int]:
    """A chain's witness ``[newest name, how many names are no newer]``; a chain the recording walk found no
    segment of has none, and no name is no newer than it."""
    if isinstance(value, (list, tuple)) and len(value) == 2 and isinstance(value[0], str) and type(value[1]) is int:
        return value[0], value[1]
    return "", 0


def _plain(entry: os.DirEntry) -> bool:
    """Not compressed yet, by one stat; a segment that cannot be asked counts as plain."""
    try:
        return not artifact_storage.is_compressed(entry.stat(follow_symlinks=False))
    except OSError:
        return True


def _settled(root: pathlib.Path, task_ids: List[str]) -> Dict[str, bool]:
    """Each task's row terminal now, in one short read."""
    from ouroboros.task_status import SETTLED_STATUSES

    with store.open_store(root, write=False) as txn:
        return {task_id: txn.row(task_id, "status") in SETTLED_STATUSES for task_id in task_ids}


WIRE_JOURNAL = "conversation-wire.jsonl"  # a live call's wire evidence (``live_conversation.WireJournal``)
_SOURCE_RECORDS = ("test-evidence", "tool_results", "delegated_activity")  # of ``source_handles``; never working material


def _evidence_part(parts: Tuple[str, ...], directory: bool) -> bool:
    """``delegated_runs/<run>/evidence*/**``, ``source_handles/<one of _SOURCE_RECORDS>/**`` and the call wire
    journal of a task folder."""
    if parts == (WIRE_JOURNAL,):
        return not directory
    depth, named = {"delegated_runs": (3, lambda: parts[2].startswith("evidence")),
                    "source_handles": (2, lambda: parts[1] in _SOURCE_RECORDS)}.get(parts[0], (0, None))
    return bool(depth) and (len(parts) < depth or named()) and (directory or len(parts) > depth)


def _has_evidence(base: pathlib.Path) -> bool:
    """A task folder holding evidence (one stat a place): only such a task is worth its proof. One
    that cannot be looked into stays a candidate: the walk then counts it ``unreadable`` and goes on."""
    for path, kind in ((base / "delegated_runs", stat.S_ISDIR),
                       *((base / "source_handles" / name, stat.S_ISDIR) for name in _SOURCE_RECORDS),
                       (base / WIRE_JOURNAL, stat.S_ISREG)):
        try:
            if kind(os.stat(path).st_mode):
                return True
        except (FileNotFoundError, NotADirectoryError):
            continue
        except OSError:
            return True
    return False


def _seat_part(parts: Tuple[str, ...], directory: bool) -> bool:
    """A home's ``evidence``, ``evidence-<capture>`` and ``.test-evidence`` folders (never its copy)."""
    return (parts[0].startswith("evidence") or parts[0] == ".test-evidence") and (directory or len(parts) > 1)


def in_scope(root: Any, path: Any) -> bool:
    """Whether the evidence or seat plane may one day ask the floor about the file at PATH: a task folder's
    evidence or a seat home's evidence folders under ROOT (the data root), never through ``.git``. By place alone,
    whatever its size, name, task state or today's floor, so a small live record that grows is one. Read-only lineage
    staging streams one not yet compressed: an APFS clone would leave it ``shared`` for good (``artifact_storage``).
    A log segment is no such place: that staging copies only what ``tool_access.lineage_read_roots`` names. A path
    that cannot be resolved answers True: not knowing is no licence to share blocks, and a stream costs only time."""
    try:
        path, root = pathlib.Path(path).resolve(), pathlib.Path(root).resolve()
    except (OSError, RuntimeError):
        return True
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        return False
    if ".git" in parts:
        return False
    if parts[:2] == ("task_results", "artifacts") and len(parts) > 3:
        return _evidence_part(parts[3:], False)
    return parts[:3] == _SEATS.parts and len(parts) > 4 and _seat_part(parts[4:], False)


def _linked(entry: os.DirEntry) -> bool:
    return entry.is_symlink() or (os.name == "nt" and bool(entry.stat(follow_symlinks=False).st_file_attributes & 0x400))


def _files(base: pathlib.Path, wanted: Callable[[Tuple[str, ...], bool], bool], after: Tuple[str, ...],
           counts: Dict[str, int]) -> Iterator[Tuple[Tuple[str, ...], os.DirEntry]]:
    """Regular files under BASE that WANTED keeps, in the order of their parts (siblings by name, a folder
    in its place among them), strictly after AFTER; never through a link, never into ``.git``. This job's
    own interrupted sibling is reaped, any other atomic-signature name passed by, and a folder that cannot
    be listed is counted (COUNTS ``reaped``, ``unreadable``), never taken for empty."""
    def walk(directory: str, parts: Tuple[str, ...]) -> Iterator[Tuple[Tuple[str, ...], os.DirEntry]]:
        try:
            with os.scandir(directory) as listing:
                entries = sorted(listing, key=lambda entry: entry.name)
        except (FileNotFoundError, NotADirectoryError):
            return
        except OSError:
            counts["unreadable"] += 1
            return
        for entry in entries:
            here = (*parts, entry.name)
            try:
                if entry.name == ".git" or _linked(entry):
                    continue
                if entry.is_dir(follow_symlinks=False):
                    if wanted(here, True) and (here >= after or after[:len(here)] == here):
                        yield from walk(entry.path, here)
                elif entry.is_file(follow_symlinks=False) and wanted(here, False):
                    if artifact_storage.is_compression_sibling(entry.name):  # this job's interrupted sibling
                        os.unlink(entry.path)
                        counts["reaped"] += 1
                    elif not is_atomic_temp(entry.name) and here > after:
                        yield here, entry
            except OSError:
                continue
    yield from walk(os.fspath(base), ())


class _Run:
    """One run of the job: the planes in order, the file budget, the tallies and where it stopped."""

    def __init__(self, root: pathlib.Path, row: Dict[str, Any], should_stop: Callable[[], bool]) -> None:
        self.root, self.row, self.should_stop = root, row, should_stop
        self.left = FILES_PER_RUN
        self.resume: Dict[str, Any] = dict(row.get("cursor") or {})
        self.cursor: Optional[Dict[str, Any]] = None
        self.census = False  # whether this run read the evidence census (its position may move)
        self.live, self.proven = _queue_live(root), {}  # one proof per task a run, both planes
        waiting = row.get("waiting") if isinstance(row.get("waiting"), dict) else {}
        self.waiting = {plane: set(waiting.get(plane) or ()) for plane in _WAITING}
        self.planes = {plane: {"compressed": 0, "saved_bytes": 0} for plane in PLANES}
        self.counts = {"reaped": 0, "unreadable": 0}
        self.skipped: Dict[str, int] = {}
        # The log plane's, once this run judged its chains: what it holds, and the closed set it listed.
        self.holds: Optional[List[Dict[str, Any]]] = None
        self.witness: Optional[Dict[str, list]] = None

    def proven_over(self, task_id: str) -> bool:
        from ouroboros.subagent_worktrees import _task_proven_over

        if task_id not in self.proven:
            self.proven[task_id] = _task_proven_over(task_id, self.root, self.live)
        return self.proven[task_id]

    def files(self, plane: str, key: str, base: pathlib.Path, wanted, owned=None) -> bool:
        """Compress KEY's files under BASE; False once the run must stop (its budget or a stop)."""
        resume, tally = self.resume, self.planes[plane]
        after = tuple(resume.get("path") or ()) if (resume.get("plane"), resume.get("key")) == (plane, key) else ()
        for parts, entry in _files(base, wanted, after, self.counts):
            if self.should_stop() or self.left <= 0:
                return False
            try:
                before = entry.stat(follow_symlinks=False)
            except OSError:
                continue  # gone since the listing
            path = pathlib.Path(entry.path)
            outcome = artifact_storage.compression_outcome(path, owned=owned, background=True)
            if outcome not in artifact_storage.STAT_SKIPS:
                self.left -= 1
            if outcome == "compressed":
                tally["compressed"] += 1
                blocks = getattr(before, "st_blocks", None)
                try:
                    saved = None if blocks is None else max(0, (blocks - os.lstat(entry.path).st_blocks) * 512)
                except OSError:
                    saved = 0
                tally["saved_bytes"] = None if saved is None or tally["saved_bytes"] is None else tally["saved_bytes"] + saved
            elif outcome not in artifact_storage.STAT_SKIPS:
                self.skipped[outcome] = self.skipped.get(outcome, 0) + 1
            self.cursor = {"plane": plane, "key": key, "path": list(parts)}
        return not self.should_stop()

    def evidence(self) -> Optional[List[str]]:
        """The evidence plane; the task ids still owed when the run stopped, else None."""
        artifacts, resume, waiting = self.root / "task_results" / "artifacts", self.resume, self.waiting["evidence"]
        if resume.get("plane") == "seats":  # this pass finished the plane before it stopped
            self.cursor = dict(resume)
            return None
        self.census = True
        if self.row.get("counter") is None:  # the backlog: every task folder, by name from the cursor's on
            try:  # an unreadable listing raises: the pass stays owed, never marked covered
                with os.scandir(artifacts) as listing:
                    names = sorted(entry.name for entry in listing if entry.is_dir(follow_symlinks=False))
            except FileNotFoundError:
                names = []
            if resume.get("plane") == "evidence":
                names = names[bisect.bisect_left(names, str(resume.get("key") or "")):]
        else:
            from ouroboros.task_store import terminal_writes_since

            written = terminal_writes_since(self.root, self.row["counter"])  # the owed begin where it stopped
            names = list(dict.fromkeys([*((self.row.get("owed") or {}).get("evidence") or ()), *sorted(waiting), *written]))
            if resume.get("plane") == "evidence" and resume.get("key") in written:
                resume["path"] = []  # written terminal again since its cursor: walked from its start
        settled = _settled(self.root, names)
        for index, task_id in enumerate(names):
            base = artifacts / task_id
            if not settled.get(task_id) or not _has_evidence(base):
                waiting.discard(task_id)  # not terminal (its next terminal write brings it back) or nothing here
                continue
            if not self.proven_over(task_id):  # terminal, a process may still write: asked again on a later pass
                waiting.add(task_id)
                continue
            waiting.discard(task_id)
            if not self.files("evidence", task_id, base, _evidence_part,
                              owned=lambda task_id=task_id: _settled(self.root, [task_id])[task_id]):
                return names[index:]
        self.cursor = {"plane": "seats", "key": "", "path": []}  # the plane is done for this pass
        return None

    def seats(self) -> Optional[List[str]]:
        """The seat plane: the homes still owed when the run stopped, else None."""
        from ouroboros.review_checking_prune import finished_home_held, finished_seat_homes

        seats, since, waiting = self.root / _SEATS, self.row.get("since"), self.waiting["seats"]
        present: Dict[str, float] = {}
        try:  # an unreadable listing raises (the pass stays owed); a home gone meanwhile is skipped
            with os.scandir(seats) as listing:
                for entry in listing:
                    try:
                        if entry.is_dir(follow_symlinks=False) and not entry.name.startswith("."):
                            # The folder's own record: a Windows listing serves a folder's times from its parent's
                            # index, which NTFS refreshes lazily, so a home changed after a pass read unchanged.
                            present[entry.name] = os.stat(entry.path, follow_symlinks=False).st_mtime
                    except FileNotFoundError:
                        continue
        except FileNotFoundError:
            pass
        wanted = set((self.row.get("owed") or {}).get("seats") or ()) | waiting
        wanted = (set(present) if since is None else wanted | {name for name, at in present.items() if at >= since}) & set(present)
        resume = self.resume  # the backlog resumes by name; a later pass owes its rest by name
        start = resume.get("key") if resume.get("plane") == "seats" and since is None else None
        key = resume.get("key") if resume.get("plane") == "seats" and since is not None else None
        if key in present and present[key] >= since:
            resume["path"] = []  # changed since the pass that stopped in it: walked from its start
        names = sorted(name for name in wanted if start is None or name >= start)
        verdicts = {home.name: verdict for home, verdict, _task in
                    finished_seat_homes(self.root, self.live, self.should_stop, self.proven, only=set(names))}
        waiting &= set(present)
        for index, name in enumerate(names):
            if self.should_stop():
                return names[index:]
            waiting.discard(name)
            if verdicts.get(name) == "in_use":
                waiting.add(name)
            if verdicts.get(name) != "offer":
                continue
            home = seats / name
            try:
                with finished_home_held(self.root, home, self.live, self.proven) as why:
                    if why:
                        waiting.add(name)
                        continue
                    if not self.files("seats", name, home, _seat_part):
                        return names[index:]
            except (OSError, ValueError, AttributeError):
                continue  # a marker that went unreadable: the home is no longer offered
        return None

    def logs(self, startup: Any) -> bool:
        """The log plane under STARTUP (the serving generation's ``startup_recovery`` record): each chain is
        judged against the row's witness (module docstring), what is held counted by class, then the segments
        an earlier generation's walk recorded are offered in name order. False once the run must stop. The
        listing it judged is the closed set the pass records when the walk ends. A listing that cannot be
        read is counted ``unreadable``: the walk ends with no record and holds, offers nothing, and the next
        start lists the folder again, while the other two planes keep what this run did."""
        recorded = self.row.get("witness")
        earlier = isinstance(recorded, dict) and recorded.get("generation") != _generation(startup)
        marks = recorded.get("chains") if earlier and isinstance(recorded.get("chains"), dict) else None
        unconfirmed = not isinstance(startup, dict) or startup.get("workers_unconfirmed") is not False
        try:
            chains = _closed(self.root)
        except OSError:  # counted, never taken for empty; the next start lists it again, and nothing is recorded
            self.counts["unreadable"] += 1
            return True
        held, offered = maintenance_jobs.Holds(), set()
        for chain, entries in chains.items():
            newest, count = ("", 0) if marks is None else _mark(marks.get(chain))
            closed = entries if marks is None else [entry for entry in entries if entry.name <= newest]
            why = ("closed_set_unrecorded" if marks is None else "prior_workers_unconfirmed" if unconfirmed
                   else "closed_set_changed" if len(closed) != count else "")
            if not why:
                offered.update(entry.name for entry in closed)
                continue
            for entry in closed:
                if _plain(entry):
                    held.note(why, entry.name)
        self.holds = held.rows()
        self.witness = {chain: [entries[-1].name, len(entries)] for chain, entries in chains.items()}
        self.cursor = {"plane": "logs", "key": "", "path": []}  # the plane's start, should the run stop before a file
        return self.files("logs", "", self.root / "archive", lambda parts, directory: not directory and (
            parts[0] in offered or artifact_storage.is_compression_sibling(parts[0])))


def _queue_live(root: pathlib.Path) -> Callable[[str], bool]:
    """The child's liveness probe: the queue snapshot's (live or unknown answers True)."""
    from ouroboros.task_status import task_has_live_queue_ownership

    return lambda task_id: task_has_live_queue_ownership(root, task_id)


def _added(total: Dict[str, Any], run: Dict[str, Any]) -> Dict[str, Any]:
    """A plane's totals with one run's added; an unknown saving stays unknown, never zero."""
    saved = total.get("saved_bytes", 0)
    return {"compressed": int(total.get("compressed") or 0) + run["compressed"],
            "saved_bytes": None if saved is None or run["saved_bytes"] is None else saved + run["saved_bytes"]}


def run(root: Any, should_stop: Callable[[], bool]) -> Dict[str, Any]:
    """One paced run (module docstring) at the utility disk tier, dataless files never read: this thread's
    policy alone (``storage.utility_io``), put back after, and each ``ditto`` through ``taskpolicy``."""
    from ouroboros.storage import utility_io

    root = pathlib.Path(root)
    row = maintenance_jobs.job(root, JOB) or {}
    if not artifact_storage.compression_supported():
        with store.open_store(root, write=True) as txn:
            maintenance_jobs.record_in(txn, JOB, state="done", unsupported=sys.platform)
            if "holds" in row:  # another platform's holds wait for a start that compresses nothing here
                maintenance_jobs.forget_in(txn, JOB, "holds")
        return {"state": "done", "items": 0}
    restore = utility_io(thread=True)
    try:
        return _pass(root, row, should_stop)
    finally:
        if restore is not None:
            restore()


def _pass(root: pathlib.Path, row: Dict[str, Any], should_stop: Callable[[], bool]) -> Dict[str, Any]:
    """The run's planes in order; its position, cursor, waits, holds and totals go to the job row. A position
    earned under another ``COVERAGE`` is dropped here, and durably only by a run that ends: one that throws
    leaves the old row, still due. A seat pass it interrupts keeps what it owes by name; the log plane walks
    only once the other two owe nothing, and its ``generation`` and ``witness`` move only when its walk ends, to
    the server generation current at the write that records them."""
    from ouroboros.utils import utc_now_iso

    if row.get("coverage") != COVERAGE:
        row = {**row, "counter": None, "cursor": None, "began": None}
    counter, now, startup = _counter(root), time.time(), _startup(root)
    generation = _generation(startup)
    began = row.get("began") or {"counter": counter, "at": now}  # a backlog pass keeps its start facts
    job = _Run(root, row, should_stop)
    owed_evidence = job.evidence()
    owed_seats = None if owed_evidence is not None else job.seats()
    owing = owed_evidence is not None or owed_seats is not None
    stopped = owing or (row.get("generation") != generation and not job.logs(startup))
    backlog = {"evidence": row.get("counter") is None, "seats": row.get("since") is None}
    owed_seats_now = (row.get("owed") or {}).get("seats") if owed_evidence is not None else owed_seats
    facts: Dict[str, Any] = {  # a backlog resumes at its cursor; a later pass owes its rest by name
        "state": "pending" if stopped else "done", "cursor": job.cursor if stopped else None,
        "coverage": COVERAGE, "counter": row.get("counter"),
        "generation": row.get("generation") if stopped else generation,
        "waiting": {plane: sorted(job.waiting[plane]) for plane in _WAITING},
        "owed": {"evidence": [] if backlog["evidence"] else owed_evidence or [],
                 "seats": [] if backlog["seats"] else owed_seats_now or []}}
    if job.census and (owed_evidence is None or not backlog["evidence"]):  # the census position moved
        facts["counter"] = began["counter"] if backlog["evidence"] else counter
    if owed_evidence is None and (owed_seats is None or not backlog["seats"]):  # the seats were looked at
        facts["since"] = began["at"] if backlog["seats"] else now
    facts["began"] = began if owing and (backlog["evidence"] or backlog["seats"]) else None
    totals = row.get("totals") if isinstance(row.get("totals"), dict) else {}
    facts["totals"] = {plane: _added(totals.get(plane) or {}, job.planes[plane]) for plane in PLANES}
    skipped = {**job.skipped, **({"unreadable": job.counts["unreadable"]} if job.counts["unreadable"] else {})}
    compressed = sum(job.planes[plane]["compressed"] for plane in PLANES)
    facts["last"] = {"at": now, "compressed": compressed, "skipped": skipped, "reaped": job.counts["reaped"]}
    if job.witness is not None and not stopped:  # the walk ended: what it listed closed, for a later generation's
        facts["witness"] = {"generation": generation, "chains": job.witness}
    if job.holds:  # the log plane judged its chains in this run: what it holds, since when
        facts["holds"] = maintenance_jobs._with_since(job.holds, row.get("holds"), utc_now_iso())
    # A supported run ends another platform's verdict; a log plane that judged its chains and holds nothing (``[]``)
    # leaves no ``holds`` field (Storage reads any, an empty list too, as a pass's report), while one that did not
    # judge them (``None``: not walked, or an archive it could not list) leaves the field as it stands.
    ended = [field for field, over in (("unsupported", True), ("holds", job.holds == [])) if over and field in row]
    with store.open_store(root, write=True) as txn:
        if "witness" in facts:
            # The generation current at this write, never the one the pass began under: a start during the pass
            # may have a reader holding, by inode, a segment the walk listed closed. Its record then names that
            # start's generation, under which nothing it lists is offered.
            from ouroboros.maintenance_retirement import STARTUP

            facts["generation"] = facts["witness"]["generation"] = _generation(txn.meta(STARTUP))
        maintenance_jobs.record_in(txn, JOB, **facts)
        if ended:
            maintenance_jobs.forget_in(txn, JOB, *ended)
    return {"state": facts["state"], "items": compressed}
