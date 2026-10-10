"""Process custody: supervised spawning plus a durable orphan ledger.

Every long-lived OS process Ouroboros spawns should go through
``spawn_supervised`` so its identity lands in ``data/state/process_ledger.jsonl``
BEFORE it can be orphaned. The reaper (server startup + periodic tick) then
kills ledger entries whose owning generation is gone, matching processes by a
STRICT (pid, start_time, cmd_sha256) fingerprint — never by command-line
class, so a dev instance can never reap a packaged instance's processes (or
vice versa).

Scopes:
  - ``task``:    dies with its owning task; reapable as soon as the task is no
                 longer running (and always across server generations).
  - ``session``: dies with the server generation (session_id mismatch → reap).
  - ``daemon``:  installation-owned processes (e.g. the shared Claudexor daemon)
                 outlive generations — never killed. Skill COMPANIONS also record
                 daemon scope but are the exception: ``reap_orphaned_processes``
                 reaps them on owner-uninstall or a foreign generation.

This module deliberately lives OUTSIDE platform_layer (primitives-only) and
adds no policy to the panic layers it complements (``_active_subprocesses``,
port sweeps, Windows Job Objects all stay).
"""

from __future__ import annotations

import hashlib
import logging
import os
import pathlib
import subprocess
import time
import uuid
from typing import Any, Dict, List, Optional

from ouroboros.platform_layer import (
    IS_WINDOWS,
    current_process_group_id,
    kill_process_group_id,
    kill_process_tree,
    pid_is_alive,
    process_command,
    process_group_id,
    process_start_time,
    process_start_time_legacy,
    subprocess_new_group_kwargs,
)
from ouroboros.process_containment import pid_is_zombie, process_group_has_live_members
from ouroboros.utils import append_jsonl, utc_now_iso

log = logging.getLogger(__name__)

LEDGER_FILENAME = "process_ledger.jsonl"
# One ledger generation per server process; recorded into every entry so the
# reaper can tell "mine" from "previous generation" without guessing.
_SESSION_ID = uuid.uuid4().hex
_VALID_SCOPES = ("task", "session", "daemon")


def current_custody_session_id() -> str:
    return _SESSION_ID


def adopt_session_id(value: str) -> None:
    """Adopt a parent process's custody session id.

    Workers started with the 'spawn' multiprocessing method (the default on
    macOS/Windows, and forced on Linux by the Terminal-Bench harness) re-import
    this module and would otherwise generate a fresh ``_SESSION_ID``. Every
    process such a worker records (its task/session-scoped services, executor
    children, local model server) would then look like a *foreign generation*
    to the server's periodic reaper — which kills task- and session-scoped
    foreign entries — so a still-running task's services get SIGKILLed at the
    next reap tick. The worker entrypoint calls this with the server's id.

    The id is passed as a spawn ARGUMENT, never via ambient env: an env var
    would survive ``server_control.restart_current_process`` (which hands the
    server over with ``os.environ.copy()``: exec on POSIX, spawn on Windows),
    making a freshly restarted server adopt the dead generation's id and treat
    leftover processes as same-session survivors — the inverse leak. A spawn
    arg survives neither transfer.
    """
    global _SESSION_ID
    v = str(value or "").strip()
    if v:
        _SESSION_ID = v


def ledger_path(drive_root: pathlib.Path) -> pathlib.Path:
    return pathlib.Path(drive_root) / "state" / LEDGER_FILENAME


def _cmd_sha256(cmd: Any) -> str:
    try:
        if isinstance(cmd, (list, tuple)):
            text = "\0".join(str(part) for part in cmd)
        else:
            text = str(cmd or "")
        return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()
    except Exception:
        return ""


def _live_cmd_sha256(pid: int) -> str:
    command = process_command(pid)
    if not command:
        return ""
    return hashlib.sha256(command.encode("utf-8", errors="replace")).hexdigest()


def record_process(
    drive_root: pathlib.Path,
    *,
    pid: int,
    cmd: Any,
    purpose: str,
    scope: str,
    owner_task_id: str = "",
    reap_process_group: bool = True,
) -> Dict[str, Any]:
    """Append a custody record for an already-spawned process."""
    if scope not in _VALID_SCOPES:
        raise ValueError(f"process custody scope must be one of {_VALID_SCOPES}, got {scope!r}")
    try:
        pgid = process_group_id(pid)
    except Exception:
        pgid = 0
    entry = {
        "ts": utc_now_iso(),
        "pid": int(pid),
        "pgid": int(pgid or 0) if reap_process_group else 0,
        "fingerprint": {
            # DOWNGRADE-SAFE split: the unversioned field keeps the legacy ``ps``
            # spelling an N-1 reader still understands (a rollback that reads a
            # boot-qualified token here would mismatch every row and prune it
            # WITHOUT killing, orphaning the process forever), while the current
            # reader prefers the boot-qualified sibling below. One extra ``ps``
            # per SPAWN (the write path always paid it before /proc-first); the
            # hot sweep path stays subprocess-free.
            "start_time": (legacy_start := process_start_time_legacy(pid)),
            **(
                {"start_time_boot": boot_start}
                if (boot_start := process_start_time(pid)) and boot_start != legacy_start
                else {}
            ),
            # The LIVE command line (what the OS reports) is the reap-time
            # comparison anchor; the argv we passed may differ cosmetically.
            "cmd_sha256": _live_cmd_sha256(pid) or _cmd_sha256(cmd),
        },
        "purpose": str(purpose or "")[:200],
        "scope": scope,
        "owner_task": str(owner_task_id or ""),
        "session_id": _SESSION_ID,
    }
    if not append_jsonl(ledger_path(drive_root), entry):
        raise OSError("process custody record could not be written")
    from ouroboros.owned_shutdown import record_ledgered_process

    # The ledger stays the reaper's custody; the installation's ownership set names the row too.
    if not record_ledgered_process(drive_root, entry):
        log.warning("process %s (%s) is ledgered but not in the ownership set", pid, purpose)
    return entry


def spawn_supervised(
    cmd: Any,
    *,
    drive_root: pathlib.Path,
    purpose: str,
    scope: str,
    owner_task_id: str = "",
    new_process_group: bool = True,
    on_spawn: Any = None,
    **popen_kwargs: Any,
) -> subprocess.Popen:
    """Popen + durable custody record (the single supervised chokepoint).

    The record is written right after ``Popen`` returns, so a spawner that dies
    any time later cannot orphan the child invisibly (the reaper finds it in the
    ledger); a hard kill INSIDE that spawn-to-record window is the disclosed
    residual — such a child is unledgered and the reaper cannot see it.
    ``on_spawn`` publishes the Popen into its existing owner before custody I/O;
    it must not wait or persist. Callback failure follows normal spawn cleanup.
    The returned Popen retains its exact durable row as ``_ouroboros_custody``.
    """
    if new_process_group:
        merged = dict(subprocess_new_group_kwargs())
        merged.update(popen_kwargs)
        popen_kwargs = merged
    from ouroboros.owner_pause import operation_start

    with operation_start():
        proc = subprocess.Popen(cmd, **popen_kwargs)  # noqa: S603 — callers pass vetted argv lists
    try:
        if on_spawn is not None:
            on_spawn(proc)
        proc._ouroboros_custody = record_process(
            drive_root,
            pid=proc.pid,
            cmd=cmd,
            purpose=purpose,
            scope=scope,
            owner_task_id=owner_task_id,
        )
    except Exception as exc:
        log.warning("process custody record failed for pid %s (%s)", proc.pid, purpose, exc_info=True)
        kill_process_tree(proc)
        try:
            proc.wait(timeout=5)
        except Exception:
            pass
        raise RuntimeError("spawned process could not enter durable custody") from exc
    return proc


def _legacy_start_matches(pid: int, recorded: str, current: str) -> bool:
    """Compare a recorded LEGACY spelling against a live boot-qualified token.

    Runs solely AFTER the direct equalities already failed, so the one extra ``ps``
    stays off the reaper's hot path and a genuinely different process still fails
    every form. ``recorded`` is the ``ps -o lstart=`` field (the spelling the ledger
    keeps writing for downgrade safety) or, from the one host class with no usable
    ``ps``, a bare tick count — which ``process_start_time_legacy`` reproduces there,
    so that shape resolves too. What this helper deliberately does NOT do is treat a
    bare tick as the tick-half of a boot-qualified live token: ticks recur across
    reboots, and a token carrying no boot id must never authorize a kill (a mismatch
    prunes; it does not kill).

    Residual, stated rather than hidden: the ``"<ticks>."`` separator form, minted
    only when the boot id AND ``ps`` both fail, string-matches its cross-boot twin on
    the direct equality BEFORE this helper is consulted. That is why the separator
    form is the mint order's last resort rather than the boot-id-less default.
    """
    ticks, sep, _ = current.partition(".")
    if not (sep and ticks.isdigit()):
        # The current token IS the legacy ``ps`` form (no /proc, or the boot id was
        # unreadable so ``ps`` won the mint order): re-running ``ps`` would return the
        # byte-identical value that already failed the equality above, and a bare-tick
        # row deliberately does NOT resolve here — see below.
        return False
    # A token carrying NO boot id must never authorize a kill: boot-relative ticks
    # recur across reboots, and a recycled pid + the same command hash is exactly the
    # cross-boot collision this change exists to refuse. The one host class that ever
    # MINTS bare ticks (no usable ``ps``) still resolves through
    # ``process_start_time_legacy``, whose own fallback IS ``str(ticks)`` there; on a
    # ``ps``-capable host a bare-tick row compares against the ``ps`` spelling, fails,
    # and is PRUNED — the safe direction (prune, never kill).
    return bool(recorded) and process_start_time_legacy(pid) == recorded


def _fingerprint_matches(entry: Dict[str, Any], *, require_measured: bool = False) -> bool:
    """STRICT identity: the live process must still BE the recorded one.

    pid alive + same start_time (when we have one) + same command hash (when
    we have one). A recycled pid fails this and is left alone. We never match
    by command-line class. The start-time comparison is DUAL-FORMAT on Linux:
    the cheap current representation first, and the pre-upgrade spelling only
    once that mismatched (see ``_legacy_start_matches``). Explicit stop sets
    ``require_measured``: both recorded dimensions must match these exact live
    observations; retention alone may tolerate unavailable measurements.
    """
    pid = int(entry.get("pid") or 0)
    if pid <= 0 or not pid_is_alive(pid) or pid_is_zombie(pid):
        return False
    fp = entry.get("fingerprint") if isinstance(entry.get("fingerprint"), dict) else {}
    recorded_boot = str(fp.get("start_time_boot") or "")
    recorded_start = str(fp.get("start_time") or "")
    recorded_cmd = str(fp.get("cmd_sha256") or "")
    if require_measured and not ((recorded_boot or recorded_start) and recorded_cmd):
        return False
    if IS_WINDOWS and not (recorded_boot or recorded_start):
        # Before native Windows observations this hash came from submitted argv,
        # not the live command spelling. Keep its old liveness-only retention;
        # it never supplies the measured identity required to signal a process.
        return not require_measured
    if recorded_boot or recorded_start:
        live_start = process_start_time(pid)
        if require_measured and not live_start:
            return False
        if live_start and not (
            # Preferred: the exact boot-qualified token of a row written by THIS line.
            (recorded_boot and live_start == recorded_boot)
            # Same-form equality (macOS/BSD ``ps`` rows, and every pre-upgrade row
            # whose spelling the live mint still produces).
            or live_start == recorded_start
            # Compatibility spellings: a boot-qualified live token against the
            # legacy ``ps`` field (boot id became readable mid-generation, or a
            # pre-upgrade row) — one ``ps``, only on this already-mismatched path,
            # and ONLY for rows carrying no boot sibling: a mismatched recorded
            # boot token is POSITIVE evidence of another boot, and on a ``ps``-less
            # host the legacy helper degrades to bare ticks, which would let a
            # cross-boot recycled pid resolve through the fallback. Refuted boot
            # evidence prunes; it never re-qualifies through a weaker spelling.
            or (not recorded_boot and _legacy_start_matches(pid, recorded_start, live_start))
        ):
            return False
    if recorded_cmd:
        live_cmd = _live_cmd_sha256(pid)
        if require_measured and not live_cmd:
            return False
        if live_cmd and live_cmd != recorded_cmd:
            return False
        if not live_cmd and not recorded_start:
            # Windows: no command line and no start time — liveness is all we
            # have (same degradation as workspace_executor records).
            return True
    return True


def _service_group_survives_leader(entry: Dict[str, Any]) -> bool:
    """Keep dead-leader evidence while a group member can still execute."""
    purpose = str(entry.get("purpose") or "")
    scope = str(entry.get("scope") or "")
    pgid = int(entry.get("pgid") or 0)
    return bool(
        purpose.startswith(("service:", "workspace_service:"))
        and scope in {"task", "session"}
        and pgid > 0
        and process_group_has_live_members(pgid)
    )


def _read_ledger_records(
    drive_root: pathlib.Path, *, strict: bool
) -> tuple[bool, List[Dict[str, Any]], bytes]:
    """Return the latest-per-PID view and the exact bytes that produced it."""
    path = ledger_path(drive_root)
    try:
        path.stat()
    except FileNotFoundError:
        if strict and path.is_symlink():
            return False, [], b""
        return True, [], b""
    except OSError:
        return False, [], b""
    entries: List[Dict[str, Any]] = []
    snapshot = b""
    try:
        import json

        snapshot = path.read_bytes()
        # Match the compactor's physical JSONL boundaries. Unicode separators
        # inside JSON strings are payload bytes, not new ledger records.
        for raw_line in snapshot.splitlines():
            line = raw_line.decode("utf-8", errors="strict" if strict else "replace").strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                if strict:
                    return False, [], snapshot
                continue
            if not isinstance(obj, dict) or not obj.get("pid"):
                if strict:
                    return False, [], snapshot
                continue
            entries.append(obj)
    except (OSError, UnicodeError):
        return False, [], snapshot
    # Last record per pid wins (a pid may be re-registered by a newer spawn).
    by_pid: Dict[int, Dict[str, Any]] = {}
    for entry in entries:
        try:
            by_pid[int(entry.get("pid") or 0)] = entry
        except (TypeError, ValueError):
            if strict:
                return False, [], snapshot
            continue
    by_pid.pop(0, None)
    return True, list(by_pid.values()), snapshot


def _read_ledger_strict(drive_root: pathlib.Path) -> tuple[bool, List[Dict[str, Any]]]:
    return _read_ledger_records(drive_root, strict=True)[:2]


def _read_ledger(drive_root: pathlib.Path) -> List[Dict[str, Any]]:
    return _read_ledger_records(drive_root, strict=False)[1]


def _rewrite_ledger(
    drive_root: pathlib.Path, entries: List[Dict[str, Any]], *,
    previous: Optional[bytes] = None,
) -> None:
    """Compact only the observed prefix, preserving concurrent and opaque bytes.

    ``previous=None`` is the explicit replacement form used by isolated fixtures.
    Lifecycle callers pass their raw read snapshot, not the deduplicated view.
    If another rewrite changed that prefix, defer to a fresh sweep. Signals and
    waits stay outside this short transaction so new spawns can enter custody.
    PIDs whose rows it drops leave the ownership set once the lock is released.
    """
    import json

    path = ledger_path(drive_root)
    dropped: set = set()
    try:
        from ouroboros.utils import jsonl_append_lock_path, replace_atomic
        from ouroboros.platform_layer import acquire_exclusive_file_lock, release_exclusive_file_lock

        lock_path = jsonl_append_lock_path(path)
        lock_fd = acquire_exclusive_file_lock(lock_path, timeout_sec=2.0, stale_sec=10.0, owner_aware_stale=True)
        if lock_fd is None:
            log.warning("process ledger rewrite skipped: append lock unavailable")
            return
        try:
            seen, named = set(), set()
            if previous is None:
                payload = "".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in entries).encode("utf-8")
            else:
                current = path.read_bytes()
                if not current.startswith(previous):
                    return
                survivors = {int(entry.get("pid") or 0): entry for entry in entries}
                kept = []
                for line in reversed(previous.splitlines(keepends=True)):
                    try:
                        row = json.loads(line)
                        pid = int(row.get("pid") or 0) if isinstance(row, dict) else 0
                    except (TypeError, ValueError, UnicodeError):
                        pid = 0
                    seen.add(pid)
                    # Only the last observed row can survive for this PID.
                    # Unparseable rows remain literal bytes, never deletion authority.
                    if not pid or survivors.pop(pid, None) == row:
                        kept.append(line)
                        named.add(pid)
                for line in current[len(previous):].splitlines():
                    try:
                        named.add(int(json.loads(line).get("pid") or 0))
                    except Exception:
                        continue
                payload = b"".join(reversed(kept)) + current[len(previous):]
            tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
            tmp.write_bytes(payload)
            replace_atomic(tmp, path)
            dropped = seen - named - {0}
        finally:
            release_exclusive_file_lock(lock_path, lock_fd)
    except Exception:
        log.debug("process ledger rewrite failed", exc_info=True)
    if dropped:
        from ouroboros.owned_shutdown import forget_ledgered_pids

        forget_ledgered_pids(drive_root, dropped)  # the ledger's close path is the set's too


def _multiprocessing_parent_sentinel() -> Optional[int]:
    """The spawner's sentinel fd inside a ``multiprocessing`` child, else None.

    Every start method hands the child the read end of a pipe whose only write
    end lives in the process that called ``Process.start()`` and stays open for
    the child's lifetime (``popen_fork``/``popen_spawn_posix`` keep it in the
    Popen finalizer; ``popen_forkserver`` dups one exactly "as a sentinel of the
    parent process used by the child"). EOF therefore means the SPAWNER died,
    under forkserver included -- there the ppid is the forkserver, which
    outlives a dead supervisor for as long as any worker holds its alive pipe.
    """
    try:
        import multiprocessing

        parent = multiprocessing.parent_process()
        return None if parent is None else int(parent.sentinel)
    except Exception:
        return None


def start_parent_lifeline(*, poll_sec: float = 5.0, label: str = "", stop_socket=None, before_exit=None) -> None:
    """Daemon watchdog: group-suicide when the spawning parent dies (POSIX).

    For OUR python entrypoints only (workers, extension runner, claude child):
    when the parent dies the child would otherwise keep burning CPU/budget
    invisibly. Inside a ``multiprocessing`` child the watched parent is the
    spawner's sentinel (EOF on its death under fork, spawn and forkserver
    alike); a plain subprocess falls back to its ppid, which changes when the
    parent dies (orphans go to init/launchd or the nearest subreaper). A ppid
    change still fires inside a multiprocessing child too: under forkserver it
    means the forkserver died, which the supervisor already reads as this
    worker's exit 255. Arbitrary-argv services and skills cannot get a watchdog
    injected -- they are covered by the ledger + reaper instead.
    """
    if os.name == "nt" and stop_socket is None:
        return  # ordinary Windows children are covered by Job Objects

    import threading
    import time as _time
    from multiprocessing.connection import wait as _mp_wait

    def _suicide(ready=()) -> None:
        try:
            # EOF/parent death is ordinary lifetime cleanup, not owner Panic:
            # it must preserve installation daemons in their separate groups.
            emergency = stop_socket is not None and stop_socket in ready and stop_socket.recv(1) == b"!"
            if emergency and before_exit is not None:
                # Callback imports, locks and even a stuck Popen cannot hold
                # this watchdog. All local owners get a bounded request chance.
                request = threading.Thread(target=before_exit, daemon=True)
                request.start()
                request.join(timeout=0.5)
        finally:
            try:
                # No imports, logging handlers, descendant scans or persistence
                # between the callback deadline and hard exit. Our own live
                # session leader is positive identity, not a guessed PID/group.
                if current_process_group_id() == os.getpid():
                    kill_process_group_id(os.getpid())
            finally:
                os._exit(1)

    initial_ppid = os.getppid()
    sentinel = _multiprocessing_parent_sentinel()
    watched = ([sentinel] if sentinel is not None else []) + ([stop_socket] if stop_socket is not None else [])
    ready = []
    try:
        ready = _mp_wait(watched, timeout=0) if watched else []
        died_early = initial_ppid <= 1 or bool(ready)
    except Exception:
        sentinel, died_early = None, initial_ppid <= 1
    if died_early:
        # The parent died before we even got here (import-delay race after an
        # abrupt supervisor kill). These entrypoints are always spawned by a
        # live Ouroboros parent, so an orphan at startup is already a leak.
        _suicide(ready)
        return

    def _watch() -> None:
        nonlocal sentinel
        while True:
            if sentinel is None:
                ready = _mp_wait([stop_socket], timeout=poll_sec) if stop_socket is not None else []
                if ready:
                    _suicide(ready)
                else:
                    _time.sleep(poll_sec)
            else:
                try:
                    ready = _mp_wait(watched, timeout=poll_sec)
                except Exception:
                    # An unusable sentinel (closed fd) must not kill a live
                    # task: degrade to the ppid watch rather than guess.
                    sentinel, ready = None, []
                if ready:
                    _suicide(ready)
            if os.getppid() != initial_ppid:
                _suicide()

    threading.Thread(target=_watch, daemon=True, name=f"parent-lifeline-{label or 'child'}").start()


def live_kept_service_pids(drive_root: pathlib.Path) -> "set[int]":
    """PIDs of still-alive, deliberately-kept (session-scope) services.

    Used by the cancel/hard-timeout worker kill to spare ``service_teardown=keep``
    services that are direct children of the worker: tree-killing the worker
    would otherwise destroy services a verifier still needs, even though the
    keep contract says they outlive the task. Only live, fingerprint-matching
    session-scoped service entries are returned.
    """
    pids: set[int] = set()
    try:
        for entry in _read_ledger(pathlib.Path(drive_root)):
            if str(entry.get("scope") or "") != "session":
                continue
            # Both the in-process service path (purpose "service:<name>") and the
            # local-executor path (purpose "workspace_service:<name>") record
            # deliberately-kept services; spare both.
            if not str(entry.get("purpose") or "").startswith(("service:", "workspace_service:")):
                continue
            if not _fingerprint_matches(entry):
                continue
            pid = int(entry.get("pid") or 0)
            if pid > 0:
                pids.add(pid)
    except Exception:
        return pids
    return pids


def live_daemon_root_pids(
    drive_root: pathlib.Path, *, retained_purposes: Optional[set[str]] = None,
    purposes: Optional[set[str]] = None, strict: bool = False,
) -> "set[int]":
    """PIDs of still-alive installation-owned (``daemon``-scope) ledger roots.

    Used by every worker tree-kill to spare a process whose lifetime belongs to
    the installation, not to the worker that happened to spawn it: the shared
    Claudexor daemon is such a root when a task worker was the first to need it,
    and its paid runs outlive that worker and the server generation alike. Only
    live, fingerprint-matching ``daemon`` rows qualify; ``kill_pid_tree`` spares
    an excluded pid together with its own descendants, so the delegated harness
    runs under the daemon survive too. Sparing is the safe direction — a row the
    reaper would keep is a row a worker teardown must not kill. Lifecycle admission
    may restrict ``purposes`` and require a readable ledger with ``strict=True``;
    absence is empty, corruption is unknown. Neither form grants signal authority.
    """
    pids: set[int] = set()
    try:
        if strict:
            readable, entries = _read_ledger_strict(pathlib.Path(drive_root))
            if not readable:
                raise OSError("process custody ledger is unreadable or corrupt")
        else:
            entries = _read_ledger(pathlib.Path(drive_root))
        for entry in entries:
            if purposes is not None and entry.get("purpose") not in purposes:
                continue
            retained = entry.get("purpose") in (retained_purposes or set())
            if (entry.get("scope") != "daemon" and not retained) or not _fingerprint_matches(entry):
                continue
            pid = int(entry.get("pid") or 0)
            if pid > 0:
                pids.add(pid)
    except Exception:
        if strict:
            raise
        return pids
    return pids


def pending_process_stops(drive_root: pathlib.Path, purposes: "set[str]") -> List[str]:
    """Read unresolved custody for stop diagnostics, never as signal authority."""
    readable, entries = _read_ledger_strict(drive_root)
    if not readable:
        return ["process custody ledger unreadable"]
    pending = []
    for entry in entries:
        if entry.get("purpose") not in purposes:
            continue
        pid, pgid = int(entry.get("pid") or 0), int(entry.get("pgid") or 0)
        if _fingerprint_matches(entry) or (pgid > 0 and process_group_has_live_members(pgid)):
            pending.append(f"process {pid} remains alive or unconfirmed")
    return pending


def process_stop_snapshot(drive_root: pathlib.Path, purposes: "set[str]") -> List[Dict[str, Any]]:
    """Capture this stop's ledger rows before an asynchronous shutdown request."""
    readable, entries = _read_ledger_strict(pathlib.Path(drive_root))
    if not readable:
        raise OSError("process custody ledger is unreadable or corrupt")
    return [entry for entry in entries if entry.get("purpose") in purposes and _fingerprint_matches(entry)]


def stop_ledgered_processes(
    drive_root: pathlib.Path, purposes: "set[str]", *, timeout_sec: float = 5.0,
    unconfirmed: Optional[List[str]] = None,
    expected_entries: Optional[List[Dict[str, Any]]] = None,
) -> List[int]:
    """Kill the installation's own processes of the named purposes, any scope.

    The lifecycle owner's explicit stop (Panic): identity is the recorded row
    under THIS drive root with a confirmed live fingerprint — never a command-
    line class, a process name or a port, so a foreign daemon that recycled our
    descriptor port is never signalled. Legacy ``session`` rows of the same
    purpose are stopped too: the process is ours whichever generation recorded
    it. A stopped row leaves the ledger with a ``process_stopped`` supervisor
    row; a row that fails identity stays for the reaper to judge. Returns the
    stopped pids; optional diagnostics retain failed signals and known children
    even after their leader exits. Each eligible row gets its own exit window.
    """
    drive_root = pathlib.Path(drive_root)
    stopped: List[int] = []
    survivors: List[Dict[str, Any]] = []
    _, entries, previous = _read_ledger_records(drive_root, strict=False)
    failures = unconfirmed if unconfirmed is not None else []
    for entry in entries:
        purpose = str(entry.get("purpose") or "")
        if (purpose not in purposes
                or (expected_entries is not None and entry not in expected_entries)
                or not _fingerprint_matches(entry, require_measured=True)):
            survivors.append(entry)
            continue
        pid = int(entry.get("pid") or 0)
        pgid = int(entry.get("pgid") or 0)
        from ouroboros.platform_layer import collect_descendant_pids, kill_pid_tree

        children = collect_descendant_pids(pid)
        deadline = time.monotonic() + max(0.0, timeout_sec)
        try:
            # Harness children lead their own groups. Capture and stop the PID
            # tree before its parent dies, then sweep the original group too.
            kill_pid_tree(pid)
            if pgid > 0:
                kill_process_group_id(pgid)
        except Exception:
            log.warning("Failed to stop ledgered process %s", pid, exc_info=True)
            failures.append(f"process {pid} signal failed")
            survivors.append(entry)
            continue
        while True:
            alive = (
                _fingerprint_matches(entry) or (pgid > 0 and process_group_has_live_members(pgid))
                or any(pid_is_alive(child) and not pid_is_zombie(child) for child in children)
            )
            if not alive or time.monotonic() >= deadline:
                break
            time.sleep(0.05)
        if alive:
            log.warning("process %s stop is unconfirmed; custody retained", pid)
            failures.append(f"process {pid} tree exit unconfirmed")
            survivors.append(entry)
            continue
        stopped.append(pid)
        append_jsonl(drive_root / "logs" / "supervisor.jsonl", {
            "ts": utc_now_iso(),
            "type": "process_stopped",
            "pid": pid,
            "pgid": pgid,
            "purpose": purpose,
            "scope": str(entry.get("scope") or ""),
            "recorded_session": str(entry.get("session_id") or ""),
            "reason": "owner_stop",
        })
    if stopped:
        _rewrite_ledger(drive_root, survivors, previous=previous)
    return stopped


def quiesce_custodied_services(
    drive_root: pathlib.Path, *, timeout_sec: float = 5.0
) -> tuple[bool, List[str]]:
    """Kill and verify every ledgered task/session service before repo replacement."""
    drive_root = pathlib.Path(drive_root)
    readable, entries, previous = _read_ledger_records(drive_root, strict=True)
    if not readable:
        return False, ["custody_ledger:unreadable"]
    targets: List[Dict[str, Any]] = []
    survivors: List[Dict[str, Any]] = []
    blockers: List[str] = []
    for entry in entries:
        purpose = str(entry.get("purpose") or "")
        is_service = purpose.startswith(("service:", "workspace_service:"))
        leader_matches = _fingerprint_matches(entry)
        if (
            is_service
            and str(entry.get("scope") or "") in {"task", "session"}
            and (leader_matches or _service_group_survives_leader(entry))
        ):
            targets.append(entry)
        elif leader_matches:
            survivors.append(entry)

    from ouroboros.platform_layer import kill_pid_tree

    for entry in targets:
        pid = int(entry.get("pid") or 0)
        pgid = int(entry.get("pgid") or 0)
        if pgid > 0:
            kill_process_group_id(pgid)
        else:
            kill_pid_tree(pid)
    deadline = time.monotonic() + max(0.0, float(timeout_sec))
    while targets and time.monotonic() < deadline:
        if not any(
            _fingerprint_matches(entry) or _service_group_survives_leader(entry)
            for entry in targets
        ):
            break
        time.sleep(0.05)
    for entry in targets:
        if _fingerprint_matches(entry) or _service_group_survives_leader(entry):
            survivors.append(entry)
            blockers.append(f"custody_service:{int(entry.get('pid') or 0)}")
        else:
            append_jsonl(drive_root / "logs" / "supervisor.jsonl", {
                "ts": utc_now_iso(),
                "type": "process_reaped_for_update",
                "pid": int(entry.get("pid") or 0),
                "pgid": int(entry.get("pgid") or 0),
                "purpose": entry.get("purpose"),
            })
    _rewrite_ledger(drive_root, survivors, previous=previous)
    return not blockers, blockers


def reap_orphaned_processes(
    drive_root: pathlib.Path,
    *,
    running_task_ids: Optional[Any] = None,
    live_owner_skills: Optional[set] = None,
    enforce_companion_reap: bool = False,
    retained_purposes: Optional[set[str]] = None,
) -> List[int]:
    """Kill ledgered processes whose owning generation/task is gone.

    ``running_task_ids`` is the live-owner set, or a zero-arg callable that
    produces it — read AFTER the ledger, never before (see below). ``None``
    still means UNKNOWN: no task-owner decision is taken at all.

    Rules:
      - dead pid / fingerprint mismatch → prune the entry, never kill;
      - current session's entries → keep (their owners are alive);
        EXCEPT task-scoped entries whose owner task is no longer running;
      - previous generations: task/session scopes → kill group + reap event;
        daemon scope → keep (installation-owned lifecycles).
      - ``retained_purposes`` lets the lifecycle owner preserve legacy records
        of its installation-owned process, only after fingerprint identity
        matches. The ledger's drive root and exact recorded process remain the
        provenance; a purpose name alone never rescues a stale/recycled row.
      - skill COMPANIONS (purpose ``companion:<skill>:<name>``, recorded under
        daemon scope) are the exception to daemon-keep: reap when the owner skill
        is UNINSTALLED (not in ``live_owner_skills``) OR the entry is from a
        FOREIGN generation (``CompanionSupervisor.start()`` always re-spawns a
        fresh pid, so a fingerprint-matching companion from another generation is
        a stale duplicate that also blocks the re-spawn on a port conflict).
        Same-session companions are killed only when their owner is uninstalled.
        ``live_owner_skills`` (installed skill names, disk-derived) is passed in
        to keep this module policy-free; if None **or empty** the companion
        clause keeps everything (fail-safe — never mass-kill on missing info; an
        explicitly empty set is normalized to None below, so no caller can
        trigger a companion mass-reap by passing an empty install set). Transient
        not-live states (disabled / review / deps) are ``stop_skill``'s job, not
        the reaper's. ``enforce_companion_reap=False`` (default) is LOG-ONLY:
        records a ``process_would_reap`` event instead of killing — a safe first
        rollout; flip to True to enforce.
    """
    drive_root = pathlib.Path(drive_root)
    # Fail-safe (defense-in-depth): an explicitly EMPTY live_owner_skills means
    # UNKNOWN (keep-all), NOT "every skill uninstalled". The only production
    # producer (server._installed_skill_names) already coalesces an empty/failed
    # discovery to None; normalizing here too guarantees no caller can trigger a
    # companion mass-reap by handing in an empty set.
    if live_owner_skills is not None and not live_owner_skills:
        live_owner_skills = None
    _, entries, previous = _read_ledger_records(drive_root, strict=False)
    if not entries:
        return []
    # CANDIDATES FIRST, LIVENESS SECOND. A candidate exists ⇒ its owner was registered
    # earlier (admission takes ``_queue_lock`` before any spawn), so an owner absent
    # from this LATER snapshot is really gone — while a set read BEFORE the ledger
    # reaps a task admitted during the read (the sweep runs off the loop thread).
    if callable(running_task_ids):
        running_task_ids = running_task_ids()
    retained_roots = {
        int(entry["pid"]) for entry in entries
        if ((entry.get("scope") == "daemon" and not str(entry.get("purpose") or "").startswith("companion:"))
            or entry.get("purpose") in (retained_purposes or set()))
        and _fingerprint_matches(entry)
    }
    reaped: List[int] = []
    survivors: List[Dict[str, Any]] = []
    for entry in entries:
        pid = int(entry.get("pid") or 0)
        scope = str(entry.get("scope") or "task")
        same_session = str(entry.get("session_id") or "") == _SESSION_ID
        # CHEAP PATH for the current generation's own session processes. The branch
        # further down keeps every same-session session entry regardless of what the
        # fingerprint said (``task_owner_gone`` is task-scope-only), so for a LIVE pid
        # the cheap path only ever KEEPS: an alive-but-RECYCLED same-session pid is
        # retained one generation (foreign next generation -> full fingerprint ->
        # prune), never killed. The skipped fingerprint only cost a `ps` per row on
        # every 600s tick (the startup sweep sees prior-generation rows as
        # foreign-session, so it saves nothing there). This is the hot majority of the ledger:
        # worker-pool members, the SyncManager, the local-model
        # server and keep-services are all scope="session".
        #
        # A DEAD pid deliberately FALLS THROUGH instead of pruning here: a session
        # service whose leader died while its process group is still alive is preserved
        # today by ``_service_group_survives_leader``, and a prune-on-dead shortcut would
        # throw that evidence away. Companions are excluded so the daemon-scope companion
        # rules below stay the only authority on them even for a malformed session-scope
        # row. This path only ever KEEPS — it can never authorize a kill.
        if (
            scope == "session"
            and same_session
            and pid > 0
            and not str(entry.get("purpose") or "").startswith("companion:")
            and pid_is_alive(pid)
            and not pid_is_zombie(pid)
        ):
            survivors.append(entry)
            continue
        leader_matches = _fingerprint_matches(entry)
        group_survives = _service_group_survives_leader(entry)
        if not leader_matches and not group_survives:
            continue  # dead/recycled pid with no surviving service group: prune silently
        if leader_matches and entry.get("purpose") in (retained_purposes or set()):
            survivors.append(entry)
            continue
        owner_task = str(entry.get("owner_task") or "")
        purpose = str(entry.get("purpose") or "")
        task_owner_gone = (
            scope == "task"
            and running_task_ids is not None
            and owner_task
            and owner_task not in running_task_ids
        )
        # Skill companions (purpose "companion:<skill>:<name>", daemon scope):
        # reap when the owner skill is uninstalled OR the entry is from a foreign
        # generation (start() always re-spawns → another generation's companion
        # is a stale duplicate). Fail-safe: keep when the live set is unknown or
        # the purpose can't be parsed. Same-session companions are killed only
        # when their owner is uninstalled; foreign-generation ones always are.
        if purpose.startswith("companion:"):
            parts = purpose.split(":", 2)
            owner_skill = parts[1] if (len(parts) >= 3 and parts[0] == "companion") else ""
            owner_uninstalled = (
                bool(owner_skill)
                and live_owner_skills is not None
                and owner_skill not in live_owner_skills
            )
            reapable = (
                live_owner_skills is not None
                and bool(owner_skill)
                and (owner_uninstalled or not same_session)
            )
            if not reapable:
                survivors.append(entry)
                continue
            if not enforce_companion_reap:
                # First-rollout safety: record the intent, do NOT kill yet.
                append_jsonl(drive_root / "logs" / "supervisor.jsonl", {
                    "ts": utc_now_iso(),
                    "type": "process_would_reap",
                    "pid": pid,
                    "pgid": int(entry.get("pgid") or 0),
                    "purpose": purpose,
                    "owner_skill": owner_skill,
                    "reason": "owner_uninstalled" if owner_uninstalled else "foreign_generation",
                    "stale_session": str(entry.get("session_id") or ""),
                })
                survivors.append(entry)
                continue
            # enforce → fall through to the kill block below
        elif scope == "daemon" or (same_session and not task_owner_gone):
            survivors.append(entry)
            continue
        try:
            pgid = int(entry.get("pgid") or 0)
            if pgid > 0:
                kill_process_group_id(pgid, **({"exclude_pids": retained_roots} if retained_roots else {}))
            if pgid <= 0 or (retained_roots and pid_is_alive(pid)):
                from ouroboros.platform_layer import kill_pid_tree

                kill_pid_tree(pid, exclude_pids=retained_roots)
            reaped.append(pid)
            append_jsonl(drive_root / "logs" / "supervisor.jsonl", {
                "ts": utc_now_iso(),
                "type": "process_reaped",
                "pid": pid,
                "pgid": int(entry.get("pgid") or 0),
                "purpose": entry.get("purpose"),
                "scope": scope,
                "owner_task": owner_task,
                "stale_session": str(entry.get("session_id") or ""),
            })
        except Exception:
            log.warning("Failed to reap ledgered process %s", pid, exc_info=True)
            survivors.append(entry)
    _rewrite_ledger(drive_root, survivors, previous=previous)
    return reaped
