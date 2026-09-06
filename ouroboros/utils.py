"""Shared low-level utilities with no ouroboros.* imports."""

from __future__ import annotations

import contextlib
import datetime as _dt
import hashlib
import json
import logging
import math
import os
import pathlib
import re as _re
import subprocess
import threading
import time
import uuid
from collections import deque
from collections.abc import Iterator
from typing import Any, Callable, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

_log_sink: Optional[Callable[[Dict[str, Any]], None]] = None


def set_log_sink(fn: Optional[Callable[[Dict[str, Any]], None]]) -> None:
    global _log_sink
    _log_sink = fn

def utc_now_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat()


# Worker processes set OUROBOROS_IN_WORKER=1 before importing the agent stack.
WORKER_PROCESS_ENV = "OUROBOROS_IN_WORKER"


def in_worker_process() -> bool:
    """Return True inside a supervisor worker process.

    Worker processes disable system proxy resolution (``trust_env=False`` /
    ``ProxyHandler({})``) on every HTTP client they create. This is the central
    network-transport policy that keeps workers fork-safe: the macOS
    ``_scproxy.get_proxies`` -> ``SCDynamicStoreCopyProxies`` lookup crashes
    (SIGSEGV) on the child side of a multi-threaded fork. It is also a clean,
    proxy-free default for spawned workers on every platform.
    """
    return os.environ.get(WORKER_PROCESS_ENV) == "1"


def emit_log_event(
    event_queue: Any,
    payload: Dict[str, Any],
    *,
    blocking: bool = False,
    log_label: str = "live log",
) -> None:
    """Best-effort log_event publish; blocking preserves critical live logs."""
    if event_queue is None:
        return
    try:
        envelope = {"type": "log_event", "data": dict(payload)}
        if blocking:
            event_queue.put(envelope)
        else:
            event_queue.put_nowait(envelope)
    except Exception:
        log.debug("Failed to emit %s event", log_label, exc_info=True)


def emit_cognitive_operation_event(
    event_queue: Any,
    *,
    task_id: str,
    operation_id: str,
    phase: str,
    kind: str,
    task_attempt: Any = None,
    lease_until: Optional[float] = None,
    **payload: Any,
) -> None:
    """Publish one typed lifecycle fact for a live cognitive operation.

    This is deliberately a direct worker event, rather than a UI ``log_event``
    envelope.  The supervisor uses it only to spare its idle rail while the
    physical call is in flight; deadlines, budgets, cancellation and the
    absolute task ceiling remain independent.  Callers may omit ``lease_until``
    so the supervisor derives a bounded ceiling from the task it owns.
    """
    if event_queue is None:
        return
    try:
        event = {
            "type": "cognitive_operation",
            "task_id": str(task_id or ""),
            "operation_id": str(operation_id or ""),
            "phase": str(phase or ""),
            "kind": str(kind or ""),
            **payload,
        }
        if task_attempt is not None:
            event["task_attempt"] = task_attempt
        if lease_until is not None:
            event["lease_until"] = float(lease_until)
        if str(phase or "").strip().lower() == "started":
            event_queue.put(event)
        else:
            event_queue.put_nowait(event)
    except Exception:
        log.debug("Failed to emit cognitive operation event", exc_info=True)


def emit_main_llm_call_state_event(
    event_queue: Any,
    *,
    task_id: str,
    task_attempt: Any,
    llm_call_id: str,
    execution_id: str,
    round_id: str,
    call_attempt: int,
    phase: str,
) -> None:
    """Publish one typed main-LLM in-flight fact to the supervisor.

    This is a direct control-plane event, not a UI ``log_event``.  It has no
    elapsed-time expiry: the supervisor uses it only to spare the idle rail
    while the exact call is active; deadline, budget, cancellation, and the
    absolute task ceiling remain independent hard axes.
    """
    if event_queue is None:
        return
    try:
        event_queue.put({
            "type": "main_llm_call_state",
            "task_id": str(task_id or ""),
            "task_attempt": task_attempt,
            "llm_call_id": str(llm_call_id or ""),
            "execution_id": str(execution_id or ""),
            "round_id": str(round_id or ""),
            "call_attempt": int(call_attempt),
            "phase": str(phase or ""),
        })
    except Exception:
        log.debug("Failed to emit main LLM call state event", exc_info=True)

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def jsonl_generation_signature(path: pathlib.Path) -> dict:
    """Identity signature of a JSONL log generation: first-line hash + size.

    SSOT shared by the chat-log consolidation writer (consolidator) and the
    memory reader so a rotation/rewrite is detected identically on both sides.
    """
    path = pathlib.Path(path)
    if not path.exists():
        return {}
    try:
        stat = path.stat()
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            first = next((line.strip() for line in handle if line.strip()), "")
        return {
            "first_line_sha256": hashlib.sha256(first.encode("utf-8", errors="replace")).hexdigest(),
            "size": int(stat.st_size),
        }
    except OSError:
        return {}

def read_text(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8")


def resolve_path_allow_missing(path: pathlib.Path) -> Optional[pathlib.Path]:
    """Resolve a path while allowing a missing tail but rejecting unusable ancestry.

    Python 3.13 changed ``Path.resolve(strict=False)`` so a symlink loop no longer
    raises and instead returns a partially resolved path. Probe strict resolution
    first to keep loops, unreadable parents, and non-directory ancestors typed as
    unusable; only an ordinary missing component earns the allow-missing fallback.
    """
    try:
        return pathlib.Path(path).resolve(strict=True)
    except FileNotFoundError:
        try:
            return pathlib.Path(path).resolve(strict=False)
        except (OSError, ValueError, RuntimeError, TypeError):
            return None
    except (OSError, ValueError, RuntimeError, TypeError):
        return None


def write_text(path: pathlib.Path, content: str) -> None:
    # Full-file overwrite -> atomic (temp-sibling + os.replace), so a crash mid-write never
    # leaves a truncated file (G, v6.39). Strictly safer for every caller of this overwrite
    # helper; APPEND paths do their own thing and never route here.
    write_text_atomic(pathlib.Path(path), content)


# Windows sharing-violation retry bound: os.replace over a file that another
# thread/process holds open (CPython opens files WITHOUT FILE_SHARE_DELETE)
# raises PermissionError (winerror 5/32) even though the reader is transient —
# e.g. the skill-review heartbeat replacing review_job.json while a poller
# reads it. ~10 attempts with doubling backoff ≈ 0.5s total, then the original
# PermissionError is raised honestly.
_REPLACE_RETRY_ATTEMPTS = 10
_REPLACE_RETRY_INITIAL_DELAY_SEC = 0.002
_REPLACE_RETRY_MAX_DELAY_SEC = 0.1


def replace_atomic(
    src: pathlib.Path | str, dst: pathlib.Path | str, *,
    precondition: Callable[[], bool] | None = None,
) -> bool:
    """``os.replace`` with a bounded retry on Windows sharing violations.

    Retries ONLY on PermissionError, which POSIX rename(2) never raises for an
    open destination — so POSIX behavior is byte-identical (one syscall, no
    sleeps). After the bound is exhausted the last PermissionError propagates
    unchanged; every other exception propagates immediately.

    ``precondition`` is asked immediately before EVERY attempt, retries
    included: a proof taken before a refused attempt is stale by the next one
    (the monetary ledger swap re-proves lock ownership and its snapshot here,
    CPL4-C6). A ``False`` answer leaves ``dst`` untouched and returns False,
    ``src`` left for the caller to remove; True once replaced.
    """
    delay = _REPLACE_RETRY_INITIAL_DELAY_SEC
    for attempt in range(_REPLACE_RETRY_ATTEMPTS):
        if precondition is not None and not precondition():
            return False
        try:
            os.replace(src, dst)
            return True
        except PermissionError:
            if attempt == _REPLACE_RETRY_ATTEMPTS - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2, _REPLACE_RETRY_MAX_DELAY_SEC)


def _atomic_overwrite(path: pathlib.Path, write_temp: Callable[[pathlib.Path], None]) -> None:
    """Run ``write_temp`` against a sibling file, then atomically replace ``path``.

    A crash (SIGKILL / power loss) between the temp create and the replace leaves the
    EXISTING file fully intact — never a half-written/truncated file (G, v6.39). The temp
    name carries the ``.tmp.<pid>.<tid>.<uuid>`` atomic signature so the stale-temp sweep
    (`sweep_stale_temp_files`) reclaims an orphaned temp. Shared SSOT for every full-file
    overwrite.

    The existing file's permission bits are PRESERVED across the replace (os.replace
    creates a new inode, so without this a tracked executable script would lose its +x);
    a brand-new file defaults to the platform mode (0644 minus umask).

    Note: a symlink at ``path`` is REPLACED with a regular file (os.replace acts on the
    link, not its target). This is intentional and confinement-preserving — writing
    THROUGH a symlink could escape the caller's allowed root — so the write always lands
    inside ``path``'s directory rather than wherever a link points.

    Being that SSOT is also why the pytest live-data guard sits here rather than on each
    writer: every full-file overwrite in the tree replaces through this seam, so the next
    writer added is guarded by construction instead of by remembering (RES-14b)."""
    path = pathlib.Path(path)
    assert_test_data_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # 0o7777 keeps the special bits (setuid/setgid/sticky) too, not just rwx.
        existing_mode = os.stat(path).st_mode & 0o7777
    except OSError:
        existing_mode = None  # new file -> keep the platform default
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex[:8]}")
    try:
        write_temp(tmp)
        if existing_mode is not None:
            with contextlib.suppress(OSError):
                os.chmod(tmp, existing_mode)
        replace_atomic(tmp, path)
    except Exception:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise


def _write_fd_fully(fd: int, data: bytes, target: pathlib.Path) -> None:
    """``os.write`` until every byte lands.

    A single call may write SHORT (POSIX permits partial writes — signals,
    quota edges, some filesystems), and treating its return as done truncates
    the record silently: the atomic lanes would publish a half document behind
    a successful rename, and the append lane a TORN line behind a successful
    append. One loop for every durable writer in this module.
    """
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError(f"short write to {target}")
        view = view[written:]


def write_bytes_atomic(path: pathlib.Path, content: bytes, *, fsync: bool = False) -> None:
    """Atomically overwrite ``path`` with exact bytes."""

    def _write(tmp: pathlib.Path) -> None:
        if not fsync:
            tmp.write_bytes(content)
            return
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_BINARY", 0)
        fd = os.open(str(tmp), flags, 0o644)
        try:
            _write_fd_fully(fd, content, tmp)
            os.fsync(fd)
        finally:
            os.close(fd)

    _atomic_overwrite(path, _write)


def write_text_atomic(path: pathlib.Path, content: str, *, fsync: bool = False) -> None:
    """Atomically overwrite ``path`` with ``content`` encoded UTF-8, BYTE-EXACT.

    The file receives exactly ``content.encode("utf-8")`` on every platform:
    the newlines the caller wrote are the newlines on disk. This is the
    contract every caller in this tree needs — durable JSON state (through
    ``atomic_write_json``), receipts, run manifests, projection files, and the
    agent's own file writes/edits, which round-trip source text that Python
    reads back with universal newlines.

    It used to be "platform newline semantics": both lanes translated ``\\n``
    to ``\\r\\n`` on Windows (``Path.write_text`` in text mode, and ``os.open``
    without ``O_BINARY``). Nothing asked for that translation, while a
    byte-compared manifest, a hashed receipt and an LF source file the agent
    merely re-saved were all silently rewritten by it.
    """
    write_bytes_atomic(pathlib.Path(path), content.encode("utf-8"), fsync=fsync)


def atomic_write_json(path: pathlib.Path, payload: Any, *, trailing_newline: bool = False,
                      fsync: bool = False) -> None:
    """Atomically persist a JSON value (object or list) via a sibling temp file."""
    content = json.dumps(payload, ensure_ascii=False, indent=2)
    if trailing_newline:
        content += "\n"
    write_text_atomic(pathlib.Path(path), content, fsync=fsync)


def sweep_stale_temp_files(root: pathlib.Path, *, min_age_sec: float = 3600.0) -> int:
    """Remove orphaned atomic-write temp files left behind by a hard kill.

    ``atomic_write_json`` writes to a unique ``.{name}.tmp.<pid>.<tid>.<uuid>``
    sibling then ``os.replace``s it into place; a SIGKILL between create and
    rename can orphan the temp file (its try/finally cleanup never runs). This
    sweeps the tree for such temp files older than ``min_age_sec`` — the age
    guard avoids deleting a temp file from an in-flight write in another process.
    Returns the number removed. Best-effort: never raises.

    Only files whose suffix after the final ``.tmp.`` is the atomic signature
    (pid/tid/uuid → hex digits and dots) are reaped, so a legitimate user dotfile
    such as ``.config.tmp.backup`` is never deleted.

    The data-root ``tmp_scripts/`` fallback is in scope too (CPL4-C20):
    ``tools/shell.py`` unlinks its ``script_<uuid>.<ext>`` files in a
    ``finally``, so one that survived is a hard-kill orphan. Only the
    TOP-LEVEL fallback dir is swept here — task-drive copies die with their
    drive's own GC prune — and only at startup, when no script can be live.
    """
    root = pathlib.Path(root)
    if not root.is_dir():
        return 0
    hex_chars = set("0123456789abcdef.")
    removed = 0
    now = time.time()
    try:
        candidates = list(root.rglob(".*.tmp.*"))
        candidates.extend(root.glob("tmp_scripts/script_*"))
    except OSError:
        return 0
    fallback_scripts = root / "tmp_scripts"
    for tmp in candidates:
        try:
            if not tmp.is_file():
                continue
            if not (tmp.parent == fallback_scripts and tmp.name.startswith("script_")):
                # Require the post-".tmp." suffix to be the atomic signature
                # (hex/dot only) so we never delete an unrelated dotfile that
                # happens to match.
                suffix = tmp.name.rsplit(".tmp.", 1)
                if len(suffix) != 2 or not suffix[1] or set(suffix[1]) - hex_chars:
                    continue
            if now - tmp.stat().st_mtime < min_age_sec:
                continue
            tmp.unlink()
            removed += 1
        except OSError:
            continue
    return removed


def read_json_dict(path: pathlib.Path) -> Optional[Dict[str, Any]]:
    """Return a JSON object from ``path`` or ``None`` when absent/invalid."""
    path = pathlib.Path(path)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        log.warning("Failed to parse JSON file %s", path, exc_info=True)
        return None
    return data if isinstance(data, dict) else None


def update_json_locked(
    path: pathlib.Path,
    mutator: Any,
    *,
    timeout_sec: float = 4.0,
    stale_sec: float = 90.0,
    strict_existing_dict: bool = False,
    reject_existing_empty_dict: bool = False,
) -> Dict[str, Any]:
    """Locked read-modify-write of a durable JSON dict file.

    Acquires a sidecar ``<file>.lock``, re-reads the CURRENT on-disk JSON
    inside the lock (so the mutator always sees the latest state, closing the
    lost-update window of unlocked load→merge→write sequences), applies
    ``mutator(current) -> dict | None`` (``None`` aborts: the file is left
    unchanged and the pre-mutation snapshot is returned), atomically writes
    the result, and releases the lock.

    Raises ``TimeoutError`` on lock timeout — proceeding unlocked would
    silently reintroduce the exact lost-update class this helper removes.
    When ``strict_existing_dict`` is true, an existing malformed/non-object
    JSON file raises ``ValueError`` instead of being mistaken for a new file.
    ``reject_existing_empty_dict`` additionally distinguishes an absent store
    from an existing empty object for schema-bearing callers: only absence may
    initialize a new schema.
    """
    from ouroboros.platform_layer import (
        acquire_exclusive_file_lock,
        release_exclusive_file_lock,
    )

    path = pathlib.Path(path)
    lock_path = path.with_name(path.name + ".lock")
    lock_fd = acquire_exclusive_file_lock(
        lock_path, timeout_sec=timeout_sec, stale_sec=stale_sec
    )
    if lock_fd is None:
        raise TimeoutError(
            f"update_json_locked: could not acquire {lock_path} within {timeout_sec}s"
        )
    try:
        existed = path.exists()
        current = read_json_dict(path)
        if current is None:
            if strict_existing_dict and path.exists():
                raise ValueError(
                    "update_json_locked: existing JSON is malformed or is not an object"
                )
            current = {}
        if reject_existing_empty_dict and existed and not current:
            raise ValueError(
                "update_json_locked: existing empty JSON object has no schema"
            )
        updated = mutator(current)
        if updated is None:
            return current
        atomic_write_json(path, updated)
        return updated
    finally:
        release_exclusive_file_lock(lock_path, lock_fd)


def jsonl_append_lock_path(path: pathlib.Path) -> pathlib.Path:
    """Sidecar lock path shared by ``append_jsonl`` writers and log rotation."""
    path_hash = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:12]
    return path.parent / f".append_jsonl_{path_hash}.lock"


def assert_test_data_path(path: pathlib.Path) -> None:
    """Fail closed when pytest resolves a writer into the live data tree.

    Lives here (the no-ouroboros-imports leaf) so the whole durable-write
    surface guards the same roots from two seams: ``append_jsonl`` below for
    appends, and ``_atomic_overwrite`` above for every full-file replace
    (``supervisor.state.atomic_write_text`` included, which also calls it
    directly). The jsonl side was the unguarded half that let the issue #455
    supervisor.jsonl leak land silently. Outside pytest this is one env read.
    """
    if os.environ.get("OUROBOROS_PYTEST_ACTIVE") != "1":
        return
    if os.environ.get("OUROBOROS_ALLOW_LIVE_DATA_TESTS") == "1":
        return
    # ``expanduser``, not ``Path.home()``: this guards the tree of the operator
    # RUNNING pytest, which the environment fixes. A suite that redirects
    # ``Path.home`` into its own tmp dir is hermetic by construction, and reading
    # the patched attribute made those writes look live; redirecting ``$HOME``
    # itself still moves the guard, as the hermetic subprocess pin needs.
    roots = {pathlib.Path(os.path.expanduser("~")) / "Ouroboros" / "data"}
    configured = str(os.environ.get("OUROBOROS_TEST_LIVE_DATA_ROOT") or "").strip()
    if configured:
        roots.add(pathlib.Path(configured))
    target = pathlib.Path(path).resolve(strict=False)
    for root in roots:
        try:
            target.relative_to(root.resolve(strict=False))
        except ValueError:
            continue
        raise RuntimeError(f"PYTEST_LIVE_DATA_WRITE_BLOCKED: {target}")


def append_jsonl(
    path: pathlib.Path, obj: Dict[str, Any], *, ensure_record_boundary: bool = False,
    require_lock: bool = False,
) -> bool:
    """Append a JSON object as a line to a JSONL file (concurrent-safe).

    ``ensure_record_boundary`` repairs a missing final newline before this
    append; it is opt-in so high-volume event logs keep their established path.
    ``require_lock`` is reserved for authority streams that also have atomic
    whole-file reconciliation: unlike high-volume observational logs, they may
    not fall back to an unlocked append after lock timeout. Both lanes take the
    shared owner-aware lock primitive, so a live holder is never displaced.
    Returns ``True`` on successful write, ``False`` when all retries
    failed (which is also logged at WARNING). Important events
    (``task_done``, ``llm_round``, escalation messages) need that signal
    so the caller can fall back to an in-memory queue or stderr instead
    of pretending the write succeeded.
    """
    if not isinstance(path, pathlib.Path):
        raise TypeError(f"append_jsonl: path must be pathlib.Path, got {type(path).__name__}")
    assert_test_data_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(obj, ensure_ascii=False)
    data = (line + "\n").encode("utf-8")

    write_retries = 3
    retry_sleep_base_sec = 0.01

    from ouroboros.platform_layer import (
        acquire_exclusive_file_lock,
        release_exclusive_file_lock,
    )

    lock_path = jsonl_append_lock_path(path)
    lock_fd = None
    _written = False

    try:
        # ONE lock primitive for both lanes. The unlocked lane used to hand-roll
        # its own O_CREAT|O_EXCL + age-reclaim loop — the duplicate this module's
        # own contract tells feature code not to write — and the copy was NOT
        # owner-aware, so a high-volume appender could delete the lock of a LIVE
        # holder (the memory-journal compactor rewrites a journal under exactly
        # this lock). Owner-aware everywhere: a live holder is waited out and the
        # non-required lane then appends unlocked, exactly as before.
        lock_fd = acquire_exclusive_file_lock(
            lock_path,
            timeout_sec=2.0,
            stale_sec=10.0,
            poll_sec=0.01,
            owner_aware_stale=True,
        )

        if require_lock and lock_fd is None:
            log.warning("append_jsonl: required lock unavailable for %s", path)
            return False

        append_data = data
        if ensure_record_boundary:
            # A crashed receipt writer may leave its final object without a
            # separator. Preserve those bytes while keeping this append a new
            # record. Ordinary high-volume logs retain their existing fast path.
            try:
                if path.stat().st_size > 0:
                    with path.open("rb") as existing:
                        existing.seek(-1, os.SEEK_END)
                        if existing.read(1) != b"\n":
                            append_data = b"\n" + data
            except FileNotFoundError:
                pass
            except OSError:
                # Preserve historical behavior for unusual write-only files.
                append_data = data

        for attempt in range(write_retries):
            try:
                fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            except Exception:
                if attempt < write_retries - 1:
                    time.sleep(retry_sleep_base_sec * (2 ** attempt))
                continue
            try:
                # One bare ``os.write`` may land SHORT: trusting its return
                # published a TORN line as a successful append (the class the
                # atomic writers already fixed). Share their full-write loop.
                _write_fd_fully(fd, append_data, path)
                _written = True
                return True
            except Exception:
                # Bytes of this record may already be in the file; re-appending
                # the whole line would duplicate that prefix. Report the failure
                # (the caller owns the fallback) and let the next append's
                # ``ensure_record_boundary`` start a clean record — a partially
                # landed record is never retried whole.
                log.warning("append_jsonl: torn write to %s", path, exc_info=True)
                return False
            finally:
                os.close(fd)

        for attempt in range(write_retries):
            try:
                with path.open("a", encoding="utf-8") as f:
                    f.write(append_data.decode("utf-8"))
                _written = True
                return True
            except Exception:
                if attempt < write_retries - 1:
                    time.sleep(retry_sleep_base_sec * (2 ** attempt))
    except Exception:
        log.warning("append_jsonl: all write attempts failed for %s", path, exc_info=True)
    finally:
        release_exclusive_file_lock(lock_path, lock_fd)
        # Live-stream only runtime LOG files. chat.jsonl has its own live
        # channel (the chat frame family), and state/memory/receipt jsonl
        # stores are durable data, not a log feed — streaming them made every
        # ledger append a raw WS "log" frame (noise the Logs panel's backfill
        # never mirrors: it requests events/tools/progress/supervisor only).
        if (
            _written
            and _log_sink is not None
            and path.parent.name == "logs"
            and path.suffix == ".jsonl"
            and path.name != "chat.jsonl"
        ):
            try:
                _log_sink(obj)
            except Exception:
                pass
    if not _written:
        log.warning("append_jsonl: all write attempts failed for %s", path)
    return _written


def iter_jsonl_objects(
    path: pathlib.Path,
    max_entries: Optional[int] = None,
    tail_bytes: Optional[int] = None,
    dict_only: bool = True,
    gap_reasons: Optional[set[str]] = None,
    _handle: Any = None,  # Internal borrowed stream; its caller owns closing.
) -> Iterator[Any]:
    """Yield parseable JSONL entries; max_entries applies to raw tail lines."""
    path = pathlib.Path(path)
    if (max_entries is not None and max_entries <= 0) or (tail_bytes is not None and tail_bytes <= 0):
        return
    try:
        with (contextlib.nullcontext(_handle) if _handle is not None else path.open("rb")) as handle:
            if tail_bytes is not None:
                file_size = path.stat().st_size
                if file_size > tail_bytes:
                    start = file_size - tail_bytes
                    handle.seek(start - 1)
                    if handle.read(1) != b"\n":
                        handle.readline()
            if max_entries:
                # Keep one raw sentinel so callers can distinguish an exact
                # bounded tail from a suffix. Parsed-object counts cannot prove
                # that: blank, malformed, non-dict, or undecodable raw rows are
                # deliberately skipped below.
                lines = deque(handle, maxlen=max_entries + 1)
                if len(lines) > max_entries:
                    if gap_reasons is not None:
                        gap_reasons.add("max_entries_truncated")
                    lines.popleft()
            else:
                lines = handle
            for raw in lines:
                try:
                    line = raw.decode("utf-8")
                except UnicodeDecodeError:
                    if gap_reasons is not None:
                        gap_reasons.add("unreadable_bytes")
                    line = raw.decode("utf-8", errors="replace")
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    if gap_reasons is not None:
                        gap_reasons.add("malformed_jsonl")
                    continue
                if not dict_only or isinstance(entry, dict):
                    yield entry
                elif gap_reasons is not None:
                    gap_reasons.add("invalid_jsonl_row")
    except FileNotFoundError:
        if _handle is not None:
            raise
        return


class JsonlChainUnreadable(OSError):
    """A rotated JSONL chain could not be enumerated or opened IN FULL.

    Authority readers (money, custody) pass ``strict=True`` so an unreadable
    ``archive/`` directory or segment is a TYPED incomplete view instead of an
    empty one — the lenient default answers "this store never rotated" for
    both, which is how a permission error silently under-counts a ledger.
    Fail-soft readers (UI tails, context backfill, boot probes) keep the
    lenient default: a missing segment degrades their window, it decides
    nothing.
    """


def jsonl_archive_segments(
    path: pathlib.Path, *, strict: bool = False,
) -> List[pathlib.Path]:
    """Rotated archive segments for a ``logs/<name>.jsonl`` store, oldest first.

    ``rotate_jsonl_log_if_needed`` renames the live file to
    ``archive/<stem>_<ts>[_<n>].jsonl`` beside the ``logs/`` directory;
    lexicographic name order is chronological by construction (the rotator's
    ``_<n>`` collision suffix sorts after ``<ts>.jsonl``). A store that never
    rotated yields an empty list.

    Enumeration is an explicit ``scandir``: ``Path.glob`` SWALLOWS a
    ``PermissionError`` on the archive directory and yields nothing, so the
    former ``except OSError`` never saw the very failure that matters. With
    ``strict`` an unreadable directory raises :class:`JsonlChainUnreadable`;
    an absent one is still positively empty on both paths.
    """
    path = pathlib.Path(path)
    archive_dir = path.parent.parent / "archive"
    stem = path.name[:-len(".jsonl")] if path.name.endswith(".jsonl") else path.stem
    prefix = f"{stem}_"
    try:
        with os.scandir(archive_dir) as entries:
            found = [
                pathlib.Path(entry.path)
                for entry in entries
                if entry.name.startswith(prefix)
                and entry.name.endswith(".jsonl")
                and entry.is_file()
            ]
    except FileNotFoundError:
        return []
    except OSError as exc:
        if strict:
            raise JsonlChainUnreadable(
                f"cannot enumerate the rotated archive of {path}: {exc}"
            ) from exc
        return []
    return sorted(found)


@contextlib.contextmanager
def jsonl_chain_handles(path, *, strict=False, start_offset=None, snapshot=None) -> Iterator[List[Tuple[pathlib.Path, Any]]]:
    """Whole ordered chain by default; snapshot selects one segment at start_offset.

    Open live first and dedup its inode; snapshot caches metadata for one fixed
    source/EOF pass, never client paths. Consumed prefixes are stat'd once, not
    opened. Rebind the original live inode after rotation; keep at most two
    handles in snapshot mode. Every handle closes on exit, including errors.
    Missing discovery entries are benign; strict enumeration/stat/read or
    shortened required segments fail explicitly. Lenient reads skip failures.
    """
    from bisect import bisect_right

    path = pathlib.Path(path)
    state = snapshot if snapshot is not None else {}
    with contextlib.ExitStack() as stack:
        live_handle = None
        if not state:
            try:
                live_handle = stack.enter_context(path.open("rb"))
                live_stat = os.fstat(live_handle.fileno())
            except OSError as exc:
                if strict and not isinstance(exc, FileNotFoundError):
                    raise JsonlChainUnreadable(f"cannot open or stat {path}: {exc}") from exc
                live_stat = None
            entries, ends, total = [], [], 0
            for segment in [*jsonl_archive_segments(path, strict=strict), path]:
                try:
                    stat = live_stat if segment == path else segment.stat()
                except OSError as exc:
                    if strict and not isinstance(exc, FileNotFoundError):
                        raise JsonlChainUnreadable(f"cannot stat {segment}: {exc}") from exc
                    continue
                if stat is None or (segment != path and live_stat is not None
                        and (stat.st_dev, stat.st_ino) == (live_stat.st_dev, live_stat.st_ino)):
                    continue
                total += stat.st_size
                entries.append([segment, stat, segment == path])
                ends.append(total)
            state.update(path=path, entries=entries, ends=ends, total=total)
        if state["path"] != path:
            raise ValueError("chain pass belongs to a different source")
        entries, ends = state["entries"], state["ends"]
        if start_offset is not None and start_offset > state["total"]:
            raise JsonlChainUnreadable(f"shortened or missing chain at {path}")
        first = bisect_right(ends, start_offset) if start_offset is not None else 0
        stop = min(len(entries), first + 1) if snapshot is not None else len(entries)
        handles = []
        for index in range(first, stop):
            segment, expected, was_live = entries[index]
            try:
                if was_live:
                    handle = live_handle
                    if handle is None:
                        with contextlib.suppress(FileNotFoundError):
                            handle = stack.enter_context(segment.open("rb"))
                    identity = (expected.st_dev, expected.st_ino)
                    actual = os.fstat(handle.fileno()) if handle is not None else None
                    if actual is None or (actual.st_dev, actual.st_ino) != identity:
                        if handle is not None:
                            handle.close()
                        for candidate in jsonl_archive_segments(path, strict=strict):
                            with contextlib.suppress(FileNotFoundError):
                                stat = candidate.stat()
                                if (stat.st_dev, stat.st_ino) == identity:
                                    segment = entries[index][0] = candidate
                                    break
                        else:
                            raise JsonlChainUnreadable(f"original live generation missing at {path}")
                        handle = stack.enter_context(segment.open("rb"))
                else:
                    handle = stack.enter_context(segment.open("rb"))
                if start_offset is not None:
                    if os.fstat(handle.fileno()).st_size < expected.st_size:
                        raise JsonlChainUnreadable(f"shortened chain segment {segment}")
                    handle.seek(max(0, start_offset - (ends[index - 1] if index else 0)))
                handles.append((segment, handle))
            except OSError as exc:
                if strict:
                    raise JsonlChainUnreadable(f"cannot read {segment}: {exc}") from exc
        yield handles

def iter_jsonl_chain_objects(
    path: pathlib.Path,
    dict_only: bool = True,
) -> Iterator[Any]:
    """``iter_jsonl_objects`` across the rotated archive chain + live file.

    Full-history readers that must not lose early rows to rotation use this
    instead of a single-file read; bounded/tail readers keep their own
    windows.
    """
    with jsonl_chain_handles(pathlib.Path(path)) as handles:
        for segment, handle in handles:
            yield from iter_jsonl_objects(segment, dict_only=dict_only, _handle=handle)


def iter_llm_usage_events(
    path: pathlib.Path,
    *,
    max_entries: Optional[int] = None,
    tail_bytes: Optional[int] = None,
) -> Iterator[Dict[str, Any]]:
    yield from (event for event in iter_jsonl_objects(path, max_entries=max_entries, tail_bytes=tail_bytes)
                if event.get("type") == "llm_usage")


def llm_usage_cost(event: Dict[str, Any]) -> float:
    usage = event.get("usage")
    value = event.get("cost")
    if value is None and isinstance(usage, dict):
        value = usage.get("cost")
    try:
        cost = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return cost if math.isfinite(cost) else 0.0


# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------

def safe_relpath(p: str) -> str:
    """Normalize relative paths and reject traversal/NUL/control-char payloads."""
    if not isinstance(p, str):
        raise ValueError("Path must be a string.")
    for ch in p:
        if ch == "\x00":
            raise ValueError("Path contains NUL byte.")
        if ord(ch) < 0x20 and ch not in ("\t", "\n", "\r"):
            raise ValueError(
                f"Path contains control character U+{ord(ch):04X}."
            )
    p = p.replace("\\", "/").lstrip("/")
    if ".." in pathlib.PurePosixPath(p).parts:
        raise ValueError("Path traversal is not allowed.")
    return p

def truncate_for_log(s: str, max_chars: int = 4000) -> str:
    if len(s) <= max_chars:
        return s
    return s[: max_chars // 2] + "\n...\n" + s[-max_chars // 2:]


def clip_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    half = max(200, max_chars // 2)
    return text[:half] + "\n...(truncated)...\n" + text[-half:]


def short(s: Any, n: int = 120) -> str:
    t = str(s or "")
    return t[:n] + "..." if len(t) > n else t


def strip_markdown(text: str) -> str:
    """Best-effort markdown-to-plain-text projection.

    Shared SSOT for every plain-text projection of markdown-shaped output: the
    Project lifecycle excerpt producer and the read-side normalization of old
    persisted lifecycle rows. Live chat delivery does NOT strip — text rides
    verbatim and plain rendering is the client's decision (system rows without
    ``markdown: true``). Line-anchored patterns (headings, list bullets) only
    match while newlines still exist, so callers must strip BEFORE flattening
    whitespace.
    """
    text = _re.sub(r"```[^\n]*\n([\s\S]*?)```", r"\1", text)
    text = _re.sub(r"`([^`]+)`", r"\1", text)
    text = _re.sub(r"\*\*\*(.+?)\*\*\*", r"\1", text)
    text = _re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = _re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\1", text)
    text = _re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"\1", text)
    text = _re.sub(r"~~(.+?)~~", r"\1", text)
    text = _re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = _re.sub(r"^#{1,6}\s+", "", text, flags=_re.MULTILINE)
    text = _re.sub(r"^[\*\-]\s+", "• ", text, flags=_re.MULTILINE)
    text = text.replace("**", "").replace("__", "").replace("~~", "")
    text = text.replace("`", "")
    return text


def estimate_tokens(text: str) -> int:
    """Rough token estimate (chars/4 heuristic)."""
    return max(1, (len(str(text or "")) + 3) // 4)


def extract_trailing_json_object(
    text: str,
    *,
    duplicate_flag_keys: tuple = (),
) -> tuple[str, Optional[Dict[str, Any]], bool]:
    """Split ``text`` into ``(prose_prefix, trailing_dict_or_None, duplicate_flag)``.

    Decodes the last complete JSON object via ``raw_decode`` anchored at a
    ``{`` scanned from the end. The object counts as TRAILING only when
    nothing but whitespace and/or one closing markdown fence follows it — an
    object with prose after it is quoted material, never a directive, and the
    whole call returns ``(text, None, False)``. Whole-text JSON yields an
    empty prefix; a dangling opening fence is trimmed off the prefix so a
    fenced object behaves like a bare one. Any duplicate key invalidates the
    object (``None``, mirroring the loop's strict parser) while the boolean
    reports whether a key from ``duplicate_flag_keys`` was the duplicate, so
    protocol-repair intent survives the failed parse. Deliberately does NOT
    scan for any specific key: key-scanning extraction has misfired on
    answers that merely quote a protocol example (see
    review_verdict_extraction's whole-text rationale).
    """
    raw = str(text or "")
    # The trailing object, if any, ends at the last non-whitespace/non-fence
    # character, which must be ``}``. Establishing that up front is O(tail) and
    # keeps every ordinary code-bearing answer (prose, a trailing ``;``, a run
    # of ``{``) off the expensive path entirely — the earlier per-``{``
    # raw_decode walk was O(n * braces) and hit ~10s on a forced-finalization
    # rail carrying a large code answer.
    end_limit = len(raw)
    for _ in range(4):
        # Walk back over trailing whitespace WITHOUT slicing (a slice per
        # fence iteration would be O(n * fences)), then peel one trailing
        # markdown fence so a `{...}` closed by ``` or ```\n``` is still
        # trailing. More than a few stacked fences is pathological output —
        # degrade to prose rather than keep peeling.
        while end_limit > 0 and raw[end_limit - 1] in " \t\r\n":
            end_limit -= 1
        if raw.endswith("```", 0, end_limit):
            end_limit -= 3
            continue
        break
    if end_limit <= 0 or raw[end_limit - 1] != "}":
        return raw, None, False

    # A forward, string-aware pass locating the outermost object whose close
    # sits at end_limit. The primary pass starts at 0; an unmatched ``{`` or
    # ``"`` in the PROSE prefix corrupts its state (prose is not JSON), so on
    # failure the scan retries from a bounded set of later anchors — the last
    # few line-starting ``{`` positions, where a real trailing directive
    # begins. Each retry is O(tail-from-anchor); the anchor count is a small
    # constant, so the pathological many-brace answer stays fast.
    def _scan(start: int) -> int:
        depth = 0
        in_str = False
        esc = False
        cand_start = -1
        found = -1
        for i in range(start, end_limit):
            c = raw[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                if depth == 0:
                    cand_start = i
                depth += 1
            elif c == "}":
                if depth > 0:
                    depth -= 1
                    if depth == 0 and i == end_limit - 1:
                        found = cand_start
        return found

    obj_start = _scan(0)
    if obj_start < 0:
        anchors: List[int] = []
        pos = raw.rfind("\n{", 0, end_limit)
        while pos != -1 and len(anchors) < 8:
            anchors.append(pos + 1)
            pos = raw.rfind("\n{", 0, pos)
        for anchor in anchors:  # rightmost first: the innermost plausible start
            obj_start = _scan(anchor)
            if obj_start >= 0:
                break
    if obj_start < 0:
        return raw, None, False

    prefix = raw[:obj_start]
    trimmed = prefix.rstrip()
    cut = trimmed.rfind("\n")
    if trimmed[cut + 1:].startswith("```"):
        prefix = trimmed[:cut] if cut != -1 else ""
    duplicate_flagged = False
    duplicate_any = False

    def _unique_object(pairs: List[tuple]) -> Dict[str, Any]:
        nonlocal duplicate_flagged, duplicate_any
        result: Dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                duplicate_any = True
                if key in duplicate_flag_keys:
                    duplicate_flagged = True
                raise ValueError(f"duplicate key: {key}")
            result[key] = item
        return result

    try:
        parsed = json.loads(raw[obj_start:end_limit], object_pairs_hook=_unique_object)
    except (ValueError, RecursionError):
        # A deeply nested body raises RecursionError out of the C decoder; on
        # the last-resort finalization rail that must degrade, not propagate.
        parsed = None
    if not isinstance(parsed, dict):
        parsed = None
    if parsed is None and not duplicate_any:
        # Contract: non-directive text comes back WHOLE. A structurally
        # balanced tail that fails to parse (single-quoted pseudo-JSON, a bare
        # word) is prose, and returning the truncated prefix here would hand a
        # future caller a silent tail loss. ANY duplicate-key rejection keeps
        # the split: that tail IS a strict-parser-refused directive shape, and
        # protocol-repair intent needs the prefix/tail boundary.
        return raw, None, False
    return prefix, parsed, duplicate_flagged


def run_cmd(
    cmd: List[str],
    cwd: Optional[pathlib.Path] = None,
    timeout: Optional[float] = None,
) -> str:
    # Tool output is PARSED (git error signatures, porcelain text), so it must not
    # depend on the operator's locale: a Russian-locale git answers «метка … уже
    # существует» where the code and its tests match "already exists".
    env = {**os.environ, "LC_ALL": "C", "LANG": "C"}
    res = subprocess.run(
        cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True, env=env,
        timeout=timeout,
    )
    if res.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\n\nSTDOUT:\n{res.stdout}\n\nSTDERR:\n{res.stderr}"
        )
    return res.stdout.strip()

def get_git_info(repo_dir: pathlib.Path) -> tuple[str, str]:
    """Best-effort retrieval of (git_branch, git_sha)."""
    branch = ""
    sha = ""
    try:
        branch = run_cmd(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo_dir, timeout=2,
        )
    except Exception:
        log.debug("Failed to get git branch", exc_info=True)
    try:
        sha = run_cmd(["git", "rev-parse", "HEAD"], cwd=repo_dir, timeout=2)
    except Exception:
        log.debug("Failed to get git SHA", exc_info=True)
    return branch, sha

def sanitize_task_for_event(
    task: Dict[str, Any], drive_logs: pathlib.Path, threshold: int = 4000,
) -> Dict[str, Any]:
    """Sanitize task event logs while persisting full oversized text."""
    try:
        sanitized = task.copy()

        keys_to_strip = [k for k in sanitized.keys() if k.endswith("_base64")]
        for key in keys_to_strip:
            value = sanitized.pop(key)
            sanitized[f"{key}_present"] = True
            if isinstance(value, str):
                sanitized[f"{key}_len"] = len(value)

        # The origin text duplicates the canonical chat row; EVERY event copy —
        # top-level AND the metadata mirror a direct turn carries — is capped like
        # ``text`` (the durable full copy lives on the binding/record).
        origin_text = sanitized.get("origin_message_text")
        if isinstance(origin_text, str) and len(origin_text) > threshold:
            sanitized["origin_message_text"] = truncate_for_log(origin_text, threshold)
        metadata = sanitized.get("metadata")
        if isinstance(metadata, dict):
            nested = metadata.get("origin_message_text")
            if isinstance(nested, str) and len(nested) > threshold:
                metadata = dict(metadata)
                metadata["origin_message_text"] = truncate_for_log(nested, threshold)
                sanitized["metadata"] = metadata

        text = task.get("text")
        if not isinstance(text, str):
            return sanitized

        text_len = len(text)
        text_hash = sha256_text(text)
        sanitized["text_len"] = text_len
        sanitized["text_sha256"] = text_hash

        if text_len > threshold:
            sanitized["text"] = truncate_for_log(text, threshold)
            sanitized["text_truncated"] = True
            try:
                task_id = task.get("id")
                filename = f"task_{task_id}.txt" if task_id else f"task_{text_hash[:12]}.txt"
                full_path = drive_logs / "tasks" / filename
                write_text(full_path, text)
                sanitized["text_full_path"] = f"tasks/{filename}"
            except Exception:
                log.debug("Failed to persist full task text to Drive during sanitization", exc_info=True)
                pass
        else:
            sanitized["text_truncated"] = False

        return sanitized
    except Exception:
        return task


_SECRET_KEYS = frozenset([
    "token", "api_key", "apikey", "authorization", "secret", "password", "passwd", "passphrase",
])

_SECRET_PATTERNS = _re.compile(
    r'ghp_[A-Za-z0-9]{30,}'       # GitHub personal access token
    r'|gh[ousr]_[A-Za-z0-9]{30,}' # GitHub OAuth/user/server/refresh tokens
    r'|github_pat_[A-Za-z0-9_]{30,}'  # GitHub fine-grained personal access token
    r'|AKIA[0-9A-Z]{16}'          # AWS access key id
    r'|sk_live_[A-Za-z0-9]{24,}'  # Stripe live secret key
    r'|sk_test_[A-Za-z0-9]{24,}'  # Stripe test secret key
    r'|sk-ant-[A-Za-z0-9_\-]{30,}' # Anthropic API key
    r'|sk-or-[A-Za-z0-9\-]{30,}'  # OpenRouter API key
    r'|sk-proj-[A-Za-z0-9_\-]{30,}'  # OpenAI project key
    r'|sk-svcacct-[A-Za-z0-9_\-]{30,}'  # OpenAI service account key
    r'|sk-admin-[A-Za-z0-9_\-]{30,}'  # OpenAI admin key
    r'|gsk_[A-Za-z0-9]{30,}'      # Groq API key
    r'|sk-[A-Za-z0-9]{40,}'       # OpenAI API key
    r'|(?:(?<=bot)|\b)[0-9]{8,}:[A-Za-z0-9_\-]{20,}\b'  # Telegram bot token (digits:secret; matches the /bot<id>:<secret>/ URL form — no \b exists between 'bot' and a digit)
)
_SECRET_URL_CREDENTIAL_RE = _re.compile(
    r'(?i)\b(?:postgres|postgresql|mysql|mariadb|mongodb(?:\+srv)?|redis)://[^/\s:@]+:[^/\s@]+@'
)
_SECRET_LITERAL_FIELDS_RE = _re.compile(
    r'(?im)(?:^|[\s,{])["\']?([A-Za-z_][A-Za-z0-9_-]*)["\']?\s*[:=]\s*["\']([^"\']+)["\']'
)
_SECRET_BRACKET_LITERAL_RE = _re.compile(
    r'(?im)\[\s*["\']([A-Za-z_][A-Za-z0-9_-]*)["\']\s*\]\s*[:=]\s*["\']([^"\']+)["\']'
)
_SECRET_UNQUOTED_ASSIGNMENT_RE = _re.compile(
    r'(?im)^([A-Za-z_][A-Za-z0-9_-]*)\s*[:=]\s*([A-Za-z0-9_\-./+=]{16,})\s*$'
)
_SECRET_FALLBACK_LITERAL_RE = _re.compile(
    r'(?i)(?:os\.getenv|os\.environ\.get|settings\.get)\(\s*[\'"]([^\'"]+)[\'"][^)]*,\s*[\'"]([^\'"]+)[\'"]'
    r'|api\.get_settings\([^)]*\)\.get\(\s*[\'"]([^\'"]+)[\'"][^)]*,\s*[\'"]([^\'"]+)[\'"]'
    r'|process\.env\.([A-Z0-9_]+)\s*(?:\|\||\?\?)\s*[\'"]([^\'"]+)[\'"]'
)
_SECRET_KEY_NAME_RE = _re.compile(
    r'(?i)^(?:'
    r'token|access_token|refresh_token|auth_token|secret|secret_key|password|passwd|passphrase|authorization|'
    r'api[_-]?key|database_url|db_url|ouroboros_network_password|aws_access_key_id|aws_secret_access_key|stripe_secret_key|'
    r'[a-z0-9_-]+(?:[_-](?:token|secret|password|passwd|passphrase|api[_-]?key))'
    r')$'
)


def _secret_key_name(key: str) -> bool:
    raw = str(key or "").strip()
    snake = raw.lower() if raw.upper() == raw else _re.sub(r"(?<!^)(?=[A-Z])", "_", raw).lower()
    normalized = _re.sub(r"[^a-z0-9]+", "_", snake).strip("_")
    return bool(_SECRET_KEY_NAME_RE.match(normalized))


CREDENTIAL_HEADER_NAMES = frozenset({
    "authorization",
    "proxy-authorization",
    "api-key",
    "x-api-key",
    "x-goog-api-key",
    "anthropic-api-key",
    "openai-api-key",
    "cookie",
})


def is_secret_key_name(key: Any) -> bool:
    """Public credential-name classifier shared by durable identity builders."""
    return _secret_key_name(str(key or ""))


def is_credential_header_name(key: Any) -> bool:
    """Whether a header is authentication state rather than route capability."""
    normalized = str(key or "").strip().lower()
    return normalized in CREDENTIAL_HEADER_NAMES or is_secret_key_name(normalized)


def _secret_placeholder_value(value: str) -> bool:
    cleaned = str(value or "").strip().rstrip(",}]").strip().strip("'\"").strip()
    if not cleaned:
        return True
    lowered = cleaned.lower()
    if lowered in {"redacted", "***redacted***", "set_via_env", "set-in-settings", "changeme", "example"}:
        return True
    if lowered == "bearer":
        return True
    if lowered.startswith("bearer "):
        bearer_value = cleaned[7:].strip()
        if _secret_placeholder_value(bearer_value):
            return True
    if lowered in {"str", "string", "int", "float", "bool", "none", "null", "undefined"}:
        return True
    if lowered.startswith(("str ", "str|", "str |", "str)", "str):", "string ", "string|", "string |", "string)", "string):")):
        return True
    if lowered.startswith(("os.environ", "os.getenv", "process.env", "settings.", "api.get_settings")):
        for literal in _re.findall(r"['\"]([^'\"]*)['\"]", cleaned):
            if literal and not _secret_placeholder_value(literal) and not _secret_key_name(literal):
                return False
        return True
    if lowered.startswith(("f\"", "f'")) and "{" in cleaned:
        return True
    if "settings" in lowered and any(word in lowered for word in ("configure", "configured", "set", "enter", "provide")):
        return True
    if "+" in cleaned and any(part in lowered for part in ("token", "key", "secret", "settings", "env")):
        return True
    if cleaned.startswith(("<", "${", "{")) and (cleaned.endswith((">", "}")) or cleaned.count("{") == 1):
        return True
    if _re.match(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+\(?[^)]*\)?$", cleaned):
        return True
    if _re.match(r"^[A-Za-z_][A-Za-z0-9_]*\([^)]*\)$", cleaned):
        return True
    if _re.match(r"^[a-z_][a-z0-9_]*$", cleaned) and cleaned in {
        "password",
        "token",
        "secret",
        "api_key",
        "auth_header",
        "access_token",
        "refresh_token",
    }:
        return True
    if cleaned.isupper() and "_" in cleaned and not any(ch.isdigit() for ch in cleaned) and _secret_key_name(cleaned):
        return True
    return False


def sanitize_tool_result_for_log(result: str) -> str:
    """Redact potential secrets before a public or durable projection."""
    if not isinstance(result, str) or len(result) < 20:
        return result
    redacted = _SECRET_PATTERNS.sub("***REDACTED***", result)
    redacted = _SECRET_URL_CREDENTIAL_RE.sub(
        lambda match: match.group(0).split("://", 1)[0] + "://***REDACTED***@",
        redacted,
    )
    try:
        from ouroboros.observability import redact_projection

        projected = redact_projection(redacted).value
        if isinstance(projected, str):
            return projected
    except Exception:
        log.debug("Failed to run observability redactor for tool result", exc_info=True)
    return redacted


def sanitize_tool_args_for_log(
    fn_name: str, args: Dict[str, Any], threshold: int = 3000,
) -> Dict[str, Any]:
    """Sanitize tool arguments for logging: redact secrets, truncate large fields."""

    def _sanitize_value(key: str, value: Any, depth: int) -> Any:
        if depth > 3:
            return {"_depth_limit": True}
        if key.lower() in _SECRET_KEYS:
            return "*** REDACTED ***"
        if isinstance(value, str):
            if len(value) > threshold:
                return f"<TRUNCATED:{key}:{len(value)}ch:sha={sha256_text(value)[:12]}>"
            return value
        if isinstance(value, dict):
            return {k: _sanitize_value(k, v, depth + 1) for k, v in value.items()}
        if isinstance(value, list):
            sanitized = [_sanitize_value(key, item, depth + 1) for item in value[:50]]
            if len(value) > 50:
                sanitized.append({"_truncated": f"... {len(value) - 50} more items"})
            return sanitized
        try:
            json.dumps(value, ensure_ascii=False)
            return value
        except (TypeError, ValueError):
            log.debug("Failed to JSON serialize value in sanitize_tool_args", exc_info=True)
            return {"_repr": sanitize_tool_result_for_log(repr(value))}

    try:
        from ouroboros.observability import redact_projection

        # Redact with the complete nested key context before limiting depth,
        # list length or field size; a value alone loses DB_PASSWORD's meaning.
        projected = redact_projection(args).value
        return {k: _sanitize_value(k, v, 0) for k, v in projected.items()}
    except Exception:
        log.debug("Failed to sanitize tool arguments for logging", exc_info=True)
        return {"_error": "sanitization_failed"}



async def collect_evolution_metrics(repo_dir: str, data_dir: str | None = None) -> list[dict]:
    """Collect evolution metrics (LOC, prompt sizes, memory) for each git tag."""
    import asyncio
    import subprocess as sp

    def _parse_journal(filepath: str, size_key: str) -> list[tuple[_dt.datetime, float]]:
        """Parse a JSONL journal into sorted (datetime, size_kb) tuples."""
        entries: list[tuple[_dt.datetime, float]] = []
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        ts = _dt.datetime.fromisoformat(obj["ts"])
                        size_chars = obj.get(size_key, 0)
                        entries.append((ts, size_chars / 1024))
                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue
        except FileNotFoundError:
            pass
        entries.sort(key=lambda x: x[0])
        return entries

    identity_journal: list[tuple[_dt.datetime, float]] = []
    scratchpad_journal: list[tuple[_dt.datetime, float]] = []
    if data_dir:
        mem_path = os.path.join(data_dir, "memory")
        identity_journal = _parse_journal(
            os.path.join(mem_path, "identity_journal.jsonl"), "new_len"
        )
        scratchpad_journal = _parse_journal(
            os.path.join(mem_path, "scratchpad_journal.jsonl"), "content_len"
        )

    def _interpolate_from_journal(
        journal_entries: list[tuple[_dt.datetime, float]], tag_date: str,
    ) -> float:
        """Find the latest journal entry whose timestamp is <= tag_date."""
        if not journal_entries or not tag_date:
            return 0
        try:
            dt = _dt.datetime.fromisoformat(tag_date)
        except ValueError:
            return 0
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=_dt.timezone.utc)
        best = 0.0
        for entry_dt, size_kb in journal_entries:
            entry_dt_aware = entry_dt if entry_dt.tzinfo else entry_dt.replace(tzinfo=_dt.timezone.utc)
            if entry_dt_aware <= dt:
                best = size_kb
            else:
                break
        return round(best, 2)

    result = sp.run(
        ["git", "tag", "-l", "--sort=creatordate",
         "--format=%(refname:short)\t%(creatordate:iso-strict)"],
        cwd=repo_dir, capture_output=True, text=True
    )

    tags = []
    for line in result.stdout.strip().split(chr(10)):
        if not line.strip():
            continue
        parts = line.split(chr(9))
        tag = parts[0]
        date = parts[1] if len(parts) > 1 else ""
        tags.append((tag, date))

    cache_path: pathlib.Path | None = None
    cached_by_tag: dict[str, dict[str, Any]] = {}
    if data_dir:
        cache_path = pathlib.Path(data_dir) / "state" / "evolution_metrics_cache.json"
        try:
            cache_obj = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(cache_obj, dict) and cache_obj.get("schema") == 1 and isinstance(cache_obj.get("points"), dict):
                cached_by_tag = {
                    str(tag): point
                    for tag, point in cache_obj["points"].items()
                    if isinstance(point, dict)
                }
        except (OSError, json.JSONDecodeError):
            cached_by_tag = {}

    def _metrics_for_tag(tag: str, date: str) -> dict | None:
        ls_result = sp.run(
            ["git", "ls-tree", "-r", "--name-only", tag],
            cwd=repo_dir, capture_output=True, text=True
        )
        if ls_result.returncode != 0:
            return None

        files = ls_result.stdout.strip().split(chr(10))

        python_lines = 0
        for f in files:
            if f.endswith(".py"):
                show = sp.run(
                    ["git", "show", f"{tag}:{f}"],
                    cwd=repo_dir, capture_output=True, text=True,
                    encoding="utf-8", errors="replace",
                )
                if show.returncode == 0 and show.stdout:
                    python_lines += len(show.stdout.splitlines())

        def get_file_size_kb(filepath: str) -> float:
            show = sp.run(
                ["git", "show", f"{tag}:{filepath}"],
                cwd=repo_dir, capture_output=True, text=True,
                encoding="utf-8", errors="replace",
            )
            if show.returncode == 0 and show.stdout:
                return round(len(show.stdout.encode("utf-8")) / 1024, 2)
            return 0

        bible_kb = get_file_size_kb("BIBLE.md")
        system_kb = get_file_size_kb("prompts/SYSTEM.md")

        identity_kb = _interpolate_from_journal(identity_journal, date)
        scratchpad_kb = _interpolate_from_journal(scratchpad_journal, date)
        memory_kb = round(identity_kb + scratchpad_kb, 2)

        return {
            "tag": tag,
            "date": date,
            "code_lines": python_lines,
            "bible_kb": bible_kb,
            "system_kb": system_kb,
            "identity_kb": identity_kb,
            "scratchpad_kb": scratchpad_kb,
            "memory_kb": memory_kb,
        }

    cached_points: list[dict[str, Any]] = []
    missing_tags: list[tuple[str, str]] = []
    for tag, date in tags:
        cached = cached_by_tag.get(tag)
        if cached and cached.get("date") == date:
            cached_points.append(dict(cached))
        else:
            missing_tags.append((tag, date))

    loop = asyncio.get_running_loop()
    semaphore = asyncio.Semaphore(4)

    async def _bounded_metrics(tag: str, date: str) -> dict | None:
        async with semaphore:
            return await loop.run_in_executor(None, _metrics_for_tag, tag, date)

    results = await asyncio.gather(*[
        _bounded_metrics(tag, date)
        for tag, date in missing_tags
    ])

    new_points = [r for r in results if r is not None]
    points_by_tag = {point["tag"]: point for point in cached_points + new_points}
    points = [points_by_tag[tag] for tag, _date in tags if tag in points_by_tag]

    if cache_path and new_points:
        try:
            atomic_write_json(cache_path, {
                "schema": 1,
                "points": points_by_tag,
                "updated_at": utc_now_iso(),
            })
        except OSError:
            log.warning("Failed to write evolution metrics cache: %s", cache_path, exc_info=True)

    # Latest tag uses live identity+scratchpad sizes.
    if data_dir and points:
        mem_dir = os.path.join(data_dir, "memory")
        if os.path.isdir(mem_dir):
            def _file_kb(path: str) -> float:
                try:
                    return os.path.getsize(path) / 1024
                except OSError:
                    return 0

            identity_kb = _file_kb(os.path.join(mem_dir, "identity.md"))
            scratchpad_kb = _file_kb(os.path.join(mem_dir, "scratchpad.md"))

            points[-1]["identity_kb"] = round(identity_kb, 2)
            points[-1]["scratchpad_kb"] = round(scratchpad_kb, 2)
            points[-1]["memory_kb"] = round(identity_kb + scratchpad_kb, 2)

    return points

def truncate_review_artifact(text: str | None, limit: int = 4000) -> str:
    """Return a display-safe preview with explicit OMISSION NOTE, never silent clipping.

    A cut that saves fewer characters than its own omission marker is pure
    damage (the historical 60-char trace caps produced markers LONGER than the
    text they destroyed), so the marker length is the truncation floor: below
    it the text passes through whole (v6.70.0 owner-facing honesty invariant,
    docs/DEVELOPMENT.md "No silent truncation")."""
    text = str(text or "")
    if len(text) <= limit:
        return text
    marker = f"\n⚠️ OMISSION NOTE: truncated at {limit} chars; original length {len(text)}"
    if len(text) - limit <= len(marker):
        return text
    return text[:limit] + marker


def truncate_within_limit(text: str | None, limit: int) -> str:
    """A STRICT disclosed bound: the result NEVER exceeds ``limit`` characters,
    with the omission marker INSIDE the budget.

    For fields whose limit is a hard wire/prompt budget rather than a display
    preference. ``truncate_review_artifact``'s anti-waste floor deliberately lets
    a small overflow pass through whole and appends its marker BEYOND the limit —
    right for logs and previews, wrong for a bounded field: a 4050-char
    assignment field rode through at 4050 against a 4000 budget (sol #9 probe),
    and a 50k harness-authored header rode a "bounded" projection to 3x the tool
    budget. Disclosure survives — the marker states the cut and the original
    length — but the budget wins. A limit too small to hold the marker returns a
    bare prefix: at that scale the marker WOULD BE the content."""
    text = str(text or "")
    if len(text) <= limit:
        return text
    marker = f"\n⚠️ OMISSION NOTE: truncated at {limit} chars; original length {len(text)}"
    if limit <= len(marker):
        return text[:max(0, limit)]
    return text[: limit - len(marker)] + marker
