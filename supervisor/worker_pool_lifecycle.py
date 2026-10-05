"""Keeping the pool populated: readiness gate, pid records, reaping, respawn.

A spawned or respawned slot is installed unassignable (``reaping=True``) and opens
only when the child's own ``worker_ready`` row is observed, which is also where the
checkout SHA sampled after initialization is compared with the managed baseline;
a child alive but silent past the readiness window is
torn down and replaced through the same respawn path, a bounded number of times.
Its own entry-progress row permits one extension, still bounded from its birth.
The pids workers ran under are recorded durably so an orphan surviving a restart
can be reaped; a replaced worker's queue is closed under the lock before the new
one takes its slot.

The lifecycle serializer lives here too: it is a decorator, so it is applied at
import time and cannot be reached through a call-time handle. It is a primitive,
not pool state — nothing rebinds it — so the parent imports it back directly.
"""

from __future__ import annotations

import logging
import json
import os
import pathlib
import subprocess
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
from ouroboros.config import WORKER_READY_CEILING_SEC, WORKER_READY_MAX_ATTEMPTS, WORKER_READY_WINDOW_SEC
from ouroboros.review_owner_custody import reconcile_confirmed_dead_review_owners
from supervisor.state import append_jsonl
from ouroboros.outcomes import EXECUTION_INFRA_FAILED, terminal_outcome_axes
from ouroboros.utils import utc_now_iso
from supervisor.queue import _queue_lock




log = logging.getLogger(__name__)


def _pool():
    """The parent module, read at call time.

    The parent owns the rebindable module state and the members tests
    monkeypatch there; reading them through the module at each call keeps
    one binding, where a from-import would freeze the value this leaf saw
    at import time (the owner-approved D18/D33 mechanical exception).
    """
    from supervisor import workers

    return workers


_WORKER_LIFECYCLE_LOCK = threading.RLock()
_WORKER_SHA_LOOKUP_TIMEOUT_SEC = 2.0  # Local diagnostic I/O, like utils.get_git_info.


def _serialized_worker_lifecycle(fn):
    def wrapped(*args, **kwargs):
        with _WORKER_LIFECYCLE_LOCK:
            return fn(*args, **kwargs)

    return wrapped


def _recorded_cancel_fields(task_id: str, *, stop_source: str = "", reason: str = "") -> dict:
    """Keep the recorded intent's cause; only an absent intent permits a door fallback.

    An unreadable authority supplies neither a recorded cause nor proof of absence.
    """
    try:
        from ouroboros.cancel_intents import active_intent
        from supervisor.cancel_publication import _intent_outcome_fields

        intent = active_intent(_pool().DRIVE_ROOT, task_id, strict=True)
        if intent:
            return _intent_outcome_fields(intent)
        return {"cancel_origin": {"source": stop_source, "reason": reason}} if stop_source else {}
    except Exception:
        log.warning("Stop cause of %s is unreadable; none recorded", task_id, exc_info=True)
        return {}


def _write_failure_result(
    task_id: str,
    reason: str = "Worker process crashed (crash storm). Task was not completed.",
    status: str = "",
    stop_source: str = "",
) -> str:
    """Write failure result for a crashed/orphaned task.

    Returns the FINAL persisted status: if the task already reached a terminal
    state, the monotonic guard preserves it and that existing status is returned
    (so the UI event matches disk); otherwise the written failure status.

    ``stop_source`` is the typed cause a known stop door passes (the owner's
    Restart, a graceful server shutdown). It lands as ``cancel_origin`` — the
    field a settled cancel intent records — and never outranks an earlier
    stop: an active intent for the task is the cause as recorded (an owner
    Stop stays a Stop), only a task no intent names takes the door's own
    cause, and an unreadable intent store records none. A cancelled result
    without a door source still retains its recorded intent (including Panic).
    """
    if not task_id:
        return ""
    try:
        from ouroboros.task_results import (
            STATUS_FAILED, STATUS_COMPLETED, STATUS_REJECTED_DUPLICATE,
            STATUS_CANCELLED, load_task_result, write_task_result,
        )
        # STATUS_INTERRUPTED is not final; it is written before requeue.
        _FINAL_STATUSES = {STATUS_COMPLETED, STATUS_FAILED, STATUS_REJECTED_DUPLICATE, STATUS_CANCELLED}
        existing = load_task_result(_pool().DRIVE_ROOT, task_id, strict=True)
        if existing and existing.get("status") in _FINAL_STATUSES:
            return str(existing.get("status") or "")
        final_status = status or STATUS_FAILED
        # Reconstruct from durable llm_usage so an abnormally-finalized task does
        # not record zero cost/rounds (understating per-task + campaign metrics).
        f_cost_fields = _pool().reconstruct_task_cost(str(task_id), fields=True)
        cause = (_recorded_cancel_fields(task_id, stop_source=stop_source, reason=reason)
                 if stop_source or final_status == STATUS_CANCELLED else {})
        stored = write_task_result(
            _pool().DRIVE_ROOT,
            task_id,
            final_status,
            strict_existing_dict=True,
            result=reason,
            reason_code="worker_terminal_failure" if final_status == STATUS_FAILED else str(final_status or ""),
            outcome_axes=terminal_outcome_axes(
                lifecycle=final_status,
                execution=EXECUTION_INFRA_FAILED if final_status == STATUS_FAILED else str(final_status or ""),
                reason_code="worker_terminal_failure" if final_status == STATUS_FAILED else str(final_status or ""),
                review_trigger="worker_terminal",
            ),
            **f_cost_fields,
            **cause,
        )
        persisted_status = str((stored or {}).get("status") or "").strip()
        if (
            not isinstance(stored, dict)
            or str(stored.get("task_id") or "") != str(task_id)
            or not persisted_status
        ):
            raise ValueError(
                f"failure result writer returned invalid durable identity for {task_id}"
            )
        return persisted_status
    except Exception:
        log.warning("Failed to write failure result for task %s", task_id, exc_info=True)
        raise


def events_log_cursor() -> Tuple[int, int, int]:
    """The live event log's ``(size, device, inode)`` — a cursor, not an offset.

    A byte offset alone cannot tell "the file I measured" from "a different
    file at the same path": ``_first_worker_event_since`` must know WHICH file
    the offset belongs to (audit #14-6c). Missing log = a zeroed cursor that
    reads from the start.
    """
    path = _pool().DRIVE_ROOT / "logs" / "events.jsonl"
    try:
        stat = path.stat()
        return int(stat.st_size), int(stat.st_dev), int(stat.st_ino)
    except Exception:
        return 0, 0, 0  # cannot measure = zeroed cursor; the reader reads from the start


def _worker_events_since(cursor: Tuple[int, int, int], event_type: str) -> List[Dict[str, Any]]:
    """Every event of one worker lifecycle type after a cursor, in log order.

    The event log rotates (CPL4-C1), so the file the cursor was taken from may
    now BE an archive segment. Rotation is detected by IDENTITY, not by size:
    the previous test ("the live file is smaller than my offset") missed every
    rotation where the new live file had already grown past the old offset —
    exactly what happens under a busy supervisor — and then read the wrong
    file's bytes from a meaningless offset. When the identity moved, the
    matching segment is read from the SAME offset (the continuation is exact);
    a cursor whose file is gone entirely falls back to the newest segment
    whole.

    A missing live file is an EMPTY read, never an error: the log may not have
    been written yet (a zeroed cursor), a poll may land in the instant between
    the rotator's ``os.replace`` and its ``touch``, or the file may have been
    removed by hand. Nothing durable is lost to that read — a row appended
    before the rename sits in the segment the cursor's identity finds, one
    appended after the touch sits in the new live file, and the next poll sees
    both.
    """
    offset_bytes, dev, ino = cursor
    path = _pool().DRIVE_ROOT / "logs" / "events.jsonl"
    chunks: list[str] = []
    try:
        with path.open("rb") as f:
            stat = os.fstat(f.fileno())
            rotated = (dev, ino) != (0, 0) and (int(stat.st_dev), int(stat.st_ino)) != (dev, ino)
            if not rotated and not 0 <= offset_bytes <= stat.st_size:
                rotated = True  # same file, truncated under the cursor
            f.seek(0 if rotated else offset_bytes)
            chunks.append(f.read().decode("utf-8", errors="replace"))
        if rotated:
            from ouroboros.utils import jsonl_archive_segments

            segments = jsonl_archive_segments(path)
            carried = None
            for segment in reversed(segments):
                try:
                    seg_stat = segment.stat()
                except OSError:
                    continue
                if (int(seg_stat.st_dev), int(seg_stat.st_ino)) == (dev, ino):
                    carried = segment
                    break
            if carried is not None:
                with carried.open("rb") as sf:
                    sf.seek(min(offset_bytes, carried.stat().st_size))
                    chunks.insert(0, sf.read().decode("utf-8", errors="replace"))
            elif segments:
                chunks.insert(0, segments[-1].read_bytes().decode("utf-8", errors="replace"))
    except FileNotFoundError:
        return []  # not written yet, the rotation gap, or removed by hand: an empty read
    except Exception:
        log.debug("Suppressed exception", exc_info=True)
        return []

    found: List[Dict[str, Any]] = []
    for line in "".join(chunks).splitlines():
        raw = line.strip()
        if not raw:
            continue
        try:
            evt = json.loads(raw)
        except Exception:
            log.debug("Suppressed exception in loop", exc_info=True)
            continue
        if isinstance(evt, dict) and str(evt.get("type") or "") == event_type:
            found.append(evt)
    return found


def _first_worker_event_since(
    cursor: Tuple[int, int, int], event_type: str = "worker_boot"
) -> Optional[Dict[str, Any]]:
    """The first event of one worker lifecycle type after a cursor, or None."""
    rows = _worker_events_since(cursor, event_type)
    return rows[0] if rows else None


def _slot_pid(slot: Any) -> int:
    try:
        return int(getattr(slot.proc, "pid", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _supervisor_row(row: Dict[str, Any]) -> None:
    append_jsonl(_pool().DRIVE_ROOT / "logs" / "supervisor.jsonl", {"ts": utc_now_iso(), **row})


def _verify_worker_sha_after_spawn(
    slots: Dict[int, Any], events_cursor: Tuple[int, int, int], attempt: int = 1,
    spawned_at: float = 0.0,
) -> None:
    """Hold freshly spawned slots unassignable until each child confirms ready.

    The ONE readiness seam for both spawn paths: ``spawn_workers`` and
    ``respawn_worker`` install a slot with ``reaping=True`` (the marker the
    assignment path, the crash detector and the reaper already honour) and hand
    it here. A slot opens only when the child's own ``worker_ready`` row
    (supervisor/worker_process.py) names its pid. That row samples the checkout
    after initialization; comparison with ``current_sha`` is diagnostic only,
    not loaded-code attestation, authorship or review evidence. A child that is alive
    but silent past ``WORKER_READY_WINDOW_SEC`` is torn down and replaced;
    its own ``worker_starting`` row permits one extension to
    ``WORKER_READY_CEILING_SEC`` from birth. Replacement still runs
    through ``respawn_worker`` — at most ``WORKER_READY_MAX_ATTEMPTS``
    consecutive times for one slot, then the slot is parked and reported. A
    child that DIED during boot is released to the crash detector, which
    already owns process death (``worker_dead_detected``, retry, the
    crash-storm fence). This is a contract distinct from process liveness and
    from the task idle rail: a deadlocked child is alive and holds no task.
    ``spawned_at`` is the instant before ``proc.start()``: the window and the
    reported wait count from the child's birth, not from this thread's start.

    The watcher owns capacity, so its body is guarded: it runs in a bare
    daemon thread, and a thread that died of an unexpected exception (the
    event reader, ``load_state``, a teardown) would leave every slot of the
    wave ``reaping`` with nothing left to open or replace it — a pool parked
    until restart. Instead the slots still booting are released to the crash
    detector (``worker_ready_released``, ``reason=watcher_error``), which is
    the pre-readiness behaviour: assignable slots, death owned by the detector.
    """
    pending = dict(slots)
    started = float(spawned_at or 0.0) or time.time()
    try:
        _watch_booting_slots(pending, events_cursor, attempt, started)
    except Exception as exc:
        log.exception(
            "Worker readiness watcher failed (attempt %d); releasing %d booting slot(s) to the crash detector",
            attempt, len(pending),
        )
        for wid, slot in list(pending.items()):
            try:
                _release_booting_slot(
                    wid, slot, started, attempt, "watcher_error",
                    error_type=type(exc).__name__, error=str(exc)[:400],
                )
            except Exception:
                log.exception("Release of booting worker slot %d failed", wid)


def _watch_booting_slots(
    pending: Dict[int, Any], events_cursor: Tuple[int, int, int], attempt: int, started: float,
) -> None:
    """Poll for each pending slot's ``worker_ready`` row; ``pending`` keeps only the unresolved slots."""
    st = _pool().load_state()
    expected_sha = str(st.get("current_sha") or "").strip()
    owner_chat_id = int(st.get("owner_chat_id") or 0)
    if not expected_sha:
        _supervisor_row({"type": "worker_sha_verify_skipped", "reason": "missing_current_sha"})
    deadline = started + max(float(WORKER_READY_WINDOW_SEC), 1.0)
    ceiling = started + max(float(WORKER_READY_CEILING_SEC), float(WORKER_READY_WINDOW_SEC), 1.0)
    extended = False
    while pending:
        ready_rows: Dict[int, Dict[str, Any]] = {}
        for row in _worker_events_since(events_cursor, "worker_ready"):
            try:
                ready_rows.setdefault(int(row.get("pid") or 0), row)
            except (TypeError, ValueError):
                continue
        for wid, slot in list(pending.items()):
            row = ready_rows.get(_slot_pid(slot))
            if row is not None:
                _open_ready_slot(wid, slot, row, expected_sha, owner_chat_id, started, attempt)
                pending.pop(wid)
            elif not slot.proc.is_alive():
                _release_booting_slot(
                    wid, slot, started, attempt, "died_during_boot",
                    exitcode=getattr(slot.proc, "exitcode", None),
                )
                pending.pop(wid)
        if not pending:
            break
        if time.time() >= deadline:
            if extended or time.time() >= ceiling:
                break
            # The cursor belongs to this spawn attempt; foreign or older progress
            # cannot buy capacity for a silent slot in the same wave.
            starting_pids = {str(row.get("pid") or "")
                             for row in _worker_events_since(events_cursor, "worker_starting")}
            for wid, slot in list(pending.items()):
                if str(_slot_pid(slot)) not in starting_pids:
                    _replace_unready_slot(wid, slot, owner_chat_id, started, attempt)
                    pending.pop(wid)
            if not pending:
                break
            extended = True
            deadline = ceiling
            _supervisor_row({
                "type": "worker_ready_window_extended", "attempt": attempt,
                "worker_ids": sorted(pending), "window_sec": float(WORKER_READY_WINDOW_SEC),
                "ceiling_sec": float(WORKER_READY_CEILING_SEC),
            })
        time.sleep(0.25)
    for wid, slot in list(pending.items()):
        _replace_unready_slot(wid, slot, owner_chat_id, started, attempt,
                              window_sec=float(WORKER_READY_CEILING_SEC if extended else WORKER_READY_WINDOW_SEC))
        pending.pop(wid)


def _worker_sha_relation(expected_sha: str, observed_sha: str) -> Dict[str, Any]:
    """Read ancestry without fetching or the mutating git index-repair path.

    Only a negative answer needs a second read: shallow history cannot prove
    non-ancestry, even when both commit objects are present locally.
    """
    if not expected_sha:
        return {"relation": "not_applicable"}
    if not observed_sha:
        return {"relation": "unavailable", "relation_error": "missing_observed_sha"}
    if observed_sha == expected_sha:
        return {"relation": "equal"}
    options = dict(cwd=_pool().REPO_DIR, capture_output=True, text=True,
                   timeout=_WORKER_SHA_LOOKUP_TIMEOUT_SEC,
                   env={**os.environ, "GIT_NO_LAZY_FETCH": "1"})
    try:
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", expected_sha, observed_sha],
            **options,
        )
        if result.returncode == 0:
            return {"relation": "descendant"}
        if result.returncode == 1:
            shallow = subprocess.run(["git", "rev-parse", "--is-shallow-repository"], **options)
            if shallow.returncode == 0 and shallow.stdout.strip() == "false":
                # An older ancestor and a different branch both belong here.
                return {"relation": "non_descendant"}
            return {"relation": "unavailable", "relation_returncode": shallow.returncode,
                    "relation_error": ("shallow_history" if shallow.stdout.strip() == "true"
                                       else "history_completeness_unavailable")}
    except Exception as exc:
        return {"relation": "unavailable", "relation_error": str(exc),
                "relation_error_type": type(exc).__name__}
    return {"relation": "unavailable", "relation_returncode": result.returncode,
            "relation_error": result.stderr.strip()}


def _open_ready_slot(
    wid: int, slot: Any, row: Dict[str, Any], expected_sha: str, owner_chat_id: int,
    started: float, attempt: int,
) -> None:
    """Open the ready slot before the diagnostic checkout comparison."""
    with _queue_lock:
        owned = (_pool().WORKERS.get(wid) is slot
                 and not getattr(slot, "readiness_exhausted", False))
        if owned:
            slot.reaping = False
    observed_sha = str(row.get("git_sha") or "").strip()
    ok = (bool(observed_sha) and observed_sha == expected_sha) if expected_sha else None
    wait_sec = round(time.time() - started, 2)
    relation = _worker_sha_relation(expected_sha, observed_sha)
    _supervisor_row({
        "type": "worker_sha_verify",
        "ok": ok,
        "expected_sha": expected_sha,
        "observed_sha": observed_sha,
        "worker_pid": row.get("pid"),
        "worker_id": wid,
        "wait_sec": wait_sec,
        "attempt": attempt,
        "slot_opened": owned,
        **relation,
    })
    if owner_chat_id and relation["relation"] in {"non_descendant", "unavailable"}:
        observation = f"baseline {expected_sha[:8]}, observed {(observed_sha or 'unknown')[:8]}"
        detail = ("is not a descendant of the recorded baseline"
                  if relation["relation"] == "non_descendant"
                  else "could not be compared with the recorded baseline")
        _pool().send_with_budget(
            owner_chat_id,
            f"⚠️ Worker checkout sampled after initialization {detail}: {observation}.",
            role="system", system_type="worker_readiness_notice")


def _release_booting_slot(
    wid: int, slot: Any, started: float, attempt: int, reason: str, **detail: Any,
) -> None:
    """Hand a booting slot to the crash detector: ``reaping`` cleared if it is still ours, one typed row.

    ``died_during_boot`` carries the exit code; ``watcher_error`` carries the error type and message.
    """
    with _queue_lock:
        owned = (_pool().WORKERS.get(wid) is slot
                 and not getattr(slot, "readiness_exhausted", False))
        if owned:
            slot.reaping = False
    _supervisor_row({
        "type": "worker_ready_released",
        "reason": reason,
        "worker_id": wid,
        "pid": _slot_pid(slot),
        "waited_sec": round(time.time() - started, 2),
        "attempt": attempt,
        "slot_released": owned,
        **detail,
    })


def kill_worker_tree(pid: int, *, keep_services: bool = False, panic_process=None):
    """The ONE worker process-tree kill, for every teardown and backstop.

    A worker's tree is not the worker's property: the installation's daemon
    roots (``daemon``-scope custody rows — the shared Claudexor daemon when a
    task worker was the first to need it) outlive every worker and server
    generation, so they and their descendants are spared. ``keep_services``
    additionally spares the deliberately kept session services a verifier
    still needs when ONE task is cancelled or timed out; a generation change
    ends those services with the generation, so the pool paths leave it off.
    The platform helper applies the same retained-subtree contract on every OS.
    """
    from ouroboros.platform_layer import kill_pid_tree, request_process_tree_kill
    if panic_process is not None:
        # Explicit Panic has already requested installation-owned daemon stops.
        # The supplied multiprocessing handle proves ownership without reading
        # custody under the pool/queue locks. Ordinary teardown is unchanged.
        if panic_process.pid != pid:
            raise ValueError("Panic worker identity mismatch")
        return request_process_tree_kill(panic_process)
    from supervisor import queue as _q

    spared = _q._retained_daemon_pids()
    if keep_services:
        spared |= _q._kept_service_pids()
    kill_pid_tree(pid, exclude_pids=spared)


@_serialized_worker_lifecycle
def _replace_unready_slot(wid: int, slot: Any, owner_chat_id: int, started: float, attempt: int,
                         window_sec: float = 0.0) -> None:
    """No worker_ready inside the window: tear the child down and replace the slot, bounded.

    One lifecycle transaction (lifecycle -> queue lock order, like every pool
    start/kill/respawn): the identity check, the teardown and the nested
    ``respawn_worker`` cannot be interleaved by a generation change
    (``kill_workers`` -> ``spawn_workers``), which would otherwise install a
    fresh live slot at ``wid`` in the gap for this stale watcher's respawn to
    evict — its process never terminated, gone from WORKERS.
    """
    with _queue_lock:
        if _pool().WORKERS.get(wid) is not slot:
            return  # a pool restart already replaced or cleared this slot
    pid = _slot_pid(slot)
    action = "respawn" if attempt < WORKER_READY_MAX_ATTEMPTS else "parked"
    _supervisor_row({
        "type": "worker_ready_timeout",
        "reason": "no_worker_ready",
        "worker_id": wid,
        "pid": pid,
        "waited_sec": round(time.time() - started, 2),
        "window_sec": float(window_sec or WORKER_READY_WINDOW_SEC),
        "attempt": attempt,
        "max_attempts": int(WORKER_READY_MAX_ATTEMPTS),
        "action": action,
    })
    if pid:
        kill_worker_tree(pid)
    try:
        slot.proc.join(timeout=2)
    except Exception:
        log.debug("Join of an unready worker failed", exc_info=True)
    if action == "respawn":
        try:
            respawn_worker(wid, ready_attempt=attempt + 1)
        except Exception:
            log.warning("Respawn of unready worker %d failed; clearing reaping for recovery", wid, exc_info=True)
            with _queue_lock:
                if _pool().WORKERS.get(wid) is slot:
                    slot.reaping = False  # the crash detector recovers the dead slot
        return
    log.error("Worker %d never confirmed ready in %d attempts; slot parked until restart", wid, attempt)
    with _queue_lock:
        if _pool().WORKERS.get(wid) is not slot:
            return
        slot.readiness_exhausted = True
        slot.reaping = True
    if owner_chat_id:
        _pool().send_with_budget(
            owner_chat_id,
            f"⚠️ Worker slot {wid} never confirmed ready in {attempt} attempts "
            f"(waited {time.time() - started:.0f}s on the last attempt); the slot is parked. Use /restart.",
            role="system", system_type="worker_readiness_notice")
    _pool().disable_exhausted_worker_pool()


def _worker_pids_path() -> pathlib.Path:
    return _pool().DRIVE_ROOT / "state" / _pool()._WORKER_PIDS_FILENAME


def _record_worker_pids() -> None:
    """Persist current worker PIDs so a later server instance can reap any that
    survive an abrupt restart. Workers run in their own ``os.setsid`` session, so
    when the parent server dies they are reparented to init and outlive it."""
    try:
        from ouroboros.utils import atomic_write_json
        recs = [{"pid": int(w.proc.pid)} for w in _pool().WORKERS.values() if w.proc.pid]
        atomic_write_json(
            _worker_pids_path(),
            {"server_pid": os.getpid(), "ts": utc_now_iso(), "workers": recs},
            trailing_newline=True,
        )
    except Exception:
        log.debug("Failed to record worker pids", exc_info=True)
    # Write-through into the custody ledger (SSOT for the generation reaper);
    # worker_pids.json stays as the legacy session-leader reap path.
    try:
        from ouroboros.process_custody import record_process

        for w in _pool().WORKERS.values():
            if w.proc.pid:
                was_alive = w.proc.is_alive()
                record = record_process(
                    _pool().DRIVE_ROOT,
                    pid=int(w.proc.pid),
                    cmd=f"ouroboros-worker-{w.wid}",
                    purpose=f"worker:{w.wid}",
                    scope="session",
                )
                if was_alive and w.proc.is_alive() and not getattr(w, "process_birth", ""):
                    fingerprint = record["fingerprint"]
                    w.process_birth = fingerprint.get("start_time_boot") or fingerprint.get("start_time") or ""
    except Exception:
        log.debug("Failed to ledger worker pids", exc_info=True)


def reap_orphaned_workers() -> int:
    """Kill leftover worker process groups left by a PRIOR server instance.

    ``kill_workers`` only walks the in-memory ``WORKERS`` dict, so workers
    orphaned by an abrupt restart (reparented to init, ~one Python interpreter
    each) were never reaped and accumulated across restarts. On startup we read
    the prior pid record and force-kill any that are still alive AND verifiably
    ours — cmdline matches this interpreter/multiprocessing and the process is
    its own session leader (``pgid == pid``) — which guards against PID reuse and
    bounds the group kill to the worker's own setsid session."""
    try:
        from ouroboros.utils import read_json_dict
        from ouroboros.platform_layer import (
            kill_process_group_id,
            process_command,
            process_group_id,
        )
        from supervisor.queue import _retained_daemon_pids
    except Exception:
        return 0
    data = read_json_dict(_worker_pids_path()) or {}
    prior = data.get("workers") or []
    if not isinstance(prior, list) or not prior:
        return 0
    current = {w.proc.pid for w in _pool().WORKERS.values() if w.proc.pid}
    killed: List[int] = []
    for rec in prior:
        try:
            pid = int((rec or {}).get("pid") or 0)
        except (TypeError, ValueError):
            continue
        if not pid or pid in current or pid == os.getpid():
            continue
        cmd = process_command(pid)
        if not cmd:
            continue  # already dead
        if sys.executable not in cmd and "multiprocessing" not in cmd:
            continue  # PID reused by an unrelated process — do not touch it
        pgid = process_group_id(pid)
        spared = _retained_daemon_pids()
        shares_retained_group = bool(pgid) and any(process_group_id(root) == pgid for root in spared)
        # Capture and stop ordinary descendants before their parent disappears.
        # This is the same selective owner as every current-worker teardown.
        kill_worker_tree(pid)
        if pgid and pgid == pid and not shares_retained_group:
            kill_process_group_id(pgid)  # the worker's own setsid session
        killed.append(pid)
    if killed:
        try:
            append_jsonl(
                _pool().DRIVE_ROOT / "logs" / "supervisor.jsonl",
                {"ts": utc_now_iso(), "type": "orphaned_workers_reaped", "pids": killed},
            )
        except Exception:
            log.debug("Failed to log orphaned worker reap", exc_info=True)
    return len(killed)


@_serialized_worker_lifecycle
def kill_workers_for_update(
    *, result_reason: str, terminal_status: str = "interrupted",
    preserve_running_task_ids: Optional[set[str]] = None,
) -> List[str]:
    """Stop the current pool and return anything whose death could not be proven."""
    with _queue_lock:
        fenced = list(_pool().WORKERS.values())
    teardown_error = ""
    try:
        kill_ok = _pool().kill_workers(
            result_reason=result_reason,
            terminal_status=terminal_status,
            disable_reason="managed_update",
            preserve_pending=True,
            preserve_running_task_ids=set(preserve_running_task_ids or ()),
            reconcile_review_custody=False,  # Final death proof below owns this batch.
        )
        if kill_ok is False:
            teardown_error = "teardown:queue_snapshot_persist_failed"
    except Exception as exc:
        teardown_error = f"teardown:{type(exc).__name__}: {exc}"
    survivors: List[str] = []
    dead_review_pids: set[int] = set()
    for worker in fenced:
        try:
            if worker.proc.is_alive() and worker.proc.pid:
                kill_worker_tree(worker.proc.pid)
                worker.proc.join(timeout=3)
            if worker.proc.is_alive():
                survivors.append(f"worker:{worker.proc.pid or worker.wid}")
            else:
                dead_review_pids.add(int(getattr(worker.proc, "pid", 0) or 0))
        except Exception as exc:
            survivors.append(f"worker:{worker.wid}:{type(exc).__name__}")
    reconcile_confirmed_dead_review_owners(_pool().DRIVE_ROOT, dead_review_pids)
    if teardown_error:
        survivors.append(teardown_error)
    return survivors


def _kill_survivors() -> None:
    """Force-kill any workers and their descendant trees (daemon roots spared)."""
    for w in _pool().WORKERS.values():
        pid = w.proc.pid
        if pid is None:
            continue
        if w.proc.is_alive():
            kill_worker_tree(pid)
            w.proc.join(timeout=2)


@_serialized_worker_lifecycle
def respawn_worker(wid: int, *, ready_attempt: int = 1) -> bool:
    """Replace one owned slot without forking under the queue RLock.

    The lifecycle lock makes the two-phase check/start/swap mutually exclusive
    with full-pool shutdown/start.  The identity check after ``proc.start()``
    prevents a replacement from being installed if the slot was removed while
    the queue lock was released.  The fresh slot is installed unassignable and
    handed to the readiness seam; ``ready_attempt`` is that seam's own
    consecutive-failure count for this slot.
    """
    with _queue_lock:
        old = _pool().WORKERS.get(wid)
    if old is None:
        return False
    if not getattr(old, "active_capacity", True):
        if old.proc.is_alive():
            return False
        with _queue_lock:
            active = sum(1 for worker in _pool().WORKERS.values()
                         if getattr(worker, "active_capacity", True))
            if _pool()._WORKER_POOL_DISABLED_REASON or active >= _pool().MAX_WORKERS:
                return retire_worker(wid, old)
        # The last sleeper can be stopped before its lent slot was created.
        # Fill only that vacancy, retaining normal failed-spawn recovery.
    return _spawn_worker_slot(wid, old, ready_attempt=ready_attempt)


def retire_worker(wid: int, slot: Any) -> bool:
    """Forget only a confirmed-dead slot; parked tasks never mint replacements."""
    with _queue_lock:
        if _pool().WORKERS.get(wid) is not slot or slot.proc.is_alive():
            return False
        _pool().WORKERS.pop(wid)
    from supervisor.worker_process import close_worker_stop_channel
    close_worker_stop_channel(slot.proc)
    try:
        slot.in_q.close()
        slot.in_q.cancel_join_thread()
    except Exception:
        log.debug("Failed to close retired worker queue", exc_info=True)
    _record_worker_pids()
    return True


def _spawn_worker_slot(wid: int, old: Any = None, *, ready_attempt: int = 1) -> bool:
    """Create or replace a slot under the lifecycle serializer, outside queue lock."""
    ctx = _pool()._get_ctx()
    in_q = ctx.Queue()
    events_cursor, spawned_at = events_log_cursor(), time.time()
    from supervisor.worker_process import close_worker_stop_channel, spawn_worker_process

    try:
        proc = spawn_worker_process(ctx, wid, in_q, _pool().get_event_q(), _pool().REPO_DIR, _pool().DRIVE_ROOT)
    except Exception:
        in_q.close()
        in_q.cancel_join_thread()
        raise
    installed = False
    with _queue_lock:
        if _pool().WORKERS.get(wid) is old:
            fresh = _pool().Worker(wid=wid, proc=proc, in_q=in_q, busy_task_id=None, reaping=True)
            _pool().WORKERS[wid] = fresh
            installed = True
    if not installed:
        try:
            if proc.pid:
                kill_worker_tree(proc.pid)
            elif proc.is_alive():
                proc.terminate()
            proc.join(timeout=2)
        finally:
            close_worker_stop_channel(proc)
            try:
                in_q.close()
                in_q.cancel_join_thread()
            except Exception:
                pass
        return False
    # Close the crashed worker's old queue now that nothing can route to it,
    # otherwise its file descriptors / semaphores leak on every respawn.
    if old is not None:
        close_worker_stop_channel(old.proc)
    if old is not None and getattr(old, "in_q", None) is not None:
        try:
            old.in_q.close()
            old.in_q.cancel_join_thread()
        except Exception:
            log.debug("Failed to close old worker queue on respawn", exc_info=True)
    _record_worker_pids()
    threading.Thread(
        target=_pool()._verify_worker_sha_after_spawn,
        args=({wid: fresh}, events_cursor, int(ready_attempt), spawned_at), daemon=True,
    ).start()
    # Do not reset _LAST_SPAWN_TIME here; respawn grace would hide crash storms.
    return True


def _worker_pool_execution_state(
    pool: Any = None, *, running: Any = None,
) -> Dict[str, Any]:
    """Queue capacity without the separate repository-writer admission policy."""
    pool = _pool().WORKERS if pool is None else pool
    running = _pool().RUNNING if running is None else running
    with _queue_lock:
        disabled_reason = str(_pool()._WORKER_POOL_DISABLED_REASON or "")
        worker_count = len(pool)
        exhausted = False
        capacity = False
        for worker in pool.values():
            exhausted = exhausted or bool(getattr(worker, "readiness_exhausted", False))
            if getattr(worker, "active_capacity", True):
                capacity = capacity or not getattr(worker, "readiness_exhausted", False)
                continue
            # The original stack can reclaim an exhausted replacement's reservation.
            task_id = getattr(worker, "busy_task_id", None)
            meta = running.get(task_id) or {}
            wait = meta.get("owner_wait") or {}
            if (not getattr(worker, "reaping", False)
                    and meta.get("worker_id") == getattr(worker, "wid", None)
                    and wait.get("state") == "waiting"
                    and wait.get("task_attempt") == meta.get("attempt")
                    and wait.get("source_ref")
                    and worker.proc.is_alive()):
                capacity = True
        if not disabled_reason and exhausted and not capacity:
            disabled_reason = "worker_readiness_exhausted"
    available = worker_count > 0 and not disabled_reason
    return {
        "available": available,
        "reason_code": "" if available else "worker_pool_unavailable",
        "disabled_reason": disabled_reason or ("no_workers" if not worker_count else ""),
        "worker_count": worker_count,
    }


def disable_exhausted_worker_pool() -> bool:
    """Use the existing terminalization owner only after all recoverable capacity is gone."""
    # Health checks must not delay event intake behind another lifecycle operation.
    # Admission already reads exhaustion independently; cleanup retries on the next tick.
    if not _WORKER_LIFECYCLE_LOCK.acquire(blocking=False):
        return False
    try:
        with _queue_lock:
            # A dead/finishing task may still own a saved child result. Its existing
            # completion/reaper must settle that result before pending-only disablement.
            if (not _pool().WORKERS or _pool().RUNNING or _pool()._worker_pool_execution_state()["disabled_reason"]
                    != "worker_readiness_exhausted"):
                return False
        return _pool().kill_workers(
            disable_reason="worker_readiness_exhausted", terminal_status="failed",
            result_reason=("Worker startup attempts were exhausted before this task could run. "
                           "Use /restart to restore the worker pool, then submit the task again."),
        )
    finally:
        _WORKER_LIFECYCLE_LOCK.release()
