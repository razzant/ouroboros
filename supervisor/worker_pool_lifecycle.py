"""Keeping the pool populated: readiness gate, pid records, reaping, respawn.

A spawned or respawned slot is installed unassignable (``reaping=True``) and opens
only when the child's own ``worker_ready`` row is observed, which is also where the
checkout SHA sampled after initialization is compared with the managed baseline;
a child alive but silent past the readiness window is
torn down and replaced through the same respawn path, a bounded number of times;
a stop it cannot confirm keeps the slot charged with that attempt until the reaper
observes the death. Its own entry-progress row permits one extension, still bounded
from its birth. Each worker's pid is recorded durably once, so an orphan surviving
a restart can be reaped; a replaced worker's queue is closed under the lock before
the new one takes its slot.

Assignment and free counts need a live current PID/birth-bound ``worker_ready``:
``_births_agree`` compares births only when both sides measured one, else the PID
(``_record_worker_processes``). SHA equality proves neither loaded code nor review.
``worker_starting`` grants one bounded extension (``runtime_limits``); watcher
errors use bounded replace/park. Readiness finiteness is ``pool_admission``'s rule.

The lifecycle serializer lives here too: it is a decorator, so it is applied at
import time and cannot be reached through a call-time handle. It is a primitive,
not pool state — nothing rebinds it — so the parent imports it back directly.
It guards pool membership, never a native wait: a reserved slot's kill and join
run outside it and the membership step re-validates identity under it, so the
loop's try-lock misses only a real generation stop, which holds it across joins.
"""

from __future__ import annotations

import logging
import json
import os
import subprocess
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple
from ouroboros.config import WORKER_READY_CEILING_SEC, WORKER_READY_MAX_ATTEMPTS, WORKER_READY_WINDOW_SEC
from ouroboros.process_identity import same_birth
from ouroboros.review_owner_custody import reconcile_confirmed_dead_review_owners
from supervisor.state import append_jsonl
from ouroboros.outcomes import EXECUTION_INFRA_FAILED, terminal_outcome_axes
from ouroboros.task_carrier import EXECUTOR_WORKER, INLINE_EXECUTORS, LANE_CONVERSATION, LANE_PROJECT
from ouroboros.utils import utc_now_iso
from supervisor import pool_admission
from supervisor.queue import _queue_lock


log = logging.getLogger(__name__)


def is_project_slot(slot: Any) -> bool:
    """A slot the project pool counts, fences and assigns queued work to (every slot not on the lane)."""
    return getattr(slot, "lane", LANE_PROJECT) != LANE_CONVERSATION


def is_lane_slot(slot: Any) -> bool:
    """A conversation-lane carrier: outside the owner's cap, memory admission and the exhaustion fence."""
    return getattr(slot, "lane", LANE_PROJECT) == LANE_CONVERSATION


def is_worker_slot(slot: Any) -> bool:
    """A custodied child process (not an inline executor's thread-backed pseudo-slot)."""
    return getattr(slot, "executor", EXECUTOR_WORKER) not in INLINE_EXECUTORS


def project_slots(pool: Any = None) -> list:
    pool = _pool().WORKERS if pool is None else pool
    return [slot for slot in pool.values() if is_project_slot(slot)]


def lane_slots(pool: Any = None) -> list:
    pool = _pool().WORKERS if pool is None else pool
    return [slot for slot in pool.values() if is_lane_slot(slot)]


def idle_project_slots(pool: Any = None) -> list:
    """Ready project slots holding no task and not parked for a wait (``active_capacity``), in slot order."""
    return [slot for slot in project_slots(pool) if getattr(slot, "busy_task_id", None) is None
            and worker_ready(slot) and getattr(slot, "active_capacity", True)]


def usable_booting() -> int:
    """Booting project slots that will carry work once ready: no task yet, not retiring, readiness not exhausted."""
    return sum(1 for slot in project_slots() if getattr(slot, "busy_task_id", None) is None
               and getattr(slot, "booting", False) and not getattr(slot, "retiring", False)
               and not getattr(slot, "readiness_exhausted", False))


def pool_slot_counts(pool: Any = None, running: Any = None) -> Dict[str, Any]:
    """The one count of the pool's slots, published by ``queue_publication._pool_counts`` and read by admission.

    ``max_workers`` is the owner's optional cap on acting project tasks (None: no number, the machine's
    memory is the limit, ``pool_admission``); ``worker_total`` counts the project processes that exist.
    ``busy_worker_count`` counts slots holding a task: ``acting_worker_count`` of them act (a launch claim
    included) and ``parked_worker_count`` wait on their own process (a park on the slot, an accepted owner
    wait or warm sleep, a live in-slot wait: ``pool_admission.waits_on_its_process``). The booting/reaping/
    retiring flags overlap (a fresh slot is booting and reaping until its readiness row);
    ``assignable_idle_workers`` is what dispatch could use this instant. Conversation carriers are another
    lane, outside the cap, counted apart. No queue, filesystem or admission work; ``worker_ready`` asks
    each slot's ``Process`` whether it is alive.
    """
    from ouroboros.runtime_limits import get_max_workers

    slots, running = _pool().WORKERS if pool is None else pool, _pool().RUNNING if running is None else running
    projects, carriers, now = project_slots(slots), lane_slots(slots), time.time()
    busy = [slot for slot in projects if getattr(slot, "busy_task_id", None) is not None]
    waiting = sum(not pool_admission.acting(slot, running.get(slot.busy_task_id), now) for slot in busy)
    return {
        "max_workers": get_max_workers(),
        "worker_total": len(projects),
        "busy_worker_count": len(busy),
        "acting_worker_count": len(busy) - waiting,
        "parked_worker_count": waiting,
        "booting_count": sum(1 for slot in projects if getattr(slot, "booting", False)),
        "reaping_count": sum(1 for slot in projects if getattr(slot, "reaping", False)),
        "retiring_count": sum(1 for slot in projects if getattr(slot, "retiring", False)),
        "assignable_idle_workers": len(idle_project_slots(slots)),
        "lane_carrier_total": len(carriers),
    }


def admissible_starts(reading: Any = None) -> int:
    """How many new project processes this pass may start, under Q: the one answer the assignment pass
    sizes its choice with (``dispatch_choice.run_pass``) and growth starts (``reconcile_worker_capacity``),
    from the pass's reading of the machine (``pool_admission``: None, not read, starts nothing) and the
    pool's own counts. Reusing an idle process, a warm return and a crash replacement never ask it."""
    slots = list(_pool().WORKERS.values())
    return pool_admission.admit(reading, pool_slot_counts(), slots, parked_now=pool_admission.parked(project_slots()))


def acting_room(pool: Any = None, running: Any = None) -> Optional[int]:
    """Under Q: what the owner's cap leaves for acting project tasks (claims count as acting); None: no cap."""
    counts = pool_slot_counts(pool, running)
    return None if counts["max_workers"] is None else max(0, counts["max_workers"] - counts["acting_worker_count"])


def project_fenced(slot: Any) -> bool:
    """Whether the project pool's stop fence owns this slot's membership (a lane slot is never fenced by it)."""
    return bool(_pool()._WORKER_POOL_DISABLED_REASON) and is_project_slot(slot)


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


class SpawnRefusedInMarkerWindow(RuntimeError):
    """No child starts inside an assisted update's conflict-marker window: a worker spawned now would import the
    tree's markers. The dead slot keeps its place and the reaper's refusal episode retries it each tick until the
    window ends (``task_reaper._respawn_refused``); the degraded executor covers the lane's turns meanwhile."""
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
    reason: str = "Worker process was stopped before this task finished. Task was not completed.",
    status: str = "",
    stop_source: str = "",
) -> str:
    """Write failure result for a crashed/orphaned task.

    Returns the FINAL persisted status: if the task already reached a terminal state, the monotonic
    guard preserves it and that existing status is returned (so the UI event matches disk);
    otherwise the written failure status.

    ``stop_source`` is the typed cause a known stop door passes (the owner's Restart, a graceful
    server shutdown). It lands as ``cancel_origin`` — the field a settled cancel intent records —
    and never outranks an earlier stop: an active intent for the task is the cause as recorded (an
    owner Stop stays a Stop), only a task no intent names takes the door's own cause, and an
    unreadable intent store records none. A cancelled result without a door source still retains its
    recorded intent (including Panic).
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

    The event log rotates (CPL4-C1), so the file the cursor was taken from may now BE an archive
    segment. Rotation is detected by IDENTITY, not by size: the previous test ("the live file is
    smaller than my offset") missed every rotation where the new live file had already grown past
    the old offset — exactly what happens under a busy supervisor — and then read the wrong file's
    bytes from a meaningless offset. When the identity moved, the matching segment is read from the
    SAME offset (the continuation is exact); a cursor whose file is gone entirely falls back to the
    newest segment whole.

    A missing live file is an EMPTY read, never an error: the log may not have been written yet (a
    zeroed cursor), a poll may land in the instant between the rotator's ``os.replace`` and its
    ``touch``, or the file may have been removed by hand. Nothing durable is lost to that read — a
    row appended before the rename sits in the segment the cursor's identity finds, one appended
    after the touch sits in the new live file, and the next poll sees both.
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
    try:  # a slot without a process (a placeholder) has no pid, never an accounting error
        return int(getattr(getattr(slot, "proc", None), "pid", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _births_agree(slot: Any, reported: Dict[str, Any]) -> bool:
    """Births are compared only when both sides measured one; otherwise the slot's own pid decides.

    The owner's rule (14 Oct): bind readiness to the process birth, falling back to the pid.
    Kernel births of one validated kind decide first (``process_identity.same_birth``): the
    parent and the child may spell one ``ps`` birth in different zones or locales.
    """
    same = same_birth(getattr(slot, "kernel_birth", ""), reported.get("kernel_birth"))
    if same is not None:
        return same
    ours, theirs = str(getattr(slot, "process_birth", "") or ""), str(reported.get("process_birth") or "")
    return not (ours and theirs) or ours == theirs


def worker_ready(slot: Any) -> bool:
    """Affirmative readiness belongs to the identity its own row opened, and requires liveness.

    That identity is ``(pid, birth)``; its birth is empty only when neither the parent nor
    the child could measure one, and the slot's own live ``Process`` then carries it.
    """
    pid, birth = _slot_pid(slot), str(getattr(slot, "process_birth", "") or "")
    return bool(pid > 0 and getattr(slot, "ready_identity", None) == (pid, birth)
                and not slot.reaping and not getattr(slot, "readiness_exhausted", False)
                and slot.proc.is_alive())


def _supervisor_row(row: Dict[str, Any]) -> None:
    append_jsonl(_pool().DRIVE_ROOT / "logs" / "supervisor.jsonl", {"ts": utc_now_iso(), **row})


def _verify_worker_sha_after_spawn(
    slots: Dict[int, Any], events_cursor: Tuple[int, int, int], attempt: int = 1,
    spawned_at: float = 0.0,
) -> None:
    """Hold freshly spawned slots unassignable until each child confirms ready.

    The ONE readiness seam for both spawn paths: ``spawn_workers`` and ``respawn_worker`` install a
    slot with ``reaping=True`` (the marker the assignment path, the crash detector and the reaper
    already honour) and hand it here. A slot opens only when the child's own ``worker_ready`` row
    (supervisor/worker_process.py) names its pid. That row samples the checkout after
    initialization; comparison with ``current_sha`` is diagnostic only, not loaded-code attestation,
    authorship or review evidence. A child that is alive but silent past ``WORKER_READY_WINDOW_SEC``
    is torn down and replaced; its own ``worker_starting`` row permits one extension to
    ``WORKER_READY_CEILING_SEC`` from birth. Replacement still runs through ``respawn_worker`` — at
    most ``WORKER_READY_MAX_ATTEMPTS`` consecutive times for one slot, then the slot is parked and
    reported. A child that DIED during boot is released to the crash detector, which already owns
    process death (``worker_dead_detected``, the one retry). This is a contract
    distinct from process liveness and from the task idle rail: a deadlocked child is alive and
    holds no task. ``spawned_at`` is the instant before ``proc.start()``: the window and the
    reported wait count from the child's birth, not from this thread's start.

    Watcher errors use the same bounded replace/park owner, never an alive-but-unready assignment
    escape. Unknown teardown keeps its charged reservation and its attempt count for the reaper
    (``finish_unready_teardown``).
    """
    pending = dict(slots)
    started = float(spawned_at or 0.0) or time.time()
    # This spawn's own thread pays the custody ledger (three ``ps`` per child on macOS),
    # never the supervisor loop; its slots stay unassignable until this thread opens them.
    _record_worker_processes(slots)
    try:
        _watch_booting_slots(pending, events_cursor, attempt, started)
    except Exception as exc:
        log.exception(
            "Worker readiness watcher failed (attempt %d); recovering %d booting slot(s) within the readiness bound",
            attempt, len(pending),
        )
        for wid, slot in list(pending.items()):
            try:
                _supervisor_row({"type": "worker_ready_watcher_error", "worker_id": wid,
                                 "attempt": attempt, "error": str(exc)[:400]})
                _replace_unready_slot(wid, slot, 0, started, attempt)
            except Exception:
                # Unknown teardown cannot lend the reservation or open an unready actor.
                _defer_unready_teardown(wid, slot, 0, started, attempt)
                log.exception("Recovery of booting worker slot %d failed", wid)


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
        rows_by_pid: Dict[int, List[Dict[str, Any]]] = {}
        for row in _worker_events_since(events_cursor, "worker_ready"):
            try:
                rows_by_pid.setdefault(int(row.get("pid") or 0), []).append(row)
            except (TypeError, ValueError):
                continue
        for wid, slot in list(pending.items()):
            row = _own_ready_row(slot, rows_by_pid.get(_slot_pid(slot), ()))
            if row is not None and _open_ready_slot(
                    wid, slot, row, expected_sha, owner_chat_id, started, attempt):
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


def _own_ready_row(slot: Any, rows: Any) -> Optional[Dict[str, Any]]:
    """The child's own ``worker_ready`` row among the rows naming its pid, or None.

    A row whose measured birth differs from the slot's belongs to another process that held
    the pid (a stale or reused pid): never this child's. An equal kernel birth wins over an
    exact spelling, which wins over a row without one. Without a slot birth the newest row is
    taken: an earlier holder of a reused pid died, and so wrote its row, before this child
    received the pid.
    """
    birth, kernel = str(getattr(slot, "process_birth", "") or ""), getattr(slot, "kernel_birth", "")
    agreeing = [row for row in rows if _births_agree(slot, row)]
    same = [row for row in agreeing if same_birth(kernel, row.get("kernel_birth")) is True]
    exact = [row for row in agreeing if birth and str(row.get("process_birth") or "") == birth]
    return (same or exact or agreeing or [None])[-1]


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
) -> bool:
    """Open the ready slot before the diagnostic checkout comparison.

    Ancestry of the observed checkout against the baseline is diagnostic only:
    a descendant stays in the supervisor log; a non-descendant or an unavailable
    check also warns the owner chat; no baseline skips the comparison.
    """
    with _queue_lock:
        pid = _slot_pid(slot)
        owned = (_pool().WORKERS.get(wid) is slot
                 and slot.booting and not slot.retiring
                 and not getattr(slot, "readiness_exhausted", False)
                 and pid > 0 and row.get("pid") == pid
                 and _births_agree(slot, row)
                 and slot.proc.is_alive())
        if owned:
            # A birth the parent could not measure is adopted from the child, which measured it alive.
            slot.process_birth = str(slot.process_birth or row.get("process_birth") or "")
            slot.ready_identity = (pid, slot.process_birth)
            slot.booting = False
            slot.reaping = False
            pool_admission.note_ready(row)  # growth after a readiness park resumes; this generation's fresh size
    if not owned:
        return False
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
    return True


def _release_booting_slot(
    wid: int, slot: Any, started: float, attempt: int, reason: str, **detail: Any,
) -> None:
    """Hand a booting slot to the crash detector: ``reaping`` cleared if it is still ours, one typed row.

    "Ours" means still registered in the pool and not readiness-exhausted; any other
    slot is left untouched and the row records ``slot_released=False``.

    ``died_during_boot`` carries the exit code.
    """
    with _queue_lock:
        owned = (_pool().WORKERS.get(wid) is slot
                 and slot.booting and not slot.retiring
                 and not getattr(slot, "readiness_exhausted", False))
        if owned:
            slot.booting = False
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

    A worker's tree is not the worker's property: the installation's daemon roots (``daemon``-scope
    custody rows — the shared Claudexor daemon when a task worker was the first to need it) outlive
    every worker and server generation, so they and their descendants are spared. ``keep_services``
    additionally spares the deliberately kept session services a verifier still needs when ONE task
    is cancelled or timed out; a generation change ends those services with the generation, so the
    pool paths leave it off. The platform helper applies the same retained-subtree contract on every
    OS.
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


def _booting_slot_current(wid: int, slot: Any) -> bool:
    """This watcher's booting slot is still the pool's own at ``wid``; a pool restart replaces or clears it."""
    with _queue_lock:
        return _pool().WORKERS.get(wid) is slot and slot.booting and not slot.retiring


def _replace_unready_slot(wid: int, slot: Any, owner_chat_id: int, started: float, attempt: int,
                         window_sec: float = 0.0) -> None:
    """No worker_ready inside the window: tear the child down and replace the slot, bounded.

    The lifecycle serializer guards pool membership, never a native wait: the kill and join run
    without it, since the booting slot is unassignable and only this watcher acts on it. The
    replace/park step takes it (lifecycle -> queue lock order, like every pool start/kill/respawn)
    and re-validates the identity first, so a generation change (``kill_workers`` ->
    ``spawn_workers``) that installed a fresh live slot at ``wid`` meanwhile is never evicted by
    this stale watcher's respawn. A stop it cannot confirm (the child outlives the kill, or the
    kill raised) stays charged with this attempt until the reaper observes the death.
    """
    if not _booting_slot_current(wid, slot):
        return
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
    try:
        if pid:
            kill_worker_tree(pid)
        slot.proc.join(timeout=2)
    except Exception:
        log.warning("Teardown of unready worker %d is unconfirmed", wid, exc_info=True)
    if slot.proc.is_alive():
        _defer_unready_teardown(wid, slot, owner_chat_id, started, attempt)
        return  # stop unconfirmed: never spawn beside it
    with _WORKER_LIFECYCLE_LOCK:
        if _booting_slot_current(wid, slot):
            _replace_or_park_unready_slot(wid, slot, owner_chat_id, started, attempt)


def _defer_unready_teardown(wid: int, slot: Any, owner_chat_id: int, started: float, attempt: int) -> None:
    """Hand a booting slot whose stop is unknown to the reaper, charged and unassignable.

    It is no longer future capacity, and its attempt count travels with the job.
    """
    from supervisor import queue

    with _queue_lock:
        if (_pool().WORKERS.get(wid) is not slot or not slot.booting or slot.retiring
                or getattr(slot, "readiness_exhausted", False)):
            return
        queue._ensure_reaper_started()
        slot.booting, slot.reaping = False, True
        queue._reap_queue.put({"kind": "unready_worker_teardown", "worker_id": wid, "worker": slot,
                               "owner_chat_id": owner_chat_id, "started": started, "attempt": attempt,
                               "drive_root": str(_pool().DRIVE_ROOT), "first_stop": True})


def _unready_teardown_current(job: dict, slot: Any) -> bool:
    """The reaper's unready slot is still this pool's own, charged and neither booting nor parked."""
    with _queue_lock:
        return (str(_pool().DRIVE_ROOT) == job["drive_root"] and _pool().WORKERS.get(job["worker_id"]) is slot
                and not slot.booting and not slot.retiring and not getattr(slot, "readiness_exhausted", False))


def finish_unready_teardown(job: dict) -> None:
    """The reaper's half: confirmed death continues the SAME bounded replace/park step.

    Unconfirmed death raises ``TerminalFileRecoveryPending``; the reaper retains the job for the
    next health tick, whose re-drive repeats one bounded tree kill and then joins without waiting.
    It is never a crash, and the count never resets. The kill and join hold no lifecycle
    serializer (the slot stays reserved under Q); the replace/park step takes it and re-validates.
    """
    from supervisor.task_reaper import TerminalFileRecoveryPending

    slot, wid = job["worker"], job["worker_id"]
    if not _unready_teardown_current(job, slot):
        return
    if slot.proc.is_alive() and _slot_pid(slot):
        kill_worker_tree(_slot_pid(slot))
        slot.proc.join(timeout=2 if job.pop("first_stop", False) else 0)  # the first stop sees an ordinary exit now
    if slot.proc.is_alive():
        raise TerminalFileRecoveryPending("unready worker stop is not yet confirmed")
    with _WORKER_LIFECYCLE_LOCK:
        if _unready_teardown_current(job, slot):
            _replace_or_park_unready_slot(wid, slot, job["owner_chat_id"], job["started"], job["attempt"])


def _replace_or_park_unready_slot(wid: int, slot: Any, owner_chat_id: int, started: float, attempt: int) -> None:
    """After a confirmed stop: retry or park; only the first park since a ready notifies the owner."""
    if attempt < WORKER_READY_MAX_ATTEMPTS:
        try:
            respawn_worker(wid, ready_attempt=attempt + 1)
        except Exception:
            log.warning("Respawn of unready worker %d failed; clearing reaping for recovery", wid, exc_info=True)
            with _queue_lock:
                if _pool().WORKERS.get(wid) is slot:
                    slot.booting = False
                    slot.reaping = False  # the crash detector recovers the dead slot
        return
    log.error("Worker %d never confirmed ready in %d attempts; slot parked until restart", wid, attempt)
    with _queue_lock:
        if _pool().WORKERS.get(wid) is not slot:
            return
        slot.readiness_exhausted = slot.reaping = True
        slot.booting = False
        pool_admission.note_park(slot)  # count this park until the next successful worker_ready
        first_park = pool_admission.parked(_pool().WORKERS.values()) == 1
    if owner_chat_id and first_park:
        _pool().send_with_budget(
            owner_chat_id,
            f"⚠️ Worker slot {wid} never confirmed ready in {attempt} attempts "
            f"(waited {time.time() - started:.0f}s on the last attempt); the slot is parked. Use /restart.",
            role="system", system_type="worker_readiness_notice")
    if is_project_slot(slot):
        _pool().disable_exhausted_worker_pool()


def _record_worker_processes(slots: Optional[Dict[int, Any]] = None) -> None:
    """Record worker identity in the process ledger for restart custody.

    A captured birth means the row exists: a spawn records only its fresh child,
    never every survivor again (linear in spawns, not pool size times churn).
    A spawn's readiness thread passes its own ``slots``; without them the whole
    pool is visited. The birth is written under Q, beside the readiness owner.
    """
    try:
        from ouroboros.process_custody import record_process

        for w in list((_pool().WORKERS if slots is None else slots).values()):
            if w.proc.pid and is_worker_slot(w) and not getattr(w, "process_birth", ""):
                was_alive = w.proc.is_alive()
                record = record_process(
                    _pool().DRIVE_ROOT,
                    pid=int(w.proc.pid),
                    cmd=f"ouroboros-worker-{w.wid}",
                    purpose=f"worker:{w.wid}",
                    scope="session",
                )
                if was_alive and w.proc.is_alive():
                    from ouroboros.process_identity import process_kernel_birth

                    fingerprint = record["fingerprint"]
                    kernel = process_kernel_birth(int(w.proc.pid)) or ""
                    with _queue_lock:
                        w.process_birth = (w.process_birth or fingerprint.get("start_time_boot")
                                           or fingerprint.get("start_time") or "")
                        # beside the legacy birth: kernel evidence for _births_agree and the exit door (worker_health)
                        w.kernel_birth = getattr(w, "kernel_birth", "") or (kernel if w.proc.is_alive() else "")
    except Exception:
        log.warning("Failed to ledger worker pids; readiness falls back to the child's birth or pid", exc_info=True)


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
            retain_saved_work=True,  # the update's restart returns what was saved (#1563)
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
            elif worker.proc.is_alive():  # an inline executor's thread: cooperative, at its next control boundary
                worker.proc.terminate()
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


def _kill_survivors(slots: Any = None) -> None:
    """Force-kill the given workers (default: every slot) and their descendant trees (daemon roots spared)."""
    for w in list(_pool().WORKERS.values() if slots is None else slots):  # read outside the queue lock while a dead slot may retire
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
    consecutive-failure count for this slot. A current pool stop fence
    (exhaustion, Stop, update) starts no child and forgets no slot: a
    death or timeout captured earlier, a lent slot's included, leaves its dead
    slot to that fence's owner, whose stop needs the membership it captured.
    """
    with _queue_lock:
        old = _pool().WORKERS.get(wid)
        fenced = old is not None and project_fenced(old)
    if old is None or fenced:
        return False
    if not is_worker_slot(old):
        return retire_worker(wid, old)  # an inline executor's pseudo-slot ends with its thread; nothing replaces it
    if not getattr(old, "active_capacity", True):
        # A sleeper's dead process mints no replacement (growth buys processes for demand); an enabled
        # pool's last member is still replaced, with normal failed-spawn recovery.
        return False if old.proc.is_alive() else bool(retire_or_replace_worker(wid, old))
    return _spawn_worker_slot(wid, old, ready_attempt=ready_attempt, lane=getattr(old, "lane", LANE_PROJECT))


def retire_worker(wid: int, slot: Any) -> bool:
    """Forget only a confirmed-dead slot; parked tasks never mint replacements.

    A fenced pool forgets nothing: its stop owns the captured membership. Survivors' rows are not re-ledgered.
    """
    with _queue_lock:
        if project_fenced(slot) or _pool().WORKERS.get(wid) is not slot or slot.proc.is_alive():
            return False
        _pool().WORKERS.pop(wid)
    from supervisor.worker_health import owe_worker_exit
    from supervisor.worker_process import close_worker_stop_channel
    owe_worker_exit(slot)  # an idle retirement also ends this process's earlier tasks' claims
    close_worker_stop_channel(slot.proc)
    try:
        slot.in_q.close()
        slot.in_q.cancel_join_thread()
    except Exception:
        log.debug("Failed to close retired worker queue", exc_info=True)
    return True


def retire_or_replace_worker(wid: int, slot: Any) -> str:
    """Forget a confirmed-dead slot, but replace an enabled pool's last member in place.

    Growth needs a member (an empty pool is stopped); a fence forgets and starts nothing. Serializer held, not Q."""
    with _queue_lock:
        last = len(_pool().WORKERS) == 1 and _pool().WORKERS.get(wid) is slot and not slot.proc.is_alive()
        fenced = project_fenced(slot)
    if not last:
        return "retired" if retire_worker(wid, slot) else ""
    try:
        return ("replaced" if not fenced and is_worker_slot(slot)
                and _spawn_worker_slot(wid, slot, lane=getattr(slot, "lane", LANE_PROJECT)) else "")
    except SpawnRefusedInMarkerWindow:
        return ""  # the window forbids the child; the dead last member stays for the tick after it


def request_worker_quiescence(slot: Any) -> None:
    """One probe per idle spell; late work can finish while the slot stays assignable.

    Called under Q. A real assignment revokes the token before its queue put.
    The child answers only once its local reviews, tool futures (late callbacks
    included) and abandoned senders settle (``worker_process.worker_main``).
    """
    if not slot.quiescence_token:
        token = uuid.uuid4().hex
        slot.in_q.put({"type": "worker_quiescence", "token": token})
        slot.quiescence_token = token
        slot.quiescence_confirmed = False


def handle_worker_quiescent(event: dict, ctx: Any) -> None:
    with _queue_lock:
        slot = ctx.WORKERS.get(event.get("worker_id"))
        if (slot is not None and slot.busy_task_id is None and worker_ready(slot)
                and event.get("pid") == _slot_pid(slot)
                and _births_agree(slot, event)
                and event.get("token") == slot.quiescence_token and slot.quiescence_token):
            slot.quiescence_confirmed = True
            # Best-effort dialogue rows it still holds: counted lost only if this slot is retired.
            slot.quiescence_retained_rows = max(0, int(event.get("retained_rows") or 0))


def retire_quiescent_worker(slot: Any) -> bool:
    """Reserve an acknowledged idle worker for the existing off-loop reaper.

    Lifecycle -> Q caller owns the demand/resume decision. Its reservation stays
    charged through a failed kill or unconfirmed death; no task terminal is made.
    """
    from supervisor import queue

    with _queue_lock:
        if (project_fenced(slot) or _pool().WORKERS.get(slot.wid) is not slot
                or slot.busy_task_id is not None or not worker_ready(slot)):
            return False
        request_worker_quiescence(slot)
        if not slot.quiescence_confirmed:
            return False
        queue._ensure_reaper_started()
        slot.retiring = slot.reaping = True
        try:
            queue._reap_queue.put({"kind": "idle_worker_retirement", "worker_id": slot.wid,
                                   "worker": slot, "token": slot.quiescence_token,
                                   "drive_root": str(_pool().DRIVE_ROOT), "first_stop": True})
        except Exception:
            slot.retiring = slot.reaping = False
            raise
        return True


def _idle_retirement_current(job: dict, slot: Any) -> bool:
    """The reserved acknowledged actor is still this pool's own, and no stop fence owns membership."""
    with _queue_lock:
        return (str(_pool().DRIVE_ROOT) == job["drive_root"]
                and _pool().WORKERS.get(slot.wid) is slot and slot.retiring
                and slot.busy_task_id is None and job["token"] == slot.quiescence_token
                # A captured fence stop keeps membership stable; its owner tears down.
                and not project_fenced(slot))


def reap_idle_worker(job: dict) -> None:
    """Kill only the reserved acknowledged actor, then forget only confirmed death (an enabled
    pool's last member is replaced). The first stop's bounded join sees an ordinary exit now; one row records it.

    The lifecycle serializer guards membership, never a native wait: the slot is reserved under Q
    (``retiring``), so the kill and join run without it and assignment keeps its turn; the
    membership step takes it after re-validating the identity.
    """
    from supervisor.task_reaper import TerminalFileRecoveryPending

    slot = job["worker"]
    if not _idle_retirement_current(job, slot):
        return
    if slot.proc.is_alive():
        kill_worker_tree(_slot_pid(slot), keep_services=True)
        # Each health-tick re-drive repeats one bounded tree kill, then joins without waiting.
        slot.proc.join(timeout=2 if job.pop("first_stop", False) else 0)
    if slot.proc.is_alive():
        raise TerminalFileRecoveryPending("idle worker stop is not yet confirmed")
    with _WORKER_LIFECYCLE_LOCK:
        outcome = _idle_retirement_current(job, slot) and retire_or_replace_worker(slot.wid, slot)
    if outcome:
        lost = int(getattr(slot, "quiescence_retained_rows", 0) or 0)
        _supervisor_row({"type": "worker_retired", "worker_id": slot.wid, "pid": _slot_pid(slot), "token": job["token"],
                         "process_birth": slot.process_birth, "replaced": outcome == "replaced",
                         **({"dialogue_rows_lost": lost} if lost else {})})


@_serialized_worker_lifecycle
def reconcile_worker_capacity(residual_demand, reading: Any = None) -> None:
    """Demand-sized growth; one stable ready spare at quiescence, with no idle timer.

    ``residual_demand()`` answers, under Q, how many rows the assignment pass chose that no idle
    worker took (``dispatch_choice.run_pass``; None: the pass was blocked); it reads no store and
    runs no chooser. Booting slots can satisfy that demand; exhausted or retiring slots cannot.
    New processes start only as ``admissible_starts`` admits them from the pass's ``reading``
    (None: the pass did not read the machine, so nothing starts).

    Growth and retirement take the lifecycle serializer, then Q; a spawn starts outside
    Q. Surplus is retired through the quiescence probe; retirement kills with kept services
    and frees a process only on confirmed death; it is intentional, never a crash, and
    there is no idle timer, rotation or setting.
    """
    from supervisor.conversation_lane import reconcile_conversation_lane

    # The conversation lane first (owner decision 4A): its one reserved spare is
    # replenished and its awaiting turns dispatched whatever the project fence says.
    reconcile_conversation_lane()
    with _queue_lock:
        if not _pool().WORKERS or _pool()._WORKER_POOL_DISABLED_REASON:
            return
        demand = residual_demand()
        if demand is None or (_pool().repo_writer_admission_closed() and not demand):
            return
        # The pool's one quiescent spare is the conversation lane's (4A): project
        # growth buys only demand, and a pooled task after idle cold-starts.
        wanted = demand
        idle = idle_project_slots()
        missing = min(max(0, wanted - len(idle) - usable_booting()), admissible_starts(reading))
        # An empty table stops growth: a pool whose lane spare is absent keeps its last
        # idle slot, as it did before the lane existed.
        keep = wanted if len(_pool().WORKERS) > len(idle) else max(1, wanted)
        # Ask all surplus candidates: a resident with late duties is not the spare.
        if len(idle) > keep:
            for slot in idle:
                request_worker_quiescence(slot)
            quiet = [w for w in idle if w.quiescence_confirmed]
            # Retire only acknowledged surplus; growth never creates a project spare during work.
            for slot in quiet[keep:]:
                retire_quiescent_worker(slot)
        next_id = max(_pool().WORKERS, default=-1) + 1
    for wid in range(next_id, next_id + missing):
        # Starts run without Q. The serializer excludes Stop/update/exhaustion teardown;
        # the readiness owner can still fence the pool while this start is in flight.
        from supervisor.worker_chat_lane import marker_window_closed

        with _queue_lock:
            if (_pool()._WORKER_POOL_DISABLED_REASON or marker_window_closed()
                    or (_pool().repo_writer_admission_closed() and not residual_demand())):
                break
        try:
            _spawn_worker_slot(wid)
        except Exception:
            log.warning("Failed to create demanded worker", exc_info=True)
            break


def _spawn_worker_slot(wid: int, old: Any = None, *, ready_attempt: int = 1, lane: str = LANE_PROJECT) -> bool:
    """Create or replace a slot under the lifecycle serializer, outside queue lock.

    ``lane`` names the slot's lane; a conversation carrier ignores the project
    pool's stop fence (its membership is not that fence's)."""
    from supervisor.worker_chat_lane import marker_window_closed

    if marker_window_closed():  # respawn, replacement and growth alike: the window forbids every child
        raise SpawnRefusedInMarkerWindow(
            f"worker {wid} not started: a child started now would import the tree's conflict markers")
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
        # Callers refuse a fenced start before it begins, and every fence or slot writer
        # holds this serializer (health only tries it), so this refusal is a backstop.
        if (_pool().WORKERS.get(wid) is old
                and not (_pool()._WORKER_POOL_DISABLED_REASON and lane == LANE_PROJECT)):
            fresh = _pool().Worker(wid=wid, proc=proc, in_q=in_q, busy_task_id=None, reaping=True, booting=True,
                                   lane=lane)
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
        from supervisor.worker_health import owe_worker_exit

        owe_worker_exit(old)  # every replacement door: a confirmed-ended member's claims retire too
        close_worker_stop_channel(old.proc)
    if old is not None and getattr(old, "in_q", None) is not None:
        try:
            old.in_q.close()
            old.in_q.cancel_join_thread()
        except Exception:
            log.debug("Failed to close old worker queue on respawn", exc_info=True)
    # The readiness thread ledgers the child first (``_verify_worker_sha_after_spawn``).
    threading.Thread(
        target=_pool()._verify_worker_sha_after_spawn,
        args=({wid: fresh}, events_cursor, int(ready_attempt), spawned_at), daemon=True,
    ).start()
    # Do not reset _LAST_SPAWN_TIME here; respawn grace would hide a dying pool's deaths.
    return True


def _worker_pool_execution_state(pool: Any = None) -> Dict[str, Any]:
    """Queue capacity without the separate repository-writer admission policy.

    Project slots alone count: the conversation lane's carriers are outside the owner's cap
    and the exhaustion fence. A table that holds only the lane's spare is a STARTED pool with
    zero project members (growth buys them on demand). Exhaustion is counted on healthy slots:
    readiness parks since the last successful ``worker_ready`` stop growth when they outnumber
    live ready project processes (``pool_admission``); with no healthy slot left nothing can run.
    """
    table = _pool().WORKERS if pool is None else pool
    projects = project_slots(table)
    with _queue_lock:
        disabled_reason = str(_pool()._WORKER_POOL_DISABLED_REASON or "")
        healthy = any(not getattr(w, "readiness_exhausted", False) for w in projects)
        if not disabled_reason and not healthy and pool_admission.parked(projects):
            disabled_reason = "worker_readiness_exhausted"
    available = len(table) > 0 and not disabled_reason
    return {
        "available": available,
        "reason_code": "" if available else "worker_pool_unavailable",
        "disabled_reason": disabled_reason or ("no_workers" if not table else ""),
        "worker_count": len(projects),
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
            # The conversation lane's rows are not the project pool's to settle.
            project_running = any(not conversation_row(meta) for meta in _pool().RUNNING.values())
            if (not project_slots() or project_running or _pool()._worker_pool_execution_state()["disabled_reason"]
                    != "worker_readiness_exhausted"):
                return False
        return _pool().kill_workers(
            disable_reason="worker_readiness_exhausted", terminal_status="failed",
            result_reason=("Worker startup attempts were exhausted before this task could run. "
                           "Use /restart to restore the worker pool, then submit the task again."),
            lanes=frozenset({LANE_PROJECT}),
        )
    finally:
        _WORKER_LIFECYCLE_LOCK.release()


def conversation_row(meta: Any) -> bool:
    """A RUNNING row the conversation lane admitted (its carrier may be awaited, a worker or inline)."""
    return isinstance(meta, dict) and meta.get("lane") == LANE_CONVERSATION
