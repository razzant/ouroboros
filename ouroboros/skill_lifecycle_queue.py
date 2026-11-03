"""Single FIFO lane for mutating skill install/review/dependency/enable work.

A job holds the lane (an in-process lock plus the cross-process
``state/skill_lifecycle.lock``) while it mutates. A long read-only step inside a
job (a review panel reading frozen candidates of a payload) lends the lane out
with ``lend_lifecycle_lane`` and takes it back before persisting, so installs,
grants and toggles do not wait for a review's model calls.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import time
import pathlib
import re
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Deque, Dict, Optional

from ouroboros.utils import utc_now_iso as _now_iso
from ouroboros.config import runtime_setting
import contextvars
from ouroboros.settings_integrity import copy_task_settings_context

log = logging.getLogger(__name__)

_MAX_EVENTS = 80
_BACKGROUND: set = set()  # strong references to answered-pending owner actions still running
JOB_ANSWER_WAIT_SEC = 10.0  # the transport bound install/update already use; a slow action answers its job
_STALE_RUNNING_JOB_SEC = int(os.environ.get("OUROBOROS_SKILL_LIFECYCLE_STALE_SEC", "1800"))


def _iso_age_seconds(value: str) -> int:
    try:
        dt = datetime.fromisoformat(str(value or "").replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return max(0, int((datetime.now(timezone.utc) - dt).total_seconds()))
    except Exception:
        return 0


@dataclass
class LifecycleJob:
    id: str
    kind: str
    target: str
    source: str = ""
    dedupe_key: str = ""
    status: str = "queued"
    message: str = ""
    error: str = ""
    # C4 (owner batch-4 3=A): the chat this job's progress belongs to. A
    # TASK-BOUND job (a review started by an agent tool) carries the task's own
    # chat; a truly UNBOUND job (HTTP/gateway lifecycle actions) stays 0 — the
    # Skill Review panel chat. Note Main is chat 1, the panel is 0 — the two
    # must never be conflated (contracts/chat_id_policy.py).
    chat_id: int = 0
    # Presentation identity is carried before the first queued notification.
    # These are domain provenance facts only: the external progress task_id
    # remains the synthetic lifecycle id minted by `_chat_task_id`.
    group_id: str = ""
    task_id: str = ""
    root_task_id: str = ""
    origin_task_id: str = ""
    origin_root_task_id: str = ""
    presentation_owner_task_id: str = ""
    queued_at: str = field(default_factory=_now_iso)
    started_at: str = ""
    finished_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        age_seconds = _iso_age_seconds(self.started_at or self.queued_at)
        stale = bool(self.status == "running" and age_seconds >= _STALE_RUNNING_JOB_SEC)
        return {
            "id": self.id,
            "kind": self.kind,
            "target": self.target,
            "source": self.source,
            "dedupe_key": self.dedupe_key,
            "chat_id": int(self.chat_id or 0),
            "group_id": self.group_id,
            "task_id": self.task_id,
            "root_task_id": self.root_task_id,
            "origin_task_id": self.origin_task_id,
            "origin_root_task_id": self.origin_root_task_id,
            "presentation_owner_task_id": self.presentation_owner_task_id,
            "status": self.status,
            "message": self.message,
            "error": self.error,
            "queued_at": self.queued_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "age_seconds": age_seconds,
            "stale": stale,
            "stale_reason": "running_too_long" if stale else "",
            "recovery_hint": (
                "Lifecycle work is still running in-process. Restart Ouroboros to clear "
                "a stuck worker after preserving logs."
                if stale else ""
            ),
        }


@dataclass
class LifecycleJobOptions:
    """Optional lifecycle hooks and formatters kept out of the main API arity."""

    drive_root: pathlib.Path | str | None = None
    result_message: Callable[[Any], str] | None = None
    result_error: Callable[[Any], str] | None = None
    progress_target: Optional["JobProgressTarget"] = None
    on_started: Callable[[LifecycleJob], None] | None = None
    on_finished: Callable[[LifecycleJob, Any, BaseException | None], None] | None = None
    presentation: Dict[str, Any] | None = None


_lock: Optional[threading.Lock] = None
_state_lock = threading.Lock()
_dedupe_jobs: Dict[str, LifecycleJob] = {}
_events: Deque[LifecycleJob] = deque(maxlen=_MAX_EVENTS)
_active: Optional[LifecycleJob] = None


@dataclass
class _LaneHandle:
    """The lane held by the lifecycle job running in this context."""

    job: LifecycleJob
    lock: threading.Lock
    lock_file: Any = None  # open cross-process lock file, or None


# Copied into the runner's worker thread with the context, so the job's own code
# can lend the lane; any other caller sees None and lending is a no-op.
_LANE_HANDLE: "contextvars.ContextVar[Optional[_LaneHandle]]" = contextvars.ContextVar(
    "skill_lifecycle_lane", default=None,
)


@contextlib.contextmanager
def lend_lifecycle_lane():
    """Let other lifecycle jobs run during a long read-only step of this job.

    The caller must not mutate skill state inside the block. The lane is taken back
    (waiting as long as the current holder needs) before the block's caller goes on,
    so whatever follows runs under the lane exactly as before. Outside a lane job
    this is a no-op.
    """
    global _active
    handle = _LANE_HANDLE.get()
    if handle is None:
        yield
        return
    from ouroboros.platform_layer import file_lock_exclusive_nb, file_unlock

    _LANE_HANDLE.set(None)  # a nested lend inside the block is a no-op
    if handle.lock_file is not None:
        file_unlock(handle.lock_file.fileno())
    with _state_lock:
        if _active is handle.job:
            _active = None
    handle.lock.release()
    try:
        yield
    finally:
        # In-process: wait for the holder (the job owns this lock again before its own
        # release runs). Cross-process: the lane's deadline, as when it was first taken.
        handle.lock.acquire()
        if handle.lock_file is not None:
            deadline = time.monotonic() + _lifecycle_deadline_sec()
            while True:
                try:
                    file_lock_exclusive_nb(handle.lock_file.fileno())
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            "skill_lifecycle.lock is held beyond the deadline — a "
                            "previous job appears wedged. Restarting Ouroboros recovers it."
                        )
                    time.sleep(0.05)
        with _state_lock:
            _active = handle.job
        _LANE_HANDLE.set(handle)


class DuplicateLifecycleJobError(RuntimeError):
    """Raised when a caller attempts to queue an already active lifecycle job."""

    def __init__(self, job: LifecycleJob) -> None:
        self.job = job
        super().__init__(f"lifecycle job already {job.status}: {job.kind}:{job.target}")


def _get_lock() -> threading.Lock:
    global _lock
    if _lock is None:
        _lock = threading.Lock()
    return _lock


def _store(job: LifecycleJob) -> None:
    if job not in _events:
        _events.append(job)


def _chat_task_id(job: LifecycleJob) -> str:
    suffix = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(job.target or "skill")).strip("_")
    job_suffix = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(job.id or "")).strip("_")
    return f"skill_lifecycle_{job.kind}_{suffix or 'skill'}_{job_suffix or 'job'}"


def _effective_chat_id(job: LifecycleJob) -> int:
    """The human chat this job reports to: its bound task chat, else panel 0.

    Routed through the ONE notification normalizer (C4): 0 is the Skill Review
    panel — a valid destination, not "no chat" — and a negative id is A2A traffic
    that must never reach a human stream, so an A2A-initiated job reports to the
    panel like an unbound one.
    """
    from supervisor.message_bus import notification_chat_route

    route = notification_chat_route(job.chat_id, 0)
    return int(route if route is not None else 0)


def _notify_chat_progress(job: LifecycleJob, phase: str) -> None:
    try:
        from supervisor.message_bus import send_with_budget

        detail = job.error or job.message or job.status
        lifecycle = job.to_dict()
        lifecycle["phase"] = str(phase or "")
        send_with_budget(
            _effective_chat_id(job),
            f"Skill {job.kind}: `{job.target}` — {phase}{f' — {detail}' if detail else ''}",
            is_progress=True,
            task_id=_chat_task_id(job),
            progress_meta={"lifecycle": lifecycle},
            role="system", system_type="skill_lifecycle")
    except Exception:
        return


def _notify_duplicate_pointer(requested: LifecycleJob, existing: LifecycleJob) -> None:
    """C4 multi-chat dedupe: the FIRST initiator owns the routing; a duplicate
    caller from ANOTHER chat gets a typed pointer ack in its own chat instead of
    silence (or a second progress stream)."""
    # Membership, not truthiness: a PANEL caller (chat 0) is a real initiator and
    # gets its pointer too — `if not requested_chat` silently dropped exactly the
    # duplicate this ack exists for. Only an identical route needs no pointer:
    # that stream already carries the original job's progress.
    requested_chat = _effective_chat_id(requested)
    if requested_chat == _effective_chat_id(existing):
        return
    try:
        from supervisor.message_bus import send_with_budget

        send_with_budget(
            requested_chat,
            f"Skill {existing.kind}: `{existing.target}` — already {existing.status} "
            f"(job {existing.id}); progress reports in its original chat.",
            is_progress=True,
            # A pointer is an acknowledgement to the duplicate caller, not a
            # scheduled task.  Giving it the synthetic lifecycle id promoted
            # this harmless line into a second task card on replay.
            task_id="",
            progress_meta={"lifecycle_pointer": {
                "job_id": existing.id,
                "kind": existing.kind,
                "target": existing.target,
                "status": existing.status,
                "chat_id": _effective_chat_id(existing),
                "group_id": existing.group_id,
                "task_id": existing.task_id,
                "root_task_id": existing.root_task_id,
                "origin_task_id": existing.origin_task_id,
                "origin_root_task_id": existing.origin_root_task_id,
                "presentation_owner_task_id": existing.presentation_owner_task_id,
                "source": existing.source,
            }},
            role="system", system_type="skill_lifecycle_pointer")
    except Exception:
        return


def _lifecycle_deadline_sec() -> float:
    """Upper bound for waiting on the lifecycle lane / running one job.

    A single wedged job used to freeze the whole skill-lifecycle queue forever
    (unbounded join + infinite lock-acquire loops); the deadline converts that
    silent wedge into a loud, attributable failure. Generous default — skill
    reviews legitimately run for many minutes.
    """
    from ouroboros.config import SETTINGS_DEFAULTS

    raw = runtime_setting("OUROBOROS_SKILL_LIFECYCLE_TIMEOUT_SEC", "")
    try:
        parsed = float(raw)
        if parsed > 0:
            return parsed
    except (TypeError, ValueError):
        pass
    # SSOT: the shipped default lives with the setting, not as a second constant here.
    return float(SETTINGS_DEFAULTS["OUROBOROS_SKILL_LIFECYCLE_TIMEOUT_SEC"])


@contextlib.asynccontextmanager
async def _async_thread_lock(lock: threading.Lock):
    deadline = time.monotonic() + _lifecycle_deadline_sec()
    acquired = False
    while not acquired:
        acquired = lock.acquire(blocking=False)
        if not acquired:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "Skill lifecycle lane is busy beyond the deadline — a previous "
                    "job appears wedged. Restarting Ouroboros recovers the lane."
                )
            await asyncio.sleep(0.01)
    try:
        yield
    finally:
        if acquired:
            lock.release()


def _register_dedupe(job: LifecycleJob) -> None:
    if not job.dedupe_key:
        _store(job)
        return
    with _state_lock:
        existing = _dedupe_jobs.get(job.dedupe_key)
        if existing is not None and existing.status in {"queued", "running"}:
            raise DuplicateLifecycleJobError(existing)
        _dedupe_jobs[job.dedupe_key] = job
        _store(job)


def _release_dedupe(job: LifecycleJob) -> None:
    if not job.dedupe_key:
        return
    with _state_lock:
        if _dedupe_jobs.get(job.dedupe_key) is job:
            _dedupe_jobs.pop(job.dedupe_key, None)


@contextlib.asynccontextmanager
async def async_skill_lifecycle_file_lock(drive_root: pathlib.Path):
    """Hold ``state/skill_lifecycle.lock``; yields the open lock file."""
    from ouroboros.platform_layer import file_lock_exclusive_nb, file_unlock

    lock_dir = pathlib.Path(drive_root) / "state"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / "skill_lifecycle.lock"
    deadline = time.monotonic() + _lifecycle_deadline_sec()
    with lock_path.open("a+") as fh:
        acquired = False
        while not acquired:
            try:
                file_lock_exclusive_nb(fh.fileno())
                acquired = True
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "skill_lifecycle.lock is held beyond the deadline — a "
                        "previous job appears wedged. Restarting Ouroboros recovers it."
                    )
                await asyncio.sleep(0.05)
        try:
            yield fh
        finally:
            if acquired:
                file_unlock(fh.fileno())


def _notify_chat(job: LifecycleJob) -> None:
    if job.status not in {"succeeded", "failed", "cancelled"}:
        return
    phase = {
        "succeeded": "completed",
        "cancelled": "cancelled",
    }.get(job.status, "failed")
    _notify_chat_progress(job, phase)


def _job_snapshot(job: LifecycleJob) -> Dict[str, Any]:
    payload = job.to_dict()
    payload["chat_task_id"] = _chat_task_id(job)
    return payload


async def run_lifecycle_job(
    *,
    kind: str,
    target: str,
    runner: Callable[[], Awaitable[Any]],
    source: str = "",
    message: str = "",
    dedupe_key: str = "",
    chat_id: int = 0,
    options: LifecycleJobOptions | None = None,
) -> Any:
    """Run runner through the lifecycle lane with optional progress updates.

    ``chat_id`` binds the job's progress to the initiating task's chat (C4);
    0 (the default) reports to the Skill Review panel chat.
    """

    global _active
    opts = options or LifecycleJobOptions()
    presentation = opts.presentation if isinstance(opts.presentation, dict) else {}
    try:
        _chat = int(chat_id or 0)
    except (TypeError, ValueError):
        _chat = 0
    job = LifecycleJob(
        id=f"skill-job-{uuid.uuid4().hex}",
        kind=str(kind or "operation"),
        target=str(target or "skill"),
        source=str(source or ""),
        dedupe_key=str(dedupe_key or ""),
        message=str(message or ""),
        chat_id=_chat,
        group_id=str(presentation.get("group_id") or ""),
        task_id=str(presentation.get("task_id") or ""),
        root_task_id=str(presentation.get("root_task_id") or ""),
        origin_task_id=str(presentation.get("origin_task_id") or ""),
        origin_root_task_id=str(presentation.get("origin_root_task_id") or ""),
        presentation_owner_task_id=str(
            presentation.get("presentation_owner_task_id") or ""
        ),
    )
    try:
        _register_dedupe(job)
    except DuplicateLifecycleJobError as duplicate:
        _notify_duplicate_pointer(job, duplicate.job)
        raise
    _notify_chat_progress(job, "queued")
    if opts.progress_target is not None:
        opts.progress_target.bind(job)
    result: Any = None
    error_obj: BaseException | None = None
    terminal_notified = False
    try:
        async with _async_thread_lock(_get_lock()):
            _active = job
            job.status = "running"
            job.started_at = _now_iso()
            _notify_chat_progress(job, "running")
            if opts.drive_root is None:
                from ouroboros.config import DATA_DIR

                lock_root = pathlib.Path(DATA_DIR)
            else:
                lock_root = pathlib.Path(opts.drive_root)
            try:
                async with async_skill_lifecycle_file_lock(lock_root) as lock_file:
                    lane = _LANE_HANDLE.set(_LaneHandle(job=job, lock=_get_lock(), lock_file=lock_file))
                    try:
                        if opts.on_started is not None:
                            opts.on_started(job)
                        result = await runner()
                    finally:
                        _LANE_HANDLE.reset(lane)
                error = opts.result_error(result) if opts.result_error else ""
                job.error = str(error or "")
                job.status = "failed" if job.error else "succeeded"
                if opts.result_message:
                    job.message = opts.result_message(result)
                elif not job.message:
                    job.message = job.status
                return result
            except BaseException as exc:
                error_obj = exc
                job.status = "cancelled" if isinstance(exc, asyncio.CancelledError) else "failed"
                job.error = str(exc) or type(exc).__name__
                raise
            finally:
                job.finished_at = _now_iso()
                if opts.on_finished is not None:
                    try:
                        opts.on_finished(job, result, error_obj)
                    except BaseException:
                        log.exception(
                            "skill lifecycle on_finished hook failed for %s:%s",
                            job.kind,
                            job.target,
                        )
                _release_dedupe(job)
                if _active is job:
                    _active = None
                if opts.progress_target is not None:
                    opts.progress_target.release()
                _notify_chat(job)
                terminal_notified = True
    except asyncio.CancelledError:
        job.status = "cancelled"
        job.error = job.error or "CancelledError"
        job.finished_at = job.finished_at or _now_iso()
        _release_dedupe(job)
        if _active is job:
            _active = None
        if opts.progress_target is not None:
            opts.progress_target.release()
        if not terminal_notified:
            _notify_chat(job)
        raise
    except BaseException as exc:
        # Lane-acquisition failures (e.g. the new _async_thread_lock deadline
        # TimeoutError) raise BEFORE the inner finally exists — without this
        # branch the job stayed "queued" in _dedupe_jobs forever, blocking
        # every future job with the same dedupe key.
        if not terminal_notified:
            job.status = "failed"
            job.error = job.error or (str(exc) or type(exc).__name__)
            job.finished_at = job.finished_at or _now_iso()
            _release_dedupe(job)
            if _active is job:
                _active = None
            if opts.progress_target is not None:
                opts.progress_target.release()
            _notify_chat(job)
        raise


async def run_blocking_preserving_cancellation(
    func: Callable[..., Any],
    *args: Any,
    log_label: str = "lifecycle work",
    **kwargs: Any,
) -> Any:
    """Keep the lifecycle lane held until non-killable thread work returns."""

    task = asyncio.create_task(asyncio.to_thread(func, *args, **kwargs))
    warned = False
    while True:
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.done():
                return task.result()
            if not warned:
                log.warning(
                    "Client task was cancelled while %s is still running; "
                    "waiting for the worker thread to finish because Python threads cannot be killed safely.",
                    log_label,
                )
                warned = True
            current = asyncio.current_task()
            if current is not None and hasattr(current, "uncancel"):
                while current.cancelling():  # type: ignore[attr-defined]
                    current.uncancel()  # type: ignore[attr-defined]
            # Python <=3.10: next shield await blocks until worker finishes.


def run_lifecycle_job_blocking(
    *,
    kind: str,
    target: str,
    runner: Callable[[], Any],
    source: str = "",
    message: str = "",
    dedupe_key: str = "",
    chat_id: int = 0,
    options: LifecycleJobOptions | None = None,
) -> Any:
    """Run a lifecycle job from a synchronous tool handler.

    Tool handlers already run outside the Starlette event loop. This wrapper
    gives them the same lifecycle lane, dedupe, and notifications as HTTP
    handlers without making each caller manage an event loop.
    """

    async def _runner() -> Any:
        return await run_blocking_preserving_cancellation(
            runner,
            log_label=f"blocking {kind} lifecycle operation",
        )

    async def _main() -> Any:
        return await run_lifecycle_job(
            kind=kind,
            target=target,
            source=source,
            message=message,
            dedupe_key=dedupe_key,
            chat_id=chat_id,
            runner=_runner,
            options=options,
        )

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_main())

    box: Dict[str, Any] = {}

    def _thread_main() -> None:
        try:
            box["result"] = asyncio.run(_main())
        except BaseException as exc:
            box["error"] = exc

    settings_context = contextvars.Context()
    copy_task_settings_context(settings_context)
    thread = threading.Thread(target=settings_context.run, args=(_thread_main,), name=f"skill-lifecycle-{kind}", daemon=False)
    thread.start()
    thread.join(timeout=_lifecycle_deadline_sec())
    if thread.is_alive():
        # The job thread is wedged. We cannot kill it safely (it may hold the
        # cross-process flock, which the OS releases only on process exit), but
        # the caller must not hang forever pretending the job is progressing.
        raise TimeoutError(
            f"Skill lifecycle job '{kind}:{target}' exceeded its deadline and is "
            "wedged; the lifecycle lane stays blocked until restart."
        )
    if "error" in box:
        raise box["error"]
    return box.get("result")


class JobProgressTarget:
    """Thread-safe relay for worker progress into the active lifecycle job."""

    __slots__ = ("_job", "_done")

    def __init__(self) -> None:
        self._job: Optional[LifecycleJob] = None
        self._done = False

    def bind(self, job: LifecycleJob) -> None:
        self._job = job

    def release(self) -> None:
        self._done = True

    def set(self, message: str) -> None:
        if self._done or self._job is None:
            return
        self._job.message = str(message or "")
        _notify_chat_progress(self._job, "running")


class JobHandle(JobProgressTarget):
    """Progress relay that also tells the request which lifecycle job it started.

    A blocking job binds from its own thread's loop; the ``bound`` event is set on the
    creating loop either way, so a request can await it.
    """

    __slots__ = ("job_id", "bound", "_loop")

    def __init__(self) -> None:
        super().__init__()
        self.job_id = ""
        self.bound = asyncio.Event()
        try:
            self._loop: Optional[asyncio.AbstractEventLoop] = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None

    def bind(self, job: LifecycleJob) -> None:
        super().bind(job)
        self.job_id = str(job.id)
        loop = self._loop
        try:
            same_loop = loop is None or asyncio.get_running_loop() is loop
        except RuntimeError:
            same_loop = loop is None
        if same_loop:
            self.bound.set()
        else:
            loop.call_soon_threadsafe(self.bound.set)

    @property
    def message(self) -> str:
        return str(getattr(self._job, "message", "") or "")


def _log_background_failure(task: "asyncio.Task") -> None:
    if not task.cancelled() and task.exception() is not None:
        log.warning("answered-pending skill action failed", exc_info=task.exception())


async def answer_or_job(work: Callable[[JobHandle], Any], *, after: Callable[[Any], None] | None = None,
                        log_label: str = "skill lifecycle operation", allow_pending: bool = False) -> Any:
    """Run ``work(handle)``, a blocking owner action that binds one lifecycle job to ``handle``, off
    the loop. Legacy callers await the terminal payload. An explicitly pending-aware caller gets
    that payload within ``JOB_ANSWER_WAIT_SEC``, else ``{ok, pending, job_id, message}`` while work
    finishes. ``after(payload)`` still runs on completion (the lifecycle broadcast).
    """
    handle = JobHandle()

    async def _run() -> Any:
        payload = await run_blocking_preserving_cancellation(work, handle, log_label=log_label)
        if after is not None:
            try:
                after(payload)
            except Exception:
                log.debug("after-hook of %s failed", log_label, exc_info=True)
        return payload

    if not allow_pending:
        return await _run()
    task = asyncio.ensure_future(_run())
    _BACKGROUND.add(task)
    task.add_done_callback(_BACKGROUND.discard)
    done, _ = await asyncio.wait({task}, timeout=JOB_ANSWER_WAIT_SEC)
    if task not in done and not handle.job_id:
        # Validation can finish without ever creating a job. Never advertise an empty job id.
        bound = asyncio.create_task(handle.bound.wait())
        try:
            done, _ = await asyncio.wait({task, bound}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            bound.cancel()
    if task in done:
        return task.result()
    task.add_done_callback(_log_background_failure)
    return {"ok": True, "pending": True, "job_id": handle.job_id, "message": handle.message}


def queue_snapshot() -> Dict[str, Any]:
    """Return a JSON-friendly view of recent lifecycle activity."""

    return {
        "active": _job_snapshot(_active) if _active else None,
        "events": [_job_snapshot(job) for job in list(_events)],
    }
