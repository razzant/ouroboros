"""Task-local custody of model quota/auth waits and role-specific continuation.

The existing task owns its worker, writer lane, mailbox and durable result.
This module keeps only the live call's wait and role overrides. It neither
schedules work nor records physical attempts: every resumed call still goes
through LLMClient and the ordinary physical-attempt ledger.

Resource recovery order (quota, sign-in, an engine-dated unavailable pool): the
transport's Auto account rotation, then every configured route of the round,
and only then the owner. A round whose later route exists returns its refusal
instead of waiting (``ResourceDeferral``); the owner question is then opened
from that retained refusal, never from another generation, and nothing sleeps
to a reset. An inline Presence turn never waits for a resource or the owner.

It also owns the one execution clock (#939) every finite execution bound reads:
the narrower of the global ceiling and a task's API/CLI ``execution_timeout_sec``
(``config.resolve_task_execution_timeout_sec``). Queue before the first dispatch,
the confirmed quota-wait union, budget/owner Pause and model sleep are excluded
(warm sleep live via ``TaskModelWait.sleep_started_monotonic``; a cold sleep and a
bounded task's exact Resume from grant to consumption via ``sleep_exclusion_since``);
tools and their result waits count, and calendar deadlines never move. A
continuation that preserves the original start already spans earlier attempts;
only a fresh-start successor carries their total as ``_execution_consumed_sec``,
decided once where assignment and the worker choose the origin
(``settle_execution_carry``). Recovery keeps confirmed quota facts newer than
its checkpoint (``fold_running_clock``), and only the author's own wait owner
moves the supervisor's RUNNING clock.
"""

from __future__ import annotations

import contextlib
import contextvars
import concurrent.futures
import copy
import functools
import inspect
import json
import math
import pathlib
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

from ouroboros.usage_accounting import current_usage_scope
from ouroboros.task_store import patch_handoff, put_handoffs
from ouroboros.utils import utc_now_iso


class ModelWaitInterrupted(RuntimeError):
    """A live model wait ended on the task's existing control/deadline rail."""

    code = "model_operation_interrupted"

    def __init__(self, reason: str, *, role: str = "", cause: Exception | None = None):
        super().__init__(f"Model wait interrupted: {reason}")
        self.control_reason = reason
        self.model_role = role
        self.previous_error = cause
        if cause is not None:
            for name in ("physical_attempt_capture", "ledger_attempt_ids", "model_result", "usage",
                         "model_role_route", "operation_id", "route", "stream_incomplete", "stream_rejected"):
                if hasattr(cause, name):
                    setattr(self, name, getattr(cause, name))


def propagate_model_control(error: Exception, *, role: str = "") -> None:
    """One typed host interruption, whether raised by the live wait or transport."""
    if isinstance(error, ModelWaitInterrupted):
        raise error
    if getattr(error, "code", "") == "model_operation_interrupted" and getattr(error, "control_reason", ""):
        raise ModelWaitInterrupted(error.control_reason, role=getattr(error, "model_role", "") or role,
                                   cause=error) from error


def model_wait_reason(error: Exception) -> str:
    """Only confirmed resource causes authorize this live waiting contract.

    The engine's typed mixed pool means auth plus quota, excluding unknown
    readiness, disabled accounts and model incompatibility. A dated generic pool
    refusal is ``unavailable`` for this request; its reset forecast proves neither
    quota exhaustion nor unavailability until that instant. An undated pool failure
    proves none of those facts and keeps its ordinary error path.
    """
    authored = getattr(error, "task_resource_wait_reason", "")
    if authored in {"auth", "quota"}:
        return authored  # confirmed by the registered caller's refusal classifier
    code = getattr(error, "code", "")
    if code in {"auth_required", "subscription_window_exhausted"}:
        return "auth" if code == "auth_required" else "quota"
    if code == "connection_unavailable":  # the route's connection is gone or disabled: typed, $0, owner's call
        return "connection_unavailable"
    problem = getattr(error, "problem", None)
    context = problem.get("context") if isinstance(problem, dict) else None
    if code == "credential_pool_exhausted" and isinstance(context, dict) and context.get("poolCause") == "mixed":
        return "auth_quota"
    if code == "credential_pool_exhausted" and str(getattr(error, "reset_at", "") or "").strip():
        return "unavailable"
    return ""


@dataclass
class PreparedModelCall:
    """One caller-reprepared send and its existing Main fit authority."""

    kwargs: dict
    physical_context: Any
    candidate_predicate: Any


@contextlib.contextmanager
def prepared_call_scope(preparation: dict | PreparedModelCall) -> Iterator[dict]:
    if isinstance(preparation, PreparedModelCall):
        from ouroboros.usage_accounting import bind_physical_attempt_context

        with bind_physical_attempt_context(preparation.physical_context, preparation.candidate_predicate):
            yield preparation.kwargs
    else:
        yield preparation


@dataclass
class _QuotaClock:
    """Union duration, not the sum of concurrent waits."""

    active: set[str] = field(default_factory=set)
    started: float | None = None
    elapsed: float = 0.0

    def enter(self, wait_id: str, now: float) -> None:
        if not self.active:
            self.started = now
        self.active.add(wait_id)

    def leave(self, wait_id: str, now: float) -> None:
        if wait_id not in self.active:
            return
        self.active.remove(wait_id)
        if not self.active and self.started is not None:
            self.elapsed += max(0.0, now - self.started)
            self.started = None

    def duration(self, now: float) -> float:
        return self.elapsed + (max(0.0, now - self.started) if self.started is not None else 0.0)


def quota_waited_seconds(meta: dict, now: float) -> float:
    """Read one union-clock snapshot, never sum the parallel waiting rows."""
    clock = meta.get("model_wait_quota_clock") or {}
    try:
        elapsed = float(clock.get("elapsed_sec") or 0.0)
        observed = float(clock.get("observed_at") or now)
        if not math.isfinite(elapsed) or not math.isfinite(observed):
            return 0.0
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, elapsed) + (max(0.0, now - observed) if clock.get("active") is True else 0.0)


def budget_paused_seconds(meta: dict) -> float:
    """The ONE budget-paused carrier (#1196), read from a RUNNING row, a resume
    handoff or a pause row alike: wall time a task spent PAUSED, which is not
    execution and is NOT part of the quota union (that clock stays its own)."""
    try:
        paused = float(meta.get("budget_paused_sec") or meta.get("paused_duration_sec") or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, paused) if math.isfinite(paused) else 0.0


def execution_elapsed_seconds(meta: dict, now: float) -> float:
    """Execution time of one task: the carried pre-origin offset plus wall clock
    minus the quota-wait union minus the separate budget-paused interval.
    ``started_at`` is never moved, so every finite-lifetime consumer (supervisor
    timeouts, owner stop, the exact-pause grant, the live wait controls)
    subtracts the same carriers and adds the same offset."""
    offset = execution_consumed_seconds(meta)
    try:
        started = float(meta.get("started_at") or 0.0)
    except (TypeError, ValueError):
        return offset
    if started <= 0 or not math.isfinite(started):
        return offset
    sleeping = meta.get("sleep_parked_at")  # a warm sleep or owner Pause (worker_owner_wait)
    return offset + max(0.0, now - started - quota_waited_seconds(meta, now) - budget_paused_seconds(meta)
                        - (max(0.0, now - float(sleeping)) if isinstance(sleeping, (int, float)) else 0.0))


def execution_consumed_seconds(meta: dict) -> float:
    """Execution spent BEFORE this clock's origin (#939), never queue time.

    A fresh-origin successor or a pre-body Pause carries one (``settle_execution_carry``);
    a preserved original start already spans
    every earlier attempt, so its offset never grows. Reads a RUNNING row's
    ``task`` or a task row itself.
    """
    task = meta.get("task") if isinstance(meta.get("task"), dict) else meta
    try:
        value = float(task.get("_execution_consumed_sec") or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, value) if math.isfinite(value) else 0.0


def stamp_execution_handoff(successor: dict, meta: dict | None, now: float | None = None) -> None:
    """Record an ended attempt's total execution on its successor (#939).

    At assignment, ``settle_execution_carry`` keeps this total: a preserved
    origin excludes the intervening queue via the existing paused carrier;
    a fresh origin carries the total as an offset. A quota wait
    still open at the last observation stops there, never extended further:
    the raw clock facts are read, never a reader's derived ``execution_sec``.
    """
    from ouroboros.runtime_limits import explicit_execution_timeout_sec

    successor.pop("_execution_handoff_sec", None)
    if explicit_execution_timeout_sec(successor) is None or not isinstance(meta, dict) or not meta:
        return
    clock = meta.get("model_wait_quota_clock")
    closed = {**meta, "model_wait_quota_clock": {**clock, "active": False}} if isinstance(clock, dict) else meta
    successor["_execution_handoff_sec"] = execution_elapsed_seconds(closed, time.time() if now is None else now)


def native_sleep_started_at(handoff: Any) -> float | None:
    """The selected native sleep's recorded open start; absent/invalid is unknown."""
    started = handoff.get("sleep_started_at") if isinstance(handoff, dict) and handoff.get("reason") == "sleep" else None
    return float(started) if (isinstance(started, (int, float)) and not isinstance(started, bool)
                             and math.isfinite(started) and started > 0) else None


def settle_execution_carry(task: dict, *, origin_preserved: bool) -> float:
    """Settle the ended total once at handoff, excluding queued time for #939."""
    handoff = task.pop("_execution_handoff_sec", None)
    if handoff is not None:
        total = execution_consumed_seconds({"_execution_consumed_sec": handoff})
        if not origin_preserved:
            task["_execution_consumed_sec"] = total
        else:
            from ouroboros.runtime_limits import explicit_execution_timeout_sec

            if explicit_execution_timeout_sec(task) is not None:
                key = next(key for key in ("_owner_wait_resume", "_budget_pause_resume", "_working_recovery")
                           if task.get(key))
                resume = dict(task[key])
                # A returning native sleep already owns the interval across stop/boot;
                # it cannot also become a closed queue exclusion at this assignment.
                sleeping = native_sleep_started_at(resume) if key == "_owner_wait_resume" else None
                queued = max(0.0, execution_elapsed_seconds(
                    {**resume, "task": task, "sleep_parked_at": sleeping}, time.time()) - total)
                resume["budget_paused_sec"] = budget_paused_seconds(resume) + queued
                task[key] = resume
    return execution_consumed_seconds(task)


def fold_running_clock(task: dict, meta: dict) -> None:
    """Keep exclusions confirmed after the saved checkpoint (#939).

    A working-recovery handoff carries the cognition checkpoint's clock; the
    supervisor's RUNNING row may hold a newer confirmed quota revision or paused
    carrier of the SAME clock (same attempt, or the same preserved origin).
    The newer fact wins, closed at its own observation; nothing is inferred.
    """
    handoff = task.get("_working_recovery")
    if not isinstance(handoff, dict) or not isinstance(meta, dict):
        return
    attempt = int(meta.get("attempt") or (meta.get("task") or {}).get("_attempt") or 1)
    try:
        same_origin = float(handoff.get("started_at") or 0.0) == float(meta.get("started_at") or -1.0)
    except (TypeError, ValueError):
        same_origin = False
    if int(handoff.get("from_attempt") or 0) != attempt and not same_origin:
        return
    clock, saved = meta.get("model_wait_quota_clock"), handoff.get("model_wait_quota_clock") or {}
    if isinstance(clock, dict) and int(clock.get("revision") or 0) > int(saved.get("revision") or 0):
        handoff["model_wait_quota_clock"] = {**clock, "active": False}
    if budget_paused_seconds({"budget_paused_sec": meta.get("budget_paused_sec")}) > budget_paused_seconds(handoff):
        handoff["budget_paused_sec"] = budget_paused_seconds({"budget_paused_sec": meta.get("budget_paused_sec")})


def adopt_handoff_clock(model_state: dict, handoff: Any) -> dict:
    """The worker side of ``fold_running_clock``: on a preserved origin, the handoff's newer
    folded facts outrank the checkpoint's own; a fresh origin keeps the checkpoint reset."""
    if not isinstance(handoff, dict) or not handoff.get("started_at"):
        return model_state
    state, newer = dict(model_state), handoff.get("model_wait_quota_clock")
    if isinstance(newer, dict) and int(newer.get("revision") or 0) > int((state.get("quota_clock") or {}).get("revision") or 0):
        state["quota_clock"] = dict(newer)
    if budget_paused_seconds(handoff) > budget_paused_seconds({"budget_paused_sec": state.get("budget_paused_sec")}):
        state["budget_paused_sec"] = budget_paused_seconds(handoff)
    return state


def explicit_execution_window_remaining(ctx: Any) -> float | None:
    """The remaining window only when the task carries an explicit execution budget, so
    pacing, finalize and tool clamps never invent one from the global ceiling alone."""
    from ouroboros.config import explicit_execution_timeout_sec

    meta = getattr(ctx, "task_metadata", None)
    if not isinstance(meta, dict) or explicit_execution_timeout_sec(meta) is None:
        return None
    return execution_window_remaining_for_context(ctx)


def execution_window_remaining_for_context(ctx: Any) -> float | None:
    """Remaining execution window: the bound live owner's clock, else the same
    arithmetic over the context's original start and carriers. ``None`` = no bound."""
    owner = getattr(ctx, "model_wait_context", None) or current_model_wait()
    if owner is not None and owner.task_id == str(getattr(ctx, "task_id", "") or ""):
        return owner.execution_window_remaining()
    from ouroboros.config import resolve_task_execution_timeout_sec

    meta = getattr(ctx, "task_metadata", None)
    meta = meta if isinstance(meta, dict) else {}
    ceiling = resolve_task_execution_timeout_sec(meta)
    if ceiling is None:
        return None
    paused = getattr(ctx, "_budget_paused_sec", None)
    if paused is None:
        paused = budget_paused_seconds(getattr(ctx, "budget_pause_resume", None) or {})
    elapsed = execution_elapsed_seconds({**meta, "started_at": getattr(ctx, "task_started_at", None),
                                         "budget_paused_sec": paused}, time.time())
    return max(0.0, ceiling - elapsed)


_CURRENT: contextvars.ContextVar[TaskModelWait | None] = contextvars.ContextVar(
    "ouroboros_model_wait", default=None)
_STRICT_CONTROL_PROBE = contextvars.ContextVar("strict_presend_control_probe", default=None)


def control_probe_requires_current() -> bool:
    """Only an unsent admission needs a temporary control-read gap to reach its wait owner."""
    return _STRICT_CONTROL_PROBE.get() is True


def control_probe_active() -> bool:
    return _STRICT_CONTROL_PROBE.get() is not None
_REPREPARE: contextvars.ContextVar[dict[str, tuple[Callable, Callable | None]] | None] = contextvars.ContextVar(
    "ouroboros_model_wait_reprepare", default=None)
_CALENDAR: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar(
    "ouroboros_model_wait_calendar", default=())
_LOGICAL: contextvars.ContextVar[tuple[tuple[float, str], ...]] = contextvars.ContextVar(
    "ouroboros_model_wait_logical", default=())
# One call's own scope, like its physical capture: never copied into a helper's context.
_DEFERRED: contextvars.ContextVar[tuple["ResourceDeferral", ...]] = contextvars.ContextVar(
    "ouroboros_model_wait_deferred", default=())


def copy_wait_context() -> contextvars.Context:
    """Carry wait ownership into an otherwise isolated worker's existing scope.

    Copying every ContextVar also transfers a previous physical capture and
    the parent's Main fit authority. Those belong to their original call.
    """
    from ouroboros.settings_integrity import copy_task_settings_context

    copied = contextvars.Context()
    copy_task_settings_context(copied)
    for variable in (_CURRENT, _REPREPARE, _CALENDAR, _LOGICAL):
        copied.run(variable.set, variable.get())
    return copied


def independent_turn_context() -> contextvars.Context:
    """The context an initiated independent turn starts from (#1536).

    Like ``copy_wait_context`` it starts empty: the initiating call's physical capture,
    usage scope, send clock, tool-result slot and billing frame stay with that call.
    Settings and explicit calendar deadlines carry; the initiating operation's own
    execution deadline, wait owner and re-preparation callbacks do not: the new turn's
    task frame installs its own controls, and a tool's call bound is not its lifetime.
    """
    from ouroboros.settings_integrity import copy_task_settings_context

    copied = contextvars.Context()
    copy_task_settings_context(copied)
    copied.run(_CALENDAR.set, _CALENDAR.get())
    return copied


@contextlib.contextmanager
def execution_deadline_scope(deadline: float, *, review_slot_id: str | None = None) -> Iterator[None]:
    """The inner waiter and its existing caller share one execution deadline."""
    scope = current_usage_scope()
    slot = review_slot_id if review_slot_id is not None else str(getattr(scope, "review_slot_id", "") or "")
    token = _LOGICAL.set((*_LOGICAL.get(), (deadline, slot)))
    try:
        yield
    finally:
        _LOGICAL.reset(token)


@contextlib.contextmanager
def calendar_scope(deadline_at: str) -> Iterator[None]:
    """Carry a narrower explicit caller deadline through its helper threads."""
    token = _CALENDAR.set((*_CALENDAR.get(), deadline_at) if deadline_at else _CALENDAR.get())
    try:
        yield
    finally:
        _CALENDAR.reset(token)


def dispatch_deadline_remaining_sec() -> float | None:
    """Read inherited calendar and quota-adjusted execution bounds, without a floor."""
    from ouroboros.deadline_utils import seconds_until

    remaining = [value for bound in _CALENDAR.get()
                 if (value := seconds_until(bound)) is not None]
    remaining.extend(max(0.0, deadline - monotonic_now(slot))
                     for deadline, slot in _LOGICAL.get())
    return min(remaining) if remaining else None


class ResourceDeferral:
    """One round's resource refusal, returned to the round instead of waited in its call.

    The round tries its configured routes first. The owner question is then this
    refusal's own wait, opened from the retained call: catalog checks only, so no
    generation precedes the owner's answer. A turn that may not wait keeps the fact.
    """

    def __init__(self, role: str):
        self.role = role
        self.fact: dict = {}
        self._retained: tuple | None = None

    def retain(self, receiver: Any, error: Exception, values: dict) -> None:
        self._retained = (receiver, error, values)
        self.fact = {"reason": model_wait_reason(error), "role": self.role, "model": str(values.get("model") or ""),
                     "reset_at": str(getattr(error, "reset_at", "") or ""),
                     "account_rotation": copy.deepcopy(getattr(error, "account_rotation", None))}

    def ask_owner(self, context: "TaskModelWait") -> dict:
        receiver, error, values = self._retained
        return context.wait(receiver, error, values)

    def terminal(self, *, fallbacks_tried: list, owner_wait: str) -> dict:
        """The typed temporary non-success when no route recovered it and no owner wait follows."""
        return {**self.fact, "fallbacks_tried": list(fallbacks_tried), "owner_wait": owner_wait, "temporary": True}


def _deferral(role: str) -> ResourceDeferral | None:
    return next((item for item in reversed(_DEFERRED.get()) if item.role == role), None)


def mutate_wait(root: Any, task_id: str, wait_id: str, transform: Callable) -> dict:
    """Mutate one wait row of the task's ``model_waits`` projection (a row write, never the body)."""
    result: dict = {}

    def update(previous):
        if previous is not None and not isinstance(previous, dict):
            raise ValueError("model wait projection is malformed")
        next_row = transform(copy.deepcopy(previous))
        result.update(previous or {} if next_row is None else next_row)
        return next_row

    try:
        patch_handoff(pathlib.Path(root), task_id, "model_waits", wait_id, update)
    except LookupError as exc:
        raise ValueError("model wait requires its lifecycle owner's task result") from exc
    return result


def mutate_live_wait(owner: "TaskModelWait", wait_id: str, transform: Callable) -> dict:
    """Use the live owner's existing rows, retaining the wait loop's row identity."""
    with owner.lock:
        previous = owner.waits.get(wait_id)
        value = transform(copy.deepcopy(previous))
        if value is not None:
            row = owner.waits.setdefault(wait_id, {})
            row.clear()
            row.update(value)
        return copy.deepcopy(owner.waits.get(wait_id) or {})


class TaskModelWait:
    """One live task's shared wait controls; copied contexts share this object."""

    def __init__(self, *, task: dict, drive_root: Any, event_queue: Any,
                 worker_slot_held: bool, row_mutator: Callable | None = None,
                 rows_reader: Callable | None = None, owner_control: Callable | None = None):
        from ouroboros.platform_layer import process_start_time
        import os

        self.task = task
        self.drive_root = drive_root
        metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
        self.canonical_root = task.get("budget_drive_root") or metadata.get("budget_drive_root") or drive_root
        self.task_id = str(task.get("id") or "")
        self.attempt = int(task.get("_attempt") or 1)
        self.event_queue = event_queue
        self.worker_slot_held = worker_slot_held
        self.row_mutator, self.rows_reader, self.owner_control = row_mutator, rows_reader, owner_control
        self.owner_id = str(task.get("model_wait_owner_id") or "")
        self.tool_context = None
        self.lock = threading.RLock()
        self.closed = False
        self.answer_consumer_id = uuid.uuid4().hex
        self.answer_consumer_bound = False
        self.answer_owner_birth = process_start_time(os.getpid())  # outside the dispatch/launch locks
        self.overrides: dict[str, dict] = {}
        self.waits: dict[str, dict] = {}
        self.clocks: dict[str, _QuotaClock] = {"": _QuotaClock()}
        self.started_monotonic = time.monotonic()
        # The SAME budget-paused carrier the queue puts on the RUNNING row
        # (#1196): a resumed task's finite lifetime excludes the paused wall
        # time while its original start stays where it was.
        self.budget_paused_sec = budget_paused_seconds(
            task.get("_budget_pause_resume") if isinstance(task.get("_budget_pause_resume"), dict) else {})
        self.revision = 0
        self.auto_continue: dict[str, bool] = {}
        self.seen_controls: set[str] = set()
        self.mailbox_stamp = None
        resume = task.get("_budget_pause_resume") or {}
        sleep_since = resume.get("sleep_exclusion_since")
        self.sleep_started_monotonic: float | None = (
            time.monotonic() - max(0.0, time.time() - float(sleep_since)) if sleep_since else None)
        # Public facts every new row of this owner carries (a review operation
        # names its exact controller); the waiting slot is stamped per row.
        self.row_facts: dict[str, dict] = {}

    @property
    def waits_allowed(self) -> bool:
        """An inline Presence turn holds its conversation slot with no owner at the
        computer: rotation and fallback still run, then a typed temporary refusal."""
        return not self.task.get("_presence_turn")

    def mutate_row(self, wait_id: str, transform: Callable) -> dict:
        if self.row_mutator is not None:
            return self.row_mutator(wait_id, transform)
        return mutate_wait(self.canonical_root, self.task_id, wait_id, transform)

    def read_rows(self) -> dict:
        if self.rows_reader is not None:
            return self.rows_reader()
        from ouroboros.task_results import load_task_result
        return (load_task_result(self.canonical_root, self.task_id, strict=True) or {}).get("model_waits", {})

    def snapshot(self) -> dict:
        """Content-free live projection; internal controls never become UI state."""
        with self.lock:
            return {"model_wait_owner_id": self.owner_id,
                    **({"chat_id": self.task.get("chat_id")} if self.owner_id else {}), "model_waits": {
                key: {k: copy.deepcopy(v) for k, v in row.items() if not k.startswith("_")}
                for key, row in self.waits.items()}}

    def executed_seconds(self, *, now: float | None = None) -> float:
        """Live execution time: the carried pre-origin offset plus elapsed minus the
        quota union minus budget pause, minus a warm sleep or owner Pause
        (``model_sleep`` folds it into the paused carrier when the task runs again,
        so it is never subtracted twice)."""
        stamp = time.monotonic() if now is None else now
        sleeping = self.sleep_started_monotonic
        return execution_consumed_seconds(self.task) + max(
            0.0, stamp - self.started_monotonic - self.paused_seconds(now=stamp) - self.budget_paused_sec
            - (max(0.0, stamp - sleeping) if sleeping is not None else 0.0))

    def execution_window_remaining(self) -> float | None:
        """A custom live owner supplies its own clock and a task without an execution
        bound has none; None invents no deadline, and 0.0 means the window is spent."""
        if self.owner_control is not None:
            return None
        from ouroboros.config import resolve_task_execution_timeout_sec
        ceiling = resolve_task_execution_timeout_sec(self.task)
        if ceiling is None:
            return None
        return max(0.0, ceiling - self.executed_seconds())

    def quota_clock_snapshot(self) -> dict:
        """The same task-wide clock fact for live publication and continuation."""
        with self.lock:
            clock = self.clocks[""]
            return {"revision": self.revision, "elapsed_sec": clock.duration(time.monotonic()),
                    "observed_at": time.time(), "active": bool(clock.active)}

    def continuation_state(self) -> dict:
        """Keep completed-call choices, accrued quota time and the paused carrier.

        The budget-paused interval (#1196) rides EVERY same-ID continuation, not
        only a budget one: a task that was paused and later parks in an owner
        wait must resume on the same execution clock, or its planned restart
        would count the paused wall time as execution and shrink the finite
        lifetime it already spent (F5). Live waiters are never carried.
        """
        with self.lock:
            return {"overrides": copy.deepcopy(self.overrides),
                    "auto_continue": dict(self.auto_continue),
                    "budget_paused_sec": self.budget_paused_sec,
                    "quota_clock": {**self.quota_clock_snapshot(), "active": False}}

    def restore_continuation(self, saved: dict, *, started_at: float | None,
                             budget_paused_sec: float | None = None) -> None:
        """Rebind one fresh task owner before Runtime context or new model work.

        ``started_at`` is the ORIGINAL start, so the restored wall clock also
        spans any budget pause; ``budget_paused_sec`` carries that interval
        separately (#1196) and is subtracted by every finite-lifetime read.
        ``None`` keeps whatever the task row already supplied.
        """
        with self.lock:
            self.overrides = copy.deepcopy(saved.get("overrides") or {})
            self.auto_continue = dict(saved.get("auto_continue") or {})
            clock = saved.get("quota_clock") or {}
            self.revision = int(clock.get("revision") or 0)
            elapsed = quota_waited_seconds({"model_wait_quota_clock": clock}, time.time())
            self.clocks = {"": _QuotaClock(elapsed=elapsed)}
            if budget_paused_sec is None:
                # The carrier the serializer saved: an owner-wait continuation of
                # a task that had been budget-paused keeps the SAME paused
                # interval across its planned restart (#1196, F5).
                budget_paused_sec = saved.get("budget_paused_sec")
            if budget_paused_sec is not None:
                self.budget_paused_sec = budget_paused_seconds(
                    {"budget_paused_sec": budget_paused_sec})
            if started_at:
                self.started_monotonic = time.monotonic() - max(0.0, time.time() - float(started_at))

    @contextlib.contextmanager
    def register_reprepare(self, role: str, callback: Callable[[dict], dict], *,
                           resource_reason: Callable[[Exception], str] | None = None) -> Iterator[None]:
        """Bind one caller's reprepare and optional confirmed-refusal classification."""
        bindings = dict(_REPREPARE.get() or {})
        bindings[role] = (callback, resource_reason)
        token = _REPREPARE.set(bindings)
        try:
            yield
        finally:
            _REPREPARE.reset(token)

    @contextlib.contextmanager
    def defer_resource_wait(self, role: str) -> Iterator[ResourceDeferral]:
        """A later route of this round, or a turn that may not wait, owns this role's refusal."""
        deferral = ResourceDeferral(role)
        token = _DEFERRED.set((*_DEFERRED.get(), deferral))
        try:
            yield deferral
        finally:
            _DEFERRED.reset(token)

    def paused_seconds(self, slot_id: str = "", *, now: float | None = None) -> float:
        with self.lock:
            clock = self.clocks.get(slot_id)
            return clock.duration(time.monotonic() if now is None else now) if clock is not None else 0.0

    def quota_enter(self, wait_id: str, slot_id: str, *, now: float | None = None) -> None:
        with self.lock:
            stamp = time.monotonic() if now is None else now
            for key in dict.fromkeys(("", slot_id)):
                self.clocks.setdefault(key, _QuotaClock()).enter(wait_id, stamp)

    def quota_leave(self, wait_id: str, slot_id: str, *, now: float | None = None) -> None:
        with self.lock:
            stamp = time.monotonic() if now is None else now
            for key in dict.fromkeys(("", slot_id)):
                clock = self.clocks.get(key)
                if clock is not None:
                    clock.leave(wait_id, stamp)

    def reprepare(self, role: str, kwargs: dict) -> dict | PreparedModelCall:
        """A route change must rebind any already-prepared Main fit authority."""
        binding = (_REPREPARE.get() or {}).get(role)
        callback = binding[0] if binding else None
        if callback is not None:
            return callback(copy.deepcopy(kwargs))
        from ouroboros.usage_accounting import current_physical_attempt_context

        if current_physical_attempt_context() is not None:
            raise ModelWaitInterrupted("model_wait_reprepare_required", role=role)
        return kwargs

    def control_reason(self, *, strict_current: bool = False) -> str | None:
        from ouroboros.store import acquisition_budget
        token = _STRICT_CONTROL_PROBE.set(strict_current)
        try:
            with acquisition_budget(0):
                return self._control_reason(strict_current=strict_current)
        finally:
            _STRICT_CONTROL_PROBE.reset(token)

    def _control_reason(self, *, strict_current: bool) -> str | None:
        """Existing task controls and explicit calendar bounds, never model prose."""
        from ouroboros.cancel_intents import cancel_pending, resolve_owner_stop_intent
        from ouroboros.config import resolve_task_execution_timeout_sec
        from ouroboros.deadline_utils import seconds_until
        from ouroboros.owner_mailbox import KIND_FINALIZE_NOW, drain_owner_entries
        from ouroboros.store import StoreBusy

        if self.closed:
            return "task_ended"
        control_gap = None
        if self.owner_control is None:
            try:
                if cancel_pending(pathlib.Path(self.canonical_root), self.task_id, propagate_busy=True):
                    _, graceful = resolve_owner_stop_intent(self.canonical_root, self.task_id, propagate_busy=True)
                    if not graceful:
                        return "cancelled"
            except StoreBusy as exc:
                control_gap = exc  # No wait here; known Stop wins, then clocks, then an admission gap.
        metadata = getattr(self.tool_context, "task_metadata", None)
        if not isinstance(metadata, dict):
            metadata = self.task.get("metadata") or {}
        contract = self.task.get("task_contract") or {}
        deadlines = (*_CALENDAR.get(), self.task.get("deadline_at"), metadata.get("deadline_at"), contract.get("deadline_at"))
        if any(seconds_until(value) == 0.0 for value in deadlines if value):
            return "deadline"
        if any(monotonic_now(slot) >= deadline for deadline, slot in _LOGICAL.get()):
            return "execution_deadline"
        if self.owner_control is not None:
            return self.owner_control()
        ceiling = resolve_task_execution_timeout_sec(self.task)  # None = unlimited; 0 = exhausted
        if ceiling is not None and self.executed_seconds() >= ceiling:
            return "absolute_ceiling"
        # The loop owns delivery. A private seen copy leaves that ownership
        # intact and excludes an already-drained or superseded stop control.
        from ouroboros.outcomes import REASON_OWNER_REQUESTED_FINALIZATION
        from supervisor.owner_stop import _owner_stop_control_is_current

        seen = set(getattr(self.tool_context, "_loop_mailbox_seen_ids", ()) or ())
        entries = drain_owner_entries(pathlib.Path(self.drive_root), self.task_id, seen,
                                      kinds={KIND_FINALIZE_NOW})
        for entry in entries:
            reason = str(entry.get("text") or "").splitlines()[0].strip()
            if reason != REASON_OWNER_REQUESTED_FINALIZATION or _owner_stop_control_is_current(
                    self.tool_context, self.canonical_root, self.task_id, entry.get("msg_id", "")):
                return "finalize_requested"
        if strict_current and control_gap is not None:
            raise control_gap  # only after every control readable here: a busy store hides no local Stop
        return None

    def pre_dispatch_pause(self) -> str | None:
        """The owner's Pause over this task's tree, for a PRE-dispatch wait only.

        Never part of ``control_reason``: that reader also polls while a sent
        model operation's result is awaited, and an owner Pause lets sent work
        finish instead of cancelling it (``owner_pause``).
        """
        from types import SimpleNamespace

        from ouroboros.owner_pause import RAIL_OWNER_PAUSE, member_fence

        source = self.tool_context or SimpleNamespace(
            task_id=self.task_id, root_task_id=str(self.task.get("root_task_id") or self.task_id),
            budget_drive_root=self.canonical_root)
        fence = member_fence(source)
        return str(fence.get("reason") or RAIL_OWNER_PAUSE) if fence else None

    def _publish(self, row: dict, *, applied_request_id: str = "") -> None:
        with self.lock:
            self.revision += 1
            row["revision"] = self.revision
            row["updated_at"] = utc_now_iso()
            # The live owner knows whether this call has a non-generating access
            # observer. A reset forecast alone cannot resume an API-only call;
            # mixed auth/quota with a viable automatic route needs no human.
            row["requires_owner_action"] = bool(row["state"] == "waiting" and (
                row.get("reason") == "auth" or row.get("auto_continue") is False
                or not row.get("source") or row.get("_unproven_account")))
            if applied_request_id:
                row["applied_request_id"] = applied_request_id
            if applied_request_id or row["state"] == "resolved":
                row.pop("pending_action", None)

            def update(previous):
                # Endpoint-owned pending/save receipts cannot be overwritten by
                # the worker's older in-memory view of the same projection.
                snapshot = {key: value for key, value in row.items()
                            if key not in {"pending_action", "saved_request_id"} and not key.startswith("_")}
                value = {**(previous or {}), **copy.deepcopy(snapshot)}
                if applied_request_id or row["state"] == "resolved":
                    value.pop("pending_action", None)
                return value

            stored = self.mutate_row(row["wait_id"], update)
            row.update(stored)
            clock_projection = self.quota_clock_snapshot()
            public = {key: copy.deepcopy(value) for key, value in row.items() if not key.startswith("_")}
            event = {"type": "task_model_wait", "ts": utc_now_iso(), "task_id": self.task_id,
                     **public, "quota_clock": clock_projection, "is_progress": False}
            if self.owner_id:
                event["model_wait_owner_id"] = self.owner_id
        # The owner's projection precedes notification. The handler owns the
        # supervisor projection and live forwarding; it writes no second ledger.
        if self.event_queue is not None:
            self.event_queue.put(event)
        else:
            from ouroboros import dialogue_store

            dialogue_store.append_quietly(pathlib.Path(self.canonical_root), "progress", event)

    def _drain_controls(self) -> None:
        from ouroboros.owner_mailbox import KIND_MODEL_WAIT, _mailbox_path, drain_owner_entries

        with self.lock:
            try:
                stat = _mailbox_path(pathlib.Path(self.drive_root), self.task_id).stat()
                stamp = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
            except FileNotFoundError:
                return
            if stamp == self.mailbox_stamp:
                return
            seen = set(self.seen_controls)
            read_status = {}
            entries = drain_owner_entries(pathlib.Path(self.drive_root), self.task_id,
                                           seen, kinds={KIND_MODEL_WAIT}, _read_status=read_status)
            if not read_status.get("complete"):
                return
            stored = self.read_rows() if entries else {}
            for entry in entries:
                try:
                    action = json.loads(entry["text"])
                except (ValueError, TypeError):
                    continue
                if not isinstance(action, dict) or action.get("task_attempt") != self.attempt:
                    continue
                row = self.waits.get(str(action.get("wait_id") or ""))
                canonical = stored.get(str(action.get("wait_id") or "")) or {}
                requested = {key: value for key, value in action.items() if key not in {"wait_id", "task_attempt"}}
                if (not row or row.get("state") != "waiting" or canonical.get("state") != "waiting"
                        or canonical.get("task_attempt") != self.attempt or canonical.get("pending_action") != requested):
                    continue
                if action.get("action") == "auto_continue":
                    row["auto_continue"] = action["auto_continue"]
                    row["_check_now"] = action["auto_continue"]
                    self.auto_continue[row["role"]] = action["auto_continue"]
                    self._publish(row, applied_request_id=action["request_id"])
                elif action.get("action") in {"switch", "retry"}:
                    row["_action"] = action
            # A failed read or application cannot acknowledge the control. Keep
            # the normal strict authority error, without poisoning its next read.
            self.seen_controls.update(seen)
            self.mailbox_stamp = stamp

    def waiting_slots(self) -> set[str]:
        with self.lock:
            return {str(row.get("_slot_id") or "") for row in self.waits.values() if row.get("state") == "waiting"}

    def wait(self, llm: Any, error: Exception, kwargs: dict,
             caller_cancel: threading.Event | None = None) -> dict:
        """Hold this call's live stack; metadata checks never generate an answer."""
        from ouroboros import config
        from ouroboros.gateways.claudexor import ClaudexorUnavailable
        from ouroboros.model_slots import MODEL_ACCOUNTS_KEY, model_role_option
        from ouroboros.provider_models import parse_claudexor_model

        role = kwargs["model_role"]
        try:
            source, native_model = parse_claudexor_model(kwargs["model"])
        except ValueError:
            source, native_model = "", kwargs["model"]  # no configured non-generating access observer
        observes_access = bool(source)
        route = getattr(error, "route", {}) or {}
        problem_context = (getattr(error, "problem", {}) or {}).get("context") or {}
        account_intent = kwargs.get("model_account_override")
        if account_intent is None:
            account_intent = model_role_option(MODEL_ACCOUNTS_KEY, role)
        wait_id = uuid.uuid4().hex
        scope = current_usage_scope()
        slot_id = str(getattr(scope, "review_slot_id", "") or "")
        reason = model_wait_reason(error)
        quota_wait = reason in {"quota", "auth_quota"}  # unavailable does not prove a quota window
        row = {"wait_id": wait_id, "task_attempt": self.attempt, "role": role,
               "_slot_id": slot_id,
               "model": kwargs["model"], "source": source,
               "credential_profile_id": "" if reason == "auth_quota" else str(route.get("credentialProfileId") or problem_context.get("credentialProfileId") or account_intent or ""),
               "credential_harness": "", "reason": reason, "reset_at": str(getattr(error, "reset_at", "") or ""),
               "auto_continue": self.auto_continue.get(role, True) if observes_access else False, "state": "waiting",
               **({"availability_observation": "unavailable"} if not observes_access else {}),
               "worker_slot_held": self.worker_slot_held, "started_at": utc_now_iso()}
        if self.owner_id:
            row["model_wait_owner_id"] = self.owner_id
        row.update({key: {**copy.deepcopy(value), "slot_id": slot_id} for key, value in self.row_facts.items()})
        if getattr(error, "account_rotation", None):
            row["account_rotation"] = copy.deepcopy(error.account_rotation)
        # A vendor refusal with no reset evidence gives the engine no cooldown for that
        # Auto account, so its catalog choosing the same account again proves nothing.
        unproven = str(route.get("credentialProfileId") or "") if (
            reason == "quota" and not account_intent and getattr(error, "status_code", 0)
            and getattr(getattr(error, "physical_attempt_capture", None), "state", None) == "settled"
            and not (problem_context.get("resetsAt") or problem_context.get("retryAfterMs"))) else ""
        row["_unproven_account"] = unproven
        with self.lock:
            self.waits[wait_id] = row
        if quota_wait:
            self.quota_enter(wait_id, slot_id)
        resolution = "task_ended"
        request_id = ""
        backoff = float(config.NETWORK_WAIT_BACKOFF_START_SEC)
        next_check = 0.0
        try:
            self._publish(row)
            while True:
                control = ("caller_cancelled" if caller_cancel is not None and caller_cancel.is_set()
                           else self.control_reason() or self.pre_dispatch_pause())
                callback = kwargs.get("model_poll_control")
                if not control and callback is not None:
                    control = callback()
                if control:
                    resolution = control
                    raise ModelWaitInterrupted(control, role=role, cause=error)
                self._drain_controls()
                with self.lock:
                    action = row.pop("_action", None)
                if action:
                    request_id = action["request_id"]
                    if action["action"] == "switch":
                        self.overrides[role] = {"model": action["model"], "use_local": action["use_local"],
                                                "model_account_override": action["credential_profile_id"]}
                        resolution = "model_switched"
                        return {**kwargs, **self.overrides[role]}
                    resolution = "retry_requested"
                    return kwargs
                now = time.monotonic()
                if row.pop("_check_now", False):
                    next_check = 0.0
                if observes_access and now >= next_check:
                    try:
                        if not row["credential_harness"]:
                            sources = llm.claudexor_model_sources()
                            match = next((item for item in sources.get("sources", []) if item.get("id") == source), {})
                            harness = str(match.get("credentialHarness") or "")
                            if harness:
                                row["credential_harness"] = harness
                                self._publish(row)
                        if row["auto_continue"]:
                            from ouroboros.llm_claudexor import catalog_admits_model

                            account = kwargs.get("model_account_override")
                            if account is None:
                                account = model_role_option(MODEL_ACCOUNTS_KEY, role)
                            catalog = llm.claudexor_model_catalog(source, account or None,
                                                                 requested_model=native_model)
                            if (catalog.get("source") == source
                                    and (not account or catalog.get("credentialProfileId") == account)
                                    and not (unproven and not account and catalog.get("credentialProfileId") == unproven)
                                    and catalog_admits_model(catalog, native_model)):
                                resolution = "resource_available"
                                return kwargs
                    except ClaudexorUnavailable:
                        # Catalog absence is not a failed generation or proof of
                        # logout. Keep the original typed resource refusal.
                        pass
                    next_check = now + backoff
                    backoff = min(backoff * 2, config.NETWORK_WAIT_BACKOFF_MAX_SEC)
                if caller_cancel is not None:
                    caller_cancel.wait(config.CLAUDEXOR_MODEL_POLL_INTERVAL_SEC)
                else:
                    time.sleep(config.CLAUDEXOR_MODEL_POLL_INTERVAL_SEC)
        finally:
            if quota_wait:
                self.quota_leave(wait_id, slot_id)
            row.update(state="resolved", resolution=resolution)
            self._publish(row, applied_request_id=request_id)
            if resolution in {"resource_available", "model_switched", "retry_requested"} and (
                    self.tool_context is not None or "review_operation" in self.row_facts):
                from types import SimpleNamespace

                from ouroboros.model_sleep import announce_resumed

                # The call goes on: the author's own, or a reviewer's for the task its panel serves.
                announce_resumed(self.tool_context or SimpleNamespace(
                    event_queue=self.event_queue, task_id=self.task_id, current_chat_id=self.task.get("chat_id")))

    def bind_answer_consumer(self) -> dict:
        """Bind this local receiver before dispatch; birth absence grants no death recovery."""
        with self.lock:
            if self.closed:
                from ouroboros.llm_attempt import _PhysicalSendNotStarted
                raise _PhysicalSendNotStarted("cancelled")
            self.answer_consumer_bound = True
            from ouroboros.tool_custody import owner_proof

            return {"local_answer_consumer_id": self.answer_consumer_id,
                    "local_answer_task_attempt": self.attempt,
                    "local_answer_owner_birth": self.answer_owner_birth,
                    "local_answer_owner_proof": owner_proof()}

    def rotate_answer_consumer(self) -> None:
        """A warm return after this receiver abandoned a sent wait (owner Pause).

        The exact old consumer identity is retired positively and durably, and a
        fresh one is minted before the next send: an answer to the abandoned
        attempt can never be read as the new call's, and the custody census sees
        the new live call under its own identity. The scope stays open.
        """
        with self.lock:
            old, bound = self.answer_consumer_id, self.answer_consumer_bound
            self.answer_consumer_id = uuid.uuid4().hex
            self.answer_consumer_bound = False
        if bound:
            try:
                retire_model_consumers(self.canonical_root, self.task_id, {old: self.attempt})
            except Exception:
                import logging
                logging.getLogger(__name__).warning("Abandoned model consumer retirement owed for %s",
                                                    self.task_id, exc_info=True)
                _owe_consumer_retirement(self.canonical_root, self.task_id, {old: self.attempt})

    def close(self) -> None:
        with self.lock:
            self.closed = True
            bound = self.answer_consumer_bound
        from types import SimpleNamespace
        from ouroboros.budget_pause import forget_tool_scope
        # Native review scopes may never enter loop cleanup. Use THIS frozen
        # owner even when close is called under a different current context.
        forget_tool_scope(SimpleNamespace(drive_root=self.canonical_root, task_id=self.task_id),
                          attempt=self.attempt)
        from ouroboros import custody_owed

        if custody_owed.pending():
            custody_owed.drain()  # scope close: one bounded attempt for what this process owes
        if not bound:
            return
        # Positive retirement of THIS consumer, not terminal task status, PID
        # absence or the pool's current assignment. Cash and remote custody stay.
        try:
            retire_model_consumers(self.canonical_root, self.task_id, {self.answer_consumer_id: self.attempt})
        except Exception:
            import logging
            logging.getLogger(__name__).warning("Model consumer retirement owed for %s", self.task_id,
                                                exc_info=True)
            # The receiver IS closed on a live host: this process owes that positive
            # fact and writes it later (custody_owed); never a second census or a PID test.
            _owe_consumer_retirement(self.canonical_root, self.task_id, {self.answer_consumer_id: self.attempt})


def _owe_consumer_retirement(root: Any, task_id: str, consumers: dict[str, int]) -> None:
    """A positive receiver closure whose durable retirement failed: this process owes the
    SAME write (``custody_owed``) instead of leaving the receiver open for a later inference."""
    from ouroboros import custody_owed

    consumers = dict(consumers)

    def write():
        from ouroboros.store import acquisition_budget
        from ouroboros._usage_wait import ACQUISITION_SLICE_SEC

        try:
            with acquisition_budget(ACQUISITION_SLICE_SEC):  # a driver's attempt never waits a full store budget
                retire_model_consumers(root, task_id, consumers)
        except ValueError as exc:
            if type(exc.__cause__) is not LookupError:  # only "the task's row is gone" is done
                raise
            import logging
            logging.getLogger(__name__).warning("Owed receiver retirement dropped: task %s is gone", task_id)
    custody_owed.owe(("consumers", str(root), str(task_id), tuple(sorted(consumers))), write,
                     label=f"consumers:{task_id}")


def retire_model_consumers(root: Any, task_id: str, consumers: dict[str, int], *, record: dict | None = None) -> None:
    """Positive receiver retirement, from scope exit or exact confirmed-worker death.

    This existing result projection releases only local answer-writing custody.
    It changes no usage row, external process or delegated-run obligation.
    ``record`` adds facts to each row (an owner's or the mind's ``accepted`` decision). A
    retirement already recorded is kept: the first writer's facts stand, decided in the write.
    """
    try:  # one `retired_model_consumers` row per consumer, one transaction, never the body
        put_handoffs(pathlib.Path(root), task_id, "retired_model_consumers",
                     {consumer: {**(record or {}), "task_attempt": attempt, "retired_at": utc_now_iso()}
                      for consumer, attempt in consumers.items()}, only_absent=True)
    except LookupError as exc:
        raise ValueError("model consumer retirement requires its task result") from exc


@contextlib.contextmanager
def task_model_wait_scope(*, task: dict, drive_root: Any, event_queue: Any,
                          worker_slot_held: bool, **owner_hooks: Any) -> Iterator[TaskModelWait]:
    """The ordinary task frame owns this context, exactly as it owns UsageScope."""
    context = TaskModelWait(task=task, drive_root=drive_root, event_queue=event_queue,
                            worker_slot_held=worker_slot_held, **owner_hooks)
    token = _CURRENT.set(context)
    try:
        yield context
    finally:
        context.close()
        _CURRENT.reset(token)


def dispatch_control_reason(owner, *, handed_off=False, before_task_start=False):
    """Resolve readable controls before surfacing a storage gap; Pause applies only before handoff."""
    from ouroboros.owner_pause import RAIL_OWNER_PAUSE, scope_fence, read_fence, fence_closed
    from ouroboros.store import StoreBusy

    gap = None
    try:
        reason = owner.control_reason(strict_current=not handed_off) if owner is not None else None
    except StoreBusy as exc:
        reason, gap = None, exc
    if reason:
        return reason
    if owner is not None:
        from ouroboros.store import acquisition_budget
        from ouroboros.work_diagnostics import control_reason, withdraw_revoked

        with acquisition_budget(0):
            reason = control_reason(owner.task, owner.canonical_root, before_handoff=not handed_off)
        if reason == "cancelled":
            withdraw_revoked(owner.task, owner.canonical_root)  # custody settles it; never a task exception
        if reason:
            if reason == RAIL_OWNER_PAUSE and before_task_start:
                fence = read_fence(owner.canonical_root, owner.task["root_task_id"])
                if not fence_closed(fence) or not fence.get("fence_id"):
                    return "owner_launch_authority_unavailable:diagnostic_pause_unbound"
                owner.prestart_pause_fence = dict(fence)
            return reason
    if handed_off:
        if gap is not None:
            raise gap
        return None
    try:
        if before_task_start and owner is not None:
            from ouroboros.owner_wait import governing_continuation
            if governing_continuation(owner.task):
                if gap is not None:
                    raise gap
                return None  # its existing grant consumer owns the saved fence
            # No lifecycle row is required yet; an existing fence still binds this input.
            meta = owner.task.get("metadata") or {}
            root_id = str(owner.task.get("root_task_id") or meta.get("root_task_id") or owner.task_id)
            fence = read_fence(owner.canonical_root, root_id)
            paused = fence_closed(fence) and owner.task_id not in (fence.get("selected_members") or {})
            if paused:
                owner.prestart_pause_fence = dict(fence)
        else:
            paused = scope_fence()
        if paused:
            return RAIL_OWNER_PAUSE
    except StoreBusy as exc:
        gap = gap or exc
    if gap is not None:
        raise gap
    return None


def current_model_wait() -> TaskModelWait | None:
    return _CURRENT.get()


_AFTER_THE_REPLY = frozenset({"post_task", "consolidation"})  # the post-task worker's usage categories


def owner_awaits_current_call() -> bool:
    """Whether this call prepares the reply of a direct turn the owner awaits (Main, a project room, a call): the
    model queue lets it ahead of background calls that have not started (``model_concurrency``). The same turn's
    work after its answer (synthesis, memory drafts, scratchpad consolidation) queues as background."""
    task = getattr(_CURRENT.get(), "task", None)
    if not isinstance(task, dict) or not task.get("_is_direct_chat"):
        return False
    from ouroboros.usage_accounting import current_usage_scope
    from supervisor.active_activity import reply_expected

    return getattr(current_usage_scope(), "category", "") not in _AFTER_THE_REPLY and reply_expected(task)


def announce_capability_refusal(error: Exception, *, drive_root: Any) -> None:
    """One owner notice per account/model episode when no account can serve a model.

    An account-service fact, not a helper's answer and not a timer: the engine's
    typed facts decide (``capability_notice``), its episode is the delivery id, and
    the unchanged durable outbox sends one row however many helpers hit it; a
    successful differential call never enters here. The task that first saw it is
    named, not part of the id. A bare library caller gets the same diagnostics on
    its error and no chat side effect.
    """
    from ouroboros.llm_substitution import capability_notice

    owner = current_model_wait()
    if owner is None or owner.closed:
        return
    notice = capability_notice(getattr(error, "problem", None) or {})
    if notice is None:
        return
    delivery_id, text = notice
    root = owner.canonical_root or drive_root
    try:
        from ouroboros.projects_registry import task_presentation_snapshot

        label = task_presentation_snapshot(root, owner.task_id, task=owner.task).get("target_label") or ""
    except Exception:
        label = ""  # the notice outranks its label
    try:
        from supervisor.schedule_notes import owner_chat_id
        from supervisor.terminal_delivery import enqueue_terminal_delivery_outcome

        enqueue_terminal_delivery_outcome(root, {
            "type": "send_message", "chat_id": owner_chat_id(root), "task_id": "",
            "text": text + (f"\nFirst seen in {label}." if label else ""), "role": "system",
            "system_type": "model_capability_notice", "delivery_id": delivery_id,
        }, event_queue=owner.event_queue)
    except Exception:
        import logging
        logging.getLogger(__name__).warning("Capability refusal notice could not be queued", exc_info=True)


@contextlib.contextmanager
def operation_wait_scope(owner: TaskModelWait) -> Iterator[TaskModelWait]:
    """Bind an already-paid operation's own wait owner for its caller and workers.

    The author's execution-lifetime scopes (a tool or call execution deadline,
    Main's re-preparation callbacks) are not the operation's and are cleared
    explicitly; explicit calendar deadlines stay. Clocks read under this scope
    are the operation's own union and per-slot quota clocks.
    """
    tokens = ((_CURRENT, _CURRENT.set(owner)), (_REPREPARE, _REPREPARE.set(None)), (_LOGICAL, _LOGICAL.set(())))
    try:
        yield owner
    finally:
        for variable, token in reversed(tokens):
            variable.reset(token)


def model_waitable(function: Callable | None = None, *, client_parameter: str = "self") -> Callable:
    """Catch resource refusals before helper catches; callers may decline waiting.

    ``wait_for_resources`` is call-local, leaving the shared task's overrides,
    controls and custody intact even when a refusal returns immediately; so is a
    round's ``defer_resource_wait``, which retains the refusal for that round.
    """
    if function is None:
        return functools.partial(model_waitable, client_parameter=client_parameter)
    signature = inspect.signature(function)

    def bind(args, kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = dict(bound.arguments)
        receiver = values.pop(client_parameter)
        for name, parameter in signature.parameters.items():
            if parameter.kind is inspect.Parameter.VAR_KEYWORD:
                values.update(values.pop(name, {}))
        if "processing_preference" in signature.parameters:
            from ouroboros.model_slots import resolve_processing_preference

            # Capture before the logical retry loop, including callers without
            # a task owner. Re-entering after a quota wait never rereads settings.
            values["processing_preference"] = resolve_processing_preference(
                str(values.get("model_role") or ""),
                override=values.get("processing_preference"),
            )
        return receiver, values

    def prepare(context, values):
        role = str(values.get("model_role") or "")
        override = context.overrides.get(role)
        if override and any(values.get(key) != value for key, value in override.items()):
            return context.reprepare(role, {**values, **override})
        return values

    def with_control(context, values):
        original = values.get("model_poll_control")
        original = getattr(original, "_model_wait_original", original)

        def poll():
            return context.control_reason() or (original() if original is not None else None)

        poll._model_wait_original = original
        return {**values, "model_poll_control": poll}

    def can_wait(context, receiver, error, values):
        """Wait here, or raise; a deferring round first retains a quota refusal for its later routes."""
        from ouroboros.llm_claudexor import ClaudexorModelError

        capture = getattr(error, "physical_attempt_capture", None)
        binding = (_REPREPARE.get() or {}).get(values.get("model_role"))
        classified = binding[1](error) if binding and binding[1] is not None else ""
        if classified:
            error.task_resource_wait_reason = classified
        reason = model_wait_reason(error)
        if not (context and not context.closed and values.get("model_role") and reason
                and (classified or reason == "connection_unavailable" or (isinstance(error, ClaudexorModelError)
                                    and getattr(capture, "state", None) in {"released", "settled"}))):
            return False
        deferral = _deferral(values["model_role"])
        # Account rotation, then every configured route, then the owner: sign-in too, so a
        # healthy configured route keeps working while the refused account awaits repair.
        if deferral is not None:
            deferral.retain(receiver, error, values)
            return False
        return bool(context.waits_allowed and values.get("wait_for_resources", True))

    def merged(result, attempts):
        result[1]["ledger_attempt_ids"] = list(dict.fromkeys([*attempts, *result[1].get("ledger_attempt_ids", [])]))
        return result

    def route_projection(values):
        from ouroboros.model_slots import MODEL_ACCOUNTS_KEY, model_role_option

        role = str(values.get("model_role") or "")
        account = values.get("model_account_override")
        return {"role": role, "model": values["model"], "use_local": bool(values.get("use_local")),
                "credential_profile_id": account if account is not None else model_role_option(MODEL_ACCOUNTS_KEY, role)}

    def record_attempts(attempts, error):
        attempts.extend(getattr(error, "ledger_attempt_ids", []))
        attempt_id = getattr(getattr(error, "physical_attempt_capture", None), "attempt_id", "")
        if attempt_id:
            attempts.append(attempt_id)

    if inspect.iscoroutinefunction(function):
        @functools.wraps(function)
        async def asynchronous(*args, **kwargs):
            import asyncio

            receiver, values = bind(args, kwargs)
            context = current_model_wait()
            if context is None:
                return await function(**{client_parameter: receiver, **values})
            attempts = []
            preparation = prepare(context, values)
            while True:
                try:
                    with prepared_call_scope(preparation) as values:
                        values = with_control(context, values)
                        route = route_projection(values)
                        control = values["model_poll_control"]()
                        if control:
                            raise ModelWaitInterrupted(control, role=values["model_role"])
                        result = await function(**{client_parameter: receiver, **values})
                        result[1]["model_role_route"] = route
                        return merged(result, attempts)
                except Exception as error:
                    error.model_role_route = route_projection(values)
                    propagate_model_control(error)
                    if not can_wait(context, receiver, error, values):
                        raise
                    record_attempts(attempts, error)
                    cancelled = threading.Event()
                    waiter = asyncio.create_task(asyncio.to_thread(context.wait, receiver, error, values, cancelled))
                    try:
                        values = await asyncio.shield(waiter)
                    except asyncio.CancelledError:
                        cancelled.set()
                        waiter.add_done_callback(lambda done: None if done.cancelled() else done.exception())
                        raise
                    preparation = context.reprepare(values["model_role"], values)
        return asynchronous

    @functools.wraps(function)
    def synchronous(*args, **kwargs):
        receiver, values = bind(args, kwargs)
        context = current_model_wait()
        if context is None:
            return function(**{client_parameter: receiver, **values})
        attempts = []
        preparation = prepare(context, values)
        while True:
            try:
                with prepared_call_scope(preparation) as values:
                    values = with_control(context, values)
                    route = route_projection(values)
                    control = values["model_poll_control"]()
                    if control:
                        raise ModelWaitInterrupted(control, role=values["model_role"])
                    result = function(**{client_parameter: receiver, **values})
                    result[1]["model_role_route"] = route
                    return merged(result, attempts)
            except Exception as error:
                error.model_role_route = route_projection(values)
                propagate_model_control(error)
                if not can_wait(context, receiver, error, values):
                    raise
                record_attempts(attempts, error)
                values = context.wait(receiver, error, values)
                preparation = context.reprepare(values["model_role"], values)
    return synchronous


def monotonic_now(review_slot_id: str | None = None) -> float:
    """Execution clock: only confirmed quota waiting is excluded.

    ISO/calendar deadlines must keep their wall clock. A reviewer reads its
    own slot's union; ordinary task/tool execution reads the task-wide union.
    """
    now = time.monotonic()
    context = current_model_wait()
    if context is None:
        return now
    scope = current_usage_scope()
    slot = review_slot_id if review_slot_id is not None else str(getattr(scope, "review_slot_id", "") or "")
    return now - context.paused_seconds(slot, now=now)


def future_result(future: Any, timeout: float) -> Any:
    """Keep the existing Future and its execution budget across a quota pause."""
    if current_model_wait() is None:
        return future.result(timeout=timeout)
    deadline = monotonic_now() + timeout
    while True:
        remaining = max(0.0, deadline - monotonic_now())
        try:
            return future.result(timeout=remaining)
        except (TimeoutError, concurrent.futures.TimeoutError):
            if future.done() or monotonic_now() >= deadline:
                raise
