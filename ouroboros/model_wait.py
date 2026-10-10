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
from ouroboros.utils import append_jsonl, update_json_locked, utc_now_iso


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
    """Execution time of one task: wall clock minus the quota-wait union minus
    the separate budget-paused interval. ``started_at`` is never moved, so every
    finite-lifetime consumer (supervisor timeouts, owner stop, the exact-pause
    grant, the live wait controls) subtracts the same two carriers."""
    try:
        started = float(meta.get("started_at") or 0.0)
    except (TypeError, ValueError):
        return 0.0
    if started <= 0 or not math.isfinite(started):
        return 0.0
    sleeping = meta.get("sleep_parked_at")  # a warm sleep or owner Pause (worker_owner_wait)
    return max(0.0, now - started - quota_waited_seconds(meta, now) - budget_paused_seconds(meta)
               - (max(0.0, now - float(sleeping)) if isinstance(sleeping, (int, float)) else 0.0))


_CURRENT: contextvars.ContextVar[TaskModelWait | None] = contextvars.ContextVar(
    "ouroboros_model_wait", default=None)
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
    """Mutate only one wait projection in the existing schema-stamped task result."""
    from ouroboros.task_results import (
        require_writable_task_result_schema, stamp_task_result_schema, task_result_path,
    )

    result: dict = {}

    def update(current):
        require_writable_task_result_schema(current)
        if not current.get("status") or current.get("task_id") != task_id:
            raise ValueError("model wait requires its lifecycle owner's task result")
        waits = current.get("model_waits", {})
        if not isinstance(waits, dict):
            raise ValueError("model_waits projection is malformed")
        previous = waits.get(wait_id)
        if previous is not None and not isinstance(previous, dict):
            raise ValueError("model wait projection is malformed")
        next_row = transform(copy.deepcopy(previous))
        result.update(previous or {} if next_row is None else next_row)
        if next_row is None:
            return None
        return stamp_task_result_schema({**current, "model_waits": {**waits, wait_id: next_row}})

    update_json_locked(task_result_path(pathlib.Path(root), task_id), update, strict_existing_dict=True)
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
        """Live execution time: elapsed minus the quota union minus budget pause,
        minus a warm sleep or owner Pause (``model_sleep`` folds it into the paused
        carrier when the task runs again, so it is never subtracted twice)."""
        stamp = time.monotonic() if now is None else now
        sleeping = self.sleep_started_monotonic
        return max(0.0, stamp - self.started_monotonic - self.paused_seconds(now=stamp) - self.budget_paused_sec
                   - (max(0.0, stamp - sleeping) if sleeping is not None else 0.0))

    def execution_window_remaining(self) -> float | None:
        """A custom live owner supplies its own clock and a task without an absolute
        lifetime has none; None invents no deadline, and 0.0 means the window is spent."""
        if self.owner_control is not None:
            return None
        from ouroboros.config import get_task_abs_ceiling_sec
        ceiling = get_task_abs_ceiling_sec()
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

    def control_reason(self) -> str | None:
        """Existing task controls and explicit calendar bounds, never model prose."""
        from ouroboros.cancel_intents import cancel_pending, resolve_owner_stop_intent
        from ouroboros.config import get_task_abs_ceiling_sec
        from ouroboros.deadline_utils import seconds_until
        from ouroboros.owner_mailbox import KIND_FINALIZE_NOW, drain_owner_entries

        if self.closed:
            return "task_ended"
        if self.owner_control is None and cancel_pending(pathlib.Path(self.canonical_root), self.task_id):
            _, graceful = resolve_owner_stop_intent(self.canonical_root, self.task_id)
            if not graceful:
                return "cancelled"
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
        ceiling = get_task_abs_ceiling_sec()  # None = unlimited; 0 = exhausted, never unlimited
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
            append_jsonl(pathlib.Path(self.canonical_root) / "logs" / "progress.jsonl", event)

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

    def bind_answer_consumer(self) -> dict:
        """Bind this local receiver before dispatch; birth absence grants no death recovery."""
        with self.lock:
            if self.closed:
                from ouroboros.llm_attempt import _PhysicalSendNotStarted
                raise _PhysicalSendNotStarted("cancelled")
            self.answer_consumer_bound = True
            return {"local_answer_consumer_id": self.answer_consumer_id,
                    "local_answer_task_attempt": self.attempt,
                    "local_answer_owner_birth": self.answer_owner_birth}

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
                logging.getLogger(__name__).warning("Abandoned model consumer retirement could not persist for %s",
                                                    self.task_id, exc_info=True)
                _retain_retirement_witness(self.canonical_root, self.task_id, {old: self.attempt})

    def close(self) -> None:
        with self.lock:
            self.closed = True
            if not self.answer_consumer_bound:
                return
        # Positive retirement of THIS consumer, not terminal task status, PID
        # absence or the pool's current assignment. Cash and remote custody stay.
        try:
            retire_model_consumers(self.canonical_root, self.task_id, {self.answer_consumer_id: self.attempt})
        except Exception:
            import logging
            logging.getLogger(__name__).warning("Model consumer retirement could not persist for %s", self.task_id,
                                                exc_info=True)
            # The receiver IS closed on a live host: keep that positive fact for the
            # addressed repair owner (#1554); never a second census or a PID test.
            _retain_retirement_witness(self.canonical_root, self.task_id, {self.answer_consumer_id: self.attempt})


def _retain_retirement_witness(root: Any, task_id: str, consumers: dict[str, int]) -> None:
    """A positive receiver closure whose durable retirement failed: retain the exact
    witness (``local_custody_repair``), so the addressed recovery owner retries the
    SAME writer instead of inferring death from the platform later."""
    try:
        from ouroboros.local_custody_repair import retain_witness

        retain_witness(root, task_id, reason="receiver_closed_retirement_unwritten",
                       model_consumers=dict(consumers))
    except Exception:
        import logging
        logging.getLogger(__name__).warning("Retirement witness for %s could not be retained", task_id,
                                            exc_info=True)


def retire_model_consumers(root: Any, task_id: str, consumers: dict[str, int]) -> None:
    """Positive receiver retirement, from scope exit or exact confirmed-worker death.

    This existing result projection releases only local answer-writing custody.
    It changes no usage row, external process or delegated-run obligation.
    """
    from ouroboros.task_results import require_writable_task_result_schema, stamp_task_result_schema, task_result_path
    def retire(current):
        require_writable_task_result_schema(current)
        if not current.get("status") or current.get("task_id") != task_id:
            raise ValueError("model consumer retirement requires its task result")
        retired = dict(current.get("retired_model_consumers") or {})
        for consumer, attempt in consumers.items():
            retired[consumer] = {"task_attempt": attempt, "retired_at": utc_now_iso()}
        return stamp_task_result_schema({**current, "retired_model_consumers": retired})
    update_json_locked(task_result_path(pathlib.Path(root), task_id), retire, strict_existing_dict=True)


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


def current_model_wait() -> TaskModelWait | None:
    return _CURRENT.get()


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
                and (classified or (isinstance(error, ClaudexorModelError)
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
