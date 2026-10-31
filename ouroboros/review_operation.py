"""An already-paid review owns its lifetime, controls and free collection.

One wait owner spans panel workers/drain, union/slot quota clocks and settlement,
beyond the author execution scope (``model_wait.operation_wait_scope``); calendar,
money, lineage and Stop/Panic still bind. Joining an entirely live panel reuses
its operation; dispatch without a task wait owner creates none.

Before sends, a sending operation records a read-back ``review_operations`` result
pointer naming its controller, panel, slots and attempt; the pointer is removed when
the operation closes, and a missing one forbids remote decisions only. Decisions use
the wait's ``pending_action``, not the author's mailbox. Only a proven controller may
act: local registration, or another process in this server generation with exact pid
birth, custody session and an open pointer matching panel/slot/attempt, with
readable, clear Stop/Panic state (``review_operation_controller``). The maintenance pass
retires a dead controller's open pointer (``retire_dead_review_pointers``).

Collection parses exact completed outcomes or proven delegated runs locally: attach-only,
never start/re-post/cancel/acknowledge/retire, ensure a daemon or call a model (Light included).
Missing records prove no absence of sends."""

from __future__ import annotations

import contextlib
import contextvars
import copy
import hashlib
import json
import logging
import os
import pathlib
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from ouroboros.model_wait import TaskModelWait, control_probe_requires_current, control_probe_active
from ouroboros.utils import utc_now_iso
from ouroboros.store import StoreBusy

log = logging.getLogger(__name__)

# Task-result field: controller owner id -> the operation's durable checkpoint pointer.
OPERATIONS_FIELD = "review_operations"
# ``retained``: pointer written, no worker released by this controller yet (a
# crash right after a release can leave it, so it never proves "not sent");
# ``dispatched``: released to physical workers. A pointer only proves liveness and
# is removed when its operation closes. ``preparing`` (and the ``closed``,
# ``unpublished`` and ``collected`` states) are only ever read on historical
# task-acceptance pointers; nothing writes them now.
OPERATION_RETAINED, OPERATION_DISPATCHED = "retained", "dispatched"
OPERATION_PREPARING = "preparing"
_OPEN_STATES = frozenset({OPERATION_PREPARING, OPERATION_RETAINED, OPERATION_DISPATCHED})
# Typed per-slot collection facts (never a verdict): a pending producer is not a
# never-dispatched one, and neither is an unknown outcome.
COLLECTED, DEFERRED, UNAVAILABLE, NEVER_DISPATCHED = (
    "collected", "deferred", "unavailable", "never_dispatched")
# ``usage.custody_state`` of a lost seat whose own row names a requested, never-bound start:
# an attach-only read cannot bind it, but its review's own collection may still replay that
# exact request under its original key (``review_pause._settle_left_record``).
START_UNBOUND = "start_unbound"
_OBSERVE_TIMEOUT_SEC = 10.0
_CONTROL_RECHECK_SEC = 1.0

_LOCK = threading.Lock()
_LIVE: Dict[str, "ReviewOperation"] = {}
_BOUND: contextvars.ContextVar[Optional["ReviewOperation"]] = contextvars.ContextVar(
    "ouroboros_review_operation", default=None)
_SELF: Dict[int, Dict[str, Any]] = {}
# Operations a paused author detached from that have since closed in THIS process
# (owner id -> closed at): what the resumed author's notice may state as known.
_CLOSED_DETACHED: Dict[str, str] = {}


def controller_identity() -> Dict[str, Any]:
    """This process as a review controller: pid, birth (the ps spelling every release
    reads), validated ``kernel_birth`` (``process_identity``) and custody session."""
    from ouroboros.platform_layer import process_start_time
    from ouroboros.process_custody import current_custody_session_id
    from ouroboros.process_identity import process_kernel_birth

    pid = os.getpid()
    if pid not in _SELF:
        _SELF.clear()
        _SELF[pid] = {"pid": pid, "birth": process_start_time(pid), "kernel_birth": process_kernel_birth(pid) or ""}
    return {**_SELF[pid], "session": str(current_custody_session_id() or "")}


def controller_state(identity: Any) -> str:
    """Return proven ``local``, ``alive``, ``dead`` or ``unknown``.
    No live pid (including zombies) proves the controller gone. Kernel births of one
    validated kind decide (``process_identity.same_birth``); without them an equal
    ``birth`` spelling proves it alive and a different one proves it gone only when both
    are one validated kind. Another zone or locale spells one live process's ``ps`` start
    differently, so that difference, like a missing or unreadable birth, is unknown: an old
    record on a reused pid stays unknown until that process ends. For our own pid
    (pid and birth survive exec; this process mints its identity
    once), a different birth names another holder, and distinct nonempty custody sessions
    prove the pre-exec controller gone; another process's session cannot prove that."""
    from ouroboros.process_identity import process_kernel_birth, same_birth

    try:
        pid = int(identity.get("pid") or 0) if isinstance(identity, dict) else 0
    except (TypeError, ValueError):
        pid = 0
    if pid <= 0:
        return "unknown"
    birth, kernel = str(identity.get("birth") or ""), str(identity.get("kernel_birth") or "")
    if pid == os.getpid():
        own = controller_identity()
        same = same_birth(own.get("kernel_birth"), kernel)
        if same is None:
            if not birth or not own["birth"]:
                return "unknown"
            same = birth == own["birth"]
        if not same:
            return "dead"
        session = str(identity.get("session") or "")
        if not session or not own["session"]:
            return "unknown"
        return "local" if session == own["session"] else "dead"
    from ouroboros.platform_layer import pid_is_alive, process_start_time
    from ouroboros.process_containment import pid_is_zombie

    if not pid_is_alive(pid) or pid_is_zombie(pid):
        return "dead"
    same = same_birth(process_kernel_birth(pid), kernel) if kernel else None
    if same is None and birth:
        live_birth = process_start_time(pid)
        same = True if live_birth and live_birth == birth else same_birth(live_birth, birth)
    return "unknown" if same is None else "alive" if same else "dead"


def operation_stop_state(root: pathlib.Path, task_id: str) -> str:
    """Current Restart, immediate/settled cancellation, clear or unknown."""
    try:
        from ouroboros.delegate_recovery import owner_restart_in_progress
        if owner_restart_in_progress(root):
            return "owner_restart"
        from ouroboros.cancel_intents import cancel_pending, resolve_owner_stop_intent
        from ouroboros.task_results import load_task_result
        if cancel_pending(root, task_id, strict=True) and not resolve_owner_stop_intent(root, task_id, propagate_busy=control_probe_active())[1]:
            return "cancelled"
        if str((load_task_result(root, task_id, strict=True) or {}).get("status") or "") == "cancelled":
            return "cancelled"
    except Exception as exc:
        if isinstance(exc, StoreBusy) and control_probe_requires_current():
            raise
        log.debug("review operation stop state unreadable for %s", task_id, exc_info=True)
        return "unknown"
    return ""


class _OperationWait(TaskModelWait):
    """The operation's wait owner: a decision is the canonical row's ``pending_action``.

    The gateway's durable claim on the wait row IS the delivery; the author's
    owner mailbox is released when the author settles and is never read here.
    """

    consumes_pending_action = True

    def _drain_controls(self) -> None:
        from ouroboros.task_store import signature

        try:
            stamp = signature(pathlib.Path(self.canonical_root), self.task_id)
        except Exception:
            return
        if stamp is None:
            return
        with self.lock:
            if stamp == self.mailbox_stamp:
                return
            waiting = [wait_id for wait_id, row in self.waits.items() if row.get("state") == "waiting"]
        try:
            stored = self.read_rows() if waiting else {}
        except (OSError, ValueError):
            return  # no control is applied without its canonical row; the next poll reads again
        with self.lock:
            for wait_id in waiting:
                row, canonical = self.waits.get(wait_id) or {}, stored.get(wait_id)
                action = canonical.get("pending_action") if isinstance(canonical, dict) else None
                if (not isinstance(action, dict) or row.get("state") != "waiting"
                        or canonical.get("state") != "waiting" or canonical.get("task_attempt") != self.attempt
                        or canonical.get("model_wait_owner_id") != self.owner_id
                        or str(action.get("request_id") or "") in self.seen_controls):
                    continue
                self.seen_controls.add(str(action.get("request_id") or ""))
                if action.get("action") == "auto_continue":
                    row["auto_continue"] = action["auto_continue"]
                    row["_check_now"] = action["auto_continue"]
                    self.auto_continue[row["role"]] = action["auto_continue"]
                    self._publish(row, applied_request_id=action["request_id"])
                elif action.get("action") in {"switch", "retry"}:
                    row["_action"] = dict(action)
            self.mailbox_stamp = stamp


class ReviewOperation:
    """One panel's live controller: its wait owner, its controls and worker lifetime."""

    def __init__(self, *, parent: Any, request: Any, task_id: str, result_root: pathlib.Path, owner_id: str = ''):
        from ouroboros.model_wait import mutate_wait
        from ouroboros.task_results import load_task_result

        self.owner_id = owner_id or f"review-operation-{uuid.uuid4().hex}"
        self.surface = str(getattr(request, "surface", "") or "")
        self.retry_key = str(getattr(request, "retry_key", "") or "")
        self.identity = controller_identity()
        self.result_root = pathlib.Path(result_root)
        self.checkpointed = False
        self.closed = False
        self._closing = self._published = False  # one final publisher at a time
        self.control_state = ""
        self._drains = 0
        self._dispatched = False
        # A background caller already left its author slot; a synchronous author
        # can also detach later while its reviewers run (``review_custody``).
        self.author_detached = bool(getattr(parent, "author_detached", False))
        self._control, self._checked_at = "", float("-inf")
        parent_task = parent.task if isinstance(parent.task, dict) else {}
        metadata = getattr(parent.tool_context, "task_metadata", None)
        metadata = metadata if isinstance(metadata, dict) else parent_task.get("metadata")
        # A fresh snapshot, never the author's mutable context: explicit deadlines,
        # contract, chat and root lineage ride it; the author's live objects do not.
        task = {key: value for key, value in parent_task.items()
                if not str(key).startswith("_") or key == "_presence_turn"}
        task.update(id=str(task_id or parent.task_id), _attempt=parent.attempt,
                    model_wait_owner_id=self.owner_id, metadata=metadata or {})
        try:
            task = copy.deepcopy(task)
        except Exception:
            task = json.loads(json.dumps(task, default=str))
        self.task_id = task["id"]
        root, tid = self.result_root, self.task_id
        self.wait = _OperationWait(
            task=task, drive_root=parent.drive_root, event_queue=parent.event_queue, worker_slot_held=False,
            owner_control=self.control,
            # Its own rows in the canonical result, independent of the author's hooks.
            row_mutator=lambda wait_id, transform: mutate_wait(root, tid, wait_id, transform),
            rows_reader=lambda: (load_task_result(root, tid, strict=True) or {}).get("model_waits", {}))
        self.wait.canonical_root = str(root)
        self.wait.overrides.update(copy.deepcopy(parent.overrides))
        self.wait.row_facts = {"review_operation": self.public()}

    def public(self) -> Dict[str, Any]:
        """The exact controller a wait row names: owner, surface, panel, process and chat."""
        return {"owner_id": self.owner_id, "surface": self.surface, "retry_key": self.retry_key,
                "controller": dict(self.identity), "chat_id": self.wait.task.get("chat_id")}

    def control(self) -> Optional[str]:
        """Latch Stop/Panic/cancellation. Sent observers reuse the short cache; admission reads afresh.

        An unreadable state is unknown, not a stop or decision authority. Failed probes never
        refresh the cache. A paid receiver keeps its call; unsent contention reaches its wait owner.
        """
        now = time.monotonic()
        if not self._control and (control_probe_requires_current() or now - self._checked_at >= _CONTROL_RECHECK_SEC):
            try:
                self.control_state = operation_stop_state(self.result_root, self.task_id)
            except StoreBusy:
                if control_probe_requires_current():
                    raise
                self.control_state = "unknown"  # A paid receiver keeps its operation and late result.
            if self.control_state in {"panic", "cancelled", "owner_restart"}:
                self._control = self.control_state
            if self.control_state != "unknown":
                self._checked_at = now
        return self._control or None

    def enter(self) -> None:
        with _LOCK:
            self._drains += 1

    def note_dispatch(self) -> None:
        """The panel released its first physical worker: a fact, never a gate."""
        if self._dispatched:
            return
        self._dispatched = True
        if self.checkpointed:
            _mark_operation(self.result_root, self.task_id, self.owner_id, OPERATION_DISPATCHED,
                            from_states={OPERATION_RETAINED})

    def drain_returned(self) -> None:
        with _LOCK:
            self._drains = max(0, self._drains - 1)
        self.maybe_close()

    def maybe_close(self) -> None:
        """Close after drains, workers and their settlement publication duties finish.

        ``closed`` ends the controller at once; registration includes the completion
        tail until the final pointer lands. A failed write or an exception keeps custody
        unknown and retryable (a later close or the idle probe), never an empty
        registry that authorizes worker death.
        """
        from ouroboros.review_custody import operation_has_live_workers

        with _LOCK:
            if self._closing or self._published or not self.closed and (
                    self._drains or operation_has_live_workers(self)):
                return
            self.closed = self._closing = True
        try:
            if not self.wait.closed:
                self.wait.close()
            # A liveness-only pointer leaves no history after close.
            self._published = not self.checkpointed or _mark_operation(
                self.result_root, self.task_id, self.owner_id, None, from_states=_OPEN_STATES)
        finally:
            with _LOCK:
                self._closing = False
                if self._published:
                    _LIVE.pop(self.owner_id, None)
                    if self.author_detached:
                        _CLOSED_DETACHED[self.owner_id] = utc_now_iso()


def local_reviews_settled() -> bool:
    """The idle probe's review half: no local operation or attempt is owed. Closed operations
    whose final pointer is unpublished retry first, bounded to this process's registry (no
    history scan; server maintenance reuses this retry); a failure keeps them registered."""
    from ouroboros.review_custody import _ACTIVE, _ACTIVE_LOCK

    with _LOCK:
        pending = [op for op in _LIVE.values() if op.closed]
    for operation in pending:
        operation.maybe_close()
    with _LOCK, _ACTIVE_LOCK:
        return not _LIVE and not _ACTIVE


@dataclass
class OperationBinding:
    """One caller's hold on an operation: the slots it may not send, with the reason."""

    operation: Optional[ReviewOperation] = None
    refused: Dict[str, str] = field(default_factory=dict)


def current_review_operation() -> Optional[ReviewOperation]:
    return _BOUND.get()


def release_review_operation(entry: Any) -> None:
    """A settled physical worker may be its operation's last one."""
    operation = getattr(entry, "operation", None)
    if operation is not None:
        operation.maybe_close()


def stamp_review_controller(actor: Any, entry: Any) -> None:
    """A row still owed by a live worker names that worker's controller."""
    operation = getattr(entry, "operation", None)
    usage = dict(getattr(actor, "usage", None) or {})
    usage["review_controller"] = {**controller_identity(), **({"owner_id": operation.owner_id} if operation else {})}
    actor.usage = usage


def _result_root(usage_ctx: Any, parent: Any = None) -> Optional[pathlib.Path]:
    meta = getattr(usage_ctx, "task_metadata", None)
    meta = meta if isinstance(meta, dict) else {}
    root = (meta.get("budget_drive_root") or getattr(usage_ctx, "budget_drive_root", None)
            or getattr(usage_ctx, "drive_root", None) or getattr(parent, "canonical_root", None))
    return pathlib.Path(root) if root else None


@contextlib.contextmanager
def review_operation_scope(*, request: Any, slots: List[Any], usage_ctx: Any,
                           task_id: str) -> Iterator[OperationBinding]:
    """Own one panel's wait, controls and lifetime before computing slot windows.

    Drain and workers share operation clocks. If every slot relays one live local
    operation, bind its controller/clocks/scopes; otherwise own only the new sends.
    Without a task wait owner (collection or dispatch outside a task frame), bind
    nothing."""
    from ouroboros.model_wait import current_model_wait, operation_wait_scope
    from ouroboros.review_custody import review_dispatch_plan
    from ouroboros.review_live_join import live_join_operation

    parent = current_model_wait()
    reconcile = bool(getattr(request, "reconcile_only", False))
    live = live_join_operation(request, slots, usage_ctx) if parent is not None and reconcile else None
    if parent is None or (reconcile and live is None):
        yield OperationBinding()
        return
    plan = review_dispatch_plan(request, slots, usage_ctx)
    sends = {slot_id: operation_id for slot_id, (kind, operation_id) in plan.items() if kind == "send"}
    relayed = {id(op): op for kind, op in plan.values() if kind == "relay" and op is not None and not op.closed}
    binding = OperationBinding()
    if live is not None:
        binding.operation = live  # no new owner, even if settlement races
    elif not sends and len(relayed) == 1:
        binding.operation = next(iter(relayed.values()))
    else:
        root = _result_root(usage_ctx, parent)
        binding.operation = ReviewOperation(parent=parent, request=request, task_id=str(task_id or ""),
                                            result_root=root or pathlib.Path(str(parent.canonical_root)))
        with _LOCK:
            _LIVE[binding.operation.owner_id] = binding.operation
        if sends:
            from ouroboros.review_pause import author_fence_closed

            if author_fence_closed(parent):
                # A NEW panel under the owner's accepted Pause never starts: every
                # send slot is refused at $0, exactly as a fenced tool handoff is.
                from ouroboros.owner_pause import NOT_STARTED_TEXT

                binding.refused = {slot_id: NOT_STARTED_TEXT for slot_id in sends}
            else:
                _retain_before_dispatch(binding.operation, sends, usage_ctx, root)
    operation = binding.operation
    operation.enter()
    try:
        token = _BOUND.set(operation)
        try:
            from ouroboros.review_pause import author_drain

            with author_drain(parent, operation), operation_wait_scope(operation.wait):
                yield binding
        finally:
            _BOUND.reset(token)
    finally:
        operation.drain_returned()


def _retain_before_dispatch(operation: ReviewOperation, sends: Dict[str, str],
                            usage_ctx: Any, root: Optional[pathlib.Path]) -> None:
    """Record the operation's pointer before sends.

    Each sending operation's task-result pointer names controller, panel, owned
    slots and attempt so other processes can address it; it is removed when the
    operation closes. A missing pointer forbids remote decisions only."""
    reserved_elsewhere = (getattr(usage_ctx, "_review_reserved_operations", None) or {}).get(operation.surface)
    reserved_elsewhere = reserved_elsewhere if isinstance(reserved_elsewhere, dict) else {}
    owned = {slot_id: operation_id or str(reserved_elsewhere.get(slot_id) or "")
             for slot_id, operation_id in sends.items()}
    try:
        if usage_ctx is None or root is None:
            raise ValueError("the operation has no task-result root to retain its checkpoint")
        _write_operation_pointer(operation, owned)
    except Exception:
        log.debug("Review operation %s has no durable pointer", operation.owner_id, exc_info=True)
        return
    operation.checkpointed = True


def _write_operation_pointer(operation: ReviewOperation, owned: Dict[str, str]) -> None:
    """Write the pointer, then prove it landed."""
    entry = {"surface": operation.surface, "retry_key": operation.retry_key, "task_attempt": operation.wait.attempt,
             "controller": dict(operation.identity), "operations": dict(owned), "recorded_at": utc_now_iso(),
             "state": OPERATION_RETAINED}
    written = _update_operations(operation.result_root, operation.task_id,
                                 lambda rows: {**rows, operation.owner_id: entry}, create=True)
    stored = (written.get(OPERATIONS_FIELD) or {}).get(operation.owner_id)
    if not isinstance(stored, dict) or stored.get("recorded_at") != entry["recorded_at"]:
        raise ValueError("the operation checkpoint pointer did not land in the task result")


def _task_operation_entries(root: Any, task_id: str, *, result_loader=None) -> Iterator[tuple]:
    """Resolve the task's existing primary/control addresses without inventing liveness.

    A ``control_only`` link exists only on historical task-acceptance late reviews:
    it resolves to its subject task's primary pointer."""
    from ouroboros.task_results import load_task_result
    read = result_loader or (lambda tid: load_task_result(root, tid, strict=True) or {})
    row = read(task_id)
    for owner, entry in (row.get(OPERATIONS_FIELD) or {}).items():
        subject = task_id
        if entry.get("control_only"):
            subject = entry.get("subject_task_id")
            target = read(subject)
            primary = (target.get(OPERATIONS_FIELD) or {}).get(owner) or {}
            # The immutable intent survives the upgrade; Stop must remain
            # addressable between primary-pointer and control-link writes.
            keys = ("controller", "intent_ref") if entry.get("intent_ref") else ("controller", "source_ref")
            if any(primary.get(key) != entry.get(key) for key in keys):
                continue
            entry = primary
        yield owner, subject, entry


def task_has_live_review_operation(root: Any, task_id: str, *, exclude_owner_id: str = '',
                                   sent_only: bool = False, result_loader=None) -> bool:
    """Physical review ownership; unsent preparation is excluded by ``sent_only``."""
    for owner, _subject, entry in _task_operation_entries(root, task_id, result_loader=result_loader):
        if owner == exclude_owner_id:
            continue
        if entry.get("state") not in _OPEN_STATES or sent_only and entry.get("state") == OPERATION_PREPARING:
            continue
        with _LOCK:
            live = _LIVE.get(owner)
        if live is not None and not live.closed:
            return True
        if controller_state(entry.get("controller")) in {"alive", "unknown"}:
            return True
    return False


def _update_operations(root: Any, task_id: str, transform: Callable[[Dict[str, Any]], Any], *,
                       create: bool = False) -> Dict[str, Any]:
    """Change only the operation pointers of one task result, under its lock; returns the stored row.

    Never a status writer: the projector re-asserts the stored status.
    """
    from ouroboros.task_results import STATUS_RUNNING, write_task_result

    def project(current: Dict[str, Any], fields: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not current:
            return None  # a pointer never creates a task result
        rows = current.get(OPERATIONS_FIELD)
        rows = dict(rows) if isinstance(rows, dict) else {}
        if not rows and not create:
            return None
        updated = transform(rows)
        if updated is None:
            return None
        return {OPERATIONS_FIELD: updated, "status": current.get("status") or fields["status"]}

    return write_task_result(root, task_id, STATUS_RUNNING, _field_projector=project, strict_existing_dict=True)


def _mark_operation(root: Any, task_id: str, owner_id: str, state: Optional[str], *, from_states: frozenset) -> bool:
    """Move one pointer from an expected state only (``None`` removes it); a fact, fail-soft."""
    def transform(rows: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        row = rows.get(owner_id)
        if not isinstance(row, dict) or row.get("state") not in from_states:
            return None
        if state is None:
            return {key: value for key, value in rows.items() if key != owner_id}
        return {**rows, owner_id: {**row, "state": state, f"{state}_at": utc_now_iso()}}

    try:
        _update_operations(root, task_id, transform)
        return True
    except Exception:
        log.warning("Review operation %s could not record %s", owner_id, state, exc_info=True)
        return False


# --- exact live controller recognition (gateway decisions, supervisor events) ---

def _open_pointer(result: Dict[str, Any], row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The still-open checkpoint pointer of the exact operation a wait row names."""
    block = row.get("review_operation") if isinstance(row, dict) else None
    owner = str(block.get("owner_id") or "") if isinstance(block, dict) else ""
    pointer = (result.get(OPERATIONS_FIELD) or {}).get(owner) if owner else None
    if (not isinstance(pointer, dict) or row.get("model_wait_owner_id") != owner
            or pointer.get("state") not in _OPEN_STATES or pointer.get("controller") != block.get("controller")
            or pointer.get("retry_key") != block.get("retry_key")
            or str(block.get("slot_id") or "") not in (pointer.get("operations") or {})
            or pointer.get("task_attempt") != row.get("task_attempt")):
        return None
    return pointer


def _remote_waiting_row(root: Any, task_id: str, wait_id: str) -> Optional[Dict[str, Any]]:
    """The canonical waiting row of a live operation in ANOTHER process of this server generation.

    The row must still be waiting under its operation's own owner id, and the
    operation's pointer must still be open (it closes when the operation ends)
    and name the same controller, panel, slot and attempt. The controller must
    be alive with its exact pid birth and this generation's custody session;
    Stop, Panic or an unreadable stop state refuse.
    """
    from ouroboros.process_custody import current_custody_session_id
    from ouroboros.task_results import load_task_result

    try:
        result = load_task_result(root, task_id, strict=True) or {}
    except Exception:
        return None
    row = (result.get("model_waits") or {}).get(str(wait_id))
    block = row.get("review_operation") if isinstance(row, dict) else None
    if (not isinstance(block, dict) or row.get("state") != "waiting" or not block.get("owner_id")
            or row.get("model_wait_owner_id") != block["owner_id"]):
        return None
    if _open_pointer(result, row) is None:
        return None
    identity = block.get("controller") or {}
    if (controller_state(identity) != "alive" or str(identity.get("session") or "") != current_custody_session_id()
            or operation_stop_state(pathlib.Path(root), str(task_id))):
        return None
    return row


class _RemoteOperationController:
    """A live review controller in another process, addressed by its exact wait row."""

    consumes_pending_action = True

    def __init__(self, root: Any, task_id: str, wait_id: str, row: Dict[str, Any]):
        self.owner_id = str(row["review_operation"]["owner_id"])
        self.root, self.task_id, self.wait_id = pathlib.Path(root), str(task_id), str(wait_id)
        self.attempt = int(row.get("task_attempt") or 1)
        self.task = {"id": self.task_id, "_attempt": self.attempt, "chat_id": row["review_operation"].get("chat_id")}
        self.lock = threading.RLock()

    @property
    def closed(self) -> bool:
        row = _remote_waiting_row(self.root, self.task_id, self.wait_id)
        return row is None or row["review_operation"].get("owner_id") != self.owner_id

    def mutate_row(self, wait_id: str, transform: Callable) -> dict:
        from ouroboros.model_wait import mutate_wait

        return mutate_wait(self.root, self.task_id, wait_id, transform)


def _live_operation_owner(task_id: str, wait_id: str) -> Any:
    with _LOCK:
        operations = [op for op in _LIVE.values() if op.task_id == str(task_id) and not op.closed]
    for operation in operations:
        with operation.wait.lock:
            held = str(wait_id) in operation.wait.waits
        if held and not operation.control() and operation.control_state != "unknown":
            return operation.wait
    return None


def names_review_operation(root: Any, task_id: str, wait_id: str) -> bool:
    """Whether the canonical wait row belongs to a review operation (then only its controller consumes it)."""
    from ouroboros.task_results import load_task_result

    row = ((load_task_result(root, task_id) or {}).get("model_waits") or {}).get(str(wait_id))
    return isinstance(row, dict) and isinstance(row.get("review_operation"), dict)


def review_operation_controller(root: Any, task_id: str, wait_id: str) -> Any:
    """The exact live controller of one review-operation wait, here or in another process.

    ``None`` whenever nobody can provably consume a decision now: no such row,
    not an operation's wait, a closed operation, a latched or unreadable
    Stop/Panic, or a controller that is dead, reused, of another server
    generation or not provably alive.
    """
    owner = _live_operation_owner(task_id, wait_id)
    if owner is not None:
        return owner
    row = _remote_waiting_row(root, task_id, wait_id)
    return _RemoteOperationController(root, task_id, wait_id, row) if row is not None else None


def review_operation_event_admitted(root: Any, payload: Dict[str, Any]) -> bool:
    """Whether a wait event of an ended author comes from its exact live operation.

    A resolution can only finish a card, never revive one; a waiting row needs
    a live controller whose canonical row still holds exactly this revision.
    """
    block = payload.get("review_operation")
    if not isinstance(block, dict) or not block.get("owner_id") or payload.get("model_wait_owner_id") != block.get("owner_id"):
        return False
    if payload.get("state") == "resolved":
        return True
    wait_id = str(payload.get("wait_id") or "")
    with _LOCK:
        operation = _LIVE.get(str(block["owner_id"]))
    if operation is not None and not operation.closed:
        with operation.wait.lock:
            current = operation.wait.waits.get(wait_id) or {}
        return current.get("revision") == payload.get("revision")
    row = _remote_waiting_row(root, str(payload.get("task_id") or ""), wait_id)
    return row is not None and row.get("revision") == payload.get("revision")


# --- pure collection -----------------------------------------------------------

def recorded_review_assignment(root: Any, *, task_id: str = "", surface: str = "", row: dict,
                               retry_key: str = "", root_task_id: str = "",
                               review_contract: str = "", assignment: Optional[Tuple[Any, Any]] = None) -> Tuple[Any, Any]:
    """Read a paid seat's original inputs, proving its wave and any supplied assignment.

    No current config: pointer, prompt and full binding must agree; missing is a gap.
    Absent session_access retains historical readonly. Old processing ``""`` meant
    unspecified (sent as None, omitted from keys); only its exact recorded binding
    restores that meaning. New captured-empty bindings stay ``""``. Evidence, never a new send.
    """
    from ouroboros.observability import read_call_payload
    from ouroboros.review_dispatch import recorded_binding_conflicts, recorded_review_slot, review_operation_binding
    from ouroboros.review_records import ReviewRequest, ReviewSlot, ReviewRouteKind

    if assignment is not None:
        current_request, current_slot = assignment
        task_id, surface, retry_key = current_request.task_id, current_request.surface, current_request.retry_key
    operation = str(row.get("operation_id") or "")
    if not operation:
        raise ValueError("recorded operation identity missing")
    manifest, prompt, _refs = read_call_payload(
        root, task_id=task_id or "review", call_id=f"{operation}_prompt")
    request = ReviewRequest(**prompt["request"])
    slot = ReviewSlot(**{**prompt["slot"], "route": ReviewRouteKind(prompt["slot"]["route"])})
    slot = recorded_review_slot(request, slot, operation, manifest.get("review_operation_binding"))
    expected = review_operation_binding(request, slot, operation)
    if (manifest.get("review_operation_binding") != expected
            or not all(expected.get(key) for key in ("subject_hash", "review_contract", "roster_hash", "root_task_id"))):
        raise ValueError("original prompt operation binding mismatch")
    if (request.task_id != task_id or request.surface != surface or request.retry_key != retry_key
            or slot.slot_id != str(row.get("slot_id") or row.get("slot") or "")):
        raise ValueError("original request task/material/slot mismatch")
    if root_task_id and expected.get("root_task_id") != root_task_id:
        raise ValueError("original request root mismatch")
    if review_contract and expected.get("review_contract") != review_contract:
        raise ValueError("original request contract mismatch")
    if recorded_binding_conflicts(row.get("recovery_binding"), expected):
        raise ValueError("recorded wave binding mismatch")
    if assignment is not None:
        slot = recorded_review_slot(current_request, current_slot, operation, expected)
        if review_operation_binding(current_request, slot, operation) != expected:
            raise ValueError("operation/task/root/material/contract/roster binding mismatch")
    return request, slot


def collect_recorded_review_actor(root: Any, *, task_id: str, surface: str, row: dict,
                                  retry_key: str, usage_ctx: Any = None, root_task_id: str = "",
                                  review_contract: str = "") -> dict:
    """Collect a frozen consumer row using its recorded assignment, never current slots."""
    from ouroboros.review_records import ReviewSlot

    try:
        request, slot = recorded_review_assignment(
            root, task_id=task_id, surface=surface, row=row, retry_key=retry_key,
            root_task_id=root_task_id, review_contract=review_contract)
        actor, fact = _collect_pending_row(root, request, slot, row, usage_ctx, None)
        if fact == COLLECTED:
            from ouroboros.review_seat_outcome import publish_seat_execution

            publish_seat_execution(request, slot, actor, usage_ctx)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        slot = ReviewSlot(slot_id=str(row.get("slot_id") or row.get("slot") or ""),
                          model=str(row.get("model") or row.get("model_id") or ""))
        actor = _row_actor(slot, str(row.get("operation_id") or ""), row, state="custody_lost",
                           reason=f"Original review assignment unavailable: {exc}")
    return asdict(actor)


def recover_review_producer(root: Any, request: Any, slot: Any, row: dict) -> Any:
    """Read complete, bound producer CAS; the caller owns aggregation, never a new send.
    Missing is pending; unreadable/differently bound is custody loss with its
    full source, never PASS. Positive pre-start evidence needs no invocation.
    """
    from ouroboros.observability import read_call_payload
    from ouroboros.review_custody import finalize_review_actor
    from ouroboros.review_dispatch import recorded_invocation_tokens, review_operation_binding
    from ouroboros.review_records import ReviewActorRecord

    operation = str(row.get("operation_id") or "")
    if not operation or not root:
        return None
    expected = dict(row.get("recovery_binding") or {})
    refs, payload = {}, None
    try:
        for suffix in ("response", "error"):
            try:
                manifest, payload, refs = read_call_payload(
                    root, task_id=request.task_id or "review", call_id=f"{operation}_{suffix}")
                break
            except FileNotFoundError:
                continue
        if payload is None:
            return None
        outcome = payload.get("producer_outcome") if isinstance(payload, dict) else None
        if not manifest.get("producer_complete") or not isinstance(outcome, dict):
            return None  # Historical/partial blobs carry no completed-producer receipt.
        try:
            original_request, original_slot = recorded_review_assignment(root, row=row, assignment=(request, slot))
        except FileNotFoundError:
            if manifest.get("collection") != "attach_only_observation":
                raise
            original_request, original_slot = request, slot  # no retained prompt: the recorded assignment binds
        expected = review_operation_binding(original_request, original_slot, operation)
        binding = outcome.get("recovery_binding")
        if not isinstance(binding, dict) or manifest.get("review_operation_binding") != binding:
            raise ValueError("producer binding missing or inconsistent")
        if {k: v for k, v in binding.items() if k != "pending_invocation_id"} != expected:
            raise ValueError("operation/task/root/material/contract/roster binding mismatch")
        try:
            _prompt_manifest, _prompt, prompt_ref = read_call_payload(
                root, task_id=request.task_id or "review", call_id=f"{operation}_prompt")
        except FileNotFoundError:
            if manifest.get("collection") != "attach_only_observation":
                raise
            prompt_ref = None  # an older invocation-only row observed cold: its assignment is the recorded one
        token = str(binding.get("pending_invocation_id") or "")
        frozen_tokens = recorded_invocation_tokens(row)
        if frozen_tokens and frozen_tokens != {token}:
            raise ValueError("recorded pending invocation mismatch")
        if str(getattr(slot.route, "value", slot.route)) == "agent_session":
            from ouroboros.review_session_custody import validate_producer_invocation
            validate_producer_invocation(root, token, expected, payload, outcome, error_source=suffix == "error")
        if outcome.get("operation_id") != operation or outcome.get("slot_id") != slot.slot_id:
            raise ValueError("producer actor identity mismatch")
        # The outcome is stamped before its failure is logged; this verified operation's row holds the receipt (#1578).
        actor = ReviewActorRecord(**{**outcome, "error_event_logged": bool(
            outcome.get("error_event_logged") or row.get("error_event_logged"))})
        message = payload.get("message")
        actor.raw_text = str(message.get("content") or "") if isinstance(message, dict) else ""
        actor.usage = dict(payload.get("usage") or {})
        actor.prompt_ref, actor.response_ref = prompt_ref, refs
        finalize_review_actor(actor, operation_id=operation, late=True)
        if actor.operation_state in {"in_flight", "custody_lost"}:
            actor.status, actor.raw_text = "error", ""
            actor.error = actor.error or "Producer outcome still lacks terminal custody; full partial source retained"
        from ouroboros.review_seat_outcome import finalize_seat_outcome

        finalize_seat_outcome(original_request, original_slot, actor)
        return actor
    except (OSError, ValueError, KeyError, TypeError) as exc:
        actor = _row_actor(slot, operation, row, state="custody_lost",
                           reason=f"Exact persisted review result unavailable: {exc}")
        actor.response_ref = refs or actor.response_ref
        actor.recovery_binding = dict(row.get("recovery_binding") or expected)
        return actor


def _row_actor(slot: Any, operation_id: str, row: Optional[dict], *, state: str, reason: str) -> Any:
    """A typed gap (``custody_lost``) or a proven zero-send row (``not_dispatched``)."""
    from ouroboros.review_records import ReviewActorRecord

    row = row or {}
    lost = state == "custody_lost"
    usage = copy.deepcopy(dict(row.get("usage") or {}))
    if lost and row.get("pending_invocation_id"):
        usage.setdefault("pending_invocation_id", row["pending_invocation_id"])
    return ReviewActorRecord(
        slot_id=slot.slot_id, model=slot.model, status="error" if lost else "not_dispatched", error=reason,
        failure_code="review_custody_lost" if lost else "", operation_id=operation_id,
        operation_state=state, late_result_pending=lost, usage=usage,
        prompt_ref=dict(row.get("prompt_ref") or {}), response_ref=dict(row.get("response_ref") or {}),
        recovery_binding=dict(row.get("recovery_binding") or {}))


def _invocation_for_operation(root: Any, slot: Any, operation_id: str, row: dict) -> Optional[Dict[str, Any]]:
    """The durable delegated invocation this operation requested, if one is on record."""
    from ouroboros import delegate_invocation

    token = str(row.get("pending_invocation_id") or (row.get("usage") or {}).get("pending_invocation_id")
                or (row.get("recovery_binding") or {}).get("pending_invocation_id") or "")
    if not token:
        return delegate_invocation.operation_record(root, operation_id, str(slot.slot_id))
    record = delegate_invocation.invocation_record(root, token)
    return {**record, "invocation_id": token} if record is not None else None


def _observe_delegated_review(root: Any, request: Any, slot: Any, row: dict, operation_id: str,
                              *, controller_known: bool = True, usage_ctx: Any = None) -> Tuple[Any, str]:
    """Attach-only reading of a delegated reviewer whose worker is gone.

    Only a terminal run read, or the daemon's definite refusal, settles the row.
    A missing start request is not proof of no send (the custody history can be
    incomplete); with an unknown controller nothing short of a terminal read
    settles it, because that controller may still be binding its run.
    """
    from ouroboros.review_custody import _frozen_actor
    from ouroboros.review_dispatch import (
        recorded_binding_conflicts, recorded_invocation_tokens, review_operation_binding)

    if recorded_binding_conflicts(row.get("recovery_binding"), review_operation_binding(request, slot, operation_id)):
        return _row_actor(slot, operation_id, row, state="custody_lost",
                          reason="The recorded review assignment binding is inconsistent."), UNAVAILABLE
    record = _invocation_for_operation(root, slot, operation_id, row)
    if record is not None and record.get("state") == "legacy_gap":
        actor = _row_actor(slot, operation_id, row, state="custody_lost", reason=(
            f"Legacy invocation custody is incomplete (legacy_gap: {record['reason']}); "
            "it cannot prove nothing was sent, so the reviewer is never re-sent."))
        if not controller_known:
            actor = _frozen_actor(row, slot)
        actor.failure_code = "review_custody_legacy_gap"
        actor.usage["custody_state"] = "legacy_gap"
        return actor, UNAVAILABLE if controller_known else DEFERRED
    if record is not None:
        expected = {"task_id": request.task_id, "surface": request.surface, "slot_id": slot.slot_id,
                    "operation_id": operation_id,
                    "root_task_id": (request.reconciliation_identity or {}).get("root_task_id") or request.task_id}
        tokens = recorded_invocation_tokens(row)
        if (any(str(record.get(k) or "") != str(value or "") for k, value in expected.items())
                or (tokens and tokens != {str(record.get("invocation_id") or "")})):
            return _row_actor(slot, operation_id, row, state="custody_lost",
                              reason="The recorded delegated invocation belongs to another operation."), UNAVAILABLE
    if record is not None and record.get("state") == "started" and record.get("run_id"):
        observed = _attach_only_run(root, request, slot, operation_id, str(record["run_id"]), record, usage_ctx)
        return (observed, COLLECTED) if observed is not None else (_frozen_actor(row, slot), DEFERRED)
    if not controller_known:
        return _frozen_actor(row, slot), DEFERRED
    if record is None:
        return _row_actor(slot, operation_id, row, state="custody_lost", reason=(
            "No durable start request was found for this reviewer; the custody history cannot prove "
            "nothing was sent, so its outcome stays unknown and it is never re-sent.")), UNAVAILABLE
    if record.get("state") == "failed_definite":
        return _row_actor(slot, operation_id, row, state="not_dispatched",
                          reason="The daemon definitively refused this reviewer's start."), NEVER_DISPATCHED
    actor = _row_actor(slot, operation_id, row, state="custody_lost", reason=(
        "A start was requested but no run was ever bound to it; its outcome stays unknown "
        "and it is never re-posted."))
    # Only the seat's own token and the retained exact body let its review's collection replay it.
    if recorded_invocation_tokens(row) and isinstance(record.get("request"), dict) and record["request"]:
        actor.usage["custody_state"] = START_UNBOUND
    return actor, UNAVAILABLE


def _attach_only_run(root: Any, request: Any, slot: Any, operation_id: str, run_id: str, invocation: dict,
                     usage_ctx: Any = None) -> Any:
    """Read one existing run: authenticated protocol handshake, GET run, GET output, local parse.

    The handshake is the engine's non-generative protocol negotiation (an HTTP
    POST); nothing else here sends, starts, cancels, acknowledges or retires.
    ``None`` means the run is still live or the daemon cannot answer now.
    """
    from ouroboros import delegate_custody as custody
    from ouroboros.claudexor_daemon import read_owned_gateway
    from ouroboros.review_dispatch import review_operation_binding
    from ouroboros.review_execution import _full_session_text, session_identity_deltas
    from ouroboros.gateways.claudexor import final_attempt_facts
    from ouroboros.review_records import ReviewActorRecord
    from ouroboros.review_seat_outcome import retain_collected_outcome
    from ouroboros.review_verdict_extraction import canonicalize_session_verdict
    from ouroboros.review_session_usage import session_usage_facts
    from ouroboros.delegate_custody_usage import observe_session_usage
    from ouroboros.tools.review_helpers import emit_review_usage
    from ouroboros._usage_rows import REVIEW_ATTRIBUTION_KEYS
    from ouroboros.triad_review import default_output_contract, request_output_shape

    try:
        gateway = read_owned_gateway()
    except Exception:
        log.debug("attach-only review observation: daemon unavailable", exc_info=True)
        return None
    try:
        detail = gateway.get_run(run_id, timeout_sec=_OBSERVE_TIMEOUT_SEC)
        if not custody.is_terminal(detail):
            return None
        summary = custody.summary_of(detail)
        state = str(summary.get("state") or "")
        observed = final_attempt_facts(detail, run_id)
        routes = (invocation.get("request") or {}).get("harnesses") or []
        requested_route = str(routes[0]) if len(routes) == 1 else ""
        usage = session_usage_facts(run_id, requested_route=requested_route, observed_attempt=observed,
                                    summary=summary, attempt_execution=detail.get("attemptExecution"))
        if usage_ctx is not None:  # Observe terminal facts even if the output read below fails.
            observe_session_usage(lambda facts: emit_review_usage(
                usage_ctx, model=str(facts.get("resolved_model") or ""), usage=facts,
                source=f"review_substrate:{request.surface}", extra={
                    "task_id": request.task_id, "root_task_id": invocation.get("root_task_id") or request.task_id,
                    "parent_task_id": invocation.get("parent_task_id") or "",
                    "delegation_role": "subagent" if invocation.get("parent_task_id") else "",
                    **{key: str(invocation.get(key) or "") for key in REVIEW_ATTRIBUTION_KEYS},
                    "surface": request.surface, "slot_id": slot.slot_id}), usage)
        text = _full_session_text(gateway, run_id, detail) if state == "succeeded" else ""
    except Exception:
        log.debug("attach-only review observation of %s did not complete", run_id, exc_info=True)
        return None
    finally:
        try:
            gateway.close()
        except Exception:
            log.debug("attach-only gateway close failed", exc_info=True)
    usage.update(invocation_id=str(invocation.get("invocation_id") or ""),
                 collection="attach_only_observation", cost=None, run_state=state)
    try:
        from ouroboros.review_checking import terminal_seat_evidence

        # The receipt is written before the seat's one-shot copy is deleted.
        evidence = terminal_seat_evidence(root, request.task_id, operation_id, detail)
    except Exception as exc:
        # The paid verdict remains collectible even if retaining its checking
        # artifacts fails; leave the original area in custody and name the gap.
        log.warning("Collected review evidence capture failed: %s", operation_id, exc_info=True)
        evidence = {"status": "capture_failed", "operation_id": operation_id,
                    "error": f"{type(exc).__name__}: {exc}"}
    if evidence:
        usage["checking_evidence"] = evidence
    if isinstance(observed.get("effort_resolution"), dict):
        usage["effort_resolution"] = observed["effort_resolution"]
    # Requested identity comes only from this paid invocation, never live settings
    # or the summary's request echoes. Absent final-attempt facts stay unknown.
    if invocation.get("prompt_cache_ttl"):  # its recorded TTL intent, where a live seat's rides
        usage["review_source_delivery"] = {"prompt_cache_ttl": copy.deepcopy(invocation["prompt_cache_ttl"])}
    usage.update(resolved_model=observed.get("model", ""), observed_attempt=observed,
                 delegated_route=observed.get("harness_id", ""), requested_route=requested_route,
                 applied_profile=observed.get("profile_id", ""))
    usage["capability_delta"] = session_identity_deltas(slot, {
        "model": usage["resolved_model"], "route_id": requested_route,
        "effective_route_ids": [usage["delegated_route"]] if usage["delegated_route"] else []})
    binding = review_operation_binding(request, slot, operation_id)
    if invocation.get("invocation_id"):
        binding["pending_invocation_id"] = str(invocation["invocation_id"])
    if state != "succeeded":
        interrupted = state in {"cancelled", "interrupted"}
        actor = ReviewActorRecord(slot_id=slot.slot_id, model=slot.model, status="error", usage=usage,
                                  error=f"delegated review run {run_id} ended {state or 'without a state'}",
                                  failure_code="review_run_cancelled" if interrupted else "review_run_failed",
                                  transport_status="interrupted" if interrupted else "provider_transport_error",
                                  operation_id=operation_id, operation_state="late_settled", recovery_binding=binding)
        return retain_collected_outcome(root, request, slot, actor)
    shape = request_output_shape(request)
    canonical, method, _unused = canonicalize_session_verdict(
        text, conformance_passed=str(summary.get("outputConformance") or "").lower() == "passed",
        contract=str((request.policy or {}).get("output_contract") or "") or default_output_contract(shape),
        shape=shape, allow_extraction=False)
    usage.update(verdict_method=method, verdict_provenance={
        "raw_transcript_chars": len(text),
        "raw_transcript_sha256": hashlib.sha256(text.encode("utf-8", "replace")).hexdigest(),
        "verdict_method": method})
    actor = ReviewActorRecord(slot_id=slot.slot_id, model=slot.model,
                              status="ok" if canonical.strip() else "empty", raw_text=canonical, usage=usage,
                              operation_id=operation_id, operation_state="late_settled", recovery_binding=binding)
    return retain_collected_outcome(root, request, slot, actor, transcript=text)


def _observe_retained_observation_usage(root: Any, request: Any, slot: Any, actor: Any, usage_ctx: Any) -> None:
    """A retained cold observation hands its usage to the first collector with an endpoint.

    The observation itself was collected without one (or its endpoint already saw it);
    the process-local once-per-run latch keeps the row single and the attribution is the
    recorded invocation's, never the collector's scope."""
    usage = actor.usage or {}
    if usage_ctx is None or usage.get("collection") != "attach_only_observation":
        return
    from ouroboros._usage_rows import REVIEW_ATTRIBUTION_KEYS
    from ouroboros.delegate_custody import invocation_record
    from ouroboros.delegate_custody_usage import observe_session_usage
    from ouroboros.tools.review_helpers import emit_review_usage

    token = str((actor.recovery_binding or {}).get("pending_invocation_id") or usage.get("invocation_id") or "")
    invocation = (invocation_record(root, token) if token else None) or {}
    observe_session_usage(lambda facts: emit_review_usage(
        usage_ctx, model=str(facts.get("resolved_model") or ""), usage=facts,
        source=f"review_substrate:{request.surface}", extra={
            "task_id": request.task_id, "root_task_id": invocation.get("root_task_id") or request.task_id,
            "parent_task_id": invocation.get("parent_task_id") or "",
            "delegation_role": "subagent" if invocation.get("parent_task_id") else "",
            **{key: str(invocation.get(key) or "") for key in REVIEW_ATTRIBUTION_KEYS},
            "surface": request.surface, "slot_id": slot.slot_id}), usage)


def _collect_pending_row(root: Any, request: Any, slot: Any, row: dict, usage_ctx: Any,
                         controller: Any) -> Tuple[Any, str]:
    from ouroboros.review_custody import (
        _ACTIVE, _ACTIVE_LOCK, _attempt_key, _frozen_actor,
    )
    from ouroboros.review_dispatch import recorded_review_slot

    operation_id = str(row.get("operation_id") or "")
    slot = recorded_review_slot(request, slot, operation_id, row.get("recovery_binding"))
    if not row.get("recovery_binding") and root and str(getattr(slot.route, "value", slot.route)) == "agent_session":
        try:
            request, slot = recorded_review_assignment(root, row=row, assignment=(request, slot))
        except FileNotFoundError:
            pass  # Older invocation-only rows may have no retained prompt.
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return _row_actor(slot, operation_id, row, state="custody_lost",
                              reason=f"Original review assignment unavailable: {exc}"), UNAVAILABLE
    key = _attempt_key(request, slot)
    with _ACTIVE_LOCK:
        live = _ACTIVE.get(key)
        if live is not None and live.operation_id == operation_id:
            if live.event.is_set() and live.actor is not None:
                return copy.deepcopy(live.actor), COLLECTED
            return _frozen_actor(row, slot), DEFERRED  # its own worker settles it
    cached = (getattr(usage_ctx, "_review_settled_attempts", None) or {}).get(key)
    if cached is not None and str(getattr(cached, "operation_id", "") or "") == operation_id:
        return copy.deepcopy(cached), COLLECTED
    recovered = recover_review_producer(root, request, slot, row)
    if recovered is not None and recovered.operation_state != "in_flight":
        if recovered.operation_state != "custody_lost":
            _observe_retained_observation_usage(root, request, slot, recovered, usage_ctx)
        return recovered, UNAVAILABLE if recovered.operation_state == "custody_lost" else COLLECTED
    identity = controller if isinstance(controller, dict) else (row.get("usage") or {}).get("review_controller")
    # A row with no controller on record keeps the pre-operation reading (its
    # worker's custody is gone); a recorded controller that cannot be verified
    # is not presumed dead.
    state = controller_state(identity) if isinstance(identity, dict) and identity else "dead"
    if state != "alive" and str(getattr(slot.route, "value", slot.route) or "") == "agent_session":
        return _observe_delegated_review(root, request, slot, row, operation_id,
                                         controller_known=state != "unknown", usage_ctx=usage_ctx)
    if state in {"alive", "unknown"}:
        return _frozen_actor(row, slot), DEFERRED  # another process owns or may own this worker
    return _row_actor(slot, operation_id, row, state="custody_lost", reason=(
        "The review worker ended without a retained producer outcome; its API outcome is unknown "
        "and is never re-sent.")), UNAVAILABLE


# --- dead controllers' liveness pointers, through the existing maintenance pass --------

def retire_dead_review_pointers_softly(drive_root: Any, *, stop: Optional[Callable[[], bool]] = None) -> None:
    """The maintenance hook: one fail-soft pass that never stops the sweep around it."""
    try:
        report = retire_dead_review_pointers(drive_root, stop=stop)
        if report.get("retired") or report.get("reclassified"):
            log.info("Dead review operation pointers retired: %s", report)
    except Exception:
        log.warning("Dead review operation pointer retirement failed", exc_info=True)


def retire_dead_review_pointers(drive_root: Any, *, stop: Optional[Callable[[], bool]] = None) -> Dict[str, Any]:
    """Remove the open review-operation pointers nobody live will close, for ended tasks.

    Runs inside the existing off-loop maintenance pass, after this process's closed operations
    retried their own failed final pointer (``local_reviews_settled``; no history scan). An open
    pointer only proves liveness (the evidence packer keeps its task's loose files for a session
    reader, ``obligations.review_collection_pending``) and its controller removes it at close.
    Once the task's actor has ended and the pointer's controller is proven dead (or is this
    process without a registration) it is removed, never settled; a live or unverifiable
    controller keeps it. The retired task acceptance's pointers are such pointers too: nothing
    settles them any more; a row an older body flagged for that collection with nothing open is
    written back to 0 (``task_store.reclassify_review_collection``), so its evidence packs."""
    from ouroboros.store import StoreError
    from ouroboros.task_results import load_task_result
    from ouroboros.task_store import reclassify_review_collection, review_collection_candidates

    local_reviews_settled()
    root = pathlib.Path(drive_root)
    report: Dict[str, Any] = {"retired": [], "reclassified": []}
    for task_id in review_collection_candidates(root, stop=stop):
        if stop is not None and stop():
            break
        try:
            row = load_task_result(root, task_id) or {}
            if reclassify_review_collection(root, task_id):  # nothing open: an older body's flag, never owed work
                report["reclassified"].append(task_id)
                continue
        except (OSError, ValueError, StoreError):
            continue
        if not _actor_ended(root, task_id, row):
            continue
        for owner_id, entry in list((row.get(OPERATIONS_FIELD) or {}).items()):
            if not isinstance(entry, dict) or entry.get("control_only") or entry.get("state") not in _OPEN_STATES:
                continue
            with _LOCK:
                registered = owner_id in _LIVE
            controller = ("dead" if not registered and entry.get("controller") == controller_identity()
                          else controller_state(entry.get("controller")))
            if controller == "dead" and _mark_operation(root, task_id, owner_id, None, from_states=_OPEN_STATES):
                report["retired"].append({"task_id": task_id, "owner_id": owner_id})
    return report


def _actor_ended(root: pathlib.Path, task_id: str, row: Dict[str, Any]) -> bool:
    """Canonical completion or a positively ended actor drain; unknown is not ended. The maintenance
    pass has no live context, so it reads the row's own child binding (file copyback may keep the
    canonical row running after the solve loop ended)."""
    from ouroboros.owner_mailbox import mailbox_drain_ended
    from ouroboros.task_results import _TRULY_TERMINAL_STATUSES
    from ouroboros.task_status import _child_drive_candidates

    if str(row.get("status") or "") in _TRULY_TERMINAL_STATUSES:
        return True
    return mailbox_drain_ended(next(iter(_child_drive_candidates(row)), root), task_id)
