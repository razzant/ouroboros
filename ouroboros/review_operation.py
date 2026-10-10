"""An already-paid review operation: its own lifetime, controls and free collection.

A reviewer panel released at the dispatch barrier outlives the author turn that
bought it. The panel therefore owns ONE model-wait owner shared by its drain and
its workers (its own union and per-slot quota clocks, its own owner controls),
closed only after settlement publication duties are durable — never when the author's
scope closes. Explicit calendar deadlines, the task's money and root lineage and
the task's Stop/Panic stay binding; the author's own execution-lifetime scopes
do not (``model_wait.operation_wait_scope``). A caller that joins a panel whose
workers are all still live binds that same operation instead of minting one.
A review dispatched with no task wait owner in context gets no operation.

Before any physical send a task-acceptance operation retains its exact request,
roster, the operation ids it will own and its controller identity through the
existing immutable artifact store plus one pointer in the task result
(``review_operations``), read back before the send; a pointer that did not land
refuses those sends at $0. Owner decisions reach the operation as the canonical
wait row's ``pending_action`` (the author's mailbox is the author's), and only a
controller proven live — this process's registration, or another process of
this server generation with the exact pid birth and custody session and a
pointer still open that matches panel, slot and attempt, with no Stop/Panic or
unreadable stop state (``review_operation_controller``) — may be told to act. A panel whose controller died is discovered by the existing
maintenance pass.

Collection here is PURE: it restores an exact completed producer outcome, reads
an existing delegated run it can prove (attach-only; never a start, re-post,
cancel, acknowledgement, retirement or daemon ensure) and parses locally. It
never calls a model, not even a Light extraction. Absence of a record is never
proof that nothing was sent.
"""

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
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from ouroboros.model_wait import TaskModelWait
from ouroboros.utils import utc_now_iso

log = logging.getLogger(__name__)

# Task-result field: controller owner id -> the operation's durable checkpoint pointer.
OPERATIONS_FIELD = "review_operations"
# ``retained``: checkpoint written, no worker released by this controller yet (a
# crash right after a release can leave it, so it never proves "not sent");
# ``dispatched``: released to physical workers; ``closed``: the operation ended;
# ``unpublished``: it ended but publication or durable notice custody did not
# land (maintenance retries it); ``collected``: a late settlement was published and read back.
# Another surface's pointer only proves liveness and is removed when it closes.
OPERATION_RETAINED, OPERATION_DISPATCHED, OPERATION_CLOSED = "retained", "dispatched", "closed"
OPERATION_UNPUBLISHED, OPERATION_COLLECTED = "unpublished", "collected"
OPERATION_PREPARING = "preparing"
_OPEN_STATES = frozenset({OPERATION_PREPARING, OPERATION_RETAINED, OPERATION_DISPATCHED})
# Typed per-slot collection facts (never a verdict): a pending producer is not a
# never-dispatched one, and neither is an unknown outcome.
COLLECTED, DEFERRED, UNAVAILABLE, NEVER_DISPATCHED, SETTLED = (
    "collected", "deferred", "unavailable", "never_dispatched", "settled")
CHECKPOINT_REFUSAL = "review_operation_checkpoint_unavailable"
_COLLECTABLE = frozenset({"pending_dispatch", "in_flight"})
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
    """This process as a review controller: pid, its birth token and custody session."""
    from ouroboros.platform_layer import process_start_time
    from ouroboros.process_custody import current_custody_session_id

    pid = os.getpid()
    if pid not in _SELF:
        _SELF.clear()
        _SELF[pid] = {"pid": pid, "birth": process_start_time(pid)}
    return {**_SELF[pid], "session": str(current_custody_session_id() or "")}


def controller_state(identity: Any) -> str:
    """``local``, ``alive``, ``dead`` or ``unknown`` — never a guess.

    No live process with the pid (or only its exited zombie) proves the
    controller gone. A live pid is the recorded controller only with the same
    birth token (a reused pid proves the original gone); a live pid with no
    recorded or readable birth is ``unknown``.
    For our own pid/birth, distinct nonempty custody sessions prove the old
    controller gone after exec. Another process's session cannot prove that.
    """
    try:
        pid = int(identity.get("pid") or 0) if isinstance(identity, dict) else 0
    except (TypeError, ValueError):
        pid = 0
    if pid <= 0:
        return "unknown"
    birth = str(identity.get("birth") or "")
    if pid == os.getpid():
        own = controller_identity()
        if not birth or not own["birth"]:
            return "unknown"
        if birth != own["birth"]:
            return "dead"
        session = str(identity.get("session") or "")
        if not session or not own["session"]:
            return "unknown"
        return "local" if session == own["session"] else "dead"
    from ouroboros.platform_layer import pid_is_alive, process_start_time
    from ouroboros.process_containment import pid_is_zombie

    if not pid_is_alive(pid) or pid_is_zombie(pid):
        return "dead"
    live_birth = process_start_time(pid) if birth else ""
    if not live_birth:
        return "unknown"
    return "alive" if live_birth == birth else "dead"


def _operation_stop_state(root: pathlib.Path, task_id: str) -> str:
    """``panic``, ``cancelled`` (a current ungraceful Stop or a settled cancellation), ``""`` or ``unknown``.

    Read strictly: an unreadable intent or result is ``unknown``, never a clear.
    """
    try:
        if any((root / "state" / name).exists() for name in ("panic_stop.flag", "owner_restart_no_resume.flag")):
            return "panic"
        from ouroboros.cancel_intents import cancel_pending, resolve_owner_stop_intent
        from ouroboros.task_results import load_task_result

        if str((load_task_result(root, task_id, strict=True) or {}).get("status") or "") == "cancelled":
            return "cancelled"
        if cancel_pending(root, task_id, strict=True) and not resolve_owner_stop_intent(root, task_id)[1]:
            return "cancelled"
    except Exception:
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
        from ouroboros.task_results import task_result_path

        try:
            stat = task_result_path(pathlib.Path(self.canonical_root), self.task_id).stat()
        except OSError:
            return
        stamp = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
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
        self.historical_purpose = copy.deepcopy((getattr(request, "policy", None) or {}).get("historical_acceptance"))
        self.identity = controller_identity()
        self.result_root = pathlib.Path(result_root)
        self.checkpointed = False
        self.preparing = False
        self.closed = False
        self.control_state = ""
        self._drains = 0
        self._dispatched = False
        # The owner paused the author while this operation's launched reviewers
        # ran: the author took pending rows and parked (``review_custody``).
        self.author_detached = False
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
        """The task's Stop/Panic and settled cancellation stay binding; latched once seen.

        Re-read at most once per ``_CONTROL_RECHECK_SEC``: the live wait polls far
        more often, and each read parses the task result. An unreadable state is
        not a stop — the live wait keeps its paid call, as the author's fail-soft
        read does — but ``control_state`` says ``unknown`` and no decision is
        accepted for this operation while it does.
        """
        now = time.monotonic()
        if not self._control and now - self._checked_at >= _CONTROL_RECHECK_SEC:
            self._checked_at = now
            self.control_state = _operation_stop_state(self.result_root, self.task_id)
            if self.historical_purpose:
                from ouroboros.acceptance_late import historical_operation_controls, owner_paused_only

                blocked = historical_operation_controls(self.result_root, self.historical_purpose,
                                                        unstarted=not self._dispatched)
                # An owner Pause defers unsent work until Resume (D10); it is never a Stop.
                if blocked and not owner_paused_only(self.result_root, self.historical_purpose, blocked):
                    self.control_state = "cancelled"
            if self.control_state in {"panic", "cancelled"}:
                self._control = self.control_state
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
        """Close after drains, workers and their settlement publication duties finish."""
        from ouroboros.review_custody import operation_has_live_workers

        with _LOCK:
            if self.closed or self._drains or operation_has_live_workers(self):
                return
            self.closed = True
            _LIVE.pop(self.owner_id, None)
            if self.author_detached:
                _CLOSED_DETACHED[self.owner_id] = utc_now_iso()
        self.wait.close()
        if not self.checkpointed:
            return
        if self.surface != "task_acceptance":
            # A liveness-only pointer ends with its operation; the task result keeps no history of it.
            _mark_operation(self.result_root, self.task_id, self.owner_id, None, from_states=_OPEN_STATES)
            return
        from ouroboros.acceptance_settlement import late_publication_owed

        # A historical panel owes its FIRST supplement even if every actor
        # answered before the drain returned. Only the settlement publisher can
        # consume that duty; crashing before its first call stays recoverable.
        state = OPERATION_UNPUBLISHED if self.historical_purpose or late_publication_owed(self.task_id, self.retry_key) else OPERATION_CLOSED
        _mark_operation(self.result_root, self.task_id, self.owner_id, state, from_states=_OPEN_STATES)


@dataclass
class OperationBinding:
    """One caller's hold on an operation: the slots it may not send, with the reason."""

    operation: Optional[ReviewOperation] = None
    refused: Dict[str, str] = field(default_factory=dict)


def current_review_operation() -> Optional[ReviewOperation]:
    return _BOUND.get()


def prepare_historical_operation(*, root: Any, purpose: dict, event_queue: Any,
                                 work: Callable, background: bool, resume_entry: Optional[dict] = None) -> dict:
    """Own historical preparation BEFORE starting it; never a ready/paid checkpoint.

    The same operation upgrades its intent to a complete canonical request at
    the ordinary dispatch seam. Its drain owns the preparation worker, so Stop,
    calendar limits and crash visibility exist before expensive source work.
    Panic never joins this worker. Recovery of an abandoned intent buys nothing.
    """
    from ouroboros.artifacts import store_actor_source_bytes
    from ouroboros.model_wait import operation_wait_scope

    retry_key = 'task_acceptance:' + hashlib.sha256(purpose['debt_id'].encode()).hexdigest()
    task = {'id': purpose['task_id'], '_attempt': purpose['task_attempt'],
            'chat_id': purpose['confirmed_delivery']['chat_id'],
            'metadata': {'root_task_id': purpose['accounting_root_task_id']}}
    parent = TaskModelWait(task=task, drive_root=root, event_queue=event_queue, worker_slot_held=False)
    request = SimpleNamespace(surface='task_acceptance', retry_key=retry_key,
                              policy={'historical_acceptance': purpose})
    operation = ReviewOperation(parent=parent, request=request, task_id=task['id'], result_root=pathlib.Path(root),
                                owner_id=str((resume_entry or {}).get('owner_id') or ''))
    parent.close()
    operation.preparing = True
    operation.enter()
    with _LOCK:
        if operation.owner_id in _LIVE:
            raise ValueError('historical preparation already has a live owner')
        _LIVE[operation.owner_id] = operation
    try:
        intent = {'schema_version': 1, 'kind': 'historical_acceptance_preparation_intent',
                  'owner_id': operation.owner_id, 'controller': operation.identity, 'purpose': purpose}
        if resume_entry:
            intent['resumed_from_intent'] = resume_entry['entry']['intent_ref']
        ref = store_actor_source_bytes(root, task['id'], category='context_checkpoints',
            source_id='historical-acceptance-intent', extension='json',
            data=json.dumps(intent, sort_keys=True, ensure_ascii=False).encode())
        operation.intent_ref = ref
        entry = {'surface': 'task_acceptance', 'retry_key': retry_key, 'task_attempt': purpose['task_attempt'],
                 'controller': operation.identity, 'state': OPERATION_PREPARING, 'intent_ref': ref,
                 'recorded_at': utc_now_iso()}
        from ouroboros.owner_pause import fence_closed, read_fence
        pause = read_fence(root, purpose['accounting_root_task_id'])
        if fence_closed(pause):
            entry['preparation_pause'] = _preparation_pause(purpose['accounting_root_task_id'], pause, entry)
        def admit(rows: dict) -> dict:
            if resume_entry:
                if rows.get(operation.owner_id) != resume_entry['entry']:
                    raise ValueError('historical preparation resume identity changed')
                return {**rows, operation.owner_id: entry}
            if any(row.get('retry_key') == retry_key and (purpose['automatic']
                   or row.get('state') != 'preparation_refused') for row in rows.values()):
                raise ValueError('historical acceptance operation already retained')
            return {**rows, operation.owner_id: entry}
        written = _update_operations(root, task['id'], admit, create=True)
        if (written.get(OPERATIONS_FIELD) or {}).get(operation.owner_id) != entry:
            raise ValueError('historical preparation intent not retained')
        _link_historical_controls(operation, entry)
    except BaseException:
        operation.preparing = False
        operation.drain_returned()
        raise

    def prepare() -> dict:
        token = _BOUND.set(operation)
        try:
            with operation_wait_scope(operation.wait):
                result = work(operation)
            if operation.checkpointed and result.get('status') == 'owed':
                _update_operations(root, task['id'], lambda rows: {**rows, operation.owner_id: {
                    **rows[operation.owner_id], 'predispatch_refusal': result}})
            if not operation.checkpointed:
                def refused(rows: dict) -> dict:
                    row = rows.get(operation.owner_id) or {}
                    return {**rows, operation.owner_id: {**row, 'state': ('preparation_unknown' if row.get('source_ref') else 'preparation_refused'),
                            'preparation_outcome': result, 'finished_at': utc_now_iso()}}
                _update_operations(root, task['id'], refused)
            return result
        except BaseException:
            # A crash/unknown preparation is NOT an absent operation. No boot
            # recovery or repeated receipt may buy from this intent.
            _mark_operation(root, task['id'], operation.owner_id, 'preparation_unknown',
                            from_states={OPERATION_PREPARING})
            if not background:
                raise
            log.exception('Historical acceptance preparation failed for %s', task['id'])
            return {'status': 'unknown', 'reason': 'historical_preparation_failed', 'dispatched': None}
        finally:
            operation.preparing = False
            _BOUND.reset(token)
            operation.drain_returned()

    if not background:
        return prepare()
    # This is the existing operation's preparation worker, retained and live
    # before start, holding its drain until request handoff or typed refusal.
    from ouroboros.settings_integrity import copy_task_settings_context
    settings_context = contextvars.Context()
    copy_task_settings_context(settings_context)
    operation.preparation_worker = threading.Thread(target=settings_context.run, args=(prepare,),
        name=f'review-prepare-{operation.owner_id}', daemon=True)
    try:
        operation.preparation_worker.start()
    except BaseException:
        _mark_operation(root, task['id'], operation.owner_id, 'preparation_unknown', from_states={OPERATION_PREPARING})
        operation.preparing = False
        operation.drain_returned()
        raise
    return {'status': 'preparing', 'reason': 'historical_operation_retained', 'owner_id': operation.owner_id,
            'task_id': task['id'], 'debt_id': purpose['debt_id'], 'dispatched': False}


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
    """Own the wait, controls and lifetime of one dispatching panel.

    Bound BEFORE any slot window is computed, so the drain and the workers read
    the same operation clocks. A call whose every slot is relayed from one live
    operation of this process binds THAT operation (its controller, clocks and
    scopes); otherwise a new operation owns exactly the slots it will send, and
    a task-acceptance checkpoint that cannot be retained and read back refuses
    those sends. All request sources must already be canonical before entry: the
    checkpoint serializes ``asdict(request)``. No task wait owner in context (collection, or a dispatch
    without a task frame) binds nothing.
    """
    from ouroboros.model_wait import current_model_wait, operation_wait_scope
    from ouroboros.review_custody import review_dispatch_plan

    parent = current_model_wait()
    if parent is None or bool(getattr(request, "reconcile_only", False)):
        yield OperationBinding()
        return
    plan = review_dispatch_plan(request, slots, usage_ctx)
    sends = {slot_id: operation_id for slot_id, (kind, operation_id) in plan.items() if kind == "send"}
    relayed = {id(op): op for kind, op in plan.values() if kind == "relay" and op is not None and not op.closed}
    binding = OperationBinding()
    restore = None
    preparing = current_review_operation()
    root = _result_root(usage_ctx, parent)
    if (preparing is not None and preparing.preparing and preparing.task_id == str(task_id)
            and preparing.retry_key == request.retry_key and preparing.result_root == root):
        binding.operation = preparing
        binding.refused, restore = _retain_before_dispatch(preparing, request, slots, sends, usage_ctx, root)
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
                binding.refused, restore = _retain_before_dispatch(
                    binding.operation, request, slots, sends, usage_ctx, root)
    operation = binding.operation
    operation.enter()
    try:
        token = _BOUND.set(operation)
        try:
            with operation_wait_scope(operation.wait):
                yield binding
        finally:
            _BOUND.reset(token)
    finally:
        if restore is not None:
            restore()
        operation.drain_returned()


def _retain_before_dispatch(operation: ReviewOperation, request: Any, slots: List[Any], sends: Dict[str, str],
                            usage_ctx: Any, root: Optional[pathlib.Path]) -> Tuple[Dict[str, str], Any]:
    """Put the operation on record before any send; ``(refused, restore)``.

    Every operation that will send gets one pointer in its task result (its
    controller, panel, the slots it owns, the attempt): the proof another
    process needs before it tells this operation to act, closed when the
    operation ends. A task-acceptance operation also retains its exact request
    and roster there (the checkpoint maintenance collects from) under the
    operation ids it will send (the exact rejoined id, or a reserved new one),
    and its sends are refused at $0 when that cannot be written and read back.
    Elsewhere a missing pointer only means no decision from another process.
    A panel whose every row is refused at $0 sends nothing and owes nothing.
    """
    from ouroboros.observability import new_call_id
    from ouroboros.review_dispatch import task_acceptance_row_refusal

    acceptance = operation.surface == "task_acceptance"
    if acceptance and all(task_acceptance_row_refusal(request, slot)
                          for slot in slots if str(slot.slot_id) in sends):
        return {}, None
    reserved_elsewhere = (getattr(usage_ctx, "_review_reserved_operations", None) or {}).get(operation.surface)
    reserved_elsewhere = reserved_elsewhere if isinstance(reserved_elsewhere, dict) else {}
    owned = {slot_id: operation_id or (new_call_id(f"review_{operation.surface}_{slot_id}") if acceptance
                                        else str(reserved_elsewhere.get(slot_id) or ""))
             for slot_id, operation_id in sends.items()}
    try:
        if usage_ctx is None or root is None:
            raise ValueError("the operation has no task-result root to retain its checkpoint")
        _write_operation_pointer(operation, request, slots, owned, retain_source=acceptance,
                                 paid_authority=getattr(usage_ctx, "_review_paid_authority", None))
    except Exception as exc:
        if not acceptance:
            log.debug("Review operation %s has no durable pointer", operation.owner_id, exc_info=True)
            return {}, None
        log.warning("Review operation checkpoint unavailable for %s", operation.task_id, exc_info=True)
        refusal = f"{CHECKPOINT_REFUSAL}: {type(exc).__name__}: {exc} (no reviewer was called)"
        return {slot_id: refusal for slot_id in owned}, None
    operation.checkpointed = True
    if not acceptance:
        return {}, None
    reserved = {slot_id: operation_id for slot_id, operation_id in owned.items() if not sends[slot_id]}
    previous = getattr(usage_ctx, "_review_reserved_operations", None)
    setattr(usage_ctx, "_review_reserved_operations", {**(previous or {}), operation.surface: reserved})
    return {}, lambda: setattr(usage_ctx, "_review_reserved_operations", previous)


def _write_operation_pointer(operation: ReviewOperation, request: Any, slots: List[Any],
                             owned: Dict[str, str], *, retain_source: bool, paid_authority: Any = None) -> None:
    """Write the pointer (and, for a checkpoint, its immutable source), then prove it landed."""
    entry = {"surface": operation.surface, "retry_key": operation.retry_key, "task_attempt": operation.wait.attempt,
             "controller": dict(operation.identity), "operations": dict(owned), "recorded_at": utc_now_iso(),
             "state": OPERATION_RETAINED}
    if operation.historical_purpose:
        entry["intent_ref"] = copy.deepcopy(operation.intent_ref)
    if retain_source:
        from ouroboros.artifacts import store_actor_source_bytes
        from ouroboros.observability import redact_projection

        source = {"schema_version": 1, "owner_id": operation.owner_id, "task_id": operation.task_id, **entry,
                  "request": asdict(request), "slot_roster": [asdict(slot) for slot in slots]}
        if paid_authority:
            source["paid_authority"] = copy.deepcopy(paid_authority)
        source.pop("state")
        raw = json.dumps(redact_projection(source).value, ensure_ascii=False, sort_keys=True,
                         default=str).encode("utf-8")
        entry["source_ref"] = store_actor_source_bytes(
            operation.result_root, operation.task_id, category="context_checkpoints",
            source_id="acceptance-operation", data=raw, extension="json")
    def admit(rows: Dict[str, Any]) -> Dict[str, Any]:
        # One debt keeps one operation even before its first paid stamp. The
        # canonical pointer is the existing handoff owner; a concurrent/new
        # rationale cannot replace it or turn an unknown send into an empty slot.
        if operation.historical_purpose and any(row.get('retry_key') == operation.retry_key
                and row.get('state') != 'preparation_refused'
                for owner, row in rows.items() if owner != operation.owner_id):
            raise ValueError('historical acceptance operation already retained')
        return {**rows, operation.owner_id: entry}
    written = _update_operations(operation.result_root, operation.task_id, admit, create=True)
    stored = (written.get(OPERATIONS_FIELD) or {}).get(operation.owner_id)
    if not isinstance(stored, dict) or any(stored.get(key) != entry.get(key) for key in ("recorded_at", "source_ref")):
        raise ValueError("the operation checkpoint pointer did not land in the task result")
    _link_historical_controls(operation, entry)


def _link_historical_controls(operation: ReviewOperation, entry: dict) -> None:
    purpose = operation.historical_purpose or {}
    for control_task in {purpose.get("caller_task_id"), purpose.get("accounting_root_task_id")} - {None, "", operation.task_id}:
        # Stop may address either owner after its author ended. This is only an
        # address to the existing operation, never another collectible panel.
        link = {"control_only": True, "subject_task_id": operation.task_id,
                "controller": entry["controller"], "source_ref": entry.get("source_ref"),
                "intent_ref": entry.get("intent_ref")}
        linked = _update_operations(operation.result_root, control_task,
                                   lambda rows: {**rows, operation.owner_id: link}, create=True)
        if (linked.get(OPERATIONS_FIELD) or {}).get(operation.owner_id) != link:
            raise ValueError("the operation control address did not land")


def _task_operation_entries(root: Any, task_id: str, *, result_loader=None) -> Iterator[tuple]:
    """Resolve the task's existing primary/control addresses without inventing liveness."""
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


def _preparation_pause(task_id: str, fence: dict, entry: dict) -> dict:
    return {'root_task_id': task_id, 'fence_id': fence['fence_id'], 'generation': fence.get('generation'),
            'intent_ref': entry['intent_ref'], 'controller': entry['controller']}


def retain_preparing_owner_pause(root: Any, task_id: str, fence: dict) -> None:
    """Bind already-retained unsent preparation to the accepted owner Pause."""
    for owner, subject, entry in _task_operation_entries(root, task_id):
        if entry.get('state') != OPERATION_PREPARING or entry.get('source_ref') or not entry.get('intent_ref'):
            continue
        def retain(rows, owner=owner, entry=entry):
            if rows.get(owner) != entry:
                return None  # It upgraded to a request; its ordinary paid custody wins.
            return {**rows, owner: {**entry, 'preparation_pause': _preparation_pause(task_id, fence, entry)}}
        _update_operations(root, subject, retain)


def paused_acceptance_preparations(root: Any, task_id: str, fence_id: str, *, result_loader=None) -> list:
    """Durable owed work, separate from whether its controller is physically alive."""
    return [(owner, subject, entry) for owner, subject, entry in _task_operation_entries(root, task_id, result_loader=result_loader)
            if (entry.get('preparation_pause') or {}).get('root_task_id') == task_id
            and (entry.get('preparation_pause') or {}).get('fence_id') == fence_id
            and entry.get('state') != 'preparation_refused']


def stop_paused_acceptance_preparations(root: Any, task_id: str) -> bool:
    """Discard only a dead controller's unsent saved remainder; keep its evidence."""
    from ouroboros.owner_pause import fence_closed, launch_lock, read_fence

    stopped = []
    with launch_lock(root, task_id):
        fence = read_fence(root, task_id)
        if not fence_closed(fence):
            return False
        for owner, subject, entry in paused_acceptance_preparations(root, task_id, fence['fence_id']):
            with _LOCK:
                live = _LIVE.get(owner)
            if ((live is not None and not live.closed) or controller_state(entry.get('controller')) in {'alive', 'unknown'}
                    or entry.get('source_ref') or entry.get('state') not in {OPERATION_PREPARING, 'preparation_unknown'}):
                continue  # Live/paid operations still settle through their existing owners.
            def stop(rows, owner=owner, entry=entry):
                if rows.get(owner) != entry:
                    return None
                stopped.append(owner)
                return {**rows, owner: {**entry, 'state': 'preparation_refused', 'finished_at': utc_now_iso(),
                    'preparation_outcome': {'status': 'owed', 'reason': 'owner_stopped', 'dispatched': False}}}
            _update_operations(root, subject, stop)
    return bool(stopped)


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


def _mark_operation(root: Any, task_id: str, owner_id: str, state: Optional[str], *, from_states: frozenset) -> None:
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
    except Exception:
        log.warning("Review operation %s could not record %s", owner_id, state, exc_info=True)


def historical_publication_retained(root: Any, task_id: str, retry_key: str) -> None:
    """The one settlement publisher has read back the supplement and its outbox custody."""
    def transform(rows: Dict[str, Any]) -> Dict[str, Any]:
        return {owner: {**row, 'state': OPERATION_COLLECTED, 'collected_at': utc_now_iso()}
                if row.get('surface') == 'task_acceptance' and row.get('retry_key') == retry_key
                and row.get('state') in _OPEN_STATES | {OPERATION_UNPUBLISHED, OPERATION_CLOSED}
                else row for owner, row in rows.items()}
    _update_operations(root, task_id, transform)


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
            or _operation_stop_state(pathlib.Path(root), str(task_id))):
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

def recover_review_producer(root: Any, request: Any, slot: Any, row: dict) -> Any:
    """Resolve one known operation from its complete existing CAS, never dispatch.

    The caller retains its own wave writer and aggregator. A missing terminal
    artifact is still in flight; an unreadable or differently bound artifact is
    custody loss with the original full source attached, never a semantic PASS.
    """
    from ouroboros.observability import read_call_payload
    from ouroboros.review_custody import finalize_review_actor
    from ouroboros.review_dispatch import review_operation_binding
    from ouroboros.review_records import ReviewActorRecord

    operation = str(row.get("operation_id") or "")
    if not operation or not root:
        return None
    expected = review_operation_binding(request, slot, operation)
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
        binding = outcome.get("recovery_binding")
        if not isinstance(binding, dict) or manifest.get("review_operation_binding") != binding:
            raise ValueError("producer binding missing or inconsistent")
        if {k: v for k, v in binding.items() if k != "pending_invocation_id"} != expected:
            raise ValueError("operation/task/root/material/contract/roster binding mismatch")
        frozen = row.get("recovery_binding")
        if frozen and {k: v for k, v in frozen.items() if k != "pending_invocation_id"} != expected:
            raise ValueError("recorded wave binding mismatch")
        prompt_manifest, prompt, prompt_ref = read_call_payload(
            root, task_id=request.task_id or "review", call_id=f"{operation}_prompt")
        if prompt_manifest.get("review_operation_binding") != expected:
            raise ValueError("original prompt operation binding mismatch")
        original_request = SimpleNamespace(**prompt["request"])
        original_slot = SimpleNamespace(**prompt["slot"])
        if review_operation_binding(original_request, original_slot, operation) != expected:
            raise ValueError("original request provenance mismatch")
        token = str(binding.get("pending_invocation_id") or "")
        frozen_token = str(row.get("pending_invocation_id") or
                           (row.get("usage") or {}).get("pending_invocation_id") or "")
        if frozen_token and token != frozen_token:
            raise ValueError("recorded pending invocation mismatch")
        if str(getattr(slot.route, "value", slot.route)) == "agent_session":
            from ouroboros.delegate_custody import invocation_record
            invocation = invocation_record(root, token) if token else None
            if not invocation or any(str(invocation.get(k) or "") != str(expected[k] or "")
                                     for k in ("task_id", "root_task_id", "surface", "slot_id", "operation_id")):
                raise ValueError("completed producer has no exact delegated invocation")
            run_id = str((payload.get("usage") or {}).get("delegated_run_id") or "")
            if run_id and run_id != str(invocation.get("run_id") or ""):
                raise ValueError("completed producer delegated run mismatch")
        if outcome.get("operation_id") != operation or outcome.get("slot_id") != slot.slot_id:
            raise ValueError("producer actor identity mismatch")
        actor = ReviewActorRecord(**outcome)
        message = payload.get("message")
        actor.raw_text = str(message.get("content") or "") if isinstance(message, dict) else ""
        actor.usage = dict(payload.get("usage") or {})
        actor.prompt_ref, actor.response_ref = prompt_ref, refs
        finalize_review_actor(actor, operation_id=operation, late=True)
        if actor.operation_state in {"in_flight", "custody_lost"}:
            actor.status, actor.raw_text = "error", ""
            actor.error = actor.error or "Producer outcome still lacks terminal custody; full partial source retained"
        return actor
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return ReviewActorRecord(
            slot_id=slot.slot_id, model=slot.model, status="error",
            error=f"Exact persisted review result unavailable: {exc}",
            failure_code="review_custody_lost", operation_id=operation,
            operation_state="custody_lost", late_result_pending=True,
            response_ref=refs, recovery_binding=expected,
        )



def _row_actor(slot: Any, operation_id: str, row: Optional[dict], *, state: str, reason: str) -> Any:
    """A typed gap (``custody_lost``) or a proven zero-send row (``not_dispatched``)."""
    from ouroboros.review_records import ReviewActorRecord

    row = row or {}
    lost = state == "custody_lost"
    return ReviewActorRecord(
        slot_id=slot.slot_id, model=slot.model, status="error" if lost else "not_dispatched", error=reason,
        failure_code="review_custody_lost" if lost else "", operation_id=operation_id,
        operation_state=state, late_result_pending=lost, usage=copy.deepcopy(dict(row.get("usage") or {})),
        response_ref=dict(row.get("response_ref") or {}), recovery_binding=dict(row.get("recovery_binding") or {}))


def _invocation_for_operation(root: Any, slot: Any, operation_id: str, row: dict) -> Optional[Dict[str, Any]]:
    """The durable delegated invocation this operation requested, if one is on record."""
    from ouroboros import delegate_custody as custody

    token = str(row.get("pending_invocation_id") or (row.get("usage") or {}).get("pending_invocation_id") or "")
    rows = list(custody.custody_rows(root))
    if not token:
        token = next((str(item.get("invocation_id") or "") for item in rows
                      if item.get("type") == custody.START_REQUESTED
                      and str(item.get("operation_id") or "") == operation_id
                      and str(item.get("slot_id") or "") == str(slot.slot_id)), "")
    return custody.invocation_record(root, token, rows=rows) if token else None


def _observe_delegated_review(root: Any, request: Any, slot: Any, row: dict, operation_id: str,
                              *, controller_known: bool = True) -> Tuple[Any, str]:
    """Attach-only reading of a delegated reviewer whose worker is gone.

    Only a terminal run read, or the daemon's definite refusal, settles the row.
    A missing start request is not proof of no send (the custody history can be
    incomplete); with an unknown controller nothing short of a terminal read
    settles it, because that controller may still be binding its run.
    """
    from ouroboros.review_custody import _frozen_actor

    record = _invocation_for_operation(root, slot, operation_id, row)
    if record is not None and record.get("state") == "started" and record.get("run_id") and (
            str(record.get("task_id") or "") == str(request.task_id or "")
            and str(record.get("operation_id") or operation_id) == operation_id):
        observed = _attach_only_run(root, request, slot, operation_id, str(record["run_id"]), record)
        return (observed, COLLECTED) if observed is not None else (_frozen_actor(row, slot), DEFERRED)
    if not controller_known:
        return _frozen_actor(row, slot), DEFERRED
    if record is None:
        return _row_actor(slot, operation_id, row, state="custody_lost", reason=(
            "No durable start request was found for this reviewer; the custody history cannot prove "
            "nothing was sent, so its outcome stays unknown and it is never re-sent.")), UNAVAILABLE
    if (str(record.get("task_id") or "") != str(request.task_id or "")
            or str(record.get("operation_id") or operation_id) != operation_id):
        return _row_actor(slot, operation_id, row, state="custody_lost",
                          reason="The recorded delegated invocation belongs to another operation."), UNAVAILABLE
    if record.get("state") == "failed_definite":
        return _row_actor(slot, operation_id, row, state="not_dispatched",
                          reason="The daemon definitively refused this reviewer's start."), NEVER_DISPATCHED
    return _row_actor(slot, operation_id, row, state="custody_lost", reason=(
        "A start was requested but no run was ever bound to it; its outcome stays unknown "
        "and it is never re-posted.")), UNAVAILABLE


def _attach_only_run(root: Any, request: Any, slot: Any, operation_id: str, run_id: str, invocation: dict) -> Any:
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
    from ouroboros.review_verdict_extraction import canonicalize_session_verdict
    from ouroboros.triad_review import default_output_contract, review_output_shape

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
        text = _full_session_text(gateway, run_id, detail) if state == "succeeded" else ""
    except Exception:
        log.debug("attach-only review observation of %s did not complete", run_id, exc_info=True)
        return None
    finally:
        try:
            gateway.close()
        except Exception:
            log.debug("attach-only gateway close failed", exc_info=True)
    usage = {"provider": "claudexor", "delegated_run_started": True, "delegated_run_id": run_id,
             "collection": "attach_only_observation", "cost": None, "run_state": state}
    observed = final_attempt_facts(detail, run_id)
    if isinstance(observed.get("effort_resolution"), dict):
        usage["effort_resolution"] = observed["effort_resolution"]
    # Requested identity comes only from this paid invocation, never live settings
    # or the summary's request echoes. Absent final-attempt facts stay unknown.
    routes = (invocation.get("request") or {}).get("harnesses") or []
    requested_route = str(routes[0]) if len(routes) == 1 else ""
    usage.update(resolved_model=observed.get("model", ""), observed_attempt=observed,
                 delegated_route=observed.get("harness_id", ""), requested_route=requested_route,
                 applied_profile=observed.get("profile_id", ""))
    usage["capability_delta"] = session_identity_deltas(slot, {
        "model": usage["resolved_model"], "route_id": requested_route,
        "effective_route_ids": [usage["delegated_route"]] if usage["delegated_route"] else []})
    binding = review_operation_binding(request, slot, operation_id)
    if state != "succeeded":
        return ReviewActorRecord(slot_id=slot.slot_id, model=slot.model, status="error", usage=usage,
                                 error=f"delegated review run {run_id} ended {state or 'without a state'}",
                                 operation_id=operation_id, operation_state="late_settled",
                                 recovery_binding=binding)
    shape = review_output_shape(request.surface)
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
    try:
        from ouroboros.observability import persist_call

        actor.response_ref = persist_call(
            root, task_id=request.task_id or "review", call_id=f"{operation_id}_collected",
            call_type=f"{request.surface}_review_collected",
            payload={"message": {"content": canonical, "session_transcript": text}, "usage": usage},
            manifest={"surface": request.surface, "slot_id": slot.slot_id, "collection": "attach_only_observation"})
    except Exception:
        log.warning("Collected review transcript could not be retained: %s", operation_id, exc_info=True)
    if method == "parse_unavailable":
        actor.parse_status = "parse_unavailable"
    return actor


def _collect_pending_row(root: Any, request: Any, slot: Any, row: dict, usage_ctx: Any,
                         controller: Any) -> Tuple[Any, str]:
    from ouroboros.review_custody import (
        _ACTIVE, _ACTIVE_LOCK, _attempt_key, _frozen_actor,
    )

    operation_id = str(row.get("operation_id") or "")
    key = _attempt_key(request, slot)
    with _ACTIVE_LOCK:
        entry = _ACTIVE.get(key)
        live = entry if entry is not None and entry.operation_id == operation_id else None
        answered = copy.deepcopy(live.actor) if live is not None and live.event.is_set() and live.actor is not None else None
    if answered is not None:
        return answered, COLLECTED
    if live is not None:
        return _frozen_actor(row, slot), DEFERRED  # its own worker settles it
    cached = (getattr(usage_ctx, "_review_settled_attempts", None) or {}).get(key)
    if cached is not None and str(getattr(cached, "operation_id", "") or "") == operation_id:
        return copy.deepcopy(cached), COLLECTED
    recovered = recover_review_producer(root, request, slot, row)
    if recovered is not None and recovered.operation_state != "in_flight":
        return recovered, UNAVAILABLE if recovered.operation_state == "custody_lost" else COLLECTED
    identity = controller if isinstance(controller, dict) else (row.get("usage") or {}).get("review_controller")
    # A row with no controller on record keeps the pre-operation reading (its
    # worker's custody is gone); a recorded controller that cannot be verified
    # is not presumed dead.
    state = controller_state(identity) if isinstance(identity, dict) and identity else "dead"
    if state == "alive":
        return _frozen_actor(row, slot), DEFERRED  # another live process owns this worker
    if str(getattr(slot.route, "value", slot.route) or "") == "agent_session":
        return _observe_delegated_review(root, request, slot, row, operation_id, controller_known=state != "unknown")
    if state == "unknown":
        return _frozen_actor(row, slot), DEFERRED
    return _row_actor(slot, operation_id, row, state="custody_lost", reason=(
        "The review worker ended without a retained producer outcome; its API outcome is unknown "
        "and is never re-sent.")), UNAVAILABLE


def collect_recorded_acceptance_run(run: Dict[str, Any], *, drive_root: Any, usage_ctx: Any = None,
                                    controller: Any = None) -> Any:
    """Collect one recorded acceptance operation purely, with its exact recorded inputs.

    No model call, send, re-post, daemon ensure, cancel or retirement happens here.
    Returns the substrate's own ``ReviewRunResult`` shape plus ``collection``:
    one typed fact per slot (``settled``/``collected``/``deferred``/
    ``unavailable``/``never_dispatched``) and the instant it was read.
    """
    from ouroboros.review_actor_aggregation import aggregate_review_actors
    from ouroboros.review_custody import _freeze_roster_rows, _frozen_actor
    from ouroboros.review_evidence import annotate_criteria_evidence_resolution
    from ouroboros.review_execution import ReviewRouteKind
    from ouroboros.review_projection import _review_actor_projection, _review_panel_id
    from ouroboros.review_records import HARDNESS_ADVISORY_VISIBLE, ReviewRequest, ReviewRunResult, ReviewSlot
    from ouroboros.review_verdict import _criteria_shape_valid

    request = ReviewRequest(**copy.deepcopy(run["request"]))
    if request.surface != "task_acceptance" or not request.retry_key:
        raise ValueError("recorded acceptance operation identity is missing")
    slots = [ReviewSlot(**{**row, "route": ReviewRouteKind(row["route"])})
             for row in copy.deepcopy(run.get("slot_roster") or [])]
    if not slots:
        raise ValueError("recorded acceptance roster is unavailable")
    frozen = _freeze_roster_rows(SimpleNamespace(), "task_acceptance", run.get("actors"))
    root = drive_root
    if usage_ctx is not None and getattr(usage_ctx, "drive_root", None):
        from ouroboros.delegate_custody import custody_root

        root = custody_root(usage_ctx)
    actors, facts = [], {}
    for slot in slots:
        row = frozen.get(slot.slot_id)
        if row is None:
            actor, fact = _row_actor(slot, "", None, state="custody_lost",
                                     reason="The recorded roster holds no row for this slot."), UNAVAILABLE
        elif str(row.get("operation_state") or "") not in _COLLECTABLE:
            actor, fact = _frozen_actor(row, slot), SETTLED
        else:
            actor, fact = _collect_pending_row(root, request, slot, row, usage_ctx, controller)
            owner = (row.get("usage") or {}).get("review_controller")
            if isinstance(owner, dict) and fact != DEFERRED:
                actor.usage = {**dict(actor.usage or {}), "review_controller": dict(owner)}
        actors.append(actor)
        facts[slot.slot_id] = fact
    aggregate = aggregate_review_actors(
        request=request, slots=slots, actors=actors, slots_by_id={slot.slot_id: slot for slot in slots},
        actor_projection=_review_actor_projection, criteria_shape_valid=_criteria_shape_valid,
        advisory_hardness=HARDNESS_ADVISORY_VISIBLE)
    result = ReviewRunResult(request=asdict(request), actors=[asdict(actor) for actor in actors], **aggregate,
                             panel_id=_review_panel_id(request, actors),
                             slot_roster=copy.deepcopy(run.get("slot_roster") or []))
    annotate_criteria_evidence_resolution(result.actors, request.evidence)
    result.collection = {"facts": facts, "collected_at": utc_now_iso()}
    return result


# --- confirmed-death recovery through the existing maintenance pass -------------

def run_from_checkpoint(root: Any, task_id: str, entry: Dict[str, Any], owner_id: str,
                        result: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """The checkpointed operation as a pending HOST run, when no publication exists.

    Only an operation the root wallet actually claimed is the host's paid panel:
    an advisory child/off-mode review uses the same seam and is never republished
    under host authority.
    """
    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros.task_results import load_task_acceptance_review_state, load_task_result, resolve_task_lineage

    retry_key = str(entry.get("retry_key") or "")
    source = json.loads(read_actor_source_bytes(root, task_id, entry.get("source_ref")))
    request = source.get("request") if isinstance(source, dict) else None
    if (not isinstance(request, dict) or source.get("schema_version") != 1
            or source.get("owner_id") != owner_id or source.get("task_id") != task_id
            or source.get("surface") != "task_acceptance" or request.get("surface") != "task_acceptance"
            or request.get("task_id") != task_id or request.get("retry_key") != retry_key
            or any(source.get(key) != entry.get(key) for key in (
                "retry_key", "task_attempt", "controller", "operations"))
            or source.get("task_attempt") != request.get("task_attempt")
            or not source.get("slot_roster")):
        return None
    if "paid_authority" in source:
        authority = source["paid_authority"]
        if not isinstance(authority, dict) or authority.get("schema_version") != 1 or authority.get("authority") != "host_root":
            return None
        lineage = authority.get("lineage") or {}
        validated = resolve_task_lineage(task_id, metadata=lineage)
        if lineage != validated or not validated["is_root_task"]:
            return None
        binding = authority.get("binding") or {}
    else:
        # Landed checkpoints predate paid_authority. Only positive canonical
        # SAME-physical-root lineage and its actual claim can qualify them.
        # Never follow a mutable retry alias or rewrite the immutable source.
        canonical = load_task_result(root, task_id, strict=True) or {}
        metadata = canonical.get("metadata") or {}
        keys = ("root_task_id", "parent_task_id", "delegation_role", "original_task_id", "timeout_retry_from")
        if any(key in canonical and key in metadata and canonical[key] != metadata[key] for key in keys):
            return None
        validated = resolve_task_lineage(task_id, metadata=metadata, **{key: canonical.get(key) for key in keys})
        pointer = (canonical.get(OPERATIONS_FIELD) or {}).get(owner_id) or {}
        if (canonical.get("task_id") != task_id or validated["root_task_id"] != task_id
                or validated["delegation_role"] != "root" or not validated["is_root_task"]
                or validated["original_task_id"] or validated["timeout_retry_from"]
                or not (canonical.get("root_task_id") or metadata.get("root_task_id"))
                or any(pointer.get(key) != entry.get(key) for key in (
                    "source_ref", "retry_key", "task_attempt", "controller", "operations"))):
            return None
        binding = None
    claims = load_task_acceptance_review_state(
        root, validated["root_task_id"], require_root_result=True)["claims_by_binding"]
    if binding is None:
        matching = [claim for claim in claims.values() if claim.get("paid_identity")
                    and retry_key == f"task_acceptance:{claim['paid_identity']}"]
        if len(matching) != 1:
            return None
        binding = matching[0]
    claim = claims.get(binding.get("binding_hash"))
    if (not claim or claim.get("claimed_by_task_id") != task_id
            or not binding.get("paid_identity") or retry_key != f"task_acceptance:{binding['paid_identity']}"
            or any(claim.get(key) != binding.get(key) for key in (
                "binding_hash", "candidate_hash", "evidence_revision", "fence_hash", "paid_identity"))
            or binding.get("candidate_hash") != hashlib.sha256(str(request.get("subject") or "").encode()).hexdigest()):
        return None
    models = {str(row.get("slot_id")): str(row.get("model") or "") for row in source["slot_roster"]}
    if (len(models) != len(source["slot_roster"]) or not all(models)
            or not source.get("operations") or not set(source["operations"]).issubset(models)
            or not all(isinstance(value, str) and value for value in source["operations"].values())):
        return None
    actors = [{"slot_id": slot_id, "model": models.get(slot_id, ""), "status": "error",
               "error": "Checkpointed before its controller ended; not yet collected.",
               "operation_id": operation_id, "operation_state": "pending_dispatch", "late_result_pending": True,
               "usage": {"review_controller": {**dict(source.get("controller") or {}), "owner_id": owner_id}}}
              for slot_id, operation_id in (source.get("operations") or {}).items()]
    return {"authority": "host_root", "request": request, "slot_roster": source["slot_roster"],
            "actors": actors, "aggregate_signal": "DEGRADED", "panel_id": f"panel_{binding['binding_hash'][:16]}",
            "candidate_hash": str(claim.get("candidate_hash") or "")
            or hashlib.sha256(str(request.get("subject") or "").encode("utf-8")).hexdigest(),
            "binding_hash": str(claim.get("binding_hash") or ""), "paid_identity": str(claim["paid_identity"]),
            "accounting_root_task_id": validated["root_task_id"],
            "task_attempt": source.get("task_attempt"), "operation_checkpoint_ref": entry.get("source_ref")}


def collect_orphaned_operations_softly(drive_root: Any, *, stop: Optional[Callable[[], bool]] = None) -> None:
    """The maintenance hook: one fail-soft pass that never stops the sweep around it."""
    try:
        report = recover_orphaned_acceptance_operations(drive_root, stop=stop)
        if report.get("settled") or report.get("errors"):
            log.info("Orphaned acceptance operations: %s", report)
    except Exception:
        log.warning("Orphaned acceptance operation recovery failed", exc_info=True)


def recover_orphaned_acceptance_operations(drive_root: Any, *, stop: Optional[Callable[[], bool]] = None) -> Dict[str, Any]:
    """Collect paid acceptance operations nobody live will settle, for terminal tasks.

    Runs inside the existing off-loop maintenance pass. An open pointer whose
    controller is proven dead (or is this process without a registration), or
    an ``unpublished`` one, is collected purely and published as a late
    settlement; only a publication read back marks it ``collected``. A
    controller that cannot be verified, a source that cannot be read or a panel
    still in flight is reported and retried — never marked unavailable.
    """
    from ouroboros.acceptance_settlement import acceptance_actor_ended, settle_acceptance_operation
    from ouroboros.task_results import load_task_result, task_results_dir

    root = pathlib.Path(drive_root)
    report: Dict[str, Any] = {"settled": [], "pending": [], "deferred": [], "errors": []}
    try:
        paths = sorted(task_results_dir(root, create=False).glob("*.json"))
    except OSError:
        return report
    marker = f'"{OPERATIONS_FIELD}"'
    for path in paths:
        if stop is not None and stop():
            break
        try:
            text = path.read_text(encoding="utf-8")
            if marker not in text and '"review_projection"' not in text:
                continue
            row = load_task_result(root, path.stem) or {}
        except (OSError, ValueError):
            continue
        ctx = SimpleNamespace(task_id=path.stem, task_attempt=None, drive_root=root,
                              budget_drive_root=root, task_metadata={}, event_queue=None)
        if not acceptance_actor_ended(ctx, path.stem, row):
            continue
        for owner_id, entry in list((row.get(OPERATIONS_FIELD) or {}).items()):
            state = entry.get("state") if isinstance(entry, dict) else None
            if state not in _OPEN_STATES | {OPERATION_UNPUBLISHED} or entry.get("surface") != "task_acceptance":
                continue
            with _LOCK:
                registered = owner_id in _LIVE
            controller = ("dead" if not registered and entry.get("controller") == controller_identity()
                          else controller_state(entry.get("controller")))
            if registered or (state != OPERATION_UNPUBLISHED and controller in {"alive", "unknown"}):
                if controller == "unknown" and not registered:
                    report["deferred"].append({"task_id": path.stem, "owner_id": owner_id,
                                               "reason": "controller_unverifiable"})
                continue
            if state == OPERATION_PREPARING:
                from ouroboros.owner_pause import fence_closed, read_fence
                pause = entry.get('preparation_pause') or {}
                fence = read_fence(root, pause.get('root_task_id', '')) if pause else {}
                if fence_closed(fence) and fence.get('fence_id') == pause.get('fence_id'):
                    from supervisor.owner_pause_control import retain_late_phase_latch
                    retain_late_phase_latch(root, pause['root_task_id'])
                    report['deferred'].append({'task_id': path.stem, 'owner_id': owner_id,
                                               'reason': 'owner_paused_preparation'})
                    continue  # Unsent remainder, not a live controller or boot dispatch authority.
                _mark_operation(root, path.stem, owner_id, 'preparation_unknown', from_states={OPERATION_PREPARING})
                report['pending'].append({'task_id': path.stem, 'owner_id': owner_id,
                                          'reason': 'preparation_controller_ended_before_request'})
                continue  # an intent is never a ready request or a paid claim
            ctx = SimpleNamespace(task_id=path.stem, task_attempt=entry.get("task_attempt"), drive_root=root,
                                  budget_drive_root=root, task_metadata={}, event_queue=None)
            try:
                status = settle_acceptance_operation(
                    ctx, retry_key=str(entry.get("retry_key") or ""), task_id=path.stem, result=row,
                    checkpoint={**entry, "owner_id": owner_id}, controller=entry.get("controller"))
            except Exception as exc:
                report["errors"].append({"task_id": path.stem, "owner_id": owner_id, "error": f"{type(exc).__name__}: {exc}"})
                log.warning("Orphaned acceptance operation %s could not be collected", owner_id, exc_info=True)
                continue
            done = status in {"announced", "published", "settled"}
            if done:
                _mark_operation(root, path.stem, owner_id, OPERATION_COLLECTED,
                                from_states=_OPEN_STATES | {OPERATION_UNPUBLISHED})
            report["settled" if done else "pending"].append({"task_id": path.stem, "owner_id": owner_id,
                                                             "status": status})
        _recover_legacy_acceptance_panels(root, path.stem, row, report, stop=stop)
    return report


def _recover_legacy_acceptance_panels(root: pathlib.Path, task_id: str, row: dict,
                                      report: dict, *, stop: Optional[Callable[[], bool]]) -> None:
    """Published host panels predate operation pointers; their own source owns the free catch-up.

    Reuse the same collector, projection and outbox, without minting a controller
    or inventing a paid request. A persisted late fact can owe only its notice.
    """
    from ouroboros.acceptance_settlement import enqueue_late_acceptance_settlement, settle_acceptance_operation
    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros.review_projection import _actor_pending
    from ouroboros.task_results import load_task_result
    from supervisor.terminal_delivery import already_delivered, pending_deliveries

    pointed = {entry.get("retry_key") for entry in (row.get(OPERATIONS_FIELD) or {}).values()
               if isinstance(entry, dict)}
    owed = None
    for panel in (row.get("review_projection") or {}).get("panels") or []:
        if stop is not None and stop():
            return
        if not isinstance(panel, dict) or panel.get("surface") != "task_acceptance" or panel.get("authority") != "host_root":
            continue
        late = panel.get("late_settlement")
        pending = any(_actor_pending(actor) for actor in panel.get("actors") or [] if isinstance(actor, dict))
        if not pending and not isinstance(late, dict):
            continue
        fact = {"task_id": task_id, "panel_id": panel.get("panel_id")}
        ctx = SimpleNamespace(task_id=task_id, task_attempt=panel.get("task_attempt"), drive_root=root,
                              budget_drive_root=root, task_metadata={}, event_queue=None)
        try:
            if isinstance(late, dict) and not pending:
                retry_key = str((late.get("reviewed_subject") or {}).get("retry_key") or "")
                if not retry_key or retry_key in pointed:
                    continue
                delivery_id = f"acceptance-late:{retry_key}"
                receipt = ((row.get("review_projection") or {}).get("late_notice_receipts") or {}).get(delivery_id)
                if not isinstance(receipt, dict) or receipt.get("panel_id") != panel.get("panel_id"):
                    continue  # old already-settled panels do not acquire a new notification duty
                if (receipt.get("custody") == "terminal_outbox"
                        and receipt.get("settled_at") == late.get("settled_at")):
                    continue
                if owed is None:
                    owed = {event.get("delivery_id") for event in pending_deliveries(root)}
                if delivery_id in owed or already_delivered(root, delivery_id):
                    _remember_legacy_notice(root, task_id, delivery_id, panel)
                    continue
                status = enqueue_late_acceptance_settlement(ctx, task_id, retry_key, row, panel)
            else:
                run = json.loads(read_actor_source_bytes(root, task_id, panel.get("applied_source_ref")))
                if not isinstance(run, dict):
                    raise ValueError("published acceptance source is not an object")
                request = run.get("request") or {}
                retry_key = str(request.get("retry_key") or "")
                if (run.get("authority") != "host_root" or request.get("surface") != "task_acceptance"
                        or request.get("task_id") != task_id or not retry_key or retry_key in pointed):
                    continue
                if not _remember_legacy_notice(root, task_id, f"acceptance-late:{retry_key}", panel,
                                                custody="publication"):
                    continue
                status = settle_acceptance_operation(ctx, retry_key=retry_key, task_id=task_id, result=row)
            if status in {"announced", "published"}:
                current = load_task_result(root, task_id, strict=True) or {}
                published = next((item for item in (current.get("review_projection") or {}).get("panels") or []
                                  if item.get("panel_id") == panel.get("panel_id")), None)
                if published is not None:
                    _remember_legacy_notice(root, task_id, f"acceptance-late:{retry_key}", published)
            report["settled" if status in {"announced", "published", "settled"} else "pending"].append(
                {**fact, "status": status})
        except (OSError, ValueError, TypeError, KeyError) as exc:
            report["errors"].append({**fact, "error": f"{type(exc).__name__}: {exc}"})


def _remember_legacy_notice(root: pathlib.Path, task_id: str, delivery_id: str, panel: dict,
                            *, custody: str = "terminal_outbox") -> bool:
    """Keep this legacy publication's notice duty until the existing outbox takes custody.

    This receipt is not delivery proof. It stays with the exact published panel,
    while the outbox alone owns retries, delivery and exhaustion disclosures. A
    durable handoff keeps bounded outbox eviction from re-owing old criticism.
    """
    from ouroboros.task_results import write_task_result

    accepted = []
    def project(current, _fields):
        projection = current.get("review_projection") or {}
        stored = next((item for item in projection.get("panels") or []
                       if isinstance(item, dict) and item.get("panel_id") == panel.get("panel_id")
                       and item.get("publication_revision") == panel.get("publication_revision")), None)
        if stored is None or stored.get("late_settlement") != panel.get("late_settlement"):
            return None  # a concurrent producer owns the new panel; next pass re-reads it
        receipt = {"panel_id": stored["panel_id"], "publication_revision": stored.get("publication_revision"),
                   "settled_at": (stored.get("late_settlement") or {}).get("settled_at"), "custody": custody}
        receipts = projection.get("late_notice_receipts") or {}
        accepted.append(True)
        if receipts.get(delivery_id) == receipt:
            return None
        return {"status": current["status"], "review_projection": {
            **projection, "late_notice_receipts": {**receipts, delivery_id: receipt}}}

    write_task_result(root, task_id, "running", strict_existing_dict=True, _field_projector=project)
    return bool(accepted)
