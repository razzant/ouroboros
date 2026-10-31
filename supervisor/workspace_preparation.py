"""The supervisor generation's off-loop owner of admission persistence.

Every ``schedule_subagent`` request is claimed on the event loop in memory only
(the existing admission reservation map under Q, nothing read or written in
storage) and handed to one of two FIFO lanes of the supervisor generation,
which run the existing schedule owner with the claim
(``events_schedule_task._handle_schedule_task(..., preparation=claim)``).
The copy lane keeps every write surface that may copy or create a root (an
isolated current-tree copy, a genesis project) in acceptance order. The
admission lane never copies: it validates existing folders, admits read-only
children whole, and runs the final admission step of every claim (the durable
receipt, the enqueue under Q, the notice), so a readonly child or a copied
child's admission never waits behind another task's copy (a child admitted on a
folder may write it while a copy of that folder still waits: the copy then
starts from those writes, and one landing during the copy is refused by its
whole-tree diff, both as in the other order). The same lane persists worker
launch marks (``worker_assignment``) and warm owner-wait Resumes
(``worker_owner_wait``): one owner for everything that must land in storage
before a queue put, outside Q. Command and event draining wait on none of it,
and the bus is never re-put: a storage acquisition gap is retried here, at the
head of its own lane (``PreparationGeneration.wait_store``: the existing 0.25 s
slice, the caller's controls re-read between slices). A host that never opened a
generation (a direct caller of the handler) keeps the historical inline path.

The marker on the row carries the token, the owning process (pid and birth) and
generation, the phase and, once the copy returned, the owned resource handle.
Phases are positive facts written before their effect: ``queued`` (claimed,
nothing started), ``copying`` (physical work may have begun), ``ready`` (the copy
returned), ``admitting`` (committed before the receipt; the SCHEDULED receipt is
written under it, outside Q, before the queue put); the put retires it to
``admitted`` and the never-admitted refusal to ``refused``. A readonly child
copies nothing and goes ``queued`` → ``admitting``. Stop, Pause, a closed
generation or lost ownership is read before the copy starts and again after it
returns; the final enqueue revalidates memory only under Q: the token, root
fences, slot accounting, Project binding.

``recover_interrupted_preparations`` (startup after queue restore, and the
periodic reconcile pass) settles ``queued``/``copying``/``ready`` as never admitted
only once the recorded owner positively ended; ``admitting`` on a REQUESTED row
(no receipt) or unreadable authority stays unresolved on the existing
``unknowns`` carrier, never inferred unsent from an absent queue row. A SCHEDULED
receipt still ``admitting`` whose owner ended and which the restored queue does
not hold was never put (every put is preceded by a published locator snapshot
naming the row), so it is refused never admitted once the saved queue is in this
process's custody; ``admitted`` and absent is a gap on ``unknowns``. A REQUESTED
row without any marker (its event died with its bus) is refused never admitted
once the stamped producer and bus owner both provably ended, or, for a row that
predates the stamp, once its parent is terminal; otherwise it stays held
(``requested_by``: ``ouroboros/tools/control_scheduling.py``). Disposal is exact:
only the recorded handle of this token, only after its refusal receipt and the
copy's return; anything else stays registered residue for the existing worktree
retention. Recovery never re-copies or dispatches.
"""

from __future__ import annotations

import collections
import copy
import functools
import logging
import os
import pathlib
import subprocess
import threading
import uuid
from types import SimpleNamespace
from typing import Any, Callable, Dict, Optional

from ouroboros.task_results import STATUS_REQUESTED, STATUS_SCHEDULED, load_task_result, write_task_result
from ouroboros.store import StoreBusy
from ouroboros.tool_capabilities import ACTING_SUBAGENT_MODE
from ouroboros.utils import append_jsonl, utc_now_iso
from supervisor.task_admission import WORKSPACE_PREPARATION_FIELD as MARKER

log = logging.getLogger(__name__)

QUEUED, COPYING, READY, ADMITTING, ADMITTED, REFUSED = (
    "queued", "copying", "ready", "admitting", "admitted", "refused")
UNSENT_PHASES = frozenset({QUEUED, COPYING, READY})
MARKER_UNWRITABLE = "preparation_marker_unwritable"
INTERRUPTED_BY_RESTART = "preparation_interrupted_by_restart"
INTERRUPTED_BY_GENERATION_EXIT = "preparation_interrupted_by_generation_exit"
EXECUTOR_UNAVAILABLE = "preparation_executor_unavailable"
ADMISSION_UNCERTAIN = "workspace_preparation_admission_uncertain"
# A request whose event died with the bus that carried it, or whose parent ended first (no marker).
REQUEST_LOST = "request_lost_with_server"
REQUEST_ORPHANED = "request_orphaned_by_parent"
_RESOURCE_KEYS = ("surface", "path", "branch", "created_at", "base_sha")
_DETAILS = {
    MARKER_UNWRITABLE: ("Subagent not scheduled: its workspace preparation could not record its progress "
                        "durably; it was never admitted and is not retried automatically."),
    INTERRUPTED_BY_RESTART: ("Subagent not scheduled: the server stopped while its workspace was being "
                             "prepared; it was never admitted and is not retried automatically."),
    INTERRUPTED_BY_GENERATION_EXIT: ("Subagent not scheduled: the supervisor generation that owned its "
                                     "workspace preparation ended first; it was never admitted and is not "
                                     "retried automatically."),
    EXECUTOR_UNAVAILABLE: ("Subagent not scheduled: the workspace preparation worker could not start; "
                           "it was never admitted."),
    REQUEST_LOST: ("Subagent not scheduled: the server that received its request ended before admitting it; "
                   "it was never admitted and is not retried automatically."),
    REQUEST_ORPHANED: ("Subagent not scheduled: its request was never admitted and its parent task has ended; "
                       "it is not retried automatically."),
}
_AUTHORITY_UNKNOWN = ("Subagent not scheduled: its durable result is unreadable, so its preparation was not "
                      "started and the existing authority was preserved.")

_STATE_LOCK = threading.Lock()
_CURRENT: list = [None]  # the open generation's owner; None keeps the historical inline path
_LIVE: Dict[str, "PreparationClaim"] = {}  # token -> claim, from its claim until its job ends
_OWNER: Dict[str, Any] = {}


def store_contention(error: BaseException) -> bool:
    """A temporary acquisition gap, not unreadable authority or an unknown commit."""
    return (isinstance(error, StoreBusy) and error.reason == "contention"
            and getattr(error, "rollback_confirmed", None) is not False)


class PreparationStopped(Exception):
    def __init__(self, refusal: Dict[str, Any]):
        self.refusal = refusal
        super().__init__(refusal["detail"])


def _process_owner() -> Dict[str, Any]:
    """This process's recorded identity, read once per process (``ps`` on macOS)."""
    if _OWNER.get("pid") != os.getpid():
        from ouroboros.platform_layer import process_start_time

        _OWNER.clear()
        _OWNER.update(pid=os.getpid(), birth=process_start_time(os.getpid()))
    return dict(_OWNER)


COPY_LANE, ADMISSION_LANE = "workspace-preparation", "workspace-preparation-validation"
VALIDATION_LANE = ADMISSION_LANE  # the admission lane's historical name (existing folders validate there)
# The resolver of these surfaces only validates an existing folder and reads its
# Git HEAD (``_resolve_subagent_constraint``): no copy, no shared worktree lock.
_VALIDATION_SURFACES = frozenset({"external_workspace"})


class PreparationGeneration:
    """One supervisor generation's admission owner: two FIFO lanes, one daemon thread each at a time.

    The copy lane keeps every surface that may copy or create a write root in
    acceptance order; the admission lane runs only steps that copy nothing (an
    existing folder's validation, a readonly child, every claim's final
    admission step, launch marks, warm Resumes), so none of them waits behind
    another task's copy, and their storage writes are serialized in one place. A
    claim's later cleanup joins its own lane, after its job. ``submit`` never
    waits: it appends and starts the lane's thread when none runs; the thread
    drains in order and exits when the lane is empty. Closing never joins: a
    running copy keeps its custody until it returns.
    """

    def __init__(self, stop: Optional[Callable[[], bool]] = None):
        self.token = uuid.uuid4().hex
        self.owner = _process_owner()
        self._stop = stop
        self._closed = threading.Event()
        self._pauses = 0  # process-ending stops under way, each its own (``pause_admission``)
        self._lock = threading.Lock()
        self._idle = threading.Condition(self._lock)
        self._jobs = {lane: collections.deque() for lane in (COPY_LANE, VALIDATION_LANE)}
        self._threads: Dict[str, Optional[threading.Thread]] = dict.fromkeys(self._jobs)

    def closed(self) -> bool:
        """Closed explicitly, or its server generation / process stop or restart was requested."""
        return self._closed.is_set() or bool(self._stop is not None and self._stop())

    def launches_closed(self) -> bool:
        """Closed, or paused by a stop under way: no worker launch or warm Resume starts a durable step
        (a child's admission reads ``closed`` alone and is queued as ever)."""
        return self._pauses > 0 or self.closed()

    def pause(self, resume: bool = False) -> None:
        """Count one stop's pause (or end one): a door that finished never lifts another's."""
        with self._lock:
            self._pauses = max(0, self._pauses - 1) if resume else self._pauses + 1

    def close(self) -> None:
        self._closed.set()

    def lane_idle(self, lane: str, timeout: Optional[float] = None) -> bool:
        """Wait until ``lane`` holds no job and runs none (a stop's bounded wait; never the lane itself)."""
        with self._idle:
            return self._idle.wait_for(lambda: self._threads[lane] is None and not self._jobs[lane], timeout)

    def submit(self, job: Callable[[], None], lane: str = COPY_LANE) -> bool:
        """Queue ``job`` behind earlier ones of its lane; False when no thread could start for it."""
        with self._lock:
            self._jobs[lane].append(job)
            if self._threads[lane] is not None:
                return True
            thread = threading.Thread(target=self._drain, args=(lane,), name=lane, daemon=True)
            self._threads[lane] = thread
            try:
                thread.start()
            except Exception:
                log.warning("Workspace preparation thread could not start", exc_info=True)
                self._threads[lane] = None
                self._jobs[lane].clear()  # no thread ran, so the lane held only this job
                return False
            return True

    def on_executor(self) -> bool:
        return threading.current_thread() in self._threads.values()

    def current_lane(self) -> str:
        """The lane whose thread runs the caller, else an empty string."""
        current = threading.current_thread()
        return next((lane for lane, thread in self._threads.items() if thread is current), "")

    def wait_store(self, operation: Callable[[], Any], *, controls: Optional[Callable[[], Optional[Dict[str, Any]]]] = None,
                   retryable: Optional[Callable[[], bool]] = None,
                   busy: Optional[Callable[[BaseException], bool]] = None) -> Any:
        """Keep a lane job's place across a known-unsent acquisition gap, outside every Q scope.

        Callers wrap one read/CAS/interlock, never a copy or arbitrary handler.
        ``busy`` names the exceptions that are such a gap (default: confirmed
        store contention; reads pass any ``StoreBusy``, launches the schedule
        owner's ``contention``). ``controls`` is re-read between slices and its
        refusal ends the job typed (``PreparationStopped``); ``retryable`` closes
        the gate before any possible queue handoff. Off a lane thread nothing
        waits: the gap propagates to the caller's own owner.
        """
        from ouroboros._usage_wait import ACQUISITION_SLICE_SEC
        from ouroboros.store import acquisition_budget
        from supervisor import queue

        gap = busy or store_contention
        while True:
            try:
                with acquisition_budget(ACQUISITION_SLICE_SEC):  # one slice per attempt, controls between
                    return operation()
            except Exception as exc:
                if not gap(exc) or not self.on_executor() or retryable is not None and not retryable():
                    raise
                if queue._queue_lock._is_owned():
                    raise RuntimeError("Preparation acquisition wait still owns Q") from exc
                if controls is not None:
                    try:
                        refusal = controls()
                    except StoreBusy as control_gap:
                        if not store_contention(control_gap):
                            raise
                    else:
                        if refusal is not None:
                            raise PreparationStopped(refusal) from exc
                    self._closed.wait(ACQUISITION_SLICE_SEC)
                else:
                    # Closing forbids new execution, not finishing this same
                    # refusal receipt. The closed event would spin here.
                    threading.Event().wait(ACQUISITION_SLICE_SEC)

    def wait_idle(self, timeout: Optional[float] = None) -> bool:
        """Wait until nothing is queued or running (tests, diagnostics; never the event loop)."""
        with self._idle:
            return self._idle.wait_for(
                lambda: not any(self._threads.values()) and not any(self._jobs.values()), timeout)

    def _drain(self, lane: str) -> None:
        jobs = self._jobs[lane]
        while True:
            with self._lock:
                if not jobs:
                    self._threads[lane] = None
                    self._idle.notify_all()
                    return
                job = jobs.popleft()
            try:
                job()
            except Exception:
                log.warning("Workspace preparation job raised", exc_info=True)


def start_generation(*, stop: Optional[Callable[[], bool]] = None) -> PreparationGeneration:
    """Open the preparation owner of a starting supervisor generation (an older one closes).

    ``stop`` answers True once that generation must not start or admit anything
    more: its own stop token, an explicit server stop, or a requested restart.
    """
    generation = PreparationGeneration(stop)
    with _STATE_LOCK:
        previous, _CURRENT[0] = _CURRENT[0], generation
    if previous is not None:
        previous.close()
    return generation


def close_generation(generation: Optional[PreparationGeneration] = None) -> None:
    """Close without waiting: queued launches and Resumes refuse before any durable step; a running
    copy keeps its custody."""
    target = generation if generation is not None else _CURRENT[0]
    if target is not None:
        target.close()


def pause_admission(timeout: Optional[float] = None) -> bool:
    """A process-ending stop's first step (owner Restart, planned restart, Update, Quit), before it reads
    the queue or writes its own restrictions: the open generation starts no new launch or warm Resume
    (``launches_closed``), and the stop waits, at most ``STOP_ADMISSION_DRAIN_SEC``, for the admission
    lane to empty, so a launch or Resume already writing its marks hands over to the pool it reserved
    while that pool still runs (``worker_assignment``, ``worker_owner_wait``). True when the lane emptied
    in time. ``resume_admission`` reopens it."""
    from ouroboros.runtime_limits import STOP_ADMISSION_DRAIN_SEC
    from supervisor import queue

    generation = _CURRENT[0]
    if generation is None:
        return True
    generation.pause()
    if generation.on_executor() or queue._queue_lock._is_owned():
        log.error("A stop paused admission from the admission lane or under Q; it cannot wait for the lane")
        return False
    if generation.lane_idle(ADMISSION_LANE, STOP_ADMISSION_DRAIN_SEC if timeout is None else timeout):
        return True
    log.warning("A stop proceeds while the admission lane still writes; a launch it commits after the pool "
                "stops restores as possibly dispatched")
    return False


def resume_admission() -> None:
    """The stop that paused admission did not end this process: its pause ends (another stop's stays)."""
    generation = _CURRENT[0]
    if generation is not None:
        generation.pause(resume=True)


def current_generation() -> Optional[PreparationGeneration]:
    """The open generation whose admission lane persists launch marks and warm Resumes; None keeps
    a caller's historical inline path (a host that never opened one)."""
    return _CURRENT[0]


def _requests_preparation(evt: Dict[str, Any]) -> bool:
    requested = evt.get("task_constraint")
    return isinstance(requested, dict) and str(requested.get("mode") or "") == ACTING_SUBAGENT_MODE


def _no_store_wait():
    """The short event-entry refusal never waits under Q."""
    from ouroboros.store import acquisition_budget
    return acquisition_budget(0)


def accept(evt: Dict[str, Any], ctx: Any, route: Dict[str, Any], *, admission: Optional[Callable] = None) -> bool:
    """Claim a subagent request for this generation's lanes; False keeps the inline path.

    Event-loop work only, memory only: the reservation under Q (the row's own
    freshness is read on the lane) and a nonblocking submit. A duplicate
    coalesces with its live claim; an id another owner holds is ignored. The
    event is consumed here and never re-put: the lane retries its own storage
    gaps at the head of its queue. An already-composed host child supplies its
    admission callback; on this lane it runs inline under the same claim.
    """
    generation = current_generation()
    if (generation is None or str(route.get("delegation_role") or "") != "subagent"
            or not str(evt.get("task_id") or "").strip()):
        return False
    with _STATE_LOCK:
        if any(value.task_id == str(evt["task_id"]) for value in _LIVE.values()):
            return True
        claim = PreparationClaim.live(generation, evt, ctx, route)
        _LIVE[claim.token] = claim
    from supervisor.task_admission import claim_requested_child

    submitted = False
    try:
        if not claim_requested_child(claim.ctx, claim.task_id, claim.token):
            log.info("Ignoring replayed schedule event for task %s", claim.task_id)
            return True
        if claim.generation.closed():
            _refuse_at_entry(claim, INTERRUPTED_BY_GENERATION_EXIT)
            return True
        if admission is not None and generation.current_lane() == claim.lane:
            submitted = claim.submitted = True
            _run(claim, admission=admission)
        else:
            submitted = claim.submitted = claim.generation.submit(functools.partial(_run, claim, admission=admission), claim.lane)
        if not submitted:
            _refuse_at_entry(claim, EXECUTOR_UNAVAILABLE)
        return True
    finally:
        if not submitted:
            claim.release()
            _finish(claim)


def _refuse_at_entry(claim: "PreparationClaim", reason: str) -> None:
    """A request no lane can take: the typed refusal, zero-wait on the event loop; a contended write
    leaves the REQUESTED row to the never-delivered recovery rule (its stamp names this server)."""
    try:
        with _no_store_wait():
            claim.refuse(_DETAILS[reason], reason)
    except Exception:
        log.warning("Entry refusal of %s (%s) was not recorded; recovery settles the request",
                    claim.task_id, reason, exc_info=True)


def _run(claim: "PreparationClaim", *, admission: Optional[Callable] = None) -> None:
    """The lane job: the ``queued`` marker, then the existing schedule owner under the claim."""
    from supervisor.events_schedule_task import _handle_schedule_task

    def step() -> None:
        landed = claim.store_gate(functools.partial(
            _transition, claim.ctx.DRIVE_ROOT, claim.task_id, claim.token, claim.queued_marker, expected=None))
        if landed is not True:
            claim.unrecorded(landed)
            return
        if admission is None:
            _handle_schedule_task(claim.event, claim.ctx, preparation=claim)
        else:
            admission(claim)

    _guarded(claim, step)


def _guarded(claim: "PreparationClaim", step: Callable[[], None], owner: Any = None) -> None:
    """One lane step under the claim: a typed stop refuses, a failure before the admission boundary
    refuses, and the claim ends unless the step handed its continuation to the admission lane.
    The end belongs to one invocation: a hand-over names the next one before its job can start."""
    owner = claim.ender = owner or object()
    try:
        step()
    except PreparationStopped as exc:
        # Raised only by a proven-unsent store gate, never around enqueue or copy.
        try:
            claim.refuse(**exc.refusal)
        except Exception:
            log.warning("Stopped preparation %s retains its recovery marker", claim.task_id, exc_info=True)
    except Exception as exc:
        log.warning("Workspace preparation of %s failed", claim.task_id, exc_info=True)
        if not claim.crossed:  # past the boundary the admission path owns the outcome
            try:
                claim.refuse(f"Subagent not scheduled: its workspace preparation failed: "
                             f"{type(exc).__name__}: {exc}", "admission_preparation_failed")
            except Exception:
                log.warning("Preparation refusal of %s failed", claim.task_id, exc_info=True)
    finally:
        if claim.ender is owner:
            claim.release()
            _finish(claim)


def _finish(claim: "PreparationClaim") -> None:
    with _STATE_LOCK:
        if _LIVE.get(claim.token) is claim:
            _LIVE.pop(claim.token, None)


class PreparationClaim:
    """One requested child's preparation, owned by one token."""

    def __init__(self, *, task_id: str, ctx: Any, route: Dict[str, Any], token: str,
                 generation: Optional[PreparationGeneration], generation_token: str,
                 owner: Dict[str, Any], event: Optional[Dict[str, Any]] = None, surface: str = "",
                 resource: Optional[Dict[str, Any]] = None, claimed_at: str = "",
                 unidentified_residue: bool = False, acting: bool = True):
        self.task_id, self.ctx, self.route, self.token = task_id, ctx, route, token
        self.generation, self.generation_token, self.owner = generation, generation_token, dict(owner)
        self.event, self.surface, self.acting = event or {}, surface, acting
        self.resource = dict(resource or {})
        self.claimed_at = claimed_at or utc_now_iso()
        self.unidentified_residue = unidentified_residue
        self.crossed = False
        self.submitted = False
        self.ender: Any = None  # the lane invocation that ends this claim
        self.queued_marker = self.marker(QUEUED)
        self.refusal_receipt = uuid.uuid4().hex

    @property
    def lane(self) -> str:
        """A copying surface takes the copy lane; a readonly child or an existing folder the admission lane."""
        return COPY_LANE if self.acting and self.surface.strip().lower() not in _VALIDATION_SURFACES else ADMISSION_LANE

    def store_gate(self, operation: Callable[[], Any], *, controls: bool = True,
                   retryable: Optional[Callable[[], bool]] = None,
                   busy: Optional[Callable[[BaseException], bool]] = None) -> Any:
        """This claim's place on its lane across a known-unsent acquisition (``wait_store``); its
        controls are the root latches and the generation. Without a generation nothing waits."""
        if self.generation is None:
            return operation()
        return self.generation.wait_store(operation, controls=self._control_refusal if controls else None,
                                          retryable=retryable, busy=busy)

    def admit(self, step: Callable[[], None]) -> None:
        """Run the admission step on the admission lane: inline there (or without a generation);
        from the copy lane it is handed over behind earlier admission steps, never behind a copy.
        An admission lane whose thread cannot start keeps this claim's step on the copy lane (still
        off the event loop, with the same gate), so one lane's failure refuses no prepared child."""
        if self.generation is None or self.generation.current_lane() != COPY_LANE:
            step()
            return
        current, handover = self.ender, object()
        self.ender = handover  # before the job can start: from here its invocation ends the claim
        if not self.generation.submit(functools.partial(_guarded, self, step, handover), ADMISSION_LANE):
            self.ender = current
            log.warning("Admission lane unavailable; %s is admitted on its copy lane", self.task_id)
            step()

    @classmethod
    def live(cls, generation: PreparationGeneration, evt: Dict[str, Any], ctx: Any,
             route: Dict[str, Any]) -> "PreparationClaim":
        requested = evt.get("task_constraint") if isinstance(evt.get("task_constraint"), dict) else {}
        return cls(task_id=str(evt.get("task_id") or "").strip(), ctx=ctx, route=dict(route),
                   token=uuid.uuid4().hex, generation=generation, generation_token=generation.token,
                   owner=generation.owner, event=copy.deepcopy(evt),
                   surface=str(requested.get("surface") or ""), acting=_requests_preparation(evt))

    @classmethod
    def recovered(cls, drive_root: Any, row: Dict[str, Any], marker: Dict[str, Any]) -> "PreparationClaim":
        """The custody an ended owner left: its identity and only the handle it recorded."""
        from supervisor.message_bus import send_with_budget

        tid = str(row.get("task_id") or "")
        phase = str(marker.get("phase") or "")
        # The copy returned once the marker says ``ready``; ``admitting`` keeps that same handle.
        resource = marker.get("resource") if phase in {READY, ADMITTING} and isinstance(marker.get("resource"), dict) else {}
        route = {"chat_id": row.get("chat_id"), "delegation_role": "subagent",
                 "parent_id": row.get("parent_task_id"), "role": str(row.get("role") or ""),
                 "root_task_id": str(row.get("root_task_id") or row.get("parent_task_id") or tid)}
        ctx = SimpleNamespace(DRIVE_ROOT=pathlib.Path(drive_root), send_with_budget=send_with_budget)
        return cls(task_id=tid, ctx=ctx, route=route, token=str(marker.get("token") or ""), generation=None,
                   generation_token=str(marker.get("generation") or ""),
                   owner=marker.get("owner") if isinstance(marker.get("owner"), dict) else {},
                   surface=str(marker.get("surface") or ""), resource=resource,
                   claimed_at=str(marker.get("claimed_at") or ""), unidentified_residue=phase == COPYING)

    def marker(self, phase: str, **facts: Any) -> Dict[str, Any]:
        return {"token": self.token, "phase": phase, "generation": self.generation_token,
                "owner": dict(self.owner), "surface": self.surface, "claimed_at": self.claimed_at,
                "updated_at": utc_now_iso(), **({"resource": dict(self.resource)} if self.resource else {}),
                **facts}

    def refusal_fields(self) -> Dict[str, Any]:
        disposition = ("after_refusal_receipt" if self.resource
                       else "unidentified_residue_retained" if self.unidentified_residue else "none")
        return {MARKER: self.marker(REFUSED, resource_disposition=disposition),
                "_admission_refusal_token": self.refusal_receipt}

    def refuse(self, detail: str, reason_code: str, *, extra_fields: Optional[Dict[str, Any]] = None,
               persist_result: bool = True) -> bool:
        """The typed never-admitted refusal through the schedule refusal and notice owner."""
        from supervisor.events_schedule_task import _reject_schedule_task

        route = self.route
        root_task_id = str(route.get("root_task_id") or self.task_id)
        return bool(self.store_gate(functools.partial(_reject_schedule_task,
            self.ctx, tid=self.task_id, chat_id=route.get("chat_id"),
            delegation_role=str(route.get("delegation_role") or "subagent"), parent_id=route.get("parent_id"),
            root_task_id=root_task_id, role=str(route.get("role") or ""),
            result_fields={"parent_task_id": route.get("parent_id"), "root_task_id": root_task_id,
                           "delegation_role": "subagent", "role": str(route.get("role") or "")},
            detail=detail, reason_code=reason_code, extra_fields=extra_fields,
            persist_result=persist_result, preparation=self), controls=False))

    def disclose_unknown(self) -> None:
        """Unreadable authority before any claim: the existing notice only, nothing reserved or written."""
        from supervisor.events_schedule_task import _reject_schedule_task

        route = self.route
        _reject_schedule_task(
            self.ctx, tid=self.task_id, chat_id=route.get("chat_id"),
            delegation_role=str(route.get("delegation_role") or "subagent"), parent_id=route.get("parent_id"),
            root_task_id=str(route.get("root_task_id") or self.task_id), role=str(route.get("role") or ""),
            result_fields={}, detail=_AUTHORITY_UNKNOWN, reason_code="scheduled_result_authority_unknown",
            persist_result=False)

    def after_refusal(self, remove_drive: Callable[[], None]) -> None:
        """After the receipt: this token's own copy, then the child drive, off the event loop.

        Recovery may clean here; live claims enqueue even from the FIFO thread
        so a caller's admission interlock is released first. Without an executor,
        retain the recorded copy and child drive for their existing retention
        owners; a confirmed refusal never lends the event thread to cleanup.
        """
        def cleanup() -> None:
            self.dispose()
            remove_drive()

        generation = self.generation
        if generation is None:
            cleanup()
        elif (generation.on_executor() or not generation.closed()) and generation.submit(cleanup, self.lane):
            return
        else:
            log.info("Refused preparation %s (%s) retains its resources for ordinary retention: "
                     "no cleanup executor", self.task_id, self.token)

    def unrecorded(self, landed: Optional[bool]) -> None:
        """A phase did not land: lost ownership is abandoned, an unwritable store refused."""
        if landed is False:
            self.abandon()
        else:
            self.refuse(_DETAILS[MARKER_UNWRITABLE], MARKER_UNWRITABLE)

    def prepare(self, resolve: Callable[..., tuple]) -> Optional[tuple]:
        """Run ``resolve(handle_out)`` (the copy) under this claim, off the event loop.

        Returns the resolver's tuple, or None once a control, a closed
        generation or lost ownership ended the claim (its refusal or
        abandonment already done). Controls are read before the copy starts
        and again after it returns; a copy is never interrupted mid-way. A
        readonly child's resolver is pure: no copy phases, one control read.
        """
        if not self._may_proceed():
            return None
        if not self.acting:
            return resolve(None)
        landed = self.store_gate(functools.partial(
            _transition, self.ctx.DRIVE_ROOT, self.task_id, self.token, self.marker(COPYING), expected=(QUEUED,)))
        if landed is not True:
            self.unrecorded(landed)
            return None
        handle: Dict[str, Any] = {}
        try:
            resolved = resolve(handle)
        finally:
            self.resource = {key: handle[key] for key in _RESOURCE_KEYS if key in handle}
        landed = self.store_gate(functools.partial(
            _transition, self.ctx.DRIVE_ROOT, self.task_id, self.token, self.marker(READY), expected=(COPYING,)))
        if landed is not True:
            self.unrecorded(landed)
            return None
        return resolved if self._may_proceed() else None

    def enter_admission(self) -> Optional[bool]:
        """Commit ``admitting`` before the admission path: True crossed, False lost, None unknown."""
        landed = self.store_gate(functools.partial(
            _transition, self.ctx.DRIVE_ROOT, self.task_id, self.token, self.marker(ADMITTING),
            expected=(READY,) if self.acting else (QUEUED,)))
        self.crossed = landed is not False
        return landed

    def close_admitted(self) -> None:
        """After the queue put: retire ``admitting`` on the SCHEDULED receipt, best effort (recovery retires
        it too once the queue holds the row); the put, not this write, is the admission."""
        try:
            self.store_gate(functools.partial(
                _transition, self.ctx.DRIVE_ROOT, self.task_id, self.token, self.marker(ADMITTED),
                expected=(ADMITTING,), status=STATUS_SCHEDULED), controls=False)
        except Exception:
            log.info("Admitted marker of %s stays for recovery to retire", self.task_id, exc_info=True)

    def release(self) -> None:
        from supervisor import queue

        queue.release_task_admission(self.task_id, self.token)

    def abandon(self) -> bool:
        """Leave the row unwritten; free this token's copy only once another owner holds the row.

        Returns whether another owner provably holds it. A row still this token's
        (an unwritten refusal) or unreadable keeps the copy for recovery.
        """
        from supervisor import queue
        from supervisor.task_admission import own_requested_child

        with queue._queue_lock:
            reserved = queue.ADMISSION_RESERVATIONS.get(self.task_id)
        self.release()
        try:
            row = load_task_result(self.ctx.DRIVE_ROOT, self.task_id, strict=True) or {}
        except Exception:
            return False
        if own_requested_child(row, self.token, reserved):
            return False
        if self.resource and not _row_references(row, self.resource):
            self.dispose()
        return True

    def dispose(self) -> str:
        """Remove this token's recorded copy (its refusal is confirmed and the copy returned)."""
        if not self.resource:
            return "none"
        resource, self.resource = self.resource, {}
        outcome = _dispose_resource(self.task_id, resource)
        try:
            append_jsonl(pathlib.Path(self.ctx.DRIVE_ROOT) / "logs" / "supervisor.jsonl", {
                "ts": utc_now_iso(), "type": "workspace_preparation_resource", "task_id": self.task_id,
                "preparation_token": self.token, "surface": str(resource.get("surface") or ""),
                "path": str(resource.get("path") or ""), "outcome": outcome})
        except Exception:
            log.debug("Preparation disposal row for %s was not written", self.task_id, exc_info=True)
        return outcome

    def _may_proceed(self) -> bool:
        owned = self.store_gate(self._owned)
        if owned is None:
            self.refuse(_AUTHORITY_UNKNOWN, "scheduled_result_authority_unknown", persist_result=False)
            return False
        if not owned:
            self.abandon()
            return False
        refusal = self.store_gate(self._control_refusal)
        if refusal is not None:
            self.refuse(refusal["detail"], refusal["reason_code"], extra_fields=refusal.get("extra_fields"))
            return False
        return True

    def _owned(self) -> Optional[bool]:
        from supervisor import queue
        from supervisor.task_admission import own_requested_child

        with queue._queue_lock:
            reserved = queue.ADMISSION_RESERVATIONS.get(self.task_id)
        if reserved and reserved != self.token:
            return False
        try:
            current = load_task_result(self.ctx.DRIVE_ROOT, self.task_id, strict=True) or {}
        except Exception as exc:
            if store_contention(exc):
                raise
            return None
        return own_requested_child(current, self.token, reserved)

    def _control_refusal(self) -> Optional[Dict[str, Any]]:
        """The refusal a current control requires: a closed generation, then the root latches
        ``enqueue_task`` reads (Stop/cancel, budget or owner Pause, acceptance fence)."""
        if self.generation is None or self.generation.closed():
            return {"detail": _DETAILS[INTERRUPTED_BY_GENERATION_EXIT], "reason_code": INTERRUPTED_BY_GENERATION_EXIT}
        from supervisor import queue
        from supervisor.task_admission import scheduled_admission_rejection

        root = str(self.route.get("root_task_id") or self.task_id)
        probe = {"id": self.task_id, "root_task_id": root, "parent_task_id": str(self.route.get("parent_id") or "")}
        with _no_store_wait(), queue._queue_lock:
            closed = queue.apply_budget_root_admission_fence(probe, root)
            fence = queue.ACCEPTANCE_FENCES.get(root)
            if not closed and isinstance(fence, dict) and str(fence.get("status") or "") in {"active", "sealed"}:
                probe.update(_admission_blocked="task_acceptance_fence",
                             _acceptance_fence_token=str(fence.get("token") or ""),
                             _acceptance_fence_status=str(fence.get("status") or "active"))
                closed = True
        if not closed:
            return None
        return scheduled_admission_rejection(probe, project_id=str(self.event.get("project_id") or ""),
                                             root_task_id=root)


def _transition(drive_root: Any, task_id: str, token: str, marker: Dict[str, Any], *,
                expected: Optional[tuple], status: str = STATUS_REQUESTED) -> Optional[bool]:
    """Write ``marker`` only onto this token's row still in ``status`` (``expected`` phases).

    ``expected=None`` is the first claim: the row may not carry any marker yet.
    True: the marker landed. False: the row left this token (progressed, or
    another owner). None: it did not land and the row could not prove otherwise.
    """
    def project(current: Dict[str, Any], incoming: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        held = current.get(MARKER)
        if current and str(current.get("status") or "") != status:
            return None
        if expected is None:
            if held is not None:
                return None
        elif not (isinstance(held, dict) and held.get("token") == token and held.get("phase") in expected):
            return None
        return {**incoming, MARKER: marker, "status": status}

    write_error = None
    try:
        write_task_result(drive_root, task_id, status, _field_projector=project, strict_existing_dict=True)
    except Exception as exc:
        write_error = exc
        if not store_contention(exc):
            log.warning("Preparation marker write for %s raised; reading it back", task_id, exc_info=True)
    try:
        stored = load_task_result(drive_root, task_id, strict=True) or {}
    except Exception as exc:
        if store_contention(exc):
            raise
        log.warning("Preparation marker for %s is unreadable", task_id, exc_info=True)
        return None
    held = stored.get(MARKER) if isinstance(stored.get(MARKER), dict) else None
    if str(stored.get("status") or "") == status and held == marker:
        return True
    if stored and (str(stored.get("status") or "") != status
                   or held is not None and held.get("token") != token
                   or expected is None and held is not None):
        return False
    if store_contention(write_error):
        raise write_error
    return None


def _row_references(row: Dict[str, Any], resource: Dict[str, Any]) -> bool:
    """Whether another owner's row names the copy as its workspace (then it is theirs to keep)."""
    path = str(resource.get("path") or "")
    constraint = row.get("task_constraint") if isinstance(row.get("task_constraint"), dict) else {}
    copy_binding = row.get("workspace_copy") if isinstance(row.get("workspace_copy"), dict) else {}
    return bool(path) and path in {str(row.get("workspace_root") or ""), str(constraint.get("write_root") or ""),
                                   str(copy_binding.get("execution_root") or "")}


def _dispose_resource(task_id: str, resource: Dict[str, Any]) -> str:
    """Remove exactly the copy ``resource`` names; a different or unknown holder is kept."""
    from ouroboros import genesis_project, subagent_worktrees

    surface, path = str(resource.get("surface") or ""), str(resource.get("path") or "")
    try:
        if surface == "self_worktree" and path:
            row = next((entry for entry in subagent_worktrees.list_worktrees() if entry.get("path") == path), None)
            if (row is None or row.get("kind") or str(row.get("task_id") or "") != task_id
                    or row.get("created_at") != resource.get("created_at")
                    or str(row.get("branch") or "") != str(resource.get("branch") or "")):
                return "retained_not_owned"
            return "removed" if subagent_worktrees.remove_worktree(path=path) else "retained_remove_failed"
        if surface == "genesis" and path:
            head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, capture_output=True, text=True,
                                  check=False) if pathlib.Path(path).is_dir() else None
            if head is None or head.returncode != 0 or head.stdout.strip() != str(resource.get("base_sha") or ""):
                return "retained_not_owned"
            return "removed" if genesis_project.remove_genesis_project(path, task_id=task_id) else "retained_remove_failed"
    except Exception:
        log.warning("Prepared copy %s of %s was kept: disposal failed", path, task_id, exc_info=True)
        return "retained_remove_failed"
    return "retained_unidentified"


def _owner_live(marker: Dict[str, Any]) -> bool:
    """A recorded owner still able to act: a job registered here, or another live process."""
    with _STATE_LOCK:
        if str(marker.get("token") or "") in _LIVE:
            return True
    owner = marker.get("owner") if isinstance(marker.get("owner"), dict) else {}
    try:
        pid = int(owner.get("pid") or 0)
    except (TypeError, ValueError):
        return True
    if pid == os.getpid():
        return False  # this process holds the pid: no registered job here means the owner ended
    from ouroboros.local_custody_repair import owner_ended

    return not owner_ended(pid, owner.get("birth"))


def _interruption_reason(marker: Dict[str, Any]) -> str:
    owner = marker.get("owner") if isinstance(marker.get("owner"), dict) else {}
    me = _process_owner()
    if owner.get("pid") != me["pid"] or owner.get("birth") != me["birth"]:
        return INTERRUPTED_BY_RESTART
    current = _CURRENT[0]
    if current is not None and marker.get("generation") == current.token and not current.closed():
        return MARKER_UNWRITABLE  # this generation's job ended without recording its outcome
    return INTERRUPTED_BY_GENERATION_EXIT


def recover_interrupted_preparations(drive_root: Any, *, stop: Optional[Callable[[], bool]] = None) -> Dict[str, list]:
    """Settle requests whose recorded owner positively ended (startup and periodic reconcile).

    Selects REQUESTED and SCHEDULED rows (the status-indexed read). A
    ``queued``/``copying``/``ready`` marker becomes the typed never-admitted
    refusal through the schedule refusal owner, then only its recorded copy is
    disposed; the queue's own admission of the token retires the marker; an
    ``admitting`` marker on a REQUESTED row stays unresolved on ``unknowns``,
    while on a SCHEDULED receipt the restored queue does not hold it proves the
    put was never entered (``_never_put``). A REQUESTED row without a marker is
    settled only once its request provably cannot arrive (``_request_ended``).
    A live owner, a foreign reservation or unreadable authority is left alone;
    a SCHEDULED row without a marker belongs to the queue restore.
    """
    from ouroboros import obligations
    from ouroboros.task_store import list_documents

    root = pathlib.Path(drive_root)
    report: Dict[str, list] = {"settled": [], "retained": [], "admitted": [], "uncertain": [], "live": [],
                               "errors": []}
    try:
        _retire_resolved_gaps(root, obligations)
        rows = list_documents(root, statuses=[STATUS_REQUESTED, STATUS_SCHEDULED])
    except Exception as exc:
        log.warning("Preparation recovery could not read requested rows", exc_info=True)
        report["errors"].append(f"requested_rows_unreadable:{type(exc).__name__}")
        return report
    for row in rows:
        marker = row.get(MARKER)
        marked = isinstance(marker, dict) and bool(str(marker.get("token") or ""))
        if not marked and str(row.get("status") or "") != STATUS_REQUESTED:
            continue
        if stop is not None and stop():
            break
        tid = str(row.get("task_id") or "")
        try:
            outcome = _recover_one(root, row, marker, obligations) if marked else _recover_unmarked(root, row)
            report[outcome].append(tid)
        except Exception:
            log.warning("Preparation recovery of %s failed; it stays for the next pass", tid, exc_info=True)
            report["errors"].append(tid)
    return report


def _recover_one(root: pathlib.Path, row: Dict[str, Any], marker: Dict[str, Any], obligations: Any) -> str:
    from supervisor import queue

    tid, token = str(row.get("task_id") or ""), str(marker.get("token") or "")
    if _owner_live(marker):
        return "live"
    with queue._queue_lock:
        reserved = queue.ADMISSION_RESERVATIONS.get(tid)
        queued = next((item for item in queue.PENDING if isinstance(item, dict) and str(item.get("id") or "") == tid),
                      None)
        meta = queue.RUNNING.get(tid)
        if queued is None and isinstance(meta, dict):
            queued = meta.get("task") if isinstance(meta.get("task"), dict) else {"id": tid}
    if reserved and reserved != token:
        return "live"
    phase, status = str(marker.get("phase") or ""), str(row.get("status") or "")
    if queued is not None:
        if phase == ADMITTING and str(queued.get("_admission_owner_token") or "") == token:
            _transition(root, tid, token, {**marker, "phase": ADMITTED, "updated_at": utc_now_iso(),
                                           "retired_by": "recovery"}, expected=(ADMITTING,), status=status)
        return "admitted"  # the queue owns admitted work
    if phase in UNSENT_PHASES or _never_put(status, phase):
        reason = _interruption_reason(marker)
        confirmed = PreparationClaim.recovered(root, row, marker).refuse(_DETAILS[reason], reason)
        return "settled" if confirmed else "retained"
    obligations.add(root, "unknowns", f"task:{tid}", {
        "task_id": tid, "reason": ADMISSION_UNCERTAIN, "phase": phase, "preparation_token": token})
    return "uncertain"


def _never_put(status: str, phase: str) -> bool:
    """A SCHEDULED receipt still ``admitting`` that the restored queue does not hold: every queue put is
    preceded by a published locator snapshot naming the row, so once this process took the saved
    queue into custody and found no row, the put was never entered. An unrestored saved queue proves
    nothing (the row may be waiting in it)."""
    from supervisor.queue_snapshot import saved_queue_unowned

    return status == STATUS_SCHEDULED and phase == ADMITTING and not saved_queue_unowned()


def _recover_unmarked(root: pathlib.Path, row: Dict[str, Any]) -> str:
    """A REQUESTED row no claim ever marked: refuse it once its event provably cannot arrive."""
    from supervisor import queue
    from supervisor.events_schedule_task import _reject_schedule_task

    tid = str(row.get("task_id") or "")
    with _STATE_LOCK:
        claimed = any(claim.task_id == tid for claim in _LIVE.values())
    with queue._queue_lock:
        held = (claimed or tid in queue.ADMISSION_RESERVATIONS or tid in queue.RUNNING
                or any(isinstance(item, dict) and str(item.get("id") or "") == tid for item in queue.PENDING))
    if held:
        return "live"
    reason = _request_ended(root, row)
    if not reason:
        return "live"
    claim = PreparationClaim.recovered(root, row, {})
    route = claim.route
    confirmed = _reject_schedule_task(
        claim.ctx, tid=tid, chat_id=route.get("chat_id"), delegation_role="subagent",
        parent_id=route.get("parent_id"), root_task_id=route["root_task_id"], role=route["role"],
        result_fields={"parent_task_id": route.get("parent_id"), "root_task_id": route["root_task_id"],
                       "delegation_role": "subagent", "role": route["role"]},
        detail=_DETAILS[reason], reason_code=reason, extra_fields={"admission_outcome": "never_admitted"})
    return "settled" if confirmed else "retained"


def _request_ended(root: pathlib.Path, row: Dict[str, Any]) -> str:
    """Why the request can no longer arrive: its stamped producer and bus owner both ended
    (``requested_by``), or, for a row that predates the stamp, its parent is terminal; else ''."""
    from ouroboros.local_custody_repair import owner_ended
    from ouroboros.task_results import _TRULY_TERMINAL_STATUSES

    stamp = row.get("requested_by") if isinstance(row.get("requested_by"), dict) else None
    if stamp is not None:
        ended = (owner_ended(stamp.get("pid"), stamp.get("birth"))
                 and owner_ended(stamp.get("host_pid"), stamp.get("host_birth")))
        return REQUEST_LOST if ended else ""
    parent = str(row.get("parent_task_id") or "")
    if not parent:
        return ""
    try:
        parent_row = load_task_result(root, parent, strict=True) or {}
    except Exception:
        return ""
    return REQUEST_ORPHANED if str(parent_row.get("status") or "") in _TRULY_TERMINAL_STATUSES else ""


def _retire_resolved_gaps(root: pathlib.Path, obligations: Any) -> None:
    """Drop this owner's ``unknowns`` rows once their row left that claim (a lifecycle owner moved it)."""
    for identity, facts in obligations.members(root, "unknowns").items():
        if facts.get("reason") != ADMISSION_UNCERTAIN:
            continue
        current = load_task_result(root, str(facts.get("task_id") or ""), strict=True) or {}
        held = current.get(MARKER) if isinstance(current.get(MARKER), dict) else {}
        if (str(current.get("status") or "") not in {STATUS_REQUESTED, STATUS_SCHEDULED}
                or held.get("token") != facts.get("preparation_token")):
            obligations.remove(root, "unknowns", identity)


def _reset_for_tests() -> None:
    """Forget this process's generation and live claims (test isolation only)."""
    with _STATE_LOCK:
        _CURRENT[0] = None
        _LIVE.clear()
