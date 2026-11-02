"""In-memory, process-local registry for active direct chat activities.

Tracks in-flight direct conversational turns so
the Gateway (/api/state) and WebSocket activity pipeline have authoritative,
thread-safe visibility into in-progress work without creating spurious queue records,
and each turn's execution (its thread and the start facts it passed) so the
liveness watchdog can say where a turn that stopped advancing stands.

An owner turn leaves one receipt (``TurnExecution.receipt``): the canonical acceptance its wait
started at, each start phase's seconds, the dispatch basis and its own store counts. It rides the
turn's first dispatch frame by value to the supervisor's events consumer, which records it
(``record_direct_turn``: one ``direct_turn`` row in ``logs/supervisor.jsonl``, written off the
turn thread without waiting on the journal, and this serving process's bounded ring); a turn
that ends without an observed dispatch records it from its unregister, ``dispatch: not_observed``.
"""

from __future__ import annotations

import copy
import datetime
import logging
import pathlib
import threading
import time
import uuid
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Iterator, List, Optional, Tuple

from ouroboros import store_timing
from ouroboros.deadline_utils import parse_deadline_ts
from ouroboros.utils import utc_now_iso

log = logging.getLogger(__name__)

# The start facts a direct turn's own thread passes up to its first model dispatch, in
# order. Each is stamped by the code that has just passed it; the liveness watchdog reads
# them (server_liveness) and never waits on them. ``requested``: its Main call started;
# ``dispatched``: that call's first observation of its request going out, or, on a route
# that observes nothing, its start once it finished (preparing, an engine queue or a slot
# is not dispatch). From ``requested`` on, the time is the call's (``CALL_PHASES``).
TURN_PHASES = ("accepted", "admitted", "started", "context_building", "context_built", "requested", "dispatched")
CALL_PHASES = ("requested", "dispatched")
_LOG_FACTS = {"context_building_started": "context_building", "context_building_finished": "context_built"}
_SENDING = ("dispatching", "awaiting_response")  # a model operation's request going out (the cards' Sending)
# How a turn's first dispatch was observed: a send seen going out; the start of a call whose route
# observes nothing, once it finished; none before the turn ended.
DISPATCH_BASES = ("observed_send", "unobserved_call_start", "not_observed")
DIRECT_TURNS_KEPT = 100  # the receipt ring's length: a memory bound, never a judgment
_DIRECT_TURNS: Deque[Dict[str, Any]] = deque(maxlen=DIRECT_TURNS_KEPT)
_RING_LOCK = threading.Lock()


def _iso(epoch: float) -> str:
    return datetime.datetime.fromtimestamp(epoch, datetime.timezone.utc).isoformat()


def _epoch(stamp: Any) -> Optional[float]:
    parsed = parse_deadline_ts(stamp)
    return None if parsed is None else parsed.timestamp()


def _calling_thread() -> Tuple[int, str]:
    current = threading.current_thread()
    return current.ident or 0, current.name


@dataclass(eq=False)
class TurnExecution:
    """One execution of a direct turn: the thread running it, a process-local identity and
    ``marks``, each start fact it passed (``TURN_PHASES``) with the monotonic instant it was
    stamped. Facts only move forward, so there is at most one mark per phase; the list is
    only appended to, so a reader's copy never pairs one fact with another fact's instant.
    ``owner_turn``: the owner awaits its answer (its entry's ``reply_expected``); only such a
    turn reports its first dispatch to the after-update facts and leaves a receipt, a wake or
    resume never does. ``call``: its latest Main call (id, start instant, whether its route
    observed it); ``ended_at``: the instant it left the registry. ``accepted_at``: the canonical
    acceptance of the message it answers (the inbound row's UTC stamp, its entry's
    ``origin_message_ref["ts"]``), where the owner's wait starts; ``entered_at`` is the wall-clock
    twin of the first mark, so that wait and the monotonic phases (one process only) add up.
    ``dispatch``: the first dispatch's basis (``DISPATCH_BASES``) once it happened; ``meter``:
    the thread's ``store_timing.metered`` scope; ``census``: the registry's count at entry."""

    chat_id: int
    client_message_id: str = ""
    task_id: str = ""
    owner_turn: bool = False
    execution_id: str = field(default_factory=lambda: f"turn_{uuid.uuid4().hex[:12]}")
    thread: Tuple[int, str] = field(default_factory=_calling_thread)
    marks: List[Tuple[str, float]] = field(default_factory=lambda: [("accepted", time.monotonic())])
    call: Optional[Tuple[str, float, bool]] = None
    ended_at: Optional[float] = None
    accepted_at: str = ""
    entered_at: float = field(default_factory=time.time)
    dispatch: str = ""
    meter: Optional[Dict[str, Any]] = field(default=None, repr=False, compare=False)
    census: Dict[str, Any] = field(default_factory=dict)

    @property
    def phase(self) -> str:
        return self.marks[-1][0]

    def bind_thread(self) -> None:
        """The calling thread runs this execution from now on (a wake is admitted on its alarm's thread)."""
        self.thread = _calling_thread()

    def advance(self, phase: str, at: Optional[float] = None) -> bool:
        if TURN_PHASES.index(phase) <= TURN_PHASES.index(self.phase):
            return False
        self.marks.append((phase, time.monotonic() if at is None else at))
        if phase == "dispatched" and self.owner_turn:  # once per execution: facts only move forward
            try:
                from ouroboros.update_generation_facts import observe_direct_dispatch

                observe_direct_dispatch(task_id=self.task_id, attempt_id=self.execution_id,
                                        accepted_at=self.marks[0][1], dispatched_at=self.marks[-1][1])
            except Exception:
                log.debug("after-update dispatch observation failed", exc_info=True)
        return True

    def observe(self, kind: str, data: Dict[str, Any]) -> None:
        """One of the turn's own frames (``TurnEventQueue``); a start fact among them advances the phase."""
        if kind == "log_event" and data.get("type") in _LOG_FACTS:
            self.advance(_LOG_FACTS[data["type"]])
        if kind != "main_llm_call_state":
            return
        call_id, phase = str(data.get("llm_call_id") or ""), data.get("phase")
        if phase == "started":
            self.call = (call_id, time.monotonic(), False)
            self.advance("requested", self.call[1])
            return
        if self.call is None or self.call[0] != call_id:
            return  # a frame of a call this execution did not see start
        if phase == "observed":
            self.call = (call_id, self.call[1], True)
            activity = data.get("model_activity")
            if isinstance(activity, dict) and activity.get("phase") in _SENDING:
                self._dispatched("observed_send", data)
        elif phase == "finished" and not self.call[2]:
            self._dispatched("unobserved_call_start", data, at=self.call[1])

    def _dispatched(self, basis: str, frame: Dict[str, Any], at: Optional[float] = None) -> None:
        """The first dispatch, once: its basis, and an owner turn's receipt by value on the frame
        that carries it to the supervisor (``events._handle_main_llm_call_state`` records it)."""
        if self.advance("dispatched", at):
            self.dispatch = basis
            if self.owner_turn:
                frame["direct_turn"] = self.receipt()

    def receipt(self, until: Optional[float] = None) -> Dict[str, Any]:
        """This execution's facts by value, as of ``until`` (monotonic: its end, else now).

        ``phases``: the seconds each start phase stood, up to the next one, the dispatch, the end or
        ``until``, the slowest named; ``handler_wait_sec``: the thread's first mark to the same point.
        ``accepted_at``, ``wait_before_entry_sec`` (wall clock: the routing pre-pass ran before this
        thread) and ``owner_wait_sec`` only when the canonical acceptance is known. ``dispatch``:
        ``not_observed`` while none happened; ``dispatched_at`` from the monotonic delta once it did.
        ``meter``: the thread's own store counts so far; ``census``: the registry's count at entry."""
        marks = list(self.marks)
        if until is None:
            until = self.ended_at if self.ended_at is not None else time.monotonic()
        dispatched = marks[-1][0] == "dispatched"
        phases: Dict[str, float] = {}
        for index, (phase, at) in enumerate(marks):
            if phase == "dispatched":
                break
            end = marks[index + 1][1] if index + 1 < len(marks) else until
            phases[phase] = round(end - at, 3)
        handler_wait = (marks[-1][1] if dispatched else until) - marks[0][1]
        row: Dict[str, Any] = {
            "type": "direct_turn", "chat_id": self.chat_id, "task_id": self.task_id,
            "execution_id": self.execution_id, "client_message_id": self.client_message_id,
            "thread": self.thread[1], "entered_at": _iso(self.entered_at),
            "handler_wait_sec": round(handler_wait, 3), "phases": phases,
            "dispatch": self.dispatch or "not_observed",
            "meter": store_timing.snapshot(self.meter), "census": dict(self.census)}
        if phases:
            row["slowest_phase"], row["slowest_phase_sec"] = max(phases.items(), key=lambda item: item[1])
        accepted = _epoch(self.accepted_at)
        if accepted is not None:
            before = self.entered_at - accepted
            row.update(accepted_at=self.accepted_at, wait_before_entry_sec=round(before, 3),
                       owner_wait_sec=round(before + handler_wait, 3))
        if dispatched:
            row["dispatched_at"] = _iso(self.entered_at + (marks[-1][1] - marks[0][1]))
        return row


@dataclass
class DirectActivityEntry:
    activity_id: str
    chat_id: int
    project_id: str = ""
    client_message_id: str = ""
    kind: str = "direct_chat"
    phase: str = "preparing"
    started_at: float = field(default_factory=time.time)
    origin_message_ref: Dict[str, Any] = field(default_factory=dict)
    # Reply-preparation facts (DESIGN §9, Telegram typing), never census fields:
    # the owner's door stamped this turn, so its answer is expected where it is
    # routed; and that answer has entered the host's delivery path.
    reply_expected: bool = False
    reply_handed_off: bool = False

    actor: Any = field(default=None, repr=False, compare=False)
    execution: Optional[TurnExecution] = field(default=None, repr=False, compare=False)
    active_llm_call: Optional[Dict[str, Any]] = field(default=None, repr=False, compare=False)
    parked_wait: Optional[Dict[str, Any]] = field(default=None, repr=False, compare=False)
    parked_root: Any = field(default=None, repr=False, compare=False)

    def _live_model_wait_projection(self, *, availability: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Read the direct turn's existing live ``TaskModelWait`` owner.

        Direct turns do not have a queue ``RUNNING`` row to carry the wait
        projection.  The actor's task-scoped ``ToolContext`` already points at
        the live owner, so use that authority while it is open instead of
        replaying a durable result or introducing a second poller/store.
        """
        try:
            actor = self.actor
            tools = getattr(actor, "tools", None)
            context = getattr(tools, "_ctx", None)
            owner = getattr(context, "model_wait_context", None)
        except Exception:
            if availability is not None:
                availability["complete"] = False
            log.debug("Direct activity owner lookup unavailable: %s", self.activity_id,
                      exc_info=True)
            return {}
        if owner is None or str(getattr(owner, "task_id", "") or "") != self.activity_id:
            return {}
        if bool(getattr(owner, "closed", False)):
            return {}
        snapshot = getattr(owner, "snapshot", None)
        if not callable(snapshot):
            if availability is not None:
                availability["complete"] = False
            return {}
        try:
            projection = snapshot()
        except Exception:
            if availability is not None:
                availability["complete"] = False
            log.debug("Direct activity model-wait snapshot unavailable: %s", self.activity_id,
                      exc_info=True)
            return {}
        if not isinstance(projection, dict):
            if availability is not None:
                availability["complete"] = False
            return {}
        result: Dict[str, Any] = {}
        waits = projection.get("model_waits")
        if isinstance(waits, dict) and waits:
            result["model_waits"] = waits
        try:
            attempt = int(getattr(owner, "attempt", 0) or 0)
        except (TypeError, ValueError):
            attempt = 0
        if attempt >= 1:
            result["task_attempt"] = attempt
        return result

    def to_dict(self, *, availability: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        row = {
            "activity_id": self.activity_id,
            "chat_id": self.chat_id,
            "project_id": self.project_id,
            "client_message_id": self.client_message_id,
            "kind": self.kind,
            "phase": self.phase,
            "started_at": self.started_at,
        }
        if self.execution is not None:
            row["turn_phase"] = self.execution.phase
        # Keep the historical static shape for an actor that has not entered
        # its task context yet. Once the direct turn owns a live model wait,
        # publish the same optional fields as managed/post-task activities.
        wait_availability: Dict[str, Any] = {}
        row.update(self._live_model_wait_projection(availability=wait_availability))
        active = self.active_llm_call
        if active and active.get("task_attempt") == row.get("task_attempt") and active.get("model_activity"):
            row["model_activity"] = dict(active["model_activity"])
        if wait_availability.get("complete") is False:
            row["phase"] = "unknown"
            if availability is not None:
                availability["complete"] = False
        return row


class DirectActivityRegistry:
    """Thread-safe registry for active direct-chat turns."""

    def __init__(self) -> None:
        self._lock = threading.Condition()
        self._activities: Dict[str, DirectActivityEntry] = {}
        # Every running turn execution, from its thread's first fact (before registration) to its end.
        self._executions: Dict[str, TurnExecution] = {}

    def begin_execution(self, chat_id: int, *, client_message_id: str = "") -> TurnExecution:
        """The calling thread starts one direct turn (``accepted``), before admission decides it."""
        execution = TurnExecution(int(chat_id or 0), str(client_message_id or ""))
        with self._lock:
            execution.census = {"direct_turns_at_entry": len(self._executions)}
            self._executions[execution.execution_id] = execution
        return execution

    def end_execution(self, execution: Optional[TurnExecution]) -> None:
        with self._lock:
            ended = self._drop(execution)
        _undispatched_receipt(ended)

    def _drop(self, execution: Optional[TurnExecution]) -> Optional[TurnExecution]:
        """Under the lock: the execution ends now, the instant a stall episode it stood in closes
        at; returns it when it was still registered (its receipt, if owed, is written after the lock)."""
        if execution is not None and self._executions.pop(execution.execution_id, None) is not None:
            execution.ended_at = time.monotonic()
            return execution
        return None

    def executions(self) -> List[TurnExecution]:
        """Every running execution, registered or not yet: what the liveness watchdog reads."""
        with self._lock:
            return list(self._executions.values())

    def register(
        self,
        activity_id: str,
        chat_id: int,
        *,
        client_message_id: str = "",
        project_id: str = "",
        kind: str = "direct_chat",
        phase: str = "preparing",
        origin_message_ref: Optional[Dict[str, Any]] = None,
        actor: Any = None,
        reply_expected: bool = False,
    ) -> DirectActivityEntry:
        """Register one turn. It adopts the execution the calling thread began and has not
        registered yet (``begin_execution``), else one begins here; the entry's unregister ends it."""
        aid = str(activity_id or "").strip()
        if not aid:
            raise ValueError("activity_id cannot be empty")
        ident = threading.get_ident()
        with self._lock:
            begun = next((row for row in self._executions.values()
                          if row.thread[0] == ident and not row.task_id), None)
        execution = begun or TurnExecution(int(chat_id or 0))
        execution.task_id = aid
        execution.owner_turn = bool(reply_expected)
        execution.client_message_id = execution.client_message_id or str(client_message_id or "")
        execution.accepted_at = str((origin_message_ref or {}).get("ts") or "")
        execution.advance("admitted")
        entry = DirectActivityEntry(
            activity_id=aid,
            chat_id=int(chat_id or 0),
            project_id=str(project_id or ""),
            client_message_id=str(client_message_id or ""),
            kind=str(kind or "direct_chat"),
            phase=str(phase or "preparing"),
            started_at=time.time(),
            origin_message_ref=dict(origin_message_ref or {}),
            actor=actor,
            reply_expected=bool(reply_expected),
            execution=execution,
        )
        with self._lock:
            replaced = self._activities.get(aid)
            ended = self._drop(replaced.execution) if replaced is not None else None
            if begun is None:
                execution.census = {"direct_turns_at_entry": len(self._executions)}
            self._activities[aid] = entry
            self._executions[execution.execution_id] = execution
        _undispatched_receipt(ended)
        log.debug("Registered direct activity: %s (chat_id=%s, kind=%s)", aid, chat_id, kind)
        return entry

    def unregister(self, activity_id: str) -> Optional[DirectActivityEntry]:
        aid = str(activity_id or "").strip()
        with self._lock:
            entry = self._activities.pop(aid, None)
            ended = self._drop(entry.execution) if entry is not None else None
            self._lock.notify_all()
        _undispatched_receipt(ended)
        if entry:
            log.debug("Unregistered direct activity: %s (chat_id=%s)", aid, entry.chat_id)
        return entry

    def snapshot(
        self, chat_id: Optional[int] = None, *, availability: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        with self._lock:
            entries = list(self._activities.values())
        return [e.to_dict(availability=availability) for e in entries
                if chat_id is None or e.chat_id == int(chat_id)]

    def get(self, activity_id: str) -> Optional[DirectActivityEntry]:
        aid = str(activity_id or "").strip()
        with self._lock:
            return self._activities.get(aid)

    def actors(self) -> List[DirectActivityEntry]:
        """Private actor handles; never include execution objects in UI snapshots."""
        with self._lock:
            return list(self._activities.values())

    def mark_parked(
        self,
        activity_id: str,
        wait: Optional[Dict[str, Any]],
        *,
        drive_root: Any = None,
    ) -> bool:
        aid = str(activity_id or "").strip()
        if not aid:
            return False
        with self._lock:
            entry = self._activities.get(aid)
            if entry is None:
                return False
            entry.parked_wait = dict(wait) if isinstance(wait, dict) else None
            entry.parked_root = drive_root
            self._lock.notify_all()
            return True

    def clear_parked(self, activity_id: str) -> bool:
        aid = str(activity_id or "").strip()
        if not aid:
            return False
        with self._lock:
            entry = self._activities.get(aid)
            if entry is None:
                return False
            entry.parked_wait = None
            entry.parked_root = None
            self._lock.notify_all()
            return True

    def wait_until_empty(self, timeout: float) -> List[str]:
        """Wait for unparked executions, including preparation and post-task work."""
        deadline = time.monotonic() + max(0.0, float(timeout))
        while True:
            with self._lock:
                entries = list(self._activities.items())
            # The parked predicate reads the task store and run custody; it runs
            # outside the registry lock so registration and census stay responsive.
            remaining_ids = [aid for aid, entry in entries if not is_parked_direct_entry(entry)]
            if not remaining_ids:
                return []
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return remaining_ids
            with self._lock:
                self._lock.wait(timeout=min(remaining, 0.05))

    def clear(self) -> None:
        """Clear registry — primarily for tests and process resets."""
        with self._lock:
            self._activities.clear()
            for execution in list(self._executions.values()):
                self._drop(execution)
            self._lock.notify_all()


# Global process-local singleton
_DIRECT_ACTIVITY_REGISTRY = DirectActivityRegistry()


def get_direct_activity_registry() -> DirectActivityRegistry:
    return _DIRECT_ACTIVITY_REGISTRY


def _undispatched_receipt(execution: Optional[TurnExecution]) -> None:
    """An owner turn that ended without an observed dispatch records its receipt here, on the
    thread that ended it (the turn is over: nothing waits on this), ``dispatch: not_observed``."""
    if execution is None or not execution.owner_turn or execution.phase == "dispatched":
        return
    try:
        record_direct_turn(execution.receipt())
    except Exception:
        log.debug("direct_turn receipt failed for %s", execution.execution_id, exc_info=True)


def record_direct_turn(row: Dict[str, Any], *, drive_root: Any = None) -> Optional[Dict[str, Any]]:
    """One owner turn's receipt: a ``direct_turn`` row into ``logs/supervisor.jsonl`` (one attempt,
    no wait on the journal lock: a diagnostic never delays the loop) and into this process's ring,
    newest last, ``journal`` saying whether the row was written. The same execution's receipt is
    recorded once (a frame handled twice records nothing); returns the recorded row, else None."""
    execution_id = str(row.get("execution_id") or "")
    with _RING_LOCK:
        if any(kept.get("execution_id") == execution_id for kept in _DIRECT_TURNS):
            return None
    row = {"ts": utc_now_iso(), **row}
    try:
        from supervisor.state import DRIVE_ROOT, append_jsonl

        root = pathlib.Path(drive_root) if drive_root else DRIVE_ROOT
        written = bool(append_jsonl(root / "logs" / "supervisor.jsonl", row, lock_timeout_sec=0))
    except Exception:
        log.debug("direct_turn row not written", exc_info=True)
        written = False
    kept = {**row, "journal": "written" if written else "not_written"}
    with _RING_LOCK:
        _DIRECT_TURNS.append(kept)
    return kept


def direct_turns(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """The newest ``limit`` receipts this process recorded (all when None), oldest first, by value."""
    with _RING_LOCK:
        rows = list(_DIRECT_TURNS)
    return copy.deepcopy(rows if limit is None else rows[-max(0, int(limit)):] if limit else [])


def is_parked_owner_wait(task_id: str, wait: Any, *, drive_root: Any = None) -> bool:
    """True when ``task_id`` is durably parked in ``owner_wait`` with no live detached or open runs."""
    tid = str(task_id or "").strip()
    if not tid or not isinstance(wait, dict) or wait.get("state") != "waiting":
        return False
    effective_wait = wait
    if drive_root is not None:
        try:
            from ouroboros.task_results import load_task_result

            row = (load_task_result(drive_root, tid, strict=True) or {}).get("owner_wait")
        except Exception:
            return False
        if not isinstance(row, dict) or row.get("state") != "waiting":
            return False
        effective_wait = row
    try:
        from ouroboros.review_pause import live_detached_operations

        if live_detached_operations(tid):
            return False
    except Exception:
        return False
    if effective_wait.get("reason") == "owner_pause":
        try:
            from ouroboros.budget_pause import observe_task_runs, task_owned_runs_open

            owner_pause = effective_wait.get("owner_pause")
            external = owner_pause.get("external_runs") if isinstance(owner_pause, dict) else None
            if task_owned_runs_open(external):
                if drive_root is None:
                    return False
                external = observe_task_runs(
                    drive_root, tid, reason="owner_pause_retained_runs", request_stop=False,
                )
                if task_owned_runs_open(external):
                    return False
        except Exception:
            return False
    return True


def is_parked_direct_entry(entry: DirectActivityEntry) -> bool:
    if entry.parked_wait is None:
        return False
    return is_parked_owner_wait(entry.activity_id, entry.parked_wait, drive_root=entry.parked_root)


def reply_expected(record: Any) -> bool:
    """Whether the owner awaits this root's answer: the owner's door stamped it
    (``run_origin``'s ``owner_ingress``, which promoted, continued and retried roots
    carry by value) and it is no Presence correspondent's turn. Consciousness,
    schedule and API/CLI roots never carry that stamp; a child counts through its root."""
    from ouroboros.dialogue_provenance import is_presence_task, run_origin

    source = record if isinstance(record, dict) else {}
    return bool(run_origin(source)["owner_ingress"]) and not is_presence_task(source)


def mark_reply_handed_off(task_id: str, running: Any = None) -> None:
    """The task's terminal answer entered the delivery path: its reply is no longer prepared.

    Process-local like both owners it marks, the direct turn's entry or the RUNNING
    row's meta (never the persisted task record), so post-answer work that keeps
    either alive cannot read as preparation (``gateway/reply_preparation.py``).
    """
    tid = str(task_id or "").strip()
    meta = running.get(tid) if isinstance(running, dict) and tid else None
    if isinstance(meta, dict):
        meta["reply_handed_off"] = True
    entry = get_direct_activity_registry().get(tid) if tid else None
    if entry is not None:
        entry.reply_handed_off = True


@contextmanager
def track_direct_activity(
    activity_id: str,
    chat_id: int,
    *,
    client_message_id: str = "",
    project_id: str = "",
    kind: str = "direct_chat",
    phase: str = "preparing",
    origin_message_ref: Optional[Dict[str, Any]] = None,
) -> Iterator[DirectActivityEntry]:
    registry = get_direct_activity_registry()
    entry = registry.register(
        activity_id,
        chat_id,
        client_message_id=client_message_id,
        project_id=project_id,
        kind=kind,
        phase=phase,
        origin_message_ref=origin_message_ref,
    )
    try:
        yield entry
    finally:
        registry.unregister(activity_id)
