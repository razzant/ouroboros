"""What the body tells a held run's owner while the run works: its speech and the engine's reach.

A delegated run once spoke to the owner only while its author sat in ``delegate_wait``'s supervising
tick, which alone told the owner the engine had stopped answering; that tick is gone (``delegate_wait``
is a sleep on runs). The body (``delegate_collection.BodyRuntime``) observes every held run whoever
waits, so both are its own here, one owner each (pool-capacity D3), and its journal read is the only
one made for a run:

* **Speech.** A run the body observes, whose task is live (running, queued or asleep, or a direct
  turn), is followed: one journal read at a time (``gateways.claudexor_run_events``; the compact wait
  carries no speech) on the run's own transport and thread, so a quiet run's held read delays no other
  run's words. Each read lasts one observation beat from the run's cursor; its events become one typed
  record, published as the wait published it (``delegate_progress.emit_stream``) and committed only once
  the frame was handed over. A new body resumes at the last retained range (re-read once: a reader keeps
  each seq once). After the body records the run's terminal the follow drains the journal to its end. An
  engine without the stream, or a journal the engine refused for good (``journal_refused``: what it served is
  shown first, it is never read again), keeps the window view: each detail the body reads gives the record of
  its new timeline rows (``delegate_progress.emit``).
* **Engine unreachable.** A pass nothing answered (a socket that carried nothing, never a slow read or
  a refusal) opens one episode; the first answer closes it. Each live task holding an observed run hears
  both once per episode through its own progress channel.
* **My human's words owed to a run.** Frames the host's placement at receipt left pending (the engine
  did not answer then, or the send's outcome was unknown) are sent by the body once the engine answers
  the run again, on its own observation pass (``delegate_message_store.replay_frames``); the starter's
  round top stays the idempotent catch-up.
* **The parent's view.** A dated record of what the body last saw of a held run
  (``state/delegate_observation/<run_id>.json``, about 300 bytes; ``read_observation``): written
  only when the run's state or question changed, or the engine stopped or resumed answering it
  (the outage class above), with the time that began, never once per read; removed once the run
  settled or left the held set (a new body sweeps what an earlier one left).

A frame reaches the owner as its worker's progress frame does (``Agent._emit_progress``): one
``send_message`` progress event on the supervisor's bus, with the task's chat and lineage.
"""
from __future__ import annotations

import json
import logging
import os
import pathlib
import threading
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Optional

from ouroboros.utils import atomic_write_json

log = logging.getLogger(__name__)
_TICK_SEC = 3  # the body's observation beat
_OBSERVATION_CURRENT_SEC = 4 * _TICK_SEC

_UNREACHABLE = "daemon_unreachable"  # the gateway's reason for a read nothing answered (``_observation_reason``)
_ENDED = "ended"
_KEPT = ("task_id", "invocation_id", "run_state", "waiting_on_user", "async_questions_only", "answered",
         "reason")  # what a record says


def _observation_dir(root: Any) -> pathlib.Path:
    return pathlib.Path(root) / "state" / "delegate_observation"


def observation_path(root: Any, run_id: str) -> pathlib.Path:
    return _observation_dir(root) / f"{run_id}.json"


def read_observation(root: Any, run_id: str) -> tuple:
    """``(record, "")`` of the body's latest record of a run, else ``(None, why)``: one read with no lock and no
    retry, so a missing or racing file (Windows ``os.replace``) is the fact ``unknown``."""
    try:
        record = json.loads(observation_path(root, run_id).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, "no_observation_record"
    except (OSError, ValueError):
        return None, "unreadable"
    return (record, "") if isinstance(record, dict) else (None, "unreadable")


def forget_observations(root: Any, run_ids: Iterable[str]) -> None:
    """Remove the records of runs that settled or left the held set (``Feed.refresh``)."""
    for run_id in run_ids:
        try:
            observation_path(root, run_id).unlink(missing_ok=True)
        except OSError:
            log.debug("Observation record of %s not removed; the next pass retries", run_id, exc_info=True)


def _bus() -> Any:
    from supervisor.workers import get_event_q

    return get_event_q(create=False)


def live_tasks(task_ids: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    """The live tasks among ``task_ids`` and where their frames go: a running task's row, a queued or
    sleeping one's pending row, a direct turn's registry entry. A task nobody runs or holds has no card
    to speak into. One look at the queue for the lot, never under the body's own lock."""
    from ouroboros.subagent_messages import initiator_meta, subagent_message_meta
    from supervisor import queue as q
    from supervisor.active_activity import get_direct_activity_registry

    wanted, rows = {str(task_id) for task_id in task_ids if task_id}, {}
    if wanted:
        with q._queue_lock:
            for task_id in wanted:
                meta = q.RUNNING.get(task_id)
                if isinstance(meta, dict) and isinstance(meta.get("task"), dict):
                    rows[task_id] = meta["task"]
            missing = wanted - set(rows)
            if missing:  # one pass over the pending rows, memory only (no I/O under the queue lock)
                rows.update((str(row.get("id")), row) for row in q.PENDING
                            if isinstance(row, dict) and str(row.get("id") or "") in missing)
        registry = get_direct_activity_registry()
        for task_id in wanted - set(rows):
            turn = registry.get(task_id)
            if turn is not None:  # its metadata as its own progress binder reads it
                metadata = getattr(getattr(turn, "actor", None), "_current_task_metadata", None)
                rows[task_id] = {**(metadata if isinstance(metadata, dict) else {}), "id": task_id,
                                 "chat_id": turn.chat_id, "_is_direct_chat": True}
    targets = {}
    for task_id, row in rows.items():
        meta = subagent_message_meta(row, task_id=task_id, event="progress")
        meta.update(initiator_meta(row))
        targets[task_id] = {"chat_id": row.get("chat_id"), "meta": meta, "attempt": row.get("_attempt"),
                            "direct": bool(row.get("_is_direct_chat"))}
    return targets


def publisher(task_id: str, target: Dict[str, Any]):
    """The task's progress channel as its worker binds it (``Agent._emit_progress``): one call, one event."""
    from ouroboros.subagent_messages import delegated_activity_meta, executor_observation_meta
    from ouroboros.utils import utc_now_iso

    def emit(text: str, *, incident=None, executor_observation=None, delegated_activity=None) -> None:
        meta = {**(incident or {}), **target["meta"]}
        if executor_observation is not None and (observation := executor_observation_meta(
                executor_observation, task_id=task_id, task_attempt=target["attempt"])):
            meta["executor_observation"] = observation
        if delegated_activity is not None and (activity := delegated_activity_meta(
                delegated_activity, task_id=task_id)):
            meta["delegated_activity"] = activity
        meta["narration"] = False
        _bus().put({"type": "send_message", "chat_id": target["chat_id"], "text": f"💬 {text}",
                    "format": "markdown", "is_progress": True, "role": "system", "system_type": "host_progress",
                    "task_id": task_id, "ts": utc_now_iso(), "progress_meta": meta,
                    **({"_is_direct_chat": True} if target["direct"] else {})})
    return emit


def _context(root: pathlib.Path, entry: Any, target: Dict[str, Any], handed: list) -> SimpleNamespace:
    """The run's task as the activity reader addresses it, with a channel that counts what it handed over."""
    emit = publisher(entry.task_id, target)

    def counted(text: str, **meta: Any) -> None:
        emit(text, **meta)
        handed.append(meta.get("delegated_activity"))
    return SimpleNamespace(task_id=entry.task_id, drive_root=str(root), task_metadata={},
                           task_attempt=target["attempt"], emit_progress_fn=counted)


def journal_refused(entry: Any) -> bool:
    """The engine refused this run's journal for good (``retryable: false``, as after retention): it is not read again."""
    from ouroboros import delegate_activity

    return delegate_activity._memo(entry.task_id, entry.run_id).refused is not None


def journal_listed(gateway: Any) -> Optional[bool]:
    """Whether this engine serves a run's own journal, from what this process already knows: no request
    (the body's observation pass reads the catalog first); ``None`` while nobody asked this engine."""
    from ouroboros.gateways.claudexor_run_events import RUN_EVENTS_OPERATION, route_known

    if gateway is None or not getattr(gateway, "engine_version", "") \
            or not callable(getattr(gateway, "open_run_events", None)):
        return False
    return route_known(gateway, RUN_EVENTS_OPERATION)


@dataclass
class _Cursor:
    """Where one run's journal was shown to; the window view's last timeline (an engine without the stream)."""
    seq: Optional[int] = None
    window: Any = None


class Feed:
    """One body generation's speech and reach duties (the module docstring)."""

    def __init__(self, root: Any) -> None:
        self.root = pathlib.Path(root)
        self.lock = threading.Lock()
        self.held: Dict[str, Any] = {}  # the body's last listing: run id -> ``_Held``
        self.live: Dict[str, Dict[str, Any]] = {}  # the live tasks of the observed and followed runs
        self.following: Dict[str, str] = {}  # run id -> task id of each follow in flight
        self.windows: Dict[str, _Cursor] = {}
        self.outage: Optional[tuple] = None  # the open episode: (its UTC start, its millisecond stamp)
        self.told: set = set()  # tasks told of the open episode
        self.placing: Dict[str, Any] = {}  # run id -> its send in flight
        self.record_lock = threading.Lock()  # one record write at a time (``record``)
        self.recorded: Dict[str, Dict[str, Any]] = {}  # run id -> the record this body wrote or read
        self.answered_at: Dict[str, float] = {}  # run id -> wall time of its latest answered read
        self.swept = False  # whether this body removed the records an earlier one left

    def refresh(self, held: Dict[str, Any]) -> None:
        """The body's latest listing, and the live tasks (with a chat their frames can go to: a worker drops a
        frame without one) of the runs it observes and of those still draining their journal."""
        with self.lock:
            wanted = {run.entry.task_id for run in held.values() if run.observe} | set(self.following.values())
        targets = live_tasks(wanted)
        with self.lock:
            self.held = held
            self.live = {task_id: target for task_id, target in targets.items() if target["chat_id"] is not None}
            for run_id in set(self.windows) - set(held):
                del self.windows[run_id]
            self.placing = {run_id: job for run_id, job in self.placing.items() if run_id in held}
        with self.record_lock:  # the parent lists open runs only: a settled or released run's record goes
            known = set(self.recorded)
            if not self.swept:
                try:
                    known |= {name[:-5] for name in os.listdir(_observation_dir(self.root)) if name.endswith(".json")}
                except OSError:
                    pass
                self.swept = True
            gone = [rid for rid in known if rid not in held or held[rid].entry.settled]
            forget_observations(self.root, gone)
            for run_id in gone:
                self.recorded.pop(run_id, None)
                self.answered_at.pop(run_id, None)

    def record(self, entry: Any, fact: Dict[str, Any], error: Any = None) -> None:
        """Keep what one read saw of a held run for its parent (``read_observation``) when it changed: the run's
        state and question as read, or that the engine stopped answering (the outage class ``reached`` tells the
        owner), dated by when that began. A slow read or a refusal says nothing; nothing is written per read."""
        now = time.time()
        if error is not None and (fact["answered"] or str(getattr(error, "observation_reason", "") or "")
                                  != _UNREACHABLE):
            return
        with self.record_lock:
            last = self.recorded.get(entry.run_id)
            if last is None:
                last = read_observation(self.root, entry.run_id)[0] or {}
            new = {"run_id": entry.run_id, "task_id": entry.task_id, "invocation_id": entry.invocation_id,
                   "run_state": fact["run_state"], "waiting_on_user": fact["waiting_on_user"],
                   "async_questions_only": bool(fact.get("async_questions_only")),
                   "answered": error is None, "reason": "" if error is None else _UNREACHABLE}
            if error is None:
                self.answered_at[entry.run_id] = now
            elif last.get("invocation_id") == entry.invocation_id:  # unanswered: its last answered state stands
                new.update(run_state=str(last.get("run_state") or ""), waiting_on_user=bool(last.get("waiting_on_user")),
                           async_questions_only=bool(last.get("async_questions_only")))
            if all(new[key] == last.get(key) for key in _KEPT):
                self.recorded[entry.run_id] = last
                return
            answered = self.answered_at.get(entry.run_id)
            new.update(since=_iso(now), last_answered_at=_iso(answered) if answered is not None
                       else str(last.get("last_answered_at") or ""))
            try:
                path = observation_path(self.root, entry.run_id)
                path.parent.mkdir(parents=True, exist_ok=True)
                atomic_write_json(path, new)
            except OSError:
                log.debug("Observation record of %s unwritten; the next change writes it", entry.run_id, exc_info=True)
                return
            self.recorded[entry.run_id] = new

    def place(self, runtime: Any, held: Dict[str, Any]) -> None:
        """Derive demand from durable frames on every pass, including after restart or a receipt in another
        process. Send once the engine answers: one job per run, idempotent with the starter's round top."""
        from ouroboros.delegate_message_store import pending_runs, replay_frames

        pending = pending_runs(self.root, held)
        with self.lock:
            due = [run_id for run_id in pending if run_id not in self.placing or self.placing[run_id].done()]
        for run_id in due:
            if runtime.observation(run_id).get("answered"):
                with self.lock:
                    self.placing[run_id] = runtime.observers.submit(replay_frames, self.root, held[run_id].entry)

    def follow(self, runtime: Any, run_id: str) -> None:
        """Read one run's journal beat after beat while the body holds it, observes it and its task is live;
        once its terminal is recorded (or custody let it go), until the journal ends or a beat brings
        nothing. A beat that could not read, or that the engine ended quiet before its time, waits out
        the beat: at most one read per run per beat, never a busy loop."""

        cursor, transport, entry = _Cursor(), [], None
        try:
            while runtime.active:
                run = self.held.get(run_id)
                entry = run.entry if run is not None else entry
                target = self.live.get(entry.task_id) if entry is not None else None
                if target is None:
                    return
                with self.lock:
                    self.following[run_id] = entry.task_id
                began = time.monotonic()
                spoke = self._beat(runtime, entry, target, cursor, transport)
                if spoke == _ENDED or (spoke is False and (run is None or not run.observe)):
                    return
                if not spoke:
                    runtime.stop.wait(_TICK_SEC if spoke is None else max(0.0, began + _TICK_SEC - time.monotonic()))
        finally:
            with self.lock:
                self.following.pop(run_id, None)
            for gateway in transport:
                gateway.close()

    def _beat(self, runtime: Any, entry: Any, target: Dict[str, Any], cursor: _Cursor, transport: list):
        """One journal read from the cursor, published: True spoke, False quiet, ``_ENDED``, None unread. A journal
        the engine refused for good is not read again; what it served is shown first (``emit_stream``), the window
        view shows the rest. Its retryable refusal is a failed read: the next beat asks again."""
        from ouroboros import delegate_activity, delegate_progress
        from ouroboros.execution_domain import run_gateway
        from ouroboros.gateways.claudexor_run_events import read_run_events

        if journal_refused(entry):
            return _ENDED
        handed: list = []
        ctx = _context(self.root, entry, target, handed)
        try:
            if not transport:  # the run's own engine and transport: a held read never waits for another's
                transport.append(run_gateway(entry, runtime._gateway))
            if cursor.seq is None:
                cursor.seq = delegate_activity.shown_cursor(ctx, entry.task_id, entry.run_id)
            read = read_run_events(transport[0], entry.run_id, after_seq=cursor.seq, through_seq=None,
                                   timeout_sec=_TICK_SEC)
        except Exception:
            log.debug("Speech of delegated run %s unread this beat", entry.run_id, exc_info=True)
            while transport:
                transport.pop().close()  # the next beat opens a fresh transport, as the observation pass does
            return None
        delegate_progress.emit_stream(ctx, entry.run_id, read, after_seq=cursor.seq)
        if read.events and not handed:
            return None  # not handed over (the bus is away): the same range is read again
        cursor.seq = max(cursor.seq, read.through_seq)
        return _ENDED if read.ended or journal_refused(entry) else bool(read.events)

    def window(self, entry: Any, gateway: Any, detail: Dict[str, Any]) -> None:
        """An engine without the run's journal, or a journal it refused for good: the new timeline rows of a
        detail the body read are the record, with the run's latest typed actor (the window view the wait
        published); words an unhanded frame left go first (``delegate_activity.window_observations``)."""
        from ouroboros import delegate_activity, delegate_progress

        seq, target = detail.get("lastSeq"), self.live.get(entry.task_id)
        if (target is None or type(seq) is not int
                or journal_listed(gateway) is not False and not journal_refused(entry)):
            return
        handed: list = []
        ctx = _context(self.root, entry, target, handed)
        with self.lock:
            state = self.windows.setdefault(entry.run_id, _Cursor())
        if state.seq is None:
            state.seq = delegate_activity.shown_cursor(ctx, entry.task_id, entry.run_id)
            state.window = delegate_progress.WindowObservations()
            state.window.observe_baseline(detail, state.seq)
        if seq > state.seq:
            delegate_progress.emit(ctx, entry.run_id, state.window.record(detail, seq), detail=detail,
                                   entry=entry, after_seq=state.seq)
            if handed:
                state.seq = seq

    def reached(self, entry: Any, *, answered: bool, error: Any = None) -> None:
        """One observation's word on the engine: an answer closes the open episode, a read no daemon
        answered opens one (or tells a task it has not told yet); a slow read or a refusal says nothing."""
        if answered:
            with self.lock:
                episode, told, self.outage, self.told = self.outage, self.told, None, set()
            for task_id in sorted(told) if episode else ():
                self._line(task_id, "Delegation daemon reachable again; the outage that began at "
                           f"{episode[0]} is over and delegated runs are being observed again.",
                           f"delegation_daemon_recovered:{episode[1]}", "ok")
            return
        if str(getattr(error, "observation_reason", "") or "") != _UNREACHABLE or entry.task_id not in self.live:
            return
        with self.lock:
            if self.outage is None:
                opened = time.time()
                self.outage = (time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(opened)),
                               str(int(opened * 1000)))
            if entry.task_id in self.told:
                return
            self.told.add(entry.task_id)
            episode = self.outage
        self._line(entry.task_id, f"Delegation daemon unreachable since {episode[0]}; delegated runs are not "
                   "being observed, the runs themselves keep going.", f"delegation_daemon_unreachable:{episode[1]}",
                   "warn")

    def _line(self, task_id: str, text: str, key: str, tone: str) -> None:
        """One owner-visible line per episode and task: the typed ``task_incident``/``toast_once`` pair the
        client toasts once, so no host state is written for the dedup."""
        target = self.live.get(task_id)
        if target is None:
            return
        try:
            publisher(task_id, target)(text, incident={"task_incident": "delegation_daemon_unreachable",
                                                       "toast_once": f"{task_id}:{key}", "toast_tone": tone})
        except Exception:
            log.debug("Delegation reach line for %s unavailable", task_id, exc_info=True)


def run_activity(observation: Any) -> Optional[bool]:
    """What one dated observation of a run says of its work (the body observer's,
    ``delegate_feed.current_activity``): True running (``async_questions_only`` questions hold
    nothing), False waiting (queued for capacity, paused on its question, terminal), None
    unanswered or unknown. A running run counts as working: an engine-internal provider wait
    carries no typed fact."""
    from ouroboros.delegate_custody import TERMINAL_STATES

    if not isinstance(observation, dict) or observation.get("answered") is not True:
        return None
    run_state = str(observation.get("run_state") or "")
    if run_state == "running":
        return not observation.get("waiting_on_user") or observation.get("async_questions_only") is True
    return False if run_state == "queued" or run_state in TERMINAL_STATES else None


def current_activity(root: Any, task_id: str, run_id: str, invocation_id: str = "") -> Optional[bool]:
    """Whether a held run works now by the body's latest read: the same run, task and (when given)
    invocation, read within a few ticks; anything else is None (unknown)."""
    from ouroboros.delegate_collection import body_runtime

    runtime = body_runtime(root)
    fact = runtime.observation(run_id) if runtime is not None else {}
    try:
        current = "at" in fact and time.monotonic() - float(fact["at"]) <= _OBSERVATION_CURRENT_SEC
    except (TypeError, ValueError):
        current = False
    if (not current or fact.get("task_id") != str(task_id)
            or (invocation_id and fact.get("invocation_id") != str(invocation_id))):
        return None
    return run_activity(fact)


def announce_working(entry: Any) -> None:
    """A held run works now and did not at its last read (or had none): say so the way a resumed attempt
    does, since its task may be a sleeper whose Telegram typing had stopped (DESIGN §9). Best effort: a
    lost frame costs one missed cue."""
    try:
        from ouroboros.model_sleep import announce_resumed
        from supervisor.workers import get_event_q

        target = live_tasks([entry.task_id]).get(entry.task_id)
        if target is not None:
            announce_resumed(SimpleNamespace(event_queue=get_event_q(create=False), task_id=entry.task_id,
                                             current_chat_id=target["chat_id"]))
    except Exception:
        log.debug("Working-again announce unavailable for %s", entry.run_id, exc_info=True)


def owe_placement(root: Any) -> None:
    """Wake this process's body sooner; any body's next pass discovers the durable pending frames itself."""
    from ouroboros.delegate_collection import body_runtime

    runtime = body_runtime(root)
    if runtime is not None:
        runtime.changed.set()


def _iso(epoch: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(epoch))
