"""Boot phases and wedge detection for the server generation.

The one boot document of the process (``BOOT``: phase, readiness flags, the named boot
steps the coordinator and the supervisor init run; what ``/api/health`` and ``/api/state``
answer as ``boot``), the boot coordinator that runs the lifespan's post-serve steps on its
own thread, and the supervisor generation's wedge detection: the two silent-wedge
predicates (a stalled supervisor loop, a heartbeat-silent in-process chat turn), the owner
alert one of them raises, the measurements the loop publishes about itself (its stall facts
and the ``supervisor_load`` row per busy minute), and the dedicated watchdog thread that
evaluates both outside the loop it watches.
"""

from __future__ import annotations

import contextlib
import os
import posixpath
import queue
import re
import sys
import threading
import time
from typing import Any, Callable, Dict, Iterator, Optional, Sequence, Tuple

from ouroboros.deadline_utils import parse_deadline_ts, utc_now
from ouroboros.runtime_limits import (
    BUDGET_PROJECTION_RETRY_SEC, SUPERVISOR_EVENT_BATCH_MAX_EVENTS, SUPERVISOR_EVENT_BATCH_MAX_SEC,
)
from ouroboros.server_process import DATA_DIR, log, _restart_requested
from ouroboros.utils import append_jsonl, utc_now_iso

# The loop hands the watchdog ONE list: [0] the monotonic stamp the watchdog
# triggers on, [1] the facts published with that stamp, [2] the loop thread's CPU
# base and [3] the worst worker-event lag of the drain in progress. The LOOP
# THREAD owns every write; the watchdog only reads them, so it never takes a
# lock, calls the daemon or touches disk — a watchdog that waits on the thread it
# watches reports nothing. Older/foreign callers may pass the stamp alone.
_STAMP, _FACTS, _CPU, _LAG = 0, 1, 2, 3

# A stall row carries the loop thread's stack at onset: at most this many
# innermost frames, each one line, so the journal row stays bounded.
_STALL_STACK_FRAMES = 12


def _supervisor_loop_stalled(last_tick: float, now: float, deadline_sec: int) -> bool:
    """True when the supervisor loop has not published a liveness tick within the
    deadline (WS3). deadline_sec<=0 disables the watchdog."""
    return deadline_sec > 0 and (now - last_tick) > deadline_sec


def _daemon_pin_matched() -> Optional[bool]:
    """Does the engine this process already PROVED serve the next-spawn pin?

    A generation whose daemon lags the installed pin is one of the suspects behind
    a stalled loop (every ``ensure`` on such a generation pays a probe/install
    path a matched one skips), so the stall row carries the answer the process
    ALREADY holds in memory: the version proven by the last successful handshake
    (``owned_engine_version``, explicitly no new I/O) against the loaded pin.
    ``None`` means unknown — no handshake has succeeded here yet, or this install
    carries no pin — never a guess.
    """
    try:
        from ouroboros.claudexor_daemon import owned_engine_version
        from ouroboros.claudexor_runtime import get_runtime_manager

        pin = get_runtime_manager().pin
        proven = owned_engine_version()
        return proven == pin.version if (proven and pin is not None) else None
    except Exception:
        log.debug("owned-daemon pin match is unknown", exc_info=True)
        return None


def note_supervisor_ready() -> None:
    """Record one durable boot fact, never a gate: this process reached readiness
    and how old it was (``process_age_sec``; on an ordinary start that is the
    whole boot, on an in-process revival the age of the surviving process).
    The lifecycle jobs record their own durations beside it
    (``usage_store_migration``, ``startup_migration``)."""
    from ouroboros.server_process import PROCESS_STARTED_MONOTONIC
    from ouroboros.utils import append_jsonl

    log.info("Supervisor ready.")
    try:
        append_jsonl(DATA_DIR / "logs" / "supervisor.jsonl", {
            "ts": utc_now_iso(), "type": "supervisor_ready",
            "process_age_sec": round(time.monotonic() - PROCESS_STARTED_MONOTONIC, 3)})
    except Exception:
        log.debug("supervisor_ready fact not recorded", exc_info=True)


def run_startup_phase(liveness: list, phase: str, step: Callable[[], Any]) -> Any:
    """Run one startup step under its own visible sub-phase (a stall inside it is
    journaled with that name, and the boot document shows it as a step), then publish
    ``startup`` again."""
    liveness[1], liveness[0] = loop_phase_facts(liveness, phase), time.monotonic()
    with BOOT.step(phase):
        result = step()
    liveness[1], liveness[0] = loop_phase_facts(liveness, "startup"), time.monotonic()
    return result


def loop_phase_facts(liveness: list, phase: str, *, new_tick: bool = False) -> dict:
    """The measurements the LOOP THREAD publishes with its own liveness stamp.

    ``phase`` is the coarse tick phase the loop is entering — ``events`` |
    ``maintenance`` | ``assign``, one stamp per phase and never per sub-step — so a
    stall names where the thread went silent instead of only how long it was.
    ``loop_thread_cpu_sec`` is the ``time.thread_time()`` delta over the interval
    that ENDS with this stamp, sampled on the loop thread itself, and
    ``cpu_interval_sec`` is that interval's wall (monotonic) length — the number
    is honest only beside the interval it covers, and at stall onset that
    interval is the last HEALTHY phase, not the stall. ``loop_thread_cpu_total_sec``
    is the thread's cumulative CPU at this stamp, so the watchdog can charge a
    whole stall (onset stamp to recovery stamp) without seeing every stamp in
    between. Read beside the wall gap they separate a thread that BURNED it from
    one blocked on a lock or starved of the GIL. ``max_event_lag_sec`` is the worst worker-stamped lag of
    the most recently completed drain, absent when no drained event carried a
    worker stamp. ``new_tick`` opens a fresh drain maximum (the events phase opens
    the tick), so a lag can never outlive the tick that observed it. No
    registered-project count rides here: the set the retire sweep walks exists
    only as a full custody-log replay, and a measurement may not pay disk on the
    thread it measures — an absent key beats a cheap-looking wrong one.
    """
    cpu = time.thread_time()
    facts = {
        "phase": phase,
        "loop_thread_cpu_sec": round(cpu - liveness[_CPU], 3),
        "cpu_interval_sec": round(max(0.0, time.monotonic() - liveness[_STAMP]), 3),
        "loop_thread_cpu_total_sec": round(cpu, 6),
        "daemon_pin_matched": _daemon_pin_matched(),
    }
    if liveness[_LAG] is not None:
        facts["max_event_lag_sec"] = round(liveness[_LAG], 1)
    liveness[_CPU] = cpu
    if new_tick:
        liveness[_LAG] = None
    return facts


def observe_worker_event_lag(liveness: list, evt: Any) -> Optional[float]:
    """Record and return how far behind the loop is on the worker event it is draining now.

    The worker stamps ``ts`` in its OWN process when it queues the event, so this
    is a cross-process wall-clock gap — a measurement, never the watchdog's
    monotonic trigger. An event without a parsable worker stamp is skipped: the
    host's own clock is not evidence about when a worker spoke (None).
    """
    stamped = parse_deadline_ts(evt.get("ts")) if isinstance(evt, dict) else None
    if stamped is None:
        return None
    lag = (utc_now() - stamped).total_seconds()
    if liveness[_LAG] is None or lag > liveness[_LAG]:
        liveness[_LAG] = lag
    return lag


def drain_worker_events(event_q: Any, ctx: Any, liveness: list, *, on_restart: Callable[..., Any]) -> bool:
    """One BOUNDED events pass of the supervisor loop: at most
    ``SUPERVISOR_EVENT_BATCH_MAX_EVENTS`` events; the ``SUPERVISOR_EVENT_BATCH_MAX_SEC``
    budget is checked between handlers (a handler already running finishes, so one
    slow handler can overrun it), then the loop runs bridge intake. The bus is one
    object with two lanes (``supervisor/event_observation.py``): every ``get`` serves
    the oldest control event while any exists, else the oldest telemetry frame, FIFO
    inside each lane — the priority lives in the queue, so this ONE consumer takes
    whatever comes out, never sorts, defers or re-puts. A ``restart_request`` goes to
    ``on_restart``; every other event is lag-observed and dispatched. The remainder
    stays queued for the next turn, so a producer that keeps the queue non-empty
    can never starve owner-message intake. A dequeued event carrying a
    ``delivery_id`` is released from the bus's outstanding-delivery membership after
    its handler returned or raised, whichever path it took. A live final answer is
    stamped as dequeued the moment it is taken (``finalization_timing``), and every
    event's service time, take to handler return, feeds ``supervisor.loop_load.CONSUMER``
    by type, with its lag by lane. Returns True
    when the pass stopped at its bound (a backlog may remain: the caller skips its idle
    sleep and polls intake without its wait)."""
    from ouroboros.finalization_timing import stamp_finalization_dequeue
    from supervisor import loop_load
    from supervisor.event_observation import event_lane
    from supervisor.events import dispatch_event

    deadline = time.monotonic() + SUPERVISOR_EVENT_BATCH_MAX_SEC
    drained = 0
    while drained < SUPERVISOR_EVENT_BATCH_MAX_EVENTS and time.monotonic() < deadline:
        try:
            evt = event_q.get_nowait()
        except queue.Empty:
            return False
        drained += 1
        taken, lag = time.monotonic(), None
        try:
            stamp_finalization_dequeue(evt)
            if evt.get("type") == "restart_request":
                on_restart(evt, ctx)
                continue
            lag = observe_worker_event_lag(liveness, evt)
            dispatch_event(evt, ctx)
        finally:
            served = time.monotonic() - taken
            _release_delivery(event_q, evt)
            kind = str(evt.get("type") or "") if isinstance(evt, dict) else ""
            loop_load.CONSUMER.note(kind, event_lane(evt), served, lag)
    return True


# The supervisor_load row's window: a reporting cadence, never a gate or a deadline.
SUPERVISOR_LOAD_WINDOW_SEC = 60.0
_LOAD_FAILURES_SEEN: set = set()


def _load_sections() -> Dict[str, Callable[[], Dict[str, Any]]]:
    """The row's further sections: a name and a zero-argument drain of its own window, one line each
    (``dialogue_retry`` answers its process-lifetime counters and current occupancy instead)."""
    from ouroboros import dialogue_retry, store_timing
    from supervisor import queue

    return {
        "queue_lock": lambda: queue._queue_lock.drain(),
        "store_locks": store_timing.drain,
        "dialogue_retry": dialogue_retry.snapshot,
    }


def publish_supervisor_load(event_q: Any, drive_root: Any) -> Optional[dict]:
    """At most one ``supervisor_load`` row per window into ``logs/supervisor.jsonl``.

    The loop thread calls this once per tick at its top level (outside Q and any store
    transaction); it returns None while the window is open or while the consumer has served
    no event in it (an idle stretch folds into the next row, whose ``window_sec`` says how
    long, so an idle install adds no rows), else the row: ``window_sec``;
    ``handlers`` (event type -> count, total_sec, max_sec of service time, take to handler
    return); ``lanes`` (control/telemetry -> ``depth`` and ``oldest_age_sec`` of the bus at
    this moment, None when the bus cannot say, and ``max_lag_sec``, the worst producer-stamp
    lag the drain saw); ``max_event_lag_sec`` over both lanes; then one key per
    ``_load_sections`` entry (Q: outermost waits, holds, longest hold's site), a failing
    source answering ``{"error": <type>}``. The append never waits on the jsonl lock and a
    failure is logged once per class: diagnostics never cost the loop or a delivery. The boot
    document keeps the row's ``ts`` and ``max_event_lag_sec`` for the health answers.
    """
    from supervisor import loop_load
    from supervisor.event_observation import lane_observations
    from supervisor.event_taxonomy import LANES

    now = time.monotonic()
    if now - loop_load.CONSUMER.started < SUPERVISOR_LOAD_WINDOW_SEC or not loop_load.CONSUMER.served():
        return None
    try:
        window = loop_load.CONSUMER.drain(now)
        lanes, lag = lane_observations(event_q) or {}, window["lag"]
        row = {"ts": utc_now_iso(), "type": "supervisor_load", "window_sec": window["window_sec"],
               "handlers": window["handlers"],
               "lanes": {lane: {"depth": (lanes.get(lane) or {}).get("depth"),
                                "oldest_age_sec": (lanes.get(lane) or {}).get("oldest_age_sec"),
                                "max_lag_sec": lag.get(lane)} for lane in LANES},
               "max_event_lag_sec": max(lag.values()) if lag else None}
        for name, drain in _load_sections().items():
            try:
                row[name] = drain()
            except Exception as exc:
                row[name] = {"error": type(exc).__name__}
        BOOT.note_supervisor_load({"ts": row["ts"], "max_event_lag_sec": row["max_event_lag_sec"]})
        append_jsonl(drive_root / "logs" / "supervisor.jsonl", row, lock_timeout_sec=0)
        return row
    except Exception as exc:
        if type(exc).__name__ not in _LOAD_FAILURES_SEEN:
            _LOAD_FAILURES_SEEN.add(type(exc).__name__)
            log.warning("supervisor_load row not recorded (%s)", type(exc).__name__, exc_info=True)
        return None


_RELEASE_FAILURES_SEEN: set = set()


def _release_delivery(event_q: Any, evt: Any) -> None:
    """The consumer's receipt for one dequeued send: fail-soft, logged once per failure class.

    A bus without the membership (a plain queue in a test, an older proxy) or a
    broken manager connection must not turn a handled event into a loop error;
    the replay duty reads an unobservable membership as unknown anyway."""
    delivery_id = str(evt.get("delivery_id") or "") if isinstance(evt, dict) else ""
    if not delivery_id:
        return
    try:
        event_q.release_delivery(delivery_id)
    except Exception as exc:
        failure_class = type(exc).__name__
        if failure_class not in _RELEASE_FAILURES_SEEN:
            _RELEASE_FAILURES_SEEN.add(failure_class)
            log.warning("Delivery membership release failed (%s); replay reads it as unknown",
                        failure_class, exc_info=True)


def flush_budget_projection(ctx: Any) -> None:
    """One compatibility budget-projection write per loop turn, AFTER bridge intake.

    ``llm_usage`` events only mark ``ctx.budget_projection_dirty``; this is the one
    place that pays the ledger render and the STATE_LOCK write. The flag clears only
    when the writer returns True; a False (unknown or stale ledger marker) or an
    exception keeps it dirty and the next attempt waits ``BUDGET_PROJECTION_RETRY_SEC``,
    logged once per attempt, so a frozen-marker install does not render every turn."""
    if not getattr(ctx, "budget_projection_dirty", False):
        return
    now = time.monotonic()
    if now < float(getattr(ctx, "budget_projection_retry_at", 0.0) or 0.0):
        return
    try:
        written = ctx.update_budget_from_usage({}) is not False
    except Exception:
        written = False
        log.error("Compatibility budget projection update failed; retrying in %.0fs",
                  BUDGET_PROJECTION_RETRY_SEC, exc_info=True)
    else:
        if not written:
            log.warning("Compatibility budget projection not written (ledger marker unknown or stale); "
                        "retrying in %.0fs", BUDGET_PROJECTION_RETRY_SEC)
    ctx.budget_projection_dirty = not written
    ctx.budget_projection_retry_at = 0.0 if written else now + BUDGET_PROJECTION_RETRY_SEC


def _published_loop_facts(liveness: list) -> dict:
    """The facts the loop published with its last stamp ({} when it published none)."""
    facts = liveness[_FACTS] if len(liveness) > _FACTS else None
    return dict(facts) if isinstance(facts, dict) else {}


def _loop_thread_stack(ident: Optional[int], *, limit: int = _STALL_STACK_FRAMES,
                       facts: Optional[dict] = None) -> list[str]:
    """The loop thread's CURRENT stack, outermost first, at most ``limit`` innermost
    ``path:line in func`` frames (repository files relative, other files absolute, a pseudo-filename
    such as ``<string>`` as the interpreter names it). Walk code
    objects instead of traceback.extract_stack: linecache could read a stalled filesystem on
    this watchdog thread; no locals, no message bodies. A cut walk sets
    ``facts["loop_stack_truncated"]`` when ``facts`` is given, so a bounded stack is never
    mistaken for the whole one. No lock, disk or cooperation from the loop; [] if its
    stack is unavailable."""
    if not ident or limit <= 0:
        return []
    try:
        frame = sys._current_frames().get(ident)
        summaries = []
        while frame is not None and len(summaries) < limit:
            code = frame.f_code
            summaries.append((code.co_filename, frame.f_lineno, code.co_name))
            frame = frame.f_back
        if frame is not None and facts is not None:
            facts["loop_stack_truncated"] = True
    except Exception:
        log.debug("loop-thread stack capture failed", exc_info=True)
        return []
    finally:
        frame = None  # never keep a foreign frame alive past this call
    try:
        from ouroboros.config import REPO_DIR

        root = str(REPO_DIR).rstrip("/\\") + "/"
    except Exception:
        root = ""
    rows = []
    for filename, lineno, name in reversed(summaries):
        path = filename.replace("\\", "/")
        if root and path.startswith(root.replace("\\", "/")):
            path = path[len(root):]
        rows.append(f"{path[:200]}:{lineno} in {name[:100]}")
    return rows


_ABSOLUTE_PATH = re.compile(r"^([A-Za-z]:)?[/\\]")
# One ``_loop_thread_stack`` row: ``path:line in func``. The path is everything before the LAST
# ``:<line> in `` (a Windows drive ``C:`` or a colon inside the path is part of the path).
_STACK_ROW = re.compile(r"^(?P<path>.*):(?P<line>\d+) in (?P<func>.*)$")


def _runtime_roots() -> tuple[str, ...]:
    """Where the Python runtime lives: ``sys.prefix`` and ``sys.base_prefix``, each as reported
    and as resolved, lexically normalized and ending in one ``/``. A relocatable interpreter may
    report its prefix through ``bin/..`` while loading modules from the normalized path, and
    Homebrew reports its ``opt`` symlink while loading the standard library from the Cellar it
    resolves to; the closing ``/`` keeps a sibling ``<prefix>-other`` outside."""
    roots = set()
    for prefix in filter(None, (sys.prefix, sys.base_prefix)):
        spellings = [prefix]
        with contextlib.suppress(OSError, ValueError):  # unresolvable: compared as reported; the watchdog lives on
            spellings.append(os.path.realpath(prefix))
        roots.update(posixpath.normpath(path.replace("\\", "/")).rstrip("/") + "/" for path in spellings)
    return tuple(roots)


def _innermost_repo_frame(stack: list[str]) -> str:
    """``path:function`` of the innermost frame inside the repository - a relative path, else
    the innermost frame outside the Python runtime (``_runtime_roots``; '' when none) - without
    the line number, so the samples of one stalled function fold into one row. A pseudo-filename
    (``<frozen importlib._bootstrap>``, ``<string>``) names no file and is neither, so an import-lock
    wait or exec'd code names its real caller. The path keeps the row's own spelling; only the
    comparison is normalized."""
    runtime = _runtime_roots()
    outside = ""
    for row in reversed(stack):
        parsed = _STACK_ROW.match(row)
        path, func = (parsed.group("path"), parsed.group("func")) if parsed else ("", "")
        if not path or (path.startswith("<") and path.endswith(">") and "/" not in path and "\\" not in path):
            continue
        key = f"{path}:{func}"
        if not _ABSOLUTE_PATH.match(path):
            return key
        if not outside and not (posixpath.normpath(path.replace("\\", "/")) + "/").startswith(runtime):
            outside = key
    return outside


def _top_frames(samples: dict) -> list:
    """Up to five innermost repository frames, the most sampled first."""
    return [{"frame": frame, "samples": count} for frame, count in
            sorted(samples.items(), key=lambda item: (-item[1], item[0]))[:5]]


def _fold_sample(episode: dict, stack: list) -> None:
    """One stack sample into an open episode: counted by its innermost repository frame, the latest kept."""
    if stack:
        episode["last_stack"] = stack
    key = _innermost_repo_frame(stack) or "(outside the repository)"
    episode["samples"][key] = episode["samples"].get(key, 0) + 1


class _TurnPhaseWatch:
    """Where a direct turn's own thread stands while its start phase does not advance.

    Each watchdog interval reads every running ``TurnExecution`` (its thread and the start
    facts it passed, each with the monotonic instant it was stamped). One whose phase has
    stood a whole interval before its Main call started gets one ``direct_turn_stall``
    row with that thread's stack, one folded stack sample per interval while the phase
    stands, and one ``direct_turn_stall_end`` once it advances (``then``: the next phase,
    ``stood_sec`` up to that phase's own stamp), the turn ends in it (``then: "ended"``, up
    to its end) or this watchdog stops (``then: "unwatched"``; a next one opens its own).
    Heartbeat age decides nothing: a turn blocked before its heartbeat starts and one
    blocked under a live heartbeat are watched alike, and the onset row only reports the
    heartbeat's age (null before it started). Journal only, as invariant 20 binds the
    watchdog: no owner message, no checkpoint row, no cancel, no verdict."""

    def __init__(self, interval: float) -> None:
        self._interval = interval
        self._open: dict = {}  # execution_id -> the open episode

    def tick(self, now: float, heartbeats: dict) -> None:
        from supervisor.active_activity import CALL_PHASES, get_direct_activity_registry

        executions = {row.execution_id: row for row in get_direct_activity_registry().executions()}
        for execution_id in [key for key in self._open if key not in executions]:
            self._close(execution_id, now)
        for execution_id, execution in executions.items():
            index = len(execution.marks) - 1
            phase, at = execution.marks[index]
            episode = self._open.get(execution_id)
            if episode is not None and episode["index"] != index:
                self._close(execution_id, now)
                episode = None
            if phase in CALL_PHASES or now - at < self._interval:
                continue
            ident = execution.thread[0]
            if episode is not None:
                _fold_sample(episode, _loop_thread_stack(ident))
                continue
            flags: dict = {}
            stack = _loop_thread_stack(ident, facts=flags)
            beat = heartbeats.get(execution.task_id) if execution.task_id else None
            self._journal({"type": "direct_turn_stall", **self._ids(execution), "thread": execution.thread[1],
                           "phase": phase, "stood_sec": round(now - at, 1),
                           "heartbeat_sec": round(now - beat, 1) if isinstance(beat, (int, float)) else None,
                           **({"stack_truncated": True} if flags else {}), **({"stack": stack} if stack else {})})
            episode = self._open[execution_id] = {"execution": execution, "index": index,
                                                  "samples": {}, "last_stack": []}
            _fold_sample(episode, stack)

    def close(self, now: float) -> None:
        """This watchdog stops: every episode it opened ends with its own row."""
        for execution_id in list(self._open):
            self._close(execution_id, now)

    def _close(self, execution_id: str, now: float) -> None:
        episode = self._open.pop(execution_id)
        execution, index = episode["execution"], episode["index"]
        marks = list(execution.marks)
        phase, at = marks[index]
        if index + 1 < len(marks):
            then, until = marks[index + 1]
        elif execution.ended_at is not None:
            then, until = "ended", execution.ended_at
        else:
            then, until = "unwatched", now
        self._journal({"type": "direct_turn_stall_end", **self._ids(execution), "phase": phase, "then": then,
                       "stood_sec": round(until - at, 1), "samples": sum(episode["samples"].values()),
                       "top_frames": _top_frames(episode["samples"]), "last_stack": episode["last_stack"]})

    @staticmethod
    def _ids(execution) -> dict:
        return {"execution_id": execution.execution_id, "task_id": execution.task_id,
                "chat_id": execution.chat_id, "client_message_id": execution.client_message_id}

    @staticmethod
    def _journal(row: dict) -> None:
        from supervisor.state import append_jsonl
        try:
            append_jsonl(DATA_DIR / "logs" / "supervisor.jsonl", {"ts": utc_now_iso(), **row})
        except Exception:
            log.debug("%s log failed", row.get("type"), exc_info=True)


def _stall_cpu_over(onset_facts: dict, latest_facts: dict) -> Optional[float]:
    """The loop thread's CPU from the stamp it went silent on to its latest stamp
    — the whole stall, however many stamps the recovery published before the
    watchdog looked — from the cumulative totals; the latest stamp's own delta
    when a foreign/older stamp published no total (never invented)."""
    onset_total, latest_total = onset_facts.get("loop_thread_cpu_total_sec"), latest_facts.get("loop_thread_cpu_total_sec")
    if isinstance(onset_total, (int, float)) and isinstance(latest_total, (int, float)):
        return round(max(0.0, latest_total - onset_total), 3)
    return latest_facts.get("loop_thread_cpu_sec")


def _chat_turn_wedged(busy: bool, last_activity_ts, now: float, deadline_sec: int) -> bool:
    """True when an IN-PROCESS direct-chat turn is busy but its liveness tick has been
    silent past the deadline (WS3). ``last_activity_ts is None`` => the turn has not
    started its liveness loop yet (not wedged). deadline_sec<=0 disables the check."""
    if not busy or last_activity_ts is None or deadline_sec <= 0:
        return False
    return (now - last_activity_ts) > deadline_sec


def _alert_chat_turn_wedge(task_id, gap: float) -> None:
    """WS3: a direct-chat turn is heartbeat-silent. New messages still get answered
    on independent native actors, but a hung IN-PROCESS turn cannot be killed
    independently of the supervisor process. Surface it +
    recommend /restart, which is the safe full recovery."""
    from supervisor.state import append_jsonl, load_state
    try:
        append_jsonl(DATA_DIR / "logs" / "supervisor.jsonl", {
            "ts": utc_now_iso(), "type": "chat_turn_wedge",
            "task_id": str(task_id or ""), "silent_sec": round(gap, 1),
        })
    except Exception:
        log.debug("chat-turn wedge log failed", exc_info=True)
    try:
        owner_chat = int((load_state() or {}).get("owner_chat_id") or 0)
        if owner_chat:
            from supervisor.message_bus import send_with_budget
            send_with_budget(
                owner_chat,
                f"⚠️ A chat turn looks wedged (~{int(gap)}s with no heartbeat). New messages "
                "still get answered, but the stuck turn can't be cleared in-process — /restart "
                "to fully recover it.",
                is_progress=True,
                task_id=str(task_id or ""),
                progress_meta={
                    "task_incident": "chat_turn_wedge",
                    "toast_once": f"{task_id or 'direct-chat'}:chat_turn_wedge",
                },
                role="system", system_type="runtime_liveness_notice")
    except Exception:
        log.debug("chat-turn wedge owner alert failed", exc_info=True)


def _start_supervisor_liveness_watchdog(
    liveness: list, stop_event=None, *, loop_thread_ident: Optional[int] = None,
) -> None:
    """Dedicated daemon thread (NOT inside the supervisor loop, so it fires even when
    that loop stalls). It observes silent-wedge classes and reports them
    DIFFERENTLY (owner decision 4C). A heartbeat-silent in-process direct-chat turn
    ALERTS the owner, because /restart is a recovery they can perform. A supervisor
    loop stall (new-message intake starvation) is JOURNAL ONLY: the ``log.error``
    and one durable ``supervisor_loop_stall`` row with the phase facts the loop
    published with its last stamp plus the loop thread's stack at onset
    (``stack``, bounded; ``loop_stack_truncated`` when cut), then one bounded stack
    sample per watchdog interval while the stall is open, closed once the loop ticks
    again by one ``supervisor_loop_stall_end`` charging the thread's CPU over the
    whole stall and carrying ``samples``, up to five ``top_frames`` (innermost
    repository frame -> samples) and the ``last_stack`` — where it spent the stall,
    not only where it began. Onset without an end is a generation that never
    recovered. Nothing reaches the
    owner's chat from that half: a stall they cannot act on is an alarm, not
    information, and the rows carry the diagnosis anyway. A direct turn whose start
    phase stops advancing before its first model request is journal only the same way,
    with its own thread's stack, whatever its heartbeat says (``_TurnPhaseWatch``). It deliberately does NOT
    kill a hung thread; independent native actors keep the chat responsive
    meanwhile. ``loop_thread_ident`` is the thread whose stack a stall row carries;
    it defaults to the CALLER, because the loop thread starts its own watchdog
    (startup phase included). ``stop_event`` is
    a PER-GENERATION token: when the supervisor loop that owns ``liveness`` exits (incl.
    the crash-storm death path, which never sets the global restart flag), it is set so
    this watchdog stops watching a now-stale liveness list (no false post-revival alert)."""
    from ouroboros.config import get_supervisor_liveness_deadline_sec

    deadline = get_supervisor_liveness_deadline_sec()
    if deadline <= 0:
        return
    watched_ident = loop_thread_ident if loop_thread_ident is not None else threading.get_ident()

    def _watch() -> None:
        from supervisor.state import append_jsonl
        interval = min(15, max(1, deadline // 3))
        loop_alerted = False
        stall_onset: tuple = ()  # (stalled stamp, phase, onset facts) of the OPEN alerted stall
        stall: dict = {"samples": {}, "last_stack": []}  # its stack samples while the stall is open
        wedged_tasks: set[str] = set()
        turn_phases = _TurnPhaseWatch(interval)
        while not _restart_requested.is_set() and not (stop_event is not None and stop_event.is_set()):
            time.sleep(interval)
            # ONE clock: both halves measure an ELAPSED GAP against stamps taken on
            # the monotonic clock (the loop-liveness tick here, and the chat-turn
            # heartbeat in agent.py), so a wall-clock jump — NTP step, DST/timezone
            # change, manual set, VM resume — can neither fabricate a stall/wedge
            # nor mask a real one on either half.
            now = time.monotonic()
            # (1) Supervisor loop stall — new-message intake starvation.
            if _supervisor_loop_stalled(liveness[_STAMP], now, deadline):
                if not loop_alerted:
                    gap = now - liveness[_STAMP]
                    facts = _published_loop_facts(liveness)
                    flags: dict = {}
                    stack = _loop_thread_stack(watched_ident, facts=flags)
                    # A startup stall is a generation that has not finished initializing: no
                    # native chat answers for it, so the line says only what is true.
                    starved = ("startup has not finished" if str(facts.get("phase") or "").startswith("startup")
                               else "new-message intake starved (native chat still answers)")
                    log.error("Supervisor loop STALLED ~%.0fs in phase %s — %s; loop thread is at: %s",
                              gap, facts.get("phase"), starved, stack[-1] if stack else "(stack unavailable)")
                    try:
                        # The facts the loop published with the stamp it went silent
                        # on: where it was, what its own thread burned over the
                        # interval BEFORE the stall (cpu_interval_sec says how long
                        # that was), how far behind the drained worker events
                        # already were — and where its thread stands right now.
                        append_jsonl(DATA_DIR / "logs" / "supervisor.jsonl", {
                            "ts": utc_now_iso(), "type": "supervisor_loop_stall",
                            "stalled_sec": round(gap, 1), **facts, **flags,
                            **({"stack": stack} if stack else {}),
                        })
                    except Exception:
                        log.debug("loop-stall log failed", exc_info=True)
                    loop_alerted = True
                    stall_onset = (liveness[_STAMP], facts.get("phase"), facts)
                    stall = {"samples": {}, "last_stack": []}
                    _fold_sample(stall, stack)
                else:
                    _fold_sample(stall, _loop_thread_stack(watched_ident))  # one bounded sample per interval
            else:
                if loop_alerted:
                    # The loop ticked again: close the episode ONCE, and only one
                    # that was alerted. Both ends are stamps the LOOP published on
                    # the monotonic clock, so the duration survives a wall-clock
                    # jump; it rounds up by at most one watchdog interval, the
                    # resolution at which recovery is observed at all.
                    try:
                        latest = _published_loop_facts(liveness)
                        stalled = round(liveness[_STAMP] - stall_onset[0], 1)
                        append_jsonl(DATA_DIR / "logs" / "supervisor.jsonl", {
                            "ts": utc_now_iso(), "type": "supervisor_loop_stall_end",
                            "stalled_sec": stalled,
                            "phase": stall_onset[1],
                            # The thread's CPU from the stamp it went silent on to its
                            # latest stamp — the stalled interval itself, whole, however
                            # many phases the recovery published before this look —
                            # beside the wall interval it covers (the same seconds as
                            # stalled_sec): a thread that burned them versus one that
                            # waited on a lock, IO or the GIL.
                            "loop_thread_cpu_sec": _stall_cpu_over(stall_onset[2], latest),
                            "cpu_interval_sec": stalled,
                            # Where the thread SPENT the stall: one stack sample per watchdog
                            # interval, folded by innermost repository frame, plus the last one.
                            "samples": sum(stall["samples"].values()),
                            "top_frames": _top_frames(stall["samples"]),
                            "last_stack": stall["last_stack"],
                        })
                    except Exception:
                        log.debug("loop-stall-end log failed", exc_info=True)
                loop_alerted = False
            # (2) Each native actor has its own liveness and alert identity.
            try:
                from supervisor.workers import chat_turn_liveness
                turns = chat_turn_liveness()
            except Exception:
                turns = []
            live_ids = {task_id for task_id, _ in turns}
            wedged_tasks.intersection_update(live_ids)
            for turn_task, turn_ts in turns:
                if _chat_turn_wedged(True, turn_ts, now, deadline) and turn_task not in wedged_tasks:
                    _alert_chat_turn_wedge(turn_task, now - turn_ts)
                    wedged_tasks.add(turn_task)
            # (3) Each direct turn's start phase: where its own thread stands while the phase holds.
            try:
                turn_phases.tick(now, dict(turns))
            except Exception:
                log.debug("direct-turn phase watch failed", exc_info=True)
        turn_phases.close(time.monotonic())

    threading.Thread(target=_watch, name="supervisor-liveness-watchdog", daemon=True).start()


# ---------------------------------------------------------------------------
# the boot document and the boot coordinator
# ---------------------------------------------------------------------------

PHASES = ("starting", "serving", "ready", "failed")
STEP_STATES = ("running", "done", "failed")


class BootDocument:
    """The process's one boot document, the ``boot`` object of ``/api/health``.

    ``phase``: ``starting`` until the HTTP socket serves (the lifespan reached its yield),
    ``serving`` while the boot steps and the supervisor init run, ``ready`` once every boot
    step ended and the supervisor reached readiness, ``failed`` when the generation cannot
    become ready (the store did not publish, the supervisor init failed, the loop died).
    ``dialogue_ready``: an owner message is accepted durably (the chat bridge is published).
    ``tasks_ready``: pooled task starts are admitted (the supervisor's readiness latch, read
    from the caller at snapshot time so one latch serves the API and this document).
    ``supervisor_load``: the latest load row's ``ts`` and ``max_event_lag_sec``
    (None before this generation's first row). ``steps``: every named boot step in the order
    it started, with its outcome. A failed
    fail-soft step (an extension that did not load) is reported as such and does not fail the
    phase; a step that ends the boot calls ``fail``. The document is per generation
    (``reset`` at every lifespan start) and lives in memory: durable boot facts stay where
    they are written (``supervisor.jsonl`` rows, the ``job:`` rows of the store).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self._since = utc_now_iso()
            self._serving = False
            self._steps_done = False
            self._dialogue_ready = False
            self._error: Optional[str] = None
            self._steps: Dict[str, Dict[str, Any]] = {}
            self._load: Optional[Dict[str, Any]] = None

    def serving(self) -> None:
        with self._lock:
            self._serving = True

    def steps_done(self) -> None:
        with self._lock:
            self._steps_done = True

    def dialogue_ready(self, value: bool = True) -> None:
        with self._lock:
            self._dialogue_ready = bool(value)

    def fail(self, error: str) -> None:
        """The boot cannot reach readiness; the first cause is kept."""
        with self._lock:
            if not self._error:
                self._error = str(error)

    def note_supervisor_load(self, headline: Dict[str, Any]) -> None:
        with self._lock:
            self._load = dict(headline)

    def step_started(self, name: str) -> None:
        with self._lock:
            self._steps[name] = {"name": name, "state": "running", "started_at": utc_now_iso(),
                                 "ended_at": None, "error": None}

    def step_ended(self, name: str, error: Optional[str] = None) -> None:
        with self._lock:
            row = self._steps.get(name)
            if row is None:
                row = self._steps[name] = {"name": name, "started_at": utc_now_iso()}
            row.update(state="failed" if error else "done", ended_at=utc_now_iso(), error=error or None)

    @contextlib.contextmanager
    def step(self, name: str) -> Iterator[None]:
        """Record one step around a block; an exception marks it failed and propagates."""
        self.step_started(name)
        try:
            yield
        except BaseException as exc:
            self.step_ended(name, error=f"{type(exc).__name__}: {exc}")
            raise
        else:
            self.step_ended(name)

    def phase(self, *, supervisor_done: bool = False, supervisor_error: Optional[str] = None) -> str:
        with self._lock:
            if self._error or supervisor_error:
                return "failed"
            if not self._serving:
                return "starting"
            return "ready" if self._steps_done and supervisor_done else "serving"

    def snapshot(self, root: Any = None, *, tasks_ready: bool = False, supervisor_done: bool = False,
                 supervisor_error: Optional[str] = None, jobs: bool = True) -> Dict[str, Any]:
        """The ``boot`` object; ``jobs`` are the store's named-job rows (``{}`` while the store
        is unpublished or absent: a read that never imports and never waits)."""
        phase = self.phase(supervisor_done=supervisor_done, supervisor_error=supervisor_error)
        with self._lock:
            document = {
                "phase": phase,
                "since": self._since,
                "dialogue_ready": self._dialogue_ready,
                "tasks_ready": bool(tasks_ready),
                "steps": [dict(row) for row in self._steps.values()],
                "jobs": {},
                "supervisor_load": dict(self._load) if self._load is not None else None,
                "error": self._error or supervisor_error or None,
            }
        if jobs and root is not None:
            document["jobs"] = job_facts(root)
        return document


BOOT = BootDocument()


def job_facts(root: Any) -> Dict[str, Dict[str, Any]]:
    """``{name: {state, done, total}}`` of the store's named jobs for the health document:
    one indexed read without waiting; an unpublished, absent or busy store reads ``{}``."""
    from ouroboros import maintenance_jobs, store

    try:
        with store.acquisition_budget(0.0):
            rows = maintenance_jobs.jobs(root)
    except Exception:
        return {}
    return {name: {"state": row.get("state"), "done": row.get("done"), "total": row.get("total")}
            for name, row in rows.items()}


def boot_document(root: Any, app_state: Any = None, *, jobs: bool = True) -> Dict[str, Any]:
    """The ``boot`` object for a gateway answer: this process's document plus the supervisor
    facts the server attached to the app (its readiness and init-outcome latches, its error)."""
    ready = getattr(app_state, "supervisor_ready_event", None)
    done = getattr(app_state, "supervisor_init_done_event", None)
    error_fn = getattr(app_state, "get_supervisor_error", None)
    return BOOT.snapshot(
        root, jobs=jobs,
        tasks_ready=bool(ready is not None and ready.is_set()),
        supervisor_done=bool(done is not None and done.is_set()),
        supervisor_error=error_fn() if callable(error_fn) else None,
    )


def boot_notice(app_state: Any = None) -> str:
    """Plain words for a chat frame that found no bridge: why the message was not saved."""
    document = boot_document(None, app_state, jobs=False)
    if document["phase"] == "failed":
        return ("⚠️ Startup failed before the chat could accept messages, so this message was not saved. "
                f"Restart Ouroboros and send it again. ({document['error']})")
    if document["phase"] == "ready":
        return "⚠️ The chat is not accepting messages right now; this message was not saved. Send it again in a moment."
    return ("⚠️ The data store is still being prepared (a migration is running), so this message was not saved. "
            "Send it again once the chat is ready.")


BootStep = Tuple[str, Callable[[], Any]]


def run_boot_steps(steps: Sequence[BootStep], *, stop: threading.Event, document: BootDocument = BOOT) -> None:
    """Run the boot steps in order on the calling thread, each one recorded in the document.

    A step's exception is logged and recorded as that step's failure; the next step still
    runs (every step is fail-soft, as the lifespan ran them), so one broken extension never
    hides the supervisor. A stop set between steps ends the walk: the generation is ending
    and nothing more of it starts. ``steps_done`` is recorded only after the last step."""
    for name, step in steps:
        if stop.is_set():
            log.info("Boot steps stopped before %s: the generation is ending", name)
            return
        document.step_started(name)
        try:
            step()
        except Exception as exc:
            log.error("Boot step %s failed", name, exc_info=True)
            document.step_ended(name, error=f"{type(exc).__name__}: {exc}")
        else:
            document.step_ended(name)
    document.steps_done()


def start_boot_coordinator(steps: Sequence[BootStep], *, stop: threading.Event) -> threading.Thread:
    """The lifespan's boot coordinator: a daemon thread running ``run_boot_steps`` while the
    socket already serves; the lifespan owns ``stop`` and sets it at teardown."""
    thread = threading.Thread(target=run_boot_steps, args=(steps,), kwargs={"stop": stop},
                              daemon=True, name="boot-coordinator")
    thread.start()
    return thread


def wait_event(event: threading.Event, *stops: threading.Event, poll_sec: float = 0.1) -> bool:
    """Wait for ``event``; False as soon as any stop is set, even beside the event."""
    while True:
        if any(stop.is_set() for stop in stops):
            return False
        if event.wait(poll_sec):
            return not any(stop.is_set() for stop in stops)
