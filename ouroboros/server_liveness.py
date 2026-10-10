"""Wedge detection for the supervisor generation.

The two silent-wedge predicates (a stalled supervisor loop, a heartbeat-silent
in-process chat turn), the owner alert one of them raises, the measurements the
loop publishes about itself, and the dedicated watchdog thread that evaluates
both outside the loop it watches.
"""

from __future__ import annotations

import posixpath
import queue
import re
import sys
import threading
import time
from typing import Any, Callable, Optional

from ouroboros.deadline_utils import parse_deadline_ts, utc_now
from ouroboros.runtime_limits import (
    BUDGET_PROJECTION_RETRY_SEC, SUPERVISOR_EVENT_BATCH_MAX_EVENTS, SUPERVISOR_EVENT_BATCH_MAX_SEC,
)
from ouroboros.server_process import DATA_DIR, log, _restart_requested
from ouroboros.utils import utc_now_iso

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
    journaled with that name), then publish ``startup`` again."""
    liveness[1], liveness[0] = loop_phase_facts(liveness, phase), time.monotonic()
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


def observe_worker_event_lag(liveness: list, evt: Any) -> None:
    """Record how far behind the loop is on the worker event it is draining now.

    The worker stamps ``ts`` in its OWN process when it queues the event, so this
    is a cross-process wall-clock gap — a measurement, never the watchdog's
    monotonic trigger. An event without a parsable worker stamp is skipped: the
    host's own clock is not evidence about when a worker spoke.
    """
    stamped = parse_deadline_ts(evt.get("ts")) if isinstance(evt, dict) else None
    if stamped is None:
        return
    lag = (utc_now() - stamped).total_seconds()
    if liveness[_LAG] is None or lag > liveness[_LAG]:
        liveness[_LAG] = lag


def drain_worker_events(event_q: Any, ctx: Any, liveness: list, *, on_restart: Callable[..., Any]) -> bool:
    """One BOUNDED events pass of the supervisor loop: FIFO, at most
    ``SUPERVISOR_EVENT_BATCH_MAX_EVENTS`` events; the ``SUPERVISOR_EVENT_BATCH_MAX_SEC``
    budget is checked between handlers (a handler already running finishes, so one
    slow handler can overrun it), then the loop runs bridge intake. A ``restart_request`` goes to
    ``on_restart``; every other event is lag-observed and dispatched. The remainder
    stays queued for the next turn, so a producer that keeps the queue non-empty
    can never starve owner-message intake. Returns True when the pass stopped at
    its bound (a backlog may remain: the caller skips its idle sleep)."""
    from supervisor.events import dispatch_event

    deadline = time.monotonic() + SUPERVISOR_EVENT_BATCH_MAX_SEC
    drained = 0
    while drained < SUPERVISOR_EVENT_BATCH_MAX_EVENTS and time.monotonic() < deadline:
        try:
            evt = event_q.get_nowait()
        except queue.Empty:
            return False
        drained += 1
        if evt.get("type") == "restart_request":
            on_restart(evt, ctx)
            continue
        observe_worker_event_lag(liveness, evt)
        dispatch_event(evt, ctx)
    return True


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
    ``path:line in func`` frames (repository files relative, others absolute). Walk code
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


def _innermost_repo_frame(stack: list[str]) -> str:
    """``path:function`` of the innermost frame inside the repository - a relative path, else
    the innermost frame outside the Python runtime ('' when none) - without the line number,
    so the samples of one stalled function fold into one row. A relocatable interpreter may
    report its prefix through ``bin/..`` while loading modules from the normalized path, so
    both sides are compared lexically normalized."""
    runtime = tuple({posixpath.normpath(prefix.replace("\\", "/"))
                     for prefix in (sys.prefix, sys.base_prefix) if prefix})
    outside = ""
    for row in reversed(stack):
        parsed = _STACK_ROW.match(row)
        path, func = (parsed.group("path"), parsed.group("func")) if parsed else ("", "")
        key = f"{path}:{func}"
        if path and not _ABSOLUTE_PATH.match(path):
            return key
        if path and not outside and not posixpath.normpath(path.replace("\\", "/")).startswith(runtime):
            outside = key
    return outside


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
    that loop stalls). It observes two silent-wedge classes and reports them
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
    information, and the rows carry the diagnosis anyway. It deliberately does NOT
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
        samples: dict = {}  # innermost repository frame -> stack samples while the stall is open
        last_stack: list = []
        wedged_tasks: set[str] = set()

        def sample(stack: list) -> None:
            nonlocal last_stack
            if stack:
                last_stack = stack
            key = _innermost_repo_frame(stack) or "(outside the repository)"
            samples[key] = samples.get(key, 0) + 1
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
                    samples, last_stack = {}, []
                    sample(stack)
                else:
                    sample(_loop_thread_stack(watched_ident))  # one bounded sample per interval
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
                            "samples": sum(samples.values()),
                            "top_frames": [{"frame": frame, "samples": count} for frame, count in
                                           sorted(samples.items(), key=lambda item: (-item[1], item[0]))[:5]],
                            "last_stack": last_stack,
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

    threading.Thread(target=_watch, name="supervisor-liveness-watchdog", daemon=True).start()
