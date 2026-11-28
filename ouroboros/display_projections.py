"""The supervisor's display and diagnostic projections, written off its loop by one host duty.

The loop only marks what a turn changed: an ``llm_usage`` event sets the compatibility budget mark
(``ctx.budget_projection_dirty``) and marks its root in ``LIVE_ROOT_COSTS``; the duty watchdog captures
an overrun at detection (its duty, running time, thread stack and instant: ``journal_duty_stall``); an
owner turn's first dispatch is stamped by its observer (``update_generation_facts``). At the end of its
maintenance look the loop takes what is due (``take``: the marked roots with the RUNNING task carrying
each and a copy of the fields the projection reads, and the budget mark) and hands it to ONE coalescing
pass (``project``), the ``display-projections`` row of ``server_maintenance.HOST_DUTIES``: its latch, its
start and its stall watchdog are the other duties'. The pass does what the loop once did inline, where one
stalled ``os.replace`` of ``state.json`` (134 s) or ``os.stat`` of ``usage.sqlite`` (145.6 s) stopped owner
intake on 27.11.2026:

* the queued diagnostics, oldest first: an overrun's warning line and ``host_duty_stall`` row, then its
  ``host_duty_stall_end`` (``journal_duty_stall_end``), each episode once and in order;
* each due root's cost figure (``LiveRootCosts.read``), published while its carrier still stands and no
  mark made since its take supersedes it;
* the compatibility budget write (``supervisor.state.update_budget_from_usage``: the ledger read before
  STATE_LOCK, its ``(compaction_epoch, seq)`` order under it); a refused or raising write leaves the
  mark owed and its next attempt no sooner than ``BUDGET_PROJECTION_RETRY_SEC``;
* the after-update facts (``update_generation_facts.publish_pending``), coalesced to the latest
  observations, with their operation, outcome and generation checks.

A mark made while a pass runs is the next pass's; a pass that cannot start leaves its work due. The
state pair keeps its synchronous writer (``supervisor.state._save_state_unlocked``), in this thread now and
under ``locks/state.lock``: while a pass stands inside that replace, the loop's own state writes wait up to
that lock's timeout (``update_state``: 4 s) and are then refused typed, and an owner message proceeds
without its activity stamp (``server.py``: "Owner activity not stamped"); the loop once stood through the
whole replace itself. All of it is display and diagnostics, never authority: admission, paid dispatch, caps
and receipts keep their synchronous owners and read ``usage.sqlite``, never ``state.json`` (ARCHITECTURE
§10.28). While this duty itself stands, the rows other duties' overruns queue wait for it in memory: two per
overrunning pass.

The process's end takes ONE last pass (``drain``), after the pass in flight and inside the stop grace: the
loop's context (``bind``) gives it the budget mark of a turn a stop or a restart ended before its look,
beside the queued rows and newer after-update facts. A restart takes it in its census, before the store
switch, which refuses while a pass still runs; no pass starts after it. What stays unwritten (a pass that
outlived its share of the grace, a mark inside a failed write's retry bound) is the exit's loss: the ledger
is the record of money.
"""

from __future__ import annotations

import collections
import threading
import time
from typing import Any, Callable, Deque, Dict, List, NamedTuple, Optional, Tuple

from ouroboros.cost_projection import LIVE_ROOT_COSTS, DueRoot
from ouroboros.runtime_limits import BUDGET_PROJECTION_RETRY_SEC
from ouroboros.server_process import DATA_DIR, log
from ouroboros.utils import utc_now_iso

DUTY_NAME = "display-projections"
LATCH = threading.Lock()  # single flight: the loop takes it before a pass starts, the pass releases it
_ROWS: Deque[Tuple[Optional[tuple], Dict[str, Any]]] = collections.deque()  # (warning line, journal row)
_ROWS_LOCK = threading.Lock()
_CLOSED = threading.Event()  # the process's end took its last pass (``drain``): no pass starts after it
_CONTEXT: Any = None  # the supervisor loop's event context (``bind``): the last pass takes its budget mark


class ProjectionWork(NamedTuple):
    """What the loop found due, taken on the loop: nothing in it is shared with the live RUNNING map."""

    drive_root: Any
    roots: Tuple[DueRoot, ...]
    budget: Optional[Callable[[], None]]  # the compatibility write and where its outcome goes, when due


def bind(ctx: Any) -> None:
    """On the supervisor thread, before its loop (``server._run_supervisor``): the loop's event context, whose
    budget mark the last pass takes (``drain``) when no tick did, a turn a stop or a restart ended first."""
    global _CONTEXT
    _CONTEXT = ctx


def take(ctx: Any) -> Optional[ProjectionWork]:
    """On the loop: the work due now, or None (no root to read, no due budget mark, no queued row, no
    newer after-update observation). Roots and the budget mark are taken: a mark from now on is the next
    pass's."""
    from ouroboros import update_generation_facts

    drive_root = getattr(ctx, "DRIVE_ROOT", None)
    roots = LIVE_ROOT_COSTS.take(getattr(ctx, "RUNNING", None), drive_root)
    budget = _take_budget_mark(ctx)
    if roots or budget is not None or _rows_queued() or update_generation_facts.pending():
        return ProjectionWork(drive_root, roots, budget)
    return None


def _take_budget_mark(ctx: Any) -> Optional[Callable[[], None]]:
    """The budget mark when it is due (set, and past the retry bound of a failed write), consumed here.
    The write it returns sets the context's two fields only while its pass holds the latch, and the loop
    consumes the mark only after taking the latch, so neither overwrites the other's news."""
    if not getattr(ctx, "budget_projection_dirty", False):
        return None
    if time.monotonic() < float(getattr(ctx, "budget_projection_retry_at", 0.0) or 0.0):
        return None
    ctx.budget_projection_dirty = False

    def write() -> None:
        now = time.monotonic()
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
        ctx.budget_projection_retry_at = 0.0 if written else now + BUDGET_PROJECTION_RETRY_SEC
        if not written:
            ctx.budget_projection_dirty = True  # owed until a later write lands

    return write


def dispatch(ctx: Any, start: Callable[..., bool], busy: Callable[[Any, float], None]) -> None:
    """The tick's look, on the loop (``server_maintenance``): a pass still running is the watchdog's
    (``busy`` journals it once past the projection's retry bound); otherwise, when anything is due, one
    pass starts with the snapshot taken here. A start that fails has released the latch; its work is due
    again. A look that raises frees the latch for the next tick's; after the last pass nothing starts."""
    if not LATCH.acquire(blocking=False):
        busy(LATCH, BUDGET_PROJECTION_RETRY_SEC)
        return
    try:
        work = None if _CLOSED.is_set() else take(ctx)
    except Exception:
        log.warning("The display projections' look failed; the next tick looks again", exc_info=True)
        work = None
    if work is None:
        LATCH.release()
    elif not start(LATCH, DUTY_NAME, project, (work, LATCH)):
        LIVE_ROOT_COSTS.restore(work.roots)
        if work.budget is not None:
            ctx.budget_projection_dirty = True


def project(work: Optional[ProjectionWork], latch: Any = None) -> None:
    """ONE pass, off the loop: the queued rows, the due root figures, the budget write, then the after-update
    facts. A step that raises is logged and the next one still runs; the pass releases its latch last."""
    from ouroboros import update_generation_facts

    steps: List[Tuple[str, Callable[[], None]]] = [("journal rows", _write_rows)]
    if work is not None:
        steps.append(("root cost figures", lambda: LIVE_ROOT_COSTS.read(work.roots, work.drive_root)))
        if work.budget is not None:
            steps.append(("budget projection", work.budget))
    steps.append(("after-update facts", update_generation_facts.publish_pending))
    try:
        for name, step in steps:
            try:
                step()
            except Exception:
                log.warning("Display projection step failed: %s", name, exc_info=True)
    finally:
        if latch is not None:
            latch.release()


def journal_duty_stall(duty: Dict[str, Any], cadence_sec: float, stack: List[str], flags: Dict[str, Any]) -> None:
    """At detection, on the loop (``server_maintenance._duty_busy``): the episode as it stands now becomes its
    warning line and ``host_duty_stall`` row, queued for the pass, and the per-duty count of the after-update
    facts moves in memory. Under the queue's lock an onset is queued before its end, or not at all when the pass
    ended while its stack was read: no stall row stands without its closing row."""
    running = round(time.time() - duty["since"], 1)
    row = {"ts": utc_now_iso(), "type": "host_duty_stall", "duty": duty["name"], "running_sec": running,
           "cadence_sec": cadence_sec, **flags, **({"stack": stack} if stack else {})}
    line = ("Off-loop duty %s has run %.0fs, past its %.0fs cadence; its thread is at: %s",
            duty["name"], running, cadence_sec, stack[-1] if stack else "(stack unavailable)")
    with _ROWS_LOCK:
        if duty.get("ended"):
            return
        duty["alerted"] = True
        _ROWS.append((line, row))
    try:
        from ouroboros.update_generation_facts import observe_duty_overrun

        observe_duty_overrun(duty["name"])  # once per alerted pass: the after-update facts count it
    except Exception:
        log.debug("generation duty observation failed", exc_info=True)


def journal_duty_stall_end(duty: Dict[str, Any]) -> None:
    """A pass ended (on its own thread, ``server_maintenance._duty_ended``): an alerted one's closing row is queued
    behind its onset row."""
    with _ROWS_LOCK:
        duty["ended"] = True
        if duty["alerted"]:
            _ROWS.append((None, {"ts": utc_now_iso(), "type": "host_duty_stall_end", "duty": duty["name"],
                                 "running_sec": round(time.time() - duty["since"], 1)}))


def _rows_queued() -> bool:
    with _ROWS_LOCK:
        return bool(_ROWS)


def _write_rows() -> None:
    """The queued lines and rows, oldest first; a row that cannot be written is logged at debug and dropped,
    as the loop's own append dropped it."""
    from supervisor.state import append_jsonl

    while True:
        with _ROWS_LOCK:
            if not _ROWS:
                return
            line, row = _ROWS.popleft()
        if line:
            log.warning(*line)
        try:
            append_jsonl(DATA_DIR / "logs" / "supervisor.jsonl", row)
        except Exception:
            log.debug("%s row failed", row.get("type"), exc_info=True)


def drain(deadline: Optional[float] = None) -> bool:
    """The process's end, once (``server_control.drain_display_projections``: the restart census before the store
    switch, else the exit or the direct re-exec); a later call and Panic wait for nothing. It closes the duty, so no
    pass starts after it, waits for a pass in flight, then runs the last pass: the bound context's due budget mark
    (``bind``), the queued rows and newer after-update facts, again while rows arrive meanwhile (a pass's own end
    row); root figures are heartbeat display and end with the process. Both waits share at most half of the stop
    grace left (``deadline``, the bound ``server_process.end_task_runs`` keeps; without a started grace no last pass
    runs). True when no pass is left running: one that is stays unjoined, its unwritten work the exit's loss."""
    from ouroboros import update_generation_facts
    from ouroboros.server_control import panic_termination_requested

    if _CLOSED.is_set() or panic_termination_requested():
        return not LATCH.locked()
    _CLOSED.set()
    wait = max(0.0, (deadline - time.monotonic()) / 2) if deadline is not None else 0.0
    cutoff = time.monotonic() + wait
    if wait <= 0.0:
        return not LATCH.locked()
    if not LATCH.acquire(timeout=wait):
        log.warning("The display projections' pass outlived its share of the stop grace; it stays unjoined")
        return False
    budget = _take_budget_mark(_CONTEXT) if _CONTEXT is not None else None
    while budget is not None or _rows_queued() or update_generation_facts.pending():
        try:
            threading.Thread(target=project, args=(ProjectionWork(None, (), budget), LATCH),
                             name=f"{DUTY_NAME}-drain", daemon=True).start()
        except Exception:
            LATCH.release()
            if budget is not None:
                _CONTEXT.budget_projection_dirty = True
            log.warning("The display projections' last pass could not start; its work stays unwritten", exc_info=True)
            return True
        budget = None
        if not LATCH.acquire(timeout=max(0.0, cutoff - time.monotonic())):
            log.warning("The display projections' last pass outlived its share of the stop grace; its rest stays unwritten")
            return False
    LATCH.release()
    return True
