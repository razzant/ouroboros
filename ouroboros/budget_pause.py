"""Exact mid-run budget pause and its owner-granted same-ID Resume (#1196).

A pooled task that meets ANY monetary rail after work has already been done
(global exhaustion, root fence, graceful in-task ceiling, last-fit wrap-up,
soft landing, a refused dispatch) no longer spends a wrap-up call and ends as
``failed/budget_exhausted``. It PAUSES instead:

1. ``pausing`` — a process-local dispatch fence closes for this task id:
   ``usage_accounting.reserve_attempt`` refuses every NEW physical send under
   the task's scope (the loop itself, tools, reviewers, verdict extraction),
   so nothing sent after this point can outrun the checkpoint. Already-sent
   local producers keep their durable identities; nothing is re-POSTed and
   nothing is extracted with a Light model on the way out. The fence does not
   reopen here: once it closes, the task is committed to pausing.
2. Quiescence HOLD — the task keeps its worker and stays NONTERMINAL in
   ``pausing`` until BOTH local producer families have settled: review
   attempts already sent, and tool futures that keep running after their
   logical timeout, including their late settlement callbacks. A producer
   still running, or a checkpoint write that fails, HOLDS the task under a
   typed ``hold_reason`` published to the owner — never a paid wrap-up call,
   never a terminal, never a claimed durable pause. Only the task's existing
   Stop/Panic/deadline/cancel controls end a hold without a pause.
3. External observation — every delegated run this task still holds is
   observed from the durable custody rows and, because pre-terminal
   subscription cost coverage is NOT provable, a stop is REQUESTED through the
   verified cancel seam (owner Q8). The typed outcome (requested / confirmed /
   failed / containment fault) is recorded per run; an unknown stop never
   licenses a second writer.
4. Checkpoint — the ONE loop serializer (``owner_wait.continuation_state``)
   captures the exact continuation plus a program counter: which tool calls of
   the last batch still have no result, so a resume executes only those and
   never replays a completed call.
5. Durable ``budget_pause`` row on the task result (state ``pausing``), THEN
   ``BudgetPauseRequested`` unwinds the loop nonterminally. The worker reports
   ``budget_pause`` with ``exact_continuation=True``; the supervisor moves the
   SAME task id back to PENDING under a ``_budget_pause`` marker (no worker,
   no slot), writes state ``paused``, and keeps it there across restarts
   without waking it (``budget_pause_restore_refusal``).

Resume is an explicit OWNER act (owner Q7/Q10): raising a budget wakes
nobody. ``queue_transitions.resume_budget_paused_task`` validates money,
Stop/cancel intent, the task deadline, the finite lifetime (with the paused
interval carried SEPARATELY — the original ``started_at`` is never moved and
the quota clock is never used as a pause clock) and the checkpoint source,
then mints ONE single-use ``grant``. The loop consumes the grant
(``resume_paused_loop``), rebinds the saved cognition, refreshes the
planning threshold within the money still authorized (Q10) and discloses
workspace drift and external custody before any new effect. Cancelled,
completed and otherwise-stopped members are never revived; a root's Resume
only makes its own budget-paused descendants ELIGIBLE — the model selects
each one explicitly through the same task control (Q9).

A direct owner-chat turn pauses the SAME way and under the SAME task id: the
live direct actor ends (its registry entry is released as on any other ending,
so no second live actor ever exists for that id), its ``budget_pause`` event
carries the turn's own task record with its ``_is_direct_chat`` lane fact, and
the supervisor parks that record in the existing PENDING carrier under the
same ``_budget_pause`` marker. The existing Resume endpoint grants it and a
pooled worker continues the checkpointed cognition cold — origin, workspace,
attempt, accounting and the opaque acceptance identities travel in the
checkpoint; browser and service handles are never resurrected. Only a turn
with no task id or no durable root is excluded, and loudly
(``resource_limit.exact_pause_unavailable``).

Every refusal to restore, grant or dispatch is typed and RETAINS the pause: a
corrupt or missing source, an unwritable revocation, a lifted root fence over
a zero-dispatch sibling all become a durable non-dispatch HOLD on the queue
row (``events_budget.budget_hold_fact``) beside the saved pause, never a
cancellation and never a silent dispatch; a grant is bound to ONE pause id
and ONE resume generation, so ``pauseA -> Resume -> pauseB -> restart`` can
never dispatch on the earlier grant. This module adds no scheduler, ledger or
recovery framework: it reuses the queue's ``_budget_pause`` carrier, the actor
source store, the task-result authority and the delegated custody rows that
already exist.
"""

from __future__ import annotations

import json
import logging
import pathlib
import threading
import time
import uuid
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional

log = logging.getLogger(__name__)

# Rails (which monetary stop produced the pause). GRACEFUL_RAILS are the
# planning stops an explicit Resume may refresh (Q10); the others are hard
# money and need an owner increase before any grant validates.
RAIL_GLOBAL_EXHAUSTED = "global_exhausted"
RAIL_DISPATCH_REFUSED = "dispatch_refused"
RAIL_GRACEFUL_CEILING = "graceful_ceiling"
RAIL_WRAPUP_LAST_FIT = "wrapup_last_fit"
RAIL_SOFT_LAND = "soft_land"
GRACEFUL_RAILS = frozenset({RAIL_GRACEFUL_CEILING, RAIL_WRAPUP_LAST_FIT, RAIL_SOFT_LAND})
# The owner's Pause of a whole tree (``owner_pause.py``) is not a monetary stop:
# it rides the SAME exact checkpoint, park and grant, under reason ``owner``,
# and never requests a stop of work already sent.
from ouroboros.owner_pause import RAIL_OWNER_PAUSE, REASON_OWNER  # noqa: E402
# The model's own COLD sleep (``model_sleep``) rides the same park too; only its
# selected sources, through the supervisor's sleep-wake policy, wake it.
RAIL_MODEL_SLEEP = "model_sleep"
REASON_SLEEP = "sleep"
RAILS = GRACEFUL_RAILS | {RAIL_GLOBAL_EXHAUSTED, RAIL_DISPATCH_REFUSED, RAIL_OWNER_PAUSE, RAIL_MODEL_SLEEP}
_RAIL_REASONS = {RAIL_OWNER_PAUSE: REASON_OWNER, RAIL_MODEL_SLEEP: REASON_SLEEP}

STATE_PAUSING = "pausing"
STATE_PAUSED = "paused"
STATE_RESUME_GRANTED = "resume_granted"
STATE_RESUMED = "resumed"
LIVE_PAUSE_STATES = frozenset({STATE_PAUSING, STATE_PAUSED, STATE_RESUME_GRANTED})

RESUME_POLICY = "owner_resume_same_id"
REASON_CODE = "budget_paused"
OWNER_REASON_CODE = "owner_paused"

# External-run stop outcomes as recorded on the pause row (independent facts,
# never collapsed into one boolean).
EXTERNAL_RUNNING = "running"
EXTERNAL_STOP_REQUESTED = "stop_requested"
EXTERNAL_STOP_CONFIRMED = "stop_confirmed"
EXTERNAL_STOP_UNKNOWN = "stop_unknown"


class BudgetPauseRequested(Exception):
    """The pause record is durable; unwind the loop WITHOUT a terminal.

    Carries the durable pause row. The worker must not emit ``task_done`` or a
    Main final for this task; the supervisor owns the queue transition.
    """

    def __init__(self, pause: Dict[str, Any]) -> None:
        super().__init__(f"budget pause {pause.get('pause_id', '')} ({pause.get('rail', '')})")
        self.pause = dict(pause)


# --- process-local dispatch fence ------------------------------------------------

_FENCE_LOCK = threading.Lock()
_FENCED: set[str] = set()


def begin_dispatch_fence(task_id: str) -> None:
    """Close NEW physical sends for ``task_id`` in this process (idempotent)."""
    tid = str(task_id or "").strip()
    if not tid:
        return
    with _FENCE_LOCK:
        _FENCED.add(tid)


def end_dispatch_fence(task_id: str) -> None:
    with _FENCE_LOCK:
        _FENCED.discard(str(task_id or "").strip())


def dispatch_fenced(task_id: str) -> bool:
    """Whether a NEW send under ``task_id`` must be refused right now."""
    tid = str(task_id or "").strip()
    if not tid:
        return False
    with _FENCE_LOCK:
        return tid in _FENCED


# --- tool-future quiescence registry (task/attempt scoped) -------------------------

# One row per REAL tool future, registered by ``loop_tool_execution`` right
# after the submit and BEFORE that call's own timeout can fire — a future first
# seen on the timeout path is invisible in exactly the window that matters.
# ``future.done()`` is not the fact recorded here: a late settlement callback
# may still be producing effects after it, so a row settles only when every
# callback that claimed it has FINISHED. Settlement OWNERSHIP is pinned from
# the registration until the registering caller releases it (its own result
# arrived, or it claimed the late-callback hold): a future that completes
# between the caller's timeout and its ``hold_tool_settlement`` claim would
# otherwise read as settled-and-unheld for one instant, and a concurrent
# registration could prune it — the late callback's effects then escape the
# quiescence gate. Pruning is per SCOPE and only over rows nobody owns or
# holds; rows of another attempt are never touched (``forget_tool_scope`` is
# the only cross-scope drop, at that attempt's own end).
_TOOL_LOCK = threading.Lock()
_TOOL_FUTURES: Dict[str, Dict[str, Dict[str, Any]]] = {}


def tool_scope_key(ctx: Any) -> str:
    """The registry scope: one task ATTEMPT, never a whole task id."""
    return f"{str(getattr(ctx, 'task_id', '') or '')}|{int(getattr(ctx, 'task_attempt', 1) or 1)}"


def _row_settled_locked(row: Dict[str, Any]) -> bool:
    """Settled = the future finished and nobody holds it: neither a late
    callback nor the registering owner's pin. Read with ``_TOOL_LOCK`` held."""
    return bool(row.get("done")) and int(row.get("holds") or 0) <= 0


def register_tool_future(ctx: Any, operation_id: str, tool: str, future: Any) -> Callable[[], None]:
    """Track one tool future until its settlement callbacks have finished.

    Returns the OWNER release: the registering caller calls it once its own
    handling of the call is over — after the result arrived, or after the
    timeout path claimed the late-settlement hold — so the row cannot be
    pruned in between. Calling it twice is harmless.
    """
    op = str(operation_id or "")
    if future is None or not op or not str(getattr(ctx, "task_id", "") or ""):
        return lambda: None
    row: Dict[str, Any] = {"operation_id": op, "tool": str(tool or ""),
                           "settled": threading.Event(), "holds": 0, "done": False}
    scope = tool_scope_key(ctx)
    with _TOOL_LOCK:
        rows = _TOOL_FUTURES.setdefault(scope, {})
        # Prune ONLY this scope's fully settled rows: done, unheld AND released.
        for op_id in [i for i, r in rows.items() if _row_settled_locked(r)]:
            rows.pop(op_id, None)
        rows[op] = row
    # The owner's pin is the FIRST hold on the row, taken before the done
    # callback is attached: a future that has already finished cannot settle
    # (and be pruned) before the registering caller has decided about a late hold.
    owner_release = hold_tool_settlement(ctx, op)

    def _done(_future: Any) -> None:
        with _TOOL_LOCK:
            row["done"] = True
            settled = _row_settled_locked(row)
        if settled:
            row["settled"].set()

    # Both producers register a ``concurrent.futures.Future`` (the stateful and
    # the abandoned-on-timeout executors): attaching the callback never raises,
    # and an already-finished future runs ``_done`` inline.
    future.add_done_callback(_done)
    return owner_release


def hold_tool_settlement(ctx: Any, operation_id: str) -> Callable[[], None]:
    """Claim a row (the registering owner's pin, or a late settlement callback's
    hold); returns its single-use release.

    A late caller attaches its callback AFTER claiming and calls the release as
    the callback's last act, so the registry reports the tool as settled only
    once that callback's own effects are over. The claim is taken while the
    registering owner still pins the row (it releases only after this claim),
    so a future that finished a moment earlier is still here to be claimed.
    An unregistered row returns a no-op: this registry never invents an
    observation it does not hold. Releasing twice is harmless.
    """
    op = str(operation_id or "")
    with _TOOL_LOCK:
        row = _TOOL_FUTURES.get(tool_scope_key(ctx), {}).get(op)
        if row is None:
            return lambda: None
        row["holds"] += 1
        row["settled"].clear()
    released = False

    def _release() -> None:
        nonlocal released
        with _TOOL_LOCK:
            if released:
                return
            released = True
            row["holds"] = max(0, int(row["holds"]) - 1)
            settled = _row_settled_locked(row)
        if settled:
            row["settled"].set()

    return _release


def forget_tool_scope(ctx: Any) -> None:
    """Drop one attempt's rows (that attempt's end, and test hygiene for this global)."""
    with _TOOL_LOCK:
        _TOOL_FUTURES.pop(tool_scope_key(ctx), None)


def drain_local_tool_futures(ctx: Any, *, timeout_sec: float) -> Dict[str, Any]:
    """Bounded observation of THIS attempt's tool futures; settles nothing.

    A call abandoned at its logical timeout keeps running, and its late
    settlement callback keeps producing effects after the worker returns.
    Releasing the native process while either is outstanding is exactly the
    escape this gate exists to refuse. A row its owner has not released yet is
    unsettled too: the owner is still deciding whether a late callback claims it.
    """
    with _TOOL_LOCK:
        rows = list(_TOOL_FUTURES.get(tool_scope_key(ctx), {}).values())
    deadline = time.monotonic() + max(0.0, float(timeout_sec or 0.0))
    settled: List[Dict[str, str]] = []
    unsettled: List[Dict[str, str]] = []
    for row in rows:
        fact = {"operation_id": row["operation_id"], "tool": row["tool"]}
        remaining = max(0.0, deadline - time.monotonic())
        if row["settled"].wait(remaining):
            settled.append(fact)
        else:
            unsettled.append({**fact, "state": "running"})
    return {"drained": not unsettled, "registry": "ok", "settled": settled, "unsettled": unsettled}


# --- program counter ---------------------------------------------------------------

def pending_tool_call_ids(messages: List[Dict[str, Any]]) -> List[str]:
    """Tool calls of the LAST assistant batch that have no recorded result.

    A missing result row is NOT proof the call never ran (a timeout or an
    exception after the effect, a parallel batch cut mid-way). These ids are
    therefore recorded as EXECUTION-UNKNOWN: a resume restores cognition only,
    never re-executes them, and tells the model to verify from authoritative
    state before repeating any of them.
    """
    last_assistant = None
    for index in range(len(messages) - 1, -1, -1):
        row = messages[index]
        if isinstance(row, dict) and row.get("role") == "assistant":
            last_assistant = index
            break
    if last_assistant is None:
        return []
    calls = messages[last_assistant].get("tool_calls") or []
    wanted = [str(call.get("id") or "") for call in calls if isinstance(call, dict) and call.get("id")]
    if not wanted:
        return []
    answered = {
        str(row.get("tool_call_id") or "")
        for row in messages[last_assistant + 1:]
        if isinstance(row, dict) and row.get("role") == "tool"
    }
    return [call_id for call_id in wanted if call_id not in answered]


# --- external custody (owner Q8) ---------------------------------------------------

def observe_task_runs(root: Any, task_id: str, *, reason: str = "budget_resume_uncovered_cost",
                      read_error: str = "", request_stop: bool = True) -> Dict[str, Any]:
    """The ONE observer body behind the loop-side pause and the supervisor-side grant.

    A FRESH custody read for ``task_id`` on ``root`` (never a pause row's saved
    summary), requesting a stop for every run still open — pre-terminal
    subscription cost coverage cannot be proved from the ledger (owner Q8), so
    its remaining cost is uncovered/unknown while it runs — and recording the
    typed outcome through the verified cancel seam. ``requested`` and
    ``unknown`` are NOT death: the run stays under this task's custody and no
    second writer may be started over it. An unreadable custody store
    (``read_error`` from the caller, or the replay failing here) is a typed
    ``custody_read=failed`` observation, never an empty (clean-looking) list.
    ``request_stop=False`` is the owner Pause's policy (owner Batch4 5A): sent
    work is OBSERVED, never cancelled, at the pause and at its Resume alike —
    each open run is reported ``running`` under ``stop_policy=observe_only``.
    """
    runs: List[Any] = []
    pending_rows: List[Dict[str, Any]] = []
    if not read_error:
        try:
            from ouroboros import delegate_custody as custody

            mine = str(task_id or "")
            # The memo silently falls back to a lenient read that skips an
            # unreadable segment; probe the chain first so hidden custody is
            # UNKNOWN, never "no open runs" (Astra run-a882315dbcd7 #2).
            if custody.custody_log_unreadable(pathlib.Path(root)):
                raise OSError("custody_log_unreadable")
            from ouroboros.delegate_custody_memo import custody_rows_with_integrity

            # ONE snapshot feeds both projections: a START_REQUESTED that becomes
            # STARTED between two reads must land in one of them (Astra 6fe5 #1),
            # and its integrity is judged on that same read. An unparseable custody
            # line naming this task may hide its request.
            rows_read, malformed = custody_rows_with_integrity(pathlib.Path(root), mine)
            snapshot = list(rows_read)
            if malformed is None or malformed:
                raise OSError(f"custody_rows_incomplete:{'unknown' if malformed is None else malformed}")
            runs = [run for run in custody.replay(pathlib.Path(root), rows=snapshot).values()
                    if str(getattr(run, "task_id", "") or "") == mine and not getattr(run, "settled", True)]
            # A START_REQUESTED whose response was lost has no run id yet but may
            # be a live remote writer: unknown custody, never absence (#3).
            from ouroboros.delegate_pending import pending_invocations

            pending = [row for row in pending_invocations(pathlib.Path(root), rows=snapshot)
                       if str(row.get("task_id") or "") == mine]
            pending_rows = [{"run_id": "", "invocation_id": str(row.get("invocation_id") or ""),
                             "route": str(row.get("route") or ""),
                             "cost_coverage": "unproven_preterminal", "stop_policy": "reconcile_first",
                             "state": EXTERNAL_STOP_UNKNOWN, "stop_outcome": "pending_invocation_unbound",
                             "detail": ""} for row in pending]
        except Exception as exc:
            log.warning("External custody rows unreadable for %s", task_id, exc_info=True)
            read_error = f"{type(exc).__name__}: {str(exc)[:200]}"
    if read_error:
        # Held as UNKNOWN on the pause row: the grant re-reads custody and
        # refuses while it stays unreadable (never "no runs").
        return {"runs": [], "observed_at": time.time(), "custody_read": "failed",
                "error": read_error, "coverage_basis": "custody_unreadable"}
    if not runs:
        if pending_rows:
            return {"runs": pending_rows, "observed_at": time.time(), "custody_read": "ok",
                    "coverage_basis": "pending_invocations_unbound"}
        return {"runs": [], "observed_at": time.time(), "custody_read": "ok", "coverage_basis": "no_open_runs"}
    rows: List[Dict[str, Any]] = []
    if not request_stop:
        rows = [{"run_id": str(getattr(run, "run_id", "") or ""),
                 "route": str(getattr(run, "route", "") or getattr(run, "route_id", "") or ""),
                 "cost_coverage": "unproven_preterminal", "stop_policy": "observe_only",
                 "state": EXTERNAL_RUNNING, "stop_outcome": "", "detail": ""} for run in runs]
        return {"runs": rows + pending_rows, "observed_at": time.time(), "custody_read": "ok",
                "coverage_basis": "observed_without_stop_request"}
    try:
        from ouroboros.gateways.claudexor import ClaudexorGateway

        gateway = ClaudexorGateway()
        gateway.handshake()
    except Exception as exc:
        log.warning("Budget pause cannot reach the harness gateway to request stops: %s", exc)
        gateway = None
    try:
        from ouroboros import delegate_custody as custody

        for run in runs:
            row = {
                "run_id": str(getattr(run, "run_id", "") or ""),
                "route": str(getattr(run, "route", "") or getattr(run, "route_id", "") or ""),
                "cost_coverage": "unproven_preterminal",
                "stop_policy": "request_stop",
                "state": EXTERNAL_RUNNING,
                "stop_outcome": "",
                "detail": "",
            }
            if gateway is None:
                row.update(state=EXTERNAL_STOP_UNKNOWN, stop_outcome="not_issued_gateway_unavailable")
            else:
                try:
                    result = custody.cancel_and_verify(pathlib.Path(root), gateway, run, reason)
                    outcome = str(result.get("outcome") or "")
                    row.update(stop_outcome=outcome, detail=str(result.get("detail") or ""))
                    if outcome == custody.CANCEL_CONFIRMED:
                        row["state"] = EXTERNAL_STOP_CONFIRMED
                    elif outcome == getattr(custody, "CANCEL_REQUESTED", "requested"):
                        row["state"] = EXTERNAL_STOP_REQUESTED
                    else:
                        row["state"] = EXTERNAL_STOP_UNKNOWN
                except Exception as exc:
                    row.update(state=EXTERNAL_STOP_UNKNOWN, stop_outcome=f"error:{type(exc).__name__}",
                               detail=str(exc)[:300])
            rows.append(row)
    finally:
        if gateway is not None:
            try:
                gateway.close()
            except Exception:
                log.debug("Gateway close after pause stop requests failed", exc_info=True)
    # Open runs still get their stop requests; unbound invocations ride beside them.
    return {"runs": rows + pending_rows, "observed_at": time.time(), "custody_read": "ok",
            "coverage_basis": "preterminal_subscription_coverage_unprovable"}


def drain_local_review_attempts(task_id: str, *, timeout_sec: float) -> Dict[str, Any]:
    """Bounded quiescence barrier over THIS task's in-flight review attempts.

    Observation only: the fence already refuses new sends; this waits for the
    requests that were ALREADY sent to settle through their own custody path
    (``review_custody`` records each actor and its response reference before
    signalling the attempt's event). An attempt still open at the bound is
    recorded by operation id as ``unsettled`` — its durable identity is what the
    late-result custody adopts when the provider answers after release; it is
    never marked settled, PASS, failed or refunded here.
    """
    settled: List[Dict[str, str]] = []
    unsettled: List[Dict[str, str]] = []
    try:
        from ouroboros import review_custody as rc

        with rc._ACTIVE_LOCK:
            mine = [entry for entry in rc._ACTIVE.values()
                    if str(task_id) in str(getattr(entry, "wave_key", "") or "").split("|")
                    or f"|{task_id}|" in f"|{getattr(entry, 'key', '')}|"]
    except Exception:
        log.debug("Review attempt registry unavailable at budget pause", exc_info=True)
        return {"drained": False, "registry": "unavailable", "settled": settled, "unsettled": unsettled}
    deadline = time.monotonic() + max(0.0, float(timeout_sec or 0.0))
    for entry in mine:
        remaining = max(0.0, deadline - time.monotonic())
        row = {"operation_id": str(getattr(entry, "operation_id", "") or ""),
               "key": str(getattr(entry, "key", "") or "")[:120]}
        if entry.event.wait(remaining):
            settled.append(row)
        else:
            unsettled.append(row)
    return {"drained": not unsettled, "registry": "ok", "timeout_sec": float(timeout_sec or 0.0),
            "settled": settled, "unsettled": unsettled}


def local_producer_observation(ctx: Any, *, timeout_sec: float) -> Dict[str, Any]:
    """Fence, then OBSERVE every local producer this task still owns.

    TWO families, not one: review attempts already sent (drained through their
    own ``review_custody`` events) and tool futures still running after their
    logical timeout, including their late settlement callbacks. No new send, no
    re-POST, no Light extraction; nothing is settled, passed, failed or
    refunded here. The paid acceptance identity travels in the checkpoint's
    ``acceptance`` block, so a resume continues that panel instead of opening a
    duplicate one. ``quiescent`` is the RELEASE gate: while it is false the
    task holds, fenced and nonterminal, and keeps its native worker.
    """
    review = drain_local_review_attempts(str(ctx.task_id), timeout_sec=timeout_sec)
    tool_futures = drain_local_tool_futures(ctx, timeout_sec=timeout_sec)
    return {
        "dispatch_fence": "closed",
        "review_attempts": review,
        "tool_futures": tool_futures,
        "quiescent": bool(review.get("drained")) and review.get("registry") == "ok"
                     and bool(tool_futures.get("drained")),
    }


# --- durable row --------------------------------------------------------------------

class BudgetPauseSuperseded(ValueError):
    """A compare-and-set lost: a newer pause, state, grant or wait owns the row."""


def set_budget_pause(root: Any, task_id: str, row: Dict[str, Any],
                     expected_pause_id: Optional[str] = None, *,
                     expected_state: Any = None,
                     expected_grant_id: Optional[str] = None,
                     expected_owner_wait: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """CAS the pause; optional warm-sleep retention transfers its exact wait.

    ``expected_owner_wait`` additionally matches the complete owner-wait row
    and marks it ``retained`` in the same replace as the new exact pause. The
    original source stays available; ordinary owner/review waits do not opt in.

    Compare-and-set: ``expected_pause_id``, ``expected_state`` (one state or a
    collection of them) and ``expected_grant_id`` are each checked under the
    file lock against the row as it is NOW, so a grant, a revocation or a
    consumption written from a stale reading refuses (``ValueError``) instead
    of overwriting a newer pause, grant or state — pauseA -> Resume -> pauseB
    -> late revoke of A must never land on B (#1196). A refusal is typed
    ``BudgetPauseSuperseded`` (an observer's lost race is benign); an empty
    ``expected_grant_id`` means "no grant", ``None`` skips that comparison.
    """
    from ouroboros.task_results import (
        _TRULY_TERMINAL_STATUSES, require_writable_task_result_schema,
        stamp_task_result_schema, task_result_path,
    )
    from ouroboros.utils import update_json_locked

    expected_states = (
        None if expected_state is None
        else ({expected_state} if isinstance(expected_state, str) else set(expected_state))
    )

    def update(current: dict) -> dict:
        require_writable_task_result_schema(current)
        if not current.get("status") or current.get("task_id") != task_id:
            raise ValueError("budget pause requires its lifecycle owner's task result")
        if current.get("status") in _TRULY_TERMINAL_STATUSES:
            raise ValueError("a terminal task cannot be budget-paused")
        if expected_owner_wait is not None and current.get("owner_wait") != expected_owner_wait:
            raise BudgetPauseSuperseded("sleep checkpoint changed before retention")
        old = current.get("budget_pause") or {}
        if expected_pause_id is not None and str(old.get("pause_id") or "") != expected_pause_id:
            raise BudgetPauseSuperseded("budget pause identity changed")
        if expected_states is not None and str(old.get("state") or "") not in expected_states:
            raise BudgetPauseSuperseded(f"budget pause state changed: {old.get('state')!r} is not {sorted(expected_states)}")
        if expected_grant_id is not None and str((old.get("grant") or {}).get("grant_id") or "") != str(expected_grant_id):
            raise BudgetPauseSuperseded("budget pause grant changed")
        retained_wait = ({"owner_wait": {**expected_owner_wait, "state": "retained"}}
                         if expected_owner_wait is not None else {})
        from ouroboros.pause_notices import notice_fields
        notice = (notice_fields(current, root, task_id, str(row.get("pause_id") or ""), "budget")
                  if row.get("state") == STATE_PAUSED and row.get("reason") == "budget"
                  and (old.get("state") != STATE_PAUSED or old.get("pause_id") != row.get("pause_id")) else {})
        return stamp_task_result_schema({**current, **retained_wait, **notice, "budget_pause": dict(row)})

    update_json_locked(task_result_path(root, task_id), update, strict_existing_dict=True)
    if row.get("state") == STATE_PAUSED and row.get("reason") == "budget":
        from ouroboros.pause_notices import track
        track(root, task_id)
    return dict(row)


def budget_pause_row(root: Any, task_id: str) -> Dict[str, Any]:
    from ouroboros.task_results import load_task_result

    row = load_task_result(pathlib.Path(root), str(task_id), strict=True) or {}
    pause = row.get("budget_pause")
    return dict(pause) if isinstance(pause, dict) else {}


# --- the pause itself -----------------------------------------------------------------

# How often a HOLD re-reads the task's EXISTING controls and re-observes its
# producers. A poll interval, not a timeout and not a settings key: nothing
# ends because of it, and no clock of this module's own bounds the hold.
_HOLD_POLL_SEC = 0.5

HOLD_PRODUCERS_UNSETTLED = "local_producers_unsettled"
HOLD_PAUSE_RECORD_UNWRITABLE = "pause_record_unwritable"
HOLD_CHECKPOINT_UNWRITABLE = "continuation_source_unwritable"
HOLD_COLD_WRITERS_UNSETTLED = "cold_sleep_writers_unsettled"
# A Resume whose grant consumption could not be published: the grant and the
# checkpoint stay exactly as the durable row carries them; the task HOLDS.
HOLD_GRANT_CONSUMPTION_UNWRITABLE = "resume_grant_consumption_unwritable"
# A ``pausing`` row the task's own controls ended before it became a pause.
STATE_ABANDONED = "abandoned"


class ColdSleepWritersUnsettled(ValueError):
    """A fresh cold-sleep census still has writers; no source write failed."""


class ColdSleepDependencyBlocked(ValueError):
    """The proposed sleep fence prevents a selected queued dependency running."""


def _hold_row(reason: str, *, exc: Optional[BaseException] = None,
              detail: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """One typed hold fact: WHY this task is still fenced and nonterminal."""
    row: Dict[str, Any] = {"hold_reason": reason, "observed_at": time.time()}
    if exc is not None:
        row["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    if detail is not None:
        row["unsettled"] = {
            "review_attempts": (detail.get("review_attempts") or {}).get("unsettled") or [],
            "tool_futures": (detail.get("tool_futures") or {}).get("unsettled") or [],
            "observation_error": str(detail.get("observation_error") or ""),
        }
    return row


# The control families that end a HOLD, named exactly: owner Stop/cancel, Panic,
# an explicit deadline and the finite lifetime. A "finalize now" request or a
# closed wait is NEITHER a stop NOR a bound, and must not abort a pause the
# fence has already committed to — the checkpoint is what preserves that work.
_HOLD_ENDING_CONTROLS = frozenset(
    {"cancelled", "panic", "deadline", "execution_deadline", "absolute_ceiling"})


def _hold_control_reason(ctx: Any) -> str:
    """The task's EXISTING controls, read during a hold. This adds none of its own.

    A live model wait already owns Stop/Panic, cancel intent, owner deadlines
    and the finite lifetime; its reader answers first. The owner-stop flags and
    cancel intents the restore gate reads are consulted either way, so a Stop
    still lands when no wait is bound or when the wait reports something else.
    """
    try:
        from ouroboros.model_wait import current_model_wait

        waiter = current_model_wait()
        if waiter is not None:
            reason = str(waiter.control_reason() or "")
            if reason in _HOLD_ENDING_CONTROLS:
                return reason
    except Exception:
        log.debug("Model-wait control reader unavailable during a budget pause hold", exc_info=True)
    try:
        root = pathlib.Path(getattr(ctx, "budget_drive_root", None) or ctx.drive_root)
        if any((root / "state" / name).exists()
               for name in ("panic_stop.flag", "owner_restart_no_resume.flag")):
            return "panic"
        from ouroboros.cancel_intents import cancel_pending

        if cancel_pending(root, str(ctx.task_id)):
            return "cancelled"
    except Exception:
        log.debug("Owner stop readers unavailable during a budget pause hold", exc_info=True)
    return ""


def _publish_hold(ctx: Any, row: Dict[str, Any]) -> None:
    """Announce one hold transition on the EXISTING worker->supervisor path.

    ``task_checkpoint`` is forwarded live and persisted to ``events.jsonl`` by
    the supervisor's nested log branch, so a retained hold reaches the owner as
    a fact instead of reading as a silent stall. Called on CHANGE only: a poll
    interval is not a ledger cadence.
    """
    try:
        from ouroboros.loop_messages import _emit_checkpoint_event

        task_id = str(ctx.task_id)
        _emit_checkpoint_event(
            getattr(ctx, "event_queue", None), task_id,
            pathlib.Path(ctx.drive_root) / "logs",
            {"checkpoint_kind": "budget_pause_hold", "owner_visible": True,
             "toast_once": f"{task_id}:budget-pause-hold:{row.get('hold_reason') or row.get('ended_by') or ''}",
             **row},
        )
    except Exception:
        log.debug("Budget pause hold could not be published for %s",
                  getattr(ctx, "task_id", ""), exc_info=True)


def _exact_continuation_row(limit_ctx: Any, ctx: Any, *, pause_id: str, rail: str, scope: str,
                            reason_text: str, root_task_id: str,
                            local: Dict[str, Any]) -> Dict[str, Any]:
    """Observe external custody, store the ONE continuation source, build the row."""
    from ouroboros.owner_wait import continuation_state, store_continuation_source

    if rail == RAIL_MODEL_SLEEP:
        from ouroboros.model_sleep import cold_blockers

        blockers = cold_blockers(ctx)
        if any(blocker.get("kind") == "queued_member" for blocker in blockers):
            raise ColdSleepDependencyBlocked(f"selected queued dependency: {blockers}")
        if blockers:
            raise ColdSleepWritersUnsettled(f"{HOLD_COLD_WRITERS_UNSETTLED}: {blockers}")
    usage = limit_ctx.accumulated_usage
    # Owner Q8, loop side: resolve THIS context's custody root, then share the
    # one observer body with the grant. A root that cannot be resolved is the
    # same typed ``custody_read=failed`` fact as an unreadable store.
    root, read_error = "", ""
    try:
        from ouroboros import delegate_custody as custody

        root = custody.custody_root(ctx)
    except Exception as exc:
        log.warning("External custody root unresolvable at budget pause", exc_info=True)
        read_error = f"{type(exc).__name__}: {str(exc)[:200]}"
    owner = rail == RAIL_OWNER_PAUSE
    reason = _RAIL_REASONS.get(rail, "budget")
    external = observe_task_runs(root, str(ctx.task_id), reason="budget_pause_uncovered_cost",
                                 read_error=read_error, request_stop=reason == "budget")
    messages = limit_ctx.messages
    trace = limit_ctx.llm_trace if isinstance(limit_ctx.llm_trace, dict) else {}
    seen = set(limit_ctx.owner_msg_seen or ())
    pending = pending_tool_call_ids(messages)
    point = {"round_idx": int(limit_ctx.round_idx),
             "phase": "partial_tool_batch_unknown" if pending else "boundary",
             "unanswered_tool_call_ids": pending,
             "unanswered_policy": "not_re_executed_execution_unknown",
             "budget_tail": getattr(limit_ctx, "budget_tail", "tool")}
    # The rail already stamped its terminal projection on the live usage, and a
    # hold leaves its own transient row there; neither may travel into the
    # resumed loop's eventual honest terminal.
    usage_for_state = {key: value for key, value in usage.items()
                       if key not in ("execution_status", "reason_code",
                                      "_best_effort_extracted", "budget_pause_hold")}
    state = {
        **continuation_state(ctx, messages, trace, usage_for_state, limit_ctx.round_idx,
                             list(limit_ctx.tool_schemas or []), seen),
        "pause_id": pause_id, "reason": reason, "rail": rail, "scope": scope,
        "resume_point": point, "external_runs": external, "local_producers": local,
    }
    source = store_continuation_source(ctx, state, "budget-pause-" + pause_id)
    cost_ceiling = getattr(ctx, "_cost_ceiling", None)
    physical_calls = None
    try:
        from ouroboros.usage_accounting import usage_breakdown

        budget_root = getattr(ctx, "budget_drive_root", None) or ctx.drive_root
        breakdown = usage_breakdown(pathlib.Path(budget_root), task_id=str(ctx.task_id))
        if not breakdown.get("integrity_degraded"):
            physical_calls = int(breakdown.get("physical_calls") or 0)
    except Exception:
        log.debug("Physical call count unavailable at budget pause", exc_info=True)
    settlement = {}
    if owner:
        # Never a false Paused: a checkpoint over work still running elsewhere
        # is an intermediate save; the park keeps it ``pausing`` until the sent
        # work settles (an unreadable custody store is unsettled too).
        open_runs = external.get("custody_read") != "ok" or bool(external.get("runs"))
        from ouroboros.owner_pause import SETTLEMENT_EXTERNAL_RUNNING, SETTLEMENT_SETTLED

        settlement = {"settlement": SETTLEMENT_EXTERNAL_RUNNING if open_runs else SETTLEMENT_SETTLED,
                      "owner_fence_id": str(getattr(ctx, "_owner_pause_fence_id", "") or "")}
    sleep = getattr(ctx, "_model_sleep", None) if reason == REASON_SLEEP else None
    if isinstance(sleep, dict):
        # What the supervisor's sleep-wake policy reads: the selected sources and
        # the mailbox ids this loop had already seen (never a second latch).
        settlement = {"sleep": dict(sleep), "sleep_seen": sorted(set(limit_ctx.owner_msg_seen or ()))}
    return {
        "pause_id": pause_id, "state": STATE_PAUSING, "reason": reason,
        **settlement,
        "pause_generation": int(getattr(ctx, "_budget_pause_generation", 0) or 0),
        "rail": rail, "scope": str(scope or "global"),
        "root_task_id": str(root_task_id or getattr(ctx, "root_task_id", "") or ""),
        "reason_text": str(reason_text or ""),
        "task_attempt": int(ctx.task_attempt or 1),
        "is_direct_chat": bool(getattr(ctx, "is_direct_chat", False)),
        "source_ref": source,
        "execution_drive_root": str(ctx.drive_root),
        "started_at": getattr(ctx, "task_started_at", None),
        "paused_at": time.time(),
        "paused_duration_sec": float(getattr(ctx, "_budget_paused_sec", 0.0) or 0.0),
        "resume_point": point,
        "cost_ceiling": asdict(cost_ceiling) if cost_ceiling is not None else None,
        "external_runs": external, "local_producers": local,
        "physical_calls": physical_calls,
        "exact_continuation": True, "replay_safe": False, "auto_resume": False,
        "resume_policy": RESUME_POLICY,
        "model_wait_quota_clock": (state.get("model_wait") or {}).get("quota_clock", {}),
    }


def request_pause(limit_ctx: Any, *, rail: str, scope: str, reason_text: str,
                  root_task_id: str = "") -> None:
    """Enter the durable pause and unwind the loop; returns only when INELIGIBLE.

    Once the fence closes, the task is committed: it stays fenced and
    NONTERMINAL until BOTH its local producers are quiescent AND the exact
    continuation source is stored. Neither an unsettled producer nor a failed
    write returns it to the old paid terminal rail — the task HOLDS instead,
    keeping its native worker and its durable ``pausing`` state, publishing a
    typed ``hold_reason`` to the owner, and buying no wrap-up call. A failed
    write is a visible retained hold, never a claimed durable pause. The task's
    existing Stop/Panic/deadline/cancel controls stay responsive throughout and
    end a hold without a pause. A cold sleep discovered to block its selected
    queued dependency instead withdraws its exact sleep seed and stays awake.
    """
    tools = getattr(limit_ctx, "tools", None)
    ctx = getattr(tools, "_ctx", None)
    if ctx is None or rail not in RAILS:
        return None
    # A context without a continuation owner, a task id or a durable root is
    # excluded LOUDLY: the typed reason lands on the terminal ``resource_limit``.
    # A direct owner-chat actor IS eligible — its continuation owner is the queue
    # carrier the supervisor parks its record in (the direct lane binds
    # ``direct_owner_wait`` to the same wait seam a pooled worker binds
    # ``worker_owner_wait`` to).
    ineligible = ("no_continuation_owner" if not callable(getattr(ctx, "owner_wait_callback", None))
                  else "no_task_id" if not str(getattr(ctx, "task_id", "") or "")
                  else "no_durable_root" if not (getattr(ctx, "budget_drive_root", None)
                                                 or getattr(ctx, "drive_root", None))
                  else "")
    usage = limit_ctx.accumulated_usage
    if ineligible:
        usage["exact_pause_unavailable"] = ineligible
        return None
    task_id = str(ctx.task_id)
    begin_dispatch_fence(task_id)
    setattr(ctx, "_budget_pausing", True)
    pause_id = uuid.uuid4().hex
    root = pathlib.Path(ctx.budget_drive_root or ctx.drive_root)
    # A direct owner-chat turn has no admission-written RUNNING row (the pool
    # mirrors one at dispatch; the direct lane writes a stub only for a turn
    # with an origin ref). The pause row is a projection ON the task result, so
    # an absent row is written as a plain RUNNING row first — merge-write, never
    # a replacement of an existing one. A failed write is not swallowed into a
    # fake pause, and an unreadable row is not ours to guess: the seed write
    # below meets the same fault and HOLDS.
    from ouroboros.task_results import STATUS_RUNNING, load_task_result, write_task_result

    try:
        pausable_row = bool(load_task_result(root, task_id, strict=False))
    except Exception:
        pausable_row = True
    if not pausable_row:
        try:
            chat_id = getattr(ctx, "current_chat_id", None)
            write_task_result(
                root, task_id, STATUS_RUNNING,
                **({"chat_id": int(chat_id)} if chat_id not in (None, "") else {}),
                _is_direct_chat=bool(getattr(ctx, "is_direct_chat", False)),
                result="Task is pausing on its budget rail; its exact continuation is being stored.",
            )
        except Exception:
            log.debug("Pausable result row for %s could not be written ahead of the seed", task_id, exc_info=True)
    # One monotonic generation per pause of this task (pauseA=1, pauseB=2, ...):
    # every grant, handoff and revocation names the generation beside the pause
    # id, so a carrier from an earlier pause can never read as the current one.
    # An unreadable row yields generation 1 with no claim about the past: the
    # seed write below meets the same fault and HOLDS.
    try:
        generation = int(budget_pause_row(root, task_id).get("pause_generation") or 0) + 1
    except Exception:
        generation = 1
    setattr(ctx, "_budget_pause_generation", generation)
    # Durable "pausing" FIRST (before any wait), so a death while the task is
    # still settling meets the crash-retry fence instead of a replay.
    seed = {"pause_id": pause_id, "state": STATE_PAUSING,
            "reason": _RAIL_REASONS.get(rail, "budget"), "rail": rail,
            "pause_generation": generation,
            "scope": str(scope or "global"), "task_attempt": int(ctx.task_attempt or 1),
            "source_ref": None, "started_at": getattr(ctx, "task_started_at", None),
            "pausing_since": time.time(), "exact_continuation": True,
            "replay_safe": False, "auto_resume": False}
    row: Dict[str, Any] = {}
    hold: Dict[str, Any] = {}
    prepared: Optional[Dict[str, Any]] = None
    published = ""
    opened = False
    while True:
        control = _hold_control_reason(ctx)
        if control:
            # The owner's own control, not a monetary decision. No durable pause
            # was reached, so the opened row is closed as abandoned (typed, never
            # silent) and the task leaves on its existing control rail. The fence
            # stays CLOSED: a stopped task does not buy a wrap-up call out of this path.
            try:
                current = budget_pause_row(root, task_id)
                if current.get("pause_id") == pause_id:
                    set_budget_pause(root, task_id, {**current, "state": STATE_ABANDONED,
                                                     "abandon_reason": f"hold_ended_by_control:{control}",
                                                     "abandon_detail": hold, "abandoned_at": time.time()},
                                     expected_pause_id=pause_id)
            except Exception:
                log.warning("Abandoned pausing row for %s could not be closed", task_id, exc_info=True)
            setattr(ctx, "_budget_pausing", False)
            usage["exact_pause_unavailable"] = "hold_ended_by_control"
            usage["budget_pause_hold"] = {**hold, "ended_by": control}
            _publish_hold(ctx, {"pause_id": pause_id, "rail": rail, "state": "hold_ended",
                                "hold_reason": str(hold.get("hold_reason") or ""),
                                "ended_by": control})
            from ouroboros.model_wait import ModelWaitInterrupted

            raise ModelWaitInterrupted(control)
        if not opened:
            try:
                from ouroboros.owner_pause import launch_lock

                # Close durable cold-tree admission before proving quiescence.
                # The lock ends before any process/remote observation or wait.
                with launch_lock(root, str(root_task_id or task_id)):
                    set_budget_pause(root, task_id, seed)
                opened = True
            except Exception as exc:
                hold = _hold_row(HOLD_PAUSE_RECORD_UNWRITABLE, exc=exc)
        if opened:
            try:
                local = local_producer_observation(ctx, timeout_sec=_HOLD_POLL_SEC)
            except Exception as exc:
                # An observation that raised proves nothing about quiescence.
                local = {"quiescent": False, "observation_error": f"{type(exc).__name__}: {exc}"}
            if not local.get("quiescent"):
                # A snapshot prepared under an earlier quiescence is stale once a
                # producer is live again: the next quiescence re-observes custody.
                prepared = None
                hold = _hold_row(HOLD_PRODUCERS_UNSETTLED, detail=local)
            else:
                try:
                    if prepared is None:
                        prepared = _exact_continuation_row(
                            limit_ctx, ctx, pause_id=pause_id, rail=rail, scope=scope,
                            reason_text=reason_text, root_task_id=root_task_id, local=local)
                    # A publication that failed is retried with the SAME prepared
                    # snapshot: its custody stop requests and its stored source are
                    # one observation, not one per poll.
                    set_budget_pause(root, task_id, prepared, expected_pause_id=pause_id)
                    row = prepared
                    break
                except ColdSleepDependencyBlocked as exc:
                    try:
                        set_budget_pause(root, task_id, {**seed, "state": STATE_ABANDONED,
                            "abandon_reason": "cold_sleep_dependency_blocked", "abandoned_at": time.time()},
                            expected_pause_id=pause_id, expected_state=STATE_PAUSING)
                    except Exception as write_error:
                        hold = _hold_row(HOLD_PAUSE_RECORD_UNWRITABLE, exc=write_error)
                    else:
                        end_dispatch_fence(task_id)
                        ctx._budget_pausing, ctx._model_sleep = False, None
                        usage.pop("budget_pause_hold", None)
                        limit_ctx.messages.append({"role": "user", "content": (
                            "[SYSTEM NOTICE]\nCold sleep was not taken: " + str(exc)
                            + ". Sleep warm to let the selected child run; no child was cancelled.")})
                        return
                except ColdSleepWritersUnsettled as exc:
                    hold = _hold_row(HOLD_COLD_WRITERS_UNSETTLED, exc=exc)
                except Exception as exc:
                    hold = _hold_row(HOLD_CHECKPOINT_UNWRITABLE, exc=exc)
        usage["budget_pause_hold"] = dict(hold)
        if str(hold.get("hold_reason") or "") != published:
            # On CHANGE only: a poll interval is neither a log nor a ledger cadence.
            published = str(hold.get("hold_reason") or "")
            log.warning("Budget pause for %s is HELD (fenced, nonterminal): %s %s",
                        task_id, published, hold.get("error") or "")
            _publish_hold(ctx, {"pause_id": pause_id, "rail": rail,
                                "state": STATE_PAUSING, **hold})
        time.sleep(_HOLD_POLL_SEC)
    usage.pop("budget_pause_hold", None)
    usage["reason_code"] = {RAIL_OWNER_PAUSE: OWNER_REASON_CODE, RAIL_MODEL_SLEEP: "model_sleeping"}.get(
        rail, REASON_CODE)
    usage["execution_status"] = "paused"
    usage["budget_pause"] = {key: row[key] for key in ("pause_id", "rail", "scope", "paused_at", "resume_point")}
    raise BudgetPauseRequested(row)


def enter_owner_pause(limit_ctx: Any) -> None:
    """A member's safe boundary under its root's closed owner fence (``owner_pause``).

    No model round is bought: the member enters the SAME exact pause the
    budget rails write, under the owner rail — sent work observed, never
    cancelled — and never returns when that pause is saved. A member that
    cannot save an exact pause (no continuation owner or durable root) HOLDS
    warm at this boundary instead, buying nothing, until the root's Resume
    releases the fence or its own Stop/Panic/deadline ends the hold. Returns
    only when the fence is open.
    """
    from ouroboros.owner_pause import member_fence

    ctx = getattr(getattr(limit_ctx, "tools", None), "_ctx", None)
    fence = member_fence(ctx) if ctx is not None else {}
    if not fence:
        return
    if fence.get("state") == "unknown":
        from ouroboros.model_wait import ModelWaitInterrupted

        raise ModelWaitInterrupted("owner_pause_authority_unreadable")
    setattr(ctx, "_owner_pause_fence_id", str(fence.get("fence_id") or ""))
    request_pause(limit_ctx, rail=RAIL_OWNER_PAUSE, scope="root",
                  reason_text="The owner paused this task tree.",
                  root_task_id=str(getattr(ctx, "root_task_id", "") or getattr(ctx, "task_id", "") or ""))
    usage = limit_ctx.accumulated_usage
    published = False
    while member_fence(ctx):
        control = _hold_control_reason(ctx)
        if control:
            from ouroboros.model_wait import ModelWaitInterrupted

            raise ModelWaitInterrupted(control)
        if not published:
            published = True
            usage["budget_pause_hold"] = _hold_row(
                "owner_pause_warm_hold", detail=None,
                exc=RuntimeError(str(usage.get("exact_pause_unavailable") or "exact_pause_unavailable")))
            _publish_hold(ctx, {"rail": RAIL_OWNER_PAUSE, "state": STATE_PAUSING,
                                **usage["budget_pause_hold"]})
        time.sleep(_HOLD_POLL_SEC)
    usage.pop("budget_pause_hold", None)


def enter_cold_sleep(limit_ctx: Any) -> None:
    """The round boundary after a COLD sleep was armed (``model_sleep``).

    The loop's own continuation is saved and the process parks under the same
    exact-pause carrier; nothing wakes it but its selected sources through the
    supervisor's sleep-wake policy. A writer of this task that appeared since
    the request (a delegated run, a live service) refuses the cold park here:
    the model is told the blockers and stays awake — never a killed helper.
    """
    ctx = getattr(getattr(limit_ctx, "tools", None), "_ctx", None)
    sleep = getattr(ctx, "_model_sleep", None) if ctx is not None else None
    if not isinstance(sleep, dict) or sleep.get("mode") != "cold":
        return
    from ouroboros.model_sleep import cold_blockers

    blockers = cold_blockers(ctx)
    if blockers:
        ctx._model_sleep = None
        limit_ctx.messages.append({"role": "user", "content": (
            "[SYSTEM NOTICE]\nYour cold sleep was not taken: this task still runs "
            + "; ".join(f"{b['kind']} {b.get('run_id') or b.get('name') or ''}" for b in blockers)
            + ". Sleep warm, wait for it, or stop it first.")})
        return
    request_pause(limit_ctx, rail=RAIL_MODEL_SLEEP, scope="task", reason_text="The model chose a cold sleep.",
                  root_task_id=str(getattr(ctx, "root_task_id", "") or getattr(ctx, "task_id", "") or ""))
    ctx._model_sleep = None  # no continuation owner: the ordinary loop continues awake


# Fields of a direct turn's task record that never travel in its pause event:
# the inline image bytes were consumed into the checkpointed transcript, and a
# multi-megabyte base64 blob has no business in the supervisor's event queue or
# the queue snapshot (which whitelists its own fields anyway).
_DIRECT_TASK_EVENT_OMITTED = frozenset({"image_base64"})


def parkable_direct_task(task: Dict[str, Any]) -> Dict[str, Any]:
    """The direct turn's own task record, as the queue may park it (#1196).

    A direct turn is never in RUNNING, so the event is the only carrier of the
    record the SAME task id continues under: origin, chat, contract, metadata,
    workspace, attachments and the lane fact all ride it unchanged.
    """
    row = {key: value for key, value in task.items() if key not in _DIRECT_TASK_EVENT_OMITTED}
    row["_is_direct_chat"] = True
    row.setdefault("_attempt", int(task.get("_attempt") or 1))
    row.setdefault("depth", 0)
    return row


def pause_event(task: Dict[str, Any], pause: Dict[str, Any]) -> Dict[str, Any]:
    """The worker->supervisor ``budget_pause`` event for an exact continuation.

    A direct owner-chat turn's event additionally carries the turn's own task
    record (``task``) and the lane fact, because that turn was never in the
    queue's RUNNING table: the supervisor parks THAT record.
    """
    from ouroboros.utils import utc_now_iso

    task_id = str(task.get("id") or "")
    direct = bool(task.get("_is_direct_chat"))
    return {
        "type": "budget_pause",
        "task_id": task_id,
        "task_type": str(task.get("type") or "task"),
        "worker_id": task.get("worker_id"),
        "chat_id": task.get("chat_id"),
        "root_task_id": str(pause.get("root_task_id") or task.get("root_task_id") or task_id),
        "resource_limit": exact_pause_marker(pause, default_root=str(task.get("root_task_id") or task_id)),
        **({"_is_direct_chat": True, "task": parkable_direct_task(task)} if direct else {}),
        "ts": utc_now_iso(),
    }


STATUS_PAUSED_EXACT = "paused_exact_continuation"


def exact_pause_marker(row: Dict[str, Any], *, default_root: str = "") -> Dict[str, Any]:
    """The queue's ``_budget_pause`` marker projected from the DURABLE pause row.

    The row on the task result is the source of truth; the marker is its
    locator inside the queue (and the worker event's ``resource_limit``).
    """
    return {
        "status": STATUS_PAUSED_EXACT,
        "scope": str(row.get("scope") or "global"),
        "root_task_id": str(row.get("root_task_id") or default_root or ""),
        "physical_calls": row.get("physical_calls"),
        "replay_safe": False,
        "exact_continuation": True,
        "auto_resume": False,
        "resume_policy": RESUME_POLICY,
        "rail": str(row.get("rail") or ""),
        # Which pause this is (``budget`` or the owner's) and, for the owner's,
        # whether its sent work had settled: the Resume and the UI read it.
        "reason": str(row.get("reason") or "budget"),
        **({"settlement": row["settlement"]} if row.get("settlement") else {}),
        **({"sleep": row["sleep"]} if isinstance(row.get("sleep"), dict) else {}),
        "paused_at": row.get("paused_at"),
        "checkpoint": {
            key: row.get(key) for key in (
                "pause_id", "pause_generation", "task_attempt", "source_ref", "execution_drive_root",
                "started_at", "paused_at", "paused_duration_sec", "resume_point",
                "model_wait_quota_clock", "external_runs", "rail", "scope", "reason_text",
            )
        },
    }


# --- resume (loop side) -----------------------------------------------------------------

def load_budget_pause(ctx: Any, handoff: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Consume ONE granted resume: the row must carry this exact single-use grant.

    Refuses (raises) a missing or corrupt source, a spent or foreign grant and
    an attempt mismatch: a stale snapshot must never revive a continuation.
    """
    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros.task_results import _TRULY_TERMINAL_STATUSES, load_task_result

    handoff = handoff or getattr(ctx, "budget_pause_resume", None)
    if not handoff:
        return {}
    root = pathlib.Path(ctx.budget_drive_root or ctx.drive_root)
    row = load_task_result(root, ctx.task_id, strict=True) or {}
    current = row.get("budget_pause") or {}
    grant = current.get("grant") or {}
    if (row.get("status") in _TRULY_TERMINAL_STATUSES
            or current.get("state") != STATE_RESUME_GRANTED
            or current.get("pause_id") != handoff.get("pause_id")
            or int(current.get("pause_generation") or 0) != int(handoff.get("pause_generation") or 0)
            or not grant.get("grant_id")
            or grant.get("grant_id") != handoff.get("grant_id")
            or int(grant.get("generation") or 0) != int(handoff.get("grant_generation") or grant.get("generation") or 0)
            or grant.get("consumed_at") or grant.get("revoked_at")):
        raise ValueError("budget pause continuation has no live single-use grant")
    state = json.loads(read_actor_source_bytes(root, ctx.task_id, current["source_ref"]))
    if (state.get("task_id") != ctx.task_id
            or state.get("pause_id") != current.get("pause_id")
            or int(state.get("task_attempt") or 0) != int(ctx.task_attempt or 1)):
        raise ValueError("budget pause continuation identity mismatch")
    return {**state, "_pause_row": dict(current)}


def _refresh_planning_threshold(ctx: Any, budget_remaining_usd: Optional[float],
                                usage: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Owner Q10: after an explicit Resume of a GRACEFUL stop, the planning
    threshold moves forward within the money still authorized, so the task is
    not paused again on the very number that paused it. The hard tree cap and
    the global ledger fence are untouched; the planning margin is what the
    owner's explicit act spends. Returns the disclosure row.

    Every number is read from the AUTHORITATIVE ledger NOW: the global wallet
    from the usage projection, the tree's cumulative spend and its ACTUAL
    root cap from a fresh root-accounting read (an owner may have raised the
    cap while the task was paused), the task's own cumulative cost from the
    restored usage. A wallet the ledger cannot answer, a degraded tree read
    or one the ledger could not answer NOW (the strict read never answers
    from the display cache, so a snapshot cached before a failed read is not
    room), or unknown spend REFUSES the refresh (the dispatch-time
    ``budget_remaining_usd`` is disclosed, never used as room): unknown money
    is not room.
    """
    from ouroboros import task_pacing
    from ouroboros.loop_budget import _loop_tree_accounting, _wrapup_global_remaining

    old = getattr(ctx, "_cost_ceiling", None)
    if not isinstance(old, task_pacing.CostCeiling):
        return {"refreshed": False, "reason": "no_ceiling"}
    try:
        fresh = _wrapup_global_remaining()
    except Exception:
        log.debug("Ledger wallet read failed at resume refresh", exc_info=True)
        fresh = None
    if fresh is None:
        return {"refreshed": False, "reason": "wallet_unavailable", "wallet_basis": "ledger_unavailable",
                "dispatch_time_remaining_usd": budget_remaining_usd}
    tree = _loop_tree_accounting(refresh=True, max_age_sec=0.0, strict=True)
    tree = tree if isinstance(tree, dict) else None
    tree_cap = tree.get("root_limit_usd") if tree else None
    tree_capped = old.root_cap_usd is not None or tree_cap is not None
    root_cap: Optional[float] = None
    root_cap_basis = "none"
    if tree_capped:
        if tree is None:
            return {"refreshed": False, "reason": "tree_spend_unavailable", "wallet_basis": "ledger_projection"}
        if tree.get("integrity_degraded"):
            return {"refreshed": False, "reason": "tree_accounting_degraded", "wallet_basis": "ledger_projection"}
        if tree.get("accounted_usd") is None:
            return {"refreshed": False, "reason": "tree_spend_unknown", "wallet_basis": "ledger_projection"}
        if tree_cap is not None:
            root_cap, root_cap_basis = float(tree_cap), "root_accounting"
        else:
            root_cap, root_cap_basis = float(old.root_cap_usd), "start_of_task"
    usage = usage if isinstance(usage, dict) else (getattr(ctx, "_accumulated_usage", None) or {})
    task_cost = usage.get("cost")
    deciding, basis = task_pacing.resolve_deciding_spend(
        tree_cost_usd=tree.get("accounted_usd") if tree else None,
        task_cost_usd=float(task_cost) if task_cost is not None else None,
        root_cap_usd=root_cap,
    )
    if deciding is None:
        return {"refreshed": False, "reason": "spend_unknown", "basis": basis, "wallet_basis": "ledger_projection"}
    spent = float(deciding)
    components: List[float] = []
    if root_cap is not None:
        components.append(root_cap - spent)
    if float(fresh) > 0:
        profile = task_pacing.resolve_budget_profile(ctx)
        pct = profile.get("cost_hard_stop_pct")
        pct = task_pacing._DEFAULT_COST_HARD_STOP_PCT if pct is None else max(0, min(100, int(pct)))
        if pct > 0:
            components.append(float(fresh) * pct / 100.0)
    room = min(components) if components else None
    if room is None or room <= 0:
        return {"refreshed": False, "reason": "no_authorized_room", "spent_usd": spent, "basis": basis,
                "wallet_basis": "ledger_projection", "global_remaining_usd": float(fresh),
                "root_cap_usd": root_cap, "root_cap_basis": root_cap_basis}
    refreshed = task_pacing.CostCeiling(
        state=task_pacing.COST_CEILING_ACTIVE, ceiling_usd=spent + room,
        root_cap_usd=root_cap, planning_margin_usd=old.planning_margin_usd,
        basis=f"owner_resume_refresh({old.basis or 'previous'})",
    )
    ctx._cost_ceiling = refreshed
    return {"refreshed": True, "previous_ceiling_usd": old.ceiling_usd,
            "ceiling_usd": refreshed.ceiling_usd, "spent_usd": spent, "basis": basis,
            "wallet_basis": "ledger_projection", "global_remaining_usd": float(fresh),
            "root_cap_usd": root_cap, "root_cap_basis": root_cap_basis}


def _reopen_owner_fence(ctx: Any, usage: Dict[str, Any], *, fence_id: str = "", grant_id: str = "") -> None:
    """Consume an explicit Resume's owner-fence authority where work starts.

    A root reopens its tree; an owner-selected child of a terminal root opens
    only its own member under that exact fence; a newer owner Pause's fence stays
    closed (``reopen_for_resume``). Failed publication HOLDS and retries; it
    never starts effects under missing authority.
    """
    from ouroboros.owner_pause import reopen_for_resume

    root_id = str(getattr(ctx, "root_task_id", "") or ctx.task_id)
    if root_id != str(ctx.task_id) and not fence_id:
        return
    root = pathlib.Path(ctx.budget_drive_root or ctx.drive_root)
    published = ""
    while True:
        try:
            reopen_for_resume(root, root_id, str(ctx.task_id), fence_id=fence_id, grant_id=grant_id)
            usage.pop("budget_pause_hold", None)
            return
        except Exception as exc:
            hold = _hold_row("owner_pause_fence_release_unwritable", exc=exc)
            usage["budget_pause_hold"] = dict(hold)
            if str(hold.get("error") or "") != published:
                published = str(hold.get("error") or "")
                _publish_hold(ctx, {"rail": RAIL_OWNER_PAUSE, "state": STATE_RESUMED, **hold})
            control = _hold_control_reason(ctx)
            if control:
                from ouroboros.model_wait import ModelWaitInterrupted

                raise ModelWaitInterrupted(control) from exc
            time.sleep(_HOLD_POLL_SEC)


def resume_paused_loop(tools: Any, state: Dict[str, Any], messages: list, trace: dict,
                       usage: dict, seen: set, *, budget_remaining_usd: Optional[float]) -> tuple:
    """Restore the paused cognition, consume the grant, disclose drift/custody.

    Returns ``(model, effort, use_local, mode, round_idx, plan)``. Cognition
    only: a tool call of the interrupted batch that has no recorded result is
    closed with a host row stating its execution is UNKNOWN — nothing is
    re-executed by the host, and the model is told to verify before repeating.
    """
    from ouroboros.owner_wait import rebind_restored_route, restore_continuation_state

    ctx = tools._ctx
    restore_continuation_state(tools, state, messages, trace, usage, seen)
    row = state.get("_pause_row") or {}
    root = pathlib.Path(ctx.budget_drive_root or ctx.drive_root)
    grant = dict(row.get("grant") or {})
    published = ""
    while True:
        grant["consumed_at"] = time.time()
        paused_sec = float(grant.get("paused_duration_sec") or row.get("paused_duration_sec") or 0.0)
        if row.get("reason") == REASON_SLEEP:
            paused_sec += max(0.0, grant["consumed_at"] - float(grant.get("granted_at_ts") or grant["consumed_at"]))
        consumed = {**row, "state": STATE_RESUMED, "grant": grant,
                    "resumed_at": grant["consumed_at"], "paused_duration_sec": paused_sec}
        # Compare-and-set on the exact pause, state and grant this worker was
        # handed: a grant revoked or superseded between the dispatch and this
        # write refuses here instead of consuming a grant that is no longer live.
        try:
            set_budget_pause(root, ctx.task_id, consumed,
                             expected_pause_id=str(row.get("pause_id") or ""),
                             expected_state=STATE_RESUME_GRANTED,
                             expected_grant_id=str(grant.get("grant_id") or ""))
            break
        except Exception as exc:
            # A consumption that cannot be published is never a paid terminal
            # (#1196): the worker HOLDS, typed and nonterminal, retrying the SAME
            # publication so the grant and checkpoint identities stay exactly as
            # the durable row carries them. A row the durable authority already
            # returned to a live pause (a revocation landed first) re-parks under
            # it: there is no grant to consume and nothing to run on.
            try:
                current = budget_pause_row(root, ctx.task_id)
            except Exception:
                current = {}
            if (current.get("pause_id") == row.get("pause_id")
                    and current.get("state") in {STATE_PAUSING, STATE_PAUSED}
                    and int(current.get("task_attempt") or 0) == int(ctx.task_attempt or 1)
                    and current.get("source_ref")):
                usage.pop("budget_pause_hold", None)
                raise BudgetPauseRequested(current) from exc
            hold = _hold_row(HOLD_GRANT_CONSUMPTION_UNWRITABLE, exc=exc)
            usage["budget_pause_hold"] = dict(hold)
            if str(hold.get("error") or "") != published:
                published = str(hold.get("error") or "")
                log.warning("Budget resume for %s is HELD (nonterminal): grant consumption unpublished %s",
                            ctx.task_id, published)
                _publish_hold(ctx, {"pause_id": row.get("pause_id"), "rail": row.get("rail"),
                                    "state": STATE_RESUME_GRANTED, **hold})
            control = _hold_control_reason(ctx)
            if control:
                usage["exact_pause_unavailable"] = "hold_ended_by_control"
                usage["budget_pause_hold"] = {**hold, "ended_by": control}
                _publish_hold(ctx, {"pause_id": row.get("pause_id"), "rail": row.get("rail"),
                                    "state": "hold_ended", "hold_reason": hold["hold_reason"],
                                    "ended_by": control})
                from ouroboros.model_wait import ModelWaitInterrupted

                raise ModelWaitInterrupted(control) from exc
            time.sleep(_HOLD_POLL_SEC)
    usage.pop("budget_pause_hold", None)
    if str(row.get("reason") or "") == REASON_OWNER or grant.get("owner_pause_fence_id"):
        _reopen_owner_fence(ctx, usage, fence_id=str(grant.get("owner_pause_fence_id") or ""),
                            grant_id=str(grant.get("grant_id") or ""))
    ctx._budget_paused_sec = consumed["paused_duration_sec"]
    waiter = getattr(ctx, "model_wait_context", None)
    if waiter is not None:
        with waiter.lock:
            waiter.budget_paused_sec = ctx._budget_paused_sec
            waiter.sleep_started_monotonic = None
    if row.get("reason") == REASON_SLEEP and getattr(ctx, "event_queue", None) is not None:
        # The existing worker event channel, with durable identity revalidation.
        try:
            ctx.event_queue.put({"type": "budget_pause", "phase": "consumed",
                "task_id": str(ctx.task_id), "task_attempt": int(ctx.task_attempt or 1),
                "pause_id": row.get("pause_id"), "grant_id": grant.get("grant_id")})
        except Exception:
            log.warning("Sleep consumption event unavailable; supervisor rereads the durable grant", exc_info=True)
    ctx.budget_pause_resume = None
    setattr(ctx, "_budget_pause_generation", int(row.get("pause_generation") or 0))
    end_dispatch_fence(str(ctx.task_id))
    # The pause is over: its quiescence rows are spent observations of a worker
    # that no longer exists, and the resumed attempt registers its own.
    forget_tool_scope(ctx)
    setattr(ctx, "_budget_pausing", False)
    if state.get("cost_ceiling") is not None:
        from ouroboros.task_pacing import CostCeiling

        ctx._cost_ceiling = CostCeiling(**state["cost_ceiling"])
    refresh: Dict[str, Any] = {"refreshed": False, "reason": "hard_rail"}
    graceful = str(row.get("rail") or "") in GRACEFUL_RAILS
    if graceful:
        refresh = _refresh_planning_threshold(ctx, budget_remaining_usd, usage)
    # Owner Q10, the other half: the last-fit rail stops one reservation EARLY
    # (it wants room for two so a wrap-up call stays affordable). After an
    # explicit Resume of a graceful stop that early margin is exactly what the
    # owner spent, so the resumed task may place the one call that fits in the
    # already-authorized remainder; the ledger fence at the full cap still binds
    # every send. Hard rails keep both reservations.
    ctx._budget_resume_last_fit_relaxed = bool(graceful)
    usage["budget_pause_resume"] = {"pause_id": row.get("pause_id"), "grant_id": grant.get("grant_id"),
                                    "grant_generation": int(grant.get("generation") or 0),
                                    "paused_duration_sec": ctx._budget_paused_sec, "threshold_refresh": refresh,
                                    "last_fit_relaxed": bool(graceful)}
    plan, mode = rebind_restored_route(tools, state, messages)
    # The grant re-observed custody at Resume time and wrote that observation
    # on the row; the checkpoint's copy is the pause-time reading. Disclose the
    # freshest one the durable rows carry.
    fresh_observation = row.get("external_runs") if isinstance(row.get("external_runs"), dict) else None
    external_observation = fresh_observation if fresh_observation is not None else (
        state.get("external_runs") or {})
    # Name WHICH reading this is: the grant's re-observation and the checkpoint's
    # pause-time copy license different amounts of trust, and a list labelled as
    # pause-time history reads as stale even when it is the fresh one.
    external_basis = ("re-observed at this Resume" if fresh_observation is not None
                      else "as recorded at the pause, NOT re-observed")
    external = external_observation.get("runs") or []
    external_lines = "".join(
        f"\n- run {run.get('run_id')}: {run.get('state')} (stop outcome: {run.get('stop_outcome') or 'n/a'})"
        for run in external if isinstance(run, dict)
    ) or "\n- none"
    pending = pending_tool_call_ids(messages)
    if str(row.get("reason") or "") == REASON_SLEEP and not pending:
        from ouroboros.model_sleep import wake_notice

        slept = max(0.0, float(grant["consumed_at"]) - float(row.get("paused_at") or grant["consumed_at"]))
        messages.append(wake_notice({**(row.get("sleep") or {}), "mode": "cold"},
                                    str((row.get("sleep_ready") or {}).get("reason") or ""), slept))
        return (ctx.active_model, ctx.active_effort, ctx.active_use_local,
                mode, int(state["round_idx"]), plan)
    kind = "owner Pause" if str(row.get("reason") or "") == REASON_OWNER else "budget pause"
    for call_id in pending:
        # Transcript validity for the provider AND the honest fact: unknown, not
        # "did not run" and not "ran". The host never re-executes it.
        messages.append({"role": "tool", "tool_call_id": call_id, "content": (
            f"[HOST NOTICE] No result for this tool call was recorded before the {kind}. "
            "Its execution state is UNKNOWN: it may have run and produced effects, or not run at all. "
            "It was NOT re-executed. Verify from authoritative state (files, git, services, custody) "
            "before repeating it.")})
    messages.append({"role": "user", "content": (
        f"[SYSTEM NOTICE]\nThis task continued from its {kind} after an explicit owner Resume "
        f"(paused {ctx._budget_paused_sec:.0f}s; rail: {row.get('rail')}; planning threshold "
        f"{'refreshed to $%.2f' % refresh['ceiling_usd'] if refresh.get('refreshed') else 'not refreshed: ' + str(refresh.get('reason'))}). "
        "Cumulative spend, rounds and elapsed execution time were NOT reset. Prior tool results remain "
        "recorded; do not repeat completed effects. The pause ended the previous browser process and "
        "task-local services; their recorded results remain evidence, not proof they are still running. "
        "The workspace may have drifted while paused: re-read any file before building on it. "
        f"Delegated runs this task holds ({external_basis}):{external_lines}\n"
        "An unknown or merely requested stop is NOT proof of termination: never start a second writer "
        "over such a run; inspect its custody first. "
        + (f"The last tool batch was interrupted: {len(pending)} call(s) have no recorded result and were "
           "NOT re-executed (see the host rows above); their execution state is unknown."
           if pending else ""))})
    return (ctx.active_model, ctx.active_effort, ctx.active_use_local,
            mode, int(state["round_idx"]), plan)


# --- restore-after-restart gate (supervisor side) -------------------------------------------

RESTORE_REFUSAL_NOT_EXACT = "not_an_exact_pause"
RESTORE_REFUSAL_RECORD_UNREADABLE = "pause_record_unreadable"
RESTORE_REFUSAL_TASK_TERMINAL = "task_terminal"
RESTORE_REFUSAL_RECORD_MISSING = "pause_record_missing"
RESTORE_REFUSAL_IDENTITY_MISMATCH = "pause_identity_mismatch"
RESTORE_REFUSAL_SOURCE_MISSING = "pause_source_missing"
RESTORE_REFUSAL_SOURCE_UNREADABLE = "pause_source_unreadable"


def budget_pause_restore_refusal(root: Any, task: Dict[str, Any]) -> str:
    """Why a paused PENDING locator cannot be restored as dispatch-eligible; "" when it can.

    A paused row survives a restart and any snapshot age WITHOUT waking: the
    queue row is a locator, and the task-result authority plus the readable
    source are what make it restorable. Every refusal here is a fact about the
    durable authority behind the locator (unreadable, terminal, a different
    pause id, no source, an unreadable source) — never a cancellation and never
    a reason to drop the row: the caller HOLDS it, typed and visible, so the
    saved pause stays where the owner can see it. Stop/Panic and the no-resume
    flag are NOT restore refusals: they are Resume refusals, read at grant time
    by the same gate that reads them live, so a flag that clears later never
    leaves a perfectly restorable pause held for nothing.
    """
    from ouroboros.artifacts import read_actor_source_bytes
    from ouroboros.task_results import _TRULY_TERMINAL_STATUSES, load_task_result

    pause = task.get("_budget_pause") if isinstance(task, dict) else None
    if not isinstance(pause, dict) or not pause.get("exact_continuation"):
        return RESTORE_REFUSAL_NOT_EXACT
    checkpoint = pause.get("checkpoint") if isinstance(pause.get("checkpoint"), dict) else {}
    root = pathlib.Path(root)
    task_id = str(task.get("id") or "")
    try:
        row = load_task_result(pathlib.Path(task.get("budget_drive_root") or root), task_id, strict=True) or {}
    except Exception:
        return RESTORE_REFUSAL_RECORD_UNREADABLE
    current = row.get("budget_pause") if isinstance(row.get("budget_pause"), dict) else {}
    if row.get("status") in _TRULY_TERMINAL_STATUSES:
        return RESTORE_REFUSAL_TASK_TERMINAL
    if not current or current.get("state") not in LIVE_PAUSE_STATES:
        return RESTORE_REFUSAL_RECORD_MISSING
    if current.get("pause_id") != checkpoint.get("pause_id"):
        return RESTORE_REFUSAL_IDENTITY_MISMATCH
    if not current.get("source_ref"):
        return RESTORE_REFUSAL_SOURCE_MISSING
    try:
        read_actor_source_bytes(pathlib.Path(task.get("budget_drive_root") or root), task_id, current["source_ref"])
    except Exception:
        return RESTORE_REFUSAL_SOURCE_UNREADABLE
    return ""
