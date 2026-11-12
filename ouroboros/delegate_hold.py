"""Unknown-provider hold: wait for the LEAF, not the provider (nanny-leaf D1-min).

A configured-session nanny supervises its physical delegated leaves. When
one of its metered rounds dies ``provider_outcome_unknown`` (dispatched request,
no terminal provider fact — never resent, per custody doctrine) while that leaf
is alive and healthy, terminalizing the nanny used to cancel the leaf through
the cause-blind terminal cleanup. Instead, the round gate latches a durable
hold and the NEXT round top holds the task on its leaves without a model call:
a wake already there is published at once, else the host arms the task's own
warm sleep on those runs (``delegate_supervision.arm_hold_sleep``), which holds
no slot; owner mail and controls wake it as they wake any sleep. A meaningful
leaf wake resumes the task with a NEW round whose transcript carries the wake
receipt: materially new input, so the unknown request itself is never
re-transmitted (owner-ratified doctrine revision; see docs/ARCHITECTURE.md).
Control wakes (Stop, deadline, finalize_now) exit through the existing unknown
no-resend terminal with zero further provider calls.

Eligibility is per run: a configured-session nanny may hold on any owned
ordinary leaves with a positive non-terminal read and no containment fault.
Pending invocations remain independent custody obligations. A terminal run
alone cannot buy a fresh paid round from an unknown provider outcome.
"""

from __future__ import annotations

import json
import logging
import pathlib
import time
from typing import Any, Dict, List

from ouroboros import delegate_custody as custody
from ouroboros.config import NETWORK_WAIT_BACKOFF_START_SEC
from ouroboros.delegate_shared import delegate_payload
from ouroboros.delegate_supervision import (
    _TICK_SEC,
    UnknownHoldUnreadable,
    _control_wakes,
    acknowledge_pending_wake,
    arm_hold_sleep,
    clear_unknown_hold,
    hold_wake,
    read_unknown_hold,
    supervision_checkpoint,
    write_unknown_hold,
)
from ouroboros.utils import append_jsonl, utc_now_iso

log = logging.getLogger(__name__)

# Control wake types/kinds that must exit the hold through the no-call terminal
# instead of buying a paid judgment round (owner Q3=A).
_CONTROL_WAKE_TYPES = frozenset({"cancellation_intent", "deadline"})
_CONTROL_WAKE_KINDS = frozenset({"finalize_now"})
# Wait statuses that must NOT buy a paid resume round: a refusal or fault is a
# transport/daemon statement, not a leaf wake — no proof of a live leaf.
_NON_WAKE_STATUSES = frozenset({"refused", "fault"})
# Inter-cycle backoff floor: repeated unknown outcomes on resume rounds must not
# tighten into a paid dispatch loop. Deliberately BELOW the 60s idle-rail floor
# so this plain bounded sleep can never outlast the silence window (the interruptible
# owner surface is the hold's sleep right after it).
_HOLD_BACKOFF_CAP_SEC = 15.0
_ACK_ATTEMPTS = 3


def _emit_hold_event(
    drive_logs: pathlib.Path, *, task_id: str, phase: str, run_id: str,
    hold_cycles: int, detail: str = "", attempt_id: str = "",
) -> None:
    try:
        append_jsonl(pathlib.Path(drive_logs) / "events.jsonl", {
            "ts": utc_now_iso(), "type": "delegate_hold", "task_id": task_id,
            "phase": phase, "run_id": run_id, "hold_cycles": int(hold_cycles),
            **({"unknown_attempt_id": attempt_id} if attempt_id else {}),
            **({"detail": detail} if detail else {}),
        })
    except Exception:
        log.debug("Failed to append delegate_hold event", exc_info=True)


def _configured_session(ctx: Any) -> bool:
    if not bool(getattr(ctx, "exact_model_route", False)):
        return False
    meta = getattr(ctx, "task_metadata", {})
    meta = meta if isinstance(meta, dict) else {}
    return isinstance(meta.get("configured_subagent"), dict)


def _leaf_probe_live(ctx: Any, run_id: str) -> bool:
    """READ-ONLY liveness probe: one short engine poll of the run's own engine, no lease/settle/cancel.

    Requires a POSITIVE non-terminal engine state. A gateway failure, refusal,
    or state-less payload is not evidence of a live leaf (fail-closed)."""
    import contextlib

    try:
        from ouroboros import claudexor_daemon
        from ouroboros.delegate_progress import bounded_poll
        from ouroboros.execution_domain import run_gateway

        entry = custody.lookup(custody.custody_root(ctx), str(getattr(ctx, "task_id", "") or ""), run_id)[1]
        with contextlib.closing(run_gateway(entry, lambda: claudexor_daemon.ensure_owned_gateway(
                admission_wait_sec=0), timeout_sec=5.0)) as gateway:
            detail = bounded_poll(gateway, run_id, 5.0)
        state = str(custody.summary_of(detail).get("state") or "")
    except Exception:
        log.debug("Hold eligibility leaf probe failed", exc_info=True)
        return False
    return bool(state) and state not in custody.TERMINAL_STATES


def _eligible_live_runs(ctx: Any) -> list[str]:
    """Every owned ordinary run with positive live evidence; no cardinality ceiling."""
    mine = str(getattr(ctx, "task_id", "") or "")
    try:
        root = custody.custody_root(ctx)
        if custody.custody_log_unreadable(root):
            return []
        faults = {str(row.get("run_id") or "") for row in custody.open_containment_faults(root)}
        return [row.run_id for row in custody.open_runs(root) if row.task_id == mine and not row.review_owned
                and row.run_id and row.run_id not in faults and _leaf_probe_live(ctx, row.run_id)]
    except Exception:
        log.debug("Hold eligibility custody audit failed", exc_info=True)
        return []


def _unknown_attempt_id() -> str:
    try:
        from ouroboros.usage_accounting import last_physical_attempt_capture

        capture = last_physical_attempt_capture()
        return str(getattr(capture, "attempt_id", "") or "")
    except Exception:
        return ""


def latch_after_unknown(
    tools: Any, *, error_kind: str, drive_logs: pathlib.Path, task_id: str,
    emit_progress: Any, transport_episode: Any = None,
) -> bool:
    """Round-gate entry: prefer a live leaf to a new model recovery episode.

    Returns True when the caller should ``continue`` to the next round top,
    clearing its transport episode; the durable latch parks in ``hold_step``.
    """
    if str(error_kind or "") != "provider_outcome_unknown":
        return False
    ctx = getattr(tools, "_ctx", None)
    if ctx is None or not _configured_session(ctx):
        return False
    run_ids = _eligible_live_runs(ctx)
    if not run_ids:
        return False
    run_id = run_ids[0]
    try:
        prior = read_unknown_hold(ctx)
    except UnknownHoldUnreadable:
        return False  # Unknown checkpoints cannot be replaced with an empty run map.
    attempt_id = _unknown_attempt_id()
    hold = {
        "run_ids": run_ids,
        "run_id": run_id,
        "entered_at": utc_now_iso(),
        "hold_cycles": int(prior.get("hold_cycles") or 0) + 1,
        **({"unknown_attempt_id": attempt_id} if attempt_id else {}),
    }
    write_unknown_hold(ctx, run_id, hold)
    if transport_episode is not None:
        from ouroboros.loop_transport import emit_network_wait_event

        emit_network_wait_event(
            drive_logs, task_id=task_id, phase="ended",
            elapsed_sec=transport_episode.waited_sec, redials=transport_episode.redials,
            model=str(getattr(ctx, "active_model", "") or ""), detail="hold_latched",
            outcome_custody=transport_episode.outcome_custody,
        )
    _emit_hold_event(
        drive_logs, task_id=task_id, phase="entered", run_id=run_id,
        hold_cycles=hold["hold_cycles"], attempt_id=attempt_id,
    )
    try:
        emit_progress(
            "🛰️ Provider outcome unknown for one nanny round, but the delegated "
            "leaf is alive — holding on the leaf ($0, no resend; the unknown "
            "attempt stays billed at its upper bound). Stop cancels."
        )
    except Exception:
        log.debug("hold entry note failed", exc_info=True)
    return True


def close_hold(
    tools: Any, *, drive_logs: pathlib.Path, task_id: str, detail: str,
) -> str:
    """Close any hold state on a non-recoverable loop exit (round limit, budget,
    abnormal exit). Returns "active"/"tombstone"/"" for what was found; emits an
    ``ended`` event so the durable episode never dangles an open bracket."""
    ctx = getattr(tools, "_ctx", None)
    if ctx is None or not _configured_session(ctx):
        return ""
    try:
        hold = read_unknown_hold(ctx)
    except Exception:
        return ""
    if not hold:
        return ""
    kind = "active" if hold.get("run_id") else "tombstone"
    try:
        clear_unknown_hold(ctx)
    except Exception:
        log.debug("hold clear failed", exc_info=True)
    if kind == "active" or detail != "loop_exit":
        # A fully-resumed hold's tombstone at ordinary loop exit is not an
        # open bracket — no spurious ended(loop_exit) event for it.
        _emit_hold_event(
            drive_logs, task_id=task_id, phase="ended",
            run_id=str(hold.get("run_id") or ""),
            hold_cycles=int(hold.get("hold_cycles") or 0), detail=detail,
            attempt_id=str(hold.get("unknown_attempt_id") or ""),
        )
    return kind


def _wake_is_control(payload: Dict[str, Any]) -> bool:
    events: List[Dict[str, Any]] = [
        item for item in (payload.get("wake_events") or []) if isinstance(item, dict)
    ]
    for item in events:
        if str(item.get("type") or "") in _CONTROL_WAKE_TYPES:
            return True
        if str(item.get("kind") or "") in _CONTROL_WAKE_KINDS:
            return True
    return False


def _whole_wake(ctx: Any, payload: Dict[str, Any]) -> Any:
    """The wake a receipt carries whole before its ACK: a spilled one's exact source (the same wake id), else
    None for the no-resend exit. A summary would acknowledge questions and mail it never showed."""
    delivery = payload.get("wake_delivery")
    if not isinstance(delivery, dict) or delivery.get("complete") is not False:
        return payload
    wake_id, marker = payload.get("supervision_wake_id"), {k: payload[k] for k in ("replay_of",) if k in payload}
    try:
        from ouroboros.artifacts import read_actor_source_bytes, task_id_for_artifacts

        whole = json.loads(read_actor_source_bytes(pathlib.Path(ctx.drive_root), task_id_for_artifacts(ctx),
                                                   delivery.get("source")))
    except Exception:
        log.warning("Spilled wake source unreadable for the hold receipt", exc_info=True)
        return None
    return {**marker, **whole} if isinstance(whole, dict) and wake_id and whole.get("supervision_wake_id") == wake_id \
        else None


def _append_wake_receipt(
    messages: List[Dict[str, Any]], payload: Dict[str, Any], run_id: str,
    attempt_id: str,
) -> None:
    bound = f" (unknown attempt {attempt_id})" if attempt_id else ""
    body = json.dumps(payload, ensure_ascii=False, indent=2)
    messages.append({
        "role": "user",
        "content": (
            "[DELEGATED LEAF WAKE / UNKNOWN-HOLD RESUME]\n"
            f"A previous nanny round was lost to an unknown provider outcome{bound} "
            "(it was never resent and stays billed at its reserved upper bound). The "
            f"task held on the live leaf run {run_id} instead of dying; this wake is "
            "the new material input for THIS round — judge it (read results, wait "
            "again, cancel, or integrate). Never rebuild the leaf's work. "
            "Use retry_of to resolve an unknown start; a new prompt is independent work.\n" + body
        ),
    })


def round_top(tools: Any, limit_ctx: Any, controls: Dict[str, Any], emit_progress: Any, *, new_input: bool) -> Any:
    """The loop's round-top hold (``hold_step`` over this round's ``limit_ctx``): ``""`` or ``"resume"`` dispatch
    the round; ``"park"`` once the hold's host-armed sleep was parked here without a model call (its wake is the
    next round top's); else the unknown no-resend terminal's ``(text, usage, trace)``, with zero provider calls."""
    step = hold_step(tools, controls=controls, messages=limit_ctx.messages, drive_logs=limit_ctx.drive_logs,
                     task_id=limit_ctx.task_id, emit_progress=emit_progress, new_input=new_input)
    if step == "park":
        from ouroboros.owner_wait import wait_after_tools

        wait_after_tools(tools._ctx, limit_ctx.messages, limit_ctx.llm_trace, limit_ctx.accumulated_usage,
                         limit_ctx.round_idx, limit_ctx.tool_schemas, limit_ctx.owner_msg_seen)
    if step != "terminal":
        return step
    from ouroboros.loop_delivery import _merge_finalization_trace
    from ouroboros.loop_round_limits import _handle_provider_unavailable

    # The no-call decision reads the usage record, not the error_kind argument: stamp first.
    limit_ctx.accumulated_usage["_last_llm_error_kind"] = "provider_outcome_unknown"
    text, usage, forced_trace = _handle_provider_unavailable(limit_ctx, error_kind="provider_outcome_unknown")
    _merge_finalization_trace(limit_ctx.llm_trace, forced_trace)
    return text, usage, limit_ctx.llm_trace


def hold_step(
    tools: Any, *, controls: Dict[str, Any], messages: List[Dict[str, Any]],
    drive_logs: pathlib.Path, task_id: str, emit_progress: Any,
    new_input: bool = False,
) -> str:
    """Round-top step for a latched hold.

    Returns "" (no latch — dispatch normally), "resume" (material new input in
    the transcript; dispatch this round), "park" (the host armed the hold's own
    warm sleep: the caller parks it now, without a model call, and its wake is
    the next round top's), or "terminal" (control wake, refused wait, or failed
    acknowledgement — the caller takes the existing unknown no-resend terminal;
    zero provider calls).
    """
    ctx = getattr(tools, "_ctx", None)
    if ctx is None or not _configured_session(ctx):
        return ""
    try:
        hold = read_unknown_hold(ctx)
    except UnknownHoldUnreadable:
        # "Cannot know whether a latch exists" must not read as "no latch":
        # dispatching here could resend an unknown request (final-pair sol #2).
        _emit_hold_event(
            drive_logs, task_id=task_id, phase="ended", run_id="",
            hold_cycles=0, detail="latch_unreadable",
        )
        return "terminal"
    run_ids = list(hold.get("run_ids") or ([hold["run_id"]] if hold.get("run_id") else []))
    if not run_ids:
        return ""
    run_id = run_ids[0]
    cycles = int(hold.get("hold_cycles") or 1)
    attempt_id = str(hold.get("unknown_attempt_id") or "")

    def _exit_terminal(detail: str) -> str:
        try:
            clear_unknown_hold(ctx)
        except Exception:
            log.debug("hold clear failed on terminal exit", exc_info=True)
        _emit_hold_event(
            drive_logs, task_id=task_id, phase="ended", run_id=run_id,
            hold_cycles=cycles, detail=detail, attempt_id=attempt_id,
        )
        return "terminal"

    if controls.get("finalize_now"):
        ctx._transport_repeat_control_reason = str(controls["finalize_now"]).splitlines()[0].strip()
        return _exit_terminal("finalize_now")
    if new_input:
        # The round-top drain already appended owner/task dialogue: that IS the
        # material new input, so resume without waiting — arrival timing must
        # not decide whether an owner message wakes the task.
        try:
            write_unknown_hold(ctx, "", {"hold_cycles": cycles})
        except Exception:
            log.warning("hold tombstone write failed; latch stays active", exc_info=True)
        _emit_hold_event(
            drive_logs, task_id=task_id, phase="resumed", run_id=run_id,
            hold_cycles=cycles, detail="owner_input", attempt_id=attempt_id,
        )
        return "resume"
    woke, ctx._delegate_hold_parked = bool(getattr(ctx, "_delegate_hold_parked", False)), False
    if cycles > 1 and not woke:  # a park already spaced this cycle's dispatch
        time.sleep(min(
            NETWORK_WAIT_BACKOFF_START_SEC * (2.0 ** min(cycles - 1, 4)),
            _HOLD_BACKOFF_CAP_SEC,
        ))
    # The wake answers with the family's NATIVE result; its own JSON payload is
    # what the hold judges. A shape this cannot read leaves the payload empty,
    # which fails the acknowledgement below into the honest no-resend terminal —
    # never an escape that would let the cleanup cancel a healthy leaf.
    wake = hold_wake(ctx, run_ids, woke=woke)
    if wake is None and callable(getattr(ctx, "owner_wait_callback", None)):
        try:
            armed = str(arm_hold_sleep(ctx, run_ids).get("reason") or "")
        except (TypeError, ValueError):
            log.warning("Hold sleep of %s not armed; it holds in place", task_id, exc_info=True)
            armed = ""
        if armed == "sleep_armed":
            ctx._delegate_hold_parked = True  # its wake publishes at the next round top
            return "park"
        wake = hold_wake(ctx, run_ids, woke=True) if armed == "ready" else None  # ready while it was armed
    while wake is None:  # no continuation owner to sleep under: hold in place on the same sources
        time.sleep(_TICK_SEC)
        wake = hold_wake(ctx, run_ids)
    try:
        payload = delegate_payload(wake)
    except (AttributeError, TypeError, ValueError):
        payload = {}
    # Controls re-checked at the SOURCE too: an oversized wake render may drop
    # the control event from the rendered list, so also read the DURABLE
    # unrendered pending-wake payload (covers finalize_now, which the
    # cancel/deadline source check cannot see) — final-pair sol #3.
    try:
        state = supervision_checkpoint(ctx)
        selected = str(payload.get("run_id") or run_id)
        pend = state.get("runs", {}).get(selected, {}).get("pending_wake")
        pend_payload = pend.get("payload") if isinstance(pend, dict) else None
    except Exception:
        pend_payload = None
    if (_wake_is_control(payload) or _control_wakes(ctx)
            or (isinstance(pend_payload, dict) and _wake_is_control(pend_payload))):
        from ouroboros.loop_transport import transport_repeat_stop_requested

        transport_repeat_stop_requested(ctx)  # Preserve a current owner cause without delivering/ACKing it.
        return _exit_terminal("control_wake")
    if str(payload.get("status") or "") in _NON_WAKE_STATUSES:
        # A refusal/fault is a daemon statement, not a leaf wake: no proof of a
        # live leaf, so no paid resume — take the honest no-resend terminal.
        return _exit_terminal("wait_" + str(payload.get("status")))
    payload = _whole_wake(ctx, payload)
    if payload is None:
        return _exit_terminal("wake_source_unreadable")
    _append_wake_receipt(messages, payload, run_id, attempt_id)
    acked = False
    for _ in range(_ACK_ATTEMPTS):
        try:
            if acknowledge_pending_wake(ctx, payload):
                acked = True
                break
        except Exception:
            # An ack exception is an ack failure, never a loop escape: the
            # outer finally would clear the latch and cleanup would cancel
            # the healthy leaf — the original incident (final-pair sol #4).
            log.debug("wake ack attempt raised", exc_info=True)
        time.sleep(1.0)
    if not acked:
        # One wake = one dispatch: an unacknowledged wake would replay into the
        # next request verbatim — a resend in disguise. Fail closed instead.
        if messages and str(messages[-1].get("content") or "").startswith(
                "[DELEGATED LEAF WAKE"):
            messages.pop()
        return _exit_terminal("ack_failed")
    # Tombstone (no run_id => inactive): the cycle count survives a resume so a
    # repeat unknown re-latches with the backoff floor instead of a tight loop.
    # A failed tombstone write keeps the latch active — worst case one extra
    # park next round (the wake is acked, nothing replays) — never an escape.
    try:
        write_unknown_hold(ctx, "", {"hold_cycles": cycles})
    except Exception:
        log.warning("hold tombstone write failed; latch stays active", exc_info=True)
    _emit_hold_event(
        drive_logs, task_id=task_id, phase="resumed", run_id=run_id,
        hold_cycles=cycles, detail=str(payload.get("status") or ""),
        attempt_id=attempt_id,
    )
    try:
        emit_progress(
            "🛰️ Leaf wake received — resuming the nanny with the new input "
            "(the lost round was never resent)."
        )
    except Exception:
        log.debug("hold resume note failed", exc_info=True)
    return "resume"
