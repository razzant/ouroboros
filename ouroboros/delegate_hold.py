"""Unknown-provider hold: wait for the LEAF, not the provider (nanny-leaf D1-min).

A configured-session nanny exists to supervise one physical delegated leaf. When
one of its metered rounds dies ``provider_outcome_unknown`` (dispatched request,
no terminal provider fact — never resent, per custody doctrine) while that leaf
is alive and healthy, terminalizing the nanny used to cancel the leaf through
the cause-blind terminal cleanup. Instead, the round gate latches a durable
hold and the NEXT round top parks the task in the same $0, provider-independent
``supervised_wait`` the nanny would have chosen — owner mail, controls, and the
idle-rail lease all stay live there. A meaningful leaf wake resumes the task
with a NEW round whose transcript carries the wake receipt: materially new
input, so the unknown request itself is never re-transmitted (owner-ratified
doctrine revision; see docs/ARCHITECTURE.md). Control wakes (Stop, deadline,
finalize_now) exit through the existing unknown no-resend terminal with zero
further provider calls.

Eligibility is deliberately narrow (owner Q8=A): configured-session/exact
actor routes with EXACTLY one open, non-terminal delegated run and no pending
invocations. The liveness probe is a READ-ONLY engine poll requiring a
positive non-terminal engine state — a refusal, fault, empty state, or open
containment fault is not evidence of a live leaf. When no hold applies, the
supervising model uses ordinary managed transport recovery; external custody
remains unchanged. A terminal-but-unsettled leaf must not hold, since an
instant terminal wake would buy repeated model calls. A successful hold takes
over any existing transport episode, so its wake alone permits the next round.
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
    UnknownHoldUnreadable,
    _control_wakes,
    acknowledge_pending_wake,
    clear_unknown_hold,
    read_unknown_hold,
    supervised_wait,
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
# so this plain bounded sleep can never race the idle reaper (the interruptible
# owner surface is the supervised_wait right after it).
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
    """READ-ONLY liveness probe: one short engine poll, no lease/settle/cancel.

    Requires a POSITIVE non-terminal engine state. A gateway failure, refusal,
    or state-less payload is not evidence of a live leaf (fail-closed)."""
    import contextlib

    try:
        from ouroboros.claudexor_daemon import ensure_owned_gateway
        from ouroboros.delegate_progress import bounded_poll

        with contextlib.closing(ensure_owned_gateway(admission_wait_sec=0)) as gateway:
            detail = bounded_poll(gateway, run_id, 5.0)
        state = str(custody.summary_of(detail).get("state") or "")
    except Exception:
        log.debug("Hold eligibility leaf probe failed", exc_info=True)
        return False
    return bool(state) and state not in custody.TERMINAL_STATES


def _single_live_run(ctx: Any) -> str:
    """Return the run id iff custody holds EXACTLY one open run, no pending
    invocations, no open containment fault, a readable log, and the read-only
    probe reads a positive non-terminal engine state."""
    mine = str(getattr(ctx, "task_id", "") or "")
    try:
        root = custody.custody_root(ctx)
        if custody.custody_log_unreadable(root):
            return ""
        open_rows = [row for row in custody.open_runs(root) if row.task_id == mine]
        if len(open_rows) != 1:
            return ""
        if any(
            str(row.get("task_id") or "") == mine
            for row in custody.pending_invocations(root)
        ):
            return ""
        run_id = str(open_rows[0].run_id or "")
        if not run_id:
            return ""
        if any(
            str(row.get("run_id") or "") == run_id
            for row in custody.open_containment_faults(root)
        ):
            return ""
    except Exception:
        log.debug("Hold eligibility custody audit failed", exc_info=True)
        return ""
    if not _leaf_probe_live(ctx, run_id):
        return ""
    return run_id


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
    run_id = _single_live_run(ctx)
    if not run_id:
        return False
    try:
        prior = read_unknown_hold(ctx)
    except UnknownHoldUnreadable:
        prior = {}  # the fresh write below replaces the unreadable file
    attempt_id = _unknown_attempt_id()
    hold = {
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
            "again, cancel, or integrate). Never rebuild the leaf's work on metered "
            "tokens and never start a duplicate leaf.\n" + body
        ),
    })


def hold_step(
    tools: Any, *, controls: Dict[str, Any], messages: List[Dict[str, Any]],
    drive_logs: pathlib.Path, task_id: str, emit_progress: Any,
    new_input: bool = False,
) -> str:
    """Round-top step for a latched hold.

    Returns "" (no latch — dispatch normally), "resume" (material new input in
    the transcript; dispatch this round), or "terminal" (control wake, refused
    wait, or failed acknowledgement — the caller takes the existing unknown
    no-resend terminal; zero provider calls).
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
    if not hold.get("run_id"):
        return ""
    run_id = str(hold.get("run_id") or "")
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
    if cycles > 1:
        time.sleep(min(
            NETWORK_WAIT_BACKOFF_START_SEC * (2.0 ** min(cycles - 1, 4)),
            _HOLD_BACKOFF_CAP_SEC,
        ))
    # The supervising wait answers with the family's NATIVE result; its own JSON
    # payload is what the hold judges. A shape this cannot read leaves the payload
    # empty, which fails the acknowledgement below into the honest no-resend
    # terminal — never an escape that would let the cleanup cancel a healthy leaf.
    wake = supervised_wait(ctx, run_id)
    try:
        payload = delegate_payload(wake)
    except (AttributeError, TypeError, ValueError):
        payload = {}
    # Controls re-checked at the SOURCE too: an oversized wake render may drop
    # the control event from the rendered list, so also read the DURABLE
    # unrendered pending-wake payload (covers finalize_now, which the
    # cancel/deadline source check cannot see) — final-pair sol #3.
    try:
        pend = supervision_checkpoint(ctx).get("pending_wake")
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
