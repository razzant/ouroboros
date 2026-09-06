"""Transport-outage wait episodes and provider-failure terminal text for the main loop.

A REMOTE pre-dispatch transport failure (typed ``released`` custody: connect
refused/timed out before request bytes left this host, $0 in the ledger) is not a
model failure. Instead of burning the fallback chain or terminalizing, the round
gate in ``loop.py`` latches a :class:`TransportWaitEpisode` and waits: durable
``network_wait`` events + owner progress notes, an interruptible backoff sleep,
then a free redial of the SAME round (the round budget is not consumed). A managed
task waits as long as its existing rails allow — owner deadline minus the
dispatch-admission reserve, budget, Stop, and the supervisor's absolute ceiling.
Every turn stamped direct-chat (owner chat and Presence turns) or ephemeral — the
``interactive`` class — waits the same way but carries no queue rails and
ordinarily no owner deadline, so its episode is bounded by the raw configured
task idle timeout (``get_task_idle_timeout_sec``): the bound limits idle WAITING,
measured from each outage episode's entry (a flapping egress starts a new
episode); a granted redial runs to its own connect timeout and a dispatched
response is always accepted — the bound never cancels in-flight work. When an
explicit deadline window also exists, the shorter window binds. When the binding
window runs out, ``_handle_provider_unavailable`` takes a deterministic no-resend
terminal keyed on the episode's ``wait_cause`` (no forced-final provider call);
the durable ``ended`` detail names the rail that expired — the bound's own
``interactive_wait_window_exhausted``, or the deadline's detail when the owner
window closed first. Interactive progress notes omit cancellation promises;
direct-turn Stop uses its existing typed control and wakes the same sleep.
Recovery is an owner note for every episode; local adoption and
error-kind change are notes for interactive turns only, because such a turn
has no progress row to show the closure — a managed task keeps the durable
row and its ordinary progress; exhaustion is a note for an interactive turn,
while a managed task's exhaustion is its terminal result; only an ephemeral
turn's episode-boundary
notes (entry, recovery/closure, exhaustion) carry the typed ``task_incident``
toast pair, because the browser renders no progress rows for that turn (direct
turns get live-card rows); periodic notes stay silent. Every episode note passes
``incident=`` (``None`` unless it is such a boundary note), so an ``emit_progress``
callable handed to ``run_llm_loop`` must accept that keyword.

Also hosts the owner-facing provider-failure text helpers and terminal salvage
readers used by that terminal path (extracted from ``loop.py``, which is at its
size-ratchet byte cap).
"""

from __future__ import annotations

import logging
import os
import pathlib
import queue
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ouroboros.config import (
    NETWORK_WAIT_BACKOFF_START_SEC,
    NETWORK_WAIT_NOTE_INTERVAL_SEC,
    get_finalization_grace_sec,
    get_task_idle_timeout_sec,
)
from ouroboros.deadline_utils import parse_deadline_ts
from ouroboros.loop_llm_call import TRANSPORT_DEATHS_KEY, _TRANSIENT_BACKOFF_CAP_SEC
from ouroboros.owner_mailbox import OwnerMailboxPeek
from ouroboros.utils import append_jsonl, utc_now_iso

log = logging.getLogger(__name__)

# Reserve for the final free redial near the owner deadline (Q14): round-top
# overhead (message drain, checkpoints, transcript seal, token measurement)
# routinely eats about a second on long transcripts, and a granted redial that
# the admission gate then refuses is a wasted grant.
_FINAL_REDIAL_MARGIN_SEC = 3.0


@dataclass
class TransportWaitEpisode:
    """Episode-local latch for one remote pre-dispatch transport outage.

    The latch — not the mutable ``_last_llm_error_kind`` projection — carries the
    terminal cause: later failures (a failed local fallback pass, the deadline
    admission gate) overwrite the usage projection, and the terminal decision
    must stay deterministic (no forced-final resend after a waited-out outage).
    ``interactive`` is the wait-class fact (every turn stamped direct-chat or
    ephemeral); ``ephemeral`` is the presentation fact (the browser renders no
    progress rows for such a turn); ``wait_bound_sec`` is the local ceiling an
    interactive turn gets in place of the queue rails it does not have.
    """

    wait_cause: str = "transport_unavailable"
    started_monotonic: float = 0.0
    interactive: bool = False
    ephemeral: bool = False
    wait_bound_sec: Optional[float] = None
    redials: int = 0
    wait_iterations: int = 0
    last_note_monotonic: float = 0.0
    local_pass_used: bool = False
    final_redial_done: bool = False
    mailbox_peek: OwnerMailboxPeek = field(default_factory=OwnerMailboxPeek, repr=False)

    @property
    def waited_sec(self) -> float:
        """Wall time spent waiting and redialing; 0.0 before the first wait
        iteration, so a zero-wait terminal never claims it waited."""
        return time.monotonic() - self.started_monotonic if self.wait_iterations else 0.0

    def incident(self, task_id: str, phase: str) -> Optional[Dict[str, str]]:
        """Typed toast pair for an ephemeral episode's owner note.

        The browser renders progress rows for managed tasks and direct turns
        but not for ephemeral decision turns, whose only owner-visible wait
        surface is the one-shot ``task_incident`` toast (``toast_once`` dedupes
        replay; the millisecond entry stamp keeps two episodes of one turn
        distinct). Managed and direct episodes carry none.
        """
        if not self.ephemeral:
            return None
        return {
            "task_incident": "network_wait",
            "toast_once": (
                f"{task_id}:network_wait:{phase}:{int(self.started_monotonic * 1000)}"
            ),
        }


def emit_network_wait_event(
    drive_logs: pathlib.Path,
    *,
    task_id: str,
    phase: str,
    elapsed_sec: float,
    redials: int,
    model: str,
    next_sleep_sec: Optional[float] = None,
    window_remaining_sec: Optional[float] = None,
    detail: str = "",
) -> None:
    """Durable episode evidence in events.jsonl (typed rows; no keyword scans).

    ``window_remaining_sec`` is the binding wait window left on a ``waiting``
    row (deadline or interactive bound); absent when no window bounds the wait.
    Closing rows describe cooperative worker exits. After external termination,
    correlate a confirmed task terminal; missing rows alone prove no outcome.
    """
    try:
        append_jsonl(pathlib.Path(drive_logs) / "events.jsonl", {
            "ts": utc_now_iso(),
            "type": "network_wait",
            "task_id": task_id,
            "phase": phase,
            "elapsed_sec": round(float(elapsed_sec), 1),
            "redials": int(redials),
            "model": model,
            "next_sleep_sec": (
                round(float(next_sleep_sec), 1) if next_sleep_sec is not None else None
            ),
            **({"window_remaining_sec": round(float(window_remaining_sec), 1)}
               if window_remaining_sec is not None else {}),
            **({"detail": detail} if detail else {}),
        })
    except Exception:
        log.debug("Failed to append network_wait event", exc_info=True)


def _use_local_fallback_configured() -> bool:
    return os.environ.get("USE_LOCAL_FALLBACK", "").lower() in ("true", "1")


def fallback_chain_allowed(
    ctx: Any, last_error_kind: str, episode: Optional[TransportWaitEpisode],
    accumulated_usage: Optional[Dict[str, Any]] = None,
) -> bool:
    """Whether this round may walk the cross-model fallback chain."""
    if bool(getattr(ctx, "exact_model_route", False)):
        return False
    if isinstance((accumulated_usage or {}).get(TRANSPORT_DEATHS_KEY), dict):
        # The round still holds an unresolved attempt (a granted transport-death
        # repeat with no usable response since): no paid candidate may dial over
        # it, whatever the last kind says.
        return False
    if episode is not None:
        # Q4: during a remote transport outage the chain runs at most ONCE per
        # episode, and only when USE_LOCAL_FALLBACK makes the whole chain local —
        # remote candidates never dial over a proven dead egress.
        if (
            last_error_kind != "transport_unavailable"
            or episode.local_pass_used
            or not _use_local_fallback_configured()
        ):
            return False
        episode.local_pass_used = True
        return True
    return last_error_kind not in (
        "context_overflow", "provider_outcome_unknown", "deadline_exhausted",
    )


def reconcile_transport_wait(
    episode: Optional[TransportWaitEpisode],
    ctx: Any,
    *,
    msg_present: bool,
    error_kind: str,
    drive_logs: pathlib.Path,
    task_id: str,
    model: str,
    emit_progress: Callable[..., None],
    after_local_pass: bool = False,
) -> Optional[TransportWaitEpisode]:
    """Reconcile the episode latch with one dispatch outcome.

    Enters a new episode on a fresh ``transport_unavailable`` failure (durable
    ``entered`` event; the first owner note fires immediately). An interactive
    turn's episode gets the idle-timeout bound at entry, because the bound is
    measured from entry and the turn has no other rail. ``emit_progress``
    honors the ``incident=`` keyword (``OuroborosAgent._emit_progress``): the
    typed toast pair rides the note for ephemeral episodes; recovery is a note
    for every episode, local adoption and error-kind change are notes for
    interactive turns only (a managed episode keeps its durable ``ended`` row
    and its ordinary progress), and exhaustion is separately noted only for
    interactive turns (``transport_wait_step``).
    The round gate reconciles twice per failed dispatch: once with the
    pre-chain kind, and once after the fallback chain with the FRESH kind — so
    an outage first observed MID-chain (a remote candidate dying pre-dispatch
    while the primary failed generically) still latches an episode instead of
    falling through to a generic terminal that would dial a forced-final call
    over the proven-dead egress. On a redial outcome: a response ends the
    episode as ``recovered`` (mandatory owner note), a NON-transport failure
    ends it as evidence the transport is passable again, while
    ``transport_unavailable`` and a pre-dispatch deadline refusal keep the
    latch for the wait/terminal step. A failed local fallback pass
    (``after_local_pass``) never clears the latched remote cause.
    """
    if episode is None:
        if msg_present or error_kind != "transport_unavailable":
            return None
        ephemeral = bool(getattr(ctx, "is_ephemeral_turn", False))
        interactive = ephemeral or bool(getattr(ctx, "is_direct_chat", False))
        episode = TransportWaitEpisode(
            started_monotonic=time.monotonic(),
            interactive=interactive,
            ephemeral=ephemeral,
            wait_bound_sec=float(get_task_idle_timeout_sec()) if interactive else None,
        )
        emit_network_wait_event(
            drive_logs, task_id=task_id, phase="entered",
            elapsed_sec=0.0, redials=0, model=model,
        )
        episode.last_note_monotonic = time.monotonic()
        # Interactive notes keep their existing wording; direct-turn Stop is
        # separately handled through its typed mailbox control.
        emit_progress(
            "🌐 Could not establish a provider connection — waiting and "
            "redialing automatically (failed attempts are $0)."
            + ("" if interactive else " Stop cancels."),
            incident=episode.incident(task_id, "entered"),
        )
        return episode
    elapsed = time.monotonic() - episode.started_monotonic
    if msg_present:
        if after_local_pass:
            emit_network_wait_event(
                drive_logs, task_id=task_id, phase="ended", elapsed_sec=elapsed,
                redials=episode.redials, model=model, detail="local_fallback_adopted",
            )
            if episode.interactive:  # a managed task keeps its durable row and ordinary progress
                emit_progress(
                    f"🌐 Provider connection still unavailable after {elapsed / 60.0:.1f} min "
                    "— continuing on the local fallback model.",
                    incident=episode.incident(task_id, "ended"),
                )
        else:
            emit_network_wait_event(
                drive_logs, task_id=task_id, phase="recovered", elapsed_sec=elapsed,
                redials=episode.redials, model=model,
            )
            emit_progress(
                f"🌐 Provider connection restored after {elapsed / 60.0:.1f} min — resuming.",
                incident=episode.incident(task_id, "recovered"),
            )
        return None
    if (
        not after_local_pass
        and error_kind not in ("transport_unavailable", "deadline_exhausted")
    ):
        # The redial got past the connect phase and failed differently: the
        # transport is provably passable, so ordinary failure policy resumes.
        emit_network_wait_event(
            drive_logs, task_id=task_id, phase="ended", elapsed_sec=elapsed,
            redials=episode.redials, model=model,
            detail=f"error_kind_changed:{error_kind}",
        )
        if episode.interactive:  # a managed task keeps its durable row and ordinary progress
            emit_progress(
                f"🌐 Provider connection restored after {elapsed / 60.0:.1f} min — the redial "
                f"got past the connect phase and failed as {error_kind}; ordinary failure policy resumes.",
                incident=episode.incident(task_id, "recovered"),
            )
        return None
    return episode


def interruptible_wait_sleep(seconds: float, wake_check: Callable[[], bool]) -> bool:
    """Sleep up to ``seconds`` in <=1s slices.

    Returns True the moment ``wake_check`` reports a pending owner signal — the
    caller re-enters the round top, whose ordinary drain delivers the message or
    control (finalize_now/hurry/dialogue) — and False after the full sleep.
    """
    deadline = time.monotonic() + max(0.0, float(seconds))
    while True:
        try:
            if wake_check():
                return True
        except Exception:
            log.debug("wake check failed during transport wait", exc_info=True)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(1.0, remaining))


def transport_repeat_stop_requested(ctx: Any) -> bool:
    """Accept a current finalize control at the unsent-repeat boundary.

    Mail remains unacknowledged; direct Stop retains its existing prohibition
    on post-task model work even though the unknown-outcome rail owns this exit.
    """
    if ctx is None or not getattr(ctx, "task_id", "") or getattr(ctx, "drive_root", None) is None:
        return False
    try:
        from ouroboros.owner_mailbox import KIND_FINALIZE_NOW, drain_owner_entries
        from ouroboros.outcomes import REASON_OWNER_REQUESTED_FINALIZATION
        from supervisor.owner_stop import REASON_OWNER_STOPPED_DIRECT_TURN, _owner_stop_control_is_current

        entries = drain_owner_entries(pathlib.Path(ctx.drive_root), ctx.task_id,
            set(getattr(ctx, "_loop_mailbox_seen_ids", ()) or ()), getattr(ctx, "task_attempt", None) or 1)
        for entry in entries:
            if entry.get("kind") != KIND_FINALIZE_NOW:
                continue
            first_line = str(entry.get("text") or "").splitlines()[0]
            if first_line.strip() != REASON_OWNER_REQUESTED_FINALIZATION or _owner_stop_control_is_current(
                ctx, ctx.drive_root, ctx.task_id, str(entry.get("msg_id") or ""),
            ):
                if first_line.strip() == REASON_OWNER_STOPPED_DIRECT_TURN:
                    ctx._skip_post_task_synthesis = True
                return True
        return False
    except Exception:
        log.debug("finalize-control peek failed during transport repeat", exc_info=True)
        return False


def wait_transport_repeat(ctx: Any) -> bool:
    """Stop a granted transport repeat only before its new physical dispatch."""
    from ouroboros.loop_llm_call import _emit_retry_deadline_exhausted, _sleep_within_deadline, _uncount_transport_death

    # Granted, counted and deadline-checked in _record_llm_call_error; only
    # the recorded backoff (by death ordinal) is left before the loop sends
    # a NEW physical attempt. Not a spent wall either — the unknown
    # no-resend terminal outranks the wall.
    backoff = (ctx.accumulated_usage.get(TRANSPORT_DEATHS_KEY) or {}).get("backoff_sec")
    if backoff is None:
        return True
    interrupted = False

    def wake_check() -> bool:
        nonlocal interrupted
        interrupted = bool(ctx.stop_retry_check())
        return interrupted

    options = {"wake_check": wake_check} if ctx.stop_retry_check is not None else {}
    if _sleep_within_deadline(backoff, ctx.deadline_ts, **options):
        return False
    _uncount_transport_death(ctx.accumulated_usage)  # only this never-sent grant; prior custody stays
    if interrupted:
        append_jsonl(ctx.drive_logs / "events.jsonl", {
            "ts": utc_now_iso(), "type": "llm_not_dispatched", "task_id": ctx.task_id,
            "round": ctx.round_idx, "model": ctx.model, "reason_code": "finalize_control_pending",
        })
    else:
        _emit_retry_deadline_exhausted(ctx.drive_logs, task_id=ctx.task_id, execution_id=ctx.execution_id,
            round_id=ctx.round_id, round_idx=ctx.round_idx, attempt=ctx.attempt,
            model=ctx.model, error_kind="provider_outcome_unknown")
    return True


def _owner_signal_pending(
    incoming_messages: Optional[queue.Queue],
    drive_root: Optional[pathlib.Path],
    task_id: str,
    owner_msg_seen: Optional[set],
    attempt: Any,
    mailbox_peek: Optional[OwnerMailboxPeek] = None,
) -> bool:
    """Non-destructive peek: is an owner message or typed control waiting?"""
    if incoming_messages is not None and not incoming_messages.empty():
        return True
    if drive_root is None or not task_id:
        return False
    try:
        from ouroboros.owner_mailbox import drain_owner_entries

        if mailbox_peek is not None:
            return mailbox_peek.pending(pathlib.Path(drive_root), task_id, set(owner_msg_seen or ()), attempt)
        # A COPY of the seen-set: this is a peek — the round top performs the
        # real drain, delivery, and acknowledgement.
        return bool(drain_owner_entries(
            pathlib.Path(drive_root), task_id, set(owner_msg_seen or ()), attempt,
        ))
    except Exception:
        log.debug("owner-signal peek failed during transport wait", exc_info=True)
        return False


def transport_wait_step(
    episode: TransportWaitEpisode,
    *,
    tools: Any,
    error_kind: str,
    drive_root: Optional[pathlib.Path],
    drive_logs: pathlib.Path,
    task_id: str,
    model: str,
    emit_progress: Callable[..., None],
    incoming_messages: Optional[queue.Queue],
    owner_msg_seen: Optional[set],
) -> bool:
    """One wait iteration of an active episode.

    Returns True to redial (the caller re-enters the round top WITHOUT consuming
    a round) and False to terminalize via the no-resend branch. The wait window
    is the None-aware minimum of the owner deadline minus the existing
    dispatch-admission reserve (so a granted redial actually dials) and, for an
    interactive turn, its ``wait_bound_sec`` measured from episode entry; the
    ``ended`` detail names the rail that expired. The acceptance-review
    percentage reserve is deliberately NOT a wait ceiling (Q18), and the
    supervisor's absolute 6h ceiling stays an external rail, not duplicated here.
    """
    elapsed = time.monotonic() - episode.started_monotonic
    # Signed windows (negative = how long ago that rail expired) decide the
    # attribution: the rail that expired EARLIER binds even when a process stall
    # inside a sleep overshot both — clamping first would erase the ordering.
    # Only the value used for sleeping and telemetry is clamped. An exact tie
    # keeps the deadline's detail (measure-zero; the owner window is the
    # stronger claim). The positive deadline case equals
    # ``dispatch_window_remaining_sec(deadline_ts, reserve=grace)``.
    deadline_ts = task_deadline_epoch(tools)
    deadline_signed = (
        None if deadline_ts is None
        else deadline_ts - max(0.0, float(get_finalization_grace_sec())) - time.time()
    )
    bound_signed = (
        None if episode.wait_bound_sec is None else episode.wait_bound_sec - elapsed
    )
    bound_binds = bound_signed is not None and (
        deadline_signed is None or bound_signed < deadline_signed
    )
    remaining = bound_signed if bound_binds else deadline_signed
    if remaining is not None:
        remaining = max(0.0, remaining)

    def _ended(detail: str) -> bool:
        emit_network_wait_event(
            drive_logs, task_id=task_id, phase="ended", elapsed_sec=elapsed,
            redials=episode.redials, model=model, detail=detail,
        )
        if episode.interactive:
            emit_progress(
                "🌐 Stopped waiting for a provider connection after "
                f"{elapsed / 60.0:.1f} min — this turn ends as a provider outage.",
                incident=episode.incident(task_id, "ended"),
            )
        return False

    if error_kind == "deadline_exhausted":
        # The redial was refused before dispatch: the owner window is spent. One
        # attribution rule everywhere — a bound that expired earlier keeps its
        # own detail; the refusal stays visible in the llm_not_dispatched row.
        return _ended(
            "interactive_wait_window_exhausted" if bound_binds else "deadline_refused_dispatch"
        )
    if episode.final_redial_done:
        return _ended(
            "interactive_wait_window_exhausted" if bound_binds else "deadline_after_final_redial"
        )
    if remaining is not None and remaining <= 0:
        return _ended("interactive_wait_window_exhausted" if bound_binds else "deadline_exhausted")
    backoff = min(
        NETWORK_WAIT_BACKOFF_START_SEC * (2.0 ** min(episode.wait_iterations, 4)),
        _TRANSIENT_BACKOFF_CAP_SEC,
    )
    note_interval = max(
        1.0,
        min(float(NETWORK_WAIT_NOTE_INTERVAL_SEC), get_task_idle_timeout_sec() / 2.0),
    )
    # The sleep never exceeds the note interval, so waiting notes keep the idle
    # rail alive even on owner-lowered idle timeouts.
    sleep_sec = min(backoff, note_interval)
    if remaining is not None and remaining < sleep_sec + _FINAL_REDIAL_MARGIN_SEC:
        # One last free redial just before the binding window closes (Q14).
        sleep_sec = max(0.0, remaining - _FINAL_REDIAL_MARGIN_SEC)
        episode.final_redial_done = True
    if time.monotonic() - episode.last_note_monotonic >= note_interval:
        episode.last_note_monotonic = time.monotonic()
        emit_progress(
            f"🌐 Still waiting for a provider connection — {elapsed / 60.0:.0f} min "
            f"elapsed, {episode.redials} redials; will resume automatically.",
            incident=None,  # a periodic note is never a toast; the episode always passes incident=
        )
    emit_network_wait_event(
        drive_logs, task_id=task_id, phase="waiting", elapsed_sec=elapsed,
        redials=episode.redials, model=model, next_sleep_sec=sleep_sec,
        window_remaining_sec=remaining,
    )
    episode.wait_iterations += 1
    interruptible_wait_sleep(
        sleep_sec,
        lambda: _owner_signal_pending(
            incoming_messages, drive_root, task_id, owner_msg_seen,
            # Same attempt key as the round-top drain (task_attempt or 1), so
            # the peek never sees acks under a different namespace.
            getattr(getattr(tools, "_ctx", None), "task_attempt", None) or 1,
            episode.mailbox_peek,
        ),
    )
    episode.redials += 1
    return True


def finalize_now_transport_terminal(
    episode: TransportWaitEpisode,
    *,
    drive_logs: pathlib.Path,
    task_id: str,
    model: str,
    handle_provider_unavailable: Callable[..., Any],
    control_reason: str = "",
) -> Any:
    """Route a finalize_now that lands during an active episode to the honest
    transport no-resend terminal.

    Every finalize_now flavor (supervisor deadline, cost ceiling, owner stop)
    normally dispatches one forced summarize call — but over a proven-dead
    egress that paid path can only fail at $0 with identical salvage, so the
    deterministic no-resend terminal wins. The episode's durable evidence is
    closed with an ``ended`` row first; the caller passes a partial of its
    ``_handle_provider_unavailable`` so terminal composition stays in loop.py.
    """
    emit_network_wait_event(
        drive_logs, task_id=task_id, phase="ended",
        elapsed_sec=time.monotonic() - episode.started_monotonic,
        redials=episode.redials, model=model, detail="finalize_now",
    )
    return handle_provider_unavailable(
        error_kind="transport_unavailable",
        wait_cause=episode.wait_cause,
        waited_sec=episode.waited_sec,
        interactive=episode.interactive,
        control_reason=control_reason,
    )


def end_episode_budget(
    episode: TransportWaitEpisode, drive_logs: pathlib.Path, task_id: str, model: str,
) -> None:
    """Close an active episode when the budget rail fires mid-wait.

    A free redial spends $0 itself, but a concurrent consumer (a child, another
    root) can exhaust the shared budget between redials; the budget terminal
    then owns the exit and the episode must not be left without its durable
    ``ended`` row.
    """
    emit_network_wait_event(
        drive_logs, task_id=task_id, phase="ended",
        elapsed_sec=time.monotonic() - episode.started_monotonic,
        redials=episode.redials, model=model, detail="budget_exhausted",
    )


def task_deadline_epoch(tools: Any) -> Optional[float]:
    """Return the task deadline for retry backoff."""
    meta = getattr(tools._ctx, "task_metadata", {})
    if not isinstance(meta, dict):
        return None
    deadline = parse_deadline_ts(meta.get("deadline_at"))
    return deadline.timestamp() if deadline is not None else None


def last_assistant_text(messages: List[Dict[str, Any]]) -> str:
    """Last real assistant text already produced this task — salvaged into the
    terminal answer when provider-death prevents a fresh final response, so
    useful work is never silently discarded (workspace files persist on disk
    regardless)."""
    for m in reversed(messages or []):
        if isinstance(m, dict) and m.get("role") == "assistant":
            content = m.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
    return ""


def provider_terminal_fallback_text(
    accumulated_usage: Dict[str, Any],
    *,
    is_context_overflow: bool,
    is_transport_wait: bool,
    waited_sec: float,
    interactive: bool = False,
    is_deadline_exhausted: bool,
    control_reason: str = "",
) -> str:
    """Owner-facing terminal text when provider death left nothing to salvage.

    ``is_context_overflow`` and ``is_transport_wait`` are the caller's resolved
    verdict, never re-derived here: ``_provider_unavailable_result`` decides the
    terminal precedence (a round record, then the latched wait cause, then the
    overflow salvage) and passes at most one of the two flags true.
    ``interactive`` is the episode's wait-class fact — a direct-chat, Presence, or
    ephemeral decision turn is "this turn", never "the task" — and ``waited_sec`` its wait
    fact (0.0 when the binding window was already spent before the first wait
    iteration, so that terminal never claims a wait). The waited-out wording
    deliberately avoids the supervisor's lifecycle term INTERRUPTED
    (STATUS_INTERRUPTED means pre-requeue, not terminal).
    """
    if is_context_overflow:
        return (
            "⚠️ The context exceeded the selected model window; no further provider call was made. "
            "Any files written so far are preserved in the workspace."
        )
    if is_transport_wait:
        from ouroboros.outcomes import REASON_OWNER_REQUESTED_FINALIZATION

        if control_reason == REASON_OWNER_REQUESTED_FINALIZATION:
            text = (
                "⚠️ The owner requested Wrap up while the provider connection was unavailable. "
                f"The wait ended after {waited_sec / 60.0:.1f} min; no new summary request was sent. "
                "Any files written so far are preserved."
            )
        elif interactive and waited_sec > 0:
            text = (
                "⚠️ Could not establish a provider connection; this turn waited and "
                f"redialed for {waited_sec / 60.0:.1f} min and ended as a provider outage, "
                "not completed. Retry when connectivity returns."
            )
        elif interactive:
            text = (
                "⚠️ Could not establish a provider connection, and no wait window was left; "
                "this turn ended as a provider outage, not completed. Retry when "
                "connectivity returns."
            )
        elif waited_sec > 0:
            text = (
                "⚠️ Could not establish a provider connection; the task waited and redialed "
                f"for {waited_sec / 60.0:.1f} min until its own limits ran out and ended as a provider outage, not completed. "
                "Any files written so far are preserved in the workspace. Retry when "
                "connectivity returns."
            )
        else:
            text = (
                "⚠️ Could not establish a provider connection, and the owner deadline left no "
                "time to wait; the task ended as a provider outage, not completed. Any files "
                "written so far are preserved in the workspace. Retry when connectivity returns."
            )
        if (isinstance(accumulated_usage.get(TRANSPORT_DEATHS_KEY), dict)
                or accumulated_usage.get("_last_llm_error_kind") == "provider_outcome_unknown"):
            # The episode redialed a granted transport-death repeat that never left the
            # host: an earlier attempt of the round is still unresolved at its upper
            # bound, and the owner text says both facts (the wait and the fence). The
            # class the repeat was released with is on the record, and the hint reads it
            # there — never the sticky kind, which by the time the window closes names a
            # LATER free redial's refusal (``deadline_exhausted``).
            text += provider_recovery_hint(accumulated_usage)
        return text
    if is_deadline_exhausted:
        text = "⚠️ The owner deadline ended primary model work; any files written so far are preserved."
        if (isinstance(accumulated_usage.get(TRANSPORT_DEATHS_KEY), dict)
                or accumulated_usage.get("_last_llm_error_kind") == "provider_outcome_unknown"):
            text += provider_recovery_hint(accumulated_usage)
        return text
    return (
        "⚠️ The model provider returned no usable response after retries and same-model reroute."
        f"{provider_failure_hint(accumulated_usage)}{provider_recovery_hint(accumulated_usage)} "
        "Any files written so far are preserved in the workspace."
    )


def provider_failure_hint(accumulated_usage: Dict[str, Any]) -> str:
    detail = " ".join(str(accumulated_usage.get("_last_llm_error") or "").split()).strip()
    if not detail:
        return ""
    return f" Last provider error: {detail}"


def provider_recovery_hint(accumulated_usage: Dict[str, Any]) -> str:
    """Explain whether retrying later is likely to help."""
    kind = str(accumulated_usage.get("_last_llm_error_kind") or "").strip()
    deaths = accumulated_usage.get(TRANSPORT_DEATHS_KEY)
    repeats = int(deaths.get("count") or 0) if isinstance(deaths, dict) else 0
    if repeats:
        # Paid repeats already spent on the last dispatched round's typed transport
        # deaths (the record is round-keyed and cleared only by a usable response):
        # name them, and the class the repeat failed with, so the terminal never
        # reads as "sent once" and never promises a retry the fence forbids. That
        # class lives on the record, stamped by the repeat's own failure once that
        # failure is classified as an exception: the sticky kind may by now belong to
        # a later free redial of the round, or to a refusal, and would misname the
        # paid attempt. A record without the stamp falls back to the sticky kind, and
        # on those paths the sticky kind IS the repeat's own outcome: a repeat that
        # returned an empty response left the host and stamps no exception class, so
        # the sticky kind is that response's own class (provider_incomplete_response,
        # rate_limit, provider_body_error, ...); a repeat refused before it was sent
        # by the admission gate or the sleep gate (each writes its own durable row) is
        # un-counted and keeps the unknown class. A budget refusal does not un-count:
        # the budget rail cannot prove the repeat never left the host, so the record
        # keeps the attempt booked and the budget terminal, not this hint's provider
        # terminal, ends the round.
        failed_as = str(deaths.get("error_kind") or kind)
        last = (
            " the dispatched request has no terminal provider outcome;"
            if failed_as == "provider_outcome_unknown" else f" the repeat failed as {failed_as};"
        )
        return (
            f" {repeats} earlier physical attempt(s) of the last dispatched round died "
            "with a typed transport death and were repeated as new attempts (the dead "
            f"ones stay unresolved at their upper bound);{last} no further retry or paid "
            "fallback was sent while the earlier request has no terminal outcome, since "
            "either could duplicate live work."
        )
    if kind == "provider_outcome_unknown":
        return (
            " The dispatched request has no terminal provider outcome, so no "
            "retry or paid fallback was sent; either could duplicate live work."
        )
    if kind == "transport_unavailable":
        return (
            " No provider connection could be established (typed pre-dispatch "
            "failure, $0 spent); the exact exception class is in the durable "
            "llm_api_error event. Retrying when connectivity returns will help."
        )
    if kind == "subscription_window_exhausted":
        reset_at = str(accumulated_usage.get("_last_llm_reset_at") or "").strip()
        when = f" It resets at {reset_at}." if reset_at else ""
        return (
            " The subscription window for the delegated route is spent. This is "
            f"TRANSIENT, not a billing refusal — waiting cures it.{when} Retrying is "
            "scheduled against that reset time, not the ordinary short backoff."
        )
    if kind in {"quota_exhausted", "auth_error", "request_too_large", "bad_request", "context_overflow"}:
        guidance = {
            "quota_exhausted": "The provider rejected the request for quota/billing reasons; retrying the same request will not help until the key/account limit changes.",
            "auth_error": "The provider rejected authentication/authorization; retrying the same request will not help until the configured key or provider access is fixed.",
            "request_too_large": "The provider rejected the request size/output-token shape; retrying the same request will not help without reducing context/output demand or changing model capacity.",
            "bad_request": "The provider rejected the request shape; retrying the same request will not help until the transcript/tool payload is fixed.",
            "context_overflow": "The context overflowed the model window; retrying the same request will not help without reducing context or changing model capacity.",
        }.get(kind, "Retrying the same provider request will not help until the underlying request/account issue changes.")
        return f" {guidance}"
    detail = str(accumulated_usage.get("_last_llm_error") or "").lower()
    if "prefill" in detail or "conversation must end with a user message" in detail:
        return (
            " This looks like a client-side transcript-shape error, not a "
            "provider outage; retrying the same input will not help."
        )
    if "provider returned incomplete response" in detail or "finish_reason=null" in detail:
        return (
            " The provider returned incomplete responses repeatedly; this may "
            "be transient, but it can also indicate malformed client input."
        )
    return " If background consciousness is running, it will retry when the provider recovers."
