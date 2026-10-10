"""Transport-outage wait episodes and provider-failure terminal text for the main loop.

A REMOTE pre-dispatch transport failure (typed ``released`` custody: connect
refused/timed out before request bytes left this host, $0 in the ledger) is not a
model failure, and one provider's failure does not prove the egress dead: the
round first tries every configured route (``loop_model_call._recover_failed_round``),
and only when none answered does the round gate in ``loop.py`` latch a
:class:`TransportWaitEpisode` and wait: durable
``network_wait`` events + owner progress notes, an interruptible backoff sleep,
then a free redial of the SAME round (the round budget is not consumed). A managed
task waits as long as its existing rails allow — owner deadline minus the
dispatch-admission reserve, budget, Stop, and the supervisor's absolute ceiling.
Every turn stamped direct-chat (owner chat and Presence turns) — the
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
An eligible unknown outcome (``new_generation_after_unknown``; its charge is disclosed)
follows the same order on every turn except inline Presence: configured routes
first, then the upstream-observed continuation inside the same wait bounds.
Recovery is an owner note for every episode; an
error-kind change is a note for interactive turns only, because such a turn
has no progress row to show the closure — a managed task keeps the durable
row and its ordinary progress; exhaustion is a note for an interactive turn,
while a managed task's exhaustion is its terminal result. Every episode note is
the host speaking about the turn: it passes ``incident=None`` and keeps the
default voice, so an ``emit_progress`` callable handed to ``run_llm_loop`` must
accept the emitter's keyword facts (``incident``, ``narration``).

Also hosts the owner-facing provider-failure text helpers and terminal salvage
readers used by that terminal path (extracted from ``loop.py``, which is at its
size-ratchet byte cap).
"""

from __future__ import annotations

import logging
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
from ouroboros.loop_llm_call import TRANSPORT_DEATHS_KEY, _TRANSIENT_BACKOFF_CAP_SEC, _exception_body
from ouroboros.llm_probe import upstream_transport_reachable
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
    """One remote outage or chosen wait for temporary primary-provider refusals.

    The latch — not the mutable ``_last_llm_error_kind`` projection — carries the
    terminal cause: later failures (a failed local fallback pass, the deadline
    admission gate) overwrite the usage projection, and the terminal decision
    must stay deterministic (no forced-final resend after a waited-out outage).
    ``interactive`` is the wait-class fact (every turn stamped direct-chat);
    ``wait_bound_sec`` is the local ceiling an interactive turn gets in place
    of the queue rails it does not have.
    """

    wait_cause: str = "transport_unavailable"
    started_monotonic: float = 0.0
    started_at: float = field(default_factory=time.time)
    interactive: bool = False
    wait_bound_sec: Optional[float] = None
    redials: int = 0
    wait_iterations: int = 0
    last_note_monotonic: float = 0.0
    final_redial_done: bool = False
    outcome_custody: Dict[str, Any] = field(default_factory=dict)
    continuation_granted: bool = False
    mailbox_peek: OwnerMailboxPeek = field(default_factory=OwnerMailboxPeek, repr=False)

    @property
    def waited_sec(self) -> float:
        """Wall time spent waiting and redialing; 0.0 before the first wait
        iteration, so a zero-wait terminal never claims it waited."""
        return time.monotonic() - self.started_monotonic if self.wait_iterations else 0.0


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
    outcome_custody: Optional[Dict[str, Any]] = None,
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
            **({"outcome_custody": dict(outcome_custody)} if outcome_custody else {}),
        })
    except Exception:
        log.debug("Failed to append network_wait event", exc_info=True)


def managed_transport_continuation(ctx: Any) -> bool:
    """Ordinary cognition rejoins its accepted operation under its existing controls."""
    return bool(ctx is not None and getattr(ctx, "task_id", "")
                and not inline_presence(ctx))


def inline_presence(ctx: Any) -> bool:
    """An inline Presence turn: its unknown outcome ends in its own typed retry refusal."""
    from ouroboros.dialogue_provenance import is_presence_task

    metadata = getattr(ctx, "task_metadata", None)
    return (getattr(ctx, "current_task_type", None) == "presence"
            or is_presence_task({"metadata": metadata if isinstance(metadata, dict) else {}}))


def new_generation_after_unknown(ctx: Any, accumulated_usage: Optional[Dict[str, Any]] = None) -> bool:
    """Whether this round may start a NEW, possibly charged, generation after its unknown outcome.

    Typed, never from prose or elapsed time: the receiving call has ended with no usable
    answer and no same-operation rejoin is viable. It is refused for an accepted operation
    whose READ failed (it may still finish; only it is rejoined), for a round still holding
    paid repeats, and for inline Presence. It proves nothing about remote computation.
    """
    usage = accumulated_usage if accumulated_usage is not None else (getattr(ctx, "_accumulated_usage", None) or {})
    pending = usage.get("_pending_transport_outcome") or {}
    return bool(ctx is not None and getattr(ctx, "task_id", "") and not inline_presence(ctx)
                and not isinstance(usage.get(TRANSPORT_DEATHS_KEY), dict)
                and not pending.get("same_operation_recoverable"))


def append_unknown_recovery_input(messages: list, accumulated_usage: dict, previous: Dict[str, Any], *,
                                  lead: str, continuation: str, connectivity: Any = None) -> bool:
    """The facts-only input a NEW generation after an unknown outcome carries, once per unknown attempt.

    The old attempt keeps its own id, route and unknown cost; the new physical attempt gets its
    own identity and may be another configured route. Returns False when this exact unknown
    attempt was already disclosed to the transcript.
    """
    noticed = (accumulated_usage.get("transport_recovery") or {}).get("previous_attempt")
    accumulated_usage["transport_recovery"] = {"previous_attempt": previous, "connectivity": connectivity or {},
                                               "continuation": continuation, "old_outcome": "unknown"}
    if previous and noticed == previous:
        return False
    messages.append({"role": "user", "content": "[SYSTEM NOTICE]\n" + (
        f"[Transport recovery] {lead} Continue from the recorded work in a NEW physical model attempt. "
        "The previous attempt's outcome and any unreported cost remain unknown; do not treat it as failed, "
        "free, completed, or an instruction to repeat completed tools. "
        f"Previous physical attempt: {previous.get('physical_attempt_id') or 'unreported'}; "
        f"operation: {previous.get('operation_id') or 'unreported'}.")})
    return True


def continue_unknown_transport(episode: TransportWaitEpisode, *, llm: Any, tools: Any,
                               messages: list, accumulated_usage: dict, drive_logs: Any,
                               task_id: str, model: str, emit_progress: Callable) -> bool:
    """Grant one NEW attempt only after upstream recovery, retaining old custody."""
    from ouroboros.config import get_llm_transport_read_timeout_sec
    from ouroboros.deadline_utils import dispatch_window_remaining_sec
    remaining = dispatch_window_remaining_sec(deadline_ts=task_deadline_epoch(tools), reserve_sec=get_finalization_grace_sec())
    if remaining is not None and remaining <= 0:
        return False
    timeout = float(get_llm_transport_read_timeout_sec())
    if remaining is not None:
        timeout = min(timeout, remaining)
    from ouroboros.model_slots import task_model_binding
    from ouroboros.model_wait import current_model_wait
    ctx, waiter = tools._ctx, current_model_wait()
    if transport_repeat_stop_requested(ctx) or (waiter is not None and waiter.control_reason()):
        return False
    role, account = task_model_binding({"model_role": getattr(ctx, "model_role", ""),
        "task_metadata": getattr(ctx, "task_metadata", {})},
        context_fit_plan=getattr(ctx, "context_fit_plan", None), overrides=waiter.overrides if waiter else None)
    observed = upstream_transport_reachable(llm, model, timeout=timeout, model_role=role,
        account_override=account, observed_after=episode.started_at,
        expected_route=(episode.outcome_custody.get("route")
                        if episode.outcome_custody.get("model", model) == model
                        and episode.outcome_custody.get("model_role", role) in ("", role) else None))
    remaining = dispatch_window_remaining_sec(deadline_ts=task_deadline_epoch(tools), reserve_sec=get_finalization_grace_sec())
    if (not observed or remaining == 0.0 or transport_repeat_stop_requested(ctx)
            or (waiter is not None and waiter.control_reason())):
        return False
    previous = dict(episode.outcome_custody)
    append_unknown_recovery_input(messages, accumulated_usage, previous, connectivity=observed,
                                  lead="Upstream connectivity is available again.", continuation="new_physical_attempt")
    accumulated_usage.pop(TRANSPORT_DEATHS_KEY, None)
    accumulated_usage.pop("_pending_transport_outcome", None)
    episode.continuation_granted = True
    emit_network_wait_event(drive_logs, task_id=task_id, phase="recovered",
        elapsed_sec=episode.waited_sec, redials=episode.redials, model=model,
        detail="new_attempt_after_unknown_outcome", outcome_custody=previous)
    emit_progress("🌐 Connection restored — continuing from saved work in a new attempt. "
                  "The prior result and unreported cost remain unknown; another charge is possible.", incident=None)
    return True


PRIMARY_REFUSAL_KINDS = frozenset({"provider_transient", "rate_limit"})


def primary_refusal_wait(ctx: Any, error_kind: str) -> bool:
    """A declared primary wait keeps temporary refusals on ordinary paced recovery."""
    return bool(getattr(ctx, "route_wait_on_primary", False) and error_kind in PRIMARY_REFUSAL_KINDS
                and not inline_presence(ctx) and not getattr(ctx, "exact_model_route", False))


def fallback_chain_allowed(
    ctx: Any, last_error_kind: str, episode: Optional[TransportWaitEpisode],
    accumulated_usage: Optional[Dict[str, Any]] = None,
) -> bool:
    """Whether this round may try configured alternatives for the current failure.

    Existing outage pacing bounds repeated attempts; no lifetime pass limit is added.
    An eligible unknown outcome tries them first, not the same route again; a declared
    wait on the primary (``switch_model(primary="wait")``) keeps its refusals and
    outages on the primary's own wait instead of paid alternatives. A context overflow
    reaches the configured routes only after the primary's own recovery ladder
    (``loop_model_call._recover_context_overflow``) left the round unanswered: the
    candidates inherit the compacted view (owner decision 7A, rung f).
    """
    usage = accumulated_usage or {}
    if bool(getattr(ctx, "exact_model_route", False)):
        return False
    if isinstance(usage.get(TRANSPORT_DEATHS_KEY), dict):
        # The round still holds an unresolved attempt (a granted transport-death
        # repeat with no usable response since): no paid candidate may dial over
        # it, whatever the last kind says.
        return False
    if primary_refusal_wait(ctx, last_error_kind):
        return False
    if getattr(ctx, "route_wait_on_primary", False) and (
            last_error_kind in ("transport_unavailable", "provider_outcome_unknown")
            or usage.get("_last_llm_resource_refusal")):
        return False
    if last_error_kind == "provider_outcome_unknown":
        return new_generation_after_unknown(ctx, accumulated_usage)
    return last_error_kind not in ("deadline_exhausted", "llm_output_exhausted")


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
) -> Optional[TransportWaitEpisode]:
    """Reconcile the episode latch with one dispatch outcome.

    A declared primary wait also admits temporary provider refusals after ordinary retries,
    without treating a ready account catalog as proof of generation recovery.
    Enters a new episode on a fresh ``transport_unavailable`` failure, or an
    eligible unknown outcome (``new_generation_after_unknown``), that the
    configured routes did not answer (durable ``entered`` event; the first owner
    note fires immediately). An interactive
    turn's episode gets the idle-timeout bound at entry, because the bound is
    measured from entry and the turn has no other rail. ``emit_progress``
    honors the ``incident=`` keyword (``OuroborosAgent._emit_progress``). Recovery
    is a note for every episode; an error-kind change is a note for
    interactive turns only (a managed episode keeps its durable ``ended`` row
    and its ordinary progress), and exhaustion is separately noted only for
    interactive turns (``transport_wait_step``).
    With no active episode the round gate reconciles once, after the configured
    routes, with the round's own failure (the chain keeps it) — or with a
    candidate's pre-dispatch failure, so an outage first observed MID-chain (a
    candidate dying pre-dispatch while the primary failed generically) still
    latches an episode instead of a generic terminal. On a redial outcome: a
    response ends the episode as ``recovered`` (mandatory owner note), a
    NON-transport failure ends it as evidence the transport is passable again,
    while ``transport_unavailable`` and a pre-dispatch deadline refusal keep the
    latch for the wait/terminal step.
    A granted continuation that fails again — unknown after dispatch, or
    released before it — returns to the SAME episode (work-order §B9 "repeated
    unknown creates no burst"), and so does a free redial that crosses dispatch
    and dies unknown: the latch keeps its elapsed clock, growing backoff and
    redial count (one per wait iteration), switches its cause to the latest
    failure, and for an unknown outcome refreshes the custody and re-arms the
    probe's freshness bound at that failure. A granted continuation whose round
    made no new attempt (a failed probe, a call that yielded to control) keeps
    its grant untouched. An answer or any other failure class ends the latch
    exactly as before.
    """
    pending = dict((getattr(ctx, "_accumulated_usage", {}) or {}).get("_pending_transport_outcome") or {})
    unknown_again = error_kind == "provider_outcome_unknown" and new_generation_after_unknown(ctx)
    refused = not msg_present and primary_refusal_wait(ctx, error_kind)
    if refused and episode is not None:
        # Catalog readiness is not generation recovery. Keep the existing episode's
        # pacing, elapsed bound and prior unknown custody; only a real answer ends it.
        episode.wait_cause, episode.continuation_granted = error_kind, False
        return episode
    if episode is not None and episode.continuation_granted:
        if (msg_present or error_kind not in ("provider_outcome_unknown", "transport_unavailable")
                or (error_kind == "provider_outcome_unknown" and pending and not unknown_again)):
            # This new physical outcome ends the episode (an answer), owns a fresh one, or is
            # an accepted operation that may still finish (only it is rejoined, never regenerated).
            episode = None
        elif error_kind == "provider_outcome_unknown" and not pending:
            return episode  # No new attempt exists yet (a failed probe, a yielded call): the grant stands.
        else:
            # §B9: the granted attempt failed again — unknown after dispatch
            # (fresh custody) or released before it ($0) — and the SAME wait
            # owner keeps its elapsed clock and growing backoff (4->8->16->32->
            # 60s) instead of restarting at 4s with a zero counter; the wait
            # iteration that granted the attempt already counted its redial. An
            # unknown repeat re-arms the probe's freshness bound at THIS failure
            # (started_at is read only as observed_after), so a catalog
            # observation taken before the latest unknown outcome cannot prove
            # recovery from it. A released repeat keeps the old custody (already
            # disclosed in the transcript) and resumes free redials.
            episode.continuation_granted = False
            episode.wait_cause = error_kind
            if error_kind == "provider_outcome_unknown":
                episode.started_at = time.time()
                episode.outcome_custody = pending
            emit_network_wait_event(
                drive_logs, task_id=task_id, phase="continued",
                elapsed_sec=time.monotonic() - episode.started_monotonic, redials=episode.redials, model=model,
                detail=("continuation_outcome_unknown" if error_kind == "provider_outcome_unknown"
                        else "continuation_transport_unavailable"),
                outcome_custody=episode.outcome_custody,
            )
            if error_kind == "transport_unavailable":  # the grant note said "continuing"; the owner must hear otherwise
                emit_progress("🌐 The new attempt could not reach the provider — waiting and redialing "
                              "automatically (that attempt was $0).", incident=None)
            return episode
    if (episode is not None and not msg_present
            and episode.wait_cause != "provider_outcome_unknown" and unknown_again):
        # A formerly free redial crossed dispatch and died unknown: the same
        # episode now needs upstream proof before its next attempt; the clock,
        # backoff and redial count carry over instead of a fresh 4s episode.
        episode.wait_cause = "provider_outcome_unknown"
        episode.started_at = time.time()
        if pending:
            episode.outcome_custody = pending
        emit_network_wait_event(
            drive_logs, task_id=task_id, phase="continued",
            elapsed_sec=time.monotonic() - episode.started_monotonic, redials=episode.redials, model=model,
            detail="redial_outcome_unknown", outcome_custody=episode.outcome_custody,
        )
        emit_progress(  # the "$0 redials" framing of the entry note no longer holds
            "🌐 Provider connection was lost after dispatch. The outcome and any unreported cost remain unknown. "
            "Waiting for connectivity, then continuing from saved work with a new attempt; another charge is possible.",
            incident=None,
        )
        return episode
    if episode is None:
        unknown = unknown_again
        if msg_present or (error_kind != "transport_unavailable" and not unknown and not refused):
            return None
        interactive = bool(getattr(ctx, "is_direct_chat", False))
        episode = TransportWaitEpisode(
            wait_cause=error_kind,
            outcome_custody=dict((getattr(ctx, "_accumulated_usage", {}) or {}).get("_pending_transport_outcome") or {}),
            started_monotonic=time.monotonic(),
            interactive=interactive,
            wait_bound_sec=float(get_task_idle_timeout_sec()) if interactive else None,
        )
        emit_network_wait_event(
            drive_logs, task_id=task_id, phase="entered",
            elapsed_sec=0.0, redials=0, model=model, detail=error_kind if refused else "",
        )
        episode.last_note_monotonic = time.monotonic()
        # Interactive notes keep their existing wording; direct-turn Stop is
        # separately handled through its typed mailbox control.
        emit_progress(
            ("🌐 The primary provider temporarily refused the request. Waiting as chosen, then continuing "
             "on the same primary with paced requests; prior attempt costs remain recorded." if refused else
             "🌐 Provider connection was lost after dispatch. The outcome and any unreported cost remain unknown. "
             "Waiting for connectivity, then continuing from saved work with a new attempt; another charge is possible."
             if unknown else "🌐 Could not establish a provider connection — waiting and "
             "redialing automatically (failed attempts are $0).")
            + ("" if interactive else " Stop cancels."),
            incident=None,
        )
        return episode
    elapsed = time.monotonic() - episode.started_monotonic
    if msg_present:
        emit_network_wait_event(
            drive_logs, task_id=task_id, phase="recovered", elapsed_sec=elapsed,
            redials=episode.redials, model=model,
        )
        emit_progress(
            (f"🌐 Primary provider answered after {elapsed / 60.0:.1f} min — resuming."
             if episode.wait_cause in PRIMARY_REFUSAL_KINDS else
             f"🌐 Provider connection restored after {elapsed / 60.0:.1f} min — resuming."),
            incident=None,
        )
        return None
    if (
        error_kind not in ("transport_unavailable", "deadline_exhausted")
        and not (error_kind == "provider_outcome_unknown" and episode.wait_cause == "provider_outcome_unknown")
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
                incident=None,
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


def transport_repeat_stop_requested(ctx: Any, *, mailbox_peek: Any = None) -> bool:
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

        root, seen, attempt = pathlib.Path(ctx.drive_root), set(getattr(ctx, "_loop_mailbox_seen_ids", ()) or ()), getattr(ctx, "task_attempt", None) or 1
        if mailbox_peek is not None and not mailbox_peek.pending(root, ctx.task_id, seen, attempt):
            return False
        entries = drain_owner_entries(root, ctx.task_id, seen, attempt)
        for entry in entries:
            if entry.get("kind") != KIND_FINALIZE_NOW:
                continue
            first_line = str(entry.get("text") or "").splitlines()[0]
            if first_line.strip() != REASON_OWNER_REQUESTED_FINALIZATION or _owner_stop_control_is_current(
                ctx, ctx.drive_root, ctx.task_id, str(entry.get("msg_id") or ""),
            ):
                if first_line.strip() == REASON_OWNER_STOPPED_DIRECT_TURN:
                    ctx._skip_post_task_synthesis = True
                ctx._transport_repeat_control_reason = first_line.strip()
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
    *,
    owner_authority_only: bool = False,
) -> bool:
    """Peek unread input; acceptance may exclude context-only task messages.

    Transport and wait callers still wake for every message. Owner admission
    uses the same typed provenance boundary as the ordinary mailbox drain.
    """
    if incoming_messages is not None and not incoming_messages.empty():
        return True
    if drive_root is None or not task_id:
        return False
    try:
        from ouroboros.loop_messages import owner_authority_kinds
        from ouroboros.owner_mailbox import drain_owner_entries

        if mailbox_peek is not None and not owner_authority_only:
            return mailbox_peek.pending(pathlib.Path(drive_root), task_id, set(owner_msg_seen or ()), attempt)
        # A COPY of the seen-set: this is a peek — the round top performs the
        # real drain, delivery, and acknowledgement.
        entries = drain_owner_entries(
            pathlib.Path(drive_root), task_id, set(owner_msg_seen or ()), attempt,
        )
        return bool(owner_authority_kinds(entries) if owner_authority_only else entries)
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
                ("🌐 Stopped waiting for the primary provider after " if episode.wait_cause in PRIMARY_REFUSAL_KINDS
                 else "🌐 Stopped waiting for a provider connection after ") +
                f"{elapsed / 60.0:.1f} min — this turn ends as a provider outage.",
                incident=None,
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
            ("🌐 Still waiting for the primary provider — " if episode.wait_cause in PRIMARY_REFUSAL_KINDS
             else "🌐 Still waiting for a provider connection — ") + f"{elapsed / 60.0:.0f} min "
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

    Supervisor deadline, cost ceiling and owner stop send no forced summary
    over a waited-out outage or chosen primary refusal wait. Preserve the actual
    cause and prior attempt costs. Close the durable episode first; the caller's
    ``_handle_provider_unavailable`` keeps terminal composition in loop.py.
    """
    emit_network_wait_event(
        drive_logs, task_id=task_id, phase="ended",
        elapsed_sec=time.monotonic() - episode.started_monotonic,
        redials=episode.redials, model=model, detail="finalize_now",
    )
    return handle_provider_unavailable(
        error_kind=episode.wait_cause,
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
    from ouroboros.context_budget import HOST_CONTEXT_KIND_KEY

    for m in reversed(messages or []):
        if isinstance(m, dict) and m.get("role") == "assistant" and not m.get(HOST_CONTEXT_KIND_KEY):
            content = m.get("content")
            if isinstance(content, str) and content.strip():
                return content
    return ""


# Unknown outcome alone establishes neither a receipt nor a numeric bound.
UNKNOWN_ATTEMPT_COST_NOTE = (
    " This does not establish the attempt's cost; any recorded estimate or "
    "upper bound is not a settled receipt."
)


def _unknown_wait_note(unknown: bool, waited_sec: float, interactive: bool) -> str:
    """The REAL wait this unknown terminal spent, on the rails that never said it.

    The transport-wait branch below states its own wait in the sentence. The
    no-call rails (``provider_no_call_source`` -> ``provider_outcome_unknown_no_resend``)
    reach the generic terminal instead, which named neither the wait nor the
    fence, so an owner whose turn had waited minutes read a bare "no usable
    response". Zero stays silent rather than claiming a wait that never ran.
    """
    if not unknown or waited_sec <= 0:
        return ""
    subject = "This turn" if interactive else "The task"
    return f" {subject} spent {waited_sec / 60.0:.1f} min in the provider wait; that wait did not confirm the attempt's outcome."


def _unknown_terminal_recovery_hint(usage: Dict[str, Any]) -> str:
    # Wording only, confined to unknown terminals. Stop/Wrap up keep their
    # existing byte-for-byte control sentence and recovery hint.
    return provider_recovery_hint(usage).replace(
        "the dead ones stay unresolved at their upper bound",
        "the dead ones stay unresolved; any recorded bound is retained",
    )


def provider_terminal_fallback_text(
    accumulated_usage: Dict[str, Any],
    *,
    is_context_overflow: bool,
    is_transport_wait: bool,
    waited_sec: float,
    interactive: bool = False,
    is_deadline_exhausted: bool,
    control_reason: str = "", wait_cause: str = "",
) -> str:
    """Owner-facing terminal text when provider death left nothing to salvage.

    ``is_context_overflow`` and ``is_transport_wait`` are the caller's resolved
    verdict, never re-derived here: ``_provider_unavailable_result`` decides the
    terminal precedence (a round record, then the latched wait cause, then the
    overflow salvage) and passes at most one of the two flags true.
    ``interactive`` is the episode's wait-class fact — a direct-chat or Presence
    turn is "this turn", never "the task" — and ``waited_sec`` its wait
    fact (0.0 when the binding window was already spent before the first wait
    iteration, so that terminal never claims a wait). The waited-out wording
    deliberately avoids the supervisor's lifecycle term INTERRUPTED
    (STATUS_INTERRUPTED means pre-requeue, not terminal).
    """
    from ouroboros.outcomes import REASON_OWNER_REQUESTED_FINALIZATION
    from supervisor.owner_stop import REASON_OWNER_STOPPED_DIRECT_TURN

    unknown = (isinstance(accumulated_usage.get(TRANSPORT_DEATHS_KEY), dict)
               or bool(accumulated_usage.get("_pending_transport_outcome"))
               or accumulated_usage.get("_last_llm_error_kind") == "provider_outcome_unknown")
    if control_reason in {REASON_OWNER_REQUESTED_FINALIZATION, REASON_OWNER_STOPPED_DIRECT_TURN}:
        action = "Stop" if control_reason == REASON_OWNER_STOPPED_DIRECT_TURN else "Wrap up"
        waited = f" The wait ended after {waited_sec / 60.0:.1f} min;" if waited_sec else ""
        subject = "the primary provider was refusing requests" if wait_cause in PRIMARY_REFUSAL_KINDS else "the provider connection was unavailable"
        return (f"⚠️ The owner requested {action} while {subject}."
                f"{waited} No new summary request was sent. Any files written so far are preserved."
                + (provider_recovery_hint(accumulated_usage) if unknown else ""))
    if wait_cause in PRIMARY_REFUSAL_KINDS and not unknown:
        return ("⚠️ The primary provider kept refusing requests; its selected wait ended "
                f"after {waited_sec / 60.0:.1f} min. No new summary request was sent. "
                "Completed work and the original attempt-cost records are preserved.")
    if is_context_overflow:
        return (
            "⚠️ The context exceeded the selected model window; no further provider call was made. "
            "Any files written so far are preserved in the workspace."
        )
    if is_transport_wait:
        if unknown:
            # A wait duration does not prove a redial, nor a numeric price bound.
            return (
                "⚠️ The dispatched attempt has no confirmed provider outcome."
                f"{_unknown_wait_note(True, waited_sec, interactive)}"
                " Inspect the preserved facts before starting another run."
                f"{_unknown_terminal_recovery_hint(accumulated_usage)}{UNKNOWN_ATTEMPT_COST_NOTE}"
            )
        advice = ("Inspect the preserved facts before starting another run." if unknown
                  else "Retry when connectivity returns.")
        if interactive and waited_sec > 0:
            text = (
                "⚠️ Could not establish a provider connection; this turn waited and "
                f"redialed for {waited_sec / 60.0:.1f} min and ended as a provider outage, "
                f"not completed. {advice}"
            )
        elif interactive:
            text = (
                "⚠️ Could not establish a provider connection, and no wait window was left; "
                f"this turn ended as a provider outage, not completed. {advice}"
            )
        elif waited_sec > 0:
            text = (
                "⚠️ Could not establish a provider connection; the task waited and redialed "
                f"for {waited_sec / 60.0:.1f} min until its own limits ran out and ended as a provider outage, not completed. "
                f"Any files written so far are preserved in the workspace. {advice}"
            )
        else:
            text = (
                "⚠️ Could not establish a provider connection, and the owner deadline left no "
                "time to wait; the task ended as a provider outage, not completed. Any files "
                f"written so far are preserved in the workspace. {advice}"
            )
        return text
    if is_deadline_exhausted:
        text = "⚠️ The owner deadline ended primary model work; any files written so far are preserved."
        if unknown:
            text += _unknown_terminal_recovery_hint(accumulated_usage) + UNKNOWN_ATTEMPT_COST_NOTE
        return text
    return (
        "⚠️ The model provider returned no usable response."
        f"{_unknown_wait_note(unknown, waited_sec, interactive)}"
        f"{provider_failure_hint(accumulated_usage)}"
        f"{_unknown_terminal_recovery_hint(accumulated_usage) if unknown else provider_recovery_hint(accumulated_usage)}"
        f"{UNKNOWN_ATTEMPT_COST_NOTE if unknown else ''} "
        "Any files written so far are preserved in the workspace."
    )



# The host's own words for a failure class when the provider left no readable
# sentence (#1369). The typed kind stays on the record and in the llm_api_error
# event; a person reads the class, never the exception repr.
_FAILURE_KIND_WORDS = {
    "auth_error": "an authentication or authorization refusal",
    "quota_exhausted": "a quota or billing refusal",
    "subscription_window_exhausted": "a spent subscription window",
    "rate_limit": "a rate limit",
    "bad_request": "a rejected request",
    "request_too_large": "a request the provider found too large",
    "context_overflow": "a context overflow",
    "transport_unavailable": "no connection to the provider",
    "provider_outcome_unknown": "an attempt with no known outcome",
    "provider_transient": "a temporary provider failure",
    "provider_incomplete_response": "an incomplete response",
    "llm_empty_response": "an empty response",
    "llm_output_exhausted": "a reply that reached its length limit before any visible output",
    "provider_body_error": "an error in the provider's response",
    "provider_error": "a provider error",
    "model_substituted": "an answer from a model other than the requested one",
    "deadline_exhausted": "the owner deadline running out before dispatch",
}
# Logs hold the host's record, not necessarily the provider's whole sentence: a
# stream error frame's text reaches the durable event only as a bounded excerpt.
_DIAGNOSTIC_POINTER = "The task's Logs keep the host's record of this failure."
# The owner's quote of a provider sentence passes the SSOT display bound; past it
# the quote is shortened in words beside the Logs pointer, never silently cut.
_OWNER_QUOTE_LIMIT = 600


def owner_provider_message(failure: Any) -> str:
    """The provider's OWN sentence about a failed call, whole, or "" when there is none.

    ``failure`` is the raised exception — a structured error body's ``message``,
    a stream error's provider message — or an empty response's body-error dict
    (its ``message``). A Python exception repr is never one of those: it stays in
    the durable ``llm_api_error`` event and ``_last_llm_error`` for diagnostics
    (#1369). Nor is a Claudexor engine error's text: its ``display_message`` joins
    host-side diagnostics (stage, cause, engine code, the engine's own message)
    and its provider body stays private, so only its typed provider fields speak
    (``owner_provider_fields``). The owner's row bounds it (``provider_failure_hint``).
    """
    from ouroboros.llm_claudexor import ClaudexorModelError
    from ouroboros.utils import sanitize_tool_result_for_log

    if isinstance(failure, ClaudexorModelError):
        return ""
    if isinstance(failure, dict):
        authored = (failure.get("message"),)
    else:
        body = _exception_body(failure)
        nested = body.get("error") if isinstance(body.get("error"), dict) else {}
        authored = (nested.get("message"), body.get("message"), getattr(failure, "provider_message", None))
    text = next((value for value in authored if isinstance(value, str) and value.strip()), "")
    return sanitize_tool_result_for_log(" ".join(text.split())) if text else ""


def owner_provider_fields(failure: Any) -> Dict[str, str]:
    """The typed fields a provider itself returned through the Claudexor engine —
    its error ``code`` and the refused ``parameter`` — or {}. An unknown outcome
    exposes none, as in ``ClaudexorModelError.display_message``."""
    from ouroboros.llm_claudexor import ClaudexorModelError
    from ouroboros.utils import sanitize_tool_result_for_log

    if not isinstance(failure, ClaudexorModelError) or failure.code == "model_outcome_unknown":
        return {}
    context = failure.problem.get("context")
    context = context if isinstance(context, dict) else {}
    return {label: " ".join(sanitize_tool_result_for_log(value).split())
            for key, label in (("vendorCode", "code"), ("parameter", "parameter"))
            if isinstance(value := context.get(key), str) and value.strip()}


def stamp_owner_provider_message(accumulated_usage: Dict[str, Any], failure: Any) -> None:
    """Replace ``_last_llm_provider_message`` with ``failure``'s own sentence and
    ``_last_llm_provider_fields`` with its typed provider fields, or clear them.
    ``_last_llm_provider_message_cut`` marks a sentence its producer already cut
    (an empty response's body error keeps only a bounded ``message``).

    Stamped by the call owner (``loop_llm_call``) at each failed call, so an
    earlier round's sentence never speaks for this one.
    """
    cut = isinstance(failure, dict) and failure.get("message_truncated") is True
    for key, value in (("_last_llm_provider_message", owner_provider_message(failure)),
                       ("_last_llm_provider_message_cut", cut),
                       ("_last_llm_provider_fields", owner_provider_fields(failure))):
        accumulated_usage.pop(key, None)
        if value:
            accumulated_usage[key] = value


def provider_failure_hint(accumulated_usage: Dict[str, Any]) -> str:
    """The provider's own sentence (a long one shortened in words, with the Logs
    pointer), or the host's classification with that pointer.

    Quotes only ``_last_llm_provider_message`` — text a provider positively
    authored (``stamp_owner_provider_message``); typed provider fields without a
    sentence are named as the provider's beside the host's classification. The
    exception repr in ``_last_llm_error`` and an engine's composed display text
    are never appended: they belong to Logs and task details.
    """
    from ouroboros.utils import sanitize_tool_result_for_log, truncate_review_artifact

    message = " ".join(sanitize_tool_result_for_log(
        str(accumulated_usage.get("_last_llm_provider_message") or "")).split()).strip()
    cut = accumulated_usage.get("_last_llm_provider_message_cut") is True
    if message and not cut and truncate_review_artifact(message, limit=_OWNER_QUOTE_LIMIT) == message:
        return f" The provider said: \"{message}\""
    if message:  # its producer or the SSOT cut it (the SSOT's anti-waste floor lets a small overflow through whole)
        return f" The provider said: \"{message[:_OWNER_QUOTE_LIMIT].rstrip()}…\" (shortened). {_DIAGNOSTIC_POINTER}"
    kind = str(accumulated_usage.get("_last_llm_error_kind") or "").strip()
    if not kind and not str(accumulated_usage.get("_last_llm_error") or "").strip():
        return ""
    words = _FAILURE_KIND_WORDS.get(kind, "a provider failure")
    fields = accumulated_usage.get("_last_llm_provider_fields")
    fields = fields if isinstance(fields, dict) else {}
    named = [f'{label} "{fields[key]}"' for key, label in (("code", "error code"), ("parameter", "parameter"))
             if isinstance(fields.get(key), str) and fields[key].strip()]
    if named:  # the provider's own typed fields; its body, if any, stays private
        return (f" The provider reported {' and '.join(named)} but no sentence the host can quote; "
                f"the host classified the failure as {words}. {_DIAGNOSTIC_POINTER}")
    return f" The provider gave no readable message; the host classified the failure as {words}. {_DIAGNOSTIC_POINTER}"


# What the host KNOWS it did. It asks again and names no account; which account
# answers the redo is the engine's choice, so no wording here may claim the
# round moved (architecture: rotation is possible, not guaranteed).
_SUBSTITUTION_DISPOSITIONS = {
    "redo": "the answer was not accepted and the round was asked again without naming an account",
    "redos_exhausted": "the answer was not accepted and no further attempt was available",
    "pinned_account": "the account is pinned, so the round was not asked again",
    "admitted_candidate": "this send was already admitted, so the round was not asked again",
    "send_budget_spent": "this caller had no send left, so the round was not asked again",
    "deadline_spent": "the task's own time was spent, so the round was not asked again",
}


def emit_model_substitution(
    accumulated_usage: Dict[str, Any], *, task_id: str, emit_progress: Optional[Callable[..., None]],
) -> None:
    """Disclose once per task and requested model that another model answered.

    A timeline row of the task that spent the round, never a chat message and
    never a toast: the round recovers by itself, and the owner's interest is
    the cognitive horizon the task ran under, not an interruption. A second
    substitution of the same model in the same task stays in the durable rows.
    The sentence names what actually happened — a recovered redo and a refusal
    are different facts and must not share one wording.
    """
    rows = accumulated_usage.get("_model_substitutions")
    notified = accumulated_usage.setdefault("_model_substitution_notified", [])
    if emit_progress is None or not isinstance(rows, list):
        return
    for row in rows:
        requested, observed = str(row.get("requested") or ""), str(row.get("observed") or "")
        if not requested or not observed or requested in notified:
            continue
        notified.append(requested)
        account = str(row.get("account") or "")
        outcome = _SUBSTITUTION_DISPOSITIONS.get(str(row.get("disposition") or ""), "")
        emit_progress(
            f"⚠️ {observed} answered instead of the requested {requested}"
            f"{f' (Claudexor account {account})' if account else ''}"
            f"{f'; {outcome}' if outcome else ''}.",
            card_row="timeline",
            card_row_id=":".join(part for part in (task_id, "model_substitution", requested) if part),
        )


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
    if kind == "provider_outcome_unknown" or accumulated_usage.get("_pending_transport_outcome"):
        if accumulated_usage.get("transport_recovery"):
            return (" An earlier dispatched attempt still has no confirmed provider outcome. "
                    "Recovery used new task requests; their separate outcomes and any unknown costs remain recorded. "
                    "No additional summary request was sent.")
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
    if (accumulated_usage.get("resource_refusal") or {}).get("reason") == "unavailable" or (
            accumulated_usage.get("_last_llm_resource_refusal") == "unavailable"):
        reset = str(accumulated_usage.get("_last_llm_reset_at") or "")
        return (" No configured account could serve this request; quota exhaustion was not established."
                + (f" The engine reported a reset time of {reset}; availability remains unconfirmed." if reset else ""))
    if kind == "subscription_window_exhausted":
        reset_at = str(accumulated_usage.get("_last_llm_reset_at") or "").strip()
        when = f" It resets at {reset_at}." if reset_at else ""
        refusal = accumulated_usage.get("resource_refusal")
        if not refusal:
            return (
                " The subscription window for the delegated route is spent. This is "
                f"TRANSIENT, not a billing refusal — waiting cures it.{when} Nothing sleeps to that reset."
            )
        rotation, tried = refusal.get("account_rotation") or {}, ", ".join(refusal.get("fallbacks_tried") or [])
        # Only the engine's own pool verdict proves every account; any other stop claims nothing.
        accounts = (" The engine reports every compatible account blocked." if rotation.get("pool_exhausted") else
                    f" Account rotation stopped ({rotation.get('stop')}); other accounts are unproven."
                    if rotation else "")
        subject = ("No managed account can currently serve this model route (an engine-dated pool refusal, "
                   "not quota)" if refusal.get("reason") == "unavailable"
                   else "The subscription quota for this model route is spent")
        return (
            f" {subject}. This is "
            f"TRANSIENT, not a billing refusal — waiting cures it.{when}{accounts}"
            f"{f' Configured fallbacks tried without an answer: {tried}.' if tried else ''} Nothing "
            "more was sent, and nothing sleeps to that reset."
        )
    if kind == "model_substituted":
        return (
            " The route answered with a different model than the one requested, so "
            "the answer was not accepted. Another account may serve the requested "
            "model right away; the engine ranks this one lower for a while after this."
        )
    if kind == "bad_request" and str(accumulated_usage.get("_last_llm_provider_code") or "") == "invalid_continuation":
        # The generic bad_request sentence below blames the caller's transcript,
        # which is wrong here: the engine refused its OWN continuation record.
        return (
            " The provider refused the stored continuation of this conversation "
            "rather than the request itself. Dropping it and sending the same "
            "conversation again is the repair, and this round already spent it."
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
