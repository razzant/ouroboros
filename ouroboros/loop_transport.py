"""Transport-outage wait episodes and provider-failure terminal text for the main loop.

A REMOTE pre-dispatch transport failure (typed ``released`` custody: connect
refused/timed out before request bytes left this host, $0 in the ledger) is not a
model failure, and one provider's failure does not prove the egress dead: the
round first tries every configured route (``loop_model_call._recover_failed_round``),
and only when none answered does the round gate in ``loop.py`` latch a
:class:`TransportWaitEpisode` and wait. Two unsent sends are no model failure and
open no walk, which would write the work's whole start on another route; their
round waits on its model (``fallback_chain_allowed``): no free slot in the install's
own queue (``model_concurrency``), and its own Claudexor daemon unanswered before
create, which every Claudexor route shares (``engine_unanswered``). A wait is durable
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
follows the same order on every turn, including inline Presence: an acting primary's
upstream-observed continuation once, then the configured routes, then that continuation
again inside the same wait bounds; another acting route walks first, primary first.
Recovery is an owner note for every episode; an
error-kind change is a note for interactive turns only, because such a turn
has no progress row to show the closure — a managed task keeps the durable
row and its ordinary progress; exhaustion is a note for an interactive turn,
while a managed task's exhaustion is its terminal result. Every episode note is
the host speaking about the turn: it passes ``incident=None`` and keeps the
default voice, so an ``emit_progress`` callable handed to ``run_llm_loop`` must
accept the emitter's keyword facts (``incident``, ``narration``, ``reasoning``).

Provider terminal notices and their source facts share this owner.
"""

from __future__ import annotations

import logging
import pathlib
import queue
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from ouroboros.config import (
    NETWORK_WAIT_BACKOFF_START_SEC,
    NETWORK_WAIT_NOTE_INTERVAL_SEC,
    get_finalization_grace_sec,
    get_task_idle_timeout_sec,
)
from ouroboros.deadline_utils import parse_deadline_ts
from ouroboros.loop_llm_call import TIMEOUT_REPEAT_KEY, TRANSPORT_DEATHS_KEY, _TRANSIENT_BACKOFF_CAP_SEC
from ouroboros.llm_claudexor import daemon_not_delivered
from ouroboros.llm_probe import upstream_transport_reachable
from ouroboros.model_concurrency import capacity_refusal
from ouroboros.owner_mailbox import OwnerMailboxPeek
from ouroboros.transport_custody import ProviderNotDispatched
from ouroboros.utils import append_jsonl, utc_now_iso

log = logging.getLogger(__name__)

# Reserve for the final free redial near the owner deadline (Q14): round-top
# overhead (message drain, checkpoints, transcript seal, token measurement)
# routinely eats about a second on long transcripts, and a granted redial that
# the admission gate then refuses is a wasted grant.
_FINAL_REDIAL_MARGIN_SEC = 3.0


@dataclass
class TransportWaitEpisode:
    """One outage or chosen wait for a temporary primary refusal.

    The latched cause survives later fallback/admission errors overwriting usage;
    exhaustion stays deterministic, without forced-final resend. Direct-chat turns
    are ``interactive``; their local ``wait_bound_sec`` replaces absent queue rails.
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
    engine_unanswered: bool = False  # its latest failure: the own Claudexor daemon, before create (``engine_unanswered``)
    # A context-only place arrival shortens no backoff: it reaches the next round's drain at the scheduled redial.
    mailbox_peek: OwnerMailboxPeek = field(default_factory=lambda: OwnerMailboxPeek(waking_only=True), repr=False)

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
    """A running turn rejoins its accepted operation under its existing controls."""
    return bool(ctx is not None and getattr(ctx, "task_id", ""))


def inline_presence(ctx: Any) -> bool:
    """An inline Presence turn, which has no owner quota/auth question channel."""
    from ouroboros.dialogue_provenance import is_presence_task

    metadata = getattr(ctx, "task_metadata", None)
    return (getattr(ctx, "current_task_type", None) == "presence"
            or is_presence_task({"metadata": metadata if isinstance(metadata, dict) else {}}))


def new_generation_after_unknown(ctx: Any, accumulated_usage: Optional[Dict[str, Any]] = None) -> bool:
    """Whether this round may start a NEW, possibly charged, generation after its unknown outcome.

    Typed, never from prose or elapsed time: the receiving call has ended with no usable
    answer and no same-operation rejoin is viable. It is refused for an accepted operation
    whose READ failed (it may still finish; only it is rejoined), and for a round still
    holding paid repeats. Completed tools stay in the transcript and do not prevent this
    model call's recovery. It proves nothing about remote computation.
    """
    usage = accumulated_usage if accumulated_usage is not None else (getattr(ctx, "_accumulated_usage", None) or {})
    pending = usage.get("_pending_transport_outcome") or {}
    return bool(ctx is not None and getattr(ctx, "task_id", "")
                and not isinstance(usage.get(TRANSPORT_DEATHS_KEY), dict)
                and not pending.get("same_operation_recoverable"))


FORCED_ATTEMPT_KEY = "_forced_finalization_attempt"


class ReadyRoundAttempts:
    """Bind a native create or a forced send to its round's own landed ready checkpoint before dispatch."""

    def __init__(self, call, tracker, ids, request_ref):
        self.call, self.tracker = call, tracker
        self.reply_source = None
        self.ids = {**ids, "request_ref": request_ref.get("manifest_ref")}
        self.alternative = getattr(call, "defer_resource_wait", None) is not None
        self.saved_plan = object() if self.alternative else getattr(call, "context_fit_plan", None)

    def __call__(self, reservation):
        from ouroboros.llm_attempt import PhysicalAttemptPreparationFailed, PhysicalDispatchInterrupted
        from ouroboros.loop_messages import owner_source_sha256
        from ouroboros.model_wait import ModelWaitInterrupted
        from ouroboros.working_checkpoint import checkpoint_path, save_model_call
        from ouroboros.utils import atomic_write_json

        if self.tracker is not None:
            self.tracker(reservation)
        owner, native = self.call.tools._ctx, reservation.provider == "claudexor"
        reason = getattr(self.call, "forced_reason_code", "")
        if native or reason:
            self.reply_source = (reservation.attempt_id, owner_source_sha256(owner))
        if reason:
            # Save the disposition with the physical identity in ONE ready write; there may be no prior ready
            # boundary (e.g. a soft landing before round one), and the reply never owns an ordinary tool turn.
            self.call.accumulated_usage[FORCED_ATTEMPT_KEY] = {
                **self.ids, "physical_attempt_id": reservation.attempt_id, "provider": reservation.provider,
                "model": reservation.model, "forced_reason_code": reason, "owner_source_sha256": self.reply_source[1],
                "owner_directive_count": len(getattr(owner, "_owner_directives", None) or ())}  # old readers
            save_model_call()
            return
        seq = getattr(owner, "_working_checkpoint_seq", None)
        if native and (getattr(owner, "_working_checkpoint_owed", False) or seq and self.saved_plan is not self.call.context_fit_plan):
            # A create binds this round's own landed ready state, never an absent or earlier one: an owed write or a changed view lands first under the hold.
            try:
                save_model_call(model_call=self.call if self.alternative else None)
            except ModelWaitInterrupted as held:
                raise PhysicalDispatchInterrupted(held.control_reason) from held
            self.saved_plan, seq = self.call.context_fit_plan, getattr(owner, "_working_checkpoint_seq", None)
        if native and getattr(owner, "_working_checkpoint_owed", False):  # still owed: no writer is bound outside a loop's model-call scope
            raise PhysicalAttemptPreparationFailed("A native create cannot name a ready state that is still owed.", attempt_id=reservation.attempt_id)
        if not seq:
            return  # no write was ever attempted: an unsupported continuation sends as before, naming no sequence
        path = checkpoint_path(reservation.drive_root, self.call.task_id, getattr(owner, "task_attempt", None) or 1).with_suffix(".operation.json")
        if native:
            atomic_write_json(path, {**self.ids, "seq": seq, "physical_attempt_id": reservation.attempt_id,
                                     "model": reservation.model, "owner_source_sha256": self.reply_source[1]})
            owner._ready_native_receipt = True
        elif getattr(owner, "_ready_native_receipt", False):
            path.unlink(missing_ok=True)
            owner._ready_native_receipt = False

    def attempt_ended(self, capture, error):
        if self.tracker is not None:
            self.tracker.attempt_ended(capture, error)


def _record_released_operation(usage, physical_id):
    usage["ledger_attempt_ids"] = list(dict.fromkeys([*usage.get("ledger_attempt_ids", []), physical_id]))
    if (usage.get("_pending_transport_outcome") or {}).get("physical_attempt_id") == physical_id:
        usage.pop("_pending_transport_outcome")
        if usage.get("_last_llm_error_kind") == "provider_outcome_unknown":
            usage.pop("_last_llm_error_kind")
            if usage.get("execution_status") == "infra_failed":
                usage.pop("execution_status")
    record = usage.get(TIMEOUT_REPEAT_KEY) or {}
    repeat = record.get("repeat") or {}
    if record.get("state") in {"dispatched", "spent"} and repeat.get("physical_attempt_id") == physical_id:
        record.update(state="spent", repeat={**repeat, "attempt_custody_state": "released",
                      "failure_code": "model_operation_not_started", "error": "The recorded repeat ended before dispatch."})
        usage["_last_llm_attempt_custody"] = "released"


def ready_operation(state, usage, root, task_id, attempt_id=""):
    """Bind a restored round to a read, never a dispatch. A forced boundary without a reader still ends. A named
    ``attempt_id`` binds only that physical attempt: another one is refused before it touches ``usage``."""
    if state.get("task_id") != task_id:
        return None
    forced = (state.get("usage") or {}).get(FORCED_ATTEMPT_KEY)
    if isinstance(forced, dict) and forced.get("forced_reason_code") and forced.get("physical_attempt_id"):
        adopted = (usage.get("_last_llm_call_meta") or {}).get("physical_attempt_id") == forced["physical_attempt_id"]
        if not adopted and forced.get("provider") == "claudexor":
            if pending := _ready_native_operation(state, usage, root, task_id, forced=forced, expected=attempt_id):
                return pending
        return {**forced, "operation_id": ""}  # no matching reader, proven unsent, or already incorporated
    return _ready_native_operation(state, usage, root, task_id, expected=attempt_id)


def _ready_native_operation(state, usage, root, task_id, *, forced=None, expected=""):
    from ouroboros import usage_store
    from ouroboros.llm_claudexor import processing_recovery_candidate
    from ouroboros.loop_messages import _recovered_owner_source_sha256
    from ouroboros.loop_output_exhaustion import kept_arrivals
    from ouroboros.observability import call_manifest_path
    from ouroboros.utils import read_json_dict
    from ouroboros.working_checkpoint import native_attempt_at

    boundary = (state.get("working") or {}).get("boundary")
    pending = usage.get("_pending_transport_outcome") or {}
    repeat = usage.get(TIMEOUT_REPEAT_KEY) or {}
    failure = usage.get("_last_llm_failure") or {}
    legacy_unknown = False
    if native := forced or native_attempt_at(state):
        attempt = native
    elif boundary == "ready" and repeat.get("state") == "dispatched":
        # The one-repeat owner wrote a newer ready sequence after the pre-dispatch tracker.
        attempt = repeat.get("repeat") or {}
    elif (pending.get("paused_receipt_read") or pending.get("same_operation_recoverable") and pending.get("llm_call_id")) and (
            state.get("resume_point") or boundary in {"ready", "resumed"}):
        attempt = pending  # This very read was checkpointed again, including by an exact pause.
    elif boundary != "ready":
        return None
    else:
        # Older ready states kept the pending physical ID and the round on its matching failure.
        legacy_unknown = pending.get("same_operation_recoverable") is not True
        if (not pending.get("physical_attempt_id") or pending["physical_attempt_id"] != failure.get("physical_attempt_id")
                or failure.get("round") is None or failure["round"] != state.get("round_idx")
                or pending.get("model") and failure.get("model") not in (None, pending["model"])
                or legacy_unknown and pending.get("outcome") != "unknown"):
            return None
        call_id = failure.get("llm_call_id") or pending.get("llm_call_id")
        request_ref = pending.get("request_ref")
        if call_id and isinstance(request_ref, dict) and request_ref.get("call_id") not in (None, f"{call_id}_request"):
            return None
        logical = (read_json_dict(call_manifest_path(root, task_id, f"{call_id}_request")) or {}) if call_id else {}
        ids = {"llm_call_id": call_id, "round": failure["round"],
               **{key: logical[key] for key in ("execution_id", "round_id") if key in logical}}
        if logical and (logical.get("task_id") != task_id or logical.get("llm_call_id", call_id) != call_id
                        or logical.get("round", failure["round"]) != failure["round"]):
            return None
        if any(pending.get(key) not in (None, value) for key, value in ids.items()):
            return None
        attempt = {**pending, **ids}
    physical_id = attempt.get("physical_attempt_id")
    role = state.get("context_model_role") or attempt.get("model_role")
    execution_id = usage.get("execution_id")
    round_id = f"{execution_id}:round:{state.get('round_idx')}" if execution_id else attempt.get("round_id")
    if (not physical_id or expected and physical_id != expected or attempt.get("round") != state.get("round_idx")
            or execution_id and attempt.get("execution_id") not in (None, execution_id)
            or round_id and attempt.get("round_id") not in (None, round_id)
            or role and attempt.get("model_role") and attempt["model_role"] != role):
        return None
    with usage_store.read(root) as store:
        row = store.attempt(physical_id)
    observed_round = ((row or {}).get("physical_context") or {}).get("round_id")
    if row and (row["task_id"] != task_id or row["provider"] != "claudexor"
                or attempt.get("model") and attempt["model"] != row["model"]
                or round_id and observed_round and observed_round != round_id):
        return None
    if row and row["state"] == "reserved":
        return None  # mark_dispatched precedes create
    request = read_json_dict(call_manifest_path(root, task_id, f"{physical_id}_model_request")) or {}
    if legacy_unknown and not row and request.get("invocation_id") != physical_id:
        return None  # A legacy transport error alone does not identify a native chat.
    receipt = (retained := read_json_dict(call_manifest_path(root, task_id, f"{physical_id}_model_response")) or {}) or request
    role = role or request.get("model_role")
    model = (row or {}).get("model") or attempt.get("model") or request.get("model")
    operation = request.get("operation_id") or receipt.get("operation_id")
    for source in (request, receipt):
        if (source.get("invocation_id") != physical_id or source.get("task_id") != task_id
                or source.get("request_kind", "chat") != "chat" or source.get("model_role", role) != role
                or source.get("model", model) != model):
            operation = None
    saved_ids = [source["operation_id"] for source in (attempt, pending, failure)
                 if source.get("physical_attempt_id") == physical_id and source.get("operation_id")]
    if any(saved_id != operation for saved_id in saved_ids) or receipt.get("operation_id") not in (None, "", operation):
        operation = None
    if row and row["state"] == "released" and not (operation and (processing_recovery_candidate(row) or retained.get("dispatch_state") == "not_started")):
        _record_released_operation(usage, physical_id)  # no retained reason to read: the owed round is admitted again
        return None
    if not operation and repeat.get("state") in {"dispatched", "spent"}:
        return None  # The timeout owner already recorded the dispatch and owns its no-resend terminal.
    # The address authorizes a read. Losing it cannot undo a dispatch or establish a zero charge.
    bound = {**attempt, "operation_id": saved_ids[0] if saved_ids else operation or "", "model_role": role, "model": model,
             "owner_source_sha256": _recovered_owner_source_sha256(state, attempt, row),
             "outcome": "unknown", "same_operation_recoverable": True}
    bound.update(kept_arrivals(pending, bound))  # a Pause kept the drains' facts with this read (``read_ready_operation``)
    usage["_pending_transport_outcome"] = bound
    usage["_last_llm_error_kind"] = "provider_outcome_unknown"
    if usage.get("execution_status") == "infra_failed":
        usage.pop("execution_status")  # Lost contact on the previous worker. This worker still owes the read.
    usage["ledger_attempt_ids"] = list(dict.fromkeys([*usage.get("ledger_attempt_ids", []), physical_id]))
    return bound if bound["operation_id"] == (operation or "") else {**bound, "operation_id": ""}  # the record itself: what a drain binds stays


def read_ready_operation(ctx, pending):
    """Consume the identified attempt in this round, without passing through any sender."""
    from ouroboros._usage_response import assistant_answer_text, refusal_content_text, usage_reply_usability
    from ouroboros.budget_pause import enter_owner_pause
    from ouroboros.gateways.claudexor import ClaudexorUnavailable
    from ouroboros.llm import add_usage
    from ouroboros.llm_claudexor import ClaudexorModelError, ClaudexorRefusalError, recover_chat_reply, restored_end
    from ouroboros.llm_observability import persist_main_response
    from ouroboros.loop_llm_call import (_account_received_error, _classify_empty_response, _remember_llm_call,
                                         _replace_response_meta, _take_response_facts, classify_llm_exception)
    from ouroboros.loop_output_exhaustion import OWNER_PAUSE_CONTROLS, paused_read, poll_read_control, record_empty_cut, response_facts
    from ouroboros.model_wait import ModelWaitInterrupted, accepted_operation_scope, current_model_wait
    from ouroboros.owner_pause import wait_member_fence
    from ouroboros.usage_accounting import BudgetExceeded

    usage, owner = ctx.accumulated_usage, ctx.tools._ctx
    mailbox = OwnerMailboxPeek()
    if pending.get("execution_id"):
        usage.setdefault("execution_id", pending["execution_id"])
    carry = not pending.get("forced_reason_code") and TIMEOUT_REPEAT_KEY not in usage  # this round's own call may follow

    def poll_control():
        return poll_read_control(ctx, owner, mailbox)

    def interrupt(reason, cause=None, *, reply=None):
        if reason in OWNER_PAUSE_CONTROLS:  # the poll met the owner's Pause: this read stays owed, its operation running
            paused_read(usage, pending)
        interrupted = ModelWaitInterrupted(reason, role=pending["model_role"], cause=cause)
        interrupted.operation_id = pending["operation_id"]
        interrupted.recovered_attempt_id = pending["physical_attempt_id"]
        interrupted.read_operation = pending  # the exact read it ended: its control owner rules, a wrap-up's rail ends it
        interrupted.ledger_attempt_ids = [pending["physical_attempt_id"]]
        if reply is not None:
            interrupted.received_main_reply = (*reply, pending.get("owner_source_sha256"))
        if isinstance(cause, ClaudexorRefusalError):
            interrupted.received_refusal = cause
        raise interrupted from cause

    ending_controls = {"cancelled", "finalize_requested", "execution_deadline", "absolute_ceiling", "deadline"}

    def ending_control():
        with accepted_operation_scope():
            waiter = current_model_wait()
            reason = poll_control() or (waiter.control_reason() if waiter else None)
        return reason if reason in ending_controls else None

    def check_control(cause=None, *, reply=None):
        if reason := pending_control or ending_control():
            interrupt(reason, cause, reply=reply)

    def terminal_failure(kind, message, *, cause=None, route=None, provider_code="", status_code=0):
        usage.update(_last_llm_error_kind=kind, _last_llm_error=message, _last_llm_attempt_custody="settled",
                     _last_llm_provider_code=provider_code, _last_llm_status_code=status_code, execution_status="infra_failed")
        usage["_last_llm_failure"] = {**{key: pending.get(key) for key in (
            "operation_id", "physical_attempt_id", "model", "model_role", "llm_call_id")}, "round": ctx.round_idx,
            "error": message, "failure_code": kind, "attempt_custody_state": "settled", "observed_route": dict(route or {}),
            "restored_attempt": True, "recovered_operation_id": pending["operation_id"],
            "provider_code": provider_code, "status_code": status_code,
            **getattr(cause, "received_status", {})}
        usage.pop("_pending_transport_outcome", None)
        usage.pop(TRANSPORT_DEATHS_KEY, None)
        check_control(cause)
        return None, None

    def received_end(error, classification, kind=""):  # a received failure or unusable reply ended known: its round's call meets it
        stamp_owner_provider_message(usage, error, classification)
        known = terminal_failure(kind or classification.kind, error.display_message, cause=error, route=error.route, provider_code=classification.provider_code or error.code, status_code=classification.status_code)
        if carry:  # (a control met first ended the read above)
            restored_end(usage, pending, ctx.round_idx, error)
        return known

    try:
        result = (recover_chat_reply(ctx.status_drive_root, ctx.task_id, pending, getattr(owner, "model_turn_state", None),
                                     poll_control=poll_control)
                  if pending["operation_id"] else None)
    except BudgetExceeded:
        # The exact budget Resume still owes this read, even if its balance now allows a new call.
        usage["_pending_transport_outcome"] = {**pending, "paused_receipt_read": True}
        raise
    except (ClaudexorModelError, ProviderNotDispatched) as error:
        if getattr(error, "control_reason", None):
            interrupt(error.control_reason, error)
        result = error
    except (ClaudexorUnavailable, OSError, ValueError, KeyError, TypeError):
        log.info("Recorded chat operation %s could not be read", pending["operation_id"], exc_info=True)
        result = None

    # Park after durable settlement and before adding the receipt to this round's usage.
    # An exact Resume will read it from retained storage and count it once.
    pending_control = ending_control()
    if not pending_control:
        with accepted_operation_scope():
            try:
                if wait_member_fence(owner):
                    paused_read(usage, pending)
                    enter_owner_pause(ctx)
            except ModelWaitInterrupted as error:
                if error.control_reason not in ending_controls:
                    raise
                pending_control = error.control_reason

    if isinstance(result, ProviderNotDispatched):
        _record_released_operation(usage, pending["physical_attempt_id"])
        check_control(result)
        if carry and isinstance(result, ClaudexorModelError):  # the engine's own typed refusal: its round's call meets it
            restored_end(usage, pending, ctx.round_idx, result)
        raise result
    if isinstance(result, ClaudexorModelError):
        error = result
        _account_received_error(usage, error, pending["physical_attempt_id"])
        if isinstance(error, ClaudexorRefusalError):
            terminal_failure(error.code, error.display_message, cause=error, route=error.route, provider_code=error.code)
            raise error
        classification = classify_llm_exception(error)
        if classification.kind != "provider_outcome_unknown":
            return received_end(error, classification)
        usage.update(_last_llm_error=error.display_message, _last_llm_provider_code=str(error.problem.get("code") or ""))  # why it stays unknown
        result = None

    if result is None:
        check_control()
        usage["execution_status"] = "infra_failed"
        return None, None
    msg, reported, _physical_ref = result
    _take_response_facts(reported, msg, usage)
    _replace_response_meta(response_facts(owner), reported, msg)
    if first := pending["physical_attempt_id"] not in usage.get("_counted_error_attempts", ()):  # a withdrawn control's or an
        add_usage(usage, reported)  # exact Resume's re-read applies it again, never counting or recording it twice
        usage.setdefault("_counted_error_attempts", []).append(pending["physical_attempt_id"])
        call_ids = {**{key: pending[key] for key in
            ("llm_call_id", "execution_id", "round_id", "attempt", "model_role") if key in pending}, "round": ctx.round_idx,
            "physical_attempt_id": pending["physical_attempt_id"], "operation_id": pending["operation_id"],
            "operation_outcome": reported["claudexor"]["outcome"]}
        model = reported["resolved_model"]
        response_ref = persist_main_response(ctx.drive_logs.parent, ctx.task_id, call_ids, msg, reported, usage,
                                             model=model, display_model=model, provider="claudexor")
        _remember_llm_call(usage, call_ids=call_ids, model=model, display_model=model, provider="claudexor", use_local=False,
            request_ref={"manifest_ref": pending.get("request_ref")}, response_ref=response_ref, reported_model=reported["resolved_model"],
            requested_profile=reported["claudexor"]["requested_profile"], observed_route=reported["claudexor"]["route"])
    content = msg.get("content")
    if usage_reply_usability(reported, msg) == "unusable" or not (
            msg.get("tool_calls") or assistant_answer_text(content).strip() or refusal_content_text(content) is not None):
        if (not pending.get("forced_reason_code") and TIMEOUT_REPEAT_KEY not in usage
                and reported.get("response_status") == "incomplete"
                and _classify_empty_response(reported, msg) == ("llm_empty_response", False, True)):
            record_empty_cut(usage, reported, custody="settled")  # no sent cap on a Claudexor reply
            usage.pop("_pending_transport_outcome", None)
            check_control()
            return None, reported["cost"]
        rejected = ClaudexorModelError({"code": "response_rejected", "message": "The recorded operation returned no usable model reply."}, model_role=pending["model_role"], operation_id=pending["operation_id"], route=reported["claudexor"]["route"])
        rejected.usage = {"claudexor": reported["claudexor"]}  # its receipt's state and outcome, counted once above: a rejected end
        return received_end(rejected, classify_llm_exception(rejected), "response_rejected")  # the read's row keeps its own code
    if pending.get("llm_call_id") and not pending.get("forced_reason_code"):
        usage["_last_llm_call_meta"]["usable_solve_response"] = True
    for stale in ("execution_status", "result_status", "reason_code", TIMEOUT_REPEAT_KEY, TRANSPORT_DEATHS_KEY, "_pending_transport_outcome"):
        usage.pop(stale, None)
    usage["rounds"] = usage.get("rounds", 0) + int(first)
    check_control(reply=(msg, reported))
    return msg, reported["cost"]


def revoke_endpoint_grant(usage, episode, *, keep_latest_failure=False):
    usage.pop("_observed_transport_endpoint", None)
    usage.setdefault("_pending_transport_outcome", usage["transport_recovery"]["previous_attempt"])
    if not keep_latest_failure:
        usage["_last_llm_error_kind"] = "provider_outcome_unknown"
    if episode is not None:
        episode.continuation_granted = False
    return None, None


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
                               task_id: str, model: str, emit_progress: Callable, use_local: bool = False) -> bool:
    """Grant one NEW attempt only after upstream recovery, retaining old custody."""
    from ouroboros.deadline_utils import dispatch_window_remaining_sec
    from ouroboros.llm import LLMClient
    remaining = dispatch_window_remaining_sec(deadline_ts=task_deadline_epoch(tools), reserve_sec=get_finalization_grace_sec())
    if remaining is not None and remaining <= 0:
        return False
    # Metadata carries no cognitive in-flight lease and may stand in front of the configured routes:
    # the ordinary connection allowance on every route, never the LLM read window.
    timeout = float(LLMClient._no_proxy_timeout().connect)
    if remaining is not None:
        timeout = min(timeout, remaining)
    from ouroboros.model_slots import route_binding, task_model_binding
    from ouroboros.model_wait import current_model_wait
    ctx, waiter = tools._ctx, current_model_wait()
    if transport_repeat_stop_requested(ctx) or (waiter is not None and waiter.control_reason()):
        return False

    def selected_binding():
        overrides = waiter.overrides if waiter else None
        role, account = task_model_binding({"model_role": getattr(ctx, "model_role", ""),
            "task_metadata": getattr(ctx, "task_metadata", {})},
            context_fit_plan=getattr(ctx, "context_fit_plan", None), overrides=overrides)
        selected, local, role_account = route_binding(model, use_local, role, overrides=overrides)
        return selected, local, role, role_account if account is None else account

    binding = selected_binding()
    selected, local, role, account = binding

    def eligible():
        return (episode.wait_cause == "provider_outcome_unknown" and not episode.continuation_granted
                and new_generation_after_unknown(ctx, accumulated_usage)
                and not episode.outcome_custody.get("same_operation_recoverable"))

    if not eligible():
        return False
    observed = upstream_transport_reachable(llm, selected, timeout=timeout, use_local=local, model_role=role,
        account_override=account, observed_after=episode.started_at,
        expected_route=(episode.outcome_custody.get("route")
                        if episode.outcome_custody.get("model", selected) == selected
                        and episode.outcome_custody.get("model_role", role) in ("", role) else None))
    if local and (observed.get("kind") != "local_server_http" or not observed.get("endpoint_sha256")):
        observed = {}
    remaining = dispatch_window_remaining_sec(deadline_ts=task_deadline_epoch(tools), reserve_sec=get_finalization_grace_sec())
    if not observed:  # the row that says why this round went on to its configured routes or kept waiting
        emit_network_wait_event(drive_logs, task_id=task_id, phase="continued", elapsed_sec=episode.waited_sec,
                                redials=episode.redials, model=selected, detail="upstream_not_observed")
    if (not observed or not eligible() or remaining == 0.0 or transport_repeat_stop_requested(ctx)
            or (waiter is not None and waiter.control_reason()) or selected_binding() != binding):
        return False
    # The latest outstanding unknown is the attempt this generation follows: since the episode opened on the
    # primary's lost answer, a configured route may have lost its own.
    previous = dict(accumulated_usage.get("_pending_transport_outcome") or episode.outcome_custody)
    append_unknown_recovery_input(messages, accumulated_usage, previous, connectivity=observed,
                                  lead="Upstream connectivity is available again.", continuation="new_physical_attempt")
    if observed.get("endpoint_sha256"):
        accumulated_usage["_observed_transport_endpoint"] = observed["endpoint_sha256"]
    else:
        accumulated_usage.pop("_observed_transport_endpoint", None)
    accumulated_usage.pop(TRANSPORT_DEATHS_KEY, None)
    accumulated_usage.pop("_pending_transport_outcome", None)
    episode.continuation_granted = True
    emit_network_wait_event(drive_logs, task_id=task_id, phase="recovered",
        elapsed_sec=episode.waited_sec, redials=episode.redials, model=selected,
        detail="new_attempt_after_unknown_outcome", outcome_custody=previous)
    cost_note = "" if local else " Any unreported cost remains unknown; another charge is possible."
    emit_progress("🌐 Connection restored — continuing from saved work in a new attempt. "
                  "The prior result is still unknown." + cost_note, incident=None)
    return True


PRIMARY_REFUSAL_KINDS = frozenset({"provider_transient", "rate_limit"})


SLOT_WAITED_KEY = "_slot_waited_sec"  # a work's seconds in the install's own queue so far (its facts line states them)


def slot_wait_note(emit_progress: Optional[Callable[[str], Any]], usage: Any = None) -> Callable[[float, bool], None]:
    """What a send waiting in the install's own queue gives its work (``model_concurrency.slot_wait_notes``): the
    words that nothing was sent or charged and the work keeps its model; and, when the send gets its slot, the
    seconds it stood in line, added to the work's own count for its next facts line."""
    def note(waited_sec: float, admitted: bool = False) -> None:
        if admitted:
            if isinstance(usage, dict):
                usage[SLOT_WAITED_KEY] = float(usage.get(SLOT_WAITED_KEY) or 0.0) + waited_sec
        elif emit_progress is not None:
            waited = f"{waited_sec / 60.0:.0f} min so far; " if waited_sec >= 60 else ""
            emit_progress("🌐 No local model slot is free under Requests at once per model — this request waits in "
                          f"line ({waited}nothing sent, $0) and keeps its model.")
    return note


def engine_unanswered(usage: Any) -> bool:
    """The acting route's latest attempt met the install's own Claudexor daemon unanswered before create
    (``llm_claudexor.daemon_not_delivered``; a failed walk's candidate row never stands in for it). Like a full local
    queue, it is no failure of the model: every Claudexor route shares that daemon, so the work waits on its model."""
    if not isinstance(usage, dict) or usage.get("_last_llm_error_kind") != "transport_unavailable":
        return False
    return daemon_not_delivered(usage.get("_acting_route_failure") or usage.get("_last_llm_failure"))


def primary_refusal_wait(ctx: Any, error_kind: str) -> bool:
    """A declared primary wait keeps temporary refusals on ordinary paced recovery."""
    return bool(getattr(ctx, "route_wait_on_primary", False) and error_kind in PRIMARY_REFUSAL_KINDS
                and not inline_presence(ctx) and not getattr(ctx, "exact_model_route", False))


def fallback_chain_allowed(
    ctx: Any, last_error_kind: str, episode: Optional[TransportWaitEpisode],
    accumulated_usage: Optional[Dict[str, Any]] = None, *, primary_acting: bool = False,
) -> bool:
    """Whether this round may try configured alternatives for the current failure.

    Existing outage pacing bounds repeated attempts; no lifetime pass limit is added.
    One lost answer is evidence about an attempt, not its route: an acting primary's
    eligible unknown outcome with no wait open takes its own continuation first (the
    episode's entry step, then ``continue_unknown_transport``; owner decision 2026-11-11).
    A missed observation walks from the open episode; the new attempt's outcome takes that kind's recovery
    (lost again or undelivered walks at once, a cut keeps its ladder), as does another acting route's unknown one. A declared
    wait on the primary (``switch_model(primary="wait")``) keeps its refusals and
    outages on the primary's own wait instead of paid alternatives; after its content filter's
    stop nothing is there to wait for, so this run ends as interrupted (owner, 12.11). A context overflow
    reaches the configured routes only after the primary's own recovery ladder
    (``loop_model_call._recover_context_overflow``) left the round unanswered: the
    candidates inherit the compacted view (owner decision 7A, rung f). A send that found
    no slot in the install's own queue (``model_concurrency``) never opens them: the model did not fail, and a move
    would write the work's whole start on another route (owner, 13.11). Nor does the acting route's own Claudexor daemon left unanswered once its same-route repeats are spent (``engine_unanswered``;
    owner wait-first, 13.11): the round's episode waits; a candidate's daemon failure mid-walk stops no walk.
    """
    usage = accumulated_usage or {}
    if bool(getattr(ctx, "exact_model_route", False)):
        return False
    if isinstance(usage.get(TRANSPORT_DEATHS_KEY), dict):
        # The round still holds an unresolved attempt (a granted transport-death
        # repeat with no usable response since): no paid candidate may dial over
        # it, whatever the last kind says.
        return False
    if primary_refusal_wait(ctx, last_error_kind) or capacity_refusal(usage) or engine_unanswered(usage):
        return False  # a chosen wait on the primary; the install's own full queue or silent daemon is no model failure
    if getattr(ctx, "route_wait_on_primary", False) and (
            last_error_kind in ("transport_unavailable", "provider_outcome_unknown", "content_filter")
            or usage.get("_last_llm_resource_refusal")):
        return False
    if last_error_kind == "provider_outcome_unknown":
        return new_generation_after_unknown(ctx, accumulated_usage) and not (primary_acting and episode is None)
    if last_error_kind == "llm_output_exhausted":
        from ouroboros.loop_output_exhaustion import ready_for_fallback
        return ready_for_fallback(usage) and not getattr(ctx, "route_wait_on_primary", False)
    if (usage.get("_last_llm_failure") or {}).get("rejected_receipt"):
        return False  # an accepted receipt judged unusable, live or restored, ended known: no candidate sends over it
    return last_error_kind not in ("deadline_exhausted", "output_recovery_interrupted")


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
    by_configured_route: bool = False, use_local: bool = False,
) -> Optional[TransportWaitEpisode]:
    """Reconcile the episode latch with one dispatch outcome.

    A declared primary wait also admits temporary provider refusals after ordinary retries,
    without treating a ready account catalog as proof of generation recovery.
    Enters a new episode on a fresh ``transport_unavailable`` failure, or an
    eligible unknown outcome (``new_generation_after_unknown``), that the
    configured routes did not answer or that opened none (a full local queue, the
    own daemon unanswered: ``fallback_chain_allowed``), or ahead of them for an acting primary's first
    lost answer (durable ``entered`` event; the first owner note fires immediately). An interactive
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
    latch for the wait/terminal step. A configured route's answer (``by_configured_route``)
    ends it as ``configured_route_answered``, no recovery note: the walk says who answered.
    A granted continuation that fails again — unknown after dispatch, or
    released before it — returns to the SAME episode (work-order §B9 "repeated
    unknown creates no burst"), and so does a free redial that crosses dispatch
    and dies unknown: the latch keeps its elapsed clock, growing backoff and
    redial count (one per wait iteration), switches its cause to the latest
    failure, and for an unknown outcome refreshes the custody and re-arms the
    probe's freshness bound at that failure. A granted continuation whose round
    made no new attempt (a failed probe, a call that yielded to control) keeps
    its grant untouched. An answer or any other failure class ends the latch
    exactly as before. ``engine_unanswered`` follows each redial's own failure (a deadline
    refusal tries nothing and keeps it): a silent own daemon sets it; any other transport
    failure, a dispatched unknown or a received refusal clears it.
    """
    lost_note = ("🌐 Provider connection was lost after dispatch. The outcome is unknown. "
                 "Waiting for connectivity before a new attempt on the built-in server." if use_local else
                 "🌐 Provider connection was lost after dispatch. The outcome and any unreported cost remain unknown. "
                 "Waiting for connectivity, then continuing from saved work with a new attempt; another charge is possible.")
    pending = dict((getattr(ctx, "_accumulated_usage", {}) or {}).get("_pending_transport_outcome") or {})
    unknown_again = error_kind == "provider_outcome_unknown" and new_generation_after_unknown(ctx)
    refused = not msg_present and primary_refusal_wait(ctx, error_kind)

    def event(phase, elapsed_sec, detail="", **custody):  # one durable row of the current episode, at its redial count
        emit_network_wait_event(drive_logs, task_id=task_id, phase=phase, elapsed_sec=elapsed_sec, redials=episode.redials, model=model, detail=detail, **custody)

    if refused and episode is not None:
        # Catalog readiness is not generation recovery. Keep the existing episode's
        # pacing, elapsed bound and prior unknown custody; only a real answer ends it.
        # A received refusal is no silent daemon: the engine's words leave with that failure.
        episode.wait_cause, episode.continuation_granted, episode.engine_unanswered = error_kind, False, False
        return episode
    if (episode is not None and not episode.continuation_granted and not msg_present and pending and not unknown_again
            and error_kind == episode.wait_cause == "provider_outcome_unknown"):
        episode = None  # as the granted case below: an operation that may still finish is only rejoined
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
            episode.engine_unanswered = engine_unanswered(getattr(ctx, "_accumulated_usage", None))
            if error_kind == "provider_outcome_unknown":
                episode.started_at = time.time()
                episode.outcome_custody = pending
            event("continued", time.monotonic() - episode.started_monotonic,
                  "continuation_outcome_unknown" if error_kind == "provider_outcome_unknown"
                  else "continuation_transport_unavailable", outcome_custody=episode.outcome_custody)
            if error_kind == "transport_unavailable":  # the grant note said "continuing"; the owner must hear otherwise
                emit_progress("🌐 The new attempt could not reach the provider — waiting and redialing "
                              "automatically (that attempt was $0).", incident=None)
            return episode
    if (episode is not None and not msg_present
            and episode.wait_cause != "provider_outcome_unknown" and unknown_again):
        # A formerly free redial crossed dispatch and died unknown: the same
        # episode now needs upstream proof before its next attempt; the clock,
        # backoff and redial count carry over instead of a fresh 4s episode.
        # That attempt was sent: no "not sent" or silent-engine words remain.
        episode.wait_cause, episode.engine_unanswered = "provider_outcome_unknown", False
        episode.started_at = time.time()
        if pending:
            episode.outcome_custody = pending
        event("continued", time.monotonic() - episode.started_monotonic, "redial_outcome_unknown",
              outcome_custody=episode.outcome_custody)
        emit_progress(  # the "$0 redials" framing of the entry note no longer holds
            lost_note,
            incident=None,
        )
        return episode
    if episode is None:
        unknown = unknown_again
        if msg_present or (error_kind != "transport_unavailable" and not unknown and not refused):
            return None
        from ouroboros.task_carrier import conversation_origin
        interactive = conversation_origin(ctx)  # the owner's foreground turn, carried or inline
        episode = TransportWaitEpisode(
            wait_cause=error_kind,
            outcome_custody=dict((getattr(ctx, "_accumulated_usage", {}) or {}).get("_pending_transport_outcome") or {}),
            started_monotonic=time.monotonic(),
            interactive=interactive,
            wait_bound_sec=float(get_task_idle_timeout_sec()) if interactive else None,
            engine_unanswered=engine_unanswered(getattr(ctx, "_accumulated_usage", None)),
        )
        event("entered", 0.0, error_kind if refused else "")
        episode.last_note_monotonic = time.monotonic()
        # Interactive notes keep their existing wording; direct-turn Stop is
        # separately handled through its typed mailbox control.
        emit_progress(
            ("🌐 The primary provider temporarily refused the request. Waiting as chosen, then continuing "
             "on the same primary with paced requests; prior attempt costs remain recorded." if refused else
             lost_note
             if unknown else "🌐 No local model slot was free, so this request was not sent — waiting under Requests "
             "at once per model and trying again automatically ($0)." if capacity_refusal(getattr(ctx, "_accumulated_usage", None)) else
             "🌐 The local Claudexor engine is not answering, so this request was not sent ($0 for it). The work keeps "
             "its model, tries no other route and retries automatically within its existing limits." if episode.engine_unanswered else
             "🌐 The request did not reach the provider — waiting and redialing automatically (failed attempts are $0).")
            + ("" if interactive else " Stop cancels."),
            incident=None,
        )
        return episode
    elapsed = time.monotonic() - episode.started_monotonic
    if msg_present and by_configured_route:  # the waited route proved nothing
        event("ended", elapsed, "configured_route_answered")
        return None
    if msg_present:
        event("recovered", elapsed)
        emit_progress(
            (f"🌐 Primary provider answered after {elapsed / 60.0:.1f} min — resuming."
             if episode.wait_cause in PRIMARY_REFUSAL_KINDS else
             f"🌐 Model response received after {elapsed / 60.0:.1f} min — resuming."
             if episode.engine_unanswered else
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
        event("ended", elapsed, f"error_kind_changed:{error_kind}")
        if episode.interactive:  # a managed task keeps its durable row and ordinary progress
            emit_progress(
                f"🌐 Provider connection restored after {elapsed / 60.0:.1f} min — the redial "
                f"got past the connect phase and failed as {error_kind}; ordinary failure policy resumes.",
                incident=None,
            )
        return None
    if error_kind == "transport_unavailable":  # the latest redial's own failure words the next notes
        episode.engine_unanswered = engine_unanswered(getattr(ctx, "_accumulated_usage", None))
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

    Transport and wait callers wake for what their ``waking_only`` peek counts: every
    message but a context-only place arrival. Owner admission uses the same typed
    provenance boundary as the ordinary mailbox drain.
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
    supervisor's optional task lifetime stays an external rail, not duplicated here.
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
    capacity = capacity_refusal(getattr(getattr(tools, "_ctx", None), "_accumulated_usage", None))
    unsent = capacity or episode.engine_unanswered  # the latest request was not sent: no slot, or no own daemon
    waiting_for = ("a free local model slot" if capacity else "the local Claudexor engine" if episode.engine_unanswered
                   else "the primary provider" if episode.wait_cause in PRIMARY_REFUSAL_KINDS else "a provider connection")

    def _ended(detail: str) -> bool:
        emit_network_wait_event(
            drive_logs, task_id=task_id, phase="ended", elapsed_sec=elapsed,
            redials=episode.redials, model=model, detail=detail,
        )
        if episode.interactive:
            emit_progress(f"🌐 Stopped waiting for {waiting_for} after {elapsed / 60.0:.1f} min — " + (
                "its last request was not sent, and this turn ends without an answer." if unsent
                else "this turn ends as an outage."), incident=None)
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
        emit_progress(f"🌐 Still waiting for {waiting_for} — {elapsed / 60.0:.0f} min elapsed, {episode.redials} " + (
            "retries; the latest request was not sent" if unsent else "redials")
            + ("; the work keeps its model" if episode.engine_unanswered else "") + "; will resume automatically.",
            incident=None)  # a periodic note is never a toast; the episode always passes incident=
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
    end_episode_budget(episode, drive_logs, task_id, model, detail="finalize_now")
    return handle_provider_unavailable(
        error_kind=episode.wait_cause,
        wait_cause=episode.wait_cause,
        waited_sec=episode.waited_sec,
        interactive=episode.interactive,
        control_reason=control_reason,
    )


def end_episode_budget(
    episode: TransportWaitEpisode, drive_logs: pathlib.Path, task_id: str, model: str,
    *, detail: str = "budget_exhausted",
) -> None:
    """Close an active episode when an admission rail fires mid-wait.

    A free redial spends $0 itself, but a concurrent consumer (a child, another
    root) can exhaust the shared budget between redials, or the billing authority
    can stop reading; that rail then owns the exit and the episode must not be
    left without its durable ``ended`` row, which names the rail.
    """
    emit_network_wait_event(
        drive_logs, task_id=task_id, phase="ended",
        elapsed_sec=time.monotonic() - episode.started_monotonic,
        redials=episode.redials, model=model, detail=detail,
    )


def task_deadline_epoch(tools: Any) -> Optional[float]:
    """Return the task deadline for retry backoff."""
    meta = getattr(tools._ctx, "task_metadata", {})
    if not isinstance(meta, dict):
        return None
    deadline = parse_deadline_ts(meta.get("deadline_at"))
    return deadline.timestamp() if deadline is not None else None


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


def provider_terminal_fallback_text(accumulated_usage: Dict[str, Any], *, round_idx: Any = None, **facts: Any) -> str:
    """The terminal text, then every failure of the ending round ``round_idx`` whose calls met a provider content refusal,
    each by its own kind (``round_failure_words``): such a refusal is never called an outage, an outage keeps its words."""
    from ouroboros.primary_route_observation import round_failure_words

    return (_provider_terminal_text(accumulated_usage, round_idx=round_idx, **facts)
            + round_failure_words(accumulated_usage, round_idx))


def _provider_terminal_text(
    accumulated_usage: Dict[str, Any],
    *,
    is_context_overflow: bool,
    is_transport_wait: bool,
    waited_sec: float,
    interactive: bool = False,
    is_deadline_exhausted: bool,
    control_reason: str = "", wait_cause: str = "", round_idx: Any = None,
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
    (STATUS_INTERRUPTED means pre-requeue, not terminal). ``round_idx``: the ending
    round, whose own latest size refusal its size facts name (``context_view_facts``).
    """
    from ouroboros.context_view_facts import refusal_frame_facts, request_window_fact
    from ouroboros.outcomes import REASON_OWNER_REQUESTED_FINALIZATION
    from supervisor.owner_stop import REASON_OWNER_STOPPED_DIRECT_TURN

    unknown = (isinstance(accumulated_usage.get(TRANSPORT_DEATHS_KEY), dict)
               or bool(accumulated_usage.get("_pending_transport_outcome"))
               or accumulated_usage.get("_last_llm_error_kind") == "provider_outcome_unknown")
    if control_reason in {REASON_OWNER_REQUESTED_FINALIZATION, REASON_OWNER_STOPPED_DIRECT_TURN}:
        action = "Stop" if control_reason == REASON_OWNER_STOPPED_DIRECT_TURN else "Wrap up"
        waited = f" The wait ended after {waited_sec / 60.0:.1f} min;" if waited_sec else ""
        subject = ("the primary provider was refusing requests" if wait_cause in PRIMARY_REFUSAL_KINDS
                   else "the local Claudexor engine was not answering" if engine_unanswered(accumulated_usage)
                   else "requests were not reaching the provider" if wait_cause == "transport_unavailable"
                   else "the provider had not returned a usable answer")
        return (f"⚠️ The owner requested {action} while {subject}."
                f"{waited} No new summary request was sent. Any files written so far are preserved."
                + (provider_recovery_hint(accumulated_usage) if unknown else ""))
    if wait_cause in PRIMARY_REFUSAL_KINDS and not unknown:
        return ("⚠️ The primary provider kept refusing requests; its selected wait ended "
                f"after {waited_sec / 60.0:.1f} min. No new summary request was sent. "
                "Completed work and the original attempt-cost records are preserved.")
    if is_context_overflow:
        facts = refusal_frame_facts(accumulated_usage, round_idx)
        return (
            "⚠️ Context-size recovery ended without a fitting request; no further provider call was made. "
            "The recovery passes, the configured fallback and a renewed view could not make it fit"
            f"{f' ({facts})' if facts else ''}. The exact transcript and any files written so far are preserved."
            + request_window_fact(accumulated_usage, overflow=True, round_idx=round_idx) + _provider_quote(accumulated_usage)
        )
    if is_transport_wait:
        if unknown:
            # A wait duration does not prove a redial, nor a numeric price bound.
            return (
                "⚠️ The dispatched attempt has no confirmed provider outcome."
                f"{_unknown_wait_note(True, waited_sec, interactive)}"
                " Inspect the preserved facts before starting another run."
                f"{provider_recovery_hint(accumulated_usage)}{UNKNOWN_ATTEMPT_COST_NOTE}"
                + request_window_fact(accumulated_usage, round_idx=round_idx, earlier_only=True)
            )
        # A released attempt proves only that the request never reached the provider: a broken
        # connect, an incomplete upload or an engine refusal before dispatch, never a provider reply.
        code = str(accumulated_usage.get("_last_llm_provider_code") or "")
        subject = "this turn" if interactive else "the task"
        kept = "" if interactive else " Any files written so far are preserved in the workspace."
        if capacity_refusal(accumulated_usage):  # the install's own limit stayed full, not a provider outage
            waited = f" after {waited_sec / 60.0:.1f} min of waiting" if waited_sec > 0 else ""
            text = ("⚠️ Local model capacity stayed unavailable: no slot under Requests at once per model freed for "
                    f"the last request, so it was not sent, and {subject} ended{waited} without completing.")
        elif engine_unanswered(accumulated_usage):  # the own daemon stayed silent before create: the model did not fail
            # The last request's facts only: an earlier round may have walked, or adopted another route, before.
            waited = f" after {waited_sec / 60.0:.1f} min of waiting" if waited_sec > 0 else ""
            text = ("⚠️ The local Claudexor engine did not answer: the last request was not sent ($0 for it), and "
                    f"{subject} ended{waited} without completing.")
        elif waited_sec > 0:
            text = (f"⚠️ The request did not reach the provider; {subject} waited and redialed for {waited_sec / 60.0:.1f} "
                    f"min{'' if interactive else ' until its own limits ran out'} and ended as an outage, not completed.")
        else:
            text = ("⚠️ The request did not reach the provider, and " + ("no wait window was left" if interactive else
                    "the owner deadline left no time to wait") + f"; {subject} ended as an outage, not completed.")
        return (text + kept + (f" Last reported code: {code}." if code else "")
                + request_window_fact(accumulated_usage, round_idx=round_idx, earlier_only=True))
    if is_deadline_exhausted:
        text = "⚠️ The owner deadline ended primary model work; any files written so far are preserved."
        if unknown:
            text += provider_recovery_hint(accumulated_usage) + UNKNOWN_ATTEMPT_COST_NOTE
        return text + request_window_fact(accumulated_usage, round_idx=round_idx, earlier_only=True)
    released = accumulated_usage.get("_last_llm_attempt_custody") == "released" and not unknown
    return (
        ("⚠️ The model request did not start generation." if released else "⚠️ The model provider returned no usable response.")
        + f"{_unknown_wait_note(unknown, waited_sec, interactive)}"
        f"{provider_failure_hint(accumulated_usage)}"
        f"{provider_recovery_hint(accumulated_usage)}"
        f"{UNKNOWN_ATTEMPT_COST_NOTE if unknown else ''} "
        "Any files written so far are preserved in the workspace."
        + request_window_fact(accumulated_usage, round_idx=round_idx)
    )


# Host vocabulary: terminal rows without provider words, and fallback's previous route.
# Typed kinds stay in the record/llm_api_error; owner text never uses exception repr.
_FAILURE_KIND_WORDS = {
    "auth_error": "an authentication or authorization refusal",
    "quota_exhausted": "a quota or billing refusal",
    "subscription_window_exhausted": "a spent subscription window",
    "rate_limit": "a rate limit",
    "bad_request": "a rejected request",
    "request_too_large": "a request the provider found too large",
    "context_overflow": "a context overflow",
    "transport_unavailable": "a request that never reached the provider",
    "provider_policy_refusal": "a refusal before any provider was reached (a connection that is unavailable, or a policy)",
    "provider_outcome_unknown": "an attempt with no known outcome",
    "provider_transient": "a temporary provider failure",
    "provider_incomplete_response": "an incomplete response",
    "llm_empty_response": "an empty response",
    "llm_output_exhausted": "a reply that reached its length limit before any visible output",
    "content_filter": "the provider's content filter stopping this request (no content returned)",
    "provider_body_error": "an error in the provider's response",
    "provider_error": "a provider error",
    "model_substituted": "an answer from a model other than the requested one",
    "deadline_exhausted": "the owner deadline running out before dispatch",
}
# Logs may retain only a bounded stream-error excerpt, not the whole provider sentence.
_DIAGNOSTIC_POINTER = "The task's Logs keep the host's record of this failure."
# Quotes past the SSOT display bound disclose shortening beside the Logs pointer.
_OWNER_QUOTE_LIMIT = 600


def owner_provider_answer(failure: Any, classification: Any = None) -> Dict[str, Any]:
    """The provider's answer as the owner line and the continuing mind read it (``ProviderAnswer.facts``): the
    sentence that carried the classifier's recognition, else the innermost party's own on the current path (an
    upstream's words, not its gateway's wrapper), with its speaker.

    An HTTP 200 body error brings the record its transport seam read from the whole object (``answer``). Exception
    repr belongs to ``llm_api_error`` and ``_last_llm_error`` diagnostics, not owner text. Claudexor's composed
    display (stage/cause/engine/capability facts) and private body are excluded; only its
    ``owner_provider_fields`` speak. ``_provider_quote`` bounds the owner's quote.
    """
    from ouroboros._usage_response import provider_answer

    if isinstance(failure, dict) and isinstance(failure.get("answer"), dict):
        return failure["answer"]
    return provider_answer(failure).facts(
        getattr(classification, "said", ("", "")), basis=getattr(classification, "basis", ""),
        untyped=getattr(classification, "untyped", False))


def owner_provider_message(failure: Any) -> str:
    """The quoted provider sentence of a failure, or "" (``owner_provider_answer``)."""
    return owner_provider_answer(failure).get("sentence", "")


def owner_provider_fields(failure: Any) -> Dict[str, str]:
    """Typed Claudexor provider code/parameter, or {} (also for an unknown outcome)."""
    from ouroboros.llm_claudexor import ClaudexorModelError
    from ouroboros.utils import sanitize_tool_result_for_log

    if not isinstance(failure, ClaudexorModelError) or failure.code == "model_outcome_unknown":
        return {}
    context = failure.problem.get("context")
    context = context if isinstance(context, dict) else {}
    return {label: " ".join(sanitize_tool_result_for_log(value).split())
            for key, label in (("vendorCode", "code"), ("parameter", "parameter"))
            if isinstance(value := context.get(key), str) and value.strip()}


def stamp_owner_provider_message(accumulated_usage: Dict[str, Any], failure: Any, classification: Any = None) -> None:
    """Replace/clear this call's provider answer: its quoted sentence, the record that sentence came from
    (speaker, recognition basis, each party's words), typed fields and producer-cut marker.

    ``loop_llm_call`` stamps each failure so older words cannot speak for it. Every later reader (the owner line,
    the window record, the sizing of a pass, the continuing request's facts) reads this one stamp.
    """
    answer = owner_provider_answer(failure, classification)
    cut = isinstance(failure, dict) and failure.get("message_truncated") is True and "answer" not in failure
    for key, value in (("_last_llm_provider_message", answer.get("sentence")), ("_last_llm_provider_answer", answer),
                       ("_last_llm_provider_message_cut", cut),
                       ("_last_llm_provider_fields", owner_provider_fields(failure))):
        accumulated_usage.pop(key, None)
        if value:
            accumulated_usage[key] = value


def _failure_words(kind: str) -> str:
    """The table's words for a failure class; a class it does not list is only a provider failure."""
    return _FAILURE_KIND_WORDS.get(kind, "a provider failure")


def _provider_quote(accumulated_usage: Dict[str, Any]) -> str:
    """The stamped sentence with its speaker (the upstream a gateway named, else "The provider"), a long one
    shortened in words beside the Logs pointer, or ""."""
    from ouroboros.utils import sanitize_tool_result_for_log, truncate_review_artifact

    message = " ".join(sanitize_tool_result_for_log(
        str(accumulated_usage.get("_last_llm_provider_message") or "")).split()).strip()
    cut = accumulated_usage.get("_last_llm_provider_message_cut") is True
    speaker = (accumulated_usage.get("_last_llm_provider_answer") or {}).get("speaker") or "The provider"
    if message and not cut and truncate_review_artifact(message, limit=_OWNER_QUOTE_LIMIT) == message:
        return f" {speaker} said: \"{message}\""
    if message:  # its producer or the SSOT cut it (the SSOT's anti-waste floor lets a small overflow through whole)
        return f" {speaker} said: \"{message[:_OWNER_QUOTE_LIMIT].rstrip()}…\" (shortened). {_DIAGNOSTIC_POINTER}"
    return ""


def provider_failure_hint(accumulated_usage: Dict[str, Any]) -> str:
    """Quote only the stamped provider sentence; disclose shortening with a Logs pointer.

    Without a sentence, attribute typed provider fields beside the host's class and
    Logs pointer. Exception repr and composed engine text stay in Logs/task details.
    """
    if quote := _provider_quote(accumulated_usage):
        return quote
    kind = str(accumulated_usage.get("_last_llm_error_kind") or "").strip()
    if not kind and not str(accumulated_usage.get("_last_llm_error") or "").strip():
        return ""
    words = _failure_words(kind)
    fields = accumulated_usage.get("_last_llm_provider_fields")
    fields = fields if isinstance(fields, dict) else {}
    named = [f'{label} "{fields[key]}"' for key, label in (("code", "error code"), ("parameter", "parameter"))
             if isinstance(fields.get(key), str) and fields[key].strip()]
    if named:  # the provider's own typed fields; its body, if any, stays private
        return (f" The provider reported {' and '.join(named)} but no sentence the host can quote; "
                f"the host classified the failure as {words}. {_DIAGNOSTIC_POINTER}")
    return f" No readable provider message is available; the host classified the failure as {words}. {_DIAGNOSTIC_POINTER}"


def _failure_is_own(accumulated_usage: Dict[str, Any], model: str, fresh: bool) -> bool:
    """Whether the last failure row is ``model``'s own call's (``fresh``: that call replaced the row it found)."""
    return fresh and (accumulated_usage.get("_last_llm_failure") or {}).get("model") == model


def fallback_reason_text(accumulated_usage: Dict[str, Any], model: str, use_local: bool = False, *,
                         named: bool = True, entry: bool = False, fresh: bool = True) -> str:
    """Explain ``model``'s last failure and surviving facts, then quote its provider sentence.

    No duplicate class/typed fields; only a shortened quote points at Logs. A host fence can keep an older kind or leave
    an earlier route's sentence: only this call's own failure row makes them this route's (``entry``: the kind the walk
    entered on is the acting route's own even without a row, its sentence still not). ``fresh=False``: the call left
    the row it found (a gate that sent nothing), which stays the earlier attempt's under the same model name too."""
    kind = str(accumulated_usage.get("_last_llm_error_kind") or "").strip()
    own = _failure_is_own(accumulated_usage, model, fresh)
    route = f" on {model}{' (local)' if use_local else ''}" if named else ""
    after = {"auth_error": "; sign-in is still needed there",
             "provider_outcome_unknown": "; the earlier attempt’s outcome and cost stay unknown"}.get(kind, "")
    cause = (f" after {_failure_words(kind)}{route}{after}" if kind and (own or entry)
             else f" after no usable answer{route}" if named else "; no cause of its own is known here")
    return f"{cause}.{_provider_quote(accumulated_usage) if own else ''}"


def fallback_last_attempt_words(accumulated_usage: Dict[str, Any], model: str, *, fresh: bool = True) -> str:
    """The walk's end row after its last attempt's route: that attempt's own result alone. An outcome the host does not
    know stays that attempt's own, true whether its operation is still readable (fenced) or not: no ending and no
    earlier attempt (``fallback_reason_text`` keeps ``the earlier attempt`` for the next attempt's row)."""
    if _failure_is_own(accumulated_usage, model, fresh) and str(
            accumulated_usage.get("_last_llm_error_kind") or "").strip() == "provider_outcome_unknown":
        return f", has no known outcome; its outcome and cost stay unknown.{_provider_quote(accumulated_usage)}"
    return f", ended without a usable reply{fallback_reason_text(accumulated_usage, model, named=False, fresh=fresh)}"


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
        # The oldest checkpoint counted permission before a repeat could be dispatched.
        # Only later checkpoints recorded the failure on that permission itself.
        recorded = deaths.get("error_kind")
        detail = f" The checkpoint records {recorded}." if recorded else ""
        return (" An earlier request has no confirmed provider outcome. The saved round "
                f"permits no further retry or paid fallback.{detail}")
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
        return (" The request did not reach the provider (typed pre-dispatch failure, $0 spent); "
                "the exact cause is in the durable llm_api_error event.")
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
    from ouroboros.llm_substitution import capability_refused

    if kind == "bad_request" and capability_refused(accumulated_usage):
        return (
            " The accounts tried cannot serve this model, and this round asks no other account. "
            "A later request may reach another one; when none can, the owner chat names them "
            "and Accounts shows any temporary marks."
        )
    if kind in {"quota_exhausted", "auth_error", "request_too_large", "bad_request", "context_overflow"}:
        guidance = {
            "quota_exhausted": "The provider rejected the request for quota/billing reasons; retrying the same request will not help until the key/account limit changes.",
            "auth_error": "Authentication or authorization was refused; retrying the same request will not help until the configured key, account or provider access is fixed.",
            "request_too_large": "The provider rejected the request size/output-token shape; retrying the same request will not help without reducing context/output demand or changing model capacity.",
            "bad_request": "The request was rejected as invalid for this route; retrying it unchanged will not help until the transcript/tool payload or the route configuration is fixed.",
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
    return ""  # no claim of an automatic retry: nothing resumes a failed task by itself
