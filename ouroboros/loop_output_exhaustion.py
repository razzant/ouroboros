"""Recovery of settled replies that did not end as a final, carried by the existing checkpoint.

An empty length reply: an informed chance, one source-address pass, then the configured route walk.
A reply with text and no typed call whose provider finish names its output limit or a tool call
(``unfinished_reply``): its text is kept as written and the informed chance is all; another one
ends the work as incomplete, saved as that end for any recovery. Only that typed finish is read,
never the reply's words.
A usable reply or new owner words ends an episode; rounds and Restart do not.
A completed commentary reply (``reply_usability`` ``intermediate``) is no cut and no episode: kept as
written with its custody, it continues as an ordinary paid round under the loop's own budget, Stop and
deadline (``intermediate_reply``). Neither it nor a cut latches a FINAL ANSWER marker (``latch_final_marker``).
Each attempt of a step is saved pending under its own id before dispatch; only a known end of the
current attempt lets its owner's replacement (wire, image, context) follow, saved pending again.
A step that answered writes no newer state: its claim, and the ready sequence a native operation's
address names, stay the saved state until the loop incorporates the reply, so work lost in between
reads that reply (``reconcile_read``) or ends unconfirmed, and nothing is generated twice. A read reply finishes
only the owner input its send provably carried: what a drain delivered since keeps it as words (``keep_unseen_reply``).
A Wrap up withdrawn once a Main call's own receipt settled, a received failure included, reads that attempt the same
way (``withdrawn_read``); met by what a restored read received, it has that read read once more, applied once
(``ordinary_read``); on the forced rail it ends on what the settled request received (``withdrawn_forced_read``). None
is ever sent again; a claim whose attempt a read proved ended is met by its own step, never read as interrupted.
A saved pending claim is uncertain, never permission to repeat: a call that returned with no known
end stays claimed, and only the execution that heard it return keeps the round's unknown-outcome
rails. No response here teaches a context-window limit.
"""
from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any

from ouroboros._usage_response import OUTPUT_LIMIT_FINISH_REASONS, TOOL_CALL_FINISH_REASONS, output_exhaustion_facts
from ouroboros.transport_custody import attempt_end_known

KEY = "_output_exhaustion_recovery"
UNFINISHED_KEY = "_unfinished_reply_recovery"
RECORD_KEYS = (KEY, UNFINISHED_KEY)  # the episodes a new execution's recovery carries (``working_checkpoint``)
KIND = "llm_output_exhausted"
REASON = "output_recovery_exhausted"


def record_empty_cut(total: dict, reported: dict, *, sent_allowance=None, custody="", message=None) -> None:
    """The same failure facts for a live or restored empty reply already known to have reached its output limit."""
    error = message or "LLM reply ended on its output limit before any visible text or tool call"
    total.update(_last_llm_error_kind=KIND, _last_llm_error=error, _last_llm_attempt_custody=custody,
                 execution_status="failed", reason_code="llm_empty_response",
                 _last_llm_output_exhausted=output_exhaustion_facts(reported, sent_allowance))
    failure = total["_last_llm_failure"] = total.get("_last_llm_call_meta", {})
    failure.update(failure_code=KIND, error=error)
    if custody:
        failure["attempt_custody_state"] = custody


def _loop():
    from ouroboros import loop
    return loop


def _save(ctx: Any) -> None:
    from ouroboros.working_checkpoint import save_model_call
    save_model_call(model_call=ctx)


def _owner_revision(ctx: Any) -> int:
    from ouroboros.loop_messages import owner_input_revision
    return owner_input_revision(ctx.tools._ctx)


def binding(ctx: Any) -> list:
    from ouroboros.model_slots import route_binding
    from ouroboros.model_wait import current_model_wait
    waiter = current_model_wait()
    role = ctx.model_role or getattr(ctx.context_fit_plan, "model_role", "") or "main"
    return list(route_binding(ctx.active_model, ctx.active_use_local, role,
                              overrides=waiter.overrides if waiter else None))


def response_facts(tool_ctx: Any) -> dict:
    """A fresh projection for one ordinary dispatch to fill (``loop_llm_call._replace_response_meta``)."""
    tool_ctx._round_response_facts = {}
    return tool_ctx._round_response_facts


def unfinished_finish(tool_ctx: Any, msg: dict) -> str:
    """That projection's finish when a usable reply with text and no typed call ended at its output
    limit or for a tool call; '' otherwise (stop, end_turn, absent, any typed call)."""
    facts = getattr(tool_ctx, "_round_response_facts", None) or {}
    finish = str(facts.get("finish_reason") or "") if facts.get("finish_reason_present") is True else ""
    typed = finish.strip().lower() in OUTPUT_LIMIT_FINISH_REASONS | TOOL_CALL_FINISH_REASONS
    return finish if typed and not msg.get("tool_calls") and str(msg.get("content") or "").strip() else ""


def _answered(ctx: Any, msg: dict) -> None:
    """A usable reply ends the episodes; an unfinished one keeps its own for ``unfinished_reply``."""
    ctx.accumulated_usage.pop(KEY, None)
    if not unfinished_finish(ctx.tools._ctx, msg):
        ctx.accumulated_usage.pop(UNFINISHED_KEY, None)


def _unfinished_cause(finish: str) -> str:
    return (f"at its output limit (finish_reason={finish})" if finish.strip().lower() in OUTPUT_LIMIT_FINISH_REASONS
            else f"with a tool-call finish (finish_reason={finish}) but no typed tool call")


def unfinished_reply(ctx: Any, msg: dict) -> Any:
    """The ordinary no-call consumer: ``False`` unless the reply is unfinished. Its text, reasoning and custody
    then stay in the transcript as written; the episode's first buys one informed ordinary turn (``None``), and
    another ends the work as incomplete with that text kept (``finish``), never as a completed answer."""
    from ouroboros.anthropic_native_custody import custody_private_key
    from ouroboros.working_checkpoint import save_or_log
    cut = unfinished_finish(ctx.tools._ctx, msg)
    if not cut:
        return False
    usage, revision = ctx.accumulated_usage, _owner_revision(ctx)
    record = usage.get(UNFINISHED_KEY)
    ctx.messages.append({"role": "assistant", "content": msg["content"], **{
        key: value for key, value in msg.items() if key in ("reasoning_details", "reasoning_content")
        or custody_private_key(key)}})
    if record and record["owner_revision"] >= revision:  # its informed turn ended the same way
        record["finishes"].append(cut)
        record["stage"] = "exhausted"
        save_or_log(ctx, "ready")  # the latest words and finishes outlive a crash before the result write
        return finish(ctx, key=UNFINISHED_KEY)
    usage[UNFINISHED_KEY] = {"stage": "informed", "finishes": [cut], "owner_revision": revision}
    _loop()._append_or_merge_user_message(ctx.messages, (
        f"[SYSTEM NOTICE]\nThe provider ended your previous reply {_unfinished_cause(cut)}; it is kept above as "
        "written and was not delivered as a final answer. Decide how to continue: complete it, use a tool, or give "
        "an honest final report. Another reply ended this way ends the work as incomplete."))
    # Words, fact and allowance are one saved transition. This runs on the loop's own context, past the model-call
    # scope ``_save`` needs; a ready state repeats its round, and a failed write is retried before the claimed send.
    save_or_log(ctx, "ready")
    return None


def _intermediate(tool_ctx: Any, msg: dict) -> bool:
    """That projection's reply is healthy intermediate progress (``reply_usability`` ``intermediate``)."""
    from ouroboros._usage_response import reply_usability
    facts = getattr(tool_ctx, "_round_response_facts", None) or {}
    finish = facts.get("finish_reason") if facts.get("finish_reason_present") is True else None
    return reply_usability(finish, facts.get("response_status"), msg) == "intermediate"


def latch_final_marker(ctx: Any, msg: dict) -> None:
    """Latch this round's FINAL ANSWER marker (``loop_messages._latch_final_answer_marker``) from a tool round or a
    reply the loop could deliver as final; a commentary or a cut latches (and withdraws) nothing, since a deadline,
    budget end or Stop would publish the latched words as the answer."""
    if not _intermediate(ctx.tools._ctx, msg) and not unfinished_finish(ctx.tools._ctx, msg):
        _loop()._latch_final_answer_marker(ctx.llm_trace, msg.get("content"))


def intermediate_reply(ctx: Any, msg: dict, emit_progress: Any) -> bool:
    """Keep a healthy intermediate reply (completed commentary, no call) as written with its reasoning and custody,
    show it as the round's update, save it, and let the caller continue the ordinary loop; ``False`` otherwise."""
    from ouroboros.anthropic_native_custody import custody_private_key
    from ouroboros.loop_messages import _emit_round_progress
    from ouroboros.working_checkpoint import save_or_log
    if not _intermediate(ctx.tools._ctx, msg):
        return False
    ctx.messages.append({"role": "assistant", "content": msg["content"], **{
        key: value for key, value in msg.items() if key in ("reasoning_details", "reasoning_content")
        or custody_private_key(key)}})
    _emit_round_progress(msg["content"], msg, emit_progress, ctx.llm_trace)  # an interim update, like a tool round's
    save_or_log(ctx, "ready")  # the kept words outlive a crash; the next round sends them
    return True


def ready_for_fallback(usage: dict) -> bool:
    return (usage.get(KEY) or {}).get("stage") == "fallback"


def _pending(usage: dict) -> str:
    """The episode whose claimed send is unconfirmed; one claim is held at a time."""
    return next((key for key in RECORD_KEYS if (usage.get(key) or {}).get("pending")), "")


def _restored_end(ctx: Any) -> dict:
    """This round's restored known end (``llm_claudexor.restored_end``), still unmet: the read proved its claimed attempt over,
    unknown no longer. A step that claimed it (``_StepAttempts``) is not interrupted; its own call meets that end as the live
    step met its attempt's, under the wait, retry and route policy that step had."""
    from ouroboros.llm_claudexor import restored_round
    usage = ctx.accumulated_usage
    return restored_round(usage, f"{usage.get('execution_id')}:round:{ctx.round_idx}").get("end") or {}


def _cut_ended(usage: dict) -> bool:
    """A typed-cut episode saved as ended: what is left is its terminal, never another send."""
    return (usage.get(UNFINISHED_KEY) or {}).get("stage") == "exhausted"


def no_recap(usage: dict) -> bool:
    spent = (usage.get(KEY) or {}).get("stage") in {"fallback", "exhausted"}
    return bool(_pending(usage) or _cut_ended(usage) or spent)


def next_step(usage: dict) -> str:
    """Only our own failure ends this episode; other failure/control rails win."""
    if usage.get("_last_llm_error_kind") == "output_recovery_interrupted":
        return "finish"  # its own gate refused the send, whatever the restored usage still says of that call
    if _loop().provider_no_call_source(usage, False)[0] or usage.get("_last_llm_error_kind") != KIND:
        return ""
    return "finish" if no_recap(usage) else "inform"


def fallback_untried(usage: dict, target: Any) -> bool:
    """Eligibility only: the physical claim below consumes the binding."""
    record = usage.get(KEY)
    return not record or list(target) not in record["tried"]


class _StepAttempts:
    """One step's physical attempts (``transport_custody.track_physical_attempts``), called before each dispatch.

    Every attempt is saved pending under its own id before dispatch. A further attempt only replaces the current
    one once that attempt's end was heard as known; whether a refusal earns a replacement stays its owner's rule."""

    def __init__(self, ctx: Any, record: dict, step: str, target: list) -> None:
        self.ctx, self.record, self.step, self.target = ctx, record, step, target
        self.attempt, self.ended, self.sent = "", "", False  # the current attempt, its known end, an earlier send

    def __call__(self, reservation: Any) -> None:
        from ouroboros.usage_accounting import PhysicalAttemptPreconditionFailed

        if self.attempt:
            if not self.ended:  # no known end: its answer could still come, so nothing may overlap it
                raise PhysicalAttemptPreconditionFailed("output recovery send already claimed")
            self.sent |= self.ended != "released"
        elif self.step == "fallback":
            self.record["tried"].append(self.target)
        self.attempt, self.ended = reservation.attempt_id, ""
        self.record.update(pending=self.step, attempt_id=reservation.attempt_id)
        _save(self.ctx)  # every send of this step is saved pending before dispatch, replacements included

    def attempt_ended(self, capture: Any, error: Any) -> None:
        if capture.attempt_id == self.attempt and attempt_end_known(capture, error):
            self.ended = capture.state


def _invoke(ctx: Any, send: Any, step: str, key: str = KEY) -> Any:
    """Track this step's physical attempts, so an unstarted Pause/budget wait spends nothing."""
    usage, record = ctx.accumulated_usage, ctx.accumulated_usage[key]
    prior_stage, target = record["stage"], binding(ctx)
    attempts = _StepAttempts(ctx, record, step, target)
    end = _restored_end(ctx)  # the read proved this step's claimed attempt ended: its call meets that end as the live one did
    if record.get("pending") == step != "fallback" and (claim := record.get("attempt_id")) and end.get("physical_attempt_id") == claim:
        attempts.attempt, attempts.ended = claim, end["custody"]

    def refund_unsent():
        if attempts.attempt and attempts.ended == "released":
            record.pop("pending", None)
            if not attempts.sent:  # nothing of this step was sent: it stays owed
                attempts.attempt = ""
                record["stage"] = prior_stage
                if step == "fallback":
                    record["tried"].remove(target)
            _save(ctx)
    prior_cap, ctx.attempt_cap = ctx.attempt_cap, 1
    prior_claim, ctx.physical_send_claim = getattr(ctx, "physical_send_claim", None), attempts
    try:
        result = send()  # preparation helpers run before the Main-only claim scope
    except BaseException:
        refund_unsent()
        raise
    finally:
        ctx.attempt_cap = prior_cap
        ctx.physical_send_claim = prior_claim
    refund_unsent()
    if result[0] is not None or attempts.ended:
        record.pop("pending", None)
    elif attempts.attempt:  # no reply, no known end: the saved claim stands; only this execution heard the call return
        ctx.tools._ctx._output_recovery_returned_unknown = attempts.attempt
    if attempts.attempt and key == KEY:
        _record_step_outcome(usage, record, step)
    return result


def _record_step_outcome(usage: dict, record: dict, step: str) -> None:
    record["last_failure"] = dict(usage.get("_last_llm_failure") or {})
    record["last_failure_kind"] = usage.get("_last_llm_error_kind", "")
    if step != "informed" or record["last_failure_kind"] != KIND:
        record["stage"] = "fallback"


def call(ctx: Any, ordinary_call: Any) -> tuple:
    """Wrap the ordinary call; preserve transport, unknown and control ownership."""
    usage = ctx.accumulated_usage
    for key in RECORD_KEYS:  # new owner words end an episode; never a claimed send's unconfirmed outcome
        if (stale := usage.get(key)) and not stale.get("pending") and stale["owner_revision"] < _owner_revision(ctx):
            usage.pop(key)
    returned_here = getattr(ctx.tools._ctx, "_output_recovery_returned_unknown", "")  # each start builds a new context
    known = _restored_end(ctx).get("physical_attempt_id")  # a read proved that attempt ended: its step meets that end below
    interrupted = (pending := _pending(usage)) and (claim := usage[pending].get("attempt_id")) != returned_here and not (
        claim and claim == known)  # a claim that names no attempt is never proved ended
    if interrupted:
        usage[pending].update(stage="exhausted", interrupted=True)
    if interrupted or _cut_ended(usage):
        usage["_last_llm_error_kind"] = "output_recovery_interrupted"
        return None, None, ctx.active_context_mode
    record = usage.get(KEY)
    fallback = bool(getattr(ctx, "output_recovery_fallback", False))
    if record and record["stage"] == "smaller" and not fallback:
        return _send_smaller(ctx, None)
    if record and record["stage"] != "informed" and not fallback:
        usage["_last_llm_error_kind"] = KIND
        return None, None, ctx.active_context_mode
    claimed = KEY if record else UNFINISHED_KEY if UNFINISHED_KEY in usage else ""  # an owed informed turn
    msg, cost, mode = (_invoke(ctx, lambda: ordinary_call(ctx), "fallback" if fallback else "informed", claimed)
                       if claimed else ordinary_call(ctx))
    if msg is not None:
        _answered(ctx, msg)
        return msg, cost, mode
    if usage.get("_last_llm_error_kind") != KIND or _loop().provider_no_call_source(usage, False)[0]:
        if claimed:
            _save(ctx)
        return msg, cost, mode
    comparison = _loop().last_physical_attempt_capture() if record and not fallback else None
    return _after_empty_cut(ctx, cost, mode, allow_shrink=not fallback, comparison=comparison)


def recover_empty_cut(ctx: Any, pending: dict, cost: Any, *, status_root: Any) -> tuple:
    """Re-enter the length ladder after the read has ended; only a matched, known end closes a claim."""
    usage, physical_id = ctx.accumulated_usage, pending["physical_attempt_id"]
    key, comparison, allow_shrink = _pending(usage), None, False
    if key:
        record = usage[key]
        if record.get("attempt_id") != physical_id:
            record.update(stage="exhausted", interrupted=True)
            usage["_last_llm_error_kind"] = "output_recovery_interrupted"
            return None, cost, ctx.active_context_mode
        step = record.pop("pending")
        if key == KEY:
            _record_step_outcome(usage, record, step)
            allow_shrink = step == "informed"
    if allow_shrink:
        from ouroboros import usage_store

        with usage_store.read(status_root) as store:
            row = store.attempt(physical_id)
        if (row and row["task_id"] == ctx.task_id and row["state"] == "settled"
                and row["provider"] == "claudexor" and row["model"] == pending.get("model")
                and row.get("physical_context") and row.get("candidate_measurement_kind") == "canonical_json_v1"
                and type(row.get("max_completion_tokens")) is int):
            comparison = SimpleNamespace(**row)
    return _after_empty_cut(ctx, cost, ctx.active_context_mode, allow_shrink=allow_shrink, comparison=comparison)


def reconcile_read(ctx: Any, operation: Any, msg: dict) -> Any:
    """A restored read's usable reply: only the exact matching claimed attempt closes its pending and its
    episode, before the reply's tools run (``recover_empty_cut``'s own match for the empty one). A mismatch
    leaves the claim untouched and returns None, so that reply's tools are never dispatched."""
    usage = ctx.accumulated_usage
    key = _pending(usage)
    if not key:
        return msg
    record = usage[key]
    if record.get("attempt_id") != (operation or {}).get("physical_attempt_id"):
        record.update(stage="exhausted", interrupted=True)
        usage["_last_llm_error_kind"] = "output_recovery_interrupted"
        return None
    record.pop("pending", None)
    _answered(ctx, msg)
    return msg


ARRIVAL_KEYS = ("reply_start", "owner_input", "owner_carried_addressed")  # what ``note_arrivals`` binds to a read, and only that


def kept_arrivals(saved: dict, read: dict) -> dict:
    """The arrival facts ``saved`` holds for ``read``'s own attempt and operation, never another read's: what a park or a
    restore carries to the read it binds again (``loop_transport._ready_native_operation``)."""
    same = all(saved.get(identity) == read.get(identity) for identity in ("physical_attempt_id", "operation_id"))
    return {key: saved[key] for key in ARRIVAL_KEYS if key in saved} if same else {}


def paused_read(usage: dict, pending: dict) -> None:
    """Keep the read a Pause met owed as it stands (``paused_receipt_read``), with the arrivals bound to it, before the pause
    is entered: Resume rejoins this exact receipt (``ready_operation``); nothing is cancelled or sent again."""
    bound = usage.setdefault("_pending_transport_outcome", dict(pending))
    bound.update(paused_receipt_read=True, **kept_arrivals(pending, bound))  # the facts the setdefault would drop


OWNER_PAUSE_CONTROLS = ("owner_pause", "owner_pause_authority_unreadable")  # a poll's Pause: the operation keeps running


def poll_read_control(ctx: Any, owner: Any, mailbox: Any) -> Any:
    """A restored read's poll (``read_ready_operation``, through its invocation's ``check_control``): Stop, the read's
    deadline (the calendar's, or the round's narrower grace, which that check narrows and never widens), then the owner's
    Pause fence, typed as the owner seam reads it (``_handle_model_wait_control``): installed ``owner_pause``, unreadable
    authority ``owner_pause_authority_unreadable``. A Pause cancels no operation (``OWNER_PAUSE_CONTROLS``) and its
    interruption keeps the read owed first (``paused_read``). A busy store proves neither: the poll goes on, and the
    settled receipt still meets ``wait_member_fence``."""
    import time

    from ouroboros.loop_round_limits import local_deadline_remaining
    from ouroboros.loop_transport import transport_repeat_stop_requested
    from ouroboros.owner_pause import member_fence
    from ouroboros.store import StoreBusy, acquisition_budget

    if transport_repeat_stop_requested(owner, mailbox_peek=mailbox):
        return "finalize_requested"
    if ((remaining := local_deadline_remaining(owner, ctx)) is not None and remaining <= 0
            or time.time() >= (getattr(ctx, "deadline_ts", None) or float("inf"))):
        return "deadline"
    try:
        with acquisition_budget(0):
            fence = member_fence(owner)
    except StoreBusy:
        return None
    if not fence:
        return None
    return OWNER_PAUSE_CONTROLS[1] if fence.get("reason") == OWNER_PAUSE_CONTROLS[1] else OWNER_PAUSE_CONTROLS[0]


def note_arrivals(reading: Any, *, place: int, owner_input: bool, carried: Any) -> None:
    """Bind what one drain delivered after an accepted operation's request to that read (``reading``, as passed to
    ``_drain_incoming_messages``): the place its reply takes, before every row the first such drain added, and the
    arrivals its send never saw, my human's rows that entered the owner corpus and carried rows addressed to this work,
    which enter none. A Hurry or a carried context row is neither. ``keep_unseen_reply`` reads only these facts. The read
    is its saved record itself (``ready_operation``), so a park or an acknowledged drain before the read keeps them."""
    if reading is None:
        return
    reading.setdefault("reply_start", place)
    if owner_input:
        reading["owner_input"] = True
    if carried:
        reading["owner_carried_addressed"] = [*(reading.get("owner_carried_addressed") or ()), *carried]


def _carried_since(error: Any) -> list:
    """The rows the control rails' own drain showed as carried to this work (``drained_controls``): their words leave
    the owner corpus unchanged by design, so a reply read after that drain cannot be shown to have seen them."""
    return list((getattr(error, "drained_controls", None) or {}).get("owner_carried_addressed") or ())


def withdrawn_read(ctx: Any, error: Any) -> Any:
    """The redial after a Wrap up withdrawn once this Main call's own receipt had settled (``retain_interrupted_main_reply``
    carried it, a received failure or missing message too) reads that exact attempt from its ready checkpoint as a
    restart would (``ready_operation``: its real call ids, address and owner binding), never sends it again; the read
    consumes what it received once, and carries what the control's drain showed (``keep_unseen_reply``). The owner is
    told the attempt first, so a sidecar naming another one changes nothing in this round's usage. Without this
    attempt's address the read has none: the round ends on the unknown no-resend rail, the settled receipt counted
    once apart from that outcome. A restored read the rails let go (``read_operation``) is owed again as it stands, its
    saved record restored for a park before its read. Every other interruption, a call that never settled, keeps the
    ordinary redial."""
    from ouroboros import working_checkpoint
    from ouroboros.loop_llm_call import _account_received_error
    from ouroboros.loop_transport import ready_operation
    from ouroboros.preflight_receipts import canonical_data_root

    if (read := getattr(error, "read_operation", None)) is not None and not read.get("forced_reason_code"):
        ctx.accumulated_usage.setdefault("_pending_transport_outcome", read)  # what it applied counts once on its re-read
        return read
    attempt = getattr(getattr(error, "physical_attempt_capture", None), "attempt_id", "")
    if (getattr(error, "control_reason", "") != "finalize_requested" or not attempt
            or not getattr(error, "received_main_reply", None)):
        return None
    owner, usage = ctx.tools._ctx, ctx.accumulated_usage
    try:
        _data, state = working_checkpoint.read_working(
            working_checkpoint._root(owner), ctx.task_id, int(getattr(owner, "task_attempt", None) or 1))
        bound = ready_operation(state, usage, canonical_data_root(owner) or ctx.drive_root, ctx.task_id, attempt)
    except Exception:
        bound = None
    own = bound if (bound or {}).get("physical_attempt_id") == attempt else None
    if own and own.get("operation_id"):  # bound on its saved record: where the handler's drain added words, what it carried
        note_arrivals(own, place=len(state.get("messages") or ()), owner_input=False, carried=_carried_since(error))
        return own
    _account_received_error(usage, error)  # this settled receipt counts once; nothing reads it this round
    usage["_last_llm_error_kind"] = "provider_outcome_unknown"
    return own or {"physical_attempt_id": attempt, "operation_id": "", "model_role": error.model_role,
                   "round": ctx.round_idx}


def ordinary_read(ctx: Any, pending: dict, read: Any) -> tuple:
    """An ordinary accepted operation's read (``_read_accepted_round``), as ``(msg, cost, None)``. A Wrap up can meet what
    the read received once its reader applied it, a reply, a received failure or an output cut, and be withdrawn by the time
    the control rails drain it (they return None): that drain bound its arrivals to this read (``read_operation``), and the
    read is read once more, its own Pause rail first and nothing counted or recorded twice (``_counted_error_attempts``), so
    what it received takes its ordinary policy exactly once (``keep_unseen_reply``, ``restored_end``, ``recover_empty_cut``).
    A control that stands ends here on its rail's terminal; every other interruption, a second one included, and an unsent
    round raise to the loop's own handler, which routes it once. A restored end whose receipt was judged unusable
    (``restored_end`` ``rejected``) is terminal here, before ordinary admission, a configured route or a deadline's grace
    could send again: the provider-unavailable rail on that receipt's own failure row, counted once by the read."""
    from ouroboros.llm_claudexor import RESTORED_END_KEY, RestoredEnd
    from ouroboros.model_wait import ModelWaitInterrupted

    try:
        try:
            return (*read(ctx, pending), None)
        except ModelWaitInterrupted as error:
            if error.control_reason != "finalize_requested":
                raise
            if (controlled := _loop()._handle_model_wait_control(ctx, error)) is not None:
                return None, None, _loop()._merged_terminal(controlled, ctx.llm_trace)
        return (*read(ctx, pending), None)
    except RestoredEnd:
        usage = ctx.accumulated_usage
        if not (usage.get(RESTORED_END_KEY) or {}).get("rejected"):
            raise
        failure = usage.setdefault("_last_llm_failure", {})
        failure["rejected_receipt"] = True  # no-call: ``provider_no_call_source``
        kind = usage["_last_llm_error_kind"] = str(failure.get("failure_code") or "response_rejected")
        return None, None, _loop()._merged_terminal(_loop()._handle_provider_unavailable(ctx, error_kind=kind), ctx.llm_trace)


def forced_read_end(ctx: Any, operation: dict, error: Any) -> tuple:
    """End a forced rail on the reply its own sent request received (``received_main_reply``), through forced delivery
    (``_read_accepted_round``) and never a new request. A control may have met that reply before its Pause check, so
    only an open fence, read at once, lets the reply answer; a closed or unreadable one withholds it. The control
    stays recorded beside the forced reason."""
    from ouroboros.owner_pause import fence_open_now

    reply = (getattr(error, "received_main_reply", None) or (None,))[0]
    if reply is not None and not fence_open_now(ctx.tools._ctx):
        reply = None
    result = _loop()._read_accepted_round(ctx, operation, lambda *_: (reply, None))[2]
    result[2].setdefault("forced_finalization", {})["control_reason"] = error.control_reason
    return result


def withdrawn_forced_read(ctx: Any, error: Any) -> Any:
    """The forced rail's own request after a control the control rails let yield (a Wrap up withdrawn or superseded
    when they drained it). Only an interruption that carries no positive physical attempt (none made, or released or
    left reserved before dispatch: ``POSITIVE_PHYSICAL_ATTEMPT_STATES``) may admit the same intent again (``None``).
    An outcome that may still answer keeps the unknown no-resend rail, and so does a settled attempt that is not
    provably this request's own (``FORCED_ATTEMPT_KEY``, recorded before its dispatch): a positive custody fact is
    never read as unsent. This request's own settled attempt is never sent again: that receipt counts once and the
    rail ends on what it received (``forced_read_end``), with the carried rows the control's drain showed; a received
    failure or missing message ends on the rail's no-send terminal."""
    from ouroboros.loop_llm_call import _account_received_error, _replace_response_meta
    from ouroboros.loop_transport import FORCED_ATTEMPT_KEY
    from ouroboros.transport_custody import _capture_on_chain, outcome_unknown_on_chain
    from ouroboros.usage_accounting import POSITIVE_PHYSICAL_ATTEMPT_STATES

    if getattr(error, "receiver_abandoned", False) or outcome_unknown_on_chain(error):
        return _loop()._handle_provider_unavailable(ctx, error_kind="provider_outcome_unknown")
    capture = _capture_on_chain(error)
    if getattr(capture, "state", None) not in POSITIVE_PHYSICAL_ATTEMPT_STATES:
        return None
    sent = ctx.accumulated_usage.get(FORCED_ATTEMPT_KEY) or {}
    if capture.attempt_id != sent.get("physical_attempt_id"):
        return _loop()._handle_provider_unavailable(ctx, error_kind="provider_outcome_unknown")
    _account_received_error(ctx.accumulated_usage, error)
    reply, reported, _source = getattr(error, "received_main_reply", None) or (None, None, None)
    _replace_response_meta(response_facts(ctx.tools._ctx), reported, reply)  # the facts a restored read's reader records
    return forced_read_end(ctx, {**sent, "owner_carried_addressed": _carried_since(error)}, error)


def keep_unseen_reply(ctx: Any, operation: Any, msg: dict, start: int) -> bool:
    """A read reply answers only the owner input its send carried. Every drain after that send binds to the read what it
    delivered (``note_arrivals``; a live call's control drain through ``withdrawn_read``): my human's rows that entered
    the owner corpus, or a carried row addressed to this work, which moves no corpus by design. With either, or a bound
    corpus that changed, the reply stays in the transcript at its own place (``reply_start``, before every row those
    drains added; else ``start``) as words only: it finalizes nothing and runs no tool, and the new input gets an
    ordinary round, never a repeat of that send. A legacy send bound no corpus, and that absence proves nothing either
    way: without such an arrival it delivers as before; a Hurry or a carried context row is none. The forced rail
    compares the same facts itself (``_read_accepted_round``)."""
    from ouroboros.loop_messages import owner_source_sha256

    operation = operation or {}
    bound = operation.get("owner_source_sha256")
    if not (operation.get("owner_input") or operation.get("owner_carried_addressed")
            or bound and bound != owner_source_sha256(ctx.tools._ctx)):
        return False
    if msg.get("content"):
        start = min(start, operation.get("reply_start", start), len(ctx.messages))
        ctx.messages.insert(start, {"role": "assistant", "content": msg["content"]})
    return True


def admit_read_reply(ctx: Any, operation: Any, msg: dict, start: int) -> tuple:
    """Admit a restored read's usable reply as ``(msg, parked, terminal)``. Its exact claim closes first (``reconcile_read``;
    a mismatch: ``(None, False, None)``). What arrived while the receipt was polled is drained and bound to this read now
    (``note_arrivals``), before the reply is accepted or a tool runs, so a withdrawn action never dispatches: new owner input
    parks the reply as words (``keep_unseen_reply``, ``parked``). A Stop or Wrap up that drain consumed keeps its disposition:
    the reply is parked and that control takes its round-top rail at once (``_maybe_early_finalize``, Stop now through
    ``_handle_direct_turn_hard_stop``), never lost to the next drain."""
    import queue

    msg = reconcile_read(ctx, operation, msg)
    if msg is None:
        return None, False, None
    seen = ctx.owner_msg_seen if ctx.owner_msg_seen is not None else set()
    controls = _loop()._drain_incoming_messages(ctx.messages, ctx.incoming_messages or queue.Queue(), ctx.drive_root,
        ctx.task_id, ctx.event_queue, seen, owner_ctx=ctx.tools._ctx, defer_content_ack=True, reading=operation)
    parked = keep_unseen_reply(ctx, operation, msg, start)
    if not controls.get("finalize_now"):
        return (None if parked else msg), parked, None
    if not parked and msg.get("content"):
        ctx.messages.append({"role": "assistant", "content": msg["content"]})  # words only: no tool of it runs
    return None, True, _loop()._maybe_early_finalize(ctx, ctx.tools, controls)


def _after_empty_cut(ctx: Any, cost: Any, mode: str, *, allow_shrink: bool, comparison: Any) -> tuple:
    usage = ctx.accumulated_usage
    if _loop().provider_no_call_source(usage, False)[0] or _pending(usage):
        return None, cost, mode  # another outcome still owns the rail
    record = usage.get(KEY)
    if record is None:
        usage[KEY] = {"stage": "informed", "owner_revision": _owner_revision(ctx),
                      "tried": [binding(ctx)], "first": dict(usage.get("_last_llm_output_exhausted") or {})}
        _loop()._append_or_merge_user_message(ctx.messages, _loop()._output_exhausted_notice(usage[KEY]["first"]))
        _save(ctx)  # facts and allowance are one saved transition
        return None, cost, mode
    if not allow_shrink:
        _save(ctx)
        return None, cost, mode
    return _smaller_request(ctx, cost, comparison)


def _smaller_request(ctx: Any, cost: Any, failed: Any) -> tuple:
    from ouroboros.loop_model_call import _failed_capture_is_comparable
    from ouroboros.loop_round_limits import _run_emergency_address_pass

    record = ctx.accumulated_usage[KEY]
    fit = _loop()._measure_round_main_fit(ctx, automatic_pass_used=False)
    record["stage"] = "fallback"
    if fit is None or not _failed_capture_is_comparable(failed):
        record["reclaim"] = "comparison_unavailable"
        _save(ctx)
        return None, cost, ctx.active_context_mode
    notice = ("Repeated empty length-limited replies: preparing a smaller request on the same model, "
              "if eligible sources can be addressed. Only superseded host facts and already exposed tool "
              "results may move to exact sources; owner words, required memory, books and unseen results stay. "
              "This establishes no context-window limit and may reduce cache reuse.")
    _loop()._append_or_merge_user_message(ctx.messages, "[SYSTEM NOTICE]\n" + notice)
    changed = False
    for rung in ("host_copies", "bodies"):
        receipt = _run_emergency_address_pass(ctx, fit, rung=rung, cause="output_exhaustion")
        changed |= receipt.status == "applied"
    record["reclaim"] = "applied" if changed else "no_eligible_shrink"
    if changed:
        # Exact prior physical comparison travels with an unsent prepared view across Pause.
        record["comparison"] = {name: getattr(failed, name) for name in (
            "provider", "model", "max_completion_tokens", "candidate_measurement_kind",
            "candidate_raw_sha256", "candidate_context_size_bytes")}
        record["stage"] = "smaller"
        if ctx.emit_progress:
            ctx.emit_progress(notice)
    _save(ctx)
    return _send_smaller(ctx, cost) if changed else (None, cost, ctx.active_context_mode)


def _send_smaller(ctx: Any, cost: Any) -> tuple:
    from ouroboros.loop_model_call import _strict_context_shrink_predicate
    from ouroboros.presence_continuation import covered_send  # only an answered send covers (``reply_anchor``)
    from ouroboros.usage_accounting import PhysicalAttemptPreconditionFailed

    usage, record = ctx.accumulated_usage, ctx.accumulated_usage[KEY]
    failed = SimpleNamespace(**record["comparison"], physical_context=None)
    fit = _loop()._measure_round_main_fit(ctx, automatic_pass_used=True)
    shrunk = True  # False: the changed view was not strictly smaller, so nothing was sent or refused
    try:
        msg, retry_cost = covered_send(ctx.tools._ctx, lambda: _invoke(
            ctx, lambda: _loop()._dispatch_round_model(
                ctx, fit, attempt_cap=1, candidate_predicate=_strict_context_shrink_predicate(failed, same_round=False),
                max_tokens=int(failed.max_completion_tokens or 0) or None), "smaller_request"))
    except PhysicalAttemptPreconditionFailed:
        msg, retry_cost, shrunk = None, None, False
        record.update(stage="fallback", reclaim="candidate_not_smaller")
    if msg is None and str(usage.get("_last_llm_error_kind") or "") == "context_overflow":
        from ouroboros.context_view_facts import remember_size_refusal
        from ouroboros.loop_llm_call import REFUSED_CANDIDATE_KEY
        from ouroboros.loop_model_call import _refused_candidate, _refusal_window

        # The same reading as the context-size path: a pre-send prediction leaves its candidate, never a capture.
        candidate = usage.get(REFUSED_CANDIDATE_KEY)
        refused = _refused_candidate(candidate) if candidate else _loop().last_physical_attempt_capture()
        if shrunk:  # this round's latest size refusal, whole; an unsent candidate leaves the earlier one
            remember_size_refusal(ctx, refused, measured=fit is not None)
        if refused is not None and getattr(refused, "predicted_free_tokens", None) is None:
            # The smaller send's own provider refusal is window evidence like any other; empty replies are not.
            from ouroboros.context_anchor import record_refused_input
            record_refused_input(usage, ctx.context_fit_plan, _refusal_window(ctx, fit, refused)[1],
                                 measured=fit.measurement.anchor_input_tokens is not None, candidate=refused)
    if msg is None:
        _save(ctx)
    else:  # no newer state is written here: the saved claim and its ready sequence stand until the loop incorporates the reply
        _answered(ctx, msg)
    return msg, float(cost or 0.0) + float(retry_cost or 0.0), ctx.active_context_mode


def finish(ctx: Any, *, reason: str = REASON, key: str = "") -> tuple:
    """Preserve existing model bytes and files, never generate an explanation."""
    from ouroboros.task_finalization import set_terminal_host_notice
    usage = ctx.accumulated_usage
    key = key or _pending(usage) or (UNFINISHED_KEY if _cut_ended(usage) else KEY)
    record = usage.get(key) or {}
    record["stage"] = "exhausted"
    notice = ("Work is incomplete: a recovery attempt was interrupted and its outcome is unconfirmed. "
              "It was not repeated." if record.get("interrupted") or record.get("pending") else
              f"Work is incomplete: the provider again ended the reply {_unfinished_cause(record['finishes'][-1])} "
              "after one informed turn; the replies stay as written and no extra generation summarized the failure."
              if key == UNFINISHED_KEY else
              "Work is incomplete: the model returned empty length-limited replies and the permitted "
              "recovery steps did not produce an answer. No extra generation was made to summarize the failure.")
    notice += " Saved work remains available; choose whether and how to continue."
    candidate, kept = _loop()._live_delivery_candidate(ctx), _loop()._last_assistant_text(ctx.messages)
    if candidate is not None and key == UNFINISHED_KEY and candidate.full_text == kept:  # its own end, saved composed
        ctx.tools._ctx._delivery_candidate = ctx.delivery_candidate = candidate = None
    saved = candidate.full_text if candidate is not None else kept
    # Stamped before composing: the composer may end this rail on the owner's Stop now.
    usage.update(execution_status="infra_failed" if reason == REASON else "failed", reason_code=reason)
    text, usage, trace = _loop()._forced_fallback_result(
        ctx, ctx.llm_trace, saved or notice, reason, source="output_exhaustion_no_recap", host_words=not saved)
    trace["output_recovery"] = copy.deepcopy(record)
    set_terminal_host_notice(usage, str(usage.get("terminal_host_notice") or ""), notice)
    return text, usage, trace


def retain_interrupted_main_reply(error: Any, source: tuple[str, str]) -> None:
    """Carry this Main call's own settled receipt, never a helper's or a prior send's: its reply, or None for a failure."""
    attempt_id, owner_source = source
    capture = getattr(error, "physical_attempt_capture", None)
    result, reported = getattr(error, "model_result", None), getattr(error, "usage", None)
    if (getattr(capture, "attempt_id", None) != attempt_id or capture.state != "settled" or capture.provider != "claudexor"
            or not isinstance(reported, dict)):  # a received failure or missing message carries usage, no ``model_result``
        return
    receipt = reported.get("claudexor") or {}
    if (receipt.get("operation_id") and receipt["operation_id"] == getattr(error, "operation_id", None)
            and receipt.get("model_role") == getattr(error, "model_role", None)
            and receipt.get("dispatch_state") == "response_received"):
        error.received_main_reply = ((result if isinstance(result, dict) else {}).get("message"), reported, owner_source)


def _received_deadline_answer(ctx: Any, error: Any, control: str, controls: dict) -> tuple[str, str | None]:
    """Select this interrupted Main call's own final prose, never a candidate replacement or a new action."""
    reply, reported, owner_source = getattr(error, "received_main_reply", (None, {}, None))
    missing = ("", None)
    deadlines = {"deadline", "execution_deadline", "absolute_ceiling"}
    if not isinstance(reply, dict) or (control and control not in deadlines):
        return missing
    if (error.control_reason not in deadlines
            and not (error.control_reason == "finalize_requested" and control)):
        return missing
    from ouroboros.loop_delivery import _completion_words, _parse_delivery_control_body
    from ouroboros.loop_llm_call import _replace_response_meta, forced_response_is_incomplete
    from ouroboros.loop_messages import owner_source_sha256
    from ouroboros.loop_transport import inline_presence
    from ouroboros.outcomes import extract_final_answer
    from ouroboros.owner_pause import fence_open_now

    # A restored read's reply answers only its exact claimed attempt, on this exit as on the ordinary one: a
    # mismatch keeps the claim and takes none of that reply's words.
    recovered = getattr(error, "recovered_attempt_id", None)
    if recovered and reconcile_read(ctx, {"physical_attempt_id": recovered}, reply) is None:
        return missing
    owner = ctx.tools._ctx
    facts = {}
    _replace_response_meta(facts, reported, reply)
    status = facts.get("response_status", (reported.get("claudexor") or {}).get("outcome"))
    if (inline_presence(owner) or _loop()._live_delivery_candidate(ctx) is not None
            or getattr(owner, "_delivery_control_required", False) or getattr(owner, "_completion_request", None)
            or getattr(owner, "_completion_selected", None) or controls.get("owner_carried_addressed")
            or owner_source != owner_source_sha256(owner) or (getattr(error, "read_operation", None) or {}).get("owner_carried_addressed")  # any drain since the read's request
            or status != "completed" or forced_response_is_incomplete(facts)):
        return missing
    content = reply.get("content")
    if (reply.get("role") != "assistant" or reply.get("function_call") or reply.get("refusal")
            or not (isinstance(content, str) or isinstance(content, list) and all(
                isinstance(part, dict) and part.get("type") in ("text", "refusal")
                and isinstance(part.get(part["type"]), str) for part in content))):
        return missing
    text, refusal_control = _completion_words(content)
    words = text if refusal_control is None else refusal_control
    if not text.strip() or (_loop()._answer_protocol_active(owner) and not extract_final_answer(words)):
        return missing
    parsed, duplicate, embedded = _parse_delivery_control_body(words)
    if duplicate or embedded or (isinstance(parsed, dict) and ("delivery_control" in parsed or
            parsed.get("action") in ("finish", "stop"))):
        return missing
    # A deadline may have ended the read before its Pause check: only an open fence, read at once, authorizes this answer.
    return (text, refusal_control) if fence_open_now(owner) else missing
