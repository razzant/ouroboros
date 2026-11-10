"""Recovery of settled replies that did not end as a final, carried by the existing checkpoint.

An empty length reply: an informed chance, one source-address pass, then the configured route walk.
A reply with text and no typed call whose provider finish names its output limit or a tool call
(``unfinished_reply``): its text is kept as written and the informed chance is all; another one
ends the work as incomplete, saved as that end for any recovery. Only that typed finish is read,
never the reply's words.
A usable reply or new owner words ends an episode; rounds and Restart do not.
Each attempt of a step is saved pending under its own id before dispatch; only a known end of the
current attempt lets its owner's replacement (wire, image, context) follow, saved pending again.
A saved pending claim is uncertain, never permission to repeat: a call that returned with no known
end stays claimed, and only the execution that heard it return keeps the round's unknown-outcome
rails. No response here teaches a context-window limit.
"""
from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any

from ouroboros._usage_response import OUTPUT_LIMIT_FINISH_REASONS, TOOL_CALL_FINISH_REASONS
from ouroboros.transport_custody import attempt_end_known

KEY = "_output_exhaustion_recovery"
UNFINISHED_KEY = "_unfinished_reply_recovery"
RECORD_KEYS = (KEY, UNFINISHED_KEY)  # the episodes a new execution's recovery carries (``working_checkpoint``)
KIND = "llm_output_exhausted"
REASON = "output_recovery_exhausted"


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


def ready_for_fallback(usage: dict) -> bool:
    return (usage.get(KEY) or {}).get("stage") == "fallback"


def _pending(usage: dict) -> str:
    """The episode whose claimed send is unconfirmed; one claim is held at a time."""
    return next((key for key in RECORD_KEYS if (usage.get(key) or {}).get("pending")), "")


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
    if attempts.attempt and key == KEY:  # the empty episode's ladder; an informed turn stays owed until a reply
        record["last_failure"] = dict(usage.get("_last_llm_failure") or {})
        record["last_failure_kind"] = usage.get("_last_llm_error_kind", "")
        if step != "informed" or usage.get("_last_llm_error_kind") != KIND:
            record["stage"] = "fallback"
    return result


def call(ctx: Any, ordinary_call: Any) -> tuple:
    """Wrap the ordinary call; preserve transport, unknown and control ownership."""
    usage = ctx.accumulated_usage
    for key in RECORD_KEYS:  # new owner words end an episode; never a claimed send's unconfirmed outcome
        if (stale := usage.get(key)) and not stale.get("pending") and stale["owner_revision"] < _owner_revision(ctx):
            usage.pop(key)
    returned_here = getattr(ctx.tools._ctx, "_output_recovery_returned_unknown", "")  # each start builds a new context
    interrupted = (pending := _pending(usage)) and usage[pending].get("attempt_id") != returned_here
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
    if record is None:
        usage[KEY] = {"stage": "informed", "owner_revision": _owner_revision(ctx),
                      "tried": [binding(ctx)], "first": dict(usage.get("_last_llm_output_exhausted") or {})}
        _loop()._append_or_merge_user_message(ctx.messages, _loop()._output_exhausted_notice(usage[KEY]["first"]))
        _save(ctx)  # facts and allowance are one saved transition
        return msg, cost, mode
    if fallback:
        _save(ctx)
        return msg, cost, mode
    return _smaller_request(ctx, cost)


def _smaller_request(ctx: Any, cost: Any) -> tuple:
    from ouroboros.loop_model_call import _failed_capture_is_comparable
    from ouroboros.loop_round_limits import _run_emergency_address_pass

    record = ctx.accumulated_usage[KEY]
    failed = _loop().last_physical_attempt_capture()
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
    from ouroboros.usage_accounting import PhysicalAttemptPreconditionFailed

    usage, record = ctx.accumulated_usage, ctx.accumulated_usage[KEY]
    failed = SimpleNamespace(**record["comparison"], physical_context=None)
    fit = _loop()._measure_round_main_fit(ctx, automatic_pass_used=True)
    try:
        msg, retry_cost = _invoke(ctx, lambda: _loop()._dispatch_round_model(
            ctx, fit, attempt_cap=1, candidate_predicate=_strict_context_shrink_predicate(failed, same_round=False),
            max_tokens=int(failed.max_completion_tokens or 0) or None), "smaller_request")
    except PhysicalAttemptPreconditionFailed:
        msg, retry_cost = None, None
        record.update(stage="fallback", reclaim="candidate_not_smaller")
    if msg is None and str(usage.get("_last_llm_error_kind") or "") == "context_overflow":
        from ouroboros.loop_llm_call import REFUSED_CANDIDATE_KEY
        from ouroboros.loop_model_call import _refused_candidate, _refusal_window

        # The same reading as the context-size path: a pre-send prediction leaves its candidate, never a capture.
        candidate = usage.get(REFUSED_CANDIDATE_KEY)
        refused = _refused_candidate(candidate) if candidate else _loop().last_physical_attempt_capture()
        if refused is not None and getattr(refused, "predicted_free_tokens", None) is None:
            # The smaller send's own provider refusal is window evidence like any other; empty replies are not.
            from ouroboros.context_anchor import record_refused_input
            record_refused_input(usage, ctx.context_fit_plan, _refusal_window(ctx, fit, refused)[1],
                                 measured=fit.measurement.anchor_input_tokens is not None)
    if msg is not None:
        _answered(ctx, msg)
    _save(ctx)
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
