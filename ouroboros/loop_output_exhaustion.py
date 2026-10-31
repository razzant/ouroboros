"""Recovery of settled empty length replies, carried by the existing checkpoint.

An informed chance, one source-address pass, then the configured route walk.
A usable reply or new owner words ends the episode; rounds and Restart do not.
Only the physical claim marks a send pending. A saved pending claim is uncertain,
never permission to repeat. No response here teaches a context-window limit.
"""
from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any

KEY = "_output_exhaustion_recovery"
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


def ready_for_fallback(usage: dict) -> bool:
    return (usage.get(KEY) or {}).get("stage") == "fallback"


def no_recap(usage: dict) -> bool:
    record = usage.get(KEY) or {}
    return bool(record.get("pending") or record.get("stage") in {"fallback", "exhausted"})


def next_step(usage: dict) -> str:
    """Only our own failure ends this episode; other failure/control rails win."""
    if (_loop().provider_no_call_source(usage, False)[0]
            or usage.get("_last_llm_error_kind") not in {KIND, "output_recovery_interrupted"}):
        return ""
    return "finish" if no_recap(usage) else "inform"


def fallback_untried(ctx: Any) -> bool:
    """Eligibility only: the physical claim below consumes the binding."""
    record = ctx.accumulated_usage.get(KEY)
    return not record or binding(ctx) not in record["tried"]


def _invoke(ctx: Any, send: Any, step: str) -> Any:
    """Use the shared physical claim, so an unstarted Pause/budget wait spends nothing."""
    from ouroboros.usage_accounting import PhysicalAttemptPreconditionFailed

    usage, record = ctx.accumulated_usage, ctx.accumulated_usage[KEY]
    prior_stage, target, claimed = record["stage"], binding(ctx), []
    def claim(reservation):
        if claimed:
            raise PhysicalAttemptPreconditionFailed("output recovery send already claimed")
        claimed.append(reservation.attempt_id)
        record.update(pending=step, attempt_id=reservation.attempt_id)
        if step == "fallback":
            record["tried"].append(target)
        _save(ctx)
    def refund_unsent():
        capture = _loop().last_physical_attempt_capture()
        if (claimed and capture is not None and capture.attempt_id == claimed[0]
                and capture.state == "released"):
            claimed.clear()
            record.pop("pending", None)
            record["stage"] = prior_stage
            if step == "fallback":
                record["tried"].remove(target)
            _save(ctx)
    prior_cap, ctx.attempt_cap = ctx.attempt_cap, 1
    prior_claim, ctx.physical_send_claim = getattr(ctx, "physical_send_claim", None), claim
    try:
        result = send()  # preparation helpers run before the Main-only claim scope
    except BaseException:
        refund_unsent()
        raise
    finally:
        ctx.attempt_cap = prior_cap
        ctx.physical_send_claim = prior_claim
    refund_unsent()
    record.pop("pending", None)
    if claimed:
        record["last_failure"] = dict(usage.get("_last_llm_failure") or {})
        record["last_failure_kind"] = usage.get("_last_llm_error_kind", "")
        if step != "informed" or usage.get("_last_llm_error_kind") != KIND:
            record["stage"] = "fallback"
    return result


def call(ctx: Any, ordinary_call: Any) -> tuple:
    """Wrap the ordinary call; preserve transport, unknown and control ownership."""
    usage = ctx.accumulated_usage
    record = usage.get(KEY)
    if record and not record.get("pending") and record["owner_revision"] < _owner_revision(ctx):
        usage.pop(KEY)
        record = None
    if record and record.get("pending"):
        record.update(stage="exhausted", interrupted=True)
        usage["_last_llm_error_kind"] = "output_recovery_interrupted"
        return None, None, ctx.active_context_mode
    fallback = bool(getattr(ctx, "output_recovery_fallback", False))
    if record and record["stage"] == "smaller" and not fallback:
        return _send_smaller(ctx, None)
    if record and record["stage"] != "informed" and not fallback:
        usage["_last_llm_error_kind"] = KIND
        return None, None, ctx.active_context_mode
    msg, cost, mode = (_invoke(ctx, lambda: ordinary_call(ctx), "fallback" if fallback else "informed")
                       if record else ordinary_call(ctx))
    if msg is not None:
        usage.pop(KEY, None)
        return msg, cost, mode
    if usage.get("_last_llm_error_kind") != KIND or _loop().provider_no_call_source(usage, False)[0]:
        if record:
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
            record_refused_input(usage, ctx.context_fit_plan, _refusal_window(ctx, fit, refused)[1])
    if msg is not None:
        usage.pop(KEY, None)
    _save(ctx)
    return msg, float(cost or 0.0) + float(retry_cost or 0.0), ctx.active_context_mode


def finish(ctx: Any, *, reason: str = REASON) -> tuple:
    """Preserve existing model bytes and files, never generate an explanation."""
    from ouroboros.task_finalization import set_terminal_host_notice
    usage = ctx.accumulated_usage
    record = usage.get(KEY) or {}
    record["stage"] = "exhausted"
    notice = ("Work is incomplete: a recovery attempt was interrupted and its outcome is unconfirmed. "
              "It was not repeated." if record.get("interrupted") else
              "Work is incomplete: the model returned empty length-limited replies and the permitted "
              "recovery steps did not produce an answer. No extra generation was made to summarize the failure.")
    notice += " Saved work remains available; choose whether and how to continue."
    candidate = _loop()._live_delivery_candidate(ctx)
    saved = candidate.full_text if candidate is not None else _loop()._last_assistant_text(ctx.messages)
    # Stamped before composing: the composer may end this rail on the owner's Stop now.
    usage.update(execution_status="infra_failed" if reason == REASON else "failed", reason_code=reason)
    text, usage, trace = _loop()._forced_fallback_result(
        ctx, ctx.llm_trace, saved or notice, reason, source="output_exhaustion_no_recap", host_words=not saved)
    trace["output_recovery"] = copy.deepcopy(record)
    set_terminal_host_notice(usage, str(usage.get("terminal_host_notice") or ""), notice)
    return text, usage, trace
