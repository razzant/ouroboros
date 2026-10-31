"""Serial Main fit and the refusal passes: a typed size refusal measured, recovered model-free, retried once per pass."""

from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ouroboros.context_budget import RECLAIM_LOW_WATER_DIVISOR


def _fit(
    *,
    action="send",
    profile="owner_max",
    mode="max",
    goal=0,
    target_deficit=None,
    capacity_deficit=None,
    used=False,
    estimated_input=120_000,
    low_water_margin=0,
    anchor_input=None,
    anchor_round=None,
    accepted_input=None,
    reply_allowance=0,
    round_id="exec:round:1",
):
    from ouroboros.context_fit import MainFitDisposition, MainFitMeasurement

    measurement = MainFitMeasurement(
        route_fp="route-a",
        round_id=round_id,
        profile=profile,
        rendered_mode=mode,
        estimated_input_tokens=estimated_input,
        response_reserve_tokens=65_536,
        target_total_tokens=250_000 if profile == "owner_low" else None,
        capacity_total_tokens=500_000,
        measurement_basis="fresh_route_usage" if anchor_input else "cold_estimate",
        measurement_density=1.0,
        target_deficit_tokens=target_deficit,
        capacity_deficit_tokens=capacity_deficit,
        reclaim_goal_tokens=goal,
        low_water_margin_tokens=low_water_margin,
        reply_allowance_tokens=reply_allowance,
        anchor_input_tokens=anchor_input,  # this conversation's own accepted input (``context_anchor``)
        anchor_round=anchor_round,
        accepted_input_tokens=accepted_input,  # the largest input this model accepted in this task
    )
    return MainFitDisposition(
        measurement=measurement,
        action=action,
        automatic_pass_used=used,
        predicted_capacity_miss=bool(capacity_deficit),
    )


def _plan(*, preferred="max"):
    def reproject(messages, mode):
        rebuilt = list(messages)
        rebuilt[0] = {"role": "system", "content": f"{mode.upper()}_SYSTEM"}
        return rebuilt

    plan = SimpleNamespace(
        model="same-model", route_fp="route-a", preferred_mode=preferred, core_sha256="a" * 64,
        status="confirmed", stale=False,  # evidence-shaped, as ``is_known`` reads the real plan
        window_tokens=500_000, output_reserve_tokens=65_536,
        projection=lambda _mode: SimpleNamespace(calibration_ratio=1.0),
        reproject_transcript=reproject, reproject_for_route=Mock(),
    )
    # The rebound plan names the selected mode, as the real projection does.
    plan.reproject_for_route.return_value = SimpleNamespace(**vars(plan), initial_mode="low")
    return plan


def _lowered_from_max(plan, messages):
    """The one rebind the floor makes when memory alone did not save the target: Low books from the level Max showed."""
    plan.reproject_for_route.assert_called_once_with(
        window_tokens=500_000, known_window=True, ratio=1.0, output_reserve=65_536,
        tool_schemas=[], start_mode="max", current_messages=messages, select_mode="low")
    assert plan.reproject_for_route.call_args.kwargs["current_messages"] is messages  # actual transcript, not a copy


def _ctx(tmp_path, *, preferred="max", mode="max"):
    from ouroboros import loop

    inner = SimpleNamespace(
        messages=[],
        active_context_mode=mode,
        task_metadata={},
        _context_overflow_retries=set(),
    )
    tools = SimpleNamespace(_ctx=inner)
    messages = [
        {"role": "system", "content": f"{mode.upper()}_SYSTEM"},
        {"role": "user", "content": "go"},
    ]
    inner.messages = messages
    return loop._RoundModelCallContext(
        llm=SimpleNamespace(),
        messages=messages,
        tools=tools,
        context_fit_plan=_plan(preferred=preferred),
        active_model="same-model",
        tool_schemas=[],
        active_effort="high",
        max_retries=3,
        drive_logs=tmp_path / "logs",
        task_id="task",
        round_idx=1,
        event_queue=None,
        accumulated_usage={"execution_id": "exec"},
        task_type="task",
        active_use_local=False,
        active_context_mode=mode,
        drive_root=tmp_path,
    )


def _failed_capture(*, profile="owner_max", mode="max", reserve=65_536, size=1_000):
    from ouroboros.usage_accounting import PhysicalAttemptCapture, PhysicalAttemptContext

    physical = PhysicalAttemptContext(
        profile=profile,
        rendered_mode=mode,
        measurement_basis="cold_estimate",
        route_fp="route-a",
        round_id="exec:round:1",
        target_total_tokens=250_000 if profile == "owner_low" else None,
        capacity_total_tokens=500_000,
        context_target_miss=False,
        automatic_pass_used=False,
    )
    return PhysicalAttemptCapture(
        attempt_id="failed",
        model="same-model",
        provider="openai",
        state="unresolved",
        candidate_measurement_kind="canonical_json_v1",
        max_completion_tokens=reserve,
        candidate_raw_sha256="old",
        candidate_raw_size_bytes=size + 100,
        candidate_context_sha256="old-context",
        candidate_context_size_bytes=size,
        physical_context=physical,
    )


def _candidate_request(disposition, *, size, reserve=65_536):
    from ouroboros import loop
    from ouroboros.usage_accounting import AttemptRequest

    return AttemptRequest(
        model="same-model",
        provider="openai",
        max_completion_tokens=reserve,
        candidate_raw_sha256="new",
        candidate_raw_size_bytes=size + 100,
        candidate_context_sha256="new-context",
        candidate_context_size_bytes=size,
        candidate_measurement_kind="canonical_json_v1",
        physical_context=loop._physical_context_for_fit(disposition),
    )


def _measure_cycle(dispositions):
    """A measurement stub that records every fit and keeps returning the last one."""
    from itertools import chain, repeat
    from ouroboros import loop

    fits = chain(dispositions[:-1], repeat(dispositions[-1]))

    def measure(ctx, **_kwargs):
        disposition = next(fits)
        loop._remember_main_fit(ctx, disposition)
        return disposition

    return measure


def _steps_applied(monkeypatch, outcome=lambda step: True):
    """Stub every refusal step; ``outcome(step)`` says whether it changed the candidate.

    A changing stub sheds 0 measured tokens, so every later step of its pass keeps the whole
    target (``loop_model_call._apply_refusal_pass``); an unchanged one answers None."""
    from ouroboros import loop_model_call

    steps = []

    def apply(ctx, step, fit, target, slack=None):
        steps.append((step, target))
        return 0 if outcome(step) else None

    monkeypatch.setattr(loop_model_call, "_apply_refusal_step", apply)
    return steps


PASS_ONE, PASS_TWO = ("exec:round:1", "host_copies+bodies"), ("exec:round:1", "floor+books+unseen_bodies")


def test_fallback_success_preserves_complete_candidate_fit_facts(tmp_path):
    from ouroboros import loop

    candidate_plan = SimpleNamespace(route_fp="fallback-route")
    messages = [{"role": "user", "content": "primary"}]
    fallback_messages = [{"role": "user", "content": "fallback"}]
    inner = SimpleNamespace()
    usage = {
        "_context_route_fp": "fallback-route",
        "_context_prompt_estimate": 123,
        "_context_profile": "task_local_low",
        "_context_measurement_basis": "fresh_route_usage",
        "_context_measurement_density": 1.7,
        "_context_target_miss": False,
        "_context_automatic_pass_used": True,
    }
    expected = dict(usage)
    loop._adopt_fallback_route(
        SimpleNamespace(active_model="primary"), SimpleNamespace(_ctx=inner),
        "fallback", False, messages, fallback_messages, candidate_plan, "low", [], usage,
    )
    assert usage == expected
    assert messages == fallback_messages
    assert inner.context_fit_plan is candidate_plan


def test_failed_fallback_restores_entire_primary_fit_bundle():
    from ouroboros import loop

    usage = {
        "_context_route_fp": "primary",
        "_context_profile": "owner_max",
        "_context_measurement_basis": "cold_estimate",
        "_context_target_total_tokens": None,
        "unrelated": "kept",
    }
    snapshot = loop._snapshot_context_fit_usage(usage)
    usage.update({
        "_context_route_fp": "rejected",
        "_context_profile": "task_local_low",
        "_context_target_miss": True,
        "_context_reclaim_goal_tokens": 999,
    })
    loop._restore_context_fit_usage(usage, snapshot)
    assert loop._snapshot_context_fit_usage(usage) == snapshot
    assert usage["unrelated"] == "kept"


def test_predicted_deficit_alone_sends_without_a_recovery_pass(tmp_path, monkeypatch):
    """7A: prediction is recorded as facts; no pass and no downgrade runs before the send."""
    from ouroboros import loop

    context = _ctx(tmp_path, preferred="low", mode="low")
    dispatched = []
    monkeypatch.setattr(loop, "_measure_round_main_fit", _measure_cycle([
        _fit(action="send_target_miss", profile="owner_low", mode="low", goal=10_000, target_deficit=10_000)]))
    steps = _steps_applied(monkeypatch)

    def dispatch(_ctx, disposition, **kwargs):
        dispatched.append((disposition, kwargs))
        return {"role": "assistant", "content": "ok", "tool_calls": []}, 0.0

    monkeypatch.setattr(loop, "_dispatch_round_model", dispatch)
    msg, _cost, mode = loop._call_round_model(context)
    assert msg["content"] == "ok" and mode == "low"
    assert steps == [] and len(dispatched) == 1
    assert dispatched[0][0].action == "send_target_miss"  # the prediction travels as a fact, not as an action
    assert dispatched[0][1].get("candidate_predicate") is None
    assert context.accumulated_usage["_context_reclaim_goal_tokens"] == 10_000
    assert context.accumulated_usage["_context_target_miss"] is True
    assert context.messages[0]["content"] == "LOW_SYSTEM"
    assert not context.tools._ctx._context_overflow_retries


# --- the measured target -------------------------------------------------------------------------------------

STATED = "This model's maximum context length is 500000 tokens. However, your messages resulted in 520000 tokens."
NUMBERLESS = "Your input exceeds the context window of this model"
SLACK = math.ceil(500_000 / RECLAIM_LOW_WATER_DIVISOR)  # the owner's return-after-fold eighth


@pytest.mark.parametrize("message,anchor,accepted,estimate,target", [
    (STATED, 480_000, 480_000, 520_000, 520_000 + 65_536 - 500_000 + SLACK),  # the provider's stated limit
    (NUMBERLESS, 480_000, 480_000, 520_000, 520_000 - 480_000 + SLACK),  # the refused estimate over the accepted floor
    (NUMBERLESS, 480_000, 480_000, 480_000, None),  # an estimate at the floor contradicts the reading: unknown
    (NUMBERLESS, 480_000, 480_000, 470_000, None),  # below the floor likewise
    (NUMBERLESS, 480_000, None, 520_000, None),  # nothing accepted yet: no floor to measure against
    (STATED, None, None, 520_000, 520_000 + 65_536 - 500_000 + SLACK),  # a stated limit measures even a cold E
])
def test_refusal_target_is_measured_from_a_stated_limit_or_the_accepted_floor_else_unknown(
        tmp_path, monkeypatch, message, anchor, accepted, estimate, target):
    from ouroboros import capability_evidence as ce, loop_model_call
    from ouroboros.context_anchor import refused_input

    # A stated limit is a durable window observation: it stays in this test's store, not the session's.
    monkeypatch.setattr(ce, "canonical_evidence_root", lambda: tmp_path)
    context = _ctx(tmp_path)
    context.accumulated_usage["_last_llm_provider_message"] = message
    fit = _fit(estimated_input=estimate, anchor_input=anchor, anchor_round=3, accepted_input=accepted,
               reply_allowance=65_536)
    assert loop_model_call._refusal_target(context, fit)[0] == target
    # The refused size is recorded whatever the target: the facts stop stating free room for this model.
    assert refused_input(context.accumulated_usage, context.context_fit_plan) == estimate


# --- the two passes ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("second_refused", [False, True])
def test_each_pass_earns_one_retry_and_the_second_pass_runs_only_after_a_refused_retry(tmp_path, monkeypatch, second_refused):
    """Both directions: pass 1 (host copies, shown bodies) earns one strictly smaller retry; when that retry
    is accepted the floor and the late rescue never run; when it is refused again pass 2 runs once and its
    retry must shrink below THAT refused candidate, not the first one."""
    from ouroboros import loop

    context = _ctx(tmp_path)
    monkeypatch.setattr(loop, "_measure_round_main_fit", _measure_cycle([_fit()]))
    steps = _steps_applied(monkeypatch)
    sends, captures = [], iter([_failed_capture(size=1_000), _failed_capture(size=900)])
    current = {"value": next(captures)}

    def dispatch(ctx, disposition, *, candidate_predicate=None, **_kwargs):
        sends.append(candidate_predicate)
        if len(sends) == 1:  # the actual request: refused
            ctx.accumulated_usage["_last_llm_error_kind"] = "context_overflow"
            return None, 0.0
        if len(sends) == 2 and second_refused:  # pass 1's smaller candidate (900 < 1000): refused again
            assert candidate_predicate(_candidate_request(disposition, size=900))
            current["value"] = next(captures)
            ctx.accumulated_usage["_last_llm_error_kind"] = "context_overflow"
            return None, 0.0
        if len(sends) == 3:
            assert not candidate_predicate(_candidate_request(disposition, size=900))  # the base moved to 900
        assert candidate_predicate(_candidate_request(disposition, size=800))
        return {"role": "assistant", "content": "fits", "tool_calls": []}, 0.0

    monkeypatch.setattr(loop, "_dispatch_round_model", dispatch)
    monkeypatch.setattr(loop, "last_physical_attempt_capture", lambda: current["value"])
    msg, _cost, mode = loop._call_round_model(context)
    assert msg["content"] == "fits" and mode == "max"
    assert len(sends) == (3 if second_refused else 2)
    expected = ["host_copies", "bodies"] + (["floor", "books", "unseen_bodies"] if second_refused else [])
    assert [step for step, _target in steps] == expected
    assert all(target is None for _step, target in steps)  # a cold first request measures no target
    assert context.tools._ctx._context_overflow_retries == ({PASS_ONE, PASS_TWO} if second_refused else {PASS_ONE})


def test_a_pass_that_changes_nothing_earns_no_retry_and_the_next_pass_still_runs(tmp_path, monkeypatch):
    from ouroboros import loop

    context = _ctx(tmp_path)
    monkeypatch.setattr(loop, "_measure_round_main_fit", _measure_cycle([_fit()]))
    steps = _steps_applied(monkeypatch, outcome=lambda step: step in ("floor", "unseen_bodies"))
    sends = []

    def dispatch(ctx, disposition, *, candidate_predicate=None, **_kwargs):
        sends.append(candidate_predicate)
        if len(sends) == 1:
            ctx.accumulated_usage["_last_llm_error_kind"] = "context_overflow"
            return None, 0.0
        assert candidate_predicate(_candidate_request(disposition, size=800))
        return {"role": "assistant", "content": "fits", "tool_calls": []}, 0.0

    monkeypatch.setattr(loop, "_dispatch_round_model", dispatch)
    monkeypatch.setattr(loop, "last_physical_attempt_capture", lambda: _failed_capture())
    msg, _cost, _mode = loop._call_round_model(context)
    assert msg["content"] == "fits" and len(sends) == 2  # no retry was bought for the unchanged candidate
    assert [step for step, _target in steps] == ["host_copies", "bodies", "floor", "books", "unseen_bodies"]
    assert context.tools._ctx._context_overflow_retries == {PASS_ONE, PASS_TWO}


@pytest.mark.parametrize("started_by", ["numbered_refusal", "prediction"])
def test_the_rounds_first_failure_decides_whether_a_just_chosen_body_pays_last(tmp_path, monkeypatch, started_by):
    """Owner 2A, both directions: pass 1 sheds part of the target and the pre-send prediction stops its retry. A round
    a provider's numbered refusal started stays that refusal's round, so pass 2 may take the just-chosen body last,
    for what its other steps left; a round a pre-send prediction started never takes one."""
    from dataclasses import asdict
    from ouroboros import capability_evidence as ce, loop, loop_model_call
    from ouroboros.loop_llm_call import REFUSED_CANDIDATE_KEY

    monkeypatch.setattr(ce, "canonical_evidence_root", lambda: tmp_path)  # a stated limit is a durable observation
    context = _ctx(tmp_path)
    monkeypatch.setattr(loop, "_measure_round_main_fit", _measure_cycle([_fit(estimated_input=520_000)]))
    shed = {"host_copies": 10_000, "floor": 5_000}  # each pass sheds part of its target, never all of it
    monkeypatch.setattr(loop_model_call, "_apply_refusal_step", lambda ctx, step, fit, target, slack=None: shed.get(step))
    chosen = []
    monkeypatch.setattr(loop_model_call, "_run_emergency_address_pass", lambda ctx, fit, *, rung, goal_tokens=None: (
        chosen.append((rung, goal_tokens)) or SimpleNamespace(status="applied")))
    sends = []

    def dispatch(ctx, disposition, *, candidate_predicate=None, **_kwargs):
        sends.append(candidate_predicate)
        if len(sends) == 3:
            assert candidate_predicate(_candidate_request(disposition, size=800))
            return {"role": "assistant", "content": "fits", "tool_calls": []}, 0.0
        ctx.accumulated_usage["_last_llm_error_kind"] = "context_overflow"
        if len(sends) == 1 and started_by == "numbered_refusal":
            ctx.accumulated_usage["_last_llm_provider_message"] = STATED
        else:  # the pre-send gate stopped this candidate: its facts, no provider words
            ctx.accumulated_usage.update({"_last_llm_provider_message": "", REFUSED_CANDIDATE_KEY: {
                **asdict(_failed_capture(size=1_000 if len(sends) == 1 else 900)),
                "predicted_free_tokens": -30_000, "predicted_slack_tokens": 0}})
        return None, 0.0

    monkeypatch.setattr(loop, "_dispatch_round_model", dispatch)
    monkeypatch.setattr(loop, "last_physical_attempt_capture", lambda: _failed_capture())
    msg, _cost, _mode = loop._call_round_model(context)
    assert msg["content"] == "fits" and len(sends) == 3
    assert context.tools._ctx._context_overflow_retries == {PASS_ONE, PASS_TWO}
    # Pass 2 measures the predicted retry (30,000 over); the floor took 5,000 of it.
    assert chosen == ([("chosen_bodies", 25_000)] if started_by == "numbered_refusal" else [])


def test_unchanged_candidate_after_every_pass_gets_no_retry(tmp_path, monkeypatch):
    """Negative: when no step changes the candidate there is no paid retry at all; the round returns
    unanswered to the ordinary recovery with both passes spent."""
    from ouroboros import loop

    context = _ctx(tmp_path, preferred="low", mode="low")
    sends = []
    monkeypatch.setattr(loop, "_measure_round_main_fit", _measure_cycle([_fit(profile="owner_low", mode="low")]))
    _steps_applied(monkeypatch, outcome=lambda step: False)

    def dispatch(ctx, disposition, **_kwargs):
        sends.append(disposition)
        ctx.accumulated_usage["_last_llm_error_kind"] = "context_overflow"
        return None, 0.0

    monkeypatch.setattr(loop, "_dispatch_round_model", dispatch)
    monkeypatch.setattr(loop, "last_physical_attempt_capture", lambda: _failed_capture(profile="owner_low", mode="low"))
    msg, _cost, mode = loop._call_round_model(context)
    assert msg is None and mode == "low" and len(sends) == 1
    assert context.accumulated_usage["_last_llm_error_kind"] == "context_overflow"  # the chain and the terminal read it
    assert context.tools._ctx._context_overflow_retries == {PASS_ONE, PASS_TWO}


def test_a_fallback_candidate_in_the_same_round_walks_no_pass_again(tmp_path, monkeypatch):
    """Both directions: a candidate of a round whose passes already ran (the configured fallback receives the
    applied view) is refused without another pass; a new round walks its passes afresh."""
    from ouroboros import loop

    context = _ctx(tmp_path)
    context.tools._ctx._context_overflow_retries = {PASS_ONE, PASS_TWO}
    monkeypatch.setattr(loop, "_measure_round_main_fit", _measure_cycle([_fit()]))
    steps = _steps_applied(monkeypatch)
    sends = []

    def dispatch(ctx, disposition, *, candidate_predicate=None, **_kwargs):
        sends.append(candidate_predicate)
        if candidate_predicate is None:
            ctx.accumulated_usage["_last_llm_error_kind"] = "context_overflow"
            return None, 0.0
        return {"role": "assistant", "content": "fits", "tool_calls": []}, 0.0

    monkeypatch.setattr(loop, "_dispatch_round_model", dispatch)
    monkeypatch.setattr(loop, "last_physical_attempt_capture", lambda: _failed_capture())
    msg, _cost, _mode = loop._call_round_model(context)
    assert msg is None and len(sends) == 1 and steps == []
    assert context.tools._ctx._context_overflow_retries == {PASS_ONE, PASS_TWO}

    context.round_idx, sends[:] = 2, []
    monkeypatch.setattr(loop, "_measure_round_main_fit", _measure_cycle([_fit(round_id="exec:round:2")]))
    msg, _cost, _mode = loop._call_round_model(context)
    assert msg["content"] == "fits" and len(sends) == 2
    assert [step for step, _target in steps] == ["host_copies", "bodies"]
    assert ("exec:round:2", "host_copies+bodies") in context.tools._ctx._context_overflow_retries


@pytest.mark.parametrize("saved", [True, False])
def test_the_floor_keeps_books_when_memory_alone_saved_the_target_and_drops_them_when_it_did_not(tmp_path, monkeypatch, saved):
    """Both directions of the owner's books-last order: the floor narrows memory by the measured excess (the target
    without its return-after-fold slack, review r1 17.10); when the narrowed view saved the whole target, Max keeps
    its books; otherwise they become navigation (task-local Low)."""
    from ouroboros import loop, loop_model_call

    context = _ctx(tmp_path)
    target = 520_000 - 480_000 + SLACK  # measured from the accepted floor
    before = _fit(estimated_input=520_000, anchor_input=480_000, anchor_round=3, accepted_input=480_000)
    after = replace(before, measurement=replace(before.measurement,
                                                estimated_input_tokens=520_000 - target if saved else 500_000))
    monkeypatch.setattr(loop, "_measure_round_main_fit", _measure_cycle([before, before, after]))
    monkeypatch.setattr(loop_model_call, "_run_emergency_address_pass",
                        lambda ctx, fit, *, rung, goal_tokens=None: SimpleNamespace(status="no_eligible"))
    coarsened = []
    monkeypatch.setattr(loop_model_call, "_coarsen_memory_view",
                        lambda ctx, fit, *, saving=None: coarsened.append(saving) or True)
    monkeypatch.setattr(loop, "_emit_checkpoint_event", lambda *_a, **_kw: None)
    sends = []

    def dispatch(ctx, disposition, *, candidate_predicate=None, **_kwargs):
        sends.append(candidate_predicate)
        if len(sends) == 1:
            ctx.accumulated_usage["_last_llm_provider_message"] = NUMBERLESS
            ctx.accumulated_usage["_last_llm_error_kind"] = "context_overflow"
            return None, 0.0
        assert candidate_predicate(_candidate_request(disposition, size=800))
        return {"role": "assistant", "content": "fits", "tool_calls": []}, 0.0

    monkeypatch.setattr(loop, "_dispatch_round_model", dispatch)
    monkeypatch.setattr(loop, "last_physical_attempt_capture", lambda: _failed_capture())
    msg, _cost, mode = loop._call_round_model(context)
    assert msg["content"] == "fits" and len(sends) == 2
    assert coarsened == [target - SLACK]  # memory narrows by the measured excess, from the level it shows
    if saved:
        assert mode == "max" and context.messages[0]["content"] == "MAX_SYSTEM"
        context.context_fit_plan.reproject_for_route.assert_not_called()
    else:
        assert mode == "low" and context.messages[0]["content"] == "LOW_SYSTEM"
        _lowered_from_max(context.context_fit_plan, context.messages)


def test_already_low_overflow_never_emits_max_to_low_toast(tmp_path, monkeypatch):
    from ouroboros import loop, loop_model_call

    context = _ctx(tmp_path, preferred="low", mode="low")
    events = []
    calls = 0
    monkeypatch.setattr(loop, "_measure_round_main_fit", _measure_cycle([_fit(profile="owner_low", mode="low")]))
    monkeypatch.setattr(loop_model_call, "_run_emergency_address_pass",
                        lambda ctx, fit, *, rung, goal_tokens=None: SimpleNamespace(status="no_eligible"))
    monkeypatch.setattr(loop_model_call, "_coarsen_memory_view", lambda ctx, fit, *, saving=None: True)

    def dispatch(ctx, disposition, *, candidate_predicate=None, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            ctx.accumulated_usage["_last_llm_error_kind"] = "context_overflow"
            return None, 0.0
        assert candidate_predicate(_candidate_request(disposition, size=700))
        return {"role": "assistant", "content": "fits", "tool_calls": []}, 0.0

    monkeypatch.setattr(loop, "_dispatch_round_model", dispatch)
    monkeypatch.setattr(
        loop, "last_physical_attempt_capture",
        lambda: _failed_capture(profile="owner_low", mode="low"),
    )
    monkeypatch.setattr(loop, "_emit_checkpoint_event", lambda *_a, **kw: events.append(kw or _a[-1]))
    msg, _cost, mode = loop._call_round_model(context)
    assert msg["content"] == "fits"
    assert mode == "low" and context.messages[0]["content"] == "LOW_SYSTEM"
    assert not any(event.get("checkpoint_kind") == "context_fit_low_retry" for event in events)
    context.context_fit_plan.reproject_for_route.assert_not_called()


def test_equal_context_releases_retry_before_provider_send(tmp_path, monkeypatch):
    from ouroboros import loop
    from ouroboros.usage_accounting import PhysicalAttemptPreconditionFailed

    context = _ctx(tmp_path)
    events = []
    provider_sends = 0
    monkeypatch.setattr(loop, "_measure_round_main_fit", _measure_cycle([_fit()]))
    _steps_applied(monkeypatch)

    def dispatch(ctx, disposition, *, candidate_predicate=None, **_kwargs):
        nonlocal provider_sends
        if provider_sends == 0:
            provider_sends += 1
            ctx.accumulated_usage["_last_llm_error_kind"] = "context_overflow"
            return None, 0.0
        if not candidate_predicate(_candidate_request(disposition, size=1_000)):
            raise PhysicalAttemptPreconditionFailed("not smaller")
        provider_sends += 1
        raise AssertionError("equal context must not dispatch")

    monkeypatch.setattr(loop, "_dispatch_round_model", dispatch)
    monkeypatch.setattr(loop, "last_physical_attempt_capture", lambda: _failed_capture())
    monkeypatch.setattr(loop, "_emit_checkpoint_event", lambda *_a, **kw: events.append(kw or _a[-1]))
    msg, _cost, _mode = loop._call_round_model(context)
    assert msg is None
    assert provider_sends == 1  # both passes ran; each equal candidate was released before any send
    skipped = [event for event in events if event.get("reason") == "context_candidate_not_strictly_smaller"]
    assert [event["rung"] for event in skipped] == ["host_copies+bodies", "floor+books+unseen_bodies"]


def test_strict_shrink_predicate_uses_failed_physical_reserve():
    from ouroboros import loop

    failed = _failed_capture(reserve=2_048)
    predicate = loop._strict_context_shrink_predicate(failed)
    disposition = _fit(profile="task_local_low", mode="low")
    assert predicate(_candidate_request(disposition, size=900, reserve=65_536)) is False
    assert predicate(_candidate_request(disposition, size=900, reserve=2_048)) is True


def test_failed_main_capture_is_snapshotted_before_the_passes(tmp_path, monkeypatch):
    """A pass's own physical receipts must not replace the failed Main comparison candidate."""
    from ouroboros import loop, loop_model_call

    context = _ctx(tmp_path)
    current_capture = {"value": _failed_capture(size=1_000)}
    monkeypatch.setattr(loop, "_measure_round_main_fit", _measure_cycle([_fit()]))

    def apply(ctx, step, fit, target, slack=None):
        current_capture["value"] = _failed_capture(size=700)  # a receipt the pass itself produced
        return True

    monkeypatch.setattr(loop_model_call, "_apply_refusal_step", apply)
    sends = 0

    def dispatch(ctx, disposition, *, candidate_predicate=None, **_kwargs):
        nonlocal sends
        sends += 1
        if sends == 1:
            ctx.accumulated_usage["_last_llm_error_kind"] = "context_overflow"
            return None, 0.0
        request = _candidate_request(disposition, size=800)
        assert candidate_predicate is not None
        assert candidate_predicate(request) is True  # smaller than failed Main (1000)
        assert loop._strict_context_shrink_predicate(current_capture["value"])(request) is False
        return {"role": "assistant", "content": "fits", "tool_calls": []}, 0.0

    monkeypatch.setattr(loop, "_dispatch_round_model", dispatch)
    monkeypatch.setattr(loop, "last_physical_attempt_capture", lambda: current_capture["value"])

    msg, _cost, mode = loop._call_round_model(context)
    assert msg["content"] == "fits"
    assert mode == "max"  # the retry fit on the same model and mode
    assert sends == 2


def test_strict_shrink_predicate_requires_entire_physical_tuple():
    from ouroboros import loop

    failed = _failed_capture(size=1_000)
    predicate = loop._strict_context_shrink_predicate(failed)
    disposition = _fit(profile="task_local_low", mode="low")
    accepted = _candidate_request(disposition, size=900)
    assert predicate(accepted) is True

    assert accepted.physical_context is not None
    variants = {
        "provider": replace(accepted, provider="anthropic"),
        "model": replace(accepted, model="other-model"),
        "larger_reserve": replace(accepted, max_completion_tokens=70_000),
        "round": replace(
            accepted,
            physical_context=replace(accepted.physical_context, round_id="exec:round:2"),
        ),
        "raw_digest": replace(accepted, candidate_raw_sha256=failed.candidate_raw_sha256),
        "equal_context": replace(accepted, candidate_context_size_bytes=1_000),
        "larger_context": replace(accepted, candidate_context_size_bytes=1_001),
    }
    for label, candidate in variants.items():
        assert predicate(candidate) is False, label
    # A smaller reply allowance is admitted: the retry's ceiling is the failed attempt's sent allowance.
    assert predicate(replace(accepted, max_completion_tokens=2_048)) is True
    # A same-model account rotation (another route fingerprint, same provider/model/round) cannot veto
    # the retry; its fresh capacity binds the retry through the rebound plan's own measurement.
    assert predicate(replace(accepted, physical_context=replace(accepted.physical_context, route_fp="other-account"))) is True


def test_overflow_retry_is_skipped_while_the_round_holds_an_unresolved_attempt(tmp_path, monkeypatch):
    """A retained explicit-repeat record blocks compaction recovery through the real dispatcher."""
    import json

    from ouroboros import loop
    from ouroboros import usage_accounting as ua
    from ouroboros.loop_llm_call import TRANSPORT_DEATHS_KEY

    def _unresolved(**extra):
        return ua.PhysicalAttemptCapture(
            attempt_id="pa", model="same-model", provider="openrouter", state="unresolved",
            candidate_measurement_kind="opaque", **extra,
        )

    class _Overflow:
        calls = 0

        def chat(self, **_kwargs):
            self.calls += 1
            exc = RuntimeError("routed to a smaller upstream window")
            exc.code = "context_length_exceeded"
            exc.physical_attempt_capture = _unresolved(provider_status_code=400, provider_code="context_length_exceeded")
            raise exc

    context = _ctx(tmp_path)
    context.llm = _Overflow()
    context.accumulated_usage.update(execution_id="exec", **{
        TRANSPORT_DEATHS_KEY: {"round_id": "exec:round:1", "count": 1, "backoff_sec": 4.0}})
    fit = _fit()  # Pure measurement may run again for the per-send facts line.

    def measure(ctx, **_kwargs):
        disposition = fit
        loop._remember_main_fit(ctx, disposition)
        return disposition

    monkeypatch.setattr(loop, "_measure_round_main_fit", measure)
    steps = _steps_applied(monkeypatch)
    monkeypatch.setattr(loop, "last_physical_attempt_capture", lambda: _failed_capture())

    msg, _cost, mode = loop._call_round_model(context)

    assert msg is None
    assert context.llm.calls == 1  # the current refusal; no compaction retry
    assert context.accumulated_usage["_last_llm_error_kind"] == "context_overflow"
    assert context.accumulated_usage[TRANSPORT_DEATHS_KEY]["count"] == 1
    assert steps == [] and not context.tools._ctx._context_overflow_retries
    assert mode == "max"  # the retry was refused before any low re-projection
    rows = [json.loads(line) for line in (tmp_path / "logs" / "events.jsonl").read_text().splitlines() if line.strip()]
    skipped = [row for row in rows if row.get("type") == "context_overflow_retry_skipped"]
    assert [row["reason"] for row in skipped] == ["round_holds_unresolved_attempt"]
    api_errors = [row for row in rows if row.get("type") == "llm_api_error"]
    assert [(row["error_kind"], row["retry_same_request"]) for row in api_errors] == [
        ("context_overflow", False),
    ]
