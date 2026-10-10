"""Overflow recovery in a rendered Nano and the reprepare rebinding of one call (A8/A9).

Shares the serial Main fit fixtures of ``tests.test_loop_compaction``.
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

from tests.test_loop_compaction import _applied, _candidate_request, _ctx, _failed_capture, _fit, _measure_cycle


def test_a_nano_overflow_stays_nano_and_the_retry_carries_the_failed_allowance(tmp_path, monkeypatch):
    """A8: an actual overflow never raises Nano (or Low) to a larger Low; the one strict-shrink retry
    sends under the failed attempt's sent allowance as its logical ceiling, so the rule can never
    produce a larger reply for the smaller input."""
    from ouroboros import loop

    context = _ctx(tmp_path, preferred="nano", mode="nano")
    events, ceilings = [], []
    measure = _measure_cycle([_fit(profile="owner_nano", mode="nano")])

    def reclaim(ctx, _disposition, **_kwargs):
        ctx.tools._ctx._context_reclaim_passes.add(("route-a", "exec:round:1"))
        return _applied()

    def dispatch(ctx, disposition, *, candidate_predicate=None, max_tokens=None, **_kwargs):
        ceilings.append(max_tokens)
        if len(ceilings) == 1:
            ctx.accumulated_usage["_last_llm_error_kind"] = "context_overflow"
            return None, 0.0
        assert candidate_predicate(_candidate_request(disposition, size=700, reserve=20_000))
        assert candidate_predicate(_candidate_request(disposition, size=700, reserve=8_192))
        assert not candidate_predicate(_candidate_request(disposition, size=700, reserve=20_001))
        return {"role": "assistant", "content": "fits", "tool_calls": []}, 0.0

    monkeypatch.setattr(loop, "_measure_round_main_fit", measure)
    monkeypatch.setattr(loop, "_run_main_reclaim", reclaim)
    monkeypatch.setattr(loop, "_dispatch_round_model", dispatch)
    monkeypatch.setattr(loop, "last_physical_attempt_capture",
                        lambda: _failed_capture(profile="owner_nano", mode="nano", reserve=20_000))
    monkeypatch.setattr(loop, "_emit_checkpoint_event", lambda *_a, **kw: events.append(kw or _a[-1]))
    msg, _cost, mode = loop._call_round_model(context)
    assert msg["content"] == "fits" and mode == "nano"
    assert context.messages[0]["content"] == "NANO_SYSTEM"  # never re-projected to Low
    assert ceilings == [None, 20_000]  # the retry's logical ceiling is the failed attempt's sent allowance
    assert not any(event.get("checkpoint_kind") == "context_fit_low_retry" for event in events)


def test_the_retry_ceiling_reaches_the_send_and_the_request_log_names_it(tmp_path, monkeypatch):
    import queue

    from ouroboros import loop_llm_call

    seen, logs = [], tmp_path / "logs"
    logs.mkdir()

    def chat(**kwargs):
        seen.append(kwargs["max_tokens"])
        return {"role": "assistant", "content": "ok"}, {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0}

    llm = SimpleNamespace(chat=chat)
    monkeypatch.setattr(loop_llm_call, "_prepare_main_messages", lambda messages, **_kw: messages)
    recorded = []
    monkeypatch.setattr(loop_llm_call, "persist_observed_call", lambda *_a, **kw: recorded.append(kw) or {})
    loop_llm_call.call_llm_with_retry(llm, [{"role": "user", "content": "go"}], "m", None, "high", 1, logs, "t", 1,
                                      queue.Queue(), {}, max_tokens=20_000)
    loop_llm_call.call_llm_with_retry(llm, [{"role": "user", "content": "go"}], "m", None, "high", 1, logs, "t", 2,
                                      queue.Queue(), {})
    assert seen == [20_000, loop_llm_call.MAIN_LOOP_MAX_TOKENS]
    requests = [kw["payload"] for kw in recorded if kw.get("call_type") == "llm_request"]
    assert [request["max_tokens"] for request in requests] == [20_000, loop_llm_call.MAIN_LOOP_MAX_TOKENS]


def test_a_local_pre_dispatch_refusal_is_compared_with_its_own_candidate_facts(tmp_path, monkeypatch):
    """A8: the local lane refuses before any send, so there is no capture of this round; the retry
    compares with the refused candidate's facts, never with an earlier round's capture."""
    from ouroboros import loop, loop_llm_call, loop_model_call
    from ouroboros.usage_accounting import PhysicalAttemptContext

    context = _ctx(tmp_path, preferred="nano", mode="nano")
    measure = _measure_cycle([_fit(profile="owner_nano", mode="nano")])
    refused = {"model": "same-model", "provider": "local", "max_completion_tokens": 4_096,
               "candidate_measurement_kind": "canonical_json_v1", "candidate_raw_sha256": "refused",
               "candidate_raw_size_bytes": 1_100, "candidate_context_sha256": "refused-context",
               "candidate_context_size_bytes": 1_000, "physical_context": {
                   "profile": "owner_nano", "rendered_mode": "nano", "measurement_basis": "cold_estimate",
                   "route_fp": "route-a", "round_id": "exec:round:1", "target_total_tokens": 85_000,
                   "capacity_total_tokens": 16_384, "context_target_miss": False, "automatic_pass_used": False,
                   "measurement_density": None}}
    stale = _failed_capture(profile="owner_nano", mode="nano")  # an earlier round's capture
    stale = replace(stale, physical_context=replace(stale.physical_context, round_id="exec:round:0"))
    checked = []

    def reclaim(ctx, _disposition, **_kwargs):
        ctx.tools._ctx._context_reclaim_passes.add(("route-a", "exec:round:1"))
        return _applied()

    def dispatch(ctx, disposition, *, candidate_predicate=None, max_tokens=None, **_kwargs):
        if candidate_predicate is None:
            ctx.accumulated_usage["_last_llm_error_kind"] = "context_overflow"
            ctx.accumulated_usage[loop_llm_call.REFUSED_CANDIDATE_KEY] = dict(refused)  # what the error recorder stashes
            return None, 0.0
        smaller = _candidate_request(disposition, size=900, reserve=4_096)
        checked.append((candidate_predicate(smaller), candidate_predicate(replace(smaller, provider="local")), max_tokens))
        return {"role": "assistant", "content": "fits", "tool_calls": []}, 0.0

    monkeypatch.setattr(loop, "_measure_round_main_fit", measure)
    monkeypatch.setattr(loop, "_run_main_reclaim", reclaim)
    monkeypatch.setattr(loop, "_dispatch_round_model", dispatch)
    monkeypatch.setattr(loop, "last_physical_attempt_capture", lambda: stale)
    monkeypatch.setattr(loop, "_emit_checkpoint_event", lambda *_a, **_kw: None)
    msg, _cost, mode = loop._call_round_model(context)
    assert msg["content"] == "fits" and mode == "nano"
    # The retry is admitted against the refused candidate (same round, local, strictly smaller, <= its allowance);
    # the stale capture's provider/round would have refused it. The key is consumed by the round.
    assert checked == [(False, True, 4_096)]
    assert loop_llm_call.REFUSED_CANDIDATE_KEY not in context.accumulated_usage
    assert loop_model_call._failed_capture_is_comparable(loop_model_call._refused_candidate(refused))
    assert isinstance(loop_model_call._refused_candidate(refused).physical_context, PhysicalAttemptContext)
    assert not loop_model_call._failed_capture_is_comparable(replace(stale, state="released"))


def test_the_error_recorder_stashes_a_refused_candidate_and_the_next_call_clears_it(tmp_path):
    import queue

    from ouroboros import loop_llm_call
    from ouroboros.context_budget import LocalContextTooLargeError

    logs = tmp_path / "logs"
    logs.mkdir()
    usage = {}
    error = LocalContextTooLargeError("the window leaves 100 for the reply, below the floor")
    error.refused_candidate = {"model": "local-model", "provider": "local", "max_completion_tokens": 2_048}
    stop = loop_llm_call._record_llm_call_error(error, loop_llm_call._LlmErrorContext(
        task_id="t", task_type="task", execution_id="e", round_id="e:round:1", llm_call_id="c", round_idx=1, attempt=0,
        model="local-model", request_ref=None, drive_logs=logs, event_queue=queue.Queue(), accumulated_usage=usage))
    assert stop and usage["_last_llm_error_kind"] == "context_overflow"
    assert usage[loop_llm_call.REFUSED_CANDIDATE_KEY] == error.refused_candidate
    plain = LocalContextTooLargeError("compacted chars still above target")
    loop_llm_call._record_llm_call_error(plain, loop_llm_call._LlmErrorContext(
        task_id="t", task_type="task", execution_id="e", round_id="e:round:1", llm_call_id="c2", round_idx=1, attempt=0,
        model="local-model", request_ref=None, drive_logs=logs, event_queue=queue.Queue(), accumulated_usage={}))
    llm = SimpleNamespace(chat=lambda **_kw: ({"role": "assistant", "content": "ok"}, {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0}))
    loop_llm_call.call_llm_with_retry(llm, [{"role": "user", "content": "go"}], "m", None, "high", 1, logs, "t", 2,
                                      queue.Queue(), usage, initial_messages=[{"role": "user", "content": "go"}])
    assert loop_llm_call.REFUSED_CANDIDATE_KEY not in usage  # a last-invocation marker, like the retry wall


def test_a_reprepare_inside_one_call_rebinds_the_context_of_its_later_attempts(tmp_path, monkeypatch):
    """A9: a wait's reprepare measures the new route; the attempts that follow in the same
    ``call_llm_with_retry`` invocation send under that context, not the one bound at entry."""
    import queue

    from ouroboros import loop_llm_call
    from ouroboros.usage_accounting import PhysicalAttemptContext, current_physical_attempt_context

    def physical(route):
        return PhysicalAttemptContext(profile="owner_nano", rendered_mode="nano", measurement_basis="cold_estimate",
                                      route_fp=route, round_id="e:round:1", target_total_tokens=85_000,
                                      capacity_total_tokens=128_000, context_target_miss=False, automatic_pass_used=False)

    entry, rebound, bound, logs = physical("route-a"), physical("route-b"), [], tmp_path / "logs"
    logs.mkdir()

    def chat(**_kwargs):
        bound.append(current_physical_attempt_context().route_fp)
        if len(bound) == 1:
            if rebind:  # what _reprepare_waiting_main leaves for the invocation
                usage[loop_llm_call.REBOUND_PHYSICAL_CONTEXT_KEY] = rebound
            error = RuntimeError("server error 502")
            error.status_code = 502
            raise error
        return {"role": "assistant", "content": "ok"}, {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0}

    monkeypatch.setattr(loop_llm_call, "_prepare_main_messages", lambda messages, **_kw: messages)
    monkeypatch.setattr(loop_llm_call, "_sleep_within_deadline", lambda *_a, **_kw: True)
    for rebind in (True, False):
        bound.clear()
        usage = {}
        msg, _cost = loop_llm_call.call_llm_with_retry(
            SimpleNamespace(chat=chat), [{"role": "user", "content": "go"}], "m", None, "high", 2, logs, "t", 1,
            queue.Queue(), usage, physical_context=entry)
        assert msg["content"] == "ok"
        assert bound == ["route-a", "route-b" if rebind else "route-a"]
        assert loop_llm_call.REBOUND_PHYSICAL_CONTEXT_KEY not in usage
