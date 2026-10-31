"""Main admission keeps the final local tokenizer observation bound to its candidate."""
from copy import deepcopy
from dataclasses import replace
import json
import queue
from types import SimpleNamespace

import pytest

from ouroboros import local_model, loop, usage_accounting as ua
from ouroboros.context_budget import LocalContextTooLargeError
from ouroboros.llm_attempt import _attempt_request
from ouroboros.loop_llm_call import _send_main_candidate
from ouroboros.loop_model_call import REFUSED_CANDIDATE_KEY, _presend_context_fit
from tests.test_context_fit_integration import _plan, _projection
from tests.test_nano_local_exact import _nano, data_root as data_root, lane as lane
from tests.test_physical_candidate_capture import _scope


def _ctx(lane):
    return SimpleNamespace(active_model="local-model", accumulated_usage={}, drive_logs=lane["root"] / "logs",
                           event_queue=queue.Queue(), task_id="local-fit", round_idx=1)


def _send(lane, ctx, *, mode="nano", window=16_384):
    # Host chars/4 overestimates this repetitive input; the serving tokenizer
    # counts it at 10,050, leaving 2,238 tokens after the 4,096-token reply.
    kwargs = {"messages": [{"role": "user", "content": "text " * 10_000}],
              "model": "local-model", "use_local": True, "max_tokens": 65_536}
    physical = replace(_nano(window), rendered_mode=mode, profile="owner_" + mode)
    with ua.usage_scope(_scope(lane["root"])), _presend_context_fit(ctx) as admit:
        return _send_main_candidate(
            lane["client"], deepcopy(kwargs), model="local-model", use_local=True, deadline_ts=None,
            physical_context=physical, candidate_predicate=None, physical_fit_predicate=admit)


@pytest.mark.parametrize("mode", ["nano", "low", "max"])
def test_exact_fit_outranks_density_on_every_main_mode(lane, mode):
    lane["count"] = lambda payload: 10_050
    ctx = _ctx(lane)
    assert _send(lane, ctx, mode=mode)[0]["content"] == "local"
    assert len(lane["measured"]) == len(lane["sent"]) == 1
    assert lane["sent"][0]["max_tokens"] == 4_096
    assert "text " * 10_000 in json.dumps(lane["sent"][0])
    assert REFUSED_CANDIDATE_KEY not in ctx.accumulated_usage


def test_confirmed_serving_window_outranks_the_plan_reading(lane):
    lane["count"] = lambda payload: 10_050
    assert _send(lane, _ctx(lane), window=12_000)[0]["content"] == "local"
    assert len(lane["measured"]) == len(lane["sent"]) == 1


@pytest.mark.parametrize("mode", ["nano", "low", "max"])
def test_exact_overflow_is_never_sent(lane, mode):
    lane["count"] = lambda payload: 15_000
    ctx = _ctx(lane)
    if mode == "nano":
        with pytest.raises(LocalContextTooLargeError) as caught:
            _send(lane, ctx, mode=mode)
        facts = caught.value.refused_candidate
    else:
        assert _send(lane, ctx, mode=mode) is None
        facts = ctx.accumulated_usage[REFUSED_CANDIDATE_KEY]
        assert facts["cause"] == "exact_local_count" and "predicted_free_tokens" not in facts
        assert ctx.accumulated_usage["_last_llm_failure"]["model"] == "local-model"
    assert facts["input_tokens"] == 15_000 and facts["context_window_tokens"] == 16_384
    assert not lane["sent"] and len(lane["measured"]) == 1


@pytest.mark.parametrize("supported", [True, False], ids=["stale-fingerprint", "no-exact-support"])
def test_unproven_count_cannot_override_admission(lane, monkeypatch, supported):
    manager = local_model.get_manager()
    measure = manager.measure_prepared_input

    def unproven(payload):
        return {**measure(payload), "supported": supported, "native_input_sha256": "wrong"}

    manager.measure_prepared_input = unproven
    monkeypatch.setattr(local_model, "get_manager", lambda: manager)
    ctx = _ctx(lane)
    assert _send(lane, ctx) is None
    assert ctx.accumulated_usage[REFUSED_CANDIDATE_KEY]["cause"] == "known_window_prediction"
    assert ctx.accumulated_usage["_last_llm_failure"]["model"] == "local-model"
    assert not lane["sent"] and len(lane["measured"]) == 1


def test_changed_serving_instance_never_reaches_admission(lane):
    def count(payload):
        lane["pid"] = 456
        return 10_050

    lane["count"] = count
    with pytest.raises(ua.PhysicalAttemptPreparationFailed, match="instance changed"):
        _send(lane, _ctx(lane))
    assert not lane["sent"] and len(lane["measured"]) == 1


def test_exact_facts_are_bound_to_the_final_input_and_local_lane(lane):
    client = lane["client"]
    target, payload = client._build_local_candidate(
        [{"role": "user", "content": "input"}], None, 65_536, "auto")
    candidate = client._finalize_local_candidate(target, payload)
    request = _attempt_request(target, candidate)
    assert request.prompt_tokens_exact == 5_000 and request.context_window_tokens_exact == 16_384
    changed = deepcopy(candidate)
    changed["messages"].append({"role": "user", "content": "new input"})
    for t, p in ((target, changed), ({**target, "provider": "openai"}, candidate)):
        stale = _attempt_request(t, p)
        assert stale.prompt_tokens_exact is None and stale.context_window_tokens_exact is None
    assert len(lane["measured"]) == 1  # transport of the facts never tokenizes again


def test_exact_fit_reaches_ordinary_round_dispatch(lane):
    lane["count"] = lambda payload: 10_050
    plan = replace(_plan(preferred="nano", window=16_384), model="local-model", provider="local",
                   output_reserve_tokens=4_096, nano_projection=_projection("nano"),
                   user_content_json=json.dumps("text " * 10_000))
    messages, events = plan.messages_for("nano"), queue.Queue()
    inner = SimpleNamespace(task_id="local-fit", task_attempt=1, task_metadata={},
                            context_fit_plan=plan, messages=messages, active_context_mode="nano",
                            active_model="local-model", event_queue=events,
                            drive_logs=lambda: lane["root"] / "logs")
    ctx = loop._RoundModelCallContext(
        llm=lane["client"], messages=messages, tools=SimpleNamespace(_ctx=inner), context_fit_plan=plan,
        active_model="local-model", tool_schemas=[], active_effort="medium", max_retries=1,
        drive_logs=lane["root"] / "logs", task_id="local-fit", round_idx=1, event_queue=events,
        accumulated_usage={}, task_type="task", active_use_local=True, active_context_mode="nano",
        drive_root=lane["root"])
    fit = loop._measure_round_main_fit(ctx, automatic_pass_used=False)
    answer, _ = loop._dispatch_round_model(ctx, fit, attempt_cap=1)
    assert answer["content"] == "local"
    assert len(lane["measured"]) == len(lane["sent"]) == 1
    assert "text " * 10_000 in json.dumps(lane["sent"][0])
