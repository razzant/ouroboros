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


def test_exact_fit_reaches_ordinary_round_dispatch(lane, monkeypatch):
    from ouroboros import llm_attempt
    from ouroboros.context_anchor import main_input_anchor
    from tests.test_physical_candidate_capture import _Response

    lane["count"] = lambda payload: 10_050
    reply = _Response(text="local")
    reply._payload["usage"]["prompt_tokens"] = 10_050  # the server reports the input it counted exactly
    monkeypatch.setattr(lane["client"], "_get_local_client", lambda: SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=lambda **candidate: lane["sent"].append(deepcopy(candidate)) or reply))))
    admitted, admission = [], llm_attempt.main_context_fit_refusal
    monkeypatch.setattr(llm_attempt, "main_context_fit_refusal",
                        lambda request: admitted.append(request.prompt_tokens_bounded_estimate) or admission(request))
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
    # The exact count admitted it (its receipt keeps no estimator count); the anchor still learns on the
    # estimator count of the bytes it answered, so this route keeps its own calibration.
    anchor, floor = main_input_anchor(ctx.accumulated_usage, plan)
    assert (anchor.input_tokens, anchor.estimate_tokens, floor) == (10_050, admitted[-1], 10_050)


@pytest.mark.parametrize("ordinary_first", [False, True])
def test_book_map_retry_submits_each_step_to_exact_local_admission(tmp_path, monkeypatch, ordinary_first):
    import math, re
    from ouroboros import context_fit as fit, loop_model_call, memory_floor
    from ouroboros.context_measurement import _physical_context_for_fit
    from ouroboros.local_model_server import input_fingerprint
    from ouroboros.llm_attempt import _physical_candidate, main_context_fit_refusal
    from ouroboros.loop_messages import append_context_facts
    from tests import test_required_development_book as books
    from tests.test_loop_compaction import _ctx as facts_ctx

    monkeypatch.setattr(fit, "main_output_reserve_tokens", lambda **kw: books.RESERVE)
    monkeypatch.setattr(fit, "_route_calibration_ratio", lambda *a, **kw: 1.0)
    monkeypatch.setattr("ouroboros.capability_evidence.resolve_main_token_density", lambda *a, **kw: (1.0, "fresh_route_usage"))
    core = books.booked(books._core(tmp_path, required=True, memory=False), tmp_path,
                        headings=600, body="Its practice. " * 10)
    roomy = books._plan(tmp_path, core, "low", 1_000_000, source="local_health")
    ctx = facts_ctx(tmp_path)
    ctx.active_model, ctx.active_context_mode = roomy.model, "low"
    ctx.tool_schemas, ctx.messages[:] = books.SCHEMAS, roomy.messages_for("low")
    superseded = {"role": "assistant", "content": "A superseded host observation. " * 30}
    if ordinary_first:
        ctx.messages.append(superseded)
    ctx.context_fit_plan = ctx.tools._ctx.context_fit_plan = roomy
    assert append_context_facts(ctx, money={}) and books._rows(roomy) == []

    def exact_count(messages, tools, *, offset=0):
        payload = _physical_candidate(dict(messages=messages, tools=tools,
                                           model=roomy.model, max_tokens=books.RESERVE))
        text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        return payload, len(re.findall(r"[a-zA-Z]+|[0-9]|[^a-zA-Z0-9\s]", text)) + offset

    _, initial_exact = exact_count(ctx.messages, ctx.tool_schemas)
    stepped_probe = memory_floor.step_book_map(roomy, "low")
    assert stepped_probe is not None
    stepped_msgs = stepped_probe.reproject_transcript(
        [m for m in ctx.messages if m is not superseded], "low"
    )
    _, stepped_exact = exact_count(stepped_msgs, ctx.tool_schemas)
    exact_saving = initial_exact - stepped_exact
    assert exact_saving > 500
    # Window holds the chars/4 estimate with start room (so build keeps body_map whole), while
    # the local tokenizer's exact count overflows by less than exact_saving.
    est_with_tools = roomy.projected_tokens_with_tools("low", books.SCHEMAS)
    min_unstepped_window = math.ceil((est_with_tools + books.RESERVE) * 8 / 7) + 2_000
    window = max(min_unstepped_window, stepped_exact + books.RESERVE + exact_saving // 2)
    density_offset = max(0, window - books.RESERVE - stepped_exact - exact_saving // 2)
    plan = books._plan(tmp_path, core, "low", window, source="local_health")
    assert books._rows(plan) == []
    ctx.context_fit_plan = ctx.tools._ctx.context_fit_plan = plan
    measured = loop._measure_round_main_fit(ctx, automatic_pass_used=False)

    def request(candidate):
        payload, count = exact_count(candidate.messages, candidate.tool_schemas, offset=density_offset)
        native = dict(supported=True, input_is_exact=True, native_input_sha256=input_fingerprint(payload),
                      input_tokens=count, context_window=window)
        with ua.bind_physical_attempt_context(_physical_context_for_fit(measured, plan)):
            candidate_request = _attempt_request({"provider": "local", "resolved_model": plan.model,
                "base_url": "http://127.0.0.1:1", "local_input_measurement": native}, payload)
        assert candidate_request.prompt_tokens_exact == count
        return candidate_request

    before = request(ctx)
    refusal = loop_model_call._refused_candidate(main_context_fit_refusal(before))
    target, slack = loop_model_call._refusal_target(ctx, measured, refusal)
    assert target - slack > 0
    sends = []
    original_pass = loop_model_call._apply_refusal_pass

    def apply_pass(c, steps, *args, **kwargs):
        if ordinary_first and steps == loop_model_call.REFUSAL_PASSES[0]:
            c.messages.remove(superseded)
            return True
        return original_pass(c, steps, *args, **kwargs)

    def dispatch(c, disposition, *, candidate_predicate=None, **kwargs):
        loop.seal_task_transcript(c.messages)
        append_context_facts(c, money={})
        maps = books._rows(c.context_fit_plan)
        sends.append(maps)
        candidate = request(c)
        assert candidate_predicate(candidate)
        answer = None
        with _presend_context_fit(c) as admit:
            if not admit(candidate):
                raise ua.PhysicalAttemptPreconditionFailed("exact local size refusal", attempt_id=str(len(sends)))
            answer = {"role": "assistant", "content": "The exact request fits."}
        return answer, 0.0

    monkeypatch.setattr(loop_model_call, "_apply_refusal_pass", apply_pass)
    monkeypatch.setattr(loop, "_dispatch_round_model", dispatch)
    answer, cost, mode = loop_model_call._recover_context_overflow(ctx, refusal, 0.0)
    assert answer and mode == "low" and cost == 0
    expected = ([[]] if ordinary_first else []) + [["body_map"]]
    assert sends == expected  # a distinct exact refusal precedes each further map step
    assert books._rows(ctx.context_fit_plan) == expected[-1]
    assert "## Body (parts & addresses)" in ctx.messages[0]["content"][0]["text"]
