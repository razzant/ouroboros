"""Known-window pressure reaches the real Main dispatch and its existing recovery."""
from dataclasses import replace
import json

import pytest

from ouroboros import loop, loop_model_call
from ouroboros.context_anchor import REFUSED_INPUTS_KEY
from tests.test_llm_claudexor import setup as gateway_fixture
from tests.test_model_wait import live_wait as wait_fixture
from tests.test_subscription_main_wait import main_call as main_fixture
from tests._usage_store_testing import dispatched_attempts

setup, live_wait, main_call = gateway_fixture, wait_fixture, main_fixture


def _large(ctx, *, known=True, stale=False, irreducible=False):
    plan = ctx.context_fit_plan
    large = "Exact source text. " * 40_000
    plan = replace(plan, window_tokens=200_000, status="confirmed" if known else "unprobeable", stale=stale,
                   max_projection=replace(plan.max_projection, system_content_json=json.dumps(large)))
    ctx.context_fit_plan = ctx.tools._ctx.context_fit_plan = plan
    ctx.messages[:] = plan.messages_for("max")
    if irreducible:
        ctx.messages[1]["content"] = large
    return large


def test_known_window_folds_before_first_provider_send(main_call, monkeypatch):
    from ouroboros import capability_evidence, context_anchor

    ctx, gateway, *_ = main_call
    large = _large(ctx)
    monkeypatch.setattr(context_anchor, "record_refused_input",
                        lambda *_a, **_kw: pytest.fail("predictions cannot teach provider refusals"))
    evidence = capability_evidence._load(ctx.drive_root).get("probes", {})
    answer, _, mode = loop._call_round_model(ctx)
    assert answer and mode == "low"
    assert len(gateway.creates) == 1
    assert large not in json.dumps(gateway.uploads[0][0])
    assert len(dispatched_attempts(ctx.drive_root)) == 1
    assert REFUSED_INPUTS_KEY not in ctx.accumulated_usage  # prediction is not a provider observation
    assert ctx.context_fit_plan.preferred_mode == "max"
    assert ctx.context_fit_plan.window_tokens == 200_000 and ctx.context_fit_plan.status == "confirmed"
    assert capability_evidence._load(ctx.drive_root).get("probes", {}) == evidence
    assert "_last_llm_error" not in ctx.accumulated_usage
    rows = [json.loads(line) for line in (ctx.drive_logs / "events.jsonl").read_text().splitlines()]
    [prediction] = [row for row in rows if row.get("checkpoint_kind") == "context_fit_refused"]
    assert prediction["cause"] == "known_window_prediction"
    assert prediction["capacity_total_tokens"] == 200_000
    assert prediction["response_reserve_tokens"] == 65_536
    assert prediction["estimated_input_tokens"] > 150_000
    assert prediction["predicted_free_tokens"] == (
        prediction["capacity_total_tokens"] - prediction["response_reserve_tokens"] - prediction["estimated_input_tokens"])
    assert prediction["physical_context"]["round_id"]
    assert prediction["candidate_context_sha256"] and "provider_message" not in prediction
    assert not any(row.get("type") in {"llm_api_error", "remote_context_overflow"} for row in rows)


@pytest.mark.parametrize("known,stale", [(False, False), (True, True)])
def test_unknown_or_stale_window_sends_the_first_full_request(main_call, known, stale):
    ctx, gateway, *_ = main_call
    large = _large(ctx, known=known, stale=stale)
    answer, _, mode = loop._call_round_model(ctx)
    assert answer and mode == "max" and len(gateway.creates) == 1
    assert large in json.dumps(gateway.uploads[0][0])
    assert not getattr(ctx.tools._ctx, "_context_overflow_retries", set())


def test_irreducible_prediction_exhausts_passes_without_a_paid_blind_round(main_call):
    ctx, gateway, *_ = main_call
    _large(ctx, irreducible=True)
    answer, cost, _ = loop._call_round_model(ctx)
    assert answer is None and cost == 0 and not gateway.creates
    assert ctx.accumulated_usage[loop_model_call.CONTEXT_REFUSAL_ROUND_KEY] == ctx.round_idx
    assert ctx.accumulated_usage["_last_llm_error_kind"] == "context_overflow"
    assert "pre-send" in ctx.accumulated_usage["_last_llm_error"]
    done = set(loop._context_overflow_retries(ctx.tools._ctx))
    assert len(done) == 2
    assert loop._call_round_model(ctx)[0] is None
    assert set(loop._context_overflow_retries(ctx.tools._ctx)) == done and not gateway.creates
    assert not dispatched_attempts(ctx.drive_root)
    assert REFUSED_INPUTS_KEY not in ctx.accumulated_usage


def test_a_claimed_change_still_needs_the_existing_strict_shrink_check(main_call, monkeypatch):
    ctx, gateway, *_ = main_call
    _large(ctx)
    monkeypatch.setattr(loop_model_call, "_apply_refusal_pass", lambda *a: True)
    assert loop._call_round_model(ctx)[0] is None
    assert not gateway.creates
    skipped = [event.get("data", {}) for event in list(ctx.event_queue.queue)]
    assert len([row for row in skipped if row.get("reason") == "context_candidate_not_strictly_smaller"]) == 2


def test_positive_result_room_sends_without_recovery(main_call):
    ctx, gateway, *_ = main_call
    assert loop._call_round_model(ctx)[0]
    assert len(gateway.creates) == 1
    assert not loop._context_overflow_retries(ctx.tools._ctx)


@pytest.mark.parametrize("known,caption_repeats", [(True, 600), (True, 5), (False, 600)])
def test_caption_admission_measures_the_final_candidate(main_call, monkeypatch, known, caption_repeats):
    from ouroboros import vision_routing
    from ouroboros.context_fit import bounded_prompt_tokens_for_payload

    ctx, gateway, *_ = main_call
    monkeypatch.setattr(vision_routing, "get_image_input_mode", lambda: "caption")
    monkeypatch.setattr(vision_routing, "resolve_vision_caption_model", lambda *a, **kw: "fixture/vision")
    caption = "The diagram shows a labelled connection. " * caption_repeats
    monkeypatch.setattr(ctx.llm, "vision_query",
                        lambda *a, **kw: (caption, {"prompt_tokens": 2, "completion_tokens": 6000, "cost": 0.01}))
    ctx.messages[-1]["content"] = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,eA=="}}]
    before = loop._measure_round_main_fit(ctx, automatic_pass_used=False).measurement
    capacity = before.estimated_input_tokens + before.response_reserve_tokens + 1000
    plan = replace(ctx.context_fit_plan, window_tokens=capacity, status="confirmed" if known else "unprobeable")
    ctx.context_fit_plan = ctx.tools._ctx.context_fit_plan = plan
    fit = loop._measure_round_main_fit(ctx, automatic_pass_used=False)
    # Dispatch appends its facts once; admission must use the already prepared request's estimate.
    monkeypatch.setattr(loop, "_measure_round_main_fit", lambda *a, **kw: fit)
    answer, _ = loop._dispatch_round_model(ctx, fit, attempt_cap=1)
    if known and caption_repeats == 600:
        assert answer is None and not gateway.creates
        refused = ctx.accumulated_usage[loop_model_call.REFUSED_CANDIDATE_KEY]
        assert refused["predicted_free_tokens"] < -3000
    else:
        assert answer and len(gateway.creates) == 1
        sent = gateway.uploads[0][0]
        assert caption.strip() in str(sent)
        if known:
            assert bounded_prompt_tokens_for_payload(sent, 0) + before.response_reserve_tokens < capacity
    assert REFUSED_INPUTS_KEY not in ctx.accumulated_usage


@pytest.mark.parametrize("large,known", [(True, True), (False, True), (True, False)])
def test_physical_fit_survives_transient_retry_and_account_reprepare(main_call, monkeypatch, large, known):
    from ouroboros import context, loop_llm_call
    from tests.test_llm_claudexor import result
    from tests.test_subscription_main_wait import ROUTE_B, _failed

    ctx, gateway, *_ = main_call
    ctx.messages.append({"role": "user", "content": "Read this exact evidence. " * (40000 if large else 100)})
    route = context._context_fit_route
    if not known:
        def unknown(*a, **kw):
            target, evidence = route(*a, **kw)
            evidence.status, evidence.window_tokens = "unprobeable", 0
            return target, evidence
        monkeypatch.setattr(context, "_context_fit_route", unknown)
    monkeypatch.setattr(loop_llm_call, "_sleep_within_deadline", lambda *a, **kw: True)
    gateway.results = [result(outcome="failed", problem={
        "code": "provider_failed", "message": "fixture transient", "retryable": True,
        "context": {"httpStatus": 429, "vendorCode": "rate_limit_exceeded"}}),
        _failed("invalid_continuation", ROUTE_B), result(route=ROUTE_B)]
    gateway.dispatch = ["not_started", "not_started", "response_received"]
    answer, _, _ = loop._call_round_model(ctx)
    if large and known:
        assert answer is None and len(gateway.creates) == 2  # the repaired oversized send never leaves
        assert loop._context_overflow_retries(ctx.tools._ctx)
    else:
        assert answer and len(gateway.creates) == 3
        assert not loop._context_overflow_retries(ctx.tools._ctx)
    assert REFUSED_INPUTS_KEY not in ctx.accumulated_usage


@pytest.mark.parametrize("predecessor", ["none", "memo", "persisted"])
def test_new_size_density_witness_reaches_real_admission(main_call, monkeypatch, predecessor):
    from ouroboros import capability_evidence as ce
    from tests.test_llm_claudexor import MODEL

    ctx, gateway, *_ = main_call
    monkeypatch.setattr(ce, "_DENSITY_MEMO", {})
    plan = replace(ctx.context_fit_plan, window_tokens=800_000)
    ctx.context_fit_plan = ctx.tools._ctx.context_fit_plan = plan
    ctx.messages[:] = plan.messages_for("max")
    ctx.messages[1]["content"] = "word " * 720_000
    root, model = ce.canonical_evidence_root(), ce._normalized_density_model(MODEL)

    def record(raw):
        ce.record_token_density(root, model, prompt_chars=raw * 4, prompt_tokens=int(raw * 0.8),
                                route_fp=plan.route_fp, basis=ce.MAIN_DENSITY_BASIS)

    if predecessor != "none":
        record(97_000)
    if predecessor == "persisted":
        ce._DENSITY_MEMO.clear()
    record(600_000)
    record(600_000)  # an unchanged comparable sample still throttles
    pairs = ce._load(root)["token_density"][model]["pairs"]
    assert len(pairs) == (1 if predecessor == "none" else 2) <= ce._TOKEN_DENSITY_MAX_PAIRS
    fit = loop._measure_round_main_fit(ctx, automatic_pass_used=False).measurement
    assert fit.measurement_density == 0.8 and 900_000 <= fit.raw_input_tokens < 910_000
    answer, _, _ = loop._call_round_model(ctx)
    assert answer and len(gateway.creates) == 1


@pytest.mark.parametrize("predicted", [True, False])
def test_size_terminal_distinguishes_prediction_from_provider_failure(main_call, predicted):
    from ouroboros.outcomes import derive_loop_outcome
    from tests.test_provider_unknown_terminal_text import _ctx

    ctx, gateway, *_ = main_call
    if predicted:
        _large(ctx, irreducible=True)
        assert loop._call_round_model(ctx)[0] is None and not gateway.creates
        usage = ctx.accumulated_usage
    else:
        usage = {"_last_llm_error_kind": "context_overflow", "reason_code": "llm_api_error",
                 REFUSED_INPUTS_KEY: {"fixture": 300_000}}
    text, usage, trace = loop._handle_provider_unavailable(_ctx(ctx.drive_root, usage), error_kind="context_overflow")
    expected = "context_overflow" if predicted else "llm_api_error"
    assert usage["reason_code"] == expected and usage["_last_llm_error_kind"] == "context_overflow"
    assert "Context-size recovery" in text and "context exceeded" not in text
    assert ("smallest refused request" in text) is (not predicted)
    outcome = derive_loop_outcome(text, usage, trace)
    assert outcome["failure"]["kind"] == ("runtime" if predicted else "provider")
    assert outcome["failure"]["error_kind"] == "context_overflow"


def test_strict_candidate_check_still_expires_after_first_transport_attempt(main_call, monkeypatch):
    from ouroboros import loop_llm_call
    from tests.test_llm_claudexor import result
    from tests.test_subscription_main_wait import ROUTE_B, _failed

    ctx, gateway, *_ = main_call
    monkeypatch.setattr(loop_llm_call, "_sleep_within_deadline", lambda *a, **kw: True)
    gateway.results = [result(outcome="failed", problem={
        "code": "provider_failed", "message": "fixture transient", "retryable": True,
        "context": {"httpStatus": 429, "vendorCode": "rate_limit_exceeded"}}),
        _failed("invalid_continuation", ROUTE_B), result(route=ROUTE_B)]
    gateway.dispatch = ["not_started", "not_started", "response_received"]
    strict_calls = []

    def strict(request):
        strict_calls.append(request)
        return len(strict_calls) == 1

    fit = loop._measure_round_main_fit(ctx, automatic_pass_used=False)
    answer, _ = loop._dispatch_round_model(ctx, fit, attempt_cap=None, candidate_predicate=strict)
    assert answer and len(gateway.creates) == 3 and len(strict_calls) == 1
