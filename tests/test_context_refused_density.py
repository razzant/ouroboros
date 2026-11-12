"""A size refusal's own density measures that refusal's recovery, once, through the round measurement (#2232).

A host at cold density 1.0 sized a stated 15k excess while its floor believed the window still had
56k of room, so no rung that only physical pressure moves could answer and recovery ended at turn 0.
The provider's stated input of the refused attempt over the estimate of the bytes that same attempt
submitted measures that request. While its passes, and a renewal after them, size that refusal, the round
measurement counts at that density (floor, books, address sizing, renewal); the retry it produces is
measured as an ordinary request, and nothing keeps the density beyond that refusal.
"""
from __future__ import annotations

import math
from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest

from ouroboros import capability_evidence as ce
from ouroboros import context_fit, loop, loop_round_limits
from ouroboros import loop_model_call as lmc
from ouroboros import usage_accounting as ua
from ouroboros.context_anchor import measured_units, refused_request_density
from ouroboros.llm_attempt import _attempt_request
from ouroboros.loop_transport import stamp_owner_provider_message
from ouroboros.physical_attempt_context import PhysicalAttemptContext
from tests.test_context_fit_integration import _plan

LIMIT, REPLY, DENSITY = 240_000, 65_536, 1.31


@pytest.fixture(autouse=True)
def _own_evidence_store(tmp_path, monkeypatch):
    monkeypatch.setattr(ce, "canonical_evidence_root", lambda: tmp_path)  # stated limits stay in this test's store
    monkeypatch.setattr(ce, "_DENSITY_MEMO", {})


def _stated(input_tokens, reply=REPLY, limit=LIMIT):
    return (f"Requested token count exceeds the model's maximum context length of {limit} tokens. You requested a "
            f"total of {input_tokens + reply} tokens: {input_tokens} tokens from the input messages and {reply} "
            "tokens for the completion.")


def _sent(route="route-a", estimate=145_078, **extra):
    context = SimpleNamespace(route_fp=route)
    return SimpleNamespace(prompt_tokens_bounded_estimate=estimate, physical_context=context, **extra)


def _measurement(route="route-a", density=1.0):
    return SimpleNamespace(route_fp=route, measurement_density=density, estimated_input_tokens=145_078,
                           anchor_input_tokens=None, accepted_input_tokens=None, capacity_total_tokens=LIMIT,
                           window_reading_tokens=None, reply_allowance_tokens=REPLY, capacity_deficit_tokens=0)


def _real_plan(*, preferred="low", window=LIMIT, book_chars=0):
    """The integration plan with a real captured core; ``book_chars`` makes this task directly require a handbook."""
    plan = _plan(preferred=preferred, window=window)
    core = context_fit.ContextCore("", "", "", "x" * book_chars, "", "", plan.user_content_json, False,
                                   development_required=bool(book_chars))
    return replace(plan, core=core)


def _round(tmp_path, plan, *, mode="low", words=0):
    """A real round-call context on ``plan``: its own measurement, usage and transcript (``words`` of owner text)."""
    tool_ctx = SimpleNamespace(messages=None, active_context_mode=mode, context_fit_plan=plan, task_metadata={},
                               _context_overflow_retries=set(), _execution_trace={})
    messages = [*plan.messages_for(mode), *([{"role": "user", "content": "word " * words}] if words else [])]
    ctx = loop._RoundModelCallContext(
        llm=SimpleNamespace(), messages=messages, tools=SimpleNamespace(_ctx=tool_ctx), context_fit_plan=plan,
        active_model=plan.model, tool_schemas=[], active_effort="high", max_retries=1, drive_logs=tmp_path / "logs",
        task_id="t", round_idx=1, event_queue=None, accumulated_usage={"execution_id": "e"}, task_type="task",
        active_use_local=False, active_context_mode=mode, drive_root=tmp_path)
    tool_ctx.messages = ctx.messages
    (tmp_path / "logs").mkdir(exist_ok=True)
    return ctx


def _refused_capture(estimate, *, route="route-a"):
    context = PhysicalAttemptContext(profile="owner_low", rendered_mode="low", measurement_basis="cold_estimate",
                                     route_fp=route, round_id="e:round:1", target_total_tokens=None,
                                     capacity_total_tokens=LIMIT, context_target_miss=False, automatic_pass_used=False)
    return ua.PhysicalAttemptCapture(attempt_id="refused", model="openai/test-model", provider="openai",
                                     state="unresolved", candidate_measurement_kind="canonical_json_v1",
                                     max_completion_tokens=REPLY, candidate_raw_sha256="old",
                                     candidate_raw_size_bytes=1_100, candidate_context_sha256="old-context",
                                     candidate_context_size_bytes=1_000, physical_context=context,
                                     prompt_tokens_bounded_estimate=estimate)


def _ctx(message):
    return SimpleNamespace(accumulated_usage={"_last_llm_provider_message": message}, active_context_mode="low",
                           context_fit_plan=SimpleNamespace(provider="claudexor", model="codex=gpt-6-astra"))


def test_only_a_stated_input_of_the_same_attempt_measures_its_density():
    usage, fit = {"_last_llm_provider_message": _stated(190_053)}, _measurement()
    assert refused_request_density(usage, _sent(), fit) == 190_053 / 145_078
    total_only = {"_last_llm_provider_message":
                  f"maximum context length is {LIMIT} tokens. You requested a total of 255589 tokens."}
    for words, refused, measured in (
            (total_only, _sent(), fit),  # a bare total holds a reply: no input of this request
            ({"_last_llm_provider_message": "context_length_exceeded"}, _sent(), fit),  # unnumbered
            ({}, _sent(), fit),  # a Claudexor refusal states no host-parsed numbers
            (usage, _sent(estimate=0), fit),  # an exact local count outranks the estimate, or none was submitted
            (usage, SimpleNamespace(predicted_free_tokens=-15_589, predicted_slack_tokens=30_000), fit),  # pre-send
            (usage, _sent(route="route-b"), fit),  # refused on another account's route, then rebound
            (usage, _sent(), _measurement(route="route-b"))):  # the round now measures another route
        assert refused_request_density(words, refused, measured) is None
    assert measured_units(None, object(), 45_589, 30_000, None) == (45_589, 30_000, None)  # no density: untouched


def test_the_client_capture_pairs_the_refusal_with_the_estimate_it_submitted(tmp_path, monkeypatch):
    """client → capture → refusal → the floor's reprojection: the provider's excess arrives once, in its units."""
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path / "data"))
    (tmp_path / "data" / "state").mkdir(parents=True)
    ua._reset_task_cache_splits()
    context = PhysicalAttemptContext(profile="owner_low", rendered_mode="low", measurement_basis="cold_estimate",
                                     route_fp="route-a", round_id="e:round:1", target_total_tokens=None,
                                     capacity_total_tokens=LIMIT, context_target_miss=False,
                                     automatic_pass_used=False, measurement_density=1.0)
    body = {"model": "acme/flint-240k", "max_tokens": REPLY, "tools": [],
            "messages": [{"role": "system", "content": "core " * 116_000}, {"role": "user", "content": "go " * 900}]}
    with ua.bind_physical_attempt_context(context):
        request = replace(_attempt_request({"provider": "openrouter", "model": "acme/flint-240k"}, body),
                          drive_root=tmp_path / "data", task_id="t", root_task_id="t", reservation_usd=1.0)
    estimate = request.prompt_tokens_bounded_estimate
    stated = math.ceil(estimate * DENSITY)

    def wire():
        raise httpx.HTTPStatusError("context_length_exceeded", request=httpx.Request("POST", "https://fixture.invalid"),
            response=httpx.Response(400, json={"error": {"code": "context_length_exceeded",
                                                         "message": _stated(stated)}}))

    with pytest.raises(httpx.HTTPStatusError) as refusal:
        ua.execute_physical_attempt(request, wire)
    capture = refusal.value.physical_attempt_capture
    assert capture.prompt_tokens_bounded_estimate == estimate and capture.physical_context.route_fp == "route-a"
    with pytest.raises(httpx.HTTPStatusError) as local:  # the same bytes counted exactly by a local serving instance
        ua.execute_physical_attempt(replace(request, prompt_tokens_exact=estimate), wire)
    assert local.value.physical_attempt_capture.prompt_tokens_bounded_estimate == 0  # the exact count outranks it
    ctx = _ctx("")
    stamp_owner_provider_message(ctx.accumulated_usage, refusal.value)  # the loop's own stamp of this failure
    target, slack = lmc._refusal_target(ctx, SimpleNamespace(measurement=_measurement()), capture)
    excess = stated + REPLY - LIMIT
    assert excess > 0 and (target, slack) == (excess + LIMIT // 8, LIMIT // 8)  # the provider's own units
    assert ctx.refused_density == stated / estimate
    plan = _real_plan()
    at_refusal = context_fit.measure_main_fit(plan, plan.messages_for("low"), [], profile="owner_low",
                                             rendered_mode="low", round_id="r", refused_density=ctx.refused_density)
    seen = []
    monkeypatch.setattr(context_fit.ContextFitPlan, "reproject_for_route", lambda self, **kw: seen.append(kw) or self)
    floor_ctx = SimpleNamespace(context_fit_plan=plan, active_context_mode="low", messages=plan.messages_for("low"),
                                tool_schemas=[])
    assert not loop_round_limits._coarsen_memory_view(floor_ctx, at_refusal, saving=max(1, target - slack))
    assert (seen[0]["ratio"], seen[0]["refused_tokens"]) == (stated / estimate, excess)  # no second conversion
    # A pre-send prediction of the next candidate is no provider count: it resets the density.
    lmc._refusal_target(ctx, SimpleNamespace(measurement=_measurement()),
                        SimpleNamespace(predicted_free_tokens=-100, predicted_slack_tokens=30_000))
    assert ctx.refused_density is None


def test_the_round_measurement_counts_at_the_refusal_density_only_while_set(tmp_path):
    plan = _real_plan()
    ctx = _round(tmp_path, plan, words=40_000)
    ordinary = loop._measure_round_main_fit(ctx, automatic_pass_used=False).measurement
    assert (ordinary.measurement_density, ordinary.measurement_basis) == (1.0, "cold_estimate")
    ctx.refused_density = DENSITY
    refused = loop._measure_round_main_fit(ctx, automatic_pass_used=True).measurement
    assert (refused.measurement_density, refused.measurement_basis) == (DENSITY, "refused_input")
    assert refused.estimated_input_tokens == math.ceil(ordinary.raw_input_tokens * DENSITY)
    assert ctx.accumulated_usage["_context_measurement_density"] == DENSITY  # what a renewal after the passes reads
    estimator = replace(plan, window_basis="estimator")  # a window declared in estimator units keeps them
    assert context_fit.measure_main_fit(estimator, ctx.messages, [], profile="owner_low", rendered_mode="low",
                                        round_id="r", refused_density=DENSITY).measurement.measurement_density == 1.0


@pytest.mark.parametrize("density", [None, DENSITY])
def test_books_to_maps_reprojects_at_the_refusal_density(tmp_path, monkeypatch, density):
    ctx = _round(tmp_path, _real_plan(preferred="max"), mode="max", words=40_000)
    ctx.refused_density = density
    seen = []
    monkeypatch.setattr(context_fit.ContextFitPlan, "reproject_for_route", lambda self, **kw: seen.append(kw) or self)
    lmc._reproject_actual_overflow_low(ctx)
    assert [(kw["ratio"], kw["select_mode"]) for kw in seen] == [(density or 1.0, "low")]
    assert seen[0]["measured_input_tokens"] is None  # no accepted preparation offset on this measurement


@pytest.mark.parametrize(("image_mode", "mode"), [("off", "low"), ("caption", "max")])
def test_recovery_prices_the_prepared_input_not_the_canonical_image_proxy(tmp_path, monkeypatch, mode, image_mode):
    """Real image preparation, then 500 tokens of excess: both recovery consumers keep Low.

    A ratio-only recovery after Off preparation saw a phantom 21,870-token excess and
    changed Low to Nano. Caption expansion has the opposite sign; both belong in the floor.
    The provider count is controlled, not a new model call or tokenizer qualification.
    """
    from ouroboros import memory_view, vision_routing
    from ouroboros.context_anchor import InputAnchor
    from tests import _memory_view_synthetic as syn

    monkeypatch.setattr(context_fit, "_route_calibration_ratio", lambda *_a, **_kw: 1.0)
    monkeypatch.setattr(loop, "_emit_checkpoint_event", lambda *_a, **_kw: None)
    monkeypatch.setattr(vision_routing, "get_image_input_mode", lambda: image_mode)
    monkeypatch.setattr(vision_routing, "resolve_vision_caption_model", lambda *_a, **_kw: "fixture/vision")
    client = SimpleNamespace(vision_query=lambda *_a, **_kw: ("caption " * 7500, {}))
    room = syn.room(notes=1)
    room["notes"][0]["text"] = "Held note. " + "n" * 8000
    core = context_fit.ContextCore("Core", "BIBLE", "", "", "Identity", "Facts", '"Assignment"', False,
        memory_view_json=memory_view.snapshot_json(syn.snapshot(room_facts=room, story=[], conversations=[])))
    plan = context_fit.build_context_fit_plan(None, core, {"id": "recovery", "_is_direct_chat": True},
        preferred_mode=mode, tool_schemas=[], route_resolver=lambda *_a, **_kw: (
            {"provider": "fixture", "model": "fixture"},
            SimpleNamespace(route_fp="", status="confirmed", stale=False, window_tokens=1_000_000)))
    routing = vision_routing.VisionRoutingContext(model="openai::fixture", llm=client, accumulated_usage={}, drive_root=tmp_path)
    prior = [*plan.messages_for(mode), {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,eA=="}}
        for _ in range(20 if image_mode == "off" else 1)]}]
    growth = {"role": "user", "content": "x" * 4000}
    def estimate(rows):
        return context_fit.estimate_context_prompt_tokens(rows, [])

    def prepare(rows):
        return vision_routing.prepare_messages_for_send(rows, routing=routing)
    window = 131_072 if image_mode == "off" else 262_144
    prior[1]["content"] += "p" * (4 * (window - REPLY + 500 - estimate(prepare([*prior, growth]))))
    raw, prepared = estimate(prior), estimate(prepare(prior))
    anchor = InputAnchor(input_tokens=prepared, estimate_tokens=prepared, transcript_tokens=raw, round=1)
    assert (prepared < raw) is (image_mode == "off")
    plan = replace(plan, window_tokens=window, window_record=None)
    ctx = _round(tmp_path, plan, mode=mode)
    ctx.messages[:] = [*prior, growth]

    def measure(c, **_kw):
        return context_fit.measure_main_fit(c.context_fit_plan, c.messages, [], profile="owner_" + c.active_context_mode,
            rendered_mode=c.active_context_mode, round_id="recovery:2", anchor=anchor)

    monkeypatch.setattr(loop, "_measure_round_main_fit", measure)
    fit = measure(ctx)
    deficit = fit.measurement.capacity_deficit_tokens
    assert estimate(prepare(ctx.messages)) + REPLY - window == 500  # actual prepared bytes
    assert 499 <= deficit <= 501  # carrying rounded prior counts may differ by one estimator token, either way
    if mode == "low":
        slack = context_fit.reclaim_low_water_margin(None, window)
        lmc._apply_refusal_step(ctx, "floor", fit, target=deficit + slack, slack=slack)
    else:
        lmc._reproject_actual_overflow_low(ctx)
    assert ctx.active_context_mode == "low" and ctx.context_fit_plan.window_tokens == window
    assert measure(ctx).measurement.response_reserve_tokens == REPLY
    assert not measure(ctx).predicted_capacity_miss


@pytest.mark.parametrize("basis", ["provider", "estimator", "refused"])
def test_recovery_offset_uses_the_effective_measurement_basis(basis):
    from ouroboros.context_anchor import InputAnchor

    plan = replace(_real_plan(), window_basis="estimator" if basis == "estimator" else "")
    anchor = InputAnchor(input_tokens=120_000, estimate_tokens=100_000, transcript_tokens=80_000, round=1)
    fit = context_fit.measure_main_fit(plan, plan.messages_for("low"), [], profile="owner_low", rendered_mode="low",
        round_id="basis", anchor=anchor, refused_density=1.5 if basis == "refused" else None).measurement
    if basis == "provider":
        assert fit.reprojection_input_tokens == math.ceil(fit.estimated_input_tokens / 1.2)
    else:
        assert fit.reprojection_input_tokens is None
        assert fit.measurement_density == (1.0 if basis == "estimator" else 1.5)


@pytest.mark.parametrize("density", [None, DENSITY])
def test_address_sizing_measures_candidates_at_the_refusal_density(tmp_path, monkeypatch, density):
    from ouroboros import context_source_view

    ctx = _round(tmp_path, _real_plan(), words=40_000)
    ctx.refused_density = density
    fit = loop._measure_round_main_fit(ctx, automatic_pass_used=True)
    seen = {}

    def view(messages, request, *, measure_candidate, usable_input, **_kw):
        seen.update(density=request.measurement_density, usable=usable_input, candidate=measure_candidate(messages))
        return messages, SimpleNamespace(status="no_eligible", reclaimed_tokens=0, checkpoint_ref=None,
                                         selected_unit_ids=())

    monkeypatch.setattr(context_source_view, "emergency_address_view", view)
    loop_round_limits._run_emergency_address_pass(ctx, fit, rung="unseen_bodies", goal_tokens=10_000)
    assert seen["density"] == (density or 1.0) and seen["usable"] == LIMIT - REPLY  # the window, in provider tokens
    assert seen["candidate"] == math.ceil(fit.measurement.raw_input_tokens * (density or 1.0))


@pytest.mark.parametrize("window,density,book_blamed", [
    (180_536, None, True),  # cold: the ordinary view seems to fit, so the handbook is blamed for the excess
    (180_536, DENSITY, False),  # at the refused request's density the ordinary view itself does not fit
    (200_536, DENSITY, True),  # where it does fit at that density, the excess is the handbook's alone: typed
])
def test_a_renewal_after_the_passes_sizes_at_the_refusal_density(tmp_path, monkeypatch, window, density, book_blamed):
    from ouroboros.context_budget import REQUIRED_BOOK_UNFIT_KEY
    from ouroboros.context_renewal import renew_after_refusal

    ctx = _round(tmp_path, _real_plan(window=window, book_chars=80_000), words=96_000)
    raw = loop._measure_round_main_fit(ctx, automatic_pass_used=False).measurement.raw_input_tokens
    ctx.accumulated_usage["_last_llm_provider_message"] = _stated(math.ceil(raw * DENSITY), limit=window)
    monkeypatch.setattr(lmc, "_apply_refusal_pass", lambda *_a, **_kw: False)  # passes that cannot change anything
    capture = _refused_capture(raw if density else 0)
    assert lmc._recover_context_overflow(ctx, capture, 0.0)[0] is None
    ctx.accumulated_usage[lmc.CONTEXT_REFUSAL_ROUND_KEY] = ctx.round_idx
    limit_ctx = SimpleNamespace(accumulated_usage=ctx.accumulated_usage, round_idx=1, messages=ctx.messages,
                                drive_root=tmp_path, drive_logs=tmp_path / "logs", task_id="t", tool_schemas=[],
                                event_queue=None, context_fit_plan=ctx.context_fit_plan)
    assert renew_after_refusal(limit_ctx, ctx.tools) is False  # no tool rows a renewal could move here
    assert (REQUIRED_BOOK_UNFIT_KEY in ctx.accumulated_usage) is book_blamed
    assert ctx.accumulated_usage["_context_measurement_density"] == (math.ceil(raw * DENSITY) / raw if density else 1.0)


def test_the_retry_is_measured_ordinarily_while_its_passes_measured_at_the_refusal(tmp_path, monkeypatch):
    ctx = _round(tmp_path, _real_plan(), words=40_000)
    raw = loop._measure_round_main_fit(ctx, automatic_pass_used=False).measurement.raw_input_tokens
    ctx.accumulated_usage["_last_llm_provider_message"] = _stated(math.ceil(raw * DENSITY))
    passes, sends = [], []
    monkeypatch.setattr(lmc, "_apply_refusal_pass", lambda c, steps, fit, *_a, **_kw: passes.append(
        fit.measurement.measurement_density) or True)
    monkeypatch.setattr(loop, "_dispatch_round_model", lambda c, disposition, **_kw: sends.append(
        disposition.measurement.measurement_density) or ({"role": "assistant", "content": "fits"}, 0.0))
    assert lmc._recover_context_overflow(ctx, _refused_capture(raw), 0.0)[0]["content"] == "fits"
    assert passes == [math.ceil(raw * DENSITY) / raw] and sends == [1.0] and ctx.refused_density is None


def test_a_changed_view_that_was_never_sent_is_not_the_refused_request(tmp_path, monkeypatch):
    from ouroboros.usage_accounting import PhysicalAttemptPreconditionFailed
    from tests.test_loop_compaction import _ctx as _round_ctx
    from tests.test_loop_compaction import _failed_capture, _measure_cycle
    from tests.test_loop_compaction import _fit as _round_fit

    context = _round_ctx(tmp_path)
    context.accumulated_usage["_last_llm_provider_message"] = _stated(524_000, limit=500_000)
    failed = replace(_failed_capture(size=1_000), prompt_tokens_bounded_estimate=400_000)
    monkeypatch.setattr(loop, "_measure_round_main_fit", _measure_cycle([_round_fit()]))  # an ordinary 1.0 measure
    seen = []
    monkeypatch.setattr(lmc, "_apply_refusal_pass", lambda ctx, steps, fit, target, slack=None, refused=False, **_kw:
                        seen.append((steps, ctx.refused_density, target, slack)) or True)

    def unsent(*_args, **_kwargs):
        raise PhysicalAttemptPreconditionFailed("the changed candidate is not strictly smaller")
    monkeypatch.setattr(loop, "_dispatch_round_model", unsent)
    assert lmc._recover_context_overflow(context, failed, 0.0)[0] is None
    target, slack = 524_000 + REPLY - 500_000 + 62_500, 62_500  # the provider's units while its density measures
    density = 524_000 / 400_000
    assert seen == [(("host_copies", "bodies", "floor"), density, target, slack),
                    (("books", "unseen_bodies"), None, *(math.ceil(n / density) for n in (target, slack)))]


def test_a_reused_call_context_starts_ordinary(tmp_path, monkeypatch):
    """A deferred candidate's context can be called again; a density its earlier recovery left prices nothing new."""
    from tests.test_loop_compaction import _ctx as _round_ctx
    from tests.test_loop_compaction import _fit as _round_fit

    context = _round_ctx(tmp_path)
    context.refused_density = DENSITY  # left by a recovery that ended without sending
    seen = []
    monkeypatch.setattr(loop, "_measure_round_main_fit",
                        lambda ctx, **_kw: seen.append(ctx.refused_density) or _round_fit())
    monkeypatch.setattr(loop, "_dispatch_round_model",
                        lambda c, d, **_kw: ({"role": "assistant", "content": "ok"}, 0.0))
    assert loop._call_round_model(context)[0]["content"] == "ok" and seen == [None]


def test_refusal_measurement_keeps_its_source_label_in_retained_capsules(tmp_path):
    from ouroboros import context_compaction as cc
    from tests.test_compaction import _authored, _fold, _large_unit

    messages = _large_unit("refused-source")
    request = replace(_authored(messages), measurement_basis="refused_input", measurement_density=DENSITY)
    folded, receipt = _fold(messages, request, tmp_path)
    assert receipt.status == "applied"
    capsule = folded[0]["content"][0]["_context_capsule"]
    assert capsule["measurement_basis"] == "refused_input"
    assert len(cc._atomic_units(folded)) == 1  # the real reader still accepts the retained capsule
    from dataclasses import asdict
    from ouroboros.tools.vision_process import _read_receipt, _restore_capture
    from ouroboros.usage_ledger import _validate_candidate_facts

    captured = replace(_refused_capture(100), candidate_raw_sha256="a" * 64,
                       candidate_context_sha256="b" * 64)
    captured = replace(captured, physical_context=replace(captured.physical_context,
                       measurement_basis="refused_input", measurement_density=DENSITY))
    raw = asdict(captured)
    _validate_candidate_facts(raw, 1)
    receipt = _read_receipt({"receipt_id": "same", "capture": raw, "custody": None}, "same")
    assert _restore_capture(receipt["capture"]) == captured
