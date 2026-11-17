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
import json
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
from ouroboros.loop_transport import provider_terminal_fallback_text, stamp_owner_provider_message
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


def _real_plan(*, preferred="low", window=LIMIT, book_chars=0, body=False):
    """The integration plan with a real captured core; ``book_chars`` makes this task directly require a handbook,
    ``body`` makes the body its subject, whose Max view holds that handbook with no addition."""
    plan = _plan(preferred=preferred, window=window)
    core = context_fit.ContextCore("", "", "", "x" * book_chars, "", "", plan.user_content_json, body,
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
    floor = 8_192  # a stated total no refusal has confirmed admits a request under way up to this short of it
    prior[1]["content"] += "p" * (4 * (window - floor + 500 - estimate(prepare([*prior, growth]))))
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
    assert estimate(prepare(ctx.messages)) + floor - window == 500  # actual prepared bytes
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
    # The line this round's request is admitted at (a stated total less the reply floor), in provider tokens.
    assert seen["density"] == (density or 1.0) and seen["usable"] == LIMIT - 8_192
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


_TERMINAL = dict(is_transport_wait=False, waited_sec=0, is_deadline_exhausted=False)


@pytest.fixture
def round_call(tmp_path, monkeypatch):
    """A real round on its own evidence store, with no clock, deadline, facts row or reasoning retirement to vary."""
    from ouroboros import loop, loop_messages

    monkeypatch.setattr(ce, "canonical_evidence_root", lambda: tmp_path / "evidence")
    monkeypatch.setattr(ce, "_DENSITY_MEMO", {})
    for name, value in (("_emit_checkpoint_event", lambda *_a, **_kw: None), ("_task_deadline_epoch", lambda *_a: None),
                        ("_retire_main_reasoning", lambda *_a, **_kw: None),
                        ("_server_web_allowed_by_task", lambda *_a: False)):
        monkeypatch.setattr(loop, name, value)
    monkeypatch.setattr(loop_messages, "append_context_facts", lambda *_a, **_kw: False)
    return lambda window, book_chars=0, body=False, **kw: _round(tmp_path, _real_plan(
        preferred=kw.get("mode", "low"), window=window, book_chars=book_chars, body=body), **kw)


@pytest.mark.parametrize("exact", [False, True], ids=["known_window_prediction", "exact_local_count"])
def test_a_size_terminal_names_the_final_candidate_it_refused(round_call, monkeypatch, exact):
    """Admission refused the final physical candidate (here the round's own clock line tips it), before any send: the
    terminal names that candidate's size, what counted it and the window it was held to, never the round's earlier
    estimate; a prediction teaches no refused size, an exact local count does."""
    from ouroboros import context_window as cw, loop, loop_model_call as lmc
    from ouroboros.llm_attempt import _attempt_request, main_context_fit_refusal
    from ouroboros.send_clock import SendClockPolicy, main_send_scope, stamp_clock_note
    from ouroboros.usage_accounting import PhysicalAttemptPreconditionFailed, bind_physical_attempt_context

    ctx = round_call(100_000, words=1_000)
    plan = ctx.context_fit_plan
    before = loop._measure_round_main_fit(ctx, automatic_pass_used=False).measurement
    if not exact:  # a local window holding the unstamped candidate exactly: only the real clock line crosses it
        bare = _attempt_request({"provider": "openai", "model": plan.model}, {
            "model": plan.model, "max_tokens": 65_536, "messages": ctx.messages, "tools": []})
        total = max(before.estimated_input_tokens, bare.prompt_tokens_bounded_estimate) + 65_536
        limit = cw.WindowLimit(total, "total", "local_health", ce.utc_now_iso())  # dated, as its read is
        plan = replace(plan, window_tokens=total, window_record=cw.WindowRecord(
            plan.route_fp, total=limit, reserve=65_536, declared=(limit,)))
        ctx.context_fit_plan = ctx.tools._ctx.context_fit_plan = plan
        before = loop._measure_round_main_fit(ctx, automatic_pass_used=False).measurement
    seen = []

    def final_candidate(*_args, **kw):  # the transport boundary: the real admission judges the final candidate
        payload = {"model": plan.model, "max_tokens": 65_536, "messages": [dict(m) for m in ctx.messages], "tools": []}
        with bind_physical_attempt_context(kw["physical_context"]), main_send_scope(None if exact else SendClockPolicy()):
            request = _attempt_request({"provider": "openai", "model": plan.model},
                                       payload if exact else stamp_clock_note(payload))
        if exact:  # the serving instance's own count
            request = replace(request, prompt_tokens_exact=50_000, context_window_tokens_exact=100_000)
        seen.append(main_context_fit_refusal(request))
        assert kw["physical_fit_predicate"](request) is False
        raise PhysicalAttemptPreconditionFailed("final candidate refused before sending")

    monkeypatch.setattr(loop, "call_llm_with_retry", final_candidate)
    monkeypatch.setattr(loop, "last_physical_attempt_capture", lambda: None)
    monkeypatch.setattr(lmc, "_apply_refusal_pass", lambda *_a, **_kw: False)
    assert lmc._call_round_model_once(ctx)[0] is None and len(seen) == 1
    size = seen[0].get("input_tokens", seen[0].get("estimated_input_tokens"))
    assert size != before.estimated_input_tokens
    text = provider_terminal_fallback_text(ctx.accumulated_usage, round_idx=ctx.round_idx, is_context_overflow=True,
                                           **_TERMINAL)
    if exact:
        assert (f"\nrequest input 50,000 tokens (exact local count before sending, no provider response) on "
                f"{plan.model}; the serving instance's window 100,000 total; reply reserve 65,536; "
                "remaining reply room 50,000; requested reply limit 65,536.") in text
    else:
        assert (f"\nrequest input estimated {size:,} tokens (host prediction before sending, no provider response) on "
                f"{plan.model}; assumed window {total:,} total: local serving window ") in text  # then its read's date
    assert f"{before.estimated_input_tokens:,} tokens" not in text
    assert ("smallest refused request ~50,000 tokens" in text) is exact


@pytest.mark.parametrize("primary,fallback", [
    ("size", "size"),  # each route's own pair: the fallback's request, its window and its smallest refused size
    ("size", "numberless_size"),  # the fallback's own sent bytes, never the primary's restored estimate
    ("size", "bad_request"),  # the primary's refusal named apart; nothing restored is set beside the fallback's words
    ("size", "empty"),
    ("uncomparable_size", "bad_request"),  # its recovery returned before any pass: the refusal was already kept whole
    ("transient", "size"),  # a fallback's refusal after the primary failed otherwise stays the round's
    ("size", "prediction"),  # the fallback's own admission: its book, window and reserve, never the restored primary's
    ("prediction", "size"),  # the provider's later refusal replaces the primary's prediction, its book arithmetic too
    ("prediction", "bad_request"),  # the primary's prediction named apart, without book arithmetic outside a size end
    # A Max primary on a body task holds the book with no addition: the walk restores its 0, the refusing Low
    # fallback's own book stays named, with its admission's arithmetic or, refused by the provider, without any.
    ("max_transient", "prediction"), ("max_size", "prediction"), ("max_transient", "size"),
    ("max_size", "bad_request"),  # outside a size end no handbook line, whoever carries the requirement
])
def test_a_failed_walk_names_each_size_refusal_with_its_own_route(round_call, monkeypatch, primary, fallback):
    """The walk restores the primary's facts after a failed candidate; the round's latest size refusal stays whole,
    with its own model, request and window, and the smallest refused size of that model alone. The task's required
    handbook is named with the refusing candidate's own size of it, also where the restored primary's view carries no
    addition, and subtracted only from that candidate's own admission (a prediction before sending), never from
    another route's view. The last failure's row and the acting route's own failure keep their meanings
    for their other readers."""
    from ouroboros import fallback_cooldown, loop, loop_model_call as lmc
    from ouroboros.context_fit import _physical_context_for_fit
    from ouroboros.llm_attempt import refused_candidate_facts
    from ouroboros.loop_llm_call import REFUSED_CANDIDATE_KEY
    from ouroboros.loop_transport import stamp_owner_provider_message

    held, primary = primary.startswith("max_"), primary.removeprefix("max_")
    mode = "max" if held else "low"  # both routes carry the task's required handbook
    ctx = round_call(200_000, book_chars=8_000, body=held, words=4_000, mode=mode)
    primary_plan = ctx.context_fit_plan
    fallback_plan = replace(_real_plan(preferred=mode, window=500_000, book_chars=8_000, body=held),
                            model="anthropic/fallback", route_fp="route-b", provider="anthropic")
    # Sizes chosen so a minimum over every model would name the primary's: only the refusing model's key may answer.
    sizes = {primary_plan.model: (210_000, 200_000), fallback_plan.model: (510_000, 500_000)}
    outcomes = {primary_plan.model: primary, fallback_plan.model: fallback}
    kinds = {"size": "context_overflow", "uncomparable_size": "context_overflow", "numberless_size": "context_overflow",
             "bad_request": "bad_request", "empty": "llm_empty_response", "transient": "provider_transient",
             "prediction": "context_overflow"}
    sent, passes, predicted = [], [], []

    def provider(call, disposition, **_kw):  # what a failure leaves, stamped as the loop stamps it
        outcome, (size, window) = outcomes[call.active_model], sizes[call.active_model]
        usage = call.accumulated_usage
        if outcome == "prediction":  # the admission refused the final candidate unsent, as ``_presend_context_fit`` keeps it
            physical = _physical_context_for_fit(disposition)
            with ua.bind_physical_attempt_context(physical):
                request = _attempt_request({"provider": call.context_fit_plan.provider, "model": call.active_model},
                    {"model": call.active_model, "messages": call.messages, "tools": [], "max_tokens": REPLY})
            free = physical.capacity_total_tokens - physical.response_reserve_tokens - size
            predicted.append(refused_candidate_facts(
                request, capacity_total_tokens=physical.capacity_total_tokens, cause="known_window_prediction",
                response_reserve_tokens=physical.response_reserve_tokens, free_tokens=free, estimated_input_tokens=size,
                predicted_free_tokens=free, predicted_slack_tokens=0))
            usage.update({REFUSED_CANDIDATE_KEY: predicted[-1], "_last_llm_error_kind": "context_overflow",
                          "reason_code": "context_overflow", "_last_llm_provider_message": "",
                          "_last_llm_failure": {"model": call.active_model, "failure_code": "context_overflow"}})
            return None, 0.0
        usage.update(_last_llm_error_kind=kinds[outcome], reason_code="llm_api_error", execution_status="infra_failed",
                     _last_llm_failure={"model": call.active_model, "failure_code": kinds[outcome]})
        stamp_owner_provider_message(usage, {"message": {
            "size": _stated(size, limit=window), "uncomparable_size": _stated(size, limit=window),
            "numberless_size": "context_length_exceeded", "bad_request": "Upstream request failed",
            "transient": "fixture transient"}.get(outcome, "")})
        sent.append(replace(_refused_capture(size), model=call.active_model, provider=call.context_fit_plan.provider,
                            physical_context=_physical_context_for_fit(disposition),
                            **({"candidate_measurement_kind": "opaque"} if outcome == "uncomparable_size" else {})))
        return None, 0.0

    monkeypatch.setattr(loop, "_dispatch_round_model", provider)
    monkeypatch.setattr(loop, "last_physical_attempt_capture", lambda: sent[-1] if sent else None)
    monkeypatch.setattr(lmc, "_apply_refusal_pass", lambda *_a, **_kw: passes.append(1) or False)  # change nothing
    monkeypatch.setenv("OUROBOROS_MODEL_FALLBACKS", fallback_plan.model)
    monkeypatch.setattr(fallback_cooldown, "is_cooling_down", lambda *_a: False)
    monkeypatch.setattr(fallback_cooldown, "mark_cooldown", lambda *_a, **_kw: None)
    monkeypatch.setattr(loop, "_rebind_context_fit_plan", lambda *_a, **_kw: (fallback_plan, "low"))
    usage = ctx.accumulated_usage
    assert loop._call_round_model(ctx)[0] is None
    walked = lmc._run_cross_model_fallback_chain(
        llm=None, ctx=ctx.tools._ctx, tools=ctx.tools, messages=ctx.messages, active_model=primary_plan.model,
        active_use_local=False, tool_schemas=[], active_effort="high", max_retries=1, drive_logs=ctx.drive_logs,
        task_id="t", round_idx=ctx.round_idx, event_queue=None, accumulated_usage=usage, task_type="task",
        emit_progress=lambda *_a, **_kw: None, context_fit_plan=primary_plan, active_context_mode="low")
    assert walked[0] is None and usage["_context_route_fp"] == primary_plan.route_fp  # the walk restored the primary
    assert (usage["_context_required_book_tokens"] == 0) is held  # whose Max view holds the book with no addition
    assert usage["_last_llm_failure"]["model"] == fallback_plan.model  # the last attempt's row, as other readers know it
    assert usage["_acting_route_failure"]["failure_code"] == kinds[primary]  # the acting route's own failure, kept
    assert bool(passes) is (primary != "uncomparable_size")  # that one round's recovery returned before its passes
    kind = usage["_last_llm_error_kind"]
    text = provider_terminal_fallback_text(usage, round_idx=ctx.round_idx,
                                           is_context_overflow=kind == "context_overflow", **_TERMINAL)
    refuser = fallback_plan if fallback in {"size", "numberless_size", "prediction"} else primary_plan
    refused, (size, window) = outcomes[refuser.model], sizes[refuser.model]
    stated = {"numberless_size": f"request input estimated {size:,} tokens (from the bytes it sent)",
              "prediction": f"request input estimated {size:,} tokens (host prediction before sending, no provider "
                            "response)"}.get(refused, f"request input {size:,} tokens (provider stated)")
    named = f"{stated} on {refuser.model}; assumed window {window:,} total: "
    if kind == "context_overflow":
        assert f"\n{named}" in text
        # A prediction is never a refused size; a provider's refusal is that model's smallest.
        assert (f"(smallest refused request ~{size:,} tokens" in text) is (refused != "prediction")
        if refused == "prediction":  # that admission's own book, window and reserve; the walk restored the primary's
            facts = predicted[-1]
            book = facts["physical_context"]["required_book_tokens"]
            usable = facts["capacity_total_tokens"] - facts["response_reserve_tokens"]
            restored = usage["_context_capacity_total_tokens"] - usage["_context_response_reserve_tokens"]
            assert refuser is fallback_plan and restored != usable and f"against {restored:,} usable" not in text
            assert (f"handbook (~{book:,} tokens), which stayed; without it the input is ~{size - book:,} against "
                    f"{usable:,} usable.") in text
        else:  # a request the provider refused was admitted with its book: its own size of it, and no arithmetic
            book = sent[-1].physical_context.required_book_tokens
            assert book and f"handbook (~{book:,} tokens), which stayed. The choices" in text and "without it" not in text
    else:  # the round's refusal, named apart from the failure that ended it, with no handbook line outside a size end
        assert f"\nEarlier in this round, a size refusal: {named}" in text
        assert text.count("request input") == 1 and "smallest refused" not in text and "handbook" not in text
    other = sizes[(fallback_plan if refuser is primary_plan else primary_plan).model][1]
    assert f"window {other:,} total" not in text  # no other route's window beside it


def test_the_round_size_refusal_lives_until_its_round_ends(round_call, monkeypatch):
    """Written where a refusal ends a send: a changed candidate that was never sent leaves it, the retry the provider
    refused again replaces it with its own facts, a cold continuation keeps it for its round, the next round's terminal
    does not read it, and an answer clears it with the round's refusal."""
    from ouroboros import loop, loop_model_call as lmc
    from ouroboros.context_anchor import REFUSED_INPUTS_KEY
    from ouroboros.context_view_facts import SIZE_REFUSAL_KEY
    from ouroboros.loop_transport import stamp_owner_provider_message
    from ouroboros.usage_accounting import PhysicalAttemptPreconditionFailed

    ctx = round_call(100_000, words=1_000)
    usage, script, sent, held = ctx.accumulated_usage, iter([120_000, None, 110_000, "answer"]), [], []

    def provider(*args, **kw):  # the transport boundary under the real dispatcher
        step = next(script)
        held.append((usage.get(SIZE_REFUSAL_KEY) or {}).get("tokens"))
        if step is None:
            raise PhysicalAttemptPreconditionFailed("the changed candidate is not strictly smaller")  # never sent
        sent.append(replace(_refused_capture(1_000), candidate_raw_sha256=f"sent-{len(sent)}",
                            physical_context=kw["physical_context"]))
        if step == "answer":
            return {"role": "assistant", "content": "fits"}, 0.0
        usage.update(_last_llm_error_kind="context_overflow", reason_code="llm_api_error",
                     _last_llm_failure={"model": args[2], "failure_code": "context_overflow"})
        stamp_owner_provider_message(usage, {"message": _stated(step, limit=100_000)})
        return None, 0.0

    monkeypatch.setattr(loop, "call_llm_with_retry", provider)
    monkeypatch.setattr(loop, "last_physical_attempt_capture", lambda: sent[-1] if sent else None)
    monkeypatch.setattr(lmc, "_apply_refusal_pass", lambda *_a, **_kw: True)  # each pass claims a change
    assert loop._call_round_model(ctx)[0] is None
    assert held == [None, 120_000, 120_000]  # the unsent candidate left the refusal as it was
    assert (usage[SIZE_REFUSAL_KEY]["tokens"], usage[SIZE_REFUSAL_KEY]["basis"]) == (110_000, "provider_stated")
    assert min(usage[REFUSED_INPUTS_KEY].values()) == 110_000  # the policy's own record, as before
    flags = dict(is_context_overflow=True, **_TERMINAL)
    text = provider_terminal_fallback_text(usage, round_idx=ctx.round_idx, **flags)
    assert "request input 110,000 tokens (provider stated)" in text
    # A cold continuation keeps the usage as JSON (``owner_wait.continuation_state``): the restored round names it alike.
    assert provider_terminal_fallback_text(json.loads(json.dumps(usage)), round_idx=ctx.round_idx, **flags) == text
    unslotted = {key: value for key, value in usage.items() if key != SIZE_REFUSAL_KEY}
    assert provider_terminal_fallback_text(usage, round_idx=ctx.round_idx + 1, **flags) == provider_terminal_fallback_text(
        unslotted, round_idx=ctx.round_idx + 1, **flags)  # another round's terminal reads none of it
    ctx.round_idx += 1
    assert loop._call_round_model(ctx)[0]["content"] == "fits"
    assert SIZE_REFUSAL_KEY not in usage and lmc.CONTEXT_REFUSAL_ROUND_KEY not in usage


# The round's facts as its last measurement left them: what a refusal they do not describe must never borrow.
_ROUND_FACTS = {"execution_id": "e", "_context_prompt_estimate": 777_777, "_context_route_fp": "route-a",
                "_context_window": {"text": "window 999,999 total: server metadata; usable input 900,000; reply reserve 65,536"},
                "_last_llm_error_kind": "context_overflow", "_context_required_book_tokens": 3_333,
                "_context_capacity_total_tokens": 999_999, "_context_response_reserve_tokens": 65_536}


def _attempt(round_idx, route="route-a", sent=0, book=2_000, **facts):
    """A refused attempt of round ``round_idx``: its physical context (reserve 4,096, a required book of ``book`` tokens
    in a 60,000 window; 0 where its Max view holds the book anyway) and the bytes it sent; ``facts`` makes it a
    pre-send prediction's candidate."""
    physical = PhysicalAttemptContext(profile="owner_low", rendered_mode="low", measurement_basis="cold_estimate",
                                      route_fp=route, round_id=f"e:round:{round_idx}", target_total_tokens=None,
                                      capacity_total_tokens=60_000, context_target_miss=False, automatic_pass_used=False,
                                      measurement_density=1.0, response_reserve_tokens=4_096, required_book_tokens=book)
    return SimpleNamespace(physical_context=physical, prompt_tokens_bounded_estimate=sent, **facts)


_PREDICTED = dict(predicted_free_tokens=-5_096, estimated_input_tokens=61_000, capacity_total_tokens=60_000,
                  response_reserve_tokens=4_096)
_UNKNOWN_BOOK = "\nThis task directly requires the whole development handbook, which stayed. The choices"
_OWN_BOOK = "\nThis task directly requires the whole development handbook (~2,000 tokens), which stayed. The choices"


@pytest.mark.parametrize("refused,measured,message,request_line,book_line", [
    # No disposition: nothing measured this request; the round's facts lend it nothing.
    (None, False, "", "request size unknown on m; its window is not known here.", _UNKNOWN_BOOK),
    # Unmeasured, its own attempt: that attempt's bytes, reserve and book, never the round's window or estimate.
    (_attempt(3, sent=5_000), False, "", "request input estimated 5,000 tokens (from the bytes it sent) on m; its "
     "window is not known here; reply reserve 4,096.", _OWN_BOOK),
    # An earlier round's capture is another attempt: neither its bytes and reserve nor, unmeasured, the round's facts.
    (_attempt(2, sent=5_000), False, "", "request size unknown on m; its window is not known here.", _UNKNOWN_BOOK),
    # Measured, an earlier round's capture: this round's own estimate and window, never that capture's bytes or reserve.
    (_attempt(2, sent=5_000), True, "", "request input estimated 777,777 tokens (this round's estimate) on m; assumed "
     "window 999,999 total: server metadata; usable input 900,000; reply reserve 65,536.", _UNKNOWN_BOOK),
    # Measured, its own attempt on another route than the round's facts: that attempt's facts alone.
    (_attempt(3, route="route-b"), True, "", "request size unknown on m; its window is not known here; reply reserve "
     "4,096.", _OWN_BOOK),
    # The provider refused what the host sent and admitted with its book: its own size of it, no decomposition.
    (_attempt(3, sent=5_000), True, _stated(70_000, limit=60_000), "request input 70,000 tokens (provider stated) on "
     "m; assumed window 999,999 total", _OWN_BOOK),
    # Its own Max view held the book with no addition: the round's later measure of the same route lends no size.
    (_attempt(3, sent=5_000, book=0), True, _stated(70_000, limit=60_000), "request input 70,000 tokens (provider "
     "stated) on m; assumed window 999,999 total", _UNKNOWN_BOOK),
    # An earlier round's prediction: its admission's book and window are not this round's.
    (_attempt(2, **_PREDICTED), True, "", "request input estimated 61,000 tokens (host prediction before sending, no "
     "provider response) on m; assumed window 999,999 total", _UNKNOWN_BOOK),
    # This round's own prediction: its admission's book, size, window and reserve, whatever the round's facts say.
    (_attempt(3, **_PREDICTED), True, "", "request input estimated 61,000 tokens (host prediction before sending, no "
     "provider response) on m; assumed window 999,999 total", "\nThis task directly requires the whole development "
     "handbook (~2,000 tokens), which stayed; without it the input is ~59,000 against 55,904 usable. The choices"),
    # A prediction whose facts state no size or window: unknown, and nothing is subtracted from it.
    (_attempt(3, predicted_free_tokens=-5_096), True, "", "request size unknown on m; assumed window 999,999 total",
     _OWN_BOOK),
], ids=["no_disposition", "unmeasured_own", "unmeasured_earlier_round", "measured_earlier_round", "own_other_route",
        "provider_refused_sent_book", "own_max_view_same_route", "earlier_round_prediction", "own_prediction",
        "own_prediction_without_its_size"])
def test_a_size_refusal_borrows_nothing_its_own_attempt_does_not_tell(refused, measured, message, request_line,
                                                                      book_line):
    """Negative controls for the refusal slot's guards (``remember_size_refusal``): a missing disposition, an
    unmeasured round, an earlier round's attempt and an own attempt on another route each leave unknown what that
    attempt does not tell, through the production terminal. An own attempt of this round names its own book, never
    the round's measure of the same route; only the host's own prediction keeps a handbook decomposition, its
    admission's own."""
    from ouroboros.context_view_facts import SIZE_REFUSAL_KEY, remember_size_refusal

    usage = json.loads(json.dumps({**_ROUND_FACTS, "_last_llm_provider_message": message}))
    remember_size_refusal(SimpleNamespace(accumulated_usage=usage, context_fit_plan=None, round_idx=3, active_model="m"),
                          refused, measured=measured)
    text = provider_terminal_fallback_text(usage, round_idx=3, is_context_overflow=True, **_TERMINAL)
    assert f"\n{request_line}" in text and book_line in text, text
    for borrowed in ("777,777", "999,999", "5,000", "4,096", "3,333", "~"):  # each only where its own attempt told it
        assert (borrowed in text) is (borrowed in request_line + book_line), (borrowed, text)
    slot = usage[SIZE_REFUSAL_KEY]
    assert ("book" in slot) is ("(~" in book_line) and ("capacity" in slot) is ("without it" in book_line)


def _diagnostic_terminal(ctx):
    """The actual round terminal carries usage and the renewed view, never a fit plan."""
    from ouroboros.context_renewal import renew_after_refusal

    limit = loop._RoundLimitContext(
        messages=ctx.messages, llm=SimpleNamespace(chat=lambda *_a, **_kw: pytest.fail("unexpected send")),
        active_model=ctx.active_model, active_effort="high", max_retries=1, drive_logs=ctx.drive_logs,
        task_id="t", round_idx=ctx.round_idx, event_queue=None, accumulated_usage=ctx.accumulated_usage,
        task_type="task", active_use_local=False, max_rounds=200, drive_root=ctx.drive_root)
    assert not hasattr(limit, "context_fit_plan")
    assert not renew_after_refusal(limit, ctx.tools)
    return loop._handle_provider_unavailable(limit, error_kind="context_overflow")[0]


def _diagnostic_candidate(ctx, physical, allowance):
    with ua.bind_physical_attempt_context(physical):
        request = _attempt_request({"provider": ctx.context_fit_plan.provider, "model": ctx.active_model}, {
            "model": ctx.active_model, "messages": ctx.messages, "tools": [],
            "max_tokens": allowance or ctx.accumulated_usage["_context_reply_allowance_tokens"]})
    capture = replace(_refused_capture(request.prompt_tokens_bounded_estimate), physical_context=physical,
        candidate_raw_sha256=request.candidate_raw_sha256, candidate_raw_size_bytes=request.candidate_raw_size_bytes,
        candidate_context_sha256=request.candidate_context_sha256,
        candidate_context_size_bytes=request.candidate_context_size_bytes)
    return request, capture


@pytest.mark.parametrize("refusal", ["provider", "prediction"])
@pytest.mark.parametrize("variant,refusals", [("unsent", 1), ("unchanged", 1), ("smaller", 2)])
def test_refused_book_arithmetic_reads_only_the_refused_candidate(round_call, monkeypatch, refusal, variant, refusals):
    """A rendered required book, the real admission and strict-shrink predicate and the production terminal. What the
    input is without the book is the host's own refusal before sending: that admission's book, size, window and
    reserve for the round's latest refused candidate, also after a changed view went unsent. The provider refused a
    request the host admitted with its book: its own size of the book then, never what fits without it, and never the
    round's later measure of the same route (the same view counted again at that refusal's own density)."""
    from ouroboros.llm_attempt import main_context_fit_refusal

    # The provider's window admits every candidate here; the smaller one starts the task but admits none of these.
    window, words, unsent, smaller = (100_000, 4_000, 1_900, 1_000) if refusal == "provider" else (
        17_000, 12_000, 5_700, 8_000)
    ctx = round_call(100_000, book_chars=8_000, words=words)
    # _real_plan's synthetic LOW_SYSTEM is useful for density tests but does not contain its core's book.
    # Build the real projection here so even the shrunk view still contains the measured handbook.
    plan = context_fit.build_context_fit_plan(None, ctx.context_fit_plan.core, {"id": "t"}, preferred_mode="low",
        route_resolver=lambda *_a, **_kw: ({"provider": "openai", "model": ctx.active_model},
            ce.CapabilityEvidence(window, "confirmed", "metadata", "route-a")), tool_schemas=[])
    ctx.context_fit_plan = ctx.tools._ctx.context_fit_plan = plan
    ctx.messages[:] = [*plan.messages_for("low"), {"role": "user", "content": "word " * words}]
    assert "x" * 8_000 in json.dumps(ctx.messages)
    sent, judged, passes, predicted = [], [], [], []

    def transport(*_a, **kw):
        request, capture = _diagnostic_candidate(ctx, kw["physical_context"], kw.get("max_tokens"))
        if kw.get("candidate_predicate"):
            judged.append(kw["candidate_predicate"](request))
            if not judged[-1]:
                raise ua.PhysicalAttemptPreconditionFailed("not strictly smaller")
        if not kw["physical_fit_predicate"](request):  # the real admission refuses it; its facts as it judged them
            predicted.append(main_context_fit_refusal(request))
            raise ua.PhysicalAttemptPreconditionFailed("refused before sending")
        sent.append(capture)
        ctx.accumulated_usage.update(_last_llm_error_kind="context_overflow", reason_code="llm_api_error")
        stamp_owner_provider_message(ctx.accumulated_usage, {
            "message": _stated(120_000 if len(sent) == 1 else 110_000, limit=100_000)})
        return None, 0.0

    def shrink(call, *_a, **_kw):
        passes.append(1)
        if len(passes) > 1 or variant == "unchanged":
            return False
        # Cyrillic carries more bytes: never strictly smaller, so never sent; fewer words are smaller.
        call.messages[-1]["content"] = "слово " * unsent if variant == "unsent" else "word " * smaller
        return True

    monkeypatch.setattr(loop, "call_llm_with_retry", transport)
    monkeypatch.setattr(loop, "last_physical_attempt_capture", lambda: sent[-1] if sent else None)
    monkeypatch.setattr(lmc, "_apply_refusal_pass", shrink)
    assert loop._call_round_model(ctx)[0] is None
    assert (len(sent), len(predicted)) == ((refusals, 0) if refusal == "provider" else (0, refusals))
    assert judged == ([] if variant == "unchanged" else [variant == "smaller"])
    usage = ctx.accumulated_usage
    assert 0 < usage["_context_required_book_tokens"] < usage["_context_prompt_estimate"]
    text = _diagnostic_terminal(ctx)
    assert "directly requires the whole development handbook" in text
    if refusal == "provider":
        from ouroboros.context_view_facts import SIZE_REFUSAL_KEY

        assert f"request input {110_000 if refusals == 2 else 120_000:,} tokens (provider stated)" in text
        own, later = sent[-1].physical_context.required_book_tokens, usage["_context_required_book_tokens"]
        assert usage[SIZE_REFUSAL_KEY]["route_fp"] == usage["_context_route_fp"]  # the same route either way
        # After a refusal the round measures its view at that refusal's density; only a changed view went unsent.
        assert (later > own) is (variant != "unsent") and (later == own) is (variant == "unsent")
        assert f"handbook (~{own:,} tokens), which stayed. The choices" in text and "without it" not in text
        assert later == own or f"~{later:,}" not in text
        return
    facts = predicted[-1]
    size, book = facts["estimated_input_tokens"], facts["physical_context"]["required_book_tokens"]
    usable = facts["capacity_total_tokens"] - facts["response_reserve_tokens"]
    assert size > usable and book == usage["_context_required_book_tokens"]  # the same handbook in every view here
    assert f"request input estimated {size:,} tokens (host prediction before sending, no provider response)" in text
    assert (f"handbook (~{book:,} tokens), which stayed; without it the input is ~{max(0, size - book):,} against "
            f"{usable:,} usable.") in text
    if variant == "unsent":  # the round's current measurement is the changed view, which went unsent
        later = usage["_context_prompt_estimate"]
        assert later != size and f"~{max(0, later - book):,} against" not in text
    if variant == "smaller":  # the first refused candidate is no longer the round's latest
        assert f"~{max(0, predicted[0]['estimated_input_tokens'] - book):,} against" not in text


@pytest.mark.parametrize("anchored", [False, True])
@pytest.mark.parametrize("seam", ["round", "smaller_send"])
def test_exact_local_nano_refusal_uses_its_own_request_facts(round_call, monkeypatch, seam, anchored):
    """The local instance's exact count names the refused request, refused in a round's send or in output recovery's
    smaller send; where the round's measure is anchored on an accepted request, the refusal becomes window evidence
    stored as an exact local count, never as a refusal that stated no number (the window Settings shows); an
    unanchored one stays task-local, as before."""
    from ouroboros import local_model, loop_llm_call
    from ouroboros import loop_output_exhaustion as recovery
    from ouroboros.context_anchor import INPUT_ANCHOR_KEY, _anchor_model_key
    from ouroboros.context_budget import LocalContextTooLargeError
    from ouroboros.context_view_facts import SIZE_REFUSAL_KEY, window_projection
    from ouroboros.context_window import observation_scope, resolve_window
    from ouroboros.llm import LLMClient
    from ouroboros.local_model_server import input_fingerprint

    ctx = round_call(100_000, words=1_000)
    if anchored:  # an accepted request of this conversation, at the estimator's own count (density 1)
        ctx.accumulated_usage[INPUT_ANCHOR_KEY] = {_anchor_model_key(ctx.context_fit_plan): {
            "input_tokens": 5_000, "estimate_tokens": 5_000, "round": 0, "rewrite_epoch": 0}}
    monkeypatch.setattr(local_model, "get_manager", lambda: SimpleNamespace(
        serving_context_evidence=lambda: {"context_window": 16_384, "confirmed": True, "process_id": 123,
                                          "source": "published_server_arguments", "port": 1},
        measure_prepared_input=lambda payload: {"supported": True, "input_is_exact": True, "input_tokens": 15_000,
            "context_window": 16_384, "process_id": 123, "native_input_sha256": input_fingerprint(payload),
            "output_limit_enforced": True, "reasoning_included_in_limit": True}))
    client = LLMClient(api_key="unused")
    monkeypatch.setattr(client, "_get_local_client", lambda: SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=lambda **_kw: pytest.fail("refused request must not send")))))
    produced = []

    def transport(*_a, **kw):
        physical = replace(kw["physical_context"], rendered_mode="nano", profile="owner_nano")
        with ua.bind_physical_attempt_context(physical), pytest.raises(LocalContextTooLargeError) as caught:
            client._chat_local([dict(m) for m in ctx.messages], None, max_tokens=65_536, tool_choice="auto")
        error = caught.value
        produced.append(error.refused_candidate)
        assert "below the floor of 4096" in str(error)
        assert "response_reserve_tokens" not in error.refused_candidate
        loop_llm_call._record_llm_call_error(error, loop_llm_call._LlmErrorContext(
            task_id="t", task_type="task", execution_id="e", round_id=physical.round_id, llm_call_id="c1",
            round_idx=ctx.round_idx, attempt=0, model=ctx.active_model, request_ref=None, drive_logs=ctx.drive_logs,
            event_queue=None, accumulated_usage=ctx.accumulated_usage))
        return None, 0.0

    monkeypatch.setattr(loop, "call_llm_with_retry", transport)
    monkeypatch.setattr(loop, "last_physical_attempt_capture", lambda: None)
    monkeypatch.setattr(lmc, "_apply_refusal_pass", lambda *_a, **_kw: False)
    if seam == "round":
        assert loop._call_round_model(ctx)[0] is None and len(produced) == 1
    else:  # the episode's checkpointing and attempt bookkeeping are not what is measured here
        monkeypatch.setattr(recovery, "_save", lambda _ctx: None)
        monkeypatch.setattr(recovery, "_invoke", lambda _ctx, send, _step, key=recovery.KEY: send())
        ctx.accumulated_usage[recovery.KEY] = {"stage": "smaller", "comparison": {
            "provider": ctx.context_fit_plan.provider, "model": ctx.active_model, "max_completion_tokens": 65_536,
            "candidate_measurement_kind": "canonical_json_v1", "candidate_raw_sha256": "earlier",
            "candidate_context_size_bytes": 10 ** 9}}
        assert recovery._send_smaller(ctx, 0.0)[0] is None and len(produced) == 1
    slot = ctx.accumulated_usage[SIZE_REFUSAL_KEY]
    assert (slot["tokens"], slot["local_window"], slot["reserve"], slot["reply_limit"]) == (15_000, 16_384, 0, 4_096)
    assert "window" not in slot  # the prior round's remote window is not this local instance's
    text = _diagnostic_terminal(ctx)
    assert "request input 15,000 tokens (exact local count before sending, no provider response)" in text
    assert "window 16,384 total; remaining reply room 1,384; requested reply limit 4,096" in text
    assert "65,536" not in text and "reply reserve" not in text
    scope, root = observation_scope(ctx.context_fit_plan), ce.canonical_evidence_root()
    rows = (((ce._load(root).get("main_windows") or {}).get(scope) or {}).get("observations") or [])
    refusals = [(row["refused"], row["kind"], row.get("basis")) for row in rows if row.get("refused")]
    assert refusals == ([(15_000, "input", "exact_local_count")] if anchored else [])
    projection = window_projection(resolve_window(ce.CapabilityEvidence(100_000, "confirmed", "provider_metadata", scope),
                                                  65_536, root=root, observation_fp=scope), under_way=True)
    assert (projection["source"], projection["basis"]) == (("refusal", "exact_local_count") if anchored
                                                           else ("provider_metadata", ""))
    assert ("(derived from an exact local count before sending); usable input 15,000" in projection["text"]) is anchored
    assert "stated no number" not in projection["text"]


@pytest.mark.parametrize("earlier_size", [False, True])
def test_network_terminal_keeps_size_facts_only_as_an_earlier_refusal(round_call, monkeypatch, earlier_size):
    from ouroboros import loop_llm_call

    ctx, sent, calls = round_call(100_000, words=1_000), [], []

    def transport(*_a, **kw):
        calls.append(1)
        if earlier_size and len(calls) == 1:
            _, capture = _diagnostic_candidate(ctx, kw["physical_context"], kw.get("max_tokens"))
            sent.append(capture)
            ctx.accumulated_usage.update(_last_llm_error_kind="context_overflow", reason_code="llm_api_error")
            stamp_owner_provider_message(ctx.accumulated_usage, {"message": _stated(120_000, limit=100_000)})
        else:
            loop_llm_call._record_llm_call_error(httpx.ConnectError("connection refused"), loop_llm_call._LlmErrorContext(
                task_id="t", task_type="task", execution_id="e", round_id=kw["physical_context"].round_id,
                llm_call_id=f"c{len(calls)}", round_idx=ctx.round_idx, attempt=0, model=ctx.active_model,
                request_ref=None, drive_logs=ctx.drive_logs, event_queue=None, accumulated_usage=ctx.accumulated_usage))
        return None, 0.0

    def shrink(call, *_a, **_kw):
        call.messages[-1]["content"] = "word " * 500
        return True

    monkeypatch.setattr(loop, "call_llm_with_retry", transport)
    monkeypatch.setattr(loop, "last_physical_attempt_capture", lambda: sent[-1] if sent else None)
    monkeypatch.setattr(lmc, "_apply_refusal_pass", shrink)
    assert loop._call_round_model(ctx)[0] is None and len(calls) == (2 if earlier_size else 1)
    assert ctx.accumulated_usage["_last_llm_error_kind"] == "provider_error"
    text = provider_terminal_fallback_text(ctx.accumulated_usage, round_idx=ctx.round_idx,
                                           is_context_overflow=False, **_TERMINAL)
    assert ("Earlier in this round, a size refusal: request input 120,000 tokens" in text) is earlier_size
    assert "smallest refused" not in text
    if not earlier_size:
        assert "window" not in text and "request input" not in text
