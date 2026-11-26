"""A start whose always-whole core cannot fit its known window is refused before the memory capture.

The ordinary build (``context.build_context_fit_plan``) over the real checkout and a private synthetic drive
(``tests/_memory_scale_fixture.install``); routes are stub resolvers shaped as the call catalogues answer. The bound
never refuses a request the full fit admits: ``test_the_bound_never_exceeds_the_fitted_minimum``.
"""
from __future__ import annotations

import datetime
import json
import math
import threading
from types import SimpleNamespace

import pytest

from ouroboros import capability_evidence as ce, context, context_fit as fit, context_start_bound as bound
from ouroboros.context_budget import NANO_MIN_HEADROOM_TOKENS
from tests import _memory_scale_fixture as scale

RESERVE = max(min(NANO_MIN_HEADROOM_TOKENS, 16_384), 10_000)  # Nano's reply reserve on these routes


def evidence(window, basis="estimator", fp="synthetic-route"):
    return SimpleNamespace(route_fp=fp, status="confirmed", stale=False, window_tokens=window, source="synthetic",
                           route_facts={"working_reserve_tokens": 10_000, "basis": basis})


def resolver(*windows, basis="estimator", calls=None):
    """Answers the windows in order (the last one repeats); ``calls`` records each call's flags."""
    answers = list(windows)

    def resolve(task, *, allow_fetch):
        if calls is not None:
            calls.append((allow_fetch, bool(task.get("route_start_read")), bool(task.get("route_read_again"))))
        window = answers.pop(0) if len(answers) > 1 else answers[0]
        return {"provider": "synthetic", "model": "synthetic"}, evidence(window, basis)
    return resolve


@pytest.fixture
def drive(tmp_path, monkeypatch):
    monkeypatch.setattr(fit, "main_output_reserve_tokens", lambda **_kw: 16_384)
    monkeypatch.setattr(context, "utc_now_iso", lambda: "2026-11-26T12:00:00+00:00")
    return scale.install(tmp_path / "drive")


def build(drive, route_resolver, *, text="Continue in this room.", mode="nano", ctx=None, task=None):
    env, memory = scale.env_on(drive)
    task = task or {"id": "start", "type": "task", "chat_id": 1, "text": text, "_is_direct_chat": True}
    return context.build_context_fit_plan(env, memory, task, preferred_mode=mode, ctx=ctx, tool_schemas=scale.tools(),
                                          route_resolver=route_resolver)


def capture_spy(monkeypatch, *, fail=False, before=None):
    seen = []
    real = context._capture_memory_view

    def spy(*args, **kwargs):
        if fail:
            pytest.fail("the memory capture ran")
        if before is not None:
            before()
        seen.append(1)
        return real(*args, **kwargs)
    monkeypatch.setattr(context, "_capture_memory_view", spy)
    return seen


def refused_by_bound(drive, monkeypatch, **kwargs):
    with monkeypatch.context() as patch, pytest.raises(fit.MinimumViewUnfit) as raised:
        patch.setattr(context, "_capture_memory_view", lambda *_a, **_kw: pytest.fail("the memory capture ran"))
        build(drive, resolver(25_000), **kwargs)
    return raised.value


def test_a_frame_the_window_cannot_hold_is_refused_before_the_memory_capture(drive, monkeypatch):
    capture_spy(monkeypatch, fail=True)
    with pytest.raises(fit.MinimumViewUnfit) as raised:
        build(drive, resolver(25_000))
    unfit = raised.value
    assert unfit.memory_measured is False and unfit.memory_view_tokens == 0
    names, sizes = [name for name, _ in unfit.parts], [size for _, size in unfit.parts]
    assert {"identity", "marks", "system prompt", "constitution", "scratchpad", "overview", "tool schemas"} <= set(names)
    assert sizes == sorted(sizes, reverse=True)
    assert unfit.window_tokens == 25_000 and unfit.response_reserve_tokens == RESERVE
    assert unfit.estimated_input_tokens + RESERVE > 25_000


def test_a_frame_just_under_the_window_reaches_the_ordinary_fit_unchanged(drive, monkeypatch):
    first = refused_by_bound(drive, monkeypatch)
    need = first.estimated_input_tokens + first.response_reserve_tokens  # the bound's own need, estimator basis
    # Runtime facts and health probes change between any two builds (process lists, clocks): held still here.
    monkeypatch.setattr(context, "build_runtime_section", lambda *_a, **_kw: "## Runtime\n\nheld still")
    monkeypatch.setattr(context, "build_health_invariants", lambda *_a, **_kw: "")

    def outcome(window):
        try:
            plan = build(drive, resolver(window))
            return plan.core_sha256, plan.initial_mode
        except fit.MinimumViewUnfit as unfit:  # the full request is larger than the frame: the fit decides
            return str(unfit), unfit.memory_measured

    for window in (need, 1_000_000):
        seen = capture_spy(monkeypatch)
        with_bound = outcome(window)
        assert seen, "the capture ran"
        with monkeypatch.context() as patch:
            patch.setattr(bound, "refuse_before_capture", lambda *_a, **_kw: None)
            assert outcome(window) == with_bound


@pytest.mark.parametrize("mode", ["max", "low", "nano"])
@pytest.mark.parametrize("route", [("grok", 80_000, "estimator"), ("gemini", 92_000, "provider"),
                                   ("openai", 133_554, "provider"), ("text-85k", 85_000, "provider")])
@pytest.mark.parametrize("shape", ["fresh", "mature"])
def test_the_bound_never_exceeds_the_fitted_minimum(tmp_path, monkeypatch, shape, route, mode):
    """The identity ``bound <= fitted minimum``: the fitter's own refusal at the window its core alone fills, memory
    rendered beside it. The windows are the shipped catalogues' (``skills/*/plugin.py``) and a hard 85k text route;
    the comparison is made at the window that forces the fitter's minimum."""
    provider, _window, basis = route
    monkeypatch.setattr(fit, "main_output_reserve_tokens", lambda **_kw: 16_384)
    drive = scale.installed(tmp_path, shape)
    unfit_bound = refused_by_bound(drive, monkeypatch, mode=mode)
    monkeypatch.setattr(bound, "refuse_before_capture", lambda *_a, **_kw: None)

    def refusal(window):
        with pytest.raises(fit.MinimumViewUnfit) as raised:
            build(drive, resolver(window, basis=basis), mode=mode)
        return raised.value

    core_alone = refusal(20_000)
    assert not core_alone.memory_measured
    minimum = refusal(core_alone.fixed_input_tokens + core_alone.response_reserve_tokens)
    assert minimum.memory_measured and minimum.memory_view_tokens > 0, provider
    assert unfit_bound.raw_input_tokens <= minimum.raw_input_tokens, provider
    assert math.ceil(unfit_bound.raw_input_tokens * unfit_bound.measurement_density) <= minimum.estimated_input_tokens


def test_an_arrived_route_answer_is_asked_again_before_an_early_refusal_stands(drive, monkeypatch):
    calls = []
    monkeypatch.setattr(fit, "_route_key", lambda _task: "synthetic-question")
    question = fit.RouteQuestion({"id": "start", "chat_id": 1}, resolver(25_000, 1_000_000, calls=calls))
    assert question.asked.wait(10)
    seen = capture_spy(monkeypatch)
    plan = build(drive, question)
    assert seen and plan.window_tokens == 1_000_000
    # The question's own call, the bound's ``again`` and the horizon guard's ``again`` (the fit took the arrived
    # answer, ``early``: ``peek`` consumed nothing); no fresh read of the route at the start.
    assert calls == [(True, False, False), (True, False, False), (True, False, False)]


def test_no_arrived_answer_and_an_unknown_window_never_refuse_early(drive, monkeypatch):
    release, calls = threading.Event(), []
    held = resolver(25_000, calls=calls)

    def waiting(task, *, allow_fetch):
        assert release.wait(30)
        return held(task, allow_fetch=allow_fetch)
    monkeypatch.setattr(fit, "_route_key", lambda _task: "synthetic-question")
    question = fit.RouteQuestion({"id": "start", "chat_id": 1}, waiting)
    seen = capture_spy(monkeypatch, before=release.set)  # the answer arrives once the capture runs
    with pytest.raises(fit.MinimumViewUnfit):  # the full fit decides on its usual answer
        build(drive, question)
    assert seen
    seen = capture_spy(monkeypatch)
    build(drive, resolver(None))
    assert seen


@pytest.mark.parametrize("start", ["child", "declared", "nanny", "held_note"])
def test_a_child_a_declared_start_a_nanny_and_a_held_note_resume_get_no_early_bound(drive, monkeypatch, start):
    from ouroboros.memory import Memory

    class Reached(Exception):
        pass

    def capture(*_a, **_kw):
        raise Reached
    monkeypatch.setattr(Memory, "load_identity", lambda self: pytest.fail("the bound read identity"))
    monkeypatch.setattr(context, "_capture_context_core", capture)
    task = {"id": "start", "type": "task", "chat_id": 1, "text": "hi"}
    ctx = None
    if start == "child":
        task["delegation_role"] = "subagent"
    elif start == "declared":
        task["input_sources"] = "declared"
    elif start == "nanny":
        monkeypatch.setattr(context, "_view_spec", lambda *_a: SimpleNamespace(role="nanny", knowledge=False))
    else:
        ctx = SimpleNamespace(_held_note_capture={"messages": []})
    with pytest.raises(Reached):
        build(drive, resolver(25_000), task=task, ctx=ctx)


def test_the_assignment_is_not_counted(drive, monkeypatch):
    long, short = refused_by_bound(drive, monkeypatch, text="Paste " * 66_667), refused_by_bound(drive, monkeypatch, text="hi")
    assert len("Paste " * 66_667) >= 400_000
    assert long.raw_input_tokens == short.raw_input_tokens and long.parts == short.parts
    seen = capture_spy(monkeypatch)
    window = short.estimated_input_tokens + short.response_reserve_tokens + 2_000
    try:  # an owner-Nano start: its exact-input pointer, not this bound, answers the paste
        build(drive, resolver(window), text="Paste " * 66_667)
    except fit.MinimumViewUnfit:
        pass
    assert seen, "the long paste reached the capture"


def test_marks_count_only_when_the_chronicle_is_already_active(tmp_path, monkeypatch):
    from ouroboros.chronicle_store import ChronicleStore

    monkeypatch.setattr(fit, "main_output_reserve_tokens", lambda **_kw: 16_384)
    drive = tmp_path / "drive"
    (drive / "memory" / "knowledge").mkdir(parents=True)
    (drive / "memory" / "identity.md").write_text(scale.text("Identity", 1800), encoding="utf-8")

    class Reached(Exception):
        pass

    def capture(*_a, **_kw):
        raise Reached
    activated = []
    monkeypatch.setattr(ChronicleStore, "ensure_activated", lambda self, **_kw: activated.append(1))
    monkeypatch.setattr(context, "_capture_context_core", capture)
    with pytest.raises(fit.MinimumViewUnfit) as raised:
        build(drive, resolver(25_000))
    assert "marks" not in dict(raised.value.parts) and "identity" in dict(raised.value.parts)
    assert not activated


def _witness(density, chars, *, hours, fp="synthetic-route", view="owner_turn"):
    observed = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=hours)
    return {"prompt_chars": chars, "prompt_tokens": int(density * chars / 4), "observed_at": observed.isoformat(),
            "basis": ce.MAIN_DENSITY_BASIS, "route_fp": fp, "view_kind": view}


def test_a_provider_route_uses_the_lowest_fresh_witness_capped_at_one(drive, monkeypatch, tmp_path):
    root = tmp_path / "evidence"
    path = ce._store_path(root)
    path.parent.mkdir(parents=True)
    monkeypatch.setattr(ce, "canonical_evidence_root", lambda: root)
    assert ce.lowest_main_token_density(root, "synthetic-route", "synthetic", view_kind="owner_turn") == 1.0  # no store
    store = {"token_density": {"synthetic": {"pairs": [_witness(0.8, 40_000, hours=5), _witness(1.2, 400_000, hours=1)]}}}
    path.write_text(json.dumps(store), encoding="utf-8")
    assert ce.lowest_main_token_density(root, "synthetic-route", "synthetic", view_kind="owner_turn") == pytest.approx(0.8)
    assert ce.resolve_main_token_density(root, "synthetic-route", "synthetic", view_kind="owner_turn")[0] == pytest.approx(1.2)

    first = refused_by_bound(drive, monkeypatch)  # estimator basis: density 1.0
    frame = first.raw_input_tokens
    window = math.ceil(frame * 0.9) + RESERVE  # refused at 1.2 and at 1.0, not at 0.8
    seen = capture_spy(monkeypatch)
    with pytest.raises(fit.MinimumViewUnfit):  # the full fit refuses; the bound did not refuse early
        build(drive, resolver(window, basis="provider"))
    assert seen
    path.write_text("{ not json", encoding="utf-8")
    assert ce.lowest_main_token_density(root, "synthetic-route", "synthetic", view_kind="owner_turn") is None
    seen.clear()
    with pytest.raises(fit.MinimumViewUnfit):
        build(drive, resolver(window - 5_000, basis="provider"))
    assert seen, "an unreadable store refuses nothing early"
