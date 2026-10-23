"""Route units and working room enter the ordinary fitter before memory selection."""
from dataclasses import asdict, replace
import json
import math
from types import SimpleNamespace

import pytest

from ouroboros import context_fit as fit, memory_view
from ouroboros.context_anchor import InputAnchor
from ouroboros.contracts.realtime import RouteFacts
from ouroboros.section_floor import HostSection
from tests import _memory_view_synthetic as syn


@pytest.fixture(autouse=True)
def evidence(monkeypatch):
    monkeypatch.setattr(fit, "main_output_reserve_tokens", lambda **_kw: 16_384)
    monkeypatch.setattr(fit, "_route_calibration_ratio", lambda *_a, **_kw: 1.0)
    monkeypatch.setattr("ouroboros.capability_evidence.resolve_main_token_density",
                        lambda *_a, **_kw: (1.0, "cold_estimate"))


def core(root, *, memory=False):
    section = HostSection("runtime", "## Runtime\n" + "observed details " * 6000,
                          "## Runtime\nThe operative facts.")
    return fit.ContextCore(base_prompt="SYSTEM", bible_md="BIBLE", architecture_md="# Architecture",
        development_md="# Development", docs_need_development=False, semi_stable_text="IDENTITY",
        dynamic_text=section.full_text, user_content_json=json.dumps("Exact owner words."),
        dynamic_sections=(section,), section_root=str(root), section_task_id="route-facts-test",
        memory_view_json=memory_view.snapshot_json(syn.snapshot(story=syn.pages(3, 4000))) if memory else "")


def plan(captured, window, *, facts=None, mode="nano"):
    evidence = SimpleNamespace(route_fp="test-route", status="confirmed" if window else "unprobeable",
                               stale=False, window_tokens=window, source="test")
    if facts is not None:
        evidence.route_facts = asdict(facts)
    return fit.build_context_fit_plan(SimpleNamespace(drive_root=captured.section_root), captured,
        {"id": "route-facts-test"}, preferred_mode=mode,
        route_resolver=lambda *_a, **_kw: ({"provider": "test", "model": "synthetic"}, evidence))


def addressed(built, mode="nano"):
    return built.projection(mode).memory_facts.get("host_sections_addressed", [])


@pytest.mark.parametrize("mode", ["max", "low", "nano"])
def test_working_reserve_is_independent_of_reply_allowance_at_initial_fit(tmp_path, mode):
    captured = core(tmp_path)
    raw = plan(captured, 0, mode=mode).projection(mode).estimated_tokens
    window = (raw + 16_384) * 4 // 3 + 10  # root start room participates once
    ordinary = plan(captured, window, mode=mode)
    route = plan(captured, window, mode=mode, facts=RouteFacts(
        conversation_window_tokens=window, accepted_input_tokens=window * 2, working_reserve_tokens=24_000))
    assert not addressed(ordinary, mode)
    assert addressed(route, mode)
    assert route.working_reserve_tokens == 24_000
    assert route.output_reserve_tokens == ordinary.output_reserve_tokens == 16_384
    assert route.window_tokens == window


def test_working_reserve_reaches_the_memory_floor_without_host_sections(tmp_path):
    snapshot = syn.snapshot(story=syn.pages(3, 55_000))
    captured = replace(core(tmp_path), dynamic_text="", dynamic_sections=(),
                       memory_view_json=memory_view.snapshot_json(snapshot))
    ordinary = plan(captured, 70_000)
    route = plan(captured, 70_000, facts=RouteFacts(
        conversation_window_tokens=70_000, working_reserve_tokens=24_000))
    assert ordinary.nano_projection.memory_facts["floor"]["steps"] == {}
    assert route.nano_projection.memory_facts["floor"]["steps"]
    assert route.nano_projection.estimated_tokens < ordinary.nano_projection.estimated_tokens


@pytest.mark.parametrize("basis", ["provider", "estimator"])
def test_unknown_conversation_window_keeps_host_sources_whole(tmp_path, monkeypatch, basis):
    monkeypatch.setattr(fit, "_route_calibration_ratio", lambda *_a, **_kw: 4.0)
    built = plan(core(tmp_path), 0, facts=RouteFacts(accepted_input_tokens=100_000,
                                                  working_reserve_tokens=24_000, basis=basis))
    assert not addressed(built)
    assert not (tmp_path / "task_results" / "artifacts").exists()
    measure = fit.measure_main_fit(built, built.messages_for("nano"), [], profile="owner_nano",
        rendered_mode="nano", round_id="r1", accepted_input_tokens=100_000).measurement
    assert measure.capacity_total_tokens is None
    assert measure.response_reserve_tokens == 24_000


@pytest.mark.parametrize("basis", ["provider", "estimator"])
@pytest.mark.parametrize("density,cutoff", [(0.25, 0.5), (4.0, 2.0)])
def test_window_basis_applies_before_the_first_memory_selection(tmp_path, monkeypatch, basis, density, cutoff):
    captured = core(tmp_path)
    raw = plan(captured, 0).nano_projection.estimated_tokens
    window = math.ceil(raw * cutoff) + 8192
    monkeypatch.setattr(fit, "main_output_reserve_tokens", lambda **_kw: 8192)
    monkeypatch.setattr(fit, "_route_calibration_ratio", lambda *_a, **_kw: density)
    projections, calls = fit._view_projections, []

    def record(*args, **kwargs):
        calls.append(kwargs["ratio"])
        return projections(*args, **kwargs)

    monkeypatch.setattr(fit, "_view_projections", record)
    built = plan(captured, window, facts=RouteFacts(conversation_window_tokens=window, basis=basis))
    expected_ratio = density if basis == "provider" else 1.0
    assert calls == [expected_ratio]
    assert built.nano_projection.calibration_ratio == expected_ratio
    assert bool(addressed(built)) == (expected_ratio > cutoff)
    if expected_ratio <= cutoff:
        assert not (tmp_path / "task_results" / "artifacts").exists()


@pytest.mark.parametrize("basis,density", [("provider", 0.5), ("provider", 2.0), ("estimator", 4.0)])
def test_route_reprojection_keeps_working_room_and_declared_units(tmp_path, basis, density):
    captured = core(tmp_path)
    built = plan(captured, 2_000_000, facts=RouteFacts(
        conversation_window_tokens=2_000_000, working_reserve_tokens=24_000, basis=basis))
    raw = built.nano_projection.estimated_tokens
    expected_ratio = density if basis == "provider" else 1.0
    window = math.ceil(raw * expected_ratio) + 24_000
    same = built.reproject_for_route(window_tokens=window, known_window=True, ratio=density,
        output_reserve=16_384, tool_schemas=[], current_messages=built.messages_for("nano"))
    assert same.messages_for("nano") == built.messages_for("nano")
    assert same.nano_projection.calibration_ratio == expected_ratio
    smaller = built.reproject_for_route(window_tokens=window - 1, known_window=True, ratio=density,
        output_reserve=16_384, tool_schemas=[], current_messages=built.messages_for("nano"))
    assert addressed(smaller)
    assert smaller.working_reserve_tokens == 24_000
    assert smaller.window_tokens == window - 1


@pytest.mark.parametrize("basis", ["provider", "estimator"])
@pytest.mark.parametrize("accepted", [20_000, 100_000])
@pytest.mark.parametrize("anchored", [False, True])
def test_measurement_keeps_conversation_capacity_and_uses_its_counter(
        tmp_path, monkeypatch, basis, accepted, anchored):
    window = 50_000
    built = plan(core(tmp_path), window, facts=RouteFacts(
        conversation_window_tokens=window, accepted_input_tokens=accepted,
        working_reserve_tokens=24_000, basis=basis))
    monkeypatch.setattr(fit, "estimate_context_prompt_tokens", lambda *_a, **_kw: 20_000)
    monkeypatch.setattr("ouroboros.capability_evidence.resolve_main_token_density",
                        lambda *_a, **_kw: (2.0, "fresh_route_usage"))
    result = fit.measure_main_fit(built, built.messages_for("nano"), [], profile="owner_nano",
        rendered_mode="nano", round_id="r1", accepted_input_tokens=accepted,
        anchor=InputAnchor(40_000, 20_000, 1) if anchored else None)
    measure = result.measurement
    assert measure.capacity_total_tokens == measure.window_reading_tokens == window
    assert measure.accepted_input_tokens == accepted
    assert measure.response_reserve_tokens == 24_000
    assert measure.reply_allowance_tokens <= 16_384
    assert measure.measurement_density == (2.0 if basis == "provider" else 1.0)
    assert measure.estimated_input_tokens == (40_000 if basis == "provider" else 20_000)
    assert result.predicted_capacity_miss == (basis == "provider")


@pytest.mark.parametrize("accepted,capacity", [(20_000, 50_000), (100_000, 108_192)])
def test_ordinary_measurement_keeps_reserve_and_density_with_accepted_input_lower_bound(
        tmp_path, monkeypatch, accepted, capacity):
    built = plan(core(tmp_path), 50_000)
    monkeypatch.setattr(fit, "estimate_context_prompt_tokens", lambda *_a, **_kw: 20_000)
    measure = fit.measure_main_fit(built, built.messages_for("nano"), [], profile="owner_nano",
        rendered_mode="nano", round_id="r1", accepted_input_tokens=accepted,
        anchor=InputAnchor(40_000, 20_000, 1)).measurement
    assert measure.response_reserve_tokens == 8192
    assert measure.estimated_input_tokens == 40_000
    assert measure.capacity_total_tokens == capacity


@pytest.mark.parametrize("mode", ["max", "low", "nano"])
@pytest.mark.parametrize("window", [0, 50_000, 2_000_000])
def test_ordinary_projection_matches_equivalent_declared_route(tmp_path, monkeypatch, mode, window):
    """The same units and reserve use one ladder; an impossible core names the same failure."""
    monkeypatch.setattr(fit, "_route_calibration_ratio", lambda *_a, **_kw: 1.5)
    captured = replace(core(tmp_path, memory=True), section_root="", dynamic_sections=())
    facts = RouteFacts(conversation_window_tokens=window or None, acceptance_can_raise=True)
    if window == 50_000:
        for route_facts in (None, facts):
            with pytest.raises(fit.MinimumViewUnfit):
                plan(captured, window, mode=mode, facts=route_facts)
        return
    ordinary = plan(captured, window, mode=mode)
    declared = plan(captured, window, mode=mode, facts=facts)
    assert ordinary.initial_mode == declared.initial_mode
    for selected in ("max", "low", "nano"):
        assert ordinary.messages_for(selected) == declared.messages_for(selected)
        assert ordinary.projection(selected).memory_facts == declared.projection(selected).memory_facts


def test_reprojection_keeps_accepted_input_evidence_when_mode_reserve_changes(tmp_path, monkeypatch):
    from ouroboros import capability_evidence as ce
    from ouroboros.context_window import record_observation

    monkeypatch.setattr(ce, "canonical_evidence_root", lambda: tmp_path)
    built = plan(core(tmp_path), 133_554, mode="max", facts=RouteFacts(
        133_554, 133_554, 10_000, source="accepted lower bound", acceptance_can_raise=True))
    record_observation(tmp_path, built.window_record.route_fp, accepted=150_000)
    refit = built.reproject_for_route(window_tokens=built.window_tokens, known_window=True,
        ratio=1, output_reserve=16_384, tool_schemas=[], start_mode="nano")
    assert refit.window_record.input.tokens == refit.window_record.usable == 150_000
    assert refit.window_tokens == 160_000 and refit.working_reserve_tokens == 10_000


@pytest.mark.parametrize("window,basis,raiseable,expected", [
    (92_000, "provider", False, 92_000),
    (80_000, "estimator", False, 80_000),
    (133_554, "provider", True, 160_000),
])
def test_window_authority_receipt_preserves_live_route_policy(
        tmp_path, monkeypatch, window, basis, raiseable, expected):
    from ouroboros import capability_evidence as ce
    from ouroboros.context_view_facts import window_projection
    from ouroboros.context_window import record_observation, resolve_window

    monkeypatch.setattr(ce, "canonical_evidence_root", lambda: tmp_path)
    built = plan(core(tmp_path), window, facts=RouteFacts(
        conversation_window_tokens=window, working_reserve_tokens=10_000,
        basis=basis, acceptance_can_raise=raiseable))
    scope = built.window_record.route_fp
    # Even on a raiseable route, a numbered refusal cannot raise a declaration.
    record_observation(tmp_path, scope, refused=200_000)
    assert resolve_window(built, 10_000, root=tmp_path).total.tokens == window
    record_observation(tmp_path, scope, accepted=150_000)
    result = fit.measure_main_fit(built, built.messages_for("nano"), [], profile="owner_nano",
        rendered_mode="nano", round_id="authority-merge").measurement
    record = resolve_window(built, 10_000, root=tmp_path)
    assert result.capacity_total_tokens == expected
    assert result.response_reserve_tokens == 10_000
    assert result.window == window_projection(record)
    assert result.window["reserve"] == 10_000
    assert result.window["kind"] == ("input" if raiseable else "total")
    assert result.window_observation_fp == scope
    if basis == "estimator":
        assert result.measurement_density == 1.0
