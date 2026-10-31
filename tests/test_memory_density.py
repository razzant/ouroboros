"""A view's startup density can only be learned from requests of that view."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
from tests._memory_presence_seams import wake_cause_seam as wake_cause_seam

from ouroboros import capability_evidence as ce
from ouroboros.context_fit import measure_main_fit, _physical_context_for_fit
from ouroboros.usage_accounting import AttemptRequest
from tests.test_context_fit_integration import _plan


@pytest.mark.parametrize("kind", ["own_wake", "focus", "owner_turn"])
def test_density_never_borrows_another_view_on_route_or_model(tmp_path, kind):
    for other in {"own_wake", "focus", "owner_turn", ""} - {kind}:
        ce.record_token_density(tmp_path, "model", prompt_chars=80_000, prompt_tokens=30_000,
                                route_fp="same-route", basis=ce.MAIN_DENSITY_BASIS, view_kind=other)
    for route in ("same-route", "another-account"):
        assert ce.resolve_main_token_density(tmp_path, route, "model", view_kind=kind) == (1.0, "cold_estimate")
    ce.record_token_density(tmp_path, "model", prompt_chars=80_000, prompt_tokens=25_000,
                            route_fp="same-route", basis=ce.MAIN_DENSITY_BASIS, view_kind=kind)
    assert ce.resolve_main_token_density(tmp_path, "same-route", "model", view_kind=kind) == (1.25, "fresh_route_usage")
    # Another route of the same model and view keeps its own basis: no model-wide borrowing (THESIS I-8, C1).
    assert ce.resolve_main_token_density(tmp_path, "another-account", "model", view_kind=kind) == (1.0, "cold_estimate")


def test_throttle_and_retention_are_independent_per_view(tmp_path):
    for kind in ("focus", "owner_turn", "own_wake"):
        ce.record_token_density(tmp_path, "model", prompt_chars=80_000, prompt_tokens=25_000,
                                route_fp="route", basis=ce.MAIN_DENSITY_BASIS, view_kind=kind)
    for number in range(8):
        ce.record_token_density(tmp_path, "model", prompt_chars=80_000, prompt_tokens=30_000 + number * 1_000,
                                route_fp=f"focus-{number}", basis=ce.MAIN_DENSITY_BASIS, view_kind="focus")
    assert ce.resolve_main_token_density(tmp_path, "route", "model", view_kind="own_wake")[0] == 1.25
    assert ce.resolve_main_token_density(tmp_path, "route", "model", view_kind="owner_turn")[0] == 1.25
    pairs = ce._load(tmp_path)["token_density"]["model"]["pairs"]
    assert sum(pair["view_kind"] == "focus" for pair in pairs) == ce._TOKEN_DENSITY_MAX_PAIRS


def test_fit_to_physical_settlement_preserves_view_kind(tmp_path, monkeypatch):
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    plan = replace(_plan(preferred="max"), view_kind="focus")
    result = measure_main_fit(plan, plan.messages_for("max"), [], profile="owner_max", rendered_mode="max", round_id="round1")
    physical = _physical_context_for_fit(result)
    assert physical.view_kind == result.measurement.view_kind == "focus"
    request = AttemptRequest(model=plan.model, provider="openai", drive_root=tmp_path,
                             physical_context=physical, prompt_tokens_bounded_estimate=20_000)
    ce.observe_token_density(request, {"prompt_tokens": 25_000}, drive_root_resolver=lambda root: root)
    assert ce.resolve_main_token_density(tmp_path, plan.route_fp, plan.model, view_kind="focus")[0] == 1.25
    assert ce.resolve_main_token_density(tmp_path, plan.route_fp, plan.model, view_kind="own_wake") == (1.0, "cold_estimate")


@pytest.mark.parametrize("metadata,kind", [
    ({}, "owner_turn"),
    ({"usage_category": "consciousness", "wake_cause": "own_clock"}, "own_wake"),
    ({"usage_category": "consciousness", "wake_cause": "focus"}, "focus"),
])
def test_startup_plan_names_the_view_before_selecting_a_density(tmp_path, monkeypatch, metadata, kind):
    from ouroboros import context_fit

    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(context_fit, "reference_doc_sections", lambda *a, **kw: ([], ""))
    core = context_fit.ContextCore(base_prompt="p", bible_md="b", architecture_md="a", development_md="d",
                                  semi_stable_text="s", dynamic_text="y", user_content_json='"hello"',
                                  docs_need_development=False)
    seen = []
    monkeypatch.setattr(context_fit, "_route_calibration_ratio", lambda *a, **kw: seen.append(kw["view_kind"]) or 1.0)
    resolver = lambda *a, **kw: ({"provider": "openai", "model": "model", "base_url": "", "use_local": False},
                                 SimpleNamespace(route_fp="route", status="confirmed", stale=False, window_tokens=500_000))
    plan = context_fit.build_context_fit_plan(SimpleNamespace(drive_root=tmp_path), core, {"metadata": metadata},
                                              preferred_mode="max", route_resolver=resolver)
    assert plan.view_kind == kind and seen == [kind]
