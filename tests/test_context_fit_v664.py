"""Ordinary-Main target-versus-capacity fit regressions."""

from __future__ import annotations

import inspect
import json
import math
from types import SimpleNamespace

import pytest

from ouroboros.context_budget import RECLAIM_LOW_WATER_DIVISOR


def _projection(mode: str):
    from ouroboros.context_fit import ContextFitProjection

    return ContextFitProjection(
        mode=mode,
        system_content_json=json.dumps(f"{mode.upper()}_SYSTEM"),
        estimated_tokens=10,
        calibrated_tokens=10,
        calibration_ratio=1.0,
        fits_known_window=None,
    )


def _plan(*, preferred: str = "max", window: int = 0, known: bool = False):
    from ouroboros.context_fit import ContextFitPlan

    return ContextFitPlan(
        core_sha256="a" * 64,
        preferred_mode=preferred,
        initial_mode=preferred,
        model="openai/test-model",
        provider="openai",
        route_fp="route-a",
        status="confirmed" if known else "unprobeable",
        stale=False,
        window_tokens=window,
        output_reserve_tokens=65_536,
        user_content_json=json.dumps("go"),
        max_projection=_projection("max"),
        low_projection=_projection("low"),
    )


def _measure(monkeypatch, tmp_path, *, plan, profile, mode, messages, used=False, density=1.0):
    from ouroboros import capability_evidence
    from ouroboros.context_fit import measure_main_fit

    monkeypatch.setattr(
        capability_evidence,
        "resolve_main_token_density",
        lambda *_a, **_kw: (density, "fresh_route_usage" if density != 1.0 else "cold_estimate"),
    )
    return measure_main_fit(
        plan,
        messages,
        [],
        drive_root=tmp_path,
        profile=profile,
        rendered_mode=mode,
        round_id="exec:round:1",
        automatic_pass_used=used,
    )


def test_unknown_owner_max_stays_max_and_sends(monkeypatch, tmp_path):
    plan = _plan()
    disposition = _measure(
        monkeypatch,
        tmp_path,
        plan=plan,
        profile="owner_max",
        mode="max",
        messages=plan.messages_for("max"),
    )
    assert plan.initial_mode == "max"
    assert disposition.action == "send"
    assert disposition.measurement.capacity_total_tokens is None
    assert disposition.measurement.target_total_tokens is None


def test_known_owner_max_miss_reclaims_without_predicted_low(monkeypatch, tmp_path):
    plan = _plan(window=70_000, known=True)
    disposition = _measure(
        monkeypatch,
        tmp_path,
        plan=plan,
        profile="owner_max",
        mode="max",
        messages=plan.messages_for("max") + [{"role": "user", "content": "x" * 40_000}],
    )
    assert plan.initial_mode == "max"
    assert disposition.action == "reclaim_once"
    assert disposition.predicted_capacity_miss is True
    assert disposition.measurement.capacity_deficit_tokens > 0
    assert disposition.measurement.target_deficit_tokens is None


def test_owner_low_uses_elastic_target_then_target_miss(monkeypatch, tmp_path):
    plan = _plan(preferred="low", window=500_000, known=True)
    messages = plan.messages_for("low") + [{"role": "user", "content": "x" * 900_000}]
    first = _measure(
        monkeypatch, tmp_path, plan=plan, profile="owner_low", mode="low", messages=messages,
    )
    after = _measure(
        monkeypatch, tmp_path, plan=plan, profile="owner_low", mode="low", messages=messages,
        used=True,
    )
    assert first.action == "reclaim_once"
    assert first.measurement.target_deficit_tokens > 0
    assert first.measurement.capacity_deficit_tokens == 0
    assert after.action == "send_target_miss"
    assert after.predicted_capacity_miss is False


def test_confirmed_window_below_target_wins_even_when_target_fits(monkeypatch, tmp_path):
    plan = _plan(preferred="low", window=65_550, known=True)
    disposition = _measure(
        monkeypatch,
        tmp_path,
        plan=plan,
        profile="owner_low",
        mode="low",
        messages=plan.messages_for("low"),
    )
    measurement = disposition.measurement
    assert measurement.target_deficit_tokens == 0
    assert measurement.capacity_deficit_tokens > 0
    # Deficit-triggered, low-water-sized: the goal carries an eighth of the BINDING
    # boundary (the 65,550 window, not the 250K target) above the deficit.
    assert measurement.low_water_margin_tokens == math.ceil(65_550 / RECLAIM_LOW_WATER_DIVISOR)
    assert measurement.reclaim_goal_tokens == (
        measurement.capacity_deficit_tokens + measurement.low_water_margin_tokens
    )
    assert disposition.action == "reclaim_once"


@pytest.mark.parametrize("profile,preferred,window,known,boundary", [
    ("owner_max", "max", 70_000, True, 70_000),  # capacity binds a Max route
    ("owner_low", "low", 500_000, True, 250_000),  # the economy target binds Low
    ("owner_low", "low", 0, False, 250_000),  # unknown capacity: the target alone binds
])
def test_low_water_margin_follows_the_binding_boundary(
    monkeypatch, tmp_path, profile, preferred, window, known, boundary,
):
    plan = _plan(preferred=preferred, window=window, known=known)
    messages = plan.messages_for(preferred) + [{"role": "user", "content": "x" * 900_000}]
    disposition = _measure(
        monkeypatch, tmp_path, plan=plan, profile=profile, mode=preferred, messages=messages,
    )
    measurement = disposition.measurement
    deficit = max(value for value in (
        measurement.target_deficit_tokens, measurement.capacity_deficit_tokens,
    ) if value is not None)
    assert deficit > 0
    assert measurement.low_water_margin_tokens == math.ceil(boundary / RECLAIM_LOW_WATER_DIVISOR)
    assert measurement.reclaim_goal_tokens == deficit + measurement.low_water_margin_tokens
    assert disposition.action == "reclaim_once"
    # The margin sizes the pass; it never re-arms the route+round latch.
    after = _measure(
        monkeypatch, tmp_path, plan=plan, profile=profile, mode=preferred, messages=messages,
        used=True,
    )
    assert after.action != "reclaim_once"
    assert after.measurement.low_water_margin_tokens == measurement.low_water_margin_tokens


@pytest.mark.parametrize("profile,preferred,window,known", [
    ("owner_max", "max", 0, False),
    ("owner_max", "max", 500_000, True),
    ("owner_low", "low", 500_000, True),
    ("task_local_low", "low", 500_000, True),
])
def test_no_deficit_means_no_margin_and_no_goal(monkeypatch, tmp_path, profile, preferred, window, known):
    plan = _plan(preferred=preferred, window=window, known=known)
    disposition = _measure(
        monkeypatch, tmp_path, plan=plan, profile=profile, mode=preferred,
        messages=plan.messages_for(preferred),
    )
    assert disposition.measurement.low_water_margin_tokens == 0
    assert disposition.measurement.reclaim_goal_tokens == 0
    assert disposition.action == "send"


def test_low_water_divisor_is_the_one_knob_in_both_directions(monkeypatch, tmp_path):
    from ouroboros import context_budget as cb

    plan = _plan(window=70_000, known=True)
    messages = plan.messages_for("max") + [{"role": "user", "content": "x" * 40_000}]
    base = _measure(
        monkeypatch, tmp_path, plan=plan, profile="owner_max", mode="max", messages=messages,
    ).measurement
    deficit = base.capacity_deficit_tokens
    assert deficit > 0 and base.low_water_margin_tokens == math.ceil(70_000 / 8)

    monkeypatch.setattr(cb, "RECLAIM_LOW_WATER_DIVISOR", 4)
    wider = _measure(
        monkeypatch, tmp_path, plan=plan, profile="owner_max", mode="max", messages=messages,
    ).measurement
    assert wider.low_water_margin_tokens == math.ceil(70_000 / 4)
    assert wider.reclaim_goal_tokens == deficit + wider.low_water_margin_tokens

    monkeypatch.setattr(cb, "RECLAIM_LOW_WATER_DIVISOR", 10 ** 9)  # the margin removed
    removed = _measure(
        monkeypatch, tmp_path, plan=plan, profile="owner_max", mode="max", messages=messages,
    ).measurement
    assert removed.low_water_margin_tokens == 1
    assert removed.reclaim_goal_tokens == deficit + 1
    assert removed.capacity_deficit_tokens == deficit  # the trigger never moved


def test_task_local_low_does_not_inherit_owner_economy_target(monkeypatch, tmp_path):
    plan = _plan(window=500_000, known=True)
    disposition = _measure(
        monkeypatch,
        tmp_path,
        plan=plan,
        profile="task_local_low",
        mode="low",
        messages=plan.messages_for("low") + [{"role": "user", "content": "x" * 600_000}],
    )
    assert disposition.measurement.target_total_tokens is None
    assert disposition.measurement.target_deficit_tokens is None
    assert disposition.measurement.capacity_deficit_tokens == 0
    assert disposition.action == "send"


def test_density_is_applied_once_on_the_declared_basis(monkeypatch, tmp_path):
    plan = _plan(preferred="low", window=500_000, known=True)
    messages = plan.messages_for("low") + [{"role": "assistant", "reasoning": "r" * 20_000}]
    neutral = _measure(
        monkeypatch, tmp_path, plan=plan, profile="owner_low", mode="low", messages=messages,
    )
    dense = _measure(
        monkeypatch, tmp_path, plan=plan, profile="owner_low", mode="low", messages=messages,
        density=2.0,
    )
    assert dense.measurement.estimated_input_tokens == 2 * neutral.measurement.estimated_input_tokens
    assert dense.measurement.measurement_basis == "fresh_route_usage"


def test_complete_shape_estimator_counts_reasoning_and_bounds_image_payloads():
    from ouroboros.context_fit import estimate_context_prompt_tokens

    base = [{"role": "assistant", "content": "ok", "tool_call_id": "call-1"}]
    rich = [{
        **base[0],
        "reasoning": "r" * 40_000,
        "reasoning_details": [{"type": "reasoning.summary", "summary": "s" * 20_000}],
        "content": [{"type": "thinking", "thinking": "t" * 20_000}],
        "response_id": "resp-1",
    }]
    assert estimate_context_prompt_tokens(rich) > estimate_context_prompt_tokens(base) + 15_000

    image_small = [{"role": "user", "content": [{
        "type": "image_url", "image_url": {"url": "data:image/png;base64," + "a" * 10_000},
    }]}]
    image_huge = [{"role": "user", "content": [{
        "type": "image_url", "image_url": {"url": "data:image/png;base64," + "a" * 1_000_000},
    }]}]
    assert estimate_context_prompt_tokens(image_small) == estimate_context_prompt_tokens(image_huge)

    visible = [{"role": "assistant", "content": [{"type": "text", "text": "summary"}]}]
    with_capsule = [{"role": "assistant", "content": [{
        "type": "text",
        "text": "summary",
        "_context_capsule": {"source_hashes": ["x" * 64] * 10_000},
    }]}]
    assert estimate_context_prompt_tokens(with_capsule) == estimate_context_prompt_tokens(visible)
    tool_schema = [{"function": {"parameters": {
        "type": "object",
        "properties": {"_context_capsule": {"type": "string", "description": "x" * 20_000}},
    }}}]
    assert estimate_context_prompt_tokens(visible, tool_schema) > estimate_context_prompt_tokens(visible)


def test_reproject_transcript_replaces_only_system_view():
    plan = _plan()
    dialogue = [
        {"role": "system", "content": "MAX_SYSTEM"},
        {"role": "user", "content": "owner"},
        {"role": "assistant", "content": "work"},
    ]
    low = plan.reproject_transcript(dialogue, "low")
    assert low[0] == plan.low_projection.system_message()
    assert low[1:] == dialogue[1:]


def test_route_rebind_keeps_owner_projection_on_small_confirmed_route(monkeypatch, tmp_path):
    from ouroboros import context, loop
    from ouroboros.tools.registry import ToolRegistry

    plan = _plan()
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.task_id = "switch"
    registry._ctx.event_queue = None
    monkeypatch.setattr(
        context,
        "_context_fit_route",
        lambda *_a, **_kw: (
            {"model": "small", "provider": "openrouter", "base_url": "", "use_local": False},
            SimpleNamespace(
                status="confirmed", stale=False, window_tokens=70_000, route_fp="small-route",
            ),
        ),
    )
    messages = plan.messages_for("max") + [{"role": "user", "content": "x" * 100_000}]
    rebound, mode = loop._rebind_context_fit_plan(
        plan,
        registry,
        messages,
        model="small",
        use_local=False,
        preferred_mode="max",
        tool_schemas=[],
    )
    assert mode == "max"
    assert messages[0] == rebound.max_projection.system_message()
    assert rebound.route_fp == "small-route"


@pytest.mark.parametrize("window,estimated,expected_fit", [
    (80_000, 35_000, True),
    (500_000, 40_000, False),
])
def test_route_rebind_preserves_nano_on_model_account_fallback(
    monkeypatch, tmp_path, window, estimated, expected_fit,
):
    from dataclasses import replace

    from ouroboros import context, context_fit, loop
    from ouroboros.tools.registry import ToolRegistry

    plan = replace(
        _plan(preferred="nano"),
        nano_projection=replace(_projection("nano"), estimated_tokens=estimated),
    )
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.task_id = "nano-fallback"
    registry._ctx.event_queue = None
    monkeypatch.setattr(
        context,
        "_context_fit_route",
        lambda task, **_kw: (
            {"model": task["model"], "provider": "claudexor", "use_local": False},
            SimpleNamespace(
                status="confirmed", stale=False, window_tokens=window,
                route_fp="fallback-account-route", source_id="codex",
                credential_profile_id="account-b", account_fingerprint="acct-fp",
            ),
        ),
    )
    monkeypatch.setattr(context_fit, "_route_calibration_ratio", lambda *_a, **_kw: 2.0)
    messages = plan.messages_for("nano")

    rebound, mode = loop._rebind_context_fit_plan(
        plan,
        registry,
        messages,
        model="claudexor::codex=fallback-model",
        use_local=False,
        preferred_mode="nano",
        tool_schemas=[],
        model_role="fallback:0",
        credential_profile_id="account-b",
    )

    assert mode == "nano"
    assert rebound.preferred_mode == rebound.initial_mode == "nano"
    assert rebound.nano_projection is not None
    assert rebound.nano_projection.calibrated_tokens == estimated * 2
    assert rebound.nano_projection.fits_known_window is expected_fit
    assert messages[0] == rebound.nano_projection.system_message()
    assert rebound.model == "claudexor::codex=fallback-model"
    assert rebound.model_route["credentialProfileId"] == "account-b"


def test_route_rebind_a_b_a_forgets_the_old_a_cache_split(monkeypatch, tmp_path):
    from ouroboros import context, loop, usage_accounting
    from ouroboros.tools.registry import ToolRegistry

    plan = _plan()
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.task_id = "switch-back"
    registry._ctx.event_queue = None
    monkeypatch.setattr(
        context,
        "_context_fit_route",
        lambda task, **_kw: (
            {"model": task["model"], "provider": "openrouter", "use_local": False},
            SimpleNamespace(
                status="confirmed", stale=False, window_tokens=500_000,
                route_fp=f"route-{task['model']}",
            ),
        ),
    )
    monkeypatch.setattr(
        usage_accounting,
        "estimate_cost_optional",
        lambda _model, _prompt, _completion, *, cache_usage, **_kw: float(
            cache_usage["cache_write_tokens"]
        ),
    )
    request = usage_accounting.AttemptRequest(
        model="anthropic/model-a", provider="openrouter", task_id="switch-back",
        prompt_tokens_estimate=1_000,
    )
    usage_accounting.stash_task_cache_split(
        "switch-back", request.model, 800, provider=request.provider, ttl_seconds=300,
    )
    assert usage_accounting._reservation_cost(request) == 200

    messages = plan.messages_for("max")
    plan, _ = loop._rebind_context_fit_plan(
        plan, registry, messages, model="anthropic/model-b", use_local=False,
        preferred_mode="max", tool_schemas=[],
    )
    loop._rebind_context_fit_plan(
        plan, registry, messages, model=request.model, use_local=False,
        preferred_mode="max", tool_schemas=[],
    )

    assert usage_accounting._reservation_cost(request) == 1_000


def test_route_switch_without_immutable_core_fails_loudly(tmp_path):
    from ouroboros import loop
    from ouroboros.tools.registry import ToolRegistry

    with pytest.raises(RuntimeError, match="CONTEXT_FIT_REBUILD_FAILED"):
        loop._rebind_context_fit_plan(
            None,
            ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path),
            [{"role": "user", "content": "go"}],
            model="new-model",
            use_local=False,
            preferred_mode="max",
            tool_schemas=[],
        )


def test_p3_commit_and_scope_review_do_not_use_task_context_fit():
    from ouroboros import review_substrate
    from ouroboros.tools import review, scope_review

    source = (
        inspect.getsource(review_substrate.ReviewCoordinator._run_slot)
        + inspect.getsource(review._run_unified_review)
        + inspect.getsource(scope_review._call_scope_llm)
    )
    assert "run_llm_loop" not in source
    assert "ContextFitPlan" not in source
