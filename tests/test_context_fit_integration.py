"""Cross-stream golden cases for Main fit, reclaim, and physical disclosure."""

from __future__ import annotations

import json

from ouroboros.context_anchor import InputAnchor

# This conversation's own anchor at density 1.0 (the estimate is its raw count, measured).
_ANCHOR = InputAnchor(input_tokens=1_000_000, estimate_tokens=1_000_000, round=1)


def _projection(mode: str):
    from ouroboros.context_fit import ContextFitProjection

    return ContextFitProjection(
        mode=mode,
        system_content_json=json.dumps(f"{mode.upper()}_SYSTEM"),
        estimated_tokens=10,
        calibrated_tokens=10,
        calibration_ratio=1.0,
    )


def _plan(*, preferred="low", window=500_000):
    from ouroboros.context_fit import ContextFitPlan

    return ContextFitPlan(
        core_sha256="a" * 64,
        preferred_mode=preferred,
        initial_mode=preferred,
        model="openai/test-model",
        provider="openai",
        route_fp="route-a",
        status="confirmed",
        stale=False,
        window_tokens=window,
        output_reserve_tokens=65_536,
        user_content_json=json.dumps("go"),
        max_projection=_projection("max"),
        low_projection=_projection("low"),
    )


def test_target_miss_is_non_terminal_fit_evidence(monkeypatch, tmp_path):
    from ouroboros import loop
    from ouroboros.context_fit import measure_main_fit

    plan = _plan()
    messages = [*plan.messages_for("low"), {"role": "user", "content": "x" * 900_000}]
    disposition = measure_main_fit(
        plan, messages, [], anchor=_ANCHOR, profile="owner_low",
        rendered_mode="low", round_id="exec:round:1", automatic_pass_used=True,
    )
    assert disposition.action == "send_target_miss"

    usage = {}
    ctx = type("Ctx", (), {"accumulated_usage": usage})()
    loop._remember_main_fit(ctx, disposition)
    assert usage["_context_target_miss"] is True
    assert "execution_status" not in usage
    assert "reason_code" not in usage

    max_disposition = measure_main_fit(
        _plan(preferred="max", window=1_000_000),
        _plan(preferred="max", window=1_000_000).messages_for("max"),
        [], anchor=_ANCHOR, profile="owner_max", rendered_mode="max",
        round_id="exec:round:2",
    )
    loop._remember_main_fit(ctx, max_disposition)
    assert usage["_context_target_miss"] is False


def test_bare_env_low_keeps_p3_owner_max_but_gets_main_target(monkeypatch, tmp_path):
    from ouroboros import config, loop
    from ouroboros.context_fit import measure_main_fit

    monkeypatch.setenv("OUROBOROS_CONTEXT_MODE", "low")
    monkeypatch.delenv("OUROBOROS_CONTEXT_MODE_AUTO_LOW", raising=False)
    plan = _plan(preferred="low")

    assert config.get_context_mode() == "low"
    assert config.get_owner_context_mode() == "max"
    assert loop._main_context_profile(plan, "low") == "owner_low"
    fit = measure_main_fit(
        plan,
        plan.messages_for("low"),
        [],
        profile=loop._main_context_profile(plan, "low"),
        rendered_mode="low",
        round_id="exec:round:1",
    )
    assert fit.measurement.target_total_tokens == 250_000

    monkeypatch.setenv("OUROBOROS_CONTEXT_MODE_AUTO_LOW", "false")
    assert config.get_owner_context_mode() == "low"


def test_round_fit_reads_its_own_conversation_never_the_witness_store(tmp_path, monkeypatch):
    """17.10.2026: another task's witness hid a full window. A fresh exact-route witness
    (density 1.8) in the canonical store leaves the round's estimate cold; this
    conversation's own accepted request (the anchor) is what calibrates it."""
    from types import SimpleNamespace

    canonical = tmp_path / "canonical"
    child = tmp_path / "child-drive"
    (canonical / "state").mkdir(parents=True)
    (child / "logs").mkdir(parents=True)
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(canonical))

    from ouroboros import loop
    from ouroboros.capability_evidence import (
        MAIN_DENSITY_BASIS,
        canonical_evidence_root,
        record_token_density,
    )

    assert canonical_evidence_root() == canonical

    plan = _plan(preferred="low")
    record_token_density(
        canonical_evidence_root(),
        plan.model,
        prompt_chars=400_000,  # above the 20K noise floor
        prompt_tokens=180_000,  # density 1.8
        source="dispatch_usage",
        route_fp=plan.route_fp,
        basis=MAIN_DENSITY_BASIS,
    )

    ctx = loop._RoundModelCallContext(
        llm=None,
        messages=plan.messages_for("low"),
        tools=SimpleNamespace(_ctx=SimpleNamespace()),
        context_fit_plan=plan,
        active_model=plan.model,
        tool_schemas=[],
        active_effort="medium",
        max_retries=1,
        drive_logs=child / "logs",
        task_id="task-canonical-density",
        round_idx=1,
        event_queue=None,
        accumulated_usage={},
        task_type="task",
        active_use_local=False,
        active_context_mode="low",
        drive_root=child,
    )
    disposition = loop._measure_round_main_fit(ctx, automatic_pass_used=False)
    assert disposition is not None
    assert disposition.measurement.measurement_basis == "cold_estimate"
    assert disposition.measurement.measurement_density == 1.0
    from ouroboros.context_anchor import record_input_anchor

    ctx.accumulated_usage.update(_last_round_cache_usage={"prompt_tokens": 130_000, "provider": plan.provider},
                                 _context_raw_input_tokens=100_000)
    record_input_anchor(ctx.accumulated_usage, plan, round_idx=1)
    anchored = loop._measure_round_main_fit(ctx, automatic_pass_used=False)
    assert anchored.measurement.measurement_basis == "fresh_route_usage"
    assert abs(anchored.measurement.measurement_density - 1.3) < 1e-6
