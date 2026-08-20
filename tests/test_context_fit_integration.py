"""Cross-stream golden cases for Main fit, reclaim, and physical disclosure."""

from __future__ import annotations

import json


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


def _tool_unit(size: int):
    return [{
        "role": "assistant",
        "content": "investigating",
        "tool_calls": [{
            "id": "call-1",
            "type": "function",
            "function": {"name": "read_file", "arguments": "x" * size},
        }],
    }, {
        "role": "tool",
        "tool_call_id": "call-1",
        "content": "y" * size,
    }]


def test_owner_low_deficit_reclaim_remeasures_on_one_basis(monkeypatch, tmp_path):
    from ouroboros import capability_evidence, context_compaction as cc
    from ouroboros.context_budget import ContextReclaimRequest
    from ouroboros.context_fit import measure_main_fit

    monkeypatch.setattr(
        capability_evidence,
        "resolve_main_token_density",
        lambda *_a, **_kw: (1.0, "fresh_route_usage"),
    )
    monkeypatch.setattr(
        cc,
        "_summarizer_spec",
        lambda: {
            "model": "summary-model", "resolved_model": "summary-model",
            "provider": "test", "route_fp": "summary-route", "effort": "low",
            "output_budget": 32_768, "use_local": False,
        },
    )
    checkpoints = []
    monkeypatch.setattr(
        cc,
        "_persist_reclaim_checkpoint",
        lambda *_a, **_kw: checkpoints.append(True) or {
            "path": "checkpoint", "sha256": "c" * 64,
        },
    )
    summaries = []

    def summarize(parts, **_kwargs):
        summaries.append(tuple(part.source_id for part in parts))
        return {part.source_id: "condensed verified evidence" for part in parts}

    monkeypatch.setattr(cc, "_call_summarizer", summarize)
    plan = _plan()
    messages = [*plan.messages_for("low"), *_tool_unit(350_000)]
    first = measure_main_fit(
        plan, messages, [], drive_root=tmp_path, profile="owner_low",
        rendered_mode="low", round_id="exec:round:1",
    )
    assert first.action == "reclaim_once"
    assert first.measurement.target_deficit_tokens > 0
    assert first.measurement.capacity_deficit_tokens == 0

    request = ContextReclaimRequest(
        route_fp=first.measurement.route_fp,
        round_id=first.measurement.round_id,
        transcript_sha256=cc.context_reclaim_transcript_sha256(messages),
        measurement_basis=first.measurement.measurement_basis,
        measurement_density=first.measurement.measurement_density,
        reclaim_goal_tokens=first.measurement.reclaim_goal_tokens,
    )
    rebuilt, receipt, _usage = cc.compact_tool_history_llm(
        messages, request=request, drive_root=tmp_path, negative_memo=set(),
    )
    after = measure_main_fit(
        plan, rebuilt, [], drive_root=tmp_path, profile="owner_low",
        rendered_mode="low", round_id="exec:round:1", automatic_pass_used=True,
    )

    assert checkpoints == [True]
    assert len(summaries) == 1
    assert receipt.status == "applied"
    assert receipt.reclaimed_tokens > 0
    assert after.automatic_pass_used is True
    assert after.action == "send"
    assert after.measurement.reclaim_goal_tokens == 0


def test_target_miss_is_non_terminal_fit_evidence(monkeypatch, tmp_path):
    from ouroboros import capability_evidence
    from ouroboros import loop_model_call
    from ouroboros.context_fit import measure_main_fit

    monkeypatch.setattr(
        capability_evidence,
        "resolve_main_token_density",
        lambda *_a, **_kw: (1.0, "fresh_route_usage"),
    )
    plan = _plan()
    messages = [*plan.messages_for("low"), {"role": "user", "content": "x" * 600_000}]
    disposition = measure_main_fit(
        plan, messages, [], drive_root=tmp_path, profile="owner_low",
        rendered_mode="low", round_id="exec:round:1", automatic_pass_used=True,
    )
    assert disposition.action == "send_target_miss"

    usage = {}
    ctx = type("Ctx", (), {"accumulated_usage": usage})()
    loop_model_call._remember_main_fit(ctx, disposition)
    assert usage["_context_target_miss"] is True
    assert "execution_status" not in usage
    assert "reason_code" not in usage

    max_disposition = measure_main_fit(
        _plan(preferred="max", window=1_000_000),
        _plan(preferred="max", window=1_000_000).messages_for("max"),
        [], drive_root=tmp_path, profile="owner_max", rendered_mode="max",
        round_id="exec:round:2",
    )
    loop_model_call._remember_main_fit(ctx, max_disposition)
    assert usage["_context_target_miss"] is False


def test_bare_env_low_keeps_p3_owner_max_but_gets_main_target(monkeypatch, tmp_path):
    from ouroboros import capability_evidence, config
    from ouroboros import loop_model_call
    from ouroboros.context_fit import measure_main_fit
    from ouroboros.tools import scope_review

    monkeypatch.setenv("OUROBOROS_CONTEXT_MODE", "low")
    monkeypatch.delenv("OUROBOROS_CONTEXT_MODE_AUTO_LOW", raising=False)
    monkeypatch.setattr(
        capability_evidence,
        "resolve_main_token_density",
        lambda *_a, **_kw: (1.0, "cold_estimate"),
    )
    plan = _plan(preferred="low")

    assert config.get_context_mode() == "low"
    assert config.get_owner_context_mode() == "max"
    assert scope_review._scope_review_skipped_in_low_context() is False
    assert loop_model_call._main_context_profile(plan, "low") == "owner_low"
    fit = measure_main_fit(
        plan,
        plan.messages_for("low"),
        [],
        drive_root=tmp_path,
        profile=loop_model_call._main_context_profile(plan, "low"),
        rendered_mode="low",
        round_id="exec:round:1",
    )
    assert fit.measurement.target_total_tokens == 200_000

    monkeypatch.setenv("OUROBOROS_CONTEXT_MODE_AUTO_LOW", "false")
    assert config.get_owner_context_mode() == "low"
    assert scope_review._scope_review_skipped_in_low_context() is True


def test_round_fit_reads_density_from_canonical_store_not_child_drive(tmp_path, monkeypatch):
    """One observation store: a forked/child task with its own empty drive must
    consume the SAME density witnesses settlement writes into the canonical host
    root — never reset to cold 1.0 by its local empty drive."""
    from types import SimpleNamespace

    canonical = tmp_path / "canonical"
    child = tmp_path / "child-drive"
    (canonical / "state").mkdir(parents=True)
    (child / "logs").mkdir(parents=True)
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(canonical))

    from ouroboros import loop
    from ouroboros import loop_model_call
    from ouroboros.capability_evidence import (
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
    disposition = loop_model_call._measure_round_main_fit(ctx, automatic_pass_used=False)
    assert disposition is not None
    assert disposition.measurement.measurement_basis == "fresh_route_usage"
    assert abs(disposition.measurement.measurement_density - 1.8) < 1e-6
