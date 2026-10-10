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
    messages = [*plan.messages_for("low"), *_tool_unit(450_000)]
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


def _growth_unit(tag: str, chars: int):
    return [{
        "role": "assistant",
        "content": "investigating",
        "tool_calls": [{
            "id": f"call-{tag}",
            "type": "function",
            "function": {"name": "read_file", "arguments": "x" * chars},
        }],
    }, {"role": "tool", "tool_call_id": f"call-{tag}", "content": "y" * chars}]


def _install_full_budget_materializer(monkeypatch):
    """A13 worst case: every summary comes back at the FULL summary budget of its
    source (never a tiny summary), the private checkpoint is stubbed, density 1.0."""
    from ouroboros import capability_evidence, context_compaction as cc

    monkeypatch.setattr(
        capability_evidence, "resolve_main_token_density",
        lambda *_a, **_kw: (1.0, "fresh_route_usage"),
    )
    monkeypatch.setattr(cc, "_summarizer_spec", lambda: {
        "model": "summary-model", "resolved_model": "summary-model", "provider": "test",
        "route_fp": "summary-route", "effort": "low", "output_budget": 32_768, "use_local": False,
    })
    monkeypatch.setattr(
        cc, "_persist_reclaim_checkpoint",
        lambda *_a, **_kw: {"path": "checkpoint", "sha256": "c" * 64},
    )
    monkeypatch.setattr(cc, "_call_summarizer", lambda parts, *, summary_budgets, **_kw: {
        part.source_id: "s" * (4 * int(summary_budgets[part.root_id])) for part in parts
    })


def _simulate_growing_transcript(tmp_path, *, window: int, rounds: int, growth_chars: int):
    """Owner Max on a known ``window``: fill to just under the capacity boundary, then
    grow one completed tool unit per round, running the REAL fit and the REAL
    materializer the way the loop does (at most one automatic pass per route+round,
    the landing re-measured on the same basis). Returns (unit tokens, per-pass rows
    of (round, requested margin, achieved headroom, receipt))."""
    from ouroboros import context_compaction as cc
    from ouroboros.context_budget import ContextReclaimRequest
    from ouroboros.context_fit import estimate_context_prompt_tokens, measure_main_fit

    plan = _plan(preferred="max", window=window)
    messages = plan.messages_for("max")
    unit_tokens = estimate_context_prompt_tokens(_growth_unit("probe", growth_chars))
    boundary_input = window - plan.output_reserve_tokens
    filler = (boundary_input - estimate_context_prompt_tokens(messages) - unit_tokens // 2) // unit_tokens
    for index in range(filler):
        messages = messages + _growth_unit(f"f{index}", growth_chars)

    def fit(current, round_idx, *, used):
        return measure_main_fit(
            plan, current, [], drive_root=tmp_path, profile="owner_max", rendered_mode="max",
            round_id=f"exec:round:{round_idx}", automatic_pass_used=used,
        )

    memo: set = set()
    rows = []
    for round_idx in range(1, rounds + 1):
        messages = messages + _growth_unit(f"g{round_idx}", growth_chars)
        before = fit(messages, round_idx, used=False)
        if before.action != "reclaim_once":
            assert before.action == "send"
            continue
        measurement = before.measurement
        request = ContextReclaimRequest(
            route_fp=measurement.route_fp, round_id=measurement.round_id,
            transcript_sha256=cc.context_reclaim_transcript_sha256(messages),
            measurement_basis=measurement.measurement_basis,
            measurement_density=measurement.measurement_density,
            reclaim_goal_tokens=measurement.reclaim_goal_tokens,
        )
        messages, receipt, _usage = cc.compact_tool_history_llm(
            messages, request=request, drive_root=tmp_path, negative_memo=memo,
        )
        after = fit(messages, round_idx, used=True).measurement
        rows.append((
            round_idx, measurement.low_water_margin_tokens,
            window - (after.estimated_input_tokens + after.response_reserve_tokens), receipt,
        ))
    return unit_tokens, rows


def test_low_water_margin_bounds_automatic_passes_under_a_full_budget_summarizer(monkeypatch, tmp_path):
    """Anti-thrash class. A pass sized to the deficit alone lands AT the boundary, so the
    next round's ordinary growth re-arms it: one summarizer pass nearly every round.
    Sized deficit + boundary/RECLAIM_LOW_WATER_DIVISOR it lands below the boundary and
    the next pass needs real growth. Under the A13 worst-case stub every selected unit
    halves, so a pass lands about HALF the requested margin below (the receipt says
    goal_reached=False): the bound proven with that stub is ceil(N*g / (margin/2)) + 1;
    the ideal-summarizer bound ceil(N*g / margin) + 1 holds for the ACHIEVED headroom and
    is asserted in that form (requested margin is not achieved headroom). With the
    margin removed the same assertions fail."""
    import math

    from ouroboros import context_budget as cb

    _install_full_budget_materializer(monkeypatch)
    window, rounds, chars = 400_000, 40, 4_000
    margin = math.ceil(window / cb.RECLAIM_LOW_WATER_DIVISOR)

    unit_tokens, passes = _simulate_growing_transcript(
        tmp_path, window=window, rounds=rounds, growth_chars=chars,
    )
    growth = rounds * unit_tokens
    bound = math.ceil(growth / (margin // 2)) + 1
    assert 2 <= len(passes) <= bound
    assert [row[1] for row in passes] == [margin] * len(passes)
    assert all(row[3].status == "applied" for row in passes)
    # Every pass reached the boundary; none achieved the full margin (partial shrink
    # under full-budget summaries), and the receipt discloses that underlanding.
    assert all(0 <= row[2] < margin for row in passes)
    assert not any(row[3].goal_reached for row in passes)
    achieved = min(row[2] for row in passes)
    assert len(passes) <= math.ceil(growth / achieved) + 1
    gaps = [later[0] - earlier[0] for earlier, later in zip(passes, passes[1:])]
    assert min(gaps) > achieved // unit_tokens

    monkeypatch.setattr(cb, "RECLAIM_LOW_WATER_DIVISOR", 10 ** 9)  # margin 1: deficit-sized passes
    _unit, thrash = _simulate_growing_transcript(
        tmp_path, window=window, rounds=rounds, growth_chars=chars,
    )
    assert len(thrash) >= rounds - 2
    assert len(thrash) > bound
    assert [row[1] for row in thrash] == [1] * len(thrash)
    # A deficit-sized pass lands at (here: still above) the boundary, which is exactly
    # what re-arms it on the next round.
    assert sum(1 for row in thrash if row[2] < 0) >= len(thrash) - 2


def test_target_miss_is_non_terminal_fit_evidence(monkeypatch, tmp_path):
    from ouroboros import capability_evidence, loop
    from ouroboros.context_fit import measure_main_fit

    monkeypatch.setattr(
        capability_evidence,
        "resolve_main_token_density",
        lambda *_a, **_kw: (1.0, "fresh_route_usage"),
    )
    plan = _plan()
    messages = [*plan.messages_for("low"), {"role": "user", "content": "x" * 900_000}]
    disposition = measure_main_fit(
        plan, messages, [], drive_root=tmp_path, profile="owner_low",
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
        [], drive_root=tmp_path, profile="owner_max", rendered_mode="max",
        round_id="exec:round:2",
    )
    loop._remember_main_fit(ctx, max_disposition)
    assert usage["_context_target_miss"] is False


def test_bare_env_low_keeps_p3_owner_max_but_gets_main_target(monkeypatch, tmp_path):
    from ouroboros import capability_evidence, config, loop
    from ouroboros.context_fit import measure_main_fit

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
    assert loop._main_context_profile(plan, "low") == "owner_low"
    fit = measure_main_fit(
        plan,
        plan.messages_for("low"),
        [],
        drive_root=tmp_path,
        profile=loop._main_context_profile(plan, "low"),
        rendered_mode="low",
        round_id="exec:round:1",
    )
    assert fit.measurement.target_total_tokens == 250_000

    monkeypatch.setenv("OUROBOROS_CONTEXT_MODE_AUTO_LOW", "false")
    assert config.get_owner_context_mode() == "low"


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
    assert disposition.measurement.measurement_basis == "fresh_route_usage"
    assert abs(disposition.measurement.measurement_density - 1.8) < 1e-6
