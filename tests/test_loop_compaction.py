"""Serial Main fit, reclaim, and actual-overflow recovery orchestration."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace


def _fit(
    *,
    action="send",
    profile="owner_max",
    mode="max",
    goal=0,
    target_deficit=None,
    capacity_deficit=None,
    used=False,
):
    from ouroboros.context_fit import MainFitDisposition, MainFitMeasurement

    measurement = MainFitMeasurement(
        route_fp="route-a",
        round_id="exec:round:1",
        profile=profile,
        rendered_mode=mode,
        estimated_input_tokens=120_000,
        response_reserve_tokens=65_536,
        target_total_tokens=200_000 if profile == "owner_low" else None,
        capacity_total_tokens=500_000,
        measurement_basis="cold_estimate",
        measurement_density=1.0,
        target_deficit_tokens=target_deficit,
        capacity_deficit_tokens=capacity_deficit,
        reclaim_goal_tokens=goal,
    )
    return MainFitDisposition(
        measurement=measurement,
        action=action,
        automatic_pass_used=used,
        predicted_capacity_miss=bool(capacity_deficit),
    )


def _plan(*, preferred="max"):
    def reproject(messages, mode):
        rebuilt = list(messages)
        rebuilt[0] = {"role": "system", "content": f"{mode.upper()}_SYSTEM"}
        return rebuilt

    return SimpleNamespace(
        model="same-model",
        route_fp="route-a",
        preferred_mode=preferred,
        core_sha256="a" * 64,
        reproject_transcript=reproject,
    )


def _ctx(tmp_path, *, preferred="max", mode="max"):
    from ouroboros import loop

    inner = SimpleNamespace(
        messages=[],
        active_context_mode=mode,
        task_metadata={},
        _context_reclaim_passes=set(),
        _context_overflow_retries=set(),
        _context_reclaim_negative_memo=set(),
    )
    tools = SimpleNamespace(_ctx=inner)
    messages = [
        {"role": "system", "content": f"{mode.upper()}_SYSTEM"},
        {"role": "user", "content": "go"},
    ]
    inner.messages = messages
    return loop._RoundModelCallContext(
        llm=SimpleNamespace(),
        messages=messages,
        tools=tools,
        context_fit_plan=_plan(preferred=preferred),
        active_model="same-model",
        tool_schemas=[],
        active_effort="high",
        max_retries=3,
        drive_logs=tmp_path / "logs",
        task_id="task",
        round_idx=1,
        event_queue=None,
        accumulated_usage={"execution_id": "exec"},
        task_type="task",
        active_use_local=False,
        active_context_mode=mode,
        drive_root=tmp_path,
    )


def _failed_capture(*, profile="owner_max", mode="max", reserve=65_536, size=1_000):
    from ouroboros.usage_accounting import PhysicalAttemptCapture, PhysicalAttemptContext

    physical = PhysicalAttemptContext(
        profile=profile,
        rendered_mode=mode,
        measurement_basis="cold_estimate",
        route_fp="route-a",
        round_id="exec:round:1",
        target_total_tokens=200_000 if profile == "owner_low" else None,
        capacity_total_tokens=500_000,
        context_target_miss=False,
        automatic_pass_used=False,
    )
    return PhysicalAttemptCapture(
        attempt_id="failed",
        model="same-model",
        provider="openai",
        state="unresolved",
        candidate_measurement_kind="canonical_json_v1",
        max_completion_tokens=reserve,
        candidate_raw_sha256="old",
        candidate_raw_size_bytes=size + 100,
        candidate_context_sha256="old-context",
        candidate_context_size_bytes=size,
        physical_context=physical,
    )


def _candidate_request(disposition, *, size, reserve=65_536):
    from ouroboros import loop
    from ouroboros.usage_accounting import AttemptRequest

    return AttemptRequest(
        model="same-model",
        provider="openai",
        max_completion_tokens=reserve,
        candidate_raw_sha256="new",
        candidate_raw_size_bytes=size + 100,
        candidate_context_sha256="new-context",
        candidate_context_size_bytes=size,
        candidate_measurement_kind="canonical_json_v1",
        physical_context=loop._physical_context_for_fit(disposition),
    )


def test_fallback_success_preserves_complete_candidate_fit_facts(tmp_path):
    from ouroboros import loop

    candidate_plan = SimpleNamespace(route_fp="fallback-route")
    messages = [{"role": "user", "content": "primary"}]
    fallback_messages = [{"role": "user", "content": "fallback"}]
    inner = SimpleNamespace()
    usage = {
        "_context_route_fp": "fallback-route",
        "_context_prompt_estimate": 123,
        "_context_profile": "task_local_low",
        "_context_measurement_basis": "fresh_route_usage",
        "_context_measurement_density": 1.7,
        "_context_target_miss": False,
        "_context_automatic_pass_used": True,
    }
    expected = dict(usage)
    loop._adopt_fallback_route(
        SimpleNamespace(active_model="primary"), SimpleNamespace(_ctx=inner),
        "fallback", False, messages, fallback_messages, candidate_plan, "low", [], usage,
    )
    assert usage == expected
    assert messages == fallback_messages
    assert inner.context_fit_plan is candidate_plan


def test_failed_fallback_restores_entire_primary_fit_bundle():
    from ouroboros import loop

    usage = {
        "_context_route_fp": "primary",
        "_context_profile": "owner_max",
        "_context_measurement_basis": "cold_estimate",
        "_context_target_total_tokens": None,
        "unrelated": "kept",
    }
    snapshot = loop._snapshot_context_fit_usage(usage)
    usage.update({
        "_context_route_fp": "rejected",
        "_context_profile": "task_local_low",
        "_context_target_miss": True,
        "_context_reclaim_goal_tokens": 999,
    })
    loop._restore_context_fit_usage(usage, snapshot)
    assert loop._snapshot_context_fit_usage(usage) == snapshot
    assert usage["unrelated"] == "kept"


def test_checkpoint_proves_automatic_materializer_attempt_even_on_binding_mismatch(
    tmp_path, monkeypatch,
):
    from ouroboros import loop
    from ouroboros.context_budget import ContextReclaimReceipt

    context = _ctx(tmp_path, preferred="low", mode="low")
    disposition = _fit(
        action="reclaim_once", profile="owner_low", mode="low", goal=100,
        target_deficit=100,
    )
    receipt = ContextReclaimReceipt(
        status="binding_mismatch",
        before_transcript_sha256="a" * 64,
        after_transcript_sha256="a" * 64,
        selection_fingerprint="b" * 64,
        selected_unit_ids=("unit",),
        reclaimed_tokens=0,
        goal_reached=False,
        checkpoint_ref={"path": "checkpoint", "sha256": "c" * 64},
        capsule_refs=(),
    )
    monkeypatch.setattr(
        loop, "compact_tool_history_llm",
        lambda *_a, **_kw: (context.messages, receipt, {"prompt_tokens": 1}),
    )
    monkeypatch.setattr(loop, "_account_compaction_usage", lambda *_a, **_kw: None)
    monkeypatch.setattr(loop, "_emit_checkpoint_event", lambda *_a, **_kw: None)
    loop._run_main_reclaim(context, disposition)
    assert ("route-a", "exec:round:1") in context.tools._ctx._context_reclaim_materializations


def test_predicted_reclaim_runs_once_then_sends_target_miss(tmp_path, monkeypatch):
    from ouroboros import loop

    context = _ctx(tmp_path, preferred="low", mode="low")
    fits = iter([
        _fit(action="reclaim_once", profile="owner_low", mode="low", goal=10_000,
             target_deficit=10_000),
        _fit(action="send_target_miss", profile="owner_low", mode="low", goal=4_000,
             target_deficit=4_000, used=True),
    ])

    def measure(ctx, **_kwargs):
        disposition = next(fits)
        loop._remember_main_fit(ctx, disposition)
        return disposition

    reclaimed = []

    def reclaim(ctx, disposition, **_kwargs):
        reclaimed.append(disposition.measurement.reclaim_goal_tokens)
        ctx.tools._ctx._context_reclaim_passes.add(("route-a", "exec:round:1"))

    dispatched = []

    def dispatch(_ctx, disposition, **kwargs):
        dispatched.append((disposition, kwargs))
        return {"role": "assistant", "content": "ok", "tool_calls": []}, 0.0

    monkeypatch.setattr(loop, "_measure_round_main_fit", measure)
    monkeypatch.setattr(loop, "_run_main_reclaim", reclaim)
    monkeypatch.setattr(loop, "_dispatch_round_model", dispatch)
    msg, _cost, mode = loop._call_round_model(context)
    assert msg["content"] == "ok"
    assert mode == "low"
    assert reclaimed == [10_000]
    assert len(dispatched) == 1
    assert dispatched[0][0].action == "send_target_miss"
    assert context.accumulated_usage["_context_target_miss"] is True


def test_actual_max_overflow_reprojects_and_retries_only_smaller_context(tmp_path, monkeypatch):
    from ouroboros import loop

    context = _ctx(tmp_path)
    fits = iter([
        _fit(),
        _fit(profile="task_local_low", mode="low"),
        _fit(profile="task_local_low", mode="low", used=True),
    ])

    def measure(ctx, **_kwargs):
        disposition = next(fits)
        loop._remember_main_fit(ctx, disposition)
        return disposition

    def reclaim(ctx, _disposition, **kwargs):
        assert kwargs["minimum_goal_tokens"] == 1
        ctx.tools._ctx._context_reclaim_passes.add(("route-a", "exec:round:1"))

    sends = []

    def dispatch(ctx, disposition, *, candidate_predicate=None, **_kwargs):
        if not sends:
            sends.append("failed-main")
            ctx.accumulated_usage["_last_llm_error_kind"] = "context_overflow"
            return None, 0.0
        request = _candidate_request(disposition, size=800)
        assert candidate_predicate is not None and candidate_predicate(request) is True
        sends.append("smaller-retry")
        return {"role": "assistant", "content": "fits", "tool_calls": []}, 0.0

    monkeypatch.setattr(loop, "_measure_round_main_fit", measure)
    monkeypatch.setattr(loop, "_run_main_reclaim", reclaim)
    monkeypatch.setattr(loop, "_dispatch_round_model", dispatch)
    monkeypatch.setattr(loop, "last_physical_attempt_capture", lambda: _failed_capture())
    msg, _cost, mode = loop._call_round_model(context)
    assert msg["content"] == "fits"
    assert sends == ["failed-main", "smaller-retry"]
    assert mode == "low"
    assert context.messages[0]["content"] == "LOW_SYSTEM"
    assert ("route-a", "exec:round:1") in context.tools._ctx._context_overflow_retries


def test_equal_context_releases_retry_before_provider_send(tmp_path, monkeypatch):
    from ouroboros import loop
    from ouroboros.usage_accounting import PhysicalAttemptPreconditionFailed

    context = _ctx(tmp_path)
    fits = iter([
        _fit(),
        _fit(profile="task_local_low", mode="low"),
        _fit(profile="task_local_low", mode="low", used=True),
    ])
    events = []
    provider_sends = 0

    def measure(ctx, **_kwargs):
        disposition = next(fits)
        loop._remember_main_fit(ctx, disposition)
        return disposition

    def reclaim(ctx, _disposition, **_kwargs):
        ctx.tools._ctx._context_reclaim_passes.add(("route-a", "exec:round:1"))

    def dispatch(ctx, disposition, *, candidate_predicate=None, **_kwargs):
        nonlocal provider_sends
        if provider_sends == 0:
            provider_sends += 1
            ctx.accumulated_usage["_last_llm_error_kind"] = "context_overflow"
            return None, 0.0
        if not candidate_predicate(_candidate_request(disposition, size=1_000)):
            raise PhysicalAttemptPreconditionFailed("not smaller")
        provider_sends += 1
        raise AssertionError("equal context must not dispatch")

    monkeypatch.setattr(loop, "_measure_round_main_fit", measure)
    monkeypatch.setattr(loop, "_run_main_reclaim", reclaim)
    monkeypatch.setattr(loop, "_dispatch_round_model", dispatch)
    monkeypatch.setattr(loop, "last_physical_attempt_capture", lambda: _failed_capture())
    monkeypatch.setattr(loop, "_emit_checkpoint_event", lambda *_a, **kw: events.append(kw or _a[-1]))
    msg, _cost, mode = loop._call_round_model(context)
    assert msg is None
    assert mode == "low"
    assert provider_sends == 1
    assert any(event.get("reason") == "context_candidate_not_strictly_smaller" for event in events)


def test_already_low_overflow_never_emits_max_to_low_toast(tmp_path, monkeypatch):
    from ouroboros import loop

    context = _ctx(tmp_path, preferred="low", mode="low")
    fits = iter([
        _fit(profile="owner_low", mode="low"),
        _fit(profile="owner_low", mode="low"),
        _fit(profile="owner_low", mode="low", used=True),
    ])
    events = []
    calls = 0

    def measure(ctx, **_kwargs):
        disposition = next(fits)
        loop._remember_main_fit(ctx, disposition)
        return disposition

    def reclaim(ctx, _disposition, **_kwargs):
        ctx.tools._ctx._context_reclaim_passes.add(("route-a", "exec:round:1"))

    def dispatch(ctx, disposition, *, candidate_predicate=None, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            ctx.accumulated_usage["_last_llm_error_kind"] = "context_overflow"
            return None, 0.0
        assert candidate_predicate(_candidate_request(disposition, size=700))
        return {"role": "assistant", "content": "fits", "tool_calls": []}, 0.0

    monkeypatch.setattr(loop, "_measure_round_main_fit", measure)
    monkeypatch.setattr(loop, "_run_main_reclaim", reclaim)
    monkeypatch.setattr(loop, "_dispatch_round_model", dispatch)
    monkeypatch.setattr(
        loop, "last_physical_attempt_capture",
        lambda: _failed_capture(profile="owner_low", mode="low"),
    )
    monkeypatch.setattr(loop, "_emit_checkpoint_event", lambda *_a, **kw: events.append(kw or _a[-1]))
    msg, _cost, mode = loop._call_round_model(context)
    assert msg["content"] == "fits"
    assert mode == "low"
    assert not any(event.get("checkpoint_kind") == "context_fit_low_retry" for event in events)


def test_one_route_round_materialization_pass_is_latched(tmp_path, monkeypatch):
    from ouroboros import loop
    from ouroboros.context_budget import ContextReclaimReceipt

    context = _ctx(tmp_path, preferred="low", mode="low")
    disposition = _fit(
        action="reclaim_once", profile="owner_low", mode="low", goal=10,
        target_deficit=10,
    )
    calls = 0

    def compact(messages, **_kwargs):
        nonlocal calls
        calls += 1
        receipt = ContextReclaimReceipt(
            status="no_eligible",
            before_transcript_sha256="a" * 64,
            after_transcript_sha256="a" * 64,
            selection_fingerprint="",
            selected_unit_ids=(),
            reclaimed_tokens=0,
            goal_reached=False,
            checkpoint_ref=None,
            capsule_refs=(),
        )
        return messages, receipt, None

    monkeypatch.setattr(loop, "compact_tool_history_llm", compact)
    loop._run_main_reclaim(context, disposition)
    assert loop._run_main_reclaim(context, disposition) is None
    assert calls == 1


def test_strict_shrink_predicate_uses_failed_physical_reserve():
    from ouroboros import loop

    failed = _failed_capture(reserve=2_048)
    predicate = loop._strict_context_shrink_predicate(failed)
    disposition = _fit(profile="task_local_low", mode="low")
    assert predicate(_candidate_request(disposition, size=900, reserve=65_536)) is False
    assert predicate(_candidate_request(disposition, size=900, reserve=2_048)) is True


def test_failed_main_capture_is_snapshotted_before_reclaim_attempt(tmp_path, monkeypatch):
    """A receipted summarizer must not replace the failed Main comparison candidate."""
    from ouroboros import loop

    context = _ctx(tmp_path)
    fits = iter([
        _fit(),
        _fit(profile="task_local_low", mode="low"),
        _fit(profile="task_local_low", mode="low", used=True),
    ])
    current_capture = {"value": _failed_capture(size=1_000)}

    def measure(ctx, **_kwargs):
        disposition = next(fits)
        loop._remember_main_fit(ctx, disposition)
        return disposition

    def reclaim(ctx, _disposition, **_kwargs):
        current_capture["value"] = _failed_capture(size=700)
        ctx.tools._ctx._context_reclaim_passes.add(("route-a", "exec:round:1"))

    sends = 0

    def dispatch(ctx, disposition, *, candidate_predicate=None, **_kwargs):
        nonlocal sends
        sends += 1
        if sends == 1:
            ctx.accumulated_usage["_last_llm_error_kind"] = "context_overflow"
            return None, 0.0
        request = _candidate_request(disposition, size=800)
        assert candidate_predicate is not None
        assert candidate_predicate(request) is True  # smaller than failed Main (1000)
        assert loop._strict_context_shrink_predicate(current_capture["value"])(request) is False
        return {"role": "assistant", "content": "fits", "tool_calls": []}, 0.0

    monkeypatch.setattr(loop, "_measure_round_main_fit", measure)
    monkeypatch.setattr(loop, "_run_main_reclaim", reclaim)
    monkeypatch.setattr(loop, "_dispatch_round_model", dispatch)
    monkeypatch.setattr(loop, "last_physical_attempt_capture", lambda: current_capture["value"])

    msg, _cost, mode = loop._call_round_model(context)
    assert msg["content"] == "fits"
    assert mode == "low"
    assert sends == 2


def test_strict_shrink_predicate_requires_entire_physical_tuple():
    from ouroboros import loop

    failed = _failed_capture(size=1_000)
    predicate = loop._strict_context_shrink_predicate(failed)
    disposition = _fit(profile="task_local_low", mode="low")
    accepted = _candidate_request(disposition, size=900)
    assert predicate(accepted) is True

    assert accepted.physical_context is not None
    variants = {
        "provider": replace(accepted, provider="anthropic"),
        "model": replace(accepted, model="other-model"),
        "reserve": replace(accepted, max_completion_tokens=2_048),
        "route": replace(
            accepted,
            physical_context=replace(accepted.physical_context, route_fp="other-route"),
        ),
        "round": replace(
            accepted,
            physical_context=replace(accepted.physical_context, round_id="exec:round:2"),
        ),
        "raw_digest": replace(accepted, candidate_raw_sha256=failed.candidate_raw_sha256),
        "equal_context": replace(accepted, candidate_context_size_bytes=1_000),
        "larger_context": replace(accepted, candidate_context_size_bytes=1_001),
    }
    for label, candidate in variants.items():
        assert predicate(candidate) is False, label


def test_overflow_retry_is_skipped_while_the_round_holds_an_unresolved_attempt(tmp_path, monkeypatch):
    """Through the REAL dispatcher: a granted transport-death repeat that is
    then rejected as a context overflow must not open the compaction retry —
    attempt #1 of the round is still unresolved, so a smaller candidate would
    be a third paid send over it. The round record is the single fact."""
    import json

    import httpx

    from ouroboros import loop, loop_llm_call
    from ouroboros import usage_accounting as ua
    from ouroboros.loop_llm_call import TRANSPORT_DEATHS_KEY

    def _unresolved(**extra):
        return ua.PhysicalAttemptCapture(
            attempt_id="pa", model="same-model", provider="openrouter", state="unresolved",
            candidate_measurement_kind="opaque", **extra,
        )

    class _DeathThenOverflow:
        calls = 0

        def chat(self, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                try:
                    raise RuntimeError("Connection error.") from httpx.ReadError("socket died")
                except RuntimeError as exc:
                    exc.physical_attempt_capture = _unresolved()
                    raise
            exc = RuntimeError("routed to a smaller upstream window")
            exc.code = "context_length_exceeded"
            exc.physical_attempt_capture = _unresolved(provider_status_code=400, provider_code="context_length_exceeded")
            raise exc

    context = _ctx(tmp_path)
    context.llm = _DeathThenOverflow()
    fits = iter([_fit(), _fit(profile="task_local_low", mode="low")])
    reclaims = []

    def measure(ctx, **_kwargs):
        disposition = next(fits)
        loop._remember_main_fit(ctx, disposition)
        return disposition

    monkeypatch.setattr(loop, "_measure_round_main_fit", measure)
    monkeypatch.setattr(loop, "_run_main_reclaim", lambda ctx, d, **_k: reclaims.append(d))
    monkeypatch.setattr(loop, "last_physical_attempt_capture", lambda: _failed_capture())
    monkeypatch.setattr(loop_llm_call, "_sleep_within_deadline", lambda _sec, _dl, **_kw: True)

    msg, _cost, mode = loop._call_round_model(context)

    assert msg is None
    assert context.llm.calls == 2  # the death and its one repeat; no compaction retry
    assert context.accumulated_usage["_last_llm_error_kind"] == "context_overflow"
    assert context.accumulated_usage[TRANSPORT_DEATHS_KEY]["count"] == 1
    assert reclaims == []
    assert ("route-a", "exec:round:1") not in context.tools._ctx._context_overflow_retries
    assert mode == "max"  # the retry was refused before any low re-projection
    rows = [json.loads(line) for line in (tmp_path / "logs" / "events.jsonl").read_text().splitlines() if line.strip()]
    skipped = [row for row in rows if row.get("type") == "context_overflow_retry_skipped"]
    assert [row["reason"] for row in skipped] == ["round_holds_unresolved_attempt"]
    api_errors = [row for row in rows if row.get("type") == "llm_api_error"]
    assert [(row["error_kind"], row["retry_same_request"]) for row in api_errors] == [
        ("provider_outcome_unknown", True), ("context_overflow", False),
    ]
