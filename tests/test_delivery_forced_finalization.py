from __future__ import annotations

import json
import queue
from types import SimpleNamespace

from tests._delivery_candidate_shared import (
    write_child as _write_child,
    write_confirmed_disposition_fixture as _write_confirmed_disposition,
)


def _forced_test_context(tmp_path, *, usage=None, incoming=None):
    import ouroboros.loop as loop
    from ouroboros.tools.registry import ToolRegistry

    trace = {"tool_calls": [], "reasoning_notes": []}
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.task_id = "parent1"
    registry._ctx.task_metadata = {
        "budget_drive_root": str(tmp_path),
        "root_task_id": "parent1",
    }
    ctx = loop._RoundLimitContext(
        [{"role": "user", "content": "task"}],
        SimpleNamespace(),
        "test-model",
        "medium",
        1,
        tmp_path / "logs",
        "parent1",
        2,
        None,
        usage if usage is not None else {},
        "",
        False,
        10,
        drive_root=tmp_path,
        incoming_messages=incoming,
        owner_msg_seen=set(),
    )
    loop._finalize_limit_ctx(ctx, registry, trace)
    return loop, registry, ctx, trace


def _bind_host_pass(loop, registry, trace, candidate):
    """Attach one exact authoritative PASS to the current delivery candidate."""

    trace["review_decision"] = {
        "eligibility": "eligible",
        "trigger": "auto_nondirect",
        "panel_id": "panel-accepted",
        "binding_hash": "binding-accepted",
    }
    trace["acceptance_decision"] = {
        "status": "accepted",
        "source": "task_acceptance_review",
        "rationale": "The exact candidate passed host acceptance.",
    }
    run = {
        "request": {
            "surface": "task_acceptance",
            "policy": {"min_successful_slots": 1},
        },
        "actors": [],
        "authority": "host_root",
        "candidate_hash": candidate.content_sha256,
        "panel_id": "panel-accepted",
        "binding_hash": "binding-accepted",
        "evidence_revision": "accepted-evidence",
        "fence_hash": "accepted-fence",
        "aggregate_signal": "PASS",
        "enforcement_impact": "allows_completion",
    }
    trace["review_runs"] = [run]
    candidate.acceptance_binding = loop._delivery_acceptance_binding(
        registry,
        trace,
        candidate.content_sha256,
    )
    registry._ctx._task_acceptance_reviewed = True
    loop._publish_delivery_candidate(registry, candidate, trace)
    return run


def _write_deferred_child(tmp_path):
    _write_child(tmp_path)
    _write_confirmed_disposition(
        tmp_path,
        disposition="deferred",
        rationale="defer until the next run",
    )


def _assert_forced_deferred_outcome(text, usage, trace, reason_code):
    from ouroboros.outcomes import derive_loop_outcome

    projection = trace["child_result_dispositions"]
    assert projection["deferred_count"] == 1
    assert projection["current"][0]["child_task_id"] == "child1"
    assert projection["current"][0]["disposition"] == "deferred"
    outcome = derive_loop_outcome(text, usage, trace)
    assert outcome["outcome_axes"]["execution"]["status"] == "best_effort"
    assert outcome["outcome_axes"]["execution"]["reason_code"] == reason_code
    assert outcome["outcome_axes"]["objective"]["status"] == "best_effort"
    assert outcome["outcome_axes"]["objective"]["source"] == "child_result_disposition"
    assert outcome["outcome_axes"]["objective"]["deferred_count"] == 1


def test_round_limit_projects_deferred_child_before_forced_return(tmp_path, monkeypatch):
    _write_deferred_child(tmp_path)
    loop, _registry, limit_ctx, _trace = _forced_test_context(tmp_path)
    monkeypatch.setattr(
        loop,
        "call_llm_with_retry",
        lambda *_args, **_kwargs: (
            {"role": "assistant", "content": "Best answer before the round limit."},
            0.0,
        ),
    )

    text, usage, returned_trace = loop._handle_round_limit(limit_ctx)

    _assert_forced_deferred_outcome(
        text, usage, returned_trace, "round_limit",
    )


def test_budget_exhaustion_projects_deferred_child_before_forced_return(
    tmp_path, monkeypatch,
):
    import ouroboros.usage_accounting as accounting

    _write_deferred_child(tmp_path)
    loop, registry, limit_ctx, trace = _forced_test_context(tmp_path)
    loop._replace_delivery_candidate(
        registry,
        limit_ctx,
        trace,
        "Best answer retained before the budget rail.",
        control="candidate",
    )
    monkeypatch.setattr(
        accounting,
        "usage_breakdown",
        lambda *_args, **_kwargs: {"physical_calls": 1, "integrity_degraded": False},
    )
    exit_ctx = loop._LoopExitContext(
        tools=registry,
        drive_root=tmp_path,
        task_id="parent1",
        event_queue=None,
        drive_logs=tmp_path / "logs",
        accumulated_usage=limit_ctx.accumulated_usage,
        llm_trace=trace,
    )

    text, usage, returned_trace = loop._handle_budget_exceeded(
        accounting.BudgetExceeded(
            "root budget closed", limit_scope="root", root_task_id="parent1",
        ),
        exit_ctx,
        limit_ctx=limit_ctx,
    )

    _assert_forced_deferred_outcome(
        text, usage, returned_trace, "budget_exhausted",
    )


def test_physical_budget_exit_discloses_stale_candidate_after_service_teardown(
    tmp_path, monkeypatch,
):
    import hashlib

    import ouroboros.usage_accounting as accounting
    from ouroboros.tools import services as services_mod

    loop, registry, limit_ctx, trace = _forced_test_context(tmp_path)
    old = loop._replace_delivery_candidate(
        registry,
        limit_ctx,
        trace,
        "Answer completed before service teardown.",
        control="replace",
    )
    old.acceptance_binding = {
        "candidate_sha256": old.content_sha256,
        "evidence_revision": old.evidence_revision,
        "acceptance_status": "pass",
        "authoritative": True,
        "panel_id": "panel-old",
        "binding_hash": "binding-old",
    }
    loop._publish_delivery_candidate(registry, old, trace)

    monkeypatch.setattr(
        accounting,
        "usage_breakdown",
        lambda *_args, **_kwargs: {"physical_calls": 1, "integrity_degraded": False},
    )
    monkeypatch.setattr(
        services_mod,
        "stop_task_services",
        lambda _ctx: [{
            "service_id": "preview",
            "name": "preview",
            "lifecycle": "stopped",
            "artifact_output_failed": True,
            "artifact_outputs": "report.html is missing",
        }],
    )
    exit_ctx = loop._LoopExitContext(
        tools=registry,
        drive_root=tmp_path,
        task_id="parent1",
        event_queue=None,
        drive_logs=tmp_path / "logs",
        accumulated_usage=limit_ctx.accumulated_usage,
        llm_trace=trace,
    )

    text, usage, returned_trace = loop._handle_budget_exceeded(
        accounting.BudgetExceeded(
            "root budget closed", limit_scope="root", root_task_id="parent1",
        ),
        exit_ctx,
        limit_ctx=limit_ctx,
    )

    candidate = registry._ctx._delivery_candidate
    assert text == candidate.full_text
    assert text.startswith(old.full_text)
    assert "STALE-EVIDENCE NOTICE — RESUME REQUIRED (host)" in text
    assert "does not claim to incorporate it" in text
    assert candidate is not old
    assert candidate.content_sha256 == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert candidate.revision > old.revision
    assert candidate.evidence_revision == old.evidence_revision
    assert candidate.acceptance_binding["acceptance_status"] == "unaccepted"
    assert candidate.acceptance_binding["authoritative"] is False
    assert candidate.acceptance_binding["stale_evidence"] is True
    assert returned_trace["delivery_candidate"]["evidence_current"] is False
    forced = returned_trace["forced_finalization"]
    assert forced["source"] == (
        "budget_stale_candidate_preserved_stale_evidence_resume_required"
    )
    assert forced["evidence_current"] is False
    assert forced["evidence_revision"] == old.evidence_revision
    assert forced["current_evidence_revision"] > old.evidence_revision
    assert returned_trace["verification_events"][0]["kind"] == "services_stopped"
    assert "SERVICE_FINALIZATION_EVIDENCE" in str(limit_ctx.messages)
    assert usage["_best_effort_extracted"] is True
    assert usage["reason_code"] == "budget_exhausted"


def test_budget_dispatch_rail_revises_candidate_for_undispositioned_child_suffix(
    tmp_path, monkeypatch,
):
    import hashlib

    import ouroboros.usage_accounting as accounting

    _write_child(tmp_path, status="running")
    loop, registry, limit_ctx, trace = _forced_test_context(tmp_path)
    original = loop._replace_delivery_candidate(
        registry,
        limit_ctx,
        trace,
        "Complete answer retained before the budget rail.",
        control="replace",
    )
    original.acceptance_binding = {
        "candidate_sha256": original.content_sha256,
        "evidence_revision": original.evidence_revision,
        "acceptance_status": "pass",
        "authoritative": True,
        "panel_id": "panel-old",
        "binding_hash": "binding-old",
    }
    loop._publish_delivery_candidate(registry, original, trace)
    monkeypatch.setattr(
        accounting,
        "usage_breakdown",
        lambda *_args, **_kwargs: {"physical_calls": 1, "integrity_degraded": False},
    )
    exit_ctx = loop._LoopExitContext(
        tools=registry,
        drive_root=tmp_path,
        task_id="parent1",
        event_queue=None,
        drive_logs=tmp_path / "logs",
        accumulated_usage=limit_ctx.accumulated_usage,
        llm_trace=trace,
    )

    text, usage, returned_trace = loop._handle_budget_exceeded(
        accounting.BudgetExceeded(
            "root budget closed", limit_scope="root", root_task_id="parent1",
        ),
        exit_ctx,
        limit_ctx=limit_ctx,
    )

    candidate = registry._ctx._delivery_candidate
    assert text == candidate.full_text
    assert text.startswith(original.full_text)
    assert "child1 [running]" in text
    assert candidate.content_sha256 == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert candidate.content_sha256 != original.content_sha256
    assert candidate.acceptance_binding["acceptance_status"] == "unaccepted"
    assert candidate.acceptance_binding["authoritative"] is False
    assert returned_trace["forced_finalization"]["source"] == "budget_preserve_with_host_suffix"
    assert usage["reason_code"] == "budget_exhausted"


def test_budget_latch_preserves_stale_candidate_with_resume_disclosure(
    tmp_path, monkeypatch,
):
    import hashlib

    import ouroboros.usage_accounting as accounting

    loop, registry, limit_ctx, trace = _forced_test_context(tmp_path)
    answer = "Latched complete answer after verified work."
    old = loop._replace_delivery_candidate(
        registry,
        limit_ctx,
        trace,
        answer,
        control="replace",
    )
    old.acceptance_binding = {
        "candidate_sha256": old.content_sha256,
        "evidence_revision": old.evidence_revision,
        "acceptance_status": "pass",
        "authoritative": True,
        "panel_id": "panel-old",
        "binding_hash": "binding-old",
    }
    loop._publish_delivery_candidate(registry, old, trace)
    loop._latch_final_answer_marker(trace, f"FINAL ANSWER: {answer}")

    # Owner evidence invalidates the candidate without adding a tool call. The
    # unconditional FINAL ANSWER latch remains useful, but its unchanged text
    # must retain its old evidence provenance and carry a loud resume disclosure.
    registry._ctx._owner_directives = [{"content": "Late answer constraint"}]
    monkeypatch.setattr(
        accounting,
        "usage_breakdown",
        lambda *_args, **_kwargs: {"physical_calls": 1, "integrity_degraded": False},
    )
    exit_ctx = loop._LoopExitContext(
        tools=registry,
        drive_root=tmp_path,
        task_id="parent1",
        event_queue=None,
        drive_logs=tmp_path / "logs",
        accumulated_usage=limit_ctx.accumulated_usage,
        llm_trace=trace,
    )

    text, usage, returned_trace = loop._handle_budget_exceeded(
        accounting.BudgetExceeded(
            "root budget closed",
            limit_scope="root",
            root_task_id="parent1",
        ),
        exit_ctx,
        limit_ctx=limit_ctx,
    )

    rebound = registry._ctx._delivery_candidate
    assert text.startswith(answer)
    assert "STALE-EVIDENCE NOTICE — RESUME REQUIRED (host)" in text
    assert "has not been regenerated or accepted" in text
    assert rebound is not old
    assert rebound.content_sha256 == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert rebound.revision > old.revision
    assert rebound.evidence_revision == old.evidence_revision
    assert rebound.degraded is True
    assert rebound.degraded_reason == "budget_exhausted"
    assert rebound.finalization_control == "forced_stale_preserve:budget_exhausted"
    assert rebound.acceptance_binding["acceptance_status"] == "unaccepted"
    assert rebound.acceptance_binding["authoritative"] is False
    assert rebound.acceptance_binding["stale_evidence"] is True
    assert returned_trace["delivery_candidate"]["evidence_current"] is False
    forced = returned_trace["forced_finalization"]
    assert forced["source"] == (
        "budget_latched_fallback_stale_evidence_resume_required"
    )
    assert forced["evidence_current"] is False
    assert forced["evidence_revision"] == old.evidence_revision
    assert forced["current_evidence_revision"] > old.evidence_revision
    assert usage["_best_effort_extracted"] is True
    assert trace["tool_calls"] == []


def test_provider_unavailable_preserves_stale_candidate_with_resume_disclosure(
    tmp_path, monkeypatch,
):
    import hashlib

    loop, registry, limit_ctx, trace = _forced_test_context(tmp_path)
    answer = "Complete answer preserved through provider failure."
    old = loop._replace_delivery_candidate(
        registry,
        limit_ctx,
        trace,
        answer,
        control="replace",
    )
    old.acceptance_binding = {
        "candidate_sha256": old.content_sha256,
        "evidence_revision": old.evidence_revision,
        "acceptance_status": "pass",
        "authoritative": True,
        "panel_id": "panel-old",
        "binding_hash": "binding-old",
    }
    loop._publish_delivery_candidate(registry, old, trace)
    registry._ctx._owner_directives = [{"content": "Late answer constraint"}]

    forced_calls = 0

    def empty_forced_call(*_args, **_kwargs):
        nonlocal forced_calls
        forced_calls += 1
        return None, 0.0

    monkeypatch.setattr(loop, "call_llm_with_retry", empty_forced_call)

    text, usage, returned_trace = loop._handle_provider_unavailable(limit_ctx)

    rebound = registry._ctx._delivery_candidate
    assert forced_calls == 1
    assert text.startswith(answer)
    assert "STALE-EVIDENCE NOTICE — RESUME REQUIRED (host)" in text
    assert "does not claim to incorporate it" in text
    assert rebound is not old
    assert rebound.content_sha256 == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert rebound.revision > old.revision
    assert rebound.evidence_revision == old.evidence_revision
    assert rebound.degraded is True
    assert rebound.degraded_reason == "provider_unavailable"
    assert rebound.finalization_control == "forced_stale_preserve:provider_unavailable"
    assert rebound.acceptance_binding["acceptance_status"] == "unaccepted"
    assert rebound.acceptance_binding["authoritative"] is False
    assert rebound.acceptance_binding["stale_evidence"] is True
    assert returned_trace["delivery_candidate"]["evidence_current"] is False
    forced = returned_trace["forced_finalization"]
    assert forced["source"] == "host_fallback_stale_evidence_resume_required"
    assert forced["evidence_current"] is False
    assert forced["evidence_revision"] == old.evidence_revision
    assert forced["current_evidence_revision"] > old.evidence_revision
    assert usage["_best_effort_extracted"] is True
    assert trace["tool_calls"] == []


def test_unknown_provider_finalization_is_host_only_and_reason_specific(
    tmp_path, monkeypatch,
):
    _write_child(tmp_path, status="running")
    usage = {
        "_last_llm_error": "APIConnectionError('Connection error.')",
        "_last_llm_error_kind": "provider_outcome_unknown",
        "_last_llm_retry_same_request": False,
        "_provider_recovery_actions": [],
    }
    loop, _registry, limit_ctx, _trace = _forced_test_context(tmp_path, usage=usage)

    def forbidden_model_call(*_args, **_kwargs):
        raise AssertionError("unknown provider outcome must not trigger another model call")

    monkeypatch.setattr(loop, "call_llm_with_retry", forbidden_model_call)
    text, returned_usage, returned_trace = loop._handle_provider_unavailable(limit_ctx)

    assert "FINALIZATION NOTICE (host)" in text
    assert "may have reached the provider" in text
    assert "No further replay, same-request retry, or paid fallback was sent" in text
    assert "after retries" not in text
    assert "same-model reroute" not in text
    assert "Finalized because a provider request had an unknown outcome" in text
    assert "hard limit" not in text
    assert "snapshot at finalization time" in text
    assert "not guaranteed to be final outcomes" in text
    assert "`get_task_result(<id>)` / `peek_task(<id>)`" in text
    assert "\\_" not in text
    assert "\\<" not in text
    assert returned_usage["reason_code"] == "provider_unavailable"
    assert returned_trace["forced_finalization"]["source"] == "provider_outcome_unknown_no_resend"


def test_round_limit_orphan_notice_names_real_hard_limit(tmp_path, monkeypatch):
    _write_child(tmp_path, status="running")
    loop, _registry, limit_ctx, _trace = _forced_test_context(tmp_path)
    monkeypatch.setattr(loop, "call_llm_with_retry", lambda *_args, **_kwargs: (None, 0.0))

    text, usage, returned_trace = loop._handle_round_limit(limit_ctx)

    assert "Finalized under the hard round limit" in text
    assert "snapshot at finalization time" in text
    assert "`get_task_result(<id>)` / `peek_task(<id>)`" in text
    assert usage["reason_code"] == "round_limit"
    assert returned_trace["forced_finalization"]["reason_code"] == "round_limit"


def test_unknown_forced_summary_keeps_original_reason_without_children(
    tmp_path, monkeypatch,
):
    loop, registry, ctx, trace = _forced_test_context(tmp_path)
    retained = loop._replace_delivery_candidate(
        registry, ctx, trace, "Retained complete answer.", control="candidate",
    )
    calls = 0

    def unknown_forced_call(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        ctx.accumulated_usage.update({
            "_last_llm_error_kind": "provider_outcome_unknown",
            "_last_llm_error": "connection ended after dispatch",
            "_llm_dispatched_attempts": 1,
        })
        return None, 0.0

    monkeypatch.setattr(loop, "call_llm_with_retry", unknown_forced_call)
    text, _usage, returned_trace = loop._forced_final_answer(
        ctx,
        prompt="finalize",
        fallback_text="round fallback",
        reason_code="round_limit",
    )

    assert calls == 1
    assert text.startswith(retained.full_text)
    assert text.count("Finalized under the hard round limit") == 1
    assert "may have reached the provider" in text
    assert "No further replay, same-request retry, or paid fallback was sent" in text
    assert "child task(s)" not in text
    assert returned_trace["forced_finalization"]["source"] == (
        "provider_outcome_unknown_no_resend_with_host_suffix"
    )


def test_supervisor_finalization_notice_preserves_exact_reason(tmp_path, monkeypatch):
    loop, _registry, limit_ctx, _trace = _forced_test_context(tmp_path)
    monkeypatch.setattr(loop, "call_llm_with_retry", lambda *_args, **_kwargs: (None, 0.0))

    text, usage, returned_trace = loop._handle_forced_finalization(
        limit_ctx, "hard_timeout",
    )

    assert "supervisor finalization request (reason: hard_timeout)" in text
    assert "under a hard limit" not in text
    assert "child task(s)" not in text
    assert usage["reason_code"] == "finalization_grace"
    assert returned_trace["forced_finalization"]["finalization_reason"] == "hard_timeout"


def test_normal_host_suffix_is_inside_candidate_and_panel_subject(tmp_path, monkeypatch):
    import hashlib

    _write_child(tmp_path)
    _write_confirmed_disposition(
        tmp_path,
        disposition="deferred",
        rationale="defer until the next run",
    )
    loop, registry, ctx, trace = _forced_test_context(tmp_path)
    captured = {}

    monkeypatch.setattr(loop, "_compute_subagent_handoff", lambda *_a, **_k: None)
    monkeypatch.setattr(loop, "_maybe_inject_finalization_nudges", lambda *_a, **_k: False)

    def capture_panel(*, content, **_kwargs):
        captured["content"] = content
        return False

    monkeypatch.setattr(loop, "_run_task_acceptance_review_once", capture_panel)
    result = loop._no_tool_final_answer(
        "Base complete answer.",
        ctx,
        trace,
        registry,
        queue.Queue(),
        set(),
        lambda _text: None,
    )

    assert result is not None
    text, _usage, returned_trace = result
    assert text == captured["content"]
    assert text.count("DEFERRED CHILD RESULTS") == 1
    assert returned_trace["delivery_candidate"]["content_sha256"] == hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()
    assert registry._ctx._delivery_candidate.full_text == text


def test_forced_retained_candidate_suffix_creates_new_unaccepted_revision(
    tmp_path, monkeypatch,
):
    import hashlib

    _write_child(tmp_path, status="running")
    loop, registry, ctx, trace = _forced_test_context(tmp_path)
    original = loop._replace_delivery_candidate(
        registry, ctx, trace, "Retained complete answer.", control="candidate",
    )
    original.acceptance_binding = {
        "candidate_sha256": original.content_sha256,
        "acceptance_status": "pass",
        "authoritative": True,
        "panel_id": "old-panel",
        "binding_hash": "old-binding",
    }
    monkeypatch.setattr(loop, "call_llm_with_retry", lambda *_a, **_k: (None, 0.0))

    text, _usage, returned_trace = loop._forced_final_answer(
        ctx,
        prompt="finalize",
        fallback_text="host fallback",
        reason_code="round_limit",
    )

    candidate = registry._ctx._delivery_candidate
    assert text == candidate.full_text
    assert "FINALIZATION NOTICE (host): Finalized under the hard round limit" in text
    assert "NOTE (host): At forced finalization" in text
    assert candidate.revision == original.revision + 1
    assert candidate.content_sha256 == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert candidate.acceptance_binding["acceptance_status"] == "unaccepted"
    assert candidate.acceptance_binding["authoritative"] is False
    assert returned_trace["delivery_candidate"]["content_sha256"] == candidate.content_sha256


def test_forced_finalization_stops_services_before_model_and_binds_evidence(
    tmp_path, monkeypatch,
):
    from ouroboros.tools import services as services_mod

    loop, registry, ctx, trace = _forced_test_context(tmp_path)
    order = []

    def stop_services(_ctx):
        order.append("services")
        return [{
            "service_id": "preview",
            "name": "preview",
            "lifecycle": "stopped",
            "artifact_output_failed": True,
            "artifact_outputs": "report.html is missing: " + ("x" * 9000),
        }]

    def forced_model(_llm, messages, *_args, **_kwargs):
        order.append("model")
        rendered = str(messages)
        assert "SERVICE_FINALIZATION_EVIDENCE" in rendered
        assert "artifact_output_failed" in rendered
        assert "report.html is missing" in rendered
        assert "OMISSION NOTE: truncated at 8000 chars" in rendered
        return {"role": "assistant", "content": "Answer disclosing the missing report."}, 0.0

    monkeypatch.setattr(services_mod, "stop_task_services", stop_services)
    monkeypatch.setattr(loop, "call_llm_with_retry", forced_model)

    text, _usage, returned_trace = loop._forced_final_answer(
        ctx,
        prompt="finalize",
        fallback_text="fallback",
        reason_code="deadline_local",
    )

    assert order == ["services", "model"]
    assert text == registry._ctx._delivery_candidate.full_text
    assert "Finalized because the task deadline was reached" in text
    assert returned_trace["verification_events"][0]["kind"] == "services_stopped"
    assert len(returned_trace["verification_events"][0]["services"][0]["artifact_outputs"]) > 8000
    assert returned_trace["delivery_candidate"]["evidence_current"] is True


def test_forced_model_call_rebinds_latest_child_result_and_suffix(tmp_path, monkeypatch):
    import hashlib

    from ouroboros.task_results import write_task_result
    from ouroboros.task_status import load_effective_task_result
    from ouroboros.tools import services as services_mod
    from ouroboros.tools.join_ledger import _child_result_sha256

    _write_child(tmp_path, status="running")
    initial_child = load_effective_task_result(tmp_path, "child1")
    initial_hash = _child_result_sha256(initial_child)
    loop, registry, ctx, trace = _forced_test_context(tmp_path)
    calls = 0

    def forced_model(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        write_task_result(
            tmp_path,
            "child1",
            "completed",
            parent_task_id="parent1",
            root_task_id="parent1",
            delegation_role="subagent",
            result="latest child result produced during forced synthesis",
            trace_summary="latest trace",
            artifact_status="ready",
            artifacts=[{
                "kind": "report",
                "name": "latest.md",
                "sha256": "b" * 64,
            }],
        )
        return {"role": "assistant", "content": "Forced answer draft."}, 0.0

    monkeypatch.setattr(services_mod, "stop_task_services", lambda _ctx: [])
    monkeypatch.setattr(loop, "call_llm_with_retry", forced_model)

    text, _usage, returned_trace = loop._forced_final_answer(
        ctx,
        prompt="finalize",
        fallback_text="fallback",
        reason_code="round_limit",
    )

    latest_child = load_effective_task_result(tmp_path, "child1")
    latest_hash = _child_result_sha256(latest_child)
    candidate = registry._ctx._delivery_candidate
    evidence_revision, evidence_fingerprint = loop._delivery_evidence_state(
        registry, ctx, trace,
    )
    assert calls == 1
    assert latest_hash != initial_hash
    assert "child1 [completed]" in text
    assert "child1 [running]" not in text
    assert text == candidate.full_text
    assert candidate.content_sha256 == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert candidate.evidence_revision == evidence_revision
    assert candidate.evidence_fingerprint == evidence_fingerprint
    assert loop._current_delivery_candidate(ctx, trace) is candidate
    assert returned_trace["delivery_candidate"]["content_sha256"] == candidate.content_sha256


def test_production_budget_wrapup_routes_through_delivery_candidate(tmp_path, monkeypatch):
    import hashlib

    loop, registry, ctx, trace = _forced_test_context(
        tmp_path, usage={"cost": 5.0},
    )
    monkeypatch.setattr(
        loop,
        "call_llm_with_retry",
        lambda *_a, **_k: ({"role": "assistant", "content": "Budget-bound answer."}, 0.0),
    )

    result = loop._check_budget_limits(
        ctx,
        budget_remaining_usd=8.0,
        cost_ceiling_usd=4.0,
    )

    assert result is not None
    text, usage, returned_trace = result
    assert text == registry._ctx._delivery_candidate.full_text
    assert returned_trace["delivery_candidate"]["content_sha256"] == hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()
    assert usage["reason_code"] == "budget_exhausted"


def test_production_budget_wrapup_propagates_budget_exceeded(tmp_path, monkeypatch):
    import pytest

    import ouroboros.usage_accounting as accounting

    loop, _registry, ctx, trace = _forced_test_context(
        tmp_path, usage={"cost": 5.0},
    )

    def reject_dispatch(*_args, **_kwargs):
        raise accounting.BudgetExceeded(
            "root budget closed", limit_scope="root", root_task_id="parent1",
        )

    monkeypatch.setattr(loop, "call_llm_with_retry", reject_dispatch)
    with pytest.raises(accounting.BudgetExceeded):
        loop._check_budget_limits(
            ctx,
            budget_remaining_usd=8.0,
            cost_ceiling_usd=4.0,
        )


def test_forced_owner_arrival_gets_one_complete_refresh(tmp_path, monkeypatch):
    import hashlib

    incoming = queue.Queue()
    loop, registry, ctx, trace = _forced_test_context(tmp_path, incoming=incoming)
    old = loop._replace_delivery_candidate(
        registry, ctx, trace, "Previously accepted answer.", control="candidate",
    )
    old.acceptance_binding = {
        "candidate_sha256": old.content_sha256,
        "acceptance_status": "pass",
        "authoritative": True,
        "panel_id": "old-panel",
        "binding_hash": "old-binding",
    }
    registry._ctx._task_acceptance_reviewed = True
    trace["review_runs"] = [{
        "authority": "host_root",
        "candidate_hash": old.content_sha256,
        "panel_id": "old-panel",
        "binding_hash": "old-binding",
        "aggregate_signal": "PASS",
    }]
    calls = []

    def forced_model(_llm, messages, *_args, **_kwargs):
        calls.append(str(messages))
        if len(calls) == 1:
            incoming.put("also include the newly requested criterion")
            return {"role": "assistant", "content": "Stale forced draft."}, 0.0
        assert "newly requested criterion" in str(messages)
        assert "FORCED_OWNER_REFRESH" in str(messages)
        return {"role": "assistant", "content": "Refreshed forced answer."}, 0.0

    monkeypatch.setattr(loop, "call_llm_with_retry", forced_model)
    text, _usage, returned_trace = loop._forced_final_answer(
        ctx,
        prompt="finalize",
        fallback_text="fallback",
        reason_code="finalization_grace",
    )

    assert len(calls) == 2
    assert text.startswith("Refreshed forced answer.")
    assert "FINALIZATION NOTICE (host)" in text
    assert "Stale forced draft" not in text
    assert registry._ctx._delivery_candidate.content_sha256 == hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()
    assert len(registry._ctx._owner_directives) == 1
    assert registry._ctx._task_acceptance_reviewed is False
    assert trace["review_runs"][0]["superseded_by_revision"] is True
    assert returned_trace["delivery_candidate"]["acceptance_binding"]["authoritative"] is False


def test_forced_replacement_supersedes_accepted_pass_in_outcome_and_projection(
    tmp_path, monkeypatch,
):
    from ouroboros.outcomes import derive_loop_outcome
    from ouroboros.review_substrate import compact_review_projection

    loop, registry, ctx, trace = _forced_test_context(tmp_path)
    accepted = loop._replace_delivery_candidate(
        registry,
        ctx,
        trace,
        "Previously accepted complete answer.",
        control="candidate",
    )
    prior_run = _bind_host_pass(loop, registry, trace, accepted)
    monkeypatch.setattr(
        loop,
        "call_llm_with_retry",
        lambda *_args, **_kwargs: (
            {"role": "assistant", "content": "Forced replacement answer."},
            0.0,
        ),
    )

    text, usage, returned_trace = loop._forced_final_answer(
        ctx,
        prompt="finalize",
        fallback_text="fallback",
        reason_code="round_limit",
    )

    assert text.startswith("Forced replacement answer.")
    assert "Finalized under the hard round limit" in text
    assert prior_run["superseded_by_revision"] is True
    assert prior_run["superseded_reason"] == "delivery_candidate_replaced"
    assert prior_run["enforcement_impact"] == "requires_revision"
    assert "panel_id" not in returned_trace["review_decision"]
    assert "binding_hash" not in returned_trace["review_decision"]
    assert returned_trace["review_decision"]["eligibility"] == (
        "pending_delivery_acceptance"
    )
    assert returned_trace["delivery_candidate"]["acceptance_binding"][
        "authoritative"
    ] is False

    outcome = derive_loop_outcome(text, usage, returned_trace)
    review = outcome["outcome_axes"]["review"]
    assert review["status"] == "degraded"
    assert review["run_count"] == 0
    assert review["superseded_run_count"] == 1
    assert review["superseded_aggregate_signals"] == ["PASS"]
    assert outcome["outcome_axes"]["objective"]["status"] != "pass"
    projection = compact_review_projection(returned_trace["review_runs"])
    assert projection["panels"][0]["aggregate_signal"] == "PASS"
    assert projection["panels"][0]["superseded"] is True
    assert projection["panels"][0]["enforcement_impact"] == "requires_revision"


def test_stale_preserve_supersedes_accepted_pass_in_outcome_and_projection(
    tmp_path, monkeypatch,
):
    from ouroboros.outcomes import derive_loop_outcome
    from ouroboros.review_substrate import compact_review_projection

    loop, registry, ctx, trace = _forced_test_context(tmp_path)
    accepted = loop._replace_delivery_candidate(
        registry,
        ctx,
        trace,
        "Accepted answer before newer evidence.",
        control="candidate",
    )
    prior_run = _bind_host_pass(loop, registry, trace, accepted)
    trace["tool_calls"].append({
        "tool": "write_file",
        "status": "ok",
        "result": "new evidence",
        "is_error": False,
    })
    monkeypatch.setattr(
        loop,
        "call_llm_with_retry",
        lambda *_args, **_kwargs: (None, 0.0),
    )

    text, usage, returned_trace = loop._forced_final_answer(
        ctx,
        prompt="finalize",
        fallback_text=accepted.full_text,
        reason_code="provider_unavailable",
    )

    assert "STALE-EVIDENCE NOTICE" in text
    assert prior_run["superseded_by_revision"] is True
    assert prior_run["superseded_reason"] == (
        "delivery_evidence_changed_after_host_acceptance"
    )
    assert prior_run["enforcement_impact"] == "requires_revision"
    assert "panel_id" not in returned_trace["review_decision"]
    assert "binding_hash" not in returned_trace["review_decision"]
    binding = returned_trace["delivery_candidate"]["acceptance_binding"]
    assert binding["authoritative"] is False
    assert binding["stale_evidence"] is True

    outcome = derive_loop_outcome(text, usage, returned_trace)
    review = outcome["outcome_axes"]["review"]
    assert review["status"] == "degraded"
    assert review["run_count"] == 0
    assert review["superseded_run_count"] == 1
    assert review["superseded_aggregate_signals"] == ["PASS"]
    assert outcome["outcome_axes"]["objective"]["status"] != "pass"
    projection = compact_review_projection(returned_trace["review_runs"])
    assert projection["panels"][0]["aggregate_signal"] == "PASS"
    assert projection["panels"][0]["superseded"] is True
    assert projection["panels"][0]["enforcement_impact"] == "requires_revision"


def test_second_forced_owner_arrival_returns_exact_resume_fallback(tmp_path, monkeypatch):
    import hashlib

    incoming = queue.Queue()
    loop, registry, ctx, trace = _forced_test_context(tmp_path, incoming=incoming)
    calls = 0

    def forced_model(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        incoming.put(f"owner directive {calls}")
        return {"role": "assistant", "content": f"stale draft {calls}"}, 0.0

    monkeypatch.setattr(loop, "call_llm_with_retry", forced_model)
    text, _usage, returned_trace = loop._forced_final_answer(
        ctx,
        prompt="finalize",
        fallback_text="fallback",
        reason_code="round_limit",
    )

    assert calls == 2
    assert "Resume the task" in text
    assert "stale draft" not in text
    assert len(registry._ctx._owner_directives) == 2
    assert registry._ctx._delivery_candidate.full_text == text
    assert returned_trace["delivery_candidate"]["content_sha256"] == hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()
    assert returned_trace["forced_finalization"]["source"] == (
        "late_owner_directive_requires_resume"
    )


def test_child_result_change_during_host_panel_supersedes_pass(tmp_path, monkeypatch):
    import hashlib

    import ouroboros.loop as loop
    import ouroboros.review_substrate as review_substrate
    from ouroboros.task_results import write_task_result
    from ouroboros.tools.registry import ToolRegistry

    _write_child(tmp_path)
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.task_id = "parent1"
    registry._ctx.drive_root = str(tmp_path)
    registry._ctx.is_direct_chat = False
    registry._ctx._task_acceptance_reviewed = False
    registry._ctx.task_metadata = {
        "budget_drive_root": str(tmp_path),
        "root_task_id": "parent1",
    }
    clean = review_substrate.ReviewRunResult(
        request={"surface": "task_acceptance", "policy": {"require_criterion_evidence": True}},
        actors=[{
            "signal": "PASS",
            "slot_id": "host-1",
            "parsed": {
                "outcome_tier": "solved",
                "completion_coach": "ship",
                "criteria_used": [{
                    "criterion": "owner request",
                    "status": "supported",
                    "evidence_refs": ["artifact:1"],
                }],
            },
        }],
        parsed_findings=[],
        aggregate_signal="PASS",
    )

    def mutate_child(_review_ctx):
        write_task_result(
            tmp_path,
            "child1",
            "completed",
            parent_task_id="parent1",
            root_task_id="parent1",
            delegation_role="subagent",
            result="fact v2 replaces fact v1",
            trace_summary="trace",
            artifact_status="ready",
            artifacts=[{"kind": "report", "name": "report.md", "sha256": "a" * 64}],
        )
        return clean

    monkeypatch.setattr(loop, "get_task_review_mode", lambda: "auto")
    monkeypatch.setattr(loop, "_execute_task_acceptance_panel", mutate_child)
    trace = {"tool_calls": [], "reasoning_notes": []}
    messages = [{"role": "system", "content": ""}, {"role": "user", "content": "goal"}]
    answer = "Answer based on fact v1."

    another_round = loop._run_task_acceptance_review_once(
        tools=registry,
        content=answer,
        task_id="parent1",
        task_type="task",
        llm_trace=trace,
        drive_root=tmp_path,
        messages=messages,
        emit_progress=lambda _text: None,
    )

    assert another_round is True
    assert registry._ctx._task_acceptance_reviewed is False
    assert trace["review_runs"][0]["superseded_by_revision"] is True
    assert trace["review_runs"][0]["superseded_reason"] == (
        "host_acceptance_evidence_revision_changed"
    )
    binding = loop._delivery_acceptance_binding(
        registry, trace, hashlib.sha256(answer.encode("utf-8")).hexdigest(),
    )
    assert binding["acceptance_status"] == "unaccepted"
    assert binding["authoritative"] is False
    assert "TASK ACCEPTANCE REFRESH" in str(messages[-1]["content"])


def test_child_result_change_after_host_panel_requires_replacement_and_fresh_panel(
    tmp_path, monkeypatch,
):
    import ouroboros.loop as loop
    import ouroboros.review_substrate as review_substrate
    from ouroboros.task_results import write_task_result

    _write_child(tmp_path)
    _loop, registry, ctx, trace = _forced_test_context(tmp_path)
    registry._ctx.is_direct_chat = False
    registry._ctx._task_acceptance_reviewed = False
    clean = review_substrate.ReviewRunResult(
        request={"surface": "task_acceptance", "policy": {"require_criterion_evidence": True}},
        actors=[{
            "signal": "PASS",
            "slot_id": "host-1",
            "parsed": {
                "outcome_tier": "solved",
                "completion_coach": "ship",
                "criteria_used": [{
                    "criterion": "owner request",
                    "status": "supported",
                    "evidence_refs": ["artifact:1"],
                }],
            },
        }],
        parsed_findings=[],
        aggregate_signal="PASS",
    )
    panel_calls = {"count": 0}

    def clean_panel(_review_ctx):
        panel_calls["count"] += 1
        return clean

    monkeypatch.setattr(loop, "get_task_review_mode", lambda: "auto")
    monkeypatch.setattr(loop, "_execute_task_acceptance_panel", clean_panel)
    monkeypatch.setattr(loop, "_compute_subagent_handoff", lambda *_a, **_k: None)
    monkeypatch.setattr(loop, "_maybe_enforce_child_absorption_gate", lambda *_a, **_k: None)
    monkeypatch.setattr(loop, "_maybe_inject_finalization_nudges", lambda *_a, **_k: False)
    monkeypatch.setattr(loop, "_finalize_task_services", lambda *_a, **_k: False)

    original_project = loop._project_child_result_dispositions
    race = {"mutated": False}

    def project_then_finish_child(round_ctx, llm_trace):
        original_project(round_ctx, llm_trace)
        if registry._ctx._task_acceptance_reviewed and not race["mutated"]:
            race["mutated"] = True
            write_task_result(
                tmp_path,
                "child1",
                "completed",
                parent_task_id="parent1",
                root_task_id="parent1",
                delegation_role="subagent",
                result="late child result v2",
                trace_summary="trace",
                artifact_status="ready",
                artifacts=[{
                    "kind": "report",
                    "name": "report.md",
                    "sha256": "b" * 64,
                }],
            )

    monkeypatch.setattr(loop, "_project_child_result_dispositions", project_then_finish_child)

    first = loop._no_tool_final_answer(
        "Answer based on child result v1.",
        ctx,
        trace,
        registry,
        queue.Queue(),
        set(),
        lambda _text: None,
    )

    assert first is None
    assert panel_calls["count"] == 1
    assert registry._ctx._task_acceptance_reviewed is False
    assert trace["review_runs"][0]["superseded_by_revision"] is True
    assert trace["review_runs"][0]["superseded_reason"] == (
        "delivery_evidence_changed_after_host_acceptance"
    )
    assert registry._ctx._delivery_control_required is True

    second = loop._no_tool_final_answer(
        json.dumps({
            "delivery_control": "replace",
            "full_answer": "Replacement answer incorporating late child result v2.",
        }),
        ctx,
        trace,
        registry,
        queue.Queue(),
        set(),
        lambda _text: None,
    )

    assert second is not None
    text, _usage, returned_trace = second
    assert text == "Replacement answer incorporating late child result v2."
    assert panel_calls["count"] == 2
    assert registry._ctx._task_acceptance_reviewed is True
    binding = returned_trace["delivery_candidate"]["acceptance_binding"]
    assert binding["authoritative"] is True
    assert binding["acceptance_status"] == "pass"


def test_orphan_label_keeps_cancelled_lifecycle_and_terminal_result(monkeypatch, tmp_path):
    import ouroboros.loop as loop

    ctx = SimpleNamespace()
    monkeypatch.setattr(
        loop,
        "_direct_child_results",
        lambda _ctx: [{
            "task_id": "child1",
            "status": "cancelled",
            "child_status": "completed",
        }],
    )
    monkeypatch.setattr(loop, "_child_disposition_state", lambda _child: "")

    note = loop._forced_orphan_note(ctx)

    assert "child1 [cancelled; terminal_result=completed]" in note
