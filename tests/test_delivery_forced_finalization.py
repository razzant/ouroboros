"""What a forced finalization preserves and discloses on the way out.

This module owns the deferred-child projection that must precede a forced return, the
swarm router's cached and confirmed receipts, and the stale candidate every hard exit
— physical budget, dispatch rail, budget latch, provider unavailable — preserves with
a resume disclosure.

The candidate/suffix binding, the owner arrival refresh, the delivery-control latch,
the children_unabsorbed acceptance rail and the typed acceptance-bypass records were
split verbatim into ``tests/test_delivery_forced_suffix_binding.py``,
``tests/test_delivery_forced_owner_refresh.py``, ``tests/test_delivery_control_latch.py``,
``tests/test_delivery_forced_absorption_acceptance.py`` and
``tests/test_delivery_forced_acceptance_bypass.py``; the two context builders they
share live in ``tests/_delivery_forced_shared.py``.
"""
from __future__ import annotations


from tests._delivery_candidate_shared import (
    write_child as _write_child,
    write_confirmed_disposition_fixture as _write_confirmed_disposition,
)
from tests._delivery_forced_shared import _forced_test_context


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


def test_blocking_open_plan_round_rail_preserves_useful_candidate(tmp_path, monkeypatch):
    from ouroboros.task_results import (
        STATUS_RUNNING,
        record_plan_review_attempt,
        write_task_result,
    )

    loop, registry, limit_ctx, trace = _forced_test_context(tmp_path)
    registry._ctx.task_metadata["force_plan"] = True
    write_task_result(tmp_path, "parent1", STATUS_RUNNING, result="running")
    record_plan_review_attempt(tmp_path, "parent1", fingerprint="a" * 64)
    loop._replace_delivery_candidate(
        registry,
        limit_ctx,
        trace,
        "Useful verified work completed before the rail.",
        control="candidate",
    )
    monkeypatch.setattr(loop, "get_review_enforcement", lambda: "blocking")
    monkeypatch.setattr(
        loop,
        "call_llm_with_retry",
        lambda *_args, **_kwargs: ({"role": "assistant", "content": ""}, 0.0),
    )

    text, usage, _returned_trace = loop._handle_round_limit(limit_ctx)

    assert text.startswith("Useful verified work completed before the rail.")
    assert "Blocking plan review remained open" in text
    assert "`round_limit`" in text
    assert usage["reason_code"] == "round_limit"


def test_forced_swarm_router_uses_cached_unconfirmed_receipt(tmp_path, monkeypatch):
    loop, registry, limit_ctx, _trace = _forced_test_context(tmp_path)
    registry._ctx.is_ephemeral_turn = True
    registry._ctx.task_metadata.update({"force_plan": True, "force_plan_source": "swarm"})
    registry._ctx._swarm_handoff_attempt = {
        "task_id": "swarm-task-1",
        "routing_token": "route-token",
        "status": "unconfirmed",
        "reason": "confirmation_timeout",
        "response": "PROMOTE_UNCONFIRMED",
    }
    monkeypatch.setattr(
        loop,
        "call_llm_with_retry",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("forced router fallback must not start a new model round")
        ),
    )

    text, usage, _returned_trace = loop._handle_round_limit(limit_ctx)

    assert "swarm-task-1" in text
    assert "admission was not confirmed" in text
    assert "No second routing event was emitted" in text
    assert usage["reason_code"] == "round_limit"


def test_forced_swarm_router_keeps_confirmed_handoff_successful(tmp_path, monkeypatch):
    loop, registry, limit_ctx, _trace = _forced_test_context(tmp_path)
    registry._ctx.is_ephemeral_turn = True
    registry._ctx.task_metadata.update({"force_plan": True, "force_plan_source": "swarm"})
    registry._ctx._swarm_handoff_attempt = {
        "task_id": "swarm-task-1",
        "routing_token": "route-token",
        "status": "scheduled",
        "reason": "",
        "response": "OK: task swarm-task-1 accepted and durably scheduled",
    }
    monkeypatch.setattr(
        loop,
        "call_llm_with_retry",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("forced router fallback must not start a new model round")
        ),
    )

    text, usage, _returned_trace = loop._handle_round_limit(limit_ctx)

    assert "Swarm admitted managed task swarm-task-1" in text
    assert usage.get("execution_status") != "failed"
    assert usage.get("reason_code") != "round_limit"


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
