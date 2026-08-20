"""What a forced finalization binds into the candidate: the child suffix and its evidence.

Split verbatim out of ``tests/test_delivery_forced_finalization.py`` by theme. This
module owns the child-result suffix that must sit inside both the candidate and the
panel subject, the new unaccepted revision a retained candidate's suffix creates, the
service teardown that precedes the model call, the rebinding of the latest child
result, and the production budget wrapup that routes through the delivery candidate.
"""

from __future__ import annotations

import queue

from tests._delivery_candidate_shared import (
    write_child as _write_child,
    write_confirmed_disposition_fixture as _write_confirmed_disposition,
)

from tests._delivery_forced_shared import _forced_test_context


def test_normal_host_suffix_is_inside_candidate_and_panel_subject(tmp_path, monkeypatch):
    import hashlib

    _write_child(tmp_path)
    _write_confirmed_disposition(
        tmp_path,
        disposition="deferred",
        rationale="defer until the next run",
    )
    loop, registry, ctx, trace = _forced_test_context(tmp_path)
    from ouroboros import loop_delivery

    captured = {}

    monkeypatch.setattr(loop_delivery, "_compute_subagent_handoff", lambda *_a, **_k: None)
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
    assert "NOTE: finalized" in text
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

    from ouroboros import task_pacing

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
        cost_ceiling=task_pacing.CostCeiling(
            state=task_pacing.COST_CEILING_ACTIVE, ceiling_usd=4.0,
        ),
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
    from ouroboros import task_pacing

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
            cost_ceiling=task_pacing.CostCeiling(
                state=task_pacing.COST_CEILING_ACTIVE, ceiling_usd=4.0,
            ),
        )
