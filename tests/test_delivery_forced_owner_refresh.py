"""The owner arrival refresh and what supersedes an already accepted pass.

Split verbatim out of ``tests/test_delivery_forced_finalization.py`` by theme. This
module owns the one complete refresh a forced owner arrival gets, the exact resume
fallback the second arrival returns, and the rule that a replacement, a stale preserve
or a child result that changed during or after the host panel supersedes an accepted
pass in both the outcome and the projection.
"""

from __future__ import annotations

import json
import queue

from tests._delivery_candidate_shared import (
    write_child as _write_child,
)

from tests._delivery_forced_shared import _bind_host_pass, _forced_test_context


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
    assert text == "Refreshed forced answer."
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

    assert text == "Forced replacement answer."
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
    from ouroboros import loop_delivery
    from ouroboros import loop_acceptance_review
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
    monkeypatch.setattr(loop_acceptance_review, "_execute_task_acceptance_panel", mutate_child)
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
    binding = loop_delivery._delivery_acceptance_binding(
        registry, trace, hashlib.sha256(answer.encode("utf-8")).hexdigest(),
    )
    assert binding["acceptance_status"] == "unaccepted"
    assert binding["authoritative"] is False
    assert "TASK ACCEPTANCE REFRESH" in str(messages[-1]["content"])


def test_child_result_change_after_host_panel_requires_replacement_and_fresh_panel(
    tmp_path, monkeypatch,
):
    import ouroboros.loop as loop
    from ouroboros import loop_delivery
    from ouroboros import loop_acceptance_review
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
    monkeypatch.setattr(loop_acceptance_review, "_execute_task_acceptance_panel", clean_panel)
    monkeypatch.setattr(loop_delivery, "_compute_subagent_handoff", lambda *_a, **_k: None)
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
