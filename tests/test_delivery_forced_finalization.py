from __future__ import annotations

import hashlib
import json
import queue
import threading
import time
from types import SimpleNamespace

import pytest

from tests._delivery_candidate_shared import (
    write_child as _write_child,
    write_confirmed_disposition_fixture as _write_confirmed_disposition,
)


def _forced_test_context(tmp_path, *, usage=None, incoming=None):
    from tests.test_loop_acceptance_gate import _seed_acceptance_root
    _seed_acceptance_root(tmp_path, "parent1", SimpleNamespace())
    import ouroboros.loop as loop
    from ouroboros.tools.registry import ToolRegistry

    trace = {"tool_calls": [], "reasoning_notes": []}
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.task_id = "parent1"
    registry._ctx.task_contract = {"expected_output": "A verified result"}
    registry._ctx.task_metadata = {
        "budget_drive_root": str(tmp_path), "root_task_id": "parent1",
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


def _select_completion(registry, ctx, trace, *, answer=None, answer_sha256=None, action="finish", rationale=""):
    """Stage the author's current observation and consume its real request."""
    from ouroboros.loop_delivery import completion_observation, consume_completion_request
    from ouroboros.tools.control_runtime import stage_completion_request

    registry._ctx._completion_observation = completion_observation(registry._ctx, trace)
    response = stage_completion_request(registry._ctx, {"action": action, "answer": answer,
        "answer_sha256": answer_sha256, "rationale": rationale})
    assert json.loads(response)["status"] == "completion_requested"
    assert consume_completion_request(registry, ctx, trace), trace.get("completion_refusals")


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


def test_post_round_finalize_control_binds_its_grace_deadline(tmp_path, monkeypatch):
    """A finalize control drained after an answer uses the mailbox grace deadline.

    The post-round drain reaches the same forced rail as the early-round path;
    it must not fall back to the task's raw transport deadline.
    """
    loop, registry, limit_ctx, trace = _forced_test_context(tmp_path)
    registry._ctx.owner_message_admission_lock = threading.RLock()
    registry._ctx.owner_message_admission_agent = SimpleNamespace(
        _accepting_owner_messages=False,
        _busy=True,
        _current_task_id="parent1",
    )
    grace_deadline = time.time() + 42.0
    monkeypatch.setattr(loop, "_resolve_delivery_control", lambda content, *_args: ("fresh", content))
    monkeypatch.setattr(loop, "_enforce_swarm_actions", lambda *_args: False)
    monkeypatch.setattr(loop, "_compute_subagent_handoff", lambda *_args: None)
    monkeypatch.setattr(loop, "_maybe_enforce_child_absorption_gate", lambda *_args: None)
    monkeypatch.setattr(loop, "_maybe_inject_finalization_nudges", lambda *_args: False)
    monkeypatch.setattr(loop, "_finalize_task_services", lambda *_args: False)
    monkeypatch.setattr(loop, "_run_task_acceptance_review_once", lambda **_kwargs: False)
    monkeypatch.setattr(
        loop,
        "_drain_incoming_messages",
        lambda *_args, **_kwargs: {
            "finalize_now": "deadline",
            "finalize_deadline_ts": grace_deadline,
        },
    )
    observed = {}

    def _capture_forced(ctx, reason):
        observed["deadline_ts"] = ctx.deadline_ts
        observed["reason"] = reason
        return "forced", {}, {}

    monkeypatch.setattr(loop, "_handle_forced_finalization", _capture_forced)
    result = loop._no_tool_final_answer(
        "answer",
        limit_ctx,
        trace,
        registry,
        queue.Queue(),
        set(),
        lambda _msg: None,
    )

    assert result[:2] == ("forced", {})
    assert observed == {"deadline_ts": grace_deadline, "reason": "deadline"}


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
    # The wave is still open at finalization, so the disclosure says so in the
    # present tense: "remained" claimed a panel had ended that nobody closed.
    assert "Blocking plan review is open" in usage["terminal_host_notice"]
    assert "`round_limit`" in usage["terminal_host_notice"]
    assert usage["reason_code"] == "round_limit"


def test_forced_managed_swarm_runs_the_ordinary_final_model_call(tmp_path, monkeypatch):
    loop, registry, limit_ctx, _trace = _forced_test_context(tmp_path)
    registry._ctx.task_metadata.update({"force_plan": True, "force_plan_source": "swarm"})
    calls = []
    monkeypatch.setattr(
        loop, "call_llm_with_retry",
        lambda *_args, **_kwargs: calls.append("forced") or (
            {"role": "assistant", "content": "Verified work before the limit."}, 0.0),
    )
    text, usage, _returned_trace = loop._handle_round_limit(limit_ctx)
    assert calls == ["forced"]
    assert text == "Verified work before the limit."
    assert usage["reason_code"] == "round_limit"
    assert "Plan review is open" in usage["terminal_host_notice"]


def test_presence_handoff_does_not_replace_ordinary_forced_finalization(tmp_path, monkeypatch):
    loop, registry, limit_ctx, _trace = _forced_test_context(tmp_path)
    registry._ctx.task_metadata["presence"] = {"binding_id": "presence-binding"}
    registry._ctx._swarm_handoff_attempt = {
        "task_id": "presence-work", "routing_token": "route-token", "status": "scheduled",
        "reason": "", "response": "OK: task presence-work accepted and durably scheduled",
    }
    monkeypatch.setattr(loop, "call_llm_with_retry", lambda *_args, **_kwargs: (
        {"role": "assistant", "content": "Current verified Presence work."}, 0.0))
    text, usage, _returned_trace = loop._handle_round_limit(limit_ctx)
    assert text == "Current verified Presence work."
    assert registry._ctx._swarm_handoff_attempt["task_id"] == "presence-work"
    assert usage["reason_code"] == "round_limit"


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
    assert "STALE-EVIDENCE NOTICE — RESUME REQUIRED (host)" in usage["terminal_host_notice"]
    assert "does not claim to incorporate it" in usage["terminal_host_notice"]
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


def test_budget_dispatch_rail_keeps_candidate_beside_undispositioned_child_notice(
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
    _bind_host_pass(loop, registry, trace, original)
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
    assert "child1 [running]" in usage["terminal_host_notice"]
    assert candidate.content_sha256 == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert candidate is original
    assert candidate.acceptance_binding["acceptance_status"] == "pass"
    assert candidate.acceptance_binding["authoritative"] is True
    assert returned_trace["forced_finalization"]["source"] == "budget_preserve"
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

    # Unprocessed owner input prevents final acceptance, but does not itself
    # change the semantic criteria. Preserve the answer and evidence provenance
    # with a loud resume disclosure until Main processes the new source.
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
    assert "STALE-EVIDENCE NOTICE — RESUME REQUIRED (host)" in usage["terminal_host_notice"]
    assert "has not been regenerated or accepted" in usage["terminal_host_notice"]
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
    assert returned_trace["delivery_candidate"]["evidence_current"] is True
    forced = returned_trace["forced_finalization"]
    assert forced["source"] == (
        "budget_latched_fallback_stale_evidence_resume_required"
    )
    assert forced["evidence_current"] is False  # Preserved stale bytes are not current evidence.
    assert forced["evidence_revision"] == old.evidence_revision
    assert forced["current_evidence_revision"] == old.evidence_revision
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
    # Source acknowledgement and semantic evidence have separate generations.
    # Provider failure cannot acknowledge this source or infer new criteria.
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
    assert "STALE-EVIDENCE NOTICE — RESUME REQUIRED (host)" in usage["terminal_host_notice"]
    assert "does not claim to incorporate it" in usage["terminal_host_notice"]
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
    assert returned_trace["delivery_candidate"]["evidence_current"] is True
    forced = returned_trace["forced_finalization"]
    assert forced["source"] == "host_fallback_stale_evidence_resume_required"
    assert forced["evidence_current"] is False  # Preserved stale bytes are not current evidence.
    assert forced["evidence_revision"] == old.evidence_revision
    assert forced["current_evidence_revision"] == old.evidence_revision
    assert usage["_best_effort_extracted"] is True
    assert usage["terminal_origin"] == "host_salvage"
    assert trace["tool_calls"] == []


def test_provider_unavailable_keeps_current_model_candidate_byte_exact(
    tmp_path, monkeypatch,
):
    loop, registry, limit_ctx, trace = _forced_test_context(tmp_path)
    answer = "Complete current model answer."
    loop._replace_delivery_candidate(registry, limit_ctx, trace, answer, control="replace")
    monkeypatch.setattr(loop, "call_llm_with_retry", lambda *_a, **_k: (None, 0.0))
    monkeypatch.setattr(loop, "_force_plan_disclosure", lambda *_a, **_k: "\nHOST FACT")

    text, usage, returned_trace = loop._handle_provider_unavailable(limit_ctx)

    assert text == answer
    assert usage["terminal_origin"] == "model_final"
    assert usage["terminal_plan_review_open"] is True
    assert "HOST FACT" not in text
    assert returned_trace["forced_finalization"]["source"] == "retained_candidate"


def test_provider_unavailable_strips_host_suffix_already_on_retained_candidate(
    tmp_path, monkeypatch,
):
    loop, registry, limit_ctx, trace = _forced_test_context(tmp_path)
    loop._replace_delivery_candidate(
        registry, limit_ctx, trace, "Model answer.\nHOST FACT",
        control="host_suffix", model_text="Model answer.",
    )
    monkeypatch.setattr(loop, "call_llm_with_retry", lambda *_a, **_k: (None, 0.0))

    text, usage, returned_trace = loop._handle_provider_unavailable(limit_ctx)

    assert text == "Model answer."
    assert usage["terminal_origin"] == "model_final"
    assert registry._ctx._delivery_candidate.full_text == text
    assert returned_trace["forced_finalization"]["candidate_sha256"] == hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def test_provider_unavailable_forced_model_answer_excludes_host_suffix(
    tmp_path, monkeypatch,
):
    loop, _registry, limit_ctx, _trace = _forced_test_context(tmp_path)
    monkeypatch.setattr(
        loop,
        "call_llm_with_retry",
        lambda *_a, **_k: ({"content": "Fresh forced model answer."}, 0.0),
    )
    monkeypatch.setattr(loop, "_force_plan_disclosure", lambda *_a, **_k: "\nHOST FACT")

    text, usage, returned_trace = loop._handle_provider_unavailable(limit_ctx)

    assert text == "Fresh forced model answer."
    assert usage["terminal_origin"] == "model_final"
    assert usage["terminal_plan_review_open"] is True
    assert returned_trace["forced_finalization"]["source"] == "model"


def test_provider_unavailable_classifies_forced_response_shape(tmp_path):
    missing = object()
    cases = (
        ("stop", "stop", [], "model_final"),
        ("end-turn", "end_turn", [], "model_final"),
        ("refusal", "refusal", [], "model_final"),
        ("absent", missing, [], "model_final"),
        ("null", None, [], "host_salvage"),
        ("length", "length", [], "host_salvage"),
        ("max-tokens", "max_tokens", [], "host_salvage"),
        ("context", "context_length_exceeded", [], "host_salvage"),
        ("function-call", "function_call", [], "host_salvage"),
        (
            "tool-call",
            "stop",
            [{"id": "call-1", "type": "function", "function": {"name": "noop", "arguments": "{}"}}],
            "host_salvage",
        ),
    )

    for label, finish_reason, tool_calls, expected_origin in cases:
        root = tmp_path / label
        (root / "logs").mkdir(parents=True)
        loop, _registry, limit_ctx, _trace = _forced_test_context(root)
        expected_text = f"forced response {label}"

        class ForcedResponseLLM:
            def chat(self, **_kwargs):
                response_usage = {"provider": "fake", "resolved_model": "fake/model"}
                if finish_reason is not missing:
                    response_usage["response_finish_reason"] = finish_reason
                return {
                    "content": expected_text,
                    "tool_calls": tool_calls,
                }, response_usage

        limit_ctx.llm = ForcedResponseLLM()
        text, usage, returned_trace = loop._handle_provider_unavailable(limit_ctx)

        assert text == expected_text, label
        assert usage["terminal_origin"] == expected_origin, label
        expected_source = (
            "model" if expected_origin == "model_final" else "forced_model_incomplete"
        )
        assert returned_trace["forced_finalization"]["source"] == expected_source, label


def test_provider_unavailable_prefers_current_candidate_over_incomplete_forced_response(
    tmp_path,
):
    loop, registry, limit_ctx, trace = _forced_test_context(tmp_path)
    answer = "Complete current model answer."
    loop._replace_delivery_candidate(registry, limit_ctx, trace, answer, control="replace")

    class LengthStoppedLLM:
        def chat(self, **_kwargs):
            return (
                {"content": "partial replacement"},
                {
                    "provider": "fake",
                    "resolved_model": "fake/model",
                    "response_finish_reason": "length",
                },
            )

    limit_ctx.llm = LengthStoppedLLM()
    text, usage, returned_trace = loop._handle_provider_unavailable(limit_ctx)

    assert text == answer
    assert usage["terminal_origin"] == "model_final"
    assert returned_trace["forced_finalization"]["source"] == "retained_candidate"


def test_provider_unavailable_without_model_answer_stamps_host_salvage(
    tmp_path, monkeypatch,
):
    loop, _registry, limit_ctx, _trace = _forced_test_context(tmp_path)
    monkeypatch.setattr(loop, "call_llm_with_retry", lambda *_a, **_k: (None, 0.0))

    text, usage, _returned_trace = loop._handle_provider_unavailable(limit_ctx)

    assert "provider returned no usable response" in text
    assert usage["terminal_origin"] == "host_salvage"


def test_deadline_exhausted_forced_finalization_keeps_local_reason(tmp_path, monkeypatch):
    import time
    from ouroboros.outcomes import derive_loop_outcome

    loop, _registry, limit_ctx, _trace = _forced_test_context(
        tmp_path, usage={"_last_llm_error_kind": "deadline_exhausted"},
    )
    limit_ctx.deadline_ts = time.time() + 30
    monkeypatch.setattr(
        loop,
        "call_llm_with_retry",
        lambda *_args, **_kwargs: (
            {"role": "assistant", "content": "Best answer before deadline."},
            0.0,
        ),
    )

    text, usage, returned_trace = loop._handle_provider_unavailable(
        limit_ctx, error_kind="deadline_exhausted",
    )

    assert text == "Best answer before deadline."
    assert usage["reason_code"] == "deadline_local"
    assert usage["execution_status"] == "failed"
    assert usage["_best_effort_extracted"] is True
    assert returned_trace["forced_finalization"]["reason_code"] == "deadline_local"
    outcome = derive_loop_outcome(text, usage, returned_trace)
    assert outcome["outcome_axes"]["execution"]["status"] == "best_effort"
    assert outcome["outcome_axes"]["execution"]["reason_code"] == "deadline_local"


def test_normal_host_notice_stays_outside_candidate_and_panel_subject(tmp_path, monkeypatch):
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
    text, usage, returned_trace = result
    assert text == captured["content"]
    assert text == "Base complete answer."
    assert usage["terminal_host_notice"].count("DEFERRED CHILD RESULTS") == 1
    assert returned_trace["delivery_candidate"]["content_sha256"] == hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()
    assert registry._ctx._delivery_candidate.full_text == text
    assert registry._ctx._delivery_candidate.model_text == "Base complete answer."


def test_forced_retained_candidate_notice_preserves_unchanged_revision(
    tmp_path, monkeypatch,
):
    import hashlib

    _write_child(tmp_path, status="running")
    loop, registry, ctx, trace = _forced_test_context(tmp_path)
    original = loop._replace_delivery_candidate(
        registry, ctx, trace, "Retained complete answer.", control="candidate",
    )
    _bind_host_pass(loop, registry, trace, original)
    monkeypatch.setattr(loop, "call_llm_with_retry", lambda *_a, **_k: (None, 0.0))

    text, usage, returned_trace = loop._forced_final_answer(
        ctx,
        prompt="finalize",
        fallback_text="host fallback",
        reason_code="round_limit",
    )

    candidate = registry._ctx._delivery_candidate
    assert text == candidate.full_text
    assert "NOTE: finalized" in usage["terminal_host_notice"]
    assert candidate is original
    assert candidate.content_sha256 == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert candidate.acceptance_binding["acceptance_status"] == "pass"
    assert candidate.acceptance_binding["authoritative"] is True
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
    assert "child1 [completed]" in ctx.accumulated_usage["terminal_host_notice"]
    assert "child1 [running]" not in ctx.accumulated_usage["terminal_host_notice"]
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

    loop, registry, ctx, trace = _forced_test_context(
        tmp_path, usage={"cost": 5.0},
    )
    # The paid wrap-up at a reached ceiling is an explicit experiment profile's authored rail.
    registry._ctx.task_contract = {**registry._ctx.task_contract, "budget_profile": {"cost_hard_stop_pct": 50}}

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


def test_wrapup_is_forced_while_real_ledger_admission_still_fits(tmp_path, monkeypatch):
    """The post-round rail reserves the last affordable send before another
    ordinary round could consume it; the forced call crosses real admission."""
    from ouroboros import task_pacing
    import ouroboros.usage_accounting as accounting

    loop, registry, ctx, _trace = _forced_test_context(
        tmp_path, usage={"cost": 1.0, "_context_prompt_estimate": 1000},
    )
    # The last-fit rail is an explicit experiment profile's authored rail (#1128).
    registry._ctx.task_contract = {**registry._ctx.task_contract, "budget_profile": {"cost_hard_stop_pct": 50}}
    scope = accounting.UsageScope(
        drive_root=tmp_path, task_id="parent1", root_task_id="parent1",
        global_limit_usd=1000.0, root_limit_usd=100.0,
    )
    with accounting.usage_scope(scope):
        prior = accounting.reserve_attempt(accounting.AttemptRequest(
            model="test-model", provider="openrouter", reservation_usd=96.5,
        ))
        accounting.mark_dispatched(prior)
        accounting.settle_attempt(prior, {}, cost_usd=96.5, cost_final=True)

        monkeypatch.setattr(accounting, "_reservation_cost", lambda _request: 2.0)
        admitted = []

        def forced_model(*_args, **_kwargs):
            reservation = accounting.reserve_attempt(accounting.AttemptRequest(
                model="test-model", provider="openrouter", reservation_usd=2.0,
            ))
            accounting.mark_dispatched(reservation)
            accounting.settle_attempt(reservation, {}, cost_usd=2.0, cost_final=True)
            admitted.append(reservation.attempt_id)
            return {"role": "assistant", "content": "Affordable final answer."}, 0.0

        monkeypatch.setattr(loop, "call_llm_with_retry", forced_model)
        result = loop._check_budget_limits(
            ctx, budget_remaining_usd=900.0,
            cost_ceiling=task_pacing.resolve_cost_ceiling(
                900.0, {"cost_hard_stop_pct": 50}, root_cap_usd=100.0,
            ),
        )

    assert result is not None and result[0].startswith("Affordable final answer.")
    assert len(admitted) == 1
    assert accounting.usage_projection(tmp_path, root_task_id="parent1")["accounted_usd"] == 98.5


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

    assert "STALE-EVIDENCE NOTICE" in usage["terminal_host_notice"]
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


def test_forced_owner_refresh_does_not_resend_unknown_provider_outcome(tmp_path, monkeypatch):
    incoming = queue.Queue()
    loop, registry, ctx, _trace = _forced_test_context(tmp_path, incoming=incoming)
    calls = 0

    def forced_model(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        ctx.accumulated_usage["_last_llm_error_kind"] = "provider_outcome_unknown"
        incoming.put("retain this owner directive for resume")
        return {"role": "assistant", "content": ""}, 0.0

    monkeypatch.setattr(loop, "call_llm_with_retry", forced_model)
    text, _usage, returned_trace = loop._forced_final_answer(
        ctx,
        prompt="finalize",
        fallback_text="fallback",
        reason_code="round_limit",
    )

    assert calls == 1
    assert "Resume the task" in text
    assert len(registry._ctx._owner_directives) == 1
    assert returned_trace["forced_finalization"]["source"] == (
        "provider_outcome_unknown_no_resend"
    )


def test_child_result_change_during_host_panel_supersedes_pass(tmp_path, monkeypatch):
    from tests.test_loop_acceptance_gate import _seed_acceptance_root
    _seed_acceptance_root(tmp_path, "parent1", SimpleNamespace())
    import hashlib

    import ouroboros.loop as loop
    import ouroboros.review_substrate as review_substrate
    from ouroboros.task_results import write_task_result
    from ouroboros.tools.registry import ToolRegistry

    _write_child(tmp_path)
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.task_id = "parent1"
    registry._ctx.drive_root = str(tmp_path)
    registry._ctx.task_contract = {"expected_output": "A verified result"}
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
        emit_progress=lambda _text, *, incident=None: None,
    )

    assert another_round is True
    assert registry._ctx._task_acceptance_reviewed is False
    assert trace["review_runs"][0]["superseded_by_revision"] is True
    # The unified subject includes material child-result evidence.
    assert trace["review_runs"][0]["superseded_reason"] == (
        "host_acceptance_subject_changed"
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

    answer = "Replacement answer incorporating late child result v2."
    _select_completion(registry, ctx, trace, answer=answer)
    second = loop._no_tool_final_answer(answer, ctx, trace, registry, queue.Queue(), set(),
                                        lambda _text: None, explicit_candidate=True)

    assert second is not None
    text, _usage, returned_trace = second
    assert text == "Replacement answer incorporating late child result v2."
    assert panel_calls["count"] == 2
    assert registry._ctx._task_acceptance_reviewed is True
    binding = returned_trace["delivery_candidate"]["acceptance_binding"]
    assert binding["authoritative"] is True
    assert binding["acceptance_status"] == "pass"


# ---------------------------------------------------------------------------
# F1 (slime saga): a forced finalization while the delivery-control latch is
# armed must RESOLVE the protocol object purely (no repair round — a hard stop
# may not re-loop), never ship raw {"delivery_control": ...} JSON to the chat
# or the durable result, and never eat legitimate JSON when the latch is off.


def _arm_latch_with_candidate(loop, registry, limit_ctx, trace, text="Retained complete answer."):
    candidate = loop._replace_delivery_candidate(
        registry, limit_ctx, trace, text, control="awaiting_control",
    )
    registry._ctx._delivery_control_required = True  # replace() resets the latch
    return candidate


def test_forced_round_limit_resolves_explicit_answer_request(tmp_path, monkeypatch):
    loop, registry, limit_ctx, trace = _forced_test_context(tmp_path)
    _arm_latch_with_candidate(loop, registry, limit_ctx, trace)
    control = json.dumps({
        "action": "finish",
        "answer": "Complete replacement answer for the owner.",
    })
    monkeypatch.setattr(
        loop, "call_llm_with_retry",
        lambda *_a, **_k: ({"role": "assistant", "content": control}, 0.0),
    )

    text, usage, _returned_trace = loop._handle_round_limit(limit_ctx)

    assert text.startswith("Complete replacement answer for the owner.")
    assert "delivery_control" not in text
    assert registry._ctx._delivery_control_required is False
    assert registry._ctx._delivery_candidate.full_text == text
    assert usage["reason_code"] == "round_limit"


def test_forced_finalization_resolves_explicit_retained_selector(tmp_path, monkeypatch):
    loop, registry, limit_ctx, trace = _forced_test_context(tmp_path)
    candidate = _arm_latch_with_candidate(loop, registry, limit_ctx, trace)
    monkeypatch.setattr(
        loop, "call_llm_with_retry",
        lambda *_a, **_k: (
            {"role": "assistant", "content": json.dumps({"action": "finish", "answer_sha256": candidate.content_sha256})}, 0.0,
        ),
    )

    text, _usage, _returned_trace = loop._forced_final_answer(
        limit_ctx, prompt="finalize", fallback_text="fallback", reason_code="finalization_grace",
    )

    assert text.startswith("Retained complete answer.")
    assert "delivery_control" not in text
    assert registry._ctx._delivery_control_required is False


def test_forced_finalization_degrades_malformed_control_to_retained_candidate(
    tmp_path, monkeypatch,
):
    loop, registry, limit_ctx, trace = _forced_test_context(tmp_path)
    _arm_latch_with_candidate(loop, registry, limit_ctx, trace)
    # Duplicate protocol key -> invalid control object with control intent.
    malformed = '{"delivery_control":"keep","delivery_control":"replace"}'
    monkeypatch.setattr(
        loop, "call_llm_with_retry",
        lambda *_a, **_k: ({"role": "assistant", "content": malformed}, 0.0),
    )

    text, _usage, returned_trace = loop._forced_final_answer(
        limit_ctx, prompt="finalize", fallback_text="fallback", reason_code="round_limit",
    )

    assert text.startswith("Retained complete answer.")
    assert "delivery_control" not in text
    candidate = registry._ctx._delivery_candidate
    assert candidate.degraded is True
    assert candidate.degraded_reason == "delivery_control_degraded"
    assert returned_trace["delivery_candidate"]["degraded_reason"] == "delivery_control_degraded"


def test_forced_finalization_passes_json_through_when_latch_not_armed(tmp_path, monkeypatch):
    """Legitimate user-facing JSON is never eaten while no control round is open."""
    loop, registry, limit_ctx, _trace = _forced_test_context(tmp_path)
    legitimate = json.dumps({"delivery_control": "keep"})
    monkeypatch.setattr(
        loop, "call_llm_with_retry",
        lambda *_a, **_k: ({"role": "assistant", "content": legitimate}, 0.0),
    )

    text, _usage, _returned_trace = loop._forced_final_answer(
        limit_ctx, prompt="finalize", fallback_text="fallback", reason_code="round_limit",
    )

    assert text.startswith(legitimate)


def test_forced_finalization_degrades_unknown_verb_control_to_retained_candidate(
    tmp_path, monkeypatch,
):
    """An armed latch treats ANY parsed object carrying the protocol key as
    protocol — an unknown verb is a mangled control, never the owner's answer."""
    loop, registry, limit_ctx, trace = _forced_test_context(tmp_path)
    _arm_latch_with_candidate(loop, registry, limit_ctx, trace)
    unknown_verb = json.dumps({
        "delivery_control": "publish",
        "full_answer": "text behind an unknown verb",
    })
    monkeypatch.setattr(
        loop, "call_llm_with_retry",
        lambda *_a, **_k: ({"role": "assistant", "content": unknown_verb}, 0.0),
    )

    text, _usage, _returned_trace = loop._forced_final_answer(
        limit_ctx, prompt="finalize", fallback_text="fallback", reason_code="round_limit",
    )

    assert text.startswith("Retained complete answer.")
    assert "delivery_control" not in text
    assert "publish" not in text
    candidate = registry._ctx._delivery_candidate
    assert candidate.degraded is True
    assert candidate.degraded_reason == "delivery_control_degraded"


def test_forced_finalization_degrades_broken_json_looking_text_to_retained_candidate(
    tmp_path, monkeypatch,
):
    """Armed latch + JSON-looking text that FAILS to parse: the model was
    explicitly instructed to answer with the protocol object, so a broken
    brace-blob is a mangled protocol attempt — resolve to the retained
    candidate with the typed degraded reason; never ship the broken JSON raw."""
    loop, registry, limit_ctx, trace = _forced_test_context(tmp_path)
    _arm_latch_with_candidate(loop, registry, limit_ctx, trace)
    broken = '{"delivery_control": "replace", "full_answer": "truncated mid-'
    monkeypatch.setattr(
        loop, "call_llm_with_retry",
        lambda *_a, **_k: ({"role": "assistant", "content": broken}, 0.0),
    )

    text, _usage, _returned_trace = loop._forced_final_answer(
        limit_ctx, prompt="finalize", fallback_text="fallback", reason_code="round_limit",
    )

    assert text.startswith("Retained complete answer.")
    assert '{"delivery_control"' not in text
    candidate = registry._ctx._delivery_candidate
    assert candidate.degraded is True
    assert candidate.degraded_reason == "delivery_control_degraded"


def test_forced_finalization_keeps_armed_prose_as_the_answer(tmp_path, monkeypatch):
    """Armed latch + plain prose (no protocol object): the fresh text stands.
    Trailing and fenced protocol objects are contained separately (tests
    below); the disclosed residual is a control object quoted MID-prose."""
    loop, registry, limit_ctx, trace = _forced_test_context(tmp_path)
    _arm_latch_with_candidate(loop, registry, limit_ctx, trace)
    prose = "A reconsidered complete prose answer for the owner."
    monkeypatch.setattr(
        loop, "call_llm_with_retry",
        lambda *_a, **_k: ({"role": "assistant", "content": prose}, 0.0),
    )

    text, _usage, _returned_trace = loop._forced_final_answer(
        limit_ctx, prompt="finalize", fallback_text="fallback", reason_code="round_limit",
    )

    assert text.startswith(prose)
    assert registry._ctx._delivery_control_required is False


def test_forced_finalization_passes_broken_json_through_when_latch_not_armed(
    tmp_path, monkeypatch,
):
    """Unarmed: broken JSON-looking output is an ordinary (bad) answer, not a
    protocol attempt — it passes through untouched."""
    loop, registry, limit_ctx, _trace = _forced_test_context(tmp_path)
    broken = '{"some_json_like": "output that never closes'
    monkeypatch.setattr(
        loop, "call_llm_with_retry",
        lambda *_a, **_k: ({"role": "assistant", "content": broken}, 0.0),
    )

    text, _usage, _returned_trace = loop._forced_final_answer(
        limit_ctx, prompt="finalize", fallback_text="fallback", reason_code="round_limit",
    )

    assert text.startswith(broken)


def test_nonforced_resolver_contains_unknown_legacy_control_in_a_migrated_episode(tmp_path):
    loop, registry, ctx, trace = _forced_test_context(tmp_path)
    candidate = _arm_latch_with_candidate(loop, registry, ctx, trace)
    candidate.control_episode_seen = True
    raw = json.dumps({"delivery_control": "finalize"})
    for _ in range(3):
        status, text = loop._resolve_delivery_control(raw, registry, ctx, trace)
        assert status == "retry" and text == candidate.full_text
        assert ctx.messages[-1]["role"] == "user"
        assert ctx.messages[-2] == {"role": "assistant", "content": raw}
        assert "latest whole held response" not in ctx.messages[-1]["content"]
    assert candidate.degraded is False
    assert "DELIVERY_CONTROL_REPAIR" not in str(ctx.messages)


def test_nonforced_resolver_selects_whole_held_prose_explicitly_after_legacy_control(tmp_path):
    loop, registry, ctx, trace = _forced_test_context(tmp_path)
    candidate = _arm_latch_with_candidate(loop, registry, ctx, trace)
    candidate.control_episode_seen = True
    loop._resolve_delivery_control('{"delivery_control":"finalize"}', registry, ctx, trace)
    prose = "A short corrected answer."
    assert loop._resolve_delivery_control(prose, registry, ctx, trace) == ("retry", candidate.full_text)
    assert registry._ctx._delivery_candidate is candidate
    _select_completion(registry, ctx, trace, answer_sha256=hashlib.sha256(prose.encode()).hexdigest())
    assert registry._ctx._delivery_candidate.full_text == prose
    assert trace["delivery_candidate"]["degraded"] is False
    assert trace["delivery_candidate"]["acceptance_binding"]["authoritative"] is False


def test_nonforced_duplicate_legacy_control_never_spends_a_repair_or_terminal_budget(tmp_path):
    loop, registry, ctx, trace = _forced_test_context(tmp_path)
    candidate = _arm_latch_with_candidate(loop, registry, ctx, trace)
    candidate.control_episode_seen = True
    raw = '{"delivery_control":"keep","delivery_control":"replace","full_answer":"Wrong answer"}'
    for _ in range(4):
        before = len(ctx.messages)
        assert loop._resolve_delivery_control(raw, registry, ctx, trace) == ("retry", candidate.full_text)
        assert len(ctx.messages) == before + 2
        assert ctx.messages[-2] == {"role": "assistant", "content": raw}
        assert ctx.messages[-1]["role"] == "user"
    assert not candidate.degraded and not candidate.degraded_reason
    _select_completion(registry, ctx, trace, answer_sha256=candidate.content_sha256)
    assert registry._ctx._delivery_candidate.full_text == candidate.full_text

# ---------------------------------------------------------------------------
# D2c (custody-absorption sprint, owner Q4=A): protocol-intent containment.
# Both latch-gated resolvers share one fence-strip normalization with
# observability._is_delivery_control_payload and treat a balanced protocol
# object at the very END of prose as a protocol attempt — never publishable
# text. Disclosed residual: a control object quoted MID-prose stays prose.


def test_forced_finalization_contains_trailing_protocol_object_in_prose(
    tmp_path, monkeypatch,
):
    """Armed latch + prose ENDING with the protocol object (the incident
    form): a protocol attempt mixed with text is never honored and never
    published raw — the retained candidate stands with the typed degraded
    reason, and the note discloses the preservation (the raw response itself
    stays persisted by the observability layer; P1)."""
    loop, registry, limit_ctx, trace = _forced_test_context(tmp_path)
    _arm_latch_with_candidate(loop, registry, limit_ctx, trace)
    mixed = (
        "Here is a summary of what happened during the run.\n"
        + json.dumps({"delivery_control": "replace", "full_answer": "smuggled"})
    )
    monkeypatch.setattr(
        loop, "call_llm_with_retry",
        lambda *_a, **_k: ({"role": "assistant", "content": mixed}, 0.0),
    )

    text, _usage, returned_trace = loop._forced_final_answer(
        limit_ctx, prompt="finalize", fallback_text="fallback", reason_code="round_limit",
    )

    assert text.startswith("Retained complete answer.")
    assert '{"delivery_control"' not in text
    assert "smuggled" not in text
    candidate = registry._ctx._delivery_candidate
    assert candidate.degraded is True
    assert candidate.degraded_reason == "delivery_control_degraded"
    assert any(
        "invalid delivery-control object" in str(note)
        for note in returned_trace.get("reasoning_notes", [])
    )


def test_forced_finalization_resolves_fenced_completion_request(tmp_path, monkeypatch):
    """A fenced completion request selects exact retained bytes, never raw JSON."""
    loop, registry, limit_ctx, trace = _forced_test_context(tmp_path)
    candidate = _arm_latch_with_candidate(loop, registry, limit_ctx, trace)
    fenced = '```json\n' + json.dumps({"action": "finish", "answer_sha256": candidate.content_sha256}) + '\n```'
    monkeypatch.setattr(
        loop, "call_llm_with_retry",
        lambda *_a, **_k: ({"role": "assistant", "content": fenced}, 0.0),
    )

    text, _usage, _returned_trace = loop._forced_final_answer(
        limit_ctx, prompt="finalize", fallback_text="fallback", reason_code="round_limit",
    )

    assert text.startswith("Retained complete answer.")
    assert "delivery_control" not in text
    assert registry._ctx._delivery_control_required is False


def test_forced_finalization_degrades_fenced_broken_control_to_retained_candidate(
    tmp_path, monkeypatch,
):
    """A fenced BROKEN protocol blob is a mangled protocol attempt after the
    fence-strip (the leading-brace heuristic applies to the normalized body):
    the retained candidate stands with the typed degraded reason."""
    loop, registry, limit_ctx, trace = _forced_test_context(tmp_path)
    _arm_latch_with_candidate(loop, registry, limit_ctx, trace)
    fenced_broken = '```json\n{"delivery_control": "replace", "full_answer": "truncated mid-\n```'
    monkeypatch.setattr(
        loop, "call_llm_with_retry",
        lambda *_a, **_k: ({"role": "assistant", "content": fenced_broken}, 0.0),
    )

    text, _usage, _returned_trace = loop._forced_final_answer(
        limit_ctx, prompt="finalize", fallback_text="fallback", reason_code="round_limit",
    )

    assert text.startswith("Retained complete answer.")
    assert '{"delivery_control"' not in text
    candidate = registry._ctx._delivery_candidate
    assert candidate.degraded is True
    assert candidate.degraded_reason == "delivery_control_degraded"


def test_forced_finalization_keeps_midprose_quotation_as_prose(tmp_path, monkeypatch):
    """A control object quoted MID-prose stays prose (the disclosed residual
    of the trailing-object containment rule): Ouroboros legitimately quotes
    the literal in its own PR bodies and docs."""
    loop, registry, limit_ctx, trace = _forced_test_context(tmp_path)
    _arm_latch_with_candidate(loop, registry, limit_ctx, trace)
    prose = (
        'The loop emits {"delivery_control": "keep"} objects during control '
        "rounds, and the documentation now says so explicitly."
    )
    monkeypatch.setattr(
        loop, "call_llm_with_retry",
        lambda *_a, **_k: ({"role": "assistant", "content": prose}, 0.0),
    )

    text, _usage, _returned_trace = loop._forced_final_answer(
        limit_ctx, prompt="finalize", fallback_text="fallback", reason_code="round_limit",
    )

    assert text.startswith(prose)


def test_nonforced_resolver_privately_holds_trailing_legacy_protocol_object(tmp_path):
    loop, registry, ctx, trace = _forced_test_context(tmp_path)
    candidate = _arm_latch_with_candidate(loop, registry, ctx, trace)
    candidate.control_episode_seen = True
    mixed = "Prose half of a contradictory answer.\n" + json.dumps({"delivery_control": "keep"})
    for _ in range(2):
        assert loop._resolve_delivery_control(mixed, registry, ctx, trace) == ("retry", candidate.full_text)
        assert ctx.messages[-2]["content"] == mixed
        assert "latest whole held response" not in ctx.messages[-1]["content"]
    assert not candidate.degraded


def test_forced_finalization_contains_trailing_fenced_protocol_object(
    tmp_path, monkeypatch,
):
    """Armed latch + prose ending with a FENCED protocol object (models fence
    JSON by default): the same protocol attempt as a bare trailing object —
    never honored, never published raw."""
    loop, registry, limit_ctx, trace = _forced_test_context(tmp_path)
    _arm_latch_with_candidate(loop, registry, limit_ctx, trace)
    mixed = (
        "Here is my final summary of the run.\n```json\n"
        + json.dumps({"delivery_control": "replace", "full_answer": "smuggled"})
        + "\n```"
    )
    monkeypatch.setattr(
        loop, "call_llm_with_retry",
        lambda *_a, **_k: ({"role": "assistant", "content": mixed}, 0.0),
    )

    text, _usage, _trace = loop._forced_final_answer(
        limit_ctx, prompt="finalize", fallback_text="fallback", reason_code="round_limit",
    )

    assert text.startswith("Retained complete answer.")
    assert '{"delivery_control"' not in text
    assert "smuggled" not in text
    candidate = registry._ctx._delivery_candidate
    assert candidate.degraded is True
    assert candidate.degraded_reason == "delivery_control_degraded"


def test_nonforced_resolver_privately_holds_trailing_fenced_legacy_protocol(tmp_path):
    loop, registry, ctx, trace = _forced_test_context(tmp_path)
    candidate = _arm_latch_with_candidate(loop, registry, ctx, trace)
    candidate.control_episode_seen = True
    mixed = 'Prose half of a contradictory answer.\n```json\n{"delivery_control":"keep"}\n```'
    for _ in range(2):
        assert loop._resolve_delivery_control(mixed, registry, ctx, trace) == ("retry", candidate.full_text)
        assert ctx.messages[-2]["content"] == mixed
        assert "latest whole held response" not in ctx.messages[-1]["content"]
    assert not candidate.degraded


def test_parse_body_survives_degenerate_nested_trailing_blob(tmp_path):
    """A repetition-loop blob (deeply nested braces) after prose must classify
    as not-a-control instead of escaping RecursionError into the round loop."""
    from ouroboros import loop

    blob = "prose answer text\n" + '{"a":' * 2000 + "1" + "}" * 2000
    parsed, duplicate, embedded = loop._parse_delivery_control_body(blob)
    assert parsed is None and duplicate is False and embedded is False


def test_nonforced_fenced_legacy_control_is_inert_until_tool_selection(tmp_path):
    loop, registry, ctx, trace = _forced_test_context(tmp_path)
    candidate = _arm_latch_with_candidate(loop, registry, ctx, trace)
    candidate.control_episode_seen = True
    fenced = '```json\n{"delivery_control":"keep"}\n```'
    assert loop._resolve_delivery_control(fenced, registry, ctx, trace) == ("retry", candidate.full_text)
    assert registry._ctx._delivery_control_required is True
    _select_completion(registry, ctx, trace, answer_sha256=candidate.content_sha256)
    assert registry._ctx._delivery_control_required is False


def test_nonforced_midprose_quotation_is_held_as_selectable_complete_prose(tmp_path):
    loop, registry, ctx, trace = _forced_test_context(tmp_path)
    candidate = loop._replace_delivery_candidate(registry, ctx, trace, "Retained complete answer.", control="candidate")
    candidate.finalization_control = "child_absorption_or_revision_required"
    prose = 'As noted, the old loop emitted {"delivery_control":"keep"} in control rounds; this is explanatory prose.'
    assert loop._resolve_delivery_control(prose, registry, ctx, trace) == ("retry", candidate.full_text)
    assert registry._ctx._completion_held_sha256 == hashlib.sha256(prose.encode()).hexdigest()
    assert ctx.messages[-2]["content"] == prose and ctx.messages[-1]["role"] == "user"


def test_nonforced_legacy_control_cannot_close_a_child_absorption_hold(tmp_path):
    loop, registry, ctx, trace = _forced_test_context(tmp_path)
    candidate = loop._replace_delivery_candidate(registry, ctx, trace, "Retained complete answer.", control="candidate")
    loop._hold_delivery_for_skill_action(registry, trace, control="child_absorption_or_revision_required")
    status, text = loop._resolve_delivery_control('{"delivery_control":"keep"}', registry, ctx, trace)
    assert status == "retry" and text == candidate.full_text
    assert candidate.finalization_control == "child_absorption_or_revision_required"
    assert registry._ctx._delivery_control_required is True
    assert "finish_task" in ctx.messages[-1]["content"] and "DELIVERY_CONTROL_REPAIR" not in str(ctx.messages)


def test_nonforced_resolver_passes_mixed_prose_json_when_latch_not_armed(tmp_path):
    """Latch OFF: prose+JSON through the ORDINARY resolver passes untouched —
    parity with the forced latch-off pin above."""
    loop, registry, limit_ctx, trace = _forced_test_context(tmp_path)
    loop._replace_delivery_candidate(
        registry, limit_ctx, trace, "Retained complete answer.", control="candidate",
    )
    registry._ctx._delivery_control_required = False
    mixed = "Final summary for the owner.\n" + json.dumps({"delivery_control": "keep"})

    status, text = loop._resolve_delivery_control(mixed, registry, limit_ctx, trace)

    assert status == "fresh"
    assert text == mixed


def test_forced_rail_truncated_trailing_fragment_is_a_disclosed_residual(
    tmp_path, monkeypatch,
):
    """Pin the THIRD disclosed containment residual: an armed forced rail
    publishes prose ending with a TRUNCATED (unbalanced) protocol fragment as
    prose — a fragment is not a parseable object, and containing it would need
    the substring scanning the containment rule deliberately rejects
    (ARCHITECTURE ~1024 / DEVELOPMENT ~2196)."""
    loop, registry, limit_ctx, trace = _forced_test_context(tmp_path)
    _arm_latch_with_candidate(loop, registry, limit_ctx, trace)
    truncated = (
        "Here is my summary before the cut.\n"
        '{"delivery_control": "replace", "full_answer": "cut'
    )
    monkeypatch.setattr(
        loop, "call_llm_with_retry",
        lambda *_a, **_k: ({"role": "assistant", "content": truncated}, 0.0),
    )

    text, _usage, _trace = loop._forced_final_answer(
        limit_ctx, prompt="finalize", fallback_text="fallback", reason_code="round_limit",
    )

    assert text.startswith("Here is my summary before the cut.")
    assert '"delivery_control"' in text  # the documented residual, not a leak fix
    candidate = registry._ctx._delivery_candidate
    # The fragment rode the rail's ORDINARY forced packaging (degraded by the
    # rail reason), never the protocol-containment path.
    assert candidate.degraded_reason == "round_limit"
    assert candidate.finalization_control == "forced_replace:round_limit"


def test_salvage_predicate_keeps_mixed_answers_and_skips_pure_protocol():
    """observability's salvage predicate keeps its latch-free whole-object
    semantics after the fence-strip sharing: a MIXED prose+object answer
    stays salvageable (never destroyed), while the pure — optionally fenced —
    protocol object is skipped as a non-answer (it remains forensic evidence
    in the observability store)."""
    from ouroboros.observability import _is_delivery_control_payload

    assert _is_delivery_control_payload('{"delivery_control": "keep"}') is True
    assert _is_delivery_control_payload(
        '```json\n{"delivery_control": "keep"}\n```'
    ) is True
    assert _is_delivery_control_payload(
        'Real answer text.\n{"delivery_control": "keep"}'
    ) is False


def test_repeated_child_holds_never_force_an_answer_or_erase_the_retained_candidate(tmp_path, monkeypatch):
    _write_child(tmp_path, status="running")
    loop, registry, ctx, trace = _forced_test_context(tmp_path)
    candidate = _arm_latch_with_candidate(loop, registry, ctx, trace)
    registry._ctx._child_absorption_reminded = True  # legacy saved state grants no termination authority
    monkeypatch.setattr(loop, "call_llm_with_retry", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("hold bought a model call")))
    for _ in range(5):
        assert loop._maybe_enforce_child_absorption_gate(registry, ctx, candidate.full_text, ctx.messages, lambda _t: None, trace) == "continue"
    assert registry._ctx._delivery_candidate is candidate
    assert "forced_finalization" not in trace and "acceptance_decision" not in trace
    assert "child1 [running]" in str(ctx.messages) and "No reminder count ends the task" in str(ctx.messages)


# ---------------------------------------------------------------------------
# Child results hold a requested finish without any reminder-count terminal rail.
# Real round limits retain their own cause; an explicit unfinished stop is free
# of new review calls and never cancels the child's independent work.








def test_exact_child_disposition_closes_the_hold_without_counter_based_finalization(tmp_path, monkeypatch):
    from ouroboros.task_status import load_effective_task_result
    from ouroboros.tools.join_ledger import _child_result_sha256

    _write_child(tmp_path)
    loop, registry, ctx, trace = _forced_test_context(tmp_path)
    monkeypatch.setattr(loop, "call_llm_with_retry", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("hold bought a model call")))
    assert loop._maybe_enforce_child_absorption_gate(registry, ctx, "", ctx.messages, lambda _t: None, trace) == "continue"
    child = load_effective_task_result(tmp_path, "child1")
    assert _child_result_sha256(child) in str(ctx.messages) and "child1 [completed]" in str(ctx.messages)
    _write_confirmed_disposition(tmp_path, disposition="integrated", rationale="Folded the exact child result into the answer")
    assert loop._maybe_enforce_child_absorption_gate(registry, ctx, "", ctx.messages, lambda _t: None, trace) is None
    assert "acceptance_decision" not in trace


def test_child_hold_refreshes_current_state_without_a_forced_model_call(tmp_path, monkeypatch):
    from ouroboros.task_status import load_effective_task_result
    from ouroboros.tools.join_ledger import _child_result_sha256

    _write_child(tmp_path, status="running")
    loop, registry, ctx, trace = _forced_test_context(tmp_path)
    monkeypatch.setattr(loop, "call_llm_with_retry", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("hold bought a model call")))
    assert loop._maybe_enforce_child_absorption_gate(registry, ctx, "", ctx.messages, lambda _t: None, trace) == "continue"
    first = str(ctx.messages)
    _write_child(tmp_path, status="completed")
    assert loop._maybe_enforce_child_absorption_gate(registry, ctx, "", ctx.messages, lambda _t: None, trace) == "continue"
    assert "child1 [running]" in first and "child1 [completed]" in str(ctx.messages[-1])
    assert _child_result_sha256(load_effective_task_result(tmp_path, "child1")) in str(ctx.messages[-1])
    assert "forced_finalization" not in trace


def test_post_tool_evidence_change_holds_while_absorption_gate_open(tmp_path):
    """New evidence retains the child hold and an explicit finish/stop affordance."""
    _write_child(tmp_path, status="running")
    loop, registry, limit_ctx, trace = _forced_test_context(tmp_path)
    candidate = loop._replace_delivery_candidate(
        registry, limit_ctx, trace, "Retained complete answer.", control="candidate",
    )
    loop._hold_delivery_for_skill_action(
        registry, trace, control="child_absorption_or_revision_required",
    )
    registry._ctx._owner_directives = ["fresh owner directive"]
    before = len(limit_ctx.messages)

    loop._prepare_post_tool_budget_context(
        registry, limit_ctx, trace, "test-model", False, "medium",
    )

    assert registry._ctx._delivery_control_required is True
    assert candidate.finalization_control == "child_absorption_or_revision_required"
    assert all(
        "[DELIVERY_FINALIZATION_CONTROL]" not in str(m.get("content") or "")
        for m in limit_ctx.messages[before:]
    )


def test_post_tool_evidence_change_arms_after_children_dispositioned(tmp_path):
    """Once every child is dispositioned the absorption gate is closed: the
    same evidence-change seam escalates through the ordinary arm (the free
    post-action escalation path)."""
    _write_child(tmp_path)
    loop, registry, limit_ctx, trace = _forced_test_context(tmp_path)
    candidate = loop._replace_delivery_candidate(
        registry, limit_ctx, trace, "Retained complete answer.", control="candidate",
    )
    loop._hold_delivery_for_skill_action(
        registry, trace, control="child_absorption_or_revision_required",
    )
    _write_confirmed_disposition(
        tmp_path, disposition="integrated", rationale="absorbed the child result",
    )

    loop._prepare_post_tool_budget_context(
        registry, limit_ctx, trace, "test-model", False, "medium",
    )

    assert registry._ctx._delivery_control_required is True
    assert candidate.finalization_control == "effect_revision_required"
    assert any(
        "[DELIVERY_FINALIZATION_CONTROL]" in str(m.get("content") or "")
        for m in limit_ctx.messages
    )


def test_real_round_limit_still_finishes_after_child_holds_with_its_own_cause(tmp_path, monkeypatch):
    _write_child(tmp_path, status="running")
    loop, registry, ctx, trace = _forced_test_context(tmp_path)
    calls = []
    monkeypatch.setattr(loop, "call_llm_with_retry", lambda *_a, **_k: (calls.append(1) or {"role": "assistant", "content": "Useful partial answer."}, 0.0))
    for _ in range(3):
        assert loop._maybe_enforce_child_absorption_gate(registry, ctx, "", ctx.messages, lambda _t: None, trace) == "continue"
    assert calls == []
    text, usage, returned_trace = loop._handle_round_limit(ctx)
    assert text == "Useful partial answer." and calls == [1]
    assert usage["reason_code"] == "round_limit"
    assert returned_trace["acceptance_decision"]["reason"] == "acceptance_bypassed_round_limit"


def test_explicit_stop_with_running_child_finishes_without_a_panel_or_cancelling_it(tmp_path, monkeypatch):
    from ouroboros.task_status import load_effective_task_result

    _write_child(tmp_path, status="running")
    loop, registry, ctx, trace = _forced_test_context(tmp_path)
    monkeypatch.setattr(loop, "get_task_review_mode", lambda: "off")
    monkeypatch.setattr(loop, "call_llm_with_retry", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("stop bought a model call")))
    monkeypatch.setattr(loop, "_run_task_acceptance_review_once", lambda **_k: (_ for _ in ()).throw(AssertionError("stop bought a panel")))
    answer = "Useful partial answer. The child is still running."
    _select_completion(registry, ctx, trace, answer=answer, action="stop", rationale="Child results remain unfinished.")
    result = loop._no_tool_final_answer(answer, ctx, trace, registry, queue.Queue(), set(), lambda _t: None, explicit_candidate=True)
    assert result is not None and result[0] == answer
    assert trace["task_completion"]["action"] == "stop"
    assert load_effective_task_result(tmp_path, "child1")["status"] == "running"
    assert "forced_finalization" not in trace


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


def test_orphan_note_names_claimed_but_failed_disposition(monkeypatch, tmp_path):
    """W2: a child whose disposition row exists on the blackboard but no longer
    binds the current result was READ and decided — the forced orphan note says
    so instead of the misleading 'unread'. It says only what the ledger PROVES:
    the row exists, so the write did NOT fail; the binding to the current result
    is what is missing."""
    import ouroboros.loop as loop
    from ouroboros.tools.join_ledger import _child_result_sha256

    child = {
        "task_id": "child1",
        "status": "completed",
        "result": "new result the parent has not re-hashed",
    }
    monkeypatch.setattr(loop, "_direct_child_results", lambda _ctx: [dict(child)])
    monkeypatch.setattr(loop, "_child_disposition_state", lambda _child: "")
    stale_sha = "0" * 64
    assert _child_result_sha256(child) != stale_sha
    monkeypatch.setattr(
        loop,
        "_claimed_child_dispositions",
        lambda _ctx: {"child1": ("integrated", stale_sha)},
    )

    note = loop._forced_orphan_note(SimpleNamespace())

    assert "integrated recorded for an EARLIER result hash" in note
    assert "the current result is not bound" in note
    assert "child1 [completed;" in note
    # The row's existence disproves a failed write; the note must not claim one.
    assert "write failed" not in note

    # Same row, hash STILL matching: the write plainly succeeded and bound, so the
    # honest gap is the projection this round, not the ledger.
    monkeypatch.setattr(
        loop,
        "_claimed_child_dispositions",
        lambda _ctx: {"child1": ("integrated", _child_result_sha256(child))},
    )
    bound_note = loop._forced_orphan_note(SimpleNamespace())
    assert "recorded for this exact result hash" in bound_note
    assert "write failed" not in bound_note


def test_claimed_child_dispositions_reads_the_blackboard(tmp_path):
    from ouroboros.task_tree_ledger import tree_ledger_append
    import ouroboros.loop as loop

    tree_ledger_append(
        "root1", "decision", "integrated after review",
        task_id="parent1", role="orchestrator",
        payload={
            "type": "child_result_disposition", "child_task_id": "child1",
            "disposition": "integrated", "child_result_sha256": "a" * 64,
        },
        allow_child_result_disposition=True,
        data_root=tmp_path,
    )
    # A plain decision note (no typed payload) and another parent's row are ignored.
    tree_ledger_append(
        "root1", "decision", "plain note", task_id="parent1", data_root=tmp_path,
    )
    ctx = SimpleNamespace(
        status_drive_root=tmp_path, drive_root=tmp_path,
        root_task_id="root1", task_id="parent1",
    )

    claims = loop._claimed_child_dispositions(ctx)

    assert claims == {"child1": ("integrated", "a" * 64)}
    # Fail-soft on junk context.
    assert loop._claimed_child_dispositions(SimpleNamespace()) == {}

# ---------------------------------------------------------------------------
# Typed acceptance-bypass records on forced rails (W2): every forced exit that
# owed an acceptance panel stamps {finalized_unaccepted, acceptance_bypassed_<rail>}
# through the COMMON terminal recorder, covering both the LLM-seam forced answer
# and the no-spend budget fence path. Pure ledger writes: no panel, no fence, no
# extra model round.


def test_round_limit_stamps_typed_acceptance_bypass(tmp_path, monkeypatch):
    loop, _registry, limit_ctx, _trace = _forced_test_context(tmp_path)
    monkeypatch.setattr(
        loop,
        "call_llm_with_retry",
        lambda *_args, **_kwargs: (
            {"role": "assistant", "content": "Best answer before the round limit."},
            0.0,
        ),
    )

    _text, _usage, trace = loop._handle_round_limit(limit_ctx)

    # Declared deliverable with no acceptance decision -> the panel was OWED.
    assert trace["review_decision"] == {
        "eligibility": "eligible",
        "trigger": "bypassed_round_limit",
    }
    assert trace["acceptance_decision"]["status"] == "finalized_unaccepted"
    assert trace["acceptance_decision"]["reason"] == "acceptance_bypassed_round_limit"
    assert trace["acceptance_decision"]["source"] == "forced_finalization"


def test_budget_fence_no_spend_path_stamps_typed_acceptance_bypass(tmp_path, monkeypatch):
    """The physical budget fence (`_handle_budget_exceeded`) re-raises around the
    LLM seam, so the stamp must ride the common recorder, not `_forced_final_answer`."""
    import ouroboros.usage_accounting as accounting

    loop, registry, limit_ctx, trace = _forced_test_context(tmp_path)
    loop._replace_delivery_candidate(
        registry, limit_ctx, trace,
        "Best answer retained before the budget rail.", control="candidate",
    )
    monkeypatch.setattr(
        accounting,
        "usage_breakdown",
        lambda *_args, **_kwargs: {"physical_calls": 1, "integrity_degraded": False},
    )
    exit_ctx = loop._LoopExitContext(
        tools=registry, drive_root=tmp_path, task_id="parent1", event_queue=None,
        drive_logs=tmp_path / "logs", accumulated_usage=limit_ctx.accumulated_usage,
        llm_trace=trace,
    )

    _text, _usage, returned_trace = loop._handle_budget_exceeded(
        accounting.BudgetExceeded(
            "root budget closed", limit_scope="root", root_task_id="parent1",
        ),
        exit_ctx,
        limit_ctx=limit_ctx,
    )

    assert returned_trace["review_decision"] == {
        "eligibility": "eligible",
        "trigger": "bypassed_budget_exhausted",
    }
    decision = returned_trace["acceptance_decision"]
    assert decision["status"] == "finalized_unaccepted"
    assert decision["reason"] == "acceptance_bypassed_budget_exhausted"
    assert decision["source"] == "forced_finalization"


def test_forced_bypass_never_overwrites_an_existing_host_decision(tmp_path, monkeypatch):
    loop, registry, limit_ctx, trace = _forced_test_context(tmp_path)
    candidate = loop._replace_delivery_candidate(
        registry, limit_ctx, trace, "Accepted answer.", control="candidate",
    )
    _bind_host_pass(loop, registry, trace, candidate)
    monkeypatch.setattr(
        loop,
        "call_llm_with_retry",
        lambda *_args, **_kwargs: (
            {"role": "assistant", "content": "Best answer before the round limit."},
            0.0,
        ),
    )

    _text, _usage, returned_trace = loop._handle_round_limit(limit_ctx)

    # The prior host decision lane keeps authority (here the forced replacement
    # superseded the PASS through the existing revision machinery); the bypass
    # recorder never overwrites an existing decision with a bypass reason. The
    # dangling revision request itself is closed, because this rail cannot take
    # the improvement pass it promised.
    decision = returned_trace["acceptance_decision"]
    assert decision["status"] == "finalized_unaccepted"
    assert decision.get("reason", "") != "acceptance_bypassed_round_limit"
    assert decision.get("reason", "") == "revision_unavailable_on_forced_rail"
    assert decision.get("source", "") == "forced_finalization"


def test_forced_bypass_stamps_over_deferred_agent_stance(tmp_path, monkeypatch):
    """A root task_acceptance_review DEFERRED to the host leaves a STATUS-LESS
    agent-stance dict in acceptance_decision (`source` + `agent_disposition`/
    `agent_rationale` — the P4.1 merge in `process_tool_results`). That is
    evidence, not a host decision: a forced rail after it must still stamp
    finalized_unaccepted with the typed rail reason (pre-fix the recorder
    early-returned on ANY non-empty dict, so the bypass went unrecorded exactly
    when the panel was still owed), and the agent stance is carried forward."""
    from ouroboros.loop_tool_execution import process_tool_results

    loop, _registry, limit_ctx, trace = _forced_test_context(tmp_path)
    deferred_payload = json.dumps({
        "status": "deferred_to_host_acceptance",
        "authoritative": False,
        "agent_decision": {
            "disposition": "pass",
            "rationale": "agent stance recorded before the host panel",
            "source": "agent_task_acceptance_review_tool",
        },
    })
    process_tool_results(
        [{
            "fn_name": "task_acceptance_review",
            "is_error": False,
            "result": deferred_payload,
            "tool_call_id": "tc1",
            "args_for_log": {},
        }],
        [],
        trace,
        lambda _msg: None,
    )
    # The production writer's exact shape: agent stance only, no canonical status.
    assert trace["acceptance_decision"]["agent_disposition"] == "pass"
    assert "status" not in trace["acceptance_decision"]
    monkeypatch.setattr(
        loop,
        "call_llm_with_retry",
        lambda *_args, **_kwargs: (
            {"role": "assistant", "content": "Best answer before the round limit."},
            0.0,
        ),
    )

    _text, _usage, returned_trace = loop._handle_round_limit(limit_ctx)

    assert returned_trace["review_decision"] == {
        "eligibility": "eligible",
        "trigger": "bypassed_round_limit",
    }
    decision = returned_trace["acceptance_decision"]
    assert decision["status"] == "finalized_unaccepted"
    assert decision["reason"] == "acceptance_bypassed_round_limit"
    assert decision["source"] == "forced_finalization"
    # Carried forward, never overwritten (the `_set_acceptance_decision` contract).
    assert decision["agent_disposition"] == "pass"
    assert decision["agent_rationale"] == "agent stance recorded before the host panel"


def test_forced_bypass_records_not_eligible_for_child_tasks(tmp_path, monkeypatch):
    loop, registry, limit_ctx, _trace = _forced_test_context(tmp_path)
    registry._ctx.task_metadata = {
        "budget_drive_root": str(tmp_path),
        "root_task_id": "root0",
        "parent_task_id": "root0",
    }
    registry._ctx.parent_task_id = "root0"
    registry._ctx.root_task_id = "root0"
    registry._ctx.delegation_role = "subagent"
    monkeypatch.setattr(
        loop,
        "call_llm_with_retry",
        lambda *_args, **_kwargs: (
            {"role": "assistant", "content": "child best effort"},
            0.0,
        ),
    )

    _text, _usage, trace = loop._handle_round_limit(limit_ctx)

    assert trace["review_decision"]["eligibility"] == "not_eligible"
    assert trace["review_decision"]["trigger"] == "skipped_child_advisory"
    assert "acceptance_decision" not in trace


def test_forced_bypass_probe_failure_records_unknown_eligibility(tmp_path, monkeypatch):
    loop, _registry, limit_ctx, _trace = _forced_test_context(tmp_path)
    monkeypatch.setattr(
        loop,
        "call_llm_with_retry",
        lambda *_args, **_kwargs: (
            {"role": "assistant", "content": "best effort"},
            0.0,
        ),
    )
    monkeypatch.setattr(
        loop,
        "_task_acceptance_eligible",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("mid-round trace")),
    )

    _text, _usage, trace = loop._handle_round_limit(limit_ctx)

    assert trace["review_decision"] == {
        "eligibility": "unknown",
        "trigger": "bypassed_round_limit",
    }
    assert "acceptance_decision" not in trace



def test_forced_final_sends_the_round_tool_envelope(tmp_path, monkeypatch):
    """The forced wrap-up call reuses the round's exact tool envelope."""

    loop, registry, limit_ctx, _trace = _forced_test_context(tmp_path)
    schemas = [{"type": "function", "function": {"name": "read_file"}}]
    limit_ctx.tool_schemas = schemas
    seen: dict = {}

    def _capture(*args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        return {"role": "assistant", "content": "wrapped up"}, 0.0

    monkeypatch.setattr(loop, "call_llm_with_retry", _capture)

    loop._handle_round_limit(limit_ctx)

    assert seen["args"][3] is schemas
    assert seen["kwargs"]["allow_server_web_search"] == loop._server_web_allowed_by_task(
        registry._ctx
    )
    assert "tool_choice" not in seen["kwargs"]


def test_forced_final_tool_call_only_reply_falls_back_to_host_text(tmp_path, monkeypatch):
    """A tool-calls-only reply on a budget rail degrades to the host fallback text."""

    loop, _registry, limit_ctx, _trace = _forced_test_context(tmp_path)
    limit_ctx.tool_schemas = [{"type": "function", "function": {"name": "read_file"}}]
    monkeypatch.setattr(
        loop,
        "call_llm_with_retry",
        lambda *_args, **_kwargs: (
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{}"},
                    }
                ],
            },
            0.0,
        ),
    )

    text, usage, trace = loop._handle_round_limit(limit_ctx)

    assert text
    assert not usage.get("_best_effort_extracted")
    assert trace.get("forced_finalization", {}).get("source") != "forced_model_incomplete"


def test_forced_final_mixed_content_and_tool_calls_is_degraded(tmp_path, monkeypatch):
    """Content beside an unexecuted tool call is a preamble, not a final."""

    loop, _registry, limit_ctx, _trace = _forced_test_context(tmp_path)
    limit_ctx.tool_schemas = [{"type": "function", "function": {"name": "read_file"}}]
    monkeypatch.setattr(
        loop,
        "call_llm_with_retry",
        lambda *_args, **kwargs: (
            kwargs["response_meta_out"].update(tool_call_count=1, finish_reason="tool_calls") or
            {
                "role": "assistant",
                "content": "here is the answer",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "{}"},
                    }
                ],
            },
            0.0,
        ),
    )

    text, usage, trace = loop._handle_round_limit(limit_ctx)

    assert "here is the answer" in text
    assert usage.get("terminal_origin") != "model_final"
    assert trace.get("forced_finalization", {}).get("source") == "forced_model_incomplete"


def test_web_forbidding_contract_keeps_the_forced_call_web_free(tmp_path, monkeypatch):
    """A task that forbids web never gets server web search on the forced call."""

    loop, registry, limit_ctx, _trace = _forced_test_context(tmp_path)
    registry._ctx.task_contract = {"allowed_resources": {"web": False}}
    seen: dict = {}

    def _capture(*args, **kwargs):
        seen.update(kwargs)
        return {"role": "assistant", "content": "wrapped up"}, 0.0

    monkeypatch.setattr(loop, "call_llm_with_retry", _capture)

    loop._handle_round_limit(limit_ctx)

    assert seen["allow_server_web_search"] is False


def test_orphan_note_names_failed_children_and_skips_rows_this_chat_already_has(tmp_path, monkeypatch):
    """Owner item spam G, corrected: the note described children it could not
    describe, and then silenced far more than it meant to.

    The two-way clause said running children may be incomplete and completed
    ones may be unread, so a FAILED or CANCELLED child was told about as
    something it is not. The first skip then asked only whether a canonical
    receipt EXISTS, which is true of every settled task, so every completed,
    failed and cancelled child vanished from the note and the forced contract
    ("the parent may not have seen completions") was void. The receipt now
    records the chat its row went to, and only that reader is spared the repeat.

    Driven through the real writers, because a fixture that omits the receipt
    production always writes cannot show either half.
    """
    import ouroboros.loop as loop
    from ouroboros.project_dialogue import append_terminal_task_projection
    from ouroboros.task_results import write_task_result
    from ouroboros.task_status import find_child_tasks

    for tid, status in (("kid-done", "completed"), ("kid-failed", "failed")):
        child = {"id": tid, "chat_id": 7, "parent_task_id": "parent",
                 "root_task_id": "parent", "delegation_role": "subagent"}
        stored = write_task_result(tmp_path, tid, status, result=f"{tid} output", **{
            key: value for key, value in child.items() if key != "id"
        })
        assert append_terminal_task_projection(
            tmp_path, tid, child, stored, {"status": status, "chat_id": 7},
        )
    rows = find_child_tasks(tmp_path, parent_task_id="parent", root_task_id="parent",
                            exclude_task_id="parent", scope="direct")
    assert len(rows) == 2 and all(r.get("canonical_terminal_projection") for r in rows)
    monkeypatch.setattr(loop, "_direct_child_results", lambda _ctx: [dict(r) for r in rows])
    monkeypatch.setattr(loop, "_child_disposition_state", lambda _child: "")
    monkeypatch.setattr(loop, "_claimed_child_dispositions", lambda _ctx: {})

    def _ctx(chat_id):
        return SimpleNamespace(tools=SimpleNamespace(
            _ctx=SimpleNamespace(current_chat_id=chat_id)))

    # The reader that already has both terminal rows is not told twice.
    assert loop._forced_orphan_note(_ctx(7)) == ""
    # Every other reader, and an unknown one, gets the whole note.
    for elsewhere in (_ctx(1), _ctx(None), SimpleNamespace()):
        note = loop._forced_orphan_note(elsewhere)
        assert "finished ones (completed, failed or cancelled) may be UNREAD" in note
        assert "kid-done [completed]" in note and "kid-failed [failed]" in note
        assert "2 child task(s) not explicitly absorbed" in note

    # A claimed disposition that no longer binds is a DIFFERENT fact: the child's
    # terminal row never carried it, so that hint survives in its own chat too.
    from ouroboros.tools.join_ledger import _child_result_sha256

    monkeypatch.setattr(loop, "_claimed_child_dispositions",
                        lambda _ctx: {"kid-done": ("integrated", "0" * 64)})
    same_chat = loop._forced_orphan_note(_ctx(7))
    assert "kid-done [completed;" in same_chat
    assert "integrated recorded for an EARLIER result hash" in same_chat
    assert "kid-failed" not in same_chat
    assert _child_result_sha256(rows[0]) != "0" * 64


@pytest.mark.parametrize("selection", ["answer", "retained_hash", "held_hash"])
def test_forced_request_uses_explicit_completion_selection_without_tool_execution(tmp_path, monkeypatch, selection):
    from ouroboros.loop_delivery import hold_completion_response

    loop, registry, ctx, trace = _forced_test_context(tmp_path)
    retained = _arm_latch_with_candidate(loop, registry, ctx, trace)
    held = "A short correction."
    hold_completion_response(held, registry, ctx, trace)
    expected = retained.full_text if selection == "retained_hash" else held
    request = {"action": "finish"}
    request.update({"answer": held} if selection == "answer" else {
        "answer_sha256": retained.content_sha256 if selection == "retained_hash" else hashlib.sha256(held.encode()).hexdigest()})
    calls = []
    monkeypatch.setattr(loop, "call_llm_with_retry", lambda *_a, **_k: (
        calls.append(1) or {"role": "assistant", "content": json.dumps(request)}, 0.0))
    text, usage, _trace = loop._handle_round_limit(ctx)
    assert text == expected and calls == [1]
    assert usage["reason_code"] == "round_limit" and trace["tool_calls"] == []
    assert registry._ctx._delivery_candidate.full_text == expected


@pytest.mark.parametrize("body", [
    '{"action":"finish","action":"stop","answer":"Do not publish"}',
    '{"action":"finish","answer":"First","answer":"Second"}',
    '{"action":"stop","answer":"Do not publish"}',
    '{"action":"finish","answer_sha256":"unoffered"}',
])
def test_forced_invalid_completion_request_preserves_answer_without_repair_call(tmp_path, monkeypatch, body):
    loop, registry, ctx, trace = _forced_test_context(tmp_path)
    retained = _arm_latch_with_candidate(loop, registry, ctx, trace)
    calls = []
    monkeypatch.setattr(loop, "call_llm_with_retry", lambda *_a, **_k: (
        calls.append(1) or {"role": "assistant", "content": body}, 0.0))
    text, usage, _trace = loop._handle_round_limit(ctx)
    assert text.startswith(retained.full_text) and calls == [1]
    assert registry._ctx._delivery_candidate.degraded is True
    assert usage["reason_code"] == "round_limit" and trace["tool_calls"] == []
    assert body not in text


@pytest.mark.parametrize("body", [
    '{"delivery_control":"keep"}',
    '{"delivery_control":"replace","full_answer":"Unselected legacy replacement"}',
])
def test_forced_legacy_control_is_containment_only_and_cannot_select_replacement(tmp_path, monkeypatch, body):
    loop, registry, ctx, trace = _forced_test_context(tmp_path)
    retained = _arm_latch_with_candidate(loop, registry, ctx, trace)
    monkeypatch.setattr(loop, "call_llm_with_retry", lambda *_a, **_k: ({"role": "assistant", "content": body}, 0.0))
    text, usage, returned_trace = loop._handle_round_limit(ctx)
    assert text == retained.full_text
    assert "delivery_control" not in text and "Unselected legacy replacement" not in text
    assert usage["reason_code"] == "round_limit"
    assert returned_trace["delivery_candidate"]["acceptance_binding"]["authoritative"] is False
