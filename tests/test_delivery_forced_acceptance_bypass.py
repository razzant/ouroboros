"""Typed acceptance-bypass records on the forced rails.

Split verbatim out of ``tests/test_delivery_forced_finalization.py`` by theme. This
module owns the ledger writes every forced exit that owed an acceptance panel makes
through the common terminal recorder — the round-limit and no-spend budget fence
paths, the existing host decision a bypass may never overwrite, the deferred agent
stance it stamps over, and the eligibility it records for child tasks and for a failed
probe.
"""

from __future__ import annotations

import json


from tests._delivery_forced_shared import _bind_host_pass, _forced_test_context

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

    # Non-direct-chat task with no acceptance decision -> the panel was OWED.
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
    # recorder never overwrites an existing decision with a bypass reason.
    decision = returned_trace["acceptance_decision"]
    assert decision["status"] in {"accepted", "revision_requested"}
    assert decision.get("reason", "") != "acceptance_bypassed_round_limit"
    assert decision.get("source", "") != "forced_finalization"


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
