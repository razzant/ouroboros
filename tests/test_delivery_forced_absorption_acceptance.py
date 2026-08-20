"""The forced children_unabsorbed rail still runs the content acceptance review.

Split verbatim out of ``tests/test_delivery_forced_finalization.py`` by theme. This
module owns the acceptance panel that must see the undispositioned-children process
debt, the honest terminalization of an improvement pass the forced rail cannot grant,
the bypass verdict kept while the subtree is not quiescent, and the orphan labels and
notes that name a claimed but failed disposition.
"""

from __future__ import annotations

from types import SimpleNamespace

from tests._delivery_candidate_shared import (
    write_child as _write_child,
)

from tests._delivery_forced_shared import _forced_test_context

# ---------------------------------------------------------------------------
# Owner Q2A (slime saga): the forced children_unabsorbed rail must still run the
# CONTENT acceptance review through the ordinary entry point (the incident task
# finalized with zero review), the panel must see the undispositioned-children
# process debt, and a requested improvement pass (which the forced rail cannot
# grant) terminalizes honestly. The process outcome stays
# best_effort/children_unabsorbed in every branch.


def _acceptance_panel_result(*, aggregate, actors, findings=()):
    import ouroboros.review_substrate as rs

    return rs.ReviewRunResult(
        request={"surface": "task_acceptance", "policy": {"min_successful_slots": 1}},
        actors=list(actors),
        parsed_findings=list(findings),
        aggregate_signal=aggregate,
    )


def _forced_absorption_acceptance_context(tmp_path, monkeypatch, panel_result):
    loop, registry, limit_ctx, trace = _forced_test_context(tmp_path)
    registry._ctx.is_direct_chat = False
    registry._ctx._child_absorption_reminded = True
    seen_evidence: dict = {}
    panel_calls = {"count": 0}

    def panel_probe(review_ctx):
        panel_calls["count"] += 1
        seen_evidence.update(review_ctx.evidence or {})
        return panel_result

    from ouroboros import loop_acceptance_review

    monkeypatch.setattr(loop, "get_task_review_mode", lambda: "auto")
    monkeypatch.setattr(loop_acceptance_review, "_execute_task_acceptance_panel", panel_probe)
    monkeypatch.setattr(
        loop, "call_llm_with_retry",
        lambda *_a, **_k: (
            {"role": "assistant", "content": "Best-effort final answer naming child1."},
            0.0,
        ),
    )
    return loop, registry, limit_ctx, trace, seen_evidence, panel_calls


def test_forced_children_unabsorbed_rail_runs_acceptance_with_debt_evidence(
    tmp_path, monkeypatch,
):
    """A quiescent-but-undispositioned subtree: the panel RUNS on the forced rail,
    sees the undispositioned children (ids/statuses/hashes) in its evidence, and a
    clean PASS lands as `accepted` while the process outcome stays
    best_effort/children_unabsorbed."""
    from ouroboros.outcomes import derive_loop_outcome
    from ouroboros.tools.join_ledger import _child_result_sha256
    from ouroboros.task_status import load_effective_task_result

    _write_child(tmp_path)
    panel = _acceptance_panel_result(
        aggregate="PASS",
        actors=[{
            "slot_id": "s0", "signal": "PASS",
            "parsed": {
                "verdict": "PASS", "outcome_tier": "solved",
                "criteria_used": [{
                    "criterion": "owner request", "status": "supported",
                    "evidence_refs": ["artifact:1"],
                }],
            },
        }],
    )
    loop, registry, limit_ctx, trace, seen_evidence, panel_calls = (
        _forced_absorption_acceptance_context(tmp_path, monkeypatch, panel)
    )

    result = loop._maybe_enforce_child_absorption_gate(
        registry, limit_ctx, "", limit_ctx.messages, lambda _t: None, trace,
    )

    assert result is not None and result != "continue"
    text, usage, returned_trace = result
    assert usage["reason_code"] == "children_unabsorbed"
    assert panel_calls["count"] == 1
    debt = seen_evidence["undispositioned_children"]
    assert [row["task_id"] for row in debt] == ["child1"]
    assert debt[0]["status"] == "completed"
    child = load_effective_task_result(tmp_path, "child1")
    assert debt[0]["child_result_sha256"] == _child_result_sha256(child)
    decision = returned_trace["acceptance_decision"]
    assert decision["status"] == "accepted"
    assert decision["reason"] == "clean_pass"
    # The ctx stash is scoped to the forced run only.
    assert registry._ctx._forced_undispositioned_children is None
    outcome = derive_loop_outcome(text, usage, returned_trace)
    assert outcome["outcome_axes"]["execution"]["status"] == "best_effort"
    assert outcome["outcome_axes"]["execution"]["reason_code"] == "children_unabsorbed"


def test_forced_rail_terminalizes_a_requested_improvement_pass(tmp_path, monkeypatch):
    """The panel asks for a revision pass, but the forced rail can never take
    another model round: the dangling `revision_requested` is downgraded to the
    honest terminal `finalized_unaccepted` with a typed reason."""
    import ouroboros.task_pacing as task_pacing

    _write_child(tmp_path)
    panel = _acceptance_panel_result(
        aggregate="FAIL",
        actors=[{
            "slot_id": "s0", "signal": "FAIL",
            "parsed": {
                "verdict": "FAIL", "outcome_tier": "blocked_with_evidence",
                "completion_coach": "fix it", "dialogue_status": "continue_actionable",
            },
        }],
        findings=[{
            "slot_id": "s0", "severity": "critical", "item": "broken",
            "recommendation": "fix the header",
        }],
    )
    loop, registry, limit_ctx, trace, _seen_evidence, panel_calls = (
        _forced_absorption_acceptance_context(tmp_path, monkeypatch, panel)
    )
    monkeypatch.setattr(
        task_pacing, "improvement_pass_allowed", lambda *_a, **_k: (True, ""),
    )

    result = loop._maybe_enforce_child_absorption_gate(
        registry, limit_ctx, "", limit_ctx.messages, lambda _t: None, trace,
    )

    assert result is not None and result != "continue"
    _text, usage, returned_trace = result
    assert usage["reason_code"] == "children_unabsorbed"
    assert panel_calls["count"] == 1
    decision = returned_trace["acceptance_decision"]
    assert decision["status"] == "finalized_unaccepted"
    assert decision["reason"] == "revision_unavailable_on_forced_rail"
    assert registry._ctx._task_acceptance_reviewed is True


def test_forced_rail_keeps_bypass_verdict_when_subtree_is_not_quiescent(
    tmp_path, monkeypatch,
):
    """A still-RUNNING child means the panel structurally cannot bind stable
    evidence (the voluntary path would WAIT, which the forced rail cannot):
    the panel never runs and the typed acceptance-bypass verdict stamped by
    the forced-finalization recorder stays as the terminal truth."""
    _write_child(tmp_path, status="running")
    panel = _acceptance_panel_result(aggregate="PASS", actors=[])
    loop, registry, limit_ctx, trace, _seen_evidence, panel_calls = (
        _forced_absorption_acceptance_context(tmp_path, monkeypatch, panel)
    )

    result = loop._maybe_enforce_child_absorption_gate(
        registry, limit_ctx, "", limit_ctx.messages, lambda _t: None, trace,
    )

    assert result is not None and result != "continue"
    _text, usage, returned_trace = result
    assert usage["reason_code"] == "children_unabsorbed"
    assert panel_calls["count"] == 0
    decision = returned_trace["acceptance_decision"]
    assert decision["status"] == "finalized_unaccepted"
    assert decision["reason"] == "acceptance_bypassed_children_unabsorbed"


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
    from ouroboros import loop_forced_finalization
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
        loop_forced_finalization,
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
        loop_forced_finalization,
        "_claimed_child_dispositions",
        lambda _ctx: {"child1": ("integrated", _child_result_sha256(child))},
    )
    bound_note = loop._forced_orphan_note(SimpleNamespace())
    assert "recorded for this exact result hash" in bound_note
    assert "write failed" not in bound_note


def test_claimed_child_dispositions_reads_the_blackboard(tmp_path):
    from ouroboros.task_tree_ledger import tree_ledger_append
    from ouroboros import loop_forced_finalization

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

    claims = loop_forced_finalization._claimed_child_dispositions(ctx)

    assert claims == {"child1": ("integrated", "a" * 64)}
    # Fail-soft on junk context.
    assert loop_forced_finalization._claimed_child_dispositions(SimpleNamespace()) == {}
