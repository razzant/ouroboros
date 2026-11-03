"""The consumer matrix for custody OWNER KINDS (issue #1006).

A custody row carries the kind of the surface that registered it, and every
DELEGATION-DOMAIN reader consumes that kind: a run a review panel started is
the panel's obligation, never the task's delegation slot. Physical custody
(settlement, ledger, containment, patch disposition, registration retirement)
keeps seeing every run, which is why the matrix below asserts sparing and
settlement side by side.

**Adding a delegation-domain reader of custody rows means adding it here.** The
rule is only as strong as its enforcing surface: the incident behind issue #1006 happened
because ARCHITECTURE already stated the invariant while the orphan sweep, the
terminal audit, the evidence counters and the hold never read it.
"""

from __future__ import annotations

import pytest

from ouroboros import delegate_custody as dc

from tests._delegated_transport_shared import (  # noqa: F401  (autouse fixture applies on import)
    _LiveRunStub,
    _owned_gateway_uses_each_test_transport,
)

TASK = "t-mixed"


def _mixed_population(drive):
    """One task owning: its own leaf, a live reviewer, a settled reviewer, and a
    review invocation that never bound a run. Reviewer rows carry the read-only
    Ask shape their producer (``review_execution``) records."""
    seat = {"access": "readonly", "mode": "ask"}
    dc._CUSTODY.clear()
    dc.record_started(drive, dc.RunCustody(
        run_id="run-leaf", task_id=TASK, route_id="codex", model="leaf-model",
        ledger_root=str(drive)))
    dc.record_started(drive, dc.RunCustody(
        run_id="run-review", task_id=TASK, route_id="codex", model="review-model",
        source="review_substrate", category="task_acceptance_review",
        review_slot_id="triad_286lhb", ledger_root=str(drive)), shape=seat)
    dc.record_started(drive, dc.RunCustody(
        run_id="run-review-done", task_id=TASK, route_id="codex",
        source="review_substrate:task_acceptance", ledger_root=str(drive)), shape=seat)
    assert dc.emit(drive, dc.SETTLED, {
        "run_id": "run-review-done", "task_id": TASK, "route": "codex",
        "model": "review-model", "state": "failed", "source": "review_substrate",
        "category": "task_acceptance_review", "cost_usd": 4.0,
        "cost_final": True, "spend_disclosed": True})
    assert dc.record_start_requested(
        drive, run_id="", task_id=TASK, invocation_id="inv-review",
        idempotency_key="inv-review", request={"prompt": "review packet", **seat},
        route="codex", source="review_substrate.extraction")
    dc._CUSTODY.clear()


@pytest.mark.parametrize("consumer", [
    "terminal_audit", "execution_evidence", "unknown_hold",
    "orphan_sweep", "pending_invocation", "owner_cancellation", "continue_writer_check",
])
def test_only_the_tasks_own_delegation_is_its_delegation(
    tmp_path, monkeypatch, consumer,
):
    """Every delegation-domain reader answers with the leaf alone; the panel's
    rows stay live, retained and out of the task's story."""
    _mixed_population(tmp_path)

    if consumer == "terminal_audit":
        from ouroboros import delegate_terminal

        result: dict = {"task_id": TASK, "outcomes": [], "unreconciled": [],
                        "audit_status": "ok"}
        delegate_terminal._audit_task_custody(
            tmp_path, TASK, result, emit_evidence=False)
        assert result["open_run_ids"] == ["run-leaf"]
        assert result["pending_invocation_ids"] == []
        assert [row["run_id"] for row in result["terminal_runs"]] == []
        assert result["unreconciled"] == ["run-leaf"]
        assert delegate_terminal.terminal_custody_notice(
            {"delegate_terminal_reconciliation": result},
        ).startswith("Open delegated execution: run-leaf")
        return

    if consumer == "execution_evidence":
        evidence = dc.task_execution_evidence(tmp_path, TASK)
        assert evidence["delegated_runs_started"] == 1
        assert evidence["delegated_runs_settled"] == 0
        assert evidence["delegated_runs_failed"] == 0
        assert evidence["delegated_run_failure_states"] == []
        assert evidence["harness_models"] == []
        assert evidence["subscription_cost_usd"] is None
        assert evidence["delegate_start_attempted"] is True
        return

    if consumer == "unknown_hold":
        import ouroboros.claudexor_daemon as daemon_mod
        import ouroboros.delegate_hold as delegate_hold
        import ouroboros.delegate_progress as progress_mod
        from types import SimpleNamespace

        from ouroboros.tools.registry import ToolContext

        monkeypatch.setattr(daemon_mod, "ensure_owned_gateway",
                            lambda **_k: SimpleNamespace(close=lambda: None),
                            raising=False)
        monkeypatch.setattr(
            progress_mod, "bounded_poll",
            lambda _gw, _run, _sec, **_k: {"summary": {"state": "running"}, "lastSeq": 1})
        (tmp_path / "repo").mkdir(parents=True, exist_ok=True)
        ctx = ToolContext(repo_dir=str(tmp_path / "repo"), drive_root=str(tmp_path))
        ctx.task_id = TASK
        assert delegate_hold._eligible_live_runs(ctx) == ["run-leaf"]
        return

    if consumer == "continue_writer_check":
        # #1572: the owner's Continue reads the panel's rows as independent reviews
        # without candidate-apply possibility; every other census keeps them.
        from ouroboros.task_results import write_task_result
        from supervisor.continuation_admission import CONTINUE_PURPOSE, conflicting_writers
        from tests._budget_pause_exact_helpers import _install_queue

        queue, _state, _workers = _install_queue(tmp_path, monkeypatch)
        write_task_result(tmp_path, TASK, "cancelled", root_task_id=TASK, chat_id=0, result="interrupted")
        delegated = lambda rows: sorted(row.get("run_id") or row.get("invocation_id")
                                        for row in rows if row["kind"] == "delegated_run")
        assert delegated(conflicting_writers(queue, TASK)) == ["inv-review", "run-leaf", "run-review"]
        assert delegated(conflicting_writers(queue, TASK, purpose=CONTINUE_PURPOSE)) == ["run-leaf"]
        dc._CUSTODY.clear()
        return

    if consumer == "pending_invocation":
        class _NeverStarts(_LiveRunStub):
            def start_run(self, request, *, idempotency_key=""):
                pytest.fail("a review panel rejoins its own invocation")

        outcomes = dc.reconcile_orphaned_runs(
            tmp_path, set(), gateway_factory=_NeverStarts, replay_pending=lambda: True)
        # The periodic sweep leaves the panel's invocation to its panel: no outcome, still pending.
        assert [row for row in outcomes if row.get("invocation_id")] == []
        assert "inv-review" in {record["invocation_id"] for record in dc.pending_invocations(tmp_path)}
        dc._CUSTODY.clear()
        return

    from ouroboros.task_results import write_task_result

    if consumer == "orphan_sweep":
        # The incident: the task completed, so the sweep read its own leaf as
        # abandoned and the reviewer as abandoned with it.
        write_task_result(tmp_path, TASK, "completed", result="verdict")
        transport = _LiveRunStub()
        by_run = {row["run_id"]: row for row in dc.reconcile_orphaned_runs(
            tmp_path, set(), gateway_factory=lambda: transport)
            if row.get("run_id")}
        assert by_run["run-leaf"]["action"] == "cancelled"
        assert "reason" not in by_run["run-leaf"]
        assert by_run["run-review"]["action"] == "left_live"
        assert by_run["run-review"]["state"] == "running"
        assert by_run["run-review"]["reason"] == "review_panel_owns_run"
        assert transport.cancels == [("run-leaf", "owner_task_gone")]
        dc._CUSTODY.clear()
        return

    # owner_cancellation: the ONE verdict that also speaks for the panel.
    write_task_result(tmp_path, TASK, "cancelled", result="stopped by owner")
    transport = _LiveRunStub()
    by_run = {row["run_id"]: row for row in dc.reconcile_orphaned_runs(
        tmp_path, set(), gateway_factory=lambda: transport) if row.get("run_id")}
    assert by_run["run-review"]["action"] == "cancelled"
    assert by_run["run-leaf"]["action"] == "cancelled"
    assert sorted(transport.cancels) == [
        ("run-leaf", "owner_task_gone"), ("run-review", "owner_task_gone")]
    dc._CUSTODY.clear()
