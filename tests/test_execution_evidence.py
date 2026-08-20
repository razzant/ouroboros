"""Completion-seam execution evidence (route labels = DECISIONS, ledger = EVIDENCE).

A live test (a1d69c6c) showed two mutating children whose durable records and UI
chips said `executor_route=claude` — a dispatch-time decision — while the custody
rows showed ZERO delegated runs: 100% of their cognition ran metered, and the card
read the label as a receipt. The fix reconciles the route against the task's own
delegate custody rows exactly once, at ``subagents.envelope_from_task``, into ONE
additive ``execution_evidence`` field. The dispatch decision is never overwritten.
"""
from __future__ import annotations

import json

from ouroboros import delegate_custody as custody
from ouroboros.subagents import envelope_from_task


def _drive(tmp_path):
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    return tmp_path


def _subagent_task(tmp_path, **extra):
    return {
        "id": "child-1",
        "parent_task_id": "root-1",
        "root_task_id": "root-1",
        "delegation_role": "subagent",
        "requested_executor": "harness",
        "effective_executor": "harness",
        "executor_route": "claude",
        "model": "openai/gpt-5.6-terra",
        "budget_drive_root": str(tmp_path),
        **extra,
    }


def _emit_started(drive, run_id="run-1", task_id="child-1", model=""):
    assert custody.emit(drive, custody.STARTED, {
        "run_id": run_id, "task_id": task_id, "route": "claude", "model": model,
        "max_seconds": 300,
    })


def _emit_settled(drive, run_id="run-1", task_id="child-1", *,
                  cost_usd=0.0, spend_disclosed=True, model="claude-sonnet",
                  spend_estimated=False, state="succeeded"):
    assert custody.emit(drive, custody.SETTLED, {
        "run_id": run_id, "task_id": task_id, "route": "claude", "model": model,
        "state": state, "cost_usd": cost_usd,
        "cost_final": spend_disclosed and not spend_estimated,
        "spend_disclosed": spend_disclosed, "spend_estimated": spend_estimated,
    })


class TestCustodyAggregation:
    def test_no_rows_is_zero_runs_not_an_error(self, tmp_path):
        drive = _drive(tmp_path)
        evidence = custody.task_execution_evidence(drive, "child-1")
        assert evidence == {
            "delegated_runs_started": 0,
            "delegated_runs_settled": 0,
            "delegated_runs_succeeded": 0,
            "delegated_runs_failed": 0,
            "delegated_run_failure_states": [],
            "evidence_read_failed": False,
            "nanny_nudge_recorded": False,
            "delegate_start_attempted": False,
            "subscription_cost_usd": None,
            "subscription_cost_estimated": False,
            "harness_models": [],
        }

    def test_started_and_settled_runs_aggregate_with_disclosed_spend(self, tmp_path):
        drive = _drive(tmp_path)
        _emit_started(drive, "run-1")
        _emit_settled(drive, "run-1", cost_usd=0.0)
        _emit_started(drive, "run-2")
        evidence = custody.task_execution_evidence(drive, "child-1")
        assert evidence["delegated_runs_started"] == 2
        assert evidence["delegated_runs_settled"] == 1
        assert evidence["subscription_cost_usd"] == 0.0
        assert evidence["harness_models"] == ["claude-sonnet"]

    def test_harness_models_lists_engine_reported_models_only(self, tmp_path):
        # A STARTED row carries the REQUESTED pin — with an owner default model
        # it is routinely non-empty, and listing it would name a model that
        # never executed. Only SETTLED rows are engine-reported.
        drive = _drive(tmp_path)
        _emit_started(drive, "run-1", model="sonnet")
        _emit_settled(drive, "run-1", model="claude-opus-5")
        _emit_started(drive, "run-2", model="sonnet")   # started, never settled
        evidence = custody.task_execution_evidence(drive, "child-1")
        assert evidence["harness_models"] == ["claude-opus-5"]

    def test_estimated_spend_is_flagged_not_dressed_as_exact(self, tmp_path):
        # The settlement row's estimated/final distinction rides into the
        # aggregate: an estimated sum must never render as an exact receipt.
        drive = _drive(tmp_path)
        _emit_started(drive, "run-1")
        _emit_settled(drive, "run-1", cost_usd=0.42, spend_estimated=True)
        evidence = custody.task_execution_evidence(drive, "child-1")
        assert evidence["subscription_cost_usd"] == 0.42
        assert evidence["subscription_cost_estimated"] is True

    def test_undisclosed_spend_never_renders_as_zero(self, tmp_path):
        drive = _drive(tmp_path)
        _emit_started(drive, "run-1")
        _emit_settled(drive, "run-1", cost_usd=None, spend_disclosed=False)
        evidence = custody.task_execution_evidence(drive, "child-1")
        assert evidence["delegated_runs_settled"] == 1
        assert evidence["subscription_cost_usd"] is None

    def test_failed_run_reads_as_attempted_route_not_zero_attempts(self, tmp_path):
        # F4 (2026-08-10 saga): a run that STARTED and FAILED is an ATTEMPTED
        # route. The terminal-state axis lets readers (the nanny nudge) tell
        # "never tried" from "tried and the run died" without accusing the child.
        drive = _drive(tmp_path)
        _emit_started(drive, "run-1")
        assert custody.emit(drive, custody.SETTLED, {
            "run_id": "run-1", "task_id": "child-1", "route": "claude",
            "model": "claude-opus-5", "state": "failed", "cost_usd": 0.0,
            "cost_final": True, "spend_disclosed": True, "spend_estimated": False,
        })
        evidence = custody.task_execution_evidence(drive, "child-1")
        assert evidence["delegated_runs_started"] == 1
        assert evidence["delegated_runs_succeeded"] == 0
        assert evidence["delegated_runs_failed"] == 1
        assert evidence["delegated_run_failure_states"] == ["failed"]
        # A succeeded run counts on the success axis and adds no failure state.
        _emit_started(drive, "run-2")
        _emit_settled(drive, "run-2")
        evidence = custody.task_execution_evidence(drive, "child-1")
        assert evidence["delegated_runs_succeeded"] == 1
        assert evidence["delegated_run_failure_states"] == ["failed"]

    def test_another_tasks_runs_do_not_leak_in(self, tmp_path):
        drive = _drive(tmp_path)
        _emit_started(drive, "run-9", task_id="other-task")
        _emit_settled(drive, "run-9", task_id="other-task")
        evidence = custody.task_execution_evidence(drive, "child-1")
        assert evidence["delegated_runs_started"] == 0
        assert evidence["delegated_runs_settled"] == 0


class TestEnvelopeReconciliation:
    def test_dispatched_route_with_no_runs_reads_zero_evidence(self, tmp_path):
        # Direction 1 (the live defect): route decided, nothing delegated —
        # the envelope now SAYS so, beside the untouched dispatch decision.
        drive = _drive(tmp_path)
        task = _subagent_task(drive)
        envelope = envelope_from_task(task, status="completed")
        assert envelope["executor_route"] == "claude"          # decision, untouched
        assert envelope["effective_executor"] == "harness"     # decision, untouched
        assert envelope["execution_evidence"] == {
            "delegated_runs_started": 0,
            "delegated_runs_settled": 0,
            "delegated_runs_succeeded": 0,
            "delegated_runs_failed": 0,
            "delegated_run_failure_states": [],
            "evidence_read_failed": False,
            "nanny_nudge_recorded": False,
            "delegate_start_attempted": False,
            "subscription_cost_usd": None,
            "subscription_cost_estimated": False,
            "harness_models": [],
        }

    def test_settled_delegate_rows_read_as_harness_evidence(self, tmp_path):
        # Direction 2: real delegated runs settle -> counts, spend and the
        # engine-reported model land on the envelope.
        drive = _drive(tmp_path)
        _emit_started(drive, "run-1")
        _emit_settled(drive, "run-1", cost_usd=0.0)
        task = _subagent_task(drive)
        envelope = envelope_from_task(task, status="completed")
        evidence = envelope["execution_evidence"]
        assert evidence["delegated_runs_started"] == 1
        assert evidence["delegated_runs_settled"] == 1
        assert evidence["subscription_cost_usd"] == 0.0
        assert evidence["harness_models"] == ["claude-sonnet"]

    def test_running_envelope_carries_no_evidence(self, tmp_path):
        # Pre-completion there is no evidence to state: the chip stays the
        # neutral "dispatched" decision, so the field must be ABSENT, not zeroed.
        drive = _drive(tmp_path)
        task = _subagent_task(drive)
        envelope = envelope_from_task(task, status="running")
        assert "execution_evidence" not in envelope

    def test_native_child_has_no_delegation_claim_to_reconcile(self, tmp_path):
        drive = _drive(tmp_path)
        task = _subagent_task(drive, executor_route="", effective_executor="native")
        envelope = envelope_from_task(task, status="completed")
        assert "execution_evidence" not in envelope

    def test_evidence_read_failure_never_breaks_completion(self, tmp_path, monkeypatch):
        drive = _drive(tmp_path)
        task = _subagent_task(drive)

        def _boom(*a, **k):
            raise OSError("event log unreadable")

        monkeypatch.setattr(custody, "task_execution_evidence", _boom)
        envelope = envelope_from_task(task, status="completed")
        assert "execution_evidence" not in envelope
        assert envelope["executor_route"] == "claude"


class TestActualSubstrate:
    """Q1A (2026-08-10 amendments): the PLAN (`effective_executor`) and the FACT
    (`actual_substrate`) are separate fields — a harness-dispatched task that ran
    everything on metered API must not read as a clean delegated execution."""

    def test_vocabulary_is_purely_factual_from_custody_counts(self):
        # Custody evidence ONLY — no usage/rounds axis, where polling and real
        # thinking are indistinguishable and any boundary would be a guess.
        from ouroboros.subagents import actual_substrate

        assert actual_substrate(None) == "native_only"
        assert actual_substrate({"delegated_runs_started": 0}) == "native_only"
        # Started-but-failed is a FAILED ATTEMPT, not "never tried".
        assert actual_substrate({"delegated_runs_started": 2,
                                 "delegated_runs_succeeded": 0}) == "harness_attempted"
        assert actual_substrate({"delegated_runs_started": 1,
                                 "delegated_runs_succeeded": 1}) == "harness_used"

    def test_attempted_run_classifies_attempted_in_the_envelope(self, tmp_path):
        drive = _drive(tmp_path)
        _emit_started(drive, "run-1")
        _emit_settled(drive, "run-1", state="failed")
        envelope = envelope_from_task(_subagent_task(drive), status="completed")
        assert envelope["actual_substrate"] == "harness_attempted"

    def test_envelope_carries_the_fact_beside_the_plan(self, tmp_path):
        drive = _drive(tmp_path)
        _emit_started(drive, "run-1")
        _emit_settled(drive, "run-1")
        envelope = envelope_from_task(_subagent_task(drive), status="completed",
                                      usage={"rounds": 4})
        assert envelope["effective_executor"] == "harness"   # the plan, untouched
        assert envelope["actual_substrate"] == "harness_used"

    def test_native_only_harness_dispatch_discloses_a_reduced_delta(self, tmp_path):
        # The e9108a09 shape: dispatched harness, zero delegated runs. The
        # completion envelope must not present a clean un-reduced execution —
        # the EXISTING capability_delta disclosure carries it (no new axis).
        drive = _drive(tmp_path)
        task = _subagent_task(drive, capability_delta={
            "requested_executor": "auto", "effective_executor": "harness",
            "reason": "", "reduced": False,
        })
        envelope = envelope_from_task(task, status="completed", usage={"rounds": 9})
        assert envelope["actual_substrate"] == "native_only"
        assert envelope["capability_delta"]["reduced"] is True
        assert "delegated_substrate_unused" in envelope["capability_delta"]["reason"]
        # The dispatch-time author's dict on the task stays untouched.
        assert task["capability_delta"]["reduced"] is False
        # And the batch-projection predicate now discloses it to the parent.
        from ouroboros.tools.control import disclosable_capability_delta

        assert disclosable_capability_delta({"capability_delta": envelope["capability_delta"]})

    def test_durable_result_fields_carry_the_raw_counts_beside_the_enum(self, tmp_path):
        from ouroboros.subagents import substrate_result_fields

        drive = _drive(tmp_path)
        _emit_started(drive, "run-1")
        envelope = envelope_from_task(_subagent_task(drive), status="completed")
        assert substrate_result_fields(envelope) == {
            "actual_substrate": "harness_attempted",
            "delegated_runs_started": 1,
            "delegated_runs_settled": 0,
            "delegated_runs_succeeded": 0,
            "delegated_runs_failed": 0,
            "native_contribution": "unknown",
        }
        assert substrate_result_fields({}) == {}  # no substrate claim, no fields

    def test_unreadable_evidence_makes_no_substrate_claim_anywhere(self, tmp_path):
        # 6c03c24e corrective wave (both sol lanes + fable): an unreadable
        # canonical custody log returns zero counts with evidence_read_failed —
        # those zeros are UNKNOWN, so the envelope must not classify them as
        # native_only, must not add the delegated_substrate_unused reduction,
        # and the durable result must carry no top-level substrate fields.
        from ouroboros import delegate_custody as custody
        from ouroboros.subagents import substrate_result_fields

        drive = _drive(tmp_path)
        log_path = custody.event_log_path(drive)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.mkdir()  # a directory where the file should be -> OSError
        task = _subagent_task(drive, capability_delta={"reduced": False, "reason": ""})
        envelope = envelope_from_task(task, status="completed")
        assert envelope["execution_evidence"]["evidence_read_failed"] is True
        assert "actual_substrate" not in envelope
        assert envelope["capability_delta"]["reduced"] is False
        assert "delegated_substrate_unused" not in str(envelope["capability_delta"].get("reason") or "")
        assert substrate_result_fields(envelope) == {}

    def test_delegated_success_does_not_amend_the_delta(self, tmp_path):
        drive = _drive(tmp_path)
        _emit_started(drive, "run-1")
        _emit_settled(drive, "run-1")
        task = _subagent_task(drive, capability_delta={"reduced": False, "reason": ""})
        envelope = envelope_from_task(task, status="completed", usage={"rounds": 4})
        assert envelope["capability_delta"]["reduced"] is False

    def test_running_and_native_envelopes_carry_no_substrate_claim(self, tmp_path):
        drive = _drive(tmp_path)
        running = envelope_from_task(_subagent_task(drive), status="running")
        assert "actual_substrate" not in running
        native = envelope_from_task(
            _subagent_task(drive, executor_route="", effective_executor="native"),
            status="completed")
        assert "actual_substrate" not in native


def test_terminal_frame_field_rides_the_history_replay_allowlist():
    # The chip's layered truth must survive a reload: the terminal frame carries
    # execution_evidence, and history replay filters progress meta by this list.
    from ouroboros.gateway.history import _PROGRESS_META_FIELDS

    assert "execution_evidence" in _PROGRESS_META_FIELDS
    assert "executor_route" in _PROGRESS_META_FIELDS


def test_run_timing_reads_the_started_row(tmp_path):
    drive = _drive(tmp_path)
    _emit_started(drive, "run-1")
    started_ts, max_seconds = custody.run_timing(drive, "run-1")
    assert started_ts  # the emit stamped ts
    assert max_seconds == 300
    assert custody.run_timing(drive, "run-unknown") == ("", 0)


def test_evidence_is_json_serializable(tmp_path):
    drive = _drive(tmp_path)
    _emit_started(drive, "run-1")
    _emit_settled(drive, "run-1")
    envelope = envelope_from_task(_subagent_task(drive), status="failed")
    json.dumps(envelope)


class TestEvidenceReadHonesty:
    def test_unreadable_log_is_flagged_not_zero(self, tmp_path):
        # Scope finding (a2a6253e gate lineage): an EXISTING but unreadable
        # canonical log must not collapse into "zero attempts established" —
        # a directory at the log path forces the open() OSError portably.
        from ouroboros import delegate_custody as custody

        log_path = custody.event_log_path(tmp_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.mkdir()  # a directory where the file should be
        evidence = custody.task_execution_evidence(tmp_path, "t1")
        assert evidence["evidence_read_failed"] is True
        assert evidence["delegated_runs_started"] == 0

    def test_nanny_never_accuses_on_unreadable_evidence(self, tmp_path):
        from types import SimpleNamespace
        from ouroboros import delegate_custody as custody
        from ouroboros.loop_nudges import _maybe_inject_finalization_nudges

        log_path = custody.event_log_path(tmp_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.mkdir()
        ctx = SimpleNamespace(_nanny_route_dispatched=True, _nanny_finalization_injected=False)
        tools = SimpleNamespace(_ctx=ctx, available_tools=lambda: ["delegate_start"])
        msgs: list = []
        assert _maybe_inject_finalization_nudges(
            tools, tmp_path, "t1",
            {"reasoning_notes": [], "tool_calls": []}, "done", msgs, lambda *_: None,
        ) is False
        assert not any("NANNY" in m.get("content", "") for m in msgs)
