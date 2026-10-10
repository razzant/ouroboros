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

import pytest

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
    def test_applied_access_rides_settled_rows(self, tmp_path):
        """D29 projection: the engine-served access (effectiveAccess) on the
        SETTLED row reaches the evidence so asked-vs-applied is answerable
        without joining the ledger; absent receipts leave the list empty."""
        drive = _drive(tmp_path)
        _emit_started(drive, "run-1")
        assert custody.emit(drive, custody.SETTLED, {
            "run_id": "run-1", "task_id": "child-1", "route": "claude",
            "model": "m", "state": "succeeded", "cost_usd": 0.0,
            "cost_final": True, "spend_disclosed": True,
            "spend_estimated": False, "access_profile": "workspace_write",
        })
        evidence = custody.task_execution_evidence(drive, "child-1")
        assert evidence["applied_access_profiles"] == ["workspace_write"]

    def test_inherited_source_debt_counts_for_the_continuing_task(self, tmp_path):
        """A continuation that adopted a partial-source predecessor's snapshot carries
        that source debt (``inherited_sources``) even when its own brief was complete,
        so the task's evidence must not count the run as a clean success."""
        drive = _drive(tmp_path)
        inherited = {
            "work_order_source_request": {"schema": 1, "kind": "complete_work_order",
                                          "coverage": "partial", "complete_chars": 100,
                                          "complete_sha256": "0" * 64},
            "work_order_fingerprint": "f" * 64, "work_order_coverage": "partial",
            "verified_source_ranges": [[0, 40]],
        }
        for run_id, request in (("run-plain", {}), ("run-cont", {"inherited_sources": [inherited]})):
            assert custody.emit(drive, custody.STARTED, {
                "run_id": run_id, "task_id": "child-1", "route": "claude", "model": "",
                "max_seconds": 300, "work_order_coverage": "complete",
                "work_order_source_request": request,
            })
            _emit_settled(drive, run_id)
        evidence = custody.task_execution_evidence(drive, "child-1")
        assert evidence["delegated_runs_source_unresolved"] == 1
        assert evidence["delegated_runs_succeeded"] == 1  # the plain run stays a success

    def test_no_rows_is_zero_runs_not_an_error(self, tmp_path):
        drive = _drive(tmp_path)
        evidence = custody.task_execution_evidence(drive, "child-1")
        assert evidence == {
            "delegated_runs_started": 0,
            "delegated_runs_settled": 0,
            "delegated_runs_succeeded": 0,
            "delegated_runs_failed": 0,
            "delegated_runs_source_unresolved": 0,
            "delegated_run_failure_states": [],
            "evidence_read_failed": False,
            "nanny_nudge_recorded": False,
            "delegate_start_attempted": False,
            "subscription_cost_usd": None,
            "subscription_cost_estimated": False,
            "harness_models": [],
            "applied_access_profiles": [],
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

    def test_the_tasks_own_reviewers_are_not_its_substrate(self, tmp_path):
        """Issue #1006: a run a REVIEW panel registered under the reviewed
        task's id is the panel's substrate. Its attempt, counters, model,
        applied access and failure state are not this task's evidence."""
        drive = _drive(tmp_path)
        _emit_started(drive, "run-leaf")
        assert custody.emit(drive, custody.START_REQUESTED, {
            "run_id": "", "task_id": "child-1", "invocation_id": "inv-review",
            "idempotency_key": "inv-review", "request": {"prompt": "packet"},
            "route": "codex", "source": "review_substrate.extraction",
        })
        assert custody.emit(drive, custody.STARTED, {
            "run_id": "run-review", "task_id": "child-1", "route": "codex",
            "model": "review-pin", "source": "review_substrate",
            "category": "task_acceptance_review",
        })
        assert custody.emit(drive, custody.SETTLED, {
            "run_id": "run-review", "task_id": "child-1", "route": "codex",
            "model": "review-model", "state": "failed", "cost_usd": 4.0,
            "cost_final": True, "spend_disclosed": True,
            "access_profile": "workspace_write",
        })
        evidence = custody.task_execution_evidence(drive, "child-1")
        assert evidence["delegated_runs_started"] == 1
        assert evidence["delegated_runs_settled"] == 0
        assert evidence["delegated_runs_failed"] == 0
        assert evidence["delegated_run_failure_states"] == []
        assert evidence["harness_models"] == []
        assert evidence["applied_access_profiles"] == []
        assert evidence["subscription_cost_usd"] is None
        # The leaf started, so delegation still provably happened.
        assert evidence["delegate_start_attempted"] is True

    def test_a_reviewers_settlement_that_outlived_its_start_names_itself(self, tmp_path):
        """After log rotation the SETTLED row may be all that survives — it
        carries its own ``source``, so it is still the panel's, not the task's.
        A settled row WITHOUT a source keeps counting as this task's run."""
        drive = _drive(tmp_path)
        assert custody.emit(drive, custody.SETTLED, {
            "run_id": "run-review", "task_id": "child-1", "route": "codex",
            "model": "review-model", "state": "failed", "source": "review_substrate",
            "cost_usd": 4.0, "cost_final": True, "spend_disclosed": True,
        })
        evidence = custody.task_execution_evidence(drive, "child-1")
        assert evidence["delegated_runs_started"] == 0
        assert evidence["delegated_run_failure_states"] == []
        assert evidence["harness_models"] == []

        _emit_settled(drive, "run-rotated", state="failed")
        evidence = custody.task_execution_evidence(drive, "child-1")
        assert evidence["delegated_runs_started"] == 1
        assert evidence["delegated_run_failure_states"] == ["failed"]
        assert evidence["harness_models"] == ["claude-sonnet"]


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
            "delegated_runs_source_unresolved": 0,
            "delegated_run_failure_states": [],
            "evidence_read_failed": False,
            "nanny_nudge_recorded": False,
            "delegate_start_attempted": False,
            "subscription_cost_usd": None,
            "subscription_cost_estimated": False,
            "harness_models": [],
            "applied_access_profiles": [],
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
            "delegated_runs_source_unresolved": 0,
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
    assert "actual_substrate" in _PROGRESS_META_FIELDS


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
        from ouroboros.loop import _maybe_inject_finalization_nudges

        log_path = custody.event_log_path(tmp_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.mkdir()
        ctx = SimpleNamespace(_nanny_route_dispatched=True, _nanny_finalization_injected=False,
                              drive_root=tmp_path)
        tools = SimpleNamespace(_ctx=ctx, available_tools=lambda: ["delegate_start"])
        msgs: list = []
        assert _maybe_inject_finalization_nudges(
            tools, tmp_path, "t1",
            {"reasoning_notes": [], "tool_calls": []}, "done", msgs, lambda *_: None,
        ) is False
        assert not any("NANNY" in m.get("content", "") for m in msgs)


class TestTerminalFrameDelivery:
    """The supervisor seam that DELIVERS the evidence to the card.

    ``_finish_task_done_dispatch`` is the one producer of the subagent terminal
    chat frame the executor chip upgrades from, and of the log-channel
    ``task_done`` the client falls back to. Routing is MEMBERSHIP, not
    truthiness (``notification_chat_route``): chat 0 is the Skill Review panel
    — a real destination the old ``if chat_id:`` guard silently dropped, so a
    panel-bound card kept its dispatch-only chip forever — and a negative id is
    A2A traffic that must never enter a human stream.
    """

    @staticmethod
    def _ctx(tmp_path, sent):
        from types import SimpleNamespace

        return SimpleNamespace(
            DRIVE_ROOT=tmp_path, RUNNING={}, PENDING=[], WORKERS={},
            send_with_budget=lambda cid, _text, **kw: sent.append(
                (cid, kw.get("progress_meta") or {})),
            persist_queue_snapshot=lambda **_k: True,
            bridge=SimpleNamespace(push_log=lambda _e: None),
        )

    @staticmethod
    def _result():
        return {
            "status": "completed",
            "executor_route": "codex=gpt-5.6-sol",
            "subagent_envelope": {
                "execution_evidence": {
                    "delegated_runs_started": 1, "delegated_runs_settled": 1,
                    "delegated_runs_succeeded": 1, "delegated_runs_failed": 0,
                },
                "actual_substrate": "harness_used",
            },
        }

    def _dispatch(self, tmp_path, sent, chat_id, monkeypatch, bound=0):
        from supervisor import events as events_mod

        monkeypatch.setattr(
            events_mod, "_bound_project_chat_id", lambda *_a, **_k: bound)
        task = {
            "id": "child-1", "chat_id": chat_id, "parent_task_id": "root-1",
            "root_task_id": "root-1", "delegation_role": "subagent",
            "role": "researcher",
        }
        task_done_event = {
            "type": "task_done", "task_id": "child-1", "status": "completed",
        }
        events_mod._finish_task_done_dispatch(
            {}, self._ctx(tmp_path, sent), task_id="child-1", worker_id=0,
            task=task, final_task_result=self._result(),
            task_done_event=task_done_event,
        )
        return task_done_event

    def _dispatch_text(self, tmp_path, monkeypatch, **terminal_facts):
        """Dispatch one subagent terminal and return its chat ``(text, meta)``."""
        from types import SimpleNamespace

        from supervisor import events as events_mod

        monkeypatch.setattr(
            events_mod, "_bound_project_chat_id", lambda *_a, **_k: 0)
        seen: list = []
        ctx = SimpleNamespace(
            DRIVE_ROOT=tmp_path, RUNNING={}, PENDING=[], WORKERS={},
            send_with_budget=lambda _cid, text, **kw: seen.append(
                (text, kw.get("progress_meta") or {})),
            persist_queue_snapshot=lambda **_k: True,
            bridge=SimpleNamespace(push_log=lambda _e: None),
        )
        events_mod._finish_task_done_dispatch(
            {}, ctx, task_id="child-1", worker_id=0,
            task={"id": "child-1", "chat_id": 7, "parent_task_id": "root-1",
                  "root_task_id": "root-1", "delegation_role": "subagent",
                  "role": "publication-auditor"},
            final_task_result=self._result(),
            task_done_event={"type": "task_done", "task_id": "child-1",
                             "status": "completed", **terminal_facts},
        )
        return seen[0]

    def test_a_clean_completion_still_reads_as_a_clean_completion(
            self, tmp_path, monkeypatch):
        text, meta = self._dispatch_text(tmp_path, monkeypatch)
        assert text == "✅ Subagent child-1 completed (publication-auditor)."
        assert meta["subagent_event"] == "completed" and meta["status"] == "completed"

    @pytest.mark.parametrize("terminal_facts", [
        {"outcome_axes": {"execution": {"status": "degraded"}}},
        {"reason_code": "configured_actor_incomplete"},
        {"reason_code": "configured_actor_unknown"},
    ])
    def test_a_degraded_completion_reads_as_a_warning(
            self, tmp_path, monkeypatch, terminal_facts):
        """The chat line and the card told two stories about ONE terminal.

        A configured actor that never started its leaf ends `completed` on the
        lifecycle axis and `degraded` on the execution axis. The web card computes
        `warn` from those axes; the server-emitted progress text read the lifecycle
        alone and said "✅ … completed", so the owner's first signal was a green
        check over a child that had done nothing. Only icon and verb move —
        `subagent_event` and `status` are what Telegram cards and the web consumers
        key on, and a terminal must not change identity to change its wording.
        """
        text, meta = self._dispatch_text(tmp_path, monkeypatch, **terminal_facts)
        assert text == "⚠️ Subagent child-1 finished with warnings (publication-auditor)."
        assert meta["subagent_event"] == "completed" and meta["status"] == "completed"

    def test_panel_chat_zero_receives_the_terminal_evidence_frame(
            self, tmp_path, monkeypatch):
        sent: list = []
        self._dispatch(tmp_path, sent, chat_id=0, monkeypatch=monkeypatch)
        assert [cid for cid, _m in sent] == [0]
        meta = sent[0][1]
        assert meta["executor_route"] == "codex=gpt-5.6-sol"
        assert meta["execution_evidence"]["delegated_runs_settled"] == 1
        assert meta["actual_substrate"] == "harness_used"

    def test_project_binding_precedes_the_task_chat(self, tmp_path, monkeypatch):
        # The binding's own 0 means "no binding" (never the panel) and falls
        # through; a real binding wins over the task's chat.
        sent: list = []
        self._dispatch(tmp_path, sent, chat_id=0, monkeypatch=monkeypatch,
                       bound=4242)
        assert [cid for cid, _m in sent] == [4242]

    def test_a2a_chat_never_receives_a_human_frame(self, tmp_path, monkeypatch):
        sent: list = []
        self._dispatch(tmp_path, sent, chat_id=-1001, monkeypatch=monkeypatch)
        assert sent == []

    def test_unbound_task_sends_no_frame(self, tmp_path, monkeypatch):
        sent: list = []
        self._dispatch(tmp_path, sent, chat_id=None, monkeypatch=monkeypatch)
        assert sent == []

    def test_log_channel_task_done_carries_the_delegation_truth(
            self, tmp_path, monkeypatch):
        # routeSubagentTerminalToCard upgrades a log-channel-only card, so the
        # pushed task_done must carry the same delegation keys as the chat
        # frame (additive; stamped after the durable events.jsonl append).
        sent: list = []
        evt = self._dispatch(tmp_path, sent, chat_id=None,
                             monkeypatch=monkeypatch)
        assert evt["executor_route"] == "codex=gpt-5.6-sol"
        assert evt["execution_evidence"]["delegated_runs_succeeded"] == 1
        assert evt["actual_substrate"] == "harness_used"
