"""``review_change(subject="result")``: one review machinery for a change and for a task's result.

Every seat of a result wave answers one question — does this task's result do what was asked — over
a subject frozen at the call (what was asked, the reply, what the task produced, the host-attested
verification). The wave is review_change's own: the same panel rules, $0 reuse of the same subject,
the ledger record, and its own paid-cycle count beside change reviews. Task review Off refuses it.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import ouroboros.review_substrate as substrate
from ouroboros import review_ledger
from ouroboros.tools.review_change import ReviewChangeArgumentError, parse_request, run_review_change
from tests import _contributor_packet_shared as shared
from tests._review_session_route_shared import (  # noqa: F401
    FakeGateway, _owned_gateway_uses_each_test_transport, _terminal_detail, fake_route as _fake_route,
)
from tests.review_pool_rosters import pool_roster, pool_seat, set_review_pool
from tests.test_review_change_end_to_end import _foreign_project

fake_route = _fake_route  # the shared offline Claudexor fixture, requested by name below

CRITICAL = '[{"item": "missing_section", "verdict": "FAIL", "severity": "critical", "reason": "The question asked for three findings; the reply gives two."}]'


def _seam(sends, answers=None):
    answers = answers or {}

    def answer(request, slot, actor, *, retry_state, pending_invocation_checkpoint):
        actor.raw_text = answers.get(slot.slot_id, "[]")  # contract A: [] is a clean answer
        return actor

    return shared.golden_physical_seam(sends, answer=answer)


def _reply(ctx, text):
    ctx._delivery_candidate = SimpleNamespace(full_text=text)


def test_every_seat_answers_the_one_result_question_and_the_same_subject_is_reused_free(tmp_path, monkeypatch):
    ctx, _project = _foreign_project(tmp_path, monkeypatch)
    _reply(ctx, "The report is ready: report.md lists the three findings.")
    sends: list[dict] = []
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", _seam(sends))

    first = run_review_change(ctx, subject="result", goal="Check the report answers the question")

    assert (first["state"], first["aggregate"], first["reused"]) == ("settled", "PASS", False), first
    assert first["per_question"] == {"result": "PASS"}, "a result wave asks no change or coupling question"
    assert {tuple(row["parts"]) for row in first["rows"]} == {("result",)}
    record = review_ledger.load_record(ctx.drive_root, first["record_id"])
    assert record["surface"] == "result" and record["subject"]["kind"] == "result"
    assert sorted(send["slot_id"] for send in sends) == ["s1", "t1", "t2"]

    again = run_review_change(ctx, subject="result", goal="Check the report answers the question")
    assert again["reused"] is True and again["record_id"] == first["record_id"] and len(sends) == 3

    _reply(ctx, "The report is ready: report.md lists four findings.")
    changed = run_review_change(ctx, subject="result", goal="Check the report answers the question")
    assert changed["reused"] is False and changed["record_id"] != first["record_id"] and len(sends) == 6


def test_a_critical_finding_fails_the_result(tmp_path, monkeypatch):
    ctx, _project = _foreign_project(tmp_path, monkeypatch)
    sends: list[dict] = []
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", _seam(sends, {"t1": CRITICAL}))

    result = run_review_change(ctx, subject="result", answer="Two findings are in report.md.")

    assert result["aggregate"] == "FAIL" and result["per_question"] == {"result": "FAIL"}
    [finding] = result["findings"]["critical_findings"]
    assert finding["seat_id"] == "t1" and finding["part"] == "result" and finding["item"] == "missing_section"


def test_task_review_off_refuses_a_result_review_before_any_seat(tmp_path, monkeypatch):
    ctx, project = _foreign_project(tmp_path, monkeypatch)
    _reply(ctx, "Done.")
    sends: list[dict] = []
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", _seam(sends))
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", "off")

    with pytest.raises(ReviewChangeArgumentError, match="result review is off"):
        run_review_change(ctx, subject="result")
    assert sends == []

    (project / "app.py").write_text("VALUE = 2  # staged\n", encoding="utf-8")
    shared.git(project, "add", "app.py")
    change = run_review_change(ctx, subject="index", goal="Bump", scope="app.py")
    assert change["aggregate"] != "NOT_DISPATCHED" and sends, "change review stays available in Off"


def test_result_reviews_keep_their_own_paid_cycle_count(tmp_path, monkeypatch):
    """Two kinds of review, two counts: a change review that used the per-task ceiling never refuses an
    owed result review, and result reviews meet their own ceiling."""
    ctx, project = _foreign_project(tmp_path, monkeypatch)
    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "1")
    sends: list[dict] = []
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", _seam(sends))
    (project / "app.py").write_text("VALUE = 2  # staged\n", encoding="utf-8")
    shared.git(project, "add", "app.py")
    assert run_review_change(ctx, subject="index", goal="Bump", scope="app.py")["aggregate"] != "NOT_DISPATCHED"

    first = run_review_change(ctx, subject="result", answer="Bumped VALUE to 2.")
    assert first["aggregate"] == "PASS" and first["dispatch_refusal"] is None

    second = run_review_change(ctx, subject="result", answer="Bumped VALUE to 2 and documented it.")
    assert second["aggregate"] == "NOT_DISPATCHED"
    assert second["dispatch_refusal"]["kind"] == "review_cycles_exhausted"


def test_result_reviews_of_one_root_share_one_allowance_wherever_their_material_lives(tmp_path, monkeypatch):
    """Two children of one root review their results from different folders (each task's own retained
    subject): they still spend the ROOT's one result allowance, never one allowance per folder."""
    ctx, _project = _foreign_project(tmp_path, monkeypatch)
    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "1")
    sends: list[dict] = []
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", _seam(sends))
    root_id = str(getattr(ctx, "task_id", "") or "root-task")
    for child in ("child-a", "child-b"):
        ctx.task_id = child
        ctx.task_metadata = {**dict(getattr(ctx, "task_metadata", None) or {}), "root_task_id": root_id,
                             "parent_task_id": root_id}
        outcome = run_review_change(ctx, subject="result", answer=f"{child} wrote its summary.")
        if child == "child-a":
            assert outcome["aggregate"] == "PASS", outcome
        else:
            assert outcome["aggregate"] == "NOT_DISPATCHED", "the root's one result allowance is spent"
            assert outcome["dispatch_refusal"]["kind"] == "review_cycles_exhausted"


@pytest.mark.parametrize("args, problem", [
    ({"subject": "result", "base": "HEAD~1"}, "root, workspace_root, base and head do not apply"),
    ({"subject": "result", "workspace_root": "/tmp"}, "root, workspace_root, base and head do not apply"),
    ({"subject": "result", "coupling_only": ["x"]}, "coupling_only belongs to a change review"),
    ({"subject": "index", "answer": "Done."}, "answer belongs to subject=result"),
    ({"subject": "index", "surface": "result"}, "subject=result goes with surface=result"),
])
def test_a_result_call_that_cannot_mean_anything_is_refused_before_payment(args, problem):
    with pytest.raises(ReviewChangeArgumentError, match=problem.replace("(", r"\(").replace(")", r"\)")):
        parse_request(args)


def test_a_result_review_binds_no_workspace_target():
    """A result review reads the task's own result, so it takes no vcs binding on a workspace (a Presence
    behavior that grants review_change without a vcs root may still ask for one); change subjects stay bound."""
    from ouroboros.tools.tool_resolution import _target_binding_operation

    assert _target_binding_operation("review_change", {"subject": "result"}) is None
    assert _target_binding_operation("review_change", {"subject": "index"}) == "vcs"
    assert _target_binding_operation("review_change", {}) == "vcs"


def test_the_result_subject_implies_its_surface():
    request = parse_request({"subject": "result", "answer": "Done."})
    assert (request.surface, request.answer) == ("result", "Done.")


# --- The delivery step: binding and the Required floor ------------------------------------------

def _floor_ctx(tmp_path, monkeypatch, *, mode="required", enforcement="advisory", root=True):
    ctx, _project = _foreign_project(tmp_path, monkeypatch)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", mode)
    monkeypatch.setenv("OUROBOROS_REVIEW_ENFORCEMENT", enforcement)
    if not root:
        ctx.parent_task_id = "parent-task"
        ctx.task_metadata = {"parent_task_id": "parent-task", "root_task_id": "parent-task"}
    return ctx


def _produced_answer_protocol(monkeypatch):
    """A benchmark-style result: the adapter's answer protocol makes the answer line produced work."""
    monkeypatch.setattr("ouroboros.contracts.task_contract.answer_protocol_active", lambda ctx: True)


def test_required_reviews_a_root_result_with_produced_work_once_and_shows_its_findings_first(tmp_path, monkeypatch):
    from ouroboros.tools.review_result import result_review_at_delivery

    ctx = _floor_ctx(tmp_path, monkeypatch)
    _produced_answer_protocol(monkeypatch)
    sends: list[dict] = []
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", _seam(sends, {"t1": CRITICAL}))
    shown, trace = [], {"tool_calls": []}
    answer = "FINAL ANSWER: 2"

    assert result_review_at_delivery(ctx, answer, trace, shown.append) is True, "Advisory: findings first"
    assert len(sends) == 3 and "[Result review]" in shown[0] and "FAIL" in shown[0] and "You decide" in shown[0]
    [run] = trace["review_runs"]
    assert (run["surface"], run["aggregate_signal"], run["authority"]) == ("result", "FAIL", "host_root")

    assert result_review_at_delivery(ctx, answer, trace, shown.append) is False, "the mind decided: deliver"
    assert len(sends) == 3 and len(shown) == 1, "the same result is never reviewed or shown twice"


def test_blocking_holds_the_same_failed_result_until_it_changes_and_passes(tmp_path, monkeypatch):
    from ouroboros.tools.review_result import result_review_at_delivery

    ctx = _floor_ctx(tmp_path, monkeypatch, enforcement="blocking")
    _produced_answer_protocol(monkeypatch)
    sends: list[dict] = []
    answers = {"t1": CRITICAL}
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", _seam(sends, answers))
    shown, trace = [], {"tool_calls": []}

    assert result_review_at_delivery(ctx, "FINAL ANSWER: 2", trace, shown.append) is True
    assert result_review_at_delivery(ctx, "FINAL ANSWER: 2", trace, shown.append) is True, "Blocking holds again"
    assert "not delivered until a review passes it" in shown[-1] and len(sends) == 3
    answers.clear()  # the revised result satisfies the reviewers
    assert result_review_at_delivery(ctx, "FINAL ANSWER: 3", trace, shown.append) is False
    assert len(sends) == 6
    current = [run for run in trace["review_runs"] if not run.get("superseded_by_revision")]
    assert [run["aggregate_signal"] for run in current] == ["PASS"]


def test_a_required_root_result_is_reviewed_through_a_real_session_seat(tmp_path, monkeypatch, fake_route):
    """The Required floor on the real Claudexor session route (offline gateway): the seat is asked for the
    findings array, never the change/coupling object; its critical finding holds a Blocking delivery and its
    clean answer on the revised result delivers it."""
    import json

    from ouroboros.tools.review_result import result_review_at_delivery

    ctx = _floor_ctx(tmp_path, monkeypatch, enforcement="blocking")
    _produced_answer_protocol(monkeypatch)
    set_review_pool(monkeypatch, pool_roster(pool_seat(
        "independent", "fake-review=fake-small", kind="agent_session", access="readonly")))
    start = FakeGateway.start_run
    monkeypatch.setattr(FakeGateway, "start_run", lambda self, request, **kwargs: {
        **start(self, request, **kwargs), "runId": f"run-{len(self.start_requests)}-{len(FakeGateway.instances)}"})
    critical = {"findings": [{"item": "missing_section", "verdict": "FAIL", "severity": "critical",
                              "reason": "The question asked for three findings; the reply gives two."}]}
    monkeypatch.setattr(FakeGateway, "detail", _terminal_detail(json.dumps(critical), conformance="passed"))
    shown, trace = [], {"tool_calls": []}

    assert result_review_at_delivery(ctx, "FINAL ANSWER: 2", trace, shown.append) is True
    assert "three findings" in shown[0], shown
    monkeypatch.setattr(FakeGateway, "detail", _terminal_detail('{"findings": []}', conformance="passed"))
    assert result_review_at_delivery(ctx, "FINAL ANSWER: 3", trace, shown.append) is False
    starts = [body for gateway in FakeGateway.instances for body in gateway.start_requests]
    assert len(starts) == 2 and all(body["outputSchema"]["required"] == ["findings"] for body in starts), starts
    current = [run for run in trace["review_runs"] if not run.get("superseded_by_revision")]
    assert [run["aggregate_signal"] for run in current] == ["PASS"]


def test_nothing_is_forced_for_a_child_auto_off_or_a_plain_text_answer(tmp_path, monkeypatch):
    from ouroboros.tools.review_result import result_review_at_delivery

    sends: list[dict] = []
    for kwargs, protocol in (({"root": False}, True), ({"mode": "auto"}, True), ({"mode": "off"}, True), ({}, False)):
        ctx = _floor_ctx(tmp_path / str(len(sends)) / repr(sorted(kwargs.items())).replace("/", ""), monkeypatch, **kwargs)
        if protocol:
            _produced_answer_protocol(monkeypatch)
        else:
            monkeypatch.setattr("ouroboros.contracts.task_contract.answer_protocol_active", lambda ctx: False)
        monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", _seam(sends))
        trace = {"tool_calls": []}
        assert result_review_at_delivery(ctx, "FINAL ANSWER: 2", trace, lambda text: None) is False, kwargs
        assert not trace.get("review_runs")
    assert sends == []


def test_required_reviews_presence_results_without_holding_them_and_ignores_a_declared_expected_output(
        tmp_path, monkeypatch):
    from ouroboros.tools.review_result import required_material, result_review_at_delivery

    sends: list[dict] = []
    ctx = _floor_ctx(tmp_path / "presence", monkeypatch)
    _produced_answer_protocol(monkeypatch)
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", _seam(sends, {"t1": CRITICAL}))
    ctx._presence_completion = {"outcome": "spoken"}
    trace, shown = {"tool_calls": []}, []
    assert result_review_at_delivery(ctx, "FINAL ANSWER: 2", trace, shown.append) is False, \
        "Presence speech is never held for a review"
    assert shown == [] and len(sends) == 3, "its owed review still runs, with nobody waiting for it"
    [run] = trace["review_runs"]
    presence = review_ledger.load_record(ctx.drive_root, run["record_id"])
    assert presence["panel"]["chosen_by"] == "host" and presence["panel"]["composition"] == "full_pool"
    assert set(presence["panel"]["assigned"]) == {"t1", "t2", "s1"}
    assert result_review_at_delivery(ctx, "FINAL ANSWER: 2", trace, shown.append) is False and len(sends) == 3
    sends.clear()

    # The expected result a task was given is intent, never production (owner, 11 Oct, #1731).
    ctx = _floor_ctx(tmp_path / "declared", monkeypatch)
    monkeypatch.setattr("ouroboros.contracts.task_contract.answer_protocol_active", lambda ctx: False)
    ctx.task_metadata = {**dict(getattr(ctx, "task_metadata", None) or {}),
                         "task_contract": {"expected_output": "A report with three findings",
                                           "acceptance_criteria": ["three findings"]}}
    answer = "The report lists three findings."
    assert required_material(ctx, {"tool_calls": []}, answer) is False
    assert result_review_at_delivery(ctx, answer, {"tool_calls": []}, lambda text: None) is False
    assert sends == []


def test_a_mind_reviewed_result_is_bound_at_delivery_in_auto_without_a_second_wave(tmp_path, monkeypatch):
    from ouroboros.outcomes import _objective_axis, _review_axis
    from ouroboros.tools.review_result import result_review_at_delivery

    ctx = _floor_ctx(tmp_path, monkeypatch, mode="auto")
    sends: list[dict] = []
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", _seam(sends))
    answer = "The summary is in summary.md."
    assert run_review_change(ctx, subject="result", answer=answer)["aggregate"] == "PASS"
    trace = {"tool_calls": []}

    assert result_review_at_delivery(ctx, answer, trace, lambda text: None) is False
    assert len(sends) == 3, "binding never buys a second wave"
    review = _review_axis(trace)
    assert review["status"] == "pass" and review["source"] == "result_review"
    assert _objective_axis(review) == {"status": "pass", "source": "result_review", "review_status": "pass"}


def test_an_advisory_finish_over_findings_is_the_authors_own_decision(tmp_path, monkeypatch):
    from ouroboros.outcomes import _objective_axis, _review_axis
    from ouroboros.tools.review_result import result_review_at_delivery

    ctx = _floor_ctx(tmp_path, monkeypatch)
    _produced_answer_protocol(monkeypatch)
    sends: list[dict] = []
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", _seam(sends, {"t1": CRITICAL}))
    trace = {"tool_calls": []}

    assert result_review_at_delivery(ctx, "FINAL ANSWER: 2", trace, lambda text: None) is True
    [run] = trace["review_runs"]
    assert not run.get("informed_finish"), "findings shown, nothing decided yet"
    assert result_review_at_delivery(ctx, "FINAL ANSWER: 2", trace, lambda text: None) is False
    assert run["informed_finish"] is True and run["enforcement"] == "advisory"
    review = _review_axis(trace)
    assert (review["status"], review["source"], review.get("author_finish")) == ("fail", "result_review", True)
    assert _objective_axis(review)["source"] == "author_acceptance" and _objective_axis(review)["status"] == "pass"


def test_blocking_with_no_review_left_to_buy_reads_blocked(tmp_path, monkeypatch):
    from ouroboros.outcomes import OUTCOME_TIER_BLOCKED, _objective_axis, _review_axis
    from ouroboros.tools.review_result import result_review_at_delivery

    ctx = _floor_ctx(tmp_path, monkeypatch, enforcement="blocking")
    _produced_answer_protocol(monkeypatch)
    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "1")
    sends: list[dict] = []
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", _seam(sends, {"t1": CRITICAL}))
    trace, shown = {"tool_calls": []}, []

    assert result_review_at_delivery(ctx, "FINAL ANSWER: 2", trace, shown.append) is True  # FAIL holds
    assert result_review_at_delivery(ctx, "FINAL ANSWER: 3", trace, shown.append) is True  # refused at $0
    assert "REVIEW_CYCLES_EXHAUSTED" in shown[-1] and "escalate to the owner" in shown[-1]
    assert result_review_at_delivery(ctx, "FINAL ANSWER: 3", trace, shown.append) is True, \
        "Blocking keeps holding an unreviewed result until its human raises the ceiling or the task stops"
    assert "raises the review ceiling" in shown[-1] and len(sends) == 3
    review = _review_axis(trace)
    assert review["blocked_reason"] == "review_cycles_exhausted" and not review.get("author_finish")
    objective = _objective_axis(review)
    assert (objective["status"], objective["outcome_tier"], objective["reason"]) == (
        "fail", OUTCOME_TIER_BLOCKED, "review_cycles_exhausted")


def test_an_advisory_delivery_after_a_shown_refusal_is_the_authors_own_decision(tmp_path, monkeypatch):
    from ouroboros.outcomes import _objective_axis, _review_axis
    from ouroboros.tools.review_result import result_review_at_delivery

    ctx = _floor_ctx(tmp_path, monkeypatch)
    _produced_answer_protocol(monkeypatch)
    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "1")
    sends: list[dict] = []
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", _seam(sends, {"t1": CRITICAL}))
    trace, shown = {"tool_calls": []}, []

    assert result_review_at_delivery(ctx, "FINAL ANSWER: 2", trace, shown.append) is True  # findings first
    assert result_review_at_delivery(ctx, "FINAL ANSWER: 3", trace, shown.append) is True  # refused, shown once
    assert "You decide" in shown[-1]
    assert result_review_at_delivery(ctx, "FINAL ANSWER: 3", trace, shown.append) is False, "the mind decided"
    current = [run for run in trace["review_runs"] if not run.get("superseded_by_revision")]
    assert [run.get("informed_finish") for run in current] == [True] and len(sends) == 3
    assert _objective_axis(_review_axis(trace))["source"] == "author_acceptance"


@pytest.mark.parametrize("owner_message", [False, True])
def test_delivery_collects_the_running_review_the_mind_started_without_buying_another(tmp_path, monkeypatch,
                                                                                      owner_message):
    """With ``owner_message`` an owner message arrives while the review runs (the usual reason its author left
    it): the running review of the same answer is still collected at $0 under its own subject, and since it
    judged the earlier request the host asks once whether the new words change it; delivering again keeps it."""
    from ouroboros.tools.review_result import result_review_at_delivery
    from tests.test_review_change_end_to_end import _pending_then_answering_seat

    for mode in ("auto", "required"):
        ctx = _floor_ctx(tmp_path / mode, monkeypatch, mode=mode)
        _produced_answer_protocol(monkeypatch)
        sends: list[dict] = []
        answer, starts = _pending_then_answering_seat("t2", f"invocation-t2-{mode}")
        monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", shared.golden_physical_seam(sends, answer=answer))
        first = run_review_change(ctx, subject="result", answer="FINAL ANSWER: 2", goal="Check the arithmetic",
                                  author_questions=["Is 2 the answer?"])
        assert first["state"] == "pending" and len(sends) == 3, (mode, first)
        if owner_message:
            ctx._owner_directives = [*(getattr(ctx, "_owner_directives", None) or []),
                                     {"source": "owner_mailbox", "content": "How far along are you?", "msg_id": "m-2"}]

        trace, shown = {"tool_calls": []}, []
        if owner_message:
            assert result_review_at_delivery(ctx, "FINAL ANSWER: 2", trace, shown.append) is True, mode
            assert "How far along are you?" in shown[0] and "deliver again" in shown[0], shown
        assert result_review_at_delivery(ctx, "FINAL ANSWER: 2", trace, shown.append) is False, mode
        assert len(sends) == 4 and sends[3]["slot_id"] == "t2" and sends[3]["reconcile_only"] is True, \
            "the delivery rejoined the running wave's own seat; no second wave was bought"
        [run] = trace["review_runs"]
        assert (run["record_id"], run["aggregate_signal"]) == (first["record_id"], "PASS"), mode
        record = review_ledger.load_record(ctx.drive_root, first["record_id"])
        assert record["brief"]["goal"] == "Check the arithmetic", "the collected record keeps the mind's brief"


def test_a_delivered_answer_never_keeps_an_earlier_answers_run_current(tmp_path, monkeypatch):
    from ouroboros.tools.review_result import result_review_at_delivery

    ctx = _floor_ctx(tmp_path, monkeypatch, mode="auto")
    sends: list[dict] = []
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", _seam(sends))
    assert run_review_change(ctx, subject="result", answer="Report A")["aggregate"] == "PASS"
    trace = {"tool_calls": []}
    assert result_review_at_delivery(ctx, "Report A", trace, lambda text: None) is False
    assert result_review_at_delivery(ctx, "Report B", trace, lambda text: None) is False
    [run] = trace["review_runs"]
    assert run["superseded_by_revision"], "A's review speaks only for A"


@pytest.mark.parametrize("enforcement, delivered", [("advisory", True), ("blocking", False)])
def test_pending_review_finish_delivers_over_a_running_review_unless_blocking(tmp_path, monkeypatch, enforcement, delivered):
    import ouroboros.tools.review_change as review_change_mod
    from ouroboros.tools.review_result import pending_review_choice_offered, result_review_at_delivery

    ctx = _floor_ctx(tmp_path, monkeypatch, enforcement=enforcement)
    _produced_answer_protocol(monkeypatch)
    monkeypatch.setattr(review_change_mod, "run_review_change",
                        lambda ctx, **kwargs: {"state": "pending", "record_id": "rl-running"})
    shown = []
    assert result_review_at_delivery(ctx, "FINAL ANSWER: 2", {"tool_calls": []}, shown.append) is True
    assert pending_review_choice_offered(ctx) is delivered
    assert ("pending_review=finish" in shown[0]) is delivered
    ctx._acceptance_pending_review_choice = "finish"
    assert (result_review_at_delivery(ctx, "FINAL ANSWER: 2", {"tool_calls": []}, shown.append) is False) is delivered


@pytest.mark.parametrize("runtime, enforcement, finish_delivers", [
    ("pro", "advisory", True), ("cyber_pro", "blocking", True), ("pro", "blocking", False)])
def test_a_running_review_the_mind_started_is_waited_for_unless_it_chose_to_finish(
        tmp_path, monkeypatch, runtime, enforcement, finish_delivers):
    import pathlib

    import ouroboros.tools.review_result as review_result
    from ouroboros.body_candidate import serving_repo_dir_for

    ctx = _floor_ctx(tmp_path, monkeypatch, mode="auto", enforcement=enforcement)
    monkeypatch.setenv("OUROBOROS_RUNTIME_MODE", runtime)
    answer = "The summary is in summary.md."
    subject = review_result.result_subject(ctx, SimpleNamespace(answer=""),
                                           governance_root=pathlib.Path(serving_repo_dir_for(ctx)), answer=answer)
    rows = [{"record_id": "rl-mind", "state": "pending", "subject": {"diff_sha": subject.diff_sha}}]
    monkeypatch.setattr(review_result, "_result_rows", lambda ctx: rows)
    collected = []
    monkeypatch.setattr(review_result, "_collect_running",
                        lambda ctx, content, record_id, subject: collected.append(record_id) or {
                            "state": "pending", "record_id": record_id})
    shown = []

    assert review_result.result_review_at_delivery(ctx, answer, {"tool_calls": []}, shown.append) is True, \
        "the default is to wait for the review the mind started (in Cyber Pro too)"
    assert collected == ["rl-mind"] and "still running" in shown[0]
    ctx._acceptance_pending_review_choice = "finish"
    assert (review_result.result_review_at_delivery(ctx, answer, {"tool_calls": []}, shown.append) is False) \
        is finish_delivers, "an explicit finish is honoured unless Blocking outside Cyber Pro"
    rows[0]["subject"] = {"diff_sha": "another-answer"}
    ctx._acceptance_pending_review_choice = "wait"
    assert review_result.result_review_at_delivery(ctx, answer, {"tool_calls": []}, shown.append) is False, \
        "a review of an earlier answer never holds this one"


# --- Transitions found by the second review round -----------------------------------------------

def test_a_raised_ceiling_lets_the_blocked_host_review_run(tmp_path, monkeypatch):
    """Blocking with the allowance spent holds the result, but asks again on every delivery: a refusal costs
    nothing, so once the human raises the ceiling the same delivery buys the review and a PASS delivers."""
    from ouroboros.tools import review_result as rr

    ctx = _floor_ctx(tmp_path, monkeypatch, enforcement="blocking")
    _produced_answer_protocol(monkeypatch)
    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "1")
    sends: list[dict] = []
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", _seam(sends, {"t1": CRITICAL}))
    trace, shown = {"tool_calls": []}, []
    assert rr.result_review_at_delivery(ctx, "FINAL ANSWER: 2", trace, shown.append)
    assert rr.result_review_at_delivery(ctx, "FINAL ANSWER: 3", trace, shown.append), "refused: held"
    records = len(review_ledger.recent_records(review_ledger.ledger_root(ctx), ctx.task_id, limit=50, surface="result"))
    assert rr.result_review_at_delivery(ctx, "FINAL ANSWER: 3", trace, shown.append), "still refused: held"
    assert rr.result_review_at_delivery(ctx, "FINAL ANSWER: 3", trace, shown.append), "still refused: held"
    assert "raises the review ceiling" in shown[-1]
    assert len(review_ledger.recent_records(review_ledger.ledger_root(ctx), ctx.task_id, limit=50,
                                           surface="result")) == records, "an unchanged ceiling writes no new refusal"
    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "2")
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", _seam(sends))
    assert rr.result_review_at_delivery(ctx, "FINAL ANSWER: 3", trace, shown.append) is False, shown[-1]


def test_a_manual_review_after_a_raised_ceiling_also_delivers(tmp_path, monkeypatch):
    from ouroboros.tools import review_result as rr

    ctx = _floor_ctx(tmp_path, monkeypatch, enforcement="blocking")
    _produced_answer_protocol(monkeypatch)
    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "1")
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", _seam([], {"t1": CRITICAL}))
    trace = {"tool_calls": []}
    assert rr.result_review_at_delivery(ctx, "FINAL ANSWER: 2", trace, lambda _: None)
    assert rr.result_review_at_delivery(ctx, "FINAL ANSWER: 3", trace, lambda _: None)
    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "2")
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", _seam([]))
    assert run_review_change(ctx, subject="result", answer="FINAL ANSWER: 3")["aggregate"] == "PASS"
    assert not rr.result_review_at_delivery(ctx, "FINAL ANSWER: 3", trace, lambda _: None)


@pytest.mark.parametrize("change", ["owner_requirement", "late_artifact"])
def test_collecting_a_running_review_keeps_the_subject_its_seats_saw(tmp_path, monkeypatch, change):
    """The delivery step collects the record it selected with the subject it froze: material that changes
    between the lookup and the collection never rewrites that record's identity or inherits its verdict."""
    from ouroboros.artifacts import store_task_artifact_bytes
    from ouroboros.tools import review_result as rr
    from tests.test_review_change_end_to_end import _pending_then_answering_seat

    ctx = _floor_ctx(tmp_path, monkeypatch)
    _produced_answer_protocol(monkeypatch)
    answer, _starts = _pending_then_answering_seat("t2", f"identity-{change}")
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", shared.golden_physical_seam([], answer=answer))
    first = run_review_change(ctx, subject="result", answer="FINAL ANSWER: 2", goal="Check the arithmetic")
    assert first["state"] == "pending"
    identity = ("kind", "root", "diff_sha", "tree_sha")  # the subject itself; its retention note settles later
    original = {k: review_ledger.load_record(review_ledger.ledger_root(ctx), first["record_id"])["subject"][k]
                for k in identity}
    collect, ask = rr._collect_running, rr._ask

    def change_then_collect(ctx, content, record_id, subject):
        if change == "late_artifact":
            store_task_artifact_bytes(ctx.drive_root, ctx.task_id, "late-result.txt", b"additional produced output")
        else:
            monkeypatch.setattr(rr, "_ask", lambda c: ask(c) + "\nNew owner requirement: answer must be 3.")
        return collect(ctx, content, record_id, subject)

    monkeypatch.setattr(rr, "_collect_running", change_then_collect)
    rr.result_review_at_delivery(ctx, "FINAL ANSWER: 2", {"tool_calls": []}, lambda _: None)
    after = review_ledger.load_record(review_ledger.ledger_root(ctx), first["record_id"])["subject"]
    assert {k: after[k] for k in identity} == original


def test_a_review_that_settled_between_the_lookup_and_the_collection_still_holds_its_fail(tmp_path, monkeypatch):
    """Blocking: the round settles FAIL after the delivery step saw it pending, so the collection finds no
    open operation; the step reads the record again and holds over its findings instead of delivering."""
    from ouroboros.tools import review_result as rr
    from tests.test_review_change_end_to_end import _pending_then_answering_seat

    ctx = _floor_ctx(tmp_path, monkeypatch, enforcement="blocking")
    _produced_answer_protocol(monkeypatch)
    sends: list[dict] = []
    pending, _starts = _pending_then_answering_seat("t2", "settlement-race")

    def answer(request, slot, actor, **kwargs):
        actor.raw_text = CRITICAL if slot.slot_id == "t1" else "[]"
        return pending(request, slot, actor, **kwargs)

    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", shared.golden_physical_seam(sends, answer=answer))
    args = dict(subject="result", answer="FINAL ANSWER: 2", goal="Check the arithmetic")
    assert run_review_change(ctx, **args)["state"] == "pending"
    collect, settled_at = rr._collect_running, []

    def settle_then_collect(ctx, content, record_id, subject):
        assert run_review_change(ctx, **args)["aggregate"] == "FAIL"  # the paid round settles first
        settled_at.append(len(sends))
        return collect(ctx, content, record_id, subject)

    monkeypatch.setattr(rr, "_collect_running", settle_then_collect)
    shown: list[str] = []
    assert rr.result_review_at_delivery(ctx, "FINAL ANSWER: 2", {"tool_calls": []}, shown.append) is True
    assert "FAIL" in shown[0] and len(sends) == settled_at[0], "the settled FAIL is shown; nothing new is bought"


@pytest.mark.parametrize("foreign", [False, True])
def test_an_unknown_or_foreign_round_is_never_collected_or_bought(tmp_path, monkeypatch, foreign):
    import copy

    from tests.test_review_change_end_to_end import _pending_then_answering_seat

    ctx = _floor_ctx(tmp_path, monkeypatch)
    sends: list[dict] = []
    answer, _starts = _pending_then_answering_seat("t2", "foreign-round")
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", shared.golden_physical_seam(sends, answer=answer))
    first = run_review_change(ctx, subject="result", answer="Answer A")
    key = review_ledger.load_record(review_ledger.ledger_root(ctx), first["record_id"])["fingerprints"]["retry_key"]
    if foreign:
        ctx = copy.copy(ctx)
        ctx.task_id = "other-task"
    else:
        key = "no-such-round"
    before = len(sends)
    with pytest.raises(ReviewChangeArgumentError):
        run_review_change(ctx, subject="result", answer="Answer A", _rejoin_retry_key=key)
    assert len(sends) == before


def test_returning_to_an_exactly_reviewed_answer_makes_its_review_current_again(tmp_path, monkeypatch):
    """A -> B -> A with unchanged material: A's own settled PASS is bound again and counts (it was only
    superseded while B was the answer); it is never revived for changed material."""
    from ouroboros.outcomes import _review_axis
    from ouroboros.tools import review_result as rr

    ctx = _floor_ctx(tmp_path, monkeypatch, mode="auto")
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", _seam([]))
    run_review_change(ctx, subject="result", answer="Reviewed answer A")
    trace = {"tool_calls": [], "delivery_candidate": {"acceptance_binding": {"authoritative": False}}}
    assert not rr.result_review_at_delivery(ctx, "Reviewed answer A", trace, lambda _: None)
    rr.supersede_result_runs(trace, "candidate-2")  # a new candidate (B) replaced it
    assert not rr.result_review_at_delivery(ctx, "Different answer B", trace, lambda _: None)
    assert not rr.result_review_at_delivery(ctx, "Reviewed answer A", trace, lambda _: None)
    assert not trace["review_runs"][0].get("superseded_by_revision")
    assert _review_axis(trace)["status"] == "pass"


def test_the_same_unchanged_result_call_collects_its_running_review_after_an_owner_message(tmp_path, monkeypatch):
    """The settlement notice promises that re-issuing the same unchanged call collects the left review at $0: an
    owner message in between does not make the running review of the same answer unreachable. Once collected, the
    same call reviews the current request, with the new words in its subject."""
    from ouroboros.tools import review_result as rr
    from tests.test_review_change_end_to_end import _pending_then_answering_seat

    ctx = _floor_ctx(tmp_path, monkeypatch, mode="auto")
    sends: list[dict] = []
    answer, _starts = _pending_then_answering_seat("t2", "invocation-same-call")
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", shared.golden_physical_seam(sends, answer=answer))
    args = dict(subject="result", answer="The report lists three findings.", goal="Check the report")
    first = run_review_change(ctx, **args)
    assert first["state"] == "pending" and len(sends) == 3
    ctx._owner_directives = [*(getattr(ctx, "_owner_directives", None) or []),
                             {"source": "owner_mailbox", "content": "Also mention the backup window.", "msg_id": "m-2"}]
    again = run_review_change(ctx, **args)
    assert again["record_id"] == first["record_id"] and again["state"] == "settled" and len(sends) == 4
    # Once collected, the same call reviews the current request: the new words are in its subject.
    current = run_review_change(ctx, **args)
    assert current["record_id"] != first["record_id"] and len(sends) == 7
    record = review_ledger.load_record(review_ledger.ledger_root(ctx), current["record_id"])
    assert record["subject"]["tree_sha"] == review_ledger.load_record(
        review_ledger.ledger_root(ctx), first["record_id"])["subject"]["tree_sha"], "the same answer"
    assert "Also mention the backup window." in rr.result_subject(
        ctx, SimpleNamespace(answer=args["answer"]), governance_root=ctx.repo_dir).diff_text


def test_collecting_a_rebuttal_bearing_review_keeps_its_provenance_and_its_reuse(tmp_path, monkeypatch):
    """The host collects the mind's running review under that dispatch's own provenance (its brief and its
    fingerprints, the reuse key over the rebuttal its seats read), so the mind's identical call afterwards is
    reused at $0, never bought again."""
    from ouroboros.tools.review_result import result_review_at_delivery
    from tests.test_review_change_end_to_end import _pending_then_answering_seat

    ctx = _floor_ctx(tmp_path, monkeypatch, mode="auto")
    sends: list[dict] = []
    answer, _starts = _pending_then_answering_seat("t2", "invocation-rebuttal")
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", shared.golden_physical_seam(sends, answer=answer))
    args = dict(subject="result", answer="The sum is 12.", goal="Check the sum",
                review_rebuttal="I re-added the column by hand; it is 12, not 11.")
    first = run_review_change(ctx, **args)
    assert first["state"] == "pending"
    root = review_ledger.ledger_root(ctx)
    pending = review_ledger.load_record(root, first["record_id"])
    assert result_review_at_delivery(ctx, "The sum is 12.", {"tool_calls": []}, lambda _: None) is False
    settled = review_ledger.load_record(root, first["record_id"])
    assert settled["state"] == "settled" and settled["verdict"]["aggregate"] == "PASS"
    assert settled["fingerprints"] == pending["fingerprints"] and settled["brief"] == pending["brief"]
    paid = len(sends)
    again = run_review_change(ctx, **args)
    assert again["reused"] is True and again["record_id"] == first["record_id"] and len(sends) == paid


def test_a_running_review_whose_operation_was_lost_never_lets_a_required_result_pass_unreviewed(tmp_path, monkeypatch):
    """A restart can close a tokenless seat's attempt while its ledger row still reads pending: the collection
    finds no open operation. Under Required the floor then buys its own review (the lost attempt produced no
    verdict) instead of delivering the result unreviewed; Blocking holds over that review's findings."""
    from ouroboros.tools import review_change as rc
    from ouroboros.tools import review_result as rr
    from tests.test_review_change_end_to_end import _pending_then_answering_seat

    ctx = _floor_ctx(tmp_path, monkeypatch, enforcement="blocking")
    _produced_answer_protocol(monkeypatch)
    sends: list[dict] = []
    pending, _starts = _pending_then_answering_seat("t2", "lost-operation")

    def answer(request, slot, actor, **kwargs):
        actor.raw_text = CRITICAL if slot.slot_id == "t1" else "[]"
        return pending(request, slot, actor, **kwargs)

    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", shared.golden_physical_seam(sends, answer=answer))
    first = run_review_change(ctx, subject="result", answer="FINAL ANSWER: 2")
    assert first["state"] == "pending"
    lost = review_ledger.load_record(review_ledger.ledger_root(ctx), first["record_id"])["fingerprints"]["retry_key"]
    lookup = rc.pending_round_attempt
    # Restart reconciliation closed that round's attempt; its ledger row still reads pending.
    monkeypatch.setattr(rc, "pending_round_attempt",
                        lambda ctx, **kw: None if kw.get("retry_key") == lost else lookup(ctx, **kw))
    shown: list[str] = []
    trace = {"tool_calls": []}
    before = len(sends)
    assert rr.result_review_at_delivery(ctx, "FINAL ANSWER: 2", trace, shown.append) is True
    # Whatever the floor's own review settles to (the lost seat may leave it without quorum), the result is
    # never delivered unreviewed: a run is bound and Blocking holds over it.
    assert [run["aggregate_signal"] for run in trace["review_runs"]] in (["FAIL"], ["DEGRADED"])
    assert "not delivered until a review passes it" in shown[0]
    assert len(sends) <= before + 3, "at most the floor's own one wave (recorded seats replay at $0)"


def test_a_late_settlement_just_before_the_rejoin_write_keeps_the_rounds_provenance(tmp_path, monkeypatch):
    """The left-review collector may settle the record an instant before the host's collection writes it: the
    rejoin still keeps the round's brief and fingerprints, so the mind's identical call is reused at $0."""
    from ouroboros.review_pause import _settle_left_record
    from ouroboros.tools import review_change as rc
    from ouroboros.tools.review_result import result_review_at_delivery
    from tests.test_review_change_end_to_end import _pending_then_answering_seat

    ctx = _floor_ctx(tmp_path, monkeypatch, mode="auto")
    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "5")
    sends: list[dict] = []
    answer, _starts = _pending_then_answering_seat("t2", "write-race")
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", shared.golden_physical_seam(sends, answer=answer))
    args = dict(subject="result", answer="The sum is 12.", goal="Check the sum", review_rebuttal="It is 12, not 11.")
    first = rc.run_review_change(ctx, **args)
    assert first["state"] == "pending"
    root = review_ledger.ledger_root(ctx)
    original = review_ledger.load_record(root, first["record_id"])
    write = rc._write

    def late_settle_then_write(drive, record, *, rejoin=False):
        if rejoin:
            seats = [{"slot_id": row["seat_id"], "model": row["requested"]["model"], "parts": ["result"],
                      "raw_text": "[]", "status": "completed", "operation_state": "settled"} for row in original["rows"]]
            settled = _settle_left_record({"record_id": record.record_id, "drive_root": drive}, seats, final=False)
            assert settled["state"] == "settled" and settled["verdict"]["aggregate"] == "PASS"
        return write(drive, record, rejoin=rejoin)

    monkeypatch.setattr(rc, "_write", late_settle_then_write)
    assert result_review_at_delivery(ctx, args["answer"], {"tool_calls": []}, lambda _: None) is False
    assert review_ledger.load_record(root, first["record_id"])["fingerprints"] == original["fingerprints"]
    assert rc.run_review_change(ctx, **args)["reused"] is True



@pytest.mark.parametrize("words_change_the_request", [True, False])
def test_new_owner_words_after_a_review_of_the_same_answer_are_asked_about_once(tmp_path, monkeypatch,
                                                                                words_change_the_request):
    """Owner, 18.10 (A): the only review of this exact answer judged an earlier request, so the host shows the
    owner's words since then once and asks. If they change what was asked, the mind asks for a review of the
    current request (the reviewers now see them and FAIL it; Blocking holds); if not, it delivers again and the
    earlier review stands, with no new seat."""
    from ouroboros.tools import review_result as rr

    ctx = _floor_ctx(tmp_path, monkeypatch, enforcement="blocking")
    _produced_answer_protocol(monkeypatch)
    sends: list[dict] = []
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", _seam(sends))
    trace, shown = {"tool_calls": []}, []
    assert rr.result_review_at_delivery(ctx, "FINAL ANSWER: 12 miles", trace, shown.append) is False, "PASS delivers"
    paid = len(sends)
    words = "Requirement change: give the distance in kilometres."
    ctx._owner_directives = [*(getattr(ctx, "_owner_directives", None) or []),
                             {"source": "owner_mailbox", "content": words, "msg_id": "m-2"}]
    assert rr.result_review_at_delivery(ctx, "FINAL ANSWER: 12 miles", trace, shown.append) is True, "asked once"
    assert words in shown[-1] and len(sends) == paid
    if words_change_the_request:
        monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", _seam(sends, {"t1": CRITICAL}))
        current = run_review_change(ctx, subject="result", answer="FINAL ANSWER: 12 miles")
        assert current["aggregate"] == "FAIL" and len(sends) == paid + 3
        reviewed = review_ledger.load_record(review_ledger.ledger_root(ctx), current["record_id"])["subject"]["diff_sha"]
        assert words in rr._retained_subject_text(ctx, reviewed), "the reviewers saw the new words"
        assert rr.result_review_at_delivery(ctx, "FINAL ANSWER: 12 miles", trace, shown.append) is True
        assert "FAIL" in shown[-1], "Blocking holds over the review of the current request"
    else:
        assert rr.result_review_at_delivery(ctx, "FINAL ANSWER: 12 miles", trace, shown.append) is False
        current = [run for run in trace["review_runs"] if not run.get("superseded_by_revision")]
        assert [(run["aggregate_signal"], run.get("earlier_request")) for run in current] == [("PASS", True)]
        assert len(sends) == paid, "the earlier review stands; nothing new is bought"


def test_a_refusal_is_no_review_of_any_request(tmp_path, monkeypatch):
    """A $0 refusal froze no reviewed request: once the ceiling allows, the floor reviews the request as it is now
    (the owner's later words included), never the one the refusal happened to carry."""
    from ouroboros.tools import review_result as rr

    ctx = _floor_ctx(tmp_path, monkeypatch, mode="required", enforcement="advisory")
    _produced_answer_protocol(monkeypatch)
    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "1")
    sends: list[dict] = []
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", _seam(sends, {"t1": CRITICAL}))
    trace, shown = {"tool_calls": []}, []
    assert rr.result_review_at_delivery(ctx, "FINAL ANSWER: 2", trace, shown.append) is True  # FAIL shown
    assert rr.result_review_at_delivery(ctx, "FINAL ANSWER: 3", trace, shown.append) is True  # refused, shown
    words = "Also state the units."
    ctx._owner_directives = [*(getattr(ctx, "_owner_directives", None) or []),
                             {"source": "owner_mailbox", "content": words, "msg_id": "m-3"}]
    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "5")
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", _seam(sends))
    paid = len(sends)
    assert rr.result_review_at_delivery(ctx, "FINAL ANSWER: 3", trace, shown.append) is False
    assert len(sends) == paid + 3, "the floor reviewed the current request"
    current = [run for run in trace["review_runs"] if not run.get("superseded_by_revision")]
    record = review_ledger.load_record(review_ledger.ledger_root(ctx), current[-1]["record_id"])
    assert record["verdict"]["aggregate"] == "PASS" and not current[-1].get("earlier_request")
    assert words in rr._retained_subject_text(ctx, record["subject"]["diff_sha"])


def test_another_call_of_the_same_answer_reviews_the_current_request_not_the_running_one(tmp_path, monkeypatch):
    """Only the same unchanged call collects the running review of an earlier request: a call with another brief
    (here another goal) reviews the request as it is now, never the earlier subject, and leaves that record alone."""
    from ouroboros.tools import review_result as rr
    from tests.test_review_change_end_to_end import _pending_then_answering_seat

    ctx = _floor_ctx(tmp_path, monkeypatch, mode="auto")
    sends: list[dict] = []
    answer, _starts = _pending_then_answering_seat("t2", "other-brief")
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", shared.golden_physical_seam(sends, answer=answer))
    first = run_review_change(ctx, subject="result", answer="The sum is 12.", goal="Check the sum")
    assert first["state"] == "pending"
    root = review_ledger.ledger_root(ctx)
    before = review_ledger.load_record(root, first["record_id"])
    words = "Also show the working."
    ctx._owner_directives = [*(getattr(ctx, "_owner_directives", None) or []),
                             {"source": "owner_mailbox", "content": words, "msg_id": "m-2"}]
    other = run_review_change(ctx, subject="result", answer="The sum is 12.", goal="Check the working too")
    assert other["record_id"] != first["record_id"]
    reviewed = review_ledger.load_record(root, other["record_id"])["subject"]["diff_sha"]
    assert words in rr._retained_subject_text(ctx, reviewed), "the current request was reviewed"
    assert review_ledger.load_record(root, first["record_id"])["subject"] == before["subject"]


def test_alternating_answers_at_a_spent_blocking_ceiling_hold_with_each_refusal_never_crash(tmp_path, monkeypatch):
    """A -> B -> A -> B under Blocking with the allowance spent: every delivery of a refused answer holds with its
    own refusal (read from its record, though another answer's run superseded its trace run), never raises."""
    from ouroboros.tools import review_result as rr

    ctx = _floor_ctx(tmp_path, monkeypatch, enforcement="blocking")
    _produced_answer_protocol(monkeypatch)
    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "1")
    sends: list[dict] = []
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", _seam(sends, {"t1": CRITICAL}))
    trace, shown = {"tool_calls": []}, []
    for answer in ("FINAL ANSWER: 2", "FINAL ANSWER: 3", "FINAL ANSWER: 2", "FINAL ANSWER: 3"):
        assert rr.result_review_at_delivery(ctx, answer, trace, shown.append) is True, answer
    assert "raises the review ceiling" in shown[-1] and len(sends) == 3


def test_the_host_floor_seats_the_whole_pool_and_records_that_the_host_chose(tmp_path, monkeypatch):
    from ouroboros.review_projection import ledger_record_panel
    from ouroboros.tools.review_result import result_review_at_delivery

    ctx = _floor_ctx(tmp_path, monkeypatch, enforcement="blocking")
    _produced_answer_protocol(monkeypatch)
    sends: list[dict] = []
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", _seam(sends))
    trace = {"tool_calls": []}

    assert result_review_at_delivery(ctx, "FINAL ANSWER: 2", trace, lambda _text: None) is False
    assert len(sends) == 3
    [run] = trace["review_runs"]
    record = review_ledger.load_record(ctx.drive_root, run["record_id"])
    assert record["panel"]["chosen_by"] == "host" and record["panel"]["composition"] == "full_pool"
    assert set(record["panel"]["assigned"]) == {"t1", "t2", "s1"} and record["panel"]["widened_from"] == []
    assert ledger_record_panel(record)["panel_facts"]["chosen_by"] == "host"


def test_a_minds_review_of_a_required_root_stays_chosen_by_ouroboros(tmp_path, monkeypatch):
    from ouroboros.tools.review_result import result_review_at_delivery

    ctx = _floor_ctx(tmp_path, monkeypatch, enforcement="blocking")
    _produced_answer_protocol(monkeypatch)
    sends: list[dict] = []
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", _seam(sends))
    answer = "FINAL ANSWER: 2"

    first = run_review_change(ctx, subject="result", answer=answer)
    assert first["panel"]["chosen_by"] == "ouroboros" and set(first["panel"]["assigned"]) == {"t1", "t2", "s1"}
    assert result_review_at_delivery(ctx, answer, {"tool_calls": []}, lambda _text: None) is False
    assert len(sends) == 3, "a review the mind already ran is not bought again as the host's"


def test_advisory_required_refuses_an_unexplained_subset_and_composes_a_reason(tmp_path, monkeypatch):
    ctx = _floor_ctx(tmp_path, monkeypatch)
    sends: list[dict] = []
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", _seam(sends))

    with pytest.raises(ReviewChangeArgumentError, match="REVIEW_PANEL_REASON_REQUIRED"):
        run_review_change(ctx, subject="result", answer="Two findings.", reviewers=["t1"])
    assert sends == []
    composed = run_review_change(ctx, subject="result", answer="Two findings, with a reason.",
                                 reviewers=["t1"], reason="the risky claim")
    assert composed["panel"]["assigned"] == ["t1"] and composed["panel"]["chosen_by"] == "ouroboros"
    assert composed["panel"]["widened_from"] == [] and [send["slot_id"] for send in sends] == ["t1"]


def test_auto_blocking_composes_a_named_subset_and_a_child_is_not_widened(tmp_path, monkeypatch):
    auto = _floor_ctx(tmp_path / "auto", monkeypatch, mode="auto", enforcement="blocking")
    sends: list[dict] = []
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", _seam(sends))
    named = run_review_change(auto, subject="result", answer="Auto keeps the named row.", reviewers=["t1"])
    assert named["panel"]["assigned"] == ["t1"] and named["panel"]["widened_from"] == []

    child = _floor_ctx(tmp_path / "child", monkeypatch, enforcement="blocking", root=False)
    widened = run_review_change(child, subject="result", answer="A child keeps the named row.", reviewers=["t1"])
    assert widened["panel"]["assigned"] == ["t1"] and widened["panel"]["widened_from"] == []
    assert sorted(send["slot_id"] for send in sends) == ["t1", "t1"]
