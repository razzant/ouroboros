"""``review_change(subject="result")``: one review machinery for a change and for a task's result.

Every seat of a result wave answers one question — does this task's result do what was asked — over
a subject frozen at the call (what was asked, the reply, what the task produced, the host-attested
verification). The wave is review_change's own: the same panel rules, $0 reuse of the same subject,
the ledger record, and its own paid-cycle count beside change reviews. Task review Off refuses it.
"""
from __future__ import annotations

import json
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

def _floor_ctx(tmp_path, monkeypatch, *, mode="required", root=True):
    ctx, _project = _foreign_project(tmp_path, monkeypatch)
    monkeypatch.setenv("OUROBOROS_TASK_REVIEW_MODE", mode)
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


def test_a_required_root_result_is_reviewed_through_a_real_session_seat(tmp_path, monkeypatch, fake_route):
    """The Required floor on the real Claudexor session route (offline gateway): the seat is asked for the
    findings array, never the change/coupling object; its critical finding is shown before delivery, and the
    author's revised result is not reviewed again unless the author asks."""
    import json

    from ouroboros.tools.review_result import result_review_at_delivery

    ctx = _floor_ctx(tmp_path, monkeypatch)
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
    # The author's revision: the earlier version's findings are put before it once more, then it delivers.
    assert result_review_at_delivery(ctx, "FINAL ANSWER: 3", trace, shown.append) is True
    assert "An earlier version of your result was reviewed" in shown[-1] and "not reviewed again unless you ask" in shown[-1]
    assert result_review_at_delivery(ctx, "FINAL ANSWER: 3", trace, shown.append) is False, "the author's revision"
    starts = [body for gateway in FakeGateway.instances for body in gateway.start_requests]
    assert len(starts) == 1 and all(body["outputSchema"]["required"] == ["findings"] for body in starts), starts


def test_nothing_is_forced_for_a_child_auto_off_or_a_plain_text_answer(tmp_path, monkeypatch):
    from ouroboros.tools.review_result import result_review_at_delivery

    sends: list[dict] = []
    for kwargs, protocol in (({"root": False}, True), ({"root": False, "mode": "auto"}, True),
                             ({"mode": "auto"}, True), ({"mode": "off"}, True), ({}, False)):
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
    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "1")
    _produced_answer_protocol(monkeypatch)
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", _seam(sends, {"t1": CRITICAL}))
    # The host stamps a Presence turn before any review (``presence_runner._build_task``).
    ctx.task_metadata = {**dict(getattr(ctx, "task_metadata", None) or {}), "presence_event_identity": "event-1"}
    trace, shown = {"tool_calls": []}, []
    assert result_review_at_delivery(ctx, "FINAL ANSWER: 2", trace, shown.append) is False, \
        "Presence speech is never held for a review"
    assert shown == [] and len(sends) == 3, "its owed review still runs, with nobody waiting for it"
    [run] = trace["review_runs"]
    presence = review_ledger.load_record(ctx.drive_root, run["record_id"])
    assert presence["panel"]["chosen_by"] == "host" and presence["panel"]["composition"] == "full_pool"
    assert set(presence["panel"]["assigned"]) == {"t1", "t2", "s1"}
    assert result_review_at_delivery(ctx, "FINAL ANSWER: 2", trace, shown.append) is False and len(sends) == 3
    for _ in range(3):  # the same answer reopened by place mail keeps one $0 refusal, without holding speech
        assert result_review_at_delivery(ctx, "FINAL ANSWER: 3", trace, shown.append) is False
    rows = review_ledger.recent_records(ctx.drive_root, ctx.task_id, limit=50, surface="result")
    refused = [row for row in rows if (row.get("verdict") or {}).get("aggregate") == "NOT_DISPATCHED"]
    assert len(refused) == 1 and len(sends) == 3 and shown == []
    current = [run for run in trace["review_runs"] if not run.get("superseded_by_revision")]
    assert len(current) == 1 and current[0]["record_id"] == refused[0]["record_id"]
    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "2")
    assert result_review_at_delivery(ctx, "FINAL ANSWER: 3", trace, shown.append) is False
    assert len(sends) == 6 and shown == [], "a raised ceiling permits its review; Presence still never holds"
    sends.clear()

    # The expected result a task was given is intent, never production (owner, 11 Oct, #1731).
    ctx = _floor_ctx(tmp_path / "declared", monkeypatch)
    monkeypatch.setattr("ouroboros.contracts.task_contract.answer_protocol_active", lambda ctx: False)
    ctx.task_metadata = {**dict(getattr(ctx, "task_metadata", None) or {}),
                         "task_contract": {"expected_output": "A report with three findings",
                                           "acceptance_criteria": ["three findings"]}}
    answer = "The report lists three findings."
    assert required_material(ctx, {"tool_calls": []}, answer) == ""
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

    assert result_review_at_delivery(ctx, answer, trace, lambda text: None) is True, "no review tool returned it"
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
    finish = run["informed_finish"]  # bound to the exact subject delivered and the outcome that reached the author
    assert finish["subject_sha"] == run["binding_hash"] and finish["record_ids"] == [run["record_id"]]
    assert finish["outcomes"] and run["enforcement"] == "advisory"
    review = _review_axis(trace)
    assert (review["status"], review["source"], review.get("author_finish")) == ("fail", "result_review", "author_finish")
    assert _objective_axis(review)["source"] == "author_acceptance" and _objective_axis(review)["status"] == "pass"


def test_money_refusal_keeps_its_recovery_not_the_review_cycle_ceiling(tmp_path, monkeypatch):
    from ouroboros import usage_accounting as ua
    from ouroboros.tools.review_result import result_review_at_delivery

    ctx = _floor_ctx(tmp_path, monkeypatch)
    _produced_answer_protocol(monkeypatch)
    monkeypatch.setattr(ua, "_reservation_cost", lambda *_a, **_kw: 1.)
    sends = []
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", _seam(sends))
    scope = ua.UsageScope(drive_root=str(ctx.drive_root), task_id=ctx.task_id,
                          root_task_id=ctx.task_id, root_limit_usd=1., global_limit_usd=10.)
    trace, shown = {"tool_calls": []}, []
    with ua.usage_scope(scope):
        ua.execute_physical_attempt(ua.AttemptRequest(model="fake", provider="test", reservation_usd=1.),
                                    lambda: {}, extractor=lambda _: ({"prompt_tokens": 1}, 1., True))
        assert result_review_at_delivery(ctx, "FINAL ANSWER: 2", trace, shown.append) is True
    assert not sends
    assert trace["review_runs"][0]["dispatch_refusal"] == "review_wave_budget_insufficient"
    assert "raises the review ceiling" not in shown[-1]
    assert "per-task budget" in shown[-1]
    assert "was reviewed before delivery" not in shown[-1]


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
    assert [run["informed_finish"]["subject_sha"] for run in current] == [current[0]["binding_hash"]]
    assert len(sends) == 3 and _objective_axis(_review_axis(trace))["source"] == "author_acceptance"


@pytest.mark.parametrize(("checkpointed", "owner_message"), [(False, False), (True, True)])
def test_delivery_collects_the_running_review_the_mind_started_without_buying_another(tmp_path, monkeypatch,
                                                                                      owner_message, checkpointed):
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
        assert result_review_at_delivery(ctx, "FINAL ANSWER: 2", trace, shown.append) is True, mode
        if owner_message:  # one question, which carries the collected outcome too
            assert "How far along are you?" in shown[0] and "deliver again" in shown[0], shown
        assert "(record " in shown[0] and "PASS" in shown[0], "the collected outcome reaches the mind first"
        assert result_review_at_delivery(ctx, "FINAL ANSWER: 2", trace, shown.append) is False, mode
        assert len(sends) == 4 and sends[3]["slot_id"] == "t2" and sends[3]["reconcile_only"] is True, \
            "the delivery rejoined the running wave's own seat; no second wave was bought"
        [run] = trace["review_runs"]
        assert (run["record_id"], run["aggregate_signal"]) == (first["record_id"], "PASS"), mode
        record = review_ledger.load_record(ctx.drive_root, first["record_id"])
        assert record["brief"]["goal"] == "Check the arithmetic", "the collected record keeps the mind's brief"
        from ouroboros.chronicle_store import ChronicleStore, is_late_review_mark
        from ouroboros.late_review_marks import replay_owed_late_reviews
        from ouroboros.task_results import write_task_result
        from ouroboros.working_checkpoint import flush_content_acks

        assert not (record.get("waiting") or {}).get("seen_outcome_states"), "showing alone is not a saved receipt"
        if checkpointed:
            flush_content_acks(ctx)  # the successful transcript checkpoint's commit point
        write_task_result(ctx.drive_root, ctx.task_id, "completed", chat_id=1)
        assert replay_owed_late_reviews(ctx.drive_root)["settled"] == 1
        marks = [mark for mark in ChronicleStore(ctx.drive_root).active_marks(None) if is_late_review_mark(mark)]
        assert bool(marks) is (not checkpointed), "only a whole saved host handback makes the unwaited mark redundant"


def test_an_earlier_answers_pass_stands_beside_the_delivered_one_as_that_versions(tmp_path, monkeypatch):
    """Advisory, 1=B (16 Sep 2026): the PASS of A, received, accompanies the revised B's delivery as an earlier
    version's PASS; it never becomes a review of B's bytes and buys nothing."""
    from ouroboros.outcomes import _objective_axis, _review_axis
    from ouroboros.tools.review_result import result_review_at_delivery

    ctx = _floor_ctx(tmp_path, monkeypatch, mode="auto")
    sends: list[dict] = []
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", _seam(sends))
    assert run_review_change(ctx, subject="result", answer="Report A")["aggregate"] == "PASS"
    trace = {"tool_calls": []}
    assert result_review_at_delivery(ctx, "Report A", trace, lambda text: None) is True  # handed back once
    assert result_review_at_delivery(ctx, "Report A", trace, lambda text: None) is False
    reviewed = trace["review_runs"][0]["binding_hash"]
    assert result_review_at_delivery(ctx, "Report B", trace, lambda text: None) is False and len(sends) == 3
    [run] = trace["review_runs"]
    assert run["earlier_revision"] is True and run["reviewed_subject_sha"] == reviewed != run["binding_hash"]
    review = _review_axis(trace)
    assert (review["status"], review["author_finish"]) == ("pass", "previous_revision_accepted")
    assert _objective_axis(review) == {"status": "pass", "source": "author_acceptance", "review_status": "pass",
                                       "reason": "previous_revision_accepted"}


def test_pending_review_finish_delivers_over_a_running_review(tmp_path, monkeypatch):
    import ouroboros.tools.review_change as review_change_mod
    from ouroboros.tools.review_result import pending_review_choice_offered, result_review_at_delivery

    ctx = _floor_ctx(tmp_path, monkeypatch)
    _produced_answer_protocol(monkeypatch)
    monkeypatch.setattr(review_change_mod, "run_review_change",
                        lambda ctx, **kwargs: {"state": "pending", "record_id": "rl-running"})
    shown = []
    assert result_review_at_delivery(ctx, "FINAL ANSWER: 2", {"tool_calls": []}, shown.append) is True
    assert pending_review_choice_offered(ctx) is True
    assert "pending_review=finish" in shown[0]
    ctx._acceptance_pending_review_choice = "finish"
    assert result_review_at_delivery(ctx, "FINAL ANSWER: 2", {"tool_calls": []}, shown.append) is False


@pytest.mark.parametrize("runtime", ["pro", "cyber_pro"])
def test_a_running_review_the_mind_started_is_waited_for_unless_it_chose_to_finish(tmp_path, monkeypatch, runtime):
    import pathlib

    import ouroboros.tools.review_result as review_result
    from ouroboros.body_candidate import serving_repo_dir_for

    ctx = _floor_ctx(tmp_path, monkeypatch, mode="auto")
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
    assert review_result.result_review_at_delivery(ctx, answer, {"tool_calls": []}, shown.append) is False, \
        "an explicit finish is honoured"
    rows[0]["subject"] = {"diff_sha": "another-answer"}
    ctx._acceptance_pending_review_choice = "wait"
    assert review_result.result_review_at_delivery(ctx, answer, {"tool_calls": []}, shown.append) is False, \
        "a review of an earlier answer never holds this one"


# --- Transitions found by the second review round -----------------------------------------------

def _sibling_spent_allowance(tmp_path, monkeypatch):
    """Two children of one root under Auto with one paid result review in the tree: the first child's PASS spends
    it, so the second child's own call is refused at $0 (``review_change._refuse_exhausted``)."""
    ctx = _floor_ctx(tmp_path, monkeypatch, mode="auto", root=False)
    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "1")
    sends: list[dict] = []
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", _seam(sends))
    ctx.task_id = "first-child"
    assert run_review_change(ctx, subject="result", answer="The first child's answer.")["aggregate"] == "PASS"
    ctx.task_id = "second-child"
    refused = run_review_change(ctx, subject="result", answer=ANSWER_2, goal="Check the second child's count")
    assert refused["aggregate"] == "NOT_DISPATCHED" and refused["dispatch_refusal"]["cap"] == 1
    return ctx, sends, refused


ANSWER_2 = "The second child counted 2."
REVISED_2 = "The second child counted 2 apples."


@pytest.mark.parametrize("case", ["new_task", "advisory"])
def test_a_revision_keeps_the_review_chosen_for_its_result_after_a_refusal_as_after_a_fail(tmp_path, monkeypatch, case):
    """Root, 9 Nov 2026 (owner Q3, Auto chooses): revising the same requested result after its chosen review was
    refused does not erase that choice, as revising after a FAIL does not (``_other_version``): the received refusal
    leaves the revision to its author's reasoned finish, beside the refusal (Done with warnings), with no wave.
    Control: another task of the same root (a new request, its own ownership) that chose no review is not held by
    this one's refusal."""
    from ouroboros.outcomes import _review_axis
    from ouroboros.tools.review_result import result_review_at_delivery

    ctx, sends, refused = _sibling_spent_allowance(tmp_path, monkeypatch)
    receipt = {"record_id": refused["record_id"], "outcome_states": []}  # the tool's whole return of the refusal
    trace, shown = {"tool_calls": [{"tool": "review_change", "tool_result_meta": {"review_delivery": receipt}}]}, []
    if case == "new_task":
        ctx.task_id = "third-child"
        assert result_review_at_delivery(ctx, REVISED_2, {"tool_calls": []}, shown.append) is False and shown == []
        assert len(sends) == 3
        return
    ctx._completion_selected = {"action": "finish", "answer": REVISED_2, "rationale": "Named the unit; the count stands."}
    assert result_review_at_delivery(ctx, REVISED_2, trace, shown.append) is False and shown == [] and len(sends) == 3
    [run] = [run for run in trace["review_runs"] if not run.get("superseded_by_revision")]
    assert run["aggregate"] == "NOT_DISPATCHED" and run["earlier_revision"] is True and run["informed_finish"]
    assert _review_axis(trace)["author_finish"] == "author_finish"


@pytest.mark.parametrize("returned", [True, False])
def test_an_advisory_author_delivers_beside_its_chosen_reviews_refusal(tmp_path, monkeypatch, returned):
    """The same refusal under Advisory is evidence, never a hold or a wave: the review tool's whole return of it is its
    author's receipt, so delivery is at once its informed decision (``returned``); one it never received is shown
    once first. Either way the result is Done with warnings beside the refusal, the chosen review that never ran."""
    from ouroboros.outcomes import _objective_axis, _review_axis
    from ouroboros.project_dialogue import outcome_phase
    from ouroboros.tools.review_result import result_review_at_delivery

    ctx, sends, refused = _sibling_spent_allowance(tmp_path, monkeypatch)
    receipt = {"record_id": refused["record_id"], "outcome_states": []}  # ``review_waiting.publish_delivery``'s shape
    trace = {"tool_calls": [{"tool": "review_change", "tool_result_meta": {"review_delivery": receipt}}] if returned else []}
    shown = []
    if not returned:
        assert result_review_at_delivery(ctx, ANSWER_2, trace, shown.append) is True and "NOT_DISPATCHED" in shown[0]
    assert result_review_at_delivery(ctx, ANSWER_2, trace, shown.append) is False
    assert len(shown) == (0 if returned else 1) and len(sends) == 3
    review = _review_axis(trace)
    assert review["status"] == "degraded" and review["author_finish"] == "author_finish" and "blocked_reason" not in review
    axes = {"execution": {"status": "ok"}, "review": review, "objective": _objective_axis(review)}
    assert outcome_phase({"status": "completed", "outcome_axes": axes}, {}) == "warn"


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
    """The round settles FAIL after the delivery step saw it pending, so the collection finds no open
    operation; the step reads the record again and shows its findings first instead of delivering."""
    from ouroboros.tools import review_result as rr
    from tests.test_review_change_end_to_end import _pending_then_answering_seat

    ctx = _floor_ctx(tmp_path, monkeypatch)
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
    assert rr.result_review_at_delivery(ctx, "Reviewed answer A", trace, lambda _: None)  # handed back once
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
    assert result_review_at_delivery(ctx, "The sum is 12.", {"tool_calls": []}, lambda _: None) is True  # collected
    settled = review_ledger.load_record(root, first["record_id"])
    assert settled["state"] == "settled" and settled["verdict"]["aggregate"] == "PASS"
    assert settled["fingerprints"] == pending["fingerprints"] and settled["brief"] == pending["brief"]
    paid = len(sends)
    again = run_review_change(ctx, **args)
    assert again["reused"] is True and again["record_id"] == first["record_id"] and len(sends) == paid


def test_a_running_review_whose_operation_was_lost_never_lets_a_required_result_pass_unreviewed(tmp_path, monkeypatch):
    """A restart can close a tokenless seat's attempt while its ledger row still reads pending: the collection
    finds no open operation. Under Required the floor then buys its own review (the lost attempt produced no
    verdict) instead of delivering the result unreviewed, and shows that review's findings first."""
    from ouroboros.tools import review_change as rc
    from ouroboros.tools import review_result as rr
    from tests.test_review_change_end_to_end import _pending_then_answering_seat

    ctx = _floor_ctx(tmp_path, monkeypatch)
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
    # never delivered unreviewed: a run is bound and its outcome is shown first.
    assert [run["aggregate_signal"] for run in trace["review_runs"]] in (["FAIL"], ["DEGRADED"])
    assert "was reviewed before delivery" in shown[0] and "delivering it is your own acceptance" in shown[0]
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
    assert result_review_at_delivery(ctx, args["answer"], {"tool_calls": []}, lambda _: None) is True  # collected
    assert review_ledger.load_record(root, first["record_id"])["fingerprints"] == original["fingerprints"]
    assert rc.run_review_change(ctx, **args)["reused"] is True



@pytest.mark.parametrize("words_change_the_request", [True, False])
def test_new_owner_words_after_a_review_of_the_same_answer_are_asked_about_once(tmp_path, monkeypatch,
                                                                                words_change_the_request):
    """Owner, 18.10 (A): the only review of this exact answer judged an earlier request, so the host shows the
    owner's words since then once and asks. If they change what was asked, the mind asks for a review of the
    current request (the reviewers now see them and FAIL it; the FAIL is shown first); if not, it delivers again and
    the earlier review stands, with no new seat.

    The question names each new row by its identity — the reference token of its stamp, whole — never by likeness
    of words or of the whole stamp: a one-word row and a Project room's row are named with their stamps; ``msg m-1``
    is not ``msg m-12`` (the earlier subject holds the longer token, which a substring match would read as the
    shorter); a row whose source or date merely changed, or that the earlier subject held under its pre-update
    ``msg`` reference, is not named; a subject that cannot be read back names every row; none of it buys a seat."""
    from ouroboros import dialogue_store as ds
    from ouroboros.projects_registry import bind_task_to_project, create_project
    from ouroboros.tools import review_result as rr

    ctx = _floor_ctx(tmp_path, monkeypatch)
    _produced_answer_protocol(monkeypatch)
    room = create_project(ctx.drive_root, "quotas", name="Quotas")["chat_id"]
    bind_task_to_project(ctx.drive_root, ctx.task_id, "quotas", room, origin={"absent": "system"})
    ctx._owner_directives = [{"source": "owner_mailbox", "content": "да", "msg_id": "m-12", "received": "2026-10-01T00:00:00+00:00"},
                             {"source": "owner_mailbox", "content": "Use miles", "msg_id": "m-3", "client_message_id": "cm-3"}]
    sends: list[dict] = []
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", _seam(sends))
    trace, shown = {"tool_calls": []}, []
    assert rr.result_review_at_delivery(ctx, "FINAL ANSWER: 12 miles", trace, shown.append) is True, "its PASS first"
    assert rr.result_review_at_delivery(ctx, "FINAL ANSWER: 12 miles", trace, shown.append) is False, "then delivers"
    paid = len(sends)
    words = "Requirement change: give the distance in kilometres."
    # Since that review: the first «да» rides on under another source and date (a Continue's corpus); the pre-update
    # row is now held with its chat row (one identity, two printed references); a second «да» (``msg m-1``, inside
    # the seen ``msg m-12`` as a substring), the change, and a «да» in the Project room are new.
    ctx._owner_directives = [{"source": "owner_corpus", "content": "да", "msg_id": "m-12", "ts": "2026-10-01T00:00:01+00:00"},
                             ctx._owner_directives[1], {"source": "owner_mailbox", "content": "да", "msg_id": "m-1"},
                             {"source": "owner_mailbox", "content": words, "msg_id": "m-2"}]
    ctx._held_owner_messages = {"row:held": {"ts": "2026-10-01T00:00:02+00:00", "direction": "in", "chat_id": 1,
                                             "text": "Use miles", "client_message_id": "cm-3", "task_id": ""}}
    ds.append(ctx.drive_root, "chat", {"ts": "2026-10-01T00:00:03+00:00", "direction": "in", "chat_id": room, "text": "да",
                                       "client_message_id": "cm-room", "task_id": ""})
    assert rr.result_review_at_delivery(ctx, "FINAL ANSWER: 12 miles", trace, shown.append) is True, "asked once"
    assert words in shown[-1] and len(sends) == paid
    assert [line for line in shown[-1].split("\n") if line.startswith("- [")] == [
        f"- [owner · owner_mailbox of task {ctx.task_id} · msg m-1]", f"- [owner · owner_mailbox of task {ctx.task_id} · msg m-2]",
        f"- [2026-10-01T00:00:03+00:00 · owner · room_message · chat {room} / cm-room]"]
    assert "msg m-12]" not in shown[-1] and "cm-3" not in shown[-1] and "Use miles" not in shown[-1]
    # A retained subject that cannot be read back names every current row, and buys nothing.
    every = rr._request_changed_text(ctx, {"record_id": "lost", "subject": {"diff_sha": "0" * 64}, "verdict": {"aggregate": "PASS"}})
    assert every.count("\n- [") == 5 and "msg m-12]" in every and "chat 1 / cm-3]" in every and len(sends) == paid
    if words_change_the_request:
        monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", _seam(sends, {"t1": CRITICAL}))
        current = run_review_change(ctx, subject="result", answer="FINAL ANSWER: 12 miles")
        assert current["aggregate"] == "FAIL" and len(sends) == paid + 3
        reviewed = review_ledger.load_record(review_ledger.ledger_root(ctx), current["record_id"])["subject"]["diff_sha"]
        assert words in rr._retained_subject_text(ctx, reviewed), "the reviewers saw the new words"
        assert rr.result_review_at_delivery(ctx, "FINAL ANSWER: 12 miles", trace, shown.append) is True
        assert "FAIL" in shown[-1], "the review of the current request is shown first"
    else:
        assert rr.result_review_at_delivery(ctx, "FINAL ANSWER: 12 miles", trace, shown.append) is False
        current = [run for run in trace["review_runs"] if not run.get("superseded_by_revision")]
        assert [(run["aggregate_signal"], run.get("earlier_request")) for run in current] == [("PASS", True)]
        assert len(sends) == paid, "the earlier review stands; nothing new is bought"


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


def _floor_review(tmp_path, monkeypatch, answers=None):
    """The Required root of the tests above over the golden seats: ``(ctx, sends, trace, shown)``."""
    ctx = _floor_ctx(tmp_path, monkeypatch)
    _produced_answer_protocol(monkeypatch)
    sends: list[dict] = []
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", _seam(sends, answers))
    return ctx, sends, {"tool_calls": []}, []


@pytest.mark.parametrize("writer", ["working", "owner_wait"])
def test_findings_shown_before_a_cold_continuation_are_not_shown_again(tmp_path, monkeypatch, writer):
    """Advisory: findings shown, the process ends; the same task continued from either saved source delivers on
    the mind's decision instead of holding for the same findings again."""
    from ouroboros.tools import review_result as rr
    from tests.test_working_checkpoint import cold_continuation

    ctx, sends, trace, shown = _floor_review(tmp_path, monkeypatch, answers={"t1": CRITICAL})
    assert rr.result_review_at_delivery(ctx, "FINAL ANSWER: 2", trace, shown.append) is True, "findings first"
    before = ctx._result_review_shown = [f"review:{'0' * 32}#FAIL", *ctx._result_review_shown]  # #2339's token form
    ctx, trace = cold_continuation(ctx, trace, monkeypatch, writer=writer)
    assert ctx._result_review_shown == before, "a token beside a record id, value for value and in order"
    assert rr.result_review_at_delivery(ctx, "FINAL ANSWER: 2", trace, shown.append) is False, "the mind decided"
    [run] = [run for run in trace["review_runs"] if not run.get("superseded_by_revision")]
    assert run["informed_finish"]["subject_sha"] == run["binding_hash"]
    assert len(shown) == 1 and len(sends) == 3, "shown once, reviewed once"


def test_a_changed_request_is_asked_about_once_per_subject_across_a_cold_continuation(tmp_path, monkeypatch):
    """The ask binds (review, subject): that pair is not asked again after the process ended, while newer owner
    words over the SAME answer form a new subject that is asked about once more."""
    from ouroboros.tools import review_result as rr
    from tests.test_working_checkpoint import cold_continuation

    ctx, sends, trace, shown = _floor_review(tmp_path, monkeypatch)
    answer = "FINAL ANSWER: 12 miles"
    assert rr.result_review_at_delivery(ctx, answer, trace, shown.append) is True, "unreceived PASS is shown"
    assert rr.result_review_at_delivery(ctx, answer, trace, shown.append) is False, "the mind decided"
    ctx._owner_directives = [{"source": "owner_mailbox", "content": "Also mention the route.", "msg_id": "m-2"}]
    assert rr.result_review_at_delivery(ctx, answer, trace, shown.append) is True, "asked once"
    ctx, trace = cold_continuation(ctx, trace, monkeypatch, writer="owner_wait")
    assert rr.result_review_at_delivery(ctx, answer, trace, shown.append) is False, "the same pair is not asked again"
    ctx._owner_directives.append({"source": "owner_mailbox", "content": "And name the start town.", "msg_id": "m-3"})
    assert rr.result_review_at_delivery(ctx, answer, trace, shown.append) is True, "a new subject is asked about"
    assert "And name the start town." in shown[-1] and len(shown) == 3 and len(sends) == 3


def test_a_shown_refusal_stands_across_a_cold_continuation(tmp_path, monkeypatch):
    """A shown $0 refusal outlives the process: the mind's informed decision delivers, writing no new refusal."""
    from ouroboros.tools import review_result as rr
    from tests.test_working_checkpoint import cold_continuation

    ctx, sends, trace, shown = _floor_review(tmp_path, monkeypatch, answers={"t1": CRITICAL})
    monkeypatch.setenv("OUROBOROS_REVIEW_MAX_CYCLES", "1")
    assert rr.result_review_at_delivery(ctx, "FINAL ANSWER: 2", trace, shown.append) is True  # FAIL shown
    assert rr.result_review_at_delivery(ctx, "FINAL ANSWER: 3", trace, shown.append) is True  # refused, shown

    root, task_id = review_ledger.ledger_root(ctx), ctx.task_id
    written = len(review_ledger.recent_records(root, task_id, limit=50, surface="result"))
    ctx, trace = cold_continuation(ctx, trace, monkeypatch)
    delivered = rr.result_review_at_delivery(ctx, "FINAL ANSWER: 3", trace, shown.append)
    assert len(review_ledger.recent_records(root, task_id, limit=50, surface="result")) == written, "no new refusal"
    assert delivered is False and len(shown) == 2, "the mind decided"


@pytest.mark.parametrize("new_id, legacy", [(True, False), (False, True)])
def test_no_presentation_is_invented_for_a_new_execution_or_a_legacy_source(tmp_path, monkeypatch, new_id, legacy):
    """A new execution id borrows no review state and an older source cannot say what was shown: both start with
    nothing shown (nor a running review), so a legacy source's findings show once more, never reviewed again."""
    from ouroboros.tools import review_result as rr
    from tests.test_working_checkpoint import cold_continuation

    ctx, sends, trace, shown = _floor_review(tmp_path, monkeypatch, answers={"t1": CRITICAL})
    assert rr.result_review_at_delivery(ctx, "FINAL ANSWER: 2", trace, shown.append) is True
    fresh, restored = cold_continuation(ctx, trace, monkeypatch, new_id=new_id, legacy=legacy)
    assert ctx._result_review_shown and fresh._result_review_pending == ""
    assert not (fresh._result_review_shown or fresh._result_review_request_asked or fresh._result_review_refused)
    if legacy:
        assert rr.result_review_at_delivery(fresh, "FINAL ANSWER: 2", restored, shown.append) is True
        assert len(shown) == 2 and len(sends) == 3


def test_the_host_floor_seats_the_whole_pool_and_records_that_the_host_chose(tmp_path, monkeypatch):
    from ouroboros.review_projection import ledger_record_panel
    from ouroboros.tools.review_result import result_review_at_delivery

    ctx = _floor_ctx(tmp_path, monkeypatch)
    _produced_answer_protocol(monkeypatch)
    sends: list[dict] = []
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", _seam(sends))
    trace = {"tool_calls": []}

    assert result_review_at_delivery(ctx, "FINAL ANSWER: 2", trace, lambda _text: None) is True  # its PASS, first
    assert len(sends) == 3
    [run] = trace["review_runs"]
    record = review_ledger.load_record(ctx.drive_root, run["record_id"])
    assert record["panel"]["chosen_by"] == "host" and record["panel"]["composition"] == "full_pool"
    assert set(record["panel"]["assigned"]) == {"t1", "t2", "s1"} and "widened_from" not in record["panel"]
    assert ledger_record_panel(record)["panel_facts"]["chosen_by"] == "host"


def test_a_minds_review_of_a_required_root_stays_chosen_by_ouroboros(tmp_path, monkeypatch):
    from ouroboros.tools.review_result import result_review_at_delivery

    ctx = _floor_ctx(tmp_path, monkeypatch)
    _produced_answer_protocol(monkeypatch)
    sends: list[dict] = []
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", _seam(sends))
    answer = "FINAL ANSWER: 2"

    first = run_review_change(ctx, subject="result", answer=answer)
    assert first["panel"]["chosen_by"] == "ouroboros" and set(first["panel"]["assigned"]) == {"t1", "t2", "s1"}
    assert result_review_at_delivery(ctx, answer, {"tool_calls": []}, lambda _text: None) is True  # unread: shown
    assert len(sends) == 3, "a review the mind already ran is not bought again as the host's"


def test_required_refuses_an_unexplained_subset_and_composes_a_reason(tmp_path, monkeypatch):
    ctx = _floor_ctx(tmp_path, monkeypatch)
    sends: list[dict] = []
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", _seam(sends))

    with pytest.raises(ReviewChangeArgumentError, match="REVIEW_PANEL_REASON_REQUIRED"):
        run_review_change(ctx, subject="result", answer="Two findings.", reviewers=["t1"])
    assert sends == []
    composed = run_review_change(ctx, subject="result", answer="Two findings, with a reason.",
                                 reviewers=["t1"], reason="the risky claim")
    assert composed["panel"]["assigned"] == ["t1"] and composed["panel"]["chosen_by"] == "ouroboros"
    assert "widened_from" not in composed["panel"] and [send["slot_id"] for send in sends] == ["t1"]


def test_auto_and_a_child_compose_a_named_subset_without_a_reason(tmp_path, monkeypatch):
    auto = _floor_ctx(tmp_path / "auto", monkeypatch, mode="auto")
    sends: list[dict] = []
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", _seam(sends))
    named = run_review_change(auto, subject="result", answer="Auto keeps the named row.", reviewers=["t1"])
    assert named["panel"]["assigned"] == ["t1"] and named["panel"]["composition"] == "composed"

    child = _floor_ctx(tmp_path / "child", monkeypatch, root=False)
    kept = run_review_change(child, subject="result", answer="A child keeps the named row.", reviewers=["t1"])
    assert kept["panel"]["assigned"] == ["t1"] and kept["panel"]["composition"] == "composed"
    assert sorted(send["slot_id"] for send in sends) == ["t1", "t1"]


# --- Owner choices of 8 Nov 2026: production, exact material, the author's decision, enforcement ---------------

def _baselined(ctx):
    """The host's own root-start capture (``agent._capture_mutation_baseline`` -> ``capture_root_baseline``)."""
    from ouroboros.mutation_attribution import capture_root_baseline

    return capture_root_baseline(ctx.drive_root, {"id": ctx.task_id}, system_repo=ctx.repo_dir,
                                 workspace=ctx.active_repo_dir(), conversation=False)


def _write(ctx, path, content):
    from ouroboros.tools.registry import ToolRegistry

    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry.set_context(ctx)
    return str(registry.execute("write_file", {"path": path, "content": content}))


def _observed_write(ctx, path, root="active_workspace", *, receipt=True):
    """The loop's own record of a successful native write (what ``reviewable_effect_projection`` reads), with the typed
    receipt its writer publishes of the target its binding resolved (``git_repo_edit._with_path_facts``);
    ``receipt=False`` is a record without it."""
    import pathlib

    meta = {"resolved_root": str(pathlib.Path(ctx.active_repo_dir() if root == "active_workspace" else ctx.repo_dir)
                                 .resolve()), "changed_paths": [path]} if receipt else {}
    trace = ctx._execution_trace = getattr(ctx, "_execution_trace", None) or {"tool_calls": []}
    trace.setdefault("tool_calls", []).append({"tool": "write_file", "args": {"path": path, "root": root},
                                               "is_error": False, "status": "ok", "result": "Written 1 file",
                                               "tool_result_meta": meta})
    return trace


def test_an_external_git_write_reaches_the_floor_with_its_exact_copy_and_none_without_its_baseline(tmp_path, monkeypatch):
    """The CyberGym class (8 Nov 2026): a file written into the admitted external Git folder is production the
    Required floor reviews before delivery, no terminal capture needed; the subject names its exact copy, which a
    later write to the live path never changes. Without the folder's baseline (the defect) nothing was reviewed."""
    import hashlib
    import pathlib

    from ouroboros.tools.review_result import produced_material, result_review_at_delivery

    for baselined in (False, True):
        ctx = _floor_ctx(tmp_path / str(baselined), monkeypatch)
        sends: list[dict] = []
        monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", _seam(sends))
        if baselined:
            assert [row["surface_type"] for row in _baselined(ctx)["baseline"]["surfaces"]] == [
                "external_workspace", "system_repo"]
        _write(ctx, "report.md", "# Findings\n\n1. the first\n2. the second\n3. the third\n")
        trace, shown = {"tool_calls": []}, []
        held = result_review_at_delivery(ctx, "The report is in report.md.", trace, shown.append)
        if not baselined:
            assert held is False and sends == [] and trace["review_decision"]["trigger"] == "no_production"
            continue
        assert held is True and len(sends) == 3 and ": PASS." in shown[0]
        [item] = [file for file in produced_material(ctx)["files"] if file["path"].endswith("report.md")]
        copy = pathlib.Path(item["retained"])
        assert item["kind"] == "workspace change" and copy.read_text(encoding="utf-8").startswith("# Findings")
        assert hashlib.sha256(copy.read_bytes()).hexdigest() == item["sha256"]
        record = review_ledger.load_record(review_ledger.ledger_root(ctx), trace["review_runs"][0]["record_id"])
        from ouroboros.tools.review_result import _retained_subject_text

        subject = _retained_subject_text(ctx, record["subject"]["diff_sha"])
        assert f"exact copy: {copy}" in subject and "1. the first" in subject, "both seat kinds read the bytes"
        (ctx.workspace_root / "report.md").write_text("rewritten later\n", encoding="utf-8")
        assert copy.read_text(encoding="utf-8").startswith("# Findings"), "the reviewed copy never follows the live path"


def test_a_produced_symbolic_link_is_its_own_output_never_its_target(tmp_path, monkeypatch):
    """Astra, 9 Nov 2026: a link the task made is listed by its own path and exact link text (``readlink``), never by
    its target's path or bytes, which making the link did not produce. Two aliases of one target, a broken link and a
    relative one are each their own output; a rename or a retarget is a new subject that selects no earlier review."""
    import hashlib
    import os
    import pathlib

    from ouroboros.tools.review_result import produced_material, result_subject, running_result_review

    ctx = _floor_ctx(tmp_path, monkeypatch)
    _baselined(ctx)
    owner_file = tmp_path / "owner-input.txt"
    owner_file.write_text("AN OWNER FILE THE TASK NEVER PRODUCED\n", encoding="utf-8")
    folder = ctx.workspace_root
    for name, target in (("a.txt", owner_file), ("alias.txt", owner_file), ("broken.txt", folder / "never-made.txt"),
                         ("relative.txt", "alias.txt")):
        (folder / name).symlink_to(target)

    def subject():
        return result_subject(ctx, SimpleNamespace(answer=""), governance_root=ctx.repo_dir, answer="Links made.")

    material = produced_material(ctx)
    links = {pathlib.Path(item["path"]).name: item for item in material["files"]}
    assert set(links) == {"a.txt", "alias.txt", "broken.txt", "relative.txt"}, material
    for name, item in links.items():
        text = os.readlink(folder / name)
        assert item["link"] == json.dumps(text) and "retained" not in item, item
        assert (item["size"], item["sha256"]) == (len(os.fsencode(text)), hashlib.sha256(os.fsencode(text)).hexdigest())
    first = subject()
    assert "AN OWNER FILE" not in json.dumps(material) and "AN OWNER FILE" not in first.diff_text
    assert f'symbolic link, its exact link text {links["a.txt"]["link"]}' in first.diff_text
    reviewing = [{"record_id": "review-of-a", "state": "pending", "subject": {"diff_sha": first.diff_sha}}]
    (folder / "a.txt").rename(folder / "b.txt")
    renamed = subject()
    assert renamed.diff_sha != first.diff_sha and running_result_review(reviewing, renamed.diff_sha) == ""
    (folder / "b.txt").unlink()
    (folder / "b.txt").symlink_to(folder / "another-target.txt")
    assert subject().diff_sha not in {first.diff_sha, renamed.diff_sha}, "a retargeted link is another output"


def test_registered_regular_file_does_not_cover_same_text_symlink(tmp_path):
    """A captured regular file and a later link with the same bytes are different output kinds."""
    from ouroboros.result_material import _file_identity, _same_file_kind

    regular = tmp_path / "result.txt"
    regular.write_text("target.txt", encoding="utf-8")
    link = tmp_path / "link.txt"
    link.symlink_to("target.txt")
    regular_identity = _file_identity(regular)
    link_identity = _file_identity(link)
    assert regular_identity["sha256"] == link_identity["sha256"]
    assert _same_file_kind(regular_identity, link_identity) is False


def test_a_process_write_in_an_unborn_admitted_folder_reaches_the_floor_before_any_commit(tmp_path, monkeypatch):
    """Astra, 8 Nov 2026: a root admitted to a freshly initialized Git folder (no commit yet) whose command writes
    its result. The real admission step baselines that folder against the empty tree beside the system repo, so the
    Required floor reviews the file a process created, with its exact copy. Before, the missing HEAD lost both
    baselines and the delivery read no_production with no review."""
    import pathlib
    import subprocess
    import sys

    from ouroboros.agent import OuroborosAgent
    from ouroboros.task_results import load_task_result
    from ouroboros.tools.registry import ToolRegistry
    from ouroboros.tools.review_result import produced_material, result_review_at_delivery

    ctx = _floor_ctx(tmp_path, monkeypatch)
    fresh = ctx.workspace_root.parent / "fresh"
    fresh.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=fresh, check=True)
    ctx.workspace_root = fresh
    host = SimpleNamespace(env=SimpleNamespace(repo_dir=ctx.repo_dir, budget_drive_root=None, drive_root=ctx.drive_root))
    OuroborosAgent._capture_mutation_baseline(host, {"id": ctx.task_id}, {}, ctx)  # the host's own admission step
    surfaces = load_task_result(ctx.drive_root, ctx.task_id)["mutation_evidence"]["baseline"]["surfaces"]
    assert sorted(row["surface_type"] for row in surfaces) == ["external_workspace", "system_repo"]
    monkeypatch.setattr("ouroboros.safety.check_safety", lambda *a, **k: (True, ""))  # no safety LLM offline
    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry.set_context(ctx)
    registry.execute("run_command", {"cmd": [sys.executable, "-c", "open('first.txt', 'w').write('made by a process\\n')"]})
    assert (fresh / "first.txt").read_text(encoding="utf-8") == "made by a process\n"
    sends: list[dict] = []
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", _seam(sends))
    trace, shown = {"tool_calls": []}, []
    assert result_review_at_delivery(ctx, "The result is in first.txt.", trace, shown.append) is True
    assert trace["review_decision"]["trigger"] == "observed_production" and len(sends) == 3 and ": PASS." in shown[0]
    [item] = [file for file in produced_material(ctx)["files"] if file["path"].endswith("first.txt")]
    assert item["kind"] == "workspace change"
    assert pathlib.Path(item["retained"]).read_text(encoding="utf-8") == "made by a process\n"


def test_an_observed_write_is_production_even_where_attribution_misses_it(tmp_path, monkeypatch):
    """No baseline, then a commit through the shell: Git attribution sees nothing, but the loop observed the native
    write, so the floor still reviews and names the file's current bytes; a read changes nothing."""
    from ouroboros.tools.review_result import produced_material, required_material, result_review_at_delivery

    ctx = _floor_ctx(tmp_path, monkeypatch)
    sends: list[dict] = []
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", _seam(sends))
    trace = ctx._execution_trace = {"tool_calls": [{"tool": "read_file", "args": {"path": "README.md"},
                                                    "is_error": False, "status": "ok", "result": "text"}]}
    assert required_material(ctx, trace, "Read it.") == "", "a read and a named old README are not production"
    _write(ctx, "final.txt", "the produced bytes\n")
    shared.git(ctx.workspace_root, "add", "final.txt")
    shared.git(ctx.workspace_root, "commit", "-qm", "task output")
    trace = _observed_write(ctx, "final.txt")
    produced = produced_material(ctx)
    [item] = [file for file in produced["files"] if file["path"].endswith("final.txt")]
    assert item["kind"] == "written file" and item["retained"] and produced["observed"]
    assert required_material(ctx, trace, "Done.") == "observed_production"
    assert result_review_at_delivery(ctx, "Done.", trace, lambda _text: None) is True and len(sends) == 3


def _loop_call(ctx, tool, args):
    """One model call as the loop runs it (``_execute_single_tool`` -> ``process_tool_results``): the trace row the
    result review reads, with the producer's typed outcome (``tool_result_meta``)."""
    import json
    import pathlib
    from unittest import mock

    from ouroboros import loop_tool_execution as execution
    from ouroboros.tools.registry import ToolRegistry

    registry = ToolRegistry(repo_dir=ctx.repo_dir, drive_root=ctx.drive_root)
    registry.set_context(ctx)
    trace = ctx._execution_trace = getattr(ctx, "_execution_trace", None) or {"tool_calls": []}
    call = {"id": f"call-{len(trace['tool_calls'])}", "function": {"name": tool, "arguments": json.dumps(args)}}
    with mock.patch("ouroboros.safety.check_safety", lambda *_a, **_kw: (True, "")):
        row = execution._execute_single_tool(registry, call, pathlib.Path(ctx.drive_root) / "logs", ctx.task_id)
    execution.process_tool_results([row], [], trace, lambda *_a, **_kw: None, registry)
    return trace["tool_calls"][-1]


def test_rewriting_the_bytes_a_file_already_holds_produces_nothing_and_buys_no_wave(tmp_path, monkeypatch):
    """A same-byte native write (8 Nov 2026 follow-up): write_file, edit_text, apply_patch and edit_batch find the exact
    bytes already there, write nothing and type the call ``completed_no_effect``, so no reader counts it and Required
    buys nothing; a mixed batch lists only its changed member. The controls stay production: changed bytes, and a
    changed write in an ordinary folder whose capture failed."""
    import pathlib

    from ouroboros.tools.registry import ToolContext
    from ouroboros.tools.review_result import produced_material, required_material, result_review_at_delivery

    ctx = _floor_ctx(tmp_path, monkeypatch)
    _baselined(ctx)
    sends: list[dict] = []
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", _seam(sends))
    target = ctx.workspace_root / "app.py"
    before = (target.read_bytes(), target.stat().st_mtime_ns)
    for tool, args in (("write_file", {"path": "app.py", "content": "VALUE = 1\n"}),
                       ("edit_text", {"path": "app.py", "old_str": "VALUE = 1", "new_str": "VALUE = 1"}),
                       ("apply_patch", {"patch": "*** Update File: app.py\n-VALUE = 1\n+VALUE = 1\n"}),
                       ("edit_batch", {"edits": [{"path": "app.py", "old_str": "VALUE = 1", "new_str": "VALUE = 1"}]})):
        row = _loop_call(ctx, tool, args)
        assert row["is_error"] is False and row["tool_result_meta"]["operation_outcome"] == "completed_no_effect", row
        assert row["result"].startswith("✅ Unchanged: active_workspace:app.py already held exactly this content")
    assert (target.read_bytes(), target.stat().st_mtime_ns) == before, "nothing was written"
    trace = ctx._execution_trace
    assert produced_material(ctx)["observed"] == [] and produced_material(ctx)["files"] == []
    assert result_review_at_delivery(ctx, "app.py already holds VALUE = 1.", trace, lambda _t: None) is False
    assert sends == [] and trace["review_decision"]["trigger"] == "no_production"

    changed = _loop_call(ctx, "write_file", {"files": [{"path": "app.py", "content": "VALUE = 2\n"},
                                                      {"path": "lib.py", "content": "LIB = 'base'\n"}]})
    assert changed["tool_result_meta"] == {"resolved_root": str(ctx.workspace_root.resolve()),
                                           "changed_paths": ["app.py"], "unchanged_paths": ["lib.py"]}
    assert not changed["is_error"]
    assert [(item["kind"], pathlib.Path(item["path"]).name) for item in produced_material(ctx)["files"]] == [
        ("workspace change", "app.py")], "the batch's unchanged member is no produced material"
    assert required_material(ctx, trace, "VALUE is 2.") == "observed_production"

    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "notes.txt").write_bytes(b"kept\n")
    folder = ToolContext(repo_dir=ctx.repo_dir, system_repo_dir=ctx.repo_dir, drive_root=ctx.drive_root,
                         workspace_root=plain, workspace_mode="external", task_id="task-plain")
    same = _loop_call(folder, "write_file", {"path": "notes.txt", "content": "kept\n"})
    assert same["tool_result_meta"]["operation_outcome"] == "completed_no_effect" and "Retained" not in same["result"]
    assert required_material(folder, folder._execution_trace, "Kept.") == "", "no new result is captured either"

    def full_disk(*_a, **_kw):
        raise OSError("No space left on device")

    monkeypatch.setattr("ouroboros.artifacts.copy_file_to_task_artifacts", full_disk)
    failed = _loop_call(folder, "write_file", {"path": "notes.txt", "content": "changed\n"})
    assert "OUTPUT_CAPTURE_FAILED" in failed["result"] and "operation_outcome" not in failed["tool_result_meta"]
    produced = produced_material(folder)
    assert produced["observed"] and [item["kind"] for item in produced["files"]] == ["written file"]
    assert required_material(folder, folder._execution_trace, "Changed.") == "observed_production"

    (plain / "crlf.txt").write_bytes(b"kept\r\n")  # one line ending apart from the payload: different bytes, on every OS
    other = _loop_call(folder, "write_file", {"path": "crlf.txt", "content": "kept\n"})
    assert "operation_outcome" not in other["tool_result_meta"] and (plain / "crlf.txt").read_bytes() == b"kept\n"


@pytest.mark.parametrize("letter", ["a", "я"])
def test_large_mixed_write_reports_incomplete_path_facts_without_inventing_changes(tmp_path, monkeypatch, letter):
    """The typed receipt's limit measures serialized bytes. A cut unchanged-path list leaves the changed list whole,
    so nothing unchanged becomes produced; a cut changed-path list names its gap and never guesses the rest, which
    the folder's own attribution still lists."""
    import pathlib

    from ouroboros.tools.review_result import produced_material, required_material

    ctx = _floor_ctx(tmp_path, monkeypatch)
    names = [f"{i:02d}-{letter * 90}.txt" for i in range(50)]
    for name in names:
        (ctx.workspace_root / name).write_bytes(b"kept\n")  # the payload's exact bytes on every OS
    shared.git(ctx.workspace_root, "add", ".")
    shared.git(ctx.workspace_root, "commit", "-qm", "fixture inputs")
    _baselined(ctx)
    row = _loop_call(ctx, "write_file", {"files": [{"path": "app.py", "content": "VALUE = 2\n"}]
                    + [{"path": name, "content": "kept\n"} for name in names]})
    assert row["is_error"] is False, row
    meta = row["tool_result_meta"]
    assert meta["changed_paths"] == ["app.py"] and meta["unchanged_paths_omitted"] > 0
    assert "changed_paths_omitted" not in meta
    material = produced_material(ctx)
    assert [(item["kind"], pathlib.Path(item["path"]).name) for item in material["files"]] == [
        ("workspace change", "app.py")]
    assert not [gap for gap in material["issues"] if "typed receipt" in gap], material["issues"]
    assert required_material(ctx, ctx._execution_trace, "Changed.") == "observed_production"

    many = _loop_call(ctx, "write_file", {"files": [{"path": name, "content": "changed\n"} for name in names]})
    assert many["is_error"] is False, many
    cut = many["tool_result_meta"]
    assert cut["changed_paths_omitted"] > 0 and len(cut["changed_paths"]) + cut["changed_paths_omitted"] == len(names)
    material = produced_material(ctx)
    assert any("more path(s) than its typed receipt could name" in gap for gap in material["issues"]), material["issues"]
    assert sorted(pathlib.Path(item["path"]).name for item in material["files"]) == sorted(["app.py", *names]), \
        "the folder's attribution lists every change; the receipt's cut invents none"


@pytest.mark.parametrize("folder_kind", ["git", "plain"])
@pytest.mark.parametrize("tool", ["write_file", "edit_text", "apply_patch", "edit_batch"])
def test_a_write_through_a_link_lists_the_target_it_wrote_never_the_one_the_link_names_later(
        tmp_path, monkeypatch, tool, folder_kind):
    """Astra, 10 Nov 2026: each native editor writes through ``alias.txt -> written.txt`` to the target its binding
    resolved when it wrote, and its typed receipt names that target. Retargeting the link afterwards to an owner file
    the call never touched never lists that file, its path or its bytes, in the material or in the reviewed subject,
    in an admitted Git folder or an ordinary one; a call record without the receipt names a gap, its arguments are
    never resolved again."""
    import pathlib

    from ouroboros.tools.registry import ToolContext
    from ouroboros.tools.review_result import produced_material, result_subject

    ctx = _floor_ctx(tmp_path, monkeypatch)
    if folder_kind == "plain":
        plain = tmp_path / "plain"
        plain.mkdir()
        ctx = ToolContext(repo_dir=ctx.repo_dir, system_repo_dir=ctx.repo_dir, drive_root=ctx.drive_root,
                          workspace_root=plain, workspace_mode="external", task_id="task-plain")
    folder = pathlib.Path(ctx.workspace_root)
    (folder / "written.txt").write_text("before\n", encoding="utf-8")
    (folder / "untouched.txt").write_text("OWNER BYTES THE CALL NEVER TOUCHED\n", encoding="utf-8")
    (folder / "alias.txt").symlink_to("written.txt")
    if folder_kind == "git":
        shared.git(folder, "add", ".")
        shared.git(folder, "commit", "-qm", "fixture inputs")
        _baselined(ctx)
    args = {"write_file": {"path": "alias.txt", "content": "after\n"},
            "edit_text": {"path": "alias.txt", "old_str": "before", "new_str": "after"},
            "apply_patch": {"patch": "*** Update File: alias.txt\n-before\n+after\n"},
            "edit_batch": {"edits": [{"path": "alias.txt", "old_str": "before", "new_str": "after"}]}}[tool]
    row = _loop_call(ctx, tool, args)
    assert row["is_error"] is False, row
    assert (folder / "written.txt").read_text(encoding="utf-8") == "after\n" and (folder / "alias.txt").is_symlink()
    (folder / "alias.txt").unlink()
    (folder / "alias.txt").symlink_to("untouched.txt")  # retargeted after the write

    def never_the_owner_file(material, subject_text):
        assert not [item for item in material["files"] if pathlib.Path(item["path"]).name == "untouched.txt"], material
        assert "OWNER BYTES" not in json.dumps(material) and "OWNER BYTES" not in subject_text

    material = produced_material(ctx)
    never_the_owner_file(material, result_subject(ctx, SimpleNamespace(answer=""), governance_root=ctx.repo_dir,
                                                  answer="Written.").diff_text)
    assert "written.txt" in {pathlib.Path(item["path"]).name for item in material["files"]}, material
    assert row["tool_result_meta"]["changed_paths"] == ["written.txt"], row["tool_result_meta"]
    assert row["tool_result_meta"]["resolved_root"] == str(folder.resolve())
    _observed_write(ctx, "alias.txt", receipt=False)  # a record without its writer's receipt
    material = produced_material(ctx)
    assert any("carries no receipt of the targets it resolved" in gap for gap in material["issues"]), material
    never_the_owner_file(material, "")


@pytest.mark.parametrize("tool", ["write_file", "edit_text", "apply_patch", "edit_batch"])
def test_partial_native_write_remains_production_when_its_capture_failed(tmp_path, monkeypatch, tool):
    """A disk/capture failure does not erase the writer's known landed paths or make the call successful."""
    from ouroboros import artifacts
    from ouroboros.tools import edit_ops, git
    from ouroboros.tools.registry import ToolContext
    from ouroboros.tools.review_result import produced_material, required_material, result_review_at_delivery

    original = _floor_ctx(tmp_path, monkeypatch)
    folder = tmp_path / "plain"
    folder.mkdir()
    ctx = ToolContext(repo_dir=original.repo_dir, system_repo_dir=original.repo_dir,
                      drive_root=original.drive_root, workspace_root=folder, workspace_mode="external",
                      task_id="partial-native")
    for name in ("first.txt", "second.txt"):
        (folder / name).write_text("before\n", encoding="utf-8")
    copy = artifacts.copy_file_to_task_artifacts

    def capture_unavailable(*args, **kwargs):
        if kwargs.get("kind") == "user_file":
            raise OSError("fixture: artifact disk unavailable")
        return copy(*args, **kwargs)

    monkeypatch.setattr(artifacts, "copy_file_to_task_artifacts", capture_unavailable)
    if tool == "write_file":
        (folder / "second.txt").unlink()
        (folder / "second.txt").mkdir()  # actual second-destination failure, after the first write
    else:
        writer = git if tool == "edit_text" else edit_ops
        write = writer.write_text

        def interrupted_write(path, content):
            if path.name == "second.txt":
                raise OSError("fixture: disk write failed")
            write(path, content)
            if tool == "edit_text":
                raise OSError("fixture: write landed before I/O failure")

        monkeypatch.setattr(writer, "write_text", interrupted_write)
    args = {"write_file": {"files": [{"path": name, "content": "after\n"}
                                     for name in ("first.txt", "second.txt")]},
            "edit_text": {"path": "first.txt", "old_str": "before", "new_str": "after"},
            "apply_patch": {"patch": "*** Update File: first.txt\n-before\n+after\n"
                                      "*** Update File: second.txt\n-before\n+after\n"},
            "edit_batch": {"edits": [{"path": name, "old_str": "before", "new_str": "after"}
                                      for name in ("first.txt", "second.txt")]}}[tool]
    row = _loop_call(ctx, tool, args)
    assert row["is_error"] is True and not row.get("artifact_registered"), row
    assert (folder / "first.txt").read_text(encoding="utf-8") == "after\n"
    material = produced_material(ctx)
    assert material["observed"] and any(item["path"].endswith("first.txt") for item in material["files"]), material
    assert required_material(ctx, ctx._execution_trace, "Partial write retained.") == "observed_production"
    sends = []
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", _seam(sends))
    assert result_review_at_delivery(ctx, "Partial write retained.", ctx._execution_trace, lambda _text: None)
    assert len(sends) == 3


def test_an_answer_its_contract_requires_reviewed_is_reviewed_without_any_file(tmp_path, monkeypatch):
    """``task_contract.review_answer``: Required reviews the whole answer (owner, 8 Nov 2026); without it a plain
    text answer stays the mind's own call, and the record says why nothing ran."""
    from ouroboros.tools.review_result import result_review_at_delivery

    for required in (False, True):
        ctx = _floor_ctx(tmp_path / str(required), monkeypatch)
        ctx.task_contract = {"review_answer": True} if required else {}
        sends: list[dict] = []
        monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", _seam(sends))
        trace = {"tool_calls": []}
        held = result_review_at_delivery(ctx, "The conclusion: X causes Y, for three reasons.", trace, lambda _t: None)
        assert held is required and len(sends) == (3 if required else 0)
        assert trace["review_decision"]["trigger"] == ("answer_required" if required else "no_production")


def test_after_a_received_fail_the_fixed_result_is_the_authors_explicit_acceptance_and_buys_nothing(tmp_path, monkeypatch):
    """Required + Advisory (owner, 8 Nov 2026): A fails, the author fixes it as B. B is not reviewed again: the
    earlier critique is put before the author once, its next finish accepts B with its reason, beside A's verdict.
    That acceptance never passes to C or to B under new owner words."""
    from ouroboros.outcomes import _objective_axis, _review_axis
    from ouroboros.project_dialogue import completion_status_label
    from ouroboros.tools.review_result import result_review_at_delivery

    ctx = _floor_ctx(tmp_path, monkeypatch)
    _produced_answer_protocol(monkeypatch)
    sends: list[dict] = []
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", _seam(sends, {"t1": CRITICAL}))
    trace, shown = {"tool_calls": []}, []
    assert result_review_at_delivery(ctx, "FINAL ANSWER: 2", trace, shown.append) is True  # A's FAIL, first
    assert result_review_at_delivery(ctx, "FINAL ANSWER: 3", trace, shown.append) is True  # B: the choice, once
    assert "An earlier version of your result was reviewed" in shown[-1] and "three findings" in shown[-1]
    ctx._completion_selected = {"action": "finish", "rationale": "Counted again: three findings are listed now."}
    assert result_review_at_delivery(ctx, "FINAL ANSWER: 3", trace, shown.append) is False and len(sends) == 3
    [run] = trace["review_runs"]
    finish = run["informed_finish"]
    assert run["earlier_revision"] is True and finish["subject_sha"] == run["binding_hash"] != run["reviewed_subject_sha"]
    assert finish["rationale"].startswith("Counted again") and finish["outcomes"] and run["aggregate_signal"] == "FAIL"
    review = _review_axis(trace)
    assert (review["status"], review["author_finish"]) == ("fail", "author_finish")
    assert review["acceptance_decision"]["author_disposition"]["rationale"].startswith("Counted again")
    objective = _objective_axis(review)
    assert (objective["status"], objective["source"]) == ("pass", "author_acceptance")
    record = {"status": "completed", "outcome_axes": {"execution": {"status": "ok"}, "review": review, "objective": objective}}
    assert completion_status_label(record, {}) == "Done with warnings"

    assert result_review_at_delivery(ctx, "FINAL ANSWER: 4", trace, shown.append) is True, "C asks again"
    assert "author_finish" not in _review_axis(trace) and len(sends) == 3, "B's acceptance never covers C"
    ctx._owner_directives = [{"source": "owner_mailbox", "content": "State the units too.", "msg_id": "m-9"}]
    assert result_review_at_delivery(ctx, "FINAL ANSWER: 3", trace, shown.append) is True, "new words: asked again"
    assert "author_finish" not in _review_axis(trace) and len(sends) == 3
    assert "State the units too." in shown[-1] and "do they change what was asked?" in shown[-1], "asked, LLM-first"
    assert result_review_at_delivery(ctx, "FINAL ANSWER: 3", trace, shown.append) is False and len(sends) == 3


def test_an_acceptance_never_authorizes_another_candidate_and_a_legacy_one_stays_readable():
    """F1: an informed finish (legacy ``True`` or bound to its subject) counts only on a current run of the subject
    it accepted; a superseded run keeps its critique and never carries its acceptance to the delivered answer."""
    from ouroboros.outcomes import _objective_axis, _review_axis

    def run(**facts):
        return {"surface": "result", "authority": "host_root", "panel_id": "rl-a", "record_id": "rl-a",
                "binding_hash": "subject-a", "aggregate_signal": "FAIL", "aggregate": "FAIL", **facts}

    for finish in (True, {"subject_sha": "subject-a", "record_ids": ["rl-a"], "outcomes": ["x"]}):
        stale = {"review_runs": [run(informed_finish=finish, superseded_by_revision="candidate-2")]}
        review = _review_axis(stale)
        assert review["status"] == "fail" and "author_finish" not in review, "the stale FAIL stands, unaccepted"
        assert _objective_axis(review)["status"] == "fail"
        current = _review_axis({"review_runs": [run(informed_finish=finish)]})
        assert current["author_finish"] == "author_finish", "on its own current subject the acceptance reads"
    misbound = _review_axis({"review_runs": [run(informed_finish={"subject_sha": "subject-b"})]})
    assert "author_finish" not in misbound, "an acceptance of another subject authorizes nothing here"


def test_a_stale_capture_never_hides_a_rewritten_original_and_a_large_file_keeps_its_exact_copy(tmp_path, monkeypatch):
    """A registered capture stands for its original only while the original holds its bytes; a 17 MiB result keeps
    an exact readable copy, never a gap (the existing verified copy, a clone where the filesystem gives one)."""
    import hashlib
    import pathlib

    from ouroboros.artifacts import copy_file_to_task_artifacts
    from ouroboros.tools.review_result import produced_material

    ctx = _floor_ctx(tmp_path, monkeypatch)
    _baselined(ctx)
    original = ctx.workspace_root / "out.txt"
    original.write_text("version A\n", encoding="utf-8")
    copy_file_to_task_artifacts(ctx, original, kind="user_file")

    def listed():
        produced = produced_material(ctx)
        assert not [issue for issue in produced["issues"] if "could not" in issue], produced["issues"]
        return {(pathlib.Path(item["path"]).name, item["kind"]): item for item in produced["files"]}

    (ctx.workspace_root / "other.txt").write_text("a second, distinct result\n", encoding="utf-8")
    _observed_write(ctx, "out.txt")  # the native write that produced the original, as the loop recorded it
    files = listed()
    assert ("out.txt", "workspace change") not in files and ("out.txt", "written file") not in files, \
        "the healthy capture stands for the original, however it is named again"
    assert ("other.txt", "workspace change") in files, "a covered original never stops the attribution after it"
    original.write_text("version B\n", encoding="utf-8")
    files = listed()
    assert pathlib.Path(files[("out.txt", "artifact")]["retained"]).read_text() == "version A\n"
    assert pathlib.Path(files[("out.txt", "workspace change")]["retained"]).read_text() == "version B\n"
    large = ctx.workspace_root / "data.bin"
    large.write_bytes(b"\1" * (17 * 1024 * 1024))
    item = listed()[("data.bin", "workspace change")]
    assert "retained_gap" not in item and hashlib.sha256(pathlib.Path(item["retained"]).read_bytes()).hexdigest() == item["sha256"]
    (ctx.workspace_root / "latin.txt").write_bytes("caf\u00e9 cr\u00e8me\n".encode("latin-1"))
    contents = produced_material(ctx)["contents_section"]
    assert f"### {ctx.workspace_root.resolve() / 'latin.txt'} (not shown: not UTF-8 text" in contents
    assert "caf" not in contents.split("latin.txt", 1)[1].split("###", 1)[0], "no bytes silently dropped as exact"


def test_an_excerpt_says_the_true_cut_and_files_past_the_spent_budget_are_named_unread(tmp_path):
    """Opus, 8 Nov 2026: a text file with no line break in its read (one-line JSON, minified code), only CR breaks,
    or a first line longer than its share shows a bounded prefix cut on a character boundary and says so, never
    "budget is spent" with the budget unused. Once the budget is spent, the remaining files are named without being
    read; a file that is not UTF-8 stays a named gap."""
    from ouroboros.result_material import _contents_section

    def retained(name, data, live=True):
        path = tmp_path / name
        if live:
            path.write_bytes(data)
        return {"path": f"/work/{name}", "retained": str(path), "size": len(data)}

    one_line = retained("one.json", ('{"key": "' + "é" * 40000 + '"}').encode("utf-8"))
    cr_only = retained("cr.txt", b"first\rsecond\r" + b"x" * 70000)
    whole = retained("whole.txt", b"alpha\nbeta\n")
    latin = retained("latin.txt", "café\n".encode("latin-1") * 10)
    section = _contents_section([one_line, cr_only, whole, latin], 6000)
    assert "budget is spent" not in section, section[:400]
    head, rest = section.split("### /work/one.json (its first 1500 characters shown; the rest is in its retained copy)\n")
    assert rest.split("\n", 2)[1] == ('{"key": "' + "é" * 40000)[:1500], "an exact prefix on a character boundary"
    assert "### /work/cr.txt (its first 2 lines shown; the rest is in its retained copy)" in section
    assert "### /work/whole.txt (the whole file shown)" in section
    assert "### /work/latin.txt (not shown: not UTF-8 text; it is in its retained copy)" in section

    lines = retained("lines.txt", ("z" * 99 + "\n").encode("utf-8") * 200)
    ghost = retained("never-read.txt", b"0123456789", live=False)  # a read would fail and drop it unnamed
    spent = _contents_section([lines, ghost], 1000)
    assert "### /work/lines.txt (its first 10 lines shown; the rest is in its retained copy)" in spent
    assert spent.endswith("Not shown: this packet's excerpt budget is spent; each is in its retained copy above: "
                          "/work/never-read.txt")


def test_every_delivery_says_why_its_result_was_not_reviewed(tmp_path, monkeypatch):
    from ouroboros.tools.review_result import result_review_at_delivery

    for mode, trigger in (("off", "mode_off"), ("auto", "not_requested"), ("required", "no_production")):
        ctx = _floor_ctx(tmp_path / mode, monkeypatch, mode=mode)
        trace = {"tool_calls": []}
        assert result_review_at_delivery(ctx, "A short reply.", trace, lambda _text: None) is False
        assert trace["review_decision"] == {"mode": mode, "trigger": trigger}
        from ouroboros.outcomes import _review_axis
        assert _review_axis(trace)["eligibility"] == "not_eligible"
        assert _review_axis(trace)["mode"] == mode


@pytest.mark.parametrize("late, warning", [("PASS", ""), ("PASS", "receipt_absent"), ("FAIL", "receipt_absent")])
def test_a_reasoned_pending_finish_warns_until_a_late_pass_of_its_exact_subject_resolves_it(tmp_path, monkeypatch, late,
                                                                                           warning):
    """Required + Advisory (owner, 8 Nov 2026): the author may finish while the floor's review still runs, with its
    reason; the task is Done with warnings and the review stays owed. When that review settles after the task ended,
    the real late path keeps its mark, and only a PASS of exactly the delivered subject clears the warning it caused.
    A warning the review never answered (Astra, 8 Nov: ``receipt_absent``) stays in the record; nothing examined the
    result is a detail, so after that PASS the headline is Done (owner, 9 Nov 2026)."""
    from ouroboros.chronicle_store import ChronicleStore, is_late_review_mark
    from ouroboros.late_review_marks import replay_owed_late_reviews
    from ouroboros.outcomes import _merge_objective_warning, _objective_axis, _review_axis
    from ouroboros.project_dialogue import completion_status_label, outcome_phase
    from ouroboros.task_results import load_task_result, write_task_result
    from ouroboros.tools import review_result as rr
    from tests.test_review_change_end_to_end import _pending_then_answering_seat

    ctx = _floor_ctx(tmp_path, monkeypatch)
    _produced_answer_protocol(monkeypatch)
    pending, _starts = _pending_then_answering_seat("t2", "invocation-t2")

    def answer(request, slot, actor, *, retry_state, pending_invocation_checkpoint):
        actor = pending(request, slot, actor, retry_state=retry_state, pending_invocation_checkpoint=pending_invocation_checkpoint)
        if late == "FAIL" and slot.slot_id == "t2" and retry_state.get("pending_invocation_id") == "invocation-t2":
            actor.raw_text = CRITICAL  # the late answer of the seat that was still running
        return actor

    sends: list[dict] = []
    monkeypatch.setattr(substrate.ReviewCoordinator, "_run_slot", shared.golden_physical_seam(sends, answer=answer))
    trace, shown = {"tool_calls": []}, []
    assert rr.result_review_at_delivery(ctx, "FINAL ANSWER: 2", trace, shown.append) is True
    assert "still running" in shown[-1] and "pending_review=finish and a rationale" in shown[-1]
    assert '"seat_id": "t1"' not in shown[-1], "the seats that answered were not handed back by that hold"
    for decision in ("first", "after reading"):
        ctx._acceptance_pending_review_choice = "finish"
        ctx._completion_selected = {"action": "finish", "pending_review": "finish",
                                    "rationale": "The owner asked for the number now; the review's verdict will follow."}
        if decision == "first":  # t1 and s1 answered and t2 still runs: their answers reach the author first
            assert rr.result_review_at_delivery(ctx, "FINAL ANSWER: 2", trace, shown.append) is True
            assert "1 still answering" in shown[-1] and all(f'"seat_id": "{seat}"' in shown[-1] for seat in ("t1", "s1"))
            assert '"seat_id": "t2"' not in shown[-1] and ctx._acceptance_pending_review_choice == "" and len(sends) == 3
    assert rr.result_review_at_delivery(ctx, "FINAL ANSWER: 2", trace, shown.append) is False, "finished beside it"
    assert len(sends) == 3 and trace["review_decision"]["trigger"] == "pending_finish", "no new paid panel"
    review = _review_axis(trace)
    assert (review["status"], review["author_finish"]) == ("awaiting", "author_finish_pending")
    assert review["acceptance_decision"]["author_disposition"]["rationale"].startswith("The owner asked")
    record_id = review["pending_review"]["record_id"]
    axes = {"execution": {"status": "ok"}, "review": review, "objective": _objective_axis(review)}
    if warning:  # what ``apply_receipt_absent_flag`` adds at the end of a turn with effects and no verification
        _merge_objective_warning(axes["objective"], warning)
    assert completion_status_label({"status": "completed", "outcome_axes": axes}, {}) == "Done with warnings"

    # The running seat answers later; then the task has ended and the owed duty is replayed.
    subject = rr.result_subject(ctx, SimpleNamespace(answer=""), governance_root=ctx.repo_dir, answer="FINAL ANSWER: 2")
    assert rr._collect_running(ctx, "FINAL ANSWER: 2", record_id, subject)["state"] == "settled"
    write_task_result(ctx.drive_root, ctx.task_id, "completed", chat_id=1, outcome_axes=axes)
    assert replay_owed_late_reviews(ctx.drive_root)["settled"] == 1
    marks = [mark for mark in ChronicleStore(ctx.drive_root).active_marks(None) if is_late_review_mark(mark)]
    assert marks, "the late verdict reaches its room either way"
    stored = load_task_result(ctx.drive_root, ctx.task_id)
    objective = stored["outcome_axes"]["objective"]
    assert objective.get("warnings", []) == ([warning] if warning else []), "an independent warning is never cleared"
    events = (ctx.drive_root / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()
    announced = [row for row in map(json.loads, events) if row.get("type") == "review_late_result" and row.get("current_outcome")]
    if late == "PASS":  # the open card reads what this reload reads; a FAIL announces no outcome and keeps the warning
        [event] = announced
        assert (event["task_id"], event["record_id"]) == (ctx.task_id, record_id)
        assert event["current_outcome"]["outcome_phase"] == outcome_phase(stored, {}) == "done"
    else:
        assert announced == []
    if late == "PASS":
        assert completion_status_label(stored, {}) == "Done", \
            "a PASS of exactly the delivered subject resolves the warning it caused; a detail stays a detail"
        resolved = stored["outcome_axes"]["review"]
        assert resolved["status"] == "pass" and resolved["late_verdict"]["record_id"] == record_id
        assert resolved["pending_finish_resolved"]["review"]["author_finish"] == "author_finish_pending"
        assert (objective["status"], objective["source"], objective["review_status"]) == ("pass", "result_review", "pass")
        assert "reason" not in objective, "the author's pending acceptance is history now, kept beside the PASS"
    else:
        assert completion_status_label(stored, {}) == "Done with warnings", "a late FAIL keeps the warning"
        assert stored["outcome_axes"]["review"]["status"] == "awaiting"
