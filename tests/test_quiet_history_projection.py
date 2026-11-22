"""Quiet terminal facts survive the browser history boundary and result retention."""
import json
from pathlib import Path

import pytest

from ouroboros import consciousness_view, dialogue_store, usage_accounting
from ouroboros.gateway.history import _assemble_history_response, _copy_task_summary_metadata
from ouroboros.projects_registry import bind_task_to_project, create_project
from ouroboros.task_results import load_task_result, write_task_result
from ouroboros.task_store import delete_document
from ouroboros.terminal_projection import _project_row
from tests._store_shared import seed_result
from tests._wake_summary_shared import seed_wake
from tests.test_chat_history_paging import isolated_runtime  # noqa: F401


@pytest.mark.parametrize(("retained", "silent", "notice"), [
    pytest.param(True, True, "", id='result-present-quiet'),
    pytest.param(True, False, "", id='result-present-ordinary'),
    pytest.param(False, True, "A tool failed.", id='result-pruned-quiet-with-notice'),
])
def test_terminal_flags_round_trip_through_room_history(tmp_path, silent, notice, retained):
    project = create_project(tmp_path, "checks", name="Scheduled checks")
    task_id = "scheduled-check"
    bind_task_to_project(tmp_path, task_id, project["id"], project["chat_id"], origin={"absent": "system"})
    stored = write_task_result(tmp_path, task_id, "completed", chat_id=project["chat_id"],
                               project_id=project["id"], root_task_id=task_id, delegation_role="root",
                               result="Private retained words", terminal_origin="model_final",
                               wake_silent=silent, terminal_host_notice=notice)
    terminal = _project_row(task_id, stored, {}, {"chat_id": project["chat_id"]})
    dialogue_store.append(tmp_path, "chat", terminal)
    if not retained:
        assert delete_document(tmp_path, task_id)
    response = json.loads(_assemble_history_response(tmp_path, project["chat_id"], 50, 200))
    rows = [row for row in response["messages"] if row.get("task_id") == task_id]
    summary = next(row for row in rows if row.get("system_type") == "task_summary"
                   and "wake_silent" in row)
    expected = {"wake_silent": silent, "quiet_effective": silent and not notice}
    assert {key: summary[key] for key in expected} == expected
    if expected["quiet_effective"]:
        assert summary["text"] == "Done · no message"
    else:
        assert summary["text"] != "Done · no message"
    if not retained:
        historical = next(row["historical_terminal"] for row in rows if "historical_terminal" in row)
        assert {key: historical[key] for key in expected} == expected
    assert "Private retained words" not in json.dumps(response)


@pytest.mark.parametrize("with_progress", [False, True], ids=["terminal-only", "with-progress"])
@pytest.mark.parametrize("reopened", [False, True], ids=["legacy-attempt", "later-attempt"])
def test_legacy_schedule_report_reference_survives_project_history(tmp_path, with_progress, reopened):
    from tests.test_schedule_report_door import ceiling, run_metadata

    project = create_project(tmp_path, "checks", name="Scheduled checks")
    chat_id, task_id = project["chat_id"], "legacy-schedule-report"
    bind_task_to_project(tmp_path, task_id, project["id"], chat_id, origin={"absent": "system"})
    report = "Private reference report."
    current_report = "Private retried report." if reopened else report
    stored = write_task_result(tmp_path, task_id, "completed", chat_id=chat_id, project_id=project["id"],
                               result=current_report, wake_summary=current_report, terminal_origin="model_final",
                               **({"task_attempt": 2} if reopened else {}),
                               metadata=run_metadata(task_id), task_contract={"capability_ceiling": ceiling()})
    # A publication from the earlier attempt's own snapshot can arrive after the next attempt ended.
    earlier = {**stored, "task_attempt": 1, "result": report, "wake_summary": report} if reopened else stored
    terminal = _project_row(task_id, earlier, {}, {"chat_id": chat_id})
    if with_progress:
        dialogue_store.append(tmp_path, "progress", {"task_id": task_id, "chat_id": chat_id, "ts": terminal["ts"],
                                                     "role": "assistant", "content": "Checking sample entries."})
    dialogue_store.append(tmp_path, "chat", terminal)
    response = json.loads(_assemble_history_response(tmp_path, chat_id, 50, 200, around=f"task:{task_id}"))
    rows = [row for row in response["messages"] if row.get("task_id") == task_id]
    assert any(row.get("is_progress") for row in rows) is with_progress
    [summary] = [row for row in rows if row.get("system_type") == "task_summary" and row.get("text")]
    assert summary["schedule_report_only"] is True
    assert summary["quiet_effective"] is True and summary["text"] == "Done · no message"
    assert report not in json.dumps(response)
    if reopened:
        current = summary["current_outcome"]
        assert summary["task_attempt"] == 1 and current["task_attempt"] == 2
        assert current["schedule_report_only"] is True and current["quiet_effective"] is True
        assert "Private retried report." not in json.dumps(response)


@pytest.mark.parametrize("flags", [{}, {"wake_silent": True}, {"wake_silent": False, "quiet_effective": False},
                                   {"schedule_report_only": False}, {"schedule_report_only": "true"}])
def test_legacy_summary_flags_stay_present_or_absent_as_recorded(tmp_path, flags):
    entry = {"type": "task_summary", "direction": "system", "chat_id": 1,
             "task_id": "legacy", "text": "Recorded terminal", "ts": "2026-10-21T12:00:00Z", **flags}
    dialogue_store.append(tmp_path, "chat", entry)
    rows = json.loads(_assemble_history_response(tmp_path, 1, 50, 200))["messages"]
    summary = next(row for row in rows if row.get("system_type") == "task_summary")
    assert {key: summary[key] for key in ("wake_silent", "quiet_effective", "schedule_report_only") if key in summary} == {
        key: value for key, value in flags.items() if type(value) is bool
    }


def test_non_summary_rows_do_not_acquire_terminal_flags():
    projected = {}
    _copy_task_summary_metadata(projected, {"type": "message", "wake_silent": True, "quiet_effective": True,
                                           "schedule_report_only": True})
    assert projected == {}


@pytest.mark.parametrize("silent,notice", [(True, ""), (True, "A tool failed."), (False, "")])
def test_stored_wake_and_history_share_phase_and_quiet_facts(tmp_path, silent, notice):
    fixture = Path(__file__).parent / "fixtures" / "wake_document_214a31b6.json"
    document = json.loads(fixture.read_text(encoding="utf-8"))
    document.update(wake_silent=silent, terminal_host_notice=notice)
    task_id = document["task_id"]
    stored = seed_result(tmp_path, task_id, document)
    usage_accounting.record_unmetered_external_dispatch(
        task_id, drive_root=tmp_path, task_id=task_id, root_task_id=task_id, category="consciousness")
    terminal = _project_row(task_id, stored, {}, {"chat_id": stored["chat_id"]})
    dialogue_store.append(tmp_path, "chat", terminal)
    history = json.loads(_assemble_history_response(tmp_path, stored["chat_id"], 50, 200))
    summary = next(row["historical_terminal"] for row in history["messages"] if "historical_terminal" in row)
    wake, = consciousness_view.recent_wakes(tmp_path, [])
    assert wake["quiet"] is (silent and not bool(notice))
    assert wake["quiet"] == summary["quiet_effective"]
    assert wake["phase"] == summary["phase"] == "done"
    assert wake["ended_at"] == terminal["terminal_time"]["occurred_at"]
    assert wake["note"] == document["result"]


@pytest.mark.parametrize(("owner_sent", "summary", "retained"), [
    (True, "Проверил состояние системы.", True),
    (False, "", False),
])
def test_wake_words_and_quiet_truth_survive_terminal_only_history(tmp_path, retained, summary, owner_sent):
    fields = {"owner_message_sent": owner_sent, "schedule_name": "Daily check"}
    if summary:
        fields.update(wake_summary=summary, suggested_name=summary)
    stored = seed_wake(tmp_path, **fields)
    tid = stored["task_id"]
    terminal = _project_row(tid, stored, {}, {"chat_id": 1})
    dialogue_store.append(tmp_path, "chat", terminal)
    if not retained:
        assert delete_document(tmp_path, tid)
    rows = json.loads(_assemble_history_response(tmp_path, 1, 50, 200))["messages"]
    projected = next(row for row in rows if row.get("system_type") == "task_summary" and "wake_silent" in row)
    assert projected["quiet_effective"] is (not owner_sent)
    assert projected.get("wake_summary", "") == summary
    assert projected["suggested_name"] == stored["suggested_name"] and projected["schedule_name"] == "Daily check"
    assert stored["result"] not in json.dumps(rows, ensure_ascii=False)


@pytest.mark.parametrize("kind", ["task_summary", "message"])
def test_terminal_copy_carries_both_authored_title_fields_only_on_summaries(kind):
    words = {"suggested_name": "Title", "wake_summary": "The mind's line.", "schedule_name": "Daily check"}
    projected = {}
    _copy_task_summary_metadata(projected, {"type": kind, **words})
    assert {key: projected[key] for key in words if key in projected} == (words if kind == "task_summary" else {})


@pytest.mark.parametrize("report_only", [False, True], ids=["ordinary-schedule", "legacy-schedule-report"])
def test_a_page_without_the_terminal_row_carries_the_runs_origin_without_a_private_report(tmp_path, isolated_runtime, report_only):  # noqa: F811
    """A long run's terminal row can lie beyond the page a work-card quote opens: the retained result's terminal
    truth, stamped on the window's last progress row, carries the authored name, the sentence and the schedule."""
    from datetime import datetime, timedelta, timezone
    from tests.test_schedule_report_door import ceiling, run_metadata

    tid, start = "scheduled-long-check", datetime(2026, 10, 28, tzinfo=timezone.utc)
    at = lambda seconds: (start + timedelta(seconds=seconds)).isoformat()  # noqa: E731
    for index in range(240):
        dialogue_store.append(tmp_path, "progress", {"task_id": tid, "chat_id": 1, "ts": at(index * 120),
                                                     "role": "assistant", "content": f"Checking item {index}"})
        if index % 3 == 0:
            dialogue_store.append(tmp_path, "chat", {"direction": "in", "chat_id": 1, "ts": at(index * 120 + 1),
                                                     "text": f"Other conversation {index}"})
    metadata = run_metadata(tid) if report_only else {"schedule_occurrence": {"schedule_id": "daily", "token": "t1"}}
    stored = write_task_result(tmp_path, tid, "completed", chat_id=1, root_task_id=tid, delegation_role="root",
                               title="Daily check", suggested_name="Daily check", schedule_name="Daily check",
                               metadata=metadata, **({"task_contract": {"capability_ceiling": ceiling()}} if report_only else {}),
                               wake_summary="The run found no faults.", wake_silent=True, result="PRIVATE NOTE",
                               terminal_origin="model_final", _terminal_observed=True)
    terminal = _project_row(tid, stored, {}, {"chat_id": 1})
    terminal["ts"] = at(30000)
    dialogue_store.append(tmp_path, "chat", terminal)
    response = json.loads(_assemble_history_response(tmp_path, 1, 50, 200, around=f"task:{tid}"))
    rows = [row for row in response["messages"] if row.get("task_id") == tid]
    assert rows and not any(row.get("summary_kind") or row.get("historical_terminal") for row in rows), "the terminal row lies beyond this page"
    settled = [row for row in rows if row.get("task_terminal_status") == "completed"]
    assert settled and all(row["suggested_name"] == "Daily check" for row in settled)
    assert any(row.get("schedule_name") == "Daily check" for row in settled)
    if report_only:
        assert "The run found no faults." not in json.dumps(response)
    else:
        assert any(row.get("wake_summary") == "The run found no faults." for row in settled)
    assert "PRIVATE NOTE" not in json.dumps(response, ensure_ascii=False)


def test_terminal_row_keeps_the_custody_fact_its_phase_uses(tmp_path):
    """The phase and its browser twin must receive the same current debt, including a known empty list."""
    for debt in (["run-x"], []):
        row = {"task_id": "debt-task", "status": "completed", "reason_code": "final_message",
               "delegated_runs_unreconciled": debt, "outcome_axes": {
                   "execution": {"status": "ok"}, "review": {"status": "pass"},
                   "objective": {"status": "pass", "source": "result_review"}}}
        projected = _project_row("debt-task", row, {}, {"chat_id": 1})
        assert projected["outcome_phase"] == ("warn" if debt else "done")
        assert projected.get("delegated_runs_unreconciled") == debt


@pytest.mark.parametrize(("verdict", "with_progress"), [("PASS", True), ("PASS", False), ("FAIL", False)])
def test_a_late_pass_gives_the_ending_row_the_current_outcome_beside_its_recorded_phase(tmp_path, monkeypatch, verdict, with_progress):
    """The ending row stays the observation it recorded (Done with warnings: Ouroboros finished before its reviewers
    answered). When the late review then PASSes exactly the delivered answer, the same history read also carries the
    task's current outcome (``late_review_marks.late_current_outcome``, the one the live event carries), so a reload
    reads what the open card shows; a late FAIL adds nothing."""
    from ouroboros import late_review_marks, review_ledger
    from ouroboros.late_review_marks import resolve_pending_result_review
    from ouroboros.outcomes import _objective_axis

    review = {"status": "awaiting", "source": "result_review", "author_finish": "author_finish_pending",
              "pending_review": {"record_id": "rec-1", "subject_sha": "subject-a"},
              "acceptance_decision": {"status": "finalized_unaccepted", "source": "result_review",
                                      "reason": "author_finish_pending", "enforcement": "advisory"}}
    stored = write_task_result(tmp_path, "late-task", "completed", chat_id=1, root_task_id="late-task",
                               delegation_role="root", result="The answer.", terminal_origin="model_final",
                               reason_code="final_message", outcome_axes={
                                   "execution": {"status": "ok"}, "review": review, "objective": _objective_axis(review)})
    ending = _project_row("late-task", stored, {}, {"chat_id": 1})
    if with_progress:
        dialogue_store.append(tmp_path, "progress", {"task_id": "late-task", "chat_id": 1, "content": "Checking",
                                                     "ts": ending["ts"], "narration": True})
    dialogue_store.append(tmp_path, "chat", ending)
    monkeypatch.setattr(review_ledger, "load_record", lambda _root, record_id: {
        "record_id": record_id, "subject": {"diff_sha": "subject-a"}, "verdict": {"aggregate": verdict}})
    assert resolve_pending_result_review(tmp_path, "late-task", "rec-1") is (verdict == "PASS")
    reads = []
    lookup = late_review_marks.late_current_outcome

    def counted(root, stored):
        reads.append(stored["task_id"])
        return lookup(root, stored)

    monkeypatch.setattr(late_review_marks, "late_current_outcome", counted)
    response = json.loads(_assemble_history_response(tmp_path, 1, 50, 200))
    assert reads.count("late-task") == 1
    [evidence] = [row for row in response["messages"] if row.get("task_id") == "late-task" and row.get("historical_terminal")]
    if not with_progress:
        assert [row for row in response["messages"] if row.get("task_id") == "late-task"] == [evidence]
    assert evidence["historical_terminal"]["phase"] == "warn", "the ending row stays as it was recorded"
    if verdict == "PASS":
        current, row = evidence["current_outcome"], load_task_result(tmp_path, "late-task")
        assert current["outcome_phase"] == "done" and current["_result_text_present"] is True
        assert current["outcome_axes"]["review"] == {"status": "pass", "source": "result_review", "aggregate_signals": ["PASS"]}
        assert {key: value for key, value in row["outcome_axes"]["review"].items()
                if key not in {"pending_finish_resolved", "late_verdict"}} == current["outcome_axes"]["review"]
        assert "The answer." not in json.dumps(current), "the card's facts, never the answer"
    else:
        assert "current_outcome" not in evidence


@pytest.mark.parametrize("shape", ["display_partial", "broken_digest"])
def test_partial_or_invalid_frozen_ceiling_refuses_adapter_yet_suppresses_report_disclosure(tmp_path, shape):
    """Strict permission requires a valid frozen ceiling + receipt to expose ``presence_finish``; conservative
    disclosure treats any frozen-ceiling legacy schedule shape as a private report in full & projected history
    and schedule previews so ``DISPLAY_KEPT`` or damaged ceilings never leak saved words."""
    from ouroboros.task_finalization import schedule_report_run
    from ouroboros.tools.registry import ToolRegistry
    from supervisor.queue_schedules import _scan_outcomes
    from supervisor.terminal_delivery import build_completed_result_event, build_unreviewed_salvage_event
    from tests.test_schedule_report_door import SCHEDULE, admit, bind, ceiling, run_metadata

    task_id = f"partial-ceiling-{shape}"
    admit(tmp_path, task_id)
    bad_ceiling = {"schema_version": 1} if shape == "display_partial" else {**ceiling(), "digest": "e" * 64}
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    ctx = bind(registry, task_id=task_id, metadata=run_metadata(task_id),
               contract={"capability_ceiling": bad_ceiling})
    assert "presence_finish" not in registry._entries and getattr(ctx, "_schedule_door", None) is None

    project = create_project(tmp_path, f"proj-{shape}", name="Scheduled checks")
    chat_id = project["chat_id"]
    bind_task_to_project(tmp_path, task_id, project["id"], chat_id, origin={"absent": "system"})
    secret, summary_line = "PRIVATE FROZEN REPORT", "Private summary sentence."
    stored = write_task_result(tmp_path, task_id, "completed", chat_id=chat_id, project_id=project["id"],
                               root_task_id=task_id, delegation_role="root", result=secret, wake_summary=summary_line,
                               wake_silent=True, terminal_origin="model_final", metadata=run_metadata(task_id),
                               task_contract={"capability_ceiling": bad_ceiling})
    full = load_task_result(tmp_path, task_id)
    assert schedule_report_run(full) is True
    completed_event = build_completed_result_event(tmp_path, {"id": task_id, "chat_id": chat_id}, task_id, full,
                                                   unreconciled_runs=["run-unsettled"])
    assert completed_event is not None and secret not in completed_event["text"] and summary_line not in completed_event["text"]
    salvage_event = build_unreviewed_salvage_event(tmp_path, {"task_id": task_id, "chat_id": chat_id}, task_id,
                                                   outcome="cancelled", salvaged_text=secret, settled_status="cancelled")
    assert secret not in salvage_event["text"]

    terminal = _project_row(task_id, stored, {}, {"chat_id": chat_id})
    dialogue_store.append(tmp_path, "chat", terminal)
    response = json.loads(_assemble_history_response(tmp_path, chat_id, 50, 200))
    serialized = json.dumps(response)
    assert secret not in serialized and summary_line not in serialized
    [row] = [r for r in response["messages"] if r.get("task_id") == task_id and r.get("system_type") == "task_summary"]
    assert row["schedule_report_only"] is True and row["quiet_effective"] is True and row["text"] == "Done · no message"

    preview = _scan_outcomes(tmp_path, {"tasks": [{"id": SCHEDULE, "last_task_id": task_id}]})[SCHEDULE]
    assert preview["report_only"] is True and preview["line"] == "" and preview["quiet"] is True
