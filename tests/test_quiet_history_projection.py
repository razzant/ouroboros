"""Quiet terminal facts survive the browser history boundary and result retention."""
import json
from pathlib import Path

import pytest

from ouroboros import consciousness_view, dialogue_store, usage_accounting
from ouroboros.gateway.history import _assemble_history_response, _copy_task_summary_metadata
from ouroboros.projects_registry import bind_task_to_project, create_project
from ouroboros.task_results import write_task_result
from ouroboros.task_store import delete_document
from ouroboros.terminal_projection import _project_row
from tests._store_shared import seed_result
from tests._wake_summary_shared import seed_wake
from tests.test_chat_history_paging import isolated_runtime  # noqa: F401


@pytest.mark.parametrize("silent,notice", [(True, ""), (True, "A tool failed."), (False, "")],
                         ids=["quiet", "quiet-with-notice", "ordinary"])
@pytest.mark.parametrize("retained", [True, False], ids=["result-present", "result-pruned"])
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


@pytest.mark.parametrize("flags", [{}, {"wake_silent": True}, {"quiet_effective": False},
                                  {"wake_silent": False, "quiet_effective": False}])
def test_legacy_summary_flags_stay_present_or_absent_as_recorded(tmp_path, flags):
    entry = {"type": "task_summary", "direction": "system", "chat_id": 1,
             "task_id": "legacy", "text": "Recorded terminal", "ts": "2026-10-21T12:00:00Z", **flags}
    dialogue_store.append(tmp_path, "chat", entry)
    rows = json.loads(_assemble_history_response(tmp_path, 1, 50, 200))["messages"]
    summary = next(row for row in rows if row.get("system_type") == "task_summary")
    assert {key: summary[key] for key in ("wake_silent", "quiet_effective") if key in summary} == flags


def test_non_summary_rows_do_not_acquire_terminal_flags():
    projected = {}
    _copy_task_summary_metadata(projected, {"type": "message", "wake_silent": True, "quiet_effective": True})
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


@pytest.mark.parametrize("retained", [True, False])
@pytest.mark.parametrize("summary", ["Проверил состояние системы.", ""])
@pytest.mark.parametrize("owner_sent", [True, False])
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


def test_a_page_without_the_terminal_row_still_carries_the_runs_sentence_and_origin(tmp_path, isolated_runtime):  # noqa: F811
    """A long run's terminal row can lie beyond the page a work-card quote opens: the retained result's terminal
    truth, stamped on the window's last progress row, carries the authored name, the sentence and the schedule."""
    from datetime import datetime, timedelta, timezone

    tid, start = "scheduled-long-check", datetime(2026, 10, 28, tzinfo=timezone.utc)
    at = lambda seconds: (start + timedelta(seconds=seconds)).isoformat()  # noqa: E731
    for index in range(240):
        dialogue_store.append(tmp_path, "progress", {"task_id": tid, "chat_id": 1, "ts": at(index * 120),
                                                     "role": "assistant", "content": f"Checking item {index}"})
        if index % 3 == 0:
            dialogue_store.append(tmp_path, "chat", {"direction": "in", "chat_id": 1, "ts": at(index * 120 + 1),
                                                     "text": f"Other conversation {index}"})
    stored = write_task_result(tmp_path, tid, "completed", chat_id=1, root_task_id=tid, delegation_role="root",
                               title="Daily check", suggested_name="Daily check", schedule_name="Daily check",
                               metadata={"schedule_occurrence": {"schedule_id": "daily", "token": "t1"}},
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
    assert any(row.get("wake_summary") == "The run found no faults." and row.get("schedule_name") == "Daily check" for row in settled)
    assert "PRIVATE NOTE" not in json.dumps(response, ensure_ascii=False)
