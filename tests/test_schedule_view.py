"""Shared schedule projections read host facts; they never invent liveness."""
import copy
import datetime
import json

import pytest

from supervisor.queue_schedules import schedule_activity_projection, schedule_tool_projection
from supervisor.schedule_view import decorate, digest

NOW = datetime.datetime(2026, 10, 21, 12, tzinfo=datetime.timezone.utc)


def row(sid="s1", **kw):
    return {"id": sid, "name": "My promise", "trigger": {"type": "once", "run_at": "2099-01-01T00:00:00+00:00"},
            "task": {"text": "Check the report\nDetails", "chat_id": 42, "project_id": "project"}, **kw}


@pytest.mark.parametrize("name,expected", [
    ("Follow-up of task abc", "Check the report"), ("Reminder of task abc", "Check the report"),
    ("My report check", "My report check"), ("Follow-up of task abc extra", "Follow-up of task abc extra"),
])
def test_only_legacy_generated_names_are_displayed_as_the_objective(name, expected):
    original = row(name=name)
    before = copy.deepcopy(original)
    view = decorate(original, NOW)
    assert view["display_name"] == expected and view["name"] == name
    assert original == before
    # A blank zone stays blank for consumers that name it (Activity: the server zone).
    assert view["kind"] == "task" and view["timezone"] == ""
    assert view["destination"] == {"chat_id": 42, "project_id": "project"}
    assert "outcome" not in view and "previous_run_open" not in view


def test_note_name_uses_written_words_and_preserves_owner_names():
    note = row(name="Reminder of task abc", kind="notify", notification={"text": "n" * 120 + "\nMore"})
    assert decorate(note, NOW)["display_name"] == "n" * 96
    assert decorate(note, NOW)["kind"] == "note"
    note["name"] = "Owner's name"
    assert decorate(note, NOW)["display_name"] == "Owner's name"


@pytest.mark.parametrize("matching", [True, False])
def test_cache_and_liveness_facts_are_distinct(matching):
    saved = {"task_id": "last" if matching else "older", "state": "completed", "quiet": True, "line": ""}
    original = row(last_task_id="last", last_outcome=saved)
    projected = decorate(original, NOW)
    assert projected["outcome"] == (saved if matching else {"task_id": "last", "state": "pending_observation"})
    assert "previous_run_open" not in projected
    original["previous_run_open"] = {"task_id": "last", "since": "earlier"}
    assert decorate(original, NOW)["previous_run_open"] == original["previous_run_open"]


@pytest.mark.parametrize("facts,expected", [
    ({}, None), ({"last_task_id": "last"}, None),
    ({"last_task_id": "last", "last_outcome": {"task_id": "older", "state": "completed"}}, None),
    ({"last_task_id": "last", "last_outcome": {"task_id": "last", "state": "completed"}}, False),
    ({"occurrence": {"phase": "claimed"}}, True),
    ({"previous_run_open": {"task_id": "last", "since": "earlier"}}, True),
])
def test_withdrawal_reports_only_known_work_facts(facts, expected):
    original = row(delete_requested_at="then", delete_requested_by="owner", delete_reason="finished", **facts)
    view = decorate(original, NOW)
    assert view["withdrawal"] == {"at": "then", "by": "owner", "reason": "finished",
                                  "work_in_flight": expected, "restorable": True}
    held = {**original, "followup_hold": {"hold_id": "h-1", "reason": "origin_stopped"}}
    assert decorate(held, NOW)["withdrawal"]["restorable"] is False  # only the exact release applies
    original["completed_at"] = "done"
    assert decorate(original, NOW)["withdrawal"]["restorable"] is False
    original.pop("delete_requested_at")
    assert "withdrawal" not in decorate(original, NOW)


def test_digest_orders_standing_first_then_attention_then_one_shots():
    rows = [row("once"), row("held", hold={"reason": "capacity", "detail": "Busy"}),
            row("active", trigger={"type": "cron", "expr": "0 * * * *"}),
            row("paused", trigger={"type": "cron", "expr": "0 * * * *"}, enabled=False),
            row("withdrawn", trigger={"type": "cron", "expr": "0 * * * *"}, delete_requested_at="then"),
            row("history", completed_at="done", enabled=False)]
    short = digest(rows, NOW, limit=3)
    assert [entry["id"] for entry in short["standing"]] == ["active", "paused", "withdrawn"]
    assert not short["attention"] and not short["one_shots"] and short["omitted_count"] == 2
    full = digest(rows, NOW)
    assert [entry["status"] for entry in full["standing"]] == ["active", "disabled", "delete_pending"]
    assert full["attention"][0]["waiting_reason"] == "capacity"
    assert full["one_shots"][0]["id"] == "once" and full["omitted_count"] == 0
    assert full["pointer"] == "manage_schedules(action='list')"
    assert "Check the report" not in json.dumps(full)


@pytest.mark.parametrize("at,enabled,marked", [
    ("2000-01-01T00:00:00+00:00", False, True), ("2099-01-01T00:00:00+00:00", False, False),
    ("2000-01-01T00:00:00+00:00", True, False),
])
def test_paused_past_due_one_shot_is_marked(at, enabled, marked):
    entry = digest([row(enabled=enabled, trigger={"type": "once", "run_at": at})], NOW)["one_shots"][0]
    assert ("time_passed_on" in entry) is marked


def test_attention_includes_wait_errors_unconfirmed_notes_and_previous_runs():
    rows = [row("wait", followup_wait="origin_unavailable"), row("error", last_error="bad time"),
            row("note", kind="notify", completed_at="done", last_error="delivery not confirmed"),
            row("open", previous_run_open={"task_id": "last", "since": "then"}, last_task_id="last"),
            row("quiet", hold={"reason": "capacity"}, last_task_id="last",
                last_outcome={"task_id": "last", "state": "completed", "quiet": True, "line": "Private"})]
    entries = digest(rows, NOW)["attention"]
    assert [entry["id"] for entry in entries] == ["wait", "error", "note", "open", "quiet"]
    assert entries[3]["outcome"] == {"state": "pending_observation"} and entries[3]["previous_run_open"]
    assert entries[4]["outcome"] == {"state": "completed", "quiet": True}
    assert "Private" not in json.dumps(entries)


def test_tool_and_activity_share_facts_and_bound_after_decoration():
    raw = row(name="Reminder of task abc", kind="notify", notification={"text": "Remember"},
              hold={"reason": "wait", "detail": "Detail"}, followup_wait="origin_unavailable")
    activity = schedule_activity_projection({"tasks": [raw]})["tasks"][0]
    tool = schedule_tool_projection({"tasks": [raw]})["tasks"][0]
    for key in ("display_name", "kind", "timezone", "destination", "waiting_detail"):
        assert activity[key] == tool[key]
    assert tool["objective_preview"] == "Remember"
    assert tool["waiting_detail"] == {"reason": "wait", "detail": "Detail", "followup_wait": "origin_unavailable"}
    large = row(destination={}, last_task_id="last", last_outcome={"task_id": "last", "state": "completed", "line": "x" * 1800})
    page = schedule_tool_projection({"tasks": [large, large]}, result_limit=5000)
    assert len(page["tasks"]) == 1 and page["next_offset"] == 1
    assert len(json.dumps(page, ensure_ascii=False, separators=(",", ":"))) <= 4000


@pytest.mark.parametrize("consumed", [True, False])
def test_withdrawal_restore_labels_do_not_promise_to_rearm_consumed_history(consumed):
    raw = row(delete_requested_at="then", enabled=False, **({"completed_at": "done"} if consumed else {}))
    for projection in (schedule_activity_projection, schedule_tool_projection):
        view = projection({"tasks": [raw]})["tasks"][0]
        assert view["restorable"] is not consumed and view["withdrawal"]["restorable"] is not consumed
        assert view["status"] == "delete_pending" and view["retained"]
