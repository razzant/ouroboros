"""Schedule reads and authored patches keep the root's authority and exact sources."""

import json

import pytest

from supervisor import queue
from tests.test_consciousness_authority import _registry, _wake_task


@pytest.fixture
def registry(tmp_path, monkeypatch):
    # update is Safety-checked at the calling work's level; these tests pin the tool's own authority.
    monkeypatch.setenv("OUROBOROS_SAFETY_MODE", "off")
    monkeypatch.setenv("OUROBOROS_CONSCIOUSNESS_SAFETY_MODE", "off")
    registry = _registry(tmp_path)
    queue.upsert_scheduled_task({
        "id": "s1", "name": "Morning review", "enabled": False, "timezone": "UTC",
        "trigger": {"type": "cron", "expr": "0 8 * * *"},
        "task": {"text": "Read the review skill", "context": "Keep these facts", "chat_id": 42,
                 "project_id": "reports", "metadata": {"source": "owner"}},
    }, drive_root=registry._ctx.drive_root, actor="owner", reason="standing promise")
    return registry


def _call(registry, action, **params):
    return registry.execute_result("manage_schedules", {"action": action, "schedule_id": "s1", **params})


def _stored(registry):
    return queue.load_schedule_store(registry._ctx.drive_root)["tasks"][0]


def test_get_returns_the_full_authored_definition_and_shared_facts_without_a_write(registry):
    path = registry._ctx.drive_root / "state/scheduled_tasks.json"
    before = path.read_bytes(), path.stat().st_mtime_ns
    result = _call(registry, "get")
    assert result.status == "ok"
    row = json.loads(result.text)["schedule"]
    assert row["task"] == _stored(registry)["task"]
    assert row["trigger"] == _stored(registry)["trigger"]
    assert row["status"] == "disabled" and row["display_name"] == "Morning review"
    assert row["destination"] == {"chat_id": 42, "project_id": "reports"}
    assert row["timezone"] == "UTC" and row["kind"] == "task"
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before


@pytest.mark.parametrize("level", [None, "act", "full"])
def test_root_and_act_full_turns_patch_only_supplied_fields(registry, level, monkeypatch):
    monkeypatch.setattr("ouroboros.config._BOOT_RUNTIME_MODE", "cyber_pro")
    if level:
        registry._ctx.task_metadata = _wake_task(level)["metadata"]
    before = _stored(registry)
    result = _call(registry, "update", reason="Use the revised procedure", name="Evening review",
                   objective="Use the review skill to check the new reports", context="", run_at="",
                   cron="0 18 * * *", timezone="Europe/Paris", reasoning_effort="high")
    assert result.status == "ok", result.text
    reply = json.loads(result.text)
    assert reply["status"] == "updated" and reply["changed"] and reply["audit"] == "recorded"
    assert reply["changed_fields"] == ["name", "objective", "reasoning_effort", "timezone", "trigger"]
    assert reply["reconsidered_claim"] is False
    row = reply["schedule"]
    assert row["name"] == "Evening review" and row["task"]["text"].endswith("new reports")
    assert row["task"]["context"] == before["task"]["context"]
    assert row["task"]["reasoning_effort"] == "high"
    assert row["task"]["metadata"] == before["task"]["metadata"]
    assert row["destination"] == {"chat_id": 42, "project_id": "reports"}
    assert row["timezone"] == "Europe/Paris" and row["next_run_at"]
    assert not row["enabled"] and row["status"] == "disabled"


def test_empty_patch_fields_are_absent_and_reason_is_required(registry):
    path = registry._ctx.drive_root / "state/scheduled_tasks.json"
    before = path.read_bytes()
    refused = _call(registry, "update", name="Changed")
    assert refused.status != "ok" and json.loads(refused.text)["status"] == "reason_required"
    assert refused.meta["operation_outcome"] == "completed_no_effect"
    assert path.read_bytes() == before
    fields = {name: " " for name in ("name", "objective", "context", "run_at", "cron", "timezone")}
    result = _call(registry, "update", reason="Keep this promise", reasoning_effort="", **fields)
    assert result.status == "ok"
    assert json.loads(result.text)["status"] == "unchanged" and path.read_bytes() == before


@pytest.mark.parametrize("params", [
    {"cron": "hourly"}, {"run_at": "soon"}, {"cron": "0 9 * * *", "run_at": "2099-01-01T00:00:00Z"},
    {"timezone": "Mars/Olympus"}, {"reasoning_effort": "invented"},
    {"objective": "x" * 4001}, {"context": "x" * 8001},
])
def test_invalid_edits_are_typed_and_leave_bytes_unchanged_with_a_valid_path_open(registry, params):
    path = registry._ctx.drive_root / "state/scheduled_tasks.json"
    before = path.read_bytes()
    result = _call(registry, "update", reason="Change the promise", **params)
    assert result.status != "ok" and result.meta["operation_outcome"] == "completed_no_effect"
    assert path.read_bytes() == before
    valid = _call(registry, "update", reason="Use a new instant", run_at="2099-01-01T12:00:00+03:00")
    assert valid.status == "ok", valid.text
    assert _stored(registry)["trigger"] == {"type": "once", "run_at": "2099-01-01T09:00:00+00:00"}


@pytest.mark.parametrize("action", ["get", "update"])
def test_missing_schedule_is_not_created_and_existing_schedule_still_works(registry, action):
    before = _stored(registry)
    result = registry.execute_result("manage_schedules", {
        "action": action, "schedule_id": "missing", "reason": "Change", "name": "Changed"})
    assert result.status != "ok" and json.loads(result.text)["status"] == "not_found"
    assert queue.load_schedule_store(registry._ctx.drive_root)["tasks"] == [before]
    assert _call(registry, action, reason="Change", name="Changed").status == "ok"


@pytest.mark.parametrize("mode", ["pro", "cyber_pro"])
def test_update_effort_keeps_the_creation_policy(registry, monkeypatch, mode):
    monkeypatch.setattr("ouroboros.config._BOOT_RUNTIME_MODE", mode)
    result = _call(registry, "update", reason="Set starting effort", reasoning_effort="high")
    assert result.status == "ok"
    reply = json.loads(result.text)
    if mode == "cyber_pro":
        assert _stored(registry)["task"]["reasoning_effort"] == "high"
        assert "argument_note" not in reply
    else:
        assert "reasoning_effort" not in _stored(registry)["task"]
        assert "reasoning_effort='high' ignored" in reply["argument_note"]


@pytest.mark.parametrize("metadata", [
    {"delegation_role": "subagent"}, _wake_task("observe")["metadata"],
    {"presence": {"event": {"source_event_id": "guest"}}},
])
def test_update_authority_refuses_without_writes_and_preserves_root_access(registry, metadata):
    before = _stored(registry)
    registry._ctx.task_metadata = metadata
    result = _call(registry, "update", reason="Change", name="Changed")
    assert result.status == "blocked" and _stored(registry) == before
    if "presence" not in metadata:
        assert _call(registry, "get").status == "ok"
        assert json.loads(_call(registry, "list").text)["total"] == 1
    else:
        assert _call(registry, "get").status == "blocked"
        assert _call(registry, "list").status == "blocked"
    registry._ctx.task_metadata = {}
    assert _call(registry, "get").status == "ok"
    assert _call(registry, "update", reason="Change", name="Changed").status == "ok"


def test_observe_keeps_existing_lifecycle_controls(registry):
    registry._ctx.task_metadata = _wake_task("observe")["metadata"]
    for action in ("disable", "delete", "restore"):
        assert _call(registry, action, reason="Revise lifecycle").status == "ok"
    assert _stored(registry)["enabled"]


@pytest.mark.parametrize("action", ["get", "update"])
def test_corrupt_store_is_unavailable_and_not_an_empty_success(registry, action):
    path = registry._ctx.drive_root / "state/scheduled_tasks.json"
    before = path.read_bytes()
    path.write_text("{torn", encoding="utf-8")
    result = _call(registry, action, reason="Change", name="Changed")
    assert result.status != "ok" and path.read_text(encoding="utf-8") == "{torn"
    path.write_bytes(before)
    assert _call(registry, action, reason="Change", name="Changed").status == "ok"


def test_update_discloses_a_durable_edit_with_incomplete_audit(registry, monkeypatch):
    from supervisor import schedule_lifecycle

    monkeypatch.setattr(schedule_lifecycle, "_audit_schedule_mutation", lambda **kw: kw["phase"] == "intent")
    result = _call(registry, "update", reason="Change", name="Changed")
    reply = json.loads(result.text)
    assert reply["status"] == "updated" and reply["changed"] and not reply["ok"]
    assert reply["audit"] == "incomplete" and _stored(registry)["name"] == "Changed"
    assert result.meta.get("operation_outcome") != "completed_no_effect"


def test_update_reports_claim_reconsideration_and_preserves_the_policy_hold(registry, monkeypatch):
    from supervisor import schedule_occurrence

    row = _stored(registry)
    row.update(hold={"reason": "capacity"}, followup_hold={"hold_id": "owner-stop"},
               occurrence={"phase": "claimed", "task_id": "unstarted", "token": "claim"})
    path = registry._ctx.drive_root / "state/scheduled_tasks.json"
    path.write_text(json.dumps({"tasks": [row]}), encoding="utf-8")
    monkeypatch.setattr(schedule_occurrence, "_read_back", lambda *_a, **_kw: {})
    reply = json.loads(_call(registry, "update", reason="Use a new label", name="New label").text)
    assert reply["reconsidered_claim"] is True
    assert reply["schedule"]["followup_hold"] == row["followup_hold"]
    assert "hold" not in reply["schedule"]


def test_note_update_changes_the_exact_words_and_rejects_task_context(registry):
    row = _stored(registry)
    row.update(kind="notify", notification={"text": "Remember", "set_at": "old"})
    path = registry._ctx.drive_root / "state/scheduled_tasks.json"
    path.write_text(json.dumps({"tasks": [row]}), encoding="utf-8")
    before = path.read_bytes()
    refused = _call(registry, "update", reason="Add hidden context", context="Hidden")
    assert refused.status != "ok" and json.loads(refused.text)["status"] == "invalid_patch"
    assert path.read_bytes() == before
    reply = json.loads(_call(registry, "update", reason="Change the reminder", objective="Call tomorrow").text)
    assert reply["schedule"]["notification"]["text"] == "Call tomorrow"
    assert reply["schedule"]["notification"]["set_at"] != "old"


@pytest.mark.parametrize("action", ["get", "update"])
def test_large_definitions_use_existing_exact_result_sources(registry, action):
    from ouroboros import artifacts
    from tests.test_tool_result_delivery import _deliver, _row, _source_ref, measured_fit

    record = _stored(registry)
    record["task"]["context"] = "START\n" + "Report data\n" * 4000 + "TAIL"
    queue.upsert_scheduled_task(record, drive_root=registry._ctx.drive_root)
    result = _call(registry, action, reason="New label", name="Changed")
    assert result.status == "ok" and json.loads(result.text)["schedule"]["task"]["context"].endswith("TAIL")
    messages, _trace = _deliver(registry._ctx, [_row("full-definition", "manage_schedules", result.text)],
                                measured_fit(window=12000, reserve=2000))
    content = messages[0]["content"]
    ref = _source_ref(content)
    recovered = artifacts.read_actor_source_bytes(registry._ctx.drive_root, registry._ctx.task_id, ref).decode()
    assert recovered == result.text and len(content) < len(result.text)
    assert json.loads(recovered)["schedule"]["task"]["context"] == record["task"]["context"]


def test_tool_schema_exposes_patch_fields_without_widening_the_lifecycle_door():
    from ouroboros.tools.followup import get_tools

    schema = get_tools()[1].schema["parameters"]
    assert set(schema["properties"]["action"]["enum"]) == {"list", "get", "update", *queue.SCHEDULE_ACTIONS}
    assert {"name", "objective", "context", "run_at", "cron", "timezone", "reasoning_effort"} <= set(schema["properties"])
    assert schema["additionalProperties"] is False and schema["required"] == ["action"]
    assert not {"anyOf", "oneOf", "allOf"}.intersection(schema)
    assert queue.SCHEDULE_ACTIONS == {"disable", "delete", "restore"}


def test_a_naive_run_at_beside_a_timezone_is_refused_as_on_creation(registry):
    path = registry._ctx.drive_root / "state/scheduled_tasks.json"
    before = path.read_bytes()
    refused = _call(registry, "update", reason="Move it", run_at="2099-01-01T12:00:00", timezone="Europe/Moscow")
    assert refused.status != "ok" and "FOLLOWUP_TIMEZONE_WITH_RUN_AT" in refused.text
    assert path.read_bytes() == before
    accepted = _call(registry, "update", reason="Move it", run_at="2099-01-01T12:00:00+03:00", timezone="Europe/Moscow")
    assert accepted.status == "ok", accepted.text
    assert _stored(registry)["trigger"] == {"type": "once", "run_at": "2099-01-01T09:00:00+00:00"}


def test_lifecycle_receipts_describe_the_row_as_stored(registry):
    from supervisor import queue_schedules as store

    root = registry._ctx.drive_root
    data = queue.load_schedule_store(root)
    data["tasks"][0].update(last_task_id="t1", last_outcome={
        "task_id": "t1", "state": "completed", "quiet": False, "notice": False, "ended_at": None, "line": "Sent"})
    store._write_scheduled_tasks(data, root)
    result = _call(registry, "delete", reason="No longer needed")
    assert result.status == "ok", result.text
    row = json.loads(result.text)["schedule"]
    # The audit row lacks these facts; the receipt reads the row as stored after the change.
    assert row["outcome"]["state"] == "completed" and row["kind"] == "task"
    assert row["destination"] == {"chat_id": 42, "project_id": "reports"}
    assert row["withdrawal"]["work_in_flight"] is False and row["status"] == "delete_pending"
