"""The patch seam changes authored fields without borrowing lifecycle authority."""
import copy
from contextlib import contextmanager

import pytest

from supervisor import queue_schedules as store, schedule_lifecycle as lifecycle, schedule_occurrence as occurrence
from tests.test_consciousness_schedule_controls import _bind, _write, _rows, _events, _once_row, _skill_row


@pytest.fixture
def root(tmp_path, monkeypatch):
    _bind(monkeypatch, tmp_path)
    _write(tmp_path, [{"id": "s1", "name": "Check", "enabled": False, "timezone": "UTC",
                      "trigger": {"type": "once", "run_at": "2099-01-01T00:00:00+00:00"},
                      "task": {"text": "Old objective", "context": "Kept", "metadata": {}},
                      "followup_relation": {"kind": "independent"}}])
    return tmp_path


def patch(root, fields, **kw):
    return lifecycle.patch_scheduled_task("s1", fields, drive_root=root, actor="task:root", reason="Changed promise", **kw)


def test_patch_changes_only_supplied_fields_and_audits_names(root):
    before = _rows(root)[0]
    reply = patch(root, {"name": "New label"})
    row = _rows(root)[0]
    # The host stamps who wrote the change; an unattributed caller is not the owner.
    assert row == {**before, "name": "New label", "updated_at": row["updated_at"],
                   "writer_kind": "consciousness", "writer_origin": row["writer_origin"]}
    assert reply["status"] == "updated" and reply["changed_fields"] == ["name"]
    assert not reply["reconsidered_claim"] and reply["audit"] == "recorded"
    events = _events(root)
    assert [e["phase"] for e in events] == ["intent", "outcome"]
    assert events[0]["operation_id"] == events[1]["operation_id"]
    assert all(e["changed_fields"] == ["name"] for e in events)


def test_unchanged_patch_does_not_write_the_table(root):
    path = root / "state/scheduled_tasks.json"
    before = path.read_bytes(), path.stat().st_mtime_ns
    assert patch(root, {"name": "Check"})["status"] == "unchanged"
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before


@pytest.mark.parametrize("fields,status", [
    ({"enabled": True}, "invalid_patch"), ({"followup_hold": None}, "invalid_patch"),
    ({"objective": ""}, "invalid_template"), ({"trigger": {}}, "invalid_template"),
    ({"trigger": {"type": "cron", "expr": "bad"}}, "invalid_template"),
    ({"trigger": {"type": "once", "run_at": "bad"}}, "invalid_template"),
    ({"timezone": "No/SuchZone"}, "invalid_template"),
    ({"reasoning_effort": "invented"}, "invalid_template"), ({"context": None}, "invalid_template"),
])
def test_invalid_patch_is_byte_identical_even_on_legacy_rows(root, fields, status):
    row = {**_rows(root)[0], "skill_content_hash": "old"}
    _write(root, [row])
    before = (root / "state/scheduled_tasks.json").read_bytes()
    assert patch(root, fields)["status"] == status
    assert (root / "state/scheduled_tasks.json").read_bytes() == before
    assert patch(root, {"objective": "Valid"})["status"] == "updated"
    assert "skill_content_hash" not in _rows(root)[0]
    assert [e["phase"] for e in _events(root) if e["action"] == "adopt"] == ["intent", "outcome"]


def test_missing_id_reason_and_withdrawal_refuse_without_writes(root):
    path = root / "state/scheduled_tasks.json"
    before = path.read_bytes()
    assert lifecycle.patch_scheduled_task("absent", {}, drive_root=root, actor="owner", reason="edit")["status"] == "not_found"
    assert lifecycle.patch_scheduled_task("s1", {}, drive_root=root, actor="owner", reason="")["status"] == "reason_required"
    assert path.read_bytes() == before
    row = {**_rows(root)[0], "delete_requested_at": "2026-10-21T00:00:00+00:00"}
    _write(root, [row])
    before = path.read_bytes()
    assert patch(root, {"objective": "Changed"})["status"] == "withdrawn"
    assert path.read_bytes() == before
    lifecycle.mutate_scheduled_task("restore", "s1", drive_root=root, actor="owner", reason="resume")
    assert patch(root, {"objective": "Changed"})["status"] == "updated"


@pytest.mark.parametrize("fields,rearmed", [
    ({"name": "History"}, False), ({"timezone": "Europe/Paris"}, False),
    ({"trigger": {"type": "once", "run_at": "2026-09-21T15:00:00+03:00"}}, False),
    ({"trigger": {"type": "once", "run_at": "2099-01-01T00:00:00+00:00"}}, True),
    ({"trigger": {"type": "cron", "expr": "0 9 * * *"}}, True),
])
def test_consumed_rearm_requires_a_new_trigger(root, fields, rearmed):
    row = _once_row(id="s1", trigger={"type": "once", "run_at": "2026-09-21T12:00:00+00:00"})
    _write(root, [row])
    reply = patch(root, fields)
    same_instant = fields.get("trigger", {}).get("run_at") == "2026-09-21T15:00:00+03:00"
    assert reply["status"] == ("unchanged" if same_instant else "updated")
    assert reply["enabled"] is rearmed
    assert ("completed_at" not in reply) is rearmed
    if rearmed:
        once = fields["trigger"]["type"] == "once"
        assert reply["next_run_at"] == "" if once else reply["next_run_at"]  # a one-shot is due at run_at


def test_paused_unconsumed_row_stays_paused_and_context_can_be_cleared(root):
    reply = patch(root, {"trigger": {"type": "once", "run_at": "2099-02-01T00:00:00+00:00"}, "context": ""})
    assert not reply["enabled"] and reply["task"]["context"] == ""
    assert reply["changed_fields"] == ["context", "trigger"]


@pytest.mark.parametrize("accepted", [False, True])
def test_fingerprint_edit_preserves_basis_and_only_reconsiders_unstarted_claim(root, monkeypatch, accepted):
    row = {**_rows(root)[0], "enabled": True, "hold": {"reason": "capacity"},
           "followup_hold": {"hold_id": "owner-stop"}, "continuation_of": {"task_id": "original"},
           "occurrence": {"phase": "claimed", "task_id": "fired", "token": "original"}}
    _write(root, [row])
    monkeypatch.setattr(occurrence, "_read_back", lambda *a, **k: {"status": "scheduled"} if accepted else {})
    reply = patch(root, {"name": "Renamed"})
    assert reply["reconsidered_claim"] is not accepted
    assert reply["occurrence"] == {"fingerprint": occurrence.fingerprint(row), **row["occurrence"]}
    assert "hold" not in reply and reply["followup_hold"] == row["followup_hold"]
    assert reply["continuation_of"] == row["continuation_of"]


def test_note_edit_preserves_authorship_and_changes_written_time(root):
    row = {**_rows(root)[0], "kind": "notify", "notification": {"text": "Remember", "set_at": "old", "author": "mind"}}
    _write(root, [row])
    assert patch(root, {"context": "extra"})["status"] == "invalid_patch"
    reply = patch(root, {"objective": "Remember this"})
    assert reply["notification"]["text"] == "Remember this"
    assert reply["notification"]["author"] == "mind" and reply["notification"]["set_at"] != "old"


def test_audit_failure_blocks_intent_but_discloses_lost_outcome(root, monkeypatch):
    before = (root / "state/scheduled_tasks.json").read_bytes()
    monkeypatch.setattr(lifecycle, "_audit_schedule_mutation", lambda **kw: False)
    assert patch(root, {"name": "Changed"})["status"] == "audit_unavailable"
    assert (root / "state/scheduled_tasks.json").read_bytes() == before
    monkeypatch.setattr(lifecycle, "_audit_schedule_mutation", lambda **kw: kw["phase"] == "intent")
    reply = patch(root, {"name": "Changed"})
    assert reply["status"] == "updated" and reply["audit"] == "incomplete"
    assert _rows(root)[0]["name"] == "Changed"


def test_corrupt_store_and_lock_timeout_are_typed(root, monkeypatch):
    path = root / "state/scheduled_tasks.json"
    valid = path.read_bytes()
    path.write_text("{torn")
    assert patch(root, {})["status"] == "store_unreadable" and path.read_text(encoding="utf-8") == "{torn"
    path.write_bytes(valid)
    transaction = lifecycle.schedule_transaction
    @contextmanager
    def busy(*a):
        raise store.ScheduleLockTimeout("busy")
        yield
    monkeypatch.setattr(lifecycle, "schedule_transaction", busy)
    assert patch(root, {})["status"] == "lock_timeout" and path.read_bytes() == valid
    monkeypatch.setattr(lifecycle, "schedule_transaction", transaction)
    assert patch(root, {"name": "Changed"})["status"] == "updated"


def test_equal_description_copies_collapse_after_claim_basis_is_remembered(root):
    row = {**_rows(root)[0], "description": "Old objective", "task": {"text": "Old objective", "description": "Old objective"},
           "occurrence": {"phase": "claimed", "task_id": "fired", "token": "same"}}
    _write(root, [row])
    reply = patch(root, {"name": "New name"})
    assert "description" not in reply and "description" not in reply["task"]
    assert reply["occurrence"]["fingerprint"] == occurrence.fingerprint(row)
    _write(root, [row])
    reply = patch(root, {"objective": "New objective"})
    assert "description" not in reply and "description" not in reply["task"]
    row["description"] = "Different description"
    _write(root, [row])
    reply = lifecycle.upsert_scheduled_task(copy.deepcopy(row), drive_root=root)
    assert reply["description"] == "Different description" and "description" not in reply["task"]
    row["task"]["description"] = "Distinct task description"
    _write(root, [row])
    reply = patch(root, {"name": "Keep distinct descriptions"})
    assert reply["description"] == "Different description" and reply["task"]["description"] == "Distinct task description"


def test_legacy_deleted_row_is_refused_without_losing_its_marker(root):
    row = _skill_row("s1", manual_override="deleted")
    _write(root, [row])
    assert patch(root, {"name": "Changed"})["status"] == "withdrawn"
    assert _rows(root) == [row]


def test_all_allowed_fields_accept_valid_values_without_enabling_a_paused_row(root):
    fields = {"name": "Weekly review", "objective": "Read the updates", "context": "Include the notes",
              "trigger": {"type": "cron", "expr": "0 8 * * 1"}, "timezone": "Europe/Paris", "reasoning_effort": "high"}
    reply = patch(root, fields)
    assert reply["status"] == "updated" and reply["changed_fields"] == sorted(fields)
    assert not reply["enabled"] and reply["next_run_at"]
    assert reply["task"]["text"] == fields["objective"] and reply["task"]["context"] == fields["context"]
    assert reply["task"]["reasoning_effort"] == "high" and reply["timezone"] == "Europe/Paris"
    assert reply["trigger"] == fields["trigger"]
