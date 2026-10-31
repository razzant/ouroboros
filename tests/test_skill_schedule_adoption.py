"""Former manifest rows converge once without changing accepted work or provenance."""
import copy

import pytest

from supervisor import schedule_lifecycle as lifecycle, queue_schedules as store
from supervisor.schedule_occurrence import fingerprint
from tests.test_consciousness_schedule_controls import _events, _skill_row
from tests.test_schedule_occurrence import q, _rows  # noqa: F401


@pytest.mark.parametrize("extra,enabled,withdrawn", [
    ({}, True, False),
    ({"enabled": False}, False, False),
    ({"manual_override": "deleted"}, False, True),
    ({"manual_override": "disabled"}, False, False),
    ({"followup_relation": {"kind": "related", "revision": "original"}}, True, False),
])
def test_adoption_preserves_the_authored_row_and_relation(extra, enabled, withdrawn):
    row = _skill_row(skill_content_hash="old", **extra)
    before = copy.deepcopy(row)
    assert lifecycle.adopt_legacy_skill_row(row)
    assert row["enabled"] is enabled
    assert bool(row.get("delete_requested_at")) is withdrawn
    assert "manual_override" not in row and "skill_content_hash" not in row
    if withdrawn:
        assert row["delete_requested_by"] == "host:skill_schedule_adoption"
        assert row["delete_reason"] == "deleted while skills managed schedules"
    for key in ("id", "name", "source", "skill", "task", "trigger", "timezone", "next_run_at"):
        assert row[key] == before[key]
    assert row.get("followup_relation") == before.get("followup_relation")
    assert ("followup_relation" in row) == ("followup_relation" in before)
    adopted = copy.deepcopy(row)
    assert not lifecycle.adopt_legacy_skill_row(row)
    assert row == adopted


@pytest.mark.parametrize("phase", ["claimed", "admitted"])
def test_adoption_keeps_claim_basis_and_admitted_identity(phase):
    row = _skill_row(manual_override="disabled", skill_content_hash="old",
                     occurrence={"phase": phase, "token": "receipt", "task_id": "fired"},
                     completed_at="2026-10-20T00:00:00+00:00", last_task_id="last",
                     hold={"reason": "capacity"})
    basis = fingerprint(row)
    assert lifecycle.adopt_legacy_skill_row(row)
    assert row["occurrence"] == {"phase": phase, "token": "receipt", "task_id": "fired", "fingerprint": basis}
    assert row["completed_at"] == "2026-10-20T00:00:00+00:00" and row["last_task_id"] == "last"
    assert row["hold"] == {"reason": "capacity"}


@pytest.mark.parametrize("enabled", [True, False])
def test_scan_adopts_before_policy_skip_and_the_second_scan_writes_nothing(q, enabled):  # noqa: F811
    row = _skill_row(enabled=enabled, skill_content_hash="old", next_run_at="2099-01-01T00:00:00+00:00")
    store._write_scheduled_tasks({"tasks": [row]}, q.root)
    q.queue.check_scheduled_tasks()
    path = q.root / "state/scheduled_tasks.json"
    first = path.read_bytes(), path.stat().st_mtime_ns
    q.queue.check_scheduled_tasks()
    assert (path.read_bytes(), path.stat().st_mtime_ns) == first
    assert "skill_content_hash" not in _rows(q)[row["id"]]
    audit = [e for e in _events(q.root) if e.get("action") == "adopt"]
    assert [e["phase"] for e in audit] == ["intent", "outcome"]
    assert audit[0]["operation_id"] == audit[1]["operation_id"]


@pytest.mark.parametrize("override", ["deleted", "disabled"])
def test_restore_before_first_scan_adopts_before_the_action(q, override):  # noqa: F811
    row = _skill_row(enabled=False, manual_override=override, skill_content_hash="old",
                     next_run_at="2099-01-01T00:00:00+00:00")
    store._write_scheduled_tasks({"tasks": [row]}, q.root)
    reply = q.queue.mutate_scheduled_task("restore", row["id"], actor="owner", reason="resume")
    assert reply["status"] == "updated"
    q.queue.check_scheduled_tasks()
    q.queue.check_scheduled_tasks()
    saved = _rows(q)[row["id"]]
    assert saved["enabled"] and not saved.get("delete_requested_at")
    assert "manual_override" not in saved and "skill_content_hash" not in saved
    assert len([e for e in _events(q.root) if e.get("action") == "adopt"]) == 2


def test_upsert_adopts_the_current_row_before_merging(q):  # noqa: F811
    row = _skill_row(manual_override="disabled", skill_content_hash="old")
    store._write_scheduled_tasks({"tasks": [row]}, q.root)
    saved = q.queue.upsert_scheduled_task({**row, "enabled": False, "name": "Changed"})
    assert saved["name"] == "Changed" and not saved["enabled"]
    assert "manual_override" not in saved and "skill_content_hash" not in saved
    assert len([e for e in _events(q.root) if e.get("action") == "adopt"]) == 2


def test_adoption_requires_an_auditable_intent(q, monkeypatch):  # noqa: F811
    row = _skill_row(enabled=False, skill_content_hash="old")
    store._write_scheduled_tasks({"tasks": [row]}, q.root)
    audit = lifecycle._audit_schedule_mutation
    monkeypatch.setattr(lifecycle, "_audit_schedule_mutation", lambda **kw: False)
    q.queue.check_scheduled_tasks()
    assert _rows(q)[row["id"]] == row
    monkeypatch.setattr(lifecycle, "_audit_schedule_mutation", audit)
    q.queue.check_scheduled_tasks()
    assert "skill_content_hash" not in _rows(q)[row["id"]]
