"""Lost-author projections require positive death evidence, never missing local custody."""
from __future__ import annotations

import pytest

from ouroboros import presence_continuation as pc
from ouroboros.review_operation import controller_identity, controller_state


@pytest.mark.parametrize("identity", [None, {}, {"pid": 0}, {"pid": "unreadable"}])
def test_missing_author_identity_does_not_invent_a_crash(identity):
    row = {"status": "running", "presence_continuation": {"initial": {}, "author_controller": identity}}
    assert pc.continuation_status(row) == "running"
    assert pc.work_view(row, "turn")[0] == 202


def test_current_process_author_remains_pending():
    row = {"status": "running", "presence_continuation": {
        "initial": {}, "author_controller": controller_identity()}}
    assert pc.continuation_status(row) == "running"
    assert pc.work_view(row, "turn")[1]["status"] == "pending"


@pytest.mark.parametrize("field", ["birth", "session"])
@pytest.mark.parametrize("observer_missing", [False, True])
def test_local_controller_with_incomplete_identity_remains_unknown(monkeypatch, field, observer_missing):
    from ouroboros import review_operation

    identity = controller_identity()
    incomplete = {key: value for key, value in identity.items() if key != field}
    if observer_missing:
        monkeypatch.setattr(review_operation, "controller_identity", lambda: {**incomplete, field: ""})
    else:
        identity = incomplete
    row = {"status": "running", "presence_continuation": {"initial": {}, "author_controller": identity}}
    assert controller_state(identity) == "unknown"
    assert pc.continuation_status(row) == "running" and pc.work_view(row, "turn")[0] == 202


def test_foreign_live_controller_without_birth_evidence_remains_unknown(monkeypatch):
    from ouroboros import platform_layer, process_containment

    identity = {**controller_identity(), "pid": controller_identity()["pid"] + 1, "session": "foreign"}
    monkeypatch.setattr(platform_layer, "pid_is_alive", lambda _pid: True)
    monkeypatch.setattr(process_containment, "pid_is_zombie", lambda _pid: False)
    monkeypatch.setattr(platform_layer, "process_start_time", lambda _pid: "")
    row = {"status": "running", "presence_continuation": {"initial": {}, "author_controller": identity}}
    assert controller_state(identity) == "unknown"
    assert pc.continuation_status(row) == "running" and pc.work_view(row, "turn")[0] == 202


@pytest.mark.parametrize("status", ["completed", "failed", "cancelled", "interrupted"])
def test_terminal_record_is_not_reclassified_by_an_author_witness(status):
    row = {"status": status, "presence_continuation": {"initial": {}, "author_controller": {"pid": 0}}}
    assert pc.continuation_status(row) == status


@pytest.mark.parametrize("status", ["running", "interrupted", "completed", "failed", "cancelled"])
@pytest.mark.parametrize("record", [
    {"initial": {"work_ref": ""}, "child_work_ref": "work-late"},
    {"initial": {"work_ref": "work-late"}},  # already retained by older hosts at the first park
])
def test_every_continuation_work_state_preserves_independent_child_discovery(status, record):
    row = {"status": status, "presence_continuation": record}
    code, body = pc.work_view(row, "author")
    assert code == (202 if status == "running" else 200)
    assert body["work_ref"] == body["continuation_ref"] == "author"
    assert body["child_work_ref"] == "work-late"


def test_terminal_promotion_after_last_park_takes_precedence_over_parked_child():
    row = {"status": "completed", "metadata": {"presence_work_ref": "work-terminal"},
           "presence_continuation": {"initial": {"work_ref": ""}, "child_work_ref": "work-parked"}}
    assert pc.work_view(row, "author")[1]["child_work_ref"] == "work-terminal"


def test_transport_queue_note_does_not_claim_leased_events_were_never_submitted():
    note = pc._queue_note({"status": "available", "snapshot": {
        "source": "slack-bridge", "pending_count": 1,
        "events": [{"source_event_id": "leased-1", "state": "leased", "text": "An in-flight event"}]}},
        bounded=False)
    assert '"state": "leased"' in note and "in flight" in note
    assert "not been submitted" not in note
    assert "not canonical chat or owner directives" in note
