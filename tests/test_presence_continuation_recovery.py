"""Lost-author projections require positive death evidence, never missing local custody."""
from __future__ import annotations

import pytest

from ouroboros import presence_continuation as pc
from ouroboros.review_operation import controller_identity, controller_state


@pytest.mark.parametrize("identity", [None, {}, {"pid": "unreadable"}])
def test_missing_author_identity_does_not_invent_a_crash(identity):
    row = {"status": "running", "presence_continuation": {"initial": {}, "author_controller": identity}}
    assert pc.continuation_status(row) == "running"
    assert pc.work_view(row, "turn")[0] == 202


def test_current_process_author_remains_pending():
    row = {"status": "running", "presence_continuation": {
        "initial": {}, "author_controller": controller_identity()}}
    assert pc.continuation_status(row) == "running"
    assert pc.work_view(row, "turn")[1]["status"] == "pending"


@pytest.mark.parametrize(("observer_missing", "fields"), [
    pytest.param(False, ("birth", "kernel_birth"), id='False-birth'),
    pytest.param(False, ("session",), id='False-session'),
    pytest.param(True, ("birth", "kernel_birth"), id='True-birth'),
])
def test_local_controller_with_incomplete_identity_remains_unknown(monkeypatch, fields, observer_missing):
    from ouroboros import review_operation

    identity = controller_identity()
    incomplete = {key: value for key, value in identity.items() if key not in fields}
    if observer_missing:
        observer = {**incomplete, **dict.fromkeys(fields, "")}
        monkeypatch.setattr(review_operation, "controller_identity", lambda: observer)
    else:
        identity = incomplete
    row = {"status": "running", "presence_continuation": {"initial": {}, "author_controller": identity}}
    assert controller_state(identity) == "unknown"
    assert pc.continuation_status(row) == "running" and pc.work_view(row, "turn")[0] == 202


def test_foreign_live_controller_without_birth_evidence_remains_unknown(monkeypatch):
    from ouroboros import platform_layer, process_containment, process_identity

    identity = {**controller_identity(), "pid": controller_identity()["pid"] + 1, "session": "foreign"}
    monkeypatch.setattr(platform_layer, "pid_is_alive", lambda _pid: True)
    monkeypatch.setattr(process_containment, "pid_is_zombie", lambda _pid: False)
    monkeypatch.setattr(platform_layer, "process_start_time", lambda _pid: "")
    monkeypatch.setattr(process_identity, "process_kernel_birth", lambda _pid: None)
    row = {"status": "running", "presence_continuation": {"initial": {}, "author_controller": identity}}
    assert controller_state(identity) == "unknown"
    assert pc.continuation_status(row) == "running" and pc.work_view(row, "turn")[0] == 202


@pytest.mark.parametrize("status", ["completed", "failed", "interrupted"])
def test_terminal_record_is_not_reclassified_by_an_author_witness(status):
    row = {"status": status, "presence_continuation": {"initial": {}, "author_controller": {"pid": 0}}}
    assert pc.continuation_status(row) == status


@pytest.mark.parametrize(("record", "status"), [
    pytest.param({"initial": {"work_ref": ""}, "child_work_ref": "work-late"}, "running", id='record0-running'),
    pytest.param({"initial": {"work_ref": ""}, "child_work_ref": "work-late"}, "completed", id='record0-completed'),
    pytest.param({"initial": {"work_ref": ""}, "child_work_ref": "work-late"}, "cancelled", id='record0-cancelled'),
    pytest.param({"initial": {"work_ref": "work-late"}}, "interrupted", id='record1-interrupted'),  # already retained by older hosts at the first park
    pytest.param({"initial": {"work_ref": "work-late"}}, "failed", id='record1-failed'),  # already retained by older hosts at the first park
])
def test_every_continuation_work_state_preserves_independent_child_discovery(status, record):
    row = {"status": status, "presence_continuation": record}
    code, body = pc.work_view(row, "author")
    assert code == (200 if status in {"completed", "failed", "cancelled"} else 202)  # ended-ness: retained_work_view
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
