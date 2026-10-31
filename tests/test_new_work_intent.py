"""Prepared intent survives crashes without an automatic second execution."""

from types import SimpleNamespace

import pytest

from ouroboros.promotion_intent import prepare_promotion_intent, claim_promotion_intent, reconcile_promotion_intents
from ouroboros.task_results import load_task_result, write_task_result
from ouroboros.tools.control_events import _emit_and_wait_for_routing
from tests.test_startup_terminal_files import roots  # noqa: F401


def event():
    return {"type": "promote_chat_to_task", "task_id": "new", "routing_token": "stable-token",
            "objective": "Separate work", "title": "Separate", "project_id": "", "source_text": "owner words",
            "source_ref": {"chat_id": 1, "client_message_id": "original"}, "attachment_uploads": [{"ref": "upload"}]}


def test_complete_intent_precedes_emit(tmp_path, monkeypatch):
    evt = event()
    emitted = []
    def emit(ctx, actual):
        stored = load_task_result(tmp_path, "new", strict=True)
        assert stored["promotion_intent"] == evt
        assert stored["promotion_admission"]["routing_token"] == evt["routing_token"]
        emitted.append(actual)
        return "deferred"
    monkeypatch.setattr("ouroboros.tools.control_events._emit_control_event", emit)
    ctx = SimpleNamespace(drive_root=tmp_path, unconfirmed_routing_acts=[])
    mode, receipt = _emit_and_wait_for_routing(ctx, evt)
    assert mode == "deferred" and receipt["status"] == "unconfirmed"
    assert emitted == [evt] and ctx.unconfirmed_routing_acts[0]["target"] == "new"


def test_failed_write_prevents_emission(tmp_path, monkeypatch):
    monkeypatch.setattr("ouroboros.promotion_intent.write_task_result",
                        lambda *a, **kw: (_ for _ in ()).throw(OSError("disk full")))
    ctx = SimpleNamespace(drive_root=tmp_path, pending_events=[])
    mode, receipt = _emit_and_wait_for_routing(ctx, event())
    assert mode == "intent_persist_failed" and receipt["status"] == "rejected"
    assert ctx.pending_events == []


@pytest.mark.parametrize("phase", ["prepared", "admitting", "possible", "admitted"])
def test_boot_distinguishes_proof_and_never_enqueues(tmp_path, phase):
    evt = event()
    prepare_promotion_intent(tmp_path, evt)
    if phase != "prepared":
        claim_promotion_intent(tmp_path, evt)
    if phase == "possible":
        write_task_result(tmp_path, "new", "requested", admitted_dispatch="possible")
    if phase == "admitted":
        write_task_result(tmp_path, "new", "scheduled", promotion_admission={
            "status": "scheduled", "routing_token": "stable-token"})
    report = reconcile_promotion_intents(tmp_path, protected=set())
    stored = load_task_result(tmp_path, "new", strict=True)
    assert stored["promotion_intent"] == evt
    if phase == "prepared":
        assert report["not_started"] == ["new"]
        assert stored["promotion_admission"]["status"] == "not_started"
        assert stored["promotion_admission"]["retry_available"] is True
        assert "Retry explicitly" in stored["result"]
    elif phase == "admitted":
        assert stored["status"] == "scheduled" and report == {"not_started": [], "held": []}
    else:
        assert report["held"] == ["new"] and stored["status"] == "requested"
        assert stored["promotion_admission"]["retry_available"] is False


def test_live_or_saved_custody_is_preserved(tmp_path):
    prepare_promotion_intent(tmp_path, event())
    assert reconcile_promotion_intents(tmp_path, protected={"new"}) == {"not_started": [], "held": []}
    assert load_task_result(tmp_path, "new")["promotion_admission"]["phase"] == "prepared"


@pytest.mark.parametrize("claimed", [False, True])
def test_startup_recovers_intent_only_after_worker_custody_proof(roots, monkeypatch, claimed):  # noqa: F811
    from ouroboros import server_maintenance
    from ouroboros.startup_migrations import prepare_startup_state
    from ouroboros.tools.control_task_results import _get_task_result
    from supervisor import queue

    root, repo = roots
    prepare_startup_state(root)
    evt = event()
    prepare_promotion_intent(root, evt)
    if claimed:
        claim_promotion_intent(root, evt)
    monkeypatch.setattr(queue, "enqueue_task", lambda *a, **k: pytest.fail("boot must not enqueue new work"))
    unknown = server_maintenance._run_startup_task_recovery(root, repo, skip_live_data=False)
    assert unknown["errors"] == ["prior_worker_ownership_unconfirmed"]
    assert load_task_result(root, "new")["promotion_admission"]["status"] == "emitted"
    report = server_maintenance._run_startup_task_recovery(root, repo, skip_live_data=False, prior_worker_pids=set())
    assert report["errors"] == []
    stored = load_task_result(root, "new")
    assert stored["promotion_admission"]["status"] == ("unconfirmed" if claimed else "not_started")
    result = _get_task_result(SimpleNamespace(drive_root=root, budget_drive_root=root), "new")
    assert ("held for reconciliation" if claimed else "explicit retry is available: start_task(") in result
    assert queue.PENDING == [] and queue.RUNNING == {}
