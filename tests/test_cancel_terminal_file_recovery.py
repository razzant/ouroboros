"""Cancellation completes interrupted file publication before delivery and cleanup."""
from types import SimpleNamespace

import pytest

from ouroboros import cancel_intents, headless
from ouroboros.task_results import load_task_result, write_task_result
from tests._cancel_intents_shared import qenv, settled_off_loop  # noqa: F401
from tests._store_shared import read_result, result_signature


@pytest.fixture
def split(qenv, monkeypatch):  # noqa: F811 -- imported pytest fixture
    task_id = "cancel-files"
    child = headless.prepare_task_drive(qenv.drive, task_id, "empty")
    task = {"id": task_id, "type": "task", "chat_id": 5, "_attempt": 1,
            "drive_root": str(child), "delegation_role": "subagent", "parent_task_id": "parent"}
    state = {"alive": True, "kills": 0}
    def terminate():
        state.update(alive=False, kills=state["kills"] + 1)
    proc = SimpleNamespace(pid=None, is_alive=lambda: state["alive"], terminate=terminate,
                           join=lambda **_kw: None)
    worker = SimpleNamespace(wid=7, busy_task_id=task_id, proc=proc, reaping=False)
    qenv.workers.WORKERS[7] = worker
    qenv.q.RUNNING[task_id] = {"task": task, "worker_id": 7, "attempt": 1}
    frames, pushed, sent = [], [], []
    monkeypatch.setattr(qenv.workers, "get_event_q", lambda: SimpleNamespace(put=frames.append))
    monkeypatch.setattr(qenv.tl, "_audit_delegated_runs_on_kill", lambda *_a, **_kw: {"unreconciled": []})
    ctx = SimpleNamespace(DRIVE_ROOT=qenv.drive, RUNNING=qenv.q.RUNNING, WORKERS=qenv.workers.WORKERS,
                          bridge=SimpleNamespace(push_log=pushed.append),
                          persist_queue_snapshot=lambda **_kw: None,
                          send_with_budget=lambda *a, **kw: sent.append((a, kw)))
    return SimpleNamespace(**vars(qenv), task=task, child=child, worker=worker, state=state,
                           frames=frames, pushed=pushed, sent=sent, ctx=ctx)


@pytest.mark.parametrize("adopted", [False, True])
def test_cancel_prepares_early_terminal_before_cleanup_and_real_dispatch(split, monkeypatch, adopted):
    from supervisor.events import dispatch_event

    s = split
    review = {"panels": [{"panel_id": "accepted", "aggregate_signal": "PASS", "actors": []}]}
    write_task_result(s.child, s.task["id"], "completed", result="The child answer", review_projection=review,
                      parent_task_id="parent", root_task_id="parent", delegation_role="subagent")
    s.task["workspace_root"] = str(s.drive / "workspace")
    if adopted:
        headless.copy_child_task_result(s.drive, s.task)
    else:
        write_task_result(s.drive, s.task["id"], "completed",
                          root_phase_checkpoint={"post_task_synthesis": "completed"})
    finalized = []
    def finalize(root, task):
        assert not s.state["alive"] and s.child.exists()
        finalized.append(task["id"])
        return write_task_result(root, task["id"], "completed", artifact_status="ready",
                                 artifact_finalized_at="fixture", artifact_bundle={"status": "ready", "artifacts": []})
    monkeypatch.setattr(headless, "finalize_task_artifacts", finalize)
    cancel_intents.request_cancel(s.drive, s.task["id"], reason="owner", allow_settled_target=True)
    assert s.q.cancel_task_custody(s.task["id"]) == s.q.CANCEL_ALREADY_SETTLED
    assert finalized == [s.task["id"]] and s.child.exists()
    for event in s.frames:
        dispatch_event(event, s.ctx)
    assert [e["status"] for e in s.pushed if e.get("type") == "task_done"] == ["completed"]
    stored = load_task_result(s.drive, s.task["id"])
    assert stored["result"] == "The child answer" and stored["review_projection"] == review
    assert stored["artifact_status"] == "ready"
    assert s.pushed[-1]["chat_id"] == 5 and s.pushed[-1]["review_projection"] == review
    from ouroboros.tools.control_task_results import _get_task_result

    parent = SimpleNamespace(task_id="parent", drive_root=s.drive, budget_drive_root=s.drive, task_metadata={})
    assert "The child answer" in _get_task_result(parent, s.task["id"])
    assert stored["child_ref_promotion"]["pending_refs"], "cancel and task_done do not wait for historical transfer"
    headless.retry_child_task_refs(s.drive, s.child, s.task["id"])
    assert settled_off_loop(s.drive, s.task["id"], s.child)


def test_cancel_file_outage_retains_intent_and_retries_dead_worker(split, monkeypatch):
    s = split
    write_task_result(s.child, s.task["id"], "completed", result="Retained child answer")
    write_task_result(s.drive, s.task["id"], "completed",
                      root_phase_checkpoint={"post_task_synthesis": "completed"})
    cancel_intents.request_cancel(s.drive, s.task["id"], reason="owner", allow_settled_target=True)
    with monkeypatch.context() as fault:
        fault.setattr(headless, "write_task_result", lambda *_a, **_kw: (_ for _ in ()).throw(OSError("disk full")))
        assert s.q.cancel_task_custody(s.task["id"]) == s.q.CANCEL_FAILED
        assert not s.state["alive"], "the worker is already dead while file publication is unresolved"
        from supervisor.events_runtime_controls import _drive_cancel_task_event

        _drive_cancel_task_event({"task_id": s.task["id"]}, s.ctx)
        assert s.sent == [], "cancel_task drives its intent without writing a chat notice"
    assert s.child.is_dir() and s.frames == [] and s.task["id"] in s.q.RUNNING
    assert cancel_intents.cancel_pending(s.drive, s.task["id"])
    assert s.q.cancel_task_custody(s.task["id"]) == s.q.CANCEL_ALREADY_SETTLED
    assert s.state["kills"] == 1 and s.child.exists()
    assert load_task_result(s.drive, s.task["id"])["result"] == "Retained child answer"
    headless.retry_child_task_refs(s.drive, s.child, s.task["id"])
    assert settled_off_loop(s.drive, s.task["id"], s.child)


@pytest.mark.parametrize("cascade", [False, True])
def test_owner_http_stop_over_a_dead_worker_retains_pending_intent(split, monkeypatch, cascade):
    """A dead worker with unresolved file publication keeps its recorded Stop pending."""
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from ouroboros.gateway.tasks import api_task_cancel

    s = split
    write_task_result(s.child, s.task["id"], "completed", result="Retained child answer")
    write_task_result(s.drive, s.task["id"], "running")  # a settled row would be completion-wins
    app = Starlette(routes=[Route("/api/tasks/{task_id}/cancel", api_task_cancel, methods=["POST"])])
    app.state.drive_root = s.drive
    with monkeypatch.context() as fault, TestClient(app) as client:
        fault.setattr(headless, "write_task_result", lambda *_a, **_kw: (_ for _ in ()).throw(OSError("disk full")))
        response = client.post(f"/api/tasks/{s.task['id']}/cancel", json={"cascade": True} if cascade else {})
    assert not s.state["alive"], "the worker is already dead while file publication is unresolved"
    assert response.status_code == 202, response.text
    intent = cancel_intents.active_intent(s.drive, s.task["id"])
    assert response.json() == {
        "ok": True, "task_id": s.task["id"], "cancel_state": "pending", "stop_policy": "immediate",
        "request_id": intent["request_id"], "reason_code": "teardown_unconfirmed",
        **({"cascade": True} if cascade else {}),
    }
    assert cancel_intents.cancel_pending(s.drive, s.task["id"])
    assert s.q.cancel_task_custody(s.task["id"]) == s.q.CANCEL_ALREADY_SETTLED
    assert load_task_result(s.drive, s.task["id"])["result"] == "Retained child answer"


def test_genuinely_ready_current_survives_cancel_byte_identical(split, monkeypatch):
    s = split
    write_task_result(s.child, s.task["id"], "completed", result="Accepted answer",
                      review_projection={"panels": [{"panel_id": "paid-PASS", "actors": []}]})
    assert not headless.prepare_terminal_task_files(s.drive, s.task)["error"]
    headless.retry_child_task_refs(s.drive, s.child, s.task["id"])
    before = result_signature(s.drive, s.task["id"])
    write_task_result(s.child, s.task["id"], "completed", result="Stale replica")
    monkeypatch.setattr(headless, "prepare_terminal_task_files", lambda *_a: pytest.fail("accepted CURRENT recopied"))
    cancel_intents.request_cancel(s.drive, s.task["id"], reason="owner", allow_settled_target=True)
    assert s.q.cancel_task_custody(s.task["id"]) == s.q.CANCEL_ALREADY_SETTLED
    assert result_signature(s.drive, s.task["id"]) == before
    assert read_result(s.drive, s.task["id"])["result"] == "Accepted answer" and settled_off_loop(s.drive, s.task["id"], s.child)


def test_reconciled_terminal_without_checkpoint_still_needs_real_file_adoption(split):
    from ouroboros.observability import write_blob
    from ouroboros.task_status import reconcile_orphaned_running_tasks

    s = split
    blob = write_blob(s.child, {"message": {"content": "Exact child evidence"}})
    write_task_result(s.child, s.task["id"], "completed", result="Child answer", trace_refs={"response": blob})
    write_task_result(s.drive, s.task["id"], "running", child_drive_root=str(s.child))
    assert reconcile_orphaned_running_tasks(s.drive) == 1
    projected = load_task_result(s.drive, s.task["id"])
    assert projected["status"] == "completed" and projected["result"] == "Child answer"
    assert "root_phase_checkpoint" not in projected and "child_ref_promotion" not in projected
    assert not headless.terminal_task_files_ready(s.drive, s.task, projected)
    assert not (s.drive / "observability").exists()
    assert not headless.prepare_terminal_task_files(s.drive, s.task)["error"]
    adopted = load_task_result(s.drive, s.task["id"])
    assert headless.terminal_task_files_ready(s.drive, s.task, adopted)
    assert adopted["result"] == projected["result"] and not (s.drive / "observability").exists()
    assert adopted["child_ref_promotion"]["pending_refs"]
    retained = headless.retry_child_task_refs(s.drive, s.child, s.task["id"])
    assert not retained["child_ref_promotion"]["pending_refs"] and (s.drive / "observability").is_dir()


def test_settled_failed_native_capture_is_not_reprepared(split, monkeypatch):
    """A settled failed capture stays failed during later dead-worker recovery."""
    task = {
        "id": "failed-native",
        "type": "task",
        "chat_id": 5,
        "workspace_root": str(split.drive / "workspace"),
    }
    write_task_result(
        split.drive,
        task["id"],
        "failed",
        result="The worker failed before capture completed",
        artifact_status="failed",
        artifact_error="native capture failed",
        native_capture_pending=True,
    )
    finalizer_calls = []
    monkeypatch.setattr(
        headless,
        "finalize_task_artifacts",
        lambda *_args, **_kwargs: finalizer_calls.append(True),
    )

    report = headless.prepare_terminal_task_files(split.drive, task)
    stored = load_task_result(split.drive, task["id"])

    assert report["error"] == ""
    assert finalizer_calls == []
    assert stored["status"] == "failed"
    assert stored["artifact_status"] == "failed"
    assert stored["native_capture_pending"] is True
