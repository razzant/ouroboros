"""Native post-task wait state and real Stop ingress through the browser card."""

import json
import threading
from types import SimpleNamespace

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros import agent_task_pipeline as pipeline, dialogue_store, model_wait
from ouroboros.gateway.history import make_chat_history_endpoint
from ouroboros.gateway.state import _chat_activities_snapshot_safe
from ouroboros.gateway.tasks import api_task_cancel
from ouroboros.post_task_checkpoint import post_task_model_wait
from ouroboros.task_results import load_task_result
from supervisor import active_activity, workers
from tests.test_post_task_model_wait import phase as post_phase_fixture, until
from tests.test_subscription_setup_browser import capture, subscription_ui as ui_fixture

phase = post_phase_fixture
subscription_ui = ui_fixture
pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]


def test_browser_stops_native_post_task_wait_without_losing_answer(subscription_ui, phase, monkeypatch):
    from supervisor import queue as task_queue

    f, ui = phase, subscription_ui
    task_id, page = f.task["id"], ui["page"]
    f.task["_is_direct_chat"] = True
    monkeypatch.setattr(task_queue, "DRIVE_ROOT", f.root)
    monkeypatch.setattr(task_queue, "RUNNING", {})
    monkeypatch.setattr(task_queue, "PENDING", [])
    monkeypatch.setattr(workers, "WORKERS", {})
    pending = [{"type": "send_message", "task_id": task_id, "chat_id": 1, "text": "Already answered"}]
    actor = SimpleNamespace(_busy=True, _accepting_owner_messages=False)
    registry = active_activity.get_direct_activity_registry()

    def run():
        registry.register(task_id, 1, actor=actor)
        try:
            with model_wait.task_model_wait_scope(task=f.task, drive_root=f.root, event_queue=f.events,
                                                  worker_slot_held=False):
                pipeline._dispatch_root_post_task(f.env, f.task, "Already answered", f.events, pending,
                    {"rounds": 3}, {}, f.root / "logs", budget_drive_root="", split_drive=False,
                    project_scoped=False, project_task=False, parent_env=None, parent_task=None)
        finally:
            registry.unregister(task_id)

    # The earlier running progress frame already carried the ordinary host's cancel
    # authority; it and the delivered answer are stored rows. Then the real endpoint.
    dialogue_store.append(f.root, "progress", {
        "ts": "2026-08-17T10:00:00+00:00", "type": "send_message", "task_id": task_id, "is_progress": True,
        "direction": "out", "chat_id": 1, "user_id": 1, "text": "step one", "content": "step one", "cancelable": True})
    dialogue_store.append(f.root, "chat", {
        "ts": "2026-08-17T10:00:05+00:00", "direction": "out", "chat_id": 1, "user_id": 1, "text": "final answer",
        "task_id": task_id, "format": "markdown"})
    thread = threading.Thread(target=run)
    thread.start()
    responses = []
    app = Starlette(routes=[Route("/api/chat/history", make_chat_history_endpoint(f.root)),
                           Route("/api/tasks/{task_id}/cancel", api_task_cancel, methods=["POST"])])
    app.state.drive_root = f.root
    try:
        until(lambda: post_task_model_wait(f.root, task_id) and any(
            row.get("credential_harness") for row in post_task_model_wait(f.root, task_id).waits.values()))
        with TestClient(app) as client:
            def history(route):
                response = client.get("/api/chat/history?chat_id=1")
                route.fulfill(status=response.status_code, content_type="application/json", body=response.content)

            def cancel(route):
                response = client.post(f"/api/tasks/{task_id}/cancel", json=route.request.post_data_json)
                responses.append(response)
                route.fulfill(status=response.status_code, content_type="application/json", body=response.content)

            page.route("**/api/chat/history*", history)
            page.route("**/api/state", lambda route: route.fulfill(content_type="application/json", body=json.dumps({
                "supervisor_ready": True, "projects": [], "active_chat_activities": _chat_activities_snapshot_safe(f.root)})))
            page.route(f"**/api/tasks/{task_id}", lambda route: route.fulfill(content_type="application/json",
                       body=json.dumps(load_task_result(f.root, task_id))))
            page.route(f"**/api/tasks/{task_id}/cancel", cancel)
            page.goto(ui["url"] + "/")
            card = page.locator(f'.chat-live-card[data-task-id="{task_id}"]')
            card.locator('[data-wait-id]').first.wait_for(timeout=5000)
            assert "worker slot" not in card.locator('.model-wait-footnote').inner_text()
            capture(page, "native-post-task-wait-before-stop")
            card.locator('[data-cancel-run]').click()
            page.locator('[data-task-control="stop_now"]').click()
            page.wait_for_function("() => !document.querySelector('.task-control-menu')")
            assert responses and responses[-1].status_code in (200, 503)
            thread.join(5)
            assert not thread.is_alive() and len(f.engine.creates) == 1
            assert load_task_result(f.root, task_id)["root_phase_checkpoint"]["post_task_synthesis"] == "degraded"
            ui["errors"].watch(page)  # the reload may cut off the page's own reads (tests/ui_console_errors.py)
            page.reload()
            page.get_by_text("final answer", exact=True).wait_for()
            page.wait_for_function("() => !document.querySelector('[data-wait-id]')")
            capture(page, "native-post-task-wait-stopped-answer-preserved")
    finally:
        f.task["_skip_post_task_synthesis"] = True
        f.ready.set()
        thread.join(5)
        assert not thread.is_alive()
