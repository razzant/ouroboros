"""Post-task checkpoint keeps the existing wait card live after answer delivery."""

import json
import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient
from ouroboros import dialogue_store
from ouroboros.gateway.history import make_chat_history_endpoint
from ouroboros.task_results import load_task_result, write_task_result
from tests import _dialogue as dialogue_test
from tests.test_project_chat_continuity import _write_history_rows
from tests import test_subscription_setup_browser as ui_fixture

subscription_ui = ui_fixture.subscription_ui
pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]


@pytest.mark.parametrize("main_status", ["completed", "failed"])
def test_answered_direct_task_waits_through_task_done_and_reload(subscription_ui, tmp_path, main_status):
    ui, page = subscription_ui, subscription_ui["page"]
    task = "post-owner"
    row = {"wait_id": "post-light", "task_attempt": 1, "revision": 1, "role": "light",
           "model": "claudexor::codex=gpt-test", "source": "codex", "credential_profile_id": "personal",
           "credential_harness": "codex", "reason": "quota", "reset_at": "", "auto_continue": True,
           "state": "waiting", "worker_slot_held": False}
    state = {"post_task_synthesis": "running"}
    # Old rows deliberately have neither failed text nor a typed failure.
    # Only the canonical task-result file can restore the outcome after reload.
    _write_history_rows(tmp_path, task)
    dialogue_test.import_all(tmp_path)  # old chain files, as the upgrade imports them
    write_task_result(
        tmp_path, task, main_status, result="stored result",
        root_phase_checkpoint=state, model_waits={row["wait_id"]: row},
        outcome_axes={"execution": {"status": "failed" if main_status == "failed" else "completed"}},
        reason_code="main_execution_failed" if main_status == "failed" else "",
    )
    dialogue_store.append(tmp_path, "progress", {"type": "task_model_wait", "task_id": task,
                                                 "chat_id": 1, "ts": "2026-09-07T00:00:00Z", **row})
    app = Starlette(routes=[Route("/history", make_chat_history_endpoint(tmp_path))])
    sockets = []
    page.route_web_socket("**/ws", lambda socket: sockets.append(socket))
    page.add_init_script("""window.postTaskFrames = 0;
        window.WebSocket = class extends window.WebSocket {
            constructor(...args) { super(...args); this.addEventListener('message', () => { window.postTaskFrames += 1; }); }
        };""")

    def reply(route, body):
        route.fulfill(content_type="application/json", body=json.dumps(body))

    def detail():
        return load_task_result(tmp_path, task)

    def history(route):
        with TestClient(app) as client:
            response = client.get("/history?limit=20")
        route.fulfill(status=response.status_code, content_type="application/json", body=response.content)

    page.route("**/api/state", lambda route: reply(route, {"supervisor_ready": True, "projects": [],
        "active_chat_activities": [] if state["post_task_synthesis"] == "completed" else [{
            "activity_id": task, "kind": "managed_task", "chat_id": 1, "phase": "finalizing", "model_waits": {row["wait_id"]: row}}]}))
    page.route("**/api/chat/history*", history)
    page.route(f"**/api/tasks/{task}", lambda route: reply(route, detail()))
    page.goto(ui["url"] + "/")
    page.wait_for_selector('[data-wait-id="post-light"]')
    assert "worker slot" not in page.locator('.model-wait-footnote').inner_text()
    before_frame = page.evaluate("window.postTaskFrames")
    sockets[-1].send(json.dumps({"type": "log", "chat_id": 1,
                                 "data": {"type": "task_done", "ts": "2026-09-07T00:01:00Z", **detail()}}))
    page.wait_for_function("before => window.postTaskFrames > before", arg=before_frame)
    assert page.locator('[data-wait-id="post-light"]').is_visible()
    if main_status == "failed":
        assert page.get_by_text("Failed", exact=True).count() > 0
    ui["errors"].watch(page)  # the reloads may cut off the page's own reads (tests/ui_console_errors.py)
    page.reload()
    page.wait_for_selector('[data-wait-id="post-light"]')
    # The wait card can rejoin before the independent history request finishes.
    page.get_by_text("final answer", exact=True).wait_for(state="visible")
    if main_status == "failed":
        assert page.get_by_text("Failed", exact=True).first.is_visible()
        assert detail()["outcome_axes"]["execution"]["status"] == "failed"
    assert page.get_by_text("final answer", exact=True).is_visible()
    ui_fixture.capture(page, f"post-task-{main_status}-light-wait-after-reload")
    state["post_task_synthesis"] = "completed"
    write_task_result(tmp_path, task, main_status, root_phase_checkpoint=state)
    page.reload()
    page.get_by_text("final answer", exact=True).wait_for(state="visible")
    page.wait_for_function("() => !document.querySelector('[data-wait-id=post-light]')")
    assert page.locator('.model-waits').count() == 0
    if main_status == "failed":
        assert page.get_by_text("Failed", exact=True).first.is_visible()
