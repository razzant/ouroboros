"""Current physical retry identity from real Pause/census through Activity and room controls.

The browser serves the actual SPA and forwards history, queue, detail and Resume
to the actual gateway. Only unrelated settings/services and OS/model work use
the shared fixtures' doubles. Screenshots and full snapshots are portable evidence.
"""
import json
import os
from pathlib import Path

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros import owner_pause
from ouroboros.gateway.state import _chat_activities_snapshot_safe
from ouroboros.task_store import patch_document
from supervisor import queue
from tests.test_execution_clock_resume_handover import clock  # noqa: F401
from tests.test_retry_pause_identity import _assign, _loop_once, reaper  # noqa: F401
from tests.test_retry_resume_stop_authority import paused_retry
from tests.test_subscription_setup_browser import subscription_ui  # noqa: F401
from tests.test_swarm_host_admission import host  # noqa: F401
from tests._work_projection_shared import state_fields

pytestmark = pytest.mark.serial


@pytest.mark.parametrize("door", ["warm", "exact", "queued", "unsaved"])
@pytest.mark.parametrize("separate_late_work", [False, True])
def test_census_uses_physical_actor_and_keeps_actual_late_work(
        host, monkeypatch, clock, reaper, door, separate_late_work):  # noqa: F811
    task, _child, _plan = paused_retry(host, monkeypatch, clock, reaper, door)
    if separate_late_work:
        # Addressed late-phase producer fact (its upkeep still open under the Pause),
        # not an extra current queue actor.
        patch_document(host.root, "held", lambda row: {**row, "status": "completed",
            "root_phase_checkpoint": {"post_task_synthesis": "running"}})
    available = {"complete": True}
    rows = _chat_activities_snapshot_safe(host.root, direct_turns=[], availability=available)
    expected = [task["id"], *(["held"] if separate_late_work else [])]
    print("Physical census", {"door": door, "late_work": separate_late_work, "rows": rows})
    assert available["complete"] and [r["activity_id"] for r in rows] == expected
    assert all(r["phase"] == "budget_paused" for r in rows)
    if separate_late_work:
        # That upkeep still writes: the tree's Resume waits for it, then admits the same actor.
        refused = queue.resume_budget_paused_task(task["id"])
        assert not refused["ok"] and "late_phase_running" in json.dumps(refused), refused
        patch_document(host.root, "held", lambda row: {
            **row, "root_phase_checkpoint": {"post_task_synthesis": "degraded"}})
    assert queue.resume_budget_paused_task(task["id"])["ok"]


@pytest.mark.ui_browser
@pytest.mark.parametrize("door", ["warm", "exact", "queued", "pausing"])
@pytest.mark.parametrize("surface,width", [("activity", 1360), ("activity", 390), ("room", 1360), ("room", 390)])
def test_browser_resumes_current_retry_once(host, monkeypatch, clock, reaper, subscription_ui, door, surface, width):  # noqa: F811
    from ouroboros.gateway.history import make_chat_history_endpoint
    from ouroboros.gateway.tasks import api_task_get, api_task_resume, api_tasks_list

    task, _child, plan = paused_retry(host, monkeypatch, clock, reaper, door)
    tid, ui = task["id"], subscription_ui
    # This real actor belongs to Main in the rendered fixture.
    task["chat_id"] = 1
    for current in [*host.pending, *(meta["task"] for meta in queue.RUNNING.values())]:
        if current["id"] == tid:
            current["chat_id"] = 1
    patch_document(host.root, tid, lambda row: {**row, "chat_id": 1})
    (host.root / "logs/progress.jsonl").write_text(json.dumps({
        "ts": "2026-10-15T10:00:00+00:00", "type": "send_message", "task_id": tid, "is_progress": True,
        "direction": "out", "chat_id": 1, "text": "Current retry work", "content": "Current retry work",
        "cancelable": True}) + "\n")
    app = Starlette(routes=[Route("/api/chat/history", make_chat_history_endpoint(host.root)),
        Route("/api/tasks", api_tasks_list), Route("/api/tasks/{task_id}", api_task_get),
        Route("/api/tasks/{task_id}/resume", api_task_resume, methods=["POST"])])
    app.state.drive_root = host.root
    page = ui["page"]
    evidence = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR") or host.root / "screenshots")
    evidence.mkdir(parents=True, exist_ok=True)
    label, posts, snapshots = f"retry-{door}-{surface}-{width}", [], []
    with TestClient(app) as client:
        def forward(route):
            path = route.request.url.split(ui["url"], 1)[-1]
            response = (client.post(path, json=route.request.post_data_json) if route.request.method == "POST"
                        else client.get(path))
            if route.request.method == "POST":
                posts.append({"path": path, "status": response.status_code, "body": response.json()})
            route.fulfill(status=response.status_code, content_type="application/json", body=response.content)
        def census(route):
            # The real census and the server's own work projection of it, which the page reads.
            facts = {"supervisor_ready": True, "projects": [], "active_chat_activities_complete": True,
                     **state_fields(host.root, {}, direct_turns=[])}
            snapshots.append(facts)
            route.fulfill(content_type="application/json", body=json.dumps(facts))
        page.route("**/api/chat/history*", forward)
        page.route("**/api/tasks?*", forward)
        page.route("**/api/tasks/**", forward)
        page.route("**/api/state", census)
        page.goto(ui["url"] + "/")
        if surface == "activity":
            page.click('[data-nav-page="dashboard"]')
            page.click('[data-dashboard-tab="activity"]')
            area = page.locator('[data-activity-section="queue"]')
            trigger = area.locator(f'[data-id="{tid}"]')
            dead = area.locator('[data-id="held"]')
        else:
            area = page.locator(f'.chat-live-card[data-task-id="{tid}"]')
            trigger = area.locator('[data-cancel-run]')
            dead = page.locator('.chat-live-card[data-task-id="held"] [data-cancel-run]')
        page.set_viewport_size({"width": width, "height": 900})
        try:
            trigger.wait_for(state="visible", timeout=15000)
            assert trigger.count() == 1 and dead.count() == 0
            if surface == "activity":
                assert trigger.get_attribute("data-root") == "1"
                assert area.locator('[data-id="unselected"]').get_attribute("data-root") is None
            if door == "pausing":
                trigger.focus()
                page.keyboard.press("Enter")
                menu = page.locator('body > .task-control-menu')
                menu.wait_for(state="visible")
                assert menu.locator('[data-task-control="resume"]').count() == 0
                assert menu.locator('[data-task-control="pause"]').count() == 1
                assert area.locator('[data-resume-run]').count() == 0
                assert not posts and owner_pause.read_fence(host.root, "held")["state"] == "requested"
                page.keyboard.press("Escape")
                return
            # Activity uses its shared menu; the room already has a visible Resume button.
            if surface == "activity":
                trigger.focus()
                page.keyboard.press("Enter")
                menu = page.locator('body > .task-control-menu')
                resume = menu.locator('[data-task-control="resume"]')
                assert menu.locator('[data-task-control="pause"]').count() == 0
            else:
                resume = area.locator('[data-resume-run]')
            resume.wait_for(state="visible", timeout=5000)
            resume.focus()
            page.screenshot(path=str(evidence / f"{label}-paused.png"), animations="disabled")
            with page.expect_response(lambda r: r.request.method == "POST" and r.url.endswith(f"/{tid}/resume")):
                page.keyboard.press("Enter")
            assert len(posts) == 1 and posts[0]["path"] == f"/api/tasks/{tid}/resume", posts
            assert posts[0]["status"] == 200 and posts[0]["body"]["ok"], posts
            if door != "warm":
                [sent] = _assign(host, monkeypatch)
                assert sent["id"] == tid
                if door == "exact":
                    assert _loop_once(host, monkeypatch, sent, plan=plan)["kind"] == "pause"
            assert owner_pause.read_fence(host.root, "held")["state"] == "released"
            assert not ui["errors"]
        finally:
            page.screenshot(path=str(evidence / f"{label}-final.png"), animations="disabled")
            (evidence / f"{label}.json").write_text(json.dumps({"posts": posts, "snapshots": snapshots,
                "page_errors": ui["errors"], "html": area.inner_html(), "viewport": page.viewport_size,
                "browser_version": page.context.browser.version}, indent=2))
