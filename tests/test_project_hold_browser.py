"""Option A through real promotion, restore, serving supervisor and browser UI.

The restore models a Quit: the owner's Resume comes first (owner S1), then the
Project verification hold alone keeps the same task until its authority heals."""
from __future__ import annotations

import copy
import json
import os
import threading
from pathlib import Path

import pytest

from tests.test_project_admission_refusal_browser import (
    PROJECT, PROJECT_NAME, _open_project, _producers,
)
from tests.test_ui_smoke_playwright import direct_server_with_data  # noqa: F401
from tests.ui_chat_viewport_smoke import _CAPTURE_TEST_SOCKET

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]
TASK = "same-id-project-wait"
WAIT = "Waiting for Project verification"


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_project_hold_reload_reconnect_and_automatic_same_id_recovery(
        direct_server_with_data, monkeypatch, tmp_path, engine):  # noqa: F811
    from playwright.sync_api import sync_playwright

    from ouroboros import projects_registry as registry
    from ouroboros.project_dialogue import build_owner_message_ref
    from ouroboros.task_results import load_task_result
    from ouroboros.utils import utc_now_iso
    from supervisor import queue
    from supervisor.events_project_routing import _handle_promote_chat_to_task
    from supervisor.message_bus import log_chat
    from tests import fixtures_mock_llm

    server = direct_server_with_data
    url, root = server["url"], server["data_dir"]
    evidence = Path(os.environ.get("OUROBOROS_TEST_TEMP_ROOT") or tmp_path) / "project-hold" / engine
    evidence.mkdir(parents=True, exist_ok=True)
    print(f"HOLD_EVIDENCE={evidence}")
    facts, calls, entered = {}, [], threading.Event()
    fixtures_mock_llm.HOLD_RELEASE.clear()
    original_handler = fixtures_mock_llm._Handler.do_POST

    def completion(handler):
        calls.append(True)
        entered.set()
        fixtures_mock_llm.HOLD_RELEASE.wait(120)
        original_handler(handler)

    monkeypatch.setattr(fixtures_mock_llm._Handler, "do_POST", completion)

    def record(label, value):
        facts[label] = value
        (evidence / "facts.json").write_text(json.dumps(facts, indent=2, default=str), encoding="utf-8")

    server["stop_server"]()
    host = _producers(monkeypatch, root, server["repo_dir"])
    folder = tmp_path / "original-folder"
    folder.mkdir()
    registry.create_project(root, PROJECT, name=PROJECT_NAME, working_dir=str(folder))
    registry.create_project(root, "neighbour-room", name="Neighbour room")
    text = "Continue my original Project assignment when it becomes readable."
    ts, client_id = utc_now_iso(), "project-hold-owner-message"
    log_chat("in", 1, 1, text, ts=ts, source="web", client_message_id=client_id, require_write=True)
    ref = build_owner_message_ref(chat_id=1, client_message_id=client_id, ts=ts, text=text)
    answer = _handle_promote_chat_to_task({
        "task_id": TASK, "routing_token": TASK + "-token", "project_id": PROJECT,
        "chat_id": 1, "objective": text, "client_message_id": client_id,
        "source_ref": ref, "source_text": text,
    }, host.ctx)
    assert answer["status"] == "scheduled", answer
    [prepared] = copy.deepcopy(host.pending)
    assert queue.persist_queue_snapshot()
    path = registry._registry_path(root)
    original = path.read_bytes()
    data = json.loads(original)
    # The room's OWN routing authority is unreadable (the display lens still lists it); a
    # malformed unrelated room would not hold this work, so the neighbour stays healthy.
    next(row for row in data["projects"] if row["id"] == PROJECT)["routing_generation"] = "0"
    path.write_text(json.dumps(data), encoding="utf-8")
    host.pending.clear()
    queue.restore_pending_from_snapshot()
    assert [task["id"] for task in host.pending] == [TASK]
    assert host.pending[0]["_project_admission_restore_hold"]
    assert queue.persist_queue_snapshot()
    record("prepared", prepared)
    server["start_server"]()
    try:
        with sync_playwright() as pw:
            browser = getattr(pw, engine).launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 900})
            page.add_init_script(f"({_CAPTURE_TEST_SOCKET})()")
            try:
                # The restore above is what a Quit leaves: accepted work also waits for the
                # owner's explicit Resume (owner S1, quiz a524d73f). Project verification is
                # independent: after that Resume it still holds the task until it heals.
                held = page.request.get(url + f"/api/tasks/{TASK}").json()
                assert held["status"] == "scheduled" and held["reason_code"] == "saved_work_hold", held
                resumed = page.request.post(url + f"/api/tasks/{TASK}/resume")
                assert resumed.ok and resumed.json().get("ok"), resumed.text()
                record("resumed", resumed.json())
                page.goto(url, wait_until="domcontentloaded")
                for label in ("waiting", "reload", "reconnect"):
                    if label == "reload":
                        page.reload(wait_until="domcontentloaded")
                    elif label == "reconnect":
                        count = page.evaluate("() => window.__testSockets.length")
                        # A socket interruption, not SIGTERM (which intentionally Stops work).
                        page.evaluate("() => window.__testSockets.at(-1).close()")
                        page.wait_for_function("n => window.__testSockets.length > n && window.__testSockets.at(-1).readyState === 1",
                                               arg=count, timeout=60_000)
                    page.wait_for_function("() => window.__testSockets?.some(s => s.readyState === 1)")
                    _open_project(page)
                    panel = page.locator(f"#panel-pchat-{PROJECT}")
                    page.wait_for_function("([id, text]) => document.getElementById(id)?.innerText.includes(text)",
                                           arg=[f"panel-pchat-{PROJECT}", WAIT], timeout=30_000)
                    detail = page.request.get(url + f"/api/tasks/{TASK}").json()
                    census = page.request.get(url + "/api/state").json()["active_chat_activities"]
                    row = next(a for a in census if a["activity_id"] == TASK)
                    assert detail["status"] == "scheduled" and row["project_admission_hold"]["label"] == WAIT
                    assert detail["project_admission_hold"]["label"] == WAIT and not calls
                    assert not any(a["phase"] == "working" for a in census)
                    page.locator('#chat-messages [data-system-type="project_handoff"]').filter(has_text=WAIT).wait_for(
                        state="attached", timeout=180_000)
                    record(label, {"detail": detail, "census": census, "panel": panel.inner_text(),
                                   "main": page.locator("#chat-messages").inner_text()})
                    page.screenshot(path=str(evidence / f"{label}.png"))
                # The real queue view provides the explanation and existing Stop control.
                page.locator('[data-nav-page="dashboard"]').click()
                page.locator('[data-dashboard-tab="activity"]').click()
                held_row = page.locator(".activity-row").filter(has_text=text)
                held_row.wait_for(state="visible")
                assert WAIT in held_row.inner_text()
                assert held_row.locator('[data-act="task-control"]').is_visible()
                record("activity", held_row.inner_text())
                page.screenshot(path=str(evidence / "activity.png"))
                # Synthetic authority only: atomic restoration, with no second prepare/admit call.
                restored = path.with_suffix(".restored")
                restored.write_bytes(original)
                restored.replace(path)
                assert entered.wait(60), "the same accepted task never resumed"
                detail = page.request.get(url + f"/api/tasks/{TASK}").json()
                assert detail["status"] == "running" and not detail.get("project_admission_hold")
                assert detail["workspace_root"] == prepared["workspace_root"]
                assert len(calls) == 1
                record("recovered", {"detail": detail, "calls": len(calls),
                    "result": load_task_result(root, TASK, strict=True)})
                page.locator('[data-nav-page="chat"]').click()
                _open_project(page)
                page.wait_for_function("([id, text]) => !document.getElementById(id)?.innerText.includes(text)",
                                       arg=[f"panel-pchat-{PROJECT}", WAIT], timeout=30_000)
                page.screenshot(path=str(evidence / "recovered.png"))
            except Exception:
                page.screenshot(path=str(evidence / "zz-failure.png"))
                record("failure", {"body": page.locator("body").inner_text(), "calls": len(calls),
                    "state": page.request.get(url + "/api/state").json()})
                raise
            finally:
                fixtures_mock_llm.HOLD_RELEASE.set()
                browser.close()
    finally:
        fixtures_mock_llm.HOLD_RELEASE.set()
