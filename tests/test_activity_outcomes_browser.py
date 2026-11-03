"""Dashboard → Activity shows a known outcome first, over real rows, census and queue.

The rows come from the actual pipeline writer (``_store_task_result``) and the real
queue publication; the page loads the production Activity module and stylesheets
against routes that call the gateway's own queue and census readers. Control POSTs
are answered at the route (Stop and Resume keep their own server suites): what is
checked here is that a click or key on a row reaches that row's endpoint with the
same arguments it always sent, and that partial or failed reads keep the queue's rows
without inventing activity. Screenshots (light and dark, desktop and narrow, per engine) go to
``OUROBOROS_UI_EVIDENCE_DIR`` before anything is asserted, so a failing run still
leaves its picture.
"""
from __future__ import annotations

import json
import os
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from types import SimpleNamespace

import pytest

from tests._budget_pause_exact_helpers import _install_queue
from tests._work_projection_shared import state_fields, with_work_fields

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]
WEB = Path(__file__).resolve().parents[1] / "web"
HTML = """<!doctype html><html lang="en" class="ouro-ui"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="/ui.css"><link rel="stylesheet" href="/style.css">
<link rel="stylesheet" href="/model_wait.css"></head><body>
<div class="dashboard-shell"><div class="dashboard-panels">
<section class="dashboard-panel active" id="activity"></section></div></div>
<script type="module">
import {initActivity} from '/modules/activity.js';
document.documentElement.dataset.theme = new URLSearchParams(location.search).get('theme') || 'dark';
window.activity = initActivity({mount: document.getElementById('activity'), ws: {send() {}}});
await window.activity.refresh();
window.ready = true;
</script></body></html>"""


def _seed(root, monkeypatch):
    from ouroboros.agent_task_pipeline import _store_task_result
    from ouroboros.task_results import write_task_result

    queue, _state, workers = _install_queue(root, monkeypatch)
    env = SimpleNamespace(drive_root=root, repo_dir=root)
    for task_id, title, usage, text in (
        ("failed-root", "Quarterly report",
         {"rounds": 1, "execution_status": "infra_failed", "reason_code": "provider_unavailable"},
         "Preserved incomplete work."),
        ("done-root", "Release notes", {"rounds": 1}, "Release notes are ready."),
        ("sleep-root", "Answered, then asleep", {"rounds": 1}, "The digest is sent."),
    ):
        task = {"id": task_id, "type": "task", "chat_id": 1, "root_task_id": task_id, "title": title, "text": title}
        write_task_result(root, task_id, "running", root_task_id=task_id, chat_id=1, title=title)
        # The real terminal writer: the answer is stored, root post-task work is still owed.
        _store_task_result(env, task, text, usage, {"tool_calls": [], "reasoning_notes": []}, review_evidence={})
        workers.RUNNING[task_id] = {"task": task, "started_at": time.time() - 42, "attempt": 1}
    # A warm model sleep is a physical queue fact: the work projection must keep it beside the answer.
    workers.RUNNING["sleep-root"]["owner_wait"] = {
        "state": "waiting", "reason": "sleep", "sleep": {"mode": "warm"}, "task_attempt": 1}
    child = {"id": "child-1", "type": "task", "chat_id": 1, "root_task_id": "failed-root",
             "parent_task_id": "failed-root", "delegation_role": "subagent", "title": "Collect figures"}
    write_task_result(root, "child-1", "running", root_task_id="failed-root", parent_task_id="failed-root",
                      delegation_role="subagent")
    workers.RUNNING["child-1"] = {"task": child, "started_at": time.time() - 30, "attempt": 1}
    write_task_result(root, "paused-root", "scheduled", root_task_id="paused-root", chat_id=1)
    workers.PENDING.append({"id": "paused-root", "type": "task", "chat_id": 1, "root_task_id": "paused-root",
                            "title": "Budget-paused draft", "_budget_pause": {"reason": "budget"}})
    assert queue.persist_queue_snapshot("activity outcome fixture")


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_activity_outcome_rows_and_their_controls_in_a_real_browser(tmp_path, monkeypatch, engine):
    from playwright.sync_api import expect, sync_playwright
    from ouroboros.gateway.tasks import _queue_snapshot
    from ouroboros.outcomes import public_task_result
    from ouroboros.task_results import load_task_result

    _seed(tmp_path, monkeypatch)
    posts = []
    reads = {"state": "complete", "queue": "ok"}

    def api(route):
        request = route.request
        path = request.url.split("/api/", 1)[1]
        if request.method == "POST":
            posts.append({"path": path, "body": request.post_data_json})
            body = {"ok": True, "status": "accepted"}
        elif path.startswith("tasks?"):
            if reads["queue"] == "failed":
                return route.fulfill(status=503, content_type="application/json", body='{"error": "offline"}')
            body = {"tasks": [], "queue": _queue_snapshot(tmp_path)}
        elif path == "state":
            if reads["state"] == "failed":
                return route.fulfill(status=503, content_type="application/json", body='{"error": "offline"}')
            # The real census and the server's own work projection of it (its Resume controls included).
            availability = {"complete": True}
            fields = state_fields(tmp_path, availability=availability)
            if reads["state"] == "partial":  # one readable row; the rest of the census is unknown
                fields = with_work_fields({"active_chat_activities_complete": False, "active_chat_activities": [
                    r for r in fields["active_chat_activities"] if r["activity_id"] == "failed-root"]})
                availability["complete"] = False
            body = {"bg_consciousness_enabled": False, **fields,
                    "active_chat_activities_complete": availability["complete"]}
        elif path == "schedules":
            body = {"tasks": []}
        elif path.startswith("tasks/"):
            body = public_task_result(load_task_result(tmp_path, path.split("/", 1)[1]) or {})
        else:
            return route.fulfill(status=404, body="{}")
        route.fulfill(status=200, content_type="application/json", body=json.dumps(body))

    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            if self.path.startswith("/fixture"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(HTML.encode())
            else:
                super().do_GET()

    evidence = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR") or tmp_path / "evidence")
    evidence.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(WEB)))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/fixture"
    try:
        with sync_playwright() as pw:
            browser = getattr(pw, engine).launch(headless=True)
            try:
                for theme, width, height in (("light", 1200, 620), ("dark", 1200, 620),
                                             ("light", 390, 780), ("dark", 390, 780)):
                    page = browser.new_page(viewport={"width": width, "height": height}, color_scheme=theme)
                    errors = []
                    page.on("pageerror", lambda exc: errors.append(str(exc)))
                    page.route("**/api/**", api)
                    page.goto(f"{url}?theme={theme}")
                    page.wait_for_function("window.ready === true")
                    queue = page.locator('[data-activity-section="queue"]')
                    page.screenshot(path=str(evidence / f"activity-outcomes-{engine}-{theme}-{width}.png"))
                    assert errors == []

                    def row(task_id):
                        return queue.locator(".activity-row").filter(has=page.locator(f'[data-id="{task_id}"]'))

                    failed = row("failed-root").locator("[data-activity-phase]")
                    expect(failed).to_have_text("Failed")
                    expect(failed).to_have_attribute("role", "status")
                    expect(failed).to_have_attribute("aria-label", "Task status: Failed, Finalizing…")
                    expect(row("failed-root").locator(".chat-live-phase-secondary")).to_have_text("Finalizing…")
                    assert row("failed-root").inner_text().count("Finalizing…") == 1
                    expect(row("failed-root")).to_contain_text("1 helper still working")
                    expect(row("done-root").locator("[data-activity-phase]")).to_have_text("Done")
                    # The queue's own wait outranks the census's generic finalization.
                    expect(row("sleep-root").locator(".chat-live-phase-secondary")).to_have_text("Waiting")
                    expect(row("sleep-root")).to_contain_text("Sleeping until it is woken")
                    # Its own row has no result yet: the child keeps its occupancy word.
                    expect(row("child-1").locator("[data-activity-phase]")).to_have_count(0)
                    expect(row("child-1")).to_contain_text("Working")
                    expect(row("paused-root")).to_contain_text("Paused")
                    expect(row("paused-root")).to_contain_text("Spending limit reached")
                    # The outcome chip is drawn in the error pair with the failed dot, the second fact in the quiet
                    # meta ink; Stop… is the neutral shared trigger, never a danger button.
                    colours = page.evaluate("""() => {
                        const probe = document.createElement('span');
                        document.body.append(probe);
                        const ink = (value) => { probe.style.color = value; return getComputedStyle(probe).color; };
                        const chip = document.querySelector('[data-id="failed-root"]').closest('.activity-row')
                            .querySelector('[data-activity-phase]');
                        const mark = chip.querySelector(':scope > .status-mark');
                        const stop = document.querySelector('[data-act="task-control"][data-id="failed-root"]');
                        const scroll = document.querySelector('.activity-scroll');
                        const out = {chip: getComputedStyle(chip).color, chipBg: getComputedStyle(chip).backgroundColor,
                            second: getComputedStyle(chip.nextElementSibling).color, error: ink('var(--status-error-fg)'),
                            errorBg: ink('var(--status-error-bg)'), meta: ink('var(--text-meta)'),
                            mark: `${mark.dataset.shape}/${mark.dataset.hue}`, markInk: getComputedStyle(mark).color,
                            red: ink('var(--red)'), stop: stop.className,
                            overflow: scroll.scrollWidth - scroll.clientWidth};
                        probe.remove();
                        return out;
                    }""")
                    assert colours["chip"] == colours["error"] and colours["second"] == colours["meta"], colours
                    assert colours["chipBg"] == colours["errorBg"], colours
                    assert (colours["mark"], colours["markInk"]) == ("dot/failed", colours["red"]), colours
                    assert colours["stop"] == "btn btn-xs btn-default", colours
                    assert colours["overflow"] <= 0, colours
                    page.close()

                page = browser.new_page(viewport={"width": 1200, "height": 620})
                page.route("**/api/**", api)
                menu = page.locator("body > .task-control-menu")

                def load(theme="dark"):
                    # A fresh page per action: Activity ignores clicks while its last action refreshes.
                    page.goto(f"{url}?theme={theme}")
                    page.wait_for_function("window.ready === true")

                def open_menu(task_id):
                    load()
                    page.locator(f'[data-act="task-control"][data-id="{task_id}"]').click()
                    expect(menu).to_be_visible()
                    return menu.locator("[data-task-control]").all_text_contents()

                def settle(count):
                    deadline = time.monotonic() + 10
                    while len(posts) < count and time.monotonic() < deadline:
                        page.wait_for_timeout(50)
                    page.wait_for_function(
                        "!document.querySelector('[data-activity-section=queue]').hasAttribute('aria-busy')")

                # Keyboard: the trigger opens the same menu on the first action; Escape continues the run.
                load()
                trigger = page.locator('[data-act="task-control"][data-id="failed-root"]')
                trigger.focus()
                page.keyboard.press("Enter")
                expect(menu).to_be_visible()
                assert page.evaluate("document.activeElement.dataset.taskControl") == "finalize"
                page.keyboard.press("Escape")
                expect(menu).to_have_count(0)
                assert posts == []

                assert open_menu("failed-root") == ["Wrap up", "Hurry up", "Pause", "Stop now"]
                page.screenshot(path=str(evidence / f"activity-outcomes-{engine}-failed-menu.png"))
                menu.locator('[data-task-control="stop_now"]').click()
                settle(1)
                assert open_menu("child-1") == ["Wrap up", "Hurry up", "Stop now"]
                page.keyboard.press("Escape")
                assert open_menu("paused-root") == ["Resume", "Stop now"]
                menu.locator('[data-task-control="resume"]').click()
                settle(2)
                page.screenshot(path=str(evidence / f"activity-outcomes-{engine}-after-controls.png"))
                assert [post["path"] for post in posts] == ["tasks/failed-root/cancel", "tasks/paused-root/resume"]
                assert posts[0]["body"]["cascade"] is True and "stop_policy" not in posts[0]["body"]
                assert posts[0]["body"]["stop_action_id"].startswith("stop-")
                expect(page.locator('[data-id="failed-root"]').locator("xpath=../..").locator(
                    "[data-activity-phase]")).to_have_text("Failed")

                queue = page.locator('[data-activity-section="queue"]')
                # A partial census describes its own rows; missing work projections stay unconfirmed.
                reads["state"] = "partial"
                load()
                expect(queue.locator("[data-activity-phase]")).to_have_count(1)
                expect(queue.locator("[data-activity-phase]")).to_have_text("Failed")
                expect(page.locator('[data-id="done-root"]').locator("xpath=../..")).to_contain_text("Activity unconfirmed")
                expect(queue).not_to_contain_text("Nothing running")
                # An unreadable census proves nothing: no outcome chip, every queue row and control stays.
                reads["state"] = "failed"
                load()
                expect(queue.locator("[data-activity-phase]")).to_have_count(0)
                expect(queue.locator('[data-act="task-control"]')).to_have_count(5)
                page.screenshot(path=str(evidence / f"activity-outcomes-{engine}-census-unreadable.png"))
                # An unreadable queue is unknown, never empty.
                reads["queue"] = "failed"
                load()
                expect(queue.locator(".activity-read-status")).to_contain_text("Current state is unknown")
                expect(queue).not_to_contain_text("Nothing running")
                (evidence / f"activity-outcomes-{engine}-posts.json").write_text(json.dumps(posts, indent=2),
                                                                                encoding="utf-8")
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()
