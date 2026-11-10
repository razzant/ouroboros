"""Activity census rows use real direct actors and the shared task controls."""
from __future__ import annotations

import json
import os
import time
import threading
from pathlib import Path

import pytest

from tests.test_ui_smoke_playwright import direct_server_with_data as direct_server_with_data
from tests.ui_chat_viewport_smoke import _CAPTURE_TEST_SOCKET


@pytest.mark.ui_browser
@pytest.mark.parametrize("action", ["hurry", "finalize", "stop_now"])
def test_activity_direct_turn_reaches_shared_control_endpoint(direct_server_with_data, monkeypatch, action):
    from playwright.sync_api import sync_playwright
    from tests import fixtures_mock_llm

    monkeypatch.setattr(fixtures_mock_llm, "HOLD_SECONDS", 90)
    fixtures_mock_llm.HOLD_RELEASE.clear()
    entered = threading.Event()

    def held_completion(handler):
        payload = json.loads(handler.rfile.read(int(handler.headers.get("Content-Length", 0))))
        entered.set()
        fixtures_mock_llm.HOLD_RELEASE.wait(90)
        streaming = payload.get("stream") is True
        answer = {"id": "held-completion", "choices": [{"index": 0, "finish_reason": "stop",
            "delta" if streaming else "message": {"role": "assistant", "content": "OK"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
        content = ("data: " + json.dumps(answer) + "\n\ndata: [DONE]\n\n" if streaming else json.dumps(answer)).encode()
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream" if streaming else "application/json")
        handler.send_header("Content-Length", str(len(content)))
        handler.end_headers()
        handler.wfile.write(content)

    monkeypatch.setattr(fixtures_mock_llm._Handler, "do_POST", held_completion)
    url = direct_server_with_data["url"]
    evidence = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", direct_server_with_data["data_dir"].parent))
    evidence.mkdir(parents=True, exist_ok=True)
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            try:
                page = browser.new_page(viewport={"width": 1440, "height": 900})
                page.add_init_script(f"({_CAPTURE_TEST_SOCKET})()")
                page.goto(url, wait_until="domcontentloaded")
                page.wait_for_function("() => window.__testSockets?.some(s => s.readyState === 1)")
                page.fill("#chat-input", "Respond with exactly OK")
                page.click("#chat-send")
                deadline = time.monotonic() + 30
                actor = None
                while time.monotonic() < deadline and (actor is None or not entered.is_set()):
                    state = page.request.get(url + "/api/state").json()
                    actor = next((a for a in state.get("active_chat_activities", []) if a["kind"] == "direct_chat"), None)
                    if actor is None or not entered.is_set():
                        page.wait_for_timeout(100)
                assert actor and entered.is_set(), state
                task_id = actor["activity_id"]
                queue = page.request.get(url + "/api/tasks?queue_only=1").json()["queue"]
                # ARCHITECTURE "Conversation lane": an owner turn is admitted by routing with its RUNNING row
                # published first, never PENDING, so the queue names it once, as a direct turn of the lane.
                rows = {part: [q for q in queue[part] if (q.get("id") or q.get("task", {}).get("id")) == task_id]
                        for part in ("running", "pending")}
                assert len(rows["running"]) == 1 and not rows["pending"], (rows, queue)
                owner = rows["running"][0]["task"]["_execution_owner"]
                assert (owner["kind"], owner["lane"]) == ("direct", "conversation"), owner
                page.click('[data-nav-page="dashboard"]')
                page.click('[data-dashboard-tab="activity"]')
                button = page.locator(f'[data-activity-section="queue"] [data-id="{task_id}"]')
                button.wait_for(state="visible")
                assert button.count() == 1
                row = button.locator("xpath=../..")
                text = row.inner_text()
                # The queue names this turn (the census adds only identities the queue lacks): by the owner's own
                # request, as a live row of the lane, never queued.
                assert "Respond with exactly OK" in text and "Queued" not in text, text
                assert "Nothing running" not in page.locator('[data-activity-section="queue"]').inner_text()
                button.click()
                menu = page.locator('body > .task-control-menu')
                menu.wait_for(state="visible")
                # A direct turn is a ROOT: it offers the owner's whole-tree Pause (Batch4 5A) too.
                assert menu.locator('[data-task-control]').all_text_contents() == [
                    "Wrap up", "Hurry up", "Pause", "Stop now"]
                page.screenshot(path=str(evidence / f"activity-{action}.png"), full_page=True)
                endpoint = "hurry" if action == "hurry" else "cancel"
                # Direct actors stop cooperatively. Let the held fake provider
                # complete after cancellation is submitted, so the real endpoint
                # can observe settlement instead of an artificial never-returning call.
                release = None
                if action == "stop_now":
                    def release_after_request(request):
                        nonlocal release
                        if request.method == "POST" and request.url.endswith(f"/{task_id}/cancel"):
                            release = threading.Timer(0.3, fixtures_mock_llm.HOLD_RELEASE.set)
                            release.start()
                    page.on("request", release_after_request)
                with page.expect_response(lambda r: f"/api/tasks/{task_id}/{endpoint}" in r.url and r.request.method == "POST") as response:
                    menu.locator(f'[data-task-control="{action}"]').click()
                if release is not None:
                    release.join(timeout=1)
                receipt = response.value
                body = receipt.json()
                assert receipt.status in (200, 202), body
                assert body.get("ok") is True, body
                request = receipt.request.post_data_json
                if action == "hurry":
                    assert request["request_id"], request
                else:
                    assert request.get("stop_policy", "immediate") == ("immediate" if action == "stop_now" else "finalize_then_cancel")
                (evidence / f"activity-{action}.json").write_text(json.dumps({
                    "actor": actor, "queue": queue, "status": receipt.status, "request": request, "response": body,
                }, indent=2), encoding="utf-8")
                fixtures_mock_llm.HOLD_RELEASE.set()
                deadline = time.monotonic() + 30
                while time.monotonic() < deadline:
                    state = page.request.get(url + "/api/state").json()
                    if state.get("active_chat_activities_complete") is True and not any(
                            a["activity_id"] == task_id for a in state["active_chat_activities"]):
                        break
                    page.wait_for_timeout(100)
                else:
                    raise AssertionError(state)
            finally:
                browser.close()
    finally:
        fixtures_mock_llm.HOLD_RELEASE.set()


_KEPT_SERVER = r'''
import os, socket, sys
sock = socket.socket()
sock.bind(("127.0.0.1", 0))
sock.listen(16)
port = sock.getsockname()[1]
print(f"  VITE ready\n  -> Local:   http://127.0.0.1:{port}/", flush=True)
with open(sys.argv[1] + ".tmp", "w") as handle:
    handle.write(f"{port} {os.getpid()}")
os.replace(sys.argv[1] + ".tmp", sys.argv[1])
while True:
    conn, _ = sock.accept()
    conn.sendall(b"pong\n")
    conn.close()
'''

# The producer: a short-lived process with the server's own environment, as a worker that kept its services and
# then ended. Its services stay, orphaned, in the server's custody; it writes the tasks' records first.
_KEPT_PRODUCER = r'''
import json, os, pathlib, subprocess, sys
from ouroboros.process_census import LAUNCH_SCOPE_PREFIX, SCOPE_VALUE, mint_marker
from ouroboros.process_custody import spawn_supervised
from ouroboros.task_results import write_task_result
data, server, ports = pathlib.Path(sys.argv[1]), sys.argv[2], pathlib.Path(sys.argv[3])
for task_id, name, title in json.loads(sys.argv[4]):
    write_task_result(data, task_id, "completed", title=title)
    log = data / "services" / task_id / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    marker = mint_marker(LAUNCH_SCOPE_PREFIX)
    with log.open("ab") as handle:
        spawn_supervised([sys.executable, server, str(ports / name)], drive_root=data, purpose=f"service:{name}",
                         scope="session", owner_task_id=task_id, scope_marker=marker,
                         env={**os.environ, marker: SCOPE_VALUE}, stdout=handle, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, start_new_session=True)
'''


@pytest.mark.ui_browser
def test_activity_background_shows_kept_services_after_their_producer_ended_and_stops_one(direct_server_with_data):
    """The real SPA over the real server: services a task kept, whose producer has ended, appear under Background with
    their task, state, since and printed address; Stop (after its confirmation) ends that one launch with a receipt, the
    other keeps running, the log stays. Frames at the owner's 1100x750 and a narrow long-name width, both themes."""
    import subprocess

    from playwright.sync_api import sync_playwright
    from tests._launch_stop_shared import _answers, _end
    from tests.test_ui_smoke_playwright import _fixture_python

    server = direct_server_with_data
    url, data = server["url"], server["data_dir"]
    evidence = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", data.parent)) / "kept-services"
    evidence.mkdir(parents=True, exist_ok=True)
    work = data.parent / "kept-fixture"
    work.mkdir()
    (work / "server.py").write_text(_KEPT_SERVER, encoding="utf-8")
    long_name = "storybook-preview-for-the-quarterly-retention-dashboard-redesign.v2"
    kept = [("task-preview", "preview", "Banner preview"),
            ("task-long", long_name, "Rebuild the onboarding retention dashboard with the new banner states and "
                                     "the long localized copy for every supported market")]
    produced = subprocess.run([_fixture_python(), "-c", _KEPT_PRODUCER, str(data), str(work / "server.py"), str(work),
                               json.dumps(kept)], cwd=server["repo_dir"], env=server["env"], capture_output=True,
                              text=True, timeout=120)
    assert produced.returncode == 0, produced.stderr
    identities = {}
    from ouroboros.process_census import measure_birth

    deadline = time.monotonic() + 60
    for _task, name, _title in kept:
        while not (work / name).exists():
            assert time.monotonic() < deadline, f"{name} never started"
            time.sleep(0.05)
        port, pid = (int(part) for part in (work / name).read_text(encoding="utf-8").split())
        identities[name] = (port, pid, measure_birth(pid))
    engine_name = os.environ.get("OUROBOROS_UI_BROWSER_ENGINE", "webkit")
    try:
        listed = None
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            import urllib.request

            with urllib.request.urlopen(url + "/api/services/kept", timeout=10) as reply:
                listed = json.loads(reply.read())
            if len(listed.get("services") or []) == 2:
                break
            time.sleep(0.2)
        (evidence / "kept-before.json").write_text(json.dumps(listed, indent=2), encoding="utf-8")
        assert listed["complete"] is True and {item["name"] for item in listed["services"]} == {"preview", long_name}
        with sync_playwright() as pw:
            engine = getattr(pw, engine_name)
            browser = engine.launch()
            try:
                frames = {}
                for theme in ("dark", "light"):
                    for width, height in ((1100, 750), (390, 844)):
                        context = browser.new_context(viewport={"width": width, "height": height}, device_scale_factor=2,
                                                      color_scheme=theme, locale="en-GB")
                        page = context.new_page()
                        page.goto(url, wait_until="domcontentloaded")
                        if width < 980:
                            page.locator(".page.active [data-mobile-nav-toggle]").click()
                            page.wait_for_selector("#primary-sidebar.open")
                        page.click('#primary-sidebar [data-nav-page="dashboard"]')
                        page.click('[data-dashboard-tab="activity"]')
                        rows = page.locator('[data-activity-section="background"] [data-kept-service]')
                        rows.first.wait_for(state="visible")
                        assert rows.count() == 2
                        section = page.locator('[data-activity-section="background"]')
                        section.scroll_into_view_if_needed()
                        shot = evidence / f"{engine_name}-{theme}-{width}x{height}-background.png"
                        page.screenshot(path=str(shot))
                        section.screenshot(path=str(evidence / f"{engine_name}-{theme}-{width}-section@2x.png"))
                        frames[f"{theme}-{width}"] = page.evaluate("""() => [...document.querySelectorAll(
                            '[data-activity-section="background"] [data-kept-service]')].map(row => {
                            const title = row.querySelector('.activity-name'), meta = row.querySelector('.activity-sub');
                            const stop = row.querySelector('[data-act="kept-stop"]');
                            const r = row.getBoundingClientRect(), b = stop.getBoundingClientRect();
                            return {text: row.innerText, row_right: r.right, viewport: innerWidth,
                                    stop_inside: b.left >= r.left && b.right <= r.right + 0.5, stop_w: b.width, stop_h: b.height,
                                    title_overflow: title.scrollWidth > title.clientWidth + 1,
                                    meta_overflow: meta.scrollWidth > meta.clientWidth + 1,
                                    title_ws: getComputedStyle(title).whiteSpace, meta_ws: getComputedStyle(meta).whiteSpace};
                        })""")
                        for row in frames[f"{theme}-{width}"]:
                            assert row["row_right"] <= row["viewport"] + 0.5 and row["stop_inside"], row
                            assert "Kept by a task · running" in row["text"] and "since " in row["text"], row
                            assert "http://127.0.0.1:" in row["text"], row
                        context.close()
                (evidence / "kept-frames.json").write_text(json.dumps(frames, indent=2), encoding="utf-8")

                context = browser.new_context(viewport={"width": 1100, "height": 750}, color_scheme="dark", locale="en-GB")
                page = context.new_page()
                page.goto(url, wait_until="domcontentloaded")
                page.click('[data-nav-page="dashboard"]')
                page.click('[data-dashboard-tab="activity"]')
                preview = next(item for item in listed["services"] if item["name"] == "preview")
                page.locator(f'[data-act="kept-stop"][data-id="{preview["launch_id"]}"]').click()
                dialog = page.get_by_role("dialog")
                dialog.wait_for(state="visible")
                page.screenshot(path=str(evidence / f"{engine_name}-dark-1100x750-confirm.png"))
                words = dialog.inner_text()
                # Said before the action and true while its task may still run: what Stop will do, not what it did.
                assert "Stop preview" in words and "ends its processes" in words and "after it ended" not in words
                with page.expect_response(lambda r: r.url.endswith(f"/api/services/kept/{preview['launch_id']}/stop")) as answer:
                    dialog.get_by_role("button", name="Stop").click()
                receipt = answer.value.json()
                request = answer.value.request.post_data_json
                assert answer.value.status == 200 and receipt["outcome"] == "stopped", receipt
                # This producer spawned it without the start path's declaration: nothing captured, not done.
                assert receipt["outputs"]["state"] == "unavailable" and receipt["ok"] is False, receipt
                assert request == {"reason": "the owner stopped it from Activity › Background"}
                page.locator('.toast').filter(has_text="Stopped preview").wait_for(state="visible")
                page.locator(f'[data-kept-service="{preview["launch_id"]}"]').wait_for(state="detached")
                page.screenshot(path=str(evidence / f"{engine_name}-dark-1100x750-after-stop.png"))
                assert not _answers(identities["preview"][0]) and _answers(identities[long_name][0])
                assert (data / "services" / "task-preview" / "preview.log").exists(), "its log is kept"
                (evidence / "kept-stop-receipt.json").write_text(json.dumps({"request": request, "receipt": receipt},
                                                                           indent=2), encoding="utf-8")
                # A synthetic owed declaration exercises the real inventory and rendered recovery state;
                # actual start/capture/retirement transitions are exercised in test_launch_stop.
                declaration = data / "services" / "kept_outputs" / "task-preview" / f"{preview['launch_id']}.kept.json"
                declaration.parent.mkdir(parents=True, exist_ok=True)
                declaration.write_text(json.dumps({"schema": 1, "launch_id": preview["launch_id"],
                    "task_id": "task-preview", "name": "preview", "outputs": ["pending.txt"],
                    "cwd": str(work), "declared_at": preview["since"]}), encoding="utf-8")
                page.reload(wait_until="domcontentloaded")
                page.click('[data-nav-page="dashboard"]')
                page.click('[data-dashboard-tab="activity"]')
                owed = page.locator(f'[data-kept-service="{preview["launch_id"]}"]')
                owed.get_by_role("button", name="Save outputs preview", exact=True).wait_for(state="visible")
                assert "ended, declared outputs not captured yet" in owed.inner_text()
                page.screenshot(path=str(evidence / f"{engine_name}-dark-1100x750-capture-owed.png"))
                owed.get_by_role("button", name="Save outputs preview", exact=True).click()
                dialog = page.get_by_role("dialog")
                assert "rechecks it, stops any remaining processes" in dialog.inner_text()
                page.screenshot(path=str(evidence / f"{engine_name}-dark-1100x750-capture-confirm.png"))
                dialog.get_by_role("button", name="Cancel", exact=True).click()
                page.set_viewport_size({"width": 390, "height": 844})
                owed.scroll_into_view_if_needed()
                page.screenshot(path=str(evidence / f"{engine_name}-dark-390x844-capture-owed.png"))
                context.close()
            finally:
                browser.close()
    finally:
        for port, pid, birth in identities.values():
            _end(pid, birth)
