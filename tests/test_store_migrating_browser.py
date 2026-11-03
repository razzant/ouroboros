"""Real client/CSS migration recovery over routed fixtures; no app server or daemon."""
import json
import pathlib
from urllib.parse import urlsplit

import pytest

from ouroboros import dialogue_store
from ouroboros.test_evidence import diagnostic_directory

pytestmark = pytest.mark.browser
ROOT = pathlib.Path(__file__).resolve().parents[1]
HTML = """<html data-theme="dark"><head><link rel="stylesheet" href="/static/ui.css">
<link rel="stylesheet" href="/static/style.css"></head><body>
<main id="chat" style="height:65vh"></main><section id="activity" style="height:35vh;overflow:auto"></section>
<script type="module">
import {createChatInstance} from '/static/modules/chat.js';
import {initActivity} from '/static/modules/activity.js';
const ws = {on: () => () => {}, send() {}, isConnected: () => true, ws: {readyState: 1}};
window.chat = createChatInstance({ws, state: {activePage: 'chat', projectChatIds: new Set(), unreadCount: 0},
 updateUnreadBadge() {}, chatId: 1, mountEl: document.querySelector('#chat'), asPanel: true,
 stateSnapshots: {begin: () => ({}), gate: async () => ({}), isCurrent: () => true, apply() {}}});
window.activity = initActivity({mount: document.querySelector('#activity'), ws});
void chat.refreshHistory({revision: 1}); void activity.refresh();
</script></body></html>"""


@pytest.mark.parametrize("width", [390, 1280])
def test_history_and_activity_recover_without_owner_action(tmp_path, width, monkeypatch):
    from playwright.sync_api import expect, sync_playwright
    from ouroboros.gateway.history import _assemble_history_response

    published = False
    data_root = tmp_path / "data"
    dialogue_store.append(data_root, "chat", {"direction": "in", "chat_id": 1,
        "text": "Saved owner message", "ts": "2026-10-11T09:00:00Z"})
    monkeypatch.setattr("ouroboros.gateway.state._chat_activities_snapshot_safe", lambda *_, **__: [])
    published_history = json.loads(_assemble_history_response(data_root, 1, 150, 150))
    assert published_history["window"]["complete"] is True

    def respond(route):
        path = urlsplit(route.request.url).path
        if path == "/":
            return route.fulfill(body=HTML, content_type="text/html")
        if path.startswith("/static/"):
            asset = ROOT / "web" / path.removeprefix("/static/")
            return route.fulfill(path=asset) if asset.is_file() else route.fulfill(status=404)
        if not published and path in {"/api/chat/history", "/api/tasks"}:
            return route.fulfill(status=503, json={"reason_code": "migrating"})
        data = {"queue": {"running": [], "pending": []}, "tasks": [], "active_direct_turns": [],
                "active_chat_activities": [], "active_chat_activities_complete": True,
                "bg_consciousness_enabled": False}
        if path == "/api/chat/history":
            data = published_history
        route.fulfill(body=json.dumps(data), content_type="application/json")

    with sync_playwright() as pw:
        chrome = pathlib.Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
        browser = pw.chromium.launch(executable_path=str(chrome) if chrome.is_file() else None)
        try:
            page = browser.new_page(viewport={"width": width, "height": 900})
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.route("**/*", respond)
            page.goto("http://fixa.test/")
            expect(page.get_by_text("Saved history is being upgraded…", exact=True)).to_be_visible()
            expect(page.get_by_text("Saved tasks are being upgraded…", exact=True)).to_be_visible()
            expect(page.get_by_role("button", name="Retry loading messages")).to_be_hidden()
            before = diagnostic_directory(tmp_path, "store-migrating") / f"migrating-{width}.png"
            page.screenshot(path=str(before), full_page=True)
            published = True
            expect(page.get_by_text("Saved owner message", exact=True)).to_be_visible()
            expect(page.get_by_text("Nothing running or queued", exact=False)).to_be_visible()
            expect(page.get_by_text("Saved history is being upgraded…", exact=True)).to_be_hidden()
            expect(page.get_by_text("Some saved history could not be loaded.", exact=True)).to_be_hidden()
            expect(page.get_by_role("button", name="Retry loading messages")).to_be_hidden()
            expect(page.get_by_text("Beginning of saved history", exact=True)).to_be_visible()
            after = diagnostic_directory(tmp_path, "store-migrating") / f"published-{width}.png"
            page.screenshot(path=str(after), full_page=True)
            page.evaluate("chat.destroy(); activity.destroy()")
            assert not errors
            print(f"Visual evidence: {before} {after}")
        finally:
            browser.close()
