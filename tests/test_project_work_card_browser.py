"""Production Main cards and motion over static gateway data, without a runtime."""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
from threading import Thread
from urllib.parse import urlparse

import pytest

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]
WEB = Path(__file__).resolve().parents[1] / "web"
BOOT = """<script type="module">
import {createChatInstance} from '/static/modules/chat.js';
const handlers=new Map();let connected=true;
const ws={on(type,fn){const rows=handlers.get(type)||[];rows.push(fn);handlers.set(type,rows);return()=>{};},
    isConnected:()=>connected,send(){}};
window.emit=(type,row)=>(handlers.get(type)||[]).forEach(fn=>fn(row));
window.connection=value=>{connected=value;window.emit(value?'open':'close');};
window.chat=createChatInstance({ws,state:{activePage:'chat',projectChatIds:new Set([7,8]),unreadCount:0},
    updateUnreadBadge(){},openSettingsTab(){},openDashboardTab(){},chatId:1,
    mountEl:document.getElementById('content'),stateSnapshots:{begin:()=>({generation:1,requestedAt:Date.now()}),
    gate(){return Promise.resolve(this.begin());},isCurrent:()=>true,
    apply(request,data){window.chat?.hydrateStateSnapshot(data);}}});
document.getElementById('reconnect-overlay').hidden=true;
document.getElementById('reconnect-overlay').style.display='none';
window.addEventListener('ouro:open-project',event=>window.openedProject=event.detail);
</script>"""


@pytest.mark.parametrize("engine,theme,width", [
    ("chromium", "light", 1360), ("chromium", "dark", 390),
    ("webkit", "light", 390), ("webkit", "dark", 1360),
])
def test_project_work_entries_share_live_phase_title_and_motion(engine, theme, width, tmp_path):
    if os.environ.get("OUROBOROS_RUN_UI_SMOKE") != "1":
        pytest.skip("Set OUROBOROS_RUN_UI_SMOKE=1 to run the browser flow")
    playwright = pytest.importorskip("playwright.sync_api")
    rows = [
        {"role": "system", "system_type": "project_started", "chat_id": 1,
         "task_id": "creation", "project_id": "alpha", "project_name": "Project Alpha",
         "task_name": "Review runner › integration", "text": "Project Alpha › Review runner › integration · Started",
         "ts": "2026-10-08T13:57:33Z"},
        {"role": "system", "system_type": "project_handoff", "chat_id": 1,
         "task_id": "transfer", "project_id": "beta", "project_name": "Project Beta",
         "handoff_id": "transfer-beta", "task_name": "A second independent work item",
         "text": "Project Beta › A second independent work item", "ts": "2026-10-08T13:58:00Z"},
    ]
    state = {"supervisor_ready": True, "active_chat_activities_complete": True,
             "active_chat_activities": [
                 {"activity_id": task, "project_id": project, "chat_id": chat, "phase": "working"}
                 for task, project, chat in [("creation", "alpha", 7), ("transfer", "beta", 8)]
             ]}
    history = {"messages": rows, "progress": [], "has_more": False, "page_cursor": "work-entries",
               "next_cursor": None, "window": {"complete": True}, "coverage": {
                   "v": 1, "view": "work", "upper": {"chat": 2, "progress": 0}, "spans": {
                       "chat": {"from": 0, "to": 2, "chain": "work", "gaps": []}, "progress": None}}}
    html = (WEB / "index.html").read_text(encoding="utf-8").replace(
        '<script type="module" src="/static/app.js"></script>', BOOT)

    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            if self.path == "/":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(html.encode("utf-8"))
                return
            self.path = self.path.removeprefix("/static")
            super().do_GET()

    def respond(route):
        path = urlparse(route.request.url).path
        data = history if path == "/api/chat/history" else state if path == "/api/state" else {}
        if path.startswith("/api/tasks/"):
            data = {"task_id": path.split("/")[-1], "status": "running"}
        route.fulfill(content_type="application/json", body=json.dumps(data))

    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(WEB)))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with playwright.sync_playwright() as pw:
            browser = getattr(pw, engine).launch(headless=True)
            try:
                page = browser.new_page(viewport={"width": width, "height": 850}, has_touch=width < 980)
                page.add_init_script(f"localStorage.setItem('ouroboros.theme', {json.dumps(theme)});")
                page.route("**/api/**", respond)
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto(f"http://127.0.0.1:{server.server_port}/")
                start = page.locator('[data-system-type="project_started"]')
                transfer = page.locator('[data-system-type="project_handoff"]')
                playwright.expect(start.locator('.chat-live-phase')).to_have_text("Working")
                playwright.expect(transfer.locator('.chat-live-phase')).to_have_text("Working")
                assert start.get_attribute("data-handoff-id") is None
                assert start.locator('.project-handoff-title').inner_text() == rows[0]["task_name"]
                assert start.locator('[data-intent="open-project"]').count() == 1
                assert "System" not in start.inner_text() and "Started" not in start.inner_text()
                assert start.evaluate("e=>getComputedStyle(e).backgroundColor") == transfer.evaluate(
                    "e=>getComputedStyle(e).backgroundColor")
                chip = start.locator('.chat-live-phase')
                assert chip.evaluate("e=>getComputedStyle(e).animationName") == "thinking-pulse"
                first = chip.evaluate("e=>getComputedStyle(e).opacity")
                page.wait_for_function("previous=>getComputedStyle(document.querySelector('[data-system-type=project_started] .chat-live-phase')).opacity!==previous", arg=first)
                page.emulate_media(reduced_motion="reduce")
                assert chip.evaluate("e=>getComputedStyle(e).animationName") == "none"
                page.emulate_media(reduced_motion="no-preference")

                for phase, extra, label, moving in [
                    ("queued", {}, "Queued", "0"),
                    ("working", {"owner_wait": {"owner_wait_state": "waiting"}}, "Waiting for your answer", "0"),
                    ("working", {"required_question": {"quiz_state": "answered", "owner_wait_state": "resumed"}}, "Working", "1"),
                    ("working", {"required_question_unavailable": True}, "Activity unconfirmed", "0"),
                ]:
                    activity = {"activity_id": "creation", "phase": phase, **extra}
                    page.evaluate("a=>chat.hydrateStateSnapshot({supervisor_ready:true,active_chat_activities_complete:true,active_chat_activities:[a]})", activity)
                    playwright.expect(chip).to_have_text(label)
                    assert chip.get_attribute("data-motion") == moving
                    assert chip.evaluate("e=>getComputedStyle(e).animationName") == ("thinking-pulse" if moving == "1" else "none")
                page.evaluate("s=>chat.hydrateStateSnapshot(s)", state)
                page.screenshot(path=str(tmp_path / f"project-work-{engine}-{theme}-{width}.png"))
                assert start.evaluate("e=>e.scrollWidth<=e.clientWidth+1")
                start.locator('[data-intent="open-project"]').click()
                assert page.evaluate("openedProject.task_id") == "creation"
                assert page.evaluate("openedProject.project.id") == "alpha"

                page.reload()
                playwright.expect(start.locator('.chat-live-phase')).to_have_text("Working")
                assert start.locator('.chat-live-phase').evaluate("e=>getComputedStyle(e).animationName") == "thinking-pulse"
                receipt = {**rows[0], "system_type": "project_handoff", "handoff_id": "transfer-alpha"}
                rows.append(receipt)
                page.evaluate("r=>emit('chat',{...r,content:r.text})", receipt)
                playwright.expect(start).to_be_hidden()
                assert page.locator('.project-handoff:not([hidden])').count() == 2
                page.reload()
                playwright.expect(start).to_have_count(1)
                playwright.expect(start).to_be_hidden()
                playwright.expect(page.locator('.project-handoff:not([hidden])')).to_have_count(2)

                # Exercise hydration and disconnect through the actual chat,
                # without pre-populating private record outcome flags.
                page.evaluate("""()=>emit('chat',{chat_id:1,task_id:'observed',role:'assistant',
                    is_progress:true,content:'Reviewing the current work',ts:'2026-10-08T14:00:00Z'})""")
                child_frame = {"chat_id": 1, "task_id": "child", "role": "assistant", "is_progress": True,
                               "content": "Inspecting a related file", "ts": "2026-10-08T14:00:01Z",
                               "subagent_task_id": "child", "parent_task_id": "observed",
                               "delegation_role": "subagent", "subagent_event": "running", "subagent_role": "scout"}
                page.evaluate("r=>emit('chat',r)", child_frame)
                child_chip = page.locator('.chat-live-card[data-task-id="child"] [data-live-phase]')
                playwright.expect(child_chip).to_have_text("Working")
                card = page.locator('.chat-live-card[data-task-id="observed"]')
                outcome = card.locator('[data-live-phase]').first
                secondary = card.locator('[data-live-phase-secondary]').first
                fact = {"activity_id": "observed", "chat_id": 1, "kind": "managed_task",
                        "phase": "finalizing", "status": "failed",
                        "root_phase_checkpoint": {"post_task_synthesis": "running"}}
                page.evaluate("a=>chat.hydrateStateSnapshot({supervisor_ready:true,active_chat_activities_complete:true,active_chat_activities:[a]})", fact)
                playwright.expect(outcome).to_have_text("Failed")
                playwright.expect(secondary).to_have_text("Finalizing…")
                assert secondary.get_attribute("data-motion") == "1"
                fact["status"] = "completed"
                page.evaluate("a=>chat.hydrateStateSnapshot({supervisor_ready:true,active_chat_activities_complete:true,active_chat_activities:[a]})", fact)
                playwright.expect(outcome).to_have_text("Done")
                state.update(active_chat_activities=[], active_chat_activities_complete=False)
                page.evaluate("connection(false)")
                playwright.expect(outcome).to_have_text("Done")
                playwright.expect(secondary).to_have_text("Activity unconfirmed")
                assert secondary.get_attribute("data-motion") == "0"
                assert secondary.evaluate("e=>getComputedStyle(e).animationName") == "none"
                playwright.expect(child_chip).to_have_text("Activity unconfirmed")
                page.evaluate("connection(true)")
                playwright.expect(secondary).to_have_text("Activity unconfirmed")
                page.evaluate("a=>chat.hydrateStateSnapshot({supervisor_ready:true,active_chat_activities_complete:true,active_chat_activities:[a]})", fact)
                playwright.expect(secondary).to_have_text("Finalizing…")
                assert secondary.get_attribute("data-motion") == "1"
                playwright.expect(child_chip).to_have_text("Activity unconfirmed")
                page.evaluate("r=>emit('chat',r)", child_frame)
                playwright.expect(child_chip).to_have_text("Working")
                assert child_chip.get_attribute("data-motion") == "1"

                ended = {**fact, "activity_id": "late-card", "status": "cancelled"}
                page.evaluate("a=>chat.hydrateStateSnapshot({supervisor_ready:true,active_chat_activities_complete:true,active_chat_activities:[a]})", ended)
                page.evaluate("""()=>emit('chat',{chat_id:1,task_id:'late-card',role:'assistant',
                    is_progress:true,narration:true,content:'Earlier progress arriving later',ts:'2026-10-08T14:00:00Z'})""")
                late = page.locator('.chat-live-card[data-task-id="late-card"]')
                playwright.expect(late.locator('[data-live-title]')).to_have_text("Earlier progress arriving later")
                playwright.expect(late.locator('[data-live-phase]')).to_have_text("Cancelled")
                assert late.locator('[data-live-phase]').get_attribute("data-motion") == "0"
                assert late.get_attribute("data-finished") == "1"
                assert not errors
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
