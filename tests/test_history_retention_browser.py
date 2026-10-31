"""Shared production Chat/Logs against an isolated static fixture, no live runtime."""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
from threading import Thread

import pytest

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]
WEB = Path(__file__).resolve().parents[1] / "web"
HTML = """<!doctype html><html class="ouro-ui"><head><meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="/ui.css"><link rel="stylesheet" href="/style.css">
<link rel="stylesheet" href="/settings.css"><script src="/purify.min.js"></script><script src="/marked.min.js"></script>
</head><body><main id="chat-mount" style="height:100vh;width:100%"></main><main id="logs-mount" hidden></main>
<script type="module">
import {createChatInstance} from '/modules/chat.js';
import {initLogs} from '/modules/logs.js';
const handlers = new Map();
const ws = {on(type, fn) {const list=handlers.get(type)||[]; list.push(fn); handlers.set(type,list); return ()=>{};},
    isConnected:()=>true, send() {}};
window.emit = (type, value) => (handlers.get(type)||[]).forEach(fn=>fn(value));
const state = {activePage:'chat', projectChatIds:new Set(), unreadCount:0};
window.chat = createChatInstance({ws,state,updateUnreadBadge(){},chatId:1,idPrefix:'chat',asPanel:true,
    mountEl:document.getElementById('chat-mount'),stateSnapshots:{begin:()=>({generation:1,requestedAt:Date.now()}),
        gate(){return Promise.resolve(this.begin());},isCurrent:()=>true,apply(){}}});
initLogs({ws,state,mount:document.getElementById('logs-mount')});
window.showLogs=()=>{document.getElementById('chat-mount').hidden=true;document.getElementById('logs-mount').hidden=false;
    state.activePage='dashboard';state.dashboardActiveSubtab='logs';};
window.fixtureReady=true;
</script></body></html>"""


@pytest.mark.parametrize(("engine", "width"), [("chromium", 1360), ("webkit", 390)])
def test_retention_details_and_problem_only_card(engine, width):
    if os.environ.get("OUROBOROS_RUN_UI_SMOKE") != "1":
        pytest.skip("Set OUROBOROS_RUN_UI_SMOKE=1")
    playwright = pytest.importorskip("playwright.sync_api")
    detail = {"task_id": "history-task", "status": "completed", "history_retention": {"status": "pending"}}

    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            if self.path == "/fixture":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(HTML.encode("utf-8"))
                return
            super().do_GET()

    def respond(route):
        path = route.request.url.split("/api/", 1)[1]
        data = detail if path.startswith("tasks/") else {"messages": [], "entries": [], "active_direct_turns": []}
        route.fulfill(content_type="application/json", body=json.dumps(data))

    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(WEB)))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with playwright.sync_playwright() as pw:
            browser = getattr(pw, engine).launch(headless=True)
            try:
                page = browser.new_page(viewport={"width": width, "height": 900}, has_touch=width < 980)
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.route("**/api/**", respond)
                page.goto(f"http://127.0.0.1:{server.server_port}/fixture")
                page.wait_for_function("window.fixtureReady === true")
                page.evaluate("""() => {
                    emit('chat',{task_id:'history-task',chat_id:1,role:'system',is_progress:true,
                        content:'Preparing the project report',ts:'2026-09-27T11:59:59Z'});
                    emit('chat',{task_id:'history-task',chat_id:1,role:'system',system_type:'task_summary',
                        content:'Finished.',task_terminal_status:'completed',ts:'2026-09-27T12:00:00Z'});
                }""")
                card = page.locator('.chat-live-card[data-task-id="history-task"]')
                card.wait_for()
                summary = card.locator(':scope > [data-live-summary-button]')
                toggle = summary.locator('[data-live-disclosure]')
                meta = card.locator(':scope > .chat-live-foot > [data-live-meta]')
                if card.get_attribute('data-expanded') == '1':
                    toggle.click()
                page.evaluate("""() => emit('log',{chat_id:1,data:{type:'history_retention',task_id:'history-task',
                    status:'deferred',pending_count:1,problem_count:0}})""")
                assert 'history' not in summary.inner_text().lower()

                def capture(name):
                    root = os.environ.get("OUROBOROS_UI_EVIDENCE_DIR")
                    if root:
                        Path(root).mkdir(parents=True, exist_ok=True)
                        page.screenshot(path=str(Path(root) / f"retention-{engine}-{width}-{name}.png"))

                capture("collapsed-normal")
                toggle.click()
                history = card.locator('[data-live-line-key="history-retention-history-task"]')
                playwright.expect(history).to_contain_text('Saving task history')
                assert 'history' not in summary.inner_text().lower()
                capture("details-pending")
                toggle.click()
                for status in ("problem", "complete"):
                    detail["history_retention"] = {"status": status, "problem_reasons":
                                                   [{"reason": "OSError: disk full", "count": 2}] if status == "problem" else []}
                    page.evaluate("""status => emit('log',{chat_id:1,data:{type:'history_retention',task_id:'history-task',status,
                        pending_count:status==='complete'?0:1,problem_count:status==='problem'?1:0,
                        problem_reasons:status==='problem'?[{reason:'OSError: disk full',count:2}]:[],
                        diagnostics:status==='problem'?{pending_refs:[{path:'retained-source',reason:'OSError: disk full'}]}:{}}})""", status)
                    if status == "problem":
                        # The problem is a fact of the card's footer meta line, not of the summary button.
                        playwright.expect(meta).to_contain_text('History storage problem')
                        assert 'history storage problem' not in toggle.inner_text().lower()
                        capture("collapsed-problem")
                        toggle.click()
                        playwright.expect(history).to_contain_text('2 × OSError: disk full')
                        capture("details-problem")
                        toggle.click()
                    else:
                        playwright.expect(meta).not_to_contain_text('History storage problem')
                    playwright.expect(card.locator('[data-live-phase]')).to_have_text('Done')
                    assert card.evaluate('(node) => node.scrollWidth <= node.clientWidth + 1')
                page.evaluate('showLogs()')
                playwright.expect(page.locator('#log-entries')).to_contain_text('Task history saved')
                playwright.expect(page.locator('#log-entries')).to_contain_text('Task history needs attention')
                playwright.expect(page.locator('#log-entries')).to_contain_text('Saving task history')
                playwright.expect(page.locator('#log-entries .log-body').filter(has_text='OSError: disk full')).to_contain_text('2 × OSError: disk full')
                page.locator('.log-entry').filter(has_text='Task history needs attention').get_by_role('button', name='Raw', exact=True).click()
                playwright.expect(page.locator('.log-raw:visible')).to_contain_text('retained-source')
                capture("logs")
                assert not errors
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()
