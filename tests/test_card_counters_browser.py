"""Selected A in the real Chat renderer and CSS, with isolated transport fixtures.

The long title comes from the owner's comparison capture. Counts and event sequences
are controlled fixtures; no live server, executor or account is used.
"""
import json
import os
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest

from tests._work_projection_shared import with_work_fields
from tests.ui_console_errors import PageErrors

pytestmark = pytest.mark.ui_browser
WEB = Path(__file__).resolve().parents[1] / "web"
TITLE = "Выпуск 4.7.0 · кандидат и 12 файлов проверены, манифесты подписаны · публикация выполняется; затем pin в Ouroboros, без установки"
HTML = """<!doctype html><html lang="en" class="ouro-ui"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="icon" href="data:,">
<link rel="stylesheet" href="/ui.css"><link rel="stylesheet" href="/style.css">
<link rel="stylesheet" href="/model_wait.css"><link rel="stylesheet" href="/room_work.css">
<link rel="stylesheet" href="/cards.css"></head><body><div id="app"><main id="content"></main></div>
<script type="module">
import {createChatInstance} from '/modules/chat.js';
const title = __TITLE__, state = __STATE__, idle = __IDLE__;
document.documentElement.dataset.theme = new URLSearchParams(location.search).get('theme');
const handlers = new Map();
window.historyRows = [];
const nativeFetch = window.fetch.bind(window);
window.fetch = (url, opts) => String(url).startsWith('/api/') ? Promise.resolve(new Response(JSON.stringify(
    String(url).startsWith('/api/chat/history') ? {messages: historyRows, window: {complete: true},
        page_cursor: 'fixture', next_cursor: null, has_more: false,
        coverage: {v: 1, view: 'fixture', upper: {chat: 1}, spans: {
            chat: {chain: 'fixture', from: 0, to: 1, gaps: []}, progress: null}}}
    : String(url) === '/api/state' ? state : {}), {status: 200, headers: {'Content-Type': 'application/json'}})) : nativeFetch(url, opts);
let generation = 0;
const chat = createChatInstance({mountEl: document.getElementById('content'), chatId: 1,
    state: {activePage: 'chat', projectChatIds: new Set(), unreadCount: 0},
    ws: {on(k, f) {handlers.set(k, f); return () => handlers.delete(k);}, isConnected: () => true, send() {}},
    stateSnapshots: {begin: () => ({generation: ++generation, requestedAt: Date.now()}),
        gate() {return Promise.resolve(this.begin());}, isCurrent: () => true, apply() {}}});
const ts = '2026-10-22T22:10:00Z';
const log = data => handlers.get('log')({chat_id: 1, data: {task_id: 'counter-root', ts, ...data}});
const say = data => handlers.get('chat')({chat_id: 1, ts, role: 'assistant', ...data});
const metrics = {rounds: 9, tool_calls: 31, tool_errors: 0, routing_tool_calls: 1,
    completion_tool_calls: 1, tool_call_counts: {read_file: 29, start_task: 1, task_complete: 1}, duration_sec: 4860};
const narration = {task_id: 'counter-root', role: 'assistant', is_progress: true, content: title, text: title,
    suggested_name: title, narration: true, model: 'gpt-6-astra', ts};
const final = {task_id: 'counter-root', role: 'assistant', task_terminal_status: 'completed', status: 'completed',
    outcome_axes: {lifecycle: {status: 'completed'}, execution: {status: 'ok'}}, ...metrics, ts};
if (new URLSearchParams(location.search).has('cold')) {
    window.historyRows = [narration, {...final, role: 'system', system_type: 'task_summary', text: ''}];
}
await chat.refreshHistory({revision: 1});
window.seed = async () => {
    chat.hydrateStateSnapshot(state, Infinity, ++generation);
    say(narration); log({type: 'task_metrics_event', ...metrics});
    window.historyRows = [narration];
};
window.finish = () => {
    window.historyRows = [narration, {...final, role: 'system', system_type: 'task_summary', text: ''}];
    chat.hydrateStateSnapshot(idle, Infinity, ++generation);
    log({...final, type: 'task_done'});
};
window.child = session => {
    say({task_id: 'counter-child', subagent_task_id: 'counter-child', parent_task_id: 'counter-root',
        root_task_id: 'counter-root', delegation_role: 'subagent', subagent_event: 'progress',
        subagent_role: 'Reviewer', is_progress: true, narration: true, content: 'Checking the complete source.',
        model: 'gpt-6-astra', ...(session ? {executor_route: 'claude'} : {})});
    log({type: 'task_metrics_event', task_id: 'counter-child', tool_calls: 7, rounds: 3});
};
window.portal = () => { window.__ouroTaskBindings = {'counter-root': {project_id: 'p', project_name: title}};
    chat.hydrateStateSnapshot({...state, active_chat_activities: state.active_chat_activities.map(r =>
        ({...r, project_id: 'p', project_name: title, tool_calls: 31}))}, Infinity, ++generation); };
window.textOnly = () => {
    log({type: 'llm_round_finished', task_id: 'plain', rounds: 1});
    say({task_id: 'plain', content: 'Hello.', task_terminal_status: 'completed'});
    log({type: 'task_done', task_id: 'plain', status: 'completed', rounds: 1});
};
window.ready = true;
</script></body></html>"""


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_selected_counter_in_chat(tmp_path, engine):
    from playwright.sync_api import expect, sync_playwright

    state = with_work_fields({"active_chat_activities": [{"activity_id": "counter-root", "kind": "managed_task",
        "phase": "working", "chat_id": 1, "task_name": TITLE}], "active_chat_activities_complete": True, "supervisor_ready": True})
    idle = with_work_fields({"active_chat_activities": [], "active_chat_activities_complete": True, "supervisor_ready": True})
    html = HTML.replace("__TITLE__", json.dumps(TITLE)).replace("__STATE__", json.dumps(state)).replace("__IDLE__", json.dumps(idle))

    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            if self.path.startswith("/fixture"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(html.encode())
            else:
                super().do_GET()

    evidence = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR") or tmp_path / "evidence")
    evidence.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(WEB)))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    measurements = []
    try:
        with sync_playwright() as pw:
            browser = getattr(pw, engine).launch(headless=True)
            try:
                for theme, width in (("light", 1280), ("dark", 1280), ("light", 390), ("dark", 390)):
                    page = browser.new_page(viewport={"width": width, "height": 844}, color_scheme=theme, device_scale_factor=2)
                    errors = PageErrors(page, engine)
                    page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)
                    url = f"http://127.0.0.1:{server.server_port}/fixture?theme={theme}"
                    page.goto(url)
                    page.wait_for_function("window.ready")
                    expect(page.get_by_text("Some saved history could not be loaded.", exact=True)).to_have_count(0)
                    expect(page.get_by_text("Some saved history is not loaded. Shown messages may have gaps.", exact=True)).to_have_count(0)
                    page.evaluate("seed()")
                    card = page.locator('.chat-live-card[data-task-id="counter-root"]')
                    meta = card.locator(":scope > [data-live-foot] > [data-live-meta]")
                    expect(meta).to_contain_text("31 tool calls")
                    expect(meta).not_to_contain_text("rounds")
                    expect(page.locator('[data-live-counters]')).to_have_count(0)
                    for phase in ("running", "done"):
                        if phase == "done":
                            page.evaluate("finish()")
                            expect(card).to_have_attribute("data-finished", "1")
                        for expanded in (False, True):
                            if (card.get_attribute("data-expanded") == "1") != expanded:
                                card.locator(":scope > [data-live-summary-button] [data-live-disclosure]").click()
                            expect(meta).to_be_visible()
                            expect(meta).to_contain_text("31 tool calls")
                            name = f"{engine}-{theme}-{width}-{phase}-{'expanded' if expanded else 'collapsed'}"
                            page.screenshot(path=str(evidence / f"{name}.png"))
                            meta.screenshot(path=str(evidence / f"{name}-meta.png"))
                            measured = meta.evaluate("""el => ({text: el.innerText, width: el.clientWidth,
                                scroll: el.scrollWidth, parts: [...el.children].map(p => ({text: p.innerText,
                                right: p.getBoundingClientRect().right, left: p.getBoundingClientRect().left})),
                                viewport: innerWidth})""")
                            measurements.append({"case": name, **measured})
                            assert measured["scroll"] <= measured["width"] + 1
                            assert all(p["left"] >= 0 and p["right"] <= width + 1 for p in measured["parts"])
                            if expanded:
                                expect(card.locator('[data-live-facts]')).to_contain_text("9 · usable solve responses")
                    page.goto(url + '&cold=1')
                    page.wait_for_function("window.ready")
                    expect(meta).to_contain_text("31 tool calls")
                    expect(card).to_have_attribute("data-finished", "1")
                    expect(page.get_by_text("Some saved history could not be loaded.", exact=True)).to_have_count(0)
                    expect(page.get_by_text("Some saved history is not loaded. Shown messages may have gaps.", exact=True)).to_have_count(0)
                    page.evaluate("textOnly()")
                    expect(page.locator('.chat-live-card[data-task-id="plain"]')).to_have_count(0)
                    page.evaluate("child(false)")
                    kid = page.locator('.chat-live-card[data-task-id="counter-child"]')
                    kidmeta = kid.locator(':scope > [data-live-foot] > [data-live-meta]')
                    expect(kidmeta).to_be_visible()  # no actions or Reviews to unhide the footer end
                    expect(kidmeta).to_contain_text("7 tool calls")
                    page.evaluate("child(true)")
                    expect(kidmeta).not_to_contain_text("tool call")
                    kid.locator('[data-live-disclosure]').click()
                    expect(kid.locator('[data-live-facts]')).to_contain_text("Supervision3 rounds · 7 tool calls")
                    page.screenshot(path=str(evidence / f"{engine}-{theme}-{width}-supervision.png"))
                    kid.screenshot(path=str(evidence / f"{engine}-{theme}-{width}-supervision-card.png"))
                    page.evaluate("portal()")
                    expect(card).to_have_attribute("data-portal", "1")
                    assert meta.inner_text().count("31 tool calls") == 1
                    assert errors.unexpected() == [], errors
                    page.close()
            finally:
                browser.close()
    finally:
        (evidence / f"{engine}-measurements.json").write_text(json.dumps(measurements, ensure_ascii=False, indent=2))
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
