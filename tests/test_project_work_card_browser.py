"""One work, one card in Main, over static gateway data in real engines (no runtime): a Project row binds the work's
own card as its portal (never a second entry), Turn into project keeps the card, the row menu opens by pointer,
keyboard and touch, and a reply sends only the quoted row's address."""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
from threading import Thread
from urllib.parse import urlparse

import pytest

from tests._shared import producer_evidence_dir
from tests._work_projection_shared import with_work_fields
from tests.ui_console_errors import PageErrors

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]
WEB = Path(__file__).resolve().parents[1] / "web"
BOOT = """<script type="module">
import {createChatInstance} from '/static/modules/chat.js';
const handlers=new Map();let connected=true;let coined=0;window.sentFrames=[];
const ws={ws:{readyState:1},on(type,fn){const rows=handlers.get(type)||[];rows.push(fn);handlers.set(type,rows);return()=>{};},
    isConnected:()=>connected,send(frame){const id=frame.client_message_id||`cm-${++coined}`;
        window.sentFrames.push({...frame,client_message_id:id});return {status:'sent',clientMessageId:id};}};
window.emit=(type,row)=>(handlers.get(type)||[]).forEach(fn=>fn(row));
window.connection=value=>{connected=value;window.emit(value?'open':'close');};
window.chat=createChatInstance({ws,state:{activePage:'chat',projectChatIds:new Set([7,8]),unreadCount:0},
    updateUnreadBadge(){},openSettingsTab(){},openDashboardTab(){},chatId:1,
    mountEl:document.getElementById('content'),stateSnapshots:{begin:()=>({generation:1,requestedAt:Date.now()}),
    gate(){return Promise.resolve(this.begin());},isCurrent:()=>true,
    apply(request,data){window.chat?.hydrateStateSnapshot(data);}}});
window.addEventListener('ouro:open-project',event=>{window.openedProject=event.detail;});
</script>"""


@pytest.fixture(params=[
    ("chromium", "light", 1360), ("chromium", "dark", 390),
    ("webkit", "light", 390), ("webkit", "dark", 1360),
], ids=["chromium-light-desktop", "chromium-dark-mobile", "webkit-light-mobile", "webkit-dark-desktop"])
def project_chat(request):
    """A real chat instance over static gateway data (Main, or a Project panel when the test is parametrized with
    `chat_id`); `rows` is the history the page reads and `state` the /api/state answer, both mutable."""
    engine, theme, width = request.param
    if os.environ.get("OUROBOROS_RUN_UI_SMOKE") != "1":
        pytest.skip("Set OUROBOROS_RUN_UI_SMOKE=1 to run the browser flow")
    playwright = pytest.importorskip("playwright.sync_api")
    rows = [
        {"role": "user", "text": "Audit the review runner and keep it as a project", "chat_id": 1, "source": "web",
         "ts": "2026-10-08T13:57:00Z", "history_id": "h-0", "row_address": "row:1@2026-10-08T13:57:00Z#aaaaaaaaaaaa"},
        {"role": "assistant", "text": "I will audit the review runner in its own room.", "chat_id": 1,
         "ts": "2026-10-08T13:57:20Z", "history_id": "h-1", "row_address": "row:1@2026-10-08T13:57:20Z#bbbbbbbbbbbb"},
        {"role": "system", "system_type": "project_started", "chat_id": 1,
         "task_id": "creation", "project_id": "alpha", "project_name": "Project Alpha",
         "task_name": "Review runner › integration", "text": "Project Alpha › Review runner › integration · Started",
         "ts": "2026-10-08T13:57:33Z", "history_id": "h-2"},
        {"role": "system", "system_type": "project_handoff", "chat_id": 1,
         "task_id": "transfer", "project_id": "beta", "project_name": "Project Beta",
         "handoff_id": "transfer-beta", "task_name": "A second independent work item",
         "text": "Project Beta › A second independent work item", "ts": "2026-10-08T13:58:00Z", "history_id": "h-3"},
    ]
    state = {"supervisor_ready": True, "active_chat_activities_complete": True,
             "task_bindings": {"creation": {"project_id": "alpha", "chat_id": 7}, "transfer": {"project_id": "beta", "chat_id": 8}},
             "projects": [{"id": "alpha", "name": "Project Alpha", "chat_id": 7}, {"id": "beta", "name": "Project Beta", "chat_id": 8}],
             "active_chat_activities": [
                 {"activity_id": "creation", "project_id": "alpha", "project_name": "Project Alpha", "chat_id": 7,
                  "kind": "managed_task", "phase": "working", "started_at": 1791467853,
                  "last_line": "Comparing the reviewer verdicts across the last three runs.",
                  "last_line_at": "2026-10-08T14:05:00Z", "tool_calls": 41, "cancelable": True},
                 {"activity_id": "transfer", "project_id": "beta", "chat_id": 8, "kind": "managed_task",
                  "phase": "budget_paused", "pause_cause": "owner", "started_at": 1791467880,
                  "stop": {"cause": "pause", "scope": "tree", "release": "resume"}},
             ]}
    conversions, details, requests, continues = {}, {}, [], {}
    history = {"messages": rows, "progress": [], "has_more": False, "page_cursor": "work-cards",
               "next_cursor": None, "window": {"complete": True}, "coverage": {
                   "v": 1, "view": "work", "upper": {"chat": len(rows), "progress": 0}, "spans": {
                       "chat": {"from": 0, "to": len(rows), "chain": "work", "gaps": []}, "progress": None}}}
    chat_id = request.node.callspec.params.get("chat_id", 1)
    boot = BOOT.replace("chatId:1,", f"chatId:{chat_id},asPanel:{str(chat_id != 1).lower()},")
    html = (WEB / "index.html").read_text(encoding="utf-8").replace(
        '<script type="module" src="/static/app.js"></script>', boot)

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
        # The census rows and, as the server sends them, their work projection.
        data = history if path == "/api/chat/history" else with_work_fields(state) if path == "/api/state" else {}
        if path == "/api/projects/from-task":
            payload = route.request.post_data_json
            requests.append(payload)
            data = conversions[payload["task_id"]]
        if path.startswith("/api/tasks/"):
            task_id = path.split("/")[-1]
            data = details.get(task_id, {"task_id": task_id, "status": "running"})
        if path.startswith("/api/tasks/") and path.endswith("/continue"):
            data = continues.get(path.split("/")[-2], {"ok": False, "error": "not admitted"})
        route.fulfill(content_type="application/json", body=json.dumps(data))

    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(WEB)))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with playwright.sync_playwright() as pw:
            browser = getattr(pw, engine).launch(headless=True)
            try:
                page = browser.new_page(viewport={"width": width, "height": 900}, has_touch=width < 980, timezone_id="UTC")
                page.add_init_script(f"localStorage.setItem('ouroboros.theme', {json.dumps(theme)});")
                page.add_init_script("""Object.defineProperty(navigator, 'clipboard', {
                    value: {writeText: async text => {window.copiedText = text;}}});""")
                page.route("**/api/**", respond)
                errors = PageErrors(page, engine)
                page.goto(f"http://127.0.0.1:{server.server_port}/")
                if chat_id == 1:
                    page.wait_for_function("() => window.chat && document.querySelector('#chat-messages .chat-bubble')")
                    page.evaluate("s=>chat.hydrateStateSnapshot(s)", with_work_fields(state))
                yield {"page": page, "expect": playwright.expect, "rows": rows, "state": state, "conversions": conversions,
                       "details": details, "requests": requests, "continues": continues, "history": history,
                       "engine": engine, "theme": theme, "width": width}
                assert not errors.unexpected(), errors
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _evidence_dir(tmp_path):
    # An exported destination stays exactly where the operator put it; otherwise the shared
    # channel keeps the shell screenshots and the geometry/keyboard JSON past scratch release.
    return producer_evidence_dir(tmp_path, "project-work-card", env="OUROBOROS_BROWSER_EVIDENCE_OUT")


def _evidence(tmp_path, chat, name):
    return str(_evidence_dir(tmp_path) / f"{name}-{chat['engine']}-{chat['theme']}-{chat['width']}.png")


def _card(page, task_id):
    return page.locator(f'#chat-messages > .chat-live-card[data-task-id="{task_id}"]')


# The portal's Project reference leads its own action row, under the meta line, its controls at the row's end
# (owner 26.10, "вариант B"): never a strip closing the card.
DOOR = ':scope > .chat-live-foot > .chat-live-foot-end > .chat-live-bound-pointer'
LEADS = "c => c.querySelector(':scope > .chat-live-foot > .chat-live-foot-end')?.firstElementChild?.classList.contains('chat-live-bound-pointer')"
# The B composition as laid out: boxes relative to the card, the capsule's hit area under a finger, the name's cut.
_PORTAL_ROW = """card => {
    const c = card.getBoundingClientRect();
    const box = (el) => { if (!el) return null; const r = el.getBoundingClientRect();
        return {top: r.top - c.top, bottom: r.bottom - c.top, left: r.left - c.left, right: c.right - r.right,
            width: r.width, height: r.height, mid: (r.top + r.bottom) / 2 - c.top}; };
    const door = card.querySelector(':scope > .chat-live-foot > .chat-live-foot-end > .chat-live-bound-pointer');
    const name = door?.querySelector('.chat-live-project-name');
    const after = door ? getComputedStyle(door, '::after') : null;
    const coarse = matchMedia('(pointer: coarse)').matches;
    return {card: box(card), summary: box(card.querySelector(':scope > .chat-live-summary-button')),
        lines: [...card.querySelectorAll(':scope > .chat-live-summary-button > *')].filter(e => e.getClientRects().length).map(box),
        meta: box(card.querySelector(':scope > .chat-live-foot > .chat-live-meta')), door: box(door),
        row: box(card.querySelector(':scope > .chat-live-foot > .chat-live-foot-end')),
        buttons: [...card.querySelectorAll(':scope > .chat-live-foot > .chat-live-foot-end > .chat-live-actions > button')].map(box),
        shown: name?.textContent, cut: name ? name.scrollWidth > name.clientWidth + 1 : null,
        hit: coarse && after && after.content !== 'none' ? door.getBoundingClientRect().height - parseFloat(after.top) - parseFloat(after.bottom) : null,
        closing: c.bottom - parseFloat(getComputedStyle(card).borderBottomWidth) - card.querySelector(':scope > .chat-live-foot > .chat-live-foot-end').getBoundingClientRect().bottom,
        overflow: card.scrollWidth > card.clientWidth + 1 || document.documentElement.scrollWidth > innerWidth + 1};
}"""


def _assert_b_row(geo, buttons):
    """One action row under the meta line: the capsule at the meta line's left edge, the controls at the row's end,
    all on one line; nothing wider than the card."""
    assert not geo["overflow"], geo
    door, meta, row = geo["door"], geo["meta"], geo["row"]
    assert meta["bottom"] <= row["top"] + 0.5, "the meta line keeps its own line above the action row"
    assert abs(door["left"] - meta["left"]) <= 1, "the capsule starts on the meta line's left edge"
    assert abs(row["height"] - door["height"]) <= 1 and door["height"] >= 24, "one row at the controls' height"
    assert len(geo["buttons"]) == buttons, geo["buttons"]
    for button in geo["buttons"]:
        assert abs(button["mid"] - door["mid"]) <= 1, "the controls share the capsule's line"
    if geo["buttons"]:
        assert abs(geo["buttons"][-1]["right"] - door["left"]) <= 1, "the row ends on the inset it starts on"
        assert door["left"] + door["width"] + 8 <= geo["buttons"][0]["left"] + 0.5, "the capsule stops before the controls"
    if geo["hit"] is not None:
        assert geo["hit"] >= 44 - 0.5, "a finger's target"


def test_project_rows_bind_the_works_own_cards_never_an_entry(project_chat, tmp_path):
    page, expect = project_chat["page"], project_chat["expect"]
    creation, transfer = _card(page, "creation"), _card(page, "transfer")
    for card, name in ((creation, "Project Alpha"), (transfer, "Project Beta")):
        expect(card).to_have_attribute("data-portal", "1")
        door = card.locator(DOOR)
        expect(door).to_have_attribute("aria-label", f"Open project {name}")
        # Named apart from its work, the Project shows its own name on the capsule.
        expect(door.locator('.chat-live-project-name')).to_have_text(name)
        assert card.evaluate(LEADS), "the reference leads the card's action row"
        assert card.locator('[data-intent="open-project"]').count() == 1, "one door, no second footer"
    assert page.locator('.project-handoff, [data-system-type="project_started"], [data-system-type="project_handoff"]').count() == 0
    # The room's present: chip, the latest line, tool calls and time; [Stop…] while it runs (Reply is the row menu's).
    expect(creation.locator('.chat-live-phase').first).to_have_text("Working")
    expect(creation.locator('[data-live-activity]')).to_contain_text("Comparing the reviewer verdicts")
    expect(creation.locator('[data-live-meta]')).to_contain_text("41 tool calls")
    expect(creation.locator('.chat-live-actions > button')).to_have_text(["Stop…"])
    assert creation.locator('[data-card-reply], .chat-live-actions > button', has_text="Reply").count() == 0
    # A paused work: Resume is the one primary, beside Stop….
    expect(transfer.locator('.chat-live-actions > button')).to_have_text(["Resume", "Stop…"])
    assert transfer.locator('.chat-live-actions > .btn-primary').all_text_contents() == ["Resume"]
    # The summary is reading content in a portal: no disclosure, no stop.
    assert creation.locator(':scope > [data-live-summary-button]').evaluate("e=>['role','tabindex','aria-expanded'].map(n=>e.getAttribute(n))") == [None, None, None]
    expect(creation.locator('[data-live-disclosure]')).to_be_hidden()
    geometry = creation.evaluate(_PORTAL_ROW)
    _assert_b_row(geometry, buttons=1)
    _assert_b_row(transfer.evaluate(_PORTAL_ROW), buttons=2)
    # Compact (B): the meta line follows the summary at the summary's own 8px gap, the row closes the card at 12px.
    assert abs(geometry["meta"]["top"] - geometry["lines"][-1]["bottom"] - 8) <= 1, geometry
    assert abs(geometry["closing"] - 12) <= 0.5, geometry
    creation.locator(DOOR).click()
    assert page.evaluate("[openedProject.project.id, openedProject.task_id]") == ["alpha", "creation"]
    creation.screenshot(path=_evidence(tmp_path, project_chat, "portal"), animations="disabled")


def test_turn_into_project_keeps_the_card_as_its_portal(project_chat):
    page, expect = project_chat["page"], project_chat["expect"]
    page.evaluate("""() => {
        emit('chat', {chat_id: 1, task_id: 'local', role: 'assistant', is_progress: true, narration: true,
            content: 'Inspecting the requested work', ts: '2026-10-08T14:10:00Z'});
        emit('task_named', {task_id: 'local', suggested_name: 'Inspect the runner'});
    }""")
    card = _card(page, "local")
    state = project_chat["state"]
    state["active_chat_activities"].append({"activity_id": "local", "chat_id": 1, "kind": "direct_chat", "phase": "working",
                                            "cancelable": True})
    page.evaluate("s=>chat.hydrateStateSnapshot(s)", with_work_fields(state))
    expect(card.locator('[data-turn-into-project]')).to_have_count(1)
    node = card.element_handle()
    project_chat["conversions"]["local"] = {"project": {"id": "gamma", "name": "Project Gamma", "chat_id": 9},
                                         "binding": {"project_id": "gamma"}, "handoff_receipt": "durable"}
    card.locator('[data-turn-into-project]').click()
    expect(card).to_have_attribute("data-portal", "1")
    assert page.evaluate("n => n.isConnected && n === document.querySelector('[data-task-id=\"local\"]')", node), "the SAME node"
    expect(card.locator('[data-turn-into-project]')).to_have_count(0)
    expect(card.locator(DOOR)).to_have_attribute("aria-label", "Open project Project Gamma")
    assert card.evaluate(LEADS), "the converted card's row leads with its Project"
    expect(card.locator(':scope > .chat-live-foot > .chat-live-foot-end > .chat-live-actions > button')).to_have_text(["Stop…"])
    assert project_chat["requests"][-1]["task_id"] == "local"


SAME = "Release checklist — signing and notarization"
LONG_TITLE = ("Narrow screen — the scrollbar disappears after the WebKit rollback, a second check and a comparison with "
              "Chromium on the same frames")
LONG_PROJECT = "Interface check after the rendering engine update and the frame comparison"


# Every observer pointed at a card title (DEVELOPMENT "UI resources carry a disposer": the capsule keeps no watch there).
_TITLE_WATCHES = """(() => { const observe = MutationObserver.prototype.observe; window.titleWatches = 0;
    MutationObserver.prototype.observe = function (target, options) {
        const el = target && (target.nodeType === 1 ? target : target.parentElement);
        if (el?.closest?.('[data-live-title]')) window.titleWatches += 1;
        return observe.call(this, target, options); }; })();"""


def test_portal_row_says_a_same_named_project_once_and_keeps_a_long_name_on_one_line(project_chat, tmp_path):
    """Owner 26.10, "вариант B": beside a title that already says its Project's name the capsule reads `Project`
    (its accessible name, tooltip and press keep the Project) and follows the title when its naming owner renames it,
    with no observer on the title; a long, differently named Project beside [Resume] [Stop…] ellipsizes inside the one
    row at desktop and phone width, the controls whole, nothing wider than the card."""
    page, expect, state = project_chat["page"], project_chat["expect"], project_chat["state"]
    page.add_init_script(_TITLE_WATCHES)
    page.reload()
    page.wait_for_function("() => window.chat && document.querySelector('#chat-messages .chat-bubble')")
    page.evaluate("s => chat.hydrateStateSnapshot(s)", with_work_fields(state))
    for task, project, name, title, at in (("same", "delta", SAME, SAME, "14:20"),
                                           ("long", "epsilon", LONG_PROJECT, LONG_TITLE, "14:21")):
        page.evaluate("""r => emit('chat', {...r, content: r.text})""", {
            "role": "system", "system_type": "project_started", "chat_id": 1, "task_id": task, "project_id": project,
            "project_name": name, "task_name": title, "text": f"{name} › {title} · Started",
            "ts": f"2026-10-08T{at}:00Z"})
        state["task_bindings"][task] = {"project_id": project, "chat_id": 9}
        state["projects"].append({"id": project, "name": name, "chat_id": 9})
    state["active_chat_activities"] += [
        {"activity_id": "same", "project_id": "delta", "project_name": SAME, "chat_id": 9, "kind": "managed_task",
         "phase": "working", "started_at": 1791469200, "tool_calls": 12, "cancelable": True},
        {"activity_id": "long", "project_id": "epsilon", "project_name": LONG_PROJECT, "chat_id": 9,
         "kind": "managed_task", "phase": "budget_paused", "pause_cause": "owner", "started_at": 1791469260,
         "last_line": "Paused by the owner while the rollback finishes; the last check showed the scrollbar gone on the "
                      "390 px frame only after the second reload, so the next step compares both engines.",
         "last_line_at": "2026-10-08T14:25:00Z", "tool_calls": 41,
         "stop": {"cause": "pause", "scope": "tree", "release": "resume"}}]
    page.evaluate("s => chat.hydrateStateSnapshot(s)", with_work_fields(state))
    same, long = _card(page, "same"), _card(page, "long")
    expect(same.locator(".chat-live-actions > button")).to_have_text(["Stop…"])
    expect(long.locator(".chat-live-actions > button")).to_have_text(["Resume", "Stop…"])
    door = same.locator(DOOR)
    expect(door.locator(".chat-live-project-name")).to_have_text("Project")
    expect(door).to_have_attribute("aria-label", f"Open project {SAME}")
    expect(door).to_have_attribute("title", f"Open project {SAME}")
    _assert_b_row(same.evaluate(_PORTAL_ROW), buttons=1)
    # The title renamed after the binding (a coined name): the capsule says the Project's name again, and back.
    page.evaluate("() => emit('task_named', {task_id: 'same', suggested_name: 'Release checklist, second pass'})")
    expect(same.locator("[data-live-title]")).to_have_text("Release checklist, second pass")
    expect(door.locator(".chat-live-project-name")).to_have_text(SAME)
    page.evaluate("name => emit('task_named', {task_id: 'same', suggested_name: name})", SAME)
    expect(door.locator(".chat-live-project-name")).to_have_text("Project")
    # The long name stays one capsule on the controls' line; at phone width it ellipsizes, its speech whole.
    long_door = long.locator(DOOR)
    expect(long_door).to_have_attribute("aria-label", f"Open project {LONG_PROJECT}")
    expect(long_door.locator(".chat-live-project-name")).to_have_text(LONG_PROJECT)
    geometry = long.evaluate(_PORTAL_ROW)
    _assert_b_row(geometry, buttons=2)
    assert min(button["width"] for button in geometry["buttons"]) >= 48, "the controls keep their words"
    if project_chat["width"] < 600:
        assert geometry["cut"], "the name gives way first"
    long.screenshot(path=_evidence(tmp_path, project_chat, "portal-long"), animations="disabled")
    same.screenshot(path=_evidence(tmp_path, project_chat, "portal-same-name"), animations="disabled")
    long_door.click()
    assert page.evaluate("[openedProject.project.id, openedProject.project.name, openedProject.task_id]") == [
        "epsilon", LONG_PROJECT, "long"]
    door.click()
    assert page.evaluate("[openedProject.project.id, openedProject.task_id]") == ["delta", "same"]
    expect(page.locator('#chat-messages > .chat-live-card[data-portal="1"]')).to_have_count(4)
    assert page.evaluate("window.titleWatches") == 0, "no portal watches its title"


def test_the_row_menu_opens_by_pointer_keyboard_and_touch_and_gives_focus_back(project_chat, tmp_path):
    page, expect = project_chat["page"], project_chat["expect"]
    card = _card(page, "creation")
    menu = page.locator('.chat-row-menu:not([data-leaving])')
    card.locator('[data-live-title]').click(button="right")
    expect(menu.locator('[role="menuitem"]')).to_have_text(["Reply", "Copy", "Open project", "Pause", "Stop…"])
    expect(menu.locator('[role="menuitem"]').first).to_be_focused()
    page.screenshot(path=_evidence(tmp_path, project_chat, "row-menu-card"), animations="disabled")
    page.keyboard.press("ArrowDown")
    expect(menu.locator('[role="menuitem"]').nth(1)).to_be_focused()
    page.keyboard.press("Escape")
    expect(menu).to_have_count(0)
    assert page.evaluate("() => document.activeElement === document.querySelector('[data-task-id=\"creation\"]')")
    # Keyboard: Shift+F10 on the answer's focused Copy.
    answer = page.locator('#chat-messages > .chat-bubble.assistant').first
    answer.locator('.chat-message-copy').focus()
    page.keyboard.press("Shift+F10")
    expect(menu.locator('[role="menuitem"]')).to_have_text(["Reply", "Copy"])
    page.keyboard.press("Escape")
    expect(answer.locator('.chat-message-copy')).to_be_focused()
    # Touch: a held finger (no movement) opens it; the lifting finger presses nothing.
    box = answer.locator('.message').bounding_box()
    point = {"x": box["x"] + 20, "y": box["y"] + 8}
    page.evaluate("""p => { const t = document.elementFromPoint(p.x, p.y);
        t.dispatchEvent(new PointerEvent('pointerdown', {bubbles: true, pointerType: 'touch', isPrimary: true, pointerId: 7,
            clientX: p.x, clientY: p.y})); }""", point)
    page.wait_for_timeout(600)
    expect(menu.locator('[role="menuitem"]')).to_have_count(2)
    # Where the finger lifts the menu now lies (it opened at the finger): the item under it is not pressed.
    page.evaluate("""p => { const t = document.elementFromPoint(p.x + 12, p.y + 12);
        document.querySelector('[data-task-id]')?.closest('#chat-messages')
            .dispatchEvent(new PointerEvent('pointerup', {bubbles: true, pointerType: 'touch', pointerId: 7}));
        t.click(); }""", point)
    expect(menu).to_have_count(1)
    assert page.evaluate("() => document.querySelectorAll('.chat-quote--composer').length") == 0, "the lift pressed no item"
    page.keyboard.press("Escape")
    expect(menu).to_have_count(0)


def test_a_reply_sends_only_the_quoted_rows_address(project_chat, tmp_path):
    page, expect = project_chat["page"], project_chat["expect"]
    answer = page.locator('#chat-messages > .chat-bubble.assistant').first
    answer.locator('.chat-message-reply').click()
    strip = page.locator('.chat-quote--composer')
    expect(strip).to_have_count(1)
    expect(strip.locator('.chat-quote-author')).to_have_text("Ouroboros")
    expect(page.locator('#chat-input')).to_be_focused()
    page.keyboard.press("Escape")
    expect(strip).to_have_count(0)
    answer.locator('.chat-message-reply').click()
    page.screenshot(path=_evidence(tmp_path, project_chat, "reply-strip"), animations="disabled")
    page.fill('#chat-input', "Which runs disagree?")
    page.locator('#chat-send').click()
    frame = page.evaluate("() => sentFrames.at(-1)")
    assert frame["reply_to"] == "row:1@2026-10-08T13:57:20Z#bbbbbbbbbbbb"
    assert frame["content"] == "Which runs disagree?"
    expect(strip).to_have_count(0)
    echo = page.locator(f'.chat-bubble.user[data-client-message-id="{frame["client_message_id"]}"]')
    expect(echo.locator('.chat-quote .chat-quote-author')).to_have_text("Ouroboros")
    # A work card's Reply is its row menu's (owner 17.10, Q9 B); it quotes the work by its task address.
    _card(page, "creation").locator('[data-live-title]').click(button="right")
    page.locator('.chat-row-menu:not([data-leaving]) [data-row-action="reply"]').click()
    page.fill('#chat-input', "Stop after the third run")
    page.locator('#chat-send').click()
    assert page.evaluate("() => sentFrames.at(-1).reply_to") == "task:creation"


SUCCESSOR = "creation-cnext"
WORK = "Review runner › integration"  # the work's first name: the ended episode's own
# DESIGN "Work portal": each episode has its own name, time and past outcome on one meta line (task_episodes.js
# renderEpisodes, #1979). Whether that line names the work is the owner's open call (#1999): this is the one place.
EPISODES_NAME_THEIR_WORK = True


def _episode(number, span, outcome="", name=""):
    """An episode's meta line as the rule above writes it; an episode the fixture never named has no name part."""
    return " · ".join(part for part in (f"Episode {number}", name if EPISODES_NAME_THEIR_WORK else "", span, outcome)
                      if part)


def _interrupted(**extra):
    return {"task_id": "creation", "status": "failed", "reason_code": "model_provider_failed",
            "terminal_time": {"occurred_at": "2026-10-08T14:09:00Z", "attempt": {"started_at": "2026-10-08T13:57:33Z"}},
            **extra}


def _continued():
    binding = {"project_id": "alpha", "chat_id": 7}
    return _interrupted(
        continued_by={"state": "admitted", "successor_task_id": SUCCESSOR, "action_nonce": "n-1", "binding": binding},
        continuation_offer={"eligible": False, "refusal": "already_continued", "state": "admitted",
                            "successor_task_id": SUCCESSOR})


def _room_continue(chat):
    """The owner pressed Continue in Project Alpha's room: the predecessor's record names its admitted successor and
    the all-chat census lists that successor in the room (Main receives none of its frames)."""
    chat["details"]["creation"] = _continued()
    state = chat["state"]
    state["active_chat_activities"] = [row for row in state["active_chat_activities"] if row["activity_id"] != "creation"]
    state["active_chat_activities"].append({
        "activity_id": SUCCESSOR, "project_id": "alpha", "project_name": "Project Alpha", "chat_id": 7,
        "kind": "managed_task", "phase": "working", "started_at": 1791468000, "last_line": "Continuing the audit",
        "last_line_at": "2026-10-08T14:12:00Z", "tool_calls": 3, "cancelable": True})
    return state


def _visible_cards(page):
    return page.evaluate("""() => [...document.querySelectorAll('#chat-messages > .chat-live-card')]
        .filter((card) => card.getClientRects().length).map((card) => card.dataset.taskId)""")


@pytest.mark.parametrize("journey", ["continued-later", "same-snapshot", "reload"])
def test_main_portal_follows_a_continue_made_in_the_project_room(project_chat, journey, tmp_path):
    """Owner QB = A and 16.10 Q1 = A: a Main request moved into Project Alpha, interrupted, then continued in the room.
    Main's ONE card for that work shows the admitted successor as its current episode (chip, line, meta, its Stop…),
    keeps the work's place, first name and Project reference, and folds the ended episode into one meta line: live
    (the census arrives while Main is open) and after a Main reload."""
    page, expect = project_chat["page"], project_chat["expect"]
    creation = _card(page, "creation")
    expect(creation.locator('.chat-live-phase').first).to_have_text("Working")
    assert _visible_cards(page) == ["creation", "transfer"]
    if journey == "continued-later":
        # The work fails first: Main reads its record and offers Continue on the same card.
        project_chat["details"]["creation"] = _interrupted(
            continuation_offer={"eligible": True, "cause": "model_provider_failed"})
        state = project_chat["state"]
        state["active_chat_activities"] = [row for row in state["active_chat_activities"] if row["activity_id"] != "creation"]
        page.evaluate("s => chat.hydrateStateSnapshot(s)", with_work_fields(state))
        expect(creation.locator('.chat-live-phase').first).to_have_text("Failed", timeout=15000)
        expect(creation.locator('[data-continue-task]')).to_have_text("Continue")
    state = _room_continue(project_chat)
    if journey == "reload":
        page.reload()
        page.wait_for_function("() => window.chat && document.querySelector('#chat-messages .chat-bubble')")
    page.evaluate("s => chat.hydrateStateSnapshot(s)", with_work_fields(state))
    card = _card(page, SUCCESSOR)
    expect(card).to_be_visible(timeout=15000)
    expect(_card(page, "creation")).to_be_hidden()
    assert _visible_cards(page) == [SUCCESSOR, "transfer"], "one card for the work, at its first place"
    expect(card).to_have_attribute("data-portal", "1")
    expect(card.locator('[data-live-title]')).to_have_text("Review runner › integration")
    expect(card.locator('.chat-live-phase').first).to_have_text("Working")
    expect(card.locator('[data-live-activity]')).to_have_text("Continuing the audit")
    expect(card.locator('[data-live-meta]')).to_contain_text("3 tool calls")
    expect(card.locator('.chat-live-actions > button')).to_have_text(["Stop…"])
    door = card.locator(DOOR)
    expect(door).to_have_attribute("aria-label", "Open project Project Alpha")
    assert card.evaluate(LEADS), "the reference leads the action row"
    _assert_b_row(card.evaluate(_PORTAL_ROW), buttons=1)
    episodes = card.locator('.chat-live-episode')
    expect(episodes).to_have_count(2)
    expect(episodes.nth(0)).to_contain_text(_episode(1, "13:57–14:09", "Failed", name=WORK))
    expect(episodes.nth(1)).to_have_text(_episode(2, "since 14:00"))
    expect(episodes.nth(1)).to_have_attribute("data-current", "1")
    assert page.locator('#chat-messages [data-continue-task]:visible').count() == 0, "no Continued stands as its action"
    assert page.locator('.project-handoff, [data-system-type="project_started"], [data-system-type="project_handoff"]').count() == 0
    card.screenshot(path=_evidence(tmp_path, project_chat, f"continue-{journey}"), animations="disabled")


def _room_history(chat, rows):
    """The room's own history as the host serves it; `rows` replace the Main rows the fixture booted with."""
    chat["rows"][:] = rows
    coverage = chat["history"]["coverage"]
    coverage["upper"]["chat"] = coverage["spans"]["chat"]["to"] = len(rows)


def _room_line(task_id, text, ts, **extra):
    return {"role": "assistant", "is_progress": True, "narration": True, "chat_id": 7, "task_id": task_id, "text": text,
            "ts": ts, "history_id": f"r-{task_id}-{ts}", **extra}


@pytest.mark.parametrize("chat_id", [7], ids=["room"])
def test_room_card_follows_a_continue_pressed_in_the_room(project_chat, chat_id, tmp_path):
    """Owner QB = A in the Project room itself: Continue pressed on the room's card is the next episode of the SAME
    card there, live (the successor takes the work's place, first name, chip and controls; the ended episode is one
    line) and after a reload of the room."""
    page, expect, state = project_chat["page"], project_chat["expect"], project_chat["state"]
    owner = {"role": "user", "text": "Audit the review runner here", "chat_id": 7, "source": "web",
             "ts": "2026-10-08T13:57:00Z", "history_id": "r-0", "row_address": "row:7@2026-10-08T13:57:00Z#cccccccccccc"}
    first_line = _room_line("creation", "Comparing the reviewer verdicts across the last three runs.", "2026-10-08T13:57:33Z",
                            suggested_name="Review runner › integration", cancelable=True)
    _room_history(project_chat, [owner, first_line])
    page.reload()
    creation = _card(page, "creation")
    expect(creation).to_be_visible(timeout=15000)
    page.evaluate("s => chat.hydrateStateSnapshot(s)", with_work_fields(state))
    # The work fails in its room: the room reads its record and offers Continue on the same card.
    project_chat["details"]["creation"] = _interrupted(continuation_offer={"eligible": True, "cause": "model_provider_failed"})
    page.evaluate("""() => emit('chat', {chat_id: 7, role: 'system', system_type: 'task_summary', task_id: 'creation',
        status: 'failed', reason_code: 'model_provider_failed', outcome_axes: {lifecycle: {status: 'failed'}},
        content: 'The model provider failed.', ts: '2026-10-08T14:09:00Z'})""")
    state["active_chat_activities"] = [row for row in state["active_chat_activities"] if row["activity_id"] != "creation"]
    page.evaluate("s => chat.hydrateStateSnapshot(s)", with_work_fields(state))
    expect(creation.locator('.chat-live-phase').first).to_have_text("Failed", timeout=15000)
    expect(creation.locator('[data-continue-task]')).to_have_text("Continue")
    # The owner presses Continue in the room; the host admits the successor and its first line reaches the room.
    project_chat["continues"]["creation"] = {"ok": True, "successor_task_id": SUCCESSOR, "status": "scheduled"}
    project_chat["details"]["creation"] = _continued()
    creation.locator('[data-continue-task]').click()
    expect(creation.locator('[data-continue-task]')).to_have_text("Continued")
    page.evaluate("""id => emit('chat', {chat_id: 7, role: 'assistant', is_progress: true, narration: true, task_id: id,
        content: 'Continuing the audit', ts: '2026-10-08T14:12:00Z', cancelable: true})""", SUCCESSOR)
    state["active_chat_activities"].append({"activity_id": SUCCESSOR, "chat_id": 7, "project_id": "alpha",
                                            "kind": "managed_task", "phase": "working", "started_at": 1791468720,
                                            "cancelable": True})
    page.evaluate("s => chat.hydrateStateSnapshot(s)", with_work_fields(state))

    def assert_one_card(label):
        card = _card(page, SUCCESSOR)
        expect(card).to_be_visible(timeout=15000)
        expect(_card(page, "creation")).to_be_hidden()
        assert _visible_cards(page) == [SUCCESSOR], f"{label}: one card for the work"
        assert page.evaluate("""id => { const cards = [...document.querySelectorAll('#chat-messages > *')];
            const at = cards.findIndex((node) => node.dataset.taskId === id);
            return cards[at - 1]?.dataset.taskId === 'creation'; }""", SUCCESSOR), f"{label}: at the work's first place"
        expect(card.locator('[data-live-title]')).to_have_text("Review runner › integration")
        expect(card.locator('.chat-live-phase').first).to_have_text("Working")
        expect(card.locator('[data-live-activity]')).to_have_text("Continuing the audit")
        expect(card.locator('.chat-live-actions > button')).to_have_text(["Stop…"])
        episodes = card.locator('.chat-live-episode')
        expect(episodes).to_have_count(2)
        expect(episodes.nth(0)).to_contain_text(_episode(1, "13:57–14:09", "Failed", name=WORK))
        expect(episodes.nth(1)).to_have_text(_episode(2, "since 14:12"))
        expect(episodes.nth(1)).to_have_attribute("data-current", "1")
        assert page.locator('#chat-messages [data-continue-task]:visible').count() == 0, "no Continued stands as its action"
        card.screenshot(path=_evidence(tmp_path, project_chat, f"room-continue-{label}"), animations="disabled")

    assert_one_card("live")
    # The room reloads: its history holds the ended episode's summary (with the admitted offer the host merges into
    # it) and the successor's line.
    summary = {"role": "system", "system_type": "task_summary", "chat_id": 7, "task_id": "creation",
               "text": "The model provider failed.", "ts": "2026-10-08T14:09:00Z", "history_id": "r-summary",
               "status": "failed", "task_terminal_status": "failed", "reason_code": "model_provider_failed",
               "outcome_axes": {"lifecycle": {"status": "failed"}}, "outcome_final": True,
               "terminal_time": {"occurred_at": "2026-10-08T14:09:00Z", "attempt": {"started_at": "2026-10-08T13:57:33Z"}},
               "continuation_offer": _continued()["continuation_offer"]}
    _room_history(project_chat, [owner, {**first_line, "task_terminal_status": "failed"}, summary,
                                 _room_line(SUCCESSOR, "Continuing the audit", "2026-10-08T14:12:00Z", cancelable=True)])
    page.reload()
    expect(_card(page, SUCCESSOR)).to_be_attached(timeout=15000)
    page.evaluate("s => chat.hydrateStateSnapshot(s)", with_work_fields(state))
    assert_one_card("reload")


# The reference as painted: its one fill, the glyph's own fill (none), and the glyph's and the name's ink against it.
_MARK_MATERIAL = """buttons => {
    const rgba = value => { const [r, g, b, a = 1] = String(value).match(/[\\d.]+/g).map(Number); return {r, g, b, a}; };
    const over = (top, base) => Object.fromEntries(['r', 'g', 'b'].map(k => [k, top[k] * top.a + base[k] * (1 - top.a)]));
    const probe = document.body.appendChild(document.createElement('i'));
    probe.style.color = 'var(--bg-primary)';
    const page = rgba(getComputedStyle(probe).color);
    // The Project ink where the reference lies: inside the dark Ouroboros voice, that voice's own (ui.css materials).
    const inkAt = node => { probe.style.color = getComputedStyle(node).getPropertyValue('--project').trim(); return getComputedStyle(probe).color; };
    const dark = document.documentElement.dataset.theme !== 'light';
    const painted = node => {
        const layers = [];
        for (let el = node; el; el = el.parentElement) {
            const style = getComputedStyle(el);
            // The shared tint is painted over its optional opaque backing (the answer fade passes behind it).
            if (el.classList.contains('chat-live-project-card-btn')) layers.unshift(rgba(style.getPropertyValue('--project-reference-fill')));
            layers.unshift(rgba(style.backgroundColor));
        }
        return layers.reduce((base, layer) => ({...over(layer, base), a: 1}), page);
    };
    const lum = c => ['r', 'g', 'b'].map(k => c[k] / 255).map(v => v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4)
        .reduce((sum, v, i) => sum + v * [0.2126, 0.7152, 0.0722][i], 0);
    const contrast = (a, b) => (Math.max(lum(a), lum(b)) + 0.05) / (Math.min(lum(a), lum(b)) + 0.05);
    const marks = buttons.map(button => {
        const icon = button.querySelector('.chat-live-project-icon'), name = button.querySelector('.chat-live-project-name');
        const fill = painted(button), beneath = painted(button.parentElement);
        return {host: button.classList.contains('chat-live-bound-pointer') ? 'footer'
                : button.closest('.msg-routing-annotation') ? 'receipt' : 'capsule', ink: inkAt(button),
            voice: dark && Boolean(button.closest('.chat-bubble.assistant, .chat-live-card')),
            tile: rgba(getComputedStyle(icon).backgroundColor).a,
            fill: rgba(getComputedStyle(button).getPropertyValue('--project-reference-fill')).a,
            glyph: getComputedStyle(icon).color, name: getComputedStyle(name).color,
            glyph_on_fill: contrast(rgba(getComputedStyle(icon).color), fill),
            fill_on_beneath: contrast(fill, beneath),
            name_on_fill: contrast(rgba(getComputedStyle(name).color), fill)};
    });
    probe.remove();
    return marks;
}"""


def test_project_reference_is_one_tinted_fill_with_no_tile_behind_its_glyph(project_chat):
    """DESIGN "Design bible" (principles 3, 4): hue says identity, fill strength importance, and a control has one
    fill. The Project reference (owner 17.10, Q2 C: one chip in two forms) is the Project's tint with the glyph, the
    name and the arrow in project ink on it, readable in each theme; the glyph stands on that fill with no tile of its
    own (a second fill read as a second button), and the tint stays far quieter than the name. Its hosts after one
    bubble per request: a moved work's portal (its action row's capsule), the capsule of a mirrored answer and of a
    System lifecycle row,
    and the owner's routing receipt line (the retired Project entries are gone)."""
    page, expect = project_chat["page"], project_chat["expect"]
    for answer in ("The review runner reports every reviewer in words.", None):
        row = {"role": "system", "system_type": "project_completion_summary", "chat_id": 1,
               "task_id": f"ended-{answer is None}", "project_id": "alpha", "project_name": "Project Alpha",
               "text": "Project Alpha · Done", "ts": "2026-10-08T14:01:00Z"}
        if answer:
            row["completion_answer"] = answer
        page.evaluate("r=>emit('chat',{...r,content:r.text})", row)
    expect(page.locator('.system-message-actions [data-intent="open-project"]')).to_have_count(2)
    # The owner's later words passed on to the moved work: his message's receipt line names its room.
    page.evaluate("""() => {
        emit('chat', {chat_id: 1, role: 'user', content: 'Also check the flaky reviewer', client_message_id: 'cm-passed',
            source: 'web', ts: '2026-10-08T14:02:00Z'});
        emit('message_annotation', {chat_id: 1, annotation_type: 'routing_ack', client_message_id: 'cm-passed',
            status: 'delivered', action: 'steer_task', project_id: 'alpha', project_name: 'Project Alpha',
            project_chat_id: 7, target: 'creation', target_titles: ['Review runner › integration']});
    }""")
    expect(page.locator('.msg-routing-annotation [data-intent="open-project"]')).to_have_count(1)
    expect(page.locator('#chat-messages > .chat-live-card > .chat-live-foot > .chat-live-foot-end > .chat-live-bound-pointer')).to_have_count(2)
    marks = page.locator('[data-intent="open-project"]').evaluate_all(_MARK_MATERIAL)
    assert sorted(mark["host"] for mark in marks) == ["capsule", "capsule", "footer", "footer", "receipt"]
    for mark in marks:
        assert mark["tile"] == 0, mark
        # A tint on the room and on white; inside the dark Ouroboros voice the voice's own opaque capsule step.
        assert mark["fill"] == 1 if mark["voice"] else 0 < mark["fill"] < 1, mark
        assert mark["glyph"] == mark["name"] == mark["ink"], mark
        assert mark["glyph_on_fill"] >= 3, mark
        assert mark["fill_on_beneath"] * 2 < mark["name_on_fill"], mark


@pytest.mark.parametrize("journey", ["live", "reload"])
def test_main_portal_follows_a_successor_that_already_finished(project_chat, journey, tmp_path):
    """Astra W2 R2: a moved work fails, is continued in its room, and the successor finishes before Main's next census
    (or Main is opened after it finished). The census lists neither ended work; the predecessor's record names the
    admitted successor and the successor's own record holds its outcome. Main's ONE card still follows it: the work's
    place, first name and Project reference, the ended episode as one line, the current episode settled from its record."""
    page, expect = project_chat["page"], project_chat["expect"]
    expect(_card(page, "creation")).to_be_visible()
    quick = "creation-cquick"
    binding = {"project_id": "alpha", "chat_id": 7}
    project_chat["details"]["creation"] = _interrupted(
        continued_by={"state": "admitted", "successor_task_id": quick, "action_nonce": "n-2", "binding": binding},
        continuation_offer={"eligible": False, "refusal": "already_continued", "state": "admitted", "successor_task_id": quick})
    project_chat["details"][quick] = {
        "task_id": quick, "status": "completed", "chat_id": 7, "project_id": "alpha",
        "outcome_axes": {"lifecycle": {"status": "completed"}, "execution": {"status": "ok"}},
        "terminal_time": {"occurred_at": "2026-10-08T14:12:00Z", "attempt": {"started_at": "2026-10-08T14:10:00Z"}}}
    state = project_chat["state"]
    state["task_bindings"][quick] = binding
    state["active_chat_activities"] = [row for row in state["active_chat_activities"] if row["activity_id"] != "creation"]
    page.evaluate("s => chat.hydrateStateSnapshot(s)", with_work_fields(state))
    # Its final reaches Main as the author's own row; that row stays a row, the card settles from the record.
    page.evaluate("""id => emit('chat', {chat_id: 1, task_id: id, role: 'assistant', content: 'The audit is complete.',
        project_id: 'alpha', project_name: 'Project Alpha', system_type: 'project_completion_summary',
        ts: '2026-10-08T14:12:00Z'})""", quick)
    if journey == "reload":
        page.reload()
        page.wait_for_function("() => window.chat && document.querySelector('#chat-messages .chat-bubble')")
        page.evaluate("s => chat.hydrateStateSnapshot(s)", with_work_fields(state))
    card = _card(page, quick)
    expect(card).to_be_visible(timeout=15000)
    expect(_card(page, "creation")).to_be_hidden()
    assert _visible_cards(page) == [quick, "transfer"], "one card for the work, at its first place"
    expect(card).to_have_attribute("data-portal", "1")
    expect(card.locator('[data-live-title]')).to_have_text("Review runner › integration")
    expect(card.locator('.chat-live-phase').first).to_have_text("Done")
    expect(card.locator('.chat-live-actions > button')).to_have_count(0)
    door = card.locator(DOOR)
    expect(door).to_have_attribute("aria-label", "Open project Project Alpha")
    # A settled work has no control: its row still stands for the door, which keeps the row's start.
    expect(door).to_be_visible()
    _assert_b_row(card.evaluate(_PORTAL_ROW), buttons=0)
    episodes = card.locator('.chat-live-episode')
    expect(episodes).to_have_count(2)
    expect(episodes.nth(0)).to_contain_text(_episode(1, "13:57–14:09", "Failed", name=WORK))
    expect(episodes.nth(1)).to_have_text(_episode(2, "14:10–14:12"))
    expect(episodes.nth(1)).to_have_attribute("data-current", "1")
    assert page.locator('#chat-messages [data-continue-task]:visible').count() == 0, "no Continued stands as its action"
    card.screenshot(path=_evidence(tmp_path, project_chat, f"finished-successor-{journey}"), animations="disabled")


@pytest.mark.parametrize("journey", ["live", "reload"])
def test_an_already_failed_portal_follows_a_later_fast_continue(project_chat, journey, tmp_path):
    """Astra W3: Main already read the moved work's Failed record and offers Continue; the owner continues in the room
    and the successor finishes before any census lists it. Main receives its new binding, its mirrored final and complete
    snapshots with no live Alpha work. The settled portal reads its own record once on those observations and follows
    its admitted link: one card for the work, settled Done, live and after a reload."""
    page, expect, state = project_chat["page"], project_chat["expect"], project_chat["state"]
    old = _card(page, "creation")
    expect(old).to_be_visible()
    project_chat["details"]["creation"] = _interrupted(continuation_offer={"eligible": True, "cause": "model_provider_failed"})
    state["active_chat_activities"] = [row for row in state["active_chat_activities"] if row["activity_id"] != "creation"]
    page.evaluate("s => chat.hydrateStateSnapshot(s)", with_work_fields(state))
    expect(old.locator('.chat-live-phase').first).to_have_text("Failed", timeout=15000)
    expect(old.locator('[data-continue-task]')).to_have_text("Continue")
    later = "creation-clater"
    binding = {"project_id": "alpha", "chat_id": 7}
    project_chat["details"]["creation"] = _interrupted(
        continued_by={"state": "admitted", "successor_task_id": later, "binding": binding},
        continuation_offer={"eligible": False, "refusal": "already_continued", "state": "admitted", "successor_task_id": later})
    project_chat["details"][later] = {
        "task_id": later, "status": "completed", "chat_id": 7, "project_id": "alpha",
        "outcome_axes": {"lifecycle": {"status": "completed"}, "execution": {"status": "ok"}},
        "terminal_time": {"occurred_at": "2026-10-08T14:12:00Z", "attempt": {"started_at": "2026-10-08T14:10:00Z"}}}
    state["task_bindings"][later] = binding
    page.evaluate("""id => emit('chat', {chat_id: 1, task_id: id, role: 'assistant', content: 'The audit is complete.',
        project_id: 'alpha', project_name: 'Project Alpha', system_type: 'project_completion_summary',
        ts: '2026-10-08T14:12:00Z'})""", later)
    for _ in range(3):
        page.evaluate("s => chat.hydrateStateSnapshot(s)", with_work_fields(state))
    if journey == "reload":
        page.reload()
        page.wait_for_function("() => window.chat && document.querySelector('#chat-messages .chat-bubble')")
        page.evaluate("s => chat.hydrateStateSnapshot(s)", with_work_fields(state))
    card = _card(page, later)
    expect(card).to_be_visible(timeout=15000)
    expect(_card(page, "creation")).to_be_hidden()
    assert _visible_cards(page) == [later, "transfer"], "one card for the work, at its first place"
    expect(card.locator('[data-live-title]')).to_have_text("Review runner › integration")
    expect(card.locator('.chat-live-phase').first).to_have_text("Done")
    expect(card.locator('.chat-live-actions > button')).to_have_count(0)
    expect(card.locator(DOOR)).to_have_attribute("aria-label", "Open project Project Alpha")
    expect(card.locator(DOOR)).to_be_visible()
    episodes = card.locator('.chat-live-episode')
    expect(episodes).to_have_count(2)
    expect(episodes.nth(0)).to_contain_text(_episode(1, "13:57–14:09", "Failed", name=WORK))
    expect(episodes.nth(1)).to_have_text(_episode(2, "14:10–14:12"))
    assert page.locator('#chat-messages [data-continue-task]:visible').count() == 0, "no Continue stands as its action"
    card.screenshot(path=_evidence(tmp_path, project_chat, f"later-continue-{journey}"), animations="disabled")


def test_empty_output_recovery_preserves_failed_card_and_continue(project_chat, tmp_path):
    """The new terminal cause uses the real card/replay/Continue consumers, not a new UI."""
    from ouroboros.owner_continue import continuation_offer
    from ouroboros.project_dialogue import TASK_CAUSE_PHRASES
    page, expect, state = project_chat["page"], project_chat["expect"], project_chat["state"]
    reason = "output_recovery_exhausted"
    record = _interrupted(reason_code=reason, root_task_id="creation",
        outcome_axes={"execution": {"status": "infra_failed"}, "objective": {"status": "not_evaluated"}})
    record["continuation_offer"] = continuation_offer(record, "creation")
    assert record["continuation_offer"]["eligible"]
    project_chat["details"]["creation"] = record
    state["active_chat_activities"] = [r for r in state["active_chat_activities"] if r["activity_id"] != "creation"]
    for replay in (False, True):
        if replay:
            page.reload()
            page.wait_for_function("() => window.chat && document.querySelector('#chat-messages .chat-bubble')")
        page.evaluate("s => chat.hydrateStateSnapshot(s)", with_work_fields(state))
        card = _card(page, "creation")
        expect(card.locator('.chat-live-phase').first).to_have_text("Failed", timeout=15000)
        expect(card.locator('[data-continue-task]')).to_have_text("Continue")
        expect(card).to_contain_text(TASK_CAUSE_PHRASES[reason])
    card.screenshot(path=_evidence(tmp_path, project_chat, "output-recovery"), animations="disabled")
    admitted = []

    def admit(route):
        # An accepted Continue persists its link before replying; the portal immediately rereads that record.
        payload = route.request.post_data_json
        assert route.request.method == "POST" and payload["action_nonce"], payload
        admitted.append(payload)
        binding = {"project_id": "alpha", "chat_id": 7}
        record["continued_by"] = {"state": "admitted", "successor_task_id": SUCCESSOR,
                                  "action_nonce": payload["action_nonce"], "binding": binding}
        record["continuation_offer"] = continuation_offer(record, "creation")
        project_chat["details"][SUCCESSOR] = {"task_id": SUCCESSOR, "status": "pending", **binding}
        state["task_bindings"][SUCCESSOR] = binding
        state["active_chat_activities"].append({"activity_id": SUCCESSOR, "kind": "managed_task", "phase": "queued",
                                                "started_at": 1791468600, "cancelable": True, **binding})
        route.fulfill(json={"ok": True, "successor_task_id": SUCCESSOR, "status": "scheduled"})

    page.route("**/api/tasks/creation/continue", admit)
    with page.expect_response(lambda r: r.url.endswith('/api/tasks/creation/continue')) as accepted:
        card.locator('[data-continue-task]').click()
    assert accepted.value.json()["ok"]
    for replay in (False, True):
        if replay:
            page.reload()
            page.wait_for_function("() => window.chat && document.querySelector('#chat-messages .chat-bubble')")
        page.evaluate("s => chat.hydrateStateSnapshot(s)", with_work_fields(state))
        successor = _card(page, SUCCESSOR)
        expect(successor).to_be_visible(timeout=15000)
        expect(successor.locator('.chat-live-phase').first).to_have_text("Queued")
        expect(card).to_be_hidden()
        assert _visible_cards(page) == [SUCCESSOR, "transfer"], "Continue keeps one card for the work"
        episodes = successor.locator('.chat-live-episode')
        expect(episodes).to_have_count(2)
        expect(episodes.first).to_contain_text(_episode(1, "13:57–14:09", "Failed", name=WORK))
        expect(episodes.first).to_contain_text(TASK_CAUSE_PHRASES[reason])
        expect(successor.locator('[data-continue-task]')).to_have_count(0)
    assert len(admitted) == 1 and not record["continuation_offer"]["eligible"], admitted
