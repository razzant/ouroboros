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


@pytest.fixture(params=[
    ("chromium", "light", 1360), ("chromium", "dark", 390),
    ("webkit", "light", 390), ("webkit", "dark", 1360),
], ids=["chromium-light-desktop", "chromium-dark-mobile", "webkit-light-mobile", "webkit-dark-desktop"])
def project_chat(request):
    engine, theme, width = request.param
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
    conversions, details, requests = {}, {}, []
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
        if path == "/api/projects/from-task":
            payload = route.request.post_data_json
            requests.append(payload)
            data = conversions[payload["task_id"]]
        if path.startswith("/api/tasks/"):
            task_id = path.split("/")[-1]
            data = details.get(task_id, {"task_id": task_id, "status": "running"})
        route.fulfill(content_type="application/json", body=json.dumps(data))

    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(WEB)))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with playwright.sync_playwright() as pw:
            browser = getattr(pw, engine).launch(headless=True)
            try:
                page = browser.new_page(viewport={"width": width, "height": 850}, has_touch=width < 980, timezone_id="UTC")
                page.add_init_script(f"localStorage.setItem('ouroboros.theme', {json.dumps(theme)});")
                page.add_init_script("""Object.defineProperty(navigator, 'clipboard', {
                    value: {writeText: async text => {window.copiedText = text;}}});""")
                page.route("**/api/**", respond)
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto(f"http://127.0.0.1:{server.server_port}/")
                yield {"page": page, "expect": playwright.expect, "rows": rows, "state": state,
                       "conversions": conversions, "details": details, "requests": requests,
                       "engine": engine, "theme": theme, "width": width}
                assert not errors
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _evidence_dir(tmp_path):
    path = Path(os.environ.get("OUROBOROS_BROWSER_EVIDENCE_OUT") or tmp_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _shell_geometry(card):
    return card.evaluate("""root => {
        const bounds = root.getBoundingClientRect(), css = getComputedStyle(root);
        const measure = selector => {
            const node = root.querySelector(selector), box = node.getBoundingClientRect();
            const style = getComputedStyle(node);
            return {left: box.left - bounds.left, top: box.top - bounds.top,
                right: box.right - bounds.left, bottom: box.bottom - bounds.top,
                width: box.width, height: box.height, overflow: node.scrollWidth > node.clientWidth + 1,
                font: style.fontSize, line: style.lineHeight};
        };
        return {width: bounds.width, height: bounds.height, overflow: root.scrollWidth > root.clientWidth + 1,
            surface: Object.fromEntries(['paddingTop', 'paddingRight', 'paddingBottom', 'paddingLeft',
                'borderTopWidth', 'borderRightWidth', 'borderBottomWidth', 'borderLeftWidth',
                'borderColor', 'borderRadius', 'backgroundColor', 'boxShadow'].map(key => [key, css[key]])),
            title: measure('.project-handoff-title'), phase: measure('.project-handoff-phase'),
            reference: measure('[data-intent="open-project"]'), footer: measure('.project-handoff-footer'),
            time: measure('.msg-time'), copy: measure('.chat-message-copy')};
    }""")


def _assert_shared_shell(cards, *, narrow):
    measurements = [_shell_geometry(card) for card in cards]
    baseline = measurements[0]
    for result in measurements:
        assert result["surface"] == baseline["surface"]
        for dimension in ("width", "height"):
            assert result[dimension] == pytest.approx(baseline[dimension], abs=1)
        assert result["surface"]["paddingTop"] == "12px"
        assert result["surface"]["paddingLeft"] == "16px"
        assert not result["overflow"]
        for part in ("title", "phase", "reference", "footer", "time", "copy"):
            geometry = result[part]
            assert geometry["left"] >= 16 and geometry["right"] <= result["width"] - 16
            assert geometry["top"] >= 12 and geometry["bottom"] <= result["height"] - 12
            if part != "reference":
                assert not geometry["overflow"], (part, result)
            for dimension in ("left", "top", "width", "height"):
                assert geometry[dimension] == pytest.approx(baseline[part][dimension], abs=1), part
            assert geometry["font"] == baseline[part]["font"]
            assert geometry["line"] == baseline[part]["line"]
        # A long Project name uses its available row, rather than the generic 32ch cap.
        assert result["reference"]["width"] >= result["width"] - 35
        if narrow:
            assert result["phase"]["bottom"] < result["title"]["top"]
            assert result["phase"]["left"] == pytest.approx(result["title"]["left"], abs=1)
        else:
            assert result["title"]["right"] < result["phase"]["left"]
            assert result["phase"]["top"] < result["title"]["top"] + float(result["title"]["line"].removesuffix("px"))
    return measurements


def _emit_convertible_work(page, task_id, title):
    page.evaluate("""({task, title}) => {
        emit('chat', {chat_id: 1, task_id: task, role: 'assistant', is_progress: true,
            content: 'Inspecting the requested work', ts: '2026-10-08T13:56:00Z'});
        emit('task_named', {task_id: task, suggested_name: title});
    }""", {"task": task_id, "title": title})
    return page.locator(f'.chat-live-card[data-task-id="{task_id}"]')


def test_conversion_replay_and_creation_share_the_whole_project_shell(project_chat, tmp_path):
    page, expect = project_chat["page"], project_chat["expect"]
    rows, state = project_chat["rows"], project_chat["state"]
    title = "Review runner integration, preserve the complete long work title and its visible hierarchy " * 3
    title = title.strip()
    name = "Project with a deliberately long name that uses its available line and remains fully accessible " * 3
    name = name.strip()
    ts = "2026-10-08T13:58:00Z"
    for row in rows:
        row.update(task_name=title, project_name=name, ts=ts)
    page.reload()
    page.emulate_media(reduced_motion="reduce")
    start = page.locator('[data-system-type="project_started"]')
    replay = page.locator('[data-handoff-id="transfer-beta"]')
    expect(start.locator('.project-handoff-title')).to_have_text(title)
    cards = [start, replay]
    saved_rows = []
    for order in ("after", "before"):
        task_id, project_id, handoff_id = f"converted-{order}", f"room-{order}", f"handoff-{order}"
        card = _emit_convertible_work(page, task_id, title)
        original_position = card.get_attribute("data-ts")
        assert original_position
        state["active_chat_activities"].append({"activity_id": task_id, "chat_id": 1, "phase": "working"})
        page.evaluate("s=>chat.hydrateStateSnapshot(s)", state)
        project_chat["conversions"][task_id] = {
            "project": {"id": project_id, "name": name, "chat_id": 9},
            "binding": {"project_id": project_id}, "handoff_id": handoff_id, "handoff_receipt": "durable"}
        receipt = {"role": "system", "system_type": "project_handoff", "chat_id": 1,
                   "task_id": task_id, "project_id": project_id, "project_name": name,
                   "handoff_id": handoff_id, "task_name": title, "text": "Stored transfer", "ts": ts}
        if order == "before":
            page.evaluate("r=>emit('chat',{...r,content:r.text})", receipt)
        card.locator('[data-turn-into-project]').click()
        expect(card).to_have_attribute("data-project-created", "1")
        assert card.locator('[data-turn-into-project]').count() == 0
        if order == "after":
            expect(card.locator('.msg-time')).to_be_hidden()
            page.evaluate("r=>emit('chat',{...r,content:r.text})", receipt)
        expect(card.locator('.msg-time')).to_have_attribute("title", "Oct 8, 2026 at 13:58")
        expect(page.locator(f'.chat-bubble[data-handoff-id="{handoff_id}"]')).to_be_hidden()
        assert card.get_attribute("data-ts") == original_position, "folding must not move the converted transcript node"
        saved_rows.append(receipt)
        cards.append(card)
    assert [request["task_id"] for request in project_chat["requests"]] == ["converted-after", "converted-before"]
    page.wait_for_function("""() => {
        const cards = [...document.querySelectorAll('.project-handoff:not([hidden])')];
        return cards.every(card => ['backgroundColor', 'borderColor', 'boxShadow'].every(
            key => getComputedStyle(card)[key] === getComputedStyle(cards[0])[key]));
    }""")
    measurements = _assert_shared_shell(cards, narrow=project_chat["width"] < 980)
    keyboard_trace = []
    for card in cards:
        assert card.locator('.message, .sender').count() == 0, "every host mounts the same content shell"
        expect(card.locator('.project-handoff-title')).to_have_text(title)
        reference = card.locator('[data-intent="open-project"]')
        expect(reference).to_have_attribute("aria-label", f"Open project {name}")
        assert reference.locator('.chat-live-project-name').evaluate("e=>e.scrollWidth>e.clientWidth")
        reference.focus()
        expect(reference).to_be_focused()
        page.keyboard.press("Tab")
        keyboard_trace.append(page.evaluate("""() => ({tag: document.activeElement.tagName,
            id: document.activeElement.id, className: document.activeElement.className})"""))
        # As in the segmented-control WebKit flow, establish keyboard modality
        # explicitly; platform Tab preferences need not visit native buttons.
        page.keyboard.press("Shift")
        reference.focus()
        expect(reference).to_be_focused()
        ring = reference.evaluate("""e=>{const s=getComputedStyle(e), r=e.getBoundingClientRect(),
            p=e.closest('.project-handoff').getBoundingClientRect(); return {visible:e.matches(':focus-visible'),
            width:s.outlineWidth, offset:s.outlineOffset,
            clearance:Math.min(r.left-p.left,p.right-r.right,r.top-p.top,p.bottom-r.bottom)};}""")
        assert ring == {"visible": True, "width": "2px", "offset": "2px", "clearance": ring["clearance"]}
        assert ring["clearance"] >= 4
        page.keyboard.press("Enter")
        assert page.evaluate("openedProject.project.id") == card.get_attribute("data-project-id")
        assert page.evaluate("openedProject.task_id") == card.get_attribute("data-task-id")
        page.evaluate("copiedText = ''")
        card.locator('.chat-message-copy').focus()
        expect(card.locator('.chat-message-copy')).to_be_focused()
        page.keyboard.press("Enter")
        page.wait_for_function("expected=>window.copiedText===expected", arg=f"{title}\n{name}")
    evidence = _evidence_dir(tmp_path)
    suffix = "-".join(str(project_chat[key]) for key in ("engine", "theme", "width"))
    page.locator('#toast-stack .toast').evaluate_all("nodes=>nodes.forEach(node=>node.click())")
    expect(page.locator('#toast-stack .toast')).to_have_count(0)
    page.screenshot(path=str(evidence / f"project-shell-{suffix}.png"), animations="disabled")
    for label, card in (("started", start), ("replay", replay), ("converted", cards[2])):
        card.screenshot(path=str(evidence / f"project-shell-{suffix}-{label}.png"), animations="disabled")
    (evidence / f"project-shell-{suffix}.json").write_text(json.dumps(measurements, indent=2), encoding="utf-8")
    (evidence / f"project-keyboard-{suffix}.json").write_text(json.dumps(keyboard_trace, indent=2), encoding="utf-8")
    if project_chat["width"] >= 980:
        # A narrow pane on a desktop must stack too: the component, not viewport, decides.
        page.locator('#chat-messages').evaluate("e=>e.style.width='420px'")
        _assert_shared_shell(cards, narrow=True)
        cards[2].screenshot(path=str(evidence / f"project-shell-{suffix}-narrow-pane.png"), animations="disabled")
        page.locator('#chat-messages').evaluate("e=>e.style.removeProperty('width')")

    # Known outcomes and unfinished finalization retain both chips after conversion.
    fact = state["active_chat_activities"][-2]
    fact.update(phase="finalizing", root_phase_checkpoint={"post_task_synthesis": "running"})
    for status, label in (("failed", "Failed"), ("completed", "Done")):
        fact["status"] = status
        page.evaluate("s=>chat.hydrateStateSnapshot(s)", state)
        expect(cards[2].locator('.chat-live-phase').first).to_have_text(label)
        expect(cards[2].locator('.chat-live-phase-secondary')).to_have_text("Finalizing…")

    # Finishing BEFORE conversion used to let the task's red shell override Project ink.
    finished = _emit_convertible_work(page, "finished", title)
    terminal = {"task_id": "finished", "chat_id": 1, "type": "task_done", "status": "completed",
                "root_phase_checkpoint": {"post_task_synthesis": "completed"}, "ts": "2026-10-08T14:00:00Z"}
    project_chat["details"]["finished"] = terminal
    page.evaluate("data=>emit('log',{chat_id:1,data})", terminal)
    expect(finished).to_have_attribute("data-finished", "1")
    project_chat["conversions"]["finished"] = {"project": {"id": "finished-room", "name": name},
        "binding": {"project_id": "finished-room"}, "handoff_id": "finished-handoff", "handoff_receipt": "durable"}
    finished.locator('[data-turn-into-project]').click()
    expect(finished.locator('.project-handoff-title')).to_have_text(title)
    page.wait_for_function("""() => {
        const card = document.querySelector('[data-task-id="finished"].project-handoff');
        const reference = document.querySelector('[data-system-type="project_started"]');
        return card && ['backgroundColor','borderColor','boxShadow'].every(key=>getComputedStyle(card)[key]===getComputedStyle(reference)[key]);
    }""")
    assert _shell_geometry(finished)["surface"] == _shell_geometry(start)["surface"]
    assert page.evaluate("""() => {
        const copies = [...document.querySelectorAll('.chat-live-card.project-handoff .chat-message-copy')];
        chat.destroy(); window.copiedText = '';
        copies.forEach(button => button.click());
        return copies.length;
    }""") == 3
    assert page.evaluate("copiedText") == "", "retired converted cards must release their copy listeners"
    rows.extend(saved_rows)
    page.reload()
    for receipt in saved_rows:
        restored = page.locator(f'[data-handoff-id="{receipt["handoff_id"]}"]')
        expect(restored).to_be_visible()
        expect(restored.locator('.msg-time')).to_have_attribute("title", "Oct 8, 2026 at 13:58")
        restored.locator('.chat-message-copy').click()
        page.wait_for_function("expected=>window.copiedText===expected", arg=f"{title}\n{name}")


def test_project_work_entries_share_live_phase_title_and_motion(project_chat, tmp_path):
    page, expect = project_chat["page"], project_chat["expect"]
    rows, state = project_chat["rows"], project_chat["state"]
    engine, theme, width = (project_chat[key] for key in ("engine", "theme", "width"))
    start = page.locator('[data-system-type="project_started"]')
    transfer = page.locator('[data-system-type="project_handoff"]')
    expect(start.locator('.chat-live-phase')).to_have_text("Working")
    expect(transfer.locator('.chat-live-phase')).to_have_text("Working")
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
        expect(chip).to_have_text(label)
        assert chip.get_attribute("data-motion") == moving
        assert chip.evaluate("e=>getComputedStyle(e).animationName") == ("thinking-pulse" if moving == "1" else "none")
    page.evaluate("s=>chat.hydrateStateSnapshot(s)", state)
    page.screenshot(path=str(_evidence_dir(tmp_path) / f"project-work-{engine}-{theme}-{width}.png"))
    assert start.evaluate("e=>e.scrollWidth<=e.clientWidth+1")
    start.locator('[data-intent="open-project"]').click()
    assert page.evaluate("openedProject.task_id") == "creation"
    assert page.evaluate("openedProject.project.id") == "alpha"

    page.reload()
    expect(start.locator('.chat-live-phase')).to_have_text("Working")
    assert start.locator('.chat-live-phase').evaluate("e=>getComputedStyle(e).animationName") == "thinking-pulse"
    receipt = {**rows[0], "system_type": "project_handoff", "handoff_id": "transfer-alpha"}
    rows.append(receipt)
    page.evaluate("r=>emit('chat',{...r,content:r.text})", receipt)
    expect(start).to_be_hidden()
    assert page.locator('.project-handoff:not([hidden])').count() == 2
    page.reload()
    expect(start).to_have_count(1)
    expect(start).to_be_hidden()
    expect(page.locator('.project-handoff:not([hidden])')).to_have_count(2)

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
    expect(child_chip).to_have_text("Working")
    card = page.locator('.chat-live-card[data-task-id="observed"]')
    outcome = card.locator('[data-live-phase]').first
    secondary = card.locator('[data-live-phase-secondary]').first
    fact = {"activity_id": "observed", "chat_id": 1, "kind": "managed_task",
            "phase": "finalizing", "status": "failed",
            "root_phase_checkpoint": {"post_task_synthesis": "running"}}
    page.evaluate("a=>chat.hydrateStateSnapshot({supervisor_ready:true,active_chat_activities_complete:true,active_chat_activities:[a]})", fact)
    expect(outcome).to_have_text("Failed")
    expect(secondary).to_have_text("Finalizing…")
    assert secondary.get_attribute("data-motion") == "1"
    fact["status"] = "completed"
    page.evaluate("a=>chat.hydrateStateSnapshot({supervisor_ready:true,active_chat_activities_complete:true,active_chat_activities:[a]})", fact)
    expect(outcome).to_have_text("Done")
    state.update(active_chat_activities=[], active_chat_activities_complete=False)
    page.evaluate("connection(false)")
    expect(outcome).to_have_text("Done")
    expect(secondary).to_have_text("Activity unconfirmed")
    assert secondary.get_attribute("data-motion") == "0"
    assert secondary.evaluate("e=>getComputedStyle(e).animationName") == "none"
    expect(child_chip).to_have_text("Activity unconfirmed")
    page.evaluate("connection(true)")
    expect(secondary).to_have_text("Activity unconfirmed")
    page.evaluate("a=>chat.hydrateStateSnapshot({supervisor_ready:true,active_chat_activities_complete:true,active_chat_activities:[a]})", fact)
    expect(secondary).to_have_text("Finalizing…")
    assert secondary.get_attribute("data-motion") == "1"
    expect(child_chip).to_have_text("Activity unconfirmed")
    page.evaluate("r=>emit('chat',r)", child_frame)
    expect(child_chip).to_have_text("Working")
    assert child_chip.get_attribute("data-motion") == "1"

    ended = {**fact, "activity_id": "late-card", "status": "cancelled"}
    page.evaluate("a=>chat.hydrateStateSnapshot({supervisor_ready:true,active_chat_activities_complete:true,active_chat_activities:[a]})", ended)
    page.evaluate("""()=>emit('chat',{chat_id:1,task_id:'late-card',role:'assistant',
        is_progress:true,narration:true,content:'Earlier progress arriving later',ts:'2026-10-08T14:00:00Z'})""")
    late = page.locator('.chat-live-card[data-task-id="late-card"]')
    expect(late.locator('[data-live-title]')).to_have_text("Earlier progress arriving later")
    expect(late.locator('[data-live-phase]')).to_have_text("Cancelled")
    assert late.locator('[data-live-phase]').get_attribute("data-motion") == "0"
    assert late.get_attribute("data-finished") == "1"
