"""A real module widget's fault reaches the host's event log from a real page (#1794), in Chromium and WebKit.

A reviewed skill's frame throws at load: its card shows the fault and ``GET /api/logs/events`` reads back one
``widget_fault`` row under the mounted identity. A frame that reads its nonce and posts thirty forged faults with
another skill's name gets at most ten reports out of its mount, all under its own name. Rejections and CSP faults
take the same path. A Keep running frame still reports when hidden; wrong-nonce/source and queued post-disposal
faults do not. An inline module belongs to its message, with no new chat or notification from its fault.
"""

from __future__ import annotations

import json
import os
import pathlib
import textwrap
import urllib.parse

import pytest

from tests.test_ui_smoke_playwright import direct_server_with_data as _direct_server_with_data

direct_server_with_data = _direct_server_with_data

SKILL = "widget_fault_probe"


def _write_fault_probe(data_dir: pathlib.Path) -> None:
    from ouroboros.skill_loader import SkillReviewState, compute_content_hash, save_review_state

    skill_dir = data_dir / "skills" / "external" / SKILL
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(textwrap.dedent(f"""\
        ---
        name: {SKILL}
        description: Isolated module widget fault reporting probe.
        version: 0.1.0
        type: extension
        entry: plugin.py
        permissions: ["widget"]
        ---
        # Widget fault probe
        """), encoding="utf-8")
    (skill_dir / "plugin.py").write_text(textwrap.dedent("""\
        def register(api):
            api.register_ui_tab("fault", "Fault", render={"kind": "module", "entry": "fault.js", "start": "auto"})
            api.register_ui_tab("burst", "Burst", render={"kind": "module", "entry": "burst.js", "start": "auto"})
            api.register_ui_tab("late", "Late", render={"kind": "module", "entry": "late.js", "start": "retain"})
            api.register_ui_tab("rejection", "Rejection", render={"kind": "module", "entry": "rejection.js", "start": "auto"})
            api.register_ui_tab("csp", "CSP", render={"kind": "module", "entry": "csp.js", "start": "auto"})
            api.register_ui_tab("isolation", "Isolation", render={"kind": "module", "entry": "isolation.js", "start": "auto"})
            api.register_ui_tab("unicode", "Unicode", render={"kind": "module", "entry": "unicode.js", "start": "auto"})
        """), encoding="utf-8")
    (skill_dir / "fault.js").write_text(
        "document.getElementById('root').textContent = 'fault probe';\n"
        "throw new Error('widget-fault-probe deliberate fault');\n", encoding="utf-8")
    (skill_dir / "burst.js").write_text(textwrap.dedent("""\
        // The widget's own code can post on its channel (the bridge's report hook): a per-mount cap still holds.
        const report = window.__ouroWidgetReport;
        for (let i = 0; i < 30; i += 1) {
            report({ type: 'ouro-widget-error', kind: 'error', message: `forged ${i}`,
                source: '', line: i, skill: 'telegram', tab_id: 'chat', revision: 'forged', where: 'message' });
        }
        document.getElementById('root').textContent = report ? 'burst sent' : 'no channel';
        report({ type: 'fault-probe-marker', name: 'burst sent' });
        """), encoding="utf-8")
    (skill_dir / "late.js").write_text(
        "document.getElementById('root').textContent = 'late armed';\n"
        "window.addEventListener('message', (event) => {\n"
        "  if (event.source === parent && event.data?.type === 'fault-probe-trigger')\n"
        "    throw new Error('widget-fault-probe late fault');\n"
        "});\n", encoding="utf-8")
    (skill_dir / "rejection.js").write_text(
        "Promise.reject(new Error('widget-fault-probe unhandled rejection'));\n", encoding="utf-8")
    (skill_dir / "unicode.js").write_text(
        "document.getElementById('root').textContent = 'clipped emoji';\n"
        "Promise.reject('x'.repeat(499) + '\\u{1F600}');\n", encoding="utf-8")
    (skill_dir / "csp.js").write_text(
        "const blocked = document.createElement('script');\n"
        "blocked.src = 'data:text/javascript,void%200';\n"
        "document.body.append(blocked);\n", encoding="utf-8")
    (skill_dir / "isolation.js").write_text(textwrap.dedent("""\
        const report = window.__ouroWidgetReport;
        document.getElementById('root').textContent = 'isolation armed';
        // A hostile frame can queue a fault behind its own stop acknowledgement. These are real channel messages,
        // already enqueued when the host removes this iframe, not a synthetic MessageEvent in the parent. The page
        // closes the channel when it handles the acknowledgement, so the marker that says the hook ran goes to the
        // window; the test then waits for the frame to be removed before it reads the fault log.
        window.__ouroWidgetOnDispose(() => {
            report({ type: 'ouro-widget-disposed' });
            report({ type: 'ouro-widget-error', kind: 'error', message: 'after disposal' });
            parent.postMessage({ type: 'fault-probe-marker', name: 'after disposal' }, '*');
        });
        """), encoding="utf-8")
    content_hash = compute_content_hash(skill_dir, manifest_entry="plugin.py")
    save_review_state(data_dir, SKILL, SkillReviewState(status="pass", content_hash=content_hash))


_FAULT_ROWS = """async (skill) => {
    const response = await fetch('/api/logs/events?limit=2000', {cache: 'no-store'});
    if (!response.ok) throw new Error(`events API: ${response.status}`);
    const body = await response.json();
    return (body.entries || []).filter((row) => row.type === 'widget_fault' && row.skill === skill);
}"""

# Observe the real socket without replacing its transport or synthesizing server frames. Marker messages are
# FIFO barriers for frame postMessages (and their host microtasks), not elapsed-time guesses.
_OBSERVE = """(() => {
    window.faultProbeFrames = [];
    window.faultProbeMarkers = [];
    window.faultProbeSocketOpen = false;
    const Socket = window.WebSocket;
    window.WebSocket = class extends Socket {
        constructor(...args) {
            super(...args);
            this.addEventListener('open', () => { window.faultProbeSocketOpen = true; });
            this.addEventListener('close', () => { window.faultProbeSocketOpen = false; });
            this.addEventListener('message', (event) => {
                try { window.faultProbeFrames.push(JSON.parse(event.data)); } catch {}
            });
        }
    };
    window.addEventListener('message', (event) => {
        if (event.data?.type === 'fault-probe-marker') window.faultProbeMarkers.push(event.data.name);
    });
    // A frame's bridge traffic arrives on its channel port: a marker there is FIFO with the frame's messages.
    const handler = Object.getOwnPropertyDescriptor(MessagePort.prototype, 'onmessage');
    Object.defineProperty(MessagePort.prototype, 'onmessage', {
        configurable: true,
        get() { return handler.get.call(this); },
        set(fn) {
            handler.set.call(this, fn && function (event) {
                const result = fn.call(this, event);
                if (event.data?.type === 'fault-probe-marker') window.faultProbeMarkers.push(event.data.name);
                return result;
            });
        },
    });
})();"""

_TOGGLE = """async (skill) => {
    const response = await fetch(`/api/skills/${encodeURIComponent(skill)}/toggle`, {
        method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({enabled: true}),
    });
    return response.status;
}"""


def _capture(page, name):
    target = os.environ.get("OUROBOROS_UI_EVIDENCE_OUT")
    if target:
        output = pathlib.Path(target)
        output.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(output / f"{name}.png"))
        (output / f"{name}.json").write_text(json.dumps(page.evaluate(_FAULT_ROWS, SKILL), indent=2), encoding="utf-8")


def _assert_log_only(page, skill, expected):
    page.wait_for_function("""({skill, expected}) => window.faultProbeFrames.filter(
        (frame) => frame.type === 'log' && frame.data?.type === 'widget_fault' && frame.data.skill === skill
    ).length >= expected""", arg={"skill": skill, "expected": expected}, timeout=30_000)
    facts = page.evaluate("""async (skill) => {
        const {classifyLiveFrame} = await import('/static/modules/notifications.js');
        const frames = window.faultProbeFrames.filter((frame) =>
            (frame.data?.type === 'widget_fault' && frame.data.skill === skill) || frame.type === 'widget_fault');
        return {frames, notifications: frames.map((frame) => classifyLiveFrame(frame.data,
            {kind: frame.type, isMain: true})), toasts: document.querySelectorAll('.toast').length,
            deliveries: window.faultProbeFrames.filter((frame) =>
                ['chat', 'card', 'quiz', 'notification'].includes(frame.type))};
    }""", skill)
    assert len(facts["frames"]) == expected, facts
    assert all(frame["type"] == "log" and frame["data"]["type"] == "widget_fault" for frame in facts["frames"]), facts
    assert all(not {"task_id", "parent_task_id", "root_task_id"} & frame["data"].keys()
               for frame in facts["frames"]), facts
    assert facts["notifications"] == [None] * expected, facts
    assert facts["toasts"] == 0, facts
    assert facts["deliveries"] == [], facts


@pytest.mark.ui_browser
@pytest.mark.parametrize("browser_name", ("chromium", "webkit"))
def test_a_running_widget_fault_reaches_the_events_log(direct_server_with_data, browser_name):
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    _write_fault_probe(direct_server_with_data["data_dir"])

    def card(tab_id: str) -> str:
        return f'[data-widget-key="{SKILL}:{tab_id}"]'

    try:
        with sync_playwright() as pw:
            browser = getattr(pw, browser_name).launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            page.add_init_script(_OBSERVE)
            reports = []
            page.on("request", lambda request: reports.append(json.loads(request.post_data or "{}"))
                    if urllib.parse.urlparse(request.url).path == "/api/widgets/fault" else None)
            try:
                # Enabling legitimately emits lifecycle chat rows. Finish that setup before opening the
                # observed page/socket: the assertions below concern faults, not the separate enable action.
                page.goto(f"{url}/api/health", wait_until="domcontentloaded")
                assert page.evaluate(_TOGGLE, SKILL) == 200
                page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                revision = page.evaluate("""async (skill) => {
                    const body = await (await fetch('/api/widgets', {cache: 'no-store'})).json();
                    return body.ui_tabs.find((tab) => tab.skill === skill && tab.tab_id === 'fault')?.revision || '';
                }""", SKILL)
                assert revision
                page.wait_for_function("() => window.faultProbeSocketOpen", timeout=30_000)
                chat_rows_before = page.locator('#page-chat .chat-bubble').count()
                page.click('[data-nav-page="widgets"]')
                page.frame_locator(f"{card('late')} iframe").locator("text=late armed").wait_for(timeout=30_000)
                # Auto widgets stop and clear their fault line when Widgets is left. Read the visible fault now.
                fault_status = page.locator(f"{card('fault')} [data-widget-status][data-widget-fault]")
                fault_status.wait_for(state="visible", timeout=30_000)
                shown = fault_status.get_attribute("data-widget-fault")
                for tab_id in ("rejection", "csp"):
                    page.locator(f"{card(tab_id)} [data-widget-status][data-widget-fault]").wait_for(
                        state="visible", timeout=30_000)
                page.wait_for_function("() => window.faultProbeMarkers.includes('burst sent')")

                isolated = page.frame_locator(f"{card('isolation')} iframe").locator("text=isolation armed")
                isolated.wait_for(timeout=30_000)
                nonce = page.locator(f"{card('isolation')} iframe").evaluate(
                    "frame => frame.srcdoc.match(/\"nonce\":\"([^\"]+)\"/)[1]")
                isolated.evaluate("""() => {
                    parent.postMessage({ type: 'ouro-widget-error', nonce: 'wrong', kind: 'error', message: 'wrong nonce' }, '*');
                    parent.postMessage({ type: 'fault-probe-marker', name: 'wrong nonce' }, '*');
                }""")
                # Correct nonce, wrong live frame: must not be attributed to the isolation mount.
                page.frame_locator(f"{card('late')} iframe").locator("#root").evaluate("""(root, nonce) => {
                    parent.postMessage({ type: 'ouro-widget-error', nonce, kind: 'error', message: 'wrong source' }, '*');
                    parent.postMessage({ type: 'fault-probe-marker', name: 'wrong source' }, '*');
                }""", nonce)
                page.wait_for_function("() => ['wrong nonce', 'wrong source'].every(x => window.faultProbeMarkers.includes(x))")
                assert page.locator(f"{card('isolation')} [data-widget-status][data-widget-fault]").count() == 0
                page.locator(f"{card('isolation')} [data-widget-power]").click()
                page.wait_for_function("() => window.faultProbeMarkers.includes('after disposal')")
                page.locator(f"{card('isolation')} iframe").wait_for(state="detached")
                fault_status.scroll_into_view_if_needed()
                _capture(page, f"widget-{browser_name}-board")
                # Leave Widgets before the Keep running frame throws: its card is retained, not shown.
                page.evaluate("""() => [...document.querySelectorAll('[data-nav-page="dashboard"]')]
                    .find((item) => getComputedStyle(item).display !== 'none')?.click()""")
                page.wait_for_function("""(selector) => {
                    const card = document.querySelector(selector);
                    return card && card.querySelector('iframe') && card.getClientRects().length === 0;
                }""", arg=card("late"), timeout=10_000)
                page.locator(f"{card('late')} iframe").evaluate(
                    "frame => frame.contentWindow.postMessage({type: 'fault-probe-trigger'}, '*')")
                # Poll the log itself: wait_for_function settles on an async predicate's first promise, true or not.
                for _attempt in range(60):
                    rows = page.evaluate(_FAULT_ROWS, SKILL)
                    if (any(row["tab_id"] == "late" for row in rows)
                            and len([row for row in rows if row["tab_id"] == "burst"]) >= 10
                            and all(any(row["tab_id"] == tab for row in rows) for tab in ("fault", "rejection", "csp", "unicode"))):
                        break
                    page.wait_for_timeout(500)
                _assert_log_only(page, SKILL, 15)
                assert not [row for row in rows if row["tab_id"] == "isolation"], rows
                assert not [report for report in reports if report.get("tab_id") == "isolation"], reports
                assert page.locator('#page-chat .chat-bubble').count() == chat_rows_before, 'faults never create chat rows'
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise

    assert shown.startswith("Widget script error: "), shown
    (fault,) = [row for row in rows if row["tab_id"] == "fault"]
    assert {key: fault[key] for key in ("revision", "where", "kind", "reported_by")} == {
        "revision": revision, "where": "board", "kind": "error", "reported_by": "page"}
    assert "task_id" not in fault
    # WebKit masks an opaque frame's error text as "Script error."; the mounted identity names it anyway.
    assert "widget-fault-probe deliberate fault" in fault["message"] or browser_name == "webkit", fault
    burst = [row for row in rows if row["tab_id"] == "burst"]
    assert sorted(row["message"] for row in burst) == sorted(f"forged {i}" for i in range(10))
    assert all(row["skill"] == SKILL and row["where"] == "board" for row in burst), "never the frame's identity"
    assert len([report for report in reports if report.get("tab_id") == "burst"]) == 10, "the mount's own cap"
    late = [row for row in rows if row["tab_id"] == "late"]
    assert len(late) == 1 and late[0]["where"] == "board"
    (unicode_fault,) = [row for row in rows if row["tab_id"] == "unicode"]
    assert unicode_fault["message"] == "x" * 499 + "\ufffd", unicode_fault
    assert unicode_fault["kind"] == "rejection" and unicode_fault["revision"] == revision
    for tab_id, kind in (("rejection", "rejection"), ("csp", "csp")):
        (row,) = [row for row in rows if row["tab_id"] == tab_id]
        assert {key: row[key] for key in ("skill", "revision", "where", "kind", "reported_by")} == {
            "skill": SKILL, "revision": revision, "where": "board", "kind": kind, "reported_by": "page"}
        assert "task_id" not in row
        # Opaque WebKit frames can mask detail; identity and kind are never relaxed.
        if browser_name != "webkit":
            assert ("unhandled rejection" if kind == "rejection" else "script-src") in row["message"], row


@pytest.mark.ui_browser
@pytest.mark.parametrize("browser_name", ("chromium", "webkit"))
def test_inline_module_fault_keeps_message_ownership(direct_server_with_data, browser_name):
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import sync_playwright

    from ouroboros import dialogue_store
    from ouroboros.card_store import write_card

    server = direct_server_with_data
    _write_fault_probe(server["data_dir"])
    # The same durable card/history contract used by test_living_cards_browser, in this fixture's data only.
    record = write_card(server["data_dir"], chat_id=1, task_id="", title="Inline fault probe",
                        fallback_text="An isolated running module", kind="widget",
                        body={"widget": {"skill": SKILL, "tab_id": "late"}})
    dialogue_store.append(server["data_dir"], "chat", {
        "ts": "2026-10-21T10:00:00+00:00", "direction": "out", "chat_id": 1, "user_id": 0, "task_id": "",
        "text": record["fallback_text"], "type": "card", "card": {
            "card_id": record["card_id"], "kind": "widget", "title": record["title"], "revision": 1},
    })
    with sync_playwright() as pw:
        browser = getattr(pw, browser_name).launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        page.add_init_script(_OBSERVE)
        try:
            # Enable before loading the chat so its first widget list is live, with no reload/start-hold race.
            page.goto(f"{server['url']}/api/health", wait_until="domcontentloaded")
            assert page.evaluate(_TOGGLE, SKILL) == 200
            page.goto(server["url"], wait_until="domcontentloaded", timeout=30_000)
            card = page.locator(f'.chat-card[data-card-id="{record["card_id"]}"]')
            card.scroll_into_view_if_needed(timeout=30_000)
            frame = page.frame_locator(f'.chat-card[data-card-id="{record["card_id"]}"] iframe')
            frame.locator("text=late armed").wait_for(timeout=30_000)
            chat_rows_before = page.locator('#page-chat .chat-bubble').count()
            assert card.count() == 1, 'one seeded card; setup may also leave a lifecycle message'
            revision = page.evaluate("""async (skill) => {
                const body = await (await fetch('/api/widgets', {cache: 'no-store'})).json();
                return body.ui_tabs.find((tab) => tab.skill === skill && tab.tab_id === 'late')?.revision;
            }""", SKILL)
            assert revision
            page.wait_for_function("() => window.faultProbeSocketOpen", timeout=30_000)
            with page.expect_response(lambda response: response.request.method == "POST" and
                                      urllib.parse.urlparse(response.url).path == "/api/widgets/fault") as written:
                card.locator("iframe").evaluate(
                    "frame => frame.contentWindow.postMessage({type: 'fault-probe-trigger'}, '*')")
            assert written.value.status == 200 and written.value.json() == {"ok": True, "logged": True}
            card.locator('.chat-card-status', has_text='Widget script error:').wait_for(timeout=30_000)
            rows = page.evaluate(_FAULT_ROWS, SKILL)
            assert len(rows) == 1, (rows, (server["data_dir"] / "logs/events.jsonl").read_text(encoding="utf-8"))
            (fault,) = rows
            assert {key: fault[key] for key in ("skill", "tab_id", "revision", "where", "kind", "reported_by")} == {
                "skill": SKILL, "tab_id": "late", "revision": revision, "where": "message", "kind": "error",
                "reported_by": "page"}
            assert "task_id" not in fault
            if browser_name != "webkit":
                assert "widget-fault-probe late fault" in fault["message"], fault
            _assert_log_only(page, SKILL, 1)
            assert page.locator('#page-chat .chat-bubble').count() == chat_rows_before
            assert card.locator('iframe').count() == 1, 'the fault stays with its running inline module'
            assert page.locator(f'[data-widget-key="{SKILL}:late"]').count() == 0, 'no board card owns this mount'
            _capture(page, f"widget-{browser_name}-inline")
        finally:
            browser.close()
