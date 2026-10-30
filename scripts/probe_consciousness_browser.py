"""Focused offline Chromium/WebKit verification of the real Dashboard module.

Run through safe_test with PLAYWRIGHT_BROWSERS_PATH and an evidence directory.
No application server, owner data or external network is used.
"""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading

from playwright.sync_api import sync_playwright

REPO = Path(__file__).resolve().parents[1]


class QuietHandler(SimpleHTTPRequestHandler):
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map, ".json": "application/json; charset=utf-8"}
    def log_message(self, *_):
        pass


def probe(output):
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(QuietHandler, directory=str(REPO)))
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    origin = f"http://127.0.0.1:{server.server_port}"
    results = []
    try:
        with sync_playwright() as playwright:
            for engine in (playwright.chromium, playwright.webkit):
                browser = engine.launch()
                try:
                    for width, height, telegram in [(1280, 900, False), (390, 844, False), (390, 780, True), (800, 780, True)]:
                        context = browser.new_context(viewport={"width": width, "height": height}, has_touch=width < 980,
                                                      color_scheme="light", timezone_id="UTC")
                        page = context.new_page()
                        errors = []
                        page.on("pageerror", lambda error: errors.append(str(error)))
                        page.route("**/*", lambda route: route.continue_() if route.request.url.startswith(origin) else route.abort())
                        page.goto(origin + "/web/tests/fixtures/consciousness_contract.json")
                        page.evaluate("""({telegram, width}) => {
                            document.documentElement.dataset.theme = 'light';
                            document.documentElement.dataset.layout = width < 980 ? 'stack' : 'docked';
                            if (telegram) { document.documentElement.dataset.host = 'telegram'; window.Telegram = { WebApp: { platform: 'ios' } }; }
                            document.head.innerHTML = ['ui', 'style', 'settings', 'cards', 'consciousness'].map(file =>
                                `<link rel="stylesheet" href="/web/${file}.css">`).join('') +
                                '<style>body{margin:0;display:block}#main-mount{height:100dvh}#main-mount[hidden],#mount[hidden]{display:none}#mount{height:100dvh;padding:16px;box-sizing:border-box}.costs-scroll{height:calc(100dvh - 32px);overflow:auto}</style>';
                            document.body.innerHTML = '<main id="mount" class="dashboard-shell"></main><main id="main-mount" hidden></main>';
                        }""", {"telegram": telegram, "width": width})
                        page.add_script_tag(url=origin + "/web/marked.min.js")
                        page.add_script_tag(url=origin + "/web/purify.min.js")
                        page.evaluate("""async () => {
                            const api = await import('/web/modules/consciousness_api.js');
                            const dashboard = await import('/web/modules/consciousness_dashboard.js');
                            window.fixture = await (await fetch('/web/tests/fixtures/consciousness_contract.json')).json();
                            window.snapshot = api.normalizeConsciousness(fixture.consciousness);
                            window.placesReads = 0; window.wakeReads = 0;
                            window.dashboardState = { activePage: 'chat', dashboardActiveSubtab: 'consciousness' };
                            window.controller = dashboard.initConsciousnessDashboard({ mount: document.querySelector('#mount'),
                                state: dashboardState, read: async () => { wakeReads++; return snapshot; },
                                readPlacesFn: async () => { placesReads++; return api.normalizePlacesResponse(fixture.places); } });
                            window.doorEvents = [];
                            window.addEventListener('ouro:open-task', event => doorEvents.push(event.detail));
                            await controller.refresh();
                            const { apiClient } = await import('/web/modules/api_client.js');
                            const { createChatInstance } = await import('/web/modules/chat.js');
                            const { tr } = await import('/web/modules/i18n.js');
                            window.historyReads = []; window.doorNotices = [];
                            apiClient.chatHistory = async ({chatId, around}) => {
                                historyReads.push({chatId, around});
                                const task = around?.startsWith('task:') ? around.slice(5) : '';
                                const history = structuredClone(fixture.history);
                                if (!task) return { ...history, messages: [] };
                                const row = history.messages.find(row => row.task_id === task);
                                return { ...history, anchor: row ? { state: 'found', task_id: task, history_id: row.history_id } : { state: 'absent' } };
                            };
                            const ws = { on() { return () => {}; }, isConnected: () => true, send() {} };
                            const chatState = { activePage: 'dashboard', projectChatIds: new Set(), unreadCount: 0 };
                            const stateSnapshots = { begin: () => ({}), gate: async () => ({}), isCurrent: () => true, apply() {} };
                            window.mainChat = createChatInstance({ ws, state: chatState, stateSnapshots,
                                mountEl: document.querySelector('#main-mount'), chatId: 1, asPanel: true });
                            const app = await (await fetch('/web/app.js')).text();
                            const door = app.slice(app.indexOf('async function revealQuotedSource('), app.indexOf('\\nfunction freshProjectRow'));
                            const showPage = async () => {
                                dashboardState.activePage = 'chat'; chatState.activePage = 'chat';
                                document.querySelector('#mount').hidden = true;
                                document.querySelector('#main-mount').hidden = false;
                                return true;
                            };
                            new Function('apiClient', 'mainChat', 'state', 'showPage', 'closeProjectPanel', 'showToast', 'tr',
                                'let projectNavigationGeneration = 0;\\n' + door)(
                                apiClient, mainChat, chatState, showPage, () => {}, text => doorNotices.push(text), tr);
                        }""")
                        assert page.evaluate("document.characterSet") == "UTF-8"
                        summary = page.locator('[data-wake="w-summary"] .consciousness-wake-summary').evaluate("""el => ({
                            whole: el.textContent === snapshot.recentWakes.find(row => row.taskId === 'w-summary').summary,
                            closed: !el.closest('[data-wake]').querySelector('details').open,
                            width: el.clientWidth, height: el.clientHeight, scrollWidth: el.scrollWidth, scrollHeight: el.scrollHeight,
                            whiteSpace: getComputedStyle(el).whiteSpace, lines: el.clientHeight / parseFloat(getComputedStyle(el).lineHeight)
                        })""")
                        assert summary['whole'] and summary['closed'] and summary['whiteSpace'] != 'nowrap', summary
                        assert summary['scrollWidth'] <= summary['width'] + 1 and summary['scrollHeight'] <= summary['height'] + 1, summary
                        if width == 390:
                            assert summary['lines'] >= 2, summary
                        if engine.name == 'chromium' and width == 1280:
                            rows = page.locator('[data-cx-wakes] [data-wake]').evaluate_all(
                                '(rows) => Object.fromEntries(rows.map(row => [row.dataset.wake, row.outerHTML]))')
                            (output / 'fixture-rows.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
                            (output / 'fixture-rows.html').write_text(page.locator('[data-cx-wakes]').inner_html(), encoding='utf-8')
                        page.locator('[data-cx-wakes]').scroll_into_view_if_needed()
                        page.screenshot(path=str(output / f"collapsed-{engine.name}-{width}-{'telegram' if telegram else 'browser'}.png"))
                        page.locator('[data-wake="w-summary"]').scroll_into_view_if_needed()
                        page.screenshot(path=str(output / f"summary-{engine.name}-{width}-{'telegram' if telegram else 'browser'}.png"))
                        page.clock.install()
                        page.locator('[data-wake="w-summary"] summary').click()
                        # The note mounts on the toggle event, which the browser queues after `open` flips: wait for the mount.
                        page.wait_for_selector('[data-wake="w-summary"] .consciousness-wake-note p')
                        page.evaluate("""() => {
                            const row = document.querySelector('[data-wake="w-summary"]');
                            const note = row.querySelector('.consciousness-wake-note');
                            const focus = row.querySelector('summary'); focus.focus();
                            const text = note.querySelector('p').firstChild;
                            const range = document.createRange(); range.setStart(text, 3); range.setEnd(text, 30);
                            getSelection().removeAllRanges(); getSelection().addRange(range);
                            window.kept = { row, note, focus, selection: getSelection().toString(), top: note.getBoundingClientRect().top };
                        }""")
                        page.evaluate("""async () => {
                            snapshot.recentWakes = [{ ...snapshot.recentWakes.find(row => row.taskId === 'w-running'), taskId: 'new-wake' }, ...snapshot.recentWakes];
                            snapshot.recentWakes.sort((a, b) => (b.taskId === 'w-summary') - (a.taskId === 'w-summary'));
                            // Both changed siblings and changed content on the open row must keep the reader's subtree.
                            snapshot.recentWakes.find(row => row.taskId === 'w-summary').note += '\\nNew detail on refresh.';
                            dashboardState.activePage = 'dashboard';
                            await controller.refresh();
                        }""")
                        page.clock.fast_forward(20000)
                        page.wait_for_function("wakeReads === 3")
                        retained = page.evaluate("""() => ({ open: kept.row.querySelector('details').open,
                            row: kept.row === document.querySelector('[data-wake="w-summary"]'),
                            note: kept.note === kept.row.querySelector('.consciousness-wake-note'),
                            focus: document.activeElement === kept.focus, selection: getSelection().toString() === kept.selection,
                            shift: Math.abs(kept.note.getBoundingClientRect().top - kept.top),
                            overflow: document.documentElement.scrollWidth > innerWidth, placesReads, wakeReads })""")
                        assert all(retained[key] for key in ("open", "row", "note", "focus", "selection")), retained
                        assert retained["shift"] < 2 and not retained["overflow"] and retained["placesReads"] == 1, retained
                        page.evaluate("window.latestWakes = snapshot.recentWakes")
                        for unavailable in (None, []):
                            page.evaluate("async wakes => { snapshot.recentWakes = wakes; await controller.refresh(); }", unavailable)
                            assert page.evaluate("kept.row.isConnected && kept.row.querySelector('details').open && document.activeElement === kept.focus && getSelection().toString() === kept.selection && Math.abs(kept.note.getBoundingClientRect().top - kept.top) < 2")
                        page.evaluate("async () => { snapshot.recentWakes = latestWakes; await controller.refresh(); }")
                        page.locator('[data-wake="w-summary"] summary').scroll_into_view_if_needed()
                        page.screenshot(path=str(output / f"wakes-{engine.name}-{width}-{'telegram' if telegram else 'browser'}.png"))
                        page.locator('[data-wake="w-summary"] summary').click()
                        page.wait_for_function("document.querySelector('[data-wake=\"w-summary\"] .consciousness-wake-note').textContent.includes('New detail on refresh.')")
                        page.locator('[data-wake="214a31b6"] summary').click()
                        page.locator('[data-wake="214a31b6"] [data-open-task]').click()
                        landed_card = page.locator('#main-mount .chat-live-card[data-task-id="214a31b6"].is-quote-target')
                        landed_card.wait_for(state='visible')
                        landing = landed_card.evaluate("""el => {
                            const card = el.getBoundingClientRect(), feed = document.querySelector('#chat-messages').getBoundingClientRect();
                            return { focused: document.activeElement === el, top: card.top, bottom: card.bottom, feedTop: feed.top,
                                feedBottom: feed.bottom, height: innerHeight, reads: historyReads.filter(row => row.around === 'task:214a31b6').length };
                        }""")
                        assert landing['focused'] and landing['reads'] == 1, landing
                        assert landing['top'] >= landing['feedTop'] - 2 and landing['top'] < min(landing['height'], landing['feedBottom']), landing
                        assert page.evaluate("doorNotices") == []
                        assert page.evaluate("doorEvents") == [{"task_id": "214a31b6", "chat_id": 1}]
                        page.screenshot(path=str(output / f"main-{engine.name}-{width}-{'telegram' if telegram else 'browser'}.png"))
                        page.evaluate("""() => {
                            document.querySelector('#main-mount').hidden = true;
                            document.querySelector('#mount').hidden = false;
                        }""")
                        # Closing a row is registered on the queued toggle event (the dashboard's capture listener runs before
                        # this target listener); refreshing before it would still find the row open and keep the list.
                        page.evaluate("""() => {
                            const more = document.querySelector('[data-wake="214a31b6"] details');
                            window.closedToggle = new Promise((resolve) => more.addEventListener('toggle', () => resolve(more.open), { once: true }));
                        }""")
                        page.locator('[data-wake="214a31b6"] summary').click()
                        assert page.evaluate("closedToggle") is False
                        page.evaluate("""async () => {
                            snapshot.recentWakes = null;
                            snapshot.coverage = 'Recent wakes could not be read: storage is busy.';
                            await controller.refresh();
                        }""")
                        assert page.locator('[data-cx-wakes] [data-wake]').count() > 0
                        warning = page.locator('[data-cx-wakes] [role="status"][data-tone="warn"][data-i18n-authored]')
                        assert warning.count() == 1 and warning.inner_text() == 'Recent wakes could not be read: storage is busy.'
                        for unavailable, sentences in [(None, ['This server does not report its recent wakes yet.']),
                                                       ([], ['No own wakes with a recorded model call in the last 48 h.',
                                                             'No wakes in places in the last 48 h.'])]:
                            page.evaluate("async wakes => { snapshot.recentWakes = wakes; snapshot.coverage = ''; await controller.refresh(); }", unavailable)
                            for words in sentences:
                                assert words in page.locator('[data-cx-wakes]').inner_text()
                        page.evaluate("""async () => {
                            snapshot.recentWakes = [{ ...latestWakes.find(row => row.taskId === '214a31b6'), note: '~~~js\\nanswer();\\n~~~' }];
                            await controller.refresh();
                        }""")
                        page.locator('[data-wake="214a31b6"] summary').click()
                        page.locator('[data-chat-markdown-enhanced] [data-code-copy]').wait_for(state='visible')
                        page.locator('[data-wake="214a31b6"] summary').click()
                        page.wait_for_function("!document.querySelector('[data-chat-markdown-enhanced]')")
                        assert not errors, errors
                        results.append({"engine": engine.name, "width": width, "height": height, "telegram": telegram, **retained, "summary": summary, "landing": landing})
                        page.evaluate("controller.destroy(); mainChat.destroy()")
                        context.close()
                finally:
                    browser.close()
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)
    (output / "browser-results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"{len(results)} browser layouts passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    probe(args.output)
