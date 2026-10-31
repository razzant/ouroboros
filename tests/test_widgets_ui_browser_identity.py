"""Production Widgets renderer over isolated browser REST/WS fixtures.

These tests never start or import the runtime. Every request is fulfilled from
the checked-out web source or a synthetic response, including framed content.
The ordinary real-extension browser suites separately cover host integration.
"""

from __future__ import annotations

import ast
import contextlib
import json
from pathlib import Path
from urllib.parse import urlsplit

import pytest


pytestmark = pytest.mark.serial

WEB = Path(__file__).resolve().parents[1] / "web"
ORIGIN = "http://widgets.test"
_BOOT = """
import {initWidgets} from '/modules/widgets.js';
const handlers = new Map();
window.fixtureWs = {
    on(type, fn) { const rows = handlers.get(type) || []; rows.push(fn); handlers.set(type, rows); },
    emit(type, message) { for (const fn of handlers.get(type) || []) fn(message); },
};
window.fixtureSources = [];
window.EventSource = class {
    constructor(url) { this.url = url; this.closed = false; fixtureSources.push(this); }
    close() { this.closed = true; }
};
initWidgets({ws: fixtureWs});
document.getElementById('page-widgets').classList.add('active');
window.showWidgets = (page = 'widgets') => window.dispatchEvent(
    new CustomEvent('ouro:page-shown', {detail: {page}}),
);
showWidgets();
"""


def _declaration():
    return {
        "skill": "identity_probe", "tab_id": "main", "ws_prefix": "ext:identity_probe:",
        "render": {"kind": "declarative", "schema_version": 1, "components": [
            {"type": "group", "id": "group", "components": [
                {"type": "callout", "condition_key": "show", "target": "live", "text": "New adjacent fact"},
                {"type": "tabs", "id": "tabs", "tabs": [
                    {"label": "Edit", "components": [
                        {"type": "form", "id": "edit", "route": "submit", "method": "POST", "fields": [
                            {"name": "query", "label": "Query"},
                            {"name": "notes", "label": "Notes", "type": "textarea"},
                            {"name": "secret", "label": "Secret", "type": "password"},
                            {"name": "mode", "label": "Mode", "type": "select", "options": ["a", "b"]},
                            {"name": "enabled", "label": "Enabled", "type": "checkbox"},
                        ]},
                    ]},
                    {"label": "Other", "components": [{"type": "markdown", "text": "Other tab"}]},
                ]},
                {"type": "action", "id": "action", "label": "Other action", "route": "action", "method": "POST"},
                {"type": "action", "id": "job", "label": "Start job", "route": "start", "method": "POST",
                 "job": True, "status_route": "status", "interval_ms": 1000},
                {"type": "chart", "id": "chart", "target": "live", "path": "chart"},
                {"type": "json", "id": "json", "target": "live"},
                {"type": "audio", "id": "audio", "route": "audio", "label": "Audio"},
                {"type": "file", "id": "file", "route": "file", "label": "Download file"},
                {"type": "kanban", "id": "kanban", "columns": [{"id": "todo", "label": "Todo"}, {"id": "done", "label": "Done"}],
                 "cards": [{"id": "task", "label": "Move me", "column": "todo"}], "on_move": {"route": "move"}},
                {"type": "stream", "id": "stream", "route": "events", "target": "stream"},
                {"type": "subscription", "id": "subscription", "event": "tick", "target": "live",
                 "render": [{"type": "metric", "label": "Count", "path": "count"}]},
            ]},
        ]},
    }


@contextlib.contextmanager
def _page(browser_name, tabs, ledger=None):
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import sync_playwright

    state = {"tabs": tabs, "list_error": False, "requests": [], "pending": [], "hold": "", "status_calls": 0,
             "preferences": {"widget_order": [], "widget_start_mode": {}}}
    with sync_playwright() as pw:
        browser = getattr(pw, browser_name).launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        page.set_default_timeout(10_000)
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))

        def route_request(route):
            url = urlsplit(route.request.url)
            if f"{url.scheme}://{url.netloc}" != ORIGIN:
                route.abort()
                return

            def reply(data, status=200):
                route.fulfill(status=status, content_type="application/json", body=json.dumps(data))

            if url.path == "/":
                route.fulfill(content_type="text/html", body=(
                    '<!doctype html><html><head><link rel="stylesheet" href="/ui.css"><link rel="stylesheet" href="/style.css">'
                    '<script src="/chart.umd.min.js"></script></head><body><div id="content"></div>'
                    f'<script type="module">{_BOOT}</script></body></html>'
                ))
            elif url.path == "/api/widgets":
                state["requests"].append(url.path)
                reply({"error": "List temporarily unavailable"}, 503) if state["list_error"] else reply({"ui_tabs": state["tabs"]})
            elif url.path == "/api/ui/preferences":
                if route.request.method == "POST":
                    state["preferences"].update(route.request.post_data_json)
                reply(state["preferences"])
            elif url.path.startswith("/api/widgets/asset-leases"):
                # A framed mount's route lease (mint, renew, revoke): the frame then loads under its URL.
                lease = "L" * 43
                reply({"ok": True} if url.path.endswith("/revoke") else
                      {"token": lease, "base_url": f"/api/widget-resources/{lease}/", "expires_in": 120, "renew_after": 40})
            elif url.path.startswith(("/api/extensions/", "/api/widget-resources/")):
                state["requests"].append(url.path + (f"?{url.query}" if url.query else ""))
                if state["hold"] and url.path.endswith(state["hold"]):
                    state["pending"].append(route)
                elif url.path.endswith("/frame"):
                    route.fulfill(content_type="text/html", body=(
                        '<button id="increment">Increment</button><output id="count">0</output>'
                        '<script>increment.onclick=()=>count.value=Number(count.value)+1</script>'
                    ))
                elif url.path.endswith("/submit"):
                    reply({"error": "Form request failed"}, 503)
                elif url.path.endswith("/action"):
                    reply({"message": "Other action completed"})
                elif url.path.endswith("/start"):
                    reply({"job_id": "same-job"})
                elif url.path.endswith("/status"):
                    state["status_calls"] += 1
                    if state["status_calls"] == 1:
                        reply({"error": "Temporary transport failure"}, 503)
                    else:
                        reply({"status": "done", "result": {"message": "Job completed"}})
                elif url.path.endswith("/audio"):
                    route.fulfill(content_type="audio/wav", body=b"")
                else:
                    reply({})
            else:
                file = (WEB / url.path.lstrip("/")).resolve()
                if not file.is_relative_to(WEB) or not file.is_file():
                    route.fulfill(status=404, body="Not found")
                    return
                mime = "text/javascript" if file.suffix == ".js" else "text/css" if file.suffix == ".css" else "application/octet-stream"
                route.fulfill(content_type=mime, body=file.read_bytes())

        page.route("**/*", route_request)
        if ledger:  # what an earlier page left in this browser's start ledger (web/modules/holds.js)
            stored = json.dumps(json.dumps({"v": 1, "entries": ledger}))
            page.add_init_script(f"if (window === window.top) localStorage.setItem('ouroboros.holds', {stored});")
        try:
            page.goto(ORIGIN, wait_until="domcontentloaded")
            yield page, state
            assert errors == []
        finally:
            for pending in state["pending"]:
                pending.abort()
            browser.close()


def _tick(page, count, show=True):
    page.evaluate("""({count, show}) => fixtureWs.emit('message', {
        type: 'ext:identity_probe:tick',
        data: {count, show, chart: {labels: ['A'], datasets: [{label: 'Count', data: [count]}]}},
    })""", {"count": count, "show": show})


@pytest.mark.ui_browser
@pytest.mark.parametrize("browser_name", ("chromium", "webkit"))
def test_widget_data_ticks_preserve_actual_controls_charts_and_resources(browser_name):
    with _page(browser_name, [_declaration()]) as (page, _state):
        page.locator('[data-widget-form="id:edit"]').wait_for()
        page.evaluate(r"""() => {
            const form = document.querySelector('[data-widget-form]');
            const input = form.elements.query;
            input.value = 'Draft'; form.elements.notes.value = 'Multiline\nnotes';
            form.elements.secret.value = 'synthetic-password'; form.elements.mode.value = 'b';
            form.elements.enabled.checked = true; input.focus(); input.setSelectionRange(2, 4);
            document.querySelector('.widget-json').open = true;
            const canvas = document.querySelector('[data-widget-chart-key]');
            window.savedNodes = {input, form, canvas, wrap: canvas.parentElement, chart: Chart.getChart(canvas),
                audio: document.querySelector('audio'), stream: fixtureSources[0]};
            input.dispatchEvent(new CompositionEvent('compositionstart', {bubbles: true, data: '輸'}));
        }""")
        for count in range(1, 5):
            _tick(page, count, show=count % 2 == 1)
        page.evaluate("fixtureSources[0].onmessage({data: JSON.stringify({text: 'stream tick'})})")
        facts = page.evaluate("""() => {
            const form = document.querySelector('[data-widget-form]'); const input = form.elements.query;
            const canvas = document.querySelector('[data-widget-chart-key]');
            return {sameInput: input === savedNodes.input, sameForm: form === savedNodes.form,
                focused: input === document.activeElement, selection: [input.selectionStart, input.selectionEnd],
                value: input.value, notes: form.elements.notes.value, secret: form.elements.secret.value,
                selected: form.elements.mode.value, checked: form.elements.enabled.checked,
                jsonOpen: document.querySelector('.widget-json').open,
                sameCanvas: canvas === savedNodes.canvas, sameWrapper: canvas.parentElement === savedNodes.wrap,
                sameChart: Chart.getChart(canvas) === savedNodes.chart, chartValue: savedNodes.chart.data.datasets[0].data[0],
                sameAudio: document.querySelector('audio') === savedNodes.audio, streams: fixtureSources.length};
        }""")
        assert facts == {
            "sameInput": True, "sameForm": True, "focused": True, "selection": [2, 4], "value": "Draft",
            "notes": "Multiline\nnotes", "secret": "synthetic-password", "selected": "b", "checked": True,
            "jsonOpen": True, "sameCanvas": True, "sameWrapper": True, "sameChart": True, "chartValue": 4,
            "sameAudio": True, "streams": 1,
        }
        page.evaluate("savedNodes.input.dispatchEvent(new CompositionEvent('compositionend', {bubbles: true, data: '輸'}))")
        page.keyboard.type("XYZ")
        assert page.locator('input[name="query"]').input_value() == "DrXYZt"
        page.evaluate("showWidgets('chat')")
        assert page.evaluate("savedNodes.stream.closed && Chart.getChart(savedNodes.canvas) === undefined")
        page.evaluate("showWidgets()")
        page.wait_for_function("fixtureSources.length === 2")
        assert page.locator('input[name="query"]').input_value() == "DrXYZt"
        assert page.locator('input[name="secret"]').input_value() == ""


@pytest.mark.ui_browser
@pytest.mark.parametrize("browser_name", ("chromium", "webkit"))
def test_widget_action_feedback_has_component_ownership_and_one_submission(browser_name):
    declaration = _declaration()
    declaration["render"]["components"][0]["components"][-1]["target"] = "result"
    with _page(browser_name, [declaration]) as (page, state):
        form = page.locator('[data-widget-form="id:edit"]')
        form.wait_for()
        for count in range(10):
            _tick(page, count)
        state["hold"] = "/submit"
        form.locator('input[name="query"]').fill("Keep this request")
        form.locator('button[type="submit"]').click()
        page.wait_for_function("document.querySelector('[data-widget-form]').getAttribute('aria-busy') === 'true'")
        assert page.locator('[data-widget-action="id:action"]').is_enabled()
        _tick(page, 20)
        assert form.locator('button[type="submit"]').is_disabled()
        assert len(state["pending"]) == 1
        state["pending"].pop().fulfill(status=503, content_type="application/json", body=json.dumps({"error": "Form request failed"}))
        state["hold"] = ""
        page.locator('[data-widget-feedback="id:edit"][data-state="error"]').wait_for()
        assert form.locator('[data-widget-feedback]').inner_text() == "Form request failed"
        assert form.locator('input[name="query"]').input_value() == "Keep this request"
        assert page.locator('[data-widget-feedback="id:action"]').count() == 0
        assert page.locator('[data-widget-feedback="id:kanban"]').count() == 0
        assert page.locator('[data-widget-kanban-move]').is_enabled()
        assert page.get_by_role('combobox', name='Move to column for Move me').evaluate("select => select.classList.contains('ui-control')")
        page.locator('[data-widget-action="id:action"]').click()
        page.locator('[data-widget-feedback="id:action"][data-state="success"]').wait_for()
        assert form.locator('[data-widget-feedback]').inner_text() == "Form request failed"
        assert len([url for url in state["requests"] if url.endswith("/submit")]) == 1
        assert len([url for url in state["requests"] if url.endswith("/action")]) == 1
        with page.expect_response("**/status?job_id=same-job") as first_status:
            page.locator('[data-widget-action="id:job"]').click()
        assert first_status.value.status == 503
        page.locator('[data-widget-feedback="id:job"][data-state="loading"]').wait_for()
        _tick(page, 30)
        assert page.locator('[data-widget-action="id:job"]').is_disabled()
        assert page.locator('[data-widget-action="id:action"]').is_enabled()
        page.evaluate("showWidgets('chat')")
        page.evaluate("showWidgets()")
        page.locator('[data-widget-feedback="id:job"][data-state="success"]').wait_for()
        status_urls = [url for url in state["requests"] if "/status?" in url]
        assert len(status_urls) == 2
        assert all("job_id=same-job" in url for url in status_urls)
        assert form.locator('[data-widget-feedback]').inner_text() == "Form request failed"


@pytest.mark.ui_browser
@pytest.mark.parametrize("browser_name", ("chromium", "webkit"))
def test_widget_download_pending_and_interrupted_action_feedback(browser_name):
    declaration = _declaration()
    declaration["render"]["components"].append({"type": "status", "id": "operation-status", "loading": "Declared working"})
    with _page(browser_name, [declaration]) as (page, state):
        page.locator('[data-widget-form]').wait_for()
        page.evaluate("""() => {
            window.downloadCalls = 0;
            window.pywebview = {api: {download_file_to_downloads() {
                downloadCalls += 1;
                return new Promise(resolve => { window.finishDownload = resolve; });
            }}};
        }""")
        download = page.locator('[data-widget-download-url]')
        download.click()
        _tick(page, 10)
        assert download.is_disabled()
        page.evaluate("document.querySelector('[data-widget-download-url]').click()")
        assert page.evaluate("downloadCalls") == 1
        page.evaluate("finishDownload({ok: true})")
        page.wait_for_function("!document.querySelector('[data-widget-download-url]').disabled")
        assert page.locator('[data-widget-feedback="id:file"]').inner_text() == "Download requested."
        state["hold"] = "/submit"
        page.locator('[data-widget-form] button[type="submit"]').click()
        page.locator('[data-widget-feedback="id:edit"][data-state="loading"]').wait_for()
        page.evaluate("showWidgets('chat')")
        page.evaluate("showWidgets()")
        page.wait_for_function("fixtureSources.length === 2")
        assert page.locator('[data-widget-feedback="id:edit"]').inner_text() == "No result received before this widget was closed."
        assert page.locator('[data-widget-feedback="id:edit"]').get_attribute("data-state") == "error"
        assert page.locator('[data-widget-form] button[type="submit"]').is_enabled()
        assert page.locator('[data-widget-component="id:operation-status"]').get_attribute("data-state") == "error"


@pytest.mark.ui_browser
@pytest.mark.parametrize("browser_name", ("chromium", "webkit"))
def test_all_declarative_component_roots_keep_existing_tree_identity(browser_name):
    samples = {
        "action": {"route": "action"}, "audio": {"route": "audio"},
        "chart": {"labels": ["A"], "datasets": [{"label": "One", "data": [1]}]},
        "code": {"text": "const n = 1;"}, "file": {"route": "file"},
        "form": {"route": "submit", "fields": [{"name": "query", "label": "Query"}]},
        "gallery": {"items": [{"type": "image", "route": "image"}]}, "image": {"route": "image"},
        "json": {}, "kv": {"fields": [{"label": "Count", "path": "count"}]},
        "key_value": {"path": "pairs"}, "markdown": {"text": "Text"},
        "poll": {"route": "poll"}, "progress": {"path": "progress"}, "status": {},
        "stream": {"route": "events", "target": "stream"},
        "subscription": {"event": "tick", "target": "result", "render": [{"type": "metric", "label": "Subscribed value", "value": 1}]},
        "tabs": {"tabs": [{"label": "One", "components": [{"type": "callout", "text": "Tab"}]}]},
        "table": {"path": "rows", "columns": [{"label": "Count", "path": "count"}]},
        "video": {"route": "video"}, "map": {"markers": [{"lat": 1, "lon": 2, "label": "Place"}]},
        "calendar": {"items": [{"label": "Event", "start": "2026-09-09"}]},
        "kanban": {"columns": [{"id": "todo", "label": "Todo"}], "cards": [{"id": "one", "label": "One", "column": "todo"}]},
        "group": {"components": [{"type": "callout", "text": "Group"}]},
        "metric": {"label": "Value", "value": 42}, "callout": {"text": "Ready"},
    }
    # Read the validator's literal without importing runtime roots or settings.
    tree = ast.parse((WEB.parent / "ouroboros" / "extension_ui_validation.py").read_text())
    supported = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id == "_DECLARATIVE_WIDGET_COMPONENTS" for target in node.targets))
    assert set(samples) == supported
    tab = {"skill": "identity_probe", "tab_id": "all", "ws_prefix": "ext:identity_probe:", "render": {
        "kind": "declarative", "schema_version": 1,
        "components": [{"type": kind, "id": kind, **sample} for kind, sample in samples.items()],
    }}
    with _page(browser_name, [tab]) as (page, _state):
        page.locator('[data-widget-component="id:form"]').wait_for()
        page.evaluate("""() => fixtureWs.emit('message', {type: 'ext:identity_probe:tick',
            data: {count: 1, progress: 25, pairs: [{key: 'One', value: 1}], rows: [{count: 1}]}})""")
        keys = page.locator('[data-widget-mount] > [data-widget-component]').evaluate_all("nodes => nodes.map(node => node.dataset.widgetComponent)")
        assert set(keys) == {f"id:{kind}" for kind in supported}



@pytest.mark.ui_browser
@pytest.mark.parametrize("browser_name", ("chromium", "webkit"))
def test_widget_failed_list_retry_preserves_frames_stop_and_initial_recovery(browser_name):
    tabs = [{"skill": "retry_probe", "tab_id": tab_id, "title": tab_id, "render": {
        "kind": "iframe", "route": "frame", "start": mode,
    }} for tab_id, mode in [("kept", "retain"), ("stopped", "auto")]]
    with _page(browser_name, tabs) as (page, state):
        kept = page.locator('[data-widget-key="retry_probe:kept"]')
        stopped = page.locator('[data-widget-key="retry_probe:stopped"]')
        kept.frame_locator("iframe").locator("#increment").click()
        stopped.locator('[data-widget-power]').click()
        page.wait_for_function("!document.querySelector('[data-widget-key=\"retry_probe:stopped\"] iframe')")
        page.evaluate("window.keptFrame = document.querySelector('[data-widget-key=\"retry_probe:kept\"] iframe')")
        state["list_error"] = True
        page.evaluate("fixtureWs.emit('open', {})")
        page.locator('#widgets-list-error').wait_for()
        assert page.get_by_role("button", name="Retry", exact=True).is_visible()
        state["list_error"] = False
        page.get_by_role("button", name="Retry", exact=True).click()
        page.locator('#widgets-list-error').wait_for(state="hidden")
        assert page.evaluate("document.querySelector('[data-widget-key=\"retry_probe:kept\"] iframe') === keptFrame")
        assert kept.frame_locator("iframe").locator("#count").inner_text() == "1"
        assert stopped.locator("iframe").count() == 0
        assert stopped.locator('[data-widget-power]').inner_text() == "Start"
        state["list_error"] = True
        page.reload(wait_until="domcontentloaded")
        page.locator('#widgets-list-error').wait_for()
        assert page.locator('[data-widget-key]').count() == 0
        assert "No live widgets" not in page.locator('#page-widgets').inner_text()
        state["list_error"] = False
        page.get_by_role("button", name="Retry", exact=True).click()
        kept.frame_locator("iframe").locator("#increment").wait_for()
        assert page.locator('#widgets-list-error').is_hidden()


@pytest.mark.ui_browser
@pytest.mark.parametrize("browser_name", ("chromium", "webkit"))
def test_a_widget_start_the_page_did_not_outlive_rests_until_start(browser_name):
    """An earlier page froze while this revision of the frame was starting: the card rests on its facade with the
    held line until Start, whose own start leaves the ledger once the frame loaded and a second passed. An entry
    for an older revision holds nothing: the new code starts as usual."""
    tabs = [{"skill": "hold_probe", "tab_id": tab_id, "title": tab_id, "revision": "r7", "render": {
        "kind": "iframe", "route": "frame", "start": "auto",
    }} for tab_id in ("frozen", "fixed")]
    ledger = {"widget:hold_probe:frozen": {"reason": "start", "rev": "r7", "at": 1},
              "widget:hold_probe:fixed": {"reason": "start", "rev": "r6", "at": 1}}
    with _page(browser_name, tabs, ledger) as (page, _state):
        frozen = page.locator('[data-widget-key="hold_probe:frozen"]')
        page.locator('[data-widget-key="hold_probe:fixed"]').frame_locator("iframe").locator("#increment").wait_for()
        frozen.locator("[data-widget-facade]").wait_for()
        assert frozen.locator("iframe").count() == 0
        assert "Held back: the page stopped while this was starting last time." in frozen.inner_text()
        assert frozen.locator("[data-widget-power]").inner_text() == "Start"
        frozen.locator("[data-widget-power]").click()
        frozen.frame_locator("iframe").locator("#increment").wait_for()
        page.wait_for_function("""() => {
            const entries = JSON.parse(localStorage.getItem('ouroboros.holds') || '{}').entries || {};
            return !entries['widget:hold_probe:frozen'] && !entries['widget:hold_probe:fixed'];
        }""", timeout=5_000)


@pytest.mark.ui_browser
@pytest.mark.parametrize("browser_name", ("chromium", "webkit"))
def test_widget_policy_menu_uses_shared_keyboard_position_and_page_lifetime(browser_name):
    tab = {"skill": "menu_probe", "tab_id": "main", "title": "Kept widget", "render": {
        "kind": "iframe", "route": "frame", "start": "retain",
    }}
    with _page(browser_name, [tab]) as (page, state):
        page.set_viewport_size({"width": 390, "height": 420})
        card = page.locator('[data-widget-key="menu_probe:main"]')
        trigger = card.locator('[data-widget-menu-trigger]')
        card.frame_locator('iframe').locator('#increment').click()
        page.evaluate("window.keptFrame = document.querySelector('iframe')")
        assert card.locator('.skills-card-menu-dialog:not([data-leaving])').is_hidden()
        trigger.click()
        menu = page.locator('body > .skills-card-menu-dialog:not([data-leaving])[open]')
        menu.wait_for()
        geometry = menu.evaluate("""menu => {
            const r = menu.getBoundingClientRect();
            return {fixed: getComputedStyle(menu).position === 'fixed',
                inViewport: r.left >= 0 && r.top >= 0 && r.right <= innerWidth && r.bottom <= innerHeight,
                focused: document.activeElement.dataset.widgetStartMode};
        }""")
        assert geometry == {"fixed": True, "inViewport": True, "focused": "retain"}
        page.keyboard.press('Home')
        assert page.evaluate("document.activeElement.dataset.widgetStartMode") == "auto"
        page.keyboard.press('ArrowDown')
        assert page.evaluate("document.activeElement.dataset.widgetStartMode") == "manual"
        page.keyboard.press('Escape')
        assert trigger.evaluate("trigger => trigger === document.activeElement")
        assert page.locator('body > .skills-card-menu-dialog:not([data-leaving])').count() == 0
        assert card.locator('.skills-card-menu-dialog:not([data-leaving])').is_hidden()
        page.keyboard.press('Enter')
        page.locator('body > .skills-card-menu-dialog:not([data-leaving])[open]').wait_for()
        page.keyboard.press('Escape')
        assert trigger.evaluate("trigger => trigger === document.activeElement")
        trigger.click()
        page.get_by_role('menuitemradio', name='Manual', exact=True).click()
        page.wait_for_function("document.querySelector('[data-widget-start-mode=manual]').getAttribute('aria-checked') === 'true'")
        assert state["preferences"]["widget_start_mode"] == {"menu_probe:main": "manual"}
        assert page.evaluate("document.querySelector('iframe') === keptFrame")
        trigger.click()
        page.get_by_role('menuitemradio', name='Keep running', exact=True).click()
        page.wait_for_function("document.querySelector('[data-widget-start-mode=retain]').getAttribute('aria-checked') === 'true'")
        trigger.click()
        page.evaluate("showWidgets('chat')")
        assert page.locator('body > .skills-card-menu-dialog:not([data-leaving])').count() == 0
        assert page.evaluate("document.querySelector('iframe') === keptFrame")
        assert card.frame_locator('iframe').locator('#count').inner_text() == '1'


@pytest.mark.ui_browser
@pytest.mark.parametrize("browser_name", ("chromium", "webkit"))
def test_job_feedback_uses_its_own_poll_payload_and_retains_progress(browser_name):
    declaration = _declaration()
    declaration["render"]["components"].append({"type": "progress", "id": "job-progress", "path": "progress"})
    with _page(browser_name, [declaration]) as (page, state):
        state["hold"] = "/status"
        job = page.locator('[data-widget-action="id:job"]')
        other = page.locator('[data-widget-action="id:action"]')
        feedback = page.locator('[data-widget-feedback="id:job"]')
        with page.expect_request("**/status?job_id=same-job"):
            job.click()
        other.click()
        page.locator('[data-widget-feedback="id:action"][data-state="success"]').wait_for()
        # A transport retry keeps this job's own last message, never the other
        # action's result, even though both write the default result target.
        with page.expect_request("**/status?job_id=same-job"):
            state["pending"].pop().fulfill(status=503, content_type="application/json", body='{"error":"Temporary"}')
        assert feedback.inner_text() == "Job started."
        with page.expect_request("**/status?job_id=same-job"):
            state["pending"].pop().fulfill(content_type="application/json", body=json.dumps({
                "status": "running", "message": "Own job progress", "progress": 40,
            }))
        assert feedback.inner_text() == "Own job progress"
        other.click()
        page.locator('[data-widget-feedback="id:action"][data-state="success"]').wait_for()
        # A pending poll without a message also must not borrow the target's
        # previous message; the existing per-job progress clamp remains intact.
        with page.expect_request("**/status?job_id=same-job"):
            state["pending"].pop().fulfill(content_type="application/json", body='{"status":"running","progress":30}')
        assert feedback.inner_text() == "Working…"
        assert page.locator('[data-widget-component="id:job-progress"] progress').get_attribute("value") == "40"
        assert job.is_disabled()
        state["pending"].pop().fulfill(content_type="application/json", body='{"status":"done","result":{"message":"Own job complete"}}')
        page.wait_for_function("document.querySelector('[data-widget-feedback=\"id:job\"]').dataset.state === 'success'")
        assert feedback.inner_text() == "Own job complete"


@pytest.mark.ui_browser
@pytest.mark.parametrize("browser_name", ("chromium", "webkit"))
@pytest.mark.parametrize("clipboard_ok", (True, False))
def test_widget_download_feedback_preserves_copy_link_or_refusal(browser_name, clipboard_ok):
    with _page(browser_name, [_declaration()]) as (page, _state):
        download = page.locator('[data-widget-download-url]')
        download.wait_for()
        page.evaluate("""clipboardOk => {
            window.pywebview = {api: {download_file_to_downloads: async () => ({ok: false, error: 'Launcher refused download'})}};
            Object.defineProperty(navigator, 'clipboard', {configurable: true, value: {
                writeText: async text => { if (!clipboardOk) throw new Error('Clipboard unavailable'); window.copiedLink = text; },
            }});
            document.execCommand = () => false;
        }""", clipboard_ok)
        download.click()
        feedback = page.locator('[data-widget-feedback="id:file"]')
        page.wait_for_function("document.querySelector('[data-widget-feedback=\"id:file\"]').dataset.state !== 'loading'")
        if clipboard_ok:
            assert feedback.get_attribute("data-state") == "warning"
            assert feedback.inner_text() == "Download unavailable. Link copied."
            assert page.evaluate("window.copiedLink") == f"{ORIGIN}/api/extensions/identity_probe/file"
        else:
            assert feedback.get_attribute("data-state") == "error"
            assert feedback.inner_text() == "Launcher refused download"
        assert download.is_enabled()


@pytest.mark.ui_browser
@pytest.mark.parametrize("browser_name", ("chromium", "webkit"))
def test_widget_policy_binding_survives_persisted_pagehide_but_disposes_on_exit(browser_name):
    tab = {"skill": "menu_probe", "tab_id": "main", "render": {"kind": "iframe", "route": "frame", "start": "retain"}}
    with _page(browser_name, [tab]) as (page, _state):
        trigger = page.locator('[data-widget-menu-trigger]')
        trigger.wait_for()
        page.evaluate("""() => {
            dispatchEvent(new PageTransitionEvent('pagehide', {persisted: true}));
            dispatchEvent(new PageTransitionEvent('pageshow', {persisted: true}));
        }""")
        trigger.click()
        page.locator('body > .skills-card-menu-dialog:not([data-leaving])[open]').wait_for()
        page.keyboard.press("Escape")
        page.evaluate("dispatchEvent(new PageTransitionEvent('pagehide', {persisted: false}))")
        trigger.click()
        assert page.locator('body > .skills-card-menu-dialog:not([data-leaving])[open]').count() == 0
