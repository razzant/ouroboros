"""Opt-in author controls through real auth, frame, bridge and OOP routes.

The TCP listener is loopback; only the fixture ASGI peer is simulated as a LAN
client. This tests the real network gate without exposing a test server on LAN.
"""
from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

import pytest

from tests import test_widget_stream_download_ui as widget_fixture
from tests._shared import producer_evidence_dir
from tests.test_author_ui_kit import EXAMPLE, _copy_installed_runtime

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]
REPO = Path(__file__).resolve().parents[1]

_HOST = """<!doctype html><html><head><meta charset="utf-8">
<script src="/static/theme.js"></script>
<link rel="stylesheet" href="/static/ui.css"><link rel="stylesheet" href="/static/style.css">
</head><body><h1>Installed author controls</h1><div id="consumers"></div>
<script type="module">
import {mountModuleWidget, mountRouteIframeWidget} from '/static/modules/widget_module.js';
window.handlers = new Set(); window.disposers = {}; window.disposed = 0;
// Each mount mints its lease for a live tab: the module tab, or the route page's tab for any own route.
const live = (await (await fetch('/api/widgets')).json()).ui_tabs.filter(row => row.skill === 'export_widget');
window.addEventListener('message', event => { if (event.data?.fixtureDisposed) window.disposed++; });
window.mountExample = async (id, kind) => {
  const card = document.createElement('section'); card.dataset.widgetKey = id;
  card.innerHTML = '<h2></h2><div class="widgets-card-status"></div><div class="mount"></div>';
  card.querySelector('h2').textContent = id;
  document.getElementById('consumers').append(card);
  const row = live.find(row => row.tab_id === (kind === 'module' ? 'module' : 'page'));
  const mount = card.querySelector('.mount'), tab = {skill:'export_widget', ws_prefix:'ext:export_widget:',
    tab_id: row.tab_id, generation: row.generation};
  window.disposers[id] = kind === 'module'
    ? await mountModuleWidget(mount, tab, {entry:'widget.js', height:420, appearance:'host'}, null, window.handlers)
    : await mountRouteIframeWidget(mount, tab, {route:kind, height:420});
};
await mountExample('module-old', 'module');
await mountExample('page-old', 'page');
await mountExample('custom', 'custom');
window.ready = true;
</script></body></html>"""

_CUSTOM = '''
def custom(request):
    return HTMLResponse('<!doctype html><title>Independent app</title>'
        '<style>body{background:rgb(255,250,220);color:rgb(35,40,50);font:19px serif}'
        'button{border-radius:3px;padding:5px}</style><button>Independent design</button>')
original_register = register
def register(api):
    original_register(api)
    api.register_route('custom', custom)
'''


def _style_document(installed, document):
    """A synthetic reference page: one product document's stylesheet links, in order, and the real primitives.

    Its root has neither `ouro-ui` nor `data-theme`, so it shows the base recipes an author area receives,
    not the product document's themed capsules.
    """
    source = (installed / "web" / document).read_text(encoding="utf-8")
    links = "\n".join(re.findall(r'<link\b[^>]*rel="stylesheet"[^>]*>', source))
    assert '/static/ui.css' in links, "the shared layer must serve both product documents"
    return f"""<!doctype html><html><head>{links}</head><body>
<main class="ouro-ui"><form id="controls"></form></main><script type="module">
import {{renderSafeField}} from '/static/modules/ui_primitives.js';
document.getElementById('controls').innerHTML = [
 {{name:'title',label:'Title',help:'A name for this example.',default:'My notes'}},
 {{name:'view',label:'View',type:'select',options:['List','Grid'],default:'List'}},
 {{name:'enabled',label:'Enabled',type:'checkbox',default:true}}
].map(field => renderSafeField(field)).join('') + '<button type="button" class="btn btn-default">Preview</button>';
</script></body></html>"""


@pytest.fixture
def author_kit_server(tmp_path, monkeypatch):
    if os.environ.get("OUROBOROS_RUN_UI_SMOKE") != "1":
        pytest.skip("set OUROBOROS_RUN_UI_SMOKE=1 to run browser UI smoke")
    import uvicorn
    from starlette.responses import HTMLResponse
    from starlette.routing import Mount, Route

    from ouroboros import server_auth
    from ouroboros.server_web import NoCacheStaticFiles
    from tests import _extension_loader_shared as extension_fixture

    installed = tmp_path / "installed"
    shutil.copytree(REPO / "web", installed / "web")
    _copy_installed_runtime(installed)
    monkeypatch.setattr(widget_fixture, "REPO", installed)
    monkeypatch.setattr(widget_fixture, "_PLUGIN", (EXAMPLE / "plugin.py").read_text(encoding="utf-8") + _CUSTOM)
    monkeypatch.setattr(widget_fixture, "_WIDGET", (EXAMPLE / "widget.js").read_text(encoding="utf-8").replace(
        "/api/extensions/author_ui_kit/", "/api/extensions/export_widget/"))
    monkeypatch.setattr(widget_fixture, "_HTML", _HOST)
    original_writer = extension_fixture._write_ext_skill

    def write_example(*args, **kwargs):
        kwargs["extra_frontmatter"] = 'plugin_api: "2.0"\n' + kwargs.get("extra_frontmatter", "")
        return original_writer(*args, **kwargs)

    monkeypatch.setattr(extension_fixture, "_write_ext_skill", write_example)
    monkeypatch.setenv("OUROBOROS_NETWORK_PASSWORD", "author-kit-test-password")
    monkeypatch.setattr("ouroboros.config.DATA_DIR", tmp_path / "drive")
    monkeypatch.setattr(server_auth, "_auth_secret_cache", None)
    requests = []
    original_config = uvicorn.Config

    def configured(gated, **kwargs):
        # widget_server already puts its app behind the real gate (leases); add the product mounts behind it.
        gated.app.router.routes.extend([
            # The product mounts: the core's builtin tree under /onboarding/static, the interface at /static.
            Mount("/onboarding/static", NoCacheStaticFiles(directory=installed / "web")),
            Mount("/static", NoCacheStaticFiles(directory=installed / "web")),
            Route("/style-spa", lambda request: HTMLResponse(_style_document(installed, "index.html"))),
            Route("/style-wizard", lambda request: HTMLResponse(_style_document(installed, "onboarding_template.html"))),
        ])

        async def simulated_peer(scope, receive, send):
            if scope["type"] not in {"http", "websocket"}:
                return await gated(scope, receive, send)
            scope = {**scope, "client": ("198.51.100.23", scope.get("client", (None, 0))[1])}
            headers = dict(scope.get("headers", []))
            record = {"path": scope["path"], "cookie": b"cookie" in headers,
                      "origin": headers.get(b"origin", b"").decode(), "status": None}
            requests.append(record)

            async def observed(message):
                if message["type"] == "http.response.start":
                    record["status"] = message["status"]
                await send(message)

            await gated(scope, receive, observed)

        return original_config(simulated_peer, **kwargs)

    monkeypatch.setattr(uvicorn, "Config", configured)
    fixture = widget_fixture.widget_server.__wrapped__(tmp_path, monkeypatch)
    with_fixture = next(fixture)
    try:
        yield {**with_fixture, "installed": installed, "requests": requests}
    finally:
        with pytest.raises(StopIteration):
            next(fixture)


def _frame(page, key):
    element = page.locator(f'[data-widget-key="{key}"] iframe').element_handle()
    assert element is not None
    return element.content_frame()


def _metrics(frame, selector):
    return frame.locator(selector).evaluate("""async element => {
        // Force the state change into style before observing finite transitions.
        getComputedStyle(element).borderColor;
        await Promise.allSettled(element.getAnimations()
            .filter(animation => Number.isFinite(animation.effect.getComputedTiming().endTime))
            .map(animation => animation.finished));
        const style = getComputedStyle(element);
        const keys = ['fontFamily','fontSize','paddingTop','paddingRight',
            'paddingBottom','paddingLeft','borderRadius','color','borderColor',
            'boxShadow','outlineStyle','outlineWidth'];
        if (element.type === 'checkbox') keys.push('width', 'height', 'accentColor', 'backgroundColor');
        // Native checkbox glyphs have geometry/color but render no text font.
        return Object.fromEntries(keys.filter(key => element.type !== 'checkbox' || !key.startsWith('font'))
            .map(key => [key,style[key]]));
    }""")


@pytest.mark.parametrize("browser_name", ["chromium", "webkit"])
def test_author_kit_authenticated_mount_and_lifetime(author_kit_server, tmp_path, browser_name):
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import sync_playwright

    server = author_kit_server
    with sync_playwright() as playwright:
        browser = getattr(playwright, browser_name).launch(headless=True)
        try:
            context = browser.new_context(viewport={"width": 1100, "height": 850}, accept_downloads=True)
            page = context.new_page()
            page.emulate_media(color_scheme="dark")
            errors = []
            style_mismatches = []
            browser_requests = []
            page.add_init_script("""window.kitCspViolations = [];
                window.addEventListener('securitypolicyviolation', event =>
                    kitCspViolations.push({directive:event.violatedDirective, blocked:event.blockedURI}));""")
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.on("request", lambda request: browser_requests.append(request))
            response = page.goto(server["url"])
            assert response.status == 401
            page.get_by_label("Password", exact=True).fill("author-kit-test-password")
            page.get_by_role("button", name="Unlock", exact=True).click()
            page.wait_for_function("window.ready === true")
            module, route, custom = [_frame(page, name) for name in ("module-old", "page-old", "custom")]
            for frame in (module, route):
                frame.locator('.ui-control').first.wait_for()
            module_node = page.locator('[data-widget-key="module-old"] iframe').element_handle()
            assert route.evaluate("window.kitCspViolations") == []
            assert module.evaluate("document.documentElement.dataset.theme") == "dark"
            page.evaluate("() => window.ouroTheme.set('light')")
            module.locator('html[data-theme="light"]').wait_for(state="attached")
            assert page.evaluate(
                "node => document.querySelector('[data-widget-key=\\\"module-old\\\"] iframe') === node",
                module_node,
            )
            assert module.evaluate(
                "getComputedStyle(document.querySelector('.ui-control')).backgroundColor"
            ) == "rgb(245, 246, 248)"
            page.evaluate("() => window.ouroTheme.set('dark')")
            module.locator('html[data-theme="dark"]').wait_for(state="attached")
            for frame in (module, route):
                frame.get_by_role("button", name="Preview", exact=True).wait_for()
                assert frame.get_by_label("Title", exact=True).input_value() == "My notes"
                frame.get_by_label("Title", exact=True).fill("")
                frame.get_by_role("button", name="Preview", exact=True).click()
                assert frame.locator('[role="status"]').inner_text() == "Enter a title."
                assert frame.locator('[role="status"]').get_attribute("data-tone") == "danger"
                frame.get_by_label("Title", exact=True).fill("Personal")
                # select_option assigns the value without focusing the control.
                # Exercise the native focus/blur cycle as part of choosing a view.
                frame.get_by_label("View", exact=True).focus()
                frame.get_by_label("View", exact=True).select_option("Grid")
                frame.get_by_label("Enabled", exact=True).uncheck()
                frame.get_by_role("button", name="Preview", exact=True).click()
                assert frame.locator('[role="status"]').inner_text() == "Personal: Grid, disabled"
                assert frame.locator('[role="status"]').get_attribute("data-tone") == "ok"
                frame.evaluate("document.activeElement.blur()")
                assert frame.evaluate("window.kitCspViolations") == []
            page.mouse.move(0, 0)
            assert route.evaluate("typeof window.OuroborosWidget") == "undefined"
            assert not any('/static/' in request.url and request.frame == route for request in browser_requests)
            assert not any('/static/' in request.url and request.frame == module for request in browser_requests)
            for element in page.locator("iframe").all():
                assert element.get_attribute("sandbox") == "allow-scripts allow-pointer-lock allow-downloads"
                assert element.get_attribute("allow") == "autoplay; fullscreen; clipboard-write"
            assert not page.locator(".widgets-card-status").all_text_contents()[0].strip()

            # Compare the author frames' controls with each synthetic reference page by the listed
            # computed metrics: not byte identity with, nor themed parity to, the product documents.
            style_pages = [context.new_page() for _ in range(2)]
            for style_page, name in zip(style_pages, ("spa", "wizard")):
                style_page.goto(server["url"] + "/style-" + name)
                style_page.locator('[name="title"]').wait_for()
                style_page.get_by_label("View", exact=True).focus()
                style_page.get_by_label("View", exact=True).select_option("Grid")
                style_page.get_by_label("Enabled", exact=True).uncheck()
                style_page.evaluate("document.activeElement.blur()")
                for selector in ('[name="title"]', '[name="view"]', '[name="enabled"]', '.btn'):
                    for frame in (module, route):
                        actual, expected = _metrics(frame, selector), _metrics(style_page, selector)
                        if actual != expected:
                            style_mismatches.append({"document": name, "selector": selector,
                                "frame": frame.url, "actual": actual, "expected": expected})
                focused = []
                for frame in (module, route, style_page):
                    frame.locator('[name="title"]').focus()
                    focused.append(_metrics(frame, '[name="title"]'))
                    frame.evaluate("document.activeElement.blur()")
                assert focused[0] == focused[1] == focused[2]
                for checked, disabled, focus in ((True, False, True), (True, True, False)):
                    checkbox_metrics = []
                    for frame in (module, route, style_page):
                        checkbox = frame.get_by_label("Enabled", exact=True)
                        checkbox.evaluate("(element, state) => Object.assign(element, state)",
                                          {"checked": checked, "disabled": disabled})
                        if focus:
                            checkbox.focus()
                        assert checkbox.is_checked() is checked
                        assert checkbox.is_disabled() is disabled
                        checkbox_metrics.append(_metrics(frame, '[name="enabled"]'))
                        checkbox.evaluate("element => { element.blur(); element.disabled=false; element.checked=false; }")
                    assert checkbox_metrics[0] == checkbox_metrics[1] == checkbox_metrics[2]

            # Own-route fetch is authenticated by the parent. A credential-less
            # opaque subresource gets a real refusal, independent of kit loading.
            assert any(row["path"].endswith("/author-kit") and row["cookie"] and row["status"] == 200
                       for row in server["requests"])
            assert any(row["path"].endswith("/page") and row["cookie"] and row["status"] == 200
                       for row in server["requests"])
            with page.expect_response(lambda response: response.url.endswith('/static/ui.css') and response.status == 401):
                opaque = page.evaluate_handle("""() => {
                    const iframe = document.createElement('iframe'); iframe.sandbox = 'allow-scripts';
                    iframe.srcdoc = '<script>fetch("/static/ui.css", {credentials:"omit",mode:"no-cors",cache:"no-store"}).catch(()=>{});<\\/script>';
                    document.body.append(iframe); return iframe;
                }""")
            assert any(row["path"] == "/static/ui.css" and not row["cookie"] and row["status"] == 401
                       for row in server["requests"])
            opaque.evaluate("element => element.remove()")

            custom_before = _metrics(custom, "button")
            old_radius = _metrics(route, ".btn")["borderRadius"]
            module.evaluate("""() => {
                const override = document.createElement('style');
                override.textContent = '.ouro-ui .btn {border-radius:37px}'; document.head.append(override);
            }""")
            assert _metrics(module, ".btn")["borderRadius"] == "37px"
            css = server["installed"] / "web/ui.css"
            css.write_text(
                css.read_text(encoding="utf-8")
                + '\n.ouro-ui .btn.btn-default { border-radius: 17px; }\n',
                encoding="utf-8",
            )
            page.evaluate("async () => {await mountExample('module-new','module'); await mountExample('page-new','page');}")
            for key in ("module-new", "page-new"):
                frame = _frame(page, key)
                frame.get_by_role("button", name="Preview", exact=True).wait_for()
                assert _metrics(frame, ".btn")["borderRadius"] == "17px"
                assert frame.evaluate("window.kitCspViolations") == []
            for style_page in style_pages:
                style_page.reload()
                style_page.locator(".btn").wait_for()
                assert _metrics(style_page, ".btn")["borderRadius"] == "17px"
            assert old_radius != "17px"
            assert _metrics(route, ".btn")["borderRadius"] == old_radius
            assert _metrics(module, ".btn")["borderRadius"] == "37px"
            assert _metrics(custom, "button") == custom_before

            # The kit leaves the existing events/download/ordered-dispose bridge usable.
            module.evaluate("""() => {
                const off = OuroborosWidget.onEvent(event => document.getElementById('root').dataset.event = event.data.value);
                window.__ouroWidgetOnDispose(() => {off(); window.parent.postMessage({fixtureDisposed:true}, '*')});
                const button = document.createElement('button'); button.textContent = 'Export example';
                button.onclick = () => OuroborosWidget.download('author-kit.txt', new Blob(['author kit export']));
                document.getElementById('root').append(button);
            }""")
            page.wait_for_function("window.handlers.size === 1")
            page.evaluate("() => handlers.forEach(handler => handler({type:'ext:export_widget:tick', data:{value:'delivered'}}))")
            module.locator('#root[data-event="delivered"]').wait_for(state="attached")
            with page.expect_download() as download:
                module.get_by_role("button", name="Export example", exact=True).click()
            assert Path(download.value.path()).read_text(encoding="utf-8") == "author kit export"

            page.route("**/api/extensions/export_widget/author-kit", lambda request: request.fulfill(status=503))
            page.evaluate("() => mountExample('unavailable', 'module')")
            failed = _frame(page, "unavailable")
            failed.get_by_text("Controls unavailable: HTTP 503", exact=True).wait_for()
            assert route.get_by_role("button", name="Preview", exact=True).is_visible()
            evidence = producer_evidence_dir(tmp_path / "evidence", "author-ui-kit",
                                             env="OUROBOROS_UI_EVIDENCE_OUT")
            evidence.mkdir(parents=True, exist_ok=True)
            page.evaluate("() => window.ouroTheme.set('light')")
            module.locator('html[data-theme="light"]').wait_for(state="attached")
            module.locator('body').screenshot(
                path=str(evidence / f"author-kit-{browser_name}-module-light.png")
            )
            for key in ("module-old", "page-old", "custom", "module-new", "page-new", "unavailable"):
                card = page.locator(f'[data-widget-key="{key}"]')
                card.scroll_into_view_if_needed()
                page.evaluate("() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))")
                card.screenshot(path=str(evidence / f"author-kit-{browser_name}-{key}.png"))
            (evidence / f"author-kit-{browser_name}-styles.json").write_text(
                json.dumps(style_mismatches, indent=2), encoding="utf-8",
            )
            page.evaluate("() => Promise.all(Object.values(disposers).map(dispose => dispose()))")
            assert page.locator("[data-widget-key] iframe").count() == 0
            assert page.evaluate("window.handlers.size") == 0
            assert page.evaluate("window.disposed") == 1
            assert errors == [], errors
            assert style_mismatches == [], style_mismatches
        finally:
            browser.close()
