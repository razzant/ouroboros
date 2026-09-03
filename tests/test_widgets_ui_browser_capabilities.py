"""Widgets phase 5 (capability unlock in the frame) browser smoke on chromium and
webkit: the module frame's document CSP and the shared sandbox / permissions
set let reviewed widget code load a sibling classic ``<script src>`` and an
``import()``-ed ES module from the skill's module prefix, instantiate a
``.wasm`` served by the skill's own route through the bridged ``fetch``, run a
``blob:`` Worker, load an image, audio and an ``@font-face`` from the skill's
own route prefix, write the clipboard and start a download from a user click;
while a foreign origin and a sibling skill's prefix stay refused by the
document policy (``connect-src`` closed, ``img-src`` scoped), a font route
without ``Access-Control-Allow-Origin`` fails (the documented CORS interlock),
and a ``kind: iframe`` route page executes its own script with network access
but no SPA cookies or DOM. Kept apart from the lifecycle file so that file
stays under the size-ratchet band; the fixtures are shared by import."""

from __future__ import annotations

import io
import pathlib
import struct
import textwrap
import wave
import zlib

import pytest

from tests.test_ui_smoke_playwright import direct_server_with_data as _direct_server_with_data
from tests.test_widgets_ui_browser_lifecycle import _click_nav

direct_server_with_data = _direct_server_with_data

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

# A 41-byte WebAssembly module exporting `add(i32, i32) -> i32`.
_ADD_WASM = bytes([
    0x00, 0x61, 0x73, 0x6D, 0x01, 0x00, 0x00, 0x00,
    0x01, 0x07, 0x01, 0x60, 0x02, 0x7F, 0x7F, 0x01, 0x7F,
    0x03, 0x02, 0x01, 0x00,
    0x07, 0x07, 0x01, 0x03, 0x61, 0x64, 0x64, 0x00, 0x00,
    0x0A, 0x09, 0x01, 0x07, 0x00, 0x20, 0x00, 0x20, 0x01, 0x6A, 0x0B,
])


def _pixel_png() -> bytes:
    def chunk(kind: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)

    raw = b"\x00" + bytes([0x33, 0x66, 0x99, 0xFF])  # one filter byte + one RGBA pixel
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def _beep_wav() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(1)
        handle.setframerate(8000)
        handle.writeframes(bytes(128 + (40 if (i // 10) % 2 else -40) for i in range(1600)))
    return buffer.getvalue()


_CAPABILITY_PLUGIN = """\
import pathlib

from starlette.responses import HTMLResponse, RedirectResponse, Response

HERE = pathlib.Path(__file__).parent
HITS = {}


def _count(name):
    HITS[name] = HITS.get(name, 0) + 1


def _asset(name, media_type, extra_headers=None):
    async def handler(_request):
        _count(name)
        return Response((HERE / name).read_bytes(), media_type=media_type, headers=extra_headers or {})
    return handler


async def font_nocors(_request):
    _count("font-nocors.woff2")
    return Response((HERE / "font.woff2").read_bytes(), media_type="font/woff2")


async def ping(request):
    # Read by the route page (opaque origin → cross-origin fetch): the CORS
    # header is what lets that page read its own skill's answer.
    _count("ping:" + str(request.query_params.get("from") or ""))
    return Response('{"ok": true}', media_type="application/json", headers={"Access-Control-Allow-Origin": "*"})


async def plain(_request):
    _count("plain")
    return {"ok": True}


async def hits(_request):
    return dict(HITS)


async def elsewhere(_request):
    # A route of THIS skill pointing out of its own prefix. The bridge checks the
    # prefix once, before the request; a followed redirect would carry the frame's
    # method, body, headers and the owner's session wherever the hop points.
    _count("elsewhere")
    return RedirectResponse("/api/state", status_code=307)


async def page(_request):
    _count("page")
    return HTMLResponse(PAGE_HTML)


PAGE_HTML = '''<!doctype html><html><head><title>route</title></head><body><div id="root">route page</div>
<script>
(async () => {
    const out = { ran: true };
    try { out.cookie = 'readable:' + document.cookie; } catch (err) { out.cookie = err.name; }
    try { void window.parent.document; out.parentDom = 'reachable'; } catch (err) { out.parentDom = err.name; }
    try { out.ping = (await fetch('/api/extensions/__SKILL__/ping?from=route')).status; } catch (err) { out.ping = err.name; }
    try { out.plain = (await fetch('/api/extensions/__SKILL__/plain')).status; } catch (err) { out.plain = err.name; }
    try { out.opaque = (await fetch('/api/extensions/__SKILL__/plain', { mode: 'no-cors' })).type; } catch (err) { out.opaque = err.name; }
    document.title = 'route-ran';
    out.done = true;
    window.__routeProbe = out;
})();
</script></body></html>'''


def register(api):
    api.register_route("add.wasm", _asset("add.wasm", "application/wasm"), methods=("GET",))
    api.register_route("pixel.png", _asset("pixel.png", "image/png"), methods=("GET",))
    api.register_route("beep.wav", _asset("beep.wav", "audio/wav"), methods=("GET",))
    # A font is fetched in CORS mode by the opaque-origin frame: the header is the interlock.
    api.register_route("font.woff2", _asset("font.woff2", "font/woff2", {"Access-Control-Allow-Origin": "*"}), methods=("GET",))
    api.register_route("font-nocors.woff2", font_nocors, methods=("GET",))
    api.register_route("ping", ping, methods=("GET",))
    api.register_route("plain", plain, methods=("GET",))
    api.register_route("hits", hits, methods=("GET",))
    api.register_route("elsewhere", elsewhere, methods=("GET",))
    api.register_route("page", page, methods=("GET",))
    api.register_ui_tab("probe", "Capability probe", render={"kind": "module", "entry": "probe.js", "height": 360, "start": "auto"})
    api.register_ui_tab("page", "Route page", render={"kind": "iframe", "route": "page", "height": 320, "start": "auto"})
"""

_HELPER_JS = """\
window.__helperLoaded = 'classic';
"""

_ESM_MJS = """\
export const helper = () => 'esm';
"""

# Child-side probe: everything runs inside the module frame under its document
# policy; the test observes the outcomes through the frame handle.
_CAPABILITY_PROBE_JS = """\
(() => {
    const root = document.getElementById('root');
    root.textContent = 'Capability probe';
    const base = '/api/extensions/__SKILL__/';
    const violations = [];
    document.addEventListener('securitypolicyviolation', (event) => {
        violations.push({ directive: event.effectiveDirective || event.violatedDirective, blocked: event.blockedURI });
    });
    const settle = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
    const loadImage = (src) => new Promise((resolve) => {
        const img = new Image();
        img.onload = () => resolve({ ok: true, width: img.naturalWidth });
        img.onerror = () => resolve({ ok: false, width: img.naturalWidth });
        img.src = src;
        document.body.appendChild(img);
    });
    const fontStatus = (family) => {
        for (const face of document.fonts) {
            if (face.family.replace(/["']/g, '') === family) return face.status;
        }
        return 'absent';
    };
    // User-activated controls: clipboard write, download and pointer lock need
    // a real click, which the test performs on these nodes.
    const copy = document.createElement('button');
    copy.id = 'copy';
    copy.textContent = 'copy';
    copy.addEventListener('click', () => {
        navigator.clipboard.writeText('widget-copy').then(
            () => { window.__clipboard = 'ok'; },
            (err) => { window.__clipboard = `${err.name}: ${err.message}`; },
        );
    });
    const link = document.createElement('a');
    link.id = 'dl';
    link.textContent = 'download';
    link.download = 'probe.txt';
    link.href = URL.createObjectURL(new Blob(['hello from the widget'], { type: 'text/plain' }));
    const lock = document.createElement('button');
    lock.id = 'lock';
    lock.textContent = 'lock';
    lock.addEventListener('click', () => {
        const outcome = { event: null, promise: null };
        window.__pointerLock = outcome;
        document.addEventListener('pointerlockchange', () => { outcome.event = document.pointerLockElement ? 'locked' : 'released'; }, { once: true });
        document.addEventListener('pointerlockerror', () => { outcome.event = 'pointerlockerror'; }, { once: true });
        try {
            const result = root.requestPointerLock();
            if (result && typeof result.then === 'function') result.then(() => { outcome.promise = 'ok'; }, (err) => { outcome.promise = `${err.name}: ${err.message}`; });
            else outcome.promise = 'no promise';
        } catch (err) { outcome.promise = `thrown ${err.name}: ${err.message}`; }
    });
    root.append(copy, link, lock);
    window.__capProbe = {
        csp() { return document.querySelector('meta[http-equiv="Content-Security-Policy"]').getAttribute('content'); },
        origin() { return { origin: window.origin, location: window.location.href }; },
        async classicScript() {
            await new Promise((resolve, reject) => {
                const script = document.createElement('script');
                script.src = base + 'module/lib/helper.js';
                script.onload = resolve;
                script.onerror = () => reject(new Error('classic sibling script blocked'));
                document.head.appendChild(script);
            });
            return window.__helperLoaded;
        },
        async esmImport() {
            const mod = await import(base + 'module/lib/esm.mjs');
            return mod.helper();
        },
        async wasm() {
            const response = await OuroborosWidget.fetch(base + 'add.wasm');
            const { instance } = await WebAssembly.instantiate(await response.arrayBuffer());
            return { sum: instance.exports.add(40, 2), contentType: response.headers.get('content-type') };
        },
        async wasmStreaming() {
            try {
                const { instance } = await WebAssembly.instantiateStreaming(OuroborosWidget.fetch(base + 'add.wasm'));
                return { sum: instance.exports.add(1, 2), error: null };
            } catch (err) { return { sum: null, error: `${err.name}: ${err.message}` }; }
        },
        async blobWorker() {
            const url = URL.createObjectURL(new Blob(['self.onmessage = (event) => postMessage(event.data * 2);'], { type: 'text/javascript' }));
            const worker = new Worker(url);
            try {
                return await Promise.race([
                    new Promise((resolve, reject) => {
                        worker.onmessage = (event) => resolve(event.data);
                        worker.onerror = (event) => reject(new Error(event.message || 'worker error'));
                        worker.postMessage(21);
                    }),
                    settle(5000).then(() => { throw new Error('worker timed out'); }),
                ]);
            } finally { worker.terminate(); URL.revokeObjectURL(url); }
        },
        ownImage() { return loadImage(base + 'pixel.png'); },
        async siblingImage() {
            const before = violations.length;
            const result = await loadImage('/api/extensions/other_skill/pixel.png');
            await settle(100);
            return { ...result, violations: violations.slice(before) };
        },
        async foreignImage() {
            const before = violations.length;
            const result = await loadImage('http://127.0.0.1:1/x.png');
            await settle(100);
            return { ...result, violations: violations.slice(before) };
        },
        foreignXhr() {
            return new Promise((resolve) => {
                const before = violations.length;
                const xhr = new XMLHttpRequest();
                const done = (outcome) => settle(100).then(() => resolve({ outcome, status: xhr.status, violations: violations.slice(before) }));
                xhr.onerror = () => done('error');
                xhr.onload = () => done('loaded');
                xhr.open('GET', 'http://127.0.0.1:1/x');
                try { xhr.send(); } catch (err) { done(`thrown ${err.name}`); }
            });
        },
        async audio() {
            const audio = document.createElement('audio');
            audio.src = base + 'beep.wav';
            document.body.appendChild(audio);
            const loaded = await Promise.race([
                new Promise((resolve) => { audio.onloadedmetadata = () => resolve('loadedmetadata'); audio.onerror = () => resolve('error'); }),
                settle(5000).then(() => 'timeout'),
            ]);
            let play;
            try { await audio.play(); play = 'playing'; } catch (err) { play = `${err.name}`; }
            audio.pause();
            return { loaded, duration: audio.duration, play };
        },
        async font(path, family) {
            const style = document.createElement('style');
            style.textContent = `@font-face { font-family: "${family}"; src: url("${base}${path}") format("woff2"); }`;
            document.head.appendChild(style);
            const span = document.createElement('span');
            span.style.fontFamily = `"${family}"`;
            span.textContent = 'x';
            document.body.appendChild(span);
            try { await document.fonts.load(`12px "${family}"`); } catch {}
            for (let i = 0; i < 50 && !['loaded', 'error'].includes(fontStatus(family)); i++) await settle(100);
            return fontStatus(family);
        },
        fullscreen() {
            return { enabled: document.fullscreenEnabled ?? null, webkitEnabled: document.webkitFullscreenEnabled ?? null };
        },
        clipboard() { return window.__clipboard ?? null; },
        async redirected() {
            try {
                const r = await OuroborosWidget.fetch(base + 'elsewhere');
                return { followed: true, status: r.status };
            } catch (err) {
                return { followed: false, error: String((err && err.message) || err) };
            }
        },
        pointerLock() { return window.__pointerLock ?? null; },
        violations() { return violations.slice(); },
    };
})();
"""


def _write_capability_widget_extension(data_dir: pathlib.Path) -> str:
    """Install the capability fixture: an `auto` module probe with a classic
    sibling script, an ES module sibling, a 41-byte `.wasm`, a 1×1 PNG, a short
    WAV and a real `.woff2` (the repo's own KaTeX font) as payload files served
    by the skill's routes (the font once with the CORS header, once without),
    plus a `kind: iframe` route page whose script probes its own reach."""
    from ouroboros.skill_loader import SkillReviewState, compute_content_hash, save_review_state

    name = "capability_widget_smoke"
    skill_dir = data_dir / "skills" / "external" / name
    (skill_dir / "lib").mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        textwrap.dedent(
            f"""\
            ---
            name: {name}
            description: Isolated widget frame capability fixture.
            version: 0.1.0
            type: extension
            entry: plugin.py
            permissions: ["route", "widget"]
            ---
            # Widget capability fixture
            """
        ),
        encoding="utf-8",
    )
    (skill_dir / "plugin.py").write_text(_CAPABILITY_PLUGIN.replace("__SKILL__", name), encoding="utf-8")
    (skill_dir / "probe.js").write_text(_CAPABILITY_PROBE_JS.replace("__SKILL__", name), encoding="utf-8")
    (skill_dir / "lib" / "helper.js").write_text(_HELPER_JS, encoding="utf-8")
    (skill_dir / "lib" / "esm.mjs").write_text(_ESM_MJS, encoding="utf-8")
    (skill_dir / "add.wasm").write_bytes(_ADD_WASM)
    (skill_dir / "pixel.png").write_bytes(_pixel_png())
    (skill_dir / "beep.wav").write_bytes(_beep_wav())
    (skill_dir / "font.woff2").write_bytes((REPO_ROOT / "web" / "vendor" / "katex" / "fonts" / "KaTeX_Main-Regular.woff2").read_bytes())
    content_hash = compute_content_hash(skill_dir, manifest_entry="plugin.py")
    save_review_state(data_dir, name, SkillReviewState(status="pass", content_hash=content_hash))
    return name


@pytest.mark.ui_browser
@pytest.mark.parametrize("browser_name", ("chromium", "webkit"))
def test_ui_smoke_widget_frame_capabilities(direct_server_with_data, browser_name):
    """Phase 5 end to end on both engines. Positive: (a) both frames carry the
    decided sandbox / allow / allowfullscreen attributes and the module CSP names
    the page origin; (b) a classic sibling `<script src>` and (c) an `import()`
    from the module prefix execute; (d) a `.wasm` from the skill's route
    instantiates through the bridged fetch and its export runs; (e) a `blob:`
    Worker answers; (f) an `<img>` and (g) an `<audio>` from the own prefix load;
    (h) an `@font-face` from the own route with the CORS header loads while (i)
    the same font without the header errors; (j) a user click writes the
    clipboard (chromium, permission granted) and (k) starts a download; (l)
    fullscreen is enabled by policy. Negative: (m) a foreign-origin `<img>`, a
    sibling skill's `<img>` and (n) a raw `XMLHttpRequest` to a foreign origin
    are refused by the document policy with `securitypolicyviolation` events
    naming `img-src` / `connect-src`. (o) The `kind: iframe` route page runs its
    script, reaches its own skill's route, and has no SPA cookie or parent DOM."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    data_dir = direct_server_with_data["data_dir"]
    skill = _write_capability_widget_extension(data_dir)
    prefix = f"/api/extensions/{skill}/"
    page_errors: list[str] = []

    def card(tab_id: str) -> str:
        return f'[data-widget-key="{skill}:{tab_id}"]'

    def probe(frame, call: str, *args):
        return frame.evaluate(f"(args) => window.__capProbe.{call}(...args)", list(args))

    def hits(page) -> dict:
        return page.evaluate("async (skill) => (await fetch(`/api/extensions/${skill}/hits`)).json()", skill)

    try:
        with sync_playwright() as pw:
            browser = getattr(pw, browser_name).launch(headless=True)
            context = browser.new_context(viewport={"width": 1440, "height": 1000}, accept_downloads=True)
            # Chromium grants `clipboard-write` (its sanitized-write permission);
            # WebKit knows no such grant and rejects it lazily when the first page
            # opens, so the fallback is a fresh context without the grant.
            clipboard_granted = True
            try:
                context.grant_permissions(["clipboard-write"], origin=url)
                page = context.new_page()
            except PlaywrightError as exc:
                if "Unknown permission" not in str(exc):
                    raise
                clipboard_granted = False
                context.close()
                context = browser.new_context(viewport={"width": 1440, "height": 1000}, accept_downloads=True)
                page = context.new_page()
            page.on("pageerror", lambda error: page_errors.append(str(error)))
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                toggled = page.evaluate(
                    """async (skill) => (await fetch(`/api/skills/${encodeURIComponent(skill)}/toggle`, {
                        method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({enabled: true}),
                    })).status""",
                    skill,
                )
                assert toggled == 200
                page.click('[data-nav-page="widgets"]')
                for tab_id in ("probe", "page"):
                    page.locator(card(tab_id)).wait_for(state="visible", timeout=30_000)
                    page.wait_for_function("(selector) => document.querySelector(`${selector} iframe`) !== null", arg=card(tab_id), timeout=10_000)

                # (a) The decided attribute set on BOTH frames; the module CSP is absolute.
                for tab_id in ("probe", "page"):
                    frame_node = page.locator(f"{card(tab_id)} iframe")
                    assert frame_node.get_attribute("sandbox") == "allow-scripts allow-pointer-lock allow-downloads", tab_id
                    assert frame_node.get_attribute("allow") == "autoplay; fullscreen; clipboard-write", tab_id
                    assert frame_node.get_attribute("allowfullscreen") == "", tab_id
                assert page.locator(f"{card('page')} iframe").get_attribute("src") == f"{prefix}page"
                frame = page.locator(f"{card('probe')} iframe").element_handle().content_frame()
                frame.wait_for_function("() => Boolean(window.__capProbe)", timeout=10_000)
                csp = probe(frame, "csp")
                assert csp == "; ".join([
                    "default-src 'none'",
                    f"script-src 'unsafe-inline' 'wasm-unsafe-eval' blob: {url}{prefix}module/",
                    "worker-src blob:",
                    "style-src 'unsafe-inline'",
                    f"img-src data: blob: {url}{prefix}",
                    f"media-src data: blob: {url}{prefix}",
                    f"font-src data: blob: {url}{prefix}",
                ]), csp
                assert probe(frame, "origin")["origin"] == "null", "the module frame must stay an opaque origin"

                # (b) classic sibling script, (c) ES module sibling — both from the module prefix.
                assert probe(frame, "classicScript") == "classic"
                assert probe(frame, "esmImport") == "esm"

                # (d) WebAssembly from the skill's own route through the bridged fetch.
                wasm = probe(frame, "wasm")
                assert wasm["sum"] == 42, wasm
                assert wasm["contentType"].startswith("application/wasm"), wasm
                streaming = probe(frame, "wasmStreaming")
                assert streaming == {"sum": 3, "error": None}, streaming

                # (e) a blob: Worker.
                assert probe(frame, "blobWorker") == 42

                # (e2) Final-gate finding WL-01: the bridge checks the owning prefix
                # once, before the request. A skill route that redirects out of that
                # prefix must NOT be followed, or the parent would replay the frame's
                # method, body, headers and the owner's session against whatever the
                # hop names — another skill's routes, or any authenticated host API.
                redirected = probe(frame, "redirected")
                assert redirected["followed"] is False, redirected
                counted = hits(page)
                assert counted.get("elsewhere") == 1, counted

                # (f) image and (g) audio from the own route prefix (passive loads).
                assert probe(frame, "ownImage") == {"ok": True, "width": 1}
                audio = probe(frame, "audio")
                assert audio["loaded"] == "loadedmetadata", audio
                assert audio["duration"] > 0, audio

                # (h) @font-face with the CORS header loads; (i) without it, Chromium
                # refuses the font (fonts are CORS-mode fetches from the opaque
                # origin) — the documented interlock; WebKit loads it anyway.
                assert probe(frame, "font", "font.woff2", "ProbeFont") == "loaded"
                font_nocors = probe(frame, "font", "font-nocors.woff2", "ProbeFontNoCors")
                assert font_nocors == ("error" if browser_name == "chromium" else "loaded"), font_nocors

                # (m) foreign origin and a sibling skill's prefix are refused by the document policy.
                foreign = probe(frame, "foreignImage")
                assert foreign["ok"] is False, foreign
                assert [v["directive"] for v in foreign["violations"]] == ["img-src"], foreign
                sibling = probe(frame, "siblingImage")
                assert sibling["ok"] is False, sibling
                assert [v["directive"] for v in sibling["violations"]] == ["img-src"], sibling
                # (n) no scriptable network of its own: a raw XHR falls to default-src 'none'.
                xhr = probe(frame, "foreignXhr")
                assert xhr["outcome"] == "error" and xhr["status"] == 0, xhr
                assert [v["directive"] for v in xhr["violations"]] == ["connect-src"], xhr

                # (l) fullscreen is enabled for the frame by policy (chromium exposes the flag).
                fullscreen = probe(frame, "fullscreen")
                if browser_name == "chromium":
                    assert fullscreen["enabled"] is True, fullscreen

                # (j) clipboard write from a user click; (k) a download from a user click.
                frame.locator("#copy").click()
                frame.wait_for_function("() => window.__clipboard !== undefined", timeout=5_000)
                clipboard = probe(frame, "clipboard")
                if clipboard_granted:
                    assert clipboard == "ok", clipboard
                else:
                    assert isinstance(clipboard, str), clipboard
                with page.expect_download(timeout=10_000) as download_info:
                    frame.locator("#dl").click()
                assert download_info.value.suggested_filename == "probe.txt"
                frame.locator("#lock").click()
                page.wait_for_timeout(500)
                pointer_lock = probe(frame, "pointerLock")

                # The server saw every own-prefix load, the CORS-less font included.
                seen = hits(page)
                for name in ("add.wasm", "pixel.png", "beep.wav", "font.woff2", "font-nocors.woff2", "page", "ping:route", "plain"):
                    assert seen.get(name, 0) >= 1, (name, seen)

                # (o) The route page's script ran with network reach and no SPA cookie / DOM.
                # Its origin is opaque, so its own fetches are cross-origin: a route
                # answering with the CORS header is readable, one without is not
                # (TypeError) unless requested `no-cors` (an opaque response).
                route_frame = page.locator(f"{card('page')} iframe").element_handle().content_frame()
                route_frame.wait_for_function("() => Boolean(window.__routeProbe && window.__routeProbe.done)", timeout=10_000)
                route = route_frame.evaluate("() => ({...window.__routeProbe, title: document.title})")
                assert route["ran"] is True and route["title"] == "route-ran", route
                assert route["ping"] == 200, route
                assert route["plain"] == "TypeError", route
                assert route["opaque"] == "opaque", route
                assert route["cookie"] in ("SecurityError", "readable:"), route
                assert route["parentDom"] == "SecurityError", route

                _click_nav(page, "dashboard")
                page.wait_for_function("() => document.querySelectorAll('#widgets-list iframe').length === 0", timeout=5_000)
                # Two engine-reported messages are expected by design: WebKit logs
                # the deliberate CORS-less `fetch` negative as a page error, and it
                # fails the Fullscreen permission-policy check for the opaque-origin
                # frame (WebKit has no opaque-`'src'` matching; fullscreen is a
                # Chromium capability there — disclosed in CREATING_SKILLS).
                unexpected = [
                    error for error in page_errors
                    if "due to access control checks" not in error and "Permission policy 'Fullscreen'" not in error
                ]
                assert unexpected == [], page_errors
                if browser_name == "webkit":
                    assert fullscreen["enabled"] is False, fullscreen
                    assert any("Permission policy 'Fullscreen'" in error for error in page_errors), page_errors
                print(f"[{browser_name}] clipboard={clipboard!r} granted={clipboard_granted} pointer_lock={pointer_lock!r} "
                      f"audio_play={audio['play']!r} fullscreen={fullscreen} font_nocors={font_nocors!r} page_errors={page_errors!r}")
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise
