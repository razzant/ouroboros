"""Widgets lifecycle phase 2–4 browser smoke: launch policy (auto / manual /
owner override), the ordered dispose → acknowledgement handshake, session-local
Stop suppression, force-stop on skill disable, the ``retain`` keep-alive
(frame identity and progress across pages, honest badge, the window reload as
the only hard reset, reorder without a reload, hidden force-stop) and the one streaming module
bridge (binary bodies byte-identical, in-process streaming observed chunk by
chunk, abort, opt-in timeout, skill WebSocket events, prefix refusal, null
bodies, dispose with an open stream) on chromium and webkit. Kept apart from
``test_widgets_ui_browser.py`` (geometry / job retry) so neither file grows past
the size-ratchet band."""

from __future__ import annotations

import json
import os
import pathlib
import textwrap

import pytest

from tests.test_ui_smoke_playwright import direct_server_with_data as _direct_server_with_data

direct_server_with_data = _direct_server_with_data


def _write_lifecycle_widget_extension(data_dir: pathlib.Path) -> str:
    """Install the launch-policy / ordered-stop fixture: a `manual` program (kind
    default), an `auto` instrument whose async dispose hook flushes through the
    bridge, an `auto` card whose hook never resolves, and a declarative gauge."""
    from ouroboros.skill_loader import SkillReviewState, compute_content_hash, save_review_state

    name = "lifecycle_widget_smoke"
    skill_dir = data_dir / "skills" / "external" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        textwrap.dedent(
            f"""\
            ---
            name: {name}
            description: Isolated widget launch-policy and ordered-stop fixture.
            version: 0.1.0
            type: extension
            entry: plugin.py
            permissions: ["route", "widget"]
            ---
            # Widget lifecycle fixture
            """
        ),
        encoding="utf-8",
    )
    (skill_dir / "plugin.py").write_text(
        textwrap.dedent(
            """\
            async def ping(_request):
                return {"ok": True}


            async def flush(_request):
                return {"ok": True}


            def register(api):
                api.register_route("ping", ping, methods=("GET",))
                api.register_route("flush", flush, methods=("POST",))
                program = {"kind": "module", "entry": "widget.js", "height": 360}
                api.register_ui_tab("manual", "Manual program", render=program)
                api.register_ui_tab("auto", "Auto instrument", render={**program, "start": "auto"})
                api.register_ui_tab("hang", "Hanging hook", render={"kind": "module", "entry": "hang.js", "height": 360, "start": "auto"})
                api.register_ui_tab(
                    "gauge",
                    "Gauge",
                    render={"kind": "declarative", "schema_version": 1, "components": [{"type": "markdown", "text": "gauge"}]},
                )
            """
        ),
        encoding="utf-8",
    )
    (skill_dir / "widget.js").write_text(
        textwrap.dedent(
            f"""\
            (() => {{
                document.getElementById('root').textContent = 'Program running';
                // Async dispose hook: flush through the bridge, then prove the
                // answer arrived by issuing a second request that carries it.
                window.__ouroWidgetOnDispose(async () => {{
                    const saved = await fetch('/api/extensions/{name}/flush', {{
                        method: 'POST',
                        headers: {{'Content-Type': 'application/json'}},
                        body: JSON.stringify({{state: 'saved'}}),
                    }});
                    await fetch('/api/extensions/{name}/ping?flushed=' + saved.status);
                }});
            }})();
            """
        ),
        encoding="utf-8",
    )
    (skill_dir / "hang.js").write_text(
        textwrap.dedent(
            """\
            (() => {
                document.getElementById('root').textContent = 'Never acknowledges';
                window.__ouroWidgetOnDispose(() => new Promise(() => {}));
            })();
            """
        ),
        encoding="utf-8",
    )
    content_hash = compute_content_hash(skill_dir, manifest_entry="plugin.py")
    save_review_state(data_dir, name, SkillReviewState(status="pass", content_hash=content_hash))
    return name


def _write_retain_widget_extension(data_dir: pathlib.Path) -> str:
    """Install the keep-alive fixture: a `retain` program whose child advances a
    `setInterval` counter (each tick also talks through the bridge) and a
    `requestAnimationFrame` counter, an `auto` instrument for contrast, and a
    declarative gauge with an auto-started poll."""
    from ouroboros.skill_loader import SkillReviewState, compute_content_hash, save_review_state

    name = "retain_widget_smoke"
    skill_dir = data_dir / "skills" / "external" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        textwrap.dedent(
            f"""\
            ---
            name: {name}
            description: Isolated widget keep-alive fixture.
            version: 0.1.0
            type: extension
            entry: plugin.py
            permissions: ["route", "widget"]
            ---
            # Widget keep-alive fixture
            """
        ),
        encoding="utf-8",
    )
    (skill_dir / "plugin.py").write_text(
        textwrap.dedent(
            """\
            async def ping(_request):
                return {"ok": True}


            async def gauge(_request):
                return {"value": 1}


            def register(api):
                api.register_route("ping", ping, methods=("GET",))
                api.register_route("gauge", gauge, methods=("GET",))
                api.register_ui_tab("kept", "Kept program", render={"kind": "module", "entry": "kept.js", "height": 360, "start": "retain"})
                api.register_ui_tab("auto", "Auto instrument", render={"kind": "module", "entry": "auto.js", "height": 360, "start": "auto"})
                api.register_ui_tab(
                    "gauge",
                    "Gauge",
                    render={
                        "kind": "declarative",
                        "schema_version": 1,
                        "components": [
                            {"type": "poll", "id": "gauge-poll", "label": "Poll", "route": "gauge", "interval_ms": 1000, "max_ticks": 100, "auto_start": True},
                            {"type": "kv", "fields": [{"label": "Value", "path": "value"}]},
                        ],
                    },
                )
            """
        ),
        encoding="utf-8",
    )
    (skill_dir / "kept.js").write_text(
        textwrap.dedent(
            f"""\
            (() => {{
                document.getElementById('root').textContent = 'Kept running';
                const counters = {{ ticks: 0, frames: 0 }};
                window.__keptCounters = counters;
                setInterval(() => {{
                    counters.ticks += 1;
                    fetch('/api/extensions/{name}/ping?tick=' + counters.ticks).catch(() => {{}});
                }}, 250);
                const loop = () => {{
                    counters.frames += 1;
                    requestAnimationFrame(loop);
                }};
                requestAnimationFrame(loop);
            }})();
            """
        ),
        encoding="utf-8",
    )
    (skill_dir / "auto.js").write_text(
        "(() => { document.getElementById('root').textContent = 'Instrument'; })();\n",
        encoding="utf-8",
    )
    content_hash = compute_content_hash(skill_dir, manifest_entry="plugin.py")
    save_review_state(data_dir, name, SkillReviewState(status="pass", content_hash=content_hash))
    return name


_BRIDGE_PLUGIN = """\
import asyncio
import os

from starlette.responses import Response, StreamingResponse

BLOB = os.urandom(65536)
STATE = {"slow_started": 0, "slow_closed": 0, "emitted": 0}
HOST = {}


def fnv1a32(data: bytes) -> int:
    value = 0x811C9DC5
    for byte in data:
        value = ((value ^ byte) * 0x01000193) & 0xFFFFFFFF
    return value


async def blob(_request):
    return Response(
        BLOB,
        media_type="application/octet-stream",
        headers={"x-blob-fnv": str(fnv1a32(BLOB)), "x-blob-len": str(len(BLOB))},
    )


async def stream(_request):
    async def chunks():
        for index in range(3):
            yield f"chunk-{index}\\n".encode()
            await asyncio.sleep(0.25)
    return StreamingResponse(chunks(), media_type="text/plain")


async def slow(_request):
    async def ticks():
        STATE["slow_started"] += 1
        try:
            while True:
                yield b"tick\\n"
                await asyncio.sleep(0.2)
        finally:
            STATE["slow_closed"] += 1
    return StreamingResponse(ticks(), media_type="text/plain")


async def state(_request):
    return dict(STATE)


async def nobody(_request):
    return Response(status_code=204)


async def ping(_request):
    return {"ok": True}


async def emit(request):
    payload = await request.json()
    STATE["emitted"] += 1
    HOST["api"].send_ws_message("tick", {"n": STATE["emitted"], "note": str(payload.get("note") or "")})
    return {"ok": True, "n": STATE["emitted"]}


def register(api):
    HOST["api"] = api
    for name, handler in (("blob", blob), ("stream", stream), ("slow", slow), ("state", state), ("nobody", nobody), ("ping", ping)):
        api.register_route(name, handler, methods=("GET",))
    api.register_route("emit", emit, methods=("POST",))
    api.register_ui_tab("probe", "Bridge probe", render={"kind": "module", "entry": "probe.js", "height": 360, "start": "auto"})
"""

# Child-side probe: every call goes through the injected bridge (`fetch` /
# `OuroborosWidget`), so what the test observes is the grammar end to end.
_BRIDGE_PROBE_JS = """\
(() => {
    document.getElementById('root').textContent = 'Bridge probe';
    const base = '/api/extensions/__SKILL__/';
    const fnv = (bytes) => bytes.reduce((value, byte) => Math.imul(value ^ byte, 0x01000193) >>> 0, 0x811c9dc5);
    const decoder = new TextDecoder();
    const readAll = async (reader) => {
        const chunks = [];
        for (;;) {
            const { done, value } = await reader.read();
            if (done) return chunks;
            chunks.push(decoder.decode(value));
        }
    };
    const events = [];
    let unsubscribe = null;
    window.__bridgeProbe = {
        async binary() {
            const r = await OuroborosWidget.fetch(base + 'blob');
            const buffer = await r.arrayBuffer();
            return { status: r.status, statusText: r.statusText, length: buffer.byteLength, fnv: fnv(new Uint8Array(buffer)),
                headerFnv: r.headers.get('x-blob-fnv'), headerLen: r.headers.get('x-blob-len'), contentType: r.headers.get('content-type') };
        },
        async stream() {
            const r = await fetch(base + 'stream');
            return { status: r.status, chunks: await readAll(r.body.getReader()) };
        },
        async abort() {
            const controller = new AbortController();
            const r = await fetch(base + 'slow', { signal: controller.signal });
            const reader = r.body.getReader();
            const first = decoder.decode((await reader.read()).value);
            controller.abort();
            try { await reader.read(); return { first, error: null }; } catch (err) { return { first, error: err.name }; }
        },
        async timeout() {
            const r = await fetch(base + 'slow', { timeoutMs: 400 });
            try { await readAll(r.body.getReader()); return { error: null }; } catch (err) { return { error: err.message }; }
        },
        async nobody() {
            const r = await fetch(base + 'nobody');
            const head = await fetch(base + 'ping', { method: 'HEAD' });
            return { status: r.status, nullBody: r.body === null, text: await r.text(), headStatus: head.status, headNullBody: head.body === null };
        },
        async outside() {
            const results = {};
            for (const [key, url] of [['sibling', '/api/extensions/other_skill/ping'], ['host', '/api/widgets'], ['absolute', 'https://example.com/api/extensions/__SKILL__/ping']]) {
                try { await fetch(url); results[key] = 'resolved'; } catch (err) { results[key] = err.message; }
            }
            return results;
        },
        subscribe() { events.length = 0; unsubscribe = OuroborosWidget.onEvent((event) => events.push(event)); return true; },
        unsubscribe() { unsubscribe?.(); unsubscribe = null; return true; },
        events() { return events.slice(); },
        async emit(note) {
            const r = await fetch(base + 'emit', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ note }) });
            return r.json();
        },
        async openSlow() {
            const r = await fetch(base + 'slow');
            const reader = r.body.getReader();
            window.__slowState = 'open';
            readAll(reader).then(() => { window.__slowState = 'ended'; }, (err) => { window.__slowState = 'errored: ' + err.message; });
            return true;
        },
    };
})();
"""


def _write_bridge_widget_extension(data_dir: pathlib.Path) -> str:
    """Install the streaming-bridge fixture: an `auto` module probe plus routes for
    a 64 KiB random blob (with its FNV-1a checksum in a header), a three-chunk
    in-process `StreamingResponse`, an endless slow stream that records its own
    teardown, a 204 route, and a POST that emits a namespaced WS event."""
    from ouroboros.skill_loader import SkillReviewState, compute_content_hash, save_review_state

    name = "bridge_widget_smoke"
    skill_dir = data_dir / "skills" / "external" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        textwrap.dedent(
            f"""\
            ---
            name: {name}
            description: Isolated module bridge streaming fixture.
            version: 0.1.0
            type: extension
            entry: plugin.py
            permissions: ["route", "widget", "ws_handler"]
            ---
            # Widget bridge fixture
            """
        ),
        encoding="utf-8",
    )
    (skill_dir / "plugin.py").write_text(_BRIDGE_PLUGIN, encoding="utf-8")
    (skill_dir / "probe.js").write_text(_BRIDGE_PROBE_JS.replace("__SKILL__", name), encoding="utf-8")
    content_hash = compute_content_hash(skill_dir, manifest_entry="plugin.py")
    save_review_state(data_dir, name, SkillReviewState(status="pass", content_hash=content_hash))
    return name


# Parent-side probe: every host fetch to an extension route is logged with the
# number of widget frames still attached and whether Widgets is the active page
# at that moment. Bridged requests (a dispose hook's flush, a kept frame's
# ticks) go through the parent's `fetch`, so this observes the ordered stop and
# the keep-alive from the host's side.
_HOST_FETCH_PROBE_SCRIPT = r"""
(() => {
    const log = [];
    window.__hostFetchLog = log;
    const original = window.fetch.bind(window);
    window.fetch = (input, init) => {
        const url = typeof input === 'string' ? input : String(input && input.url || input);
        if (url.includes('/api/extensions/')) {
            log.push({
                url,
                t: performance.now(),
                frames: document.querySelectorAll('#widgets-list iframe').length,
                widgetsActive: Boolean(document.getElementById('page-widgets')?.classList.contains('active')),
            });
        }
        return original(input, init);
    };
})();
"""


def _click_nav(page, target: str) -> None:
    page.evaluate(
        """(target) => {
            const button = [...document.querySelectorAll(`[data-nav-page="${target}"]`)]
                .find((item) => getComputedStyle(item).display !== 'none');
            button?.click();
        }""",
        target,
    )


@pytest.mark.ui_browser
@pytest.mark.parametrize("browser_name", ("chromium", "webkit"))
def test_ui_smoke_widget_launch_policy_and_ordered_stop(direct_server_with_data, browser_name):
    """Widgets lifecycle phase 2 end to end, on both engines: manual facade until
    Start, auto mounts on show, ordered dispose with acknowledgement (async hook
    flushes through the bridge before the frame goes; a hook that never resolves
    is cut after ~1 s without delaying the page switch), one frame per key across
    a leave/return race, session-local Stop suppression, owner override over the
    author default (menu and API), and force-stop + removal on skill disable."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    data_dir = direct_server_with_data["data_dir"]
    skill = _write_lifecycle_widget_extension(data_dir)
    evidence_dir = pathlib.Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", str(data_dir.parent)))
    evidence_dir.mkdir(parents=True, exist_ok=True)

    def card(tab_id: str) -> str:
        return f'[data-widget-key="{skill}:{tab_id}"]'

    def frame_count(page, tab_id: str) -> int:
        return page.locator(f"{card(tab_id)} iframe").count()

    def wait_frame(page, tab_id: str, present: bool, timeout: int = 10_000) -> None:
        page.wait_for_function(
            "([selector, present]) => (document.querySelector(`${selector} iframe`) !== null) === present",
            arg=[card(tab_id), present],
            timeout=timeout,
        )

    try:
        with sync_playwright() as pw:
            browser = getattr(pw, browser_name).launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            page.add_init_script(_HOST_FETCH_PROBE_SCRIPT)
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                toggled = page.evaluate(
                    """async (skill) => {
                        const response = await fetch(`/api/skills/${encodeURIComponent(skill)}/toggle`, {
                            method: 'POST', headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({enabled: true}),
                        });
                        return {status: response.status, body: await response.json()};
                    }""",
                    skill,
                )
                assert toggled["status"] == 200, toggled
                page.click('[data-nav-page="widgets"]')
                for tab_id in ("manual", "auto", "hang", "gauge"):
                    page.locator(card(tab_id)).wait_for(state="visible", timeout=30_000)

                # auto → mounts on show (the `start:"auto"` survival path); manual → facade,
                # no frame; declarative → host-drawn, no Start/Stop and no policy menu.
                wait_frame(page, "auto", True)
                wait_frame(page, "hang", True)
                page.locator(f"{card('manual')} [data-widget-facade]").wait_for(state="visible", timeout=10_000)
                assert frame_count(page, "manual") == 0
                assert page.locator(f"{card('manual')} [data-widget-power]").inner_text() == "Start"
                assert page.locator(f"{card('gauge')} [data-widget-power]").count() == 0
                assert page.locator(f"{card('gauge')} [data-widget-menu-trigger]").count() == 0
                page.wait_for_function(
                    "(selector) => document.querySelector(`${selector} [data-widget-power]`)?.textContent === 'Stop'",
                    arg=card("auto"),
                    timeout=10_000,
                )
                assert page.locator(f"{card('auto')} [data-widget-status]").inner_text() == "Running"
                facade_height = page.locator(f"{card('manual')} [data-widget-facade]").evaluate(
                    "node => node.getBoundingClientRect().height"
                )
                assert facade_height == 360, facade_height
                page.screenshot(path=str(evidence_dir / f"widget-lifecycle-cards-{browser_name}.png"), full_page=True)
                page.locator(f"{card('manual')} [data-widget-menu-trigger]").click()
                page.locator(f"{card('manual')} [data-widget-start-mode=\"manual\"]").wait_for(state="visible", timeout=5_000)
                page.screenshot(path=str(evidence_dir / f"widget-lifecycle-menu-{browser_name}.png"), full_page=True)
                page.keyboard.press("Escape")
                page.locator(f"{card('manual')} [data-widget-start-mode=\"manual\"]").wait_for(state="hidden", timeout=5_000)

                # Leave: the page switches at once while both frames still stand in their
                # cards; the auto card's async hook flushes through the bridge and gets its
                # answer before the frame goes; the hanging hook is cut after ~1 s.
                page.evaluate("window.__hostFetchLog.length = 0")
                left_at = page.evaluate("performance.now()")
                _click_nav(page, "dashboard")
                page.wait_for_function(
                    "() => !document.getElementById('page-widgets').classList.contains('active')",
                    timeout=2_000,
                )
                assert frame_count(page, "hang") == 1, "the page switch must not wait for the acknowledgement"
                wait_frame(page, "auto", False, timeout=5_000)
                wait_frame(page, "hang", False, timeout=5_000)
                hang_gone_at = page.evaluate("performance.now()")
                assert 700 <= hang_gone_at - left_at <= 4_000, hang_gone_at - left_at
                fetch_log = page.evaluate("window.__hostFetchLog")
                flush = [row for row in fetch_log if row["url"].endswith(f"/api/extensions/{skill}/flush")]
                answered = [row for row in fetch_log if f"/api/extensions/{skill}/ping?flushed=200" in row["url"]]
                assert len(flush) == 1, fetch_log
                assert flush[0]["frames"] >= 1, fetch_log
                assert flush[0]["widgetsActive"] is False, fetch_log
                assert len(answered) == 1, fetch_log
                assert answered[0]["frames"] >= 1, fetch_log

                # Return: auto cards remount (one frame each — the hang card's remount waited
                # for its pending stop), the manual card stays a facade.
                _click_nav(page, "widgets")
                wait_frame(page, "auto", True)
                wait_frame(page, "hang", True)
                page.wait_for_timeout(300)
                assert frame_count(page, "auto") == 1
                assert frame_count(page, "hang") == 1
                assert frame_count(page, "manual") == 0

                # Leave and return within the acknowledgement window: the remount waits
                # for the pending stop, so the card never holds two frames and ends with
                # one fresh frame (the opaque-origin document is unreadable from here, so
                # freshness is an expando on the old node and a visible child root).
                page.evaluate(
                    """(selector) => {
                        const cardNode = document.querySelector(selector);
                        cardNode.querySelector('iframe').__ouroOldFrame = true;
                        window.__maxHangFrames = 1;
                        new MutationObserver(() => {
                            window.__maxHangFrames = Math.max(window.__maxHangFrames, cardNode.querySelectorAll('iframe').length);
                        }).observe(cardNode, {subtree: true, childList: true});
                    }""",
                    card("hang"),
                )
                _click_nav(page, "dashboard")
                page.wait_for_timeout(150)
                assert frame_count(page, "hang") == 1
                _click_nav(page, "widgets")
                page.wait_for_function(
                    """(selector) => {
                        const frames = document.querySelectorAll(`${selector} iframe`);
                        return frames.length === 1 && frames[0].__ouroOldFrame !== true;
                    }""",
                    arg=card("hang"),
                    timeout=10_000,
                )
                page.frame_locator(f"{card('hang')} iframe").locator("#root").wait_for(state="visible", timeout=10_000)
                page.wait_for_timeout(300)
                assert frame_count(page, "hang") == 1
                assert page.evaluate("window.__maxHangFrames") == 1

                # Owner Stop → facade; the Stop is remembered across leave/return until Start.
                page.locator(f"{card('auto')} [data-widget-power]").click()
                wait_frame(page, "auto", False)
                page.locator(f"{card('auto')} [data-widget-facade]").wait_for(state="visible", timeout=5_000)
                assert page.locator(f"{card('auto')} [data-widget-power]").inner_text() == "Start"
                _click_nav(page, "dashboard")
                wait_frame(page, "hang", False, timeout=5_000)
                _click_nav(page, "widgets")
                wait_frame(page, "hang", True)
                page.wait_for_timeout(300)
                assert frame_count(page, "auto") == 0, "an owner-stopped auto card must not restart on return"
                page.locator(f"{card('auto')} [data-widget-power]").click()
                wait_frame(page, "auto", True)

                # Owner override from the card menu: Manual program → Auto starts it now and
                # persists the whole map through the preferences API.
                page.locator(f"{card('manual')} [data-widget-menu-trigger]").click()
                page.locator(f"{card('manual')} [data-widget-start-mode=\"auto\"]").click()
                wait_frame(page, "manual", True)
                prefs = page.evaluate("async () => (await fetch('/api/ui/preferences')).json()")
                assert prefs["widget_start_mode"] == {f"{skill}:manual": "auto"}, prefs
                assert page.locator(f"{card('manual')} [data-widget-start-mode=\"auto\"]").get_attribute("aria-checked") == "true"

                # Owner override through the API beats the author default in both directions
                # on the next Widgets entry: the author-auto card waits, the author-manual card runs.
                saved = page.evaluate(
                    """async (payload) => (await fetch('/api/ui/preferences', {
                        method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload),
                    })).status""",
                    {"widget_start_mode": {f"{skill}:manual": "auto", f"{skill}:auto": "manual"}},
                )
                assert saved == 200
                _click_nav(page, "dashboard")
                page.wait_for_timeout(300)
                _click_nav(page, "widgets")
                wait_frame(page, "hang", True, timeout=15_000)
                wait_frame(page, "manual", True, timeout=15_000)
                page.locator(f"{card('auto')} [data-widget-facade]").wait_for(state="visible", timeout=10_000)
                assert frame_count(page, "auto") == 0
                assert page.locator(f"{card('auto')} [data-widget-start-mode=\"manual\"]").get_attribute("aria-checked") == "true"

                # Disabling the skill while its cards run force-stops them in order and
                # removes every card of that skill.
                disabled = page.evaluate(
                    """async (skill) => (await fetch(`/api/skills/${encodeURIComponent(skill)}/toggle`, {
                        method: 'POST', headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({enabled: false}),
                    })).status""",
                    skill,
                )
                assert disabled == 200
                page.wait_for_function(
                    "(prefix) => document.querySelectorAll(`[data-widget-key^=\"${prefix}\"]`).length === 0",
                    arg=f"{skill}:",
                    timeout=20_000,
                )
                assert page.locator("#widgets-list iframe").count() == 0
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise


@pytest.mark.ui_browser
@pytest.mark.parametrize("browser_name", ("chromium", "webkit"))
def test_ui_smoke_widget_retain_keeps_running_across_pages(direct_server_with_data, browser_name):
    """Widgets lifecycle phase 3 end to end, on both engines. A `retain` card
    starts on the first visit like `auto` and says "Keeps running"; leaving the
    page stops the `auto` frame but keeps the retained frame mounted — same
    iframe node, same child window, its `setInterval` counter advancing while
    hidden (the `requestAnimationFrame` counter is asserted advanced on webkit
    only: Chromium pauses animation frames of a hidden frame, no rate is
    promised) and its bridged ticks still reaching the host while the
    declarative poll issues nothing. A keyboard reorder changes the visible
    position without moving the node or reloading the frame. The page carries
    no Refresh control: the window reload is the only hard reset, it ends the
    kept frame with its window, and it forgets an owner Stop (which lives in
    the page session only). Owner Stop frees the frame and its timers.
    Disabling the skill while Widgets is hidden force-stops the kept frame
    before the next visit."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    data_dir = direct_server_with_data["data_dir"]
    skill = _write_retain_widget_extension(data_dir)
    evidence_dir = pathlib.Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", str(data_dir.parent)))
    evidence_dir.mkdir(parents=True, exist_ok=True)

    def card(tab_id: str) -> str:
        return f'[data-widget-key="{skill}:{tab_id}"]'

    def frame_count(page, tab_id: str) -> int:
        return page.locator(f"{card(tab_id)} iframe").count()

    def wait_frame(page, tab_id: str, present: bool, timeout: int = 10_000) -> None:
        page.wait_for_function(
            "([selector, present]) => (document.querySelector(`${selector} iframe`) !== null) === present",
            arg=[card(tab_id), present],
            timeout=timeout,
        )

    def wait_status(page, tab_id: str, text: str, timeout: int = 10_000) -> None:
        page.wait_for_function(
            "([selector, text]) => document.querySelector(`${selector} [data-widget-status]`)?.textContent === text",
            arg=[card(tab_id), text],
            timeout=timeout,
        )

    def wait_active(page, active: bool) -> None:
        page.wait_for_function(
            "(active) => document.getElementById('page-widgets').classList.contains('active') === active",
            arg=active,
            timeout=5_000,
        )

    def kept_frame(page):
        return page.locator(f"{card('kept')} iframe").element_handle().content_frame()

    def counters(page) -> dict:
        return kept_frame(page).evaluate("() => ({...window.__keptCounters})")

    def same_frame(page) -> bool:
        return page.evaluate(
            "(selector) => document.querySelector(`${selector} iframe`)?.__ouroKeptFrame === true",
            card("kept"),
        )

    def masonry_x(page, tab_id: str) -> str:
        return page.locator(card(tab_id)).evaluate("node => node.style.getPropertyValue('--masonry-x')")

    def dom_index(page, tab_id: str) -> int:
        return page.evaluate(
            "(selector) => [...document.querySelectorAll('#widgets-list .widgets-card')].indexOf(document.querySelector(selector))",
            card(tab_id),
        )

    def toggle(page, enabled: bool) -> int:
        return page.evaluate(
            """async ([skill, enabled]) => (await fetch(`/api/skills/${encodeURIComponent(skill)}/toggle`, {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({enabled}),
            })).status""",
            [skill, enabled],
        )

    try:
        with sync_playwright() as pw:
            browser = getattr(pw, browser_name).launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            page.add_init_script(_HOST_FETCH_PROBE_SCRIPT)
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                assert toggle(page, True) == 200
                page.click('[data-nav-page="widgets"]')
                for tab_id in ("kept", "auto", "gauge"):
                    page.locator(card(tab_id)).wait_for(state="visible", timeout=30_000)

                # First visit starts the kept card like auto; its badge is honest.
                wait_frame(page, "kept", True)
                wait_frame(page, "auto", True)
                wait_status(page, "kept", "Keeps running")
                wait_status(page, "auto", "Running")
                assert page.locator(f"{card('kept')} [data-widget-power]").inner_text() == "Stop"
                kept_frame(page).wait_for_function("() => window.__keptCounters.ticks >= 2", timeout=5_000)
                kept_frame(page).evaluate("() => { window.__keptMark = 'same-window'; }")
                page.evaluate("(selector) => { document.querySelector(`${selector} iframe`).__ouroKeptFrame = true; }", card("kept"))
                page.screenshot(path=str(evidence_dir / f"widget-retain-{browser_name}.png"), full_page=True)

                # Leave: the auto frame goes, the kept frame stays and keeps working.
                before = counters(page)
                page.evaluate("window.__hostFetchLog.length = 0")
                _click_nav(page, "dashboard")
                wait_active(page, False)
                wait_frame(page, "auto", False, timeout=5_000)
                page.wait_for_timeout(1_500)
                assert frame_count(page, "kept") == 1
                assert same_frame(page)
                hidden = counters(page)
                assert hidden["ticks"] > before["ticks"], (browser_name, before, hidden)
                hidden_frames = hidden["frames"] - before["frames"]
                # Chromium pauses requestAnimationFrame in a hidden frame; the
                # observation is recorded as evidence per engine, asserted on webkit only.
                (evidence_dir / f"widget-retain-{browser_name}.json").write_text(
                    json.dumps({"engine": browser_name, "before": before, "hidden": hidden, "hidden_frames": hidden_frames}),
                    encoding="utf-8",
                )
                if browser_name == "webkit":
                    assert hidden_frames > 0, (browser_name, before, hidden)
                fetch_log = page.evaluate("window.__hostFetchLog")
                kept_ticks = [row for row in fetch_log if f"/api/extensions/{skill}/ping?tick=" in row["url"]]
                hidden_polls = [
                    row for row in fetch_log
                    if f"/api/extensions/{skill}/gauge" in row["url"] and row["widgetsActive"] is False
                ]
                assert kept_ticks, fetch_log
                assert all(row["widgetsActive"] is False and row["frames"] >= 1 for row in kept_ticks), fetch_log
                assert hidden_polls == [], "a hidden declarative poll must issue nothing"

                # Return: same node, same window, badge still honest; auto remounts.
                _click_nav(page, "widgets")
                wait_active(page, True)
                wait_frame(page, "auto", True)
                page.wait_for_timeout(300)
                assert frame_count(page, "kept") == 1
                assert same_frame(page)
                assert kept_frame(page).evaluate("() => window.__keptMark") == "same-window"
                assert page.locator(f"{card('kept')} [data-widget-status]").inner_text() == "Keeps running"

                # Keyboard reorder moves the kept card to the other end of the visible
                # order (the list arrives sorted by tab id, so it normally sits last and
                # Home brings it first) while its node stays where it is and its frame
                # never reloads.
                page.wait_for_function(
                    "(selector) => document.querySelector(selector)?.style.getPropertyValue('--masonry-x') !== ''",
                    arg=card("kept"),
                    timeout=5_000,
                )
                x_before = masonry_x(page, "kept")
                key_press = "Home" if x_before != "0px" else "End"
                index_before = dom_index(page, "kept")
                page.locator(f"{card('kept')} [data-widget-reorder-handle]").focus()
                page.keyboard.press(key_press)
                page.wait_for_function(
                    """async ([key, first]) => {
                        const prefs = await (await fetch('/api/ui/preferences')).json();
                        const order = prefs.widget_order || [];
                        return order.length > 0 && order[first ? 0 : order.length - 1] === key;
                    }""",
                    arg=[f"{skill}:kept", key_press == "Home"],
                    timeout=5_000,
                )
                page.wait_for_function(
                    "([selector, before]) => document.querySelector(selector)?.style.getPropertyValue('--masonry-x') !== before",
                    arg=[card("kept"), x_before],
                    timeout=5_000,
                )
                if key_press == "Home":
                    assert masonry_x(page, "kept") == "0px", masonry_x(page, "kept")
                assert dom_index(page, "kept") == index_before, "a reorder must not move the card node"
                assert frame_count(page, "kept") == 1
                assert same_frame(page)
                assert kept_frame(page).evaluate("() => window.__keptMark") == "same-window"
                assert page.evaluate("document.activeElement?.hasAttribute('data-widget-reorder-handle')")

                # Owner decision Q20: the page carries no Refresh control, so nothing in it
                # stops a kept-running program behind the owner's back.
                assert page.locator("#widgets-refresh").count() == 0
                page.screenshot(path=str(evidence_dir / f"widget-retain-no-refresh-{browser_name}.png"), full_page=True)
                assert frame_count(page, "kept") == 1
                assert same_frame(page)
                assert kept_frame(page).evaluate("() => window.__keptMark") == "same-window"

                # The window reload is the hard reset that remains: the kept frame dies
                # with the window and the card starts a fresh one on the next entry.
                page.reload(wait_until="domcontentloaded", timeout=30_000)
                page.click('[data-nav-page="widgets"]')
                page.locator(card("kept")).wait_for(state="visible", timeout=30_000)
                wait_frame(page, "kept", True, timeout=15_000)
                wait_status(page, "kept", "Keeps running", timeout=15_000)
                assert kept_frame(page).evaluate("() => window.__keptMark ?? null") is None
                assert frame_count(page, "kept") == 1

                # Owner Stop frees the frame; nothing of it keeps talking.
                page.locator(f"{card('kept')} [data-widget-power]").click()
                wait_frame(page, "kept", False)
                page.locator(f"{card('kept')} [data-widget-facade]").wait_for(state="visible", timeout=5_000)
                assert page.locator(f"{card('kept')} [data-widget-power]").inner_text() == "Start"
                page.evaluate("window.__hostFetchLog.length = 0")
                page.wait_for_timeout(900)
                late = [row for row in page.evaluate("window.__hostFetchLog") if "ping?tick=" in row["url"]]
                assert late == [], late

                # The owner Stop lives in the page session only: a window reload forgets it
                # and the retain card starts again on the next entry.
                page.reload(wait_until="domcontentloaded", timeout=30_000)
                page.click('[data-nav-page="widgets"]')
                page.locator(card("kept")).wait_for(state="visible", timeout=30_000)
                wait_frame(page, "kept", True, timeout=15_000)
                wait_status(page, "kept", "Keeps running", timeout=15_000)

                # Leave, then disable the skill while Widgets is hidden: the kept frame
                # is force-stopped without a visit; the return finds the cards gone.
                _click_nav(page, "dashboard")
                wait_active(page, False)
                page.wait_for_timeout(300)
                assert frame_count(page, "kept") == 1
                assert toggle(page, False) == 200
                wait_frame(page, "kept", False, timeout=15_000)
                wait_active(page, False)
                _click_nav(page, "widgets")
                page.wait_for_function(
                    "(prefix) => document.querySelectorAll(`[data-widget-key^=\"${prefix}\"]`).length === 0",
                    arg=f"{skill}:",
                    timeout=20_000,
                )
                assert page.locator("#widgets-list iframe").count() == 0
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise


@pytest.mark.ui_browser
@pytest.mark.parametrize("browser_name", ("chromium", "webkit"))
def test_ui_smoke_module_bridge_streams_binary_events_and_abort(direct_server_with_data, browser_name):
    """Widgets lifecycle phase 4 end to end, on both engines, all through the
    child's bridged `fetch` / `OuroborosWidget`: a 64 KiB random body arrives
    byte-identical (FNV-1a in the child equals the server's header) with every
    response header; an in-process `StreamingResponse` is read incrementally —
    at least two separate chunks before `end`; `AbortController` mid-stream
    rejects the reader with `AbortError` and the server sees the disconnect;
    the opt-in `timeoutMs` aborts the same way; HEAD and 204 carry a null body;
    a sibling skill, a host API and an absolute URL are refused; the skill's
    `send_ws_message` reaches `onEvent` as `{type, data}` and stops after
    unsubscribe; leaving the page with a stream open tears everything down
    (server-side generator closed) without a page error."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    data_dir = direct_server_with_data["data_dir"]
    skill = _write_bridge_widget_extension(data_dir)
    card = f'[data-widget-key="{skill}:probe"]'
    page_errors: list[str] = []

    def probe(frame, call: str, *args):
        return frame.evaluate(f"(args) => window.__bridgeProbe.{call}(...args)", list(args))

    def server_state(page) -> dict:
        return page.evaluate("async (skill) => (await fetch(`/api/extensions/${skill}/state`)).json()", skill)

    def wait_closed(page, expected: int) -> None:
        page.wait_for_function(
            "async ([skill, expected]) => (await (await fetch(`/api/extensions/${skill}/state`)).json()).slow_closed >= expected",
            arg=[skill, expected],
            timeout=10_000,
        )

    try:
        with sync_playwright() as pw:
            browser = getattr(pw, browser_name).launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            page.on("pageerror", lambda error: page_errors.append(str(error)))
            page.on("console", lambda message: page_errors.append(message.text) if message.type == "error" and "widget" in message.text.lower() else None)
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
                page.locator(card).wait_for(state="visible", timeout=30_000)
                page.wait_for_function("(selector) => document.querySelector(`${selector} iframe`) !== null", arg=card, timeout=10_000)
                frame = page.locator(f"{card} iframe").element_handle().content_frame()
                frame.wait_for_function("() => Boolean(window.__bridgeProbe)", timeout=10_000)

                # (a) binary, byte-identical, every header forwarded.
                binary = probe(frame, "binary")
                assert binary["status"] == 200 and binary["length"] == 65536, binary
                assert str(binary["fnv"]) == binary["headerFnv"], binary
                assert binary["headerLen"] == "65536", binary
                assert binary["contentType"].startswith("application/octet-stream"), binary
                assert binary["statusText"] in ("OK", ""), binary

                # (b) in-process streaming: separate chunks observed before end.
                stream = probe(frame, "stream")
                assert stream["status"] == 200, stream
                assert len(stream["chunks"]) >= 2, stream
                assert "".join(stream["chunks"]) == "chunk-0\nchunk-1\nchunk-2\n", stream

                # (c) abort mid-stream: the reader rejects, the server generator is closed.
                aborted = probe(frame, "abort")
                assert aborted == {"first": "tick\n", "error": "AbortError"}, aborted
                wait_closed(page, 1)

                # (h) the author's opt-in timeoutMs aborts the same way; no default exists.
                timed_out = probe(frame, "timeout")
                assert timed_out == {"error": "widget request timed out"}, timed_out
                wait_closed(page, 2)

                # (f) null bodies; (e) refusal outside the owning prefix on the new channel.
                assert probe(frame, "nobody") == {"status": 204, "nullBody": True, "text": "", "headStatus": 200, "headNullBody": True}
                outside = probe(frame, "outside")
                assert set(outside) == {"sibling", "host", "absolute"}, outside
                assert all(value == "module widget fetch outside extension route prefix" for value in outside.values()), outside

                # (d) skill WS events through the bridge; unsubscribe stops delivery.
                assert probe(frame, "subscribe") is True
                emitted = probe(frame, "emit", "one")
                assert emitted["ok"] is True, emitted
                frame.wait_for_function("() => window.__bridgeProbe.events().length >= 1", timeout=10_000)
                events = probe(frame, "events")
                assert events == [{"type": "tick", "data": {"n": emitted["n"], "note": "one"}}], events
                assert probe(frame, "unsubscribe") is True
                probe(frame, "emit", "two")
                page.wait_for_timeout(700)
                assert probe(frame, "events") == events, "an unsubscribed frame must receive nothing"

                # (g) dispose while a stream is open: the frame goes, the parent aborts,
                # the server closes its generator, and the page logs no error.
                assert probe(frame, "openSlow") is True
                frame.wait_for_function("() => window.__slowState === 'open'", timeout=5_000)
                before = server_state(page)
                assert before["slow_started"] == before["slow_closed"] + 1, before
                _click_nav(page, "dashboard")
                page.wait_for_function("(selector) => document.querySelector(`${selector} iframe`) === null", arg=card, timeout=5_000)
                wait_closed(page, before["slow_started"])
                assert page.locator("#widgets-list iframe").count() == 0
                assert page_errors == [], page_errors
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise
