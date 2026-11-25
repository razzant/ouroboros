from __future__ import annotations

import json
import os
import pathlib
import re
import textwrap
import time
import urllib.parse

import pytest

from tests.test_ui_smoke_playwright import direct_server_with_data as _direct_server_with_data

direct_server_with_data = _direct_server_with_data


def _write_module_widget_smoke_extension(data_dir: pathlib.Path) -> str:
    """Install reviewed module tabs that exercise geometry, faults, and teardown."""
    from ouroboros.skill_loader import SkillReviewState, compute_content_hash, save_review_state

    name = "module_widget_smoke"
    skill_dir = data_dir / "skills" / "external" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        textwrap.dedent(
            f"""\
            ---
            name: {name}
            description: Isolated module widget geometry smoke.
            version: 0.1.0
            type: extension
            entry: plugin.py
            permissions: ["route", "widget"]
            ---
            # Module widget geometry smoke
            """
        ),
        encoding="utf-8",
    )
    (skill_dir / "plugin.py").write_text(
        textwrap.dedent(
            """\
            async def ping(_request):
                return {"ok": True}


            def register(api):
                # Geometry probes are cheap instruments: `start: "auto"` so they mount on show
                # (framed cards default to `manual` and would wait behind Start otherwise).
                api.register_route("ping", ping, methods=("GET",))
                api.register_ui_tab("auto", "Auto module", render={"kind": "module", "entry": "widget.js", "start": "auto"})
                api.register_ui_tab("fixed", "Fixed module", render={"kind": "module", "entry": "widget.js", "height": 480, "start": "auto"})
                api.register_ui_tab("capped", "Capped module", render={"kind": "module", "entry": "widget.js", "max_height": 640, "start": "auto"})
                api.register_ui_tab("fault", "Fault module", render={"kind": "module", "entry": "fault.js", "max_height": 640, "start": "auto"})
                api.register_ui_tab("small", "Small module", render={"kind": "module", "entry": "small.js", "start": "auto"})
            """
        ),
        encoding="utf-8",
    )
    (skill_dir / "widget.js").write_text(
        textwrap.dedent(
            """\
            (() => {
                const root = document.getElementById('root');
                const style = document.createElement('style');
                style.textContent = 'body{margin:0;padding:12px;height:100vh;box-sizing:border-box;overflow-y:auto;font:14px sans-serif;color:#e8ecf3;background:#111}#root{display:block}.row{padding:8px 0;border-bottom:1px solid #445}button{margin:8px 0;padding:6px 10px}';
                root.appendChild(style);
                const heading = document.createElement('h2');
                heading.textContent = 'Module geometry probe';
                root.appendChild(heading);
                const addRows = (count) => {
                    for (let index = 0; index < count; index += 1) {
                        const row = document.createElement('div');
                        row.className = 'row';
                        row.textContent = `Measured row ${index + 1}`;
                        root.appendChild(row);
                    }
                };
                addRows(28);
                const button = document.createElement('button');
                button.textContent = 'Add rows';
                button.addEventListener('click', () => addRows(18));
                root.appendChild(button);
                fetch('/api/extensions/module_widget_smoke/ping').then((response) => {
                    if (!response.ok) throw new Error('ping failed');
                }).catch(() => {});
            })();
            """
        ),
        encoding="utf-8",
    )
    (skill_dir / "small.js").write_text(
        "document.getElementById('root').textContent = 'Small content';\n",
        encoding="utf-8",
    )
    (skill_dir / "fault.js").write_text(
        textwrap.dedent(
            """\
            const root = document.getElementById('root');
            root.innerHTML = '<div style="height: 900px">Deterministic fault fixture</div>';
            throw new Error('Deterministic module widget fault: ' + 'x'.repeat(320));
            """
        ),
        encoding="utf-8",
    )
    content_hash = compute_content_hash(skill_dir, manifest_entry="plugin.py")
    save_review_state(data_dir, name, SkillReviewState(status="pass", content_hash=content_hash))
    return name


def _write_temporal_module_widget_extension(data_dir: pathlib.Path) -> str:
    """Install a Cache-derived width/height feedback fixture."""
    from ouroboros.skill_loader import SkillReviewState, compute_content_hash, save_review_state

    name = "module_widget_temporal"
    skill_dir = data_dir / "skills" / "external" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        textwrap.dedent(
            f"""\
            ---
            name: {name}
            description: Isolated module widget convergence fixture.
            version: 0.1.0
            type: extension
            entry: plugin.py
            permissions: ["widget"]
            ---
            # Module widget convergence fixture
            """
        ),
        encoding="utf-8",
    )
    (skill_dir / "plugin.py").write_text(
        textwrap.dedent(
            """\
            def register(api):
                module = {"kind": "module", "entry": "widget.js", "start": "auto"}
                api.register_ui_tab("auto", "Temporal auto", render={**module, "span": 2})
                api.register_ui_tab("capped", "Temporal capped", render={**module, "span": 2, "max_height": 1000})
                api.register_ui_tab("fixed", "Temporal fixed", render={**module, "height": 480})
                api.register_ui_tab("floor", "Temporal floor", render={"kind": "module", "entry": "small.js", "max_height": 320, "start": "auto"})
                api.register_ui_tab("wide", "Temporal wide", render={"kind": "module", "entry": "wide.js", "span": 2, "max_height": 700, "start": "auto"})
                api.register_ui_tab("sibling", "Temporal sibling", render={"kind": "module", "entry": "small.js", "height": 360, "start": "auto"})
            """
        ),
        encoding="utf-8",
    )
    (skill_dir / "widget.js").write_text(
        textwrap.dedent(
            """\
            (() => {
                const root = document.getElementById('root');
                const style = document.createElement('style');
                style.textContent = `
                    *{box-sizing:border-box;margin:0;padding:0}
                    html{overflow-y:auto!important}
                    body,#root{width:100%;min-height:100%;overflow-x:hidden;padding:16px 18px}
                    body{overflow-y:auto!important;padding-bottom:14px;border-bottom:0 solid #e85d6f;font:14px sans-serif;color:#edf2f7;background:#111016}
                    body.bottom-accounting{border-bottom-width:2px}
                    main{display:flex;flex-direction:column;gap:10px;width:100%}
                    .controls{display:flex;gap:8px;flex-wrap:wrap}.controls button{padding:5px 9px}
                    .thresholds{display:flex;flex-direction:column;gap:1px}
                    .probe{display:flex;flex-wrap:wrap;gap:4px;min-width:0}
                    .probe i{display:block;height:10px;background:#48536a;border-radius:2px}
                    .probe .lead{flex:0 0 220px}.probe .tail{flex:0 0 var(--tail)}
                    .rows{display:flex;flex-direction:column;gap:5px}.row{height:50px;border:1px solid #48536a;background:#191923}
                    #bottom-marker{height:20px;border:1px solid #e85d6f;background:#251820}
                `;
                root.appendChild(style);
                root.insertAdjacentHTML('beforeend', '<main><div class="controls"><button data-action="grow">Grow</button><button data-action="shrink">Shrink</button></div><div class="thresholds"></div><div class="rows"></div><div id="bottom-marker">Bottom marker</div></main>');
                const thresholds = root.querySelector('.thresholds');
                for (let width = 214; width <= 310; width += 4) {
                    thresholds.insertAdjacentHTML('beforeend', `<div class="probe" style="--tail:${width}px"><i class="lead"></i><i class="tail"></i></div>`);
                }
                const rows = root.querySelector('.rows');
                const renderRows = (count) => {
                    rows.replaceChildren(...Array.from({length: count}, (_, index) => {
                        const row = document.createElement('div');
                        row.className = 'row';
                        row.textContent = `Dense 7D-like row ${index + 1}`;
                        return row;
                    }));
                };
                renderRows(3);
                root.querySelector('[data-action="grow"]').addEventListener('click', () => {
                    document.body.classList.remove('bottom-accounting');
                    renderRows(18);
                });
                root.querySelector('[data-action="shrink"]').addEventListener('click', () => {
                    renderRows(1);
                    document.body.classList.add('bottom-accounting');
                });
                const geometry = () => {
                    const body = document.body;
                    const marker = document.getElementById('bottom-marker');
                    const bodyStyle = getComputedStyle(body);
                    const rootStyle = getComputedStyle(root);
                    return {
                        height: innerHeight,
                        width: innerWidth,
                        clientWidth: document.documentElement.clientWidth,
                        scrollbarWidth: Math.max(0, innerWidth - document.documentElement.clientWidth),
                        scrollHeight: Math.max(document.documentElement.scrollHeight, body.scrollHeight),
                        scrollTop: Math.max(scrollY, document.documentElement.scrollTop, body.scrollTop),
                        rootHeight: root.getBoundingClientRect().height,
                        markerBottom: marker.getBoundingClientRect().bottom,
                        bottomSpacing: [rootStyle.paddingBottom, bodyStyle.paddingBottom, bodyStyle.borderBottomWidth]
                            .reduce((total, value) => total + (parseFloat(value) || 0), 0),
                    };
                };
                window.__fixtureGeometry = [];
                const report = () => {
                    const sample = {t: performance.now(), ...geometry()};
                    window.__fixtureGeometry.push(sample);
                    window.parent.postMessage({type: 'fixture-geometry', sample}, '*');
                };
                const observer = new ResizeObserver(report);
                observer.observe(root);
                observer.observe(document.body);
                window.addEventListener('resize', report);
                window.__resetFixtureGeometry = () => { window.__fixtureGeometry = []; };
                window.__ouroWidgetOnDispose?.(() => {
                    observer.disconnect();
                    window.removeEventListener('resize', report);
                });
                report();
            })();
            """
        ),
        encoding="utf-8",
    )
    (skill_dir / "small.js").write_text(
        textwrap.dedent(
            """\
            (() => {
                const root = document.getElementById('root');
                root.innerHTML = '<style>*{box-sizing:border-box}body{margin:0;padding:12px 14px 16px;border-bottom:2px solid #e85d6f;background:#111;color:#eee;font:14px sans-serif}</style><div id="bottom-marker">Small bottom marker</div>';
            })();
            """
        ),
        encoding="utf-8",
    )
    (skill_dir / "wide.js").write_text(
        textwrap.dedent(
            """\
            (() => {
                const root = document.getElementById('root');
                root.innerHTML = `
                    <style>
                        *{box-sizing:border-box}
                        html,body{margin:0}
                        body{padding:12px 14px 16px;background:#111;color:#eee;font:14px sans-serif}
                        #root{width:1200px}
                        #wide-track{width:1200px;height:84px;padding:18px;border:1px solid #48a6d9;white-space:nowrap;background:linear-gradient(90deg,#182335,#35182f)}
                        #bottom-marker{margin-top:10px;width:1200px;height:20px;border:1px solid #e85d6f}
                    </style>
                    <div id="wide-track">Document-level horizontal overflow must stay wheel-reachable all the way to this far edge.</div>
                    <div id="bottom-marker">Wide bottom marker</div>
                `;
            })();
            """
        ),
        encoding="utf-8",
    )
    content_hash = compute_content_hash(skill_dir, manifest_entry="plugin.py")
    save_review_state(data_dir, name, SkillReviewState(status="pass", content_hash=content_hash))
    return name


_WIDGET_TEMPORAL_TRACE_SCRIPT = r"""
(() => {
    const trace = {
        events: [], lastActivity: performance.now(),
        reset() {
            this.events = [];
            this.lastActivity = performance.now();
        },
    };
    window.__widgetTemporalTrace = trace;
    const roundedRect = (node) => {
        const box = node.getBoundingClientRect();
        return {x: +box.x.toFixed(2), y: +box.y.toFixed(2), width: +box.width.toFixed(2), height: +box.height.toFixed(2)};
    };
    const keyFor = (node) => node?.closest?.('[data-widget-key]')?.dataset.widgetKey || 'unknown';
    const record = (kind, node, extra = {}) => {
        trace.lastActivity = performance.now();
        const sibling = document.querySelector('[data-widget-key$=":sibling"]');
        trace.events.push({
            kind, key: keyFor(node), t: trace.lastActivity,
            ...(node ? roundedRect(node) : {}),
            sibling: sibling ? roundedRect(sibling) : null,
            ...extra,
        });
    };
    const frames = new WeakSet();
    const cards = new WeakSet();
    const frameObserver = new ResizeObserver((rows) => rows.forEach((row) => record('frame', row.target)));
    const cardObserver = new ResizeObserver((rows) => rows.forEach((row) => record('card', row.target)));
    const scan = () => {
        document.querySelectorAll('[data-widget-key]').forEach((card) => {
            if (!cards.has(card)) { cards.add(card); cardObserver.observe(card); }
        });
        document.querySelectorAll('[data-widget-key] iframe').forEach((frame) => {
            if (!frames.has(frame)) { frames.add(frame); frameObserver.observe(frame); }
        });
    };
    const frameOf = (source) => [...document.querySelectorAll('[data-widget-key] iframe')]
        .find((candidate) => candidate.contentWindow === source);
    // Resize reports travel on each frame's bridge channel: its hello names the frame that owns the port.
    const portFrames = new WeakMap();
    addEventListener('message', (event) => {
        const message = event.data || {};
        if (message.type === 'ouro-widget-hello' && event.ports?.[0]) portFrames.set(event.ports[0], frameOf(event.source));
        if (message.type !== 'fixture-geometry') return;
        record('child', frameOf(event.source), { heightValue: message.height, child: message.sample || null });
    }, true);
    const handler = Object.getOwnPropertyDescriptor(MessagePort.prototype, 'onmessage');
    Object.defineProperty(MessagePort.prototype, 'onmessage', {
        configurable: true,
        get() { return handler.get.call(this); },
        set(fn) {
            const port = this;
            handler.set.call(this, fn && function (event) {
                const message = event.data || {};
                if (message.type === 'ouro-widget-resize') {
                    record('message', portFrames.get(port), { heightValue: message.height, child: null });
                }
                return fn.call(this, event);
            });
        },
    });
    const start = () => {
        const observer = new MutationObserver((records) => {
            records.forEach((row) => {
                // Masonry writes its plan as custom properties (widgets lifecycle
                // phase 3): `--masonry-h` on the list is the one write per layout.
                if (row.type === 'attributes' && row.target.matches?.('.widgets-list')) {
                    record('masonry', row.target, {height: row.target.style.getPropertyValue('--masonry-h')});
                }
            });
            scan();
        });
        observer.observe(document.documentElement, {
            subtree: true, childList: true, attributes: true, attributeFilter: ['style'],
        });
        scan();
    };
    if (document.readyState === 'loading') addEventListener('DOMContentLoaded', start, {once: true});
    else start();
})();
"""


def _wait_for_widget_quiet(page, timeout_ms: int = 6_000) -> bool:
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

    try:
        page.wait_for_function(
            "() => performance.now() - window.__widgetTemporalTrace.lastActivity >= 1000",
            timeout=timeout_ms,
        )
        return True
    except PlaywrightTimeoutError:
        return False


def _has_sustained_alternation(values, minimum: int = 6) -> bool:
    compressed = [value for index, value in enumerate(values) if index == 0 or value != values[index - 1]]
    return any(
        len(set(compressed[index:index + minimum])) == 2
        and all(compressed[offset] == compressed[offset + 2] for offset in range(index, index + minimum - 2))
        for index in range(max(0, len(compressed) - minimum + 1))
    )


def _write_job_widget_smoke_extension(data_dir: pathlib.Path) -> str:
    """Install a tiny declarative job widget for retry-preservation E2E."""
    from ouroboros.skill_loader import SkillReviewState, compute_content_hash, save_review_state

    name = "job_widget_smoke"
    skill_dir = data_dir / "skills" / "external" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        textwrap.dedent(
            f"""\
            ---
            name: {name}
            description: Isolated declarative job retry smoke.
            version: 0.1.0
            type: extension
            entry: plugin.py
            permissions: ["route", "widget"]
            ---
            # Job retry smoke
            """
        ),
        encoding="utf-8",
    )
    (skill_dir / "plugin.py").write_text(
        textwrap.dedent(
            """\
            async def start(_request):
                return {"job_id": "retry-job"}


            async def status(_request):
                return {"status": "queued", "progress": 10}


            def register(api):
                api.register_route("start", start, methods=("POST",))
                api.register_route("status", status, methods=("GET",))
                api.register_ui_tab(
                    "main",
                    "Job retry",
                    render={
                        "kind": "declarative",
                        "schema_version": 1,
                        "components": [
                            {"type": "action", "id": "job-action", "label": "Start job", "route": "start", "method": "POST", "job": True, "status_route": "status"},
                            {"type": "status", "id": "job-status", "target": "result", "loading": "Waiting", "success": "Done", "error": "Failed"},
                        ],
                    },
                )
            """
        ),
        encoding="utf-8",
    )
    content_hash = compute_content_hash(skill_dir, manifest_entry="plugin.py")
    save_review_state(data_dir, name, SkillReviewState(status="pass", content_hash=content_hash))
    return name


@pytest.mark.ui_browser
def test_ui_smoke_job_poll_preserves_id_after_transient_failure(direct_server_with_data):
    """Prove a retryable status failure keeps the same job id and resumes."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    skill = _write_job_widget_smoke_extension(direct_server_with_data["data_dir"])
    status_urls = []

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 800})
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

                def fulfill_status(route):
                    status_urls.append(route.request.url)
                    if len(status_urls) == 1:
                        route.fulfill(status=503, content_type="application/json", body=json.dumps({"error": "temporary"}))
                    elif len(status_urls) == 2:
                        route.fulfill(status=200, content_type="application/json", body=json.dumps({"status": "queued", "progress": 20}))
                    else:
                        route.fulfill(status=200, content_type="application/json", body=json.dumps({"status": "done", "result": {"message": "finished"}}))

                page.route(f"**/api/extensions/{skill}/status**", fulfill_status)
                page.click('[data-nav-page="widgets"]')
                card = page.locator(f'[data-widget-key="{skill}:main"]')
                card.wait_for(state="visible", timeout=30_000)
                card.locator('[data-widget-action="id:job-action"]').click()
                for _ in range(100):
                    if status_urls:
                        break
                    page.wait_for_timeout(50)
                assert len(status_urls) == 1, status_urls
                page.evaluate(
                    """() => {
                        const button = [...document.querySelectorAll('[data-nav-page="dashboard"]')]
                            .find((item) => getComputedStyle(item).display !== 'none');
                        button?.click();
                    }"""
                )
                # The declarative card has no iframe. Waiting beyond the
                # existing 2s interval proves its disposed poller did not
                # issue the queued retry while Widgets was hidden.
                page.wait_for_timeout(2_200)
                assert len(status_urls) == 1, status_urls
                page.evaluate(
                    """() => {
                        const button = [...document.querySelectorAll('[data-nav-page="widgets"]')]
                            .find((item) => getComputedStyle(item).display !== 'none');
                        button?.click();
                    }"""
                )
                card.wait_for(state="visible", timeout=10_000)
                page.wait_for_function(
                    "() => document.querySelector('.widget-status[data-state=\"success\"]')",
                    timeout=10_000,
                )
                assert len(status_urls) >= 3, status_urls
                job_ids = {url.split("job_id=", 1)[1].split("&", 1)[0] for url in status_urls}
                assert job_ids == {"retry-job"}, status_urls
                assert card.locator('.widget-status[data-state="error"]').count() == 0
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise


@pytest.mark.ui_browser
def test_ui_smoke_module_widgets_geometry_lifecycle(direct_server_with_data):
    """Prove module auto-height, fixed/capped contracts, and framed teardown."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    data_dir = direct_server_with_data["data_dir"]
    skill = _write_module_widget_smoke_extension(data_dir)
    evidence_dir = pathlib.Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", str(data_dir.parent)))
    evidence_dir.mkdir(parents=True, exist_ok=True)

    def card_selector(tab_id: str) -> str:
        return f'[data-widget-key="{skill}:{tab_id}"]'

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
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
                auto = page.locator(card_selector("auto"))
                auto.wait_for(state="visible", timeout=30_000)
                auto_frame = page.frame_locator(f'{card_selector("auto")} iframe')
                auto_frame.locator("#root").wait_for(state="visible", timeout=10_000)

                page.wait_for_function(
                    """(selector) => {
                        const frame = document.querySelector(`${selector} iframe`);
                        return frame && frame.getBoundingClientRect().height > 320;
                    }""",
                    arg=card_selector("auto"),
                    timeout=10_000,
                )
                auto_metrics = page.locator(f'{card_selector("auto")} iframe').evaluate(
                    """frame => ({height: frame.getBoundingClientRect().height, client: frame.clientHeight})"""
                )
                child_metrics = auto_frame.locator("body").evaluate(
                    """body => {
                        const root = document.querySelector('#root');
                        const style = getComputedStyle(body);
                        return {
                            client: body.clientHeight,
                            scroll: body.scrollHeight,
                            root: root?.scrollHeight || 0,
                            cssHeight: style.height,
                            padding: `${style.paddingTop}/${style.paddingBottom}`,
                            marginBottom: style.marginBottom,
                            bodyRect: body.getBoundingClientRect().toJSON(),
                            rootRect: root?.getBoundingClientRect().height || 0,
                            rootBottom: root?.getBoundingClientRect().bottom || 0,
                            scrollTop: body.scrollTop,
                        };
                    }"""
                )
                assert auto_metrics["height"] > 320, auto_metrics
                # Chromium can retain a two-pixel fractional/border rounding delta;
                # the host's own resize reserve is tested separately below.
                assert child_metrics["scroll"] <= child_metrics["client"] + 3, child_metrics
                assert child_metrics["root"] <= auto_metrics["client"] + 3, child_metrics

                first_height = auto_metrics["height"]
                page.wait_for_timeout(250)
                second_height = page.locator(f'{card_selector("auto")} iframe').evaluate(
                    "frame => frame.getBoundingClientRect().height"
                )
                assert abs(first_height - second_height) <= 1
                auto_frame.get_by_role("button", name="Add rows").click()
                page.wait_for_function(
                    """([selector, previous]) => document.querySelector(`${selector} iframe`)?.getBoundingClientRect().height > previous + 100""",
                    arg=[card_selector("auto"), second_height],
                    timeout=10_000,
                )

                fixed_frame = page.locator(f'{card_selector("fixed")} iframe')
                capped_frame = page.locator(f'{card_selector("capped")} iframe')
                small_frame = page.locator(f'{card_selector("small")} iframe')
                page.wait_for_function(
                    """(selectors) => selectors.every((selector) => document.querySelector(`${selector} iframe`))""",
                    arg=[card_selector("fixed"), card_selector("capped"), card_selector("small")],
                    timeout=10_000,
                )
                assert fixed_frame.evaluate("frame => frame.getBoundingClientRect().height") == 480
                capped_height = capped_frame.evaluate("frame => frame.getBoundingClientRect().height")
                assert capped_height == 640, capped_height
                small_height = small_frame.evaluate("frame => frame.getBoundingClientRect().height")
                # Measured short content: the board's frame is as tall as its document (320 px is a loading height
                # only, docs/DESIGN.md "Widgets board"), and nothing of it is cut.
                assert 0 < small_height < 320, small_height
                assert small_frame.element_handle().content_frame().evaluate(
                    "() => document.documentElement.scrollHeight <= innerHeight + 1"), "short content is cut"

                page.screenshot(path=str(evidence_dir / "module-widgets-desktop.png"), full_page=True)
                page.set_viewport_size({"width": 430, "height": 932})
                page.wait_for_timeout(300)
                assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")
                narrow_height = page.locator(f'{card_selector("auto")} iframe').evaluate(
                    "frame => frame.getBoundingClientRect().height"
                )
                assert narrow_height > 320
                page.screenshot(path=str(evidence_dir / "module-widgets-narrow.png"), full_page=True)

                # Leave/return: leaving disposes the framed mount, but the card
                # keeps its DOM identity (an expando survives only on the same
                # node), a hidden page issues no list request, and the return
                # issues exactly one `GET /api/widgets` before mounting again.
                widgets_list_requests = []

                def record_widgets_request(request):
                    if urllib.parse.urlparse(request.url).path == "/api/widgets":
                        widgets_list_requests.append(request.url)

                page.on("request", record_widgets_request)
                page.evaluate(
                    "(selector) => { document.querySelector(selector).__ouroCardIdentity = true; }",
                    card_selector("auto"),
                )
                page.evaluate(
                    """() => {
                        const button = [...document.querySelectorAll('[data-nav-page="dashboard"]')]
                            .find((item) => getComputedStyle(item).display !== 'none');
                        button?.click();
                    }"""
                )
                page.wait_for_function(
                    "(selector) => document.querySelector(`${selector} iframe`) === null",
                    arg=card_selector("auto"),
                    timeout=10_000,
                )
                page.wait_for_timeout(300)
                assert widgets_list_requests == [], widgets_list_requests
                page.evaluate(
                    """() => {
                        const button = [...document.querySelectorAll('[data-nav-page="widgets"]')]
                            .find((item) => getComputedStyle(item).display !== 'none');
                        button?.click();
                    }"""
                )
                page.locator(card_selector("auto")).locator("iframe").wait_for(state="attached", timeout=10_000)
                assert page.evaluate(
                    "(selector) => document.querySelector(selector).__ouroCardIdentity === true",
                    card_selector("auto"),
                ), "returning to Widgets must reuse the existing card node, not rebuild the list"
                assert len(widgets_list_requests) == 1, widgets_list_requests
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise


@pytest.mark.ui_browser
@pytest.mark.parametrize("browser_name", ("chromium", "webkit"))
def test_ui_smoke_module_widget_temporal_convergence(direct_server_with_data, browser_name):
    """Prove module geometry converges across wrapping, cap, and shrink transitions."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    data_dir = direct_server_with_data["data_dir"]
    skill = _write_temporal_module_widget_extension(data_dir)
    evidence_dir = data_dir.parent / "widget-convergence-evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    def card_selector(tab_id: str) -> str:
        return f'[data-widget-key="{skill}:{tab_id}"]'

    def frame_selector(tab_id: str) -> str:
        return f'{card_selector(tab_id)} iframe'

    def child_metrics(frame):
        return frame.locator("body").evaluate(
            """() => {
                const body = document.body;
                const root = document.querySelector('#root');
                const marker = document.querySelector('#bottom-marker');
                const bodyStyle = getComputedStyle(body);
                const rootStyle = getComputedStyle(root);
                const htmlStyle = getComputedStyle(document.documentElement);
                const scrollTop = Math.max(scrollY, document.documentElement.scrollTop, body.scrollTop);
                const scrollLeft = Math.max(scrollX, document.documentElement.scrollLeft, body.scrollLeft);
                return {
                    height: innerHeight,
                    width: innerWidth,
                    clientWidth: document.documentElement.clientWidth,
                    scrollbarWidth: Math.max(0, innerWidth - document.documentElement.clientWidth),
                    scrollHeight: Math.max(document.documentElement.scrollHeight, body.scrollHeight),
                    scrollWidth: Math.max(document.documentElement.scrollWidth, body.scrollWidth),
                    scrollTop,
                    scrollLeft,
                    htmlOverflowX: htmlStyle.overflowX,
                    htmlOverflowY: htmlStyle.overflowY,
                    bodyOverflowX: bodyStyle.overflowX,
                    bodyOverflowY: bodyStyle.overflowY,
                    markerTop: marker?.getBoundingClientRect().top || 0,
                    markerBottom: marker?.getBoundingClientRect().bottom || 0,
                    tailExtent: (marker?.getBoundingClientRect().bottom || 0)
                        + [rootStyle.paddingBottom, bodyStyle.paddingBottom, bodyStyle.borderBottomWidth]
                            .reduce((total, value) => total + (parseFloat(value) || 0), 0),
                    bottomSpacing: [rootStyle.paddingBottom, bodyStyle.paddingBottom, bodyStyle.borderBottomWidth]
                        .reduce((total, value) => total + (parseFloat(value) || 0), 0),
                    geometrySamples: window.__fixtureGeometry?.length || 0,
                };
            }"""
        )

    def reset_trace(page, frame=None):
        page.evaluate("window.__widgetTemporalTrace.reset()")
        if frame is not None:
            frame.locator("body").evaluate("() => window.__resetFixtureGeometry?.()")

    def assert_converged(page, phase: str, *, masonry_limit: int | None = None):
        quiet = _wait_for_widget_quiet(page)
        trace = page.evaluate("window.__widgetTemporalTrace")
        heights = [round(row["heightValue"]) for row in trace["events"] if row["kind"] == "message"]
        sibling_pairs = [
            (round(row["sibling"]["x"], 1), round(row["sibling"]["width"], 1))
            for row in trace["events"] if row.get("sibling")
        ]
        diagnostic = json.dumps(
            {
                "engine": browser_name,
                "phase": phase,
                "quiet": quiet,
                "events": len(trace["events"]),
                "heights": heights[-30:],
            },
            sort_keys=True,
        )
        assert quiet, diagnostic
        assert not _has_sustained_alternation(heights), diagnostic
        assert not _has_sustained_alternation(sibling_pairs), diagnostic
        masonry_count = sum(row["kind"] == "masonry" for row in trace["events"])
        if masonry_limit is not None:
            assert masonry_count <= masonry_limit, diagnostic
        sibling = page.locator(card_selector("sibling")).evaluate(
            "node => { const box = node.getBoundingClientRect(); return {x: box.x, width: box.width}; }"
        )
        assert sibling["width"] > 0
        return trace

    try:
        with sync_playwright() as pw:
            browser = getattr(pw, browser_name).launch(headless=True)
            page = browser.new_page(viewport={"width": 1478, "height": 972})
            page.add_init_script(_WIDGET_TEMPORAL_TRACE_SCRIPT)
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
                frames = {
                    tab_id: page.frame_locator(frame_selector(tab_id))
                    for tab_id in ("auto", "capped", "fixed", "floor", "wide", "sibling")
                }
                for frame in frames.values():
                    frame.locator("#bottom-marker").wait_for(state="visible", timeout=30_000)

                assert_converged(page, "initial")

                initial_heights = {
                    tab_id: page.locator(frame_selector(tab_id)).evaluate(
                        "frame => frame.getBoundingClientRect().height"
                    )
                    for tab_id in frames
                }
                assert 320 < initial_heights["auto"] < 1000, initial_heights
                assert 320 < initial_heights["capped"] < 1000, initial_heights
                assert initial_heights["fixed"] == 480, initial_heights
                # Short content under its author's 320 px ceiling, and the wide strip: as tall as their documents.
                assert 0 < initial_heights["floor"] < 320, initial_heights
                assert 0 < initial_heights["wide"] < 700, initial_heights
                for tab_id in frames:
                    assert page.locator(frame_selector(tab_id)).get_attribute("scrolling") is None
                auto_initial = child_metrics(frames["auto"])
                assert auto_initial["scrollbarWidth"] >= 0
                assert auto_initial["htmlOverflowY"] == "hidden", auto_initial
                assert auto_initial["bodyOverflowY"] == "hidden", auto_initial
                assert auto_initial["htmlOverflowX"] != "hidden", auto_initial
                assert auto_initial["bodyOverflowX"] == "hidden", auto_initial

                # It fits under its ceiling, so the frame shows all of it and the document's scrollbar stays hidden.
                floor_initial = child_metrics(frames["floor"])
                assert floor_initial["htmlOverflowY"] == "hidden", floor_initial
                assert floor_initial["bodyOverflowY"] == "hidden", floor_initial
                assert floor_initial["scrollHeight"] <= floor_initial["height"] + 1, ("short content is cut", floor_initial)

                fixed_initial = child_metrics(frames["fixed"])
                assert fixed_initial["htmlOverflowY"] != "hidden", fixed_initial
                assert fixed_initial["bodyOverflowY"] == "auto", fixed_initial
                page.screenshot(path=str(evidence_dir / f"{browser_name}-dense.png"), full_page=True)

                wide_before = child_metrics(frames["wide"])
                assert wide_before["scrollWidth"] > wide_before["width"] + 400, wide_before
                assert wide_before["scrollLeft"] == 0, wide_before
                assert wide_before["scrollTop"] == 0, wide_before
                assert wide_before["htmlOverflowY"] == "hidden", wide_before
                assert wide_before["bodyOverflowY"] == "hidden", wide_before
                assert wide_before["htmlOverflowX"] != "hidden", wide_before
                assert wide_before["bodyOverflowX"] != "hidden", wide_before
                reset_trace(page, frames["wide"])
                frames["wide"].locator("#wide-track").hover()
                page.mouse.wheel(700, 240)
                wide_content = page.locator(frame_selector("wide")).element_handle().content_frame()
                wide_content.wait_for_function(
                    "() => Math.max(scrollX, document.documentElement.scrollLeft, document.body.scrollLeft) > 0",
                    timeout=5_000,
                )
                wide_after = child_metrics(frames["wide"])
                assert wide_after["scrollLeft"] > 0, wide_after
                assert wide_after["scrollTop"] == 0, wide_after
                assert_converged(page, "wide-horizontal-wheel", masonry_limit=8)

                reset_trace(page, frames["auto"])
                frames["auto"].locator('[data-action="grow"]').click()
                page.wait_for_function(
                    "([selector, prior]) => document.querySelector(selector)?.getBoundingClientRect().height > prior + 300",
                    arg=[frame_selector("auto"), initial_heights["auto"]],
                    timeout=10_000,
                )
                grown_auto_height = page.locator(frame_selector("auto")).evaluate(
                    "frame => frame.getBoundingClientRect().height"
                )
                assert grown_auto_height < 8192
                auto_grown = child_metrics(frames["auto"])
                assert auto_grown["htmlOverflowY"] == "hidden", auto_grown
                assert auto_grown["bodyOverflowY"] == "hidden", auto_grown
                assert_converged(page, "auto-grow", masonry_limit=8)

                reset_trace(page, frames["auto"])
                frames["auto"].locator('[data-action="shrink"]').click()
                page.wait_for_function(
                    "([selector, grown]) => document.querySelector(selector)?.getBoundingClientRect().height < grown - 300",
                    arg=[frame_selector("auto"), grown_auto_height],
                    timeout=10_000,
                )
                assert_converged(page, "auto-shrink", masonry_limit=8)
                auto_shrunk = child_metrics(frames["auto"])
                assert auto_shrunk["htmlOverflowY"] == "hidden", auto_shrunk
                assert auto_shrunk["bodyOverflowY"] == "hidden", auto_shrunk
                assert auto_shrunk["bottomSpacing"] == 32, auto_shrunk
                assert auto_shrunk["scrollHeight"] <= auto_shrunk["height"] + 3, auto_shrunk
                assert auto_shrunk["tailExtent"] <= auto_shrunk["height"] + 3, auto_shrunk
                page.screenshot(path=str(evidence_dir / f"{browser_name}-after-shrink.png"), full_page=True)

                reset_trace(page, frames["capped"])
                frames["capped"].locator('[data-action="grow"]').click()
                page.wait_for_function(
                    "selector => document.querySelector(selector)?.getBoundingClientRect().height === 1000",
                    arg=frame_selector("capped"),
                    timeout=10_000,
                )
                capped_before_key = child_metrics(frames["capped"])
                assert capped_before_key["scrollHeight"] > capped_before_key["height"] + 20, capped_before_key
                assert capped_before_key["htmlOverflowY"] != "hidden", capped_before_key
                assert capped_before_key["bodyOverflowY"] == "auto", capped_before_key
                frames["capped"].locator('[data-action="grow"]').focus()
                page.keyboard.press("PageDown")
                capped_content = page.locator(frame_selector("capped")).element_handle().content_frame()
                capped_content.wait_for_function(
                    "() => Math.max(scrollY, document.documentElement.scrollTop, document.body.scrollTop) > 0",
                    timeout=5_000,
                )
                after_page_down = child_metrics(frames["capped"])
                assert after_page_down["scrollTop"] > 0, after_page_down
                page.keyboard.press("End")
                capped_content.wait_for_function(
                    "() => { const marker = document.querySelector('#bottom-marker'); return marker && marker.getBoundingClientRect().bottom <= innerHeight + 1; }",
                    timeout=5_000,
                )
                capped_bottom = child_metrics(frames["capped"])
                assert 0 < capped_bottom["markerBottom"] <= capped_bottom["height"] + 1, capped_bottom
                assert_converged(page, "capped-grow-and-keyboard", masonry_limit=8)
                page.screenshot(path=str(evidence_dir / f"{browser_name}-capped.png"), full_page=True)

                reset_trace(page, frames["capped"])
                frames["capped"].locator('[data-action="shrink"]').click()
                page.wait_for_function(
                    "selector => document.querySelector(selector)?.getBoundingClientRect().height < 1000",
                    arg=frame_selector("capped"),
                    timeout=10_000,
                )
                assert_converged(page, "capped-shrink", masonry_limit=8)
                capped_shrunk = child_metrics(frames["capped"])
                assert capped_shrunk["htmlOverflowY"] == "hidden", capped_shrunk
                assert capped_shrunk["bodyOverflowY"] == "hidden", capped_shrunk
                assert capped_shrunk["bottomSpacing"] == 32, capped_shrunk
                assert capped_shrunk["scrollHeight"] <= capped_shrunk["height"] + 3, capped_shrunk
                assert capped_shrunk["tailExtent"] <= capped_shrunk["height"] + 3, capped_shrunk

                reset_trace(page, frames["fixed"])
                frames["fixed"].locator('[data-action="grow"]').click()
                assert_converged(page, "fixed-grow", masonry_limit=8)
                reset_trace(page, frames["fixed"])
                frames["fixed"].locator('[data-action="shrink"]').click()
                assert_converged(page, "fixed-shrink", masonry_limit=8)
                assert page.locator(frame_selector("fixed")).evaluate(
                    "frame => frame.getBoundingClientRect().height"
                ) == 480
                fixed_final = child_metrics(frames["fixed"])
                assert fixed_final["htmlOverflowY"] != "hidden", fixed_final
                assert fixed_final["bodyOverflowY"] == "auto", fixed_final
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise


@pytest.mark.ui_browser
def test_ui_smoke_widgets_page_has_a_typed_address_and_one_nav_landmark(
    direct_server_with_data,
):
    """A page could not be linked to and the sidebar was not a landmark.

    Opening the application with a `#widgets` fragment must land on Widgets
    (browsers gain a shareable link; the desktop shell and the Linux fallback
    get the same load-time route with no address bar to break), and an
    automation or assistive client must be able to resolve the navigation by
    its landmark instead of by an id.
    """
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            try:
                page.goto(f"{url}/#widgets", wait_until="domcontentloaded", timeout=30_000)
                page.wait_for_selector("nav#primary-sidebar", timeout=15_000)
                page.wait_for_selector("#page-widgets.active", timeout=15_000)
                assert page.locator('[data-nav-page="widgets"].active').count() == 1
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise



# In the top window only: every message the Beautiful Context frame posts to it (none is expected; a probe proves
# the recorder hears one).
_REPORT_MESSAGES = """(() => {
  if (window.top !== window) return;
  window.__reportMessages = [];
  window.addEventListener('message', (event) => {
    const report = document.querySelector("[data-widget-key='beautiful_context:viewer'] iframe.widgets-frame");
    if (report && event.source === report.contentWindow) window.__reportMessages.push(String(event.data));
  }, true);
})();"""

# A route lease is a credential for the frame's page. It stays out of every recorded request, assertion message and
# picture: only its place in the address (`/api/widget-resources/<lease>/viewer`) is ever compared.
_LEASE = re.compile(r"(/api/widget-resources/)[^/?#]+")
_VIEWER_PATH = re.compile(r"/api/widget-resources/[A-Za-z0-9_-]+/viewer")

# What a person sees under one element: its ink and the backgrounds beneath it, element outward. Any CSS colour is read
# as sRGB through a canvas, so the frame's own palette tokens and their syntax are never part of the measurement.
_LAYERS = """(element) => {
  const canvas = document.createElement('canvas').getContext('2d', { willReadFrequently: true });
  const rgba = (css) => {
    canvas.clearRect(0, 0, 1, 1);
    canvas.fillStyle = css;
    canvas.fillRect(0, 0, 1, 1);
    const [red, green, blue, alpha] = canvas.getImageData(0, 0, 1, 1).data;
    return [red, green, blue, alpha / 255];
  };
  const layers = [];
  for (let node = element; node; node = node.parentElement) layers.push(rgba(getComputedStyle(node).backgroundColor));
  const root = element.ownerDocument.documentElement;  // how this document reads its scheme, for a failure to explain itself
  return { ink: rgba(getComputedStyle(element).color), layers, shadow: getComputedStyle(element).boxShadow,
           scheme: [getComputedStyle(root).colorScheme, window.matchMedia('(prefers-color-scheme: dark)').matches] };
}"""


def _redacted(address: str) -> str:
    return _LEASE.sub(r"\1[lease]", address)


def _reader_command(task: str, attempt: str, max_bytes: int | None = None, *, report: bool = False) -> re.Pattern:
    """The reader's command a help line shows, whatever the words around it: these arguments, then one quoted absolute
    file in a private folder (`…/private-directory/ATTEMPT_ID.json` for a report, `…-request.json` for the request)."""
    arguments = f" --data-root DATA {task} --attempt {attempt}" + (f" --max-bytes {max_bytes}" if max_bytes else "")
    file = rf"(?:ATTEMPT_ID|{re.escape(attempt)})" + (r"\.json" if report else r"-request\.json")
    return re.compile(re.escape("python -m ouroboros.context_attempt_reader" + arguments + (" --report" if report else ""))
                      + rf'\s*>\s*"/[^"]*private-directory/{file}"')


def _under(layers, canvas):
    """The colour the eye gets from `layers` (innermost first) lying over `canvas`."""
    color = canvas
    for red, green, blue, alpha in reversed(layers):
        color = tuple(below * (1 - alpha) + above * alpha for below, above in zip(color, (red, green, blue)))
    return color


def _luminance(color) -> float:
    red, green, blue = ((value / 255 / 12.92 if value <= 10.31475 else ((value / 255 + 0.055) / 1.055) ** 2.4)
                        for value in color)
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def _painted(inner: dict, outer: dict, theme: str) -> dict:
    """One piece of text as painted: the frame's layers (`inner`) lie over those of the card that holds the frame (`outer`)."""
    background = _under(inner["layers"] + outer["layers"], (255, 255, 255) if theme == "light" else (0, 0, 0))
    ink = _under([inner["ink"]], background)
    light, dark = sorted((_luminance(ink), _luminance(background)), reverse=True)
    return {"ink": tuple(round(each) for each in ink), "background": tuple(round(each) for each in background),
            "ink_lighter": _luminance(ink) > _luminance(background), "contrast": round((light + 0.05) / (dark + 0.05), 2),
            "shadow": inner["shadow"], "frame_scheme": inner["scheme"]}


@pytest.mark.ui_browser
@pytest.mark.parametrize("browser_name", ("chromium", "webkit"))
def test_ui_smoke_beautiful_context_reads_the_selected_attempt(
    direct_server_with_data, browser_name, tmp_path, monkeypatch, capsys,
):
    """Beautiful Context is an optional bundled widget: absent by default, live on Widgets once enabled, gone once disabled.

    One page keeps the Widgets board open while a second page does what an owner does in Extensions: Browse, the Built
    in source, Install on the skill's card (which turns it on once ready, and whose job ends with no
    `extension_lifecycle`), later the switch on its Installed card off and on again. The card must arrive on and leave
    the already mounted board through the host's lifecycle notices (no reload), and be there, or gone, after a reload.
    The reports are the real reader's over real sealed attempts. A stale CSP hash, a body shown under another
    attempt's facts, an earlier result left under a failed selection, text drawn as markup, a request held
    twice in the frame, a gap explained only inside a closed disclosure or a ledger state read as a final
    charge each fail here. So does unreadable text when System follows an emulated device (dark to light) or when
    the host manually switches away from and back to the browser's unforced preference on the open report. The
    frame must not send anything out: a message to its parent, a request or a WebSocket frame, before or
    after a file is chosen (the frame's own page load is the observers' positive control). The route lease is a
    credential, so no address with it is kept, shown or pictured.
    """
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import expect, sync_playwright

    from tests._context_reports import context_reports
    from tests.ui_console_errors import record_console_errors, record_failed_loads, unexpected_console_errors

    reports = context_reports(tmp_path / "report-data", monkeypatch, capsys)
    first, second = reports["attempts"]
    (failed, failed_report), (retried, retried_report) = reports["failed"], reports["retried"]
    stored = json.loads(reports["first"])["stored_json"]
    # What no request or socket frame may carry: words of the exported files (each different report has its own) and their ids.
    private = ("Marigold", "fenders", "tide table", "Tide sensor", "Where did the attachment go?",
               "sk-secret-model-send-seal", first, second, failed, retried)
    skill = "beautiful_context"
    card_selector = f"[data-widget-key='{skill}:viewer']"
    selector = f"{card_selector} iframe.widgets-frame"
    url = direct_server_with_data["url"]
    home = urllib.parse.urlsplit(url)
    evidence = pathlib.Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR") or tmp_path / "evidence") / "beautiful-context"
    evidence.mkdir(parents=True, exist_ok=True)
    phase, requests, sockets = ["boot"], [], []

    def on_request(request):
        try:
            where = request.frame.url
        except PlaywrightError:  # a request of a frame already gone
            where = ""
        requests.append({"phase": phase[0], "path": _redacted(urllib.parse.urlsplit(request.url).path),
                         "url": _redacted(request.url), "frame": _redacted(where),
                         "navigation": request.is_navigation_request(),
                         "data": (request.post_data_buffer or b"").decode("utf-8", "replace")})

    def on_socket(socket):
        record = {"phase": phase[0], "url": socket.url, "sent": []}
        socket.on("framesent", lambda payload: record["sent"].append(str(payload)))
        sockets.append(record)

    try:
        with sync_playwright() as pw:
            browser = getattr(pw, browser_name).launch(headless=True)
            context = browser.new_context(viewport={"width": 1280, "height": 800}, color_scheme="null")
            context.add_init_script(_REPORT_MESSAGES)
            context.on("request", on_request)
            page = context.new_page()
            page.emulate_media(color_scheme="dark")  # Start in System with a deterministic device preference.
            page.on("websocket", on_socket)
            page.set_default_timeout(15_000)
            errors, failed_loads = record_console_errors(page), record_failed_loads(page)
            try:
                card, report = page.locator(card_selector), page.locator(selector)
                frame = page.frame_locator(selector)

                def listed(response) -> bool:
                    return urllib.parse.urlsplit(response.url).path == "/api/widgets" and response.ok

                def board_is_read() -> None:
                    # The board holds the list: a missing card is then a fact about the list, not a read still going.
                    expect(page.locator("#widgets-list")).not_to_contain_text("Loading widgets")
                    expect(page.locator("#widgets-list-error")).to_be_hidden()

                def shoot(name: str, target=None) -> None:
                    # WebKit's screenshot puts a stylesheet into the frame, which the frame's policy refuses: that one
                    # console line per picture is the screenshot's, never the page's, and is set aside here.
                    seen = len(errors)
                    (card if target is None else target).screenshot(path=str(evidence / f"{browser_name}-{name}.png"), caret="initial")
                    page.wait_for_timeout(200)
                    errors[seen:] = [each for each in errors[seen:] if browser_name != "webkit"
                                     or "/api/widget-resources/" not in each[1]
                                     or not each[0].startswith("Refused to apply a stylesheet because its hash")]

                def viewer_frame():
                    # The frame's page is this app's own, on its widget-resource route. The lease in that address is
                    # compared in place and never shown: each verdict is a plain boolean, which is all pytest can print.
                    (inner,) = [each for each in page.frames if _VIEWER_PATH.search(urllib.parse.urlsplit(each.url).path)]
                    address = report.get_attribute("src") or ""
                    where = urllib.parse.urlsplit(address)
                    own_origin = (where.scheme, where.netloc) == (home.scheme, home.netloc)
                    on_route = not where.query and not where.fragment and bool(_VIEWER_PATH.fullmatch(where.path))
                    the_card_address = inner.url == address
                    assert own_origin, "the viewer is not served from the app's own origin"
                    assert on_route, "the viewer is not on its widget-resource route"
                    assert the_card_address, "the frame is not on the address its card gave it"
                    return inner

                with page.expect_websocket(), page.expect_response(listed):  # the page's own socket and its first list
                    page.goto(f"{url}/#widgets", wait_until="domcontentloaded", timeout=30_000)
                page.wait_for_selector("#page-widgets.active")
                board_is_read()
                # Nothing about appearance is stored yet, so the device decides: dark here.
                assert page.evaluate("() => [window.ouroTheme.choice, document.documentElement.dataset.themeChoice]") == [
                    "system", "system"]
                kept = page.evaluate("""() => [...Object.keys(localStorage), ...document.cookie.split(';')
                    .map((pair) => pair.split('=')[0].trim())].filter((name) => /^ouroboros[.-]theme/.test(name))""")
                assert kept == [], kept
                expect(page.locator("html")).to_have_attribute("data-theme", "dark")
                # Absent by default: no permanent button, no Beautiful Context card, no frame. The board's own words
                # for an empty board are the product's to change; the picture keeps what they look like.
                assert page.locator("[data-context-report]").count() == 0
                assert page.locator("[data-context-report-mount]").count() == 0
                assert page.get_by_role("button", name="Beautiful Context").count() == 0
                assert card.count() == 0 and report.count() == 0
                shoot("0-board-off", page.locator("#page-widgets"))
                page.evaluate("() => { window.__boardKept = true; }")  # still this very document at every step below

                # What an owner does in Extensions, on a second page while the board above stays open: Browse, the Built
                # in source, Install on the skill's card. A new installation first asks once how the Ouroboros hub
                # installs; Later saves nothing and is not asked again on this visit.
                phase[0] = "install"
                extensions = context.new_page()
                extensions.set_default_timeout(15_000)
                extensions.goto(f"{url}/#skills", wait_until="domcontentloaded", timeout=30_000)
                extensions.locator('#page-skills [data-extensions-open="browse"]').click()
                hub = extensions.get_by_role("dialog", name="How should skills from the Ouroboros hub install?")
                hub.get_by_role("button", name="Later").click()
                expect(hub).to_have_count(0)
                extensions.locator("[data-browse-source]").select_option("builtin")
                with extensions.expect_request(lambda request: request.method == "POST" and urllib.parse.urlsplit(
                        request.url).path == "/api/skill-catalog/install") as asked:
                    extensions.locator(f'[data-browse-board] .marketplace-card[data-item="builtin/{skill}"] '
                                       '[data-browse-action="install"]').click()
                # Browse's Install turns the skill on once ready: the end of that job is all that reaches the open board.
                assert asked.value.post_data_json == {"id": f"builtin/{skill}", "turn_on_when_ready": True}
                expect(card).to_be_visible(timeout=30_000)  # on the board that never reloaded
                assert page.evaluate("() => window.__boardKept === true")
                assert "widgets-card-span-2" in (card.get_attribute("class") or "")
                expect(report).to_be_visible()
                # Opaque and bridge-free: sandboxed without allow-same-origin, loaded via its own route lease.
                sandbox = report.get_attribute("sandbox") or ""
                assert "allow-scripts" in sandbox and "allow-same-origin" not in sandbox
                expect(frame.locator("#report-file")).to_be_attached()
                viewer_frame()
                phase[0] = "select"
                shoot("1-board-on")
                facts = frame.locator("#attempt-facts")
                rows = frame.locator("#request .rows[data-kind='messages'] > .row")
                stored_body = frame.locator("#stored-body")

                def choose(content, shown: str, where: str = "#report-status") -> None:
                    # Through the picker a person uses: the control must take the click itself.
                    with page.expect_file_chooser() as picker:
                        frame.locator("#report-file").click()
                    picker.value.set_files([] if content is None else content if isinstance(content, str) else {
                        "name": "report.json", "mimeType": "application/json", "buffer": content})
                    if where == "#attempt-facts":
                        expect(frame.locator("#report-status")).to_have_text("Showing report.json.")
                        expect(facts).to_contain_text(shown)
                    else:
                        frame.locator(where).get_by_text(shown).wait_for()

                def gap_reads_without_opening(words) -> None:
                    # What a person sees with every disclosure closed: the gap's headline and its explanation; the
                    # stored JSON, where a gap also names itself, stays folded and is no evidence of either.
                    assert frame.locator("#report-result details[open]:not(#origins)").count() == 0
                    expect(frame.locator("#headline")).to_be_visible()
                    expect(frame.locator("#explain")).to_be_visible()
                    explained = frame.locator("#explain").inner_text()
                    assert words.search(explained) if isinstance(words, re.Pattern) else words in explained, explained
                    expect(stored_body).to_be_hidden()
                    assert rows.count() == 0 and frame.locator("#privacy").is_hidden()

                def gap(**facts_of_gap) -> bytes:
                    return json.dumps({"format": "ouroboros.context-report/1", "stored_json": None, "facts": {
                        "task_id": "report-task", "attempt_id": "earlier-attempt", **facts_of_gap}}).encode("utf-8")

                frame.get_by_role("button", name="Try an example").click()
                expect(frame.locator("#explain")).to_contain_text("Nothing was sent to a model")
                expect(frame.locator("#overview")).to_contain_text("Tools offered")
                expect(frame.get_by_text("Can I return a book on Sunday?", exact=True)).to_be_visible()
                expect(facts).to_be_hidden()
                choose(reports["first"], first, "#attempt-facts")
                expect(frame.locator("#overview")).to_contain_text("Low used · Max selected")
                expect(frame.locator("#overview")).to_contain_text("Cache read in ledger")
                expect(frame.locator("#source-limit")).to_be_visible()
                expect(facts).to_be_hidden()
                assert "example-request" not in facts.text_content()
                technical = frame.locator("#technical > summary")
                technical.click()
                expect(facts).to_be_visible()
                technical.click()
                build = frame.locator("#build-facts")
                value = lambda label: build.get_by_text(label, exact=True).locator("+ dd").text_content().strip()
                assert value("Context mode") == "Low · chosen Max", ascii(value("Context mode"))
                assert value("Sizing basis") == "Calibrated on recent usage from this route"
                assert value("Host prefix within this task") == "First change: system blocks · position 3 · shared understanding"
                assert value("Ledger input") == "50 tokens"
                assert value("Ledger output") == "2 tokens"
                assert value("Ledger cache read").startswith("0 tokens · ")
                assert value("Input outside the recorded cache read") == "50 tokens"
                assert value("Ledger cache write") == "Not recorded"
                assert 'Ledger text' not in build.text_content()
                assert "private-fixture-route" not in frame.locator("#record-body").text_content()
                sources = frame.get_by_text("Where these blocks came from", exact=True)
                expect(frame.locator("#source-versions")).to_be_visible()
                expect(frame.locator("#source-versions")).to_contain_text("BIBLE.md")
                expect(frame.locator("#source-versions")).to_contain_text("Your messages · 1 with a matching request row")
                expect(frame.locator("#source-versions")).to_contain_text("Messages · 4 · 3 without an exact source anchor")
                expect(frame.locator("#source-technical")).to_contain_text("in prepared text before redaction")
                assert "SHA-256" not in frame.locator("#source-versions").inner_text()
                sources.click()
                choose(reports['media'], 'media-task', '#attempt-facts')
                media_sources = frame.locator('#source-versions')
                expect(media_sources).to_be_visible()
                expect(media_sources).to_contain_text('6 entries considered · 0 non-image or path-only entries')
                expect(media_sources).to_contain_text('Entry 6 · image limit reached; not opened')
                assert value('Ledger input') == '1,000 tokens'
                assert value('Ledger output') == '100 tokens'
                assert value('Input outside the recorded cache read') == '900 tokens'
                shoot('attachment-label')
                choose(reports['first'], first, '#attempt-facts')
                frame.locator("#comparison > summary").click()
                pair = frame.locator("#comparison-facts")
                second_input = frame.locator("#comparison-file")

                def compare(contents):
                    second_input.set_input_files({"name": "second.json", "mimeType": "application/json", "buffer": contents})
                    expect(frame.locator("#comparison-status")).not_to_have_text("Reading the second report…")

                compare(reports["second"])
                expect(pair).to_contain_text("Identical redacted JSON")
                assert first not in pair.inner_text() and second not in pair.inner_text()
                ids = frame.locator("#comparison-identities")
                expect(ids).not_to_be_visible()
                frame.locator("#comparison-technical > summary").click()
                expect(ids).to_contain_text(first)
                expect(ids).to_contain_text(second)
                compare(reports["noisy"])
                expect(pair).to_contain_text("First difference · Messages")
                expect(pair).to_contain_text("Item 4 differs")
                expect(pair).to_contain_text("0 tokens → 0 tokens")
                pair.scroll_into_view_if_needed()
                shoot("compare-dark")
                frame.locator("h1").scroll_into_view_if_needed()
                compare(reports["over_limit"])
                expect(pair).to_contain_text("at least one report has no stored body")
                compare(b"{")
                expect(frame.locator("#comparison-status")).to_contain_text("Cannot compare")
                assert pair.text_content() == "" and stored_body.text_content() == stored
                frame.locator("#comparison > summary").click()
                assert "receipt not confirmed" in facts.text_content() and second not in facts.text_content()
                assert "Request recorded" in frame.locator("#headline").inner_text()
                assert "Private text can remain" in frame.locator("#privacy").text_content()
                # Its help writes a report only to a file of the owner's, in a private folder they name.
                assert _reader_command("TASK_ID", "ATTEMPT_ID", report=True).search(
                    frame.locator("#report-help").text_content())
                # The request is in the frame once, whole and literal; the overview names every message and
                # tool, and a row's text exists only while that row is open.
                assert stored_body.text_content() == stored
                assert rows.count() == 4 and frame.locator(".part").count() == 0
                # Measure the painted result: both the report headline and the button text must have the expected
                # polarity and contrast against their backgrounds, and the button keeps its shadow.
                def painted(theme: str) -> dict:
                    outside = report.evaluate(_LAYERS)
                    return {part: _painted(frame.locator(node).evaluate(_LAYERS), outside, theme)
                            for part, node in (("headline", "#headline"), ("button", "label[for='report-file']"))}

                def readable(theme: str, label: str) -> None:
                    def right(seen: dict) -> bool:
                        return seen["button"]["shadow"] != "none" and all(
                            each["ink_lighter"] == (theme == "dark") and each["contrast"] >= 4.5 for each in seen.values())

                    def witness(seen: dict) -> str:
                        host = report.evaluate("""iframe => ({choice: document.documentElement.dataset.themeChoice,
                            theme: document.documentElement.dataset.theme,
                            topPrefersLight: matchMedia('(prefers-color-scheme: light)').matches,
                            embeddingScheme: getComputedStyle(iframe).colorScheme})""")
                        return f"{label}: expected readable {theme}; host={host}, report={seen}"

                    deadline = time.monotonic() + 8
                    while not right(seen := painted(theme)):
                        assert time.monotonic() < deadline, witness(seen)
                        page.wait_for_timeout(100)
                    # A first readable paint is not enough if styles then settle to the other scheme.
                    page.wait_for_timeout(300)
                    assert right(seen := painted(theme)), witness(seen)

                appearance = page.locator("html")

                def check_theme(label: str, theme: str) -> None:
                    expect(appearance).to_have_attribute("data-theme", theme)
                    readable(theme, label)
                    assert first in facts.text_content() and stored_body.text_content() == stored and rows.count() == 4
                    expect(frame.locator("#report-result")).to_be_visible()
                    shoot(label)

                frame.locator("h1").scroll_into_view_if_needed()  # the pictures below show the report from its top
                check_theme("2-default-dark", "dark")
                page.emulate_media(color_scheme="light")  # Only System: browser emulation can force the iframe too.
                check_theme("3-device-light", "light")
                page.emulate_media(color_scheme="null")
                ambient = "light" if page.evaluate("() => matchMedia('(prefers-color-scheme: light)').matches") else "dark"
                expect(appearance).to_have_attribute("data-theme", ambient)  # Still System before either manual choice.
                opposite = "dark" if ambient == "light" else "light"
                for label, theme in (("4-manual-opposite", opposite), ("5-manual-return", ambient)):
                    # Same setter as Appearance, without leaving Widgets and replacing the open frame.
                    page.evaluate("(choice) => window.ouroTheme.set(choice)", theme)
                    expect(appearance).to_have_attribute("data-theme-choice", theme)
                    check_theme(f"{label}-{theme}", theme)
                page.evaluate("() => window.ouroTheme.set('system')")
                page.emulate_media(color_scheme="dark")
                expect(page.locator("html")).to_have_attribute("data-theme", "dark")
                inner = viewer_frame()
                asked = rows.nth(1)
                assert "secret masked" in asked.locator("summary").inner_text()
                assert "calls inspect" in rows.nth(2).locator("summary").inner_text()
                assert "inspect" in frame.locator("#request .rows[data-kind='tools']").inner_text()
                asked.locator("summary").click()
                said = asked.locator(".part pre").inner_text()
                assert "Can the Marigold berth at 14?\nShe arrives at 06:30.\n" in said and "<b>now</b>" in said
                assert "sk-secret-model-send-seal" not in said and frame.locator("#request b").count() == 0
                asked.locator("summary").click()
                expect(frame.locator(".part")).to_have_count(0)  # `toggle` arrives after the click returns
                # The same request sent again is another attempt, and it replaces what was shown.
                choose(reports["second"], second, "#attempt-facts")
                assert value("Context mode") == "Not recorded" and value("Host prefix within this task") == "Not recorded"
                assert first not in frame.locator("body").text_content()
                assert stored_body.text_content() == stored
                # The first send failed and its retry was answered: the failed attempt's report is that attempt's,
                # its request included, and its ledger state says how the record was closed, never a price.
                choose(failed_report, failed, "#attempt-facts")
                assert pair.text_content() == ""
                frame.locator("#comparison > summary").click()
                compare(retried_report)
                expect(pair).to_contain_text("identical UTF-8 bytes at the start")
                expect(pair).to_contain_text("temperature")
                frame.locator("#comparison > summary").click()
                assert value("Ledger cache read") == "Not recorded"
                words = facts.text_content()
                assert "The send raised _Rejected · HTTP 400" in words and retried not in words
                assert "Settled — the ledger closed this attempt’s accounting record without a receipt (abandoned)" in words
                assert "Request recorded" in frame.locator("#headline").inner_text()
                assert json.loads(stored_body.text_content())["messages"] == json.loads(stored)["messages"]
                shown = " ".join(frame.locator(node).inner_text() for node in ("#headline", "#explain", "#attempt-facts"))
                for claim in ("final", "charge", "price", "cost", "$", "paid"):
                    assert claim not in shown.lower(), claim
                choose(retried_report, retried, "#attempt-facts")
                assert failed not in frame.locator("body").text_content() and "Failure" not in facts.text_content()
                assert "abandoned" not in facts.text_content()
                # An open record names only its open accounting; what the transport returned stays the Transport's.
                for state, words_of_state in (
                        ("dispatched", "marked this send as started; its accounting record is still open"),
                        ("unresolved", "keeps this attempt’s accounting record open")):
                    choose(gap(body="not_linked", attempt_id=f"open-{state}", ledger_state=state,
                               transport_observation="error_reply_recorded"), f"open-{state}", "#attempt-facts")
                    words = facts.text_content()
                    assert words_of_state in words and "Transport returned an error reply" in words
                    assert "no outcome" not in words and "does not know how" not in words
                # An opened tool call shows its recorded input to the digit, as text or as an object (Anthropic's
                # tool_use), and so do the request's other fields: no number is parsed and written again. A closed
                # row formats no JSON.
                exact = {"model": "gpt-x", "messages": [
                    {"role": "assistant", "content": None, "tool_calls": [{"id": "call_9", "type": "function", "function": {
                        "name": "find", "arguments": '{"record": 9007199254740993, "fresh": []}'}}]},
                    {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_9", "name": "find",
                                                        "input": {"record": 9007199254740993, "fresh": []}}]}],
                    "max_tokens": 9007199254740993}
                inner.evaluate("""() => { window.formatted = 0; const indent = window.indentJson;
                    window.indentJson = (source) => { window.formatted += 1; return indent(source); }; }""")
                choose(json.dumps({"format": "ouroboros.context-report/1", "stored_json": json.dumps(exact), "facts": {
                    "task_id": "report-task", "attempt_id": "exact-attempt", "body": "redacted_host_candidate"}}).encode(
                    "utf-8"), "exact-attempt", "#attempt-facts")
                assert inner.evaluate("() => window.formatted") == 0 and rows.count() == 2
                for index in (0, 1):
                    rows.nth(index).locator("summary").click()
                    assert '"record": 9007199254740993' in rows.nth(index).locator(".part pre").inner_text()
                assert inner.evaluate("() => window.formatted") == 2
                assert "9007199254740993" in frame.locator("#request dl").inner_text()
                # A secret masked in a row marks that row alone, however deep its path runs ("[1]" is not "[10]"); an
                # exclusion of another class marks nothing, and a section that is no list is marked whole.
                paths = ("$.messages[1].content[0].text", "$.messages[10]", "$.tools[1].function.name", "$.system.x")
                masked_rows = {"messages": [{"role": "user", "content": f"m{index}"} for index in range(12)],
                               "tools": [{"name": "a"}, {"name": "b"}], "system": "be brief"}
                choose(json.dumps({"format": "ouroboros.context-report/1", "stored_json": json.dumps(masked_rows), "facts": {
                    "task_id": "report-task", "attempt_id": "masked-rows", "body": "redacted_host_candidate",
                    "host_exclusions": [{"class": "secret_redaction", "path": path} for path in paths]
                                       + [{"class": "provider_native_custody", "path": "$.messages[2]"}]}}).encode("utf-8"),
                       "masked-rows", "#attempt-facts")
                for kind, marked in (("messages", [1, 10]), ("tools", [1]), ("system", [0])):
                    heads = frame.locator(f"#request .rows[data-kind='{kind}'] > .row summary")
                    assert [at for at in range(heads.count()) if "secret masked" in heads.nth(at).inner_text()] == marked, kind
                # A report of a gap says which gap at the top; an earlier /1 report without the later facts opens.
                choose(reports["over_limit"], "larger than this report", "#headline")
                gap_reads_without_opening(f"--max-bytes {len(stored.encode('utf-8'))}")
                assert stored_body.text_content() == "Request body unavailable: over_limit"
                # Its record is named under the data folder, never by this computer's folders, and still in full.
                record = json.loads(frame.locator("#record-body").text_content())["redacted_ref"]
                assert record == json.loads(reports["over_limit"])["facts"]["redacted_ref"]
                assert record["path"].startswith("observability/blobs/")
                assert {"sha256", "size", "kind", "encoding"} <= set(record)
                choose(gap(body="not_linked", file_trace="not_read"), "No saved request is linked", "#headline")
                gap_reads_without_opening("links no saved request")
                assert "earlier-attempt" in facts.text_content()
                assert "The sealed record was not read" in frame.locator("#source-versions").text_content()
                # The first reader wrote `not_recorded` beside `not_linked` without opening any record: its versions are
                # as unknown as when not read, and the report's own fields stay as written. A usage-reported attempt
                # never had a record, so its `not_recorded` keeps saying that none were recorded.
                unread = frame.locator("#source-versions").text_content()
                choose(gap(body="not_linked", file_trace="not_recorded", attempt_id="unlinked-first-reader"),
                       "unlinked-first-reader", "#attempt-facts")
                assert frame.locator("#source-versions").text_content() == unread
                assert "No source versions were recorded" not in frame.locator("#source-versions").text_content()
                written = json.loads(frame.locator("#record-body").text_content())
                assert (written["body"], written["file_trace"]) == ("not_linked", "not_recorded")
                choose(gap(body="receipt_only", file_trace="not_recorded", attempt_id="usage-reported"),
                       "usage-reported", "#attempt-facts")
                assert "No source versions were recorded" in frame.locator("#source-versions").text_content()
                for facts_of_gap, headline, words in (
                        ({"body": "manifest_over_limit", "recorded_bytes": 2258277, "max_bytes": 2097152,
                          "retry_max_bytes": 2258277}, "sealed record is larger",
                         "--max-bytes 2258277 to open it; if the request itself is larger, that run names"),
                        ({"body": "evidence_unavailable", "evidence_reason": "busy"}, "could not be located just now",
                         "run the reader again later"),
                        ({"body": "receipt_only"}, "recorded from a usage report", "No request was saved for it"),
                        # The allowance admits the request and its sealed record: the larger of the two.
                        ({"body": "over_page_limit", "recorded_bytes": 52428800, "page_limit_bytes": 50331648,
                          "retry_max_bytes": 60000000}, "larger than this page opens",
                         _reader_command("report-task", "earlier-attempt", 60000000)),
                        # A report too large names the request's size and the report's, or a floor when never built.
                        ({"body": "over_page_limit", "recorded_bytes": 40000000, "report_bytes": 52500000,
                          "page_limit_bytes": 50331648, "retry_max_bytes": 40000000}, "larger than this page opens",
                         "The request is 38.2 MiB; its report would be 50.1 MiB, and Beautiful Context opens"),
                        ({"body": "over_page_limit", "recorded_bytes": 52428800, "report_bytes_at_least": 52428800,
                          "page_limit_bytes": 50331648, "retry_max_bytes": 52428800}, "larger than this page opens",
                         "its report would be at least 50 MiB"),
                        # IDs that are no plain word never enter the command; the facts still show them as text.
                        ({"body": "over_page_limit", "recorded_bytes": 52428800, "page_limit_bytes": 50331648,
                          "retry_max_bytes": 52428800, "task_id": "x; echo owned", "attempt_id": "-x|cat"},
                         "larger than this page opens", _reader_command("TASK_ID", "ATTEMPT_ID", 52428800)),
                        ({"body": "not_one_attempt", "ledger_kind": "subscription_session"},
                         "names no single model request", "List the task’s attempts")):
                    choose(gap(**facts_of_gap), headline, "#headline")
                    gap_reads_without_opening(words)
                    if facts_of_gap.get("task_id"):
                        assert "echo owned" not in frame.locator("#explain").inner_text()
                        assert "x; echo owned" in facts.text_content() and "-x|cat" in facts.text_content()
                assert "Ledger row" in facts.text_content() and "subscription_session" in facts.text_content()
                assert "Ledger state" not in facts.text_content() and "Transport" not in facts.text_content()
                # A gap keeps the failure's type and the file trace; a trace the reader did not open says so.
                trace = [{"source_path": "prompts/SYSTEM.md", "source_sha256": "ab" * 32,
                          "contribution": "leading_system_prefix", "text_sha256": "cd" * 32}]
                choose(gap(body="over_page_limit", recorded_bytes=52428800, report_bytes_at_least=52428800,
                           page_limit_bytes=50331648, retry_max_bytes=52428800, failure={
                               "stage": "raised_exception", "exception_type": "_Rejected", "provider_status_code": 400},
                           file_trace=trace), "larger than this page opens", "#headline")
                assert "The send raised _Rejected · HTTP 400" in facts.text_content()
                assert frame.locator("#source-versions").text_content().startswith("prompts/SYSTEM.md · Base instructions")
                choose(gap(body="manifest_over_limit", recorded_bytes=2258277, max_bytes=2097152,
                           retry_max_bytes=2258277, file_trace="not_read"), "sealed record is larger", "#headline")
                assert "The sealed record was not read" in frame.locator("#source-versions").text_content()
                # Design reference groups show the recorded form without overclaiming when retained_source is absent.
                retained_path = "source_handles/context_checkpoints/host-design_bible.md"
                choose(gap(body="not_linked", attempt_id="design-bible-forms", assembly_source_map=[
                    {"block": 0, "status": "located", "groups_omitted": 0, "chars": 900, "groups": [
                        {"family": "design_bible", "form": "rendered", "start": 0, "chars": 600},
                        {"family": "design_bible", "form": "shorter", "reason": "physical_window",
                         "retained_source": {"path": retained_path, "sha256": "ab" * 32}, "start": 600, "chars": 300},
                    ]},
                    {"block": 1, "status": "position_unverified", "groups_omitted": 0, "chars": 162, "groups": [
                        {"family": "design_bible", "form": "shorter", "reason": "mode_or_start_allowance",
                         "start": 0, "chars": 120},
                        {"family": "design_bible", "form": "unknown_form", "start": 120, "chars": 42},
                    ]},
                ]), "design-bible-forms", "#attempt-facts")
                expect(frame.locator("#source-versions")).to_be_visible()
                design_lines = frame.locator("#source-versions").inner_text().splitlines()
                assert design_lines == [
                    "Design reference · Design bible section in full; rest as table of contents · 600 characters",
                    "Design reference · table of contents only; Design bible retained · to fit the model window · 300 characters",
                    "Design reference · shorter form · to fit the starting allowance · position not verified · 120 characters",
                    "Design reference · form not recorded · position not verified · 42 characters",
                ], design_lines
                assert retained_path not in "\n".join(design_lines)
                assert retained_path in frame.locator("#source-technical").text_content()
                # A file the page cannot open says how to read that request elsewhere, not only that it is too large.
                large = tmp_path / "large-report.json"
                with large.open("wb") as handle:
                    handle.truncate(48 * 1024 * 1024 + 1)
                choose(str(large), "more than the 48 MiB this page opens")
                assert _reader_command("TASK_ID", "ATTEMPT_ID", 48 * 1024 * 1024 + 1).search(
                    frame.locator("#report-status").inner_text())
                assert frame.locator("#report-result").is_hidden()
                # A selection that is no report removes the result instead of leaving the last one in view.
                for content, words in ((b"", "the file is empty"), (reports["first"][:200], "not complete JSON"),
                                       (b'{"stored_json": null}', "another kind of document"),
                                       (None, "No report selected.")):
                    choose(content, words)
                    assert content is None or "Cannot read this Beautiful Context JSON report" in frame.locator(
                        "#report-status").inner_text()
                    assert frame.locator("#report-result").is_hidden() and facts.text_content() == ""
                    assert stored_body.text_content() == ""
                # The lease is a credential: a failure shows the console's words without it.
                unexpected = [_redacted(text) for text in unexpected_console_errors(errors, failed_loads)]
                assert unexpected == [], unexpected
                # Nothing left the frame: no message to its parent (a probe from the frame itself reaches the same
                # recorder), no request but its own page load, and no socket frame with the reports' words.
                assert page.evaluate("() => window.__reportMessages") == []
                viewer_frame().evaluate("() => window.parent.postMessage('probe', '*')")
                page.wait_for_function("() => window.__reportMessages.length === 1")
                assert page.evaluate("() => window.__reportMessages") == ["probe"]

                # A reload keeps the card: the board reads its list again and mounts the viewer once more.
                phase[0] = "reload"
                page.reload(wait_until="domcontentloaded", timeout=30_000)
                page.wait_for_selector("#page-widgets.active")
                expect(card).to_be_visible()
                expect(report).to_be_visible()
                expect(frame.locator("#report-file")).to_be_attached()
                viewer_frame()
                assert page.evaluate("() => window.__reportMessages") == []
                page.evaluate("() => { window.__boardKept = true; }")

                # Turned off with the switch on its Installed card, the card leaves the board that stays open ...
                phase[0] = "disable"
                extensions.locator('#page-skills [data-extensions-open="installed"]').click()
                switch = extensions.locator(f'.skills-installed-card[data-skill="{skill}"] input.skills-toggle')
                press = extensions.locator(f'.skills-installed-card[data-skill="{skill}"] label.ui-switch')
                expect(switch).to_be_checked()
                expect(switch).to_be_enabled()
                press.click()
                expect(card).to_have_count(0)
                expect(report).to_have_count(0)
                assert page.evaluate("() => window.__boardKept === true")
                board_is_read()
                shoot("6-board-off-again", page.locator("#page-widgets"))
                # ... is still gone after a reload ...
                phase[0] = "gone"
                with page.expect_response(listed):
                    page.reload(wait_until="domcontentloaded", timeout=30_000)
                page.wait_for_selector("#page-widgets.active")
                board_is_read()
                expect(card).to_have_count(0)
                expect(report).to_have_count(0)
                page.evaluate("() => { window.__boardKept = true; }")
                # ... and comes back to that board, live, when the same switch turns the skill on again.
                phase[0] = "enable"
                expect(switch).not_to_be_checked()
                expect(switch).to_be_enabled()
                press.click()
                expect(card).to_be_visible()
                expect(frame.locator("#report-file")).to_be_attached()
                assert page.evaluate("() => window.__boardKept === true")
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise
    # The viewer's page was loaded when the open board mounted it live after Install, when the reload mounted it again
    # and when the switch turned it on again: nowhere else.
    loads = [(each["phase"], each["navigation"]) for each in requests
             if each["path"].startswith("/api/widget-resources/") and each["path"].endswith("/viewer")]
    assert loads == [("install", True), ("reload", True), ("enable", True)], loads
    outside = [(each["phase"], each["path"]) for each in requests
               if "/api/widget-resources/" in each["frame"] and not each["navigation"]]
    assert not outside, outside
    leaked = [(each["phase"], each["path"]) for each in requests if any(word in each["url"] + each["data"] for word in private)]
    assert not leaked, leaked
    assert sockets and sockets[0]["phase"] == "boot", [each["phase"] for each in sockets]  # the page's own socket, from its start
    assert len({each["url"] for each in sockets}) == 1, len({each["url"] for each in sockets})
    assert not [each["phase"] for each in sockets for sent in each["sent"] if any(word in sent for word in private)]


@pytest.mark.ui_browser
@pytest.mark.parametrize("browser_name", ("chromium", "webkit"))
def test_private_report_finds_late_copies_and_clears_old_details_before_read(browser_name, tmp_path):
    """Real report HTML and controls, replacing the native archive only (no Windows claim)."""
    playwright = pytest.importorskip("playwright.sync_api")
    source = pathlib.Path(__file__).resolve().parents[1] / "web" / "context-report.html"
    document = tmp_path / "archive.html"
    document.write_text(source.read_text(encoding="utf-8").replace('<html lang="en">',
                        '<html lang="en" data-private-archive>', 1), encoding="utf-8")
    with playwright.sync_playwright() as pw:
        try:
            browser = getattr(pw, browser_name).launch(headless=True)
        except playwright.Error as exc:
            if "Executable doesn't exist" in str(exc):
                pytest.skip(str(exc))
            raise
        try:
            page = browser.new_page(viewport={"width": 1030, "height": 760}, color_scheme="dark")
            page.route("**/*", lambda route: route.continue_() if route.request.url == document.as_uri() else route.abort())
            page.add_init_script("""(() => {
              const ok = value => ({ok: true, value});
              const document = (id, read) => JSON.stringify({format: 'ouroboros.context-report/1',
                stored_json: read ? JSON.stringify({messages: [{role: 'user', content: id}]}) : null,
                facts: {task_id: 'fixture-only', attempt_id: id, body: read ? 'redacted_host_candidate' : 'not_linked',
                  assembly_source_map: read ? [{status: 'located', observation: `assembly-for-${id}`,
                    groups: [{family: 'other_conversation', form: 'rendered', chars: 8}]}] : []}});
              window.reportCalls = [];
              window.pywebview = {api: {
                rooms: () => window.holdRefresh ? new Promise(resolve => { window.finishRefresh = resolve; })
                  : ok([{id: 1, name: 'Tide charts'}, {id: 2, name: 'Library hours'}]),
                requests: room => ok(Array.from({length: room === 1 ? 40 : 1}, (_, index) => ({
                  attempt_id: `${room === 1 ? 'tide' : 'library'}-${index}`, model: 'fixture-model'}))),
                report: id => {
                  window.reportCalls.push(id);
                  if (id === 'library-0') return new Promise(resolve => { window.finishRead = resolve; });
                  return ok(document(id, id === 'tide-6' || id === 'tide-39'));
                },
              }};
            })();""")
            page.goto(document.as_uri(), wait_until="domcontentloaded")
            page.evaluate("() => window.dispatchEvent(new Event('pywebviewready'))")
            expect = playwright.expect
            status = page.locator("#report-status")
            technical = page.locator("#source-technical")
            attempts = page.locator("#archive-attempt")
            room = page.locator("#archive-room")
            compare = page.locator("#archive-compare")
            expect(attempts).to_have_value("tide-6")
            expect(page.locator("#comparison")).to_be_hidden()
            expect(technical).to_contain_text("assembly-for-tide-6", use_inner_text=False)
            expect(page.locator("#privacy")).to_be_visible()
            expect(page.locator("#privacy")).to_contain_text("Keep this file on this computer")
            assert page.evaluate("() => window.reportCalls") == [f"tide-{n}" for n in range(7)]
            compare.click()
            expect(page.locator("#comparison-status")).to_contain_text("Selected request compared with Earlier 39")
            expect(page.locator("#comparison")).to_be_visible()
            assert page.evaluate("() => window.reportCalls") == [f"tide-{n}" for n in range(40)]

            def clean():
                expect(page.locator("#report-result")).to_be_hidden()
                assert page.locator("#comparison").get_attribute("hidden") is not None
                assert "assembly-for-tide" not in page.locator("body").text_content()
                assert technical.text_content() == ""
                assert page.locator("#comparison-facts").text_content() == ""
                assert page.locator("#stored-body").text_content() == ""
                expect(compare).to_be_disabled()

            room.select_option("2")
            page.wait_for_function("() => typeof window.finishRead === 'function'")
            expect(status).to_have_text("Opening the saved request…")
            clean()
            page.evaluate("() => window.finishRead({ok: false})")
            expect(status).to_contain_text("No readable copy")
            expect(attempts).to_have_value("library-0")
            clean()

            room.select_option("1")
            expect(technical).to_contain_text("assembly-for-tide-6", use_inner_text=False)
            page.evaluate("() => { window.holdRefresh = true; }")
            page.locator("#archive-reload").click()
            page.wait_for_function("() => typeof window.finishRefresh === 'function'")
            expect(attempts.locator("option")).to_have_count(0)
            clean()
            page.evaluate("() => window.finishRefresh({ok: false, error: 'Fixture refresh refused'})")
            expect(status).to_have_text("Fixture refresh refused")
            clean()
        finally:
            browser.close()
