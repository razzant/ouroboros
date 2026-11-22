from __future__ import annotations

import json
import os
import pathlib
import textwrap
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
    const report = document.querySelector("iframe[title='Beautiful Context']");
    if (report && event.source === report.contentWindow) window.__reportMessages.push(String(event.data));
  }, true);
})();"""


@pytest.mark.ui_browser
@pytest.mark.parametrize("browser_name", ("chromium", "webkit"))
def test_ui_smoke_beautiful_context_reads_the_selected_attempt(
    direct_server_with_data, browser_name, tmp_path, monkeypatch, capsys,
):
    """Widgets opens one local report at a time and shows that attempt, its messages and tools, or its gap.

    The reports are the real reader's over real sealed attempts. A stale CSP hash, a body shown under another
    attempt's facts, an earlier result left under a failed selection, text drawn as markup, a request held
    twice in the frame, a gap explained only inside a closed disclosure or a ledger state read as a final
    charge each fail here. So does a frame that keeps its old palette when the device's theme changes while a
    report is open, or that sends anything out: a message to its parent, a request or a WebSocket frame, before or
    after a file is chosen (the frame's own page load is the observers' positive control).
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
    private = ("Marigold", "fenders", first, second, failed, retried)  # what no request or socket frame may carry
    selector = "iframe[title='Beautiful Context']"
    url = direct_server_with_data["url"]
    evidence = pathlib.Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR") or tmp_path / "evidence") / "beautiful-context"
    evidence.mkdir(parents=True, exist_ok=True)
    phase, requests, sockets = ["boot"], [], []

    def on_request(request):
        try:
            where = request.frame.url
        except PlaywrightError:  # a request of a frame already gone
            where = ""
        requests.append({"phase": phase[0], "path": urllib.parse.urlsplit(request.url).path, "url": request.url,
                         "frame": where, "navigation": request.is_navigation_request(),
                         "data": (request.post_data_buffer or b"").decode("utf-8", "replace")})

    def on_socket(socket):
        record = {"phase": phase[0], "url": socket.url, "sent": []}
        socket.on("framesent", lambda payload: record["sent"].append(str(payload)))
        sockets.append(record)

    try:
        with sync_playwright() as pw:
            browser = getattr(pw, browser_name).launch(headless=True)
            context = browser.new_context(viewport={"width": 1280, "height": 800}, color_scheme="dark")
            context.add_init_script(_REPORT_MESSAGES)
            context.on("request", on_request)
            page = context.new_page()
            page.on("websocket", on_socket)
            page.set_default_timeout(15_000)
            errors, failed_loads = record_console_errors(page), record_failed_loads(page)
            try:
                with page.expect_websocket():  # the page's own socket, observed from its start
                    page.goto(f"{url}/#widgets", wait_until="domcontentloaded", timeout=30_000)
                page.wait_for_selector("#page-widgets.active")
                page.evaluate("() => window.ouroTheme.set('system')")  # Appearance follows the device, as installed
                expect(page.locator("html")).to_have_attribute("data-theme", "dark")
                assert page.locator(selector).count() == 0
                phase[0] = "open"
                page.get_by_role("button", name="Beautiful Context").click()
                report = page.locator(selector)
                # Opaque and bridge-free: scripts only, its own address, and the palette as the one thing told.
                assert report.get_attribute("sandbox") == "allow-scripts"
                assert report.get_attribute("src") == "/static/context-report.html"
                frame = page.frame_locator(selector)
                expect(frame.locator("html")).to_have_attribute("data-theme", "dark")
                facts = frame.locator("#attempt-facts")
                rows = frame.locator("#request .rows[data-kind='messages'] > .row")
                stored_body = frame.locator("#stored-body")
                mount = page.locator("[data-context-report-mount]")

                def choose(content, shown: str, where: str = "#report-status") -> None:
                    # Through the picker a person uses: the control must take the click itself.
                    with page.expect_file_chooser() as picker:
                        frame.locator("#report-file").click()
                    picker.value.set_files([] if content is None else content if isinstance(content, str) else {
                        "name": "report.json", "mimeType": "application/json", "buffer": content})
                    frame.locator(where).get_by_text(shown).wait_for()

                def gap_reads_without_opening(words: str) -> None:
                    # What a person sees with every disclosure closed: the gap's headline and its explanation; the
                    # stored JSON, where a gap also names itself, stays folded and is no evidence of either.
                    assert frame.locator("#report-result details[open]").count() == 0
                    expect(frame.locator("#headline")).to_be_visible()
                    expect(frame.locator("#explain")).to_be_visible()
                    assert words in frame.locator("#explain").inner_text()
                    expect(stored_body).to_be_hidden()
                    assert rows.count() == 0 and frame.locator("#privacy").is_hidden()

                def gap(**facts_of_gap) -> bytes:
                    return json.dumps({"format": "ouroboros.context-report/1", "stored_json": None, "facts": {
                        "task_id": "report-task", "attempt_id": "earlier-attempt", **facts_of_gap}}).encode("utf-8")

                def shoot(name: str) -> None:
                    # WebKit's screenshot puts a stylesheet into the frame, which the frame's policy refuses: that one
                    # console line per picture is the screenshot's, never the page's, and is set aside here.
                    seen = len(errors)
                    mount.screenshot(path=str(evidence / f"{browser_name}-{name}.png"), caret="initial")
                    page.wait_for_timeout(200)
                    errors[seen:] = [each for each in errors[seen:] if browser_name != "webkit"
                                     or each[1] != "/static/context-report.html"
                                     or not each[0].startswith("Refused to apply a stylesheet because its hash")]

                def inks() -> tuple:
                    # The host's text on the card the frame lies on, and the frame's own headline, each as painted.
                    host = page.evaluate("""() => {
                        const probe = document.createElement('span');
                        probe.style.color = 'var(--text-primary)';
                        document.querySelector('[data-context-report-mount]').append(probe);
                        const ink = getComputedStyle(probe).color;
                        probe.remove();
                        return [document.documentElement.dataset.theme, ink];
                    }""")
                    inside = frame.locator("#headline").evaluate(
                        "(node) => [document.documentElement.dataset.theme, getComputedStyle(node).color]")
                    return host, inside

                phase[0] = "select"
                choose(reports["first"], first, "#attempt-facts")
                assert "receipt not confirmed" in facts.inner_text() and second not in facts.inner_text()
                assert "Request recorded" in frame.locator("#headline").inner_text()
                assert "Private text can remain" in frame.locator("#privacy").inner_text()
                # The request is in the frame once, whole and literal; the overview names every message and
                # tool, and a row's text exists only while that row is open.
                assert stored_body.text_content() == stored
                assert rows.count() == 4 and frame.locator(".part").count() == 0
                # The device turns light and back while the report is open: the frame takes each palette from its
                # card, its text reads as the card's own text, and the report stays where it was.
                host, inside = inks()
                assert inside == host and host[0] == "dark", (host, inside)
                shoot("1-dark-opened")
                for step, scheme in enumerate(("light", "dark"), 2):
                    page.emulate_media(color_scheme=scheme)
                    expect(page.locator("html")).to_have_attribute("data-theme", scheme)
                    expect(frame.locator("html")).to_have_attribute("data-theme", scheme)
                    host, inside = inks()
                    assert inside == host and host[0] == scheme, (scheme, host, inside)
                    assert first in facts.inner_text() and stored_body.text_content() == stored and rows.count() == 4
                    expect(frame.locator("#report-result")).to_be_visible()
                    # Its one pressable control stands on the rest shadow in either theme (DESIGN principle 4).
                    assert frame.locator(".button").evaluate("(node) => getComputedStyle(node).boxShadow") != "none"
                    shoot(f"{step}-{scheme}-switched")
                (inner,) = [each for each in page.frames if each.url.endswith("/static/context-report.html")]
                inner.evaluate("""() => window.postMessage({type: 'ouro-context-report-theme', name: 'light',
                                                           tokens: {'--ouro-text': 'rgb(255, 0, 0)'}}, '*')""")
                page.wait_for_timeout(300)  # a message the frame would apply is applied within one task
                assert inks()[1] == host, "a palette message not from the parent window must change nothing"
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
                assert first not in frame.locator("body").text_content()
                assert stored_body.text_content() == stored
                # The first send failed and its retry was answered: the failed attempt's report is that attempt's,
                # its request included, and its ledger state says how the record was closed, never a price.
                choose(failed_report, failed, "#attempt-facts")
                words = facts.inner_text()
                assert "The send raised _Rejected · HTTP 400" in words and retried not in frame.locator("body").text_content()
                assert "Settled — the ledger closed this attempt’s accounting record without a receipt (abandoned)" in words
                assert "Request recorded" in frame.locator("#headline").inner_text()
                assert json.loads(stored_body.text_content())["messages"] == json.loads(stored)["messages"]
                shown = " ".join(frame.locator(node).inner_text() for node in ("#headline", "#explain", "#attempt-facts"))
                for claim in ("final", "charge", "price", "cost", "$", "paid"):
                    assert claim not in shown.lower(), claim
                choose(retried_report, retried, "#attempt-facts")
                assert failed not in frame.locator("body").text_content() and "Failure" not in facts.inner_text()
                assert "abandoned" not in facts.inner_text()
                # An open record names only its open accounting; what the transport returned stays the Transport's.
                for state, words_of_state in (
                        ("dispatched", "marked this send as started; its accounting record is still open"),
                        ("unresolved", "keeps this attempt’s accounting record open")):
                    choose(gap(body="not_linked", attempt_id=f"open-{state}", ledger_state=state,
                               transport_observation="error_reply_recorded"), f"open-{state}", "#attempt-facts")
                    words = facts.inner_text()
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
                assert "earlier-attempt" in facts.inner_text()
                assert frame.locator("#source-versions").text_content().startswith("Not read: this report did not open")
                # The first reader wrote `not_recorded` beside `not_linked` without opening any record: its versions are
                # as unknown as when not read, and the report's own fields stay as written. A usage-reported attempt
                # never had a record, so its `not_recorded` keeps saying that none were recorded.
                unread = frame.locator("#source-versions").text_content()
                choose(gap(body="not_linked", file_trace="not_recorded", attempt_id="unlinked-first-reader"),
                       "unlinked-first-reader", "#attempt-facts")
                assert frame.locator("#source-versions").text_content() == unread
                assert "No supported source versions were recorded" not in frame.locator("#source-versions").text_content()
                written = json.loads(frame.locator("#record-body").text_content())
                assert (written["body"], written["file_trace"]) == ("not_linked", "not_recorded")
                choose(gap(body="receipt_only", file_trace="not_recorded", attempt_id="usage-reported"),
                       "usage-reported", "#attempt-facts")
                assert frame.locator("#source-versions").text_content().startswith("No supported source versions were recorded")
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
                         "python -m ouroboros.context_attempt_reader --data-root DATA report-task --attempt "
                         "earlier-attempt --max-bytes 60000000 > request.json"),
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
                         "larger than this page opens", "--data-root DATA TASK_ID --attempt ATTEMPT_ID --max-bytes "
                         "52428800 > request.json"),
                        ({"body": "not_one_attempt", "ledger_kind": "subscription_session"},
                         "names no single model request", "List the task’s attempts")):
                    choose(gap(**facts_of_gap), headline, "#headline")
                    gap_reads_without_opening(words)
                    if facts_of_gap.get("task_id"):
                        assert "echo owned" not in frame.locator("#explain").inner_text()
                        assert "x; echo owned" in facts.inner_text() and "-x|cat" in facts.inner_text()
                assert "Ledger row" in facts.inner_text() and "subscription_session" in facts.inner_text()
                assert "Ledger state" not in facts.inner_text() and "Transport" not in facts.inner_text()
                # A gap keeps the failure's type and the file trace; a trace the reader did not open says so.
                trace = [{"source_path": "prompts/SYSTEM.md", "source_sha256": "ab" * 32,
                          "contribution": "leading_system_prefix", "text_sha256": "cd" * 32}]
                choose(gap(body="over_page_limit", recorded_bytes=52428800, report_bytes_at_least=52428800,
                           page_limit_bytes=50331648, retry_max_bytes=52428800, failure={
                               "stage": "raised_exception", "exception_type": "_Rejected", "provider_status_code": 400},
                           file_trace=trace), "larger than this page opens", "#headline")
                assert "The send raised _Rejected · HTTP 400" in facts.inner_text()
                assert frame.locator("#source-versions").text_content().startswith("prompts/SYSTEM.md\nFile SHA-256: ab")
                choose(gap(body="manifest_over_limit", recorded_bytes=2258277, max_bytes=2097152,
                           retry_max_bytes=2258277, file_trace="not_read"), "sealed record is larger", "#headline")
                assert frame.locator("#source-versions").text_content().startswith("Not read: this report did not open")
                # A file the page cannot open says how to read that request elsewhere, not only that it is too large.
                large = tmp_path / "large-report.json"
                with large.open("wb") as handle:
                    handle.truncate(48 * 1024 * 1024 + 1)
                choose(str(large), "more than the 48 MiB this page opens")
                assert (f"without --report: python -m ouroboros.context_attempt_reader --data-root DATA TASK_ID "
                        f"--attempt ATTEMPT_ID --max-bytes {48 * 1024 * 1024 + 1} > request.json") in frame.locator(
                    "#report-status").inner_text()
                assert frame.locator("#report-result").is_hidden()
                # A selection that is no report removes the result instead of leaving the last one in view.
                for content, words in ((b"", "the file is empty"), (reports["first"][:200], "not complete JSON"),
                                       (b'{"stored_json": null}', "another kind of document"),
                                       (None, "No report selected.")):
                    choose(content, words)
                    assert content is None or "Cannot read this Beautiful Context JSON report" in frame.locator(
                        "#report-status").inner_text()
                    assert frame.locator("#report-result").is_hidden() and facts.inner_text() == ""
                    assert stored_body.text_content() == ""
                assert unexpected_console_errors(errors, failed_loads) == [], errors
                # Nothing left the frame: no message to its parent (a probe from the frame itself reaches the same
                # recorder), no request but its own page load, and no socket frame with the reports' words.
                assert page.evaluate("() => window.__reportMessages") == []
                (inner,) = [each for each in page.frames if each.url.endswith("/static/context-report.html")]
                inner.evaluate("() => window.parent.postMessage('probe', '*')")
                page.wait_for_function("() => window.__reportMessages.length === 1")
                assert page.evaluate("() => window.__reportMessages") == ["probe"]
                phase[0] = "leave"
                page.locator("nav#primary-sidebar [data-nav-page]:not([data-nav-page='widgets'])").first.click()
                report.wait_for(state="detached")
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise
    loads = [each for each in requests if each["path"] == "/static/context-report.html"]
    assert [(each["phase"], each["navigation"]) for each in loads] == [("open", True)], loads
    assert not [each for each in requests if "context-report.html" in each["frame"] and not each["navigation"]]
    assert not [each for each in requests if any(word in each["url"] + each["data"] for word in private)]
    assert sockets and sockets[0]["phase"] == "boot", sockets  # the page's own socket, observed from its start
    assert {each["url"] for each in sockets} == {sockets[0]["url"]}, sockets
    assert not [each for each in sockets for sent in each["sent"] if any(word in sent for word in private)]
