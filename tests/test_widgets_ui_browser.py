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
    """Install reviewed module tabs that exercise host geometry and teardown."""
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
    addEventListener('message', (event) => {
        const message = event.data || {};
        if (!['ouro-widget-resize', 'fixture-geometry'].includes(message.type)) return;
        const frame = [...document.querySelectorAll('[data-widget-key] iframe')]
            .find((candidate) => candidate.contentWindow === event.source);
        record(message.type === 'ouro-widget-resize' ? 'message' : 'child', frame, {
            heightValue: message.height,
            child: message.sample || null,
        });
    }, true);
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
                assert 320 <= small_height < 700, small_height

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
                assert initial_heights["floor"] == 320, initial_heights
                assert 320 <= initial_heights["wide"] < 700, initial_heights
                for tab_id in frames:
                    assert page.locator(frame_selector(tab_id)).get_attribute("scrolling") is None
                auto_initial = child_metrics(frames["auto"])
                assert auto_initial["scrollbarWidth"] >= 0
                assert auto_initial["htmlOverflowY"] == "hidden", auto_initial
                assert auto_initial["bodyOverflowY"] == "hidden", auto_initial
                assert auto_initial["htmlOverflowX"] != "hidden", auto_initial
                assert auto_initial["bodyOverflowX"] == "hidden", auto_initial

                floor_initial = child_metrics(frames["floor"])
                assert floor_initial["htmlOverflowY"] != "hidden", floor_initial
                assert floor_initial["bodyOverflowY"] != "hidden", floor_initial

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
