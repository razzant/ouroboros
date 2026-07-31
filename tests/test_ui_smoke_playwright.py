from __future__ import annotations

import json
import os
import pathlib
import socket
import subprocess
import sys
import textwrap
import time
import urllib.request

import pytest

from tests.fixtures_mock_llm import MockLLMServer

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_health(url: str, timeout_sec: int = 30) -> None:
    deadline = time.time() + timeout_sec
    last = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/api/health", timeout=2) as resp:  # noqa: S310 - local test server
                if resp.status == 200:
                    return
        except Exception as exc:
            last = str(exc)
        time.sleep(0.5)
    raise RuntimeError(f"server did not become healthy: {last}")


def _wait_supervisor_ready(url: str, timeout_sec: int = 45) -> None:
    """Wait past port readiness until the direct test runtime can serve history."""
    deadline = time.time() + timeout_sec
    last = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{url}/api/state", timeout=2) as resp:  # noqa: S310 - local test server
                payload = json.loads(resp.read().decode("utf-8"))
                if payload.get("supervisor_ready") is True:
                    return
        except Exception as exc:
            last = str(exc)
        time.sleep(0.25)
    raise RuntimeError(f"server supervisor did not become ready: {last}")


def _run_core_ui_assertions(url: str) -> None:
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 390, "height": 844})
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                page.wait_for_selector("#page-chat", timeout=30_000)
                assert page.locator("#page-chat").count() == 1
                page.evaluate(
                    """() => {
                        const transfer = new DataTransfer();
                        transfer.items.add(new File(['hello'], 'drop-check.txt', { type: 'text/plain' }));
                        const target = document.querySelector('#page-chat');
                        for (const type of ['dragenter', 'dragover', 'drop']) {
                            target.dispatchEvent(new DragEvent(type, {
                                bubbles: true,
                                cancelable: true,
                                dataTransfer: transfer,
                            }));
                        }
                    }"""
                )
                page.wait_for_selector("#chat-attachment-preview.visible .attach-badge", timeout=5_000)
                assert "drop-check.txt" in page.locator("#chat-attachment-preview").inner_text(timeout=5_000)
                input_area_class = page.locator("#chat-input-area").get_attribute("class", timeout=5_000) or ""
                assert "drag-active" not in input_area_class
                # v6.32.0 redesign: nav rows use data-nav-page (the old data-page
                # rail is gone), and on this mobile viewport (390px) the sidebar is
                # a drawer behind the header toggle — open it before navigating.
                page.click('[data-mobile-nav-toggle]')
                page.wait_for_selector('#primary-sidebar.open', timeout=5_000)
                page.click('[data-nav-page="dashboard"]')
                page.click('[data-dashboard-tab="updates"]')
                assert page.locator("#updates-summary").count() == 1
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise


@pytest.mark.ui_browser
def test_ui_projects_sidebar_unread_and_keyboard_menu(direct_server_with_data):
    """Projects stays compact, paint-ACKs unread, and exposes a real keyboard menu."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    from ouroboros.projects_registry import create_project, increment_project_visible_revision

    url = direct_server_with_data["url"]
    data_dir = direct_server_with_data["data_dir"]
    create_project(data_dir, "alpha", name="Alpha project")
    increment_project_visible_revision(data_dir, project_id="alpha")

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                row = page.locator('.nav-project-row[data-project-id="alpha"]')
                row.wait_for(state="visible", timeout=30_000)
                add = page.locator("#nav-projects-add")
                assert add.is_visible()
                assert add.get_attribute("aria-label") == "New project"
                assert page.locator("#nav-projects-count").inner_text() == "1"
                assert row.locator(".nav-unread-dot").count() == 1

                # A real room paint advances the monotonic cursor and clears unread.
                row.click()
                page.wait_for_selector('#project-panel:not([hidden])', timeout=30_000)
                page.wait_for_function(
                    "() => document.querySelector('#nav-projects-count')?.textContent === ''",
                    timeout=30_000,
                )
                page.click("#project-panel-close")

                kebab = page.locator('.nav-project-kebab[aria-label="Actions for Alpha project"]')
                kebab.focus()
                kebab.press("Enter")
                menu = page.locator('.project-row-menu[role="menu"]')
                menu.wait_for(state="visible", timeout=5_000)
                assert page.locator(':focus').get_attribute("data-prm") == "rename"
                box = menu.bounding_box()
                assert box is not None
                assert box["x"] >= 0 and box["y"] >= 0
                assert box["x"] + box["width"] <= 1280
                assert box["y"] + box["height"] <= 800
                page.keyboard.press("Escape")
                assert menu.count() == 0
                assert page.locator(':focus').get_attribute("aria-label") == "Actions for Alpha project"

                # Collapse is keyboard-operable while the add action stays available.
                toggle = page.locator("#nav-projects-toggle")
                toggle.focus()
                toggle.press("Space")
                assert toggle.get_attribute("aria-expanded") == "false"
                assert page.locator("#nav-projects-list").is_hidden()
                assert add.is_visible()
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise


def _run_docker_ui_assertions(url: str) -> None:
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 390, "height": 844})
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                if page.locator("#onboarding-overlay").count():
                    overlay_text = page.locator("#onboarding-overlay").inner_text(timeout=5_000)
                    if "Ouroboros" in overlay_text:
                        return
                page.wait_for_selector("#page-chat", timeout=30_000)
                assert page.locator("#page-chat").count() == 1
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise


@pytest.fixture()
def direct_server_with_data(tmp_path):
    if os.environ.get("OUROBOROS_RUN_UI_SMOKE") != "1":
        pytest.skip("set OUROBOROS_RUN_UI_SMOKE=1 to run browser UI smoke")
    with MockLLMServer() as llm:
        port = _free_port()
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True)
        model = "openai-compatible::mock-model"
        (data_dir / "settings.json").write_text(
            json.dumps(
                {
                    "OPENAI_COMPATIBLE_API_KEY": "ui-smoke-key",
                    "OPENAI_COMPATIBLE_BASE_URL": llm.base_url,
                    "OUROBOROS_MODEL": model,
                    "OUROBOROS_MODEL_HEAVY": model,
                    "OUROBOROS_MODEL_LIGHT": model,
                    "OUROBOROS_MODEL_FALLBACKS": model,
                    # Every smoke case is single-task or deterministic log replay;
                    # a ten-process default pool adds only process churn and makes
                    # sequential browser history fetches flaky on shared hosts.
                    "OUROBOROS_MAX_WORKERS": 1,
                    "OUROBOROS_RUNTIME_MODE": "light",
                }
            ),
            encoding="utf-8",
        )
        env = {
            **os.environ,
            "OUROBOROS_APP_ROOT": str(tmp_path),
            "OUROBOROS_DATA_DIR": str(data_dir),
            "OUROBOROS_SETTINGS_PATH": str(data_dir / "settings.json"),
            "OUROBOROS_REPO_DIR": REPO_ROOT,
            "OUROBOROS_SERVER_HOST": "127.0.0.1",
            "OUROBOROS_SERVER_PORT": str(port),
            "OUROBOROS_HOST_SERVICE_PORT": str(port + 1),
            "OUROBOROS_NETWORK_PASSWORD": "ui-smoke-password",
        }
        url = f"http://127.0.0.1:{port}"
        active_proc = None

        def stop_server() -> None:
            nonlocal active_proc
            if active_proc is None or active_proc.poll() is not None:
                return
            from ouroboros.platform_layer import IS_WINDOWS, kill_process_tree

            # Windows terminate() is an immediate TerminateProcess, so the parent
            # can disappear before its worker tree and bypass the timeout cleanup.
            # taskkill /T must own that path from the start.
            if IS_WINDOWS:
                kill_process_tree(active_proc)
                active_proc.wait(timeout=5)
                active_proc = None
                return
            active_proc.terminate()
            try:
                active_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                # A timed-out UI-smoke server still owns its worker pool. Killing
                # only the parent leaks ten orphan workers into later smoke tests,
                # producing suite-order history/card timeouts. The server starts in
                # its own process group below, so the shared cross-platform helper
                # can close the complete tree without touching pytest.
                kill_process_tree(active_proc)
                active_proc.wait(timeout=5)
            finally:
                active_proc = None

        def start_server() -> None:
            nonlocal active_proc
            from ouroboros.platform_layer import subprocess_new_group_kwargs

            active_proc = subprocess.Popen(
                [sys.executable, "server.py"],
                cwd=REPO_ROOT,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                **subprocess_new_group_kwargs(),
            )
            _wait_health(url)
            _wait_supervisor_ready(url)

        def restart_server() -> None:
            stop_server()
            start_server()

        try:
            start_server()
            yield {"url": url, "data_dir": data_dir, "restart_server": restart_server}
        finally:
            stop_server()


@pytest.fixture()
def direct_server(direct_server_with_data):
    return direct_server_with_data["url"]


def _write_phase3_widget_smoke_extension(data_dir: pathlib.Path) -> str:
    """Install an exact-hash reviewed extension for the real Widgets flow."""
    from ouroboros.skill_loader import (
        SkillReviewState,
        compute_content_hash,
        save_review_state,
    )

    name = "phase3_widget_smoke"
    skill_dir = data_dir / "skills" / "external" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        textwrap.dedent(
            f"""\
            ---
            name: {name}
            description: Isolated declarative widget visual smoke.
            version: 0.1.0
            type: extension
            entry: plugin.py
            permissions: ["route", "widget", "ws_handler"]
            ---
            # Phase 3 widget smoke
            """
        ),
        encoding="utf-8",
    )
    (skill_dir / "plugin.py").write_text(
        textwrap.dedent(
            """\
            import asyncio


            _STATE = {
                "metric": 87.5,
                "message": "The nested form completed successfully.",
                "rows": [
                    {"label": "Safe reference", "url": "https://example.com/report", "status": "ready"},
                    {"label": "Unsafe reference", "url": "javascript:alert(1)", "status": "blocked"},
                ],
                "chart": {
                    "labels": ["Warm", "Gap", "Hot"],
                    "datasets": [{"label": "Hit rate", "data": [74, None, 91]}],
                },
                "long_json": {"token": "x" * 600},
                "cards": [
                    {"id": "card-1", "label": "Inspect visual evidence", "column": "todo"},
                    {"id": "card-2", "label": "Ship reviewed flow", "column": "done"},
                ],
            }


            def _snapshot():
                return {
                    **_STATE,
                    "rows": [dict(row) for row in _STATE["rows"]],
                    "chart": {
                        "labels": list(_STATE["chart"]["labels"]),
                        "datasets": [
                            {"label": item["label"], "data": list(item["data"])}
                            for item in _STATE["chart"]["datasets"]
                        ],
                    },
                    "cards": [dict(card) for card in _STATE["cards"]],
                }


            async def submit(request):
                body = await request.json()
                await asyncio.sleep(0.25)
                _STATE["message"] = f"Submitted {body.get('query') or 'request'} safely."
                return _snapshot()


            async def move(request):
                body = await request.json()
                await asyncio.sleep(0.15)
                for card in _STATE["cards"]:
                    if card["id"] == body.get("card_id"):
                        card["column"] = body.get("column_id")
                return _snapshot()


            async def save(request):
                body = await request.json()
                await asyncio.sleep(0.2)
                return {"message": f"Saved {body.get('mode') or 'safe'} mode."}


            async def tick(request):
                await asyncio.sleep(0.4)
                data = _STATE["chart"]["datasets"][0]["data"]
                data[0] = (data[0] or 0) + 1
                return _snapshot()


            def register(api):
                async def emit_live(_request):
                    api.send_ws_message("live", {
                        "count": "42.25",
                        "progress": 67,
                        "label": "streaming",
                        "state": "healthy",
                        "unknown": {"nested": True},
                        "nonfinite": "1e999",
                        "image_src": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=",
                        "gallery_image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=",
                        "file_src": "/api/extensions/phase3_widget_smoke/live-file?name=report",
                    })
                    return {"sent": True}

                api.register_route("submit", submit, methods=("POST",))
                api.register_route("move", move, methods=("POST",))
                api.register_route("save", save, methods=("POST",))
                api.register_route("tick", tick, methods=("POST",))
                api.register_route("emit-live", emit_live, methods=("POST",))
                api.register_ui_tab(
                    "main",
                    "Phase 3 design system",
                    render={
                        "kind": "declarative",
                        "schema_version": 1,
                        "components": [
                            {
                                "type": "group",
                                "id": "operations",
                                "title": "Operations overview",
                                "description": "Host-owned composition with stable nested identity.",
                                "layout": "grid",
                                "columns": 2,
                                "components": [
                                    {"type": "metric", "id": "hit-rate", "label": "Cache hit", "path": "metric", "unit": "%", "precision": 1, "tone": "success"},
                                    {"type": "callout", "id": "result-callout", "path": "message", "tone": "warning"},
                                    {
                                        "type": "tabs",
                                        "id": "flows",
                                        "tabs": [
                                            {
                                                "label": "Submit",
                                                "components": [
                                                    {
                                                        "type": "form",
                                                        "id": "query-form",
                                                        "title": "Nested request",
                                                        "route": "submit",
                                                        "method": "POST",
                                                        "columns": 2,
                                                        "submit_label": "Run request",
                                                        "busy_label": "Running…",
                                                        "fields": [
                                                            {"name": "query", "label": "Query", "placeholder": "Ada", "help": "Rendered and escaped by the host.", "span": 2, "required": True},
                                                            {"name": "limit", "label": "Limit", "type": "number", "min": 1, "max": 10, "step": 1, "default": 3},
                                                            {"name": "secret", "label": "Ephemeral secret", "type": "password", "placeholder": "not persisted"},
                                                        ],
                                                    }
                                                ],
                                            },
                                            {
                                                "label": "Data",
                                                "components": [
                                                    {
                                                        "type": "table",
                                                        "id": "result-table",
                                                        "path": "rows",
                                                        "columns": [
                                                            {"label": "Reference", "path": "label", "presentation": "link", "href_path": "url"},
                                                            {"label": "Status", "path": "status", "presentation": "status"},
                                                        ],
                                                    },
                                                    {"type": "chart", "id": "gap-chart", "path": "chart", "chart_type": "line", "unit": "%", "aria_label": "Cache hit rate with an intentional gap"},
                                    {"type": "status", "id": "poll-status", "loading": "Loading data"},
                                    {"type": "poll", "id": "chart-poll", "route": "tick", "method": "POST", "interval_ms": 1000, "max_ticks": 3, "label": "Refresh chart"},
                                                ],
                                            },
                                        ],
                                    },
                                ],
                            },
                            {
                                "type": "subscription",
                                "id": "live-subscription",
                                "event": "live",
                                "target": "live",
                                "render": [
                                    {
                                        "type": "group",
                                        "id": "live-group",
                                        "title": "Live telemetry",
                                        "layout": "grid",
                                        "columns": 2,
                                        "components": [
                                            {"type": "metric", "id": "live-direct-count", "label": "Direct count", "path": "count", "precision": 1},
                                            {
                                                "type": "tabs",
                                                "id": "live-tabs",
                                                "tabs": [
                                                    {
                                                        "label": "Stream",
                                                        "components": [
                                                            {"type": "metric", "id": "live-tab-count", "label": "Tab count", "path": "count", "precision": 1},
                                                            {"type": "metric", "id": "live-state", "label": "State", "path": "state"},
                                                            {"type": "metric", "id": "live-unknown", "label": "Unknown", "path": "unknown"},
                                                            {"type": "metric", "id": "live-nonfinite", "label": "Non-finite", "path": "nonfinite"},
                                                            {"type": "progress", "id": "live-progress", "path": "progress", "label_key": "label"},
                                                        ],
                                                    },
                                                    {
                                                        "label": "Media",
                                                        "components": [
                                                            {"type": "image", "id": "live-image", "path": "image_src", "label": "Live image", "alt": "Live image"},
                                                            {"type": "file", "id": "live-file", "path": "file_src", "label": "Live file", "filename": "live-report.txt"},
                                                            {
                                                                "type": "gallery",
                                                                "id": "live-gallery",
                                                                "items": [
                                                                    {"type": "image", "path": "gallery_image", "label": "Gallery image", "alt": "Gallery image"}
                                                                ],
                                                            },
                                                        ],
                                                    },
                                                ],
                                            },
                                        ],
                                    }
                                ],
                            },
                            {
                                "type": "markdown",
                                "id": "notes",
                                "text": "### Notes\\n\\n- first bullet with an unbroken token abcdefghijklmnopqrstuvwxyz0123456789abcdefghijklmnopqrstuvwxyz0123456789\\n- second bullet\\n\\n1. ordered item one\\n2. ordered item two",
                            },
                            {"type": "json", "id": "long-json", "path": "long_json", "label": "Long JSON"},
                            {
                                "type": "kanban",
                                "id": "delivery-board",
                                "path": "cards",
                                "columns": [
                                    {"id": "todo", "label": "To do"},
                                    {"id": "done", "label": "Done"},
                                ],
                                "on_move": {"route": "move", "method": "POST"},
                            },
                        ],
                    },
                )
                api.register_settings_section(
                    "config",
                    "Phase 3 settings",
                    schema={
                        "components": [
                            {
                                "type": "form",
                                "id": "settings-form",
                                "route": "save",
                                "method": "POST",
                                "submit_label": "Save mode",
                                "busy_label": "Saving mode…",
                                "fields": [
                                    {"name": "mode", "label": "Mode", "type": "select", "default": "safe", "options": [{"label": "Safe", "value": "safe"}, {"label": "Fast", "value": "fast"}]},
                                    {"name": "token", "label": "Temporary token", "type": "password", "placeholder": "not persisted"},
                                ],
                            }
                        ]
                    },
                )
            """
        ),
        encoding="utf-8",
    )
    content_hash = compute_content_hash(skill_dir, manifest_entry="plugin.py")
    save_review_state(
        data_dir,
        name,
        SkillReviewState(status="pass", content_hash=content_hash),
    )
    return name


@pytest.mark.ui_browser
def test_ui_smoke_phase3_declarative_widgets_and_settings(direct_server_with_data):
    """Exercise the real reviewed-extension consumer flow for schema v1."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    data_dir = direct_server_with_data["data_dir"]
    skill = _write_phase3_widget_smoke_extension(data_dir)
    evidence_dir = pathlib.Path(
        os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", str(data_dir.parent))
    )
    evidence_dir.mkdir(parents=True, exist_ok=True)

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                toggled = page.evaluate(
                    """async (skill) => {
                        const response = await fetch(`/api/skills/${encodeURIComponent(skill)}/toggle`, {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({enabled: true}),
                        });
                        return {status: response.status, body: await response.json()};
                    }""",
                    skill,
                )
                assert toggled["status"] == 200, toggled
                assert toggled["body"].get("enabled") is True, toggled

                page.click('[data-nav-page="widgets"]')
                card = page.locator(f'[data-widget-key="{skill}:main"]')
                card.wait_for(state="visible", timeout=30_000)
                assert "Operations overview" in card.inner_text()
                cache_metric = card.locator('.widget-metric').filter(has_text="Cache hit")
                assert cache_metric.locator('strong').inner_text() == "—"

                form = card.locator('[data-widget-form="id:query-form"]')
                form.locator('input[name="query"]').fill("Ada")
                form.locator('input[name="limit"]').fill("4")
                form.locator('input[name="secret"]').fill("ephemeral")
                submit = form.locator('button[type="submit"]')
                submit.click()
                page.wait_for_function(
                    "() => document.querySelector('[data-widget-form=\"id:query-form\"] button')?.disabled === true"
                )
                assert submit.inner_text() == "Running…"
                page.wait_for_function(
                    "() => document.querySelector('[data-widget-form=\"id:query-form\"] button')?.disabled === false",
                    timeout=10_000,
                )
                assert "Submitted Ada safely." in card.inner_text()
                assert "87.5 %" in cache_metric.locator('strong').inner_text()
                metric = cache_metric
                callout = card.locator('.widget-callout')
                assert metric.get_attribute("data-tone") == "ok"
                assert callout.get_attribute("data-tone") == "warn"
                assert metric.evaluate("element => getComputedStyle(element).borderLeftColor") == "rgb(52, 211, 153)"
                assert callout.evaluate("element => getComputedStyle(element).borderLeftColor") == "rgb(251, 191, 36)"

                emitted = page.evaluate(
                    """async (skill) => {
                        const response = await fetch(`/api/extensions/${encodeURIComponent(skill)}/emit-live`, {method: 'POST'});
                        return {status: response.status, body: await response.json()};
                    }""",
                    skill,
                )
                assert emitted == {"status": 200, "body": {"sent": True}}
                page.wait_for_function(
                    """() => {
                        const live = document.querySelector('.widget-subscription-render');
                        if (!live) return false;
                        const metrics = [...live.querySelectorAll('.widget-metric')];
                        const value = (label) => metrics.find((item) => item.querySelector('span')?.textContent === label)?.querySelector('strong')?.textContent.trim();
                        return value('Direct count') === '42.3'
                            && value('Tab count') === '42.3'
                            && value('State') === 'healthy'
                            && value('Unknown') === '—'
                            && value('Non-finite') === '—'
                            && live.querySelector('.widget-progress span')?.textContent.includes('67% · streaming');
                    }""",
                    timeout=10_000,
                )
                live = card.locator('.widget-subscription-render')
                live.get_by_role("button", name="Media").click()
                live_image = live.locator('img[alt="Live image"]')
                gallery_image = live.locator('img[alt="Gallery image"]')
                live_image.wait_for(state="visible", timeout=5_000)
                gallery_image.wait_for(state="visible", timeout=5_000)
                assert live_image.get_attribute("src").startswith("data:image/png;base64,")
                assert gallery_image.get_attribute("src").startswith("data:image/png;base64,")
                assert live.get_by_role("button", name="Live file").get_attribute("data-widget-download-url") == "/api/extensions/phase3_widget_smoke/live-file?name=report"
                live.get_by_role("button", name="Stream").click()

                card.get_by_role("button", name="Data").click()
                chart = card.locator('[data-widget-chart-key="id:gap-chart"]')
                chart.wait_for(state="visible", timeout=5_000)
                # Bounded canvas (v6.71.0): the chart box clamps to 260-360px and
                # never grows the card unbounded.
                canvas_box = chart.bounding_box()
                assert canvas_box and 250 <= canvas_box["height"] <= 370, canvas_box
                chart_config = json.loads(chart.get_attribute("data-widget-chart-config"))
                assert chart_config["data"]["datasets"][0]["data"] == [74, None, 91]
                assert chart_config["data"]["datasets"][0]["spanGaps"] is False
                assert chart_config["options"]["spanGaps"] is False
                assert chart.get_attribute("aria-label") == "Cache hit rate with an intentional gap"
                # Consumer flow (v6.71.0): a poll refetch updates the SAME live
                # canvas in place (wrapper adoption keeps Chart.js resize alive),
                # the config attribute stays fresh, and the SWR status keeps the
                # content with a 'refreshing' indicator instead of a loading swap.
                chart.evaluate("el => { el.__adoptMarker = 42; }")
                first_point = chart_config["data"]["datasets"][0]["data"][0]
                card.get_by_role("button", name="Refresh chart").click()
                page.wait_for_function(
                    """() => document.querySelector('.widget-status')?.dataset.state === 'refreshing'""",
                    timeout=5_000,
                )
                assert card.locator('.widget-status').inner_text() == "Loading data"  # declared loading label reused
                assert card.locator('canvas[data-widget-chart-key="id:gap-chart"]').count() == 1  # content kept during refetch
                # The fixture declares max_ticks=3. Wait for its final value so
                # no scheduled renderAll can detach the geometry probes below.
                page.wait_for_function(
                    """(prev) => {
                        const el = document.querySelector('canvas[data-widget-chart-key="id:gap-chart"]');
                        if (!el) return false;
                        const cfg = JSON.parse(el.dataset.widgetChartConfig || '{}');
                        return cfg.data?.datasets?.[0]?.data?.[0] >= prev + 3;
                    }""",
                    arg=first_point,
                    timeout=10_000,
                )
                assert chart.evaluate("el => el.__adoptMarker") == 42  # SAME canvas node — adopted, not recreated
                page.wait_for_function(
                    """() => document.querySelector('.widget-status')?.dataset.state === 'success'""",
                    timeout=10_000,
                )

                table = card.locator('.widget-chart-data table')
                table_text = table.text_content() or ""
                assert "Gap" in table_text
                assert "—" in table_text
                unsafe_row = card.locator('.widget-table tbody tr').filter(has_text="Unsafe reference")
                assert unsafe_row.locator('a').count() == 0

                # Force the supported no-Chart.js path, then re-render through the
                # same tab lifecycle. The semantic table remains the authority.
                page.evaluate("window.Chart = undefined")
                card.get_by_role("button", name="Submit").click()
                card.get_by_role("button", name="Data").click()
                fallback = card.locator('.widget-chart-fallback')
                fallback.wait_for(state="visible", timeout=5_000)
                assert fallback.locator('canvas').count() == 0
                assert fallback.locator('details[open] table').count() == 1

                move = card.locator('[data-widget-kanban-card="card-1"] [data-widget-kanban-move]')
                move.select_option("done")
                page.wait_for_function(
                    """() => Boolean(
                        document.querySelector('[data-widget-kanban-col="done"] [data-widget-kanban-card="card-1"]')
                        || document.querySelector('.widget-kanban .widget-status[data-state="error"]')
                    )""",
                    timeout=10_000,
                )
                assert card.locator(
                    '[data-widget-kanban-col="done"] [data-widget-kanban-card="card-1"]'
                ).count() == 1, card.inner_text()
                long_json = card.locator('.widget-json').filter(has_text="Long JSON")
                long_json.locator('summary').click()
                json_pre = long_json.locator('pre')
                json_pre.wait_for(state="visible", timeout=5_000)
                assert json_pre.evaluate(
                    "element => getComputedStyle(element).maxHeight"
                ) == "360px"

                # Host layout owns density. A lone data-rich widget uses the
                # available desktop canvas, its nested data surface spans the
                # group grid, and real kanban columns share the row.
                list_box = page.locator('#widgets-list').bounding_box()
                card_box = card.bounding_box()
                operations_group = card.locator('.widget-group').filter(has_text="Operations overview")
                group_box = operations_group.bounding_box()
                tabs_box = operations_group.locator('.widget-tabs').bounding_box()
                kanban_columns = card.locator('.widget-kanban-col')
                todo_box = kanban_columns.nth(0).bounding_box()
                done_box = kanban_columns.nth(1).bounding_box()
                assert list_box and card_box and group_box and tabs_box
                assert todo_box and done_box
                assert card_box["width"] >= list_box["width"] * 0.9
                # Rich-content contract (v6.71.0): list markers render INSIDE the
                # card (the gutter is reserved), and a long unbroken token wraps
                # instead of overflowing the card box.
                markdown_block = card.locator('.widget-markdown.ui-rich-content')
                assert markdown_block.locator('li').count() >= 4
                first_li_box = markdown_block.locator('li').first.bounding_box()
                assert first_li_box and card_box
                assert first_li_box["x"] >= card_box["x"]
                assert first_li_box["x"] + first_li_box["width"] <= card_box["x"] + card_box["width"] + 1
                assert tabs_box["width"] >= group_box["width"] * 0.9
                assert abs(todo_box["y"] - done_box["y"]) < 2
                assert done_box["x"] > todo_box["x"] + todo_box["width"]
                page.screenshot(
                    path=str(evidence_dir / "phase3-widgets-desktop.png"),
                    full_page=True,
                )

                # The same real flow must collapse without page-level
                # horizontal overflow at a narrow viewport.
                page.set_viewport_size({"width": 430, "height": 932})
                page.wait_for_timeout(100)
                assert page.evaluate(
                    "document.documentElement.scrollWidth <= document.documentElement.clientWidth"
                )
                narrow_card = card.bounding_box()
                narrow_columns = card.locator('.widget-kanban-col')
                narrow_todo = narrow_columns.nth(0).bounding_box()
                narrow_done = narrow_columns.nth(1).bounding_box()
                assert narrow_card and narrow_todo and narrow_done
                assert narrow_done["y"] > narrow_todo["y"] + narrow_todo["height"]
                empty_column = card.locator('.widget-kanban-col.is-empty')
                assert empty_column.count() == 1
                empty_box = empty_column.bounding_box()
                assert empty_box and empty_box["height"] < 56
                json_geometry = long_json.evaluate(
                    """element => {
                        const pre = element.querySelector('pre');
                        const card = element.closest('[data-widget-key]');
                        return {
                            cardClient: card.clientWidth,
                            cardScroll: card.scrollWidth,
                            jsonClient: element.clientWidth,
                            jsonScroll: element.scrollWidth,
                            preClient: pre.clientWidth,
                            preScroll: pre.scrollWidth,
                        };
                    }"""
                )
                assert json_geometry["cardScroll"] <= json_geometry["cardClient"]
                assert json_geometry["jsonScroll"] <= json_geometry["jsonClient"]
                assert json_geometry["preScroll"] <= json_geometry["preClient"]
                card.screenshot(
                    path=str(evidence_dir / "phase3-widgets-narrow.png"),
                )
                page.locator('.widgets-scroll').evaluate(
                    "element => { element.scrollTop = element.scrollHeight; }"
                )
                page.wait_for_timeout(100)
                page.screenshot(
                    path=str(evidence_dir / "phase3-widgets-narrow-kanban.png"),
                )

                page.set_viewport_size({"width": 1440, "height": 1000})
                page.wait_for_timeout(100)

                page.click('[data-nav-page="settings"]')
                page.locator('[data-settings-tab="advanced"]').click()
                section = page.locator('.settings-extension-section').filter(
                    has_text="Phase 3 settings"
                )
                section.wait_for(state="visible", timeout=30_000)
                settings_form = section.locator('[data-extension-settings-form]')
                settings_form.locator('select[name="mode"]').select_option("fast")
                settings_form.locator('input[name="token"]').fill("discard-me")
                save = settings_form.locator('button[type="submit"]')
                save.click()
                page.wait_for_function(
                    "() => [...document.querySelectorAll('[data-extension-settings-form] button')].some((button) => button.disabled && button.textContent === 'Saving mode…')"
                )
                section.locator('[data-extension-settings-status]').filter(
                    has_text="Saved fast mode."
                ).wait_for(state="visible", timeout=10_000)
                assert save.is_enabled()
                page.screenshot(
                    path=str(evidence_dir / "phase3-settings-desktop.png"),
                    full_page=True,
                )
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise


@pytest.mark.ui_browser
def test_ui_smoke_direct_mode_loads_chat_and_dashboard(direct_server):
    _run_core_ui_assertions(direct_server)


@pytest.mark.ui_browser
def test_ui_smoke_review_truth_is_visible_in_chat_and_logs(direct_server_with_data):
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    data_dir = direct_server_with_data["data_dir"]
    logs_dir = data_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    projection = {
        "panels": [{
            "panel_id": "panel_visual_truth",
            "surface": "task_acceptance",
            "authority": "host_root",
            "aggregate_signal": "DEGRADED",
            "transport_status": "partial",
            "parse_status": "malformed",
            "quorum": {"required": 2, "contributed": 1, "configured": 3},
            "enforcement_impact": "degrades_completion",
            "reason": "One reviewer timed out, so the panel did not reach quorum.",
            "candidate_hash": "candidate-visual",
            "evidence_revision": "evidence-visual",
            "fence_hash": "fence-visual-hash",
            "actors": [
                {
                    "slot_id": "fable",
                    "actor_role": "task acceptance",
                    "provider": "anthropic",
                    "model": "anthropic/claude-fable-5",
                    "transport_status": "success",
                    "parse_status": "valid",
                    "semantic_verdict": "DEGRADED",
                    "quorum_contribution": True,
                    "enforcement_impact": "supports_pass",
                    "reason": "The browser evidence is incomplete.",
                },
                {
                    "slot_id": "sol",
                    "actor_role": "task acceptance",
                    "provider": "openai",
                    "model": "openai/gpt-5.6-sol",
                    "transport_status": "timeout",
                    "parse_status": "malformed",
                    "semantic_verdict": "",
                    "quorum_contribution": False,
                    "enforcement_impact": "abstains",
                    "reason": "Provider request timed out.",
                },
            ],
        }],
    }
    axes = {
        "lifecycle": {"status": "completed"},
        "execution": {"status": "ok"},
        "objective": {"status": "best_effort"},
        "review": {"status": "degraded"},
        "artifacts": {"status": "ready"},
    }
    summary = {
        "ts": "2026-07-15T10:00:00+00:00",
        "direction": "system",
        "type": "task_summary",
        "task_id": "review-ui",
        "chat_id": 1,
        "text": "Task finished with review evidence.",
        "tool_calls": 0,
        "rounds": 1,
        "outcome_axes": axes,
        "review_projection": projection,
    }
    event = {
        "ts": "2026-07-15T10:00:01+00:00",
        "type": "task_done",
        "task_id": "review-ui",
        "task_type": "task",
        "status": "completed",
        "outcome_axes": axes,
        "review_projection": projection,
    }
    ordinary_final = {
        "ts": "2026-07-15T10:00:00.500000+00:00",
        "direction": "out",
        "chat_id": 1,
        "task_id": "review-no-summary",
        "text": "Normal final answer after the terminal progress anchor.",
        "format": "markdown",
    }
    (logs_dir / "chat.jsonl").write_text(
        json.dumps(summary) + "\n" + json.dumps(ordinary_final) + "\n",
        encoding="utf-8",
    )
    (logs_dir / "events.jsonl").write_text(json.dumps(event) + "\n", encoding="utf-8")
    (logs_dir / "progress.jsonl").write_text(json.dumps({
        "ts": "2026-07-15T09:59:59+00:00",
        "chat_id": 1,
        "task_id": "review-no-summary",
        "content": "Terminal review must survive without a task summary.",
    }) + "\n", encoding="utf-8")
    task_results = data_dir / "task_results"
    task_results.mkdir(parents=True, exist_ok=True)
    (task_results / "review-no-summary.json").write_text(json.dumps({
        "task_id": "review-no-summary",
        "status": "completed",
        "reason_code": "acceptance_degraded",
        "outcome_axes": axes,
        "review_projection": projection,
    }) + "\n", encoding="utf-8")

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                card = page.locator('.chat-live-card[data-task-id="review-ui"]')
                card.wait_for(state="attached", timeout=30_000)
                assert card.is_visible()
                assert card.get_attribute("data-expanded") == "1"
                chat_text = card.inner_text()
                assert "Notice" in chat_text
                assert "Review panel panel_visual_truth" in chat_text
                assert "Reviewer fable" in chat_text
                assert "Reviewer sol" in chat_text
                no_summary = page.locator('.chat-live-card[data-task-id="review-no-summary"]')
                no_summary.wait_for(state="attached", timeout=30_000)
                assert no_summary.is_visible()
                assert no_summary.get_attribute("data-expanded") == "1"
                assert no_summary.locator('[data-live-phase]').first.get_attribute("data-phase") == "warn"
                assert "Review panel panel_visual_truth" in no_summary.inner_text()
                page.wait_for_timeout(900)  # cover the routine background history sync
                assert no_summary.locator('.chat-live-line-repeat:not([hidden])').count() == 0
                assert card.locator('.chat-live-line-repeat:not([hidden])').count() == 0
                page.screenshot(path=str(data_dir.parent / "review-truth-chat.png"), full_page=True)

                page.click('[data-nav-page="dashboard"]')
                page.click('[data-dashboard-tab="logs"]')
                log_card = page.locator('.log-task-card[data-task-group="review-ui"]')
                log_card.wait_for(state="attached", timeout=30_000)
                assert log_card.is_visible()
                review = log_card.locator('[data-task-review]')
                assert review.is_visible()
                log_text = review.inner_text()
                assert "Review panel panel_visual_truth" in log_text
                assert "Reviewer fable" in log_text
                assert "Reviewer sol" in log_text
                assert log_card.locator('[data-task-phase]').inner_text() == "warn"
                review.scroll_into_view_if_needed()
                review.screenshot(path=str(data_dir.parent / "review-truth-logs.png"))
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise


@pytest.mark.ui_browser
def test_ui_smoke_collapsed_activity_line_named_vs_unnamed(direct_server_with_data):
    """v6.82 P1: a collapsed NAMED card shows the latest activity in the
    dedicated [data-live-activity] line under the coined title; an UNNAMED card
    hides the line (its title already shows the activity — no duplication)."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    data_dir = direct_server_with_data["data_dir"]
    logs_dir = data_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "chat.jsonl").write_text("", encoding="utf-8")
    (logs_dir / "progress.jsonl").write_text(
        json.dumps({
            "ts": "2026-07-29T10:00:00+00:00",
            "chat_id": 1,
            "task_id": "named-act",
            "content": "Analyzing the dataset",
        }) + "\n" + json.dumps({
            "ts": "2026-07-29T10:00:01+00:00",
            "chat_id": 1,
            "task_id": "unnamed-act",
            "content": "Doing things without a name",
        }) + "\n",
        encoding="utf-8",
    )
    task_results = data_dir / "task_results"
    task_results.mkdir(parents=True, exist_ok=True)
    (task_results / "named-act.json").write_text(json.dumps({
        "task_id": "named-act",
        "status": "completed",
        "suggested_name": "Data Analysis",
        "cost_usd": 0.42,
        "cost_accounting_status": "available",
        "cost_final": True,
    }) + "\n", encoding="utf-8")

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                named = page.locator('.chat-live-card[data-task-id="named-act"]')
                named.wait_for(state="attached", timeout=30_000)
                unnamed = page.locator('.chat-live-card[data-task-id="unnamed-act"]')
                unnamed.wait_for(state="attached", timeout=30_000)
                # The coined name holds the title; the activity line carries the
                # latest activity and stays populated on the finished card.
                page.wait_for_function(
                    "() => document.querySelector('.chat-live-card[data-task-id=\"named-act\"]"
                    " [data-live-title]')?.textContent === 'Data Analysis'",
                    timeout=30_000,
                )
                named_activity = named.locator('[data-live-activity]')
                assert named_activity.text_content().strip() == "Analyzing the dataset"
                assert named_activity.is_visible()
                # Sticky terminal cost renders in the meta row from the result truth.
                assert "cost=$0.42" in named.locator('[data-live-meta]').inner_text()
                # The unnamed card's title IS its activity; the line stays empty
                # and takes no space (CSS :empty).
                assert "Doing things without a name" in unnamed.locator('[data-live-title]').text_content()
                unnamed_activity = unnamed.locator('[data-live-activity]')
                assert unnamed_activity.text_content().strip() == ""
                assert not unnamed_activity.is_visible()
                page.screenshot(path=str(data_dir.parent / "collapsed-activity.png"), full_page=True)
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise


@pytest.mark.ui_browser
def test_ui_smoke_chat_chronology_reconnect_and_plain_answer_marker(direct_server_with_data):
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    data_dir = direct_server_with_data["data_dir"]
    logs_dir = data_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir = pathlib.Path(
        os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", str(data_dir.parent))
    )
    evidence_dir.mkdir(parents=True, exist_ok=True)
    anchor_summary = {
        "ts": "2025-07-18T10:00:03+00:00",
        "direction": "system",
        "type": "task_summary",
        "system_type": "task_summary",
        "task_id": "chronology-anchor",
        "chat_id": 1,
        "text": "Mounted task card whose earliest event will be backfilled.",
        "tool_calls": 1,
        "rounds": 2,
        "outcome_axes": {
            "lifecycle": {"status": "completed"},
            "execution": {"status": "ok"},
            "objective": {"status": "pass"},
            "review": {"status": "pass"},
            "artifacts": {"status": "ready"},
        },
    }
    t3 = {
        "ts": "2025-07-18T10:00:03.200000+00:00",
        "direction": "out",
        "chat_id": 1,
        "text": "Third historical message.\n" + "\n".join(
            f"Scrollable historical detail {index}." for index in range(80)
        ),
        "format": "markdown",
    }
    (logs_dir / "chat.jsonl").write_text(
        json.dumps(anchor_summary) + "\n" + json.dumps(t3) + "\n",
        encoding="utf-8",
    )

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            context = browser.new_context(viewport={"width": 1280, "height": 800})
            page = context.new_page()
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                third = page.locator(".chat-bubble", has_text="Third historical message.").first
                third.wait_for(state="attached", timeout=30_000)
                assert third.is_visible()
                mounted_anchor = page.locator(
                    '.chat-live-card[data-task-id="chronology-anchor"]'
                )
                mounted_anchor.wait_for(state="attached", timeout=30_000)
                assert mounted_anchor.is_visible()

                t1 = {
                    "ts": "2025-07-18T10:00:01+00:00",
                    "direction": "out",
                    "chat_id": 1,
                    "text": "First historical message.\nFINAL ANSWER: 41",
                    "format": "markdown",
                }
                t2 = {
                    "ts": "2025-07-18T10:00:02+00:00",
                    "direction": "system",
                    "type": "notice",
                    "chat_id": 1,
                    "text": "Second historical system message.\nFINAL ANSWER: 42",
                    "format": "markdown",
                }
                disconnected_summary = {
                    "ts": "2025-07-18T10:00:02.500000+00:00",
                    "direction": "system",
                    "type": "task_summary",
                    "system_type": "task_summary",
                    "task_id": "chronology-disconnected",
                    "chat_id": 1,
                    "text": "Disconnected summary-only card.",
                    "tool_calls": 1,
                    "rounds": 2,
                    "outcome_axes": {
                        "lifecycle": {"status": "completed"},
                        "execution": {"status": "ok"},
                        "objective": {"status": "pass"},
                        "review": {"status": "pass"},
                        "artifacts": {"status": "ready"},
                    },
                }
                t4 = {
                    "ts": "2025-07-18T10:00:04+00:00",
                    "direction": "out",
                    "chat_id": 1,
                    "text": "Fourth new message below the reading anchor.",
                    "format": "markdown",
                }
                (logs_dir / "chat.jsonl").write_text(
                    "".join(
                        json.dumps(row) + "\n"
                        for row in (anchor_summary, t3, t1, t2, disconnected_summary, t4)
                    ),
                    encoding="utf-8",
                )
                (logs_dir / "progress.jsonl").write_text(
                    json.dumps({
                        "ts": "2025-07-18T10:00:01.500000+00:00",
                        "chat_id": 1,
                        "task_id": "chronology-progress-only",
                        "content": "Progress-only terminal card.",
                    }) + "\n" + json.dumps({
                        "ts": "2025-07-18T10:00:01.750000+00:00",
                        "chat_id": 1,
                        "task_id": "chronology-anchor",
                        "content": "Earlier progress backfilled for the mounted anchor card.",
                    }) + "\n",
                    encoding="utf-8",
                )
                task_results = data_dir / "task_results"
                task_results.mkdir(parents=True, exist_ok=True)
                (task_results / "chronology-progress-only.json").write_text(json.dumps({
                    "task_id": "chronology-progress-only",
                    "status": "completed",
                    "outcome_axes": {
                        "lifecycle": {"status": "completed"},
                        "execution": {"status": "ok"},
                        "objective": {"status": "best_effort"},
                        "review": {"status": "degraded"},
                        "artifacts": {"status": "ready"},
                    },
                }) + "\n", encoding="utf-8")

                scroll_before = page.evaluate(
                    """() => {
                        const messages = document.querySelector('#chat-messages');
                        const anchor = messages.querySelector(
                            '.chat-live-card[data-task-id="chronology-anchor"]'
                        );
                        messages.scrollTop = Math.max(1, anchor.offsetTop - 40);
                        return {
                            top: messages.scrollTop,
                            height: messages.scrollHeight,
                            remaining: messages.scrollHeight - messages.scrollTop - messages.clientHeight,
                            anchorTop: anchor?.getBoundingClientRect().top,
                        };
                    }"""
                )
                assert scroll_before["top"] > 0
                assert scroll_before["remaining"] > 160
                direct_server_with_data["restart_server"]()
                page.wait_for_function(
                    "() => [...document.querySelectorAll('.chat-bubble.system')]"
                    ".some((node) => node.textContent.includes('Reconnected'))",
                    timeout=20_000,
                )
                page.wait_for_selector(
                    '.chat-live-card[data-task-id="chronology-progress-only"][data-finished="1"]',
                    timeout=30_000,
                )
                page.wait_for_selector(
                    '.chat-live-card[data-task-id="chronology-disconnected"][data-finished="1"]',
                    timeout=30_000,
                )
                state = page.evaluate(
                    """() => [...document.querySelector('#chat-messages').children]
                        .filter((node) => !node.classList.contains('typing-bubble')
                            && !node.textContent.includes('Reconnected'))
                        .map((node) => ({
                            text: node.textContent,
                            ts: node.dataset.ts || '',
                            card: node.classList.contains('chat-live-card'),
                            taskId: node.dataset.taskId || '',
                        }))"""
                )
                assert [item["card"] for item in state] == [
                    False, True, True, False, True, False, False,
                ]
                assert "First historical message." in state[0]["text"]
                assert "Progress-only terminal card." in state[1]["text"]
                assert state[2]["taskId"] == "chronology-anchor"
                assert "Earlier progress backfilled" in state[2]["text"]
                assert "Second historical system message." in state[3]["text"]
                assert state[4]["taskId"] == "chronology-disconnected"
                assert "Third historical message." in state[5]["text"]
                assert "Fourth new message below the reading anchor." in state[6]["text"]
                assert all(item["ts"].isdigit() for item in state)
                assert page.locator(".final-answer-chip").count() == 0
                assert "FINAL ANSWER: 41" in page.locator("#chat-messages").inner_text()
                assert "FINAL ANSWER: 42" in page.locator("#chat-messages").inner_text()
                assert "2025" in page.locator(
                    '.chat-live-card[data-task-id="chronology-progress-only"]'
                ).inner_text()
                scroll_after = page.evaluate(
                    """() => {
                        const messages = document.querySelector('#chat-messages');
                        const anchor = messages.querySelector(
                            '.chat-live-card[data-task-id="chronology-anchor"]'
                        );
                        return {
                            top: messages.scrollTop,
                            height: messages.scrollHeight,
                            anchorTop: anchor?.getBoundingClientRect().top,
                        };
                    }"""
                )
                assert abs(scroll_after["anchorTop"] - scroll_before["anchorTop"]) <= 6
                page.locator("#chat-messages").evaluate("(messages) => { messages.scrollTop = 0; }")
                page.screenshot(
                    path=str(evidence_dir / "phase3-chat-chronology-desktop.png"),
                    full_page=True,
                )

                page.set_viewport_size({"width": 390, "height": 844})
                page.keyboard.press("Escape")
                page.wait_for_selector("#primary-sidebar:not(.open)", timeout=5_000)
                backdrop = page.locator(".nav-drawer-backdrop")
                backdrop.wait_for(state="attached", timeout=5_000)
                assert backdrop.is_hidden()
                page.wait_for_timeout(250)
                page.locator("#chat-messages").evaluate("(messages) => { messages.scrollTop = 0; }")
                narrow_top_geometry = page.evaluate(
                    """() => {
                        const header = document.querySelector('.chat-page-header');
                        const first = document.querySelector('#chat-messages > :not(.typing-bubble)');
                        return {
                            headerBottom: header?.getBoundingClientRect().bottom,
                            firstTop: first?.getBoundingClientRect().top,
                        };
                    }"""
                )
                assert narrow_top_geometry["firstTop"] >= narrow_top_geometry["headerBottom"] - 2
                page.screenshot(
                    path=str(evidence_dir / "phase3-chat-chronology-narrow.png"),
                    full_page=True,
                )

                page.goto(f"{url}/?_ouro_reason=sha-change", wait_until="domcontentloaded", timeout=30_000)
                page.get_by_text("Restart complete").wait_for(state="visible", timeout=30_000)
                first = page.locator(".chat-bubble", has_text="First historical message.").first
                first.wait_for(state="attached", timeout=30_000)
                assert first.is_visible()
                assert page.locator(".final-answer-chip").count() == 0
                page.screenshot(
                    path=str(evidence_dir / "phase3-chat-chronology-reload.png"),
                    full_page=True,
                )
            finally:
                context.close()
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise


@pytest.mark.ui_browser
def test_ui_smoke_direct_mode_creates_task_with_mock_provider(direct_server):
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page()
            try:
                page.goto(direct_server, wait_until="domcontentloaded", timeout=30_000)
                page.fill("#chat-input", "Respond with exactly OK")
                page.click("#chat-send")
                page.wait_for_selector(".chat-bubble.assistant", timeout=60_000)
                assert "OK" in page.locator("#chat-messages").inner_text(timeout=5_000)
                metrics = page.evaluate(
                    """() => {
                        const messages = document.querySelector('#chat-messages');
                        const remaining = messages.scrollHeight - messages.scrollTop - messages.clientHeight;
                        return {
                            scrollTop: messages.scrollTop,
                            scrollHeight: messages.scrollHeight,
                            clientHeight: messages.clientHeight,
                            remaining,
                        };
                    }"""
                )
                assert metrics["remaining"] <= 4, metrics
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise


@pytest.mark.ui_browser
def test_ui_smoke_direct_mode_nests_subagent_child_cards(direct_server_with_data):
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    data_dir = direct_server_with_data["data_dir"]
    logs_dir = data_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    child_review_projection = {
        "panels": [{
            "panel_id": "panel_child_review",
            "surface": "task_acceptance",
            "authority": "host_root",
            "aggregate_signal": "DEGRADED",
            "transport_status": "success",
            "parse_status": "valid",
            "quorum": {"required": 1, "contributed": 0, "configured": 1},
            "enforcement_impact": "degrades_completion",
            "reason": "Child evidence was incomplete.",
            "actors": [{
                "slot_id": "child_actor",
                "actor_role": "task acceptance",
                "provider": "openrouter",
                "model": "anthropic/claude-fable-5",
                "transport_status": "success",
                "parse_status": "valid",
                "semantic_verdict": "DEGRADED",
                "quorum_contribution": False,
                "enforcement_impact": "abstains",
                "reason": "Missing child visual evidence.",
            }],
        }],
    }
    child_outcome_axes = {
        "lifecycle": {"status": "completed"},
        "execution": {"status": "ok"},
        "objective": {"status": "best_effort"},
        "review": {"status": "degraded"},
        "artifacts": {"status": "ready"},
    }
    rows = [
        {
            "ts": "2026-05-25T10:00:00+00:00",
            "chat_id": 1,
            "task_id": "parent1",
            "content": "Parent task started",
            "is_progress": True,
        },
        {
            "ts": "2026-05-25T10:00:01+00:00",
            "chat_id": 1,
            "task_id": "child1",
            "content": "Scheduled subagent child1",
            "is_progress": True,
            "delegation_role": "subagent",
            "subagent_event": "scheduled",
            "subagent_task_id": "child1",
            "parent_task_id": "parent1",
            "root_task_id": "parent1",
            "subagent_role": "researcher",
        },
        {
            "ts": "2026-05-25T10:00:02+00:00",
            "chat_id": 1,
            "task_id": "child1",
            "content": "Subagent child1 running",
            "is_progress": True,
            "delegation_role": "subagent",
            "subagent_event": "running",
            "subagent_task_id": "child1",
            "parent_task_id": "parent1",
            "root_task_id": "parent1",
            "subagent_role": "researcher",
            "status": "running",
        },
        {
            "ts": "2026-05-25T10:00:02.500000+00:00",
            "chat_id": 1,
            "task_id": "child1",
            "content": "Searching evidence",
            "is_progress": True,
            "delegation_role": "subagent",
            "subagent_event": "progress",
            "subagent_task_id": "child1",
            "parent_task_id": "parent1",
            "root_task_id": "parent1",
            "subagent_role": "researcher",
            "status": "running",
        },
        {
            "ts": "2026-05-25T10:00:03+00:00",
            "chat_id": 1,
            "task_id": "child1",
            "content": "Subagent child1 completed",
            "is_progress": True,
            "delegation_role": "subagent",
            "subagent_event": "completed",
            "subagent_task_id": "child1",
            "parent_task_id": "parent1",
            "root_task_id": "parent1",
            "subagent_role": "researcher",
            "status": "completed",
            "cost_usd": 0.125,
            "result": "Child result with evidence table\n| source | verdict |\n| A | pass |",
            "trace_summary": "searched sources\ncompared output",
            "outcome_axes": child_outcome_axes,
            "reason_code": "acceptance_degraded",
            "review_projection": child_review_projection,
        },
        {
            "ts": "2026-05-25T10:00:03.100000+00:00",
            "chat_id": 1,
            "task_id": "grandchild1",
            "content": "Scheduled nested subagent grandchild1",
            "is_progress": True,
            "delegation_role": "subagent",
            "subagent_event": "scheduled",
            "subagent_task_id": "grandchild1",
            "parent_task_id": "child1",
            "root_task_id": "parent1",
            "subagent_role": "evidence-mapper",
        },
        {
            "ts": "2026-05-25T10:00:03.200000+00:00",
            "chat_id": 1,
            "task_id": "grandchild1",
            "content": "Nested subagent grandchild1 completed",
            "is_progress": True,
            "delegation_role": "subagent",
            "subagent_event": "completed",
            "subagent_task_id": "grandchild1",
            "parent_task_id": "child1",
            "root_task_id": "parent1",
            "subagent_role": "evidence-mapper",
            "status": "completed",
            "result": "Nested evidence result",
        },
    ]
    (logs_dir / "progress.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    (logs_dir / "chat.jsonl").write_text(
        json.dumps({
            "ts": "2026-05-25T10:00:03.500000+00:00",
            "chat_id": 1,
            "direction": "out",
            "task_id": "child1",
            "text": "Final child answer should stay inside the child card.",
            "format": "markdown",
        }) + "\n",
        encoding="utf-8",
    )

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                page.wait_for_selector(".chat-live-card", state="attached", timeout=30_000)
                assert page.locator(".chat-live-card").first.is_visible()
                # Subagents render as always-visible child cards nested under
                # the parent card. Child completion must not finish the parent.
                page.wait_for_function("() => document.querySelectorAll('.chat-live-card').length === 3", timeout=30_000)
                page.wait_for_function(
                    "() => { const p = document.querySelector('.chat-live-card:not(.subagent)');"
                    " const c = document.querySelector('.chat-live-card.subagent[data-parent-task-id=\"parent1\"]');"
                    " const g = document.querySelector('.chat-live-card.subagent[data-parent-task-id=\"child1\"]');"
                    " return !!p && !!c && c.closest('.chat-subagents') && c.parentElement.closest('.chat-live-card') === p"
                    " && !!g && g.closest('.chat-subagents') && g.parentElement.closest('.chat-live-card') === c"
                    " && /researcher \\(child1\\)/.test(c.innerText) && /role=researcher/.test(c.innerText)"
                    " && /evidence-mapper \\(grandchi/.test(g.innerText); }",
                    timeout=30_000,
                )
                parent = page.locator(".chat-live-card:not(.subagent)").first
                child = page.locator('.chat-live-card.subagent[data-parent-task-id="parent1"]').first
                grandchild = page.locator('.chat-live-card.subagent[data-parent-task-id="child1"]').first
                parent_ts = int(parent.get_attribute("data-ts"))
                child_ts = int(child.get_attribute("data-ts"))
                grandchild_ts = int(grandchild.get_attribute("data-ts"))
                assert parent_ts < child_ts < grandchild_ts
                parent_count = parent.locator(':scope > [data-live-summary-button] [data-live-count]').first
                child_count = child.locator(':scope > [data-live-summary-button] [data-live-count]').first
                parent_text = parent.inner_text()
                child_text = child.inner_text()
                assert "Parent task started" in parent_text
                assert "1 child" in parent_count.inner_text()
                assert "researcher (child1)" in child_text
                assert "1 child" in child_count.inner_text()
                assert "child=child1" in child_text
                assert "role=researcher" in child_text
                assert "panel_child_review" in child_text
                assert "claude-fable-5" in child_text
                assert "verdict=DEGRADED" in child_text
                assert "evidence-mapper (grandchi" in grandchild.inner_text()
                assert child.get_attribute("data-task-id") == "child1"
                assert page.locator(
                    '.chat-live-card[data-task-id="parent1"] > .chat-subagents > '
                    '.chat-live-card.subagent[data-task-id="child1"]'
                ).count() == 1
                assert page.locator(
                    '.chat-live-card.subagent[data-task-id="child1"] > .chat-subagents > '
                    '.chat-live-card.subagent[data-task-id="grandchild1"]'
                ).count() == 1
                assert page.locator("#chat-messages > .chat-live-card.subagent").count() == 0
                assert parent.get_attribute("data-finished") == "0"
                assert child.get_attribute("data-finished") == "1"
                assert child.locator(":scope > [data-live-summary-button] [data-live-phase]").first.get_attribute("data-phase") == "warn"
                assert child.get_attribute("data-subagent-role") == "researcher"
                assert grandchild.get_attribute("data-finished") == "1"
                assert grandchild.get_attribute("data-subagent-role") == "evidence-mapper"
                assert page.locator(".chat-bubble.progress").count() == 0
                assert page.locator(".chat-bubble").filter(
                    has_text="Final child answer should stay inside the child card."
                ).count() == 0

                # Review actor/model details are disclosed immediately even on a
                # nested child; ordinary nested cards remain collapsed.
                assert child.get_attribute("data-expanded") == "1"
                assert grandchild.get_attribute("data-expanded") == "0"
                child_summary = child.locator(":scope > [data-live-summary-button]").first
                line_toggles = child.locator(".chat-live-line-toggle:visible")
                if line_toggles.count():
                    line_toggles.last.click()
                expanded_text = child.inner_text(timeout=5_000)
                assert "Final child answer should stay inside the child card." in expanded_text
                assert "Child result with evidence table" in expanded_text
                assert "| source | verdict |" in expanded_text
                assert "searched sources" in expanded_text
                assert "compared output" in expanded_text
                assert "done" in expanded_text.lower()
                assert "Scheduled subagent child1" not in expanded_text
                assert child_summary.get_attribute("aria-expanded") == "true"
                assert child.locator("[data-live-timeline]").first.get_attribute("id")
                if line_toggles.count():
                    assert line_toggles.last.get_attribute("aria-controls")

                page.reload(wait_until="domcontentloaded", timeout=30_000)
                page.wait_for_function("() => document.querySelectorAll('.chat-live-card').length === 3", timeout=30_000)
                page.wait_for_function(
                    "() => { const p = document.querySelector('.chat-live-card:not(.subagent)');"
                    " const c = document.querySelector('.chat-live-card.subagent[data-parent-task-id=\"parent1\"]');"
                    " const g = document.querySelector('.chat-live-card.subagent[data-parent-task-id=\"child1\"]');"
                    " return !!p && !!c && c.closest('.chat-subagents') && c.parentElement.closest('.chat-live-card') === p"
                    " && !!g && g.closest('.chat-subagents') && g.parentElement.closest('.chat-live-card') === c; }",
                    timeout=30_000,
                )
                replay_parent = page.locator(".chat-live-card:not(.subagent)").first
                replay_child = page.locator('.chat-live-card.subagent[data-parent-task-id="parent1"]').first
                replay_grandchild = page.locator('.chat-live-card.subagent[data-parent-task-id="child1"]').first
                assert replay_parent.get_attribute("data-finished") == "0"
                assert replay_child.get_attribute("data-finished") == "1"
                assert replay_child.locator(":scope > [data-live-summary-button] [data-live-phase]").first.get_attribute("data-phase") == "warn"
                assert replay_grandchild.get_attribute("data-finished") == "1"
                assert replay_child.get_attribute("data-expanded") == "1"
                assert replay_grandchild.get_attribute("data-expanded") == "0"
                assert "researcher (child1)" in replay_child.inner_text()
                assert "child=child1" in replay_child.inner_text()
                assert "Final child answer should stay inside the child card." in replay_child.inner_text()
                page.wait_for_timeout(900)  # cover the routine background history sync
                assert replay_child.locator('.chat-live-line-repeat:not([hidden])').count() == 0
                page.screenshot(path=str(data_dir.parent / "review-truth-child-reconnect.png"), full_page=True)
                assert page.locator(".chat-bubble.progress").count() == 0
                assert page.locator(".chat-bubble", has_text="Final child answer should stay inside the child card.").count() == 0

                page.evaluate(
                    """async () => {
                        const resp = await fetch('/api/ui/preferences', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ nested_subagents_expanded: true }),
                        });
                        if (!resp.ok) throw new Error(await resp.text());
                    }"""
                )
                page.reload(wait_until="domcontentloaded", timeout=30_000)
                page.wait_for_function("() => document.querySelectorAll('.chat-live-card').length === 3", timeout=30_000)
                const_pref_check = (
                    "() => {"
                    " const c = document.querySelector('.chat-live-card.subagent[data-parent-task-id=\"parent1\"]');"
                    " const g = document.querySelector('.chat-live-card.subagent[data-parent-task-id=\"child1\"]');"
                    " return !!c && !!g && c.dataset.expanded === '1' && g.dataset.expanded === '1';"
                    " }"
                )
                page.wait_for_function(const_pref_check, timeout=30_000)
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise


@pytest.mark.ui_browser
def test_ui_smoke_desktop_composer_chips_above_input_send_inside(direct_server):
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            try:
                page.goto(direct_server, wait_until="domcontentloaded", timeout=30_000)
                page.wait_for_selector("#chat-input", timeout=30_000)
                metrics = page.evaluate(
                    """() => {
                        const rect = (selector) => {
                            const el = document.querySelector(selector);
                            const r = el.getBoundingClientRect();
                            return { left: r.left, right: r.right, top: r.top, bottom: r.bottom, width: r.width, height: r.height };
                        };
                        return {
                            input: rect('#chat-input'),
                            toolbar: rect('.chat-toolbar-row'),
                            send: rect('.chat-send-group'),
                            sendButton: rect('.chat-send-inline'),
                            swarm: rect('.chat-swarm'),
                            contextMode: rect('.chat-context-mode'),
                        };
                    }"""
                )
                # v6.32.0 composer redesign (owner: "чипы правильнее НАД полем ввода"):
                # the chips row (Swarm + Low/Max) sits ABOVE the text input...
                assert metrics["toolbar"]["bottom"] <= metrics["input"]["top"] + 4, metrics
                assert metrics["swarm"]["bottom"] <= metrics["input"]["top"] + 4, metrics
                assert metrics["contextMode"]["bottom"] <= metrics["input"]["top"] + 4, metrics
                # ...the two chips share that row (aligned tops)...
                assert abs(metrics["swarm"]["top"] - metrics["contextMode"]["top"]) <= 2, metrics
                # ...and the Send button stays INSIDE the input's vertical band (same text row).
                assert metrics["send"]["top"] >= metrics["input"]["top"] - 4, metrics
                assert metrics["send"]["bottom"] <= metrics["input"]["bottom"] + 4, metrics
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise


@pytest.mark.ui_browser
def test_ui_smoke_mobile_composer_toolbar_does_not_overlap_input(direct_server):
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 390, "height": 844}, is_mobile=True, has_touch=True)
            try:
                page.goto(direct_server, wait_until="domcontentloaded", timeout=30_000)
                page.wait_for_selector("#chat-input", timeout=30_000)
                metrics = page.evaluate(
                    """() => {
                        const rect = (selector) => {
                            const el = document.querySelector(selector);
                            const r = el.getBoundingClientRect();
                            return { left: r.left, right: r.right, top: r.top, bottom: r.bottom, width: r.width, height: r.height };
                        };
                        const inputStyle = getComputedStyle(document.querySelector('#chat-input'));
                        return {
                            input: rect('#chat-input'),
                            toolbar: rect('.chat-toolbar-row'),
                            pills: rect('.chat-composer-pills'),
                            send: rect('.chat-send-group'),
                            sendButton: rect('.chat-send-inline'),
                            swarm: rect('.chat-swarm'),
                            contextMode: rect('.chat-context-mode'),
                            paddingRight: inputStyle.paddingRight,
                        };
                    }"""
                )
                # Mobile (390px): chips ride ABOVE the input row, while the input
                # shares its row with the attach button (left) and the Send button
                # (right). The usable input width is therefore naturally below the
                # old desktop-era 300px target; assert it stays usable (>= half the
                # viewport) and never runs under the Send button.
                assert metrics["input"]["width"] >= 190, metrics
                assert metrics["input"]["right"] <= metrics["send"]["left"] + 2, metrics
                assert metrics["toolbar"]["bottom"] <= metrics["input"]["top"] + 1, metrics
                assert metrics["send"]["top"] >= metrics["input"]["top"] - 1, metrics
                assert metrics["send"]["bottom"] <= metrics["input"]["bottom"] + 1, metrics
                assert abs(metrics["swarm"]["height"] - metrics["sendButton"]["height"]) <= 1, metrics
                assert abs(metrics["contextMode"]["height"] - metrics["sendButton"]["height"]) <= 1, metrics
                assert metrics["paddingRight"] != "256px", metrics
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise


def _install_controlled_visual_viewport(page, initial_height: int) -> None:
    """Install a deterministic viewport-height signal before application JS.

    This exercises Ouroboros's viewport/focus state machine, not a native OS
    keyboard. The assertions below separately inspect the rendered drawer.
    """
    page.add_init_script(
        f"""(() => {{
            let height = {int(initial_height)};
            const viewport = new EventTarget();
            Object.defineProperty(viewport, 'height', {{ get: () => height }});
            Object.defineProperty(window, 'visualViewport', {{
                configurable: true,
                value: viewport,
            }});
            window.__setTestVisualViewportHeight = (nextHeight) => {{
                height = Number(nextHeight);
                viewport.dispatchEvent(new Event('resize'));
            }};
        }})()"""
    )


def _mobile_keyboard_drawer_assertions(page, url: str, screenshot_path: pathlib.Path) -> None:
    page.goto(url, wait_until="domcontentloaded", timeout=30_000)
    page.wait_for_selector("#chat-input", timeout=30_000)

    # A transient Telegram/WebView viewport shrink with no focused editable must
    # never claim that the software keyboard is open.
    page.evaluate("() => window.__setTestVisualViewportHeight(500)")
    page.wait_for_timeout(50)
    assert not page.locator("body").evaluate("el => el.classList.contains('keyboard-open')")

    # Restore the stable app viewport, then prove the same shrink is recognized
    # while the composer really owns focus.
    page.evaluate("() => window.__setTestVisualViewportHeight(844)")
    page.wait_for_timeout(50)
    page.focus("#chat-input")
    page.evaluate("() => window.__setTestVisualViewportHeight(500)")
    page.wait_for_function("() => document.body.classList.contains('keyboard-open')", timeout=5_000)

    toggle = page.locator("#page-chat [data-mobile-nav-toggle]")
    toggle.click()
    page.wait_for_function(
        "() => document.body.classList.contains('nav-drawer-open')"
        " && !document.body.classList.contains('keyboard-open')"
        " && document.activeElement?.id !== 'chat-input'",
        timeout=5_000,
    )
    page.wait_for_timeout(220)  # finish the 180ms drawer transition

    state = page.evaluate(
        """() => {
            const sidebar = document.querySelector('#primary-sidebar');
            const backdrop = document.querySelector('#nav-drawer-backdrop');
            const toggle = document.querySelector('#page-chat [data-mobile-nav-toggle]');
            const rect = sidebar.getBoundingClientRect();
            return {
                bodyOpen: document.body.classList.contains('nav-drawer-open'),
                sidebarOpen: sidebar.classList.contains('open'),
                sidebarDisplay: getComputedStyle(sidebar).display,
                sidebarVisibility: getComputedStyle(sidebar).visibility,
                sidebarRect: {left: rect.left, right: rect.right, width: rect.width, height: rect.height},
                backdropHidden: backdrop.hidden,
                backdropDisplay: getComputedStyle(backdrop).display,
                ariaExpanded: toggle.getAttribute('aria-expanded'),
                activeId: document.activeElement?.id || '',
                keyboardBody: document.body.classList.contains('keyboard-open'),
                keyboardRoot: document.documentElement.classList.contains('keyboard-open'),
            };
        }"""
    )
    assert state["bodyOpen"] and state["sidebarOpen"], state
    assert state["ariaExpanded"] == "true", state
    assert not state["backdropHidden"] and state["backdropDisplay"] != "none", state
    assert state["sidebarDisplay"] != "none" and state["sidebarVisibility"] != "hidden", state
    assert state["sidebarRect"]["width"] > 200 and state["sidebarRect"]["height"] > 400, state
    assert state["sidebarRect"]["left"] >= -1 and state["sidebarRect"]["right"] > 0, state
    assert state["activeId"] != "chat-input", state
    assert not state["keyboardBody"] and not state["keyboardRoot"], state

    # The now-visible drawer must still own a vertically scrollable content
    # surface even though the keyboard touch lock was active one frame earlier.
    scroll = page.evaluate(
        """() => {
            const scroller = document.querySelector('#primary-sidebar .sidebar-scroll');
            for (let i = 0; i < 60; i += 1) {
                const row = document.createElement('button');
                row.className = 'nav-row';
                row.type = 'button';
                row.textContent = `Drawer overflow probe ${i}`;
                scroller.appendChild(row);
            }
            scroller.scrollTop = scroller.scrollHeight;
            return {
                scrollTop: scroller.scrollTop,
                scrollHeight: scroller.scrollHeight,
                clientHeight: scroller.clientHeight,
                overflowY: getComputedStyle(scroller).overflowY,
            };
        }"""
    )
    assert scroll["scrollHeight"] > scroll["clientHeight"], scroll
    assert scroll["scrollTop"] > 0, scroll
    assert scroll["overflowY"] in {"auto", "scroll"}, scroll
    page.screenshot(path=str(screenshot_path), full_page=True)

    # Exercise the real backdrop click in the visible strip to the right of the
    # 320px drawer, then require all state/ARIA projections to close together.
    page.locator("#nav-drawer-backdrop").click(position={"x": 380, "y": 400})
    page.wait_for_function(
        "() => !document.body.classList.contains('nav-drawer-open')"
        " && !document.querySelector('#primary-sidebar').classList.contains('open')"
        " && document.querySelector('#nav-drawer-backdrop').hidden"
        " && document.querySelector('#page-chat [data-mobile-nav-toggle]').getAttribute('aria-expanded') === 'false'",
        timeout=5_000,
    )


@pytest.mark.ui_browser
def test_ui_smoke_mobile_keyboard_state_cannot_hide_open_drawer_chromium(direct_server_with_data):
    """Controlled visualViewport state plus real Chromium drawer geometry."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 390, "height": 844}, is_mobile=True, has_touch=True)
            try:
                _install_controlled_visual_viewport(page, 844)
                _mobile_keyboard_drawer_assertions(
                    page,
                    direct_server_with_data["url"],
                    direct_server_with_data["data_dir"].parent / "mobile-keyboard-drawer-chromium.png",
                )
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise


@pytest.mark.ui_browser
def test_ui_smoke_mobile_keyboard_state_cannot_hide_open_drawer_webkit(direct_server_with_data):
    """Same controlled state-machine check in WebKit with an iPhone profile."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    try:
        with sync_playwright() as pw:
            iphone = pw.devices.get("iPhone 13")
            if not iphone:
                pytest.skip("Playwright iPhone 13 device descriptor unavailable")
            try:
                browser = pw.webkit.launch(headless=True)
            except PlaywrightError as exc:
                if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
                    pytest.skip(f"Playwright WebKit browser is not installed: {exc}")
                raise
            context = browser.new_context(**iphone)
            page = context.new_page()
            try:
                # The controller's threshold is driven by our deterministic app
                # viewport; the iPhone descriptor still owns rendering/input.
                _install_controlled_visual_viewport(page, 844)
                _mobile_keyboard_drawer_assertions(
                    page,
                    direct_server_with_data["url"],
                    direct_server_with_data["data_dir"].parent / "mobile-keyboard-drawer-webkit.png",
                )
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise


@pytest.mark.ui_browser
def test_ui_smoke_direct_mode_chat_scrolls_on_desktop(direct_server):
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    def scroll_metrics(page):
        return page.evaluate(
            """() => {
                const messages = document.querySelector('#chat-messages');
                if (!messages) return null;
                messages.scrollTop = 0;
                const top = messages.scrollTop;
                messages.scrollTop = messages.scrollHeight;
                const bottom = messages.scrollTop;
                return {
                    clientHeight: messages.clientHeight,
                    scrollHeight: messages.scrollHeight,
                    top,
                    bottom,
                    overflowY: getComputedStyle(messages).overflowY,
                    runtimeVvh: document.getElementById('runtime-vvh')?.textContent || '',
                    bodyHeight: Math.round(document.body.getBoundingClientRect().height),
                    windowHeight: window.innerHeight,
                };
            }"""
        )

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            try:
                page.goto(direct_server, wait_until="domcontentloaded", timeout=30_000)
                page.get_by_role("button", name="Chat").click()
                page.wait_for_selector("#chat-messages", timeout=30_000)
                # Wait for the initial history rebuild to finish before injecting
                # synthetic rows; otherwise that authoritative rebuild may erase
                # the probe immediately after insertion on slower startup paths.
                page.wait_for_selector("#chat-messages .chat-bubble.assistant", timeout=30_000)
                # A viewport change can re-render the chat from the (empty) real
                # history and drop injected probe nodes, so injection is a helper
                # re-run before every measurement instead of a one-shot setup.
                inject_probe_bubbles = """() => {
                    const messages = document.querySelector('#chat-messages');
                    messages.replaceChildren();
                    for (let i = 0; i < 48; i += 1) {
                        const bubble = document.createElement('div');
                        bubble.className = 'chat-bubble assistant';
                        bubble.textContent = `Desktop scroll probe ${i} `.repeat(16);
                        bubble.style.minHeight = '48px';
                        messages.appendChild(bubble);
                    }
                }"""
                page.evaluate(inject_probe_bubbles)

                metrics = scroll_metrics(page)
                assert metrics is not None
                assert metrics["overflowY"] in {"auto", "scroll"}
                assert metrics["scrollHeight"] > metrics["clientHeight"] + 100
                assert metrics["bottom"] > metrics["top"] + 100
                assert "--vvh:100dvh" in metrics["runtimeVvh"]
                assert abs(metrics["bodyHeight"] - metrics["windowHeight"]) <= 2

                page.set_viewport_size({"width": 1280, "height": 400})
                page.wait_for_timeout(100)
                page.set_viewport_size({"width": 1280, "height": 800})
                page.wait_for_timeout(100)
                page.evaluate(inject_probe_bubbles)

                metrics_after_resize = scroll_metrics(page)
                assert metrics_after_resize is not None
                assert metrics_after_resize["scrollHeight"] > metrics_after_resize["clientHeight"] + 100
                assert metrics_after_resize["bottom"] > metrics_after_resize["top"] + 100
                assert "--vvh:100dvh" in metrics_after_resize["runtimeVvh"]
                assert abs(metrics_after_resize["bodyHeight"] - metrics_after_resize["windowHeight"]) <= 2
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise


@pytest.mark.ui_browser
def test_ui_smoke_finished_cards_keep_height_when_transcript_overflows(direct_server):
    """Regression: live cards / skill_review bubbles use overflow:hidden, which
    gives them an automatic flex min-height of 0. When the transcript column
    overflows they must NOT be shrunk to a 1px strip — the list scrolls instead.
    (rc.1 removed the inline min-height that previously masked this collapse.)"""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 600})
            try:
                page.goto(direct_server, wait_until="domcontentloaded", timeout=30_000)
                page.wait_for_selector("#chat-messages", timeout=30_000)
                result = page.evaluate(
                    """() => {
                        const messages = document.querySelector('#chat-messages');
                        messages.replaceChildren();
                        // Overflow the column with collapsed, overflow:hidden cards.
                        for (let i = 0; i < 24; i += 1) {
                            const card = document.createElement('div');
                            card.className = 'chat-live-card';
                            card.dataset.finished = '1';
                            card.dataset.expanded = '0';
                            const btn = document.createElement('div');
                            btn.className = 'chat-live-summary-button';
                            btn.style.minHeight = '48px';
                            btn.textContent = `Finished card ${i}`;
                            card.appendChild(btn);
                            messages.appendChild(card);
                        }
                        const heights = [...messages.querySelectorAll('.chat-live-card')]
                            .map((el) => Math.round(el.getBoundingClientRect().height));
                        return {
                            heights,
                            scrollHeight: messages.scrollHeight,
                            clientHeight: messages.clientHeight,
                        };
                    }"""
                )
                assert result["heights"], "no cards rendered"
                # Without flex-shrink:0 the overflow:hidden cards collapse to ~1px.
                assert min(result["heights"]) >= 40, result
                # The column should scroll rather than absorb the overflow.
                assert result["scrollHeight"] > result["clientHeight"] + 100, result
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise


@pytest.mark.ui_browser_docker
def test_ui_smoke_docker_mode_loads_health():
    if os.environ.get("OUROBOROS_RUN_DOCKER_UI_SMOKE") != "1":
        pytest.skip("set OUROBOROS_RUN_DOCKER_UI_SMOKE=1 to run Docker UI smoke")
    image = os.environ.get("OUROBOROS_DOCKER_UI_IMAGE", "ouroboros-web:test")
    probe = subprocess.run(["docker", "image", "inspect", image], capture_output=True, text=True, timeout=20)
    if probe.returncode != 0:
        pytest.skip(f"Docker image missing: {image}")
    port = _free_port()
    run = subprocess.run(
        ["docker", "run", "-d", "--rm", "-p", f"{port}:8765", image],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if run.returncode != 0:
        pytest.skip(f"Docker daemon unavailable or container failed: {run.stderr}")
    cid = run.stdout.strip()
    try:
        url = f"http://127.0.0.1:{port}"
        _wait_health(url, timeout_sec=45)
        _run_docker_ui_assertions(url)
    finally:
        subprocess.run(["docker", "stop", cid], capture_output=True, text=True, timeout=30)


@pytest.mark.ui_browser
@pytest.mark.parametrize("browser_engine", ["chromium", "webkit"])
def test_ui_smoke_live_cards_keep_usable_geometry_at_depth_and_in_project_panel(
    direct_server_with_data,
    browser_engine,
):
    """Rendered regression for the one-letter-wide nested-card failure.

    A real replayed task tree reaches the configured hard depth of ten. The
    narrow checks use geometry instead of CSS declarations, then reload to
    cover replay. Wide Main and a narrow Project panel prove the card-local
    container responds to its actual consumer width rather than the viewport.
    """
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    from ouroboros.projects_registry import create_project

    data_dir = direct_server_with_data["data_dir"]
    url = direct_server_with_data["url"]
    project = create_project(data_dir, "layout-project", name="Layout Project")
    logs_dir = data_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    rows = [{
        "ts": "2026-05-25T10:00:00+00:00",
        "chat_id": 1,
        "task_id": "layout-root",
        "content": "Root task started",
        "suggested_name": "Deep nested layout regression",
        "is_progress": True,
    }]
    long_url = "https://example.com/" + "nested-segment-without-breaks-" * 18 + "final"
    parent_id = "layout-root"
    for depth in range(1, 11):
        child_id = f"layout-child-{depth:02d}"
        rows.append({
            "ts": f"2026-05-25T10:00:{depth:02d}+00:00",
            "chat_id": 1,
            "task_id": child_id,
            "content": f"Depth {depth} subagent completed",
            "is_progress": True,
            "delegation_role": "subagent",
            "subagent_event": "completed",
            "subagent_task_id": child_id,
            "parent_task_id": parent_id,
            "root_task_id": "layout-root",
            "subagent_role": "pty-tests",
            "model": "google/gemini-3.6-flash",
            "status": "completed",
            "result": f"Depth {depth} complete. Full evidence: {long_url}",
        })
        parent_id = child_id
    panel_row = {
        "ts": "2026-05-25T10:01:00+00:00",
        "chat_id": project["chat_id"],
        "task_id": "panel-root",
        "content": "Inspecting a narrow Project panel with a long unbroken reference " + long_url,
        "suggested_name": "Narrow Project panel keeps a usable title column",
        "is_progress": True,
    }
    (logs_dir / "progress.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    mobile_geometry = """() => {
        const messages = document.querySelector('#page-chat #chat-messages');
        const messagesStyle = getComputedStyle(messages);
        const usableMessageWidth = messages.clientWidth
            - parseFloat(messagesStyle.paddingInlineStart || messagesStyle.paddingLeft || '0')
            - parseFloat(messagesStyle.paddingInlineEnd || messagesStyle.paddingRight || '0');
        const cards = [...messages.querySelectorAll('.chat-live-card')];
        const root = messages.querySelector(':scope > .chat-live-card[data-task-id="layout-root"]');
        const deepest = messages.querySelector('.chat-live-card[data-task-id="layout-child-10"]');
        const cardFacts = cards.map((card) => {
            const title = card.querySelector(':scope > .chat-live-summary-button [data-live-title]');
            const style = getComputedStyle(title);
            const lineHeight = parseFloat(style.lineHeight);
            const titleRect = title.getBoundingClientRect();
            return {
                id: card.dataset.taskId,
                clientWidth: card.clientWidth,
                scrollWidth: card.scrollWidth,
                titleWidth: titleRect.width,
                titleHeight: titleRect.height,
                titleLines: lineHeight > 0 ? titleRect.height / lineHeight : 99,
            };
        });
        const main = root.querySelector(':scope > .chat-live-summary-button .chat-live-summary-main').getBoundingClientRect();
        const side = root.querySelector(':scope > .chat-live-summary-button .chat-live-summary-side').getBoundingClientRect();
        return {
            messageWidth: usableMessageWidth,
            rootWidth: root.getBoundingClientRect().width,
            deepestWidth: deepest.getBoundingClientRect().width,
            rootMainBottom: main.bottom,
            rootSideTop: side.top,
            cardFacts,
        };
    }"""

    def assert_mobile_geometry(page):
        page.wait_for_selector(
            '#page-chat .chat-live-card[data-task-id="layout-root"]',
            state="attached",
            timeout=30_000,
        )
        page.wait_for_timeout(500)
        rendered_ids = page.locator("#page-chat .chat-live-card").evaluate_all(
            "cards => cards.map(card => card.dataset.taskId)"
        )
        assert len(rendered_ids) == 11, rendered_ids
        facts = page.evaluate(mobile_geometry)
        assert facts["rootWidth"] >= facts["messageWidth"] * 0.95, facts
        assert facts["rootWidth"] - facts["deepestWidth"] <= 40, facts
        assert facts["rootSideTop"] >= facts["rootMainBottom"] - 1, facts
        assert all(card["scrollWidth"] <= card["clientWidth"] + 1 for card in facts["cardFacts"]), facts
        assert min(card["titleWidth"] for card in facts["cardFacts"]) >= 160, facts
        assert max(card["titleLines"] for card in facts["cardFacts"]) <= 3.2, facts
        deepest = page.locator('.chat-live-card[data-task-id="layout-child-10"]')
        assert "pty-tests · gemini-3.6-flash" in deepest.inner_text()
        deepest.locator(":scope > [data-live-summary-button]").click()
        line_toggle = deepest.locator(":scope > [data-live-timeline] .chat-live-line-toggle").last
        line_toggle.wait_for(state="visible", timeout=5_000)
        line_toggle.click()
        expanded = deepest.evaluate(
            """card => {
                const line = card.querySelector(':scope > [data-live-timeline] .chat-live-line');
                const title = line.querySelector('.chat-live-line-title');
                return {
                    cardClient: card.clientWidth,
                    cardScroll: card.scrollWidth,
                    lineClient: line.clientWidth,
                    lineScroll: line.scrollWidth,
                    titleWidth: title.getBoundingClientRect().width,
                    text: line.innerText,
                };
            }"""
        )
        assert expanded["cardScroll"] <= expanded["cardClient"] + 1, expanded
        assert expanded["lineScroll"] <= expanded["lineClient"] + 1, expanded
        assert expanded["titleWidth"] >= 150, expanded
        assert long_url in expanded["text"], expanded

    try:
        with sync_playwright() as pw:
            browser_type = getattr(pw, browser_engine)
            try:
                browser = browser_type.launch(headless=True)
            except PlaywrightError as exc:
                if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
                    pytest.skip(f"Playwright {browser_engine} browser is not installed: {exc}")
                raise
            try:
                mobile_context = browser.new_context(
                    viewport={"width": 390, "height": 844},
                    is_mobile=True,
                    has_touch=True,
                )
                mobile = mobile_context.new_page()
                mobile.goto(url, wait_until="domcontentloaded", timeout=30_000)
                assert_mobile_geometry(mobile)
                mobile.screenshot(
                    path=str(data_dir.parent / f"live-card-depth-10-{browser_engine}.png"),
                    full_page=True,
                )
                mobile.reload(wait_until="domcontentloaded", timeout=30_000)
                assert_mobile_geometry(mobile)
                mobile_context.close()

                wide = browser.new_page(viewport={"width": 1280, "height": 900})
                wide.goto(url, wait_until="domcontentloaded", timeout=30_000)
                wide.wait_for_selector(
                    '#page-chat .chat-live-card[data-task-id="layout-root"]',
                    state="attached",
                    timeout=30_000,
                )
                wide.wait_for_timeout(500)
                rendered_ids = wide.locator("#page-chat .chat-live-card").evaluate_all(
                    "cards => cards.map(card => card.dataset.taskId)"
                )
                assert len(rendered_ids) == 11, rendered_ids
                wide_facts = wide.evaluate(
                    """() => {
                        const root = document.querySelector('#page-chat #chat-messages > .chat-live-card[data-task-id="layout-root"]');
                        const summary = root.querySelector(':scope > .chat-live-summary-button .chat-live-summary');
                        const main = summary.querySelector('.chat-live-summary-main').getBoundingClientRect();
                        const side = summary.querySelector('.chat-live-summary-side').getBoundingClientRect();
                        return {
                            wrap: getComputedStyle(summary).flexWrap,
                            mainTop: main.top,
                            mainBottom: main.bottom,
                            sideTop: side.top,
                            sideBottom: side.bottom,
                            rootClient: root.clientWidth,
                            rootScroll: root.scrollWidth,
                        };
                    }"""
                )
                assert wide_facts["wrap"] == "nowrap", wide_facts
                assert min(wide_facts["mainBottom"], wide_facts["sideBottom"]) \
                    > max(wide_facts["mainTop"], wide_facts["sideTop"]), wide_facts
                assert wide_facts["rootScroll"] <= wide_facts["rootClient"] + 1, wide_facts

                with (logs_dir / "progress.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(panel_row) + "\n")
                wide.evaluate(
                    "() => document.documentElement.style.setProperty('--project-panel-width', '440px')"
                )
                project_row = wide.locator('[data-project-id="layout-project"]')
                project_row.wait_for(state="visible", timeout=30_000)
                project_row.click()
                panel_card = wide.locator(
                    '.chat-instance-panel .chat-live-card[data-task-id="panel-root"]'
                )
                panel_card.wait_for(state="visible", timeout=30_000)
                panel_facts = panel_card.evaluate(
                    """card => {
                        const panel = card.closest('.chat-instance-panel');
                        const summary = card.querySelector(':scope > .chat-live-summary-button .chat-live-summary');
                        const main = summary.querySelector('.chat-live-summary-main').getBoundingClientRect();
                        const side = summary.querySelector('.chat-live-summary-side').getBoundingClientRect();
                        const title = summary.querySelector('[data-live-title]').getBoundingClientRect();
                        return {
                            panelWidth: panel.getBoundingClientRect().width,
                            cardWidth: card.getBoundingClientRect().width,
                            cardClient: card.clientWidth,
                            cardScroll: card.scrollWidth,
                            titleWidth: title.width,
                            mainBottom: main.bottom,
                            sideTop: side.top,
                        };
                    }"""
                )
                assert panel_facts["panelWidth"] <= 620, panel_facts
                assert panel_facts["cardWidth"] >= panel_facts["panelWidth"] * 0.9, panel_facts
                assert panel_facts["cardScroll"] <= panel_facts["cardClient"] + 1, panel_facts
                assert panel_facts["titleWidth"] >= 180, panel_facts
                assert panel_facts["sideTop"] >= panel_facts["mainBottom"] - 1, panel_facts
                wide.screenshot(
                    path=str(data_dir.parent / f"live-card-project-panel-{browser_engine}.png"),
                    full_page=True,
                )
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise


@pytest.mark.ui_browser
def test_ui_smoke_v639_skip_review_button(direct_server_with_data):
    # C1: the owner-only "⚠️ Skip review" action is offered for the owner's OWN (external)
    # skill and hash-verified official-hub payloads that still need review, and NEVER for
    # native/ClawHub/unverified marketplace payloads.
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    data_dir = direct_server_with_data["data_dir"]
    url = direct_server_with_data["url"]
    manifest = ("---\nname: {n}\ntype: instruction\ndescription: smoke skill\n"
                "version: 0.1.0\n---\n# {n}\nDo a thing.\n")
    ext = data_dir / "skills" / "external" / "owntool"
    ext.mkdir(parents=True, exist_ok=True)
    (ext / "SKILL.md").write_text(manifest.format(n="owntool"), encoding="utf-8")
    mk = data_dir / "skills" / "clawhub" / "markettool"
    mk.mkdir(parents=True, exist_ok=True)
    (mk / "SKILL.md").write_text(manifest.format(n="markettool"), encoding="utf-8")
    # A real marketplace skill carries clawhub provenance -> resolves to source=clawhub
    # (without it, an unprovenanced clawhub-bucket payload is treated as owner-own external).
    (mk / ".clawhub.json").write_text(
        json.dumps({"slug": "markettool", "version": "0.1.0"}), encoding="utf-8")
    # An already owner-attested skill: must show the distinct 'owner-attested' badge.
    att = data_dir / "skills" / "external" / "attestedtool"
    att.mkdir(parents=True, exist_ok=True)
    (att / "SKILL.md").write_text(manifest.format(n="attestedtool"), encoding="utf-8")
    att_state = data_dir / "state" / "skills" / "attestedtool"
    att_state.mkdir(parents=True, exist_ok=True)
    (att_state / "review.json").write_text(json.dumps({
        "status": "clean", "content_hash": "seed", "review_profile": "owner_attested",
        "reviewer_models": ["owner_attestation"],
        "findings": [{"item": "owner_attestation", "verdict": "PASS", "severity": "info", "reason": "owner attested"}],
    }), encoding="utf-8")
    (att_state / "owner_attestation.json").write_text(
        json.dumps({"attested_at": "now", "content_hash": "seed"}), encoding="utf-8")

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            try:
                page = browser.new_page(viewport={"width": 1280, "height": 900})
                page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                page.click('[data-nav-page="skills"]')
                page.wait_for_selector("#page-skills", timeout=30_000)
                page.wait_for_selector('.skills-card[data-skill="owntool"]', timeout=30_000)
                own = page.locator('.skills-card[data-skill="owntool"]').first
                market = page.locator('.skills-card[data-skill="markettool"]').first
                # owner-own external skill that still needs review -> Skip review offered.
                assert own.locator(".skills-attest-review").count() == 1
                assert "Skip review" in (
                    own.locator(".skills-attest-review").first.text_content() or "")
                # ClawHub marketplace skill -> never attestable, no Skip review action.
                assert market.locator(".skills-attest-review").count() == 0
                # owner-attested skill -> distinct 'owner-attested' badge (review_profile surfaced).
                page.wait_for_selector('.skills-card[data-skill="attestedtool"]', timeout=30_000)
                att_card = page.locator('.skills-card[data-skill="attestedtool"]').first
                assert att_card.locator(".skills-badge").filter(
                    has_text="owner-attested").count() >= 1
                # submitHubReady guard: an owner-attested skill must NOT offer an enabled
                # publish (the hub refuses to publish owner-attested skills). Render the card
                # WITH a github token configured (in-page module import — node exec is blocked)
                # and assert Submit-to-OuroborosHub is disabled for the owner-attested reason.
                submit_html = page.evaluate(
                    """async () => {
                        const m = await import('/static/modules/skill_card_renderer.js');
                        return m.renderInstalledSkillCard(
                            { name: 'att', type: 'instruction', version: '0.1.0', source: 'external',
                              is_self_authored: true, review_status: 'clean',
                              review_gate: { executable_review: true }, review_stale: false,
                              review_profile: 'owner_attested', grants: {}, permissions: [],
                              payload_root: 'skills/external/att', enabled: true },
                            new Set(), new Set(), {}, { githubTokenConfigured: true });
                    }"""
                )
                assert 'data-submit-disabled="true"' in submit_html
                assert "owner-attested" in submit_html.lower()
                # Defense-in-depth (mirrors the backend source gate): a marketplace skill
                # mislabeled self-authored must STILL NOT offer Skip review.
                market_self_html = page.evaluate(
                    """async () => {
                        const m = await import('/static/modules/skill_card_renderer.js');
                        return m.renderInstalledSkillCard(
                            { name: 'mk2', type: 'instruction', version: '0.1.0', source: 'clawhub',
                              is_self_authored: true, review_status: 'pending',
                              review_gate: { executable_review: false }, review_stale: false,
                              review_profile: '', grants: {}, permissions: [],
                              payload_root: 'skills/clawhub/mk2', enabled: false },
                            new Set(), new Set(), {}, {});
                    }"""
                )
                assert "skills-attest-review" not in market_self_html
                # Unverified OuroborosHub payloads also stay blocked; only the official_hub
                # profile is a cheap UI hint, and the backend still re-verifies.
                hub_html = page.evaluate(
                    """async () => {
                        const m = await import('/static/modules/skill_card_renderer.js');
                        return {
                          unverified: m.renderInstalledSkillCard(
                            { name: 'hub1', type: 'instruction', version: '0.1.0', source: 'ouroboroshub',
                              is_self_authored: false, review_status: 'pending',
                              review_gate: { executable_review: false }, review_stale: false,
                              review_profile: '', grants: {}, permissions: [],
                              payload_root: 'skills/ouroboroshub/hub1', enabled: false },
                            new Set(), new Set(), {}, {}),
                          verified: m.renderInstalledSkillCard(
                            { name: 'hub2', type: 'instruction', version: '0.1.0', source: 'ouroboroshub',
                              is_self_authored: false, review_status: 'pending',
                              review_gate: { executable_review: false }, review_stale: false,
                              review_profile: '', owner_attestable: true, official_hub_verified: true,
                              grants: {}, permissions: [],
                              payload_root: 'skills/ouroboroshub/hub2', enabled: false },
                            new Set(), new Set(), {}, {}),
                          staleProfile: m.renderInstalledSkillCard(
                            { name: 'hub3', type: 'instruction', version: '0.1.0', source: 'ouroboroshub',
                              is_self_authored: false, review_status: 'pending',
                              review_gate: { executable_review: false }, review_stale: true,
                              review_profile: 'official_hub', owner_attestable: false,
                              official_hub_verified: false, grants: {}, permissions: [],
                              payload_root: 'skills/ouroboroshub/hub3', enabled: false },
                            new Set(), new Set(), {}, {})
                        };
                    }"""
                )
                assert "skills-attest-review" not in hub_html["unverified"]
                assert "skills-attest-review" in hub_html["verified"]
                assert "skills-attest-review" not in hub_html["staleProfile"]
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise


@pytest.mark.ui_browser
def test_ui_smoke_v679_subagent_depth_zero_round_trips_through_settings(direct_server_with_data):
    """v6.79.0: the owner can actually reach, save, and re-read a Subagent Depth of 0.

    The structural fix (``_bounded_positive_int_setting`` honouring a configured 0) is
    unreachable if the visible control refuses the value or the load path rewrites it back to
    the fallback. This drives the real consumer flow — Settings -> Advanced -> type -> save ->
    full page reload — and pins the three neighbouring states so the fix cannot silently break
    them: 0 (no delegation), a normal positive value, and empty (falls back, does not persist
    an invalid value). Screenshots are written for vision inspection; a saved screenshot is
    not verification on its own (docs/DEVELOPMENT.md "Browser/mobile verification").
    """
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    data_dir = direct_server_with_data["data_dir"]
    settings_path = data_dir / "settings.json"
    evidence_dir = pathlib.Path(
        os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", str(data_dir.parent))
    )
    evidence_dir.mkdir(parents=True, exist_ok=True)

    def saved_depth():
        return json.loads(settings_path.read_text(encoding="utf-8")).get(
            "OUROBOROS_MAX_SUBAGENT_DEPTH", "<absent>"
        )

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1400, "height": 1000})
            try:
                def open_settings_advanced():
                    """Reload the whole app the way an owner would, then reopen the field."""
                    page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                    page.wait_for_selector('[data-nav-page="settings"]', timeout=30_000)
                    page.click('[data-nav-page="settings"]')
                    page.wait_for_selector("#s-subagent-depth", state="attached", timeout=30_000)
                    page.click('[data-settings-tab="advanced"]')
                    depth = page.locator("#s-subagent-depth")
                    depth.wait_for(state="visible", timeout=30_000)
                    # Saving is blocked until the settings load succeeds; waiting on the real
                    # enablement avoids racing the first fetch.
                    page.wait_for_function(
                        "() => document.querySelector('#btn-save-settings')?.disabled === false",
                        timeout=30_000,
                    )
                    depth.scroll_into_view_if_needed()
                    return depth

                def type_depth(value):
                    page.fill("#s-subagent-depth", value)
                    page.dispatch_event("#s-subagent-depth", "input")
                    page.dispatch_event("#s-subagent-depth", "change")

                def save_and_wait():
                    page.click("#btn-save-settings")
                    page.wait_for_function(
                        "() => (document.querySelector('#settings-status')?.textContent || '')"
                        ".includes('Settings saved')",
                        timeout=30_000,
                    )

                depth = open_settings_advanced()
                # The control must admit 0 at all: a min of 1 would make it unreachable.
                assert depth.get_attribute("min") == "0"
                assert depth.get_attribute("max") == "10"
                assert depth.is_enabled()
                assert depth.input_value() == "2"  # unset -> visible fallback
                page.screenshot(path=str(evidence_dir / "v679-depth-01-initial-unset.png"))

                # 0 is a valid value for the control, not a validation error.
                type_depth("0")
                assert page.evaluate(
                    "() => document.querySelector('#s-subagent-depth').validity.valid"
                ) is True
                assert page.evaluate(
                    "() => document.querySelector('#s-subagent-depth').validationMessage"
                ) == ""
                page.screenshot(path=str(evidence_dir / "v679-depth-02-typed-zero.png"))

                save_and_wait()
                assert page.locator("#s-subagent-depth").input_value() == "0"
                assert saved_depth() == 0
                page.screenshot(path=str(evidence_dir / "v679-depth-03-saved-zero.png"))

                # The round trip is the point: a reload must not rewrite 0 back to 2.
                open_settings_advanced()
                assert page.locator("#s-subagent-depth").input_value() == "0"
                assert saved_depth() == 0
                page.screenshot(path=str(evidence_dir / "v679-depth-04-reload-zero.png"))

                # Neighbouring state: an ordinary positive value still round-trips.
                type_depth("3")
                save_and_wait()
                assert saved_depth() == 3
                open_settings_advanced()
                assert page.locator("#s-subagent-depth").input_value() == "3"
                page.screenshot(path=str(evidence_dir / "v679-depth-05-reload-three.png"))

                # Neighbouring state: empty is not a value — it falls back to 2 rather than
                # persisting an unparsable setting.
                type_depth("")
                page.screenshot(path=str(evidence_dir / "v679-depth-06-empty-typed.png"))
                save_and_wait()
                assert saved_depth() == 2
                open_settings_advanced()
                assert page.locator("#s-subagent-depth").input_value() == "2"
                page.screenshot(path=str(evidence_dir / "v679-depth-07-reload-after-empty.png"))
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise
@pytest.mark.ui_browser
def test_ui_owner_context_mode_autolow_and_scope_review_ack(direct_server_with_data):
    """Owner-visible v6.80.0 flows, driven in a real browser (BIBLE P3 / UI verification rule).

    Two claimed-complete owner flows that source-string tests cannot certify:

    1. AUTO-LOW RE-SELECTION. When the effective `low` is a SYSTEM auto-downgrade, the segmented
       control already displays Low, so the old ``next === current`` short-circuit swallowed the
       click and the derived flag could never be cleared — an install whose route cannot be
       confirmed >=1M stayed wedged with scope review blocking every commit. Re-picking the
       displayed Low must still POST the idempotent owner endpoint.
    2. SCOPE-REVIEW CAPABILITY ACK. Saving a scope-review slot whose route has no >=1M evidence
       must raise the owner confirm and, on accept, persist a route-scoped capability ack and say
       so in the settings status line.
    """
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    data_dir = direct_server_with_data["data_dir"]
    settings_path = data_dir / "settings.json"
    evidence_dir = pathlib.Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", str(data_dir.parent)))
    evidence_dir.mkdir(parents=True, exist_ok=True)

    # Boot into the state under test: effective low that is a system auto-downgrade, not an
    # owner selection. The derived flag is disk-authored, so it must be in the file before start.
    seeded = json.loads(settings_path.read_text(encoding="utf-8"))
    seeded["OUROBOROS_CONTEXT_MODE"] = "low"
    seeded["OUROBOROS_CONTEXT_MODE_AUTO_LOW"] = "true"
    seeded["OUROBOROS_SCOPE_REVIEW_MODELS"] = seeded["OUROBOROS_MODEL"]
    settings_path.write_text(json.dumps(seeded), encoding="utf-8")
    direct_server_with_data["restart_server"]()

    with urllib.request.urlopen(f"{url}/api/state", timeout=5) as resp:  # noqa: S310 - local test server
        boot_state = json.loads(resp.read().decode("utf-8"))
    assert boot_state["context_mode"] == "low"
    assert boot_state["context_mode_auto_low"] is True, "the fixture must boot in the auto-downgraded state"

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            dialogs: list[str] = []
            page.on("dialog", lambda dialog: (dialogs.append(dialog.message), dialog.accept()))
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=60_000)
                toggle = page.locator("#chat-context-mode")
                toggle.wait_for(state="visible", timeout=60_000)
                page.wait_for_function(
                    "() => document.querySelector('#chat-context-mode')?.dataset.contextModeAutoLow === 'true'",
                    timeout=30_000,
                )
                assert toggle.get_attribute("data-context-mode") == "low"
                page.screenshot(path=str(evidence_dir / "v6800-autolow-before.png"))

                # The click the old short-circuit swallowed: Low is ALREADY displayed.
                toggle.locator('.chat-seg[data-mode="low"]').click()
                page.wait_for_function(
                    "() => document.querySelector('#chat-context-mode')?.dataset.contextModeAutoLow === 'false'",
                    timeout=30_000,
                )
                page.screenshot(path=str(evidence_dir / "v6800-autolow-after.png"))
                # It reached the owner endpoint: the derived flag is cleared on disk and on the wire.
                assert json.loads(settings_path.read_text(encoding="utf-8"))["OUROBOROS_CONTEXT_MODE_AUTO_LOW"] == "false"
                with urllib.request.urlopen(f"{url}/api/state", timeout=5) as resp:  # noqa: S310
                    after = json.loads(resp.read().decode("utf-8"))
                assert after["context_mode"] == "low", "confirming Low must not flip the horizon to max"
                assert after["context_mode_auto_low"] is False

                # 2. Scope-review capability notice -> owner confirm -> route-scoped ack.
                page.click('[data-nav-page="settings"]')
                page.wait_for_selector("#s-context-mode", state="attached", timeout=30_000)
                page.locator('[data-settings-tab="models"]').click()
                page.wait_for_selector("#s-scope-review-models", timeout=30_000)
                page.locator("#s-scope-review-models").fill("openai-compatible::scope-reviewer-x")
                page.locator("#btn-save-settings").click()
                page.wait_for_function(
                    "() => (document.querySelector('#settings-status')?.textContent || '')"
                    ".includes('scope-review route')",
                    timeout=60_000,
                )
                page.screenshot(path=str(evidence_dir / "v6800-scope-review-ack.png"), full_page=True)

                assert dialogs, "the owner was never asked to confirm the reviewer's window"
                assert "1,000,000-token context window" in dialogs[0]
                assert "openai-compatible::scope-reviewer-x" in dialogs[0], "the ack must name the exact route"
                status_text = page.locator("#settings-status").inner_text()
                assert "Confirmed a 1M-token window for 1 scope-review route(s)." in status_text
                evidence = json.loads((data_dir / "state" / "capability_evidence.json").read_text(encoding="utf-8"))
                acked = [
                    entry for entry in (evidence.get("acks") or evidence.get("probes") or {}).values()
                    if str(entry.get("model") or "") == "openai-compatible::scope-reviewer-x"
                ]
                assert acked, "no route-scoped capability evidence was stored for the acked reviewer"
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise


@pytest.mark.ui_browser
def test_ui_smoke_cancel_run_button_eligibility_and_cancelled_state(direct_server_with_data):
    """v6.82 P5: "Cancel run" renders ONLY on live marker-attested root cards
    (never on marker-less direct-turn-shaped cards, subagent children, or the
    reusable background slot), opens a confirm dialog, and a cancelled root
    replays as an honest warn-toned "Cancelled" — never a generic "Done"."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    data_dir = direct_server_with_data["data_dir"]
    logs_dir = data_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "chat.jsonl").write_text("", encoding="utf-8")
    rows = [
        # Pooled live root: carries the supervisor's host-attested marker.
        {"ts": "2026-07-29T10:00:00+00:00", "chat_id": 1, "task_id": "live-root",
         "content": "Working on the big thing", "cancelable": True},
        # Direct-chat-turn shape: same card shape, NO marker -> no button.
        {"ts": "2026-07-29T10:00:01+00:00", "chat_id": 1, "task_id": "direct-turn",
         "content": "Inline turn narration"},
        # Subagent child of the live root: marker present but child cards never
        # offer the action (the root cascade covers them).
        {"ts": "2026-07-29T10:00:02+00:00", "chat_id": 1, "task_id": "sub-child1",
         "content": "Collecting evidence", "delegation_role": "subagent",
         "subagent_event": "scheduled", "subagent_task_id": "sub-child1",
         "parent_task_id": "live-root", "subagent_role": "researcher",
         "cancelable": True},
        # Reusable background-consciousness slot: never eligible.
        {"ts": "2026-07-29T10:00:03+00:00", "chat_id": 1, "task_id": "bg-consciousness",
         "content": "Background thinking", "cancelable": True},
        # A root that was force-cancelled before this reload.
        {"ts": "2026-07-29T10:00:04+00:00", "chat_id": 1, "task_id": "gone-root",
         "content": "Was working before the cancel", "cancelable": True},
    ]
    (logs_dir / "progress.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8",
    )
    task_results = data_dir / "task_results"
    task_results.mkdir(parents=True, exist_ok=True)
    (task_results / "gone-root.json").write_text(json.dumps({
        "task_id": "gone-root",
        "status": "cancelled",
        "reason_code": "cancelled",
        "outcome_axes": {
            "lifecycle": {"status": "cancelled"},
            "execution": {"status": "cancelled"},
        },
    }) + "\n", encoding="utf-8")

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                live = page.locator('.chat-live-card[data-task-id="live-root"]')
                live.wait_for(state="attached", timeout=30_000)
                cancel_btn = live.locator('[data-cancel-run]')
                cancel_btn.wait_for(state="attached", timeout=30_000)
                assert cancel_btn.inner_text().strip() == "Cancel run"
                # Marker-less direct-turn shape, subagent child, reusable slot,
                # and the finished cancelled root must NOT offer the action.
                for absent_id in ("direct-turn", "sub-child1", "bg-consciousness", "gone-root"):
                    card = page.locator(f'.chat-live-card[data-task-id="{absent_id}"]')
                    card.wait_for(state="attached", timeout=30_000)
                    assert card.locator('[data-cancel-run]').count() == 0, absent_id
                # The cancelled root replays as an honest Cancelled state.
                gone_phase = page.locator('.chat-live-card[data-task-id="gone-root"] [data-live-phase]')
                page.wait_for_function(
                    "() => document.querySelector('.chat-live-card[data-task-id=\"gone-root\"]"
                    " [data-live-phase]')?.textContent === 'Cancelled'",
                    timeout=30_000,
                )
                assert "cancelled" in (gone_phase.get_attribute("class") or "")
                # Confirm dialog wiring: open, then keep the run running.
                cancel_btn.click()
                dialog = page.locator('.confirm-dialog')
                dialog.wait_for(state="visible", timeout=10_000)
                assert "Cancel this run and all its subagents?" in dialog.inner_text()
                dialog.locator('[data-confirm-cancel]').last.click()
                dialog.wait_for(state="detached", timeout=10_000)
                assert cancel_btn.is_enabled()
                page.screenshot(path=str(data_dir.parent / "cancel-run.png"), full_page=True)
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise
