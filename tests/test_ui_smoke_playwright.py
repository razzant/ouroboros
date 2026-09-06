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
from tests.ui_chat_viewport_smoke import _CAPTURE_TEST_SOCKET, _emit_ws_frame

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))


def _open_review_checkpoint(card, *, open_card=True):
    if open_card:
        card.locator(":scope > [data-live-summary-button]").click()
    assert card.is_visible()
    section = card.locator(":scope > [data-live-reviews-host] [data-review-section]")
    section.wait_for(state="visible", timeout=5_000)
    assert section.get_attribute("data-expanded") == "0"
    section.locator("[data-review-section-toggle]").click()
    assert section.get_attribute("data-expanded") == "1"
    group = section.locator("[data-review-group]").first
    assert group.locator("[data-review-group-toggle]").get_attribute("aria-expanded") == "false"
    group.locator("[data-review-group-toggle]").click()
    attempt = group.locator("[data-review-attempt-toggle]").first
    attempt.click()
    assert attempt.get_attribute("aria-expanded") == "true"


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


@pytest.mark.ui_browser
def test_ui_smoke_project_panel_lifecycle_does_not_leak(direct_server_with_data):
    """Open/close cycles keep one live panel, flat ws listeners, and flat DOM.

    P3 lifecycle concrete: closing or switching a project DESTROYS its chat
    instance (disposing every ws.on subscription, the ResizeObserver, the
    window/document listeners, and all timers), so repeated open/close cycles
    cannot accumulate hidden panels, listeners, or DOM nodes. Panels marked
    data-pending-work (staged attachments / in-flight upload) are the one
    sanctioned exception and are excluded from the live-panel count.
    """
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    from ouroboros.projects_registry import create_project

    url = direct_server_with_data["url"]
    data_dir = direct_server_with_data["data_dir"]
    project_ids = [f"leak-{idx}" for idx in range(1, 4)]
    for idx, project_id in enumerate(project_ids, start=1):
        create_project(data_dir, project_id, name=f"Leak project {idx}")

    # window.__ouroWs is the loopback debug hook app.js exposes for exactly
    # this count; the module-scoped ws is unreachable from page.evaluate.
    count_listeners = """() => {
        const ws = window.__ouroWs;
        return Object.values(ws.listeners).reduce((total, set) => total + set.size, 0);
    }"""
    live_panels = """() => [...document.querySelectorAll('.chat-instance-panel')]
        .filter((panel) => panel.dataset.pendingWork !== '1').length"""
    dom_count = "() => document.getElementsByTagName('*').length"

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                for project_id in project_ids:
                    page.wait_for_selector(
                        f'.nav-project-row[data-project-id="{project_id}"]', timeout=30_000
                    )

                def open_project(project_id):
                    page.click(f'.nav-project-row[data-project-id="{project_id}"]')
                    page.wait_for_selector("#project-panel:not([hidden])", timeout=30_000)
                    page.wait_for_selector(
                        f'[id="panel-pchat-{project_id}"]:not([hidden])', timeout=30_000
                    )

                def close_project():
                    page.click("#project-panel-close")
                    page.wait_for_function(
                        "() => !document.getElementById('project-panel')"
                        ".classList.contains('open')",
                        timeout=30_000,
                    )

                # Baseline AFTER one full open/close cycle so one-time lazy
                # registrations cannot masquerade as leaks.
                open_project(project_ids[0])
                close_project()
                listeners_baseline = page.evaluate(count_listeners)
                dom_baseline = page.evaluate(dom_count)
                assert listeners_baseline > 0

                # Small slack for churn outside the panel (badges, toasts);
                # a leaked panel or card timeline is hundreds of nodes.
                dom_slack = 30
                for project_id in project_ids:
                    open_project(project_id)
                    assert page.evaluate(live_panels) <= 1
                    close_project()
                    assert page.evaluate(live_panels) == 0
                    # Every cycle returns to the baseline: no monotonic growth.
                    cycle_dom = page.evaluate(dom_count)
                    assert cycle_dom <= dom_baseline + dom_slack, (dom_baseline, cycle_dom)
                    assert page.evaluate(count_listeners) == listeners_baseline

                # Direct project-to-project switch (no explicit close) also
                # destroys the previous instance: one live panel, ever.
                open_project(project_ids[0])
                open_project(project_ids[1])
                assert page.evaluate(live_panels) == 1
                close_project()
                assert page.evaluate(live_panels) == 0

                assert page.evaluate(count_listeners) == listeners_baseline
                final_dom = page.evaluate(dom_count)
                assert final_dom <= dom_baseline + dom_slack, (dom_baseline, final_dom)
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
                assert metric.evaluate("element => getComputedStyle(element).borderLeftColor") == "rgb(110, 231, 183)"
                assert callout.evaluate("element => getComputedStyle(element).borderLeftColor") == "rgb(252, 211, 77)"

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
                canvas_box = chart.bounding_box()
                assert canvas_box and 250 <= canvas_box["height"] <= 370, canvas_box
                chart_config = json.loads(chart.get_attribute("data-widget-chart-config"))
                assert chart_config["data"]["datasets"][0]["data"] == [74, None, 91]
                assert chart_config["data"]["datasets"][0]["spanGaps"] is False
                assert chart_config["options"]["spanGaps"] is False
                assert chart.get_attribute("aria-label") == "Cache hit rate with an intentional gap"
                # Poll refetch adopts the live canvas and preserves stale content.
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
@pytest.mark.parametrize("browser_engine", ["chromium", "webkit"])
def test_ui_smoke_collapsed_activity_line_named_vs_unnamed(
    direct_server_with_data,
    browser_engine,
):
    """Collapsed root summaries stay compact without destroying full activity."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    data_dir = direct_server_with_data["data_dir"]
    logs_dir = data_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "chat.jsonl").write_text("", encoding="utf-8")
    unique_tail = "UNIQUE_FULL_ACTIVITY_TAIL"
    long_activity = (
        "Analyzing the dataset and comparing every source. " * 18
        + "https://example.com/" + "unbroken-segment-" * 18 + unique_tail
    )
    (logs_dir / "progress.jsonl").write_text(
        json.dumps({
            "ts": "2026-07-29T10:00:00+00:00",
            "chat_id": 1,
            "task_id": "named-act",
            "content": long_activity,
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
        "_schema_version": 1,
        "task_id": "named-act",
        "status": "completed",
        "suggested_name": "Data Analysis",
        "cost_usd": 0.42,
        "cost_accounting_status": "available",
        "cost_final": True,
    }) + "\n", encoding="utf-8")
    try:
        with sync_playwright() as pw:
            browser_type = getattr(pw, browser_engine)
            try:
                browser = browser_type.launch(headless=True)
            except PlaywrightError as exc:
                if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
                    pytest.fail(f"required Playwright {browser_engine} browser is not installed: {exc}")
                raise
            try:
                for width, height, mobile in [(1440, 1000, False), (390, 844, True)]:
                    context = browser.new_context(
                        viewport={"width": width, "height": height},
                        is_mobile=mobile,
                        has_touch=mobile,
                    )
                    page = context.new_page()
                    page.add_init_script(f"({_CAPTURE_TEST_SOCKET})()")
                    page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                    page.wait_for_function(
                        "() => window.__testSockets?.some(socket => socket.readyState === WebSocket.OPEN)",
                        timeout=30_000,
                    )
                    named = page.locator('.chat-live-card[data-task-id="named-act"]')
                    named.wait_for(state="attached", timeout=30_000)
                    unnamed = page.locator('.chat-live-card[data-task-id="unnamed-act"]')
                    unnamed.wait_for(state="attached", timeout=30_000)
                    page.wait_for_function(
                        "() => document.querySelector('.chat-live-card[data-task-id=\"named-act\"]"
                        " [data-live-title]')?.textContent === 'Data Analysis'",
                        timeout=30_000,
                    )
                    _emit_ws_frame(page, {
                        "type": "chat", "role": "assistant", "is_progress": True,
                        "chat_id": 1, "task_id": "running-act", "suggested_name": "Short task",
                        "content": "Short update", "ts": "2026-07-29T10:00:02+00:00",
                    })
                    running = page.locator('.chat-live-card[data-task-id="running-act"]')
                    running.wait_for(state="attached", timeout=30_000)
                    assert running.locator('[data-live-title]').text_content().strip() == "Short task"
                    assert named.get_attribute("data-expanded") == "0"
                    named_activity = named.locator('[data-live-activity]')
                    activity_text = named_activity.text_content().strip()
                    assert activity_text
                    assert len(activity_text) <= 240
                    assert activity_text.endswith(("…", "..."))
                    assert unique_tail not in activity_text
                    assert named_activity.get_attribute("title") is None
                    geometry = named.evaluate(
                        """card => {
                            const facts = selector => {
                                const el = card.querySelector(selector);
                                const style = getComputedStyle(el);
                                const lineHeight = parseFloat(style.lineHeight);
                                const rect = el.getBoundingClientRect();
                                return { lines: rect.height / lineHeight, width: rect.width };
                            };
                            return {
                                title: facts('[data-live-title]'),
                                activity: facts('[data-live-activity]'),
                                clientWidth: card.clientWidth,
                                scrollWidth: card.scrollWidth,
                            };
                        }"""
                    )
                    assert geometry["title"]["lines"] <= 2.2, geometry
                    assert geometry["activity"]["lines"] <= 2.2, geometry
                    assert geometry["scrollWidth"] <= geometry["clientWidth"] + 1, geometry
                    named_meta = named.locator('[data-live-meta]').inner_text().split()
                    # Final ledger: the plain amount, never the open-ledger ceiling.
                    assert "$0.42" in named_meta and "up" not in named_meta, named_meta

                    unnamed_activity = unnamed.locator('[data-live-activity]')
                    assert "Doing things without a name" in unnamed.locator('[data-live-title]').text_content()
                    assert unnamed_activity.text_content().strip() == ""
                    assert not unnamed_activity.is_visible()
                    bands = page.evaluate(
                        """() => Object.fromEntries(['named-act', 'unnamed-act', 'running-act'].map(id => {
                            const card = document.querySelector(`.chat-live-card[data-task-id="${id}"]`);
                            const facts = selector => {
                                const node = card.querySelector(selector);
                                const style = getComputedStyle(node);
                                const height = node.getBoundingClientRect().height;
                                const lineHeight = parseFloat(style.lineHeight);
                                return {height, lines: height / lineHeight,
                                    display: style.display, visibility: style.visibility};
                            };
                            return [id, {
                                title: facts('[data-live-title]'),
                                activity: facts('[data-live-activity]'),
                                meta: facts('[data-live-meta]'),
                                finished: card.dataset.finished === '1',
                                clipped: card.scrollHeight > card.clientHeight + 1,
                                reviews: card.querySelector('[data-live-review-summary]')?.textContent || '',
                            }];
                        }))"""
                    )
                    assert bands["named-act"]["finished"] is True, bands
                    assert bands["unnamed-act"]["finished"] is False, bands
                    assert bands["running-act"]["finished"] is False, bands
                    for slot, low in (("title", 0.9), ("activity", 1.9)):
                        heights = [bands[task_id][slot]["height"] for task_id in bands]
                        assert max(heights) - min(heights) <= 1, bands
                        assert all(low <= bands[task_id][slot]["lines"] <= low + 0.3 for task_id in bands), bands
                    assert bands["unnamed-act"]["activity"]["display"] != "none", bands
                    assert bands["unnamed-act"]["activity"]["visibility"] == "hidden", bands
                    # D23 (owner, 2026-09-02): a FINISHED card folds an empty activity
                    # band; a running card keeps the two-line reserve (31.08 seam).
                    _emit_ws_frame(page, {
                        "type": "chat", "role": "assistant", "is_progress": True,
                        "chat_id": 1, "task_id": "done-empty", "suggested_name": "Quick task",
                        "content": "", "ts": "2026-07-29T10:00:03+00:00",
                    })
                    done_empty = page.locator('.chat-live-card[data-task-id="done-empty"]')
                    done_empty.wait_for(state="attached", timeout=30_000)
                    _emit_ws_frame(page, {
                        "type": "chat", "role": "system", "system_type": "task_summary",
                        "chat_id": 1, "task_id": "done-empty", "content": "Done",
                        "ts": "2026-07-29T10:00:04+00:00",
                    })
                    page.wait_for_function(
                        "() => document.querySelector('.chat-live-card[data-task-id=\"done-empty\"]')"
                        "?.dataset.finished === '1'",
                        timeout=30_000,
                    )
                    fold = done_empty.evaluate(
                        """card => { const node = card.querySelector('[data-live-activity]');
                            return {text: node.textContent, display: getComputedStyle(node).display,
                                height: card.getBoundingClientRect().height}; }"""
                    )
                    assert fold["text"].strip() == "", fold
                    assert fold["display"] == "none", fold
                    running_height = running.evaluate("el => el.getBoundingClientRect().height")
                    assert fold["height"] <= running_height - 20, (fold, running_height)
                    assert all(bands[task_id]["meta"]["lines"] >= 0.9 for task_id in bands), bands
                    button_height = "el => el.getBoundingClientRect().height"
                    before_reviews = running.locator(":scope > [data-live-summary-button]").evaluate(button_height)
                    # A root card's acceptance evidence rides the log channel (task detail seam).
                    _emit_ws_frame(page, {"type": "log", "chat_id": 1, "data": {
                        "type": "task_metrics_event", "task_id": "running-act",
                        "ts": "2026-07-29T10:00:03+00:00",
                        "review_projection": {"panels": [{
                            "panel_id": "act-review", "surface": "task_acceptance",
                            "aggregate_signal": "PASS", "reason": "smoke", "actors": [],
                        }]},
                    }})
                    page.wait_for_function(
                        "() => document.querySelector('.chat-live-card[data-task-id=\"running-act\"]"
                        " [data-live-review-summary]')?.textContent === 'Reviews 1'",
                        timeout=10_000,
                    )
                    row = running.evaluate(
                        """card => {
                            const btn = card.querySelector(':scope > [data-live-summary-button]');
                            const meta = btn.querySelector('[data-live-meta]').getBoundingClientRect();
                            const review = btn.querySelector('[data-live-review-summary]').getBoundingClientRect();
                            return {height: btn.getBoundingClientRect().height, metaTop: meta.top,
                                reviewTop: review.top, reviewRight: review.right,
                                buttonRight: btn.getBoundingClientRect().right};
                        }"""
                    )
                    # The quiet count shares the metadata row, docked right, without a new row.
                    assert abs(row["reviewTop"] - row["metaTop"]) <= 2, row
                    assert row["buttonRight"] - row["reviewRight"] <= 20, row
                    assert abs(row["height"] - before_reviews) <= 1, row
                    assert all(not bands[task_id]["clipped"] for task_id in bands), bands
                    assert all("Reviews" not in bands[task_id]["reviews"] for task_id in bands), bands

                    named.locator(':scope > [data-live-summary-button]').click()
                    line_toggle = named.locator(':scope > [data-live-timeline] .chat-live-line-toggle').first
                    line_toggle.wait_for(state="visible", timeout=5_000)
                    line_toggle.click()
                    page.wait_for_function(
                        "tail => document.querySelector('.chat-live-card[data-task-id=\"named-act\"]')"
                        ".innerText.includes(tail)",
                        arg=unique_tail,
                    )
                    page.screenshot(
                        path=str(data_dir.parent / f"compact-activity-{browser_engine}-{width}.png"),
                        full_page=True,
                    )
                    context.close()
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.fail(f"required Playwright {browser_engine} browser is not installed: {exc}")
        raise


@pytest.mark.ui_browser
@pytest.mark.parametrize("browser_engine", ["chromium", "webkit"])
def test_ui_smoke_live_card_mutations_preserve_viewport(
    direct_server_with_data,
    browser_engine,
):
    from tests.ui_chat_viewport_smoke import run_chat_viewport_smoke

    run_chat_viewport_smoke(direct_server_with_data, browser_engine)

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
                    "_schema_version": 1,
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
    child_activity_tail = "UNIQUE_CHILD_ACTIVITY_TAIL"
    child_activity_early = "UNIQUE_CHILD_ACTIVITY_EARLY"
    child_activity = (
        child_activity_early + " Searching evidence across repositories.\n"
        + "Comparing every source and preserving the complete routed narration. " * 14
        + "\nhttps://example.com/" + "child-evidence-segment-" * 14 + child_activity_tail
    )
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
            # Real rejection carrier shape: task_id addresses the parent chat,
            # while subagent_task_id is the child presentation identity.
            "ts": "2026-05-25T10:00:01.500000+00:00",
            "chat_id": 1,
            "task_id": "parent1",
            "content": "Rejected child should stay a child",
            "is_progress": True,
            "delegation_role": "subagent",
            "subagent_event": "rejected",
            "subagent_task_id": "rejected1",
            "parent_task_id": "parent1",
            "root_task_id": "parent1",
            "subagent_role": "rejected-reader",
            "status": "rejected_duplicate",
            "error": "Active-subagent cap rejected this child",
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
            "content": child_activity,
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
            "delegation_role": "subagent",
            "subagent_event": "completed",
            "subagent_task_id": "child1",
            "parent_task_id": "parent1",
            "root_task_id": "parent1",
            "subagent_role": "researcher",
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
                page.wait_for_function("() => document.querySelectorAll('.chat-live-card').length === 4", timeout=30_000)
                page.wait_for_function(
                    "() => { const p = document.querySelector('.chat-live-card:not(.subagent)');"
                    " const c = document.querySelector('.chat-live-card.subagent[data-parent-task-id=\"parent1\"]');"
                    " const g = document.querySelector('.chat-live-card.subagent[data-parent-task-id=\"child1\"]');"
                    " return !!p && !!c && c.closest('.chat-subagents') && c.parentElement.closest('.chat-live-card') === p"
                    " && !!g && g.closest('.chat-subagents') && g.parentElement.closest('.chat-live-card') === c"
                    " && /researcher/.test(c.innerText)"
                    " && /evidence-mapper/.test(g.innerText); }",
                    timeout=30_000,
                )
                parent = page.locator(".chat-live-card:not(.subagent)").first
                child = page.locator('.chat-live-card.subagent[data-task-id="child1"]')
                rejected_child = page.locator('.chat-live-card.subagent[data-task-id="rejected1"]')
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
                assert "2 children" in parent_count.inner_text()
                assert "researcher" in child_text and "(child1)" not in child_text
                assert "1 child" in child_count.inner_text()
                assert "child=child1" not in child_text
                assert "role=researcher" not in child_text
                assert "panel_child_review" not in child_text
                assert "claude-fable-5" not in child_text
                assert "verdict=DEGRADED" not in child_text
                assert "evidence-mapper" in grandchild.inner_text()
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
                assert rejected_child.get_attribute("data-finished") == "1"
                assert rejected_child.locator(
                    ":scope > [data-live-summary-button] [data-live-phase]"
                ).get_attribute("data-phase") == "warn"
                assert child.get_attribute("data-finished") == "1"
                assert child.locator(":scope > [data-live-summary-button] [data-live-phase]").first.get_attribute("data-phase") == "warn"
                assert child.get_attribute("data-subagent-role") == "researcher"
                child_activity_el = child.locator(":scope > [data-live-summary-button] [data-live-activity]")
                assert len(child_activity_el.text_content().strip()) <= 240
                assert child_activity_tail not in child_activity_el.text_content()
                assert child_activity_el.get_attribute("title") is None
                assert grandchild.get_attribute("data-finished") == "1"
                assert grandchild.get_attribute("data-subagent-role") == "evidence-mapper"
                assert page.locator(".chat-bubble.progress").count() == 0
                assert page.locator(".chat-bubble").filter(
                    has_text="Final child answer should stay inside the child card."
                ).count() == 0

                assert child.get_attribute("data-expanded") == "0"
                assert grandchild.get_attribute("data-expanded") == "0"
                child_summary = child.locator(":scope > [data-live-summary-button]").first
                child_summary.click()
                _open_review_checkpoint(child, open_card=False)
                assert "panel_child_review" in child.inner_text()
                assert "claude-fable-5" in child.inner_text()
                assert "verdict=DEGRADED" in child.inner_text()
                progress_line = child.locator(".chat-live-line", has_text="Searching evidence").first
                progress_toggle = progress_line.locator(".chat-live-line-toggle")
                progress_toggle.wait_for(state="visible", timeout=5_000)
                progress_toggle.click()
                assert child_activity_early in progress_line.inner_text()
                assert child_activity_tail in progress_line.inner_text()
                result_line = child.locator(".chat-live-line", has_text="Child result with evidence table").first
                result_toggle = result_line.locator(".chat-live-line-toggle")
                result_toggle.wait_for(state="visible", timeout=5_000)
                result_toggle.click()
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
                assert result_toggle.get_attribute("aria-controls")

                page.reload(wait_until="domcontentloaded", timeout=30_000)
                page.wait_for_function("() => document.querySelectorAll('.chat-live-card').length === 4", timeout=30_000)
                page.wait_for_function(
                    "() => { const p = document.querySelector('.chat-live-card:not(.subagent)');"
                    " const c = document.querySelector('.chat-live-card.subagent[data-parent-task-id=\"parent1\"]');"
                    " const g = document.querySelector('.chat-live-card.subagent[data-parent-task-id=\"child1\"]');"
                    " return !!p && !!c && c.closest('.chat-subagents') && c.parentElement.closest('.chat-live-card') === p"
                    " && !!g && g.closest('.chat-subagents') && g.parentElement.closest('.chat-live-card') === c; }",
                    timeout=30_000,
                )
                replay_parent = page.locator(".chat-live-card:not(.subagent)").first
                replay_child = page.locator('.chat-live-card.subagent[data-task-id="child1"]')
                replay_rejected = page.locator('.chat-live-card.subagent[data-task-id="rejected1"]')
                replay_grandchild = page.locator('.chat-live-card.subagent[data-parent-task-id="child1"]').first
                assert replay_parent.get_attribute("data-finished") == "0"
                assert replay_rejected.get_attribute("data-finished") == "1"
                assert replay_rejected.locator(
                    ":scope > [data-live-summary-button] [data-live-phase]"
                ).get_attribute("data-phase") == "warn"
                assert replay_child.get_attribute("data-finished") == "1"
                assert replay_child.locator(":scope > [data-live-summary-button] [data-live-phase]").first.get_attribute("data-phase") == "warn"
                assert replay_grandchild.get_attribute("data-finished") == "1"
                assert replay_child.get_attribute("data-expanded") == "0"
                assert replay_grandchild.get_attribute("data-expanded") == "0"
                assert "researcher" in replay_child.inner_text()
                assert "child=child1" not in replay_child.inner_text()
                assert "role=researcher" not in replay_child.inner_text()
                assert page.locator(".chat-bubble").filter(
                    has_text="Final child answer should stay inside the child card."
                ).count() == 0
                _open_review_checkpoint(replay_child)
                assert "Final child answer should stay inside the child card." in replay_child.inner_text()
                replay_progress = replay_child.locator(".chat-live-line", has_text="Searching evidence").first
                replay_progress.locator(".chat-live-line-toggle").click()
                assert child_activity_early in replay_progress.inner_text()
                assert child_activity_tail in replay_progress.inner_text()
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
                page.wait_for_function("() => document.querySelectorAll('.chat-live-card').length === 4", timeout=30_000)
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
    # Wait for the drawer to actually arrive instead of sleeping past the 180ms
    # transform transition: a fixed 220ms budget left ~40ms of margin and lost
    # that race on the Linux WebKit runner, which then measured the drawer at its
    # closed position (-105% => left -336) and failed. A drawer that never opens
    # still fails here, now naming the cause instead of a stale geometry read.
    # The predicate is byte-for-byte the one asserted below, so the wait can
    # never pass on a value the assertion would reject (a rounded variant let
    # left=-1.017 through and failed one line later).
    page.wait_for_function(
        "() => document.querySelector('#primary-sidebar')"
        ".getBoundingClientRect().left >= -1",
        timeout=5_000,
    )

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
    cover replay. Launcher-default Main and a narrow Project panel prove the
    card-local container responds to its actual consumer width rather than the
    viewport.
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
            const activity = card.querySelector(':scope > .chat-live-summary-button [data-live-activity]');
            const style = getComputedStyle(title);
            const lineHeight = parseFloat(style.lineHeight);
            const titleRect = title.getBoundingClientRect();
            const activityStyle = getComputedStyle(activity);
            const activityLineHeight = parseFloat(activityStyle.lineHeight);
            const activityRect = activity.getBoundingClientRect();
            return {
                id: card.dataset.taskId,
                clientWidth: card.clientWidth,
                scrollWidth: card.scrollWidth,
                titleWidth: titleRect.width,
                titleHeight: titleRect.height,
                titleLines: lineHeight > 0 ? titleRect.height / lineHeight : 99,
                activityLines: activityLineHeight > 0 ? activityRect.height / activityLineHeight : 99,
                activityTitle: activity.getAttribute('title'),
            };
        });
        const main = root.querySelector(':scope > .chat-live-summary-button .chat-live-summary-main').getBoundingClientRect();
        const side = root.querySelector(':scope > .chat-live-summary-button .chat-live-summary-side').getBoundingClientRect();
        const title = root.querySelector(':scope > .chat-live-summary-button [data-live-title]').getBoundingClientRect();
        return {
            messageWidth: usableMessageWidth,
            rootWidth: root.getBoundingClientRect().width,
            deepestWidth: deepest.getBoundingClientRect().width,
            rootMainBottom: main.bottom,
            rootSideTop: side.top,
            rootSideBottom: side.bottom,
            rootTitleTop: title.top,
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
        # Narrow regime: the side controls share the chip row, the title takes its own
        # full-width row below both.
        assert facts["rootSideTop"] < facts["rootMainBottom"], facts
        assert facts["rootTitleTop"] >= max(facts["rootMainBottom"], facts["rootSideBottom"]) - 1, facts
        assert all(card["scrollWidth"] <= card["clientWidth"] + 1 for card in facts["cardFacts"]), facts
        assert min(card["titleWidth"] for card in facts["cardFacts"]) >= 160, facts
        assert 0.9 <= min(card["titleLines"] for card in facts["cardFacts"]), facts
        assert max(card["titleLines"] for card in facts["cardFacts"]) <= 2.2, facts
        assert max(card["activityLines"] for card in facts["cardFacts"]) <= 2.2, facts
        assert all(card["activityTitle"] is None for card in facts["cardFacts"]), facts
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

    def assert_jump_geometry(page, scope_selector, *, require_overflow=True):
        page.evaluate(
            """({scopeSelector, requireOverflow}) => {
                const scope = document.querySelector(scopeSelector);
                const messages = scope
                    .querySelector('.chat-messages, #chat-messages');
                messages.scrollTop = 0;
                messages.dispatchEvent(new Event('scroll'));
            }""",
            {"scopeSelector": scope_selector, "requireOverflow": require_overflow},
        )
        page.wait_for_timeout(250)
        facts = page.evaluate(
            """({scopeSelector, forceVisible}) => {
                const scope = document.querySelector(scopeSelector);
                const messages = scope.querySelector('.chat-messages, #chat-messages');
                const button = scope.querySelector('.chat-scroll-bottom-btn');
                const priorTransition = button.style.transition;
                button.style.transition = 'none';
                if (forceVisible) button.classList.add('visible');
                const wrap = scope.querySelector('.chat-input-wrap');
                const toolbar = scope.querySelector('.chat-toolbar-row');
                const preview = scope.querySelector('.chat-attachment-preview.visible');
                const b = button.getBoundingClientRect();
                const w = wrap.getBoundingClientRect();
                const t = toolbar.getBoundingClientRect();
                const overlap = (a, c) => Math.max(0, Math.min(a.right, c.right) - Math.max(a.left, c.left))
                    * Math.max(0, Math.min(a.bottom, c.bottom) - Math.max(a.top, c.top));
                const facts = {
                    remaining: messages.scrollHeight - messages.scrollTop - messages.clientHeight,
                    visible: button.classList.contains('visible'),
                    width: b.width, height: b.height,
                    centerDelta: Math.abs((b.left + b.width / 2) - (w.left + w.width / 2)),
                    gap: w.top - b.bottom,
                    toolbarOverlap: overlap(b, t),
                    previewOverlap: preview ? overlap(b, preview.getBoundingClientRect()) : 0,
                    boxShadow: getComputedStyle(button).boxShadow,
                };
                if (priorTransition) button.style.transition = priorTransition;
                else button.style.removeProperty('transition');
                return facts;
            }""",
            {"scopeSelector": scope_selector, "forceVisible": not require_overflow},
        )
        assert facts["visible"], facts
        if require_overflow:
            assert facts["remaining"] > 48, facts
        assert abs(facts["width"] - 32) <= 1 and abs(facts["height"] - 32) <= 1, facts
        assert facts["centerDelta"] <= 1 and abs(facts["gap"] - 8) <= 1, facts
        assert facts["toolbarOverlap"] == 0 and facts["previewOverlap"] == 0, facts
        assert facts["boxShadow"] == "none", facts
        return facts

    try:
        with sync_playwright() as pw:
            browser_type = getattr(pw, browser_engine)
            try:
                browser = browser_type.launch(headless=True)
            except PlaywrightError as exc:
                if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
                    pytest.fail(f"required Playwright {browser_engine} browser is not installed: {exc}")
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
                assert_jump_geometry(mobile, "#page-chat")
                mobile.screenshot(
                    path=str(data_dir.parent / f"live-card-depth-10-{browser_engine}.png"),
                    full_page=True,
                )
                mobile.reload(wait_until="domcontentloaded", timeout=30_000)
                assert_mobile_geometry(mobile)
                mobile_context.close()

                wide = browser.new_page(viewport={"width": 1100, "height": 750})
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
                        const ids = ['layout-root', 'layout-child-01', 'layout-child-02'];
                        return ids.map((id) => {
                            const card = document.querySelector(`#page-chat .chat-live-card[data-task-id="${id}"]`);
                            const summary = card.querySelector(':scope > .chat-live-summary-button .chat-live-summary');
                            const main = summary.querySelector('.chat-live-summary-main').getBoundingClientRect();
                            const side = summary.querySelector('.chat-live-summary-side').getBoundingClientRect();
                            const rect = card.getBoundingClientRect();
                            return {
                                id,
                                left: rect.left,
                                width: rect.width,
                                wrap: getComputedStyle(summary).flexWrap,
                                mainTop: main.top,
                                mainBottom: main.bottom,
                                sideTop: side.top,
                                sideBottom: side.bottom,
                                client: card.clientWidth,
                                scroll: card.scrollWidth,
                            };
                        });
                    }"""
                )
                assert [card["wrap"] for card in wide_facts] == ["nowrap", "nowrap", "wrap"], wide_facts
                assert wide_facts[1]["left"] - wide_facts[0]["left"] >= 30, wide_facts
                assert wide_facts[2]["left"] - wide_facts[1]["left"] >= 30, wide_facts
                assert all(card["scroll"] <= card["client"] + 1 for card in wide_facts), wide_facts
                for card in wide_facts[:2]:
                    assert min(card["mainBottom"], card["sideBottom"]) \
                        > max(card["mainTop"], card["sideTop"]), wide_facts

                # The 620-700px column (laptop with the project panel open): the root
                # card takes up to 620px there and keeps its single-row header.
                wide.set_viewport_size({"width": 1004, "height": 750})
                wide.wait_for_timeout(250)
                owner_facts = wide.evaluate(
                    """() => {
                        const card = document.querySelector('#page-chat .chat-live-card[data-task-id="layout-root"]');
                        const summary = card.querySelector(':scope > .chat-live-summary-button .chat-live-summary');
                        return {column: document.querySelector('#page-chat #chat-messages').clientWidth,
                            width: card.getBoundingClientRect().width,
                            wrap: getComputedStyle(summary).flexWrap};
                    }"""
                )
                assert 700 <= owner_facts["column"] <= 740, owner_facts
                # 80% of a 700-740px column is below the 620px floor, so the floor wins.
                assert abs(owner_facts["width"] - 620) <= 1 and owner_facts["wrap"] == "nowrap", owner_facts
                # The width is monotonic across the 620px chatcol breakpoint: against
                # its containing block's content width the card is
                # min(content, max(80% of content, 620px)) at every column width.
                width_formula = (
                    """() => {
                        const card = document.querySelector('#page-chat .chat-live-card[data-task-id="layout-root"]');
                        const block = card.parentElement;
                        const style = getComputedStyle(block);
                        const content = block.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight);
                        return {content, width: card.getBoundingClientRect().width};
                    }"""
                )
                for viewport_width in (880, 920, 1100):
                    wide.set_viewport_size({"width": viewport_width, "height": 750})
                    wide.wait_for_timeout(250)
                    sample = wide.evaluate(width_formula)
                    expected = min(sample["content"], max(0.8 * sample["content"], 620))
                    assert abs(sample["width"] - expected) <= 1, (viewport_width, sample, expected)
                wide.set_viewport_size({"width": 1100, "height": 750})
                wide.wait_for_timeout(250)

                assert_jump_geometry(wide, "#page-chat")
                jump = wide.locator("#page-chat .chat-scroll-bottom-btn")
                before_hover = jump.bounding_box()
                jump.hover()
                after_hover = jump.bounding_box()
                assert before_hover is not None and after_hover is not None
                for key in ("x", "y", "width", "height"):
                    assert abs(before_hover[key] - after_hover[key]) <= 1, (before_hover, after_hover)
                jump.focus()
                focus = jump.evaluate(
                    "node => ({style: getComputedStyle(node).outlineStyle, "
                    "width: parseFloat(getComputedStyle(node).outlineWidth)})"
                )
                assert focus["style"] != "none" and focus["width"] >= 2, focus
                wide.emulate_media(reduced_motion="reduce")
                assert jump.evaluate(
                    "node => getComputedStyle(node).transitionDuration.split(',')"
                    ".every(value => parseFloat(value) === 0)"
                )
                wide.emulate_media(reduced_motion="no-preference")

                wide.evaluate(
                    """() => {
                        const transfer = new DataTransfer();
                        transfer.items.add(new File(['preview'], 'preview.txt', {type: 'text/plain'}));
                        const target = document.querySelector('#page-chat');
                        for (const type of ['dragenter', 'dragover', 'drop']) {
                            target.dispatchEvent(new DragEvent(type, {
                                bubbles: true, cancelable: true, dataTransfer: transfer,
                            }));
                        }
                    }"""
                )
                wide.locator("#chat-attachment-preview.visible").wait_for(
                    state="visible", timeout=5_000
                )
                assert_jump_geometry(wide, "#page-chat")

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
                            titleTop: title.top,
                            mainBottom: main.bottom,
                            sideTop: side.top,
                            sideBottom: side.bottom,
                        };
                    }"""
                )
                assert panel_facts["panelWidth"] <= 560, panel_facts
                assert panel_facts["cardWidth"] >= panel_facts["panelWidth"] * 0.9, panel_facts
                assert panel_facts["cardScroll"] <= panel_facts["cardClient"] + 1, panel_facts
                assert panel_facts["titleWidth"] >= 180, panel_facts
                assert panel_facts["sideTop"] < panel_facts["mainBottom"], panel_facts
                assert panel_facts["titleTop"] >= max(panel_facts["mainBottom"], panel_facts["sideBottom"]) - 1, panel_facts
                assert_jump_geometry(
                    wide, "#panel-pchat-layout-project", require_overflow=False
                )
                wide.screenshot(
                    path=str(data_dir.parent / f"live-card-project-panel-{browser_engine}.png"),
                    full_page=True,
                )
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.fail(f"required Playwright {browser_engine} browser is not installed: {exc}")
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
                assert att_card.locator(".skills-badge").filter(has_text="owner-attested").count() >= 1
                # The backend is the sole publication classifier. Owner-attested review is
                # not publication-ready, but the selected flow may start an ordinary task
                # that repairs/reviews the bytes before opening a PR.
                submit_html = page.evaluate(
                    """async () => {
                        const m = await import('/static/modules/skill_card_renderer.js');
                        return m.renderInstalledSkillCard(
                            { name: 'att', type: 'instruction', version: '0.1.0', source: 'external',
                              is_self_authored: true, review_status: 'clean',
                              review_gate: { executable_review: true }, review_stale: false,
                              review_profile: 'owner_attested', grants: {}, permissions: [],
                              payload_root: 'skills/external/att', enabled: true,
                              submit_hub: { visible: true, publication_ready: false,
                                task_start_allowed: true, state: 'needs_attention',
                                reason: 'Owner-attested review needs attention' } },
                            new Set(), new Set(), {}, { githubTokenConfigured: true });
                    }"""
                )
                assert 'data-submit-disabled="false"' in submit_html
                assert 'data-publication-ready="false"' in submit_html
                assert 'data-submit-state="needs_attention"' in submit_html
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
    not verification on its own (docs/DEVELOPMENT.md "Responsive and accessible
    behavior").
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
                    # D-10: subagent depth bounds the AGENTS, so it moved out of
                    # Advanced -> Runtime Limits into Agents -> Delegation.
                    page.click('[data-settings-tab="agents"]')
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
                assert depth.input_value() == "3"  # unset -> visible fallback
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

                # The round trip is the point: a reload must not rewrite 0 back to 3.
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

                # Neighbouring state: empty is not a value — it falls back to 3 rather than
                # persisting an unparsable setting.
                type_depth("")
                page.screenshot(path=str(evidence_dir / "v679-depth-06-empty-typed.png"))
                save_and_wait()
                assert saved_depth() == 3
                open_settings_advanced()
                assert page.locator("#s-subagent-depth").input_value() == "3"
                page.screenshot(path=str(evidence_dir / "v679-depth-07-reload-after-empty.png"))
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise
@pytest.mark.ui_browser
def test_ui_owner_context_mode_and_scope_review_ack(direct_server_with_data):
    """Owner context intent and scope-review ack, driven in a real browser.

    Two claimed-complete owner flows that source-string tests cannot certify:

    1. OWNER MAX. Switching an explicit Low to Max succeeds without a Main-route
       context-window confirmation; the frozen compatibility field remains false.
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

    # Boot into explicit owner Low with the one-window false provenance tombstone.
    seeded = json.loads(settings_path.read_text(encoding="utf-8"))
    seeded["OUROBOROS_CONTEXT_MODE"] = "low"
    seeded["OUROBOROS_CONTEXT_MODE_AUTO_LOW"] = "false"
    seeded["OUROBOROS_SCOPE_REVIEW_MODELS"] = seeded["OUROBOROS_MODEL"]
    settings_path.write_text(json.dumps(seeded), encoding="utf-8")
    direct_server_with_data["restart_server"]()

    with urllib.request.urlopen(f"{url}/api/state", timeout=5) as resp:  # noqa: S310 - local test server
        boot_state = json.loads(resp.read().decode("utf-8"))
    assert boot_state["context_mode"] == "low"
    assert boot_state["context_mode_auto_low"] is False

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
                page.wait_for_function("() => document.querySelector('#chat-context-mode')?.dataset.contextMode === 'low'", timeout=30_000)
                assert toggle.get_attribute("data-context-mode") == "low"
                page.screenshot(path=str(evidence_dir / "context-mode-low-before.png"))

                toggle.locator('.chat-seg[data-mode="max"]').click()
                page.wait_for_function(
                    "() => document.querySelector('#chat-context-mode')?.dataset.contextMode === 'max'",
                    timeout=30_000,
                )
                assert page.locator(".confirm-dialog:not([hidden])").count() == 0
                page.screenshot(path=str(evidence_dir / "context-mode-max-after.png"))
                persisted = json.loads(settings_path.read_text(encoding="utf-8"))
                assert persisted["OUROBOROS_CONTEXT_MODE"] == "max"
                assert persisted["OUROBOROS_CONTEXT_MODE_AUTO_LOW"] == "false"
                with urllib.request.urlopen(f"{url}/api/state", timeout=5) as resp:  # noqa: S310
                    after = json.loads(resp.read().decode("utf-8"))
                assert after["context_mode"] == "max"
                assert after["context_mode_auto_low"] is False

                # 2. Scope-review capability notice -> owner confirm -> route-scoped ack.
                # 6.2: the scope route is a review-lane row — pick the API-model
                # route in the grouped combobox and type the id. D-10 moved the
                # lanes out of Models into their own Agents tab.
                page.click('[data-nav-page="settings"]')
                page.wait_for_selector("#s-context-mode", state="attached", timeout=30_000)
                page.locator('[data-settings-tab="agents"]').click()
                page.wait_for_selector("#reviewer-slots-section", timeout=30_000)
                scope_route = page.locator(
                    '#reviewer-scope-rows .reviewer-slot-row [data-slot-route]'
                ).first
                scope_route.wait_for(state="visible", timeout=30_000)
                scope_route.select_option("api")
                custom_input = page.locator(
                    '#reviewer-scope-rows .reviewer-slot-row [data-slot-custom-api]'
                ).first
                custom_input.wait_for(state="visible", timeout=30_000)
                custom_input.fill("openai-compatible::scope-reviewer-x")
                page.locator("#btn-save-settings").click()
                # The capability ack is an in-app dialog since the native-dialog
                # class ban (tests/test_web_dialogs_static.py); Playwright's
                # page.on("dialog") hook only fires for window.alert/confirm/prompt.
                ack_dialog = page.locator(".confirm-dialog")
                ack_dialog.wait_for(state="visible", timeout=60_000)
                ack_text = ack_dialog.inner_text()
                page.screenshot(path=str(evidence_dir / "v6800-scope-review-ack.png"), full_page=True)
                ack_dialog.locator("[data-confirm-ok]").last.click()
                page.wait_for_function(
                    "() => (document.querySelector('#settings-status')?.textContent || '')"
                    ".includes('scope-review route')",
                    timeout=60_000,
                )

                assert "1,000,000-token context window" in ack_text
                assert "openai-compatible::scope-reviewer-x" in ack_text, "the ack must name the exact route"
                status_text = page.locator("#settings-status").inner_text()
                assert "Confirmed the required context window for 1 scope-review route(s)." in status_text
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
def test_ui_smoke_superseded_input_dialog_resolves_object_result(direct_server_with_data):
    """v6.90.3 dialog contract: superseding an INPUT dialog with a newer dialog
    resolves the documented {confirmed: false, value: ''} — never a bare false
    the docs do not promise (the supersession close is mode-aware)."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            try:
                page = browser.new_page()
                page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                first_result = page.evaluate(
                    """
                    async () => {
                        const m = await import('/static/modules/confirm_dialog.js');
                        const first = m.openConfirmDialog({
                            title: 'first', body: 'input dialog', input: true,
                        });
                        const second = m.openConfirmDialog({
                            title: 'second', body: 'supersedes the first',
                        });
                        const r1 = await first;
                        document.querySelector('[data-confirm-cancel]')?.click();
                        await second;
                        return r1;
                    }
                    """
                )
                assert first_result == {"confirmed": False, "value": ""}
                assert page.locator(".confirm-dialog").count() == 0
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise


@pytest.mark.ui_browser
def test_ui_smoke_login_recovery_reconcile_detach_and_retry_are_explicit(direct_server_with_data):
    """Recovery lifecycle."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            try:
                page = browser.new_page()
                posts: list[str] = []
                deletes: list[str] = []
                reconciles: list[str] = []

                def handle_create(route):
                    posts.append(route.request.url)
                    if len(posts) == 1:
                        job = '{"state": "running"}'
                    elif len(posts) == 2:
                        job = ('{"state": "failed", "outcome": '
                               '{"reason": "termination_unconfirmed"}}')
                    else:
                        job = '{"state": "running"}'
                    route.fulfill(
                        status=200,
                        content_type="application/json",
                        body='{"job_id":"job-recovery","job":' + job + '}',
                    )

                def handle_job(route):
                    if route.request.url.endswith("/reconcile"):
                        reconciles.append(route.request.url)
                        if len(reconciles) == 1:
                            route.fulfill(status=409, content_type="application/json", body=(
                                '{"error":"still present","code":"setup_termination_unconfirmed",'
                                '"required_actions":["retry_setup_reconciliation"]}'
                            ))
                        else:
                            route.fulfill(status=200, content_type="application/json", body=(
                                '{"job":{"state":"failed","outcome":'
                                '{"reason":"termination_unconfirmed"},'
                                '"terminationReconciliation":{"status":"empty"}}}'
                            ))
                    else:
                        deletes.append(route.request.url)
                        route.fulfill(status=200, content_type="application/json", body=(
                            '{"job":{"state":"failed","outcome":'
                            '{"reason":"termination_unconfirmed"}}}'
                        ))

                page.route("**/api/claudexor/login", handle_create)
                page.route("**/api/claudexor/login/*", handle_job)
                page.route("**/api/claudexor/login/*/reconcile", handle_job)
                page.goto(url, wait_until="domcontentloaded", timeout=30_000)

                setup_result = page.evaluate(
                    """
                    async () => {
                        const loginHost = () =>
                            document.querySelector('[data-family-login="codex"]')
                            || document.getElementById('harness-login-card');
                        if (!loginHost()) return 'NO-HOST';
                        const m = await import('/static/modules/harness_accounts.js');
                        const wait = async (sel) => {
                            for (let i = 0; i < 100; i++) {
                                const b = loginHost()?.querySelector(sel);
                                if (b && !b.disabled) return b;
                                await new Promise((r) => setTimeout(r, 20));
                            }
                        };
                        const p1 = m.startLogin('codex', 'race-a');
                        const p2 = m.startLogin('codex', 'race-a');
                        await Promise.all([p1, p2]);
                        loginHost()?.querySelector('[data-login-dismiss]')?.click();
                        (await wait('[data-login-reconcile]'))?.click();
                        return 'RECONCILE-CLICKED';
                    }
                    """
                )
                assert setup_result == "RECONCILE-CLICKED"
                # Deterministic settle wait (a fixed sleep was flaky under load:
                # the card could still say "Checking…"). The reconcile round-trip
                # is settled only when the card re-renders the retained-custody
                # recovery face: the outcome detail note exists (it is absent
                # before the click) and "Check again" is enabled again.
                page.wait_for_function(
                    "() => { const host = document.querySelector('[data-family-login=\"codex\"]')"
                    " || document.getElementById('harness-login-card');"
                    " const btn = host?.querySelector('[data-login-reconcile]');"
                    " return Boolean(host?.querySelector('[data-login-detail]'))"
                    " && Boolean(btn) && !btn.disabled; }",
                    timeout=30_000,
                )
                recovery_html = page.evaluate(
                    "() => (document.querySelector('[data-family-login=\"codex\"]')"
                    " || document.getElementById('harness-login-card')).innerHTML"
                )
                assert len(posts) == 1
                assert len(deletes) == 1
                assert len(reconciles) == 1
                assert "Check again" in recovery_html and "job-recovery" in reconciles[0]

                before_detach = (len(posts), len(deletes), len(reconciles))
                detached_html = page.evaluate(
                    """async () => {
                        const loginHost = () =>
                            document.querySelector('[data-family-login="codex"]')
                            || document.getElementById('harness-login-card');
                        loginHost()?.querySelector('[data-login-dismiss]')?.click();
                        await new Promise((r) => setTimeout(r, 50));
                        return loginHost()?.innerHTML || '';
                    }"""
                )
                assert detached_html == ""
                assert (len(posts), len(deletes), len(reconciles)) == before_detach

                final_html = page.evaluate(
                    """async () => {
                        const loginHost = () =>
                            document.querySelector('[data-family-login="codex"]')
                            || document.getElementById('harness-login-card');
                        const m = await import('/static/modules/harness_accounts.js');
                        await m.startLogin('codex', 'race-a');
                        loginHost()?.querySelector('[data-login-reconcile]')?.click();
                        for (let i = 0; i < 100
                            && !loginHost()?.querySelector('[data-login-retry]'); i++)
                            await new Promise((r) => setTimeout(r, 20));
                        loginHost()?.querySelector('[data-login-retry]')?.click();
                        await new Promise((r) => setTimeout(r, 100));
                        return loginHost()?.innerHTML || '';
                    }"""
                )
                assert len(posts) == 3
                assert len(deletes) == 1
                assert len(reconciles) == 2 and all("job-recovery" in u for u in reconciles)
                assert "Starting" in final_html or "sign-in" in final_html
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise


@pytest.mark.ui_browser
def test_ui_smoke_dismiss_overlapping_start_cannot_drop_a_live_job(direct_server_with_data):
    """Queued start follows slow Dismiss."""
    import time as _time

    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            try:
                page = browser.new_page()
                events: list = []

                def handle_create(route):
                    events.append("post")
                    route.fulfill(
                        status=200,
                        content_type="application/json",
                        body='{"job_id": "job-ov-%d", "job": {"state": "running"},'
                             ' "attach_command": ""}' % len(events),
                    )

                def handle_cancel(route):
                    events.append("delete-open")
                    _time.sleep(0.35)
                    events.append("delete-done")
                    route.fulfill(status=200, content_type="application/json",
                                  body='{"job":{"state":"cancelled"}}')

                page.route("**/api/claudexor/login", handle_create)
                page.route("**/api/claudexor/login/*", handle_cancel)
                page.goto(url, wait_until="domcontentloaded", timeout=30_000)

                result = page.evaluate(
                    """
                    async () => {
                        const loginHost = () =>
                            document.querySelector('[data-family-login="codex"]')
                            || document.getElementById('harness-login-card');
                        if (!loginHost()) return { error: 'NO-HOST' };
                        const m = await import('/static/modules/harness_accounts.js');
                        await m.startLogin('codex', 'ov-a');
                        loginHost()?.querySelector('[data-login-dismiss]')?.click();
                        await m.startLogin('codex', 'ov-b');
                        await new Promise((r) => setTimeout(r, 600));
                        const cardAfterQueuedStart = (loginHost()?.innerHTML || '').length > 0;
                        await m.startLogin('codex', 'ov-c');
                        return {
                            cardAfterQueuedStart,
                            finalHasCard: (loginHost()?.innerHTML || '').length > 0,
                        };
                    }
                    """
                )
                assert result.get("error") is None
                assert result["cardAfterQueuedStart"] is True
                assert result["finalHasCard"] is True
                posts = events.count("post")
                deletes = events.count("delete-open")
                assert posts == 3
                assert deletes == 2
                first_delete_done = events.index("delete-done")
                second_post = [i for i, e in enumerate(events) if e == "post"][1]
                assert first_delete_done < second_post
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise


@pytest.mark.ui_browser
@pytest.mark.parametrize("face", ["recovery", "reconciled", "unavailable"])
def test_ui_smoke_stale_get_cannot_overwrite_login_terminal_faces(
    direct_server_with_data, face,
):
    """Stale GET cannot repaint custody."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            try:
                page = browser.new_page()
                creates: list[str] = []
                deletes: list[str] = []
                reconciles: list[str] = []

                def handle_create(route):
                    creates.append(route.request.url)
                    route.fulfill(
                        status=200,
                        content_type="application/json",
                        body='{"job_id": "job-stale", "job": {"state": "running"},'
                             ' "attach_command": ""}',
                    )

                def handle_job(route):
                    if route.request.method == "DELETE":
                        deletes.append(route.request.url)
                        if face == "unavailable":
                            route.fulfill(status=404, content_type="application/json", body="{}")
                        else:
                            route.fulfill(status=200, content_type="application/json", body=(
                                '{"job":{"state":"failed","outcome":'
                                '{"reason":"termination_unconfirmed"}}}'
                            ))
                        return
                    route.fulfill(status=200, content_type="application/json",
                                  body='{"job": {"state": "running"}}')

                def handle_reconcile(route):
                    reconciles.append(route.request.url)
                    route.fulfill(status=200, content_type="application/json", body=(
                        '{"job":{"state":"failed","outcome":'
                        '{"reason":"termination_unconfirmed"},'
                        '"terminationReconciliation":{"status":"empty"}}}'
                    ))

                page.route("**/api/claudexor/login", handle_create)
                page.route("**/api/claudexor/login/*", handle_job)
                page.route("**/api/claudexor/login/*/reconcile", handle_reconcile)
                page.goto(url, wait_until="domcontentloaded", timeout=30_000)

                result = page.evaluate(
                    """
                    async (face) => {
                        const realFetch = window.fetch.bind(window);
                        let releaseStale;
                        const stale = new Promise((resolve) => { releaseStale = resolve; });
                        let gets = 0;
                        window.fetch = (input, init = {}) => {
                            const url = String(input?.url || input);
                            const method = String(init.method || input?.method || 'GET').toUpperCase();
                            if (method === 'GET' && url.includes('/api/claudexor/login/job-stale')) {
                                gets += 1;
                                return stale;
                            }
                            return realFetch(input, init);
                        };
                        const loginHost = () =>
                            document.querySelector('[data-family-login="codex"]')
                            || document.getElementById('harness-login-card');
                        const m = await import('/static/modules/harness_accounts.js');
                        await m.startLogin('codex', 'stale-' + face);
                        await new Promise((r) => setTimeout(r, 3200));
                        loginHost()?.querySelector('[data-login-dismiss]')?.click();
                        await new Promise((r) => setTimeout(r, 100));
                        if (face === 'reconciled') {
                            loginHost()?.querySelector('[data-login-reconcile]')?.click();
                            await new Promise((r) => setTimeout(r, 100));
                        }
                        const before = loginHost()?.innerHTML || '';
                        releaseStale(new Response('{"job":{"state":"running"}}', {
                            status: 200, headers: { 'Content-Type': 'application/json' },
                        }));
                        await new Promise((r) => setTimeout(r, 3400));
                        return { before, after: loginHost()?.innerHTML || '', gets };
                    }
                    """, face,
                )
                assert result["gets"] == 1
                assert result["before"] == result["after"]
                marker = {
                    "recovery": "could not prove",
                    "reconciled": "no longer blocking",
                    "unavailable": "no longer available",
                }[face]
                assert marker in result["after"]
                assert len(creates) == 1 and len(deletes) == 1
                assert len(reconciles) == (1 if face == "reconciled" else 0)
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise


@pytest.mark.ui_browser
def test_ui_smoke_window_pagehide_detaches_login_without_lifecycle_http(direct_server_with_data):
    """Window pagehide detaches locally."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            try:
                page = browser.new_page()
                page.goto(direct_server_with_data["url"], wait_until="domcontentloaded")
                result = page.evaluate(
                    """
                    async () => {
                        const {createAgentsStep} = await import('/static/modules/onboarding_agents_step.js');
                        let connect, release;
                        const pending = new Promise((r) => { release = r; });
                        const calls = {create: 0, delete: 0, reconcile: 0, get: 0};
                        const button = {getAttribute: () => 'claude',
                            addEventListener: (_t, fn) => { connect = fn; }};
                        const host = {innerHTML: '', querySelector: () => null};
                        const list = {innerHTML: '', querySelectorAll: () => [button]};
                        const other = document.createElement('div');
                        const doc = {defaultView: window, getElementById: (id) =>
                            id === 'agents-login-host' ? host
                                : id === 'agents-family-list' ? list : other};
                        const store = {
                            accountsKnown: false, snapshot: null, subscribe: () => () => {},
                            refresh: () => {}, unavailableNote: () => null,
                        };
                        const fetchImpl = async (input, init={}) => {
                            const url=String(input), method=init.method || 'GET';
                            if (url === '/api/claudexor/login' && method === 'POST') {
                                calls.create++; return pending;
                            }
                            if (url.endsWith('/reconcile')) calls.reconcile++;
                            else if (method === 'DELETE') calls.delete++; else calls.get++;
                            return new Response('{"job":{"state":"running"}}', {
                                status: 200, headers: {'Content-Type':'application/json'}});
                        };
                        const step = createAgentsStep({doc, store, fetchImpl});
                        step.mount(); connect(); await Promise.resolve();
                        window.dispatchEvent(new PageTransitionEvent('pagehide',{persisted:true}));
                        const cached=host.innerHTML, before={...calls};
                        window.dispatchEvent(new PageTransitionEvent('pagehide',{persisted:false}));
                        const immediate=host.innerHTML;
                        connect(); await Promise.resolve();
                        release(new Response('{"job_id":"late","job":{"state":"running"}}',
                            {status:200,headers:{'Content-Type':'application/json'}}));
                        await new Promise((r) => setTimeout(r, 50));
                        return {cached, immediate, final:host.innerHTML, before, after:calls};
                    }
                    """
                )
                assert result["cached"]
                assert result["immediate"] == result["final"] == ""
                assert result["before"] == result["after"] == dict(
                    create=1, delete=0, reconcile=0, get=0)
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise


@pytest.mark.ui_browser
def test_ui_smoke_dismiss_overlapping_settle_never_freezes_the_card(direct_server_with_data):
    """Terminal GET wins over slow Dismiss."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            try:
                page = browser.new_page()
                creates: list[str] = []
                gets: list[str] = []
                reconciles: list[str] = []

                def handle_create(route):
                    creates.append(route.request.url)
                    route.fulfill(status=200, content_type="application/json",
                                  body='{"job_id": "job-os-1", "job": {"state": "running"},'
                                       ' "attach_command": ""}')

                def handle_job(route):
                    gets.append(route.request.url)
                    route.fulfill(status=200, content_type="application/json",
                                  body='{"job": {"state": "succeeded"}}')

                def handle_reconcile(route):
                    reconciles.append(route.request.url)
                    route.fulfill(status=500, content_type="application/json", body="{}")

                page.route("**/api/claudexor/login", handle_create)
                page.route("**/api/claudexor/login/*", handle_job)
                page.route("**/api/claudexor/login/*/reconcile", handle_reconcile)
                page.goto(url, wait_until="domcontentloaded", timeout=30_000)

                result = page.evaluate(
                    """
                    async () => {
                        const realFetch = window.fetch.bind(window);
                        let deletes = 0;
                        window.fetch = (input, init = {}) => {
                            const url = String(input && input.url ? input.url : input);
                            const method = String((init && init.method)
                                || (input && input.method) || 'GET').toUpperCase();
                            if (method === 'DELETE' && url.includes('/api/claudexor/login/')) {
                                deletes += 1;
                                return new Promise((resolve) => setTimeout(() => resolve(
                                    new Response('{"error": "daemon busy"}',
                                        { status: 503,
                                          headers: { 'Content-Type': 'application/json' } })
                                ), 4000));
                            }
                            return realFetch(input, init);
                        };
                        const loginHost = () =>
                            document.querySelector('[data-family-login="codex"]')
                            || document.getElementById('harness-login-card');
                        const m = await import('/static/modules/harness_accounts.js');
                        await m.startLogin('codex', 'os-a');
                        loginHost()?.querySelector('[data-login-dismiss]')?.click();
                        await new Promise((r) => setTimeout(r, 5200));
                        const host = loginHost();
                        return { html: host?.innerHTML || '', deletes,
                            cardCount: host?.querySelectorAll('[data-login-card]').length || 0,
                            verdict: host?.querySelector('[data-login-verdict]')?.textContent.trim() };
                    }
                    """
                )
                assert result["cardCount"] == 1
                assert result["verdict"] == "Connected."
                assert "Could not cancel" not in result["html"]
                assert (len(creates), result["deletes"], len(gets), len(reconciles)) == (1, 1, 1, 0)
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise


@pytest.mark.ui_browser
def test_ui_smoke_cancel_run_button_eligibility_and_cancelled_state(direct_server_with_data):
    """v6.82 P5 / S3 Q2: the stop control renders ONLY on live marker-attested
    root cards (never marker-less direct-turn cards, subagent children, or the
    reusable background slot), opens the dropdown, and a cancelled root
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
        "_schema_version": 1,
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
                assert cancel_btn.inner_text().strip() == "Stop…"
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
                # Dropdown wiring (S3 Q2): open, then dismiss = keep running.
                cancel_btn.click()
                menu = page.locator('body > .task-control-menu')
                menu.wait_for(state="visible", timeout=10_000)
                assert "Wrap up" in menu.inner_text()
                page.keyboard.press("Escape")
                menu.wait_for(state="detached", timeout=10_000)
                assert cancel_btn.is_enabled()
                page.screenshot(path=str(data_dir.parent / "cancel-run.png"), full_page=True)
            finally:
                browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise


# The in-flight indicator lifecycle smoke test lives in
# tests/test_ui_smoke_inflight_indicator.py and the Settings → Agents list-editor
# acceptance in tests/test_ui_smoke_agents_panel.py (size-ratchet byte gate on this module).
