"""The browser smoke that proves the shipped UI boots and does its first job.

This module owns the direct-mode load of chat and dashboard, the Docker-mode health
check, the projects sidebar and panel lifecycle, and the task a mock provider creates.

The declarative widgets, the live cards, the chat surface, the login lifecycle and the
review controls were split verbatim into ``tests/test_ui_smoke_widgets.py``,
``tests/test_ui_smoke_cards.py``, ``tests/test_ui_smoke_chat.py``,
``tests/test_ui_smoke_login.py`` and ``tests/test_ui_smoke_review_controls.py``; the
server fixtures they share live in ``tests/_ui_smoke_shared.py``.

Every test here launches a real browser and is marked ``ui_browser`` or
``ui_browser_docker``, so the default local run deselects the whole module.
"""

from __future__ import annotations

import os
import subprocess

import pytest


from tests._ui_smoke_shared import (
    _free_port,
    _wait_health,
)
from tests._ui_smoke_shared import direct_server as _direct_server
from tests._ui_smoke_shared import direct_server_with_data as _direct_server_with_data

# Fixtures are requested by name as test parameters, so they are re-bound through a
# module attribute: a direct import of a name that reappears as a parameter is an F811
# redefinition under the CI ruff gate.
direct_server = _direct_server
direct_server_with_data = _direct_server_with_data


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

@pytest.mark.ui_browser
def test_ui_smoke_direct_mode_loads_chat_and_dashboard(direct_server):
    _run_core_ui_assertions(direct_server)

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
