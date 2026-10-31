"""Dashboard › Consciousness names the busy store beside the wake rows it already read.

The page is the shipped SPA (``web/``). ``/api/consciousness`` and ``/api/places`` are
answered by the real gateway over a seeded store; the busy answer is read while another
connection holds the dialogue store's exclusive lock. Screens (light/dark, 1100/390,
per engine) go to ``OUROBOROS_UI_EVIDENCE_DIR`` before anything is asserted.
"""
from __future__ import annotations

import contextlib
from urllib.parse import urlparse

import pytest

from ouroboros import dialogue_store
from tests import test_recent_wake_store_busy as store_busy, test_subscription_setup_browser as setup_browser
from tests.test_recent_wake_store_busy import busy, exclusive

subscription_ui, client, wake = setup_browser.subscription_ui, store_busy.client, store_busy.wake
pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]
STATE = {"supervisor_ready": True, "active_chat_activities": [], "projects": [], "runtime_mode": "advanced",
         "bg_consciousness_enabled": True, "bg_consciousness_state": {"status": "sleeping"}}


@pytest.fixture(params=["chromium", "webkit"])
def engine_ui(request, monkeypatch):
    monkeypatch.setenv("OUROBOROS_UI_BROWSER_ENGINE", request.param)
    return {**request.getfixturevalue("subscription_ui"), "engine": request.param}


@pytest.mark.parametrize("scheme", ["light", "dark"])
@pytest.mark.parametrize("width", [1100, 390])
def test_the_busy_sentence_stands_over_the_rows_already_read(engine_ui, client, wake, tmp_path, scheme, width):
    ui, page = engine_ui, engine_ui["page"]
    page.set_viewport_size({"width": width, "height": 844 if width < 500 else 750})
    page.emulate_media(color_scheme=scheme)
    origin = urlparse(ui["url"]).netloc
    page.route("**/*", lambda route: route.fallback() if urlparse(route.request.url).netloc == origin else route.abort())
    phase = {"busy": False}

    def gateway(route):
        path = urlparse(route.request.url).path
        with contextlib.ExitStack() as held:
            if phase["busy"] and path == "/api/consciousness":
                held.enter_context(exclusive(dialogue_store.path(tmp_path)))
            answer = client.get(path)
        route.fulfill(status=answer.status_code, content_type="application/json", body=answer.text)

    page.route("**/api/consciousness", gateway)
    page.route("**/api/places", gateway)
    page.route("**/api/state", lambda route: route.fulfill(json=STATE))
    page.goto(ui["url"] + "/#dashboard")
    page.locator('[data-dashboard-tab="consciousness"]').click()
    region = page.locator("[data-cx-wakes]")
    row = region.locator(f'[data-wake="{wake["task_id"]}"]')
    row.wait_for()
    name = f"recent-wakes-{ui['engine']}-{scheme}-{width}"
    region.scroll_into_view_if_needed()
    setup_browser.capture(page, name + "-read")
    phase["busy"] = True
    page.locator('[data-dashboard-tab="logs"]').click()  # showing the tab again reads the panel again
    page.locator('[data-dashboard-tab="consciousness"]').click()
    warning = region.locator('p[role="status"][data-tone="warn"]')
    try:
        warning.wait_for()
    finally:
        region.scroll_into_view_if_needed()
        setup_browser.capture(page, name + "-busy")
    assert warning.inner_text() == busy("dialogue")
    assert row.is_visible() and region.locator("[data-wake]").count() == 1, "the rows already read stay"
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), "horizontal overflow"
