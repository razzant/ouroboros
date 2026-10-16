"""Task-owned Chrome MCP controls through the real Settings save/reload consumer.

Uses the existing isolated direct-server fixture; the disabled synthetic stdio
entry is never launched or paired with a browser. Screenshot evidence is local.
"""
from __future__ import annotations

import json

import pytest

from tests.test_mcp_headers_browser import _capture, _open_mcp, _save
from tests.test_ui_smoke_playwright import direct_server_with_data  # noqa: F401

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]


@pytest.mark.parametrize("engine,width", [("chromium", 1440), ("webkit", 390)])
def test_chrome_bridge_settings_save_reload(direct_server_with_data, engine, width):  # noqa: F811
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    data_dir = direct_server_with_data["data_dir"]
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch(headless=True)
        page = browser.new_page(viewport={"width": width, "height": 1000})
        page._evidence_engine = engine
        try:
            _open_mcp(page, url, open_drawer=width == 390)
            page.click("#btn-mcp-add-server")
            card = page.locator("[data-mcp-card]").last
            card.locator('[data-mcp-field="id"]').fill("chrome_fixture")
            card.locator('[data-mcp-field="transport"]').select_option("stdio")
            card = page.locator("[data-mcp-card]").last
            card.locator('[data-mcp-field="command"]').fill("node")
            bridge = card.locator('[data-mcp-field="browser_bridge"]')
            protocol = card.locator('[data-mcp-field="browser_protocol"]')
            bridge.check()
            protocol.select_option("ouroboros_chrome")
            protocol.click()  # Locator action re-resolves a Settings status redraw.
            assert bridge.is_checked()
            assert protocol.input_value() == "ouroboros_chrome"
            assert not card.locator('[data-mcp-field="enabled"]').is_checked()
            _capture(page, "chrome-bridge-controls")
            _save(page)
            saved = json.loads((data_dir / "settings.json").read_text(encoding="utf-8"))
            entry = next(row for row in saved["MCP_SERVERS"] if row["id"] == "chrome_fixture")
            assert entry["transport"] == "stdio" and entry["command"] == "node"
            assert entry["browser_bridge"] is True
            assert entry["browser_protocol"] == "ouroboros_chrome"
            assert entry["enabled"] is False
            page.reload(wait_until="domcontentloaded")
            _open_mcp(page, url, open_drawer=width == 390)
            card = page.locator("[data-mcp-card]").last
            protocol = card.locator('[data-mcp-field="browser_protocol"]')
            protocol.click()  # Locator action re-resolves a Settings status redraw.
            assert card.locator('[data-mcp-field="browser_bridge"]').is_checked()
            assert protocol.input_value() == "ouroboros_chrome"
            assert not card.locator('[data-mcp-field="enabled"]').is_checked()
            _capture(page, "chrome-bridge-reloaded")
        finally:
            browser.close()
