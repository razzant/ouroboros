"""The shared reader of Main's tab-kept slot (tests/_room_browser_shared.py MAIN_SLOT_JS) takes exactly Main's key.

A reader that took the first key with the right prefix would be satisfied by a legacy key, another profile's Main
slot or this profile's Project-room slot, and would reject the right writer when stale keys came first.
"""
from __future__ import annotations

import json

import pytest

from tests._room_browser_shared import MAIN_SLOT_JS

pytestmark = [pytest.mark.ui_browser]

SCOPE, OTHER = "0123456789abcdef", "fedcba9876543210"
BASE = "ouro_chat_reply"
PAGE = '<!doctype html><meta name="ouroboros-chat-preview-scope" content="%s"><title>t</title>'
BARE = "<!doctype html><title>t</title>"


def _read(browser, html, seeded):
    page = browser.new_page()
    page.route("http://slot.test/**", lambda route: route.fulfill(status=200, content_type="text/html", body=html))
    page.goto("http://slot.test/")
    page.evaluate("items => { for (const [k, v] of items) sessionStorage.setItem(k, v); }",
                  [[key, json.dumps(value)] for key, value in seeded.items()])
    try:
        return page.evaluate(MAIN_SLOT_JS, BASE)
    finally:
        page.close()


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_main_slot_reader_takes_only_the_exact_key(engine):
    from playwright.sync_api import sync_playwright

    stale = {BASE: {"who": "legacy"}, f"{BASE}:{OTHER}": {"who": "other profile"},
             f"{BASE}:{SCOPE}:7": {"who": "project room"}}
    right = {f"{BASE}:{SCOPE}": {"who": "main"}}
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch()
        try:
            scoped = PAGE % SCOPE
            assert _read(browser, scoped, {**stale, **right}) == {"who": "main"}
            assert _read(browser, scoped, {**right, **stale}) == {"who": "main"}
            assert _read(browser, scoped, stale) is None, "no wrong slot may stand in for Main's own"
            assert _read(browser, scoped, {}) is None
            assert _read(browser, BARE, {BASE: {"who": "legacy"}, **right}) == {"who": "legacy"}
            assert _read(browser, BARE, right) is None
        finally:
            browser.close()
