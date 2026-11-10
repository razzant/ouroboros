"""The Settings LAN address on the real page, in Chromium and WebKit: one accessible Copy link button beside the link.

The page, its `/api/settings` read and its toast host are the real ones (the subscription suite's fixture answers the
read); the clipboard is an inert recorder (tests/ui_clipboard_sink.py), so nothing here reaches the machine's
clipboard. Addresses come from the documentation range (RFC 5737).
"""
from __future__ import annotations

import os
import re

import pytest

from tests.test_subscription_setup_browser import capture
from tests.test_subscription_setup_browser import subscription_ui as subscription_ui  # the suite's fixture
from tests.ui_clipboard_sink import INSTALL, recorded, set_sink

pytestmark = [pytest.mark.ui_browser]

FIRST, SECOND = "http://192.0.2.44:8765", "http://192.0.2.45:9000"
WARNING = "Set a Network Password before sharing this address."
LAN_FIT = """() => {
    const hint = document.getElementById('settings-lan-hint');
    const button = hint.querySelector('[data-copy-lan-url]');
    const scroller = document.querySelector('#page-settings .settings-scroll');
    const box = (node) => { const r = node.getBoundingClientRect(); return [r.left, r.top, r.right, r.bottom]; };
    return {hint: box(hint), button: button && box(button), width: innerWidth,
            sideways: scroller.scrollWidth - scroller.clientWidth, page: document.scrollingElement.scrollWidth - innerWidth};
}"""


def test_settings_lan_url_has_one_copy_button_that_copies_the_current_address(subscription_ui):
    """The real Settings page, its real `/api/settings` read answered by the fixture: the LAN address keeps its link
    and warning, gains one accessible Copy link button, and that button writes whatever address the latest read drew
    through the shared copy (toast, fallback and refusal included); loopback or a placeholder offers nothing to copy."""
    from playwright.sync_api import expect

    ui, page = subscription_ui, subscription_ui["page"]
    engine = os.environ.get("OUROBOROS_UI_BROWSER_ENGINE", "chromium")
    assert page.context.browser.browser_type.name == engine
    ui["settings"]["_meta"].update(reachability="lan_reachable", recommended_url=FIRST, warning=WARNING)
    page.add_init_script(INSTALL)
    page.goto(ui["url"])
    page.click('[data-nav-page="settings"]')
    hint = page.locator("#settings-lan-hint")
    toasts = page.locator("#toast-stack .toast")
    button = hint.get_by_role("button", name="Copy link")
    expect(button).to_have_count(1, timeout=30_000)

    # The address keeps its own link and warning; the button is one more control beside them.
    link = hint.get_by_role("link", name=FIRST)
    assert [link.get_attribute("href"), link.get_attribute("target"), link.get_attribute("rel")] == [FIRST, "_blank", "noopener"]
    assert WARNING in hint.inner_text() and hint.inner_text().startswith(f"LAN URL: {FIRST}")
    assert hint.locator("button").count() == 1
    assert {"btn", "btn-default", "btn-sm", "ui-hit-area"} <= set(button.get_attribute("class").split())
    assert button.get_attribute("type") == "button"

    def fit(where):
        # A resized window keeps the old layout for a frame in WebKit: measure once the hint has its column.
        hint.evaluate("el => el.scrollIntoView({block: 'center'})")
        page.wait_for_function("() => document.getElementById('settings-lan-hint').getBoundingClientRect().width > 250")
        found = page.evaluate(LAN_FIT)
        left, top, right, bottom = found["hint"]
        b_left, b_top, b_right, b_bottom = found["button"]
        assert left <= b_left and b_right <= right + 0.5 and top <= b_top and b_bottom <= bottom + 0.5, (where, found)
        assert right <= found["width"] and found["sideways"] <= 0 and found["page"] <= 0, (where, found)

    fit("desktop")
    # By key, as a keyboard owner would: the shared copy writes the address and reports through the toast host.
    button.focus()
    button.press("Enter")
    expect(toasts).to_have_text(["LAN URL copied."])  # one report, from the shared host, and no second voice
    assert recorded(page)["writes"] == [FIRST] and recorded(page)["selections"] == []
    expect(button).to_be_focused()
    fit("desktop")
    capture(page, f"settings-lan-copy-{engine}-desktop")

    # A later read (the bind changed) draws one button again, and it carries that read's address.
    ui["settings"]["_meta"].update(recommended_url=SECOND)
    page.click("#btn-reload-settings")
    expect(hint.get_by_role("link", name=SECOND)).to_have_attribute("href", SECOND)
    expect(hint.get_by_role("button", name="Copy link")).to_have_count(1)
    hint.get_by_role("button", name="Copy link").click()
    expect(toasts.last).to_have_text("LAN URL copied.")
    assert recorded(page)["writes"] == [FIRST, SECOND]

    # Only a real LAN address can be copied.
    for meta in ({"reachability": "loopback_only", "recommended_url": ""},
                 {"reachability": "host_ip_unknown", "recommended_url": "http://your-host-ip:8765"}):
        ui["settings"]["_meta"].update(meta)
        page.click("#btn-reload-settings")
        expect(hint.locator("[data-copy-lan-url]")).to_have_count(0)
        expect(hint).to_be_visible()
    ui["settings"]["_meta"].update(reachability="lan_reachable", recommended_url=SECOND)
    page.click("#btn-reload-settings")
    expect(hint.get_by_role("button", name="Copy link")).to_have_count(1)

    # No async clipboard (a non-secure LAN page): the selection path copies exactly the address.
    set_sink(page, "missing", True, reset=True)
    hint.get_by_role("button", name="Copy link").click()
    expect(toasts.last).to_have_text("LAN URL copied.")
    assert recorded(page)["selections"] == [{"command": "copy", "tag": "TEXTAREA", "text": SECOND, "readOnly": True, "inDialog": False}]
    # Refused by the API and by the engine: the failure is said, and no success follows it.
    set_sink(page, "refused", False, reset=True)
    hint.get_by_role("button", name="Copy link").click()
    expect(toasts.last).to_have_text("Could not copy: Clipboard access is unavailable")
    assert recorded(page)["writes"] == []
    assert page.evaluate("() => document.querySelectorAll('.chat-copy-fallback').length") == 0

    # A phone-wide Settings page: the action stays inside the hint and the page does not scroll sideways.
    page.set_viewport_size({"width": 390, "height": 844})
    page.evaluate("() => document.querySelectorAll('#toast-stack .toast').forEach((toast) => toast.click())")
    fit("phone")
    capture(page, f"settings-lan-copy-{engine}-phone")
    page.set_viewport_size({"width": 1360, "height": 900})

    # One listener lives as long as the page: a page kept for back/forward still copies, a gone one does not.
    set_sink(page, "ok", True, reset=True)
    page.evaluate("() => window.dispatchEvent(new PageTransitionEvent('pagehide', {persisted: true}))")
    hint.get_by_role("button", name="Copy link").click()
    expect(toasts.last).to_have_text("LAN URL copied.")
    assert recorded(page)["writes"] == [SECOND]
    page.evaluate("() => window.dispatchEvent(new PageTransitionEvent('pagehide', {persisted: false}))")
    hint.get_by_role("button", name="Copy link").click()
    page.wait_for_timeout(300)
    assert recorded(page)["writes"] == [SECOND], "a disposed page's button copies nothing"
    assert toasts.count() == 0


def test_lan_result_reflows_with_its_container_and_keeps_the_full_address(subscription_ui):
    from playwright.sync_api import expect

    ui, page = subscription_ui, subscription_ui["page"]
    address = FIRST + "/" + "long-deployment-prefix-" * 5
    ui["settings"]["_meta"].update(reachability="lan_reachable", recommended_url=address, warning=WARNING)
    page.add_init_script(INSTALL)
    page.set_viewport_size({"width": 1100, "height": 750})
    page.goto(ui["url"] + "/#settings")
    hint = page.locator("#settings-lan-hint")
    button = hint.get_by_role("button", name="Copy link")
    expect(button).to_be_visible()
    row = button.locator("..")
    row.evaluate("e => { e.style.width = '320px'; e.style.maxWidth = '100%'; }")
    hint.scroll_into_view_if_needed()
    capture(page, "settings-lan-readable-inline")
    # At 320px the short action can share readable text space; at 280px it
    # cannot. Preserve the full-line assertion at the actual wrapping boundary.
    row.evaluate("e => e.style.width = '280px'")
    text_box, action_box, row_box = row.locator(":scope > span").bounding_box(), button.bounding_box(), row.bounding_box()
    assert text_box["width"] >= row_box["width"] - 2
    assert action_box["y"] >= text_box["y"] + text_box["height"] - 1
    assert row.evaluate("e => e.scrollWidth <= e.clientWidth + 1")
    expect(hint.get_by_role("link", name=address)).to_have_attribute("href", address)
    assert WARNING in hint.inner_text()
    button.focus()
    button.press("Enter")
    expect(page.locator("#toast-stack .toast").last).to_have_text("LAN URL copied.")
    assert recorded(page)["writes"] == [address]
    expect(button).to_be_focused()
    capture(page, "settings-lan-container-reflow")


def test_settings_lan_url_in_a_translated_ui_keeps_the_link_and_its_label(subscription_ui):
    """With a non-English dictionary the overlay translates the sentence around the link as one inline composite and
    the button as chrome; the link keeps its own label and address, and the button still copies the read's address,
    never a translated word. A later read whose sentence the dictionary does not know stays as the app wrote it."""
    from playwright.sync_api import expect

    ui, page = subscription_ui, subscription_ui["page"]
    ui["settings"]["_meta"].update(reachability="lan_reachable", recommended_url=FIRST, warning=WARNING)
    dictionary = {"language": "ru", "english": False, "revision": 1, "entries": {
        f"LAN URL: <1>{FIRST}</1> — <2>{WARNING}</2>": {
            "text": f"Адрес в сети: <1>{FIRST}</1> — <2>Задайте сетевой пароль, прежде чем делиться этим адресом.</2>",
            "provenance": "imported"},
        "Copy link": {"text": "Копировать адрес", "provenance": "imported"},
    }}
    page.route("**/api/ui/i18n", lambda route: route.fulfill(json=dictionary))
    page.add_init_script(INSTALL)
    page.goto(ui["url"])
    page.click('[data-nav-page="settings"]')
    hint = page.locator("#settings-lan-hint")
    button = hint.get_by_role("button", name="Копировать адрес")
    expect(button).to_have_count(1, timeout=30_000)
    link = hint.get_by_role("link", name=FIRST)
    expect(link).to_have_count(1)
    assert [link.get_attribute("href"), link.get_attribute("target"), link.get_attribute("rel")] == [FIRST, "_blank", "noopener"]
    # Settings may rebuild the hint after page-show; wait for its next translation pass.
    expect(hint).to_have_text(re.compile(r"^" + re.escape(f"Адрес в сети: {FIRST} — Задайте сетевой пароль")), use_inner_text=True)
    expect(hint.locator("strong")).to_have_text(re.compile(r"^Задайте сетевой пароль"), use_inner_text=True)
    button.click()
    expect(page.locator("#toast-stack .toast").last).to_have_text("LAN URL copied.")
    assert recorded(page)["writes"] == [FIRST], "the address, not a word of the sentence"

    ui["settings"]["_meta"].update(recommended_url=SECOND)
    page.click("#btn-reload-settings")
    expect(hint.get_by_role("link", name=SECOND)).to_have_attribute("href", SECOND)
    assert hint.inner_text().startswith(f"LAN URL: {SECOND} — {WARNING}"), "no entry for this sentence: the app's words stand"
    hint.get_by_role("button", name="Копировать адрес").click()
    assert recorded(page)["writes"] == [FIRST, SECOND]
