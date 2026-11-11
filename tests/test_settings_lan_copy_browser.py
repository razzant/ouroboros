"""The address other devices type (Settings › This computer › Who can reach it) on the real page, in Chromium and
WebKit: a row that shows the address as a link with one accessible Copy link button beside it.

The page, its `/api/settings` read and its toast host are the real ones (the subscription suite's fixture answers the
read); the clipboard is an inert recorder (tests/ui_clipboard_sink.py), so nothing here reaches the machine's
clipboard. Addresses come from the documentation range (RFC 5737).
"""
from __future__ import annotations

import os

import pytest

from tests.settings_browser_helpers import go_back, open_settings
from tests.test_subscription_setup_browser import capture
from tests.test_subscription_setup_browser import subscription_ui as subscription_ui  # the suite's fixture
from tests.ui_clipboard_sink import INSTALL, recorded, set_sink

pytestmark = [pytest.mark.ui_browser]

FIRST, SECOND = "http://192.0.2.44:8765", "http://192.0.2.45:9000"
WARNING = "Set a Network Password before sharing this address."
ROW = '#page-settings [data-setting-row="computer:address"]'
LAN_FIT = """() => {
    const row = document.querySelector('#page-settings [data-setting-row="computer:address"]');
    const button = row.querySelector('[data-copy-lan-url]');
    const level = row.closest('[data-settings-page]');
    const box = (node) => { const r = node.getBoundingClientRect(); return [r.left, r.top, r.right, r.bottom]; };
    return {row: box(row), button: button && box(button), width: innerWidth, inside: row.scrollWidth - row.clientWidth,
            sideways: level.scrollWidth - level.clientWidth, page: document.scrollingElement.scrollWidth - innerWidth};
}"""


def _open(page, ui, **meta):
    ui["settings"]["_meta"].update(meta)
    page.add_init_script(INSTALL)
    page.goto(ui["url"] + "/#settings")
    return _read_again(page)


def _read_again(page):
    """Show This computer: a page reads the settings whenever it is shown, so a later read needs no button."""
    open_settings(page, "computer")
    return page.locator(ROW)


def test_the_address_row_has_one_copy_button_that_copies_the_current_address(subscription_ui):
    """The real page, its real `/api/settings` read answered by the fixture: the address keeps its link, the server's
    warning stands in the row as a warning, one accessible Copy link button writes whatever address the latest read drew
    through the shared copy (toast, fallback and refusal included); loopback offers no row and a placeholder no button."""
    from playwright.sync_api import expect

    ui, page = subscription_ui, subscription_ui["page"]
    engine = os.environ.get("OUROBOROS_UI_BROWSER_ENGINE", "chromium")
    assert page.context.browser.browser_type.name == engine
    row = _open(page, ui, reachability="lan_reachable", recommended_url=FIRST, warning=WARNING)
    toasts = page.locator("#toast-stack .toast")
    button = row.get_by_role("button", name="Copy link")
    expect(button).to_have_count(1, timeout=30_000)

    # The address is a link of its own; the button is the one control beside it; the row's lines say the rest.
    link = row.get_by_role("link", name=FIRST)
    assert [link.get_attribute("href"), link.get_attribute("target"), link.get_attribute("rel")] == [FIRST, "_blank", "noopener"]
    assert row.locator(".ui-row-title").inner_text() == "Address for other devices"
    assert row.locator("[data-setting-meta]").inner_text() == "Type it in a browser on the same network."
    warning = row.locator("[data-setting-result]")
    expect(warning).to_have_text(WARNING)
    assert warning.get_attribute("data-tone") == "warn"
    assert row.locator("button").count() == 1
    assert {"btn", "btn-default", "btn-sm", "ui-hit-area"} <= set(button.get_attribute("class").split())
    assert button.get_attribute("type") == "button"

    def fit(where):
        row.evaluate("el => el.scrollIntoView({block: 'center'})")
        found = page.evaluate(LAN_FIT)
        left, top, right, bottom = found["row"]
        b_left, b_top, b_right, b_bottom = found["button"]
        assert left <= b_left and b_right <= right + 0.5 and top <= b_top and b_bottom <= bottom + 0.5, (where, found)
        assert right <= found["width"] and found["inside"] <= 0 and found["sideways"] <= 0 and found["page"] <= 0, (where, found)

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
    ui["settings"]["_meta"].update(recommended_url=SECOND, warning="")
    _read_again(page)
    expect(row.get_by_role("link", name=SECOND)).to_have_attribute("href", SECOND)
    expect(row.get_by_role("button", name="Copy link")).to_have_count(1)
    expect(warning).to_have_text("")  # the warning leaves with its cause
    row.get_by_role("button", name="Copy link").click()
    expect(toasts.last).to_have_text("LAN URL copied.")
    assert recorded(page)["writes"] == [FIRST, SECOND]

    # Only a real LAN address can be copied: a placeholder shows no button, and loopback no row at all.
    ui["settings"]["_meta"].update(reachability="host_ip_unknown", recommended_url="http://your-host-ip:8765")
    _read_again(page)
    expect(row.locator("[data-copy-lan-url]")).to_have_count(0)
    expect(row).to_be_visible()
    expect(row.locator("[data-setting-meta]")).to_contain_text("could not be found")
    ui["settings"]["_meta"].update(reachability="loopback_only", recommended_url="")
    _read_again(page)
    expect(row).to_be_hidden()
    ui["settings"]["_meta"].update(reachability="lan_reachable", recommended_url=SECOND)
    _read_again(page)
    expect(row.get_by_role("button", name="Copy link")).to_have_count(1)

    # No async clipboard (a non-secure LAN page): the selection path copies exactly the address.
    set_sink(page, "missing", True, reset=True)
    row.get_by_role("button", name="Copy link").click()
    expect(toasts.last).to_have_text("LAN URL copied.")
    assert recorded(page)["selections"] == [{"command": "copy", "tag": "TEXTAREA", "text": SECOND, "readOnly": True, "inDialog": False}]
    # Refused by the API and by the engine: the failure is said, and no success follows it.
    set_sink(page, "refused", False, reset=True)
    row.get_by_role("button", name="Copy link").click()
    expect(toasts.last).to_have_text("Could not copy: Clipboard access is unavailable")
    assert recorded(page)["writes"] == []
    assert page.evaluate("() => document.querySelectorAll('.chat-copy-fallback').length") == 0

    # A phone: the action stays inside the row and nothing scrolls sideways.
    page.set_viewport_size({"width": 320, "height": 640})
    page.evaluate("() => document.querySelectorAll('#toast-stack .toast').forEach((toast) => toast.click())")
    fit("phone")
    capture(page, f"settings-lan-copy-{engine}-phone")
    page.set_viewport_size({"width": 1360, "height": 900})

    # One listener lives as long as the page is open: opened again after Back, one press still copies once.
    set_sink(page, "ok", True, reset=True)
    assert go_back(page) == ""
    _read_again(page)
    row.get_by_role("button", name="Copy link").click()
    expect(toasts.last).to_have_text("LAN URL copied.")
    assert recorded(page)["writes"] == [SECOND]


def test_a_long_address_wraps_inside_its_row_and_stays_whole(subscription_ui):
    from playwright.sync_api import expect

    ui, page = subscription_ui, subscription_ui["page"]
    address = FIRST + "/" + "long-deployment-prefix-" * 5
    page.set_viewport_size({"width": 320, "height": 640})
    row = _open(page, ui, reachability="lan_reachable", recommended_url=address, warning=WARNING)
    button = row.get_by_role("button", name="Copy link")
    expect(button).to_be_visible()
    row.scroll_into_view_if_needed()
    found = page.evaluate(LAN_FIT)
    assert found["inside"] <= 0 and found["sideways"] <= 0 and found["page"] <= 0, found
    expect(row.get_by_role("link", name=address)).to_have_attribute("href", address)
    assert row.get_by_role("link", name=address).inner_text() == address, "the whole address, never a cut one"
    expect(row.locator("[data-setting-result]")).to_have_text(WARNING)
    button.focus()
    button.press("Enter")
    expect(page.locator("#toast-stack .toast").last).to_have_text("LAN URL copied.")
    assert recorded(page)["writes"] == [address]
    expect(button).to_be_focused()
    capture(page, "settings-lan-long-address-phone")


def test_the_address_row_in_a_translated_ui_keeps_the_link_and_copies_the_address(subscription_ui):
    """With a non-English dictionary the overlay translates the row's words and the button as chrome; the link keeps
    its own label and address, and the button still copies the read's address, never a translated word."""
    from playwright.sync_api import expect

    ui, page = subscription_ui, subscription_ui["page"]
    dictionary = {"language": "ru", "english": False, "revision": 1, "entries": {
        "Address for other devices": {"text": "Адрес для других устройств", "provenance": "imported"},
        "Type it in a browser on the same network.": {"text": "Введите его в браузере в той же сети.", "provenance": "imported"},
        "Copy link": {"text": "Копировать адрес", "provenance": "imported"},
    }}
    page.route("**/api/ui/i18n", lambda route: route.fulfill(json=dictionary))
    row = _open(page, ui, reachability="lan_reachable", recommended_url=FIRST, warning="")
    button = row.get_by_role("button", name="Копировать адрес")
    expect(button).to_have_count(1, timeout=30_000)
    expect(row.locator(".ui-row-title")).to_have_text("Адрес для других устройств")
    expect(row.locator("[data-setting-meta]")).to_have_text("Введите его в браузере в той же сети.")
    link = row.get_by_role("link", name=FIRST)
    expect(link).to_have_count(1)
    assert [link.get_attribute("href"), link.get_attribute("target"), link.get_attribute("rel")] == [FIRST, "_blank", "noopener"]
    button.click()
    expect(page.locator("#toast-stack .toast").last).to_have_text("LAN URL copied.")
    assert recorded(page)["writes"] == [FIRST], "the address, not a word of the sentence"

    ui["settings"]["_meta"].update(recommended_url=SECOND)
    _read_again(page)
    expect(row.get_by_role("link", name=SECOND)).to_have_attribute("href", SECOND)
    row.get_by_role("button", name="Копировать адрес").click()
    assert recorded(page)["writes"] == [FIRST, SECOND]
