"""The shared browser fixture's page errors in real Chromium and WebKit (tests/ui_console_errors.py).

``subscription_ui`` watches every page it opens and keeps each page's every error; on a page the test declares it
navigates, only WebKit's report of a read of that page's own origin is set aside. The real generator reads a fixed
address the page has already read successfully, again from ``pagehide`` while a reload unloads the page: no nonce, as
the app polls. ``route.abort("accessdenied")`` is a labelled contract control, a test-only stand-in WebKit words the
same way. 127.0.0.1 and localhost are different origins: a read across them is a real refusal. Static-server fixture;
no runtime starts.
"""
from __future__ import annotations

import pytest

from tests import test_subscription_setup_browser as setup_browser

subscription_ui = setup_browser.subscription_ui
pytestmark = [pytest.mark.ui_browser]

BLANK = "<!doctype html><title>Blank</title>"  # a document of the fixture's origin that reads nothing itself
# Uncaught a moment after the call returns: a page error, where a throw inside evaluate only fails the call.
THROW = "text => new Promise((done) => setTimeout(() => { setTimeout(done, 100); throw new TypeError(text); }))"
READ = "url => fetch(url).then(() => 'read', () => 'refused')"  # the page catches its own failed read
READ_WHILE_LEAVING = "url => { addEventListener('pagehide', () => { fetch(url).catch(() => {}); }); }"
LOSE_READ = "url => { fetch(url); }"  # no handler: the rejection is the page's own


@pytest.fixture(params=["chromium", "webkit"])
def engine_ui(request, monkeypatch):
    monkeypatch.setenv("OUROBOROS_UI_BROWSER_ENGINE", request.param)
    ui = request.getfixturevalue("subscription_ui")
    ui["engine"] = request.param
    return ui


def _open(page, address):
    page.route(address, lambda route: route.fulfill(content_type="text/html", body=BLANK))
    page.route("**/api/refused*", lambda route: route.abort("accessdenied"))
    page.goto(address)


def _arrived(page, errors, count):
    """Playwright delivers page errors while a call waits; give ``count`` of them time, and the comparison shows the rest."""
    for _ in range(200):
        if len(errors) >= count:
            return
        page.wait_for_timeout(50)


def test_a_read_a_real_reload_cuts_off_is_set_aside_on_a_declared_page_and_kept_on_a_strict_one(engine_ui):
    ui = engine_ui
    errors, address = ui["errors"], ui["url"] + "/blank"
    strict, navigating = ui["page"], ui["new_page"]()
    for page in (strict, navigating):
        _open(page, address)
        assert page.evaluate(READ, "/api/state") == "read"
        page.evaluate(READ_WHILE_LEAVING, "/api/state")
    errors.watch(navigating)
    for page in (strict, navigating):
        page.reload()
    if ui["engine"] != "webkit":  # Chromium reports no read a navigation cut off
        assert not errors, errors.report()
        return
    _arrived(navigating, errors, 2)
    line = "/" + ui["url"].removeprefix("http://") + "/api/state due to access control checks."
    # At least one line from each page: WebKit coalesces identical lines, so their number is never the claim.
    assert {(entry.page, str(entry), entry.own_fetch) for entry in errors} == {(1, line, False), (2, line, True)}, errors.report()
    assert set(errors.unexpected()) == {line}, errors.report()
    errors.clear()  # every entry was checked above; the fixture's teardown judges whatever comes after


def test_a_declared_page_sets_aside_only_its_own_refused_read_and_every_page_keeps_its_errors(engine_ui):
    """The contract control: a read the page caught after ``accessdenied``, set aside only where declared, from then on."""
    ui = engine_ui
    errors, address = ui["errors"], ui["url"] + "/blank"
    strict, navigating = ui["page"], ui["new_page"]()
    for page in (strict, navigating):
        _open(page, address)
    assert navigating.evaluate(READ, "/api/refused-first") == "refused"  # before the declaration: it stays strict
    _arrived(navigating, errors, 1 if ui["engine"] == "webkit" else 0)
    errors.watch(navigating)
    navigating.reload()
    for page, words in ((strict, "never navigates"), (navigating, "reloaded")):
        assert page.evaluate(READ, "/api/refused") == "refused"
        page.evaluate(THROW, f"thrown on the page that {words}")
    elsewhere = ui["url"].replace("127.0.0.1", "localhost") + "/index.html"
    assert navigating.evaluate(READ, elsewhere) == "refused"
    own = "/" + ui["url"].removeprefix("http://") + "/api/refused due to access control checks."
    first = own.replace("/api/refused", "/api/refused-first")
    foreign = "/" + elsewhere.removeprefix("http://") + " due to access control checks."
    expected = [(1, own, False), (1, "thrown on the page that never navigates", False),
                (2, first, False), (2, own, True), (2, "thrown on the page that reloaded", False), (2, foreign, False)]
    if ui["engine"] != "webkit":  # Chromium reports no refused read as a page error
        expected = [row for row in expected if "access control" not in row[1]]
    assert sorted((entry.page, str(entry), entry.own_fetch) for entry in errors) == sorted(expected), errors.report()
    assert {entry.url for entry in errors} == {address}, errors.report()
    errors.clear()  # every entry was checked above; the fixture's teardown judges whatever comes after


def test_a_read_the_page_left_uncaught_after_its_declared_reload_fails_the_consumers_assertion(engine_ui):
    ui = engine_ui
    errors, address, page = ui["errors"], ui["url"] + "/blank", ui["page"]
    _open(page, address)
    errors.watch(page)
    page.reload()
    assert not errors.unexpected(), errors.report()  # the transition itself left nothing
    with page.expect_event("pageerror", lambda error: "access control" not in str(error)):
        page.evaluate(LOSE_READ, "/api/refused")
    with pytest.raises(AssertionError):
        assert not errors.unexpected(), errors.report()  # the exact statement a consumer ends with
    assert errors.unexpected() == [{"chromium": "Failed to fetch", "webkit": "TypeError: Load failed"}[ui["engine"]]], errors.report()
    errors.clear()  # the failure above is this test's subject; the fixture's teardown judges whatever comes after
