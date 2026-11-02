"""The shared browser fixture's page errors in real Chromium and WebKit (tests/ui_console_errors.py).

``subscription_ui`` watches every page it opens and keeps each page's every error; on a page the test declares it
navigates, only WebKit's report of a read of that page's own origin is set aside. A reload's cancellation cannot be
produced on demand, so the page's own read is refused with ``accessdenied``, which WebKit reports in the same words.
127.0.0.1 and localhost are different origins: a read across them is a real refusal. Static-server fixture; no
runtime starts.
"""
from __future__ import annotations

import pytest

from tests import test_subscription_setup_browser as setup_browser

subscription_ui = setup_browser.subscription_ui
pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]

BLANK = "<!doctype html><title>Blank</title>"  # a document of the fixture's origin that reads nothing itself
# Uncaught a moment after the call returns: a page error, where a throw inside evaluate only fails the call.
THROW = "text => new Promise((done) => setTimeout(() => { setTimeout(done, 100); throw new TypeError(text); }))"
READ = "url => fetch(url).then(() => 'read', () => 'refused')"  # the page catches its own failed read


@pytest.fixture(params=["chromium", "webkit"])
def engine_ui(request, monkeypatch):
    monkeypatch.setenv("OUROBOROS_UI_BROWSER_ENGINE", request.param)
    ui = request.getfixturevalue("subscription_ui")
    ui["engine"] = request.param
    return ui


def test_a_declared_page_sets_aside_only_its_own_refused_read_and_every_page_keeps_its_errors(engine_ui):
    ui = engine_ui
    errors, address = ui["errors"], ui["url"] + "/blank"
    strict, navigating = ui["page"], ui["new_page"]()
    for page in (strict, navigating):
        page.route(address, lambda route: route.fulfill(content_type="text/html", body=BLANK))
        page.route("**/api/refused", lambda route: route.abort("accessdenied"))
        page.goto(address)
    errors.watch(navigating)
    navigating.reload()
    for page, words in ((strict, "never navigates"), (navigating, "reloaded")):
        assert page.evaluate(READ, "/api/refused") == "refused"
        page.evaluate(THROW, f"thrown on the page that {words}")
    elsewhere = ui["url"].replace("127.0.0.1", "localhost") + "/index.html"
    assert navigating.evaluate(READ, elsewhere) == "refused"
    own = "/" + ui["url"].removeprefix("http://") + "/api/refused due to access control checks."
    foreign = "/" + elsewhere.removeprefix("http://") + " due to access control checks."
    expected = [(1, own, False), (1, "thrown on the page that never navigates", False),
                (2, own, True), (2, "thrown on the page that reloaded", False), (2, foreign, False)]
    if ui["engine"] != "webkit":  # Chromium reports no refused read as a page error
        expected = [row for row in expected if "access control" not in row[1]]
    assert sorted((entry.page, str(entry), entry.own_fetch) for entry in errors) == sorted(expected), errors.report()
    assert {entry.url for entry in errors} == {address}, errors.report()
    errors.clear()  # every entry was checked above; the fixture's teardown judges whatever comes after
