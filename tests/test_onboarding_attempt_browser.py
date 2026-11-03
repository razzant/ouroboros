"""Check status after an unknown onboarding save, on a real install that is already ready (#682).

``direct_server_with_data`` runs the real server over a settings file whose provider is
startup-ready, so ``GET /api/onboarding`` answers 204 whatever happens to a new save: the
condition under which the old Check status closed the builtin wizard on a save nobody had
seen land. Playwright decides only what reaches the browser. The first Start never reaches
the server (the page gets the typed unknown 503); the explicit Retry reaches it and its
answer is dropped. Everything else is the real wizard reading back its own attempt from
the real endpoint. Screenshots go to ``OUROBOROS_UI_EVIDENCE_DIR``.
"""

from __future__ import annotations

import json
import os
import pathlib

import pytest

pytest_plugins = ("tests.test_ui_smoke_playwright",)
pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]

SAVE_TIMEOUT = {"code": "settings_save_timeout", "saved": None, "error": (
    "the settings save is still running in the server after 60s and was left to finish on its own; "
    "reload Settings to see what landed")}
UNKNOWN = "The save result is unknown. Your draft has not been discarded."
NOT_CONFIRMED = "This save is not confirmed. Your draft remains here. You can check again or choose Retry save."
STORED = "This save is stored. Setup completion and restart requirements are not yet confirmed."
# engine, theme, viewport, how far the run goes: "complete" saves for real, "pro" also changes the access level.
RUNS = (("chromium", "light", (1100, 750), "complete"), ("chromium", "dark", (1100, 750), "unknown"),
        ("webkit", "light", (1100, 750), "pro"), ("webkit", "dark", (390, 844), "unknown"))


def _saved_attempt(data_dir: pathlib.Path) -> str:
    return str(json.loads((data_dir / "settings.json").read_text(encoding="utf-8")).get("OUROBOROS_ONBOARDING_ATTEMPT_ID") or "")


def _to_summary(page, *, pro: bool) -> None:
    page.wait_for_selector(".wizard-step", timeout=60_000)
    assert page.locator(".wizard-step").count() == 5
    for _ in range(4):  # Accounts → Models → Review → Budget → Summary
        title = page.locator(".step-title").inner_text()
        if pro and page.locator('[data-runtime-mode="pro"]').count():
            page.locator('[data-runtime-mode="pro"]').click()
        page.locator("#next-btn:not([disabled])").click(timeout=60_000)
        page.wait_for_function("t => document.querySelector('.step-title')?.textContent !== t", arg=title, timeout=30_000)
    page.wait_for_selector(".summary-card", timeout=30_000)


def _start(page, posts) -> None:
    """Start once the wizard's own preview gate opens (it refuses to send while subagents update)."""
    sent = len(posts)
    for _ in range(60):
        page.locator("#next-btn:not([disabled])").click(timeout=30_000)
        page.wait_for_timeout(500)
        if len(posts) > sent:
            return
    raise AssertionError("the wizard never sent the completion")


def test_check_status_confirms_only_this_attempt_on_an_already_ready_install(direct_server_with_data):
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import expect, sync_playwright

    url, data_dir = direct_server_with_data["url"], direct_server_with_data["data_dir"]
    evidence = pathlib.Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", str(data_dir.parent)))
    evidence.mkdir(parents=True, exist_ok=True)
    try:
        with sync_playwright() as pw:
            for engine, theme, (width, height), reach in RUNS:
                browser = getattr(pw, engine).launch(headless=True)
                try:
                    context = browser.new_context(viewport={"width": width, "height": height}, color_scheme=theme)
                    page = context.new_page()
                    errors, posts, outcomes = [], [], []
                    page.on("pageerror", lambda error, errors=errors: errors.append(str(error)))
                    assert page.request.get(url + "/api/onboarding").status == 204, "the install is ready before this save"

                    def complete(route, posts=posts, outcomes=outcomes):
                        posts.append(json.loads(route.request.post_data or "{}"))
                        if len(posts) == 1:  # never reaches the server: the page only learns "unknown"
                            route.fulfill(status=503, content_type="application/json", body=json.dumps(SAVE_TIMEOUT))
                            return
                        answer = route.fetch()  # the real completion runs; its answer is then lost
                        outcomes.append((answer.status, answer.json()))
                        route.abort("connectionreset")

                    page.route("**/api/onboarding/complete", lambda route: complete(route)
                               if route.request.method == "POST" else route.fallback())
                    before = _saved_attempt(data_dir)
                    page.goto(url + "/onboarding?builtin=1", wait_until="domcontentloaded", timeout=30_000)
                    _to_summary(page, pro=reach == "pro")
                    _start(page, posts)
                    first = posts[0]["attempt_id"]
                    error = page.locator(".wizard-error")
                    expect(error).to_contain_text(UNKNOWN)
                    expect(page.locator("#check-save-btn")).to_have_class("btn btn-primary")
                    expect(page.locator(".footer-copy")).to_have_text("The last save did not return a confirmed result.")
                    page.screenshot(path=str(evidence / f"682-{engine}-{theme}-{width}-unknown.png"))

                    page.locator("#check-save-btn").click()
                    expect(error).to_have_text(NOT_CONFIRMED)
                    error.scroll_into_view_if_needed()
                    assert "/onboarding" in page.url, "a ready older provider no longer closes the wizard"
                    assert _saved_attempt(data_dir) == before != first
                    overflow = page.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
                    assert overflow <= 1, overflow
                    page.screenshot(path=str(evidence / f"682-{engine}-{theme}-{width}-not-confirmed.png"))
                    if reach == "unknown":
                        assert not errors, errors
                        continue

                    # The stored-but-finishing answer, rendered by the real page (only this GET is stubbed;
                    # the server side of that state is tests/test_onboarding_attempt_status.py).
                    def asks_first(address, first=first):
                        return address.endswith(f"/api/onboarding/complete?attempt={first}")

                    page.route(asks_first, lambda route: route.fulfill(
                        content_type="application/json", body=json.dumps({"saved": True})))
                    page.locator("#check-save-btn").click()
                    expect(error).to_have_text(STORED)
                    page.screenshot(path=str(evidence / f"682-{engine}-{theme}-{width}-stored-stubbed-get.png"))
                    page.unroute(asks_first)

                    page.locator("#next-btn").click()  # "Retry save": a new attempt, whose answer is lost
                    expect(error).to_contain_text("The answer to this save did not arrive")
                    second = posts[1]["attempt_id"]
                    assert second != first and outcomes[0][0] == 200, outcomes
                    assert _saved_attempt(data_dir) == second, "the server saved the retry"
                    page.screenshot(path=str(evidence / f"682-{engine}-{theme}-{width}-reply-lost.png"))

                    page.locator("#check-save-btn").click()
                    if reach == "pro":
                        expect(page.locator(".step-title")).to_have_text("Setup saved — restart to apply it")
                        expect(page.locator(".panel-card code")).to_have_text("pro")
                        expect(page.locator(".panel-card code")).to_be_visible()
                        page.screenshot(path=str(evidence / f"682-{engine}-{theme}-{width}-restart.png"))
                    else:
                        page.wait_for_url(url + "/", timeout=30_000)
                        page.screenshot(path=str(evidence / f"682-{engine}-{theme}-{width}-completed.png"))
                    read_back = {attempt: page.request.get(url + "/api/onboarding/complete", params={"attempt": attempt}).json()
                                 for attempt in (first, second)}
                    assert read_back[first] == {"saved": None}, read_back
                    assert read_back[second] == {"saved": True, "receipt": outcomes[0][1]}, read_back
                    assert len(posts) == 2, "checking never sends the save again"
                    assert not errors, errors
                finally:
                    browser.close()
    except PlaywrightError as exc:
        if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower():
            pytest.skip(str(exc))
        raise
