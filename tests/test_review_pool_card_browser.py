"""The review pool's owner-facing facts in a real browser, at phone and desktop widths (IMPL_SPEC §5).

A ledger record made by the real builder, carrying the facts the strictness-keyed composer
records, reaches the task card through the real projection; the card's Reviews detail says
them in words. Settings → Agents says what a Reviewer tick means under the SAVED strictness:
a drafted Blocking changes nothing until Save, and the saved choice repaints the line. Every
fact asserted is visible text, nothing depends on hover, and neither width scrolls sideways.
"""
from __future__ import annotations

import os
import pathlib
from types import SimpleNamespace

import pytest

from ouroboros import review_ledger as rl
from ouroboros.agent_task_pipeline import _store_task_result
from ouroboros.post_task_synthesis import _record_task_facts
from tests.test_review_record_card_projection import _pool_record
from tests.test_ui_smoke_playwright import _browser_missing, _open_review_checkpoint
from tests.ui_chat_viewport_smoke import wait_drawer_closed

pytest_plugins = ("tests.test_ui_smoke_playwright",)
pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]

WIDTHS = {"phone": {"width": 390, "height": 844}, "desktop": {"width": 1280, "height": 900}}
TASK_ID = "pool-card"
CARD_LINES = (
    "Panel: Advisory · chosen by Ouroboros · 2 of 3 ticked reviewers · 1 adviser, not counted",
    "Why these reviewers: two engines are enough for a docs fix",
    "Not reached: critic (engine refused token ***REDACTED***)",
    "Wave total: $0.42 for the priced seats · 1 seat on a subscription · 1 seat cost unknown",
)
TICK = {
    "advisory": "Counts when chosen: for each check Ouroboros asks one, several or all ticked reviewers"
                " and writes why; it never skips a check that is owed.",
    "blocking": "Counts toward review: every ticked reviewer checks each change Ouroboros makes to itself,"
                " each skill and each required result; a critical finding stops it.",
}
NO_SIDEWAYS = "el => el.scrollWidth <= el.clientWidth + 1"


def _seed_card(fixture) -> None:
    """While the server is stopped: the record in the ledger, then the task's real terminal write and
    host task-facts row in Main, both projecting that record (the producers a finished task runs)."""
    root = fixture["data_dir"]
    fixture["stop_server"]()
    rl.write_record(root, _pool_record(TASK_ID))
    env, task = SimpleNamespace(drive_root=root, repo_dir=root), {"id": TASK_ID, "type": "task", "chat_id": 1}
    _store_task_result(env, task, "done", {"rounds": 1, "cost": 0}, {"tool_calls": []})
    _record_task_facts(env, task, {"rounds": 1}, {"tool_calls": []}, root / "logs")
    fixture["start_server"]()


def _go(page, place: str) -> None:
    """Open a place the way an owner does: from the drawer on a phone, from the column on a desktop."""
    phone = page.viewport_size["width"] < 720
    if phone:
        page.locator("[data-mobile-nav-toggle]:visible").first.click()
        page.wait_for_function("() => document.body.classList.contains('nav-drawer-open')", timeout=10_000)
    page.click(f'[data-nav-page="{place}"]')
    if phone:  # choosing a place closes the drawer; nothing is read until the room has settled back over it
        wait_drawer_closed(page, timeout=10_000)


def _open_agents(page, url: str):
    page.goto(url, wait_until="domcontentloaded", timeout=30_000)
    _go(page, "settings")
    page.wait_for_function("() => document.querySelector('#btn-save-settings')?.disabled === false", timeout=30_000)
    page.click('[data-settings-tab="agents"]')
    caption = page.locator("#subagents-section [data-reviewer-tick-caption]")
    caption.wait_for(state="visible", timeout=30_000)
    return caption


def _assert_contained(page, element, width: int) -> None:
    assert element.evaluate(NO_SIDEWAYS), "the element scrolls sideways"
    assert page.evaluate("() => document.documentElement.scrollWidth <= window.innerWidth + 1"), "the page scrolls sideways"
    box = element.bounding_box()
    assert box and box["x"] >= 0 and box["x"] + box["width"] <= width + 1, box


def test_review_pool_card_and_reviewer_caption_at_phone_and_desktop_widths(direct_server_with_data, tmp_path):
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    evidence = pathlib.Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR") or tmp_path / "evidence")
    evidence.mkdir(parents=True, exist_ok=True)
    _seed_card(direct_server_with_data)
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            try:
                for name, viewport in WIDTHS.items():
                    page = browser.new_page(viewport=viewport)
                    page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                    card = page.locator(f'.chat-live-card[data-task-id="{TASK_ID}"]')
                    card.wait_for(state="attached", timeout=30_000)
                    _open_review_checkpoint(card)
                    detail = card.locator("[data-review-attempt-detail]").first
                    detail.wait_for(state="visible", timeout=10_000)
                    # Record facts is a definition list: each visible label/value pair still
                    # reconstructs the exact sentence the old plain-text detail printed.
                    shown = detail.locator('dl').evaluate("""dl => [...dl.querySelectorAll('dd')].map(dd => {
                        const label = dd.previousElementSibling;
                        return (label?.tagName === 'DT' ? label.textContent + ': ' : '') + dd.textContent;
                    })""")
                    for line in CARD_LINES:
                        assert line in shown, (name, line, shown)
                    assert any(line.startswith("Reviewer critic: adviser, not counted · cost unknown") for line in shown)
                    _assert_contained(page, detail, viewport["width"])
                    # Record facts keeps its values readable at every width: a long label never squeezes the value
                    # column below the label column (a phone once read the record id two characters a line).
                    columns = detail.locator('dl').evaluate("""dl => { const dt = dl.querySelector('dt');
                        const dd = dt?.nextElementSibling; return dt && dd ? [dt.getBoundingClientRect().width,
                        dd.getBoundingClientRect().width] : null; }""")
                    assert columns and columns[1] >= columns[0], (name, columns)
                    detail.scroll_into_view_if_needed()
                    card.screenshot(path=str(evidence / f"review-pool-card-{name}.png"))
                    page.close()

                # Desktop: the saved Advisory, a drafted Blocking that changes nothing, then the saved Blocking.
                page = browser.new_page(viewport=WIDTHS["desktop"])
                caption = _open_agents(page, url)
                assert caption.inner_text() == TICK["advisory"]
                page.click('[data-settings-tab="behavior"]')
                page.click('[data-effort-target="s-review-enforcement"] [data-effort-value="blocking"]')
                page.click('[data-settings-tab="agents"]')
                assert caption.inner_text() == TICK["advisory"], "a draft is not the saved strictness"
                with page.expect_response(lambda r: r.url.endswith("/api/settings") and r.request.method == "POST",
                                          timeout=30_000) as answer:
                    page.click("#btn-save-settings")
                assert answer.value.request.post_data_json["OUROBOROS_REVIEW_ENFORCEMENT"] == "blocking"
                page.wait_for_function("t => document.querySelector('#subagents-section [data-reviewer-tick-caption]')"
                                       "?.textContent === t", arg=TICK["blocking"], timeout=30_000)
                _assert_contained(page, caption, WIDTHS["desktop"]["width"])
                caption.scroll_into_view_if_needed()
                page.screenshot(path=str(evidence / "reviewer-caption-desktop.png"))
                page.close()

                # Phone: a fresh page reads the saved Blocking and the line wraps inside the column.
                page = browser.new_page(viewport=WIDTHS["phone"])
                caption = _open_agents(page, url)
                assert caption.inner_text() == TICK["blocking"]
                assert "each of them reviews every change" not in page.locator("#subagents-section").inner_text()
                _assert_contained(page, caption, WIDTHS["phone"]["width"])
                caption.scroll_into_view_if_needed()
                page.screenshot(path=str(evidence / "reviewer-caption-phone.png"))
                page.close()
            finally:
                browser.close()
    except PlaywrightError as exc:
        if _browser_missing(exc):
            pytest.skip(str(exc))
        raise
