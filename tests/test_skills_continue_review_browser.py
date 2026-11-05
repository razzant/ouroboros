"""The Skills page continues its own stopped reviewer through the existing card tray (no running backend).

The card row is what the real server projects after a Skills-page review whose session seat stopped with
continuable work (``skill_review_runner._ledger_history_rows``; the end-to-end door is
``test_skill_review_continuation_api``): the job failed without a verdict, its record's history row offers
the seat. The tray keeps a new review (Try again, ``{}``) and the continuation of that saved work as two
choices; only the latter posts ``SkillReviewRequest``, exactly the offered record, bytes and seat.
"""
from __future__ import annotations

import copy

import pytest

from tests._skills_page_harness import URL, capture, no_overflow, open_skills, skills_browser as _skills_browser

pytestmark = [pytest.mark.ui_browser]
skills_browser = _skills_browser
HASH = "e4525ee1b665f43e71d181e2a7e557978932b6bd4726ba34ab46f134e7d44d73"
RECORD = "rl-20261102T151335-70f9fdc1d8b9"
OPERATION = "commit_review_multi_model_review_review_s_1524e07cdfb64c889c2ae8836763f0b1"
REVIEW = "/api/skills/weather/review"
STOPPED = {
    "name": "weather", "source": "external", "type": "script", "version": "0.1.0", "description": "Simple greeter.",
    "payload_root": "skills/external/weather", "enabled": False, "content_hash": HASH, "permissions": [],
    "grants": {"all_granted": True, "usable": False}, "owner_attestable": True, "review_status": "pending",
    "review_stale": False, "executable_review": False, "review_profile": "", "removal": "delete",
    "review_gate": {"status": "pending", "stale": False, "executable_review": False, "blocking_reason": "review_pending"},
    "skill_review": {
        "current": {"status": "failed", "lifecycle_status": "failed", "content_hash": HASH, "record_id": RECORD,
                    "task_id": "api_skill_review", "source": "skills",
                    "terminal_reason": "no verdict was formed (QUORUM_FAILED: {'skill': 'unanswered'})"},
        "history": [{"status": "pending", "lifecycle_status": "succeeded", "content_hash": HASH, "record_id": RECORD,
                     "task_id": "api_skill_review", "source": "skills", "continuable_seats": [{
                         "seat_id": "review-s", "operation_id": OPERATION, "run_id": "run-1", "continues_run": "run-1",
                         "cause": "Account limit reached", "engine_stop": {"cause": "pool_exhausted", "carriers": ["native"]},
                         "cost_so_far": {"usd": 0, "unknown": True}}]}],
        "history_omitted": 0,
    },
}
ANSWERED = copy.deepcopy(STOPPED) | {"review_status": "clean", "executable_review": True,
                                     "review_gate": {"status": "clean", "stale": False, "executable_review": True}}
ANSWERED["skill_review"]["current"].update(status="clean", lifecycle_status="succeeded", terminal_reason="")
ANSWERED["skill_review"]["history"][0].pop("continuable_seats")
ANSWERED["skill_review"]["history"][0]["status"] = "clean"


@pytest.mark.parametrize("width", [1440, 390])
def test_the_card_continues_the_saved_review_only_as_its_own_choice_beside_a_new_one(skills_browser, width):
    from playwright.sync_api import expect

    browser = skills_browser.chromium.launch(headless=True)
    try:
        page = browser.new_page(viewport={"width": width, "height": 900})
        skills = open_skills(page, extensions=[STOPPED])
        page.goto(URL, wait_until="networkidle")
        tray = page.locator(".skills-tray", has_text="The review didn’t finish")
        resume, again = tray.get_by_role("button", name="Continue review"), tray.get_by_role("button", name="Try again")
        assert resume.is_visible() and again.is_visible() and no_overflow(page)
        assert "A reviewer stopped with work it can continue from where it stopped." in tray.inner_text()
        resume.focus()
        assert page.evaluate("() => document.activeElement.textContent.trim()") == "Continue review"  # keyboard reachable
        capture(page, "chromium", width, "continue-offered")

        dialog = page.locator(".confirm-dialog")
        resume.click()
        assert "Continue the review of weather?" in dialog.inner_text()
        capture(page, "chromium", width, "continue-confirm")
        dialog.locator("[data-confirm-cancel]").last.click()
        assert skills.posted(REVIEW) == []  # declined: nothing posted

        # The other choice stays an ordinary new review of the current bytes (answered: its wave stopped again).
        skills.answers[REVIEW] = {"skill": "weather", "status": "pending", "record_id": "rl-new", "content_hash": HASH,
                                  "error": "no verdict was formed (QUORUM_FAILED: {'skill': 'unanswered'})"}
        again.click()
        dialog.locator("[data-confirm-ok]").click()
        skills.wait_for_count(("POST", REVIEW), 1)
        assert skills.posted(REVIEW) == [{}]

        skills.hold.add(REVIEW)
        page.locator(".skills-tray", has_text="The review didn’t finish").get_by_role("button", name="Continue review").click()
        dialog.locator("[data-confirm-ok]").click()
        skills.wait_for_request(REVIEW)
        assert skills.posted(REVIEW)[-1] == {"review_reference": {"record_id": RECORD, "content_hash": HASH},
                                             "continue_seats": [{"seat_id": "review-s", "operation_id": OPERATION}]}
        skills.set_skills([ANSWERED])
        skills.release(REVIEW, {"skill": "weather", "status": "clean", "record_id": RECORD, "content_hash": HASH})
        page.get_by_role("button", name="Continue review").wait_for(state="detached", timeout=10_000)
        # The running-state repaint already detaches the button; await the response's own completion notice.
        expect(page.locator(".toast").last).to_contain_text("weather: review clean", timeout=10_000)
        page.wait_for_function("() => !document.querySelector('.toast')", timeout=15_000)
        assert page.locator(".skills-tray", has_text="The review didn’t finish").count() == 0 and no_overflow(page)
        capture(page, "chromium", width, "continue-answered")
    finally:
        skills_page = locals().get("skills")
        if skills_page is not None:
            skills_page.close(browser)
        else:
            browser.close()
