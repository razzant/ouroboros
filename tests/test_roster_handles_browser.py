"""Roster rows are named by their card in a REAL render: one row and the full 26,
desktop and phone width; the Duplicate draft, a repeated review and a switched-off
reviewer in the Available-subagents editor. Controlled API responses; no runtime,
no model calls."""
from __future__ import annotations

import pytest
from tests import test_subscription_role_routes_browser as roles

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]
subscription_ui = roles.subscription_ui
role_ui = roles.role_ui

EFFORTS = ("", "low", "medium", "high", "xhigh")
MAX_ROWS = 26
REPEAT = "another independent run of the same model, not a different reviewer"


def _roster(count):
    """Stored ids deliberately carry the rotten role labels the owner must never see."""
    rows = []
    for index in range(count):
        effort = EFFORTS[index % len(EFFORTS)]
        row = {
            "subagent_id": "fast-scout" if not index else f"fast-scout_copy_{index}",
            "recommended_use": f"Notes {index}",
            "route": ({"kind": "agent_session", "target_id": f"codex=gpt-test-{index}"} if index % 2 == 0
                      else {"kind": "api_model", "target_id": f"openai::gpt-api-{index}"}),
        }
        if effort:
            row["effort"] = effort
        rows.append(row)
    return rows


def _configure(ui, count, marked=()):
    roles.configure_mixed(ui)
    rows = _roster(count)
    for index in marked:
        rows[index]["review_eligible"] = True
    ui["settings"].update(OUROBOROS_SUBAGENTS={"enabled": True, "items": rows})
    return rows


def _saved(ui):
    return roles.saved_catalog(ui)["items"]


@pytest.mark.parametrize("width", [1360, 390])
@pytest.mark.parametrize("count", [1, MAX_ROWS])
def test_every_row_is_named_by_its_card_at_any_roster_size(role_ui, count, width):
    ui = role_ui
    marked = tuple(range(0, count, 3))
    _configure(ui, count, marked)
    ui["page"].set_viewport_size({"width": width, "height": 900})
    page = roles.open_agents(ui)
    cards = page.locator("[data-subagent-row]")
    assert cards.count() == count
    assert page.locator(".available-subagent-heading").all_text_contents() == [
        f"Subagent {index + 1}" for index in range(count)]
    editor = page.locator("#available-subagents-editor")
    text = editor.inner_text()
    assert "fast-scout" not in text and "_copy_" not in text, "a stored id is never the owner's name for a row"
    assert editor.locator(".available-subagents-count").first.inner_text() == f"{count}/{MAX_ROWS}"
    assert page.locator("[data-review-pool-count]").inner_text() == f"Reviewers: {len(marked)}"
    checks = editor.locator('[data-subagent-field="review_eligible"]').evaluate_all("els => els.map(e => e.checked)")
    assert checks == [index in marked for index in range(count)]
    assert page.locator("[data-subagent-add]").is_disabled() is (count == MAX_ROWS)
    cards.nth(count - 1).scroll_into_view_if_needed()
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
    roles.capture(page, f"roster-handles-{count}-rows-{width}")


def test_duplicate_is_a_draft_whose_card_names_its_twin_until_the_engine_changes(role_ui):
    ui = role_ui
    _configure(ui, 2, marked=(1,))  # the reviewer is another engine: the pool is never the subject
    page = roles.open_agents(ui)
    page.locator("[data-subagent-duplicate]").first.click()
    cards = page.locator("[data-subagent-row]")
    assert cards.count() == 3
    copy = cards.nth(1)
    # Named on the card BEFORE any Save click, tinted as an error; the source stays clean.
    meta = copy.locator("[data-subagent-meta]")
    assert "Subagent 2 runs the same engine as Subagent 1" in meta.text_content()
    assert "mark both as Reviewer for a repeated review" in meta.text_content()
    assert meta.get_attribute("data-tone") == "error"
    assert copy.get_attribute("data-invalid") is not None
    assert cards.nth(0).get_attribute("data-invalid") is None
    roles.capture(page, "roster-handles-duplicate-draft")

    posts_before = len([1 for path, _ in ui["posts"] if path == "/api/settings"])
    page.locator("#btn-save-settings").click()
    assert "runs the same engine as Subagent 1" in page.locator("#settings-status").text_content()
    assert len([1 for path, _ in ui["posts"] if path == "/api/settings"]) == posts_before, "a twin is never sent"

    copy.locator('[data-subagent-field="effort"]').select_option("low")
    assert "same engine" not in (copy.locator("[data-subagent-meta]").text_content() or "")
    assert copy.get_attribute("data-invalid") is None
    with page.expect_response("**/api/settings"):
        page.locator("#btn-save-settings").click()
    saved = _saved(ui)
    assert [row["route"]["target_id"] for row in saved] == ["codex=gpt-test-0"] * 2 + ["openai::gpt-api-1"]
    assert saved[1]["effort"] == "low" and "review_eligible" not in saved[1]
    assert saved[1]["subagent_id"].startswith("subagent_") and "fast-scout" not in saved[1]["subagent_id"]


def test_two_marked_twins_are_a_repeated_review_and_say_so(role_ui):
    ui = role_ui
    _configure(ui, 1, marked=(0,))
    page = roles.open_agents(ui)
    page.locator("[data-subagent-duplicate]").first.click()
    cards = page.locator("[data-subagent-row]")
    copy = cards.nth(1)
    # A copy of a reviewer is a reviewer: one engine twice is a repeat, not a slip.
    assert copy.locator('[data-subagent-field="review_eligible"]').is_checked()
    assert copy.get_attribute("data-invalid") is None
    assert copy.locator("[data-subagent-review-notes]").inner_text() == (
        f"Repeat of Subagent 1: {REPEAT}.")
    assert page.locator("[data-review-pool-count]").inner_text() == "Reviewers: 2"
    copy.scroll_into_view_if_needed()
    roles.capture(page, "roster-handles-repeated-review")
    # Unmarking the copy turns it back into an ordinary twin, which blocks Save.
    copy.locator('[data-subagent-field="review_eligible"]').uncheck()
    assert copy.locator("[data-subagent-review-notes]").is_hidden()
    posts = len(ui["posts"])
    page.locator("#btn-save-settings").click()
    assert "Subagent 2 runs the same engine as Subagent 1" in page.locator("#settings-status").text_content()
    assert len(ui["posts"]) == posts
    copy.locator('[data-subagent-field="review_eligible"]').check()
    with page.expect_response("**/api/settings"):
        page.locator("#btn-save-settings").click()
    saved = _saved(ui)
    assert [(row["route"]["target_id"], row.get("review_eligible")) for row in saved] == [("codex=gpt-test-0", True)] * 2


def test_twins_saved_earlier_are_hinted_and_never_block_an_unrelated_save(role_ui):
    ui = role_ui
    rows = _configure(ui, 2, marked=(1,))
    rows.append({**rows[0], "subagent_id": "fast-scout_copy_legacy"})
    page = roles.open_agents(ui)
    twin = page.locator("[data-subagent-row]").nth(2)
    meta = twin.locator("[data-subagent-meta]")
    assert "Runs the same engine as Subagent 1" in meta.text_content()
    assert meta.get_attribute("data-tone") is None and twin.get_attribute("data-invalid") is None
    roles.capture(page, "roster-handles-legacy-twin-hint")
    with page.expect_response("**/api/settings"):
        page.locator("#btn-save-settings").click()
    assert [row["subagent_id"] for row in _saved(ui)] == ["fast-scout", "fast-scout_copy_1", "fast-scout_copy_legacy"]
    # Editing the roster is what makes the twin a Save-blocking error.
    page.locator("[data-subagent-row]").first.locator('[data-subagent-field="recommended_use"]').fill("New words")
    posts = len(ui["posts"])
    page.locator("#btn-save-settings").click()
    assert "Subagent 3 runs the same engine as Subagent 1" in page.locator("#settings-status").text_content()
    assert twin.get_attribute("data-invalid") is not None and len(ui["posts"]) == posts


def test_a_switched_off_reviewer_keeps_its_mark_and_leaves_the_pool(role_ui):
    ui = role_ui
    rows = _configure(ui, 3, marked=(0, 1))
    rows[0]["enabled"] = False
    page = roles.open_agents(ui)
    card = page.locator("[data-subagent-row]").first
    assert not card.locator('[data-subagent-field="enabled"]').is_checked()
    assert card.locator('[data-subagent-field="review_eligible"]').is_checked()
    assert "Switched off, so not in the review pool" in card.locator("[data-subagent-review-facts]").inner_text()
    assert "In the review pool" in page.locator("[data-subagent-row]").nth(1).locator(
        "[data-subagent-review-facts]").inner_text()
    assert page.locator("[data-review-pool-count]").inner_text() == "Reviewers: 1"
    assert "review stays on for rows marked Reviewer" in page.locator("[data-review-pool-stays]").inner_text()
    roles.capture(page, "roster-handles-switched-off-row")


def test_an_empty_pool_is_refused_until_the_owner_saves_without_reviewers(role_ui):
    ui = role_ui
    _configure(ui, 2, marked=(0,))
    page = roles.open_agents(ui)
    empty = page.locator("[data-review-pool-empty]")
    assert empty.is_hidden()
    page.locator("[data-subagent-row]").first.locator('[data-subagent-field="review_eligible"]').uncheck()
    assert page.locator("[data-review-pool-count]").inner_text() == "Reviewers: 0"
    assert "reviews will not run and will report “not performed”" in empty.inner_text()
    posts = len(ui["posts"])
    page.locator("#btn-save-settings").click()
    assert "No row is marked Reviewer" in page.locator("#settings-status").text_content()
    assert len(ui["posts"]) == posts, "an empty pool is never saved by accident"
    empty.scroll_into_view_if_needed()
    roles.capture(page, "roster-handles-empty-pool-refused")
    page.locator("[data-review-pool-allow-empty]").check()
    with page.expect_response("**/api/settings"):
        page.locator("#btn-save-settings").click()
    body = [body for path, body in ui["posts"] if path == "/api/settings"][-1]
    assert body["allow_empty_review_pool"] is True
    assert not any(row.get("review_eligible") for row in body["OUROBOROS_SUBAGENTS"]["items"])
