"""Settings › Agents in a REAL render: every agent is named by its route, never by a stored id, for one row and
for the full 26, at desk and phone width; a Duplicate is an unsaved agent whose page names its twin; a repeated
review; a switched-off reviewer; the last reviewer's question. Controlled API responses; no runtime, no model
calls."""
from __future__ import annotations

import pytest
from tests import test_subscription_role_routes_browser as roles
from tests.settings_browser_helpers import open_settings, settings_page
from tests.test_agents_settings_browser import ASKS, RosterServer

pytestmark = [pytest.mark.ui_browser]
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


def _roster_posts(ui):
    return len(roles.roster_writes(ui))


def _level(page, path="agents"):
    return settings_page(page, path)


@pytest.mark.parametrize("width", [1360, 390])
@pytest.mark.parametrize("count", [1, MAX_ROWS])
def test_every_agent_is_named_by_its_route_at_any_roster_size(role_ui, count, width):
    ui = role_ui
    marked = tuple(range(0, count, 3))
    rows = _configure(ui, count, marked)
    ui["page"].set_viewport_size({"width": width, "height": 900})
    page = roles.open_agents(ui)
    level = _level(page)
    heads = level.locator("[data-entry-id]")
    assert heads.count() == count
    # The model in short, then the facets the handle spells; a stored id is never the owner's name for an agent.
    names = [title.split("\n")[0].strip() for title in level.locator("[data-entry-id] .ui-row-title").all_inner_texts()]
    assert names == [row["route"]["target_id"].replace("codex=", "").replace("openai::", "")
                     + (f"/{row['effort']}" if row.get("effort") else "") for row in rows]
    text = level.inner_text()
    assert "fast-scout" not in text and "_copy_" not in text and "Subagent " not in text
    assert level.locator(".agents-card-title").nth(0).inner_text().strip() == f"{count} agent{'' if count == 1 else 's'}"
    assert level.locator('[data-agents-filter="reviewers"]').inner_text().strip() == f"Reviewers {len(marked)}"
    assert level.locator("[data-entry-id] .agent-chip", has_text="Reviewer").count() == len(marked)
    assert heads.locator("[data-entry-switch]").evaluate_all("els => els.every(e => e.checked)")
    add = level.locator("[data-subagent-add]")
    assert add.is_disabled() is (count == MAX_ROWS)
    assert ("The list holds at most 26 agents." in text) is (count == MAX_ROWS)
    heads.nth(count - 1).scroll_into_view_if_needed()
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
    assert level.evaluate("el => el.scrollWidth <= el.clientWidth + 1")
    roles.capture(page, f"roster-handles-{count}-rows-{width}")


def test_duplicate_is_an_unsaved_agent_whose_page_names_its_twin_until_the_engine_changes(role_ui):
    ui = role_ui
    _configure(ui, 2, marked=(1,))  # the reviewer is another engine: the pool is never the subject
    form = roles.open_agent(ui, "fast-scout")
    page = ui["page"]
    form.locator("[data-subagent-duplicate]").click()
    copy = _level(page, "agents/new").locator("[data-subagent-row]")
    copy.wait_for(state="visible")
    assert _level(page, "agents/new").locator("[data-settings-back]").inner_text().strip() == "gpt-test-0"
    # Named on the page BEFORE any Save, tinted as an error, in the agents' names (a page has no numbers).
    meta = copy.locator("[data-subagent-meta]")
    assert "gpt-test-0 runs the same engine as gpt-test-0" in meta.text_content()
    assert "mark both as Reviewer for a repeated review" in meta.text_content() and "Subagent" not in meta.text_content()
    assert meta.get_attribute("data-tone") == "error"
    assert copy.get_attribute("data-invalid") is not None
    # An unsaved agent's on/off is a field of its form, and it has nothing to remove yet.
    assert copy.locator('[data-subagent-field="enabled"]').get_attribute("role") is None
    assert copy.locator("[data-editor-remove]").inner_text().strip() == "Discard"
    roles.capture(page, "roster-handles-duplicate-draft")

    copy.locator("[data-editor-save]").click()
    assert "runs the same engine as" in copy.locator("[data-subagents-validation]").text_content()
    assert _roster_posts(ui) == 0, "a twin is never sent"

    copy.locator('[data-subagent-field="effort"]').select_option("low")
    assert "same engine" not in (copy.locator("[data-subagent-meta]").text_content() or "")
    assert copy.get_attribute("data-invalid") is None
    copy.locator("[data-editor-save]").click()
    # Saved, the page gives way to the list, which shows the new agent and gives it the focus.
    _level(page).locator("[data-entry-id]").nth(2).wait_for(state="visible")
    saved = _saved(ui)
    assert [row["route"]["target_id"] for row in saved] == ["codex=gpt-test-0", "openai::gpt-api-1", "codex=gpt-test-0"]
    assert saved[2]["effort"] == "low" and "review_eligible" not in saved[2]
    assert saved[2]["subagent_id"].startswith("subagent_") and "fast-scout" not in saved[2]["subagent_id"]
    assert page.evaluate("id => document.activeElement?.closest('[data-entry-id]')?.dataset.entryId === id",
                         saved[2]["subagent_id"])


def test_two_marked_twins_are_a_repeated_review_and_say_so(role_ui):
    ui = role_ui
    _configure(ui, 1, marked=(0,))
    form = roles.open_agent(ui, "fast-scout")
    page = ui["page"]
    form.locator("[data-subagent-duplicate]").click()
    copy = _level(page, "agents/new").locator("[data-subagent-row]")
    mark = copy.locator('[data-subagent-field="review_eligible"]')
    # A copy of a reviewer is a reviewer: one engine twice is a repeat, not a slip.
    assert mark.is_checked()
    assert copy.get_attribute("data-invalid") is None
    assert copy.locator("[data-subagent-review-notes]").inner_text() == f"Repeat of gpt-test-0: {REPEAT}."
    roles.capture(page, "roster-handles-repeated-review")
    # Unmarking the copy turns it back into an ordinary twin, which its Save does not send.
    mark.uncheck()
    assert copy.locator("[data-subagent-review-notes]").is_hidden()
    copy.locator("[data-editor-save]").click()
    assert "runs the same engine as gpt-test-0" in copy.locator("[data-subagents-validation]").text_content()
    assert _roster_posts(ui) == 0
    mark.check()
    copy.locator("[data-editor-save]").click()
    _level(page).locator("[data-entry-id]").nth(1).wait_for(state="visible")
    saved = _saved(ui)
    assert [(row["route"]["target_id"], row.get("review_eligible")) for row in saved] == [("codex=gpt-test-0", True)] * 2
    assert _level(page).locator('[data-agents-filter="reviewers"]').inner_text().strip() == "Reviewers 2"


def test_twins_saved_earlier_are_hinted_and_a_switch_elsewhere_is_still_sent(role_ui):
    ui = role_ui
    rows = _configure(ui, 2, marked=(1,))
    rows.append({**rows[0], "subagent_id": "fast-scout_copy_legacy"})
    page = roles.open_agents(ui)
    # A switch on another agent is one write of its entry: the client refuses nothing for a pair it never touched.
    _level(page).locator('[data-entry-id="fast-scout_copy_1"] .ui-switch').click()
    roles.wait_for_switch(page, "fast-scout_copy_1", False)
    assert [row["subagent_id"] for row in _saved(ui)] == ["fast-scout", "fast-scout_copy_1", "fast-scout_copy_legacy"]
    assert _saved(ui)[1]["enabled"] is False
    # The twin's own page hints at the pair without calling it an error...
    twin = roles.open_agent(ui, "fast-scout_copy_legacy", fresh=False)
    meta = twin.locator("[data-subagent-meta]")
    assert "Runs the same engine as gpt-test-0" in meta.text_content()
    assert meta.get_attribute("data-tone") is None and twin.get_attribute("data-invalid") is None
    roles.capture(page, "roster-handles-legacy-twin-hint")
    # ...and editing the twin is what makes it an error its Save does not send.
    twin.locator('[data-subagent-field="recommended_use"]').fill("New words")
    posts = _roster_posts(ui)
    twin.locator("[data-editor-save]").click()
    assert "runs the same engine as gpt-test-0" in twin.locator("[data-subagents-validation]").text_content()
    assert twin.get_attribute("data-invalid") is not None and _roster_posts(ui) == posts


def test_a_switched_off_reviewer_keeps_its_mark_and_leaves_the_pool(role_ui):
    ui = role_ui
    rows = _configure(ui, 3, marked=(0, 1))
    rows[0]["enabled"] = False
    page = roles.open_agents(ui)
    level = _level(page)
    head = level.locator('[data-entry-id="fast-scout"]')
    assert not head.locator("[data-entry-switch]").is_checked()
    assert head.locator(".agent-chip", has_text="Reviewer").count() == 1
    assert [chip.inner_text().strip() for chip in level.locator("[data-agents-filter]").all()][:3] == [
        "All 3", "Reviewers 2", "Off 1"]
    # The roster's own switch off: one write, and the page says what still reviews.
    level.locator('[data-setting-row="OUROBOROS_SUBAGENTS"] .ui-switch').click()
    page.wait_for_function("() => document.querySelector('#page-settings [data-review-pool-stays]')?.hidden === false")
    assert level.locator("[data-review-pool-stays]").inner_text() == "Helpers are off; agents marked Reviewer still review."
    assert roles.saved_catalog(ui)["enabled"] is False and len(_saved(ui)) == 3
    roles.capture(page, "roster-handles-switched-off-row")
    # Its own page: the mark holds, and the exception is said under the switches.
    form = roles.open_agent(ui, "fast-scout", fresh=False)
    assert not form.locator('[data-subagent-field="enabled"]').is_checked()
    assert form.locator('[data-subagent-field="review_eligible"]').is_checked()
    assert form.locator("[data-subagent-review-exception]").inner_text() == "Switched off, so not in the review pool."
    other = roles.open_agent(ui, "fast-scout_copy_1", fresh=False)
    assert other.locator("[data-subagent-review-exception]").is_hidden()


def test_the_last_reviewer_is_not_unmarked_until_the_owner_confirms(role_ui):
    """The server refuses a roster that leaves no reviewer; the page asks in the shared confirm and sends the same
    body again with the owner's confirmation (LANE A's refusal, answered here by the roster fake)."""
    ui = role_ui
    rows = _configure(ui, 2, marked=(0,))
    server = RosterServer(ui, rows)
    server.serve(ui["page"])
    form = roles.open_agent(ui, "fast-scout")
    page = ui["page"]
    mark = form.locator('[data-subagent-field="review_eligible"]')
    switch = form.locator('.agent-use:has([data-subagent-field="review_eligible"]) .ui-switch')
    assert mark.is_checked() and _level(page, "agents/fast-scout").locator("[data-review-pool-empty]").count() == 0
    switch.click()
    dialog = page.locator(".confirm-dialog")
    dialog.wait_for(state="visible")
    assert ASKS in dialog.inner_text()
    roles.capture(page, "roster-handles-empty-pool-refused")
    dialog.locator("[data-confirm-cancel]").click()
    page.wait_for_function("""() => { const input = document.querySelector('#page-settings [data-subagent-field="review_eligible"]');
        return input && input.checked && input.getAttribute('aria-busy') !== 'true'; }""")
    assert len(server.posts) == 1 and server.agent("fast-scout").get("review_eligible") is True, "No keeps the reviewer"
    switch.click()
    dialog.wait_for(state="visible")
    dialog.locator("[data-confirm-ok]").click()
    page.wait_for_function("""() => { const input = document.querySelector('#page-settings [data-subagent-field="review_eligible"]');
        return input && !input.checked && input.getAttribute('aria-busy') !== 'true'; }""")
    assert server.posts[2] == {**server.posts[1], "allow_empty_review_pool": True}
    assert not any(row.get("review_eligible") for row in server.roster["items"])
    assert not [body for path, body in ui["posts"] if path == "/api/settings"], "the roster never takes the generic save"
    # The list then says what an empty pool means.
    open_settings(page, "agents")
    empty = _level(page).locator("[data-review-pool-empty]")
    page.wait_for_function("() => document.querySelector('#page-settings [data-review-pool-empty]')?.hidden === false")
    assert "so the main model reviews alone: one model, no second opinion" in empty.inner_text()
