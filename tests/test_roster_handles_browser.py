"""Settings › Agents in a REAL render: every agent is named by its route, never by a stored id, for one row and
for the full 50 (whose held Add and Duplicate say why in their hint, in Settings and in onboarding, through a scroll
and a touch), at desk, the owner's window and phone width; a Duplicate is an unsaved agent whose page names its twin;
a repeated review; a switched-off reviewer; the last reviewer's question. Controlled API responses; no runtime, no
model calls."""
from __future__ import annotations

import pytest
from tests import test_subscription_role_routes_browser as roles
from tests.settings_browser_helpers import open_settings, settings_page
from tests.settings_section_helpers import header_back
from tests.test_agents_settings_browser import ASKS, RosterServer

pytestmark = [pytest.mark.ui_browser]
subscription_ui = roles.subscription_ui
role_ui = roles.role_ui

EFFORTS = ("", "low", "medium", "high", "xhigh")
MAX_ROWS = 50
FULL = "The list holds at most 50 agents, including those switched off. Remove one to add another."
FULL_SETUP = FULL.replace("agents", "subagents")
OFF = (7, MAX_ROWS - 1)  # switched off, and still two of the 50
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


def _configure(ui, count=MAX_ROWS, marked=()):
    roles.configure_mixed(ui)
    rows = _roster(count)
    for index in marked:
        rows[index]["review_eligible"] = True
    ui["settings"].update(OUROBOROS_SUBAGENTS={"enabled": True, "items": rows})
    return rows


def _configure_full(ui):
    """The full 50 with two switched off, for Settings and for onboarding's preview."""
    rows = _configure(ui)
    for index in OFF:
        rows[index]["enabled"] = False
    ui["fixture"]["preview"]["available_subagents"] = ui["settings"]["OUROBOROS_SUBAGENTS"]
    return rows


def _saved(ui):
    return roles.saved_catalog(ui)["items"]


def _roster_posts(ui):
    return len(roles.roster_writes(ui))


def _level(page, path="agents"):
    return settings_page(page, path)


@pytest.mark.parametrize("width,height", [(1360, 900), (1100, 750), (390, 844)])
@pytest.mark.parametrize("count", [1, MAX_ROWS])
def test_every_agent_is_named_by_its_route_at_any_roster_size(role_ui, count, width, height):
    ui = role_ui
    marked = tuple(range(0, count, 3))
    rows = _configure(ui, count, marked)
    ui["page"].set_viewport_size({"width": width, "height": height})
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
    assert "holds at most" not in text, "no line of the page says the limit: the held control's hint does"
    if count == MAX_ROWS:  # a held Add: the mouse shows its one hint, inside the window
        _shows_why(page, level, add, f"roster-handles-full-add-hint-{width}")
    heads.nth(count - 1).scroll_into_view_if_needed()
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
    assert level.evaluate("el => el.scrollWidth <= el.clientWidth + 1")
    roles.capture(page, f"roster-handles-{count}-rows-{width}")
    if count == MAX_ROWS:  # the last agent's page: its held Duplicate says the same, and only its hint says it
        form = roles.open_agent(ui, rows[-1]["subagent_id"], fresh=False)
        duplicate = form.locator("[data-subagent-duplicate]")
        duplicate.scroll_into_view_if_needed()
        _shows_why(page, form, duplicate, f"roster-handles-full-duplicate-hint-{width}", keyboard=True)


def _shows_why(page, scope, control, shot, keyboard=False):
    """A held control (aria-disabled, never disabled) shows its one hint to the mouse or the keyboard: the shared fact
    of the full list, inside the window, as the control's description; nothing else in ``scope`` says it."""
    hint = scope.locator("[data-full-hint] .ui-hint")
    assert control.get_attribute("aria-disabled") == "true" and control.get_attribute("disabled") is None
    assert hint.count() == 1 and hint.is_hidden() and control.get_attribute("aria-describedby") == hint.get_attribute("id")
    control.focus() if keyboard else control.hover()
    hint.wait_for(state="visible")
    assert " ".join(hint.inner_text().split()) == FULL
    box, view = hint.bounding_box(), page.viewport_size
    assert box["x"] >= 0 and box["y"] >= 0 and box["x"] + box["width"] <= view["width"] and box["y"] + box["height"] <= view["height"]
    _unclipped(hint)
    roles.capture(page, shot)


def _unclipped(hint):
    assert hint.evaluate("""el => {
        const r = el.getBoundingClientRect();
        for (let a = el.parentElement; a; a = a.parentElement) {
            const c = getComputedStyle(a), b = a.getBoundingClientRect();
            if (/(auto|scroll|hidden|clip)/.test(c.overflowY) &&
                (r.top < b.top + a.clientTop - 1 || r.bottom > b.top + a.clientTop + a.clientHeight + 1)) return false;
            if (/(auto|scroll|hidden|clip)/.test(c.overflowX) &&
                (r.left < b.left + a.clientLeft - 1 || r.right > b.left + a.clientLeft + a.clientWidth + 1)) return false;
        }
        return [.15, .5, .85].every(f => el.contains(document.elementFromPoint(r.left + r.width / 2, r.top + r.height * f)));
    }"""), "the full explanation must be visible through every clipping ancestor"


@pytest.mark.parametrize("width,height", [(1100, 750), (390, 844)])
@pytest.mark.parametrize("theme", ["light", "dark"])
def test_full_add_hint_follows_the_settings_scroll_boundary(role_ui, width, height, theme):
    ui = role_ui
    _configure(ui, MAX_ROWS)
    page = ui["page"]
    page.set_viewport_size({"width": width, "height": height})
    page.add_init_script(f"localStorage.setItem('ouroboros.theme', '{theme}')")
    roles.open_agents(ui)
    level = _level(page)
    add = level.locator('[data-subagent-add]')
    add.focus()
    hint = level.locator('[data-full-hint] .ui-hint')
    hint.wait_for(state='visible')
    # Scroll an already open hint to the independently reproduced clipping edge.
    add.evaluate("e => { const s=e.closest('[data-settings-page]'); s.scrollTop += Math.max(0, e.getBoundingClientRect().top-s.getBoundingClientRect().top-40); }")
    page.evaluate("() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))")
    assert hint.is_visible()
    _unclipped(hint)
    assert ' '.join(hint.inner_text().split()) == FULL
    roles.capture(page, f'roster-add-scroll-{width}-{theme}')
    # New focus at that same edge also places it within the scroller.
    add.evaluate('e => e.blur()')
    _shows_why(page, level, add, f'roster-add-refocus-{width}-{theme}', keyboard=True)


@pytest.mark.parametrize("theme", ["light", "dark"])
@pytest.mark.parametrize("action", ["add", "duplicate"])
def test_full_hints_dismiss_after_an_outside_touch_and_page_return(role_ui, theme, action):
    """Real browser touch emulation, not native iOS: a tap never leaves a limit hint covering the rows."""
    ui = role_ui
    _configure_full(ui)
    page = ui["new_page"](has_touch=True, is_mobile=True, viewport={"width": 390, "height": 844})
    ui["page"] = page
    page.add_init_script(f"localStorage.setItem('ouroboros.theme', '{theme}')")
    roles.open_agents(ui)
    assert _level(page).locator('[data-agents-filter="off"]').inner_text().strip() == f"Off {len(OFF)}"
    scope = roles.open_agent(ui, 'fast-scout', fresh=False) if action == 'duplicate' else _level(page)
    control = scope.locator(f'[data-subagent-{action}]')
    control.scroll_into_view_if_needed()
    control.tap(force=True)  # aria-disabled explains the limit but accepts a touch.
    hint = scope.locator('[data-full-hint] .ui-hint')
    hint.wait_for(state='visible')
    _unclipped(hint)
    roles.capture(page, f'roster-touch-{action}-open-{theme}')
    # The page heading is outside the hint host and does not save or change a row.
    page.locator('#page-settings .app-page-header').tap(position={"x": 150, "y": 15})
    roles.capture(page, f'roster-touch-{action}-outside-{theme}')
    dismissed = hint.is_hidden()
    if action == 'add':
        roles.open_agent(ui, 'fast-scout', fresh=False)
    page.locator('#page-settings .app-page-header').get_by_role('button', name='Back to Agents', exact=True).tap()
    level = _level(page)
    assert level.locator('[data-entry-id]').count() == MAX_ROWS
    assert level.locator('.ui-hint:visible').count() == 0
    roles.capture(page, f'roster-touch-{action}-return-{theme}')
    assert dismissed, 'an outside touch must dismiss the hint before leaving the page'
    assert _roster_posts(ui) == 0, 'showing and dismissing the limit writes nothing'


def _setup_agents(ui):
    """Onboarding's Models step with its Subagents section open; returns its held Add."""
    page = ui["page"]
    page.goto(ui["url"] + "/onboarding")
    page.wait_for_selector("#quick-start-btn:not([hidden])")
    page.click("#next-btn")
    page.wait_for_selector('[data-model-role="main"]')
    page.locator('[data-collapse="subagents"] > summary').click()
    add = page.locator("#onboarding-available-subagents [data-subagent-add]")
    add.wait_for(state="visible")
    return add


def _setup_full(page, add):
    """The setup list counts all 50 and holds Add, two of them switched off."""
    rows = page.locator("#onboarding-available-subagents .available-subagent-row")
    assert rows.count() == MAX_ROWS
    assert page.locator("#onboarding-available-subagents .available-subagents-count").first.inner_text().strip() == f"{MAX_ROWS}/{MAX_ROWS}"
    switches = rows.locator('[data-subagent-field="enabled"]').evaluate_all("els => els.map(e => e.checked)")
    assert [index for index, on in enumerate(switches) if not on] == list(OFF)
    assert add.get_attribute("aria-disabled") == "true" and add.get_attribute("disabled") is None


def _frames(page):
    page.evaluate("() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))")


@pytest.mark.parametrize("width,height", [(1100, 750), (390, 844)])
def test_setup_full_hint_stays_in_the_window_through_a_document_scroll(role_ui, width, height):
    """Onboarding scrolls the document, not a Settings scroller: an open hint is placed again as its control nears the
    top, so its explanation never passes the window (the reviewer's reproduction: 50 rows, Add 40 px from the top)."""
    ui = role_ui
    _configure_full(ui)
    page = ui["page"]
    page.set_viewport_size({"width": width, "height": height})
    add = _setup_agents(ui)
    _setup_full(page, add)
    add.evaluate("e => window.scrollBy(0, e.getBoundingClientRect().top - 350)")
    add.focus()
    hint = page.locator("#onboarding-available-subagents [data-full-hint] .ui-hint").first
    hint.wait_for(state="visible")
    assert " ".join(hint.inner_text().split()) == FULL_SETUP
    _unclipped(hint)
    add.evaluate("e => window.scrollBy(0, e.getBoundingClientRect().top - 40)")
    _frames(page)
    assert page.evaluate("window.scrollY") > 0 and abs(add.bounding_box()["y"] - 40) <= 2
    box, view = hint.bounding_box(), page.viewport_size
    assert hint.is_visible() and hint.get_attribute("data-side") == "below", hint.get_attribute("data-side")
    assert box["y"] >= 0 and box["y"] + box["height"] <= view["height"], box
    _unclipped(hint)
    roles.capture(page, f"setup-full-hint-scrolled-{width}")
    # Scrolled back down the window, it is placed again and still whole.
    add.evaluate("e => window.scrollBy(0, e.getBoundingClientRect().top - 350)")
    _frames(page)
    assert hint.is_visible()
    _unclipped(hint)


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_setup_full_hint_dismisses_after_an_outside_touch_and_a_return(role_ui, theme):
    """Real browser touch emulation, not native iOS: in onboarding too a tap outside dismisses the limit hint, and
    after leaving the step and returning the held Add shows it again and a tap outside dismisses it again."""
    ui = role_ui
    _configure_full(ui)
    page = ui["new_page"](has_touch=True, is_mobile=True, viewport={"width": 390, "height": 844})
    ui["page"] = page
    page.add_init_script(f"localStorage.setItem('ouroboros.theme', '{theme}')")
    add = _setup_agents(ui)
    _setup_full(page, add)
    hint = page.locator("#onboarding-available-subagents [data-full-hint] .ui-hint").first
    count = page.locator("#onboarding-available-subagents .available-subagents-count").first
    for visit in ("first", "return"):
        add.scroll_into_view_if_needed()
        add.tap(force=True)  # aria-disabled explains the limit but accepts a touch
        hint.wait_for(state="visible")
        assert " ".join(hint.inner_text().split()) == FULL_SETUP
        _unclipped(hint)
        roles.capture(page, f"setup-touch-open-{visit}-{theme}")
        count.tap()  # beside Add, outside its hint host; it changes nothing
        _frames(page)
        assert hint.is_hidden(), f"an outside touch dismisses the hint ({visit})"
        roles.capture(page, f"setup-touch-outside-{visit}-{theme}")
        _setup_full(page, add)
        if visit == "first":  # leave the step for the one before it (the synthetic Main holds Continue) and return
            page.locator("#back-btn").tap()
            page.wait_for_selector("#quick-start-btn:not([hidden])")
            page.locator("#next-btn").tap()
            page.wait_for_selector('[data-model-role="main"]')
            if not page.locator('[data-collapse="subagents"]').evaluate("el => el.open"):
                page.locator('[data-collapse="subagents"] > summary').tap()
            add.wait_for(state="visible")
            assert page.locator("#onboarding-available-subagents .ui-hint:visible").count() == 0
    assert not any(path == "/api/onboarding/complete" for path, _ in ui["posts"])


def test_duplicate_is_an_unsaved_agent_whose_page_names_its_twin_until_the_engine_changes(role_ui):
    ui = role_ui
    _configure(ui, 2, marked=(1,))  # the reviewer is another engine: the pool is never the subject
    form = roles.open_agent(ui, "fast-scout")
    page = ui["page"]
    form.locator("[data-subagent-duplicate]").click()
    copy = _level(page, "agents/new").locator("[data-subagent-row]")
    copy.wait_for(state="visible")
    # Its Back, in the section's header, leads to the twin it was copied from, named by its route.
    back = header_back(page)
    assert back.is_visible() and back.get_attribute("aria-label") == "Back to gpt-test-0"
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

    back.click()
    form = _level(page, "agents/fast-scout")
    form.locator("[data-subagent-duplicate]").wait_for(state="visible")
    assert _roster_posts(ui) == 0
    form.locator("[data-subagent-duplicate]").click()
    copy = _level(page, "agents/new").locator("[data-subagent-row]")

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
    assert level.locator("[data-review-pool-stays]").inner_text() == "Subagents are off; agents marked Reviewer still review."
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
