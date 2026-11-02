"""Browse, Sources and the Skills dialogs through the production page, with no backend.

Every answer is the gateway's skill-source contract as the fixtures in web/tests/fixtures/skills_v4
mirror it; the tests read what the page asks for (method, address, body) and what it says back.
"""

from __future__ import annotations

import copy

import pytest

from tests._skills_page_harness import (
    URL,
    capture,
    fixture,
    no_overflow,
    open_skills,
    skills_browser as _skills_browser,
)

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]
skills_browser = _skills_browser

LISTING = fixture("extensions_v4.json")["skills"]
CATALOG = fixture("skill_catalog.json")
DIGEST = {item["id"]: item["digest"] for item in CATALOG["items"]}


def _sources(*, confirmed=True, extra=()):
    sources = fixture("skill_sources.json")
    for row in sources["sources"]:
        row["confirmed"] = confirmed if row["kind"] == "hub" else True
    sources["sources"].extend(copy.deepcopy(list(extra)))
    return sources


def _row(name):
    return copy.deepcopy(next(skill for skill in LISTING if skill["name"] == name))


def _open(page, *, sources=None, catalog=None, listing=None):
    skills = open_skills(page, extensions=listing if listing is not None else LISTING,
                         sources=sources or _sources(), catalog=catalog or CATALOG)
    page.goto(URL, wait_until="networkidle")
    page.locator('.skills-card[data-skill="weather"]').wait_for()
    return skills


def _tab(page, name):
    page.locator(f'.skills-tab[data-tab="{name}"]').click()


def _card(page, item_id):
    return page.locator(f'[data-browse-board] .marketplace-card[data-item="{item_id}"]').first


@pytest.mark.parametrize("browser_name,width", [("chromium", 1280), ("webkit", 390)])
def test_browse_install_says_its_consequence_and_follows_the_host(skills_browser, browser_name, width):
    browser = getattr(skills_browser, browser_name).launch(headless=True)
    skills = None
    try:
        page = browser.new_page(viewport={"width": width, "height": 900})
        listing = copy.deepcopy(LISTING)
        skills = open_skills(page, extensions=listing, sources=_sources(), catalog=CATALOG)
        skills.answers["/api/extensions"] = lambda _request: {"skills": listing, "live": {}}
        page.goto(URL, wait_until="networkidle")
        _tab(page, "browse")
        # An install that predates the starter skills gets the recommended shelf.
        shelf = page.locator('[data-browse-grid="recommended"]')
        shelf.wait_for()
        assert shelf.locator(".marketplace-card").count() == 2
        assert page.locator('[data-browse-action="install_all"]').inner_text() == "Install all 2"
        duck = _card(page, "ouroboros/duckduckgo")
        assert "Installs without review — you trust Ouroboros" in duck.inner_text()
        scholion = _card(page, "ouroboros-community/scholion")
        assert "Ouroboros reviews it before it runs" in scholion.inner_text()
        capture(page, browser_name, width, "browse-shelf")

        # Trusted: Installing…, then the card follows the installed listing.
        skills.hold = {"/api/skill-catalog/install"}
        duck.locator('[data-browse-action="install"]').click()
        skills.wait_for_request("/api/skill-catalog/install")
        assert "Installing…" in duck.inner_text()
        listing.append({"name": "duckduckgo", "source": "ouroboroshub", "description": "Web search", "version": "1.0.0",
                        "enabled": True, "review_status": "pending", "review_gate": {"executable_review": True, "blocking_reason": "trusted_source"},
                        "content_hash": DIGEST["ouroboros/duckduckgo"], "payload_root": "skills/ouroboroshub/duckduckgo",
                        "grants": {}, "permissions": ["net", "tool"], "category": ["search"],
                        "install": {"source": {"id": "ouroboros", "kind": "hub", "name": "Ouroboros", "part": "official"},
                                    "official": True, "release": {"version": "1.0.0", "digest": DIGEST["ouroboros/duckduckgo"]},
                                    "basis": "trusted", "base_known": True, "modified": False, "turn_on_when_ready": True,
                                    "installed_by": "owner", "installed_at": "2026-10-17T15:00:00Z", "update_available": None,
                                    "withdrawn": None, "left_source": False}})
        skills.release("/api/skill-catalog/install", fixture("skill_install.json")["trusted"])
        page.wait_for_function("document.querySelector('[data-item=\"ouroboros/duckduckgo\"].marketplace-card').textContent.includes('On · v1.0.0')")
        skills.hold.clear()
        assert skills.posted("/api/skill-catalog/install") == [{"id": "ouroboros/duckduckgo", "turn_on_when_ready": True}]

        # Review basis: the card says the review runs, and Skip review is one press away.
        skills.answers["/api/skill-catalog/install"] = {"ok": True, "job_id": "job-3", "name": "scholion", "basis": "review"}
        scholion.locator('[data-browse-action="install"]').click()
        page.wait_for_function("document.querySelector('[data-item=\"ouroboros-community/scholion\"].marketplace-card').textContent.includes('Reviewing · started just now')")
        assert scholion.locator('[data-browse-action="skip"]').is_visible()
        assert "you can leave this page" in scholion.inner_text()

        # A typed refusal of this install: the card says it and that nothing was installed.
        skills.answers["/api/skill-catalog/install"] = (409, {"ok": False, "code": "withdrawn",
            "error": "happy_farm was withdrawn by Ouroboros Community: no longer maintained."})
        farm = _card(page, "ouroboros-community/happy_farm")
        farm.scroll_into_view_if_needed()
        farm.locator('[data-browse-action="install"]').click()
        page.wait_for_function("document.querySelector('[data-item=\"ouroboros-community/happy_farm\"].marketplace-card').textContent.includes('Nothing was installed')")
        assert "Couldn’t install happy_farm from Ouroboros Community" in farm.inner_text()
        assert farm.locator('[data-browse-action="install"]').inner_text() == "Try again"

        # A refusal about what is already installed is said once; the card is not a failed install.
        skills.answers["/api/skill-catalog/install"] = (409, {"ok": False, "code": "name_taken",
            "error": "A skill named pdf_tools is already installed (My skills). Keep it, or replace it.",
            "occupant": {"name": "pdf_tools", "source": "My skills", "locations": ["user_repo"]}})
        pdf = _card(page, "ouroboros/pdf_tools")
        pdf.scroll_into_view_if_needed()
        pdf.locator('[data-browse-action="install"]').click()
        page.locator(".toast", has_text="already installed (My skills)").wait_for()
        assert "Nothing was installed" not in pdf.inner_text()
        assert "Installs without review — you trust Ouroboros" in pdf.inner_text()
        capture(page, browser_name, width, "browse-install-states")
        assert no_overflow(page)

        # Hide keeps the shelf hidden for these skills, across a reload.
        page.locator('[data-browse-action="hide_shelf"]').click()
        assert page.locator('[data-browse-grid="recommended"]').count() == 0
        page.reload(wait_until="networkidle")
        _tab(page, "browse")
        _card(page, "ouroboros/pdf_tools").wait_for()
        assert page.locator('[data-browse-grid="recommended"]').count() == 0
    finally:
        if skills:
            skills.close(browser)


@pytest.mark.parametrize("browser_name", ["webkit"])
def test_the_hub_rows_are_confirmed_once_with_their_suggested_trust(skills_browser, browser_name):
    browser = getattr(skills_browser, browser_name).launch(headless=True)
    skills = None
    try:
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        skills = _open(page, sources=_sources(confirmed=False))
        sources = skills.answers["/api/skill-sources"]

        def change(request):
            source_id = request.url.rsplit("/", 1)[-1]
            row = next(row for row in sources["sources"] if row["id"] == source_id)
            row.update(request.post_data_json)
            return {"ok": True, "sources": sources["sources"], "source": row}

        for source_id in ("ouroboros", "ouroboros-community"):
            skills.answers[f"/api/skill-sources/{source_id}"] = change
        _tab(page, "browse")
        dialog = page.locator(".skills-dialog")
        dialog.wait_for()
        assert "How should skills from the Ouroboros hub install?" in dialog.inner_text()
        assert "Suggested choices apply until you choose." in dialog.inner_text()
        rows = dialog.locator(".skills-first-row")
        assert rows.nth(0).locator('[aria-checked="true"]').inner_text() == "Install without review"
        assert rows.nth(1).locator('[aria-checked="true"]').inner_text() == "Review each skill"
        capture(page, browser_name, 1280, "first-start")
        dialog.locator("[data-dialog-cancel]").last.click()
        assert dialog.count() == 0 and skills.posts == [], "Later saves nothing"

        # Asked once per visit: an install after Later uses the rows' suggested trust without asking again.
        _card(page, "ouroboros/duckduckgo").locator('[data-browse-action="install"]').click()
        skills.wait_for_count(("POST", "/api/skill-catalog/install"), 1)
        assert dialog.count() == 0
        assert skills.posted("/api/skill-catalog/install") == [{"id": "ouroboros/duckduckgo", "turn_on_when_ready": True}]

        # Sources keeps the question open until the owner confirms; the choice is saved row by row.
        _tab(page, "sources")
        tray = page.locator(".skills-sources-card .skills-tray", has_text="Confirm how skills from the Ouroboros hub install")
        tray.locator('[data-source-action="first_start"]').click()
        dialog.wait_for()
        rows.nth(1).locator('[data-trust="trusted"]').click()
        dialog.locator("[data-first-ok]").click()
        page.locator(".toast", has_text="Saved. You can change it any time here.").wait_for()
        assert skills.posted("/api/skill-sources/ouroboros") == [{"trust": "trusted", "confirmed": True}]
        assert skills.posted("/api/skill-sources/ouroboros-community") == [{"trust": "trusted", "confirmed": True}]
        page.wait_for_function("!document.querySelector('.skills-sources-card .skills-trust-value.is-suggested')")
        assert tray.count() == 0
    finally:
        if skills:
            skills.close(browser)


@pytest.mark.parametrize("browser_name,width", [("webkit", 1280), ("webkit", 390)])
def test_one_name_two_sources_compare_and_replace_is_one_install(skills_browser, browser_name, width):
    browser = getattr(skills_browser, browser_name).launch(headless=True)
    skills = None
    try:
        page = browser.new_page(viewport={"width": width, "height": 900})
        skills = _open(page)
        skills.answers["/api/skill-catalog/search"] = fixture("skill_catalog_search.json")
        _tab(page, "browse")
        search = page.locator("[data-browse-search]")
        search.fill("weather")
        search.press("Enter")
        claw = _card(page, "clawhub/steipete/weather")
        claw.wait_for()
        assert page.locator("[data-browse-sort]").input_value() == "best"
        assert "★ 429 · 171.8k downloads · 6 comments" in claw.inner_text()
        assert "You have another “weather” (from Ouroboros)" in claw.inner_text()
        claw.locator('[data-browse-action="compare"]').click()
        dialog = page.locator(".skills-dialog")
        dialog.wait_for()
        assert "Two skills are called “weather”" in dialog.inner_text()
        assert "replaces yours and keeps a backup of it; the new one is reviewed before it runs" in dialog.inner_text()
        assert dialog.locator("[data-dialog-ok]").inner_text() == "Keep mine"
        capture(page, browser_name, width, "compare")
        dialog.locator("[data-dialog-ok]").click()
        assert dialog.count() == 0 and not skills.posts, "keeping yours changes nothing"
        claw.locator('[data-browse-action="compare"]').click()
        dialog.locator('[data-choice="replace"]').click()
        assert dialog.locator("[data-dialog-ok]").inner_text() == "Use this one"
        dialog.locator("[data-dialog-ok]").click()
        skills.wait_for_count(("POST", "/api/skill-catalog/install"), 1)
        assert skills.posted("/api/skill-catalog/install") == [
            {"id": "clawhub/steipete/weather", "turn_on_when_ready": True, "replace": True}]
        assert [path for path, _ in skills.posts] == ["/api/skill-catalog/install"], "no separate uninstall"
        assert no_overflow(page)
    finally:
        if skills:
            skills.close(browser)


@pytest.mark.parametrize("browser_name,width", [("chromium", 1280), ("webkit", 390)])
def test_the_detail_sheet_reads_the_entry_and_returns_focus(skills_browser, browser_name, width):
    browser = getattr(skills_browser, browser_name).launch(headless=True)
    skills = None
    try:
        page = browser.new_page(viewport={"width": width, "height": 844})
        skills = _open(page)
        skills.answers["/api/skill-catalog/item/ouroboros-community/scholion"] = fixture("skill_catalog_item.json")
        skills.answers["/api/skill-catalog/search"] = fixture("skill_catalog_search.json")
        _tab(page, "browse")
        name = _card(page, "ouroboros-community/scholion").locator(".skills-card-name")
        name.scroll_into_view_if_needed()
        name.click()
        sheet = page.locator(".skills-detail-sheet")
        sheet.locator(".skills-sheet-about").wait_for()
        page.wait_for_function("!document.querySelector('.skills-detail-sheet .skills-sheet-status')")
        text = sheet.inner_text()
        assert "Local second-opinion layer" in text
        assert "Your keys: OPENROUTER_API_KEY" in text
        assert "Ouroboros will review it on this computer before it runs. You can skip the review." in text
        assert "★ 4 stars on CrossRead/scholion" in text
        assert "Discussion (4)" in text
        assert ("GET", "/api/skill-catalog/item/ouroboros-community/scholion") in skills.requests
        capture(page, browser_name, width, "detail-sheet")
        page.keyboard.press("Escape")
        assert sheet.count() == 0
        assert name.evaluate("element => element === document.activeElement")

        # A registry hit has no entry to read: the sheet shows what the card had.
        search = page.locator("[data-browse-search]")
        search.fill("weather")
        search.press("Enter")
        claw = _card(page, "clawhub/steipete/weather")
        claw.locator(".skills-card-name").click()
        page.locator(".skills-detail-sheet .skills-sheet-about").wait_for()
        assert not any("/api/skill-catalog/item/clawhub" in path for _, path in skills.requests)
        page.locator("[data-sheet-close]").click()
    finally:
        if skills:
            skills.close(browser)


@pytest.mark.parametrize("browser_name,width", [("chromium", 1280), ("webkit", 390)])
def test_sources_trust_menu_and_the_two_step_add(skills_browser, browser_name, width):
    browser = getattr(skills_browser, browser_name).launch(headless=True)
    skills = None
    try:
        page = browser.new_page(viewport={"width": width, "height": 900})
        sources = _sources()
        skills = _open(page, sources=sources)
        _tab(page, "sources")
        rows = page.locator(".skills-source-row")
        rows.first.wait_for()
        assert rows.count() == 5
        community = page.locator('[data-source-row="ouroboros-community"]')
        assert "couldn’t be read: GitHub didn't answer (timed out after 60 s) — showing the list from" in community.inner_text()
        assert "4 skills by 4 authors" in community.inner_text()
        capture(page, browser_name, width, "sources")

        # Trust is a setting: the dialog names the consequence and changes nothing already installed.
        community.locator('[data-source-action="trust"]').click()
        dialog = page.locator(".skills-dialog")
        dialog.wait_for()
        assert "keep running as they are" in dialog.inner_text()
        assert dialog.locator('[data-choice="review"]').get_attribute("aria-checked") == "true"
        dialog.locator('[data-choice="trusted"]').click()
        assert dialog.locator("[data-dialog-ok]").inner_text() == "Trust Ouroboros Community"
        dialog.locator("[data-dialog-ok]").click()
        page.locator(".toast", has_text="Skills from Ouroboros Community now install without review.").wait_for()
        assert skills.posted("/api/skill-sources/ouroboros-community") == [{"trust": "trusted", "confirmed": True}]

        # ⋮ Refresh, Rename, Remove; the built-in source has no Remove.
        page.locator('[data-source-menu="builtin"]').click()
        menu = page.locator("body > .skills-card-menu-dialog:not([data-leaving])")
        assert [item.strip() for item in menu.locator("[data-menu-action]").all_inner_texts()] == ["Refresh", "Rename…"]
        page.keyboard.press("Escape")
        folder = page.locator('[data-source-menu="my-folder"]')
        folder.click()
        menu.locator('[data-menu-action="refresh"]').click()
        skills.wait_for_count(("POST", "/api/skill-sources/my-folder/refresh"), 1)
        folder.click()
        menu.locator('[data-menu-action="rename"]').click()
        confirm = page.locator(".confirm-dialog")
        confirm.locator("input").fill("Work skills")
        confirm.locator("[data-confirm-ok]").click()
        skills.wait_for_count(("POST", "/api/skill-sources/my-folder"), 1)
        assert skills.posted("/api/skill-sources/my-folder") == [{"name": "Work skills"}]
        folder.click()
        menu.locator('[data-menu-action="remove"]').click()
        assert "Remove My skills?" in confirm.inner_text()
        confirm.locator("[data-confirm-ok]").click()
        skills.wait_for_count(("POST", "/api/skill-sources/my-folder/remove"), 1)

        # Add a source: an unreadable address stays on step 1 and saves nothing.
        skills.answers["/api/skill-sources/probe"] = lambda request: (
            fixture("skill_source_probe.json")["unreadable"] if "missing" in request.post_data_json.get("url", "")
            else fixture("skill_source_probe.json")["ok"])
        page.locator('[data-source-action="add"]').click()
        address = page.locator("[data-source-address]")
        address.fill("https://github.com/someone/missing")
        address.press("Enter")
        page.locator(".skills-dialog-error").wait_for()
        assert "could not read https://github.com/someone/missing" in page.locator(".skills-dialog-error").inner_text()
        assert skills.posted("/api/skill-sources") == []
        page.locator("[data-source-address]").fill("https://github.com/kuznetsoffandrey/ru-skills")
        page.locator("[data-add-check]").click()
        name = page.locator("[data-source-name]")
        name.wait_for()
        assert name.input_value() == "Russian services"
        step_two = page.locator(".skills-dialog").inner_text()
        assert "Found 7 skills" in step_two and "2gis, vkusvill" in step_two
        assert "A review sends the skill’s files to your review models." in step_two
        assert "While automatic keys are on, they get the keys they ask for." in step_two
        assert page.locator('.skills-dialog [data-choice="review"]').get_attribute("aria-checked") == "true"
        capture(page, browser_name, width, "add-source-step-2")
        page.locator("[data-add-ok]").click()
        skills.wait_for_count(("POST", "/api/skill-sources"), 1)
        assert skills.posted("/api/skill-sources") == [
            {"url": "https://github.com/kuznetsoffandrey/ru-skills", "name": "Russian services", "trust": "review"}]
        assert skills.posted("/api/skill-sources/probe") == [
            {"url": "https://github.com/someone/missing"}, {"url": "https://github.com/kuznetsoffandrey/ru-skills"}]
        assert no_overflow(page)
    finally:
        if skills:
            skills.close(browser)


@pytest.mark.parametrize("browser_name,width", [("chromium", 1280), ("webkit", 390)])
def test_update_asks_what_to_keep_and_skip_review_names_the_author(skills_browser, browser_name, width):
    browser = getattr(skills_browser, browser_name).launch(headless=True)
    skills = None
    try:
        page = browser.new_page(viewport={"width": width, "height": 900})
        skills = _open(page)
        anime = _row("anime_studio")
        update = page.locator('.skills-card[data-skill="anime_studio"] [data-skill-action="update"]')
        update.scroll_into_view_if_needed()
        update.click()
        dialog = page.locator(".skills-dialog")
        dialog.wait_for()
        text = dialog.inner_text()
        assert "Update anime_studio to v3.2.0?" in text
        assert "You changed anime_studio on this computer: its files differ from v2.12.0 you installed (your copy says v3.3.0)." in text
        assert dialog.locator('[data-choice="merge"]').get_attribute("aria-checked") == "true"
        assert "v3.2.0 isn’t offered again; a newer version will be." in text
        capture(page, browser_name, width, "update-dialog")
        page.keyboard.press("Escape")
        assert dialog.count() == 0 and skills.posts == [], "Escape updates nothing"
        update.click()
        dialog.locator('[data-choice="keep"]').click()
        assert dialog.locator("[data-dialog-ok]").inner_text() == "Keep my version"
        skills.answers["/api/skills/anime_studio/update"] = {"ok": True, "name": "anime_studio", "mode": "keep",
                                                             "skipped": {"version": "3.2.0", "digest": anime["install"]["update_available"]["digest"]}}
        dialog.locator("[data-dialog-ok]").click()
        skills.wait_for_count(("POST", "/api/skills/anime_studio/update"), 1)
        assert skills.posted("/api/skills/anime_studio/update") == [{
            "mode": "keep", "digest": anime["install"]["update_available"]["digest"], "local_digest": anime["content_hash"]}]

        # Unmodified, from a trusted row: one press replaces, with the digests the card was read with.
        weather = _row("weather")
        page.locator('.skills-card[data-skill="weather"] [data-skill-action="update"]').click()
        skills.wait_for_count(("POST", "/api/skills/weather/update"), 1)
        assert skills.posted("/api/skills/weather/update") == [{
            "mode": "replace", "digest": weather["install"]["update_available"]["digest"], "local_digest": weather["content_hash"]}]

        # A refusal of a stale dialog is said with the host's words.
        skills.answers["/api/skills/weather/update"] = (409, {"ok": False, "code": "candidate_changed",
            "error": "Ouroboros now offers a different release of weather; look at it again.",
            "candidate": {"version": "0.3.3", "digest": "f" * 64}})
        page.locator('.skills-card[data-skill="weather"] [data-skill-action="update"]').click()
        page.locator(".toast", has_text="now offers a different release of weather").wait_for()

        # Skip review: the source and its author are named before the owner attests these bytes.
        whoop = _row("whoop-health")
        skip = page.locator('.skills-card[data-skill="whoop-health"] [data-skill-action="attest_review"]').first
        skip.scroll_into_view_if_needed()
        skip.click()
        confirm = page.locator(".confirm-dialog")
        confirm.wait_for()
        assert "Skip review for whoop-health?" in confirm.inner_text()
        assert "It comes from Ouroboros Community, by kuznetsoffandrey." in confirm.inner_text()
        capture(page, browser_name, width, "skip-review")
        confirm.locator("[data-confirm-ok]").click()
        skills.wait_for_count(("POST", "/api/owner/skills/whoop-health/attest-review"), 1)
        assert skills.posted("/api/owner/skills/whoop-health/attest-review") == [{"expected_content_hash": whoop["content_hash"]}]

        # Review found problems: the same attestation is «Turn on anyway».
        blocked = page.locator('.skills-card[data-skill="minecraft-widget"] [data-skill-action="attest_review"]').first
        blocked.scroll_into_view_if_needed()
        blocked.click()
        confirm.wait_for()
        assert "Turn on minecraft-widget anyway?" in confirm.inner_text()
        assert confirm.locator("[data-confirm-ok]").inner_text() == "Turn on anyway"
        confirm.locator("[data-confirm-cancel]").last.click()
        assert skills.posted("/api/owner/skills/minecraft-widget/attest-review") == []
        assert no_overflow(page)
    finally:
        if skills:
            skills.close(browser)


@pytest.mark.parametrize("browser_name", ["webkit"])
def test_publish_destination_reaches_the_preflight_and_the_task(skills_browser, browser_name):
    browser = getattr(skills_browser, browser_name).launch(headless=True)
    skills = None
    try:
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        listing = copy.deepcopy(LISTING)
        anime = next(row for row in listing if row["name"] == "anime_studio")
        anime["submit_hub"] = {"visible": True, "task_start_allowed": True, "publication_ready": True, "state": "ready"}
        mine = {"id": "mine", "name": "My skills repo", "kind": "git", "url": "https://github.com/me/skills",
                "trust": "trusted", "confirmed": True, "state": "ready", "last_read_at": "2026-10-17T15:00:00Z"}
        skills = _open(page, listing=listing, sources=_sources(extra=[mine]))
        skills.answers["/api/skills/anime_studio/publish-preflight"] = {
            "ok": True, "skill": "anime_studio", "repository": "razzant/OuroborosHub", "state": "ready",
            "publication_ready": True, "task_start_allowed": True}
        skills.answers["/api/tasks"] = {"ok": True, "task_id": "t-publish"}
        card = page.locator('.skills-card[data-skill="anime_studio"]')
        card.scroll_into_view_if_needed()
        card.locator("[data-skill-menu-trigger]").click()
        page.locator("body > .skills-card-menu-dialog:not([data-leaving]) .skills-submit-hub").click()
        dialog = page.locator(".skills-dialog")
        dialog.wait_for()
        labels = [label.strip() for label in dialog.locator(".chat-quiz-option-label").all_inner_texts()]
        assert labels == ["Back to Ouroboros Community — where it came from", "My skills repo — your own source",
                          "Ouroboros Community — as a new skill"]
        assert "Where should your changed version go?" in dialog.inner_text()
        capture(page, browser_name, 1280, "publish-destination")
        dialog.locator('[data-choice="new:ouroboros-community"]').click()
        dialog.locator("[data-dialog-ok]").click()
        confirm = page.locator(".confirm-dialog")
        confirm.wait_for()
        confirm.locator("[data-confirm-ok]").click()
        skills.wait_for_count(("POST", "/api/tasks"), 1)
        destination = {"source_id": "ouroboros-community", "mode": "new_entry"}
        assert skills.posted("/api/skills/anime_studio/publish-preflight") == [{"destination": destination}]
        task = skills.posted("/api/tasks")[0]
        assert task["type"] == "skill_publish"
        assert task["metadata"]["skill_publish_target"] == {
            "skill": "anime_studio", "repository": "razzant/OuroborosHub", "destination": destination}
    finally:
        if skills:
            skills.close(browser)
