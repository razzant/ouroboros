"""Browse, Sources and the Skills dialogs through the production page, with no backend.

Every answer is the gateway's skill-source contract as the fixtures in web/tests/fixtures/skills_v4
mirror it; the tests read what the page asks for (method, address, body) and what it says back.
"""

from __future__ import annotations

import copy
import re
from urllib.parse import parse_qs, urlsplit

import pytest

from tests._skills_page_harness import (
    open_view,
    URL,
    capture,
    fixture,
    no_overflow,
    open_skills,
    skills_browser as _skills_browser,
)

pytestmark = [pytest.mark.ui_browser]
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


def _catalog(sources=None, catalog=None):
    """The catalog as the host derives it from the same rows as the list of sources: its row of a source says whether
    the owner confirmed its trust, and that row and every entry of the source carry the trust itself."""
    catalog = copy.deepcopy(catalog or CATALOG)
    rows = {row["id"]: row for row in (sources or _sources())["sources"]}
    for status in catalog["sources"]:
        if status["id"] in rows:
            status.update(trust=rows[status["id"]]["trust"], confirmed=rows[status["id"]]["confirmed"])
    for item in catalog["items"]:
        if item["source_id"] in rows:
            item["trust"] = rows[item["source_id"]]["trust"]
    return catalog


def _row(name):
    return copy.deepcopy(next(skill for skill in LISTING if skill["name"] == name))


def _open(page, *, sources=None, catalog=None, listing=None):
    sources = sources or _sources()
    skills = open_skills(page, extensions=listing if listing is not None else LISTING,
                         sources=sources, catalog=_catalog(sources, catalog))
    page.goto(URL, wait_until="networkidle")
    page.locator('.skills-card[data-skill="weather"]').wait_for()
    return skills


def _tab(page, name):
    open_view(page, name)


def _card(page, item_id):
    return page.locator(f'[data-browse-board] .marketplace-card[data-item="{item_id}"]').first


@pytest.mark.parametrize("browser_name", ["chromium", "webkit"])
def test_addressed_installed_card_opens_through_search_and_filter(skills_browser, browser_name):
    from playwright.sync_api import expect

    browser = getattr(skills_browser, browser_name).launch(headless=True)
    page = browser.new_page(viewport={"width": 1280, "height": 900})
    listing = [_row("telegram"), _row("weather"), {"name": "calendar", "enabled": False}]
    listing[0]["secret_config"] = {"fields": [{"id": "example", "key": "EXAMPLE", "state": "missing"}]}
    skills = _open(page, listing=listing)
    try:
        off = page.locator('[data-installed-filter="off"]')
        search = page.locator("#skills-installed-search")
        off.click()
        search.fill("other words")
        _tab(page, "browse")
        _card(page, "builtin/telegram").locator('[data-browse-action="installed"]').click()
        expect(page.locator('.skills-card[data-skill="telegram"]')).to_be_in_viewport()

        off.click()
        search.fill("still filtering")
        _tab(page, "browse")
        _tab(page, "installed")
        expect(off).to_have_attribute("aria-pressed", "true")
        expect(search).to_have_value("still filtering")
        _tab(page, "browse")
        skills.hold.add("/api/extensions")
        page.evaluate("""() => window.dispatchEvent(new CustomEvent('ouro:open-skills', { detail: { skill: 'notes' } }))""")
        skills.wait_for_request("/api/extensions")
        skills.release("/api/extensions", {"skills": [*listing, {"name": "notes", "enabled": True}], "live": {}})
        expect(page.locator('.skills-card[data-skill="notes"]')).to_be_in_viewport()
    finally:
        skills.close(browser)


@pytest.mark.parametrize("width", [390, 1280])
def test_skills_count_follows_the_visible_extensions_list(skills_browser, width):
    from playwright.sync_api import expect

    browser = skills_browser.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": width, "height": 900})
    skills = open_skills(page, extensions=[{"name": "notes", "enabled": False}])
    try:
        skills.answers["/api/ui/contributions"] = {}
        page.goto(URL + "?host_modules", wait_until="networkidle")
        _tab(page, "tools")
        answer = page.locator('[data-extensions-open="installed"] [data-extensions-answer]')
        rows = [{"name": name, "enabled": False} for name in ("notes", "calendar")]
        skills.set_skills(rows)
        reads = skills.requests.count(("GET", "/api/extensions"))
        page.evaluate("""() => window.dispatchEvent(new CustomEvent('ouro:skill-lifecycle', { detail: { action: 'install', name: 'calendar' } }))""")
        if width == 390:
            expect(answer).to_be_hidden()
            page.wait_for_timeout(450)  # beyond the coalesced lifecycle notification
            assert skills.requests.count(("GET", "/api/extensions")) == reads
            page.evaluate("() => window.skillsView.pressedAgain()")
        expect(answer).to_contain_text("2 installed")
        assert skills.requests.count(("GET", "/api/extensions")) == reads + 1
        visibility = """hidden => {
            Object.defineProperty(document, 'hidden', {configurable: true, value: hidden});
            document.dispatchEvent(new Event('visibilitychange'));
        }"""
        page.evaluate(visibility, True)
        skills.set_skills([*rows, {"name": "index", "enabled": False}])
        page.evaluate(visibility, False)
        expect(answer).to_contain_text("3 installed")
    finally:
        skills.close(browser)


@pytest.mark.parametrize("browser_name,width", [("chromium", 1280), ("webkit", 1280), ("webkit", 390)])
def test_browse_install_says_its_consequence_and_follows_the_host(skills_browser, browser_name, width):
    browser = getattr(skills_browser, browser_name).launch(headless=True)
    skills = None
    try:
        page = browser.new_page(viewport={"width": width, "height": 900})
        listing = copy.deepcopy(LISTING)
        skills = open_skills(page, extensions=listing, sources=_sources(), catalog=_catalog())
        skills.answers["/api/extensions"] = lambda _request: {"skills": listing, "live": {}}
        page.goto(URL, wait_until="networkidle")
        _tab(page, "browse")
        # An install that predates the starter skills gets the recommended shelf.
        shelf = page.locator('[data-browse-grid="recommended"]')
        shelf.wait_for()
        assert shelf.locator(".marketplace-card").count() == 2
        assert page.locator('[data-browse-action="install_all"]').inner_text() == "Install all 2"
        duck = _card(page, "ouroboros/duckduckgo")
        assert "Installs without review while you trust Ouroboros" in duck.inner_text()
        scholion = _card(page, "ouroboros-community/scholion")
        assert "Reviewed before it runs while that is the setting for Ouroboros Community" in scholion.inner_text()
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
        assert skills.posted("/api/skill-catalog/install") == [
            {"id": "ouroboros/duckduckgo", "turn_on_when_ready": True,
             "expected_digest": DIGEST["ouroboros/duckduckgo"]}]

        # Review basis: the card says the review runs, and Skip review is one press away.
        skills.answers["/api/skill-catalog/install"] = {"ok": True, "job_id": "job-3", "name": "scholion", "basis": "review"}
        scholion.locator('[data-browse-action="install"]').click()
        page.wait_for_function("document.querySelector('[data-item=\"ouroboros-community/scholion\"].marketplace-card').textContent.includes('Reviewing · started just now')")
        assert scholion.locator('[data-browse-action="skip"]').is_visible()
        assert "you can leave this page" in scholion.inner_text()
        if width == 1280:
            # A short result must not reserve the long-message measure and push
            # a small action below it when both fit in the actual card.
            short_row = scholion.locator(".settings-action-row")
            text_box = short_row.locator(":scope > :first-child").bounding_box()
            action_box = short_row.locator(".skills-card-actions-end").bounding_box()
            short_row.evaluate("e => e.scrollIntoView({block: 'center'})")
            capture(page, browser_name, width, "browse-short-status")
            assert action_box["y"] < text_box["y"] + text_box["height"]
            assert short_row.evaluate("e => e.scrollWidth <= e.clientWidth + 1")

        # A typed refusal of this install: the card says it and that nothing was installed.
        skills.answers["/api/skill-catalog/install"] = (409, {"ok": False, "code": "withdrawn",
            "error": "happy_farm was withdrawn by Ouroboros Community: no longer maintained."})
        farm = _card(page, "ouroboros-community/happy_farm")
        farm.scroll_into_view_if_needed()
        farm.locator('[data-browse-action="install"]').click()
        page.wait_for_function("document.querySelector('[data-item=\"ouroboros-community/happy_farm\"].marketplace-card').textContent.includes('Nothing was installed')")
        assert "Couldn’t install happy_farm from Ouroboros Community" in farm.inner_text()
        assert farm.locator('[data-browse-action="install"]').inner_text() == "Try again"
        result_row = farm.locator(".settings-action-row")
        result_row.evaluate("e => { e.style.width = '280px'; e.style.maxWidth = '100%'; }")
        text_box = result_row.locator(":scope > :first-child").bounding_box()
        action_box = result_row.locator(".skills-card-actions-end").bounding_box()
        assert text_box["width"] >= result_row.bounding_box()["width"] - 2
        assert action_box["y"] >= text_box["y"] + text_box["height"] - 1
        assert result_row.evaluate("e => e.scrollWidth <= e.clientWidth + 1")
        result_row.evaluate("e => e.scrollIntoView({block: 'center'})")
        capture(page, browser_name, width, "browse-failure-container-reflow")

        # A refusal about what is already installed is said once; the card is not a failed install.
        skills.answers["/api/skill-catalog/install"] = (409, {"ok": False, "code": "name_taken",
            "error": "A skill named pdf_tools is already installed (My skills). Keep it, or replace it.",
            "occupant": {"name": "pdf_tools", "source": "My skills", "locations": ["user_repo"]}})
        pdf = _card(page, "ouroboros/pdf_tools")
        pdf.scroll_into_view_if_needed()
        pdf.locator('[data-browse-action="install"]').click()
        page.locator(".toast", has_text="already installed (My skills)").wait_for()
        assert "Nothing was installed" not in pdf.inner_text()
        assert "Installs without review while you trust Ouroboros" in pdf.inner_text()
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


@pytest.mark.parametrize("browser_name", ["chromium", "webkit"])
def test_withdrawn_browse_result_uses_the_row_without_an_empty_action(skills_browser, browser_name):
    from playwright.sync_api import expect

    browser = getattr(skills_browser, browser_name).launch(headless=True)
    skills = None
    try:
        page = browser.new_page(viewport={"width": 1100, "height": 750})
        catalog = copy.deepcopy(CATALOG)
        item = next(item for item in catalog["items"] if item["id"] == "ouroboros-community/happy_farm")
        item["withdrawn"] = {"reason": "No longer maintained"}
        skills = _open(page, catalog=catalog)
        _tab(page, "browse")
        row = _card(page, item["id"]).locator(".settings-action-row")
        expect(row).to_contain_text("Withdrawn by")
        row.evaluate("e => { e.style.width = '280px'; e.style.maxWidth = '100%'; }")
        expect(row.locator(".skills-card-actions-end")).to_be_empty()
        expect(row.locator(".skills-card-actions-end")).not_to_be_visible()
        assert row.locator(":scope > :first-child").bounding_box()["width"] >= row.bounding_box()["width"] - 2
        assert row.evaluate("e => e.scrollWidth <= e.clientWidth + 1")
        row.evaluate("e => e.scrollIntoView({block: 'center'})")
        capture(page, browser_name, 1100, "browse-no-action")
    finally:
        if skills:
            skills.close(browser)


@pytest.mark.parametrize("browser_name", ["webkit"])
def test_the_hub_rows_are_confirmed_once_with_their_suggested_trust(skills_browser, browser_name):
    from playwright.sync_api import expect

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
        skills.answers["/api/skill-catalog"] = lambda _request: _catalog(sources)  # the host reads both from one list
        dialog = page.locator(".skills-dialog")
        skills.hold.add("/api/skill-catalog")
        _tab(page, "sources")
        skills.wait_for_request("/api/skill-catalog")
        _tab(page, "installed")
        skills.hold.clear()
        skills.release("/api/skill-catalog")
        page.locator('[data-source-action="first_start"]').wait_for(state="attached")
        assert dialog.count() == 0 and skills.posts == []
        _tab(page, "browse")
        dialog.wait_for()
        assert "How should skills from the Ouroboros hub install?" in dialog.inner_text()
        assert "Suggested choices apply until you choose." in dialog.inner_text()
        rows = dialog.locator(".skills-first-row")
        assert rows.nth(0).locator('[aria-checked="true"]').inner_text() == "Install without review"
        assert rows.nth(1).locator('[aria-checked="true"]').inner_text() == "Review each skill"
        capture(page, browser_name, 1280, "first-start")
        dialog.locator("[data-dialog-cancel]").last.click()
        assert dialog.count() == 0 and skills.posts == [], "Later saves nothing"
        # Until the owner chooses, a card says the setting the host installs by, as the suggested one.
        assert ("Installs without review while that is the suggested setting for Ouroboros"
                in _card(page, "ouroboros/duckduckgo").inner_text())
        assert ("Reviewed before it runs while that is the suggested setting for Ouroboros Community"
                in _card(page, "ouroboros-community/scholion").inner_text())

        # Asked once per visit: an install after Later uses the rows' suggested trust without asking again.
        _card(page, "ouroboros/duckduckgo").locator('[data-browse-action="install"]').click()
        skills.wait_for_count(("POST", "/api/skill-catalog/install"), 1)
        assert dialog.count() == 0
        assert skills.posted("/api/skill-catalog/install") == [
            {"id": "ouroboros/duckduckgo", "turn_on_when_ready": True,
             "expected_digest": DIGEST["ouroboros/duckduckgo"]}]

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
        # Browse reads its catalog again on arrival: its cards say the confirmed choice.
        _tab(page, "browse")
        expect(_card(page, "ouroboros-community/scholion")).to_contain_text(
            "Installs without review while you trust Ouroboros Community")
    finally:
        if skills:
            skills.close(browser)


@pytest.mark.parametrize("browser_name", ["chromium", "webkit"])
def test_browse_words_hold_when_the_source_list_answers_late_with_another_setting(skills_browser, browser_name):
    """A card and an open detail say how their entry installs only while its source's setting holds: the host reads
    the source again when Install is pressed. A list of sources that answers late, with the same source set
    otherwise, therefore makes nothing on screen false, and repaints and refocuses nothing."""
    from playwright.sync_api import expect

    browser = getattr(skills_browser, browser_name).launch(headless=True)
    skills = None
    try:
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        skills = _open(page)
        skills.answers["/api/skill-catalog/item/ouroboros-community/scholion"] = fixture("skill_catalog_item.json")
        # Showing the place reads its sources again; here that answer is still on its way while Browse is used.
        skills.hold.add("/api/skill-sources")
        _tab(page, "browse")
        skills.wait_for_request("/api/skill-sources")
        card = _card(page, "ouroboros-community/scholion")
        said = "Reviewed before it runs while that is the setting for Ouroboros Community"
        expect(card).to_contain_text(said)
        card.locator(".skills-card-name").click()
        sheet = page.locator(".skills-detail-sheet")
        sheet.locator(".skills-sheet-about").wait_for()
        page.wait_for_function("!document.querySelector('.skills-detail-sheet .skills-sheet-status')")
        detail = ("Ouroboros reviews it on this computer before it runs while that is the setting for"
                  " the Ouroboros Community source. You can skip the review.")
        assert detail in sheet.inner_text()
        install = sheet.locator('[data-sheet-action="install"]')
        install.focus()
        # The late list: the same source, now set to install without review, and one source fewer (the rail's count
        # says the list arrived). An Install pressed now would go without review; no sentence here says otherwise.
        late = _sources()
        late["sources"] = [row for row in late["sources"] if row["id"] != "my-folder"]
        next(row for row in late["sources"] if row["id"] == "ouroboros-community")["trust"] = "trusted"
        skills.hold.clear()
        skills.release("/api/skill-sources", late)
        expect(page.locator('[data-extensions-open="sources"] [data-extensions-answer]')).to_contain_text("4 connected")
        assert said in card.inner_text(), "the shown entry keeps its own conditional sentence"
        assert detail in sheet.inner_text()
        assert install.evaluate("element => element === document.activeElement"), "the open detail keeps the keyboard"
        page.keyboard.press("Escape")
        assert sheet.count() == 0
    finally:
        if skills:
            skills.close(browser)


@pytest.mark.parametrize("browser_name,width", [("webkit", 1280), ("webkit", 390)])
def test_a_namespaced_registry_offer_installs_beside_a_same_leaf_skill(skills_browser, browser_name, width):
    from playwright.sync_api import expect

    browser = getattr(skills_browser, browser_name).launch(headless=True)
    skills = None
    try:
        page = browser.new_page(viewport={"width": width, "height": 900})
        skills = _open(page)
        search_result = fixture("skill_catalog_search.json")
        offer = next(item for item in search_result["items"] if item["id"] == "clawhub/steipete/weather")
        registry_key = "registry:" + "b" * 32
        offer["source_key"] = registry_key
        skills.answers["/api/skill-catalog/search"] = search_result
        _tab(page, "browse")
        search = page.locator("[data-browse-search]")
        search.fill("weather")
        search.press("Enter")
        claw = _card(page, "clawhub/steipete/weather")
        claw.wait_for()
        assert page.locator("[data-browse-sort]").input_value() == "best"
        assert "★ 429 · 171.8k downloads · 6 comments" in claw.inner_text()
        # A registry leaf is not a local install identity: it can neither Compare with nor replace the hub's weather.
        assert "You have another" not in claw.inner_text()
        assert claw.locator('[data-browse-action="compare"]').count() == 0
        def install_registry(_request):
            # The queue accepts the full slug before the adapter names the installed folder.
            landed_digest = "b" * 64
            skills.set_skills([*LISTING, {
                "name": "steipete__weather", "source": "clawhub", "registry_slug": "steipete/weather",
                "source_key": registry_key, "version": "1.0.4", "enabled": True, "review_status": "clean",
                "executable_review": True, "content_hash": landed_digest, "manifest_file": "SKILL.md",
                "grants": {"all_granted": True},
                "install": {"source": {"id": "clawhub", "kind": "registry", "name": "ClawHub"},
                            "release": {"version": "1.0.4", "digest": landed_digest},
                            "basis": "review", "modified": False},
            }])
            return {"ok": True, "job_id": "job-w", "name": "steipete/weather", "basis": "review"}

        skills.answers["/api/skill-catalog/install"] = install_registry
        claw.locator('[data-browse-action="install"]').click()
        skills.wait_for_count(("POST", "/api/skill-catalog/install"), 1)
        # Ordinary registry install: pinned to the shown exact version, no digest supplied, no overwrite, no uninstall.
        assert skills.posted("/api/skill-catalog/install") == [
            {"id": "clawhub/steipete/weather", "turn_on_when_ready": True, "version": "1.0.4",
             "expected_source_key": registry_key}]
        assert [path for path, _ in skills.posts] == ["/api/skill-catalog/install"], "no separate uninstall"
        expect(claw).to_contain_text("On · v1.0.4")
        _tab(page, "installed")
        expect(page.locator('.skills-card[data-skill="steipete__weather"]')).to_be_visible()
        expect(page.locator('.skills-card[data-skill="weather"]')).to_be_visible()
    finally:
        if skills:
            skills.close(browser)


@pytest.mark.parametrize("browser_name,width", [("webkit", 1280), ("webkit", 390)])
def test_one_name_two_sources_compare_and_replace_is_one_install(skills_browser, browser_name, width):
    browser = getattr(skills_browser, browser_name).launch(headless=True)
    skills = None
    try:
        page = browser.new_page(viewport={"width": width, "height": 900})
        # Compare/Replace stays with a non-registry offer against a same-named occupant: a git source's "weather".
        git_url = "https://example.com/my-git"
        git_digest = "d" * 64
        git_key = "git:" + "d" * 32
        sources = _sources(extra=[{"id": "my-git", "name": "My git", "kind": "git", "url": git_url,
                                   "ref": "main", "trust": "review", "confirmed": True, "state": "ready"}])
        catalog = copy.deepcopy(CATALOG)
        catalog["sources"].append({"id": "my-git", "name": "My git", "state": "ready"})
        git_weather = {"id": "my-git/weather", "source_id": "my-git", "source_name": "My git", "name": "weather",
                       "description": "Weather from my own repository.", "category": [], "version": "0.1.0",
                       "digest": git_digest, "source_key": git_key, "official": False, "recommended": False, "owner": "me",
                       "updated_at": "2026-11-01T10:00:00+00:00", "signals": None, "withdrawn": None,
                       "type": "script", "needs_keys": [], "permissions": ["net"], "installed": None,
                       "trust": "review", "repo_url": git_url}
        catalog["items"].append(git_weather)
        skills = _open(page, sources=sources, catalog=catalog)
        skills.answers["/api/skill-catalog/install"] = {"ok": True, "job_id": "job-git", "name": "weather", "basis": "review"}
        _tab(page, "browse")
        git_card = _card(page, "my-git/weather")
        git_card.wait_for()
        assert "You have another “weather” (from Ouroboros)" in git_card.inner_text()
        git_card.locator('[data-browse-action="compare"]').click()
        dialog = page.locator(".skills-dialog")
        dialog.wait_for()
        assert "Two skills are called “weather”" in dialog.inner_text()
        assert ("replaces yours and keeps a backup of it; its source’s setting in Sources decides whether it is reviewed first"
                in dialog.inner_text())
        assert dialog.locator("[data-dialog-ok]").inner_text() == "Keep mine"
        capture(page, browser_name, width, "compare")
        dialog.locator("[data-dialog-ok]").click()
        assert dialog.count() == 0 and not skills.posts, "keeping yours changes nothing"
        git_card.locator('[data-browse-action="compare"]').click()
        dialog.locator('[data-choice="replace"]').click()
        assert dialog.locator("[data-dialog-ok]").inner_text() == "Use this one"
        dialog.locator("[data-dialog-ok]").click()
        skills.wait_for_count(("POST", "/api/skill-catalog/install"), 1)
        assert skills.posted("/api/skill-catalog/install") == [
            {"id": "my-git/weather", "turn_on_when_ready": True, "replace": True,
             "expected_digest": git_digest, "expected_source_key": git_key}]
        assert [path for path, _ in skills.posts] == ["/api/skill-catalog/install"], "no separate uninstall"
        assert no_overflow(page)
    finally:
        if skills:
            skills.close(browser)


def _claw_item(slug, downloads, *, updated="2026-11-01T10:00:00+00:00"):
    return {"id": f"clawhub/{slug}", "source_id": "clawhub", "source_name": "ClawHub", "trust": "review",
            "name": slug.replace("-", "_"), "registry_slug": slug, "description": f"{slug} helps with one job.",
            "category": [], "version": "1.0.0", "digest": "", "official": False, "recommended": False, "signals": None,
            "withdrawn": None, "type": "", "needs_keys": [], "permissions": [], "installed": None,
            "registry_stats": {"downloads": downloads, "stars": downloads // 100}, "owner": "someone", "updated_at": updated}


class _ClawHub:
    """The search route as the host answers it: pages by cursor, popular or updated."""

    def __init__(self, skills):
        self.skills, self.queries = skills, []
        skills.answers["/api/skill-catalog/search"] = self.answer

    def answer(self, request):
        query = parse_qs(urlsplit(request.url).query)
        self.queries.append({key: values[0] for key, values in query.items()})
        if query.get("q"):
            return {"items": [_claw_item("typed-hit", 5)]}
        if query.get("sort") == ["updated"]:
            return {"items": [_claw_item("just-uploaded", 0)]}
        if query.get("cursor"):
            return {"items": [_claw_item("third", 300), _claw_item("second", 400)]}
        return {"items": [_claw_item("first", 491000), _claw_item("second", 400)], "next_cursor": "skillcat:{\"offset\":2}"}


def _board_names(page):
    return page.locator("[data-browse-board] .marketplace-card").evaluate_all("cards => cards.map(card => card.dataset.item)")


def _choose_clawhub(page):
    _tab(page, "browse")
    page.locator('[data-browse-source] option[value="clawhub"]').wait_for(state="attached")
    assert page.locator('[data-browse-source] option[value="clawhub"]').inner_text() == "ClawHub"
    page.locator("[data-browse-source]").select_option("clawhub")


@pytest.mark.parametrize("browser_name,width", [("chromium", 1280), ("webkit", 1280), ("chromium", 390)])
def test_clawhub_chosen_alone_shows_its_popular_page_and_pages_on(skills_browser, browser_name, width):
    browser = getattr(skills_browser, browser_name).launch(headless=True)
    skills = None
    try:
        page = browser.new_page(viewport={"width": width, "height": 700})
        skills = _open(page)
        claw = _ClawHub(skills)
        _tab(page, "browse")
        _card(page, "ouroboros/weather").wait_for()
        assert claw.queries == [], "All sources never asks the registry when Browse opens"
        all_status = page.locator("[data-browse-status]").inner_text()
        assert "ClawHub joins the list when you search" in all_status
        capture(page, browser_name, width, "clawhub-all-sources")

        skills.hold = {"/api/skill-catalog/search"}
        _choose_clawhub(page)
        skills.wait_for_request("/api/skill-catalog/search")
        assert "Asking ClawHub for its most downloaded skills" in page.locator("[data-browse-status]").inner_text()
        assert "Nothing to show" not in page.locator("[data-browse-board]").inner_text()
        skills.hold.clear()
        skills.release("/api/skill-catalog/search")
        _card(page, "clawhub/first").wait_for()
        assert claw.queries == [{"source": "clawhub", "sort": "popular"}]
        assert _board_names(page) == ["clawhub/first", "clawhub/second"], "the registry's order, never re-sorted here"
        status = page.locator("[data-browse-status]").inner_text()
        assert "2 skills from ClawHub, most downloaded first" in status and "search all of ClawHub" in status
        assert "491.0k downloads" in _card(page, "clawhub/first").inner_text()
        assert "Reviewed before it runs while that is the setting for ClawHub" in _card(page, "clawhub/first").inner_text()
        sort = page.locator("[data-browse-sort]")
        assert sort.locator("option").all_inner_texts() == ["Popular", "Recently updated"] and sort.input_value() == "popular"
        assert no_overflow(page)
        capture(page, browser_name, width, "clawhub-popular")

        # A page that comes back after the person moved on hands the keyboard nowhere: not when another source
        # was chosen meanwhile (that answer is left behind), nor when they put the focus somewhere else.
        more = page.locator('[data-browse-action="browse_more"]')
        search = page.locator("[data-browse-search]")
        source = page.locator("[data-browse-source]")
        skills.hold = {"/api/skill-catalog/search"}
        more.focus()
        more.press("Enter")
        skills.wait_for_request("/api/skill-catalog/search")
        source.select_option("all")
        _card(page, "ouroboros/weather").wait_for()
        skills.hold.clear()
        with page.expect_request_finished(lambda request: "/api/skill-catalog/search" in request.url):
            skills.release("/api/skill-catalog/search")
        page.wait_for_timeout(150)
        assert page.evaluate("document.activeElement.closest('.marketplace-card')") is None, "a left-behind page moves no focus"
        assert _card(page, "clawhub/third").count() == 0
        source.select_option("clawhub")
        _card(page, "clawhub/second").wait_for()
        skills.hold = {"/api/skill-catalog/search"}
        more.focus()
        more.press("Enter")
        skills.wait_for_request("/api/skill-catalog/search")
        search.focus()
        skills.hold.clear()
        skills.release("/api/skill-catalog/search")
        _card(page, "clawhub/third").wait_for()
        page.wait_for_timeout(150)
        assert page.evaluate("document.activeElement.matches('[data-browse-search]')"), "the focus stays where the person put it"
        source.select_option("all")
        _card(page, "ouroboros/weather").wait_for()
        source.select_option("clawhub")
        _card(page, "clawhub/second").wait_for()
        assert _board_names(page) == ["clawhub/first", "clawhub/second"]
        skills.hold = {"/api/skill-catalog/search"}
        more.focus()
        more.press("Enter")
        skills.wait_for_request("/api/skill-catalog/search")
        _card(page, "clawhub/first").locator('[data-browse-action="open"]').focus()
        skills.hold.clear()
        skills.release("/api/skill-catalog/search")
        _card(page, "clawhub/third").wait_for()
        page.wait_for_timeout(150)
        assert page.evaluate("document.activeElement.closest('.marketplace-card')?.dataset.item") == "clawhub/first", \
            "a card the person moved to keeps the keyboard when the page grows"
        source.select_option("all")
        _card(page, "ouroboros/weather").wait_for()
        source.select_option("clawhub")
        _card(page, "clawhub/second").wait_for()

        # Show more appends the next page by its cursor, keeps the place and hands the keyboard on; a repeat is one card.
        assert more.inner_text() == "Show more"
        more.focus()
        scroller = "document.querySelector('.skills-scroll')"
        page.evaluate(f"{scroller}.scrollTop = {scroller}.scrollHeight")
        before = page.evaluate(f"{scroller}.scrollTop")
        more.press("Enter")
        _card(page, "clawhub/third").wait_for()
        assert claw.queries[-1] == {"source": "clawhub", "sort": "popular", "cursor": 'skillcat:{"offset":2}'}
        assert _board_names(page) == ["clawhub/first", "clawhub/second", "clawhub/third"]
        assert not more.is_visible(), "no next page, no button"
        assert page.evaluate("document.activeElement.closest('.marketplace-card')?.dataset.item") == "clawhub/third"
        assert page.evaluate(f"{scroller}.scrollTop") >= before - 1, "the reader stays where the new cards landed"
        assert no_overflow(page)
        capture(page, browser_name, width, "clawhub-show-more")

        # A card installs through the same id as a search hit.
        skills.answers["/api/skill-catalog/install"] = {"ok": True, "job_id": "job-c", "name": "first", "basis": "review"}
        _card(page, "clawhub/first").locator('[data-browse-action="install"]').click()
        skills.wait_for_count(("POST", "/api/skill-catalog/install"), 1)
        assert skills.posted("/api/skill-catalog/install") == [
            {"id": "clawhub/first", "turn_on_when_ready": True, "version": "1.0.0"}]

        # Changing the order reads the first page of that order.
        sort.select_option("updated")
        _card(page, "clawhub/just-uploaded").wait_for()
        assert claw.queries[-1] == {"source": "clawhub", "sort": "updated"}
        assert _board_names(page) == ["clawhub/just-uploaded"]
        assert "recently updated first" in page.locator("[data-browse-status]").inner_text()

        # A typed search is the same as before, and clearing it brings the registry's page back.
        count = len(claw.queries)
        search.fill("weather")
        search.press("Enter")
        _card(page, "clawhub/typed-hit").wait_for()
        assert claw.queries[count] == {"q": "weather", "source": "clawhub"}
        search.fill("")
        _card(page, "clawhub/first").wait_for()
        assert sort.input_value() == "popular" and claw.queries[-1] == {"source": "clawhub", "sort": "popular"}

        # Back to All sources: no registry cards and no further request.
        count = len(claw.queries)
        page.locator("[data-browse-source]").select_option("all")
        _card(page, "ouroboros/weather").wait_for()
        assert _card(page, "clawhub/first").count() == 0 and len(claw.queries) == count
        assert sort.locator("option").all_inner_texts()[:2] == ["Popular", "Name"]

        # An order or a category of the catalogs doesn't empty the registry's page: it has its own, and no categories.
        sort.select_option("name")
        page.locator('[data-browse-category="games"]').click()
        page.locator("[data-browse-source]").select_option("clawhub")
        _card(page, "clawhub/first").wait_for()
        assert claw.queries[-1] == {"source": "clawhub", "sort": "popular"}
        assert not page.locator("[data-browse-categories]").is_visible()
    finally:
        if skills:
            skills.close(browser)


class _LongClawHub:
    """A first page taller than the window, then a second page of two."""

    def __init__(self, skills):
        skills.answers["/api/skill-catalog/search"] = self.answer

    def answer(self, request):
        if parse_qs(urlsplit(request.url).query).get("cursor"):
            return {"items": [_claw_item("late-one", 2), _claw_item("late-two", 1)]}
        return {"items": [_claw_item(f"skill-{n:02d}", 10000 - n) for n in range(24)], "next_cursor": "skillcat:{\"offset\":24}"}


@pytest.mark.parametrize("browser_name,width", [("chromium", 1280), ("webkit", 1280), ("chromium", 390)])
def test_show_more_keeps_the_reader_in_place_in_an_overflowing_shell(skills_browser, browser_name, width):
    browser = getattr(skills_browser, browser_name).launch(headless=True)
    skills = None
    try:
        page = browser.new_page(viewport={"width": width, "height": 700})
        skills = _open(page)
        # The harness lays the page out at its full height; the app's shell holds it to the window,
        # so the Skills panel itself scrolls. Hold it the same way here.
        page.add_style_tag(content="main { height: 100vh; }")
        _LongClawHub(skills)
        _choose_clawhub(page)
        _card(page, "clawhub/skill-23").wait_for()
        scroller = "document.querySelector('.skills-scroll')"
        page.wait_for_function(f"{scroller}.scrollHeight > {scroller}.clientHeight + 400")
        more = page.locator('[data-browse-action="browse_more"]')
        more.focus()
        page.evaluate(f"{scroller}.scrollTop = {scroller}.scrollHeight")
        before = page.evaluate(f"{scroller}.scrollTop")
        assert before > 400, "the page overflows the shell, so a jump to the top would show"

        # While the next page is read, the shown cards stay exactly where they are.
        skills.hold = {"/api/skill-catalog/search"}
        more.press("Enter")
        skills.wait_for_request("/api/skill-catalog/search")
        assert more.inner_text() == "Reading…"
        assert page.evaluate(f"{scroller}.scrollTop") >= before - 1
        skills.hold.clear()
        skills.release("/api/skill-catalog/search")

        # The appended page lands below them; the reader stays and the keyboard is on the first new card, on screen.
        _card(page, "clawhub/late-two").wait_for()
        page.wait_for_function("document.activeElement.closest('.marketplace-card')?.dataset.item === 'clawhub/late-one'")
        page.wait_for_timeout(150)
        assert page.evaluate(f"{scroller}.scrollTop") >= before - 1, "the reader stays where the new cards landed"
        top, bottom, height = page.evaluate(
            "(() => { const r = document.activeElement.getBoundingClientRect(); return [r.top, r.bottom, innerHeight]; })()")
        assert top >= 0 and bottom <= height, "the focused new card is on screen"
        assert no_overflow(page)
        capture(page, browser_name, width, "clawhub-show-more-long")
        # Sources is shorter. Returning to Browse restores its place against the tall Browse panel.
        _tab(page, "sources")
        page.locator('[data-source-action="add"]').wait_for()
        _tab(page, "browse")
        assert page.evaluate(f"{scroller}.scrollTop") >= before - 1
    finally:
        if skills:
            skills.close(browser)


@pytest.mark.parametrize("browser_name,width", [("chromium", 1280), ("webkit", 390)])
def test_clawhub_that_cannot_be_read_says_so_and_can_be_retried(skills_browser, browser_name, width):
    browser = getattr(skills_browser, browser_name).launch(headless=True)
    skills = None
    try:
        page = browser.new_page(viewport={"width": width, "height": 700})
        skills = _open(page)
        skills.answers["/api/skill-catalog/search"] = {"items": [], "registry_error": "ClawHub: HTTP 503 from registry"}
        _choose_clawhub(page)
        status = page.locator("[data-browse-status]")
        page.wait_for_function("document.querySelector('[data-browse-status]').textContent.includes('Couldn’t read ClawHub')")
        assert "HTTP 503 from registry" in status.inner_text() and status.get_attribute("data-tone") == "danger"
        board = page.locator("[data-browse-board]")
        assert "Nothing to show" not in board.inner_text() and "Nothing was loaded from ClawHub" in board.inner_text()
        assert not page.locator('[data-browse-action="browse_more"]').is_visible()
        assert no_overflow(page)
        capture(page, browser_name, width, "clawhub-error")

        # Try again asks once more; a transport failure reads the same way, and a good answer clears it.
        retry = page.locator('[data-browse-action="browse_retry"]')
        skills.fail("/api/skill-catalog/search", "no route to host")
        retry.click()
        page.wait_for_function("document.querySelector('[data-browse-status]').textContent.includes('no route')")
        assert "Couldn’t read ClawHub" in status.inner_text()
        claw = _ClawHub(skills)
        page.locator('[data-browse-action="browse_retry"]').click()
        _card(page, "clawhub/first").wait_for()
        assert status.get_attribute("data-tone") == "muted"

        # A page that fails keeps what is shown, says why, and Show more retries the same cursor.
        skills.answers["/api/skill-catalog/search"] = {"items": [], "registry_error": "ClawHub: rate limited"}
        more = page.locator('[data-browse-action="browse_more"]')
        more.click()
        page.wait_for_function("document.querySelector('[data-browse-status]').textContent.includes('rate limited')")
        assert "couldn’t read more from ClawHub" in status.inner_text()
        assert _board_names(page) == ["clawhub/first", "clawhub/second"] and more.is_visible()
        skills.answers["/api/skill-catalog/search"] = claw.answer
        more.click()
        _card(page, "clawhub/third").wait_for()

        # Official only hides unmarked registry skills and says so, instead of an unexplained blank.
        page.locator(".skills-official-toggle").click()
        page.wait_for_function("document.querySelector('[data-browse-board]').textContent.includes('Official only hides')")
        assert "Nothing to show" not in page.locator("[data-browse-board]").inner_text()
        assert not more.is_visible()
    finally:
        if skills:
            skills.close(browser)


@pytest.mark.parametrize("browser_name,width", [("chromium", 1280), ("webkit", 390)])
def test_a_source_change_reads_the_installed_listing_again(skills_browser, browser_name, width):
    """The listing's source facts follow the sources (a name added again for another view no longer names its id):
    after a change on Sources the page reads the listing again, and Browse shows what the host now says."""
    browser = getattr(skills_browser, browser_name).launch(headless=True)
    skills = None
    try:
        page = browser.new_page(viewport={"width": width, "height": 900})
        skills = _open(page)
        _tab(page, "browse")
        card = _card(page, "ouroboros/weather")
        card.wait_for()
        assert "available" in card.inner_text() and card.locator('[data-browse-action="update"]').count() == 1
        listing = copy.deepcopy(LISTING)
        next(skill for skill in listing if skill["name"] == "weather")["install"]["source"].pop("id")
        skills.set_skills(listing)
        reads = skills.requests.count(("GET", "/api/extensions"))
        _tab(page, "sources")
        page.locator('[data-source-menu="my-folder"]').click()
        page.locator('body > .skills-card-menu-dialog:not([data-leaving]) [data-menu-action="remove"]').click()
        page.locator(".confirm-dialog [data-confirm-ok]").click()
        skills.wait_for_count(("POST", "/api/skill-sources/my-folder/remove"), 1)
        skills.wait_for_count(("GET", "/api/extensions"), reads + 1)
        _tab(page, "browse")
        page.wait_for_function("""(document.querySelector('[data-browse-board] .marketplace-card[data-item="ouroboros/weather"]')
                                   ?.textContent || '').includes('You have another')""")
        assert "You have another “weather” (from Ouroboros)" in card.inner_text()
        assert card.locator('[data-browse-action="compare"]').count() == 1
        # The sheet describes this catalog entry, not the review, version or Places of the other installed copy.
        skills.answers["/api/skill-catalog/item/ouroboros/weather"] = {"about": "Weather from the new source."}
        card.locator(".skills-card-name").click()
        sheet = page.locator(".skills-detail-sheet")
        sheet.locator(".skills-sheet-about", has_text="Weather from the new source.").wait_for()
        text = sheet.inner_text()
        assert "Installs without review while you trust the Ouroboros source." in text
        assert "Installed without review" not in text and "You have v" not in text
        assert "Installed folder" not in text and sheet.locator('[data-sheet-places]:not([hidden])').count() == 0
        assert sheet.locator('[data-sheet-action="compare"]').count() == 1
        capture(page, browser_name, width, "other-view-detail")
        page.locator("[data-sheet-close]").click()
        assert no_overflow(page)
    finally:
        if skills:
            skills.close(browser)


@pytest.mark.parametrize("browser_name,theme", [("chromium", "light"), ("webkit", "dark")])
def test_browse_fades_only_clipped_prose_and_keeps_the_detail_door(skills_browser, browser_name, theme):
    from playwright.sync_api import expect

    browser = getattr(skills_browser, browser_name).launch(headless=True)
    try:
        page = browser.new_page(viewport={"width": 390, "height": 844})
        description = (
            "Check the forecast before heading out. Compare the next few hours to find a dry stretch for a walk, "
            "or look up the evening forecast in another town before catching a train. The daily view shows when "
            "the chance of rain rises, while the hourly view helps decide whether to wait it out. See wind and "
            "temperature alongside the rain forecast before heading to the coast. Ask for a second place to compare "
            "conditions on either end of a longer trip. A morning summary puts the coldest hour and any change in "
            "wind direction beside the day's high, so there's time to change plans before leaving the house. "
            "Save those two places for the next trip."
        )
        catalog = copy.deepcopy(CATALOG)
        catalog["starter"] = {}
        catalog["items"] = [item for item in catalog["items"] if item["id"] in {"ouroboros/weather", "ouroboros/duckduckgo"}]
        entry = next(item for item in catalog["items"] if item["id"] == "ouroboros/weather")
        entry.update(description=description, signals={**entry["signals"], "votes": 50})
        skills = _open(page, catalog=catalog)
        page.evaluate("theme => document.documentElement.dataset.theme = theme", theme)
        path = "/api/skill-catalog/item/ouroboros/weather"
        skills.answers[path] = {"about": "Full guide with a forecast example."}
        _tab(page, "browse")
        weather = _card(page, "ouroboros/weather")
        long = weather.locator(".skills-card-purpose")
        short = _card(page, "ouroboros/duckduckgo").locator(".skills-card-purpose")
        opener = weather.locator(".skills-read-more")

        expect(opener).to_be_visible()
        expect(long).to_have_attribute("data-fold-clipped", "")
        expect(long).to_have_text(description)
        expect(short).to_be_visible()
        assert short.evaluate("p => p.scrollHeight <= p.clientHeight && !p.hasAttribute('data-fold-clipped')")
        expect(long).to_have_css("mask-image", re.compile("linear-gradient"))
        expect(short).to_have_css("mask-image", "none")
        expect(opener).to_have_css("mask-image", "none")
        opener.scroll_into_view_if_needed()
        capture(page, browser_name, 390, "browse-fold")
        short.scroll_into_view_if_needed()
        capture(page, browser_name, 390, "browse-short-control")

        # Below the two-column layout, the same two cards get an entire wide column each.
        page.set_viewport_size({"width": 620, "height": 844})
        expect(long).not_to_have_attribute("data-fold-clipped", "")
        expect(long).to_have_css("mask-image", "none")
        assert long.evaluate("p => p.scrollHeight <= p.clientHeight")
        # The catalog text now fits, but the detail has more than the catalog description.
        expect(opener).to_be_visible()
        opener.scroll_into_view_if_needed()
        capture(page, browser_name, 620, "browse-unfolded")

        page.set_viewport_size({"width": 390, "height": 844})
        expect(long).to_have_attribute("data-fold-clipped", "")
        _tab(page, "installed")
        _tab(page, "browse")
        expect(long).to_have_attribute("data-fold-clipped", "")
        opener.click()
        sheet = page.locator(".skills-detail-sheet")
        expect(sheet.locator(".skills-sheet-about")).to_have_text("Full guide with a forecast example.")
        assert ("GET", path) in skills.requests
        page.keyboard.press("Escape")
        expect(sheet).to_have_count(0)
        expect(opener).to_be_focused()
    finally:
        browser.close()


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
        assert ("Ouroboros reviews it on this computer before it runs while that is the setting for"
                " the Ouroboros Community source. You can skip the review.") in text
        assert "★ 4 stars on CrossRead/scholion" in text
        assert "Discussion (4)" in text
        assert ("GET", "/api/skill-catalog/item/ouroboros-community/scholion") in skills.requests
        capture(page, browser_name, width, "detail-sheet")
        page.keyboard.press("Escape")
        assert sheet.count() == 0
        assert name.evaluate("element => element === document.activeElement")

        # A press that leaves focus on the page's own region, as WebKit on macOS does with a pressed button: the
        # browser's own close hands focus back to that region, and the sheet still returns it to the pressed name.
        page.locator("[data-extensions-views]").focus()
        name.evaluate("element => element.click()")
        sheet.locator(".skills-sheet-about").wait_for()
        assert page.evaluate("() => document.activeElement.closest('.skills-detail-sheet') !== null")
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
        assert ("Ouroboros reviews it on this computer before it runs while that is the setting for the ClawHub source."
                " You can skip the review.") in page.locator(".skills-detail-sheet").inner_text()
        page.locator("[data-sheet-close]").click()
    finally:
        if skills:
            skills.close(browser)


@pytest.mark.parametrize("browser_name", ["chromium", "webkit"])
def test_the_installed_sheet_takes_access_from_its_own_manifest_never_the_catalog(skills_browser, browser_name):
    """Installed uses its own manifest and grants and only borrows prose for the same release.

    Browse keeps the catalog's own words and labels and leaves out an id it gives neither.
    """
    from playwright.sync_api import expect

    browser = getattr(skills_browser, browser_name).launch(headless=True)
    skills = None
    try:
        page = browser.new_page(viewport={"width": 390, "height": 900})
        catalog = copy.deepcopy(CATALOG)
        entry = next(item for item in catalog["items"] if item["id"] == "ouroboros/weather")
        other_release = {key: entry[key] for key in ("version", "digest")}
        installed = _row("weather")
        installed.update(catalog_id=entry["id"], source_key="git:" + "a" * 32,
                         manifest_file="SKILL.md", permissions=["page_module"])
        installed["install"]["update_available"] = None
        release = {"version": installed["version"], "digest": installed["content_hash"]}
        # Competing, plausible declarations: a known word, and an id that no word or label names.
        entry.update(**release, source_key=installed["source_key"], permissions=["net", "cached_unlabeled"],
                     needs_keys=["WEATHER_KEY"],
                     installed={"state": "installed", **release, "modified": False, "update_available": False})
        skills = _open(page, catalog=catalog, listing=[installed])
        detail_path = "/api/skill-catalog/item/ouroboros/weather"
        detail = {
            "item": {**entry, "permissions": ["net", "feed_reader", "fetched_unlabeled"]},
            "about": "Weather from the source.",
            "uses": [{"id": "net", "kind": "permission"},
                     {"id": "feed_reader", "kind": "permission", "label": "Reads the weather feed"},
                     {"id": "fetched_unlabeled", "kind": "permission"},
                     {"id": "WEATHER_KEY", "kind": "key"}],
            "versions": [{**release, "current": True}],
        }
        skills.answers[detail_path] = detail
        _tab(page, "browse")  # the catalog is now cached, so the installed sheet could join it
        _card(page, "ouroboros/weather").wait_for()
        _tab(page, "installed")
        sheet = page.locator(".skills-detail-sheet")
        card = page.locator('.skills-card[data-skill="weather"]')
        can_use = card.locator(".skills-detail-row", has_text="Can use").locator(".skills-detail-value")

        def open_installed(*, accepts_catalog=True):
            card.locator(".skills-card-name").click()
            status = sheet.locator(".skills-sheet-status")
            if accepts_catalog:
                expect(sheet.locator(".skills-sheet-about")).to_have_text(detail["about"])
                expect(status).to_have_count(0)
            else:
                expect(status).to_contain_text("Showing the installed description.")
                expect(sheet.locator(".skills-sheet-about")).to_have_text(installed["description"])
            return sheet.inner_text()

        text = open_installed()  # a parsed manifest: its own access, with the row's grants (no keys requested)
        assert "What it can use" in text and "Your keys: none" in text and "WEATHER_KEY" not in text
        assert "page_module" in text and "Nothing beyond its own folder" not in text
        for claimed in ("Internet", "Reads the weather feed", "cached_unlabeled", "fetched_unlabeled"):
            assert claimed not in text, f"{claimed} is the catalog's claim, not this manifest's"
        page.locator("[data-sheet-close]").click()
        card.locator(".skills-details > summary").click()
        assert can_use.inner_text() == "page_module"  # the card says the same, not "Nothing beyond its own folder"

        # An id with no friendlier word stands as declared beside the known words: once, in order, and escaped.
        mixed = {**installed, "permissions": ["net", "page_module", 'odd"<i>x</i>', "page_module"],
                 "grants": {**installed["grants"], "requested_keys": ["LOCAL_FORECAST_KEY"]}}
        skills.set_skills([mixed])
        page.locator("#skills-refresh").click()
        page.wait_for_function("!document.querySelector('#skills-refresh').disabled")
        shown = 'Internet · page_module · odd"<i>x</i>'
        assert can_use.inner_text() == shown and card.locator(".skills-detail-value i").count() == 0
        text = open_installed()
        assert shown in text and sheet.locator(".skills-sheet-section i").count() == 0
        assert "Your keys: LOCAL_FORECAST_KEY" in text and "WEATHER_KEY" not in text
        page.locator("[data-sheet-close]").click()

        # The address still resolves, but the currently offered files have changed since this copy was installed.
        skills.answers[detail_path] = {**detail, "item": {**detail["item"], **other_release},
                                       "about": "Full guide for another release.",
                                       "versions": [{**other_release, "current": True}]}
        text = open_installed(accepts_catalog=False)
        assert "Full guide for another release." not in text
        assert shown in text and "Your keys: LOCAL_FORECAST_KEY" in text and "WEATHER_KEY" not in text
        page.locator("[data-sheet-close]").click()

        # Even a match on source and release cannot supply claims for an unread installed manifest.
        skills.answers[detail_path] = detail
        unread = {**installed, "manifest_file": "", "type": "instruction",
                  "load_error": "manifest parse error: bad", "grants": {}}  # its placeholder permissions declare nothing
        skills.set_skills([unread])
        page.locator("#skills-refresh").click()
        page.wait_for_function("!document.querySelector('#skills-refresh').disabled")
        assert "Can use" not in card.inner_text() and "page_module" not in card.inner_text()
        text = open_installed(accepts_catalog=False)
        assert detail["about"] not in text
        assert "What it can use" not in text and "WEATHER_KEY" not in text and "Your keys" not in text
        assert "Nothing beyond its own folder" not in text and "page_module" not in text
        sheet.locator(".skills-sheet-details > summary").click()
        assert unread["payload_root"] in sheet.locator(".skills-sheet-facts").inner_text()  # the exact folder remains
        page.locator("[data-sheet-close]").click()

        # Browse keeps the catalog's own words: the known one and the source's label, and neither unlabeled id.
        _tab(page, "browse")
        skills.hold = {detail_path}
        _card(page, "ouroboros/weather").locator(".skills-card-name").click()
        skills.wait_for_request(detail_path)
        reading = sheet.inner_text()  # before the entry is read: the cached claim
        assert "Internet" in reading and "cached_unlabeled" not in reading and "page_module" not in reading
        skills.release(detail_path)
        skills.hold.clear()
        sheet.locator(".skills-sheet-about", has_text="Weather from the source.").wait_for()
        page.wait_for_function("!document.querySelector('.skills-detail-sheet .skills-sheet-status')")
        text = sheet.inner_text()
        assert "Internet · Reads the weather feed" in text and "Your keys: WEATHER_KEY" in text
        for hidden in ("cached_unlabeled", "fetched_unlabeled", "page_module"):
            assert hidden not in text, hidden
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
        sources["suggestions"] = [
            {"name": "Team skills", "url": "https://git.example.invalid/team/skills.git", "ref": "stable",
             "path": "skills", "about": "Our curated skills", "connected": False},
            {"name": "Ouroboros", "url": "https://github.com/razzant/OuroborosHub", "ref": "main", "about": "",
             "connected": True}]
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

        # A revision and a folder go with the check. While it runs its fields are read-only and another Enter
        # starts nothing, so step two confirms, and Add sends, exactly what was checked.
        page.locator('[data-source-action="add"]').click()
        page.locator("[data-source-address]").fill("https://github.com/team/skills")
        page.locator(".skills-add-more > summary").click()
        page.locator("[data-source-revision]").fill("next")
        page.locator("[data-source-subfolder]").fill("skills/team")
        capture(page, browser_name, width, "add-source-revision-folder")
        skills.hold.add("/api/skill-sources/probe")
        page.locator("[data-add-check]").click()
        skills.wait_for_request("/api/skill-sources/probe")
        fields = ("[data-source-address]", "[data-source-revision]", "[data-source-subfolder]")
        assert [page.locator(field).evaluate("element => element.readOnly") for field in fields] == [True] * 3
        revision = page.locator("[data-source-revision]")
        revision.evaluate("element => { element.value = 'other'; }")  # whatever reaches the field meanwhile
        revision.press("Enter")
        skills.hold.discard("/api/skill-sources/probe")
        skills.release("/api/skill-sources/probe", {"kind": "git", "suggested_name": "skills", "owner": "team",
                                                     "found": ["alpha", "beta"], "commit": "a" * 40, "problems": []})
        page.locator("[data-source-name]").wait_for()
        assert page.locator("[data-source-view]").inner_text() == \
            "Follows next (now at aaaaaaa), reading the folder skills/team."
        page.locator("[data-add-ok]").click()
        skills.wait_for_count(("POST", "/api/skill-sources"), 2)
        checked = {"url": "https://github.com/team/skills", "revision": "next", "subfolder": "skills/team"}
        assert skills.posted("/api/skill-sources/probe")[2:] == [checked]
        assert skills.posted("/api/skill-sources")[1] == {**checked, "name": "skills", "trust": "review"}

        # A build's suggestion fills the address, revision and folder, checks them and names the source; a
        # connected one is not offered.
        page.locator('[data-source-action="add"]').click()
        suggested = page.locator("[data-source-suggestion]")
        suggested.first.wait_for()
        assert suggested.count() == 1 and "git.example.invalid/team/skills.git at stable · folder skills" in \
            suggested.inner_text()
        suggested.click()
        page.locator("[data-source-name]").wait_for()
        assert page.locator("[data-source-name]").input_value() == "Team skills"
        page.locator("[data-add-ok]").click()
        skills.wait_for_count(("POST", "/api/skill-sources"), 3)
        offered = {"url": "https://git.example.invalid/team/skills.git", "revision": "stable", "subfolder": "skills"}
        assert skills.posted("/api/skill-sources/probe")[3:] == [offered]
        assert skills.posted("/api/skill-sources")[2] == {**offered, "name": "Team skills", "trust": "review"}
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


# Every notice there is: where it stands (the sheet, the page), whether it can be seen, and whether a press at its × reaches it.
NOTICES = """() => [...document.querySelectorAll('.toast:not(.is-hiding)')].map((toast) => {
    const close = toast.querySelector('.toast-dismiss');
    const r = close.getBoundingClientRect();
    const at = document.elementFromPoint((r.left + r.right) / 2, (r.top + r.bottom) / 2);
    return { text: toast.querySelector('.toast-words').textContent,
        where: toast.closest('#toast-stack') ? 'page' : toast.closest('dialog.skills-detail-sheet') ? 'sheet' : 'elsewhere',
        visible: getComputedStyle(toast).visibility === 'visible', hit: close.contains(at) };
})"""


def _where(page):
    return [(n["where"], n["visible"], n["hit"]) for n in page.evaluate(NOTICES)]


@pytest.mark.parametrize("browser_name", ["chromium", "webkit"])
@pytest.mark.parametrize("width", [1280, 390])
def test_what_a_press_in_the_browse_sheet_answers_stands_in_the_sheet(skills_browser, browser_name, width):
    """Install and an installed skill's Update, pressed in the detail sheet (a modal dialog: the page behind it is inert),
    answer where they were pressed. The refusal of a name already taken is the one place the card says why; hidden until
    the sheet closes it would be lost. A press on the board, outside any sheet, still answers on the page."""
    browser = getattr(skills_browser, browser_name).launch(headless=True)
    skills = None
    try:
        page = browser.new_page(viewport={"width": width, "height": 844}, has_touch=width < 720)
        # The notices' own six seconds are stretched: the case reads where they stand, across steps of a loaded machine.
        page.add_init_script("(() => { const set = window.setTimeout.bind(window);"
                             " window.setTimeout = (fn, ms, ...rest) => set(fn, ms === 6000 ? 120000 : ms, ...rest); })()")
        skills = _open(page)
        _tab(page, "browse")
        sheet = page.locator("dialog.skills-detail-sheet[open]")
        taken = (409, {"ok": False, "code": "name_taken",
                       "error": "A skill named pdf_tools is already installed (My skills). Keep it, or replace it.",
                       "occupant": {"name": "pdf_tools", "source": "My skills", "locations": ["user_repo"]}})

        def open_sheet(item_id):
            card = _card(page, item_id)
            card.scroll_into_view_if_needed()
            card.locator(".skills-card-name").click()
            sheet.wait_for()

        # Install, refused: said in the sheet, where its × is reached; the × hides it and nothing else.
        skills.answers["/api/skill-catalog/install"] = taken
        open_sheet("ouroboros/pdf_tools")
        sheet.locator('[data-sheet-action="install"]').click()
        sheet.locator(".toast", has_text="already installed (My skills)").wait_for()
        capture(page, browser_name, width, "sheet-install-refused")
        assert _where(page) == [("sheet", True, True)], "said in the sheet and nowhere else"
        sheet.locator(".toast").get_by_role("button", name="Dismiss notice").click()
        page.wait_for_function("() => !document.querySelector('.toast:not(.is-hiding)')")
        assert sheet.count() == 1, "the × leaves the sheet open"
        # Update of an installed skill, refused in the host's words: the same.
        page.keyboard.press("Escape")
        sheet.wait_for(state="detached")
        skills.answers["/api/skills/weather/update"] = (409, {"ok": False, "code": "candidate_changed",
            "error": "Ouroboros now offers a different release of weather; look at it again.",
            "candidate": {"version": "0.3.3", "digest": "f" * 64}})
        open_sheet("ouroboros/weather")
        sheet.locator('[data-sheet-action="update"]').click()
        sheet.locator(".toast", has_text="now offers a different release of weather").wait_for()
        assert _where(page) == [("sheet", True, True)], "an installed skill's action answers in the sheet too"
        # An answer that arrives after the sheet went is the page's.
        skills.hold = {"/api/skills/weather/update"}
        sheet.locator('[data-sheet-action="update"]').click()
        skills.wait_for_request("/api/skills/weather/update")
        page.keyboard.press("Escape")
        sheet.wait_for(state="detached")
        skills.hold.clear()
        skills.release("/api/skills/weather/update")
        page.locator("#toast-stack .toast", has_text="now offers a different release of weather").wait_for()
        assert _where(page) == [("page", True, True)], "the page's, once the sheet has left"
        page.locator("#toast-stack .toast").get_by_role("button", name="Dismiss notice").click()
        # No sheet: the board's own press answers on the page, as before.
        skills.answers["/api/skill-catalog/install"] = taken
        pdf = _card(page, "ouroboros/pdf_tools")
        pdf.scroll_into_view_if_needed()
        pdf.locator('[data-browse-action="install"]').click()
        page.locator("#toast-stack .toast", has_text="already installed (My skills)").wait_for()
        assert _where(page) == [("page", True, True)]
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
