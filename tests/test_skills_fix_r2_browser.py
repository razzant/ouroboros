"""Round-2 card and publication regressions on the production Skills page."""
from __future__ import annotations

import pytest

from tests._skills_page_harness import (
    URL, capture, fixture, no_overflow, open_skills, refresh, skills_browser as skills_browser,
)

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]
MATRIX = [(engine, width, theme) for engine in ("chromium", "webkit")
          for width in (1280, 390) for theme in ("light", "dark")]


def _open(browser, engine, width, theme):
    page = browser.new_page(viewport={"width": width, "height": 900}, has_touch=width == 390)
    page.set_default_timeout(60_000)
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.add_init_script("window.prompt = window.confirm = window.alert = () => { throw Error('native dialog used'); };")
    if engine == "webkit" and width == 1280:
        page.add_init_script("window.pywebview = {api: {open_external_url: async () => true}};")
    elif engine == "webkit" and width == 390:
        page.add_init_script("window.Telegram = {WebApp: {platform: 'ios', ready() {}, expand() {}}};")
    rows = fixture("extensions_fix_r2.json")["skills"] + fixture("extensions_fix_r1.json")["skills"]
    sources = fixture("skill_sources.json")
    for source in sources["sources"]:
        source["confirmed"] = True
        if source["id"] == "ouroboros":
            source["trust"] = "review"
    catalog = fixture("skill_catalog.json")
    template = catalog["items"][0]
    for row in rows:
        row["submit_hub"] = {"visible": True, "task_start_allowed": True, "publication_ready": True, "state": "ready"}
        catalog["items"].append({**template, "id": f"ouroboros/{row['name']}", "name": row["name"],
                                 "source_id": "ouroboros", "source_name": "Ouroboros", "recommended": False})
    skills = open_skills(page, extensions=rows, sources=sources, catalog=catalog)
    page.goto(URL, wait_until="networkidle", timeout=120_000)
    page.evaluate("theme => document.documentElement.dataset.theme = theme", theme)
    page.locator('.skills-card[data-skill="rejected_newer"]').wait_for()
    return page, skills, errors


def _card(page, name):
    page.locator("#skills-installed-search").fill(name)
    card = page.locator(f'.skills-card[data-skill="{name}"]')
    card.scroll_into_view_if_needed()
    return card


def _contained(page):
    assert no_overflow(page)
    assert page.locator(".marketplace-modal-body").evaluate_all(
        "nodes => nodes.every(el => el.scrollWidth <= el.clientWidth + 1)")


@pytest.mark.parametrize("engine,width,theme", MATRIX)
def test_rejected_updates_keep_reviewed_retry_and_current_candidate(skills_browser, engine, width, theme):
    browser = getattr(skills_browser, engine).launch(headless=True)
    skills = None
    try:
        page, skills, errors = _open(browser, engine, width, theme)
        cases = [("rejected_newer", "its review found problems", True, False),
                 ("review_unfinished", "its review didn't finish", False, True),
                 ("landing_failed", "it couldn't be installed", False, True),
                 ("preflight_failed", "its file checks failed", True, True)]
        for name, cause, findings, skip in cases:
            card = _card(page, name)
            tray = card.locator("[data-skill-trays]")
            assert f"{cause}; the previous version runs." in tray.inner_text()
            assert card.locator(".skills-toggle").is_checked()
            assert tray.get_by_role("button", name="Update", exact=True).is_visible()
            assert tray.get_by_role("button", name="See findings", exact=True).count() == int(findings)
            assert tray.get_by_role("button", name="Update without review", exact=True).count() == int(skip)
            if name == "rejected_newer":
                assert "v0.3.3 is available." in tray.inner_text()
            _contained(page)
            capture(page, engine, width, f"r2-{theme}-{name}")
            if skip:
                tray.get_by_role("button", name="Update without review", exact=True).click()
                dialog = page.locator(".confirm-dialog")
                assert cause in dialog.inner_text()
                dialog.locator("[data-confirm-cancel]").last.click()
                assert skills.posted(f"/api/skills/{name}/update") == []

        newer = _card(page, "rejected_newer")
        newer.get_by_role("button", name="Update", exact=True).click()
        skills.wait_for_count(("POST", "/api/skills/rejected_newer/update"), 1, timeout_ms=60_000)
        row = fixture("extensions_fix_r2.json")["skills"][0]
        expected = {"mode": "replace", "digest": row["install"]["update_available"]["digest"], "local_digest": row["content_hash"]}
        assert skills.posted("/api/skills/rejected_newer/update") == [expected]

        page.locator('[data-tab="browse"]').click()
        page.locator('[data-browse-search]').fill("rejected_newer")
        browse = page.locator('[data-browse-board] .marketplace-card[data-item="ouroboros/rejected_newer"]').first
        assert "v0.3.3 available" in browse.inner_text()
        assert browse.get_by_role("button", name="Open in Installed", exact=True).count() == 0
        browse.get_by_role("button", name="Update", exact=True).click()
        skills.wait_for_count(("POST", "/api/skills/rejected_newer/update"), 2, timeout_ms=60_000)
        assert skills.posted("/api/skills/rejected_newer/update") == [expected, expected]
        _contained(page)
        assert errors == []
    finally:
        if skills:
            skills.close(browser)
        else:
            browser.close()


@pytest.mark.parametrize("engine,width,theme", MATRIX)
def test_publication_says_actual_visibility_through_confirmation_and_task(skills_browser, engine, width, theme):
    browser = getattr(skills_browser, engine).launch(headless=True)
    skills = None
    public = "Everything in the skill's folder becomes public."
    private = "Everything in the skill's folder goes to your private repository."
    path = "/api/skills/rejected_newer/publish-preflight"
    try:
        page, skills, errors = _open(browser, engine, width, theme)
        for index, target in enumerate(("origin", "hub", "public", "private"), start=1):
            card = _card(page, "rejected_newer")
            card.locator("[data-skill-menu-trigger]").click()
            page.locator("body > .skills-card-menu-dialog:not([data-leaving]) .skills-submit-hub").click()
            dialog = page.locator(".skills-dialog")
            dialog.wait_for()
            disclosure = dialog.locator("[data-publication-disclosure]")
            assert disclosure.inner_text() == public
            before = len(skills.posted("/api/skill-sources/my-skills"))
            repository = "razzant/OuroborosHub"
            destination = {"source_id": "ouroboros", "mode": "origin"}
            expected = public
            if target == "hub":
                dialog.locator('[data-choice="new:ouroboros-community"]').click()
                destination = {"source_id": "ouroboros-community", "mode": "new_entry"}
            elif target in ("public", "private"):
                answer = fixture("my_skills.json")[target]
                skills.answers["/api/skill-sources/my-skills"] = answer
                dialog.locator("[data-create-my-skills]").click()
                chosen = dialog.locator('[data-choice="source:my-skills"]')
                chosen.wait_for()
                assert chosen.get_attribute("aria-checked") == "true"
                assert f"owner/ouroboros-skills ({target})" in chosen.inner_text()
                assert answer["source"]["name"] in chosen.inner_text()
                status = dialog.locator("[data-my-skills-status]")
                assert status.inner_text() == f"{'Created' if answer['created'] else 'Connected'} owner/ouroboros-skills ({target})."
                expected = private if target == "private" else public
                assert disclosure.inner_text() == expected
                # Keyboard selection changes the disclosure without replacing the focused option.
                chosen.focus()
                chosen.press("Home")
                assert disclosure.inner_text() == public
                page.keyboard.press("End")
                assert chosen.evaluate("el => document.activeElement === el")
                assert disclosure.inner_text() == expected
                status.scroll_into_view_if_needed()
                repository = answer["repository"]
                destination = {"source_id": "my-skills", "mode": "entry"}
            assert len(skills.posted("/api/skill-sources/my-skills")) == before + int(target in ("public", "private"))
            assert len(skills.posted(path)) == index - 1, 'choosing a repository starts no publication'
            _contained(page)
            capture(page, engine, width, f"r2-{theme}-publish-{target}")
            skills.answers[path] = {"ok": True, "skill": "rejected_newer", "repository": repository,
                                    "state": "ready", "publication_ready": True, "task_start_allowed": True}
            skills.answers["/api/tasks"] = {"ok": True, "task_id": f"publish-{target}"}
            dialog.locator("[data-dialog-ok]").click()
            confirm = page.locator(".confirm-dialog")
            confirm.wait_for()
            assert expected in confirm.inner_text()
            _contained(page)
            if target in ("public", "private"):
                capture(page, engine, width, f"r2-{theme}-confirm-{target}")
            confirm.locator("[data-confirm-ok]").click()
            skills.wait_for_count(("POST", "/api/tasks"), index, timeout_ms=60_000)
            payload = skills.posted("/api/tasks")[-1]
            assert skills.posted(path)[-1] == {"destination": destination}
            assert payload["metadata"]["skill_publish_target"] == {"skill": "rejected_newer", "repository": repository, "destination": destination}
            for field in ("description", "constraints", "context"):
                assert expected in payload[field]
                assert ("becomes public" in payload[field]) == (target != "private")
            toast = page.locator(".toast", has_text=f"publication task publish-{target} started")
            toast.wait_for()
            toast.wait_for(state="detached")
        assert errors == []
    finally:
        if skills:
            skills.close(browser)
        else:
            browser.close()


@pytest.mark.parametrize("engine,width,theme", [("chromium", 1280, "light"), ("webkit", 390, "dark")])
def test_withdrawn_version_can_update_to_the_offered_reverted_release(skills_browser, engine, width, theme):
    browser = getattr(skills_browser, engine).launch(headless=True)
    skills = None
    try:
        page, skills, errors = _open(browser, engine, width, theme)
        rows = fixture("extensions_fix_r1.json")["skills"]
        withdrawn = next(row for row in rows if row["name"] == "old_tool")
        withdrawn["install"]["update_available"] = {"version": "0.1.9", "digest": "c" * 64}
        skills.set_skills(rows)
        refresh(page)
        card = _card(page, "old_tool")
        tray = card.locator("[data-skill-trays]")
        assert "Version 0.2.0 was withdrawn" in tray.inner_text()
        assert "You can update to v0.1.9, or uninstall it." in tray.inner_text()
        assert tray.get_by_role("button", name="Uninstall", exact=True).is_visible()
        _contained(page)
        capture(page, engine, width, f"r2-{theme}-withdrawn-update")
        tray.get_by_role("button", name="Update", exact=True).click()
        skills.wait_for_count(("POST", "/api/skills/old_tool/update"), 1, timeout_ms=60_000)
        assert skills.posted("/api/skills/old_tool/update") == [
            {"mode": "replace", "digest": "c" * 64, "local_digest": withdrawn["content_hash"]}]
        toast = page.locator(".toast", has_text="old_tool: updating to v0.1.9")
        toast.wait_for()
        toast.wait_for(state="detached")
        checked = _card(page, "self_checked")
        checked.locator(".skills-details > summary").click()
        review = checked.locator("[data-skill-detail-review]")
        assert "Ouroboros read it before accepting it" in review.inner_text()
        review.scroll_into_view_if_needed()
        capture(page, engine, width, f"r2-{theme}-self-checked")
        assert errors == []
    finally:
        if skills:
            skills.close(browser)
        else:
            browser.close()
