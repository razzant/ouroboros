"""Round-3 Skills regressions, including the real registry install/listing path."""
from __future__ import annotations

import pytest

from tests._skills_page_harness import (
    URL, capture, fixture, no_overflow, open_skills, skills_browser as skills_browser,
)
from tests.test_skill_install_checks import registry_release as registry_release
from tests.ui_console_errors import PageErrors

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]
MATRIX = [(engine, width, theme) for engine in ("chromium", "webkit")
          for width in (1280, 390) for theme in ("light", "dark")]


def _open(browser, engine, width, theme, rows, sources=None):
    page = browser.new_page(viewport={"width": width, "height": 900}, has_touch=width == 390)
    errors = PageErrors(page, engine)
    page.add_init_script("window.prompt = window.confirm = window.alert = () => { throw Error('native dialog used'); };")
    skills = open_skills(page, extensions=rows, sources=sources)
    page.goto(URL, wait_until="networkidle")
    page.evaluate("theme => document.documentElement.dataset.theme = theme", theme)
    page.locator(".skills-card").first.wait_for()
    page.wait_for_function("() => !document.querySelector('#skills-refresh').disabled")
    return page, skills, errors


def _contained(page):
    assert no_overflow(page)
    assert page.locator(".marketplace-modal-body").evaluate_all(
        "nodes => nodes.every(el => el.scrollWidth <= el.clientWidth + 1)")


@pytest.mark.parametrize("engine,width,theme", MATRIX)
def test_registry_rejected_release_from_host_listing_can_skip_review(
    skills_browser, registry_release, engine, width, theme,
):
    from ouroboros import skill_source_install as installer
    from ouroboros.gateway.extensions import _build_extensions_index
    from ouroboros.skill_install_record import read_install_record
    from ouroboros.skill_loader import SkillReviewState, find_skill, save_enabled, save_review_state
    from playwright.sync_api import expect

    drive, _storage, registry = registry_release
    save_review_state(drive, "demo", SkillReviewState(status="clean", content_hash=find_skill(drive, "demo").content_hash))
    save_enabled(drive, "demo", True, actor="owner")
    registry["latest"] = "2.0.0"
    assert installer.update_installed(drive, "demo", mode="replace")["needs_review"]
    rejected_digest = find_skill(drive, "demo").content_hash
    save_review_state(drive, "demo", SkillReviewState(status="blockers", content_hash=rejected_digest))
    assert installer.settle_pending_update(drive, "demo")["status"] == "restored"
    listing = _build_extensions_index(drive, None)
    row = next(skill for skill in listing["skills"] if skill["name"] == "demo")
    static = next(skill for skill in fixture("extensions_fix_r2.json")["skills"]
                  if skill["name"] == "registry_rejected")
    assert row["install"]["source"]["kind"] == static["install"]["source"]["kind"] == "registry"
    assert row["install"]["update_available"] is static["install"]["update_available"] is None
    assert row["install"]["update_rejected"]["digest"] == rejected_digest
    previous = read_install_record(drive, "demo")

    browser = getattr(skills_browser, engine).launch(headless=True)
    skills = None
    try:
        page, skills, errors = _open(browser, engine, width, theme, [row])
        card = page.locator('.skills-card[data-skill="demo"]')
        skip = card.get_by_role("button", name="Update without review", exact=True)
        expect(skip).to_be_enabled()
        assert card.locator(".skills-toggle").is_checked()
        assert card.get_by_role("button", name="Update", exact=True).count() == 0
        _contained(page)
        capture(page, engine, width, f"r3-{theme}-registry-rejected")
        skip.click()
        confirm = page.locator(".confirm-dialog")
        assert "Update demo to 2.0.0 without review?" in confirm.inner_text()
        assert "its review found problems" in confirm.inner_text()
        _contained(page)
        capture(page, engine, width, f"r3-{theme}-registry-confirm")
        confirm.locator("[data-confirm-cancel]").last.click()
        assert skills.posted("/api/skills/demo/update") == []

        # Browser requests reach the real installer; registry transport alone is synthetic.
        def update(request):
            body = request.post_data_json
            try:
                result = installer.update_installed(drive, "demo", **body)
            except installer.SkillInstallError as error:
                return error.status, {"ok": False, "code": error.code, "error": str(error)}
            skills.set_skills(_build_extensions_index(drive, None)["skills"])
            return 200, result

        skills.answers["/api/skills/demo/update"] = update
        registry.update(latest="3.0.0", changed=True)
        skip.click()
        confirm.locator("[data-confirm-ok]").click()
        skills.wait_for_count(("POST", "/api/skills/demo/update"), 1)
        assert registry["staged_versions"][-1] == "2.0.0"
        assert read_install_record(drive, "demo") == previous
        toast = page.locator(".toast", has_text="ClawHub now offers different bytes")
        toast.wait_for()
        toast.click()
        toast.wait_for(state="detached")

        registry["changed"] = False
        skip.click()
        confirm.locator("[data-confirm-ok]").click()
        skills.wait_for_count(("POST", "/api/skills/demo/update"), 2)
        expected = {"mode": "replace", "digest": rejected_digest, "skip_review": True}
        assert skills.posted("/api/skills/demo/update") == [expected, expected]
        accepted = read_install_record(drive, "demo")
        assert accepted["basis"] == "owner_skip"
        assert accepted["release"]["version"] == "2.0.0"
        assert accepted["release"]["digest"] == rejected_digest
        assert registry["staged_versions"][-1] == "2.0.0"
        assert "update_rejected" not in accepted
        assert errors == []
    finally:
        skills.close(browser) if skills else browser.close()


@pytest.mark.parametrize("engine,width,theme", MATRIX)
def test_publication_visibility_survives_saved_sources_confirmation_and_task(skills_browser, engine, width, theme):
    from playwright.sync_api import expect

    answers = fixture("my_skills.json")
    sources = fixture("skill_sources.json")
    sources["sources"].append(answers["private"]["source"])
    for source in sources["sources"]:
        source["confirmed"] = True
    row = fixture("extensions_fix_r2.json")["skills"][0]
    row["submit_hub"] = {"visible": True, "task_start_allowed": True, "publication_ready": True, "state": "ready"}
    origin = fixture("extensions_fix_r2.json")["skills"][0]
    origin.update(name="private_origin", submit_hub=row["submit_hub"])
    origin["install"]["source"] = answers["private"]["source"]
    public = "Everything in the skill's folder becomes public."
    private = "Everything in the skill's folder goes to your private repository."
    unknown = "Everything in the skill's folder goes to owner/ouroboros-skills; whoever can read that repository can see it."
    browser = getattr(skills_browser, engine).launch(headless=True)
    skills = None
    try:
        page, skills, errors = _open(browser, engine, width, theme, [row, origin], sources)
        for index, target in enumerate(("saved", "origin", "hub", "public", "private", "reopened"), start=1):
            if target == "reopened":
                # A fresh page sees only the saved source, which carries no visibility.
                page.reload(wait_until="networkidle")
                page.evaluate("theme => document.documentElement.dataset.theme = theme", theme)
            name = "private_origin" if target == "origin" else "rejected_newer"
            page.locator("#skills-installed-search").fill(name)
            card = page.locator(f'.skills-card[data-skill="{name}"]')
            card.locator("[data-skill-menu-trigger]").click()
            page.locator("body > .skills-card-menu-dialog:not([data-leaving]) .skills-submit-hub").click()
            dialog = page.locator(".skills-dialog")
            dialog.wait_for()
            repository = "owner/ouroboros-skills"
            destination = {"source_id": "my-skills", "mode": "origin" if target == "origin" else "entry"}
            expected = unknown
            before = len(skills.posted("/api/skill-sources/my-skills"))
            if target in ("private", "public"):
                skills.answers["/api/skill-sources/my-skills"] = answers[target]
                dialog.locator("[data-create-my-skills]").click()
                expect(dialog.locator("[data-my-skills-status]")).to_contain_text(f"({target})")
                expected = private if target == "private" else public
            elif target == "hub":
                dialog.locator('[data-choice="new:ouroboros-community"]').click()
                destination = {"source_id": "ouroboros-community", "mode": "new_entry"}
                repository = "razzant/OuroborosHub"
                expected = public
            elif target != "origin":
                dialog.locator('[data-choice="source:my-skills"]').click()
            disclosure = dialog.locator("[data-publication-disclosure]")
            assert disclosure.inner_text() == expected
            assert len(skills.posted("/api/skill-sources/my-skills")) == before + int(target in ("private", "public"))
            assert len(skills.posted("/api/tasks")) == index - 1
            # The same focused choice survives keyboard navigation while the disclosure changes.
            chosen = dialog.locator('[aria-checked="true"]')
            value = chosen.get_attribute("data-choice")
            chosen.focus()
            chosen.press("Home")
            first_disclosure = unknown if target == "origin" else public
            assert disclosure.inner_text() == first_disclosure
            dialog.locator(f'[data-choice="{value}"]').click()
            assert disclosure.inner_text() == expected
            disclosure.scroll_into_view_if_needed()
            disclosure.hover()
            # Capture the settled choice tint, after its real hover/selection transitions.
            dialog.evaluate("async el => await Promise.all(el.getAnimations({subtree: true}).map(a => a.finished.catch(() => {})))")
            _contained(page)
            capture(page, engine, width, f"r3-{theme}-publish-{target}")
            path = f"/api/skills/{name}/publish-preflight"
            skills.answers[path] = {"ok": True, "skill": name, "repository": repository,
                                    "state": "ready", "publication_ready": True, "task_start_allowed": True}
            skills.answers["/api/tasks"] = {"ok": True, "task_id": f"publish-{target}"}
            dialog.locator("[data-dialog-ok]").click()
            confirm = page.locator(".confirm-dialog")
            expect(confirm).to_contain_text(expected)
            _contained(page)
            capture(page, engine, width, f"r3-{theme}-confirm-{target}")
            confirm.locator("[data-confirm-ok]").click()
            skills.wait_for_count(("POST", "/api/tasks"), index)
            payload = skills.posted("/api/tasks")[-1]
            assert skills.posted(path)[-1] == {"destination": destination}
            assert payload["metadata"]["skill_publish_target"] == {
                "skill": name, "repository": repository, "destination": destination}
            for field in ("description", "constraints", "context"):
                assert expected in payload[field]
                assert ("becomes public" in payload[field]) == (expected == public)
                assert ("private repository" in payload[field]) == (expected == private)
            toast = page.locator(".toast", has_text=f"publication task publish-{target} started")
            toast.wait_for()
            toast.click()
            toast.wait_for(state="detached")
        assert errors.unexpected() == [], errors
    finally:
        skills.close(browser) if skills else browser.close()
