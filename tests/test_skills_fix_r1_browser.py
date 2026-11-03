"""Fix-round UI journeys through production modules, with common A–F host responses."""
from __future__ import annotations

import pytest

from tests._skills_page_harness import (
    URL, capture, fixture, no_overflow, open_skills, skills_browser as skills_browser,
)

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]
MATRIX = [(engine, width, theme) for engine in ("chromium", "webkit")
          for width in (1280, 390) for theme in ("light", "dark")]


def _open(browser, width, theme, browser_name):
    page = browser.new_page(viewport={"width": width, "height": 900}, has_touch=width == 390)
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    # Native dialogs must never be needed in the desktop bridge or Telegram webview.
    page.add_init_script("""
        window.prompt = window.confirm = window.alert = () => { throw Error('native dialog used'); };
    """)
    if browser_name == 'webkit' and width == 1280:
        page.add_init_script('window.pywebview = {api: {open_external_url: async () => true}};')
    elif browser_name == 'webkit' and width == 390:
        page.add_init_script("window.Telegram = {WebApp: {platform: 'ios', ready() {}, expand() {}}};")
    sources = fixture("skill_sources.json")
    for source in sources["sources"]:
        source["confirmed"] = True
    listing = fixture("extensions_fix_r1.json")["skills"] + fixture("extensions_v4.json")["skills"][:2]
    listing[0]["submit_hub"] = {"visible": True, "task_start_allowed": True, "publication_ready": True, "state": "ready"}
    skills = open_skills(page, extensions=listing, sources=sources, catalog=fixture("skill_catalog.json"))
    page.goto(URL, wait_until="networkidle")
    page.evaluate("theme => document.documentElement.dataset.theme = theme", theme)
    page.locator('[data-skill="update_rejected"].skills-card').wait_for()
    return page, skills, errors


def _only(page, name):
    page.locator("#skills-installed-search").fill(name)
    card = page.locator(f'.skills-card[data-skill="{name}"]')
    card.scroll_into_view_if_needed()
    return card


def _contained(page):
    assert no_overflow(page)
    assert page.locator(".marketplace-modal-body").evaluate_all(
        "nodes => nodes.every(el => el.scrollWidth <= el.clientWidth + 1)")


@pytest.mark.parametrize("browser_name,width,theme", MATRIX)
def test_acceptance_and_update_outcomes(skills_browser, browser_name, width, theme):
    browser = getattr(skills_browser, browser_name).launch(headless=True)
    skills = None
    try:
        page, skills, errors = _open(browser, width, theme, browser_name)
        rejected = _only(page, "update_rejected")
        assert "Update to 0.3.2: its review found problems; the previous version runs." in rejected.inner_text()
        assert rejected.locator('.skills-toggle').is_checked()
        capture(page, browser_name, width, f"{theme}-update-rejected")
        rejected.get_by_role("button", name="See findings", exact=True).click()
        assert rejected.locator('.skills-review-history').evaluate("el => el.open")
        assert "blockers" in rejected.locator('.skills-review-history').inner_text()
        rejected.get_by_role("button", name="Update without review", exact=True).click()
        dialog = page.locator('.confirm-dialog')
        assert "Update update_rejected to 0.3.2 without review?" in dialog.inner_text()
        dialog.locator('[data-confirm-cancel]').last.click()
        assert skills.posted('/api/skills/update_rejected/update') == []
        rejected.get_by_role("button", name="Update without review", exact=True).click()
        dialog.locator('[data-confirm-ok]').click()
        skills.wait_for_count(('POST', '/api/skills/update_rejected/update'), 1)
        digest = fixture('extensions_fix_r1.json')['skills'][0]['install']['update_rejected']['digest']
        assert skills.posted('/api/skills/update_rejected/update') == [{'mode': 'replace', 'digest': digest, 'skip_review': True}]

        pending = _only(page, 'pending_update')
        assert "Updating to 2.0.0: it stays off until the review finishes." in pending.inner_text()
        assert pending.locator('.skills-toggle').is_disabled()
        assert not pending.locator('.skills-toggle').is_checked()
        page.locator('.toast').wait_for(state='detached', timeout=10_000)
        capture(page, browser_name, width, f"{theme}-pending-update")

        for name, sentence in [
            ('weather', 'because you trust Ouroboros.'),
            ('telegram', 'because you trust Built in.'),
            ('self_checked', 'Ouroboros read it before accepting it and recorded why:'),
        ]:
            card = _only(page, name)
            card.locator('.skills-details > summary').click()
            review = card.locator('[data-skill-detail-review]')
            assert sentence in review.inner_text()
            assert 'pending' not in review.inner_text() and 'Reviewed on this computer' not in review.inner_text()
            review.scroll_into_view_if_needed()
            if name == 'telegram':
                assert card.locator('[data-official-mark]').count() == 1
                assert card.locator('.skills-card-marks').inner_text().strip() == 'Built in\nOfficial'
            if name == 'self_checked':
                notes = fixture('extensions_fix_r1.json')['skills'][2]['install']['check']['notes']
                assert notes in card.locator('[data-skill-detail-facts]').text_content()
                assert len(review.inner_text()) < len(notes)
                assert card.locator('script').count() == 0
            capture(page, browser_name, width, f"{theme}-{name}-acceptance")
            _contained(page)
            if name == 'self_checked':
                detail = fixture('skill_catalog_item.json')
                detail['item'].update(name=name, source_id='ouroboros', source_name='Ouroboros')
                skills.answers[f'/api/skill-catalog/item/ouroboros/{name}'] = detail
                card.locator('.skills-card-name').click()
                sheet = page.locator('.skills-detail-sheet')
                page.wait_for_function("!document.querySelector('.skills-detail-sheet .skills-sheet-status')")
                sheet.locator('.skills-sheet-details > summary').click()
                assert notes in sheet.locator('.skills-sheet-facts').text_content()
                assert sheet.locator('script').count() == 0
                sheet.locator('[data-sheet-close]').click()

        withdrawn = _only(page, 'old_tool')
        assert 'Version 0.2.0 was withdrawn by Ouroboros Community' in withdrawn.inner_text()
        page.locator('[data-tab="browse"]').click()
        builtin = page.locator('[data-browse-board] [data-item="builtin/telegram"]').first
        builtin.wait_for()
        assert builtin.locator('[data-official-mark]').count() == 1
        assert 'Built in' in builtin.inner_text()
        builtin.locator('[data-official-mark]').click()
        popup = page.locator('.skills-mark-popup:not([data-leaving])')
        assert 'the Built in source' in popup.inner_text()
        assert 'github.com' not in popup.inner_text()
        popup.locator('[data-popup-close]').click()
        assert errors == []
    finally:
        if skills:
            skills.close(browser)
        else:
            browser.close()


@pytest.mark.parametrize("browser_name,width,theme", MATRIX)
def test_source_warning_and_my_skills_opt_in(skills_browser, browser_name, width, theme):
    browser = getattr(skills_browser, browser_name).launch(headless=True)
    skills = None
    try:
        page, skills, errors = _open(browser, width, theme, browser_name)
        skills.answers['/api/skill-sources/probe'] = {'kind': 'git', 'suggested_name': 'Team skills',
            'found': ['weather'], 'warning': 'not_encrypted'}
        page.locator('[data-tab="sources"]').click()
        page.locator('[data-source-action="add"]').click()
        page.locator('[data-source-address]').fill('http://git.example/skills')
        page.locator('[data-add-check]').click()
        warning = page.locator('[data-source-warning]')
        warning.wait_for()
        assert warning.inner_text() == 'This address is not encrypted: anyone on the network path could change what you install.'
        _contained(page)
        capture(page, browser_name, width, f'{theme}-source-warning')
        assert page.locator('[data-add-ok]').is_enabled()
        page.locator('[data-add-ok]').click()
        skills.wait_for_count(('POST', '/api/skill-sources'), 1)
        assert skills.posted('/api/skill-sources') == [{'url': 'http://git.example/skills', 'name': 'Team skills', 'trust': 'review'}]

        page.locator('[data-tab="installed"]').click()
        card = _only(page, 'update_rejected')
        card.locator('[data-skill-menu-trigger]').click()
        page.locator('body > .skills-card-menu-dialog:not([data-leaving]) .skills-submit-hub').click()
        dialog = page.locator('.skills-dialog')
        dialog.wait_for()
        assert skills.posted('/api/skill-sources/my-skills') == [], 'opening Publish has no repository side effect'
        assert dialog.get_by_role('button', name='Create or connect «My skills» on GitHub').count() == 1
        _contained(page)
        page.locator('.toast').wait_for(state='detached', timeout=10_000)
        capture(page, browser_name, width, f'{theme}-publish-my-skills')
        for code, next_step in [('no_token', 'Add GITHUB_TOKEN'), ('cannot_create', 'allow repository access'), ('failed', 'check your GitHub connection')]:
            skills.answers['/api/skill-sources/my-skills'] = (400, {'ok': False, 'code': code, 'error': 'fixture refusal'})
            dialog.locator('[data-create-my-skills]').click()
            dialog.locator('[role="alert"]', has_text=next_step).wait_for()
            assert next_step in dialog.locator('[role="alert"]').inner_text()
            assert dialog.locator('[data-dialog-ok]').is_enabled()

        skills.answers['/api/skill-sources/my-skills'] = {'ok': True, 'created': True, 'private': True, 'repository': 'owner/ouroboros-skills',
            'source': {'id': 'my-skills', 'name': 'My skills', 'kind': 'git', 'url': 'https://github.com/owner/ouroboros-skills',
                       'trust': 'trusted', 'confirmed': True}}
        dialog.locator('[data-create-my-skills]').click()
        chosen = dialog.locator('[data-choice="source:my-skills"]')
        chosen.wait_for()
        assert chosen.get_attribute('aria-checked') == 'true'
        assert 'owner/ouroboros-skills (private)' in chosen.inner_text()
        assert dialog.locator('[data-create-my-skills]').count() == 0
        assert skills.posted('/api/skills/update_rejected/publish-preflight') == []
        _contained(page)
        capture(page, browser_name, width, f'{theme}-my-skills-selected')
        skills.answers['/api/skills/update_rejected/publish-preflight'] = {'ok': True, 'skill': 'update_rejected',
            'repository': 'owner/ouroboros-skills', 'state': 'ready', 'publication_ready': True, 'task_start_allowed': True}
        skills.answers['/api/tasks'] = {'ok': True, 'task_id': 'publish-private'}
        dialog.locator('[data-dialog-ok]').click()
        confirm = page.locator('.confirm-dialog')
        confirm.wait_for()
        assert 'public request' not in confirm.inner_text()
        confirm.locator('[data-confirm-ok]').click()
        skills.wait_for_count(('POST', '/api/tasks'), 1)
        destination = {'source_id': 'my-skills', 'mode': 'entry'}
        assert skills.posted('/api/skills/update_rejected/publish-preflight') == [{'destination': destination}]
        assert skills.posted('/api/tasks')[0]['metadata']['skill_publish_target']['destination'] == destination
        assert skills.posted('/api/skill-sources/my-skills') == [{}, {}, {}, {}]
        assert errors == []
    finally:
        if skills:
            skills.close(browser)
        else:
            browser.close()
