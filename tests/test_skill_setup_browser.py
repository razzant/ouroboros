"""Real Skills controllers and rendering, with controlled API outcomes.

Threaded production endpoint/installer behavior is separately exercised in test_skill_setup_http.
This browser boundary checks pending, grant-failure continuation, retry and host-named removal.
"""
from __future__ import annotations

import pytest
from tests._skills_page_harness import (
    URL, capture, no_overflow, open_skills, refresh, skills_browser as _skills_browser,
)

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]
skills_browser = _skills_browser


def skill(**changes):
    return {
        'name': 'setup_demo', 'type': 'script', 'source': 'native', 'description': 'Readiness example',
        'version': '1.0', 'content_hash': 'a' * 64, 'payload_root': 'skills/native/setup_demo',
        'enabled': False, 'review_status': 'clean', 'review_stale': False,
        'review_gate': {'executable_review': True, 'blocking_reason': 'ready'}, 'executable_review': True,
        'grants': {'all_granted': True}, 'permissions': [], 'removal': 'uninstall', **changes,
    }


@pytest.mark.parametrize('browser_name,width', [('chromium', 1280), ('webkit', 390)])
def test_pending_enable_and_failed_retry_keep_truth_and_removal(skills_browser, browser_name, width):
    browser = getattr(skills_browser, browser_name).launch(headless=True)
    try:
        page = browser.new_page(viewport={'width': width, 'height': 850})
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        facts = open_skills(page, extensions=[skill()])
        target = '/api/skills/setup_demo/toggle'
        job = {'id': 'enable-one', 'target': 'setup_demo', 'kind': 'enable', 'status': 'running',
               'message': 'Installing packages…'}

        def enable(request):
            assert request.headers.get('prefer') == 'respond-async'
            facts.answers['/api/skills/lifecycle-queue'] = {'active': job, 'events': [job]}
            return {'ok': True, 'pending': True, 'job_id': job['id'], 'message': job['message']}

        facts.answers[target] = enable
        page.goto(URL, wait_until='networkidle')
        card = page.locator('.skills-card[data-skill="setup_demo"]')
        card.locator('label.skills-switch').click()
        page.wait_for_function("document.querySelector('.skills-card-status').textContent.includes('Installing packages')")
        assert card.locator('.skills-toggle').is_disabled()
        assert len(facts.posted(target)) == 1
        assert 'setup_demo is on' not in page.locator('body').inner_text()
        capture(page, browser_name, width, 'packages-pending')

        failed = {**job, 'status': 'failed', 'error': 'Its packages did not install.'}
        facts.answers['/api/skills/lifecycle-queue'] = {'active': None, 'events': [failed]}
        facts.set_skills([skill(enabled=True, live_reason='deps_failed', load_error='Its packages did not install.')])
        refresh(page)
        retry = card.locator('[data-skill-action="turn_on"]')
        assert retry.inner_text() == 'Install packages'
        assert not card.locator('[data-skill-action="repair"]').count()
        capture(page, browser_name, width, 'packages-failed')
        retry.click()
        page.wait_for_function("document.querySelector('.skills-card-status').textContent.includes('Installing packages')")
        assert len(facts.posted(target)) == 2, 'one explicit retry, not an automatic replay'

        facts.set_skills([skill(enabled=True)])
        facts.answers['/api/skills/lifecycle-queue'] = {'active': None, 'events': [{**job, 'status': 'succeeded'}]}
        refresh(page)
        assert card.locator('.skills-toggle').is_checked()
        assert card.locator('.skills-toggle').is_enabled()
        card.locator('[data-skill-menu-trigger]').click()
        assert page.locator('body > .skills-card-menu-dialog:not([data-leaving]) .skills-uninstall').is_visible()
        assert not page.locator('body > .skills-card-menu-dialog:not([data-leaving]) .skills-delete-local').count()
        capture(page, browser_name, width, 'packages-ready-menu')
        assert no_overflow(page) and not errors
    finally:
        browser.close()


@pytest.mark.parametrize('browser_name', ['chromium', 'webkit'])
@pytest.mark.parametrize('confirmed', [False, True])
def test_old_native_bridge_keeps_confirmation_and_server_owned_enable(skills_browser, browser_name, confirmed):
    browser = getattr(skills_browser, browser_name).launch(headless=True)
    try:
        page = browser.new_page(viewport={'width': 1000, 'height': 850})
        facts = open_skills(page, extensions=[skill(grants={'all_granted': False, 'missing_permissions': ['inject_chat']})])
        page.add_init_script('''window.__legacyGrants = [];
            window.pywebview = {api: {request_skill_key_grant: async (...args) => {
                window.__legacyGrants.push(args);
                return {ok: CONFIRMED, error: CONFIRMED ? '' : 'Skill grant cancelled.'};
            }}};'''.replace('CONFIRMED', 'true' if confirmed else 'false'))

        def toggle(request):
            facts.set_skills([skill(enabled=True)])
            return {'skill': 'setup_demo', 'enabled': True}

        facts.answers['/api/skills/setup_demo/toggle'] = toggle
        page.goto(URL, wait_until='networkidle')
        page.locator('.skills-card[data-skill="setup_demo"] label.skills-switch').click()
        page.get_by_role('dialog').get_by_role('button', name='Grant access', exact=True).click()
        text = 'setup_demo is on' if confirmed else 'Skill grant cancelled.'
        page.wait_for_function('(text) => document.body.textContent.includes(text)', arg=text)
        assert page.evaluate('window.__legacyGrants') == [['setup_demo', ['inject_chat']]]
        assert facts.posted('/api/skills/setup_demo/grants') == []
        assert facts.posted('/api/skills/setup_demo/toggle') == ([{'enabled': True}] if confirmed else [])
    finally:
        browser.close()


@pytest.mark.parametrize('browser_name', ['chromium', 'webkit'])
@pytest.mark.parametrize('outcome', ['fast_failure', 'pending_success', 'pending_failure'])
def test_grant_setup_failure_or_pending_does_not_post_enable(skills_browser, browser_name, outcome):
    pending = outcome.startswith('pending')
    browser = getattr(skills_browser, browser_name).launch(headless=True)
    try:
        page = browser.new_page(viewport={'width': 1000, 'height': 850})
        facts = open_skills(page, extensions=[skill(grants={'all_granted': False, 'missing_permissions': ['inject_chat']})])
        result = ({'ok': True, 'pending': True, 'job_id': 'grant-one', 'message': 'Installing packages…'} if pending else
                  {'ok': True, 'granted_permissions': ['inject_chat'], 'setup_error': 'Its packages did not install.'})
        def grant(request):
            assert request.headers.get('prefer') == 'respond-async'
            if not pending:
                facts.set_skills([skill(live_reason='deps_failed', load_error='Its packages did not install.')])
            return result

        facts.answers['/api/skills/setup_demo/grants'] = grant
        page.goto(URL, wait_until='networkidle')
        card = page.locator('.skills-card[data-skill="setup_demo"]')
        card.locator('label.skills-switch').click()
        page.get_by_role('dialog').get_by_role('button', name='Grant access', exact=True).click()
        message = 'Installing packages' if pending else 'Access granted, but'
        page.wait_for_function('(text) => document.body.textContent.includes(text)', arg=message)
        # A following GET acts as a barrier after the click handler, not a timing sleep.
        refresh(page)
        assert len(facts.posted('/api/skills/setup_demo/grants')) == 1
        assert facts.posted('/api/skills/setup_demo/grants')[0]['enable_after_grant'] is True
        assert facts.posted('/api/skills/setup_demo/toggle') == []
        assert 'setup_demo is on' not in page.locator('body').inner_text()
        capture(page, browser_name, 1000, 'grant-pending' if pending else 'grant-setup-failed')
        if outcome == 'pending_success':
            facts.set_skills([skill(enabled=True)])
            refresh(page)
            assert card.locator('.skills-toggle').is_checked()
            assert facts.posted('/api/skills/setup_demo/toggle') == []
        else:
            facts.set_skills([skill(live_reason='deps_failed', load_error='Its packages did not install.')])
            facts.answers['/api/skills/lifecycle-queue'] = {'active': None, 'events': [{
                'id': 'grant-one', 'target': 'setup_demo', 'kind': 'grant', 'status': 'succeeded',
                'message': 'Access granted; Its packages did not install.'}]}
            refresh(page)
            assert 'Grant failed' not in card.inner_text() and 'Needs access' not in card.inner_text()
            assert card.locator('[data-skill-action="turn_on"]').inner_text() == 'Install packages'
            capture(page, browser_name, 1000, outcome)
    finally:
        browser.close()
