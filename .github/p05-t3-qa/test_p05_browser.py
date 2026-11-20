"""UI oracles: real SPA controls, isolated candidate, desktop Chromium and phone WebKit."""
import json
import re
from urllib.parse import urlsplit

import httpx
import pytest
from playwright.sync_api import expect, sync_playwright
from ouroboros import places
from tests.settings_browser_helpers import settings_page
from tests.test_telegram_connection_browser import _NAME, _install

from p05_live import ROLE, ROOT, live, publish_fact
from p05_place_fixture import DIRECT, LEGACY, LEGACY_ID, get_row, old_same_list, pending, sha, typed, wakes

pytestmark = pytest.mark.ui_browser
# Accessible names of the two choices a person sees, found through roles instead of the page's private attributes.
WAKE_NAMES = {LEGACY[0]: re.compile(r'^messages addressed to it$', re.I), DIRECT: re.compile(r'^direct messages?$', re.I)}
KEEP_NAME = re.compile(r'^keep my current choice$', re.I)
ADD_NAME = re.compile(r'^add the suggested kinds$', re.I)
MATRIX = [('chromium', 1280, 920, 'light'), ('webkit', 390, 844, 'dark')]


def browser_page(browser, url, width, height, theme):
    page = browser.new_page(viewport={'width': width, 'height': height}, color_scheme=theme,
                            device_scale_factor=1, service_workers='block')
    page.set_default_timeout(16_000)
    expect.set_options(timeout=12_000)
    errors = []
    page.on('pageerror', lambda _error: errors.append('pageerror'))
    page.route('**/*', lambda route: route.continue_() if urlsplit(route.request.url).netloc == urlsplit(url).netloc
               else route.abort())
    page.add_init_script(f"localStorage.setItem('ouroboros.theme',{json.dumps(theme)})")
    return page, errors


def go(page, name, width):
    if width == 390:
        page.locator('[data-mobile-nav-toggle]:visible').click()
    page.locator(f'#primary-sidebar button[data-nav-page="{name}"]').click()
    expect(page.locator(f'#page-{name}')).to_be_visible()


def screenshot(page, target, name):
    expect(target).to_be_visible()
    target.scroll_into_view_if_needed()
    dimensions = page.evaluate('() => [document.documentElement.scrollWidth, innerWidth]')
    assert dimensions[0] <= dimensions[1] + 1
    path = ROOT / 'work' / 'proof' / 'screens' / (name + '.png')
    page.screenshot(path=str(path), animations='disabled')


def is_response(response, url, method):
    return response.url == url and response.request.method == method


@pytest.mark.parametrize('engine,width,height,theme', MATRIX, ids=['chromium', 'webkit'])
def test_telegram_form_and_accounts_key(tmp_path, monkeypatch, engine, width, height, theme):
    name = ('base-' if ROLE == 'base' else '') + f'telegram-{engine}'
    with live(tmp_path / 'telegram') as server:
        server.start()
        # Reviewed public test fixture: bundled register/forms, no supervised sender, no companion.
        _install(server.data, tmp_path / 'fixture-homes', monkeypatch)
        file = server.data / 'state' / 'skills' / _NAME / 'settings.json'
        settings = json.loads(file.read_text(encoding='utf-8'))
        assert isinstance(settings, dict) and settings['TELEGRAM_CHAT_ID'] == '42'
        settings['TELEGRAM_MINIAPP_ENABLED'] = 'off'
        file.write_text(json.dumps(settings) + '\n', encoding='utf-8')
        response = httpx.post(server.url + f'/api/skills/{_NAME}/toggle', json={'enabled': True},
                              timeout=15, trust_env=False)
        assert response.status_code == 200 and response.json()['enabled'] is True
        assert json.loads(file.read_text(encoding='utf-8')) == settings
        saved_sha = sha(file)
        save_url = server.url + f'/api/extensions/{_NAME}/settings/save'
        with sync_playwright() as playwright:
            browser = getattr(playwright, engine).launch(headless=True)
            try:
                page, errors = browser_page(browser, server.url, width, height, theme)
                writes, failed_save_reads = [], []
                page.on('request', lambda request: writes.append(True) if request.method in {'POST', 'PUT', 'PATCH', 'DELETE'}
                        and (request.url == save_url or urlsplit(request.url).path.startswith('/api/settings')) else None)
                page.on('requestfailed', lambda request: failed_save_reads.append(True) if
                        request.url == save_url and request.method == 'GET' else None)
                page.goto(server.url, wait_until='domcontentloaded')
                go(page, 'skills', width)
                card = page.locator(f'.skills-installed-card[data-skill="{_NAME}"]')
                card.locator('[data-skill-menu-trigger]').click()
                with page.expect_response(lambda resp: is_response(resp, save_url, 'GET')) as caught:
                    page.locator(f'[role="menuitem"][data-skill-action="open_settings"][data-skill="{_NAME}"]:visible').click()
                hydration = caught.value
                assert hydration.status == 200 and hydration.finished() is None
                assert hydration.json()['TELEGRAM_MINIAPP_ENABLED'] == 'off'
                level = settings_page(page, f'skill/{_NAME}')
                old_copy = level.get_by_text('The bot token lives in Settings → Secrets', exact=False)
                new_copy = level.get_by_text('The bot token lives in Settings → Accounts and keys', exact=False)
                shown, absent = (old_copy, new_copy) if ROLE == 'base' else (new_copy, old_copy)
                expect(shown).to_be_visible()
                expect(absent).to_have_count(0)
                form = level.locator('[data-extension-settings-form][data-route="settings/save"]')
                mini = form.locator('[name="TELEGRAM_MINIAPP_ENABLED"]')
                expect(mini).to_have_value('off')  # Stored; the bundle defaults to on.
                assert set(mini.locator('option').evaluate_all('nodes => nodes.map(n => n.value)')) == {'on', 'off'}
                expect(form.locator('[name="TELEGRAM_PROXY"]')).to_be_visible()
                save = form.get_by_role('button', name='Save Telegram settings', exact=True)
                expect(save).to_be_visible()
                expect(save).to_be_enabled()
                expect(form.locator('[data-extension-settings-edit-anyway]')).to_have_count(0)
                screenshot(page, mini, name + '-skill-form')
                go(page, 'settings', width)
                settings_page(page).locator('[data-settings-open="accounts"]').click()
                accounts = settings_page(page, 'accounts')
                row = accounts.locator('[data-setting-row="TELEGRAM_BOT_TOKEN"]')
                opener = row.locator('[data-secret-open]')
                expect(opener).to_have_text('Add')
                opener.click()
                editor = row.locator('[data-secret-field] input[data-setting-control]')
                expect(editor).to_be_visible()
                expect(editor).to_have_value('')
                screenshot(page, row, name + '-accounts-empty-editor')
                assert not errors and not failed_save_reads and not writes and sha(file) == saved_sha
            finally:
                browser.close()
    publish_fact(name, {'settings_get': 200, 'saved_miniapp': 'off', 'save_enabled': True,
                        'accounts_editor_empty': True, 'settings_writes': 0,
                        'setup_copy': 'secrets' if ROLE == 'base' else 'accounts_and_keys'})


def wake_box(section, fact):
    return section.locator(f'input[type="checkbox"][data-fact="{fact}"]')


def assert_wake_list(section, row):
    declared = {entry if isinstance(entry, str) else entry['id'] for entry in row['facts']}
    assert declared == {DIRECT}, 'test fixture must explicitly declare only DM'
    stored = set(wakes(row))
    actual = section.locator('input[data-fact]').evaluate_all('nodes => nodes.map(n => n.dataset.fact)')
    assert sorted(actual) == sorted(declared | stored), 'only declared additions and saved removable values'
    for fact in declared | stored:
        box = wake_box(section, fact)
        expect(box).to_be_visible()
        expect(box).to_be_enabled()
        if fact in stored:
            expect(box).to_be_checked()
        else:
            expect(box).not_to_be_checked()


def assert_accessible_wakes(section, row):
    """The same list through roles: each kind is one named checkbox whose state is the stored choice."""
    declared = {entry if isinstance(entry, str) else entry['id'] for entry in row['facts']}
    stored = set(wakes(row))
    boxes = section.get_by_role('checkbox')
    expect(boxes).to_have_count(len(declared | stored))
    for fact in declared | stored:
        box = section.get_by_role('checkbox', name=WAKE_NAMES[fact])
        expect(box).to_have_count(1)
        assert box.get_attribute('data-fact') == fact
        if fact in stored:
            expect(box).to_be_checked()
        else:
            expect(box).not_to_be_checked()


def click_wakes(page, control, server, before, expected):
    url = server.url + '/api/places/' + LEGACY_ID
    with page.expect_response(lambda resp: is_response(resp, url, 'POST')) as caught:
        control.click()
    response = caught.value
    assert response.status == 200 and response.finished() is None
    assert response.request.post_data_json == {'wakes': expected, 'expected_revision': before['revision']}
    body = response.json()
    assert body['ok'] is True
    changed = body['place']
    assert changed['place_id'] == LEGACY_ID and wakes(changed) == expected and 'wake_review' not in changed
    path = places.places_path(server.data)
    digest = sha(path)
    stored = get_row(server)
    assert sha(path) == digest and stored['revision'] == changed['revision'] and wakes(stored) == expected
    assert 'wake_review' not in stored
    return stored


@pytest.mark.parametrize('choice', ['keep', 'add'])
@pytest.mark.parametrize('engine,width,height,theme', MATRIX, ids=['chromium', 'webkit'])
def test_place_sheet_keeps_or_accepts_wakes(tmp_path, engine, width, height, theme, choice):
    name = f'place-{engine}-{choice}'
    with live(tmp_path / 'candidate') as server:
        pending(server.data)
        old_same_list(server, tmp_path / 'old')
        typed(server.data)
        server.start()
        path = places.places_path(server.data)
        digest = sha(path)
        initial = get_row(server)
        assert initial['wake_review'] == 'ambiguous' and initial['wake_suggestions'] == [DIRECT] and wakes(initial) == LEGACY
        with sync_playwright() as playwright:
            browser = getattr(playwright, engine).launch(headless=True)
            try:
                page, errors = browser_page(browser, server.url, width, height, theme)
                page.goto(server.url, wait_until='domcontentloaded')
                go(page, 'circles', width)
                row = page.locator(f'#page-circles button[data-place-open="{LEGACY_ID}"]')
                sheet = page.locator('dialog.place-sheet:visible')
                section = sheet.locator('[data-place-section="wakes"]')

                def reopen():
                    sheet.locator('[data-place-close]').click()
                    expect(sheet).to_have_count(0)
                    row.click()
                    expect(section).to_be_visible()

                row.click()
                expect(section).to_be_visible()
                expect(page.get_by_role('dialog', name=re.compile(r'.+'))).to_have_count(1)
                expect(section).to_contain_text('This older connection has unchecked suggestions.')
                assert_wake_list(section, initial)
                assert_accessible_wakes(section, initial)
                # The new connector kind is offered, not checked, and says so beside its own checkbox.
                expect(section.locator(f'li.place-wake-item:has(input[data-fact="{DIRECT}"])')).to_contain_text('Suggested')
                keep, add = (section.get_by_role('button', name=name) for name in (KEEP_NAME, ADD_NAME))
                for button, act in ((keep, 'keep-wakes'), (add, 'add-suggestions')):
                    expect(button).to_have_count(1)
                    expect(button).to_be_enabled()
                    assert button.get_attribute('data-place-act') == act
                screenshot(page, wake_box(section, LEGACY[0]), name + '-wakes-before')
                assert sha(path) == digest, 'reading/opening the sheet must not write choices'
                action = 'keep-wakes' if choice == 'keep' else 'add-suggestions'
                expected = LEGACY if choice == 'keep' else [*LEGACY, DIRECT]
                changed = click_wakes(page, keep if choice == 'keep' else add, server, initial, expected)
                expect(section.get_by_role('status')).to_contain_text('Current choice kept.' if choice == 'keep'
                                                                     else 'Suggested kinds added.')
                reopen()
                expect(section.locator('[data-place-act="keep-wakes"]')).to_have_count(0)
                expect(section.get_by_role('button', name=KEEP_NAME)).to_have_count(0)
                expect(section.get_by_role('button', name=ADD_NAME)).to_have_count(0)
                assert_wake_list(section, changed)
                assert_accessible_wakes(section, changed)
                screenshot(page, wake_box(section, LEGACY[0]), name + '-wakes-reopened')
                for fact in expected:
                    remaining = [entry for entry in wakes(changed) if entry != fact]
                    expect(wake_box(section, fact)).to_be_checked()
                    control = section.locator(f'label.place-wake-toggle:has(input[data-fact="{fact}"])')
                    changed = click_wakes(page, control, server, changed, remaining)
                    assert_wake_list(section, changed)
                    assert_accessible_wakes(section, changed)
                reopen()
                assert wakes(get_row(server)) == []
                assert_wake_list(section, changed)
                assert not errors
            finally:
                browser.close()
    publish_fact(name, {'old_same_list_http_save': True, 'choice': choice, 'reopened': True,
                        'removed_saved': len(expected), 'cleared': True, 'declared_additions_only': True})
