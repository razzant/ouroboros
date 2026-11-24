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

from p05_live import ROLE, ROOT, live, publish_fact, publish_step
from p05_place_fixture import DIRECT, INBOX, LEGACY, LEGACY_ID, get_row, old_same_list, pending, sha, typed, wakes

pytestmark = pytest.mark.ui_browser
# Accessible names of the two choices a person sees, found through roles instead of the page's private attributes.
WAKE_NAMES = {LEGACY[0]: re.compile(r'^messages addressed to it$', re.I), DIRECT: re.compile(r'^direct messages?$', re.I)}
PENDING_SIGNAL_NAME = re.compile(r'^Priority inbox.+once the connector is on', re.I)
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


def screenshot(page, target, name, case=None):
    expect(target).to_be_visible()
    target.scroll_into_view_if_needed()
    dimensions = page.evaluate('() => [document.documentElement.scrollWidth, innerWidth]')
    if case is not None:
        # The measurement lands before the assertion: if the layout overflows, the numbers that
        # explain it are already in the proof, under the step the case reached.
        publish_step(case, scroll_width=dimensions[0], inner_width=dimensions[1])
    assert dimensions[0] <= dimensions[1] + 1
    path = ROOT / 'work' / 'proof' / 'screens' / (name + '.png')
    page.screenshot(path=str(path), animations='disabled')


def is_response(response, url, method):
    return response.url == url and response.request.method == method


@pytest.mark.parametrize('engine,width,height,theme', MATRIX, ids=['chromium', 'webkit'])
def test_telegram_form_and_accounts_key(tmp_path, monkeypatch, engine, width, height, theme):
    name = ('base-' if ROLE == 'base' else '') + f'telegram-{engine}'
    publish_step(name, 'case_started', viewport_width=width, viewport_height=height)
    with live(tmp_path / 'telegram') as server:
        server.start()
        publish_step(name, 'server_ready')
        # Reviewed public test fixture: bundled register/forms, no supervised sender, no companion.
        _install(server.data, tmp_path / 'fixture-homes', monkeypatch)
        file = server.data / 'state' / 'skills' / _NAME / 'settings.json'
        settings = json.loads(file.read_text(encoding='utf-8'))
        assert isinstance(settings, dict) and settings['TELEGRAM_CHAT_ID'] == '42'
        settings['TELEGRAM_MINIAPP_ENABLED'] = 'off'
        file.write_text(json.dumps(settings) + '\n', encoding='utf-8')
        response = httpx.post(server.url + f'/api/skills/{_NAME}/toggle', json={'enabled': True},
                              timeout=15, trust_env=False)
        publish_step(name, 'skill_toggle_responded', http_status=response.status_code)
        assert response.status_code == 200 and response.json()['enabled'] is True
        assert json.loads(file.read_text(encoding='utf-8')) == settings
        publish_step(name, 'skill_enabled')
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
                publish_step(name, 'page_loaded')
                go(page, 'skills', width)
                publish_step(name, 'entered_extensions')
                if width == 390:
                    # Below the 40rem section breakpoint (extensions.js), Extensions lands on its home list;
                    # tap the visible Skills row through the normal UI to open the installed skills view.
                    skills_row = page.locator('#page-skills [data-extensions-open="installed"]:visible')
                    expect(skills_row).to_have_count(1)
                    expect(skills_row).to_be_visible()
                    skills_row.click()
                expect(page.locator('#skills-pane-installed')).to_be_visible()
                publish_step(name, 'skills_list_open')
                card = page.locator(f'.skills-installed-card[data-skill="{_NAME}"]')
                expect(card).to_be_visible()
                publish_step(name, 'skill_card_visible')
                trigger = card.locator('[data-skill-menu-trigger]')
                expect(trigger).to_be_visible()
                trigger.click()
                publish_step(name, 'skill_menu_clicked')
                open_settings = page.locator(
                    f'[role="menuitem"][data-skill-action="open_settings"][data-skill="{_NAME}"]:visible')
                expect(open_settings).to_be_visible()
                publish_step(name, 'skill_menu_open')
                with page.expect_response(lambda resp: is_response(resp, save_url, 'GET')) as caught:
                    open_settings.click()
                hydration = caught.value
                publish_step(name, 'skill_settings_responded', http_status=hydration.status)
                assert hydration.status == 200 and hydration.finished() is None
                assert hydration.json()['TELEGRAM_MINIAPP_ENABLED'] == 'off'
                publish_step(name, 'skill_form_hydrated')
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
                publish_step(name, 'skill_form_ready')
                screenshot(page, mini, name + '-skill-form', case=name)
                publish_step(name, 'skill_form_captured')
                if width == 390:
                    # Compact (< 40rem): the skill form sits at depth 2 inside #page-skills, which hides the
                    # mobile drawer toggle while any page level is open. Walk Back to Skills, then Back to
                    # Extensions home, and open Settings through the real visible drawer toggle.
                    ext_page = page.locator('#page-skills')
                    back_btn = page.locator('#page-skills > .app-page-header [data-shell-back]:visible')
                    for expected_depth, step_name in (('1', 'back_to_skills_list'), ('0', 'back_to_extensions_home')):
                        expect(back_btn).to_have_count(1)
                        expect(back_btn).to_have_attribute('aria-label', re.compile(r'^Back to .+'))
                        back_btn.click()
                        expect(ext_page).to_have_attribute('data-extensions-depth', expected_depth)
                        publish_step(name, step_name)
                    expect(page.locator('#page-skills[data-section-view="home"]')).to_be_visible()
                    expect(page.locator('#page-skills > .app-page-header [data-mobile-nav-toggle]:visible')).to_have_count(1)
                    go(page, 'settings', width)
                    expect(page.locator('#page-settings .settings-shell[data-settings-view="home"]:visible')).to_be_visible()
                else:
                    go(page, 'settings', width)
                publish_step(name, 'entered_settings', viewport_width=width, viewport_height=height)
                # The home's Accounts door is the person-facing path. The dock and the stack
                # layouts can each hold a copy of the home, so the door the person can see in
                # this layout is the one to press; if none is visible, the overview row the
                # product's own settings helper opens first is opened.
                doors = settings_page(page).locator('[data-settings-open="accounts"]:visible')
                if not doors.count():
                    overview = settings_page(page).locator('[data-settings-overview-row]:visible')
                    if overview.count() == 1:
                        overview.click()
                expect(doors).to_be_visible()
                doors.click()
                publish_step(name, 'accounts_door_clicked')
                accounts = settings_page(page, 'accounts')
                expect(accounts).to_be_attached()
                expect(page.locator('#page-settings section[data-settings-page="accounts"]:visible')).to_have_count(1)
                publish_step(name, 'accounts_level_attached')
                row = accounts.locator('[data-setting-row="TELEGRAM_BOT_TOKEN"]')
                expect(row).to_be_visible()
                publish_step(name, 'secret_row_found')
                opener = row.locator('[data-secret-open]')
                expect(opener).to_have_text('Add')
                publish_step(name, 'secret_opener_label_ok')
                opener.click()
                editor = row.locator('[data-secret-field] input[data-setting-control]')
                expect(editor).to_be_visible()
                publish_step(name, 'secret_editor_visible')
                expect(editor).to_have_value('')
                publish_step(name, 'secret_editor_empty')
                screenshot(page, row, name + '-accounts-empty-editor', case=name)
                publish_step(name, 'accounts_captured')
                assert not errors and not failed_save_reads and not writes and sha(file) == saved_sha
            finally:
                browser.close()
    publish_fact(name, {'settings_get': 200, 'saved_miniapp': 'off', 'save_enabled': True,
                        'accounts_editor_empty': True, 'settings_writes': 0,
                        'setup_copy': 'secrets' if ROLE == 'base' else 'accounts_and_keys'})
    publish_step(name, 'fact_published')


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
    publish_step(name, 'case_started', viewport_width=width, viewport_height=height)
    with live(tmp_path / 'candidate') as server:
        pending(server.data)
        old_same_list(server, tmp_path / 'old')
        typed(server.data)
        server.start()
        publish_step(name, 'server_ready')
        path = places.places_path(server.data)
        digest = sha(path)
        initial = get_row(server)
        assert initial['wake_review'] == 'ambiguous' and initial['wake_suggestions'] == [DIRECT] and wakes(initial) == LEGACY
        assert 'unpublished_signals' not in initial
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
                publish_step(name, 'sheet_open')
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
                screenshot(page, wake_box(section, LEGACY[0]), name + '-wakes-before', case=name)
                publish_step(name, 'wakes_before_captured')
                assert sha(path) == digest, 'reading/opening the sheet must not write choices'
                action = 'keep-wakes' if choice == 'keep' else 'add-suggestions'
                expected = LEGACY if choice == 'keep' else [*LEGACY, DIRECT]
                changed = click_wakes(page, keep if choice == 'keep' else add, server, initial, expected)
                publish_step(name, 'choice_saved')
                expect(section.get_by_role('status')).to_contain_text('Current choice kept.' if choice == 'keep'
                                                                     else 'Suggested kinds added.')
                reopen()
                publish_step(name, 'sheet_reopened')
                expect(section.locator('[data-place-act="keep-wakes"]')).to_have_count(0)
                expect(section.get_by_role('button', name=KEEP_NAME)).to_have_count(0)
                expect(section.get_by_role('button', name=ADD_NAME)).to_have_count(0)
                assert_wake_list(section, changed)
                assert_accessible_wakes(section, changed)
                screenshot(page, wake_box(section, LEGACY[0]), name + '-wakes-reopened', case=name)
                for fact in expected:
                    remaining = [entry for entry in wakes(changed) if entry != fact]
                    expect(wake_box(section, fact)).to_be_checked()
                    control = section.locator(f'label.place-wake-toggle:has(input[data-fact="{fact}"])')
                    changed = click_wakes(page, control, server, changed, remaining)
                    assert_wake_list(section, changed)
                    assert_accessible_wakes(section, changed)
                reopen()
                assert wakes(get_row(server)) == []
                publish_step(name, 'wakes_cleared')
                assert_wake_list(section, changed)
                assert not errors
            finally:
                browser.close()
    publish_fact(name, {'old_same_list_http_save': True, 'choice': choice, 'reopened': True,
                        'removed_saved': len(expected), 'cleared': True, 'declared_additions_only': True})
    publish_step(name, 'fact_published')


@pytest.mark.parametrize('engine,width,height,theme', MATRIX, ids=['chromium', 'webkit'])
def test_place_sheet_pending_unpublished_signal(tmp_path, engine, width, height, theme):
    name = f'place-pending-{engine}'
    publish_step(name, 'case_started', viewport_width=width, viewport_height=height)
    with live(tmp_path / 'candidate') as server:
        pending(server.data)
        typed(server.data, inbox=True, enabled=False)
        server.start()
        publish_step(name, 'server_ready')
        path = places.places_path(server.data)
        digest = sha(path)
        initial = get_row(server)
        declared = {entry if isinstance(entry, str) else entry['id'] for entry in initial['facts']}
        assert initial['wake_review'] == 'pending'
        assert initial['readiness']['text'] == 'Off in Skills'
        assert declared == {DIRECT, INBOX}
        assert any(isinstance(entry, dict) and entry.get('id') == INBOX and entry.get('label') == 'Priority inbox'
                   for entry in initial['facts'])
        assert initial['wake_suggestions'] == [DIRECT, INBOX]
        assert initial['unpublished_signals'] == [INBOX]
        assert wakes(initial) == LEGACY
        with sync_playwright() as playwright:
            browser = getattr(playwright, engine).launch(headless=True)
            try:
                page, errors = browser_page(browser, server.url, width, height, theme)
                writes = []
                page.on('request', lambda request: writes.append(True)
                        if request.method in {'POST', 'PUT', 'PATCH', 'DELETE'}
                        and urlsplit(request.url).path.startswith('/api/places') else None)
                page.goto(server.url, wait_until='domcontentloaded')
                go(page, 'circles', width)
                row = page.locator(f'#page-circles button[data-place-open="{LEGACY_ID}"]')
                sheet = page.locator('dialog.place-sheet:visible')
                section = sheet.locator('[data-place-section="wakes"]')
                row.click()
                expect(section).to_be_visible()
                publish_step(name, 'sheet_open')
                expect(page.get_by_role('dialog', name=re.compile(r'.+'))).to_have_count(1)
                expect(section).to_contain_text('Once the connector supplies its suggested kinds')
                expect(section).not_to_contain_text('This older connection')
                actual = section.locator('input[data-fact]').evaluate_all('nodes => nodes.map(n => n.dataset.fact)')
                assert sorted(actual) == sorted([LEGACY[0], DIRECT, INBOX])
                expect(section.get_by_role('checkbox')).to_have_count(3)
                legacy_box = wake_box(section, LEGACY[0])
                expect(legacy_box).to_be_visible()
                expect(legacy_box).to_be_checked()
                expect(legacy_box).to_be_enabled()
                assert legacy_box.get_attribute('data-place-act') == 'remove-wake'
                expect(section.get_by_role('checkbox', name=WAKE_NAMES[LEGACY[0]])).to_have_count(1)
                dm_box = wake_box(section, DIRECT)
                expect(dm_box).to_be_visible()
                expect(dm_box).not_to_be_checked()
                expect(dm_box).to_be_enabled()
                assert dm_box.get_attribute('data-place-act') == 'add-wake'
                assert dm_box.get_attribute('data-disabled') is None
                expect(section.get_by_role('checkbox', name=WAKE_NAMES[DIRECT])).to_have_count(1)
                expect(section.locator(f'li.place-wake-item:has(input[data-fact="{DIRECT}"]) [data-place-waiting]')).to_have_count(0)
                signal_box = wake_box(section, INBOX)
                expect(signal_box).to_be_visible()
                expect(signal_box).not_to_be_checked()
                expect(signal_box).to_be_disabled()
                assert signal_box.get_attribute('data-place-act') == 'add-wake'
                assert signal_box.get_attribute('data-disabled') is not None
                signal_by_role = section.get_by_role('checkbox', name=PENDING_SIGNAL_NAME)
                expect(signal_by_role).to_have_count(1)
                assert signal_by_role.get_attribute('data-fact') == INBOX
                expect(signal_by_role).to_be_disabled()
                waiting = section.locator(f'li.place-wake-item:has(input[data-fact="{INBOX}"]) [data-place-waiting]')
                expect(waiting).to_have_count(1)
                expect(waiting).to_be_visible()
                expect(waiting).to_contain_text('once the connector is on')
                expect(section.locator('[data-place-waiting]')).to_have_count(1)
                add = section.get_by_role('button', name=ADD_NAME)
                keep = section.get_by_role('button', name=KEEP_NAME)
                expect(add).to_have_count(1)
                assert add.get_attribute('data-place-act') == 'add-suggestions'
                expect(add).to_be_disabled()
                assert add.get_attribute('data-disabled') is not None
                expect(keep).to_have_count(1)
                assert keep.get_attribute('data-place-act') == 'keep-wakes'
                expect(keep).to_be_enabled()
                screenshot(page, section, name + '-wakes-pending', case=name)
                publish_step(name, 'wakes_pending_captured')
                after = get_row(server)
                assert after['revision'] == initial['revision'] and wakes(after) == LEGACY
                assert after['wake_review'] == 'pending' and after['unpublished_signals'] == [INBOX]
                assert sha(path) == digest and not writes and not errors
            finally:
                browser.close()
    publish_fact(name, {'wake_review': 'pending', 'readiness': 'off_in_skills',
                        'offered_facts': [DIRECT, INBOX], 'unpublished_signals': [INBOX],
                        'custom_signal_disabled': True, 'custom_signal_waiting': True,
                        'core_direct_enabled': True, 'saved_legacy_removable': True,
                        'bulk_add_disabled': True, 'keep_enabled': True,
                        'place_writes': 0, 'places_unchanged': True})
    publish_step(name, 'fact_published')
