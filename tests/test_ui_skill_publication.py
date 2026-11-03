"""Real Skills card geometry and the local publication record against an isolated server:
listings, receipts and Forget this record are the real backend (no catalog or source read is
needed by Installed). Forgetting goes through `POST /api/skills/{skill}/publication/clear`, the
skill-sources gateway's route."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

from ouroboros.marketplace.provenance import read_publication_record, write_publication_record, merge_state_record
from ouroboros.skill_loader import SkillReviewState, compute_content_hash, save_review_state, save_enabled
from tests.test_ui_smoke_playwright import direct_server_with_data as _direct_server_with_data

pytestmark = pytest.mark.ui_browser
direct_server_with_data = _direct_server_with_data


def _skill(data: Path, name: str, *, reviewed=True, grant=False, version='2.0.0', bucket='external') -> Path:
    payload = data / 'skills' / bucket / name
    (payload / 'scripts').mkdir(parents=True)
    if bucket == 'ouroboroshub':
        (payload / '.ouroboroshub.json').write_text(json.dumps(
            {'schema_version': 1, 'source': 'ouroboroshub', 'slug': name, 'sanitized_name': name}))
    (payload / 'SKILL.md').write_text(
        f'---\nname: {name}\ndescription: Publication layout fixture\nversion: "{version}"\n'
        'type: script\nruntime: python3\nscripts:\n  - name: check.py\n    description: Fixture\n'
        + ('env_from_settings: [OPENROUTER_API_KEY]\n' if grant else '') + '---\n# Fixture\n')
    (payload / 'scripts/check.py').write_text("print('fixture')\n")
    if reviewed:
        save_review_state(data, name, SkillReviewState(status='pass', content_hash=compute_content_hash(payload)))
    save_enabled(data, name, False)
    return payload


def _screenshot(page, name):
    directory = os.environ.get('OUROBOROS_PUBLICATION_SCREENSHOTS')
    if directory:
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(output / name))


@pytest.mark.parametrize('engine', ['chromium', 'webkit'])
def test_skills_card_geometry_tracks_card_width_and_keeps_grant_on_card(direct_server_with_data, engine):
    from playwright.sync_api import sync_playwright, expect

    data, url = direct_server_with_data['data_dir'], direct_server_with_data['url']
    names = ['publication_pending_with_a_long_but_valid_skill_name',
             'publication_grant_access_with_a_very_long_skill_name',
             'publication_ready_with_a_long_but_valid_skill_name']
    _skill(data, names[0], reviewed=False)
    _skill(data, names[1], grant=True)
    _skill(data, names[2])
    with sync_playwright() as playwright:
        browser = getattr(playwright, engine).launch(headless=True)
        try:
            # Docked, like the first measured width: below 720 the sidebar is an off-screen drawer.
            page = browser.new_page(viewport={'width': 720, 'height': 900})
            page.goto(url, wait_until='domcontentloaded')
            page.click('[data-nav-page="skills"]')
            page.wait_for_selector(f'.skills-card[data-skill="{names[1]}"]')
            measurements = []
            for viewport, sidebar in [(720, 280), (1100, 560), (1440, 280)]:  # 720 = the docked layout's first width (layout.js)
                page.set_viewport_size({'width': viewport, 'height': 900})
                page.evaluate("width => document.getElementById('app').style.setProperty('--sidebar-width', `${width}px`)", sidebar)
                for name in names:
                    card = page.locator(f'.skills-card[data-skill="{name}"]')
                    card.scroll_into_view_if_needed()
                    facts = card.evaluate('''card => {
                        const rect = card.getBoundingClientRect();
                        const title = card.querySelector('.skills-card-title').getBoundingClientRect();
                        const actions = [...card.querySelectorAll('.skills-switch, .skills-card-menu-trigger, .skills-tray-actions button')]
                            .map(control => ({text:control.textContent.trim(), x:control.getBoundingClientRect().x,
                                right:control.getBoundingClientRect().right}));
                        return {card:rect.width, title:title.width, x:rect.x, right:rect.right, actions,
                            direction:getComputedStyle(card.querySelector('.skills-card-head')).flexDirection,
                            sidebar:document.getElementById('primary-sidebar').getBoundingClientRect().width};
                    }''')
                    assert abs(facts['sidebar'] - sidebar) <= 2, facts
                    assert facts['title'] >= 180, facts
                    assert facts['actions'] and all(action['x'] >= facts['x'] and action['right'] <= facts['right']
                                                    for action in facts['actions']), facts
                    assert name != names[1] or 'Grant access' in [action['text'] for action in facts['actions']], facts
                    assert facts['direction'] == ('column' if viewport != 1440 else 'row'), facts
                    measurements.append({'viewport': viewport, 'name': name, **facts})
                grant = page.locator(f'.skills-card[data-skill="{names[1]}"]')
                grant.scroll_into_view_if_needed()
                expect(grant.get_by_role('button', name='Grant access', exact=True)).to_be_visible()
                _screenshot(page, f'{engine}-skills-width-{viewport}-sidebar-{sidebar}.png')
            grant.get_by_role('button', name='Grant access', exact=True).click()
            expect(page.locator('.confirm-dialog')).to_contain_text(f'Grant access to {names[1]}')
            _screenshot(page, f'{engine}-grant-access-dialog.png')
            page.locator('.confirm-dialog [data-confirm-cancel]').last.click()
            if directory := os.environ.get('OUROBOROS_PUBLICATION_SCREENSHOTS'):
                (Path(directory) / f'{engine}-geometry.json').write_text(json.dumps(measurements, indent=2))
        finally:
            browser.close()


def _receipt(data: Path, name: str, version: str, content_hash: str, number: int = 60, url: str = ''):
    record = {'slug': name, 'version': version, 'content_hash': content_hash,
              'repository': 'hub/project', 'pr_number': number,
              'pr_url': url or f'https://github.com/hub/project/pull/{number}', 'published_at': '2026-09-06T00:00:00Z'}
    write_publication_record(data, name, record)
    return record


@pytest.mark.parametrize('bucket', ['external', 'ouroboroshub'])
def test_forget_publication_record_clears_only_the_record_shown(direct_server_with_data, bucket):
    from playwright.sync_api import sync_playwright, expect

    data, url = direct_server_with_data['data_dir'], direct_server_with_data['url']
    name = 'publication_waiting_fixture'
    payload = _skill(data, name, bucket=bucket)
    original_hash = compute_content_hash(payload)
    first = _receipt(data, name, '2.0.0', original_hash, 7)
    merge_state_record(data, name, 'ouroboroshub.json', {'future': {'keep': True}})
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            page = browser.new_page(viewport={'width': 1100, 'height': 900})
            external = []
            page.route('https://github.com/**', lambda route: (external.append(route.request.url), route.abort()))
            page.goto(url, wait_until='domcontentloaded')
            page.click('[data-nav-page="skills"]')
            own = page.locator(f'.skills-card[data-skill="{name}"]')
            own.locator('.skills-details > summary').click()
            # The row labelled exactly «Publication» (a folder path may contain the word).
            label = page.locator('.skills-detail-label', has_text=re.compile(r'^Publication$'))
            record = own.locator('.skills-detail-row').filter(has=label)
            expect(record).to_contain_text('Submitted v2.0.0 · PR #7')
            expect(own.locator('.skills-card-title')).not_to_contain_text('Submitted')
            assert record.locator('a').get_attribute('href') == 'https://github.com/hub/project/pull/7'
            _screenshot(page, f'publication-record-{bucket}.png')

            # The record on screen cannot clear a newer one written meanwhile: the host refuses.
            newer = _receipt(data, name, '2.0.0', original_hash, 8)
            dialog = page.locator('.confirm-dialog')
            record.locator('[data-skill-action="forget_publication"]').click()
            expect(dialog).to_contain_text(f'Forget the publication record of {name}?')
            dialog.locator('[data-confirm-ok]').click()
            expect(page.locator('.toast').last).to_contain_text(name)
            assert read_publication_record(data, name)[0] == newer != first
            expect(record).to_contain_text('PR #8')

            record.locator('[data-skill-action="forget_publication"]').click()
            dialog.locator('[data-confirm-ok]').click()
            expect(page.locator('.toast', has_text='publication record forgotten')).to_have_count(1)
            assert read_publication_record(data, name) == (None, None)
            assert json.loads((data / f'state/skills/{name}/ouroboroshub.json').read_text())['future'] == {'keep': True}
            expect(record).to_have_count(0)
            expect(own.locator('.skills-details')).to_have_attribute('open', '')
            assert compute_content_hash(payload) == original_hash
            assert not external
        finally:
            browser.close()
