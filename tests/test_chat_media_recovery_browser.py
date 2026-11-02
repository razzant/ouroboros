"""Real decoder failures recovered through existing chat controls, without page reload."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests._media_recovery_harness import media_server

pytestmark = pytest.mark.ui_browser


@pytest.mark.parametrize('browser_name', ['chromium', 'webkit'])
@pytest.mark.parametrize('action', ['play', 'surface', 'repeat', 'failed-press'])
def test_failed_video_reloads_only_on_deliberate_play(browser_name, action):
    if os.environ.get('OUROBOROS_RUN_UI_SMOKE') != '1':
        pytest.skip('set OUROBOROS_RUN_UI_SMOKE=1')
    from playwright.sync_api import sync_playwright

    with media_server() as (url, state), sync_playwright() as pw:
        browser = getattr(pw, browser_name).launch(headless=True)
        try:
            page = browser.new_page(viewport={'width': 1100, 'height': 750})
            page.set_default_timeout(6000)
            page.goto(url, wait_until='domcontentloaded')
            page.wait_for_function('window.media?.error')
            before = page.evaluate('mediaFacts()')
            assert before['error'] == 4
            assert before['paused'] is True
            assert before['muted'] is False
            page.evaluate("""() => {
                window.keptMedia = media;
                window.loadCalls = 0;
                const nativeLoad = media.load.bind(media);
                media.load = () => { loadCalls++; return nativeLoad(); };
            }""")
            page.locator('[data-media-action="speed"]').click()
            assert page.evaluate('media.playbackRate') == 1.5
            if action == 'failed-press':
                page.locator('[data-media-action="play"]').click()
                # A real failed play settles before we restore the source. In WebKit
                # it can leave paused=false; the next gesture must still recover.
                page.get_by_text('Could not play media:', exact=False).wait_for()
                failed_press = page.evaluate('mediaFacts()')
                assert failed_press['error'] == 4
                assert page.evaluate('loadCalls') == 1
            state['available'] = True
            if action == 'surface':
                page.locator('video').click()
            else:
                control = 'play' if action == 'failed-press' else action
                page.locator(f'[data-media-action="{control}"]').click()
            page.wait_for_function('!media.error && !media.paused && media.currentTime > 0.25')
            after = page.evaluate('mediaFacts()')
            assert after['rate'] == 1.5
            assert after['muted'] is False
            expected_loads = 2 if action == 'failed-press' else 1
            assert page.evaluate('media === keptMedia && media.videoWidth === 160')
            assert page.evaluate('loadCalls') == expected_loads
            assert any(click['trusted'] for click in page.evaluate('clicks'))
            assert any(request['available'] for request in state['requests'])
            page.locator('[data-media-action="play"]').click()
            page.wait_for_function('media.paused')
            page.locator('[data-media-action="play"]').click()
            page.wait_for_function('!media.paused')
            assert page.evaluate('loadCalls') == expected_loads, 'healthy pause/resume must not reload the source'
            if directory := os.environ.get('OUROBOROS_UI_EVIDENCE_OUT'):
                target = Path(directory)
                target.mkdir(parents=True, exist_ok=True)
                (target / f'media-{browser_name}-{action}.json').write_text(json.dumps({
                    'before': before, 'after': after, 'observations': page.evaluate('observations'),
                    'clicks': page.evaluate('clicks'), 'source_requests': state['requests'],
                }, indent=2), encoding='utf-8')
                page.screenshot(path=str(target / f'media-{browser_name}-{action}.png'), animations='disabled')
            page.evaluate('releasePlayer()')
            assert page.evaluate('!media.isConnected && media.paused')
            if action != 'failed-press':
                assert page.locator('.toast').count() == 0
        finally:
            browser.close()
