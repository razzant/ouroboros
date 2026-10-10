"""The real Accounts module keeps stop/memory facts legible on desktop and phone."""
from pathlib import Path

import pytest

pytest_plugins = ('tests.test_ui_smoke_playwright',)


@pytest.mark.ui_browser
@pytest.mark.serial
def test_engine_facts_status_line_at_both_widths(direct_server_with_data, tmp_path):
    from playwright.sync_api import sync_playwright

    payload = {
        'daemon': {'state': 'running', 'engine_version': '3.22.1', 'runtime': {'state': 'ready'},
                   'last_exit': {'classification': 'heap_exhausted', 'phase': 'serving', 'exit_signal': 6,
                                 'engine_version': '3.22.0', 'observed_at': '2026-10-07T19:34:11Z'},
                   'memory': {'heapUsedBytes': 3 * 2**30, 'heapLimitBytes': 16 * 2**30}},
        'config_dir': '/temporary-test-root/claudexor', 'harnesses': [],
        'profiles': {'profiles': []}, 'quota': [], 'quota_absences': [],
        'reads': {'catalog': 'ok', 'accounts': 'ok', 'quota': 'ok'},
    }
    # A system browser binary uses a fresh Playwright-owned temporary profile.
    # It reads no operator Chrome profile; other platforms use the test browser.
    chrome = Path('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome')
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(**({'executable_path': str(chrome)} if chrome.is_file() else {}))
        try:
            for name, width in [('desktop', 1440), ('phone', 390)]:
                page = browser.new_page(viewport={'width': width, 'height': 1000})
                page.route('**/api/claudexor/status**', lambda route: route.fulfill(json=payload))
                page.goto(direct_server_with_data['url'] + '/#settings', wait_until='domcontentloaded')
                page.wait_for_selector('.settings-shell')
                page.click('[data-settings-tab="providers"]')
                line = page.locator('#agents-service-banner')
                line.wait_for(state='visible')
                assert 'Last stop: heap exhausted while serving (signal 6)' in line.inner_text()
                assert 'headroom 13.0 GiB' in line.inner_text()
                assert 'engine 3.22.0, seen 2026-10-07 19:34 UTC' in line.inner_text()
                assert line.evaluate('(el) => el.scrollWidth <= el.clientWidth + 1')
                line.scroll_into_view_if_needed()
                destination = tmp_path / f'engine-facts-{name}.png'
                page.screenshot(path=str(destination))
                print(f'UI_ENGINE_EVIDENCE {destination}')
                page.close()
        finally:
            browser.close()
