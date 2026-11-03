"""The Accounts module keeps engine measurements and recovery legible at both widths."""
from pathlib import Path

import pytest
from ouroboros.test_evidence import diagnostic_directory

pytest_plugins = ('tests.test_ui_smoke_playwright',)


@pytest.mark.ui_browser
@pytest.mark.serial
def test_engine_facts_status_line_at_both_widths(direct_server_with_data, tmp_path):
    from playwright.sync_api import sync_playwright

    payload = {
        'daemon': {'state': 'running', 'engine_version': '4.0.0', 'runtime': {'state': 'ready'},
                   'servingMode': 'normal',
                   'last_exit': {'classification': 'heap_exhausted', 'phase': 'serving', 'exit_signal': 6,
                                 'engine_version': '3.22.0', 'observed_at': '2026-10-07T19:34:11Z'},
                   'memory': {'heapUsedBytes': 3 * 2**30, 'heapLimitBytes': 16 * 2**30},
                   'loop': {'windowMs': 10004, 'delay': {'p99Ms': 3.4, 'maxMs': 9}, 'utilization': 0.25},
                   'store': {'integrity': 'ok', 'flusher': {'state': 'up'}, 'flush_lag_ms': 7,
                             'wal_bytes': 2**20, 'obligations_open': 0, 'migration': None},
                   'capacity': {'maxConcurrent': 'unlimited', 'maxConcurrentModelOperations': 8,
                                'maxConcurrentNonModelJobs': 'unlimited', 'sources': {
                                    'max_concurrent': 'default', 'max_concurrent_model_operations': 'config',
                                    'max_concurrent_non_model_jobs': 'environment'}},
                   'admission': {'active': {'model': 1, 'non_model': 24}, 'queued': {'model': 0, 'non_model': 3}}},
        'config_dir': '/temporary-test-root/claudexor',
        'harnesses': [{'id': 'claude', 'display_name': 'Claude Code', 'enabled': True, 'status': 'ok'}],
        'profiles': {'profiles': [{'profile': {'harness_id': 'claude', 'profile_id': 'kept-account', 'enabled': True},
                                  'status': {'verification': 'passed', 'verification_source': 'vendor'}}]},
        'quota': [], 'quota_absences': [], 'unified_accounts': True,
        'reads': {'catalog': 'ok', 'accounts': 'ok', 'quota': 'ok'},
    }
    # A system browser binary uses a fresh Playwright-owned temporary profile.
    # It reads no operator Chrome profile; other platforms use the test browser.
    chrome = Path('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome')
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(**({'executable_path': str(chrome)} if chrome.is_file() else {}))
        try:
            for name, width in [('desktop', 1440), ('phone', 390), ('recovery-desktop', 1440), ('recovery-phone', 390)]:
                if name.startswith('recovery'):
                    payload['daemon']['servingMode'] = 'recovery_only'
                    payload['daemon']['store'] = {'integrity': None, 'flusher': None, 'flush_lag_ms': None,
                        'wal_bytes': None, 'obligations_open': None, 'migration': {
                            'phase': 'verifying', 'completedPartitions': 2, 'totalPartitions': 3,
                            'processedBytes': 2**20, 'totalBytes': 2**21, 'currentPartition': 'project-fixture'}}
                    payload['reads']['accounts'] = 'failed'
                    payload['facets'] = {'accounts': {'stale': True, 'observed_at': '2026-10-11T01:02:03Z',
                                                      'error': 'daemon_recovery_only'}}
                page = browser.new_page(viewport={'width': width, 'height': 1000})
                page.route('**/api/claudexor/status**', lambda route: route.fulfill(json=payload))
                page.goto(direct_server_with_data['url'] + '/#settings', wait_until='domcontentloaded')
                page.wait_for_selector('.settings-shell')
                page.click('[data-settings-tab="providers"]')
                line = page.locator('#agents-service-banner')
                line.wait_for(state='visible')
                page.locator('.harness-account-row[data-profile="kept-account"]').wait_for(state='visible')
                toggle = page.locator('#agents-service-diagnostics-toggle')
                diagnostics = page.locator('#agents-service-diagnostics')
                # Refined A (owner 21.10): the verdict and the engine version are on the line; the measurements wait
                # under Diagnostics, which opens by pointer and stays open, focused, across a status refresh.
                assert diagnostics.is_hidden() and toggle.get_attribute('aria-expanded') == 'false'
                toggle.click()
                diagnostics.wait_for(state='visible')
                page.evaluate("""async () => {
                    const {claudexorStatus} = await import('/static/modules/claudexor_status_store.js');
                    await claudexorStatus.refresh();
                }""")
                assert diagnostics.is_visible() and toggle.get_attribute('aria-expanded') == 'true'
                assert toggle.evaluate('(el) => el === document.activeElement')
                measured = diagnostics.inner_text()
                assert 'headroom 13.0 GiB' in measured and 'p99 3.4 ms' in measured
                assert 'total unlimited (default), model 8 (config)' in measured
                assert '/temporary-test-root/claudexor' in measured
                assert 'Loop' not in line.inner_text() and 'headroom' not in line.inner_text()
                if name.startswith('recovery'):
                    # Not ready: what blocks work, and why the engine stopped, stay on the line.
                    assert 'Recovery only; product work is not admitted' in line.inner_text()
                    assert 'Import verifying: 2/3 partitions, 1.0/2.0 MiB' in line.inner_text()
                    assert 'accounts' in line.inner_text().lower() and 'could not be read' in line.inner_text()
                    assert 'Last stop: heap exhausted while serving (signal 6)' in line.inner_text()
                    assert 'integrity unknown' in measured
                else:
                    assert line.inner_text().strip() == 'Claudexor ready'
                    assert page.locator('#agents-service-version').inner_text() == 'engine 4.0.0'
                    # A stop the ready engine recovered from is a diagnostic, with its own engine and time.
                    assert 'heap exhausted while serving (signal 6) · engine 3.22.0, seen 2026-10-07 19:34 UTC' in measured
                    assert 'integrity ok, flusher up' in measured
                assert page.locator('.agents-service').evaluate('(el) => el.scrollWidth <= el.clientWidth + 1')
                line.scroll_into_view_if_needed()
                output = diagnostic_directory(tmp_path, "engine-facts")
                destination = output / f'engine-facts-{name}.png'
                page.screenshot(path=str(destination))
                print(f'UI_ENGINE_EVIDENCE {destination}')
                line.screenshot(path=str(output / f'engine-line-{name}.png'))
                page.close()
        finally:
            browser.close()
