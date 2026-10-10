"""Actual Settings and Agents (reviewer rows included) flows with local gateway-backed previews."""
from urllib.parse import urlsplit

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from tests import test_subscription_role_routes_browser as roles

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]
subscription_ui = roles.subscription_ui
role_ui = roles.role_ui


@pytest.fixture
def route_ui(role_ui, tmp_path, monkeypatch):
    from ouroboros.gateway import settings as gateway
    ui = role_ui
    roles.configure_mixed(ui)
    ui['settings'].update(OPENROUTER_API_KEY='fixture-key', OUROBOROS_WEBSEARCH_BACKEND='auto',
                          OUROBOROS_WEBSEARCH_MODEL='gpt-5.2', OUROBOROS_MODEL='openai-compatible::future',
                          OPENAI_COMPATIBLE_BASE_URL='https://compatible.invalid/v1', OPENAI_COMPATIBLE_API_KEY='fixture-key')
    monkeypatch.setattr(gateway, 'load_settings', lambda: ui['settings'])
    monkeypatch.setattr(gateway, 'request_drive_root', lambda _request: tmp_path)
    monkeypatch.setattr(gateway, '_owner_audit', lambda *a: None)
    app = Starlette(routes=[Route('/api/settings', gateway.api_settings_get),
        Route('/api/owner/capability-ack', gateway.api_acknowledge_capability, methods=['POST'])])
    with TestClient(app) as client:
        def bridge(route):
            url = urlsplit(route.request.url)
            path = url.path + ('?' + url.query if url.query else '')
            response = (client.post(path, json=route.request.post_data_json) if route.request.method == 'POST' else client.get(path))
            route.fulfill(status=response.status_code, content_type='application/json', body=response.text)
        ui['page'].route('**/api/settings?*', bridge)
        ui['page'].route('**/api/owner/capability-ack', bridge)
        yield ui


def test_settings_search_and_exact_response_maximum(route_ui):
    page = route_ui['page']
    page.goto(route_ui['url'] + '/#settings')
    page.locator('[data-settings-tab="models"]').click()
    page.wait_for_function("document.querySelector('#s-websearch-preview').textContent.includes('openrouter')")
    assert page.locator('#s-websearch-model').input_value() == 'gpt-5.2'
    page.locator('#s-websearch-source').select_option('openrouter')
    page.locator('#s-websearch-model').fill('vendor/future-search')
    page.locator('#s-websearch-model').blur()
    page.wait_for_function("document.querySelector('#s-websearch-preview').textContent.includes('vendor/future-search')")
    main = page.locator('[data-model-role="main"]')
    main.locator('summary').click()
    maximum = main.locator('[data-response-limit]')
    maximum.fill('4096')
    with page.expect_response('**/api/owner/capability-ack') as saved:
        main.locator('[data-response-apply]').click()
    assert saved.value.status == 200, saved.value.text()
    assert saved.value.json()['ack']['max_output_tokens'] == 4096
    page.wait_for_function("document.querySelector('[data-model-role=main] [data-response-note]').textContent.includes('4,096')")
    roles.capture(page, 'model-routes-settings-maximum')
    page.set_viewport_size({'width': 390, 'height': 844})
    maximum.scroll_into_view_if_needed()
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    roles.capture(page, 'model-routes-maximum-mobile')
    page.set_viewport_size({'width': 1360, 'height': 900})
    main.locator('[data-model-role-model]').fill('other-model')
    assert maximum.input_value() == ''
    main.locator('summary').click()
    main.locator('summary').click()
    page.wait_for_function("document.querySelector('[data-model-role=main] [data-response-note]').textContent.includes('unknown')")
    assert maximum.input_value() == ''
    page.set_viewport_size({'width': 390, 'height': 844})
    page.locator('#s-websearch-model').scroll_into_view_if_needed()
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    roles.capture(page, 'model-routes-search-mobile')
    page.locator('#btn-save-settings').click()
    page.wait_for_function("() => !document.querySelector('#btn-save-settings').disabled")
    writes = [body for path, body in route_ui['posts'] if path == '/api/settings']
    assert writes[-1]['OUROBOROS_WEBSEARCH_BACKEND'] == 'openrouter'
    assert writes[-1]['OUROBOROS_WEBSEARCH_MODEL'] == 'vendor/future-search'
    page.locator('#btn-reload-settings').click()
    assert page.locator('#s-websearch-model').input_value() == 'vendor/future-search'
    assert not route_ui['errors']


@pytest.mark.parametrize('edit_order', ['before_toggle', 'pending_preview', 'reopen'])
def test_response_maximum_preview_keeps_owner_draft(route_ui, edit_order):
    from playwright.sync_api import expect

    page = route_ui['page']
    page.goto(route_ui['url'] + '/#settings')
    page.locator('[data-settings-tab="models"]').click()
    main = page.locator('[data-model-role="main"]')
    maximum = main.locator('[data-response-limit]')
    note = main.locator('[data-response-note]')
    if edit_order == 'before_toggle':
        # Native details queues its toggle event. Exercise an input arriving
        # before that event starts the preview, without depending on timing.
        main.evaluate("""row => {
            row.querySelector('summary').click();
            const field = row.querySelector('[data-response-limit]');
            field.value = '4096';
            field.dispatchEvent(new Event('input', {bubbles: true}));
        }""")
    elif edit_order == 'pending_preview':
        page.evaluate("""() => {
            const fetch = window.fetch;
            window.fetch = async (...args) => {
                if (String(args[0]).includes('response_limit_preview=1')) {
                    await new Promise(resolve => { window.releaseMaximumPreview = resolve; });
                }
                return fetch(...args);
            };
            window.restoreMaximumFetch = () => { window.fetch = fetch; };
        }""")
        main.locator('summary').click()
        page.wait_for_function('Boolean(window.releaseMaximumPreview)')
        maximum.fill('4096')
        page.evaluate('() => { window.restoreMaximumFetch(); window.releaseMaximumPreview(); }')
    else:
        main.locator('summary').click()
        expect(note).to_contain_text('Auto: maximum response unknown')
        maximum.fill('4096')
        main.locator('summary').click()
        note.evaluate("node => { node.textContent = ''; }")
        main.locator('summary').click()
    expect(note).to_contain_text('Auto: maximum response unknown')
    with page.expect_response('**/api/owner/capability-ack') as saved:
        main.locator('[data-response-apply]').click()
    assert saved.value.json()['ack']['max_output_tokens'] == 4096
    expect(maximum).to_have_value('4096')
    expect(note).to_contain_text('4,096 tokens · your maximum')
    roles.capture(page, f'maximum-preview-draft-{edit_order}')
    # A fresh, untouched control still reads the persisted owner maximum.
    page.reload()
    page.locator('[data-settings-tab="models"]').click()
    main.locator('summary').click()
    expect(maximum).to_have_value('4096')
    expect(note).to_contain_text('4,096 tokens · your maximum')
    # Clearing to Auto is an edit too; a new preview must not restore the ack.
    maximum.fill('')
    main.locator('summary').click()
    note.evaluate("node => { node.textContent = ''; }")
    main.locator('summary').click()
    expect(note).to_contain_text('4,096 tokens · your maximum')
    with page.expect_response('**/api/owner/capability-ack') as cleared:
        main.locator('[data-response-apply]').click()
    assert cleared.value.json()['ack']['max_output_tokens'] == 0
    expect(maximum).to_have_value('')
    # Changing the exact route clears its draft and reads that route's facts.
    maximum.fill('2048')
    main.locator('[data-model-role-model]').fill('other-model')
    expect(maximum).to_have_value('')
    main.locator('summary').click()
    main.locator('summary').click()
    expect(note).to_contain_text('Auto: maximum response unknown')
    expect(maximum).to_have_value('')
    assert not route_ui['errors']


def test_response_maximum_deferred_toggle_cannot_cancel_apply(route_ui):
    from playwright.sync_api import expect

    page = route_ui['page']
    page.goto(route_ui['url'] + '/#settings')
    page.locator('[data-settings-tab="models"]').click()
    main = page.locator('[data-model-role="main"]')
    with page.expect_response('**/api/owner/capability-ack', timeout=5000) as saved:
        main.evaluate("""row => {
            row.querySelector('summary').click();
            const field = row.querySelector('[data-response-limit]');
            field.value = '4096';
            field.dispatchEvent(new Event('input', {bubbles: true}));
            row.querySelector('[data-response-apply]').click();
        }""")
    assert saved.value.json()['ack']['max_output_tokens'] == 4096
    expect(main.locator('[data-response-note]')).to_contain_text('4,096 tokens · your maximum')
    assert not route_ui['errors']


def test_claude_variant_agent_and_reviewer_rows_keep_exact_selection(route_ui):
    ui = route_ui
    target = 'claude=claude-future[1M]'
    for entry in ui['fixture']['status']['profiles']['profiles']:
        entry['profile']['harness_id'] = 'claude'
    ui['fixture']['status']['harnesses'] = [{'id': 'claude', 'display_name': 'Claude Code', 'enabled': True,
        'status': 'ok', 'models': [{'id': 'claude-future', 'credential_profile_id': 'personal'}]}]
    ui['fixture']['status']['quota'] = [{'subject': {'harness': 'claude', 'subject_id': 'personal'},
                                        'freshness': 'fresh', 'constraints': []}]
    items = ui['settings']['OUROBOROS_SUBAGENTS']['items']
    # A reviewer is a catalog row marked Reviewer; an unmarked twin is an ordinary agent.
    items[2].update(review_eligible=True, route={'kind': 'agent_session', 'target_id': target,
                                                 'credential_profile_id': 'personal'})
    items.append({'subagent_id': 'claude-agent', 'recommended_use': 'Long-context agent.', 'effort': 'max',
                  'route': {'kind': 'agent_session', 'target_id': target}})
    page = roles.open_agents(ui)
    rows = page.locator('[data-subagent-row]')
    page.wait_for_function("() => [...document.querySelectorAll('[data-subagent-row]')]"
                           ".filter(x => x.textContent.includes('Base model listed')).length === 2")
    for index, reviews in ((2, True), (3, False)):
        row = rows.nth(index)
        assert row.locator('[data-subagent-field="model"]').input_value() == 'claude-future[1M]'
        assert row.locator('[data-subagent-field="review_eligible"]').is_checked() is reviews
        assert 'Available' in row.inner_text() and 'Unavailable' not in row.inner_text()
    rows.nth(2).scroll_into_view_if_needed()
    roles.capture(page, 'model-routes-claude-reviewer-row')
    page.set_viewport_size({'width': 390, 'height': 844})
    rows.nth(3).scroll_into_view_if_needed()
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    # The conditional availability stays readable on a phone: natural height, nothing clipped.
    meta = rows.nth(3).locator('[data-subagent-meta]')
    assert meta.get_attribute('data-availability-qualifier') is not None
    assert 'checked by the engine at session start' in meta.inner_text()
    assert meta.evaluate('el => el.scrollWidth <= el.clientWidth && el.scrollHeight <= el.clientHeight')
    assert meta.evaluate('el => el.getBoundingClientRect().height > 1.5 * parseFloat(getComputedStyle(el).lineHeight)')
    assert rows.nth(1).locator('[data-subagent-meta]').get_attribute('data-availability-qualifier') is None
    roles.capture(page, 'model-routes-claude-agent-mobile')
    # An unrelated edit saves the catalog; neither selected spelling is rewritten.
    rows.nth(3).locator('[data-subagent-field="recommended_use"]').fill('Long-context agent, edited.')
    page.locator('#btn-save-settings').click()
    page.wait_for_function("() => !document.querySelector('#btn-save-settings').disabled")
    saved = roles.saved_catalog(ui)['items']
    assert [row['route']['target_id'] for row in saved[2:]] == [target, target]
    assert saved[2]['review_eligible'] is True and 'review_eligible' not in saved[3]
    assert not ui['errors']
