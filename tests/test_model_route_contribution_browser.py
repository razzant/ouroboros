"""Actual Settings and Agents (reviewer rows included) flows with local gateway-backed previews."""
from urllib.parse import urlsplit

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from tests import test_subscription_role_routes_browser as roles
from tests.settings_browser_helpers import open_settings, press_editor_save

pytestmark = [pytest.mark.ui_browser]
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
    monkeypatch.setattr(gateway, 'settings_read_view', lambda: (ui['settings'], {}))
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
    open_settings(page, 'models/role/websearch')
    page.wait_for_function("document.querySelector('[data-websearch-preview]').textContent.includes('OpenRouter')")
    assert page.locator('#s-websearch-model').input_value() == 'gpt-5.2'
    page.locator('#s-websearch-source').select_option('openrouter')
    page.locator('#s-websearch-model').fill('vendor/future-search')
    page.locator('#s-websearch-model').blur()
    page.wait_for_function("document.querySelector('[data-websearch-preview]').textContent.includes('vendor/future-search')")
    assert press_editor_save(page).startswith('Saved')
    open_settings(page, 'models/role/main')
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
    main.locator('[data-model-role-model]').press('Escape')
    press_editor_save(page)
    open_settings(page, 'models/role/websearch')
    page.locator('#s-websearch-model').scroll_into_view_if_needed()
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    roles.capture(page, 'model-routes-search-mobile')
    writes = [body for path, body in route_ui['posts'] if path == '/api/settings']
    assert writes[-1]['OUROBOROS_WEBSEARCH_BACKEND'] == 'openrouter'
    assert writes[-1]['OUROBOROS_WEBSEARCH_MODEL'] == 'vendor/future-search'
    open_settings(page, 'models')
    open_settings(page, 'models/role/websearch')
    assert page.locator('#s-websearch-model').input_value() == 'vendor/future-search'
    assert not route_ui['errors']


@pytest.mark.parametrize('retired', [True, False])
def test_settings_window_history_disagreement_and_clear(route_ui, tmp_path, retired):
    from ouroboros import capability_evidence as ce

    route = dict(provider='openai-compatible', model='openai-compatible::future', base_url='https://compatible.invalid/v1')
    ack = ce.record_owner_ack(tmp_path, **route, window_tokens=1000000,
        note='owner-confirmed via context-mode toggle' if retired else 'typed')
    ce._store_evidence(tmp_path, 'probes', ack['route_fp'], ce.CapabilityEvidence(
        400000, 'confirmed', 'provider_metadata', ack['route_fp'], ts=ce.utc_now_iso()).to_json())
    page = route_ui['page']
    page.goto(route_ui['url'] + '/#settings')
    open_settings(page, 'models/role/main')
    main = page.locator('[data-model-role="main"]')
    main.locator('summary').click()
    page.wait_for_function("() => !document.querySelector('[data-model-role=main] [data-window-clear]').hidden")
    note = main.locator('[data-model-context-note]')
    assert ('old confirmation 1M' in note.inner_text()) is retired
    assert ('not used' in note.inner_text()) is retired
    assert ('the server says 400k' in note.inner_text()) is not retired
    assert ('Window 400k' if retired else 'Window 1M') in note.inner_text()
    page.set_viewport_size({'width': 390, 'height': 844})
    note.scroll_into_view_if_needed()
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    roles.capture(page, f'window-authority-{retired}')
    with page.expect_response('**/api/owner/capability-ack') as cleared:
        main.locator('[data-window-clear]').click()
    assert cleared.value.json()['ack']['removed']
    page.wait_for_function("() => document.querySelector('[data-model-role=main] [data-window-clear]').hidden")
    assert 'old confirmation' not in note.inner_text() and 'Window 400k' in note.inner_text()
    assert not ce.list_owner_acks(tmp_path)
    assert not route_ui['errors']


@pytest.mark.parametrize('edit_order', ['before_toggle', 'pending_preview', 'reopen'])
def test_response_maximum_preview_keeps_owner_draft(route_ui, edit_order):
    from playwright.sync_api import expect

    page = route_ui['page']
    page.goto(route_ui['url'] + '/#settings')
    open_settings(page, 'models/role/main')
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
    route_ui['errors'].watch(page)  # the reload may cut off the page's own reads (tests/ui_console_errors.py)
    # A fresh, untouched control still reads the persisted owner maximum.
    page.reload()
    open_settings(page, 'models/role/main')
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
    assert not route_ui['errors'].unexpected(), route_ui['errors'].report()


def test_response_maximum_deferred_toggle_cannot_cancel_apply(route_ui):
    from playwright.sync_api import expect

    page = route_ui['page']
    page.goto(route_ui['url'] + '/#settings')
    open_settings(page, 'models/role/main')
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
    # Available is the engine's verdict for the exact scope it answered (#1981), never a bare fresh row, and the
    # engine's "available" names a request only beside one measured every-model window (docs/PASSIVE_QUOTA_READ.md).
    quota = {'subject': {'harness': 'claude', 'subject_id': 'personal'},
             'freshness': 'fresh', 'constraints': [], 'availability': {'state': 'available'}}
    ui['fixture']['status']['quota'] = [quota]
    ui['fixture']['status']['availability_scope'] = {'model': 'claude-future[1M]', 'requestKind': 'text',
                                                     'basis': 'quota_block'}
    items = ui['settings']['OUROBOROS_SUBAGENTS']['items']
    # A reviewer is a catalog row marked Reviewer; an unmarked twin is an ordinary agent.
    items[2].update(review_eligible=True, route={'kind': 'agent_session', 'target_id': target,
                                                 'credential_profile_id': 'personal'})
    items.append({'subagent_id': 'claude-agent', 'recommended_use': 'Long-context agent.', 'effort': 'max',
                  'route': {'kind': 'agent_session', 'target_id': target}})
    page = roles.open_agents(ui)
    # No measured window yet: the exact echo and the engine's verdict alone prove nothing, so each row says Not checked
    # (neither Available nor Unavailable) and keeps its catalog qualifier, the exact selection and its Reviewer mark.
    for identity, reviews in (('agent', True), ('claude-agent', False)):
        row = roles.open_agent(ui, identity, fresh=False)
        row.locator('[data-subagent-meta]').filter(has_text='Base model listed').wait_for()
        status = row.locator('[data-subagent-status]')
        assert status.get_attribute('data-tone') == 'neutral'
        assert ('quota availability not proven' if reviews else 'live availability not checked') in status.inner_text()
        assert 'available now' not in status.inner_text() and 'currently unavailable' not in status.inner_text()
        assert row.locator('[data-subagent-field="model"]').input_value() == 'claude-future[1M]'
        assert row.locator('[data-subagent-field="review_eligible"]').is_checked() is reviews
    roles.capture(page, 'model-routes-claude-not-proven')
    # One measured window beside the engine's own verdict makes it Available.
    quota['constraints'] = [{'id': 'primary', 'used_ratio': 0.2}]
    ui['errors'].watch(page)  # the reload may cut off the page's own reads (tests/ui_console_errors.py)
    page.reload()
    page = roles.open_agents(ui)
    for identity, reviews in (('agent', True), ('claude-agent', False)):
        row = roles.open_agent(ui, identity, fresh=False)
        row.locator('[data-subagent-meta]').filter(has_text='Base model listed').wait_for()
        assert row.locator('[data-subagent-field="model"]').input_value() == 'claude-future[1M]'
        assert row.locator('[data-subagent-field="review_eligible"]').is_checked() is reviews
        status = row.locator('[data-subagent-status]')
        assert status.get_attribute('data-tone') == 'ok'
        assert 'available now' in status.inner_text()
        assert 'currently unavailable' not in status.inner_text()
    roles.capture(page, 'model-routes-claude-reviewer-row')
    page.set_viewport_size({'width': 390, 'height': 844})
    row.scroll_into_view_if_needed()
    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    # The conditional availability stays readable on a phone: natural height, nothing clipped.
    meta = row.locator('[data-subagent-meta]')
    assert meta.get_attribute('data-availability-qualifier') is not None
    assert 'checked by the engine at session start' in meta.inner_text()
    assert meta.evaluate('el => el.scrollWidth <= el.clientWidth && el.scrollHeight <= el.clientHeight')
    assert meta.evaluate('el => el.getBoundingClientRect().height > 1.5 * parseFloat(getComputedStyle(el).lineHeight)')
    roles.capture(page, 'model-routes-claude-agent-mobile')
    # An unrelated edit saves the catalog; neither selected spelling is rewritten.
    direct = roles.open_agent(ui, 'direct', fresh=False)
    assert direct.locator('[data-subagent-meta]').get_attribute('data-availability-qualifier') is None
    row = roles.open_agent(ui, 'claude-agent', fresh=False)
    row.locator('[data-subagent-field="recommended_use"]').fill('Long-context agent, edited.')
    assert press_editor_save(page).startswith('Saved')
    saved = roles.saved_catalog(ui)['items']
    assert [row['route']['target_id'] for row in saved[2:]] == [target, target]
    assert saved[2]['review_eligible'] is True and 'review_eligible' not in saved[3]
    # The reviewer's account pin survives the save, and the unpinned agent stays on automatic rotation.
    assert saved[2]['route'].get('credential_profile_id') == 'personal'
    assert not saved[3]['route'].get('credential_profile_id')
    assert not ui['errors']
