"""First-run named servers through the real atomic writer and Settings readback."""
from __future__ import annotations

import json
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

from ouroboros.test_evidence import diagnostic_directory
from tests.test_ui_smoke_playwright import direct_server_with_data as direct_server_with_data
from tests.settings_browser_helpers import open_settings
from tests.ui_console_errors import record_console_errors, record_failed_loads, unexpected_console_errors

pytestmark = [pytest.mark.ui_browser]


@contextmanager
def _server():
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            calls.append({"path": self.path, "authorization": self.headers.get("Authorization")})
            body = json.dumps({"data": [{"id": "same-model"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            calls.append({"path": self.path, "authorization": self.headers.get("Authorization"), "model": body.get("model")})
            data = json.dumps({"id": "fixture", "choices": [{"message": {"role": "assistant", "content": "OK"}}],
                               "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


def _frame(page, evidence, name):
    page.screenshot(path=str(evidence / f"{name}.png"), animations="disabled")
    return page.evaluate("""() => ({viewport: [innerWidth, innerHeight],
        body: [document.body.clientWidth, document.body.scrollWidth],
        scrollBodies: [...document.querySelectorAll('[data-settings-page], .wizard-content')]
            .filter(el => el.getBoundingClientRect().width).map(el => [el.clientWidth, el.scrollWidth]),
        overflow: [...document.querySelectorAll('input, select, .connection-row, .settings-provider-card')]
            .filter(el => el.getBoundingClientRect().width && el.getBoundingClientRect().right > innerWidth + 1)
            .map(el => ({tag: el.tagName, id: el.id, right: el.getBoundingClientRect().right}))})""")


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_first_run_named_servers_keep_routes_and_credentials_in_settings(direct_server_with_data, tmp_path, engine):
    from playwright.sync_api import expect, sync_playwright

    ui = direct_server_with_data
    evidence = diagnostic_directory(tmp_path / "frames", f"named-servers-{engine}", publish=True)
    # The running fixture owns this isolated root. Absence is the actual first-install proof.
    (ui["data_dir"] / "settings.json").unlink()
    report = {"candidate_identity": ui["candidate_identity"], "sentinel": ui["candidate_sentinel"].decode(),
              "version": ui["candidate_version"], "engine": engine, "frames": {}}
    with _server() as (alpha_url, alpha_calls), _server() as (beta_url, beta_calls), sync_playwright() as pw:
        browser = getattr(pw, engine).launch(headless=True)
        context = browser.new_context(viewport={"width": 1360, "height": 900}, device_scale_factor=2)
        page = context.new_page()
        errors, failed = record_console_errors(page), record_failed_loads(page)
        # These are deliberate no-engine fixture reads; retain all raw diagnostics below.
        page.goto(ui["url"] + "/onboarding", wait_until="networkidle")
        expect(page.locator('[data-connection-add="openai-compatible"]')).to_be_visible()
        # The card's shared renderer resolves assets beside its own builtin module,
        # independently of the main interface's selected /static tree.
        icon = page.locator('[data-provider-card="compatible"] .settings-provider-icon')
        expect(icon).to_have_attribute('src', ui["url"] + '/onboarding/static/providers/openai-compatible.svg')
        assert icon.evaluate('(img) => img.complete && img.naturalWidth > 0')
        page.locator('#s-openai-compatible-base-url').fill('')
        page.locator('#s-openai-compatible-key').fill('')
        default = page.locator('[data-connection-row][data-kind="openai-compatible"][data-builtin="1"]')
        expect(default.locator('[data-connection-status]')).to_have_text('Not saved')
        report['cleared_default'] = {
            'base_url': default.locator('[data-connection-field="base_url"]').input_value(),
            'status': default.locator('[data-connection-status]').inner_text(),
        }
        for label, url, key in (("Local reasoning server with a long descriptive name", alpha_url, ""),
                                ("Research server", beta_url, "b")):
            page.locator('[data-connection-add="openai-compatible"]').click()
            row = page.locator('[data-connection-row][data-uid]').last
            row.locator('[data-connection-field="base_url"]').fill(url)
            expect(page.locator('#next-btn')).to_be_disabled()
            row.locator('[data-connection-label]').fill(label)
            row.locator('[data-connection-field="api_key"]').fill(key)
        for theme in ("light", "dark"):
            page.evaluate("theme => document.documentElement.dataset.theme = theme", theme)
            page.locator('[data-provider-card="compatible"]').scroll_into_view_if_needed()
            report["frames"][f"accounts-{theme}"] = _frame(page, evidence, f"after-accounts-{theme}")
        page.click('#next-btn')
        page.wait_for_selector('[data-model-role="main"]')
        expect(page.locator('[data-model-role="heavy"], #s-model-heavy')).to_have_count(0)
        # Names are derived by the shared editor; read the exact saved values, never invent a second slugger.
        choices = page.locator('[data-model-role="main"] [data-model-role-source] option').evaluate_all(
            "nodes => nodes.map(n => n.value).filter(v => v.startsWith('api:openai-compatible@'))")
        assert len(choices) == 2
        sources = [choice.removeprefix('api:') for choice in choices]
        for role, source in zip(("main", "light"), sources):
            page.locator(f'[data-model-role="{role}"] [data-model-role-source]').select_option('api:' + source)
            page.locator('#compatible-model-source').select_option(source)
            with page.expect_response('**/api/openai-compatible/models') as discovery:
                page.click('#load-compatible-models')
            assert discovery.value.status == 200
            field = page.locator(f'[data-model-role="{role}"] [data-model-role-model]')
            if role == 'main':
                field.click()
                page.get_by_role('option', name='same-model', exact=True).click()
            else:
                field.fill('same-model')  # manual IDs stay valid even without discovery
        # Fallback is an explicit connection+model choice, independent of Main's identical model ID.
        page.locator('[data-model-role-group="fallback"] [data-model-add]').click()
        fallback = page.locator('[data-model-role-group="fallback"] [data-model-role]')
        expect(fallback).to_have_count(1)
        fallback.locator('[data-model-role-source]').select_option('api:' + sources[1])
        fallback.locator('[data-model-role-model]').fill('same-model')
        fallback_route = sources[1] + '::same-model'
        assert alpha_calls and beta_calls
        assert all(call['authorization'] is None for call in alpha_calls)
        assert all(call['authorization'] == 'Bearer b' for call in beta_calls)
        # An endpoint may have no model-list API. A failed lookup cannot erase the manual model.
        page.route('**/api/openai-compatible/models', lambda route: route.fulfill(
            status=502, json={"error": "Model discovery unavailable in fixture"}))
        page.click('#load-compatible-models')
        expect(page.locator('#compatible-load-status')).to_contain_text('You can type a model ID below.')
        expect(page.locator('[data-model-role="light"] [data-model-role-model]')).to_have_value('same-model')
        for theme in ('light', 'dark'):
            page.evaluate('theme => document.documentElement.dataset.theme = theme', theme)
            report['frames'][f'discovery-failure-{theme}'] = _frame(page, evidence, f'after-discovery-failure-{theme}')
        page.unroute('**/api/openai-compatible/models')
        # Back keeps both rows and their immutable names; editing a display label keeps Light's identity.
        page.click('#back-btn')
        beta = page.locator('[data-connection-row][data-uid]').last
        beta.locator('[data-connection-label]').fill('Research server renamed')
        beta.locator('.secret-toggle').click()
        expect(beta.locator('.settings-secret-value')).to_have_text('b')
        beta.locator('.secret-toggle').click()
        page.click('#next-btn')
        for role, source in zip(("main", "light"), sources):
            expect(page.locator(f'[data-model-role="{role}"] [data-model-role-source]')).to_have_value('api:' + source)
            expect(page.locator(f'[data-model-role="{role}"] [data-model-role-model]')).to_have_value('same-model')
        expect(fallback.locator('[data-model-role-source]')).to_have_value('api:' + sources[1])
        expect(fallback.locator('[data-model-role-model]')).to_have_value('same-model')
        for theme in ("light", "dark"):
            page.evaluate("theme => document.documentElement.dataset.theme = theme", theme)
            report["frames"][f"models-{theme}"] = _frame(page, evidence, f"after-models-{theme}")
            fallback.scroll_into_view_if_needed()
            report['frames'][f'fallback-{theme}'] = _frame(page, evidence, f'after-fallback-{theme}')
            page.evaluate('window.scrollTo(0, 0)')
        page.set_viewport_size({'width': 390, 'height': 844})
        report['frames']['narrow-models'] = _frame(page, evidence, 'after-models-narrow')
        page.set_viewport_size({'width': 1360, 'height': 900})
        page.click('#next-btn')  # Review
        page.locator('[data-runtime-mode="light"]').click()
        page.click('#next-btn')  # Budget
        page.click('#next-btn')  # Summary
        # Disable optional starter installation: this test makes no public network requests.
        page.locator('[data-starter-toggle]').click()
        page.locator('[data-starter-all="0"]').click()
        receipts = []

        def keep_receipt(route):
            response = route.fetch()
            receipts.append(response.json())
            report['completion_routes'] = {key: route.request.post_data_json.get(key, '') for key in (
                'OUROBOROS_MODEL', 'OUROBOROS_MODEL_LIGHT', 'OUROBOROS_MODEL_FALLBACKS')}
            route.fulfill(response=response)

        page.route('**/api/onboarding/complete', keep_receipt)
        with page.expect_response('**/api/onboarding/complete') as completion:
            page.click('#next-btn')
        assert completion.value.status == 200, completion.value.text()
        report['receipt'] = receipts[-1]
        saved = json.loads((ui['data_dir'] / 'settings.json').read_text(encoding="utf-8"))
        assert saved['OUROBOROS_MODEL'] == sources[0] + '::same-model'
        assert saved['OUROBOROS_MODEL_LIGHT'] == sources[1] + '::same-model'
        assert report['completion_routes']['OUROBOROS_MODEL_FALLBACKS'] == fallback_route
        assert saved['OUROBOROS_MODEL_FALLBACKS'] == fallback_route
        report['saved_routes'] = {key: saved[key] for key in report['completion_routes']}
        assert saved['OUROBOROS_CONNECTIONS'][0].get('api_key', '') == ''
        assert saved['OUROBOROS_CONNECTIONS'][1]['api_key'] == 'b'
        # Follow the wizard's completion handoff before opening Settings. A second goto
        # here races its location.replace and can abort either navigation.
        page.wait_for_url(ui['url'] + '/', wait_until='domcontentloaded')
        for role, source in zip(('main', 'light'), sources):
            open_settings(page, f"models/role/{role}")
            expect(page.locator(f'[data-model-role="{role}"] [data-model-role-source]')).to_have_value('api:' + source)
            expect(page.locator(f'[data-model-role="{role}"] [data-model-role-model]')).to_have_value('same-model')
        fallback = open_settings(page, "models/role/fallback/1")
        expect(fallback.locator('[data-model-role-source]')).to_have_value('api:' + sources[1])
        expect(fallback.locator('[data-model-role-model]')).to_have_value('same-model')
        settings = page.request.get(ui['url'] + '/api/settings').json()
        assert settings['OUROBOROS_MODEL_FALLBACKS'] == fallback_route
        report['reloaded_routes'] = {key: settings[key] for key in report['completion_routes']}
        for theme in ('light', 'dark'):
            page.evaluate('theme => document.documentElement.dataset.theme = theme', theme)
            report['frames'][f'settings-fallback-{theme}'] = _frame(page, evidence, f'after-settings-fallback-{theme}')
        listing = open_settings(page, "accounts")
        for source in sources:
            expect(listing.locator(f'[data-entry-id="{source}"]')).to_have_count(1)
        # Each saved destination has its own editor; its Test uses only that connection's saved credential.
        for source in sources:
            kind, name = source.split('@', 1)
            editor = open_settings(page, f"accounts/key/{kind}/{name}")
            row = editor.locator('[data-connection-row]')
            if source == sources[1]:
                expect(row.locator('[data-connection-label]')).to_have_value('Research server renamed')
            with page.expect_response('**/api/providers/test') as tested:
                row.locator('[data-connection-test]').click()
            assert tested.value.status == 200 and tested.value.json()['ok'], tested.value.text()
            assert tested.value.request.post_data_json == {
                'provider_id': 'openai-compatible', 'connection': name, 'overrides': {},
            }
            expect(row.locator('[data-editor-result]')).to_have_text('Works')
            # A verdict is about the tested draft; editing clears it, without saving anything.
            field = row.locator('[data-connection-field="base_url"]')
            original = field.input_value()
            field.fill('')
            expect(row.locator('[data-editor-result]')).to_have_text('')
            field.fill(original)
        assert any(call.get('model') == 'same-model' for call in alpha_calls)
        assert any(call.get('model') == 'same-model' for call in beta_calls)
        assert all(call['authorization'] is None for call in alpha_calls)
        assert all(call['authorization'] == 'Bearer b' for call in beta_calls)
        report.update(alpha_calls=alpha_calls, beta_calls=beta_calls, console_errors=errors, failed_loads=failed,
                      unexpected_console_errors=unexpected_console_errors(errors, failed))
        page.set_viewport_size({'width': 390, 'height': 844})
        row.scroll_into_view_if_needed()
        report['frames']['narrow-accounts'] = _frame(page, evidence, 'after-settings-accounts-narrow')
        (evidence / 'consumer-proof.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        assert all(not frame['overflow'] and frame['body'][1] <= frame['viewport'][0] + 1 for frame in report['frames'].values())
        assert all(width >= scroll - 1 for frame in report['frames'].values() for width, scroll in frame['scrollBodies'])
        browser.close()


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_undo_named_server_edits_restores_the_current_wizard_draft(direct_server_with_data, tmp_path, engine):
    from playwright.sync_api import expect, sync_playwright

    ui = direct_server_with_data
    evidence = diagnostic_directory(tmp_path / "frames", f"undo-server-{engine}", publish=True)
    (ui["data_dir"] / "settings.json").unlink()
    with _server() as (url, calls), sync_playwright() as pw:
        browser = getattr(pw, engine).launch(headless=True)
        page = browser.new_page(viewport={"width": 1360, "height": 900})
        errors, failed = record_console_errors(page), record_failed_loads(page)
        page.goto(ui["url"] + "/onboarding", wait_until="networkidle")
        default = page.locator('[data-connection-row][data-kind="openai-compatible"][data-builtin="1"]')
        base = default.locator('[data-connection-field="base_url"]')
        initial_url = base.input_value()
        expect(default.locator('[data-connection-status]')).to_have_text('Ready')
        page.locator('#s-openai-compatible-base-url').fill('')
        expect(default.locator('[data-connection-status]')).to_have_text('Not saved')
        base.fill(initial_url)
        expect(default.locator('[data-connection-status]')).to_have_text('Ready')
        base.fill('')
        page.locator('#s-openai-compatible-key').fill('')
        page.locator('[data-connection-add="openai-compatible"]').click()
        row = page.locator('[data-connection-row][data-uid]').last
        row.locator('[data-connection-label]').fill('Research server')
        row.locator('[data-connection-field="base_url"]').fill(url)
        row.locator('[data-connection-field="api_key"]').fill('b')
        page.click('#next-btn')
        main = page.locator('[data-model-role="main"]')
        source = main.locator('[data-model-role-source] option').evaluate_all(
            "nodes => nodes.map(n => n.value).find(v => v.startsWith('api:openai-compatible@'))")
        main.locator('[data-model-role-source]').select_option(source)
        main.locator('[data-model-role-model]').fill('same-model')
        page.click('#back-btn')  # The editor now has a nonempty mount baseline.
        row = page.locator('[data-connection-row][data-uid]').last
        row.locator('[data-connection-toggle]').click()
        expect(row.locator('[data-connection-status]')).to_have_text('Disabled')
        expect(page.locator('#next-btn')).to_be_disabled()
        row.locator('[data-connection-toggle]').click()
        expect(row.locator('[data-connection-status]')).to_have_text('Ready')
        expect(page.locator('#next-btn')).to_be_enabled()
        report = {"candidate_identity": ui["candidate_identity"], "engine": engine,
                  "toggle_back": _frame(page, evidence, 'toggle-back')}
        # Returning fields to the loaded values emits an empty patch, not a new document.
        row.locator('[data-connection-field="base_url"]').fill('http://127.0.0.1:1/v1')
        expect(row.locator('[data-connection-status]')).to_have_text('Not saved')
        row.locator('[data-connection-field="base_url"]').fill(url)
        expect(row.locator('[data-connection-status]')).to_have_text('Ready')
        row.locator('[data-connection-field="api_key"]').fill('temporary-draft-key')
        expect(row.locator('[data-connection-status]')).to_have_text('Not saved')
        row.locator('[data-connection-field="api_key"]').fill('b')
        expect(row.locator('[data-connection-status]')).to_have_text('Ready')
        page.click('#next-btn')
        expect(main.locator('[data-model-role-source]')).to_have_value(source)
        expect(main.locator('[data-model-role-model]')).to_have_value('same-model')
        page.locator('#compatible-model-source').select_option(source.removeprefix('api:'))
        with page.expect_response('**/api/openai-compatible/models') as discovery:
            page.click('#load-compatible-models')
        assert discovery.value.status == 200
        sent = discovery.value.request.post_data_json
        assert sent['baseUrl'] == url and sent['apiKey'] == 'b'
        assert calls and all(call['authorization'] == 'Bearer b' for call in calls)
        report.update(discovery_payload=sent, upstream_calls=calls, console_errors=errors,
                      failed_loads=failed, models=_frame(page, evidence, 'undone-models'))
        (evidence / 'undo-proof.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        browser.close()
