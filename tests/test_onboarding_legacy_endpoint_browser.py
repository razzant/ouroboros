"""Reopened setup over an older install's OPENAI_BASE_URL + OPENAI_API_KEY pair (#2497).

The real server, wizard, discovery proxy, atomic writer and Settings over an isolated root:
the pair is shown read-only, its row, sources and discovery target follow the draft, and a
Finish that edits no key keeps both fields byte-for-byte without manufacturing a dedicated pair.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest

from ouroboros.test_evidence import diagnostic_directory
from tests.test_ui_smoke_playwright import direct_server_with_data as direct_server_with_data
from tests.ui_chat_viewport_smoke import wait_drawer_closed
from tests.ui_console_errors import record_console_errors, record_failed_loads, unexpected_console_errors

pytestmark = [pytest.mark.ui_browser]

LEGACY_KEY = "sk-legacy-pair-0123456789"
EDITED_KEY = "sk-edited-legacy-pair-0123456789"


@contextmanager
def _endpoint(models):
    """An inert OpenAI-compatible server that records where each request went and with which auth."""
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def _answer(self, body):
            data = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            calls.append({"method": "GET", "path": self.path, "authorization": self.headers.get("Authorization")})
            self._answer({"data": [{"id": model} for model in models]})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            calls.append({"method": "POST", "path": self.path, "authorization": self.headers.get("Authorization"),
                          "model": body.get("model")})
            self._answer({"id": "fixture", "choices": [{"message": {"role": "assistant", "content": "OK"}}],
                          "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})

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


def _listed(calls):
    return [call for call in calls if call["method"] == "GET" and call["path"].endswith("/models")]


def _frame(page, evidence, name, element=None):
    """A full-viewport frame (and the element itself, when named) plus the overflow facts beside it."""
    page.screenshot(path=str(evidence / f"{name}.png"), animations="disabled")
    if element is not None:
        element.screenshot(path=str(evidence / f"{name}-element.png"), animations="disabled")
    return page.evaluate("""() => ({viewport: [innerWidth, innerHeight],
        body: [document.body.clientWidth, document.body.scrollWidth],
        overflow: [...document.querySelectorAll('input, select, .connection-row, .settings-provider-card, .harness-account-meta')]
            .filter(el => el.getBoundingClientRect().width && el.getBoundingClientRect().right > innerWidth + 1)
            .map(el => ({tag: el.tagName, id: el.id, right: el.getBoundingClientRect().right}))})""")


def _seed_legacy_install(ui, endpoint_url):
    """An older install: only the legacy pair reaches its server; seeded while the server is stopped."""
    ui["stop_server"]()
    path = ui["data_dir"] / "settings.json"
    settings = json.loads(path.read_text(encoding="utf-8"))
    for key in ("OPENAI_COMPATIBLE_BASE_URL", "OPENAI_COMPATIBLE_API_KEY"):
        settings.pop(key, None)
    settings.update({
        "OPENAI_BASE_URL": endpoint_url, "OPENAI_API_KEY": LEGACY_KEY,
        "OUROBOROS_MODEL": "openai-compatible::legacy-model", "OUROBOROS_MODEL_LIGHT": "openai-compatible::legacy-model",
        "OUROBOROS_MODEL_FALLBACKS": "openai-compatible::legacy-model",
    })
    path.write_text(json.dumps(settings), encoding="utf-8")
    ui["start_server"]()
    return settings


def _summary(page):
    return dict(page.locator('.summary-kv').evaluate_all(
        "rows => rows.map(row => [row.querySelector('strong').textContent.trim(), row.querySelector('span').textContent.trim()])"))


def _test_row(page, row, verdict):
    """The row's real Test button: what the page asked, once its verdict is shown."""
    from playwright.sync_api import expect

    with page.expect_response('**/api/providers/test') as tested:
        row.locator('[data-connection-test]').click()
    expect(row.locator('[data-connection-test-status]')).to_have_text(verdict)
    return tested.value.request.post_data_json


def _legacy_test(url, key="***set***", **overrides):
    return {"provider_id": "openai-compatible", "api_key_setting": "OPENAI_API_KEY",
            "overrides": {**overrides, "OPENAI_BASE_URL": url, "OPENAI_API_KEY": key}}


def _sent(calls, method="POST"):
    return {call["authorization"] for call in calls if call["method"] == method}


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_reopened_setup_reaches_and_keeps_a_legacy_compatible_pair(direct_server_with_data, tmp_path, engine):
    from playwright.sync_api import expect, sync_playwright

    ui = direct_server_with_data
    evidence = diagnostic_directory(tmp_path / "frames", f"legacy-endpoint-{engine}")
    report = {"candidate_identity": ui["candidate_identity"], "version": ui["candidate_version"], "engine": engine, "frames": {}}
    with _endpoint(["legacy-model", "second-model"]) as (legacy_url, legacy_calls), \
            _endpoint(["dedicated-model"]) as (dedicated_url, dedicated_calls), sync_playwright() as pw:
        seeded = _seed_legacy_install(ui, legacy_url)
        report["seeded"] = {key: seeded[key] for key in ("OPENAI_BASE_URL", "OUROBOROS_MODEL")}
        browser = getattr(pw, engine).launch(headless=True)
        # The desktop setup window's own size (launcher_onboarding: 980x780).
        context = browser.new_context(viewport={"width": 980, "height": 780}, device_scale_factor=2)
        page = context.new_page()
        errors, failed = record_console_errors(page), record_failed_loads(page)
        uncaught = []
        page.on("pageerror", lambda error: uncaught.append(str(error)))
        page.goto(ui["url"] + "/onboarding?builtin=1", wait_until="networkidle")

        # Accounts: the saved pair is the built-in server's, shown read-only; no field is prefilled from it.
        card = page.locator('[data-provider-card="compatible"]')
        default = card.locator('[data-connection-row][data-builtin="1"]')
        status, meta = default.locator('[data-connection-status]'), default.locator('[data-connection-meta]')
        expect(card.locator('.settings-provider-body')).to_be_visible()
        expect(status).to_have_text('Ready')
        expect(meta).to_be_visible()
        expect(meta).to_contain_text(f'Legacy OpenAI Base URL {legacy_url} with the OpenAI API key'
                                     ' · change the address in Settings → Accounts · used by Main')
        expect(default.locator('[data-connection-field="base_url"]')).to_have_value('')
        expect(default.locator('[data-connection-field="api_key"]')).to_have_value('')
        expect(default.locator('[data-connection-remove]')).to_be_hidden()
        expect(page.locator('#openai-key')).to_have_value('***set***')
        report["accounts_meta"] = meta.inner_text()

        # Test probes the pair a save runs: the page names it, the kept key stays masked and the server sends it
        # only to its own address. An edit of the OpenAI key beside the row withdraws the verdict and is what Test
        # sends next; with no key, the row says No key and nothing is sent, as at runtime.
        before = len(legacy_calls)
        assert _test_row(page, default, 'Works') == _legacy_test(legacy_url)
        assert _sent(legacy_calls[before:]) == {f"Bearer {LEGACY_KEY}"}
        result = default.locator('[data-connection-test-status]')
        page.locator('#openai-key').fill(EDITED_KEY)
        expect(result).to_have_text('')
        before = len(legacy_calls)
        assert _test_row(page, default, 'Works') == _legacy_test(legacy_url, EDITED_KEY)
        assert _sent(legacy_calls[before:]) == {f"Bearer {EDITED_KEY}"}
        assert not dedicated_calls
        page.locator('#openai-key').fill('')
        expect(status).to_have_text('No key')
        before = len(legacy_calls)
        assert _test_row(page, default, 'Not ready — Provider is not configured') == _legacy_test(legacy_url, '')
        assert len(legacy_calls) == before
        page.locator('#openai-key').fill('***set***')
        expect(status).to_have_text('Ready')
        expect(result).to_have_text('')

        for theme in ("light", "dark"):
            page.evaluate("theme => document.documentElement.dataset.theme = theme", theme)
            default.scroll_into_view_if_needed()
            report["frames"][f"accounts-{theme}"] = _frame(page, evidence, f"accounts-980x780-{theme}", card)
        page.evaluate("document.documentElement.dataset.theme = 'light'")
        page.set_viewport_size({"width": 390, "height": 844})
        default.scroll_into_view_if_needed()
        report["frames"]["accounts-narrow"] = _frame(page, evidence, "accounts-390x844", card)
        page.set_viewport_size({"width": 980, "height": 780})

        # Models: the legacy server is a usable source with its own discovery; official OpenAI stays a choice.
        page.click('#next-btn')
        main = page.locator('[data-model-role="main"]')
        expect(main.locator('[data-model-role-source]')).to_have_value('api:openai-compatible')
        options = main.locator('[data-model-role-source] option').evaluate_all(
            "nodes => nodes.filter(n => n.value.startsWith('api:')).map(n => [n.value, n.textContent.trim()])")
        report["main_api_sources"] = options
        assert ['api:openai-compatible', 'OpenAI-compatible endpoint'] in options
        assert ['api:openai', 'OpenAI'] in options
        assert not any('(no key)' in label for _value, label in options)
        expect(page.locator('#compatible-model-source option')).to_have_count(1)
        expect(page.locator('#compatible-model-source')).to_have_value('openai-compatible')
        expect(page.locator('#next-btn')).to_be_enabled()
        listed_before = len(_listed(legacy_calls))
        with page.expect_response('**/api/openai-compatible/models') as discovery:
            page.click('#load-compatible-models')
        assert discovery.value.status == 200, discovery.value.text()
        report["discovery_request"] = discovery.value.request.post_data_json
        # The masked key resolves on the server for the address it was saved with, and nowhere else.
        assert report["discovery_request"] == {"baseUrl": legacy_url, "apiKey": "***set***", "apiKeySetting": "OPENAI_API_KEY"}
        assert len(_listed(legacy_calls)) == listed_before + 1
        assert {call["authorization"] for call in _listed(legacy_calls)} == {f"Bearer {LEGACY_KEY}"}
        expect(page.locator('#compatible-load-status')).to_contain_text('2 models listed')
        light = page.locator('[data-model-role="light"] [data-model-role-model]')
        light.fill('second')  # the chooser lists the discovered models matching what is typed
        page.get_by_role('option', name='second-model', exact=True).click()
        expect(light).to_have_value('second-model')
        report["frames"]["models"] = _frame(page, evidence, "models-980x780", page.locator('#compatible-model-loader'))

        page.click('#next-btn')  # Review
        page.click('#next-btn')  # Budget
        page.click('#next-btn')  # Summary
        summary = _summary(page)
        report["summary"] = summary
        assert summary["Connected access"] == "OpenAI-compatible endpoint"
        assert summary["OpenAI"] == f"configured · also used by the legacy address {legacy_url}"
        report["frames"]["summary"] = _frame(page, evidence, "summary-980x780")
        page.locator('[data-starter-toggle]').click()  # no public starter install from this isolated root
        page.locator('[data-starter-all="0"]').click()
        sent = {}

        def keep_body(route):
            sent.update(route.request.post_data_json)
            route.continue_()

        page.route('**/api/onboarding/complete', keep_body)
        with page.expect_response('**/api/onboarding/complete') as completion:
            page.click('#next-btn')
        assert completion.value.status == 200, completion.value.text()
        report["completion_body"] = {key: sent.get(key) for key in (
            "OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_COMPATIBLE_BASE_URL", "OPENAI_COMPATIBLE_API_KEY",
            "OUROBOROS_MODEL", "OUROBOROS_MODEL_LIGHT")}
        assert report["completion_body"]["OPENAI_API_KEY"] == "***set***"  # never the key itself
        saved = json.loads((ui["data_dir"] / "settings.json").read_text(encoding="utf-8"))
        report["saved"] = {key: saved.get(key) for key in report["completion_body"] if key != "OPENAI_API_KEY"}
        assert (saved["OPENAI_BASE_URL"], saved["OPENAI_API_KEY"]) == (legacy_url, LEGACY_KEY)
        assert saved.get("OPENAI_COMPATIBLE_BASE_URL", "") == saved.get("OPENAI_COMPATIBLE_API_KEY", "") == ""
        assert saved["OUROBOROS_MODEL"] == "openai-compatible::legacy-model"
        assert saved["OUROBOROS_MODEL_LIGHT"] == "openai-compatible::second-model"

        # Settings reads the same pair: Ready with the same address, and Main's source keeps its access.
        page.wait_for_url(ui["url"] + "/", wait_until="domcontentloaded")
        page.click('[data-nav-page="settings"]')
        expect(page.locator('#settings-status')).to_have_text('Settings refreshed')
        settings_card = page.locator('[data-provider-card="compatible"]')
        if settings_card.locator('.settings-provider-body').is_hidden():  # Settings opens it only for added keys
            settings_card.locator('[data-provider-toggle]').click()
        settings_row = settings_card.locator('[data-connection-row][data-builtin="1"]')
        expect(settings_row.locator('[data-connection-status]')).to_have_text('Ready')
        expect(settings_row.locator('[data-connection-meta]')).to_contain_text(legacy_url)
        expect(page.locator('#s-openai-base-url')).to_have_value(legacy_url)
        report["settings_meta"] = settings_row.locator('[data-connection-meta]').inner_text()
        settings_row.scroll_into_view_if_needed()
        report["frames"]["settings-accounts"] = _frame(page, evidence, "settings-accounts", settings_row)
        # Settings' Test reads the same pair, the key as typed in its own OpenAI row; official OpenAI asks for itself
        # alone (answered here: no request leaves for the vendor).
        kept = page.locator('#s-openai').input_value()
        assert kept and kept != LEGACY_KEY, "Settings holds its own mask, never the key"
        before = len(legacy_calls)
        assert _test_row(page, settings_row, 'Works') == _legacy_test(legacy_url, kept)
        assert _sent(legacy_calls[before:]) == {f"Bearer {LEGACY_KEY}"}
        openai_card = page.locator('[data-provider-card="openai"]')
        if openai_card.locator('.settings-provider-body').is_hidden():
            openai_card.locator('[data-provider-toggle]').click()
        page.locator('#s-openai').fill(EDITED_KEY)
        expect(settings_row.locator('[data-connection-test-status]')).to_have_text('')
        before = len(legacy_calls)
        assert _test_row(page, settings_row, 'Works') == _legacy_test(legacy_url, EDITED_KEY)
        assert _sent(legacy_calls[before:]) == {f"Bearer {EDITED_KEY}"}
        official = openai_card.locator('[data-connection-row][data-builtin="1"]')
        page.route('**/api/providers/test', lambda route: route.fulfill(json={"ok": True}))
        assert _test_row(page, official, 'Works') == {"provider_id": "openai", "overrides": {"OPENAI_API_KEY": EDITED_KEY}}
        page.unroute('**/api/providers/test')
        page.locator('#s-openai').fill(kept)
        # An unsaved legacy address is what Test asks about too: the kept key stays on its saved address, nothing
        # reaches the new one, and a typed key may test it.
        address = page.locator('#s-openai-base-url')
        address.fill(dedicated_url)
        expect(settings_row.locator('[data-connection-test-status]')).to_have_text('')
        moved = "Not ready — This server's address changed since its key was saved. Enter its key again to test the new address."
        assert _test_row(page, settings_row, moved) == _legacy_test(dedicated_url, kept)
        assert not dedicated_calls
        page.locator('#s-openai').fill(EDITED_KEY)
        assert _test_row(page, settings_row, 'Works') == _legacy_test(dedicated_url, EDITED_KEY)
        assert _sent(dedicated_calls) == {f"Bearer {EDITED_KEY}"}
        dedicated_calls.clear()
        page.locator('#s-openai').fill(kept)
        address.fill(legacy_url)
        page.click('[data-settings-tab="models"]')
        settings_main = page.locator('[data-model-role="main"] [data-model-role-source]')
        expect(settings_main).to_have_value('api:openai-compatible')
        assert '(no key)' not in settings_main.locator('option:checked').inner_text()

        # Reopened again, the row, the sources and the discovery target follow each draft edit together.
        page.goto(ui["url"] + "/onboarding?builtin=1", wait_until="networkidle")
        expect(status).to_have_text('Ready')
        page.locator('#openai-key').fill('')
        expect(status).to_have_text('No key')
        expect(meta).to_contain_text(legacy_url)
        expect(page.locator('#next-btn')).to_be_disabled()
        report["frames"]["key-cleared"] = _frame(page, evidence, "accounts-key-cleared", card)
        page.locator('#openai-key').fill('***set***')
        expect(status).to_have_text('Ready')
        expect(page.locator('#next-btn')).to_be_enabled()
        base = default.locator('[data-connection-field="base_url"]')
        base.fill(dedicated_url)
        expect(status).to_have_text('Not saved')
        expect(meta).to_be_hidden()
        # An edit beside the row while it is a draft, then undo: the row reads the current draft again.
        page.locator('#openai-key').fill('')
        page.locator('#openai-key').fill('***set***')
        base.fill('')
        expect(status).to_have_text('Ready')
        expect(meta).to_contain_text(legacy_url)
        base.fill(dedicated_url)
        expect(status).to_have_text('Not saved')
        page.click('#next-btn')
        listed_before = len(_listed(legacy_calls))
        with page.expect_response('**/api/openai-compatible/models') as discovery:
            page.click('#load-compatible-models')
        assert discovery.value.status == 200, discovery.value.text()
        # A keyless dedicated address now serves the built-in source; the legacy key stays home.
        report["dedicated_discovery_request"] = discovery.value.request.post_data_json
        assert report["dedicated_discovery_request"] == {"baseUrl": dedicated_url, "apiKey": "", "apiKeySetting": "OPENAI_COMPATIBLE_API_KEY"}
        assert [call["authorization"] for call in _listed(dedicated_calls)] == [None]
        assert len(_listed(legacy_calls)) == listed_before
        report["frames"]["dedicated-models"] = _frame(page, evidence, "models-dedicated-draft")
        # Official OpenAI stays a supported explicit choice beside the compatible endpoint.
        main.locator('[data-model-role-source]').select_option('api:openai')
        main.locator('[data-model-role-model]').fill('gpt-5.6-terra')
        expect(page.locator('#next-btn')).to_be_enabled()
        report.update(legacy_calls=legacy_calls, dedicated_calls=dedicated_calls, console_errors=errors,
                      failed_loads=failed, uncaught_page_errors=uncaught,
                      unexpected_console_errors=unexpected_console_errors(errors, failed))
        (evidence / "legacy-endpoint-proof.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        assert all(not frame["overflow"] and frame["body"][1] <= frame["viewport"][0] + 1 for frame in report["frames"].values())
        assert not uncaught
        browser.close()


DEDICATED_KEY = "sk-dedicated-pair-9876543210"
NAMED_KEY = "sk-named-server-5555555555"


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
@pytest.mark.parametrize("shape", ["one-address", "two-addresses"])
def test_both_pairs_test_and_discovery_follow_the_cleared_dedicated_address(direct_server_with_data, tmp_path, shape, engine):
    """Both pairs saved, at one address with two keys (the address cannot tell them apart) or at two. Clearing the
    dedicated address selects the legacy pair: Test and discovery name that pair and reach its address with the
    legacy key. Restoring the address brings the dedicated pair and its key back; Finish saves what Test probed."""
    from playwright.sync_api import expect, sync_playwright

    ui = direct_server_with_data
    evidence = diagnostic_directory(tmp_path / "frames", f"both-pairs-{shape}-{engine}")
    report = {"candidate_identity": ui["candidate_identity"], "frames": {}, "steps": []}
    with _endpoint(["shared-model"]) as (legacy_url, legacy_calls), _endpoint(["shared-model"]) as (other_url, other_calls), \
            _endpoint(["lab-model"]) as (named_url, named_calls), sync_playwright() as pw:
        url = legacy_url if shape == "one-address" else other_url
        calls = {legacy_url: legacy_calls, url: other_calls if url == other_url else legacy_calls}
        ui["stop_server"]()  # seed only while stopped
        path = ui["data_dir"] / "settings.json"
        settings = json.loads(path.read_text(encoding="utf-8"))
        settings.update({"OPENAI_COMPATIBLE_BASE_URL": url, "OPENAI_COMPATIBLE_API_KEY": DEDICATED_KEY,
                         "OPENAI_BASE_URL": legacy_url, "OPENAI_API_KEY": LEGACY_KEY,
                         "OUROBOROS_CONNECTIONS": [{"kind": "openai-compatible", "name": "lab", "label": "Lab",
                                                    "base_url": named_url, "api_key": NAMED_KEY}],
                         **dict.fromkeys(("OUROBOROS_MODEL", "OUROBOROS_MODEL_LIGHT", "OUROBOROS_MODEL_FALLBACKS"),
                                         "openai-compatible::shared-model")})
        path.write_text(json.dumps(settings), encoding="utf-8")
        ui["start_server"]()
        browser = getattr(pw, engine).launch(headless=True)
        page = browser.new_context(viewport={"width": 980, "height": 780}, device_scale_factor=2).new_page()
        uncaught = []
        page.on("pageerror", lambda error: uncaught.append(str(error)))
        page.goto(ui["url"] + "/onboarding?builtin=1", wait_until="networkidle")
        row = page.locator('[data-provider-card="compatible"] [data-connection-row][data-builtin="1"]')
        base, status = row.locator('[data-connection-field="base_url"]'), row.locator('[data-connection-status]')
        meta = row.locator('[data-connection-meta]')
        expect(base).to_have_value(url)
        expect(status).to_have_text('Ready')
        dedicated_test = {"provider_id": "openai-compatible", "api_key_setting": "OPENAI_COMPATIBLE_API_KEY", "overrides": {}}
        for name, typed, setting, target, key, serving, tested in (
                ("saved", None, "OPENAI_COMPATIBLE_API_KEY", url, DEDICATED_KEY, "Enabled", dedicated_test),
                ("cleared", "", "OPENAI_API_KEY", legacy_url, LEGACY_KEY, f"Legacy OpenAI Base URL {legacy_url} with the OpenAI API key",
                 _legacy_test(legacy_url, OPENAI_COMPATIBLE_BASE_URL="")),
                ("restored", url, "OPENAI_COMPATIBLE_API_KEY", url, DEDICATED_KEY, "Enabled", dedicated_test)):
            if typed is not None:
                base.fill(typed)
            if name == "cleared":
                expect(status).to_have_text('Not saved')  # an edited row is a draft; it vouches for nothing yet
            step = {"step": name, "row": status.inner_text(), "meta": meta.inner_text() if meta.is_visible() else ""}
            # Test first: the pair this draft reaches, with that pair's key only, at that pair's address.
            before = {endpoint: len(sent) for endpoint, sent in calls.items()}
            row.scroll_into_view_if_needed()
            step["test_request"] = _test_row(page, row, 'Works')
            assert step["test_request"] == tested
            step["test_upstream"] = calls[target][before[target]:]
            assert _sent(step["test_upstream"]) == {f"Bearer {key}"}
            for endpoint, sent in calls.items():
                if sent is not calls[target]:
                    assert len(sent) == before[endpoint], "the other pair's address hears nothing"
            report["frames"][f"accounts-{name}"] = _frame(page, evidence, f"accounts-{name}", row)
            page.click('#next-btn')
            before = len(calls[target])
            with page.expect_response('**/api/openai-compatible/models') as discovery:
                page.click('#load-compatible-models')
            assert discovery.value.status == 200, discovery.value.text()
            step["request"] = discovery.value.request.post_data_json
            # The page names the pair its draft selects; the key behind the mask is read from that pair only.
            assert step["request"] == {"baseUrl": target, "apiKey": "***set***", "apiKeySetting": setting}
            step["upstream"] = [call for call in calls[target][before:] if call["path"].endswith("/models")]
            # The server's own catalog reads use the saved effective (dedicated) pair, so the legacy key can only
            # reach the server through this discovery.
            assert f"Bearer {key}" in {call["authorization"] for call in step["upstream"]}
            report["frames"][f"models-{name}"] = _frame(page, evidence, f"models-{name}",
                                                        page.locator('#compatible-model-loader'))
            page.click('#back-btn')
            # Remounted on this draft, the row says which pair now serves it: the one Test and discovery just used.
            expect(status).to_have_text('Ready')
            expect(meta).to_contain_text(serving)
            if name != "cleared":
                expect(meta).not_to_contain_text("Legacy")
            step["remounted"] = {"row": status.inner_text(), "meta": meta.inner_text()}
            report["frames"][f"accounts-{name}-remounted"] = _frame(page, evidence, f"accounts-{name}-remounted", row)
            report["steps"].append(step)

        # A named server keeps its own request and key, whichever pair the built-in row reaches.
        named = page.locator('[data-provider-card="compatible"] [data-connection-row][data-name="lab"]')
        before = len(named_calls)
        row.scroll_into_view_if_needed()
        assert _test_row(page, named, 'Works') == {"provider_id": "openai-compatible", "connection": "lab", "overrides": {}}
        assert _sent(named_calls[before:]) == {f"Bearer {NAMED_KEY}"}

        # The negative control: the old request for this same draft, without a selector, keeps its Clear and blanks
        # both addresses, so it cannot test the pair the row, discovery and a save all use.
        base.fill("")
        before = {endpoint: len(sent) for endpoint, sent in calls.items()}
        old = page.evaluate("""async () => { const response = await fetch('/api/providers/test', {method: 'POST',
            headers: {'Content-Type': 'application/json'}, body: JSON.stringify({provider_id: 'openai-compatible',
            overrides: {OPENAI_COMPATIBLE_BASE_URL: ''}})}); return [response.status, await response.json()]; }""")
        report["old_request_after_clear"] = old
        assert old[0] == 200 and old[1]["ok"] is False
        assert {endpoint: len(sent) for endpoint, sent in calls.items()} == before

        # The edited legacy key is what Test sends. A kept key is not: the masked value goes back to the saved one.
        page.locator('#openai-key').fill(EDITED_KEY)
        before = len(legacy_calls)
        assert _test_row(page, row, 'Works') == _legacy_test(legacy_url, EDITED_KEY, OPENAI_COMPATIBLE_BASE_URL="")
        assert _sent(legacy_calls[before:]) == {f"Bearer {EDITED_KEY}"}
        page.click('#next-btn')
        before = len(legacy_calls)
        with page.expect_response('**/api/openai-compatible/models') as discovery:
            page.click('#load-compatible-models')
        assert discovery.value.request.post_data_json == {"baseUrl": legacy_url, "apiKey": EDITED_KEY, "apiKeySetting": "OPENAI_API_KEY"}
        assert f"Bearer {EDITED_KEY}" in _sent(legacy_calls[before:], "GET")
        page.click('#back-btn')
        page.locator('#openai-key').fill('***set***')

        # Finish from the cleared draft saves the pair Test probed: the legacy pair untouched, the dedicated address
        # blank, and no key moved or copied between pairs.
        for _ in range(4):
            page.click('#next-btn')
        page.locator('[data-starter-toggle]').click()  # no public starter install from this isolated root
        page.locator('[data-starter-all="0"]').click()
        with page.expect_response('**/api/onboarding/complete') as completion:
            page.click('#next-btn')
        assert completion.value.status == 200, completion.value.text()
        saved = json.loads(path.read_text(encoding="utf-8"))
        assert (saved["OPENAI_BASE_URL"], saved["OPENAI_API_KEY"]) == (legacy_url, LEGACY_KEY)
        assert (saved["OPENAI_COMPATIBLE_BASE_URL"], saved["OPENAI_COMPATIBLE_API_KEY"]) == ("", DEDICATED_KEY)
        assert saved["OUROBOROS_CONNECTIONS"][0]["api_key"] == NAMED_KEY
        report["saved_after_clear"] = {key: saved[key] for key in ("OPENAI_BASE_URL", "OPENAI_COMPATIBLE_BASE_URL")}
        report.update(uncaught_page_errors=uncaught)
        (evidence / "both-pairs-proof.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        assert all(not frame["overflow"] for frame in report["frames"].values())
        assert not uncaught
        browser.close()


MOVED = "Not ready — This server's address changed since its key was saved. Enter its key again to test the new address."


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
@pytest.mark.parametrize("shape", ["legacy", "both-pairs"])
def test_settings_row_describes_the_pair_its_test_sends(direct_server_with_data, tmp_path, shape, engine):
    """Settings' built-in compatible row reads the saved pair's facts only while its draft still reaches that pair,
    the one its Test sends. The legacy address and the OpenAI key are edited beside the row: moving the legacy pair
    there makes the row a draft (it vouches for nothing yet), and going back restores the saved facts. With a
    dedicated pair saved, both are beside the point until the dedicated address is cleared. Nothing is saved."""
    from playwright.sync_api import expect, sync_playwright

    ui = direct_server_with_data
    evidence = diagnostic_directory(tmp_path / "frames", f"settings-pair-{shape}-{engine}")
    report = {"candidate_identity": ui["candidate_identity"], "engine": engine, "shape": shape, "frames": {}, "steps": []}
    with _endpoint(["legacy-model"]) as (legacy_url, legacy_calls), _endpoint(["other-model"]) as (other_url, other_calls), \
            _endpoint(["dedicated-model"]) as (dedicated_url, dedicated_calls), sync_playwright() as pw:
        if shape == "legacy":
            _seed_legacy_install(ui, legacy_url)
        else:
            ui["stop_server"]()  # seed only while stopped
            path = ui["data_dir"] / "settings.json"
            settings = json.loads(path.read_text(encoding="utf-8"))
            settings.update({"OPENAI_COMPATIBLE_BASE_URL": dedicated_url, "OPENAI_COMPATIBLE_API_KEY": DEDICATED_KEY,
                             "OPENAI_BASE_URL": legacy_url, "OPENAI_API_KEY": LEGACY_KEY,
                             **dict.fromkeys(("OUROBOROS_MODEL", "OUROBOROS_MODEL_LIGHT", "OUROBOROS_MODEL_FALLBACKS"),
                                             "openai-compatible::dedicated-model")})
            path.write_text(json.dumps(settings), encoding="utf-8")
            ui["start_server"]()
        saved_before = (ui["data_dir"] / "settings.json").read_bytes()
        browser = getattr(pw, engine).launch(headless=True)
        page = browser.new_context(viewport={"width": 980, "height": 780}, device_scale_factor=2).new_page()
        # Inject only a dated provider fact into the real Settings response: the test exercises its display,
        # not a paid provider learning a cache-hint refusal. The server and every Test request stay real.
        def with_cache_hint(route):
            response = route.fetch()
            data = response.json()
            entry = next(item for item in data["_meta"]["connections"] if item["id"] == "openai-compatible")
            entry["cache_hints"] = {"refused": True, "learned_at": "2026-10-20T10:00:00Z"}
            route.fulfill(response=response, json=data)

        page.route("**/api/settings", with_cache_hint)
        uncaught = []
        page.on("pageerror", lambda error: uncaught.append(str(error)))
        page.goto(ui["url"], wait_until="domcontentloaded")
        page.wait_for_selector("#page-chat", timeout=30_000)
        page.click('[data-nav-page="settings"]')
        expect(page.locator("#settings-status")).to_have_text("Settings refreshed", timeout=30_000)
        for card in ("compatible", "openai"):
            toggle = page.locator(f'[data-provider-card="{card}"] [data-provider-toggle]')
            if toggle.get_attribute("aria-expanded") != "true":
                toggle.click()
        row = page.locator('[data-provider-card="compatible"] [data-connection-row][data-builtin="1"]')
        official = page.locator('[data-provider-card="openai"] [data-connection-row][data-builtin="1"]')
        status, meta, verdict = (row.locator(f"[data-connection-{part}]") for part in ("status", "meta", "test-status"))
        cache_note = row.locator("[data-connection-cache]")
        address, key, base = page.locator("#s-openai-base-url"), page.locator("#s-openai"), row.locator('[data-connection-field="base_url"]')
        kept = key.input_value()
        assert kept and kept != LEGACY_KEY, "Settings holds its own mask, never the key"
        calls = {legacy_url: legacy_calls, other_url: other_calls, dedicated_url: dedicated_calls}
        legacy_meta = f"Legacy OpenAI Base URL {legacy_url} with the OpenAI API key"
        dedicated_test = {"provider_id": "openai-compatible", "api_key_setting": "OPENAI_COMPATIBLE_API_KEY", "overrides": {}}

        mismatches = []

        def painted_as(name, check):
            """What the row should say; a miss is kept with the step, so one run shows every state it got wrong."""
            try:
                check()
            except AssertionError as error:
                mismatches.append({"step": name, "miss": " ".join(str(error).split())[:300]})

        def step(name, row_status, *, shown=None, kept_verdict=None, test=None, answer=None, reached=None):
            """The row as painted, a frame of it, then (when asked) its real Test: what it sent and who heard it."""
            painted_as(name, lambda: expect(status).to_have_text(row_status, timeout=3000))
            if row_status == "Not saved":
                painted_as(name, lambda: expect(cache_note).to_be_hidden(timeout=3000))
            else:
                painted_as(name, lambda: expect(cache_note).to_be_visible(timeout=3000))
                painted_as(name, lambda: expect(cache_note).to_contain_text("refused by this server"))
            if shown is None:
                painted_as(name, lambda: expect(meta).to_be_hidden(timeout=3000))
            else:
                painted_as(name, lambda: expect(meta).to_contain_text(shown, timeout=3000))
                if shown != legacy_meta:
                    painted_as(name, lambda: expect(meta).not_to_contain_text("Legacy", timeout=3000))
            if kept_verdict is not None:
                painted_as(name, lambda: expect(verdict).to_have_text(kept_verdict, timeout=3000))
            row.scroll_into_view_if_needed()
            painted = {"step": name, "row": status.inner_text(), "meta": meta.inner_text() if meta.is_visible() else "",
                       "verdict": verdict.inner_text(), "official": official.locator("[data-connection-status]").inner_text()}
            report["frames"][name] = _frame(page, evidence, name, row)
            if test is not None:
                before = {url: len(sent) for url, sent in calls.items()}
                painted["test_request"] = _test_row(page, row, answer)
                assert painted["test_request"] == test, painted
                # The probe is the one POST; a model-list read the page makes meanwhile is not this Test.
                probes = {url: [call for call in sent[before[url]:] if call["method"] == "POST"] for url, sent in calls.items()}
                painted["upstream"] = {url: sent for url, sent in probes.items() if sent}
                assert {url: _sent(sent) for url, sent in painted["upstream"].items()} == (reached or {}), painted
            report["steps"].append(painted)

        if shape == "legacy":
            expect(address).to_have_value(legacy_url)
            expect(base).to_have_value("")
            step("saved", "Ready", shown=legacy_meta, test=_legacy_test(legacy_url, kept), answer="Works",
                 reached={legacy_url: {f"Bearer {LEGACY_KEY}"}})
            # The legacy address, edited beside the row: the row no longer says Ready at the saved address, and the
            # kept key stays on its own address.
            address.fill(other_url)
            step("address-edited", "Not saved", kept_verdict="", test=_legacy_test(other_url, kept), answer=MOVED)
            address.fill(legacy_url)
            step("address-restored", "Ready", shown=legacy_meta, kept_verdict="")
            # The OpenAI key the pair travels with: its Clear, then its row's Remove, each followed by the saved mask.
            page.locator('.secret-clear[data-target="s-openai"]').click()
            step("key-cleared", "Not saved", test=_legacy_test(legacy_url, ""), answer="Not ready — Provider is not configured")
            key.fill(kept)
            step("key-restored", "Ready", shown=legacy_meta, kept_verdict="")
            official.locator("[data-connection-remove]").click()
            step("key-removed", "Not saved", kept_verdict="")
            key.fill(kept)
            step("key-restored-again", "Ready", shown=legacy_meta)
            # A dedicated address typed in the row selects the dedicated pair; the legacy address is then beside the
            # point, so editing it keeps the verdict. Clearing the row returns to the saved legacy pair.
            base.fill(dedicated_url)
            step("dedicated-typed", "Not saved", kept_verdict="", answer="Works", reached={dedicated_url: {None}},
                 test={"provider_id": "openai-compatible", "api_key_setting": "OPENAI_COMPATIBLE_API_KEY",
                       "overrides": {"OPENAI_COMPATIBLE_BASE_URL": dedicated_url}})
            address.fill(other_url)
            step("dedicated-typed-address-edited", "Not saved", kept_verdict="Works")
            address.fill(legacy_url)
            base.fill("")
            step("dedicated-cleared", "Ready", shown=legacy_meta, kept_verdict="")
        else:
            expect(base).to_have_value(dedicated_url)
            step("saved", "Ready", shown="Enabled", test=dedicated_test, answer="Works",
                 reached={dedicated_url: {f"Bearer {DEDICATED_KEY}"}})
            # The dedicated pair is the endpoint: the legacy address and the OpenAI key change nothing it sends.
            address.fill(other_url)
            step("legacy-address-edited", "Ready", shown="Enabled", kept_verdict="Works")
            page.locator('.secret-clear[data-target="s-openai"]').click()
            step("legacy-key-cleared", "Ready", shown="Enabled", kept_verdict="Works")
            key.fill(kept)
            address.fill(legacy_url)
            # Clearing the dedicated address selects the legacy pair: a draft whose Test reaches the legacy address
            # with the legacy key, until the legacy address moves too.
            base.fill("")
            step("dedicated-cleared", "Not saved", kept_verdict="", answer="Works", reached={legacy_url: {f"Bearer {LEGACY_KEY}"}},
                 test=_legacy_test(legacy_url, kept, OPENAI_COMPATIBLE_BASE_URL=""))
            address.fill(other_url)
            step("dedicated-cleared-address-edited", "Not saved", kept_verdict="", answer=MOVED,
                 test=_legacy_test(other_url, kept, OPENAI_COMPATIBLE_BASE_URL=""))
            address.fill(legacy_url)
            base.fill(dedicated_url)
            step("restored", "Ready", shown="Enabled", kept_verdict="")

        page.set_viewport_size({"width": 390, "height": 844})
        menu = page.locator("#page-settings [data-mobile-nav-toggle]")
        if menu.is_visible():
            if menu.get_attribute("aria-expanded") == "true":
                menu.click()
            wait_drawer_closed(page)
        address.fill(other_url)
        painted_as("narrow-address-edited", lambda: expect(status).to_have_text(
            "Ready" if shape == "both-pairs" else "Not saved", timeout=3000))
        row.scroll_into_view_if_needed()
        report["frames"]["narrow-address-edited"] = _frame(page, evidence, "narrow-address-edited", row)
        report["steps"].append({"step": "narrow-address-edited", "row": status.inner_text()})
        address.fill(legacy_url)
        assert not other_calls, "no request reached the edited address"
        assert (ui["data_dir"] / "settings.json").read_bytes() == saved_before, "nothing was saved"
        report.update(mismatches=mismatches, uncaught_page_errors=uncaught)
        (evidence / "settings-pair-proof.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        assert not mismatches, json.dumps(mismatches, indent=1)
        assert all(not frame["overflow"] and frame["body"][1] <= frame["viewport"][0] + 1 for frame in report["frames"].values())
        assert not uncaught
        browser.close()
