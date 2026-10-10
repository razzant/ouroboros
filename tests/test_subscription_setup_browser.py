"""Real UI modules against controlled subscription API responses, without runtime startup."""
from __future__ import annotations

import copy
import json
import os
from datetime import datetime, timedelta, timezone
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from urllib.parse import parse_qs, urlparse

import pytest
from tests.test_onboarding_complete_endpoint import (
    LIVE_SNAPSHOT, _profile, _profile_account,
    onboarding as onboarding,  # explicit re-export of the real atomic settings fixture
)

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]
WEB = Path(__file__).resolve().parents[1] / "web"


@pytest.fixture
def subscription_ui():
    if os.environ.get("OUROBOROS_RUN_UI_SMOKE") != "1":
        pytest.skip("Set OUROBOROS_RUN_UI_SMOKE=1 to run the browser flow")
    playwright = pytest.importorskip("playwright.sync_api")
    bootstrap = json.loads((WEB / "tests/fixtures/onboarding_bootstrap.json").read_text(encoding="utf-8"))
    bootstrap["freshInstall"] = True
    fixture = json.loads((WEB / "tests/fixtures/subscription_setup.json").read_text(encoding="utf-8"))
    fixture['status']['quota'] = [{
        'subject': {'harness': 'codex', 'subject_id': 'personal'}, 'freshness': 'fresh',
        'constraints': [{'id': 'window', 'used_ratio': 0.38,
                         'resets_at': (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()}],
    }]
    posts = []
    reads = []
    page_errors = []
    backend = {}
    settings = {
        **fixture["preview"]["model_settings"],
        "OUROBOROS_SUBAGENTS": fixture["preview"]["available_subagents"],
        "OUROBOROS_RUNTIME_MODE": "advanced",
        "OUROBOROS_CONTEXT_MODE": "max",
        "_meta": {"setup_contract": bootstrap["contract"]},
    }

    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            if self.path == "/onboarding":
                body = (WEB / "onboarding_template.html").read_text(encoding="utf-8").replace(
                    "__ONBOARDING_BOOTSTRAP__", json.dumps(bootstrap))
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(body.encode())
                return
            if self.path.startswith("/api/tasks/") and backend.get("task_gateway"):
                response = backend["task_gateway"].get(self.path)
                self.send_response(response.status_code)
                for key in ("content-type", "content-disposition", "content-length"):
                    if key in response.headers:
                        self.send_header(key, response.headers[key])
                self.end_headers()
                self.wfile.write(response.content)
                return
            self.path = self.path.removeprefix("/static")
            super().do_GET()

    def respond(route):
        request = route.request
        path = urlparse(request.url).path
        reads.append(request.url)
        body = {}
        if request.method == "POST":
            payload = request.post_data_json or {}
            posts.append((path, payload))
            if path == "/api/onboarding/subagents/preview":
                if backend.get("preview_client"):
                    result = backend["preview_client"].post(path, json=payload)
                    route.fulfill(status=result.status_code, content_type="application/json", body=result.text)
                    return
                if backend.get("recovery_error") and payload.get("skipSubscriptionPresets"):
                    route.fulfill(status=400, content_type="application/json", body=json.dumps({
                        "ok": False, "error": "Reviewer recovery unavailable.", "detail": backend["recovery_error"], "saved": False,
                    }))
                    return
                if backend.get("preview_error") and not payload.get("skipSubscriptionPresets"):
                    route.fulfill(status=503, content_type="application/json", body=json.dumps({
                        "ok": False, "error": "Automatic assignments unavailable.", "code": "models_unavailable",
                        "detail": backend["preview_error"], "can_skip": True, "saved": False,
                    }))
                    return
                body = copy.deepcopy(fixture["preview"])
                if not payload.get('subscriptionsConnected'):
                    body['model_settings'] = {key: payload.get(key, '') for key in body['model_settings']}
                    main = payload.get('OUROBOROS_MODEL')
                    # The endpoint mints the factory reviewer row of an API-only draft.
                    body['available_subagents'] = {'enabled': True, 'items': [{
                        'subagent_id': 'review-1', 'recommended_use': 'Independent review.', 'effort': 'high',
                        'route': {'kind': 'api_model', 'target_id': main},
                        'review_eligible': True, 'minted_from': 'factory_default'}] if main else []}
                if payload.get("skipSubscriptionPresets"):
                    body["model_settings"] = {key: payload.get(key, '') for key in body["model_settings"]}
                    catalog = payload.get("OUROBOROS_SUBAGENTS") or body["available_subagents"]
                    pin = payload.get("OUROBOROS_MODEL_ACCOUNTS", {}).get("main")
                    # Like ``review_rows_on_main``: every marked row moves onto Main; the rest stay.
                    catalog["items"] = [{
                        **{key: row[key] for key in ("subagent_id", "recommended_use", "effort", "review_eligible",
                                                     "minted_from", "enabled") if key in row},
                        "route": {"kind": "api_model", "target_id": payload["OUROBOROS_MODEL"],
                                  **({"credential_profile_id": pin} if pin else {})},
                    } if row.get("review_eligible") is True else row for row in catalog["items"]]
                    body["available_subagents"] = catalog
            elif path == "/api/onboarding/complete":
                assert "OUROBOROS_REVIEWER_SLOTS" not in payload, "the wizard never authors review lanes"
                assert isinstance(payload["OUROBOROS_SUBAGENTS"], dict)
                if backend.get("client"):
                    result = backend["client"].post(path, json=payload)
                    route.fulfill(status=result.status_code, content_type="application/json", body=result.text)
                    return
                body = {"ok": True, "runtime_mode": "advanced", "restart_required": False}
            elif path == "/api/settings":
                body = {"ok": True, "saved": True}
        elif path == "/api/claudexor/status":
            body = fixture["status"]
        elif path == "/api/settings":
            body = settings
        elif path == "/api/review-pool":
            body = backend.get("review_pool", {"pool": [], "excluded": [], "last_executions": {},
                                               "row_costs": {}, "config_error": "", "migration": None})
        elif path == "/api/model-catalog":
            body = copy.deepcopy(backend.get("catalog_response", fixture["catalog"]))
            if backend.get("catalog_status"):
                route.fulfill(status=backend["catalog_status"], content_type="application/json", body=json.dumps(body))
                return
            profile = parse_qs(urlparse(request.url).query).get("credential_profile_id", [""])[0]
            if profile == "work":
                body["items"][0].update(credential_profile_id="work", max_context_window=500000)
        elif path == "/api/onboarding":
            route.fulfill(status=204)
            return
        elif path == "/api/state":
            body = {"supervisor_ready": True, "active_chat_activities": [], "projects": []}
        elif path == "/api/projects":
            body = {"projects": []}
        elif path == "/api/chat/history":
            body = {"messages": [], "progress": []}
        elif path == "/api/health":
            body = {"ok": True, "version": "fixture"}
        route.fulfill(content_type="application/json", body=json.dumps(body))

    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(WEB)))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with playwright.sync_playwright() as pw:
            engine = os.environ.get("OUROBOROS_UI_BROWSER_ENGINE", "chromium")
            if engine not in {"chromium", "webkit", "firefox"}:
                raise ValueError(f"Unsupported browser engine: {engine}")
            browser = getattr(pw, engine).launch(
                headless=True, executable_path=os.environ.get("OUROBOROS_UI_BROWSER_EXECUTABLE") or None)
            try:
                def new_page(**options):
                    page = browser.new_page(viewport={"width": 1360, "height": 900},
                                            has_touch=os.environ.get("OUROBOROS_UI_HAS_TOUCH") == "1", **options)
                    page.route_web_socket('**/ws', lambda ws: ws.send(json.dumps({"type": "heartbeat"})))
                    page.route("**/api/**", respond)
                    page.on("pageerror", lambda error: page_errors.append(str(error)))
                    return page

                yield {"page": new_page(), "new_page": new_page, "url": f"http://127.0.0.1:{server.server_port}",
                       "posts": posts, "reads": reads, "fixture": fixture,
                       "settings": settings, "errors": page_errors, "backend": backend, "bootstrap": bootstrap}
                assert not page_errors
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def capture(page, name):
    root = os.environ.get("OUROBOROS_UI_EVIDENCE_DIR")
    if root:
        target = Path(root)
        target.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(target / f"{name}.png"), animations="disabled")


def test_codex_quick_setup_computes_skipped_steps_and_finishes_once(subscription_ui, onboarding):
    ui = subscription_ui
    ui['backend']['client'] = onboarding.client
    onboarding.calls['snapshot_payload'] = {
        **LIVE_SNAPSHOT, 'harnesses': [LIVE_SNAPSHOT['harnesses'][1]],
        'profiles': {'harnessAccounts': [_profile_account('codex', 'personal')],
                     'profiles': [_profile('codex', 'personal')]},
        'model_catalog': ui['fixture']['catalog']['items'],
    }
    ui['fixture']['status']['profiles']['profiles'] = ui['fixture']['status']['profiles']['profiles'][:1]
    page = ui["page"]
    page.goto(ui["url"] + "/onboarding")
    page.wait_for_selector('#quick-start-btn:not([hidden])')
    assert page.locator('.wizard-step').count() == 5
    capture(page, "accounts-connected")
    page.click('#quick-start-btn')
    page.wait_for_selector('.summary-card')
    assert "gpt-test" in page.locator('.summary-card').inner_text()
    reviewers = page.locator('.summary-kv').filter(has=page.get_by_text('Reviewers', exact=True))
    assert "codex=gpt-test" in reviewers.inner_text()
    capture(page, "summary-quick")
    page.click('#next-btn')
    page.wait_for_url(ui["url"] + "/")
    writes = [body for path, body in ui["posts"] if path == '/api/onboarding/complete']
    assert len(writes) == 1
    assert writes[0]["OPENROUTER_API_KEY"] == ""
    assert writes[0]["OUROBOROS_MODEL"] == 'claudexor::codex=gpt-test'
    # The reviewers are the catalog rows marked Reviewer; no lane is sent or kept.
    assert [row.get("review_eligible") for row in writes[0]["OUROBOROS_SUBAGENTS"]["items"]] == [True]
    assert not any(path == '/api/settings' for path, _ in ui["posts"])
    saved = onboarding.saved()
    assert saved['OUROBOROS_MODEL'] == writes[0]['OUROBOROS_MODEL']
    catalog = saved['OUROBOROS_SUBAGENTS']
    catalog = json.loads(catalog) if isinstance(catalog, str) else catalog
    posted = writes[0]["OUROBOROS_SUBAGENTS"]["items"]
    # The visible rows are saved as shown (the subscription preset may only append its own).
    assert [(row['subagent_id'], row.get('review_eligible')) for row in catalog['items'][:len(posted)]] == [('codex', True)]
    assert 'OUROBOROS_REVIEWER_SLOTS' not in saved
    assert not saved['OPENAI_API_KEY'] and not saved['OPENROUTER_API_KEY']
    assert onboarding.calls['supervisor'] == 1


@pytest.mark.parametrize("credential_harness, expected", [("codex", True), ("claude", False)])
def test_quick_start_requires_a_model_source_for_the_connected_harness(subscription_ui, credential_harness, expected):
    ui, page = subscription_ui, subscription_ui["page"]
    ui["fixture"]["catalog"]["model_sources"] = [
        {"id": "opaque-source", "label": "Managed models", "credentialHarness": credential_harness},
    ]
    with page.expect_response("**/api/model-catalog"):
        page.goto(ui["url"] + "/onboarding")
    page.wait_for_function(
        "expected => document.querySelector('#quick-start-btn').hidden !== expected", arg=expected)
    assert page.locator("#quick-start-btn").is_visible() is expected


def test_model_roles_pin_context_fallback_and_manual_draft_survive_preview(subscription_ui):
    ui = subscription_ui
    page = ui["page"]
    page.goto(ui["url"] + "/onboarding")
    page.wait_for_selector('#quick-start-btn:not([hidden])')
    page.click('#next-btn')
    page.wait_for_selector('[data-model-role="main"]')
    main = page.locator('[data-model-role="main"]')
    light = page.locator('[data-model-role="light"]')
    main.locator('[data-model-role-account]').select_option('personal')
    held_catalog = []
    page.route('**/api/model-catalog?*credential_profile_id=work', lambda route: held_catalog.append(route))
    light.locator('[data-model-role-account]').select_option('work')
    main.locator('summary').click()
    light.locator('summary').click()
    assert 'not known' in light.locator('[data-model-context-note]').inner_text()
    page.wait_for_function("() => document.querySelector('[data-model-role=light] [data-model-role-account]').value === 'work'")
    assert held_catalog
    work_catalog = copy.deepcopy(ui['fixture']['catalog'])
    work_catalog['items'][0].update(credential_profile_id='work', max_context_window=500000)
    for route in held_catalog:
        route.fulfill(content_type='application/json', body=json.dumps(work_catalog))
    page.unroute('**/api/model-catalog?*credential_profile_id=work')
    page.wait_for_function("() => document.querySelector('[data-model-role=light] [data-model-context-note]').textContent.includes('500,000')")
    assert '872,000' in main.locator('[data-model-context-note]').inner_text()
    main.locator('[data-model-role-context]').fill('1000000')
    assert 'set by you' in main.locator('[data-model-context-note]').inner_text()
    page.locator('[data-model-role="vision"] [data-model-role-account]').select_option('work')
    page.click('[data-model-add]')
    fallback = page.locator('[data-model-role-group="fallback"]')
    fallback.locator('[data-model-role-source]').select_option('subscription:codex')
    fallback.locator('[data-model-role-model]').fill('first')
    fallback.locator('[data-model-role-account]').select_option('work')
    page.click('[data-model-add]')
    fallback.locator('[data-model-role-model]').last.fill('openai::second')
    fallback.locator('[data-model-up]').last.click()
    main.locator('[data-model-role-model]').fill('owner-model')
    page.evaluate('window.scrollTo(0, 0)')
    capture(page, "models-desktop")
    # A reviewer is a catalog row marked Reviewer, edited where the row is.
    page.locator('[data-collapse="subagents"] > summary').click()
    reviewer = page.locator('[data-subagent-row]').first
    assert reviewer.locator('[data-subagent-field="review_eligible"]').is_checked()
    assert page.locator('[data-review-pool-count]').inner_text() == 'Reviewers: 1'
    default_effort = reviewer.locator('[data-subagent-field="effort"] option[value=""]')
    assert default_effort.inner_text() == 'Default (reviews at high)'
    reviewer.locator('[data-subagent-field="effort"]').select_option('high')
    assert reviewer.locator('[data-subagent-field="effort"]').input_value() == 'high'
    reviewer.scroll_into_view_if_needed()
    capture(page, "reviewer-row-editable")
    page.click('#next-btn')
    assert page.locator('[data-reviewers-note]').inner_text().startswith('Reviewers: 1 — the rows marked Reviewer')
    capture(page, "review-editable")
    page.click('#next-btn')
    page.wait_for_selector('[data-collapse="api-budget"]')
    assert not page.locator('[data-collapse="api-budget"]').evaluate('(el) => el.open')
    capture(page, "budget-subscription")
    page.click('#next-btn')
    page.wait_for_selector('.summary-card')
    page.wait_for_function("() => !document.querySelector('.wizard-error').textContent")
    summary = page.locator('.summary-card')
    assert '1,000,000 tokens, set by you' in summary.inner_text()
    vision = summary.locator('.summary-kv').filter(has=page.get_by_text('Vision', exact=True))
    assert 'Uses Main' in vision.inner_text()
    assert 'Account: work' in vision.inner_text()
    reviewers = summary.locator('.summary-kv').filter(has=page.get_by_text('Reviewers', exact=True))
    assert 'codex=gpt-test · Effort: high' in reviewers.inner_text()
    capture(page, "summary-manual")
    page.click('#next-btn')
    page.wait_for_url(ui["url"] + "/")
    body = [body for path, body in ui["posts"] if path == '/api/onboarding/complete'][0]
    assert body['OUROBOROS_MODEL'] == 'claudexor::codex=owner-model'
    assert body['OUROBOROS_MODEL_LIGHT'] == 'claudexor::codex=gpt-test'
    assert body['OUROBOROS_MODEL_ACCOUNTS']['main'] == 'personal'
    assert body['OUROBOROS_MODEL_ACCOUNTS']['light'] == 'work'
    assert body['OUROBOROS_MODEL_ACCOUNTS']['vision'] == 'work'
    assert body['OUROBOROS_MODEL_ACCOUNTS']['fallback'] == ['', 'work']
    assert body['OUROBOROS_MODEL_CONTEXT_WINDOWS']['main'] == 1000000
    assert body['OUROBOROS_MODEL_FALLBACKS'] == 'openai::second, claudexor::codex=first'
    (row,) = body['OUROBOROS_SUBAGENTS']['items']
    assert (row['subagent_id'], row['review_eligible'], row['effort']) == ('codex', True, 'high')


def test_settings_accounts_and_model_roles_use_the_same_compact_components(subscription_ui):
    ui = subscription_ui
    page = ui['page']
    page.goto(ui['url'] + '/#settings')
    page.wait_for_selector('#settings-model-roles .model-role-row', state='attached')
    assert page.locator('[data-settings-tab="providers"]').inner_text() == 'Accounts'
    assert page.locator('[data-settings-panel="providers"] #harness-accounts-section').count() == 1
    assert page.locator('[data-settings-panel="agents"] #harness-accounts-section').count() == 0
    page.click('[data-settings-tab="models"]')
    page.wait_for_function("() => document.querySelector('[data-settings-tab=models]').getAttribute('aria-selected') === 'true'")
    main = page.locator('[data-model-role="main"]')
    main.locator('[data-model-role-account]').select_option('work')
    inputs = main.locator('.model-role-controls > input, .model-role-controls > select')
    tops = inputs.evaluate_all('(els) => els.filter(el => !el.hidden).map(el => el.getBoundingClientRect().top)')
    assert max(tops) - min(tops) <= 2
    assert main.locator('input').first.evaluate('(el) => getComputedStyle(el).fontSize') == '14px'
    capture(page, 'settings-models-desktop')
    page.set_viewport_size({'width': 390, 'height': 844})
    page.wait_for_function("() => document.querySelector('#primary-sidebar').getBoundingClientRect().right <= 1")
    assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
    capture(page, 'settings-models-narrow')


def test_cursor_only_does_not_claim_model_access_and_api_only_finishes(subscription_ui):
    ui = subscription_ui
    ui['fixture']['status']['profiles']['profiles'] = [{
        'profile': {'profile_id': 'cursor-only', 'harness_id': 'cursor', 'enabled': True},
        'status': {'verification': 'passed'},
    }]
    page = ui['page']
    page.goto(ui['url'] + '/onboarding')
    page.wait_for_function("() => document.querySelector('[data-agent-family=cursor]').textContent.includes('Connected')")
    assert page.locator('#quick-start-btn').is_hidden()
    assert page.locator('#next-btn').is_disabled()
    assert 'Main still needs Codex' in page.locator('#agents-outcome').inner_text()
    # Removing the optional agent gives the API-only path the same five steps.
    ui['fixture']['status']['profiles']['profiles'] = []
    page.reload()
    page.locator('[data-collapse="api-access"] > summary').click()
    page.locator('#openai-key').fill('fixture-api-credential')
    page.click('#next-btn')
    page.wait_for_selector('#main-model')
    assert page.locator('#main-model').input_value()
    for _ in range(3):
        page.click('#next-btn')
    page.wait_for_selector('.summary-card')
    assert 'no API key' not in page.locator('.summary-card').inner_text()
    page.click('#next-btn')
    page.wait_for_url(ui['url'] + '/')
    body = [body for path, body in ui['posts'] if path == '/api/onboarding/complete'][0]
    assert body['OPENAI_API_KEY'] == 'fixture-api-credential'
    assert body['OUROBOROS_MODEL'].startswith('openai::')
    assert not body['subscriptionsConnected']


@pytest.mark.parametrize('edit_after_recovery', [False, True, 'main'])
def test_failed_preview_allows_manual_main_and_visible_reviewer_recovery_before_save(subscription_ui, onboarding, edit_after_recovery):
    ui, page = subscription_ui, subscription_ui['page']
    ui['backend'].update(preview_client=onboarding.client, client=onboarding.client)
    onboarding.calls['snapshot_payload'] = {
        **LIVE_SNAPSHOT,
        'harnesses': [{**LIVE_SNAPSHOT['harnesses'][1], 'models': []}],
        'profiles': {'harnessAccounts': [_profile_account('codex', 'personal')],
                     'profiles': [_profile('codex', 'personal')]},
        'model_catalog': ui['fixture']['catalog']['items'],
    }
    page.goto(ui['url'] + '/onboarding')
    page.wait_for_selector('#next-btn:not([disabled])')
    page.wait_for_function("() => document.querySelector('#onboarding-access-note').textContent.includes('listed no models')")
    assert not any(path == '/api/onboarding/complete' for path, _ in ui['posts'])
    page.set_viewport_size({'width': 390, 'height': 844})
    assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
    capture(page, 'accounts-preview-failure-narrow')
    page.set_viewport_size({'width': 1360, 'height': 900})
    page.click('#next-btn')
    main = page.locator('[data-model-role="main"]')
    assert main.locator('[data-model-role-model]').input_value() == ''
    assert page.locator('#next-btn').is_disabled()
    main.locator('[data-model-role-source]').select_option('subscription:codex')
    main.locator('[data-model-role-model]').fill('owner-main')
    main.locator('[data-model-role-account]').select_option('personal')
    for _ in range(3):
        page.click('#next-btn')
    page.wait_for_selector('.summary-card')
    page.wait_for_selector('#skip-presets-btn:not([hidden])')
    page.click('#skip-presets-btn')
    page.wait_for_function("() => document.querySelector('.wizard-inline-note')?.textContent.includes('Reviewers were assigned to Main')")
    assert not any(path == '/api/onboarding/complete' for path, _ in ui['posts']), 'Recovery is a preview, not a write'
    assert not onboarding.settings_path.exists()
    reviewers = page.locator('.summary-kv').filter(has=page.get_by_text('Reviewers', exact=True))
    assert 'claudexor::codex=owner-main · Account: personal' in reviewers.inner_text()
    capture(page, 'manual-main-reviewer-recovery')
    if edit_after_recovery == 'main':
        for _ in range(3):
            page.click('#back-btn')
        page.locator('[data-model-role="main"] [data-model-role-model]').fill('new-main')
        for _ in range(3):
            page.click('#next-btn')
        page.wait_for_selector('#skip-presets-btn:not([hidden])')
        assert page.locator('#next-btn').is_enabled()
        assert 'Main changed; reviewers keep the assignments shown above' in page.locator('.wizard-inline-note').inner_text()
        assert 'owner-main' in reviewers.inner_text() and 'new-main' not in reviewers.inner_text()
    elif edit_after_recovery:
        # The reviewers are rows of the Models step's catalog: edit one there.
        for _ in range(3):
            page.click('#back-btn')
        page.locator('[data-collapse="subagents"] > summary').click()
        first = page.locator('[data-subagent-row]').filter(has=page.locator('[data-subagent-field="review_eligible"]:checked')).first
        assert first.count() == 1
        first.locator('[data-subagent-field="model"]').fill('owner-deep')
        for _ in range(3):
            page.click('#next-btn')
        page.wait_for_function("() => !document.querySelector('.wizard-error').textContent")
        assert 'claudexor::codex=owner-deep' in reviewers.inner_text()
    page.click('#next-btn')
    page.wait_for_url(ui['url'] + '/')
    bodies = [body for path, body in ui['posts'] if path == '/api/onboarding/complete']
    assert len(bodies) == 1 and bodies[0]['skipSubscriptionPresets'] is True
    assert bodies[0]['OUROBOROS_MODEL'] == ('claudexor::codex=new-main' if edit_after_recovery == 'main' else 'claudexor::codex=owner-main')
    saved = onboarding.saved()
    catalog = saved['OUROBOROS_SUBAGENTS']
    catalog = json.loads(catalog) if isinstance(catalog, str) else catalog
    marked = [row for row in catalog['items'] if row.get('review_eligible') is True]
    assert marked == [row for row in bodies[0]['OUROBOROS_SUBAGENTS']['items'] if row.get('review_eligible') is True]
    assert 'OUROBOROS_REVIEWER_SLOTS' not in saved
    assert onboarding.calls['supervisor'] == 1
    assert marked, 'Use Main kept the reviewers: none were dropped'
    for index, row in enumerate(marked):
        expected = 'owner-deep' if edit_after_recovery is True and index == 0 else 'owner-main'
        assert row['route'] == {'kind': 'api_model', 'target_id': f'claudexor::codex={expected}',
                                'credential_profile_id': 'personal'}


def test_failed_main_reviewer_recovery_does_not_latch_finish(subscription_ui):
    # A refused recovery preview must leave the wizard on its ordinary path:
    # the packaged setup window has no reload, so the owner needs both the
    # normal Finish and a working Use Main retry after the backend refuses once.
    ui, page = subscription_ui, subscription_ui['page']
    ui['backend'].update(preview_error='Agent model discovery unavailable.',
                         recovery_error='A Main account pin requires a managed model source.')
    page.goto(ui['url'] + '/onboarding')
    page.wait_for_selector('#next-btn:not([disabled])')
    page.click('#next-btn')
    main = page.locator('[data-model-role="main"]')
    main.locator('[data-model-role-source]').select_option('subscription:codex')
    main.locator('[data-model-role-model]').fill('owner-main')
    for _ in range(3):
        page.click('#next-btn')
    page.wait_for_selector('#skip-presets-btn:not([hidden])')
    page.click('#skip-presets-btn')
    page.wait_for_function("() => document.querySelector('.wizard-error').textContent.includes('managed model source')")
    page.wait_for_selector('#skip-presets-btn:not([hidden])')
    page.click('#next-btn')
    page.wait_for_function("() => document.querySelector('.wizard-error').textContent")
    assert 'Use Main for reviewers to prepare' not in page.locator('.wizard-error').inner_text()
    assert not any(path == '/api/onboarding/complete' for path, _ in ui['posts'])
    ui['backend'].pop('recovery_error')
    page.click('#skip-presets-btn')
    page.wait_for_function("() => document.querySelector('.wizard-inline-note')?.textContent.includes('Reviewers were assigned to Main')")
    page.click('#next-btn')
    page.wait_for_url(ui['url'] + '/')
    bodies = [body for path, body in ui['posts'] if path == '/api/onboarding/complete']
    assert len(bodies) == 1 and bodies[0]['skipSubscriptionPresets'] is True
    assert bodies[0]['OUROBOROS_MODEL'] == 'claudexor::codex=owner-main'


@pytest.mark.parametrize('http_status', [200, 500])
def test_accounts_catalog_failure_is_visible_and_retry_keeps_the_draft(subscription_ui, http_status):
    ui, page = subscription_ui, subscription_ui['page']
    ui['backend'].update(preview_error='Agent model discovery unavailable.', catalog_status=http_status,
                         catalog_response={'error': 'Raw model service unavailable.'})
    page.goto(ui['url'] + '/onboarding')
    page.wait_for_function("() => document.querySelector('#onboarding-access-note').textContent.includes('Raw model service unavailable')")
    assert page.locator('#next-btn').is_disabled()
    page.locator('[data-collapse="api-access"] > summary').click()
    page.locator('#openai-key').fill('owner-draft-credential')
    ui['backend'].pop('catalog_status')
    ui['backend'].pop('catalog_response')
    page.click('#onboarding-access-retry')
    page.wait_for_selector('#quick-start-btn:not([hidden])')
    assert page.locator('#openai-key').input_value() == 'owner-draft-credential'
    assert page.locator('[data-collapse="api-access"]').evaluate('(el) => el.open')
    assert not any(path == '/api/onboarding/complete' for path, _ in ui['posts'])


@pytest.mark.parametrize('fresh_install', [False, True])
def test_declared_source_with_failed_inventory_preserves_stored_or_owner_authored_defaults(subscription_ui, fresh_install):
    ui, page = subscription_ui, subscription_ui['page']
    ui['bootstrap']['freshInstall'] = fresh_install
    initial = ui['bootstrap']['initialState']['mainModel']
    ui['backend'].update(preview_error='Agent inventory unavailable.', catalog_response={
        'items': [], 'model_sources': ui['fixture']['catalog']['model_sources'],
        'read_state': 'partial', 'errors': [{'error': 'Account model inventory unavailable.'}],
    })
    page.goto(ui['url'] + '/onboarding')
    page.wait_for_selector('#next-btn:not([disabled])')
    page.click('#next-btn')
    main = page.locator('[data-model-role="main"]')
    if fresh_install:
        assert main.locator('[data-model-role-model]').input_value() == ''
        main.locator('[data-model-role-model]').fill(initial)
        page.click('#back-btn')
        page.click('#onboarding-access-retry')
        page.wait_for_selector('#onboarding-access-retry:not([disabled])')
        page.click('#next-btn')
    assert main.locator('[data-model-role-model]').input_value() == initial
    assert page.locator('#next-btn').is_disabled()
    assert 'Main uses OpenRouter' in page.locator('.wizard-error').inner_text()
    main.locator('[data-model-role-source]').select_option('subscription:codex')
    main.locator('[data-model-role-model]').fill('typed-without-inventory')
    assert page.locator('#next-btn').is_enabled()


def test_late_catalog_does_not_restore_access_after_the_account_disconnects(subscription_ui):
    ui, page = subscription_ui, subscription_ui['page']
    ui['backend']['preview_error'] = 'Agent inventory unavailable.'
    held = []
    page.route('**/api/model-catalog', lambda route: held.append(route))
    page.goto(ui['url'] + '/onboarding')
    page.wait_for_function("() => document.querySelector('#onboarding-access-note').textContent.includes('Checking model sources')")
    assert held
    ui['fixture']['status']['profiles']['profiles'] = []
    page.evaluate("async () => (await import('/static/modules/claudexor_status_store.js')).claudexorStatus.refresh()")
    page.wait_for_function("() => !document.querySelector('[data-agent-family=codex]').textContent.includes('Connected')")
    for route in held:
        route.fulfill(content_type='application/json', body=json.dumps(ui['fixture']['catalog']))
    page.unroute('**/api/model-catalog')
    assert page.locator('#next-btn').is_disabled()
    assert page.locator('#quick-start-btn').is_hidden()


def _zai_key(page, plan):
    page.locator('[data-collapse="api-access"] > summary').click()
    page.locator('[data-collapse="more-providers"] > summary').click()
    page.locator('#zai-key').fill('zai-fixture-credential')
    page.locator('#zai-plan').fill(plan)


@pytest.mark.parametrize('plan, owner_vision', [('', None), ('coding', None), ('payg', ''), ('payg', 'zai::glm-ocr')])
def test_fresh_zai_setup_proposes_the_image_capable_vision_the_owner_may_change(subscription_ui, plan, owner_vision):
    """PR #1560: a fresh Z.ai-only setup shows and saves Flash in Vision; an owner's
    clear or replacement survives Back/Next and is what gets saved."""
    ui, page = subscription_ui, subscription_ui['page']
    ui['fixture']['status']['profiles']['profiles'] = []
    page.goto(ui['url'] + '/onboarding')
    _zai_key(page, plan)
    page.click('#next-btn')
    vision = page.locator('[data-model-role="vision"] [data-model-role-model]')
    vision.wait_for()
    assert vision.input_value().endswith('glm-5.3-flash')
    capture(page, f'zai-models-{plan or "default"}')
    if owner_vision is not None:
        vision.fill(owner_vision)
        page.click('#back-btn')
        page.click('#next-btn')
        assert vision.input_value() == owner_vision.rpartition('::')[2]   # the source picker holds "zai"
    for _ in range(3):
        page.click('#next-btn')
    page.wait_for_selector('.summary-card')
    capture(page, f'zai-summary-{plan or "default"}-{"owner" if owner_vision is not None else "default"}')
    page.click('#next-btn')
    page.wait_for_url(ui['url'] + '/')
    body = [body for path, body in ui['posts'] if path == '/api/onboarding/complete'][0]
    assert (body['ZAI_API_KEY'], body['ZAI_PLAN'], body['OUROBOROS_MODEL']) == ('zai-fixture-credential', plan, 'zai::glm-5.3')
    assert body['OUROBOROS_MODEL_VISION'] == ('zai::glm-5.3-flash' if owner_vision is None else owner_vision)


def test_a_connected_subscription_keeps_the_zai_vision_default_out(subscription_ui):
    """The subscription path owns model selection: a Z.ai key beside it adds no Vision default."""
    ui, page = subscription_ui, subscription_ui['page']
    page.goto(ui['url'] + '/onboarding')
    page.wait_for_selector('#next-btn:not([disabled])')
    _zai_key(page, '')
    page.click('#next-btn')
    page.wait_for_function("() => document.querySelector('[data-model-role=\"main\"] [data-model-role-model]')?.value === 'gpt-test'")
    assert page.locator('[data-model-role="vision"] [data-model-role-model]').input_value() == ''
    capture(page, 'zai-with-subscription-models')


@pytest.mark.parametrize('scenario, expected_vision', [
    ('key-first', ''), ('local-first', ''), ('remote-default', ''),
    ('remote-main-edit', ''), ('remote-clear', ''),
    ('remote-custom', 'zai::glm-ocr'), ('remote-explicit-default', 'zai::glm-5.3-flash'),
    ('local-fallback', 'zai::glm-5.3-flash'),
])
def test_local_main_with_zai_defaults_only_untouched_vision(subscription_ui, monkeypatch, scenario, expected_vision):
    ui, page = subscription_ui, subscription_ui['page']
    ui['fixture']['status']['profiles']['profiles'] = []
    page.goto(ui['url'] + '/onboarding')

    def disclosure(name):
        details = page.locator(f'[data-collapse="{name}"]')
        if not details.evaluate('el => el.open'):
            details.locator(':scope > summary').click()

    def key():
        disclosure('api-access')
        disclosure('more-providers')
        page.locator('#zai-key').fill('zai-fixture-credential')

    def local():
        disclosure('api-access')
        disclosure('local-model')
        page.locator('#local-source').fill('/models/local-test.gguf')
        mode = 'fallback' if scenario == 'local-fallback' else 'all'
        page.locator(f'[data-local-mode="{mode}"]').click()

    vision = page.locator('[data-model-role="vision"] [data-model-role-model]')
    if scenario == 'local-first':
        local()
        key()
    else:
        key()
        if scenario.startswith('remote-'):
            page.click('#next-btn')
            assert vision.input_value() == 'glm-5.3-flash'
            if scenario == 'remote-main-edit':
                page.locator('[data-model-role="main"] [data-model-role-model]').fill('owner-main')
            elif scenario in {'remote-clear', 'remote-custom', 'remote-explicit-default'}:
                # Even an explicit choice equal to the suggestion belongs to the owner.
                vision.fill(expected_vision.rpartition('::')[2])
            page.click('#back-btn')
        local()
    page.click('#next-btn')
    capture(page, f'local-zai-models-{scenario}')
    assert vision.input_value() == expected_vision.rpartition('::')[2]
    for _ in range(3):
        page.click('#next-btn')
    page.wait_for_selector('.summary-card')
    capture(page, f'local-zai-summary-{scenario}')
    page.click('#next-btn')
    page.wait_for_url(ui['url'] + '/')
    writes = [body for path, body in ui['posts'] if path == '/api/onboarding/complete']
    assert len(writes) == 1
    body = writes[0]
    assert body['OUROBOROS_MODEL_VISION'] == expected_vision
    assert body['LOCAL_ROUTING_MODE'] == ('fallback' if scenario == 'local-fallback' else 'all')
    if scenario == 'remote-main-edit':
        assert body['OUROBOROS_MODEL'] == 'zai::owner-main'
    if not expected_vision:
        # Continue the actual browser draft through the validator and caption
        # consumer: a present Z.ai key must not turn local Main into a paid call.
        from ouroboros.llm import LLMClient
        from ouroboros.settings_setup_contract import validate_setup_payload
        from ouroboros.vision_routing import VisionRoutingContext, prepare_messages_for_send

        settings, error = validate_setup_payload(body, {})
        assert not error
        assert settings['USE_LOCAL_MAIN'] is True
        for key, value in settings.items():
            if isinstance(value, (str, bool, int, float)):
                monkeypatch.setenv(key, str(value))
        monkeypatch.setenv('OUROBOROS_IMAGE_INPUT_MODE', 'caption')
        calls = []
        monkeypatch.setattr(LLMClient, 'vision_query', lambda *_a, **_k: calls.append(_k))
        message = [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {
            'url': 'data:image/png;base64,aW1hZ2U='}}]}]
        result = prepare_messages_for_send(message, routing=VisionRoutingContext(
            settings['OUROBOROS_MODEL'], LLMClient(), {}, use_local=settings['USE_LOCAL_MAIN']))
        assert calls == []
        assert result[0]['content'][0]['text'] == (
            '[image omitted: our local llama.cpp transport lane cannot carry images; no caption route is available]')


@pytest.mark.parametrize('choice', ['generated', 'main-edited', 'deliberate-Flash'])
def test_review_shortcut_reconciles_local_vision_before_preview(
        subscription_ui, onboarding, monkeypatch, record_property, choice):
    """The shortcut must reconcile the same draft as Next, before preview or save."""
    import httpx
    from ouroboros import net_transport
    from ouroboros.llm import LLMClient
    from ouroboros.vision_routing import VisionRoutingContext, prepare_messages_for_send

    ui, page = subscription_ui, subscription_ui['page']
    profiles = ui['fixture']['status']['profiles']['profiles']
    ui['fixture']['status']['profiles']['profiles'] = []
    ui['backend'].update(preview_client=onboarding.client, client=onboarding.client)
    page.goto(ui['url'] + '/onboarding')
    _zai_key(page, 'coding')
    page.click('#next-btn')
    vision = page.locator('[data-model-role="vision"] [data-model-role-model]')
    assert vision.input_value() == 'glm-5.3-flash'
    if choice == 'deliberate-Flash':
        vision.fill('glm-5.3-flash')
    elif choice == 'main-edited':
        page.locator('[data-model-role="main"] [data-model-role-model]').fill('owner-main')
    capture(page, f'shortcut-initial-vision-{choice}')
    page.click('#back-btn')
    # Observe a newly connected Codex account without a live login or daemon.
    ui['fixture']['status']['profiles']['profiles'] = profiles
    onboarding.calls['snapshot_payload'] = {
        **LIVE_SNAPSHOT, 'harnesses': [LIVE_SNAPSHOT['harnesses'][1]],
        'profiles': {'harnessAccounts': [_profile_account('codex', 'personal')],
                     'profiles': [_profile('codex', 'personal')]},
        'model_catalog': ui['fixture']['catalog']['items'],
    }
    page.evaluate("async () => (await import('/static/modules/claudexor_status_store.js')).claudexorStatus.refresh()")
    page.wait_for_selector('#quick-start-btn:not([hidden])')
    page.wait_for_selector('#onboarding-access-retry:not([disabled])')
    page.locator('[data-collapse="local-model"] > summary').click()
    page.locator('#local-source').fill('/models/local-test.gguf')
    page.locator('[data-local-mode="all"]').click()
    capture(page, f'shortcut-local-accounts-{choice}')
    page.click('#quick-start-btn')
    page.wait_for_selector('.summary-card')
    summary = page.locator('.summary-kv').filter(has=page.get_by_text('Vision', exact=True)).inner_text()
    capture(page, f'shortcut-summary-{choice}')
    page.click('#next-btn')
    page.wait_for_url(ui['url'] + '/')
    writes = [body for path, body in ui['posts'] if path == '/api/onboarding/complete']
    assert len(writes) == 1
    saved = onboarding.saved()
    assert saved['USE_LOCAL_MAIN'] is True
    for key, value in saved.items():
        if isinstance(value, (str, bool, int, float)):
            monkeypatch.setenv(key, str(value))
    monkeypatch.setenv('OUROBOROS_IMAGE_INPUT_MODE', 'caption')
    wire = []

    def answer(request):
        wire.append({'url': str(request.url), 'body': json.loads(request.content)})
        return httpx.Response(200, json={
            'id': 'shortcut-test', 'object': 'chat.completion', 'created': 0, 'model': 'glm-5.3-flash',
            'choices': [{'index': 0, 'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': 'Dark red'}}],
            'usage': {'prompt_tokens': 9, 'completion_tokens': 2, 'total_tokens': 11},
        })

    monkeypatch.setattr(net_transport, 'remote_httpx_transport', lambda *_a, **_k: httpx.MockTransport(answer))
    image = {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,aW1hZ2U='}}
    result = prepare_messages_for_send([{'role': 'user', 'content': [image]}], routing=VisionRoutingContext(
        saved['OUROBOROS_MODEL'], LLMClient(), {}, use_local=saved['USE_LOCAL_MAIN']))
    record_property('caption_consumer', json.dumps({'vision': saved['OUROBOROS_MODEL_VISION'], 'wire': wire, 'result': result}))
    expected = 'zai::glm-5.3-flash' if choice == 'deliberate-Flash' else ''
    local_previews = [body for path, body in ui['posts']
                      if path == '/api/onboarding/subagents/preview' and body.get('LOCAL_ROUTING_MODE') == 'all']
    assert local_previews and all(body['OUROBOROS_MODEL_VISION'] == expected for body in local_previews)
    assert writes[0]['subscriptionsConnected'] is True
    assert writes[0]['OUROBOROS_MODEL_VISION'] == saved['OUROBOROS_MODEL_VISION'] == expected
    if choice == 'main-edited':
        assert saved['OUROBOROS_MODEL'] == 'zai::owner-main'
    if expected:
        assert 'glm-5.3-flash' in summary and 'Uses Main' not in summary
        assert len(wire) == 1
        assert wire[0]['url'] == 'https://api.z.ai/api/coding/paas/v4/chat/completions'
        assert wire[0]['body']['model'] == 'glm-5.3-flash'
        assert image in wire[0]['body']['messages'][0]['content']
        assert result[0]['content'][0]['text'] == '[image caption: Dark red]'
    else:
        assert 'Uses Main' in summary
        assert wire == []
        assert result[0]['content'][0]['text'] == (
            '[image omitted: our local llama.cpp transport lane cannot carry images; no caption route is available]')
