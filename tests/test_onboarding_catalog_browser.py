"""Wizard model discovery through its production host with controlled API reads."""
from __future__ import annotations

import copy
import json

import pytest

from ouroboros.test_evidence import diagnostic_directory
from tests.test_subscription_setup_browser import subscription_ui as subscription_ui

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]


@pytest.mark.parametrize("viewport", [(980, 780), (900, 900), (390, 844), (1360, 900)])
@pytest.mark.parametrize("theme", ["light", "dark"])
def test_connection_discovery_feedback_stays_visible_on_compact_screens(subscription_ui, viewport, theme, tmp_path):
    from playwright.sync_api import expect

    ui, page = subscription_ui, subscription_ui["page"]
    page.set_viewport_size(dict(zip(("width", "height"), viewport)))
    ui["bootstrap"]["initialState"].update(compatibleBaseUrl="http://127.0.0.1:1/v1")
    response = {"status": 400, "body": {
        "error": "This server's address changed since its key was saved. Enter its key again to list its models.",
        "code": "masked_key_endpoint_changed"}}
    page.route("**/api/openai-compatible/models", lambda route: route.fulfill(
        status=response["status"], json=response["body"]))
    page.goto(ui["url"] + "/onboarding", wait_until="networkidle")
    page.evaluate("theme => document.documentElement.dataset.theme = theme", theme)
    evidence = diagnostic_directory(tmp_path / "frames", "discovery-feedback")
    compact = viewport[0] <= 900 or viewport[1] <= 820
    columns = 1 if viewport[0] <= 760 else 2
    measurements = {}

    def measure_grid(name, selector):
        grid = page.locator(selector).first
        expect(grid).to_be_visible()
        facts = grid.evaluate("""el => ({
            columns: getComputedStyle(el).gridTemplateColumns.split(/\\s+/).length,
            width: el.clientWidth, scrollWidth: el.scrollWidth,
            titleSize: getComputedStyle(document.querySelector('.step-title')).fontSize,
            bodyWidth: document.body.clientWidth, bodyScrollWidth: document.body.scrollWidth,
        })""")
        measurements[name] = facts
        assert facts["columns"] == columns, facts
        assert facts["scrollWidth"] <= facts["width"] + 1, facts
        assert facts["bodyScrollWidth"] <= facts["bodyWidth"] + 1, facts
        assert facts["titleSize"] == "24px", facts

    # Accounts' ordinary fields stay paired on a short desktop and stack on a
    # phone. Compactness comes from hiding help, never from shrinking text.
    api_access = page.locator('[data-collapse="api-access"]')
    if not api_access.evaluate("el => el.open"):
        api_access.locator("summary").first.click()
    measure_grid("accounts", ".field-grid")
    note = page.locator('.field-grid .field-note').first
    expect(note).to_be_hidden() if compact else expect(note).to_be_visible()
    footer = page.locator('.footer-copy')
    expect(footer).to_be_hidden() if compact else expect(footer).to_be_visible()
    label = page.locator('.field-grid label').first
    expect(label).to_have_css("font-size", "12px")
    page.screenshot(path=str(evidence / f"{viewport[0]}x{viewport[1]}-accounts.png"), animations="disabled")
    page.click("#next-btn")
    status = page.locator("#compatible-load-status")
    for name, code, body, text in (
        ("error", 400, response["body"], "Enter its key again"),
        ("empty", 200, {"models": []}, "No models listed"),
        ("success", 200, {"models": ["same-model"]}, "1 model listed"),
    ):
        response.update(status=code, body=body)
        page.click("#load-compatible-models")
        expect(status).to_contain_text(text)
        page.screenshot(path=str(evidence / f"{viewport[0]}x{viewport[1]}-{name}.png"), animations="disabled")
        expect(status).to_be_visible()
        expect(status).to_have_css("font-size", "12px")

    page.click("#next-btn")
    # The Review choices are two columns; the separate Access grid deliberately
    # has its own count, so it is not part of this two-column contract.
    measure_grid("review", ".wizard-choice-grid")
    page.click("#next-btn")
    budget = page.locator('[data-collapse="api-budget"]')
    if not budget.evaluate("el => el.open"):
        budget.locator("summary").click()
    measure_grid("budget", ".grid.two")
    (evidence / "layout.json").write_text(json.dumps(measurements, indent=2))


def test_connection_discovery_survives_general_refresh_without_replacing_other_sources(subscription_ui, tmp_path):
    from playwright.sync_api import expect

    ui, page = subscription_ui, subscription_ui['page']
    initial = ui['bootstrap']['initialState']
    initial.update(openaiKey='fixture', compatibleBaseUrl='http://127.0.0.1:1/v1',
                   mainModel='openai::same-model', lightModel='openai-compatible@alpha::same-model',
                   safetyModel='claudexor::service=gpt-test', modelAccounts={'safety': 'personal'},
                   connections=[{'kind': 'openai-compatible', 'name': name, 'label': name.title(),
                                 'base_url': 'http://127.0.0.1:1/v1'} for name in ('alpha', 'beta')])
    # No provider is contacted: these responses exercise the served wizard and shared editor.
    general = copy.deepcopy(ui['fixture']['catalog'])
    general['model_sources'][0]['id'] = 'service'  # credential harness and source id are independent
    general['items'][0]['value'] = 'claudexor::service=gpt-test'
    general['items'][0]['source_id'] = 'service'
    general['items'] += [{'value': value} for value in (
        'openai::same-model', 'openai-compatible::same-model',
        'openai-compatible@alpha::old-only', 'openai-compatible@beta::same-model')]
    general.update(observed_at='general-observation', coverage={'complete': True})
    ui['backend']['catalog_response'] = general
    ui['fixture']['preview']['model_settings'].update(
        OUROBOROS_MODEL=initial['mainModel'], OUROBOROS_MODEL_LIGHT=initial['lightModel'],
        OUROBOROS_MODEL_SAFETY=initial['safetyModel'], OUROBOROS_MODEL_ACCOUNTS=initial['modelAccounts'])
    discovered = ['same-model']
    page.route('**/api/openai-compatible/models', lambda route: route.fulfill(json={'models': discovered}))
    page.goto(ui['url'] + '/onboarding', wait_until='networkidle')
    expect(page.locator('#onboarding-access-retry')).to_be_enabled()
    page.click('#next-btn')
    main = page.locator('[data-model-role="main"]')
    light = page.locator('[data-model-role="light"]')
    safety = page.locator('[data-model-role="safety"]')
    # Protect these owner choices from the automatic preset while probing refreshes.
    main.locator('[data-model-role-model]').fill('same-model')

    def choices(row):
        field = row.locator('[data-model-role-model]')
        field.focus()
        values = page.locator('[role="listbox"]:visible [data-model-value]').evaluate_all(
            'nodes => nodes.map(node => node.dataset.modelValue)')
        field.press('Escape')
        return values

    def kept():
        expect(main.locator('[data-model-role-source]')).to_have_value('api:openai')
        expect(main.locator('[data-model-role-model]')).to_have_value('same-model')
        expect(light.locator('[data-model-role-source]')).to_have_value('api:openai-compatible@alpha')
        expect(light.locator('[data-model-role-model]')).to_have_value('same-model')
        expect(safety.locator('[data-model-role-account]')).to_have_value('personal')

    assert choices(main) == ['same-model']
    page.locator('#compatible-model-source').select_option('openai-compatible@alpha')
    page.click('#load-compatible-models')
    expect(page.locator('#compatible-load-status')).to_contain_text('1 model listed')
    assert choices(main) == ['same-model'], 'a connection list must not replace unrelated API suggestions'
    assert choices(light) == ['same-model']
    expect(main.locator('[data-model-role-source] option[value="subscription:service"]')).to_have_text('Codex (model)')
    expect(main.locator('[data-model-role-status]')).to_have_text('')
    kept()
    evidence = diagnostic_directory(tmp_path / 'frames', 'mixed-catalog')
    report = []
    for label, response, status, expected_main, note in (
        ('transport', {'error': 'Controlled offline'}, 503, ['same-model'], 'could not be read'),
        ('failed', {'items': [], 'model_sources': [], 'errors': [{'error': 'Controlled failure'}]}, 200,
         ['same-model'], 'could not be read'),
        ('partial', {'items': [{'value': 'openai::same-model'}], 'model_sources': [], 'partial': True}, 200,
         ['same-model'], 'Some model sources'),
        ('full', {'items': [{'value': 'openai::same-model'}], 'model_sources': general['model_sources']}, 200,
         ['same-model'], ''),
        ('empty', {'items': [], 'model_sources': [], 'errors': []}, 200, [], ''),
    ):
        ui['backend'].update(catalog_response=response, catalog_status=status)
        page.click('#back-btn')
        with page.expect_response('**/api/model-catalog'):
            page.click('#onboarding-access-retry')
        expect(page.locator('#onboarding-access-retry')).to_be_enabled()
        page.click('#next-btn')
        kept()
        assert choices(main) == expected_main, label
        assert choices(light) == ['same-model'], f'{label} refresh must retain unsaved connection discovery'
        if note:
            expect(main.locator('[data-model-role-status]')).to_contain_text(note)
        else:
            expect(main.locator('[data-model-role-status]')).to_have_text('')
        if label == 'empty':
            expect(main.locator('[data-model-role-source] option[value="subscription:service"]')).to_have_count(0)
            expect(main.locator('[data-model-role-source]')).to_contain_text('No model sources listed')
        report.append({'refresh': label, 'main': choices(main), 'connection': choices(light),
                       'status': main.locator('[data-model-role-status]').inner_text()})
    # A successful empty connection list clears only that source's suggestions, even after a remount.
    discovered.clear()
    page.locator('#compatible-model-source').select_option('openai-compatible@alpha')
    page.click('#load-compatible-models')
    expect(page.locator('#compatible-load-status')).to_contain_text('No models listed')
    assert choices(light) == []
    kept()
    page.click('#back-btn')
    page.click('#next-btn')
    assert choices(light) == []
    kept()
    page.screenshot(path=str(evidence / 'mixed-catalog-empty.png'), animations='disabled')
    (evidence / 'mixed-catalog.json').write_text(json.dumps(report, indent=2))
