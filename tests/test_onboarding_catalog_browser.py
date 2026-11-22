"""Wizard model discovery through its production host with controlled API reads."""
from __future__ import annotations

import copy
import json

import pytest
from tests._wizard_walk import step_order

from ouroboros.test_evidence import diagnostic_directory
from tests.test_subscription_setup_browser import subscription_ui as subscription_ui

pytestmark = [pytest.mark.ui_browser]


def test_old_core_subscription_only_setup_can_retry_without_implicitly_reading_providers(subscription_ui):
    from playwright.sync_api import expect

    ui, page = subscription_ui, subscription_ui["page"]
    del ui["bootstrap"]["contract"]["scopedModelCatalogs"]
    page.goto(ui["url"] + "/onboarding", wait_until="networkidle")
    retry = page.locator("#onboarding-access-retry")
    expect(retry).to_be_enabled()
    expect(page.locator("#next-btn")).to_be_disabled()
    assert not any("/model-catalog" in url for url in ui["reads"])
    with page.expect_response("**/api/onboarding/model-catalog"):
        retry.click()
    expect(page.locator("#next-btn")).to_be_enabled()
    assert not any("/api/model-catalog?view=" in url or "provider_id=" in url for url in ui["reads"])


def test_wizard_uses_the_current_subscription_list_when_an_earlier_full_read_arrives(subscription_ui):
    from playwright.sync_api import expect

    ui, page = subscription_ui, subscription_ui["page"]
    pending = []
    full = copy.deepcopy(ui["fixture"]["catalog"])
    page.route("**/api/onboarding/model-catalog", lambda route: pending.append(route))
    page.goto(ui["url"] + "/onboarding", wait_until="networkidle")
    next_step = page.locator("#next-btn")
    expect(next_step).to_be_enabled()
    with page.expect_request("**/api/onboarding/model-catalog"):
        page.locator("#onboarding-access-retry").click()
    ui["fixture"]["catalog"]["model_sources"] = []
    ui["fixture"]["status"]["profiles"]["profiles"].pop()
    with page.expect_response("**/api/model-catalog?view=sources"):
        page.evaluate("async () => (await import('/onboarding/static/modules/claudexor_status_store.js')).claudexorStatus.refresh()")
    expect(next_step).to_be_disabled()
    with page.expect_response("**/api/onboarding/model-catalog"):
        pending[0].fulfill(json=full)
    expect(page.locator("#onboarding-access-note")).not_to_contain_text("Checking model sources")
    expect(next_step).to_be_disabled()


def test_wizard_waiting_on_a_pin_only_suggests_models_from_that_account(subscription_ui):
    from playwright.sync_api import expect

    ui, page = subscription_ui, subscription_ui["page"]
    ui["bootstrap"]["initialState"].update(mainModel="claudexor::codex=saved", automaticModels=[])
    ui["fixture"]["preview"]["model_settings"] = {}
    catalog = ui["fixture"]["catalog"]
    for item in catalog["items"]:
        model = item["credential_profile_id"] + "-exclusive"
        item.update(id=model, value="claudexor::codex=" + model, name=model, label=model)
    pending = []
    page.route("**/api/model-catalog?source_id=codex&credential_profile_id=work", lambda r: pending.append(r))
    page.goto(ui["url"] + "/onboarding")
    with page.expect_response("**/api/onboarding/model-catalog"):
        page.locator("#onboarding-access-retry").click()
    page.click("#next-btn")
    main = page.locator('[data-model-role="main"]')
    model = main.locator("[data-model-role-model]")
    model.fill("exclusive")
    choices = page.locator('#' + model.get_attribute("aria-controls"))
    expect(choices.locator('[data-model-value="personal-exclusive"]')).to_be_visible()
    expect(choices.locator('[data-model-value="work-exclusive"]')).to_be_visible()
    model.press("Escape")
    with page.expect_request("**/api/model-catalog?source_id=codex&credential_profile_id=work"):
        main.locator("[data-model-role-account]").select_option("work")
    model.fill("exclusive")
    expect(choices.locator('[data-model-value="personal-exclusive"]')).to_have_count(0)
    for route in pending:
        route.fulfill(json={**catalog, "items": [row for row in catalog["items"] if row["credential_profile_id"] == "work"]})
    expect(choices.locator('[data-model-value="work-exclusive"]')).to_be_visible()
    expect(model).to_have_value("exclusive")


def test_wizard_lists_a_subscription_while_the_selected_draft_provider_is_slow(subscription_ui):
    from playwright.sync_api import expect

    ui, page = subscription_ui, subscription_ui["page"]
    ui["bootstrap"]["initialState"].update(
        mainModel="openai::saved", automaticModels=[], lightModel="", fallbackModel="",
        visionModel="", safetyModel="", consciousnessModel="")
    ui["fixture"]["preview"]["model_settings"] = {}
    pending = []
    page.route("**/api/onboarding/model-catalog?provider_id=openai", lambda route: pending.append(route))
    page.goto(ui["url"] + "/onboarding")
    page.locator('[data-collapse="api-access"] > summary').click()
    page.locator("#openai-key").fill("openai-draft-test-credential")
    page.locator("#openrouter-key").fill("openrouter-unselected-test-credential")
    with page.expect_request("**/api/onboarding/model-catalog?provider_id=openai"):
        page.click("#next-btn")
    main = page.locator('[data-model-role="main"]')
    expect(main.locator("[data-model-role-model]")).to_have_value("saved")
    main.locator("[data-source-picker]").click()
    page.get_by_role("option", name="Codex (model)", exact=True).click()
    model = main.locator("[data-model-role-model]")
    model.fill("gpt")
    page.locator(f'#{model.get_attribute("aria-controls")} [data-model-value="gpt-test"]').click()
    assert pending and pending[0].request.method == "POST"
    assert pending[0].request.post_data_json["OPENAI_API_KEY"] == "openai-draft-test-credential"
    for route in pending:
        route.fulfill(json={"items": [{"value": "openai::a-late-model"}], "errors": [], "model_sources": []})
    expect(model).to_have_value("gpt-test")
    expect(main.locator("[data-model-role-source]")).to_have_value("subscription:codex")
    assert not any(url.endswith(("/api/onboarding/model-catalog", "/api/model-catalog",
                                 "provider_id=openrouter")) for url in ui["reads"])
    assert not any("/api/model-catalog?provider_id=" in url for url in ui["reads"]), "draft keys read only the setup endpoint"


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
    evidence = diagnostic_directory(tmp_path / "frames", "discovery-feedback", publish=True)
    compact = viewport[0] <= 900 or viewport[1] <= 820
    columns = 1 if viewport[0] <= 760 else 2
    measurements = {}

    def measure_grid(name, selector, expected_columns=columns):
        grid = page.locator(selector).first
        expect(grid).to_be_visible()
        facts = grid.evaluate("""el => ({
            columns: getComputedStyle(el).gridTemplateColumns.split(/\\s+/).length,
            width: el.clientWidth, scrollWidth: el.scrollWidth,
            titleSize: getComputedStyle(document.querySelector('.step-title')).fontSize,
            bodyWidth: document.body.clientWidth, bodyScrollWidth: document.body.scrollWidth,
        })""")
        measurements[name] = facts
        assert facts["columns"] == expected_columns, facts
        assert facts["scrollWidth"] <= facts["width"] + 1, facts
        assert facts["bodyScrollWidth"] <= facts["bodyWidth"] + 1, facts
        assert facts["titleSize"] == "24px", facts

    # Accounts' ordinary fields stay paired on a short desktop and stack on a
    # phone. Compactness drops ornament (the rail's sub-labels), never a
    # field's note or the size of the text.
    api_access = page.locator('[data-collapse="api-access"]')
    if not api_access.evaluate("el => el.open"):
        api_access.locator("summary").first.click()
    measure_grid("accounts", ".field-grid")
    note = page.locator('.field-grid .field-note').first
    expect(note).to_be_visible()
    rail = page.locator('.wizard-step-copy').first
    expect(rail).to_be_hidden() if compact else expect(rail).to_be_visible()
    # The footer's aside leaves only a narrow window, where the pinned footer would take the step's own lines.
    aside = page.locator('.footer-copy')
    expect(aside).to_be_hidden() if viewport[0] <= 900 else expect(aside).to_be_visible()
    # A warning in the aside is no ornament: every footer keeps it while no reason beside Continue takes its slot.
    assert page.evaluate("""() => { const aside = document.querySelector('.footer-copy');
        aside.classList.add('footer-warning'); const shown = getComputedStyle(aside).display !== 'none';
        aside.classList.remove('footer-warning'); return shown; }"""), viewport
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
    # Reviewers are chosen under Models; this step has the four Access choices and the reviewer summary.
    measure_grid("access", ".runtime-mode-card .wizard-choice-grid", 1 if viewport[0] <= 760 else 4)
    expect(page.locator('[data-reviewers-note]')).to_be_visible()
    assert page.locator('[data-runtime-mode]').count() == 4
    # What each choice means, and the step's own sentence, are readable at every size: the step scrolls under
    # the pinned footer instead.
    expect(page.locator('.step-copy')).to_be_visible()
    meanings = page.locator('.wizard-choice p')
    assert meanings.count() == 4, meanings.count()
    for index in range(meanings.count()):
        expect(meanings.nth(index)).to_be_visible()
    expect(page.locator('#next-btn')).to_be_in_viewport()
    page.click("#next-btn")
    if "consciousness" in step_order(page):  # its own step before limits on a root with no state yet
        expect(page.locator('[data-consciousness-choice]')).to_have_count(2)
        expect(page.locator('#next-btn')).to_be_in_viewport()
        page.click("#next-btn")
    if "skills" in step_order(page):  # then Skills, while this root's starter list never ran
        expect(page.locator('[data-starter-skills="step"]')).to_have_count(1)
        expect(page.locator('#next-btn')).to_be_in_viewport()
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
    # These are saved owner choices, not the wizard's replaceable first-run proposals (#2969).
    initial.update(openaiKey='fixture', compatibleBaseUrl='http://127.0.0.1:1/v1',
                   automaticModels=[],
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
    # DESIGN "Onboarding density": an API role without a published price says so.
    expect(main.locator('[data-model-role-status]')).to_have_text('Price unknown')
    kept()
    evidence = diagnostic_directory(tmp_path / 'frames', 'mixed-catalog', publish=True)
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
        with page.expect_response('**/api/onboarding/model-catalog'):
            page.click('#onboarding-access-retry')
        expect(page.locator('#onboarding-access-retry')).to_be_enabled()
        page.click('#next-btn')
        kept()
        assert choices(main) == expected_main, label
        assert choices(light) == ['same-model'], f'{label} refresh must retain unsaved connection discovery'
        if note:
            expect(main.locator('[data-model-role-status]')).to_contain_text(note)
        else:
            expect(main.locator('[data-model-role-status]')).to_have_text('Price unknown')
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
