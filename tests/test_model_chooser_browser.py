"""Editable model chooser on real Settings/actor/reviewer consumers, no model calls."""
from __future__ import annotations

import pytest

from tests import test_subscription_role_routes_browser as roles
from tests.settings_browser_helpers import open_settings, press_editor_save

pytestmark = [pytest.mark.ui_browser]
subscription_ui = roles.subscription_ui
role_ui = roles.role_ui


def catalog(ui):
    value = ui["fixture"]["catalog"]
    value["items"] = [{"value": f"choice-{i:02}", "id": f"choice-{i:02}"} for i in range(30)]
    return value


def select_model_field(ui, consumer):
    roles.configure_mixed(ui)
    # The catalog above is unprefixed, so every consumer is put on the same
    # OpenRouter API lane the suggestions belong to.
    lane = roles.api_lane(ui)
    data = catalog(ui)
    if consumer in ['Models', 'Fallback']:
        page = roles.open_models(ui, 'main' if consumer == 'Models' else 'fallback/1')
        page.evaluate("""detail => document.dispatchEvent(new CustomEvent(
            'settings-model-catalog:updated', {detail}))""", data)
        row = page.locator('[data-model-role]').first
        row.locator('[data-model-role-source]').select_option(lane)
        return page, row.locator('[data-model-role-model]')
    row = roles.open_agent(ui, 'direct')
    page = ui['page']
    page.evaluate("""detail => document.dispatchEvent(new CustomEvent(
        'settings-model-catalog:updated', {detail}))""", data)
    # Actor: a row marked Reviewer, so the reviewer chooser is this same field.
    row.locator('[data-subagent-field="route"]').select_option(lane)
    return page, row.locator('[data-subagent-field="model"]')


@pytest.mark.parametrize("consumer", ["Models", "Fallback", "Actor"])
def test_chooser_keyboard_free_values_and_catalog_identity(role_ui, consumer):
    page, field = select_model_field(role_ui, consumer)
    field.fill('choice')
    popup = page.locator('[id="' + field.get_attribute('aria-controls') + '"]')
    assert popup.is_visible()
    assert popup.locator('[role="option"]').count() == 30
    field.press('ArrowDown')
    assert field.get_attribute('aria-activedescendant')
    field.press('Escape')
    assert field.input_value() == 'choice'
    assert not popup.is_visible()
    field.press('ArrowDown')
    field.press('Enter')
    assert field.input_value() == 'choice-00'
    assert field.get_attribute('aria-expanded') == 'false'
    field.fill('owner/new-unknown-id')
    field.evaluate("""e => { window.keptModelField = e; e.setSelectionRange(3, 7);
        e.dispatchEvent(new CompositionEvent('compositionstart', {bubbles:true})); }""")
    page.evaluate("""() => document.dispatchEvent(new CustomEvent(
        'settings-model-catalog:updated', {detail:{items:[],model_sources:[]}}))""")
    assert field.evaluate('e => e === window.keptModelField && document.activeElement === e')
    assert field.evaluate('e => [e.selectionStart,e.selectionEnd]') == [3, 7]
    assert field.input_value() == 'owner/new-unknown-id'
    field.evaluate("e => e.dispatchEvent(new CompositionEvent('compositionend', {bubbles:true}))")
    field.press('Tab')
    assert field.input_value() == 'owner/new-unknown-id'
    assert not popup.is_visible()
    roles.capture(page, f"chooser-{consumer.lower()}-unknown-kept")


def test_agent_session_uses_same_chooser_and_keeps_native_short_choices(role_ui):
    roles.configure_mixed(role_ui)
    row = roles.open_agent(role_ui, 'agent')
    page = role_ui['page']
    model = row.locator('[data-subagent-field="model"]')
    assert model.evaluate('e => e.tagName') == 'INPUT'
    assert model.get_attribute('role') == 'combobox'
    for field in ['route', 'account', 'effort']:
        assert row.locator(f'[data-subagent-field="{field}"]').evaluate('e => e.tagName') == 'SELECT'
    model.fill('future-model')
    page.evaluate("""detail => document.dispatchEvent(new CustomEvent(
        'settings-model-catalog:updated', {detail}))""", role_ui['fixture']['catalog'])
    assert model.input_value() == 'future-model'
    assert row.locator('[data-subagent-field="account"]').input_value() == 'personal'
    roles.capture(page, 'chooser-session-reference-preserved')


def test_incomplete_reviewer_holds_own_save_until_corrected(role_ui):
    roles.configure_mixed(role_ui)
    reviewer = roles.open_agent(role_ui, 'native')
    page = role_ui['page']
    assert reviewer.locator('[data-subagent-field="review_eligible"]').is_checked()
    model = reviewer.locator('[data-subagent-field="model"]')
    model.fill('')
    page.locator('[data-editor-save]').click()
    assert not [path for path, _ in role_ui['posts'] if path in ('/api/settings', '/api/owner/safety-model')]
    assert model.input_value() == ''
    assert model.get_attribute('aria-invalid') == 'true'
    model.fill('owner-unknown-review-model')
    assert model.get_attribute('aria-invalid') == 'false'
    assert press_editor_save(page).startswith('Saved')
    writes = roles.roster_writes(role_ui)
    assert len(writes) == 1
    roles.capture(page, 'chooser-reviewer-corrected-save')


@pytest.fixture
def touch_role_ui(monkeypatch, request):
    monkeypatch.setenv('OUROBOROS_UI_HAS_TOUCH', '1')
    return request.getfixturevalue('role_ui')


def test_chooser_touch_selection(touch_role_ui):
    page, field = select_model_field(touch_role_ui, 'Models')
    page.set_viewport_size({'width': 390, 'height': 600})
    field.tap()
    field.fill('choice-')
    popup = page.locator('[id="' + field.get_attribute('aria-controls') + '"]')
    popup.locator('[data-model-value="choice-01"]').tap()
    assert field.input_value() == 'choice-01'
    assert field.get_attribute('aria-expanded') == 'false'
    roles.capture(page, 'chooser-touch-selected')


def test_partial_numeric_context_blocks_save_and_preserves_auto(role_ui):
    roles.configure_mixed(role_ui)
    role_ui['settings']['OUROBOROS_MODEL_CONTEXT_WINDOWS'] = {'main': 8192}
    page = roles.open_models(role_ui)
    row = page.locator('[data-model-role="main"]')
    row.locator('summary').click()
    field = row.locator('[data-model-role-context]')
    assert field.input_value() == '8192'
    field.fill('')
    field.press('1')
    field.press('e')
    assert field.evaluate('e => e.validity.badInput')
    page.locator('[data-editor-save]').click()
    assert not [path for path, _ in role_ui['posts'] if path == '/api/settings']
    assert field.get_attribute('aria-invalid') == 'true'
    field.fill('16384')
    assert field.get_attribute('aria-invalid') == 'false'
    press_editor_save(page)
    saved = roles.settings_saves(role_ui)[-1]
    assert saved['OUROBOROS_MODEL_CONTEXT_WINDOWS']['main'] == 16384
    row.locator('summary').click()
    field.fill('')
    assert not field.evaluate('e => e.validity.badInput')
    press_editor_save(page)
    saved = roles.settings_saves(role_ui)[-1]
    assert saved['OUROBOROS_MODEL_CONTEXT_WINDOWS']['main'] == 0
    roles.capture(page, 'numeric-context-corrected-auto')
    row.locator('summary').click()
    field.press('1')
    field.press('e')
    assert field.evaluate('e => e.validity.badInput')
    assert page.locator('[data-editor-discard]').is_visible()


def test_chooser_scrolls_with_mobile_keyboard_boundary(role_ui):
    page = role_ui['page']
    page.set_viewport_size({'width': 390, 'height': 844})
    page.add_init_script("""(() => {
        const viewport = new EventTarget();
        Object.assign(viewport, {width:390,height:844,offsetLeft:0,offsetTop:0,scale:1});
        Object.defineProperty(window,'visualViewport',{value:viewport,configurable:true});
        window.testViewport = viewport;
    })();""")
    page, field = select_model_field(role_ui, 'Models')
    field.click()
    field.fill('choice')
    popup = page.locator('[id="' + field.get_attribute('aria-controls') + '"]')
    page.evaluate("""() => { testViewport.height=400;testViewport.dispatchEvent(new Event('resize')); }""")
    page.wait_for_function("document.body.classList.contains('keyboard-open')")
    assert popup.evaluate('e => e.scrollHeight > e.clientHeight')
    def gesture(target):
        return target.evaluate("""node => {
            for (const [type,y] of [['touchstart',220],['touchmove',170]]) {
                const event=new Event(type,{bubbles:true,cancelable:true});
                Object.defineProperty(event,'touches',{value:[{clientX:100,clientY:y}]});
                node.dispatchEvent(event);
                if(type==='touchmove') return event.defaultPrevented;
            }
        }""")
    assert gesture(popup) is False
    assert gesture(page.locator('#page-settings .app-page-header')) is True
    assert field.evaluate('e => document.activeElement === e')
    roles.capture(page, 'chooser-keyboard-touch-scroll')


@pytest.mark.parametrize("width,height", [(320, 480), (390, 600), (640, 360), (641, 540),
    (760, 600), (768, 600), (980, 600), (981, 540), (1440, 900), (1920, 900)])
def test_chooser_popup_reachable_in_narrow_short_view(role_ui, width, height):
    page, field = select_model_field(role_ui, 'Models')
    page.set_viewport_size({'width': width, 'height': height})
    field.click()
    field.fill('choice')
    popup = page.locator('[id="' + field.get_attribute('aria-controls') + '"]')
    assert popup.is_visible()
    box = popup.bounding_box()
    roles.capture(page, f'chooser-open-{width}-{height}')
    assert box and box['x'] >= 0 and box['y'] >= 0
    assert box['x'] + box['width'] <= width + 1
    assert box['y'] + box['height'] <= height + 1, {'popup': box, 'anchor': field.bounding_box()}
    option = popup.locator('[role="option"]').nth(29)
    option.scroll_into_view_if_needed()
    option.click()
    assert field.input_value() == 'choice-29'
    roles.capture(page, f'chooser-narrow-{width}-{height}')


def vendor_scene(ui, count=1):
    """The real consumer catalog, including an empty-independent API source and six editors."""
    import copy
    from urllib.parse import parse_qs, urlparse
    from tests.test_harness_maintenance_browser import _inventory_row

    roles.configure_mixed(ui)
    roles.api_lane(ui, "openai")
    profiles = ui["fixture"]["status"]["profiles"]["profiles"]
    seed = copy.deepcopy(profiles[0])
    profiles[:] = []
    for index in range(count):
        profile = copy.deepcopy(seed)
        profile["profile"]["profile_id"] = "personal" if not index else f"account-{index}"
        profiles.append(profile)
    actors = [{"subagent_id": f"worker{index}", "recommended_use": f"Work {index}", "review_eligible": index == 0,
               "route": {"kind": "api_model" if index == 0 else "agent_session",
                         "target_id": "claudexor::opaque-source=choice-old" if index == 0 else "codex=choice-old"}}
              for index in range(4)]
    actors += [{"subagent_id": "current", "recommended_use": "Other program", "route": {"kind": "agent_session", "target_id": "claude=choice-old"}},
               {"subagent_id": "api", "recommended_use": "Direct API", "route": {"kind": "api_model", "target_id": "openai::choice-api"}}]
    ui["settings"].update(OUROBOROS_SUBAGENTS={"enabled": True, "items": actors}, OUROBOROS_MODEL="claudexor::opaque-source=choice-old")
    world = {"version": "2.0.0", "op": None, "new": False, "posts": [], "inventories": [], "catalogs": 0,
             "catalog_requests": [], "catalog_revision": "2.0.0"}
    row = _inventory_row("codex")
    row.update(updateAvailable="yes", newest={"status": "available", "version": "3.0.0", "observedAt": "2026-11-10T10:00:00Z"})
    other = _inventory_row("claude")
    other.update(updateAvailable="no", newest={"status": "available", "version": "2.0.0", "observedAt": "2026-11-10T10:00:00Z"})

    def catalogs():
        version = world["version"]
        producer = {"path": "/vendor/codex", "version": version}
        models = [{"id": "choice-old"}] + ([{"id": "choice-new"}] if world["new"] else [])
        raw, native, flat = [], [], []
        for profile in profiles:
            pin = profile["profile"]["profile_id"]
            common = {"credentialProfileId": pin, "provenance": "vendor", "observedAt": "2026-11-10T12:00:00Z", "sourceProgram": producer, "models": models}
            raw.append({"credentialProfileId": pin, "catalog": {**common, "source": "opaque-source"}, "availability": "available", "problem": None})
            native.append({"credentialProfileId": pin, "catalog": {**common, "harnessId": "codex", "source": "api"}, "availability": "available", "problem": None})
            flat += [{**model, "value": f"claudexor::opaque-source={model['id']}", "source_id": "opaque-source", "source": "Claudexor", "credential_profile_id": pin} for model in models]
        ui["fixture"]["catalog"] = {"items": flat + [{"id": "choice-api", "value": "openai::choice-api"}],
            "model_sources": [{"id": "opaque-source", "label": "Codex", "credentialHarness": "codex"}],
            "account_catalogs": [{"source": "opaque-source", "accounts": raw, "partial": False}], "errors": []}
        ui["fixture"]["status"]["harnesses"] = [
            {"id": "codex", "display_name": "Codex", "enabled": True, "status": "ok", "models": [{**model, "credential_profile_id": account["credentialProfileId"]} for account in native for model in models],
             "model_catalog": {"harnessId": "codex", "accounts": native, "partial": False}},
            {"id": "claude", "display_name": "Claude Code", "enabled": True, "status": "ok", "models": [{"id": "choice-old"}],
             "model_catalog": {"harnessId": "claude", "sourceProgram": {"path": "/vendor/claude", "version": "2.0.0"}, "models": [{"id": "choice-old"}]}},
        ]

    catalogs()

    def catalog_reply(route):
        world["catalogs"] += 1
        world["catalog_requests"].append(route.request.url)
        if world["catalog_revision"] != world["version"]:
            catalogs()
            world["catalog_revision"] = world["version"]
        payload = copy.deepcopy(ui["fixture"]["catalog"])
        source = parse_qs(urlparse(route.request.url).query).get("source_id", [""])[0]
        if source:
            payload["items"] = [item for item in payload["items"] if item.get("source_id") == source]
        route.fulfill(json=payload)

    def maintenance(route):
        path = urlparse(route.request.url).path
        if path.endswith("/harnesses"):
            if world["catalog_revision"] != world["version"]:
                catalogs()
                world["catalog_revision"] = world["version"]
            world["inventories"].append(route.request.url)
            row["selection"]["version"] = world["version"]
            row["updateAvailable"] = "no" if world["new"] else "yes"
            row["operation"] = world["op"]
            row["observedAt"] = "2026-11-10T12:01:00Z"
            route.fulfill(json={"observedAt": row["observedAt"], "harnesses": [row, other]})
        elif route.request.method == "POST":
            world["posts"].append(route.request.post_data_json)
            world["op"] = {"id": "vendor-update", "harness": "codex", "state": "running", "phase": "installing",
                "before": {"path": "/vendor/codex", "version": "2.0.0"}, "after": None, "target": {"kind": "latest", "version": None},
                "mutation": "unknown", "termination": "not_applicable", "outcome": "unknown", "limitations": [], "progress": ["Vendor is updating."]}
            route.fulfill(json=world["op"])
        else:
            route.fulfill(json=world["op"])

    ui["page"].route("**/api/model-catalog*", catalog_reply)
    ui["page"].route("**/api/claudexor/maintenance/**", maintenance)
    world["catalogs_refresh"] = catalogs
    world["other_inventory"] = other
    return world


def vendor_role_field(page, slot):
    if slot == "fallback":
        page.locator("[data-model-add]").click()
    group = page.locator(f'[data-model-role-group="{slot}"]')
    local = group.locator('[data-model-local]')
    if local.count() and local.is_checked():
        local.uncheck()
    row = group.locator('.model-role-row').last
    row.locator("[data-model-role-source]").select_option("subscription:opaque-source")
    return row.locator("[data-model-role-model]")


def vendor_consumer(ui, consumer):
    page = ui['page']
    if consumer not in ('Reviewer', 'Agent'):
        slot = 'main' if consumer == 'Models' else 'fallback/1' if consumer == 'Fallback' else consumer.lower()
        open_settings(page, 'models/role/' + slot)
        row = page.locator('[data-model-role]').first
        local = row.locator('[data-model-local]')
        if local.count() and local.is_checked():
            local.uncheck()
        row.locator('[data-model-role-source]').select_option('subscription:opaque-source')
        return row.locator('[data-model-role-model]')
    row = roles.open_agent(ui, 'worker0' if consumer == 'Reviewer' else 'worker1', fresh=False)
    return row.locator('[data-subagent-field="model"]')


def open_circle_editor(ui):
    page = ui["page"]
    page.route("**/api/places", lambda route: route.fulfill(json={
        "connectors": [], "places": [], "circles": [{"circle_id": "work", "name": "Work", "text": "",
        "text_author": "owner", "revision": 1, "model": ""}],
    }))
    page.goto(ui["url"] + "/#circles")
    page.locator('[data-circles-act="edit"][data-circle="work"]').click()
    return page.locator('[data-circles-draft="work:model"]')


def general_catalog_reads(world):
    from urllib.parse import urlsplit

    # Settings roles can renew source-scoped catalogs when any reader publishes
    # a catalog. Circles and the shared settlement path read the general list.
    return sum(not urlsplit(url).query for url in world["catalog_requests"])


@pytest.mark.parametrize("width", [1440, 390])
@pytest.mark.parametrize("account_count", [1, 18])
def test_circle_editor_uses_its_catalog_and_names_only_its_supplied_entries(role_ui, width, account_count):
    import copy
    import re
    from playwright.sync_api import expect

    ui, page = role_ui, role_ui["page"]
    world = vendor_scene(ui, account_count)
    page.set_viewport_size({"width": width, "height": 950})
    field = open_circle_editor(ui)
    expect(field).to_have_attribute("data-model-programs", re.compile("2.0.0"))
    reads, inventories = world["catalogs"], len(world["inventories"])
    general_reads = general_catalog_reads(world)
    expect(page.locator("[data-model-footer]:visible")).to_have_count(0)
    field.fill("choice")
    footer = page.locator("[data-model-footer]:visible")
    expect(footer).to_have_count(1)
    expect(footer.locator("[data-program]")).to_have_count(1)
    expect(footer).to_contain_text("Claudexor entries supplied by Codex 2.0.0")
    expect(footer.get_by_role("button", name="Update", exact=True)).to_be_enabled()
    expect(page.locator('[role="listbox"]:visible [data-model-value="openai::choice-api"]')).to_be_visible()
    assert world["catalogs"] == reads, "the chooser uses the catalog already read by the circle editor"
    assert len(world["inventories"]) == inventories + 1
    field.press("Escape")
    with page.expect_response("**/api/claudexor/maintenance/harnesses?*"):
        field.press("ArrowDown")
    expect(footer.get_by_role("button", name="Update", exact=True)).to_be_enabled()
    assert world["catalogs"] == reads, world["catalog_requests"][reads:]
    assert len(world["inventories"]) == inventories + 2
    pure = copy.deepcopy(ui["fixture"]["catalog"])
    pure["items"] = [item for item in pure["items"] if item.get("source_id") == "opaque-source"]
    publish = "detail => document.dispatchEvent(new CustomEvent('settings-model-catalog:updated', {detail}))"
    page.evaluate(publish, pure)
    expect(footer).to_contain_text("List from Codex 2.0.0")
    expect(footer).not_to_contain_text("entries supplied by")
    assert len(world["inventories"]) == inventories + 2, "unchanged producers do not cause another read"
    assert field.input_value() == "choice"
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    roles.capture(page, f"program-circle-{account_count}-{width}")
    page.evaluate(publish, {"items": [{"id": "choice-api", "value": "openai::choice-api"}],
                            "model_sources": [], "account_catalogs": []})
    expect(footer).to_have_count(0)
    assert field.input_value() == "choice"
    field.press("Escape")
    page.locator('[data-circles-act="new"]').click()
    new_field = page.locator('[data-circles-draft="new:model"]')
    new_field.fill("choice")
    expect(footer).to_have_count(0)
    assert general_catalog_reads(world) == general_reads, world["catalog_requests"][reads:]
    assert len(world["inventories"]) == inventories + 2
    assert not world["posts"]


@pytest.mark.parametrize("width", [1440, 390])
def test_circle_update_keeps_the_open_field_and_receives_the_refreshed_list(touch_role_ui, width):
    from playwright.sync_api import expect

    ui, page = touch_role_ui, touch_role_ui["page"]
    world = vendor_scene(ui, 44)
    page.set_viewport_size({"width": width, "height": 950})
    field = open_circle_editor(ui)
    field.fill("choice")
    field.evaluate("e => { window.circleModelField = e; e.setSelectionRange(2, 3); }")
    footer = page.locator("[data-model-footer]:visible")
    action = footer.get_by_role("button", name="Update", exact=True)
    expect(action).to_be_enabled()
    if width == 390:
        action.tap()
    else:
        field.press("Tab")
        expect(action).to_be_focused()
        action.press("Enter")
    expect(footer).to_contain_text("Updating…")
    expect(field).to_have_attribute("aria-expanded", "true")
    before = general_catalog_reads(world)
    world.update(version="3.0.0", new=True)
    world["op"].update(state="succeeded", phase="settled", outcome="changed", mutation="applied", termination="confirmed",
        after={"path": "/vendor/codex", "version": "3.0.0"}, finishedAt="2026-11-10T12:00:00Z", progress=["Vendor finished."])
    world["catalogs_refresh"]()
    page.evaluate("async () => (await import('/static/modules/claudexor_status_store.js')).claudexorStatus.refresh()")
    expect(footer).to_contain_text("Updated from 2.0.0")
    expect(footer).to_contain_text("Claudexor entries supplied by Codex 3.0.0")
    expect(page.locator('[role="listbox"]:visible [data-model-value="claudexor::opaque-source=choice-new"]')).to_be_visible()
    expect(field).to_have_attribute("aria-expanded", "true")
    assert field.evaluate("e => e === window.circleModelField")
    assert field.input_value() == "choice"
    assert general_catalog_reads(world) == before + 1, world["catalog_requests"]
    assert world["posts"] == [{"harness": "codex", "target": {"kind": "latest"}}]
    expect(footer.locator("button")).to_have_count(0)
    roles.capture(page, f"program-circle-updated-44-{width}")


@pytest.mark.parametrize("width", [1440, 390])
@pytest.mark.parametrize("account_count", [1, 18])
def test_vendor_footer_every_settings_consumer_at_scale(role_ui, width, account_count):
    from playwright.sync_api import expect

    ui, page = role_ui, role_ui["page"]
    world = vendor_scene(ui, account_count)
    page.set_viewport_size({"width": width, "height": 950})
    roles.open_models(ui)
    expect(page.locator("[data-model-footer]:visible")).to_have_count(0)
    roles.capture(page, f"program-editors-closed-{account_count}-{width}")
    for consumer in ("Models", "Light", "Safety", "Vision", "Consciousness", "Fallback"):
        field = vendor_consumer(ui, consumer)
        field.fill("choice")
        footer = page.locator("[data-model-footer]:visible")
        expect(footer).to_have_count(1)
        expect(footer).to_contain_text("List from Codex 2.0.0")
        expect(footer.locator("[data-program]")).to_have_count(1)
        expect(footer.get_by_role("button", name="Update", exact=True)).to_be_enabled()
        expect(page.locator('[role="listbox"]:visible [role="option"]')).to_have_count(1)
        assert footer.locator('[role="option"]').count() == 0
        field.fill("owner-new-id")
        expect(field.locator("..").locator("[data-model-custom]")).to_be_visible()
        field.press("Escape")
        expect(page.locator("[data-model-footer]:visible")).to_have_count(0)
        assert field.input_value() == "owner-new-id"
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    assert not world['posts']


@pytest.mark.parametrize("width", [1440, 390])
@pytest.mark.parametrize("account_count", [1, 18])
def test_vendor_footer_agent_consumers_at_scale(role_ui, width, account_count):
    from playwright.sync_api import expect
    ui, page = role_ui, role_ui['page']
    world = vendor_scene(ui, account_count)
    page.set_viewport_size({'width': width, 'height': 950})
    roles.open_agents(ui)
    for consumer in ("Reviewer", "Agent"):
        field = vendor_consumer(ui, consumer)
        field.fill("choice")
        footer = page.locator("[data-model-footer]:visible")
        expect(footer).to_have_count(1)
        expect(footer).to_contain_text("List from Codex 2.0.0")
        expect(footer.locator("[data-program]")).to_have_count(1)
        expect(footer.get_by_role("button", name="Update", exact=True)).to_be_enabled()
        expect(page.locator('[role="listbox"]:visible [role="option"]')).to_have_count(1)
        assert footer.locator('[role="option"]').count() == 0
        field.fill("owner-new-id")
        expect(field.locator("..").locator("[data-model-custom]")).to_be_visible()
        field.press("Escape")
        expect(page.locator("[data-model-footer]:visible")).to_have_count(0)
        assert field.input_value() == "owner-new-id"
    current = roles.open_agent(ui, 'current', fresh=False).locator('[data-subagent-field="model"]')
    current.fill("choice")
    footer = page.locator("[data-model-footer]:visible")
    expect(footer).to_have_text("List from Claude Code 2.0.0")
    expect(footer.locator("button")).to_have_count(0)
    current.press("Escape")
    api = roles.open_agent(ui, 'api', fresh=False).locator('[data-subagent-field="model"]')
    api.fill("choice")
    expect(page.locator("[data-model-footer]:visible")).to_have_count(0)
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    api.press("Escape")
    vendor_consumer(ui, "Agent").fill("choice")
    roles.capture(page, f"program-editors-open-{account_count}-{width}")
    before = len(world["inventories"])
    page.evaluate("async () => (await import('/static/modules/claudexor_status_store.js')).claudexorStatus.refresh()")
    assert len(world["inventories"]) == before, "passive status never reads newest or inventory"
    assert not world["posts"]


@pytest.mark.parametrize("interaction,consumer", [("mouse", "Agent"), ("keyboard", "Models"), ("touch", "Reviewer")])
def test_vendor_update_keeps_chooser_open_and_refreshes_its_real_catalog(touch_role_ui, interaction, consumer):
    from playwright.sync_api import expect

    ui, page = touch_role_ui, touch_role_ui["page"]
    world = vendor_scene(ui, 18)
    page.set_viewport_size({"width": 390 if interaction == "touch" else 1440, "height": 950})
    roles.open_models(ui)
    field = vendor_consumer(ui, consumer)
    field.fill("choice")
    field.evaluate("e => {window.ownerModelField = e; e.setSelectionRange(2, 3);}")
    footer = page.locator("[data-model-footer]:visible")
    action = footer.get_by_role("button", name="Update", exact=True)
    expect(action).to_be_enabled()
    if interaction == "keyboard":
        field.press("Tab")
        expect(action).to_be_focused()
        action.press("Shift+Tab")
        expect(field).to_be_focused()
        field.press("Tab")
        action.press("Enter")
    elif interaction == "touch":
        action.tap()
    else:
        action.click()
    expect(footer).to_contain_text("Updating…")
    expect(field).to_have_attribute("aria-expanded", "true")
    assert field.input_value() == "choice"
    assert world["posts"] == [{"harness": "codex", "target": {"kind": "latest"}}]
    before = world["catalogs"]
    world.update(version="3.0.0", new=True)
    world["op"].update(state="succeeded", phase="settled", outcome="changed", mutation="applied", termination="confirmed",
        after={"path": "/vendor/codex", "version": "3.0.0"}, finishedAt="2026-11-10T12:00:00Z", progress=["Vendor finished."],
        exitCode=7, limitations=["Installer exited 7; replacement verified."])
    world["catalogs_refresh"]()
    page.evaluate("async () => (await import('/static/modules/claudexor_status_store.js')).claudexorStatus.refresh()")
    expect(footer).to_contain_text("Updated from 2.0.0")
    expect(footer).to_contain_text("List from Codex 3.0.0")
    expect(footer).to_contain_text("Installer exited 7; replacement verified.")
    expect(page.locator('[role="listbox"]:visible [data-model-value="choice-new"]')).to_be_visible()
    expect(field).to_have_attribute("aria-expanded", "true")
    assert field.evaluate("e => e === window.ownerModelField")
    assert field.input_value() == "choice"
    assert world["catalogs"] > before
    expect(footer.locator("button")).to_have_count(0)
    roles.capture(page, f"program-chooser-{interaction}-settled")
    field.press("Tab")
    expect(page.locator("[data-model-footer]:visible")).to_have_count(0)


@pytest.mark.parametrize("width", [1440, 390])
@pytest.mark.parametrize("account_count", [1, 18])
def test_vendor_footer_in_wizard_models_reviewers_and_place_sheet(role_ui, width, account_count):
    import json
    from pathlib import Path
    from playwright.sync_api import expect

    ui, page = role_ui, role_ui["page"]
    world = vendor_scene(ui, account_count)
    ui["bootstrap"]["initialState"].update(mainModel="claudexor::opaque-source=choice-old", openaiKey="fixture")
    ui["fixture"]["preview"]["model_settings"]["OUROBOROS_MODEL"] = "claudexor::opaque-source=choice-old"
    ui["fixture"]["preview"]["available_subagents"] = ui["settings"]["OUROBOROS_SUBAGENTS"]
    page.set_viewport_size({"width": width, "height": 950})
    page.goto(ui["url"] + "/onboarding")
    page.click("#next-btn")
    field = page.locator('[data-model-role="main"] [data-model-role-model]')
    page.locator('[data-model-role="main"] [data-model-role-source]').select_option("subscription:opaque-source")
    field.fill("choice")
    footer = page.locator("[data-model-footer]:visible")
    expect(footer).to_have_count(1)
    expect(footer).to_contain_text("List from Codex 2.0.0")
    expect(footer.locator("[data-program]")).to_have_count(1)
    expect(footer.get_by_role("button", name="Update", exact=True)).to_be_enabled()
    roles.capture(page, f"program-wizard-model-{account_count}-{width}")
    field.press("Escape")
    for slot in ("light", "safety", "vision", "consciousness", "fallback"):
        field = vendor_role_field(page, slot)
        field.fill("choice")
        expect(footer).to_have_count(1)
        expect(footer.locator("[data-program]")).to_have_count(1)
        expect(footer).to_contain_text("List from Codex 2.0.0")
        expect(footer.get_by_role("button", name="Update", exact=True)).to_be_enabled()
        field.press("Escape")
    page.locator('[data-collapse="subagents"] > summary').click()
    reviewer = page.locator('[data-subagent-row]').first
    expect(reviewer.locator('[data-subagent-field="review_eligible"]')).to_be_checked()
    field = reviewer.locator('[data-subagent-field="model"]')
    field.fill("choice")
    expect(footer).to_have_count(1)
    expect(footer).to_contain_text("List from Codex 2.0.0")
    expect(footer.get_by_role("button", name="Update", exact=True)).to_be_enabled()
    roles.capture(page, f"program-wizard-reviewer-{account_count}-{width}")
    field.press("Escape")
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")

    # Open the production Place sheet through its ordinary entry point (as Circles does), then More / Model.
    # This general chooser can contain several vendors. Each gets one closing
    # row regardless of account count, and every action must be keyboard reachable.
    ui["fixture"]["catalog"]["model_sources"].append({"id": "other-source", "label": "Claude Code", "credentialHarness": "claude"})
    ui["fixture"]["catalog"]["items"].append({"id": "choice-local", "value": "choice-local", "use_local": True})
    ui["fixture"]["catalog"]["items"] += [{"id": "choice-other", "value": "claudexor::other-source=choice-other",
        "source_id": "other-source", "source": "Claudexor", "credential_profile_id": f"other-{index}",
        "sourceProgram": {"path": "/vendor/claude", "version": "2.0.0"}} for index in range(account_count)]
    ui["fixture"]["catalog"]["account_catalogs"].append({"source": "other-source", "accounts": [{"catalog": {
        "source": "other-source", "credentialProfileId": f"other-{index}", "models": [{"id": "choice-other"}],
        "sourceProgram": {"path": "/vendor/claude", "version": "2.0.0"}}} for index in range(account_count)]})
    world["other_inventory"].update(updateAvailable="yes", newest={"status": "available", "version": "3.0.0"})
    ui["errors"].watch(page)
    page.goto(ui["url"] + "/#settings")
    data = json.loads((Path(__file__).resolve().parents[1] / "web/tests/fixtures/consciousness_contract.json").read_text(encoding="utf-8"))
    page.evaluate("""async raw => {
        const [{openPlaceSheet}, {normalizePlace}] = await Promise.all([
            import('/static/modules/place_sheet.js'), import('/static/modules/consciousness_api.js')]);
        openPlaceSheet(normalizePlace(raw), {circles: []});
    }""", data["places"]["places"][0])
    page.locator(".place-more > summary").click()
    field = page.locator('[data-place-draft="model"]')
    expect(field).to_have_attribute("aria-describedby", "place-model-help")
    expect(page.locator("#place-model-help")).to_be_visible()
    field.fill("choice")
    expect(footer).to_have_count(1)
    expect(footer).to_contain_text("Claudexor entries supplied by Codex 2.0.0")
    expect(footer.locator("[data-program]")).to_have_count(2)
    expect(footer).to_contain_text("Claudexor entries supplied by Claude Code 2.0.0")
    expect(footer).not_to_contain_text("List from")
    expect(page.locator('[role="listbox"]:visible [data-model-value="openai::choice-api"]')).to_be_visible()
    expect(page.locator('[role="listbox"]:visible [data-model-value="choice-local"]')).to_have_count(0)
    action = footer.locator('[data-program="codex"]').get_by_role("button", name="Update", exact=True)
    other_action = footer.locator('[data-program="claude"]').get_by_role("button", name="Update", exact=True)
    expect(action).to_be_enabled()
    field.press("Tab")
    expect(action).to_be_focused()
    action.press("Tab")
    expect(other_action).to_be_focused()
    other_action.press("Shift+Tab")
    expect(action).to_be_focused()
    action.press("Shift+Tab")
    expect(field).to_be_focused()
    expect(field).to_have_attribute("aria-expanded", "true")
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    roles.capture(page, f"program-place-{account_count}-{width}")
    field.press("Escape")
    expect(footer).to_have_count(0)


@pytest.mark.parametrize("width", [1440, 390])
@pytest.mark.parametrize("account_count", [1, 18])
def test_models_role_names_only_program_supplied_entries_in_a_mixed_source(role_ui, width, account_count):
    from playwright.sync_api import expect

    ui, page = role_ui, role_ui["page"]
    vendor_scene(ui, account_count)
    # A source can also supply hints with no program provenance; same source is
    # not proof that every suggestion came from its executable.
    ui["fixture"]["catalog"]["items"].append({"id": "choice-hint", "value": "claudexor::opaque-source=choice-hint",
        "source_id": "opaque-source", "credential_profile_id": "personal", "sourceProgram": None})
    page.set_viewport_size({"width": width, "height": 950})
    roles.open_models(ui)
    field = vendor_consumer(ui, "Models")
    field.fill("choice")
    footer = page.locator("[data-model-footer]:visible")
    expect(footer).to_have_count(1)
    expect(footer.locator("[data-program]")).to_have_count(1)
    expect(footer).to_contain_text("Claudexor entries supplied by Codex 2.0.0")
    expect(footer).not_to_contain_text("List from")
    expect(page.locator('[role="listbox"]:visible [role="option"]')).to_have_count(2)
    expect(footer.get_by_role("button", name="Update", exact=True)).to_be_enabled()
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    roles.capture(page, f"program-models-mixed-{account_count}-{width}")


@pytest.mark.parametrize("width", [1440, 390])
@pytest.mark.parametrize("consumer", ["Agent", "Models"])
def test_outside_update_then_chooser_open_renews_list_once_at_44_accounts(role_ui, width, consumer):
    from playwright.sync_api import expect
    ui, page = role_ui, role_ui["page"]
    world = vendor_scene(ui, 44)
    page.set_viewport_size({"width": width, "height": 950})
    roles.open_models(ui)
    field = vendor_consumer(ui, consumer)
    field.fill("choice")
    footer = page.locator("[data-model-footer]:visible")
    expect(footer).to_contain_text("List from Codex 2.0.0")
    expect(footer.get_by_role("button", name="Update", exact=True)).to_be_enabled()
    before = world["catalogs"]
    field.press("Escape")
    with page.expect_response("**/api/claudexor/maintenance/harnesses?*"):
        field.press("ArrowDown")
    expect(footer.get_by_role("button", name="Update", exact=True)).to_be_enabled()
    expect(footer).to_contain_text("List from Codex 2.0.0")
    assert world["catalogs"] == before
    field.press("Escape")
    world.update(version="3.0.0", new=True)  # Terminal/vendor updater, no receipt or button.
    field.press("ArrowDown")
    expect(footer).to_contain_text("List from Codex 3.0.0")
    expect(page.locator('[role="listbox"]:visible [role="option"]').filter(has_text="choice-new")).to_be_visible()
    assert field.input_value() == "choice"
    after = world["catalogs"]
    assert after > before and not world["posts"]
    field.press("Escape"); field.press("ArrowDown")
    expect(footer).to_contain_text("List from Codex 3.0.0")
    assert world["catalogs"] == after


@pytest.mark.parametrize("width", [1440, 390])
def test_failed_catalog_refresh_recovers_on_next_open_at_44_accounts(role_ui, width):
    from playwright.sync_api import expect
    ui, page = role_ui, role_ui["page"]
    world = vendor_scene(ui, 44)
    page.set_viewport_size({"width": width, "height": 950})
    roles.open_models(ui)
    field = vendor_consumer(ui, "Models")
    field.fill("choice")
    footer = page.locator("[data-model-footer]:visible")
    expect(footer).to_contain_text("List from Codex 2.0.0")
    field.press("Escape")
    failures = []
    def refuse(route):
        failures.append(route.request.url)
        route.fulfill(status=503, json={"error": "Catalog temporarily unavailable"})
    page.route("**/api/model-catalog?*", refuse)
    world.update(version="3.0.0", new=True)
    field.press("ArrowDown")
    expect(page.locator('[data-model-role="main"] [data-model-role-status]')).to_contain_text("temporarily unavailable")
    expect(footer).to_contain_text("List from Codex 2.0.0")
    count = len(failures)
    page.wait_for_timeout(250)
    assert len(failures) == count and count > 0, "No retry loop while opened"
    page.unroute("**/api/model-catalog?*", refuse)
    field.press("Escape"); field.press("ArrowDown")
    expect(footer).to_contain_text("List from Codex 3.0.0")
    expect(page.locator('[role="listbox"]:visible [role="option"]').filter(has_text="choice-new")).to_be_visible()
    after = world["catalogs"]
    field.press("Escape"); field.press("ArrowDown")
    expect(footer).to_contain_text("List from Codex 3.0.0")
    assert world["catalogs"] == after and not world["posts"]
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")


@pytest.mark.parametrize("width", [1440, 390])
def test_two_chooser_openings_share_producer_refresh_at_44_accounts(role_ui, width):
    from playwright.sync_api import expect
    from urllib.parse import parse_qs, urlparse
    ui, page = role_ui, role_ui["page"]
    world = vendor_scene(ui, 44)
    page.set_viewport_size({"width": width, "height": 950})
    roles.open_models(ui)
    first = vendor_consumer(ui, "Agent")
    first.fill("choice")
    footer = page.locator("[data-model-footer]:visible")
    expect(footer).to_contain_text("List from Codex 2.0.0")
    first.press("Escape")
    pending, requests = [], []
    def hold(route):
        if parse_qs(urlparse(route.request.url).query).get("harness") == ["codex"]:
            pending.append(route)
        else:
            route.fallback()
    pattern = "**/api/claudexor/maintenance/harnesses?*"
    page.route(pattern, hold)
    page.on("request", lambda request: requests.append(request.url))
    first.press("ArrowDown")
    expect(footer).to_contain_text("List from Codex 2.0.0")
    second = roles.open_agent(ui, "worker2", fresh=False).locator('[data-subagent-field="model"]')
    second.fill("choice")  # The first opening still awaits the same program read.
    expect(footer).to_contain_text("List from Codex 2.0.0")
    page.wait_for_timeout(100)
    assert len(pending) == 1
    # Page navigation reads its own catalogue; count the producer refresh separately.
    requests.clear()
    world.update(version="3.0.0", new=True)
    page.unroute(pattern, hold)
    pending[0].fallback()
    expect(footer).to_contain_text("List from Codex 3.0.0")
    expect(page.locator('[role="listbox"]:visible [role="option"]').filter(has_text="choice-new")).to_be_visible()
    full_catalogs = [url for url in requests if urlparse(url).path == "/api/model-catalog"
                     and "source_id" not in parse_qs(urlparse(url).query)]
    native_catalogs = [url for url in requests if urlparse(url).path == "/api/claudexor/status"
                       and parse_qs(urlparse(url).query).get("include") == ["models"]]
    assert len(full_catalogs) == len(native_catalogs) == 1
    assert not world["posts"] and page.evaluate("document.documentElement.scrollWidth <= innerWidth")
