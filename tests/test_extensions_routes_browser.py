"""Settings and Extensions in one app, at a desk width and a phone width (docs/DESIGN.md §10 "A section with pages",
"Settings"; web/modules/extensions.js, settings.js, section_levels.js).

The routes the owner takes between the two sections, on the real modules: Settings' one door to Extensions and the
names it answers to (Skills, Tools, MCP servers); an old Settings address (a search result on MCP servers) that opens
its page in Extensions at its row, with Back to the Settings page it was pressed on; Back inside Extensions to the
control that opened a level; a page asked for from Main with its own return; an
explicit `ouro:open-skills` request over the page this device last left the section on; a server's revealed
credential put away when Extensions is left; and a skill form's read that lands after the owner left, which draws
nothing until the form is shown again. Settings' own landing stands as before the shared levels (Models the first time).

Settings' levels, Escape, browser Back and field commits: tests/test_settings_shell_browser.py; Settings' home and
search: tests/test_settings_home_browser.py; a covered or hidden form's late read in isolation:
web/tests/settings_skill_list.test.js and web/tests/settings_tools_models.test.js; the levels themselves:
web/tests/extensions_section.test.js."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone

import pytest

from tests import test_subscription_setup_browser as setup_browser
from tests._room_browser_shared import _nav
from tests.settings_browser_helpers import HOME, open_settings, search_settings, settings_page, top_path
from tests.settings_section_helpers import header_back, lead_name

subscription_ui = setup_browser.subscription_ui
pytestmark = [pytest.mark.ui_browser]

EXTENSIONS = "#page-skills"
LAST_PAGE = "() => localStorage.getItem('ouroboros.extensions.last-page')"
SECRET = "Bearer synthetic-extensions-route-0001"
SKILLS = {"skills": [dict(name="gate_skill", description="An installed skill with a form.", enabled=True, loaded=True,
                          live_loaded=True, type="extension", review_status="pass", review_gate="pass", source="self_authored")],
          "live": {"settings_sections": [dict(skill="gate_skill", section_id="config", key="gate_skill:config",
              title="Gate skill settings", render={"components": [{"type": "form", "id": "form", "route": "settings",
                  "submit_label": "Save", "fields": [{"name": "address", "label": "Address", "type": "text"}]}]})]}}
SERVER = dict(id="docs", name="Docs", enabled=True, transport="streamable_http", url="https://docs.example.test/mcp",
              auth_header="Authorization", auth_token="Bearer ...", auth_configured=True)
SIZES = [(1360, 900), (390, 844)]


def _page(ui, width, height):
    """A page of the app with one MCP server whose credential is saved, one installed skill with a form, and the
    server's saved credential answered by Show."""
    ui["settings"]["MCP_ENABLED"] = True
    ui["settings"]["MCP_SERVERS"] = [dict(SERVER)]
    page = ui["new_page"](viewport={"width": width, "height": height}, has_touch=width < 720)
    status = {"sdk_available": True, "enabled": True, "servers": [{"id": "docs", "enabled": True, "tool_count": 2,
              "listed_at": datetime.now(timezone.utc).isoformat(), "tools": []}]}
    page.route("**/api/mcp/status", lambda route: route.fulfill(json=status))
    page.route("**/api/extensions", lambda route: route.fulfill(json=SKILLS))
    reveals = []

    def reveal(route):
        reveals.append(route.request.post_data_json)
        route.fulfill(json={"value": SECRET})

    page.route("**/api/settings/secret", reveal)
    return page, reveals


@pytest.mark.parametrize("width,height", SIZES, ids=["desk", "phone"])
def test_settings_and_extensions_lead_to_each_other_and_back(subscription_ui, width, height):
    ui = subscription_ui
    page, _reveals = _page(ui, width, height)
    wide = width >= 720
    page.goto(ui["url"] + "/#settings")
    page.locator(HOME).wait_for(state="visible")
    # Settings' own landing, as before the levels it shares with Extensions: Models the first time beside the list.
    if wide:
        settings_page(page, "models").wait_for(state="visible")
    assert top_path(page) == ("models" if wide else "")

    # One door to Extensions, found by the names the Skills and Tools rows had; a row of a page that moved says where
    # it lives now.
    for words in ("Skills", "Tools", "MCP servers"):
        assert "home:skills" in search_settings(page, words), words
    assert "MCP_ENABLED" in search_settings(page, "Use MCP servers")
    where = page.locator(f'{HOME} [data-settings-result="MCP_ENABLED"] [data-settings-answer]').inner_text().strip()
    assert where == "Extensions › MCP servers"
    search_settings(page, "")
    door = page.locator(f'{HOME} [data-settings-open="page:skills"]')
    assert door.locator(".settings-door-title").inner_text().strip() == "Extensions"

    # The door: Extensions with Back to Settings. Two panes: beside the page last opened (Skills the first time);
    # one column: the list alone, Back where ☰ stood.
    before = top_path(page)
    door.click()
    page.locator(EXTENSIONS).wait_for(state="visible")
    if wide:
        page.locator(f'{EXTENSIONS} [data-extensions-open="installed"][aria-current="page"]').wait_for()
        back = header_back(page)
    else:
        page.locator(f"{EXTENSIONS} [data-extensions-rail]").wait_for(state="visible")
        back = header_back(page)
    assert back.is_visible() and back.get_attribute("aria-label") == "Back to Settings"
    setup_browser.capture(page, f"extensions-from-settings-{width}x{height}")
    back.click()
    page.locator(HOME).wait_for(state="visible")
    assert top_path(page) == before, "Back returns to the Settings page the door stood on"

    # An old address, here a search result on MCP servers: its page in Extensions at its row, Back to Settings.
    before = top_path(page)
    search_settings(page, "Use MCP servers")
    page.locator(f'{HOME} [data-settings-result="MCP_ENABLED"]').click()
    servers = settings_page(page, "tools")
    servers.locator('[data-setting-row="MCP_ENABLED"]').wait_for(state="visible")
    page.wait_for_function("() => document.activeElement?.closest('[data-setting-row]')?.dataset.settingRow === 'MCP_ENABLED'")
    assert lead_name(page) == "MCP servers"
    assert header_back(page).get_attribute("aria-label") == "Back to Settings"
    header_back(page).click()
    page.locator(HOME).wait_for(state="visible")
    assert top_path(page) == before

    # The two old Tools links diverge at their rows.
    for key, path in [("OUROBOROS_TOOL_TIMEOUT_SEC", "computer/advanced"), ("MCP_TOOL_TIMEOUT_SEC", "tools")]:
        assert page.evaluate("row => window.__ouroSettings.open('tools', {row})", key)
        field = settings_page(page, path).locator(f'[data-setting-row="{key}"]')
        field.wait_for(state="visible")
        page.wait_for_function("key => document.activeElement?.closest('[data-setting-row]')?.dataset.settingRow === key", arg=key)
        header_back(page).click()

    # Inside Extensions: a server's head opens its page over MCP servers, and Back returns to that head.
    _nav(page, "skills")
    row = page.locator(f'{EXTENSIONS} [data-extensions-open="tools"]')
    if not row.is_visible():  # one column: a page is open over the list
        header_back(page).click()
    row.click()
    head = settings_page(page, "tools").locator('[data-entry-id="docs"] [data-entry-open]')
    head.focus()
    page.keyboard.press("Enter")
    server = settings_page(page, "tools/docs")
    server.locator("[data-mcp-card]").wait_for(state="visible")
    assert header_back(page).get_attribute("aria-label") == "Back to MCP servers"
    assert page.locator(f'{EXTENSIONS} [data-settings-page^="tools/"]').count() == 1
    header_back(page).click()
    server.wait_for(state="detached")
    page.wait_for_function("() => document.activeElement?.closest('[data-entry-id]')?.dataset.entryId === 'docs'")
    assert page.evaluate(LAST_PAGE) == "tools"

    # From Main (a place's Connect does the same): a fresh jump returns to Main.
    _nav(page, "chat")
    assert page.evaluate("() => window.__ouroSettings.open('tools/docs')") is not False
    server = settings_page(page, "tools/docs")
    server.locator("[data-mcp-card]").wait_for(state="visible")
    assert header_back(page).get_attribute("aria-label") == "Back to Main"
    assert page.locator(f"{EXTENSIONS} [data-shell-back]:visible").count() == 1

    # An explicit request (the dictation chooser, a place's connector) opens its page over the one this device last
    # left the section on, and that page is remembered.
    _nav(page, "chat")
    page.evaluate("() => window.dispatchEvent(new CustomEvent('ouro:open-skills', { detail: { skill: 'gate_skill' } }))")
    page.locator('.skills-card[data-skill="gate_skill"]').wait_for(state="visible")
    assert page.evaluate(LAST_PAGE) == "installed"
    if wide:
        assert page.locator(f'{EXTENSIONS} [data-extensions-open="installed"]').get_attribute("aria-current") == "page"
    page.close()


def test_leaving_extensions_for_settings_closes_the_server_import(subscription_ui):
    page, _ = _page(subscription_ui, 1360, 900)
    page.goto(subscription_ui["url"] + "/#skills")
    servers = open_settings(page, "tools")
    servers.locator("[data-settings-fold] > summary").click()
    servers.locator('[data-tools-action="tools:import"]').click()
    dialog = page.get_by_role("dialog", name="Import mcp.json")
    dialog.locator("textarea").fill('{"mcpServers": {}}')
    open_settings(page, "models")
    dialog.wait_for(state="detached")
    page.close()


@pytest.mark.parametrize("answer", ["found", "absent", "failed", "left"])
def test_direct_skill_request_uses_a_fresh_listing_once(subscription_ui, answer):
    from playwright.sync_api import expect

    page, _ = _page(subscription_ui, 1360, 900)
    page.goto(subscription_ui["url"] + "/#skills")
    page.locator('.skills-card[data-skill="gate_skill"]').wait_for()
    page.wait_for_load_state("networkidle")
    search = page.locator("#skills-installed-search")
    search.fill("unrelated")
    fresh = {**SKILLS, "skills": [*SKILLS["skills"], dict(SKILLS["skills"][0], name="new_skill")]}
    card = page.locator('.skills-card[data-skill="new_skill"]')
    held = []
    route_pattern = "**/api/extensions"

    def hold(route):
        held.append(route)
        page.evaluate("count => window.heldSkillReads = count", len(held))

    page.route(route_pattern, hold)
    page.locator("#skills-refresh").click()
    page.wait_for_function("() => window.heldSkillReads === 1")
    page.evaluate("() => window.dispatchEvent(new CustomEvent('ouro:open-skills', {detail: {skill: 'new_skill'}}))")
    held[0].fulfill(json=SKILLS)  # Refresh started before the direct request.
    page.wait_for_function("() => window.heldSkillReads === 2")
    if answer == "left":
        _nav(page, "chat")
    held[1].fulfill(json=SKILLS if answer == "absent" else fresh, status=503 if answer == "failed" else 200)
    page.wait_for_load_state("networkidle")
    assert len(held) == 2
    if answer == "found":
        expect(card).to_be_in_viewport()
        expect(search).to_have_value("new_skill")
    else:
        # A later ordinary listing now includes the card. It must respect the filter the owner left.
        page.route(route_pattern, lambda route: route.fulfill(json=fresh))
        if answer == "left":
            _nav(page, "skills")
        else:
            page.locator("#skills-refresh").click()
        expect(card).to_have_count(1)
        expect(card).to_be_hidden()
        expect(search).to_have_value("unrelated")
    page.close()


@pytest.mark.parametrize("start_width,end_width", [(390, 1087), (1440, 390)], ids=["phone-to-desk", "desk-to-phone"])
@pytest.mark.parametrize("method", ["button", "escape", "browser-back"])
def test_settings_return_survives_extensions_layout_change(subscription_ui, start_width, end_width, method):
    page, _reveals = _page(subscription_ui, start_width, 844)
    page.goto(subscription_ui["url"] + "/#settings")
    page.locator(HOME).wait_for(state="visible")
    if start_width == 1440:
        settings_page(page, "models").wait_for(state="visible")
    before = top_path(page)
    page.locator(f'{HOME} [data-settings-open="page:skills"]').click()
    page.locator(EXTENSIONS).wait_for(state="visible")
    page.set_viewport_size({"width": end_width, "height": 844})
    # Wait for the actual responsive landing, not a fixed delay or just the viewport dimensions.
    page.locator(f'{EXTENSIONS}[data-section-view="page"]').wait_for(state="visible")
    page.locator(f"{EXTENSIONS} [data-extensions-rail]").wait_for(state="visible" if end_width == 1087 else "hidden")
    back = page.locator(f"{EXTENSIONS} > .app-page-header [data-shell-back]")
    back.wait_for(state="visible")
    assert back.get_attribute("aria-label") == "Back to Settings"
    assert page.locator(f"{EXTENSIONS} [data-shell-back]:visible").count() == 1
    if method == "button":
        back.click()
    elif method == "escape":
        page.keyboard.press("Escape")
    else:
        page.go_back()
    page.locator("#page-settings").wait_for(state="visible")
    # Widening the empty Settings list opens its default page, as on the first wide landing.
    assert top_path(page) == (before or "models"), "resize preserves the Settings page the door stood on"
    assert page.url.startswith(subscription_ui["url"]), "Back must not leave the app for about:blank"
    assert page.evaluate("() => window.__ouroLayers.returnPage() === null")
    page.close()


@pytest.mark.parametrize("width,height", SIZES, ids=["desk", "phone"])
@pytest.mark.parametrize("method", ["button", "escape", "browser-back"])
def test_voice_skill_return_restores_the_pressed_door(subscription_ui, monkeypatch, width, height, method):
    from tests import test_live_conversation_browser as calls

    page, _ = calls.open_call_ui(subscription_ui, monkeypatch, width=width, scheme="dark")
    voice = open_settings(page, "voice")
    door = voice.locator('[data-setting-row="voice:skill:live_conversation"]')
    door.wait_for(state="visible")
    # No preparatory focus(): pointer presses in WebKit do not focus buttons.
    door.click()
    page.locator(f'{EXTENSIONS} [data-extension-settings-form][data-skill="live_conversation"]').wait_for(state="visible")
    assert header_back(page).get_attribute("aria-label") == "Back to Settings › Voice and calls"
    if method == "button":
        header_back(page).click()
    elif method == "escape":
        page.keyboard.press("Escape")
    else:
        page.go_back()
    voice.wait_for(state="visible")
    page.wait_for_function("() => document.activeElement?.dataset.settingRow === 'voice:skill:live_conversation'")
    page.close()


@pytest.mark.parametrize("start_width,end_width", [(390, 1087), (1440, 390)], ids=["phone-to-desk", "desk-to-phone"])
@pytest.mark.parametrize("method", ["button", "escape", "browser-back"])
def test_accounts_return_to_skill_survives_layout_change(subscription_ui, start_width, end_width, method):
    page, _reveals = _page(subscription_ui, start_width, 844)
    listing = {"skills": [dict(SKILLS["skills"][0],
        grants={"requested_keys": ["WEATHER_TOKEN"], "granted_keys": ["WEATHER_TOKEN"], "ready": True},
        secret_config={"fields": [{"key": "WEATHER_TOKEN", "state": "missing"}]})], "live": SKILLS["live"]}
    page.route("**/api/extensions", lambda route: route.fulfill(json=listing))
    page.goto(subscription_ui["url"] + "/#skills")
    page.locator(EXTENSIONS).wait_for(state="visible")
    page.evaluate("() => window.dispatchEvent(new CustomEvent('ouro:open-skills', {detail: {skill: 'gate_skill'}}))")
    card = page.locator('.skills-card[data-skill="gate_skill"]')
    card.wait_for(state="visible")
    opener = card.locator('[data-skill-action="open_secrets"]')
    original = opener.element_handle()
    return_id = "() => window.__ouroLayers.returnPage()?.id ?? null"
    prior_return = page.evaluate(return_id)
    opener.click()
    settings_page(page, "accounts").wait_for(state="visible")
    assert page.evaluate(return_id) not in (None, prior_return), "Accounts owns a new return"
    # Separate the return from the refresh it starts. A real changed description must paint without
    # replacing the unchanged action that just regained focus.
    renewed = {**listing, "skills": [{**listing["skills"][0], "description": "Updated after returning to Skills."}]}
    held = []
    hold = True

    def read_listing(route):
        if hold:
            held.append(route)
        else:
            route.fulfill(json=renewed)

    page.route("**/api/extensions", read_listing)
    assert header_back(page).get_attribute("aria-label") == "Back to Extensions › Skills"
    page.set_viewport_size({"width": end_width, "height": 844})
    page.locator(HOME).wait_for(state="visible" if end_width == 1087 else "hidden")
    assert top_path(page) == "accounts"
    assert header_back(page).get_attribute("aria-label") == "Back to Extensions › Skills"
    with page.expect_request(lambda req: req.method == "GET" and req.url.endswith("/api/extensions")):
        if method == "button":
            header_back(page).click()
        elif method == "escape":
            page.keyboard.press("Escape")
        else:
            page.go_back()
    page.locator(f"{EXTENSIONS}.active").wait_for(state="visible")
    card.wait_for(state="visible")
    page.wait_for_function("el => el === document.activeElement", arg=original)
    marker = original.get_attribute("data-return-focus")
    assert marker in {"pointer", "keyboard"} and held
    hold = False
    for route in held:
        route.fulfill(json=renewed)
    card.get_by_text(renewed["skills"][0]["description"], exact=True).wait_for()
    assert original.evaluate("el => el.isConnected && el === document.activeElement")
    assert original.get_attribute("data-return-focus") == marker
    assert page.url.startswith(subscription_ui["url"]), "Back must not leave the app for about:blank"
    assert page.evaluate(return_id) == prior_return, "Back consumes only the Accounts return"
    page.close()


@pytest.mark.parametrize("width,height", SIZES, ids=["desk", "phone"])
def test_a_revealed_secret_is_put_away_and_a_late_form_read_draws_nothing_while_hidden(subscription_ui, width, height):
    ui = subscription_ui
    page, reveals = _page(ui, width, height)
    values = {"address": "first"}
    held, hold = [], {"on": False}

    def form_values(route):
        if route.request.method != "GET":
            route.fallback()
        elif hold["on"]:
            held.append(route)
        else:
            route.fulfill(json=dict(values))

    page.route("**/api/extensions/gate_skill/settings", form_values)
    page.goto(ui["url"] + "/#settings")
    page.locator(HOME).wait_for(state="visible")

    # A server's revealed credential is put away when the owner leaves Extensions; its page stays open.
    server = open_settings(page, "tools/docs")
    card = server.locator("[data-mcp-card]")
    card.locator("[data-mcp-more] > summary").click()
    card.locator("[data-mcp-header-toggle]").first.click()
    revealed = card.locator(".settings-secret-value").first
    revealed.wait_for(state="visible")
    assert revealed.inner_text() == SECRET and reveals[-1] == {"mcp_server_id": "docs"}
    _nav(page, "chat")
    _nav(page, "skills")
    server.wait_for(state="visible")
    assert SECRET not in page.locator(EXTENSIONS).evaluate("el => el.textContent"), "a revealed secret outlived the place"

    # Local Back puts a dirty form aside. Its typed value survives and its Show state ends.
    name = card.locator('[data-mcp-field="name"]')
    name.fill("My draft")
    card.locator("[data-mcp-header-toggle]").first.click()
    revealed.wait_for(state="visible")
    header_back(page).click()
    server = open_settings(page, "tools/docs")
    assert name.input_value() == "My draft"
    assert revealed.is_hidden()

    # A skill's form: a read on its way when the owner leaves draws nothing into the hidden form; shown again, the
    # form is read and drawn.
    form = open_settings(page, "skill/gate_skill").locator("[data-extension-settings-form]")
    form.wait_for(state="visible")
    field = form.locator('[name="address"]')
    assert field.input_value() == "first"
    hold["on"] = True
    values["address"] = "second"
    with page.expect_request(lambda request: request.url.endswith("/api/extensions/gate_skill/settings")):
        page.evaluate("() => window.dispatchEvent(new CustomEvent('ouro:skill-lifecycle', "
                      "{ detail: { action: 'review', name: 'gate_skill' } }))")
    _nav(page, "chat")
    held.pop().fulfill(json=dict(values))
    page.wait_for_timeout(300)  # the late answer has been taken in
    assert field.input_value() == "first", "a read that landed after the owner left drew into the hidden form"
    hold["on"] = False
    _nav(page, "skills")
    page.wait_for_function("() => document.querySelector('#page-skills [data-extension-settings-form] [name=\"address\"]')"
                           "?.value === 'second'")
    page.close()


def _return(page, method):
    if method == "button":
        header_back(page).click()
    elif method == "escape":
        page.keyboard.press("Escape")
    else:
        page.go_back()


@pytest.mark.parametrize("edited", ["settings", "notes"], ids=["same-form", "sibling-form"])
def test_skill_save_preserves_edits_until_their_own_save(subscription_ui, edited):
    from playwright.sync_api import expect

    page, _ = _page(subscription_ui, 1360, 900)
    listing = deepcopy(SKILLS)
    components = listing["live"]["settings_sections"][0]["render"]["components"]
    components.append({**components[0], "id": "notes", "route": "notes"})
    page.route("**/api/extensions", lambda route: route.fulfill(json=listing))
    stored = {"settings": "initial setting", "notes": "initial notes"}
    writes, held = [], []

    def form_route(route):
        key = route.request.url.rsplit("/", 1)[-1]
        if route.request.method == "POST":
            stored[key] = route.request.post_data_json["address"]
            writes.append((key, stored[key]))
            if len(writes) == 1:
                held.append(route)
                return
        route.fulfill(json={"address": stored[key]})

    for key in stored:
        page.route(f"**/api/extensions/gate_skill/{key}", form_route)
    page.goto(subscription_ui["url"] + "/#skills")
    editor = open_settings(page, "skill/gate_skill")
    first = editor.locator('form[data-route="settings"]')
    draft = editor.locator(f'form[data-route="{edited}"]')
    first.locator("input").fill("sent")
    with page.expect_request(lambda req: req.method == "POST" and req.url.endswith("/gate_skill/settings")):
        first.get_by_role("button", name="Save").click()
    draft.locator("input").fill("new draft")
    draft.locator("input").blur()
    held.pop().fulfill(json={"ok": True})
    expect(first.get_by_role("button", name="Save")).to_be_enabled()
    expect(first.locator("[data-extension-settings-status]")).to_have_text(
        "Earlier changes saved. Your latest edits are unsaved." if edited == "settings" else "Saved.")
    _nav(page, "chat")
    _nav(page, "skills")
    # If the return starts a hydration, let it settle before checking the draft.
    page.wait_for_load_state("networkidle")
    expect(draft.locator("input")).to_have_value("new draft")
    assert writes == [("settings", "sent")]
    draft.get_by_role("button", name="Save").click()
    expect(draft.locator("[data-extension-settings-status]")).to_have_text("Saved.")
    assert writes[-1] == (edited, "new draft")
    page.close()


def test_new_mcp_server_survives_back_until_discard(subscription_ui):
    page, _ = _page(subscription_ui, 1360, 900)
    page.goto(subscription_ui["url"] + "/#skills")
    editor = open_settings(page, "tools/new")
    name = editor.locator('[data-mcp-field="name"]')
    name.fill("Draft server")
    header_back(page).click()
    add = settings_page(page, "tools").locator("[data-tools-add]")
    add.click()
    assert name.input_value() == "Draft server"
    editor.get_by_role("button", name="Discard", exact=True).click()
    add.click()
    assert name.input_value() == ""
    page.close()


@pytest.mark.parametrize("start_width,end_width", [(390, 1087), (1440, 390)])
@pytest.mark.parametrize("method", ["button", "escape", "browser-back"])
@pytest.mark.parametrize("family", ["mcp", "skill"])
@pytest.mark.parametrize("draft", [False, True], ids=["clean", "draft"])
def test_retained_form_survives_accounts_then_another_extension(subscription_ui, start_width, end_width, method, family, draft):
    """S → Accounts → T → Accounts → S uses the real shared stack, not a callback that reopens a route.
    Both a clean form and explicit-Save words survive the second visit to the same section."""
    from playwright.sync_api import expect

    ui = subscription_ui
    page, _reveals = _page(ui, start_width, 844)
    values, saves = {"address": "stored address"}, []
    saved_servers = [dict(SERVER)]

    def mcp_settings(route):
        if route.request.method == "POST":
            saved_servers[:] = route.request.post_data_json["MCP_SERVERS"]
            route.fallback()  # the fixture records the write and acknowledges it
        else:
            route.fulfill(json={**ui["settings"], "MCP_SERVERS": saved_servers})

    def skill_values(route):
        if route.request.method == "POST":
            saves.append(route.request.post_data_json)
            values.update(saves[-1])
        route.fulfill(json=dict(values))

    page.route("**/api/extensions/gate_skill/settings", skill_values)
    if family == "mcp" and draft:
        page.route("**/api/settings", mcp_settings)
    page.goto(ui["url"] + "/#skills")
    path = "tools/docs" if family == "mcp" else "skill/gate_skill"
    selector = '[data-mcp-field="url"]' if family == "mcp" else '[name="address"]'
    original = open_settings(page, path)
    field = original.locator(selector)
    field.wait_for(state="visible")
    expected = "https://draft.example.test/mcp" if family == "mcp" else "typed address"
    if draft:
        field.fill(expected)
    else:
        expected = field.input_value()
    field.focus()
    page.evaluate("""path => { window.__retainedS = document.querySelector(`[data-settings-page="${path}"]`); }""", path)
    assert page.evaluate("() => window.__ouroSettings.open('accounts')") is not False
    settings_page(page, "accounts").wait_for(state="visible")
    first_return = page.evaluate("() => window.__ouroLayers.returnPage().id")
    assert page.evaluate("() => window.__ouroSettings.open('tools/new')") is not False
    settings_page(page, "tools/new").locator('[data-mcp-card]').wait_for(state="visible")
    assert page.evaluate("() => !window.__retainedS.isConnected"), "T displaces S, so custody must preserve it off DOM"
    page.set_viewport_size({"width": end_width, "height": 844})
    page.locator(f"{EXTENSIONS} [data-extensions-rail]").wait_for(state="visible" if end_width == 1087 else "hidden")
    assert header_back(page).get_attribute("aria-label") == "Back to Settings › Accounts and keys"
    _return(page, method)
    settings_page(page, "accounts").wait_for(state="visible")
    assert page.locator("#page-settings.active").count() == 1
    assert page.evaluate("() => window.__ouroLayers.page().id") == first_return
    assert page.url.startswith(ui["url"]), "the first return stays in the app"
    _return(page, method)
    original.wait_for(state="visible")
    assert page.evaluate("""path => document.querySelector(`[data-settings-page="${path}"]`) === window.__retainedS""", path)
    assert field.input_value() == expected
    page.wait_for_function("""([path, selector]) => document.querySelector(`[data-settings-page="${path}"] ${selector}`) === document.activeElement""",
                           arg=[path, selector])
    assert page.evaluate("() => window.__ouroLayers.returnPage() === null"), "both owed returns were consumed exactly once"
    assert page.url.startswith(ui["url"]), "the second return must not navigate to about:blank"
    if draft:
        if family == "mcp":
            before = len(ui["posts"])
            original.locator('[data-editor-save]').click()
            page.wait_for_function("() => document.querySelector('#page-skills [data-editor-result]')?.textContent.includes('Saved')")
            writes = [body for endpoint, body in ui["posts"][before:] if endpoint == "/api/settings" and "MCP_SERVERS" in body]
            assert len(writes) == 1 and writes[0]["MCP_SERVERS"][0]["url"] == expected
        else:
            assert saves == [], "navigation does not change the explicit Save policy"
            original.locator('button[type="submit"]').click()
            page.wait_for_function("() => document.querySelector('#page-skills [data-extension-settings-status]')?.textContent.includes('Saved')")
            assert saves == [{"address": expected}]
        header_back(page).click()
        original.wait_for(state="detached")
        endpoint = "/api/settings" if family == "mcp" else "/api/extensions/gate_skill/settings"
        with page.expect_response(lambda res: res.request.method == "GET" and res.url.endswith(endpoint)) as readback:
            reopened = open_settings(page, path)
        stored = readback.value.json()
        assert (stored["MCP_SERVERS"][0]["url"] if family == "mcp" else stored["address"]) == expected
        expect(reopened.locator(selector)).to_have_value(expected)
        assert reopened.evaluate("el => el !== window.__retainedS")
    page.close()


@pytest.mark.parametrize("draft", [False, True], ids=["clean", "draft"])
def test_map_choice_releases_clean_mcp_origin_and_keeps_existing_unsaved_draft(subscription_ui, draft):
    ui = subscription_ui
    page, _reveals = _page(ui, 390, 844)
    page.goto(ui["url"] + "/#skills")
    original = open_settings(page, "tools/docs")
    field = original.locator('[data-mcp-field="url"]')
    field.wait_for(state="visible")
    if draft:
        field.fill("https://draft.example.test/mcp")
    page.evaluate("""window.__retainedS = document.querySelector('[data-settings-page="tools/docs"]')""")
    page.evaluate("() => window.__ouroSettings.open('accounts')")
    settings_page(page, "accounts").wait_for(state="visible")
    page.evaluate("() => window.__ouroSettings.open('tools/new')")
    settings_page(page, "tools/new").locator('[data-mcp-card]').wait_for(state="visible")
    _nav(page, "chat")
    assert page.evaluate("() => window.__ouroLayers.returnPage() === null")
    page.set_viewport_size({"width": 1087, "height": 844})
    reopened = open_settings(page, "tools/docs")
    reopened.locator('[data-mcp-card]').wait_for(state="visible")
    same = page.evaluate("""() => document.querySelector('[data-settings-page="tools/docs"]') === window.__retainedS""")
    assert same is draft, "the map releases a clean held origin, while the editor's unsaved policy retains a draft"
    assert reopened.locator('[data-mcp-field="url"]').input_value() == ("https://draft.example.test/mcp" if draft else SERVER["url"])
    page.close()
