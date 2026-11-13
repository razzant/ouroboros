"""The plain pages of Settings in a browser (docs/DESIGN.md "Settings"; web/modules/settings_ui.js): Permissions and
checks, Your rules, Appearance, Notifications, Voice and calls, This computer with its Advanced and About.

Each page is opened by its path, at the owner's window (1087 x 740) and at a phone's (320 x 640), dark and light:
nothing scrolls sideways and every strip fits its row or wraps inside it. Then the rows that ask first and the ones
that say two values: raising Access asks and then says the level in force beside the saved one; lowering a Safety
level asks and raising one does not; Your rules keeps the text when the server refuses it; opening to the network
asks and carries the typed password in the same write; a stored password is shown, replaced and removed like any
stored key.

The settings server is this file's own fake, shaped like the server's answer: it keeps a document, answers a write
with its receipt (`changes`: old, new and the installed value's token) and a read with what the server derives
(`_meta.effects`, `_meta.tokens`, `_meta.policy_state`, `_meta.restart_state.running`). The same rows against the real
server are in test_settings_policy_browser.py, test_deployment_policy_ui_browser.py and test_ui_selection_browser.py."""
from __future__ import annotations

import json
from urllib.parse import urlparse

import pytest

from tests import test_live_conversation_browser as calls
from tests import test_subscription_setup_browser as setup_browser
from tests import settings_browser_helpers as helpers
from tests.settings_browser_helpers import commit_row, go_back, open_settings, setting_row, settings_page

subscription_ui = setup_browser.subscription_ui
pytestmark = [pytest.mark.ui_browser]

PAGES = ["permissions", "rules", "appearance", "notifications", "voice", "computer", "computer/advanced", "computer/about"]
SIZES = [(1087, 740), (320, 640)]
EFFECTS = {"OUROBOROS_RUNTIME_MODE": "restart", "OUROBOROS_SERVER_HOST": "restart", "OUROBOROS_MAX_WORKERS": "restart",
           "OUROBOROS_UI_DIR": "restart", "OUROBOROS_TASK_REVIEW_MODE": "next_task",
           "OUROBOROS_REVIEW_MAX_CYCLES": "next_task", "OUROBOROS_SAFETY_MODE": "next_task",
           "OUROBOROS_CONSCIOUSNESS_SAFETY_MODE": "next_task", "OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS": "now",
           "OUROBOROS_NETWORK_PASSWORD": "now", "OUROBOROS_EXTRA_CA_BUNDLE": "next_task"}


class FakeSettings:
    """The settings document behind a page: reads, writes with their receipt, and the owner endpoints."""

    def __init__(self, ui, **values):
        self.posts = []
        self.refuse = {}  # path -> (status, body): the next write to it is refused
        self.effects = dict(EFFECTS)
        self.versions = {}  # key -> how many times it was written: the token of its installed value
        self.running = {"OUROBOROS_RUNTIME_MODE": "pro", "OUROBOROS_SERVER_HOST": "127.0.0.1"}
        self.values = {
            **{key: value for key, value in ui["settings"].items() if key != "_meta"},
            "OUROBOROS_RUNTIME_MODE": "pro", "OUROBOROS_SAFETY_MODE": "full", "OUROBOROS_CONSCIOUSNESS_SAFETY_MODE": "light",
            "OUROBOROS_SAFETY_OWNER_RULES": "Never email my clients without asking me first.",
            "OUROBOROS_TASK_REVIEW_MODE": "auto", "OUROBOROS_REVIEW_MAX_CYCLES": "2",
            "OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS": "false", "OUROBOROS_SERVER_HOST": "127.0.0.1", "OUROBOROS_NETWORK_PASSWORD": "",
            "OUROBOROS_MAX_WORKERS": 10, "OUROBOROS_UI_DIR": "", "OUROBOROS_EXTRA_CA_BUNDLE": "", "OUROBOROS_MODEL_SAFETY": "",
            **values,
        }
        self.setup_contract = ui["settings"]["_meta"]["setup_contract"]

    def document(self):
        values = self.values
        pending = sorted(key for key, value in self.running.items() if str(values.get(key)) != str(value))
        lan = self.running["OUROBOROS_SERVER_HOST"] == "0.0.0.0"
        meta = {
            "setup_contract": self.setup_contract, "effects": self.effects,
            "tokens": {key: self._token(key) for key in values if key != "OUROBOROS_NETWORK_PASSWORD"},
            "bind_host": self.running["OUROBOROS_SERVER_HOST"], "bind_port": 8765,
            "reachability": "lan_reachable" if lan else "loopback_only",
            "recommended_url": "http://192.168.1.20:8765" if lan else "", "warning": "",
            "policy_state": {
                "access": {"configured": values["OUROBOROS_RUNTIME_MODE"], "effective": self.running["OUROBOROS_RUNTIME_MODE"],
                           "current_process": self.running["OUROBOROS_RUNTIME_MODE"],
                           "restart_required": values["OUROBOROS_RUNTIME_MODE"] != self.running["OUROBOROS_RUNTIME_MODE"]},
            },
            "restart_state": {"restart_required": bool(pending), "restart_keys": pending,
                              "running": {key: self.running[key] for key in pending}},
        }
        masked = {**values, "OUROBOROS_NETWORK_PASSWORD": "***set***" if values["OUROBOROS_NETWORK_PASSWORD"] else ""}
        return {**masked, "_meta": meta}

    def _token(self, key):
        return f"{key}#{self.versions.get(key, 0)}"

    def _write(self, changed):
        changes = {}
        for key, value in changed.items():
            if str(self.values.get(key)) == str(value):
                continue
            old, self.values[key] = self.values.get(key), value
            self.versions[key] = self.versions.get(key, 0) + 1
            changes[key] = ({"withheld": "secret"} if key == "OUROBOROS_NETWORK_PASSWORD"
                            else {"old": old, "new": value, "token": self._token(key)})
        return changes

    def answer(self, route):
        request = route.request
        path = urlparse(request.url).path
        if request.method != "POST":
            route.fulfill(content_type="application/json", body=json.dumps(self.document()))
            return
        body = request.post_data_json or {}
        self.posts.append((path, body))
        if path in self.refuse:
            status, refusal = self.refuse.pop(path)
            route.fulfill(status=status, content_type="application/json", body=json.dumps(refusal))
            return
        if path == "/api/settings":
            changes = self._write({key: value for key, value in body.items() if key not in ("if_unchanged", "unset")})
            answer = {"status": "saved", "changes": changes, **({} if changes else {"no_changes": True})}
        elif path == "/api/owner/runtime-mode":
            answer = {"ok": True, "runtime_mode": body["mode"], "changes": self._write({"OUROBOROS_RUNTIME_MODE": body["mode"]})}
        elif path == "/api/owner/safety-mode":
            key = "OUROBOROS_CONSCIOUSNESS_SAFETY_MODE" if body.get("kind") == "consciousness" else "OUROBOROS_SAFETY_MODE"
            answer = {"ok": True, "safety_mode": body["mode"], "changes": self._write({key: body["mode"]})}
        elif path == "/api/owner/safety-rules":
            answer = {"ok": True, "rules": body["rules"], "changes": self._write({"OUROBOROS_SAFETY_OWNER_RULES": body["rules"]})}
        else:  # /api/owner/auto-grant
            value = "true" if body.get("enabled") else "false"
            answer = {"ok": True, "enabled": body.get("enabled"), "changes": self._write({"OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS": value})}
        route.fulfill(content_type="application/json", body=json.dumps(answer))

    def writes(self, path):
        return [body for posted, body in self.posts if posted == path]


def _page(ui, width, height, theme="dark", **values):
    page = ui["new_page"](viewport={"width": width, "height": height}, has_touch=width < 720)
    server = FakeSettings(ui, **values)
    # The page's own routes answer before the fixture's catch-all.
    page.route(lambda url: urlparse(url).path == "/api/settings" and not urlparse(url).query, server.answer)
    page.route("**/api/owner/**", server.answer)
    page.route("**/api/desktop/autostart", lambda route: route.fulfill(content_type="application/json", body='{"state": "on"}'))
    page.route("**/api/desktop/background", lambda route: route.fulfill(content_type="application/json", body='{"state": "off"}'))
    page.route("**/api/logs/events**", lambda route: route.fulfill(content_type="application/json", body='{"entries": []}'))
    page.route("**/api/ui/i18n", lambda route: route.fulfill(content_type="application/json", body=json.dumps(
        {"language": "en", "english": True, "chosen": False, "languages": [], "revision": 1, "entries": {}})))
    page.goto(ui["url"] + "/#settings")
    page.wait_for_function("() => Boolean(window.__ouroSettings && window.ouroTheme)")
    page.evaluate("theme => window.ouroTheme.set(theme)", theme)
    return page, server


def _settled(page, path):
    """The page at `path` has drawn its body and holds no write in flight."""
    level = settings_page(page, path)
    level.locator(".settings-page-body > *").first.wait_for(state="visible")
    page.wait_for_function("() => !document.querySelector('#page-settings [aria-busy=\"true\"]')")
    return level


@pytest.mark.parametrize("width,height", SIZES)
@pytest.mark.parametrize("theme", ["dark", "light"])
def test_every_plain_page_fits_its_window_without_sideways_scroll(subscription_ui, width, height, theme):
    page, _server = _page(subscription_ui, width, height, theme, OUROBOROS_RUNTIME_MODE="cyber_pro")
    for path in PAGES:
        open_settings(page, path)
        level = _settled(page, path)
        if path == "permissions":  # a restart waits: the Access row says both values
            page.wait_for_function("() => /Running: Pro/.test(document.querySelector("
                                   "'[data-setting-row=\"OUROBOROS_RUNTIME_MODE\"] [data-setting-meta]')?.textContent || '')")
        fits = level.evaluate("""level => {
            const wide = (el) => el.scrollWidth - el.clientWidth;
            const out = [...level.querySelectorAll('.ui-segment-group, .settings-choices, .settings-slider, .ui-row-actions')]
                .filter((el) => el.getClientRects().length)
                .filter((el) => { const row = el.closest('.ui-row, .settings-card'); const a = el.getBoundingClientRect();
                    const b = row.getBoundingClientRect(); return a.left < b.left - 0.5 || a.right > b.right + 0.5; })
                .map((el) => el.className);
            return { page: wide(level), doc: wide(document.documentElement), out };
        }""")
        assert fits == {"page": 0, "doc": 0, "out": []}, f"{path} at {width}x{height} {theme}: {fits}"
        setup_browser.capture(page, f"c2-{path.replace('/', '-')}-{width}x{height}-{theme}")
        if path == "permissions":
            level.locator("details[data-settings-fold] > summary").click()
            level.evaluate("level => { level.scrollTop = level.scrollHeight; }")
            setup_browser.capture(page, f"c2-permissions-end-{width}x{height}-{theme}")


def test_raising_access_asks_first_and_then_says_both_values(subscription_ui):
    page, server = _page(subscription_ui, 1087, 740)
    open_settings(page, "permissions")
    _settled(page, "permissions")
    row = setting_row(page, "OUROBOROS_RUNTIME_MODE")
    assert row.locator('[role="radio"][aria-checked="true"]').inner_text().startswith("Pro")
    assert row.locator("[data-setting-meta]").inner_text() == "How far it reaches into its own code and this computer."

    # Lowering Access asks nothing and writes at once; a restart row says no "was" (its meta says both values).
    assert commit_row(page, "OUROBOROS_RUNTIME_MODE", "advanced") == "Saved · applies after restart"
    page.wait_for_function("() => /Running: Pro · after restart: Advanced/.test(document.querySelector("
                           "'[data-setting-row=\"OUROBOROS_RUNTIME_MODE\"] [data-setting-meta]').textContent)")
    assert server.writes("/api/owner/runtime-mode") == [{"mode": "advanced"}]
    assert [mark for mark in row.locator("[data-setting-mark]").all_inner_texts() if mark] == ["After restart", "Now"]

    # Raising it asks first; declined, nothing is sent.
    row.locator('[data-setting-value="cyber_pro"]').click()
    dialog = page.locator("dialog[open], .confirm-dialog").last
    dialog.wait_for(state="visible")
    assert "Raise Access from Advanced to Cyber Pro" in dialog.inner_text()
    dialog.get_by_role("button", name="Cancel").click()
    dialog.wait_for(state="hidden")
    assert len(server.writes("/api/owner/runtime-mode")) == 1
    row.locator('[data-setting-value="cyber_pro"]').click()
    dialog = page.locator("dialog[open], .confirm-dialog").last
    dialog.get_by_role("button", name="Raise Access").click()
    page.wait_for_function("() => /Running: Pro · after restart: Cyber Pro/.test(document.querySelector("
                           "'[data-setting-row=\"OUROBOROS_RUNTIME_MODE\"] [data-setting-meta]').textContent)")
    assert server.writes("/api/owner/runtime-mode")[-1] == {"mode": "cyber_pro"}
    assert row.locator("[data-setting-result]").inner_text() == "Saved · applies after restart"
    assert row.locator("[data-setting-undo]").count() == 0, "Access is undone by choosing again, which may ask"
    setup_browser.capture(page, "c2-permissions-restart-waits-1087x740-dark")


def test_lowering_a_safety_level_asks_and_raising_does_not(subscription_ui):
    from playwright.sync_api import expect

    page, server = _page(subscription_ui, 1087, 740)
    open_settings(page, "permissions")
    _settled(page, "permissions")
    row = setting_row(page, "OUROBOROS_SAFETY_MODE")
    assert row.locator(".ui-segment").all_inner_texts() == ["Off", "Outgoing", "+ Commands"]
    assert row.locator("[data-setting-meta]").inner_text() == "Shell commands and scripts are read first too. File edits are not."
    waiting = []

    def hold_safety(route):
        waiting.append(route)
        page.evaluate("window.safetyWriteHeld = true")

    page.route("**/api/owner/safety-mode", hold_safety)
    row.locator('[data-setting-value="light"]').click()
    dialog = page.locator("dialog[open], .confirm-dialog").last
    dialog.wait_for(state="visible")
    assert "from + Commands to Outgoing" in dialog.inner_text()
    dialog.get_by_role("button", name="Lower the level").click()
    page.wait_for_function("() => document.querySelector('[data-setting-row=\"OUROBOROS_SAFETY_MODE\"] "
                           "[data-setting-value=\"light\"]').getAttribute('aria-pressed') === 'true'")
    page.wait_for_function("window.safetyWriteHeld === true")
    # The selected option is optimistic; a held answer has no saved result yet.
    expect(row.locator("[data-setting-result]")).to_have_text("")
    page.unroute("**/api/owner/safety-mode", hold_safety)
    server.answer(waiting.pop())
    expect(row.locator("[data-setting-result]")).to_have_text("Saved · applies to new tasks · was + Commands")
    assert server.writes("/api/owner/safety-mode") == [{"mode": "light"}]
    # Raising the other level asks nothing.
    assert commit_row(page, "OUROBOROS_CONSCIOUSNESS_SAFETY_MODE", "full") == "Saved · applies to new tasks · was Outgoing"
    assert server.writes("/api/owner/safety-mode")[-1] == {"mode": "full", "kind": "consciousness"}
    assert not page.locator("dialog[open]").count()


def test_your_rules_saves_by_its_own_button_and_keeps_the_text_the_server_refused(subscription_ui):
    page, server = _page(subscription_ui, 1087, 740)
    level = open_settings(page, "rules")
    field = level.locator("#settings-rules")
    page.wait_for_function("() => document.querySelector('#settings-rules').value.startsWith('Never email')")
    typed = "Never email my clients.\nAlways ask before spending more than $20."
    server.refuse["/api/owner/safety-rules"] = (400, {"error": "Rules are limited to 64,000 bytes.", "saved": False})
    field.fill(typed)
    level.locator("[data-editor-save]").click()
    result = level.locator("[data-editor-result]")
    page.wait_for_function("() => /Not saved/.test(document.querySelector('[data-editor-result]').textContent)")
    assert result.inner_text() == "Not saved. Rules are limited to 64,000 bytes."
    assert field.input_value() == typed, "a refusal keeps the text in the field"
    level.locator("[data-editor-save]").click()
    page.wait_for_function("() => /^Saved/.test(document.querySelector('[data-editor-result]').textContent)")
    assert result.inner_text() == "Saved · applies from the next task or turn"
    assert server.writes("/api/owner/safety-rules")[-1] == {"rules": typed}
    assert "Running work keeps the text it started with." in level.inner_text()


def test_opening_to_the_network_asks_and_carries_the_typed_password(subscription_ui):
    page, server = _page(subscription_ui, 1087, 740)
    level = open_settings(page, "computer")
    _settled(page, "computer")
    row = setting_row(page, "OUROBOROS_SERVER_HOST")
    assert row.locator(".ui-segment").all_inner_texts() == ["Only here", "My network"]
    assert not level.locator('[data-setting-row="computer:address"]').is_visible(), "no address while only this computer reaches it"
    password = setting_row(page, "OUROBOROS_NETWORK_PASSWORD")
    assert password.locator("[data-secret-open]").inner_text() == "Add", "no password is set: its field opens on Add"
    password.locator("[data-secret-open]").click()
    level.locator('[id="setting-OUROBOROS_NETWORK_PASSWORD"]').type("correct horse battery")
    row.locator('[data-setting-value="0.0.0.0"]').click()
    dialog = page.locator("dialog[open], .confirm-dialog").last
    dialog.wait_for(state="visible")
    assert "Other devices on your network will reach Ouroboros with the network password" in dialog.inner_text()
    dialog.get_by_role("button", name="Open to the network").click()
    page.wait_for_function("() => /Running: Only here · after restart: My network/.test(document.querySelector("
                           "'[data-setting-row=\"OUROBOROS_SERVER_HOST\"] [data-setting-meta]').textContent)")
    hosts = [body for body in server.writes("/api/settings") if "OUROBOROS_SERVER_HOST" in body]
    assert hosts == [{"OUROBOROS_SERVER_HOST": "0.0.0.0", "OUROBOROS_NETWORK_PASSWORD": "correct horse battery"}], \
        "the host and the typed password travel in one body"
    assert server.values["OUROBOROS_NETWORK_PASSWORD"] == "correct horse battery"
    assert row.locator("[data-setting-result]").inner_text() == "Saved · applies after restart · Undo"
    # A secret says it is saved and nothing of its value; it offers no Undo.
    assert password.locator("[data-setting-result]").inner_text() == "Saved"
    assert password.locator("[data-secret-state]").inner_text() == "Saved"
    assert password.locator("[data-setting-undo]").count() == 0
    setup_browser.capture(page, "c2-computer-restart-waits-1087x740-dark")
    # Undo puts the address back with the token of the value this write installed.
    row.locator("[data-setting-undo]").click()
    page.wait_for_function("() => document.querySelector('[data-setting-row=\"OUROBOROS_SERVER_HOST\"] "
                           "[data-setting-meta]').textContent === 'Only this computer.'")
    assert server.writes("/api/settings")[-1] == {"OUROBOROS_SERVER_HOST": "127.0.0.1",
                                                  "if_unchanged": {"OUROBOROS_SERVER_HOST": "OUROBOROS_SERVER_HOST#1"}}


def test_a_stored_network_password_is_shown_replaced_and_removed_like_a_stored_key(subscription_ui):
    page, server = _page(subscription_ui, 1087, 740, OUROBOROS_NETWORK_PASSWORD="old horse")
    revealed = []

    def reveal(route):
        body = route.request.post_data_json or {}
        revealed.append(body)
        route.fulfill(content_type="application/json", body=json.dumps({"value": server.values[body["key"]]}))

    page.route("**/api/settings/secret", reveal)
    open_settings(page, "computer")
    _settled(page, "computer")
    password = setting_row(page, "OUROBOROS_NETWORK_PASSWORD")
    opener, show, remove = (password.locator(hook) for hook in ("[data-secret-open]", "[data-secret-show]", "[data-secret-remove]"))
    page.wait_for_function("() => document.querySelector('[data-setting-row=\"OUROBOROS_NETWORK_PASSWORD\"] "
                           "[data-secret-state]')?.textContent === 'Saved'")
    assert opener.inner_text() == "Replace" and not show.is_visible(), "at rest the row says only that one is saved"
    # Replace opens its field: Show and Remove stand beside it; Show reads the stored value on the press.
    opener.click()
    assert show.is_visible() and remove.is_visible()
    show.click()
    shown = password.locator(".settings-secret-value")
    shown.wait_for(state="visible")
    assert shown.inner_text() == "old horse" and revealed == [{"key": "OUROBOROS_NETWORK_PASSWORD"}]
    setup_browser.capture(page, "c2-computer-password-shown-1087x740-dark")
    # A typed new one replaces it: Saved alone, no Undo, and the field is put away.
    show.click()
    assert commit_row(page, "OUROBOROS_NETWORK_PASSWORD", "new horse") == "Saved"
    assert server.values["OUROBOROS_NETWORK_PASSWORD"] == "new horse"
    assert password.locator("[data-setting-undo]").count() == 0
    password.locator("[data-secret-field]").wait_for(state="hidden")
    # Remove asks first. Still open to the network, the server refuses: its sentence is the row's line.
    refusal = "Cannot clear Network Password while the running server is still bound to a non-localhost interface."
    server.refuse["/api/settings"] = (400, {"error": refusal, "saved": False})
    opener.click()
    remove.click()
    dialog = page.locator("dialog[open], .confirm-dialog").last
    dialog.wait_for(state="visible")
    assert "Remove Password?" in dialog.inner_text()
    dialog.get_by_role("button", name="Remove").click()
    page.wait_for_function("() => /^Not saved/.test(document.querySelector('[data-setting-row=\"OUROBOROS_NETWORK_PASSWORD\"] "
                           "[data-setting-result]').textContent)")
    assert password.locator("[data-setting-result]").inner_text() == f"Not saved. {refusal}"
    assert server.values["OUROBOROS_NETWORK_PASSWORD"] == "new horse"
    # Accepted, it is cleared through the row's own write, and the row offers Add.
    remove.click()
    page.locator("dialog[open], .confirm-dialog").last.get_by_role("button", name="Remove").click()
    page.wait_for_function("() => document.querySelector('[data-setting-row=\"OUROBOROS_NETWORK_PASSWORD\"] "
                           "[data-secret-open]').textContent === 'Add'")
    assert server.writes("/api/settings")[-2:] == [{"OUROBOROS_NETWORK_PASSWORD": ""}] * 2
    assert server.values["OUROBOROS_NETWORK_PASSWORD"] == ""
    assert password.locator("[data-setting-result]").inner_text() == "Saved"
    assert not remove.is_visible()


def test_voice_and_calls_says_the_saved_choices_and_opens_the_form_of_the_skill_that_registered_a_call(subscription_ui, monkeypatch):
    """No setting lives on this page. The skill is the bundled Calls skill as its own tests stand it up: its real
    registration (the conversation it contributes, the settings section "Calls (beta)") and its real page module."""
    page, _posts = calls.open_call_ui(subscription_ui, monkeypatch, width=1087, scheme="dark")
    writes = []
    page.on("request", lambda request: request.method == "POST" and writes.append(request.url))
    level = open_settings(page, "voice")
    door = level.locator('[data-setting-row="voice:skill:live_conversation"]')
    door.wait_for(state="visible")  # drawn once the skills were read: a skill with a form gets a door
    assert level.locator("[data-setting-row]").evaluate_all("rows => rows.map((row) => row.dataset.settingRow)") == \
        ["voice:dictation", "voice:read-aloud", "voice:skill:live_conversation"]
    assert level.locator("input, select, textarea, [role='switch']").count() == 0
    dictation = setting_row(page, "voice:dictation")
    assert dictation.locator("[data-setting-meta]").inner_text() == "Chosen at the message field: +, then Add dictation."
    assert dictation.locator('[data-fact="voice:dictation"]').inner_text() == "Whisper", "the engine saved in the preferences"
    assert setting_row(page, "voice:read-aloud").locator("[data-setting-meta]").inner_text().startswith("Chosen in a reply")
    assert door.locator(".ui-row-title").inner_text() == "Calls (beta)", "the door is named by the form's own title"
    setup_browser.capture(page, "c2-voice-with-calls-1087x740-dark")
    door.click()
    opened = settings_page(page, "skill/live_conversation")
    opened.wait_for(state="visible")
    assert opened.locator("[data-settings-back]").inner_text().strip() == "Voice and calls"
    assert opened.locator(".settings-page-title").inner_text().strip() == "Calls (beta)"
    form = opened.locator('[data-extension-settings-form][data-skill="live_conversation"]')
    form.wait_for(state="visible")
    assert form.locator('[name="enabled"]').count() == 1 and form.locator('button[type="submit"]').count() == 1
    setup_browser.capture(page, "c2-voice-calls-form-1087x740-dark")
    assert go_back(page) == "voice"
    assert not [url for url in writes if "/api/settings" in url or "/api/owner/" in url or "/config" in url], writes


def test_the_shared_helper_settles_a_save_that_says_the_same_words_and_presses_a_switch_by_its_track(subscription_ui):
    """tests/settings_browser_helpers.py `commit_row`. Two Access saves both say "Saved · applies after restart": the
    second settles because the row was seen busy, not because its words changed; and an old line alone, with no write,
    is no sign. A switch is pressed on its track: its input lies under it."""
    page, server = _page(subscription_ui, 1087, 740, OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS="true")
    server.effects["OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS"] = "next_task"  # a delayed switch says a line; one that acts now is quiet
    open_settings(page, "permissions")
    _settled(page, "permissions")
    assert commit_row(page, "OUROBOROS_RUNTIME_MODE", "advanced") == "Saved · applies after restart"
    assert commit_row(page, "OUROBOROS_RUNTIME_MODE", "light") == "Saved · applies after restart"
    assert server.writes("/api/owner/runtime-mode") == [{"mode": "advanced"}, {"mode": "light"}]
    # The watch alone, with nothing pressed: the old line is still there and is not taken for a new save.
    page.evaluate(helpers._WATCH_ROW, "OUROBOROS_RUNTIME_MODE")
    page.wait_for_timeout(300)
    assert setting_row(page, "OUROBOROS_RUNTIME_MODE").locator("[data-setting-result]").inner_text() == "Saved · applies after restart"
    assert page.evaluate("() => window.__ouroRowWatch.seen.moved") is False
    # The switch: Playwright cannot press the input (the track lies over it); the helper presses the track.
    box = setting_row(page, "OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS").locator("[data-setting-control]")
    assert box.is_checked()
    with pytest.raises(Exception, match="intercepts pointer events"):
        box.click(timeout=1_500)
    assert commit_row(page, "OUROBOROS_AUTO_GRANT_REVIEWED_SKILLS", False) == "Saved · applies to new tasks"
    assert not box.is_checked() and server.writes("/api/owner/auto-grant") == [{"enabled": False}]
