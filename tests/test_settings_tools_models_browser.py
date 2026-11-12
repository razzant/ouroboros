"""Settings › Tools and Settings › Models in a browser (docs/DESIGN.md "Settings"; web/modules/mcp_settings.js,
model_roles.js, settings_local_model.js).

Both lists and their editors are opened by path at the owner's window (1087 x 740) and at a phone's (320 x 640), dark
and light: nothing scrolls sideways. Then the pages are reached the way the owner reaches them (a home row, search)
and used: a server's switch writes its one entry, a stale list is refused and shown again, a server's own Save and
Remove, a role's Save through the checker's transaction or the generic save, the fallbacks' order, the Context door.

The settings server is this file's own fake, shaped like the server's answer: it keeps a document, checks
`if_unchanged` against the token of each key, answers a write with its receipt (`changes`) and a read with
`_meta.effects` and `_meta.tokens`."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from urllib.parse import urlparse

import pytest

from tests import test_subscription_setup_browser as setup_browser
from tests.settings_browser_helpers import (
    HOME, commit_row, go_back, open_entry, open_settings, press_editor_save, search_settings, setting_row, settings_page, top_path)

subscription_ui = setup_browser.subscription_ui
pytestmark = [pytest.mark.ui_browser]

SIZES = [(1087, 740), (320, 640)]
CHECKER_KEYS = {"OUROBOROS_MODEL_SAFETY", "OUROBOROS_MODEL_LIGHT", "OUROBOROS_MODEL", "USE_LOCAL_LIGHT", "USE_LOCAL_MAIN",
                "OUROBOROS_MODEL_ACCOUNTS", "OUROBOROS_MODEL_CONTEXT_WINDOWS", "OUROBOROS_MODEL_PROCESSING_PREFERENCES",
                "OUROBOROS_SUBAGENTS"}
SERVERS = [
    {"id": "slack", "name": "Slack", "enabled": True, "transport": "streamable_http", "url": "https://mcp.example.com/slack",
     "auth_header": "Authorization", "auth_token": "xoxb...", "auth_configured": True},
    {"id": "github", "name": "GitHub", "enabled": True, "transport": "streamable_http", "url": "https://mcp.example.com/github",
     "headers": {"X-API-Key": "***set***"}},
    {"id": "browser", "name": "Browser (Playwright)", "enabled": False, "transport": "stdio", "command": "npx",
     "args": ["@playwright/mcp@latest", "--headless"], "task_session": True},
]


def _effect(key):
    if key.startswith("LOCAL_MODEL_"):
        return "local_model"
    if key.startswith("MCP_") or key in ("OUROBOROS_TOOL_TIMEOUT_SEC", "OUROBOROS_MODEL_MAX_CONCURRENCY"):
        return "now"
    if key == "OUROBOROS_MODEL_CONTEXT_WINDOWS":
        return "next_task"
    return "next_call" if ("MODEL" in key or "USE_LOCAL" in key or "WEBSEARCH" in key or "PROCESSING" in key) else "next_task"


class FakeSettings:
    """The settings document behind the pages: reads, both write endpoints with their receipt, the tool status."""

    def __init__(self, ui, **values):
        self.posts = []          # (path, settings, if_unchanged)
        self.stale_next = False  # the next write finds the list changed by another device
        self.versions = {}
        self.values = {
            **{key: value for key, value in ui["settings"].items() if key != "_meta"},
            "MCP_ENABLED": True, "MCP_TOOL_TIMEOUT_SEC": 60, "OUROBOROS_TOOL_TIMEOUT_SEC": 600, "MCP_SERVERS": json.loads(json.dumps(SERVERS)),
            "OUROBOROS_MODEL": "claudexor::codex=gpt-test", "OUROBOROS_MODEL_LIGHT": "openai::gpt-light", "OUROBOROS_MODEL_SAFETY": "",
            "OUROBOROS_MODEL_VISION": "", "OUROBOROS_MODEL_CONSCIOUSNESS": "",
            "OUROBOROS_MODEL_FALLBACKS": "anthropic::claude-opus-test, moonshotai/kimi-k3-pro, openai::gpt-sol",
            "USE_LOCAL_MAIN": False, "USE_LOCAL_LIGHT": False, "USE_LOCAL_CONSCIOUSNESS": False, "USE_LOCAL_FALLBACK": False,
            "OUROBOROS_MODEL_ACCOUNTS": {}, "OUROBOROS_MODEL_CONTEXT_WINDOWS": {"fallback": [0, 128000, 0]},
            "OUROBOROS_MODEL_PROCESSING_PREFERENCES": {}, "OUROBOROS_PROCESSING_PREFERENCE": "",
            "OUROBOROS_WEBSEARCH_BACKEND": "auto", "OUROBOROS_WEBSEARCH_MODEL": "", "OUROBOROS_CONTEXT_MODE": "max",
            "OUROBOROS_IMAGE_INPUT_MODE": "auto", "OUROBOROS_PROMPT_CACHE_TTL": "1h", "OUROBOROS_MODEL_MAX_CONCURRENCY": 3,
            "OUROBOROS_MODEL_SLOT_MAX_WAIT_SEC": 180, "OPENAI_API_KEY": "***set***", "ANTHROPIC_API_KEY": "***set***", "OPENROUTER_API_KEY": "***set***",
            "LOCAL_MODEL_SOURCE": "bartowski/Llama-3.3-70B-Instruct-GGUF", "LOCAL_MODEL_FILENAME": "Llama-3.3-70B-Instruct-Q4_K_M.gguf",
            "LOCAL_MODEL_PORT": 8766, "LOCAL_MODEL_N_GPU_LAYERS": -1, "LOCAL_MODEL_CONTEXT_LENGTH": 16384, "LOCAL_MODEL_CHAT_FORMAT": "",
            **values,
        }
        self.setup_contract = ui["settings"]["_meta"]["setup_contract"]

    def _token(self, key):
        return f"{key}#{self.versions.get(key, 0)}"

    def document(self):
        return {**json.loads(json.dumps(self.values)), "_meta": {
            "setup_contract": self.setup_contract, "effects": {key: _effect(key) for key in self.values},
            "tokens": {key: self._token(key) for key in self.values if not key.endswith("_API_KEY")}}}

    def status(self):
        now = datetime.now(timezone.utc).isoformat()
        rows = {"slack": {"tool_count": 38, "listed_at": now, "schema_tokens": 9200, "map_tokens": 310,
                          "sign_in": {"state": "not_signed_in", "reason": "expired"},
                          "tools": [{"name": "search_messages", "description": "Search messages in channels."},
                                    {"name": "post_message", "description": "Post a message."}]},
                "github": {"tool_count": 51, "listed_at": now, "schema_tokens": 14800, "map_tokens": 420, "tools": []},
                "browser": {"tool_count": 0, "listed_at": "", "schema_tokens": None,
                            "latest_attempt": {"ok": False, "at": now, "error": "npx: command not found"}, "last_error": "npx: command not found", "tools": []}}
        servers = [{"id": entry["id"], **rows.get(entry["id"], {"tool_count": 0, "listed_at": "", "tools": []})} for entry in self.values["MCP_SERVERS"]]
        return {"sdk_available": True, "enabled": bool(self.values["MCP_ENABLED"]), "servers": servers,
                "weights": {"map_tokens": 730, "schema_tokens_from_task_start": 0}}

    def answer(self, route):
        request = route.request
        path = urlparse(request.url).path
        if path == "/api/mcp/status":
            route.fulfill(content_type="application/json", body=json.dumps(self.status()))
            return
        if request.method != "POST":
            route.fulfill(content_type="application/json", body=json.dumps(self.document()))
            return
        body = request.post_data_json or {}
        settings = body.get("settings") if path == "/api/owner/safety-model" else {key: value for key, value in body.items() if key != "if_unchanged"}
        held = body.get("if_unchanged") or {}
        self.posts.append((path, settings, held))
        if self.stale_next:  # another device wrote the first server off between this page's read and its write
            self.stale_next = False
            self.values["MCP_SERVERS"][0]["enabled"] = not self.values["MCP_SERVERS"][0]["enabled"]
            self.versions["MCP_SERVERS"] = self.versions.get("MCP_SERVERS", 0) + 1
        stale = [key for key, token in held.items() if token != self._token(key)]
        if stale:
            route.fulfill(status=409, content_type="application/json", body=json.dumps(
                {"saved": False, "code": "settings_changed_elsewhere", "error": f"{stale[0]} changed on another device."}))
            return
        changes = {}
        for key, value in settings.items():
            old = self.values.get(key)
            self.values[key] = value
            self.versions[key] = self.versions.get(key, 0) + 1
            changes[key] = ({"withheld": "structure", "token": self._token(key)} if isinstance(value, (dict, list))
                            else {"old": old, "new": value, "token": self._token(key)})
        route.fulfill(content_type="application/json", body=json.dumps({"ok": True, "status": "saved", "changes": changes}))

    def writes(self, path="/api/settings"):
        return [settings for posted, settings, _held in self.posts if posted == path]


def _page(ui, width=1087, height=740, theme="dark", **values):
    page = ui["new_page"](viewport={"width": width, "height": height}, has_touch=width < 720)
    server = FakeSettings(ui, **values)
    # The page's own routes answer before the fixture's catch-all.
    page.route(lambda url: urlparse(url).path == "/api/settings" and not urlparse(url).query, server.answer)
    page.route("**/api/owner/safety-model", server.answer)
    page.route("**/api/mcp/status", server.answer)
    page.route("**/api/local-model/status", lambda route: route.fulfill(content_type="application/json", body=json.dumps(
        {"status": "ready", "context_length": 16384, "runtime_status": "ok",
         "settings_application": {"status": "ready", "pending_keys": [], "unknown_keys": [], "summary": ""}})))
    page.route(lambda url: urlparse(url).path == "/api/settings" and "websearch_preview" in urlparse(url).query,
               lambda route: route.fulfill(content_type="application/json", body=json.dumps(
                   {"legs": [{"source": "openai", "model": "gpt-search", "model_source": "provider_default"}], "unavailable": []})))
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


FITS = """level => {
    const wide = (el) => el.scrollWidth - el.clientWidth;
    const out = [...level.querySelectorAll('.ui-row-actions, .ui-action-row, .ui-segment-group, .model-role-controls, .mcp-server-form, .ui-control, .btn')]
        .filter((el) => el.getClientRects().length)
        .filter((el) => { const box = el.closest('.settings-card, .settings-page-body'); const a = el.getBoundingClientRect();
            const b = box.getBoundingClientRect(); return a.left < b.left - 0.5 || a.right > b.right + 0.5; })
        .map((el) => el.className || el.tagName);
    return { page: wide(level), doc: wide(document.documentElement), out };
}"""
# Each view with what it must show before it is measured.
VIEWS = [
    ("tools", '[data-entry-id="browser"]'), ("tools/github", "[data-mcp-card] [data-editor-save]"), ("tools/browser", "[data-mcp-card] [data-editor-save]"),
    ("tools/new", "[data-mcp-card] [data-editor-save]"), ("models", '[data-entry-id="fallback/3"]'), ("models/role/main", "[data-editor-save]"),
    ("models/role/vision", "[data-editor-save]"), ("models/role/fallback/2", "[data-editor-save]"), ("models/role/websearch", "[data-editor-save]"),
    ("models/local", "#btn-local-start"),
]


@pytest.mark.parametrize("width,height", SIZES)
@pytest.mark.parametrize("theme", ["dark", "light"])
def test_both_lists_and_their_editors_fit_the_window_without_sideways_scroll(subscription_ui, width, height, theme):
    page, _server = _page(subscription_ui, width, height, theme)
    for path, ready in VIEWS:
        open_settings(page, path)
        level = _settled(page, path)
        level.locator(ready).first.wait_for(state="visible")
        if path == "tools":
            page.wait_for_function("() => /tools · listed/.test(document.querySelector('#page-settings [data-entry-id=slack]')?.textContent || '')")
        if path == "models":
            page.wait_for_function("() => document.querySelector('#page-settings [data-setting-row=\"models:light\"] [data-row-value]')?.textContent")
        fits = level.evaluate(FITS)
        assert fits == {"page": 0, "doc": 0, "out": []}, f"{path} at {width}x{height} {theme}: {fits}"
        setup_browser.capture(page, f"e2-{path.replace('/', '-')}-{width}x{height}-{theme}")
        fold = level.locator("details[data-settings-fold] > summary, details[data-mcp-more] > summary, details.model-role-details > summary")
        if fold.count():
            fold.first.click()
            level.evaluate("level => { level.scrollTop = level.scrollHeight; }")
            assert level.evaluate(FITS) == {"page": 0, "doc": 0, "out": []}, f"{path} with its fold open at {width}x{height} {theme}"
            setup_browser.capture(page, f"e2-{path.replace('/', '-')}-end-{width}x{height}-{theme}")
    page.close()


@pytest.mark.parametrize("width,height", SIZES)
def test_tools_from_the_home_a_switch_writes_one_entry_and_a_stale_list_is_refused(subscription_ui, width, height):
    page, server = _page(subscription_ui, width, height)
    page.locator(f'{HOME} [data-settings-open="tools"]').click()
    level = _settled(page, "tools")
    slack = level.locator('[data-entry-id="slack"]')
    slack.wait_for(state="visible")
    page.wait_for_function("() => /38 tools · listed/.test(document.querySelector('#page-settings [data-entry-id=slack]')?.textContent || '')")
    # The head: the name, the status's words, Sign in because the status asks, the switch. No refresh was started.
    assert "Slack" in slack.inner_text() and "HTTP" in slack.inner_text()
    assert slack.locator("[data-entry-sign-in]").inner_text() == "Sign in"
    assert level.locator('[data-entry-id="github"] [data-entry-sign-in]').count() == 0
    assert "Listing failed" in level.locator('[data-entry-id="browser"]').inner_text()
    assert "Connections map ≈730 tokens in every Low/Max request" in level.locator("[data-mcp-global]").inner_text()
    assert not [url for url in subscription_ui["reads"] if "/api/mcp/refresh" in url or "/api/mcp/test" in url]

    # The switch of a saved head: one write of the list with one entry changed and the token of its read.
    level.locator('[data-entry-id="browser"] .ui-switch').click()
    page.wait_for_function("() => document.querySelector('#page-settings [data-entry-id=browser] [data-entry-switch]')?.checked === true"
                           " && !document.querySelector('#page-settings [aria-busy=\"true\"]')")
    [written] = server.writes()
    assert list(written) == ["MCP_SERVERS"]
    assert [(entry["id"], entry["enabled"]) for entry in written["MCP_SERVERS"]] == [("slack", True), ("github", True), ("browser", True)]
    assert server.posts[0][2] == {"MCP_SERVERS": "MCP_SERVERS#0"}
    assert level.locator("[data-entry-result]").first.inner_text().strip() == ""

    # Another device changed the list meanwhile: the write is refused, said, and the list shows what the server holds.
    server.stale_next = True
    level.locator('[data-entry-id="github"] .ui-switch').click()
    page.wait_for_function("() => /Changed on another device/.test(document.querySelector('#page-settings [data-entry-result]')?.textContent || '')")
    assert len(server.writes()) == 2, "one attempt, never a silent retry"
    assert level.locator('[data-entry-id="github"] [data-entry-switch]').is_checked(), "its own switch is back at the stored value"
    assert not level.locator('[data-entry-id="slack"] [data-entry-switch]').is_checked(), "the other device's change is shown"

    # The client's switch is a plain row that acts at once; the fold holds the two waits.
    assert commit_row_quiet(page, "MCP_ENABLED") == ""
    assert server.writes()[-1] == {"MCP_ENABLED": False}
    page.wait_for_function("() => /All tool servers are off/.test(document.querySelector('#page-settings [data-mcp-global]')?.textContent || '')")
    level.locator("details[data-settings-fold] > summary").click()
    assert commit_row(page, "MCP_TOOL_TIMEOUT_SEC", 90).startswith("Saved · was 60")
    page.close()


def _type_model(editor, model):
    """Type a model id and leave the field, as the owner does before Save: leaving closes the chooser's list."""
    field = editor.locator("[data-model-role-model]")
    field.fill(model)
    field.evaluate("element => element.blur()")


def commit_row_quiet(page, row_id):
    """Press a switch row that acts now and return its result line once the write settled (nothing, when it landed)."""
    row = setting_row(page, row_id)
    row.locator(".ui-switch").click()
    page.wait_for_function("id => { const row = document.querySelector(`#page-settings [data-setting-row=\"${id}\"]`);"
                           " return row && !row.querySelector('[aria-busy=\"true\"]'); }", arg=row_id)
    return row.locator("[data-setting-result]").inner_text().strip()


def test_a_tool_servers_page_saves_its_entry_keeps_typed_text_and_remove_asks_first(subscription_ui):
    page, server = _page(subscription_ui)
    open_settings(page, "tools")
    _settled(page, "tools").locator('[data-entry-id="github"]').wait_for(state="visible")
    editor = open_entry(page, "github")
    assert top_path(page) == "tools/github"
    assert editor.locator(".settings-page-title").inner_text() == "GitHub"
    assert editor.locator("[data-settings-back]").inner_text().strip() == "Tools"
    name = editor.locator('[data-mcp-field="name"]')
    name.fill("GitHub (work)")
    # A status read lands while the form holds typed text: the text stays, and so does the caret.
    page.evaluate("() => window.dispatchEvent(new CustomEvent('ouro:settings-page-shown', { detail: { path: 'tools/github' } }))")
    page.wait_for_function("() => document.querySelector('#page-settings [data-mcp-field=name]').value === 'GitHub (work)'")
    assert editor.locator("[data-editor-discard]").is_visible()
    assert press_editor_save(page) == "Saved"
    [written] = server.writes()
    assert [entry["name"] for entry in written["MCP_SERVERS"]] == ["Slack", "GitHub (work)", "Browser (Playwright)"]
    assert written["MCP_SERVERS"][1]["headers"] == {"X-API-Key": "***set***"}, "an untouched saved value is kept as its mask"
    assert written["MCP_SERVERS"][0] == SERVERS[0], "the other entries are the fresh read's"
    assert server.posts[0][2] == {"MCP_SERVERS": "MCP_SERVERS#0"}
    # A field its transport does not use is refused beside Save, before anything is sent.
    editor = settings_page(page, "tools/github")
    editor.locator('[data-mcp-field="transport"]').select_option("stdio")
    editor.locator("[data-editor-save]").click()
    page.wait_for_function("() => /not used by Local process/.test(document.querySelector('#page-settings [data-editor-result]')?.textContent || '')")
    assert len(server.writes()) == 1
    editor.locator("[data-editor-discard]").click()
    page.wait_for_function("() => document.querySelector('#page-settings [data-mcp-field=transport]').value === 'streamable_http'")
    # Remove asks first; declining keeps the server, accepting removes it and returns to the list.
    editor.locator("[data-editor-remove]").click()
    dialog = page.locator(".confirm-dialog-backdrop")
    dialog.wait_for(state="visible")
    assert "Remove GitHub (work)?" in dialog.inner_text()
    dialog.get_by_role("button", name="Cancel").click()
    dialog.wait_for(state="detached")
    assert len(server.writes()) == 1
    editor.locator("[data-editor-remove]").click()
    dialog.wait_for(state="visible")
    dialog.get_by_role("button", name="Remove").click()
    page.wait_for_function("() => /Removed GitHub/.test(document.querySelector('#page-settings [data-settings-page=tools] [data-entry-result]')?.textContent || '')")
    assert top_path(page) == "tools"
    assert [entry["id"] for entry in server.writes()[-1]["MCP_SERVERS"]] == ["slack", "browser"]
    # Add opens an unsaved server; its Save appends it, off, and the list says so.
    settings_page(page, "tools").locator("[data-tools-add]").click()
    new = settings_page(page, "tools/new")
    new.locator('[data-mcp-field="name"]').fill("Docs")
    new.locator('[data-mcp-field="url"]').fill("https://docs.example.com/mcp")
    new.locator("[data-editor-save]").click()
    page.wait_for_function("() => /Added Docs/.test(document.querySelector('#page-settings [data-settings-page=tools] [data-entry-result]')?.textContent || '')")
    assert [(entry["id"], entry["enabled"]) for entry in server.writes()[-1]["MCP_SERVERS"]] == [("slack", True), ("browser", False), ("docs", False)]
    page.close()


@pytest.mark.parametrize("width,height", SIZES)
def test_models_from_search_a_role_saves_through_its_writer_and_the_context_door_posts_nothing(subscription_ui, width, height):
    page, server = _page(subscription_ui, width, height)
    subscription_ui["reads"].clear()
    page.locator(f"{HOME} [data-settings-card]").wait_for(state="visible")
    assert not [url for url in subscription_ui["reads"] if "/api/model-catalog" in url], "the home reads no catalogue"
    assert "OUROBOROS_PROMPT_CACHE_TTL" in search_settings(page, "prompt cache")
    page.locator(f'{HOME} [data-settings-result="OUROBOROS_PROMPT_CACHE_TTL"]').click()
    level = _settled(page, "models")
    page.wait_for_function("() => document.querySelector('#page-settings [data-setting-row=\"OUROBOROS_PROMPT_CACHE_TTL\"]')?.closest('details')?.open")
    assert [url for url in subscription_ui["reads"] if "/api/model-catalog" in url], "the catalogue is read when Models is shown"
    # The roles in the picture's order, each with its route.
    page.wait_for_function("() => document.querySelector('#page-settings [data-setting-row=\"models:light\"] [data-row-value]')?.textContent")
    assert level.locator("[data-entry-id]").evaluate_all("rows => rows.map((row) => row.dataset.entryId)") == [
        "main", "fallback/1", "fallback/2", "fallback/3", "light", "vision", "consciousness", "safety", "websearch"]
    assert "gpt-test" in level.locator('[data-entry-id="main"]').inner_text()
    values = level.locator("[data-row-value]").evaluate_all("spans => spans.map((span) => span.textContent)")
    assert values[:6] == ["gpt-light", "Main model", "Main model", "Light model", "Auto", "Max"]

    # The Context door shows the install's value, leads to Main and writes nothing.
    level.locator('[data-setting-row="models:context"]').click()
    page.wait_for_function("() => document.querySelector('#page-chat')?.classList.contains('active')")
    assert not server.posts and not [path for path, _body in subscription_ui["posts"] if "context-mode" in path]

    # Vision gets a model of its own: its route is a plain key, written by the generic save alone.
    open_settings(page, "models/role/vision")
    editor = _settled(page, "models/role/vision")
    editor.locator("[data-editor-save]").wait_for(state="visible")
    editor.locator("[data-model-role-source]").select_option("api:openai")
    _type_model(editor, "gpt-vision")
    assert press_editor_save(page).startswith("Saved · applies from the next model call")
    assert server.posts[-1][:2] == ("/api/settings", {"OUROBOROS_MODEL_VISION": "openai::gpt-vision"})
    assert server.posts[-1][2] == {"OUROBOROS_MODEL_VISION": "OUROBOROS_MODEL_VISION#0"}

    # Light is the checker's: its route goes through the owner's checker transaction, never the generic save.
    open_settings(page, "models/role/light")
    editor = _settled(page, "models/role/light")
    _type_model(editor, "gpt-lighter")
    before = len(server.posts)
    assert press_editor_save(page).startswith("Saved")
    sent = server.posts[before:]
    assert [(path, sorted(settings)) for path, settings, _held in sent] == [("/api/owner/safety-model", ["OUROBOROS_MODEL_LIGHT"])]
    assert not [key for path, settings, _held in server.posts if path == "/api/settings" for key in settings if key in CHECKER_KEYS]

    # The fallbacks' order is changed on the list: the order by the generic save, their windows by the checker.
    open_settings(page, "models")
    level = _settled(page, "models")
    level.locator('[data-entry-id="fallback/2"] [data-fallback-move="-1"]').click()
    page.wait_for_function("() => /Order saved/.test(document.querySelector('#page-settings [data-settings-page=models] [data-entry-result]')?.textContent || '')")
    assert server.values["OUROBOROS_MODEL_FALLBACKS"] == "moonshotai/kimi-k3-pro, anthropic::claude-opus-test, openai::gpt-sol"
    assert server.values["OUROBOROS_MODEL_CONTEXT_WINDOWS"] == {"fallback": [128000, 0, 0]}
    # Remove asks first.
    level.locator('[data-entry-id="fallback/3"] [data-fallback-remove]').click()
    dialog = page.locator(".confirm-dialog-backdrop")
    dialog.wait_for(state="visible")
    dialog.get_by_role("button", name="Remove").click()
    page.wait_for_function("() => /^Removed/.test(document.querySelector('#page-settings [data-settings-page=models] [data-entry-result]')?.textContent || '')")
    assert server.values["OUROBOROS_MODEL_FALLBACKS"] == "moonshotai/kimi-k3-pro, anthropic::claude-opus-test"
    page.close()


def test_a_role_left_with_unsaved_words_comes_back_as_it_was_left_and_a_saved_one_is_read_again(subscription_ui):
    """Back from an editor with unsaved words puts its page aside: its path opens it again as it was left, and Save
    then writes those words. Around the typed role the page follows the read: a key saved in Accounts meanwhile is
    offered. With nothing unsaved the page is let go and shows what is stored the next time."""
    from playwright.sync_api import expect

    page, server = _page(subscription_ui, 1087, 740, ANTHROPIC_API_KEY="")
    open_settings(page, "models/role/vision")
    editor = _settled(page, "models/role/vision")
    editor.locator("[data-editor-save]").wait_for(state="visible")
    assert editor.locator('[data-model-role-source] option[value="api:anthropic"]').count() == 0, "no Anthropic key yet"
    editor.locator("[data-model-role-source]").select_option("api:openai")
    _type_model(editor, "gpt-vision")
    go_back(page)
    assert not server.posts, "leaving wrote nothing"
    server.values["ANTHROPIC_API_KEY"] = "***set***"  # saved in Accounts and keys while the role waited aside
    open_settings(page, "models/role/vision")
    editor = _settled(page, "models/role/vision")
    expect(editor.locator('[data-model-role-source] option[value="api:anthropic"]')).to_have_count(1)
    assert editor.locator("[data-model-role-source]").input_value() == "api:openai"
    assert editor.locator("[data-model-role-model]").input_value() == "gpt-vision"
    assert editor.locator("[data-editor-discard]").is_visible()
    assert press_editor_save(page).startswith("Saved")
    assert server.posts[-1][:2] == ("/api/settings", {"OUROBOROS_MODEL_VISION": "openai::gpt-vision"})
    # Saved, the page holds nothing of its own: left and opened again, it reads what is stored now.
    go_back(page)
    server.values["OUROBOROS_MODEL_VISION"] = "openai::gpt-other"
    open_settings(page, "models/role/vision")
    editor = _settled(page, "models/role/vision")
    page.wait_for_function("() => document.querySelector('#page-settings [data-settings-page=\"models/role/vision\"] "
                           "[data-model-role-model]')?.value === 'gpt-other'")
    assert not editor.locator("[data-editor-discard]").is_visible()
    page.close()


def test_a_kept_role_saves_only_what_was_changed_and_a_model_chosen_elsewhere_meanwhile_stays(subscription_ui):
    """A role's page kept with one changed field writes that field and nothing it left as read: a model saved
    elsewhere while the page waited (Main from the composer, another device) stays."""
    from playwright.sync_api import expect

    page, server = _page(subscription_ui, 1087, 740, OUROBOROS_MODEL_VISION="openai::gpt-vision-old")
    open_settings(page, "models/role/vision")
    editor = _settled(page, "models/role/vision")
    expect(editor.locator("[data-model-role-model]")).to_have_value("gpt-vision-old")
    editor.locator(".model-role-details > summary").click()  # processing and window wait behind More
    editor.locator("[data-model-role-processing]").select_option("fast")
    go_back(page)
    server.values["OUROBOROS_MODEL_VISION"] = "openai::gpt-vision-new"  # chosen elsewhere while the page waited aside
    open_settings(page, "models/role/vision")
    editor = _settled(page, "models/role/vision")
    editor.locator(".model-role-details").evaluate("node => { node.open = true; }")
    assert editor.locator("[data-model-role-processing]").input_value() == "fast", "the typed field stays"
    assert press_editor_save(page).startswith("Saved")
    path, written, _held = server.posts[-1]
    assert "OUROBOROS_MODEL_VISION" not in written, written
    assert written.get("OUROBOROS_MODEL_PROCESSING_PREFERENCES", {}).get("vision") == "fast", written
    assert server.values["OUROBOROS_MODEL_VISION"] == "openai::gpt-vision-new"
    page.close()


def test_the_local_model_page_saves_a_row_for_its_own_restart_and_never_offers_a_restart_of_ouroboros(subscription_ui):
    page, server = _page(subscription_ui)
    open_settings(page, "models")
    _settled(page, "models").locator('[data-setting-row="models:local"]').click()
    level = _settled(page, "models/local")
    assert level.locator("[data-settings-back]").inner_text().strip() == "Models"
    page.wait_for_function("() => /Ready/.test(document.querySelector('#local-model-status')?.textContent || '')")
    assert commit_row(page, "LOCAL_MODEL_CONTEXT_LENGTH", 32768) == "Saved · applies when the local model is restarted · was 16384 · Undo"
    assert server.writes() == [{"LOCAL_MODEL_CONTEXT_LENGTH": 32768}]
    words = level.inner_text()
    assert "Restart" not in words and "restart of Ouroboros" not in words
    assert [button.strip() for button in level.locator(".ui-action-row .btn:visible").all_inner_texts()] == ["Start", "Stop", "Test tool calling"]
    assert go_back(page) == "models"
    page.close()


@pytest.mark.parametrize('width', [1087, 320])
def test_every_role_editor_keeps_vendor_program_inside_open_chooser(subscription_ui, width):
    from playwright.sync_api import expect
    from tests.test_model_chooser_browser import vendor_scene

    ui, page = subscription_ui, subscription_ui['page']
    world = vendor_scene(ui, 18)
    ui['settings']['OUROBOROS_MODEL_FALLBACKS'] = 'claudexor::opaque-source=choice-old'
    page.set_viewport_size({'width': width, 'height': 640})
    page.goto(ui['url'] + '/#settings')
    for slot in ['main', 'light', 'safety', 'vision', 'consciousness', 'fallback/1']:
        level = open_settings(page, 'models/role/' + slot)
        source = level.locator('[data-model-role-source]')
        expect(source.locator('option[value="subscription:opaque-source"]')).to_have_count(1)
        source.select_option('subscription:opaque-source')
        field = level.locator('[data-model-role-model]')
        field.fill('choice')
        popup = page.locator('[id="' + field.get_attribute('aria-controls') + '"]')
        popup = popup.locator('..')
        footer = popup.locator('[data-model-footer]')
        expect(footer).to_be_visible()
        expect(footer).to_contain_text('List from Codex 2.0.0')
        expect(footer.get_by_role('button', name='Update', exact=True)).to_be_enabled()
        assert footer.locator('[role="option"]').count() == 0
        assert popup.evaluate('e => e.lastElementChild.matches("[data-model-footer]")')
        field.press('Escape')
        expect(footer).to_be_hidden()
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
    assert not world['posts']
