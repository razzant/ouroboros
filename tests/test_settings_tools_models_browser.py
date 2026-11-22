"""Extensions › MCP servers and Settings › Models in a browser (docs/DESIGN.md "Settings"; web/modules/mcp_settings.js,
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
from copy import deepcopy
from datetime import datetime, timezone
from urllib.parse import urlparse

import pytest

from tests import test_subscription_setup_browser as setup_browser
from tests.settings_browser_helpers import (
    HOME, commit_row, go_back, open_entry, open_settings, press_editor_save, search_settings, setting_row, settings_page, top_path)
from tests.settings_section_helpers import header_back, lead_name, show_card, show_page

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
            "OPENAI_API_KEY": "***set***", "ANTHROPIC_API_KEY": "***set***", "OPENROUTER_API_KEY": "***set***",
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
    page.wait_for_function("() => !document.querySelector('.page.active [aria-busy=\"true\"]')")
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
            page.wait_for_function("() => /tools · listed/.test(document.querySelector('.page.active [data-entry-id=slack]')?.textContent || '')")
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
    # MCP servers is a page of Extensions: Settings' one door leads there, and the list's row opens it.
    page.locator(f'{HOME} [data-settings-open="page:skills"]').click()
    page.locator('#page-skills [data-extensions-open="tools"]').click()
    level = _settled(page, "tools")
    slack = level.locator('[data-entry-id="slack"]')
    slack.wait_for(state="visible")
    page.wait_for_function("() => /38 tools · listed/.test(document.querySelector('.page.active [data-entry-id=slack]')?.textContent || '')")
    # The head: the name, the status's words, Sign in because the status asks, the switch. No refresh was started.
    assert "Slack" in slack.inner_text() and "HTTP" in slack.inner_text()
    assert slack.locator("[data-entry-sign-in]").inner_text() == "Sign in"
    assert level.locator('[data-entry-id="github"] [data-entry-sign-in]').count() == 0
    assert "Listing failed" in level.locator('[data-entry-id="browser"]').inner_text()
    assert "Connections map ≈730 tokens in every Low/Max request" in level.locator("[data-mcp-global]").inner_text()
    assert not [url for url in subscription_ui["reads"] if "/api/mcp/refresh" in url or "/api/mcp/test" in url]

    # The switch of a saved head: one write of the list with one entry changed and the token of its read.
    level.locator('[data-entry-id="browser"] .ui-switch').click()
    page.wait_for_function("() => document.querySelector('.page.active [data-entry-id=browser] [data-entry-switch]')?.checked === true"
                           " && !document.querySelector('.page.active [aria-busy=\"true\"]')")
    [written] = server.writes()
    assert list(written) == ["MCP_SERVERS"]
    assert [(entry["id"], entry["enabled"]) for entry in written["MCP_SERVERS"]] == [("slack", True), ("github", True), ("browser", True)]
    assert server.posts[0][2] == {"MCP_SERVERS": "MCP_SERVERS#0"}
    assert level.locator("[data-entry-result]").first.inner_text().strip() == ""

    # Another device changed the list meanwhile: the write is refused, said, and the list shows what the server holds.
    server.stale_next = True
    level.locator('[data-entry-id="github"] .ui-switch').click()
    page.wait_for_function("() => /Changed on another device/.test(document.querySelector('#page-skills [data-entry-result]')?.textContent || '')")
    assert len(server.writes()) == 2, "one attempt, never a silent retry"
    assert level.locator('[data-entry-id="github"] [data-entry-switch]').is_checked(), "its own switch is back at the stored value"
    assert not level.locator('[data-entry-id="slack"] [data-entry-switch]').is_checked(), "the other device's change is shown"

    # The client's switch is a plain row that acts at once; the fold holds the two waits.
    assert commit_row_quiet(page, "MCP_ENABLED") == ""
    assert server.writes()[-1] == {"MCP_ENABLED": False}
    page.wait_for_function("() => /All MCP servers are off/.test(document.querySelector('#page-skills [data-mcp-global]')?.textContent || '')")
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
    page.wait_for_function("id => { const row = document.querySelector(`.page.active [data-setting-row=\"${id}\"]`);"
                           " return row && !row.querySelector('[aria-busy=\"true\"]'); }", arg=row_id)
    return row.locator("[data-setting-result]").inner_text().strip()


def test_a_tool_servers_page_saves_its_entry_keeps_typed_text_and_remove_asks_first(subscription_ui):
    page, server = _page(subscription_ui)
    open_settings(page, "tools")
    _settled(page, "tools").locator('[data-entry-id="github"]').wait_for(state="visible")
    editor = open_entry(page, "github")
    assert top_path(page) == "tools/github"
    # The header names the server's page, and its Back leads to the list it was opened from.
    assert lead_name(page) == "GitHub"
    back = header_back(page)
    assert back.is_visible() and back.get_attribute("aria-label") == back.get_attribute("title") == "Back to MCP servers"
    back.click()
    settings_page(page, "tools").wait_for(state="visible")
    assert top_path(page) == "tools", "Back opens the list it names"
    editor = open_entry(page, "github")
    name = editor.locator('[data-mcp-field="name"]')
    name.fill("GitHub (work)")
    # A status read lands while the form holds typed text: the text stays, and so does the caret.
    page.evaluate("() => window.dispatchEvent(new CustomEvent('ouro:settings-page-shown', { detail: { path: 'tools/github' } }))")
    page.wait_for_function("() => document.querySelector('#page-skills [data-mcp-field=name]').value === 'GitHub (work)'")
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
    page.wait_for_function("() => /not used by Local process/.test(document.querySelector('#page-skills [data-editor-result]')?.textContent || '')")
    assert len(server.writes()) == 1
    editor.locator("[data-editor-discard]").click()
    page.wait_for_function("() => document.querySelector('#page-skills [data-mcp-field=transport]').value === 'streamable_http'")
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
    page.wait_for_function("() => /Removed GitHub/.test(document.querySelector('#page-skills [data-settings-page=tools] [data-entry-result]')?.textContent || '')")
    assert top_path(page) == "tools"
    assert [entry["id"] for entry in server.writes()[-1]["MCP_SERVERS"]] == ["slack", "browser"]
    # Add opens an unsaved server; its Save appends it, off, and the list says so.
    settings_page(page, "tools").locator("[data-tools-add]").click()
    new = settings_page(page, "tools/new")
    new.locator('[data-mcp-field="name"]').fill("Docs")
    new.locator('[data-mcp-field="url"]').fill("https://docs.example.com/mcp")
    new.locator("[data-editor-save]").click()
    page.wait_for_function("() => /Added Docs/.test(document.querySelector('#page-skills [data-settings-page=tools] [data-entry-result]')?.textContent || '')")
    assert [(entry["id"], entry["enabled"]) for entry in server.writes()[-1]["MCP_SERVERS"]] == [("slack", True), ("browser", False), ("docs", False)]
    page.close()


@pytest.mark.parametrize("new", [False, True], ids=["existing", "new"])
@pytest.mark.parametrize("edit_header", [False, True], ids=["address", "header-name"])
def test_mcp_address_typed_after_save_belongs_to_the_next_save(subscription_ui, new, edit_header):
    from playwright.sync_api import expect

    first, second = "https://first.example.test/mcp", "https://second.example.test/mcp"
    existing = {"id": "notes", "name": "Notes", "url": "https://notes.example.test/mcp", "enabled": False}
    page, server = _page(subscription_ui, MCP_SERVERS=[] if new else [existing])
    editor = open_settings(page, "tools/new" if new else "tools/notes")
    if new:
        editor.locator('[data-mcp-field="name"]').fill("Notes")
    address = '[data-mcp-field="url"]'
    editor.locator(address).fill(first)
    pending = []
    page.route("**/api/settings", lambda route: pending.append(route) if route.request.method == "POST" else route.fallback())
    with page.expect_request(lambda req: req.method == "POST" and req.url.endswith("/api/settings")):
        editor.locator("[data-editor-save]").click()
    editor.locator(address).fill(second)
    field, start = address, len("https://")
    if edit_header:
        editor.locator("[data-mcp-more] > summary").click()
        editor.locator("[data-mcp-header-add]").click()
        field, start = '[data-mcp-header-field="name"]', len("X-")
        editor.locator(field).fill("X-second")
    editor.locator(field).evaluate("(el, start) => el.setSelectionRange(start, start + 6, 'backward')", start)
    server.answer(pending.pop())

    saved_page = settings_page(page, "tools/notes")
    expect(saved_page.locator("[data-editor-result]")).to_contain_text("unsaved")
    expect(saved_page.locator(address)).to_have_value(second)
    expect(saved_page.locator("[data-editor-discard]")).to_be_visible()
    assert len(server.writes()) == 1 and server.values["MCP_SERVERS"][0]["url"] == first
    current = saved_page.locator(field)
    expect(current).to_be_focused()
    assert current.evaluate("el => [el.selectionStart, el.selectionEnd, el.selectionDirection]") == [start, start + 6, "backward"]
    page.keyboard.insert_text("final")
    final_url = second if edit_header else second.replace("second", "final")
    expect(current).to_have_value("X-final" if edit_header else final_url)
    page.unroute("**/api/settings")
    assert press_editor_save(page) == "Saved"
    assert [(row["id"], row["url"]) for row in server.values["MCP_SERVERS"]] == [("notes", final_url)]
    if edit_header:
        assert server.values["MCP_SERVERS"][0]["headers"] == {"X-final": ""}
    assert [held for _, _, held in server.posts] == [{"MCP_SERVERS": f"MCP_SERVERS#{i}"} for i in (0, 1)]
    page.close()


@pytest.mark.parametrize("external", ["neighbour", "same-entry", "masked-value", "overlapping-header", "removed"])
def test_mcp_draft_after_save_and_another_list_write(subscription_ui, external):
    from playwright.sync_api import expect
    from ouroboros.gateway.secret_rows import mask_mcp_servers, rehydrate_mcp_servers

    rows = [dict(id=key, name=key, url=f"https://{key}.example.test") for key in ("notes", "other")]
    page, server = _page(subscription_ui, MCP_SERVERS=rows)
    pending = []

    def settings(route):
        if route.request.method == "POST":
            if not server.posts:
                pending.append(route)
            else:
                previous = server.values["MCP_SERVERS"]
                server.answer(route)
                server.values["MCP_SERVERS"] = rehydrate_mcp_servers(server.values["MCP_SERVERS"], previous)
            return
        if server.versions.get("MCP_SERVERS") == 1:
            # Another writer commits after our receipt but before the first readback.
            saved = server.values["MCP_SERVERS"] = deepcopy(server.values["MCP_SERVERS"])
            saved[1]["name"] = "Other updated"
            if external == "same-entry":
                saved[0]["name"] = "Notes renamed elsewhere"
            elif external in ("masked-value", "overlapping-header"):
                saved[0]["headers"]["X-Mode"] = "external-fixture"
            elif external == "removed":
                saved.pop(0)
            server.versions["MCP_SERVERS"] = 2
        document = server.document()
        document["MCP_SERVERS"] = mask_mcp_servers(document["MCP_SERVERS"])
        route.fulfill(json=document)

    page.route("**/api/settings", settings)
    editor = open_settings(page, "tools/notes")
    page.wait_for_load_state("networkidle")
    editor.locator("[data-mcp-more] > summary").click()
    editor.locator("[data-mcp-header-add]").click()
    editor.locator('[data-mcp-header-field="name"]').fill("X-Mode")
    editor.locator('[data-mcp-header-field="value"]').fill("submitted-fixture")
    save = editor.locator("[data-editor-save]")
    with page.expect_request(lambda req: req.method == "POST" and req.url.endswith("/api/settings")):
        save.click()
    address = editor.locator('[data-mcp-field="url"]')
    latest = "https://next.example.test"
    address.fill(latest)
    if external == "overlapping-header":
        editor.locator('[data-mcp-header-field="value"]').fill("next-fixture")
    server.answer(pending.pop())
    result = editor.locator("[data-editor-result]")
    expect(result).to_contain_text("latest edits are unsaved")
    expect(address).to_have_value(latest)
    expect(save).to_be_enabled()
    if external in ("same-entry", "overlapping-header", "removed"):
        expect(result).to_contain_text("could not be confirmed")
        if external == "overlapping-header":
            field = editor.locator('[data-mcp-header-field="value"]')
            show = editor.locator('[data-mcp-header-toggle]')
            expect(field).to_have_value("next-fixture")
            expect(field).to_have_attribute("type", "password")
            expect(show).to_be_disabled()
            expect(show).to_have_attribute("aria-expanded", "false")
        save.click()
        expect(result).to_contain_text("Discard changes")
        editor.locator("[data-editor-discard]").click()
        if external == "removed":
            expect(editor).to_contain_text("no longer on the list")
        elif external == "same-entry":
            expect(editor.locator('[data-mcp-field="name"]')).to_have_value("Notes renamed elsewhere")
        else:
            expect(address).to_have_value("https://notes.example.test")
            assert server.values["MCP_SERVERS"][0]["headers"] == {"X-Mode": "external-fixture"}
        assert len(server.writes()) == 1
    else:
        expect(editor.locator('[data-mcp-header-field="value"]')).to_have_value("***set***")
        assert press_editor_save(page) == "Saved"
        notes, other = server.values["MCP_SERVERS"]
        assert notes["url"] == latest and other["name"] == "Other updated"
        assert notes["headers"] == {"X-Mode": "external-fixture" if external == "masked-value" else "submitted-fixture"}
        assert server.posts[-1][2] == {"MCP_SERVERS": "MCP_SERVERS#2"}
    page.close()


def test_mcp_header_stored_by_one_save_is_kept_by_the_next(subscription_ui):
    """One Save stores X-A while X-B is typed; another window then replaces X-A. The next Save keeps X-A's mask."""
    from playwright.sync_api import expect
    from ouroboros.gateway.secret_rows import mask_mcp_servers, rehydrate_mcp_servers

    notes = {"id": "notes", "name": "Notes", "url": "https://notes.example.test/mcp",
             "headers": {"X-A": "stored-a-fixture", "X-B": "stored-b-fixture"}}
    page, server = _page(subscription_ui, MCP_SERVERS=[notes])
    pending = []

    def commit(route):
        previous = server.values["MCP_SERVERS"]
        server.answer(route)
        server.values["MCP_SERVERS"] = rehydrate_mcp_servers(server.values["MCP_SERVERS"], previous)

    def settings(route):
        if route.request.method == "POST":
            if server.posts or pending:
                commit(route)
            else:
                pending.append(route)  # the first Save waits until the other row is typed
            return
        document = server.document()
        document["MCP_SERVERS"] = mask_mcp_servers(document["MCP_SERVERS"])
        route.fulfill(json=document)

    page.route("**/api/settings", settings)
    editor = open_settings(page, "tools/notes")
    page.wait_for_load_state("networkidle")
    editor.locator("[data-mcp-more] > summary").click()
    values = editor.locator('[data-mcp-header-field="value"]')
    values.nth(0).fill("submitted-a-fixture")
    with page.expect_request(lambda req: req.method == "POST" and req.url.endswith("/api/settings")):
        editor.locator("[data-editor-save]").click()
    values.nth(1).fill("next-b-fixture")
    commit(pending.pop())
    expect(editor.locator("[data-editor-result]")).to_contain_text("latest edits are unsaved")
    expect(values.nth(0)).to_have_value("***set***")
    expect(values.nth(1)).to_have_value("next-b-fixture")
    expect(values.nth(1)).to_be_focused()
    assert server.values["MCP_SERVERS"][0]["headers"] == {"X-A": "submitted-a-fixture", "X-B": "stored-b-fixture"}

    server.values["MCP_SERVERS"][0]["headers"]["X-A"] = "other-window-fixture"
    server.versions["MCP_SERVERS"] += 1
    save = editor.locator("[data-editor-save]")
    dialog = page.locator(".confirm-dialog-backdrop")
    save.click()
    expect(dialog).to_contain_text("A masked value you edited or removed may have changed elsewhere")
    dialog.get_by_role("button", name="Cancel").click()
    expect(values.nth(1)).to_have_value("next-b-fixture")
    assert len(server.writes()) == 1
    save.click()
    dialog.get_by_role("button", name="Save changes", exact=True).click()
    expect(editor.locator("[data-editor-result]")).to_have_text("Saved")
    assert len(server.writes()) == 2 and server.posts[-1][2] == {"MCP_SERVERS": "MCP_SERVERS#2"}
    assert server.writes()[-1]["MCP_SERVERS"][0]["headers"] == {"X-A": "***set***", "X-B": "next-b-fixture"}
    assert server.values["MCP_SERVERS"][0]["headers"] == {"X-A": "other-window-fixture", "X-B": "next-b-fixture"}
    page.close()


# Who asked for each catalogue read, recorded as it starts: its stack names the module that read (the home's own reads
# start in settings_home.js), since another page may read the catalogue for itself.
_RECORD_CATALOGUE_READERS = """() => {
    const fetch = window.fetch;
    window.__catalogueReaders = [];
    window.fetch = (url, init) => {
        if (String(url).split('?')[0].endsWith('/api/model-catalog')) window.__catalogueReaders.push(String(new Error().stack));
        return fetch(url, init);
    };
}"""


@pytest.mark.parametrize("width,height", SIZES)
def test_models_from_search_a_role_saves_through_its_writer_and_the_context_door_posts_nothing(subscription_ui, width, height):
    page, server = _page(subscription_ui, width, height)
    # The list with Ouroboros's card (beside a page, the card is the pane its row opens). The list reads its answers when
    # it comes on screen (settings.js look): Settings shown again after another page reads every one of them, and none
    # of the home's reads is the catalogue.
    show_card(page)
    page.evaluate(_RECORD_CATALOGUE_READERS)
    show_page(page, "skills")
    subscription_ui["reads"].clear()
    show_page(page, "settings")
    for _ in range(200):  # its storage read, one of the answers it reads each time, has reached the server
        if any(urlparse(url).path == "/api/storage" for url in subscription_ui["reads"]):
            break
        page.wait_for_timeout(50)
    else:
        raise AssertionError(("the home did not read its answers again", subscription_ui["reads"]))
    page.wait_for_function("() => document.querySelector('#page-settings [data-settings-open=rules] [data-settings-answer]')?.textContent")
    readers = page.evaluate("() => window.__catalogueReaders")
    assert not [stack for stack in readers if "settings_home.js" in stack], ("the home reads no catalogue", readers)
    subscription_ui["reads"].clear()  # from here a catalogue read is the one Models makes when it is shown
    assert "OUROBOROS_PROMPT_CACHE_TTL" in search_settings(page, "prompt cache")
    page.locator(f'{HOME} [data-settings-result="OUROBOROS_PROMPT_CACHE_TTL"]').click()
    level = _settled(page, "models")
    page.wait_for_function("() => document.querySelector('#page-settings [data-setting-row=\"OUROBOROS_PROMPT_CACHE_TTL\"]')?.closest('details')?.open")
    assert not [url for url in subscription_ui["reads"] if url.endswith("/api/model-catalog")], "navigating to Models does not read every provider"
    # The roles in the picture's order, each with its route.
    page.wait_for_function("() => document.querySelector('#page-settings [data-setting-row=\"models:light\"] [data-row-value]')?.textContent")
    assert level.locator("[data-entry-id]").evaluate_all("rows => rows.map((row) => row.dataset.entryId)") == [
        "main", "fallback/1", "fallback/2", "fallback/3", "light", "vision", "consciousness", "safety", "websearch"]
    assert "gpt-test" in level.locator('[data-entry-id="main"]').inner_text()
    values = level.locator("[data-row-value]").evaluate_all("spans => spans.map((span) => span.textContent)")
    # An empty Safety Supervisor names the route it checks with now (Light's model) and whose it is.
    assert values[:6] == ["gpt-light", "Main model", "Main model", "gpt-light · as Light", "Auto", "Max"]

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


def _open_role(page, role):
    """A role's page as the owner opens it: from Models, so it lies over that page and its Back leads there (beside
    the list a first-level page has no Back: the list is its way back, docs/DESIGN.md "Settings")."""
    open_settings(page, "models")
    _settled(page, "models").locator(f'[data-entry-id="{role}"]').click()  # the role's row is its door
    settings_page(page, f"models/role/{role}").wait_for(state="visible")
    back = header_back(page)
    assert back.is_visible() and back.get_attribute("aria-label") == "Back to Models"
    return _settled(page, f"models/role/{role}")


def test_a_role_left_with_unsaved_words_comes_back_as_it_was_left_and_a_saved_one_is_read_again(subscription_ui):
    """Back from an editor with unsaved words puts its page aside: its path opens it again as it was left (from
    outside Settings too, as a first-level page beside the list), its row in Models does too, and Save then writes
    those words. Around the typed role the page follows the read: a key saved in Accounts meanwhile is offered. With
    nothing unsaved the page is let go and shows what is stored the next time."""
    from playwright.sync_api import expect

    page, server = _page(subscription_ui, 1087, 740, ANTHROPIC_API_KEY="")
    editor = _open_role(page, "vision")
    editor.locator("[data-editor-save]").wait_for(state="visible")
    assert editor.locator('[data-model-role-source] option[value="api:anthropic"]').count() == 0, "no Anthropic key yet"
    editor.locator("[data-model-role-source]").select_option("api:openai")
    _type_model(editor, "gpt-vision")
    go_back(page)
    assert not server.posts, "leaving wrote nothing"
    # Its path, asked from outside Settings (a deep link), brings the page back as it was left; another page opened
    # beside the list puts it aside again.
    open_settings(page, "models/role/vision")
    editor = _settled(page, "models/role/vision")
    assert editor.locator("[data-model-role-source]").input_value() == "api:openai"
    assert editor.locator("[data-model-role-model]").input_value() == "gpt-vision"
    assert editor.locator("[data-editor-discard]").is_visible()
    open_settings(page, "models")
    assert not server.posts, "leaving wrote nothing"
    server.values["ANTHROPIC_API_KEY"] = "***set***"  # saved in Accounts and keys while the role waited aside
    editor = _open_role(page, "vision")
    expect(editor.locator('[data-model-role-source] option[value="api:anthropic"]')).to_have_count(1)
    assert editor.locator("[data-model-role-source]").input_value() == "api:openai"
    assert editor.locator("[data-model-role-model]").input_value() == "gpt-vision"
    assert editor.locator("[data-editor-discard]").is_visible()
    assert press_editor_save(page).startswith("Saved")
    assert server.posts[-1][:2] == ("/api/settings", {"OUROBOROS_MODEL_VISION": "openai::gpt-vision"})
    # Saved, the page holds nothing of its own: left and opened again, it reads what is stored now.
    go_back(page)
    server.values["OUROBOROS_MODEL_VISION"] = "openai::gpt-other"
    editor = _open_role(page, "vision")
    page.wait_for_function("() => document.querySelector('#page-settings [data-settings-page=\"models/role/vision\"] "
                           "[data-model-role-model]')?.value === 'gpt-other'")
    assert not editor.locator("[data-editor-discard]").is_visible()
    page.close()


def test_a_kept_role_saves_only_what_was_changed_and_a_model_chosen_elsewhere_meanwhile_stays(subscription_ui):
    """A role's page kept with one changed field writes that field and nothing it left as read: a model saved
    elsewhere while the page waited (Main from the composer, another device) stays."""
    from playwright.sync_api import expect

    page, server = _page(subscription_ui, 1087, 740, OUROBOROS_MODEL_VISION="openai::gpt-vision-old")
    editor = _open_role(page, "vision")
    expect(editor.locator("[data-model-role-model]")).to_have_value("gpt-vision-old")
    editor.locator(".model-role-details > summary").click()  # processing and window wait behind More
    editor.locator("[data-model-role-processing]").select_option("fast")
    go_back(page)
    server.values["OUROBOROS_MODEL_VISION"] = "openai::gpt-vision-new"  # chosen elsewhere while the page waited aside
    editor = _open_role(page, "vision")
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
    back = header_back(page)
    assert back.is_visible() and back.get_attribute("aria-label") == back.get_attribute("title") == "Back to Models"
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


# A window a size refusal set, in context_view_facts.window_projection's shape: made up here, no provider was asked.
REFUSED_WINDOW = {"tokens": 883_899, "kind": "input", "reserve": 65_536, "source": "refusal",
                  "observed_at": "2026-11-15T12:00:00+00:00", "contradiction": None, "retired_ack": {},
                  "usable": 883_899, "admitted": 883_899}


@pytest.mark.parametrize("width,height", SIZES)
def test_a_window_a_refusal_set_says_how_its_number_was_known(subscription_ui, width, height):
    """A role's More: the window a size refusal set says whether the provider stated its number, a local instance
    counted it exactly before sending, Ouroboros derived it from a refusal that stated none, or an earlier refusal kept
    no basis; a projection naming none (a server from before bases were kept) claims none. Each opening of the fold
    reads the route again, the page fits in both themes and nothing is written."""
    from playwright.sync_api import expect

    shown = {}
    subscription_ui["backend"]["response_preview"] = lambda query: {
        "route": {"provider": "claudexor", "model": query["model"][0], "base_url": "", "use_local": False, "options": None},
        "response_limit": {"max_output_tokens": 0, "status": "unknown", "source": "none", "observed_at": "",
                           "stale": False, "route_fp": "fixture"},
        "window": {**REFUSED_WINDOW, **shown}, "window_route": {}, "window_ack": False}
    page, server = _page(subscription_ui, width, height)
    open_settings(page, "models/role/main")
    editor = _settled(page, "models/role/main")
    fold, note = editor.locator(".model-role-details"), editor.locator("[data-model-context-note]")
    for basis, words in [(None, "from a refusal, basis unknown"),
                         ("unrecorded", "from an earlier refusal, basis not recorded"),
                         ("provider_stated", "stated by the provider in a refusal"),
                         ("anchored_estimate", "derived from a refusal that stated no number"),
                         ("exact_local_count", "derived from an exact local count before sending")]:
        shown.clear()
        shown.update({"basis": basis} if basis else {})
        if fold.evaluate("node => node.open"):
            fold.locator(":scope > summary").click()
        fold.locator(":scope > summary").click()  # opening it reads the route's window again
        expect(note).to_have_text(f"Window 884k tokens ({words}) · 884k usable for input")
        note.scroll_into_view_if_needed()
        assert editor.evaluate(FITS) == {"page": 0, "doc": 0, "out": []}, (basis, width)
        setup_browser.capture(page, f"refusal-basis-{basis or 'none'}-{width}x{height}-dark")
    page.evaluate("theme => window.ouroTheme.set(theme)", "light")  # the longest words, in the other theme
    assert editor.evaluate(FITS) == {"page": 0, "doc": 0, "out": []}, ("light", width)
    setup_browser.capture(page, f"refusal-basis-{basis}-{width}x{height}-light")
    assert not server.posts
    page.close()
