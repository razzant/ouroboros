"""Settings › Agents in a browser (docs/DESIGN.md "Settings"): the list of heads with a switch, and one agent's page.

One scene (consilium, astra's worry 4): an agent's page holds an unsaved description while switches on OTHER agents
write at once: the last reviewer's asks first and sends the same body again with the flag, a re-check row's goes
through the checker's endpoint like every roster write, and after leaving Settings and returning the switches hold
and the unsaved description was published by none of them. Then the look: the list and the page at the owner's
window and at 320 x 640, both themes, with no sideways scroll.

The roster's server half is a fake here, answering as PINS section 6 pins it (`_meta.tokens`, `if_unchanged`, a
receipt in `changes`, the `empty_review_pool` refusal): marked LANE A below, to be run against the real endpoint on
the integrated tree."""
from __future__ import annotations

import copy
import json

import pytest

from tests import test_subscription_setup_browser as setup_browser
from tests.settings_browser_helpers import go_back, open_entry, open_settings, press_editor_save, settings_page, top_path

subscription_ui = setup_browser.subscription_ui
pytestmark = [pytest.mark.ui_browser]

KEY = "OUROBOROS_SUBAGENTS"
EMPTY_POOL = "no reviewers marked; mark at least one row or save with `allow_empty_review_pool`"  # the server's own
ASKS = "No agent marked Reviewer would be on, so the main model reviews alone: one model, no second opinion."


def _session(agent_id, target, use, **extra):
    return {"subagent_id": agent_id, "recommended_use": use, "route": {"kind": "agent_session", "target_id": target}, **extra}


def _api(agent_id, target, use, **extra):
    return {"subagent_id": agent_id, "recommended_use": use, "route": {"kind": "api_model", "target_id": target}, **extra}


SCENE = [
    _session("reviewer", "codex=gpt-test", "Meticulous reviewer. Give it the whole repository.", review_eligible=True),
    _api("recheck", "deepseek/deepseek-v5", "Reads only the packet it is given.", delivery="packet", rechecks_flags=True),
    _api("scout", "openai::gpt-6-luna", "Quick edits and scouting through code."),
]

# A realistically long list: sessions and API models, some reviewers, two switched off, one long description.
LOOK = [
    _session("a01", "claude=claude-test", "Main executor: code, front end and documents. Strong plans and long refactors across "
             "many files; double-check its numbers and anything it says about dates. Give it the whole task with the "
             "acceptance test, not a piece of it.", review_eligible=True),
    _session("a02", "codex=gpt-test", "Meticulous reviewer and code researcher. Slow, rarely wrong.", review_eligible=True, effort="xhigh"),
    _session("a03", "cursor=grok-4.7-xhigh", "Another point of view, bold hypotheses, a good critic of plans.", review_eligible=True),
    _api("a04", "moonshotai/kimi-k3.1-pro", "Cheap brainstorming in numbers; a fresh lens on product questions."),
    _api("a05", "deepseek/deepseek-v5", "Reviewer for maths and algorithms. Reads only the packet it is given.",
         review_eligible=True, delivery="packet"),
    _api("a06", "openai::gpt-6-sol", "Reviewer by API key: counted when subscriptions are exhausted.", review_eligible=True),
    _api("a07", "anthropic::claude-opus-5-5", "Opus by API key. Used when every subscription account waits for its reset."),
    _api("a08", "xiaomi/mimo-3", "Tried for translation; kept off until the next comparison.", enabled=False),
    _session("a09", "codex=gpt-6-terra", "Balanced Codex agent for everyday tasks in projects."),
    _api("a10", "meta/llama-5-405b", "Open weights, for comparisons only.", enabled=False),
]


class RosterServer:
    """LANE A: the settings server's roster half as PINS section 6 pins it, kept in memory."""

    def __init__(self, ui, items, *, saved=True):
        self.ui = ui
        self.roster = {"enabled": True, "items": copy.deepcopy(items)}
        self.saved = saved
        self.version = 0
        self.posts = []

    @property
    def token(self):
        return f"roster@{self.version}"

    def agent(self, agent_id):
        return next(item for item in self.roster["items"] if item["subagent_id"] == agent_id)

    def document(self):
        meta = {**self.ui["settings"]["_meta"], "tokens": {KEY: self.token}, "effects": {KEY: "next_task"},
                "available_subagents": {"source": "configured" if self.saved else "undecided", "diagnostics": [],
                                        "candidate": None if self.saved else self.roster, "automatic": False}}
        return {**self.ui["settings"], KEY: json.dumps(self.roster) if self.saved else "",
                "OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS": "", "OUROBOROS_MAX_ACTIVE_SUBAGENTS_PER_ROOT": 6,
                "OUROBOROS_MAX_SUBAGENT_DEPTH": 3, "OUROBOROS_SUBAGENT_WORKTREE_ROOT": "",
                "OUROBOROS_SUBAGENT_PROJECTS_ROOT": "", "_meta": meta}

    def write(self, body):
        self.posts.append(copy.deepcopy(body))
        roster = (body.get("settings") or {}).get(KEY)
        if (body.get("if_unchanged") or {}).get(KEY) != self.token:
            return 409, {"saved": False, "code": "settings_changed_elsewhere", "error": "The agents list changed elsewhere."}
        pool = [item for item in roster["items"] if item.get("review_eligible") is True and item.get("enabled") is not False]
        if roster["items"] and not pool and roster != self.roster and body.get("allow_empty_review_pool") is not True:
            return 400, {"saved": False, "code": "empty_review_pool", "error": EMPTY_POOL}
        self.roster = {"enabled": roster["enabled"], "items": [
            {key: value for key, value in item.items()
             if not (key in {"review_eligible", "rechecks_flags"} and value is False) and not (key == "enabled" and value is True)}
            for item in roster["items"]]}
        self.saved = True
        self.version += 1
        return 200, {"ok": True, "changes": {KEY: {"withheld": "structure", "token": self.token}}}

    def serve(self, page):
        def settings(route):
            if route.request.method == "GET" and "preview" not in route.request.url:
                route.fulfill(content_type="application/json", body=json.dumps(self.document()))
            else:
                route.fallback()

        def checker(route):
            status, body = self.write(route.request.post_data_json or {})
            route.fulfill(status=status, content_type="application/json", body=json.dumps(body))

        page.route("**/api/settings*", settings)
        page.route("**/api/owner/safety-model", checker)
        return page


def _agents(ui, items, *, viewport, saved=True):
    page = ui["new_page"](viewport=viewport, has_touch=viewport["width"] < 720)
    server = RosterServer(ui, items, saved=saved)
    server.serve(page)
    page.goto(ui["url"] + "/#settings")
    level = open_settings(page, "agents")
    level.locator("[data-entry-id]").first.wait_for(state="visible")
    return page, server, level


def _switch(level, agent_id):
    return level.locator(f'[data-entry-id="{agent_id}"] [data-entry-switch]')


def _leave_and_return(page):
    """Go to Dashboard and come back to Settings by the app's own navigation (the drawer on a phone)."""
    for here, target in (("settings", "dashboard"), ("dashboard", "settings")):
        if page.locator("html").get_attribute("data-layout") == "stack":
            page.locator(f"#page-{here} [data-mobile-nav-toggle]").first.click()
        page.locator(f'#primary-sidebar [data-nav-page="{target}"]').click()
        page.locator(f"#page-{target}").wait_for(state="visible")
        page.locator(f"#page-{here}").wait_for(state="hidden")


def _no_sideways_scroll(page, path):
    level = settings_page(page, path)
    widths = level.evaluate("el => [el.scrollWidth, el.clientWidth, document.documentElement.scrollWidth, innerWidth]")
    assert widths[0] <= widths[1] + 1 and widths[2] <= widths[3] + 1, f"{path!r} scrolls sideways: {widths}"
    spill = level.evaluate("""el => { const edge = el.getBoundingClientRect().right + 1;
        return [...el.querySelectorAll('*')].filter((node) => node.getClientRects().length
            && node.getBoundingClientRect().right > edge && getComputedStyle(node).position !== 'fixed')
            .map((node) => node.className || node.tagName).slice(0, 5); }""")
    assert not spill, f"{path!r}: content past the page's edge: {spill}"


@pytest.mark.parametrize("width,height", [(1087, 740), (320, 640)])
def test_a_switch_writes_its_own_entry_and_never_an_open_form(subscription_ui, width, height):
    ui = subscription_ui
    page, server, level = _agents(ui, SCENE, viewport={"width": width, "height": height})
    assert [row.get_attribute("data-entry-id") for row in level.locator("[data-entry-id]").all()] == ["reviewer", "recheck", "scout"]
    _no_sideways_scroll(page, "agents")

    # An agent's page: an unsaved description, then Back. Nothing is written.
    editor = open_entry(page, "scout")
    assert top_path(page) == "agents/scout"
    _no_sideways_scroll(page, "agents/scout")
    editor.locator('[data-subagent-field="recommended_use"]').fill("UNSAVED WORDS")
    assert go_back(page) == "agents"
    assert server.posts == []

    # The last reviewer off: the server refuses, the shared confirm asks, and the SAME body goes again with the flag.
    level = settings_page(page, "agents")
    level.locator('[data-entry-id="reviewer"] .ui-switch').click()
    dialog = page.locator(".confirm-dialog")
    dialog.wait_for(state="visible")
    assert ASKS in dialog.inner_text() and "allow_empty_review_pool" not in dialog.inner_text()
    assert len(server.posts) == 1 and "allow_empty_review_pool" not in server.posts[0]
    dialog.locator("[data-confirm-ok]").click()
    page.wait_for_function("() => !document.querySelector('.confirm-dialog')")
    _switch(level, "reviewer").wait_for(state="attached")
    page.wait_for_function("""() => { const input = document.querySelector('#page-settings [data-entry-id="reviewer"] [data-entry-switch]');
        return input && !input.checked && input.getAttribute('aria-busy') !== 'true'; }""")
    first, second = server.posts
    assert second == {**first, "allow_empty_review_pool": True}, "the confirmed write is the refused body, never a newer list"
    assert server.agent("reviewer").get("enabled") is False

    # A re-check row off: the same owner transaction (the generic save would refuse it outside Cyber Pro). No
    # reviewer is on now, and the server judges every changed roster that leaves none: it asks again, for THIS roster.
    level.locator('[data-entry-id="recheck"] .ui-switch').click()
    dialog.wait_for(state="visible")
    dialog.locator("[data-confirm-ok]").click()
    page.wait_for_function("""() => { const input = document.querySelector('#page-settings [data-entry-id="recheck"] [data-entry-switch]');
        return input && !input.checked && input.getAttribute('aria-busy') !== 'true'; }""")
    assert len(server.posts) == 4 and server.posts[3] == {**server.posts[2], "allow_empty_review_pool": True}
    assert server.posts[2]["settings"] != first["settings"], "the second question is about its own roster"
    assert server.agent("recheck").get("enabled") is False
    assert server.agent("recheck").get("rechecks_flags") is True, "the mark stays with its row"
    assert not any(path == "/api/settings" for path, _body in ui["posts"]), "no roster write takes the generic save"
    assert level.locator("[data-entry-result]").inner_text().strip() == "Saved · applies to new tasks"

    # Leave Settings and return: the switches hold, and no write carried the unsaved description.
    _leave_and_return(page)
    level = open_settings(page, "agents")
    page.wait_for_function("""() => ['reviewer', 'recheck'].every((id) => {
        const input = document.querySelector(`#page-settings [data-entry-id="${id}"] [data-entry-switch]`);
        return input && !input.checked; })""")
    assert _switch(level, "scout").is_checked()
    assert "UNSAVED WORDS" not in json.dumps(server.posts) and "UNSAVED WORDS" not in json.dumps(server.roster)
    assert server.agent("scout")["recommended_use"] == "Quick edits and scouting through code."

    # A reviewer on again: with one in the pool the server asks nothing.
    level.locator('[data-entry-id="reviewer"] .ui-switch').click()
    page.wait_for_function("""() => { const input = document.querySelector('#page-settings [data-entry-id="reviewer"] [data-entry-switch]');
        return input && input.checked && input.getAttribute('aria-busy') !== 'true'; }""")
    assert len(server.posts) == 5 and "enabled" not in server.agent("reviewer")

    # The page that was left kept its unsaved form; its own Save is what publishes it, into its own entry only.
    editor = open_entry(page, "scout")
    field = editor.locator('[data-subagent-field="recommended_use"]')
    assert field.input_value() == "UNSAVED WORDS"
    assert press_editor_save(page) == "Saved · applies to new tasks"
    assert server.agent("scout")["recommended_use"] == "UNSAVED WORDS"
    assert "enabled" not in server.agent("reviewer") and server.agent("recheck").get("enabled") is False
    assert not ui["errors"].unexpected(), ui["errors"].report()


def test_a_row_the_page_is_opened_for_takes_the_focus_with_its_value(subscription_ui):
    """Search and other pages open Agents on a row: a row of the fold is revealed (the fold opens) holding its
    stored value, though the page reads after it is opened and a focused field is never rewritten; the roster's own
    row takes the focus on its switch."""
    ui = subscription_ui
    page = ui["new_page"](viewport={"width": 1087, "height": 740})
    RosterServer(ui, SCENE).serve(page)
    page.goto(ui["url"] + "/#settings")
    depth = "OUROBOROS_MAX_SUBAGENT_DEPTH"
    level = open_settings(page, "agents", depth)
    page.wait_for_function("""row => { const at = document.activeElement;
        return at?.closest('[data-setting-row]')?.dataset.settingRow === row && at.value === '3'; }""", arg=depth)
    assert level.locator("details[data-settings-fold]").evaluate("el => el.open")
    open_settings(page, "")
    level = open_settings(page, "agents", KEY)
    page.wait_for_function("""row => { const at = document.activeElement;
        return at?.closest('[data-setting-row]')?.dataset.settingRow === row && at.getAttribute('role') === 'switch'
            && at.checked; }""", arg=KEY)
    assert not level.locator("details[data-settings-fold]").evaluate("el => el.open")
    assert not ui["errors"].unexpected(), ui["errors"].report()


@pytest.mark.parametrize("theme", ["dark", "light"])
@pytest.mark.parametrize("width,height", [(1087, 740), (320, 640)])
def test_the_list_and_an_agents_page_keep_their_shape(subscription_ui, width, height, theme):
    ui = subscription_ui
    ui["fixture"]["status"]["harnesses"][2]["models"] = [{"id": "grok-4.7-xhigh"}]
    page, _server, level = _agents(ui, LOOK, viewport={"width": width, "height": height})
    page.evaluate("theme => { document.documentElement.dataset.theme = theme; }", theme)
    name = f"agents-{{}}-{width}x{height}-{theme}"

    # The list: the roster's switch, a head per agent (mark, name, what it is for, switch), the filter, the fold.
    assert level.locator(f'[data-setting-row="{KEY}"] [data-setting-control]').is_checked()
    assert level.locator("[data-entry-id]").count() == len(LOOK)
    assert level.locator(".agents-card-title").nth(0).inner_text().strip() == f"{len(LOOK)} agents"
    assert [chip.inner_text().strip() for chip in level.locator("[data-agents-filter]").all()][:3] == [
        f"All {len(LOOK)}", "Reviewers 5", "Off 2"]
    head = level.locator('[data-entry-id="a02"]')
    assert head.locator(".ui-row-title").inner_text().split() == ["gpt-test/xhigh", "Reviewer"]
    assert head.locator(".ui-row-meta").inner_text().startswith("Codex · agent · Meticulous reviewer")
    assert not _switch(level, "a08").is_checked() and _switch(level, "a01").is_checked()
    box, toggle = head.bounding_box(), head.locator(".ui-switch").bounding_box()
    assert toggle["x"] + toggle["width"] <= box["x"] + box["width"] and toggle["x"] > box["x"] + box["width"] / 2, (
        "the switch keeps the row's end at every width")
    _no_sideways_scroll(page, "agents")
    setup_browser.capture(page, name.format("list"))

    level.locator('[data-agents-filter="off"]').click()
    assert [row.get_attribute("data-entry-id") for row in level.locator("[data-entry-id]").all()] == ["a08", "a10"]
    level.locator('[data-agents-filter="all"]').click()
    fold = level.locator("details[data-settings-fold]")
    assert not fold.evaluate("el => el.open")
    fold.locator(":scope > summary").click()
    assert [row.get_attribute("data-setting-row") for row in fold.locator("[data-setting-row]").all()] == [
        "OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS", "OUROBOROS_MAX_ACTIVE_SUBAGENTS_PER_ROOT", "OUROBOROS_MAX_SUBAGENT_DEPTH",
        "OUROBOROS_SUBAGENT_WORKTREE_ROOT", "OUROBOROS_SUBAGENT_PROJECTS_ROOT"]
    assert fold.locator('[data-setting-row="OUROBOROS_ALLOW_MUTATIVE_SUBAGENTS"] [aria-pressed="true"]').inner_text().strip() == "Auto"
    fold.scroll_into_view_if_needed()
    _no_sideways_scroll(page, "agents")
    setup_browser.capture(page, name.format("list-fold"))

    # An agent's page: its name is the title, then In use, what it is good at, how it runs with Save, and its removal.
    editor = open_entry(page, "a05")
    assert editor.locator(".settings-page-title").inner_text().strip() == "deepseek-v5"
    # The shell shows the level before the editor's body has drawn; read the cards once its Save stands.
    editor.locator("[data-editor-save]").wait_for(state="visible")
    assert [title.inner_text().strip() for title in editor.locator(".agents-card-title").all()] == [
        "In use", "What it is good at", "Runs on"]
    assert editor.locator('[data-subagent-field="rechecks_flags"]').count() == 1, "a packet reviewer may take the re-check"
    assert editor.locator("[data-editor-save]").is_visible() and editor.locator("[data-editor-remove]").count() == 1
    _no_sideways_scroll(page, "agents/a05")
    setup_browser.capture(page, name.format("agent"))
    editor.locator("[data-editor-remove]").scroll_into_view_if_needed()
    _no_sideways_scroll(page, "agents/a05")
    setup_browser.capture(page, name.format("agent-end"))
    assert not ui["errors"].unexpected(), ui["errors"].report()
