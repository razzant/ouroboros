"""Skills in the page, over the static SPA in a real engine (no runtime): a saved reply's contributed action between
Reply and Copy and in the row menu; a page module that registers its provider and writes into the draft beside one
that throws (one notice, the page keeps working); the "+" menu's skill sections and host items doing their verbs by
pointer and keyboard; contributed chips on the capsule row at 320, 360 and 1280 px in both themes, folding into one
"+N" whose list holds the rest; and the start ledger: a module whose start the page did not outlive is held next time
and named with Start it, `?hold=all` holds every module for one load, and the chat header offers no reload door."""
from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import urlparse

import pytest

from tests import test_subscription_setup_browser as setup_browser

subscription_ui = setup_browser.subscription_ui
pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]

VOICE_URL = "/api/extensions/voice/ui/r1/ui/tts.js"
BROKEN_URL = "/api/extensions/broken/ui/r1/ui/panel.js"
MODULES = {
    VOICE_URL: """export function mount(ctx) {
        window.voiceContext = Object.keys(ctx).sort();
        window.voiceCtx = ctx;
        ctx.providers.register('tts', {label: 'Device voice', voices: () => [], speak: async () => {}, stop() {}});
        return {dispose() { window.voiceDisposed = true; }};
    }""",
    BROKEN_URL: "export function mount() { throw new Error('the panel broke'); }",
}
TONES = ["ok", "warn", "bad", "muted"]


def _item(key, **extra):
    skill, ident = key.split(":")
    return {"key": key, "skill": skill, "skill_label": skill.title(), "id": ident, "label": skill.title(), "title": "",
            "icon": "", "order": 0, "revision": "r1", "render": {"kind": "declarative", "url": ""}, **extra}


def _chip(index, action=None):
    return _item(f"chips{index}:chip", placement="composer.chips", label=f"Meter {index}", order=100 + index,
                 source={"url": f"/api/extensions/chips{index}/chip", "refresh_s": 60},
                 action=action or {"verb": "notice", "text": f"Meter {index} pressed"})


def projection(chips=0, chip_action=None):
    return {
        "contributions": [
            _item("templates:simpler", placement="composer.plus", label="Explain simpler", icon="sparkle", order=10,
                  action={"verb": "draft.insert", "text": "Explain your last answer more simply."}),
            _item("lists:add", placement="composer.plus", label="Add to list", icon="list", order=20,
                  action={"verb": "skill.request", "route": "add", "with": "draft", "then": "clear_draft",
                          "url": "/api/extensions/lists/add"}, route_url="/api/extensions/lists/add"),
            _item("lists:save", placement="message.action", label="Save to list", title="Keep this answer in a list",
                  icon="list", action={"verb": "notice", "text": "Saved to Groceries"}),
            _item("voice:tts", provides="tts", runtime="client", destination="device", skill_label="Device voice",
                  label="Device voice", render={"kind": "host", "url": VOICE_URL}),
            _item("broken:panel", placement="composer.plus", action={"verb": "notice", "text": "unused"},
                  render={"kind": "host", "url": BROKEN_URL}),
            *(_chip(index, chip_action) for index in range(1, chips + 1)),
        ],
        "offers": {"stt": []}, "voice": {}, "icons": [], "problems": {},
        "preferences": {"voice": {"stt": "", "tts": ""}, "contributions": {"hidden": [], "order": []}},
    }


HISTORY = {"messages": [
    {"role": "user", "text": "What do we need from the shop?", "chat_id": 1, "source": "web",
     "ts": "2026-10-17T09:00:00Z", "history_id": "h-0", "row_address": "row:1@2026-10-17T09:00:00Z#aaaaaaaaaaaa"},
    {"role": "assistant", "text": "Milk, eggs and bread.", "chat_id": 1,
     "ts": "2026-10-17T09:00:20Z", "history_id": "h-1", "row_address": "row:1@2026-10-17T09:00:20Z#bbbbbbbbbbbb"},
], "progress": []}


def open_app(ui, path="/", *, width=1280, scheme="dark", state=None):
    """The SPA over routed gateway data; `state["chips"]` sets how many chips the projection carries (mutable)."""
    page = ui["page"]
    state = state if state is not None else {}
    state.setdefault("chips", 0)
    loaded, posted = [], []
    origin = urlparse(ui["url"]).netloc
    page.route("**/*", lambda route: route.fallback()
               if urlparse(route.request.url).netloc == origin else route.abort())
    page.route("**/api/ui/contributions*", lambda route: route.fulfill(
        content_type="application/json", body=json.dumps(projection(state["chips"], state.get("chip_action")))))
    page.route("**/api/chat/history*", lambda route: route.fulfill(
        content_type="application/json", body=json.dumps(HISTORY)))

    def extension(route):
        request = route.request
        path = urlparse(request.url).path
        if path in MODULES:
            loaded.append(path)
            route.fulfill(content_type="text/javascript", body=MODULES[path])
        elif path.endswith("/chip"):
            index = int(path.split("/")[3].removeprefix("chips"))
            route.fulfill(content_type="application/json", body=json.dumps(
                {"text": f"{index * 7}%", "tone": TONES[index % 4], "title": f"Meter {index} in detail"}))
        elif path == "/api/extensions/lists/add" and request.method == "POST":
            posted.append(request.post_data_json)
            route.fulfill(content_type="application/json", body=json.dumps({"notice": "Added to Groceries"}))
        else:
            loaded.append(path)
            route.fulfill(status=404, body="")
    page.route("**/api/extensions/**", extension)
    page.set_viewport_size({"width": width, "height": 800})
    page.emulate_media(color_scheme=scheme)
    page.goto(ui["url"] + path)
    page.wait_for_selector("#chat-input", state="attached")
    return page, loaded, posted


def toast(page, text):
    return page.locator(".toast", has_text=text)


@pytest.mark.parametrize("width,scheme", [(360, "light"), (360, "dark"), (1280, "light"), (1280, "dark")])
def test_a_saved_reply_offers_the_skill_action_and_a_failing_page_module_costs_only_itself(subscription_ui, width, scheme):
    page, loaded, _posted = open_app(subscription_ui, width=width, scheme=scheme)
    answer = page.locator(".chat-bubble.assistant").first
    action = answer.locator(".chat-message-actions > [data-message-action='contribution:lists:save']")
    action.wait_for()
    # The host's Read aloud (the voice module registered a device engine) comes first, the skill's action after it.
    answer.locator("[data-message-action='read-aloud']").wait_for()
    controls = answer.locator(".chat-message-actions > *").evaluate_all(
        "nodes => nodes.map(node => node.dataset.messageAction || node.className.split(' ')[0])")
    assert controls == ["chat-message-reply", "read-aloud", "contribution:lists:save", "chat-message-copy"], controls
    targets = answer.locator(".chat-message-actions > *").evaluate_all(
        "nodes => nodes.every(node => node.classList.contains('ui-hit-area'))")
    assert targets, "under a finger every control of the row has the same hit area"
    owner = page.locator(".chat-bubble.user").first
    assert owner.locator(".chat-message-action").count() == 0, "a skill's action is for Ouroboros's replies"
    assert action.get_attribute("aria-label") == "Save to list"
    geometry = answer.evaluate("""bubble => {
        const time = bubble.querySelector('.msg-time').getBoundingClientRect();
        const row = bubble.querySelector('.chat-message-actions').getBoundingClientRect();
        const box = bubble.getBoundingClientRect();
        return {timeRight: time.right, rowLeft: row.left, rowRight: row.right, boxRight: box.right,
            sameLine: time.bottom > row.top && time.top < row.bottom};
    }""")
    assert not geometry["sameLine"] or geometry["timeRight"] <= geometry["rowLeft"] + 0.5, geometry
    assert geometry["rowRight"] <= geometry["boxRight"] + 0.5, geometry
    action.click()
    toast(page, "Saved to Groceries").wait_for()
    answer.locator(".message").click(button="right")
    menu = page.locator(".chat-row-menu:not([data-leaving])")
    menu.wait_for()
    assert menu.locator("[role=menuitem]").all_inner_texts()[-1] == "Save to list", "after the row's own doors, in words"
    page.keyboard.press("Escape")
    # The module that registers its provider runs; the one that throws is off with one notice; the page still works.
    page.wait_for_function("Array.isArray(window.voiceContext)")
    assert page.evaluate("window.voiceContext") == [
        "api", "draft", "key", "notice", "origin", "providers", "signal", "skill", "theme"]
    toast(page, "Broken stopped working: the panel broke.").wait_for()
    providers = page.evaluate("""async () => (await import('/static/modules/host_modules.js'))
        .getHostModules().clientProviders('tts').map((entry) => entry.key)""")
    assert providers == ["voice:tts"]
    assert sorted(set(loaded)) == [BROKEN_URL, VOICE_URL]
    # A page module writes into the visible room's draft through its context, never by reaching into the composer.
    assert page.evaluate("window.voiceCtx.draft.insert('Read this back to me')") is True
    assert page.locator("#chat-input").input_value() == "Read this back to me"
    page.locator("#chat-input").fill("still typing")
    assert page.locator("#chat-input").input_value() == "still typing"
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth"), "no horizontal page scroll"


@pytest.mark.parametrize("width,scheme", [(1280, "light"), (360, "dark")])
def test_plus_lists_skill_sections_then_host_items_and_each_does_its_verb(subscription_ui, width, scheme):
    page, _loaded, posted = open_app(subscription_ui, width=width, scheme=scheme)
    toast(page, "Broken stopped working").wait_for()  # its "+" item is off with it
    field = page.locator("#chat-input")
    field.click()
    page.click("#page-chat #chat-plus")
    menu = page.locator(".chat-plus-menu:not([data-leaving])")
    menu.wait_for()
    items = menu.locator("[role=menuitem]").all_inner_texts()
    assert [text.strip() for text in items] == [
        "Attach file", "Explain simpler", "Add to list", "Add dictation…", "Make a button with Ouroboros…"], items
    assert menu.locator(".chat-plus-label").all_inner_texts() == ["Templates", "Lists", "Ouroboros"], \
        "each skill under its label, the host's own items last under Ouroboros's"
    assert menu.locator("[data-plus-item='templates:simpler'] svg").count() == 1, "a skill's glyph from the host set"
    box = menu.bounding_box()
    assert box["x"] >= 0 and box["x"] + box["width"] <= width + 0.5, box
    assert page.evaluate("document.activeElement.id") == "chat-input", "a pointer leaves the field its focus"
    capture_name = f"plus-menu-{width}-{scheme}"
    setup_browser.capture(page, capture_name)
    menu.locator("[data-plus-item='templates:simpler']").click()
    assert field.input_value() == "Explain your last answer more simply.", "a template only fills the draft"
    assert not posted
    # The keyboard: Enter on "+" moves into the menu; Add to list sends the draft to its own route, then clears it.
    field.fill("Buy milk")
    page.focus("#page-chat #chat-plus")
    page.keyboard.press("Enter")
    menu.wait_for()
    assert page.evaluate("document.activeElement.dataset.plusItem") == "attach"
    page.keyboard.press("ArrowDown")
    page.keyboard.press("ArrowDown")
    assert page.evaluate("document.activeElement.dataset.plusItem") == "lists:add"
    page.keyboard.press("Enter")
    toast(page, "Added to Groceries").wait_for()
    assert posted == [{"text": "Buy milk"}]
    page.wait_for_function("document.querySelector('#chat-input').value === ''")
    # The host's own item starts a request for the owner to finish; nothing is sent.
    page.click("#page-chat #chat-plus")
    menu.locator("[data-plus-item='make-button']").click()
    assert field.input_value() == "Make me a button in the + menu that "
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth"), "no horizontal page scroll"


ROW = """() => {
    const page = document.querySelector('#page-chat');
    const row = page.querySelector('.chat-toolbar-row');
    const slot = row.querySelector('[data-composer-chips]');
    const rect = (node) => node.getBoundingClientRect();
    const chips = [...slot.querySelectorAll('.chat-composer-chip:not(.chat-chip-more)')];
    const more = slot.querySelector('.chat-chip-more');
    const style = getComputedStyle(slot);
    return {
        slotStyle: { overflowX: style.overflowX, maxWidth: style.maxWidth, flex: style.flex, width: rect(slot).width,
            left: rect(slot).left, clientWidth: slot.clientWidth, scrollWidth: slot.scrollWidth, folded: slot.dataset.folded || '' },
        children: [...row.children].map((node) => [node.className, rect(node).left, rect(node).right]),
        moreWidth: more && !more.hidden ? rect(more).width : null,
        lines: new Set([...row.children].filter((node) => rect(node).width > 0).map((node) => Math.round(rect(node).top))).size,
        rowHeight: rect(row).height, rowRight: rect(row).right, compact: row.dataset.ownerCompact || '',
        hosts: [...row.children].filter((node) => node !== slot && rect(node).width > 0)
            .map((node) => [node.className, Math.round(rect(node).width * 2) / 2]),
        shown: chips.filter((chip) => !chip.hidden).map((chip) => ({ key: chip.dataset.chipKey, right: rect(chip).right,
            whole: chip.scrollWidth <= chip.clientWidth + 1, name: chip.getAttribute('aria-label') })),
        valued: chips.every((chip) => chip.querySelector('.chat-chip-value').textContent !== ''),
        folded: more && !more.hidden ? Number(more.textContent.slice(1)) : 0,
        moreRight: more && !more.hidden ? rect(more).right : null,
        slot: { right: rect(slot).right, hidden: slot.hidden },
        roomless: slot.dataset.roomless === '1',
        overflow: document.documentElement.scrollWidth - window.innerWidth,
    };
}"""


def _record(evidence):
    root = os.environ.get("OUROBOROS_UI_EVIDENCE_DIR")
    if root:
        (Path(root) / "chips-rows.json").write_text(json.dumps(evidence, indent=1), encoding="utf-8")


def test_chips_take_the_room_the_capsules_leave_and_fold_into_one_more_chip(subscription_ui):
    state = {"chips": 0}
    page, _loaded, _posted = open_app(subscription_ui, state=state)
    evidence = {}
    subscription_ui["errors"].watch(page)  # each reopen may cut off the page's own reads (tests/ui_console_errors.py)
    for scheme in ("light", "dark"):
        for width in (320, 360, 1280):
            page.set_viewport_size({"width": width, "height": 800})
            page.emulate_media(color_scheme=scheme)
            state["chips"] = 0
            page.goto(subscription_ui["url"] + "/")
            page.wait_for_selector("#chat-input", state="attached")
            page.wait_for_timeout(300)
            before = page.evaluate(ROW)
            for count in (1, 6):
                where = f"{scheme} {width}px {count} chip(s)"
                state["chips"] = count
                page.goto(subscription_ui["url"] + "/")
                page.wait_for_function(
                    """(count) => { const chips = document.querySelectorAll('#page-chat .chat-composer-chip:not(.chat-chip-more)');
                        return chips.length === count
                            && [...chips].every((chip) => chip.querySelector('.chat-chip-value').textContent !== ''); }""",
                    arg=count)
                page.wait_for_timeout(300)
                after = page.evaluate(ROW)
                evidence[where] = {"before": before, "after": after}
                _record(evidence)
                # One line; the host capsules keep their place, width and words; every chip shown is whole and inside.
                assert after["lines"] == 1 and after["rowHeight"] <= before["rowHeight"] + 0.5, (where, before, after)
                assert after["hosts"] == before["hosts"] and after["compact"] == before["compact"], (where, before, after)
                assert all(chip["whole"] and chip["right"] <= after["slot"]["right"] + 0.5 for chip in after["shown"]), (where, after)
                assert after["overflow"] <= 0, (where, after)
                if after["roomless"]:
                    # The row has no room even for "+N" (☰ leads a phone's row): "+" lists every chip under its skill.
                    assert after["shown"] == [] and after["moreRight"] is None, (where, after)
                    page.click("#page-chat [data-composer-plus]")
                    listed = page.locator(".chat-plus-menu:not([data-leaving]) [data-plus-item^='chip:']")
                    listed.first.wait_for()
                    assert listed.count() == count, (where, listed.all_inner_texts())
                    assert listed.first.inner_text().strip() == "Meter 1 7%", where
                    listed.first.click()
                    toast(page, "Meter 1 pressed").wait_for()
                    setup_browser.capture(page, f"chips-{scheme}-{width}-{count}")
                    continue
                assert len(after["shown"]) + after["folded"] == count, (where, after)
                for chip in after["shown"]:
                    index = int(chip["key"].split(":")[0].removeprefix("chips"))
                    assert chip["name"] == f"Meter {index} {index * 7}%", (where, chip)
                setup_browser.capture(page, f"chips-{scheme}-{width}-{count}")
                if after["folded"]:
                    assert after["moreRight"] <= after["slot"]["right"] + 0.5, (where, after)
                    page.click("#page-chat .chat-chip-more")
                    listed = page.locator(".chat-plus-menu:not([data-leaving]) [role=menuitem]")
                    listed.first.wait_for()
                    assert listed.count() == after["folded"], (where, listed.all_inner_texts())
                    hidden_first = count - after["folded"] + 1
                    assert listed.first.inner_text().strip() == f"Meter {hidden_first} {hidden_first * 7}%", where
                    listed.first.click()
                    toast(page, f"Meter {hidden_first} pressed").wait_for()
                else:
                    page.click("#page-chat .chat-composer-chip[data-chip-key='chips1:chip']")
                    toast(page, "Meter 1 pressed").wait_for()
    afters = [value["after"] for value in evidence.values()]
    assert any(after["folded"] or after["roomless"] for after in afters), "six chips on a phone fold (into +N or into +)"
    assert any(not after["folded"] and len(after["shown"]) == 1 for after in afters), "one chip shows whole"


def test_a_chip_that_opens_its_widgets_takes_the_owner_to_the_widgets_page(subscription_ui):
    page, _loaded, _posted = open_app(subscription_ui, state={"chips": 1, "chip_action": {"verb": "open.tab"}})
    chip = page.locator("#page-chat .chat-composer-chip[data-chip-key='chips1:chip']")
    chip.wait_for()
    chip.click()
    page.wait_for_function("document.querySelector('#page-widgets')?.classList.contains('active')")
    assert not page.evaluate("document.querySelector('#page-chat').classList.contains('active')")


def test_hold_all_starts_without_page_code_and_start_it_mounts_it_in_place(subscription_ui):
    page, loaded, _posted = open_app(subscription_ui, "/?hold=all#chat")
    notice = toast(page, "did not start: this page was opened with everything held.")
    notice.wait_for()
    page.locator(".chat-bubble.assistant .chat-message-action").wait_for()
    assert loaded == [], "no page module is even fetched"
    assert page.evaluate("window.voiceContext") is None
    assert urlparse(page.url).query == "", "the flag leaves the address once read: a reload starts as usual"
    page.evaluate("window.samePage = true")
    notice.get_by_role("button", name="Start it").click()
    page.wait_for_function("Array.isArray(window.voiceContext)")
    assert page.evaluate("window.samePage") is True, "no reload: the held code starts in this page"


def test_a_start_the_page_did_not_outlive_is_held_next_time_and_the_bar_offers_no_reload_door(subscription_ui):
    page, loaded, _posted = open_app(subscription_ui)
    page.wait_for_function("Array.isArray(window.voiceContext)")
    bar = page.locator("#page-chat .chat-page-header")
    bar.locator('[data-chat-command="panic"]').wait_for()
    assert bar.get_by_role("button", name="Restart without plugins").count() == 0
    # The entry a page that froze during the voice module's start leaves in the ledger.
    page.evaluate("""() => localStorage.setItem('ouroboros.holds', JSON.stringify({v: 1, entries: {
        'module:voice:tts': {reason: 'start', rev: 'r1', at: Date.now()}}}))""")
    loaded.clear()
    subscription_ui["errors"].watch(page)  # the reopen may cut off the page's own reads (tests/ui_console_errors.py)
    page.goto(subscription_ui["url"] + "/")
    held = toast(page, "Device voice did not start: the page stopped while it was starting last time.")
    held.wait_for()
    toast(page, "stopped working: the panel broke").wait_for()
    assert BROKEN_URL in loaded, "a module the ledger does not hold starts as usual"
    assert VOICE_URL not in loaded and page.evaluate("window.voiceContext") is None
    held.get_by_role("button", name="Start it").click()
    page.wait_for_function("Array.isArray(window.voiceContext)")
    assert loaded.count(VOICE_URL) == 1
    page.wait_for_function("""() => !(JSON.parse(localStorage.getItem('ouroboros.holds') || '{}').entries || {})['module:voice:tts']""")
