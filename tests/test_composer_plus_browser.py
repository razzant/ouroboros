"""The composer's "+" and its empty mount points in real engines (docs/DESIGN.md "Composer owner controls").

On a real server in Chromium and WebKit, light and dark, at 1280 and 360 px: "+" stands where the paperclip
stood, a 36 px target that keeps the text row's height; a pointer opens its menu above it without taking the
field's focus, and Attach file opens the file chooser and stages the chosen file; by keyboard, Enter opens the
menu onto Attach file, Esc closes it and returns focus to "+", Tab leaves it; a drop on the room and an image
pasted into the field still stage files, and the menu fits a 320 px phone. The voice and chips slots take no
room while empty; chips put into the capsule row stay on its one line, taking only the room the host capsules
leave (one chip or six, at 320 and 360 px: no wrap, no horizontal page scroll), and step aside with the other
controls while an owner capsule stands alone.
"""
from __future__ import annotations

import json

import pytest

from tests.test_effort_range_browser import _evidence_dir, _launch, _open_chat
from tests.test_ui_smoke_playwright import direct_server_with_data  # noqa: F401 - pytest fixture import

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]

PLUS = "#page-chat #chat-plus"
MENU = ".chat-plus-menu:not([data-leaving])"
ATTACH = f"{MENU} [data-plus-item='attach']"
FACTS = """() => {
    const box = (node) => { if (!node) return null; const r = node.getBoundingClientRect();
        return { left: r.left, right: r.right, top: r.top, bottom: r.bottom, width: r.width, height: r.height }; };
    const plus = document.querySelector('#page-chat #chat-plus');
    const menu = document.querySelector('.chat-plus-menu:not([data-leaving])');
    const page = document.querySelector('#page-chat');
    const active = document.activeElement;
    const input = page.querySelector('#chat-input'), words = getComputedStyle(input);
    return {
        plus: box(plus), row: box(plus.closest('.chat-text-row')), input: box(page.querySelector('#chat-input')),
        send: box(page.querySelector('.chat-send-group')), menu: box(menu),
        lastLine: box(input).bottom - parseFloat(words.paddingBottom) - parseFloat(words.lineHeight) / 2,
        items: menu ? [...menu.querySelectorAll('[role="menuitem"]')].map((item) => item.textContent.trim()) : [],
        expanded: plus.getAttribute('aria-expanded'), name: plus.getAttribute('aria-label'),
        focus: active === plus ? 'plus' : active?.id === 'chat-input' ? 'input'
            : active?.dataset?.plusItem ? 'item:' + active.dataset.plusItem : (active?.tagName || '').toLowerCase(),
        slots: ['[data-composer-voice]', '[data-composer-chips]'].map((sel) => {
            const node = page.querySelector(sel);
            return { display: getComputedStyle(node).display, width: node.getBoundingClientRect().width };
        }),
        overflow: document.documentElement.scrollWidth - window.innerWidth,
        view: window.innerWidth,
    };
}"""
TEXT_FILE = {"name": "plus-attach.txt", "mimeType": "text/plain", "buffer": b"staged through the plus menu\n"}
# The two other routes a file takes into the composer, unchanged by "+": a drop anywhere on the room, an image
# pasted into the field. An engine that cannot build the synthetic event says so (`*Supported`).
DROP_AND_PASTE = """() => {
    const page = document.querySelector('#page-chat');
    const badges = () => page.querySelectorAll('#chat-attachment-preview .attach-badge').length;
    const drop = new DataTransfer();
    drop.items.add(new File(['dropped'], 'dropped.txt', { type: 'text/plain' }));
    const dragSupported = Boolean(new DragEvent('drop', { dataTransfer: drop }).dataTransfer);
    for (const type of ['dragenter', 'dragover', 'drop']) {
        page.dispatchEvent(new DragEvent(type, { bubbles: true, cancelable: true, dataTransfer: drop }));
    }
    const dropped = badges();
    const clip = new DataTransfer();
    clip.items.add(new File([new Uint8Array([137, 80, 78, 71])], 'pasted.png', { type: 'image/png' }));
    const paste = new ClipboardEvent('paste', { bubbles: true, cancelable: true, clipboardData: clip });
    const pasteSupported = Boolean(paste.clipboardData);
    page.querySelector('#chat-input').dispatchEvent(paste);
    return { dragSupported, dropped, pasteSupported, pasted: badges() - dropped };
}"""


def _facts(page) -> dict:
    return page.evaluate(FACTS)


def _assert_plus_in_place(facts: dict, where: str) -> None:
    plus, row, field = facts["plus"], facts["row"], facts["input"]
    assert abs(plus["width"] - 36) <= 0.5 and abs(plus["height"] - 36) <= 0.5, (where, plus)
    # The finger-sized target keeps the paperclip's place: the row is no taller than its field needs.
    assert row["height"] <= field["height"] + 18.5, (where, row, field)
    # The field can wrap even its placeholder on a phone. The controls ride on its last line.
    assert abs((plus["top"] + plus["bottom"]) / 2 - facts["lastLine"]) <= 1, (where, facts)
    assert plus["right"] <= field["left"] + 0.5, (where, plus, field)
    assert facts["name"] == "More actions" and facts["expanded"] == "false", (where, facts)
    assert all(slot == {"display": "none", "width": 0} for slot in facts["slots"]), (where, facts["slots"])
    assert facts["overflow"] <= 0, (where, facts["overflow"])


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_plus_menu_attaches_by_pointer_and_keyboard_in_both_themes(direct_server_with_data, engine, tmp_path):  # noqa: F811
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    evidence = {}
    with sync_playwright() as pw:
        browser, page = _launch(pw, engine, viewport={"width": 1280, "height": 900})
        try:
            page.close()
            for theme in ("light", "dark"):
                for width in (1280, 360):
                    where = f"{engine} {theme} {width}px"
                    page = browser.new_page(viewport={"width": width, "height": 800})
                    try:
                        _open_chat(page, url, width=width, height=800, theme=theme)
                        closed = _facts(page)
                        _assert_plus_in_place(closed, where)

                        # A pointer: the field keeps its focus; the menu opens above "+", inside the window.
                        page.click("#chat-input")
                        page.click(PLUS)
                        page.wait_for_selector(MENU, state="visible", timeout=5_000)
                        opened = _facts(page)
                        assert opened["focus"] == "input", (where, opened["focus"])
                        # Attach file first; the host's own "Make a button with Ouroboros…" closes the list.
                        assert opened["expanded"] == "true" and opened["items"][0] == "Attach file", (where, opened)
                        assert opened["items"][-1] == "Make a button with Ouroboros…", (where, opened)
                        menu = opened["menu"]
                        assert menu["bottom"] <= opened["plus"]["top"] + 0.5, (where, menu, opened["plus"])
                        assert menu["left"] >= 0 and menu["right"] <= opened["view"] + 0.5, (where, menu)
                        assert opened["overflow"] <= 0, (where, opened["overflow"])
                        page.screenshot(path=str(_evidence_dir(tmp_path) / f"composer-plus-{engine}-{theme}-{width}.png"),
                                        animations="disabled")
                        with page.expect_file_chooser(timeout=5_000) as chooser:
                            page.click(ATTACH)
                        chooser.value.set_files([TEXT_FILE])
                        page.wait_for_function(
                            "() => document.querySelectorAll('#chat-attachment-preview .attach-badge').length === 1")
                        attached = _facts(page)
                        assert attached["menu"] is None and attached["expanded"] == "false", (where, attached)
                        page.click("#chat-attachment-preview .attach-remove")
                        page.wait_for_function(
                            "() => !document.querySelector('#chat-attachment-preview .attach-badge')")

                        # The keyboard: Enter opens onto Attach file, Esc closes back to "+", Space opens, Tab leaves.
                        page.focus(PLUS)
                        page.keyboard.press("Enter")
                        page.wait_for_selector(MENU, state="visible", timeout=5_000)
                        assert _facts(page)["focus"] == "item:attach", where
                        # The live install's own items follow (a skill's sections, the host's last): arrows move
                        # among them and wrap from the last back to Attach file.
                        page.keyboard.press("ArrowDown")
                        moved = _facts(page)["focus"]
                        assert moved.startswith("item:") and moved != "item:attach", f"{where}: arrows move among the items"
                        page.keyboard.press("End")
                        assert _facts(page)["focus"] == "item:make-button", where
                        page.keyboard.press("ArrowDown")
                        assert _facts(page)["focus"] == "item:attach", f"{where}: arrows stay among the items"
                        page.keyboard.press("Escape")
                        page.wait_for_selector(MENU, state="detached", timeout=5_000)
                        escaped = _facts(page)
                        assert escaped["focus"] == "plus" and escaped["expanded"] == "false", (where, escaped)
                        page.keyboard.press(" ")
                        page.wait_for_selector(MENU, state="visible", timeout=5_000)
                        page.keyboard.press("Tab")
                        page.wait_for_selector(MENU, state="detached", timeout=5_000)
                        evidence[where] = {"closed": closed, "opened": opened}
                        if theme == "light" and width == 360:
                            routes = page.evaluate(DROP_AND_PASTE)
                            evidence[where]["routes"] = routes
                            if engine == "chromium":
                                assert routes["dragSupported"] and routes["pasteSupported"], (where, routes)
                            if routes["dragSupported"]:
                                assert routes["dropped"] == 1, (where, routes)
                            if routes["pasteSupported"]:
                                assert routes["pasted"] == 1, (where, routes)
                    finally:
                        page.close()
        finally:
            (_evidence_dir(tmp_path) / f"composer-plus-{engine}.json").write_text(json.dumps(evidence, indent=1),
                                                                                 encoding="utf-8")
            browser.close()


CHIPS = """(count) => {
    const slot = document.querySelector('#page-chat [data-composer-chips]');
    for (let i = 0; i < count; i += 1) {
        const chip = document.createElement('button');
        chip.type = 'button';
        chip.className = 'chat-composer-chip';
        chip.textContent = `Contributed chip ${i + 1}`;
        slot.append(chip);
    }
    slot.hidden = false;
    const voice = document.querySelector('#page-chat [data-composer-voice]');
    const mic = document.createElement('button');
    mic.type = 'button';
    mic.className = 'voice-btn voice-mic';
    mic.textContent = 'Mic';
    voice.append(mic);
    voice.hidden = false;
}"""
ROW = """() => {
    const page = document.querySelector('#page-chat');
    const row = page.querySelector('.chat-toolbar-row');
    const slot = row.querySelector('[data-composer-chips]');
    const rect = (node) => node.getBoundingClientRect();
    const view = window.innerWidth;
    const style = getComputedStyle(slot);
    return {
        rowHeight: rect(row).height, rowRight: rect(row).right, compact: row.dataset.ownerCompact || '',
        hosts: [...row.children].filter((node) => node !== slot && rect(node).width > 0)
            .map((node) => [node.className, Math.round(rect(node).width * 2) / 2]),
        lines: new Set([...row.children].filter((node) => rect(node).width > 0).map((node) => Math.round(rect(node).top))).size,
        slot: { display: style.display, overflow: style.overflowX, left: rect(slot).left, right: rect(slot).right,
            clips: slot.scrollWidth > slot.clientWidth + 1 },
        chipsWhole: [...slot.children].every((chip) => chip.scrollWidth <= chip.clientWidth + 1),
        overflow: document.documentElement.scrollWidth - view,
        outside: [page.querySelector('[data-composer-voice] button'), page.querySelector('.chat-send-inline'),
            page.querySelector('#chat-plus')].filter(Boolean)
            .filter((node) => rect(node).right > view + 0.5 || rect(node).left < -0.5).map((node) => node.textContent.trim() || node.id),
    };
}"""


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_filled_slots_keep_the_capsules_line_and_step_aside_with_the_capsules(direct_server_with_data, engine, tmp_path):  # noqa: F811
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import sync_playwright

    url = direct_server_with_data["url"]
    evidence = {}
    with sync_playwright() as pw:
        browser, page = _launch(pw, engine, viewport={"width": 360, "height": 800})
        try:
            for width in (320, 360):
                for count in (1, 6):
                    where = f"{engine} {width}px {count} chip(s)"
                    _open_chat(page, url, width=width, height=800)
                    if count == 6:
                        page.locator('#chat-input').fill('one\ntwo\nthree')
                        page.wait_for_function("document.querySelector('#chat-input').getBoundingClientRect().height > 60")
                    _assert_plus_in_place(_facts(page), where)
                    if count == 1:
                        # The menu fits the narrowest phone.
                        page.click(PLUS)
                        page.wait_for_selector(MENU, state="visible", timeout=5_000)
                        menu = _facts(page)["menu"]
                        assert menu["left"] >= 0 and menu["right"] <= width + 0.5, (where, menu)
                        page.keyboard.press("Escape")
                        page.wait_for_selector(MENU, state="detached", timeout=5_000)
                    before = page.evaluate(ROW)
                    page.evaluate(CHIPS, count)
                    page.wait_for_timeout(250)
                    after = page.evaluate(ROW)
                    evidence[where] = {"before": before, "after": after}
                    # The capsule row never wraps and the host capsules keep their place and words: the chips
                    # take only the room left, clip what does not fit, and never shrink a chip.
                    assert after["rowHeight"] <= before["rowHeight"] + 0.5 and after["lines"] == 1, (where, before, after)
                    assert after["hosts"] == before["hosts"] and after["compact"] == before["compact"], (where, before, after)
                    # It clips sideways ("clip" keeps the chips' shadow below them where the engine supports it).
                    assert after["slot"]["display"] == "flex" and after["slot"]["overflow"] in ("hidden", "clip"), (where, after)
                    assert after["slot"]["right"] <= after["rowRight"] + 0.5, (where, after)
                    assert after["chipsWhole"], (where, after)
                    if count == 6:
                        assert after["slot"]["clips"], (where, after)
                    assert after["overflow"] <= 0 and not after["outside"], (where, after)
                    page.screenshot(path=str(_evidence_dir(tmp_path) / f"composer-slots-{engine}-{width}-{count}.png"),
                                    animations="disabled")
                    # An owner capsule open alone: the chips step aside with the other controls.
                    page.evaluate("() => { document.querySelector('#page-chat .chat-toolbar-row').dataset.ownerSolo = 'effort'; }")
                    assert page.evaluate(ROW)["slot"]["display"] == "none", where
        finally:
            (_evidence_dir(tmp_path) / f"composer-slots-{engine}.json").write_text(json.dumps(evidence, indent=1),
                                                                                  encoding="utf-8")
            browser.close()
