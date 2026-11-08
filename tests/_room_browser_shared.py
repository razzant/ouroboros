"""Project room browser helpers the room tests share: a room seeded as the server stores it, a client opening the
app, a room opened and turned back into older history, a page reached through the navigation.

Shared by tests/test_project_unread_read_band_browser.py, tests/test_room_return_browser.py,
tests/test_room_suspension_browser.py and tests/test_secondary_place_browser.py: a test module is not imported by
another test module (docs/development/02-naming-and-boundaries.md).
"""
from __future__ import annotations

import json

from ouroboros import dialogue_store
from tests import _dialogue as dialogue_test
from tests._dialogue import advance_room_revision
from tests.ui_chat_viewport_smoke import _CAPTURE_TEST_SOCKET, _SETTLE_RESTORE_FRAMES


def _seed_room(data, project_id, name, rows, archives=()):
    """A Project room holding ``rows`` as the server stores them, after ``archives``: an older
    install's chain segments ``(name, rows)`` (a raw string is a line as written), imported first."""
    from ouroboros.projects_registry import create_project

    chat_id = create_project(data, project_id, name=name)["chat_id"]
    if archives:
        dialogue_test.seed(data, archives={segment: [line if isinstance(line, str) else {**line, "chat_id": chat_id}
                                                     for line in lines] for segment, lines in archives})
        dialogue_test.import_all(data)
    for row in rows:
        dialogue_store.append(data, "chat", {**row, "chat_id": chat_id})
    return chat_id, advance_room_revision(data, chat_id=chat_id)["visible_revision"]


def _unread(row):
    """A Project row's unread fact: its own attribute, never its mark's colour, which the owner's amber outranks."""
    return row.and_(row.page.locator("[data-unread]"))


def _client(browser, url, project_id, acks, init=(), **page_options):
    """One client showing the room's row unread; ``acks`` collects its read receipts."""
    page = browser.new_page(viewport={"width": 1187, "height": 734}, **page_options)
    page.on("request", lambda request: acks.append(json.loads(request.post_data or "{}"))
            if request.method == "POST" and request.url.endswith("/api/ui/preferences") else None)
    for script in (_CAPTURE_TEST_SOCKET, *init):
        page.add_init_script(f"({script})()")
    page.goto(url, wait_until="domcontentloaded")
    page.wait_for_function("() => window.__testSockets?.some(s => s.readyState === 1)")
    row = page.locator(f'.nav-project-row[data-project-id="{project_id}"]')
    _unread(row).wait_for(state="attached")
    return page, row


def _wait_for(page, check, attempts=100):
    for _ in range(attempts):
        if check():
            return True
        page.wait_for_timeout(100)
    return check()


_ROW_AT_TOP = """pid => {
    const feed = document.querySelector(`[id="panel-pchat-${pid}"] .chat-messages`);
    const top = feed.getBoundingClientRect().top;
    const node = [...feed.querySelectorAll('[data-history-id]')].find(n => n.getBoundingClientRect().bottom > top + 1);
    return node && {id: node.dataset.historyId, offset: Math.round(node.getBoundingClientRect().top - top)};
}"""
_SELECT = """([pid, text]) => {
    const feed = document.querySelector(`[id="panel-pchat-${pid}"] .chat-messages`);
    const node = [...feed.querySelectorAll('[data-history-id] .message')].find(n => n.textContent.includes(text));
    const textNode = document.createTreeWalker(node, NodeFilter.SHOW_TEXT).nextNode();
    const range = document.createRange();
    range.setStart(textNode, 0);
    range.setEnd(textNode, Math.min(9, textNode.length));
    const selection = getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    return selection.toString();
}"""


def _rows(prefix, count):
    return [{"ts": f"2026-09-28T{10 + index // 60:02d}:{index % 60:02d}:00Z", "direction": "out",
             "text": f"{prefix} row {index:02d} " + "words " * 12} for index in range(count)]


def _open(page, pid):
    page.locator(f'.nav-project-row[data-project-id="{pid}"]').evaluate("el => el.click()")
    page.locator(f'[id="panel-pchat-{pid}"]:not([hidden])').wait_for()
    page.wait_for_function("() => !document.querySelector('.chat-instance-panel[data-leaving]')")
    page.locator(f'[id="panel-pchat-{pid}"] [data-history-id]').first.wait_for(state="attached")
    page.evaluate(_SETTLE_RESTORE_FRAMES)


def _read_older(page, pid):
    """The owner's own wheel turns the room back into older history (a gesture, not the app's write)."""
    box = page.locator(f'[id="panel-pchat-{pid}"] .chat-messages').bounding_box()
    page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    for _ in range(4):
        page.mouse.wheel(0, -500)
        page.wait_for_timeout(120)
    page.evaluate(_SETTLE_RESTORE_FRAMES)
    page.wait_for_timeout(300)
    return page.evaluate(_ROW_AT_TOP, pid)


def _nav(page, name):
    page.locator(f'[data-nav-page="{name}"]').evaluate("el => el.click()")
    page.wait_for_function("name => document.getElementById(`page-${name}`)?.classList.contains('active')", arg=name)
    page.evaluate(_SETTLE_RESTORE_FRAMES)
