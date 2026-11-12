"""The shell has one layout switch (docs/DESIGN.md "Shell", plan P1).

`web/theme.js` writes `html[data-layout="docked"|"stack"]` before CSS from one number pair; the
shell's sheets key off that attribute, and the shell's JS reads it through `web/modules/layout.js`.
A second width breakpoint for the shell (a media query in the sheets, a `matchMedia` in app code)
is the drift this guard closes: CSS and JS used to disagree about where "mobile" starts (640,
900 and 980 at once)."""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# The shell's structure: what the switch moves. Component sheets may keep their own width queries.
SHELL_SELECTORS = (
    "#app", "#primary-sidebar", ".nav-drawer-backdrop", ".mobile-nav-toggle", ".project-panel",
    ".project-panel-backdrop", ".resize-handle", ".app-page-header", ".page-header", "#nav-version",
    "#content", ".nav-row", ".nav-projects", ".sidebar-", ".chat-page-header", "keyboard-open",
)
WIDTH_QUERY = re.compile(r"@media[^{]*\((?:max|min)-width:[^{]*\{")


def _read(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


def _media_blocks(css: str):
    """Yield the body of every width media query (balanced braces)."""
    for match in WIDTH_QUERY.finditer(css):
        depth, start = 1, match.end()
        pos = start
        while depth and pos < len(css):
            if css[pos] == "{":
                depth += 1
            elif css[pos] == "}":
                depth -= 1
            pos += 1
        yield match.group(0), css[start:pos]


def test_shell_sheets_key_structure_off_the_layout_attribute_not_width_queries():
    for rel in ("web/style.css", "web/ui.css"):
        for head, body in _media_blocks(_read(rel)):
            offenders = [sel for sel in SHELL_SELECTORS if sel in body]
            assert not offenders, f"{rel}: {head.strip()} styles shell structure {offenders}; key it off html[data-layout]"
    css = _read("web/style.css")
    assert 'html[data-layout="stack"] #app' in css
    assert 'html[data-layout="stack"] #primary-sidebar' in css
    assert 'html[data-layout="stack"] .project-panel' in css
    # The docked sidebar keeps its hairline beside the room; the stack drawer lies under the room card, whose curved
    # edge is its edge, so it has none (owner 12.11: a straight line doubled the card's edge).
    desk = css.split("\n#primary-sidebar {", 1)[1].split("}", 1)[0]
    assert "border-right: 1px solid" in desk, desk
    drawer = css.split('\nhtml[data-layout="stack"] #primary-sidebar {', 1)[1].split("}", 1)[0]
    assert "border-right: 0;" in drawer, drawer
    # Docked, the Project panel keeps its own width beside Main and never covers the sidebar (owner 18.10, 4A): it
    # is placed in the columns after the sidebar's, and its cap is that area.
    placed = css.split('html[data-layout="docked"] .project-panel-backdrop,\nhtml[data-layout="docked"] .project-panel {', 1)[1]
    assert "position: absolute;" in placed.split("}", 1)[0] and "grid-column: 2 / -1;" in placed.split("}", 1)[0]
    capped = placed.split('html[data-layout="docked"] .project-panel {', 1)[1].split("}", 1)[0]
    assert "max-width: 100%;" in capped
    assert "data-project-fill" not in css and "data-project-fill" not in _read("web/app.js")
    # What a drag writes, and when, is behaviour: tests/test_shell_interactions_browser.py drags both handles (the
    # stored widths change once, at the end; a Project closed mid-drag keeps its width).


def test_a_finger_is_the_switchs_other_fact_and_opens_a_project_in_mains_place():
    """The primary input (owner 06.11): theme.js paints `html[data-pointer]` from one `(pointer: coarse)` query beside the
    layout, and only the shell's sheets read it; docked under a finger the Project takes the content column with no
    scrim and no width handle of its own (the sidebar keeps its). It is an input fact, never a width rule: a desk with a
    mouse keeps the panel beside Main at its own width (owner 18.10, 4A). The painting and the door are behaviour
    (web/tests/layout.test.js); the rendered room is tests/test_shell_project_open_browser.py."""
    css = _read("web/style.css")
    whole = css.split('html[data-layout="docked"][data-pointer="coarse"] .project-panel {', 1)[1].split("}", 1)[0]
    assert "width: 100%;" in whole and "box-shadow: none;" in whole and "border-left: none;" in whole, whole
    hidden = css.split('html[data-layout="docked"][data-pointer="coarse"] .project-panel-backdrop,', 1)[1].split("}", 1)[0]
    assert 'html[data-layout="docked"][data-pointer="coarse"] .project-panel > .resize-handle {' in hidden, hidden
    assert "display: none;" in hidden, hidden
    # The shell asks the door; the pointer queries outside theme.js are a closed list that only shrinks (a component's
    # own fact: the surface packet, a row's long press, the reader, the Interface size), never a new reader.
    readers = sorted(path.relative_to(REPO).as_posix() for path in (REPO / "web").glob("**/*.js")
                     if "tests" not in path.parts and "vendor" not in path.parts
                     and "(pointer: coarse)" in path.read_text(encoding="utf-8"))
    assert set(readers) <= {"web/theme.js", "web/modules/client_surface.js", "web/modules/chat_row_menu.js",
                            "web/modules/document_reader.js", "web/modules/interface_size.js"}, readers


def test_mains_count_and_read_follow_one_predicate():
    """While a Project is open Main is not the room on screen (owner 06.11; DESIGN "Unread"): its row counts its new
    messages and nothing reads them until Main is back. The count, ☰'s dot and the read ask the same `mainOnScreen()`,
    so a covered Main can neither hide its number nor be read behind the Project. The read is behaviour
    (web/tests/main_unread_counter.test.js, project_history_ack.test.js); the count and the dot have no other home in
    the default lane (the browser case, tests/test_shell_project_open_browser.py, is opt-in)."""
    app = _read("web/app.js")
    badge = app.split("function updateUnreadBadge()", 1)[1].split("\n}\n", 1)[0]
    assert "mainOnScreen() ? 0 : mainUnreadCount()" in badge, badge
    assert "(!mainOnScreen() && mainUnreadCount() > 0)" in app
