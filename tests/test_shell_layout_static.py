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


SECTION_PAGE_ALONE = "@container sectionplace (max-width: calc(40rem - 0.02px)) {"


def test_layers_is_the_only_browser_history_entry_writer():
    modules = {path.name: path.read_text(encoding='utf8') for path in (REPO / 'web/modules').glob('*.js')}
    assert sorted(name for name, source in modules.items() if re.search(r'\bhistory\.pushState\(', source)) == ['layers.js']
    replacements = sorted(name for name, source in modules.items() if re.search(r'\bhistory\.replaceState\(', source))
    assert replacements == ['layers.js', 'read_ahead.js']
    # The residual observer-mode handoff only consumes its boot parameter, preserving the current sentinel state.
    assert "history.replaceState(history.state, '', address)" in modules['read_ahead.js']
    assert not re.search(r'\bhistory\.(?:pushState|back|go)\(', modules['read_ahead.js'])
    assert not re.search(r'\bhistory\.(?:pushState|replaceState)\(', _read('web/app.js'))


def _block(css: str, head: str) -> str:
    """Bodies governed by this query, including rules beside their component (balanced braces)."""
    assert head in css, head
    bodies, end = [], 0
    while (index := css.find(head, end)) >= 0:
        start = index + len(head)
        depth, end = 1, start
        while depth and end < len(css):
            depth += {"{": 1, "}": -1}.get(css[end], 0)
            end += 1
        bodies.append(css[start:end - 1])
    return "\n".join(bodies)


def test_folded_rings_leave_each_others_touch_reach():
    css = _read("web/style.css")
    rings = _block(css, '.chat-toolbar-row[data-owner-glyphs="true"] {')
    assert "--composer-glyph-gap: max(var(--composer-row-gap), calc(var(--touch-target-min) - var(--composer-control-height)));" in rings
    assert "column-gap: var(--composer-glyph-gap);" in rings
    # Only a measured tight line gives up the extra reach; both states share the chips' matching give-back below.
    tight = _block(css, '.chat-toolbar-row[data-owner-glyphs="true"][data-owner-tight] {')
    assert "--composer-glyph-gap: var(--composer-row-gap);" in tight
    # The chips slot gives back that same wider gap, or a folded row with chips would wrap onto a second line.
    chips = _block(css, '.chat-toolbar-row[data-owner-glyphs="true"] > .chat-composer-chips {')
    assert "margin-inline-start: calc(-1 * var(--composer-glyph-gap));" in chips
    lead = _block(css, '.chat-toolbar-row[data-owner-glyphs="true"] > .chat-composer-chips::before {')
    assert "width: calc(var(--composer-glyph-gap) - 6px);" in lead


def test_logs_long_filter_word_never_crowds_its_neighbours_or_clear():
    """The longest filter (Consciousness, wider in some system faces) widens no neighbour and never pushes Clear: the
    filters are one sideways rail that scrolls beside a Clear that never shrinks, and on a page alone the rail takes
    its own full row under Clear. (This guard once read the two-row chip grid the band's rail replaced.)"""
    css = _read("web/style.css")
    assert "flex: none;" in _block(css, ".logs-clear-list {")
    assert "flex: 1;" in _block(css, ".logs-kinds {")
    strip = _block(css, ".app-tab-strip {")
    assert "flex-wrap: nowrap;" in strip and "overflow-x: auto;" in strip and "min-width: 0;" in strip
    assert 'class="logs-kinds app-tab-strip"' in _read("web/modules/logs.js")
    alone = _block(css, SECTION_PAGE_ALONE)
    assert 'html[data-layout="stack"] #page-dashboard .logs-kinds { grid-area: 2 / 1 / auto / -1;' in alone
    assert 'html[data-layout="stack"] #page-dashboard .logs-clear-list { grid-area: 1 / 3; }' in alone


def test_an_automatic_section_page_keeps_its_list_when_the_container_narrows_before_javascript():
    css = _read("web/ui.css")
    two = _block(css, "@container sectionplace (min-width: 40rem) {")
    assert '.section-shell[data-section-view="auto"] > .section-pane { display: grid; }' in two
    assert '.section-place[data-section-view="auto"] .section-lead-page { display: flex; }' in two
    # Only an explicit page hides the rail; auto shows the list immediately below the two-pane query.
    assert '.section-shell[data-section-view="page"] > .section-rail,\n.section-shell:is([data-section-view="home"], [data-section-view="auto"]) > .section-pane { display: none; }' in css.replace(two, "")
    phone = _block(_read("web/style.css"), SECTION_PAGE_ALONE)
    assert '#page-dashboard[data-section-view="auto"] > .app-page-header .app-page-lead { display: none; }' in phone


def test_a_section_gives_the_places_control_up_only_where_a_page_replaced_its_list():
    """A level's Back replaces the picture only when its page replaced the list. A jump independently owes a
    return in either layout; its writer marks the header, so its single capsule also replaces the picture."""
    for rel, page in (("web/style.css", "#page-dashboard"), ("web/settings.css", "#page-settings")):
        css = _read(rel)
        hide = f'html[data-layout="stack"] {page}[data-section-view="page"] > .app-page-header > .app-page-leading'
        inside = _block(css, SECTION_PAGE_ALONE)
        assert hide in inside, f"{rel}: a page that replaced the list keeps one leading control, its Back"
        assert css.count(hide) == inside.count(hide), f"{rel}: the level rule belongs below 40rem"
    css = _read("web/style.css")
    assert 'html[data-layout="stack"] .app-page-header[data-return] > .app-page-leading { display: none; }' in css
    assert "@container sectionplace (min-width: 40rem) {" in _read("web/ui.css"), "the two panes start where the query above ends"


def test_a_touch_window_keeps_a_sections_two_columns_in_one_header_row_and_an_empty_lead_draws_nothing():
    """DESIGN:3293,3345,3468: two panes from 40rem still have one header row. Places takes room from the rail's lead;
    the section name remains available to readers. Empty captions cannot keep a second row alive; Settings puts
    Search in its list, with its one round Back and level in the header when owed."""
    two = _block(_read("web/ui.css"), "@container sectionplace (min-width: 40rem) {")
    header = 'html[data-layout="stack"] .section-place > .app-page-header'
    assert f'{header} {{ grid-template-areas: "leading lead lead"; }}' in two
    assert f"{header} > .app-page-title-block {{ display: contents; }}" in two
    title = _block(two, f"{header} .app-page-title-row {{")
    assert "position: absolute;" in title and "clip-path: inset(50%);" in title
    assert f"{header} .app-page-lead {{ grid-area: lead; min-height: var(--touch-target-min); }}" in two
    lead = _block(two, 'html[data-layout="stack"] .section-place:is([data-section-view="page"], [data-section-view="auto"]) .section-lead {')
    assert "margin-inline-start: 0;" in lead
    assert ("grid-template-columns: calc(var(--section-rail-w) - var(--section-rail-end) - var(--space-3) - 44px - var(--space-2)) minmax(0, 1fr);") in lead
    assert '"leading title toolbar" "lead lead lead"' not in _read("web/settings.css")
    assert "padding: var(--chrome-inset) var(--space-3);" in _read("web/style.css"), "the stack header's inset the lead counts"
    css = _read('web/style.css')
    for selector in ['html[data-layout="docked"] .app-page-header[data-empty]:not([data-return]) {',
                     'html[data-layout="stack"] #page-settings > .app-page-header:not([data-return]) .app-page-lead {']:
        hidden = _block(css, selector)
        assert 'display: none' not in hidden and 'visibility: hidden' not in hidden
        assert 'position: absolute;' in hidden and 'clip-path: inset(50%);' in hidden, 'the heading remains accessible'
    dashboard = _block(_block(css, ".dashboard-row-mark[hidden] { display: none; }\n@container sectionplace (min-width: 40rem) {"),
                       'html[data-layout="stack"] #page-dashboard > .app-page-header:not([data-controls]):not([data-return]) .app-page-lead {')
    assert 'display: none' not in dashboard and 'position: absolute;' in dashboard and 'clip-path: inset(50%);' in dashboard, \
        "an empty two-pane band keeps the page's heading for readers"
    assert '#page-settings > .app-page-header .app-page-lead { display: flex;' in _read('web/settings.css')
