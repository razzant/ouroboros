"""One header for every page and the Project panel (docs/DESIGN.md "Shell", plan P5, owner Q3=A, Q4=A, 19.10 A).

Docked: no visible title where the sidebar names the page (the h2 stays); the page's tabs or location take its place
and its actions the end; the Project keeps its name, and a header with nothing to show renders no bar. Stack: one
row of at most 56px, `title · status · consciousness … Panic`, tabs under it, the title one line and ellipsized first;
Restart and the budget in the status Shade. No ⋯, no More, no icon for a place."""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


def _rule(css: str, selector: str) -> str:
    assert selector in css, selector
    return css.split(selector, 1)[1].split("}", 1)[0]


def test_docked_header_hides_a_title_the_sidebar_names_and_an_empty_bar():
    css = _read("web/style.css")
    hidden = _rule(css, 'html[data-layout="docked"] .app-page-header[data-sidebar-names] .app-page-title {')
    assert "position: absolute" in hidden and "clip-path: inset(50%)" in hidden, "an accessible h2, not a removed one"
    assert "display: none" in _rule(css, 'html[data-layout="docked"] .app-page-header[data-empty] {')
    title = _rule(css, ".app-page-title {")
    assert "font-size: var(--type-section)" in title and "font-weight: 500" in title
    assert "white-space: nowrap" in title and "text-overflow: ellipsis" in title
    assert ".app-page-icon" not in css and "app-page-description" not in css
    assert ".app-page-lede" not in css, "a page has no caption line"


def test_stack_header_is_one_row_of_at_most_56px_with_restart_and_budget_in_the_status_shade():
    css = _read("web/style.css")
    row = _rule(css, 'html[data-layout="stack"] .app-page-header {')
    # 44px controls plus the 5px chrome inset above and below: 54, held at 56 by the min-height.
    assert "min-height: 56px" in row and "padding: var(--chrome-inset) var(--space-3)" in row, row
    assert "var(--inset-top)" in row
    assert "flex-wrap: nowrap" in _rule(css, 'html[data-layout="stack"] .chat-page-header {')
    assert "flex-wrap: nowrap" in _rule(css, 'html[data-layout="stack"] .chat-header-actions {')
    assert "chat-header-more" not in css and "chat-header-menu" not in css, "no More, no ⋯"
    room_work = _read("web/room_work.css")
    assert ".rw-extras:empty { display: none; }" in room_work
    assert ":root[data-theme] .rw-extras > :is(.chat-header-btn, .chat-budget-pill).ui-chrome-control {" in room_work
    # The note is a floating sibling under the bar, never a second row of it; the move into the Shade follows the
    # bar's measured width as well as the layout (page_header.js), so a narrow docked column keeps one row too.
    assert '.chat-page-note {' in css and "top: var(--chat-header-bar, 56px)" in _rule(css, ".chat-page-note {")
    assert re.search(r"createHistoryControls\(messagesDiv, pageNote \|\| page\.querySelector\('\.chat-panel-statusbar'\)[,)]",
                     _read("web/modules/chat.js"))
    header_js = _read("web/modules/page_header.js")
    assert "export const HEADER_FULL_ROW_MIN_WIDTH = 640;" in header_js
    assert "bar.clientWidth < HEADER_FULL_ROW_MIN_WIDTH" in header_js and "new Observer(place)" in header_js
    # The ☰ and the × share the shell's one icon-control recipe; the Project's × is that recipe too.
    assert 'class="shell-icon-control project-panel-close"' in _read("web/app.js")
    header_js = _read("web/modules/page_header.js")
    assert "'shell-icon-control mobile-nav-toggle composer-menu ui-hit-area' : 'shell-icon-control mobile-nav-toggle'" in header_js
    assert ".project-panel-title" not in css, "one title style"


def test_header_actions_take_the_chrome_height_under_a_finger():
    css = _read("web/style.css")
    gated = css.split("@media (pointer: coarse) {\n    .app-page-toolbar :is(.btn, button),", 1)[1].split("}", 1)[0]
    assert "min-height: var(--chrome-control-h)" in gated
    assert re.search(r"@media \(pointer: coarse\) \{[^}]*--chrome-control-h: var\(--touch-target-min\)", _read("web/ui.css"), re.S)


def test_the_place_name_wins_over_the_status_word_and_the_status_keeps_its_target():
    room_work = _read("web/room_work.css")
    trigger = _rule(room_work, ":is(.chat-page-header, .project-panel-bar) .app-page-title-row > .rw-trigger {")
    assert "flex: 0 1000 auto" in trigger, "the status yields a thousand times faster than the title"
    assert "min-width: var(--touch-target-min)" in trigger and "5ch" not in trigger
    title = _rule(room_work, ":is(.chat-page-header, .project-panel-bar) .app-page-title-row > .app-page-title {")
    assert "flex: 0 1 auto" in title and "min-width: 0" in title and "text-overflow: ellipsis" in title


def test_main_title_is_the_registry_label_and_the_panel_bar_keeps_its_hooks():
    chat_js = _read("web/modules/chat.js")
    assert "asPanel = false, title = pageLabel('chat')," in chat_js
    assert "PAGE_ICONS" not in chat_js
    # room_work.js still finds the Project bar and Main's header to seat the status after the title.
    room_work = _read("web/modules/room_work.js")
    assert "panel?.querySelector?.('.project-panel-bar')" in room_work
    assert "bar?.querySelector('.app-page-title-row')" in room_work


def test_the_status_word_clips_past_its_line_so_descenders_stay_whole():
    # The word ellipsizes (overflow hidden); its clip box reaches past the line and the margin gives the room back,
    # so the tail of g, p or y is never cut ("Working" lost its g).
    badge = _rule(_read("web/room_work.css"), ".rw-trigger .status-badge {")
    assert "overflow: hidden" in badge and "text-overflow: ellipsis" in badge
    assert "padding-block: 0.15em" in badge and "margin-block: -0.15em" in badge


def test_the_desktop_title_band_over_an_empty_header_is_the_windows_handle_not_a_row():
    # Owner 22.10: the empty strip an empty header kept for the band cut the Widgets board. The band is the window's:
    # the empty header lies over the top of its page as the window's handle (out of the flow, above the content, clear
    # of the scrollbar), and the page's scroll body starts at the top edge with its first row padded below the band.
    css = _read("web/style.css")
    band = _rule(css, ':root[data-window-chrome="integrated"][data-layout="docked"] .app-page-header[data-empty] {')
    assert "position: absolute" in band and "inset: 0 6px auto 0" in band and "z-index: 2" in band, band
    assert "height: var(--window-chrome-top)" in band and "padding: 0" in band, band
    board = _rule(css, ':root[data-window-chrome="integrated"][data-layout="docked"] .app-page-header[data-empty] ~ .widgets-scroll {')
    assert "padding-top: calc(var(--space-4) + var(--window-chrome-top))" in board, board
    # Outside the desktop window an empty header still takes no room.
    assert "display: none" in _rule(css, 'html[data-layout="docked"] .app-page-header[data-empty] {')


def test_a_desk_bar_starts_with_its_pages_navigation_and_ends_with_its_actions():
    # Owner 19.10 (A): on a desk the tabs or Files' location take the hidden title's place and the actions keep the
    # end, every control on Main's line (the band's centre in the macOS window); the stack layout keeps them under
    # the row. The nodes move (page_header.js bindPageHeader), so selection, listeners and focus stay theirs.
    css = _read("web/style.css")
    bar = _rule(css, 'html[data-layout="docked"] .app-page-header[data-lead] {')
    assert 'grid-template-areas: "title toolbar"' in bar and "padding-block: 0" in bar and "min-height: 50px" in bar
    # The band's height only where the shell measured it (tests/test_launcher_window_chrome.py: no phantom spacer).
    # At every Interface size the row is the band and the tabs' top inset shrinks with it (10px at 100 %), so the
    # tabs stay level with the traffic lights where a 50px floor left them below once the band shrank.
    band = _rule(css, ':root[data-window-chrome="integrated"][data-layout="docked"] .app-page-header[data-lead] {')
    assert "min-height: var(--window-chrome-top)" in band and "50px" not in band, band
    assert "--lead-inset: min(10px, max(0px, calc((var(--window-chrome-top) - var(--chrome-control-h)) / 2)))" in band
    inset = _rule(css, ':root[data-window-chrome="integrated"][data-layout="docked"] .app-page-lead .app-tab-strip {')
    # Both insets shrink together, so the actions centred in the row stay level with the tabs.
    assert "padding-top: var(--lead-inset)" in inset and "padding-bottom: calc(16px + var(--lead-inset))" in inset
    controls = _rule(css, 'html[data-layout="docked"] .app-page-header[data-lead] > .app-page-toolbar :is(.btn, button) {')
    tab = _rule(css, 'html[data-layout="docked"] .app-page-lead .app-tab {')
    assert "height: var(--chrome-control-h)" in controls and "height: var(--chrome-control-h)" in tab
    # The strip lends its tabs' shadows room below the bar without taking presses there.
    strip = _rule(css, 'html[data-layout="docked"] .app-page-lead .app-tab-strip {')
    assert "pointer-events: none" in strip and "margin: 0 0 -16px" in strip and "pointer-events: auto" in tab
    header_js = _read("web/modules/page_header.js")
    assert "export function bindPageHeader(" in header_js and "toggleAttribute('data-lead', docked)" in header_js
    for rel in ("web/modules/dashboard.js", "web/modules/settings_ui.js", "web/modules/skills.js", "web/modules/files.js"):
        assert "bindPageHeader(" in _read(rel), rel
    # Widgets has nothing of its own for the bar: its list starts one inset under it, never two.
    assert "margin-top" not in _rule(css, ".widgets-list {")
