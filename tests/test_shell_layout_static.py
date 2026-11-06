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
    # Docked, the Project panel keeps its own width beside Main and never covers the sidebar (owner 18.10, 4A): it
    # is placed in the columns after the sidebar's, and its cap is that area.
    placed = css.split('html[data-layout="docked"] .project-panel-backdrop,\nhtml[data-layout="docked"] .project-panel {', 1)[1]
    assert "position: absolute;" in placed.split("}", 1)[0] and "grid-column: 2 / -1;" in placed.split("}", 1)[0]
    capped = placed.split('html[data-layout="docked"] .project-panel {', 1)[1].split("}", 1)[0]
    assert "max-width: 100%;" in capped
    assert "data-project-fill" not in css and "data-project-fill" not in _read("web/app.js")
    # What a drag writes, and when, is behaviour: tests/test_shell_interactions_browser.py drags both handles (the
    # stored widths change once, at the end; a Project closed mid-drag keeps its width).
