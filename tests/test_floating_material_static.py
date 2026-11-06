"""A floating surface carries its own material (docs/DESIGN.md principle 2).

Two defects of one class shipped: the work list kept the room's ink after its bar's backing was removed, and the
Official explanation (a ``role="dialog"`` popup) had no paint because the popup material was selected by
``role="menu"``. The class is "a surface's paint belongs to another surface or to its ARIA role", so this guard reads the
first-party sheets as text and refuses exactly that shape, with scanner examples pinned so a miss shows up here first.
A third shape of the same promise: a menu or dialog painted with the translucent float colour, which let a row's menu
show the words under it. Real rendering is in ``tests/test_work_list_panel_browser.py`` and
``tests/test_floating_material_browser.py``.
"""
from __future__ import annotations

import re
PAINT = ("background", "box-shadow", "backdrop-filter", "-webkit-backdrop-filter")


def _rules(css: str):
    """``(selector, body)`` of every flat rule, nested at-rules flattened, comments dropped."""
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)
    for match in re.finditer(r"([^{}]+)\{([^{}]*)\}", css):
        yield " ".join(match.group(1).split()), match.group(2)


def _declares(body: str, props=PAINT) -> list[str]:
    return [name for name in props if re.search(rf"(?:^|[;\s]){re.escape(name)}\s*:", body)]


def role_keyed_popup_paint(css: str) -> list[str]:
    """Popup rules that choose paint by an ARIA role instead of by being a popup."""
    return [f"{selector} → {', '.join(found)}" for selector, body in _rules(css)
            for found in [_declares(body)]
            if found and re.search(r"\.ui-popup(?:\.[\w-]+)*\[role", selector)]


def borrowed_room_ink(css: str) -> list[str]:
    """Rules that paint the work list with the room's own ink (what a continuous bar once made right)."""
    return [selector for selector, body in _rules(css)
            if re.search(r"\.rw-(shade|surface)\b", selector) and re.search(r"background[^;]*--shell-rgb", body)]


def translucent_popup_paint(css: str) -> list[str]:
    """Menus and dialogs painted with a translucent float colour, so the words under them read through."""
    return [selector for selector, body in _rules(css)
            if re.search(r"\.(?:ui-popup|[\w-]+-menu|[\w-]+-modal)(?![\w-])", selector)
            and re.search(r"(?:^|[;\s])background\s*:\s*var\(--(?:surface|chip)-float\)\s*(?:;|$)", body)]


def test_scanners_see_the_shape_they_refuse():
    assert role_keyed_popup_paint('.ui-popup[role="menu"] { padding: 4px; background: var(--surface-float); }') == [
        '.ui-popup[role="menu"] → background']
    assert role_keyed_popup_paint('.ui-popup { background: var(--surface-float); } '
                                  '.ui-popup[role="menu"] { display: flex; width: max-content; }') == []
    assert borrowed_room_ink('.chat-page-header .rw-shade { background: rgb(var(--shell-rgb)); }') == [
        '.chat-page-header .rw-shade']
    assert borrowed_room_ink('.rw-shade { background: var(--sheet-float); }') == []
    assert translucent_popup_paint(':root[data-theme] :is(.chat-header-menu, .task-control-menu) '
                                   '{ border-color: transparent; background: var(--surface-float); }') == [
        ':root[data-theme] :is(.chat-header-menu, .task-control-menu)']
    assert translucent_popup_paint('.task-control-menu { background: var(--popup-float); } '
                                   '.chat-header-menu-item { background: var(--surface-float); }') == []
