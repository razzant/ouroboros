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


# One glass and one shadow for what floats: each object below rests on the float shadow token, the toast and the staged
# file wear the float surface and the one glass with no visible rim (the staged file also its resting shadow, its edge on a
# light room), the status badge the one glass, and the three named veils soften with the one scrim filter. Every
# unconditioned rule naming the bare selector (alone or in a list) counts, so a later override is seen too.
FLOAT_SHADOW = (".toast", ".files-preview-image", ".files-drop-card", ".onboarding-restart-card",
                ".chat-file-dialog", ".document-reader", ".project-panel")
FLOAT_SURFACE = (".toast", ".attach-badge")
GLASS = (".toast", ".attach-badge", ".status-badge")
SCRIMS = (".onboarding-overlay-backdrop", ".files-drop-overlay", ".project-panel-backdrop")
_NO_RIM = re.compile(r"^(?:none|0|0px|transparent|\d+px solid transparent|0 none)$")


def _declarations(css: str, selector: str) -> dict[str, list[str]]:
    """Every value each property takes in the flat rules whose selector list names `selector` itself (not a descendant,
    a state or a theme), in sheet order."""
    found: dict[str, list[str]] = {}
    for selectors, body in _rules(css):
        if selector not in [part.strip() for part in selectors.split(",")]:
            continue
        for decl in body.split(";"):
            name, _, value = decl.partition(":")
            if value.strip():
                found.setdefault(name.strip(), []).append(" ".join(value.split()))
    return found


def floating_material_drift(css: str) -> list[str]:
    """Named floating objects whose rules paint a material other than the shared tokens."""
    found = []
    for selector in FLOAT_SHADOW:
        if set(_declarations(css, selector).get("box-shadow", ["missing"])) != {"var(--elev-float)"}:
            found.append(f"{selector} shadow")
    for selector in FLOAT_SURFACE:
        decls = _declarations(css, selector)
        rims = [v for name in ("border", "border-color") for v in decls.get(name, [])]
        if set(decls.get("background", ["missing"])) != {"var(--surface-float)"} or not all(_NO_RIM.match(v) for v in rims):
            found.append(f"{selector} surface")
    for selector in GLASS:
        if set(_declarations(css, selector).get("backdrop-filter", ["missing"])) != {"var(--glass)"}:
            found.append(f"{selector} glass")
    if set(_declarations(css, ".attach-badge").get("box-shadow", ["missing"])) != {"var(--elev-rest)"}:
        found.append(".attach-badge edge")
    for selector in SCRIMS:
        if set(_declarations(css, selector).get("backdrop-filter", ["missing"])) != {"var(--scrim-filter)"}:
            found.append(f"{selector} veil")
    return found


GOOD = (".toast { background: var(--surface-float); box-shadow: var(--elev-float); backdrop-filter: var(--glass);"
        " outline: 1px solid transparent; }"
        ".attach-badge { background: var(--surface-float); border: 1px solid transparent; box-shadow: var(--elev-rest);"
        " backdrop-filter: var(--glass); } .status-badge { backdrop-filter: var(--glass); }"
        ".files-preview-image { box-shadow: var(--elev-float); } .files-drop-card { box-shadow: var(--elev-float); }"
        ".onboarding-restart-card { box-shadow: var(--elev-float); }"
        ".chat-file-dialog, .document-reader { box-shadow: var(--elev-float); }"
        ".project-panel { box-shadow: var(--elev-float); }"
        ".onboarding-overlay-backdrop { backdrop-filter: var(--scrim-filter); }"
        ".files-drop-overlay { backdrop-filter: var(--scrim-filter); }"
        ".project-panel-backdrop { backdrop-filter: var(--scrim-filter); }")


def test_floating_material_scanner_sees_drift():
    assert floating_material_drift(GOOD) == []
    assert floating_material_drift(GOOD + ".toast:hover { border: 1px solid red; }") == [], "a state is not the object"
    for broken, expected in (
            (".toast { border: 1px solid var(--divider); }", [".toast surface"]),           # a plain rim, later
            (".toast { box-shadow: 0 2px 4px red; }", [".toast shadow"]),                   # an override at the end
            (".status-badge { backdrop-filter: blur(8px); }", [".status-badge glass"]),
            (".attach-badge { box-shadow: none; }", [".attach-badge edge"]),
            (".x, .files-drop-overlay { backdrop-filter: blur(3px); }", [".files-drop-overlay veil"])):
        assert floating_material_drift(GOOD + broken) == expected, broken


def test_floating_objects_wear_the_shared_material():
    from pathlib import Path

    css = (Path(__file__).resolve().parents[1] / "web" / "style.css").read_text(encoding="utf-8")
    assert floating_material_drift(css) == []
