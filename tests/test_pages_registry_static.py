"""The page registry (docs/DESIGN.md "Shell", plan P9/P11): one list of places renders the sidebar's rows,
names the stack layout's header titles, and matches app.js's page init map and the page sections one to one.
No nav markup or glyph copy lives in index.html; the shell's glyphs come from page_icons.js once."""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# Where a page's section is mounted and where its header is rendered.
MODULES = {"chat": ("chat.js", "chat.js"), "circles": ("circles.js", "circles.js"), "files": ("files.js", "files.js"),
           "skills": ("skills.js", "skills.js"), "widgets": ("widgets.js", "widgets.js"),
           "dashboard": ("dashboard.js", "dashboard.js"), "settings": ("settings.js", "settings.js")}


def _read(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


def _registry():
    source = _read("web/modules/pages.js")
    rows = re.findall(r"\{ id: '([a-z]+)', label: '([^']+)', icon: '([a-z]+)', zone: '([a-z]+)'(?:, hint: '[^']+')?, init: '([A-Za-z]+)' \}", source)
    assert rows, "the registry rows must stay readable to this parity guard"
    return rows


def test_registry_matches_the_init_map_and_the_page_sections():
    app_js = _read("web/app.js")
    # A page may take more than the shared context (Files takes the shell's layer stack): `initFiles({ ...ctx, layers })`.
    inits = set(re.findall(r"\b(init[A-Z][A-Za-z]+)\((?:ctx\)|\{ \.\.\.ctx[,}])", app_js))
    rows = _registry()
    assert [r[0] for r in rows] == ["chat", "circles", "files", "skills", "widgets", "dashboard", "settings"]
    for page_id, label, _icon, _zone, init in rows:
        assert init in inits, f"{page_id}: app.js never calls {init}(ctx)"
        mount, header = (_read(f"web/modules/{name}") for name in MODULES[page_id])
        assert f"page-{page_id}" in mount, f"{page_id}: its module mounts no page-{page_id} section"
        if page_id == "chat":
            assert "title = pageLabel('chat')" in header, "Main's header title is the registry label"
        else:
            # The header names the place by its registry id, so the title IS the label (no second copy to drift).
            assert f"page: '{page_id}'" in header, f"{page_id}: the header must read its title from the registry"
            assert f"title: '{label}'" not in header, f"{page_id}: a literal title copies the registry label"
    # Every init app.js calls for a page is in the registry (the shell's pages, not the Dashboard's panels).
    page_inits = {init for init in inits if init in {r[4] for r in rows}}
    assert page_inits == {r[4] for r in rows}
