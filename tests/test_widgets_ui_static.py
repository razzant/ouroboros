"""Static contract checks for the Widgets page renderer."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _widgets_js() -> str:
    return (REPO_ROOT / "web" / "modules" / "widgets.js").read_text(
        encoding="utf-8"
    )


def _render_js() -> str:
    """The declarative renderer, split out of widgets.js for its second consumer."""
    return _read("web/modules/widget_render.js")


def _framed_widget_sources() -> str:
    """widgets.js (page host, dispatcher) plus the declarative renderer split
    out of it (widget_render.js), the framed mounts (widget_module.js), the
    in-frame bootstrap (widget_frame.js and its classic bridge,
    widget_frame_bridge.js), the framed-card chrome
    (widget_card.js), the card reorder handles (widget_reorder.js), the
    declarative chart helpers (widget_chart.js) and Beautiful Context's frame
    host (context_report_host.js). Negative pins run against this union so the
    moved code never leaves their coverage."""
    return (
        _widgets_js()
        + _render_js()
        + _read("web/modules/widget_module.js")
        + _read("web/modules/widget_frame.js")
        + _read("web/modules/widget_frame_bridge.js")
        + _read("web/modules/widget_card.js")
        + _read("web/modules/widget_reorder.js")
        + _read("web/modules/widget_chart.js")
        + _read("web/modules/context_report_host.js")
    )


def _read(rel: str) -> str:
    return (REPO_ROOT / rel).read_text(encoding="utf-8")


def test_widgets_use_design_radius_tokens():
    style = _read("web/ui.css")
    block_start = style.index(".ui-control {")
    block_end = style.index("}", block_start)
    block = style[block_start:block_end]
    assert "border-radius: var(--radius-inner);" in block
    assert "border-radius: 9px;" not in block


def test_context_report_page_is_admitted_by_its_own_csp_hashes():
    """Beautiful Context is one self-contained document: its policy admits its inline style and script by hash.

    An edit that keeps an old hash ships a page whose style or script the browser refuses, and only the opt-in
    browser lane opens that page (``tests/test_widgets_ui_browser.py``).
    """
    import base64
    import hashlib
    import re

    page = _read("web/context-report.html")
    policy = re.search(r'http-equiv="Content-Security-Policy" content="([^"]+)"', page).group(1)
    directives = dict(part.strip().split(" ", 1) for part in policy.split(";"))
    assert directives["default-src"] == "'none'"
    for tag, directive in (("style", "style-src"), ("script", "script-src")):
        (inline,) = re.findall(rf"<{tag}\b[^>]*>(.*?)</{tag}>", page, re.S)
        digest = base64.b64encode(hashlib.sha256(inline.encode("utf-8")).digest()).decode("ascii")
        assert directives[directive] == f"'sha256-{digest}'", directive


def test_context_report_page_sends_nothing_and_shares_the_readers_page_limit():
    """The page has no way to send what it reads (its palette arrives one way: the browser lane,
    ``tests/test_widgets_ui_browser.py``), and the page and the reader name one page limit, so a report the reader
    writes is one the page opens."""
    import re

    from ouroboros.context_attempt_reader import PAGE_LIMIT

    (script,) = re.findall(r"<script\b[^>]*>(.*?)</script>", _read("web/context-report.html"), re.S)
    assert not re.search(r"postMessage|fetch\(|WebSocket|XMLHttpRequest|sendBeacon|EventSource", script)
    (limit,) = re.findall(r"const maxBytes = (\d+) \* 1024 \* 1024;", script)
    assert int(limit) * 1024 * 1024 == PAGE_LIMIT


def test_widgets_schema_v1_composition_uses_stable_tree_keys():
    source = _widgets_js()
    renderer = _render_js()
    assert "function componentIdentity" in renderer
    assert "function indexComponentTree" in renderer
    assert "indexComponentTree(components)" in source
    assert "type === 'group'" in renderer
    assert "type === 'metric'" in renderer
    assert "type === 'callout'" in renderer
    assert "visibleKeys.forEach((key)" in source
    assert "components[Number(" not in _framed_widget_sources()
    assert "data-widget-kanban-key" in source
