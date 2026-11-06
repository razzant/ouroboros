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
    (widget_card.js), the card reorder handles (widget_reorder.js) and the
    declarative chart helpers (widget_chart.js). Negative pins run against this
    union so the moved code never leaves their coverage."""
    return (
        _widgets_js()
        + _render_js()
        + _read("web/modules/widget_module.js")
        + _read("web/modules/widget_frame.js")
        + _read("web/modules/widget_frame_bridge.js")
        + _read("web/modules/widget_card.js")
        + _read("web/modules/widget_reorder.js")
        + _read("web/modules/widget_chart.js")
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
