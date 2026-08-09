"""Static contract checks for the Widgets page renderer."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _widgets_js() -> str:
    return (REPO_ROOT / "web" / "modules" / "widgets.js").read_text(
        encoding="utf-8"
    )


def _read(rel: str) -> str:
    return (REPO_ROOT / rel).read_text(encoding="utf-8")


def test_widgets_support_declarative_schema_components():
    """Spot-check that widgets.js exposes the declarative schema entry point
    and a representative set of components. Trimmed in v5.15.x — the full
    type-marker enumeration (15+ entries) was brittle to schema evolution
    and added little signal over a smoke check. Security/lifecycle pins
    moved to the dedicated tests below (escape/sanitize, media source guard,
    download host helper, etc.)."""
    source = _widgets_js()
    assert "render.kind === 'declarative'" in source
    # Sentinel components — proof the declarative router is wired
    assert "type === 'form'" in source
    assert "type === 'action'" in source
    assert "type === 'table'" in source
    assert "type === 'markdown'" in source
    # Lifecycle / cleanup discipline
    assert "disposeMountedWidgets();" in source
    assert "let widgetsMounted = false;" in source
    assert "let renderGeneration = 0;" in source
    page_shown_branch = source.split("window.addEventListener('ouro:page-shown'")[1]
    assert "disposeMountedWidgets();" in page_shown_branch


def test_widgets_escape_and_sanitize_untrusted_content():
    """Widgets must reach the sanitised markdown helper through the v5.8.3-rc.5
    SSOT (``web/modules/utils.js::renderMarkdownSafe``); the DOMPurify
    allowlist itself moved to that module and is pinned by
    ``tests/test_web_utils_ssot.py::test_render_markdown_safe_strips_dangerous_tags_and_attrs``.
    Widgets-side this test now only verifies the import and the
    escapeHtml-around-untrusted-content discipline that remains local
    (table cells, JSON dumps).
    """
    source = _widgets_js()
    assert "renderMarkdownSafe" in source
    # Widgets must NOT redeclare the SSOT helper locally.
    assert "function renderMarkdownSafe" not in source, (
        "widgets.js must use renderMarkdownSafe from utils.js (SSOT), not a local copy"
    )
    assert "escapeHtml(JSON.stringify(value, null, 2))" in source
    assert "renderTableCell(row, c)" in source
    assert "function safeTableHref" in source


def test_widgets_media_sources_are_constrained_to_extension_routes_or_data_urls():
    source = _widgets_js()
    assert "function safeMediaSrc" in source
    assert "effectiveTarget = ''" in source
    assert "state[effectiveTarget || spec.target || 'result']" in source
    assert "safeMediaSrc(tab, component, state, target)" in source
    assert "const route = spec.route || spec.api_route || '';" in source
    assert "extensionRoutePath(tab.skill, route, params)" in source
    assert "data:(image\\/" in source
    assert "parsed.pathname.startsWith(expectedPrefix)" in source
    assert "parsed.origin === window.location.origin" in source
    assert "javascript:" not in source
    assert "`${treePath}.gallery.${idx}`, passiveTarget" in source


def test_widgets_downloads_use_host_handler_not_navigation():
    source = _widgets_js()
    helper = _read("web/modules/ui_helpers.js")
    assert "data-widget-download-url" in source
    assert "event.preventDefault();" in source
    assert "downloadViaHostBridge(" in source
    assert "download_file_to_downloads" in helper
    assert "URL.createObjectURL" in helper
    assert "window.location.href" not in source
    assert "window.location.assign" not in source
    assert '<a class="btn btn-default" href' not in source


def test_widgets_treat_head_as_no_body_request():
    source = _widgets_js()
    assert "const noBody = method === 'GET' || method === 'HEAD';" in source
    assert "const init = noBody" in source


def test_widgets_keep_iframe_sandbox_locked_down():
    """The legacy ``kind: "iframe"`` widget surface mounts an extension
    route inside a <iframe> with the *empty* sandbox attribute (no
    permissions at all). v5.7.0 added ``kind: "module"``, which mounts
    extension-supplied JS inside a separate <iframe srcdoc> with
    ``sandbox="allow-scripts"`` BUT no ``allow-same-origin`` token —
    so the iframe is still an opaque origin (no SPA cookie / storage
    access) and is further constrained by a strict CSP. We check both
    invariants here:

    1. The legacy iframe path still uses the empty sandbox.
    2. The module iframe path adds ``allow-scripts`` but never adds
       ``allow-same-origin`` (the only token that would re-expose
       parent storage).
    """
    source = _widgets_js()
    assert 'sandbox=""' in source
    # ``allow-scripts`` is now legitimately present, but only inside the
    # ``kind === 'module'`` branch. The dangerous combined sandbox token
    # must never appear in an actual iframe attribute.
    assert 'sandbox="allow-scripts"' in source
    assert 'sandbox="allow-scripts allow-same-origin"' not in source
    assert 'sandbox="allow-scripts allow-forms allow-same-origin"' not in source
    assert "render.kind === 'module'" in source
    # Verify the module iframe carries a CSP that does NOT grant network
    # access directly. The parent injects a postMessage fetch bridge instead,
    # restricted to /api/extensions/<skill>/... from the parent side.
    assert "default-src 'none'" in source
    assert "script-src 'unsafe-inline'" in source
    assert "OuroborosWidget = { fetch: window.fetch }" in source
    assert "module widget fetch outside extension route prefix" in source


def test_widgets_use_design_radius_tokens():
    style = (REPO_ROOT / "web" / "style.css").read_text(encoding="utf-8")
    block_start = style.index(".widget-field input,")
    block_end = style.index("}", block_start)
    block = style[block_start:block_end]
    assert "border-radius: var(--radius-sm);" in block
    assert "border-radius: 9px;" not in block


def test_widgets_refresh_button_shows_loading_state():
    source = _widgets_js()
    css = (REPO_ROOT / "web" / "style.css").read_text(encoding="utf-8")

    assert "refreshBtn.classList.add('is-loading')" in source
    assert "refreshBtn.classList.remove('is-loading')" in source
    assert "refreshBtn.disabled = true" in source
    assert "#widgets-refresh.is-loading::after" in css


def test_widgets_cards_do_not_stretch_to_row_height():
    source = _widgets_js()
    css = (REPO_ROOT / "web" / "style.css").read_text(encoding="utf-8")
    masonry = (REPO_ROOT / "web" / "modules" / "masonry.js").read_text(encoding="utf-8")
    assert "const span = Number(tab.span || tab.grid_span || 1);" in source
    assert "widgets-card-span-2" in source
    assert "applyMasonry(list)" in source
    assert "function layout(container, config)" in masonry
    assert "item.classList.contains(spanClass) ? 2 : 1" in masonry
    assert "Math.min(desiredColumns, availableColumns)" in masonry
    assert "itemResizeObserver" in masonry
    assert "observeItems()" in masonry
    widgets_block = css.split(".widgets-list {", 1)[1].split("}", 1)[0]
    assert "display: grid" not in widgets_block
    assert "position: relative;" in widgets_block
    assert ".widgets-card-span-2" in css


def test_widget_form_label_is_accessible_heading_fallback():
    source = _widgets_js()
    assert "const heading = component.title || component.label || '';" in source
    assert 'aria-label="${escapeHtml(heading)}"' in source
    assert "heading ? `<h4>${escapeHtml(heading)}</h4>` : ''" in source


def test_widget_json_wraps_inside_its_host_card():
    style = _read("web/style.css")
    json_block = style.split(".widget-json pre {", 1)[1].split("}", 1)[0]
    assert "max-width: 100%;" in json_block
    assert "max-height: min(360px, 50vh);" in json_block
    assert "overflow: auto;" in json_block
    assert "white-space: pre-wrap;" in json_block
    assert "overflow-wrap: anywhere;" in json_block


def test_widgets_card_order_is_owner_ui_preference():
    source = _widgets_js()
    css = (REPO_ROOT / "web" / "style.css").read_text(encoding="utf-8")
    api_client = (REPO_ROOT / "web" / "modules" / "api_client.js").read_text(encoding="utf-8")

    assert 'data-widget-reorder-handle' in source
    assert "function sortTabsByWidgetOrder" in source
    assert "originalIndex" in source
    assert "return a.originalIndex - b.originalIndex;" in source
    assert "Move widget: drag or use arrow keys" in source
    assert "handle.addEventListener('keydown'" in source
    assert "event.key === 'ArrowUp'" in source
    assert "apiClient.uiPreferences()" in source
    assert "apiClient.saveUiPreferences({ widget_order: normalized })" in source
    assert "currentWidgetOrderFromDom(list)" in source
    assert ".widgets-card-drag" in css
    assert ".widgets-card.drag-over" in css
    assert "uiPreferences: () => fetchJson('/api/ui/preferences'" in api_client
    assert "saveUiPreferences: (payload) => jsonPost('/api/ui/preferences', payload)" in api_client


def test_widgets_inline_card_host_path_removed():
    source = _widgets_js()
    assert "render.kind === 'inline_card'" not in source
    assert "skill-widget-weather" not in source
    assert "const saved = widgetSessionState.get(persistenceKey) || {};" in source


def test_widgets_v5_7_0_new_components_render():
    """v5.7.0 host-owned declarative components: ``map`` (Leaflet-ready
    fallback list), ``calendar`` (host SVG-style row list), ``kanban``
    (HTML5 drag with on_move POST). All three must be present in the
    declarative renderer so authors can reference them in widgets, and
    none of them may bring skill-supplied JS into the SPA origin."""
    source = _widgets_js()
    assert "type === 'map'" in source
    assert "type === 'calendar'" in source
    assert "type === 'kanban'" in source
    # Module / arbitrary <script> from the skill must NEVER be inserted
    # into the host origin. ``data-widget-map-config`` carries the spec
    # as JSON in a data attribute (host renders); no runtime eval of
    # extension JS is acceptable in any of the new component renderers.
    assert "data-widget-map-config" in source
    assert "widget-kanban-card" in source


def test_widgets_render_subscription_children():
    source = _widgets_js()
    assert "type === 'subscription'" in source
    assert "component.render" in source
    assert "widget-subscription-render" in source
    assert "inheritedTarget = ''" in source
    assert "component.target || inheritedTarget || 'result'" in source
    assert "renderComponent(tab, child, view, `${treePath}.render.${idx}`, target)" in source
    assert "const passiveTarget = inheritedTarget ? target : '';" in source
    assert "value_key" in source
    assert "items_key" in source
    assert "route_prefix" in source
    assert "type === 'key_value'" in source


def test_widgets_schema_v1_composition_uses_stable_tree_keys():
    source = _widgets_js()
    assert "function componentIdentity" in source
    assert "function indexComponentTree" in source
    assert "type === 'group'" in source
    assert "type === 'metric'" in source
    assert "type === 'callout'" in source
    assert "visibleKeys.forEach((key)" in source
    assert "components[Number(" not in source
    assert "data-widget-kanban-key" in source


def test_widgets_forms_charts_and_kanban_keep_host_owned_contracts():
    source = _widgets_js()
    helper = _read("web/modules/ui_helpers.js")
    assert "renderSafeField(" in source
    assert "collectSafeFieldValues(" in source
    assert "includePasswords: false" in source
    assert "pendingActions.has(key)" in source
    assert "spanGaps: false" in source
    assert "finiteChartValue" in source
    assert "aria-label=" in source
    assert "renderChartDataTable" in source
    assert "data-widget-kanban-move" in source
    assert "widget-kanban-empty" in source
    assert "mount.querySelectorAll('[data-widget-kanban-key]')" in source
    assert "{ card_id: cardId, column_id: columnId }" in source
    assert "SAFE_FIELD_TYPES" in helper
    assert "autocomplete=\"new-password\"" in helper


def test_canvas_palettes_read_only_defined_root_tokens():
    """Canvas token seam (flat redesign, Phase D).

    Chart.js paints on a canvas and cannot resolve `var(...)`, so both chart
    palettes resolve their colours through `readThemeTokens` at build time.
    `getComputedStyle().getPropertyValue('--x')` returns the COMPUTED value, so an
    alias like `--accent-system: var(--amber)` resolves THROUGH to the hex and is
    perfectly paintable (verified in Chromium and WebKit). The real failure mode
    is a token that is not DEFINED in :root: that comes back as the empty string,
    and a canvas handed '' silently drops the series. So this pins EXISTENCE —
    every token name these modules hand to the seam is defined in :root — and
    deliberately says nothing about leaves vs aliases.
    """
    style = _read("web/style.css")
    root_block = style.split(":root {", 1)[1].split("\n}", 1)[0]
    definitions = dict(
        re.findall(r"^\s*(--[\w-]+)\s*:\s*([^;]+);", root_block, flags=re.MULTILINE)
    )
    assert definitions, "could not parse the :root token block"

    named: set[str] = set()
    for rel in ("web/modules/evolution.js", "web/modules/widgets.js"):
        source = _read(rel)
        # Both modules keep their token names in flat string lists / maps, so the
        # names are literal source text by construction (that is the point of the
        # seam: no computed token names, nothing to resolve at runtime).
        for block in re.findall(
            r"const (?:CHART_TOKENS|CHART_SERIES_TOKENS|CHART_GRID_TOKEN)\b[^;]*;",
            source,
            flags=re.DOTALL,
        ):
            named.update(re.findall(r"'(--[\w-]+)'", block))
    # Sanity: the sweep actually found the ramps, not an empty set that trivially
    # passes (six evolution series + four widget series + axes/grid/surface/mono).
    assert len(named) >= 12, f"token sweep found too few names: {sorted(named)}"
    assert "--accent-light" in named and "--font-mono" in named

    for token in sorted(named):
        assert token in definitions, (
            f"{token} is read by a chart but not defined in :root: readThemeTokens "
            f"would hand Chart.js '' and the series would vanish without an error."
        )


def test_widgets_responsive_design_system_styles_are_host_owned():
    style = _read("web/style.css")
    assert ".widget-group-grid" in style
    assert ".widget-metric" in style
    assert ".widget-callout" in style
    assert ".widget-form-fields.widget-grid-cols-4" in style
    assert "content: attr(data-label);" in style
    assert ".widget-kanban-move" in style
    assert ".widget-kanban-col.is-empty" in style
    narrow = style.rsplit("@media (max-width: 640px) {", 1)[1]
    assert ".widget-kanban-col.is-empty" in narrow
    assert "min-height: 0;" in narrow
    assert "padding-block: 8px;" in narrow
    assert ".widget-group-components > * { margin-top: 0; }" in style
    assert "repeat(auto-fit, minmax(min(220px, 100%), 1fr))" in style
    assert ".widget-group-grid > .widget-group-components > :is(" in style


def test_widget_public_tones_share_the_host_normalizer_and_canonical_css():
    source = _widgets_js()
    helper = _read("web/modules/ui_helpers.js")
    style = _read("web/style.css")
    assert "function widgetTone" not in source
    assert "normalizeTone(component.tone)" in source
    assert "normalizeTone(component.tone, 'info')" in source
    assert "success: 'ok'" in helper
    assert "warning: 'warn'" in helper
    assert "neutral: 'muted'" in helper
    assert '.widget-metric[data-tone="ok"], .widget-callout[data-tone="ok"]' in style
    assert '.widget-metric[data-tone="warn"], .widget-callout[data-tone="warn"]' in style


def test_widget_metrics_share_the_standard_empty_value_and_numeric_formatter():
    source = _widgets_js()
    assert "const numericValue = text ? Number(text) : Number.NaN;" in source
    assert "!Number.isNaN(numericValue) && !Number.isFinite(numericValue)" in source
    assert "const structured = raw !== null && typeof raw === 'object';" in source
    assert "nonFiniteText" in source
    assert "typeof raw === 'number' || numericText ? formatNumber" in source
