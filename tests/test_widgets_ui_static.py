"""Static contract checks for the Widgets page renderer."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _widgets_js() -> str:
    return (REPO_ROOT / "web" / "modules" / "widgets.js").read_text(
        encoding="utf-8"
    )


def _framed_widget_sources() -> str:
    """widgets.js (page host, dispatcher, declarative renderer) plus the framed
    mounts split out of it (widget_module.js), the in-frame bootstrap
    (widget_frame.js), the framed-card chrome (widget_card.js), the card
    reorder handles (widget_reorder.js) and the declarative chart helpers
    (widget_chart.js). Negative pins run against this union so the moved code
    never leaves their coverage."""
    return (
        _widgets_js()
        + _read("web/modules/widget_module.js")
        + _read("web/modules/widget_frame.js")
        + _read("web/modules/widget_card.js")
        + _read("web/modules/widget_reorder.js")
        + _read("web/modules/widget_chart.js")
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
    # Lifecycle / cleanup discipline: a list rebuild disposes every mounted
    # card first; leaving stops everything except the frames the owner keeps
    # running (phase 3).
    assert "disposeMountedWidgets();" in source
    assert "let widgetsMounted = false;" in source
    assert "let renderGeneration = 0;" in source
    page_shown_branch = source.split("window.addEventListener('ouro:page-shown'")[1]
    assert "disposeMountedWidgets(retainsWhileHidden);" in page_shown_branch
    assert "disposeMountedWidgets();" not in page_shown_branch


def test_widgets_page_reads_cheap_list_and_reconciles_by_signature():
    """Widgets lifecycle phase 1: the page reads the passive ``GET /api/widgets``
    projection — never the fat ``/api/extensions`` catalogue, which stays the
    Skills page's read — paints the shell from the last known list before its
    first await, and after every fetch compares an order-independent list
    signature: unchanged → not one ``<article>`` is touched; changed → keyed
    patch (``web/modules/widget_list.js`` holds the pure helpers). The same
    sync runs on a visible ``extension_lifecycle`` event and on every WebSocket
    (re)connect, never on a timer. The page has no Refresh control (owner
    decision Q20): the window reload is the only hard reset, so nothing in the
    page stops every kept-running card behind the owner's back."""
    source = _widgets_js()
    helpers = _read("web/modules/widget_list.js")
    assert "apiClient.widgets()" in source
    assert "apiClient.extensions()" not in source
    assert "live.ui_tabs" not in source
    assert "live?.ui_tabs" not in source
    assert "from './widget_list.js'" in source
    assert "export function widgetTabsSignature" in helpers
    assert "export function planWidgetListPatch" in helpers
    assert "const signature = widgetTabsSignature(tabs);" in source
    assert "if (signature !== lastSignature) patchWidgetCards(list, lastTabs, tabs);" in source
    assert "ctx.ws.on('extension_lifecycle', reconcileWidgetList);" in source
    assert "ctx.ws.on('open', reconcileWidgetList);" in source
    assert "setInterval(" not in source
    # Entry paints the shell before the first await, then syncs.
    assert source.index("paintShell(lastTabs);") < source.index("await syncWidgets(generation);")
    # Owner decision Q20: the page has no Refresh control at all. A window reload
    # is the only hard reset, so nothing in the page can stop every kept-running
    # program behind the owner's back, and no confirmation dialog is needed.
    css = (REPO_ROOT / "web" / "style.css").read_text(encoding="utf-8")
    card = _read("web/modules/widget_card.js")
    for absent in ("widgets-refresh", "refreshBtn", "refreshWidgets", "confirmWidgetsRestart"):
        assert absent not in source, absent
    assert "confirmWidgetsRestart" not in card
    assert "openConfirmDialog" not in card
    assert "widgets-refresh" not in css
    assert "actionsHtml" not in source
    # `render()` takes no force flag: there is no path that clears every owner
    # Stop and rebuilds every card while the page stays open.
    assert "async function render() {" in source
    assert "stoppedByOwner.clear()" not in source
    # A vanished card's declarative session state and the owner's page-session Stop
    # are evicted on BOTH removal paths: the keyed patch, and the rebuild branch the
    # patch never sees (the last card leaving, or the first list arriving). Eviction
    # runs after disposal, because a declarative disposer writes that snapshot as it
    # goes. Without it, re-enabling the only skill restores values the owner never
    # re-entered and keeps its card suppressed.
    rebuild = source.split("// Rebuilding the shell destroys frames", 1)[1].split("renderShell(list, tabs);", 1)[0]
    assert "await disposeMountedWidgets();" in rebuild
    assert rebuild.index("await disposeMountedWidgets();") < rebuild.index("widgetSessionState.delete(key);")
    assert "stoppedByOwner.delete(key);" in rebuild
    patch = source.split("function patchWidgetCards(", 1)[1].split("for (const tab of nextTabs)", 1)[0]
    assert "widgetSessionState.delete(key);" in patch and "stoppedByOwner.delete(key);" in patch


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
    assert "function renderMarkdownSafe" not in _framed_widget_sources(), (
        "widgets.js must use renderMarkdownSafe from utils.js (SSOT), not a local copy"
    )
    assert "escapeHtml(JSON.stringify(value, null, 2))" in source
    assert "renderTableCell(row, c)" in source
    # The table cell renderer and its http(s)-only link guard moved to
    # widget_chart.js (cycle-A fix round, ratchet room); the guard still exists.
    assert "function safeTableHref" in _read("web/modules/widget_chart.js")


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
    assert "javascript:" not in _framed_widget_sources()
    assert "`${treePath}.gallery.${idx}`, passiveTarget" in source


def test_widgets_downloads_use_host_handler_not_navigation():
    source = _widgets_js()
    helper = _read("web/modules/ui_helpers.js")
    assert "data-widget-download-url" in source
    assert "event.preventDefault();" in source
    assert "downloadViaHostBridge(" in source
    assert "download_file_to_downloads" in helper
    assert "URL.createObjectURL" in helper
    framed = _framed_widget_sources()
    assert "window.location.href" not in framed
    assert "window.location.assign" not in framed
    assert '<a class="btn btn-default" href' not in framed


def test_widgets_treat_head_as_no_body_request():
    source = _widgets_js()
    assert "const noBody = method === 'GET' || method === 'HEAD';" in source
    assert "const init = noBody" in source


def test_widgets_keep_iframe_sandbox_locked_down():
    """Both framed mounts — the ``kind: "iframe"`` route frame and the
    ``kind: "module"`` ``srcdoc`` frame — carry ONE decided capability set
    (widgets lifecycle sprint, Q14=B / Q16=A): ``sandbox="allow-scripts
    allow-pointer-lock allow-downloads"``, ``allow="autoplay; fullscreen;
    clipboard-write"`` and ``allowfullscreen``. The route frame's former empty
    sandbox is gone deliberately (its page is the skill's own, reviewed as
    payload; the opaque origin still keeps the SPA's cookies and DOM away). The
    tokens that would re-expose the SPA origin or widen the frame beyond the
    decision never appear in the framed sources: no ``allow-same-origin`` (the
    only token that re-exposes parent storage), no top navigation, popups,
    forms (``form-action`` does not fall back to ``default-src``) or modals,
    and no clipboard read. The module frame is a created ``<iframe>`` whose
    document is assigned through the ``srcdoc`` PROPERTY (no attribute-escaping
    round-trip of a module-sized payload); the route frame's URL goes in
    through the ``src`` property the same way (never an interpolated
    attribute string).
    """
    source = _framed_widget_sources()
    module = _read("web/modules/widget_module.js")
    assert "export const WIDGET_FRAME_SANDBOX = 'allow-scripts allow-pointer-lock allow-downloads';" in module
    assert "export const WIDGET_FRAME_ALLOW = 'autoplay; fullscreen; clipboard-write';" in module
    assert "iframe.setAttribute('sandbox', WIDGET_FRAME_SANDBOX);" in module
    assert "iframe.setAttribute('allow', WIDGET_FRAME_ALLOW);" in module
    assert "iframe.setAttribute('allowfullscreen', '');" in module
    assert 'sandbox=""' not in source, "the route iframe shares the module frame's capability set now"
    assert "iframe.setAttribute('sandbox', 'allow-scripts');" not in source
    assert "iframe.srcdoc = srcdoc;" in source
    assert 'srcdoc="' not in source
    assert "iframe.src = src;" in module
    assert 'src="${src}"' not in source
    for forbidden in (
        "allow-same-origin",
        "allow-top-navigation",
        "allow-popups",
        "allow-forms",
        "allow-modals",
        "clipboard-read",
    ):
        assert forbidden not in source, forbidden
    assert "render.kind === 'module'" in source
    # The module document CSP: sources are ABSOLUTE (an opaque frame's 'self' is
    # nothing), scripts run inline / from blob: / from this skill's module
    # prefix with 'wasm-unsafe-eval', workers from blob:, passive image / media
    # / font loads from data: / blob: / this skill's route prefix — and nothing
    # else: no connect-src (falls to default-src 'none'; the parent bridge is the
    # frame's one request path) and never plain 'unsafe-eval'.
    assert "export function moduleFrameCsp(skill, origin = window.location.origin)" in module
    assert "const prefix = `${origin}${extensionRoutePrefix(skill)}`;" in module
    assert "default-src 'none'" in module
    assert "script-src 'unsafe-inline' 'wasm-unsafe-eval' blob: ${prefix}module/" in module
    assert "'worker-src blob:'" in module
    assert "style-src 'unsafe-inline'" in module
    assert "img-src data: blob: ${prefix}" in module
    assert "media-src data: blob: ${prefix}" in module
    assert "font-src data: blob: ${prefix}" in module
    assert "const csp = moduleFrameCsp(tab.skill);" in module
    assert "connect-src" not in source
    assert "'unsafe-eval'" not in source
    assert "window.OuroborosWidget = { fetch: request, onEvent };" in source
    assert "module widget fetch outside extension route prefix" in source


def test_widgets_frame_geometry_and_teardown_contract():
    source = _framed_widget_sources()
    style = _read("web/style.css")
    assert "--widget-frame-height" in source
    assert "height: var(--widget-frame-height, 320px);" in style
    assert "type: 'ouro-widget-resize'" in source
    assert "new ResizeObserver(report)" in source
    assert "box.bottom - bodyTop + bodyBottomSpacing" in source
    assert "fixedViewportBody" in source
    assert 'scrolling="no"' not in source
    assert "syncModuleFrameScrolling" not in source
    assert "getPropertyValue('overflow-y')" in source
    assert "getPropertyPriority('overflow-y')" in source
    assert "style.setProperty('overflow-y', 'hidden', 'important')" in source
    assert "style.setProperty('overflow-x'" not in source
    assert source.index("setVerticalOverflowSuppressed(true);") < source.index("const report = () =>")
    assert "setVerticalOverflowSuppressed(outerHeight <" in source
    assert "setVerticalOverflowSuppressed(false);" in source
    assert "nonce, WIDGET_FRAME_DEFAULT_HEIGHT, maxHeight, WIDGET_FRAME_BORDER_RESERVE," in source
    assert source.index("<script>${resizeBridge}</script>") < source.index("<script>${escapeScript(moduleSource)}</script>")
    assert "ouro-widget-dispose" in source
    assert "if (iframe?.parentNode === mount) iframe.remove();" in source
    assert "pendingRequests.forEach((controller) => controller.abort());" in source
    assert "if (!isCurrent())" in source
    assert "WIDGET_FRAME_MAX_HEIGHT = 8192" in source
    assert "widget module request timed out" in source
    assert source.index("moduleSource = await resp.text();") < source.index("clearTimeout(sourceTimeout);")
    assert "widgetMountControllers.forEach((controller) => controller.abort());" in source


def test_widgets_framed_dispose_is_ordered_and_acknowledged():
    """Widgets lifecycle phase 2, both sides of the frame. Child
    (``widget_frame.js``): on ``ouro-widget-dispose`` every registered hook
    runs first — async hooks are awaited, the fetch bridge stays live for
    them — then the bootstrap posts ``ouro-widget-disposed`` and only then
    fails pending fetches (rejecting unsettled ones, erroring open body
    streams) and removes its listener. Parent
    (``widget_module.js``): the disposer posts the dispose message, keeps
    ``onMessage`` answering bridged fetches, and the abort → unlisten →
    ``iframe.remove()`` tail runs from ``finish`` on the acknowledgement or
    after ``WIDGET_DISPOSE_ACK_TIMEOUT_MS`` (1 s, beside the 25 s request
    timeout) — asynchronously, never blocking a page switch. The page keeps
    one settle promise per key so a remount waits for the pending stop."""
    child = _read("web/modules/widget_frame.js")
    parent = _read("web/modules/widget_module.js")
    jobs = _read("web/modules/widget_job.js")
    page = _widgets_js()
    assert "export const WIDGET_DISPOSE_ACK_TIMEOUT_MS = 1000;" in jobs
    assert jobs.index("WIDGET_REQUEST_TIMEOUT_MS = 25000") < jobs.index("WIDGET_DISPOSE_ACK_TIMEOUT_MS = 1000")
    # Child: hooks (awaited) → ack → reject pending → unlisten.
    dispose_body = child.split("const dispose = async () =>", 1)[1].split("const onMessage", 1)[0]
    assert "await Promise.allSettled(hooks.map((fn) => Promise.resolve().then(fn)));" in dispose_body
    assert dispose_body.index("Promise.allSettled") < dispose_body.index("type: 'ouro-widget-disposed'")
    assert dispose_body.index("type: 'ouro-widget-disposed'") < dispose_body.index("item.fail(new Error('widget disposed'))")
    assert dispose_body.index("item.fail(new Error('widget disposed'))") < dispose_body.index("window.removeEventListener('message', onMessage);")
    # The bridge answers during the hooks: frames are refused only once `disposed`,
    # and the dispose message itself is honoured before that gate.
    frames = child.split("const onMessage = (event) =>", 1)[1].split("const request =", 1)[0]
    assert frames.index("if (msg.type === 'ouro-widget-dispose') {") < frames.index("if (disposed) return;")
    assert frames.index("if (disposed) return;") < frames.index("if (msg.type !== 'ouro-widget-fetch-chunk') return;")
    # Parent: the old synchronous tail is now the post-ack `finish`.
    assert "if (msg.type === 'ouro-widget-disposed') {" in parent
    tail = parent.split("const finish = () =>", 1)[1]
    assert tail.index("pendingRequests.forEach((controller) => controller.abort());") < tail.index("window.removeEventListener('message', onMessage);")
    assert tail.index("window.removeEventListener('message', onMessage);") < tail.index("if (iframe?.parentNode === mount) iframe.remove();")
    assert "onDisposed = finish;" in tail
    assert "setTimeout(finish, WIDGET_DISPOSE_ACK_TIMEOUT_MS)" in tail
    assert "postMessage({ type: 'ouro-widget-dispose', nonce }, '*');" in tail
    assert "if (disposing) return disposing;" in parent
    # Page: one settle promise per key; a remount and the facade wait for it.
    assert "const widgetDisposing = new Map();" in page
    assert "if (settling) await settling;" in page
    assert "await widgetDisposing.get(key);" in page
    assert "return Promise.allSettled(Array.from(widgetDisposing.values()));" in page
    # Cycle A, CA-1: a CHANGED card whose frame runs (revision / render change)
    # follows the removed-card shape — the old <article> stands, marked and
    # `stopping`, until the settle; the fresh card is inserted beside it and its
    # mount waits on the same settle promise. Never `replaceWith` over a
    # settling frame.
    retire = page.split("function retireCard(card, settling) {", 1)[1].split("function patchWidgetCards", 1)[0]
    assert "card.setAttribute('data-widget-removed', '');" in retire
    assert "syncWidgetCardControls(card, 'stopping');" in retire
    assert "settling.then(() => card.remove());" in retire
    changed_branch = page.split("for (const tab of nextTabs) {", 1)[1].split("function mountTrackedTab", 1)[0]
    assert "const settling = disposeWidgetByKey(key);" in changed_branch
    assert "else if (!settling) card.replaceWith(fresh);" in changed_branch
    assert "card.after(fresh);" in changed_branch
    assert "retireCard(card, settling);" in changed_branch
    # CA-3: one mount in flight per key — a second request for the SAME card
    # joins it. CA-19 / A3-10: never a stale one (another card node, or a mount
    # whose page generation moved on) — wait for it, then mount on your own.
    assert "const widgetMounting = new Map();" in page
    assert "if (inFlight && inFlight.card === card && inFlight.isCurrent()) return inFlight.promise;" in page
    assert "? inFlight.promise.catch(() => {}).then(() => mountTabOnce(card, tab, key, isCurrent))" in page
    assert "widgetMounting.set(key, { card, isCurrent, promise: mounting });" in page
    assert "if (!isCurrent() || !card.isConnected) return;" in page


def test_widgets_launch_policy_controls_and_stop_suppression():
    """Widgets lifecycle phase 2, host side. Framed (module / route-iframe)
    cards carry exactly one primary control (Start / Stop) plus a secondary
    launch-policy menu built on the Skills card menu primitive; declarative
    cards get neither. The effective policy is owner override > author
    ``render.start`` > kind default (``widget_card.js``, node-tested). A card
    that is not to run shows a facade at the declared frame height through
    the frame's own custom property. Owner Stop is remembered for the page
    session only; Start, a policy change to Auto / Keep running, and a window
    reload forget it. A vanished card is stopped in order and evicts its session
    state. Phase 3: a FRAMED card whose effective policy is `retain` keeps its
    frame mounted in the hidden page when the owner leaves (declarative cards
    always dispose), its badge says "Keeps running", a lifecycle event while
    hidden force-stops a kept frame whose skill left the list, and the window
    reload still ends every frame with its window."""
    page = _widgets_js()
    card = _read("web/modules/widget_card.js")
    style = _read("web/style.css")
    assert "export function effectiveStartMode(tab, prefs)" in card
    assert "export function isRetainedWidget(tab, prefs)" in card
    assert "return isFramedWidget(tab) && effectiveStartMode(tab, prefs) === 'retain';" in card
    assert "? 'Keeps running'" in card
    assert "const retainsWhileHidden = (key) => isRetainedWidget(tabByKey(key), uiPreferences);" in page
    assert "if (!keep?.(key)) disposeWidgetByKey(key);" in page
    hidden_branch = page.split("async function stopVanishedRetainedWidgets() {", 1)[1].split("async function syncWidgets", 1)[0]
    assert "if (widgetsVisible) return;" in hidden_branch
    assert "kept.filter((key) => !live.has(key)).forEach(disposeWidgetByKey);" in hidden_branch
    reconcile = page.split("function reconcileWidgetList() {", 1)[1].split("async function stopVanishedRetainedWidgets", 1)[0]
    assert "stopVanishedRetainedWidgets();" in reconcile
    assert reconcile.index("listDirty = true;") < reconcile.index("stopVanishedRetainedWidgets();")
    assert "const KIND_DEFAULT_START = { declarative: 'auto', module: 'manual', iframe: 'manual' };" in card
    assert "if (!isFramedWidget(tab)) return '';" in card
    assert card.count("btn btn-primary") == 1
    assert 'role="menuitemradio"' in card
    assert '<dialog class="skills-card-menu-dialog" role="menu"' in card
    assert 'class="skills-card-menu-trigger"' in card
    assert '<span class="ui-status" data-tone="neutral" data-widget-status hidden>' in card
    # Owner-facing menu wording, not the enum name (CA-11).
    assert "retain: 'Keep running'," in card
    assert "(retain)" not in card
    # Facade: never over a settling frame (CA-13); `icon` is a glyph — an
    # identifier-like NAME (the `extension` default) falls back to the page glyph (CA-15).
    assert "mount.querySelector('[data-widget-facade], iframe')" in card
    assert "const ICON_NAME = /^[a-z][a-z0-9_-]*$/i;" in card
    assert "setFrameHeight(mount.firstElementChild, frameHeight(tab.render || {}));" in card
    # Closing a menu hands focus back to its trigger when it was inside (CA-16, WebKit).
    assert "trigger?.focus({ preventScroll: true });" in card
    assert ".widgets-facade {" in style
    assert "height: var(--widget-frame-height, 320px);" in style.split(".widgets-facade {", 1)[1].split("}", 1)[0]
    assert ".widgets-card-controls .ui-status[data-tone]::before" in style
    # An open policy menu is never painted under a sibling card (CA-14).
    assert ".widgets-card:has(.skills-card-menu-dialog[open]) {" in style
    # Widgets is not a migrated surface (DESIGN.md section 8): the phase-2/3 card
    # controls / menu / facade rules keep the surface's literals, not type tokens;
    # the shared `.ui-status[data-tone]` pair (section 4) is the one exception.
    controls_css = style.split("/* Framed-card head controls", 1)[1].split(".widgets-card-source {", 1)[0]
    for token in ("var(--type-", "var(--line-", "var(--text-meta)"):
        assert token not in controls_css, token
    # Page: policy gate, suppression, owner controls, whole-map persistence.
    assert "const stoppedByOwner = new Set();" in page
    assert "effectiveStartMode(tab, uiPreferences) !== 'manual'" in page
    assert "&& !stoppedByOwner.has(widgetKey(tab));" in page
    assert "if (isFramedWidget(tab) && !startsOnShow(tab)) await settleStopped(card, tab);" in page
    assert "stoppedByOwner.add(widgetKey(tab));" in page
    assert "stoppedByOwner.delete(widgetKey(tab));" in page
    assert "apiClient.saveUiPreferences({ widget_start_mode: next })" in page
    assert "const next = withWidgetStartMode(current, key, mode);" in page
    # Launch-policy writes are serialized (AUD-13): the card applies after its write.
    assert "let startModeWrites = Promise.resolve();" in page
    assert "startModeWrites = write.catch(() => {});" in page
    set_mode = page.split("async function setWidgetStartMode(key, mode) {", 1)[1].split("bindWidgetCardMenus", 1)[0]
    assert set_mode.index("await write;") < set_mode.index("const card = liveCardFor(list, key);")
    assert "bindWidgetCardMenus(list, setWidgetStartMode);" in page
    assert "event.target.closest('[data-widget-power]')" in page
    # Force-stop + eviction on a vanished card; the frame keeps its ack window.
    removed_branch = page.split("for (const key of plan.removed) {", 1)[1].split("for (const tab of nextTabs) {", 1)[0]
    assert "disposeWidgetByKey(key)" in removed_branch
    assert "widgetSessionState.delete(key);" in removed_branch
    assert "stoppedByOwner.delete(key);" in removed_branch
    assert "if (settling) retireCard(card, settling);" in removed_branch
    # A stopped framed card ends with its facade and Start (CA-13): on leave for
    # every card not kept running, and while a card starts (idempotent facade).
    page_shown_branch = page.split("window.addEventListener('ouro:page-shown'", 1)[1]
    assert "if (card && tab && isFramedWidget(tab)) settleStopped(card, tab);" in page_shown_branch
    start_widget = page.split("async function startWidget(card, tab, isCurrent) {", 1)[1].split("async function settleStopped", 1)[0]
    assert "if (isFramedWidget(tab)) renderWidgetFacade(mount, tab);" in start_widget
    assert "else if (isFramedWidget(tab)) await settleStopped(card, tab);" in start_widget
    # CA-20: a mount under way for the key decides the body first — wait for it
    # and re-check; never an empty body behind a mount that bailed.
    settle = page.split("async function settleStopped(card, tab) {", 1)[1].split("async function stopWidgetByOwner", 1)[0]
    assert "if (widgetMounting.has(key)) await widgetMounting.get(key).promise.catch(() => {});" in settle
    assert "if (widgetDisposers.has(key) || !card.isConnected) return;" in settle
    # Hidden reconcile force-stops vanished kept frames even mid-sync (CA-18).
    assert "if (!widgetsVisible) stopVanishedRetainedWidgets();" in page
    assert "localStorage" not in _framed_widget_sources()


def test_widgets_job_poll_retries_transient_transport_without_dropping_id():
    source = _widgets_js()
    assert "error.retryable = resp.status === 408" in source
    assert "isRetryableWidgetError(err) && ticks < maxTicks" in source
    assert "classifyWidgetJobStatus" in source
    assert "invalid job status response" in source
    assert "status[target] = 'loading';" in source
    assert "schedule(pollJob, intervalMs);" in source
    assert "delete componentState[`job:${key}`];" in source


def test_widgets_use_design_radius_tokens():
    style = (REPO_ROOT / "web" / "style.css").read_text(encoding="utf-8")
    block_start = style.index(".widget-field input,")
    block_end = style.index("}", block_start)
    block = style[block_start:block_end]
    assert "border-radius: var(--radius-sm);" in block
    assert "border-radius: 9px;" not in block


def test_widgets_cards_do_not_stretch_to_row_height():
    """Masonry packs unequal cards by absolute position. Phase 3: `layout()`
    packs the cards in the page's explicit key order (`widget_order`), never the
    DOM order, and writes the plan back only as narrow custom properties
    (`--masonry-w/-x/-y` per card, `--masonry-h` on the list) that one static
    rule set in web/style.css applies; the generated per-container `<style>`
    with `:nth-child` rules is gone, and `applyMasonry` returns an idempotent
    disposer for its three observers and the pending frame."""
    source = _widgets_js()
    css = (REPO_ROOT / "web" / "style.css").read_text(encoding="utf-8")
    masonry = (REPO_ROOT / "web" / "modules" / "masonry.js").read_text(encoding="utf-8")
    assert "const span = Number(tab.span || tab.grid_span || 1);" in source
    assert "widgets-card-span-2" in source
    assert "const relayout = () => applyMasonry(list, { order: currentWidgetOrder() });" in source
    assert "function layout(container, config)" in masonry
    assert "item.classList.contains(spanClass) ? 2 : 1" in masonry
    assert "Math.min(desiredColumns, availableColumns)" in masonry
    assert "itemResizeObserver" in masonry
    assert "observeItems()" in masonry
    # Custom properties in, static CSS out; no generated stylesheet, no DOM id.
    for name in ("--masonry-w", "--masonry-x", "--masonry-y", "--masonry-h"):
        assert f"setProperty('{name}'" in masonry, name
    for gone in ("createElement('style')", "document.head", "getElementById", "data-masonry-id", "masonryId", "nth-child", "textContent"):
        assert gone not in masonry, gone
    assert "resizeObserver.disconnect();" in masonry
    assert "itemResizeObserver.disconnect();" in masonry
    assert "mutationObserver.disconnect();" in masonry
    assert "cancelAnimationFrame(frame)" in masonry
    widgets_block = css.split(".widgets-list {", 1)[1].split("}", 1)[0]
    assert "display: grid" not in widgets_block
    assert "position: relative;" in widgets_block
    assert "height: var(--masonry-h, auto);" in widgets_block
    card_block = css.split("height: var(--masonry-h, auto);", 1)[1].split(".widgets-card {", 1)[1].split("}", 1)[0]
    assert "width: var(--masonry-w, auto);" in card_block
    assert "transform: translate(var(--masonry-x, 0px), var(--masonry-y, 0px));" in card_block
    # `widgets-card-span-2` is the JS span signal only (masonry reads the class);
    # the inert `grid-column: span 2` rules from the grid era are gone (CA-9).
    assert ".widgets-card-span-2" not in css


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


def test_widget_fault_status_wraps_inside_narrow_card():
    style = _read("web/style.css")
    root = style.split(":root {", 1)[1].split("}", 1)[0]
    controls = style.split(".widgets-card-controls {", 1)[1].split("}", 1)[0]
    status = style.split(".widgets-card-controls .ui-status:not([hidden]) {", 1)[1].split("}", 1)[0]
    assert "--widget-fault-status-max-width: min(240px, 35vw);" in root
    assert "flex-wrap: wrap;" in controls
    assert "min-width: 0;" in controls
    assert "min-width: 0;" in status
    assert "max-width: var(--widget-fault-status-max-width);" in status
    assert "white-space: normal;" in status
    assert "overflow-wrap: anywhere;" in status


def test_widgets_card_order_is_owner_ui_preference():
    """The reorder handles live in ``widget_reorder.js``; the card markup and
    the preference write stay in the page host. Phase 3: a reorder (drag or
    keys) is a pure move in the KEY order (``moveWidgetKey``) handed back to the
    page, which re-sorts, relayouts through masonry and persists — no
    ``<article>`` is ever moved, so a running frame never reloads on reorder.
    Disclosed residual: the Tab/focus order follows the DOM and may differ from
    the visible order until a window reload rebuilds the cards."""
    source = _widgets_js()
    reorder = _read("web/modules/widget_reorder.js")
    css = (REPO_ROOT / "web" / "style.css").read_text(encoding="utf-8")
    api_client = (REPO_ROOT / "web" / "modules" / "api_client.js").read_text(encoding="utf-8")

    assert 'data-widget-reorder-handle' in source
    assert "from './widget_reorder.js'" in source
    assert "export function sortTabsByWidgetOrder" in reorder
    assert "originalIndex" in reorder
    assert "return a.originalIndex - b.originalIndex;" in reorder
    assert "Move widget: drag or use arrow keys" in source
    assert "handle.addEventListener('keydown'" in reorder
    assert "event.key === 'ArrowUp'" in reorder
    assert "apiClient.uiPreferences()" in source
    assert "apiClient.saveUiPreferences({ widget_order: normalized })" in source
    assert "export function moveWidgetKey(order, key, toIndex)" in reorder
    assert "export function bindWidgetCardReorder(list, currentOrder, onOrderChange)" in reorder
    assert "bindWidgetCardReorder(list, currentWidgetOrder, persistWidgetOrder);" in source
    for moved in (".before(", ".after(", ".prepend(", ".append(", "insertBefore", "appendChild", "replaceWith"):
        assert moved not in reorder, moved
    # The keyed patch inserts and replaces nodes but never moves one.
    assert "anchor.after(" not in source
    assert "list.prepend(" not in source
    assert "previousLiveCard" not in source
    assert ".widgets-card-drag" in css
    assert ".widgets-card.drag-over" in css
    assert "uiPreferences: () => fetchJson('/api/ui/preferences'" in api_client
    assert "saveUiPreferences: (payload) => jsonPost('/api/ui/preferences', payload)" in api_client


def test_widgets_inline_card_host_path_removed():
    source = _widgets_js()
    framed = _framed_widget_sources()
    assert "render.kind === 'inline_card'" not in framed
    assert "skill-widget-weather" not in framed
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
    assert "components[Number(" not in _framed_widget_sources()
    assert "data-widget-kanban-key" in source


def test_widgets_forms_charts_and_kanban_keep_host_owned_contracts():
    """The chart helpers (config, finite-value coercion, accessible data table)
    and the shared dotted-path reader moved unchanged into
    ``widget_chart.js`` (phase 3 made room in widgets.js); the table cell
    renderer with its number formatter and http(s)-only link guard followed in
    the cycle-A fix round."""
    source = _widgets_js()
    helper = _read("web/modules/ui_helpers.js")
    chart = _read("web/modules/widget_chart.js")
    assert "renderSafeField(" in source
    assert "collectSafeFieldValues(" in source
    assert "includePasswords: false" in source
    assert "pendingActions.has(key)" in source
    assert "from './widget_chart.js'" in source
    assert "export function getPath(root, path, fallback = '')" in chart
    assert "export function formatNumber(value, precision)" in chart
    assert "export function renderTableCell(row, column)" in chart
    assert "['http:', 'https:'].includes(parsed.protocol)" in chart
    assert "spanGaps: false" in chart
    assert "export function finiteChartValue" in chart
    assert "data.map(finiteChartValue)" in chart
    assert "aria-label=" in source
    assert "renderChartDataTable(config, label, !chartAvailable)" in source
    assert "export function renderChartDataTable" in chart
    assert "data-widget-kanban-move" in source
    assert "widget-kanban-empty" in source
    assert "mount.querySelectorAll('[data-widget-kanban-key]')" in source
    assert "{ card_id: cardId, column_id: columnId }" in source
    assert "SAFE_FIELD_TYPES" in helper
    assert "autocomplete=\"new-password\"" in helper


def test_widgets_responsive_design_system_styles_are_host_owned():
    style = _read("web/style.css")
    assert ".widget-group-grid" in style
    assert ".widget-metric" in style
    assert ".widget-callout" in style
    assert ".widget-form-fields.widget-grid-cols-4" in style
    assert "content: attr(data-label);" in style
    assert ".widget-kanban-move" in style
    assert ".widget-kanban-col.is-empty" in style
    # The widget narrow block is found by its CONTENT, not by being the last
    # `@media (max-width: 640px)` in the file. Position was never the fact under
    # test, and any surface that later adds its own 640px block (the Agents tab's
    # account rows did) would silently steal this assertion and fail it.
    blocks = style.split("@media (max-width: 640px) {")[1:]
    narrow = [b for b in blocks if ".widget-kanban-col.is-empty" in b]
    assert narrow, "no @media (max-width: 640px) block carries the widget kanban rules"
    assert any("min-height: 0;" in b for b in narrow)
    assert any("padding-block: 8px;" in b for b in narrow)
    assert ".widget-group-components > * { margin-top: 0; }" in style
    assert "repeat(auto-fit, minmax(min(220px, 100%), 1fr))" in style
    assert ".widget-group-grid > .widget-group-components > :is(" in style


def test_widget_public_tones_share_the_host_normalizer_and_canonical_css():
    source = _widgets_js()
    helper = _read("web/modules/ui_helpers.js")
    style = _read("web/style.css")
    assert "function widgetTone" not in _framed_widget_sources()
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


def test_widgets_module_bridge_is_one_streaming_grammar():
    """Widgets lifecycle phase 4: the module frame has ONE parent-mediated I/O
    grammar on the existing nonce. Child → parent: ``ouro-widget-fetch``,
    ``ouro-widget-fetch-abort``, ``ouro-widget-events {subscribe|unsubscribe}``
    and the ``ouro-widget-disposed`` ack. Parent → child: ``ouro-widget-fetch-chunk``
    frames (``headers`` with status/statusText/every header first, one ``data``
    frame per body chunk as a transferred ArrayBuffer, then ``end``; ``error`` on
    failure) and ``ouro-widget-event`` for the skill's ``ws_prefix`` WebSocket
    messages. The child rebuilds a real ``Response`` over a ``ReadableStream``
    (binary by default; null body for HEAD/204/205/304). Negative pins: the old
    string ``-fetch-result`` path and its ``r.text()`` are gone with no alias,
    the frame opens no ``EventSource``/``WebSocket`` of its own, and no default
    timeout constant is applied to a bridged fetch — the only timer in the relay
    is the author's opt-in ``init.timeoutMs`` (declarative requests and the
    module source load keep ``WIDGET_REQUEST_TIMEOUT_MS``)."""
    child = _read("web/modules/widget_frame.js")
    parent = _read("web/modules/widget_module.js")
    page = _widgets_js()
    framed = _framed_widget_sources()
    # Child grammar and Response construction.
    for name in (
        "type: 'ouro-widget-fetch'", "type: 'ouro-widget-fetch-abort'", "type: 'ouro-widget-events'",
        "'ouro-widget-fetch-chunk'", "'ouro-widget-event'", "type: 'ouro-widget-disposed'",
        "type: 'ouro-widget-error'",
    ):
        assert name in child, name
    assert "new ReadableStream({" in child
    assert "resolve(new Response(stream, {" in child
    assert "const nullBody = method === 'HEAD' || [204, 205, 304].includes(Number(msg.status));" in child
    assert "headers: Array.from(new Headers(init.headers || {}))," in child
    assert "timeoutMs: init.timeoutMs ?? null," in child
    assert "signal?.addEventListener('abort', onAbort, { once: true });" in child
    assert "const onEvent = (callback) =>" in child
    assert "post({ type: 'ouro-widget-events', op: 'subscribe' });" in child
    assert "post({ type: 'ouro-widget-events', op: 'unsubscribe' });" in child
    assert "window.fetch = request;" in child
    # Parent relay: prefix check, headers → data (transferred) → end, error; abort;
    # event forwarding under the card's ws_prefix through the page's handler Set.
    relay = parent.split("const relayFetch = async (msg) =>", 1)[1].split("const onMessage = (event) =>", 1)[0]
    assert "module widget fetch outside extension route prefix" in relay
    # The prefix is checked once, before the request. A followed redirect would
    # carry the frame's method, body, headers and the owner's session wherever the
    # hop points, past that check, so the relay refuses the hop instead.
    assert "redirect: 'error'," in relay
    assert relay.index("phase: 'headers'") < relay.index("phase: 'data'") < relay.index("phase: 'end'") < relay.index("phase: 'error'")
    assert "statusText: r.statusText, headers: Array.from(r.headers)" in relay
    assert "const reader = r.body?.getReader();" in relay
    assert "const chunk = bridgeChunkBuffer(value);" in relay
    assert "frame({ phase: 'data', chunk }, [chunk]);" in relay
    assert "export function bridgeChunkBuffer(view)" in child
    assert "pendingRequests.get(msg.id)?.abort();" in parent
    assert "if (msg.type === 'ouro-widget-fetch') relayFetch(msg);" in parent
    assert "const wsPrefix = String(tab.ws_prefix || '').trim();" in parent
    assert "if (!wsPrefix || !type.startsWith(wsPrefix)) return;" in parent
    assert "post({ type: 'ouro-widget-event', event: type.slice(wsPrefix.length), data: msg.data ?? {} });" in parent
    assert "if (msg.op === 'subscribe') messageHandlers?.add(onWsMessage);" in parent
    assert "else if (msg.op === 'unsubscribe') messageHandlers?.delete(onWsMessage);" in parent
    assert "return mountModuleWidget(mount, tab, render, mountSignal, widgetMessageHandlers);" in page
    # Dispose tail: open streams aborted with the pending requests, forwarding dropped.
    tail = parent.split("const finish = () =>", 1)[1]
    assert tail.index("pendingRequests.forEach((controller) => controller.abort());") < tail.index("messageHandlers?.delete(onWsMessage);")
    assert tail.index("messageHandlers?.delete(onWsMessage);") < tail.index("window.removeEventListener('message', onMessage);")
    # Negative pins: no string path, no alias, no second transport, no default bridge timeout.
    assert "ouro-widget-fetch-result" not in framed
    assert ".text()" not in relay
    assert "WIDGET_REQUEST_TIMEOUT_MS" not in relay
    assert "withWidgetRequestTimeout" not in relay
    assert relay.count("setTimeout(") == 1
    assert "Number.isFinite(timeoutMs) && timeoutMs > 0" in relay
    for forbidden in ("EventSource(", "WebSocket(", "XMLHttpRequest", "setTimeout("):
        assert forbidden not in child, forbidden
    jobs = _read("web/modules/widget_job.js")
    assert "BRIDGE" not in jobs and "bridge" not in jobs.split("Ordered stop", 1)[0]


def test_widgets_module_frame_faults_reach_the_card_status_slot():
    """A module widget's frame has a bounded fault channel, and the height cap is
    readable from the DOM.

    In-frame script errors, unhandled rejections and CSP refusals are posted as
    one ``ouro-widget-error`` message per distinct kind+message (deduplicated,
    capped, clipped) through the existing nonce-bound ``post`` helper, and the
    parent writes them into the card's own ``[data-widget-status]`` slot. The
    lifecycle state is deliberately untouched: the frame is still mounted, so
    ``syncWidgetCardControls`` is NOT called with an error state, which would
    flip a live Stop button back to Start. All three listeners are registered on
    ``window`` (``securitypolicyviolation`` bubbles there, and the child bridge
    runs in hosts with no ``document``) and all three are removed in the single
    ``dispose()``. The two height-cap attributes are stamped BEFORE the
    ``nextHeight === appliedHeight`` early return, which is exactly the case they
    exist to explain."""
    child = _read("web/modules/widget_frame.js")
    parent = _read("web/modules/widget_module.js")
    card = _read("web/modules/widget_card.js")

    # Child: three window listeners, one bounded post, removed on dispose.
    for listener in ("'error'", "'unhandledrejection'", "'securitypolicyviolation'"):
        assert f"window.addEventListener({listener}" in child, listener
        assert f"window.removeEventListener({listener}" in child, listener
    assert "document.addEventListener('error'" not in child
    assert "if (faultCount >= 10 || seenFaults.has(key)) return;" in child
    assert ".slice(0, 500)" in child and ".slice(0, 200)" in child

    # Parent: routed to the status slot only; no lifecycle flip.
    assert "msg.type === 'ouro-widget-error'" in parent
    assert "setWidgetCardFault(mount.closest('[data-widget-key]')" in parent
    assert "syncWidgetCardControls(card, 'error'" not in parent
    assert "export function setWidgetCardFault(card, text)" in card
    assert "status.dataset.tone = 'error';" in card

    # Ordering regression pin: the stamps precede the early return.
    assert parent.index("widgetFrameCapped") < parent.index("nextHeight === appliedHeight")
