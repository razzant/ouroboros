"""Static contracts for the responsive navigation shell (v6.32.0).

These checks pin the high-risk parts of the multi-project navigation rewrite:
desktop sidebar + mobile drawer share one DOM/state model, project rows use
explicit slots, and the old bottom icon rail does not come back.
"""

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


def test_navigation_shell_dom_has_sidebar_drawer_and_project_slots():
    html = _read("web/index.html")
    chat_js = _read("web/modules/chat.js")
    assert 'id="primary-sidebar"' in html
    # The sidebar is the document's one navigation landmark.
    assert '<nav id="primary-sidebar"' in html
    assert '<aside id="primary-sidebar"' not in html
    assert '<aside id="project-panel"' in html
    assert 'data-mobile-nav-toggle' not in html
    assert 'id="nav-drawer-backdrop"' in html
    # The rows come from the page registry (pages.js), rendered by app.js into the sidebar's zones.
    pages_js = _read("web/modules/pages.js")
    assert 'data-nav-page="${escapeHtml(page.id)}"' in pages_js
    assert '<span class="nav-row-label">${escapeHtml(page.label)}</span>' in pages_js
    assert 'id="nav-projects-list" class="nav-projects-list"' in html
    # The mobile trigger is shell-level (available on Skills/Dashboard/etc),
    # not a chat-only duplicate.
    assert 'data-mobile-nav-toggle' not in chat_js
    assert "data-mobile-nav-toggle" in _read("web/modules/page_header.js")


def test_navigation_shell_replaces_legacy_rail_selectors():
    combined = "\n".join(
        _read(path)
        for path in (
            "web/index.html",
            "web/app.js",
            "web/style.css",
            "web/modules/page_icons.js",
            "web/modules/skills.js",
            "web/modules/skill_browse.js",
        )
    )
    assert "#nav-rail" not in combined
    assert ".nav-btn" not in combined
    assert "data-page=" not in combined
    assert "--nav-width" not in combined


def test_navigation_state_and_mobile_drawer_are_first_class():
    app_js = _read("web/app.js")
    css = _read("web/style.css")
    assert "const navState = {" in app_js
    assert "function syncNavigationState()" in app_js
    assert "mobileDrawerOpen" in app_js
    assert "navState.activeProjectId" in app_js
    assert "#primary-sidebar.open" in css
    assert ".nav-drawer-backdrop" in css
    assert ".mobile-nav-toggle" in css
    # The sidebar's column follows the sidebar's own box.
    assert "grid-template-columns: auto minmax(0, 1fr);" in css.split("\n#app {", 1)[1].split("}", 1)[0]
    assert "#app > #divider { display: none; }" in css
    assert "#content {\n    grid-column: 2;" in css
    # One client-side route, read once on load and validated against the DOM
    # rather than a duplicated page list; never written back on navigation
    # (the desktop shell and the Telegram mini app have no address bar).
    assert "function pageFromHash()" in app_js
    assert "document.getElementById(`page-${name}`)" in app_js
    assert "hashchange" not in app_js
    assert "replaceState" not in app_js
    assert "pushState" not in app_js
    # The one history write is the layer stack's same-URL sentinel: pushed only while a layer is
    # open, popped by Back, consumed by a programmatic close (web/tests/layers.test.js).
    layers_js = _read("web/modules/layers.js")
    assert "history.pushState(LAYER_STATE, '')" in layers_js
    assert layers_js.count("pushState(") == 1
    assert "if (stack.length && !hist.armed)" in layers_js
    assert "replaceState" not in layers_js and "hashchange" not in layers_js and "location." not in layers_js
    assert "const layers = createLayerStack();" in app_js
    assert "layers.push({ id: 'drawer'" in app_js and "layers.push({ id: 'project'" in app_js
    assert ".mobile-nav-toggle {\n        position: fixed" not in css
    assert "flex: 0 0 44px;" in css


def test_page_header_is_shared_foundation_for_top_pages():
    header_js = _read("web/modules/page_header.js")
    css = _read("web/style.css")
    assert "export function renderMobileNavToggle({ inComposer = false } = {})" in header_js  # one ☰, a page's or a room's
    assert "toolbarHtml" in header_js
    assert "trailingHtml" in header_js
    # Actual toolbar class composition/escaping is exercised by page_header.test.js.
    assert "'app-page-toolbar'" in header_js and "'app-page-actions'" in header_js
    # One row: leading (☰) · title (with the room's status) · toolbar/trailing; a tab strip is its own strip, so
    # the row stays one row (P5). A page has no caption line: its sidebar row and its tabs name it.
    assert 'grid-template-areas: "leading title toolbar";' in css
    assert '"tabs tabs tabs"' not in css
    assert "app-page-lede" not in header_js and "app-page-description" not in header_js
    assert "description" not in header_js.split("export function renderPageHeader(", 1)[1].split(") {", 1)[0]
    for rel in ("web/modules/dashboard.js", "web/modules/settings_ui.js", "web/modules/skills.js", "web/modules/widgets.js"):
        page = _read(rel)
        header_call = page.split("renderPageHeader({", 1)[1].split("})", 1)[0]
        assert "description:" not in header_call, f"{rel} passes a caption line to its header"
    assert "app-page-icon" not in header_js and "app-page-icon" not in css
    # The Project panel's bar is the same component, rendered by app.js; index.html keeps no hand-written bar.
    html = _read("web/index.html")
    assert "project-panel-bar" not in html and "project-panel-close" not in html
    app_js = _read("web/app.js")
    assert "titleId: 'project-panel-title'" in app_js and "className: 'project-panel-bar project-inspector-header'" in app_js
    assert "named: false" in app_js
    assert "import { renderPageHeader } from './modules/page_header.js';" in app_js
    for path in (
        "web/modules/chat.js",
        "web/modules/files.js",
        "web/modules/skills.js",
        "web/modules/widgets.js",
        "web/modules/dashboard.js",
        "web/modules/settings_ui.js",
    ):
        assert "renderPageHeader({" in _read(path)


def test_project_rows_use_slots_not_generic_spans():
    app_js = _read("web/app.js")
    css = _read("web/style.css")
    assert "PROJECTS_VISIBLE_LIMIT" not in app_js
    assert "className = 'nav-row nav-project-row'" in app_js
    # Project rows are a dotless, compact, indented list (no green status dot).
    assert "nav-project-dot" not in app_js
    assert "nav-project-dot" not in css
    assert "className = 'nav-row-label'" in app_js
    assert ".nav-project-btn span" not in css
    assert ".nav-projects {" in css and ".nav-projects-list" in css
    assert "['active', 'deleting'].includes" in app_js
    assert "p.visible_revision" in app_js
    assert "#page-skills,\n#page-widgets {\n    flex-direction: column;\n    min-height: 0;\n    overflow: hidden;" in css
    assert "#page-skills,\n#page-widgets {\n    flex-direction: column;\n    min-height: 0;\n    padding:" not in css


def test_project_panel_composer_and_welcome_contracts():
    chat_js = _read("web/modules/chat.js")
    css = _read("web/style.css")
    send_group_css = css.split(".chat-send-group {", 1)[1].split(
        ".chat-send-group[data-busy", 1
    )[0]
    # The empty-Main greeting is host copy: mounted for Main only, withdrawn when a
    # recent read starts and decided by that read's own coverage (a failed read keeps it
    # withdrawn), set as text and never minted as a chat bubble. What counts as an empty
    # feed is one rule shared with the history loading state. Its preference is hidden:
    # Settings has no editor.
    welcome_js = _read("web/modules/welcome_preference.js")
    assert "welcome" not in _read("web/modules/settings.js").lower()
    assert "welcome" not in _read("web/modules/settings_ui.js").lower()
    assert "const emptyWelcome = isMain ? mountEmptyChatWelcome(messagesDiv) : null;" in chat_js
    sync_start = chat_js.split("historySyncPromise = (async () => {", 1)[1].split("await fetchHistory(null);", 1)[0]
    assert "emptyWelcome?.historyPending();" in sync_start
    assert "emptyWelcome?.historyRead(data.window?.complete === true);" in chat_js
    assert "lastHistorySyncSucceeded = false;\n                emptyWelcome?.historyRead(false);" in chat_js
    assert "import { feedIsEmpty } from './chat_render_batch.js';" in welcome_js
    assert "if (!confirmedEmpty || !copy || !feedIsEmpty(messages)) {" in welcome_js
    assert "node.lastElementChild.textContent = copy;" in welcome_js
    assert "addMessage" not in welcome_js
    assert "padding: 10px 292px" not in css
    assert "right: 8px;\n    bottom: 6px" not in send_group_css
    # The field is one object under the row of capsules: it takes the focus ring, the text row inside it none.
    assert ".chat-field:focus-within" in css and ".chat-text-row:focus-within" not in css
    assert ".chat-toolbar-row {\n    order: 1;" in css
    assert ".chat-field {\n    order: 2;" in css
    # The Project is a reading surface, not a particular blur implementation.
    # Its rendered CLOSED/OPEN backing and transparent falsifier are exercised by
    # test_status_sheet_and_shade_geometry_light_dark_and_screenshots.
    panel_css = css.split(".project-panel {", 1)[1].split("}", 1)[0]
    assert "rgb(var(--shell-rgb))" in panel_css
    assert 'id="project-panel-backdrop" class="project-panel-backdrop"' in _read("web/index.html")
    assert ".project-panel-backdrop" in css
    assert "transition: transform 180ms ease, opacity 180ms ease;" in css
    assert ".project-panel.open" in css
    # The sidebar stays clickable beside the backdrop: docked, the dimmed room is the columns after the sidebar's.
    assert 'html[data-layout="docked"] .project-panel-backdrop,' in css and "grid-column: 2 / -1;" in css
    assert ".chat-header-actions {\n        display: none;" not in css
    # Gateway Boundary: Turn into project (work_portal.js, the card's move) consumes the endpoint via
    # the api_client wrapper, and the raw route lives in api_client.js (no raw fetch in a chat module).
    portal_js = _read("web/modules/work_portal.js")
    assert "apiClient.projectFromTask(" in portal_js
    assert "/api/projects/from-task" not in portal_js and "/api/projects/from-task" not in chat_js
    assert "/api/projects/from-task" in _read("web/modules/api_client.js")


def test_room_bars_hold_states_and_panic_without_more():
    chat_js = _read("web/modules/chat.js")
    app_js = _read("web/app.js")
    # A room's bar is declared once in chat_header.js; chat.js mounts Main's, app.js the Project panel's.
    header_markup = _read("web/modules/chat_header.js")
    assert "actionsHtml: mainHeaderActionsHtml()," in chat_js
    # Restart and Panic are bar buttons, each one node with one handler; Panic stands in every room's bar.
    assert 'class="chat-header-btn ui-chrome-control" type="button" data-chat-command="restart"' in header_markup
    assert 'class="chat-header-btn danger ui-chrome-control" type="button" data-chat-command="panic"' in header_markup
    assert (chat_js + header_markup).count('type="button" data-chat-command="restart"') == 1, "one Restart node, never a second copy"
    assert 'trailingHtml: `${panicButtonHtml()}<button id="project-panel-close"' in app_js
    assert "await confirmAndSendRestart({ openConfirmDialog, ws });" in header_markup
    # No More and no ⋯ on any width: a state has its own control (the consciousness chip), a command lives in the
    # Shade of the state it changes, and Evolve and Review stay in Dashboard and Settings.
    for gone in ("chat-header-more", "chat-header-menu", 'data-chat-command="bg"', 'data-chat-command="evolve"',
                 'data-chat-command="review"', "SHELL_ICONS.more"):
        assert gone not in header_markup and gone not in chat_js, gone
    assert "createConsciousnessStatus(" in header_markup
    # A narrow bar moves the same Restart and budget nodes into the room status's Shade.
    header_js = _read("web/modules/page_header.js")
    placement = header_js.split("export function bindHeaderControlPlacement(", 1)[1].split("\n}\n", 1)[0]
    assert "if (next === 'shade') {" in placement and "extras.append(restart);" in placement
    assert "headerActions.insertBefore(restart, panic);" in placement
    # Driven by the layout AND the bar's own measured width, so a narrow docked column keeps one row too.
    assert "layout === 'stack'" in placement and "bar.clientWidth < HEADER_FULL_ROW_MIN_WIDTH" in placement
    assert "onLayoutChange(place)" in placement and "observer?.observe(bar);" in placement
    assert "createRoomHeader({ bar: pageHeader, ws, shade: roomWork," in chat_js
    assert "header?.destroy();" in chat_js
