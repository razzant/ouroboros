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
    assert 'data-nav-page="chat"' in html
    assert 'class="nav-row-label"' in html
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
            "web/modules/marketplace.js",
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
    assert "grid-template-columns: var(--sidebar-width) minmax(0, 1fr) auto;" in css
    # One client-side route, read once on load and validated against the DOM
    # rather than a duplicated page list; never written back on navigation
    # (the desktop shell and the Telegram mini app have no address bar).
    assert "function pageFromHash()" in app_js
    assert "document.getElementById(`page-${name}`)" in app_js
    assert "hashchange" not in app_js
    assert "replaceState" not in app_js
    assert ".mobile-nav-toggle {\n        position: fixed" not in css
    assert "flex: 0 0 44px;" in css


def test_page_header_is_shared_foundation_for_top_pages():
    header_js = _read("web/modules/page_header.js")
    css = _read("web/style.css")
    assert "export function renderMobileNavToggle()" in header_js
    assert "toolbarHtml" in header_js
    assert "trailingHtml" in header_js
    assert "app-page-toolbar app-page-actions" in header_js
    assert "grid-template-areas:" in css
    assert '"leading title toolbar"' in css
    assert '"tabs tabs tabs"' in css
    assert "align-items: start;" in css
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
    assert "Show more" not in app_js
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
    assert "if (!isMain) return;" in chat_js  # ensureWelcomeMessage is main-only
    assert "padding: 10px 292px" not in css
    assert "right: 8px;\n    bottom: 6px" not in send_group_css
    assert ".chat-text-row:focus-within" in css
    assert ".chat-toolbar-row {\n    order: 1;" in css
    assert ".chat-text-row {\n    order: 2;" in css
    assert "chat-header-more" in chat_js
    assert "chat-header-menu" in css
    assert "backdrop-filter: blur(22px)" in css
    assert 'id="project-panel-backdrop" class="project-panel-backdrop"' in _read("web/index.html")
    assert ".project-panel-backdrop" in css
    assert "transition: transform 180ms ease, opacity 180ms ease;" in css
    assert ".project-panel.open" in css
    assert "left: var(--sidebar-width);" in css  # sidebar stays clickable under backdrop
    assert ".chat-header-actions {\n        display: none;" not in css
    # Gateway Boundary: chat.js consumes the endpoint via the api_client wrapper,
    # and the raw route lives in api_client.js (not a raw fetch in chat.js).
    assert "projectFromTask" in chat_js
    assert "/api/projects/from-task" in _read("web/modules/api_client.js")


def test_chat_header_controls_reorg_and_more_autodismiss():
    chat_js = _read("web/modules/chat.js")
    # Restart + Panic are promoted to always-visible header buttons...
    assert 'class="chat-header-btn" type="button" data-chat-command="restart"' in chat_js
    assert 'class="chat-header-btn danger" type="button" data-chat-command="panic"' in chat_js
    # ...and Consciousness / Evolve / Review move into the More overflow menu.
    menu = chat_js.split('class="chat-header-menu"', 1)[1].split("</details>", 1)[0]
    assert 'data-chat-command="bg"' in menu
    assert 'data-chat-command="evolve"' in menu
    assert 'data-chat-command="review"' in menu
    # The More <details> auto-collapses on an outside click or Escape (never sticks).
    assert "details.chat-header-more[open]" in chat_js
    assert "event.key === 'Escape'" in chat_js
