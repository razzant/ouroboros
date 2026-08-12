"""Static contracts for the responsive navigation shell (v6.32.0).

These checks pin the high-risk parts of the multi-project navigation rewrite:
desktop sidebar + mobile drawer share one DOM/state model, project rows use
explicit slots, and the old bottom icon rail does not come back.
"""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _read(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


def test_navigation_shell_dom_has_sidebar_drawer_and_project_slots():
    html = _read("web/index.html")
    chat_js = _read("web/modules/chat.js")
    assert 'id="primary-sidebar"' in html
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
    assert "if (!isMain) return;" in chat_js  # ensureWelcomeMessage is main-only
    assert "padding: 10px 292px" not in css
    assert "right: 8px;\n    bottom: 6px" not in css
    assert ".chat-text-row:focus-within" in css
    assert ".chat-toolbar-row {\n    order: 1;" in css
    assert ".chat-text-row {\n    order: 2;" in css
    assert "chat-header-more" in chat_js
    assert "chat-header-menu" in css
    # Flat redesign: the project panel is a solid rail surface, not a glass pane.
    # The whole main SPA is blur-free (the standalone onboarding wizard keeps its
    # own formulas in web/onboarding.css).
    assert "backdrop-filter" not in css.split("*/", 1)[1]
    assert "background: var(--bg-panel);" in css
    # Project threads (T1): the right-side project panel is RETIRED — a thread
    # opens in the CENTRE stage instead, so the panel aside, its backdrop and
    # their slide-in transition must be gone rather than merely unused. The
    # backdrop was the mobile failure this phase fixes: a second full-screen
    # surface stacked over the content area with its own close affordance.
    html = _read("web/index.html")
    assert "project-panel-backdrop" not in html
    assert 'id="project-panel"' not in html
    assert ".project-panel-backdrop" not in css
    assert ".project-panel.open" not in css
    assert "body.project-panel-open" not in css
    # The lean bar/title/close chrome SURVIVES: the centre stage reuses it.
    assert ".project-panel-bar" in css
    assert ".thread-stage-bar" in css
    assert ".chat-header-actions {\n        display: none;" not in css
    # Gateway Boundary: chat.js consumes the endpoint via the api_client wrapper,
    # and the raw route lives in api_client.js (not a raw fetch in chat.js).
    assert "projectFromTask" in chat_js
    assert "/api/projects/from-task" in _read("web/modules/api_client.js")


def test_chat_header_controls_reorg_and_more_autodismiss():
    chat_js = _read("web/modules/chat.js")
    # Evolve / Review / Restart are ghost buttons and Panic is the one danger
    # button; all four are always visible in the header.
    for command in ("evolve", "review", "restart"):
        assert f'class="chat-header-btn" type="button" data-chat-command="{command}"' in chat_js
    assert 'class="chat-header-btn danger" type="button" data-chat-command="panic"' in chat_js
    # More is a slim overflow holding ONLY Consciousness now.
    menu = chat_js.split('class="chat-header-menu"', 1)[1].split("</details>", 1)[0]
    assert 'data-chat-command="bg"' in menu
    assert menu.count("chat-header-menu-item") == 1
    assert 'data-chat-command="evolve"' not in menu
    assert 'data-chat-command="review"' not in menu
    # The More <details> auto-collapses on an outside click or Escape (never sticks).
    assert "details.chat-header-more[open]" in chat_js
    assert "event.key === 'Escape'" in chat_js
    # The budget meter left the chat header for the sidebar; chat.js keeps only
    # the formatting projection, not a pill.
    assert "chat-budget" not in chat_js
    assert "headerBudgetPresentation" in chat_js


def test_sidebar_brand_sections_and_budget_block():
    html = _read("web/index.html")
    css = _read("web/style.css")
    app_js = _read("web/app.js")
    # Brand row: 26px app mark, product name, version + liveness sub line.
    assert 'class="nav-brand" id="nav-brand"' in html
    assert 'src="/static/favicon.png"' in html
    assert 'id="nav-version"' in html
    assert 'class="nav-status-dot"' in html
    assert "border-radius: var(--radius-7);" in css
    # Sections: Main Chat, Projects (unchanged mechanics), Workspace, System.
    assert '<div class="nav-section-label">Workspace</div>' in html
    assert '<div class="nav-section-label">System</div>' in html
    assert "Utilities" not in html
    assert 'data-nav-page="changes"' in html
    assert 'id="nav-projects-list" class="nav-projects-list"' in html
    assert "changes: icon(" in _read("web/modules/page_icons.js")
    # Budget block pinned at the sidebar bottom; the fill is a CSS custom
    # property (the accepted dynamic-value exception), never a style.width write.
    assert 'class="nav-budget" id="nav-budget"' in html
    assert 'id="nav-budget-amount"' in html
    assert 'id="nav-budget-bar"' in html
    assert "width: var(--budget-fill, 0%);" in css
    assert "navBudgetBar.style.setProperty('--budget-fill'" in app_js
    assert ".style.width =" not in app_js
    assert ".style.width=" not in app_js


def test_single_api_state_poll_owner():
    """ONE poll owner, and its RULES are covered by tests rather than by grepping.

    The scheduling/coalescing core lives in `web/modules/state_poll.js` and is
    exercised by `web/tests/state_poll.test.js` against a fake clock — single-flight
    dedup, re-arm on settle (including a rejected read), the hidden-document pause,
    subscriber fan-out, replay and unsubscribe. What is worth PINNING statically is
    only what a node test cannot see: that app.js still wires that core to the real
    fetch, live page/visibility getters, and the real timer functions, and that the
    two timers it replaced have not grown back.
    """
    app_js = _read("web/app.js")
    chat_js = _read("web/modules/chat.js")
    core_js = _read("web/modules/state_poll.js")
    core_test = _read("web/tests/state_poll.test.js")

    # The core is imported and wired, not reimplemented alongside.
    assert "import { createStatePoll } from './modules/state_poll.js';" in app_js
    assert "createStatePoll({" in app_js
    assert "read: readStateSnapshot," in app_js
    # Live GETTERS: the cadence must follow navigation and tab visibility, so a value
    # captured once at startup would silently freeze the interval.
    # An open project THREAD takes the chat cadence: it IS a chat surface.
    assert "activePage: () => (state.activePage === 'thread' ? 'chat' : state.activePage)," in app_js
    assert "hidden: () => document.hidden," in app_js
    assert "async function readStateSnapshot()" in app_js
    # The cadence and the pause are the CORE's decisions now — pinned where they live.
    assert "activePage === 'chat' ? STATE_POLL_CHAT_MS : STATE_POLL_IDLE_MS" in core_js
    assert "if (hidden()) return;" in core_js
    assert "STATE_POLL_CHAT_MS = 3000" in core_js
    assert "STATE_POLL_IDLE_MS = 20000" in core_js
    # ...and those decisions are executably covered, not just spelled correctly.
    for behaviour in ("ONE request, not three", "re-arms on SETTLE", "PAUSES the timer"):
        assert behaviour in core_test, behaviour
    # No second copy of the poll machinery in app.js.
    assert "statePollInFlight" not in app_js
    # ...and no LEFTOVER reference to the deleted timer handle either: the visibility
    # handler kept `clearTimeout(statePollTimer)` after the handle moved inside the
    # core, which is a ReferenceError on every tab hide, not a dead line.
    assert "statePollTimer" not in app_js
    assert "statePoll.stop();" in app_js
    assert "function publishState(" not in app_js
    # The two old timers are gone.
    assert "setInterval(refreshProjectsNav" not in app_js
    assert "setInterval(refreshHeaderControlState" not in chat_js
    assert "refreshHeaderControlState" not in chat_js
    # Chat consumes the shared snapshot instead of polling.
    assert "subscribeState(syncHeaderControlState)" in chat_js
    # Preserved refresh entry points.
    assert "refreshProjectsNav().finally(() => ws.connect());" in app_js
    assert "if (cid) state.projectChatIds.add(cid);" in app_js


def test_right_panel_state_machine_and_stream_anchors():
    app_js = _read("web/app.js")
    css = _read("web/style.css")
    assert "panelKind" in app_js
    assert "function registerRightPanel(kind, handlers)" in app_js
    assert "async function openRightPanel(kind, opts = {})" in app_js
    assert "function closeRightPanel({ sync = true } = {})" in app_js
    # `project` is no longer a right-panel kind — a thread owns the centre — but
    # the NAME stays reserved so a module cannot re-register the retired panel.
    assert "navState.panelKind = 'project';" not in app_js
    assert "if (!name || name === 'project') return () => {};" in app_js
    assert "const threadStage = createThreadStage({" in app_js
    assert "--inspector-width: 320px;" in css
    # Append-only ownership anchors for the parallel streams.
    assert "/* [anchor:phase-B] right-panel registrations */" in app_js
    assert "/* [anchor:phase-C] global capture hotkey */" in app_js
    assert "[stream A: chat" in css
    assert "[stream B: changes screen + task inspector]" in css
    assert "[stream C: files" in css
    assert "[stream D: dashboard, skills, settings, widgets]" in css
    # Empty Changes container registered by the shell.
    assert "changesPage.id = 'page-changes';" in app_js


def test_chat_controller_is_kept_and_exposes_parts_adapters():
    app_js = _read("web/app.js")
    chat_js = _read("web/modules/chat.js")
    assert "const chatController = initChat(ctx);" in app_js
    assert "export function getChatController()" in app_js
    assert "function setDraftParts(parts)" in chat_js
    assert "function sendParts(parts" in chat_js
    # The adapters still ride the EXISTING sendMessage transport authority and the
    # ONE codec, so the serialized string stays the message identity. Phase A
    # swapped WHERE the parts land: the Phase-0 adapter wrote the serialized text
    # into the bare textarea (`input.value = text`), the parts editor now takes the
    # ordered parts themselves, so a handed-over chip arrives as a removable CHIP.
    assert "const text = serializeParts(parts);" in chat_js
    assert "composerParts.setParts([...kept, ...normalizeParts(parts)]);" in chat_js
    assert "createComposerParts({" in chat_js
    # ...and the send path reads the whole field (parts + typed draft) through the
    # shared serializer rather than the raw textarea value.
    assert "let text = composerParts.serialize().trim();" in chat_js
    # A handoff ADDS to the chat composer; it never wipes a draft already there.
    assert "const kept = composerParts.commitDraft();" in chat_js
    # sendParts reports the REAL dispatch outcome, so a source dock never clears the
    # owner's draft after a refused or failed send.
    assert ".then((ok) => ok === true)" in chat_js
    assert ".then(() => true)" not in chat_js


def test_matrix_rain_is_gone():
    for rel in ("web/app.js", "web/modules/utils.js", "web/style.css", "web/index.html"):
        assert "matrix" not in _read(rel).lower(), rel


def test_inline_nav_glyphs_byte_match_the_canonical_page_icons():
    """Every sidebar row's inline SVG is a pre-hydration FALLBACK, not a second glyph.

    `hydrateNavIcons` replaces these at boot from `PAGE_ICONS`, so the inline copy is
    what the owner sees for one frame — and any drift between the two is a flicker
    into a different icon, invisible to whoever edits only one of the files. The
    fallback pattern itself is fine; two spellings of the same glyph is the defect.
    """
    icons = _read("web/modules/page_icons.js")
    html = _read("web/index.html")
    checked = 0
    for key in ("changes", "files", "skills", "widgets", "dashboard", "settings"):
        canonical = re.search(rf"{key}: icon\('(.*?)'\)", icons, re.S)
        assert canonical, f"{key} missing from PAGE_ICONS"
        inline = re.search(
            rf'data-nav-page="{key}".*?<svg[^>]*>(.*?)</svg>', html, re.S,
        )
        assert inline, f"{key} nav row has no inline svg"
        assert inline.group(1) == canonical.group(1), (
            f"{key}: inline nav glyph diverged from PAGE_ICONS"
        )
        checked += 1
    assert checked == 6


def test_an_unavailable_thread_menu_row_is_greyed_not_only_inert():
    """T4 vision pass: `disabled` in the DOM is not a visible state.

    The thread menu greys an action it cannot offer and keeps its reason on the
    row (rather than omitting it, which teaches nothing). `.project-row-menu
    button` had no `:disabled` rule at all, so `Merge back` on a thread with no
    checkout rendered at full contrast with `cursor: pointer` and a live hover —
    indistinguishable from `Fork` beside it, with the reason only in a tooltip.
    Every DOM assertion was green; only the rendered menu showed it.

    Pinned here because the failure is invisible to the Playwright DOM checks
    that already cover this menu: `[disabled]` was — and still is — present.
    """
    css = _read("web/style.css")
    assert ".project-row-menu button:disabled {" in css
    disabled_rule = css.split(".project-row-menu button:disabled {", 1)[1].split("}", 1)[0]
    assert "opacity" in disabled_rule
    assert "cursor: not-allowed" in disabled_rule
    # ...and the hover must not light a row the owner cannot use.
    assert ".project-row-menu button:disabled:hover {" in css
    hover_rule = css.split(".project-row-menu button:disabled:hover {", 1)[1].split("}", 1)[0]
    assert "background: transparent" in hover_rule
