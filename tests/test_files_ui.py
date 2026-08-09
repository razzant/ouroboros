"""Static contract for the READ-ONLY Files page (tree + viewer + ⌘L capture).

The redesign made Files a reader (owner decision 18): tree rail, highlighted
per-line viewer, download, and context capture into its own composer dock. Every
mutating affordance was removed from the UI while the backend write endpoints
stay untouched, so these checks pin both halves: the removals must not creep
back, and the capture substrate (`data-line-number`, the sticky selection
button, the dock handoff order) must not silently change shape.
"""

import os
import pathlib

REPO = pathlib.Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _read(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Read-only: what the UI must NOT contain any more
# ---------------------------------------------------------------------------


def test_files_ui_has_no_editor_or_mutating_affordances():
    source = _read("web/modules/files.js")
    css = _read("web/style.css")

    # Editor + save (and therefore the ⌘S handler and the dirty-state guard).
    assert "files-editor" not in source
    assert "files-editor" not in css
    assert "renderEditor" not in source
    assert "saveCurrentFile" not in source
    assert "editorDirty" not in source
    assert "beforeunload" not in source

    # Create / upload / clipboard / delete / context menu.
    for gone in (
        "files-new-file",
        "files-new-dir",
        "files-paste",
        "files-context-menu",
        "files-drop-overlay",
        "createNewFile",
        "uploadFiles",
        "pasteClipboard",
        "deleteSelectedEntry",
        "state.clipboard",
        "dragenter",
        "dropEffect",
    ):
        assert gone not in source, gone
    for gone in (".files-context-menu", ".files-drop-overlay", ".files-drop-card"):
        assert gone not in css, gone

    # No write endpoint is reachable from the page (the backend keeps them).
    for endpoint in (
        "/api/files/write",
        "/api/files/mkdir",
        "/api/files/upload",
        "/api/files/delete",
        "/api/files/transfer",
    ):
        assert endpoint not in source, endpoint

    # Reads only.
    assert "/api/files/list" in source
    assert "/api/files/read" in source
    assert "/api/files/download" in source


def test_files_backend_write_endpoints_remain_untouched():
    """Decision 18 removes UI, not contract: the gateway still serves writes."""
    gateway = _read("ouroboros/gateway/files.py")
    for endpoint in ("/api/files/write", "/api/files/mkdir", "/api/files/upload", "/api/files/delete"):
        assert endpoint in gateway, endpoint


def test_files_page_no_longer_registers_a_navigation_guard():
    """No editor means no unsaved state, so the files guard is gone — while the
    app-level seam itself stays for the module that still needs it."""
    files_source = _read("web/modules/files.js")
    app_source = _read("web/app.js")
    settings_source = _read("web/modules/settings.js")

    assert "setBeforePageLeave(" not in files_source
    assert "canLeaveEditor" not in files_source
    # The seam is still provided by app.js and still consumed by Settings.
    assert "setBeforePageLeave: (handler) =>" in app_source
    assert "beforePageLeaveHandlers" in app_source
    assert "setBeforePageLeave" in settings_source


# ---------------------------------------------------------------------------
# Tree rail
# ---------------------------------------------------------------------------


def test_files_tree_rail_replaces_the_breadcrumb_strip():
    source = _read("web/modules/files.js")

    assert 'placeholder="Go to file…"' in source
    assert "files-tree-row" in source
    assert "files-tree-twist" in source
    # Lazy expansion: one list call per expanded directory, expansion is state.
    assert "async function loadDir(path)" in source
    assert "node.expanded = !node.expanded" in source
    assert "if (node.expanded && !node.loaded)" in source
    # Dir/file glyphs and the active-file highlight.
    assert "expanded ? '▾' : '▸'" in source
    assert "is-active" in source
    # Breadcrumbs are gone; the viewer header carries the current path.
    assert "files-breadcrumb" not in source
    assert "renderBreadcrumb" not in source
    assert "files-crumb" not in source
    assert "files-viewer-path" in source
    # Depth is a CSS custom property, never an inline style rule.
    assert "row.style.setProperty('--files-indent', String(depth));" in source
    assert 'style="' not in source


def test_files_filter_scope_is_disclosed_not_implied():
    """There is no server-side search: the filter matches loaded entries, and the
    empty state says so instead of implying a whole-tree search found nothing."""
    source = _read("web/modules/files.js")
    assert "No matches in the folders opened so far." in source
    assert "Listing truncated by the server." in source


# ---------------------------------------------------------------------------
# Viewer: per-line rows, honest truncation, highlighting
# ---------------------------------------------------------------------------


def test_files_viewer_renders_one_row_per_line_with_line_numbers():
    source = _read("web/modules/files.js")

    assert "row.dataset.lineNumber = String(index + 1);" in source
    assert "files-code-row" in source
    assert "files-code-num" in source
    assert "files-code-text" in source
    # Highlighting goes through the shared XSS-safe tokenizer.
    assert "import { highlightLine, languageForPath } from './code_highlight.js';" in source
    assert "text.innerHTML = highlightLine(line, language);" in source


def test_files_viewer_never_claims_a_line_total_it_cannot_know():
    source = _read("web/modules/files.js")

    assert "${state.activeTruncated ? ' shown · preview truncated' : ''}" in source
    # Pluralization matches the chip label: "1 line", "42 lines".
    assert "${count} line${count === 1 ? '' : 's'}" in source
    # An empty file renders ZERO rows, not a phantom line 1.
    assert "if (text === '') return [];" in source
    # No mtime / "modified by task" fiction in the meta line.
    assert "modified" not in source
    assert "mtime" not in source


def test_files_tree_says_why_a_listing_failed_instead_of_loading_forever():
    """A failed expand/refresh must replace "Loading…" with the recorded reason."""
    source = _read("web/modules/files.js")

    # loadDir records the failure on the node before re-raising, status included.
    assert "node.error = err instanceof Error ? err.message : String(err);" in source
    assert "failure.status = resp.status;" in source
    # The repaint after an expand is unconditional.
    assert "            } finally {\n                renderTree();\n            }" in source
    assert "treeNote(child.error || 'Loading…')" in source
    assert "treeNote(root?.error || 'No files listed.')" in source

    # Refresh: per-path isolation, always repaint + re-open, ONE summary toast,
    # and a folder the server no longer has is forgotten rather than kept as a
    # ghost row that fails again on every refresh.
    refresh = source.split("async function refreshAll()", 1)[1].split("\n    }\n", 1)[0]
    assert "failures.push(" in refresh
    assert "if (err?.status === 404 && path !== '.') state.dirs.delete(path);" in refresh
    assert refresh.index("renderTree();") < refresh.index("await openFile(path)")
    assert "failed to refresh" in refresh


def test_files_media_previews_stay_sandboxed_and_download_uses_the_bridge():
    source = _read("web/modules/files.js")
    download_helper = _read("web/modules/ui_helpers.js")
    launcher = _read("launcher.py")

    assert 'class="files-preview-frame" sandbox="allow-same-origin"' in source
    assert "escapeHtmlAttr(data.content_url)" in source
    assert "encodeURI(data.content_url)" not in source
    assert "files-preview-image" in source

    assert "downloadViaHostBridge(" in source
    assert "download_file_to_downloads" in download_helper
    assert "URL.createObjectURL" in download_helper
    assert 'parsed.path != "/api/files/download"' in launcher
    assert 'parsed.path.startswith("/api/extensions/")' in launcher
    assert "parsed.port != actual_port" in launcher


# ---------------------------------------------------------------------------
# Selection → range → chip, and the dock handoff
# ---------------------------------------------------------------------------


def test_selection_capture_maps_boundaries_to_lines_of_the_active_viewer():
    source = _read("web/modules/files.js")

    assert "export function selectionLineRange(boundaries = {})" in source
    assert "export function readViewerSelection(root)" in source
    assert "selection.getRangeAt(0)" in source
    assert "if (selection.rangeCount === 0 || selection.isCollapsed) return null;" in source or (
        "!selection || selection.rangeCount === 0 || selection.isCollapsed" in source
    )
    # Both boundaries must resolve to rows of THIS viewer.
    assert "const start = resolveBoundary(range.startContainer, range.startOffset, root);" in source
    assert "const end = resolveBoundary(range.endContainer, range.endOffset, root);" in source
    assert "if (start === null || end === null) return null;" in source
    assert "element.closest('[data-line-number]')" in source
    # Ordering, collapsed rejection, and the offset-zero exclusion.
    assert "[first, last] = [last, first];" in source
    assert "if (first.line === last.line && first.offset === last.offset) return null;" in source
    assert "if (lineEnd > first.line && last.offset === 0) lineEnd -= 1;" in source


def test_element_range_boundaries_are_not_read_as_character_offsets():
    """DOM spec: a boundary on an ELEMENT carries a CHILD INDEX, so a boundary that
    sits before the line's first character must read as offset 0 rather than as "one
    char in" — whatever index the engine reports.

    This is SPEC-HARDENING, not a reproduced bug: probing Chromium and WebKit, a drag
    STARTED over the line-number gutter never yielded a nonzero child index. The case
    actually observed there is a drag whose END lands over the gutter, arriving as
    element offset 0 on the row. The normalization makes the two readings agree so the
    mapping does not depend on an engine's choice of index."""
    source = _read("web/modules/files.js")

    assert "function boundaryBeforeText(container, offset, row)" in source
    assert "if (!(container instanceof Element)) return false;" in source
    # The question is answered positionally, against the row's own code text.
    assert "row.querySelector('.files-code-text')" in source
    assert "probe.setStart(text, 0);" in source
    assert "probe.setEnd(container, offset);" in source
    assert "return probe.toString() === '';" in source
    # The pure mapper honors the flag over the numeric offset.
    assert "const offsetOf = (value, beforeText) => {" in source
    assert "if (beforeText === true) return 0;" in source
    assert "offsetOf(boundaries.startOffset, boundaries.startBeforeText)" in source
    assert "offsetOf(boundaries.endOffset, boundaries.endBeforeText)" in source
    assert "startBeforeText: start.beforeText," in source
    assert "endBeforeText: end.beforeText," in source


def test_truncated_preview_capture_sends_the_range_without_inline_bytes():
    """The LAST shown line of a prefix can be a fragment the server cut mid-line,
    so a range touching it degrades to the ranged bare marker — and says so."""
    source = _read("web/modules/files.js")

    assert "export function captureInlinesContent(" in source
    assert "if (!truncated) return true;" in source
    assert "return end < shown;" in source
    # The capture site consults it and drops the bytes, keeping the true range.
    assert "shownLines: state.activeLines.length," in source
    assert "content: inlineContent ? state.activeLines.slice(range.lineStart - 1, range.lineEnd).join('\\n') : null," in source
    # Disclosed exactly once per opened file.
    assert "if (range && !inlineContent && !state.truncatedNoticeShown) {" in source
    assert "state.truncatedNoticeShown = true;" in source
    assert "Preview is truncated — sending the line range without inline bytes." in source
    assert "state.truncatedNoticeShown = false;" in source


def test_capture_builds_a_chip_through_the_shared_codec_and_discloses_refusals():
    source = _read("web/modules/files.js")

    assert "import { createComposerParts, makeChipPart } from './composer_parts.js';" in source
    assert "state.activeLines.slice(range.lineStart - 1, range.lineEnd).join('\\n')" in source
    assert "path: state.activeDisplayPath" in source
    # A whole-file capture is a bare-path chip.
    assert ": makeChipPart({ path: state.activeDisplayPath });" in source
    # makeChipPart returning null is DISCLOSED, never silently dropped.
    assert "if (!chip) {" in source
    assert "cannot be written as a context reference" in source
    assert "import { showToast } from './toast.js';" in source


def test_sticky_selection_button_survives_the_click_that_uses_it():
    source = _read("web/modules/files.js")
    css = _read("web/style.css")

    assert "files-selection-btn" in source
    assert "selectionBtn.addEventListener('mousedown', (event) => event.preventDefault());" in source
    assert "captureBtn.addEventListener('mousedown', (event) => event.preventDefault());" in source
    assert "viewerBodyEl.addEventListener('mouseup', () => syncSelectionButton());" in source
    assert "document.addEventListener('selectionchange', () => syncSelectionButton());" in source
    assert "state.selectionRange = readViewerSelection(viewerBodyEl);" in source
    assert "capture({ selectionOnly: true })" in source

    # The mouseup cache is readable by the BUTTON and by a ⌘L pressed with focus
    # inside the dock ("select code, type a comment, then ⌘L" — focusing the dock
    # is what collapsed the selection). A ⌘L anywhere else with nothing selected
    # falls back to a whole-file chip, never to a remembered range.
    assert "export function resolveCaptureRange(" in source
    assert "if (live) return live;" in source
    assert "if (selectionOnly || focusInDock) return cached || null;" in source
    assert "return null;" in source
    assert "cached: state.lastSelectionRange," in source
    assert "focusInDock: Boolean(activeElement instanceof Element && activeElement.closest('[data-capture-dock]'))," in source
    assert "if (state.selectionRange) state.lastSelectionRange = state.selectionRange;" in source
    # No stale-range fallback survives at the capture site.
    assert "|| state.selectionRange" not in source

    assert ".files-selection-btn {" in css
    assert "transform: translateX(-50%);" in css


def test_files_dock_sends_ordered_parts_before_navigating_to_chat():
    source = _read("web/modules/files.js")

    assert "createComposerParts({ container: dockPartsEl, input: dockInputEl })" in source
    assert "data-capture-dock" in source
    assert "dock.addChip(chip)" in source
    assert "dock.commitDraft();" in source
    assert "const parts = dock.getParts();" in source
    assert "await controller.sendParts(parts)" in source
    # Order is load-bearing: clear + navigate happen ONLY after a successful send.
    send_body = source.split("async function sendDock()", 1)[1]
    send_body = send_body.split("// ---", 1)[0]
    assert send_body.index("sendParts(parts)") < send_body.index("dock.clear();")
    assert send_body.index("dock.clear();") < send_body.index("showPage('chat')")
    assert "Your draft is kept." in source
    # Enter sends, Shift+Enter keeps writing.
    assert "if (event.key !== 'Enter' || event.shiftKey) return;" in source


def test_global_capture_hotkey_lives_in_the_phase_c_anchor():
    app_source = _read("web/app.js")
    files_source = _read("web/modules/files.js")

    anchor = app_source.split("/* [anchor:phase-C] global capture hotkey */", 1)[1]
    assert "event.metaKey || event.ctrlKey" in anchor
    assert "String(event.key).toLowerCase() !== 'l'" in anchor
    assert "CAPTURE_PAGES = new Set(['files', 'changes'])" in anchor
    # ONE editable test for the whole app (shared with the mobile-keyboard code),
    # not a second broader selector living inside the anchor.
    assert "isKeyboardEditable(document.activeElement)" in anchor
    assert "CAPTURE_EDITABLE_SELECTOR" not in app_source
    assert "function isKeyboardEditable(node) {" in app_source
    assert "const KEYBOARD_EDITABLE_SELECTOR = [" in app_source
    assert app_source.count("[contenteditable]:not([contenteditable=\"false\"])") == 1
    assert "editable.closest('[data-capture-dock]')" in anchor
    # The keystroke is swallowed ONLY when a page actually consumed the capture:
    # cancelable event, preventDefault gated on defaultPrevented.
    assert "cancelable: true," in anchor
    assert "if (request.defaultPrevented) event.preventDefault();" in anchor
    assert "new CustomEvent('ouro:capture-selection'" in anchor
    # The page that owns the surface does the capture and claims the keystroke.
    assert "window.addEventListener('ouro:capture-selection'" in files_source
    assert "if (event.detail?.page !== 'files') return;" in files_source
    handler = files_source.split("window.addEventListener('ouro:capture-selection'", 1)[1]
    handler = handler.split("});", 1)[0]
    # ...and it claims it ONLY when the capture actually happened. An unconditional
    # preventDefault makes the global handler's `defaultPrevented` gate above a
    # tautology, swallowing ⌘L even on the "open a file first" no-op path.
    assert "if (capture()) event.preventDefault();" in handler


def test_files_page_exposes_no_dead_surface():
    """Every export/return the page does not actually use is a maintenance lie."""
    files_source = _read("web/modules/files.js")
    app_source = _read("web/app.js")
    highlight_source = _read("web/modules/code_highlight.js")

    # initFiles wires itself up; it hands nothing back and writes no app state.
    assert "export function initFiles({ showPage, getChatController } = {}) {" in files_source
    assert "return { capture, state };" not in files_source
    assert "filesState" not in files_source
    assert "filesState" not in app_source
    # The highlighter exports only what the viewer and its tests consume.
    assert "isSupportedLanguage" not in highlight_source
    assert "isSupportedLanguage" not in files_source


# ---------------------------------------------------------------------------
# Highlighter safety (decision 19)
# ---------------------------------------------------------------------------


def test_code_highlighter_escapes_every_lexeme_and_names_no_hue():
    source = _read("web/modules/code_highlight.js")
    css = _read("web/style.css")

    assert "import { escapeHtmlAttr as escapeHtml } from './utils.js';" in source
    assert '`<span class="tok-${' in source
    assert "escapeHtml(token.text)" in source
    # ONE escape site, and it is the same expression that emits the wrapper:
    # there is no code path from source bytes to the DOM that skips escaping.
    assert source.count("escapeHtml(") == 1
    # The tokenizer names lexeme ROLES; every hue stays in the CSS tokens.
    for hue in ("#f07a86", "#6e96d2", "#e5b567", "#f59e0b", "rgba(255, 255, 255"):
        assert hue not in source, hue
    for token in ("--code-keyword", "--code-self", "--code-string", "--code-number", "--code-comment"):
        assert token in css, token
    for cls in (".tok-keyword {", ".tok-self {", ".tok-string {", ".tok-number {", ".tok-comment {", ".tok-default {"):
        assert cls in css, cls


# ---------------------------------------------------------------------------
# Layout: internal scroll contract
# ---------------------------------------------------------------------------


def test_files_layout_uses_internal_scroll_contract():
    css = _read("web/style.css")

    assert "[stream C: files (tree rail, viewer, code highlighting)]" in css
    assert ".files-layout {" in css
    assert 'grid-template-areas:\n        "rail viewer"\n        "dock dock";' in css
    assert "grid-template-columns: 280px minmax(0, 1fr);" in css
    assert "grid-template-rows: minmax(0, 1fr) auto;" in css
    # Rail, code body and dock each own their scrolling; the page never scrolls.
    assert ".files-rail {" in css
    assert ".files-tree {" in css
    assert ".files-viewer-body {" in css
    assert "overscroll-behavior: contain;" in css
    # The viewer is the positioning context for the sticky capture button.
    assert ".files-viewer {\n    grid-area: viewer;" in css
    # Per-line grid + non-selectable gutter (a selection must not contain digits).
    assert "grid-template-columns: 52px minmax(0, 1fr);" in css
    assert ".files-code-num {" in css
    assert "user-select: none;" in css
    # Mobile stacks rail / viewer / dock.
    assert 'grid-template-areas:\n            "rail"\n            "viewer"\n            "dock";' in css


def test_files_dock_reuses_the_shared_composer_part_contract():
    css = _read("web/style.css")
    assert ".files-dock {" in css
    assert ".files-dock-field:focus-within {" in css
    # Chips themselves are the shared contract, not a files-local restyle.
    assert ".composer-part-chip {" in css
    assert ".files-dock-chip" not in css
    # …and the dock does NOT try to re-align them: `.composer-parts` (declared
    # later, same specificity) always won, so the override was dead weight
    # pretending the dock's chip row differed from every other one.
    assert ".files-dock-parts {" not in css


# ---------------------------------------------------------------------------
# Neighbouring contracts this page depends on
# ---------------------------------------------------------------------------


def test_chat_document_bubble_opens_externally_and_downloads_separately():
    chat = _read("web/modules/chat.js")
    helper = _read("web/modules/ui_helpers.js")
    launcher = _read("launcher.py")
    css = _read("web/style.css")

    # Desktop bridge: open in the OS default app without navigating the WebView.
    assert "def open_file_with_default_app(self, url: str, filename: str) -> dict:" in launcher
    assert "open_path_external(target)" in launcher
    assert 'tempfile.mkdtemp(prefix="ouroboros-open-")' in launcher
    # Shared loopback guard reused by both bridge methods (DRY).
    assert "_resolve_bridge_file_url(url)" in launcher

    # JS open helper prefers the native open bridge, degrades to the long-shipped
    # download_file_to_downloads(open_external=true) bridge when a packaged
    # launcher predates open_file_with_default_app (version skew), and only falls
    # back to a new tab on true web.
    assert "export async function openViaHostBridge(url, filename = 'file')" in helper
    assert "api?.open_file_with_default_app" in helper
    assert "api?.download_file_to_downloads" in helper
    assert "await downloadBridge(url, filename, true)" in helper

    # Bubble body click = open externally; separate ↓ button = download.
    assert "import { downloadViaHostBridge, openViaHostBridge } from './ui_helpers.js';" in chat
    assert "await openViaHostBridge(downloadUrl, filename);" in chat
    assert "await downloadViaHostBridge(downloadUrl, filename);" in chat
    assert 'class="chat-file-download"' in chat
    assert ".chat-file-download {" in css


def test_toast_tone_normalization_is_shared():
    """Files reports capture refusals through the shared toast, so its tone
    normalization stays the one code path."""
    toast = _read("web/modules/toast.js")
    helper = _read("web/modules/ui_helpers.js")
    assert "normalizeTone(tone || 'info', 'info')" in toast
    assert "export function normalizeTone(tone = 'muted', fallback = 'muted')" in helper
