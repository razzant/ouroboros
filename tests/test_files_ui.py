"""Regression checks for Files tab navigation and context menu behavior."""

import os
import pathlib

REPO = pathlib.Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _read(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


def test_files_page_registers_navigation_guard():
    app_source = _read("web/app.js")
    files_source = _read("web/modules/files.js")

    assert "beforePageLeave" in app_source
    assert "setBeforePageLeave" in app_source
    assert "setBeforePageLeave(async ({ from })" in files_source
    assert "if (from !== 'files') return true;" in files_source


def test_new_file_discard_and_context_menu_clamp_regressions():
    source = _read("web/modules/files.js")

    assert "createNewFile({ force: true })" in source
    assert "window.innerWidth - rect.width" in source
    assert "window.innerHeight - rect.height" in source


def test_files_page_explains_manager_role_and_directory_affordance():
    source = _read("web/modules/files.js")

    assert "This is a file manager, not a chat attachment picker." in source
    assert "Open a folder or file from the left panel to browse, preview, or edit its contents." in source
    assert "button.type = 'button';" in source
    assert "(entry.type === 'file' ? formatFileSize(entry.size) : 'open')" in source


def test_files_layout_uses_internal_scroll_contract():
    css = _read("web/style.css")

    assert "flex: 1;" in css
    assert ".files-layout {" in css
    assert 'grid-template-areas: "sidebar preview";' in css
    assert ".files-sidebar {" in css
    assert "min-height: 0;" in css
    assert "overflow: hidden;" in css
    assert ".files-list {" in css
    assert "overscroll-behavior: contain;" in css
    assert "grid-template-rows: minmax(220px, 320px) minmax(0, 1fr);" in css
    assert 'max-height: none;' in css


def test_files_pdf_preview_and_download_bridge_are_safe():
    source = _read("web/modules/files.js")
    download_helper = _read("web/modules/ui_helpers.js")
    launcher = _read("launcher.py")
    assert 'class="files-preview-frame" sandbox="allow-same-origin"' in source
    assert "downloadViaHostBridge(" in source
    assert "download_file_to_downloads" in download_helper
    assert "URL.createObjectURL" in download_helper
    assert "encodeURI(data.content_url)" not in source
    assert 'parsed.path != "/api/files/download"' in launcher
    assert 'parsed.path.startswith(("/api/extensions/", "/api/tasks/"))' in launcher
    assert "parsed.port != actual_port" in launcher


def test_chat_document_card_uses_dialog_and_safe_download_fallbacks():
    chat = _read("web/modules/chat_media.js")
    helper = _read("web/modules/ui_helpers.js")
    launcher = _read("launcher.py")
    css = _read("web/style.css")

    # Desktop bridge: open in the OS default app without navigating the WebView.
    assert "def open_file_with_default_app(self, url: str, filename: str) -> dict:" in launcher
    assert "open_path_external(target)" in launcher
    assert 'tempfile.mkdtemp(prefix="ouroboros-open-")' in launcher
    # Shared loopback guard reused by both bridge methods (DRY).
    assert "_resolve_bridge_file_url(url)" in launcher

    assert "export async function openViaHostBridge(url, filename = 'file', { browserUrl = '' } = {})" in helper
    assert "api?.open_file_with_default_app" in helper
    assert "api?.download_file_to_downloads" in helper
    assert "await downloadBridge(url, filename, true)" in helper

    # A card opens the explicit action dialog. Download retains the native bridge
    # for durable files and the object-URL anchor fallback for live base64 bytes.
    assert "className = 'chat-file-dialog'" in chat
    assert 'data-file-action="open"' in chat
    assert "open.hidden = !file.source.durable;" in chat
    # Host-bridge calls prefer the launcher-compatible address for the same
    # bytes and fall back to the canonical one; the browser keeps the canonical.
    assert "dialogFile.source.bridge || dialogFile.source.durable," in chat
    assert 'data-file-action="download"' in chat
    assert "await downloadViaHostBridge(source.bridge || source.durable, filename, { browserUrl: source.durable });" in chat
    assert "URL.createObjectURL(blob)" in chat
    assert ".chat-file-dialog {" in css

    # Share was removed EVERYWHERE by owner decision (postfix sprint D7):
    # navigator.share/canShare are unavailable in the desktop WebView and the
    # affordance was judged redundant next to Open/Download/Copy.
    assert 'data-file-action="share"' not in chat
    assert 'data-photo-action="share"' not in chat
    assert "navigator.share" not in chat
    assert "navigator.canShare" not in chat


def test_desktop_bridge_exposes_external_open_and_byte_save():
    launcher = _read("launcher.py")
    helper = _read("web/modules/ui_helpers.js")
    app = _read("web/app.js")

    # New MainApi methods keep the established {ok, error} bridge shape.
    assert "def open_external_url(self, url: str) -> dict:" in launcher
    assert 'raw.lower().startswith(("http://", "https://", "mailto:"))' in launcher
    # Bounded join on the detached opener: a settled failure (False/exception
    # recorded in the outcome list) is reported honestly; a still-running open
    # keeps the detached semantics.
    assert "_open_browser_detached(raw, outcome).join(timeout=3.0)" in launcher
    assert "if outcome and outcome[0] is not True:" in launcher
    assert "def save_bytes_to_downloads(self, filename: str, b64: str) -> dict:" in launcher
    assert 'base64.b64decode(str(b64 or ""), validate=True)' in launcher
    # Byte saves reuse the shared ~/Downloads collision helper.
    assert launcher.count('_unique_bridge_target(pathlib.Path.home() / "Downloads", filename)') == 2

    # The loopback guard now admits durable chat-media artifact paths.
    assert 'parsed.path.startswith(("/api/extensions/", "/api/tasks/"))' in launcher

    # The shell-only interceptor is wired from the app bootstrap and classifies
    # every escape intent (file / external / bytes).
    assert "installDesktopShellLinkInterceptor();" in app
    assert "export function installDesktopShellLinkInterceptor(" in helper
    assert "export function classifyShellUrl(" in helper
    assert "'pywebviewready'" in helper


def test_desktop_bridge_version_skew_fallback_chain():
    """An OLD packaged launcher (frontend updates via managed repo, launcher
    only on reinstall) must degrade honestly, never into a silently dead
    control: external links copy to clipboard with a toast, byte saves toast
    that saving is unavailable, and file opens keep the long-shipped
    open_file_with_default_app -> download_file_to_downloads -> window.open
    chain."""
    helper = _read("web/modules/ui_helpers.js")

    assert "api?.open_external_url" in helper
    assert "'Link copied — open it in your browser.'" in helper
    assert "api?.save_bytes_to_downloads" in helper
    assert '"Saving isn\'t available in the app — open in a browser."' in helper
    # Existing file-method chain stays intact for the interceptor to reuse.
    assert "api?.open_file_with_default_app" in helper
    assert "api?.download_file_to_downloads" in helper
    assert "window.open(browserUrl || url, '_blank', 'noopener');" in helper
    # Without ANY file bridge the interceptor degrades the file class to the
    # copy-link fallback instead of looping the helpers back into its own shim.
    assert "fileBridgeReady" in helper
    # Framed-wizard parity: the helpers resolve the bridge through the shared
    # resolver (the bridge lives on the PARENT window inside the overlay iframe).
    assert helper.count("shellBridgeApi(window)") == 2


def test_open_browser_detached_records_outcome(monkeypatch):
    """The detached opener reports its settled result through the outcome list:
    ``webbrowser.open`` returning False and raising are both honest failures
    the desktop bridge surfaces instead of a silent {ok: True}."""
    import launcher

    monkeypatch.setattr(launcher.webbrowser, "open", lambda url: True)
    outcome: list = []
    launcher._open_browser_detached("https://example.com", outcome).join(timeout=5)
    assert outcome == [True]

    monkeypatch.setattr(launcher.webbrowser, "open", lambda url: False)
    outcome = []
    launcher._open_browser_detached("https://example.com", outcome).join(timeout=5)
    assert outcome == [False]

    boom = RuntimeError("no browser association")
    def _raise(url):
        raise boom
    monkeypatch.setattr(launcher.webbrowser, "open", _raise)
    outcome = []
    launcher._open_browser_detached("https://example.com", outcome).join(timeout=5)
    assert outcome == [boom]

    # Callers without an outcome list keep the fire-and-forget contract.
    launcher._open_browser_detached("https://example.com").join(timeout=5)


def test_files_confirm_dialog_results_are_normalized():
    source = _read("web/modules/files.js")
    helper = _read("web/modules/ui_helpers.js")
    toast = _read("web/modules/toast.js")

    assert "typeof result === 'boolean' ? { confirmed: result, value: '' } : result" in source
    assert "return Boolean(result?.confirmed);" in source
    assert "if (!result?.confirmed) return;" in source
    assert "normalizeTone(tone || 'info', 'info')" in toast
    assert "export function normalizeTone(tone = 'muted', fallback = 'muted')" in helper
