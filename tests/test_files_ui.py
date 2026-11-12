"""Regression checks for Files tab navigation and context menu behavior."""

import os
import pathlib

REPO = pathlib.Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _read(rel: str) -> str:
    return (REPO / rel).read_text(encoding="utf-8")


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
