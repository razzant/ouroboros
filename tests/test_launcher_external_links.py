"""Both emitted desktop APIs share one bounded browser handoff, never a save."""

import ast
import inspect

import pytest

from tests.test_onboarding_host import _install_fake_webview


@pytest.mark.parametrize("settled", [True, False, RuntimeError("no browser"), None])
def test_setup_and_main_bridge_share_the_bounded_external_opener(monkeypatch, settled):
    import launcher
    from ouroboros import launcher_onboarding

    opened, joins = [], []

    class OpenThread:
        def join(self, timeout):
            joins.append(timeout)

    def open_browser(url, outcome):
        opened.append(url)
        if settled is not None:
            outcome.append(settled)
        return OpenThread()

    monkeypatch.setattr(launcher, "_open_browser_detached", open_browser)
    created, _ = _install_fake_webview(monkeypatch)
    result = launcher_onboarding.present_first_run_onboarding(
        {}, 8765, open_external_url=launcher._open_external_url,
    )
    # Evaluate the actual main-window API class without launching a server/UI.
    source = inspect.getsource(launcher.main)
    node = next(node for node in ast.walk(ast.parse(source))
                if isinstance(node, ast.ClassDef) and node.name == "MainApi")
    namespace = {"_open_external_url": launcher._open_external_url}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "MainApi", "exec"), namespace)
    assert "open_external_url=_open_external_url" in source
    for api in (created["js_api"], namespace["MainApi"]()):
        for url in ("https://example.test/signin", "http://example.test/", "mailto:owner@example.test"):
            answer = api.open_external_url(url)
            assert answer["ok"] is (settled is True or settled is None)
            if not answer["ok"]:
                assert "default browser could not be opened" in answer["error"]
        before = len(opened)
        for url in ("javascript:alert(1)", "file:///tmp/signin", "/relative", ""):
            assert api.open_external_url(url)["ok"] is False
        assert len(opened) == before
    assert len(opened) == 6 and joins == [3.0] * 6
    assert result == {"saved": False, "restart_required": False}
    assert not hasattr(created["js_api"], "save_wizard")


def test_external_opener_returns_thread_start_failure(monkeypatch):
    import launcher

    def fail(*_):
        raise RuntimeError("thread unavailable")

    monkeypatch.setattr(launcher, "_open_browser_detached", fail)
    assert launcher._open_external_url("https://example.test/") == {
        "ok": False, "error": "thread unavailable",
    }


def test_main_bridge_request_attention_delegates_to_the_background_policy(monkeypatch):
    """The bridge only forwards: the window-or-banner choice belongs to launcher_background
    (a window hidden on purpose is never raised). `notify_owner` is the same method under the
    name newer pages feature-detect to send the alert text; older launchers lack it."""
    import launcher

    source = inspect.getsource(launcher.main)
    node = next(node for node in ast.walk(ast.parse(source))
                if isinstance(node, ast.ClassDef) and node.name == "MainApi")
    seen = []
    namespace = {
        "_open_external_url": launcher._open_external_url,
        "background": type("Background", (), {
            "attention": lambda self, *args: seen.append(args) or {"ok": True}})(),
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), "MainApi", "exec"), namespace)
    api = namespace["MainApi"]()
    assert api.request_attention(False) == {"ok": True}  # an older page: sound only
    assert api.notify_owner(1, "Task finished", None) == {"ok": True}
    assert api.notify_owner(True, "Task finished", "", False) == {"ok": True}  # a page with its own banner asks first
    assert seen == [(False, "", "", True), (True, "Task finished", "", True), (True, "Task finished", "", False)]
