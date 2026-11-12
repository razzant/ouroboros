"""Phase 4 (v6.39) J: visual verification — SwiftShader launch, bounded paint wait,
evaluate IIFE retry."""

from __future__ import annotations


import pytest

from ouroboros.tools import browser


def test_evaluate_runtime_error_not_misreported_as_syntax(monkeypatch):
    # raw SyntaxError -> IIFE retry -> the wrapped code throws a RUNTIME error: it must
    # surface as that runtime error, NOT be misreported as a syntax parse failure.
    class _Page:
        def set_default_timeout(self, ms):
            pass

        def evaluate(self, js):
            if "})()" not in js:
                raise RuntimeError("SyntaxError: Illegal return statement")
            raise RuntimeError("ReferenceError: missingFn is not defined")

    class _BrowserState:
        last_screenshot_b64 = ""

    class _Ctx:
        browser_state = _BrowserState()

    monkeypatch.setattr(browser, "_ensure_browser", lambda ctx, *a, **k: (_Page(), ctx.browser_state))
    monkeypatch.setattr(browser, "_readonly_subagent", lambda ctx: False)
    monkeypatch.setattr(browser, "_is_infrastructure_error", lambda ctx: False)
    with pytest.raises(Exception) as exc:
        browser._browser_action(_Ctx(), "evaluate", value="return missingFn();")
    assert "ReferenceError" in str(exc.value)
    assert "BROWSER_EVALUATE_SYNTAX_ERROR" not in str(exc.value)


def test_browser_disconnect_after_click_never_replays_action(monkeypatch):
    seen = []
    page = _RecordingPage()
    def click(*args, **kwargs):
        seen.append("click")
        raise RuntimeError("connection disconnected after dispatch")
    page.click = click
    monkeypatch.setattr(browser, "_ensure_browser", lambda ctx, *a, **k: (page, ctx.browser_state))
    monkeypatch.setattr(browser, "_readonly_subagent", lambda ctx: False)
    monkeypatch.setattr(browser, "_is_infrastructure_error", lambda ctx: True)
    result = browser._browser_action(_action_ctx(), "click", selector="#submit")
    assert seen == ["click"]
    assert "BROWSER_ACTION_OUTCOME_UNKNOWN" in result


class _RecordingPage:
    """Fake page: records evaluate payloads and the session default timeout."""

    def __init__(self, result="ok"):
        self.evaluated = []
        self.default_ms = None
        self.url = "http://example.test/"
        self._result = result

    def set_default_timeout(self, ms):
        self.default_ms = ms

    def evaluate(self, js):
        self.evaluated.append(js)
        return self._result

    def title(self):
        return "t"

    def inner_text(self, _selector):
        return "body text"

    def goto(self, url, timeout=None, wait_until=None):
        self.evaluated.append(("goto", url, timeout))


def _action_ctx():
    class _BrowserState:
        last_screenshot_b64 = ""
        _browser_engine = "chromium"
        _browser_device = ""

    class _Ctx:
        browser_state = _BrowserState()

    return _Ctx()


def test_action_evaluate_and_scroll_route_through_bounded_wrapper(monkeypatch):
    page = _RecordingPage()
    monkeypatch.setattr(browser, "_ensure_browser", lambda ctx, *a, **k: (page, ctx.browser_state))
    monkeypatch.setattr(browser, "_readonly_subagent", lambda ctx: False)

    out = browser._browser_action(_action_ctx(), "evaluate", value="document.title")
    assert "ok" in out
    assert "Promise.race" in page.evaluated[-1] and "document.title" in page.evaluated[-1]
    # The 5s action default cannot strangle default-honoring capture calls.
    assert page.default_ms == 30000

    browser._browser_action(_action_ctx(), "scroll", value="bottom")
    assert "Promise.race" in page.evaluated[-1]
    assert "document.body.scrollHeight" in page.evaluated[-1]

    # An explicitly larger caller timeout widens the session default and the
    # in-page evaluate deadline alike.
    browser._browser_action(_action_ctx(), "evaluate", value="1 + 1", timeout=120000)
    assert page.default_ms == 120000
    assert "120000ms" in page.evaluated[-1]
