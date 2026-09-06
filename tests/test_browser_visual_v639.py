"""Phase 4 (v6.39) J: visual verification — SwiftShader launch, bounded paint wait,
evaluate IIFE retry."""

from __future__ import annotations

import pytest

from ouroboros.tools import browser


def test_launch_args_include_swiftshader():
    captured = {}

    class _Chromium:
        def launch(self, **kw):
            captured.update(kw)
            return "browser"

    class _PW:
        chromium = _Chromium()

    browser._launch_browser_with_fallback(_PW(), engine="chromium")
    args = captured.get("args", [])
    for flag in ("--use-gl=angle", "--use-angle=swiftshader",
                 "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"):
        assert flag in args


def test_wait_for_page_paint_is_bounded():
    calls = []

    class _Page:
        def wait_for_function(self, expr, timeout=None):
            calls.append(("wff", expr, timeout))

        def evaluate(self, js):
            calls.append(("eval", js))

    browser._wait_for_page_paint(_Page(), 9999)
    wffs = [c for c in calls if c[0] == "wff"]
    assert any(c[2] <= 3000 for c in wffs)  # readyState wait is bounded
    # the paint-flag wait has a HARD Playwright timeout (page timers are never trusted to
    # unblock us, so a page that suppresses rAF cannot hang the capture)
    assert any(c[2] == 500 for c in wffs)
    raf = next(c for c in calls if c[0] == "eval")
    assert "requestAnimationFrame" in raf[1] and "__obo_painted" in raf[1]


def test_wait_for_page_paint_never_raises():
    class _BadPage:
        def wait_for_function(self, *a, **k):
            raise RuntimeError("nav in progress")

        def evaluate(self, *a, **k):
            raise RuntimeError("rAF suppressed")

    # Best-effort contract: a hostile/unready page must not break capture.
    browser._wait_for_page_paint(_BadPage(), 3000)


def test_evaluate_retries_statement_snippet_in_iife(monkeypatch):
    # A statement-style snippet (top-level `return`) is a SyntaxError as a raw evaluate
    # expression; the action must retry it wrapped in an IIFE before surfacing a parse error.
    # Both attempts arrive inside the _evaluate_bounded Promise.race deadline wrapper
    # (the "})()" suffix is the IIFE retry marker; the raw attempt lacks it).
    seen = []

    class _Page:
        def set_default_timeout(self, ms):
            pass

        def evaluate(self, js):
            seen.append(js)
            if "})()" not in js:
                raise RuntimeError("SyntaxError: Illegal return statement")
            return "iife-ok"

    class _BrowserState:
        last_screenshot_b64 = ""

    class _Ctx:
        browser_state = _BrowserState()

    monkeypatch.setattr(browser, "_ensure_browser", lambda ctx, *a, **k: (_Page(), ctx.browser_state))
    monkeypatch.setattr(browser, "_readonly_subagent", lambda ctx: False)
    monkeypatch.setattr(browser, "_blocks_context_mode_self_lowering_js", lambda v: False)
    monkeypatch.setattr(browser, "_blocks_mutative_toggle_js", lambda v: False)
    monkeypatch.setattr(browser, "_blocks_post_task_evolution_js", lambda v: False)

    out = browser._browser_action(_Ctx(), "evaluate", value="return 1 + 1;")
    assert "iife-ok" in out
    assert len(seen) == 2  # raw then IIFE-wrapped
    assert "})()" not in seen[0] and "})()" in seen[1]
    assert all("Promise.race" in js for js in seen)  # both attempts stay bounded


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
    for _g in ("_blocks_context_mode_self_lowering_js",
               "_blocks_mutative_toggle_js", "_blocks_post_task_evolution_js"):
        monkeypatch.setattr(browser, _g, lambda v: False)
    monkeypatch.setattr(browser, "_is_infrastructure_error", lambda ctx: False)
    with pytest.raises(Exception) as exc:
        browser._browser_action(_Ctx(), "evaluate", value="return missingFn();")
    assert "ReferenceError" in str(exc.value)
    assert "BROWSER_EVALUATE_SYNTAX_ERROR" not in str(exc.value)


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


def test_evaluate_bounded_wraps_in_promise_race_with_deadline():
    page = _RecordingPage()
    browser._evaluate_bounded(page, "1 + 1", 7000)
    assert len(page.evaluated) == 1
    js = page.evaluated[0]
    # The in-page race carries the expression and a setTimeout rejection at the deadline.
    assert "Promise.race" in js and "setTimeout" in js
    assert "1 + 1" in js and "7000ms" in js
    # The expression rides through eval(<json-string>) — statement-list and
    # completion-value semantics identical to an un-bounded evaluate (the
    # parenthesised const-wrapper silently returned undefined for those).
    assert '"1 + 1"' in js
    # AND a function-valued result is invoked, exactly like the driver's
    # UtilityScript does for a raw-string evaluate with isFunction unset —
    # page.evaluate("() => {...}") / _MARKDOWN_JS depend on this branch
    # (dropping it serialized the uninvoked function to undefined).
    assert "typeof __obo_result === 'function' ? __obo_result() : __obo_result" in js
    # INDIRECT eval (global scope, like the driver's global.eval): var/function
    # declarations persist across evaluate calls and the wrapper's lexical
    # binding stays invisible to user code (direct eval was a TDZ trap).
    assert "(0, eval)(" in js


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


def test_health_snapshot_and_markdown_extraction_use_bounded_evaluate():
    page = _RecordingPage()
    browser._page_health_snapshot(page)
    health_js = [js for js in page.evaluated if isinstance(js, str)]
    assert any("querySelectorAll('canvas')" in js and "Promise.race" in js for js in health_js)

    page2 = _RecordingPage(result="# markdown")
    out = browser._extract_page_output(page2, "markdown", _action_ctx())
    assert out.startswith("# markdown")
    assert any("Promise.race" in js for js in page2.evaluated if isinstance(js, str))


def test_browse_page_sets_session_default_timeout_from_caller(monkeypatch):
    page = _RecordingPage(result="text out")
    monkeypatch.setattr(browser, "_ensure_browser", lambda ctx, *a, **k: (page, ctx.browser_state))
    monkeypatch.setattr(browser, "_readonly_subagent", lambda ctx: False)

    out = browser._browse_page(_action_ctx(), "http://example.test/", output="text", timeout=45000)
    assert "body text" in out
    assert page.default_ms == 45000
    assert ("goto", "http://example.test/", 45000) in page.evaluated
