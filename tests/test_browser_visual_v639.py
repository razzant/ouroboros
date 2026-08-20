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
    seen = []

    class _Page:
        def evaluate(self, js):
            seen.append(js)
            if not js.lstrip().startswith("(()"):
                raise RuntimeError("SyntaxError: Illegal return statement")
            return "iife-ok"

    class _BrowserState:
        last_screenshot_b64 = ""

    class _Ctx:
        browser_state = _BrowserState()

    monkeypatch.setattr(browser, "_ensure_browser", lambda *a, **k: _Page())
    monkeypatch.setattr(browser, "is_restricted_subagent_profile", lambda ctx: False)
    monkeypatch.setattr(browser, "_blocks_context_mode_self_lowering_js", lambda v: False)
    monkeypatch.setattr(browser, "_blocks_scope_review_floor_self_lowering_js", lambda v: False)
    monkeypatch.setattr(browser, "_blocks_mutative_toggle_js", lambda v: False)
    monkeypatch.setattr(browser, "_blocks_post_task_evolution_js", lambda v: False)

    out = browser._browser_action(_Ctx(), "evaluate", value="return 1 + 1;")
    assert "iife-ok" in out
    assert len(seen) == 2 and seen[1].lstrip().startswith("(()")  # raw then IIFE-wrapped


def test_evaluate_runtime_error_not_misreported_as_syntax(monkeypatch):
    # raw SyntaxError -> IIFE retry -> the wrapped code throws a RUNTIME error: it must
    # surface as that runtime error, NOT be misreported as a syntax parse failure.
    class _Page:
        def evaluate(self, js):
            if not js.lstrip().startswith("(()"):
                raise RuntimeError("SyntaxError: Illegal return statement")
            raise RuntimeError("ReferenceError: missingFn is not defined")

    class _BrowserState:
        last_screenshot_b64 = ""

    class _Ctx:
        browser_state = _BrowserState()

    monkeypatch.setattr(browser, "_ensure_browser", lambda *a, **k: _Page())
    monkeypatch.setattr(browser, "is_restricted_subagent_profile", lambda ctx: False)
    for _g in ("_blocks_context_mode_self_lowering_js", "_blocks_scope_review_floor_self_lowering_js",
               "_blocks_mutative_toggle_js", "_blocks_post_task_evolution_js"):
        monkeypatch.setattr(browser, _g, lambda v: False)
    monkeypatch.setattr(browser, "_is_infrastructure_error", lambda ctx: False)
    with pytest.raises(Exception) as exc:
        browser._browser_action(_Ctx(), "evaluate", value="return missingFn();")
    assert "ReferenceError" in str(exc.value)
    assert "BROWSER_EVALUATE_SYNTAX_ERROR" not in str(exc.value)
