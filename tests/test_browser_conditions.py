"""#1606: the built-in browser no longer empties srcdoc frames, and says how it observed.

Only playwright-stealth's ``iframe.contentWindow`` evasion is disabled; every other
evasion stays. Each browser generation records its own engine, actual version and
stealth facts after a complete setup, and every successful observation (DOM text,
HTML, markdown, screenshot, evaluate) carries them as a host note through
``producer_text``/``host_annotations``: the model sees the note, programs read the
clean data. Refusals, action receipts and errors are never relabelled.

The real-engine qualification against the production module-widget mount lives in the
CI browser lane, ``tests/test_browser_tools_smoke.py``.
"""
from __future__ import annotations

from tests._tool_result_delivery_shared import measured_fit

import base64
import json
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor

import pytest

import ouroboros.tools.browser as browser_mod
from ouroboros import owner_pause
from ouroboros.task_results import write_task_result
from ouroboros.tools.registry import ToolRegistry

PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")


class FakeStealth:
    applied = []

    def __init__(self, **options):
        self.options = options

    def apply_stealth_sync(self, page):
        FakeStealth.applied.append((self.options, page))


class FakePage:
    def __init__(self, ctx, text="Fake page body", on_read=None):
        self.ctx, self.text, self.on_read, self.url = ctx, text, on_read, "https://observed.example/"

    def set_default_timeout(self, _ms):
        pass

    def on(self, *_a):
        pass

    def goto(self, url, **_kw):
        self.url = url

    def title(self):
        return "Observed"

    def inner_text(self, _selector):
        if self.on_read:
            self.on_read()
        return self.text

    def content(self):
        return f"<html><body>{self.text}</body></html>"

    def evaluate(self, expression, *_a):
        if "__obo_result" in expression and "MARKDOWN_PROBE" in expression:
            return "# Fake markdown"
        if "__obo_result" in expression and "querySelectorAll('canvas')" in expression:
            return {"canvas": 0, "img": 0}
        if "__obo_result" in expression:
            return 42
        return None

    def wait_for_function(self, *_a, **_kw):
        pass

    def wait_for_timeout(self, _ms):
        pass

    def screenshot(self, **_kw):
        return PNG

    def click(self, *_a, **_kw):
        pass


def _fake_playwright(ctx, launches, page_factory):
    def engine(name, version):
        def launch(**_kw):
            launches.append(name)
            context = types.SimpleNamespace(new_page=lambda: page_factory(ctx), route=lambda *_a: None,
                                            close=lambda: None)
            return types.SimpleNamespace(new_context=lambda **_kw: context, is_connected=lambda: True,
                                         close=lambda: None, version=version)
        return types.SimpleNamespace(launch=launch)
    return types.SimpleNamespace(chromium=engine("chromium", "141.0.7390.37"), webkit=engine("webkit", "26.0"),
                                 devices={}, stop=lambda: None)


@pytest.fixture
def fake_browser(monkeypatch):
    def install(ctx, *, stealth=True, page_factory=FakePage):
        launches = []
        playwright = _fake_playwright(ctx, launches, page_factory)
        sync_api = types.SimpleNamespace(sync_playwright=lambda: types.SimpleNamespace(start=lambda: playwright))
        monkeypatch.setitem(__import__("sys").modules, "playwright.sync_api", sync_api)
        monkeypatch.setattr(browser_mod, "_ensure_playwright_installed", lambda *a, **k: None)
        monkeypatch.setattr(browser_mod, "_HAS_STEALTH", stealth)
        monkeypatch.setattr(browser_mod, "Stealth", FakeStealth, raising=False)
        monkeypatch.setattr(browser_mod.browser_policy, "browser_url_block_reason", lambda *a, **k: "")
        FakeStealth.applied = []
        return launches
    return install


def _ctx():
    from ouroboros.tools.registry import BrowserState

    return types.SimpleNamespace(browser_state=BrowserState(), task_constraint=None)


def test_setup_disables_only_the_iframe_evasion_and_records_generation_facts(fake_browser):
    ctx = _ctx()
    launches = fake_browser(ctx)

    page, generation = browser_mod._ensure_browser(ctx)

    ((options, applied_page),) = FakeStealth.applied
    assert options == {"iframe_content_window": False} and applied_page is page
    assert generation.browser_conditions == ("Browser conditions: chromium 141.0.7390.37 (Playwright); "
                                             "playwright-stealth evasions applied except iframe.contentWindow.")
    again, same = browser_mod._ensure_browser(ctx)  # reuse: the generation's own facts, no second setup
    assert (again, same) == (page, generation) and len(FakeStealth.applied) == 1 and launches == ["chromium"]
    _page, recreated = browser_mod._ensure_browser(ctx, engine="webkit")
    assert recreated is not generation and launches == ["chromium", "webkit"]
    assert recreated.browser_conditions.startswith("Browser conditions: webkit 26.0 (Playwright); ")
    assert generation.browser_conditions.startswith("Browser conditions: chromium 141.0.7390.37")


def test_missing_stealth_is_reported_absent(fake_browser):
    ctx = _ctx()
    fake_browser(ctx, stealth=False)

    _page, generation = browser_mod._ensure_browser(ctx)

    assert FakeStealth.applied == []
    assert generation.browser_conditions == ("Browser conditions: chromium 141.0.7390.37 (Playwright); "
                                             "playwright-stealth is not installed, so no stealth evasions were applied.")


def test_an_incomplete_setup_records_no_facts(fake_browser):
    ctx = _ctx()

    def broken(_ctx):
        raise RuntimeError("new_page failed")

    fake_browser(ctx, page_factory=broken)
    with pytest.raises(RuntimeError, match="new_page failed"):
        browser_mod._ensure_browser(ctx)
    assert not getattr(ctx.browser_state, "browser_conditions", "")


def test_a_retired_call_reports_its_own_generation_never_the_replacement(monkeypatch):
    from ouroboros.tools.registry import BrowserState

    own, replacement = BrowserState(), BrowserState()
    own.browser_conditions, replacement.browser_conditions = "Browser conditions: OWN", "Browser conditions: NEW"
    ctx = types.SimpleNamespace(browser_state=own, task_constraint=None)
    page = FakePage(ctx)

    def ensure_and_retire(c, engine="chromium", device=""):
        c.browser_state = replacement  # the mid-call timeout detach
        return page, own

    monkeypatch.setattr(browser_mod, "_ensure_browser", ensure_and_retire)
    monkeypatch.setattr(browser_mod, "_readonly_subagent", lambda c: False)
    monkeypatch.setattr(browser_mod.browser_policy, "browser_url_block_reason", lambda *a, **k: "")

    out = browser_mod._browse_page(ctx, "https://observed.example/")

    assert out == "Fake page body\n\nBrowser conditions: OWN"


@pytest.fixture
def sticky(tmp_path, monkeypatch, fake_browser):
    """The production path: the registry invocation handed to one sticky executor."""
    import ouroboros.safety as safety

    monkeypatch.setattr(safety, "check_safety", lambda *_a, **_k: (True, ""))
    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(tmp_path))
    write_task_result(tmp_path, "root", "running", root_task_id="root")
    registry = ToolRegistry(repo_dir=tmp_path, drive_root=tmp_path)
    registry._ctx.task_id = registry._ctx.root_task_id = "root"
    registry._ctx.messages = []
    executor = ThreadPoolExecutor(max_workers=1)

    def call(name, args):
        return owner_pause.submit_tool(registry._ctx, name, executor.submit, registry.execute_result,
                                       name, dict(args)).result()

    world = types.SimpleNamespace(registry=registry, call=call, install=lambda **kw: fake_browser(registry._ctx, **kw))
    try:
        yield world
    finally:
        executor.submit(browser_mod.cleanup_browser, registry._ctx).result()
        executor.shutdown()


CONDITIONS = ("Browser conditions: chromium 141.0.7390.37 (Playwright); "
              "playwright-stealth evasions applied except iframe.contentWindow.")


@pytest.mark.parametrize("tool,args,producer", [
    ("browse_page", {"url": "https://observed.example/", "output": "text"}, "Fake page body"),
    ("browse_page", {"url": "https://observed.example/", "output": "html"}, "<html><body>Fake page body</body></html>"),
    ("browse_page", {"url": "https://observed.example/", "output": "screenshot"}, None),
    ("browser_action", {"action": "evaluate", "value": "document.title"}, "42"),
    ("browser_action", {"action": "screenshot"}, None),
], ids=["text", "html", "screenshot", "evaluate", "action-screenshot"])
def test_successful_observations_carry_the_note_and_keep_clean_data(sticky, tool, args, producer):
    sticky.install()
    if tool == "browser_action":
        sticky.call("browse_page", {"url": "https://observed.example/"})

    result = sticky.call(tool, args)

    assert (result.status, result.code) == ("ok", "OK"), result.text
    assert result.host_annotations == (CONDITIONS,)
    assert result.text == result.producer_text + "\n\n" + CONDITIONS
    if producer is not None:
        assert result.producer_text == producer
    else:
        assert result.producer_text.startswith("Screenshot captured (") and CONDITIONS not in result.producer_text


def test_markdown_output_is_annotated_like_the_other_reads(sticky, monkeypatch):
    sticky.install()
    monkeypatch.setattr(browser_mod, "_MARKDOWN_JS", "() => 'MARKDOWN_PROBE'")

    result = sticky.call("browse_page", {"url": "https://observed.example/", "output": "markdown"})

    assert result.producer_text == "# Fake markdown" and result.host_annotations == (CONDITIONS,)


def test_a_blocked_subrequest_and_the_conditions_are_both_host_notes(sticky):
    ctx = sticky.registry._ctx
    reason = "BROWSER_REQUEST_BLOCKED: a subresource target was refused"
    sticky.install(page_factory=lambda c: FakePage(c, on_read=lambda: setattr(
        ctx.browser_state, "request_block_reason", reason)))

    result = sticky.call("browse_page", {"url": "https://observed.example/"})

    assert result.producer_text == "Fake page body" and result.host_annotations == (reason, CONDITIONS)
    assert result.text == "Fake page body\n\n" + reason + "\n\n" + CONDITIONS


def test_refusals_and_action_receipts_are_never_relabelled(sticky, monkeypatch):
    sticky.install()
    clicked = sticky.call("browser_action", {"action": "click", "selector": "#go"})
    assert clicked.text == "Clicked: #go" and clicked.producer_text is None
    missing = sticky.call("browser_action", {"action": "evaluate", "value": ""})
    assert missing.producer_text is None and CONDITIONS not in missing.text
    monkeypatch.setattr(browser_mod, "_navigation_block_reason", lambda *a, **k: "BROWSER_URL_BLOCKED: refused")
    blocked = sticky.call("browse_page", {"url": "https://observed.example/"})
    assert blocked.text == "⚠️ BROWSER_URL_BLOCKED: refused" and blocked.status != "ok"
    assert blocked.producer_text is None


class DyingPage(FakePage):
    """A page whose navigation dies under it (a crashed renderer, a closed target)."""

    def __init__(self, ctx, release=None):
        super().__init__(ctx)
        self.closed, self.release = False, release
        self.generation = ctx.browser_state

    def goto(self, url, **_kw):
        if self.release is not None:
            assert self.release.wait(10)
        self.generation.browser_conditions = "Browser conditions: FAILED FIRST ATTEMPT"
        self.closed = True
        raise RuntimeError("Target page, context or browser has been closed")

    def is_closed(self):
        return self.closed


def _browse_through_the_loop(registry, tmp_path, timeout_sec, *, executor=None, engine="chromium"):
    """The production stateful branch: submit wrapper, generation pin, sticky worker."""
    from ouroboros import loop_tool_execution as execution

    (tmp_path / "logs").mkdir(exist_ok=True)
    executor = executor or execution.StatefulToolExecutor()
    tc = {"id": "call-retry", "function": {"name": "browse_page",
                                           "arguments": json.dumps({"url": "https://observed.example/",
                                                                    "engine": engine})}}
    return execution._execute_with_timeout(registry, tc, tmp_path / "logs", timeout_sec, "root",
                                           stateful_executor=executor), executor


def test_an_infrastructure_retry_delivers_its_own_generations_observation(sticky, tmp_path):
    ctx = sticky.registry._ctx
    pages = iter([DyingPage, FakePage])
    launches = sticky.install(page_factory=lambda c: next(pages)(c))
    first = ctx.browser_state

    row, executor = _browse_through_the_loop(sticky.registry, tmp_path, 30)
    try:
        result = row["tool_result"]
        assert (result.status, result.producer_text) == ("ok", "Fake page body"), result.text
        assert result.host_annotations == (CONDITIONS,) and "FAILED FIRST ATTEMPT" not in result.text
        assert launches == ["chromium", "chromium"] and first._cleanup_done
        assert ctx.browser_state is not first
        assert executor.submit(browser_mod._browser_call.get).result() is None
    finally:
        executor.submit(browser_mod.cleanup_browser, ctx).result()
        executor.shutdown()


def test_a_timeout_retirement_during_the_retry_checks_wins_the_swap(fake_browser, monkeypatch):
    ctx = _ctx()
    launches = fake_browser(ctx, page_factory=DyingPage)
    own = ctx.browser_state
    retirements = []

    def retired_while_checking(generation):
        assert generation is own
        retirements.append(browser_mod._detach_browser(ctx))  # the call's own timeout, from the main thread
        return True

    monkeypatch.setattr(browser_mod, "_is_infrastructure_error", retired_while_checking)
    monkeypatch.setattr(browser_mod, "_readonly_subagent", lambda c: False)

    out = browser_mod._browse_page(ctx, "https://observed.example/")

    ((retired, replacement),) = retirements
    assert out == browser_mod._SESSION_RETIRED_VOID and retired is own and own._cleanup_done
    assert ctx.browser_state is replacement and replacement.browser is None and launches == ["chromium"]


def test_engine_switch_then_infrastructure_retry_keeps_its_own_expectation(sticky, tmp_path):
    ctx = sticky.registry._ctx
    pages = iter([FakePage, DyingPage, FakePage])
    generations = []

    def page_factory(c):
        generations.append(c.browser_state)
        return next(pages)(c)

    launches = sticky.install(page_factory=page_factory)
    _, executor = _browse_through_the_loop(sticky.registry, tmp_path, 30)
    try:
        row, _ = _browse_through_the_loop(sticky.registry, tmp_path, 30, executor=executor, engine="webkit")
        result = row["tool_result"]
        assert (result.status, result.producer_text) == ("ok", "Fake page body"), result.text
        assert result.host_annotations == (CONDITIONS.replace("chromium 141.0.7390.37", "webkit 26.0"),)
        assert launches == ["chromium", "webkit", "webkit"]
        assert all(g._cleanup_done for g in generations[:2])
        assert ctx.browser_state is generations[2]
    finally:
        executor.submit(browser_mod.cleanup_browser, ctx).result()
        executor.shutdown()


@pytest.mark.parametrize("replacement_kind", ["engine", "disconnected", "thread"])
def test_timeout_wins_ensure_replacement_race(fake_browser, monkeypatch, replacement_kind):
    ctx = _ctx()
    launches = fake_browser(ctx)
    _, own = browser_mod._ensure_browser(ctx)
    if replacement_kind == "disconnected":
        own.browser.is_connected = lambda: False
    elif replacement_kind == "thread":
        own._thread_id = -1
    detach = browser_mod._detach_browser
    retirements = []

    def timeout_first(c, expected=None):
        if not retirements:
            retirements.append(detach(c))
        return detach(c, expected=expected)

    monkeypatch.setattr(browser_mod, "_detach_browser", timeout_first)
    out = browser_mod._browse_page(ctx, "https://observed.example/",
                                   engine="webkit" if replacement_kind == "engine" else "chromium")
    ((retired, replacement),) = retirements
    assert out == browser_mod._SESSION_RETIRED_VOID
    assert retired is own and own._cleanup_done
    assert ctx.browser_state is replacement and replacement.browser is None
    assert launches == ["chromium"]


def test_timeout_wins_action_cleanup_race_without_replay(fake_browser, monkeypatch):
    ctx = _ctx()
    clicks, retirements = [], []

    class Page(FakePage):
        def click(self, *_a, **_kw):
            clicks.append("click")
            raise RuntimeError("connection lost after dispatch")

    launches = fake_browser(ctx, page_factory=Page)
    own = ctx.browser_state

    def timeout_during_check(generation):
        assert generation is own
        retirements.append(browser_mod._detach_browser(ctx))
        return True

    monkeypatch.setattr(browser_mod, "_is_infrastructure_error", timeout_during_check)
    out = browser_mod._browser_action(ctx, "click", selector="#submit")
    ((retired, replacement),) = retirements
    assert out == browser_mod._SESSION_RETIRED_VOID and clicks == ["click"]
    assert retired is own and own._cleanup_done
    assert ctx.browser_state is replacement and replacement.browser is None
    assert launches == ["chromium"]


def test_a_call_abandoned_by_its_timeout_never_retries_into_the_replacement(sticky, tmp_path):
    ctx = sticky.registry._ctx
    release = threading.Event()
    launches = sticky.install(page_factory=lambda c: DyingPage(c, release))
    first = ctx.browser_state

    row, executor = _browse_through_the_loop(sticky.registry, tmp_path, 1)
    replacement = ctx.browser_state
    release.set()  # the abandoned call's navigation now dies on the retired generation
    deadline = time.monotonic() + 10
    while not getattr(first, "_cleanup_done", False) and time.monotonic() < deadline:
        time.sleep(0.05)
    executor.shutdown()

    assert row["tool_result"].code == "TOOL_TIMEOUT"
    assert first._cleanup_done and launches == ["chromium"]
    assert ctx.browser_state is replacement is not first and replacement.browser is None


@pytest.mark.parametrize("a_stage", ["navigation", "click", "before_session", "before_wrapper"])
def test_abandoned_a_unwinds_while_b_opens_on_the_replacement(sticky, tmp_path, monkeypatch, a_stage):
    """A times out, B enters the real bound wrapper, THEN A's navigation dies."""
    from ouroboros import loop_tool_execution as execution

    ctx = sticky.registry._ctx
    first = ctx.browser_state
    a_entered, release_a = threading.Event(), threading.Event()
    a_cleaned = threading.Event()
    b_opening, release_b = threading.Event(), threading.Event()
    trace, navigations, clicks, a_futures = [], [], [], []

    def hold_a():
        trace.append(f"A entered {a_stage}")
        a_entered.set()
        assert release_a.wait(10)
        trace.append("A released")

    class Page(FakePage):
        def __init__(self, c):
            super().__init__(c)
            self.generation = c.browser_state

        def goto(self, url, **kw):
            navigations.append(url)
            if self.generation is first:
                hold_a()
                trace.append("A raises")
                raise RuntimeError("Target page, context or browser has been closed")
            return super().goto(url, **kw)

        def click(self, selector, **_kw):
            clicks.append(selector)
            hold_a()
            raise RuntimeError("Target page, context or browser has been closed")

    sticky.install(page_factory=Page)
    cleanup = browser_mod.cleanup_browser_handles

    def record_cleanup(generation):
        cleanup(generation)
        if generation is first:
            a_cleaned.set()

    monkeypatch.setattr(browser_mod, "cleanup_browser_handles", record_cleanup)

    def opening(**_kw):
        if threading.get_ident() == b_thread[0]:
            trace.append("B opening")
            b_opening.set()
            assert release_b.wait(10)

    # Capture B's worker identity at the existing wrapper, without replacing it.
    bound = execution._execute_browser_tool_bound
    b_thread = [None]

    def record_bound(tools, tc, *args):
        if tc["id"] == "call-b":
            b_thread[0] = threading.get_ident()
        elif a_stage == "before_wrapper":
            hold_a()
        return bound(tools, tc, *args)

    execute_single = execution._execute_single_tool

    def before_session(tools, tc, *args, **kwargs):
        if tc["id"] == "call-a" and a_stage == "before_session":
            hold_a()  # wrapper bound A, but the handler has not captured a page yet
        return execute_single(tools, tc, *args, **kwargs)

    monkeypatch.setattr(execution, "_execute_browser_tool_bound", record_bound)
    monkeypatch.setattr(execution, "_execute_single_tool", before_session)
    monkeypatch.setattr(browser_mod, "_ensure_playwright_installed", opening)
    wait = execution.future_result

    def timeout_a(future, seconds):
        if not a_futures:
            a_futures.append(future)
            assert a_entered.wait(10)
            raise TimeoutError()  # deterministically enter the production timeout branch
        return wait(future, seconds)

    monkeypatch.setattr(execution, "future_result", timeout_a)
    executor = execution.StatefulToolExecutor()
    (tmp_path / "logs").mkdir(exist_ok=True)

    def call(letter):
        name, args = "browse_page", {"url": f"https://observed.example/{letter}"}
        if letter == "a" and a_stage == "click":
            name, args = "browser_action", {"action": "click", "selector": "#submit"}
        tc = {"id": f"call-{letter}", "function": {"name": name, "arguments": json.dumps(args)}}
        return execution._execute_with_timeout(sticky.registry, tc, tmp_path / "logs", 30, "root",
                                                stateful_executor=executor)

    with ThreadPoolExecutor(max_workers=1) as caller:
        try:
            a_row = call("a")
            replacement = ctx.browser_state
            trace.append("A timeout retired its generation")
            b_future = caller.submit(call, "b")
            assert b_opening.wait(10)
            release_a.set()
            a_late = a_futures[0].result(timeout=10)
            assert a_cleaned.wait(10)  # include the cleanup queued on the retired worker
            trace.append("A settled")
            b_still_owned = ctx.browser_state is replacement
            b_not_closed = not getattr(replacement, "_cleanup_done", False)
            release_b.set()
            b_row = b_future.result(timeout=10)
            trace.append("B settled")
            evidence = {"stage": a_stage, "trace": trace, "navigations": navigations, "clicks": clicks,
                        "b_still_owned": b_still_owned,
                        "b_not_closed": b_not_closed, "a_late": a_late["result"]}
            print(json.dumps(evidence))
            assert a_row["tool_result"].code == "TOOL_TIMEOUT"
            assert b_still_owned and b_not_closed, evidence
            assert "BROWSER_SESSION_RETIRED" in a_late["result"], evidence
            expected = ["https://observed.example/a"] if a_stage == "navigation" else []
            assert navigations == expected + ["https://observed.example/b"], evidence
            assert clicks == (["#submit"] if a_stage == "click" else []), evidence
            assert b_row["tool_result"].producer_text == "Fake page body"
            assert b_row["tool_result"].host_annotations == (CONDITIONS,)
            assert ctx.browser_state is replacement and first._cleanup_done
        finally:
            release_a.set()
            release_b.set()
            executor.submit(browser_mod.cleanup_browser, ctx).result(timeout=10)
            executor.shutdown()


def test_the_note_survives_loop_truncation_of_a_large_page(sticky, tmp_path, monkeypatch):
    from ouroboros import loop_tool_execution as execution

    ctx = sticky.registry._ctx
    (tmp_path / "logs").mkdir(exist_ok=True)
    sticky.install(page_factory=lambda c: FakePage(c, text="x" * 29000))
    monkeypatch.setattr(execution, "persist_call", lambda *a, **k: {})
    row = execution._execute_single_tool(
        types.SimpleNamespace(_ctx=ctx, CODE_TOOLS=frozenset(),
                              execute_result=lambda name, args: sticky.call(name, args)),
        {"id": "call-browse", "function": {"name": "browse_page",
                                           "arguments": json.dumps({"url": "https://observed.example/"})}},
        tmp_path / "logs", "root")
    messages, trace = [], {"tool_calls": []}
    execution.process_tool_results([row], messages, trace, lambda *a, **k: None,
                                   types.SimpleNamespace(_ctx=ctx),
                                   fit_candidate=measured_fit(window=16_000, reserve=4_000))
    content = messages[0]["content"]
    assert len(content) < 29000 and CONDITIONS in content
    assert trace["tool_calls"][0]["host_annotations"] == [CONDITIONS]
