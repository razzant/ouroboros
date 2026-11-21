"""Transport-only pytest entry point. Run from the product root in the offline Linux guest."""
import json
import linecache
import os
import struct
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path.cwd()
OUT = Path("/results")
sys.path.insert(0, str(ROOT))


@pytest.fixture
def screen(request):
    from tests.test_subscription_setup_browser import subscription_ui

    engine = os.environ["OUROBOROS_UI_BROWSER_ENGINE"]
    width, height, theme = (1440, 900, "light") if engine == "chromium" else (390, 844, "dark")
    width = getattr(request, "param", width)
    with contextmanager(subscription_ui.__wrapped__)() as ui:
        servers = [
            dict(id=key, name=name, enabled=enabled, transport="streamable_http",
                 url=f"https://{key}.example.test/mcp")
            for key, name, enabled in (("docs", "Docs", True), ("calendar", "Team calendar", True),
                                       ("archive", "Research archive", False))
        ]
        ui["settings"].update(MCP_ENABLED=True, MCP_SERVERS=servers)
        status = [dict(row, tool_count=0, tools=[]) for row in servers]
        status[0].update(tool_count=2, listed_at=datetime.now(timezone.utc).isoformat(),
                         tools=[{"name": "list_topics"}, {"name": "read_topic"}])
        status[1]["sign_in"] = {"state": "not_signed_in", "reason": "not_signed_in"}
        page = ui["new_page"](viewport={"width": width, "height": height}, has_touch=width < 720)
        fixtures = ROOT / "web/tests/fixtures/skills_v4"

        def respond(payload):
            def answer(route):
                route.fulfill(json=payload)
            return answer

        for endpoint, name in (("extensions", "extensions_v4"), ("skill-sources", "skill_sources"),
                               ("skill-catalog", "skill_catalog")):
            payload = json.loads((fixtures / f"{name}.json").read_bytes())
            if endpoint == "skill-sources":
                for source in payload["sources"]:
                    source["confirmed"] = True
            page.route(f"**/api/{endpoint}*", respond(payload))
        page.route("**/api/skills/lifecycle-queue", lambda r: r.fulfill(json={"active": None, "events": []}))
        page.route("**/api/mcp/status", lambda r: r.fulfill(json={"sdk_available": True, "enabled": True,
                                                                  "servers": status}))
        page.goto(ui["url"] + "/#skills")
        page.wait_for_function("() => Boolean(window.ouroTheme)")
        page.evaluate("theme => window.ouroTheme.set(theme)", theme)
        assert page.context.browser.browser_type.name == engine
        yield page, width, theme


@pytest.mark.ui_browser
def test_picture(screen):
    from playwright.sync_api import expect

    page, width, theme = screen
    section = page.locator("#page-skills.active")
    door = section.locator('[data-extensions-open="tools"]')
    expect(door).to_contain_text("2 of 3 on")
    door.click()
    docs = section.locator('[data-entry-id="docs"]')
    expect(docs).to_contain_text("2 tools")
    expect(section.locator('[data-entry-id="calendar"] [data-entry-sign-in]')).to_have_text("Sign in")
    for key, checked in (("docs", True), ("calendar", True), ("archive", False)):
        expect(section.locator(f'[data-entry-id="{key}"] [data-entry-switch]')).to_be_checked(checked=checked)
    expect(page.locator("html")).to_have_attribute("data-theme", theme)
    section.locator("[data-settings-arrive]:visible").wait_for(state="hidden")
    page.evaluate("() => document.fonts.ready")
    engine = page.context.browser.browser_type.name
    png = page.screenshot(animations="disabled")
    assert png.startswith(b"\x89PNG\r\n\x1a\n") and 24 < len(png) <= 3_000_000
    assert struct.unpack(">II", png[16:24]) == (width, page.viewport_size["height"])
    (OUT / f"extensions-{engine}-{width}-{theme}.png").write_bytes(png)


@pytest.mark.ui_browser
@pytest.mark.parametrize("screen", [390], indirect=True)
def test_four_pages_and_returns(screen):
    from playwright.sync_api import expect
    from tests.settings_section_helpers import header_back
    from tests.settings_browser_helpers import settings_page

    page, _, _ = screen
    section = page.locator("#page-skills.active")
    rail = section.locator("[data-extensions-rail]")
    for key, title in (("installed", "Skills"), ("tools", "MCP servers"),
                       ("browse", "Browse"), ("sources", "Sources")):
        door = rail.locator(f'[data-extensions-open="{key}"]')
        door.press("Enter")
        expect(section.locator("[data-extensions-lead-name]")).to_have_text(title)
        panel = settings_page(page, "tools") if key == "tools" else section.locator(f"#skills-pane-{key}")
        expect(panel).to_be_visible()
        if key == "sources":
            for action, target in (("auto_grant", "permissions"), ("skill_sources", "skill-sources")):
                opener = panel.locator(f'[data-source-action="{action}"]')
                opener.press("Enter")
                expect(settings_page(page, target)).to_be_visible()
                header_back(page).press("Enter")
                expect(opener).to_be_focused()
        header_back(page).press("Enter")
        expect(door).to_be_focused()


@pytest.mark.ui_browser
def test_status_loss_and_recovery(screen):
    """The mounted list and form receive ordinary status results; no sign-in action is invoked."""
    from playwright.sync_api import expect

    page, _, _ = screen
    section = page.locator("#page-skills.active")
    section.locator('[data-extensions-open="tools"]').click()
    docs = section.locator('[data-entry-id="docs"]')
    calendar = section.locator('[data-entry-id="calendar"]')
    action = calendar.locator('[data-entry-sign-in]')
    expect(docs).to_contain_text('2 tools')
    expect(action).to_be_visible()

    pattern = '**/api/mcp/status'
    def unavailable(route):
        route.fulfill(status=503, json={})

    # Await the real client's read and render. Reconnect and visibility delivery are covered separately.
    read = "async () => { const m = await import('/static/modules/mcp_settings.js'); await m.refreshStatus(); }"
    page.route(pattern, unavailable)
    page.evaluate(read)
    expect(docs.locator('[data-mcp-state]')).to_have_text('MCP status unavailable')
    expect(action).to_have_count(0)
    page.unroute(pattern, unavailable)
    page.evaluate(read)
    expect(docs).to_contain_text('2 tools')
    expect(action).to_be_visible()

    calendar.locator('[data-entry-open]').click()
    section.locator('.mcp-sign-in-details:visible > summary').click()
    field = section.locator('[data-mcp-sign-in-client-id]:visible')
    field.fill('demo-app')
    field.evaluate('el => { window.qaEditedField = el; }')
    page.route(pattern, unavailable)
    page.evaluate(read)
    expect(field).to_have_value('demo-app')
    expect(field).to_be_focused()
    assert field.evaluate('el => el === window.qaEditedField')
    page.unroute(pattern, unavailable)
    page.evaluate(read)
    expect(field).to_have_value('demo-app')
    expect(field).to_be_focused()
    assert field.evaluate('el => el === window.qaEditedField')


@pytest.mark.ui_browser
def test_circles_return(screen):
    from playwright.sync_api import expect
    from tests.settings_browser_helpers import open_settings

    page, _, _ = screen
    page.route("**/api/places", lambda r: r.fulfill(json={"circles": [], "connectors": [], "places": []}))
    page.goto(page.url.partition("#")[0] + "#settings")
    servers = open_settings(page, "tools")
    servers.get_by_text("Circles", exact=True).click()
    page.locator('[data-circles-act="add-place"]:visible').first.click()
    dialog = page.locator("dialog.place-connect:visible")
    field = dialog.locator('[data-connect-draft="words"]')
    field.fill("Local demonstration")
    dialog.locator('[data-connect-act="next"]').click()
    dialog.locator('[data-connect-act="back-words"]').click()
    expect(field).to_have_value("Local demonstration")
    dialog.locator("[data-place-close]").click()
    page.locator("#page-circles > .app-page-header").get_by_role(
        "button", name="Back to Extensions › MCP servers", exact=True).click()
    expect(servers.locator('[data-entry-id="docs"]')).to_be_visible()


MAP_FILE = "tests/test_docs_sync.py"
MAP_TEST = "test_architecture_component_map_covers_every_live_runtime_module"
MAP_CASE = f"{MAP_FILE}::{MAP_TEST}"
MAP_GAP = ["ouroboros/gateway/places_api.py"]


def map_assertion(error):
    """Only the real owner-map assertion supplies a comparable missing-module list."""
    if type(error) is not AssertionError:
        return {}
    tb = error.__traceback__
    while tb and tb.tb_next:
        tb = tb.tb_next
    if not tb:
        return {}
    code = tb.tb_frame.f_code
    if (Path(code.co_filename).resolve() != ROOT / MAP_FILE or code.co_name != MAP_TEST
            or linecache.getline(code.co_filename, tb.tb_lineno).strip() != "assert not missing, ("):
        return {}
    missing = tb.tb_frame.f_locals.get("missing")
    if not isinstance(missing, list) or not all(isinstance(path, str) for path in missing):
        return {}
    message = f"docs/ARCHITECTURE.md names no owner for these live modules: {missing}"
    return {"missing_count": len(missing), "missing_modules": [p[:160] for p in missing[:8]],
            "known_map_assertion": missing == MAP_GAP and str(error).split("\n", 1)[0] == message}


def skip_reason(report):
    reason = getattr(report, "wasxfail", None)
    if reason is None and isinstance(report.longrepr, tuple):
        reason = report.longrepr[-1]
    return str(reason or "No reason supplied").split("\n", 1)[0][:240]


class Results:
    def __init__(self, mode, targets):
        self.mode = mode
        self.strict = mode != "broad"
        self.targets = [t for t in targets if not t.startswith("-") and (t.endswith(".py") or ".py::" in t)]
        self.expected_engines = ({"chromium", "webkit"} if mode == "native" else
                                 {mode} if mode in ("chromium", "webkit") else set())
        self.selected = set()
        self.deselected = 0
        self.finished = set()
        self.passed = set()
        self.missing_targets = []
        self.engines = set()
        self.case_engines = set()
        self.skips = 0
        self.failures = 0
        self.collection_errors = 0
        self.map_match = False
        self.map_only_failure = False
        self.verdict = None

    def record(self, **row):
        with (OUT / "cases.jsonl").open("a", encoding="utf-8") as stream:
            print(json.dumps(row, ensure_ascii=True), file=stream)

    def pytest_sessionstart(self):
        if not self.expected_engines:
            return
        from playwright.sync_api import BrowserType

        driver_node = os.environ["PLAYWRIGHT_NODEJS_PATH"]
        launch = BrowserType.launch

        def observed(browser_type, *args, **kwargs):
            browser = launch(browser_type, *args, **kwargs)
            # The Python SDK's live transport process, not the 'node' found on PATH.
            proc = browser._impl_obj._connection._transport._proc
            assert os.path.samefile(f"/proc/{proc.pid}/exe", driver_node)
            if browser_type.name not in self.engines:
                self.record(browser=browser_type.name, version=browser.version, driver_node="24.22.3")
            self.engines.add(browser_type.name)
            self.case_engines.add(browser_type.name)
            return browser
        BrowserType.launch = observed

    def pytest_runtest_logstart(self):
        self.case_engines.clear()

    def pytest_deselected(self, items):
        self.deselected += len(items)

    def pytest_collectreport(self, report):
        if report.skipped:
            self.skips += 1
            self.record(case=report.nodeid[:320], phase="collect", outcome="skipped", reason=skip_reason(report))
        elif report.failed:
            self.collection_errors += 1
            self.record(case=report.nodeid[:320], phase="collect", outcome="failed")

    def pytest_collection_finish(self, session):
        self.selected = {item.nodeid for item in session.items}
        if self.strict:
            collected = {(item.path.resolve(), name) for item in session.items
                         for name in ("", item.name, getattr(item, "originalname", None))}
            for target in self.targets:
                path, _, name = target.partition("::")
                if (Path(path).resolve(), name) not in collected:
                    self.missing_targets.append(target)
        self.record(selected=len(self.selected), missing_targets=self.missing_targets)

    @pytest.hookimpl(hookwrapper=True, tryfirst=True)
    def pytest_runtest_makereport(self, item, call):
        report = (yield).get_result()
        if call.when != "call" and report.passed:
            return
        outcome = report.outcome
        if hasattr(report, "wasxfail"):
            outcome = "xfailed" if report.skipped else "xpassed"
        self.finished.add(item.nodeid)
        row = dict(case=item.nodeid[:320], phase=call.when, outcome=outcome)
        if self.expected_engines and call.when == "call":
            row["browsers_started"] = sorted(self.case_engines)
            if (outcome == "passed" and item.originalname ==
                    "test_skills_card_geometry_tracks_card_width_and_keeps_grant_on_card"
                    and self.case_engines != {item.callspec.params["engine"]}):
                outcome = row["outcome"] = "wrong_browser"
        if outcome == "passed":
            self.passed.add(item.nodeid)
            if not self.strict:
                return
        elif report.skipped:
            self.skips += 1
            row["reason"] = skip_reason(report)
        else:
            self.failures += 1
        if report.failed and call.excinfo:
            row["type"] = call.excinfo.typename
            for frame in reversed(call.excinfo.traceback):
                path = Path(str(frame.path)).resolve()
                if path.is_relative_to(ROOT) or path == Path(__file__).resolve():
                    row["at"] = f"{path.name}:{frame.lineno + 1}"
                    break
            if self.mode == "docs-map" and item.nodeid == MAP_CASE and call.when == "call":
                row.update(map_assertion(call.excinfo.value))
                self.map_match = row.get("known_map_assertion", False)
        self.record(**row)

    def pytest_sessionfinish(self, session, exitstatus):
        missing = self.selected - self.finished
        self.map_only_failure = (self.map_match and self.selected == self.finished == {MAP_CASE}
                                 and self.failures == 1 and not self.passed and not self.skips
                                 and not self.collection_errors and not self.missing_targets
                                 and session.testsfailed == 1 and exitstatus == pytest.ExitCode.TESTS_FAILED)
        if (not self.passed or missing or self.missing_targets or self.failures or self.collection_errors
                or self.engines != self.expected_engines
                or (self.strict and (self.skips or self.passed != self.selected))):
            session.exitstatus = session.exitstatus or pytest.ExitCode.TESTS_FAILED
        partial = self.skips or (not self.strict and self.deselected)
        self.verdict = "failed" if session.exitstatus else "partial" if partial else "passed"
        self.record(verdict=self.verdict, exit=int(session.exitstatus), selected=len(self.selected), passed=len(self.passed),
                    deselected=self.deselected,
                    skipped=self.skips, failed=self.failures, collection_errors=self.collection_errors,
                    missing=sorted(missing)[:15], engines=sorted(self.engines))
        with (OUT / "summary.txt").open("a", encoding="utf-8") as stream:
            print(f"pytest={self.verdict} selected={len(self.selected)} passed={len(self.passed)} "
                  f"skipped_or_xfailed={self.skips} failed_or_xpassed={self.failures} deselected={self.deselected}", file=stream)


if __name__ == "__main__":
    mode, *targets = sys.argv[1:]
    if mode not in {"chromium", "webkit", "native", "python", "docs-map", "broad"} or not targets:
        raise SystemExit("Expected a qualification mode and explicit test targets")
    results = Results(mode, targets)
    rc = pytest.main(["-c", "pyproject.toml", "-o", "addopts=", "-p", "no:cacheprovider",
                      "--tb=line", "--show-capture=no", "--timeout=90", *targets], plugins=[results])
    if results.map_only_failure and rc == pytest.ExitCode.TESTS_FAILED:
        (OUT / "known-map-failure.txt").write_text(MAP_GAP[0] + "\n", encoding="utf-8")
    if rc == pytest.ExitCode.OK and results.verdict not in {"passed", "partial"}:
        rc = pytest.ExitCode.TESTS_FAILED
    raise SystemExit(rc)
