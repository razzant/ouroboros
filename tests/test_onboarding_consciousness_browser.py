"""The first-run Consciousness offer on a real server, through the real wizard (#1603).

A contained candidate checkout serves each case over its own data root, with the scripted loopback model as
its only provider and a skill-source list naming only the built-in source, so no hub is fetched. Every claim
is read back from the server: the setup save, the one `/bg start` the page sends through the ordinary
command queue, and the control `/api/state` shows afterwards. A later ordinary Stop goes through the same
queue, a restart resumes nothing, and an unticked Finish, a completed setup, a Panic during the wait and a page
left during the wait and shown again from the browser's back/forward cache send nothing. Screenshots go to
``OUROBOROS_UI_EVIDENCE_DIR`` when a launcher names one, else to the test's own temporary directory.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import time
import types
import urllib.error
import urllib.request

import pytest

from tests.candidate_checkout import case_checkout, require_candidate_interpreter, verify_checkout
from tests.fixtures_mock_llm import MockLLMServer
from tests.test_onboarding_attempt_browser import _to_summary
from tests.test_ui_smoke_playwright import (
    REPO_ROOT, _assert_served_candidate, _browser_missing, _fixture_interpreter, _fixture_python, _free_port,
    _wait_health, _wait_supervisor_ready,
)
from tests.ui_console_errors import PageErrors
from ouroboros.test_environment import isolated_environment, owned_process_temp
from ouroboros.test_evidence import keep_startup_log

pytestmark = [pytest.mark.ui_browser]

# No hub, no network: the built-in source alone, and the bundled starter installed off (#1995).
OFFLINE = {
    "OUROBOROS_SKILL_SOURCES": json.dumps([{"id": "builtin", "name": "Built in", "kind": "builtin",
                                            "trust": "trusted", "confirmed": True}]),
    "OUROBOROS_STARTER_SKILLS": '["builtin/telegram"]', "OUROBOROS_STARTER_SKILLS_TURN_ON": False,
    "OUROBOROS_MAX_WORKERS": 1, "OUROBOROS_RUNTIME_MODE": "light",
}
MOCK_MODEL = "openai-compatible::mock-model"
ROW = '[data-consciousness-offer] .consciousness-offer-row'
DEFAULT_ROW = ("Consciousness Start consciousness with Ouroboros Between your requests it continues topics that interest "
               "it and you, notices opportunities and brings you suggestions. Autonomy: Acts. Also acts on its own: starts work, "
               "runs commands, speaks and acts in places. It does not change itself. Up to $20.00 a day in recorded "
               "spending. Subscription quota is separate. At most 2 of its own tasks at once. Change the allowance in "
               "Dashboard › Costs. Its other settings, and Stop and Start, are in Dashboard › Consciousness.")


def _provider(llm_url: str) -> dict:
    return {**dict.fromkeys(("OUROBOROS_MODEL", "OUROBOROS_MODEL_LIGHT", "OUROBOROS_MODEL_FALLBACKS"), MOCK_MODEL),
            "OPENAI_COMPATIBLE_API_KEY": "ui-smoke-key", "OPENAI_COMPATIBLE_BASE_URL": llm_url}


@pytest.fixture()
def setup_server(tmp_path, ui_candidate_checkout):
    """One real server over one data root; ``seed`` rewrites the root while it is stopped, ``start`` waits for
    the gateway (and, with ``supervisor``, the running supervisor and the served-candidate proof)."""
    with case_checkout(pathlib.Path(REPO_ROOT), ui_candidate_checkout) as candidate, MockLLMServer() as llm:
        checkout = candidate.path
        port = _free_port()
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True)
        env = isolated_environment(tmp_path, checkout, process_temp=owned_process_temp())
        env.update({"OUROBOROS_SERVER_HOST": "127.0.0.1", "OUROBOROS_SERVER_PORT": str(port),
                    "OUROBOROS_HOST_SERVICE_PORT": str(_free_port()), "OUROBOROS_NETWORK_PASSWORD": "ui-smoke-password"})
        require_candidate_interpreter(_fixture_interpreter(), env, checkout)
        url = f"http://127.0.0.1:{port}"
        live = types.SimpleNamespace(proc=None, container=None)

        def stop() -> None:
            proc, container = live.proc, live.container
            live.proc = live.container = None
            if container is None:
                return
            try:
                if proc is not None and proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        pass
            finally:
                try:
                    error = container.reap()  # this case's own process tree, never anything else
                finally:
                    container.close()
                if error:
                    raise RuntimeError(f"consciousness offer fixture cleanup failed: {error}")
                if proc is not None:
                    proc.wait(timeout=5)
            candidate.release()

        def prove() -> None:
            _assert_served_candidate(url, checkout, data_dir, live.proc.pid, candidate)

        @keep_startup_log(tmp_path, "server-stdio.log")
        def start(*, supervisor: bool) -> None:
            from ouroboros.process_containment import ProcessContainer

            verify_checkout(checkout, candidate)
            live.container = ProcessContainer()
            candidate.hold()
            with (tmp_path / "server-stdio.log").open("ab") as log:
                live.proc = live.container.spawn([_fixture_python(), "server.py"], cwd=checkout, env=env,
                                                 stdout=log, stderr=subprocess.STDOUT)
            _wait_health(url, timeout_sec=90)
            if supervisor:
                _wait_supervisor_ready(url, timeout_sec=120)
                prove()

        def seed(settings: dict | None) -> None:
            stop()
            for child in data_dir.iterdir():
                shutil.rmtree(child) if child.is_dir() else child.unlink()
            if settings is not None:
                (data_dir / "settings.json").write_text(json.dumps(settings), encoding="utf-8")

        try:
            yield types.SimpleNamespace(url=url, data_dir=data_dir, llm_url=llm.base_url, start=start, stop=stop,
                                        seed=seed, prove=prove, live=live)
        finally:
            stop()


def _get(url: str, path: str) -> dict:
    with urllib.request.urlopen(url + path, timeout=5) as resp:  # noqa: S310 - the case's own loopback server
        return json.load(resp)


def _command(url: str, cmd: str) -> int:
    request = urllib.request.Request(url + "/api/command", data=json.dumps({"cmd": cmd}).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=10) as resp:  # noqa: S310
            return resp.status
    except urllib.error.HTTPError as exc:
        return exc.code


def _control(url: str):
    return _get(url, "/api/state").get("bg_consciousness_enabled")


def _wait_control(url: str, expected, timeout: float = 90.0):
    deadline, seen = time.time() + timeout, None
    while time.time() < deadline:
        try:
            seen = _control(url)
        except Exception as exc:  # a restart in progress
            seen = f"unreadable: {exc}"
        if seen is expected:
            return seen
        time.sleep(0.5)
    raise AssertionError(f"bg_consciousness_enabled stayed {seen!r}, expected {expected!r}")


def _shots(tmp_path: pathlib.Path) -> pathlib.Path:
    """The launcher's evidence directory when it names one, else this test's own: never the current directory."""
    named = os.environ.get("OUROBOROS_UI_EVIDENCE_DIR")
    out = (pathlib.Path(named) if named else tmp_path / "ui-evidence") / "1603"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _text(page, selector: str) -> str:
    return " ".join(page.locator(selector).inner_text().split())


def _record(page) -> dict:
    seen = {"commands": [], "completions": [], "errors": PageErrors(None, page.context.browser.browser_type.name)}
    seen["errors"].watch(page, navigates=False)  # strict: only a caller driving a hand-off declares it

    def request(req):
        if req.url.endswith("/api/command") and req.method == "POST":
            seen["commands"].append(json.loads(req.post_data or "{}"))
        if req.url.endswith("/api/onboarding/complete") and req.method == "POST":
            seen["completions"].append(json.loads(req.post_data or "{}"))
    page.on("request", request)
    return seen


def _fresh_accounts_and_models(page, llm_url: str) -> None:
    """Accounts → Models on a providerless first setup: the compatible endpoint, Main and Light on its model."""
    from playwright.sync_api import expect

    page.locator('details[data-collapse="api-access"] > summary').click()
    card = page.locator('[data-provider-card="compatible"]')
    toggle = card.locator('[data-provider-toggle]')
    if toggle.get_attribute("aria-expanded") != "true":
        toggle.click()
    row = card.locator('[data-connection-row][data-builtin="1"]')
    row.locator('[data-connection-field="base_url"]').fill(llm_url)
    row.locator('[data-connection-field="api_key"]').fill("ui-smoke-key")
    page.locator("#next-btn:not([disabled])").click(timeout=30_000)
    expect(page.locator('[data-model-role="main"]')).to_be_visible(timeout=30_000)
    for role in ("main", "light"):
        slot = page.locator(f'[data-model-role="{role}"]')
        slot.locator("[data-model-role-source]").select_option("api:openai-compatible")
        slot.locator("[data-model-role-model]").fill("mock-model")
    for _ in range(3):  # Models → Review → Budget → Summary
        title = page.locator(".step-title").inner_text()
        page.locator("#next-btn:not([disabled])").click(timeout=60_000)
        page.wait_for_function("t => document.querySelector('.step-title')?.textContent !== t", arg=title, timeout=30_000)
    page.wait_for_selector(".summary-card", timeout=30_000)


def _finish(page) -> None:
    """Start Ouroboros once the wizard's own preview gate opens."""
    for _ in range(60):
        button = page.locator("#next-btn")
        if button.count() and button.is_enabled():
            button.click()
            return
        page.wait_for_timeout(500)
    raise AssertionError("Start Ouroboros never became available")


def _browser(pw, engine="webkit", theme="light", size=(980, 752)):
    browser = getattr(pw, engine).launch(headless=True)
    context = browser.new_context(viewport={"width": size[0], "height": size[1]}, color_scheme=theme, device_scale_factor=2)
    return browser, context.new_page()


def test_first_setup_ticked_starts_once_through_the_queue_and_a_later_stop_wins(setup_server, tmp_path):
    """A first setup without a provider: the save starts the supervisor, its state is initialized, the page sends
    `/bg start` once and calls it on only when the state shows it. A later ordinary Stop, and a Start and Stop sent
    back to back through the same queue, end off; a restart resumes nothing."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import expect, sync_playwright

    srv, out = setup_server, _shots(tmp_path)
    srv.seed(OFFLINE)  # settings present, setup never completed, no provider
    srv.start(supervisor=False)
    assert urllib.request.urlopen(srv.url + "/api/onboarding", timeout=5).status == 200, "providerless before setup"
    assert not (srv.data_dir / "state" / "state.json").exists(), "no state before the setup save"
    # Before setup there is no command queue: an ordinary Stop is refused, never held for later.
    assert _command(srv.url, "/bg stop") == 409
    with sync_playwright() as pw:
        browser, page = _browser(pw)
        try:
            seen = _record(page)
            page.goto(srv.url + "/onboarding", wait_until="networkidle", timeout=60_000)
            _fresh_accounts_and_models(page, srv.llm_url)
            expect(page.locator(ROW)).to_be_visible()
            assert _text(page, ROW) == DEFAULT_ROW
            box = page.locator("[data-consciousness-start]")
            assert not box.is_checked(), "the offer starts unticked"
            page.screenshot(path=str(out / "first-setup-summary-unticked-webkit-light.png"), full_page=True)
            box.check()
            page.locator(ROW).screenshot(path=str(out / "first-setup-row-ticked-webkit-light.png"))
            seen["errors"].watch(page)  # Finish hands off to the app: leaving may cut off the page's own reads (tests/ui_console_errors.py)
            _finish(page)
            page.wait_for_selector("[data-consciousness-run]", timeout=60_000)
            page.screenshot(path=str(out / "first-setup-after-save-webkit-light.png"))
            page.wait_for_url(srv.url + "/", timeout=240_000)
            page.wait_for_timeout(1500)
            # The press is the owner's act, its command the host's words (#2778): Main shows the one System
            # receipt, never a "You · /bg start" bubble, on the live page and in the history a reload reads.
            receipt = page.locator("#chat-messages .chat-bubble").filter(has_text="Background consciousness")
            receipt.first.wait_for(state="visible", timeout=60_000)
            page.screenshot(path=str(out / "first-setup-main-after-handoff-webkit-light.png"))
            assert page.locator("#chat-messages .chat-bubble.user").count() == 0, "a press writes no owner bubble"
            history = _get(srv.url, "/api/chat/history?limit=50")["messages"]
            assert [row["text"] for row in history if row.get("role") == "user"] == [], history
            assert [row["system_type"] for row in history if "Background consciousness" in row.get("text", "")] == ["command_reply"]
            page.reload(wait_until="networkidle")
            page.locator("#chat-messages .chat-bubble").filter(has_text="Background consciousness").first.wait_for(
                state="visible", timeout=60_000)
            assert page.locator("#chat-messages .chat-bubble.user").count() == 0, "nor after a reload"
            page.screenshot(path=str(out / "first-setup-main-after-reload-webkit-light.png"))
            assert seen["commands"] == [{"cmd": "/bg start", "if_boot": _get(srv.url, "/api/health")["boot"]["since"],
                                         "ui_control": {"name": "setup_consciousness"}}], seen["commands"]
            assert len(seen["completions"]) == 1
            assert not any("onscious" in key for key in seen["completions"][0]), "nothing rides the save"
            assert not seen["errors"].unexpected(), seen["errors"].report()
        finally:
            browser.close()
    assert _control(srv.url) is True, "observed on through the real state"
    srv.prove()  # the supervisor the save started serves this candidate
    saved = json.loads((srv.data_dir / "settings.json").read_text(encoding="utf-8"))
    assert saved.get("OUROBOROS_ONBOARDING_COMPLETED_AT")
    assert not [key for key in saved if "CONSCIOUS" in key and any(word in key for word in ("START", "CONSENT", "OPT", "PENDING"))]

    # A later ordinary Stop through the same queue wins over the setup's Start.
    assert _command(srv.url, "/bg stop") == 200
    _wait_control(srv.url, False)
    # Start then Stop, back to back: the queue keeps their order and the Stop is the last word.
    assert _command(srv.url, "/bg start") == 200 and _command(srv.url, "/bg stop") == 200
    time.sleep(6)
    assert _control(srv.url) is False
    # Nothing durable re-arms the setup's Start: a restart keeps it off.
    srv.stop()
    srv.start(supervisor=True)
    time.sleep(5)
    assert _control(srv.url) is False


def test_markerless_install_offers_its_own_limits_and_an_unticked_finish_sends_nothing(setup_server, tmp_path):
    """Settings present, a provider ready, setup never completed (a markerless older install): the unticked offer
    shows this install's limits; Finish without the tick sends nothing; the completed setup shows no row."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import expect, sync_playwright

    srv, out = setup_server, _shots(tmp_path)
    srv.seed({**OFFLINE, **_provider(srv.llm_url), "OUROBOROS_CONSCIOUSNESS_AUTONOMY": "observe",
              "OUROBOROS_CONSCIOUSNESS_DAILY_USD": 5, "OUROBOROS_CONSCIOUSNESS_MAX_TASKS": 1})
    srv.start(supervisor=True)
    assert _control(srv.url) is False
    with sync_playwright() as pw:
        for engine, theme in (("webkit", "dark"), ("chromium", "light"), ("chromium", "dark")):
            browser, page = _browser(pw, engine, theme)
            try:
                page.goto(srv.url + "/onboarding?builtin=1", wait_until="networkidle", timeout=60_000)
                _to_summary(page, pro=False)
                expect(page.locator(ROW)).to_be_visible()
                assert _text(page, ROW) == (
                    "Consciousness Start consciousness with Ouroboros Between your requests it continues topics that "
                    "interest it and you, notices opportunities and brings you suggestions. Autonomy: Advises. On its own it "
                    "studies, keeps its memory and tells you; it changes nothing outside. Up to $5.00 a day in recorded "
                    "spending. Subscription quota is separate. At most 1 of its own tasks at once. Change the "
                    "allowance in Dashboard › Costs. Its other settings, and Stop and Start, are in Dashboard › Consciousness.")
                page.screenshot(path=str(out / f"markerless-summary-{engine}-{theme}.png"), full_page=True)
                page.locator(".summary-card").screenshot(path=str(out / f"markerless-summary-card-{engine}-{theme}.png"))
                overflow = page.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
                assert overflow <= 1, overflow
            finally:
                browser.close()
        browser, page = _browser(pw, "webkit", "light", (390, 844))
        try:
            page.goto(srv.url + "/onboarding?builtin=1", wait_until="networkidle", timeout=60_000)
            _to_summary(page, pro=False)
            page.locator(ROW).scroll_into_view_if_needed()
            page.screenshot(path=str(out / "markerless-summary-webkit-light-390.png"))
            assert page.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth") <= 1
        finally:
            browser.close()
        browser, page = _browser(pw)
        try:
            seen = _record(page)
            page.goto(srv.url + "/onboarding?builtin=1", wait_until="networkidle", timeout=60_000)
            _to_summary(page, pro=False)
            assert not page.locator("[data-consciousness-start]").is_checked()
            seen["errors"].watch(page)  # Finish hands off to the app: leaving may cut off the page's own reads (tests/ui_console_errors.py)
            _finish(page)
            page.wait_for_url(srv.url + "/", timeout=120_000)
            assert seen["commands"] == [], "an unticked Finish sends no Start"
            assert len(seen["completions"]) == 1 and not any("onscious" in key for key in seen["completions"][0])
            time.sleep(3)
            assert _control(srv.url) is False
            # The completed setup re-entering the wizard: no row, so no old choice can be repeated.
            page.goto(srv.url + "/onboarding?builtin=1", wait_until="networkidle", timeout=60_000)
            _to_summary(page, pro=False)
            assert page.locator(ROW).count() == 0
            page.screenshot(path=str(out / "completed-setup-summary-no-row-webkit-light.png"), full_page=True)
            assert not seen["errors"].unexpected(), seen["errors"].report()
        finally:
            browser.close()


def test_panic_while_the_page_waits_sends_nothing_and_says_so(setup_server, tmp_path):
    """The page waits for this server's chat queue; a Panic during that wait ends the server, the page says the
    Start was not sent and sends nothing, and the restarted install keeps consciousness off."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import expect, sync_playwright

    srv, out = setup_server, _shots(tmp_path)
    srv.seed(OFFLINE)
    srv.start(supervisor=False)
    since = _get(srv.url, "/api/health")["boot"]["since"]
    held = json.dumps({"status": "ok", "boot": {"phase": "ready", "since": since, "dialogue_ready": False}})
    with sync_playwright() as pw:
        browser, page = _browser(pw)
        try:
            seen = _record(page)
            page.goto(srv.url + "/onboarding", wait_until="networkidle", timeout=60_000)
            _fresh_accounts_and_models(page, srv.llm_url)
            page.locator("[data-consciousness-start]").check()
            # Only the page's readiness read is held "not ready"; the save and everything else are real.
            page.route("**/api/health", lambda route: route.fulfill(status=200, content_type="application/json", body=held))
            _finish(page)
            expect(page.locator('[data-consciousness-run="waiting"]')).to_be_visible(timeout=60_000)
            _wait_supervisor_ready(srv.url, timeout_sec=120)  # the save started the supervisor; its state exists
            page.screenshot(path=str(out / "panic-wait-screen-webkit-light.png"))
            assert _command(srv.url, "/panic") == 200
            deadline = time.time() + 30
            while time.time() < deadline and srv.live.proc.poll() is None:
                time.sleep(0.2)
            assert srv.live.proc.poll() is not None, "Panic ended the server"
            page.unroute("**/api/health")
            expect(page.locator('[data-consciousness-run="not_started"]')).to_be_visible(timeout=30_000)
            assert _text(page, '[data-consciousness-run="not_started"]') == (
                "Consciousness was not started Ouroboros stopped answering before the Start could be sent, so nothing "
                "was sent. Dashboard › Consciousness shows whether it is on; its Start is there. Open Ouroboros")
            page.screenshot(path=str(out / "panic-not-started-screen-webkit-light.png"))
            assert seen["commands"] == [], "nothing was sent"
            assert not seen["errors"], seen["errors"]
        finally:
            browser.close()
    srv.stop()
    srv.start(supervisor=True)
    time.sleep(3)
    assert _control(srv.url) is False, "the restarted install keeps consciousness off"


# Records each pageshow/pagehide with its `persisted` flag before the wizard's own listeners exist.
LIFECYCLE = ("window.__lifecycle = []; for (const type of ['pageshow', 'pagehide']) "
             "addEventListener(type, (event) => window.__lifecycle.push(`${type}:${event.persisted}`));")


def _cached_chromium(pw):
    """Chromium that keeps a left page in its back/forward cache, as people's browsers do. Playwright launches it
    with `--disable-back-forward-cache`, and its headless shell reloads instead of restoring."""
    browser = pw.chromium.launch(headless=True, channel="chromium", ignore_default_args=["--disable-back-forward-cache"])
    context = browser.new_context(viewport={"width": 980, "height": 752}, color_scheme="light", device_scale_factor=2)
    return browser, context.new_page()


def test_a_page_left_during_the_wait_and_shown_again_from_the_cache_sends_nothing(setup_server, tmp_path):
    """Tick, a confirmed save, the page waits for readiness; the owner opens another page and comes Back. The
    browser shows the very same page again from its back/forward cache (pagehide and pageshow persisted), its
    timers resumed, and the server is ready by then. The page sends nothing, says it was left, and hands off
    only on Open Ouroboros; consciousness stays off."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import expect, sync_playwright

    srv, out = setup_server, _shots(tmp_path)
    srv.seed(OFFLINE)
    srv.start(supervisor=False)
    since = _get(srv.url, "/api/health")["boot"]["since"]
    held = json.dumps({"status": "ok", "boot": {"phase": "ready", "since": since, "dialogue_ready": False}})
    hold = {"on": True}

    def health(route):  # only the page's readiness read is held "not ready" while the owner is on the page
        if hold["on"]:
            route.fulfill(status=200, content_type="application/json", body=held)
        else:
            route.continue_()

    with sync_playwright() as pw:
        try:
            browser, page = _cached_chromium(pw)
        except PlaywrightError as exc:
            if _browser_missing(exc):
                pytest.skip(str(exc))
            raise
        try:
            seen = _record(page)
            page.add_init_script(LIFECYCLE)
            page.goto(srv.url + "/onboarding", wait_until="networkidle", timeout=60_000)
            _fresh_accounts_and_models(page, srv.llm_url)
            page.locator("[data-consciousness-start]").check()
            page.route("**/api/health", health)
            _finish(page)
            expect(page.locator('[data-consciousness-run="waiting"]')).to_be_visible(timeout=60_000)
            _wait_supervisor_ready(srv.url, timeout_sec=120)  # ready for real: a resumed old wait would send now
            page.goto(srv.url + "/static/candidate-sentinel.txt", timeout=60_000)  # the owner opens another page
            hold["on"] = False
            page.go_back(wait_until="commit", timeout=60_000)  # a restore fires no load event
            page.wait_for_function("() => (window.__lifecycle || []).includes('pageshow:true')", timeout=30_000)
            assert page.evaluate("window.__lifecycle") == ["pageshow:false", "pagehide:true", "pageshow:true"], (
                "the browser reloaded instead of restoring the page, so this case proves nothing")
            expect(page.locator('[data-consciousness-run="not_started"]')).to_be_visible(timeout=30_000)
            page.wait_for_timeout(3_000)  # three of the wait's polls: an old wait resumed here would send
            assert seen["commands"] == [], "no Start after the page was left"
            assert _text(page, '[data-consciousness-run="not_started"]') == (
                "Consciousness was not started You left this page before the Start could be sent, so nothing was "
                "sent. Dashboard › Consciousness shows whether it is on; its Start is there. Open Ouroboros")
            page.screenshot(path=str(out / "left-and-back-not-started-chromium-light.png"))
            page.locator("#consciousness-open-btn").click()
            page.wait_for_url(srv.url + "/", timeout=120_000)
            assert seen["commands"] == [], "nor on the hand-off"
            assert len(seen["completions"]) == 1
            assert not seen["errors"], seen["errors"]
        finally:
            browser.close()
    assert _control(srv.url) is False, "consciousness stays off"
