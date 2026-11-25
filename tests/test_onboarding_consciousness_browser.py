"""The first-run Consciousness step on a real server, through the real wizard (owner decision 2026-11-20).

A contained candidate checkout serves each case over its own data root, with the scripted loopback model as its only
provider and a skill-source list naming only the built-in source, so no hub is fetched. Every claim is read back from
the server: the completion request carries the switch's position, the page sends no command, the first state is born
with consciousness on, and Main holds no System row after a first message and a restart; a later Stop wins across a
restart. An older install that already has a state shows no step and stays as its state says, and a completed setup
shows no step. Screenshots go to ``OUROBOROS_UI_EVIDENCE_DIR`` when a launcher names one, else to the test's own
temporary directory.
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
from tests.test_onboarding_attempt_browser import _start, _to_summary
from tests._wizard_walk import step_order
from tests.test_ui_smoke_playwright import (
    REPO_ROOT, _assert_served_candidate, _fixture_interpreter, _fixture_python, _free_port, _wait_health,
    _wait_supervisor_ready,
)
from tests.ui_console_errors import PageErrors
from ouroboros.test_environment import isolated_environment, owned_process_temp
from ouroboros.test_evidence import keep_startup_log
from ouroboros.consciousness_view import FIRST_RUN

pytestmark = [pytest.mark.ui_browser]

# No hub, no network: the built-in source alone, and the bundled starter installed off (#1995).
OFFLINE = {
    "OUROBOROS_SKILL_SOURCES": json.dumps([{"id": "builtin", "name": "Built in", "kind": "builtin",
                                            "trust": "trusted", "confirmed": True}]),
    "OUROBOROS_STARTER_SKILLS": '["builtin/telegram"]', "OUROBOROS_STARTER_SKILLS_TURN_ON": False,
    "OUROBOROS_MAX_WORKERS": 1, "OUROBOROS_RUNTIME_MODE": "light",
}
MOCK_MODEL = "openai-compatible::mock-model"
STEP = "[data-consciousness-step]"
LIMITS = ("Autonomy: Acts. Also acts on its own: starts work, runs commands, speaks and acts in places. It does not "
          "change itself. Up to $15.00 a day in recorded spending. Subscription quota is separate. At most 2 of its own "
          "tasks at once.")
# The step's words are the host's (consciousness_view.first_run_words, owner 15.11, 5A); the meaning leads as its copy.
STEP_TEXT = f"Off {FIRST_RUN['off']} On {FIRST_RUN['on']} {LIMITS} {FIRST_RUN['door']}"


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
    out = (pathlib.Path(named) if named else tmp_path / "ui-evidence") / "consciousness-step"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _text(page, selector: str) -> str:
    return " ".join(page.locator(selector).inner_text().split())


def _system_rows(url: str) -> list:
    """Main's System rows as ``(system_type, text)``: what the owner reads as the host speaking. Rows a work card
    draws (web/modules/chat_preview.js ``CARD_ROWS``: a task's summary and evidence, a quiz, a card) are not one, nor
    a work's progress rows (``is_progress``, that filter's other half: a task's checkpoints fold into its card)."""
    return [(row.get("system_type"), row.get("text", "")) for row in _get(url, "/api/chat/history?limit=100")["messages"]
            if (row.get("role") == "system" or row.get("system_type")) and not row.get("is_progress")
            and row.get("system_type") not in {"task_evidence", "task_summary", "quiz", "card"}]


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
    """Accounts → Models on a providerless first setup: the compatible endpoint, Main and Light on its model; then
    Access, and the Consciousness step."""
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
    _next(page, 2)  # Models → Access → Consciousness
    page.wait_for_selector(STEP, timeout=30_000)


def _next(page, steps: int) -> None:
    for _ in range(steps):
        title = page.locator(".step-title").inner_text()
        page.locator("#next-btn:not([disabled])").click(timeout=60_000)
        page.wait_for_function("t => document.querySelector('.step-title')?.textContent !== t", arg=title, timeout=30_000)


def _meetings(data_dir: pathlib.Path) -> list:
    """The rows ``supervisor.first_meeting`` wrote to ``logs/events.jsonl``: one a start that tried."""
    events = data_dir / "logs" / "events.jsonl"
    lines = events.read_text(encoding="utf-8").splitlines() if events.exists() else []
    return [row for row in map(json.loads, filter(None, lines)) if row.get("type") == "first_meeting"]


def _settled(url: str, task_id: str, timeout: float = 240.0) -> str:
    """The task's status once it is terminal (``task_results._TRULY_TERMINAL_STATUSES``): a stop before that holds the
    work for Resume, and the restart's notice says so in Main."""
    deadline, status = time.time() + timeout, None
    while time.time() < deadline:
        try:
            status = _get(url, f"/api/tasks/{task_id}").get("status")
        except Exception as exc:  # not readable yet
            status = f"unreadable: {exc}"
        if status in {"completed", "failed", "cancelled", "rejected_duplicate"}:
            return status
        time.sleep(1)
    raise AssertionError(f"task {task_id} stayed {status!r}")


def _browser(pw, engine="webkit", theme="light", size=(980, 752)):
    browser = getattr(pw, engine).launch(headless=True)
    context = browser.new_context(viewport={"width": size[0], "height": size[1]}, color_scheme=theme, device_scale_factor=2)
    return browser, context.new_page()


def test_a_first_setup_is_born_on_with_no_command_and_main_holds_no_system_row(setup_server, tmp_path):
    """A first setup without a provider: the step starts on, the completion carries it, the page sends no command and
    the supervisor the save starts mints a state born on, which admits the one first meeting Start opens, in Main, and
    it completes on its own. After a first message and a restart Main holds no System row, the state still shows on
    and no second meeting opened; a later ordinary Stop wins and survives a restart."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import sync_playwright

    srv, out = setup_server, _shots(tmp_path)
    # Settings present (the offline skill sources), setup never completed, no provider, no state. A document that
    # predates setup reads the optional task bounds as their legacy finite values and is told so once
    # (upgrade_notices.optional_bound_facts); a fresh install writes no document before setup, so its bounds are
    # seeded as the unlimited a fresh install ships with.
    srv.seed({**OFFLINE, "OUROBOROS_MAX_ROUNDS": "unlimited", "OUROBOROS_TASK_ABS_CEILING_SEC": "unlimited"})
    srv.start(supervisor=False)
    assert not (srv.data_dir / "state" / "state.json").exists(), "no state before the setup save"
    with sync_playwright() as pw:
        browser, page = _browser(pw)
        try:
            seen = _record(page)
            page.goto(srv.url + "/onboarding", wait_until="networkidle", timeout=60_000)
            _fresh_accounts_and_models(page, srv.llm_url)
            assert page.locator(".step-title").inner_text() == "Consciousness"
            assert _text(page, ".step-copy") == FIRST_RUN["meaning"]
            assert _text(page, STEP) == STEP_TEXT
            assert page.locator('[data-consciousness-choice="on"]').get_attribute("aria-pressed") == "true"
            page.screenshot(path=str(out / "first-setup-consciousness-step-webkit-light.png"), full_page=True)
            order = step_order(page)  # Consciousness → (Skills, while the starter list never ran) → Budget → Summary
            _next(page, order.index("summary") - order.index("consciousness"))
            row = page.locator(".summary-kv").filter(has=page.get_by_text("Consciousness", exact=True))
            assert " ".join(row.inner_text().split()) == f"Consciousness On · {FIRST_RUN['on']}"
            seen["errors"].watch(page)  # Finish hands off to the app: leaving may cut off the page's own reads
            _start(page, seen["completions"])  # a click while the preview updates sends nothing; it is pressed again
            page.wait_for_url(srv.url + "/", timeout=240_000)
            assert seen["commands"] == [], "the builtin page sends no /bg"
            assert len(seen["completions"]) == 1 and seen["completions"][0]["consciousness"] is True
            assert not seen["errors"].unexpected(), seen["errors"].report()
            _wait_control(srv.url, True)
            page.fill("#chat-input", "Respond with exactly OK")  # the owner's first message binds the owner
            page.click("#chat-send")
            page.locator("#chat-messages .chat-bubble.assistant:not(.typing-bubble)").first.wait_for(timeout=120_000)
        finally:
            browser.close()
    saved = json.loads((srv.data_dir / "settings.json").read_text(encoding="utf-8"))
    assert saved.get("OUROBOROS_ONBOARDING_COMPLETED_AT") and saved.get("OUROBOROS_ONBOARDING_CONSCIOUSNESS") == "on"
    [meeting] = _meetings(srv.data_dir)  # Start opened one first meeting, in Main, and it ends on its own
    assert (meeting["outcome"], meeting["chat_id"]) == ("admitted", 1)
    assert _settled(srv.url, meeting["task_id"]) == "completed"
    srv.stop()
    srv.start(supervisor=True)
    assert _control(srv.url) is True, "born on, and still on after a restart"
    assert _meetings(srv.data_dir) == [meeting], "the restart opened no second meeting"
    assert _system_rows(srv.url) == [], "no System row in Main after setup, a first message and a restart"
    with sync_playwright() as pw:
        browser, page = _browser(pw)
        try:
            page.goto(srv.url + "/", wait_until="networkidle", timeout=60_000)
            page.locator("#chat-messages .chat-bubble.assistant:not(.typing-bubble)").first.wait_for(timeout=60_000)
            page.screenshot(path=str(out / "first-setup-main-after-restart-webkit-light.png"))
            assert page.locator("#chat-messages .chat-bubble.system").count() == 0
        finally:
            browser.close()
    assert _system_rows(srv.url) == [], "nor once the restarted page has read its history"
    # A later ordinary Stop wins and a restart keeps it; its reply is the System row the setup no longer writes.
    assert _command(srv.url, "/bg stop") == 200
    _wait_control(srv.url, False)
    deadline = time.time() + 30
    while not _system_rows(srv.url) and time.time() < deadline:
        time.sleep(0.5)
    assert _system_rows(srv.url), "the same reader sees a command's System row"
    srv.stop()
    srv.start(supervisor=True)
    time.sleep(3)
    assert _control(srv.url) is False


def test_an_older_install_with_a_state_shows_no_step_and_stays_as_its_state_says(setup_server, tmp_path):
    """Settings present, a provider ready, setup never completed, and a state that says off (a markerless older
    install): the wizard has no Consciousness step, Finish sends no field and no command, and consciousness stays
    off; Dashboard › Consciousness is where it changes."""
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import sync_playwright

    srv = setup_server
    srv.seed({**OFFLINE, **_provider(srv.llm_url)})
    srv.start(supervisor=True)
    assert _control(srv.url) is False
    with sync_playwright() as pw:
        browser, page = _browser(pw)
        try:
            seen = _record(page)
            page.goto(srv.url + "/onboarding?builtin=1", wait_until="networkidle", timeout=60_000)
            _to_summary(page, pro=False)
            assert page.locator(".wizard-step").count() == 5 and page.locator(STEP).count() == 0
            assert page.locator(".summary-kv").filter(has=page.get_by_text("Consciousness", exact=True)).count() == 0
            seen["errors"].watch(page)
            _start(page, seen["completions"])  # a click while the preview updates sends nothing; it is pressed again
            page.wait_for_url(srv.url + "/", timeout=120_000)
            assert seen["commands"] == [], "nothing is sent"
            assert len(seen["completions"]) == 1 and "consciousness" not in seen["completions"][0]
            time.sleep(3)
            assert _control(srv.url) is False
            assert _meetings(srv.data_dir) == [], "a setup finished over an existing state opens no first meeting"
            assert not seen["errors"].unexpected(), seen["errors"].report()
        finally:
            browser.close()
