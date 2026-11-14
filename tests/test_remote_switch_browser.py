"""The Remote capsule in Main's bar, on this checkout's own running server, in WebKit (what the Mac window shows).

The candidate's ``server.py`` serves the SPA over a temporary data folder (``candidate_server``): ``/api/state`` and
``/api/settings`` are the real handlers, so what the capsule says comes from the server's projection
(``remote_placement.state_view``) and every press is the real settings save. The engine is attach-only on an empty
home (``OUROBOROS_CLAUDEXOR_ATTACH_HOME``): nothing is started, installed or contacted, and no host, account, token or
model is reached. Frames: the owner's window (1512 x 949, scale 2) and a phone (390 x 844), light and dark, under
``OUROBOROS_UI_EVIDENCE_DIR`` (else beside the data folder).
"""
from __future__ import annotations

import json
import os
import pathlib
import socket
import subprocess
import sys
import time
import urllib.request
from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.ui_browser]
REPO = pathlib.Path(__file__).resolve().parents[1]
OWNER = {"width": 1512, "height": 949}
PHONE = {"width": 390, "height": 844}
KEY = "OUROBOROS_REMOTE_EXECUTION"
LOCAL = "claudexor::codex=gpt-6"
VPS = {"name": "vps", "label": "VDSina", "ssh_host": "vds", "remote_root": "/home/ouroboros/runs"}
LAB = {"name": "lab", "label": "Lab box", "ssh_host": "lab"}
CAPSULE = ".chat-page-header .rx-capsule"


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _answers(url, path, check, timeout_sec=60):
    deadline, last = time.time() + timeout_sec, ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{url}{path}", timeout=2) as resp:  # noqa: S310 - local test server
                if check(json.loads(resp.read().decode("utf-8"))):
                    return
        except Exception as exc:  # not up yet
            last = str(exc)
        time.sleep(0.25)
    raise RuntimeError(f"{path} did not become ready: {last}")


def _browser_missing(exc) -> bool:
    """Playwright's report of an engine that was never installed."""
    return "Executable doesn't exist" in str(exc) or "playwright install" in str(exc).lower()


@pytest.fixture
def candidate_server(tmp_path, ui_candidate_checkout):
    """This checkout's own ``server.py`` from the session's origin-proof copy (``tests/candidate_checkout.py``) in a
    contained process group over a temporary data folder, the model a local mock (``tests/fixtures_mock_llm.py``) and
    the engine attach-only on an empty home. ``seed(project="", **values)`` writes the document (a value ``None``
    removes its key) and makes a Project while the server is stopped, then starts it again; teardown proves the
    process group gone."""
    from ouroboros.process_containment import ProcessContainer
    from ouroboros.test_environment import isolated_environment, owned_process_temp
    from tests.candidate_checkout import (assert_served_candidate, case_checkout, require_candidate_interpreter,
                                          verify_checkout)
    from tests.fixtures_mock_llm import MockLLMServer

    with case_checkout(REPO, ui_candidate_checkout) as candidate, MockLLMServer() as llm:
        data_dir, home, port = tmp_path / "data", tmp_path / "attach-only-engine", _free_port()
        data_dir.mkdir()
        home.mkdir()
        settings = dict.fromkeys(("OUROBOROS_MODEL", "OUROBOROS_MODEL_LIGHT", "OUROBOROS_MODEL_FALLBACKS"),
                                 "openai-compatible::mock-model")
        settings.update(OPENAI_COMPATIBLE_API_KEY="ui-key", OPENAI_COMPATIBLE_BASE_URL=llm.base_url,
                        OUROBOROS_MAX_WORKERS=1, OUROBOROS_RUNTIME_MODE="light",
                        OUROBOROS_STARTER_SKILLS='["builtin/telegram"]', OUROBOROS_STARTER_SKILLS_TURN_ON=False)
        (data_dir / "settings.json").write_text(json.dumps(settings), encoding="utf-8")
        env = isolated_environment(tmp_path, candidate.path, process_temp=owned_process_temp())
        env.update(OUROBOROS_SERVER_HOST="127.0.0.1", OUROBOROS_SERVER_PORT=str(port),
                   OUROBOROS_HOST_SERVICE_PORT=str(_free_port()), OUROBOROS_NETWORK_PASSWORD="ui-password",
                   OUROBOROS_CLAUDEXOR_ATTACH_HOME=str(home))  # never a started, installed or contacted engine
        require_candidate_interpreter(sys.executable, env, candidate.path)
        url, running = f"http://127.0.0.1:{port}", {}

        def stop():
            proc, container = running.pop("proc", None), running.pop("container", None)
            if container is None:
                return
            try:
                if proc is not None and proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        pass  # the container reaps what is left
            finally:
                try:
                    error = container.reap()
                finally:
                    container.close()
                if error:
                    raise RuntimeError(f"server process cleanup failed: {error}")
            candidate.release()  # only a proven teardown releases the copy

        def start():
            verify_checkout(candidate.path, candidate)
            running["container"] = ProcessContainer()
            candidate.hold()
            with (tmp_path / "server-stdio.log").open("ab") as log:
                running["proc"] = running["container"].spawn([sys.executable, "server.py"], cwd=candidate.path, env=env,
                                                             stdout=log, stderr=subprocess.STDOUT)
            _answers(url, "/api/health", lambda _body: True)
            _answers(url, "/api/state", lambda body: body.get("supervisor_ready") is True)
            assert_served_candidate(url, candidate.path, data_dir, running["proc"].pid, candidate)

        def seed(project="", **values):
            stop()
            if project:  # a Project room, made by the registry's own writer while nothing else writes
                from ouroboros.projects_registry import create_project

                create_project(data_dir, project, name="Remote demo")
            path = data_dir / "settings.json"
            document = {**json.loads(path.read_text(encoding="utf-8")), **values}
            path.write_text(json.dumps({key: value for key, value in document.items() if value is not None}),
                            encoding="utf-8")
            start()

        start()
        try:
            yield SimpleNamespace(url=url, data_dir=data_dir, seed=seed)
        finally:
            stop()


def _state(url):
    with urllib.request.urlopen(f"{url}/api/state", timeout=10) as resp:  # noqa: S310 - local test server
        return json.loads(resp.read().decode("utf-8"))


def _disk(server):
    return json.loads((server.data_dir / "settings.json").read_text(encoding="utf-8")).get(KEY)


def _panic_stays(page, bar):
    """Beside Remote, Panic stays whole, in view and clear of the capsule, whose computer stays drawn: the capsule's
    word yields first where the bar is short (DESIGN §10 "Page header", room_work.js ``fitWord``)."""
    facts = page.locator(bar).evaluate("""(bar) => {
        const panic = bar.querySelector('[data-chat-command="panic"]'), capsule = bar.querySelector('.rx-capsule');
        const p = panic.getBoundingClientRect(), c = capsule.getBoundingClientRect();
        const g = capsule.querySelector('.rx-glyph').getBoundingClientRect();
        return { whole: p.width > 0 && panic.scrollWidth <= panic.clientWidth + 1,
                 inView: p.left >= -0.5 && p.right <= innerWidth + 0.5,
                 apart: c.right <= p.left + 0.5 || c.left >= p.right - 0.5, glyph: g.width > 0 && g.height > 0 };
    }""")
    assert facts == {"whole": True, "inView": True, "apart": True, "glyph": True}, (bar, facts)


def _frame(page, folder, name, *, sizes=(OWNER, PHONE)):
    """The first screen at each size and scheme, and a 2x crop of the bar; nothing wider than the window."""
    for size in sizes:
        page.set_viewport_size(size)
        for scheme in ("light", "dark"):
            page.emulate_media(color_scheme=scheme)
            page.wait_for_timeout(250)
            assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth"), (name, size)
            _panic_stays(page, ".chat-page-header")
            tag = f"{name}-{scheme}-{size['width']}"
            page.screenshot(path=str(folder / f"{tag}.png"))
            page.locator(".chat-page-header").screenshot(path=str(folder / f"{tag}-bar.png"))
    page.emulate_media(color_scheme="light")
    page.set_viewport_size(OWNER)


def _open(page, url):
    page.goto(url, wait_until="domcontentloaded", timeout=60_000)
    page.locator(CAPSULE).wait_for(state="visible", timeout=60_000)
    return page.locator(CAPSULE)


def test_remote_capsule_setup_toggle_choice_reload_and_refusal(candidate_server):
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import Error as PlaywrightError
    from playwright.sync_api import expect, sync_playwright

    server, url = candidate_server, candidate_server.url
    folder = pathlib.Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR") or server.data_dir.parent) / "remote_switch"
    folder.mkdir(parents=True, exist_ok=True)
    server.seed(OUROBOROS_MODEL=LOCAL, OUROBOROS_ENGINE_CONNECTIONS=[])
    try:
        with sync_playwright() as pw:
            browser = pw.webkit.launch(headless=True)
            page = browser.new_context(viewport=OWNER, device_scale_factor=2, color_scheme="light").new_page()
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            try:
                # 1. No engine saved: Off, and the press opens Settings › Accounts and keys › Remote engines.
                capsule = _open(page, url)
                toggle, more = capsule.locator(".rx-toggle"), capsule.locator(".rx-more")
                expect(capsule.locator(".rx-word")).to_have_text("Off")
                assert toggle.get_attribute("aria-pressed") == "false"
                assert "no remote engine is set up" in toggle.get_attribute("aria-label")
                assert more.is_hidden()
                assert capsule.evaluate("(el) => el.previousElementSibling?.classList.contains('cx-chip')"), "after the chip"
                _frame(page, folder, "01-unconfigured")
                toggle.click()
                page.locator('#page-settings section[data-setting-row="OUROBOROS_ENGINE_CONNECTIONS"]').wait_for(
                    state="visible", timeout=30_000)
                page.screenshot(path=str(folder / "02-unconfigured-press-opens-remote-engines.png"))
                assert _disk(server) is None, "the setup door writes nothing"

                # 2. One engine: one press (keyboard) turns it on; the server shows it; a reload keeps it.
                server.seed(project="remote-demo", OUROBOROS_ENGINE_CONNECTIONS=[VPS])
                capsule = _open(page, url)
                toggle, more = capsule.locator(".rx-toggle"), capsule.locator(".rx-more")
                assert "Press to run Main and new agents on VDSina" in toggle.get_attribute("aria-label")
                toggle.focus()
                page.keyboard.press("Enter")
                expect(capsule).to_have_attribute("data-on", "true", timeout=30_000)
                expect(capsule.locator(".rx-word")).to_have_text("VDSina")
                assert toggle.get_attribute("aria-pressed") == "true"
                assert _disk(server) == '{"engine":"vps","on":true}'
                state = _state(url)
                assert state["main_route"] == {"model": "claudexor@vps::codex=gpt-6", "use_local": False}
                assert state["remote_execution"]["main"]["scope"] == "follows"
                assert capsule.get_attribute("data-mixed") == "false"
                page.locator(".chat-page-header").screenshot(path=str(folder / "03-keyboard-focus-light-1512-bar.png"))
                toggle.evaluate("(el) => el.blur()")
                assert page.locator(".project-panel-bar .rx-capsule").count() == 1, "a Project's bar has its own"
                _frame(page, folder, "03-on-main-and-agents")
                more.click()
                menu = page.locator(".rx-menu")
                expect(menu).to_be_visible()
                notes = menu.inner_text()
                assert "Main and new agents run on VDSina." in notes and "Emergency Stop closes the tunnel" in notes
                assert "plan turns, in-place edits and folders that are not Git projects are refused" in notes
                assert "A press moves Main at its next call and the agents started after it" in notes
                assert menu.locator('[role="menuitemradio"][aria-checked="true"]').inner_text().startswith("VDSina")
                for scheme in ("light", "dark"):
                    page.emulate_media(color_scheme=scheme)
                    page.wait_for_timeout(200)
                    page.screenshot(path=str(folder / f"04-chooser-{scheme}-1512.png"))
                page.emulate_media(color_scheme="light")
                page.keyboard.press("Escape")
                expect(menu).to_have_count(0)
                capsule = _open(page, url)  # a reload shows what the server keeps
                expect(capsule).to_have_attribute("data-on", "true", timeout=30_000)
                # A Project's bar carries the same switch (one preference for the installation).
                page.locator('.nav-project-row[data-project-id="remote-demo"]').evaluate("el => el.click()")
                page.locator('[id="panel-pchat-remote-demo"]:not([hidden])').wait_for(timeout=30_000)
                in_project = page.locator(".project-panel-bar .rx-capsule")
                expect(in_project).to_have_attribute("data-on", "true", timeout=30_000)
                expect(in_project.locator(".rx-word")).to_have_text("VDSina")
                for size in (OWNER, PHONE):
                    page.set_viewport_size(size)
                    for scheme in ("light", "dark"):
                        page.emulate_media(color_scheme=scheme)
                        page.wait_for_timeout(250)
                        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth"), size
                        _panic_stays(page, ".project-panel-bar")
                        page.screenshot(path=str(folder / f"04b-project-room-{scheme}-{size['width']}.png"))
                page.emulate_media(color_scheme="light")
                page.set_viewport_size(OWNER)
                page.locator("#project-panel-close").click()
                page.locator('[id="panel-pchat-remote-demo"]').wait_for(state="hidden", timeout=30_000)
                capsule.locator(".rx-toggle").click()
                expect(capsule).to_have_attribute("data-on", "false", timeout=30_000)
                assert _disk(server) == '{"engine":"vps","on":false}', "Off keeps the remembered engine"
                assert _state(url)["main_route"] == {"model": LOCAL, "use_local": False}

                # 3. Several engines and no choice made: the press asks; the keyboard picks Lab box.
                server.seed(OUROBOROS_ENGINE_CONNECTIONS=[VPS, LAB], **{KEY: None})
                capsule = _open(page, url)
                assert "Press to choose a remote engine" in capsule.locator(".rx-toggle").get_attribute("aria-label")
                capsule.locator(".rx-toggle").click()
                menu = page.locator(".rx-menu")
                expect(menu).to_be_visible()
                assert _disk(server) is None, "never the first engine by default"
                page.screenshot(path=str(folder / "05-several-engines-choose-light-1512.png"))
                page.keyboard.press("ArrowDown")
                page.keyboard.press("ArrowDown")
                page.keyboard.press("Enter")
                expect(capsule.locator(".rx-word")).to_have_text("Lab box", timeout=30_000)
                assert _disk(server) == '{"engine":"lab","on":true}'

                # 4. On while Main uses an API key: the capsule says it moves agents only.
                server.seed(OUROBOROS_MODEL="openai-compatible::mock-model")
                capsule = _open(page, url)
                expect(capsule).to_have_attribute("data-mixed", "true", timeout=30_000)
                expect(capsule.locator(".rx-scope")).to_have_text("· agents")
                assert "Main uses an API key and stays as it is" in capsule.locator(".rx-toggle").get_attribute("aria-label")
                _frame(page, folder, "06-on-agents-only-api-main", sizes=(OWNER,))

                # 5. The target switched off in Settings while On: the owner's amber dot and why.
                server.seed(OUROBOROS_MODEL=LOCAL, OUROBOROS_ENGINE_CONNECTIONS=[VPS, {**LAB, "enabled": False}])
                capsule = _open(page, url)
                mark = capsule.locator(".rx-mark")
                expect(mark).to_have_attribute("data-hue", "owner", timeout=30_000)
                assert "switched off in Settings: new work for it is refused" in capsule.locator(".rx-toggle").get_attribute("aria-label")
                _frame(page, folder, "07-target-switched-off", sizes=(OWNER,))
                capsule.locator(".rx-toggle").click()  # Off is the way back; nothing stops or disconnects
                expect(capsule).to_have_attribute("data-on", "false", timeout=30_000)
                assert _disk(server) == '{"engine":"lab","on":false}'

                # 6. A save the server refuses: the page still believed Lab box takes new work (its state held);
                # the real save refuses by name, the toast says so and the capsule never shows On.
                server.seed(OUROBOROS_ENGINE_CONNECTIONS=[VPS, LAB])
                capsule = _open(page, url)
                held = json.dumps(_state(url))
                page.route("**/api/state**", lambda route: route.fulfill(content_type="application/json", body=held))
                server_path = server.data_dir / "settings.json"
                document = json.loads(server_path.read_text(encoding="utf-8"))
                document["OUROBOROS_ENGINE_CONNECTIONS"] = [VPS, {**LAB, "enabled": False}]
                request = urllib.request.Request(f"{url}/api/settings", method="POST", data=json.dumps(
                    {"OUROBOROS_ENGINE_CONNECTIONS": document["OUROBOROS_ENGINE_CONNECTIONS"]}).encode("utf-8"),
                    headers={"Content-Type": "application/json", "X-Ouroboros-Surface": "app", "X-Ouroboros-Actor": "owner"})
                with urllib.request.urlopen(request, timeout=30) as resp:  # noqa: S310 - local test server
                    assert resp.status == 200
                capsule.locator(".rx-toggle").click()
                toast = page.locator(".toast", has_text="Lab box is switched off in Settings")
                expect(toast).to_be_visible(timeout=30_000)
                expect(capsule).to_have_attribute("data-on", "false")
                expect(capsule.locator(".rx-word")).to_have_text("Off")
                assert _disk(server) == '{"engine":"lab","on":false}'
                page.wait_for_timeout(400)  # past the toast's entrance fade
                page.screenshot(path=str(folder / "08-refused-save-light-1512.png"))
                page.unroute("**/api/state**")

                # 7. On with every engine since removed: one press is still the way back, never the setup.
                server.seed(OUROBOROS_ENGINE_CONNECTIONS=[], **{KEY: '{"engine":"vps","on":true}'})
                capsule = _open(page, url)
                expect(capsule).to_have_attribute("data-on", "true", timeout=30_000)
                expect(capsule.locator(".rx-mark")).to_have_attribute("data-hue", "owner")
                assert capsule.locator(".rx-more").is_visible(), "the chooser stays reachable while On"
                assert "no longer saved" in capsule.locator(".rx-toggle").get_attribute("aria-label")
                _frame(page, folder, "09-on-every-engine-removed", sizes=(OWNER,))
                capsule.locator(".rx-toggle").click()
                expect(capsule).to_have_attribute("data-on", "false", timeout=30_000)
                assert _disk(server) == '{"engine":"vps","on":false}'
                assert page.locator("#page-settings.active").count() == 0, "the press opened no setup"
                assert not errors, errors
            finally:
                browser.close()
    except PlaywrightError as exc:
        if _browser_missing(exc):
            pytest.skip(str(exc))
        raise


def test_remote_capsule_leaves_panic_reachable_beside_long_room_states(candidate_server):
    """Real candidate header with a visible Remote capsule, including mixed scope; only the room status is supplied
    by a fixture. The ordinary room-status consumer paints and fits the word, not a direct DOM edit."""
    from playwright.sync_api import expect, sync_playwright

    server = candidate_server
    server.seed(OUROBOROS_MODEL="openai-compatible::mock-model", OUROBOROS_ENGINE_CONNECTIONS=[VPS],
                **{KEY: '{"engine":"vps","on":true}'})
    state = _state(server.url)
    state["rooms"] = {"1": {"activity": "waiting", "transition": "none", "call": "",
                             "turn": [{"kind": "question", "task_id": "fixture", "quiz_id": "fixture-question"}],
                             "amber": True, "mark": "still", "confirmed": True, "works": []}}
    folder = pathlib.Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR") or server.data_dir.parent) / "remote_switch"
    folder.mkdir(parents=True, exist_ok=True)
    observations = []
    with sync_playwright() as pw:
        browser = pw.webkit.launch(headless=True)
        page = browser.new_context(viewport=OWNER, device_scale_factor=2).new_page()
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.route("**/api/state**", lambda route: route.fulfill(content_type="application/json", body=json.dumps(state)))
        try:
            capsule = _open(page, server.url)
            expect(capsule).to_have_attribute("data-mixed", "true")
            expect(page.locator("#chat-status")).to_have_text("Waiting for your answer", timeout=30_000)
            for width in (375, 390, 768, 1512):
                page.set_viewport_size({"width": width, "height": 949 if width > 390 else 844})
                for scheme in ("light", "dark"):
                    page.emulate_media(color_scheme=scheme)
                    page.wait_for_timeout(300)
                    _panic_stays(page, ".chat-page-header")
                    expect(capsule.locator(".rx-scope")).to_be_visible()
                    facts = page.locator(".chat-page-header").evaluate("""bar => {
                        const panic = bar.querySelector('[data-chat-command="panic"]');
                        const r = panic.getBoundingClientRect();
                        const hit = document.elementFromPoint(r.x+r.width/2, r.y+r.height/2);
                        return {overflow: document.documentElement.scrollWidth > innerWidth,
                                panicHit: hit === panic || panic.contains(hit),
                                status: bar.querySelector('#chat-status').textContent,
                                remote: bar.querySelector('.rx-capsule').getAttribute('data-mixed')};
                    }""")
                    observations.append({"width": width, "scheme": scheme, **facts})
                    assert facts["panicHit"] and not facts["overflow"], facts
                    page.screenshot(path=str(folder / f"10-long-status-mixed-{scheme}-{width}.png"))
            assert not errors, errors
        finally:
            (folder / "long-status-observations.json").write_text(json.dumps(observations, indent=2), encoding="utf-8")
            browser.close()
