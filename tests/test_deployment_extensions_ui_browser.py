"""The real SPA and a real worker under a policy that requires extensions, the same server unmanaged beside it.

Unmanaged, a skill is the employee's to turn off or delete and a chat turn runs its body. Required by the file, the
same package reads on, its card says so with no Off, removal or update, the extensions that do not run are said
above the list and a fixed setting shows no value. Under ``hold`` a chat turn the worker cannot serve ends with the
typed held notice before any model call, and the worker lives on; under ``warn`` the turn runs; once every
requirement runs, ``hold`` holds nothing. A turn that runs is answered: the fixture's model replies in the framing
the request asks for (an SSE stream, as today's client asks), its words reach the chat and the turn's journal row
says completed. With the tracker running, its own widget draws what each process's transport observed.
Frames go to ``OUROBOROS_UI_EVIDENCE_DIR`` for a look.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import shutil
import textwrap
import time
import urllib.request
from datetime import datetime, timezone

import pytest

from tests.settings_browser_helpers import open_settings, settings_page
from tests.test_ui_smoke_playwright import direct_server_with_data  # noqa: F401

REPO = pathlib.Path(__file__).resolve().parents[1]
NOTES = "org_notes"
DSN = "http://fixturekey@127.0.0.1:9/1"  # never contacted: the tracker is not granted in this server
ANSWER = "Answered by the fixture model, reply {}."  # numbered, so an earlier turn's answer never stands in


def _seed(data_dir: pathlib.Path) -> dict:
    """A small reviewed extension, saved Off, and the error_tracker example, reviewed and not granted."""
    from ouroboros.skill_loader import SkillReviewState, load_skill, save_enabled, save_review_state

    notes = data_dir / "skills" / "external" / NOTES
    notes.mkdir(parents=True, exist_ok=True)
    (notes / "SKILL.md").write_text(textwrap.dedent(f"""\
        ---
        name: {NOTES}
        description: Organization notes for the team, a fixture extension.
        version: 1.0.0
        type: extension
        entry: plugin.py
        plugin_api: "2.0"
        permissions: [tool]
        ---
        # Organization notes
        """), encoding="utf-8")
    (notes / "plugin.py").write_text("def register(api):\n    api.register_tool('note', lambda ctx: 'noted', "
                                     "description='note', schema={})\n", encoding="utf-8")
    tracker = data_dir / "skills" / "external" / "error_tracker"
    shutil.copytree(REPO / "docs" / "examples" / "error_tracker", tracker)
    pins = {}
    for skill_dir in (notes, tracker):
        loaded = load_skill(skill_dir, data_dir)
        save_review_state(data_dir, loaded.name, SkillReviewState(status="pass", content_hash=loaded.content_hash))
        save_enabled(data_dir, loaded.name, False, actor="owner_ui")
        pins[loaded.name] = {"name": loaded.name, "path": str(loaded.skill_dir), "content_hash": loaded.content_hash}
    pins["error_tracker"]["settings"] = {"SENTRY_DSN": DSN}
    return pins


def _policy(path: pathlib.Path, mode: str, rows: list) -> None:
    path.write_text(json.dumps({"version": 1, "extensions": {"when_unmet": mode, "required": rows}}), encoding="utf-8")


def _settled(page) -> None:
    """Two frames after the last read: what the screen shows, not a transition."""
    page.evaluate("() => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)))")


def _shot(page, evidence, name, locator=None):
    _settled(page)
    (locator or page).screenshot(path=str(evidence / f"{name}.png"))


def _skills(page, url, theme, width):
    page.set_viewport_size({"width": width, "height": 1000 if width > 500 else 844})
    page.add_init_script(f"localStorage.setItem('ouroboros.theme', {json.dumps(theme)})")
    page.goto(url + "/#skills", wait_until="domcontentloaded")
    page.locator(f'.skills-card[data-skill="{NOTES}"]').wait_for(state="visible", timeout=60_000)
    page.wait_for_function("() => !(document.querySelector('#skills-status')?.textContent || '').includes('Loading')")


_HELD = "() => [...document.querySelectorAll('.chat-bubble.system .message')].filter((n) => n.textContent.includes('Not started')).length"


def _model_answers(monkeypatch) -> list:
    """The fixture's model answers every completion in the framing its request asks for: an SSE stream of chunks,
    a finish reason, usage and ``[DONE]`` when ``stream`` is true (today's client), one JSON body otherwise. Returns
    the requests it took as ``(UTC time, streamed, the answer's words)``."""
    import threading

    from tests import fixtures_mock_llm

    calls, lock = [], threading.Lock()

    def answer(handler):
        if handler.path != "/v1/chat/completions":
            handler.send_error(404)
            return
        payload = json.loads(handler.rfile.read(int(handler.headers.get("Content-Length", 0))) or b"{}")
        streamed = payload.get("stream") is True
        with lock:
            text = ANSWER.format(len(calls) + 1)
            calls.append((datetime.now(timezone.utc).isoformat(), streamed, text))
        message, usage = {"role": "assistant", "content": text}, {"prompt_tokens": 1, "completion_tokens": 1,
                                                                   "total_tokens": 2}
        if streamed:
            chunks = [{"id": "fixture", "object": "chat.completion.chunk",
                       "choices": [{"index": 0, "delta": message, "finish_reason": None}]},
                      {"id": "fixture", "object": "chat.completion.chunk",
                       "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                      {"id": "fixture", "object": "chat.completion.chunk", "choices": [], "usage": usage}]
            body = ("".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n").encode()
        else:
            body = json.dumps({"id": "fixture", "object": "chat.completion", "usage": usage,
                               "choices": [{"index": 0, "finish_reason": "stop", "message": message}]}).encode()
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream" if streamed else "application/json")
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)

    monkeypatch.setattr(fixtures_mock_llm._Handler, "do_POST", answer)
    return calls


def _ask(page, url, text):
    """Send ``text`` from Main; answers when it was sent and how many held notices the page showed before."""
    page.goto(url, wait_until="domcontentloaded", timeout=30_000)
    page.wait_for_selector("#chat-input", timeout=30_000)
    before = (datetime.now(timezone.utc).isoformat(), page.evaluate(_HELD))
    page.fill("#chat-input", text)
    page.click("#chat-send")
    return before


def _turn_events(data_dir: pathlib.Path, since: str) -> list:
    path = data_dir / "logs" / "events.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    return [row for row in (json.loads(line) for line in lines if line.strip()) if str(row.get("ts") or "") >= since]


def _answered(page, data_dir: pathlib.Path, calls: list, before) -> None:
    """The turn ran to its answer: the model was asked in a stream, the turn's journal row says completed, one of
    this turn's own numbered answers is an assistant reply in the chat, and nothing was held."""
    deadline = time.monotonic() + 90
    while not (asked := [text for at, streamed, text in calls if at >= before[0] and streamed]):
        assert time.monotonic() < deadline, f"the model was never asked in a stream: {calls}"
        time.sleep(0.25)
    deadline = time.monotonic() + 60
    while not (done := [row for row in _turn_events(data_dir, before[0]) if row.get("type") == "task_done"]):
        assert time.monotonic() < deadline, "the answered turn has no task_done row"
        time.sleep(0.25)
    assert all(row.get("status") == "completed" for row in done), done
    page.wait_for_function(
        "(texts) => [...document.querySelectorAll('.chat-bubble.assistant:not(.typing-bubble)')]"
        ".some((bubble) => texts.some((text) => bubble.textContent.includes(text)))", arg=asked, timeout=60_000)
    assert not [row for row in _turn_events(data_dir, before[0]) if row.get("reason_code") == "deployment_requirement_unmet"]
    assert not [row for row in _turn_events(data_dir, before[0]) if row.get("type") == "llm_api_error"]


def _rows(path: pathlib.Path, kind: str) -> list:
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    return [row for row in (json.loads(line) for line in lines if line.strip()) if row.get("type") == kind]


def _post(url, path, body):
    request = urllib.request.Request(url + path, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=60) as answer:
            return answer.status, json.loads(answer.read() or b"{}")
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read() or b"{}")


@pytest.mark.ui_browser
def test_required_extensions_in_skills_settings_and_the_worker(direct_server_with_data, tmp_path, monkeypatch):  # noqa: F811
    from playwright.sync_api import expect, sync_playwright

    calls = _model_answers(monkeypatch)
    fixture = direct_server_with_data
    data_dir, url = fixture["data_dir"], fixture["url"]
    evidence = pathlib.Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", str(tmp_path / "frames")))
    evidence.mkdir(parents=True, exist_ok=True)
    fixture["stop_server"]()
    pins = _seed(data_dir)
    enabled_file = data_dir / "state" / "skills" / NOTES / "enabled.json"
    saved_off = enabled_file.read_bytes()
    fixture["start_server"]()
    policy = tmp_path / "admin" / "deployment-policy.json"
    policy.parent.mkdir()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            # --- unmanaged: the employee's switch and removal; a chat turn is answered ---
            page = browser.new_page()
            _skills(page, url, "light", 1400)
            card = page.locator(f'.skills-card[data-skill="{NOTES}"]')
            assert card.locator(".skills-toggle").is_enabled() and not card.locator(".skills-toggle").is_checked()
            assert not page.locator("[data-skills-org-requirements]").is_visible()
            assert card.locator("[data-required-by-organization]").count() == 0
            _shot(page, evidence, "unmanaged-skills-light-1400", card)
            _answered(page, data_dir, calls, _ask(page, url, "Reply briefly."))
            _shot(page, evidence, "unmanaged-chat-light-1400")
            # Unmanaged, SENTRY_DSN is the owner's: a key absent from Settings is no request yet
            # (skill_loader.requested_core_setting_keys), so the owner adds it as a key of their own; the
            # tracker then asks for it on its row, which offers Replace and Remove, and Remove clears the value.
            own = open_settings(page, "accounts/key/custom")
            own.locator("[data-custom-secret-key]").fill("SENTRY_DSN")
            own.locator("[data-custom-secret-value]").fill("owner-typed-dsn")
            own.locator("[data-editor-save]").click()
            asked = settings_page(page, "accounts").locator('[data-setting-row="SENTRY_DSN"]')
            expect(asked).to_be_visible(timeout=30_000)
            expect(asked.locator("[data-setting-meta]")).to_contain_text("error_tracker", timeout=30_000)
            expect(asked.locator("[data-secret-state]")).to_have_text("Saved")
            expect(asked.locator("[data-secret-open]")).to_have_text("Replace")
            asked.locator("[data-secret-open]").click()
            asked.locator("[data-secret-remove]").click()
            page.locator("[data-confirm-ok]").click()
            # The row stays while the page holds its skills read; it now offers Add over an empty value.
            expect(asked.locator("[data-secret-open]")).to_have_text("Add", timeout=30_000)
            expect(asked.locator("[data-secret-state]")).to_have_text("")
            assert not json.loads((data_dir / "settings.json").read_text(encoding="utf-8")).get("SENTRY_DSN")
            page.close()

            # --- hold: the tracker is required but not granted, so this worker holds ordinary work ---
            fixture["stop_server"]()
            _policy(policy, "hold", [pins[NOTES], pins["error_tracker"]])
            fixture["env"]["OUROBOROS_DEPLOYMENT_POLICY"] = str(policy)
            fixture["start_server"]()
            for theme, width in (("light", 1400), ("dark", 1400), ("dark", 390)):
                page = browser.new_page()
                _skills(page, url, theme, width)
                assert page.evaluate("document.documentElement.dataset.theme") == theme
                notice = page.locator("[data-skills-org-requirements]")
                notice.wait_for(state="visible", timeout=30_000)
                text = notice.inner_text()
                assert "error_tracker" in text and "It needs your grant for SENTRY_DSN." in text, text
                assert NOTES not in text and "Ordinary work waits until it runs." in text
                card = page.locator(f'.skills-card[data-skill="{NOTES}"]')
                assert card.locator("[data-required-by-organization]").is_visible()
                toggle = card.locator(".skills-toggle")
                assert toggle.is_checked() and toggle.is_disabled(), "on by the organization, not the employee's switch"
                tracker = page.locator('.skills-card[data-skill="error_tracker"]')
                assert tracker.locator("[data-required-unmet]").is_visible()
                assert page.evaluate("document.querySelector('#page-skills').scrollWidth <= innerWidth + 1")
                _shot(page, evidence, f"managed-hold-skills-{theme}-{width}")
                _shot(page, evidence, f"managed-hold-tracker-{theme}-{width}", tracker)
                menu = card.locator(".skills-card-menu-dialog")  # the card's own menu items, as drawn
                assert menu.locator('[data-skill-action="delete"]').count() == 0
                assert menu.locator('[data-skill-action="open_detail"]').count() == 1
                if (theme, width) == ("light", 1400):
                    page.close()
                    page = browser.new_page(viewport={"width": 1400, "height": 1000})
                    page.add_init_script("localStorage.setItem('ouroboros.theme', 'light')")
                    page.goto(url + "/#settings", wait_until="domcontentloaded")
                    # The key the tracker asked for is a row of Accounts and keys; its value is the organization's.
                    listing = open_settings(page, "accounts", row="SENTRY_DSN", timeout=30_000)
                    row = listing.locator('[data-setting-row="SENTRY_DSN"]')
                    row.wait_for(state="visible", timeout=30_000)
                    expect(row.locator("[data-secret-state]")).to_have_text("Set by your organization for error_tracker")
                    assert row.locator("[data-secret-open]").is_hidden(), "no Add or Replace for a fixed value"
                    assert row.locator("[data-secret-field]").is_hidden(), "no field, Show or Remove for a fixed value"
                    _shot(page, evidence, "managed-hold-secret-light-1400", row)
                    before = _ask(page, url, "Reply briefly.")
                    page.wait_for_function(f"() => ({_HELD})() > {before[1]}", timeout=90_000)
                    messages = page.locator("#chat-messages").inner_text()
                    assert "error_tracker: It needs your grant for SENTRY_DSN. Nothing ran;" in messages, messages
                    page.locator("#typing-indicator").wait_for(state="hidden", timeout=30_000)
                    page.wait_for_timeout(100)
                    for _ in range(300):  # the terminal's journal row follows the notice the page shows
                        turn = _turn_events(data_dir, before[0])
                        done = [row.get("reason_code") for row in turn if row.get("type") == "task_done"]
                        if done:
                            break
                        page.wait_for_timeout(100)
                    assert done == ["deployment_requirement_unmet"], turn
                    assert not [row for row in turn if str(row.get("type") or "").startswith("llm_")], "no model call"
                    assert not [call for call in calls if call[0] >= before[0]], "the model was asked nothing"
                    _shot(page, evidence, "managed-hold-chat-light-1400")
                    (evidence / "managed-hold-chat.txt").write_text(messages, encoding="utf-8")
                    assert not _rows(data_dir / "logs" / "supervisor.jsonl", "worker_crash"), "a hold never crashes a worker"
                    held_rows = _rows(data_dir / "logs" / "events.jsonl", "deployment_requirement_unmet")
                    assert held_rows and all(DSN not in json.dumps(row) for row in held_rows)
                page.close()
            refusals = {"off": _post(url, f"/api/skills/{NOTES}/toggle", {"enabled": False}),
                        "delete": _post(url, f"/api/skills/{NOTES}/delete", {})}
            assert {name: (status, body.get("code")) for name, (status, body) in refusals.items()} == {
                "off": (403, "managed_by_deployment"), "delete": (403, "managed_by_deployment")}, refusals
            assert enabled_file.read_bytes() == saved_off, "the employee's saved Off stays as saved"
            assert DSN not in json.dumps(refusals)

            # --- warn: the same gap, and the turn is answered ---
            fixture["stop_server"]()
            _policy(policy, "warn", [pins[NOTES], pins["error_tracker"]])
            fixture["start_server"]()
            page = browser.new_page()
            _skills(page, url, "light", 1400)
            notice = page.locator("[data-skills-org-requirements]")
            notice.wait_for(state="visible", timeout=30_000)
            assert "Work goes on while it isn’t running." in notice.inner_text()
            _shot(page, evidence, "managed-warn-skills-light-1400", notice)
            _answered(page, data_dir, calls, _ask(page, url, "Reply briefly."))
            _shot(page, evidence, "managed-warn-chat-light-1400")
            warned = _rows(data_dir / "logs" / "events.jsonl", "deployment_requirement_unmet")
            assert warned[-1]["when_unmet"] == "warn" and warned[-1]["process"]["role"] in {"server", "worker"}
            page.close()

            # --- hold, every requirement running: nothing is held ---
            fixture["stop_server"]()
            _policy(policy, "hold", [pins[NOTES]])
            fixture["start_server"]()
            page = browser.new_page()
            _skills(page, url, "dark", 1400)
            assert not page.locator("[data-skills-org-requirements]").is_visible()
            _shot(page, evidence, "managed-met-skills-dark-1400", page.locator(f'.skills-card[data-skill="{NOTES}"]'))
            _answered(page, data_dir, calls, _ask(page, url, "Reply briefly."))
            _shot(page, evidence, "managed-met-chat-dark-1400")
            page.close()
        finally:
            browser.close()


def _loopback_receiver():
    """A synthetic Sentry endpoint on this machine that takes every report (HTTP 200); nothing leaves the host."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 - stdlib callback name
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *_args):
            return None

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


@pytest.mark.ui_browser
def test_the_loaded_error_tracker_widget_draws_what_each_process_observed(direct_server_with_data, tmp_path):  # noqa: F811
    """The strict sample with the tracker running: the file requires it and fixes its loopback DSN, the employee's
    switch stays Off, and the owner grants SENTRY_DSN on its card. Its widget, drawn by the real SPA from the plugin's
    own route,
    names each process with what its transport observed: the server's own running record, an ended worker and a
    worker of unknown liveness whose last report failed, which is the problem it states. No DSN reaches the page.
    The SDK is the real sentry-sdk 2.71.0 this test process imports, handed to the server's environment."""
    sentry_sdk = pytest.importorskip("sentry_sdk")
    if sentry_sdk.VERSION != "2.71.0":
        pytest.skip("the transport hooks are verified on sentry-sdk 2.71.0")
    fixture = direct_server_with_data
    data_dir, url = fixture["data_dir"], fixture["url"]
    evidence = pathlib.Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", str(tmp_path / "frames")))
    evidence.mkdir(parents=True, exist_ok=True)
    fixture["stop_server"]()
    site = str(pathlib.Path(sentry_sdk.__file__).resolve().parents[1])
    fixture["env"]["PYTHONPATH"] = os.pathsep.join(path for path in (site, fixture["env"].get("PYTHONPATH")) if path)
    receiver, serving = _loopback_receiver()
    try:
        _draw_tracker(fixture, data_dir, url, evidence, tmp_path, receiver)
    finally:
        receiver.shutdown()
        receiver.server_close()
        serving.join(timeout=10)


def _tracker_card(page, url):
    page.goto(url + "/#skills", wait_until="domcontentloaded")
    card = page.locator('.skills-card[data-skill="error_tracker"]')
    card.wait_for(state="visible", timeout=60_000)
    page.wait_for_function("() => !(document.querySelector('#skills-status')?.textContent || '').includes('Loading')")
    return card


def _tracker_status(url) -> dict:
    deadline = time.monotonic() + 90
    while True:
        try:
            with urllib.request.urlopen(url + "/api/extensions/error_tracker/status", timeout=10) as answer:
                return json.loads(answer.read())
        except urllib.error.HTTPError as error:
            assert time.monotonic() < deadline, f"the tracker never ran in the server: {error.code} {error.read()!r}"
            time.sleep(0.5)


def _requirements_met(url) -> list:
    """The server's own requirement rows once every one reads met on two reads in a row: a grant settles in a
    background job, which may still reload the extension after its route first answers."""
    deadline, streak = time.monotonic() + 90, 0
    while True:
        with urllib.request.urlopen(url + "/api/extensions", timeout=10) as answer:
            rows = (json.loads(answer.read()).get("deployment") or {}).get("required") or []
        streak = streak + 1 if rows and all(row.get("state") == "met" for row in rows) else 0
        if streak >= 2:
            return rows
        assert time.monotonic() < deadline, f"the server never read every requirement met: {rows}"
        time.sleep(1)


def _draw_tracker(fixture, data_dir, url, evidence, tmp_path, receiver):
    """Seed the strict sample and synthetic records, start the server, and draw the widget in each theme and width."""
    import subprocess
    import sys

    from playwright.sync_api import sync_playwright

    from ouroboros.skill_loader import SkillReviewState, load_skill, save_enabled, save_review_state

    dsn = f"http://fixturekey@127.0.0.1:{receiver.server_address[1]}/1"
    tracker = data_dir / "skills" / "external" / "error_tracker"
    shutil.copytree(REPO / "docs" / "examples" / "error_tracker", tracker)
    loaded = load_skill(tracker, data_dir)
    save_review_state(data_dir, loaded.name, SkillReviewState(status="pass", content_hash=loaded.content_hash))
    save_enabled(data_dir, loaded.name, False, actor="owner_ui")  # no grant yet: the owner gives it on the card
    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait()
    delivery = data_dir / "state" / "skills" / "error_tracker" / "delivery"
    delivery.mkdir(parents=True, exist_ok=True)
    records = {
        f"worker-{gone.pid}": {"role": "worker", "pid": gone.pid, "started": "an earlier start",
                               "loaded_at": "2026-11-08T09:00:00Z", "source": "settings", "sdk": "2.71.0",
                               "counts": {"queued": 5, "accepted": 3, "refused": 1, "unreachable": 1},
                               "last": {"outcome": "refused", "detail": "HTTP 500", "at": "2026-11-08T09:40:00Z"}},
        f"worker-{os.getpid()}": {"role": "worker", "pid": os.getpid(), "started": "",
                                  "loaded_at": "2026-11-08T09:30:00Z", "source": "settings", "sdk": "2.71.0",
                                  "counts": {"queued": 2, "accepted": 1, "unreachable": 1},
                                  "last": {"outcome": "unreachable", "detail": "no HTTP answer",
                                           "at": "2026-11-08T09:45:00Z"}},
    }
    for name, record in records.items():  # synthetic observations in the plugin's own state, as its processes write
        (delivery / f"{name}.json").write_text(json.dumps(record), encoding="utf-8")
    policy = tmp_path / "admin" / "deployment-policy.json"
    policy.parent.mkdir()
    _policy(policy, "hold", [{"name": loaded.name, "path": str(loaded.skill_dir), "content_hash": loaded.content_hash,
                              "settings": {"SENTRY_DSN": dsn}}])
    fixture["env"]["OUROBOROS_DEPLOYMENT_POLICY"] = str(policy)
    fixture["start_server"]()
    unknown = f"worker · pid {os.getpid()} · liveness unknown"
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            page = browser.new_page(viewport={"width": 1400, "height": 1000})
            page.add_init_script("localStorage.setItem('ouroboros.theme', 'light')")
            card = _tracker_card(page, url)
            assert card.locator("[data-required-unmet]").is_visible(), "required and not running: it needs the grant"
            card.locator('[data-skill-action="grant"]').click()
            dialog = page.locator(".confirm-dialog")
            dialog.wait_for(state="visible", timeout=10_000)
            assert "SENTRY_DSN" in dialog.inner_text() and "fixturekey" not in dialog.inner_text()
            dialog.locator("[data-confirm-ok]").click()
            dialog.wait_for(state="detached", timeout=30_000)
            status = _tracker_status(url)  # the plugin's own route answers once the tracker runs in the server
            assert unknown in [row["process"] for row in status["processes"]], status
            assert [row["name"] for row in _requirements_met(url)] == ["error_tracker"]
            card = _tracker_card(page, url)
            assert not page.locator("[data-skills-org-requirements]").is_visible() and card.locator(
                "[data-required-unmet]").count() == 0, "every requirement runs: no notice, no tray"
            assert card.locator(".skills-toggle").is_checked(), "on by the organization over the saved Off"
            _shot(page, evidence, "tracker-granted-skills-light-1400", card)
            page.close()
            for theme, width in (("light", 1400), ("dark", 1400), ("dark", 390), ("light", 390)):
                page = browser.new_page(viewport={"width": width, "height": 1000 if width > 500 else 844})
                page.add_init_script(f"localStorage.setItem('ouroboros.theme', {json.dumps(theme)})")
                page.goto(url, wait_until="domcontentloaded")
                page.wait_for_selector("#chat-input", timeout=30_000)
                if width <= 500:
                    page.click("[data-mobile-nav-toggle]")
                    page.wait_for_selector("#primary-sidebar.open", timeout=5_000)
                page.click('[data-nav-page="widgets"]')
                page.wait_for_function("""() => { const bar = document.querySelector('#primary-sidebar');
                    return !bar || (!bar.classList.contains('open')
                        && bar.getAnimations({subtree: true}).every((a) => a.playState !== 'running')); }""")
                card = page.locator('[data-widget-key="error_tracker:delivery"]')
                card.wait_for(state="visible", timeout=60_000)
                page.wait_for_function("(text) => document.querySelector('[data-widget-key=\"error_tracker:delivery\"]')"
                                       "?.textContent.includes(text)", arg=unknown, timeout=60_000)
                assert page.evaluate("document.documentElement.dataset.theme") == theme
                text = card.inner_text()
                assert "Error tracker" in text and "Settings" in text
                assert re.search(r"server · pid \d+ · running", text), text
                assert f"worker · pid {gone.pid} · ended" in text and "refused (HTTP 500)" in text
                assert f"Last outcome not accepted in {unknown}: unreachable (no HTTP answer)" in text, text
                assert "liveness unknown: unreachable" in text and f"pid {gone.pid} · ended:" not in text
                assert "fixturekey" not in page.content() and str(receiver.server_address[1]) not in text
                assert page.evaluate("document.querySelector('#page-widgets').scrollWidth <= innerWidth + 1")
                _shot(page, evidence, f"tracker-widget-{theme}-{width}", card if width > 500 else None)
                page.close()
        finally:
            browser.close()
