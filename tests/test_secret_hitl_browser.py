"""Qualify the integration-secret form in the served SPA without a model call.

The page is the real interface (static ``web/``) in Chromium and WebKit. Its question card arrives
as the frame the real ``LocalChatBridge.send_quiz`` broadcasts for a ``showSecretHitl`` request;
its reads and saves reach the real ``POST /api/decisions/secret`` handler over a real task store and
a real Settings file (Playwright forwards the page's requests to the in-process app); its lifecycle
frames are the ones the bridge broadcasts; a reload reads the real chat-history endpoint. The full
``server.py`` process is not booted here. The canary matches no token pattern; each flow proves it
reached ``settings.json`` and nothing a page, a reply, storage or the data root keeps. Screenshots
go to ``OUROBOROS_UI_EVIDENCE_DIR``.
"""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from ouroboros import owner_quiz
from ouroboros import secret_requests as sr
from ouroboros.task_results import write_task_result
from tests.test_secret_hitl import TASK, _write_skill
from tests.test_subscription_setup_browser import capture
from tests.test_subscription_setup_browser import subscription_ui as subscription_ui

pytestmark = [pytest.mark.ui_browser]
CANARY = "zq7-CANARY-Ωlj3~p0+kx/2 browser"
CANARY_B = "Second·canary·browser-9"
STORAGE = "() => JSON.stringify([Object.entries(localStorage), Object.entries(sessionStorage)])"


@pytest.fixture(params=["chromium", "webkit"])
def engine_ui(request, monkeypatch):
    monkeypatch.setenv("OUROBOROS_UI_BROWSER_ENGINE", request.param)
    root = os.environ.get("OUROBOROS_UI_EVIDENCE_DIR")
    if root:
        monkeypatch.setenv("OUROBOROS_UI_EVIDENCE_DIR", str(Path(root) / request.param))
    return request.getfixturevalue("subscription_ui")


@pytest.fixture
def backend(tmp_path, monkeypatch):
    import supervisor.message_bus as mb
    import supervisor.queue as q
    from ouroboros import config as cfg
    from ouroboros.gateway import secret_decision as sd
    from ouroboros.gateway import task_decision as td
    from ouroboros.gateway.history import make_chat_history_endpoint
    from ouroboros.skill_loader import save_enabled

    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setattr(cfg, "DATA_DIR", data)
    monkeypatch.setattr(cfg, "SETTINGS_PATH", data / "settings.json")
    monkeypatch.setattr(cfg, "get_skills_repo_path", lambda: "")
    _write_skill(data)
    _write_skill(data, "mailer", env="FAKEPLUG_TOKEN", fields="  token: FAKEPLUG_TOKEN")
    save_enabled(data, "fakeplug", True)
    write_task_result(data, TASK, "running", chat_id=1)
    frames = []
    bridge = mb.LocalChatBridge({})
    bridge._broadcast_fn = frames.append
    monkeypatch.setattr(mb, "DATA_DIR", data)
    monkeypatch.setattr(mb, "load_state", lambda: {"owner_id": 1})
    monkeypatch.setattr(mb, "publish_event", lambda *args: None)
    monkeypatch.setattr(mb, "get_bridge", lambda: bridge)
    live = {"task": True}
    monkeypatch.setattr(td, "_live_root_task", lambda task_id, holds_mail=False: (
        ({"id": task_id}, "") if live["task"] else (None, "task_not_live")))
    monkeypatch.setattr(q, "_task_drive_for_task", lambda task, task_id: data, raising=False)
    app = Starlette(routes=[Route("/api/decisions/secret", endpoint=sd.api_secret_decision, methods=["POST"]),
                            Route("/api/chat/history", endpoint=make_chat_history_endpoint(data))])
    app.state.drive_root = data

    def ask():
        request = sr.resolve_secret_request(data, "fakeplug", ["botToken"])
        quiz_id = uuid.uuid4().hex
        owner_quiz.record_asked(data, TASK, quiz_id=quiz_id, question=sr.question_text(request), options=[],
                                wait_for_answer=True, chat_id=1, secret_request=request)
        before = len(frames)
        ok, _ = bridge.send_quiz(1, quiz_id=quiz_id, question=sr.question_text(request), options=[],
                                 task_id=TASK, wait_for_answer=True)
        assert ok
        [frame] = [frame for frame in frames[before:] if frame.get("type") == "quiz"]
        return quiz_id, frame

    return type("Backend", (), {"data": data, "frames": frames, "client": TestClient(app), "ask": staticmethod(ask),
                                "live": live, "hold": False, "drop": False})


def _open(page, backend):
    sockets, seen, held = [], [], []
    console = []
    page.on("console", lambda message: console.append(message.text))

    def forward(route, *, drop=False):
        response = backend.client.post("/api/decisions/secret", content=route.request.post_data,
                                       headers={"Content-Type": "application/json"})
        seen.append(response.text)
        if drop:
            route.abort("connectionreset")  # the server finished; the reply never reached the page
        else:
            route.fulfill(status=response.status_code, content_type="application/json", body=response.text)

    def secret_route(route):
        if backend.hold and '"status"' not in (route.request.post_data or ""):
            held.append(route)
            return
        forward(route, drop=backend.drop and '"status"' not in (route.request.post_data or ""))

    def history_route(route):
        url = route.request.url
        response = backend.client.get(url[url.index("/api/chat/history"):])
        seen.append(response.text)
        route.fulfill(status=response.status_code, content_type="application/json", body=response.text)

    def socket_route(ws):
        sockets.append(ws)
        ws.send(json.dumps({"type": "heartbeat"}))

    page.route("**/api/state", lambda route: route.fulfill(content_type="application/json", body=json.dumps(
        {"supervisor_ready": True, "active_chat_activities": [], "projects": [], "sha": "secret-hitl"})))
    page.route("**/api/chat/history*", history_route)
    page.route("**/api/decisions/secret", secret_route)
    page.route_web_socket("**/ws", socket_route)
    return sockets, seen, held, forward, console


def _load(ui, page, backend):
    wired = _open(page, backend)
    page.goto(ui["url"])
    page.wait_for_selector("#chat-input")
    sockets = wired[0]
    push = lambda frame: sockets[-1].send(json.dumps(frame))  # noqa: E731 - the live socket of this load
    return (push, *wired[1:])


def _settings(backend):
    path = backend.data / "settings.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _push_states(backend, push, since=0):
    for live in [item for item in backend.frames[since:] if item.get("type") == "quiz_state"]:
        push(live)


def _no_canary(page, backend, seen, console, *values):
    texts = [page.content(), page.evaluate(STORAGE), "\n".join(console), *seen]
    for value in values:
        for text in texts:
            assert value not in text and json.dumps(value)[1:-1] not in text
        for path in backend.data.rglob("*"):
            if path.is_file() and not path.name.startswith("settings.json"):
                assert value.encode() not in path.read_bytes(), path


@pytest.mark.parametrize("theme", ["light", "dark"])
@pytest.mark.parametrize("viewport", [{"width": 1148, "height": 808}, {"width": 390, "height": 844}])
def test_secret_form_first_fill_and_replace_screens(engine_ui, backend, viewport, theme):
    """The owner-visible states at the owner's sizes: card, first fill, Replace warning, saved card."""
    ui = engine_ui
    page = ui["page"]
    page.emulate_media(color_scheme=theme)
    page.set_viewport_size(viewport)
    push, seen, _held, _forward, console = _load(ui, page, backend)
    tag = f"{viewport['width']}x{viewport['height']}-{theme}"
    quiz_id, frame = backend.ask()
    push(frame)
    card = page.locator(f'.chat-quiz-card[data-quiz-id="{quiz_id}"]')
    card.locator(".chat-secret-open").wait_for()
    assert card.locator(".chat-secret-field-key").inner_text() == "Saved in Settings as FAKEPLUG_TOKEN"
    capture(page, f"secret-card-{tag}")
    card.locator(".chat-secret-open").click()
    dialog = page.locator(".secret-request-dialog")
    dialog.wait_for(state="visible")
    dialog.locator(".secret-request-key").filter(has_text="not set").wait_for()
    assert dialog.evaluate("node => getComputedStyle(node).backgroundColor").startswith("rgb(")
    readers = dialog.locator(".secret-request-reader").all_inner_texts()
    assert readers == ["fakeplug · enabled · needs FAKEPLUG_TOKEN granted in Skills; after saving, this enabled skill cannot run until the keys are granted",
                       "mailer · disabled · needs FAKEPLUG_TOKEN granted in Skills"]
    ok = dialog.locator(".secret-request-ok")
    assert ok.inner_text() == "Save" and ok.is_disabled()
    dialog.locator("input.secret-request-input").fill(CANARY)
    assert ok.is_enabled()
    box = dialog.bounding_box()
    assert box["x"] >= 0 and box["x"] + box["width"] <= viewport["width"] + 0.5
    for selector in (".marketplace-modal-head", ".marketplace-modal-actions"):
        region = dialog.locator(selector).bounding_box()
        assert region["y"] >= 0 and region["y"] + region["height"] <= viewport["height"]
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    capture(page, f"secret-form-first-fill-{tag}")
    dialog.locator(".secret-request-show").click()  # the typed draft only
    dialog.locator(".settings-secret-value").filter(has_text=CANARY).wait_for()
    dialog.locator(".secret-request-show").click()
    assert dialog.locator(".settings-secret-value").inner_text() == ""
    since = len(backend.frames)
    ok.click()
    dialog.wait_for(state="detached")
    _push_states(backend, push, since)
    card.locator(".chat-secret-status").filter(has_text="Saved to Settings: botToken").wait_for()
    capture(page, f"secret-card-saved-{tag}")
    assert _settings(backend)["FAKEPLUG_TOKEN"] == CANARY

    second, frame = backend.ask()
    push(frame)
    card = page.locator(f'.chat-quiz-card[data-quiz-id="{second}"]')
    card.locator(".chat-secret-open").click()
    dialog.wait_for(state="visible")
    dialog.locator(".secret-request-key").filter(has_text="a value is set (hidden)").wait_for()
    assert dialog.locator(".secret-request-ok").inner_text() == "Replace"
    warning = dialog.locator(".secret-request-warning").inner_text()
    assert warning.startswith("Replace overwrites the saved value. Enabled skills that read it (fakeplug)")
    dialog.locator("input.secret-request-input").fill(CANARY_B)
    capture(page, f"secret-form-replace-{tag}")
    dialog.locator(".secret-request-close").click()
    _no_canary(page, backend, seen, console, CANARY, CANARY_B)


def test_a_key_saved_elsewhere_turns_save_into_replace_and_keeps_the_draft(engine_ui, backend):
    ui = engine_ui
    page = ui["page"]
    push, seen, _held, _forward, console = _load(ui, page, backend)
    quiz_id, frame = backend.ask()
    push(frame)
    card = page.locator(f'.chat-quiz-card[data-quiz-id="{quiz_id}"]')
    card.locator(".chat-secret-open").click()
    dialog = page.locator(".secret-request-dialog")
    dialog.locator(".secret-request-key").filter(has_text="not set").wait_for()
    field = dialog.locator("input.secret-request-input")
    field.fill(CANARY_B)
    # Another request (another task, another window) saves the key while this form is open.
    other, _ = backend.ask()
    observed = {"botToken": False}
    response = backend.client.post("/api/decisions/secret", json={"request_id": "elsewhere", "observed": observed,
        "decision_id": f"quiz:{TASK}:{other}", "values": {"botToken": CANARY}})
    assert response.status_code == 200
    dialog.locator(".secret-request-ok").click()
    dialog.locator(".secret-request-error").filter(has_text="after the form opened, so your entry was not saved").wait_for()
    assert field.input_value() == CANARY_B, "the draft survives the refusal"
    # Focus stays in the form (WebKit focuses the panel, not a clicked button), never the page behind it.
    assert page.evaluate("document.querySelector('.secret-request-dialog').contains(document.activeElement)")
    assert dialog.locator(".secret-request-ok").inner_text() == "Replace"
    assert _settings(backend)["FAKEPLUG_TOKEN"] == CANARY
    capture(page, "secret-form-stale-now-replace")
    since = len(backend.frames)
    dialog.locator(".secret-request-ok").click()
    dialog.wait_for(state="detached")
    _push_states(backend, push, since)
    card.locator(".chat-secret-status").filter(has_text="Replaced in Settings: botToken").wait_for()
    assert _settings(backend)["FAKEPLUG_TOKEN"] == CANARY_B
    ui["errors"].watch(page, navigates=True)
    page.reload()
    page.wait_for_selector("#chat-input")
    page.locator(f'.chat-quiz-card[data-quiz-id="{quiz_id}"] .chat-secret-status').filter(
        has_text="Replaced in Settings").wait_for()
    _no_canary(page, backend, seen, console, CANARY, CANARY_B)


def test_a_lost_reply_stays_unknown_until_status_and_duplicate_presses_send_once(engine_ui, backend):
    ui = engine_ui
    page = ui["page"]
    push, seen, held, forward, console = _load(ui, page, backend)
    quiz_id, frame = backend.ask()
    push(frame)
    card = page.locator(f'.chat-quiz-card[data-quiz-id="{quiz_id}"]')
    card.locator(".chat-secret-open").click()
    dialog = page.locator(".secret-request-dialog")
    dialog.locator(".secret-request-key").filter(has_text="not set").wait_for()
    dialog.locator("input.secret-request-input").fill(CANARY)
    backend.hold = True
    ok = dialog.locator(".secret-request-ok")
    ok.click()
    page.wait_for_function("document.querySelector('.secret-request-ok')?.textContent === 'Saving…'")
    ok.click(force=True)  # a second press while the first is in flight sends nothing
    assert card.locator(".chat-quiz-mark").get_attribute("data-hue") == "neutral"
    capture(page, "secret-form-saving")
    assert len(held) == 1
    backend.hold = False
    forward(held.pop(), drop=True)  # the save landed; its reply was lost
    dialog.locator(".secret-request-error").filter(has_text="its result is unknown").wait_for()
    assert ok.is_disabled() and dialog.locator(".secret-request-check").is_visible()
    capture(page, "secret-form-unknown")
    dialog.locator(".secret-request-check").click()
    dialog.wait_for(state="detached")
    card.locator(".chat-secret-status").filter(has_text="Saved to Settings").wait_for()
    assert _settings(backend)["FAKEPLUG_TOKEN"] == CANARY
    assert len([text for text in seen if '"secret_filled"' in text]) == 2  # the dropped reply and the status read
    _no_canary(page, backend, seen, console, CANARY)


def test_close_waits_for_a_save_in_flight_and_its_refusal_leaves_a_live_form(engine_ui, backend):
    """Close, like Escape and a click outside, waits for the reply that settles a save; a refusal then
    leaves Save live on the reply's fresh facts. A form closed and reopened mid-save stayed loading."""
    ui = engine_ui
    page = ui["page"]
    push, seen, held, forward, console = _load(ui, page, backend)
    quiz_id, frame = backend.ask()
    push(frame)
    card = page.locator(f'.chat-quiz-card[data-quiz-id="{quiz_id}"]')
    card.locator(".chat-secret-open").click()
    dialog = page.locator(".secret-request-dialog")
    dialog.locator(".secret-request-key").filter(has_text="not set").wait_for()
    dialog.locator("input.secret-request-input").fill(CANARY_B)
    backend.hold = True
    dialog.locator(".secret-request-ok").click()
    page.wait_for_function("document.querySelector('.secret-request-ok')?.textContent === 'Saving…'")
    assert dialog.locator(".secret-request-close").is_disabled()
    assert dialog.locator(".secret-request-cancel").is_disabled()
    dialog.locator(".secret-request-close").click(force=True)
    dialog.locator(".secret-request-cancel").click(force=True)
    page.keyboard.press("Escape")
    page.mouse.click(4, 4)  # the backdrop, outside the panel
    assert dialog.is_visible(), "the form waits for the reply that settles it"
    capture(page, "secret-form-saving-close-waits")
    # Another request saves the key meanwhile, so the held save is refused with fresh facts.
    other, _ = backend.ask()
    assert backend.client.post("/api/decisions/secret", json={"request_id": "elsewhere", "observed": {"botToken": False},
                               "decision_id": f"quiz:{TASK}:{other}", "values": {"botToken": CANARY}}).status_code == 200
    backend.hold = False
    forward(held.pop())
    dialog.locator(".secret-request-error").filter(has_text="after the form opened, so your entry was not saved").wait_for()
    ok = dialog.locator(".secret-request-ok")
    assert ok.inner_text() == "Replace" and ok.is_enabled()
    assert dialog.locator(".secret-request-close").is_enabled()
    assert not dialog.locator(".secret-request-check").is_visible()
    capture(page, "secret-form-refused-after-held-save")
    since = len(backend.frames)
    ok.click()
    dialog.wait_for(state="detached")
    _push_states(backend, push, since)
    card.locator(".chat-secret-status").filter(has_text="Replaced in Settings: botToken").wait_for()
    assert _settings(backend)["FAKEPLUG_TOKEN"] == CANARY_B
    _no_canary(page, backend, seen, console, CANARY, CANARY_B)


def test_a_held_save_releases_the_owner_after_the_wait_and_its_late_reply_settles(engine_ui, backend):
    """One real held response over the default 15 s wait, in both engines: the page stops holding Close,
    offers Check status and resends nothing; the server's late reply then settles the card."""
    ui = engine_ui
    page = ui["page"]
    push, seen, held, forward, console = _load(ui, page, backend)
    quiz_id, frame = backend.ask()
    push(frame)
    card = page.locator(f'.chat-quiz-card[data-quiz-id="{quiz_id}"]')
    card.locator(".chat-secret-open").click()
    dialog = page.locator(".secret-request-dialog")
    dialog.locator(".secret-request-key").filter(has_text="not set").wait_for()
    dialog.locator("input.secret-request-input").fill(CANARY)
    backend.hold = True
    ok = dialog.locator(".secret-request-ok")
    ok.click()
    page.wait_for_function("document.querySelector('.secret-request-ok')?.textContent === 'Saving…'")
    assert dialog.locator(".secret-request-close").is_disabled()
    dialog.locator(".secret-request-error").filter(has_text="was not confirmed").wait_for(timeout=30000)
    assert dialog.locator(".secret-request-close").is_enabled() and dialog.locator(".secret-request-cancel").is_enabled()
    assert ok.is_disabled() and dialog.locator(".secret-request-check").is_visible()
    assert card.locator(".chat-quiz-mark").get_attribute("data-hue") == "neutral"
    capture(page, "secret-form-held-save-released")
    ok.click(force=True)
    dialog.locator("input.secret-request-input").press("Enter")
    assert len(held) == 1, "an unconfirmed save is never resent"
    dialog.locator(".secret-request-close").click()  # Close is available again
    dialog.wait_for(state="detached")
    assert page.locator("input.secret-request-input").count() == 0
    backend.hold = False
    since = len(backend.frames)
    forward(held.pop())  # the late reply of the same save
    card.locator(".chat-secret-status").filter(has_text="Saved to Settings").wait_for()
    _push_states(backend, push, since)
    assert _settings(backend)["FAKEPLUG_TOKEN"] == CANARY
    assert len([text for text in seen if '"secret_filled"' in text]) == 1
    _no_canary(page, backend, seen, console, CANARY)


def test_close_is_not_cancel_reload_reopens_empty_and_cancel_is_recorded(engine_ui, backend):
    ui = engine_ui
    page = ui["page"]
    push, seen, _held, _forward, console = _load(ui, page, backend)
    quiz_id, frame = backend.ask()
    push(frame)
    dialog = page.locator(".secret-request-dialog")
    card = page.locator(f'.chat-quiz-card[data-quiz-id="{quiz_id}"]')
    open_button = card.locator(".chat-secret-open")
    open_button.click()
    dialog.wait_for(state="visible")
    dialog.locator("input.secret-request-input").fill(CANARY)
    dialog.locator(".secret-request-cancel").click()  # footer Close is dismissal, not cancellation
    dialog.wait_for(state="detached")
    assert owner_quiz.quiz_states(backend.data, TASK)[quiz_id]["state"] == "open"
    open_button.click()
    assert dialog.locator("input.secret-request-input").input_value() == ""
    page.keyboard.press("Escape")
    dialog.wait_for(state="detached")
    ui["errors"].watch(page, navigates=True)
    page.reload()
    page.wait_for_selector("#chat-input")
    card = page.locator(f'.chat-quiz-card[data-quiz-id="{quiz_id}"]')
    card.locator(".chat-secret-open").wait_for()
    assert page.locator(".secret-request-dialog").count() == 0  # a replay never opens it by itself
    card.locator(".chat-secret-cancel").click()
    card.locator(".chat-secret-actions").wait_for(state="detached")
    assert "FAKEPLUG_TOKEN" not in _settings(backend)
    assert owner_quiz.quiz_states(backend.data, TASK)[quiz_id]["secret_outcome"]["event"] == "secret_cancelled"
    capture(page, "secret-card-cancelled")
    _no_canary(page, backend, seen, console, CANARY)


def test_a_finished_task_keeps_the_form_and_a_late_save_is_recorded(engine_ui, backend):
    ui = engine_ui
    page = ui["page"]
    push, seen, _held, _forward, console = _load(ui, page, backend)
    quiz_id, frame = backend.ask()
    push(frame)
    owner_quiz.reconcile_terminal(backend.data, TASK)
    backend.live["task"] = False
    push({"type": "quiz_state", "quiz_id": quiz_id, "task_id": TASK, "state": "expired_terminal",
          "ts": "2026-11-02T00:00:00Z", "chat_id": 1})
    card = page.locator(f'.chat-quiz-card[data-quiz-id="{quiz_id}"]')
    card.locator(".chat-secret-status").filter(has_text="it does not resume that work").wait_for()
    card.locator(".chat-secret-open").click()
    dialog = page.locator(".secret-request-dialog")
    dialog.locator(".secret-request-late").filter(has_text="has finished").wait_for()
    dialog.locator(".secret-request-key").filter(has_text="not set").wait_for()
    dialog.locator("input.secret-request-input").fill(CANARY)
    capture(page, "secret-form-after-task-finished")
    since = len(backend.frames)
    dialog.locator(".secret-request-ok").click()
    dialog.wait_for(state="detached")
    _push_states(backend, push, since)
    card.locator(".chat-secret-status").filter(has_text="The task had already finished; nothing was resumed.").wait_for()
    block = owner_quiz.quiz_states(backend.data, TASK)[quiz_id]
    assert block["answered_after_terminal"] is True and block["delivery"]["route"] == "history"
    _no_canary(page, backend, seen, console, CANARY)


@pytest.mark.parametrize("held", [True, False])
def test_a_remote_settlement_moves_focus_to_the_question_only_if_the_form_held_it(engine_ui, backend, held):
    """Another window saves while this form's input (or the composer) has focus: the lifecycle frame closes the form."""
    ui = engine_ui
    page = ui["page"]
    push, seen, _held, _forward, console = _load(ui, page, backend)
    quiz_id, frame = backend.ask()
    push(frame)
    card = page.locator(f'.chat-quiz-card[data-quiz-id="{quiz_id}"]')
    card.locator(".chat-secret-open").click()
    dialog = page.locator(".secret-request-dialog")
    dialog.locator(".secret-request-key").filter(has_text="not set").wait_for()
    field = dialog.locator("input.secret-request-input")
    field.fill(CANARY)
    (field if held else page.locator("#chat-input")).focus()
    since = len(backend.frames)
    assert backend.client.post("/api/decisions/secret", json={
        "request_id": "elsewhere", "observed": {"botToken": False}, "decision_id": f"quiz:{TASK}:{quiz_id}",
        "values": {"botToken": CANARY_B}}).status_code == 200
    _push_states(backend, push, since)
    dialog.wait_for(state="detached")
    card.locator(".chat-secret-status").filter(has_text="Saved to Settings").wait_for()
    focused = page.evaluate("""() => { const node = document.activeElement;
        return { question: Boolean(node?.classList?.contains('chat-quiz-question')), id: node?.id || '', body: node === document.body }; }""")
    if held:
        assert focused["question"] and not focused["body"], focused
    else:
        assert focused["id"] == "chat-input", focused
    capture(page, f"secret-form-remote-settled-focus-{'held' if held else 'elsewhere'}")
    _no_canary(page, backend, seen, console, CANARY, CANARY_B)


def test_another_window_settles_an_open_form(engine_ui, backend):
    ui = engine_ui
    first = ui["page"]
    push_first, seen_first, *_rest = _load(ui, first, backend)
    second = ui["new_page"](viewport={"width": 1148, "height": 808})
    push_second, seen_second, _held, _forward, console = _load(ui, second, backend)
    quiz_id, frame = backend.ask()
    push_first(frame)
    push_second(frame)
    for page in (first, second):
        page.locator(f'.chat-quiz-card[data-quiz-id="{quiz_id}"] .chat-secret-open').click()
        page.locator(".secret-request-key").filter(has_text="not set").wait_for()
    second.locator("input.secret-request-input").fill(CANARY_B)
    first.locator("input.secret-request-input").fill(CANARY)
    since = len(backend.frames)
    first.locator(".secret-request-ok").click()
    first.locator(".secret-request-dialog").wait_for(state="detached")
    _push_states(backend, push_second, since)
    second.locator(".secret-request-dialog").wait_for(state="detached")
    second.locator(f'.chat-quiz-card[data-quiz-id="{quiz_id}"] .chat-secret-status').filter(
        has_text="Saved to Settings").wait_for()
    assert _settings(backend)["FAKEPLUG_TOKEN"] == CANARY
    _no_canary(second, backend, seen_second + seen_first, console, CANARY, CANARY_B)
    second.close()
