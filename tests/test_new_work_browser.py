"""The model a room remembers, from the page to the provider's wire and back: the Main chat's capsule, a busy room,
two windows on one room, and Replies.

The real server on its own port and data root, the real page, and a loopback provider that stands for the models
(``fixtures_mock_llm``, ``tests._model_choice_browser.Provider``): no paid call and no account login is possible
here, and nothing of the server is answered by the test. Every room, model and word is made up.

What is proved end to end:

- the Main chat's capsule saves the Main chat's own model without a question and without writing Main the role; its
  next message is a new task on that model; a room made afterwards copies it, a room that existed keeps following
  Main;
- in a room where a work runs on one model, a pick of another makes Send say ``New task`` before the press, and the
  message starts work of its own on the picked model while the running work keeps its own; the capsule keeps the pick;
- two windows on one room: where the other window changed the model first, the page holds plain words back and keeps
  them, and the next Send is an ordinary message; where the page had not heard of the change, the host refuses the
  message, its words return to the field, and the next Send is an ordinary message that succeeds;
- a Reply is never held back by a pick that no longer stands: the host gives a Reply to a work's row to that work,
  and takes a quote of the owner's own words as the new task.
"""
from __future__ import annotations

import json
import os
import pathlib
import re

import pytest

from tests import fixtures_mock_llm
from tests._model_choice_browser import (
    _CAPSULE,
    MAIN,
    OTHER,
    PICKED,
    QUEUED,
    SOURCE,
    Provider,
    _answered,
    _capsule_ready,
    _idle,
    _open_project,
    _post,
    _send,
    _sheet_pick,
    _state,
)
from tests.test_effort_range_browser import _launch
from tests.test_ui_smoke_playwright import direct_server_with_data  # noqa: F401 -- fixture

pytestmark = [pytest.mark.ui_browser]  # its server, data root and provider are its own: nothing on the host is shared

_SEND = """(prefix) => { const send = document.querySelector('#' + prefix + '-send'), input = document.querySelector('#' + prefix + '-input');
    return { word: send.textContent, placeholder: input.placeholder, value: input.value }; }"""
# What the page reads the next press as: '1' a new task, 'reply' a Reply the host still has to place, '0' ordinary.
_READING = "(prefix) => document.querySelector('#' + prefix + '-send').closest('.chat-send-group').dataset.newWork"
_VALUE = "([sel, value]) => document.querySelector(sel + ' .chat-capsule-value').textContent === value"
_WORD = "([prefix, word]) => document.querySelector('#' + prefix + '-send').textContent === word"
CHANGED = "changed in another window"
DESK = {"width": 1512, "height": 949}
PHONE = {"viewport": {"width": 390, "height": 844}, "is_mobile": True, "has_touch": True}
_TOP = "(prefix) => Math.round(document.querySelector('#' + prefix + '-input').closest('.chat-field').getBoundingClientRect().top)"


def _theme(page, theme: str) -> None:
    """As Settings › Appearance sets it (web/theme.js). The page's surfaces and inks ease to the new theme (the
    composer's field longest), so a frame taken before they end shows both themes at once: wait for them."""
    page.evaluate(f"() => window.ouroTheme.set('{theme}')")
    page.wait_for_function(f"() => document.documentElement.dataset.theme === '{theme}'")
    page.wait_for_function("""() => document.getAnimations()
        .filter((a) => a.effect?.getComputedTiming().endTime !== Infinity)
        .every((a) => !a.pending && a.playState !== 'running')""", timeout=10_000)


def _frames(page) -> list:
    """Every chat frame this page writes to its socket, as the host receives it."""
    sent: list = []

    def on_socket(socket) -> None:
        def on_frame(payload) -> None:
            try:
                frame = json.loads(payload)
            except (TypeError, ValueError):
                return
            if isinstance(frame, dict) and frame.get("type") == "chat":
                sent.append(frame)
        socket.on("framesent", on_frame)

    page.on("websocket", on_socket)
    return sent


def _frame(frames: list, marker: str) -> dict:
    found = [frame for frame in frames if marker in str(frame.get("content"))]
    assert found, (marker, [frame.get("content") for frame in frames])
    return found[-1]


def _pick(page, capsule: str, route: str, value: str, *, finger: bool = False) -> None:
    _sheet_pick(page, capsule, route, finger=finger)
    page.wait_for_function(_VALUE, arg=[capsule, value], timeout=20_000)
    page.wait_for_function("() => !document.querySelector('.ui-popup.model-sheet')", timeout=10_000)


def _reply(page, prefix: str, role: str, words: str) -> bool:
    """A press on the Reply control of the first saved bubble of `role` whose words include `words`."""
    bubbles = page.locator(f"#{prefix}-messages .chat-bubble.{role}").filter(has_text=words).filter(
        has=page.locator(".chat-message-reply"))
    if not bubbles.count():
        return False
    bubbles.first.locator(".chat-message-reply").first.click()
    return True


def _asked_within(page, provider, marker: str, seconds: float) -> bool:
    for _ in range(int(seconds * 4)):
        if provider.models_of(marker):
            return True
        page.wait_for_timeout(250)
    return False


def test_a_remembered_model_starts_new_work_and_two_windows_never_send_on_a_stale_pick(direct_server_with_data, monkeypatch, tmp_path):  # noqa: F811
    from playwright.sync_api import sync_playwright

    server = direct_server_with_data
    server["stop_server"]()
    settings_path = server["data_dir"] / "settings.json"
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    settings.update(OUROBOROS_MODEL=MAIN, OUROBOROS_MODEL_LIGHT=f"{SOURCE}::mock-light",
                    OUROBOROS_MODEL_FALLBACKS=f"{SOURCE}::mock-fallback",
                    OUROBOROS_TASK_REVIEW_MODE="off", OUROBOROS_SAFETY_MODE="off",
                    OUROBOROS_SUBAGENTS=json.dumps({"enabled": False, "items": []}))
    settings_path.write_text(json.dumps(settings), encoding="utf-8")
    provider = Provider()
    monkeypatch.setattr(fixtures_mock_llm._Handler, "do_POST", lambda handler: provider.post(handler))
    monkeypatch.setattr(fixtures_mock_llm._Handler, "do_GET", lambda handler: provider.get(handler), raising=False)
    server["start_server"]()
    evidence = pathlib.Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", str(tmp_path)))
    evidence.mkdir(parents=True, exist_ok=True)
    seen: dict = {}

    def saved() -> dict:
        return json.loads(settings_path.read_text(encoding="utf-8"))

    with sync_playwright() as pw:
        engine = os.environ.get("OUROBOROS_UI_BROWSER_ENGINE") or "webkit"  # WebKit unless a run names another engine
        browser, blank = _launch(pw, engine)
        blank.close()
        windows = browser.new_context(viewport=DESK)  # one profile: its pages are two windows
        seen["looked_at"] = {"engine": engine, "desktop": {**DESK, "themes": ["dark", "light"]},
                             "phone": {**PHONE["viewport"], "mobile": True, "touch": True, "themes": ["dark", "light"]}}
        page = windows.new_page()
        frames = _frames(page)
        writes: list = []
        page.on("request", lambda request: writes.append((request.url.split("/api/", 1)[1], json.loads(request.post_data or "{}")))
                if request.method == "POST" and "/api/owner/chat-" in request.url else None)
        other = None
        try:
            page.goto(server["url"], wait_until="domcontentloaded")
            _capsule_ready(page, "#chat-model-mode")
            state = _state(page)
            assert state["main_route"] == {"model": MAIN, "use_local": False}
            assert state["main_chat"] == {"model_choice": {"kind": "main"}, "processing_choice": ""}
            assert page.evaluate(_SEND, "chat")["word"] == "Send"

            # --- 1. The Main chat's capsule: its own model, saved without a question; Main the role stays.
            alpha = _post(page, "/api/projects", {"name": "Chip alpha"})["project"]  # exists before the pick
            _pick(page, "#chat-model-mode", OTHER, "mock-other")
            assert not page.query_selector(".confirm-dialog"), "the Main chat's pick asks nothing"
            assert writes[-1] == ("owner/chat-model", {"kind": "model", "model": OTHER, "use_local": False,
                                                       "scope": "main_chat"}), writes
            state = _state(page)
            assert state["main_route"] == {"model": MAIN, "use_local": False}, "Main the role is not the capsule's to write"
            assert state["main_chat"]["model_choice"] == {"kind": "model", "model": OTHER, "use_local": False}
            assert (saved()["OUROBOROS_MODEL"], saved()["OUROBOROS_MAIN_CHAT_MODEL"]) == (MAIN, OTHER)
            page.wait_for_function(_WORD, arg=["chat", "New task"], timeout=10_000)
            said = page.evaluate(_SEND, "chat")
            seen["main_armed"] = said
            assert said == {"word": "New task", "placeholder": "New task...", "value": ""}, said
            page.screenshot(path=str(evidence / "01-main-capsule-new-task.png"))
            _send(page, "chat", "ROOM_WIRE_MAIN_CHIP")
            _answered(page, "chat", "ROOM_WIRE_MAIN_CHIP")
            _idle(page)
            assert provider.models_of("ROOM_WIRE_MAIN_CHIP") == ["mock-other"], provider.models_of("ROOM_WIRE_MAIN_CHIP")
            assert re.fullmatch(r"[0-9a-f]{24}", _frame(frames, "ROOM_WIRE_MAIN_CHIP").get("new_work", "")), frames[-1]
            page.wait_for_function(_WORD, arg=["chat", "Send"], timeout=10_000)
            # A room made now copies the Main chat's model; the one that existed keeps following Main.
            gamma = _post(page, "/api/projects", {"name": "Chip gamma"})["project"]
            rooms = {row["id"]: row.get("model_choice") for row in _state(page)["projects"]}
            seen["rooms"] = rooms
            assert rooms[gamma["id"]] == {"kind": "model", "model": OTHER, "use_local": False}, rooms
            assert rooms[alpha["id"]] == {"kind": "main"}, rooms
            gamma_prefix = _open_project(page, gamma)
            copied = page.evaluate(_CAPSULE, f"#{gamma_prefix}-model-mode")
            assert copied["value"] == "mock-other" and copied["own"] == "true", copied
            assert page.evaluate(_SEND, gamma_prefix)["word"] == "Send", "a copied model arms nothing"
            page.screenshot(path=str(evidence / "02-new-room-copied-the-model.png"))

            # --- 2. A busy room: a work runs on Main's model; a pick of another makes the next message a new task.
            prefix = _open_project(page, alpha)
            capsule = f"#{prefix}-model-mode"
            closed = page.evaluate(_CAPSULE, capsule)
            assert (closed["name"], closed["value"], closed["own"]) == ("Main", "mock-main", "false"), closed
            provider.hold("ROOM_WIRE_BUSY_A")
            _send(page, prefix, "ROOM_WIRE_BUSY_A")
            assert provider.held("ROOM_WIRE_BUSY_A"), "the room's first work never reached its held call"
            assert "new_work" not in _frame(frames, "ROOM_WIRE_BUSY_A"), "nothing was picked: an ordinary message"
            _pick(page, capsule, PICKED, "mock-picked")
            page.wait_for_function(_WORD, arg=[prefix, "New task"], timeout=10_000)
            said = page.evaluate(_SEND, prefix)
            seen["busy_armed"] = said
            assert said == {"word": "New task", "placeholder": "New task...", "value": ""}, said
            page.screenshot(path=str(evidence / "03-busy-room-says-new-task-before-the-press.png"))
            _theme(page, "light")
            page.screenshot(path=str(evidence / "03-busy-room-says-new-task-before-the-press-light.png"))
            _theme(page, "dark")
            _send(page, prefix, "ROOM_WIRE_BUSY_B")
            # Whether the new work starts beside the held one or after it is the host's scheduling; its model is not.
            seen["new_work_ran_beside_the_held_one"] = _asked_within(page, provider, "ROOM_WIRE_BUSY_B", 45)
            if not seen["new_work_ran_beside_the_held_one"]:
                provider.release("ROOM_WIRE_BUSY_A")
            _answered(page, prefix, "ROOM_WIRE_BUSY_B")
            assert provider.models_of("ROOM_WIRE_BUSY_B")[:1] == ["mock-picked"], provider.models_of("ROOM_WIRE_BUSY_B")
            assert set(provider.models_of("ROOM_WIRE_BUSY_A")) == {"mock-main"}, "the running work kept its own model"
            assert re.fullmatch(r"[0-9a-f]{24}", _frame(frames, "ROOM_WIRE_BUSY_B").get("new_work", ""))
            page.wait_for_function(_WORD, arg=[prefix, "Send"], timeout=10_000)
            assert page.evaluate(_CAPSULE, capsule)["value"] == "mock-picked", "the pick stays in the capsule"
            provider.release("ROOM_WIRE_BUSY_A")
            _answered(page, prefix, "ROOM_WIRE_BUSY_A")
            _idle(page)
            assert set(provider.models_of("ROOM_WIRE_BUSY_A")) == {"mock-main"}
            page.screenshot(path=str(evidence / "04-busy-room-both-works-answered.png"))

            # --- 2b. The same on a phone (390×844, touch), in both themes: Send says it before the press, the field
            #         keeps its height, and the message goes as the new task on the pick.
            pocket = browser.new_context(**PHONE)
            phone = pocket.new_page()
            phone_frames = _frames(phone)
            phone.goto(server["url"], wait_until="domcontentloaded")
            _capsule_ready(phone, "#chat-model-mode")
            phone_prefix = _open_project(phone, gamma)
            phone.wait_for_timeout(800)  # the room's own entrance and its row's fit settle before the field is measured
            top = phone.evaluate(_TOP, phone_prefix)
            _pick(phone, f"#{phone_prefix}-model-mode", QUEUED, "mock-queued", finger=True)  # its bottom sheet, by taps
            phone.wait_for_function(_WORD, arg=[phone_prefix, "New task"], timeout=10_000)
            phone.wait_for_timeout(800)  # the sheet's leave, likewise
            said = phone.evaluate(_SEND, phone_prefix)
            seen["phone_armed"] = {**said, "field_top_before": top, "field_top_armed": phone.evaluate(_TOP, phone_prefix)}
            assert said == {"word": "New task", "placeholder": "New task...", "value": ""}, said
            assert abs(seen["phone_armed"]["field_top_armed"] - top) <= 1, ("the armed field keeps its height on a phone", seen["phone_armed"])
            for theme in ("dark", "light"):
                _theme(phone, theme)
                phone.screenshot(path=str(evidence / f"09-phone-390x844-new-task-before-the-press-{theme}.png"))
            _theme(phone, "dark")
            _send(phone, phone_prefix, "ROOM_WIRE_PHONE", finger=True)
            _answered(phone, phone_prefix, "ROOM_WIRE_PHONE")
            assert provider.models_of("ROOM_WIRE_PHONE")[:1] == ["mock-queued"], provider.models_of("ROOM_WIRE_PHONE")
            assert re.fullmatch(r"[0-9a-f]{24}", _frame(phone_frames, "ROOM_WIRE_PHONE").get("new_work", ""))
            phone.wait_for_function(_WORD, arg=[phone_prefix, "Send"], timeout=10_000)
            phone.screenshot(path=str(evidence / "10-phone-390x844-new-task-answered-dark.png"))
            pocket.close()
            _idle(page)

            # --- 3. Two windows on one room. The other one changes the model first; this page has heard of it.
            other = windows.new_page()
            other.goto(server["url"], wait_until="domcontentloaded")
            _capsule_ready(other, "#chat-model-mode")
            other_prefix = _open_project(other, alpha)
            other_capsule = f"#{other_prefix}-model-mode"
            _pick(page, capsule, QUEUED, "mock-queued")
            page.wait_for_function(_WORD, arg=[prefix, "New task"], timeout=10_000)
            _pick(other, other_capsule, OTHER, "mock-other")
            page.wait_for_function(_VALUE, arg=[capsule, "mock-other"], timeout=30_000)  # this page hears of it
            before = len(frames)
            page.fill(f"#{prefix}-input", "ROOM_WIRE_TWO_TABS")
            page.click(f"#{prefix}-send")
            page.wait_for_function("(words) => [...document.querySelectorAll('.toast')].some((toast) => toast.innerText.includes(words))",
                                   arg=CHANGED, timeout=10_000)
            held = page.evaluate(_SEND, prefix)
            seen["held_back"] = held
            assert held == {"word": "Send", "placeholder": "Message Ouroboros...", "value": "ROOM_WIRE_TWO_TABS"}, held
            assert len(frames) == before and provider.models_of("ROOM_WIRE_TWO_TABS") == [], "nothing went on either model"
            page.screenshot(path=str(evidence / "05-two-windows-words-held-back.png"))
            page.click(f"#{prefix}-send")
            page.wait_for_function("(prefix) => document.querySelector('#' + prefix + '-input').value === ''", arg=prefix, timeout=20_000)
            _answered(page, prefix, "ROOM_WIRE_TWO_TABS")
            _idle(page)
            assert "new_work" not in _frame(frames, "ROOM_WIRE_TWO_TABS"), "pressed again it is an ordinary message"
            assert provider.models_of("ROOM_WIRE_TWO_TABS") == ["mock-other"], "of the room as it is saved now"

            # --- 4. The same, before this page hears of the change: the host refuses, the words come back.
            _pick(page, capsule, PICKED, "mock-picked")
            page.wait_for_function(_WORD, arg=[prefix, "New task"], timeout=10_000)
            page.wait_for_timeout(1500)  # the page's own read of the state after its save
            page.route("**/api/state*", lambda route: route.abort())  # this page's state reads lag behind
            _pick(other, other_capsule, QUEUED, "mock-queued")
            assert page.evaluate(_CAPSULE, capsule)["value"] == "mock-picked", "this page still shows its own pick"
            page.fill(f"#{prefix}-input", "ROOM_WIRE_REFUSED")
            page.click(f"#{prefix}-send")
            page.wait_for_function("(prefix) => document.querySelector('#' + prefix + '-messages').innerText.includes('changed elsewhere')",
                                   arg=prefix, timeout=30_000)
            page.wait_for_function("([prefix, words]) => document.querySelector('#' + prefix + '-input').value === words",
                                   arg=[prefix, "ROOM_WIRE_REFUSED"], timeout=10_000)
            back = page.evaluate(_SEND, prefix)
            seen["refused"] = back
            assert back == {"word": "Send", "placeholder": "Message Ouroboros...", "value": "ROOM_WIRE_REFUSED"}, back
            assert re.fullmatch(r"[0-9a-f]{24}", _frame(frames, "ROOM_WIRE_REFUSED").get("new_work", ""))
            assert provider.models_of("ROOM_WIRE_REFUSED") == [], "the refused message reached no model"
            page.screenshot(path=str(evidence / "06-host-refused-words-back-in-the-field.png"))
            page.unroute("**/api/state*")
            page.click(f"#{prefix}-send")
            page.wait_for_function("(prefix) => document.querySelector('#' + prefix + '-input').value === ''", arg=prefix, timeout=20_000)
            _answered(page, prefix, "ROOM_WIRE_REFUSED")
            _idle(page)
            assert "new_work" not in _frame(frames, "ROOM_WIRE_REFUSED"), "the next Send is an ordinary one, never the refused intent"
            assert provider.models_of("ROOM_WIRE_REFUSED") == ["mock-queued"], provider.models_of("ROOM_WIRE_REFUSED")
            page.wait_for_function(_VALUE, arg=[capsule, "mock-queued"], timeout=30_000)

            # --- 5. Replies. A pick that no longer stands holds no Reply back: the host gives it to its work.
            _pick(page, capsule, OTHER, "mock-other")
            page.wait_for_function(_WORD, arg=[prefix, "New task"], timeout=10_000)
            _pick(other, other_capsule, PICKED, "mock-picked")
            page.wait_for_function(_VALUE, arg=[capsule, "mock-picked"], timeout=30_000)
            toasts = page.evaluate("() => document.querySelectorAll('.toast').length")
            assert _reply(page, prefix, "assistant", "WIRE_OK ROOM_WIRE_BUSY_B"), "no Reply on the work's answer"
            page.wait_for_selector(f"#{prefix}-input-area .chat-quote, #{prefix}-input-area [id$='-reply']", timeout=10_000)
            replying = {**page.evaluate(_SEND, prefix), "reading": page.evaluate(_READING, prefix)}
            seen["replying_on_a_stale_pick"] = replying
            page.screenshot(path=str(evidence / "07-reply-on-a-stale-pick.png"))
            # The page knows a row inside a work's card is that work's ('0'); any other row is the host's to place.
            assert replying["word"] == "Send" and replying["reading"] in ("0", "reply"), replying
            before = len(frames)
            page.fill(f"#{prefix}-input", "ROOM_WIRE_REPLY_WORK")
            page.click(f"#{prefix}-send")
            page.wait_for_function("(prefix) => document.querySelector('#' + prefix + '-input').value === ''", arg=prefix, timeout=20_000)
            assert len(frames) == before + 1 and re.match(r"(row|task):", frames[-1].get("reply_to", "")), "the Reply went at once"
            assert ("new_work" in frames[-1]) == (replying["reading"] == "reply"), frames[-1]
            _answered(page, prefix, "ROOM_WIRE_REPLY_WORK")
            _idle(page)
            assert page.evaluate("() => document.querySelectorAll('.toast').length") <= toasts, "nothing was said against the Reply"
            assert page.evaluate(_SEND, prefix)["value"] == "", "the host took the Reply: nothing came back to the field"
            assert not page.evaluate("(prefix) => document.querySelector('#' + prefix + '-messages').innerText.split('ROOM_WIRE_REPLY_WORK')"
                                     ".at(-1).includes('changed elsewhere')", prefix), "and refused nothing"
            seen["reply_models"] = provider.models_of("ROOM_WIRE_REPLY_WORK")
            assert seen["reply_models"], "the Reply reached a model"
            # A quote of the owner's own words is a plain quote: with a pick that stands it is the new task.
            _pick(page, capsule, QUEUED, "mock-queued")
            page.wait_for_function(_WORD, arg=[prefix, "New task"], timeout=10_000)
            assert _reply(page, prefix, "user", "ROOM_WIRE_BUSY_B"), "no Reply on the owner's own message"
            page.wait_for_function("(prefix) => document.querySelector('#' + prefix + '-input').placeholder.startsWith('Reply or new task')",
                                   arg=prefix, timeout=10_000)
            quoting = {**page.evaluate(_SEND, prefix), "reading": page.evaluate(_READING, prefix)}
            seen["quoting_own_words"] = quoting
            assert quoting == {"word": "Send", "placeholder": "Reply or new task...", "value": "",
                               "reading": "reply"}, quoting
            page.fill(f"#{prefix}-input", "ROOM_WIRE_PLAIN_QUOTE")
            page.click(f"#{prefix}-send")
            page.wait_for_function("(prefix) => document.querySelector('#' + prefix + '-input').value === ''", arg=prefix, timeout=20_000)
            _answered(page, prefix, "ROOM_WIRE_PLAIN_QUOTE")
            _idle(page)
            quoted = _frame(frames, "ROOM_WIRE_PLAIN_QUOTE")
            assert quoted.get("reply_to", "").startswith("row:") and re.fullmatch(r"[0-9a-f]{24}", quoted.get("new_work", "")), quoted
            assert provider.models_of("ROOM_WIRE_PLAIN_QUOTE")[:1] == ["mock-queued"], provider.models_of("ROOM_WIRE_PLAIN_QUOTE")
            page.wait_for_function(_WORD, arg=[prefix, "Send"], timeout=10_000)
            page.screenshot(path=str(evidence / "08-plain-quote-started-the-new-task.png"))
        finally:
            provider.release_all()
            (evidence / "new-work-seen.json").write_text(json.dumps(seen, indent=2, default=str), encoding="utf-8")
            (evidence / "new-work-models.json").write_text(json.dumps(
                [{"model": body.get("model"), "marker": provider.marker_of(body)} for body in provider.received]), encoding="utf-8")
            (evidence / "new-work-frames.json").write_text(json.dumps(
                [{key: frame.get(key) for key in ("content", "chat_id", "project_id", "reply_to", "new_work") if key in frame}
                 for frame in frames], indent=1), encoding="utf-8")
            (evidence / "new-work-writes.json").write_text(json.dumps(writes, indent=1), encoding="utf-8")
            try:
                page.screenshot(path=str(evidence / "new-work-last.png"))
                (evidence / "new-work-page.txt").write_text(page.locator("body").inner_text(), encoding="utf-8")
            except Exception:  # noqa: BLE001 - diagnostics only: the page may be gone
                pass
            browser.close()
