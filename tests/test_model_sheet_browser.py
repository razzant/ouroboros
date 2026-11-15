"""The Model sheet on the real server: a running work's tier, pins across windows, the keyboard, a finger, the Main
chat's confirm and tier, and a paused work without a model control.

Owner 12.11 «A» (docs/DESIGN.md "Composer owner controls", "A work's model on its card"): a press on the Model capsule
or on a work's model fact opens one floating sheet, a bottom sheet in the stack layout: the field, Main, Pinned,
Recent, the sources, and at its foot the processing tier strip. The real server on its own port and data root, the
real page, and the loopback provider of tests/_model_choice_browser.py: no paid call, no account, no key, and nothing
of the server answered by the test. The loopback openai-compatible transport is sent no tier, so every call's receipt
says `processing_not_submitted` and a card names what it asked beside how it was sent.
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
    _SHEET,
    MAIN,
    NEXT_STEP,
    OFFERED,
    OTHER,
    ROOM_SCOPE_OWN,
    SOURCE,
    WORK_SCOPE_OWN,
    Provider,
    _capsule_ready,
    _close_project,
    _close_sheet,
    _held_focus,
    _open_project,
    _open_sheet,
    _post,
    _preferences,
    _row,
    _send,
    _state,
    _work_of,
)
from tests.test_effort_range_browser import _launch
from tests.test_ui_smoke_playwright import direct_server_with_data  # noqa: F401 -- fixture

pytestmark = [pytest.mark.ui_browser]  # its server, data root and provider are its own: nothing on the host is shared

DESK = {"width": 1280, "height": 850}
PHONE = {"viewport": {"width": 390, "height": 844}, "is_mobile": True, "has_touch": True}
MAIN_CAPSULE = "#chat-model-mode"
MARK = "ROOM_WIRE_TIER"
_SEG = '.ui-popup.model-sheet .model-tier-seg[data-tier="{}"]'
_PINNED = "() => [...document.querySelectorAll('.ui-popup.model-sheet .chat-model-row[data-pinned=\"true\"]')].map((row) => row.dataset.row)"
# The tier words after a work's model on its card: the words alone (the line's separator rides at their end).
_CARD_TIER = """(sel) => { const node = document.querySelector(sel + ' .card-model-tier');
    return node && !node.hidden ? node.querySelector('.card-model-words').innerText.replace(/\\s+/g, ' ').trim() : ''; }"""
_PINS = """() => [...document.querySelectorAll('.ui-popup.model-sheet .chat-model-pin')].filter((pin) => pin.getClientRects().length)
    .map((pin) => { const box = pin.getBoundingClientRect(); return { row: pin.closest('.chat-model-row').dataset.row,
        width: box.width, height: box.height, opacity: getComputedStyle(pin).opacity }; })"""
# The card's Details line of its tier and its newest call's whole receipt (card_model.js, chat_activity.js).
_PROCESSING = """(sel) => { const fact = [...document.querySelectorAll(sel + ' .chat-live-fact')]
    .find((row) => row.querySelector('dt')?.textContent === 'Processing');
    return fact ? fact.querySelector('dd').innerText.replace(/\\s+/g, ' ').trim() : null; }"""
_COMPOSER = """(prefix) => { const box = (sel) => { const r = document.querySelector(sel).getBoundingClientRect();
    return [r.left, r.top, r.width, r.height].map((v) => Math.round(v * 10) / 10); };
    return { area: box('#' + prefix + '-input-area'), field: box('#' + prefix + '-input') }; }"""


def _route(model: str) -> dict:
    return {"model": model, "use_local": False}


def _serve(server: dict, monkeypatch) -> Provider:
    """The harness server again, on the loopback provider's models, with Main, Light and a fallback of its own."""
    server["stop_server"]()
    settings_path = server["data_dir"] / "settings.json"
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    settings.update(OUROBOROS_MODEL=MAIN, OUROBOROS_MODEL_LIGHT=f"{SOURCE}::mock-light",
                    OUROBOROS_MODEL_FALLBACKS=f"{SOURCE}::mock-fallback",
                    OUROBOROS_TASK_REVIEW_MODE="off", OUROBOROS_SAFETY_MODE="off",
                    OUROBOROS_SUBAGENTS=json.dumps({"enabled": False, "items": []}))
    settings_path.write_text(json.dumps(settings), encoding="utf-8")
    provider = Provider()
    # Plain functions, so the handler class binds each request's handler as their argument.
    monkeypatch.setattr(fixtures_mock_llm._Handler, "do_POST", lambda handler: provider.post(handler))
    monkeypatch.setattr(fixtures_mock_llm._Handler, "do_GET", lambda handler: provider.get(handler), raising=False)
    server["start_server"]()
    return provider


def _record(page) -> list:
    """Every owner write the page sends: the chat's model and tier, the pins, a work's model act, a Pause."""
    writes: list = []
    pattern = re.compile(r"/api/(owner/chat-(?:model|processing)|ui/preferences|tasks/[^/]+/(?:model|pause))$")
    page.on("request", lambda request: writes.append((pattern.search(request.url).group(1), json.loads(request.post_data or "{}")))
            if request.method == "POST" and pattern.search(request.url) else None)
    return writes


def _pin_write(response) -> bool:
    """A pin's own write: the preferences also take the page's read cursors, which are not pins."""
    return (response.url.endswith("/api/ui/preferences") and response.request.method == "POST"
            and "model_pins" in (response.request.post_data or ""))


def _sent(writes: list, path: str, key: str | None = None) -> list:
    """The bodies posted to one path (the preferences' read cursors are not pins: `key` names what counts)."""
    return [body for where, body in writes if where == path and (key is None or key in body)]


def _load(page, url: str) -> None:
    page.goto(url, wait_until="domcontentloaded", timeout=30_000)
    _capsule_ready(page, MAIN_CAPSULE)
    # WebKit measures the textarea on its next frame; catalogue readiness can precede that
    # 33.2 -> 33 px startup adjustment. Compare sheet motion only once the field is sized.
    page.wait_for_function("""() => { const field = document.querySelector('#chat-input');
        return field?.dataset.sizing === 'content' || Boolean(field?.style.getPropertyValue('--chat-input-height')); }""")


def _said(page, words: str) -> None:
    page.wait_for_function("(words) => [...document.querySelectorAll('.toast')].some((toast) => toast.innerText.includes(words))",
                           arg=words, timeout=15_000)


def _shot(page, evidence: pathlib.Path, name: str) -> None:
    page.screenshot(path=str(evidence / f"{name}.png"), animations="disabled")


def _evidence(tmp_path) -> pathlib.Path:
    evidence = pathlib.Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", str(tmp_path)))
    evidence.mkdir(parents=True, exist_ok=True)
    return evidence


def _wait_work(page, task: str, ready, what: str, timeout: float = 60) -> dict:
    """The work's row of `/api/state` once `ready(work)` holds."""
    work = None
    for _ in range(int(timeout * 4)):
        work = _work_of(_state(page), task)
        if work is not None and ready(work):
            return work
        page.wait_for_timeout(250)
    raise AssertionError((what, work))


def _usage(data_dir: pathlib.Path, task: str) -> list:
    """The work's own model calls the supervisor recorded (``logs/events.jsonl`` ``llm_usage``), in their order. A
    call's receipt becomes the work's ``processing.last`` in the same handler, before its row is written."""
    log = pathlib.Path(data_dir) / "logs" / "events.jsonl"
    rows = []
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines() if log.exists() else []:
        if '"llm_usage"' not in line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get("type") == "llm_usage" and row.get("task_id") == task and "mock-light" not in str(row.get("model")):
            rows.append(row)
    return rows


def _wait_usage(page, data_dir: pathlib.Path, task: str, count: int, timeout: float = 60) -> list:
    for _ in range(int(timeout * 4)):
        rows = _usage(data_dir, task)
        if len(rows) >= count:
            return rows
        page.wait_for_timeout(250)
    raise AssertionError((task, f"{count} recorded calls expected", _usage(data_dir, task)))


def _receipt(work: dict) -> tuple:
    last = (work.get("processing") or {}).get("last") or {}
    return last.get("requested"), last.get("submitted"), last.get("observed"), last.get("reason")


# ---------------------------------------------------------------------------------------------------------------------
# (a) A running work's tier switches at its next call; (f) paused, it has no model control
# ---------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_a_works_tier_switches_at_its_next_call_and_a_paused_work_has_no_model_control(
        direct_server_with_data, monkeypatch, engine, tmp_path):  # noqa: F811
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import sync_playwright

    server = direct_server_with_data
    provider = _serve(server, monkeypatch)
    evidence = _evidence(tmp_path)
    seen: dict = {}
    with sync_playwright() as pw:
        browser, page = _launch(pw, engine, viewport=DESK)
        writes = _record(page)
        try:
            _load(page, server["url"])
            room = _post(page, "/api/projects", {"name": "Fixture · Tiers"})["project"]
            prefix = _open_project(page, room)
            capsule = f"#{prefix}-model-mode"

            # The room's own tier, an explicit Standard (so each call of its work leaves a receipt): a press posts at
            # once, asks nothing first and leaves the sheet open; the capsule never names Standard.
            _open_sheet(page, capsule)
            with page.expect_response(lambda r: r.url.endswith("/api/owner/chat-processing")) as reply:
                page.click(_SEG.format("standard"))
            answer = reply.value.json()
            assert answer["ok"] is True and answer["processing_choice"] == "standard", answer
            assert _sent(writes, "owner/chat-processing") == [{"processing": "standard", "project_id": room["id"]}], writes
            page.wait_for_function("(sel) => document.querySelector(sel)?.getAttribute('aria-checked') === 'true'",
                                   arg=_SEG.format("standard"), timeout=10_000)
            sheet = page.evaluate(_SHEET)
            seen["room_sheet"] = sheet
            assert sheet["chosen"] == "standard" and sheet["reset"] and not page.query_selector(".confirm-dialog"), sheet
            assert sheet["scope"] == f"Standard · {ROOM_SCOPE_OWN}", sheet
            closed = page.evaluate(_CAPSULE, capsule)
            assert (closed["tier"], closed["dataTier"]) == ("", "standard"), ("Standard is never named on the capsule", closed)
            _close_sheet(page)
            assert next(p for p in _state(page)["projects"] if p["id"] == room["id"])["processing_choice"] == "standard"

            # A held work in the room: its first call answers with one tool, its second waits at the provider.
            provider.hold(MARK)
            _send(page, prefix, MARK)
            assert provider.held(MARK), "the room's work never reached its held call"
            card = f"#{prefix}-messages .chat-live-card"
            page.wait_for_selector(f"{card} button.card-model-fact", timeout=60_000)
            task = page.get_attribute(card, "data-task-id")
            first = _wait_work(page, task, lambda work: _receipt(work)[0], "the first call's receipt")
            seen["first"] = first
            assert {key: first["processing"].get(key) for key in ("choice", "source")} == {"choice": "standard", "source": "room"}, first
            assert _receipt(first) == ("standard", None, "unknown", "processing_not_submitted"), first
            recorded = len(_wait_usage(page, server["data_dir"], task, 1))
            in_flight = len(provider.models_of(MARK))
            assert in_flight == 2, ("one call answered, the next one in flight", provider.models_of(MARK))
            page.wait_for_timeout(500)
            seen["standard_words"] = page.evaluate(_CARD_TIER, card)

            # Fast, pressed on the card while that call is in flight: one act, applied from the next call.
            page.click(f"{card} button.card-model-fact")
            page.wait_for_selector(".ui-popup.model-sheet .card-model-head", timeout=20_000)
            sheet = page.evaluate(_SHEET)
            seen["card_sheet"] = sheet
            assert sheet["offered"] == OFFERED and sheet["chosen"] == "standard", ("no Ultrafast where the source lists none", sheet)
            assert sheet["scope"].startswith(f"Standard · {WORK_SCOPE_OWN}"), sheet
            seen["standard_scope"] = sheet["scope"]
            _shot(page, evidence, f"tier-card-sheet-{engine}")
            with page.expect_response(lambda r: r.url.endswith(f"/api/tasks/{task}/model") and r.request.method == "POST") as reply:
                page.click(_SEG.format("fast"))
            act = json.loads(reply.value.request.post_data)
            answer = reply.value.json()
            seen["act"], seen["answer"] = act, answer
            assert act.pop("request_id") and act == {"action": "processing", "processing": "fast"}, act
            assert reply.value.status == 200 and answer["applied"] == "next_step", answer
            assert answer["processing"] == {"choice": "fast", "source": "owner", "resolved": "fast"}, answer
            _said(page, f"Fast for this work. {NEXT_STEP}")
            page.wait_for_function("(sel) => document.querySelector(sel)?.getAttribute('aria-checked') === 'true'",
                                   arg=_SEG.format("fast"), timeout=10_000)
            assert page.query_selector(".ui-popup.model-sheet .card-model-head"), "a tier leaves the sheet open"
            strip_at = "() => document.querySelector('.ui-popup.model-sheet .model-tier').getBoundingClientRect().top"
            page.wait_for_timeout(300)  # the press's own repaint settles; from here only the work's calls move facts
            seen["strip_top_before_receipt"] = page.evaluate(strip_at)
            assert len(provider.models_of(MARK)) == in_flight, "nothing is sent again: the call in flight runs on"
            assert not _sent(writes, "owner/chat-model"), "a tier moves no model"

            # The call in flight finishes on Standard ...
            provider.advance(MARK)
            assert provider.held(MARK), "the work never reached its next call"
            assert len(provider.models_of(MARK)) == in_flight + 1
            _wait_usage(page, server["data_dir"], task, recorded + 1)
            second = _work_of(_state(page), task)
            seen["second"] = second
            assert second["processing"]["choice"] == "fast" and _receipt(second)[0] == "standard", \
                ("the call in flight when Fast was pressed finished on Standard", second)
            # ... and the next one asks Fast; this transport sends no tier, and the card says so.
            provider.advance(MARK)
            assert provider.held(MARK), "the work never reached the call after its first Fast one"
            _wait_usage(page, server["data_dir"], task, recorded + 2)
            third = _work_of(_state(page), task)
            seen["third"] = third
            assert _receipt(third) == ("fast", None, "unknown", "processing_not_submitted"), ("the next call asked Fast", third)
            assert {key: third["processing"].get(key) for key in ("choice", "source")} == {"choice": "fast", "source": "owner"}, third
            assert not any("service_tier" in body or "speed" in body for body in provider.received), "the loopback transport is sent no tier"
            page.wait_for_function(f"(sel) => ({_CARD_TIER})(sel) === 'Fast · Fast asked · sent as Standard'", arg=card, timeout=30_000)
            # The open sheet says it in its head, above the field: the strip at its foot keeps its words and place.
            page.wait_for_function("(words) => document.querySelector('.ui-popup.model-sheet .card-model-served')?.innerText === words",
                                   arg="Fast asked · sent as Standard", timeout=30_000)
            seen["strip_top_after_receipt"] = page.evaluate(strip_at)
            assert seen["strip_top_after_receipt"] == seen["strip_top_before_receipt"], \
                ("the receipt filled its held line: the strip at the sheet's foot did not move", seen)
            seen["fast_sheet"] = page.evaluate(_SHEET)
            assert seen["fast_sheet"]["scope"] == f"Fast · {WORK_SCOPE_OWN}", seen["fast_sheet"]
            _shot(page, evidence, f"tier-card-fast-{engine}")
            # The whole receipt (asked, sent, served) is the card's Details line, not its meta line's.
            _close_sheet(page)
            page.click(f"{card} > [data-live-summary-button] [data-live-disclosure]")
            page.wait_for_function(f"(sel) => Boolean(({_PROCESSING})(sel))", arg=card, timeout=15_000)
            seen["processing_detail"] = page.evaluate(_PROCESSING, card)
            assert seen["processing_detail"] == "Fast · last call: asked Fast · sent with no tier · served as not reported", seen
            page.click(f"{card} button.card-model-fact")
            page.wait_for_selector(".ui-popup.model-sheet .card-model-head", timeout=20_000)

            # (f) Paused: the server offers no model control, so the card offers none, and its open sheet goes with it.
            assert page.query_selector(".ui-popup.model-sheet .card-model-head"), "the card's sheet is still open"
            paused = page.evaluate("""async ([task, id]) => { const r = await fetch(`/api/tasks/${task}/pause`, { method: 'POST',
                headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ request_id: id, actor: 'owner' }) });
                return { status: r.status, body: await r.json() }; }""", [task, f"model-sheet-pause-{engine}"])
            seen["paused"] = paused
            assert paused["status"] in (200, 202) and paused["body"]["ok"] is True, paused
            work = _wait_work(page, task, lambda work: work["activity"] == "paused", "the work pauses")
            seen["paused_work"] = work
            assert not [control for control in work["controls"] if control["action"] == "model"], work["controls"]
            page.wait_for_function("(sel) => document.querySelector(sel + ' [data-live-phase]')?.innerText.includes('Paused')",
                                   arg=card, timeout=30_000)
            page.wait_for_function("(sel) => !document.querySelector(sel + ' button.card-model-fact')", arg=card, timeout=30_000)
            page.wait_for_function("() => !document.querySelector('.ui-popup.model-sheet')", timeout=10_000)
            _shot(page, evidence, f"tier-card-paused-{engine}")
            assert not _sent(writes, "owner/chat-model") and len(_sent(writes, f"tasks/{task}/model")) == 1, writes
            # DESIGN "A work's model on its card": a tier other than Standard is named, and what the newest call was
            # served as only when it was served otherwise. Standard sent with no tier is the ordinary service.
            assert (seen["standard_words"], seen["standard_scope"]) == ("", f"Standard · {WORK_SCOPE_OWN}"), \
                ("an explicit Standard sent with no tier is no deviation", seen["standard_words"], seen["standard_scope"])
        finally:
            provider.release_all()
            (evidence / f"tier-seen-{engine}.json").write_text(json.dumps(seen, indent=1, default=str), encoding="utf-8")
            (evidence / f"tier-writes-{engine}.json").write_text(json.dumps(writes, indent=1), encoding="utf-8")
            browser.close()


# ---------------------------------------------------------------------------------------------------------------------
# (b) A pin reaches every window, survives a reload and chooses nothing
# ---------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_a_pin_reaches_every_window_survives_a_reload_and_chooses_nothing(direct_server_with_data, monkeypatch, engine, tmp_path):  # noqa: F811
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import sync_playwright

    server = direct_server_with_data
    _serve(server, monkeypatch)
    evidence = _evidence(tmp_path)
    other = _route(OTHER)
    with sync_playwright() as pw:
        browser, one = _launch(pw, engine, viewport=DESK)
        two = browser.new_context(viewport=DESK).new_page()  # a second window on the same server
        writes = {"one": _record(one), "two": _record(two)}
        try:
            for page in (one, two):
                _load(page, server["url"])
            room = _post(one, "/api/projects", {"name": "Pins"})["project"]
            for page in (one, two):
                _open_project(page, room)
            capsule = f"#pchat-{room['id']}-model-mode"
            for page in (one, two):
                _open_sheet(page, capsule)
                assert page.evaluate(_PINNED) == [], "nothing is pinned yet"

            # One window pins with the mouse: an unpinned row's Pin shows on the row's hover.
            one.hover(f"{_row(OTHER)} .chat-model-cell")
            one.wait_for_function("(sel) => getComputedStyle(document.querySelector(sel)).opacity === '1'",
                                  arg=f"{_row(OTHER)} .chat-model-pin", timeout=5_000)
            with one.expect_response(_pin_write) as reply:
                one.click(f"{_row(OTHER)} .chat-model-pin")
            assert json.loads(reply.value.request.post_data) == {"model_pins": {"pin": [other]}}
            assert reply.value.json()["model_pins"] == [other], reply.value.json()
            one.wait_for_function(f"() => ({_PINNED})().includes('route|{OTHER}')", timeout=10_000)
            sheet = one.evaluate(_SHEET)
            assert sheet["headings"][0] == "Pinned" and [row["id"] for row in sheet["groups"][1]["rows"]] == ["mock-other"], sheet
            # The other window: without a reload (ui_preferences_changed over its socket), its open sheet lists it too.
            two.wait_for_function(f"() => ({_PINNED})().includes('route|{OTHER}')", timeout=20_000)
            _shot(two, evidence, f"pins-second-window-{engine}")
            assert one.query_selector(".ui-popup.model-sheet") and two.query_selector(".ui-popup.model-sheet"), "a Pin leaves both sheets open"

            # A Pin chooses nothing: no model write, the room still follows Main, no recent model, both capsules as they were.
            assert not _sent(writes["one"], "owner/chat-model") and not _sent(writes["two"], "owner/chat-model"), writes
            assert next(p for p in _state(one)["projects"] if p["id"] == room["id"])["model_choice"] == {"kind": "main"}
            prefs = _preferences(one)
            assert prefs["model_pins"] == [other] and prefs["model_recents"] == [], prefs
            for page in (one, two):
                capsule_now = page.evaluate(_CAPSULE, capsule)
                assert (capsule_now["name"], capsule_now["value"], capsule_now["own"]) == ("Main", "mock-main", "false"), capsule_now

            # A reload keeps it, and the Main chat lists it too: a pin is the owner's, for every room.
            one.reload(wait_until="domcontentloaded")
            _capsule_ready(one, MAIN_CAPSULE)
            if one.evaluate("() => document.querySelector('#project-panel-backdrop')?.classList.contains('open')"):
                _close_project(one)  # a reload may bring the room back over Main
            _open_sheet(one, MAIN_CAPSULE)
            assert one.evaluate(_PINNED) == [f"route|{OTHER}"], "the Main chat's sheet after a reload"
            _close_sheet(one)
            _open_project(one, room)
            _open_sheet(one, capsule)
            assert one.evaluate(_PINNED) == [f"route|{OTHER}"], "the room's sheet after a reload"

            # The second window unpins; the first sees it go without a reload.
            with two.expect_response(_pin_write) as reply:
                two.click(f"{_row(OTHER)} .chat-model-pin")
            assert json.loads(reply.value.request.post_data) == {"model_pins": {"unpin": [other]}}
            assert reply.value.json()["model_pins"] == [], reply.value.json()
            two.wait_for_function(f"() => ({_PINNED})().length === 0", timeout=10_000)
            one.wait_for_function(f"() => ({_PINNED})().length === 0", timeout=20_000)
            assert "Pinned" not in one.evaluate(_SHEET)["headings"]
            assert _preferences(two)["model_pins"] == [], _preferences(two)
            assert not _sent(writes["one"], "owner/chat-model") and not _sent(writes["two"], "owner/chat-model"), writes
        finally:
            (evidence / f"pins-writes-{engine}.json").write_text(json.dumps(writes, indent=1), encoding="utf-8")
            browser.close()


# ---------------------------------------------------------------------------------------------------------------------
# (e) the Main chat, (d) a finger, (c) the keyboard
# ---------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_the_main_chat_a_finger_and_the_keyboard(direct_server_with_data, monkeypatch, engine, tmp_path):  # noqa: F811
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import sync_playwright

    server = direct_server_with_data
    _serve(server, monkeypatch)
    evidence = _evidence(tmp_path)
    # Safari's Tab reaches only fields unless "Press Tab to highlight each item" is on; Option+Tab reaches every control.
    tab, back = ("Alt+Tab", "Alt+Shift+Tab") if engine == "webkit" else ("Tab", "Shift+Tab")
    seen: dict = {}
    with sync_playwright() as pw:
        browser, page = _launch(pw, engine, viewport=DESK)
        writes = _record(page)
        try:
            _load(page, server["url"])

            # --- (e) The Main chat: a pick of another model asks first; a tier posts at once, with no dialog.
            _open_sheet(page, MAIN_CAPSULE)
            sheet = page.evaluate(_SHEET)
            assert [(row["row"], row["id"], row["tag"], row["selected"]) for row in sheet["groups"][0]["rows"]] == \
                [("main", "mock-main", "Main", True)], ("the Main chat's sheet starts with Main itself", sheet)
            page.click(f"{_row(OTHER)} .chat-model-cell")
            page.wait_for_selector(".confirm-dialog [data-main-followers]", timeout=20_000)
            assert page.inner_text(".confirm-dialog h3") == "Change Main" and not page.query_selector(".ui-popup.model-sheet")
            page.click(".confirm-dialog [data-confirm-cancel]")
            page.wait_for_function("() => !document.querySelector('.confirm-dialog')")
            assert not _sent(writes, "owner/chat-model") and _state(page)["main_route"] == _route(MAIN), "a cancelled sheet writes nothing"

            for tier, word in (("fast", "Fast"), ("economy", "Flex"), ("", "")):
                _open_sheet(page, MAIN_CAPSULE)
                assert page.evaluate(_SHEET)["offered"] == OFFERED, "Main's strip offers Flex, Standard and Fast, never Ultrafast"
                with page.expect_response(lambda r: r.url.endswith("/api/owner/chat-processing")) as reply:
                    page.click(_SEG.format(tier) if tier else ".ui-popup.model-sheet .model-tier-default")
                answer = reply.value.json()
                assert answer["ok"] is True and answer["project_id"] == "" and answer["main_processing"] == tier, answer
                page.wait_for_function("(tier) => (document.querySelector('.ui-popup.model-sheet .model-tier-seg[aria-checked=\"true\"]')?.dataset.tier || '') === tier",
                                       arg=tier, timeout=10_000)
                page.wait_for_timeout(300)
                sheet = page.evaluate(_SHEET)
                seen[f"main_tier_{tier or 'default'}"] = sheet
                assert not page.query_selector(".confirm-dialog") and sheet is not None, ("a tier asks nothing first and leaves the sheet open", tier)
                assert sheet["scope"] == f"{word or 'Default · Standard'} · for Main and its new subagents; no running work follows it now", sheet
                assert sheet["reset"] is bool(tier), ("Default stands beside the strip only while there is a choice to leave", sheet)
                _close_sheet(page)
                closed = page.evaluate(_CAPSULE, MAIN_CAPSULE)
                assert (closed["tier"], closed["dataTier"]) == (word, tier or "default"), ("the closed capsule names a tier other than Standard", closed)
                assert closed["focused"], ("Escape gives the focus back to the capsule", closed)
                assert _state(page)["processing_tiers"]["main"] == tier
                if tier == "economy":
                    _shot(page, evidence, f"main-capsule-flex-{engine}")
            assert _sent(writes, "owner/chat-processing") == [{"processing": "fast"}, {"processing": "economy"}, {"processing": ""}], writes
            assert not _sent(writes, "owner/chat-model"), "a tier moves no model"

            # --- (d) A finger: a 390 px phone. A tap opens a bottom sheet over a scrim and raises no keyboard; every
            #     Pin is a finger's size and shows; a Pin leaves the sheet open; the field takes focus on its own tap.
            phone = browser.new_context(**PHONE).new_page()
            phone_writes = _record(phone)
            _load(phone, server["url"])
            assert phone.evaluate("() => [document.documentElement.dataset.layout, document.documentElement.dataset.pointer]") == ["stack", "coarse"]
            composer = phone.evaluate(_COMPOSER, "chat")
            phone.tap(f"{MAIN_CAPSULE} .chat-model-head")
            phone.wait_for_selector(".ui-popup.model-sheet .chat-model-row", timeout=20_000)
            phone.wait_for_timeout(600)
            sheet = phone.evaluate(_SHEET)
            seen["phone_sheet"] = sheet
            assert sheet["bottom"] and sheet["scrim"] and sheet["inside"], ("a bottom sheet over a scrim, on screen", sheet)
            assert sheet["focus"] != "field", ("opening under a finger raises no keyboard", sheet)  # where it is: _held_focus
            assert phone.evaluate(_COMPOSER, "chat") == composer, "the composer does not move"
            pins = phone.evaluate(_PINS)
            seen["phone_pins"] = pins
            assert pins and all(pin["width"] >= 44 and pin["height"] >= 44 and pin["opacity"] == "1" for pin in pins), \
                ("under a finger every Pin shows, at least 44 px each way", pins)
            _shot(phone, evidence, f"phone-sheet-{engine}")
            # The bottom sheet is modal: a hardware keyboard's Tab and Shift+Tab stay in it; Escape closes it and gives
            # the focus back to the capsule; a tap opens it again, again without the field's focus.
            seen["phone_held"] = _held_focus(phone, tab, back)
            if engine == "webkit":
                phone.focus(".ui-popup.model-sheet .chat-model-query")  # Safari's own Tab, from the field: fields only
                seen["phone_held_plain_tab"] = _held_focus(phone, "Tab", "Shift+Tab", finger=False, every_stop=False)
            phone.keyboard.press("Escape")
            phone.wait_for_function("() => !document.querySelector('.ui-popup.model-sheet') && !document.querySelector('.model-sheet-scrim')",
                                    timeout=10_000)
            assert phone.evaluate(_CAPSULE, MAIN_CAPSULE)["focused"], "Escape gives the focus back to the capsule"
            phone.tap(f"{MAIN_CAPSULE} .chat-model-head")
            phone.wait_for_selector(".ui-popup.model-sheet .chat-model-row", timeout=20_000)
            phone.wait_for_timeout(600)
            assert phone.evaluate(_SHEET)["focus"] == "sheet", ("opened again by a finger, still no field focus", phone.evaluate(_SHEET))
            with phone.expect_response(_pin_write) as reply:
                phone.tap(f"{_row(OTHER)} .chat-model-pin")
            assert json.loads(reply.value.request.post_data) == {"model_pins": {"pin": [_route(OTHER)]}}
            phone.wait_for_function(f"() => ({_PINNED})().includes('route|{OTHER}')", timeout=10_000)
            assert phone.query_selector(".ui-popup.model-sheet"), "a Pin leaves the sheet open"
            with phone.expect_response(_pin_write) as reply:
                phone.tap(f"{_row(OTHER)} .chat-model-pin")  # and a second tap takes it back
            assert json.loads(reply.value.request.post_data) == {"model_pins": {"unpin": [_route(OTHER)]}}
            phone.wait_for_function(f"() => ({_PINNED})().length === 0", timeout=10_000)
            phone.tap(".ui-popup.model-sheet .chat-model-query")
            assert phone.evaluate(_SHEET)["focus"] == "field", "the field takes the focus on its own tap"
            # The dismissing tap closes the sheet and reaches nothing beneath the scrim (here the header's controls).
            phone.evaluate("() => { window.__clicks = []; document.addEventListener('click', (event) => window.__clicks.push("
                           "event.target.className || event.target.tagName), true); }")
            beneath = phone.evaluate("() => document.elementsFromPoint(20, 20).slice(0, 4).map((node) => node.className || node.tagName)")
            phone.tap(".model-sheet-scrim", position={"x": 20, "y": 20})
            phone.wait_for_function("() => !document.querySelector('.ui-popup.model-sheet') && !document.querySelector('.model-sheet-scrim')",
                                    timeout=10_000)
            phone.wait_for_timeout(400)
            seen["phone_dismiss"] = {"beneath": beneath, "clicks": phone.evaluate("() => window.__clicks")}
            assert seen["phone_dismiss"]["clicks"] == ["model-sheet-scrim"], ("the tap reached what lies beneath", seen["phone_dismiss"])
            assert phone.evaluate("() => document.querySelector('[data-mobile-nav-toggle]')?.getAttribute('aria-expanded')") != "true"
            # In a room the sheet is on top: Back closes it and leaves the room open; a second Back closes the room.
            phone_room = _post(phone, "/api/projects", {"name": "Thumb"})["project"]
            phone_prefix = _open_project(phone, phone_room)
            phone.tap(f"#{phone_prefix}-model-mode .chat-model-head")
            phone.wait_for_selector(".ui-popup.model-sheet .chat-model-row", timeout=20_000)
            assert phone.evaluate("() => document.querySelector('.ui-popup.model-sheet').getAttribute('aria-modal')") == "true"
            phone.evaluate("() => history.back()")
            phone.wait_for_function("() => !document.querySelector('.ui-popup.model-sheet') && !document.querySelector('.model-sheet-scrim')",
                                    timeout=10_000)
            phone.wait_for_timeout(400)
            assert phone.evaluate("() => document.querySelector('#project-panel-backdrop').classList.contains('open')"), \
                "Back closed the sheet over the room, not the room under it"
            phone.evaluate("() => history.back()")
            phone.wait_for_function("() => !document.querySelector('#project-panel-backdrop').classList.contains('open')", timeout=20_000)
            assert not _sent(phone_writes, "owner/chat-model") and not _sent(phone_writes, "owner/chat-processing"), phone_writes
            phone.context.close()

            # --- (c) The keyboard, in a room: Enter on the capsule opens it in the field; Tab reaches the tier strip,
            #     whose arrows move without choosing; Enter chooses; back in the field Down and Right reach a row's Pin
            #     and Enter toggles it; Escape closes and gives the focus back to the capsule.
            # What the keyboard finds wrong is collected and asserted at the end, so the rest of the walk still runs.
            problems: list = []
            room = _post(page, "/api/projects", {"name": "Keys"})["project"]
            prefix = _open_project(page, room)
            capsule = f"#{prefix}-model-mode"
            page.focus(f"{capsule} .chat-model-head")
            page.keyboard.press("Enter")
            page.wait_for_selector(".ui-popup.model-sheet .chat-model-row", timeout=20_000)
            page.wait_for_function("() => !document.querySelector('.ui-popup.model-sheet .chat-model-note')", timeout=20_000)
            sheet = page.evaluate(_SHEET)
            assert sheet["focus"] == "field" and page.evaluate(_CAPSULE, capsule)["expanded"] == "true", sheet
            page.keyboard.press(tab)
            sheet = page.evaluate(_SHEET)
            assert sheet["focus"] == "tier:standard", ("Tab leaves the list for the tier; Default's Standard holds its stop", sheet)
            assert sheet["chosen"] == "" and [tier["tier"] for tier in sheet["tiers"] if tier["resolved"]] == ["standard"], sheet
            assert sheet["offered"] == OFFERED, ("no Ultrafast where the source lists none", sheet)
            moves = []
            for key in ("ArrowLeft", "ArrowRight", "ArrowRight"):
                page.keyboard.press(key)
                moves.append(page.evaluate(_SHEET)["focus"])
            assert moves == ["tier:economy", "tier:standard", "tier:fast"], moves
            assert page.evaluate(_SHEET)["chosen"] == "" and not _sent(writes, "owner/chat-processing", "project_id"), \
                "arrows move among the tiers without choosing"
            with page.expect_response(lambda r: r.url.endswith("/api/owner/chat-processing")) as reply:
                page.keyboard.press("Enter")
            assert reply.value.json()["processing_choice"] == "fast"
            assert _sent(writes, "owner/chat-processing", "project_id") == [{"processing": "fast", "project_id": room["id"]}], writes
            page.wait_for_function("(sel) => document.querySelector(sel)?.getAttribute('aria-checked') === 'true'",
                                   arg=_SEG.format("fast"), timeout=10_000)
            assert page.query_selector(".ui-popup.model-sheet"), "a tier leaves the sheet open"
            seen["focus_after_tier_enter"] = page.evaluate(_SHEET)["focus"]
            if seen["focus_after_tier_enter"] != "tier:fast":
                problems.append(("the tier chosen with Enter keeps the keyboard focus", seen["focus_after_tier_enter"]))
                page.focus(_SEG.format("fast"))  # carry on from where the keyboard was
            page.keyboard.press(back)
            assert page.evaluate(_SHEET)["focus"] == "field", "Shift+Tab goes back to the field"

            def walk_to(index: int) -> dict:
                """Down from the field to row `index` (Main's row is 0), then Right to its Pin."""
                for _ in range(index + 1):
                    page.keyboard.press("ArrowDown")
                page.keyboard.press("ArrowRight")
                sheet = page.evaluate(_SHEET)
                rows = [row for group in sheet["groups"] for row in group["rows"]]
                active = [row for row in rows if row["active"]]
                assert len(active) == 1 and active[0]["cell"] == "1" and rows.index(active[0]) == index, ("Right reaches the row's Pin", sheet)
                assert sheet["descendant"].endswith(f"-r{index}c1") and sheet["focus"] == "field", ("the field keeps the focus", sheet)
                return active[0]

            target = walk_to(2)
            route = target["row"].split("|", 1)[1]
            with page.expect_response(_pin_write) as reply:
                page.keyboard.press("Enter")
            assert json.loads(reply.value.request.post_data) == {"model_pins": {"pin": [_route(route)]}}
            page.wait_for_function(f"() => ({_PINNED})().includes('route|{route}')", timeout=10_000)
            assert page.query_selector(".ui-popup.model-sheet") and not _sent(writes, "owner/chat-model"), "a Pin chooses nothing and leaves the sheet open"
            _shot(page, evidence, f"keyboard-pinned-{engine}")
            page.keyboard.press("Escape")
            page.wait_for_function("() => !document.querySelector('.ui-popup.model-sheet')", timeout=10_000)
            assert page.evaluate(_CAPSULE, capsule)["focused"], "Escape closes and gives the focus back to the capsule"

            # Opened again: the pinned model stands first under Main; Enter on its Pin takes it back.
            page.keyboard.press("Enter")
            page.wait_for_selector(".ui-popup.model-sheet .chat-model-row", timeout=20_000)
            assert page.evaluate(_SHEET)["focus"] == "field"
            assert walk_to(1)["row"] == f"route|{route}", "the pinned model is the first row under Main"
            with page.expect_response(_pin_write) as reply:
                page.keyboard.press("Enter")
            assert json.loads(reply.value.request.post_data) == {"model_pins": {"unpin": [_route(route)]}}
            page.wait_for_function(f"() => ({_PINNED})().length === 0", timeout=10_000)
            assert next(p for p in _state(page)["projects"] if p["id"] == room["id"])["model_choice"] == {"kind": "main"}

            # The strip's ends: past Fast (the last tier offered here) the arrows and End stay on Fast, which keeps the
            # strip's one tab stop; Home goes to Flex.
            page.keyboard.press("Escape")
            page.wait_for_function("() => !document.querySelector('.ui-popup.model-sheet')", timeout=10_000)
            _open_sheet(page, capsule)
            page.focus(_SEG.format("fast"))  # the room's own tier now: the strip's one tab stop
            edges = []
            for key in ("ArrowRight", "End", "Home", "End"):
                page.keyboard.press(key)
                sheet = page.evaluate(_SHEET)
                edges.append((key, sheet["focus"], sheet["stop"]))
            seen["tier_edges"] = edges
            if edges != [("ArrowRight", "tier:fast", ["fast"]), ("End", "tier:fast", ["fast"]), ("Home", "tier:economy", ["economy"]),
                         ("End", "tier:fast", ["fast"])]:
                problems.append(("the strip's ends: a hidden tier is never the focus or the tab stop", edges))

            # A toggle stays where it was made: after Enter pins a row further down (the row moves up into Pinned),
            # the active Pin is still that model's, so a second Enter takes back the same pin.
            page.keyboard.press("Escape")
            page.wait_for_function("() => !document.querySelector('.ui-popup.model-sheet')", timeout=10_000)
            page.keyboard.press("Enter")
            page.wait_for_selector(".ui-popup.model-sheet .chat-model-row", timeout=20_000)
            target = walk_to(3)
            with page.expect_response(_pin_write):
                page.keyboard.press("Enter")
            page.wait_for_function(f"() => ({_PINNED})().includes('{target['row']}')", timeout=10_000)
            sheet = page.evaluate(_SHEET)
            seen["after_toggle"] = sheet
            active = [row for group in sheet["groups"] for row in group["rows"] if row["active"]]
            if [(row["row"], row["cell"]) for row in active] != [(target["row"], "1")]:
                problems.append(("the active Pin follows the model it pinned", target["row"], active))
            # Back with the room's sheet open on the desk: the room leaves and takes its sheet with it, nothing stays
            # over Main.
            page.evaluate("() => history.back()")
            page.wait_for_function("() => !document.querySelector('#project-panel-backdrop').classList.contains('open')", timeout=20_000)
            page.wait_for_function("() => !document.querySelector('.ui-popup.model-sheet') && !document.querySelector('.model-sheet-scrim')",
                                   timeout=10_000)
            if page.evaluate(_CAPSULE, capsule)["expanded"] != "false":
                problems.append(("the room's capsule says its sheet is closed", page.evaluate(_CAPSULE, capsule)))
            assert not problems, problems
        finally:
            (evidence / f"keys-seen-{engine}.json").write_text(json.dumps(seen, indent=1, default=str), encoding="utf-8")
            (evidence / f"keys-writes-{engine}.json").write_text(json.dumps(writes, indent=1), encoding="utf-8")
            browser.close()


@pytest.mark.parametrize('engine', ['chromium', 'webkit'])
def test_default_fallback_tier_matches_the_real_call_and_explicit_standard(direct_server_with_data, monkeypatch, engine, tmp_path):  # noqa: F811
    from playwright.sync_api import sync_playwright

    server = direct_server_with_data
    provider = _serve(server, monkeypatch)
    server['stop_server']()
    settings_path = server['data_dir'] / 'settings.json'
    settings = json.loads(settings_path.read_text())
    settings['OUROBOROS_MODEL_PROCESSING_PREFERENCES'] = json.dumps({'main': 'fast', 'fallback': ['economy']})
    settings_path.write_text(json.dumps(settings))
    provider.broken = {'mock-main'}
    server['start_server']()
    marker = 'ROOM_WIRE_FALLBACK_TIER'
    provider.hold(marker)
    evidence = _evidence(tmp_path)
    with sync_playwright() as pw:
        browser, page = _launch(pw, engine, viewport=DESK)
        try:
            _load(page, server['url'])
            room = _post(page, '/api/projects', {'name': 'Fixture · Fallback tier'})['project']
            prefix = _open_project(page, room)
            _send(page, prefix, marker)
            assert provider.held(marker)
            card = f'#{prefix}-messages .chat-live-card'
            page.wait_for_selector(f'{card} button.card-model-fact.model-fallback', timeout=60_000)
            task = page.get_attribute(card, 'data-task-id')
            work = _wait_work(page, task, lambda w: _receipt(w)[0] == 'economy', 'fallback Flex receipt')
            assert work['lead']['role'] == 'fallback:0'
            assert work['processing']['choice'] == '' and work['processing']['resolved'] == 'economy'
            assert _receipt(work) == ('economy', None, 'unknown', 'processing_not_submitted')
            assert provider.models_of(marker)[:2] == ['mock-main', 'mock-fallback']
            page.click(f'{card} button.card-model-fact')
            page.wait_for_function("() => document.querySelector('.model-tier-scope')?.textContent.startsWith('Default · Flex')")
            assert 'Flex' in page.evaluate(_CARD_TIER, card)
            _shot(page, evidence, f'fallback-default-{engine}')
            recorded = len(_wait_usage(page, server['data_dir'], task, 1))
            with page.expect_response(lambda r: r.url.endswith(f'/api/tasks/{task}/model')) as reply:
                page.click(_SEG.format('standard'))
            assert reply.value.json()['processing']['resolved'] == 'standard'
            page.wait_for_function("() => document.querySelector('.model-tier-scope')?.textContent.startsWith('Standard ·')")
            # The in-flight Flex call finishes; only the following actual call asks Standard.
            provider.advance(marker)
            assert provider.held(marker)
            provider.advance(marker)
            assert provider.held(marker)
            _wait_usage(page, server['data_dir'], task, recorded + 2)
            work = _wait_work(page, task, lambda w: _receipt(w)[0] == 'standard', 'explicit Standard receipt')
            assert work['lead']['basis'] == 'fallback'
            assert work['processing']['resolved'] == 'standard'
            _shot(page, evidence, f'fallback-standard-{engine}')
            (evidence / f'fallback-receipts-{engine}.json').write_text(json.dumps(work, indent=2))
        finally:
            provider.release_all()
            browser.close()
