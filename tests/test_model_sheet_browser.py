"""The Model sheet on the real server: a running work's tier, pins across windows, the keyboard, a finger, the Main
chat's confirm and tier, and a paused work without a model control.

Owner 12.11 «A» (docs/DESIGN.md "Composer owner controls", "A work's model on its card"): a press on the Model capsule
or on a work's model fact opens one floating sheet, a bottom sheet in the stack layout: the field, Main, Pinned,
Recent, the sources, and at its foot the processing tier strip. The real server on its own port and data root, the
real page, and the loopback provider of tests/_model_choice_browser.py: no paid call or live account. Some cases
project an older discovery response without transport evidence so that Fast and Flex can be requested but not assumed
supported. The actual loopback transport sends no tier: receipts distinguish the request from what went over HTTP.
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
    _sheet_pick,
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
_LEGACY_OFFERED = ["economy", "standard", "fast"]
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


def _legacy_catalog(page) -> None:
    """Only this page sees the old catalog shape for the compatible endpoint; its POSTs and model calls are real."""
    def without_ceiling(route):
        response = route.fetch()
        if not response.ok:
            route.fulfill(response=response)
            return
        catalog = response.json()
        catalog.pop("api_sources", None)  # a current source descriptor would still give the compatible adapter's ceiling
        for item in catalog["items"]:
            if item.get("provider_id") == SOURCE:
                item.pop("processing_transport", None)
        route.fulfill(json=catalog)

    page.route("**/api/model-catalog", without_ceiling)


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
            _legacy_catalog(page)
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
            # The card paints the receipt on the next state body it receives: wait for its words, never a fixed pause
            # (an absent fact would also read as nothing).
            page.wait_for_function(f"(sel) => ({_CARD_TIER})(sel) === 'Standard asked · sent with no tier'", arg=card, timeout=30_000)
            seen["standard_words"] = page.evaluate(_CARD_TIER, card)

            # Fast, pressed on the card while that call is in flight: one act, applied from the next call.
            page.click(f"{card} button.card-model-fact")
            page.wait_for_selector(".ui-popup.model-sheet .card-model-head", timeout=20_000)
            sheet = page.evaluate(_SHEET)
            seen["card_sheet"] = sheet
            assert sheet["offered"] == _LEGACY_OFFERED and sheet["chosen"] == "standard", sheet
            assert page.get_attribute(_SEG.format("fast"), "data-support") == "unknown", sheet
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
            page.wait_for_function(f"(sel) => ({_CARD_TIER})(sel) === 'Fast · Fast asked · sent with no tier'", arg=card, timeout=30_000)
            # The open sheet says it in its head, above the field: the strip at its foot keeps its words and place.
            page.wait_for_function("(words) => document.querySelector('.ui-popup.model-sheet .card-model-served')?.innerText === words",
                                   arg="Fast asked · sent with no tier", timeout=30_000)
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
            # served as only when it was served otherwise. An omission leaves the explicit Standard request unsubmitted.
            assert (seen["standard_words"], seen["standard_scope"]) == ("Standard asked · sent with no tier", f"Standard · {WORK_SCOPE_OWN}"), \
                ("the unsent Standard request stays visible", seen["standard_words"], seen["standard_scope"])
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
            # Unmodified discovery includes the compatible adapter's ceiling: Default and Standard are redundant.
            one.wait_for_function("() => document.querySelector('.ui-popup.model-sheet .model-tier-row')?.hidden === true")
            assert one.evaluate(_SHEET)["offered"] == []

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
            _legacy_catalog(page)
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
                assert page.evaluate(_SHEET)["offered"] == _LEGACY_OFFERED
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
                assert sheet["scope"].startswith(f"{word or 'Default'} · for Main and work that inherits its tier"), sheet
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
            _legacy_catalog(phone)
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
            assert sheet["chosen"] == "" and [tier["tier"] for tier in sheet["tiers"] if tier["resolved"]] == [], sheet
            assert sheet["offered"] == _LEGACY_OFFERED, sheet
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
            # Back with the room's sheet open on the desk: one Back for every input, a popup first (layers.js; RULING
            # §1, as on the phone above), so the sheet closes and the room stays; the next Back closes the room and
            # nothing stays over Main.
            page.evaluate("() => history.back()")
            page.wait_for_function("() => !document.querySelector('.ui-popup.model-sheet') && !document.querySelector('.model-sheet-scrim')",
                                   timeout=10_000)
            page.wait_for_timeout(400)
            if not page.evaluate("() => document.querySelector('#project-panel-backdrop').classList.contains('open')"):
                problems.append(("Back closed the sheet over the room, not the room under it", "desk"))
            if page.evaluate(_CAPSULE, capsule)["expanded"] != "false":
                problems.append(("the room's capsule says its sheet is closed", page.evaluate(_CAPSULE, capsule)))
            page.evaluate("() => history.back()")
            page.wait_for_function("() => !document.querySelector('#project-panel-backdrop').classList.contains('open')", timeout=20_000)
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
            _legacy_catalog(page)
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


def test_unavailable_inherited_ultrafast_can_be_left_by_keyboard(direct_server_with_data, monkeypatch):  # noqa: F811
    from playwright.sync_api import expect, sync_playwright

    server = direct_server_with_data
    _serve(server, monkeypatch)
    server["stop_server"]()
    path = server["data_dir"] / "settings.json"
    settings = json.loads(path.read_text())
    settings["OUROBOROS_MODEL_PROCESSING_PREFERENCES"] = json.dumps({"main": "ultrafast"})
    path.write_text(json.dumps(settings))
    server["start_server"]()

    with sync_playwright() as pw, pw.chromium.launch(headless=True) as browser:
        page = browser.new_page(viewport=DESK)
        writes = _record(page)
        _load(page, server["url"])
        room = _post(page, "/api/projects", {"name": "Fixture · inherited Ultra"})["project"]
        prefix = _open_project(page, room)
        _open_sheet(page, f"#{prefix}-model-mode")
        ultra = page.locator(_SEG.format("ultrafast"))
        standard = page.locator(_SEG.format("standard"))
        expect(page.locator(_SEG.format("fast"))).to_have_attribute("data-support", "unlisted")
        assert page.evaluate(_SHEET)["offered"] == ["standard", "ultrafast"]
        expect(ultra).to_have_attribute("data-resolved", "true")
        expect(ultra).to_have_attribute("aria-disabled", "true")
        ultra.click(force=True)
        ultra.press("Enter")
        ultra.press("Space")
        ultra.press("Home")
        expect(standard).to_be_focused()
        page.keyboard.press("ArrowRight")
        expect(ultra).to_be_focused()
        page.keyboard.press("ArrowLeft")
        expect(standard).to_be_focused()
        assert not _sent(writes, "owner/chat-processing"), writes
        with page.expect_response(lambda r: r.url.endswith("/api/owner/chat-processing")) as saved:
            page.keyboard.press("Space")
        assert saved.value.json()["processing_choice"] == "standard"
        assert _sent(writes, "owner/chat-processing") == [{"processing": "standard", "project_id": room["id"]}]
        expect(ultra).to_be_hidden()
        expect(standard).to_have_attribute("aria-checked", "true")


def _main_ultrafast_tabs(server, browser, extra: tuple = (), *, context_options=None, processing=None):
    """Two pages share the real local Settings document and writers. Only discovery and preview are canned; `extra`
    lists more Codex models the catalog offers beside Main's `gpt-test`."""
    path = server["data_dir"] / "settings.json"
    server["stop_server"]()
    initial = json.loads(path.read_text())
    initial.update({"OUROBOROS_MODEL": "claudexor::codex=gpt-test", "OUROBOROS_MODEL_ACCOUNTS": '{}',
                    "OUROBOROS_MODEL_PROCESSING_PREFERENCES": json.dumps(processing or {}), "USE_LOCAL_MAIN": False})
    path.write_text(json.dumps(initial))
    server["start_server"]()

    fixture = json.loads((pathlib.Path(__file__).resolve().parents[1] / "web/tests/fixtures/subscription_setup.json").read_text())
    catalog = fixture["catalog"]
    catalog["model_sources"][0]["processingPreferences"] = ["standard", "fast", "ultrafast"]
    catalog["store_id"] = "fixture-store"
    first = catalog["items"][0]
    catalog["items"] += [{**first, "id": model, "name": model, "value": f"claudexor::codex={model}"} for model in extra]
    context = browser.new_context(**(context_options or {"viewport": DESK}))
    context.route("**/api/model-catalog*", lambda route: route.fulfill(json=catalog))
    context.route("**/api/settings?response_limit_preview=*", lambda route: route.fulfill(json={}))
    context.route("**/api/claudexor/status*", lambda route: route.fulfill(json=fixture["status"]))
    return path, context.new_page(), context.new_page()


def test_reopening_model_sheet_reads_a_pin_changed_in_another_tab(direct_server_with_data):  # noqa: F811
    from playwright.sync_api import expect, sync_playwright
    from tests.settings_browser_helpers import open_settings, press_editor_save

    server = direct_server_with_data
    with sync_playwright() as pw, pw.chromium.launch(headless=True) as browser:
        path, first, other = _main_ultrafast_tabs(server, browser)
        other.goto(server["url"], wait_until="domcontentloaded")
        editor = open_settings(other, "models/role/main").locator('[data-model-role="main"]')
        account = editor.locator("[data-model-role-account]")
        account.select_option("personal")
        assert press_editor_save(other).startswith("Saved")
        _load(first, server["url"])
        sheet = first.locator(".ui-popup.model-sheet")
        means = sheet.locator(".model-tier-means")
        _open_sheet(first, MAIN_CAPSULE)
        expect(means).to_contain_text("Pinned account personal")
        _close_sheet(first)

        account.select_option("work")
        assert press_editor_save(other).startswith("Saved")
        saved = json.loads(path.read_text())
        assert saved["OUROBOROS_MODEL"] == "claudexor::codex=gpt-test"
        assert json.loads(saved["OUROBOROS_MODEL_ACCOUNTS"])["main"] == "work"

        with first.expect_response("**/api/settings") as refreshed:
            _open_sheet(first, MAIN_CAPSULE)
        assert refreshed.value.ok
        expect(means).to_contain_text("Pinned account work")
        expect(sheet).to_have_count(1)
        expect(sheet.locator(".chat-model-query")).to_be_focused()


@pytest.mark.parametrize("race", [
    "unchanged", "edited_model", "before_save_model", "before_save_account", "after_read", "edited_model_already_saved",
])
def test_settings_main_ultrafast_is_bound_across_two_tabs(direct_server_with_data, race):  # noqa: F811
    """Real Settings pages, GET tokens and owner writes to the case's settings.json."""
    from playwright.sync_api import expect, sync_playwright
    from tests.settings_browser_helpers import open_settings, press_editor_save

    server = direct_server_with_data
    model_key = "OUROBOROS_MODEL"
    account_key = "OUROBOROS_MODEL_ACCOUNTS"
    tier_key = "OUROBOROS_MODEL_PROCESSING_PREFERENCES"
    target = "claudexor::codex="
    checker = "**/api/owner/safety-model"

    with sync_playwright() as pw, pw.chromium.launch(headless=True) as browser:
        path, first, other = _main_ultrafast_tabs(server, browser)

        def main_editor(page):
            page.goto(server["url"], wait_until="domcontentloaded")
            return open_settings(page, "models/role/main").locator('[data-model-role="main"]')

        form, second = main_editor(first), main_editor(other)
        model = '[data-model-role-model]'
        account = '[data-model-role-account]'
        def edit_model(row, value):
            field = row.locator(model)
            field.fill(value)
            field.press("Tab")

        level = form.locator('[data-model-role-processing]')
        form.locator("summary").click()
        expect(form.locator(account)).to_have_value("")
        expect(second.locator(account)).to_have_value("")
        own_edit = race in ("edited_model", "edited_model_already_saved")
        succeeds = race in ("unchanged", "edited_model")
        if own_edit:
            edit_model(form, "gpt-next")

        # Cancel authors nothing. One subsequent confirmation suffices through Save.
        writes = []
        first.on("request", lambda req: writes.append(req.post_data_json)
                 if req.method == "POST" and req.url.endswith("/api/owner/safety-model") else None)
        level.select_option("ultrafast")
        dialog = first.get_by_role("dialog", name="Ultrafast for Main")
        expect(dialog).to_contain_text("Account: Auto")
        expect(dialog).to_contain_text("settings file")
        expect(dialog).to_contain_text("even on a fallback model")
        expect(dialog).to_contain_text("subagents inherit the work")
        expect(dialog).to_contain_text("costs on this route is not known")
        dialog.get_by_role("button", name="Cancel").click()
        expect(level).to_have_value("")
        assert not writes
        level.select_option("ultrafast")
        expect(dialog).to_contain_text("gpt-next" if own_edit else "gpt-test")
        dialog.get_by_role("button", name="Use Ultrafast").click()
        expect(form.locator("summary")).to_contain_text("Ultrafast (override)")

        before_save = race.startswith("before_save")
        if before_save or race == "edited_model_already_saved":
            if race == "before_save_account":
                second.locator(account).select_option("work")
            else:
                edit_model(second, "gpt-next")
            assert press_editor_save(other).startswith("Saved")

        if race in ("after_read", "edited_model_already_saved"):
            # Hold tab one's real POST after its GET. Tab two saves only the pin, or only the same field
            # tab one edited and the fresh GET already matched. Both must invalidate the pending write.
            held = []
            first.route(checker, lambda route: held.append(route))
            with first.expect_request(checker):
                form.locator("[data-editor-save]").click()
            if race == "after_read":
                second.locator(account).select_option("work")
            else:
                edit_model(second, "gpt-third")
            assert press_editor_save(other).startswith("Saved")
            assert len(held) == 1
            with first.expect_response(checker) as response:
                held[0].continue_()
            assert response.value.status == 409
            expect(form.locator("[data-editor-result]")).to_contain_text(re.compile("changed", re.I))
        else:
            line = press_editor_save(first)
            assert line.startswith("Saved" if succeeds else "Not saved")

        assert len(writes) == (0 if before_save else 1)
        if writes:
            expected = {tier_key: {"main": "ultrafast"}}
            if race == "edited_model":
                expected[model_key] = target + "gpt-next"
            assert writes[0]["settings"] == expected
            assert set(writes[0]["if_unchanged"]) == {model_key, account_key, "USE_LOCAL_MAIN", tier_key, "ultrafast"}
        stored = json.loads(path.read_text())
        assert json.loads(stored[tier_key]).get("main", "") == ("ultrafast" if succeeds else "")
        assert json.loads(stored[account_key]).get("main", "") == ("work" if race in ("before_save_account", "after_read") else "")
        final_model = {"before_save_model": "gpt-next", "edited_model": "gpt-next", "edited_model_already_saved": "gpt-third"}
        assert stored[model_key] == target + final_model.get(race, "gpt-test")


@pytest.mark.parametrize(
    "engine, change_binding", [("chromium", False), ("chromium", True), ("webkit", False)],
    ids=["unchanged", "delayed-post", "phone-webkit"],
)
def test_composer_main_ultrafast_is_bound_across_two_tabs(direct_server_with_data, engine, change_binding, tmp_path):  # noqa: F811
    from playwright.sync_api import expect, sync_playwright
    from tests.settings_browser_helpers import open_settings, press_editor_save

    server = direct_server_with_data
    post = "**/api/owner/chat-processing"
    model, pin, tier = "OUROBOROS_MODEL", "OUROBOROS_MODEL_ACCOUNTS", "OUROBOROS_MODEL_PROCESSING_PREFERENCES"
    phone = engine == "webkit"
    processing = {"light": "standard"}
    evidence = _evidence(tmp_path)
    with sync_playwright() as pw, getattr(pw, engine).launch(headless=True) as browser:
        path, first, other = _main_ultrafast_tabs(server, browser, context_options=PHONE if phone else None, processing=processing)
        writes = _record(first)
        _load(first, server["url"])
        if change_binding:
            other.goto(server["url"], wait_until="domcontentloaded")
            editor = open_settings(other, "models/role/main").locator('[data-model-role="main"]')
            account = editor.locator("[data-model-role-account]")
            expect(account).to_have_value("")
        dialog = first.get_by_role("dialog", name="Ultrafast for Main")
        press = first.tap if phone else first.click
        before = path.read_bytes()

        def choose():
            _open_sheet(first, MAIN_CAPSULE, finger=phone)
            expect(first.locator(".ui-popup.model-sheet .model-tier-means")).to_contain_text("Account: Auto")
            with first.expect_response("**/api/settings") as read:
                press(_SEG.format("ultrafast"))
            expect(dialog).to_be_visible()
            expect(dialog).to_contain_text("gpt-test")
            expect(dialog).to_contain_text("Account: Auto")
            expect(dialog).to_contain_text("even on a fallback model")
            expect(dialog).to_contain_text("subagents inherit the work")
            expect(dialog).to_contain_text("Safety falling back to Main")
            expect(dialog).to_contain_text("Not confirmed for gpt-test here")
            expect(dialog).to_contain_text("What Ultrafast costs on this route is not known here.")
            expect(dialog).to_contain_text("Before rolling back, choose Standard or Default for Main.")
            return read.value.json()

        choose()
        dialog.get_by_role("button", name="Cancel").click()
        expect(dialog).to_have_count(0)
        assert _sent(writes, "owner/chat-processing") == [] and path.read_bytes() == before

        shown = choose()
        assert shown[model] == "claudexor::codex=gpt-test" and not json.loads(shown[pin]).get("main")
        tokens = {key: shown["_meta"]["tokens"][key] for key in (model, "USE_LOCAL_MAIN", pin, tier)}
        tokens["ultrafast"] = json.dumps([shown[model], False, "", "", ""], separators=(",", ":"))
        confirm = dialog.get_by_role("button", name="Use Ultrafast")
        if change_binding:
            held = []
            first.route(post, lambda route: held.append(route))
            with first.expect_request(post):
                confirm.click()
            field = editor.locator("[data-model-role-model]")
            field.fill("gpt-next")
            field.press("Tab")
            account.select_option("work")
            assert press_editor_save(other).startswith("Saved")
            after_other = path.read_bytes()
            saved = json.loads(after_other)
            assert saved[model] == "claudexor::codex=gpt-next" and json.loads(saved[pin])["main"] == "work"
            assert not json.loads(saved[tier]).get("main")
            assert len(held) == 1
            with first.expect_response(post) as response:
                held[0].continue_()
            assert response.value.status == 409
            refusal = response.value.json()
            assert refusal["code"] == "settings_changed_elsewhere" and refusal["saved"] is False
            _said(first, "Main changed")
            assert path.read_bytes() == after_other
        else:
            _shot(first, evidence, f"main-ultrafast-warning-{engine}")
            with first.expect_response(post) as response:
                confirm.click()
            assert response.value.status == 200 and response.value.json()["main_processing"] == "ultrafast"
            saved = json.loads(path.read_bytes())
            assert saved[model] == shown[model] and not json.loads(saved[pin]).get("main")
            assert json.loads(saved[tier]) == {**processing, "main": "ultrafast"}
        expect(dialog).to_have_count(0)
        confirmed = {"processing": "ultrafast", "if_unchanged": tokens}
        assert _sent(writes, "owner/chat-processing") == [confirmed]

        if not change_binding:
            first.reload(wait_until="domcontentloaded")
            _capsule_ready(first, MAIN_CAPSULE)
            expect(first.locator(MAIN_CAPSULE)).to_have_attribute("data-tier", "ultrafast")
            _open_sheet(first, MAIN_CAPSULE, finger=phone)
            ultra = first.locator(_SEG.format("ultrafast"))
            expect(ultra).to_have_attribute("aria-checked", "true")
            reset = ".ui-popup.model-sheet .model-tier-default"
            # One press, one real POST and no second confirmation. Default removes Main's own override.
            with first.expect_response(post) as response:
                press(reset)
            assert response.value.status == 200 and response.value.json()["main_processing"] == ""
            expect(first.locator(reset)).to_be_hidden()
            expect(ultra).to_have_attribute("aria-checked", "false")
            expect(first.locator(MAIN_CAPSULE)).to_have_attribute("data-tier", "default")
            expect(first.locator(".confirm-dialog")).to_have_count(0)
            assert _sent(writes, "owner/chat-processing") == [confirmed, {"processing": ""}]
            after = json.loads(path.read_bytes())
            assert json.loads(after[tier]) == processing, "the literal main override is gone; Light's preference remains"
            assert all(after[key] == saved[key] for key in (model, pin, "USE_LOCAL_MAIN")), "the route and account are unchanged"
            _shot(first, evidence, f"main-ultrafast-default-{engine}")


MAIN_BINDING = {"OUROBOROS_MODEL", "USE_LOCAL_MAIN", "OUROBOROS_MODEL_ACCOUNTS", "OUROBOROS_MODEL_PROCESSING_PREFERENCES"}


def _role_editor(server, page, role: str = "main"):
    from tests.settings_browser_helpers import open_settings

    page.goto(server["url"], wait_until="domcontentloaded")
    return open_settings(page, f"models/role/{role}").locator(f'[data-model-role="{role}"]')


def _type_model(row, value: str) -> None:
    field = row.locator("[data-model-role-model]")
    field.fill(value)
    field.press("Tab")


def _save_main_tier(server, page, tier: str) -> None:
    """The second tab's one real Settings write: Main's own tier, through its page's Save."""
    from tests.settings_browser_helpers import press_editor_save

    editor = _role_editor(server, page)
    editor.locator("summary").click()
    editor.locator("[data-model-role-processing]").select_option(tier)
    assert press_editor_save(page).startswith("Saved")


def _save_light(server, page, model: str) -> None:
    """The second tab's one real unrelated write: Light's model, through its own page's Save."""
    from tests.settings_browser_helpers import press_editor_save

    _type_model(_role_editor(server, page, "light"), model)
    assert press_editor_save(page).startswith("Saved")


def test_settings_main_model_save_stands_on_the_tier_shown_across_two_tabs(direct_server_with_data):  # noqa: F811
    """Tab one types a new Main model with Main's tier shown as Default; tab two's real write sets Fast. Tab one's Save
    says why it is not saved and writes nothing, instead of carrying Fast to the new model unasked. Typed again, its
    Save succeeds although tab two's unrelated write (Light) lands between tab one's read and its held POST."""
    from playwright.sync_api import expect, sync_playwright
    from tests.settings_browser_helpers import press_editor_save

    server = direct_server_with_data
    model, tier = "OUROBOROS_MODEL", "OUROBOROS_MODEL_PROCESSING_PREFERENCES"
    checker = "**/api/owner/safety-model"
    with sync_playwright() as pw, pw.chromium.launch(headless=True) as browser:
        path, first, other = _main_ultrafast_tabs(server, browser)
        writes = []
        first.on("request", lambda req: writes.append(req.post_data_json)
                 if req.method == "POST" and req.url.endswith(("/api/owner/safety-model", "/api/settings")) else None)
        form = _role_editor(server, first)
        _type_model(form, "gpt-next")
        _save_main_tier(server, other, "fast")
        assert json.loads(json.loads(path.read_text())[tier])["main"] == "fast"
        before = path.read_bytes()
        line = press_editor_save(first)
        assert line.startswith("Not saved. Main's processing is Fast now, set on another device."), line
        assert not writes and path.read_bytes() == before, "nothing of the refused save is sent or stored"

        _type_model(form, "gpt-next")
        held = []
        first.route(checker, lambda route: held.append(route))
        with first.expect_request(checker):
            form.locator("[data-editor-save]").click()
        _save_light(server, other, "gpt-test")
        assert len(held) == 1
        with first.expect_response(checker) as response:
            held[0].continue_()
        assert response.value.status == 200, response.value.text()
        expect(form.locator("[data-editor-result]")).to_contain_text("Saved")
        stored = json.loads(path.read_text())
        assert stored[model] == "claudexor::codex=gpt-next" and json.loads(stored[tier])["main"] == "fast"
        # Light keeps its own source (this fixture seeds ``openai-compatible``): the other tab typed only its model.
        assert stored["OUROBOROS_MODEL_LIGHT"] == "openai-compatible::gpt-test", "the unrelated write stands too"
        assert set(writes[-1]["if_unchanged"]) == MAIN_BINDING, writes[-1]


@pytest.mark.parametrize("second", ["before_pick", "while_asked", "unrelated"])
def test_composer_main_model_change_stands_on_the_tier_shown_across_two_tabs(direct_server_with_data, second):  # noqa: F811
    """Tab one changes Main's model in the composer; tab two makes one real Settings write. Set before the pick, the
    confirming sheet says the tier the new model keeps; set while the sheet asks, the write is refused and said, and
    nothing of it lands; an unrelated write while it asks lets it land."""
    from playwright.sync_api import expect, sync_playwright

    server = direct_server_with_data
    model, tier = "OUROBOROS_MODEL", "OUROBOROS_MODEL_PROCESSING_PREFERENCES"
    target = "claudexor::codex=gpt-next"
    with sync_playwright() as pw, pw.chromium.launch(headless=True) as browser:
        path, first, other = _main_ultrafast_tabs(server, browser, extra=("gpt-next",))
        writes = _record(first)
        _load(first, server["url"])
        dialog = first.get_by_role("dialog", name="Change Main")
        if second == "before_pick":
            _save_main_tier(server, other, "fast")
        _sheet_pick(first, MAIN_CAPSULE, target)
        expect(dialog).to_have_count(1)
        if second == "before_pick":
            expect(dialog).to_contain_text("Main keeps its tier, Fast")
        elif second == "while_asked":
            _save_main_tier(server, other, "fast")
        else:
            _save_light(server, other, "gpt-test")
        before = path.read_bytes()
        with first.expect_response("**/api/owner/chat-model") as response:
            dialog.locator("[data-confirm-ok]").click()
        posted = _sent(writes, "owner/chat-model")
        assert len(posted) == 1 and set(posted[0]["if_unchanged"]) == MAIN_BINDING, posted
        stored = json.loads(path.read_text())
        if second == "while_asked":
            assert response.value.status == 409 and response.value.json()["saved"] is False
            _said(first, "Main changed on another device")
            assert path.read_bytes() == before and stored[model] == "claudexor::codex=gpt-test"
        else:
            assert response.value.status == 200, response.value.text()
            assert stored[model] == target
            assert json.loads(stored[tier]).get("main", "") == ("fast" if second == "before_pick" else "")
            if second == "unrelated":
                assert stored["OUROBOROS_MODEL_LIGHT"] == "openai-compatible::gpt-test"  # Light's own source, as seeded


@pytest.mark.parametrize("engine, premium", [("chromium", True), ("webkit", False)], ids=["wide-priced", "phone-unknown"])
def test_main_model_confirmation_prices_only_the_refreshed_tier(direct_server_with_data, engine, premium, tmp_path):  # noqa: F811
    """The old view quotes Standard, but the confirming read names Ultra. Neither old rate nor cache price belongs in it."""
    from playwright.sync_api import expect, sync_playwright

    server, target = direct_server_with_data, "fixture/next-model"
    phone = engine == "webkit"
    with sync_playwright() as pw, getattr(pw, engine).launch(headless=True) as browser:
        path, page, _ = _main_ultrafast_tabs(server, browser, context_options=PHONE if phone else None, processing={"main": "ultrafast"})
        writes = _record(page)
        picked = False

        def as_seen(route):
            response = route.fetch()
            shown = response.json()
            if not picked:
                shown["OUROBOROS_MODEL_PROCESSING_PREFERENCES"] = json.dumps({"main": "standard"})
            shown["OUROBOROS_PROMPT_CACHE_TTL"] = "5m"
            shown["_meta"]["connections"] = [{"id": "openrouter", "configured": True, "revision": "openrouter#fixture"}]
            route.fulfill(response=response, json=shown)

        def state_before_confirmation(route):
            response = route.fetch()
            shown = response.json()
            shown["processing_tiers"] = {"main": "standard", "default": "standard"}
            route.fulfill(response=response, json=shown)

        page.route("**/api/settings", as_seen)
        page.route("**/api/state*", state_before_confirmation)
        page.route("**/api/model-catalog", lambda route: route.fulfill(json={
            "items": [{"id": target, "value": target, "provider_id": "openrouter", "context_window": 128000,
                       "price": {"input": 2, "output": 5, "cache_write": 7, "cache_write_ttl": "5m"}}],
            "api_sources": [{"id": "openrouter", "label": "Fixture key", "processing_transport": ["standard", "ultrafast"]}],
        }))
        page.route("**/api/model-catalog?processing_model=*", lambda route: route.fulfill(json={
            "value": target, "processing": {"complete": True, "modes": ["standard", "ultrafast"],
                                            "endpoints": [{"mode": "ultrafast", "price": {"input": 17, "output": 31} if premium else None}]},
        }))
        _load(page, server["url"])
        _open_sheet(page, MAIN_CAPSULE, finger=phone)
        expect(page.locator(".model-tier-means")).to_contain_text("Account: Auto")  # the opening GET has answered
        row = page.locator(_row(target))
        expect(row).to_contain_text("$2 in / $5 out")
        expect(page.locator(MAIN_CAPSULE)).to_have_attribute("data-tier", "standard")
        choose = page.tap if phone else page.click
        dialog = page.get_by_role("dialog", name="Change Main")
        choose(f"{_row(target)} .chat-model-cell")
        expect(dialog).to_contain_text("Cache write: $7 per 1M tokens.")
        dialog.locator("[data-confirm-cancel]").click()
        expect(dialog).to_have_count(0)

        _open_sheet(page, MAIN_CAPSULE, finger=phone)
        expect(page.locator(".model-tier-means")).to_contain_text("Account: Auto")
        picked = True
        choose(f"{_row(target)} .chat-model-cell")
        expect(dialog).to_contain_text("Fixture key · paid by key · 128K window")
        expect(dialog).to_contain_text("Main keeps its tier, Ultrafast")
        expect(dialog).to_contain_text("$17 in / $31 out per 1M tokens" if premium else "Price unknown.")
        expect(dialog).not_to_contain_text(re.compile(r"\$2 in|\$5 out|Cache write: \$7"))
        _shot(page, _evidence(tmp_path), f"main-ultrafast-fresh-price-{engine}")
        before = path.read_bytes()
        dialog.locator("[data-confirm-cancel]").click()
        expect(dialog).to_have_count(0)
        assert not _sent(writes, "owner/chat-model") and path.read_bytes() == before


def test_settings_pinned_ultrafast_conflict_rereads_the_catalog_without_resubmitting(direct_server_with_data):  # noqa: F811
    from playwright.sync_api import expect, sync_playwright
    from tests.settings_browser_helpers import press_editor_save

    server = direct_server_with_data
    with sync_playwright() as pw, pw.chromium.launch(headless=True) as browser:
        path, page, _ = _main_ultrafast_tabs(server, browser)
        writes, reads = [], []

        def catalog_for(route):
            replaced = bool(writes)
            query = route.request.url.partition("?")[2]
            kind = "scoped" if "credential_profile_id=work" in query else "other" if query else "general"
            reads.append((replaced, kind))
            route.fulfill(json={
                "store_id": "replacement-store" if replaced else "initial-store",
                "items": [{"value": "claudexor::codex=gpt-test", "id": "gpt-test", "credential_profile_id": "work"}],
                "model_sources": [{"id": "codex", "label": "Codex", "credentialHarness": "codex",
                                   "processingPreferences": ["standard", "fast", "ultrafast"]}],
                "account_catalogs": [{"source": "codex", "accounts": [{"credentialProfileId": "work", "catalog": {
                    "source": "codex", "credentialProfileId": "work", "accountFingerprint": "work-new" if replaced else "work-old",
                    "provenance": "live", "models": [{"id": "gpt-test", "processing": {
                        "modes": ["standard", "fast"] if replaced else ["standard", "fast", "ultrafast"],
                    }}],
                }}]}],
            })

        def refuse(route):
            writes.append(route.request.post_data_json)
            route.fulfill(status=409, json={"ok": False, "saved": False, "code": "settings_changed_elsewhere", "error": "The pinned account changed."})

        page.route("**/api/model-catalog*", catalog_for)
        page.route("**/api/owner/safety-model", refuse)
        before = path.read_bytes()
        form = _role_editor(server, page)
        form.locator("[data-model-role-account]").select_option("work")
        form.locator("summary").click()
        tier = form.locator("[data-model-role-processing]")
        expect(tier.locator("option[value=ultrafast]")).to_have_text("Ultrafast")
        tier.select_option("ultrafast")
        dialog = page.get_by_role("dialog", name="Ultrafast for Main")
        expect(dialog).to_contain_text("Pinned account work")
        dialog.get_by_role("button", name="Use Ultrafast").click()
        assert "changed" in press_editor_save(page).lower()
        assert len(writes) == 1 and path.read_bytes() == before
        assert json.loads(writes[0]["if_unchanged"]["ultrafast"])[2:] == ["work", "initial-store", "work-old"]
        assert (True, "general") in reads

        # Existing conflict recovery restores the saved form. Re-select the same pin: it no longer confirms Ultra.
        form.locator("[data-model-role-account]").select_option("work")
        form.locator("summary").click()
        expect(tier.locator("option[value=fast]")).to_have_text("Fast")  # the new scope answered, not merely absent
        expect(tier.locator("option[value=ultrafast]")).to_have_text("Ultrafast · not confirmed")
        assert (True, "scoped") in reads
        tier.select_option("ultrafast")
        expect(dialog).to_contain_text("Not confirmed for gpt-test here")
        dialog.get_by_role("button", name="Cancel").click()
        assert len(writes) == 1, "only the refused save was sent"
