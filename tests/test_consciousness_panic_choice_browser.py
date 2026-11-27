"""After Panic the owner's Turn on in an ordinary browser or on a phone turns consciousness back on as in 7.12.1 (the
owner 27.11), handing Ouroboros no choice of Panic-stopped tasks: only the desktop app's native confirmation hands that
over (docs/DESIGN.md "Dashboard › Consciousness"; ``supervisor/panic_choice.py``). Manual per-task Resume stays.

Two scenes. The first runs the shipped SPA in desktop and phone browser viewports over the shared fixture's endpoints,
the consciousness reads answered with what the host's own functions produce: pressing Turn on asks Turn on / Keep off
with the host's words and a note that this Turn on hands over no choice; Keep off sends nothing, Turn on sends one
WebSocket ``/bg start``. The second runs one whole browser path on a REAL isolated server (``direct_server_with_data``:
its own data root, port and mock model): an installation stopped by Panic is written with the product's own writers
while the server is down (no Panic is invoked), the server boots on it, Keep off changes nothing, manual per-task Resume
in Main returns the chosen task while consciousness stays off, and then Turn on in the browser turns consciousness on,
clears Panic's stop, leaves the other task's Panic hold standing and writes no choice key to ``state.json``.
Screenshots go to ``OUROBOROS_UI_EVIDENCE_DIR`` when set, before assertions.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests.test_subscription_setup_browser import subscription_ui as subscription_ui
from tests.test_ui_smoke_playwright import direct_server_with_data as direct_server_with_data

pytestmark = [pytest.mark.ui_browser]
BASIS = "0123456789abcdef0123456789abcdef"
DIALOG = ".confirm-dialog"
BROWSER_NOTE = ("Turning it on here hands Ouroboros no choice: the tasks stopped by Panic stay stopped until you "
                "resume them.")
HOST_LINES = ("Panic turned consciousness off.",
              "Nothing starts right now: 2 tasks stopped by Panic stay stopped at this press.",
              "Confirmed in the desktop app, Ouroboros may on its later wakes decide which of them to resume, even "
              "one the Panic was pressed for.",
              "A task you stopped or paused separately stays stopped, and deadlines, money and autonomy limits stay "
              "in force.",
              "Turning consciousness off, or restarting Ouroboros, ends that choice.")


def _asks_with_the_host_words_and_the_note(page, width=None, height=None):
    words = page.locator(DIALOG).inner_text()
    assert "Turn on consciousness?" in words
    assert all(line in words for line in HOST_LINES), words
    assert BROWSER_NOTE in words and "desktop app on that computer" not in words
    assert [button.inner_text() for button in page.locator(f"{DIALOG} button").all()] == ["Keep off", "Turn on"]
    if width:
        box = page.locator(DIALOG).bounding_box()
        assert box["x"] >= 0 and box["x"] + box["width"] <= width and box["y"] + box["height"] <= height, box


def _host_answers():
    from ouroboros.consciousness_view import describe_consciousness_state, switch_preview

    status = {**describe_consciousness_state({"stop_source": "panic"}, False), "enabled": False, "recent_wakes": []}
    preview = switch_preview("on", [], autonomy="act", daily_usd=20,
                             after_panic={"why": "panic", "held": 2, "choice": True, "basis": BASIS})
    return status, preview


def _shot(page, name):
    if root := os.environ.get("OUROBOROS_UI_EVIDENCE_DIR"):
        target = Path(root) / "consciousness-panic-choice"
        target.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(target / f"{name}.png"))


@pytest.mark.parametrize("width,height,theme", [(1100, 750, "dark"), (390, 844, "light")])
def test_turn_on_after_panic_in_browser_asks_and_sends_the_plain_start_only_on_yes(
        subscription_ui, width, height, theme):
    ui = subscription_ui
    status, preview = _host_answers()
    frames = []
    page = ui["new_page"](viewport={"width": width, "height": height}, color_scheme=theme)

    def socket(ws):
        ws.on_message(lambda message: frames.append(json.loads(message)))
        ws.send(json.dumps({"type": "heartbeat"}))

    page.route_web_socket("**/ws", socket)
    page.route("**/api/consciousness*", lambda route: route.fulfill(content_type="application/json", body=json.dumps(
        preview if "switch=on" in route.request.url else status)))
    page.goto(ui["url"] + "/")
    page.wait_for_selector("#page-chat")
    if width < 720:
        page.locator(".page.active [data-mobile-nav-toggle]:visible").click()
    page.click('[data-nav-page="dashboard"]')
    page.wait_for_selector("#page-dashboard.active")
    page.click('[data-dashboard-tab="consciousness"]')
    turn_on = page.locator('#page-dashboard [data-cx-act="toggle"]')
    turn_on.wait_for()
    page.wait_for_function("(s) => !document.querySelector(s).disabled", arg='#page-dashboard [data-cx-act="toggle"]')
    _shot(page, f"dashboard-off-after-panic-{width}-{theme}")
    assert turn_on.inner_text() == "Turn on" and turn_on.get_attribute("data-enabled") == "0"
    commands = lambda: [frame for frame in frames if frame.get("type") == "command"]

    turn_on.click()
    page.wait_for_selector(DIALOG)
    _shot(page, f"browser-plain-turn-on-question-{width}-{theme}")
    _asks_with_the_host_words_and_the_note(page, width, height)
    assert commands() == [], "nothing is sent before the answer"

    page.click(f"{DIALOG} [data-confirm-cancel]")  # Keep off
    page.wait_for_selector(DIALOG, state="detached")
    page.wait_for_function(
        "(s) => { const b = document.querySelector(s); return b && !b.disabled && b.dataset.enabled === '0'; }",
        arg='#page-dashboard [data-cx-act="toggle"]',
    )
    page.wait_for_timeout(200)
    assert commands() == [], "Keep off sends no /bg start and makes no retry"

    turn_on.click()
    page.wait_for_selector(DIALOG)
    page.click(f"{DIALOG} [data-confirm-ok]")  # Turn on
    page.wait_for_selector(DIALOG, state="detached")
    for _ in range(50):
        if commands():
            break
        page.wait_for_timeout(100)
    assert [frame.get("cmd") for frame in commands()] == ["/bg start"], "Turn on sends the plain /bg start once"


# --- one whole browser path on a real isolated server ------------------------------------------------------------

def _bind(monkeypatch, data_dir):
    """This process's supervisor state and queue over the stopped server's data root: fresh, empty lists, no workers."""
    from supervisor import queue, state, workers

    monkeypatch.setenv("OUROBOROS_DATA_DIR", str(data_dir))
    monkeypatch.setenv("OUROBOROS_SETTINGS_PATH", str(data_dir / "settings.json"))
    monkeypatch.setattr(state, "DRIVE_ROOT", data_dir)
    for name, relative in (("STATE_PATH", "state/state.json"), ("STATE_LAST_GOOD_PATH", "state/state.last_good.json"),
                           ("STATE_LOCK_PATH", "locks/state.lock")):
        monkeypatch.setattr(state, name, data_dir / relative)
    pending, running = [], {}
    for module in (queue, workers):
        monkeypatch.setattr(module, "DRIVE_ROOT", data_dir)
        monkeypatch.setattr(module, "PENDING", pending)
        monkeypatch.setattr(module, "RUNNING", running)
    monkeypatch.setattr(queue, "BUDGET_ROOT_FENCES", {})
    monkeypatch.setattr(queue, "ACCEPTANCE_FENCES", {})
    monkeypatch.setattr(queue, "PRIOR_DIRECT_ROOTS", {})
    monkeypatch.setattr(queue, "QUEUE_SEQ_COUNTER_REF", {"value": 0})
    monkeypatch.setattr(workers, "WORKERS", {})
    monkeypatch.setattr(queue, "INITIALIZED", True)
    monkeypatch.setattr(queue, "OWNER_PID", os.getpid())
    monkeypatch.setattr(queue, "QUEUE_SNAPSHOT_PATH", data_dir / "state" / "queue_snapshot.json")
    return queue, state, pending, running


def _stopped_by_panic(monkeypatch, data_dir):
    """Write, with the product's writers, what a received Panic leaves: its receipt, consciousness off by it, and two
    invented works under their own Panic holds in the published queue."""
    import time

    from ouroboros.installation_stop import PANIC_RECEIPT
    from ouroboros.task_results import write_task_result
    from ouroboros.work_stop import panic_hold_id
    from supervisor import events_budget as eb

    with monkeypatch.context() as patch:
        queue, state, pending, _ = _bind(patch, data_dir)
        queue.restore_pending_from_snapshot()
        state.update_state(lambda live: live.update({
            PANIC_RECEIPT: {"mtime_ns": int((time.time() - 3600) * 1e9)},
            "bg_consciousness_enabled": False, "bg_consciousness_stop_source": "panic"}))
        for task_id, title in (("report", "Weekly report"), ("digest", "News digest")):
            row = {"id": task_id, "root_task_id": task_id, "chat_id": 1, "type": "task", "_attempt": 1,
                   "admitted_dispatch": "none", "text": title, "title": title}
            write_task_result(data_dir, task_id, "scheduled", root_task_id=task_id, chat_id=1, title=title)
            eb.hold_budget_row(row, reason=eb.HOLD_PANIC)
            pending.append(row)
        assert queue.publish_queue_snapshot(reason="fixture_panic_holds")
        return {row["id"]: panic_hold_id(row) for row in pending if row["id"] in ("report", "digest")}


def _saved_state(data_dir):
    return json.loads((data_dir / "state" / "state.json").read_text(encoding="utf-8"))


def test_a_real_server_turns_consciousness_on_from_the_browser_after_panic_with_no_choice_and_manual_resume_works(
        direct_server_with_data, monkeypatch):
    from playwright.sync_api import sync_playwright

    server = direct_server_with_data
    url, data_dir = server["url"], server["data_dir"]
    server["stop_server"]()  # seed only while stopped
    holds = _stopped_by_panic(monkeypatch, data_dir)
    server["start_server"]()

    toggle = '#page-dashboard [data-cx-act="toggle"]'
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        try:
            page = browser.new_page(viewport={"width": 1100, "height": 750}, color_scheme="dark")
            page.goto(url, wait_until="domcontentloaded")
            page.wait_for_selector("#page-chat")
            listed = lambda: {row["task_id"]: row["panic_hold_id"]
                              for row in page.request.get(url + "/api/state").json()["panic_recovery"]}
            page.wait_for_function("async () => (await (await fetch('/api/state')).json()).panic_recovery?.length === 2")
            assert listed() == holds, "the booted server holds both works, as the list shows them"
            page.click('[data-nav-page="dashboard"]')
            page.click('[data-dashboard-tab="consciousness"]')
            page.wait_for_function(
                "(s) => { const b = document.querySelector(s); return b && !b.disabled && b.dataset.enabled === '0'; }",
                arg=toggle,
            )
            _shot(page, "real-server-off-after-panic")

            # A plain browser (no pywebview bridge): Turn on asks in the real host's words; Keep off changes nothing.
            page.click(toggle)
            page.wait_for_selector(DIALOG)
            _shot(page, "real-server-browser-plain-turn-on-question")
            _asks_with_the_host_words_and_the_note(page)
            assert _saved_state(data_dir)["bg_consciousness_enabled"] is False, "asked before anything changes"

            page.click(f"{DIALOG} [data-confirm-cancel]")  # Keep off
            page.wait_for_selector(DIALOG, state="detached")
            page.wait_for_function(
                "(s) => { const b = document.querySelector(s); return b && !b.disabled && b.dataset.enabled === '0'; }",
                arg=toggle,
            )
            state_body = page.request.get(url + "/api/state").json()
            assert state_body["bg_consciousness_enabled"] is False
            assert listed() == holds, "Keep off leaves consciousness off and both Panic holds standing"

            # Manual per-task Resume in Main remains available to the browser user while consciousness stays off.
            page.click('[data-nav-page="chat"]')
            page.wait_for_selector('#page-chat.active [data-owner-action="panic-recovery"]')
            page.wait_for_function(
                "() => { const box = document.querySelector('.panic-recovery-row[data-task-id=\"report\"] input[type=\"checkbox\"]');"
                " return box && !box.disabled; }"
            )
            page.locator('.panic-recovery-row[data-task-id="report"] input[type="checkbox"]').check()
            page.wait_for_function("() => !document.querySelector('[data-panic-resume]')?.disabled")
            page.click("[data-panic-resume]")
            page.wait_for_function(
                "async () => { const rows = (await (await fetch('/api/state')).json()).panic_recovery || [];"
                " return rows.length === 1 && rows[0].task_id === 'digest'; }",
                timeout=30_000,
            )
            _shot(page, "real-server-browser-manual-resume")
            assert listed() == {"digest": holds["digest"]}, "manual Resume returned only the selected task"
            assert page.request.get(url + "/api/state").json()["bg_consciousness_enabled"] is False

            # The owner's plain Turn on from this browser turns consciousness on, as in 7.12.1, handing over no choice.
            page.click('[data-nav-page="dashboard"]')
            page.click('[data-dashboard-tab="consciousness"]')
            page.wait_for_function(
                "(s) => { const b = document.querySelector(s); return b && !b.disabled && b.dataset.enabled === '0'; }",
                arg=toggle,
            )
            page.click(toggle)
            page.wait_for_selector(DIALOG)
            page.click(f"{DIALOG} [data-confirm-ok]")  # Turn on
            said = lambda: json.dumps(page.request.get(url + "/api/chat/history?chat_id=1").json(), ensure_ascii=False)
            for _ in range(300):  # its own reply says so, in Main
                if "Consciousness on" in said():
                    break
                page.wait_for_timeout(100)
            assert "🧠 Consciousness on" in said(), said()[-3000:]
            _shot(page, "real-server-browser-turned-on-after-panic")
            saved = _saved_state(data_dir)
            assert saved["bg_consciousness_enabled"] is True and "bg_consciousness_stop_source" not in saved, {
                key: saved.get(key) for key in ("bg_consciousness_enabled", "bg_consciousness_stop_source")}
            assert listed() == {"digest": holds["digest"]}, "the Turn on returned no work: the other Panic hold stands"
        finally:
            browser.close()

    saved = _saved_state(data_dir)
    assert saved["bg_consciousness_enabled"] is True and "bg_consciousness_stop_source" not in saved
    assert "bg_consciousness_panic_choice" not in saved
