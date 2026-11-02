"""Real Chat replay and controls over isolated durable logs, with a controlled census and its work projection."""
from __future__ import annotations

from tests import _dialogue as dialogue_test

import json
import time
import urllib.parse
import urllib.request

import pytest

from ouroboros import dialogue_store

from tests.test_ui_smoke_playwright import direct_server_with_data as direct_server_with_data
from tests._store_shared import seed_result
from tests._work_projection_shared import work_fields
from tests.ui_chat_viewport_smoke import state_reads
from ouroboros.test_evidence import diagnostic_directory
from tests.ui_console_errors import PageErrors

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_speechless_history_and_direct_resume(direct_server_with_data, engine):
    from playwright.sync_api import expect, sync_playwright
    from ouroboros.task_results import load_task_result

    fixture = direct_server_with_data
    root, task_id, turn_id = fixture["data_dir"], "tool-only-pause", "direct-turn"
    fixture["stop_server"]()
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    dialogue_store.append(root, "chat", {"direction": "in", "chat_id": 1,
        "task_id": task_id, "text": "Inspect the saved sources", "ts": "2026-10-04T09:00:00Z"})
    calls = [{"type": kind, "task_id": task_id, "invocation_id": f"call-{index}",
              "tool": "read_file", "is_error": False, "ts": "2026-10-04T09:00:01Z"}
             for index in range(24) for kind in ("tool_call_started", "tool_call")]
    (logs / "tools.jsonl").write_text("".join(json.dumps(row) + "\n" for row in calls), encoding="utf-8")
    archive = root / "archive"
    archive.mkdir(exist_ok=True)
    for index in range(4):
        (archive / f"tools_2026100{index}.jsonl").write_text(json.dumps({
            **calls[1], "task_id": "unrelated", "invocation_id": f"unrelated-{index}"}) + "\n", encoding="utf-8")
    results = root / "task_results"
    results.mkdir(exist_ok=True)
    # An earlier version saved this answered root's upkeep as the owner's Pause, for a Resume.
    seed_result(root, task_id, {"_schema_version": 1, "task_id": task_id,
        "status": "completed", "suggested_name": "Read source records", "_is_direct_chat": True,
        "root_phase_checkpoint": {"post_task_synthesis": "paused"}})
    # A direct turn still running: its card comes from its own progress row.
    dialogue_store.append(root, "progress", {"task_id": turn_id, "chat_id": 1,
        "content": "Reading the saved sources", "ts": "2026-10-04T09:02:00Z"})
    fixture["start_server"]()
    # A Pause now ends the upkeep at once, so the maintenance pass ends that saved remainder once, disclosed.
    deadline, checkpoint = time.monotonic() + 60, {}
    while time.monotonic() < deadline:
        checkpoint = (load_task_result(root, task_id) or {}).get("root_phase_checkpoint") or {}
        if checkpoint.get("post_task_synthesis") == "degraded":
            break
        time.sleep(0.2)
    assert checkpoint.get("post_task_synthesis") == "degraded", checkpoint
    assert str(checkpoint.get("post_task_stop_reason") or "").startswith("owner_pause:skipped="), checkpoint
    census = {"phase": "budget_pausing", "pause_cause": "owner"}
    with urllib.request.urlopen(fixture["url"] + "/api/state", timeout=10) as response:
        state = json.load(response)

    def state_route(route):
        data = dict(state)
        pausing = census["phase"] == "budget_pausing"
        rows = [{"activity_id": turn_id, "chat_id": 1, "kind": "direct_chat", "live": pausing, **census}]
        # The projection also reads physical liveness and the current Pause fence.
        # No author is declared by this fixture; it must not invent who pressed Pause.
        data.update(active_chat_activities=rows, active_chat_activities_complete=True,
                    supervisor_ready=True, **work_fields(rows, fences={
                        turn_id: {"state": "requested" if pausing else "paused"}}))
        route.fulfill(json=data)

    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch(headless=True)
        try:
            for width in (1440, 390):
                context = browser.new_context(viewport={"width": width, "height": 900},
                    is_mobile=width == 390, has_touch=width == 390)
                page = context.new_page()
                page.route(state_reads, state_route)
                census["phase"] = "budget_pausing"
                page.goto(fixture["url"], wait_until="domcontentloaded")
                card = page.locator(f'.chat-live-card[data-task-id="{task_id}"]')
                card.wait_for(state="visible")
                # The ended upkeep leaves the answer Done, with nothing to resume or stop.
                page.wait_for_function("id => document.querySelector(`.chat-live-card[data-task-id='${id}']`)?.dataset.finished === '1'", arg=task_id)
                assert card.locator("[data-live-phase]").inner_text() == "Done"
                assert card.locator("[data-resume-run]").count() == 0
                assert card.locator("[data-cancel-run]").count() == 0
                assert card.locator("[data-live-title]").inner_text() == "Read source records"
                assert page.locator(".chat-bubble.assistant:not(.typing-bubble)").count() == 0
                assert "outcome unknown" not in card.inner_text(), "unrelated archives cannot unsettle known calls"
                # The running direct turn's Pause is still settling: the work's word alone (who paused is its line;
                # this controlled row's press declared nobody).
                turn = page.locator(f'.chat-live-card[data-task-id="{turn_id}"]')
                turn.wait_for(state="visible")
                page.wait_for_function("id => document.querySelector(`.chat-live-card[data-task-id='${id}'] [data-live-phase]`)?.textContent === 'Pausing…'", arg=turn_id)
                assert turn.locator("[data-resume-run]").count() == 0
                assert turn.locator("[data-cancel-run]").is_visible()
                page.screenshot(path=str(diagnostic_directory(root.parent, "task-evidence") / f"task-evidence-{engine}-{width}-pausing.png"), full_page=True)

                census["phase"] = "budget_paused"
                page.reload(wait_until="domcontentloaded")
                resume = turn.locator("[data-resume-run]")
                resume.wait_for(state="visible")
                assert resume.inner_text() == "Resume"
                # The first read lands before the socket opens: Activity unconfirmed until the census over it lands.
                expect(turn.locator("[data-live-phase]")).to_have_text("Paused")
                assert page.locator("#chat-messages").evaluate("el => el.scrollWidth <= el.clientWidth + 1")
                assert resume.evaluate("el => { const r=el.getBoundingClientRect(); return r.left>=0 && r.right<=innerWidth; }")
                turn.locator("[data-cancel-run]").click()
                assert page.locator('[role="menuitem"]', has_text="Resume").count() == 0
                assert page.locator('[role="menuitem"]', has_text="Stop now").is_visible()
                page.keyboard.press("Escape")
                page.screenshot(path=str(diagnostic_directory(root.parent, "task-evidence") / f"task-evidence-{engine}-{width}-paused.png"), full_page=True)
                card.locator(":scope > [data-live-summary-button] [data-live-disclosure]").click()
                expect(card.locator(":scope > [data-live-foot] > [data-live-meta]")).to_contain_text("24 tool calls")
                lines = card.locator(".chat-live-line")
                # The tool calls, then the answer's own Done line.
                assert lines.count() == 2, lines.all_inner_texts()
                assert "24 tool calls" in lines.nth(0).inner_text() and lines.nth(1).inner_text().startswith("Done")
                page.screenshot(path=str(diagnostic_directory(root.parent, "task-evidence") / f"task-evidence-{engine}-{width}-expanded.png"), full_page=True)
                tools = lines.nth(0)
                tools.locator("[data-live-line-toggle]").click()
                assert "Only recent tool history was read." in tools.locator(".chat-live-line-body").inner_text()
                page.screenshot(path=str(diagnostic_directory(root.parent, "task-evidence") / f"tool-history-bounded-{engine}-{width}.png"), full_page=True)

                # The controlled projection offers the door; the real server still
                # refuses a turn with no resumable queue/fence authority.
                resume.click()
                page.locator(".toast", has_text="Resume refused").wait_for(state="visible")
                assert turn.locator("[data-live-phase]").inner_text() == "Paused"
                assert card.locator("[data-live-phase]").inner_text() == "Done"
                context.close()
        finally:
            browser.close()


def seed_rotated_tool_history(root):
    """A finished 14-call run whose rows rotated into the oldest of five archives (one history read
    opens the newest three), beside a run whose start alone is in the live log (#2400)."""
    stamp = "2026-10-28T06:3{}:00Z".format
    for task_id, ask, answer, name, counts, minute in (
            ("archived-run", "Run the scheduled health check", "Health check finished: every service answered.",
             "Scheduled health check", {"run_command": 6, "read_file": 5, "web_search": 3}, 0),
            ("started-run", "Fetch the release notes", "Stopped before the fetch returned.",
             "Release notes fetch", {"web_fetch": 1}, 5)):
        dialogue_store.append(root, "chat", {"direction": "in", "chat_id": 1, "text": ask, "ts": stamp(minute)})
        dialogue_store.append(root, "chat", {"direction": "out", "chat_id": 1, "task_id": task_id, "text": answer,
                                             "ts": stamp(minute + 4)})
        dialogue_store.append(root, "chat", {"direction": "system", "type": "task_summary", "chat_id": 1,
            "task_id": task_id, "text": answer, "tool_calls": sum(counts.values()), "tool_errors": 0,
            "tool_call_counts": counts, "ts": stamp(minute + 4)})
        seed_result(root, task_id, {"_schema_version": 1, "task_id": task_id, "status": "completed", "chat_id": 1,
                                    "suggested_name": name, "tool_calls": sum(counts.values())})
    def row(kind, task_id, invocation, tool, minute=1):
        return json.dumps({"type": kind, "task_id": task_id, "invocation_id": invocation, "tool": tool,
                           "is_error": False, "ts": stamp(minute)}) + "\n"
    archive = root / "archive"
    archive.mkdir(exist_ok=True)
    tools = [tool for tool, n in {"run_command": 6, "read_file": 5, "web_search": 3}.items() for _ in range(n)]
    (archive / "tools_20261028T063749.jsonl").write_text("".join(row(kind, "archived-run", f"call-{n}", tool)
        for n, tool in enumerate(tools) for kind in ("tool_call_started", "tool_call")), encoding="utf-8")
    for later in ("064239", "070101", "073012", "080455"):
        (archive / f"tools_20261028T{later}.jsonl").write_text(row("tool_call", f"other-{later}", later, "read_file"),
                                                                encoding="utf-8")
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    (logs / "tools.jsonl").write_text(row("tool_call_started", "started-run", "fetch-1", "web_fetch", 6), encoding="utf-8")


def _reconnect(page):
    """A real socket close re-enters the production reconnect and history path, without a reload."""
    page.wait_for_function("() => window.__ouroWs?._lastSha && window.__ouroWs?.ws?.readyState === 1", timeout=30000)
    page.evaluate("window.__sameDocument = true; window.__oldSocket = window.__ouroWs.ws")
    with page.expect_response(lambda response: urllib.parse.urlsplit(response.url).path == "/api/chat/history",
                              timeout=60000):
        page.evaluate("window.__oldSocket.close()")
    page.wait_for_function("() => window.__ouroWs.ws !== window.__oldSocket && window.__ouroWs.ws?.readyState === 1",
                           timeout=30000)
    assert page.evaluate("window.__sameDocument === true"), "reconnect reloaded the document"


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_rotated_out_tool_history_keeps_host_totals(direct_server_with_data, engine):
    """#2400 in the real Chat: the finished run keeps "14 tool calls" with the bounded-read disclosure on
    load, reload and reconnect, never "outcome unknown"; a run whose start alone was read keeps it."""
    from playwright.sync_api import expect, sync_playwright

    fixture = direct_server_with_data
    root = fixture["data_dir"]
    fixture["stop_server"]()
    seed_rotated_tool_history(root)
    fixture["start_server"]()
    evidence = diagnostic_directory(root.parent, "task-evidence")
    with sync_playwright() as pw:
        browser = getattr(pw, engine).launch(headless=True)
        try:
            for width, height, theme in ((1100, 735, "light"), (390, 844, "dark")):
                context = browser.new_context(viewport={"width": width, "height": height}, color_scheme=theme,
                                              is_mobile=width == 390, has_touch=width == 390)
                page = context.new_page()
                errors = PageErrors(page, engine)
                page.goto(fixture["url"], wait_until="domcontentloaded")
                for step in ("load", "reload", "reconnect"):
                    if step == "reload":
                        page.reload(wait_until="domcontentloaded")
                    elif step == "reconnect":
                        _reconnect(page)
                    rows = {}
                    for task_id in ("archived-run", "started-run"):
                        card = page.locator(f'.chat-live-card[data-task-id="{task_id}"]')
                        card.wait_for(state="visible")
                        page.wait_for_function("id => document.querySelector(`.chat-live-card[data-task-id='${id}']`)"
                                               "?.dataset.finished === '1'", arg=task_id)
                        if card.get_attribute("data-expanded") != "1":
                            card.locator(":scope > [data-live-summary-button] [data-live-disclosure]").click()
                        rows[task_id] = card.locator(".chat-live-line", has_text="tool call").first
                    expect(rows["archived-run"]).to_contain_text("14 tool calls")
                    assert "outcome unknown" not in rows["archived-run"].inner_text(), step
                    expect(rows["started-run"]).to_contain_text("1 tool call · outcome unknown")
                rows["archived-run"].locator("[data-live-line-toggle]").click()
                body = rows["archived-run"].locator(".chat-live-line-body")
                expect(body).to_contain_text("Only recent tool history was read. Source: logs/tools.jsonl.")
                assert "incomplete" not in body.inner_text()
                assert page.locator("#chat-messages").evaluate("el => el.scrollWidth <= el.clientWidth + 1")
                assert not errors.unexpected(), errors.report()
                rows["archived-run"].scroll_into_view_if_needed()
                page.screenshot(path=str(evidence / f"rotated-tool-history-{engine}-{width}-{theme}.png"))
                context.close()
        finally:
            browser.close()


def test_pause_notice_translation_and_historical_wake_status(direct_server_with_data):
    from playwright.sync_api import expect, sync_playwright

    fixture = direct_server_with_data
    root = fixture["data_dir"]
    english = "The task paused because of the owner's Pause. Its work is retained. Use Resume on the task when available."
    translated = "Задача приостановлена владельцем. Работа сохранена. Продолжение доступно в карточке задачи."
    fixture["stop_server"]()
    notices = [{"direction": "system", "type": "task_pause_notice", "task_id": task,
                "card_row": "timeline", "card_row_id": f"pause:{task}:one", "narration": False,
                "text": english, "chat_id": 1, "ts": "2026-10-04T12:00:00Z"}
               for task in ("with-card", "without-card")]
    for row in notices:
        dialogue_store.append(root, "chat", row)
    dialogue_store.append(root, "progress", {"task_id": "with-card", "chat_id": 1,
        "content": "Model words stay unchanged", "ts": "2026-10-04T11:59:00Z"})
    fixture["start_server"]()
    with urllib.request.urlopen(fixture["url"] + "/api/state", timeout=10) as response:
        state = json.load(response)
    status = "wake_paused"
    owner_wait = {"owner_wait_state": "waiting", "quiz_state": "open"}
    details = {"wake_paused": "The last wake-up returned while paused; its task card shows the current state. Next wake check at 18:00.",
               "wake_outcome_unknown": "The last wake-up's outcome is unconfirmed; next check at 18:00."}
    payload = {"language": "ru", "english": False, "entries": {
        english: {"text": translated, "provenance": "imported"},
        "code:consciousness.wake_paused": {"text": "приостановлено", "provenance": "imported"},
        "code:consciousness.wake_outcome_unknown": {"text": "исход неизвестен", "provenance": "imported"}}}
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        try:
            page = browser.new_page(viewport={"width": 1280, "height": 900})
            def rows():
                return [{"activity_id": "with-card", "chat_id": 1, "kind": "managed_task",
                         "phase": "working", "owner_wait": owner_wait, "required_question_unavailable": True}]
            page.route(state_reads, lambda route: route.fulfill(json={**state, "bg_consciousness_enabled": True,
                "active_chat_activities": rows(), "active_chat_activities_complete": True, **work_fields(rows()),
                "bg_consciousness_state": {"status": status, "detail": details[status]}}))
            page.route("**/api/ui/i18n", lambda route: route.fulfill(json=payload))
            page.goto(fixture["url"], wait_until="domcontentloaded")
            card = page.locator('.chat-live-card[data-task-id="with-card"]')
            card.wait_for(state="visible")
            if card.get_attribute("data-expanded") != "1":
                card.locator(":scope > [data-live-summary-button] [data-live-disclosure]").click()
            expect(card).to_contain_text(translated)
            expect(card).to_contain_text("Model words stay unchanged")
            fallback = page.locator('.chat-bubble.system[data-task-id="without-card"]')
            expect(fallback).to_contain_text(translated)
            assert english in dialogue_test.log_text(root / "logs/chat.jsonl")
            page.screenshot(path=str(diagnostic_directory(root.parent, "task-evidence") / "pause-notice-translated-chat.png"), full_page=True)
            for wait_state, quiz_state, label in (("waiting", "open", "Waiting for your answer"),
                ("waiting", "answered", "Working"), ("resumed", "answered", "Working")):
                owner_wait = {"owner_wait_state": wait_state, "quiz_state": quiz_state}
                page.reload(wait_until="domcontentloaded")
                expect(card.locator("[data-live-phase]")).to_have_text(label)
                expect(page.locator("#chat-status")).not_to_have_text("Activity unconfirmed")
                page.screenshot(path=str(diagnostic_directory(root.parent, "task-evidence") / f"main-wait-{wait_state}-{quiz_state}.png"), full_page=True)
            page.locator('[data-nav-page="dashboard"]').click()
            page.locator('[data-dashboard-tab="evolution"]').click()
            sentence = page.locator("#evo-consciousness")
            for next_status, label in (("wake_paused", "приостановлено"), ("wake_outcome_unknown", "исход неизвестен")):
                status = next_status
                page.evaluate("() => window.dispatchEvent(new CustomEvent('ouro:dashboard-subtab-shown', {detail: {tab: 'evolution'}}))")
                expect(sentence).to_have_text(f"Consciousness {label}")
                expect(sentence).to_have_attribute("data-tone", "neutral")  # goes on by itself: never the owner's amber
                expect(page.locator("#evo-consciousness-detail")).to_contain_text(details[status])
                page.screenshot(path=str(diagnostic_directory(root.parent, "task-evidence") / f"evolution-{status}.png"), full_page=True)
        finally:
            browser.close()
