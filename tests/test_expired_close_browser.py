"""A park past its hard deadline in the real SPA (docs/DESIGN.md "Room work status"), over synthetic census rows.

The shipped client (``web/``) is served as it is and every ``/api/*`` read is answered at the route, so no Ouroboros
server, task, engine or credential is involved and nothing is closed or resumed for real. What is checked is what the
owner sees and can press: Main's card and pill, the room's work list, Activity's late row and queue row, and the Panic
choice, each with ``Deadline expired`` as a status, one ``Close`` at its exact task, and the Stop a row is still
granted. Screenshots go to ``OUROBOROS_UI_EVIDENCE_DIR`` when it is set.
"""
from __future__ import annotations

import copy
import json
import os
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from urllib.parse import urlsplit

import pytest

pytestmark = [pytest.mark.ui_browser]
WEB = Path(__file__).resolve().parents[1] / "web"
DESKTOP = {"width": 1100, "height": 800}
PHONE = {"width": 390, "height": 844}
TOKEN = "a" * 64
OTHER = "b" * 64
EMPTY_HISTORY = {"messages": [], "progress": [], "has_more": False, "page_cursor": "fixture-page",
                 "coverage": {"v": 1, "view": "main", "upper": {"chat": 0},
                              "spans": {"chat": {"chain": "empty", "from": 0, "to": 0, "gaps": []}, "progress": None}}}
EXPIRED = {"code": "deadline_expired", "text": "Deadline expired"}
PAUSED = {"code": "paused_by_owner", "text": "You paused it", "since": "2026-11-23T07:06:00Z"}
WAITING = {"code": "awaiting_selection", "text": "Waiting for Ouroboros or you to resume it"}


def close(target, token=TOKEN, scope="single"):
    return {"action": "close", "target": target, "scope": scope, "close_expired": token}


def stop(target, scope="cascade"):
    return {"action": "stop", "target": target, "scope": scope}


def work(task_id, name, *, activity="paused", because=(), controls=(), live=True, members=None, confirmed=True):
    return {"work_id": task_id, "room": 1, "name": name, "kind": "task", "project_id": "",
            "episode": {"task_id": task_id, "live": live}, "activity": activity, "transition": "none", "call": "",
            "because": list(because), "tails": [], "turn": [], "release": "none", "members": members or {"ids": [task_id]},
            "controls": list(controls), "since": 0, "confirmed": confirmed,
            "mark": "ring" if not confirmed else ("moving" if activity == "working" else "pause"),
            "live": activity == "working"}


def attention(task_id, name, **patch):
    return {"task_id": task_id, "activity_id": task_id, "task_name": name, "project_id": "", "chat_id": 1,
            "phase": "budget_paused", "pause_cause": "budget", "can_resume": True, "cancelable": True,
            "attention_reason": "budget", "model_waits": {}, **patch}


class Core:
    """The census and the doors, synthetic: every POST is recorded, and a Close removes its row."""

    def __init__(self, **state):
        self.extra = state
        self.history = copy.deepcopy(EMPTY_HISTORY)
        self.queue = {"running": [], "pending": []}
        self.cancels: list[tuple[str, dict]] = []
        self.resumes: list[tuple[str, dict]] = []
        self.writes: list[tuple[str, dict]] = []
        self.cancel_reply = None

    def state(self):
        body = {"sha": "fixture", "supervisor_ready": True, "projects": [], "project_chat_ids": [],
                "active_chat_activities": [], "active_chat_activities_complete": True,
                "current_questions": [], "current_questions_complete": True,
                "main_attention": [], "main_attention_complete": True,
                "works": [], "rooms": {}, "works_complete": True, "works_gaps": [],
                "work_summary": {"counts": {}, "turn": 0, "confirmed": True},
                "panic_recovery": [], "panic_recovery_complete": True}
        body.update(copy.deepcopy(self.extra))
        return body

    def closed(self, task_id):
        for key in ("main_attention", "panic_recovery", "active_chat_activities"):
            self.extra[key] = [row for row in self.extra.get(key, [])
                               if task_id not in (row.get("task_id"), row.get("activity_id"))]
        for item in self.extra.get("works", []):
            item["controls"] = [c for c in item.get("controls", []) if c.get("target") != task_id]
            states = (item.get("members") or {}).get("states") or {}
            if task_id in states:
                states[task_id].pop("close_expired", None)


@pytest.fixture(scope="module")
def _spa_browser():
    playwright = pytest.importorskip("playwright.sync_api")

    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            self.path = self.path.removeprefix("/static")
            super().do_GET()

    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(WEB)))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with playwright.sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True,
                                         executable_path=os.environ.get("OUROBOROS_UI_BROWSER_EXECUTABLE") or None)
            yield browser, server.server_port
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture
def spa(_spa_browser):
    browser, port = _spa_browser
    opened = []

    def open_page(core, *, viewport=DESKTOP, theme="dark", ready="#chat-messages"):
        context = browser.new_context(viewport=viewport)
        page = context.new_page()
        opened.append(context)
        errors: list[str] = []
        page.on("pageerror", lambda error: errors.append(str(error)))

        def api(route):
            request = route.request
            path = urlsplit(request.url).path
            if request.method == "POST" and path.startswith("/api/tasks/") and path.endswith("/cancel"):
                task_id = path.split("/")[3]
                core.cancels.append((task_id, request.post_data_json))
                core.closed(task_id)
                reply = (core.cancel_reply(task_id, request.post_data_json) if callable(core.cancel_reply)
                         else core.cancel_reply) if core.cancel_reply is not None else {"ok": True, "task_id": task_id}
                route.fulfill(content_type="application/json", body=json.dumps(reply))
                return
            if request.method == "POST" and path.startswith("/api/tasks/") and path.endswith("/resume"):
                task_id = path.split("/")[3]
                core.resumes.append((task_id, request.post_data_json))
                core.extra["panic_recovery"] = [
                    row for row in core.extra.get("panic_recovery", []) if row.get("task_id") != task_id]
                route.fulfill(content_type="application/json", body=json.dumps({"ok": True, "task_id": task_id}))
                return
            if path == "/api/onboarding":
                route.fulfill(status=204)
                return
            if request.method == "POST":
                core.writes.append((path, request.post_data_json if request.post_data else {}))
                route.fulfill(content_type="application/json", body="{}")
                return
            body = core.state() if path == "/api/state" else {
                "/api/health": {"status": "ok", "version": "fixture", "runtime_version": "fixture", "app_version": "fixture"},
                "/api/chat/history": core.history,
                "/api/projects": {"projects": []},
                "/api/tasks": {"queue": core.queue, "tasks": []},
                "/api/services/kept": {"services": []},
                "/api/schedules": {"schedules": []},
            }.get(path, {})
            route.fulfill(content_type="application/json", body=json.dumps(body))

        page.add_init_script(f"localStorage.setItem('ouroboros.theme', {json.dumps(theme)});")
        page.route_web_socket("**/ws", lambda ws: ws.send(json.dumps({"type": "heartbeat"})))
        page.route("**/api/**", api)
        page.goto(f"http://127.0.0.1:{port}/", wait_until="domcontentloaded")
        page.locator(ready).first.wait_for(state="visible", timeout=30_000)
        return page, errors

    yield open_page
    for context in opened:
        context.close()


def shot(page, name, locator=None):
    root = os.environ.get("OUROBOROS_UI_EVIDENCE_DIR")
    if not root:
        return
    target = Path(root)
    target.mkdir(parents=True, exist_ok=True)
    # Write bytes on the test thread instead of passing path= to Playwright (which spawns an asyncio executor thread).
    (target / f"{name}.png").write_bytes((locator or page).screenshot(animations="disabled"))


def buttons(page, scope, text):
    return page.locator(f"{scope} button", has_text=text)


def exact(page, scope, text):
    return page.locator(scope).get_by_role("button", name=text, exact=True)


# ---------------------------------------------------------------- Main: card, pill, a crossing repaint
@pytest.mark.parametrize("theme", ["dark", "light"])
def test_main_card_keeps_its_stop_beside_close_and_never_counts_as_waiting(spa, theme):
    refusal = {"at": "2026-11-23T08:00:00Z", "by": {"actor": "owner"}, "error": "deadline_passed", "text": "Deadline passed."}
    helper = attention("task-helper", "Windows window check", attention_reason="deadline_expired", can_resume=False,
                       close_expired=close("task-helper", scope="cascade"))
    need = attention("task-spend", "Sweep the weekly spend")
    helper_member = {"confirmed": True, "activity": "paused", "transition": "none", "call": "", "release": "none",
                     "turn": [], "tails": [], "because": [EXPIRED, PAUSED, WAITING], "last_resume": refusal, "mark": "pause"}
    core = Core(main_attention=[helper, need],
                works=[work("task-helper", "Windows window check", activity="paused", because=[EXPIRED, PAUSED, WAITING],
                            controls=[close("task-helper", scope="cascade"), stop("task-helper")],
                            members={"ids": ["task-helper"], "states": {"task-helper": helper_member}})])
    page, errors = spa(core, theme=theme, ready=".owner-action-body")
    card = '.owner-action-body[data-task-id="task-helper"]'
    page.locator(card).wait_for(state="visible")
    assert "Deadline expired" in page.locator(card).inner_text()
    reason_text = page.locator(f"{card} .owner-action-reason").inner_text()
    assert "You paused it" in reason_text
    assert reason_text.count("Resume refused at") == 1, reason_text
    assert "Deadline passed." in reason_text
    close_btn = exact(page, card, "Close")
    assert close_btn.is_visible() and exact(page, card, "Stop…").is_visible()
    described = close_btn.get_attribute("aria-describedby")
    assert described and page.locator(f"#{described}").inner_text() == "Windows window check"
    assert page.locator(f"{card} button", has_text="Resume").count() == 0
    # Only the spending-limit card waits for the owner; the optional Close is not counted beside it.
    page.evaluate("() => { const feed = document.querySelector('#chat-messages'); feed.scrollTop = feed.scrollHeight; }")
    pill = page.locator(".chat-question-nav")
    assert pill.count() == 0 or "2" not in pill.inner_text()
    shot(page, f"main-card-{theme}", page.locator("#chat-messages"))
    close_btn.click()
    page.locator(card).wait_for(state="detached", timeout=15_000)
    assert core.cancels == [("task-helper", {"stop_policy": "close_expired", "close_expired": TOKEN})]
    assert not errors


def test_main_card_that_crosses_its_deadline_drops_every_dead_action(spa):
    before = attention("task-helper", "Windows window check")
    core = Core(main_attention=[before])
    page, errors = spa(core, ready=".owner-action-body")
    card = '.owner-action-body[data-task-id="task-helper"]'
    assert exact(page, card, "Resume").is_visible() and exact(page, card, "Stop…").is_visible()
    core.extra["main_attention"] = [attention("task-helper", "Windows window check", attention_reason="deadline_expired",
                                              can_resume=False, cancelable=False,
                                              close_expired=close("task-helper", scope="cascade"))]
    page.evaluate("() => window.dispatchEvent(new Event('focus'))")
    exact(page, card, "Close").wait_for(state="visible", timeout=30_000)
    assert page.locator(f"{card} button", has_text="Stop").count() == 0
    assert page.locator(f"{card} button", has_text="Resume").count() == 0
    assert not page.locator(".chat-question-nav").is_visible()
    shot(page, "main-card-crossed", page.locator("#chat-messages"))
    assert not errors


# ---------------------------------------------------------------- Room: the work list
def test_room_list_draws_close_only_at_its_own_episode(spa):
    root = work("task-report", "Review weekly report", activity="paused", because=[EXPIRED, PAUSED],
                controls=[close("task-report", scope="cascade"), stop("task-report")])
    busy = work("task-busy", "Index the archive", activity="working",
                controls=[close("helper-a"), close("helper-b", OTHER), stop("task-busy")])
    core = Core(works=[root, busy], rooms={"1": {"activity": "working", "transition": "none", "call": "", "turn": [],
                                                  "amber": False, "mark": "moving", "confirmed": True,
                                                  "works": ["task-report", "task-busy"]}})
    page, errors = spa(core, ready=".rw-trigger")
    page.locator(".rw-trigger").first.click()
    rows = page.locator(".rw-row")
    rows.first.wait_for(state="visible")
    report = page.locator('.rw-row[data-id="task-report"]')
    index = page.locator('.rw-row[data-id="task-busy"]')
    assert report.get_by_role("button", name="Close", exact=True).count() == 1
    assert index.get_by_role("button", name="Close", exact=True).count() == 0, "a helper's Close is not drawn beside another name"
    assert "Deadline expired" in report.inner_text() and "resume it" not in report.inner_text()
    shot(page, "room-list", page.locator(".rw-surface").first)
    report.get_by_role("button", name="Close", exact=True).click()
    page.wait_for_timeout(500)
    assert core.cancels == [("task-report", {"stop_policy": "close_expired", "close_expired": TOKEN})]
    assert not errors


# ---------------------------------------------------------------- Activity: late row, healthy paused & expired groups
def test_activity_late_row_says_deadline_expired_and_keeps_stop(spa):
    late_work = work("late-root", "Weekly report", activity="paused", because=[EXPIRED, PAUSED, WAITING],
                     controls=[close("late-root", scope="cascade"), stop("late-root")])
    helper = work("task-helper", "Windows window check", activity="paused", because=[EXPIRED, WAITING],
                  controls=[close("task-helper", scope="cascade")])
    healthy = work("task-healthy", "Draft quarterly summary", activity="paused", because=[PAUSED],
                   controls=[{"action": "resume", "target": "task-healthy", "scope": "cascade"}, stop("task-healthy")])
    core = Core(works=[late_work, helper, healthy],
                active_chat_activities=[{"activity_id": "late-root", "kind": "direct_chat", "phase": "budget_paused",
                                         "started_at": 1_700_000_000, "cancelable": True, "chat_id": 1,
                                         "late_work": {"title": "Weekly report", "answered_at": "2026-11-23T06:00:00Z"},
                                         "close_expired": close("late-root", scope="cascade")},
                                        # A census-only row: no work or member names it, only its own Close offer.
                                        {"activity_id": "census-root", "kind": "managed_task", "phase": "budget_paused",
                                         "started_at": int(time.time()) - 7200, "cancelable": False, "chat_id": 1,
                                         "suggested_name": "Census only task",
                                         "close_expired": close("census-root", OTHER, "cascade")}])
    core.queue = {"running": [], "pending": [
        {"id": "task-healthy", "type": "task", "attempt": 1,
         "task": {"id": "task-healthy", "suggested_name": "Draft quarterly summary"}},
        {"id": "task-helper", "type": "task", "attempt": 1,
         "task": {"id": "task-helper", "suggested_name": "Windows window check"}},
    ]}
    page, errors = spa(core)
    page.locator('[data-nav-page="dashboard"]').click()
    page.locator('[data-dashboard-tab="activity"]').click()
    section = '[data-activity-section="queue"]'
    page.locator(f"{section} .activity-row").first.wait_for(state="visible", timeout=30_000)
    paused_group = page.locator(f'{section} details[data-open-key="paused"]')
    expired_group = page.locator(f'{section} details[data-open-key="expired"]')
    assert paused_group.locator("> summary").inner_text() == "Paused (1)"
    assert "Draft quarterly summary" in paused_group.inner_text()
    assert "Deadline expired" not in paused_group.inner_text()
    assert expired_group.locator("> summary").inner_text() == "Deadline expired (3)"
    text = expired_group.inner_text()
    assert text.count("Deadline expired") >= 4, text  # summary + 3 rows
    assert "Waiting for Ouroboros or you to resume it" not in text
    assert "You paused it" in text
    shot(page, "activity-rows", page.locator(section))
    assert page.locator(f"{section} [data-act='close-expired']").count() == 3
    # The census-only row's Close is pressable: the press re-reads that same exact offer, never only a work's.
    only = page.locator(f"{section} [data-act='close-expired'][data-id='census-root']")
    assert only.get_attribute("aria-describedby")
    only.click()
    page.wait_for_timeout(500)
    assert core.cancels == [("census-root", {"stop_policy": "close_expired", "close_expired": OTHER})]
    assert not errors


# ---------------------------------------------------------------- Panic: mobile fit (mixed + all-expired live in test_panic_recovery_browser.py)
def _panic_rows(*ids):
    rows = []
    for index, (task_id, name) in enumerate(ids):
        rows.append({"task_id": task_id, "task_name": name, "panic_hold_id": f"hold-{index}", "chat_id": 1})
    return rows


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_phone_cards_fit(spa, theme):
    rows = _panic_rows(("task-a", "Summarise this week's reading list"), ("task-c", "Check the release notes for broken links"))
    rows[1]["close_expired"] = close("task-c", scope="cascade")
    core = Core(panic_recovery=rows, main_attention=[
        attention(r["task_id"], r["task_name"], pause_cause="panic", attention_reason="resume") for r in rows])
    page, errors = spa(core, viewport=PHONE, theme=theme, ready=".panic-recovery")
    message = page.locator(".panic-recovery")
    message.wait_for(state="visible")
    assert page.evaluate("() => document.scrollingElement.scrollWidth <= document.scrollingElement.clientWidth")
    shot(page, f"expired-panic-phone-{theme}", message)
    assert not errors


# ---------------------------------------------------------------- Chat card: authoritative controls & helper Close descriptions
@pytest.mark.parametrize("theme", ["dark", "light"])
def test_chat_card_uses_authoritative_controls_and_describes_each_helper_close(spa, theme):
    """A managed root card with a Close-only work offer renders no dummy legacy Stop, keeps granted Stop when
    the server grants one, and describes each expired helper Close button by its own instance-unique card title."""
    root_id = "root-release"
    helper_a = "helper-win11111"
    helper_b = "helper-mac22222"
    member_a = {"confirmed": True, "activity": "paused", "transition": "none", "call": "", "release": "none",
                "turn": [], "tails": [], "because": [EXPIRED, WAITING], "mark": "pause",
                "close_expired": close(helper_a, TOKEN)}
    member_b = {"confirmed": True, "activity": "paused", "transition": "none", "call": "", "release": "none",
                "turn": [], "tails": [], "because": [EXPIRED, WAITING], "mark": "pause",
                "close_expired": close(helper_b, OTHER)}
    root_work = work(
        root_id, "Release verification", activity="paused", because=[EXPIRED, PAUSED],
        controls=[close(root_id, TOKEN, scope="cascade")],
        members={"ids": [root_id, helper_a, helper_b],
                 "states": {
                     root_id: {"confirmed": True, "activity": "paused", "transition": "none", "call": "",
                               "release": "none", "turn": [], "tails": [], "because": [EXPIRED, PAUSED], "mark": "pause"},
                     helper_a: member_a,
                     helper_b: member_b,
                 }},
    )
    core = Core(
        works=[root_work],
        rooms={"1": {"activity": "paused", "transition": "none", "call": "", "turn": [], "amber": False,
                     "mark": "pause", "confirmed": True, "works": [root_id]}},
        active_chat_activities=[{"activity_id": root_id, "kind": "managed_task", "phase": "budget_paused",
                                 "started_at": 1_700_000_000, "cancelable": False, "chat_id": 1,
                                 "suggested_name": "Release verification", "deadline_expired": True,
                                 "close_expired": close(root_id, TOKEN, "cascade")}],
    )
    core.history = {
        **EMPTY_HISTORY,
        "messages": [
            {"ts": "2026-11-23T07:05:00Z", "role": "assistant", "task_id": root_id, "is_progress": True,
             "cancelable": True, "suggested_name": "Release verification", "text": "Checking release artifacts.",
             "content": "Checking release artifacts."},
            {"ts": "2026-11-23T07:05:10Z", "role": "assistant", "task_id": helper_a, "parent_task_id": root_id,
             "root_task_id": root_id, "subagent_event": "started", "subagent_task_id": helper_a,
             "delegation_role": "subagent", "subagent_role": "Verifier", "is_progress": True,
             "text": "Running Windows check.", "content": "Running Windows check."},
            {"ts": "2026-11-23T07:05:20Z", "role": "assistant", "task_id": helper_b, "parent_task_id": root_id,
             "root_task_id": root_id, "subagent_event": "started", "subagent_task_id": helper_b,
             "delegation_role": "subagent", "subagent_role": "Verifier", "is_progress": True,
             "text": "Running macOS check.", "content": "Running macOS check."},
        ],
    }
    page, errors = spa(core, theme=theme, ready=f'.chat-live-card[data-task-id="{root_id}"]')
    root_card = f'.chat-live-card[data-task-id="{root_id}"]'
    card_a = f'.chat-live-card.subagent[data-task-id="{helper_a}"]'
    card_b = f'.chat-live-card.subagent[data-task-id="{helper_b}"]'
    page.locator(f"{root_card} > .chat-live-foot [data-close-expired-run]").wait_for(state="visible", timeout=15_000)
    # Close-only authoritative work: no dummy legacy Stop… on the managed root card.
    assert page.locator(f"{root_card} > .chat-live-foot [data-cancel-run]").count() == 0
    # Two expired helper cards: each Close button is described by its own unique card title.
    page.locator(f"{card_a} [data-close-expired-run]").wait_for(state="visible", timeout=15_000)
    page.locator(f"{card_b} [data-close-expired-run]").wait_for(state="visible", timeout=15_000)
    desc_a = page.locator(f"{card_a} [data-close-expired-run]").get_attribute("aria-describedby")
    desc_b = page.locator(f"{card_b} [data-close-expired-run]").get_attribute("aria-describedby")
    desc_root = page.locator(f"{root_card} > .chat-live-foot [data-close-expired-run]").get_attribute("aria-describedby")
    assert desc_root and desc_a and desc_b and len({desc_root, desc_a, desc_b}) == 3
    title_a = page.locator(f"#{desc_a}").inner_text()
    title_b = page.locator(f"#{desc_b}").inner_text()
    assert helper_a[:8] in title_a and helper_b[:8] in title_b and title_a != title_b
    assert page.locator(f"#{desc_root}").inner_text() == "Release verification"
    shot(page, f"chat-card-close-only-{theme}", page.locator(root_card))

    # Now the server grants Stop beside Close on the root: Stop… appears and opens its menu with Stop now.
    core.extra["works"][0]["controls"] = [close(root_id, TOKEN, scope="cascade"), stop(root_id)]
    core.extra["active_chat_activities"][0]["cancelable"] = True
    page.evaluate("() => window.dispatchEvent(new Event('focus'))")
    root_stop = page.locator(f"{root_card} > .chat-live-foot [data-cancel-run]")
    root_stop.wait_for(state="visible", timeout=15_000)
    assert page.locator(f"{root_card} > .chat-live-foot [data-close-expired-run]").is_visible()
    shot(page, f"chat-card-close-and-stop-{theme}", page.locator(root_card))
    root_stop.click()
    menu = page.locator(".task-control-menu [data-task-control]")
    menu.first.wait_for(state="visible", timeout=10_000)
    assert menu.evaluate_all("items => items.map(item => item.dataset.taskControl)") == ["stop_now"]
    page.keyboard.press("Escape")

    # Pressing helper A's Close targets only helper A with its exact token.
    page.locator(f"{card_a} [data-close-expired-run]").click()
    page.locator(f"{card_a} [data-close-expired-run]").wait_for(state="detached", timeout=15_000)
    assert core.cancels == [(helper_a, {"stop_policy": "close_expired", "close_expired": TOKEN})]
    assert page.locator(f"{card_b} [data-close-expired-run]").is_visible()
    assert not errors


# ---------------------------------------------------------------- Activity: non-close expired helper Stop & result_kept toast
# (Partial-Panic authoritative expiry + Resume selection is pinned in web/tests/expired_close_surfaces.test.js.)
def test_activity_preserves_stop_on_non_close_expired_helper_without_enabling_unknown(spa):
    """Public queue + root-only census: a confirmed paused expired helper with matching attempt keeps Stop even when
    its owning root has NO Stop grant, while unknown/stale-attempt neighbors stay denied even when their root has Stop."""
    root_no_stop = "root-no-stop"
    root_with_stop = "root-coord"
    helper_stop = "helper-stop"
    helper_unknown = "helper-unknown"
    helper_stale = "helper-stale"
    no_stop_work = work(
        root_no_stop, "Ended parent", activity="paused", because=[EXPIRED, PAUSED], controls=[],
        members={
            "ids": [root_no_stop, helper_stop],
            "states": {
                root_no_stop: {"confirmed": True, "activity": "paused", "transition": "none", "call": "",
                               "release": "none", "turn": [], "tails": [], "because": [EXPIRED, PAUSED],
                               "queue_attempt": 1, "mark": "pause"},
                helper_stop: {"confirmed": True, "activity": "paused", "transition": "none", "call": "",
                              "release": "none", "turn": [], "tails": [], "because": [EXPIRED, PAUSED],
                              "queue_attempt": 1, "mark": "pause"},
            },
        },
    )
    stoppable_work = work(
        root_with_stop, "Release coordinator", activity="working", controls=[stop(root_with_stop)],
        members={
            "ids": [root_with_stop, helper_unknown, helper_stale],
            "states": {
                root_with_stop: {"confirmed": True, "activity": "working", "transition": "none", "call": "working",
                                 "live": True, "release": "self", "turn": [], "tails": [], "because": [],
                                 "queue_attempt": 1, "mark": "moving"},
                helper_unknown: {"confirmed": False, "activity": "unknown", "transition": "none", "call": "",
                                 "release": "none", "turn": [], "tails": [], "because": [],
                                 "queue_attempt": 1, "mark": "ring"},
                helper_stale: {"confirmed": True, "activity": "paused", "transition": "none", "call": "",
                               "release": "none", "turn": [], "tails": [], "because": [EXPIRED, PAUSED],
                               "queue_attempt": 2, "mark": "pause"},
            },
        },
    )
    core = Core(
        works=[no_stop_work, stoppable_work],
        active_chat_activities=[{"activity_id": root_with_stop, "kind": "managed_task", "phase": "working",
                                 "started_at": 1_700_000_000, "cancelable": True, "chat_id": 1, "task_attempt": 1,
                                 "suggested_name": "Release coordinator"}],
        main_attention=[attention(helper_unknown, "Unconfirmed neighbor", attention_reason="deadline_expired",
                                  deadline_expired=True, can_resume=False, cancelable=False)],
    )
    core.queue = {
        "running": [{"id": root_with_stop, "type": "task", "attempt": 1, "runtime_sec": 30,
                     "task": {"id": root_with_stop, "root_task_id": root_with_stop,
                              "suggested_name": "Release coordinator"}}],
        "pending": [
            {"id": helper_stop, "type": "task", "attempt": 1,
             "task": {"id": helper_stop, "root_task_id": root_no_stop, "parent_task_id": root_no_stop,
                      "delegation_role": "subagent", "suggested_name": "Windows window check"}},
            {"id": helper_unknown, "type": "task", "attempt": 1,
             "task": {"id": helper_unknown, "root_task_id": root_with_stop, "parent_task_id": root_with_stop,
                      "delegation_role": "subagent", "suggested_name": "Unconfirmed neighbor"}},
            {"id": helper_stale, "type": "task", "attempt": 1,
             "task": {"id": helper_stale, "root_task_id": root_with_stop, "parent_task_id": root_with_stop,
                      "delegation_role": "subagent", "suggested_name": "Stale attempt helper"}},
        ],
    }
    page, errors = spa(core)
    page.locator('[data-nav-page="dashboard"]').click()
    page.locator('[data-dashboard-tab="activity"]').click()
    section = '[data-activity-section="queue"]'
    page.locator(f"{section} .activity-row").first.wait_for(state="visible", timeout=30_000)

    stop_row = page.locator(f"{section} .activity-row", has_text="Windows window check")
    unknown_row = page.locator(f"{section} .activity-row", has_text="Unconfirmed neighbor")
    stale_row = page.locator(f"{section} .activity-row", has_text="Stale attempt helper")
    root_row = page.locator(f"{section} .activity-row", has_text="Release coordinator")
    assert "Deadline expired" in stop_row.inner_text()
    assert stop_row.locator('[data-act="close-expired"]').count() == 0
    assert stop_row.locator("button", has_text="Resume").count() == 0
    stop_btn = stop_row.locator(f'[data-act="task-control"][data-id="{helper_stop}"]')
    assert stop_btn.is_visible() and stop_btn.inner_text() == "Stop now"
    assert stop_btn.get_attribute("data-root") is None
    assert stop_btn.get_attribute("data-resume") is None
    assert root_row.locator(f'[data-act="task-control"][data-id="{root_with_stop}"][data-root="1"]').is_visible()
    assert page.locator(f'{section} details[data-open-key="paused"]').count() == 0
    assert page.locator(f'{section} details[data-open-key="expired"] > summary').inner_text() == "Deadline expired (2)"
    for denied in (unknown_row, stale_row):
        assert denied.locator('[data-act="task-control"]').count() == 0
        assert denied.locator('[data-act="close-expired"]').count() == 0
        assert denied.locator("button", has_text="Resume").count() == 0
    shot(page, "activity-expired-helper-stop-r2", page.locator(section))
    # Narrow dark viewport: named expired, healthy and stale rows stay readable without horizontal overflow.
    narrow, narrow_errors = spa(core, viewport=PHONE, theme="dark")
    narrow.locator('[data-mobile-nav-toggle]').first.click()
    narrow.locator('#primary-sidebar [data-nav-page="dashboard"]').click()
    narrow.locator('[data-dashboard-tab="activity"]').click()
    narrow.locator(f'{section} details[data-open-key="expired"]').wait_for(state="visible", timeout=30_000)
    assert narrow.evaluate("() => document.scrollingElement.scrollWidth <= document.scrollingElement.clientWidth")
    shot(narrow, "activity-expired-helper-narrow-dark", narrow.locator(section))
    assert not narrow_errors

    # Clicking the helper's Stop confirms and posts the exact ordinary immediate-Stop cancel target/body
    # (api_client.cancelTask omits stop_policy for 'immediate' -> {cascade: True, stop_action_id}, never close_expired).
    stop_btn.click()
    confirm = page.locator(".confirm-dialog")
    confirm.wait_for(state="visible", timeout=10_000)
    confirm.get_by_role("button", name="Stop now", exact=True).click()
    page.wait_for_timeout(500)
    assert len(core.cancels) == 1
    target_id, body = core.cancels[0]
    assert target_id == helper_stop
    assert body.get("cascade") is True
    assert "stop_policy" not in body and "close_expired" not in body
    assert isinstance(body.get("stop_action_id"), str) and body["stop_action_id"].startswith("stop-")
    assert set(body.keys()) == {"cascade", "stop_action_id"}
    assert not core.resumes and not core.writes and not errors


def test_activity_rechecks_expired_helper_stop_after_crossed_repaint(spa):
    """If an expired helper's attempt changes while its Stop confirmation dialog is open, the crossed repaint drops
    the button and confirming the open dialog submits nothing."""
    root_no_stop = "root-no-stop"
    helper_stop = "helper-stop"
    core = Core(
        works=[work(
            root_no_stop, "Ended parent", activity="paused", because=[EXPIRED, PAUSED], controls=[],
            members={
                "ids": [root_no_stop, helper_stop],
                "states": {
                    helper_stop: {"confirmed": True, "activity": "paused", "transition": "none", "call": "",
                                  "release": "none", "turn": [], "tails": [], "because": [EXPIRED, PAUSED],
                                  "queue_attempt": 1, "mark": "pause"},
                },
            },
        )],
        active_chat_activities=[],
    )
    core.queue = {
        "running": [],
        "pending": [{"id": helper_stop, "type": "task", "attempt": 1,
                     "task": {"id": helper_stop, "root_task_id": root_no_stop, "parent_task_id": root_no_stop,
                              "delegation_role": "subagent", "suggested_name": "Windows window check"}}],
    }
    page, errors = spa(core)
    page.locator('[data-nav-page="dashboard"]').click()
    page.locator('[data-dashboard-tab="activity"]').click()
    section = '[data-activity-section="queue"]'
    stop_btn = page.locator(f'{section} [data-act="task-control"][data-id="{helper_stop}"]')
    stop_btn.wait_for(state="visible", timeout=30_000)

    # Open the Stop confirmation dialog while attempt 1 is confirmed.
    stop_btn.click()
    confirm = page.locator(".confirm-dialog")
    confirm.wait_for(state="visible", timeout=10_000)

    # Before the owner confirms, queue advances to attempt 2 (census stays at 1) and Activity refreshes.
    core.queue["pending"][0]["attempt"] = 2
    page.evaluate("() => window.dispatchEvent(new CustomEvent('ouro:dashboard-subtab-shown', { detail: { tab: 'activity' } }))")
    stop_btn.wait_for(state="detached", timeout=15_000)
    shot(page, "activity-expired-helper-crossed-r2", page.locator(section))

    # Confirming the already-open dialog must recheck and submit NO cancel request.
    confirm.get_by_role("button", name="Stop now", exact=True).click()
    page.wait_for_timeout(500)
    assert core.cancels == [] and not core.resumes and not core.writes and not errors


def test_close_expired_acknowledges_preserved_result_distinctly(spa):
    """A 200 close_expired response with reason_code=result_kept acknowledges the preserved result instead of Closed."""
    late = attention("task-kept", "Weekly summary", attention_reason="deadline_expired", can_resume=False,
                     cancelable=False, close_expired=close("task-kept", scope="cascade"))
    core = Core(main_attention=[late])
    core.cancel_reply = {"ok": True, "task_id": "task-kept", "request_id": "ci-kept", "reason_code": "result_kept"}
    page, errors = spa(core, ready=".owner-action-body")
    card = '.owner-action-body[data-task-id="task-kept"]'
    exact(page, card, "Close").click()
    toast = page.locator(".toast").first
    toast.wait_for(state="visible", timeout=10_000)
    toast_text = toast.inner_text()
    assert "The saved result was kept." in toast_text and "Closed." not in toast_text, toast_text
    shot(page, "close-result-kept-toast", page)
    assert core.cancels == [("task-kept", {"stop_policy": "close_expired", "close_expired": TOKEN})]
    assert not errors


def test_expired_quiz_card_keeps_answer_saved_without_promising_resume(spa):
    """Confirmed per-member expiry survives an unrelated unknown sibling (aggregate work confirmed=False) and shows
    truthful saved status with no promise of Resume, while healthy paused and unknown recipients keep their wording."""
    saved_delivery = {"state": "owed", "reason_code": "awaiting_recipient_read"}
    core = Core(
        works=[
            # Real server shape (gateway/work_status.py:661): unknown sibling makes the aggregate work unconfirmed.
            work(
                "task-exp", "Release check", activity="unknown", because=[], confirmed=False,
                controls=[close("task-exp", TOKEN, scope="cascade")],
                members={
                    "ids": ["task-exp", "task-unknown"],
                    "states": {
                        "task-exp": {"confirmed": True, "activity": "paused", "transition": "none", "call": "",
                                     "release": "none", "turn": [], "tails": [], "because": [EXPIRED, PAUSED], "mark": "pause"},
                        "task-unknown": {"confirmed": False, "activity": "paused", "transition": "none", "call": "",
                                         "release": "none", "turn": [], "tails": [], "because": [EXPIRED], "mark": "ring"},
                    },
                },
            ),
            work(
                "task-healthy", "macOS sign-off", activity="paused", because=[PAUSED], confirmed=True,
                controls=[{"action": "resume", "target": "task-healthy", "scope": "cascade"}],
                members={
                    "ids": ["task-healthy"],
                    "states": {
                        "task-healthy": {"confirmed": True, "activity": "paused", "transition": "none", "call": "",
                                         "release": "owner", "turn": [], "tails": [], "because": [PAUSED], "mark": "pause"},
                    },
                },
            ),
        ],
    )
    core.history = {
        **EMPTY_HISTORY,
        "messages": [
            {"ts": "2026-11-23T07:05:00Z", "role": "assistant", "msg_type": "quiz", "task_id": "task-unknown",
             "text": "Upload the archive?",
             "quiz": {"quiz_id": "qz-unknown", "question": "Upload the archive?",
                      "options": [{"label": "Upload"}, {"label": "Skip"}], "state": "answered",
                      "answered_index": 0, "question_delivery": saved_delivery}},
            {"ts": "2026-11-23T07:06:00Z", "role": "assistant", "msg_type": "quiz", "task_id": "task-healthy",
             "text": "Proceed with macOS sign-off?",
             "quiz": {"quiz_id": "qz-healthy", "question": "Proceed with macOS sign-off?",
                      "options": [{"label": "Proceed"}, {"label": "Hold"}], "state": "answered",
                      "answered_index": 0, "question_delivery": saved_delivery}},
            {"ts": "2026-11-23T07:07:00Z", "role": "assistant", "msg_type": "quiz", "task_id": "task-exp",
             "text": "Keep the Windows verification draft?",
             "quiz": {"quiz_id": "qz-exp", "question": "Keep the Windows verification draft?",
                      "options": [{"label": "Keep"}, {"label": "Discard"}], "state": "answered",
                      "answered_index": 0, "question_delivery": saved_delivery}},
        ],
    }
    page, errors = spa(core, theme="dark", ready='.chat-quiz-card[data-quiz-id="qz-exp"]')
    exp_status = page.locator('.chat-quiz-card[data-quiz-id="qz-exp"] .chat-quiz-status-text')
    healthy_status = page.locator('.chat-quiz-card[data-quiz-id="qz-healthy"] .chat-quiz-status-text')
    unknown_status = page.locator('.chat-quiz-card[data-quiz-id="qz-unknown"] .chat-quiz-status-text')
    page.wait_for_function(
        "() => document.querySelector('.chat-quiz-card[data-quiz-id=\"qz-exp\"] .chat-quiz-status-text')?.textContent"
        " === 'Answer saved · task deadline expired'",
        timeout=15_000,
    )
    assert exp_status.inner_text() == "Answer saved · task deadline expired"
    assert healthy_status.inner_text() == "Answer saved · read after Resume"
    assert unknown_status.inner_text() == "Answer saved · the task reads it when it continues"
    shot(page, "expired-quiz-status-dark", page.locator("#chat-messages"))
    assert not errors
