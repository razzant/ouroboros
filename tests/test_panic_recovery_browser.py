"""After Panic, the real Main page offers the stopped works for the owner's choice (docs/DESIGN.md "Project question
mirror"), over synthetic census rows.

The shipped SPA (``web/index.html``, ``app.js``, every stylesheet) is served from ``web/``; every ``/api/*`` read is
answered at the route and the socket is routed, so no Ouroboros server, task, engine or process runs and nothing is
resumed. What is checked is what the owner sees and can do: the names, a subset or all chosen deliberately, one Resume
per chosen row carrying only the stop it was chosen under, each row's answer (an earlier stop's said as such, a refusal
in the host's words), an incomplete list without ``Choose all``, a choice whose stop changed, and the Stop… each work's
card offered. Screenshots (the owner's 1100×750 window and a phone, light and dark) go to ``OUROBOROS_UI_EVIDENCE_DIR``
when it is set, before anything is asserted.
"""
from __future__ import annotations

import copy
import json
import os
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from urllib.parse import urlsplit

import pytest

pytestmark = [pytest.mark.ui_browser]
WEB = Path(__file__).resolve().parents[1] / "web"
DESKTOP = {"width": 1100, "height": 750}
PHONE = {"width": 390, "height": 844}
MESSAGE = "#chat-messages .panic-recovery"
# Main's history, saved and whole but empty: its one page and its coverage (CLIENT_CONTRACT §5), so the feed says only
# that it begins here.
EMPTY_HISTORY = {"messages": [], "progress": [], "has_more": False, "page_cursor": "fixture-page",
                 "coverage": {"v": 1, "view": "main", "upper": {"chat": 0},
                              "spans": {"chat": {"chain": "empty", "from": 0, "to": 0, "gaps": []}, "progress": None}}}

# Synthetic works only: invented names, rooms and stops.
PROJECTS = [
    {"id": "proj-notes", "name": "Research notes", "chat_id": 21, "lifecycle": "active",
     "created_at": "2026-11-01T09:00:00Z", "last_message_at": "2026-11-20T09:00:00Z", "visible_revision": 0},
    {"id": "proj-ops", "name": "Release checks", "chat_id": 22, "lifecycle": "active",
     "created_at": "2026-11-02T09:00:00Z", "last_message_at": "2026-11-21T09:00:00Z", "visible_revision": 0},
]
WORKS = [
    {"task_id": "task-digest", "task_name": "Summarise this week's reading list", "panic_hold_id": "hold-digest-1",
     "project_id": "proj-notes", "chat_id": 21},
    {"task_id": "task-outline", "task_name": "Compare three draft outlines", "panic_hold_id": "hold-outline-1", "chat_id": 1},
    {"task_id": "task-links", "task_name": "Check the release notes for broken links", "panic_hold_id": "hold-links-1",
     "project_id": "proj-ops", "chat_id": 22, "root_task_id": "task-links"},
    {"task_id": "task-tidy", "task_name": "Tidy the sample data folder", "panic_hold_id": "hold-tidy-1", "chat_id": 1},
    {"task_id": "task-reply", "task_name": "Answer the open thread in a connected place", "panic_hold_id": "hold-reply-1",
     "chat_id": 777},
]

_ROWS = """selector => [...document.querySelectorAll(`${selector} .panic-recovery-row`)].map(row => ({
    id: row.dataset.taskId,
    shown: !row.hidden,
    name: row.querySelector('.panic-recovery-name').textContent,
    meta: row.querySelector('.panic-recovery-meta').textContent,
    outcome: row.querySelector('.panic-recovery-outcome').textContent,
    tone: row.querySelector('.panic-recovery-outcome').dataset.tone || '',
    box: (box => box && !box.hidden ? {checked: box.checked, disabled: box.disabled} : null)(row.querySelector('input')),
    stop: (stop => stop && !stop.hidden ? {disabled: stop.disabled} : null)(
        row.querySelector(':scope .panic-recovery-acts > button:not([data-close-expired-run])')),
}))"""
_PARTS = """selector => {
    const root = document.querySelector(selector);
    // What the owner sees: a part that renders no box (hidden, or an empty status line) is absent.
    const text = node => (node && node.getClientRects().length ? node.textContent : null);
    return {
        title: root.querySelector('strong').textContent,
        lead: text(root.querySelector('.owner-action-reason')),
        gap: text(root.querySelector('[data-panic-gap]')),
        count: text(root.querySelector('.panic-recovery-count')),
        all: text(root.querySelector('[data-panic-all]')),
        none: text(root.querySelector('[data-panic-none]')),
        summary: text(root.querySelector('[data-panic-summary]')),
        resume: text(root.querySelector('[data-panic-resume]')),
        resumeDisabled: root.querySelector('[data-panic-resume]').disabled,
    };
}"""
# Nothing in the message reaches past its bubble, and the page does not scroll sideways.
_CONTAINED = """selector => {
    const bubble = document.querySelector(selector);
    const edge = bubble.getBoundingClientRect().right + 0.5;
    const over = [...bubble.querySelectorAll('*')].filter(node => node.getClientRects().length
        && node.getBoundingClientRect().right > edge).map(node => node.className || node.tagName);
    return {over, page: document.scrollingElement.scrollWidth <= document.scrollingElement.clientWidth};
}"""


class FakeCore:
    """The census and the Resume door, synthetic: what each POST is answered with, recorded."""

    def __init__(self, works=WORKS, complete=True):
        self.works = copy.deepcopy(list(works))
        self.complete = complete
        self.posts: list[tuple[str, dict]] = []   # Resume requests: (task id, body)
        self.stops: list[tuple[str, dict]] = []   # Stop requests: (task id, body)
        self.writes: list[tuple[str, dict]] = []  # any other write the page made (preferences, read receipts)
        self.answers: dict[str, object] = {}
        self.state_reads = 0                      # /api/state bodies served
        self.since = "run-1"                      # the host run (`boot.since`): a restart is a new one
        self.held: dict[str, object] = {}         # Resume requests left unanswered until release()

    def attention(self):
        """The same census's other projection (main_decisions.owner_actions): every held work a readable
        room vouches for is a `budget_paused`, `can_resume` row whose stop is Panic, beside one ordinary
        spending-limit pause the Panic list does not name. A work no room proves is offered nowhere."""
        rows = [{"task_id": work["task_id"], "activity_id": work["task_id"], "task_name": work["task_name"],
                 "project_id": work.get("project_id", ""), "chat_id": work.get("chat_id", 1),
                 "phase": "budget_paused", "pause_cause": "panic", "can_resume": True, "cancelable": True,
                 "attention_reason": "resume", "model_waits": {}}
                for work in self.works if work.get("chat_id") in (1, 21, 22)]
        rows.append({"task_id": "task-budget", "activity_id": "task-budget", "task_name": "Sweep the weekly spend",
                     "project_id": "proj-ops", "chat_id": 22, "phase": "budget_paused", "pause_cause": "budget",
                     "can_resume": True, "cancelable": True, "attention_reason": "budget", "model_waits": {}})
        return rows

    def state(self):
        self.state_reads += 1
        return {"sha": "fixture", "supervisor_ready": True, "boot": {"tasks_ready": True, "since": self.since},
                "projects": PROJECTS, "project_chat_ids": [21, 22],
                "active_chat_activities": [], "active_chat_activities_complete": True,
                "current_questions": [], "current_questions_complete": True,
                "main_attention": self.attention(), "main_attention_complete": True,
                "works": [], "rooms": {}, "works_complete": True, "works_gaps": [],
                "panic_recovery": copy.deepcopy(self.works), "panic_recovery_complete": self.complete}

    def resume(self, route, task_id):
        body = route.request.post_data_json
        self.posts.append((task_id, body))
        answer = self.answers.get(task_id, "ok")
        if answer == "hold":
            self.held[task_id] = route  # answered later, by the test (release)
            return
        self.answer(route, task_id, answer)

    def release(self, task_id, answer="ok"):
        self.answer(self.held.pop(task_id), task_id, answer)

    def cancel(self, route, task_id):
        """The work's Stop: recorded, accepted, and the work leaves the census for good."""
        self.stops.append((task_id, route.request.post_data_json if route.request.post_data else {}))
        self.works = [work for work in self.works if work["task_id"] != task_id]
        route.fulfill(content_type="application/json", body=json.dumps({"ok": True, "task_id": task_id}))

    def answer(self, route, task_id, answer):
        if answer == "abort":
            route.abort()
            return
        if answer in ("ok", "ok, still listed"):
            # The door used the stop: the census no longer holds that work (unless the test says it was stopped again).
            if answer == "ok":
                self.works = [work for work in self.works if work["task_id"] != task_id]
            route.fulfill(content_type="application/json", body=json.dumps({"ok": True, "task_id": task_id}))
            return
        status, payload = answer
        route.fulfill(status=status, content_type="application/json", body=json.dumps({"task_id": task_id, **payload}))


@pytest.fixture
def spa():
    playwright = pytest.importorskip("playwright.sync_api")

    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            # The core serves the SPA's assets under /static; the page itself at /.
            self.path = self.path.removeprefix("/static")
            super().do_GET()

    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(WEB)))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with playwright.sync_playwright() as pw:
            engines = {}

            def open_page(core, *, engine="chromium", viewport=DESKTOP, theme="dark", touch=False, ready=MESSAGE):
                if engine not in engines:
                    engines[engine] = getattr(pw, engine).launch(
                        headless=True, executable_path=os.environ.get("OUROBOROS_UI_BROWSER_EXECUTABLE") or None)
                page = engines[engine].new_page(viewport=viewport, has_touch=touch)
                errors: list[str] = []
                page.on("pageerror", lambda error: errors.append(str(error)))

                def api(route):
                    request = route.request
                    path = urlsplit(request.url).path
                    if request.method == "POST" and path.startswith("/api/tasks/") and path.endswith("/resume"):
                        core.resume(route, path.split("/")[3])
                        return
                    if request.method == "POST" and path.startswith("/api/tasks/") and path.endswith("/cancel"):
                        core.cancel(route, path.split("/")[3])
                        return
                    if path == "/api/onboarding":
                        route.fulfill(status=204)  # an installed app: no setup over Main
                        return
                    if request.method == "POST":
                        # No other write belongs to this screen: answered, recorded, never acted on.
                        core.writes.append((path, request.post_data_json if request.post_data else {}))
                        route.fulfill(content_type="application/json", body="{}")
                        return
                    body = core.state() if path == "/api/state" else {
                        "/api/health": {"status": "ok", "version": "fixture", "runtime_version": "fixture",
                                        "app_version": "fixture"},
                        "/api/chat/history": EMPTY_HISTORY,
                        "/api/projects": {"projects": PROJECTS},
                    }.get(path, {})
                    route.fulfill(content_type="application/json", body=json.dumps(body))

                page.add_init_script(f"localStorage.setItem('ouroboros.theme', {json.dumps(theme)});")
                page.route_web_socket("**/ws", lambda ws: ws.send(json.dumps({"type": "heartbeat"})))
                page.route("**/api/**", api)
                page.goto(f"http://127.0.0.1:{server.server_port}/", wait_until="domcontentloaded")
                page.locator(ready).wait_for(state="visible", timeout=30_000)
                return page, errors

            yield open_page
            for browser in engines.values():
                browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def capture(page, name, *, crop=True):
    root = os.environ.get("OUROBOROS_UI_EVIDENCE_DIR")
    if not root:
        return
    target = Path(root)
    target.mkdir(parents=True, exist_ok=True)
    if page.locator(MESSAGE).count():
        page.locator(MESSAGE).scroll_into_view_if_needed()
    page.screenshot(path=str(target / f"{name}.png"), animations="disabled")
    if crop:
        page.locator(MESSAGE).screenshot(path=str(target / f"{name}-message.png"), animations="disabled")


def rows(page):
    return {row["id"]: row for row in page.evaluate(_ROWS, MESSAGE)}


def parts(page):
    return page.evaluate(_PARTS, MESSAGE)


def wait_settled(page):
    """The press's answers are in and its reconciling read has repainted the message."""
    page.wait_for_function("selector => !document.querySelector(`${selector} [data-panic-resume]`)"
                           ".hasAttribute('aria-busy')", arg=MESSAGE, timeout=30_000)


def test_expired_panic_row_crosses_to_close_without_joining_the_resume_selection(spa):
    core = FakeCore(works=WORKS[:3])
    target = "task-digest"
    page, errors = spa(core, viewport=DESKTOP, theme="dark")
    expired = page.locator(f'{MESSAGE} [data-task-id="{target}"]')
    expired.locator('input').check()
    capture(page, "expired-park-before", crop=False)
    token = "a" * 64
    core.works[0]["close_expired"] = {"action": "close", "target": target, "scope": "cascade", "close_expired": token}
    page.wait_for_function("selector => !document.querySelector(selector).getClientRects().length",
                           arg=f'{MESSAGE} [data-task-id="{target}"] input', timeout=15_000)
    close = expired.get_by_role("button", name="Close", exact=True)
    assert close.is_visible() and "Deadline expired" in expired.inner_text()
    # The card still grants its own Stop: a deadline takes the Resume away, never the owner's right to stop.
    assert expired.get_by_role("button", name="Stop…", exact=True).is_visible()
    assert not core.posts and not core.stops
    assert parts(page)["all"] == "Choose all 2"
    capture(page, "expired-park-after", crop=True)
    close.click()
    expired.wait_for(state="detached", timeout=15_000)
    assert core.stops == [(target, {"stop_policy": "close_expired", "close_expired": token})]
    # When every remaining Panic row is expired too, the Resume button and its empty actions footer both hide.
    for index, work in enumerate(core.works):
        work["close_expired"] = {
            "action": "close", "target": work["task_id"], "scope": "cascade",
            "close_expired": ("b" if index == 0 else "c") * 64,
        }
    page.evaluate("() => window.dispatchEvent(new Event('focus'))")
    page.locator(f"{MESSAGE} [data-panic-resume]").wait_for(state="hidden", timeout=15_000)
    assert page.locator(f"{MESSAGE} .system-message-actions").is_hidden()
    capture(page, "expired-park-all-expired", crop=True)
    assert not core.posts and not errors


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_the_owner_chooses_a_subset_and_reads_what_came_back(spa, theme):
    core = FakeCore()
    core.answers["task-links"] = (409, {"error": "update_in_progress",
                                        "message": "An update is in progress; press Resume after the restart."})
    page, errors = spa(core, theme=theme)
    capture(page, f"panic-desktop-{theme}-1-opened")
    seen = parts(page)
    assert seen["title"] == "Stopped by Panic"
    assert seen["lead"] == "Choose which to resume. The others stay stopped."
    assert seen["gap"] is None and seen["all"] == "Choose all 5" and seen["count"] == "0 of 5 chosen"
    assert seen["resume"] == "Resume selected" and seen["resumeDisabled"]
    listed = rows(page)
    assert [row["name"] for row in listed.values()] == [work["task_name"] for work in WORKS]
    assert listed["task-digest"]["meta"] == "Research notes" and listed["task-outline"]["meta"] == "Main"
    assert listed["task-reply"]["meta"] == ""
    assert all(row["box"] == {"checked": False, "disabled": False} for row in listed.values()), "nothing chosen for the owner"
    assert core.posts == [], "opening sends nothing"
    assert not [path for path, _ in core.writes if "/tasks/" in path], "no other task write either"

    # The whole row is the box's label; Space on a focused box chooses it too.
    page.locator(f"{MESSAGE} .panic-recovery-row[data-task-id='task-digest'] .panic-recovery-name").click()
    page.locator(f"{MESSAGE} .panic-recovery-row[data-task-id='task-links'] input").focus()
    page.keyboard.press("Space")
    assert parts(page)["resume"] == "Resume 2 selected"
    capture(page, f"panic-desktop-{theme}-2-chosen")
    page.locator(f"{MESSAGE} [data-panic-resume]").click()
    wait_settled(page)
    capture(page, f"panic-desktop-{theme}-3-answered")

    assert core.posts == [("task-digest", {"expected_panic_hold_id": "hold-digest-1"}),
                          ("task-links", {"expected_panic_hold_id": "hold-links-1"})]
    after = rows(page)
    assert (after["task-digest"]["outcome"], after["task-digest"]["tone"]) == ("Resuming: the task returns to the queue.", "ok")
    assert after["task-digest"]["box"] is None, "what came back keeps no box"
    assert (after["task-links"]["outcome"], after["task-links"]["tone"]) == (
        "An update is in progress; press Resume after the restart.", "error")
    assert after["task-links"]["box"] == {"checked": False, "disabled": False}, "a refused work stays stopped and choosable"
    assert all(after[key]["box"] == {"checked": False, "disabled": False}
               for key in ("task-outline", "task-tidy", "task-reply")), "what was not chosen stays stopped"
    assert parts(page)["summary"] == "1 accepted · 4 still stopped"
    assert page.evaluate(_CONTAINED, MESSAGE) == {"over": [], "page": True}
    assert errors == []


def test_one_card_per_panic_work_beside_the_actions_it_does_not_name(spa):
    """The realistic mixed body: the same census also offers every held work's Resume as an individual
    `main_attention` row (main_decisions.owner_actions). Each such work keeps one card — its row in the
    choice's message — while the spending-limit pause the message does not name keeps its own card and
    its own Resume, and a choice still resumes only what it named, each with the stop it was chosen under."""
    core = FakeCore()
    page, errors = spa(core)
    capture(page, "panic-desktop-dark-0-mixed")
    # No second feed card for a work the message already offers, from any room.
    for task in ("task-digest", "task-outline", "task-links", "task-tidy"):
        assert page.locator(f".chat-bubble.owner-action[data-owner-action='{task}']").count() == 0, task
    # The action the message does not name keeps its card, its reason and its working Resume.
    budget = page.locator(".chat-bubble.owner-action[data-owner-action='task-budget']")
    budget.wait_for(state="visible")
    assert "spending limit" in budget.locator(".owner-action-reason").inner_text().lower()
    resume = budget.locator("[data-resume-run]")
    assert resume.count() == 1 and not resume.is_disabled()
    # The message still names every room's work once; a choice resumes only what it named.
    assert [row["id"] for row in rows(page).values()] == [work["task_id"] for work in WORKS]
    # Each work keeps the Stop… its card offered, in its row; one no room proves had none and gets none here.
    listed = rows(page)
    assert all(listed[task]["stop"] == {"disabled": False} for task in ("task-digest", "task-outline", "task-links", "task-tidy"))
    assert listed["task-reply"]["stop"] is None
    page.locator(f"{MESSAGE} .panic-recovery-row[data-task-id='task-outline'] .panic-recovery-name").click()
    page.locator(f"{MESSAGE} [data-panic-resume]").click()
    wait_settled(page)
    assert core.posts == [("task-outline", {"expected_panic_hold_id": "hold-outline-1"})]
    # Its Stop… opens that work's stop alone (its box is its Resume) and stops only that work.
    page.locator(f"{MESSAGE} .panic-recovery-row[data-task-id='task-tidy'] .panic-recovery-acts > button").click()
    menu = page.locator(".task-control-menu [data-task-control]")
    assert menu.evaluate_all("items => items.map(item => item.dataset.taskControl)") == ["stop_now"]
    menu.first.click()
    page.wait_for_function("selector => !document.querySelector(`${selector} .panic-recovery-row[data-task-id='task-tidy']`)",
                           arg=MESSAGE, timeout=15_000)
    assert [task for task, _ in core.stops] == ["task-tidy"]
    assert core.posts == [("task-outline", {"expected_panic_hold_id": "hold-outline-1"})], "a Stop resumes nothing"
    assert errors == []


def test_choose_all_on_a_complete_list_and_each_answer_in_its_own_words(spa):
    core = FakeCore()
    core.answers.update({
        "task-outline": (409, {"error": "panic_selection_changed"}),
        "task-tidy": "abort",
        "task-reply": (404, {"error": "task_not_pending"}),
    })
    core.answers["task-digest"] = "hold"
    page, errors = spa(core)
    page.locator(f"{MESSAGE} [data-panic-all]").click()
    assert parts(page)["count"] == "5 of 5 chosen"
    assert parts(page)["resume"] == "Resume 5 selected"
    page.locator(f"{MESSAGE} [data-panic-resume]").click()
    # While the first answer is out: nothing claims success, asks to choose, or says nothing waits.
    page.wait_for_function("selector => document.querySelector(`${selector} [data-panic-resume]`)"
                           ".getAttribute('aria-busy') === 'true'", arg=MESSAGE, timeout=15_000)
    page.wait_for_timeout(300)
    capture(page, "panic-desktop-dark-4a-pending")
    pending = parts(page)
    assert (pending["resume"], pending["resumeDisabled"], pending["lead"], pending["summary"]) == ("Sending…", True, None, None)
    assert all((row["outcome"], row["tone"]) == ("Sending…", "") for row in rows(page).values())
    assert [task for task, _ in core.posts] == ["task-digest"], "one at a time, the first still unanswered"
    core.release("task-digest")
    wait_settled(page)
    capture(page, "panic-desktop-dark-4-every-answer")
    assert [task for task, _ in core.posts] == [work["task_id"] for work in WORKS], "each chosen work once, in order"
    assert all(set(body) == {"expected_panic_hold_id"} for _, body in core.posts)
    said = {key: (row["outcome"], row["tone"]) for key, row in rows(page).items()}
    assert said == {
        "task-digest": ("Resuming: the task returns to the queue.", "ok"),
        "task-outline": ("Not resumed: its stop changed after you chose it.", "warn"),
        "task-links": ("Resuming: the task returns to the queue.", "ok"),
        "task-tidy": ("No answer came back; it may already be back in the queue.", "neutral"),
        "task-reply": ("Not resumed: it no longer waits for Resume.", "neutral"),
    }
    assert parts(page)["summary"] == "2 accepted · 2 still stopped · 1 without an answer"
    page.wait_for_timeout(3500)  # a later poll of the census
    assert len(core.posts) == len(WORKS), "nothing is sent again by itself"
    assert errors == []


def test_an_incomplete_list_never_offers_all_and_a_changed_stop_is_chosen_again(spa):
    core = FakeCore(works=WORKS[:3], complete=False)
    page, errors = spa(core)
    capture(page, "panic-desktop-dark-5-incomplete")
    seen = parts(page)
    assert seen["all"] is None, "no Choose all while the list may be partial"
    assert seen["gap"] == "Some saved work could not be read, so this list may be incomplete."
    page.locator(f"{MESSAGE} .panic-recovery-row[data-task-id='task-outline'] .panic-recovery-name").click()
    assert parts(page)["resume"] == "Resume 1 selected"
    # Before the press, the next census names a new stop for that work: the choice was for the old one.
    core.works[1]["panic_hold_id"] = "hold-outline-2"
    page.wait_for_function("selector => document.querySelector(`${selector} [data-panic-resume]`).disabled",
                           arg=MESSAGE, timeout=15_000)
    assert rows(page)["task-outline"]["box"] == {"checked": False, "disabled": False}
    assert parts(page)["resume"] == "Resume selected"
    capture(page, "panic-desktop-dark-6-stale-choice")
    assert core.posts == []

    # A Resume of its old stop is out when a fresh read names the work under a newer stop; the old stop's
    # acceptance then arrives. It stays in view as the earlier stop's; the newer stop is not chosen, and can be.
    core.answers["task-digest"] = "hold"
    page.locator(f"{MESSAGE} .panic-recovery-row[data-task-id='task-digest'] .panic-recovery-name").click()
    page.locator(f"{MESSAGE} [data-panic-resume]").click()
    page.wait_for_function("selector => document.querySelector(`${selector} [data-panic-resume]`)"
                           ".getAttribute('aria-busy') === 'true'", arg=MESSAGE, timeout=15_000)
    core.works[0]["panic_hold_id"] = "hold-digest-2"
    page.wait_for_function("""selector => document.querySelector(
        `${selector} .panic-recovery-row[data-task-id='task-digest'] .panic-recovery-outcome`).textContent
        === 'Earlier stop: Sending…'""", arg=MESSAGE, timeout=15_000)
    core.release("task-digest", "ok, still listed")
    wait_settled(page)
    capture(page, "panic-desktop-dark-7-newer-stop")
    assert core.posts == [("task-digest", {"expected_panic_hold_id": "hold-digest-1"})]
    digest = rows(page)["task-digest"]
    assert (digest["outcome"], digest["tone"], digest["box"]) == (
        "Earlier stop: Resuming: the task returns to the queue.", "", {"checked": False, "disabled": False})
    assert parts(page)["summary"] == "3 still stopped", "no acceptance is counted for the newer stop"
    assert parts(page)["all"] is None and parts(page)["lead"] == "Choose which to resume. The others stay stopped."
    assert parts(page)["gap"] == "Some saved work could not be read, so this list may be incomplete."
    assert errors == []


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_the_choice_fits_a_phone(spa, theme):
    core = FakeCore()
    page, errors = spa(core, viewport=PHONE, theme=theme, touch=True)
    page.locator(f"{MESSAGE} .panic-recovery-row[data-task-id='task-links'] .panic-recovery-name").tap()
    assert parts(page)["resume"] == "Resume 1 selected"
    capture(page, f"panic-phone-{theme}")
    assert page.evaluate(_CONTAINED, MESSAGE) == {"over": [], "page": True}
    heights = page.evaluate(f"""() => [...document.querySelectorAll('{MESSAGE} .panic-recovery-pick')]
        .map(node => node.getBoundingClientRect().height)""")
    assert min(heights) >= 44, "each row is a finger's target"
    assert core.posts == []
    assert errors == []


@pytest.mark.parametrize(("viewport", "theme"), [(DESKTOP, "dark"), (PHONE, "light")], ids=["desktop-dark", "phone-light"])
def test_the_final_state_after_a_press_on_an_incomplete_list(spa, viewport, theme):
    """One press over an incomplete list: a work that came back, one whose stop changed while its Resume was out
    (its answer reads as the earlier stop's, its newer stop unchosen beside its Stop…), one the host refused in its own
    words, and the works not chosen. The list still says it may be incomplete after the last answer."""
    core = FakeCore(complete=False)
    core.answers.update({
        "task-digest": "hold",
        # Synthetic wording: the host's own sentence for this code is shown as it comes.
        "task-links": (409, {"error": "installation_panic_pending",
                             "message": "Resume refused: the installation's Panic stop is still in force."}),
    })
    phone = viewport is PHONE
    page, errors = spa(core, viewport=viewport, theme=theme, touch=phone)
    for task in ("task-digest", "task-outline", "task-links"):
        name = page.locator(f"{MESSAGE} .panic-recovery-row[data-task-id='{task}'] .panic-recovery-name")
        name.tap() if phone else name.click()
    assert parts(page)["resume"] == "Resume 3 selected"
    page.locator(f"{MESSAGE} [data-panic-resume]").click()
    page.wait_for_function("selector => document.querySelector(`${selector} [data-panic-resume]`)"
                           ".getAttribute('aria-busy') === 'true'", arg=MESSAGE, timeout=15_000)
    core.works[0]["panic_hold_id"] = "hold-digest-2"
    page.wait_for_function("""selector => document.querySelector(
        `${selector} .panic-recovery-row[data-task-id='task-digest'] .panic-recovery-outcome`).textContent
        === 'Earlier stop: Sending…'""", arg=MESSAGE, timeout=15_000)
    core.release("task-digest", "ok, still listed")
    wait_settled(page)
    if not phone:  # the owner's window holds the whole result; the phone's frame is the empty first list below
        capture(page, f"panic-final-desktop-{theme}", crop=False)

    assert [task for task, _ in core.posts] == ["task-digest", "task-outline", "task-links"]
    after = rows(page)
    said = {task: (row["outcome"], row["tone"], row["box"], row["stop"]) for task, row in after.items()}
    free = {"checked": False, "disabled": False}
    assert said == {
        "task-digest": ("Earlier stop: Resuming: the task returns to the queue.", "", free, {"disabled": False}),
        "task-outline": ("Resuming: the task returns to the queue.", "ok", None, None),
        "task-links": ("Resume refused: the installation's Panic stop is still in force.", "error", free, {"disabled": False}),
        "task-tidy": ("", "", free, {"disabled": False}),
        "task-reply": ("", "", free, None),
    }
    seen = parts(page)
    assert seen["gap"] == "Some saved work could not be read, so this list may be incomplete."
    assert seen["all"] is None and seen["summary"] == "1 accepted · 4 still stopped"
    assert seen["count"] == "0 of 4 chosen"
    assert page.evaluate(_CONTAINED, MESSAGE) == {"over": [], "page": True}
    if phone:
        heights = page.evaluate(f"""() => [...document.querySelectorAll('{MESSAGE} .panic-recovery-pick')]
            .map(node => node.getBoundingClientRect().height)""")
        assert min(heights) >= 44, "each row is a finger's target"
    assert errors == []


def test_an_empty_first_list_that_is_not_whole_is_one_notice(spa):
    """A ready host's first answer names no work and says its list is not whole: Main holds no System message for it;
    the page's one notice at the top right says so once, claiming no Panic, stop or work, with a Retry that only reads
    again and, when that read still is not whole, says so again; the spending-limit card stays its own."""
    core = FakeCore(works=[], complete=False)
    page, errors = spa(core, viewport=PHONE, theme="light", touch=True, ready="#toast-stack .toast")
    seen = core.state_reads
    deadline = 30_000
    while core.state_reads < seen + 2 and deadline > 0:  # at least two more state reads: the notice is not repeated
        page.wait_for_timeout(250)
        deadline -= 250
    assert core.state_reads >= seen + 2, core.state_reads
    capture(page, "panic-final-phone-light-empty-incomplete", crop=False)
    assert page.locator(MESSAGE).count() == 0
    toast = page.locator("#toast-stack .toast")
    assert toast.count() == 1
    assert toast.locator(".toast-words").inner_text() == "Saved work could not be checked for anything waiting to resume."
    box, view = toast.bounding_box(), page.viewport_size
    assert box["x"] >= 0 and box["x"] + box["width"] <= view["width"] and box["y"] >= 0, (box, view)
    assert page.locator(".chat-bubble.owner-action[data-owner-action='task-budget']").count() == 1
    reads = core.state_reads
    toast.get_by_role("button", name="Retry").click()
    page.wait_for_function("() => document.querySelectorAll('#toast-stack .toast:not(.is-hiding)').length === 1")
    assert core.state_reads > reads, "Retry read the state again"
    assert page.locator(MESSAGE).count() == 0, "a list still not whole is answered by the notice, never a card"
    # A successful Retry: its read is whole, and the notice leaves with nothing said in its place.
    core.complete = True
    reads = core.state_reads
    page.locator("#toast-stack .toast:not(.is-hiding)").get_by_role("button", name="Retry").click()
    page.wait_for_function("() => document.querySelectorAll('#toast-stack .toast').length === 0")
    assert core.state_reads > reads, "Retry read the state again"
    page.wait_for_timeout(400)  # past the Retry's own answer to that read
    assert page.locator("#toast-stack .toast").count() == 0 and page.locator(MESSAGE).count() == 0, (
        "a whole empty list: nothing waits, nothing to say")
    # The host's word again is a new episode. It stands across a reload of the tab; the person's × closes it for this
    # host run, a reload included; a host that restarted tells it again.
    core.complete = False
    page.locator("#toast-stack .toast").wait_for()
    page.reload()
    page.locator("#toast-stack .toast").wait_for()
    page.locator("#toast-stack .toast:not(.is-hiding)").get_by_role("button", name="Dismiss notice").click()
    page.wait_for_function("() => document.querySelectorAll('#toast-stack .toast').length === 0")
    page.reload()
    page.wait_for_selector("#chat-input", state="attached")
    seen = core.state_reads
    deadline = 30_000
    while core.state_reads < seen + 2 and deadline > 0:
        page.wait_for_timeout(250)
        deadline -= 250
    assert core.state_reads >= seen + 2, ("two more reads, still not whole", core.state_reads)
    assert page.locator("#toast-stack .toast").count() == 0, "closed by the person in this run: none after a reload"
    core.since = "run-2"
    page.locator("#toast-stack .toast").wait_for()
    assert core.posts == [] and core.stops == [], "the notice and its × run nothing"
    assert errors == []
