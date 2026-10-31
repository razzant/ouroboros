"""The room's work status (DESIGN "Room work status") over controlled transport.

The real SPA (index.html, app.js, chat.js, room_work.js, chat_decision.js, task_control_menu.js and
their CSS) runs in Chromium and WebKit against the ``subscription_ui`` static server. The census
(``/api/state``), history, task detail and the WebSocket are answered here, so states a live backend
reaches only by chance are exact: unknown, a partial census, an access wait, Pausing, an outcome
beside Finalizing…, a Project hold, a new attempt of the same id, a question outside the loaded
history and a read that lands after a gesture or after the room was left. Task-control POSTs are
answered with acknowledgements and recorded; no decision may ever be posted from the list.

Screenshots (1100x722 and 390x844, light and dark, closed and open over scrolling content) and the
facts read in the page go to ``OUROBOROS_UI_EVIDENCE_DIR`` when it is set.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
import os
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from ouroboros.consciousness_view import describe_consciousness_state
from tests import test_subscription_setup_browser as setup_browser
from tests._work_projection_shared import work_fields

subscription_ui = setup_browser.subscription_ui
pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]

# The facts one room's control and list expose to the owner, read in one frame.
ROOM_FACTS = """(scope) => {
    const root = scope === 'main' ? document.querySelector('#page-chat') : document.querySelector('#project-panel');
    const trigger = root?.querySelector('.rw-trigger');
    const surface = trigger ? document.getElementById(trigger.getAttribute('aria-controls')) : null;  // the room's work Shade, never the chip's
    const text = (el) => (el?.textContent || '').replace(/\\s+/g, ' ').trim();
    const shown = (el) => Boolean(el) && !el.hidden, markOf = (el) => (shown(el) ? `${el.dataset.shape} ${el.dataset.hue}` : '');
    const rows = [...(surface?.querySelectorAll('.rw-row') || [])].map((row) => {
        const main = row.querySelector('.rw-main');
        const detail = row.querySelector('.rw-detail');
        const secondary = row.querySelector('.rw-secondary');
        return {
            id: row.dataset.id, kind: row.dataset.kind, tone: row.dataset.tone, slot: row.dataset.slot,
            settled: row.dataset.settled === '1', open: row.dataset.open === '1',
            reading: row.dataset.reading || '', name: text(row.querySelector('.rw-name')),
            word: text(row.querySelector('.rw-word')), secondary: shown(secondary) ? text(secondary) : '',
            target: main?.dataset.target || '', state: text(row.querySelector('.rw-state')),
            label: main?.getAttribute('aria-label') || '', expanded: main?.getAttribute('aria-expanded'),
            question: Boolean(row.querySelector('.rw-pill-question')), resume: Boolean(row.querySelector('.rw-pill-resume')),
            more: Boolean(row.querySelector('.rw-more')), moreId: row.querySelector('.rw-more')?.dataset.id || '',
            moreLabel: row.querySelector('.rw-more')?.getAttribute('aria-label') || '',
            mark: markOf(row.querySelector('.status-mark')), detail: shown(detail) ? text(detail) : '',
        };
    });
    const back = root?.querySelector('.rw-back');
    const note = surface?.querySelector('.rw-note');
    const active = document.activeElement;
    const menu = document.querySelector('body > .task-control-menu');
    return {
        exists: Boolean(trigger), tone: trigger?.dataset.tone || '', disabled: Boolean(trigger?.disabled),
        hidden: Boolean(trigger?.hidden), expanded: trigger?.getAttribute('aria-expanded') || '',
        badge: text(trigger?.querySelector('.status-badge')), mark: markOf(trigger?.querySelector('.status-mark')),
        markColor: trigger ? getComputedStyle(trigger.querySelector('.status-mark')).color : '', ownerColor: (() => {
            const probe = document.body.appendChild(document.createElement('i')); probe.style.color = 'var(--mark-owner)';
            const color = getComputedStyle(probe).color; probe.remove(); return color; })(),
        waitNote: shown(trigger?.querySelector('.rw-sr')), surfaceHidden: surface ? surface.hidden : null,
        rows, groups: [...(surface?.querySelectorAll('.rw-group') || [])].map(text),
        note: shown(note) ? text(note) : '', back: shown(back) ? text(back) : '',
        menu: menu ? [...menu.querySelectorAll('.task-control-item')].map((item) => item.dataset.taskControl) : null,
        active: active ? { tag: active.tagName, cls: String(active.className || ''), id: active.id || '',
            inSurface: Boolean(surface?.contains(active)), isTrigger: active === trigger,
            row: active.closest?.('.rw-row')?.dataset.id || '' } : null,
    };
}"""

_SETTLE_FRAMES = "() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))"

# A row's ⋯: its rectangle at the next click, and a reader of its rectangle now.
_ANCHOR_AT_CLICK = """(id) => {
    const more = document.querySelector(`#page-chat .rw-row[data-id="${id}"] .rw-more`);
    const rect = () => { const r = more.getBoundingClientRect(); return [Math.round(r.top), Math.round(r.bottom)]; };
    window.__rwAnchor = { now: rect };
    document.addEventListener('click', () => { window.__rwAnchor.click = rect(); }, { capture: true, once: true });
}"""

# Rows' rectangles at a press and at the click its release makes.
_PRESS_RECORDER = """(ids) => {
    const rects = () => ids.map((id) => { const r = document.querySelector(`#page-chat .rw-row[data-id="${id}"] .rw-main`)
        .getBoundingClientRect(); return [Math.round(r.top), Math.round(r.bottom)]; });
    window.__rwPress = {};
    for (const type of ['pointerdown', 'click'])
        document.addEventListener(type, () => { window.__rwPress[type] = rects(); }, { capture: true, once: true });
}"""

# Every frame for a while: one row's word, the reason beside it and settledness, its ⋯ and any open shared menu.
_FRAME_RECORDER = """([selector, ms]) => {
    window.__rwFrames = [];
    const start = performance.now();
    const tick = () => {
        const row = document.querySelector(selector);
        const menu = document.querySelector('body > .task-control-menu');
        window.__rwFrames.push({ t: Math.round(performance.now() - start),
            word: row?.querySelector('.rw-word')?.textContent || '',
            secondary: row?.querySelector('.rw-secondary')?.textContent || '', settled: row?.dataset.settled || '',
            more: Boolean(row?.querySelector('.rw-more')),
            menu: menu ? [...menu.querySelectorAll('.task-control-item')].map((item) => item.dataset.taskControl) : null });
        if (performance.now() - start < ms) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
}"""

_CLIPBOARD = """(() => {
    window.__copied = [];
    const clipboard = { writeText: async (text) => { window.__copied.push(String(text)); } };
    try { Object.defineProperty(navigator, 'clipboard', { configurable: true, get: () => clipboard }); } catch {}
})();"""

# Long names in the languages the product is used in; a row shows two lines, focus shows all.
LONG_NAME = ("Сверить итоговые ссылки отчёта с первоисточниками и собрать таблицу расхождений для приложения Б "
             "— einschließlich der Übersetzungsvarianten aller Zitate, Quellenangaben und Anmerkungen")
LONGER_NAME = f"{LONG_NAME} — {LONG_NAME}"  # clamps at two lines even in a Shade as wide as the bar
ASK_TEXT = "Please check which evidence the weekly summary should cite, and ask me before you choose."
PROJECT = {"id": "rw-proj", "name": "Archive evidence review", "chat_id": 42, "lifecycle": "active", "visible_revision": 0}
OTHER_PROJECT = {"id": "rw-proj-two", "name": "Release notes", "chat_id": 43, "lifecycle": "active",
                 "visible_revision": 0}


def settle_frames(page):
    page.evaluate(_SETTLE_FRAMES)


def _ts(minute, second=0):
    return f"2026-10-14T09:{minute:02d}:{second:02d}Z"


def owner(text, cmid, minute):
    return {"role": "user", "text": text, "content": text, "ts": _ts(minute), "client_message_id": cmid}


def assistant(text, minute, second=0):
    return {"role": "assistant", "text": text, "content": text, "ts": _ts(minute, second)}


def progress(task_id, text, minute, **extra):
    return {"role": "assistant", "direction": "out", "is_progress": True, "task_id": task_id, "text": text,
            "content": text, "ts": _ts(minute), "cancelable": True, **extra}


def reading(count, minute=0, room="Main"):
    rows = []
    for index in range(count):
        rows.append(owner(f"{room} question {index}: what changed in section {index}?", f"{room}-read-{index}", minute))
        rows.append(assistant(f"{room} answer {index}. " + "The reading notes continue with ordinary context. " * 5,
                              minute, 30))
    return rows


def activity(task_id, phase="working", *, kind="managed_task", chat_id=1, attempt=1, **extra):
    return {"activity_id": task_id, "kind": kind, "phase": phase, "chat_id": chat_id, "task_attempt": attempt,
            "started_at": 1792000000, **extra}


def waited(quiz_id):
    return {"owner_wait": {"quiz_id": quiz_id, "owner_wait_state": "waiting", "quiz_state": "open"}}


def quiz_detail(task_id, quiz_id, question, **stamp):
    room = stamp.pop("question_room", {"state": "known", "chat_id": stamp["chat_id"],
                     "project_id": stamp.get("project_id", "")} if "chat_id" in stamp else {"state": "unknown"})
    return {"task_id": task_id, "status": "running", **stamp,
            "question_rooms": {quiz_id: room},
            "owner_quiz": {quiz_id: {"quiz_id": quiz_id, "state": "open", "question": question,
                                     "options": ["The primary notes", "The replication"], "wait_for_answer": True,
                                     "asked_at": _ts(40)}},
            "owner_wait": {"quiz_id": quiz_id, "state": "waiting"}}


class Transport:
    """Census, history, task detail and the WebSocket of one page, answered from these fields."""

    def __init__(self, page, *, activities=(), history=None, details=None, projects=()):
        self.page = page
        self.activities = list(activities)
        self.complete = True
        self.history = history or {1: []}
        self.details = dict(details or {})
        self.projects = list(projects)
        # What the projection reads besides the census rows: queue facts, stored results and owner fences.
        self.raw: dict = {}
        self.results: dict = {}
        self.fences: dict = {}
        self.hold = set()
        self.held: list = []
        self.sockets: list = []
        self.state_reads = 0
        self.requests: list = []
        self.posts: list = []
        self.around_reads: list = []  # presses on rows whose card is not loaded (``around=task:<id>``)
        self.console: list = []
        page.on("request", lambda request: self.requests.append((request.method, urlparse(request.url).path))
                if "/api/" in request.url else None)
        page.on("console", lambda msg: self.console.append(f"{msg.type}: {msg.text}")
                if msg.type in {"error", "warning"} else None)
        page.route_web_socket("**/ws", self._connect)
        page.route("**/api/state*", self._state)
        page.route("**/api/projects*", lambda route: route.fulfill(json={"projects": self.projects}))
        page.route("**/api/chat/history*", self._history)
        page.route("**/api/tasks/**", self._tasks)
        page.route("**/api/decisions*", self._decision)

    # -- transport
    def _connect(self, ws):
        self.sockets.append(ws)
        ws.send(json.dumps({"type": "heartbeat"}))

    def body(self, *, ready=True):
        """The ``/api/state`` body for the current census, with the server's own work projection of the same rows."""
        rows = json.loads(json.dumps(self.activities))
        raw, fences = dict(self.raw), dict(self.fences)
        # A phase alone grants no Resume and proves no execution. Supply the
        # queue observation and current fence that the real census reads.
        for item in rows:
            if item.get("pause_cause") == "owner" and item["phase"] in {"budget_pausing", "budget_paused"}:
                task_id = item["activity_id"]
                pausing = item["phase"] == "budget_pausing"
                raw[task_id] = {"running": pausing, **raw.get(task_id, {})}
                fences.setdefault(item.get("root_task_id") or task_id, {
                    "state": "requested" if pausing else "paused", "parked_members": [] if pausing else [task_id]})
        # Consciousness facts show each bar's chip: one more neighbour no word may move (DESIGN bible 6).
        next_wake = datetime.now(timezone.utc) + timedelta(hours=1)
        return {"sha": "room-work-fixture", "supervisor_ready": ready, "active_chat_activities": rows,
                "bg_consciousness_enabled": True, "bg_consciousness_state": describe_consciousness_state(
                    {"next_wake_at": next_wake.isoformat()}, True),
                "active_chat_activities_complete": self.complete, "projects": self.projects,
                "project_chat_ids": [p["chat_id"] for p in self.projects],
                **work_fields(rows, raw, results=self.results, fences=fences, complete=self.complete)}

    def _state(self, route):
        self.state_reads += 1
        route.fulfill(json=self.body())

    def _history(self, route):
        query = parse_qs(urlparse(route.request.url).query)
        chat_id, around = int(query.get("chat_id", ["1"])[0]), query.get("around", [""])[0]
        if around:  # these rooms' saved history holds no message of the census-only work (gateway: ``absent``)
            self.around_reads.append(around)
            route.fulfill(json={"messages": [], "has_more": False, "next_cursor": None, "page_cursor": None,
                                "window": {"complete": False, "truncated_by": ["page"]},
                                "anchor": {"state": "absent", "task_id": around.partition(":")[2]}})
            return
        route.fulfill(json={"messages": self.history.get(chat_id, []), "progress": []})

    def _tasks(self, route):
        request = route.request
        path = urlparse(request.url).path
        parts = path.split("/")
        task_id, action = parts[3], (parts[4] if len(parts) > 4 else "")
        if request.method == "POST":
            self.posts.append((path, request.post_data_json or {}))
            states = {"pause": "pausing", "resume": "resumed", "hurry": "requested"}
            route.fulfill(json={"ok": True, "task_id": task_id, "state": states.get(action, "pending"),
                                "cancel_state": "pending", "request_id": (request.post_data_json or {}).get("request_id", "")})
            return
        if task_id in self.hold:
            self.held.append((task_id, route))
            return
        self._detail(route, task_id)

    def _detail(self, route, task_id):
        detail = self.details.get(task_id)
        if detail is None:
            route.fulfill(status=404, json={"error": "task not found"})
        else:
            route.fulfill(json=detail)

    def _decision(self, route):
        self.posts.append((urlparse(route.request.url).path, route.request.post_data_json or {}))
        route.fulfill(json={"ok": True, "state": "answered"})

    # -- driving
    def release(self, task_id):
        for held_id, route in [item for item in self.held if item[0] == task_id]:
            self.held.remove((held_id, route))
            self._detail(route, held_id)

    def send(self, frame):
        self.sockets[-1].send(json.dumps(frame))

    def refresh(self):
        """Force one page-wide census read (the app's own `projects_changed` path) and let it paint."""
        before = self.state_reads
        self.send({"type": "projects_changed"})
        deadline = time.monotonic() + 15
        while self.state_reads <= before and time.monotonic() < deadline:
            self.page.wait_for_timeout(40)
        assert self.state_reads > before, "the census read never happened"
        self.page.wait_for_timeout(150)
        settle_frames(self.page)

    def disconnect(self):
        count = len(self.sockets)
        self.sockets[-1].close()
        return count

    def await_reconnect(self, count, timeout=20):
        deadline = time.monotonic() + timeout
        while len(self.sockets) <= count and time.monotonic() < deadline:
            self.page.wait_for_timeout(100)
        assert len(self.sockets) > count, "the page never reconnected"
        self.page.wait_for_timeout(300)

    def set(self, task_id, **fields):
        row = next(row for row in self.activities if row["activity_id"] == task_id)
        row.update(fields)
        for key in [key for key, value in fields.items() if value is None]:
            row.pop(key)

    def drop(self, task_id):
        self.activities = [row for row in self.activities if row["activity_id"] != task_id]

    def decision_posts(self):
        return [post for post in self.posts if "/api/decisions" in post[0]]


@pytest.fixture(params=["chromium", "webkit"])
def room_ui(request, monkeypatch):
    """``subscription_ui`` in each engine; evidence per engine under ``room-work``."""
    monkeypatch.setenv("OUROBOROS_UI_BROWSER_ENGINE", request.param)
    root = os.environ.get("OUROBOROS_UI_EVIDENCE_DIR")
    if root:
        monkeypatch.setenv("OUROBOROS_UI_EVIDENCE_DIR", str(Path(root) / "room-work" / request.param))
    ui = request.getfixturevalue("subscription_ui")
    return {**ui, "engine": request.param}


def evidence(name, value):
    root = os.environ.get("OUROBOROS_UI_EVIDENCE_DIR")
    if root:
        Path(root).mkdir(parents=True, exist_ok=True)
        (Path(root) / f"{name}.json").write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str),
                                                 encoding="utf-8")


def facts(page, scope="main"):
    return page.evaluate(ROOM_FACTS, scope)


def wait_facts(page, scope, predicate, what, timeout=20):
    deadline = time.monotonic() + timeout
    current = None
    while time.monotonic() < deadline:
        current = facts(page, scope)
        if predicate(current):
            return current
        page.wait_for_timeout(100)
    raise AssertionError(f"timed out waiting for {what}: {json.dumps(current, ensure_ascii=False, indent=1)}")


def row(current, task_id):
    return next((item for item in current["rows"] if item["id"] == task_id), None)


def open_app(ui, transport, *, theme="dark", viewport=None):
    page = ui["page"]
    if viewport:
        page.set_viewport_size(viewport)
    origin = urlparse(ui["url"]).netloc
    page.route("**/*", lambda route: route.fallback() if urlparse(route.request.url).netloc == origin else route.abort())
    page.add_init_script(f"localStorage.setItem('ouroboros.theme', {json.dumps(theme)})")
    page.add_init_script(_CLIPBOARD)
    page.goto(ui["url"] + "/")
    page.wait_for_selector("#page-chat .rw-trigger", state="attached")
    deadline = time.monotonic() + 15
    while not transport.sockets and time.monotonic() < deadline:
        page.wait_for_timeout(50)
    assert transport.sockets, "the page never opened its socket"
    transport.refresh()
    return page


def open_project(page, project):
    page.evaluate("p => window.dispatchEvent(new CustomEvent('ouro:open-project', {detail: {project: p}}))", project)
    page.wait_for_selector(f"#panel-pchat-{project['id']}", state="attached")
    page.wait_for_function("() => !document.getElementById('project-panel').hidden")
    page.wait_for_selector("#project-panel .project-panel-bar .rw-trigger", state="attached")
    # Docked the panel comes in from the window's edge once its room has painted: geometry is read where it rests.
    page.wait_for_function("() => { const panel = document.getElementById('project-panel'); return panel.classList.contains('open')"
                           " && !panel.getAnimations().some((a) => a.playState === 'running'); }")
    settle_frames(page)


def trigger(page, scope="main"):
    return page.locator("#page-chat .rw-trigger" if scope == "main" else "#project-panel .project-panel-bar .rw-trigger")


def row_main(page, task_id, scope="main"):
    root = "#page-chat" if scope == "main" else "#project-panel"
    return page.locator(f"{root} .rw-row[data-id='{task_id}'] .rw-main")


def toasts(page):
    return page.locator(".toast").all_inner_texts()


def capture(page, name):
    setup_browser.capture(page, name)


# ------------------------------------------------------------------------------------------------
# Main: each census state in its own words, tone and controls.

def test_main_rows_say_each_census_state_with_its_own_tone_and_controls(room_ui):
    history = {1: [*reading(10), owner(ASK_TEXT, "cm-ask", 30),
                   progress("rw-live", "Comparing the cited pages.", 31, suggested_name=LONGER_NAME),
                   progress("rw-access", "Preparing the quota check.", 32, suggested_name="Check the subscription quota"),
                   progress("rw-pausing", "Collecting appendix evidence.", 33, suggested_name="Collect appendix evidence"),
                   progress("rw-failfin", "Rebuilding the index.", 34, suggested_name="Rebuild the citation index"),
                   *reading(3, 35, "Later")]}
    # The model-wait owner's own row shape (web/tests/model_wait.test.js): a quota wait, not an answer wait.
    wait = {"wait_id": "w-access", "revision": 1, "task_attempt": 1, "role": "main", "reason": "quota",
            "model": "claudexor::source-a=same-model", "source": "source-a", "credential_harness": "codex",
            "credential_profile_id": "personal", "auto_continue": True, "state": "waiting", "worker_slot_held": True}
    transport = Transport(room_ui["page"], history=history, details={
        "rw-unknown": {"task_id": "rw-unknown", "status": "running", "suggested_name": "Reconcile the import ledger"}},
        activities=[
            activity("rw-live"),
            activity("rw-access", model_waits={"w-access": wait}),
            activity("rw-pausing", "budget_pausing", pause_cause="owner"),
            activity("rw-failfin", "finalizing", status="failed",
                     root_phase_checkpoint={"post_task_synthesis": "running"},
                     outcome_axes={"lifecycle": {"status": "failed"}, "execution": {"status": "infra_failed"}}),
            activity("rw-unknown", "unknown"),
            activity("rw-queued", "queued"),
            activity("rw-ask", kind="direct_chat", client_message_id="cm-ask", **waited("q-main")),
        ])
    # The live work's own focus line (update_focus), which the row shows quieter under its state.
    focus_text = "Cited pages · 3 of 5 compared · two left"
    transport.raw["rw-live"] = {"focus": {"text": focus_text, "at": "2026-10-19T09:00:00Z"}}
    page = open_app(room_ui, transport)
    closed = wait_facts(page, "main", lambda f: f["tone"] == "live" and not f["disabled"], "a live control")
    # Moving work with something waiting keeps its live word; the owner's amber dot takes its mark's place, in words too.
    assert closed["mark"] == "dot owner" and closed["waitNote"], closed
    assert closed["markColor"] == closed["ownerColor"], closed  # the one mark is painted in the owner's token
    assert closed["surfaceHidden"] and closed["expanded"] == "false"
    trigger(page).click()
    opened = wait_facts(page, "main", lambda f: len(f["rows"]) == 7 and row(f, "rw-unknown")["name"]
                        == "Reconcile the import ledger", "seven named rows")
    evidence("main-states", opened)
    capture(page, "main-states-open")
    ids = [item["id"] for item in opened["rows"]]
    assert ids[:2] == ["rw-ask", "rw-live"], "the owner's turn first, then moving work"
    assert ids[-1] == "rw-unknown", "unknown work goes last"
    by = {item["id"]: item for item in opened["rows"]}
    # Each row says its work's word and the reason beside it, from the server's projection of these rows.
    expect = {
        "rw-live": ("Working", "", "live"),
        # A model wait no one has to act on continues by itself: Waiting, with its reason, and nothing amber.
        "rw-access": ("Waiting", "Waiting for the model to become available", "held"),
        # A Pause still settling: neutral ink. Its press declared nobody (this transport records no author), so its
        # line names nobody either; the word never carries a who.
        "rw-pausing": ("Pausing…", "", "held"),
        "rw-failfin": ("Failed", "Finalizing…", "failed"),
        "rw-unknown": ("Activity unconfirmed", "", "unknown"),
        "rw-queued": ("Queued", "Waiting for a free worker", "queued"),
        "rw-ask": ("Waiting for your answer", "", "wait"),
    }
    for task_id, (word, secondary, tone) in expect.items():
        assert (by[task_id]["word"], by[task_id]["secondary"], by[task_id]["tone"]) == (word, secondary, tone), by[task_id]
    focus_lines = page.evaluate("""() => Object.fromEntries([...document.querySelectorAll('#page-chat .rw-row')]
        .map(r => [r.dataset.id, (r.querySelector('.rw-focus:not([hidden])')?.textContent || '').trim()]))""")
    assert focus_lines["rw-live"] == focus_text and focus_lines["rw-queued"] == "", focus_lines
    # An access wait is not an answer wait: only the waited question offers View question.
    assert [item["id"] for item in opened["rows"] if item["question"]] == ["rw-ask"]
    assert by["rw-ask"]["name"].startswith("Reply to “Please check which evidence"), by["rw-ask"]
    assert by["rw-ask"]["target"] == "card", "the owner's own message is the row's place"
    # Names: the card's coined name whole in the DOM; census-only work reads its detail; else a neutral name.
    assert by["rw-live"]["name"] == LONGER_NAME and by["rw-live"]["target"] == "card"
    assert by["rw-live"]["label"].startswith(f"Go to task: {LONGER_NAME}, Working")
    # A row whose card is not loaded claims nothing about where its message is until a press asks the room's
    # history (DESIGN "Room work status"); every row's second line is its state (5A).
    assert by["rw-unknown"]["target"] == "history" and by["rw-unknown"]["label"].startswith("Open: "), by["rw-unknown"]
    assert "loaded" not in by["rw-unknown"]["label"] and "loaded" not in by["rw-unknown"]["state"]
    assert by["rw-unknown"]["state"].startswith("Activity unconfirmed")
    assert page.locator("#page-chat .rw-place").count() == 0
    assert by["rw-queued"]["name"].startswith("Task"), by["rw-queued"]
    # One mark per row, its work's shape; an outcome's dot beside its word; the owner's amber dot where a question waits.
    assert [by[key]["mark"] for key in ("rw-live", "rw-pausing", "rw-queued", "rw-access", "rw-unknown", "rw-failfin",
            "rw-ask")] == ["moving ink", "moving ink", "still ink", "still ink", "ring ink", "dot failed", "dot owner"], by
    for task_id in ("rw-live", "rw-unknown", "rw-queued", "rw-ask"):
        assert by[task_id]["more"] and by[task_id]["moreId"] == task_id, by[task_id]

    # Two lines at rest; keyboard focus shows the whole name; the row's ⋯ never grows it under the pointer.
    heights = page.evaluate("""() => { const name = document.querySelector('#page-chat .rw-row[data-id="rw-live"] .rw-name');
        return { rest: name.getBoundingClientRect().height, line: parseFloat(getComputedStyle(name).lineHeight) }; }""")
    assert heights["rest"] <= heights["line"] * 2 + 1, heights
    row_main(page, "rw-ask").focus()
    page.keyboard.press("ArrowDown")
    focused = wait_facts(page, "main", lambda f: f["active"]["row"] == "rw-live", "keyboard focus on the long row")
    grown = page.evaluate("""() => document.querySelector('#page-chat .rw-row[data-id="rw-live"] .rw-name')
        .getBoundingClientRect().height""")
    if room_ui["engine"] == "chromium":  # WebKit's :focus-visible after a programmatic focus is engine-specific
        assert row(focused, "rw-live")["reading"] == "1" and grown > heights["rest"] + 1, (focused, grown, heights)
    capture(page, "main-states-long-name-focus")

    # A card-less row asks the room's history on an explicit press, never a hover, and says what it found:
    # here the saved history holds no message of it, so it opens its details (finished work included later).
    # The press comes while the long row above holds keyboard focus and its whole name, and blurs it. Nothing may
    # move under the pointer: a fold at mouse-down moved the pressed row up from under it (36 px with wider fonts)
    # and the click landed on the list. A press folds nothing.
    page.evaluate(_PRESS_RECORDER, ["rw-live", "rw-unknown"])
    row_main(page, "rw-unknown").click()
    detail = wait_facts(page, "main", lambda f: "No message" in row(f, "rw-unknown")["detail"], "an unloaded row's lookup")
    press = page.evaluate("() => window.__rwPress")
    assert press["pointerdown"] == press["click"], press
    assert row(detail, "rw-live")["reading"] == row(focused, "rw-live")["reading"], "a press leaves the long row as it was"
    assert transport.around_reads == ["task:rw-unknown"]
    assert row(detail, "rw-unknown")["expanded"] == "true" and row(detail, "rw-unknown")["open"]
    assert row(detail, "rw-unknown")["detail"].startswith("WhereNo message yet.")
    assert row(detail, "rw-unknown")["label"].startswith("Details: ")
    assert "Not in loaded messages" not in json.dumps(detail) and "cannot take you to it" not in json.dumps(detail)
    assert row(detail, "rw-unknown")["detail"].endswith("WhatA task"), "label and words (textContent joins them)"
    assert not detail["surfaceHidden"]
    assert transport.decision_posts() == [] and transport.posts == []
    if room_ui["engine"] == "chromium":  # keyboard reading needs :focus-visible, engine-specific after a focus()
        # Keyboard focus reading another row folds the long one; the list never holds two grown names.
        page.keyboard.press("ArrowUp")
        moved = wait_facts(page, "main", lambda f: f["active"]["row"] not in ("rw-unknown", "rw-live"),
                           "keyboard focus on the row above")
        assert row(moved, "rw-live")["reading"] == "0", moved
        # A ⋯ pressed while a name is grown opens its menu on a button that stays where it was pressed.
        row_main(page, "rw-ask").focus()
        page.keyboard.press("ArrowDown")
        wait_facts(page, "main", lambda f: row(f, "rw-live")["reading"] == "1", "the long row grown again")
        page.evaluate(_ANCHOR_AT_CLICK, "rw-unknown")
        page.locator("#page-chat .rw-row[data-id='rw-unknown'] .rw-more").click()
        menu = page.locator("body > .task-control-menu")
        menu.wait_for(state="visible")
        page.evaluate(_SETTLE_FRAMES)
        anchor = page.evaluate("() => ({ click: window.__rwAnchor.click, now: window.__rwAnchor.now() })")
        assert anchor["click"] == anchor["now"] and menu.is_visible(), anchor
        # Pressing the grown row itself near its bottom edge, with that menu still open: the menu closes and hands
        # focus back to its ⋯ a moment later, inside a press a person holds; the row stays whole until the click
        # goes to its card.
        page.evaluate(_PRESS_RECORDER, ["rw-live"])
        box = row_main(page, "rw-live").bounding_box()
        page.mouse.move(box["x"] + 12, box["y"] + box["height"] - 4)
        page.mouse.down()
        page.wait_for_timeout(150)
        page.mouse.up()
        wait_facts(page, "main", lambda f: f["surfaceHidden"], "the grown row's press went to its card")
        press = page.evaluate("() => window.__rwPress")
        assert press["pointerdown"] == press["click"], press
    evidence("main-states-console", transport.console)


# ------------------------------------------------------------------------------------------------
# Exact id and attempt; the shared controls; partial census; disconnect; finish while open.

def test_controls_follow_the_exact_attempt_partial_census_disconnect_and_finish(room_ui):
    history = {1: [*reading(8), progress("rw-live", "Comparing the cited pages.", 30, suggested_name="Compare the cited pages")]}
    transport = Transport(room_ui["page"], history=history, details={
        "rw-other": {"task_id": "rw-other", "status": "running", "suggested_name": "Index the archived notes"}},
        activities=[activity("rw-live"), activity("rw-other")])
    page = open_app(room_ui, transport)
    trigger(page).click()
    wait_facts(page, "main", lambda f: len(f["rows"]) == 2 and row(f, "rw-other")["name"] == "Index the archived notes",
               "two named rows")
    more = page.locator("#page-chat .rw-row[data-id='rw-live'] .rw-more")

    # A new attempt of the same id never inherits the menu opened for the previous one.
    more.click()
    page.locator("body > .task-control-menu").wait_for(state="visible")
    assert more.get_attribute("aria-expanded") == "true"
    transport.set("rw-live", task_attempt=2)
    transport.refresh()
    attempt = wait_facts(page, "main", lambda f: f["menu"] is None, "the old attempt's menu closed")
    assert row(attempt, "rw-live")["more"], "the new attempt keeps its own controls"

    # Pause from the row reaches the exact id; Resume is the row's own button for that id.
    more.click()
    menu = page.locator("body > .task-control-menu")
    menu.wait_for(state="visible")
    assert menu.evaluate("n => [...n.querySelectorAll('.task-control-item')].map(i => i.dataset.taskControl)") == [
        "finalize", "hurry", "pause", "stop_now"]
    menu.locator('[data-task-control="pause"]').click()
    page.wait_for_function("() => [...document.querySelectorAll('.toast')].some(t => t.textContent.includes('Pausing'))")
    assert [path for path, _ in transport.posts] == ["/api/tasks/rw-live/pause"], transport.posts
    assert transport.posts[0][1].get("request_id"), transport.posts
    transport.set("rw-live", phase="budget_pausing", pause_cause="owner")
    transport.refresh()
    # The row says the work's word; a press this transport records no author for names nobody beside it, and a
    # Pause is never amber.
    wait_facts(page, "main", lambda f: (row(f, "rw-live")["word"], row(f, "rw-live")["secondary"]) == ("Pausing…", ""),
               "Pausing in place")
    transport.set("rw-live", phase="budget_paused")
    transport.refresh()
    paused = wait_facts(page, "main", lambda f: row(f, "rw-live")["resume"], "Resume on the paused row")
    assert (row(paused, "rw-live")["word"], row(paused, "rw-live")["secondary"], row(paused, "rw-live")["tone"]) == (
        "Paused", "", "held"), row(paused, "rw-live")
    page.locator("#page-chat .rw-row[data-id='rw-live'] .rw-pill-resume").click()
    page.wait_for_function("() => [...document.querySelectorAll('.toast')].some(t => t.textContent.includes('Resuming'))")
    assert [path for path, _ in transport.posts] == ["/api/tasks/rw-live/pause", "/api/tasks/rw-live/resume"]
    transport.set("rw-live", phase="working", pause_cause=None)
    transport.refresh()
    wait_facts(page, "main", lambda f: row(f, "rw-live")["word"] == "Working" and not row(f, "rw-live")["resume"],
               "working again without Resume")

    # A partial census confirms only what it lists: the rest stays, unconfirmed and without controls.
    row_main(page, "rw-other").click()
    transport.complete = False
    transport.drop("rw-other")
    transport.refresh()
    partial = wait_facts(page, "main", lambda f: row(f, "rw-other")["tone"] == "unknown", "an unconfirmed row")
    evidence("partial", partial)
    assert partial["note"] == "Not all current work could be confirmed; this list may be incomplete."
    assert not row(partial, "rw-other")["more"] and "Why no controls" in row(partial, "rw-other")["detail"]
    assert row(partial, "rw-live")["more"], "a listed row stays confirmed"
    capture(page, "partial-census")

    # A socket interruption: the last known rows stay, unconfirmed, without controls, and say so.
    transport.complete = True
    transport.activities.append(activity("rw-other"))
    transport.refresh()
    wait_facts(page, "main", lambda f: row(f, "rw-other")["more"], "both confirmed again")
    count = transport.disconnect()
    offline = wait_facts(page, "main", lambda f: f["note"].startswith("Reconnecting"), "the reconnecting note")
    evidence("offline", offline)
    assert offline["note"] == "Reconnecting — showing the last known state."
    assert all(item["tone"] == "unknown" and not item["more"] and not item["resume"] for item in offline["rows"]), offline
    assert [item["id"] for item in offline["rows"]] == ["rw-live", "rw-other"], "order holds while open"
    capture(page, "offline-open")
    transport.await_reconnect(count)
    transport.refresh()
    online = wait_facts(page, "main", lambda f: not f["note"] and all(item["more"] for item in f["rows"]),
                        "controls back after reconnect")
    assert [item["tone"] for item in online["rows"]] == ["live", "live"], online

    # Finish while open: names and opened details survive; controls go; outcome only where known.
    transport.details["rw-live"] = {"task_id": "rw-live", "status": "completed", "suggested_name": "Compare the cited pages"}
    transport.details["rw-other"] = {"task_id": "rw-other", "status": "completed", "suggested_name": "Index the archived notes"}
    transport.activities = []
    transport.refresh()
    done = wait_facts(page, "main", lambda f: all(item["settled"] for item in f["rows"])
                      and row(f, "rw-live")["word"] == "Done", "both rows settled", timeout=30)
    evidence("finished-open", done)
    capture(page, "finished-open")
    live, other = row(done, "rw-live"), row(done, "rw-other")
    assert live["name"] == "Compare the cited pages" and live["tone"] == "done" and live["target"] == "card", live
    assert other["name"] == "Index the archived notes" and other["word"] == "No longer listed as active", other
    assert other["open"] and "no longer listed as active; its outcome is not known here" in other["detail"].lower(), other
    assert not any(item["more"] or item["resume"] or item["question"] for item in done["rows"]), done
    # Closed by Escape from inside, Main keeps no finished list, and the keyboard reader keeps a focus.
    row_main(page, "rw-live").focus()
    page.keyboard.press("Escape")
    gone = wait_facts(page, "main", lambda f: f["surfaceHidden"] and f["disabled"], "no finished list in Main")
    assert gone["rows"] == [] or all(item["settled"] for item in gone["rows"])
    assert transport.decision_posts() == []
    focus = page.evaluate("() => document.activeElement === document.body || !document.activeElement ? 'body'"
                          " : `${document.activeElement.tagName}.${document.activeElement.className}`")
    evidence("focus-after-escape-on-finished-list", focus)
    assert focus != "body", "Escape on a list of finished work dropped focus to the document"


# ------------------------------------------------------------------------------------------------
# Parent hypothesis 1: an open shared menu never outlives the census state it was opened for.

FULL_MENU = ["finalize", "hurry", "pause", "stop_now"]


@pytest.mark.parametrize("ending", ["finished", "offline"])
@pytest.mark.parametrize("subject", ["card", "cardless"])
def test_open_task_menu_never_outlives_its_census_state(room_ui, subject, ending):
    """The shared menu opened from a row is for that row's state when it opened. Pausing keeps a retry of Pause;
    Paused offers only Stop now (Resume is the row's own button); a finished or offline row offers nothing.
    A stale menu must go (or match) by the frame the row says its new state, and a
    withdrawn Pause or Hurry pressed in it must post nothing (the menu calls them before any caller guard)."""
    task_id = "rw-live" if subject == "card" else "rw-far"
    history = {1: [*reading(6), *([progress("rw-live", "Comparing the cited pages.", 30,
                                            suggested_name="Compare the cited pages")] if subject == "card" else [])]}
    transport = Transport(room_ui["page"], history=history, activities=[activity(task_id)], details={
        "rw-far": {"task_id": "rw-far", "status": "running", "suggested_name": "Index the archived notes"}})
    page = open_app(room_ui, transport)
    trigger(page).click()
    wait_facts(page, "main", lambda f: row(f, task_id) and row(f, task_id)["more"] and row(f, task_id)["name"],
               "the row's controls")
    selector = f"#page-chat .rw-row[data-id='{task_id}']"
    more = page.locator(f"{selector} .rw-more")
    observed: dict = {}
    count = None

    def open_menu():
        if facts(page)["menu"] is None:
            more.click()
            page.locator("body > .task-control-menu").wait_for(state="visible")
        return facts(page)["menu"]

    def transition(label, reached, change):
        before = facts(page)["menu"]
        page.evaluate(_FRAME_RECORDER, [selector, 1500])
        change()
        page.wait_for_timeout(1600)
        frames = page.evaluate("() => window.__rwFrames")
        first = next((frame for frame in frames if reached(frame)), None)
        observed[label] = {"menu_before": before, "first_frame": first, "last_frame": frames[-1],
                           "transitions": [frame for frame in frames if frame["menu"] != before or reached(frame)][:6]}
        assert first, (label, frames[-3:])
        return first, frames[-1]

    def census(**fields):
        def change():
            transport.set(task_id, **fields)
            transport.refresh()
        return change

    problems = []

    def press_stale(stage, menu_items):
        """A stale menu's consequence: its Hurry/Pause run before any caller guard and post."""
        forbidden = ("hurry",) if stage == "pausing" else ("hurry", "pause")
        action = next((name for name in forbidden if name in (menu_items or [])), "")
        if not action:
            return
        posted = len(transport.posts)
        page.locator(f'body > .task-control-menu [data-task-control="{action}"]').click(force=True)
        page.wait_for_timeout(600)
        observed[f"pressed_{stage}"] = {"action": action, "posts": transport.posts[posted:]}
        if transport.posts[posted:]:
            problems.append(f"{stage}: a stale menu's {action} posted {transport.posts[posted:]}")

    try:
        assert open_menu() == FULL_MENU
        # Pausing retains the coalescing Pause retry, but withdraws Hurry and Wrap up.
        first, last = transition("pausing", lambda f: (f["word"], f["secondary"]) == ("Pausing…", ""),
                                 census(phase="budget_pausing", pause_cause="owner"))
        if first["menu"] not in (None, ["pause", "stop_now"]) or last["menu"] not in (None, ["pause", "stop_now"]):
            problems.append(f"pausing: a stale menu outlived Working → Pausing (first frame {first['menu']}, "
                            f"after {last['t']} ms {last['menu']})")
        press_stale("pausing", facts(page)["menu"])
        assert open_menu() == ["pause", "stop_now"], "a fresh menu keeps the accepted Pause retry"
        posted = len(transport.posts)
        page.locator('body > .task-control-menu [data-task-control="pause"]').click()
        page.wait_for_function("() => [...document.querySelectorAll('.toast')].some(t => t.textContent.includes('Pausing'))")
        assert [path for path, _ in transport.posts[posted:]] == [f"/api/tasks/{task_id}/pause"]
        assert transport.posts[-1][1].get("request_id"), transport.posts[-1]
        # Hover pauses a toast's lifetime (#2193); move away so it expires while the Work list stays open.
        notice = page.locator('#toast-stack .toast').filter(has_text="Pausing:")
        page.mouse.move(0, 0)
        notice.wait_for(state="detached")
        assert open_menu() == ["pause", "stop_now"]
        # Paused: a fresh open offers only Stop now (Resume is the row's own button); the old menu must go or match.
        first, last = transition("paused", lambda f: (f["word"], f["secondary"]) == ("Paused", ""),
                                 census(phase="budget_paused"))
        if first["menu"] not in (None, ["stop_now"]) or last["menu"] not in (None, ["stop_now"]):
            problems.append(f"paused: a stale menu outlived Pausing → Paused (first frame {first['menu']}, "
                            f"after {last['t']} ms {last['menu']})")
        press_stale("paused", facts(page)["menu"])
        assert open_menu() == ["stop_now"]
        if facts(page)["menu"]:
            page.keyboard.press("Escape")

        # Working again, a fresh menu; then the row ends while it is open.
        transport.set(task_id, phase="working", pause_cause=None)
        transport.refresh()
        wait_facts(page, "main", lambda f: row(f, task_id)["word"] == "Working" and row(f, task_id)["more"], "working")
        assert open_menu() == FULL_MENU
        if ending == "finished":
            def end():
                transport.details[task_id] = {"task_id": task_id, "status": "completed"}
                transport.drop(task_id)
                transport.refresh()
            first, last = transition("ended", lambda f: f["settled"] == "1" and not f["more"], end)
        else:
            count = len(transport.sockets)

            def end():
                transport.disconnect()
            first, last = transition("ended", lambda f: not f["more"], end)
        stale = facts(page)["menu"]
        observed["stale_menu_after_end"] = stale
        if first["menu"] is not None or last["menu"] is not None or stale is not None:
            problems.append(f"{ending}: a menu outlived its row (first frame {first['menu']}, "
                            f"after {last['t']} ms {last['menu']}, now {stale})")
        press_stale(ending, stale)
        assert not problems, problems
    finally:
        observed["problems"] = problems
        observed["posts"] = list(transport.posts)
        evidence(f"menu-lifetime-{subject}-{ending}", observed)
        if count is not None:
            transport.await_reconnect(count)


# ------------------------------------------------------------------------------------------------
# Parent hypothesis 2: a card-less row's controls never open after close, leaving or going offline.

@pytest.mark.parametrize("leave", ["escape", "outside", "project", "offline"])
def test_cardless_controls_never_open_after_close_leave_or_offline(room_ui, leave):
    """The ⋯ of a card-less row first reads the durable record. When that read lands after the list
    closed (Escape, an outside press), the room was left or the connection went offline, no menu may
    open: not a visible one and not one hung on a hidden anchor (whose Pause would still post)."""
    transport = Transport(room_ui["page"], history={1: reading(6), 42: reading(2, 0, "Project")},
                          projects=[PROJECT], details={"rw-far": {"task_id": "rw-far", "status": "running",
                                                                  "suggested_name": "Index the archived notes"}},
                          activities=[activity("rw-far")])
    page = open_app(room_ui, transport)
    trigger(page).click()
    wait_facts(page, "main", lambda f: row(f, "rw-far") and row(f, "rw-far")["name"] == "Index the archived notes",
               "the card-less row")
    transport.hold.add("rw-far")
    page.locator("#page-chat .rw-row[data-id='rw-far'] .rw-more").click()
    page.wait_for_timeout(200)
    assert any(task_id == "rw-far" for task_id, _ in transport.held), "the strict detail read is in flight"
    assert facts(page)["menu"] is None
    count = None
    if leave == "escape":
        page.keyboard.press("Escape")
        wait_facts(page, "main", lambda f: f["surfaceHidden"], "the list closed")
    elif leave == "outside":
        box = page.locator("#chat-messages").bounding_box()
        page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] - 40)
        wait_facts(page, "main", lambda f: f["surfaceHidden"], "the list closed by an outside press")
    elif leave == "project":
        open_project(page, PROJECT)
        wait_facts(page, "main", lambda f: f["surfaceHidden"], "the list closed by leaving")
    else:
        count = transport.disconnect()
        wait_facts(page, "main", lambda f: f["note"].startswith("Reconnecting") or f["surfaceHidden"], "offline")
    transport.hold.discard("rw-far")
    page.evaluate(_FRAME_RECORDER, ["#page-chat .rw-row[data-id='rw-far']", 1200])
    transport.release("rw-far")
    page.wait_for_timeout(1300)
    frames = page.evaluate("() => window.__rwFrames")
    expanded = page.evaluate("() => document.querySelector(\"#page-chat .rw-row[data-id='rw-far'] .rw-more\")"
                             "?.getAttribute('aria-expanded') ?? null")
    evidence(f"cardless-{leave}", {"frames": frames[:5] + frames[-3:], "facts": facts(page), "expanded": expanded})
    try:
        assert all(frame["menu"] is None for frame in frames), (
            "a menu opened after " + leave, [f for f in frames if f["menu"]][:3])
        assert page.locator(".task-control-menu").count() == 0
        assert expanded in (None, "false"), ("the ⋯ claims an open menu", expanded)
        assert transport.posts == []
    finally:
        if count is not None:
            transport.await_reconnect(count)


# ------------------------------------------------------------------------------------------------
# View question: the exact question, never an answer; Main proves its room; Project still matches.

def test_view_question_reaches_the_exact_question_and_proves_its_room(room_ui):
    rows = [owner(ASK_TEXT, "cm-ask", 0), *reading(12, 1)]
    stamped = {"rw-nostamp": {}, "rw-elsewhere": {"chat_id": 2}, "rw-project-stamp": {"chat_id": 1, "project_id": "x",
               "question_room": {"state": "known", "chat_id": 99, "project_id": "x"}}}  # converted, not contextual
    transport = Transport(room_ui["page"], history={1: rows, 42: reading(4, 0, "Project")}, projects=[PROJECT], details={
        "rw-ask": quiz_detail("rw-ask", "q-main", "Which evidence should the weekly summary cite first?", chat_id=1),
        **{task_id: quiz_detail(task_id, f"q-{task_id}", f"Question of {task_id}", **stamp)
           for task_id, stamp in stamped.items()},
        "p-ask": quiz_detail("p-ask", "q-proj", "Which archive goes first?", project_id=PROJECT["id"], chat_id=42),
        "p-wrong": quiz_detail("p-wrong", "q-wrong", "A question of another Project", project_id="other", chat_id=42),
    }, activities=[
        activity("rw-ask", kind="direct_chat", client_message_id="cm-ask", **waited("q-main")),
        *[activity(task_id, **waited(f"q-{task_id}")) for task_id in stamped],
        activity("p-ask", chat_id=42, project_id=PROJECT["id"], **waited("q-proj")),
        activity("p-wrong", chat_id=42, project_id=PROJECT["id"], **waited("q-wrong")),
    ])
    page = open_app(room_ui, transport, viewport={"width": 1100, "height": 722})
    feed = page.locator("#chat-messages")
    assert page.locator("#chat-messages .chat-quiz-card").count() == 0, "the question is outside the loaded history"
    feed.evaluate("el => { el.scrollTop = Math.round(el.scrollHeight * 0.45); }")
    page.wait_for_timeout(200)
    reading_at = feed.evaluate("el => el.scrollTop")
    page.locator("#chat-input").fill("Draft stays for Main")
    trigger(page).click()
    wait_facts(page, "main", lambda f: row(f, "rw-ask") and row(f, "rw-ask")["question"], "View question")
    capture(page, "question-open")
    page.locator("#page-chat .rw-row[data-id='rw-ask'] .rw-pill-question").click()
    card = page.locator('#chat-messages .chat-quiz-card[data-task-id="rw-ask"][data-quiz-id="q-main"]')
    card.wait_for(state="visible")
    page.wait_for_function("() => document.activeElement?.classList.contains('chat-quiz-question')")
    landed = page.evaluate("""() => { const card = document.querySelector('.chat-quiz-card[data-quiz-id="q-main"]');
        const feed = document.querySelector('#chat-messages'); const box = card.getBoundingClientRect(), view = feed.getBoundingClientRect();
        return { focusInCard: card.contains(document.activeElement), focusClass: document.activeElement.className,
            comment: card.querySelector('.chat-quiz-comment')?.value ?? null,
            selected: [...card.querySelectorAll('.chat-quiz-option')].filter(b => b.getAttribute('aria-pressed') === 'true'
                || b.classList.contains('selected')).length,
            enabled: [...card.querySelectorAll('.chat-quiz-option')].filter(b => !b.disabled).length,
            visible: box.bottom > view.top && box.top < view.bottom }; }""")
    evidence("question-landed", landed)
    assert landed["focusInCard"] and "chat-quiz-question" in landed["focusClass"], landed
    assert landed["comment"] in ("", None) and landed["selected"] == 0 and landed["enabled"] == 2, landed
    assert landed["visible"], landed
    # The list is closed at once and folds away (the Shade's own motion) before it is hidden.
    after = wait_facts(page, "main", lambda f: f["surfaceHidden"], "the list folded away")
    assert after["back"] == "Back to reading", after
    assert page.locator("#chat-input").input_value() == "Draft stays for Main"
    assert transport.decision_posts() == [], "View question answers nothing"
    capture(page, "question-landed")
    page.locator("#page-chat .rw-back").click()
    page.wait_for_timeout(300)
    assert abs(feed.evaluate("el => el.scrollTop") - reading_at) <= 40, "Back to reading returns to the passage left"
    assert facts(page)["back"] == ""

    # A Main question whose detail does not prove Main's exact chat is not rendered.
    for task_id in stamped:
        trigger(page).click()
        wait_facts(page, "main", lambda f, t=task_id: row(f, t) and row(f, t)["question"], f"View question of {task_id}")
        page.locator(f"#page-chat .rw-row[data-id='{task_id}'] .rw-pill-question").click()
        page.wait_for_function("() => [...document.querySelectorAll('.toast')].some(t => t.textContent.includes('Question unavailable'))")
        page.wait_for_timeout(200)
        assert page.locator(f'.chat-quiz-card[data-task-id="{task_id}"]').count() == 0, task_id
        assert facts(page)["back"] == "", f"no Back to reading after a refused reveal ({task_id})"
        page.evaluate("() => document.querySelectorAll('.toast').forEach(t => t.remove())")

    # A Project question still proves its Project.
    open_project(page, PROJECT)
    trigger(page, "project").click()
    wait_facts(page, "project", lambda f: row(f, "p-ask") and row(f, "p-ask")["question"], "the Project's View question")
    assert {item["id"] for item in facts(page, "project")["rows"]} == {"p-ask", "p-wrong"}
    page.locator("#project-panel .rw-row[data-id='p-ask'] .rw-pill-question").click()
    page.locator('#project-panel .chat-quiz-card[data-task-id="p-ask"][data-quiz-id="q-proj"]').wait_for(state="visible")
    page.wait_for_function("() => document.activeElement?.classList.contains('chat-quiz-question')")
    trigger(page, "project").click()
    wait_facts(page, "project", lambda f: row(f, "p-wrong") and row(f, "p-wrong")["question"], "the other question")
    page.locator("#project-panel .rw-row[data-id='p-wrong'] .rw-pill-question").click()
    page.wait_for_function("() => [...document.querySelectorAll('.toast')].some(t => t.textContent.includes('Question unavailable'))")
    assert page.locator('.chat-quiz-card[data-task-id="p-wrong"]').count() == 0
    assert transport.decision_posts() == []


# ------------------------------------------------------------------------------------------------
# A late question read: after a gesture it never steals reading; after leaving it never reveals or
# offers Back to reading (parent hypothesis 3).

@pytest.mark.parametrize("interrupt", ["wheel", "project", "project-and-back", "socket-reopen"])
def test_late_question_read_never_steals_reading_or_offers_back(room_ui, interrupt):
    rows = [owner(ASK_TEXT, "cm-ask", 0), *reading(14, 1)]
    transport = Transport(room_ui["page"], history={1: rows, 42: reading(3, 0, "Project")}, projects=[PROJECT], details={
        "rw-ask": quiz_detail("rw-ask", "q-main", "Which evidence should the weekly summary cite first?", chat_id=1)},
        activities=[activity("rw-ask", kind="direct_chat", client_message_id="cm-ask", **waited("q-main"))])
    page = open_app(room_ui, transport, viewport={"width": 1100, "height": 722})
    feed = page.locator("#chat-messages")
    feed.evaluate("el => { el.scrollTop = Math.round(el.scrollHeight * 0.4); }")
    page.wait_for_timeout(200)
    trigger(page).click()
    wait_facts(page, "main", lambda f: row(f, "rw-ask") and row(f, "rw-ask")["question"], "View question")
    transport.hold.add("rw-ask")
    page.locator("#page-chat .rw-row[data-id='rw-ask'] .rw-pill-question").click()
    page.wait_for_timeout(200)
    assert any(task_id == "rw-ask" for task_id, _ in transport.held), "the question read is in flight"
    if interrupt == "socket-reopen":
        count = transport.disconnect()
        transport.await_reconnect(count)
    elif interrupt == "wheel":
        box = feed.bounding_box()
        page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        page.mouse.wheel(0, -500)
        page.wait_for_timeout(300)
    else:
        open_project(page, PROJECT)
        if interrupt == "project-and-back":
            page.locator("#project-panel-close").click()
            page.wait_for_function("() => document.getElementById('project-panel').hidden")
            page.wait_for_timeout(300)
    position = feed.evaluate("el => el.scrollTop")
    active = page.evaluate("() => document.activeElement?.className || ''")
    transport.hold.discard("rw-ask")
    transport.release("rw-ask")
    page.wait_for_timeout(800)
    settle_frames(page)
    result = {"before": position, "after": feed.evaluate("el => el.scrollTop"),
              "active_before": active, "active_after": page.evaluate("() => document.activeElement?.className || ''"),
              "facts": facts(page), "cards": page.locator('.chat-quiz-card[data-quiz-id="q-main"]').count()}
    evidence(f"late-question-{interrupt}", result)
    assert result["facts"]["back"] == "", ("a late read offered Back to reading", result)
    assert "chat-quiz-question" not in result["active_after"], ("a late read took focus", result)
    assert abs(result["after"] - result["before"]) <= 2, ("a late read moved the reader", result)
    if interrupt == "socket-reopen":
        assert result["cards"] == 0, ("a prior connection's Header read created an answer form", result)
        if facts(page)["surfaceHidden"]:
            trigger(page).click()
        page.locator("#page-chat .rw-row[data-id='rw-ask'] .rw-pill-question").click()
        page.locator('.chat-quiz-card[data-quiz-id="q-main"]').wait_for(state="visible")
        page.wait_for_function("() => document.activeElement?.classList.contains('chat-quiz-question')")
        assert page.locator('.chat-quiz-card[data-quiz-id="q-main"]').count() == 1
        capture(page, "header-fresh-question-after-reopen")
    assert transport.decision_posts() == []


# ------------------------------------------------------------------------------------------------
# Project: its Shade lists only its room; holds, outside press, close, switch and its latest root.

def test_project_shade_hold_outside_press_switch_and_latest_root(room_ui):
    history = {1: reading(6), 42: [*reading(8, 0, "Project"), progress("p-done", "Finished the archive pass.", 20,
                                                                      suggested_name="Archive pass")],
               43: reading(3, 0, "Release")}
    transport = Transport(room_ui["page"], history=history, projects=[PROJECT, OTHER_PROJECT], details={
        "p-done": {"task_id": "p-done", "status": "completed", "suggested_name": "Archive pass"},
        "p-hold": {"task_id": "p-hold", "status": "scheduled", "suggested_name": "Verify the archive folder"}},
        activities=[activity("p-hold", "queued", chat_id=42, project_id=PROJECT["id"],
                             project_admission_hold={"label": "Waiting for Project access"})])
    page = open_app(room_ui, transport, viewport={"width": 1100, "height": 722})
    main = wait_facts(page, "main", lambda f: f["exists"], "Main's control")
    assert main["disabled"] and main["rows"] == [], "Main never lists a Project's work"
    open_project(page, PROJECT)
    closed = wait_facts(page, "project", lambda f: f["tone"] == "held" and not f["disabled"], "the Project's control")
    capture(page, "project-closed")
    trigger(page, "project").click()
    shade = wait_facts(page, "project", lambda f: [item["id"] for item in f["rows"]] == ["p-hold"]
                       and row(f, "p-hold")["name"] == "Verify the archive folder", "the held row")
    evidence("project-shade", {"closed": closed, "open": shade})
    held = row(shade, "p-hold")
    # The room waits for the host's verification: the row says Waiting, and the hold's own label beside it.
    assert (held["word"], held["secondary"], held["tone"]) == ("Waiting", "Waiting for Project access", "held"), held
    settle_unfold(page, "project")
    geometry = page.evaluate(_BAR_GEOMETRY, "project")
    shade, panel = geometry["shade"], page.locator("#project-panel").bounding_box()
    assert_hangs_under_capsule(shade, geometry["capsule"], "the held Project's list")
    assert shade["x"] >= panel["x"] - 0.5 and shade["x"] + shade["w"] <= panel["x"] + panel["width"] + 0.5, (shade, panel)
    assert shade["bottom"] <= geometry["vh"], geometry
    capture(page, "project-open")
    # The first outside press on the dimmed Main only dismisses the Shade, never the room.
    backdrop = page.locator("#project-panel-backdrop")
    if backdrop.is_visible():
        backdrop.click(position={"x": 20, "y": 300})
        wait_facts(page, "project", lambda f: f["surfaceHidden"], "the Shade dismissed")
        assert page.evaluate("() => !document.getElementById('project-panel').hidden"), "the room stayed open"
        trigger(page, "project").click()
        wait_facts(page, "project", lambda f: not f["surfaceHidden"], "the Shade again")
    # The panel's own close takes the Shade along.
    page.locator("#project-panel-close").click()
    page.wait_for_function("() => document.getElementById('project-panel').hidden")
    assert page.locator("#project-panel .rw-shade:not([hidden])").count() == 0
    # Its latest represented root stays reachable when nothing else is listed.
    transport.drop("p-hold")
    transport.details["p-hold"] = {"task_id": "p-hold", "status": "completed"}
    open_project(page, PROJECT)
    transport.refresh()
    trigger(page, "project").click()
    latest = wait_facts(page, "project", lambda f: f["groups"] == ["Latest task"] and len(f["rows"]) == 1,
                        "the latest root", timeout=30)
    evidence("project-latest", latest)
    capture(page, "project-latest")
    done = latest["rows"][0]
    assert done["id"] == "p-done" and done["word"] == "Done" and done["target"] == "card", done
    assert not (done["more"] or done["resume"] or done["question"]), done
    # Switching rooms takes the control along; the new room shows only its own.
    open_project(page, OTHER_PROJECT)
    switched = wait_facts(page, "project", lambda f: f["exists"], "the other Project's control")
    assert switched["disabled"] and switched["rows"] == [], switched
    assert page.locator("#panel-pchat-rw-proj").count() == 0
    assert page.locator("#project-panel .rw-trigger").count() == 1
    page.locator("#project-panel-close").click()
    main = wait_facts(page, "main", lambda f: f["exists"], "Main's control")
    assert main["disabled"], "Main keeps no finished list"
    assert transport.decision_posts() == [] and transport.posts == []


# ------------------------------------------------------------------------------------------------
# Keyboard and focus; Files, Copy and the composer stay independent of the list.

def test_keyboard_focus_escape_and_independent_controls(room_ui):
    history = {1: [*reading(8), progress("rw-live", "Comparing the cited pages.", 30, suggested_name="Compare the cited pages"),
                   assistant("The copyable answer stays independent of the list.", 31)]}
    transport = Transport(room_ui["page"], history=history, details={
        "rw-other": {"task_id": "rw-other", "status": "running", "suggested_name": "Index the archived notes"}},
        activities=[activity("rw-live"), activity("rw-other", "queued")])
    page = open_app(room_ui, transport)
    control = trigger(page)
    wait_facts(page, "main", lambda f: not f["disabled"], "an enabled control")
    control.focus()
    page.keyboard.press("Enter")
    entered = wait_facts(page, "main", lambda f: not f["surfaceHidden"] and f["active"]["inSurface"], "a keyboard open")
    assert entered["active"]["row"] == "rw-live" and "rw-main" in entered["active"]["cls"], entered
    page.keyboard.press("ArrowDown")
    assert wait_facts(page, "main", lambda f: f["active"]["row"] == "rw-other", "ArrowDown")["active"]["row"] == "rw-other"
    page.keyboard.press("ArrowUp")
    wait_facts(page, "main", lambda f: f["active"]["row"] == "rw-live", "ArrowUp")
    page.keyboard.press("Escape")
    closed = wait_facts(page, "main", lambda f: f["surfaceHidden"], "Escape")
    assert closed["active"]["isTrigger"], closed
    page.keyboard.press("ArrowDown")
    wait_facts(page, "main", lambda f: not f["surfaceHidden"] and f["active"]["inSurface"], "ArrowDown opens")
    # Escape in the shared menu closes only the menu; the next Escape closes the list.
    page.locator("#page-chat .rw-row[data-id='rw-live'] .rw-more").focus()
    page.keyboard.press("Enter")
    page.locator("body > .task-control-menu").wait_for(state="visible")
    page.keyboard.press("Escape")
    menu_closed = wait_facts(page, "main", lambda f: f["menu"] is None, "the menu closed")
    assert not menu_closed["surfaceHidden"], "Escape in the menu leaves the list open"
    page.keyboard.press("Escape")
    wait_facts(page, "main", lambda f: f["surfaceHidden"], "the list closed by its own Escape")
    # Tabbing out of the list dismisses it.
    control.focus()
    page.keyboard.press("Enter")
    wait_facts(page, "main", lambda f: not f["surfaceHidden"], "open again")
    for _ in range(12):
        page.keyboard.press("Tab")
        if facts(page)["expanded"] == "false":
            break
    wait_facts(page, "main", lambda f: f["surfaceHidden"], "Tab out of the list closes it")
    # Copy: one press both dismisses the list and copies; nothing is swallowed in Main.
    control.click()
    wait_facts(page, "main", lambda f: not f["surfaceHidden"], "open for Copy")
    bubble = page.locator("#chat-messages .chat-bubble.assistant", has_text="The copyable answer").last
    bubble.hover()
    bubble.locator(".chat-message-copy").click()
    assert page.evaluate("() => window.__copied") == ["The copyable answer stays independent of the list."]
    wait_facts(page, "main", lambda f: f["surfaceHidden"], "the press that copied dismissed the list")
    # Files: the composer's "+" → Attach file opens its chooser with the list open.
    control.click()
    wait_facts(page, "main", lambda f: not f["surfaceHidden"], "open for Files")
    with page.expect_file_chooser(timeout=5_000):
        page.locator("#chat-plus").click()
        page.locator(".chat-plus-menu:not([data-leaving]) [data-plus-item='attach']").click()
    wait_facts(page, "main", lambda f: f["surfaceHidden"], "Attach file dismissed the list")
    assert transport.posts == []


# ------------------------------------------------------------------------------------------------
# Responsive light/dark geometry and screenshots over real scrolling content.

_BAR_GEOMETRY = """(scope) => {
    const bar = scope === 'main' ? document.querySelector('#page-chat .chat-page-header')
        : document.querySelector('#project-panel .project-panel-bar');
    const box = (el) => { const r = el.getBoundingClientRect(); return { x: r.x, y: r.y, w: r.width, h: r.height, bottom: r.bottom }; };
    const visible = (el) => el.getClientRects().length > 0 && getComputedStyle(el).visibility !== 'hidden';
    const trigger = bar.querySelector('.rw-trigger');
    const badge = trigger.querySelector('.status-badge');
    const title = bar.querySelector('.app-page-title');
    const neighbours = [...bar.querySelectorAll('button, .chat-budget-pill')]
        .filter((el) => visible(el) && el !== trigger && !trigger.contains(el) && !el.closest('.rw-shade-anchor'))
        .map((el) => ({ name: el.id || el.dataset.chatCommand || String(el.className).split(' ')[0], ...box(el) }));
    const shade = document.getElementById(trigger.getAttribute('aria-controls'));  // the work Shade, not the chip's
    const alpha = (color) => { const m = color.match(/rgba?\\(([^)]+)\\)/); if (!m) return color === 'transparent' ? 0 : 1;
        const parts = m[1].split(/[ ,/]+/).filter(Boolean); return parts.length > 3 ? parseFloat(parts[3]) : 1; };
    const reachable = [...(shade?.querySelectorAll('.rw-more, .rw-main') || [])].filter(visible).every((el) => {
        const r = el.getBoundingClientRect(); const hit = document.elementFromPoint(r.x + r.width / 2, r.y + r.height / 2);
        return Boolean(hit) && (hit === el || el.contains(hit)); });
    return { bar: box(bar), capsule: box(bar.querySelector('.app-page-title-row')), title: box(title), trigger: box(trigger),
             neighbours, word: badge.textContent.trim(),
             clipped: badge.scrollWidth > badge.clientWidth + 0.5, note: !trigger.querySelector('.rw-sr').hidden,
             shade: shade && !shade.hidden ? box(shade) : null, shadeAlpha: shade ? alpha(getComputedStyle(shade).backgroundColor) : null,
             reachable, vw: innerWidth, vh: innerHeight, overflow: document.documentElement.scrollWidth - innerWidth };
}"""

# The model-wait owner's own row shape (web/tests/model_wait.test.js), for the access-wait word.
_ACCESS_WAIT = {"wait_id": "w-geo", "revision": 1, "task_attempt": 1, "role": "main", "reason": "quota",
                "model": "claudexor::source-a=same-model", "source": "source-a", "credential_harness": "codex",
                "credential_profile_id": "personal", "auto_continue": True, "state": "waiting", "worker_slot_held": True}


def header_states(room):
    """Named census states that give the room's header its different words, one after another."""
    where = {} if room == "main" else {"chat_id": PROJECT["chat_id"], "project_id": PROJECT["id"]}
    ask = activity("g-ask", kind="direct_chat", client_message_id="cm-geo", **waited("q-geo"), **where)
    states = [
        ("idle", []),
        ("live", [activity("g-live", **where)]),
        ("queued", [activity("g-queued", "queued", **where)]),
        ("finalizing", [activity("g-fin", "finalizing", **where)]),
        ("access", [activity("g-access", model_waits={"w-geo": _ACCESS_WAIT}, **where)]),
        ("answer", [ask]),
        ("pausing", [activity("g-pausing", "budget_pausing", pause_cause="owner", **where)]),
        ("paused", [activity("g-paused", "budget_paused", pause_cause="owner", **where)]),
        ("unknown", [activity("g-unknown", "unknown", **where)]),
        # A fresh question (its own id and quiz): the client remembers a question that left the census.
        ("live+answer", [activity("g-live", **where), activity("g-ask2", kind="direct_chat", client_message_id="cm-geo2",
                                                             **waited("q-geo2"), **where)]),
    ]
    if room == "project":
        # A Project hold (ouroboros/project_admission.py): the room waits; the hold's label is its row's reason.
        states.append(("hold", [activity("g-hold", "queued", project_admission_hold={
            "label": "Waiting for task scope verification"}, **where)]))
    return states


# The room words the header says for each named state; a reason is said on the room's rows, never here.
HEADER_WORDS = {"idle": "Online", "live": "Working", "queued": "Queued", "finalizing": "Finalizing…", "access": "Waiting",
                "answer": "Waiting for your answer", "pausing": "Pausing…", "paused": "Paused",
                "unknown": "Activity unconfirmed", "live+answer": "Working", "hold": "Waiting"}


def settle_unfold(page, room):
    """Wait until the room's Shade has finished unfolding (its 170 ms entrance slides it 6px), so geometry reads its
    resting place rather than a frame of the animation."""
    root = "#page-chat" if room == "main" else "#project-panel"
    page.wait_for_function("""(root) => { const s = document.querySelector(`${root} .rw-shade:not(.cx-shade)`);
        return Boolean(s) && !s.hidden && !(s.getAnimations?.() || []).some((a) => a.playState === 'running'); }""",
                           arg=root, timeout=10_000)
    settle_frames(page)


def settled_bar(page, room, expected, previous, timeout=15):
    """The bar once the census has repainted it: the expected word when known, else a word that changed from the
    previous state's and held still for a moment (a census read lands asynchronously, more so under load)."""
    deadline, last, stable = time.monotonic() + timeout, None, 0
    while time.monotonic() < deadline:
        sample = page.evaluate(_BAR_GEOMETRY, room)
        if expected is not None and sample["word"] == expected:
            settle_frames(page)
            return page.evaluate(_BAR_GEOMETRY, room)
        if expected is None and (previous is None or sample["word"] != previous["word"]):
            stable = stable + 1 if last and last["word"] == sample["word"] else 0
            if stable >= 3:
                return sample
        last = sample
        page.wait_for_timeout(100)
    raise AssertionError(("the header never settled", room, expected, last))


def assert_word_never_moves_the_bar(page, transport, room, tag):
    """Every word the header can say leaves the bar's height and every other control's box where they were,
    keeps the word whole (the title yields first) and overflows nothing."""
    samples, previous = {}, None
    for name, activities in header_states(room):
        transport.activities = activities
        transport.refresh()
        samples[name] = previous = settled_bar(page, room, HEADER_WORDS.get(name), previous)
    evidence(f"bar-words-{room}-{tag}", samples)
    words = [sample["word"] for sample in samples.values()]
    assert len(set(words)) >= 7, ("the census states reached the header as different words", words)
    first = samples["idle"]
    for name, sample in samples.items():
        assert abs(sample["bar"]["h"] - first["bar"]["h"]) <= 0.5, ("a word never resizes the bar", name, sample, first)
        assert abs(sample["trigger"]["h"] - first["trigger"]["h"]) <= 0.5, ("the status never wraps", name, sample)
        assert abs(sample["title"]["x"] - first["title"]["x"]) <= 0.5, ("the title keeps its place", name, sample)
        assert [n["name"] for n in sample["neighbours"]] == [n["name"] for n in first["neighbours"]], (name, sample)
        moved = [(a, b) for a, b in zip(first["neighbours"], sample["neighbours"])
                 if any(abs(a[k] - b[k]) > 0.5 for k in ("x", "y", "w", "h"))]
        assert not moved, ("a word never moves a neighbouring control", name, sample["word"], moved)
        assert not sample["clipped"], ("the status word stays whole; the title ellipsizes first", name, sample)
        assert sample["overflow"] <= 0, (name, sample)
    # What waits for the owner is said in words beside another word only, never beside its own (the marks of every
    # state: tests/test_status_marks_room_browser.py).
    assert [name for name, sample in samples.items() if sample["note"]] == ["live+answer"], samples
    assert samples["live+answer"]["word"] == samples["live"]["word"], samples
    if room == "project":
        assert samples["hold"]["word"] == "Waiting", samples["hold"]


@pytest.mark.parametrize("width,height", [(390, 844), (1040, 720), (1440, 860)])
def test_header_words_never_move_a_neighbour_or_resize_the_bar(room_ui, width, height):
    long_project = {**PROJECT, "name": "Archive evidence review — cross-checking every cited page against its source"}
    transport = Transport(room_ui["page"], history={1: reading(8), 42: reading(6, 0, "Project")},
                          projects=[long_project])
    page = open_app(room_ui, transport, theme="light", viewport={"width": width, "height": height})
    tag = f"{width}x{height}"
    assert_word_never_moves_the_bar(page, transport, "main", tag)
    capture(page, f"bar-words-main-{tag}")
    transport.activities = []
    transport.refresh()
    open_project(page, long_project)
    assert_word_never_moves_the_bar(page, transport, "project", tag)
    capture(page, f"bar-words-project-{tag}")
    evidence(f"bar-words-console-{tag}", transport.console)


def assert_hangs_under_capsule(shade, capsule, what):
    """The work list is a sheet of its own under the bar's capsules: from the status capsule's left edge, a gap
    (``--space-2``, 8px) below it (DESIGN "Room work status")."""
    assert abs((shade["y"] - capsule["bottom"]) - 8) <= 1.5, (what, "a gap below the status capsule", shade, capsule)
    assert abs(shade["x"] - capsule["x"]) <= 1.5, (what, "from the status capsule's left edge", shade, capsule)


def assert_project_backing(page):
    facts = page.locator("#project-panel").evaluate("""el => {
        const s = getComputedStyle(el);
        const m = s.backgroundColor.match(/rgba?\\(([^)]+)\\)/);
        const parts = m ? m[1].split(/[ ,/]+/).filter(Boolean) : [];
        return { color: s.backgroundColor, opacity: Number(s.opacity),
                 alpha: parts.length === 3 ? 1 : Number(parts[3] || 0) };
    }""")
    assert facts["alpha"] == 1 and facts["opacity"] == 1, ("Project reading backing", facts)


@pytest.mark.parametrize("theme", ["dark", "light"])
@pytest.mark.parametrize("width,height", [(1100, 722), (390, 844)])
def test_shade_geometry_light_dark_and_screenshots(room_ui, width, height, theme):
    history = {1: [*reading(14), progress("rw-live", "Comparing the cited pages.", 30, suggested_name=LONG_NAME),
                   progress("rw-pausing", "Collecting appendix evidence.", 31, suggested_name="Collect appendix evidence"),
                   *reading(6, 32, "Later")],
               42: [*reading(12, 0, "Project"), progress("p-live", "Reading the archive.", 30,
                                                          suggested_name="Read the archive folder")]}
    transport = Transport(room_ui["page"], history=history, projects=[PROJECT], details={
        "rw-far": {"task_id": "rw-far", "status": "running", "suggested_name": "Index the archived notes"}},
        activities=[activity("rw-live"), activity("rw-pausing", "budget_paused", pause_cause="owner"),
                    activity("rw-far", "queued"),
                    activity("p-live", chat_id=42, project_id=PROJECT["id"]),
                    activity("p-hold", "queued", chat_id=42, project_id=PROJECT["id"],
                             project_admission_hold={"label": "Waiting for Project access"})])
    page = open_app(room_ui, transport, theme=theme, viewport={"width": width, "height": height})
    tag = f"{width}x{height}-{theme}"
    for room in ("main", "project"):
        if room == "project":
            open_project(page, PROJECT)
            page.locator("#panel-pchat-rw-proj .chat-messages, #pchat-rw-proj-messages").first.evaluate(
                "el => { el.scrollTop = Math.round(el.scrollHeight * 0.5); }")
            # Wait for the panel's opening transition before judging its settled material.
            page.wait_for_function("getComputedStyle(document.querySelector('#project-panel')).opacity === '1'")
            assert_project_backing(page)
        else:
            page.locator("#chat-messages").evaluate("el => { el.scrollTop = Math.round(el.scrollHeight * 0.5); }")
        wait_facts(page, room, lambda f: f["tone"] == "live", f"the {room} live control")
        page.wait_for_timeout(250)
        closed = page.evaluate(_BAR_GEOMETRY, room)
        capture(page, f"{room}-closed-{tag}")
        trigger(page, room).click()
        wait_facts(page, room, lambda f: len(f["rows"]) == (3 if room == "main" else 2), f"the {room} rows")
        settle_unfold(page, room)
        opened = page.evaluate(_BAR_GEOMETRY, room)
        capture(page, f"{room}-open-{tag}")
        evidence(f"geometry-{room}-{tag}", {"closed": closed, "open": opened})
        moved = [(a, b) for a, b in zip(closed["neighbours"], opened["neighbours"])
                 if any(abs(a[k] - b[k]) > 0.5 for k in ("x", "y", "w", "h"))]
        assert not moved, ("opening moves no neighbouring control", moved)
        assert abs(opened["bar"]["h"] - closed["bar"]["h"]) <= 0.5, ("opening never resizes the bar", closed, opened)
        shade, bar = opened["shade"], opened["bar"]
        assert shade, opened
        assert_hangs_under_capsule(shade, opened["capsule"], f"{room} {tag}")
        assert shade["x"] >= bar["x"] - 0.5 and shade["x"] + shade["w"] <= bar["x"] + bar["w"] + 0.5, (shade, bar)
        assert shade["x"] >= 0 and shade["x"] + shade["w"] <= opened["vw"] and shade["bottom"] <= opened["vh"], opened
        assert opened["shadeAlpha"] == 1, "content never shows through the Shade"
        assert opened["reachable"], "every row target is the element hit at its centre"
        assert opened["overflow"] <= 0, opened
        if room == "project":
            assert_project_backing(page)
        page.keyboard.press("Escape") if facts(page, room)["active"]["inSurface"] else trigger(page, room).click()
        wait_facts(page, room, lambda f: f["surfaceHidden"], f"the {room} Shade closed")
    # Falsify the same consumer oracle, rather than accepting a new source spelling.
    panel = page.locator("#project-panel")
    original_style = panel.get_attribute("style")
    panel.evaluate("el => el.style.setProperty('background', 'transparent', 'important')")
    try:
        with pytest.raises(AssertionError, match="Project reading backing"):
            assert_project_backing(page)
    finally:
        panel.evaluate("(el, value) => value === null ? el.removeAttribute('style') : el.setAttribute('style', value)", original_style)
    assert_project_backing(page)
    evidence(f"console-{tag}", transport.console)
