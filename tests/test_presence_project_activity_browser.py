"""A live inline Presence turn, seen and addressed through a real isolated host and the shipped SPA (#2566 S1).

One real isolated server (the served copy of this checkout: real ``server.py``, supervisor, gateway, loop, queue,
stores and WebSocket) with a loopback scripted model, and Chromium on the shipped SPA at the owner's desktop size
in Dark and Light. The inline author is started by a real authenticated Host ``POST /presence/turn`` and is held
inside a model round (``ModelGate``) while the owner looks, moves it with the real ``POST /api/projects/from-task``,
types into the Project's composer, and presses Stop in the room's work list. Every state is read three ways: the
server's own ``GET /api/state``, the DOM the SPA built from it, and its pixels (screenshots, read by a person; the
DOM facts only keep the run honest).

Real boundaries: Host HTTP, admission, the gate and lease, ``run_presence_turn`` and its author agent, the loop's
owner-mail drain, the owner-message router, ``move_task_to_project``, the cancel route, ``/api/state`` and the
work projection, the SPA. Substituted: the model (a loopback script that never reaches a provider), the place
(a synthetic ``fixture-chat`` connector kept Silent, so nothing is sent anywhere) and Safety/review judgment
(off, as in the Presence contact consumer). Python children refuse non-loopback sockets and the browser refuses
every URL but the host's.

Checks are collected, not raised, so a red run names every difference it saw (the run against the base this
change started from lists exactly what it fixes); the test fails when any was recorded.
"""
from __future__ import annotations

import contextlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import threading
import time
from urllib.parse import urlparse

import pytest

from ouroboros import dialogue_store, places
from tests._shared import producer_evidence_dir
from tests.system_e2e.harness import (
    ArtifactOracle,
    ModelGate,
    ScriptedStubModel,
    body_text,
    keyless_settings,
    message_text,
    wait_until,
    write_settings_file,
)
from tests.test_host_service_api import _seed_token
from tests.test_owner_wait_integration import wait_clone as clone_fixture
from tests.test_presence_contact_browser import LABEL, SKILL, TOKEN, _event, _http, _Server
from tests.test_room_work_browser import ROOM_FACTS

wait_clone = clone_fixture
pytestmark = [pytest.mark.browser]

PROJECT, OTHER = "activity-room", "activity-other"
VIEWPORT = {"width": 1148, "height": 808}
ACTOR_WORDS = "Please look into the shared status report."  # the place member's words: the author's wake
OWNER_WORDS = "Keep it to the public summary only."  # typed into the Project's composer
# The Restart confirmation's words for one live inline turn: what the restart does to it, and no promise of return.
RESTART_INLINE = "1 place conversation is mid-turn: the restart interrupts it and it does not resume by itself"
STOP_RECORDED = "Stop requested"  # beside Working while a held model round keeps a recorded Stop pending
LATER_WORDS = "Is anything still running here?"  # typed after the author ended
MIND_ONE, MIND_TWO = "Mind view one: who is working?", "Mind view two: who is working?"  # ordinary Main heads
MAIN_WORDS = "Quick question in Main."  # an ordinary direct turn, held: the positive control of every door
PROJECT_WORDS = "Quick question in the Project."  # an ordinary Project head, held: the control of the room's bar
FIRST_ANSWER, SECOND_ANSWER, ORDINARY_ANSWER = "First pass is done.", "Understood: the public summary only.", "Noted."
ORDINARY = (LATER_WORDS, MIND_ONE, MIND_TWO, MAIN_WORDS, PROJECT_WORDS)
MARKERS = (ACTOR_WORDS, OWNER_WORDS, *ORDINARY)

_SETTLE = "() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))"
_ACTIVITY_ROWS = """() => [...document.querySelectorAll('[data-activity-section="queue"] .activity-row')].map((row) => {
    const button = row.querySelector('[data-act="task-control"]'), box = row.getBoundingClientRect();
    return {id: button?.dataset.id || '', name: (row.querySelector('.activity-name')?.textContent || '').trim(),
            sub: (row.querySelector('.activity-sub')?.textContent || '').replace(/\\s+/g, ' ').trim(),
            button: (button?.textContent || '').trim(), box: {x: box.x, y: box.y, w: box.width, h: box.height}};
})"""
_ACTIVITY_EMPTY = "() => (document.querySelector('[data-activity-section=\"queue\"] .activity-empty')?.textContent || '').trim()"
_MENU = "() => [...document.querySelectorAll('body > .task-control-menu .task-control-item')].map((i) => i.textContent.trim())"
_DIALOG = """() => { const d = document.querySelector('.confirm-dialog[role="dialog"]');
    return d ? {title: d.querySelector('h3')?.textContent || '', body: d.innerText} : null; }"""
_OVERFLOW = "() => ({scroll: document.documentElement.scrollWidth, client: document.documentElement.clientWidth})"


class Problems(list):
    """Observations that must hold; none stops the walk."""

    def check(self, ok, message):
        if not ok:
            self.append(message)
        return bool(ok)


def last_user(body):
    return next((message_text(m) for m in reversed(body.get("messages") or []) if m.get("role") == "user"), "")


def kind_of(body):
    """Which actor sent this model request. An inline author's request ends with the loop's own ``[CONTEXT_FACTS] round N``
    line (an owner chat head's ends with the host clock), so its round is read from that; a head is read from the latest of
    the words only it carries (the host appends its own lines last). ``owner-head`` is a fresh turn that carries the owner's
    Project words, which the author's mailbox would otherwise have taken."""
    text = body_text(body)
    round_line = re.match(r"\[CONTEXT_FACTS\] round (\d+) of", last_user(body))
    if round_line and ACTOR_WORDS in text:
        return f"author-{round_line.group(1)}"
    seen = [(text.rfind(marker), marker) for marker in MARKERS if marker in text]
    if not seen:
        return "other"
    marker = max(seen)[1]
    return {ACTOR_WORDS: "place-head", OWNER_WORDS: "owner-head"}.get(marker, marker)


def _row(state, task_id):
    return next((row for row in state.get("active_chat_activities") or [] if row.get("activity_id") == task_id), None)


def _work(state, task_id):
    return next((work for work in state.get("works") or [] if (work.get("episode") or {}).get("task_id") == task_id), None)


def _actions(work):
    return sorted({control["action"] for control in (work or {}).get("controls") or []})


def _mail(root, task_id):
    from ouroboros.owner_mailbox import _mailbox_path

    path = _mailbox_path(root, task_id)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()] if path.exists() else []


def _snippets(text, needle, width=260, limit=3):
    out, start = [], 0
    while len(out) < limit and (at := text.find(needle, start)) >= 0:
        out.append(text[max(0, at - width):at + width])
        start = at + len(needle)
    return out


class Consumer:
    """One real isolated host, its scripted model, and one Chromium page on the shipped SPA."""

    def __init__(self, wait_clone, tmp_path, name, *, extra_gates=()):
        from ouroboros.presence_bindings import conversation_key
        from ouroboros.presence_runner import presence_chat_id, presence_turn_task_id
        from ouroboros.projects_registry import create_project

        self.root = tmp_path / "instance" / "data"
        self.root.mkdir(parents=True)
        self.evidence = producer_evidence_dir(tmp_path / "evidence", name)
        self.problems, self.facts, self.states = Problems(), {"stages": {}}, {}
        self.errors, self.blocked, self.console, self.shots = [], [], [], []
        self.bodies, self.post = {}, {}
        self.wait_clone = wait_clone
        self.first = ModelGate(lambda b: bool(b.get("tools")) and kind_of(b) == "author-1", timeout=300)
        self.second = ModelGate(lambda b: bool(b.get("tools")) and kind_of(b) == "author-2", timeout=300)
        self.main_gate = ModelGate(lambda b: bool(b.get("tools")) and kind_of(b) == MAIN_WORDS, timeout=300)
        self.project_gate = ModelGate(lambda b: bool(b.get("tools")) and kind_of(b) == PROJECT_WORDS, timeout=300)
        self.gates = [self.first, self.second, self.main_gate, self.project_gate, *extra_gates]
        self.stub = ScriptedStubModel([self._dispatch] * 60, final_answer=ORDINARY_ANSWER, gate=self._gate)
        self.tid = None
        self._create_project, self._presence_turn_task_id = create_project, presence_turn_task_id
        self._key, self._chat_id = conversation_key, presence_chat_id
        self.place_chat = presence_chat_id(conversation_key("fixture-chat", "fixture-account", "community", ""))

    def _gate(self, body):
        self.bodies.setdefault(kind_of(body), body)  # recorded as it arrives: a held round is read while it is held
        for gate in self.gates:
            gate(body)

    def _dispatch(self, body):
        kind = kind_of(body)
        if kind in ORDINARY:
            return {"final": ORDINARY_ANSWER}
        return {"final": SECOND_ANSWER if kind == "author-2" else FIRST_ANSWER}

    # ---- the stack
    def __enter__(self):
        from playwright.sync_api import sync_playwright

        self.stub.__enter__()
        self.pw = sync_playwright().start()
        settings = keyless_settings(self.stub, OUROBOROS_MAX_WORKERS=2, OUROBOROS_CONSCIOUSNESS_SAFETY_MODE="off",
                                    OUROBOROS_CONSCIOUSNESS_DAILY_USD=10, OUROBOROS_BG_WAKEUP_MIN=3600,
                                    OUROBOROS_BG_WAKEUP_MAX=7200)
        write_settings_file(self.root / "settings.json", settings)
        _seed_token(self.root, skill=SKILL, token=TOKEN, permissions=["presence"], manifest_permissions=["presence"])
        self.place = places.connect_place(self.root, connector=SKILL, provider="fixture-chat", account_id="fixture-account",
                                          conversation_id="community", label=LABEL, wakes=["mention"])
        places.edit_place(self.root, self.place.place_id, "set_state", author="owner", state="silent")
        self.project = self._create_project(self.root, PROJECT, name="Activity room")
        self.other = self._create_project(self.root, OTHER, name="Other room")
        self.server = _Server(self.wait_clone, self.root, self.root / "settings.json")
        self.oracle = ArtifactOracle(self.root)
        self.tid = self._presence_turn_task_id(self.place.place_id, "live-inline")
        self.browser = self.pw.chromium.launch()
        self.context = self.browser.new_context(viewport=VIEWPORT, color_scheme="dark", reduced_motion="reduce")
        origin = urlparse(self.server.base_url).netloc
        self.context.route("**/*", lambda request: request.fallback() if urlparse(request.request.url).netloc == origin
                           else (self.blocked.append(request.request.url), request.abort()))
        self.context.on("page", lambda page: (page.on("pageerror", lambda error: self.errors.append(str(error))),
                                              page.on("console", lambda m: self.console.append(f"{m.type}: {m.text}")
                                                      if m.type in {"error", "warning"} else None)))
        self.page = self.context.new_page()
        self.http_failures = []
        self.page.on("response", lambda r: self.http_failures.append((r.status, r.request.method, urlparse(r.url).path + "?" + urlparse(r.url).query))
                     if r.status >= 400 else None)
        try:
            self.server.start(ready_timeout=120)
            self.facts["source"] = {"head": self.wait_clone.state.head.decode().strip(), "candidate": self.wait_clone.identity,
                                    "served_candidate": self.wait_clone.checkout_identity,
                                    "attestation": self.server.attestation}
            assert _http(self.server.port, "POST", "/api/command", {"cmd": "/bg start"})["status"] == 200
            assert wait_until(lambda: self.oracle.state().get("bg_consciousness_enabled") is True, 10)
            self.page.goto(self.server.base_url, wait_until="domcontentloaded")
            self.page.locator("#chat-status").filter(has_text="Online").wait_for(timeout=60000)
            self.page.locator(f'.nav-project-row[data-project-id="{PROJECT}"]').wait_for(state="attached", timeout=30000)
        except BaseException:
            self._teardown(export=True)
            raise
        return self

    def __exit__(self, *exc):
        self._teardown(export=True)

    def _teardown(self, *, export):
        for gate in self.gates:  # nothing may keep a model round open past the run
            gate.release.set()
        try:
            if export:
                self._export()
        finally:
            try:
                with contextlib.suppress(Exception):
                    self.browser.close()
                with contextlib.suppress(Exception):
                    self.pw.stop()
            finally:
                try:
                    self.server.stop()
                    assert self.server.proc is None or self.server.proc.poll() is not None
                finally:
                    self.stub.__exit__(None, None, None)

    def _export(self):
        with contextlib.suppress(Exception):
            if not self.page.is_closed():
                self.page.screenshot(path=str(self.evidence / "last-page.png"), animations="disabled")
                (self.evidence / "last-page.html").write_text(self.page.content())
        self.facts.update(errors=self.errors, blocked_browser_requests=self.blocked, console=self.console[-80:],
                          http_failures=self.http_failures,
                          screenshots=self.shots, problems=list(self.problems), model_kinds=self.stub.kinds(),
                          model_calls=len(self.stub.calls))
        (self.evidence / "receipt.json").write_text(json.dumps(self.facts, indent=2, default=str))
        (self.evidence / "states.json").write_text(json.dumps(self.states, indent=2, default=str))
        (self.evidence / "model-calls.json").write_text(json.dumps([
            {"kind": kind, "actor": kind_of(body), "tools": bool(body.get("tools")),
             "roles": [m.get("role") for m in body.get("messages") or []][-6:], "last_user": last_user(body)[:160],
             "mentions": {"actor": ACTOR_WORDS in body_text(body), "owner": OWNER_WORDS in body_text(body)}}
            for kind, body in list(self.stub.calls)], indent=2))
        (self.evidence / "dialogue.json").write_text(json.dumps([m.row for m in dialogue_store.iter_rows(self.root, "chat")], indent=2))
        with contextlib.suppress(Exception):
            from ouroboros.task_store import task_facts

            (self.evidence / "tasks.json").write_text(json.dumps(
                {t: self.oracle.task_result(t) for t in task_facts(self.root)}, indent=2, default=str))
        for path in (self.root.parent / "network.jsonl", self.root / "logs" / "supervisor.jsonl", self.root / "logs" / "server.log"):
            if path.is_file():
                shutil.copyfile(path, self.evidence / path.name)

    # ---- reading the server
    def state(self):
        reply = _http(self.server.port, "GET", "/api/state")
        assert reply["status"] == 200, reply
        return reply["body"]

    def dialogue(self):
        return [m.row for m in dialogue_store.iter_rows(self.root, "chat")]

    # ---- driving the host
    def start_turn(self, *, event_id="live-inline"):
        payload = _event(self.place, event_id)
        payload["continuation_version"] = 2
        payload["event"]["text"] = ACTOR_WORDS

        def post():
            try:
                self.post["reply"] = _http(self.server.host_service_port, "POST", "/presence/turn", payload, host=True)
            except Exception as exc:  # recorded; the walk reads the server, not this reply
                self.post["error"] = repr(exc)

        thread = threading.Thread(target=post, daemon=True)
        thread.start()
        return thread

    # ---- driving and reading the page
    def shot(self, name, *, themes=("dark", "light")):
        """The viewport as it stands, in each theme; the page returns to Dark."""
        for theme in themes:
            self.page.emulate_media(color_scheme=theme)
            self.page.evaluate(_SETTLE)
            path = self.evidence / f"{name}-{theme}.png"
            self.page.screenshot(path=str(path), animations="disabled")
            self.shots.append(path.name)
        self.page.emulate_media(color_scheme="dark")
        self.page.evaluate(_SETTLE)
        overflow = self.page.evaluate(_OVERFLOW)
        self.problems.check(overflow["scroll"] <= overflow["client"] + 1, f"{name}: the page scrolls sideways {overflow}")

    def go(self, view):
        self.page.locator(f'[data-nav-page="{view}"]').click()
        self.page.evaluate(_SETTLE)

    def activity_rows(self, ready=lambda rows, empty: True, *, timeout=30):
        """Activity's queue section, re-read fresh (leaving and re-entering its tab makes it refresh) until ``ready``."""
        deadline, rows, empty = time.monotonic() + timeout, [], ""
        while True:
            self.go("dashboard")
            self.page.locator('[data-dashboard-tab="costs"]').click()
            self.page.locator('[data-dashboard-tab="activity"]').click()
            self.page.locator('[data-activity-section="queue"]').wait_for(timeout=30000)
            self.page.wait_for_function("() => !document.querySelector('[data-activity-section=\"queue\"] .ui-status:not([hidden]):not(:empty)')",
                                        timeout=30000)
            rows, empty = self.page.evaluate(_ACTIVITY_ROWS), self.page.evaluate(_ACTIVITY_EMPTY)
            if ready(rows, empty) or time.monotonic() > deadline:
                return rows
            self.page.wait_for_timeout(500)

    def activity_menu(self, task_id):
        """What the row's control offers: its menu's items, or, when the control is its one action (its label names
        it), that action, whose press opens the shared confirmation (closed by close_menu, confirmed by stop_pending)."""
        trigger = self.page.locator(f'[data-activity-section="queue"] [data-act="task-control"][data-id="{task_id}"]')
        trigger.click()
        self.page.locator("body > .task-control-menu, .confirm-dialog").first.wait_for(timeout=10000)
        self.page.evaluate(_SETTLE)
        return self.page.evaluate(_MENU) or [trigger.inner_text().strip()]

    def close_menu(self):
        self.page.keyboard.press("Escape")
        self.page.evaluate(_SETTLE)

    def restart_dialog(self, name):
        """The one Restart confirmation, read and dismissed with Cancel (a restart is never sent)."""
        self.go("chat")
        self.page.locator('button[data-chat-command="restart"]').click()
        self.page.locator('.confirm-dialog[role="dialog"]').wait_for(timeout=15000)
        self.page.evaluate(_SETTLE)
        reading = self.page.evaluate(_DIALOG)
        self.shot(name)
        self.page.locator('.confirm-dialog button[data-confirm-cancel]:has-text("Cancel")').click()
        self.page.locator('.confirm-dialog[role="dialog"]').wait_for(state="detached", timeout=10000)
        return reading

    def room(self, scope):
        return self.page.evaluate(ROOM_FACTS, scope)

    def open_list(self, scope):
        """Open the room's work list (a closed one holds no rows) and return its facts."""
        trigger = ("#page-chat .rw-trigger" if scope == "main" else "#project-panel .project-panel-bar .rw-trigger")
        facts = self.room(scope)
        if facts["disabled"] or facts["hidden"] or not facts["exists"]:
            return facts  # a quiet room's status is no door: it has nothing to list
        if facts["expanded"] != "true":
            try:
                self.page.locator(trigger).click(timeout=8000)
                self.page.wait_for_function("(sel) => document.querySelector(sel)?.getAttribute('aria-expanded') === 'true'",
                                            arg=trigger, timeout=5000)
            except Exception:  # a bar that offers nothing to open (or is between two words) lists nothing
                return self.room(scope)
        self.page.evaluate(_SETTLE)
        return self.room(scope)

    def close_list(self, scope):
        if self.room(scope)["expanded"] == "true":
            self.page.keyboard.press("Escape")
            self.page.evaluate(_SETTLE)

    def open_project(self, project_id):
        row = self.page.locator(f'.nav-project-row[data-project-id="{project_id}"]')
        row.wait_for(state="attached", timeout=30000)
        row.click()
        self.page.locator("#project-panel:not([hidden])").wait_for(timeout=30000)
        self.page.wait_for_function("() => { const p = document.getElementById('project-panel'); return p.classList.contains('open')"
                                    " && !p.getAnimations().some((a) => a.playState === 'running'); }", timeout=30000)
        self.page.evaluate(_SETTLE)

    def close_project(self):
        self.page.locator("#project-panel-close").click()
        self.page.evaluate(_SETTLE)

    def settle_heads(self, *, timeout=120):
        """Wait until no ordinary head (an owner chat turn) still stands in the census: the host keeps one live for a while
        after its answer, and a room's bar says so. The inline author is no head."""
        return wait_until(lambda: not [r for r in self.state().get("active_chat_activities") or []
                                       if r.get("kind") != "presence"] or None, timeout)

    def wait_row(self, scope, task_id, what, predicate=lambda row: True, timeout=30):
        """The row of ``task_id`` in the room's (opened) work list, once ``predicate`` holds; else a recorded problem."""
        deadline, row = time.monotonic() + timeout, None
        while time.monotonic() < deadline:
            row = next((r for r in self.open_list(scope)["rows"] if r["id"] == task_id), None)
            if row is not None and predicate(row):
                return row
            self.page.wait_for_timeout(200)
        self.problems.append(f"timed out waiting for {what}: {row}")
        return row

    def row_menu(self, scope, task_id):
        root = "#page-chat" if scope == "main" else "#project-panel"
        more = self.page.locator(f'{root} .rw-row[data-id="{task_id}"] .rw-more')
        more.wait_for(timeout=15000)
        more.click()
        self.page.locator("body > .task-control-menu").wait_for(timeout=10000)
        self.page.evaluate(_SETTLE)
        return self.page.evaluate(_MENU)

    def say(self, composer, words, *, feed):
        box = self.page.locator(composer)
        box.fill(words)
        box.press("Enter")
        self.page.locator(f"{feed} .chat-bubble.user").filter(has_text=words).wait_for(timeout=30000)

    def stop_requested(self, task_id):
        """Whether the supervisor durably recorded a Stop request for ``task_id`` (its own append-only log)."""
        path = self.root / "logs" / "supervisor.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines() if path.is_file() else ():
            with contextlib.suppress(ValueError):
                row = json.loads(line)
                if row.get("type") == "cancel_intent" and row.get("event") == "requested" and row.get("task_id") == task_id:
                    return True
        return False

    def wait_for_browser(self, predicate, timeout=30):
        """Bounded observation while Python's Playwright route/response callbacks keep running."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            value = predicate()
            if value:
                return value
            self.page.wait_for_timeout(50)
        return None

    def stop_recorded(self, task_id, *, timeout=20):
        return bool(self.wait_for_browser(lambda: self.stop_requested(task_id) or task_id in self.oracle.cancel_intents(), timeout))

    def stop_pending(self, task_id, *, settle=30):
        """Press ``Stop now`` in the menu or the confirmation now open; returns once the Stop is durably recorded (the held model round is the
        turn's, so the request answers only when the turn reaches its next stop check)."""
        answers = []
        self.page.on("response", lambda r: answers.append((r.status, r.url.split("/api/")[-1]))
                     if r.request.method == "POST" and "/cancel" in r.url else None)
        confirm = self.page.locator(".confirm-dialog [data-confirm-ok]")
        if confirm.count():  # the row's control was Stop now itself: its confirmation is open
            confirm.click()
        else:
            self.page.locator("body > .task-control-menu .task-control-item").filter(has_text="Stop now").click()
        self.wait_for_browser(lambda: self.stop_requested(task_id), settle)
        return answers


def _sent_to(dialogue, text, chat_id):
    return [r for r in dialogue if r.get("text") == text and r.get("chat_id") == chat_id and r.get("direction") == "out"]


def test_a_live_inline_turn_is_seen_moved_addressed_and_stopped_in_the_real_spa(wait_clone, tmp_path):
    with Consumer(wait_clone, tmp_path, "presence-project-activity-browser") as c:
        page, problems, facts, stage, tid = c.page, c.problems, c.facts, c.facts["stages"], c.tid
        project_chat = int(c.project["chat_id"])
        try:
            # --- 0. Quiet: nothing live anywhere, the Project empty.
            c.states["0-quiet"] = c.state()
            problems.check(_row(c.states["0-quiet"], tid) is None, "0: a turn is listed before any event")
            c.open_project(PROJECT)
            stage["0-project"] = {k: c.room("project")[k] for k in ("badge", "mark")}
            c.shot("0-project-quiet")
            c.close_project()
            quiet_restart = c.restart_dialog("0-restart-no-work")  # the confirmation with no work anywhere
            stage["0-restart"] = quiet_restart
            problems.check(bool(quiet_restart) and "place conversation" not in quiet_restart["body"],
                           f"0: Restart names a live place conversation with no work anywhere: {quiet_restart}")

            # --- 1. A real Host event starts the inline author; its first model round stays open.
            c.start_turn()
            assert c.first.arrived.wait(120), "1: the author never reached its first model round"
            stage["1-live"] = {"task_id": tid, "model_calls": len(c.stub.calls)}

            # --- 2. Unmoved: the census says it, the place's room is where it works, nowhere else.
            c.states["2-unmoved"] = state = c.state()
            row, work = _row(state, tid), _work(state, tid)
            problems.check(row is not None, "2: the live inline turn is missing from active_chat_activities")
            if row:
                problems.check(row.get("kind") == "presence" and row.get("phase") == "working" and row.get("cancelable") is True,
                               f"2: its census row says {row.get('kind')}/{row.get('phase')}/{row.get('cancelable')}")
                problems.check(row.get("chat_id") == c.place_chat and not row.get("project_id"),
                               f"2: it works in room {row.get('chat_id')}, project {row.get('project_id')!r}")
            problems.check(work is not None, "2: the work projection does not list it")
            problems.check(_actions(work) == ["stop"], f"2: its controls are {_actions(work)}, expected Stop alone")
            problems.check(state.get("active_chat_activities_complete") is True and state.get("works_complete") is not False,
                           "2: the census is not complete while a turn is live")
            problems.check(tid not in {str(x.get("activity_id")) for x in state.get("active_direct_turns") or []},
                           "2: the inline turn leaked into active_direct_turns")

            rows = c.activity_rows(lambda rows, empty: any(r["id"] == tid for r in rows), timeout=20)
            stage["2-activity"] = rows
            mine = [r for r in rows if r["id"] == tid]
            problems.check(len(mine) == 1, f"2: Activity lists the turn {len(mine)} times: {rows}")
            if mine:
                problems.check(mine[0]["name"] == (work or {}).get("name") and mine[0]["name"] != "Live turn",
                               f"2: Activity does not name the turn by its work: {mine[0]['name']!r}, work {(work or {}).get('name')!r}")
                menu = c.activity_menu(tid)
                stage["2-activity-menu"] = menu
                problems.check(menu == ["Stop now"], f"2: Activity's Stop… offers {menu}, expected Stop now alone")
                c.shot("2-activity-menu-unmoved")
                c.close_menu()
            c.shot("2-activity-unmoved")
            c.go("chat")
            main = c.open_list("main")
            stage["2-main"] = {"badge": main["badge"], "rows": [r["id"] for r in main["rows"]]}
            problems.check(tid not in [r["id"] for r in main["rows"]], "2: the unmoved turn is listed in Main's work list")
            c.close_list("main")
            c.shot("2-main-unmoved")
            warning = c.restart_dialog("2-restart-unmoved")
            stage["2-restart"] = warning
            stage["2-restart-identical-to-no-work"] = bool(warning) and warning["body"] == quiet_restart["body"]
            problems.check(bool(warning) and "Pause status could not be read" not in warning["body"],
                           f"2: Restart cannot read the census while the turn is live: {warning}")
            problems.check(bool(warning) and RESTART_INLINE in warning["body"] and not stage["2-restart-identical-to-no-work"],
                           f"2: Restart does not warn about the live inline turn: {warning}")
            # The owner's Pause, over the real route, is refused for a turn not moved; nothing changes for it.
            pause = _http(c.server.port, "POST", f"/api/tasks/{tid}/pause", {"request_id": "pause-live-author", "actor": "owner"})
            stage["2-pause"] = {"status": pause["status"], "body": pause["body"]}
            problems.check((pause["status"], pause["body"].get("reason_code")) == (409, "presence_pause_unsupported"),
                           f"2: the owner's Pause of an unmoved live inline author did not refuse it as unsupported: {pause}")
            state = c.state()
            problems.check((_row(state, tid) or {}).get("phase") == "working" and _actions(_work(state, tid)) == ["stop"],
                           "2: a refused Pause changed the author's census row or its controls")
            # The mind's own view (an ordinary Main head asks nothing of the author): unmoved, known to every mind
            # as live place work, never an addressable root.
            c.say("#chat-input", MIND_ONE, feed="#chat-messages")
            wait_until(lambda: MIND_ONE in c.bodies or None, 60)
            text = body_text(c.bodies.get(MIND_ONE, {}))
            stage["2-mind"] = {"snippets": _snippets(text, tid), "head_settled": bool(c.settle_heads())}
            problems.check(tid in text and "inline_presence" in text,
                           "2: a mind's context does not know the unmoved live place turn (kind inline_presence)")

            # --- 3. The owner's own move, over the real HTTP route.
            moved = _http(c.server.port, "POST", "/api/projects/from-task", {"task_id": tid, "id": PROJECT, "name": "Activity room"})
            stage["3-move"] = {"status": moved["status"], "body": {k: moved["body"].get(k) for k in ("status", "project_id", "chat_id", "reason")}}
            problems.check(moved["status"] == 200, f"3: the move answered {moved}")
            c.states["3-moved"] = state = c.state()
            row, work = _row(state, tid), _work(state, tid)
            problems.check(row is not None and (row.get("chat_id"), row.get("project_id")) == (project_chat, PROJECT),
                           f"3: after the move the census row says {row and (row.get('chat_id'), row.get('project_id'))}")
            problems.check(work is not None and work.get("room") == project_chat and _actions(work) == ["pause", "stop"],
                           f"3: after the move its work says room {work and work.get('room')}, controls {_actions(work)}")
            homes = [str(w.get("room")) for w in state.get("works") or [] if (w.get("episode") or {}).get("task_id") == tid]
            problems.check(homes == [str(project_chat)], f"3: the work is listed in rooms {homes}")
            problems.check(str(c.place_chat) not in (state.get("rooms") or {}) and str(project_chat) in (state.get("rooms") or {}),
                           f"3: the room statuses are {sorted((state.get('rooms') or {}).keys())}")
            problems.check(state.get("task_bindings", {}).get(tid, {}).get("project_id") == PROJECT,
                           "3: the binding does not name the Project")

            c.open_project(PROJECT)
            opened = c.open_list("project")
            mine = c.wait_row("project", tid, "the moved turn in the Project's work list")
            stage["3-project"] = {"badge": opened["badge"], "mark": opened["mark"], "row": mine}
            problems.check(opened["badge"] == "Working", f"3: the Project's status says {opened['badge']!r}")
            problems.check(mine is not None and mine["word"] == "Working" and mine["mark"].startswith("moving"),
                           f"3: its row says {mine and (mine['word'], mine['mark'])}")
            problems.check(len([r for r in c.room("project")["rows"] if r["id"] == tid]) == 1, "3: the Project lists the turn more than once")
            c.shot("3-project-work-list")
            if mine and mine["more"]:
                menu = c.row_menu("project", tid)
                stage["3-project-menu"] = menu
                problems.check(set(menu) == {"Pause", "Stop now"}, f"3: the Project's task actions offer {menu}, expected Pause and Stop now")
                c.shot("3-project-work-menu")
                c.close_menu()
            else:
                problems.append("3: the Project's row has no task actions control")
            c.close_list("project")
            c.shot("3-project-moved")
            # One narrow viewport: the same Project, work list opened, where the bar has the least room.
            try:
                page.set_viewport_size({"width": 390, "height": 844})
                page.evaluate(_SETTLE)
                c.shot("3-project-narrow", themes=("dark", "light"))
                narrow = c.open_list("project")
                stage["3-project-narrow"] = {"badge": narrow["badge"], "rows": [(r["name"], r["word"]) for r in narrow["rows"]]}
                c.shot("3-project-work-list-narrow", themes=("dark", "light"))
                c.close_list("project")
            except Exception as exc:  # recorded: the walk goes on at the owner's desktop size
                problems.append(f"3: the narrow view failed: {type(exc).__name__}: {str(exc)[:200]}")
            finally:
                page.set_viewport_size(VIEWPORT)
                page.evaluate(_SETTLE)
            # Its Pause is pressed in its own scenario (``test_a_moved_turn_pauses_from_its_project_...``): here it runs on.
            c.close_project()
            c.go("chat")
            main = c.open_list("main")
            problems.check(tid not in [r["id"] for r in main["rows"]], "3: the moved turn is listed in Main's work list")
            c.close_list("main")
            c.shot("3-main-moved")
            rows = c.activity_rows(lambda rows, empty: any(r["id"] == tid for r in rows), timeout=20)
            problems.check(len([r for r in rows if r["id"] == tid]) == 1, f"3: Activity lists the moved turn {[r['id'] for r in rows]}")
            c.shot("3-activity-moved")
            warning = c.restart_dialog("3-restart-moved")
            stage["3-restart"] = warning
            stage["3-restart-identical-to-no-work"] = bool(warning) and warning["body"] == quiet_restart["body"]
            problems.check(bool(warning) and "Pause status could not be read" not in warning["body"],
                           f"3: Restart cannot read the census while the moved turn is live: {warning}")
            problems.check(bool(warning) and RESTART_INLINE not in warning["body"],
                           f"3: Restart still says the moved text turn is interrupted, though a restart saves it: {warning}")
            c.open_project(OTHER)
            quiet = c.room("project")
            problems.check(not quiet["rows"] and quiet["badge"] != "Working", f"3: the other Project shows {quiet['badge']!r} {quiet['rows']}")
            c.shot("3-other-project")
            c.close_project()
            # The mind's view once moved: a listed Project work, addressable like any root.
            c.say("#chat-input", MIND_TWO, feed="#chat-messages")
            wait_until(lambda: MIND_TWO in c.bodies or None, 60)
            text = body_text(c.bodies.get(MIND_TWO, {}))
            stage["3-mind"] = {"snippets": _snippets(text, tid), "head_settled": bool(c.settle_heads())}
            problems.check(tid in text and "live place conversation turn" in text and f"project={PROJECT}" in text,
                           "3: a mind's roster does not list the moved turn as a live place conversation turn of its Project")

            # --- 4. The owner's words in the Project reach this one author's mailbox, and its loop reads them.
            c.open_project(PROJECT)
            composer = f"#pchat-{PROJECT}-input"
            page.locator(composer).fill(OWNER_WORDS)
            c.shot("4-composer-filled", themes=("dark",))
            page.locator(composer).press("Enter")
            mail = wait_until(lambda: [m for m in _mail(c.root, tid) if m.get("text") == OWNER_WORDS] or None, 30)
            stage["4-mailbox"] = [{k: m.get(k) for k in ("kind", "msg_id", "client_message_id", "text")} for m in _mail(c.root, tid)]
            problems.check(bool(mail) and [m.get("kind") for m in mail] == ["owner_text"],
                           f"4: the owner's words did not reach the author's mailbox as owner text: {_mail(c.root, tid)}")
            page.locator(f"#pchat-{PROJECT}-messages .chat-bubble.user").filter(has_text=OWNER_WORDS).wait_for(timeout=30000)
            c.shot("4-project-owner-words")
            problems.check(len([r for r in c.state().get("active_chat_activities") or [] if r.get("project_id") == PROJECT]) == 1,
                           "4: a second turn stands in the Project beside the author")
            c.first.release.set()
            problems.check(c.second.arrived.wait(60), "4: the author's loop never read the owner's words in a second round")
            problems.check(OWNER_WORDS in body_text(c.bodies.get("author-2", {})),
                           "4: the second round did not carry the owner's words")
            stage["4-model"] = {"calls": len(c.stub.calls), "owner_words": _snippets(body_text(c.bodies.get("author-2", {})), OWNER_WORDS, 160, 2)}

            # --- 5. Stop, in the Project's work list, ends this author alone.
            c.states["5-before-stop"] = c.state()
            problems.check((_row(c.states["5-before-stop"], tid) or {}).get("kind") == "presence", "5: the author is not live in its second round")
            if (_row(c.states["5-before-stop"], tid) or {}).get("kind") == "presence":
                c.open_list("project")
                problems.check(set(c.row_menu("project", tid)) == {"Pause", "Stop now"}, "5: the Project's task actions changed")
                calls = len(c.stub.calls)
                answers = c.stop_pending(tid)
                problems.check(c.stop_recorded(tid), "5: Stop was not durably recorded")
                page.wait_for_timeout(600)
                pending = c.room("project")
                stage["5-pending"] = {"badge": pending["badge"], "mark": pending["mark"],
                                      "rows": [(r["word"], r["secondary"], r["state"], r["more"]) for r in pending["rows"]]}
                # A press through a loaded card says Cancelling… at once; one that beat the card to the room (no record
                # to mark) reads the server's recorded Stop beside Working. Bare Working is the one false answer.
                problems.check(any(r["word"] == "Cancelling…" or STOP_RECORDED in r["secondary"] for r in pending["rows"]),
                               f"5: the Project's list does not say the recorded Stop while its round is held: {stage['5-pending']}")
                c.shot("5-stop-pending")
                c.second.release.set()
                ended = wait_until(lambda: _row(c.state(), tid) is None or None, 60)
                problems.check(bool(ended), "5: the census still lists the author after Stop")
                c.wait_for_browser(lambda: answers, 30)
                stage["5-cancel-answers"] = answers
                problems.check(bool(answers) and all(status in (200, 202) for status, _url in answers), f"5: Stop answered {answers}")
                result = c.oracle.task_result(tid)
                stage["5-task"] = {key: result.get(key) for key in ("status", "result", "reason_code")}
                problems.check(result.get("status") == "cancelled", f"5: the author's row says {result.get('status')!r}")
                problems.check(wait_until(lambda: len(c.stub.calls) == calls + 1 or None, 20) is not None,
                               f"5: expected exactly the held round to end, saw {len(c.stub.calls) - calls} model calls")
                page.wait_for_timeout(3000)
                problems.check(len(c.stub.calls) == calls + 1, f"5: the model was called {len(c.stub.calls) - calls - 1} more times after Stop")
                c.states["5-stopped"] = c.state()
                problems.check(_row(c.states["5-stopped"], tid) is None and not ((_work(c.states["5-stopped"], tid) or {}).get("episode") or {}).get("live"),
                               "5: the census or the work still reads live after Stop")
                settled = c.wait_row("project", tid, "the Project's row to settle after Stop", lambda r: r["settled"])
                after = c.room("project")
                stage["5-project"] = {"badge": after["badge"], "mark": after["mark"], "row": settled}
                problems.check(after["badge"] != "Working" and settled is not None and settled["word"] == "Stopped",
                               f"5: after Stop the Project says {after['badge']!r} / {settled and settled['word']!r}")
                c.close_list("project")
                c.shot("5-project-stopped")
                dialogue = c.dialogue()
                late = _sent_to(dialogue, SECOND_ANSWER, project_chat)
                stage["5-late-answer"] = {"rows_in_project": len(late), "rows_to_the_place": len(_sent_to(dialogue, SECOND_ANSWER, c.place_chat)),
                                          "out_rows_of_the_task": [(r.get("chat_id"), (r.get("text") or "")[:40]) for r in dialogue
                                                                   if r.get("task_id") == tid and r.get("direction") == "out"]}
                problems.check(not _sent_to(dialogue, SECOND_ANSWER, c.place_chat), "5: the stopped author's answer was sent to the place")
                c.shot("5-project-after-late-answer")
                # A cold reload: the Project's saved history alone gives back the author's terminal row, once, and Main none.
                page.reload(wait_until="domcontentloaded")
                page.locator("#chat-status").filter(has_text="Online").wait_for(timeout=60000)
                c.open_project(PROJECT)
                feed = f"#pchat-{PROJECT}-messages"
                page.wait_for_timeout(2500)
                cards = page.locator(feed).get_by_text("Stopped", exact=True).filter(visible=True)
                history = {chat: [m for m in _http(c.server.port, "GET", f"/api/chat/history?chat_id={chat}")["body"].get("messages") or []
                                  if m.get("task_id") == tid] for chat in (project_chat, c.place_chat, 1)}
                stage["5-reload"] = {"cards_in_project": cards.count(),
                                     "history_rows": {str(chat): [(m.get("role"), m.get("system_type"), (m.get("text") or "")[:50])
                                                                  for m in rows] for chat, rows in history.items()}}
                problems.check(cards.count() == 1, f"5: after a reload the Project shows {cards.count()} Stopped cards of the author")
                problems.check(not history[1], f"5: after a reload Main's history holds rows of the moved turn: {history[1]}")
                c.shot("5-project-reloaded")

            # --- 6. Nothing is delivered to a finished author; the owner's words take the ordinary Project path.
            before = len(_mail(c.root, tid))
            c.say(composer, LATER_WORDS, feed=f"#pchat-{PROJECT}-messages")
            handled = wait_until(lambda: LATER_WORDS in c.bodies or None, 60)
            problems.check(len(_mail(c.root, tid)) == before, "6: a finished author's mailbox took the owner's later words")
            problems.check(bool(handled), "6: the owner's later words reached no turn at all")
            c.shot("6-project-after", themes=("dark",))
            # Control of the room's bar: an ordinary Project head, held, in the same room at the same size.
            c.say(composer, PROJECT_WORDS, feed=f"#pchat-{PROJECT}-messages")
            held = c.project_gate.arrived.wait(60)
            problems.check(held, "6: the ordinary Project head never reached its model round")
            if held:
                page.wait_for_timeout(1500)  # the room's own poll reads the new head
                bar = c.room("project")
                stage["6-ordinary-project-bar"] = {"badge": bar["badge"], "mark": bar["mark"]}
                c.shot("6-ordinary-project-bar")
            c.project_gate.release.set()
            c.close_project()

            # --- 7. Observational, not a verified control: an ordinary direct turn's richer doors in the same surfaces,
            # and what its Stop does with the answer in flight. Its Stop's 404 in some runs is unattributed (no base run).
            c.go("chat")
            c.settle_heads()  # the walk's earlier heads have ended: the one live head below is the control's
            c.say("#chat-input", MAIN_WORDS, feed="#chat-messages")
            held = c.main_gate.arrived.wait(60)
            problems.check(held, "7: the ordinary direct turn never reached its model round")
            if held:
                head_ids = set()
                wait_until(lambda: head_ids.update(r["activity_id"] for r in c.state().get("active_chat_activities") or []
                                                   if r.get("kind") == "direct_chat" and r.get("chat_id") == 1) or head_ids or None, 30)
                rows = c.activity_rows(lambda rows, empty: any(r["id"] in head_ids for r in rows), timeout=20)
                direct = [r for r in rows if r["id"] in head_ids]
                stage["7-activity"] = rows
                if direct:
                    direct_id = direct[0]["id"]
                    menu = c.activity_menu(direct_id)
                    stage["7-control-menu"] = menu
                    problems.check({"Pause", "Stop now"} <= set(menu), f"7: an ordinary turn's Stop… offers {menu}")
                    c.shot("7-activity-menu-direct")
                    # A direct turn still Preparing has nothing to stop yet (its Stop answers 404, existing behaviour): the
                    # control's Stop is judged only once its census row says Working.
                    working = wait_until(lambda: any(r["activity_id"] == direct_id and r.get("phase") == "working"
                                                     for r in c.state().get("active_chat_activities") or []) or None, 30)
                    stage["7-head-working-before-stop"] = bool(working)
                    # What its Stop does is recorded, never asserted: the route answered 202/200 or 404 in different runs of
                    # this control (an ordinary direct turn, outside S1), so it can say nothing about the inline author.
                    answers = c.stop_pending(direct_id, settle=3)  # answers only once the held round has returned
                    stage["7-stop-recorded"] = c.stop_recorded(direct_id, timeout=3)
                    c.main_gate.release.set()
                    wait_until(lambda: _row(c.state(), direct_id) is None or None, 60)
                    c.wait_for_browser(lambda: answers, 10)
                    stage["7-cancel-answers"] = answers
                    page.wait_for_timeout(3000)
                    dialogue = c.dialogue()
                    own = [(r.get("chat_id"), (r.get("text") or "")[:40]) for r in dialogue if r.get("task_id") == direct_id and r.get("direction") == "out"]
                    stage["7-direct-late-answer"] = {"out_rows_of_the_task": own}

                    c.go("chat")
                    c.shot("7-main-after-direct-stop", themes=("dark",))
                else:
                    problems.append(f"7: Activity lists no ordinary direct turn: {rows}")
            c.main_gate.release.set()

            facts["host_reply"] = c.post.get("reply") or c.post.get("error")
            assert not c.blocked and not c.errors, (c.errors, c.blocked)
        finally:
            facts["problems"] = list(problems)
        network = [json.loads(line) for line in (c.root.parent / "network.jsonl").read_text().splitlines()]
        assert not [r for r in network if r["kind"] == "blocked"], network
        assert not problems, "\n".join(problems)


def test_a_moved_turn_that_finishes_leaves_its_answer_in_the_project_not_the_place(wait_clone, tmp_path):
    """The same author, left alone: its answer stands once in the Project; Main and the place gain nothing; no host work is
    left running after it (a tool-less Done turn leaves no block, DESIGN "Conversation activity block")."""
    with Consumer(wait_clone, tmp_path, "presence-project-finish-browser") as c:
        page, problems, stage, tid = c.page, c.problems, c.facts["stages"], c.tid
        project_chat = int(c.project["chat_id"])
        try:
            c.start_turn()
            assert c.first.arrived.wait(120), "the author never reached its first model round"
            moved = _http(c.server.port, "POST", "/api/projects/from-task", {"task_id": tid, "id": PROJECT, "name": "Activity room"})
            problems.check(moved["status"] == 200, f"the move answered {moved}")
            c.open_project(PROJECT)
            c.wait_row("project", tid, "the moved turn in the Project's work list")
            c.shot("1-project-working")
            c.close_list("project")
            calls = len(c.stub.calls)
            c.first.release.set()  # no owner words: this answer is the author's last
            ended = wait_until(lambda: _row(c.state(), tid) is None or None, 60)
            problems.check(bool(ended), "the census still lists the author after its answer")
            page.wait_for_timeout(8000)  # whatever else the host asks of the model after the answer is upkeep of this work
            after = [{"kind": kind, "tools": bool(body.get("tools")), "actor": kind_of(body)} for kind, body in list(c.stub.calls)[calls:]]
            stage["after-answer-model-calls"] = after
            state = c.state()
            stage["after-answer-works"] = [{"task": (w.get("episode") or {}).get("task_id"), "room": w.get("room"),
                                            "activity": w.get("activity"), "tails": w.get("tails")} for w in state.get("works") or []]
            problems.check(not [w for w in state.get("works") or [] if (w.get("episode") or {}).get("task_id") == tid],
                           f"a work still stands for the finished turn: {stage['after-answer-works']}")
            after_bar = c.room("project")
            stage["after-answer-project"] = {"badge": after_bar["badge"], "mark": after_bar["mark"], "rows": [r["id"] for r in after_bar["rows"]]}
            problems.check(after_bar["badge"] != "Working", f"the Project still says {after_bar['badge']!r} after the answer")
            feed = f"#pchat-{PROJECT}-messages"
            bubbles = page.locator(f"{feed} .chat-bubble").filter(has_text=FIRST_ANSWER)
            bubbles.first.wait_for(timeout=30000)
            c.shot("2-project-answered")
            dialogue = c.dialogue()
            stage["after-answer-rows"] = [(r.get("chat_id"), r.get("direction"), r.get("type"), (r.get("text") or "")[:60])
                                          for r in dialogue if r.get("task_id") == tid]
            problems.check(len(_sent_to(dialogue, FIRST_ANSWER, project_chat)) == 1, "the answer does not stand once in the Project")
            problems.check(not _sent_to(dialogue, FIRST_ANSWER, c.place_chat) and not _sent_to(dialogue, FIRST_ANSWER, 1),
                           "the answer was sent to the place or to Main")
            page.reload(wait_until="domcontentloaded")
            page.locator("#chat-status").filter(has_text="Online").wait_for(timeout=60000)
            c.open_project(PROJECT)
            page.wait_for_timeout(2500)
            bubbles = page.locator(f"{feed} .chat-bubble").filter(has_text=FIRST_ANSWER)
            stage["reload"] = {"answer_bubbles": bubbles.count(), "texts": page.locator(f"{feed} .chat-bubble").all_inner_texts()[-6:]}
            problems.check(bubbles.count() == 1, f"after a reload the Project shows the answer {bubbles.count()} times")
            c.shot("3-project-reloaded")
            c.close_project()
            c.go("chat")
            main = c.room("main")
            problems.check(main["badge"] == "Online" and not main["rows"], f"Main shows {main['badge']!r} {main['rows']}")
            rows = c.activity_rows(lambda rows, empty: not rows, timeout=20)
            problems.check(not [r for r in rows if r["id"] == tid], f"Activity still lists the finished turn: {rows}")
            c.shot("4-activity-after")
        finally:
            c.facts["problems"] = list(problems)
        assert not c.blocked and not c.errors, (c.errors, c.blocked)
        assert not problems, "\n".join(problems)


def test_an_unmoved_inline_turn_is_stopped_from_activity_and_leaves_no_trace_in_main_or_a_project(wait_clone, tmp_path):
    """An author still in its place's conversation has no Project room: Activity is its one door, and its Stop ends it
    (zero further model rounds) without a row, a bar or a card in Main or in any Project."""
    with Consumer(wait_clone, tmp_path, "presence-unmoved-stop-browser") as c:
        page, problems, stage, tid = c.page, c.problems, c.facts["stages"], c.tid
        try:
            c.start_turn()
            assert c.first.arrived.wait(120), "the author never reached its first model round"
            rows = c.activity_rows(lambda rows, empty: any(r["id"] == tid for r in rows), timeout=20)
            problems.check(len([r for r in rows if r["id"] == tid]) == 1, f"Activity does not list the unmoved turn once: {rows}")
            calls = len(c.stub.calls)
            if [r for r in rows if r["id"] == tid]:
                menu = c.activity_menu(tid)
                problems.check(menu == ["Stop now"], f"Activity's Stop… offers {menu}, expected Stop now alone")
                # Force the request to arrive after click() returns: a blocking Python poll starves its route.
                page.evaluate("""() => {
                    const send = window.fetch.bind(window);
                    window.fetch = (input, init) => String(input).includes('/cancel')
                        ? new Promise(resolve => setTimeout(resolve, 1000)).then(() => send(input, init))
                        : send(input, init);
                }""")
                answers = c.stop_pending(tid)
            else:  # no door in the SPA: the same route, so the rest of the walk still says what a Stop does
                answers = []
                reply = _http(c.server.port, "POST", f"/api/tasks/{tid}/cancel", {})
                answers.append((reply["status"], f"tasks/{tid}/cancel"))
                wait_until(lambda: c.stop_requested(tid) or None, 30)
            problems.check(c.stop_recorded(tid), "Stop was not durably recorded")
            page.wait_for_timeout(600)
            rows = c.activity_rows(timeout=10)
            stage["pending-activity"] = rows
            problems.check(any(r["id"] == tid and STOP_RECORDED in r["sub"] for r in rows),
                           f"Activity does not say the recorded Stop while the round is held: {rows}")
            c.shot("1-activity-stop-pending")
            c.first.release.set()
            problems.check(bool(wait_until(lambda: _row(c.state(), tid) is None or None, 60)), "the census still lists the author after Stop")
            c.wait_for_browser(lambda: answers, 30)
            stage["cancel-answers"] = answers
            problems.check(bool(answers) and all(status in (200, 202) for status, _url in answers), f"Stop answered {answers}")
            result = c.oracle.task_result(tid)
            stage["task"] = {key: result.get(key) for key in ("status", "result", "reason_code")}
            dialogue = c.dialogue()
            stage["late-answer"] = {"out_rows_of_the_task": [(r.get("chat_id"), (r.get("text") or "")[:40]) for r in dialogue
                                                              if r.get("task_id") == tid and r.get("direction") == "out"]}
            problems.check(result.get("status") == "cancelled", f"the author's row says {result.get('status')!r}")
            problems.check(wait_until(lambda: len(c.stub.calls) == calls + 1 or None, 20) is not None, "the held round did not end")
            page.wait_for_timeout(3000)
            problems.check(len(c.stub.calls) == calls + 1, f"the model was called {len(c.stub.calls) - calls - 1} more times after Stop")
            rows = c.activity_rows(lambda rows, empty: not rows, timeout=20)
            problems.check(not [r for r in rows if r["id"] == tid], f"Activity still lists the stopped turn: {rows}")
            c.shot("2-activity-after-stop")
            main = c.room("main")
            problems.check(main["badge"] == "Online" and not main["rows"], f"Main shows {main['badge']!r} {main['rows']}")
            for project in (PROJECT, OTHER):
                c.open_project(project)
                quiet = c.room("project")
                problems.check(quiet["badge"] != "Working" and not quiet["rows"], f"{project} shows {quiet['badge']!r} {quiet['rows']}")
                feed = f"#pchat-{project}-messages"
                problems.check(page.locator(feed).get_by_text("Stopped", exact=True).filter(visible=True).count() == 0,
                               f"{project} shows a Stopped card of a turn that never moved there")
                c.shot(f"3-{project}-after-stop", themes=("dark",))
                c.close_project()
            history = {chat: [m for m in _http(c.server.port, "GET", f"/api/chat/history?chat_id={chat}")["body"].get("messages") or []
                              if m.get("task_id") == tid] for chat in (1, int(c.project["chat_id"]), int(c.other["chat_id"]))}
            stage["history"] = {str(chat): len(rows) for chat, rows in history.items()}
            problems.check(not any(history.values()), f"Main or a Project holds rows of the unmoved turn: {history}")
        finally:
            c.facts["problems"] = list(problems)
        assert not c.blocked and not c.errors, (c.errors, c.blocked)
        assert not problems, "\n".join(problems)


_HOLDER = r'''
import os, sys
from pathlib import Path
from ouroboros.platform_layer import file_lock_exclusive
fd = os.open(Path(sys.argv[1]) / "author-zz-unreadable.live", os.O_CREAT | os.O_RDWR, 0o600)
file_lock_exclusive(fd)
os.write(fd, b"{not json")  # a live foreign stack whose witness this census cannot name
print("held", flush=True)
sys.stdin.read()
'''


def test_an_unreadable_author_census_is_disclosed_and_never_hides_a_local_authors_doors(wait_clone, tmp_path):
    """A held witness the census cannot read makes it incomplete: the owner is told so (census, Activity, Restart), the
    mind's context says it, and the moved author this process runs is still addressed in its Project, never replaced
    by a fresh head."""
    with Consumer(wait_clone, tmp_path, "presence-census-gap-browser") as c:
        page, problems, stage, tid = c.page, c.problems, c.facts["stages"], c.tid
        holder = None
        try:
            c.start_turn()
            assert c.first.arrived.wait(120), "the author never reached its first model round"
            moved = _http(c.server.port, "POST", "/api/projects/from-task", {"task_id": tid, "id": PROJECT, "name": "Activity room"})
            problems.check(moved["status"] == 200, f"the move answered {moved}")
            whole = c.state()
            problems.check(whole.get("active_chat_activities_complete") is True, "the census is incomplete before any gap")
            gate = c.root / "state" / "presence_turn_gate"
            problems.check(gate.is_dir(), "the author left no gate directory")
            repo = pathlib.Path(__file__).resolve().parents[1]
            holder = subprocess.Popen([sys.executable, "-c", _HOLDER, str(gate)], cwd=str(repo), stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE, text=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
            problems.check(holder.stdout.readline().strip() == "held", "the foreign witness holder did not start")
            c.states["gap"] = state = c.state()
            row, work = _row(state, tid), _work(state, tid)
            stage["gap-state"] = {"activities_complete": state.get("active_chat_activities_complete"),
                                  "works_complete": state.get("works_complete"), "works_gaps": state.get("works_gaps"),
                                  "row": row and {k: row.get(k) for k in ("kind", "phase", "chat_id", "project_id")},
                                  "controls": _actions(work), "ready": state.get("supervisor_ready")}
            problems.check(state.get("active_chat_activities_complete") is False, "an unreadable witness left the census complete")
            problems.check(row is not None and row.get("kind") == "presence", "the local author left the census with the gap")
            rows = c.activity_rows(timeout=20)
            stage["gap-activity"] = {"rows": rows, "empty": page.evaluate(_ACTIVITY_EMPTY)}
            c.shot("1-activity-gap")
            warning = c.restart_dialog("2-restart-gap")
            stage["gap-restart"] = warning
            problems.check("could not be read" in (warning or {}).get("body", ""), f"Restart does not say the census is unread: {warning}")
            # The turn it still sees through the gap is moved as text: saved work a restart returns (#2566 S2), as in
            # the walk above, so the gap never makes Restart call it interrupted.
            problems.check(RESTART_INLINE not in (warning or {}).get("body", ""),
                           f"Restart says the moved text turn it sees through the gap is interrupted, though a restart saves it: {warning}")
            # The mind's context names the gap rather than a quiet list.
            c.say("#chat-input", MIND_ONE, feed="#chat-messages")
            wait_until(lambda: MIND_ONE in c.bodies or None, 60)
            text = body_text(c.bodies.get(MIND_ONE, {}))
            works = _snippets(text, '"live_works_present"', 0, 1)
            at = text.find('"live_works_present"')
            flag = re.search(r'"incomplete":\s*(true|false)', text[at:at + 600]) if at >= 0 else None
            stage["gap-mind"] = {"live_works_incomplete": flag and flag.group(1), "present": bool(works),
                                 "roster": _snippets(text, "live place conversation turn", 200, 1)}
            problems.check(bool(flag) and flag.group(1) == "true",
                           f"a mind's live-works view does not say it is incomplete while the author census is unreadable: {flag and flag.group(0)}")
            # The owner's words in the Project still reach this author (it is this process's), not a fresh head.
            c.open_project(PROJECT)
            heads = len(c.stub.calls)
            c.say(f"#pchat-{PROJECT}-input", OWNER_WORDS, feed=f"#pchat-{PROJECT}-messages")
            mail = wait_until(lambda: [m for m in _mail(c.root, tid) if m.get("text") == OWNER_WORDS] or None, 30)
            problems.check(bool(mail), f"during the gap the owner's words did not reach the local author: {_mail(c.root, tid)}")
            page.wait_for_timeout(1500)
            problems.check(len(c.stub.calls) == heads, "during the gap a fresh head started instead")
            c.shot("3-project-gap")
            c.close_project()
        finally:
            if holder is not None:
                with contextlib.suppress(Exception):
                    holder.stdin.close()
                try:
                    holder.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    holder.kill()
                    holder.wait(timeout=10)
                c.facts["holder_returncode"] = holder.returncode
            c.facts["problems"] = list(problems)
        problems.check(c.state().get("active_chat_activities_complete") is True, "the census stays incomplete after the witness is gone")
        assert not c.blocked and not c.errors, (c.errors, c.blocked)
        assert not problems, "\n".join(problems)
