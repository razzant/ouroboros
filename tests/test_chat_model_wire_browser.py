"""A room's model, Main's own write and a work's own model: from the page to the provider's wire and back.

The real server on its own port and data root, the real page, and a loopback provider that stands for the models
(``fixtures_mock_llm``): no paid call and no account login is possible here. Nothing of the server is answered by
the test. The provider lists its models on ``GET /v1/models``, so the real ``/api/model-catalog`` suggests them, and
it can hold a marked request, so a work stays running (and another queued behind it) while the page is read.

One row is written by the test, through the dialogue store's own writer: a task created over ``POST /api/tasks``
says nothing in its room, so a queued one has no card there. In the product a queued root's card comes from the
frame the work that scheduled it sends (``events_schedule_task``: its line and the ``cancelable`` marker); the test
appends that row for the task the server admitted, and everything about the task itself is the server's.

What is proved end to end (owner 10.11.2026: a room's model stays with its room; a pick in the Main chat writes
Main itself after a confirming sheet; a queued work's model is changed before it starts; the recent models are the
server's):

- a pick in a Project room's capsule is saved for that room, the next Send reaches the provider on that model, the
  running card says whose model it is, and a reload keeps the room's choice;
- the Main chat's capsule changes Main through the sheet, the settings document holds it, and the next work runs
  on it;
- a queued work is moved from its card before it starts (``at_start``) and then runs on the model it was given;
- the recent models are what the picks were, newest first, the same on a second page load.
"""
from __future__ import annotations

import datetime
import json
import os
import pathlib
import re
import threading

import pytest

from ouroboros import dialogue_store
from tests import fixtures_mock_llm
from tests.test_effort_range_browser import _launch
from tests.test_ui_smoke_playwright import direct_server_with_data  # noqa: F401 -- fixture

pytestmark = [pytest.mark.ui_browser]  # its server, data root and provider are its own: nothing on the host is shared

SOURCE = "openai-compatible"
MAIN, PICKED, OTHER, QUEUED = (f"{SOURCE}::mock-{name}" for name in ("main", "picked", "other", "queued"))
MODELS = ("mock-main", "mock-light", "mock-fallback", "mock-picked", "mock-other", "mock-queued")
NEXT_STEP, AT_START = "Switches at the next step; the running call finishes.", "Applies when it starts."


class Provider:
    """The loopback provider: every request it was sent, and the marked ones it holds until released.

    A held work first calls one harmless tool, so it has a row of work and with it a card (a turn that has done
    nothing yet is a bare block without a card's bottom line); its next call waits for the release."""

    def __init__(self) -> None:
        self.received: list = []
        self.lock = threading.Lock()
        self.gates: dict[str, threading.Event] = {}
        self.reached: dict[str, threading.Event] = {}
        self.calls: dict[str, int] = {}

    def hold(self, marker: str) -> None:
        self.gates[marker], self.reached[marker] = threading.Event(), threading.Event()

    def held(self, marker: str, timeout: float = 120) -> bool:
        """True once the marked work's call is waiting here: it has its row of work and it runs."""
        return self.reached[marker].wait(timeout)

    def release(self, marker: str) -> None:
        self.gates[marker].set()

    def release_all(self) -> None:
        for gate in self.gates.values():
            gate.set()

    @staticmethod
    def marker_of(body: dict) -> str:
        """The request's own work: its newest marker (a room's later work carries the earlier ones in its context).
        The production wire may carry text blocks and host framing, not a bare prompt string."""
        markers = re.findall(r"\bROOM_WIRE_[A-Z_]+\b", json.dumps(body.get("messages", [])))
        return markers[-1] if markers else ""

    def models_of(self, marker: str) -> list:
        with self.lock:
            return [body["model"] for body in self.received if body["model"] != "mock-light" and self.marker_of(body) == marker]

    def get(self, handler) -> None:
        if handler.path.split("?")[0] != "/v1/models":
            handler.send_error(404)
            return
        self._send(handler, 200, {"object": "list", "data": [{"id": name, "object": "model"} for name in MODELS]})

    def post(self, handler) -> None:
        body = json.loads(handler.rfile.read(int(handler.headers["Content-Length"])))
        with self.lock:
            self.received.append(body)
        marker = self.marker_of(body)
        gate = self.gates.get(marker)
        message, finish = {"role": "assistant", "content": f"WIRE_OK {marker}"}, "stop"
        if gate is not None and body.get("tools"):
            with self.lock:
                ordinal = self.calls[marker] = self.calls.get(marker, 0) + 1
            if ordinal == 1:
                message, finish = {"role": "assistant", "content": "", "tool_calls": [{
                    "id": "call_inventory", "type": "function", "function": {"name": "list_available_tools", "arguments": "{}"}}]}, "tool_calls"
            else:
                self.reached[marker].set()
                gate.wait(240)
        streaming = body.get("stream") is True
        if streaming and message.get("tool_calls"):
            message = {**message, "tool_calls": [{**call, "index": index} for index, call in enumerate(message["tool_calls"])]}
        answer = {"id": "mock-chat", "object": "chat.completion.chunk" if streaming else "chat.completion", "model": body["model"],
                  "choices": [{"index": 0, "finish_reason": finish, ("delta" if streaming else "message"): message}],
                  "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
        if streaming:
            self._write(handler, 200, "text/event-stream", f"data: {json.dumps(answer)}\n\ndata: [DONE]\n\n".encode())
        else:
            self._send(handler, 200, answer)

    def _send(self, handler, status: int, payload: dict) -> None:
        self._write(handler, status, "application/json", json.dumps(payload).encode())

    @staticmethod
    def _write(handler, status: int, content_type: str, data: bytes) -> None:
        try:
            handler.send_response(status)
            handler.send_header("Content-Type", content_type)
            handler.send_header("Content-Length", str(len(data)))
            handler.end_headers()
            handler.wfile.write(data)
        except OSError:
            pass  # the server gave up on a held call (a stopped work): nothing waits for this answer


_CAPSULE = """(sel) => {
    const el = document.querySelector(sel);
    const text = (part) => el.querySelector(`.chat-capsule-${part}`)?.textContent || '';
    return { known: el.dataset.known, own: el.dataset.own, name: text('name'), value: text('value'),
        strip: [...el.querySelectorAll('.chat-model-strip .chat-seg')].map((seg) => seg.textContent.trim()),
        checked: [...el.querySelectorAll('.chat-model-strip .chat-seg[aria-checked="true"]')].map((seg) => seg.textContent.trim()) };
}"""

_CARD = """(sel) => {
    const text = (node) => (node ? node.innerText.replace(/\\s+/g, ' ').trim() : null);
    const card = document.querySelector(sel);
    if (!card) return null;
    const part = card.querySelector('.card-model');
    const fact = part?.querySelector('.card-model-fact');
    return { task: card.dataset.taskId, chip: text(card.querySelector('[data-live-phase]')), fact: text(fact), tag: fact?.tagName || null,
        why: text(part?.querySelector('.card-model-why')), pending: text(part?.querySelector('.card-model-pending')),
        basis: part?.dataset.basis || null, meta: text(card.querySelector('[data-live-meta]')), finished: card.dataset.finished };
}"""


def _state(page) -> dict:
    return page.evaluate("async () => (await fetch('/api/state', { cache: 'no-store' })).json()")


def _preferences(page) -> dict:
    return page.evaluate("async () => (await fetch('/api/ui/preferences', { cache: 'no-store' })).json()")


def _post(page, path: str, data: dict) -> dict:
    result = page.evaluate("""async ([path, data]) => { const r = await fetch(path, { method: 'POST',
        headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) }); return { status: r.status, body: await r.json() }; }""",
                           [path, data])
    assert result["status"] in (200, 201), result
    return result["body"]


def _idle(page) -> None:
    page.wait_for_function("""async () => { const s = await (await fetch('/api/state', { cache: 'no-store' })).json();
        return s.active_chat_activities_complete && s.active_chat_activities.length === 0; }""", timeout=90_000)


def _asked(page, provider, marker: str, timeout: float = 90) -> None:
    """Waits until the marked work has asked a model at all (a queued work starts when the worker is free)."""
    for _ in range(int(timeout * 4)):
        if provider.models_of(marker):
            return
        page.wait_for_timeout(250)
    raise AssertionError(f"{marker} never reached the provider")


def _capsule_ready(page, capsule: str) -> None:
    page.wait_for_function("(sel) => document.querySelector(sel)?.dataset.known === 'true'", arg=capsule, timeout=30_000)


def _search_pick(page, capsule: str, route: str) -> None:
    """The capsule's own way: open it, Search, and press the catalog's row for the route."""
    page.click(f"{capsule} .chat-model-head")
    page.wait_for_selector(f'{capsule}[data-open="true"] .chat-model-find', timeout=10_000)
    page.click(f"{capsule} .chat-model-find")
    row = f'.ui-popup.chat-model-list .chat-model-row[data-row="route|{route}"]'
    page.wait_for_selector(row, timeout=20_000)
    page.click(row)


def _send(page, prefix: str, marker: str) -> None:
    page.fill(f"#{prefix}-input", marker)
    page.click(f"#{prefix}-send")
    page.wait_for_function("(prefix) => document.querySelector('#' + prefix + '-input').value === ''", arg=prefix, timeout=20_000)


def _answered(page, prefix: str, marker: str) -> None:
    # Reloaded history may arrive after Send: only this request's unique reply proves completion.
    page.wait_for_function("([prefix, marker]) => document.querySelector('#' + prefix + '-messages')?.innerText.includes('WIRE_OK ' + marker)",
                           arg=[prefix, marker], timeout=90_000)


def _open_project(page, project: dict) -> str:
    page.evaluate("project => window.dispatchEvent(new CustomEvent('ouro:open-project', { detail: { project } }))", project)
    prefix = f"pchat-{project['id']}"
    page.wait_for_selector(f"#{prefix}-input", timeout=30_000)
    _capsule_ready(page, f"#{prefix}-model-mode")
    return prefix


def _announce(page, data_dir, state: dict, task_id: str, line: str) -> None:
    """The scheduled task's own row in its room (see the module's note), then the page reads its history again."""
    work = _work_of(state, task_id)
    dialogue_store.append(data_dir, "progress", {"ts": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                                 "chat_id": int(work["room"]), "task_id": task_id, "content": line, "cancelable": True})


def _close_project(page) -> None:
    """Back to the Main chat alone: an open Project panel stands over it."""
    page.click("#project-panel-close")
    page.wait_for_function("() => !document.querySelector('#project-panel-backdrop').classList.contains('open')", timeout=20_000)


def _work_of(state: dict, task_id: str) -> dict | None:
    return next((work for work in state.get("works", []) if work["episode"]["task_id"] == task_id), None)


def test_a_rooms_model_mains_write_and_a_queued_works_model_reach_the_provider(direct_server_with_data, monkeypatch, tmp_path):  # noqa: F811
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
    # Plain functions, so the handler class binds each request's handler as their argument.
    monkeypatch.setattr(fixtures_mock_llm._Handler, "do_POST", lambda handler: provider.post(handler))
    monkeypatch.setattr(fixtures_mock_llm._Handler, "do_GET", lambda handler: provider.get(handler), raising=False)
    server["start_server"]()
    evidence = pathlib.Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", str(tmp_path)))
    evidence.mkdir(parents=True, exist_ok=True)
    seen: dict = {}
    with sync_playwright() as pw:
        browser, page = _launch(pw, "webkit", viewport={"width": 1512, "height": 949})
        writes: list = []
        page.on("request", lambda request: writes.append((request.url.split("/api/", 1)[1], json.loads(request.post_data or "{}")))
                if request.method == "POST" and ("/api/owner/chat-model" in request.url or re.search(r"/api/tasks/[^/]+/model$", request.url)) else None)
        try:
            page.goto(server["url"], wait_until="domcontentloaded")
            _capsule_ready(page, "#chat-model-mode")
            assert _state(page)["main_route"] == {"model": MAIN, "use_local": False}
            assert _preferences(page)["model_recents"] == [], "no model was chosen yet"

            # --- 1. A Project room: its capsule follows Main until it chooses; a pick is saved for the room at once.
            alpha = _post(page, "/api/projects", {"name": "Wire alpha"})["project"]
            beta = _post(page, "/api/projects", {"name": "Wire beta"})["project"]
            prefix = _open_project(page, alpha)
            capsule = f"#{prefix}-model-mode"
            closed = page.evaluate(_CAPSULE, capsule)
            assert (closed["name"], closed["value"], closed["own"]) == ("Main", "mock-main", "false"), closed
            _search_pick(page, capsule, PICKED)
            page.wait_for_function("(sel) => document.querySelector(sel).dataset.own === 'true'", arg=capsule, timeout=20_000)
            picked = page.evaluate(_CAPSULE, capsule)
            assert picked["value"] == "mock-picked" and picked["name"] != "Main", picked
            assert writes[-1] == ("owner/chat-model", {"kind": "model", "model": PICKED, "use_local": False, "project_id": alpha["id"]}), writes
            state = _state(page)
            rooms = {row["id"]: row.get("model_choice") for row in state["projects"]}
            assert rooms[alpha["id"]] == {"kind": "model", "model": PICKED, "use_local": False}, rooms
            assert rooms[beta["id"]] == {"kind": "main"}, "a choice stays with its room"
            assert state["main_route"]["model"] == MAIN, "a room's pick never writes Main"

            # --- 2. Send: the provider is asked for the room's model; the running card says whose model it is.
            provider.hold("ROOM_WIRE_ALPHA")
            _send(page, prefix, "ROOM_WIRE_ALPHA")
            card = f"#{prefix}-messages .chat-live-card"
            assert provider.held("ROOM_WIRE_ALPHA"), "the room's work never reached its held call"
            page.wait_for_function("(sel) => document.querySelector(sel + ' .card-model-fact')?.innerText.includes('mock-picked')",
                                   arg=card, timeout=60_000)
            running = page.evaluate(_CARD, card)
            seen["running"] = running
            assert provider.models_of("ROOM_WIRE_ALPHA")[:1] == ["mock-picked"], provider.models_of("ROOM_WIRE_ALPHA")
            assert running["why"] == "this room's model" and running["basis"] == "own", running
            assert running["fact"].endswith("· mock-picked"), running
            state = _state(page)
            work = _work_of(state, running["task"])
            seen["running_work"] = work
            assert work and work["lead"] == {"model": PICKED, "use_local": False, "basis": "room"}, work
            assert {"action": "model", "target": running["task"]} in work["controls"], work["controls"]
            assert running["tag"] == "BUTTON", "the server offers the control: the fact is pressable"

            # --- 3. A queued work is moved from its card before it starts. Project work is bounded by the one worker:
            #        a held task keeps it, so the next one stays queued behind it.
            provider.hold("ROOM_WIRE_GAMMA")
            gamma = _post(page, "/api/tasks", {"description": "ROOM_WIRE_GAMMA hold the worker", "project_id": beta["id"], "title": "Gamma"})["task_id"]
            assert provider.held("ROOM_WIRE_GAMMA"), "the first task never reached its held call"
            delta = _post(page, "/api/tasks", {"description": "ROOM_WIRE_DELTA wait for the worker", "project_id": beta["id"], "title": "Delta"})["task_id"]
            page.wait_for_function("async (id) => { const s = await (await fetch('/api/state', { cache: 'no-store' })).json();"
                                   " return s.works.some((w) => w.episode.task_id === id && w.activity === 'queued'); }", arg=delta, timeout=60_000)
            queued_work = _work_of(_state(page), delta)
            seen["queued_work"] = queued_work
            assert queued_work["lead"] == {"model": MAIN, "use_local": False, "basis": "main"}, queued_work
            assert {"action": "model", "target": delta} in queued_work["controls"], queued_work["controls"]
            _announce(page, server["data_dir"], _state(page), delta, "Queued: waiting for the worker")
            beta_prefix = _open_project(page, beta)
            page.reload(wait_until="domcontentloaded")
            _capsule_ready(page, "#chat-model-mode")
            beta_prefix = _open_project(page, beta)
            beta_card = f'#{beta_prefix}-messages .chat-live-card[data-task-id="{delta}"]'
            page.wait_for_function("(sel) => Boolean(document.querySelector(sel + ' button.card-model-fact'))", arg=beta_card, timeout=60_000)
            queued = page.evaluate(_CARD, beta_card)
            seen["queued"] = queued
            assert queued["chip"] == "Queued" and queued["fact"].endswith("· mock-main") and queued["why"] == "", queued
            assert provider.models_of("ROOM_WIRE_DELTA") == [], "a queued work has asked no model yet"
            page.click(f"{beta_card} button.card-model-fact")
            page.wait_for_selector(".card-model-pop .chat-model-row", timeout=20_000)
            popover = page.evaluate("""() => ({ title: document.querySelector('.card-model-pop-title').innerText,
                acts: document.querySelectorAll('.card-model-pop .card-model-act').length,
                main: document.querySelector('.card-model-pop .chat-model-row').innerText.replace(/\\s+/g, ' ').trim(),
                rule: document.querySelector('.card-model-pop .chat-model-rule').innerText })""")
            seen["queued_popover"] = popover
            assert popover["title"] == "Model for this work · follows Main" and popover["acts"] == 0, popover
            assert popover["main"] == "mock-main Main this work follows Main" and popover["rule"] == AT_START, popover
            page.click(f'.card-model-pop .chat-model-row[data-row="route|{QUEUED}"]')
            page.wait_for_function("(words) => [...document.querySelectorAll('.toast')].some((toast) => toast.innerText.includes(words))",
                                   arg=AT_START, timeout=20_000)
            task_write = writes[-1]
            assert task_write[0] == f"tasks/{delta}/model", writes
            assert {key: value for key, value in task_write[1].items() if key != "request_id"} == {"action": "set", "model": QUEUED, "use_local": False}
            assert task_write[1]["request_id"], task_write
            page.wait_for_function("(sel) => document.querySelector(sel + ' .card-model-fact').innerText.includes('mock-queued')",
                                   arg=beta_card, timeout=20_000)
            moved = page.evaluate(_CARD, beta_card)
            assert moved["why"] == "you chose it" and not moved["pending"], moved
            moved_work = _work_of(_state(page), delta)
            assert moved_work["lead"] == {"model": QUEUED, "use_local": False, "basis": "owner"}, moved_work

            # The held works finish on their own models; the queued one starts on the model it was given.
            seen["before_release"] = [(w["name"], w["activity"], w["live"]) for w in _state(page)["works"]]
            provider.release("ROOM_WIRE_ALPHA")
            provider.release("ROOM_WIRE_GAMMA")
            _asked(page, provider, "ROOM_WIRE_DELTA")
            _idle(page)
            assert provider.models_of("ROOM_WIRE_DELTA") == ["mock-queued"], provider.models_of("ROOM_WIRE_DELTA")
            assert set(provider.models_of("ROOM_WIRE_GAMMA")) == {"mock-main"}, "a follower ran on Main"
            assert set(provider.models_of("ROOM_WIRE_ALPHA")) == {"mock-picked"}, provider.models_of("ROOM_WIRE_ALPHA")
            assert _state(page)["projects"] and {row["id"]: row.get("model_choice") for row in _state(page)["projects"]}[beta["id"]] == {"kind": "main"}, \
                "a work's own model is the work's: its room still follows Main"

            # --- 4. Reload: the room keeps its choice, its finished card names the model it ran on, the next Send too.
            page.reload(wait_until="domcontentloaded")
            _capsule_ready(page, "#chat-model-mode")
            prefix = _open_project(page, alpha)
            capsule = f"#{prefix}-model-mode"
            page.wait_for_function("(sel) => document.querySelector(sel).dataset.own === 'true'", arg=capsule, timeout=20_000)
            assert page.evaluate(_CAPSULE, capsule)["value"] == "mock-picked"
            page.wait_for_function("(sel) => document.querySelector(sel + ' [data-live-meta]')?.innerText.includes('mock-picked')",
                                   arg=f"#{prefix}-messages .chat-live-card", timeout=30_000)
            _send(page, prefix, "ROOM_WIRE_RELOADED")
            _answered(page, prefix, "ROOM_WIRE_RELOADED")
            _idle(page)
            assert provider.models_of("ROOM_WIRE_RELOADED") == ["mock-picked"]

            # --- 5. The Main chat's capsule is Main itself: a pick asks first, then writes Main; the next work runs on it.
            _close_project(page)
            _search_pick(page, "#chat-model-mode", OTHER)
            page.wait_for_selector(".confirm-dialog [data-main-followers]", timeout=20_000)
            sheet = page.evaluate("""() => { const d = document.querySelector('.confirm-dialog'); const text = (n) => n.innerText.replace(/\\s+/g, ' ').trim();
                return { title: text(d.querySelector('h3')), row: text(d.querySelector('.chat-model-sheet-row')), count: text(d.querySelector('[data-main-followers]')),
                    rooms: [...d.querySelectorAll('.chat-model-off')].map((row) => [text(row.querySelector('.chat-model-off-name')), row.querySelector('.ui-checkbox')?.checked]),
                    ok: text(d.querySelector('[data-confirm-ok]')) }; }""")
            seen["sheet"] = sheet
            assert sheet["title"] == "Change Main" and "mock-other" in sheet["row"] and "was mock-main" in sheet["row"], sheet
            assert sheet["count"] == "No running work follows Main now.", sheet
            assert sheet["rooms"] == [["Wire alpha", False]] and sheet["ok"] == "Change Main", "the room with its own model is listed, unticked"
            assert json.loads(settings_path.read_text(encoding="utf-8"))["OUROBOROS_MODEL"] == MAIN, "nothing is written before he confirms"
            page.click(".confirm-dialog [data-confirm-ok]")
            page.wait_for_function("() => document.querySelector('#chat-model-mode .chat-capsule-value').textContent === 'mock-other'", timeout=20_000)
            assert writes[-1] == ("owner/chat-model", {"kind": "model", "model": OTHER, "use_local": False}), writes
            assert _state(page)["main_route"] == {"model": OTHER, "use_local": False}
            saved = json.loads(settings_path.read_text(encoding="utf-8"))
            assert saved["OUROBOROS_MODEL"] == OTHER, saved["OUROBOROS_MODEL"]
            assert (saved["OUROBOROS_MODEL_LIGHT"], saved["OUROBOROS_MODEL_FALLBACKS"]) == \
                (settings["OUROBOROS_MODEL_LIGHT"], settings["OUROBOROS_MODEL_FALLBACKS"]), "only Main was written"
            _send(page, "chat", "ROOM_WIRE_MAIN")
            _answered(page, "chat", "ROOM_WIRE_MAIN")
            _idle(page)
            assert provider.models_of("ROOM_WIRE_MAIN") == ["mock-other"], provider.models_of("ROOM_WIRE_MAIN")
            # A follower follows the new Main; the room with its own model does not.
            beta_prefix = _open_project(page, beta)
            _send(page, beta_prefix, "ROOM_WIRE_FOLLOWER")
            _answered(page, beta_prefix, "ROOM_WIRE_FOLLOWER")
            _idle(page)
            assert provider.models_of("ROOM_WIRE_FOLLOWER") == ["mock-other"]

            # --- 6. The recent models are the server's: what was picked, newest first, the same on a second page load.
            recents = _preferences(page)["model_recents"]
            seen["recents"] = recents
            assert recents == [{"model": OTHER, "use_local": False}, {"model": QUEUED, "use_local": False},
                               {"model": PICKED, "use_local": False}], recents
            _close_project(page)
            for _ in range(2):
                page.reload(wait_until="domcontentloaded")
                _capsule_ready(page, "#chat-model-mode")
                page.click("#chat-model-mode .chat-model-head")
                page.wait_for_function("() => document.querySelectorAll('#chat-model-mode .chat-model-strip .chat-seg').length >= 4", timeout=20_000)
                strip = page.evaluate(_CAPSULE, "#chat-model-mode")
                assert strip["strip"] == ["mock-other", "mock-queued", "mock-picked", "Search"], strip
                assert strip["checked"] == ["mock-other"], strip
                assert _preferences(page)["model_recents"] == recents
        finally:
            provider.release_all()
            (evidence / "wire-seen.json").write_text(json.dumps(seen, indent=2, default=str), encoding="utf-8")
            (evidence / "wire-models.json").write_text(json.dumps(
                [{"model": body.get("model"), "marker": provider.marker_of(body)} for body in provider.received]), encoding="utf-8")
            (evidence / "wire-writes.json").write_text(json.dumps(writes, indent=2), encoding="utf-8")
            for name in ("events.jsonl", "supervisor.jsonl"):
                log = server["data_dir"] / "logs" / name
                if log.exists():
                    (evidence / f"wire-{name}").write_text("\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-400:]), encoding="utf-8")
            try:
                (evidence / "wire-state.json").write_text(json.dumps(_state(page), indent=1, default=str), encoding="utf-8")
                (evidence / "wire-page.txt").write_text(page.locator("body").inner_text(), encoding="utf-8")
                page.screenshot(path=str(evidence / "wire-last.png"))
            except Exception:  # noqa: BLE001 - diagnostics only: the page may be gone
                pass
            browser.close()
