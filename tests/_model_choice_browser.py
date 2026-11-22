"""Loopback model provider and room actions shared by the model browser tests."""
from __future__ import annotations

import datetime
import json
import re
import threading

from ouroboros import dialogue_store

SOURCE = "openai-compatible"
MAIN, PICKED, OTHER, QUEUED = (f"{SOURCE}::mock-{name}" for name in ("main", "picked", "other", "queued"))
MODELS = ("mock-main", "mock-light", "mock-fallback", "mock-picked", "mock-other", "mock-queued")
NEXT_STEP, AT_START = "Switches at the next step; the running call finishes.", "Applies when it starts."
# What the tier strip at the Model sheet's foot applies to (card_model.js, composer_model.js): a running work, a queued
# one, a room's new work; the tier they run on, their own or Default's (Main's included), reaches their new subagents
# without one of theirs too (owner 13.11), so a scope with a tier of its own says the same.
WORK_SCOPE = WORK_SCOPE_OWN = ("for this work from its next call and its new subagents without a tier of their own; "
                               "the running call and started subagents keep theirs")
QUEUED_SCOPE = "for this work when it starts, and its new subagents without a tier of their own"
ROOM_SCOPE = ROOM_SCOPE_OWN = "for new work here and its new subagents without a tier of their own; running work keeps its own"


class Provider:
    """The loopback provider: every request it was sent, and the marked ones it holds until released.

    A held work first calls one harmless tool, so it has a row of work and with it a card (a turn that has done
    nothing yet is a bare block without a card's bottom line); its next call waits for the release. A model in
    `broken` refuses every call for good, as a provider that does not have it would."""

    def __init__(self, models: tuple = MODELS, broken: frozenset = frozenset()) -> None:
        self.models, self.broken = tuple(models), frozenset(broken)
        self.received: list = []
        self.lock = threading.Lock()
        self.gates: dict[str, threading.Event] = {}
        self.reached: dict[str, threading.Event] = {}
        self.calls: dict[str, int] = {}
        self.steps: set[threading.Event] = set()

    def hold(self, marker: str) -> None:
        self.gates[marker], self.reached[marker] = threading.Event(), threading.Event()

    def held(self, marker: str, timeout: float = 120) -> bool:
        """True once the marked work's call is waiting here: it has its row of work and it runs."""
        return self.reached[marker].wait(timeout)

    def release(self, marker: str) -> None:
        self.gates[marker].set()

    def advance(self, marker: str) -> None:
        """Finish the held call with one tool, then hold the next ordinary call for observation."""
        with self.lock:
            previous = self.gates[marker]
            self.steps.add(previous)
            self.gates[marker], self.reached[marker] = threading.Event(), threading.Event()
        previous.set()

    @staticmethod
    def tool_reply(ordinal: int) -> dict:
        return {"role": "assistant", "content": "", "tool_calls": [{
            "id": f"call_inventory_{ordinal}", "type": "function",
            "function": {"name": "list_available_tools", "arguments": "{}"}}]}

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
        self._send(handler, 200, {"object": "list", "data": [{"id": name, "object": "model"} for name in self.models]})

    def post(self, handler) -> None:
        body = json.loads(handler.rfile.read(int(handler.headers["Content-Length"])))
        with self.lock:
            self.received.append(body)
        marker = self.marker_of(body)
        if body["model"] in self.broken:
            self._send(handler, 404, {"error": {"message": "model unavailable", "type": "invalid_request_error", "code": "model_not_found"}})
            return
        gate = self.gates.get(marker)
        message, finish = {"role": "assistant", "content": f"WIRE_OK {marker}"}, "stop"
        if gate is not None and body.get("tools"):
            with self.lock:
                ordinal = self.calls[marker] = self.calls.get(marker, 0) + 1
            if ordinal == 1:
                message, finish = self.tool_reply(ordinal), "tool_calls"
            else:
                self.reached[marker].set()
                gate.wait()
                with self.lock:
                    if gate in self.steps:
                        self.steps.remove(gate)
                        message, finish = self.tool_reply(ordinal), "tool_calls"
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
    const tier = el.querySelector('.chat-capsule-tier');
    const head = el.querySelector('.chat-model-head');
    return { known: el.dataset.known, own: el.dataset.own, name: text('name'), value: text('value'),
        tier: tier && !tier.hidden ? tier.textContent : '', dataTier: el.dataset.tier || '', open: el.dataset.sheetOpen === 'true',
        expanded: head?.getAttribute('aria-expanded') || null, focused: document.activeElement === head };
}"""

# The one open Model sheet (model_sheet.js), as its owner reads it: the head a card puts first, the list grouped under
# its sections (the first group has no heading), and the tier strip at its foot.
_SHEET = """() => {
    const sheet = document.querySelector('.ui-popup.model-sheet');
    if (!sheet) return null;
    const text = (node) => (node ? node.innerText.replace(/\\s+/g, ' ').trim() : null);
    const groups = [{ heading: '', rows: [] }];
    for (const node of sheet.querySelectorAll('.chat-model-rows > *')) {
        if (node.classList.contains('chat-model-section')) groups.push({ heading: node.textContent, rows: [] });
        else if (node.classList.contains('chat-model-row')) {
            const pin = node.querySelector('.chat-model-pin');
            groups.at(-1).rows.push({ row: node.dataset.row, id: node.querySelector('.chat-model-row-id')?.textContent || '',
                tag: node.querySelector('.chat-model-row-tag')?.textContent || '', facts: node.querySelector('.chat-model-row-facts')?.textContent || '',
                selected: node.getAttribute('aria-selected') === 'true', pinned: pin ? pin.getAttribute('aria-pressed') === 'true' : null,
                active: node.dataset.active === 'true', cell: node.dataset.activeCell ?? null });
        }
    }
    const box = sheet.getBoundingClientRect();
    const field = sheet.querySelector('.chat-model-query');
    const focused = document.activeElement;
    return {
        label: sheet.getAttribute('aria-label'), bottom: sheet.dataset.sheet === 'bottom',
        scrim: Boolean(document.querySelector('.model-sheet-scrim')),
        inside: box.left >= -0.5 && box.top >= -0.5 && box.right <= innerWidth + 0.5 && box.bottom <= innerHeight + 0.5,
        title: text(sheet.querySelector('.card-model-pop-title')),
        acts: [...sheet.querySelectorAll('.card-model-act')].map((node) => [node.dataset.act, text(node.querySelector('.card-model-act-label'))]),
        groups, headings: groups.slice(1).map((group) => group.heading),
        tiers: [...sheet.querySelectorAll('.model-tier-seg')].map((seg) => ({ tier: seg.dataset.tier, hidden: seg.hidden,
            checked: seg.getAttribute('aria-checked') === 'true', resolved: seg.dataset.resolved === 'true', stop: seg.tabIndex === 0 })),
        offered: [...sheet.querySelectorAll('.model-tier-seg')].filter((seg) => !seg.hidden && seg.getClientRects().length).map((seg) => seg.dataset.tier),
        stop: [...sheet.querySelectorAll('.model-tier-seg')].filter((seg) => seg.tabIndex === 0).map((seg) => seg.dataset.tier),
        chosen: sheet.querySelector('.model-tier-seg[aria-checked="true"]')?.dataset.tier || '',
        reset: !sheet.querySelector('.model-tier-default')?.hidden,
        scope: text(sheet.querySelector('.model-tier-scope')), means: text(sheet.querySelector('.model-tier-means')),
        focus: focused === field ? 'field' : focused?.classList?.contains('model-tier-seg') ? `tier:${focused.dataset.tier}`
            : sheet.contains(focused) ? 'sheet' : 'outside',
        descendant: field?.getAttribute('aria-activedescendant') || '',
    };
}"""


def _row(route: str) -> str:
    """The sheet's row of one model through its source (Main's own row is `[data-row="main"]`)."""
    return f'.ui-popup.model-sheet .chat-model-row[data-row="route|{route}"]'

_CARD = """(sel) => {
    const attached = (part, fact) => {
        const sep = part?.querySelector('.chat-live-meta-sep');
        if (!sep || sep.hidden) return true;
        const words = sep.parentElement.querySelector('.card-model-words');
        const range = document.createRange();
        if (words) range.selectNodeContents(words);
        const last = words ? [...range.getClientRects()].at(-1) : fact.getBoundingClientRect();
        const box = sep.getBoundingClientRect(), middle = (box.top + box.bottom) / 2;
        return Boolean(last) && middle >= last.top - 0.5 && middle <= last.bottom + 0.5;
    };
    const text = (node) => (node ? node.innerText.replace(/\\s+/g, ' ').trim() : null);
    const card = document.querySelector(sel);
    if (!card) return null;
    const part = card.querySelector('.card-model');
    const fact = part?.querySelector('.card-model-fact');
    const model = part?.querySelector('.card-model-id');
    const words = document.createRange();
    if (model) words.selectNodeContents(model);
    const modelWhole = Boolean(model) && words.getBoundingClientRect().width <= model.getBoundingClientRect().width + 0.01;
    return { modelWhole, separatorAttached: attached(part, fact), task: card.dataset.taskId, chip: text(card.querySelector('[data-live-phase]')), fact: text(fact), tag: fact?.tagName || null,
        why: text(part?.querySelector('.card-model-why .card-model-words')), pending: text(part?.querySelector('.card-model-pending .card-model-words')),
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


def _open_sheet(page, capsule: str, *, finger: bool = False) -> None:
    """A press on the capsule (never hover) opens its Model sheet, whose list the catalog's read fills."""
    (page.tap if finger else page.click)(f"{capsule} .chat-model-head")
    page.wait_for_selector(".ui-popup.model-sheet .chat-model-row", timeout=20_000)
    page.wait_for_function("() => !document.querySelector('.ui-popup.model-sheet .chat-model-note')?.innerText.includes('Reading')",
                           timeout=20_000)


def _close_sheet(page) -> None:
    """Escape: the field's own first (it clears what was typed), then the sheet's."""
    for _ in range(2):
        if not page.query_selector(".ui-popup.model-sheet"):
            break
        page.keyboard.press("Escape")
        page.wait_for_timeout(150)
    page.wait_for_function("() => !document.querySelector('.ui-popup.model-sheet') && !document.querySelector('[data-sheet-open=\"true\"]')",
                           timeout=10_000)


# Where the focus is while a Model sheet is open, and what of the page a click reached outside it (_held_focus).
_FOCUS = """() => { const sheet = document.querySelector('.ui-popup.model-sheet'), at = document.activeElement;
    const stops = sheet ? [...sheet.querySelectorAll('a[href], button, input, select, textarea, summary, [tabindex], [contenteditable="true"]')]
        .filter((node) => node.tabIndex >= 0 && !node.matches(':disabled, [aria-disabled="true"]') && !node.closest('[hidden], [inert]')
            && node.getClientRects().length && getComputedStyle(node).visibility !== 'hidden') : [];
    return { open: Boolean(sheet), modal: sheet?.getAttribute('aria-modal') || null,
        inside: Boolean(sheet) && (at === sheet || sheet.contains(at)), stop: at === sheet ? -1 : stops.indexOf(at),
        stops: stops.length, typing: Boolean(at?.matches?.('input, textarea, [contenteditable="true"]')),
        at: at === sheet ? 'sheet' : `${at?.tagName || ''}.${at?.className || ''}`, beneath: window.__beneath || [] }; }"""
_BENEATH = """() => { if (window.__beneath) return; window.__beneath = [];
    document.addEventListener('click', (event) => { const sheet = document.querySelector('.ui-popup.model-sheet');
        if (!sheet?.contains(event.target)) window.__beneath.push(event.target.className || event.target.tagName); }, true); }"""


def _held_focus(page, forward: str, backward: str, finger: bool = True, every_stop: bool = True) -> dict:
    """A bottom Model sheet is modal: opened by a finger the sheet itself has the focus (no field, so no keyboard
    rises), by a mouse or a key its field; the hardware Tab and Shift+Tab go round its own stops and back, never to the
    page under its scrim, so an Enter at any of those stops has only the sheet's control to press. Returns what was
    seen. `every_stop=False` for Safari's own Tab, which visits only the fields: it still never leaves the sheet."""
    page.evaluate(_BENEATH)
    seen = {"opened": page.evaluate(_FOCUS)}
    count = max(seen["opened"]["stops"], 2)
    for key in (forward, backward):  # walked first, judged after: a sheet that lets the focus go shows where it went
        seen[key] = []
        for _ in range(2 * count + 1):
            page.keyboard.press(key)
            seen[key].append(page.evaluate(_FOCUS))
    seen["beneath"] = page.evaluate(_FOCUS)["beneath"]
    assert seen["opened"]["open"] and seen["opened"]["modal"] == "true", seen
    if finger:
        assert seen["opened"]["at"] == "sheet" and not seen["opened"]["typing"], ("a finger's sheet holds the focus itself", seen)
    else:
        assert seen["opened"]["typing"] and seen["opened"]["inside"], ("a mouse's or a key's sheet starts in its field", seen)
    assert seen["opened"]["stops"] >= 2, ("the sheet has its field and its tier strip to reach", seen)
    for key in (forward, backward):
        walk = seen[key]
        assert all(step["open"] and step["inside"] for step in walk), (f"{key} left the modal sheet", walk)
        stops = [step["stop"] for step in walk]
        if not every_stop:
            continue
        assert sorted(set(stops)) == list(range(count)), (f"{key} reaches every stop of the sheet", stops)
        assert stops[count:2 * count] == stops[:count], (f"{key} wraps round the sheet's stops", stops)
        step = 1 if key == forward else -1
        assert all((b - a) % count == step % count for a, b in zip(stops, stops[1:])), (f"{key} goes round in order", stops)
    assert not seen["beneath"], ("nothing beneath the scrim was clicked", seen)
    return seen


def _sheet_pick(page, capsule: str, route: str) -> None:
    """The capsule's own way: a press opens its Model sheet, a press on the route's row picks it."""
    _open_sheet(page, capsule)
    page.wait_for_selector(_row(route), timeout=20_000)
    page.click(f"{_row(route)} .chat-model-cell")


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
