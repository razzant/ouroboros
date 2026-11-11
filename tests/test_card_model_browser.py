"""A work's model on its card, the fallback in amber and the Main sheet's lists, on the real page.

Owner, 10.11.2026: the model stands first in a work card's bottom line and is pressable; a work that went to a
fallback by itself is amber until that is resolved; before Main changes, the sheet says how many running works
follow it and lists what is not on Main, each with its own tick and nothing ticked beforehand. 11.11.2026: the amber
line names where the fallback came from and gives three exits.

The real page, with ``/api/state``, the chat history, the catalog and the two writes answered by the test in the
published shapes (CLIENT_CONTRACT §5 and §9); the work rows are the server's own projection of the census rows,
with the published ``lead`` and the ``model`` control added. Both ends run together in
``test_chat_model_wire_browser.py``; this test walks the states one server run cannot hold at once, at the owner's
two pane widths and on a phone, in both themes, and asserts that nothing is wider than the screen and that the
transcript does not move when the popover opens.
"""
from __future__ import annotations

import copy
import datetime
import json
import os
import pathlib
import re

import pytest

from tests._work_projection_shared import with_work_fields
from tests.test_effort_range_browser import _launch
from tests.test_ui_smoke_playwright import direct_server_with_data  # noqa: F401 - pytest fixture import

pytestmark = [pytest.mark.ui_browser]

ASTRA, GLM, OPUS = "claudexor::codex=gpt-6-astra", "z-ai/glm-5.3", "anthropic::claude-opus-5.5"
KIMI = "moonshotai/kimi-k3-thinking-preview"  # a long name, as real ones are
COGNITION = {"context_mode": "max", "subagents_mode": "auto", "effort_range": {"min": "low", "recommended": "high", "max": "max"}}
NEXT_STEP, AT_START = "Switches at the next step; the running call finishes.", "Applies when it starts."
# (viewport tag, the transcript width it is sized to or None for a phone's own width, phone)
LAYOUTS = [("desk581", 581, False), ("desk282", 282, False), ("phone390", None, True)]


def _route(model: str) -> dict:
    return {"model": model, "use_local": False}


def _lead(model: str, basis: str, **extra) -> dict:
    return {**_route(model), "basis": basis, **extra}


class Stand:
    """The server's side of the look: Main, each work's published fact, the rooms, and every write it was sent."""

    TASKS = [  # task id, name, census phase, whether the server offers its `model` control
        ("t-follow", "Compare the two parsers", "working", True),
        ("t-own", "Draft the release letter", "working", True),
        ("t-queued", "Sort the inbox", "queued", True),
        ("t-fallback", "Read last night's logs", "working", True),
        ("t-plain", "Summarise the week", "working", False),
        ("t-old", "An older record", "working", False),
    ]

    def __init__(self) -> None:
        # The server dates a departure as an ISO instant in UTC; the page says it in the viewer's clock.
        since = datetime.datetime.now().replace(hour=2, minute=14, second=0, microsecond=0).astimezone(datetime.timezone.utc).isoformat()
        self.main = _route(ASTRA)
        self.leads = {
            "t-follow": _lead(ASTRA, "main"),
            "t-own": _lead(KIMI, "owner"),
            "t-queued": _lead(ASTRA, "main"),
            "t-fallback": _lead(OPUS, "fallback", role="fallback:0", primary=_lead(ASTRA, "main"), since=since,
                                cause="subscription_window_exhausted"),
            "t-plain": _lead(GLM, "mind", since=since, cause="switch_model"),
        }  # t-old publishes no fact: its card keeps today's line
        self.room = {"id": "p7", "name": "Research", "chat_id": 7, "pinned": False, "lifecycle": "active",
                     "cognition": COGNITION, "model_choice": {"kind": "model", **_route(KIMI)}}
        self.writes: list = []
        rows = [{"activity_id": task, "kind": "managed_task", "chat_id": 1, "phase": phase, "task_name": name}
                for task, name, phase, _ in self.TASKS]
        self.census = with_work_fields({"active_chat_activities": rows, "active_chat_activities_complete": True})
        self.history = [{"task_id": task, "chat_id": 1, "role": "assistant", "is_progress": True, "narration": True,
                         "content": f"{name}: reading what is there.", "text": f"{name}: reading what is there.",
                         "suggested_name": name, "model": "gpt-6-astra", "ts": "2026-11-11T02:00:00Z"}
                        for task, name, _, _ in self.TASKS]

    def state(self, real: dict) -> dict:
        body = {**real, **copy.deepcopy(self.census), "supervisor_ready": True, "main_route": self.main,
                "projects": [*real.get("projects", []), self.room]}
        offered = {task for task, _, _, control in self.TASKS if control}
        names = {task: name for task, name, _, _ in self.TASKS}
        for work in body["works"]:
            task = work["episode"]["task_id"]
            work["name"] = names[task]  # the server names a work from its stored record; this stand has none
            if task in self.leads:
                work["lead"] = self.leads[task]
            if task in offered:
                work["controls"] = [*work["controls"], {"action": "model", "target": task}]
        return body

    def queued(self, task: str) -> bool:
        return next(phase for name, _, phase, _ in self.TASKS if name == task) == "queued"

    def task_model(self, task: str, body: dict) -> dict:
        """CLIENT_CONTRACT §5: one act on a work's model; the answer says when it applies and carries the work's fact."""
        assert body.get("request_id"), body
        act = {key: value for key, value in body.items() if key != "request_id"}
        self.writes.append(("task", task, act))
        now, queued = self.leads[task], self.queued(task)
        acting = _route(now["model"])
        if act["action"] == "set":
            wanted = _route(act["model"])
            lead = _lead(wanted["model"], "owner") if queued else _lead(acting["model"], "owner", pending=wanted)
        elif act["action"] == "follow":
            lead = _lead(self.main["model"], "main") if queued or acting == self.main else _lead(acting["model"], "main", pending=self.main)
        elif act["action"] == "keep":
            lead = _lead(acting["model"], "owner")
        else:  # return, wait: back to what it left
            primary = now["primary"]
            lead = _lead(acting["model"], primary["basis"], pending=_route(primary["model"]))
        self.leads[task] = lead
        return {"ok": True, "task_id": task, "applied": "at_start" if queued else "next_step", "lead": lead}

    def chat_model(self, body: dict) -> dict:
        """CLIENT_CONTRACT §9: a room's own model, or Main itself; a running follower then has a pending route."""
        if body.get("project_id"):
            self.writes.append(("room", body["project_id"], {key: value for key, value in body.items() if key != "project_id"}))
            previous, self.room["model_choice"] = self.room["model_choice"], {"kind": "main"}
            return {"ok": True, "project_id": body["project_id"], "model_choice": {"kind": "main"}, "previous_model_choice": previous}
        self.writes.append(("main", "", body))
        previous, self.main = self.main, _route(body["model"])
        for task, lead in self.leads.items():
            if lead["basis"] == "main" and not self.queued(task):
                acting = _route(lead["model"])
                self.leads[task] = _lead(acting["model"], "main", **({} if acting == self.main else {"pending": self.main}))
            elif lead["basis"] == "main":
                self.leads[task] = _lead(self.main["model"], "main")
        return {"ok": True, "project_id": "", "main_route": self.main, "previous_main_route": previous, "account": "kept", "window_reset": False}


def _catalog() -> dict:
    codex = lambda model: {"provider_id": "claudexor", "provider": "Codex", "source_id": "codex", "id": model, "name": model,  # noqa: E731
                           "value": f"claudexor::codex={model}", "credential_profile_id": "home", "availability": "available"}
    api = lambda provider, label, model, value: {"provider_id": provider, "provider": label, "id": model, "name": model, "value": value}  # noqa: E731
    return {"model_sources": [{"id": "codex", "label": "Codex", "credentialHarness": "codex"}], "errors": [],
            "items": [codex("gpt-6-astra"), codex("gpt-6.1-sol"), api("openrouter", "OpenRouter", GLM, GLM),
                      api("openrouter", "OpenRouter", KIMI, KIMI), api("anthropic", "Anthropic", "claude-opus-5.5", OPUS)]}


def _serve(page, stand: Stand) -> None:
    def state(route):
        route.fulfill(json=stand.state(route.fetch().json()))

    def history(route):
        main = "chat_id=1" in route.request.url or "chat_id" not in route.request.url
        route.fulfill(json={"messages": stand.history if main else [], "window": {"complete": True}, "page_cursor": "fixture",
                            "next_cursor": None, "has_more": False, "coverage": {"v": 1, "view": "fixture", "upper": {"chat": 1}, "spans": {
                                "chat": {"chain": "fixture", "from": 0, "to": 1, "gaps": []}, "progress": None}}})

    def preferences(route):
        if route.request.method != "GET":
            return route.continue_()
        route.fulfill(json={**route.fetch().json(), "model_recents": [_route(GLM), _route(KIMI)]})

    def task(route):
        task_id = route.request.url.rstrip("/").split("/")[-2]
        route.fulfill(json=stand.task_model(task_id, json.loads(route.request.post_data or "{}")))

    page.route(re.compile(r"/api/state(\?.*)?$"), state)  # the first read, and every later one that names what it holds
    page.route("**/api/chat/history*", history)
    page.route("**/api/ui/preferences", preferences)
    page.route("**/api/model-catalog*", lambda route: route.fulfill(json=_catalog()))
    page.route("**/api/tasks/*/model", task)
    page.route("**/api/owner/chat-model", lambda route: route.fulfill(json=stand.chat_model(json.loads(route.request.post_data or "{}"))))


_CARDS = """() => {
    const text = (node) => (node ? node.innerText.replace(/\\s+/g, ' ').trim() : null);
    const probe = document.createElement('span');
    probe.style.color = 'var(--status-warn-fg)';
    document.body.append(probe);
    const warn = getComputedStyle(probe).color;
    probe.remove();
    const cards = {};
    for (const card of document.querySelectorAll('.chat-live-card[data-task-id]')) {
        const part = card.querySelector('.card-model');
        const fact = part?.querySelector('.card-model-fact');
        const box = card.getBoundingClientRect();
        cards[card.dataset.taskId] = {
            fact: text(fact), tag: fact?.tagName || null, basis: part?.dataset.basis || null,
            why: text(part?.querySelector('.card-model-why')), pending: text(part?.querySelector('.card-model-pending')),
            amber: Boolean(fact) && getComputedStyle(fact).color === warn,
            first: Boolean(part) && card.querySelector('[data-live-meta]').firstElementChild === part,
            meta: text(card.querySelector('[data-live-meta]')),
            box: [box.left, box.top, box.width, box.height].map((value) => Math.round(value * 10) / 10),
            over: [...card.querySelectorAll('[data-live-meta], [data-live-meta] *')]
                .filter((node) => node.getClientRects().length && node.getBoundingClientRect().right > box.right + 0.5).map((node) => node.className),
        };
    }
    return { cards, wide: document.documentElement.scrollWidth - window.innerWidth,
        transcript: document.querySelector('#chat-messages').clientWidth };
}"""

_POPOVER = """() => {
    const text = (node) => (node ? node.innerText.replace(/\\s+/g, ' ').trim() : null);
    const popup = document.querySelector('.card-model-pop');
    if (!popup) return null;
    const box = popup.getBoundingClientRect();
    return {
        title: text(popup.querySelector('.card-model-pop-title')),
        acts: [...popup.querySelectorAll('.card-model-act')].map((node) => [node.dataset.act, text(node.querySelector('.card-model-act-label')), text(node.querySelector('.card-model-act-sub'))]),
        rows: [...popup.querySelectorAll('.chat-model-row')].map((node) => [node.dataset.row, node.getAttribute('aria-selected'), text(node)]),
        sections: [...popup.querySelectorAll('.chat-model-section')].map(text),
        rule: text(popup.querySelector('.chat-model-rule')),
        field: Boolean(popup.querySelector('input.chat-model-query')),
        inside: box.left >= 0 && box.top >= 0 && box.right <= window.innerWidth + 0.5 && box.bottom <= window.innerHeight + 0.5,
        expanded: document.querySelector('.card-model-fact[aria-expanded="true"]')?.closest('.chat-live-card')?.dataset.taskId || null,
    };
}"""

_SHEET = """() => {
    const text = (node) => (node ? node.innerText.replace(/\\s+/g, ' ').trim() : null);
    const dialog = document.querySelector('.confirm-dialog');
    if (!dialog) return null;
    const box = dialog.getBoundingClientRect();
    return {
        title: text(dialog.querySelector('h3')),
        count: text(dialog.querySelector('[data-main-followers]')),
        heads: [...dialog.querySelectorAll('.chat-model-sheet-sub')].map(text),
        rows: [...dialog.querySelectorAll('.chat-model-off')].map((row) => ({
            name: text(row.querySelector('.chat-model-off-name')), meta: text(row.querySelector('.chat-model-off-meta')),
            tick: row.querySelector('.ui-checkbox') ? row.querySelector('.ui-checkbox').checked : null,
            move: row.querySelector('.ui-checkbox')?.dataset.move || null,
            amber: text(row.querySelector('.model-fallback')), result: row.dataset.result || null,
        })),
        ok: text(dialog.querySelector('[data-confirm-ok]')),
        body: text(dialog.querySelector('.marketplace-modal-body > p')),
        inside: box.left >= 0 && box.right <= window.innerWidth + 0.5,
        wide: document.documentElement.scrollWidth - window.innerWidth,
    };
}"""


def _fact(task: str) -> str:
    return f'.chat-live-card[data-task-id="{task}"] .card-model-fact'


def _show(page, task: str) -> None:
    """The card in the middle of the transcript, clear of the room's bar and the composer."""
    page.evaluate("(task) => document.querySelector(`.chat-live-card[data-task-id=\"${task}\"]`).scrollIntoView({ block: 'center' })", task)
    page.wait_for_timeout(250)


def _quiet(page) -> None:
    """No toast left over the page: a picture shows the state it names."""
    page.mouse.move(1, 1)  # a toast under the pointer does not expire
    page.wait_for_function("() => !document.querySelector('.toast')", timeout=15_000)


def _said(page, words: str) -> None:
    """A toast says the words (the popover's own rule line says them before the act; this is the answer's)."""
    page.wait_for_function("(words) => [...document.querySelectorAll('.toast')].some((toast) => toast.innerText.includes(words))",
                           arg=words, timeout=10_000)


def _open(page, url: str, tag: str, transcript: int | None) -> None:
    """The real page with its cards up; a desk viewport is then sized until the transcript is as wide as asked."""
    page.goto(url, wait_until="domcontentloaded", timeout=30_000)
    page.wait_for_selector("#chat-input", timeout=30_000)
    for _ in range(4 if transcript else 0):
        page.wait_for_timeout(200)
        now = page.evaluate("() => document.querySelector('#chat-messages').clientWidth")
        if now == transcript:
            break
        size = page.viewport_size
        page.set_viewport_size({"width": size["width"] + transcript - now, "height": size["height"]})
    page.wait_for_function("() => document.querySelectorAll('.chat-live-card .card-model').length >= 5", timeout=30_000)
    page.wait_for_timeout(400)
    if transcript:
        assert page.evaluate("() => document.querySelector('#chat-messages').clientWidth") == transcript, tag


def _walk(page, url: str, shots: pathlib.Path, tag: str, transcript: int | None) -> None:
    stand = Stand()
    page.unroute_all()
    _serve(page, stand)
    _open(page, url, tag, transcript)

    def shoot(name: str) -> None:
        for theme in ("dark", "light"):
            page.evaluate(f"() => window.ouroTheme.set('{theme}')")
            page.wait_for_function(f"() => document.documentElement.dataset.theme === '{theme}'")
            page.wait_for_timeout(150)
            page.screenshot(path=str(shots / f"{tag}-{name}-{theme}.png"), animations="disabled")
        page.evaluate("() => window.ouroTheme.set('dark')")
        page.wait_for_function("() => document.documentElement.dataset.theme === 'dark'")
        page.wait_for_timeout(300)  # inks settle (a control's colour eases): what is read next is the dark theme's

    def held(seen: dict) -> None:
        assert seen["wide"] <= 0, (tag, seen["wide"])
        assert not any(card["over"] for card in seen["cards"].values()), (tag, {k: v["over"] for k, v in seen["cards"].items()})

    # 1. The cards: a follower, a model of the owner's choice, a queued work, a fallback in amber, a fact the server
    #    offers no control for, and a work that publishes no fact.
    seen = page.evaluate(_CARDS)
    cards = seen["cards"]
    held(seen)
    assert all(cards[task]["first"] for task in stand.leads), (tag, cards)
    assert (cards["t-follow"]["fact"], cards["t-follow"]["tag"], cards["t-follow"]["why"], cards["t-follow"]["basis"]) == \
        ("Codex · gpt-6-astra", "BUTTON", "", "main"), cards["t-follow"]
    assert (cards["t-own"]["fact"], cards["t-own"]["why"], cards["t-own"]["basis"]) == \
        ("OpenRouter · kimi-k3-thinking-preview", "you chose it", "own"), cards["t-own"]
    assert cards["t-queued"]["tag"] == "BUTTON" and cards["t-queued"]["fact"] == "Codex · gpt-6-astra", cards["t-queued"]
    fell = cards["t-fallback"]
    assert fell["fact"].startswith("fallback from Settings · claude-opus-5.5 · since 02:14"), fell
    assert (fell["why"], fell["basis"], fell["amber"]) == ("left Main (gpt-6-astra)", "fallback", True), fell
    assert not any(cards[task]["amber"] for task in ("t-follow", "t-own", "t-queued", "t-plain")), "only a fallback is amber"
    assert "$" not in fell["fact"] + fell["why"], "no money on the amber line"
    assert (cards["t-plain"]["tag"], cards["t-plain"]["fact"], cards["t-plain"]["why"]) == \
        ("SPAN", "OpenRouter · glm-5.3", "Ouroboros switched to it"), "no offered control: plain text"
    assert cards["t-old"]["fact"] is None and "gpt-6-astra" in cards["t-old"]["meta"], "no fact: the line it has today"
    assert "gpt-6-astra" not in cards["t-plain"]["meta"].replace(cards["t-plain"]["fact"], ""), "the line names the model once"
    _show(page, "t-queued")
    shoot("1-cards")

    # 2. The popover of a work on its own model: Main (the one way to follow it), the recent models and Search. Nothing in the
    #    transcript moves when it opens.
    _show(page, "t-own")
    before = page.evaluate(_CARDS)["cards"]
    page.click(_fact("t-own"))
    page.wait_for_selector(".card-model-pop .chat-model-row", timeout=10_000)
    page.wait_for_timeout(350)
    pop = page.evaluate(_POPOVER)
    assert pop["title"] == "Model for this work · you chose it" and pop["expanded"] == "t-own", pop
    assert pop["acts"] == [], "no separate action: the list's Main row is the way to follow Main"
    assert pop["rows"][0][0] == f"route|{ASTRA}" and pop["rows"][0][2] == "gpt-6-astra Main this work will follow Main", pop
    assert [row[0] for row in pop["rows"][1:3]] == [f"route|{GLM}", f"route|{KIMI}"] and pop["sections"][0] == "Recent", pop
    assert pop["rows"][2][1] == "true", "the work's own model is the chosen row"
    assert pop["field"] and pop["rule"] == NEXT_STEP and pop["inside"], pop
    after = page.evaluate(_CARDS)
    assert {task: card["box"] for task, card in after["cards"].items()} == {task: card["box"] for task, card in before.items()}, \
        "the popover floats: no card moved or changed size"
    held(after)
    shoot("2-popover")
    page.click(f'.card-model-pop .chat-model-row[data-row="route|{ASTRA}"]')
    _said(page, NEXT_STEP)
    page.wait_for_function("() => !document.querySelector('.card-model-pop')")
    assert stand.writes[-1] == ("task", "t-own", {"action": "follow"}), stand.writes
    page.wait_for_function(f"""() => document.querySelector('{_fact("t-own")}').closest('.card-model').querySelector('.card-model-pending').innerText.includes('gpt-6-astra')""")
    moved = page.evaluate(_CARDS)["cards"]["t-own"]
    assert (moved["pending"], moved["why"], moved["basis"]) == ("→ gpt-6-astra at the next step", "", "main"), moved

    # 3. A queued work is changed before it starts: a pick from the list is its own model.
    _quiet(page)
    _show(page, "t-queued")
    page.click(_fact("t-queued"))
    page.wait_for_selector(".card-model-pop .chat-model-row", timeout=10_000)
    page.wait_for_timeout(350)
    pop = page.evaluate(_POPOVER)
    assert pop["acts"] == [] and pop["title"] == "Model for this work · follows Main" and pop["rule"] == AT_START, pop
    assert pop["rows"][0][1] == "true" and pop["inside"], pop
    shoot("3-queued")
    page.click(f'.card-model-pop .chat-model-row[data-row="route|{GLM}"]')
    _said(page, AT_START)
    assert stand.writes[-1] == ("task", "t-queued", {"action": "set", "model": GLM, "use_local": False}), stand.writes
    page.wait_for_function(f"""() => document.querySelector('{_fact("t-queued")}').innerText.includes('glm-5.3')""")

    # 4. A fallback: where it came from and since when on the card, why and the three exits in its popover.
    _quiet(page)
    _show(page, "t-fallback")
    page.click(_fact("t-fallback"))
    page.wait_for_selector(".card-model-pop .card-model-act", timeout=10_000)
    page.wait_for_timeout(350)
    pop = page.evaluate(_POPOVER)
    assert pop["title"].startswith("Left Main at 02:14") and pop["title"].endswith("· Waiting for quota"), \
        ("the cause is the server's code, said in the wait card's words", pop)
    assert [act[:2] for act in pop["acts"]] == [["return", "Return to Main"], ["wait", "Wait for Main"], ["keep", "Keep on claude-opus-5.5"]], pop
    assert pop["inside"], pop
    held(page.evaluate(_CARDS))
    shoot("4-fallback")
    page.keyboard.press("Escape")
    page.wait_for_function("() => !document.querySelector('.card-model-pop')")
    assert page.evaluate(_CARDS)["cards"]["t-fallback"]["amber"], "amber stays until the server's fact says otherwise"

    # 5. Main changes: the sheet's count and lists, nothing ticked; two ticks, then one act each after the Main write.
    page.click("#chat-model-mode .chat-model-head")
    page.wait_for_selector('#chat-model-mode[data-open="true"] .chat-seg', timeout=10_000)
    page.locator("#chat-model-mode .chat-seg", has_text="glm-5.3").first.click()
    page.wait_for_selector(".confirm-dialog [data-main-followers]", timeout=10_000)
    sheet = page.evaluate(_SHEET)
    assert sheet["title"] == "Change Main" and sheet["count"] == "2 running works follow Main now.", sheet
    assert sheet["heads"] == ["Not on Main now · tick to move too", "Rooms that start new work on their own model · 1"], sheet
    rows = {row["name"]: row for row in sheet["rows"]}  # the works in the server's own order, then the rooms
    assert {name: (row["move"], row["tick"]) for name, row in rows.items()} == {
        "Sort the inbox": ("work:t-queued", False), "Read last night's logs": ("work:t-fallback", False),
        "Summarise the week": (None, None), "Research": ("room:p7", False)}, sheet["rows"]
    assert sheet["rows"][-1]["name"] == "Research" and rows["Sort the inbox"]["meta"] == "OpenRouter · glm-5.3 · you chose it · Main Chat", sheet["rows"]
    assert rows["Read last night's logs"]["amber"].startswith("fallback since 02:14"), sheet["rows"]
    assert [name for name, row in rows.items() if row["amber"]] == ["Read last night's logs"], "only the fallback's word is amber"
    assert sheet["ok"] == "Change Main" and sheet["inside"] and sheet["wide"] <= 0, sheet
    page.click('.confirm-dialog [data-move="work:t-fallback"]')
    page.click('.confirm-dialog [data-move="room:p7"]')
    assert page.evaluate(_SHEET)["ok"] == "Change Main and move 2 more"
    shoot("5-main-sheet")
    sent = len(stand.writes)
    page.click(".confirm-dialog [data-confirm-ok]")
    page.wait_for_function("() => document.querySelector('.confirm-dialog h3')?.innerText === 'Main changed'", timeout=10_000)
    assert stand.writes[sent:] == [("main", "", {"kind": "model", "model": GLM, "use_local": False}),
                                   ("task", "t-fallback", {"action": "follow"}), ("room", "p7", {"kind": "main"})], stand.writes
    moves = page.evaluate(_SHEET)
    assert moves["body"] == "2 more were moved to Main." and moves["inside"], moves
    assert [(row["name"], row["meta"], row["result"], row["tick"]) for row in moves["rows"]] == [
        ("Read last night's logs", NEXT_STEP, "moved", None), ("Research", "New work here starts on Main.", "moved", None)], moves
    shoot("6-main-moved")
    page.click(".confirm-dialog [data-confirm-ok]")

    # 6. The arrow after a Main change: a follower's call in flight finishes on the model it has.
    page.wait_for_function(f"""() => document.querySelector('{_fact("t-follow")}').closest('.card-model').querySelector('.card-model-pending').innerText.includes('glm-5.3')""",
                           timeout=15_000)
    seen = page.evaluate(_CARDS)
    held(seen)
    assert (seen["cards"]["t-follow"]["fact"], seen["cards"]["t-follow"]["pending"]) == ("Codex · gpt-6-astra", "→ glm-5.3 at the next step"), seen["cards"]["t-follow"]
    _quiet(page)
    _show(page, "t-follow")
    shoot("7-pending")


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_a_works_model_on_its_card_the_fallback_and_the_main_sheet(direct_server_with_data, engine, tmp_path):  # noqa: F811
    pytest.importorskip("playwright.sync_api", reason="Playwright is not installed")
    from playwright.sync_api import sync_playwright

    shots = pathlib.Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR", str(tmp_path)))
    shots.mkdir(parents=True, exist_ok=True)
    url = direct_server_with_data["url"]
    with sync_playwright() as pw:
        for tag, transcript, phone in LAYOUTS:
            context = ({"viewport": {"width": 390, "height": 844}, "is_mobile": True, "has_touch": True} if phone
                       else {"viewport": {"width": 1000 if transcript > 400 else 360, "height": 900}})
            browser, page = _launch(pw, engine, **context)
            try:
                _walk(page, url, shots, f"{engine}-{tag}", transcript)
            finally:
                browser.close()
