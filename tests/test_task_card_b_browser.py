"""Ordinary root task cards in the approved B composition (owner 30.10, B), in the real shell, chat module and
stylesheets over static gateway data (no runtime, no account).

A root card's head holds its status, the existing clock (`updated`, else `started`; none once finished) and the quiet
`Reviews N` and `Details` doors; its name (16px) and latest authored line (14px) wrap whole below; its foot is one band,
the facts at its left and the host's commands at its right end, the command cluster wrapping whole under the facts on a
narrow card. What Reviews and Details open lies after the foot and before the children. Children keep their identity
row and docked Reviews door; a Main portal keeps its Project-reference row and its two-line line. Every word here is
synthetic: the quiet scene has a real feed's geometry (summaries of two and three lines, short meta lines, unknown
costs), the work scene is illustrative.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
from threading import Thread
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import pytest

from tests._work_projection_shared import with_work_fields
from tests.ui_console_errors import PageErrors

pytestmark = pytest.mark.ui_browser
WEB = Path(__file__).resolve().parents[1] / "web"
BOOT = """<script type="module">
import {createChatInstance} from '/static/modules/chat.js';
const handlers=new Map();
const ws={ws:{readyState:1},on(type,fn){const rows=handlers.get(type)||[];rows.push(fn);handlers.set(type,rows);return()=>{};},
    isConnected:()=>true,send(frame){return {status:'sent',clientMessageId:frame.client_message_id||'cm'};}};
window.emit=(type,row)=>(handlers.get(type)||[]).forEach(fn=>fn(row));
window.chat=createChatInstance({ws,state:{activePage:'chat',projectChatIds:new Set([7]),unreadCount:0},
    updateUnreadBadge(){},openSettingsTab(){},openDashboardTab(){},chatId:1,
    mountEl:document.getElementById('content'),stateSnapshots:{begin:()=>({generation:1,requestedAt:Date.now()}),
    gate(){return Promise.resolve(this.begin());},isCurrent:()=>true,
    apply(request,data){window.chat?.hydrateStateSnapshot(data);}}});
</script>"""

# A quiet feed in synthetic words with a real feed's geometry (Europe/Moscow clock). Its clock times are read on the
# day the test runs, so a clock never reads `Yesterday`.
MSK = ZoneInfo("Europe/Moscow")
DAY = datetime.now(MSK).date()


def _at(clock):
    """`HH:MM[:SS]` today in Moscow, as the host's UTC ISO stamp."""
    parts = [int(part) for part in clock.split(":")] + [0]
    local = datetime(DAY.year, DAY.month, DAY.day, parts[0], parts[1], parts[2], tzinfo=MSK)
    return local.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _epoch(clock):
    return int(datetime.fromisoformat(_at(clock).replace("Z", "+00:00")).timestamp())


Q1 = "Проверил черновик плана поездки: все даты уже согласованы с участниками, новых вопросов нет, напоминание отправлять не стал."
Q2 = ("Просмотрел итоговую таблицу расходов за квартал: бухгалтер объяснил выбранную категорию вместо отдельной статьи; "
      "новых замечаний нет, обсуждение закрыто.")
Q3 = "Сверил обновлённый список покупок: все позиции уже отмечены купленными, поэтому повторное уведомление не понадобилось."
MODEL = "codex=model-alpha"
UNKNOWN = {"cost_presentation": None}
TRACKED = {"cost_presentation": {"has_rows": True, "tracked_amount": 0.42, "has_unpriced": True, "tracked_final": True}}
WORK_TITLE = "Отчёт — сводка остатков по складу"
WORK_LINE = "Сравниваю остатки трёх складов на одной выборке; проверяю пересчёт и длинные названия."
LONG_ROLE = "Аналитик складских остатков и связанных поставок клиентов"


def _say(task_id, text, ts, **extra):
    return ("chat", {"chat_id": 1, "task_id": task_id, "role": "assistant", "is_progress": True, "narration": True,
                     "content": text, "text": text, "ts": ts, **extra})


def _log(task_id, kind, ts, **extra):
    return ("log", {"chat_id": 1, "data": {"type": kind, "task_id": task_id, "ts": ts, **extra}})


def _done(task_id, ts, status="completed", **extra):
    return _log(task_id, "task_done", ts, status=status, task_terminal_status=status,
                outcome_axes={"lifecycle": {"status": status}, "execution": {"status": "ok" if status == "completed" else "failed"}},
                **extra)


def _ended(task_id, ts, **extra):
    """A settled ending as a Project room's history keeps it."""
    return {"chat_id": 7, "task_id": task_id, "role": "system", "system_type": "task_summary", "text": "",
            "status": "failed", "task_terminal_status": "failed", "ts": ts, "history_id": f"end-{task_id}",
            "outcome_axes": {"lifecycle": {"status": "failed"}, "execution": {"status": "failed"}}, **extra}


def _child(child_id, role, event, text, ts, **extra):
    return ("chat", {"chat_id": 1, "task_id": child_id, "subagent_task_id": child_id, "parent_task_id": "w-root",
                     "root_task_id": "w-root", "delegation_role": "subagent", "subagent_event": event,
                     "subagent_role": role, "role": "assistant", "is_progress": True, "narration": True,
                     "content": text, "text": text, "ts": ts, **extra})


QUIET = [
    _say("q1", Q1, _at("22:42:20"), model=MODEL, cancelable=True),
    _log("q1", "task_metrics_event", _at("22:45:23"), tool_calls=4, duration_sec=183, **UNKNOWN),
    _say("q2", Q2, _at("22:42:30"), model=MODEL),
    _done("q2", _at("22:45:18"), tool_calls=4, duration_sec=168, **UNKNOWN),
    # Named by its admission, with a host note and no narration yet: its clock is when it started. The host offers
    # no Stop for it (it grants Stop on the first card alone), so it has none.
    _say("q3", "Context ready", _at("22:43"), narration=False, model=MODEL),
    ("task_named", {"task_id": "q3", "suggested_name": Q3}),
    _log("q3", "task_metrics_event", _at("22:45:59"), tool_calls=4, duration_sec=169, **UNKNOWN),
]
# Illustrative, as the reference's work scene: two reviews, two children and the host's two commands.
WORK = [
    _say("w-root", WORK_LINE, _at("22:47"), suggested_name=WORK_TITLE, model="model-bravo-5.5", cancelable=True),
    _log("w-root", "task_metrics_event", _at("22:48"), tool_calls=18, duration_sec=720, **TRACKED,
         review_projection={"panels": [{"panel_id": "scenario-2", "surface": "task_acceptance", "aggregate_signal": "FAIL",
                                        "reason": "Итог не совпадает с выгрузкой за прошлый месяц.", "actors": []}]}),
    ("chat", {"chat_id": 1, "role": "system", "system_type": "skill_review", "task_id": "w-root", "ts": _at("22:48:10"),
              "review_group": {"surface": "skill", "id": "task:w-root:ledger", "presentation_owner_task_id": "w-root",
                               "skill": "ledger", "status": "clean",
                               "attempts": [{"job_id": "job-ledger", "skill": "ledger", "status": "clean"}]}}),
    _child("w-kid-7c2e", LONG_ROLE, "completed", "Два отчёта готовы; подробные таблицы сохранены.",
           _at("22:47:30"), model="model-charlie-5.5", status="completed",
           result="Два отчёта готовы; подробные таблицы сохранены."),
    _child("w-kid-9b41", "Независимая проверка расчётов", "running", "Продолжение ждёт решения родителя.",
           _at("22:47:40"), model="model-delta"),
]
CENSUS = [
    {"activity_id": "q1", "chat_id": 1, "kind": "managed_task", "phase": "working", "started_at": _epoch("22:42"),
     "cancelable": True},
    {"activity_id": "w-root", "chat_id": 1, "kind": "managed_task", "phase": "working", "started_at": _epoch("22:46"),
     "cancelable": True},
]

ROOT_FACTS = """id => {
    const card = document.querySelector(`.chat-live-card[data-task-id="${id}"]:not(.subagent)`);
    if (!card) return null;
    const own = (sel) => card.querySelector(`:scope > .chat-live-summary-button ${sel}`);
    const box = (el) => { if (!el || !el.getClientRects().length) return null;
        const r = el.getBoundingClientRect(); return {l: r.left, t: r.top, r: r.right, b: r.bottom, h: r.height, w: r.width}; };
    const title = own('[data-live-title]'), line = own('[data-live-activity]'), when = own('[data-live-when]');
    const reviews = own('[data-live-review-summary]'), details = own('[data-live-disclosure]');
    const words = (el) => { if (!el) return null; const range = document.createRange(); range.selectNodeContents(el);
        const r = range.getBoundingClientRect(); return {l: r.left, r: r.right}; };
    const foot = card.querySelector(':scope > .chat-live-foot'), meta = card.querySelector(':scope > .chat-live-foot > .chat-live-meta');
    const receipt = card.querySelector(':scope > .chat-live-foot > .chat-live-swarm-receipt');
    const buttons = [...card.querySelectorAll(':scope > .chat-live-foot > .chat-live-foot-end > .chat-live-actions > *')];
    const style = (el, key) => el ? getComputedStyle(el)[key] : null;
    const inner = card.getBoundingClientRect(), pad = parseFloat(getComputedStyle(card).borderRightWidth) || 0;
    const kids = [...card.querySelectorAll(':scope > .chat-subagents > .chat-live-card.subagent')];
    const parts = [...card.children].map((el) => el.className.split(' ')[0] || el.getAttribute('data-live-reviews-host') !== null && 'reviews-host');
    return {
        chip: own('[data-live-phase]').textContent.trim(), chipBox: box(own('[data-live-phase]')),
        when: when && !when.hidden ? when.textContent : '', whenBox: box(when),
        title: title.textContent, titleBox: box(title), titleFont: style(title, 'fontSize'), titleWeight: style(title, 'fontWeight'),
        titleClamp: style(title, 'webkitLineClamp'), titleCut: title.scrollHeight > title.clientHeight + 1,
        titleLineHeight: style(title, 'lineHeight'),
        line: line.textContent, lineBox: box(line), lineFont: style(line, 'fontSize'), lineClamp: style(line, 'webkitLineClamp'),
        lineCut: line.scrollHeight > line.clientHeight + 1,
        reviews: reviews && !reviews.hidden ? reviews.textContent.trim() : '', reviewsBox: box(reviews), reviewsWords: words(reviews),
        reviewsColor: style(reviews, 'color'), detailsColor: style(own('[data-live-toggle]'), 'color'),
        reviewsFont: style(reviews, 'fontSize'), detailsFont: style(own('[data-live-toggle]'), 'fontSize'),
        detailsBox: box(details), detailsWords: words(own('[data-live-toggle]')), chevronBox: box(details?.querySelector('.chat-live-chevron')),
        reviewsInHead: Boolean(reviews && reviews.closest('.chat-live-summary-button')),
        footShown: Boolean(foot && foot.getClientRects().length), footBox: box(foot),
        footInnerBottom: foot ? foot.getBoundingClientRect().bottom - parseFloat(getComputedStyle(foot).paddingBottom) : null,
        cardInnerBottom: inner.bottom - (parseFloat(getComputedStyle(card).borderBottomWidth) || 0),
        expanded: card.dataset.expanded === '1',
        receipt: box(receipt), receiptText: receipt ? receipt.textContent : '', receiptFirst: Boolean(receipt) && foot.firstElementChild === receipt,
        footWidth: foot ? foot.clientWidth - parseFloat(getComputedStyle(foot).paddingLeft) - parseFloat(getComputedStyle(foot).paddingRight) : 0,
        meta: meta ? meta.textContent.replace(/\u00a0/g, ' ') : '', metaBox: box(meta), metaParts: [...(meta?.querySelectorAll('.chat-live-meta-part') || [])].map(box),
        buttons: buttons.map((el) => ({name: el.textContent.trim(), cls: el.className, ...box(el)})),
        card: box(card), contentRight: inner.right - pad - 20, contentLeft: inner.left + pad + 20,
        hscroll: card.scrollWidth > card.clientWidth + 1, order: parts,
        kids: kids.map((kid) => ({id: kid.dataset.taskId, role: kid.querySelector('[data-live-title]').textContent,
            box: box(kid), titleBox: box(kid.querySelector(':scope > .chat-live-summary-button [data-live-title]')),
            chipBox: box(kid.querySelector(':scope > .chat-live-summary-button [data-live-phase]')),
            sideBox: box(kid.querySelector(':scope > .chat-live-summary-button .chat-live-summary-side')),
            when: kid.querySelector('[data-live-when]')?.hidden === false,
            reviewsInFoot: Boolean(kid.querySelector(':scope > .chat-live-foot > .chat-live-foot-end > [data-live-review-summary]')),
            controls: [...kid.querySelectorAll(':scope > .chat-live-foot .chat-live-actions > *')].map((el) => el.textContent.trim())})),
    };
}"""


def _respond(chat):
    def respond(route):
        path = urlparse(route.request.url).path
        data = {}
        if path == "/api/chat/history":
            rows = chat["rows"]
            data = {"messages": rows, "progress": [], "has_more": False, "page_cursor": "cards-b", "next_cursor": None,
                    "window": {"complete": True}, "coverage": {
                        "v": 1, "view": "cards-b", "upper": {"chat": len(rows), "progress": 0}, "spans": {
                            "chat": {"from": 0, "to": len(rows), "chain": "cards-b", "gaps": []}, "progress": None}}}
        elif path == "/api/state":
            data = with_work_fields(chat["state"])
        elif path.startswith("/api/tasks/"):
            task_id = path.split("/")[3]
            data = chat["details"].get(task_id, {"task_id": task_id, "status": "running"})
            if task_id in chat["hold"]:  # answered when the test releases it, after the card painted without it
                chat["held"].append((route, data))
                return
        route.fulfill(content_type="application/json", body=json.dumps(data))
    return respond


@contextmanager
def card_room(pw, engine, theme, width, height, *, coarse=False, web=WEB, rows=None, state=None, details=None, chat_id=1):
    """A real chat instance in the real shell (`web`'s index.html and stylesheets), Main or the Project room `chat_id`:
    `rows` is the history it reads, `state` the census its `/api/state` answers, `details` its task reads; all three
    stay mutable. A read of a task in `hold` waits in `held` until the test answers it."""
    chat = {"rows": list(rows or []), "details": dict(details or {}), "hold": set(), "held": [],
            "state": state or {"supervisor_ready": True, "active_chat_activities_complete": True, "active_chat_activities": []}}
    boot = BOOT.replace("chatId:1,", f"chatId:{chat_id},asPanel:{str(chat_id != 1).lower()},")
    html = (Path(web) / "index.html").read_text(encoding="utf-8").replace(
        '<script type="module" src="/static/app.js"></script>', boot)

    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            if self.path == "/":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(html.encode("utf-8"))
                return
            self.path = self.path.removeprefix("/static")
            super().do_GET()

    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(web)))
    Thread(target=server.serve_forever, daemon=True).start()
    browser = getattr(pw, engine).launch(headless=True)
    try:
        page = browser.new_page(viewport={"width": width, "height": height}, has_touch=coarse,
                                timezone_id="Europe/Moscow", device_scale_factor=2)
        page.add_init_script(f"localStorage.setItem('ouroboros.theme', {json.dumps(theme)});")
        page.route("**/api/**", _respond(chat))
        errors = PageErrors(page, engine)

        def emit(frames):
            for channel, row in frames:
                page.evaluate("([type, row]) => window.emit(type, row)", [channel, row])

        def load():
            page.goto(f"http://127.0.0.1:{server.server_port}/")
            page.wait_for_function("() => window.chat?.hasPaintedHistory()")
            page.evaluate("s => chat.hydrateStateSnapshot(s)", with_work_fields(chat["state"]))

        load()
        yield {**chat, "page": page, "emit": emit, "load": load, "errors": errors, "engine": engine, "theme": theme,
               "width": width, "coarse": coarse, "facts": lambda task_id: page.evaluate(ROOT_FACTS, task_id)}
        assert not errors.unexpected(), errors
    finally:
        browser.close()
        server.shutdown()
        server.server_close()


def _settle(page):
    page.wait_for_timeout(250)
    page.evaluate("() => new Promise((done) => requestAnimationFrame(() => requestAnimationFrame(done)))")


def _evidence(name):
    root = os.environ.get("OUROBOROS_BROWSER_EVIDENCE_OUT", "").strip()
    if not root:
        return None
    path = Path(root) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def _assert_b_root(facts, *, clock=None, reviews="", buttons=()):
    """The B composition of one ordinary root card, as measured on its own render."""
    chip, title, line = facts["chipBox"], facts["titleBox"], facts["lineBox"]
    details = facts["detailsBox"]
    # Head: the chip, its clock and the doors share one row; the doors stand at its right end, the last door's
    # chevron on the card's content edge (each door pays its padding back).
    assert details and abs((details["t"] + details["b"]) / 2 - (chip["t"] + chip["b"]) / 2) <= 2, facts
    assert abs(facts["chevronBox"]["r"] - facts["contentRight"]) <= 2, facts
    if clock is not None:
        assert facts["when"] == clock, facts
        if clock:
            # Beside the chip; on a card too narrow for chip, clock and both doors, under the chip (never a door).
            when = facts["whenBox"]
            beside = abs((when["t"] + when["b"]) / 2 - (chip["t"] + chip["b"]) / 2) <= 2
            under = when["t"] >= chip["b"] - 1 and abs(when["l"] - chip["l"]) <= 1 and bool(reviews)
            assert beside or under, facts
    assert "updated" not in facts["meta"] and "started" not in facts["meta"], facts  # the clock moved once, whole
    assert facts["reviews"] == reviews, facts
    if reviews:
        r = facts["reviewsBox"]
        assert facts["reviewsInHead"] and abs(r["t"] - details["t"]) <= 1 and r["r"] <= details["l"], facts
        assert facts["reviewsColor"] == facts["detailsColor"] and facts["reviewsFont"] == facts["detailsFont"], facts
    # The name and the line wrap whole: no clamp and no cut, at 16px and 14px, the line close under the name.
    head = max(box["b"] for box in (chip, facts["whenBox"], details) if box)
    assert title is None or (title["t"] >= head + 4 and facts["titleFont"] == "16px" and facts["titleWeight"] == "500"), facts
    assert facts["titleClamp"] in ("none", None, "") and not facts["titleCut"], facts
    if facts["line"]:
        assert facts["lineFont"] == "14px" and facts["lineClamp"] in ("none", None, "") and not facts["lineCut"], facts
        assert not facts["line"].endswith("…") and 2 <= line["t"] - title["b"] <= 6, facts
    # The foot: one band, facts left and the host's commands right, never a fact under a button.
    names = [button["name"] for button in facts["buttons"]]
    assert names == list(buttons), facts
    if facts["footShown"]:
        last = facts["lineBox"] or facts["titleBox"] or chip
        assert 10 <= facts["footBox"]["t"] - last["b"] <= 14, facts
        if not facts["kids"] and not facts["expanded"]:
            assert abs(facts["cardInnerBottom"] - facts["footInnerBottom"] - 16) <= 1.5, facts
    for button in facts["buttons"]:
        assert button["r"] <= facts["contentRight"] + 1 and "btn-sm" in button["cls"] and "ui-hit-area" in button["cls"], facts
        for part in facts["metaParts"]:
            assert part["b"] <= button["t"] + button["h"], ("a fact under a button", facts)
    if facts["buttons"]:
        assert facts["buttons"][-1]["r"] >= facts["contentRight"] - 1, facts
    assert not facts["hscroll"], facts


@pytest.mark.parametrize("engine,theme,width,height,coarse", [
    ("webkit", "light", 1100, 750, False), ("webkit", "dark", 1100, 750, False),
    ("webkit", "dark", 390, 844, True), ("webkit", "light", 390, 844, True),
    ("chromium", "dark", 1100, 750, False), ("chromium", "light", 390, 844, True),
], ids=["webkit-light-1100", "webkit-dark-1100", "webkit-dark-390", "webkit-light-390", "chromium-dark-1100",
        "chromium-light-390"])
def test_root_cards_compose_head_whole_text_and_one_foot_band(engine, theme, width, height, coarse):
    if os.environ.get("OUROBOROS_RUN_UI_SMOKE") != "1":
        pytest.skip("Set OUROBOROS_RUN_UI_SMOKE=1 to run the browser flow")
    playwright = pytest.importorskip("playwright.sync_api")
    state = {"supervisor_ready": True, "active_chat_activities_complete": True, "active_chat_activities": CENSUS}
    with playwright.sync_playwright() as pw, card_room(pw, engine, theme, width, height, coarse=coarse, state=state) as room:
        page = room["page"]
        room["emit"](QUIET)
        page.evaluate("s => chat.hydrateStateSnapshot(s)", with_work_fields(state))
        _settle(page)
        shot = _evidence(f"quiet-{engine}-{theme}-{width}.png")
        if shot:
            page.screenshot(path=shot)
        q1, q2, q3 = (room["facts"](task_id) for task_id in ("q1", "q2", "q3"))
        _assert_b_root(q1, clock="updated 22:42", buttons=("Turn into project", "Stop…"))
        _assert_b_root(q2, clock="", buttons=("Turn into project",))
        _assert_b_root(q3, clock="started 22:43", buttons=("Turn into project",))
        # The words stand whole as the card's text; the facts keep their meta line.
        assert (q1["title"], q2["title"], q3["title"]) == (Q1, Q2, Q3)
        for facts, duration in ((q1, "3m 3s"), (q2, "2m 48s"), (q3, "2m 49s")):
            assert facts["meta"].split(" · ")[1:] == [duration, "4 tool calls", "Cost unknown"], facts

        room["emit"](WORK)
        _settle(page)
        page.locator('#chat-messages > .chat-live-card[data-task-id="w-root"]').scroll_into_view_if_needed()
        _settle(page)
        work = room["facts"]("w-root")
        _assert_b_root(work, clock="updated 22:47", reviews="Reviews 2", buttons=("Turn into project", "Stop…"))
        assert work["title"] == WORK_TITLE and work["line"] == WORK_LINE, work
        assert work["meta"] == "model-bravo-5.5 · 12m 0s · 18 tool calls · Tracked: $0.42", work
        # Children keep their identity row and carry no root control and no clock of a root.
        assert [kid["role"] for kid in work["kids"]] == [LONG_ROLE, "Независимая проверка расчётов"], work
        for kid in work["kids"]:
            assert not kid["when"] and "Turn into project" not in kid["controls"] and "Stop…" not in kid["controls"], kid
            assert abs(kid["titleBox"]["t"] - kid["chipBox"]["t"]) <= 4 and kid["sideBox"]["r"] <= kid["box"]["r"], kid
        if coarse:  # before any screenshot: Chromium's element capture resets the touch emulation
            hits = page.evaluate(FINGER_TARGETS, "w-root")
            assert hits["coarse"] and hits["overlaps"] == [], hits
        shot = _evidence(f"work-{engine}-{theme}-{width}.png")
        if shot:
            page.locator('#chat-messages > .chat-live-card[data-task-id="w-root"]').screenshot(path=shot)

        # Reviews and Details open independently, after the foot and before the children, moving no node.
        card = page.locator('#chat-messages > .chat-live-card[data-task-id="w-root"]')
        page.evaluate("""() => { const card = document.querySelector('#chat-messages > .chat-live-card[data-task-id="w-root"]');
            [...card.querySelectorAll('*')].forEach((node, index) => { node.__b = index; }); }""")
        card.locator(":scope > .chat-live-summary-button [data-live-review-summary]").click()
        card.locator(":scope > .chat-live-summary-button [data-live-disclosure]").click()
        _settle(page)
        opened = page.evaluate("""() => { const card = document.querySelector('#chat-messages > .chat-live-card[data-task-id="w-root"]');
            const r = (sel) => card.querySelector(sel)?.getBoundingClientRect();
            return {foot: r(':scope > .chat-live-foot').bottom, reviews: r(':scope > [data-live-reviews-host] [data-review-section]')?.top,
                details: r(':scope > .chat-live-details')?.top, kids: r(':scope > .chat-subagents')?.top,
                reviewsBottom: r(':scope > [data-live-reviews-host]')?.bottom, notesBottom: r(':scope > .chat-live-timeline')?.bottom,
                kept: [...card.querySelectorAll(':scope > .chat-live-summary-button *, :scope > .chat-live-foot *')]
                    .every((node) => node.__b !== undefined)}; }""")
        assert opened["reviews"] >= opened["foot"] - 1 and opened["details"] >= opened["reviewsBottom"] - 1, opened
        assert opened["kids"] >= opened["notesBottom"] - 1 and opened["kept"], opened
        assert card.locator(":scope > .chat-live-summary-button [data-live-review-summary]").get_attribute("aria-expanded") == "true"
        assert card.get_attribute("data-expanded") == "1"
        shot = _evidence(f"work-open-{engine}-{theme}-{width}.png")
        if shot:
            card.screenshot(path=shot)


# Every finger target of one card's head and foot (ui.css .ui-hit-area: centred, at least the touch minimum each way).
FINGER_TARGETS = """id => {
    const card = document.querySelector(`#chat-messages > .chat-live-card[data-task-id="${id}"]`);
    const min = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--touch-target-min'));
    const hits = [...card.querySelectorAll(':scope > .chat-live-summary-button .ui-hit-area, :scope > .chat-live-foot .ui-hit-area')]
        .filter((el) => el.getClientRects().length).map((el) => { const r = el.getBoundingClientRect();
            const dx = Math.max(0, (min - r.width) / 2), dy = Math.max(0, (min - r.height) / 2);
            return {name: el.textContent.trim(), l: r.left - dx, t: r.top - dy, r: r.right + dx, b: r.bottom + dy}; });
    const overlaps = [];
    hits.forEach((a, i) => hits.slice(i + 1).forEach((b) => {
        const w = Math.min(a.r, b.r) - Math.max(a.l, b.l), h = Math.min(a.b, b.b) - Math.max(a.t, b.t);
        if (w > 0.5 && h > 0.5) overlaps.push([a.name, b.name, w, h]);
    }));
    return {coarse: matchMedia('(pointer: coarse)').matches, names: hits.map((hit) => hit.name), overlaps};
}"""


def _history(frames):
    """The rows a reload reads back for these live frames: chat rows as they were, and each task's log facts on its
    saved summary row (the history carries no log stream)."""
    rows, facts = [], {}
    for channel, row in frames:
        if channel == "chat":
            rows.append(dict(row))
        elif channel == "log":
            data = {key: value for key, value in row["data"].items() if key != "type"}
            facts.setdefault(data["task_id"], {}).update(data)
    for task_id, data in facts.items():
        if data.get("task_terminal_status"):
            rows.append({**data, "chat_id": 1, "role": "system", "system_type": "task_summary", "text": ""})
    return rows


def _skip_without_smoke():
    if os.environ.get("OUROBOROS_RUN_UI_SMOKE") != "1":
        pytest.skip("Set OUROBOROS_RUN_UI_SMOKE=1 to run the browser flow")
    return pytest.importorskip("playwright.sync_api")


@pytest.mark.parametrize("engine", ["webkit", "chromium"])
def test_root_card_live_terminal_and_reload_keep_one_composition(engine):
    """Live, ended and read back after a reload, the same card keeps its head, whole text and one foot band; the clock
    leaves when the work ends, and Stop leaves with it."""
    playwright = _skip_without_smoke()
    state = {"supervisor_ready": True, "active_chat_activities_complete": True, "active_chat_activities": CENSUS[-1:]}
    idle = {"supervisor_ready": True, "active_chat_activities_complete": True, "active_chat_activities": []}
    end = [_done("w-root", _at("22:59"), tool_calls=18, duration_sec=720, **TRACKED)]
    with playwright.sync_playwright() as pw, card_room(pw, engine, "light", 1100, 750, state=state) as room:
        page = room["page"]
        room["emit"](WORK)
        _settle(page)
        live = room["facts"]("w-root")
        _assert_b_root(live, clock="updated 22:47", reviews="Reviews 2", buttons=("Turn into project", "Stop…"))
        room["state"].update(idle)
        page.evaluate("s => chat.hydrateStateSnapshot(s)", with_work_fields(idle))
        room["emit"](end)
        page.wait_for_function("""() => document.querySelector('#chat-messages > .chat-live-card[data-task-id="w-root"]')
            ?.dataset.finished === '1'""")
        _settle(page)
        ended = room["facts"]("w-root")
        _assert_b_root(ended, clock="", reviews="Reviews 2", buttons=("Turn into project",))
        assert ended["chip"] == "Done" and (ended["title"], ended["line"]) == (WORK_TITLE, WORK_LINE), ended
        room["rows"][:] = _history(WORK + end)
        room["load"]()
        page.locator('#chat-messages > .chat-live-card[data-task-id="w-root"]').wait_for()
        _settle(page)
        replayed = room["facts"]("w-root")
        _assert_b_root(replayed, clock="", reviews="Reviews 2", buttons=("Turn into project",))
        for key in ("chip", "title", "line", "meta"):
            assert replayed[key] == ended[key], (key, ended, replayed)
        assert [kid["role"] for kid in replayed["kids"]] == [kid["role"] for kid in ended["kids"]], replayed


FOOT = """id => { const card = document.querySelector(`.chat-live-card[data-task-id="${id}"]:not(.subagent)`);
    const foot = card.querySelector(':scope > .chat-live-foot'), s = card.querySelector(':scope > .chat-live-summary-button');
    const shown = foot.getClientRects().length > 0, last = shown ? foot : s, style = getComputedStyle(last);
    return {shown, buttons: [...foot.querySelectorAll('.chat-live-actions > *')].map((b) => b.textContent.trim()),
        inset: card.getBoundingClientRect().bottom - (parseFloat(getComputedStyle(card).borderBottomWidth) || 0)
            - (last.getBoundingClientRect().bottom - parseFloat(style.paddingBottom))}; }"""


@pytest.mark.parametrize("engine", ["webkit", "chromium"])
def test_a_late_command_paints_a_bare_or_meta_only_foot_in_a_room(engine):
    """In a Project room a card offers no conversion, so its foot can hold nothing or only its facts. A foot with
    nothing to say closes the card on its summary; a command that lands after the first paint (the Continue a settled
    record offers after a reload, a Stop the host grants) paints its band at once, in WebKit too (`:has()`, never
    `:empty`, re-matched). A card that exists for its reviews shows its door in its head, never an empty foot."""
    playwright = _skip_without_smoke()
    offer = {"status": "failed", "task_terminal_status": "failed",
             "continuation_offer": {"eligible": True, "cause": "technical_interruption"}}
    rows = [_ended("nw", _at("22:41")), _ended("mo", _at("22:42"), **UNKNOWN)]
    details = {"nw": {"task_id": "nw", **offer}, "mo": {"task_id": "mo", **offer}}
    with playwright.sync_playwright() as pw, card_room(pw, engine, "light", 1100, 750, rows=rows, details=details,
                                                        chat_id=7) as room:
        page = room["page"]
        foot = lambda task_id: page.evaluate(FOOT, task_id)  # noqa: E731
        # The record reads that offer Continue wait until the cards painted without it.
        room["hold"].update({"nw", "mo"})
        room["load"]()
        page.wait_for_function("() => document.querySelectorAll('.chat-live-card[data-finished=\"1\"]').length >= 2")
        _settle(page)
        assert not foot("nw")["shown"] and abs(foot("nw")["inset"] - 16) <= 1.5, foot("nw")
        meta_only = room["facts"]("mo")
        assert meta_only["footShown"] and meta_only["meta"] == "Cost unknown" and not meta_only["buttons"], meta_only
        for _ in range(100):
            if len(room["held"]) >= 2:
                break
            page.wait_for_timeout(50)
        assert len(room["held"]) >= 2, room["held"]
        room["hold"].clear()
        held, room["held"][:] = list(room["held"]), []
        for route, data in held:
            route.fulfill(content_type="application/json", body=json.dumps(data))
        for task_id in ("nw", "mo"):
            page.locator(f'.chat-live-card[data-task-id="{task_id}"] [data-continue-task]').wait_for()
        _settle(page)
        bare = foot("nw")
        assert bare["shown"] and bare["buttons"] == ["Continue"] and abs(bare["inset"] - 16) <= 1.5, bare
        meta_only = room["facts"]("mo")
        assert meta_only["meta"] == "Cost unknown" and [b["name"] for b in meta_only["buttons"]] == ["Continue"], meta_only
        assert abs((meta_only["buttons"][0]["t"] + meta_only["buttons"][0]["b"]) / 2
                   - (meta_only["metaBox"]["t"] + meta_only["metaBox"]["b"]) / 2) <= 2, meta_only
        # A card that exists for its reviews: the door in its head, no empty foot; a Stop granted later paints the band.
        room["emit"]([("log", {"chat_id": 7, "data": {"type": "task_metrics_event", "task_id": "rv", "ts": _at("22:43"),
                                                      "review_projection": {"panels": [{"panel_id": "rv-1", "surface": "task_acceptance",
                                                                                         "aggregate_signal": "PASS", "reason": "ok", "actors": []}]}}})])
        page.locator('.chat-live-card[data-task-id="rv"]').wait_for()
        _settle(page)
        reviewed = room["facts"]("rv")
        assert reviewed["reviews"] == "Reviews 1" and reviewed["reviewsInHead"], reviewed
        assert not foot("rv")["shown"] and abs(foot("rv")["inset"] - 16) <= 1.5, foot("rv")
        room["emit"]([("log", {"chat_id": 7, "data": {"type": "task_metrics_event", "task_id": "rv", "ts": _at("22:44"),
                                                      "cancelable": True}})])
        page.locator('.chat-live-card[data-task-id="rv"] [data-cancel-run]').wait_for()
        _settle(page)
        granted = foot("rv")
        assert granted["shown"] and granted["buttons"] == ["Stop…"] and abs(granted["inset"] - 16) <= 1.5, granted
        shot = _evidence(f"room-late-commands-{engine}-light-1100.png")
        if shot:
            page.screenshot(path=shot)


@pytest.mark.parametrize("engine", ["webkit", "chromium"])
def test_a_main_portal_keeps_its_own_composition(engine):
    """A Main portal is not an ordinary card: its Project reference leads its row under the meta line, its clock ends
    that line, its Reviews and Details doors stay hidden and its latest line keeps two lines cut on a word."""
    playwright = _skip_without_smoke()
    state = {"supervisor_ready": True, "active_chat_activities_complete": True, "task_bindings": {"pt": {"project_id": "alpha"}},
             "projects": [{"id": "alpha", "name": "Project Alpha", "chat_id": 7}],
             "active_chat_activities": [{"activity_id": "pt", "project_id": "alpha", "project_name": "Project Alpha",
                                         "chat_id": 7, "kind": "managed_task", "phase": "working", "started_at": _epoch("22:40"),
                                         "last_line": "Comparing the reviewer verdicts across the last three runs. " * 4,
                                         "last_line_at": _at("22:44"), "tool_calls": 41, "cancelable": True}]}
    rows = [{"chat_id": 1, "role": "system", "system_type": "project_started", "task_id": "pt", "project_id": "alpha",
             "project_name": "Project Alpha", "task_name": WORK_TITLE, "text": "Project Alpha › Started", "ts": _at("22:40"),
             "history_id": "start-pt"}]
    with playwright.sync_playwright() as pw, card_room(pw, engine, "dark", 390, 844, coarse=True, rows=rows, state=state) as room:
        page = room["page"]
        page.locator('#chat-messages > .chat-live-card[data-task-id="pt"][data-portal="1"]').wait_for()
        _settle(page)
        portal = page.evaluate("""() => { const card = document.querySelector('#chat-messages > .chat-live-card[data-task-id="pt"]');
            const meta = card.querySelector(':scope > .chat-live-foot > .chat-live-meta');
            const row = card.querySelector(':scope > .chat-live-foot > .chat-live-foot-end');
            const line = card.querySelector(':scope > .chat-live-summary-button [data-live-activity]');
            const lh = parseFloat(getComputedStyle(line).lineHeight);
            return {leads: row.firstElementChild?.classList.contains('chat-live-bound-pointer'),
                under: row.getBoundingClientRect().top >= meta.getBoundingClientRect().bottom - 1, meta: meta.textContent,
                when: card.querySelector('[data-live-when]').hidden, doors: getComputedStyle(card.querySelector('[data-live-review-summary]')).display,
                group: getComputedStyle(card.querySelector('.chat-live-doors')).display,
                status: card.querySelector('.chat-live-summary-main').getBoundingClientRect().width,
                chip: card.querySelector('[data-live-phase]').getBoundingClientRect().width,
                details: getComputedStyle(card.querySelector('[data-live-disclosure]')).display,
                lines: Math.round(line.getBoundingClientRect().height / lh), cut: line.textContent.endsWith('…'),
                buttons: [...row.querySelectorAll('.chat-live-actions > *')].map((b) => b.textContent.trim())}; }""")
        assert portal["leads"] and portal["under"] and portal["buttons"] == ["Stop…"], portal
        assert "updated 22:44" in portal["meta"] and portal["when"] and portal["doors"] == "none", portal
        assert portal["details"] == "none" and portal["group"] == "none" and portal["lines"] <= 2 and portal["cut"], portal
        assert abs(portal["status"] - portal["chip"]) <= 1, portal  # no door to make room for: the status keeps its chip's box
        shot = _evidence(f"portal-{engine}-dark-390.png")
        if shot:
            page.locator('#chat-messages > .chat-live-card[data-task-id="pt"]').screenshot(path=shot)


@pytest.mark.parametrize("engine", ["webkit", "chromium"])
def test_a_root_card_growing_above_the_reader_keeps_the_reading_position(engine):
    """A root's whole line grows (a longer narration lands) while the reader reads below it: the passage under the
    reader's eyes stays where it was."""
    playwright = _skip_without_smoke()
    rows = [_say("grow-root", "Short start.", _at("21:00"), suggested_name="Card above the reader")[1]]
    for index in range(36):
        rows.append({"chat_id": 1, "role": "user" if index % 2 == 0 else "assistant", "ts": _at(f"21:{index + 1:02d}"),
                     "text": f"Message {index}: " + "a passage long enough to take two lines of the column. " * 2,
                     "history_id": f"h-{index}"})
    with playwright.sync_playwright() as pw, card_room(pw, engine, "light", 1100, 750, rows=rows) as room:
        page = room["page"]
        page.locator('#chat-messages > .chat-live-card[data-task-id="grow-root"]').wait_for(state="attached")
        _settle(page)
        anchor = page.locator("#chat-messages .chat-bubble", has_text="Message 18:").first
        anchor.evaluate("el => el.scrollIntoView({block: 'start'})")
        anchor.hover()
        page.mouse.wheel(0, -40)  # the reader's own gesture: off the live edge, reading
        _settle(page)
        before = anchor.evaluate("el => el.getBoundingClientRect().top")
        card_bottom = page.evaluate("""() => document.querySelector('#chat-messages > .chat-live-card[data-task-id="grow-root"]')
            .getBoundingClientRect().bottom""")
        assert card_bottom < 0, "the card is above the reader"
        grown = "A longer narration: " + "the whole line wraps and grows the card above. " * 4  # under the preview bound
        room["emit"]([_say("grow-root", grown, _at("21:40"))])
        page.wait_for_function("""text => document.querySelector('#chat-messages > .chat-live-card[data-task-id="grow-root"] [data-live-activity]')
            ?.textContent === text""", arg=grown.strip())
        _settle(page)
        after = anchor.evaluate("el => el.getBoundingClientRect().top")
        assert abs(after - before) <= 1, (before, after)
        height = page.evaluate("""() => document.querySelector('#chat-messages > .chat-live-card[data-task-id="grow-root"] [data-live-activity]')
            .getBoundingClientRect().height""")
        assert height > 40, height  # it really grew by lines, unclamped


# Each door as [says open (`aria-expanded`), chevron turned]: a root's two doors and its first child's Details.
DOORS = """id => { const card = document.querySelector(`#chat-messages > .chat-live-card[data-task-id="${id}"]`);
    const door = (owner, sel) => { const el = owner.querySelector(`:scope > .chat-live-summary-button ${sel}`);
        const turn = getComputedStyle(el.querySelector('.chat-live-chevron')).transform;
        return [el.getAttribute('aria-expanded') === 'true', turn !== 'none' && new DOMMatrix(turn).a < 0]; };
    return {reviews: door(card, '[data-live-review-summary]'), details: door(card, '[data-live-disclosure]'),
        kid: door(card.querySelector(':scope > .chat-subagents > .chat-live-card.subagent'), '[data-live-disclosure]')}; }"""


@pytest.mark.parametrize("engine", ["webkit", "chromium"])
def test_each_door_chevron_says_its_own_door_in_every_state(engine):
    """Reviews and Details open independently, so each chevron turns with its own door's `aria-expanded` alone, whichever
    door opened first; a child's Details keeps its own state under its open parent."""
    playwright = _skip_without_smoke()
    state = {"supervisor_ready": True, "active_chat_activities_complete": True, "active_chat_activities": CENSUS[-1:]}
    with playwright.sync_playwright() as pw, card_room(pw, engine, "light", 1100, 750, state=state) as room:
        page = room["page"]
        room["emit"](WORK)
        _settle(page)
        card = page.locator('#chat-messages > .chat-live-card[data-task-id="w-root"]')
        reviews = card.locator(":scope > .chat-live-summary-button [data-live-review-summary]")
        details = card.locator(":scope > .chat-live-summary-button [data-live-disclosure]")
        kid = card.locator(":scope > .chat-subagents > .chat-live-card.subagent").first.locator(
            ":scope > .chat-live-summary-button [data-live-disclosure]")
        for door, reviews_open, details_open, kid_open in (
                (None, False, False, False), (details, False, True, False), (kid, False, True, True), (kid, False, True, False),
                (reviews, True, True, False), (details, True, False, False), (reviews, False, False, False)):
            if door:
                door.click()
                _settle(page)
            facts = page.evaluate(DOORS, "w-root")
            assert facts == {"reviews": [reviews_open] * 2, "details": [details_open] * 2, "kid": [kid_open] * 2}, facts
            if details_open and not reviews_open and not kid_open:
                shot = _evidence(f"doors-details-open-reviews-closed-{engine}-light-1100.png")
                if shot:
                    card.screenshot(path=shot)


# A Swarm root's stored detail in the host's receipt facts. Details reads it, the way the line reaches a card.
SWARM = {"status": "failed", "metadata": {"subagents_mode": "swarm"}, "swarm_efficiency": {"subagent_count": 2}}
SWARM_SHORT = "Swarm · 2 helpers requested"
SWARM_LONG = {**SWARM, "swarm_efficiency": {"subagent_count": 12}, "swarm_receipt": {"cost": {
    "known_final_usd": 12.34, "unknown_attempts": 3, "attempts": [
        {"attempt_id": "a1", "task_id": "w-root", "model": "model-alpha"},
        {"attempt_id": "a2", "task_id": "s1", "model": "model-bravo", "kind": "subscription_session"},
        {"attempt_id": "a3", "task_id": "s2", "model": "model-charlie", "kind": "subscription_session"}]}}}
SWARM_LONG_TEXT = "Swarm · 12 helpers requested · 3 models · $12.34 known + 3 not final · 2 subscription sessions"


def _read_receipt(room, task_id):
    """Details' first opening reads the root's detail and the line lands; closed again, the card is measured collapsed."""
    door = room["page"].locator(f'.chat-live-card[data-task-id="{task_id}"] > .chat-live-summary-button [data-live-disclosure]')
    door.click()
    room["page"].locator(f'.chat-live-card[data-task-id="{task_id}"] > .chat-live-foot > .chat-live-swarm-receipt').wait_for()
    door.click()
    _settle(room["page"])
    return room["facts"](task_id)


def _assert_receipt_line(facts, text):
    """The receipt stands first in the foot on a full-width line of its own; no fact and no command shares that line."""
    receipt = facts["receipt"]
    assert receipt and facts["receiptFirst"] and facts["receiptText"] == text, facts
    assert abs(receipt["w"] - facts["footWidth"]) <= 1, facts
    assert all(other["t"] >= receipt["b"] for other in ([facts["metaBox"]] if facts["meta"] else []) + facts["buttons"]), facts


@pytest.mark.parametrize("engine,theme,width,height,coarse,detail,text", [
    ("webkit", "light", 1100, 750, False, SWARM, SWARM_SHORT), ("chromium", "light", 1100, 750, False, SWARM, SWARM_SHORT),
    ("webkit", "dark", 1100, 750, False, SWARM_LONG, SWARM_LONG_TEXT), ("webkit", "dark", 390, 844, True, SWARM_LONG, SWARM_LONG_TEXT),
    ("chromium", "light", 390, 844, True, SWARM, SWARM_SHORT),
], ids=["webkit-light-1100-short", "chromium-light-1100-short", "webkit-dark-1100-long", "webkit-dark-390-long", "chromium-light-390-short"])
def test_a_swarm_receipt_stands_on_its_own_line_above_the_foot_band(engine, theme, width, height, coarse, detail, text):
    """A short receipt never joins the facts' row: it stays its own line, first in the foot, a long one wrapping inside
    that line, and the facts and commands keep their one band under it."""
    playwright = _skip_without_smoke()
    state = {"supervisor_ready": True, "active_chat_activities_complete": True, "active_chat_activities": CENSUS[-1:]}
    with playwright.sync_playwright() as pw, card_room(pw, engine, theme, width, height, coarse=coarse, state=state,
                                                        details={"w-root": {"task_id": "w-root", **detail}}) as room:
        room["emit"](WORK)
        _settle(room["page"])
        facts = _read_receipt(room, "w-root")
        _assert_b_root(facts, clock="updated 22:47", reviews="Reviews 2", buttons=("Turn into project", "Stop…"))
        _assert_receipt_line(facts, text)
        if width > 600:  # the facts at the band's left, the commands at its right end
            button = facts["buttons"][0]
            assert abs((button["t"] + button["b"]) / 2 - (facts["metaBox"]["t"] + facts["metaBox"]["b"]) / 2) <= 2, facts
        shot = _evidence(f"swarm-{engine}-{theme}-{width}-{'long' if detail is SWARM_LONG else 'short'}.png")
        if shot:
            room["page"].locator('#chat-messages > .chat-live-card[data-task-id="w-root"]').screenshot(path=shot)


@pytest.mark.parametrize("engine", ["webkit", "chromium"])
def test_a_receipt_alone_keeps_its_foot_and_takes_one_line(engine):
    """A foot holding only a receipt stands: its `:has()` re-matches when the line lands late, WebKit too. The card closes
    16px under that band, with no empty facts line under it and no second inset on the summary above it."""
    playwright = _skip_without_smoke()
    rows = [_ended("sw-bare", _at("22:41")), _ended("sw-meta", _at("22:42"), **UNKNOWN)]
    details = {task_id: {"task_id": task_id, **SWARM} for task_id in ("sw-bare", "sw-meta")}
    with playwright.sync_playwright() as pw, card_room(pw, engine, "light", 1100, 750, rows=rows, details=details, chat_id=7) as room:
        page = room["page"]
        page.wait_for_function("() => document.querySelectorAll('.chat-live-card[data-finished=\"1\"]').length >= 2")
        _settle(page)
        bare = page.evaluate(FOOT, "sw-bare")
        assert not bare["shown"] and abs(bare["inset"] - 16) <= 1.5, bare  # nothing to say yet: no foot
        facts, bare = _read_receipt(room, "sw-bare"), page.evaluate(FOOT, "sw-bare")
        assert bare["shown"] and abs(bare["inset"] - 16) <= 1.5, bare
        _assert_receipt_line(facts, SWARM_SHORT)
        assert facts["metaBox"] is None and not facts["buttons"], facts  # the empty facts take no line of their own
        above = max(box["b"] for box in (facts["chipBox"], facts["titleBox"], facts["lineBox"]) if box)
        assert 10 <= facts["footBox"]["t"] - above <= 14 and facts["footBox"]["h"] <= 28 + 16 + 1, facts
        with_meta = _read_receipt(room, "sw-meta")
        _assert_receipt_line(with_meta, SWARM_SHORT)
        assert with_meta["meta"] == "Cost unknown" and with_meta["metaBox"]["t"] >= with_meta["receipt"]["b"], with_meta
        shot = _evidence(f"swarm-receipt-alone-{engine}-light-1100.png")
        if shot:
            page.screenshot(path=shot)


# Past the 240-character preview, with Markdown the line's projection removes and a tail only whole text keeps.
LONG = ("**Reading** the `ledger` rows\n\n" + "Comparing every saved row with the ledger, one source at a time. " * 6
        + "UNIQUE_ROOT_TAIL")
LONG_WHOLE = "Reading the ledger rows " + "Comparing every saved row with the ledger, one source at a time. " * 6 + "UNIQUE_ROOT_TAIL"
LONG_TEXT = """id => { const card = document.querySelector(`.chat-live-card[data-task-id="${id}"]`);
    const get = (sel) => card.querySelector(`:scope > .chat-live-summary-button ${sel}`);
    const lines = (el) => Math.round(el.getBoundingClientRect().height / parseFloat(getComputedStyle(el).lineHeight));
    const title = get('[data-live-title]'), line = get('[data-live-activity]');
    return {title: title.textContent, line: line.textContent, lineLines: line.textContent ? lines(line) : 0,
        titleCut: title.scrollHeight > title.clientHeight + 1, lineCut: line.scrollHeight > line.clientHeight + 1}; }"""


@pytest.mark.parametrize("engine", ["webkit", "chromium"])
def test_long_root_title_and_line_are_whole_live_and_after_reload(engine):
    """An ordinary root's chosen title (an unnamed root: its newest authored words) and its latest authored line (a
    named root) stay whole past the 240-character preview, live and read back after a reload; a child keeps one fitted
    line of its bounded preview and a Main portal its two lines cut on a word."""
    playwright = _skip_without_smoke()
    assert len(LONG_WHOLE) > 380
    frames = [_say("long-unnamed", LONG, _at("22:50")), _say("long-named", LONG, _at("22:51"), suggested_name="Data Analysis"),
              ("chat", {"chat_id": 1, "task_id": "long-kid", "subagent_task_id": "long-kid", "parent_task_id": "long-named",
                        "root_task_id": "long-named", "delegation_role": "subagent", "subagent_event": "running",
                        "subagent_role": "Reader", "role": "assistant", "is_progress": True, "narration": True,
                        "content": LONG, "text": LONG, "ts": _at("22:52")})]
    state = {"supervisor_ready": True, "active_chat_activities_complete": True, "task_bindings": {"long-portal": {"project_id": "alpha"}},
             "projects": [{"id": "alpha", "name": "Project Alpha", "chat_id": 7}],
             "active_chat_activities": [{"activity_id": "long-portal", "project_id": "alpha", "project_name": "Project Alpha",
                                         "chat_id": 7, "kind": "managed_task", "phase": "working", "started_at": _epoch("22:40"),
                                         "last_line": LONG_WHOLE, "last_line_at": _at("22:44"), "tool_calls": 3}]}
    portal_row = {"chat_id": 1, "role": "system", "system_type": "project_started", "task_id": "long-portal", "project_id": "alpha",
                  "project_name": "Project Alpha", "task_name": "Long portal", "text": "Project Alpha › Started",
                  "ts": _at("22:40"), "history_id": "start-long-portal"}

    def check(room, when):
        page = room["page"]
        unnamed, named = page.evaluate(LONG_TEXT, "long-unnamed"), page.evaluate(LONG_TEXT, "long-named")
        assert unnamed["title"] == LONG_WHOLE and not unnamed["titleCut"] and unnamed["line"] == "", (when, unnamed)
        assert named["title"] == "Data Analysis" and named["line"] == LONG_WHOLE and not named["lineCut"], (when, named)
        assert named["lineLines"] >= 3, (when, named)
        portal = page.evaluate(LONG_TEXT, "long-portal")
        assert portal["lineLines"] <= 2 and portal["line"].endswith("…") and "UNIQUE_ROOT_TAIL" not in portal["line"], (when, portal)

    with playwright.sync_playwright() as pw, card_room(pw, engine, "light", 1100, 750, rows=[portal_row], state=state) as room:
        page = room["page"]
        page.locator('#chat-messages > .chat-live-card[data-task-id="long-portal"][data-portal="1"]').wait_for()
        room["emit"](frames)
        _settle(page)
        check(room, "live")
        kid = page.evaluate(LONG_TEXT, "long-kid")
        assert kid["lineLines"] == 1 and kid["line"].endswith("…") and "UNIQUE_ROOT_TAIL" not in kid["line"], kid
        shot = _evidence(f"long-text-live-{engine}-light-1100.png")
        if shot:
            page.screenshot(path=shot, full_page=True)
        room["rows"][:] = [portal_row] + [dict(row, history_id=f"long-{index}") for index, (channel, row) in enumerate(frames[:2])]
        room["load"]()
        page.locator('#chat-messages > .chat-live-card[data-task-id="long-named"]').wait_for()
        _settle(page)
        check(room, "reload")
