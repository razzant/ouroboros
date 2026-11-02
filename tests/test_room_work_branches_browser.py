"""Missing branches of the room's work status (DESIGN "Room work status").

A test-only addition beside ``tests/test_room_work_browser.py`` (controlled transport) and
``tests/test_room_work_backend_browser.py`` (the real server). Both modules are imported unchanged and
their fixtures, transport and helpers are reused; nothing here patches product code.

Controlled transport (the real SPA in Chromium and WebKit, census/history/detail/socket answered here):
  * translated chrome (an install language whose memory holds every ``tr``/``fmt`` seam of
    web/modules, pseudo-translated as ‹…›) with long Cyrillic/German names and an unbroken path,
    under a coarse pointer at 320 and 390 px, light and dark, closed and open, Main and Project;
  * an unconfirmed send (``Sending…``) handing its slot to the turn the census lists with its exact
    client message id, while another client's turn and the other send stay their own;
  * Back to reading by keyboard, and after the passage it saved has left the feed;
  * a host that is not ready (``supervisor_ready`` false) at boot and while the list is open;
  * subagent and reusable ids (``active``);
  * a pending cancel leaving only Stop now in the row's menu (card and card-less);
  * census reads that fail while the socket stays open (HTTP 500, network);
  * HTTP offline across a socket close and reopen.
Real server (``direct_server_with_data``: candidate checkout with origin proof, isolated data root,
``server.py`` with one worker, the scripted mock model):
  * the existing Project Shade scenario, run in WebKit;
  * Hurry, Wrap up and Stop now from a row, and a server restart with the page and the list open.

Policies the documents do not state are recorded as observations in the evidence, not asserted.
Evidence goes to ``OUROBOROS_UI_EVIDENCE_DIR`` when it is set.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import pytest

from tests import test_room_work_backend_browser as backend
from tests import test_room_work_browser as rw

subscription_ui = rw.subscription_ui
direct_server_with_data = backend.direct_server_with_data
pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]

REPO = Path(__file__).resolve().parents[1]
FULL_MENU = ["finalize", "hurry", "pause", "stop_now"]


@pytest.fixture(params=["chromium", "webkit"])
def branch_ui(request, monkeypatch):
    """``subscription_ui`` in each engine; evidence per engine under ``room-work-branches``."""
    monkeypatch.setenv("OUROBOROS_UI_BROWSER_ENGINE", request.param)
    root = os.environ.get("OUROBOROS_UI_EVIDENCE_DIR")
    if root:
        monkeypatch.setenv("OUROBOROS_UI_EVIDENCE_DIR", str(Path(root) / "room-work-branches" / request.param))
    ui = request.getfixturevalue("subscription_ui")
    return {**ui, "engine": request.param}


class BranchTransport(rw.Transport):
    """``rw.Transport`` plus the failure modes and inputs these branches need: a host that is not ready,
    a census read that fails (HTTP 500 or a network error) while the socket stays open, every HTTP read
    offline, the install's language memory, and the frames the page sends on its socket."""

    def __init__(self, page, *, i18n=None, **kwargs):
        self.ready = True
        self.state_mode = ""       # "" | "http500" | "network"
        self.http_offline = False
        self.state_failures = 0
        self.i18n = i18n
        self.misses: list = []
        self.frames: list = []
        super().__init__(page, **kwargs)
        page.route("**/api/ui/i18n", self._i18n)
        page.route("**/api/ui/i18n/missing", self._missing)
        page.route("**/api/**", self._offline)  # registered last: consulted first

    def _connect(self, ws):
        ws.on_message(lambda message: self.frames.append(message))
        super()._connect(ws)

    def _offline(self, route):
        if self.http_offline:
            route.abort("internetdisconnected")
        else:
            route.fallback()

    def _state(self, route):
        self.state_reads += 1
        if self.state_mode == "http500":
            self.state_failures += 1
            route.fulfill(status=500, json={"error": "state unavailable"})
            return
        if self.state_mode == "network":
            self.state_failures += 1
            route.abort("connectionreset")
            return
        route.fulfill(json=self.body(ready=self.ready))

    def _i18n(self, route):
        if self.i18n is None:
            route.fallback()
        else:
            route.fulfill(json=self.i18n)

    def _missing(self, route):
        self.misses.extend((route.request.post_data_json or {}).get("items") or [])
        route.fulfill(json={"ok": True})

    def chat_frames(self):
        out = []
        for raw in self.frames:
            try:
                frame = json.loads(raw)
            except (TypeError, ValueError):
                continue
            if isinstance(frame, dict) and frame.get("type") == "chat":
                out.append(frame)
        return out


# Every frame for a while, recording only changes: the list's note, its rows and an open shared menu.
ROWS_RECORDER = """([scope, ms]) => {
    window.__rwRows = [];
    const start = performance.now();
    const root = scope === 'main' ? '#page-chat' : '#project-panel';
    let last = '';
    const tick = () => {
        const surface = document.querySelector(`${root} .rw-surface:not(.cx-shade)`);  // the work Shade, not the chip's
        const note = surface?.querySelector('.rw-note');
        const menu = document.querySelector('body > .task-control-menu');
        const state = { hidden: surface ? surface.hidden : null, note: note && !note.hidden ? note.textContent : '',
            tone: document.querySelector(`${root} .rw-trigger`)?.dataset.tone || '',
            menu: menu ? [...menu.querySelectorAll('.task-control-item')].map((i) => i.dataset.taskControl) : null,
            rows: [...(surface?.querySelectorAll('.rw-row') || [])].map((row) => ({ id: row.dataset.id,
                tone: row.dataset.tone, word: row.querySelector('.rw-word')?.textContent || '', settled: row.dataset.settled,
                more: Boolean(row.querySelector('.rw-more')), resume: Boolean(row.querySelector('.rw-pill-resume')),
                question: Boolean(row.querySelector('.rw-pill-question')) })) };
        const key = JSON.stringify(state);
        if (key !== last) { last = key; window.__rwRows.push({ t: Math.round(performance.now() - start), ...state }); }
        if (performance.now() - start < ms) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
}"""

ACTIVE = """() => { const a = document.activeElement;
    if (!a || a === document.body) return { where: 'body' };
    return { where: `${a.tagName}.${String(a.className || '')}`, id: a.id || '', row: a.closest?.('.rw-row')?.dataset.id || '',
             inFeed: Boolean(a.closest?.('#chat-messages')) }; }"""


def record(name, value):
    rw.evidence(name, value)


def frames_now(page):
    return page.evaluate("() => window.__rwRows || []")


def menu_items(page):
    return rw.facts(page)["menu"]


def open_menu(page, task_id, scope="main"):
    root = "#page-chat" if scope == "main" else "#project-panel"
    page.locator(f"{root} .rw-row[data-id='{task_id}'] .rw-more").click()
    page.locator("body > .task-control-menu").wait_for(state="visible", timeout=10_000)
    return menu_items(page)


# ------------------------------------------------------------------------------------------------
# Translated chrome, long names, a coarse pointer at 320 and 390 px, light and dark.

MARK = "‹{}›"
OVERLAY_KEYS = ["Back to reading", "A question waits for your answer", "Earlier place is no longer loaded",
                "The earlier reading place is no longer loaded.", "Returned to where you were reading.",
                "Returned to the nearest loaded place."]
_TR = re.compile(r"""\btr\(\s*(['"])([A-Za-z0-9_.\-]+)\1\s*,\s*(['"])((?:\\.|(?!\3).)*?)\3\s*\)""", re.S)
_FMT = re.compile(r"""\bfmt\(\s*(['"])((?:\\.|(?!\1).)*?)\1""", re.S)


def _unescape(text):
    return text.replace("\\'", "'").replace('\\"', '"').replace("\\n", "\n").replace("\\\\", "\\")


def pseudo_memory():
    """An install language whose memory knows every literal ``tr`` code and ``fmt`` template in
    web/modules, the chrome strings room_work.js writes as plain text, and the server's reason
    sentences the web reads by their code (gateway/work_status.py REASON_TEXT; a reason whose text
    is the host's own words, a Project hold's label, stays host text). Each translation is the
    English wrapped in ‹…›, so any chrome still in plain English is a string no seam translated."""
    from ouroboros.gateway.work_status import REASON_TEXT

    entries = {f"code:work.because.{code}": {"text": MARK.format(phrase)} for code, phrase in REASON_TEXT.items()
               if isinstance(phrase, str) and code not in {"project_admission", "held"}}
    for path in sorted((REPO / "web" / "modules").glob("*.js")):
        source = path.read_text(encoding="utf-8")
        for match in _TR.finditer(source):
            entries["code:" + match.group(2)] = {"text": MARK.format(_unescape(match.group(4)))}
        for match in _FMT.finditer(source):
            template = _unescape(match.group(2))
            if "${" not in template:
                entries.setdefault(template, {"text": MARK.format(template)})
    for key in OVERLAY_KEYS:
        entries[key] = {"text": MARK.format(key)}
    return {"language": "ru", "english": False, "revision": 7, "entries": entries, "languages": []}


# Chrome text in one room's control, list, Back and live region that no seam translated.
LEAKS = r"""(scope) => {
    const root = scope === 'main' ? document.querySelector('#page-chat') : document.querySelector('#project-panel');
    const leaks = [], seen = [];
    const lettered = (text) => (String(text || '').match(/\p{L}/gu) || []).length >= 2;
    const note = (where, text) => {
        const value = String(text || '').replace(/\s+/g, ' ').trim();
        if (!value) return;
        seen.push({ where, text: value });
        if (lettered(value) && !value.includes('‹')) leaks.push({ where, text: value });
    };
    const shown = (el) => Boolean(el) && !el.hidden && el.getClientRects().length > 0;
    const trigger = root?.querySelector('.rw-trigger');
    const surface = trigger ? document.getElementById(trigger.getAttribute('aria-controls')) : null;  // the room's work Shade, never the chip's
    const waitNote = trigger?.querySelector('.rw-sr');
    if (waitNote && !waitNote.hidden) note('trigger .rw-sr (wait note)', waitNote.textContent);
    note('surface[aria-label]', surface?.getAttribute('aria-label'));
    for (const el of surface?.querySelectorAll('.rw-group, .rw-note') || []) if (shown(el)) note(el.className, el.textContent);
    for (const rowEl of surface?.querySelectorAll('.rw-row') || []) {
        const id = rowEl.dataset.id;
        for (const sel of ['.rw-word', '.rw-secondary', '.rw-pill', '.rw-detail-label',
                           '.rw-detail-line > span:not(.rw-detail-label)']) {
            for (const el of rowEl.querySelectorAll(sel)) if (shown(el)) note(`${id} ${sel}`, el.textContent);
        }
        const main = rowEl.querySelector('.rw-main');
        if (main?.hasAttribute('aria-label')) note(`${id} .rw-main[aria-label]`, main.getAttribute('aria-label'));
        for (const el of rowEl.querySelectorAll('.rw-more, .rw-pill-question')) {
            if (el.hasAttribute('aria-label')) note(`${id} ${el.className.split(' ').pop()}[aria-label]`, el.getAttribute('aria-label'));
        }
        // A composed name (no card, no coined name) is chrome as well.
        const name = (rowEl.querySelector('.rw-name')?.textContent || '').trim();
        if (/^(Reply to “|Task started |Turn started |Task$|Turn in this chat$|Reply to an earlier message$)/.test(name)) {
            leaks.push({ where: `${id} .rw-name (composed)`, text: name });
        }
    }
    const back = root?.querySelector('.rw-back');
    if (shown(back)) note('.rw-back', back.textContent);
    for (const el of root?.querySelectorAll('.rw-sr[role="status"]') || []) note('live region', el.textContent);
    return { leaks, seen, lang: document.documentElement.lang };
}"""

TARGETS = """(scope) => {
    const root = scope === 'main' ? document.querySelector('#page-chat') : document.querySelector('#project-panel');
    const min = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--touch-target-min')) || 0;
    const visible = (el) => Boolean(el) && !el.hidden && el.getClientRects().length > 0 && getComputedStyle(el).visibility !== 'hidden';
    const box = (el) => { const r = el.getBoundingClientRect(); return { w: Math.round(r.width * 10) / 10, h: Math.round(r.height * 10) / 10 }; };
    const of = (sel, kind) => [...root.querySelectorAll(sel)].filter(visible)
        .map((el) => ({ kind, id: el.closest('.rw-row')?.dataset.id || '', ...box(el) }));
    const scrollLatest = root.querySelector('.chat-scroll-bottom-btn');
    return { coarse: matchMedia('(pointer: coarse)').matches, hoverNone: matchMedia('(hover: none)').matches, min,
        targets: [...of('.rw-trigger', 'trigger'), ...of('.rw-main[tabindex="0"]', 'row'), ...of('.rw-more', 'more'),
                  ...of('.rw-pill', 'pill')],
        back: visible(root.querySelector('.rw-back')) ? box(root.querySelector('.rw-back')) : null,
        scrollLatest: scrollLatest ? box(scrollLatest) : null };
}"""

NAMES = """(scope) => {
    const root = scope === 'main' ? document.querySelector('#page-chat') : document.querySelector('#project-panel');
    return [...root.querySelectorAll('.rw-surface:not([hidden]) .rw-row')].map((row) => {
        const name = row.querySelector('.rw-name');
        const line = parseFloat(getComputedStyle(name).lineHeight);
        const r = name.getBoundingClientRect(), rr = row.getBoundingClientRect();
        return { id: row.dataset.id, h: Math.round(r.height * 10) / 10, line, reading: row.dataset.reading || '',
                 open: row.dataset.open, rowOverflow: row.scrollWidth - row.clientWidth,
                 nameOverflow: name.scrollWidth - name.clientWidth, left: rr.left, right: rr.right, vw: innerWidth,
                 chars: name.textContent.length };
    });
}"""

SHADE_GEOMETRY = """() => { const box = s => { const r = document.querySelector(s).getBoundingClientRect();
        return {x: r.x, y: r.y, right: r.right, bottom: r.bottom}; };
    const shade = document.querySelector('#project-panel .rw-shade:not(.cx-shade)');
    const reachable = [...shade.querySelectorAll('.rw-more, .rw-main[tabindex="0"], .rw-pill')].filter((el) => el.getClientRects().length)
        .every((el) => { const r = el.getBoundingClientRect(); const hit = document.elementFromPoint(r.x + r.width / 2, r.y + r.height / 2);
            return Boolean(hit) && (hit === el || el.contains(hit)); });
    return {bar: box('#project-panel .project-panel-bar'), shade: box('#project-panel .rw-shade:not(.cx-shade)'),
            capsule: box('#project-panel .app-page-title-row'),
            panel: box('#project-panel'), vw: innerWidth, vh: innerHeight, reachable,
            overflow: document.documentElement.scrollWidth - innerWidth}; }"""

def leaks(page, scope):
    """LEAKS after the overlay's own frame-batched pass over the latest mutation has run."""
    rw.settle_frames(page)
    page.wait_for_timeout(120)
    rw.settle_frames(page)
    return page.evaluate(LEAKS, scope)


# The census's own hold sentence (``project_admission_hold.label``) is shown verbatim by the shared
# presenter (task_phase_chip.js), as on the card: host text from before this change, recorded apart.
HOST_LABELS = {"Waiting for Project access"}


UNBROKEN = "docs/architecture/03-web-ui-pages-and-buttons.md#room-work-status-back-to-reading-and-the-shade-under-the-bar"
LONG_ASK = ("Пожалуйста, проверь, какие источники должен цитировать еженедельный отчёт, и спроси меня, прежде чем "
            "выбрать — bitte auch alle Übersetzungsvarianten der Zitate prüfen")


@pytest.mark.parametrize("theme", ["dark", "light"])
@pytest.mark.parametrize("width", [320, 390])
def test_translated_chrome_long_names_coarse_pointer(branch_ui, width, theme):
    ui = branch_ui
    page = ui["new_page"](viewport={"width": width, "height": 844}, has_touch=True)
    history = {1: [*rw.reading(10), rw.owner(LONG_ASK, "cm-ask", 30),
                   rw.progress("rw-live", "Comparing the cited pages.", 31, suggested_name=rw.LONG_NAME),
                   rw.progress("rw-paused", "Collecting appendix evidence.", 32, suggested_name=f"Сверить {UNBROKEN}"),
                   *rw.reading(4, 35, "Later")],
               42: [*rw.reading(8, 0, "Project"),
                    rw.progress("p-live", "Reading the archive.", 30, suggested_name=rw.LONG_NAME)]}
    transport = BranchTransport(page, i18n=pseudo_memory(), history=history, projects=[rw.PROJECT], details={
        "rw-far": {"task_id": "rw-far", "status": "running", "suggested_name": f"Индексировать {UNBROKEN}"}},
        activities=[rw.activity("rw-live"), rw.activity("rw-paused", "budget_paused", pause_cause="owner"),
                    rw.activity("rw-far", "queued"),
                    rw.activity("rw-ask", kind="direct_chat", client_message_id="cm-ask", **rw.waited("q-main")),
                    rw.activity("p-live", chat_id=42, project_id=rw.PROJECT["id"]),
                    rw.activity("p-hold", "queued", chat_id=42, project_id=rw.PROJECT["id"],
                                project_admission_hold={"label": "Waiting for Project access"})])
    rw.open_app({**ui, "page": page}, transport, theme=theme)
    page.wait_for_function("() => document.documentElement.lang === 'ru'", timeout=15_000)
    transport.refresh()
    tag = f"{width}-{theme}"
    problems, observations, out = [], [], {}

    # Closed: Main's bar at this width.
    rw.wait_facts(page, "main", lambda f: not f["disabled"], "an enabled control")
    page.locator("#chat-messages").evaluate("el => { el.scrollTop = Math.round(el.scrollHeight * 0.5); }")
    page.wait_for_timeout(250)
    out["closed_geometry"] = page.evaluate(rw._BAR_GEOMETRY, "main")
    out["closed_targets"] = page.evaluate(TARGETS, "main")
    rw.capture(page, f"coarse-{tag}-main-closed")
    if not out["closed_targets"]["coarse"]:
        problems.append("the engine did not report (pointer: coarse) with a touch context")

    # Open by a tap: rows, names, targets, translated chrome.
    rw.trigger(page).tap()
    opened = rw.wait_facts(page, "main", lambda f: len(f["rows"]) == 4 and (rw.row(f, "rw-far") or {}).get("name", "")
                           .startswith("Индексировать"), "four named rows")
    rw.settle_unfold(page, "main")
    out["open_facts"] = opened
    out["open_geometry"] = page.evaluate(rw._BAR_GEOMETRY, "main")
    out["open_targets"] = page.evaluate(TARGETS, "main")
    out["open_names"] = page.evaluate(NAMES, "main")
    out["open_leaks"] = leaks(page, "main")
    rw.capture(page, f"coarse-{tag}-main-open")
    geometry = out["open_geometry"]
    shade, bar = geometry["shade"], geometry["bar"]
    if not (shade and shade["x"] >= -0.5 and shade["x"] + shade["w"] <= geometry["vw"] + 0.5
            and shade["bottom"] <= geometry["vh"] + 0.5):
        problems.append(f"the Shade leaves the viewport: {shade} vw={geometry['vw']} vh={geometry['vh']}")
    capsule = geometry["capsule"]
    if shade and (abs((shade["y"] - capsule["bottom"]) - 8) > 1.5 or abs(shade["x"] - capsule["x"]) > 1.5):
        problems.append(f"the work list does not hang under its status capsule: {shade} {capsule} (bar {bar})")
    if geometry["overflow"] > 0:
        problems.append(f"page overflows horizontally by {geometry['overflow']}px")
    if not geometry["reachable"]:
        problems.append("a row target is not the element hit at its centre")
    moved = [(a, b) for a, b in zip(out["closed_geometry"]["neighbours"], geometry["neighbours"])
             if any(abs(a[k] - b[k]) > 0.5 for k in ("x", "y", "w", "h"))]
    if moved:
        problems.append(f"opening moved a neighbouring target: {moved}")
    small = [t for t in out["open_targets"]["targets"]
             if t["kind"] in {"trigger", "more", "pill"} and min(t["w"], t["h"]) < out["open_targets"]["min"] - 0.5
             or t["kind"] == "row" and t["h"] < out["open_targets"]["min"] - 0.5]
    if small:
        problems.append(f"targets under --touch-target-min {out['open_targets']['min']}px: {small}")
    for name in out["open_names"]:
        if name["h"] > name["line"] * 2 + 1:
            problems.append(f"{name['id']}: name not clamped to two lines at rest ({name})")
        if name["rowOverflow"] > 1 or name["nameOverflow"] > 1 or name["right"] > name["vw"] + 0.5:
            problems.append(f"{name['id']}: a long name overflows its row ({name})")
    for leak in out["open_leaks"]["leaks"]:
        problems.append(f"untranslated chrome (open list): {leak}")

    # Details of the card-less row by a tap: translated labels, the whole name.
    rw.row_main(page, "rw-far").tap()
    rw.wait_facts(page, "main", lambda f: (rw.row(f, "rw-far") or {}).get("open"), "details by a tap")
    out["detail_names"] = page.evaluate(NAMES, "main")
    out["detail_leaks"] = leaks(page, "main")
    rw.capture(page, f"coarse-{tag}-main-details")
    far = next(n for n in out["detail_names"] if n["id"] == "rw-far")
    if far["h"] <= far["line"] * 2 + 1 and far["chars"] > 60:
        observations.append(f"open details did not show the whole long name: {far}")
    for leak in out["detail_leaks"]["leaks"]:
        if leak not in out["open_leaks"]["leaks"]:
            problems.append(f"untranslated chrome (details): {leak}")

    # The name goes to the card; Back to reading at this width, by a tap.
    rw.row_main(page, "rw-live").tap()
    rw.wait_facts(page, "main", lambda f: f["surfaceHidden"] and f["back"], "Back to reading after a tap")
    out["back_targets"] = page.evaluate(TARGETS, "main")
    out["back_leaks"] = leaks(page, "main")
    rw.capture(page, f"coarse-{tag}-main-back")
    back = out["back_targets"]["back"]
    if back and back["h"] < out["back_targets"]["min"] - 0.5:
        problems.append(f"Back to reading is {back['h']}px tall under a coarse pointer (touch minimum "
                            f"{out['back_targets']['min']}px; Scroll to latest {out['back_targets']['scrollLatest']})")
    for leak in out["back_leaks"]["leaks"]:
        if leak not in out["open_leaks"]["leaks"]:
            problems.append(f"untranslated chrome (Back): {leak}")
    page.locator("#page-chat .rw-back").tap()
    page.wait_for_timeout(400)
    out["after_back_leaks"] = leaks(page, "main")
    for leak in out["after_back_leaks"]["leaks"]:
        if leak not in out["open_leaks"]["leaks"]:
            problems.append(f"untranslated chrome (after Back): {leak}")

    # The Project's Shade at this width.
    rw.open_project(page, rw.PROJECT)
    rw.wait_facts(page, "project", lambda f: not f["disabled"], "the Project's control")
    rw.capture(page, f"coarse-{tag}-project-closed")
    rw.trigger(page, "project").tap()
    rw.wait_facts(page, "project", lambda f: len(f["rows"]) == 2, "two Project rows")
    rw.settle_unfold(page, "project")
    out["shade"] = page.evaluate(SHADE_GEOMETRY)
    out["shade_targets"] = page.evaluate(TARGETS, "project")
    out["shade_names"] = page.evaluate(NAMES, "project")
    out["shade_leaks"] = leaks(page, "project")
    rw.capture(page, f"coarse-{tag}-project-open")
    shade = out["shade"]
    if (abs((shade["shade"]["y"] - shade["capsule"]["bottom"]) - 8) > 1.5
            or abs(shade["shade"]["x"] - shade["capsule"]["x"]) > 1.5):
        problems.append(f"the work list does not hang under its status capsule: {shade}")
    if shade["shade"]["x"] < -0.5 or shade["shade"]["right"] > shade["vw"] + 0.5 or shade["overflow"] > 0:
        problems.append(f"the Shade leaves the viewport: {shade}")
    if not shade["reachable"]:
        problems.append("a Shade target is not the element hit at its centre")
    small = [t for t in out["shade_targets"]["targets"]
             if t["kind"] in {"trigger", "more", "pill"} and min(t["w"], t["h"]) < out["shade_targets"]["min"] - 0.5]
    if small:
        problems.append(f"Shade targets under --touch-target-min: {small}")
    for name in out["shade_names"]:
        if name["h"] > name["line"] * 2 + 1 or name["rowOverflow"] > 1 or name["right"] > name["vw"] + 0.5:
            problems.append(f"{name['id']}: Shade name not clamped or overflowing ({name})")
    for leak in out["shade_leaks"]["leaks"]:
        if leak["text"] in HOST_LABELS:
            observations.append(f"inherited host text shown verbatim (census hold label): {leak}")
        else:
            problems.append(f"untranslated chrome (Project): {leak}")

    page.wait_for_timeout(1700)  # the miss reporter flushes after 1.5 s
    out["misses_reported"] = sorted({item.get("key", "") for item in transport.misses})
    out["problems"], out["observations"] = problems, observations
    out["posts"], out["console"] = transport.posts, transport.console
    record(f"coarse-{tag}", out)
    assert transport.decision_posts() == [] and transport.posts == []
    assert not problems, problems


def test_open_sheet_repaints_translated_chrome_without_replacing_focus(branch_ui):
    page = branch_ui['page']
    transport = BranchTransport(page, history=_host_history(), activities=_host_activities(), details={
        'rw-far': {'task_id': 'rw-far', 'status': 'running', 'suggested_name': 'Index the archived notes'}})
    rw.open_app(branch_ui, transport)
    rw.trigger(page).focus()
    page.keyboard.press('Enter')
    rw.wait_facts(page, 'main', _all_offered, 'four actionable rows')
    page.locator('#page-chat .rw-pill-question').focus()
    page.evaluate('() => { window.__languageFocusNode = document.activeElement; }')
    for payload, language, question in [
        (pseudo_memory(), 'ru', '‹View question›'),
        ({'language': 'en', 'english': True, 'revision': 8, 'entries': {}}, 'en', 'View question'),
    ]:
        transport.i18n = payload
        transport.send({'type': 'ui_language_changed'})
        page.wait_for_function('(lang) => document.documentElement.lang === lang', arg=language)
        rw.settle_frames(page)
        page.wait_for_function('(label) => document.querySelector("#page-chat .rw-pill-question").textContent === label', arg=question)
        assert page.evaluate('() => document.activeElement === window.__languageFocusNode'), 'language repaint replaced focus'
        expected = '‹Work in this chat›' if language == 'ru' else 'Work in this chat'
        assert page.locator('#page-chat .rw-surface:not(.cx-shade)').get_attribute('aria-label') == expected
        assert page.locator('#page-chat .rw-pill-resume').inner_text() == ('‹Resume›' if language == 'ru' else 'Resume')
    record('language-switch', {'focus': page.evaluate(ACTIVE), 'posts': transport.posts})
    assert transport.posts == [] and transport.decision_posts() == []


# ------------------------------------------------------------------------------------------------
# Sending… hands its slot to the turn with its exact client message id.

def _send(page, transport, text):
    before = len(transport.chat_frames())
    page.locator("#chat-input").fill(text)
    page.locator("#page-chat .chat-send-inline").first.click()
    deadline = time.monotonic() + 10
    while len(transport.chat_frames()) <= before and time.monotonic() < deadline:
        page.wait_for_timeout(50)
    frame = next(frame for frame in transport.chat_frames() if frame.get("content") == text)
    return frame["client_message_id"]


def test_pending_send_hands_its_slot_to_the_turn_with_its_exact_client_id(branch_ui):
    page = branch_ui["page"]
    history = {1: [*rw.reading(6), rw.progress("rw-live", "Comparing the cited pages.", 30,
                                               suggested_name="Compare the cited pages")]}
    transport = BranchTransport(page, history=history, activities=[rw.activity("rw-live")])
    rw.open_app(branch_ui, transport)
    rw.wait_facts(page, "main", lambda f: not f["disabled"], "an enabled control")
    first = _send(page, transport, "RW-SEND-A: compare the first appendix")
    second = _send(page, transport, "RW-SEND-B: compare the second appendix")
    out = {"cmids": {"A": first, "B": second}, "frames": transport.chat_frames()}
    problems = []

    # A keyboard open; the reader goes to B's row.
    rw.trigger(page).focus()
    page.keyboard.press("Enter")
    opened = rw.wait_facts(page, "main", lambda f: rw.row(f, f"pending:{second}") and rw.row(f, f"pending:{first}")
                           and f["active"] and f["active"]["inSurface"], "both sends listed, focus in the list")
    out["opened"] = opened
    for _ in range(4):
        if rw.facts(page)["active"]["row"] == f"pending:{second}":
            break
        page.keyboard.press("ArrowDown")
        page.wait_for_timeout(80)
    before = rw.facts(page)
    out["before"] = before
    order = [item["id"] for item in before["rows"]]
    pending_b = rw.row(before, f"pending:{second}")
    assert pending_b["word"] == "Sending…" and pending_b["name"].startswith("Reply to “RW-SEND-B"), pending_b
    assert not pending_b["more"] and not pending_b["resume"], "an unconfirmed send offers no controls"
    if before["active"]["row"] != f"pending:{second}":
        problems.append(f"the keyboard could not reach B's row: {before['active']}")
    page.evaluate("() => { window.__pendingFocusNode = document.activeElement; }")

    # The census lists B's turn by its exact client id: it takes B's place, nothing else moves.
    page.evaluate(ROWS_RECORDER, ["main", 1500])
    transport.activities.append(rw.activity("turn-b", kind="direct_chat", client_message_id=second))
    transport.refresh()
    handed = rw.wait_facts(page, "main", lambda f: rw.row(f, "turn-b") and not rw.row(f, f"pending:{second}"),
                           "B's turn in B's slot")
    page.wait_for_timeout(300)
    handed = rw.facts(page)
    out["handed"], out["handed_frames"] = handed, frames_now(page)
    expected = [("turn-b" if key == f"pending:{second}" else key) for key in order]
    if [item["id"] for item in handed["rows"]] != expected:
        problems.append(f"the hand-off moved rows: {order} -> {[item['id'] for item in handed['rows']]}")
    turn = rw.row(handed, "turn-b")
    if turn["name"] != pending_b["name"]:
        problems.append(f"the turn renamed the send: {pending_b['name']!r} -> {turn['name']!r}")
    if (turn["word"], turn["tone"]) != ("Working", "live"):
        problems.append(f"the turn's own words: {turn}")
    if rw.row(handed, f"pending:{first}")["word"] != "Sending…":
        problems.append("the other send changed with B's turn")
    if any(frame["rows"] and len({r["id"] for r in frame["rows"]}) != len(frame["rows"]) for frame in out["handed_frames"]):
        problems.append("a frame listed a row twice")
    if any(any(r["id"] == f"pending:{second}" for r in frame["rows"]) and any(r["id"] == "turn-b" for r in frame["rows"])
           for frame in out["handed_frames"]):
        problems.append("a frame showed both B's send and B's turn")
    active = handed["active"] or {}
    out["focus_after_handoff"] = page.evaluate(ACTIVE)
    assert page.evaluate("() => document.activeElement === window.__pendingFocusNode"), "handoff rebuilt the focused node"
    if active.get("row") != "turn-b":
        problems.append(f"keyboard focus did not hold in B's slot through the hand-off: {out['focus_after_handoff']}")

    # Another client's turn appends; A's own turn takes A's place.
    transport.activities.append(rw.activity("turn-x", kind="direct_chat", client_message_id="msg-other-tab"))
    transport.refresh()
    other = rw.wait_facts(page, "main", lambda f: rw.row(f, "turn-x"), "another client's turn")
    if [item["id"] for item in other["rows"]][-1] != "turn-x" or not rw.row(other, f"pending:{first}"):
        problems.append(f"another client's turn took a slot: {[item['id'] for item in other['rows']]}")
    transport.activities.append(rw.activity("turn-a", kind="direct_chat", client_message_id=first))
    transport.refresh()
    both = rw.wait_facts(page, "main", lambda f: rw.row(f, "turn-a") and not rw.row(f, f"pending:{first}"), "A's turn")
    out["both"] = both
    expected_a = [("turn-a" if key == f"pending:{first}" else key) for key in [item["id"] for item in other["rows"]]]
    if [item["id"] for item in both["rows"]] != expected_a:
        problems.append(f"A's hand-off moved rows: {[item['id'] for item in both['rows']]} != {expected_a}")
    rw.capture(page, "pending-handoff-open")
    page.keyboard.press("Escape")
    rw.wait_facts(page, "main", lambda f: f["surfaceHidden"], "closed")
    rw.trigger(page).click()
    reopened = rw.wait_facts(page, "main", lambda f: not f["surfaceHidden"] and len(f["rows"]) == 4, "four rows again")
    out["reopened"] = reopened
    if any(item["kind"] == "pending" for item in reopened["rows"]):
        problems.append("a confirmed send still listed as Sending…")
    out["problems"] = problems
    record("pending-handoff", out)
    assert transport.decision_posts() == [] and transport.posts == []
    assert not problems, problems


# ------------------------------------------------------------------------------------------------
# Back to reading: by keyboard, and after its passage left the feed.

MARK_VISIBLE = """() => { const feed = document.querySelector('#chat-messages'); const v = feed.getBoundingClientRect();
    const nodes = [...feed.children].filter((n) => { const r = n.getBoundingClientRect(); return r.bottom > v.top && r.top < v.bottom; });
    nodes.forEach((n) => n.setAttribute('data-qa-reading', '1'));
    return nodes.map((n) => ({ cls: String(n.className || ''), history: n.dataset.historyId || '', cmid: n.dataset.clientMessageId || '' })); }"""
# Retire the nodes the reader was reading the way the feed retires a message node: it leaves the DOM.
EVICT = """() => { const feed = document.querySelector('#chat-messages');
    const marked = [...feed.querySelectorAll('[data-qa-reading="1"]')];
    const ids = new Set(marked.map((n) => n.dataset.historyId).filter(Boolean));
    const gone = new Set(marked);
    for (const n of feed.querySelectorAll('[data-history-id]')) if (ids.has(n.dataset.historyId)) gone.add(n.closest('#chat-messages > *') || n);
    gone.forEach((n) => n.remove());
    return gone.size; }"""


@pytest.mark.parametrize("passage", ["kept", "evicted", "evicted-away"])
def test_back_to_reading_by_keyboard_and_after_its_passage_left(branch_ui, passage):
    page = branch_ui["page"]
    history = {1: [*rw.reading(14), rw.progress("rw-live", "Comparing the cited pages.", 30,
                                                suggested_name="Compare the cited pages"), *rw.reading(3, 31, "Later")]}
    transport = BranchTransport(page, history=history, activities=[rw.activity("rw-live")])
    rw.open_app(branch_ui, transport, viewport={"width": 1100, "height": 722})
    feed = page.locator("#chat-messages")
    feed.evaluate("el => { el.scrollTop = Math.round(el.scrollHeight * 0.3); }")
    page.wait_for_timeout(300)
    reading_at = feed.evaluate("el => el.scrollTop")
    out = {"reading_at": reading_at, "marked": page.evaluate(MARK_VISIBLE)}
    problems, observations = [], []
    rw.trigger(page).focus()
    page.keyboard.press("Enter")
    rw.wait_facts(page, "main", lambda f: f["active"] and f["active"]["row"] == "rw-live", "keyboard in the row")
    page.keyboard.press("Enter")
    landed = rw.wait_facts(page, "main", lambda f: f["surfaceHidden"] and f["back"], "the card and Back to reading")
    out["landed"] = {"back": landed["back"], "focus": page.evaluate(ACTIVE), "scroll": feed.evaluate("el => el.scrollTop")}
    if passage.startswith("evicted"):
        out["evicted_nodes"] = page.evaluate(EVICT)
        page.wait_for_timeout(150)
    page.locator("#page-chat .rw-back").focus()
    page.keyboard.press("Enter")
    page.wait_for_timeout(350)
    after = rw.facts(page)
    out["after"] = {"back": after["back"], "focus": page.evaluate(ACTIVE), "scroll": feed.evaluate("el => el.scrollTop"),
                    "live": page.evaluate("() => document.querySelector('#page-chat .rw-sr[role=status]')?.textContent || ''")}
    rw.capture(page, f"back-{passage}-after-press")
    if passage == "kept":
        if abs(out["after"]["scroll"] - reading_at) > 40:
            problems.append(f"Back did not return to the passage: {reading_at} -> {out['after']['scroll']}")
        if out["after"]["live"] != "Returned to where you were reading." or out["after"]["back"]:
            problems.append(f"the return was not said or Back stayed: {out['after']}")
        if out["after"]["focus"]["where"] == "body":
            problems.append("Back to reading by keyboard dropped focus to the document")
    else:
        out["after_4s"] = None
        if out["after"]["back"] != "Earlier place is no longer loaded":
            observations.append(f"Back after eviction said {out['after']['back']!r} / {out['after']['live']!r}")
        if not out["after"]["live"]:
            problems.append("the reader was not told the earlier place is gone")
        if passage == "evicted-away":
            page.locator('#chat-input').focus()
        page.wait_for_timeout(4300)
        later = rw.facts(page)
        out["after_4s"] = {"back": later["back"], "focus": page.evaluate(ACTIVE)}
        rw.capture(page, "back-evicted-after-4s")
        if later["back"]:
            problems.append("the Earlier place note stayed beyond its time")
        if out["after"]["focus"]["where"] == "body" or out["after_4s"]["focus"]["where"] == "body":
            problems.append(f"Back after eviction dropped keyboard focus to the document: {out['after']['focus']} / "
                            f"{out['after_4s']['focus']}")
        if passage == "evicted-away":
            assert page.locator('#chat-input').evaluate('el => document.activeElement === el'), "late notice stole focus"
    out["problems"], out["observations"] = problems, observations
    record(f"back-{passage}", out)
    assert transport.decision_posts() == [] and transport.posts == []
    assert not problems, problems


# ------------------------------------------------------------------------------------------------
# A host that is not ready: every control withdrawn and said, at boot and while open.

def _host_history():
    return {1: [*rw.reading(6), rw.owner(rw.ASK_TEXT, "cm-ask", 30),
                rw.progress("rw-live", "Comparing the cited pages.", 31, suggested_name="Compare the cited pages"),
                rw.progress("rw-paused", "Collecting appendix evidence.", 32, suggested_name="Collect appendix evidence")]}


def _host_activities():
    return [rw.activity("rw-live"), rw.activity("rw-paused", "budget_paused", pause_cause="owner"), rw.activity("rw-far"),
            rw.activity("rw-ask", kind="direct_chat", client_message_id="cm-ask", **rw.waited("q-main"))]


def _all_withdrawn(current):
    return all(not (item["more"] or item["resume"] or item["question"]) and item["tone"] == "unknown"
               for item in current["rows"])


def _all_offered(current):
    by = {item["id"]: item for item in current["rows"]}
    return (len(by) == 4 and by["rw-live"]["more"] and by["rw-paused"]["resume"] and by["rw-far"]["more"]
            and by["rw-ask"]["question"])


@pytest.mark.parametrize("when", ["boot", "while-open"])
def test_host_not_ready_withdraws_every_control_and_says_starting(branch_ui, when):
    page = branch_ui["page"]
    transport = BranchTransport(page, history=_host_history(), activities=_host_activities(), details={
        "rw-far": {"task_id": "rw-far", "status": "running", "suggested_name": "Index the archived notes"}})
    transport.ready = when != "boot"
    rw.open_app(branch_ui, transport)
    out, problems = {}, []
    if when == "boot":
        closed = rw.wait_facts(page, "main", lambda f: f["exists"], "the control")
        out["closed"] = closed
        if closed["disabled"]:
            out["observation"] = "the control is disabled while the host is starting"
        else:
            rw.trigger(page).click()
        starting = rw.wait_facts(page, "main", lambda f: closed["disabled"] or len(f["rows"]) == 4, "four rows")
    else:
        rw.trigger(page).click()
        rw.wait_facts(page, "main", _all_offered, "every control offered")
        assert open_menu(page, "rw-live") == FULL_MENU
        page.evaluate(ROWS_RECORDER, ["main", 2000])
        transport.ready = False
        transport.refresh()
        starting = rw.wait_facts(page, "main", lambda f: f["note"] == "Ouroboros is starting…", "the starting note")
        page.wait_for_timeout(300)
        starting = rw.facts(page)
        out["frames"] = frames_now(page)
        if [item["id"] for item in starting["rows"]] != [item["id"] for item in out["frames"][0]["rows"]]:
            problems.append("order changed when the host stopped being ready")
        if starting["menu"] is not None:
            problems.append(f"the open menu outlived the host's readiness: {starting['menu']}")
    out["starting"] = starting
    rw.capture(page, f"host-not-ready-{when}")
    if not starting["disabled"]:
        if not _all_withdrawn(starting):
            problems.append(f"a control offered while the host is not ready: {starting['rows']}")
        if starting["note"] != "Ouroboros is starting…":
            problems.append(f"the list did not say the host is starting: {starting['note']!r}")
    transport.ready = True
    transport.refresh()
    if rw.facts(page)["surfaceHidden"]:
        rw.trigger(page).click()
    ready = rw.wait_facts(page, "main", lambda f: _all_offered(f) and not f["note"], "controls back once ready")
    out["ready"] = ready
    out["problems"] = problems
    record(f"host-not-ready-{when}", out)
    assert transport.decision_posts() == [] and transport.posts == []
    assert not problems, problems


# ------------------------------------------------------------------------------------------------
# Subagent and reusable ids.

def test_subagent_and_reusable_ids_get_no_row_or_no_controls(branch_ui):
    page = branch_ui["page"]
    history = {1: [*rw.reading(6), rw.progress("rw-root", "Coordinating the reviewers.", 30,
                                               suggested_name="Coordinate the reviewers"),
                   rw.progress("rw-sub", "Reviewing chapter two.", 31, delegation_role="subagent",
                               parent_task_id="rw-root", subagent_role="reviewer"),
                   rw.owner("A question asked in the reusable slot", "cm-active", 32)],
               42: [*rw.reading(3, 0, "Project"),
                    {**rw.progress("p-root", "Finished the archive pass.", 20, suggested_name="Archive pass"),
                     "task_terminal_status": "completed"},
                    {**rw.progress("p-sub", "Reviewed the archive pass.", 25, delegation_role="subagent",
                                   parent_task_id="p-root", subagent_role="reviewer"), "task_terminal_status": "completed"}]}
    transport = BranchTransport(page, history=history, projects=[rw.PROJECT], details={
        "p-root": {"task_id": "p-root", "status": "completed", "suggested_name": "Archive pass"},
        "p-sub": {"task_id": "p-sub", "status": "completed", "delegation_role": "subagent", "parent_task_id": "p-root"}},
        # The census names a child with its lineage, as the server's census does: it is a member of its root's work.
        activities=[rw.activity("rw-root"), rw.activity("rw-sub", parent_task_id="rw-root", root_task_id="rw-root"),
                    rw.activity("active", kind="direct_chat", client_message_id="cm-active")])
    rw.open_app(branch_ui, transport)
    cards = page.evaluate("""() => [...document.querySelectorAll('#chat-messages .chat-live-card')].map((c) => ({
        id: c.dataset.taskId, subagent: c.classList.contains('subagent') || c.dataset.subagent === '1' }))""")
    rw.trigger(page).click()
    listed = rw.wait_facts(page, "main", lambda f: rw.row(f, "rw-root") and rw.row(f, "active"), "the root and the slot")
    page.wait_for_timeout(300)
    listed = rw.facts(page)
    out = {"cards": cards, "listed": listed}
    problems = []
    if not any(card["id"] == "rw-sub" and card["subagent"] for card in cards):
        out["observation"] = "the history did not mint a subagent card for rw-sub; its exclusion rests on the census alone"
    if rw.row(listed, "rw-sub"):
        problems.append(f"a subagent got a row: {rw.row(listed, 'rw-sub')}")
    slot = rw.row(listed, "active")
    if slot["more"] or slot["resume"]:
        problems.append(f"the reusable slot offers controls: {slot}")
    if not rw.row(listed, "rw-root")["more"]:
        problems.append("the root lost its controls")
    rw.capture(page, "subagent-reusable-main")
    rw.open_project(page, rw.PROJECT)
    rw.trigger(page, "project").click()
    latest = rw.wait_facts(page, "project", lambda f: f["groups"] == ["Latest task"] and len(f["rows"]) == 1,
                           "the Project's latest root", timeout=30)
    out["project_latest"] = latest
    if latest["rows"][0]["id"] != "p-root":
        problems.append(f"the latest represented root is not the root: {latest['rows'][0]}")
    out["problems"] = problems
    record("subagent-reusable", out)
    assert transport.decision_posts() == [] and transport.posts == []
    assert not problems, problems


# ------------------------------------------------------------------------------------------------
# A pending cancel leaves only Stop now.

@pytest.mark.parametrize("subject", ["card", "cardless"])
def test_pending_cancel_leaves_only_stop_now_in_the_row_menu(branch_ui, subject):
    page = branch_ui["page"]
    task_id = "rw-live" if subject == "card" else "rw-far"
    history = {1: [*rw.reading(6), *([rw.progress("rw-live", "Comparing the cited pages.", 30,
                                                  suggested_name="Compare the cited pages")] if subject == "card" else [])]}
    transport = BranchTransport(page, history=history, activities=[rw.activity(task_id), rw.activity("rw-other")], details={
        task_id: {"task_id": task_id, "status": "running", "suggested_name": "Compare the cited pages"},
        "rw-other": {"task_id": "rw-other", "status": "running", "suggested_name": "Index the archived notes"}})
    rw.open_app(branch_ui, transport)
    rw.trigger(page).click()
    rw.wait_facts(page, "main", lambda f: (rw.row(f, task_id) or {}).get("more") and (rw.row(f, "rw-other") or {}).get("more")
                  and rw.row(f, "rw-other")["name"] == "Index the archived notes", "two rows with controls")
    out, problems, observations = {}, [], []
    assert open_menu(page, task_id) == FULL_MENU
    # The durable record says the stop is pending from the moment the host accepts it.
    transport.details[task_id] = {**transport.details[task_id], "cancel_state": "pending", "stop_policy": "finalize_then_cancel"}
    page.locator('body > .task-control-menu [data-task-control="finalize"]').click()
    rw.wait_facts(page, "main", lambda f: f["menu"] is None, "the menu closed after Wrap up")
    page.wait_for_timeout(600)
    out["after_wrap_up"] = rw.facts(page)
    out["posts_after_wrap_up"] = list(transport.posts)
    if [path for path, _ in transport.posts] != [f"/api/tasks/{task_id}/cancel"]:
        problems.append(f"Wrap up posted {transport.posts}")
    if out["after_wrap_up"]["surfaceHidden"]:
        observations.append("Wrap up from the row closed the list")
        rw.trigger(page).click()
    row = rw.row(rw.facts(page), task_id)
    out["row_after_wrap_up"] = row
    if subject == "card" and row["word"] != "Finalizing…":
        problems.append(f"the card row does not say Finalizing… after Wrap up: {row}")
    if not row["more"]:
        observations.append("the row withdrew its ⋯ after Wrap up (Stop now unreachable from the list)")
    else:
        again = open_menu(page, task_id)
        out["menu_again"] = again
        rw.capture(page, f"cancel-pending-{subject}-menu")
        if again != ["stop_now"]:
            problems.append(f"after Wrap up the row's menu offers {again}, not only Stop now")
        page.locator('body > .task-control-menu [data-task-control="stop_now"]').click()
        page.wait_for_timeout(600)
        out["posts_after_stop_now"] = list(transport.posts)
        stops = [body.get("stop_policy") or body for path, body in transport.posts if path.endswith(f"/{task_id}/cancel")]
        out["stop_bodies"] = stops
        if len(stops) != 2:
            problems.append(f"Stop now did not post a second cancel: {transport.posts}")
    # Another device's Wrap up while this row's menu is open: there is no census fact for it.
    other = open_menu(page, "rw-other")
    transport.details["rw-other"] = {**transport.details["rw-other"], "cancel_state": "pending",
                                     "stop_policy": "finalize_then_cancel"}
    transport.refresh()
    page.wait_for_timeout(500)
    out["other_menu_before"], out["other_menu_after_remote_wrap_up"] = other, menu_items(page)
    if menu_items(page):
        page.keyboard.press("Escape")
        rw.wait_facts(page, "main", lambda f: f["menu"] is None, "the other menu closed")
    if rw.facts(page)["surfaceHidden"]:
        rw.trigger(page).click()
        rw.wait_facts(page, "main", lambda f: not f["surfaceHidden"], "open again")
    fresh = open_menu(page, "rw-other")
    out["other_menu_reopened"] = fresh
    if fresh != ["stop_now"]:
        problems.append(f"a fresh menu after another device's Wrap up offers {fresh}")
    if out["other_menu_after_remote_wrap_up"] not in (None, ["stop_now"]):
        observations.append(f"an open card-less menu kept {out['other_menu_after_remote_wrap_up']} after another "
                            "device's Wrap up (no census fact carries a pending cancel)")
    page.keyboard.press("Escape")
    out["problems"], out["observations"], out["posts"] = problems, observations, list(transport.posts)
    record(f"cancel-pending-{subject}", out)
    assert transport.decision_posts() == []
    assert not problems, problems


# ------------------------------------------------------------------------------------------------
# Census reads that fail while the socket stays open; HTTP offline across a socket reopen.

@pytest.mark.parametrize("failure", ["http500", "network"])
def test_failed_census_reads_while_the_socket_stays_open(branch_ui, failure):
    """Observation: no document bounds how long a last census may authorize controls while every census
    read fails; this records what the list says and offers, and asserts only order, honesty of the
    surviving rows and recovery."""
    page = branch_ui["page"]
    history = {1: [*rw.reading(6), rw.progress("rw-live", "Comparing the cited pages.", 30,
                                               suggested_name="Compare the cited pages")]}
    transport = BranchTransport(page, history=history, activities=[rw.activity("rw-live"), rw.activity("rw-far")], details={
        "rw-far": {"task_id": "rw-far", "status": "running", "suggested_name": "Index the archived notes"}})
    rw.open_app(branch_ui, transport)
    rw.trigger(page).click()
    before = rw.wait_facts(page, "main", lambda f: all(item["more"] for item in f["rows"]) and len(f["rows"]) == 2
                           and rw.row(f, "rw-far")["name"] == "Index the archived notes", "two confirmed rows")
    out, problems = {"before": before}, []
    page.evaluate(ROWS_RECORDER, ["main", 9000])
    if failure == "http500":
        transport.state_mode = "http500"
    else:
        transport.http_offline = True
    started = time.monotonic()
    reads = transport.state_reads
    for _ in range(3):
        transport.send({"type": "projects_changed"})
        page.wait_for_timeout(2500)
    out["seconds_failing"] = round(time.monotonic() - started, 1)
    out["census_reads_while_failing"] = transport.state_reads - reads
    during = rw.facts(page)
    out["during"] = during
    out["frames"] = frames_now(page)
    rw.capture(page, f"census-{failure}-during")
    if [item["id"] for item in during["rows"]] != [item["id"] for item in before["rows"]]:
        problems.append("order changed while census reads failed")
    # Pressing the ⋯ during the failure: a card row opens at once; a card-less row reads its detail first.
    page.locator("#page-chat .rw-row[data-id='rw-far'] .rw-more").click() if rw.row(during, "rw-far")["more"] else None
    page.wait_for_timeout(800)
    out["cardless_press"] = {"menu": menu_items(page), "toasts": rw.toasts(page)}
    if menu_items(page):
        page.keyboard.press("Escape")
    page.locator("#page-chat .rw-row[data-id='rw-live'] .rw-more").click() if rw.row(rw.facts(page), "rw-live")["more"] else None
    page.wait_for_timeout(500)
    out["card_press"] = {"menu": menu_items(page)}
    if menu_items(page):
        page.keyboard.press("Escape")
    out["observation"] = {
        "controls_offered_through_failure": all(item["more"] for item in during["rows"]),
        "note_during_failure": during["note"], "trigger_tone": during["tone"], "badge": during["badge"]}
    transport.state_mode = ""
    transport.http_offline = False
    transport.refresh()
    if rw.facts(page)["surfaceHidden"]:
        rw.trigger(page).click()
    recovered = rw.wait_facts(page, "main", lambda f: all(item["more"] and item["tone"] in {"live", "queued"}
                                                        for item in f["rows"]) and not f["note"], "recovered")
    out["recovered"] = recovered
    out["problems"], out["posts"] = problems, list(transport.posts)
    record(f"census-{failure}", out)
    assert transport.posts == [] and transport.decision_posts() == []
    assert not problems, problems


def test_offline_http_across_a_socket_reopen_keeps_stale_actions_withdrawn(branch_ui):
    page = branch_ui["page"]
    transport = BranchTransport(page, history=_host_history(), activities=_host_activities(), details={
        "rw-far": {"task_id": "rw-far", "status": "running", "suggested_name": "Index the archived notes"}})
    rw.open_app(branch_ui, transport)
    rw.trigger(page).click()
    before = rw.wait_facts(page, "main", _all_offered, "every control offered")
    out, problems, observations = {"before": before}, [], []
    assert open_menu(page, "rw-live") == FULL_MENU
    page.evaluate(ROWS_RECORDER, ["main", 12000])
    transport.http_offline = True
    count = transport.disconnect()
    rw.wait_facts(page, "main", lambda f: f["note"].startswith("Reconnecting"), "the reconnecting note")
    transport.await_reconnect(count)
    page.wait_for_timeout(2500)
    reopened = rw.facts(page)
    out["reopened_http_offline"] = reopened
    out["frames"] = frames_now(page)
    out["state_reads_offline"] = transport.state_reads
    rw.capture(page, "offline-http-socket-reopened")
    offered = [item for item in reopened["rows"] if item["more"] or item["resume"] or item["question"]]
    if offered:
        problems.append(f"after the socket reopened with HTTP still offline a row offers a stale control: {offered}")
    if any(frame["menu"] for frame in out["frames"][1:]):
        problems.append("a menu was open while offline")
    if [item["id"] for item in reopened["rows"]] != [item["id"] for item in before["rows"]]:
        problems.append("order changed across the reopen")
    if reopened["note"] != "Reconnecting — showing the last known state.":
        observations.append(f"with the socket open but HTTP offline the list says {reopened['note']!r}")
    transport.http_offline = False
    transport.refresh()
    if rw.facts(page)["surfaceHidden"]:
        rw.trigger(page).click()
    recovered = rw.wait_facts(page, "main", lambda f: _all_offered(f) and not f["note"], "controls back with HTTP")
    out["recovered"] = recovered
    out["problems"], out["observations"], out["posts"] = problems, observations, list(transport.posts)
    record("offline-http-socket-reopen", out)
    assert transport.posts == [] and transport.decision_posts() == []
    assert not problems, problems


# ------------------------------------------------------------------------------------------------
# A root asleep on the helper it chose, which still awaits its own Resume: Waiting, and the line says so.

_SLEEP_ON_HELD = {"state": "waiting", "reason": "sleep", "task_attempt": 1,
                  "sleep": {"mode": "warm", "tasks": ["rw-child"], "senders": ["rw-child"]}}
_HELD_LINE = "Sleeping until it is woken · 1 helper it waits for is paused until Ouroboros resumes it"
# A Continue episode holds no request door for its predecessor's helper: no releaser is named.
_COLD_LINE = "Sleeping until it is woken · 1 helper it waits for is paused"
# Whether a status line shows whole where it is drawn: inside the viewport, nothing cut, nothing covering it.
_LINE_FIT = """(el) => {
    const r = el.getBoundingClientRect();
    const hit = document.elementFromPoint(r.left + Math.min(4, r.width / 2), r.top + r.height / 2);
    return { x: r.left, right: r.right, bottom: r.bottom, vw: innerWidth, vh: innerHeight,
             cut: el.scrollWidth > el.clientWidth + 1 || el.scrollHeight > el.clientHeight + 1,
             covered: !(hit === el || el.contains(hit)), lines: Math.round(r.height / parseFloat(getComputedStyle(el).lineHeight)) };
}"""


def _held_line(branch_ui, transport, task_id, tag, *, width, height, theme, work_id=None, line=_HELD_LINE):
    """Open the room, wait for the sleeper's card and its work's list row to say the held line; measure and
    capture both. A work is listed under its id: its first episode's, for a Continue."""
    work_id = work_id or task_id
    page = branch_ui["page"]
    rw.open_app(branch_ui, transport, theme=theme, viewport={"width": width, "height": height})
    bar = rw.wait_facts(page, "main", lambda f: f["badge"] == "Waiting", "the room says Waiting")
    card = page.locator(f'#chat-messages .chat-live-card[data-task-id="{task_id}"]')
    card.scroll_into_view_if_needed()
    page.wait_for_function("""([task, line]) => document.querySelector(
        `#chat-messages .chat-live-card[data-task-id="${task}"] [data-live-phase-secondary]`)?.textContent === line""",
                           arg=[task_id, line], timeout=20000)
    rw.settle_frames(page)
    chip = {"word": card.locator("[data-live-phase]").inner_text().strip(),
            "line": card.locator("[data-live-phase-secondary]").inner_text().strip(),
            "fit": card.locator("[data-live-phase-secondary]").evaluate(_LINE_FIT)}
    rw.capture(page, f"{tag}-card")
    rw.trigger(page).click()
    listed = rw.wait_facts(page, "main", lambda f: (rw.row(f, work_id) or {}).get("secondary"), "the sleeping row")
    rw.settle_frames(page)
    fit = page.locator(f'#page-chat .rw-row[data-id="{work_id}"] .rw-secondary').evaluate(_LINE_FIT)
    rw.capture(page, f"{tag}-list")
    record(tag, {"bar": bar, "chip": chip, "listed": listed, "row_line_fit": fit, "console": transport.console})
    for where, measured in (("card", chip["fit"]), ("list", fit)):
        assert not measured["cut"] and not measured["covered"], (where, measured)
        assert 0 <= measured["x"] and measured["right"] <= measured["vw"] + 0.5, (where, measured)
    assert (chip["word"], chip["line"]) == ("Waiting", line), chip
    return bar, listed


@pytest.mark.parametrize("theme", ["dark", "light"])
@pytest.mark.parametrize("width,height", [(1100, 722), (390, 844)])
def test_a_sleep_on_a_helper_awaiting_resume_says_so_on_card_and_list_without_a_door(branch_ui, width, height, theme):
    history = {1: [*rw.reading(8), rw.progress("rw-sleep", "Waiting for the builder it chose.", 30,
                                               suggested_name=rw.LONG_NAME)]}
    transport = BranchTransport(branch_ui["page"], history=history, activities=[
        rw.activity("rw-sleep"),
        rw.activity("rw-child", "budget_paused", pause_cause="unknown", can_resume=False,
                    parent_task_id="rw-sleep", root_task_id="rw-sleep")])
    transport.raw = {"rw-sleep": {"running": True, "owner_wait": _SLEEP_ON_HELD},
                     "rw-child": {"_budget_pause": {"reason": "owner", "exact_continuation": True}}}
    bar, listed = _held_line(branch_ui, transport, "rw-sleep", f"held-helper-{width}x{height}-{theme}",
                             width=width, height=height, theme=theme)
    row = rw.row(listed, "rw-sleep")
    assert (row["word"], row["secondary"], row["resume"]) == ("Waiting", _HELD_LINE, False), row
    assert rw.row(listed, "rw-child") is None, "a helper is a member of its root's work, never a row of its own"
    assert bar["mark"].startswith("still") and not bar["waitNote"], ("still dots, no owner's move", bar)
    assert transport.posts == [] and transport.decision_posts() == []


@pytest.mark.parametrize("theme", ["dark", "light"])
@pytest.mark.parametrize("width,height", [(1100, 722), (390, 844)])
def test_a_cold_sleep_of_a_continue_says_which_held_helper_of_the_earlier_episode_it_waits_for(
        branch_ui, width, height, theme):
    """The rows the real census gives a Continue episode cold-asleep on its predecessor's held helper
    (``test_current_stop_readers``): the saved selection rides on the exact pause; the ended predecessor's
    helpers are the owner's to select, the one the sleep chose is named on the line, the other is not."""
    history = {1: [*rw.reading(8), rw.progress("rw-cold", "Waiting for the builder of the first attempt.", 30,
                                               suggested_name=rw.LONG_NAME)]}
    transport = BranchTransport(branch_ui["page"], history=history, activities=[
        rw.activity("rw-cold", "budget_paused", pause_cause="sleep", can_resume=False),
        *(rw.activity(helper, "budget_paused", pause_cause="unknown", can_resume=True,
                      parent_task_id="rw-first", root_task_id="rw-first") for helper in ("rw-run", "rw-new"))])
    transport.raw = {
        "rw-cold": {"root_task_id": "rw-cold", "continuation": {"predecessor_task_id": "rw-first"}, "_budget_pause": {
            "reason": "sleep", "scope": "task", "rail": "model_sleep", "exact_continuation": True, "root_task_id": "rw-cold",
            "sleep": {"mode": "cold", "tasks": ["rw-run"], "senders": [], "runs": [], "wake_at": "", "any_mail": False}}},
        "rw-run": {"root_task_id": "rw-first", "parent_task_id": "rw-first", "_budget_pause": {
            "reason": "owner", "scope": "root", "rail": "owner_pause", "exact_continuation": True,
            "root_task_id": "rw-first"}},
        "rw-new": {"root_task_id": "rw-first", "parent_task_id": "rw-first", "_budget_pause_hold": {
            "reason": "root_fence_lifted_pending_selection", "release": "resume", "root_task_id": "rw-first"}}}
    transport.results = {"rw-first": {"task_id": "rw-first", "status": "interrupted", "chat_id": 1,
                                      "root_task_id": "rw-first"}}
    bar, listed = _held_line(branch_ui, transport, "rw-cold", f"cold-held-helper-{width}x{height}-{theme}",
                             width=width, height=height, theme=theme, work_id="rw-first", line=_COLD_LINE)
    row = rw.row(listed, "rw-first")
    assert (row["word"], row["secondary"], row["resume"]) == ("Waiting", _COLD_LINE, False), row
    assert all(rw.row(listed, task) is None for task in ("rw-cold", "rw-run", "rw-new")), "one work, one row"
    # The earlier episode's helpers are the owner's to select: the owner's mark, as the projection's turn says.
    assert bar["mark"].startswith("dot owner") and bar["waitNote"], bar
    assert transport.posts == [] and transport.decision_posts() == []


# ------------------------------------------------------------------------------------------------
# Real server: the Project Shade in WebKit.

def test_real_project_shade_in_webkit(direct_server_with_data, monkeypatch, tmp_path):
    """``backend.test_project_shade_lists_only_its_room_and_survives_disconnect`` byte for byte, with its
    browser launch pointed at WebKit (its evidence folder keeps the original name ``project-chromium``)."""
    launch = backend._launch
    launched = []

    def webkit(pw, engine, **options):
        browser, page, log = launch(pw, "webkit", **options)
        launched.append({"requested": engine, "launched": browser.browser_type.name, "version": browser.version,
                         "user_agent": page.evaluate("() => navigator.userAgent")})
        return browser, page, log

    monkeypatch.setattr(backend, "_launch", webkit)
    root = os.environ.get("OUROBOROS_UI_EVIDENCE_DIR")
    if root:
        monkeypatch.setenv("OUROBOROS_UI_EVIDENCE_DIR", str(Path(root) / "project-webkit"))
    try:
        backend.test_project_shade_lists_only_its_room_and_survives_disconnect(direct_server_with_data, monkeypatch, tmp_path)
    finally:
        if root:
            (Path(root) / "project-webkit").mkdir(parents=True, exist_ok=True)
            (Path(root) / "project-webkit" / "engine.json").write_text(json.dumps(launched, indent=2), encoding="utf-8")
    assert [item["launched"] for item in launched] == ["webkit"], launched


# ------------------------------------------------------------------------------------------------
# Real server: Hurry, Wrap up and Stop now from a row; a server restart with the page and list open.

def _press(page, task_id, action, url_suffix):
    menu = page.locator("body > .task-control-menu")
    with page.expect_response(lambda r: r.url.endswith(url_suffix) and r.request.method == "POST", timeout=30_000) as answer:
        menu.locator(f'[data-task-control="{action}"]').click()
    if action in {"hurry", "finalize"}:
        # Let the acknowledgement expire away from the pointer, keeping the Work list open (#2193).
        notice = page.locator('#toast-stack .toast').filter(has_text=("Hurry up: accepted" if action == "hurry"
            else f"The owner asked task {task_id} to summarize and stop"))
        notice.wait_for(state="visible")
        page.mouse.move(0, 0)
        notice.wait_for(state="detached")
    return {"status": answer.value.status, "body": answer.value.json(), "request": answer.value.request.post_data_json}


def _status(url, task_id):
    try:
        return str(backend._http(url, "GET", f"/api/tasks/{task_id}").get("status") or "")
    except Exception as exc:  # noqa: BLE001 - recorded, never hidden
        return f"error: {exc}"


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_real_row_hurry_wrap_up_stop_and_restart_with_page_open(direct_server_with_data, monkeypatch, engine):
    from playwright.sync_api import sync_playwright

    server = direct_server_with_data
    url, data_dir = server["url"], server["data_dir"]
    alpha, bravo, charlie = backend.ALPHA, backend.BRAVO, backend.CHARLIE
    evidence = backend.Evidence(data_dir, f"row-actions-restart-{engine}")
    marks = {alpha: "RW-ALPHA-9K", bravo: "RW-BRAVO-9K"}
    model = backend.ScriptedModel(monkeypatch, holds=marks.values())
    server["stop_server"]()
    backend._seed_roots(data_dir, (alpha, bravo, charlie), filler=10)
    server["start_server"]()
    backend._resume_after_app_stop(url, alpha, bravo, charlie)
    problems, observations = [], []
    try:
        with sync_playwright() as pw:
            browser, page, log = backend._launch(pw, engine)
            try:
                assert model.entered[marks[alpha]].wait(90), "ALPHA never reached its first model call"
                backend._wait(lambda: backend._phase(url, alpha) == "working" and backend._phase(url, bravo) == "queued",
                              60, "ALPHA working, BRAVO queued")
                backend._open_chat(page, url)
                backend._wait_facts(page, "main", lambda f: f["tone"] == "live" and not f["disabled"], "a live control")
                page.locator("#page-chat .rw-trigger").click()
                opened = backend._wait_facts(page, "main", lambda f: len(f["rows"]) == 3 and all(r["name"] for r in f["rows"]),
                                             "three named roots")
                ids = [row["id"] for row in opened["rows"]]
                evidence.record("opened", opened)

                # Hurry from the row: the real endpoint, that exact id.
                more = page.locator(f"#page-chat .rw-row[data-id='{alpha}'] .rw-more")
                more.click()
                page.locator("body > .task-control-menu").wait_for(state="visible", timeout=10_000)
                evidence.record("menu_working", backend._facts(page)["menu"])
                hurry = _press(page, alpha, "hurry", f"/api/tasks/{alpha}/hurry")
                evidence.record("hurry", hurry)
                assert hurry["status"] == 200, hurry
                evidence.shot(page, "01-after-hurry")

                # Wrap up from the row: the real cancel with the finalize policy; then only Stop now.
                more.click()
                page.locator("body > .task-control-menu").wait_for(state="visible", timeout=10_000)
                wrap = _press(page, alpha, "finalize", f"/api/tasks/{alpha}/cancel")
                evidence.record("wrap_up", wrap)
                assert wrap["status"] == 202 and wrap["body"]["cancel_state"] == "pending", wrap
                finalizing = backend._wait_facts(page, "main", lambda f: (backend._row(f, alpha) or {}).get("word")
                                                 in {"Finalizing…", "Cancelling…"}, "ALPHA finalizing in place")
                evidence.record("after_wrap_up", finalizing)
                evidence.record("detail_after_wrap_up", {k: v for k, v in backend._http(url, "GET", f"/api/tasks/{alpha}").items()
                                                         if k in ("status", "cancel_state", "stop_policy")})
                assert backend._row(finalizing, alpha)["more"], "Stop now must stay reachable after Wrap up"
                if backend._row(finalizing, alpha)["more"]:
                    more.click()
                    page.locator("body > .task-control-menu").wait_for(state="visible", timeout=10_000)
                    after_menu = backend._facts(page)["menu"]
                    evidence.record("menu_after_wrap_up", after_menu)
                    evidence.shot(page, "02-menu-after-wrap-up")
                    if after_menu != ["stop_now"]:
                        problems.append(f"after a real Wrap up the row's menu offers {after_menu}")
                    if "stop_now" in (after_menu or []):
                        stopped_reply = _press(page, alpha, "stop_now", f"/api/tasks/{alpha}/cancel")
                        evidence.record("stop_now", stopped_reply)
                        assert stopped_reply["status"] == 200, stopped_reply
                else:
                    observations.append("ALPHA's row withdrew its ⋯ after Wrap up")
                model.release[marks[alpha]].set()
                backend._wait(lambda: _status(url, alpha) in {"cancelled", "completed", "failed"}, 120, "ALPHA ended")
                ended = backend._wait_facts(page, "main", lambda f: (backend._row(f, alpha) or {}).get("settled"),
                                            "ALPHA settled in place", timeout=60)
                evidence.record("alpha_ended", {"status": _status(url, alpha), "facts": ended})
                assert _status(url, alpha) == "cancelled", "Stop now did not settle cancellation"
                if [row["id"] for row in ended["rows"]] != ids:
                    problems.append("order changed when ALPHA ended")
                if backend._row(ended, alpha)["more"] or backend._row(ended, alpha)["resume"]:
                    problems.append("ALPHA kept controls after it ended")

                # Stop now from a queued root's row, when its row offers controls.
                backend._wait(lambda: backend._phase(url, bravo) == "working", 90, "BRAVO working")
                current = backend._wait_facts(page, "main", lambda f: (backend._row(f, bravo) or {}).get("word") == "Working",
                                              "BRAVO working in place")
                charlie_row = backend._row(current, charlie)
                evidence.record("charlie_before_stop", charlie_row)
                assert charlie_row["more"], "queued root has no Stop door"
                if charlie_row["more"]:
                    page.locator(f"#page-chat .rw-row[data-id='{charlie}'] .rw-more").click()
                    page.locator("body > .task-control-menu").wait_for(state="visible", timeout=10_000)
                    evidence.record("charlie_menu", backend._facts(page)["menu"])
                    evidence.record("charlie_stop", _press(page, charlie, "stop_now", f"/api/tasks/{charlie}/cancel"))
                    backend._wait(lambda: _status(url, charlie) in {"cancelled", "completed", "failed"}, 90, "CHARLIE ended")
                    stopped = backend._wait_facts(page, "main", lambda f: (backend._row(f, charlie) or {}).get("settled"),
                                                  "CHARLIE settled", timeout=60)
                    evidence.record("charlie_ended", {"status": _status(url, charlie), "facts": stopped})
                    assert _status(url, charlie) == "cancelled", "queued Stop now did not settle cancellation"
                else:
                    observations.append("a queued root's row offered no ⋯ (its card has no Stop predicate yet)")
                evidence.shot(page, "03-after-stops")

                # Working is admission, not a saved continuation. Stop only after
                # this exact attempt has real recovery bytes, never a seeded source.
                assert model.entered[marks[bravo]].wait(90), "BRAVO never reached its first model call"
                from ouroboros.working_checkpoint import read_working

                def saved_bravo():
                    detail = backend._http(url, "GET", f"/api/tasks/{bravo}")
                    attempt = int(detail.get("task_attempt") or 0)
                    raw, state = read_working(data_dir, bravo, attempt)
                    return ({"task_id": state["task_id"], "attempt": attempt,
                             "bytes": len(raw), "working": state["working"]} if raw else None)

                evidence.record("bravo_checkpoint_before_restart", backend._wait(
                    saved_bravo, 90, "BRAVO's current attempt checkpoint"))
                # A server restart with the page and the list open.
                page.evaluate(ROWS_RECORDER, ["main", 120_000])
                sockets = page.evaluate("() => window.__testSockets.length")
                restart_at = time.monotonic()
                server["stop_server"]()
                down = backend._wait_facts(page, "main", lambda f: f["note"].startswith("Reconnecting"), "the list says reconnecting")
                evidence.record("server_down", down)
                evidence.shot(page, "04-server-down")
                if any(r["more"] or r["resume"] or r["question"] for r in down["rows"] if not r["settled"]):
                    problems.append(f"a control offered while the server is down: {down['rows']}")
                server["start_server"]()
                page.wait_for_function("n => window.__testSockets.length > n && window.__testSockets.at(-1).readyState === 1",
                                       arg=sockets, timeout=120_000)
                census = backend._wait(lambda: backend._census(url).get(bravo), 60, "BRAVO in the census after restart")
                evidence.record("census_after_restart", census)
                evidence.record("bravo_detail_after_restart", {k: v for k, v in backend._http(
                    url, "GET", f"/api/tasks/{bravo}").items() if k in ("status", "reason_code", "cancel_state")})
                back = backend._wait_facts(page, "main", lambda f: not f["note"] and (backend._row(f, bravo) or {}).get("tone")
                                           != "unknown", "the list confirmed after the restart", timeout=90)
                evidence.record("restart_seconds", round(time.monotonic() - restart_at, 1))
                evidence.record("after_restart", back)
                evidence.shot(page, "05-after-restart")
                if [row["id"] for row in back["rows"]] != ids:
                    problems.append(f"order changed across the restart: {[row['id'] for row in back['rows']]}")
                bravo_row = backend._row(back, bravo)
                assert census.get("phase") == "budget_paused" and bravo_row["resume"], (census, bravo_row)
                if census.get("phase") == "budget_paused" and not bravo_row["resume"]:
                    problems.append(f"the census says budget_paused but the row offers no Resume: {bravo_row}")
                if census.get("phase") != "budget_paused" and bravo_row["resume"]:
                    problems.append(f"Resume offered without a paused census: {bravo_row} / {census}")
                if bravo_row["resume"]:
                    with page.expect_response(lambda r: r.url.endswith(f"/api/tasks/{bravo}/resume")
                                              and r.request.method == "POST", timeout=30_000) as resumed:
                        page.locator(f"#page-chat .rw-row[data-id='{bravo}'] .rw-pill-resume").click()
                    evidence.record("bravo_resume", {"status": resumed.value.status, "body": resumed.value.json()})
                    assert resumed.value.status == 200, resumed.value.text()
                model.finish()
                backend._wait(lambda: _status(url, bravo) == "completed", 180, "BRAVO completed")
                done = backend._wait_facts(page, "main", lambda f: (backend._row(f, bravo) or {}).get("settled")
                                           and backend._row(f, bravo)["word"] == "Done", "BRAVO Done in place", timeout=90)
                evidence.record("bravo_done", done)
                evidence.record("restart_frames", page.evaluate("() => (window.__rwRows || []).slice(0, 40)"))
                evidence.shot(page, "06-bravo-done")
                page.locator(f"#page-chat .rw-row[data-id='{bravo}'] .rw-main").focus()
                page.keyboard.press("Escape")
                closed = backend._wait_facts(page, "main", lambda f: f["surfaceHidden"], "closed")
                evidence.record("closed", closed)
                assert backend._decision_posts(log) == []
                assert log["pageerrors"].unexpected() == [], log["pageerrors"]
            except Exception:
                for label, path in (("census_on_failure", "/api/state"),
                                    ("bravo_detail_on_failure", f"/api/tasks/{bravo}")):
                    try:
                        evidence.record(label, backend._http(url, "GET", path))
                    except Exception as exc:
                        evidence.record(label, {"read_error": str(exc)})
                for relative in ("logs/supervisor.jsonl", "logs/server.log", "state/queue_snapshot.json"):
                    try:
                        evidence.record(relative, (data_dir / relative).read_text(encoding="utf-8"))
                    except (OSError, UnicodeError) as exc:
                        evidence.record(relative, {"read_error": str(exc)})
                evidence.shot(page, "failure")
                raise
            finally:
                evidence.record("problems", problems)
                evidence.record("observations", observations)
                evidence.record("log", log)
                browser.close()
    finally:
        model.finish()
        evidence.record("model_calls", list(model.calls))
    assert not problems, problems
