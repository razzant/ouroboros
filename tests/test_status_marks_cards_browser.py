"""Task-card chips, Activity, Logs and the budget pill speak the status marks (docs/DESIGN.md "Status marks").

The page loads the production stylesheets and modules and paints with their own producers: every row of the approved
chip sheet (its body tone and its shared mark), a running parent whose helper waits for the owner, Activity's words
and its neutral Stop…, a Logs task headline that settled Stopped or Done with warnings beside a warning event, and the
budget pill as a neutral fact unless a spending-limit stop waits for the owner. Colours are compared with the tokens
they must resolve to, in both themes and both engines; screenshots go to ``OUROBOROS_UI_EVIDENCE_DIR`` first.
"""
from __future__ import annotations

import json
import os
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]
WEB = Path(__file__).resolve().parents[1] / "web"

HTML = """<!doctype html><html lang="en" class="ouro-ui"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="/ui.css"><link rel="stylesheet" href="/style.css">
<link rel="stylesheet" href="/model_wait.css"></head><body>
<div id="header" style="position: relative; height: 96px"></div>
<div id="sheet" style="display: grid; gap: 6px; padding: 12px; justify-items: start"></div>
<div class="dashboard-panel active" id="activity"></div>
<div id="logs"></div>
<script type="module">
import { censusTaskPhase, desiredLiveCardPhase, paintTaskPhase, taskActivityPresentation } from '/modules/task_phase_chip.js';
import { waitingModel } from '/modules/project_activity.js';
import { applyCardWorkViews, emptyWorkState, reconcileWorkState } from '/modules/work_status.js';
import { mainHeaderActionsHtml } from '/modules/chat_header.js';
import { headerBudgetPresentation } from '/modules/chat_activity.js';
import { initActivity } from '/modules/activity.js';
import { initLogs } from '/modules/logs.js';

document.documentElement.dataset.theme = new URLSearchParams(location.search).get('theme') || 'dark';
const wait = (patch = {}) => ({ w: { wait_id: 'w', revision: 1, task_attempt: 1, state: 'waiting', reason: 'quota', ...patch } });
const census = (row) => { const full = { activity_id: 't', ...row }; return censusTaskPhase(full, null, true, waitingModel(full)); };
// A running parent whose helper waits for the owner's answer: the server takes the turn from every member.
const parent = { groupId: 'parent' };
applyCardWorkViews(new Map([['parent', parent]]), reconcileWorkState(emptyWorkState(), { supervisor_ready: true,
    works_complete: true, rooms: {}, works: [{ work_id: 'parent', room: 1, episode: { task_id: 'parent', live: true },
        activity: 'working', transition: 'none', call: '', because: [], mark: 'moving', confirmed: true,
        turn: [{ kind: 'question', task_id: 'helper', quiz_id: 'q1' }], controls: [] }] }, 1));
const rows = {
    working: census({ phase: 'working' }),
    preparing: census({ phase: 'preparing' }),
    finalizing: census({ phase: 'finalizing' }),
    queued: census({ phase: 'queued' }),
    access_automatic: census({ phase: 'working', model_waits: wait() }),
    access_owner: census({ phase: 'working', model_waits: wait({ reason: 'auth', requires_owner_action: true }) }),
    paused: census({ phase: 'budget_paused', pause_cause: 'owner' }),
    paused_limit: census({ phase: 'budget_paused', pause_cause: 'budget', can_resume: true }),
    pausing: census({ phase: 'budget_pausing', pause_cause: 'owner' }),
    question: census({ phase: 'working', owner_wait: { owner_wait_state: 'waiting', quiz_state: 'open' } }),
    unconfirmed: census({ phase: 'unknown' }),
    done: taskActivityPresentation({ ended: true, outcome: 'done' }),
    warnings: taskActivityPresentation({ ended: true, outcome: 'warn' }),
    stopped: taskActivityPresentation({ ended: true, outcome: 'cancelled' }),
    failed: taskActivityPresentation({ ended: true, outcome: 'error' }),
    parent: desiredLiveCardPhase(parent),
};
const sheet = document.getElementById('sheet');
for (const [name, view] of Object.entries(rows)) {
    const chip = document.createElement('span');
    chip.dataset.row = name;
    sheet.append(chip);
    paintTaskPhase(chip, view);
}

// The budget pill as chat.js writes it from the /api/state body.
const header = document.getElementById('header');
header.innerHTML = mainHeaderActionsHtml();
window.paintBudget = (body) => {
    const budget = headerBudgetPresentation(body);
    document.getElementById('chat-budget-text').textContent = budget.label;
    document.getElementById('chat-budget-bar-fill').style.width = `${budget.fillPct}%`;
    document.getElementById('chat-budget-pill').dataset.attention = budget.attention ? '1' : '0';
};
window.paintBudget({ spent_usd: 10, budget_limit: 10, accounting: { available: true }, main_attention: [] });

window.activity = initActivity({ mount: document.getElementById('activity'), ws: { send() {} } });
await window.activity.refresh();

const handlers = new Map();
initLogs({ mount: document.getElementById('logs'), ws: { on: (type, fn) => handlers.set(type, fn), send() {} },
    state: { activePage: 'dashboard', dashboardActiveSubtab: 'logs' } });
window.emitLog = (data) => handlers.get('log')({ data });
window.ready = true;
</script></body></html>"""

# Every row of the approved chip sheet: the chip's body tone and its mark's shape and hue.
SHEET = {
    "working": ("Working", "live", "moving", "ink"),
    "preparing": ("Preparing", "live", "moving", "ink"),
    "finalizing": ("Finalizing…", "live", "moving", "ink"),
    "queued": ("Queued", "neutral", "still", "ink"),
    "access_automatic": ("Waiting for access", "neutral", "still", "ink"),
    "access_owner": ("Waiting for access", "owner", "dot", "owner"),
    "paused": ("Paused", "neutral", "pause", "ink"),
    # The chip says its word alone; why it waits is said beside it (the work's line, the owner's message).
    "paused_limit": ("Paused", "owner", "dot", "owner"),
    "pausing": ("Pausing…", "neutral", "moving", "ink"),
    "question": ("Waiting for your answer", "owner", "dot", "owner"),
    "unconfirmed": ("Activity unconfirmed", "neutral", "ring", "ink"),
    "done": ("Done", "ok", "dot", "ok"),
    "warnings": ("Done with warnings", "ok", "dot", "ok"),
    "stopped": ("Stopped", "neutral", "dot", "neutral"),
    "failed": ("Failed", "failed", "dot", "failed"),
    # A running parent keeps its live word; its helper's question is the owner's amber dot.
    "parent": ("Working", "live", "dot", "owner"),
}
# The token each body tone's text and each mark hue resolves to (ui.css, style.css "Status marks").
TONE_INK = {"live": "--accent-light", "owner": "--status-warn-fg", "ok": "--status-ok-fg",
            "failed": "--status-error-fg", "neutral": "--text-secondary"}
HUE_INK = {"owner": "--amber", "ok": "--green", "failed": "--red", "neutral": "--text-meta"}

PROBE_JS = """() => {
    const probe = document.createElement('span');
    document.body.append(probe);
    const ink = (token) => { probe.style.color = `var(${token})`; return getComputedStyle(probe).color; };
    const tokens = Object.fromEntries(['--accent-light', '--status-warn-fg', '--status-ok-fg', '--status-error-fg',
        '--text-secondary', '--text-meta', '--amber', '--green', '--red', '--tone-red', '--status-warn-bg'].map((t) => [t, ink(t)]));
    probe.remove();
    const chips = Object.fromEntries([...document.querySelectorAll('#sheet [data-row]')].map((chip) => {
        const mark = chip.querySelector(':scope > .status-mark');
        const dots = [...mark.querySelectorAll('i')].map((dot) => getComputedStyle(dot).animationName);
        return [chip.dataset.row, { text: chip.textContent, tone: chip.dataset.chipTone, shape: mark.dataset.shape,
            hue: mark.dataset.hue, hidden: mark.hidden, first: chip.firstElementChild === mark,
            color: getComputedStyle(chip).color, markColor: getComputedStyle(mark).color,
            width: mark.getBoundingClientRect().width, opacity: getComputedStyle(mark).opacity,
            before: getComputedStyle(chip, '::before').content, animation: dots.find((name) => name !== 'none') || 'none' }];
    }));
    return { tokens, chips };
}"""


def _serve():
    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            if self.path.startswith("/fixture"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(HTML.encode())
            else:
                super().do_GET()

    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(WEB)))
    Thread(target=server.serve_forever, daemon=True).start()
    return server


def _api(route):
    """Activity's three reads and the Logs backfill: one census-only direct turn, nothing queued or logged."""
    path = route.request.url.split("/api/", 1)[1]
    if path.startswith("tasks?"):
        body = {"tasks": [], "queue": {"running": [], "pending": []}}
    elif path == "state":
        body = {"bg_consciousness_enabled": False, "active_chat_activities_complete": True, "supervisor_ready": True,
                "active_chat_activities": [{"activity_id": "direct-1", "kind": "direct_chat", "phase": "thinking",
                                            "chat_id": 1, "started_at": 1}]}
    elif path == "schedules":
        body = {"tasks": []}
    elif path.startswith("logs/"):
        body = {"entries": []}
    else:
        return route.fulfill(status=404, body="{}")
    route.fulfill(status=200, content_type="application/json", body=json.dumps(body))


@pytest.mark.parametrize("engine", ["chromium", "webkit"])
def test_cards_activity_logs_and_budget_speak_the_status_marks(tmp_path, engine):
    from playwright.sync_api import expect, sync_playwright

    evidence = Path(os.environ.get("OUROBOROS_UI_EVIDENCE_DIR") or tmp_path / "evidence")
    evidence.mkdir(parents=True, exist_ok=True)
    server = _serve()
    url = f"http://127.0.0.1:{server.server_port}/fixture"
    try:
        with sync_playwright() as pw:
            browser = getattr(pw, engine).launch(headless=True)
            try:
                for theme, motion in (("light", "no-preference"), ("dark", "no-preference"), ("light", "reduce")):
                    page = browser.new_page(viewport={"width": 760, "height": 1400}, color_scheme=theme,
                                            reduced_motion=motion)
                    errors = []
                    page.on("pageerror", lambda exc: errors.append(str(exc)))
                    page.route("**/api/**", _api)
                    page.goto(f"{url}?theme={theme}")
                    page.wait_for_function("window.ready === true")
                    page.locator("#sheet").screenshot(path=str(evidence / f"chips-{engine}-{theme}-{motion}.png"),
                                                      animations="disabled")
                    assert errors == [], errors
                    facts = page.evaluate(PROBE_JS)
                    tokens, chips = facts["tokens"], facts["chips"]
                    assert set(chips) == set(SHEET)
                    for name, (text, tone, shape, hue) in SHEET.items():
                        chip = chips[name]
                        # The mark is the chip's first child, its words the text beside it; no pseudo-element dot.
                        assert (chip["text"], chip["tone"], chip["shape"], chip["hue"]) == (text, tone, shape, hue), name
                        assert chip["first"] and not chip["hidden"] and chip["before"] in ("none", "normal"), (name, chip)
                        assert chip["color"] == tokens[TONE_INK[tone]], (name, chip, tokens)
                        # A mark in the ink hue takes the chip's own text colour; a fact's dot its saturated hue.
                        assert chip["markColor"] == (chip["color"] if hue == "ink" else tokens[HUE_INK[hue]]), (name, chip)
                        moving = shape == "moving" and motion != "reduce"
                        assert (chip["animation"] != "none") == moving, (name, motion, chip)
                    # A live chip keeps the moving dots' width when the owner's dot takes their place: Working stays put.
                    assert abs(chips["parent"]["width"] - chips["working"]["width"]) <= 0.5, (chips["parent"], chips["working"])
                    # The unconfirmed ring is faint.
                    assert chips["unconfirmed"]["opacity"] == "0.55", chips["unconfirmed"]

                    # Activity: a census row says the chip's word; Stop… is the neutral trigger, never danger red.
                    sub = page.locator('[data-activity-section="queue"] .activity-sub').first
                    expect(sub).to_contain_text("Thinking")
                    assert "thinking" not in sub.inner_text()
                    stop = page.locator('[data-act="task-control"][data-id="direct-1"]')
                    assert stop.get_attribute("class") == "btn btn-xs btn-default"
                    stop_paint = stop.evaluate("el => [getComputedStyle(el).backgroundColor, getComputedStyle(el).color]")
                    # Danger red is the red tone's body and the red ink. On dark that body is the same neutral veil as
                    # a neutral capsule's, so there the ink is what tells them apart.
                    assert stop_paint[1] != tokens["--status-error-fg"], stop_paint
                    assert theme == "dark" or stop_paint[0] != tokens["--tone-red"], stop_paint

                    # Logs: a settled headline takes the cards' tone; a warn-level event stays amber.
                    page.evaluate("""() => {
                        const at = (s) => `2026-10-19T10:00:0${s}+00:00`;
                        window.emitLog({ type: 'task_started', task_id: 'stopped', task_type: 'task', ts: at(1) });
                        window.emitLog({ type: 'task_done', task_id: 'stopped', status: 'cancelled', ts: at(2) });
                        window.emitLog({ type: 'task_started', task_id: 'warned', task_type: 'task', ts: at(3) });
                        window.emitLog({ type: 'task_done', task_id: 'warned', status: 'completed',
                            outcome_axes: { execution: { status: 'degraded' } }, ts: at(4) });
                        window.emitLog({ type: 'task_started', task_id: 'checking', task_type: 'task', ts: at(5) });
                        window.emitLog({ type: 'task_checkpoint', task_id: 'checking',
                            checkpoint_kind: 'context_fit_low_retry', ts: at(6) });
                    }""")
                    def headline(task_id):
                        phase = page.locator(f'.log-task-card[data-task-group="{task_id}"] [data-task-phase]')
                        return phase.inner_text(), phase.evaluate("el => getComputedStyle(el).color")
                    assert headline("stopped") == ("cancelled", tokens["--text-secondary"])
                    assert headline("warned") == ("warn", tokens["--status-ok-fg"])
                    assert headline("checking") == ("warn", tokens["--status-warn-fg"])

                    # The budget pill: a neutral fact at the very limit; amber only while its stop waits for the owner.
                    pill = page.locator("#chat-budget-pill")
                    budget_text, budget_fill = page.locator("#chat-budget-text"), page.locator("#chat-budget-bar-fill")
                    color = lambda node, prop="color": node.evaluate(f"el => getComputedStyle(el)['{prop}']")  # noqa: E731
                    assert pill.get_attribute("data-attention") == "0"
                    assert color(budget_text) == tokens["--text-secondary"]
                    assert color(budget_fill, "backgroundColor") == tokens["--text-meta"]
                    page.evaluate("""() => window.paintBudget({ spent_usd: 10, budget_limit: 10, accounting: { available: true },
                        main_attention: [{ task_id: 's', attention_reason: 'budget', can_resume: true }] })""")
                    assert pill.get_attribute("data-attention") == "1"
                    assert color(budget_text) == tokens["--status-warn-fg"]
                    assert color(budget_fill, "backgroundColor") == tokens["--amber"]
                    panic = page.locator('[data-chat-command="panic"]')  # the fixture page's one Panic
                    # Light adds its red tint; dark's neutral danger veil is already in the floating capsule.
                    paint = panic.evaluate("""(el, theme) => {
                        const probe = document.createElement('span'); document.body.append(probe);
                        probe.style.background = theme === 'light'
                            ? 'linear-gradient(var(--tone-red), var(--tone-red)), var(--chip-float)' : 'var(--chip-float)';
                        probe.style.color = 'var(--panic-ink)';
                        const read = node => { const s = getComputedStyle(node);
                            return [s.backgroundColor, s.backgroundImage, s.color]; };
                        const expected = read(probe); probe.remove();
                        return {actual: read(el), expected};
                    }""", theme)
                    assert paint["actual"] == paint["expected"], paint
                    page.screenshot(path=str(evidence / f"cards-surfaces-{engine}-{theme}-{motion}.png"))
                    page.close()
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()
