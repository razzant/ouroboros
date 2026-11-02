"""What a task card's Details opens stays reachable by keyboard while it scrolls inside (real chat module, synthetic HTTP/WS).

While a card holds children its opened facts and notes are short boxes that scroll inside (style.css). WebKit gives no
keyboard way into a scroller with nothing focusable in it, so each part is a named region (`Scrollable task details`,
`Scrollable task notes`) exactly while it scrolls: Tab from the Details door reaches it, PageDown scrolls it, the one
focus ring shows, and a closed card adds no tab stop. The door keeps its focus and `aria-controls`, and pressing it moves
no node.
"""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import threading
from urllib.parse import urlsplit

import pytest

from tests.ui_console_errors import PageErrors

ROOT = Path(__file__).resolve().parents[1]
T = '2026-10-29T09:00:00Z'
HTML = '''<!doctype html><html data-theme="light"><head><meta charset="utf-8">
<link rel="stylesheet" href="/web/ui.css"><link rel="stylesheet" href="/web/style.css">
<style>html,body,#mount{height:100%;margin:0}</style></head><body><div id="mount"></div>
<script type="module">
import {createChatInstance} from '/web/modules/chat.js';
const handlers=new Map();window.fixture={handlers};
fixture.instance=createChatInstance({ws:{on(t,f){handlers.set(t,f);return()=>handlers.delete(t)},
isConnected:()=>true,send(){return {status:'sent'}}},
state:{activePage:'chat',projectChatIds:new Set(),unreadCount:0},updateUnreadBadge(){},
chatId:1,idPrefix:'chat',mountEl:document.getElementById('mount'),
stateSnapshots:{begin:()=>({generation:1,requestedAt:Date.now()}),gate(){return Promise.resolve(this.begin())},
isCurrent:()=>true,apply(){},fail(){}}});
</script></body></html>'''
CARD = '.chat-live-card[data-task-id="root"]'
PART = {'facts': f'{CARD} > .chat-live-details', 'notes': f'{CARD} > .chat-live-timeline'}
NAME = {'facts': 'Scrollable task details', 'notes': 'Scrollable task notes'}
STATE = """sel => { const e = document.querySelector(sel); return {tabindex: e.getAttribute('tabindex'), role: e.getAttribute('role'),
    label: e.getAttribute('aria-label'), focused: document.activeElement === e, scrollTop: e.scrollTop,
    scrolls: e.scrollHeight > e.clientHeight + 1, shown: e.getClientRects().length > 0,
    outline: getComputedStyle(e).outlineStyle + ' ' + getComputedStyle(e).outlineWidth}; }"""


def _serve():
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            path = urlsplit(self.path).path
            content_type, status = 'application/json', 200
            if path == '/':
                data, content_type = HTML.encode(), 'text/html'
            elif path.startswith('/web/'):
                source = (ROOT / path.lstrip('/')).resolve()
                if source.is_relative_to(ROOT / 'web') and source.is_file():
                    data = source.read_bytes()
                    content_type = mimetypes.guess_type(source.name)[0] or 'text/plain'
                else:
                    data, status = b'not found', 404
            elif path == '/api/chat/history':
                data = json.dumps({'messages': [], 'window': {'complete': True}, 'page_cursor': 'recent',
                                   'has_more': False, 'next_cursor': None}).encode()
            elif path.startswith('/api/tasks/'):
                data = json.dumps({'task_id': path.split('/')[3], 'status': 'running'}).encode()
            else:
                data = json.dumps({'active_direct_turns': [], 'supervisor_ready': True}).encode()
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            self.send_response(500)
            self.end_headers()

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


@pytest.mark.ui_browser
@pytest.mark.parametrize('engine', ['chromium', 'webkit'])
@pytest.mark.parametrize('width,height', [(1100, 735), (390, 844)])
def test_capped_details_parts_are_keyboard_regions_only_while_they_scroll(engine, width, height):
    from playwright.sync_api import sync_playwright

    server = _serve()
    try:
        with sync_playwright() as playwright:
            browser = getattr(playwright, engine).launch()
            try:
                page = browser.new_page(viewport={'width': width, 'height': height})
                errors = PageErrors(page, engine)
                page.goto(f'http://127.0.0.1:{server.server_port}/')
                page.wait_for_function('window.fixture?.instance.hasPaintedHistory()')

                def emit(row):
                    page.evaluate('row => fixture.handlers.get("chat")(row)', row)

                # A root with fifteen notes, then two children (its own notes make it the parent card first).
                for index in range(15):
                    emit({'chat_id': 1, 'task_id': 'root', 'role': 'assistant', 'is_progress': True, 'narration': True,
                          'suggested_name': 'Delivery check', 'ts': f'2026-10-29T09:{index:02d}:00Z',
                          'text': f'Step {index + 1}: checking the delivery order for case {index + 1} and noting it in the report.'})
                for child, role in (('kid-1', 'Builder'), ('kid-2', 'Independent critic')):
                    emit({'chat_id': 1, 'task_id': child, 'role': 'assistant', 'is_progress': True, 'narration': True,
                          'subagent_event': 'running', 'subagent_task_id': child, 'parent_task_id': 'root', 'root_task_id': 'root',
                          'delegation_role': 'subagent', 'subagent_role': role, 'text': f'{role} at work', 'ts': T})
                card = page.locator(CARD)
                card.wait_for(state='visible')
                door = card.locator(':scope > [data-live-summary-button] [data-live-disclosure]')
                assert card.get_attribute('data-children') is not None
                # Closed: nothing Details would open is a tab stop.
                for part in PART.values():
                    closed = page.evaluate(STATE, part)
                    assert not closed['shown'] and closed['tabindex'] is None, closed
                page.evaluate("""card => { const nodes = [...document.querySelector(card).children];
                    nodes.forEach((node, index) => { node.__order = index; }); }""", CARD)

                door.click()
                # The card's facts beyond its line, longer than the facts box holds (eight labelled rows).
                page.evaluate("""sel => { const dl = document.querySelector(sel + ' > .chat-live-details > [data-live-facts]');
                    for (let i = 1; i <= 8; i += 1) { const row = document.createElement('div'); row.className = 'chat-live-fact';
                        row.innerHTML = `<dt>Fact ${i}</dt><dd>A recorded value long enough to take its own line ${i}</dd>`; dl.append(row); }
                    dl.hidden = false; }""", CARD)
                for key, part in PART.items():
                    page.wait_for_function('sel => document.querySelector(sel)?.getAttribute("role") === "region"', arg=part)
                    state = page.evaluate(STATE, part)
                    assert state['scrolls'] and state['tabindex'] == '0' and state['label'] == NAME[key], (key, state)
                assert page.evaluate("card => [...document.querySelector(card).children].map((node) => node.__order ?? 'new')", CARD) \
                    == list(range(page.evaluate('card => document.querySelector(card).children.length', CARD))), 'Details moved no node'
                assert door.get_attribute('aria-expanded') == 'true'
                assert door.get_attribute('aria-controls') == page.evaluate(f"document.querySelector('{PART['notes']}').id")

                # Tab from the door reaches each part in reading order; PageDown scrolls it; the one ring shows.
                door.focus()
                for key, part in PART.items():
                    for _ in range(12):
                        page.keyboard.press('Tab')
                        if page.evaluate(STATE, part)['focused']:
                            break
                    focused = page.evaluate(STATE, part)
                    assert focused['focused'], (key, 'Tab reaches it from the door', focused)
                    assert focused['outline'] == 'solid 2px', (key, focused)
                    page.keyboard.press('PageDown')
                    page.wait_for_function('sel => document.querySelector(sel).scrollTop > 0', arg=part, timeout=3000)

                # Closing: the door keeps the focus, and what it opened leaves the tab order with it.
                door.focus()
                page.keyboard.press('Enter')
                assert door.get_attribute('aria-expanded') == 'false'
                assert page.evaluate("sel => document.activeElement === document.querySelector(sel)",
                                     f'{CARD} > .chat-live-summary-button [data-live-disclosure]')
                for part in PART.values():
                    page.wait_for_function('sel => !document.querySelector(sel).hasAttribute("tabindex")', arg=part)
                assert not errors.unexpected(), errors
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()


CHILD_ROW = """() => [...document.querySelectorAll('.chat-live-card.subagent')].map((card) => {
    const r = (e) => e.getBoundingClientRect(), s = card.querySelector(':scope > .chat-live-summary-button');
    const title = s.querySelector('[data-live-title]'), phase = s.querySelector('[data-live-phase]');
    const side = s.querySelector('.chat-live-summary-side'), chevron = side.querySelector('.chat-live-chevron');
    const lh = parseFloat(getComputedStyle(title).lineHeight);
    return {role: title.textContent, phase: phase.textContent.trim(), cardRight: r(card).right,
        sideRight: r(side).right, chevronRight: r(chevron).right, chevronShown: r(chevron).width > 1,
        sideTop: r(side).top, titleTop: r(title).top, titleBottom: r(title).bottom, chipTop: r(phase).top,
        titleLines: Math.round(r(title).height / lh), titleCut: title.scrollHeight > title.clientHeight + 1,
        phaseSpills: phase.scrollWidth > phase.clientWidth + 1};
})"""


@pytest.mark.ui_browser
@pytest.mark.parametrize('engine', ['chromium', 'webkit'])
@pytest.mark.parametrize('width', [320, 390])
def test_a_disconnected_child_row_keeps_its_role_chip_and_controls_inside_the_card(engine, width):
    """№2 at a phone's narrowest: after a disconnect paints `Activity unconfirmed` the child's identity row still keeps
    the whole chip, a readable role and `Details` with its chevron inside the card. At 320px the floors of chip and role
    leave no room for the controls, which move under the role; at 390px the row stays one row."""
    from playwright.sync_api import sync_playwright

    server = _serve()
    try:
        with sync_playwright() as playwright:
            browser = getattr(playwright, engine).launch()
            try:
                page = browser.new_page(viewport={'width': width, 'height': 800})
                errors = PageErrors(page, engine)
                page.goto(f'http://127.0.0.1:{server.server_port}/')
                page.wait_for_function('window.fixture?.instance.hasPaintedHistory()')

                def emit(row):
                    page.evaluate('row => fixture.handlers.get("chat")(row)', row)

                emit({'chat_id': 1, 'task_id': 'root', 'role': 'assistant', 'is_progress': True, 'narration': True,
                      'suggested_name': 'Delivery check', 'ts': T, 'text': 'Root at work'})
                for child, role in (('kid-1', 'Builder'), ('kid-2', 'Independent critic')):
                    emit({'chat_id': 1, 'task_id': child, 'role': 'assistant', 'is_progress': True, 'narration': True,
                          'subagent_event': 'running', 'subagent_task_id': child, 'parent_task_id': 'root', 'root_task_id': 'root',
                          'delegation_role': 'subagent', 'subagent_role': role, 'text': f'{role} at work', 'ts': T})
                page.locator('.chat-live-card.subagent').nth(1).wait_for(state='visible')
                page.evaluate('() => fixture.handlers.get("close")()')
                page.wait_for_function("""() => [...document.querySelectorAll('.chat-live-card.subagent [data-live-phase]')]
                    .every((chip) => chip.textContent.includes('Activity unconfirmed'))""")
                rows = page.evaluate(CHILD_ROW)
                for row in rows:
                    assert row['chevronShown'] and row['chevronRight'] <= row['cardRight'] - 4, row
                    assert row['sideRight'] <= row['cardRight'], row
                    assert not row['phaseSpills'] and abs(row['titleTop'] - row['chipTop']) <= 4, row
                    assert row['titleLines'] <= 2 and not row['titleCut'], row
                    under = row['sideTop'] >= row['titleBottom'] - 1
                    assert under == (width == 320), row
                assert not errors.unexpected(), errors
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()


FOOT_HITS = """() => {
    const card = document.querySelector('.chat-live-card[data-task-id="root"]');
    const min = parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--touch-target-min'));
    // The shared recipe's hit area (ui.css .ui-hit-area): centred on the control, at least the touch minimum each way.
    // The card's own doors (its head) and commands (its foot), never a child's.
    const hits = [...card.querySelectorAll(':scope > .chat-live-summary-button .ui-hit-area, :scope > .chat-live-foot .ui-hit-area')]
        .filter((el) => el.getClientRects().length)
        .map((el) => { const r = el.getBoundingClientRect(), dx = Math.max(0, (min - r.width) / 2), dy = Math.max(0, (min - r.height) / 2);
            return {name: el.querySelector('[data-live-toggle]')?.textContent.trim() || el.textContent.trim(), top: r.top,
                head: Boolean(el.closest('.chat-live-summary-button')), l: r.left - dx, t: r.top - dy, r: r.right + dx, b: r.bottom + dy}; });
    const overlaps = [];
    hits.forEach((a, i) => hits.slice(i + 1).forEach((b) => {
        const w = Math.min(a.r, b.r) - Math.max(a.l, b.l), h = Math.min(a.b, b.b) - Math.max(a.t, b.t);
        if (w > 0.5 && h > 0.5) overlaps.push([a.name, b.name, w, h]);
    }));
    return {coarse: matchMedia('(pointer: coarse)').matches, names: hits.map((hit) => hit.name),
        head: hits.filter((hit) => hit.head).map((hit) => hit.name), rows: new Set(hits.map((hit) => Math.round(hit.top))).size, overlaps};
}"""


@pytest.mark.ui_browser
@pytest.mark.parametrize('engine', ['chromium', 'webkit'])
def test_a_wrapped_card_foot_parts_its_rows_so_finger_targets_never_overlap(engine):
    """№4 on a 320px phone: a card's `Reviews 2 · 1 active` door, now Details' twin in the card's head (owner 30.10, B),
    and `Turn into project` in its foot never share a row, and the rows part by the hit areas' overshoot, so no point
    of the card belongs to two targets."""
    from playwright.sync_api import sync_playwright

    server = _serve()
    try:
        with sync_playwright() as playwright:
            browser = getattr(playwright, engine).launch()
            try:
                page = browser.new_page(viewport={'width': 320, 'height': 800}, has_touch=True)
                errors = PageErrors(page, engine)
                page.goto(f'http://127.0.0.1:{server.server_port}/')
                page.wait_for_function('window.fixture?.instance.hasPaintedHistory()')

                def emit(row):
                    page.evaluate('row => fixture.handlers.get("chat")(row)', row)

                emit({'chat_id': 1, 'task_id': 'root', 'role': 'assistant', 'is_progress': True, 'narration': True,
                      'suggested_name': 'Delivery check', 'ts': T, 'text': 'Root at work'})
                for skill, status in (('alpha', 'clean'), ('beta', 'running')):
                    emit({'chat_id': 1, 'role': 'system', 'system_type': 'skill_review', 'task_id': 'root', 'ts': T,
                          'review_group': {'surface': 'skill', 'id': f'task:root:{skill}', 'presentation_owner_task_id': 'root',
                                           'skill': skill, 'status': status,
                                           'attempts': [{'job_id': f'job-{skill}', 'skill': skill, 'status': status}]}})
                page.wait_for_function("""() => document.querySelector('.chat-live-card[data-task-id="root"] [data-live-review-summary]')
                    ?.textContent === 'Reviews 2 · 1 active'""")
                foot = page.evaluate(FOOT_HITS)
                assert foot['coarse'] and foot['names'] == ['Reviews 2 · 1 active', 'Details', 'Turn into project'], foot
                assert foot['head'] == ['Reviews 2 · 1 active', 'Details'], foot
                assert foot['rows'] == 2 and foot['overlaps'] == [], foot
                assert not errors.unexpected(), errors
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()
