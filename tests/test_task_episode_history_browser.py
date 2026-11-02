"""Physical episode disclosures through the real chat consumer; synthetic HTTP/WS only."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import hashlib
import mimetypes
import os
from pathlib import Path
import threading
from urllib.parse import urlsplit

import pytest

from tests.ui_console_errors import PageErrors

ROOT = Path(__file__).resolve().parents[1]
T = '2026-10-17T08:00:00Z'
HTML = '''<!doctype html><html data-theme="light"><head><meta charset="utf-8">
<link rel="stylesheet" href="/web/ui.css"><link rel="stylesheet" href="/web/style.css">
<style>html,body,#mount{height:100%;margin:0}</style></head><body><div id="mount"></div>
<script type="module">
import {createChatInstance} from '/web/modules/chat.js';
const handlers=new Map(),sent=[];window.fixture={handlers,sent};
fixture.instance=createChatInstance({ws:{on(t,f){handlers.set(t,f);return()=>handlers.delete(t)},
isConnected:()=>true,send(v){sent.push(v);return {status:'sent'}}},
state:{activePage:'chat',projectChatIds:new Set(),unreadCount:0},updateUnreadBadge(){},
chatId:1,idPrefix:'chat',mountEl:document.getElementById('mount'),
stateSnapshots:{begin:()=>({generation:1,requestedAt:Date.now()}),gate(){return Promise.resolve(this.begin())},
isCurrent:()=>true,apply(){},fail(){}}});
</script></body></html>'''


@pytest.mark.ui_browser
@pytest.mark.parametrize('engine', ['chromium', 'webkit'])
def test_physical_episode_content_survives_continue(tmp_path, engine):
    from playwright.sync_api import sync_playwright

    review_bytes = b'Original physical review source\nREVIEW_SOURCE_TAIL\n'
    review_ref = {'kind': 'task_source', 'root': 'artifact_store',
                  'path': 'source_handles/context_checkpoints/old-review.txt',
                  'sha256': hashlib.sha256(review_bytes).hexdigest(), 'size': len(review_bytes)}
    details = {
        'e1': {'task_id': 'e1', 'status': 'failed', 'reason_code': 'provider_unavailable',
               'title': 'Original report', 'started_at': T, 'ts': T,
               'review_projection': {'panels': [{'panel_id': 'old-review', 'surface': 'task_acceptance',
                   'aggregate_signal': 'FAIL', 'reason': 'Preserved independent review detail', 'actors': [],
                   'applied_source_status': 'available', 'applied_source_ref': review_ref}]},
               'artifacts': [{'name': 'earlier.txt', 'size': 16, 'status': 'ready'}]},
        'e2': {'task_id': 'e2', 'status': 'running', 'title': 'Continue: check totals', 'started_at': T, 'ts': T},
        'e3': {'task_id': 'e3', 'status': 'running', 'title': 'Continue: publish totals', 'started_at': T, 'ts': T},
    }
    history = []
    reads, posts = [], []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_HEAD(self):
            self.do_GET()

        def do_GET(self):
            path = urlsplit(self.path).path
            reads.append(path)
            content_type = 'application/json'
            status = 200
            if path == '/':
                data, content_type = HTML.encode(), 'text/html'
            elif path.startswith('/web/'):
                source = (ROOT / path.lstrip('/')).resolve()
                if source.is_relative_to(ROOT / 'web') and source.is_file():
                    data = source.read_bytes()
                    content_type = mimetypes.guess_type(source.name)[0] or 'text/plain'
                else:
                    data, status = b'not found', 404
            elif path == '/api/tasks/e1/artifacts/old-review.txt':
                data, content_type = review_bytes, 'text/plain'
            elif '/artifacts/' in path:
                data, content_type = b'earlier evidence\n', 'text/plain'
            elif path.startswith('/api/tasks/'):
                data = json.dumps(details.get(path.split('/')[3], {})).encode()
            elif path == '/api/chat/history':
                data = json.dumps({'messages': history, 'window': {'complete': True},
                                   'page_cursor': 'recent', 'has_more': False, 'next_cursor': None}).encode()
            else:
                data = json.dumps({'active_direct_turns': [], 'supervisor_ready': True}).encode()
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            if self.command != 'HEAD':
                self.wfile.write(data)

        def do_POST(self):
            posts.append(self.path)
            self.send_response(500)
            self.end_headers()

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    evidence = Path(os.environ.get('OUROBOROS_UI_EVIDENCE_DIR', tmp_path)) / engine
    evidence.mkdir(parents=True, exist_ok=True)
    facts = {}
    try:
        with sync_playwright() as playwright:
            browser = getattr(playwright, engine).launch()
            try:
                page = browser.new_page(viewport={'width': 1100, 'height': 850})
                errors = PageErrors(page, engine)
                page.goto(f'http://127.0.0.1:{server.server_port}/')
                page.wait_for_function('window.fixture?.instance.hasPaintedHistory()')

                def emit(kind, row):
                    page.evaluate('([kind,row])=>fixture.handlers.get(kind)(row)', [kind, row])

                def narrate(task_id, name, text):
                    row = {'chat_id': 1, 'task_id': task_id, 'role': 'assistant', 'is_progress': True,
                           'narration': True, 'text': text, 'suggested_name': name, 'ts': T}
                    emit('chat', row)
                    emit('task_named', {'task_id': task_id, 'suggested_name': name})
                    return row

                def expand(card):
                    if card.get_attribute('data-expanded') != '1':
                        card.locator(':scope > [data-live-summary-button] [data-live-toggle]').click()

                first_row = narrate('e1', 'Original report', 'Checking original report')
                review_row = {'chat_id': 1, 'task_id': 'e1', 'role': 'system',
                              'system_type': 'acceptance_late_settlement', 'card_row': 'reviews',
                              'card_row_id': 'review:e1', 'text': 'Earlier independent criticism', 'ts': T}
                emit('chat', review_row)
                first = page.locator('.chat-live-card[data-task-id="e1"]')
                expand(first)
                first.locator('[data-result-files]').wait_for()
                page.evaluate('window.originalReviewDoor=document.querySelector(".chat-live-card[data-task-id=e1] [data-live-review-summary]")')
                page.screenshot(path=str(evidence / 'before.png'))
                # A bound claim never folds the predecessor, even with a successor present.
                details['e1']['continued_by'] = {'state': 'bound', 'successor_task_id': 'e2'}
                second_row = narrate('e2', 'Continue: check totals', 'Checking new totals')
                assert first.is_visible()
                details['e1']['continued_by']['state'] = 'admitted'
                emit('chat', {'chat_id': 1, 'task_id': 'e1', 'role': 'system', 'system_type': 'task_summary',
                              'status': 'failed', 'continued_by': details['e1']['continued_by'], 'ts': T})
                second = page.locator('.chat-live-card[data-task-id="e2"]')
                expand(second)
                second.scroll_into_view_if_needed()
                page.screenshot(path=str(evidence / 'names.png'))
                disclosure = second.locator('details[data-episode-task="e1"]')
                disclosure.locator(':scope > summary').click()
                disclosure.locator('[data-result-files]').wait_for(state='visible')
                disclosure.locator('[data-live-review-summary]').click()
                disclosure.locator('[data-review-group-toggle]').click()
                disclosure.locator('[data-review-facts] > summary').click()
                assert page.evaluate('window.originalReviewDoor===document.querySelector("details[data-episode-task=e1] [data-live-review-summary]")')
                read = disclosure.locator('[data-review-read-url]')
                source_url = read.get_attribute('data-review-read-url')
                assert source_url.startswith('/api/tasks/e1/artifacts/old-review.txt?source=')
                assert page.request.get(f'http://127.0.0.1:{server.server_port}' + source_url).body() == review_bytes
                read.click()
                reader = page.locator('dialog[open]:has(.document-reader-panel)')
                reader.locator('.document-reader-body', has_text='REVIEW_SOURCE_TAIL').wait_for()
                with page.expect_download() as review_download:
                    reader.locator('[data-reader-action="download"]').click()
                review_target = evidence / 'old-review.txt'
                review_download.value.save_as(review_target)
                assert review_target.read_bytes() == review_bytes
                page.keyboard.press('Escape')
                assert disclosure.locator('[data-review-attempt-detail]').get_by_text('Preserved independent review detail', exact=True).is_visible()
                assert 'Earlier independent criticism' in disclosure.inner_text()
                assert 'Continue: check totals' in second.locator('[data-live-episodes]').inner_text()
                assert second.locator(':scope > [data-live-summary-button] [data-live-title]').inner_text() == 'Original report'
                assert first.is_hidden()
                assert page.locator('.chat-live-card:visible').count() == 1
                # The original physical timeline remains functional, not a DOM screenshot.
                link = disclosure.locator('a').filter(has_text='earlier.txt')
                assert '/e1/' in link.get_attribute('href')
                with page.expect_download() as download:
                    link.click()
                target = evidence / 'downloaded-earlier.txt'
                download.value.save_as(target)
                assert target.read_bytes() == b'earlier evidence\n'
                # Late predecessor rows update the SAME disclosed nodes, not the current episode.
                details['e1']['artifacts'].append({'name': 'late.txt', 'size': 16, 'status': 'ready'})
                emit('chat', {**review_row, 'card_row_id': 'review:e1:late', 'text': 'Late predecessor criticism'})
                assert 'Late predecessor criticism' in disclosure.inner_text()
                assert page.evaluate('window.originalReviewDoor===document.querySelector("details[data-episode-task=e1] [data-live-review-summary]")')
                assert disclosure.locator('[data-review-read-url]').get_attribute('data-review-read-url') == source_url
                late_link = disclosure.locator('a').filter(has_text='late.txt')
                late_link.wait_for(state='visible')
                assert '/e1/' in late_link.get_attribute('href')
                with page.expect_download() as late_download:
                    late_link.click()
                late_target = evidence / 'downloaded-late.txt'
                late_download.value.save_as(late_target)
                assert late_target.read_bytes() == b'earlier evidence\n'
                assert second.locator(':scope > [data-live-summary-button] [data-live-title]').inner_text() == 'Original report'
                assert second.locator(':scope > [data-live-summary-button] [data-live-phase]').inner_text() == 'Working'
                assert not disclosure.locator('[data-continue-task], [data-cancel-run]').count()
                page.evaluate('fixture.instance.hydrateStateSnapshot({supervisor_ready:true,active_chat_activities_complete:true,active_chat_activities:[{task_id:"e2",activity_id:"e2",chat_id:1,kind:"managed_task",phase:"working",started_at:1792224000}]})')
                assert second.get_by_role('button', name='Stop…', exact=True).count() == 1
                page.screenshot(path=str(evidence / 'after.png'))
                facts['live'] = {'one_visible_card': page.locator('.chat-live-card:visible').count(),
                                 'earlier_text': disclosure.inner_text(), 'current_text': second.inner_text()}
                # A third Continue rehomes disclosures without nesting or copying their contents.
                details['e2'].update(status='failed', continued_by={'state': 'admitted', 'successor_task_id': 'e3'})
                third_row = narrate('e3', 'Continue: publish totals', 'Publishing totals')
                emit('chat', {'chat_id': 1, 'task_id': 'e2', 'role': 'system', 'system_type': 'task_summary',
                              'status': 'failed', 'continued_by': details['e2']['continued_by'], 'ts': T})
                third = page.locator('.chat-live-card[data-task-id="e3"]')
                expand(third)
                assert third.locator('details[data-episode-task]').count() == 2
                assert third.locator('details[data-episode-task="e1"]').get_attribute('open') is not None
                assert page.evaluate('window.originalReviewDoor===document.querySelector("details[data-episode-task=e1] [data-live-review-summary]")')
                assert third.locator('details[data-episode-task="e1"] [data-review-read-url]').get_attribute('data-review-read-url') == source_url
                assert third.locator('[data-result-files]').count() == 1
                assert page.locator('.chat-live-card:visible').count() == 1
                # Rebuild from history in reverse physical arrival order, not the live objects.
                history[:] = [dict(third_row, history_id='h3'), dict(second_row, history_id='h2'),
                              dict(first_row, history_id='h1'), dict(review_row, history_id='hr'),
                              *({**first_row, 'text': f'Earlier step {n}', 'history_id': f'he{n}'} for n in range(40)),
                              {'chat_id': 1, 'task_id': 'e2', 'role': 'system', 'system_type': 'task_summary',
                               'status': 'failed', 'continued_by': details['e2']['continued_by'], 'ts': T, 'history_id': 'hs2'},
                              {'chat_id': 1, 'task_id': 'e1', 'role': 'system', 'system_type': 'task_summary',
                               'status': 'failed', 'continued_by': details['e1']['continued_by'], 'ts': T, 'history_id': 'hs1'}]
                page.reload()
                page.wait_for_function('window.fixture?.instance.hasPaintedHistory()')
                third = page.locator('.chat-live-card[data-task-id="e3"]')
                expand(third)
                assert third.locator('details[data-episode-task]').evaluate_all(
                    '(nodes)=>nodes.map(node=>node.dataset.episodeTask)') == ['e1', 'e2'], 'newer-link-first backfill keeps episode order'
                old = third.locator('details[data-episode-task="e1"]')
                old.locator(':scope > summary').click()
                old.locator('[data-result-files]').wait_for(state='visible')
                assert 'Earlier independent criticism' in old.inner_text()
                assert 'Continue: check totals' in third.locator('[data-live-episodes]').inner_text()
                assert page.locator('.chat-live-card:visible').count() == 1
                # A newest-page refresh evicts the old physical rows. Their content controller is
                # still owned by the mounted successor; re-adding the page must use it, not a dead copy.
                page.evaluate('window.oldTimeline=document.querySelector("details[data-episode-task=e1] [data-live-timeline]")')
                full_history = history[:]
                newer_row = {'chat_id': 1, 'role': 'user', 'text': '\n'.join(f'Newer message line {n}' for n in range(60)),
                             'ts': '2026-10-17T09:00:00Z', 'history_id': 'newer'}
                history[:] = [dict(third_row, history_id='h3'), newer_row]
                old.locator(':scope > summary').click()
                page.locator('#chat-input').focus()
                page.evaluate('fixture.instance.showLatest()')
                page.evaluate('new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)))')
                page.evaluate('fixture.instance.refreshHistory({revision:1})')
                assert old.locator('[data-result-files]').count() == 1
                assert 'Earlier independent criticism' not in old.text_content(), 'the prior page really was evicted'
                assert page.evaluate('window.oldTimeline===document.querySelector("details[data-episode-task=e1] [data-live-timeline]")')
                history[:] = full_history
                page.evaluate('fixture.instance.refreshHistory({revision:2})')
                old.locator(':scope > summary').click()
                assert 'Earlier independent criticism' in old.inner_text()
                facts['eviction_and_readd'] = True
                old.locator('[data-live-review-summary]').click()
                old.locator('[data-review-group-toggle]').click()
                old.locator('[data-review-facts] > summary').click()
                assert old.locator('[data-review-attempt-detail]').get_by_text('Preserved independent review detail', exact=True).is_visible()
                # A long earlier episode keeps the expanded card's bounded timeline box and the one focus ring.
                box = old.locator(':scope > [data-live-timeline]').evaluate(
                    '(n)=>[getComputedStyle(n).maxHeight,getComputedStyle(n).overflowY,n.clientHeight,n.scrollHeight]')
                assert box[:2] == ['420px', 'auto'] and box[2] <= 420 < box[3], box
                page.keyboard.press('Shift')  # keyboard modality: script focus then counts as keyboard focus
                old.locator(':scope > summary').focus()
                ring = old.locator(':scope > summary').evaluate('''(node) => {
                    const probe = document.createElement('span');
                    probe.style.color = 'var(--focus-accent-border)';
                    document.body.append(probe);
                    const token = getComputedStyle(probe).color;
                    probe.remove();
                    const style = getComputedStyle(node);
                    return {visible: node.matches(':focus-visible'), token,
                            outline: [style.outlineStyle, style.outlineWidth, style.outlineOffset, style.outlineColor]};
                }''')
                if engine == 'chromium':  # WebKit's :focus-visible after a programmatic focus is engine-specific
                    assert ring['visible'] and ring['outline'] == ['solid', '2px', '2px', ring['token']], ring
                page.screenshot(path=str(evidence / 'history-box-focus.png'))
                facts.update(history_box=box, focus_ring=ring)
                page.set_viewport_size({'width': 390, 'height': 850})
                page.screenshot(path=str(evidence / 'reload-narrow.png'))
                facts['reload'] = third.inner_text()
                page.evaluate('fixture.instance.destroy(); fixture.instance.destroy()')
                assert page.evaluate('fixture.handlers.size') == 0
                assert page.locator('.chat-live-card').count() == 0
                assert not errors.unexpected(), errors
                assert not posts, posts
                facts.update(errors=errors, posts=posts, sent=page.evaluate('fixture.sent'), destroyed=True)
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)
        (evidence / 'facts.json').write_text(json.dumps(facts, indent=2), encoding='utf-8')
