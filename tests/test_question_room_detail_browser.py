"""Real task-detail and census into native Main forms, including hidden publication."""
import json
import re
import socket
import traceback

import pytest

from tests.test_question_room_detail import question_world
from tests._shared import producer_evidence_dir
from tests.test_ui_smoke_playwright import direct_server_with_data as direct_server_with_data
from tests.test_main_shared_state_browser import _REAL_SOCKET_DROP

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]


def test_actual_detail_recovers_native_main_context_and_hidden_publication(direct_server_with_data):
    from playwright.sync_api import expect, sync_playwright
    from ouroboros.owner_quiz import mark_published, quiz_states
    from supervisor.message_bus import log_chat

    server = direct_server_with_data
    root = server['data_dir']
    server['stop_server']()
    question_world(root)
    tids = ['context-main', 'hidden-main', 'plain-main']
    for tid in tids:
        block = quiz_states(root, tid)['q']
        log_chat('out', 1, 1, block['question'], task_id=tid, record_type='quiz', quiz=block,
                 drive_root=root, require_write=True)
        mark_published(root, tid, 'q')
    server['start_server']()
    evidence = producer_evidence_dir(root.parent, "room-proof",
                                     env='OUROBOROS_BROWSER_EVIDENCE_OUT') / 'room-proof'
    evidence.mkdir(parents=True, exist_ok=True)
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        refused = f'ws://127.0.0.1:{probe.getsockname()[1]}/ws'
    held, reads, posts, errors, stages = [], [], [], [], []
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={'width': 1100, 'height': 900})
        page.add_init_script(f'({_REAL_SOCKET_DROP})({json.dumps(refused)})')
        page.on('pageerror', lambda e: errors.append(str(e)))
        page.on('request', lambda r: posts.append(r.url) if r.method == 'POST' and '/api/decisions' in r.url else None)

        def hold(route):
            response = route.fetch()  # actual task-detail producer; no body replacement
            reads.append(response.json())
            held.append((route, response))

        try:
            page.goto(server['url'], wait_until='domcontentloaded')
            page.wait_for_function('() => window.__ouroWs?.isConnected()')
            # The served-page guard reloads when its first SHA is learned only after a reconnect.
            # Wait for that existing proof before deliberately dropping the socket.
            served_sha = page.request.get(server['url'] + '/api/state').json().get('sha')
            if served_sha:
                page.wait_for_function('sha => window.__ouroWs?._lastSha === sha', arg=served_sha)
            cards = [page.locator(f'#chat-messages .chat-quiz-card[data-task-id="{tid}"]') for tid in tids]
            for card, tid in zip(cards, tids):
                expect(card).to_have_count(1)
                expect(card.locator('.chat-quiz-comment')).to_be_enabled()
                card.locator('.chat-quiz-comment').fill('Keep ' + tid)
                card.evaluate('(node, tid) => { (window.__originalForms ||= {})[tid] = node; }', tid)
            stages.append('three native Main forms with real canonical records')
            page.route(re.compile(r'/api/tasks/(context-main|hidden-main|plain-main)$'), hold)

            def close_and_open():
                page.evaluate('window.__refuseSockets = true; window.__ouroWs.ws.close()')
                page.wait_for_function('() => !window.__ouroWs.isConnected()')
                for card in cards:
                    expect(card.locator('.chat-quiz-comment')).to_be_disabled()
                page.evaluate('window.__refuseSockets = false')
                page.wait_for_function('() => window.__ouroWs.isConnected()', timeout=30_000)

            def wait_reads(count):
                for _ in range(300):
                    if len(held) >= count:
                        return
                    page.wait_for_timeout(50)
                raise AssertionError(f'expected {count} real detail reads, got {len(held)}')

            close_and_open()
            wait_reads(3)
            stale = list(held)
            held.clear()
            close_and_open()
            wait_reads(3)
            for route, response in stale:
                route.fulfill(response=response)
            page.wait_for_timeout(300)
            for card in cards:
                expect(card.locator('.chat-quiz-comment')).to_be_disabled()
            stages.append('pre-close detail landed after reopen without restoring authority')
            page.screenshot(path=str(evidence / 'stale.png'))
            while held:
                route, response = held.pop(0)
                route.fulfill(response=response)
            page.unroute(re.compile(r'/api/tasks/(context-main|hidden-main|plain-main)$'), hold)
            for card, tid in zip(cards, tids):
                expect(card.locator('.chat-quiz-comment')).to_be_enabled(timeout=10_000)
                expect(card.locator('.chat-quiz-comment')).to_have_value('Keep ' + tid)
                assert card.evaluate('(node, tid) => node === window.__originalForms[tid]', tid)
            stages.append('fresh producer proof restores same forms and drafts')
            # The same shipped reader consumes real HTTP detail for Project-local and mirrored
            # forms too. No server body is formatted by this fixture, and no decision is posted.
            proof = page.evaluate("""async () => {
                const { createChatDecision } = await import('/static/modules/chat_decision.js');
                const make = isMain => createChatDecision({ isMain, apiFetch: () => { throw Error('unexpected POST'); },
                    frameNode: (_, node) => node, showToast: () => {},
                    fetchDetail: async id => (await fetch('/api/tasks/' + id)).json() });
                const main = make(true), project = make(false), outcomes = {};
                try {
                    for (const [id, room] of [['converted', 'near-b'], ['project-local', 'context-a']]) {
                        outcomes[id] = Boolean(await main.readQuestion(id, 'q', room))
                            && Boolean(await project.readQuestion(id, 'q', room));
                        outcomes[id + '-foreign-main'] = await main.readQuestion(id, 'q', '') === null;
                    }
                    outcomes['absent-parent'] = Boolean(await main.readQuestion('missing-parent', 'q', ''));
                    for (const id of ['foreign', 'missing-address'])
                        outcomes[id] = await main.readQuestion(id, 'q', '') === null;
                    outcomes['wrong-quiz'] = await main.readQuestion('plain-main', 'absent', '') === null;
                    return outcomes;
                } finally { main.destroy(); project.destroy(); }
            }""")
            assert all(proof.values()), proof
            stages.append({'shared_reader_real_http': proof})
            assert not posts and not errors
            page.screenshot(path=str(evidence / 'recovered.png'))
        except Exception:
            (evidence / 'failure.txt').write_text(traceback.format_exc())
            raise
        finally:
            page.screenshot(path=str(evidence / 'end.png'))
            (evidence / 'receipt.json').write_text(json.dumps({'stages': stages, 'details': reads,
                'posts': posts, 'page_errors': errors, 'candidate_identity': server['candidate_identity']}, indent=2))
            for route, _ in held:
                try:
                    route.abort()
                except Exception as exc:
                    (evidence / 'cleanup-error.txt').write_text(str(exc))
            browser.close()
