"""Production chat card over real terminal/review projections, isolated static server."""
import json
import os
import re
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest

from tests.test_review_skill import harness  # noqa: F401 -- shared offline pytest fixture
from tests.test_skill_review_runner import _reset_queue
from tests.test_history_retention_browser import HTML, WEB
from ouroboros.test_evidence import diagnostic_directory

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]


@pytest.fixture
def continuity_rows(harness):  # noqa: F811 -- imported pytest fixture
    from ouroboros import skill_source_install as install
    from ouroboros.gateway.history import _collect_chat_rows, _fold_task_bound_skill_reviews
    from ouroboros.terminal_projection import _project_row
    from ouroboros.outcomes import derive_loop_outcome
    from tests.test_candidate_remainder import _start
    from ouroboros import mutation_attribution as attribution
    from ouroboros.task_results import load_task_result, write_task_result
    from ouroboros.gateway.tasks import _task_get_response
    from types import SimpleNamespace
    import threading

    h = harness
    # The real completion producer reads a Git status captured after the task's commit.
    repo, data = _start(h.drive.parent / 'candidate-proof')
    (repo / 'clean.txt').write_text('post-commit edit\n')
    (repo / 'new.py').write_text('value = 2\n')
    attribution.record_terminal_mutation_candidates(data, 'candidate-task')
    projection = attribution.load_mutation_evidence_projection(data, 'candidate-task')
    outcome = derive_loop_outcome('The candidate is saved.', {}, {'mutation_attribution': projection})
    write_task_result(data, 'candidate-task', 'completed', outcome_axes=outcome['outcome_axes'])
    response = _task_get_response(SimpleNamespace(path_params={'task_id': 'candidate-task'},
        app=SimpleNamespace(state=SimpleNamespace(drive_root=data))))
    assert response.status_code == 200
    detail = json.loads(response.body)
    assert detail['mutation_evidence'] == load_task_result(data, 'candidate-task')['mutation_evidence']
    done = _project_row('candidate-task', {'status': 'completed', 'outcome_axes': outcome['outcome_axes'],
        'reason_code': 'final_message'}, {}, {'chat_id': 1})
    done.update(role='system', system_type='task_summary', content=done['text'],
                task_terminal_status='completed', suggested_name='Settings persistence repair')
    h.ctx.current_chat_id = 1
    h.ctx.task_metadata = {'root_task_id': h.ctx.task_id}
    _reset_queue()
    try:
        install.start_review(h.drive, 'weather', ctx=h.ctx, reviewers=['t1'], reason='small payload')
        [thread] = [t for t in threading.enumerate() if t.name == 'skill-install-review-weather']
        thread.join(120)
        assert not thread.is_alive()
    finally:
        _reset_queue()
    rows, _ = _collect_chat_rows(h.drive / 'logs' / 'chat.jsonl', h.drive / 'archive', 30, lambda *_: True, {})
    reviews = [row for row in _fold_task_bound_skill_reviews(rows) if row.get('system_type') == 'skill_review']
    assert len(reviews) == 1 and reviews[0]['review_group']['presentation_owner_task_id'] == h.ctx.task_id
    return done, detail, reviews[0], h.ctx.task_id


@pytest.mark.parametrize(('engine', 'width', 'theme', 'inherited'), [
    ('webkit', 1100, 'dark', False), ('webkit', 1100, 'light', False),
    ('webkit', 390, 'dark', False), ('webkit', 390, 'light', False), ('chromium', 1100, 'dark', False),
    ('webkit', 1100, 'dark', True),
])
def test_actual_candidate_and_origin_review_cards(engine, width, theme, inherited, continuity_rows, tmp_path):
    if os.environ.get('OUROBOROS_RUN_UI_SMOKE') != '1':
        pytest.skip('Set OUROBOROS_RUN_UI_SMOKE=1')
    playwright = pytest.importorskip('playwright.sync_api')
    done, detail, review, review_task = continuity_rows
    candidate_id = 'candidate-child' if inherited else 'candidate-task'
    done['task_id'] = candidate_id  # A child can cite its owning root's existing detail route.

    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            if self.path == '/fixture':
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.end_headers()
                self.wfile.write(HTML.encode())
            else:
                super().do_GET()

    server = ThreadingHTTPServer(('127.0.0.1', 0), partial(Handler, directory=str(WEB)))
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    evidence = Path(os.environ.get('OUROBOROS_UI_EVIDENCE_DIR') or diagnostic_directory(
        tmp_path / 'browser-evidence', 'candidate-review-continuity'))
    evidence.mkdir(parents=True, exist_ok=True)
    prefix = f'{engine}-{width}-{theme}' + ('-inherited' if inherited else '')
    (evidence / f'{prefix}-producer-rows.json').write_text(json.dumps([done, review], ensure_ascii=False, indent=2))
    try:
        with playwright.sync_playwright() as pw:
            browser = getattr(pw, engine).launch(headless=True)
            try:
                page = browser.new_page(viewport={'width': width, 'height': 750})
                errors = []
                page.on('pageerror', lambda error: errors.append(str(error)))
                source_reads = []

                def api(route):
                    url = route.request.url
                    payload = detail if '/api/tasks/candidate-task' in url else {'messages': [], 'entries': [], 'active_direct_turns': []}
                    if inherited and '/api/tasks/candidate-task' in url:
                        source_reads.append(url)
                        if len(source_reads) == 1:
                            route.fulfill(status=503, content_type='application/json', body='{"error":"temporarily unavailable"}')
                            return
                        if len(source_reads) == 2:
                            payload = json.loads(json.dumps(detail))
                            payload['mutation_evidence']['terminal_candidate_snapshot']['captured_at'] = 'different observation'
                    if '/api/tasks/candidate-child' in url:
                        payload = done
                    if '/api/tasks/' + review_task in url:
                        payload = {'task_id': review_task, 'status': 'running'}
                    route.fulfill(content_type='application/json', body=json.dumps(payload))

                page.route('**/api/**', api)
                page.goto(f'http://127.0.0.1:{server.server_port}/fixture')
                page.wait_for_function('window.fixtureReady === true')
                page.evaluate('(theme) => document.documentElement.dataset.theme=theme', theme)
                page.evaluate("""({done,review,review_task}) => {
                    emit('chat',{task_id:done.task_id,chat_id:1,role:'system',is_progress:true,
                        content:'Saving the reviewed change',ts:'2026-10-25T12:00:00Z'});
                    emit('chat',done);
                    emit('chat',{task_id:review_task,chat_id:1,role:'system',is_progress:true,
                        suggested_name:'Install weather',content:'Reviewing the installed weather skill',ts:'2026-10-25T12:02:00Z'});
                    emit('chat',review);
                }""", {'done': done, 'review': review, 'review_task': review_task})
                card = page.locator(f'.chat-live-card[data-task-id="{candidate_id}"]')
                playwright.expect(card.locator('[data-live-phase]')).to_have_text('Done')
                playwright.expect(card.locator('[data-live-activity]')).to_contain_text('2 uncommitted files remain in the saved candidate')
                toggle = card.locator(':scope > [data-live-summary-button] [data-live-disclosure]')
                if card.get_attribute('data-expanded') == '1':
                    toggle.click()
                page.screenshot(path=str(evidence / f'{prefix}-candidate-collapsed.png'))
                toggle.click()
                paths = card.locator(f'[data-live-line-key="candidate-remainder-{candidate_id}"]')
                if inherited:
                    playwright.expect(paths).to_contain_text('2 recorded file paths are not loaded')
                    paths.locator('[data-live-line-toggle]').click()
                    playwright.expect(paths).to_contain_text('could not be loaded')
                    paths.locator('[data-live-line-toggle]').click()
                    paths.locator('[data-live-line-toggle]').click()
                    playwright.expect(paths).to_contain_text('unavailable for this observation')
                    paths.locator('[data-live-line-toggle]').click()
                    paths.locator('[data-live-line-toggle]').click()
                playwright.expect(paths).to_contain_text('clean.txt')
                playwright.expect(paths).to_contain_text('new.py')
                playwright.expect(paths).to_contain_text('HEAD:')
                literal = paths.locator('[data-body-format="text"]')
                playwright.expect(literal).to_have_count(1)
                assert literal.evaluate('(el) => getComputedStyle(el).whiteSpace') == 'pre-wrap'
                assert '\n\nFiles:\nclean.txt\nnew.py' in literal.inner_text()
                page.screenshot(path=str(evidence / f'{prefix}-candidate-details.png'))
                toggle.click()
                owner = page.locator(f'.chat-live-card[data-task-id="{review_task}"]')
                playwright.expect(owner.locator('[data-live-review-summary]')).to_contain_text(re.compile(r'Reviews\s*1'))
                assert page.locator('.chat-live-card[data-task-id="marketplace_install"]').count() == 0
                owner.locator(':scope > [data-live-summary-button] [data-live-disclosure]').click()
                # Reviews opens through its own door since #2278 (docs/DESIGN.md "Reviews inside task cards").
                owner.locator('[data-live-review-summary]').click()
                playwright.expect(owner.locator('[data-review-group-toggle]')).to_be_visible()
                playwright.expect(owner.locator('[data-review-group-toggle]')).to_contain_text('Skill review')
                owner.scroll_into_view_if_needed()
                page.screenshot(path=str(evidence / f'{prefix}-origin-review.png'))
                assert card.evaluate('(el) => el.scrollWidth <= el.clientWidth + 1')
                assert owner.evaluate('(el) => el.scrollWidth <= el.clientWidth + 1')
                assert not errors
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)
        assert not thread.is_alive()
