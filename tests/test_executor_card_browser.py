"""Actual chat renderers over deterministic progress/history, without model calls."""
from __future__ import annotations
import json
import pytest
from tests.test_subscription_setup_browser import subscription_ui as subscription_ui, capture
from tests.test_delegated_activity_browser import engine_ui as engine_ui  # chromium and webkit

pytestmark = [pytest.mark.ui_browser, pytest.mark.serial]


def executor_sentence(card):
    # The chip's detail sentence (#360 A) is the first row of the card's own details (refined A, 21.10).
    row = card.locator(':scope > .chat-live-details > [data-live-facts] > div').first
    assert row.locator('dt').text_content() == 'Executor'
    return row.locator('dd')


def progress(task, text, **extra):
    return dict(role='assistant', direction='out', is_progress=True, task_id=task,
                text=text, content=text, ts='2026-09-09T10:00:00Z', **extra)


@pytest.mark.parametrize('width', [390, 1440])
@pytest.mark.parametrize('linux_metrics', [False, True], ids=['native-font', 'linux-metrics'])
def test_observed_executor_role_activity_and_project_pointer_live_replay(subscription_ui, width, linux_metrics):
    ui = subscription_ui
    page = ui['page']
    page.set_viewport_size({'width':width,'height':800})
    observation = dict(task_id='child',task_attempt='1',run_id='run-a',attempt_id='a01',
                       harness_id='cursor',phase='harness.event',revision=8,
                       model='cursor-grok-4.6-xhigh-fast',model_source='requested')
    child = progress('child','Checking every visible control and its alignment.',
                     subagent_event='running',subagent_task_id='child',parent_task_id='root',
                     root_task_id='root',delegation_role='subagent',subagent_role='UI reviewer',
                     model='openai/gpt-6-astra',executor_route='cursor',executor_observation=observation)
    rows = [progress('root','Inspecting the shared UI system. '+ 'Useful activity text. '*20,suggested_name='UI coherence', model='openai/gpt-6-astra'), child]
    history = {'messages': rows, 'progress': [], 'page_cursor': 'executor-recent',
               'next_cursor': 'executor-older', 'has_more': True,
               'window': {'complete': False, 'truncated_by': ['quota']},
               'coverage': {'v': 1, 'view': 'executor-proof', 'upper': {'chat': 0, 'progress': 7},
                            'spans': {'chat': {'from': 0, 'to': 0, 'chain': 'empty', 'gaps': []},
                                      'progress': {'from': 5, 'to': 7, 'chain': 'retained', 'gaps': ['incomplete_live_line']}}}}
    page.route('**/api/chat/history*',lambda route:route.fulfill(content_type='application/json',body=json.dumps(history)))
    page.goto(ui['url'])
    page.wait_for_selector('#chat-input')
    # A real secondary instance with its own mock WS boundary and the same history API.
    page.evaluate('''async () => {
        const {createChatInstance} = await import('/static/modules/chat.js');
        const {createStateSnapshotSequencer} = await import('/static/modules/chat_activity.js');
        const mount=document.createElement('aside'); mount.id='executor-proof';
        mount.className='project-panel open'; document.body.append(mount);
        const handlers=new Map(); const ws={on(type,fn){
            if(!handlers.has(type))handlers.set(type,new Set());handlers.get(type).add(fn);
            return()=>handlers.get(type).delete(fn);},isConnected:()=>true,send:()=>{}};
        window.executorProof={handlers,instance:createChatInstance({ws,state:{activePage:'chat',projectChatIds:new Set()},
            updateUnreadBadge:()=>{},stateSnapshots:createStateSnapshotSequencer(()=>{}),chatId:101,projectId:'proof',idPrefix:'proof',mountEl:mount,asPanel:true})};
        await executorProof.instance.refreshHistory();
    }''')
    card = page.locator('#executor-proof [data-task-id="child"]')
    card.wait_for()
    root_card = page.locator('#executor-proof [data-task-id="root"]')
    root_meta = root_card.locator(':scope > .chat-live-foot > .chat-live-meta')
    root_facts = root_card.locator(':scope > .chat-live-details > [data-live-facts]')
    # A root runs no executor: its compact line names its own model; the labelled row is in its details.
    assert 'gpt-6-astra' in root_meta.inner_text()
    assert 'Agent model' in root_facts.text_content() and 'gpt-6-astra' in root_facts.text_content()
    # A helper's usage shares the task id but must not replace its coordinator.
    page.evaluate('''async () => {
        for (const fn of executorProof.handlers.get('log') || []) fn({chat_id:101, data:{
            type:'llm_usage', task_id:'root', model:'helper-model', model_category:'websearch',
            ts:'2026-09-09T10:00:30Z'}});
        await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    }''')
    assert 'gpt-6-astra' in root_meta.inner_text() and 'gpt-6-astra' in root_facts.text_content()
    assert 'helper-model' not in root_meta.inner_text() + root_facts.text_content()
    assert card.locator('.chat-live-title').first.inner_text() == 'UI reviewer'
    # The child's compact line names its executor and the executor's own (requested) model; the coordinating
    # model is a labelled row of its details, so it never reads as the executor's.
    child_meta = card.locator(':scope > .chat-live-foot > .chat-live-meta')
    child_facts = card.locator(':scope > .chat-live-details > [data-live-facts]').text_content()
    assert 'requested cursor-grok-4.6-xhigh-fast' in child_meta.inner_text()
    assert 'gpt-6-astra' not in child_meta.inner_text()
    assert 'Coordinator' in child_facts and 'gpt-6-astra' in child_facts
    # #360 A: the chip keeps the observation word; the detail sentence is a row of the
    # expanded card's details, so the collapsed child stays one quiet identity row.
    assert card.locator('.chat-live-executor-chip').inner_text().strip() == 'Cursor · last update'
    detail = executor_sentence(card)
    assert '(requested)' in detail.text_content() and not detail.is_visible()
    activity=card.locator('.chat-live-activity').first
    assert activity.is_visible() and activity.inner_text().startswith('Checking every')
    assert activity.evaluate('e=>getComputedStyle(e).webkitLineClamp') == '1'
    # root card one press away. Outside the real panel bar it stays in the instance's status bar.
    trigger = page.locator('#executor-proof .rw-trigger')
    assert trigger.locator('.status-badge').count() == 1
    if linux_metrics:
        # Exercise the Linux status-pill width without bundling another font.
        page.add_style_tag(content='''
            #executor-proof .rw-trigger .status-badge { min-width: 81px; }
        ''')
    assert trigger.bounding_box()['height'] <= 40
    # The common history gap disclosure stays at the in-feed Load-older edge, not in the header bar.
    bar = page.locator('#executor-proof .chat-panel-statusbar')
    assert bar.locator('.chat-history-status').count() == 0
    note = page.locator('#executor-proof .chat-load-older .chat-load-older-note')
    assert note.inner_text() == 'Some saved history is not loaded. Shown messages may have gaps.'
    assert page.locator('#executor-proof .chat-load-older-note').count() == 1
    geometry = page.locator('#executor-proof').evaluate('''panel => {
        const box = selector => panel.querySelector(selector).getBoundingClientRect();
        return {control: box('.rw-trigger'), note: box('.chat-load-older-note'), bar: box('.chat-panel-statusbar')};
    }''')
    assert geometry['note']['y'] >= geometry['control']['bottom']
    assert geometry['note']['x'] >= geometry['bar']['x']
    assert geometry['note']['right'] <= geometry['bar']['right'] + 1
    input_box=page.locator('#proof-input')
    input_box.fill('Message stays in this Project')
    trigger.click()
    page.locator('#executor-proof .rw-shade .rw-main', has_text='UI coherence').click()
    assert input_box.input_value() == 'Message stays in this Project'
    page.locator('#executor-proof .rw-shade').wait_for(state='hidden')  # it folds away, then hides
    terminal={**child,'is_progress':True,'subagent_event':'completed','status':'completed','result':'Checks complete',
              'ts':'2026-09-09T10:01:00Z','execution_evidence':{'delegated_runs_started':1,'delegated_runs_settled':1,
              'delegated_runs_failed':0,'harness_models':['Cursor Grok 4.6 Extra High Fast']}}
    page.evaluate('''frame=>{for(const fn of executorProof.handlers.get('chat')||[])fn({...frame,chat_id:101})}''',terminal)
    # The settled observed model replaces the progress report's requested one on the compact line.
    page.wait_for_function("document.querySelector('#executor-proof [data-task-id=child] > .chat-live-foot > .chat-live-meta')"
                           ".textContent.includes('Cursor Grok 4.6 Extra High Fast')")
    assert 'requested cursor-grok' not in child_meta.inner_text() and 'gpt-6-astra' not in child_meta.inner_text()
    child_facts = card.locator(':scope > .chat-live-details > [data-live-facts]').text_content()
    assert 'Coordinator' in child_facts and 'gpt-6-astra' in child_facts
    capture(page,f'executor-card-{width}-{"linux-metrics" if linux_metrics else "native-font"}')
    page.evaluate('executorProof.instance.destroy()')
    assert page.locator('#executor-proof .rw-trigger').count()==0


# #360 A matrix: the chip names the helper; exceptions stay one visible phrase; the
# detail sentence is text on the expanded card (pointer, keyboard, touch, copy).
OBSERVED = dict(task_attempt='1', run_id='run-m', attempt_id='a01', harness_id='cursor', phase='harness.event',
                revision=3, model='cursor-grok-4.6-xhigh-fast', model_source='requested')


def child(task, role, event='running', evidence=None, **extra):
    if evidence is not None:
        extra['execution_evidence'] = {'subscription_cost_usd': None, 'harness_models': [], **evidence}
    if event != 'running':
        extra = {'status': 'failed' if event == 'failed' else 'completed', 'result': f'{role} result', **extra}
    return progress(task, f'{role} reports its step.', subagent_event=event, subagent_task_id=task,
                    parent_task_id='matrix-root', root_task_id='matrix-root', delegation_role='subagent',
                    subagent_role=role, model='openai/gpt-6-astra', **extra)


def runs(started, settled, ok=None, **extra):
    counts = {'delegated_runs_started': started, 'delegated_runs_settled': settled}
    if ok is not None:
        counts.update(delegated_runs_succeeded=ok, delegated_runs_failed=settled - ok)
    return {**counts, **extra}


MATRIX = (  # task, frame, visible label, a fragment of the detail sentence
    ('m-assigned', child('m-assigned', 'assigned', executor_route='codex'), 'Codex', 'Dispatched to Codex'),
    ('m-observed', child('m-observed', 'observed', executor_route='cursor',
                         executor_observation={**OBSERVED, 'task_id': 'm-observed'}),
     'Cursor · last update', 'model cursor-grok-4.6-xhigh-fast (requested)'),
    ('m-clean', child('m-clean', 'clean', 'completed', runs(2, 2, 2, subscription_cost_usd=0.25), executor_route='codex'),
     'Codex', '2 runs settled (2 ok), $0.25 subscription'),
    ('m-mixed', child('m-mixed', 'mixed', 'completed', runs(2, 2, 1), executor_route='codex'),
     'Codex · 1 of 2 runs failed', '(1 ok, 1 failed)'),
    ('m-unverified', child('m-unverified', 'unverified', 'completed',
                           runs(2, 2, 1, delegated_runs_source_unresolved=1), executor_route='codex'),
     'Codex · 1 of 2 runs unverified', 'unverified work-order coverage'),
    ('m-unsettled', child('m-unsettled', 'unsettled', 'completed', runs(1, 0, 0), executor_route='codex'),
     'Codex · not settled', 'started, none settled'),
    ('m-legacy', child('m-legacy', 'legacy', 'completed', runs(1, 1), executor_route='codex'),
     'Codex · run outcome unknown', '1 run settled, subscription spend undisclosed'),
    ('m-missing', child('m-missing', 'missing evidence', 'completed', executor_route='codex'),
     'Codex · run outcome unknown', 'terminal record has no delegated-run evidence'),
    ('m-no-run', child('m-no-run', 'no run', 'completed', runs(0, 0), executor_route='claude'),
     'Claude Code · no run yet', 'no durable record of a delegated run'),
    ('m-unreadable', child('m-unreadable', 'unreadable', 'completed', runs(0, 0, evidence_read_failed=True),
                           executor_route='codex'), 'Codex · evidence unavailable', 'could not be read'),
    ('m-blocked', child('m-blocked', 'blocked', 'failed', executor_route='claude',
                        reason_code='subagent_executor_unavailable'), 'Claude Code · blocked', 'NOT run'),
)
MOUNT = '''async rows => {
    const {createChatInstance} = await import('/static/modules/chat.js');
    const {createStateSnapshotSequencer} = await import('/static/modules/chat_activity.js');
    const mount = document.createElement('aside'); mount.id = 'executor-matrix';
    mount.className = 'project-panel open'; document.body.append(mount);
    const handlers = new Map(); const ws = {on(type, fn) {
        if (!handlers.has(type)) handlers.set(type, new Set()); handlers.get(type).add(fn);
        return () => handlers.get(type).delete(fn);}, isConnected: () => true, send: () => {}};
    window.executorMatrix = {handlers, instance: createChatInstance({ws, state: {activePage: 'chat', projectChatIds: new Set()},
        updateUnreadBadge: () => {}, stateSnapshots: createStateSnapshotSequencer(() => {}), chatId: 102, projectId: 'matrix',
        idPrefix: 'matrix', mountEl: mount, asPanel: true})};
    await executorMatrix.instance.refreshHistory();
}'''
COPY = '''el => { const s = getSelection(); s.removeAllRanges(); const r = document.createRange();
    r.selectNodeContents(el); s.addRange(r); const text = s.toString(); s.removeAllRanges(); return text; }'''


def open_matrix(ui, page, theme, size):
    page.set_viewport_size(size)
    page.emulate_media(color_scheme=theme)
    rows = [progress('matrix-root', 'Coordinating the helpers.', suggested_name='Helper matrix',
                     model='openai/gpt-6-astra')] + [frame for _, frame, _, _ in MATRIX]
    page.route('**/api/chat/history*', lambda route: route.fulfill(content_type='application/json', body=json.dumps(
        {'messages': rows, 'progress': [], 'has_more': False, 'window': {'complete': True}})))
    page.goto(ui['url'])
    page.wait_for_selector('#chat-input')
    assert page.locator('html').get_attribute('data-theme') == theme
    page.evaluate(MOUNT, rows)
    page.locator('#executor-matrix [data-task-id="m-blocked"]').wait_for()


def part(card, selector):
    # The summary holds chip, title and line; the meta line (executor chip and detail) leads the card's foot.
    return card.locator(f':scope > .chat-live-summary-button {selector}, :scope > .chat-live-foot {selector}')


@pytest.mark.parametrize('size', [{'width': 1100, 'height': 721}, {'width': 375, 'height': 812}],
                         ids=['desktop', 'narrow'])
@pytest.mark.parametrize('theme', ['light', 'dark'])
def test_quiet_executor_label_matrix_and_details_by_pointer_keyboard_touch(engine_ui, theme, size):
    ui = engine_ui
    page = ui['page']
    engine = page.context.browser.browser_type.name
    shot = f'executor-label-{engine}-{theme}-{size["width"]}x{size["height"]}'
    open_matrix(ui, page, theme, size)
    cards = {task: page.locator(f'#executor-matrix [data-task-id="{task}"]') for task, *_ in MATRIX}
    for task, _, label, fragment in MATRIX:
        card = cards[task]
        assert card.get_attribute('data-expanded') == '0', task
        assert part(card, '.chat-live-executor-chip').inner_text().strip() == label, task
        detail = executor_sentence(card)
        assert fragment in detail.text_content(), task
        # Collapsed: the sentence is out of sight and out of the copied line, with no dangling dot.
        assert not detail.is_visible(), task
        copied = ' '.join(card.locator(':scope > .chat-live-foot [data-live-meta]').evaluate(COPY).split())
        assert copied.startswith(label) and fragment not in copied and '· ·' not in copied, (task, copied)
        meta = part(card, '[data-live-meta]')
        assert meta.evaluate('e => e.scrollWidth <= e.clientWidth + 1'), task
    capture(page, f'{shot}-collapsed')
    # The rendered markup equals its own serialization (quotes included), so the
    # meta line's no-op check holds and a reader's selection is never rebuilt away.
    unreadable = next(frame for task, frame, *_ in MATRIX if task == 'm-unreadable')
    # The detail sentence (quotes included) is the Executor row of the details, which round-trips the same way.
    assert page.evaluate('''async frame => {
        const {executorChip} = await import('/static/modules/log_events.js');
        const {executorIdentityMarkup} = await import('/static/modules/harness_presentation.js');
        const html = executorIdentityMarkup(executorChip(frame), {model: 'gpt-6-astra'});
        const probe = document.createElement('div'); probe.innerHTML = html;
        const facts = document.querySelector('#executor-matrix [data-task-id="m-unreadable"] > .chat-live-details > [data-live-facts]');
        const again = document.createElement('dl'); again.innerHTML = facts.innerHTML;
        return probe.innerHTML === html && facts.innerHTML.includes('"none"') && again.innerHTML === facts.innerHTML;
    }''', unreadable)
    # Pointer: the card's Details button opens the sentence as the details' Executor row; the compact line stays as it was.
    mixed = cards['m-mixed']
    part(mixed, '[data-live-disclosure]').click()
    assert mixed.get_attribute('data-expanded') == '1'
    detail_text = executor_sentence(mixed)
    assert detail_text.is_visible()
    sentence = detail_text.inner_text().strip()
    assert sentence == part(mixed, '.chat-live-executor-chip').get_attribute('title')
    copied = ' '.join(part(mixed, '[data-live-meta]').evaluate(COPY).split())
    assert copied.startswith('Codex · 1 of 2 runs failed') and sentence not in copied, copied
    facts_copy = ' '.join(mixed.locator(':scope > .chat-live-details > [data-live-facts]').evaluate(COPY).split())
    assert f'Executor {sentence}' in facts_copy, facts_copy
    assert part(mixed, '.chat-live-executor-chip').inner_text().strip() == 'Codex · 1 of 2 runs failed'
    # The phase chip stays the work indicator: the quiet name is not a running proof.
    assert cards['m-assigned'].locator(':scope > .chat-live-summary-button [data-live-phase]').inner_text() == 'Working'
    capture(page, f'{shot}-expanded')
    # Keyboard: Enter closes and Space reopens through the same control.
    mixed.locator(':scope > .chat-live-summary-button [data-live-disclosure]').focus()
    page.keyboard.press('Enter')
    assert mixed.get_attribute('data-expanded') == '0' and not detail_text.is_visible()
    page.keyboard.press(' ')
    assert mixed.get_attribute('data-expanded') == '1' and detail_text.is_visible()
    # Live: an assignment, then the run's own report, then the terminal receipt.
    emit = 'frame => { for (const fn of executorMatrix.handlers.get("chat") || []) fn({...frame, chat_id: 102}); }'
    live = page.locator('#executor-matrix [data-task-id="m-live"]')
    page.evaluate(emit, {**child('m-live', 'live', executor_route='codex'), 'ts': '2026-09-09T10:02:00Z'})
    live.wait_for()
    assert part(live, '.chat-live-executor-chip').inner_text().strip() == 'Codex'
    page.evaluate(emit, {**child('m-live', 'live', executor_route='codex',
                                executor_observation={**OBSERVED, 'harness_id': 'codex', 'task_id': 'm-live'}),
                         'ts': '2026-09-09T10:02:10Z'})
    page.wait_for_function("document.querySelector('#executor-matrix [data-task-id=\"m-live\"] .chat-live-executor-chip')"
                           ".textContent.includes('last update')")
    page.evaluate(emit, {**child('m-live', 'live', 'completed', runs(3, 3, 2), executor_route='codex'),
                         'ts': '2026-09-09T10:03:00Z'})
    page.wait_for_function("document.querySelector('#executor-matrix [data-task-id=\"m-live\"] .chat-live-executor-chip')"
                           ".textContent.includes('1 of 3 runs failed')")
    # Touch: the same sentence opens by tap on a touch context, not by hover.
    touch = ui['new_page'](has_touch=True, viewport={'width': 375, 'height': 812})
    open_matrix(ui, touch, theme, {'width': 375, 'height': 812})
    unverified = touch.locator('#executor-matrix [data-task-id="m-unverified"]')
    unverified.locator(':scope > .chat-live-summary-button [data-live-disclosure]').tap()
    assert unverified.get_attribute('data-expanded') == '1'
    assert executor_sentence(unverified).is_visible()
    capture(touch, f'{shot}-touch-expanded')
    page.evaluate('executorMatrix.instance.destroy()')
    touch.evaluate('executorMatrix.instance.destroy()')


@pytest.mark.parametrize('theme', ['light', 'dark'])
def test_logs_print_the_executor_detail_beside_the_quiet_chip(engine_ui, theme):
    ui = engine_ui
    page = ui['page']
    page.set_viewport_size({'width': 1100, 'height': 721})
    page.emulate_media(color_scheme=theme)
    entries = [{**frame, 'type': 'send_message'} for _, frame, _, _ in MATRIX]
    page.route('**/api/logs/*', lambda route: route.fulfill(content_type='application/json', body=json.dumps(
        {'entries': entries if '/api/logs/progress' in route.request.url else []})))
    page.goto(ui['url'])
    page.wait_for_selector('#chat-input')
    page.locator('[data-nav-page="dashboard"]').click()
    page.locator('[data-dashboard-tab="logs"]').click()
    page.locator('.log-task-card[data-task-group="m-blocked"]').wait_for()
    for task, _, label, fragment in MATRIX:
        card = page.locator(f'.log-task-card[data-task-group="{task}"]')
        chip = card.locator('[data-task-executor]')
        assert chip.inner_text().strip() == label, task
        detail = card.locator('[data-task-executor-detail]')
        assert detail.is_visible() and fragment in detail.inner_text(), task
        assert detail.inner_text().strip() == chip.get_attribute('title'), task
    page.locator('.log-task-card[data-task-group="m-mixed"]').scroll_into_view_if_needed()
    capture(page, f'executor-logs-{page.context.browser.browser_type.name}-{theme}')
    # Standalone Logs owns an attribute sink as well as the new text detail.
    # Quotes in a valid producer-shaped value must remain text, not new attributes.
    assert page.evaluate('''async () => {
        const {initLogs} = await import('/static/modules/logs.js');
        const mount = document.createElement('section'); document.body.append(mount);
        const handlers = new Map(); const ws = {on(type, fn) { handlers.set(type, fn); return () => {}; }};
        initLogs({ws, state: {activePage: 'dashboard', dashboardActiveSubtab: 'logs'}, mount});
        handlers.get('log')({data: {type: 'executor_note', task_id: 'quote-child', executor_route: 'codex',
            ts: '2026-10-13T06:00:00Z', executor_observation: {task_id: 'quote-child', run_id: 'run-quote',
                attempt_id: 'a01', harness_id: 'codex', revision: 1, model_source: 'requested',
                model: 'probe" data-g5-boundary="escaped'}}});
        const chip = mount.querySelector('.log-executor-chip');
        const valid = !chip.hasAttribute('data-g5-boundary')
            && chip.title === mount.querySelector('.log-executor-detail').textContent;
        // This page-scoped Logs mount ends with the fixture's page/context teardown.
        return valid;
    }''')
