"""Actual chat renderers over deterministic progress/history, without model calls."""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.test_delegated_activity_browser import engine_ui as engine_ui  # chromium and webkit
from tests.test_subscription_setup_browser import PREVIEW_SCOPE, capture
from tests.test_subscription_setup_browser import subscription_ui as subscription_ui

pytestmark = [pytest.mark.ui_browser]


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
    history = {'messages': rows, 'progress': [], 'profile_scope': PREVIEW_SCOPE, 'page_cursor': 'executor-recent',
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
        {'messages': rows, 'progress': [], 'profile_scope': PREVIEW_SCOPE, 'has_more': False, 'window': {'complete': True}})))
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
        const {executorChip} = await import('/static/modules/executor_card_facts.js');
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
    # The phase chip stays the work indicator: the quiet name is not a running proof. A nested card says its own
    # member's state and a saved row proves no current one (#3929, DESIGN "Room work status"), so this replayed running
    # helper, which no server read lists, is unconfirmed, whatever its executor chip names.
    assert cards['m-assigned'].locator(':scope > .chat-live-summary-button [data-live-phase]').inner_text() == \
        'Activity unconfirmed'
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
        # The quiet chip stands among the card's facts and its detail sentence is text beside them: the card is its one
        # fold, and its head (the summary) opens it.
        card.locator(':scope > summary').click()
        chip = card.locator(':scope > .log-facts > .log-executor-chip')
        assert chip.inner_text().strip() == label, task
        detail = card.locator(':scope > .log-executor-detail')
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


# The owner asked to see which model a subagent was spawned on (2026-11-22): from Queued on, its card names the
# executor and model the delegation selected, observed evidence replaces it, and the request stays in Details.
SEL_ROOT, SEL_KID, SEL_API, SEL_MISSED, SEL_CANCELLED = 'sel-root', 'sel-kid', 'sel-api', 'sel-missed', 'sel-cancelled'
SEL_LOCAL = 'sel-local'
SEL_LONG_ROLE = 'Cross-surface consistency reviewer for the long-named settings and history migration'
SEL_LONG_MODEL = 'claude-opus-5-5-20261101-extended-thinking-preview'
SEL_CHILDREN = {  # task id: (role, row route, the coordinator's model, observed models, delivered live)
    SEL_KID: ('Implementation reviewer', {'kind': 'agent_session', 'target_id': 'codex=gpt-5.5'},
              'openai/gpt-6-astra', ['gpt-5.5-codex'], True),
    SEL_API: ('API researcher', {'kind': 'api_model', 'target_id': 'openai::gpt-5.5-pro'}, 'openai::gpt-5.5-pro', [], True),
    # Its coordinator runs on a subscription route: that recorded billing must survive the missed-terminal recovery.
    SEL_MISSED: (SEL_LONG_ROLE, {'kind': 'agent_session', 'target_id': f'claude={SEL_LONG_MODEL}'},
                 'claudexor::claude=claude-sonnet-5-5', ['claude-opus-5-5'], False),
    SEL_CANCELLED: ('Queued-only checker', {'kind': 'agent_session', 'target_id': 'codex=gpt-5.5-mini'}, '', [], True),
    # A local API row: its target is spelled `… (local)` (configured_subagents), and its actor runs the bare model locally.
    SEL_LOCAL: ('Local summarizer', {'kind': 'api_model', 'target_id': 'qwen.gguf (local)'}, 'qwen.gguf', [], True),
}
SEL_UNDISPATCHED = {SEL_CANCELLED}  # cancelled while queued: never dispatched, so its terminal names no route
SEL_PARENT_MODEL = 'openai/gpt-6-astra'  # the parent's cognitive route, the snapshot an undispatched child keeps
SEL_META = '''ids => Object.fromEntries(ids.map(id => {
    const card = document.querySelector(`.chat-live-card[data-task-id="${id}"]`);
    const meta = card?.querySelector(':scope > .chat-live-foot [data-live-meta]');
    const facts = [...(card?.querySelectorAll(':scope > .chat-live-details > [data-live-facts] > div') || [])]
        .map(row => [row.querySelector('dt').textContent, row.querySelector('dd').textContent]);
    const chip = card?.querySelector(':scope > .chat-live-foot .chat-live-executor-chip');
    const box = meta?.getBoundingClientRect(), cardBox = card?.getBoundingClientRect();
    return [id, {line: meta ? meta.innerText.replace(/\\s+/g, ' ').trim() : null, chip: chip ? chip.innerText.trim() : null,
        facts: Object.fromEntries(facts), phase: card?.querySelector('[data-live-phase]')?.innerText || null,
        finished: card?.dataset.finished === '1', fits: meta ? meta.scrollWidth <= meta.clientWidth + 1 : null,
        inside: box && cardBox ? box.right <= cardBox.right + 1 && box.left >= cardBox.left - 1 : null}];
}))'''
SEL_EXPAND = '''ids => { for (const id of ids) {
    const card = document.querySelector(`.chat-live-card[data-task-id="${id}"]`);
    if (card && card.dataset.expanded !== '1') card.querySelector('[data-live-disclosure]').click();
} }'''


class SelectionProducers:
    """The real producers of a child's frames, in process: the admission notice, the worker's running notice through
    supervisor delivery, a progress report and the supervisor's terminal dispatch, all through ``message_bus``."""

    def __init__(self, data, monkeypatch):
        from ouroboros.task_results import write_task_result
        from supervisor import events, events_chat_delivery, message_bus

        self.data, self.live, self.message_bus, self.monkeypatch = data, [], message_bus, monkeypatch
        bridge = message_bus.LocalChatBridge()
        bridge._broadcast_fn = self.live.append
        monkeypatch.setattr(message_bus, 'DATA_DIR', data)
        monkeypatch.setattr(message_bus, 'load_state', lambda: {'owner_id': 1})
        monkeypatch.setattr(message_bus, '_BRIDGE', bridge)
        monkeypatch.setattr(message_bus, 'publish_event', lambda *_: None)
        monkeypatch.setattr(events_chat_delivery, '_bound_project_chat_id', lambda *_: None)
        monkeypatch.setattr(events, '_bound_project_chat_id', lambda *_: None)
        self.tasks = {tid: {'id': tid, 'type': 'task', 'chat_id': 1, '_attempt': 0, 'delegation_role': 'subagent',
                            'root_task_id': SEL_ROOT, 'parent_task_id': SEL_ROOT, 'role': role, 'subagent_role': role,
                            'configured_subagent': {'schema': 1, 'selected_subagent_id': tid, 'config_fingerprint': 'fp',
                                                    'route': {**route, 'credential_profile_id': 'work-account'}},
                            # The cognitive snapshot admission froze (a session child runs on it as its coordinator).
                            'parent_cognitive_route': {'model': model or SEL_PARENT_MODEL, 'use_local_model': False,
                                                       'chosen_by': 'inherited'}}
                      for tid, (role, route, model, *_rest) in SEL_CHILDREN.items()}
        self.delivery = SimpleNamespace(DRIVE_ROOT=data, RUNNING={}, send_with_budget=message_bus.send_with_budget,
                                        append_jsonl=lambda *_: pytest.fail('delivery raised'))
        write_task_result(data, SEL_ROOT, 'running', chat_id=1)

    def _taken(self, before):
        return self.live[before:]

    def queued(self):
        from ouroboros.task_results import write_task_result
        from supervisor.events_subagent_admission import _subagent_scheduled_meta

        before = len(self.live)
        for tid, task in self.tasks.items():  # the receipt and fields events_schedule_task writes and sends
            write_task_result(self.data, tid, 'scheduled', chat_id=1, **{key: task[key] for key in (
                'delegation_role', 'root_task_id', 'parent_task_id', 'role', 'configured_subagent', 'parent_cognitive_route')})
            meta = {'root_task_id': SEL_ROOT, 'parent_task_id': SEL_ROOT, 'delegation_role': 'subagent', 'cancelable': False,
                    **_subagent_scheduled_meta(tid=tid, role=task['role'], task_constraint={}, task_group_id='',
                                               requested_model_lane='auto', active_subagent_count=1, max_active_subagents=None,
                                               configured_subagent=task['configured_subagent']),
                    'parent_cognitive_route': task['parent_cognitive_route']}
            self.message_bus.send_with_budget(1, f"🗓️ Scheduled subagent {tid} ({task['role']}): check the card", is_progress=True,
                                              task_id=tid, progress_meta=meta, role='system', system_type='task_scheduled')
        return self._taken(before)

    def running(self):
        from ouroboros.subagent_messages import subagent_message_meta
        from supervisor import events_chat_delivery

        before = len(self.live)
        for tid, task in self.tasks.items():  # dispatch stamps the route and the coordinating model, as the record does
            if tid in SEL_UNDISPATCHED:
                continue
            role, route, model, *_ = SEL_CHILDREN[tid]
            task.update(executor_route=route['target_id'], model=model, use_local_model=route['target_id'].endswith(' (local)'))
            self.delivery.RUNNING[tid] = {'task': task}
            events_chat_delivery._handle_send_message({
                'type': 'send_message', 'chat_id': 1, 'role': 'system', 'system_type': 'subagent_started',
                'text': f'Subagent running ({role}).', 'is_progress': True, 'task_id': tid,
                'progress_meta': subagent_message_meta(task, task_id=tid, event='running')}, self.delivery)
        # The run's own report names its harness but no model: the selection stands for that harness.
        events_chat_delivery._handle_send_message({
            'type': 'send_message', 'chat_id': 1, 'text': 'Reading the card renderer.', 'is_progress': True,
            'task_id': SEL_KID, 'progress_meta': {'narration': True, 'executor_observation': {
                'task_id': SEL_KID, 'task_attempt': '0', 'run_id': 'run-sel', 'attempt_id': 'a01', 'harness_id': 'codex',
                'phase': 'harness.event', 'revision': 1}}}, self.delivery)
        return self._taken(before)

    def finish(self, tid, status='completed', **terminal_facts):
        from ouroboros.headless import prepare_terminal_task_files
        from ouroboros.task_results import write_task_result

        task = self.tasks.get(tid) or {'id': tid, 'type': 'task', 'chat_id': 1, '_attempt': 0}
        extra = {}
        if tid in SEL_CHILDREN:
            observed = SEL_CHILDREN[tid][3]
            extra = {key: task[key] for key in ('delegation_role', 'root_task_id', 'parent_task_id', 'configured_subagent',
                                                'executor_route', 'model') if key in task}
            if observed:
                extra['subagent_envelope'] = {'actual_substrate': 'harness_used', 'execution_evidence': {
                    'delegated_runs_started': 1, 'delegated_runs_settled': 1, 'delegated_runs_succeeded': 1,
                    'delegated_runs_failed': 0, 'harness_models': observed}}
            if task.get('use_local_model'):  # the worker's own solve evidence: the bare model, on the local lane
                extra['model_execution'] = {'source': 'usable_solve_response', 'requested_model': task['model'],
                                            'requested_use_local': True, 'used_model': task['model'], 'used_local': True,
                                            'provider': 'local', 'llm_call_id': 'call-local'}
        write_task_result(self.data, tid, status, result=f'{tid} finished its check.', chat_id=1, **{**extra, **terminal_facts})
        assert not prepare_terminal_task_files(self.data, task)['error']
        return self._dispatch({'type': 'task_done', 'task_id': tid, 'task_type': 'task', 'worker_id': 0, 'chat_id': 1,
                               'ts': '2026-11-22T18:40:00Z', '_files_prepared_attempt': 1}, {tid: {'task': task}})

    def _dispatch(self, done_event, running):
        from supervisor.events import _handle_task_done

        before = len(self.live)
        ctx = SimpleNamespace(DRIVE_ROOT=self.data, RUNNING=running, WORKERS={}, PENDING=[], bridge=self.message_bus._BRIDGE,
                              persist_queue_snapshot=lambda **_: None, send_with_budget=self.message_bus.send_with_budget,
                              load_state=lambda: {}, save_state=lambda _st: None, sort_pending=lambda: None,
                              append_jsonl=lambda path, row: __import__('ouroboros.utils').utils.append_jsonl(path, row))
        _handle_task_done(done_event, ctx)
        return self._taken(before)

    def cancel_queued(self, tid):
        """The real pending cancellation (cancel_task_by_id -> _finish_captured_pending -> _emit_cancel_task_done): the
        child leaves PENDING without ever entering RUNNING, so its task_done has no prepared-files marker and the
        supervisor's dispatch finds no RUNNING row for it."""
        from ouroboros import cancel_intents
        from supervisor import queue, task_lifecycle, workers

        emitted, patch = [], self.monkeypatch
        for name, value in (('DRIVE_ROOT', self.data), ('PENDING', [self.tasks[tid]]), ('RUNNING', {}),
                            ('ADMISSION_RESERVATIONS', {})):
            patch.setattr(queue, name, value)
        patch.setattr(queue, 'persist_queue_snapshot', lambda reason='': None)
        patch.setattr(workers, 'WORKERS', {}, raising=False)
        patch.setattr(workers, 'get_event_q', lambda *_a, **_k: SimpleNamespace(put=emitted.append))
        patch.setattr(task_lifecycle, 'CANCELLED_ROOT_FENCES', {}, raising=False)
        patch.setattr(task_lifecycle, '_ACTIVE_CASCADE_FENCES', {}, raising=False)
        cancel_intents.request_cancel(self.data, tid, source='http_single')
        assert task_lifecycle.cancel_task_by_id(tid) and queue.PENDING == [] and queue.RUNNING == {}
        [done] = [event for event in emitted if event.get('type') == 'task_done']
        assert '_files_prepared_attempt' not in done, done
        return self._dispatch(done, {})


@pytest.mark.parametrize('theme', ['light', 'dark'])
def test_selected_model_from_queued_through_done_reload_and_missed_terminal_detail(engine_ui, tmp_path, monkeypatch, theme):
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.testclient import TestClient

    from ouroboros.gateway.history import make_chat_history_endpoint
    from tests.test_task_file_serving import _client

    ui = engine_ui
    page = ui['page']
    engine = page.context.browser.browser_type.name
    data = tmp_path / 'data'
    (data / 'logs').mkdir(parents=True)
    producers = SelectionProducers(data, monkeypatch)
    app = Starlette(routes=[Route('/api/chat/history', endpoint=make_chat_history_endpoint(data))])
    app.state.served_ui = SimpleNamespace(serving_profile_scope=lambda: PREVIEW_SCOPE)  # the proof the page's scope meta names
    history = TestClient(app)
    ui['backend']['task_gateway'] = _client(data)
    sockets, details = [], []

    def history_route(route):
        url = route.request.url
        response = history.get(url[url.index('/api/chat/history'):])
        route.fulfill(status=response.status_code, content_type='application/json', body=response.text)

    def task_route(route):
        details.append(route.request.url)
        route.continue_()  # the served origin answers from the real task route (``task_gateway``)

    page.route('**/api/state', lambda route: route.fulfill(content_type='application/json', body=json.dumps(
        {'supervisor_ready': True, 'active_chat_activities': [], 'projects': [], 'sha': 'sel-candidate'})))
    page.route('**/api/chat/history*', history_route)
    page.route('**/api/tasks/**', task_route)
    page.route_web_socket('**/ws', lambda ws: (sockets.append(ws),  # the serving socket's proof precedes every other frame
                                               ws.send(json.dumps({'type': 'connection_profile', 'profile_scope': PREVIEW_SCOPE})),
                                               ws.send(json.dumps({'type': 'heartbeat'}))))
    page.set_viewport_size({'width': 1728, 'height': 1084})
    page.emulate_media(color_scheme=theme)
    page.goto(ui['url'])
    page.wait_for_selector('#chat-input')
    assert page.locator('html').get_attribute('data-theme') == theme
    frames = []

    def push(payloads):
        for payload in payloads:
            frames.append(payload)
            sockets[-1].send(json.dumps(payload))

    ids = [SEL_KID, SEL_API, SEL_MISSED, SEL_CANCELLED, SEL_LOCAL]
    measured = {}

    def measure(stage):
        page.evaluate(SEL_EXPAND, [SEL_ROOT, *ids])
        page.wait_for_timeout(200)
        measured[stage] = page.evaluate(SEL_META, ids)
        capture(page, f'selected-model-{engine}-{theme}-{stage}')
        if evidence := os.environ.get('OUROBOROS_UI_EVIDENCE_DIR'):  # each card whole, so none is judged off-screen
            top = page.evaluate("document.getElementById('chat-messages').scrollTop")
            for tid in ids:  # centred clear of the fixed header and composer, then the reader's place is restored
                card = page.locator(f'.chat-live-card[data-task-id="{tid}"]')
                card.evaluate("el => el.scrollIntoView({block: 'center'})")
                card.screenshot(path=str(Path(evidence) / f'selected-model-{engine}-{theme}-{stage}-{tid}.png'),
                                animations='disabled')
            page.evaluate("top => { document.getElementById('chat-messages').scrollTop = top; }", top)
        return measured[stage]

    producers.message_bus.send_with_budget(1, 'Coordinating three helpers.', is_progress=True, task_id=SEL_ROOT,
                                           narration=True)
    push(producers.live)
    push(producers.queued())
    page.locator(f'.chat-live-card[data-task-id="{SEL_MISSED}"]').wait_for()
    queued = measure('1-queued')
    # Queued: each card names its selected executor and model as a request; an API row has no harness chip.
    assert queued[SEL_KID]['chip'] == 'Codex' and 'requested gpt-5.5' in queued[SEL_KID]['line'], queued
    assert queued[SEL_API]['chip'] is None and 'requested gpt-5.5-pro' in queued[SEL_API]['line'], queued
    assert queued[SEL_MISSED]['chip'] == 'Claude Code' and f'requested {SEL_LONG_MODEL}' in queued[SEL_MISSED]['line']
    assert 'work-account' not in json.dumps(frames), 'the row account pin never travels with the selection'
    # A local API row is marked local once, never `(local) (local)`.
    assert queued[SEL_LOCAL]['chip'] is None and 'requested qwen.gguf (local)' in queued[SEL_LOCAL]['line'], queued
    assert '(local) (local)' not in json.dumps(queued[SEL_LOCAL]), queued[SEL_LOCAL]
    # The coordinator is the scheduling snapshot admission sent: Details only, its unstated billing read as unknown.
    for tid in (SEL_KID, SEL_CANCELLED):
        assert queued[tid]['facts']['Coordinator'] == 'gpt-6-astra (inherited) · bills as unknown', queued[tid]
    page.set_viewport_size({'width': 390, 'height': 844})  # the long requested model on a phone's compact line
    narrow_queued = measure('1b-queued-narrow')
    assert all(narrow_queued[tid]['inside'] for tid in ids), narrow_queued
    page.set_viewport_size({'width': 1728, 'height': 1084})
    push(producers.running())
    page.wait_for_function(f"document.querySelector('.chat-live-card[data-task-id=\"{SEL_KID}\"] .chat-live-executor-chip')"
                           ".textContent.includes('last update')")
    running = measure('2-running')
    # Running: a model-less report of the selected harness keeps the request; the coordinator stays in Details.
    assert running[SEL_KID]['chip'] == 'Codex · last update' and 'requested gpt-5.5' in running[SEL_KID]['line']
    assert 'gpt-6-astra' not in running[SEL_KID]['line'] and running[SEL_KID]['facts']['Coordinator'].startswith('gpt-6-astra')
    # The progress report's chip carries no coordinator facts: the recorded route, provenance and billing stay.
    coordinators = {SEL_KID: 'gpt-6-astra (inherited) · bills as unknown',
                    SEL_MISSED: 'claude=claude-sonnet-5-5 (inherited) · bills as subscription'}
    for tid, coordinator in coordinators.items():
        assert running[tid]['facts'].get('Coordinator') == coordinator, (tid, running[tid])
    assert 'model gpt-5.5 (requested)' in running[SEL_KID]['facts']['Executor'], running[SEL_KID]
    assert running[SEL_API]['chip'] is None and 'gpt-5.5-pro' in running[SEL_API]['line'], running[SEL_API]
    push(producers.finish(SEL_KID, model='openai/gpt-6-sol', coordinator_route='openai/gpt-6-sol',
                          coordinator_billing='measured', coordinator_provenance='parent_fallback'))
    push(producers.finish(SEL_API))
    push(producers.finish(SEL_LOCAL))
    cancelled = producers.cancel_queued(SEL_CANCELLED)  # the real pending cancellation: no RUNNING row, no prepared files
    assert [frame['data'].get('requested_executor_route') for frame in cancelled if frame.get('type') == 'log'
            and frame['data'].get('type') == 'task_done'] == [SEL_CHILDREN[SEL_CANCELLED][1]], cancelled
    push(cancelled)
    producers.finish(SEL_MISSED)  # its terminal frames never reach this page
    page.wait_for_function(f"document.querySelector('.chat-live-card[data-task-id=\"{SEL_KID}\"]').dataset.finished === '1'")
    done = measure('3-done')
    # Done: the observed model takes the line and the request moves to Details; the missed child is still open.
    assert 'gpt-5.5-codex' in done[SEL_KID]['line'] and 'requested' not in done[SEL_KID]['line'], done[SEL_KID]
    assert done[SEL_KID]['facts'].get('Requested') == 'Codex · gpt-5.5', done[SEL_KID]
    assert not done[SEL_MISSED]['finished'] and f'requested {SEL_LONG_MODEL}' in done[SEL_MISSED]['line']
    # Cancelled before dispatch: it lost no run, so it keeps its selection and request with no missing-receipt warning.
    undispatched = 'Selected Codex, model gpt-5.5-mini (requested); no dispatch was recorded before it ended'
    assert done[SEL_CANCELLED]['finished'] and done[SEL_CANCELLED]['chip'] == 'Codex', done[SEL_CANCELLED]
    assert 'requested gpt-5.5-mini' in done[SEL_CANCELLED]['line'], done[SEL_CANCELLED]
    assert done[SEL_CANCELLED]['facts']['Executor'] == undispatched, done[SEL_CANCELLED]
    assert done[SEL_CANCELLED]['phase'] == 'Stopped', done[SEL_CANCELLED]
    # The ended child's own coordinator changed after start. The queued cancellation retains its recorded snapshot.
    assert done[SEL_KID]['facts'].get('Coordinator') == "gpt-6-sol (parent's fallback) · bills as measured", done[SEL_KID]
    assert done[SEL_CANCELLED]['facts'].get('Coordinator') == coordinators[SEL_KID], done[SEL_CANCELLED]
    # The local child ran the model it asked for: its solve response names the bare model, so no false request.
    assert done[SEL_LOCAL]['finished'] and 'qwen.gguf' in done[SEL_LOCAL]['line'], done[SEL_LOCAL]
    assert 'requested' not in done[SEL_LOCAL]['line'] and 'Requested' not in done[SEL_LOCAL]['facts'], done[SEL_LOCAL]
    # The root ends: the page reads the missed child's stored detail (the real task route) and settles it from there.
    push(producers.finish(SEL_ROOT))
    page.wait_for_function(f"document.querySelector('.chat-live-card[data-task-id=\"{SEL_MISSED}\"]').dataset.finished === '1'")
    recovered = measure('4-missed-terminal-recovered')
    assert any(SEL_MISSED in url for url in details), details
    assert 'claude-opus-5-5' in recovered[SEL_MISSED]['line'] and 'requested' not in recovered[SEL_MISSED]['line']
    assert recovered[SEL_MISSED]['chip'] == 'Claude Code', recovered[SEL_MISSED]
    assert recovered[SEL_MISSED]['facts'].get('Requested') == f'Claude Code · {SEL_LONG_MODEL}', recovered[SEL_MISSED]
    assert recovered[SEL_MISSED]['facts'].get('Coordinator') == coordinators[SEL_MISSED], recovered[SEL_MISSED]
    page.reload()
    # A finished root opens collapsed, its finished children folded inside it (DESIGN §5): measure() opens it.
    page.locator(f'.chat-live-card[data-task-id="{SEL_MISSED}"]').wait_for(state='attached')
    reloaded = measure('5-reloaded')
    for tid in ids:  # history replays what live showed
        assert reloaded[tid]['line'].split(' · ')[:2] == recovered[tid]['line'].split(' · ')[:2], (tid, reloaded, recovered)
        assert reloaded[tid]['facts'].get('Requested') == recovered[tid]['facts'].get('Requested'), tid
        assert reloaded[tid]['facts'].get('Coordinator') == recovered[tid]['facts'].get('Coordinator'), (tid, reloaded[tid])
    assert reloaded[SEL_CANCELLED]['chip'] == 'Codex' and reloaded[SEL_CANCELLED]['facts']['Executor'] == undispatched
    page.set_viewport_size({'width': 390, 'height': 844})
    narrow = measure('6-reloaded-narrow')
    assert all(narrow[tid]['inside'] for tid in ids), narrow
    root = os.environ.get('OUROBOROS_UI_EVIDENCE_DIR')
    if root:
        Path(root).mkdir(parents=True, exist_ok=True)
        (Path(root) / f'selected-model-{engine}-{theme}.json').write_text(json.dumps(
            {'measured': measured, 'frames': frames, 'detail_reads': details}, ensure_ascii=False, indent=2), encoding='utf-8')
