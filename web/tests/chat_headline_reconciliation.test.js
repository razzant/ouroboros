import assert from 'node:assert/strict';
import test from 'node:test';
import { createChatInstance } from '../modules/chat.js';
import { installDom, restoreDom, walkCard } from './chat_dom_fixture.js';

const taskId = 'headline-reconciliation';
const at = second => `2026-10-25T12:00:${String(second).padStart(2, '0')}Z`;
const narration = (second, text, extra = {}) => ({
    task_id: taskId, chat_id: 2, role: 'assistant', is_progress: true, narration: true,
    text, ts: at(second), history_id: `progress:${second}`,
    history_position: { source: 'progress', offset: second }, ...extra,
});
const older = narration(10, 'Checking the first source');
const newer = narration(20, 'Checking the independent replication');
const cause = 'Only some of the plan reviewers answered; the work went on with their notes.';
const warned = { task_id: taskId, chat_id: 2, role: 'system', is_progress: false, system_type: 'task_summary',
    text: 'Done with warnings.', ts: at(40), outcome_final: true, outcome_phase: 'warn', reason_code: 'plan_review_advisory',
    outcome_axes: { execution: { status: 'degraded', reason_code: 'plan_review_advisory', plan_review: 'unanswered' } } };
const history = (messages, next = null) => ({ messages, page_cursor: 'recent', next_cursor: next,
    has_more: Boolean(next), window: { complete: !next, truncated_by: next ? ['quota'] : [] } });

function fixture(t) {
    let response = history([]), revision = 0;
    const handlers = new Map();
    const { prior, mount } = installDom(async url => ({ ok: true, json: async () =>
        String(url).startsWith('/api/chat/history') ? response : { active_direct_turns: [] } }));
    const instance = createChatInstance({ chatId: 2, mountEl: mount, asPanel: true,
        state: { activePage: 'chat', projectChatIds: new Set(), unreadCount: 0 },
        ws: { on(type, handler) { handlers.set(type, handler); return () => handlers.delete(type); },
            isConnected: () => true, send() {} },
        stateSnapshots: { begin: () => ({ generation: 1, requestedAt: Date.now() }),
            gate() { return Promise.resolve(this.begin()); }, isCurrent: () => true, apply() {} },
    });
    t.after(() => { instance.destroy(); restoreDom(prior); });
    const messages = document.byId.get('chat-messages');
    const log = data => handlers.get('log')({ chat_id: 2, data: { chat_id: 2, task_id: taskId, ...data } });
    return {
        card: () => walkCard(messages, taskId),
        title: () => walkCard(messages, taskId)?.querySelector('[data-live-title]').textContent,
        activity: () => walkCard(messages, taskId)?.querySelector('[data-live-activity]').textContent,
        emit: row => handlers.get('chat')({ ...row, content: row.text, history_id: undefined }),
        name: name => handlers.get('task_named')({ task_id: taskId, suggested_name: name }),
        log,
        tool: () => log({ type: 'tool_call_started', ts: at(30), tool: 'read_file', invocation_id: 'first-tool' }),
        terminal: status => instance.hydrateStateSnapshot({ active_chat_activities: [{
            activity_id: taskId, chat_id: 2, kind: 'managed_task', phase: 'working', status,
        }], active_chat_activities_complete: true, supervisor_ready: true }),
        async read(rows, next = null) {
            response = history(rows, next);
            const result = await instance.refreshHistory({ revision: ++revision });
            assert.equal(result.painted, true, 'wait for this history paint, never for the desired title');
        },
        async older(rows) {
            response = history(rows);
            const button = messages.querySelector('.chat-load-older-btn');
            for (const handler of button.listeners.get('click')) await handler({ target: button });
        },
    };
}

for (const order of ['tool-first', 'history-first', 'finished-first', 'failed-first', 'stopped-first']) {
    for (const reverse of [false, true]) test(`${order}: newest authored headline survives ${reverse ? 'reverse' : 'forward'} history`, async t => {
        const f = fixture(t);
        const rows = reverse ? [newer, older] : [older, newer];
        const terminal = { 'finished-first': ['completed', 'done'], 'failed-first': ['failed', 'error'],
            'stopped-first': ['cancelled', 'cancelled'] }[order];
        if (order !== 'history-first') f.tool();
        if (terminal) f.terminal(terminal[0]);
        const mounted = f.card();
        await f.read(rows);
        if (order === 'history-first') f.tool();
        const card = f.card();
        if (mounted) assert.equal(card, mounted, 'reconcile the mounted card in place');
        assert.equal(card.querySelector('[data-live-title]').textContent, newer.text);
        // The collapsed line keeps the same clock: it hides while it repeats the title and
        // would otherwise show the older narration beneath the newer one.
        assert.equal(f.activity(), '', 'no older narration under the newer title');
        assert.equal(card.dataset.ts, String(Date.parse(older.ts)), 'the earliest anchor is independent');
        if (terminal) {
            assert.equal(card.dataset.finished, '1');
            assert.equal(card.querySelector('[data-live-phase]').dataset.phase, terminal[1]);
            assert.equal(card.querySelector('[data-cancel-run]'), null);
        }
        await f.read(rows);
        assert.equal(f.card(), card);
        assert.equal(card.querySelector('[data-live-title]').textContent, newer.text, 'duplicate history is stable');
        assert.equal(f.activity(), '');
    });
}

test('an older page, a delayed live note and newer host notes cannot replace the latest narration', async t => {
    const f = fixture(t);
    await f.read([newer], 'older');
    const card = f.card();
    await f.older([older]);
    f.emit(older);
    assert.equal(f.activity(), '', 'the delayed older note takes neither the title nor the collapsed line');
    f.emit(narration(40, 'Host checkpoint', { role: 'system', narration: false }));
    await f.read([narration(50, 'Host recovery', { role: 'system', narration: false })]);
    assert.equal(f.card(), card);
    assert.equal(card.querySelector('[data-live-title]').textContent, newer.text);
    assert.equal(f.activity(), '');
    f.emit(narration(55, 'Legacy authored update', { narration: undefined }));
    assert.equal(card.querySelector('[data-live-title]').textContent, 'Legacy authored update');
    assert.equal(f.activity(), '');
});

// Replaces the retired task_presentation.test.js source pins with the behaviour they guarded.
test('nonterminal diagnostics stay rows but never title the card or take its collapsed line', t => {
    const f = fixture(t);
    const notes = () => parseInt(f.card().querySelector('[data-live-count]').textContent, 10) || 0;
    f.log({ type: 'llm_round_error', error: 'temporary provider error', ts: at(11) });
    assert.ok(f.card(), 'a diagnostic is content');
    assert.equal(f.title(), '', 'which names nothing: the chip already says Working (#1369)');
    f.emit(older);
    assert.equal(f.title(), older.text);
    const before = notes();
    for (const [second, diagnostic] of [[12, { type: 'tool_timeout', tool: 'delegate_wait' }],
        [13, { type: 'tool_call_finished', tool: 'run_command', is_error: true, invocation_id: 'failed-call' }],
        [14, { type: 'task_checkpoint', checkpoint_kind: 'context_fit_low_retry' }]]) f.log({ ...diagnostic, ts: at(second) });
    assert.ok(notes() > before, 'the diagnostics are visible rows');
    assert.equal(f.title(), older.text);
    assert.equal(f.activity(), '');
    assert.equal(f.card().dataset.finished, '0');
});

test('a terminal cause keeps the collapsed line while saved newer narration names the closed card', async t => {
    const f = fixture(t);
    f.emit(older);
    f.emit(warned);
    assert.equal(f.activity(), cause);
    await f.read([older, newer]);
    assert.equal(f.title(), newer.text);
    assert.equal(f.activity(), cause, 'saved narration is not the cause');
    assert.equal(f.card().dataset.finished, '1');
});

// Promoted from the independent review's probe: equal words are not equal ownership.
test('a terminal cause that repeats earlier narration still owns the collapsed line', async t => {
    const f = fixture(t);
    const said = narration(10, cause);
    f.name('Compare the measurements');
    f.emit(said);
    f.emit(warned);
    assert.equal(f.activity(), cause);
    await f.read([said, newer]);
    assert.equal(f.title(), 'Compare the measurements');
    assert.equal(f.card().dataset.finished, '1');
    assert.equal(f.card().querySelector('[data-live-phase]').dataset.phase, 'warn');
    assert.equal(f.activity(), cause, 'saved narration older than the outcome cannot replace its cause');
});

test('equal-time history uses physical source order, including an older page arriving last', async t => {
    const f = fixture(t);
    const second = { ...newer, ts: older.ts };
    await f.read([second, older], 'older');
    assert.equal(f.card().querySelector('[data-live-title]').textContent, second.text);
    await f.older([older]);
    assert.equal(f.card().querySelector('[data-live-title]').textContent, second.text);
});

for (const when of ['before', 'after']) test(`an explicit task name ${when} narration keeps precedence`, async t => {
    const f = fixture(t);
    if (when === 'before') f.name('Compare the measurements');
    f.tool();
    await f.read([older, newer]);
    if (when === 'after') f.name('Compare the measurements');
    f.terminal('completed');
    await f.read([newer, older]);
    assert.equal(f.card().querySelector('[data-live-title]').textContent, 'Compare the measurements');
    assert.equal(f.activity(), newer.text, 'a named card shows the newest narration on its collapsed line');
});

// Promoted from the independent review's probe: the name came before the lineage.
test('late typed lineage on a named, already closed shell restores child identity', t => {
    const f = fixture(t);
    f.emit(narration(5, 'Delegating the verification', { task_id: 'parent' }));
    f.emit(narration(10, 'Reading the input'));
    f.name('Root-shaped name');
    f.terminal('cancelled');
    f.emit(narration(20, 'Late child progress', { delegation_role: 'subagent', subagent_task_id: taskId,
        parent_task_id: 'parent', subagent_role: 'Verifier', model: 'model-x' }));
    const card = f.card();
    assert.equal(card.dataset.subagent, '1');
    assert.equal(card.dataset.parentTaskId, 'parent');
    assert.equal(card.dataset.finished, '1');
    assert.equal(f.title(), 'Verifier', 'a child is titled by its role, not the name it had as a root-shaped shell');
});
