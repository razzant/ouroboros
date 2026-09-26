import assert from 'node:assert/strict';
import test from 'node:test';
import { createChatInstance } from '../modules/chat.js';
import { ElementStub, installDom, restoreDom, walkCard } from './chat_dom_fixture.js';
// Preserve the production test fixture, adding actual descendant lookup for
// lazily inserted controls, which the original flat fixture does not model.
const originalQuery = ElementStub.prototype.querySelector;
ElementStub.prototype.querySelector = function(selector) {
    const direct = originalQuery.call(this, selector);
    if (direct) return direct;
    for (const child of this.children) { const found = child.querySelector(selector); if (found) return found; }
    return null;
};
function makeInstance(rows, details = {}) {
    const env = installDom(async url => String(url).startsWith('/api/tasks/')
        ? (details[String(url).split('/').at(-1)] ? { ok: true, json: async () => details[String(url).split('/').at(-1)] }
            : { ok: false, status: 404, json: async () => ({error: 'missing'}) })
        : { ok: true, json: async () => String(url).startsWith('/api/chat/history')
            ? {messages: rows} : {active_direct_turns: []} });
    const handlers = new Map();
    const instance = createChatInstance({
        ws: {on(type, fn) { handlers.set(type, fn); return () => handlers.delete(type); }, isConnected: () => true, send() {}},
        state: {activePage: 'chat', projectChatIds: new Set(), unreadCount: 0}, updateUnreadBadge() {},
        stateSnapshots: {begin: () => ({generation: 1, requestedAt: Date.now()}), gate() { return Promise.resolve(this.begin()); }, isCurrent: () => true, apply() {}},
        chatId: 1, idPrefix: 'chat', mountEl: env.mount,
    });
    return {...env, handlers, instance, card: id => walkCard(globalThis.document.byId.get('chat-messages'), id)};
}
test(' current activity restores previously attested Stop after cold unconfirmed replay', async () => {
    const fx = makeInstance([{task_id: 'live', is_progress: true, text: 'Working on the report',
        cancelable: true, ts: '2026-09-09T09:00:00Z'}]);
    try {
        await fx.instance.refreshHistory({revision: 1});
        const card = fx.card('live');
        assert.equal(card.querySelector('[data-live-phase]').hidden, true);
        assert.equal(card.querySelector('[data-cancel-run]'), null);
        fx.instance.hydrateStateSnapshot({active_chat_activities: [{activity_id: 'live', chat_id: 1, kind: 'managed_task', phase: 'working'}],
            active_chat_activities_complete: true, supervisor_ready: true}, Infinity, 2);
        assert.equal(card.querySelector('[data-live-phase]').hidden, false);
        assert.ok(card.querySelector('[data-cancel-run]'), 'positive current activity restores existing control authority');
        fx.handlers.get('chat')({chat_id: 1, task_id: 'live', role: 'assistant', is_progress: true,
            cancelable: true, content: 'Continued real work'});
        assert.ok(card.querySelector('[data-cancel-run]'), 'fresh positive evidence must restore Stop');
    } finally {fx.instance.destroy(); restoreDom(fx.prior);}
});
test(' retained cancelled fact outranks untyped historical answer after result quarantine', async () => {
    const historical = {status: 'cancelled', phase: 'cancelled', ts: '2026-09-09T09:02:00Z', provenance: 'canonical_task_result_after_finalization'};
    const fx = makeInstance([
        {task_id: 'past', is_progress: true, text: 'Work before cancellation', ts: '2026-09-09T09:00:00Z', historical_terminal: historical},
        {task_id: 'past', role: 'assistant', text: 'Preserved partial answer', ts: '2026-09-09T09:02:00Z', historical_terminal: historical},
    ]);
    try {
        await fx.instance.refreshHistory({revision: 1});
        fx.instance.hydrateStateSnapshot({active_chat_activities: [], active_chat_activities_complete: false, supervisor_ready: true}, Infinity, 1);
        assert.equal(fx.card('past').dataset.finished, '0');
        fx.instance.hydrateStateSnapshot({active_chat_activities: [], active_chat_activities_complete: true, supervisor_ready: true}, Infinity, 2);
        await new Promise(resolve => setImmediate(resolve));
        assert.equal(fx.card('past').querySelector('[data-live-phase]').textContent, 'Cancelled');
    } finally {fx.instance.destroy(); restoreDom(fx.prior);}
});

test('terminal-root history projection settles a card after a later finalizing summary', async () => {
    const terminal = {status: 'completed', phase: 'done', ts: '2026-09-09T09:02:00Z',
        provenance: 'canonical_task_result_after_finalization'};
    const fx = makeInstance([
        {task_id: 'replayed', is_progress: true, text: 'Working', ts: '2026-09-09T09:00:00Z'},
        {task_id: 'replayed', summary_kind: 'terminal_root_projection', historical_terminal: terminal,
            role: 'system', system_type: 'task_summary', outcome_final: true, ts: '2026-09-09T09:02:00Z'},
        {task_id: 'replayed', summary_kind: 'authored_root_summary', role: 'system',
            system_type: 'task_summary', task_phase: 'finalizing', outcome_final: false,
            status: 'completed', ts: '2026-09-09T09:03:00Z'},
    ]);
    try {
        await fx.instance.refreshHistory({revision: 1});
        assert.equal(fx.card('replayed').dataset.finished, '1');
        assert.equal(fx.card('replayed').querySelector('[data-live-phase]').textContent, 'Done');
    } finally { fx.instance.destroy(); restoreDom(fx.prior); }
});
for (const via of ['log', 'detail']) test(`child terminal ${via} retains the producer model observation`, async () => {
    const observation = {source: 'usable_solve_response', used_model: 'fallback', requested_model: 'initial',
        used_local: false, requested_use_local: false, llm_call_id: 'call', provider: 'openrouter'};
    const executor = {task_id: 'child', task_attempt: '1', run_id: 'run', attempt_id: 'a01',
        harness_id: 'cursor', phase: 'harness.event', revision: 1, model: 'cursor-model', model_source: 'requested'};
    const fx = makeInstance([], {child: {task_id: 'child', status: 'completed', model_execution: observation}});
    try {
        await fx.instance.refreshHistory({revision: 1});
        fx.handlers.get('chat')({chat_id: 1, role: 'assistant', is_progress: true, content: 'child working',
            task_id: 'child', subagent_task_id: 'child', parent_task_id: 'root', delegation_role: 'subagent',
            subagent_role: 'reader', subagent_event: 'running', model: 'initial', executor_observation: executor});
        fx.handlers.get('log')({chat_id: 1, data: {type: 'task_done', task_id: via === 'log' ? 'child' : 'root', status: 'completed',
            ...(via === 'log' ? {model_execution: observation} : {})}});
        await new Promise(resolve => setImmediate(resolve));
        const meta = fx.card('child').querySelector('[data-live-meta]').innerHTML;
        assert.match(meta, /Last solve response: fallback/);
        assert.match(meta, /Coordinator: initial/);
        assert.match(meta, /Cursor/);
    } finally {fx.instance.destroy(); restoreDom(fx.prior);}
});

for (const fresh of [false, true]) test(`pruned model history preserves current evidence: ${fresh}`, async () => {
    const historical = {status: 'completed', phase: 'done', ts: '2026-09-09T09:02:00Z',
        provenance: 'canonical_task_result_after_finalization',
        model_execution: {source: 'usable_solve_response', used_model: 'old-fallback', used_local: false}};
    const rows = [{task_id: 'past', is_progress: true, text: 'Historical work',
        ts: '2026-09-09T09:00:00Z', historical_terminal: historical}];
    const fx = makeInstance(rows);
    try {
        if (fresh) {
            rows[0] = {...rows[0], historical_terminal: undefined,
                model_execution: {source: 'usable_solve_response', used_model: 'current-model', used_local: false}};
            await fx.instance.refreshHistory({revision: 1});
            rows[0] = {...rows[0], model_execution: undefined, historical_terminal: historical};
        }
        await fx.instance.refreshHistory({revision: 2});
        const meta = () => fx.card('past').querySelector('[data-live-meta]').innerHTML;
        assert.doesNotMatch(meta(), /Last solve response: old-fallback/);
        if (fresh) assert.match(meta(), /Last solve response: current-model/);
        fx.instance.hydrateStateSnapshot({active_chat_activities: [],
            active_chat_activities_complete: true, supervisor_ready: true}, Infinity, 2);
        await new Promise(resolve => setImmediate(resolve));
        assert.equal(fx.card('past').querySelector('[data-live-phase]').textContent, 'Done');
        assert.match(meta(), fresh ? /Last solve response: current-model/ : /Last solve response: old-fallback/);
    } finally {fx.instance.destroy(); restoreDom(fx.prior);}
});

for (const external of [false, true]) test(`terminal child retains late lineage model independently of solve evidence: ${external}`, async () => {
    const fx = makeInstance([]);
    const child = {chat_id: 1, role: 'assistant', is_progress: true, task_id: 'planner',
        subagent_task_id: 'planner', parent_task_id: 'root', delegation_role: 'subagent',
        subagent_role: 'planner', subagent_event: 'scheduled', content: 'queued', model: '',
        ...(external ? {executor_observation: {task_id: 'planner', task_attempt: '1', run_id: 'run',
            attempt_id: 'a01', harness_id: 'cursor', phase: 'harness.event', revision: 1}} : {})};
    try {
        await fx.instance.refreshHistory({revision: 1});
        fx.handlers.get('chat')(child);
        fx.handlers.get('chat')({chat_id: 1, role: 'system', system_type: 'task_summary',
            task_id: 'planner', subagent_task_id: 'planner', parent_task_id: 'root',
            delegation_role: 'subagent', subagent_role: 'planner', content: 'Done',
            task_terminal_status: 'completed',
            model_execution: {source: 'usable_solve_response', used_model: 'actual-solver', used_local: false}});
        fx.handlers.get('chat')({...child, subagent_event: 'running', content: 'planning', model: 'google/gemini-3.6-flash'});
        const card = fx.card('planner');
        const meta = card.querySelector('[data-live-meta]').innerHTML;
        assert.match(meta, external ? /Coordinator: gemini-3\.6-flash/ : /Agent model: gemini-3\.6-flash/);
        assert.match(meta, /Last solve response: actual-solver/);
        assert.equal(card.dataset.finished, '1', 'late model evidence must not revive the terminal child');
    } finally {fx.instance.destroy(); restoreDom(fx.prior);}
});
