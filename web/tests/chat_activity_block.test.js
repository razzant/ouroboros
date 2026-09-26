// The conversation activity block (docs/DESIGN.md "Conversation activity
// block"): one predicate over the record's facts decides whether a turn's block
// is in the transcript — always-shown kinds, open attention (a model wait, a
// pending stop, a host-attested Stop), children, reviews, content rows, a
// terminal outcome other than Done. A receipt row (a host-stamped addressing
// call, or the replay summary of a turn that ran only such calls) is not
// content. Presence is the same live, on reload and on reconnect; a wait-only
// block leaves with its wait and its resolved episode cannot reopen it; the
// header keeps the census verdict beside a block.
import assert from 'node:assert/strict';
import test, { after } from 'node:test';
import { createChatInstance } from '../modules/chat.js';
import {
    clearStickyCardState, noteToolCall, noteToolHostMetrics, toolEvidenceView,
} from '../modules/chat_activity.js';
import { upsertToolFoldRow } from '../modules/chat_render_batch.js';
import { summarizeChatLiveEvent } from '../modules/log_events.js';
import { ElementStub, installDom, restoreDom, walkCard } from './chat_dom_fixture.js';

// The folded row a sequence of observations produces, with no DOM in the way.
const fold = (...observations) => {
    const record = {};
    for (const observation of observations) noteToolCall(record, observation);
    return toolEvidenceView(record.toolFold);
};
const frame = (type, task, row) => summarizeChatLiveEvent({ type, task_id: task, ...row });

// The flat fixture's querySelector does not descend; the status badge and
// card internals need a real descendant lookup.
const originalQuery = ElementStub.prototype.querySelector;
after(() => { ElementStub.prototype.querySelector = originalQuery; });
ElementStub.prototype.querySelector = function (selector) {
    const direct = originalQuery.call(this, selector);
    if (direct) return direct;
    for (const child of this.children) { const found = child.querySelector(selector); if (found) return found; }
    return null;
};

// The flat fixture parses markup into bare children; give each parsed child
// the markup from its own tag onward so a rendered row's text is assertable.
const innerHTMLDescriptor = Object.getOwnPropertyDescriptor(ElementStub.prototype, 'innerHTML');
Object.defineProperty(ElementStub.prototype, 'innerHTML', {
    configurable: true,
    get() { return innerHTMLDescriptor.get.call(this); },
    set(value) {
        innerHTMLDescriptor.set.call(this, value);
        const html = String(value || '');
        let cursor = 0;
        for (const child of this.children) {
            const index = html.indexOf(`<${child.tagName.toLowerCase()}`, cursor);
            if (index < 0) continue;
            child._innerHTML = html.slice(index);
            cursor = index + 1;
        }
    },
});

const TS = '2026-09-15T12:00:00Z';
const TASK = 'turn-a';

function fixture(history = []) {
    const env = installDom(async (url) => ({ ok: true, json: async () =>
        String(url).startsWith('/api/chat/history')
            ? { messages: history, window: { complete: true } }
            : { active_direct_turns: [] } }));
    const handlers = new Map();
    let generation = 0;
    const instance = createChatInstance({
        ws: { on(type, fn) { handlers.set(type, fn); return () => handlers.delete(type); },
            isConnected: () => true, send() {} },
        state: { activePage: 'chat', projectChatIds: new Set(), unreadCount: 0 },
        updateUnreadBadge() {}, chatId: 1, idPrefix: 'chat', mountEl: env.mount,
        stateSnapshots: { begin: () => ({ generation: ++generation, requestedAt: Date.now() }), gate() { return Promise.resolve(this.begin()); },
            isCurrent: () => true, apply() {} },
    });
    const messages = document.byId.get('chat-messages');
    const nodes = (node) => [node, ...(node?.children || []).flatMap(nodes)];
    return {
        instance, messages,
        card: (id = TASK) => walkCard(messages, id),
        rows: (id = TASK) => nodes(walkCard(messages, id)).filter((n) => n.classList?.contains('chat-live-line')),
        meta: (id = TASK) => walkCard(messages, id)?.querySelector('[data-live-meta]')?.innerHTML || '',
        status: () => env.mount.querySelector('.status-badge')?.textContent,
        typingHidden: () => messages.children
            .find((node) => String(node.className || '').includes('typing-bubble'))?.style.display === 'none',
        emit: (type, row) => handlers.get(type)({ chat_id: 1, ts: TS, ...row }),
        log: (row) => handlers.get('log')({ chat_id: 1, data: { task_id: TASK, ts: TS, ...row } }),
        census: (rows) => instance.hydrateStateSnapshot({
            active_chat_activities: rows, active_chat_activities_complete: true, supervisor_ready: true,
        }, Infinity, ++generation),
        close() { instance.destroy(); restoreDom(env.prior); },
    };
}

const direct = (id = TASK) => [{ activity_id: id, chat_id: 1, kind: 'direct_chat', phase: 'thinking' }];
const managed = (id = TASK) => [{ activity_id: id, chat_id: 1, kind: 'managed_task', phase: 'working' }];
const final = { task_id: TASK, role: 'assistant', content: 'Here is the answer.', text: 'Here is the answer.',
    task_terminal_status: 'completed', outcome_axes: { execution: { status: 'ok' } },
    accounted_upper_bound_usd: 0.12, cost_final: true, cost_accounting_status: 'available' };
const wait = (patch = {}) => ({ role: 'system', system_type: 'task_model_wait', task_id: TASK,
    wait_id: 'wait-main', revision: 1, task_attempt: 1, role_name: 'main', model: 'openai::gpt-5.5',
    reason: 'quota', state: 'waiting', auto_continue: true, ...patch });

test('a wait with zero tools opens a block with the controls and leaves with the wait; a stale revision cannot resurrect it', () => {
    const f = fixture();
    try {
        f.census(direct());
        f.emit('chat', wait({ role: 'system' }));
        const card = f.card();
        assert.ok(card, 'the open wait is the block');
        assert.ok(card.querySelector('.model-waits'), 'the wait controls live inside it');
        assert.equal(card.dataset.modelWaiting, '1');
        assert.equal(f.status(), 'Waiting for access');
        f.emit('chat', wait({ role: 'system', revision: 2, state: 'resolved', resolution: 'quota_restored' }));
        assert.equal(f.card(), null, 'no block remains once the wait resolved');
        f.emit('chat', wait({ role: 'system', revision: 1 }));
        assert.equal(f.card(), null, 'the resolved episode never reopens');
        f.emit('chat', final);
        f.log({ ...final, type: 'task_done', status: 'completed' });
        assert.equal(f.card(), null, 'a done zero-tool turn keeps no block');
    } finally { f.close(); }
});

test('a tool frame mints the task card before any census lists the turn: a tool row is content', () => {
    const f = fixture();
    try {
        f.log({ type: 'tool_call_started', tool: 'read_file', tool_call_id: 'c1', args: { path: 'README.md' }, _is_direct_chat: true });
        assert.ok(f.card(), 'the first tool call mints the block');
        assert.ok(f.card().querySelector('[data-turn-into-project]'), 'conversion is offered in Main from the first content row');
        assert.equal(f.card().querySelector('[data-live-title]').textContent, 'Working...', 'the placeholder title of a running task card');
        f.log({ type: 'tool_call_finished', tool: 'read_file', tool_call_id: 'c1', args: { path: 'README.md' }, duration_sec: 0.3, _is_direct_chat: true });
        f.census(direct());
        assert.equal(f.card().querySelector('[data-live-title]').textContent, 'Working...', 'the census lane fact does not change chrome');
        assert.ok(f.card().querySelector('[data-turn-into-project]'));
    } finally { f.close(); }
});

test('a tool-only direct turn offers Stop from its stamped tool frame, without a narration row', () => {
    const f = fixture();
    try {
        f.log({ type: 'llm_round_started', model: 'm', round: 1, _is_direct_chat: true });
        assert.equal(f.card(), null, 'a round frame is not content and carries no marker');
        f.log({ type: 'tool_call_started', tool: 'read_file', tool_call_id: 'c1', args: { path: 'README.md' }, _is_direct_chat: true, cancelable: true });
        assert.ok(f.card(), 'the tool row mints the block');
        assert.ok(f.card().querySelector('[data-cancel-run]'), 'the host marker on the work frame offers Stop');
        assert.ok(f.card().querySelector('[data-turn-into-project]'), 'a tool row is work: conversion is offered');
        f.census(direct());
        assert.ok(f.card().querySelector('[data-cancel-run]'));
        f.log({ ...final, type: 'task_done', status: 'completed', _is_direct_chat: true });
        assert.equal(f.card()?.querySelector('[data-cancel-run]') ?? null, null, 'no Stop on a finished turn');
    } finally { f.close(); }
});

test('a direct turn with two successful tools is the task card: ONE evidence row live and the same row after a reload', async () => {
    const f = fixture();
    try {
        f.census(direct());
        f.log({ type: 'tool_call_started', tool: 'read_file', tool_call_id: 'c1', args: { path: 'README.md' } });
        f.log({ type: 'tool_call_finished', tool: 'read_file', tool_call_id: 'c1', args: { path: 'README.md' }, duration_sec: 0.3 });
        f.log({ type: 'tool_call_started', tool: 'web_search', tool_call_id: 'c2', args: { query: 'ouroboros' } });
        f.log({ type: 'tool_call_finished', tool: 'web_search', tool_call_id: 'c2', args: { query: 'ouroboros' }, duration_sec: 1.2 });
        assert.ok(f.card(), 'the first tool call mints the block live');
        assert.ok(f.card().querySelector('[data-turn-into-project]'), 'a working direct turn is offered conversion in Main');
        assert.equal(f.card().querySelector('[data-live-title]').textContent, 'Working...', 'the running placeholder title');
        assert.equal(f.rows().length, 1, 'four frames about two calls are the block\'s one evidence row');
        // The stub cannot repaint an in-place patch; the producer states the folded row.
        const c1 = { tool: 'read_file', tool_call_id: 'c1', args: { path: 'README.md' } };
        const c2 = { tool: 'web_search', tool_call_id: 'c2', args: { query: 'ouroboros' } };
        const started = frame('tool_call_started', TASK, c1);
        const finished = frame('tool_call_finished', TASK, { ...c1, duration_sec: 0.3 });
        assert.equal(started.dedupeKey, finished.dedupeKey);
        assert.deepEqual([started.phase, started.visible, finished.phase, finished.dedupeKey],
            ['calling', true, 'ok', `tools|${TASK}`]);
        const row = fold(started.toolCall, finished.toolCall,
            frame('tool_call_finished', TASK, { ...c2, duration_sec: 1.2 }).toolCall);
        assert.deepEqual([row.headline, row.phase, row.body, row.fullBody, row.receipt],
            ['2 tool calls', 'result', '', 'read_file · web_search', false],
            'the closed row counts; the tool names wait behind Expand');
        f.emit('chat', { ...final, tool_calls: 2 });
        assert.equal(f.card().dataset.finished, '1');
        assert.match(f.meta(), /2 tool calls/);
        assert.match(f.meta(), /\$0\.12/);
    } finally { f.close(); }
    const g = fixture([
        { role: 'user', text: 'read the readme', ts: TS, chat_id: 1 },
        { ...final, ts: '2026-09-15T12:00:05Z', chat_id: 1, _is_direct_chat: true },
        { ...final, role: 'system', system_type: 'task_summary', text: 'Read the readme.', rounds: 2,
            tool_calls: 2, tool_errors: 0, tool_call_counts: { read_file: 1, web_search: 1 },
            _is_direct_chat: true, ts: '2026-09-15T12:00:06Z', chat_id: 1 },
    ]);
    try {
        await g.instance.refreshHistory({ revision: 1 });
        assert.ok(g.card(), 'the same turn shows the block after a reload');
        assert.ok(g.card().querySelector('[data-turn-into-project]'), 'conversion survives a reload');
        assert.notEqual(g.card().querySelector('[data-live-title]').textContent, '', 'a finished task card carries a title');
        // The replay row is the same row: the host's totals reach the same builder.
        assert.equal(g.rows().length, 2);
        assert.ok(g.rows().some((n) => /2 tool calls/.test(n.innerHTML)));
        assert.match(g.meta(), /2 tool calls/);
    } finally { g.close(); }
});

test('a greeting with zero tools shows nothing live and nothing on reload', async () => {
    const f = fixture();
    try {
        f.census(direct());
        f.log({ type: 'task_started' });
        f.log({ type: 'llm_round_started', model: 'm', round: 1 });
        f.emit('chat', { ...final, content: 'Hi!', text: 'Hi!' });
        f.log({ ...final, type: 'task_done', status: 'completed', _is_direct_chat: true });
        f.log({ type: 'task_metrics_event', tool_calls: 0, tool_errors: 0, tool_call_counts: {} });
        assert.equal(f.card(), null);
        assert.equal(f.messages.children.filter((n) => n.classList.contains('chat-live-card')).length, 0);
    } finally { f.close(); }
    const g = fixture([
        { role: 'user', text: 'привет', ts: TS, chat_id: 1 },
        { ...final, content: 'Привет!', text: 'Привет!', ts: '2026-09-15T12:00:02Z', chat_id: 1, _is_direct_chat: true },
        { ...final, role: 'system', system_type: 'task_summary', text: 'Greeted.', rounds: 1, tool_calls: 0,
            tool_errors: 0, tool_call_counts: {}, _is_direct_chat: true, ts: '2026-09-15T12:00:03Z', chat_id: 1 },
    ]);
    try {
        await g.instance.refreshHistory({ revision: 1 });
        assert.equal(g.card(), null);
        assert.ok(g.messages.children.some((n) => /Привет!/.test(n.innerHTML)), 'the reply is a plain bubble');
    } finally { g.close(); }
});

test('a managed Swarm root keeps the task card with Turn into project; an origin-bound root gets no button', () => {
    const f = fixture();
    try {
        f.census(managed());
        f.emit('chat', { task_id: TASK, role: 'assistant', is_progress: true, content: 'Planning the swarm.' });
        assert.ok(f.card());
        assert.ok(f.card().querySelector('[data-turn-into-project]'));
        assert.equal(f.status(), 'Working...');
        globalThis.window.__ouroTaskBindings = { 'bound-root': { project_id: 'p1', chat_id: 7 } };
        f.census([...managed(), ...managed('bound-root')]);
        f.emit('chat', { task_id: 'bound-root', role: 'assistant', is_progress: true, content: 'Bound work.' });
        assert.ok(f.card('bound-root'));
        assert.equal(f.card('bound-root').querySelector('[data-turn-into-project]'), null);
    } finally { delete globalThis.window.__ouroTaskBindings; f.close(); }
});

test('lineage reclassifies a root-shaped shell: a child offers no conversion, whichever frame arrives first', () => {
    // A child's own frame can outrun the frame that names its parent (a
    // reconnecting socket, a partial window): the shell is minted root-shaped,
    // with the conversion control. A child is never a convertible unit — it
    // inherits its root's Project by lineage.
    const lineage = { delegation_role: 'subagent', parent_task_id: TASK, root_task_id: TASK, subagent_role: 'researcher' };
    const work = { task_id: 'kid-1', role: 'assistant', is_progress: true, content: 'Reading the registry.' };
    const scheduled = { ...lineage, task_id: TASK, role: 'assistant', is_progress: true, content: 'scheduled',
        subagent_event: 'scheduled', subagent_task_id: 'kid-1' };
    // Lineage on a frame that renders nothing: no row, no lifecycle event.
    const silent = { ...lineage, task_id: 'kid-1', role: 'assistant', is_progress: true, content: '' };
    const ownRow = (card) => card.children.find((node) => node.classList.contains('chat-live-actions'));
    // `rootFirst: false` — the child's frame is the first the client sees of the
    // whole tree: the root has no actions row of its own yet.
    for (const rootFirst of [true, false]) for (const order of [[work, scheduled], [scheduled, work], [work, silent]]) {
        const f = fixture();
        try {
            f.census(managed());
            if (rootFirst) f.emit('chat', { task_id: TASK, role: 'assistant', is_progress: true, content: 'Planning the swarm.' });
            for (const row of order) f.emit('chat', row);
            const kid = f.card('kid-1');
            assert.ok(kid, 'the reclassified child stays in the transcript');
            assert.equal(kid.dataset.subagent, '1');
            assert.equal(kid.dataset.parentTaskId, TASK);
            // Booleans, not nodes: a failed node comparison prints the whole stub graph.
            assert.equal(Boolean(kid.querySelector('[data-turn-into-project]')), false, 'a child offers no conversion');
            // The root's conversion sits on the root's OWN row, never on a row its child left behind.
            assert.ok(ownRow(f.card())?.querySelector('[data-turn-into-project]'), 'the root keeps its conversion');
        } finally { f.close(); }
    }
});

test('a child frame that outruns the first history load leaves no root control inside the child card', async () => {
    // Main opened mid-swarm: the child's tool frame lands before the history
    // that holds the root's own row and the lineage.
    const f = fixture([
        { role: 'assistant', is_progress: true, text: 'Planning the swarm.', content: 'Planning the swarm.', task_id: TASK, ts: TS, chat_id: 1,
            cancelable: true },
        { role: 'assistant', is_progress: true, text: 'scheduled', content: 'scheduled', task_id: TASK, ts: '2026-09-15T12:00:01Z',
            chat_id: 1, subagent_event: 'scheduled', subagent_task_id: 'kid-1', parent_task_id: TASK,
            root_task_id: TASK, delegation_role: 'subagent', subagent_role: 'researcher' },
    ]);
    try {
        f.census(managed());
        f.log({ type: 'tool_call_started', task_id: 'kid-1', tool: 'read_file', tool_call_id: 'k1' });
        await f.instance.refreshHistory({ revision: 1 });
        const kid = f.card('kid-1');
        assert.equal(kid.dataset.parentTaskId, TASK);
        for (const control of ['[data-turn-into-project]', '[data-cancel-run]']) {
            assert.equal(Boolean(kid.querySelector(control)), false, `no ${control} inside the child card`);
        }
        const own = f.card().children.find((node) => node.classList.contains('chat-live-actions'));
        assert.ok(own?.querySelector('[data-turn-into-project]'), 'the root holds its conversion on its own row');
    } finally { f.close(); }
});

test('a wake-up is an ordinary direct block: an empty frame mints nothing, a tool call mints the block', () => {
    const f = fixture();
    try {
        // No always-shown kind survives: origin does not buy a card.
        f.emit('chat', { task_id: 'wake-1', role: 'assistant', is_progress: true,
            content: '', initiator: 'consciousness' });
        assert.equal(f.card('wake-1'), null, 'an empty wake frame mints nothing');
        f.emit('chat', { task_id: TASK, role: 'assistant', is_progress: true, content: '' });
        assert.equal(f.card(), null);
        f.log({ type: 'tool_call_started', task_id: 'wake-1', tool: 'read_file', tool_call_id: 'w1',
            initiator: 'consciousness' });
        assert.ok(f.card('wake-1'), 'a tool-only wake shows the same block any direct turn shows');
    } finally { f.close(); }
});

test('the header keeps the census verdict beside a block: Thinking… for a direct block, Working… for a managed card', () => {
    const f = fixture();
    try {
        f.census(direct());
        assert.equal(f.status(), 'Thinking...');
        assert.equal(f.typingHidden(), false, 'a zero-tool turn shows the thinking indicator only');
        f.log({ type: 'tool_call_started', tool: 'read_file', tool_call_id: 'c1' });
        assert.ok(f.card());
        assert.equal(f.status(), 'Thinking...', 'a direct block is not a managed live card');
        assert.equal(f.typingHidden(), true, 'the block hosts its own running indicator');
        f.emit('chat', { ...final, tool_calls: 1 });
        f.census([]);
        assert.equal(f.status(), 'Online');
        f.census(managed('root-m'));
        f.emit('chat', { task_id: 'root-m', role: 'assistant', is_progress: true, content: 'Working on it.' });
        assert.equal(f.status(), 'Working...');
        assert.equal(f.typingHidden(), true);
    } finally { f.close(); }
});

test('a recovered tool error on replay keeps a block that names the error', async () => {
    const f = fixture([
        { role: 'user', text: 'steer it', ts: TS, chat_id: 1 },
        { ...final, ts: '2026-09-15T12:00:05Z', chat_id: 1 },
        { ...final, role: 'system', system_type: 'task_summary', text: 'One step failed and was retried.',
            rounds: 3, tool_calls: 2, tool_errors: 1, tool_call_counts: { steer_task: 2 },
            ts: '2026-09-15T12:00:06Z', chat_id: 1 },
    ]);
    try {
        await f.instance.refreshHistory({ revision: 1 });
        assert.ok(f.card());
        assert.match(f.rows()[0].innerHTML, /2 tool calls · 1 error/);
        assert.ok(f.rows()[0].classList.contains('warn'));
        assert.match(f.meta(), /1 error/);
    } finally { f.close(); }
});

// The old client forced a card whenever a terminal summary looked warn, error
// or cancelled (a sticky flag written from two places, live and on replay).
// The predicate needs no such writer: a terminal outcome other than Done is
// itself a reason to exist, so the honest chip survives every source.
test('a zero-tool turn that ended failed keeps its block live and after a reload', async () => {
    const failed = { ...final, task_terminal_status: 'failed',
        outcome_axes: { lifecycle: { status: 'failed' }, execution: { status: 'failed' } } };
    const phaseOf = (card) => card?.querySelector('[data-live-phase]')?.dataset?.phase;
    const f = fixture();
    try {
        f.census(direct());
        f.emit('chat', failed);
        f.log({ ...failed, type: 'task_done', status: 'failed' });
        assert.ok(f.card(), 'a terminal outcome other than Done is its own reason to exist');
        assert.equal(phaseOf(f.card()), 'error');
        assert.equal(f.card().querySelector('[data-live-title]').textContent, '', 'a failed greeting keeps no title placeholder');
        assert.equal(f.card().querySelector('[data-turn-into-project]'), null, 'and offers no conversion: it did no work');
        assert.equal(f.rows().length, 1, 'only the terminal note, which the predicate never counts as content');
        assert.match(f.rows()[0].innerHTML, />Failed</);
    } finally { f.close(); }
    const g = fixture([
        { role: 'user', text: 'run it', ts: TS, chat_id: 1 },
        { ...failed, ts: '2026-09-15T12:00:05Z', chat_id: 1 },
        { ...failed, role: 'system', system_type: 'task_summary', text: 'It failed.', rounds: 1,
            tool_calls: 0, tool_errors: 0, tool_call_counts: {}, ts: '2026-09-15T12:00:06Z', chat_id: 1 },
    ]);
    try {
        await g.instance.refreshHistory({ revision: 1 });
        assert.ok(g.card(), 'presence is the same on reload, with no replay-only force branch');
        assert.equal(phaseOf(g.card()), 'error');
        assert.equal(g.card().querySelector('[data-turn-into-project]'), null);
    } finally { g.close(); }
});

test('an acceptance review row keeps a block for a zero-tool turn', () => {
    const f = fixture();
    try {
        f.census(direct());
        f.emit('chat', { ...final, review_projection: { panels: [
            { surface: 'task_acceptance', panel_id: 'p1', aggregate_signal: 'PASS', reason: 'Answer matches the ask.' },
        ] } });
        assert.ok(f.card(), 'the review is content the block stands on');
        assert.ok(f.card().querySelector('[data-turn-into-project]'), 'a review group is work: conversion is offered');
        assert.equal(f.card().dataset.finished, '1');
        assert.match(f.card().querySelector('[data-live-review-summary]')?.textContent || '', /Reviews 1/);
    } finally { f.close(); }
});

// Owner decision 11.09 (2A): a turn that only addressed work draws no block —
// the annotation on the owner's message is the receipt. The host stamps the
// call (`routing_action`) and counts it (`routing_tool_calls`); no client list.
const ownerRow = { role: 'user', content: 'Turn this into a project', text: 'Turn this into a project',
    client_message_id: 'owner-1', ts: TS, chat_id: 1 };
const receipt = { annotation_type: 'routing_ack', client_message_id: 'owner-1', action: 'promote_chat_to_task',
    status: 'scheduled', target: 'managed-root', target_title: 'Requested work' };
const promote = (row = {}) => ({ tool: 'promote_chat_to_task', routing_action: 'promote_chat_to_task', tool_call_id: 'p1', ...row });

test('an addressing-only turn keeps no block live or on reload; the owner message carries the receipt', async () => {
    const f = fixture();
    try {
        f.census(direct());
        f.emit('chat', ownerRow);
        f.log({ type: 'tool_call_started', ...promote({ args: { objective: 'the project' } }) });
        f.emit('message_annotation', receipt);
        f.log({ type: 'tool_call_finished', ...promote({ duration_sec: 0.4 }) });
        f.emit('chat', { ...final, tool_calls: 1 });
        f.log({ ...final, type: 'task_done', status: 'completed', _is_direct_chat: true });
        f.log({ type: 'task_metrics_event', tool_calls: 1, tool_errors: 0, routing_tool_calls: 1, tool_call_counts: { promote_chat_to_task: 1 } });
        assert.equal(f.card(), null);
        assert.equal(f.messages.children.filter((n) => n.classList.contains('chat-live-card')).length, 0);
        const owner = f.messages.children.find((n) => n.dataset.clientMessageId === 'owner-1');
        assert.match(owner?.querySelector('.msg-routing-annotation')?.textContent || '', /Started task/);
    } finally { f.close(); }
    const g = fixture([
        { ...ownerRow, chat_annotation: receipt },
        { ...final, tool_calls: 1, ts: '2026-09-15T12:00:05Z', chat_id: 1, _is_direct_chat: true },
        { ...final, role: 'system', system_type: 'task_summary', text: 'Started the project task.', rounds: 2,
            tool_calls: 1, tool_errors: 0, routing_tool_calls: 1, tool_call_counts: { promote_chat_to_task: 1 },
            addressing_only: 'promote_chat_to_task', _is_direct_chat: true, ts: '2026-09-15T12:00:06Z', chat_id: 1 },
    ]);
    try {
        await g.instance.refreshHistory({ revision: 1 });
        assert.equal(g.card(), null, 'the replay summary of receipt-only calls is a receipt row');
        const owner = g.messages.children.find((n) => n.dataset.clientMessageId === 'owner-1');
        assert.match(owner?.querySelector('.msg-routing-annotation')?.textContent || '', /Started task/);
        assert.ok(g.messages.children.some((n) => /Here is the answer/.test(n.innerHTML)), 'the reply is a plain bubble');
    } finally { g.close(); }
});

test('addressing beside real work keeps the block: the addressing call joins the same row live and on reload', async () => {
    const f = fixture();
    try {
        f.census(direct());
        f.log({ type: 'tool_call_started', tool: 'read_file', tool_call_id: 'c1', args: { path: 'README.md' } });
        f.log({ type: 'tool_call_started', ...promote() });
        assert.ok(f.card());
        assert.equal(f.rows().length, 1, 'both calls are counted by the block\'s one row');
        const row = fold(frame('tool_call_started', TASK, { tool: 'read_file', tool_call_id: 'c1' }).toolCall,
            frame('tool_call_started', TASK, promote()).toolCall);
        assert.deepEqual([row.headline, row.fullBody, row.receipt],
            ['2 tool calls', 'read_file · promote_chat_to_task', false],
            'the addressing call is counted and named, and real work keeps the row content');
    } finally { f.close(); }
    const g = fixture([
        { role: 'user', text: 'read it and start the work', ts: TS, chat_id: 1 },
        { ...final, ts: '2026-09-15T12:00:05Z', chat_id: 1, _is_direct_chat: true },
        { ...final, role: 'system', system_type: 'task_summary', text: 'Read and started.', rounds: 2,
            tool_calls: 2, tool_errors: 0, routing_tool_calls: 1, tool_call_counts: { read_file: 1, promote_chat_to_task: 1 },
            addressing_only: 'promote_chat_to_task', _is_direct_chat: true, ts: '2026-09-15T12:00:06Z', chat_id: 1 },
    ]);
    try {
        await g.instance.refreshHistory({ revision: 1 });
        assert.ok(g.card());
        assert.ok(g.rows().some((n) => /2 tool calls/.test(n.innerHTML)));
    } finally { g.close(); }
});

// The host states the turn's totals more than once (the metrics event, the
// terminal, a replayed summary) and each fact carries its own subset. They
// merge field-wise: a later fact that states a total says nothing about the
// routing or error count, so it can neither erase one nor reclassify the row.
test('a later partial host fact keeps the routing count, the error count and the tool names of a complete one', () => {
    const addressing = {};
    noteToolCall(addressing, { key: 'p1', status: 'ok', receipt: true, tool: 'promote_chat_to_task' });
    noteToolHostMetrics(addressing, { calls: 1, errors: 0, routing: 1, counts: { promote_chat_to_task: 1 } });
    const later = noteToolHostMetrics(addressing, { calls: 1, errors: null, routing: null, counts: undefined });
    assert.deepEqual([later.headline, later.fullBody, later.receipt],
        ['1 tool call', 'promote_chat_to_task', true],
        'the addressing-only turn keeps its receipt: the second fact states a total, not the absence of routing');
    // Cold replay: no live frame ever reached this record, so the host's own
    // memory is the only source the row can read.
    const cold = {};
    noteToolHostMetrics(cold, { calls: 4, errors: 1, routing: 0, counts: { read_file: 3, web_search: 1 } });
    const after = noteToolHostMetrics(cold, { calls: 5, errors: null, routing: null, counts: {} });
    assert.deepEqual([after.headline, after.phase, after.fullBody, after.receipt],
        ['5 tool calls · 1 error', 'warn', 'read_file ×3 · web_search', false],
        'the new total lands; the known error count stays and an empty counts map never empties Expand');
});

test('a terminal that states the total alone keeps an addressing-only turn blockless when no call frame was seen', () => {
    const f = fixture();
    try {
        f.census(direct());
        f.emit('chat', ownerRow);
        f.emit('message_annotation', receipt);
        f.log({ type: 'task_metrics_event', tool_calls: 1, tool_errors: 0, routing_tool_calls: 1,
            tool_call_counts: { promote_chat_to_task: 1 } });
        assert.ok(!f.card(), 'the complete snapshot says the turn\'s one call addressed work');
        f.log({ ...final, type: 'task_done', status: 'completed', tool_calls: 1, _is_direct_chat: true });
        assert.ok(!f.card(), 'the terminal states a total; the routing fact it omits is still known');
        const owner = f.messages.children.find((n) => n.dataset.clientMessageId === 'owner-1');
        assert.match(owner?.querySelector('.msg-routing-annotation')?.textContent || '', /Started task/);
    } finally { f.close(); }
});

// V1: Stop stays reachable while a turn runs. A replayed progress row with the
// host-attested marker is "Activity unconfirmed" until a live source vouches
// for the root; the census that lists it restores Stop — on the managed card
// and on the direct block alike — and the same reading is the block's own
// Stop term, so a block never stands on a Stop it hides.
test('Stop stays reachable on a census-vouched root, managed card and direct block alike', async () => {
    const progress = { task_id: 'live-root', role: 'assistant', is_progress: true, text: 'Working on the big thing',
        content: 'Working on the big thing', cancelable: true, ts: TS, chat_id: 1 };
    for (const kind of ['managed_task', 'direct_chat']) {
        const f = fixture([progress]);
        try {
            await f.instance.refreshHistory({ revision: 1 });
            assert.ok(f.card('live-root'), 'the narration is content');
            assert.equal(f.card('live-root').querySelector('[data-cancel-run]'), null, 'unconfirmed until a live source answers');
            assert.match(f.meta('live-root'), /Activity unconfirmed/);
            f.census([{ activity_id: 'live-root', chat_id: 1, kind, phase: kind === 'managed_task' ? 'working' : 'thinking' }]);
            const stop = f.card('live-root').querySelector('[data-cancel-run]');
            assert.ok(stop, `the census restores Stop on a ${kind} root`);
            assert.equal(stop.textContent, 'Stop…');
            assert.ok(f.card('live-root').querySelector('[data-turn-into-project]'), `narration is work on a ${kind} root`);
            assert.doesNotMatch(f.meta('live-root'), /unconfirmed|unavailable/);
        } finally { f.close(); }
    }
});

// Owner decision 16.09 (Q1=A): chrome follows content, not the lane. A block
// that exists only for open attention is compact; its first content row makes
// it the task card. The header pill keeps reading the lane fact (Thinking…).
test('a wait-only block carries no title and no conversion until its first row of work', () => {
    const f = fixture();
    try {
        f.census(direct());
        f.emit('chat', wait({ role: 'system' }));
        const card = f.card();
        assert.equal(card.querySelector('[data-live-title]').textContent, '', 'no placeholder title without work');
        assert.equal(card.querySelector('[data-turn-into-project]'), null, 'nothing to convert yet');
        f.emit('chat', wait({ role: 'system', revision: 2, state: 'resolved', resolution: 'quota_restored' }));
        assert.equal(f.card(), null, 'the block leaves with its attention');
        f.log({ type: 'tool_call_started', tool: 'read_file', tool_call_id: 'c1', args: { path: 'README.md' } });
        assert.equal(f.card().querySelector('[data-live-title]').textContent, 'Working...');
        assert.ok(f.card().querySelector('[data-turn-into-project]'));
        assert.equal(f.status(), 'Thinking...', 'the header keeps the census verdict for a direct turn');
    } finally { f.close(); }
});

test('a managed root waiting for access before its first row shows its admission name and no conversion until work arrives', () => {
    const f = fixture();
    try {
        f.census(managed());
        f.emit('task_named', { task_id: TASK, suggested_name: 'Ship release' });
        f.emit('chat', wait({ role: 'system' }));
        const card = f.card();
        assert.equal(card.querySelector('[data-live-title]').textContent, 'Ship release', 'the admission name is still the title');
        assert.equal(card.querySelector('[data-turn-into-project]'), null);
        f.emit('chat', { task_id: TASK, role: 'assistant', is_progress: true, content: 'Working on it.' });
        assert.ok(card.querySelector('[data-turn-into-project]'));
        assert.equal(card.querySelector('[data-live-title]').textContent, 'Ship release');
    } finally { f.close(); }
});

test('a coined name titles a direct task card live and stays after its final', () => {
    const f = fixture();
    try {
        f.census(direct());
        f.log({ type: 'tool_call_started', tool: 'read_file', tool_call_id: 'c1', args: { path: 'README.md' } });
        assert.equal(f.card().querySelector('[data-live-title]').textContent, 'Working...');
        f.emit('task_named', { task_id: TASK, suggested_name: 'Проверка карточки' });
        assert.equal(f.card().querySelector('[data-live-title]').textContent, 'Проверка карточки');
        f.emit('chat', { ...final, tool_calls: 1 });
        assert.equal(f.card().dataset.finished, '1');
        assert.equal(f.card().querySelector('[data-live-title]').textContent, 'Проверка карточки');
        assert.equal(f.card().querySelector('[data-live-phase]').textContent, 'Done', 'the Done chip is the card\'s own');
        assert.ok(f.card().querySelector('[data-turn-into-project]'));
    } finally { f.close(); }
});

// Promote-refusal placement (Q1=A): a refused addressing call inside a live
// turn is a FAILED call — an error row, content the block stands on — while
// the owner message carries the host's sentence (`cause`) as its receipt.
// Live and on reload the block holds exactly one error row.
test('a refused promote keeps a block with exactly one error row live and on reload; the owner message carries the cause', async () => {
    const cause = 'Not started: the working folder can\'t be used';
    const refused = { annotation_type: 'routing_ack', client_message_id: 'owner-1', action: 'promote_chat_to_task',
        status: 'needs_manual_target', target: 'never-started', target_label: 'Requested work', cause };
    const f = fixture();
    try {
        f.census(direct());
        f.emit('chat', ownerRow);
        f.log({ type: 'tool_call_started', ...promote({ args: { objective: 'the project' } }) });
        assert.equal(f.card(), null, 'still a receipt while the call runs');
        f.emit('message_annotation', refused);
        f.log({ type: 'tool_call_finished', ...promote({ is_error: true, error: 'workspace_unusable', duration_sec: 0.4 }) });
        assert.ok(f.card(), 'the failed call is content');
        assert.equal(f.rows().length, 2, 'the failure keeps its own row beside the folded count');
        assert.match(f.rows()[1].innerHTML, /promote_chat_to_task/);
        // The stub cannot repaint an in-place patch; the producer states the failed row.
        const started = summarizeChatLiveEvent({ type: 'tool_call_started', task_id: TASK, ...promote({ args: { objective: 'the project' } }) });
        const failure = summarizeChatLiveEvent({ type: 'tool_call_finished', task_id: TASK, ...promote({ is_error: true, error: 'workspace_unusable', duration_sec: 0.4 }) });
        assert.deepEqual([started.dedupeKey, failure.dedupeKey], [`tools|${TASK}`, `tool:${TASK}:p1`],
            'the start feeds the folded row; the failure keeps the call\'s own row');
        assert.equal(failure.toolCall.key, started.toolCall.key, 'both frames report the same invocation');
        assert.deepEqual([started.receipt, failure.phase, failure.visible, Boolean(failure.receipt)], [true, 'error', true, false]);
        const row = fold(started.toolCall, failure.toolCall);
        assert.deepEqual([row.headline, row.phase, row.receipt], ['1 tool call · 1 error', 'warn', false],
            'a failed call counts once in the total and once as the error it explains');
        const owner = f.messages.children.find((n) => n.dataset.clientMessageId === 'owner-1');
        assert.equal(owner?.querySelector('.msg-routing-annotation')?.textContent, cause);
        f.emit('chat', { ...final, tool_calls: 1, tool_errors: 1 });
        f.log({ ...final, type: 'task_done', status: 'completed', tool_calls: 1, tool_errors: 1, _is_direct_chat: true });
        f.log({ type: 'task_metrics_event', tool_calls: 1, tool_errors: 1, routing_tool_calls: 1, tool_call_counts: { promote_chat_to_task: 1 } });
        assert.ok(f.card());
        assert.equal(f.rows().length, 3, 'the folded count, the call\'s own error row and the terminal note');
        assert.equal(f.rows().filter((n) => /promote_chat_to_task/.test(n.innerHTML)).length, 1,
            'the failure is explained once; the fold only counts it');
        assert.match(f.meta(), /1 error/);
    } finally { f.close(); }
    const g = fixture([
        { ...ownerRow, chat_annotation: refused },
        { ...final, tool_calls: 1, ts: '2026-09-15T12:00:05Z', chat_id: 1, _is_direct_chat: true },
        { ...final, role: 'system', system_type: 'task_summary', text: 'The working folder could not be used.', rounds: 2,
            tool_calls: 1, tool_errors: 1, routing_tool_calls: 1, tool_call_counts: { promote_chat_to_task: 1 },
            addressing_only: 'promote_chat_to_task', _is_direct_chat: true, ts: '2026-09-15T12:00:06Z', chat_id: 1 },
    ]);
    try {
        await g.instance.refreshHistory({ revision: 1 });
        assert.ok(g.card(), 'a counted error is content, so the replay summary is not a receipt');
        const errorRows = g.rows().filter((n) => /1 tool call · 1 error/.test(n.innerHTML));
        assert.equal(errorRows.length, 1);
        assert.ok(errorRows[0].classList.contains('warn'));
        assert.match(g.meta(), /1 error/);
        const owner = g.messages.children.find((n) => n.dataset.clientMessageId === 'owner-1');
        assert.equal(owner?.querySelector('.msg-routing-annotation')?.textContent, cause);
    } finally { g.close(); }
});

// The fold is a per-invocation state map, not a pair of counters: the transport
// may deliver a call's frames twice, out of order, or interleaved with another
// call's, and the row must still read the turn correctly.
test('the fold counts one invocation once, whatever order and however many frames report it', () => {
    const observe = (key, status, tool) => ({ key, status, receipt: false, tool });
    const row = fold(observe('c1', 'ok', 'read_file'), observe('c1', 'calling', 'read_file'),
        observe('c1', 'ok', 'read_file'));
    assert.deepEqual([row.headline, row.phase, row.fullBody, row.calls],
        ['1 tool call', 'result', 'read_file', 1], 'a late start cannot reopen a finished call');
});

test('concurrent calls keep the row calling until the last one lands', () => {
    const record = {};
    const phase = () => toolEvidenceView(record.toolFold).phase;
    noteToolCall(record, { key: 'a', status: 'calling', receipt: false, tool: 'read_file' });
    noteToolCall(record, { key: 'b', status: 'calling', receipt: false, tool: 'web_search' });
    noteToolCall(record, { key: 'b', status: 'ok', receipt: false, tool: 'web_search' });
    assert.deepEqual([toolEvidenceView(record.toolFold).headline, phase()], ['2 tool calls', 'calling']);
    noteToolCall(record, { key: 'a', status: 'ok', receipt: false, tool: 'read_file' });
    assert.equal(phase(), 'result', 'the row rests only when nothing is in flight');
});

test('a timeout and a failure for one call are one error', () => {
    const observe = (status) => ({ key: 'a', status, receipt: false, tool: 'bash' });
    const row = fold(observe('calling'), observe('error'), observe('error'), observe('ok'));
    assert.deepEqual([row.headline, row.phase, row.errors], ['1 tool call · 1 error', 'warn', 1],
        'an error is counted once and no later frame can take it back');
});

test('a recycled card counts no invocation from the cycle before it', () => {
    const record = {};
    noteToolCall(record, { key: 'a', status: 'ok', receipt: false, tool: 'read_file' });
    noteToolHostMetrics(record, { calls: 3, errors: 1, routing: 0, counts: { read_file: 3 } });
    clearStickyCardState(record);
    assert.equal(record.toolFold, null);
    assert.deepEqual([toolEvidenceView(record.toolFold).headline, record.toolCalls, record.toolErrors],
        ['0 tool calls', null, null]);
});

// Stationary row (owner decision): the evidence stays at the first call's place
// and time, so a burst of calls never walks it down past the narration.
test('the folded row keeps the place and the time of the first frame it counted', () => {
    const record = { groupId: 'g', items: [] };
    const view = (headline) => ({ phase: 'calling', headline, body: '', fullBody: '', receipt: false });
    upsertToolFoldRow(record, view('1 tool call'), '12:00:00', '2026-09-15T12:00:00Z');
    record.items.push({ dedupeKey: 'progress:note', headline: 'Inspecting the source.', ts: '12:00:30', count: 1 });
    const second = upsertToolFoldRow(record, view('2 tool calls'), '12:01:00', '2026-09-15T12:01:00Z');
    assert.deepEqual([second.timelineUpdate, second.patchIndex], ['patch-at', 0]);
    assert.deepEqual([record.items[0].headline, record.items[0].ts], ['2 tool calls', '12:00:00'],
        'the count moves with the turn, the timestamp does not');
    assert.equal(record.items.length, 2, 'and no second row appears');
});

test('the folded row is one item: the block counts notes, not tool calls', () => {
    const f = fixture();
    try {
        f.census(direct());
        for (const [id, tool] of [['c1', 'read_file'], ['c2', 'read_file'], ['c3', 'web_search']]) {
            f.log({ type: 'tool_call_started', tool, tool_call_id: id });
            f.log({ type: 'tool_call_finished', tool, tool_call_id: id, duration_sec: 0.2 });
        }
        assert.equal(f.rows().length, 1, 'six frames about three calls are one row');
        const count = f.card().querySelector('[data-live-count]');
        assert.equal(count.hidden, true, 'one item is not a list');
        f.emit('chat', { task_id: TASK, role: 'assistant', is_progress: true, content: 'Inspecting the source.' });
        assert.equal(f.rows().length, 2);
        assert.match(f.rows()[1].innerHTML, /Inspecting the source/, 'the narration lands after the stationary row');
        assert.equal(count.textContent, '2 notes', 'the fold counts as one note beside the narration');
    } finally { f.close(); }
});

// A child card reads the same voice fact its parent's card reads: the worker
// stamps every progress frame it emits, inside a child's turn as well, so a
// checkpoint or fallback note there is a visible row and nothing more.
const CHILD = 'child-1';
const childNote = (patch = {}) => ({ role: 'assistant', is_progress: true, task_id: CHILD,
    subagent_task_id: CHILD, parent_task_id: TASK, root_task_id: TASK, delegation_role: 'subagent',
    subagent_event: 'progress', subagent_role: 'scout', model: 'm', ...patch });

test('a host note inside a child leaves the child\'s collapsed line and title alone; the child\'s own notes lead', () => {
    const f = fixture();
    try {
        f.census(managed());
        f.emit('chat', { task_id: TASK, role: 'assistant', is_progress: true, content: 'Planning the swarm.' });
        f.emit('chat', childNote({ content: 'Reading the spec.', narration: true }));
        const card = f.card(CHILD);
        assert.ok(card, 'the child frame mints the child card');
        const activity = () => card.querySelector('[data-live-activity]').textContent;
        const title = () => card.querySelector('[data-live-title]').textContent;
        const lineage = title();
        assert.equal(activity(), 'Reading the spec.', 'the child\'s own note is its collapsed activity');
        f.emit('chat', childNote({ content: 'Falling back to the second model.', narration: false }));
        assert.equal(activity(), 'Reading the spec.', 'the host note inside the child claims nothing');
        assert.equal(title(), lineage, 'and the child keeps its lineage title');
        f.emit('chat', childNote({ content: 'Comparing the two runs.', narration: true }));
        assert.equal(activity(), 'Comparing the two runs.', 'the child\'s next note moves the line again');
        f.emit('chat', childNote({ content: 'A frame from before the fact existed.' }));
        assert.equal(activity(), 'A frame from before the fact existed.', 'an absent fact stays legacy narration');
        f.emit('chat', childNote({ subagent_event: 'completed', status: 'completed',
            result: 'Scouted the module.', narration: false }));
        assert.equal(activity(), 'Scouted the module.', 'the child\'s terminal is the host\'s own account and leads');
    } finally { f.close(); }
});

// The disclosed residual: a child folds its calls while they happen, and the
// host's at-rest metrics stay the owner's (`noteToolMetrics` skips children),
// so a reloaded child card carries no evidence row.
test('a child card folds its own tool calls live and takes no at-rest evidence row', () => {
    const f = fixture();
    try {
        f.census(managed());
        f.emit('chat', childNote({ content: 'Reading the spec.', narration: true }));
        assert.ok(f.card(CHILD), 'the child card exists before its first call');
        f.log({ type: 'tool_call_started', task_id: CHILD, tool: 'read_file', tool_call_id: 'k1' });
        f.log({ type: 'tool_call_finished', task_id: CHILD, tool: 'read_file', tool_call_id: 'k1', duration_sec: 0.2 });
        f.log({ type: 'tool_call_started', task_id: CHILD, tool: 'web_search', tool_call_id: 'k2' });
        // A child card renders its timeline only while it is open.
        f.card(CHILD).querySelector('[data-live-summary-button]').listeners.get('click')[0]({ detail: 0 });
        const folded = () => f.rows(CHILD).filter((n) => /tool call/.test(n.innerHTML));
        assert.equal(folded().length, 1, 'three frames about two calls are the child\'s one evidence row');
        assert.match(folded()[0].innerHTML, /2 tool calls/);
        f.log({ type: 'task_metrics_event', task_id: CHILD, tool_calls: 5, tool_errors: 0,
            tool_call_counts: { read_file: 5 } });
        assert.equal(folded().length, 1, 'the at-rest fact belongs to the owning turn: a child takes no row from it');
        assert.doesNotMatch(folded()[0].innerHTML, /5 tool calls/);
    } finally { f.close(); }
});

test('#931 a typed checkpoint is a real expanded row without stealing narration', () => {
    const f = fixture();
    try {
        f.census(managed());
        f.emit('chat', { task_id: TASK, role: 'assistant', is_progress: true,
            narration: true, content: 'Reading the source' });
        const card = f.card();
        const title = card.querySelector('[data-live-title]').textContent;
        f.log({ type: 'task_checkpoint', checkpoint_kind: 'context_view', round: 3 });
        if (card.dataset.expanded !== '1') {
            card.querySelector('[data-live-summary-button]').listeners.get('click')[0]({ detail: 0 });
        }
        assert.ok(f.rows().some((row) => row.innerHTML.includes('Context inspected')));
        assert.equal(card.querySelector('[data-live-title]').textContent, title);
        f.log({ type: 'task_checkpoint', checkpoint_kind: 'context_view', round: 3 });
        assert.equal(f.rows().filter((row) => row.innerHTML.includes('Context inspected')).length, 1);
        const count = f.rows().length;
        f.log({ type: 'worker_starting', worker_id: 0 });
        assert.equal(f.rows().length, count);
    } finally { f.close(); }
});
