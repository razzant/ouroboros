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
import test from 'node:test';
import { createChatInstance } from '../modules/chat.js';
import { summarizeChatLiveEvent } from '../modules/log_events.js';
import { ElementStub, installDom, restoreDom, walkCard } from './chat_dom_fixture.js';

// The flat fixture's querySelector does not descend; the status badge and
// card internals need a real descendant lookup.
const originalQuery = ElementStub.prototype.querySelector;
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
        stateSnapshots: { begin: () => ({ generation: ++generation, requestedAt: Date.now() }),
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

test('a tool frame stamped with the lane fact mints a direct block before any census lists the turn', () => {
    const f = fixture();
    try {
        f.log({ type: 'tool_call_started', tool: 'read_file', tool_call_id: 'c1', args: { path: 'README.md' }, _is_direct_chat: true });
        assert.ok(f.card(), 'the first tool call mints the block');
        assert.equal(f.card().dataset.direct, '1', 'the frame carries the lane, the census is not awaited');
        assert.equal(f.card().querySelector('[data-turn-into-project]'), null, 'no conversion control in the pre-census window');
        f.log({ type: 'tool_call_finished', tool: 'read_file', tool_call_id: 'c1', args: { path: 'README.md' }, duration_sec: 0.3, _is_direct_chat: true });
        f.census(direct());
        assert.equal(f.card().dataset.direct, '1');
        assert.equal(f.card().querySelector('[data-turn-into-project]'), null);
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
        assert.equal(f.card().dataset.direct, '1');
        f.census(direct());
        assert.ok(f.card().querySelector('[data-cancel-run]'));
        f.log({ ...final, type: 'task_done', status: 'completed', _is_direct_chat: true });
        assert.equal(f.card()?.querySelector('[data-cancel-run]') ?? null, null, 'no Stop on a finished turn');
    } finally { f.close(); }
});

test('a direct turn with two successful tools shows two compact rows live and the summary row after a reload', async () => {
    const f = fixture();
    try {
        f.census(direct());
        f.log({ type: 'tool_call_started', tool: 'read_file', tool_call_id: 'c1', args: { path: 'README.md' } });
        f.log({ type: 'tool_call_finished', tool: 'read_file', tool_call_id: 'c1', args: { path: 'README.md' }, duration_sec: 0.3 });
        f.log({ type: 'tool_call_started', tool: 'web_search', tool_call_id: 'c2', args: { query: 'ouroboros' } });
        f.log({ type: 'tool_call_finished', tool: 'web_search', tool_call_id: 'c2', args: { query: 'ouroboros' }, duration_sec: 1.2 });
        assert.ok(f.card(), 'the first tool call mints the block live');
        assert.equal(f.card().dataset.direct, '1', 'direct chrome from the census fact');
        assert.equal(f.card().querySelector('[data-turn-into-project]'), null, 'no conversion on a direct block');
        assert.equal(f.card().querySelector('[data-live-title]').textContent, '', 'no placeholder title');
        assert.equal(f.rows().length, 2, 'start and finish of one call share a row');
        assert.match(f.rows()[0].innerHTML, /read_file · README\.md/);
        assert.match(f.rows()[1].innerHTML, /web_search · ouroboros/);
        // The stub cannot repaint an in-place patch; the producer states the finished row.
        const started = summarizeChatLiveEvent({ type: 'tool_call_started', task_id: TASK, tool: 'read_file', tool_call_id: 'c1', args: { path: 'README.md' } });
        const finished = summarizeChatLiveEvent({ type: 'tool_call_finished', task_id: TASK, tool: 'read_file', tool_call_id: 'c1', args: { path: 'README.md' }, duration_sec: 0.3 });
        assert.equal(started.dedupeKey, finished.dedupeKey);
        assert.deepEqual([started.phase, started.visible, finished.phase, finished.headline], ['calling', true, 'ok', 'read_file · README.md · ✓ 0.3s']);
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
        assert.equal(g.card().dataset.direct, '1', 'replay reads the same host fact from the summary row');
        assert.equal(g.card().querySelector('[data-turn-into-project]'), null);
        // Per-tool rows are live-only: replay carries the summary row and the completion note.
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
        assert.equal(f.card().dataset.direct, '0');
        assert.ok(f.card().querySelector('[data-turn-into-project]'));
        assert.equal(f.status(), 'Working...');
        globalThis.window.__ouroTaskBindings = { 'bound-root': { project_id: 'p1', chat_id: 7 } };
        f.census([...managed(), ...managed('bound-root')]);
        f.emit('chat', { task_id: 'bound-root', role: 'assistant', is_progress: true, content: 'Bound work.' });
        assert.ok(f.card('bound-root'));
        assert.equal(f.card('bound-root').querySelector('[data-turn-into-project]'), null);
    } finally { delete globalThis.window.__ouroTaskBindings; f.close(); }
});

test('background consciousness keeps its card without content while an ordinary empty frame mints nothing', () => {
    const f = fixture();
    try {
        f.emit('chat', { task_id: 'bg-consciousness', role: 'assistant', is_progress: true, content: '' });
        assert.ok(f.card('bg-consciousness'), 'the always-shown kind is a block on its own');
        f.emit('chat', { task_id: TASK, role: 'assistant', is_progress: true, content: '' });
        assert.equal(f.card(), null);
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

test('addressing beside real work keeps the block: the receipt row renders inside it live, the summary row on reload', async () => {
    const f = fixture();
    try {
        f.census(direct());
        f.log({ type: 'tool_call_started', tool: 'read_file', tool_call_id: 'c1', args: { path: 'README.md' } });
        f.log({ type: 'tool_call_started', ...promote() });
        assert.ok(f.card());
        assert.equal(f.rows().length, 2, 'the receipt row is shown honestly once the block exists');
        assert.match(f.rows()[1].innerHTML, /promote_chat_to_task/);
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
            assert.equal(f.card('live-root').dataset.direct, kind === 'managed_task' ? '0' : '1');
            assert.doesNotMatch(f.meta('live-root'), /unconfirmed|unavailable/);
        } finally { f.close(); }
    }
});
