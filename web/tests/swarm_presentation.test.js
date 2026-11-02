// A Swarm message and its root (docs/DESIGN.md "Composer owner controls"): the quiet `Swarm` in the time
// line of an owner message the host accepted under Swarm, and one line on a Swarm root's card in exactly the receipt's
// words (SYN2 A.8): helpers started, models, the money known and what is not final, subscription sessions.
import assert from 'node:assert/strict';
import test from 'node:test';
import { markSwarmLabel, swarmReceiptText, syncSwarmReceipt } from '../modules/swarm_presentation.js';
import { createChatInstance } from '../modules/chat.js';
import { installDom, restoreDom, walkCard } from './chat_dom_fixture.js';

const TS = '2026-10-25T00:31:00Z';
const swarmRoot = (extra = {}) => ({ task_id: 'root-1', status: 'completed', metadata: { subagents_mode: 'swarm' }, ...extra });
const attempt = (task_id, model, kind = 'attempt', extra = {}) => ({ attempt_id: `${task_id}-${model}-${kind}`, task_id, model, kind, ...extra });
// The receipt's actors as the host folds them (ouroboros/swarm_receipt_sources.py): the root a supervisor, a child a
// task, a delegated run on the task that owns it; `start` is the recorded start, null before one.
const actor = (id, kind, task_id, start = TS, extra = {}) => ({ id, kind, task_id, start, ...extra });
const children = (count, root = 'root-1') => [actor(`task:${root}`, 'supervisor', root),
    ...Array.from({ length: count }, (_, at) => actor(`task:child-${at + 1}`, 'task', `child-${at + 1}`, TS, { parent_task_id: root }))];
const FORBIDDEN = /mind|wave|famil|peak|parallel|%|\$0\.00/i;

test('the receipt line says helpers, models, known money + not final, and subscription sessions; nothing else', () => {
    const attempts = [
        attempt('root-1', 'claude-opus-5-5'), attempt('child-1', 'gpt-6'), attempt('child-2', 'gemini-4'),
        ...['s1', 's2', 's3', 's4', 's5', 's6', 's7'].map((id) => attempt(id, 'gpt-6', 'subscription_session')),
        attempt('s1', 'gpt-6', 'subscription_session', { attempt_id: 's1-resumed' }),
    ];
    const full = swarmRoot({ swarm_efficiency: { subagent_count: 5, fanout_count: 2 }, swarm_receipt: { state: 'closed',
        actor_intervals: children(5), cost: { known_final_usd: 2.3, unknown_attempts: 2, total_usd: null, attempts } } });
    assert.equal(swarmReceiptText(full), 'Swarm · 5 helpers started · 3 models · $2.30 known + 2 not final · 7 subscription sessions');
    const one = swarmRoot({ swarm_efficiency: { subagent_count: 1 }, swarm_receipt: { actor_intervals: children(1),
        cost: { known_final_usd: 0.4, unknown_attempts: 0, attempts: [attempt('child-1', 'gpt-6', 'subscription_session')] } } });
    assert.equal(swarmReceiptText(one), 'Swarm · 1 helper started · 1 model · $0.40 known · 1 subscription session');
    // An amount nobody knows yet is a word, never $0; a free tree has no money part at all.
    const unknown = swarmRoot({ swarm_efficiency: { subagent_count: 2 }, swarm_receipt: { actor_intervals: children(2),
        cost: { known_final_usd: 0, unknown_attempts: 3, attempts: [attempt('child-1', 'gpt-6')] } } });
    assert.equal(swarmReceiptText(unknown), 'Swarm · 2 helpers started · 1 model · 3 not final');
    const free = swarmRoot({ swarm_efficiency: { subagent_count: 2 }, swarm_receipt: { actor_intervals: children(2),
        cost: { known_final_usd: 0, unknown_attempts: 0 } } });
    assert.equal(swarmReceiptText(free), 'Swarm · 2 helpers started');
    // No helper started: said so, with no reason the record does not hold.
    const solo = swarmRoot({ swarm_efficiency: { intent_source: 'swarm', planned: null, observed_started: 0, status: 'no_fanout_observed' },
        swarm_receipt: { cost: { known_final_usd: 0.12, unknown_attempts: 0, attempts: [attempt('root-1', 'claude-opus-5-5')] } } });
    assert.equal(swarmReceiptText(solo), 'Swarm requested · no helpers started');
    for (const detail of [full, one, unknown, free, solo]) assert.doesNotMatch(swarmReceiptText(detail), FORBIDDEN);
});

test('a start is said only from a start fact: a request the host refused is no helper that started', () => {
    // Emitted, then refused before it started (the room went Off): the request is counted, the child never started.
    const refused = swarmRoot({ swarm_efficiency: { subagent_count: 1, fanout_count: 1 }, swarm_receipt: { actor_intervals: [
        actor('task:root-1', 'supervisor', 'root-1'), actor('task:child-1', 'task', 'child-1', null, { parent_task_id: 'root-1' })],
    cost: { known_final_usd: 0.1, unknown_attempts: 0, attempts: [attempt('root-1', 'claude-opus-5-5')] } } });
    assert.equal(swarmReceiptText(refused), 'Swarm requested · no helpers started');
    // Each helper once: a child's waits split its actor, its own runs are that child; the root's own run is a helper of
    // its own; the root, a review and a start still pending are none.
    const started = swarmRoot({ swarm_efficiency: { subagent_count: 4 }, swarm_receipt: { actor_intervals: [
        actor('task:root-1', 'supervisor', 'root-1'), actor('task:root-1:before:w1', 'supervisor', 'root-1'),
        actor('task:child-1', 'task', 'child-1'), actor('task:child-1:before:w2', 'task', 'child-1'),
        actor('run:r1', 'delegated_run', 'child-1', 1798162260.5),
        actor('task:child-2', 'task', 'child-2', null), actor('run:r2', 'delegated_run', 'child-2', '2026-10-25T00:32:00+00:00'),
        actor('task:child-3', 'task', 'child-3', null),
        actor('run:r3', 'delegated_run', 'root-1'),
        actor('invocation:i1', 'delegated_start', 'root-1', null, { state: 'pending' }),
        actor('review:o1', 'review', 'root-1'), actor('review_operation:op1', 'review_custody', 'root-1', null)],
    cost: { known_final_usd: 0.6, unknown_attempts: 0, attempts: [attempt('child-1', 'gpt-6')] } } });
    assert.equal(swarmReceiptText(started), 'Swarm · 3 helpers started · 1 model · $0.60 known',
        'child-1 once, child-2 by its run, the root\'s run; never child-3, the root, the review or the pending start');
    // No start recorded yet (a receipt without actors, or none at all): the requests are what is known.
    const requested = swarmRoot({ swarm_efficiency: { subagent_count: 3 },
        swarm_receipt: { cost: { known_final_usd: 0.2, unknown_attempts: 1, attempts: [attempt('child-1', 'gpt-6')] } } });
    assert.equal(swarmReceiptText(requested), 'Swarm · 3 helpers requested · 1 model · $0.20 known + 1 not final');
    assert.equal(swarmReceiptText(swarmRoot({ swarm_efficiency: { subagent_count: 1 } })), 'Swarm · 1 helper requested');
    for (const detail of [refused, started, requested]) assert.doesNotMatch(swarmReceiptText(detail), FORBIDDEN);
});

test('only a Swarm root with recorded facts has a line', () => {
    const facts = { swarm_efficiency: { subagent_count: 3 }, swarm_receipt: { cost: { known_final_usd: 1, unknown_attempts: 0 } } };
    assert.equal(swarmReceiptText(swarmRoot()), '', 'a running root has recorded nothing yet');
    assert.equal(swarmReceiptText({ ...swarmRoot(facts), metadata: { subagents_mode: 'auto' } }), '', 'Auto: no Swarm line');
    assert.equal(swarmReceiptText({ ...facts, task_id: 'root-1' }), '', 'a root admitted before the mode existed');
    assert.equal(swarmReceiptText(swarmRoot({ ...facts, parent_task_id: 'root-0' })), '', 'a helper inherits the mode, not the line');
    assert.equal(swarmReceiptText(null), '');
});

test('the card takes the line first in its foot, keeps it through frames without facts, and revises it', () => {
    const { prior } = installDom();
    try {
        const root = document.createElement('div');
        const footEl = document.createElement('div');
        const meta = document.createElement('div');
        footEl.append(meta);
        root.append(footEl);
        const record = { root, footEl };
        const detail = swarmRoot({ swarm_efficiency: { subagent_count: 2 }, swarm_receipt: { actor_intervals: children(2),
            cost: { known_final_usd: 1.5, unknown_attempts: 1 } } });
        assert.equal(syncSwarmReceipt(record, detail), true);
        const line = footEl.children[0];
        assert.equal(line.className, 'chat-live-swarm-receipt');
        assert.equal(line.textContent, 'Swarm · 2 helpers started · $1.50 known + 1 not final');
        assert.equal(line.hasAttribute('data-i18n-skip'), true, 'data, not interface words');
        assert.equal(syncSwarmReceipt(record, detail), false, 'the same facts write nothing');
        assert.equal(syncSwarmReceipt(record, { task_id: 'root-1', status: 'completed' }), false, 'an event without the facts');
        assert.equal(footEl.children[0], line, 'the line stays');
        detail.swarm_receipt.cost = { known_final_usd: 1.9, unknown_attempts: 0 };
        assert.equal(syncSwarmReceipt(record, detail), true, 'a later revision');
        assert.equal(line.textContent, 'Swarm · 2 helpers started · $1.90 known');
        assert.equal(footEl.children.length, 2);
        assert.equal(syncSwarmReceipt({ root, footEl, isSubagent: true }, detail), false, 'a helper\'s card has none');
        assert.equal(syncSwarmReceipt(undefined, detail), false);
    } finally { restoreDom(prior); }
});

test('the label: `Swarm ·` before the time, only for an owner row the host accepted under Swarm, once', () => {
    const { prior } = installDom();
    try {
        const feed = document.createElement('div');
        const bubble = document.createElement('div');
        bubble.className = 'chat-bubble user';
        bubble.dataset.clientMessageId = 'cm-1';
        const time = document.createElement('div');
        time.className = 'msg-time';
        time.textContent = '00:31';
        bubble.append(time);
        feed.append(bubble);
        const row = { role: 'user', client_message_id: 'cm-1', subagents_mode: 'swarm' };
        assert.equal(markSwarmLabel(feed, { ...row, subagents_mode: 'auto' }), false);
        assert.equal(markSwarmLabel(feed, { ...row, role: 'assistant' }), false);
        assert.equal(markSwarmLabel(feed, { ...row, client_message_id: 'cm-2' }), false, 'no bubble of that id');
        assert.equal(markSwarmLabel(feed, row), true, 'the owner\'s own echo finds its bubble by id');
        const [label, dot] = time.children;
        assert.equal(label.className, 'msg-swarm');
        assert.equal(label.children.at(-1).textContent, 'Swarm');
        assert.equal(label.title, 'Sent while Subagents was on Swarm');
        assert.deepEqual([dot.className, dot.textContent, dot.getAttribute('aria-hidden')], ['msg-swarm-sep', '·', 'true']);
        assert.equal(markSwarmLabel(feed, row, bubble), false, 'once');
    } finally { restoreDom(prior); }
});

function chatFixture(history = [], detail = null) {
    const env = installDom(async (url) => ({ ok: true, json: async () =>
        String(url).startsWith('/api/chat/history') ? { messages: history, window: { complete: true } }
            : String(url).startsWith('/api/tasks/') ? detail
                : { active_chat_activities: [], active_chat_activities_complete: true, supervisor_ready: true } }));
    globalThis.sessionStorage.setItem('ouro_chat_session_id', 'session-me');
    const handlers = new Map();
    const instance = createChatInstance({
        ws: { on(type, fn) { handlers.set(type, fn); return () => handlers.delete(type); }, isConnected: () => true,
            send: () => ({ status: 'sent', clientMessageId: 'cm-1' }) },
        state: { activePage: 'chat', projectChatIds: new Set(), unreadCount: 0 }, updateUnreadBadge() {},
        chatId: 1, idPrefix: 'chat', mountEl: env.mount,
        stateSnapshots: { begin: () => ({ generation: 1, requestedAt: Date.now() }), gate() { return Promise.resolve(this.begin()); },
            isCurrent: () => true, apply() {}, fail() {}, latest: () => null },
    });
    const messages = document.byId.get('chat-messages');
    const label = (bubble) => bubble?.querySelector('.msg-time')?.querySelector('.msg-swarm') || null;
    return { instance, handlers, messages, label, close() { instance.destroy(); restoreDom(env.prior); } };
}
const settle = async () => { for (let i = 0; i < 8; i += 1) await new Promise((resolve) => setTimeout(resolve, 0)); };

test('chat: history and live owner rows carry the label by their own fact; my own echo labels my bubble', async () => {
    const row = (id, extra = {}) => ({ role: 'user', text: `message ${id}`, ts: TS, chat_id: 1, source: 'telegram', history_id: `h-${id}`, ...extra });
    const f = chatFixture([row('a', { subagents_mode: 'swarm' }), row('b')]);
    try {
        await f.instance.refreshHistory({ revision: 1 });
        const bubble = (id) => f.messages.children.find((node) => node.dataset.historyId === `h-${id}`);
        assert.ok(f.label(bubble('a')), 'a row without a client id still says Swarm');
        assert.equal(f.label(bubble('b')), null, 'no fact, no label');
        f.handlers.get('chat')({ role: 'user', chat_id: 1, ts: TS, source: 'web', content: 'from my phone',
            sender_session_id: 'session-other', client_message_id: 'cm-other', subagents_mode: 'swarm' });
        assert.ok(f.label(f.messages.children.find((node) => node.dataset.clientMessageId === 'cm-other')));
        document.byId.get('chat-input').value = 'think it through with everyone';
        for (const fn of document.byId.get('chat-send').listeners.get('click')) fn();
        await settle();
        const mine = () => f.messages.children.find((node) => node.dataset.clientMessageId === 'cm-1');
        assert.equal(f.label(mine()), null, 'nothing is claimed before the host accepts it');
        f.handlers.get('chat')({ role: 'user', chat_id: 1, ts: TS, source: 'web', content: 'think it through with everyone',
            sender_session_id: 'session-me', client_message_id: 'cm-1', subagents_mode: 'swarm' });
        assert.ok(f.label(mine()), 'the accepted row says it');
        assert.equal(mine().querySelector('.msg-time').querySelectorAll('.msg-swarm').length, 1);
    } finally { f.close(); }
});

test('chat: a Swarm root\'s card shows the receipt line once Details reads its detail', async () => {
    const id = 'swarm-root';
    const detail = swarmRoot({ task_id: id, swarm_efficiency: { subagent_count: 4 }, swarm_receipt: { actor_intervals: children(4, id),
        cost: { known_final_usd: 0.85, unknown_attempts: 1, attempts: [attempt(id, 'claude-opus-5-5'), attempt('c1', 'gpt-6')] } } });
    const f = chatFixture([], detail);
    try {
        await settle();
        f.handlers.get('chat')({ task_id: id, chat_id: 1, role: 'system', is_progress: true, content: 'Asking four helpers', ts: TS });
        f.handlers.get('chat')({ task_id: id, chat_id: 1, role: 'system', system_type: 'task_summary', content: 'Done.',
            task_terminal_status: 'completed', ts: TS });
        const card = walkCard(f.messages, id);
        const line = () => card.querySelector('[data-live-foot]').querySelector('.chat-live-swarm-receipt');
        assert.equal(line(), null, 'no line from events alone');
        const toggle = () => card.querySelector('[data-live-disclosure]').listeners.get('click')[0]({ detail: 0 });
        if (card.dataset.expanded === '1') toggle();
        toggle();
        await settle();
        assert.equal(line()?.textContent, 'Swarm · 4 helpers started · 2 models · $0.85 known + 1 not final');
        assert.equal(card.querySelector('[data-live-foot]').children[0], line(), 'first in the foot, above the meta line');
    } finally { f.close(); }
});
