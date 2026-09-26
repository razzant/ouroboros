// The origin label of a self-initiated turn (a consciousness wake-up): every
// frame of the turn carries `initiator: 'consciousness'`, the block's meta
// line names it, the final bubble signs it through the same sender line the
// user bubble uses, Logs files the rows under Consciousness — live and on
// reload alike. An owner's turn carries no label and renders as before.
import assert from 'node:assert/strict';
import test from 'node:test';
import { createChatInstance } from '../modules/chat.js';
import { senderLabel } from '../modules/chat_activity.js';
import { categorizeLogEvent, summarizeChatLiveEvent, taskTerminalSummary } from '../modules/log_events.js';
import { ElementStub, installDom, restoreDom, walkCard } from './chat_dom_fixture.js';

// The flat fixture's querySelector does not descend; card internals need a
// real descendant lookup (same patch as chat_activity_block.test.js).
const originalQuery = ElementStub.prototype.querySelector;
ElementStub.prototype.querySelector = function (selector) {
    const direct = originalQuery.call(this, selector);
    if (direct) return direct;
    for (const child of this.children) { const found = child.querySelector(selector); if (found) return found; }
    return null;
};

const TS = '2026-09-16T12:00:00Z';
const TASK = 'wake-a';

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
        instance,
        meta: (id = TASK) => walkCard(messages, id)?.querySelector('[data-live-meta]')?.innerHTML || '',
        bubbles: () => nodes(messages).filter((n) => n.classList?.contains('chat-bubble') && n.classList.contains('assistant')),
        emit: (type, row) => handlers.get(type)({ chat_id: 1, ts: TS, ...row }),
        census: (rows) => instance.hydrateStateSnapshot({
            active_chat_activities: rows, active_chat_activities_complete: true, supervisor_ready: true,
        }, Infinity, ++generation),
        close() { instance.destroy(); restoreDom(env.prior); },
    };
}

const direct = (id = TASK) => [{ activity_id: id, chat_id: 1, kind: 'direct_chat', phase: 'thinking' }];
const progress = (patch = {}) => ({ task_id: TASK, role: 'assistant', is_progress: true, text: '💬 Reading the day',
    content: '💬 Reading the day', cancelable: true, initiator: 'consciousness', ...patch });
const final = (patch = {}) => ({ task_id: TASK, role: 'assistant', content: 'Nothing needs you today.',
    text: 'Nothing needs you today.', task_terminal_status: 'completed',
    outcome_axes: { execution: { status: 'ok' } }, initiator: 'consciousness', ...patch });

test('the sender line signs a wake-up final and leaves every other bubble alone', () => {
    assert.equal(senderLabel('assistant', false, '', { initiator: 'consciousness' }), 'Ouroboros · Consciousness');
    assert.equal(senderLabel('assistant', false, '', {}), 'Ouroboros');
    assert.equal(senderLabel('assistant', true, '', { initiator: 'consciousness' }), '💬 Thought');
    assert.equal(senderLabel('system', false, 'task_summary', { initiator: 'consciousness' }), '📋 Task Summary');
    assert.equal(senderLabel('user', false, '', { initiator: 'consciousness' }), 'You');
});

test('Logs files a wake-up row under Consciousness by its origin fact; errors stay errors', () => {
    const wake = { initiator: 'consciousness' };
    assert.equal(categorizeLogEvent({ type: 'tool_call_finished', tool: 'read_file', is_error: false, task_id: 'w', ...wake }), 'consciousness');
    assert.equal(categorizeLogEvent({ type: 'task_heartbeat', task_id: 'w', phase: 'running', ...wake }), 'consciousness');
    assert.equal(categorizeLogEvent({ type: 'send_message', is_progress: true, task_id: 'w', ...wake }), 'consciousness');
    assert.equal(categorizeLogEvent({ type: 'tool_call_finished', tool: 'run_command', is_error: true, task_id: 'w', ...wake }), 'errors');
    assert.equal(categorizeLogEvent({ type: 'tool_call_finished', tool: 'read_file', is_error: false, task_id: 't' }), 'tools');
    assert.equal(categorizeLogEvent({ type: 'send_message', is_progress: true, task_id: 't' }), 'tasks');
    // The origin fact is the ONLY signal: the retired loop's pseudo id is not one.
    assert.equal(categorizeLogEvent({ type: 'send_message', is_progress: true, task_id: 'bg-consciousness' }), 'tasks');
});

test('the live projections carry the label so any frame can name the block', () => {
    assert.equal(summarizeChatLiveEvent({ type: 'send_message', is_progress: true, task_id: 'w', content: '💬 hi', initiator: 'consciousness' }).initiator, 'consciousness');
    assert.equal(summarizeChatLiveEvent({ type: 'tool_call_started', tool: 'read_file', task_id: 'w', initiator: 'consciousness' }).initiator, 'consciousness');
    assert.equal('initiator' in summarizeChatLiveEvent({ type: 'send_message', is_progress: true, task_id: 't', content: '💬 hi' }), false);
    assert.equal(taskTerminalSummary({ task_id: 'w', task_terminal_status: 'completed', initiator: 'consciousness' }).initiator, 'consciousness');
    assert.equal('initiator' in taskTerminalSummary({ task_id: 't', task_terminal_status: 'completed' }), false);
});

test('live: the block meta line says Consciousness and the final bubble is signed', () => {
    const f = fixture();
    try {
        f.census(direct());
        f.emit('chat', progress());
        assert.match(f.meta(), /Consciousness/);
        f.emit('chat', final());
        const [bubble] = f.bubbles();
        assert.ok(bubble, 'the final is an ordinary assistant bubble');
        assert.match(bubble.innerHTML, /class="sender">Ouroboros · Consciousness</);
        assert.match(f.meta(), /Consciousness/, 'the terminal frame keeps the label on the block');
    } finally { f.close(); }
});

test('live: a tool-only wake names its block from the stamped tool frame', () => {
    const f = fixture();
    try {
        f.census(direct());
        f.emit('log', { chat_id: 1, data: { type: 'tool_call_finished', tool: 'read_file', is_error: false,
            task_id: TASK, ts: TS, initiator: 'consciousness', _is_direct_chat: true, cancelable: true } });
        assert.match(f.meta(), /Consciousness/);
    } finally { f.close(); }
});

test('an owner turn shows no label anywhere', () => {
    const f = fixture();
    try {
        f.census(direct('turn-o'));
        f.emit('chat', progress({ task_id: 'turn-o', initiator: undefined }));
        assert.doesNotMatch(f.meta('turn-o'), /Consciousness/);
        f.emit('chat', final({ task_id: 'turn-o', initiator: undefined }));
        assert.match(f.bubbles()[0].innerHTML, /class="sender">Ouroboros</);
        assert.doesNotMatch(f.bubbles()[0].innerHTML, /Consciousness/);
    } finally { f.close(); }
});

test('reload: the replayed rows keep the label on the block and on the bubble', async () => {
    const f = fixture([
        { ...progress(), ts: TS, chat_id: 1 },
        { ...final(), ts: '2026-09-16T12:00:05Z', chat_id: 1 },
    ]);
    try {
        await f.instance.refreshHistory({ revision: 1 });
        assert.match(f.meta(), /Consciousness/);
        assert.match(f.bubbles()[0].innerHTML, /class="sender">Ouroboros · Consciousness</);
    } finally { f.close(); }
});
