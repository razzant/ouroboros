// The real Chat consumer: typing a host row must not turn it into a final.
import assert from 'node:assert/strict';
import test from 'node:test';
import { createChatInstance } from '../modules/chat.js';
import { installDom, restoreDom, walkCard } from './chat_dom_fixture.js';

function openChat(rows = []) {
    const { prior, mount } = installDom(async (url) => ({ ok: true, json: async () =>
        String(url).startsWith('/api/chat/history') ? { messages: rows }
            : { active_direct_turns: [], status: 'running' } }));
    const handlers = new Map();
    const ws = { on(type, fn) { handlers.set(type, fn); return () => handlers.delete(type); },
        isConnected: () => true, send() {} };
    const instance = createChatInstance({ ws,
        state: { activePage: 'chat', projectChatIds: new Set(), unreadCount: 0 },
        updateUnreadBadge() {}, stateSnapshots: { begin: () => ({ generation: 1, requestedAt: Date.now() }), gate() { return Promise.resolve(this.begin()); },
            isCurrent: () => true, apply() {} }, chatId: 1, idPrefix: 'chat', mountEl: mount });
    return { prior, instance, handlers, messages: () => globalThis.document.byId.get('chat-messages') };
}
const bubbles = (messages, role) => messages.children.filter(node =>
    node.classList.contains('chat-bubble') && node.classList.contains(role)
    && !node.classList.contains('typing-bubble'));

test('command replies use System voice while typed proactive model speech stays assistant', () => {
    const c = openChat();
    try {
        c.handlers.get('chat')({ chat_id: 1, role: 'system', system_type: 'command_reply',
            content: 'Background consciousness: enabled <plain>', markdown: false,
            ts: '2026-09-19T00:00:00Z' });
        c.handlers.get('chat')({ chat_id: 1, role: 'assistant', system_type: 'proactive_message',
            content: 'My own reply', markdown: false, ts: '2026-09-19T00:00:01Z' });
        assert.equal(bubbles(c.messages(), 'system').length, 1);
        assert.match(bubbles(c.messages(), 'system')[0].innerHTML, /&lt;plain&gt;/);
        assert.equal(bubbles(c.messages(), 'assistant').length, 1);
        assert.match(bubbles(c.messages(), 'assistant')[0].innerHTML, /My own reply/);
    } finally { c.instance.destroy(); restoreDom(c.prior); }
});

test('a typed nonterminal host notice preserves its working card on live delivery and reload', async () => {
    const progress = { chat_id: 1, role: 'assistant', task_id: 'voice-task',
        is_progress: true, narration: true, system_type: 'model_narration',
        text: 'Reading a source', ts: '2026-09-19T00:00:00Z' };
    const notice = { chat_id: 1, role: 'system', task_id: 'voice-task',
        system_type: 'attachment_notice', text: 'Some attachments were rejected.',
        ts: '2026-09-19T00:00:01Z' };
    const c = openChat([progress, notice]);
    try {
        c.handlers.get('chat')({ ...progress, content: progress.text });
        const card = walkCard(c.messages(), 'voice-task');
        assert.ok(card);
        c.handlers.get('chat')({ ...notice, content: notice.text });
        assert.notEqual(card.dataset.finished, '1');
        await c.instance.refreshHistory({ revision: 1 });
        assert.notEqual(walkCard(c.messages(), 'voice-task').dataset.finished, '1');
        assert.ok(bubbles(c.messages(), 'system').some(node => node.innerHTML.includes(notice.text)));
    } finally { c.instance.destroy(); restoreDom(c.prior); }
});

test('a typed host failure with an explicit terminal fact still concludes on replay', async () => {
    const c = openChat([
        { task_id: 'failed-voice', role: 'assistant', is_progress: true,
            text: 'Working', ts: '2026-09-19T00:00:00Z' },
        { task_id: 'failed-voice', role: 'system', system_type: 'task_error',
            task_terminal_status: 'failed', text: 'Runtime error', ts: '2026-09-19T00:00:01Z' },
    ]);
    try {
        await c.instance.refreshHistory({ revision: 1 });
        assert.equal(walkCard(c.messages(), 'failed-voice').dataset.finished, '1');
    } finally { c.instance.destroy(); restoreDom(c.prior); }
});
