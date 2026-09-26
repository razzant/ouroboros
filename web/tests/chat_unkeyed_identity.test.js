import assert from 'node:assert/strict';
import test from 'node:test';
import { createChatInstance } from '../modules/chat.js';
import { installDom, restoreDom, walkCard } from './chat_dom_fixture.js';

test('unkeyed live progress cannot mutate or finish the current task card', async () => {
    const { prior, mount } = installDom(async () => ({ ok: true,
        json: async () => ({ active_direct_turns: [] }) }));
    const handlers = new Map();
    const ws = {
        on(type, fn) { handlers.set(type, fn); return () => handlers.delete(type); },
        isConnected: () => true, send() {},
    };
    let generation = 0, instance;
    const stateSnapshots = {
        begin: () => ({ generation: ++generation, requestedAt: Date.now() }), gate() { return Promise.resolve(this.begin()); },
        isCurrent: () => true, apply() {},
    };
    try {
        instance = createChatInstance({
            ws, state: { activePage: 'chat', projectChatIds: new Set(), unreadCount: 0 },
            updateUnreadBadge() {}, stateSnapshots, chatId: 2, idPrefix: 'chat', mountEl: mount,
            asPanel: true,
        });
        await new Promise((resolve) => setTimeout(resolve, 0));
        const messages = globalThis.document.byId.get('chat-messages');
        const emit = (task_id, content, extra = {}) => handlers.get('chat')({
            chat_id: 2, role: 'system', is_progress: true, task_id, content,
            ts: '2026-08-25T00:00:00Z', ...extra,
        });
        emit('task-a', 'Task A is working');
        const card = walkCard(messages, 'task-a');
        assert.ok(card, 'keyed progress creates the owner card');
        const title = card.querySelector('[data-live-title]').textContent;
        const timeline = card.querySelector('[data-live-timeline]');
        const noteCount = timeline.children.length;

        emit('', 'Review task B failed', { narration: false });
        assert.equal(card.querySelector('[data-live-title]').textContent, title,
            'an ownerless note cannot rename the current card');
        assert.equal(timeline.children.length, noteCount,
            'an ownerless note cannot append to the current card');

        emit('', 'Task B done', { is_progress: false, system_type: 'task_summary', text: 'Task B done' });
        assert.equal(card.dataset.finished, '0',
            'an ownerless summary cannot finish the current card');

        emit('task-a', 'Task A continues');
        assert.ok(timeline.children.length > noteCount,
            'a keyed note still updates its own card');
    } finally {
        instance?.destroy();
        restoreDom(prior);
    }
});
