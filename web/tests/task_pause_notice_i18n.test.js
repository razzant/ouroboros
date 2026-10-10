import assert from 'node:assert/strict';
import test from 'node:test';
import { createChatInstance } from '../modules/chat.js';
import { buildTimelineItemHtml, cardRowSummary } from '../modules/chat_activity.js';
import { applyPayload, tx } from '../modules/i18n.js';
import { installDom, restoreDom } from './chat_dom_fixture.js';

const ENGLISH = "The task paused because of the owner's Pause. Its work is retained. Use Resume on the task when available.";
const TRANSLATED = 'Задача приостановлена владельцем. Работа сохранена. Продолжение доступно в карточке задачи.';
const notice = { role: 'system', system_type: 'task_pause_notice', task_id: 'paused-root',
    card_row: 'timeline', card_row_id: 'pause:paused-root:one', narration: false,
    text: ENGLISH, content: ENGLISH, ts: '2026-10-04T12:00:00Z', chat_id: 1 };
const russian = () => applyPayload({ language: 'ru', english: false, entries: {
    [ENGLISH]: { text: TRANSLATED, provenance: 'imported' },
} });

test('typed System pause card rows translate through the installed memory without changing source bytes', () => {
    const env = installDom();
    try {
        russian();
        assert.equal(tx(ENGLISH), TRANSLATED, 'the exact sentence is in the translation memory');
        for (const [row, expected] of [[notice, TRANSLATED], [{ ...notice, role: 'assistant' }, ENGLISH],
            [{ ...notice, system_type: 'other_notice' }, ENGLISH]]) {
            const original = JSON.stringify(row);
            const summary = cardRowSummary(row, 'warn', row.ts);
            const html = buildTimelineItemHtml({ ...summary, lineKey: 'pause-one' },
                { groupId: row.task_id, expandedLineKeys: new Set() });
            assert.equal(summary.headline, expected);
            assert.ok(html.includes(expected));
            assert.equal(JSON.stringify(row), original, 'neither host nor model evidence is rewritten');
        }
    } finally { applyPayload({ language: 'en', english: true }); restoreDom(env.prior); }
});

for (const history of [false, true]) test(`pause notice without a card translates ${history ? 'on history replay' : 'live'} but preserves model speech and stored English`, async () => {
    const env = installDom(async url => ({ ok: true, json: async () => String(url).startsWith('/api/chat/history')
        ? { messages: history ? [notice] : [], window: { complete: true } } : { active_direct_turns: [] } }));
    let instance;
    try {
        russian();
        const handlers = new Map();
        let generation = 0;
        instance = createChatInstance({ ws: { on(type, fn) { handlers.set(type, fn); return () => handlers.delete(type); },
            isConnected: () => true, send() {} },
            state: { activePage: 'chat', projectChatIds: new Set(), unreadCount: 0 }, updateUnreadBadge() {},
            chatId: 1, idPrefix: 'chat', mountEl: env.mount,
            stateSnapshots: { begin: () => ({ generation: ++generation, requestedAt: Date.now() }),
                gate() { return Promise.resolve(this.begin()); }, isCurrent: () => true, apply() {} },
        });
        await instance.refreshHistory({ revision: 1 });
        if (!history) handlers.get('chat')({ ...notice });
        const feed = document.byId.get('chat-messages');
        const bubble = feed.children.find(node => node.classList.contains('system') && node.classList.contains('chat-bubble'));
        assert.ok(bubble, 'a missing card leaves the real System fallback');
        assert.ok(bubble.innerHTML.includes(TRANSLATED));
        assert.equal(bubble.innerHTML.includes(ENGLISH), false);
        assert.equal(feed.children.some(node => node.classList.contains('chat-live-card')), false);
        const saved = JSON.parse(sessionStorage.getItem('ouro_chat'));
        assert.equal(saved.find(row => row.systemType === 'task_pause_notice').text, ENGLISH);
        handlers.get('chat')({ ...notice, role: 'assistant', task_id: 'model-root' });
        const speech = feed.children.find(node => node.classList.contains('assistant')
            && node.classList.contains('chat-bubble') && !node.classList.contains('typing-bubble'));
        assert.ok(speech.innerHTML.includes(ENGLISH), 'typed metadata never turns model words into host copy');
        assert.equal(speech.innerHTML.includes(TRANSLATED), false);
    } finally { instance?.destroy(); applyPayload({ language: 'en', english: true }); restoreDom(env.prior); }
});
