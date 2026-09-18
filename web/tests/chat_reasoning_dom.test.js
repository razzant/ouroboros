import assert from 'node:assert/strict';
import test from 'node:test';
import { createChatInstance } from '../modules/chat.js';
import { setReasoningVisible } from '../modules/log_events.js';
import { installDom, restoreDom, walkCard } from './chat_dom_fixture.js';

// ---------------------------------------------------------------------------
// Visible reasoning: a `reasoning: true` progress frame (live or replayed)
// reaches the card as its own collapsed "Thinking" line; the stamp is forwarded
// by updateLiveCardFromProgressMessage, so an unstamped frame stays a working line.
// ---------------------------------------------------------------------------
function walkLines(node, out = []) {
    if (node?.classList?.contains('chat-live-line')) out.push(node);
    for (const child of node?.children || []) walkLines(child, out);
    return out;
}
test('a reasoning-stamped progress frame renders as a collapsed Thinking line, live and from history', async () => {
    // Reasoning rows are hidden by default (the `show_reasoning` preference);
    // this test is the shown state, restored at the end.
    setReasoningVisible(true);
    // Long enough that the preview body is shorter than fullBody (the line's Expand toggle).
    const reasoning = 'Weigh the two migration paths before touching the schema. '.repeat(6).trim();
    const rows = [
        { chat_id: 2, role: 'system', is_progress: true, reasoning: true, task_id: 'think-h',
          content: `💬 ${reasoning}`, ts: '2026-09-11T10:00:00Z' },
        { chat_id: 2, role: 'system', is_progress: true, task_id: 'think-h',
          content: '💬 editing the schema', ts: '2026-09-11T10:00:01Z' },
    ];
    const { prior, mount } = installDom(async (url) => {
        if (String(url).startsWith('/api/chat/history')) {
            return { ok: true, json: async () => ({ messages: rows, window: { complete: true } }) };
        }
        return { ok: true, json: async () => ({ active_direct_turns: [] }) };
    });
    const handlers = new Map();
    const ws = {
        on(type, fn) { handlers.set(type, fn); return () => handlers.delete(type); },
        isConnected: () => true, send() {},
    };
    let generation = 0;
    const stateSnapshots = {
        begin: () => ({ generation: ++generation, requestedAt: Date.now() }),
        isCurrent: () => true, apply() {},
    };
    let instance;
    try {
        instance = createChatInstance({
            ws, state: { activePage: 'chat', projectChatIds: new Set(), unreadCount: 0 },
            updateUnreadBadge() {}, stateSnapshots, chatId: 2, idPrefix: 'chat', mountEl: mount, asPanel: true,
        });
        const messages = globalThis.document.byId.get('chat-messages');
        // A collapsed card defers its timeline DOM; expand through the summary button.
        const expand = (card) => card.querySelector('[data-live-summary-button]').listeners.get('click')[0]({ detail: 0 });
        for (const row of rows) handlers.get('chat')({ ...row, task_id: 'think-l' });
        expand(walkCard(messages, 'think-l'));
        const live = walkLines(walkCard(messages, 'think-l'));
        assert.deepEqual(live.map((n) => n.classList.contains('thinking')), [true, false]);
        assert.ok(live[0].classList.contains('expandable'), 'the reasoning line is collapsible');
        assert.equal(live[1].classList.contains('working'), true);
        await instance.refreshHistory({ revision: 1 });
        expand(walkCard(messages, 'think-h'));
        const replayed = walkLines(walkCard(messages, 'think-h'));
        assert.deepEqual(replayed.map((n) => n.classList.contains('thinking')), [true, false]);
    } finally {
        setReasoningVisible(false);
        instance?.destroy();
        restoreDom(prior);
    }
});
