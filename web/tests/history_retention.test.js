import assert from 'node:assert/strict';
import test from 'node:test';
import { historyRetentionView, syncHistoryRetentionItem } from '../modules/history_retention.js';
import { renderLiveCardMeta } from '../modules/chat_activity.js';
import { categorizeLogEvent, summarizeChatLiveEvent, summarizeLogEvent, taskTerminalPhase } from '../modules/log_events.js';
import { createChatInstance } from '../modules/chat.js';
import { installDom, restoreDom, walkCard } from './chat_dom_fixture.js';

test('normal history progress is detail-only; problems clear after successful retry', () => {
    const { prior } = installDom();
    try {
        const record = { groupId: 'done', items: [], metaEl: { isConnected: true, innerHTML: '' } };
        for (const status of ['pending', 'problem', 'complete']) {
            const detail = { status: 'completed', history_retention: { status } };
            assert.equal(syncHistoryRetentionItem(record, detail), true);
            assert.equal(syncHistoryRetentionItem(record, detail), false);
            assert.equal(record.items.length, 1);
            assert.equal(record.items[0].body, historyRetentionView(detail).body);
            renderLiveCardMeta(record);
            assert.equal(record.metaEl.innerHTML.includes('History storage problem'), status === 'problem');
            assert.equal(taskTerminalPhase(detail), 'done', 'placement never changes task outcome');
        }
        assert.equal(syncHistoryRetentionItem(record, {}), false);
        assert.equal(record.items[0].headline, 'Task history saved');
    } finally { restoreDom(prior); }
});

test('retention Logs name the work and counts without minting chat messages', () => {
    for (const status of ['deferred', 'pending', 'problem', 'complete']) {
        const event = { type: 'history_retention', task_id: 'done', status, pending_count: 2, problem_count: 0 };
        const view = summarizeLogEvent(event);
        assert.equal(view.headline, historyRetentionView(event).headline);
        assert.ok(view.meta.includes('pending_count=2'));
        assert.equal(categorizeLogEvent(event), 'tasks');
        assert.equal(summarizeChatLiveEvent(event).visible, false);
    }
});

test('details and Logs show every grouped failure reason instead of a circular pointer', () => {
    const history_retention = { status: 'problem', problem_reasons: [
        { reason: 'source_missing', count: 2 }, { reason: 'OSError: disk full', count: 1 },
    ] };
    const detail = historyRetentionView({ history_retention });
    const log = summarizeLogEvent({ type: 'history_retention', ...history_retention });
    assert.equal(detail.body, log.body);
    assert.match(log.body, /2 × source_missing\nOSError: disk full/);
    assert.doesNotMatch(log.body, /See Logs/);
    assert.match(historyRetentionView({ history_retention: { status: 'problem' } }).body, /No failure reason was recorded/);
});

test('Details hydrates retention on the real finished card and live retry clears its problem', async () => {
    const id = 'history-task';
    const detail = { task_id: id, status: 'completed', history_retention: { status: 'pending' } };
    const { prior, mount } = installDom(async (url) => ({ ok: true, json: async () =>
        String(url).startsWith('/api/tasks/') ? detail
            : String(url).startsWith('/api/chat/history') ? { messages: [] } : { active_direct_turns: [] } }));
    const handlers = new Map();
    const ws = { on(type, fn) { handlers.set(type, fn); return () => handlers.delete(type); }, isConnected: () => true, send() {} };
    let instance;
    try {
        instance = createChatInstance({ ws, state: { activePage: 'chat', projectChatIds: new Set(), unreadCount: 0 },
            updateUnreadBadge() {}, stateSnapshots: { begin: () => ({ generation: 1, requestedAt: Date.now() }),
                gate() { return Promise.resolve(this.begin()); }, isCurrent: () => true, apply() {} },
            chatId: 1, idPrefix: 'chat', mountEl: mount });
        await new Promise((resolve) => setTimeout(resolve, 0));
        handlers.get('chat')({ task_id: id, chat_id: 1, role: 'system', is_progress: true,
            content: 'Building the report', ts: '2026-09-27T11:59:59Z' });
        handlers.get('chat')({ task_id: id, chat_id: 1, role: 'system', system_type: 'task_summary',
            content: 'Finished.', task_terminal_status: 'completed', ts: '2026-09-27T12:00:00Z' });
        const card = walkCard(globalThis.document.byId.get('chat-messages'), id);
        assert.equal(card.dataset.finished, '1');
        assert.doesNotMatch(card.querySelector('[data-live-meta]').innerHTML, /history/i);
        const toggle = () => card.querySelector('[data-live-disclosure]').listeners.get('click')[0]({ detail: 0 });
        if (card.dataset.expanded === '1') toggle();
        toggle();
        assert.equal(card.dataset.expanded, '1');
        await new Promise((resolve) => setTimeout(resolve, 0));
        assert.equal(card.querySelector('[data-live-timeline]').children
            .filter((node) => node.dataset.liveLineKey === `history-retention-${id}`).length, 1);
        assert.doesNotMatch(card.querySelector('[data-live-meta]').innerHTML, /history/i);
        for (const status of ['problem', 'complete']) {
            handlers.get('log')({ chat_id: 1, data: { type: 'history_retention', task_id: id, status } });
            assert.equal(card.querySelector('[data-live-meta]').innerHTML.includes('History storage problem'), status === 'problem');
            assert.equal(card.dataset.finished, '1');
            assert.equal(card.querySelector('[data-live-phase]').textContent, 'Done');
        }
        assert.equal(card.querySelector('[data-live-timeline]').children
            .filter((node) => node.dataset.liveLineKey === `history-retention-${id}`).length, 1);
    } finally { instance?.destroy(); restoreDom(prior); }
});

test('cold history shows storage problems before expansion and a newer summary clears them', async () => {
    const id = 'cold-history';
    const rows = [
        { task_id: id, chat_id: 1, role: 'system', is_progress: true, content: 'Preparing a report',
            ts: '2026-09-27T11:59:59Z' },
        { task_id: id, chat_id: 1, role: 'system', system_type: 'task_summary', content: 'Finished.',
            task_terminal_status: 'completed', ts: '2026-09-27T12:00:00Z', history_retention: { status: 'problem' } },
    ];
    const { prior, mount } = installDom(async (url) => ({ ok: true, json: async () =>
        String(url).startsWith('/api/chat/history') ? { messages: rows } : { active_direct_turns: [] } }));
    let instance;
    try {
        instance = createChatInstance({ ws: { on() { return () => {}; }, isConnected: () => true, send() {} },
            state: { activePage: 'chat', projectChatIds: new Set(), unreadCount: 0 }, updateUnreadBadge() {},
            stateSnapshots: { begin: () => ({ generation: 1, requestedAt: Date.now() }),
                gate() { return Promise.resolve(this.begin()); }, isCurrent: () => true, apply() {} },
            chatId: 1, idPrefix: 'chat', mountEl: mount });
        await instance.refreshHistory({ revision: 1 });
        const card = () => walkCard(globalThis.document.byId.get('chat-messages'), id);
        assert.match(card().querySelector('[data-live-meta]').innerHTML, /History storage problem/);
        rows[1].history_retention = { status: 'complete' };
        await instance.refreshHistory({ revision: 2 });
        assert.doesNotMatch(card().querySelector('[data-live-meta]').innerHTML, /History storage problem/);
        assert.equal(card().querySelector('[data-live-phase]').textContent, 'Done');
    } finally { instance?.destroy(); restoreDom(prior); }
});
