import assert from 'node:assert/strict';
import test from 'node:test';
import { reviewGateLine, syncReviewGateItem } from '../modules/review_gate_card.js';
import { renderLiveCardMeta } from '../modules/chat_activity.js';
import { createChatInstance } from '../modules/chat.js';
import { installDom, restoreDom, walkCard } from './chat_dom_fixture.js';

const LONG = `[CRITICAL] [triad] tests_affected: ${'the changed contract has no test of its refusal; '.repeat(8)}`.trim();
const fact = (count, reason = 'critical_findings', taskId = 'work-1') => ({ count, latest: {
    reason, task_id: taskId, text: LONG, at: '2026-11-11T10:00:00+00:00' } });

test('the field is a count and the latest reason, twin of review_overrides.card_line', () => {
    assert.equal(reviewGateLine({ count: 2, latest: { reason: 'review_quorum' } }),
        'Review: 2× continued where Blocking would have stopped · latest: review quorum');
    assert.equal(reviewGateLine({ count: 12 }), 'Review: 12× continued where Blocking would have stopped · latest: unknown');
    for (const none of [undefined, null, {}, { count: 0 }, { count: '3' }]) assert.equal(reviewGateLine(none), '');
});

test('one Details item and one meta phrase, replaced in place; an older count never wins', () => {
    const record = { groupId: 'work-1', items: [], metaEl: { isConnected: true, innerHTML: '' } };
    assert.equal(syncReviewGateItem(record, {}), false);
    assert.equal(syncReviewGateItem(record, { review_overrides: fact(11) }), true);
    assert.equal(syncReviewGateItem(record, { review_overrides: fact(11) }), false, 'the same decision again changes nothing');
    assert.equal(syncReviewGateItem(record, { review_overrides: fact(12, 'review_quorum', 'child-1') }), true);
    assert.equal(syncReviewGateItem(record, { review_overrides: fact(11) }), false, 'a stale read keeps the newer count');
    assert.equal(record.items.length, 1);
    const [item] = record.items;
    assert.equal(item.headline, 'Review gate did not prohibit continuation');
    assert.match(item.body, /^12 times the review gate let this work continue where Blocking enforcement would have stopped it/);
    assert.match(item.body, /\nLatest \(task child-1\): review quorum — \[CRITICAL\]/);
    assert.ok(item.body.endsWith(LONG), 'Details shows the stored reason as stored (its head, review_overrides.py)');
    const { prior } = installDom();
    try {
        renderLiveCardMeta(record);
    } finally { restoreDom(prior); }
    assert.match(record.metaEl.innerHTML, /Review: 12× continued where Blocking would have stopped · latest: review quorum/);
    assert.doesNotMatch(record.metaEl.innerHTML, /CRITICAL/, 'the compact line never carries the long text');
});

function chat(mount, ws) {
    return createChatInstance({ ws, state: { activePage: 'chat', projectChatIds: new Set(), unreadCount: 0 },
        updateUnreadBadge() {}, stateSnapshots: { begin: () => ({ generation: 1, requestedAt: Date.now() }),
            gate() { return Promise.resolve(this.begin()); }, isCurrent: () => true, apply() {} },
        chatId: 1, idPrefix: 'chat', mountEl: mount });
}

test('a live decision marks the work card, never a row or a child card, and Details keeps it after the end', async () => {
    const id = 'work-1';
    const { prior, mount } = installDom(async (url) => ({ ok: true, json: async () =>
        String(url).startsWith(`/api/tasks/${id}`) ? { task_id: id, status: 'running', review_overrides: fact(11) }
            : String(url).startsWith('/api/chat/history') ? { messages: [] } : { active_direct_turns: [] } }));
    const handlers = new Map();
    const ws = { on(type, fn) { handlers.set(type, fn); return () => handlers.delete(type); }, isConnected: () => true, send() {} };
    let instance;
    try {
        instance = chat(mount, ws);
        await new Promise((resolve) => setTimeout(resolve, 0));
        handlers.get('chat')({ task_id: id, chat_id: 1, role: 'system', is_progress: true,
            content: 'Changing my body', ts: '2026-11-11T09:59:59Z' });
        const messages = globalThis.document.byId.get('chat-messages');
        const before = messages.children.length;
        handlers.get('log')({ chat_id: 1, data: { type: 'review_advisory_override', task_id: 'child-1', work_task_id: id,
            block_reason: 'critical_findings', review_overrides: fact(12, 'critical_findings', 'child-1') } });
        const card = walkCard(messages, id);
        assert.match(card.querySelector('[data-live-meta]').innerHTML,
            /Review: 12× continued where Blocking would have stopped · latest: critical findings/);
        assert.equal(messages.children.length, before, 'no row joins the feed');
        assert.equal(walkCard(messages, 'child-1'), null, "the work's card holds it, no card is minted for the actor");
        const toggle = () => card.querySelector('[data-live-disclosure]').listeners.get('click')[0]({ detail: 0 });
        if (card.dataset.expanded !== '1') toggle();
        await new Promise((resolve) => setTimeout(resolve, 0));  // the Details read answers an older count
        assert.match(card.querySelector('[data-live-meta]').innerHTML, /12× continued/);
        assert.equal(card.querySelector('[data-live-timeline]').children
            .filter((node) => node.dataset.liveLineKey === `review-gate-${id}`).length, 1);
        handlers.get('chat')({ task_id: id, chat_id: 1, role: 'system', system_type: 'task_summary',
            content: 'Finished.', task_terminal_status: 'completed', ts: '2026-11-11T10:05:00Z' });
        assert.equal(card.dataset.finished, '1');
        assert.match(card.querySelector('[data-live-meta]').innerHTML, /12× continued/);
    } finally { instance?.destroy(); restoreDom(prior); }
});

test('a page load shows the field from its history row, before Details and without a detail read', async () => {
    // The history endpoint carries the work's fact on its summary row (gateway/history.py); a work without one shows
    // nothing. Opening the card is not needed, and no card fetches its detail for it.
    const rows = ['cold-work', 'plain-work'].flatMap((id) => [
        { task_id: id, chat_id: 1, role: 'system', is_progress: true, content: 'Changed my body', ts: '2026-11-11T09:00:00Z' },
        { task_id: id, chat_id: 1, role: 'system', system_type: 'task_summary', content: 'Finished.',
            task_terminal_status: 'completed', ts: '2026-11-11T09:10:00Z',
            ...(id === 'cold-work' ? { review_overrides: fact(3) } : {}) },
    ]);
    const reads = [];
    const { prior, mount } = installDom(async (url) => ({ ok: true, json: async () => {
        if (String(url).startsWith('/api/tasks/')) reads.push(String(url));
        return String(url).startsWith('/api/chat/history') ? { messages: rows } : { active_direct_turns: [] };
    } }));
    let instance;
    try {
        instance = chat(mount, { on() { return () => {}; }, isConnected: () => true, send() {} });
        await instance.refreshHistory({ revision: 1 });
        const meta = (id) => walkCard(globalThis.document.byId.get('chat-messages'), id).querySelector('[data-live-meta]').innerHTML;
        assert.match(meta('cold-work'), /Review: 3× continued where Blocking would have stopped · latest: critical findings/);
        assert.doesNotMatch(meta('plain-work'), /continued where Blocking/, 'a row without the fact shows no field');
        assert.deepEqual(reads, [], 'no per-card detail read');
    } finally { instance?.destroy(); restoreDom(prior); }
});
