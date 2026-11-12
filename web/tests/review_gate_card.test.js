import assert from 'node:assert/strict';
import test from 'node:test';
import { reviewGateLine, syncReviewGateItem } from '../modules/review_gate_card.js';
import { renderLiveCardMeta } from '../modules/chat_activity.js';
import { createChatInstance } from '../modules/chat.js';
import { installDom, restoreDom, walkCard } from './chat_dom_fixture.js';

const LONG = `[CRITICAL] [triad] tests_affected: ${'the changed contract has no test of its refusal; '.repeat(8)}`.trim();
const WHY = `The lock order is intended: ${'the writer takes the store first and the counter second; '.repeat(5)}`.trim();
const fact = (count, reason = 'critical_findings', taskId = 'work-1') => ({ count, latest: {
    reason, task_id: taskId, text: LONG, at: '2026-11-11T10:00:00+00:00' } });
const continued = (count, rationale = WHY) => ({ count, latest: { at: '2026-11-11T10:01:00+00:00',
    disposition: 'rejected', rationale, reason: 'critical_findings', record_id: 'rec-1' } });
const landed = (count) => ({ count, latest: { sha: 'abc123def4567890'.padEnd(40, '0'), at: '2026-11-11T10:02:00+00:00' } });

test('the field names each recorded part, twin of review_overrides.card_line', () => {
    assert.equal(reviewGateLine({ count: 2, latest: { reason: 'review_quorum' } }),
        'Review: FAIL or no verdict 2× (latest: review quorum)');
    assert.equal(reviewGateLine({ count: 12 }), 'Review: FAIL or no verdict 12× (latest: unknown)');
    assert.equal(reviewGateLine({ ...fact(3), continued: continued(1, 'the lock order is intended'), landed: landed(2) }),
        'Review: FAIL or no verdict 3× (latest: critical findings) · author continued 1× (latest: rejected — "the lock '
        + 'order is intended") · landed over it: 2 commits');
    assert.equal(reviewGateLine({ continued: continued(2), landed: landed(1) }),
        `Review: author continued 2× over critical findings (latest: rejected — "${Array.from(WHY).slice(0, 79).join('')
            .trimEnd()}…") · landed over it: 1 commit`, 'a part another work counted is absent here, not zero: the next '
        + 'part names what it went over');
    assert.equal(reviewGateLine({ landed: landed(2) }), 'Review: landed over a FAIL or no verdict: 2 commits',
        'a landing alone names what it went over');
    assert.equal(reviewGateLine({ count: 1, continued: {}, landed: { count: 0 } }),
        'Review: FAIL or no verdict 1× (latest: unknown)', 'an old field shows no zero for a part it never recorded');
    for (const none of [undefined, null, {}, { count: 0 }, { count: '3' }, { continued: { count: 0 } }, 'fact']) {
        assert.equal(reviewGateLine(none), '');
    }
    assert.doesNotMatch(reviewGateLine({ ...fact(1), continued: continued(1), landed: landed(1) }), /Blocking|PASS|merged/);
});

test('one Details item and one meta phrase, replaced in place; an older read never wins', () => {
    const record = { groupId: 'work-1', items: [], metaEl: { isConnected: true, innerHTML: '' } };
    assert.equal(syncReviewGateItem(record, {}), false);
    assert.equal(syncReviewGateItem(record, { review_overrides: fact(11) }), true);
    assert.equal(syncReviewGateItem(record, { review_overrides: fact(11) }), false, 'the same decision again changes nothing');
    assert.equal(syncReviewGateItem(record, { review_overrides: fact(12, 'review_quorum', 'child-1') }), true);
    assert.equal(syncReviewGateItem(record, { review_overrides: fact(11) }), false, 'a stale read keeps the newer count');
    const decided = { ...fact(12, 'review_quorum', 'child-1'), continued: continued(1) };
    assert.equal(syncReviewGateItem(record, { review_overrides: decided }), true, "the author's continuation is news");
    assert.equal(syncReviewGateItem(record, { review_overrides: fact(12, 'review_quorum', 'child-1') }), false,
        'a read from before the continuation keeps it');
    assert.equal(syncReviewGateItem(record, { review_overrides: { ...decided, landed: landed(1) } }), true, 'so is a landing');
    assert.equal(record.items.length, 1);
    const [item] = record.items;
    assert.equal(item.headline, 'Review returned FAIL or no verdict');
    assert.equal(item.sourceTs, '2026-11-11T10:02:00+00:00', 'the newest part dates the item');
    assert.match(item.body, /^A review of this work returned FAIL or no verdict 12 times\./);
    assert.match(item.body, /\nLatest \(task child-1\): review quorum — \[CRITICAL\]/);
    assert.ok(item.body.includes(`${LONG}\n`), 'Details shows the stored reason as stored (its head, review_overrides.py)');
    assert.ok(item.body.includes(`\nThe author explicitly continued once; latest: rejected — "${WHY}"\n`),
        "Details quotes the author's stored head whole");
    assert.ok(item.body.endsWith('\nLanded over it: 1 commit; latest: abc123def456'));
    assert.doesNotMatch(item.body, /Blocking|prohibit/);
    const successor = { groupId: 'work-2', items: [] };  // a Continue successor counts no outcome of its own
    assert.equal(syncReviewGateItem(successor, { review_overrides: { continued: continued(1), landed: landed(1) } }), true);
    assert.ok(successor.items[0].body.startsWith('The author explicitly continued once over critical findings; latest: '));
    assert.ok(successor.items[0].body.endsWith('\nLanded over it: 1 commit; latest: abc123def456'));
    const alone = { groupId: 'work-3', items: [] };
    assert.equal(syncReviewGateItem(alone, { review_overrides: { landed: landed(2) } }), true);
    assert.equal(alone.items[0].body, 'Landed over a FAIL or no verdict: 2 commits; latest: abc123def456');
    const { prior } = installDom();
    try {
        renderLiveCardMeta(record);
    } finally { restoreDom(prior); }
    assert.ok(record.metaEl.innerHTML.includes(reviewGateLine({ ...decided, landed: landed(1) })));
    assert.doesNotMatch(record.metaEl.innerHTML, /CRITICAL/, 'the compact line never carries the long text');
    assert.ok(!record.metaEl.innerHTML.includes(WHY), "nor the author's whole reason");
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
            /Review: FAIL or no verdict 12× \(latest: critical findings\)/);
        assert.equal(messages.children.length, before, 'no row joins the feed');
        assert.equal(walkCard(messages, 'child-1'), null, "the work's card holds it, no card is minted for the actor");
        const toggle = () => card.querySelector('[data-live-disclosure]').listeners.get('click')[0]({ detail: 0 });
        if (card.dataset.expanded !== '1') toggle();
        await new Promise((resolve) => setTimeout(resolve, 0));  // the Details read answers an older count
        assert.match(card.querySelector('[data-live-meta]').innerHTML, /FAIL or no verdict 12×/);
        assert.equal(card.querySelector('[data-live-timeline]').children
            .filter((node) => node.dataset.liveLineKey === `review-gate-${id}`).length, 1);
        handlers.get('chat')({ task_id: id, chat_id: 1, role: 'system', system_type: 'task_summary',
            content: 'Finished.', task_terminal_status: 'completed', ts: '2026-11-11T10:05:00Z' });
        assert.equal(card.dataset.finished, '1');
        assert.match(card.querySelector('[data-live-meta]').innerHTML, /FAIL or no verdict 12×/);
    } finally { instance?.destroy(); restoreDom(prior); }
});

test('a page load shows the field from its history row, before Details and without a detail read', async () => {
    // The history endpoint carries the work's fact on its summary row (gateway/history.py); a work without one shows
    // nothing. Opening the card is not needed, and no card fetches its detail for it.
    const rows = ['cold-work', 'plain-work'].flatMap((id) => [
        { task_id: id, chat_id: 1, role: 'system', is_progress: true, content: 'Changed my body', ts: '2026-11-11T09:00:00Z' },
        { task_id: id, chat_id: 1, role: 'system', system_type: 'task_summary', content: 'Finished.',
            task_terminal_status: 'completed', ts: '2026-11-11T09:10:00Z',
            ...(id === 'cold-work' ? { review_overrides: { ...fact(3), continued: continued(1), landed: landed(1) } } : {}) },
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
        assert.ok(meta('cold-work').includes(reviewGateLine({ ...fact(3), continued: continued(1), landed: landed(1) })));
        assert.match(meta('cold-work'), /· landed over it: 1 commit/);
        assert.doesNotMatch(meta('plain-work'), /Review:/, 'a row without the fact shows no field');
        assert.deepEqual(reads, [], 'no per-card detail read');
    } finally { instance?.destroy(); restoreDom(prior); }
});
