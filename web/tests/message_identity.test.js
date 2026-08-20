// Behavioural characterization of the message identity owner, exercised where
// the code now lives. The dedup contract (one durable row -> one key on both
// the live socket and history replay) and the bounded seen-key window are the
// load-bearing parts; the presentation helpers only touch a node's dataset, so
// a plain object stands in for the DOM node.

import assert from 'node:assert/strict';
import test from 'node:test';

import { createMessageIdentity } from '../modules/chat_message_identity.js';

function identity({ chatSessionId = 'session-a' } = {}) {
    const seenMessageKeys = new Set();
    const messageKeyOrder = [];
    return {
        seenMessageKeys,
        messageKeyOrder,
        ...createMessageIdentity({ chatSessionId, seenMessageKeys, messageKeyOrder }),
    };
}

// ──────────────────────────── message keys ────────────────────────────

test('a client message id is the whole key: the live echo and its replay collapse', () => {
    const { buildMessageKey } = identity();
    const live = buildMessageKey('user', 'hello', '2026-01-01T00:00:00Z', { clientMessageId: 'm-1' });
    const replay = buildMessageKey('user', 'hello (edited by the server)', '2026-01-02T00:00:00Z', {
        clientMessageId: 'm-1',
        source: 'telegram',
    });
    assert.equal(live, 'client|m-1');
    assert.equal(replay, live, 'the durable identity wins over text and timestamp drift');
});

test('a task-scoped assistant row keys on the task, not the timestamp', () => {
    const { buildMessageKey } = identity();
    const first = buildMessageKey('assistant', 'done', '2026-01-01T00:00:00Z', { taskId: 't-1' });
    const later = buildMessageKey('assistant', 'done', '2026-01-01T09:99:99Z', { taskId: 't-1' });
    assert.equal(first, 'task|assistant|||t-1|done');
    assert.equal(later, first, 'replay with a different ts must not double-insert the row');

    // Progress frames are explicitly NOT task-keyed: many arrive per task.
    const progress = buildMessageKey('assistant', 'done', '2026-01-01T00:00:00Z', {
        taskId: 't-1',
        isProgress: true,
    });
    assert.notEqual(progress, first);
});

test('a user row is never task-keyed even when it carries a task id', () => {
    const { buildMessageKey } = identity();
    const key = buildMessageKey('user', 'go', '2026-01-01T00:00:00Z', { taskId: 't-1' });
    assert.ok(!key.startsWith('task|'));
    assert.ok(key.includes('t-1'));
});

test('a row with no durable identity and no timestamp gets no key at all', () => {
    const { buildMessageKey } = identity();
    assert.equal(buildMessageKey('assistant', 'text', '', {}), '');
    assert.equal(buildMessageKey('user', 'text', null, {}), '');
});

test('sender identity is part of the key, so two web tabs do not collapse', () => {
    const { buildMessageKey } = identity();
    const mine = buildMessageKey('user', 'hi', '2026-01-01T00:00:00Z', { senderSessionId: 'aaa' });
    const theirs = buildMessageKey('user', 'hi', '2026-01-01T00:00:00Z', { senderSessionId: 'bbb' });
    assert.notEqual(mine, theirs);
});

// ─────────────────────── bounded seen-key window ───────────────────────

test('remembering a key is idempotent and empty keys are ignored', () => {
    const it = identity();
    it.rememberMessageKey('k1');
    it.rememberMessageKey('k1');
    it.rememberMessageKey('');
    assert.deepEqual(it.messageKeyOrder, ['k1']);
    assert.equal(it.seenMessageKeys.size, 1);
});

test('the window is bounded at 2000 keys and evicts oldest-first', () => {
    const it = identity();
    for (let i = 0; i < 2000; i += 1) it.rememberMessageKey(`k${i}`);
    assert.equal(it.messageKeyOrder.length, 2000);
    assert.equal(it.seenMessageKeys.has('k0'), true);

    it.rememberMessageKey('k2000');
    assert.equal(it.messageKeyOrder.length, 2000, 'the window never grows past its bound');
    assert.equal(it.seenMessageKeys.has('k0'), false, 'the oldest key left the set, not just the list');
    assert.equal(it.seenMessageKeys.has('k1'), true);
    assert.equal(it.seenMessageKeys.has('k2000'), true);
    assert.equal(it.seenMessageKeys.size, 2000);
});

test('two instances keep separate dedup windows', () => {
    const main = identity();
    const panel = identity();
    main.rememberMessageKey('shared');
    assert.equal(main.seenMessageKeys.has('shared'), true);
    assert.equal(panel.seenMessageKeys.has('shared'), false);
});

// ───────────────────────────── timestamps ─────────────────────────────

test('a node is stamped with the sortable epoch of its raw timestamp', () => {
    const { stampNodeTimestamp } = identity();
    const node = { dataset: {} };
    assert.equal(stampNodeTimestamp(node, '2026-01-01T00:00:00.000Z'), false);
    assert.equal(node.dataset.ts, String(Date.parse('2026-01-01T00:00:00.000Z')));
});

test('an unparseable or missing stamp leaves the node untouched', () => {
    const { stampNodeTimestamp } = identity();
    const node = { dataset: {} };
    assert.equal(stampNodeTimestamp(node, 'not-a-date'), false);
    assert.equal(node.dataset.ts, undefined);
    assert.equal(stampNodeTimestamp(null, '2026-01-01T00:00:00Z'), false);
});

test('anchor mode keeps a card at its EARLIEST timestamp and reports the move', () => {
    const { stampNodeTimestamp } = identity();
    const late = '2026-01-01T00:00:10.000Z';
    const early = '2026-01-01T00:00:05.000Z';
    const node = { dataset: {} };

    // First anchored stamp just records; there is nothing to move earlier than.
    assert.equal(stampNodeTimestamp(node, late, { anchor: true }), false);
    assert.equal(node.dataset.ts, String(Date.parse(late)));

    // An earlier event lowers the anchor and says so, so the card is re-sorted.
    assert.equal(stampNodeTimestamp(node, early, { anchor: true }), true);
    assert.equal(node.dataset.ts, String(Date.parse(early)));

    // A later event must not push the card down the transcript again.
    assert.equal(stampNodeTimestamp(node, late, { anchor: true }), false);
    assert.equal(node.dataset.ts, String(Date.parse(early)));
});

test('without anchor mode a later stamp overwrites', () => {
    const { stampNodeTimestamp } = identity();
    const node = { dataset: { ts: String(Date.parse('2026-01-01T00:00:10.000Z')) } };
    stampNodeTimestamp(node, '2026-01-01T00:00:05.000Z');
    assert.equal(node.dataset.ts, String(Date.parse('2026-01-01T00:00:05.000Z')));
});

test('display time is null when there is nothing to show', () => {
    const { formatMsgTime } = identity();
    assert.equal(formatMsgTime(''), null);
    assert.equal(formatMsgTime('nonsense'), null);
});

test('todays messages show a bare clock, older ones carry the date', () => {
    const { formatMsgTime } = identity();
    const now = new Date();
    now.setHours(13, 5, 0, 0);
    const today = formatMsgTime(now.toISOString());
    assert.equal(today.short, '13:05');
    assert.match(today.full, /at 13:05$/);

    const yesterday = new Date(now);
    yesterday.setDate(now.getDate() - 1);
    assert.equal(formatMsgTime(yesterday.toISOString()).short, 'Yesterday, 13:05');

    const old = new Date(now);
    old.setFullYear(now.getFullYear() - 1);
    old.setMonth(2, 4);
    assert.equal(formatMsgTime(old.toISOString()).short, 'Mar 4, 13:05');
    assert.equal(formatMsgTime(old.toISOString()).full, `Mar 4, ${old.getFullYear()} at 13:05`);
});

// ─────────────────────────── sender labels ───────────────────────────

test('this tab is You; another web tab is disambiguated by its session prefix', () => {
    const { getSenderLabel } = identity({ chatSessionId: 'abcdefgh-1234' });
    assert.equal(getSenderLabel('user'), 'You');
    assert.equal(getSenderLabel('user', false, '', { senderSessionId: 'abcdefgh-1234' }), 'You');
    assert.equal(
        getSenderLabel('user', false, '', { senderSessionId: 'zzzzzzzz-9999' }),
        'WebUI (zzzzzzzz)',
    );
});

test('a telegram origin wins over session disambiguation', () => {
    const { getSenderLabel } = identity({ chatSessionId: 'abcdefgh-1234' });
    assert.equal(
        getSenderLabel('user', false, '', { source: 'telegram', senderSessionId: 'zzzzzzzz' }),
        'Telegram',
    );
    assert.equal(
        getSenderLabel('user', false, '', { source: 'telegram', senderLabel: 'Anton' }),
        'Anton',
    );
});

test('system, progress and assistant rows get their own labels', () => {
    const { getSenderLabel } = identity();
    assert.equal(getSenderLabel('system', false, 'task_summary'), '📋 Task Summary');
    assert.equal(getSenderLabel('system', false, 'skill_review'), '📋 Skill Review');
    assert.equal(getSenderLabel('system', false, 'anything-else'), '📋 System');
    assert.equal(getSenderLabel('assistant', true), '💬 Thought');
    assert.equal(getSenderLabel('assistant'), 'Ouroboros');
});
