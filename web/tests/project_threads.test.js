import assert from 'node:assert/strict';
import test from 'node:test';

import {
    MAIN_THREAD_ID,
    applyManualOrder,
    extraThreadRows,
    isMainThreadUnread,
    isThreadUnread,
    normalizeSeenRevision,
    orderProjectRows,
    orderThreadRows,
    projectThreadRows,
    rememberSeenRevision,
    reorderIds,
    seenRevisionFor,
    threadKey,
    unreadThreadCount,
} from '../modules/project_threads.js';
import { forgetThreadTranscriptCache, threadTranscriptCacheKey } from '../modules/chat.js';

// ---------------------------------------------------------------------------
// The nested read cursor (X6) — the browser half of a BREAKING ABI migration.
// ---------------------------------------------------------------------------

test('normalizeSeenRevision maps a FLAT pre-T1 cursor onto thread #0', () => {
    // This is the whole compatibility window: every value stored before the
    // migration is flat, and thread #0 IS the chat that number described
    // (thread_chat_id(pid, 0) == project_chat_id(pid)). Getting this wrong does
    // not throw — it silently shows every project as unread forever.
    assert.deepEqual(normalizeSeenRevision({ alpha: 7 }), { alpha: { 0: 7 } });
    assert.deepEqual(normalizeSeenRevision({ alpha: { 0: 7, 3: 2 } }), { alpha: { 0: 7, 3: 2 } });
    // Mixed documents are normal DURING the window and are handled per entry.
    assert.deepEqual(
        normalizeSeenRevision({ alpha: 7, beta: { 2: 1 } }),
        { alpha: { 0: 7 }, beta: { 2: 1 } },
    );
});

test('normalizeSeenRevision refuses to invent cursors from junk', () => {
    assert.deepEqual(normalizeSeenRevision(null), {});
    assert.deepEqual(normalizeSeenRevision('nope'), {});
    assert.deepEqual(normalizeSeenRevision([1, 2]), {});
    // A negative or unparsable revision clamps to 0 (unread) rather than to a
    // number that would mark unseen output as read.
    assert.deepEqual(normalizeSeenRevision({ a: -4, b: 'x' }), { a: { 0: 0 }, b: { 0: 0 } });
    // A non-numeric thread key is dropped, not coerced to thread #0 — silently
    // acknowledging thread 0 for "main" would hide real unread activity.
    assert.deepEqual(normalizeSeenRevision({ a: { main: 5 } }), { a: {} });
});

test('seenRevisionFor / rememberSeenRevision keep the cursor monotonic per thread', () => {
    const cursor = {};
    rememberSeenRevision(cursor, 'alpha', 3, 5);
    assert.equal(seenRevisionFor(cursor, 'alpha', 3), 5);
    rememberSeenRevision(cursor, 'alpha', 3, 2);   // a stale tab must not rewind
    assert.equal(seenRevisionFor(cursor, 'alpha', 3), 5);
    assert.equal(seenRevisionFor(cursor, 'alpha', 0), 0);
    assert.equal(seenRevisionFor(cursor, 'missing', 0), 0);
});

// ---------------------------------------------------------------------------
// Unread aggregation
// ---------------------------------------------------------------------------

const project = (overrides = {}) => ({
    id: 'alpha',
    name: 'Alpha',
    lifecycle: 'active',
    chat_id: 111,
    visible_revision: 9,
    threads: [
        { id: 0, chat_id: 111, name: 'Alpha', visible_revision: 3 },
        { id: 1, chat_id: 222, name: 'Side', visible_revision: 2 },
    ],
    ...overrides,
});

test('projectThreadRows falls back to a synthetic thread #0 without a projection', () => {
    const rows = projectThreadRows({ id: 'a', name: 'A', chat_id: 5, visible_revision: 4 });
    assert.deepEqual(rows, [{ id: MAIN_THREAD_ID, chat_id: 5, name: 'A', visible_revision: 4 }]);
    assert.deepEqual(extraThreadRows(project()).map((t) => t.id), [1]);
});

test('a sibling thread never marks the project main thread read', () => {
    const cursor = normalizeSeenRevision({ alpha: { 0: 3, 1: 0 } });
    const [zero, side] = projectThreadRows(project());
    assert.equal(isThreadUnread(zero, cursor, 'alpha'), false);
    assert.equal(isThreadUnread(side, cursor, 'alpha'), true);
    // The project ROW's dot is thread #0's own; the group aggregate is the
    // `#nav-projects-count` pill, which is what `unreadThreadCount` feeds and
    // what survives the collapse. Either way the project-wide `visible_revision`
    // (9) is deliberately unused — no dot and no pill is ever computed from it.
    assert.equal(unreadThreadCount(project(), cursor), 1);
});

test('a deleting project reports no unread threads', () => {
    assert.equal(unreadThreadCount(project({ lifecycle: 'deleting' }), {}), 0);
    assert.equal(isMainThreadUnread(project({ lifecycle: 'deleting' }), {}), false);
});

test('the project ROW dot is thread #0 only — the aggregate is the pill', () => {
    // One unread SIBLING. The sibling's own row is already lit, and clicking the
    // project row opens thread #0, so an aggregate dot on the row would be a
    // second dot for the same message that the click could never clear.
    const siblingOnly = normalizeSeenRevision({ alpha: { 0: 3, 1: 0 } });
    assert.equal(isMainThreadUnread(project(), siblingOnly), false);
    assert.equal(unreadThreadCount(project(), siblingOnly), 1);   // still the pill's 1

    // Thread #0 itself unread: the row lights, and clicking it clears it.
    const mainOnly = normalizeSeenRevision({ alpha: { 0: 0, 1: 2 } });
    assert.equal(isMainThreadUnread(project(), mainOnly), true);
    assert.equal(unreadThreadCount(project(), mainOnly), 1);

    // The project-wide aggregate `visible_revision` (9) is never the row's dot:
    // with thread #0 acknowledged, a project whose aggregate has run far ahead
    // because of sibling traffic still shows no dot on its own row.
    assert.equal(isMainThreadUnread(project({ visible_revision: 99 }), siblingOnly), false);

    // Without a threads projection the row IS thread #0, so it answers for it.
    const legacy = { id: 'a', name: 'A', chat_id: 5, visible_revision: 4 };
    assert.equal(isMainThreadUnread(legacy, {}), true);
    assert.equal(isMainThreadUnread(legacy, normalizeSeenRevision({ a: 4 })), false);
});

test('an empty cursor means every thread with activity is unread', () => {
    assert.equal(unreadThreadCount(project(), {}), 2);
    // ...but a thread that has never produced owner-visible output is NOT unread.
    const fresh = project({ threads: [{ id: 0, chat_id: 111, name: 'Alpha', visible_revision: 0 }] });
    assert.equal(unreadThreadCount(fresh, {}), 0);
});

// ---------------------------------------------------------------------------
// Ordering (D3): new on top, owner's manual order wins, grouped under its project
// ---------------------------------------------------------------------------

test('orderThreadRows puts the newest thread on top by default', () => {
    const threads = [{ id: 1 }, { id: 7 }, { id: 4 }];
    assert.deepEqual(orderThreadRows(threads, []).map((t) => t.id), [7, 4, 1]);
});

test('a manual order is an explicit PREFIX, not a full sort', () => {
    const threads = [{ id: 1 }, { id: 7 }, { id: 4 }];
    // Only 1 was placed by the owner; the rest keep the default newest-first.
    assert.deepEqual(orderThreadRows(threads, ['1']).map((t) => t.id), [1, 7, 4]);
    // A stale id in the stored order is ignored rather than scrambling the list.
    assert.deepEqual(orderThreadRows(threads, ['99', '4']).map((t) => t.id), [4, 7, 1]);
});

test('orderProjectRows defaults to newest-active first and honours the manual order', () => {
    const rows = [
        { id: 'a', last_active_at: '2026-01-01T00:00:00Z' },
        { id: 'b', last_active_at: '2026-03-01T00:00:00Z' },
        { id: 'c', created_at: '2026-02-01T00:00:00Z' },
    ];
    assert.deepEqual(orderProjectRows(rows, []).map((r) => r.id), ['b', 'c', 'a']);
    assert.deepEqual(orderProjectRows(rows, ['a']).map((r) => r.id), ['a', 'b', 'c']);
});

test('applyManualOrder never drops or duplicates a row', () => {
    const rows = [{ id: 'x' }, { id: 'y' }, { id: 'z' }];
    const out = applyManualOrder(rows, ['z', 'gone', 'x'], (r) => r.id);
    assert.deepEqual(out.map((r) => r.id), ['z', 'x', 'y']);
    assert.equal(out.length, rows.length);
});

test('reorderIds produces the FULL new order for a drop above or below', () => {
    assert.deepEqual(reorderIds(['a', 'b', 'c'], 'c', 'a', false), ['c', 'a', 'b']);
    assert.deepEqual(reorderIds(['a', 'b', 'c'], 'a', 'c', true), ['b', 'c', 'a']);
    // Dropping onto itself, or onto something not in the list, is a no-op.
    assert.deepEqual(reorderIds(['a', 'b'], 'a', 'a', true), ['a', 'b']);
    assert.deepEqual(reorderIds(['a', 'b'], 'a', 'zz', true), ['a', 'b']);
});

test('threadKey separates two threads of the same project', () => {
    assert.equal(threadKey('alpha', 0), 'alpha#0');
    assert.notEqual(threadKey('alpha', 0), threadKey('alpha', 1));
});

// ---------------------------------------------------------------------------
// Destroying a thread releases its REBUILDABLE session storage
// ---------------------------------------------------------------------------

test('a destroyed thread drops its transcript cache and keeps its draft', () => {
    const store = new Map();
    const storage = {
        setItem: (k, v) => store.set(k, v),
        getItem: (k) => (store.has(k) ? store.get(k) : null),
        removeItem: (k) => store.delete(k),
    };
    storage.setItem(threadTranscriptCacheKey(7), '[{"id":1}]');
    storage.setItem('ouro_chat_draft:7', 'half a sentence');
    storage.setItem('ouro_chat_input_history:7', '["earlier"]');

    assert.equal(threadTranscriptCacheKey(7), 'ouro_chat:7');
    assert.equal(forgetThreadTranscriptCache(7, storage), true);
    assert.equal(storage.getItem('ouro_chat:7'), null, 'the paint accelerator goes');
    // The two keys holding text nobody can rebuild stay. Dropping the cache is
    // what buys them the quota they need to keep being writable.
    assert.equal(storage.getItem('ouro_chat_draft:7'), 'half a sentence');
    assert.equal(storage.getItem('ouro_chat_input_history:7'), '["earlier"]');
});

test('the MAIN chat transcript is never dropped — Main is not a thread', () => {
    const store = new Map([['ouro_chat', '[]']]);
    const storage = {
        getItem: (k) => (store.has(k) ? store.get(k) : null),
        removeItem: (k) => store.delete(k),
    };
    assert.equal(threadTranscriptCacheKey(1), 'ouro_chat');
    assert.equal(forgetThreadTranscriptCache(1, storage), false);
    assert.equal(forgetThreadTranscriptCache(0, storage), false);
    assert.equal(forgetThreadTranscriptCache(undefined, storage), false);
    assert.equal(storage.getItem('ouro_chat'), '[]');
});

test('a storage that throws is survivable — the drop is best-effort', () => {
    const hostile = { removeItem() { throw new DOMException('QuotaExceededError'); } };
    assert.equal(forgetThreadTranscriptCache(9, hostile), false);
    assert.equal(forgetThreadTranscriptCache(9, null), false);  // no sessionStorage at all
});
