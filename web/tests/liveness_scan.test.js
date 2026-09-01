// Card-set durable-truth reconcile (stuck "Working..." pill class): the pure
// selector that hands snapshot-unconfirmed foreground cards to the existing
// observeMissingManagedTask/reconcileMissingManagedTask path, plus source pins
// proving the header badge has exactly one writer (the status reducer) after
// the replay-batch bypass was removed.
import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';

import {
    reconcileHydratedDirectActivities,
    unconfirmedForegroundCardIds,
} from '../modules/chat_activity.js';
import { createRebuildBatch } from '../modules/chat_render_batch.js';
import { REUSABLE_TASK_IDS } from '../modules/task_control_menu.js';

const chatSource = readFileSync(new URL('../modules/chat.js', import.meta.url), 'utf8');

const live = (id, extra = {}) => ({ id, finished: false, isSubagent: false, connected: true, ...extra });

test('unconfirmedForegroundCardIds: a mounted unfinished foreground card the snapshot does not list is returned', () => {
    assert.deepEqual(unconfirmedForegroundCardIds([live('orphan-root')], new Set()), ['orphan-root']);
    // Order is the card-map order; every orphan is reported, not just the first.
    assert.deepEqual(
        unconfirmedForegroundCardIds([live('a'), live('b')], new Set(['unrelated'])),
        ['a', 'b'],
    );
});

test('unconfirmedForegroundCardIds: snapshot and request-barrier live evidence block detail reconcile', () => {
    assert.deepEqual(
        unconfirmedForegroundCardIds([live('running-root'), live('orphan')], new Set(['running-root'])),
        ['orphan'],
    );
    // Map keys work as the confirmed set too (anything with .has).
    assert.deepEqual(unconfirmedForegroundCardIds([live('r')], new Map([['r', {}]])), []);

    const existing = new Map([[
        'fresh-root',
        { activityId: 'fresh-root', kind: 'managed_task', phase: 'working', startedAt: 9_000 },
    ]]);
    const stale = reconcileHydratedDirectActivities(existing, [], 1, 5_000);
    const staleConfirmed = new Set([
        ...stale.globallyActiveActivityIds, ...stale.activities.keys(),
    ]);
    assert.deepEqual(stale.departedManagedTaskIds, []);
    assert.deepEqual(unconfirmedForegroundCardIds([live('fresh-root')], staleConfirmed), []);

    const fresh = reconcileHydratedDirectActivities(existing, [], 1, 10_000);
    const freshConfirmed = new Set([
        ...fresh.globallyActiveActivityIds, ...fresh.activities.keys(),
    ]);
    assert.deepEqual(fresh.departedManagedTaskIds, ['fresh-root']);
    assert.deepEqual(unconfirmedForegroundCardIds([live('fresh-root')], freshConfirmed), ['fresh-root']);
});

test('unconfirmedForegroundCardIds: finished cards are skipped', () => {
    assert.deepEqual(unconfirmedForegroundCardIds([live('done', { finished: true })], new Set()), []);
});

test('unconfirmedForegroundCardIds: detached (unmounted) roots are skipped — they are not in the reducer scan', () => {
    assert.deepEqual(unconfirmedForegroundCardIds([live('gone', { connected: false })], new Set()), []);
    assert.deepEqual(unconfirmedForegroundCardIds([live('unknown', { connected: undefined })], new Set()), []);
});

test('unconfirmedForegroundCardIds: subagent cards are skipped (the parent owns the lineage)', () => {
    assert.deepEqual(unconfirmedForegroundCardIds([live('child', { isSubagent: true })], new Set()), []);
});

test('unconfirmedForegroundCardIds: reusable slots and the chat fallback group are skipped', () => {
    assert.ok(REUSABLE_TASK_IDS.has('bg-consciousness') && REUSABLE_TASK_IDS.has('active'));
    const cards = [live('bg-consciousness'), live('active'), live('chat'), live('real-orphan')];
    assert.deepEqual(unconfirmedForegroundCardIds(cards, new Set()), ['real-orphan']);
});

test('unconfirmedForegroundCardIds: empty, missing and malformed inputs yield nothing', () => {
    assert.deepEqual(unconfirmedForegroundCardIds([], new Set()), []);
    assert.deepEqual(unconfirmedForegroundCardIds(null, new Set()), []);
    assert.deepEqual(unconfirmedForegroundCardIds(undefined, undefined), []);
    assert.deepEqual(unconfirmedForegroundCardIds([null, {}, live('')], null), []);
    // A missing confirmed set means nothing is confirmed: the orphan is still reported.
    assert.deepEqual(unconfirmedForegroundCardIds([live('x')], null), ['x']);
});

// ───────────── source pins: reducer is the sole header-badge writer ─────────────

test('chat.js hands the card projection to the selector inside hydrateDirectActivities', () => {
    const fn = chatSource.slice(
        chatSource.indexOf('function hydrateDirectActivities('),
        chatSource.indexOf('const isKnownProjectFrame ='),
    );
    assert.match(fn, /unconfirmedForegroundCardIds\(/);
    assert.match(
        fn,
        /new Set\(\[\.\.\.globallyActiveActivityIds, \.\.\.activeDirectActivities\.keys\(\)\]\)/,
    );
    // The projection is built from the live card map, not from a DOM query.
    assert.match(fn, /Array\.from\(liveCardRecords, \(\[id, r\]\) =>/);
    assert.match(fn, /connected: r\.root\?\.isConnected/);
    // The scan precedes the retry loop so a freshly observed id is not double-read
    // (reconcileMissingManagedTask dedupes on managedTaskDetailReads anyway).
    assert.ok(fn.indexOf('unconfirmedForegroundCardIds(') < fn.indexOf('for (const taskId of missingManagedTaskIds)'));
    assert.match(fn.slice(fn.lastIndexOf('missingManagedTaskIds')), /syncChatStatus\(\);/);
});

test('the replay batch no longer bypasses the status reducer', () => {
    const fn = chatSource.slice(
        chatSource.indexOf('function finalizeRebuildBatch('),
        chatSource.indexOf('async function syncHistory('),
    );
    assert.doesNotMatch(fn, /setStatus\(/);
    assert.doesNotMatch(fn, /batch\.status/);
    assert.doesNotMatch(chatSource, /_rebuildBatch\.status/);
    assert.equal(Object.prototype.hasOwnProperty.call(createRebuildBatch(), 'status'), false);
    // setStatus has exactly three callsites: the reducer (syncChatStatus) and the
    // panel-boot 'Online' seed — the one documented exception — plus its own
    // definition. The replay-batch "Working..." write is gone.
    const calls = chatSource.match(/setStatus\(/g) || [];
    assert.equal(calls.length, 3);
    assert.match(chatSource, /const derived = deriveChatStatus\(\);\s*setStatus\(derived\.kind, derived\.text\);/);
    assert.match(chatSource, /if \(ws\.isConnected\?\.\(\)\) setStatus\('online', 'Online'\);/);
    // The reducer still runs unconditionally right after the replay dispatch, so
    // a replayed unfinished foreground card reaches the badge through it.
    const replayEnd = chatSource.indexOf('_historyReplayActive = false;', chatSource.indexOf('_historyReplayActive = true;'));
    assert.match(chatSource.slice(replayEnd, replayEnd + 200), /syncChatStatus\(\);/);
});
