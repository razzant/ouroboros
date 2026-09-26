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

test('unconfirmedForegroundCardIds: only census evidence blocks detail reconcile', () => {
    assert.deepEqual(
        unconfirmedForegroundCardIds([live('running-root'), live('orphan')], new Set(['running-root'])),
        ['orphan'],
    );
    // Map keys work as the confirmed set too (anything with .has).
    assert.deepEqual(unconfirmedForegroundCardIds([live('r')], new Map([['r', {}]])), []);

    const existing = new Map([[
        'fresh-root',
        { activityId: 'fresh-root', kind: 'managed_task', phase: 'working' },
    ]]);
    // A partial census vouches for nothing it does not list, but concludes nothing.
    const partial = reconcileHydratedDirectActivities(existing, [], 1, null, false);
    assert.deepEqual(partial.departedManagedTaskIds, []);
    assert.deepEqual(
        unconfirmedForegroundCardIds([live('fresh-root')], partial.globallyActiveActivityIds),
        ['fresh-root'],
    );

    const complete = reconcileHydratedDirectActivities(existing, [], 1);
    assert.deepEqual(complete.departedManagedTaskIds, ['fresh-root']);
    assert.deepEqual(
        unconfirmedForegroundCardIds([live('fresh-root')], complete.globallyActiveActivityIds),
        ['fresh-root'],
    );

    const listed = reconcileHydratedDirectActivities(
        existing, [{ activity_id: 'fresh-root', chat_id: 1, kind: 'managed_task' }], 1,
    );
    assert.deepEqual(
        unconfirmedForegroundCardIds([live('fresh-root')], listed.globallyActiveActivityIds),
        [],
    );
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
    // A consciousness wake-up is an ordinary direct turn with its own durable
    // result, so its id is NOT a reusable slot and its card is scanned.
    assert.ok(REUSABLE_TASK_IDS.has('active') && !REUSABLE_TASK_IDS.has('bg-consciousness'));
    const cards = [live('active'), live('chat'), live('wake-1'), live('real-orphan')];
    assert.deepEqual(unconfirmedForegroundCardIds(cards, new Set()), ['wake-1', 'real-orphan']);
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
    // Census ids ONLY: the live set is a projection of the census, so unioning
    // its keys in shielded cards from durable reconcile for nothing (#866).
    assert.match(fn, /\r?\n\s+globallyActiveActivityIds,\r?\n\s+\)\) \{/);
    assert.doesNotMatch(fn, /activeDirectActivities\.keys\(\)/);
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
        chatSource.indexOf('function applyHistoryMessages('),
        chatSource.indexOf('async function syncHistory('),
    );
    assert.doesNotMatch(fn, /setStatus\(/);
    assert.doesNotMatch(fn, /batch\.status/);
    assert.doesNotMatch(chatSource, /_rebuildBatch\.status/);
    assert.doesNotMatch(chatSource, /createRebuildBatch/);
    // setStatus has exactly three callsites: the reducer (syncChatStatus) and the
    // panel-boot seed — which asks the same pure reducer, so a late panel says
    // Starting… until the host is ready (В9) — plus its own definition. The
    // replay-batch "Working..." write and the literal 'Online' seed are gone.
    const calls = chatSource.match(/setStatus\(/g) || [];
    assert.equal(calls.length, 3);
    assert.match(chatSource, /const derived = deriveChatStatus\(\);\s*setStatus\(derived\.kind, derived\.text\);/);
    assert.match(chatSource, /const seed = computeDerivedChatStatus\(\{ supervisorStarting: !hostReady \}\);\s*if \(ws\.isConnected\?\.\(\)\) setStatus\(seed\.kind, seed\.text\);/);
    assert.doesNotMatch(chatSource, /setStatus\('online', 'Online'\)/);
    // The reducer still runs unconditionally right after the replay dispatch, so
    // a replayed unfinished foreground card reaches the badge through it.
    const replayEnd = fn.lastIndexOf('_historyReplayActive = false;');
    assert.match(fn.slice(replayEnd), /syncChatStatus\(\);/);
});
