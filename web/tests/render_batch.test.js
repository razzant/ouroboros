import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';

import {
    LOAD_OLDER_QUOTA_STEPS,
    createHistoryResyncScheduler,
    createRebuildBatch,
    loadOlderControlState,
    nextQuotaEscalation,
    orderBatchNodes,
    timelineNodeSortKey,
} from '../modules/chat_render_batch.js';

const chatSource = readFileSync(new URL('../modules/chat.js', import.meta.url), 'utf8');
// The feed/history owner carries the pinned regions since the W3 wave D move.
const historySource = readFileSync(new URL('../modules/chat_history_sync.js', import.meta.url), 'utf8');
const anchorSource = readFileSync(new URL('../modules/chat_timeline_anchor.js', import.meta.url), 'utf8');

function makeNode(id, ts = null) {
    return { id, dataset: ts == null ? {} : { ts: String(ts) }, parentNode: null };
}

// ───────────────────────── stable batch ordering ─────────────────────────

test('batch order sorts by stamped ts', () => {
    const ordered = orderBatchNodes([makeNode('c', 3), makeNode('a', 1), makeNode('b', 2)]);
    assert.deepEqual(ordered.map((n) => n.id), ['a', 'b', 'c']);
});

test('equal timestamps preserve collection (arrival) order', () => {
    // Mirrors the insertTimelineNode pin (chat_chronology.test.js:59-64):
    // equal ts = arrival order, reproduced by the stable tie-break.
    const ordered = orderBatchNodes([
        makeNode('first', 5), makeNode('second', 5), makeNode('third', 5),
    ]);
    assert.deepEqual(ordered.map((n) => n.id), ['first', 'second', 'third']);
});

test('timestamp-free nodes land at the end in arrival order', () => {
    const ordered = orderBatchNodes([
        makeNode('undatedA'), makeNode('late', 9), makeNode('undatedB'), makeNode('early', 1),
    ]);
    assert.deepEqual(ordered.map((n) => n.id), ['early', 'late', 'undatedA', 'undatedB']);
    assert.equal(timelineNodeSortKey(makeNode('u')), Infinity);
    assert.equal(timelineNodeSortKey(makeNode('d', 42)), 42);
});

// ───────────────────────── batch mount semantics ─────────────────────────

function makeFakeDom() {
    const doc = {
        createDocumentFragment() {
            return {
                _isFragment: true,
                children: [],
                appendChild(node) { this.children.push(node); },
            };
        },
    };
    const makeMessages = () => ({
        children: [],
        insertBefore(node, before) {
            const incoming = node._isFragment ? node.children : [node];
            const index = this.children.indexOf(before);
            this.children.splice(index === -1 ? this.children.length : index, 0, ...incoming);
        },
        appendChild(node) { this.insertBefore(node, null); },
    });
    return { doc, makeMessages };
}

test('mount inserts one sorted fragment before typing; typing stays last', () => {
    const { doc, makeMessages } = makeFakeDom();
    const messages = makeMessages();
    const typing = makeNode('typing');
    messages.children.push(typing);
    typing.parentNode = messages;

    const batch = createRebuildBatch(doc);
    const b = makeNode('b', 2);
    batch.collect(b);
    batch.collect(makeNode('a', 1));
    batch.collect(b);  // duplicate collect is a no-op
    batch.collect(makeNode('c', 3));
    batch.mount(messages, typing);
    assert.deepEqual(messages.children.map((n) => n.id), ['a', 'b', 'c', 'typing']);
});

test('mount without a mounted typing node appends at the end', () => {
    const { doc, makeMessages } = makeFakeDom();
    const messages = makeMessages();
    const batch = createRebuildBatch(doc);
    batch.collect(makeNode('only', 7));
    batch.mount(messages, null);
    assert.deepEqual(messages.children.map((n) => n.id), ['only']);
});

test('touch() registers each record once for the per-card finals', () => {
    const batch = createRebuildBatch();
    const record = { groupId: 't1' };
    batch.touch(record);
    batch.touch(record);
    batch.touch(null);
    assert.deepEqual([...batch.touched], [record]);
});

// ───────────────────────── Load-older escalation ─────────────────────────

test('quota escalation ladder: default window -> 400 -> caps -> exhausted', () => {
    const first = nextQuotaEscalation(null);
    assert.deepEqual(first, LOAD_OLDER_QUOTA_STEPS[0]);
    assert.equal(first.n_human, 400);
    const second = nextQuotaEscalation(first);
    assert.deepEqual(second, { n_human: 1500, n_progress: 600 });
    assert.equal(nextQuotaEscalation(second), null);
});

test('load-older control follows the SERVER window verdict', () => {
    // Complete window (or a server predating the field): no control at all —
    // a short history must never be told about phantom archives.
    assert.equal(loadOlderControlState({ complete: true, truncated_by: [] }).mode, 'hidden');
    assert.equal(loadOlderControlState(null).mode, 'hidden');
    // Quota truncation with escalation headroom: a real button.
    const btn = loadOlderControlState({ complete: false, truncated_by: ['quota'] }, null);
    assert.equal(btn.mode, 'button');
    // Quota truncation at the caps: honest boundary notice instead.
    const capped = loadOlderControlState(
        { complete: false, truncated_by: ['quota'] },
        { n_human: 1500, n_progress: 600 },
    );
    assert.equal(capped.mode, 'notice');
    // Archive-floor / lineage-cap truncation cannot be escalated away.
    const floor = loadOlderControlState(
        { complete: false, truncated_by: ['archive_floor', 'lineage_cap'] }, null,
    );
    assert.equal(floor.mode, 'notice');
    // The notice names BOTH boundaries: on-disk archives AND the lineage cap.
    assert.match(floor.label, /archive/i);
    assert.match(floor.label, /lineage/i);
});

// ─────────────── source pins: routine path & sticky boundary ───────────────

test('routine path still routes through chronological insertTimelineNode', () => {
    // [Fable#9] The batch collector only diverts while a rebuildAll batch is
    // active; the routine/live insertMessageNode path must keep flowing into
    // insertTimelineNode (the py pin becomes provable structure, not luck).
    const fn = historySource.slice(
        historySource.indexOf('function insertMessageNode('),
        historySource.indexOf('function addMessage('),
    );
    const guard = fn.indexOf('if (_rebuildBatch) {');
    const divert = fn.indexOf('_rebuildBatch.collect(node);');
    const chronological = fn.indexOf('insertTimelineNode(messagesDiv, node, typing');
    assert.ok(guard !== -1 && divert !== -1 && chronological !== -1);
    // The batch divert RETURNS before the chronological insertion, which stays
    // the one and only mounted-DOM path.
    assert.ok(guard < divert && divert < chronological);
    assert.match(fn.slice(divert, chronological), /return;/);
});

test('sticky single-flight never swallows the post-completion resync', () => {
    // [GPT#12 + Fable#1] scheduleHistorySync (700ms debounce after task
    // completion) must do a REAL fetch — a lost task_done is healed only by
    // refetching — so it calls syncHistory directly, never the sticky
    // awaitInitialHydration shortcut.
    const fn = historySource.slice(
        historySource.indexOf('function scheduleHistorySync('),
        // the Load-older control bounds the same scheduler region in the
        // feed/history owner (W3 wave D).
        historySource.indexOf('function syncLoadOlderControl('),
    );
    assert.match(fn, /syncHistory\(\{ includeUser: false \}\)/);
    assert.doesNotMatch(fn, /awaitInitialHydration/);
    // The reconnect branch of the open handler also always refetches.
    assert.match(
        historySource,
        /\? syncHistory\(\{ includeUser: !historyLoaded, fromReconnect: isReconnect \}\)/,
    );
    // Every failed sync resets the sticky promise.
    assert.ok((historySource.match(/initialHydrationPromise = null;/g) || []).length >= 3);
});

test('rebuild batch runs inside ONE outer withStableViewport with a synchronous mount', () => {
    // [GPT#14] the clearing→mount critical section must stay synchronous and
    // wrapped once; the fragment mounts before the typing bubble.
    const start = historySource.indexOf('_rebuildBatch = createRebuildBatch();');
    assert.ok(start !== -1);
    const section = historySource.slice(start, start + 900);
    assert.match(section, /withStableViewport\(\(\) => \{/);
    assert.match(section, /applySyncedMessages\(\);/);
    assert.match(section, /batch\.mount\(messagesDiv, messagesDiv\.querySelector\('\.typing-bubble'\)\);/);
    assert.match(section, /finalizeRebuildBatch\(batch\);/);
    assert.doesNotMatch(section.slice(0, section.indexOf('finalizeRebuildBatch')), /await /);
});

// ──────────── replay-time resync suppression (double-fetch fix) ────────────

function makeFakeTimers() {
    const pending = [];
    return {
        pending,
        setTimer(fn, ms) {
            const id = { fn, ms };
            pending.push(id);
            return id;
        },
        clearTimer(id) {
            const i = pending.indexOf(id);
            if (i !== -1) pending.splice(i, 1);
        },
        fire() {
            const jobs = pending.splice(0);
            for (const job of jobs) job.fn();
        },
    };
}

test('a finished transition during a history replay does NOT schedule the resync', () => {
    const timers = makeFakeTimers();
    let runs = 0;
    let replayActive = true;
    const scheduler = createHistoryResyncScheduler({
        isReplayActive: () => replayActive,
        run: () => { runs += 1; },
        setTimer: timers.setTimer,
        clearTimer: timers.clearTimer,
    });
    assert.equal(scheduler.schedule(), false);
    assert.equal(timers.pending.length, 0);
    timers.fire();
    assert.equal(runs, 0);
    // The suppression is not sticky: the same scheduler works once the replay ends.
    replayActive = false;
    assert.equal(scheduler.schedule(), true);
});

test('a LIVE finished transition (outside a replay) schedules a real 700ms resync', () => {
    const timers = makeFakeTimers();
    let runs = 0;
    const scheduler = createHistoryResyncScheduler({
        isReplayActive: () => false,
        run: () => { runs += 1; },
        setTimer: timers.setTimer,
        clearTimer: timers.clearTimer,
    });
    assert.equal(scheduler.schedule(), true);
    assert.equal(timers.pending.length, 1);
    assert.equal(timers.pending[0].ms, 700);
    // Re-scheduling debounces: the previous timer is replaced, not stacked.
    scheduler.schedule();
    assert.equal(timers.pending.length, 1);
    timers.fire();
    assert.equal(runs, 1);
    // cancel() (instance destroy) clears a pending resync.
    scheduler.schedule();
    scheduler.cancel();
    assert.equal(timers.pending.length, 0);
});

test('chat.js wires the replay flag around the replay and keeps live callsites intact', () => {
    // scheduleHistorySync delegates to the scheduler, whose replay gate reads
    // _historyReplayActive; the flag wraps the whole replay dispatch (both the
    // rebuildAll batch and the routine branch) and drops in a finally.
    assert.match(historySource, /isReplayActive: \(\) => _historyReplayActive,/);
    const flagUp = historySource.indexOf('_historyReplayActive = true;');
    assert.ok(flagUp !== -1);
    // (search from flagUp: the earlier `let … = false;` declaration also matches)
    const replaySection = historySource.slice(flagUp, historySource.indexOf('_historyReplayActive = false;', flagUp));
    assert.match(replaySection, /if \(rebuildAll\) \{/);
    assert.match(replaySection, /applySyncedMessages\(\);/);
    assert.doesNotMatch(replaySection, /await /);
    // The LIVE path is untouched: both finished-transition callsites (the
    // task_done frame in applyLiveCardStateMutation and finishLiveCardMutation)
    // still call scheduleHistorySync() unconditionally — the replay decision
    // lives ONLY behind the scheduler's gate. Both callsites moved with their
    // owners into the live-card store (W3 wave D).
    const liveCardsSource = readFileSync(new URL('../modules/chat_live_cards.js', import.meta.url), 'utf8');
    assert.equal((liveCardsSource.match(/scheduleHistorySync\(\);/g) || []).length, 2);
    assert.doesNotMatch(historySource, /_historyReplayActive[^\n]*scheduleHistorySync/);
});

test('the Load-older control is mounted ONLY while it has something to show', () => {
    // A permanently-present (even hidden) control is an extra top-level feed
    // child that breaks child-order consumers (ui-smoke chronology pattern)
    // and diverges from the pre-P4 feed layout on complete windows.
    const fn = historySource.slice(
        historySource.indexOf('function syncLoadOlderControl('),
        historySource.indexOf('async function loadOlderHistory('),
    );
    assert.match(fn, /if \(control\.mode === 'hidden'\) \{[\s\n]*loadOlderEl\.remove\(\);/);
    assert.match(fn, /if \(!loadOlderEl\.isConnected\) messagesDiv\.prepend\(loadOlderEl\);/);
    // The ONLY mount site is the on-demand one inside syncLoadOlderControl —
    // no unconditional prepend at instance construction.
    assert.equal((historySource.match(/messagesDiv\.prepend\(loadOlderEl\);/g) || []).length, 1);
});

test('the Load-older control is excluded from viewport anchoring like typing', () => {
    // [GPT#13] the anchor must land on the first visible TIMESTAMPED node.
    // The anchor pair now lives in its own owner; the behavioural counterpart of
    // this pin is timeline_anchor.test.js ("capture picks the first visible node
    // and skips the typing bubble and load-older control").
    const fn = anchorSource.slice(
        anchorSource.indexOf('function captureVisibleTimelineAnchor('),
        anchorSource.indexOf('function restoreVisibleTimelineAnchor('),
    );
    assert.match(fn, /!node\.classList\.contains\('typing-bubble'\)/);
    assert.match(fn, /!node\.classList\.contains\('chat-load-older'\)/);
});
