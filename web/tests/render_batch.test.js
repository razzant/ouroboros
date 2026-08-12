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
    // A FORK whose shared past was not fully read is a DIFFERENT boundary: those
    // rows are not "in the archive", the chain was never followed to them.
    // Reusing the archive wording would misname where the conversation is (A3b).
    const ancestry = loadOlderControlState(
        { complete: false, truncated_by: ['ancestry_depth'] }, null,
    );
    assert.equal(ancestry.mode, 'notice');
    assert.match(ancestry.label, /shared past/i);
    assert.doesNotMatch(ancestry.label, /archive/i);
    // Quota headroom still wins: that part of the window IS loadable.
    assert.equal(
        loadOlderControlState(
            { complete: false, truncated_by: ['quota', 'ancestry_depth'] }, null,
        ).mode,
        'button',
    );
    // TWO unescalatable causes at once: both boundaries get named. Stopping at
    // the first match told a forked thread about its fork chain while silently
    // dropping the on-disk archive floor that was ALSO cutting its window.
    const both = loadOlderControlState(
        { complete: false, truncated_by: ['archive_floor', 'ancestry_depth'] }, null,
    );
    assert.equal(both.mode, 'notice');
    assert.match(both.label, /shared past/i);
    assert.match(both.label, /archive/i);
    assert.match(both.label, /lineage/i);
    // ...and order of the causes cannot change what is disclosed.
    assert.equal(
        loadOlderControlState(
            { complete: false, truncated_by: ['ancestry_depth', 'archive_floor'] }, null,
        ).label,
        both.label,
    );
    // A capped quota alongside a fork gap names both too.
    const cappedFork = loadOlderControlState(
        { complete: false, truncated_by: ['quota', 'ancestry_depth'] },
        { n_human: 1500, n_progress: 600 },
    );
    assert.equal(cappedFork.mode, 'notice');
    assert.match(cappedFork.label, /shared past/i);
    assert.match(cappedFork.label, /archive/i);
    // P6: `lens_unavailable` is a THIRD boundary — the lens could not be BUILT, so
    // whether this thread has a shared past is unknown rather than known-and-cut.
    // The server sets it together with `ancestry_depth`; it must add its own
    // clause and must NOT fall through to the archive/lineage wording.
    const unavailable = loadOlderControlState(
        { complete: false, truncated_by: ['ancestry_depth', 'lens_unavailable'] }, null,
    );
    assert.equal(unavailable.mode, 'notice');
    assert.match(unavailable.label, /shared past/i);
    assert.match(unavailable.label, /could not be looked up/i);
    assert.doesNotMatch(unavailable.label, /archive/i);
    // On its own it still says the right thing rather than borrowing the archive text.
    const alone = loadOlderControlState(
        { complete: false, truncated_by: ['lens_unavailable'] }, null,
    );
    assert.match(alone.label, /could not be looked up/i);
    assert.doesNotMatch(alone.label, /archive/i);
});

// ─────────────── source pins: routine path & sticky boundary ───────────────

test('routine path still routes through chronological insertTimelineNode', () => {
    // [Fable#9] The batch collector only diverts while a rebuildAll batch is
    // active; the routine/live insertMessageNode path must keep flowing into
    // insertTimelineNode (the py pin becomes provable structure, not luck).
    const fn = chatSource.slice(
        chatSource.indexOf('function insertMessageNode('),
        chatSource.indexOf('function isBackgroundTaskId('),
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
    const fn = chatSource.slice(
        chatSource.indexOf('function scheduleHistorySync('),
        chatSource.indexOf('function applyLiveCardState('),
    );
    assert.match(fn, /syncHistory\(\{ includeUser: false \}\)/);
    assert.doesNotMatch(fn, /awaitInitialHydration/);
    // The reconnect branch of the open handler also always refetches.
    assert.match(
        chatSource,
        /\? syncHistory\(\{ includeUser: !historyLoaded, fromReconnect: isReconnect \}\)/,
    );
    // Every failed sync resets the sticky promise.
    assert.ok((chatSource.match(/initialHydrationPromise = null;/g) || []).length >= 3);
});

test('rebuild batch runs inside ONE outer withStableViewport with a synchronous mount', () => {
    // [GPT#14] the clearing→mount critical section must stay synchronous and
    // wrapped once; the fragment mounts before the typing bubble.
    const start = chatSource.indexOf('_rebuildBatch = createRebuildBatch();');
    assert.ok(start !== -1);
    const section = chatSource.slice(start, start + 900);
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
    assert.match(chatSource, /isReplayActive: \(\) => _historyReplayActive,/);
    const flagUp = chatSource.indexOf('_historyReplayActive = true;');
    assert.ok(flagUp !== -1);
    // (search from flagUp: the earlier `let … = false;` declaration also matches)
    const replaySection = chatSource.slice(flagUp, chatSource.indexOf('_historyReplayActive = false;', flagUp));
    assert.match(replaySection, /if \(rebuildAll\) \{/);
    assert.match(replaySection, /applySyncedMessages\(\);/);
    assert.doesNotMatch(replaySection, /await /);
    // The LIVE path is untouched: both finished-transition callsites (the
    // task_done frame in applyLiveCardStateMutation and finishLiveCardMutation)
    // still call scheduleHistorySync() unconditionally — the replay decision
    // lives ONLY behind the scheduler's gate.
    assert.equal((chatSource.match(/scheduleHistorySync\(\);/g) || []).length, 2);
    assert.doesNotMatch(chatSource, /_historyReplayActive[^\n]*scheduleHistorySync/);
});

test('the Load-older control is mounted ONLY while it has something to show', () => {
    // A permanently-present (even hidden) control is an extra top-level feed
    // child that breaks child-order consumers (ui-smoke chronology pattern)
    // and diverges from the pre-P4 feed layout on complete windows.
    const fn = chatSource.slice(
        chatSource.indexOf('function syncLoadOlderControl('),
        chatSource.indexOf('async function loadOlderHistory('),
    );
    assert.match(fn, /if \(control\.mode === 'hidden'\) \{[\s\n]*loadOlderEl\.remove\(\);/);
    assert.match(fn, /if \(!loadOlderEl\.isConnected\) messagesDiv\.prepend\(loadOlderEl\);/);
    // The ONLY mount site is the on-demand one inside syncLoadOlderControl —
    // no unconditional prepend at instance construction.
    assert.equal((chatSource.match(/messagesDiv\.prepend\(loadOlderEl\);/g) || []).length, 1);
});

test('the Load-older control is excluded from viewport anchoring like typing', () => {
    // [GPT#13] the anchor must land on the first visible TIMESTAMPED node.
    const fn = chatSource.slice(
        chatSource.indexOf('function captureVisibleTimelineAnchor('),
        chatSource.indexOf('function restoreVisibleTimelineAnchor('),
    );
    assert.match(fn, /!node\.classList\.contains\('typing-bubble'\)/);
    assert.match(fn, /!node\.classList\.contains\('chat-load-older'\)/);
});
