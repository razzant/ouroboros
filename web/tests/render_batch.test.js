import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';

import {
    LOAD_OLDER_QUOTA_STEPS,
    createHistoryResyncScheduler,
    createLiveCardBound,
    createRebuildBatch,
    createTimelineAnchors,
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
});

// ─────────────── sticky hydration / replay contracts ──────────────────────

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

test('the live-card bound arms past the cap relative to the last rebuild', () => {
    const bound = createLiveCardBound(200);
    assert.equal(bound.isArmed(), false);
    bound.observe(200);
    assert.equal(bound.isArmed(), false, 'the bound is exceeded-by-one, not reached');
    bound.observe(201);
    assert.equal(bound.isArmed(), true);
    // The rebuild consumes the arm and the population it produced becomes the floor:
    // a window that itself holds more than the cap must not re-arm on the next card.
    bound.settle({ rebuilt: true, size: 210 });
    assert.equal(bound.isArmed(), false);
    bound.observe(211);
    assert.equal(bound.isArmed(), false);
    bound.observe(411);
    assert.equal(bound.isArmed(), true);
});

test('an arm raised while a sync is in flight is not consumed by that sync', () => {
    // The window that sync fetched predates the arm, so it cannot answer for the
    // cards that raised it: consuming the arm there would rebuild from a stale
    // window, drop the newest cards, and leave nothing armed to replay them.
    const bound = createLiveCardBound(200);
    const armedAtStart = bound.begin();
    assert.equal(armedAtStart, false);
    bound.beginReplay();
    bound.observe(201);                       // the window's own rows cross the cap
    bound.settle({ rebuilt: false, size: 201 });
    assert.equal(bound.isArmed(), true, 'the arm survives for the next sync');
    const nextArmed = bound.begin();
    assert.equal(nextArmed, true);
    bound.settle({ rebuilt: true, size: 0 });
    assert.equal(bound.isArmed(), false);
});

test('an arm raised while the fetch was in flight outlives that rebuild', () => {
    // A reconnect (or first load, or Load older) fetches a window, and live cards
    // cross the cap while it is in flight. That rebuild erases those cards from a
    // window that never contained them, so its arm is NOT answered: the next sync
    // fetches a window that does contain them.
    const bound = createLiveCardBound(200);
    assert.equal(bound.begin(), false);
    bound.observe(201);                       // live frames, still in flight
    bound.beginReplay();                      // the synchronous replay starts here
    bound.settle({ rebuilt: true, size: 0 });
    assert.equal(bound.isArmed(), true, 'the older window did not answer this arm');
    assert.equal(bound.begin(), true);
    bound.beginReplay();
    bound.settle({ rebuilt: true, size: 0 });
    assert.equal(bound.isArmed(), false, 'the fresh window did');
});

test('a rebuild consumes an arm its own replay raised, and does not loop', () => {
    // The bootstrap rebuild replays a window that itself holds more than the cap.
    // That mints the cards, so the arm it raises is answered by the very replay that
    // raised it: the new floor is that population and the next sync stays routine.
    const bound = createLiveCardBound(200);
    assert.equal(bound.begin(), false);
    bound.beginReplay();
    bound.observe(210);                       // the window's own rows mint these
    assert.equal(bound.isArmed(), true);
    bound.settle({ rebuilt: true, size: 210 });
    assert.equal(bound.isArmed(), false);
    bound.observe(211);
    assert.equal(bound.isArmed(), false, 'no rebuild storm on a window above the cap');
});

test('while a full rebuild is armed, later completions cannot push the resync out', () => {
    // The live-card bound arms the rebuild and waits for the next history sync. That
    // sync is this debounced resync, and every completion used to restart its timer:
    // completions arriving faster than 700ms starved it for as long as the burst
    // lasted, which is precisely the busy session the bound exists for.
    const timers = makeFakeTimers();
    let runs = 0;
    const scheduler = createHistoryResyncScheduler({
        isReplayActive: () => false,
        run: () => { runs += 1; },
        setTimer: timers.setTimer,
        clearTimer: timers.clearTimer,
    });
    const armed = true;
    assert.equal(scheduler.schedule(armed), true);
    const first = timers.pending[0];
    for (let i = 0; i < 50; i += 1) scheduler.schedule(armed);
    assert.equal(timers.pending.length, 1);
    assert.equal(timers.pending[0], first, 'the original deadline survived the burst');
    timers.fire();
    assert.equal(runs, 1);
    // Unarmed scheduling keeps debouncing: a quiet session still coalesces.
    scheduler.schedule();
    const second = timers.pending[0];
    scheduler.schedule();
    assert.equal(timers.pending.length, 1);
    assert.notEqual(timers.pending[0], second);
});

test('chat.js hands the armed flag to the scheduler', () => {
    assert.match(chatSource, /historyResyncScheduler\.schedule\(liveCardBound\.isArmed\(\)\);/);
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
    // plus a terminal task-bound review lifecycle call scheduleHistorySync()
    // unconditionally — the replay decision lives ONLY behind the scheduler's
    // gate.
    // Three LIVE callsites call scheduleHistorySync() unconditionally. The fourth
    // occurrence is the scheduler re-arming itself when a run settles with the bound
    // still armed, which is how a run that only JOINED an older in-flight fetch (and
    // spent its timer on a window fetched before the arm) keeps the deadline alive.
    // It is gated on !destroyed: a joined sync that settles after teardown must not
    // install a fresh timer on a dead instance (the disposer invariant).
    assert.equal((chatSource.match(/scheduleHistorySync\(\);/g) || []).length, 4);
    assert.match(
        chatSource,
        /if \(!destroyed && lastHistorySyncSucceeded && liveCardBound\.isArmed\(\)\) scheduleHistorySync\(\);/,
    );
    assert.doesNotMatch(chatSource, /_historyReplayActive[^\n]*scheduleHistorySync/);
});

test('the Load-older control is excluded from viewport anchoring like typing', () => {
    // [GPT#13] the anchor must land on the first visible TIMESTAMPED node.
    // The anchor pair lives in chat_render_batch.js (extracted verbatim from
    // chat.js at the byte ratchet); the contract is unchanged.
    const anchorSource = readFileSync(new URL('../modules/chat_render_batch.js', import.meta.url), 'utf8');
    const fn = anchorSource.slice(
        anchorSource.indexOf('function captureVisibleTimelineAnchor('),
        anchorSource.indexOf('function restoreVisibleTimelineAnchor('),
    );
    assert.match(fn, /!node\.classList\.contains\('typing-bubble'\)/);
    assert.match(fn, /!node\.classList\.contains\('chat-load-older'\)/);
});

test('a reader inside Reviews stays anchored when content grows above the attempt', () => {
    const box = (top, bottom) => ({ top, bottom, left: 0, right: 600, width: 600, height: bottom - top });
    const makeAnchorNode = (name, bounds, classes = [], selectors = []) => {
        const node = {
            name,
            dataset: {},
            isConnected: true,
            parentElement: null,
            bounds,
            classNames: new Set(classes),
            selectors: new Set(selectors),
        };
        node.classList = { contains: (value) => node.classNames.has(value) };
        node.getBoundingClientRect = () => node.bounds;
        node.getClientRects = () => [node.bounds];
        node.matches = (selector) => node.selectors.has(selector);
        node.contains = (candidate) => {
            for (let current = candidate; current; current = current.parentElement) {
                if (current === node) return true;
            }
            return false;
        };
        node.closest = (selector) => {
            for (let current = node; current; current = current.parentElement) {
                if (selector === '.chat-live-card' && current.classNames?.has('chat-live-card')) {
                    return current;
                }
            }
            return null;
        };
        node.querySelectorAll = () => [];
        return node;
    };

    const messages = makeAnchorNode('messages', box(0, 500));
    messages.scrollTop = 1000;
    const card = makeAnchorNode('card', box(-1000, 1200), ['chat-live-card']);
    card.dataset.taskId = 'review-task';
    card.parentElement = messages;
    const summary = makeAnchorNode('summary', box(-1000, -900), [], ['[data-live-summary-button]']);
    const timeline = makeAnchorNode('timeline', box(-800, -100), ['chat-live-line'], ['.chat-live-line']);
    const reviewHost = makeAnchorNode('review-host', box(-100, 900));
    const reviewSection = makeAnchorNode('review-section', box(-100, 900), [], ['[data-review-section]']);
    const review = makeAnchorNode('review-attempt', box(-100, 900), [], ['[data-review-attempt]']);
    summary.parentElement = card;
    timeline.parentElement = card;
    reviewHost.parentElement = card;
    reviewSection.parentElement = reviewHost;
    review.parentElement = reviewSection;
    const descendants = [summary, timeline, reviewSection, review];
    card.querySelectorAll = (selector) => descendants.filter(
        (candidate) => [...candidate.selectors].some((token) => selector.includes(token)),
    );
    messages.children = [card];
    messages.contains = (candidate) => candidate === card || card.contains(candidate);

    const anchors = createTimelineAnchors({
        messagesDiv: messages,
        liveCardRecords: new Map([['review-task', { root: card }]]),
    });
    const anchor = anchors.captureVisibleTimelineAnchor();
    review.bounds = box(20, 1020);
    assert.equal(anchor.node, review);
    assert.equal(anchors.restoreVisibleTimelineAnchor(anchor), true);
    assert.equal(messages.scrollTop, 1120);
});
