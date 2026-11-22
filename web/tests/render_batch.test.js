import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';

import {
    createHistoryControls,
    createHistoryResyncScheduler,
    createLiveCardBound,
    createTimelineAnchors,
    feedIsEmpty,
} from '../modules/chat_render_batch.js';
import { ElementStub } from './chat_dom_fixture.js';

const chatSource = readFileSync(new URL('../modules/chat.js', import.meta.url), 'utf8');

test('hidden hydration defers the document notice; a completed hidden window leaves no stale notice', () => {
    const doc = { byId: new Map(), createElement: tag => new ElementStub(tag, doc) };
    const messages = new ElementStub('div', doc);
    messages.isConnected = true;
    let visible = false;
    const notices = [];
    const controls = createHistoryControls(messages, null, {
        isVisible: () => visible, onGapNotice: text => notices.push(text),
    });
    const snapshot = { initialized: true, canOlder: true };
    visible = true;
    controls.render({ initialized: false }, { gaps: true });
    controls.showNotice();
    assert.equal(notices.length, 0, 'unknown initial coverage is not a read gap');
    visible = false;
    controls.render(snapshot, { gaps: true });
    assert.equal(notices.length, 0);
    assert.match(messages.querySelector('.chat-load-older-note').textContent, /gaps/);
    controls.render(snapshot, { complete: true });
    visible = true;
    controls.showNotice();
    assert.equal(notices.length, 0, 'resolved coverage cannot leave a pending gap');
    visible = false;
    controls.render(snapshot, { gaps: true });
    visible = true;
    controls.showNotice();
    controls.showNotice();
    assert.equal(notices.length, 1);
    controls.destroy();
});

test('gaps stay at the feed edge with a once-per-document notice while failure and approximate restoration use persistent chrome', () => {
    const doc = { byId: new Map(), createElement: tag => new ElementStub(tag, doc) };
    const messages = new ElementStub('div', doc), chrome = new ElementStub('div', doc);
    messages.isConnected = chrome.isConnected = true;
    const notices = [];
    const controls = createHistoryControls(messages, chrome, { onGapNotice: (text) => notices.push(text) });
    const snapshot = { initialized: true, canOlder: true };
    controls.render(snapshot, { gaps: true }, false);
    assert.equal(chrome.querySelector('.chat-history-status'), null);
    const feedNote = messages.querySelector('.chat-load-older').querySelector('.chat-load-older-note');
    assert.equal(feedNote.textContent, 'Some saved history is not loaded. Shown messages may have gaps.');
    assert.deepEqual(notices, ['Some saved history is not loaded. Shown messages may have gaps.']);
    const otherMessages = new ElementStub('div', doc), otherChrome = new ElementStub('div', doc);
    otherMessages.isConnected = otherChrome.isConnected = true;
    const otherControls = createHistoryControls(otherMessages, otherChrome, { onGapNotice: (text) => notices.push(text) });
    otherControls.render(snapshot, { gaps: true }, false);
    assert.equal(notices.length, 1, 'gap notice fires once per document across room switches and reopens');
    controls.render(snapshot, { gaps: true }, true);
    const note = chrome.querySelector('.chat-history-status');
    assert.equal(note.textContent, 'Saved position could not be restored exactly.');
    const gap = messages.querySelector('.chat-history-gap');
    assert.equal(gap.hidden, false);
    assert.equal(gap.textContent, 'Some saved history is not loaded. Shown messages may have gaps.');
    controls.render({ ...snapshot, error: new Error('read failed') }, { gaps: true }, true);
    assert.equal(gap.hidden, false, 'load failure must not erase the reading-edge gap explanation');
    controls.render({ ...snapshot, error: new Error('read failed') }, {}, true);
    assert.equal(chrome.querySelector('.chat-history-status'), note);
    assert.match(note.textContent, /could not be loaded.*could not be restored exactly/);
    controls.render(snapshot, { complete: true }, false);
    assert.equal(gap.hidden, true);
    assert.equal(chrome.children.length, 0);
    assert.equal(messages.querySelector('.chat-load-older').querySelector('.chat-load-older-note'), note);
    assert.equal(note.textContent, 'Beginning of saved history');
});

test('a room tells its history gap at the edge it concerns, never as a notice over its newest rows (U021)', () => {
    const doc = { byId: new Map(), createElement: tag => new ElementStub(tag, doc) };
    const messages = new ElementStub('div', doc);
    messages.isConnected = true;
    const announced = [];
    const prior = globalThis.document;
    // Any toast the default path raised would land in this document's body.
    globalThis.document = { ...doc, getElementById: () => null, body: { appendChild: (node) => announced.push(node) } };
    try {
        const controls = createHistoryControls(messages, null, { isVisible: () => true });
        controls.render({ initialized: true, canOlder: true }, { gaps: true });
        controls.showNotice();
        assert.deepEqual(announced, [], 'the room raises no notice of its own');
        assert.equal(messages.children[0].classList.contains('chat-load-older'), true, 'the fact heads the loaded window');
        assert.match(messages.querySelector('.chat-load-older-note').textContent, /may have gaps/);
        controls.destroy();
    } finally { globalThis.document = prior; }
});

test('a status moved into persistent chrome leaves no empty history control padding while gaps stay at the feed edge', () => {
    const doc = { byId: new Map(), createElement: tag => new ElementStub(tag, doc) };
    const messages = new ElementStub('div', doc), chrome = new ElementStub('div', doc);
    messages.isConnected = chrome.isConnected = true;
    const controls = createHistoryControls(messages, chrome, { onGapNotice() {} });
    controls.render({ initialized: true, canOlder: false, canNewer: false }, { gaps: true });
    assert.equal(controls.olderButton.hidden, true);
    assert.equal(chrome.querySelector('.chat-history-status'), null);
    assert.equal(messages.querySelector('.chat-load-older')?.hidden, false);
    assert.equal(messages.querySelector('.chat-load-older')?.querySelector('.chat-load-older-note')?.hidden, false);
    controls.render({ initialized: true, canOlder: false, canNewer: false }, {}, true);
    assert.equal(chrome.querySelector('.chat-history-status')?.hidden, false);
    assert.equal(messages.querySelector('.chat-load-older')?.hidden, true);
});

test('history chrome shares one control and derives completeness from physical coverage', () => {
    const doc = { byId: new Map(), createElement: (tag) => new ElementStub(tag, doc) };
    const messages = new ElementStub('div', doc);
    messages.isConnected = true;
    const controls = createHistoryControls(messages);
    assert.equal('newerButton' in controls, false);
    const snapshot = { initialized: true, canOlder: true, canNewer: true,
        olderExhausted: false, loading: '', error: null };
    assert.equal(controls.render(snapshot).complete, false);
    assert.equal(controls.olderButton.textContent, 'Load more history');
    assert.deepEqual(messages.children.map((node) => node.className), ['chat-load-older']);
    assert.equal(messages.querySelector('.chat-load-newer'), null);
    const exhausted = { ...snapshot, canOlder: false, canNewer: false, olderExhausted: true };
    assert.equal(controls.render(exhausted, { complete: false, gaps: true }).complete, false,
        'EOF cannot certify physical coverage');
    assert.equal(controls.render(exhausted, { complete: true, gaps: false }).complete, true);
    assert.deepEqual(messages.children.map((node) => node.className), ['chat-load-older']);
    assert.equal(messages.querySelector('.chat-load-older')
        .querySelector('.chat-load-older-note').textContent, 'Beginning of saved history');
    assert.equal(controls.olderButton.hidden, true);
});

test('at its older end a grown room offers the way on, a chain a navigation began does not', () => {
    const doc = { byId: new Map(), createElement: (tag) => new ElementStub(tag, doc) };
    const messages = new ElementStub('div', doc);
    messages.isConnected = true;
    const controls = createHistoryControls(messages);
    const exhausted = { initialized: true, canOlder: false, canNewer: false, olderExhausted: true, loading: '', error: null };
    controls.render(exhausted, { horizonGap: true });
    assert.equal(controls.olderButton.hidden, false, 'a grown room re-anchors to the present from here');
    controls.render({ ...exhausted, away: true }, { horizonGap: true });
    assert.equal(controls.olderButton.hidden, true, 'a page around a task never sends Load more to the present');
});

test('the empty-Main greeting and the reconnect notice are chrome: a read over them still shows its loading and failure', () => {
    const doc = { byId: new Map(), createElement: (tag) => new ElementStub(tag, doc) };
    const messages = new ElementStub('div', doc);
    messages.isConnected = true;
    const chrome = new ElementStub('div', doc);
    chrome.isConnected = true;
    const controls = createHistoryControls(messages, chrome);
    const node = (className) => { const element = doc.createElement('div'); element.className = className; return element; };
    messages.appendChild(node('chat-bubble assistant typing-bubble'));
    messages.appendChild(node('chat-empty-welcome'));
    const notice = node('chat-bubble system');
    notice.dataset.ephemeral = '1';
    messages.appendChild(notice);
    assert.equal(feedIsEmpty(messages), true);
    assert.equal(controls.beginRecent(), true);
    controls.render({ initialized: false }, { gaps: true });
    assert.equal(chrome.children.length, 0, 'unanswered first read is loading, not a known gap');
    assert.equal(controls.olderButton.textContent, 'Loading saved history…');
    assert.equal(messages.querySelector('.chat-load-older').querySelector('.chat-load-older-note').hidden, true);
    controls.endRecent(new Error('offline'));
    assert.equal(controls.recentFailed(), true);
    controls.endRecent();
    messages.appendChild(node('chat-bubble assistant'));
    assert.equal(feedIsEmpty(messages), false);
    assert.equal(controls.beginRecent(), false, 'a painted transcript gets no loading chrome');
});

test('a migrating store is waiting for ↓, never a failed read, also while its automatic retry is out', () => {
    const doc = { byId: new Map(), createElement: (tag) => new ElementStub(tag, doc) };
    const messages = new ElementStub('div', doc);
    messages.isConnected = true;
    const controls = createHistoryControls(messages);
    const migrating = Object.assign(new Error('importing'), { status: 503, body: { reason_code: 'migrating' } });
    const snapshot = { initialized: true, canOlder: true };
    const facts = (state = snapshot) => { const { waiting, failed } = controls.render(state); return { waiting, failed }; };
    controls.endRecent(migrating);
    assert.equal(controls.recentFailed(), false);
    assert.deepEqual(facts(), { waiting: true, failed: false });
    assert.equal(controls.beginRecent(), true);
    assert.deepEqual(facts(), { waiting: true, failed: false }, 'the read that asks again still waits');
    controls.endRecent(new Error('offline'));
    assert.equal(controls.recentFailed(), true);
    assert.deepEqual(facts(), { waiting: false, failed: true }, 'an ordinary failure is one');
    controls.endRecent();
    assert.deepEqual(facts({ ...snapshot, error: migrating, retryDirection: 'latest' }), { waiting: true, failed: false },
        '↓\'s own read refused by the upgrade waits too');
    assert.deepEqual(facts({ ...snapshot, error: new Error('offline'), retryDirection: 'latest' }), { waiting: false, failed: true });
    controls.endRecent(new Error('cut short'), true);  // a partial read's note heads the edge
    assert.deepEqual(facts({ ...snapshot, error: migrating, retryDirection: 'latest' }), { waiting: true, failed: false },
        '↓\'s own read refused by the upgrade waits, whichever note heads the edge');
    controls.endRecent(new Error('offline'));
    assert.deepEqual(facts({ ...snapshot, error: migrating, retryDirection: 'latest' }), { waiting: false, failed: true },
        'an earlier refusal never hides the room\'s later ordinary failure');
});

// ─────────────── sticky hydration / replay contracts ──────────────────────

test('sticky single-flight never swallows the post-completion resync', () => {
    // [GPT#12 + Fable#1] scheduleHistorySync (700ms debounce after task
    // completion) must do a REAL fetch — a lost task_done is healed only by
    // refetching — so it reads through the pager (what arrived since the newest
    // read, else the newest page; chat_history_arrivals.test.js), never the
    // sticky awaitInitialHydration shortcut.
    const fn = chatSource.slice(
        chatSource.indexOf('function scheduleHistorySync('),
        chatSource.indexOf('function applyLiveCardState('),
    );
    assert.match(fn, /run: \(\) => historyPager\.read\(\)/);
    assert.doesNotMatch(fn, /awaitInitialHydration/);
    // The reconnect branch of the open handler reads the newest page whole (frames sent while away are lost),
    // through the pager. chat_history_integration.test.js exercises queued reconnects and the actual re-fetch.
    assert.match(chatSource, /isReconnect \? historyPager\.read\(\{ fromReconnect: true, whole: true,/);
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
    assert.match(replaySection, /withStableViewport\(\(\) => \{/);
    assert.match(replaySection, /learnSubagentLineage\(msg\)/);
    assert.doesNotMatch(replaySection, /await /);
    // Both finished-transition paths share settleLiveCard; the task-bound
    // review lifecycle keeps its own trigger. The replay decision stays ONLY
    // behind the scheduler's gate, so sharing cleanup cannot mute either path.
    assert.equal((chatSource.match(/settleLiveCard\(record, wasFinished\);/g) || []).length, 2);
    assert.match(chatSource, /if \(!wasFinished && blockVisible\(record\)\) scheduleHistorySync\(\);/);
    // The third occurrence is the scheduler re-arming when a run settles with the bound
    // still armed, which is how a run that only JOINED an older in-flight fetch (and
    // spent its timer on a window fetched before the arm) keeps the deadline alive.
    // It is gated on !destroyed: a joined sync that settles after teardown must not
    // install a fresh timer on a dead instance (the disposer invariant).
    // The fourth is a live terminal receipt landing on a card that already settled (an
    // evolution cleanup a later campaign Stop ran): the card's stored line changed after
    // its terminal, and a row replayed from history never asks.
    assert.equal((chatSource.match(/scheduleHistorySync\(\);/g) || []).length, 4);
    assert.match(
        chatSource,
        /if \(!destroyed && !leaving && lastHistorySyncSucceeded && liveCardBound\.isArmed\(\)\) scheduleHistorySync\(\);/,
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

test('adopted live line bookmark serializes its row and restores the cold line with a different DOM key', () => {
    const box = (top, bottom) => ({ top, bottom, left: 0, right: 600, width: 600, height: bottom - top });
    const makeNode = (bounds, classes = []) => {
        const node = { bounds, dataset: {}, isConnected: true, parentElement: null };
        node.classList = { contains: value => classes.includes(value) };
        node.getBoundingClientRect = () => node.bounds;
        node.getClientRects = () => [node.bounds];
        node.matches = selector => classes.some(value => selector === `.${value}`);
        node.contains = candidate => {
            for (let current = candidate; current; current = current.parentElement) if (current === node) return true;
            return false;
        };
        node.closest = selector => {
            for (let current = node; current; current = current.parentElement) {
                if (selector === '.chat-live-card' && current.classList.contains('chat-live-card')) return current;
            }
            return null;
        };
        node.querySelectorAll = selector => selector.includes('.chat-live-line') && node.line ? [node.line] : [];
        return node;
    };
    const messages = makeNode(box(0, 400));
    messages.scrollTop = 200;
    const card = makeNode(box(-100, 500), ['chat-live-card']);
    card.dataset.taskId = 'owner'; card.parentElement = messages;
    const line = makeNode(box(20, 100), ['chat-live-line']);
    line.dataset.liveLineKey = 'line-random'; line.dataset.expanded = '1';
    line.parentElement = card; card.line = line; messages.children = [card];
    messages.contains = candidate => candidate === card || card.contains(candidate);
    const records = new Map([['owner', { root: card, items: [{
        lineKey: 'line-random', historyId: 'progress:41',
    }] }]]);
    const anchors = createTimelineAnchors({ messagesDiv: messages, liveCardRecords: records });
    const saved = anchors.serializeTimelineAnchor();
    assert.equal(saved.lineKey, 'line-random');
    assert.equal(saved.lineHistoryId, 'progress:41');
    assert.equal(saved.lineExpanded, true);
    assert.equal('node' in saved, false);
    records.get('owner').items[0] = { lineKey: 'line-random',
        dedupeKey: 'subagent-lifecycle:child' };
    const liveLifecycle = anchors.serializeTimelineAnchor();
    assert.equal(liveLifecycle.lineLifecycleKey, 'subagent-lifecycle:child');
    assert.equal(liveLifecycle.lineHistoryId, '');

    card.isConnected = line.isConnected = false;
    const coldCard = makeNode(box(-20, 600), ['chat-live-card']);
    coldCard.dataset.taskId = 'owner'; coldCard.parentElement = messages;
    const coldLine = makeNode(box(120, 200), ['chat-live-line']);
    coldLine.dataset.liveLineKey = 'history-progress-41';
    coldLine.parentElement = coldCard; coldCard.line = coldLine; messages.children = [coldCard];
    messages.contains = candidate => candidate === coldCard || coldCard.contains(candidate);
    records.set('owner', { root: coldCard, groupId: 'owner', items: [{
        lineKey: coldLine.dataset.liveLineKey, historyId: 'progress:41',
    }] });
    assert.equal(anchors.restoreVisibleTimelineAnchor(saved, { exact: true }), true);
    assert.equal(messages.scrollTop, 300);
});

test('a card crossing the top with nothing anchorable inside keeps the reader on what follows it', () => {
    // A wait-only block above the viewport (no title, no actions, no timeline
    // line) used to anchor on its own top; when a wait update shrank the block,
    // the messages the reader was on moved up. The reader's view of what
    // FOLLOWS the card is the anchor there.
    const box = (top, bottom) => ({ top, bottom, left: 0, right: 600, width: 600, height: bottom - top });
    const makeNode = (name, bounds, classes = [], selectors = []) => {
        const node = { name, dataset: {}, isConnected: true, parentElement: null, bounds,
            classNames: new Set(classes), selectors: new Set(selectors) };
        node.classList = { contains: (value) => node.classNames.has(value) };
        node.getBoundingClientRect = () => node.bounds;
        node.getClientRects = () => [node.bounds];
        node.matches = (selector) => node.selectors.has(selector);
        node.contains = (candidate) => {
            for (let current = candidate; current; current = current.parentElement) if (current === node) return true;
            return false;
        };
        node.closest = (selector) => {
            for (let current = node; current; current = current.parentElement) {
                if (selector === '.chat-live-card' && current.classNames?.has('chat-live-card')) return current;
            }
            return null;
        };
        node.querySelectorAll = () => [];
        return node;
    };
    const messages = makeNode('messages', box(0, 900));
    messages.scrollTop = 300;
    const card = makeNode('card', box(-244, 124), ['chat-live-card']);
    card.dataset.taskId = 'wait-task';
    card.parentElement = messages;
    const summary = makeNode('summary', box(-243, -200), [], ['[data-live-summary-button]']);
    summary.parentElement = card;
    card.querySelectorAll = (selector) => (selector.includes('[data-live-summary-button]') ? [summary] : []);
    const bubble = makeNode('bubble', box(124, 300), ['chat-bubble']);
    bubble.dataset.ts = '2026-09-06T21:02:00Z';
    bubble.parentElement = messages;
    messages.children = [card, bubble];
    messages.contains = (candidate) => candidate === card || candidate === bubble || card.contains(candidate);

    const anchors = createTimelineAnchors({ messagesDiv: messages, liveCardRecords: new Map([['wait-task', { root: card }]]) });
    const anchor = anchors.captureVisibleTimelineAnchor();
    assert.equal(anchor.node, bubble, 'the following message is the anchor, not the card top');
    // The wait update shrank the card by 40 px: everything below moved up.
    card.bounds = box(-244, 84);
    bubble.bounds = box(84, 260);
    assert.equal(anchors.restoreVisibleTimelineAnchor(anchor), true);
    assert.equal(messages.scrollTop, 260, 'the reader stays on the same message');
});

test('a restore that finds its node in place writes nothing to the feed', () => {
    // Even a write of the same scrollTop stops a scroll WebKit animates (End, PageDown).
    let scrollTop = 500, writes = 0;
    const bubble = { isConnected: true, bounds: { top: 40, width: 600, height: 80 },
        getBoundingClientRect() { return this.bounds; }, getClientRects() { return [this.bounds]; } };
    const messages = { getBoundingClientRect: () => ({ top: 0 }), contains: node => node === bubble,
        get scrollTop() { return scrollTop; }, set scrollTop(value) { writes += 1; scrollTop = value; } };
    const { restoreVisibleTimelineAnchor } = createTimelineAnchors({ messagesDiv: messages, liveCardRecords: new Map() });
    assert.equal(restoreVisibleTimelineAnchor({ node: bubble, offset: 40 }), true);
    assert.deepEqual({ writes, scrollTop }, { writes: 0, scrollTop: 500 });
    bubble.bounds = { top: 70, width: 600, height: 80 }; // a card above grew 30 px
    assert.equal(restoreVisibleTimelineAnchor({ node: bubble, offset: 40 }), true);
    assert.deepEqual({ writes, scrollTop }, { writes: 1, scrollTop: 530 }, 'a moved node is put back');
});

test('scaled drawer restoration uses scroll pixels and ignores subpixel rounding noise', () => {
    let scale = .96, top = 3458, writes = 0;
    const feed = layoutNode({ height: 780, scroll: 6000 });
    feed.offsetHeight = 780;
    const row = layoutNode({ classes: ['chat-bubble'], data: {'data-history-id':'row'}, y:3401, height:150, parent:feed });
    // Model WebKit's integer scroll offset while CSS scales its entire room.
    const box = (y,h) => ({top:y*scale,bottom:(y+h)*scale,height:h*scale,width:390*scale});
    feed.getBoundingClientRect = () => box(0,780);
    row.getBoundingClientRect = () => box(3401-top,150);
    const messages = Object.create(feed);
    Object.defineProperty(messages,'scrollTop',{get:()=>top,set:v=>{writes++;top=Math.floor(v);}});
    messages.contains = n => n === row;
    const anchors=createTimelineAnchors({messagesDiv:messages,liveCardRecords:new Map()});
    const saved=anchors.captureVisibleTimelineAnchor();
    assert.ok(Math.abs(saved.offset+57)<.001);
    top=0;
    anchors.restoreVisibleTimelineAnchor(saved);
    assert.equal(top,3458);
    for (let i=0;i<30;i++) {scale=.96+i*.04/29; anchors.restoreVisibleTimelineAnchor(saved);}
    assert.equal(top,3458);
    assert.equal(writes,1);
});

// A small layout model: a child's top follows its parent's top, the parent's own
// scrollTop when the parent scrolls, and its offset in the parent's content.
function layoutNode({ classes = [], data = {}, y = 0, height = 0, scroll = null, parent = null, tag = '' } = {}) {
    const node = { classNames: new Set(classes), attrs: { ...data }, y, height, children: [], parentElement: null, isConnected: true, tag };
    let top = 0;
    Object.defineProperty(node, 'scrollTop', { get: () => top,
        set: value => { top = scroll === null ? 0 : Math.max(0, Math.min(scroll, value)); } });
    Object.defineProperty(node, 'clientHeight', { get: () => node.height });
    Object.defineProperty(node, 'scrollHeight', { get: () => node.height + (scroll || 0) });
    node.dataset = new Proxy({}, { get: (_, key) => node.attrs[`data-${String(key).replace(/[A-Z]/g, c => `-${c.toLowerCase()}`)}`] });
    node.classList = { contains: value => node.classNames.has(value) };
    const one = selector => selector.startsWith('.') ? node.classNames.has(selector.slice(1))
        : selector.startsWith('[') ? selector.slice(1, -1).split('=')[0] in node.attrs : selector === node.tag;
    node.matches = selector => selector.split(',').some(part => one(part.trim()));
    node.getBoundingClientRect = () => {
        const parent = node.parentElement;
        const at = parent ? parent.getBoundingClientRect().top - parent.scrollTop + node.y : node.y;
        return { top: at, bottom: at + node.height, left: 0, right: 600, width: 600, height: node.height };
    };
    node.getClientRects = () => [node.getBoundingClientRect()];
    const descendants = () => node.children.flatMap(child => [child, ...child.querySelectorAll('*')]);
    node.querySelectorAll = selector => selector === '*' ? descendants() : descendants().filter(child => child.matches(selector));
    node.querySelector = selector => selector.startsWith(':scope > ')
        ? node.children.find(child => child.matches(selector.slice(9))) || null : node.querySelectorAll(selector)[0] || null;
    node.contains = candidate => {
        for (let current = candidate; current; current = current.parentElement) if (current === node) return true;
        return false;
    };
    node.closest = selector => {
        for (let current = node; current; current = current.parentElement) if (current.matches(selector)) return current;
        return null;
    };
    node.append = (...children) => { for (const child of children) { child.parentElement = node; node.children.push(child); } return node; };
    if (parent) parent.append(node);
    return node;
}

// One Project card: a bounded timeline holding `before` lines and the read line,
// whose fetched full output is itself bounded; one more line follows it.
function readingCard(feed, { before = 1, bodyMax = 900, full = true, key = 'line-warm', rowHeight = 60, timelineMax = 2000 } = {}) {
    const card = layoutNode({ classes: ['chat-live-card'], data: { 'data-task-id': 'reader' }, y: 0, height: 900, parent: feed });
    layoutNode({ data: { 'data-live-summary-button': '' }, y: 0, height: 40, parent: card });
    const timeline = layoutNode({ data: { 'data-live-timeline': '' }, y: 40, height: 420, scroll: timelineMax, parent: card });
    const items = [];
    let y = 0;
    for (let index = 0; index < before; index += 1) {
        layoutNode({ classes: ['chat-live-line'], data: { 'data-live-line-key': `${key}-before-${index}` }, y, height: rowHeight, parent: timeline });
        items.push({ lineKey: `${key}-before-${index}`, historyId: `progress:${index}` });
        y += rowHeight + 8;
    }
    const line = layoutNode({ classes: ['chat-live-line'], y, height: 450, parent: timeline,
        data: { 'data-live-line-key': key, 'data-expanded': '1' } });
    layoutNode({ y: 0, height: 30, parent: line });
    const body = layoutNode({ classes: ['chat-live-line-body', ...(full ? ['chat-live-line-body-full'] : [])],
        y: 30, height: 420, scroll: full ? bodyMax : null, parent: line });
    const next = layoutNode({ classes: ['chat-live-line'], data: { 'data-live-line-key': `${key}-next` }, y: y + 458, height: 60, parent: timeline });
    items.push({ lineKey: key, historyId: 'progress:7', truncated: true, fullRef: 'child' }, { lineKey: `${key}-next`, historyId: 'progress:8' });
    return { card, timeline, line, body, next, record: { root: card, groupId: 'reader', timelineEl: timeline, items } };
}

test('a bookmark inside a bounded full output keeps its line, timeline offset and own scroll by identity', () => {
    const feed = layoutNode({ height: 800, scroll: 5000 });
    const records = new Map();
    const anchors = createTimelineAnchors({ messagesDiv: feed, liveCardRecords: records });
    const warm = readingCard(feed);
    records.set('reader', warm.record);
    warm.timeline.scrollTop = 24; warm.body.scrollTop = 360;
    feed.scrollTop = 150; // the read line's head is above the top edge; its output crosses it
    assert.ok(warm.body.getBoundingClientRect().top < 0 && warm.body.getBoundingClientRect().bottom > 0);
    assert.ok(warm.next.getBoundingClientRect().top >= 0, 'a later line is visible below the top edge');
    feed.scrollTop = 0; // the card's own chrome is first at the top edge; the output is fully in view
    assert.equal(anchors.captureVisibleTimelineAnchor().node, warm.line, 'a scrolled output on screen is the place being read');
    warm.body.scrollTop = 0;
    assert.equal(anchors.captureVisibleTimelineAnchor().node, warm.card, 'an unscrolled output below the edge is not: the first visible node anchors');
    warm.body.scrollTop = 360; feed.scrollTop = 150;
    const live = anchors.captureVisibleTimelineAnchor();
    assert.equal(live.node, warm.line, 'the top edge is inside the output: the reader is there, not at the next line');
    warm.body.scrollTop = 200; warm.timeline.scrollTop = 30;
    assert.equal(anchors.restoreVisibleTimelineAnchor(live), true);
    assert.deepEqual([warm.body.scrollTop, warm.timeline.scrollTop], [200, 30], 'live restores never rewrite the reader\'s own box scrolling');
    warm.body.scrollTop = 360; warm.timeline.scrollTop = 24;
    const saved = anchors.serializeTimelineAnchor();
    assert.equal(saved.lineHistoryId, 'progress:7');
    assert.equal(saved.innerTop, 360);
    assert.equal(saved.timelineOffset, 68 - 24);
    assert.deepEqual(JSON.parse(JSON.stringify(saved)), saved, 'the bookmark stays plain data');
    assert.equal(saved.nested, undefined, 'no positional DOM path');
    feed.children.length = 0;

    // Two more lines above the read one, new keys and new markup: identity, not position.
    for (const [bodyMax, full, exact] of [[900, true, true], [100, true, false], [900, false, false]]) {
        const cold = readingCard(feed, { before: 3, bodyMax, full, key: `line-cold-${bodyMax}-${full}` });
        records.set('reader', cold.record);
        assert.equal(anchors.restoreVisibleTimelineAnchor(saved, { exact: true }), exact, { bodyMax, full });
        const within = cold.line.getBoundingClientRect().top - cold.timeline.getBoundingClientRect().top;
        assert.equal(within, saved.timelineOffset, 'the timeline puts the line back at its offset there');
        assert.equal(cold.line.getBoundingClientRect().top - feed.getBoundingClientRect().top, saved.offset);
        if (full) assert.equal(cold.body.scrollTop, Math.min(360, bodyMax));
        assert.equal(anchors.restoreVisibleTimelineAnchor(saved, { cardOnly: true }), true,
            'a shorter or unfetched output still restores its line; the caller discloses the approximation');
        feed.children.length = 0;
    }

    // An earlier row reflowed and the timeline cannot scroll far enough (or at
    // all): at its limit the line still takes its exact feed offset. Only the
    // reader's own output box must reach its scroll.
    for (const [rowHeight, timelineMax, bodyMax, exact] of [[20, 0, 900, true], [100, 0, 900, true], [100, 10, 900, true], [100, 0, 100, false]]) {
        const cold = readingCard(feed, { rowHeight, timelineMax, bodyMax, key: `line-reflow-${rowHeight}-${timelineMax}-${bodyMax}` });
        records.set('reader', cold.record);
        assert.equal(anchors.restoreVisibleTimelineAnchor(saved, { exact: true }), exact, { rowHeight, timelineMax, bodyMax });
        assert.equal(cold.timeline.scrollTop, Math.min(timelineMax, Math.max(0, rowHeight + 8 - saved.timelineOffset)));
        assert.equal(cold.line.getBoundingClientRect().top - feed.getBoundingClientRect().top, saved.offset);
        feed.children.length = 0;
    }
});

test('a full output overlapping the feed and its timeline at different places is not visible: it cannot take a visible Review anchor', () => {
    const feed = layoutNode({ height: 800, scroll: 5000 });
    const records = new Map();
    const anchors = createTimelineAnchors({ messagesDiv: feed, liveCardRecords: records });
    const card = readingCard(feed, { before: 4 });
    records.set('reader', card.record);
    // The card's Review detail follows its timeline; the reader scrolled inside it.
    const detail = layoutNode({ data: { 'data-review-attempt-detail': 'attempt-1' }, y: 480, height: 300, scroll: 400, parent: card.card });
    detail.scrollTop = 50;
    // The timeline has left the top edge. Its full output still overlaps the feed,
    // but only below the timeline, which clips it there.
    feed.scrollTop = 470;
    const body = card.body.getBoundingClientRect(), timeline = card.timeline.getBoundingClientRect();
    const top = feed.getBoundingClientRect().top;
    assert.ok(body.top < top && body.bottom > top, 'the output crosses the feed top');
    assert.ok(body.top < timeline.bottom && timeline.bottom <= top, 'and its timeline, but only above the feed');
    const captured = anchors.captureVisibleTimelineAnchor();
    assert.equal(captured.node, detail, 'the Review detail being read anchors');
    assert.equal(anchors.serializeTimelineAnchor().reviewValue, 'attempt-1');
    // Where the feed and its timeline overlap on the output, the output is read.
    feed.scrollTop = 400;
    assert.equal(anchors.captureVisibleTimelineAnchor().node, card.line);
    feed.children.length = 0;
});

test('header chrome followed directly by an expanded line anchors that line, so a reopen expands it again', () => {
    const feed = layoutNode({ height: 800, scroll: 5000 });
    const records = new Map();
    const anchors = createTimelineAnchors({ messagesDiv: feed, liveCardRecords: records });
    const card = readingCard(feed, { before: 0 });
    records.set('reader', card.record);
    // The card header is first at the top edge; its unscrolled output fills the view below.
    assert.equal(card.body.scrollTop, 0);
    const saved = anchors.serializeTimelineAnchor();
    assert.equal(saved.lineKey, 'line-warm', 'not the header alone');
    assert.equal(saved.lineHistoryId, 'progress:7');
    assert.equal(saved.lineExpanded, true);
    assert.equal(saved.anchorRole, '');
    assert.equal(saved.offset, 40);
    assert.deepEqual([saved.timelineOffset, saved.innerTop], [0, 0]);
    card.line.attrs['data-expanded'] = '0';
    assert.equal(anchors.captureVisibleTimelineAnchor().node, card.card, 'a collapsed line discloses nothing: the header anchors');
    feed.children.length = 0;
});

test('saved-place readiness waits for the Review detail and an expanded line full output', () => {
    const item = { lineKey: 'line', historyId: 'progress:3', _fetchingFull: true };
    const records = new Map([['owner', { groupId: 'owner', items: [item] }]]);
    const { anchorOwnersReady } = createTimelineAnchors({ messagesDiv: {}, liveCardRecords: records });
    const line = { lineKey: 'line', lineHistoryId: 'progress:3', lineExpanded: true, cardChain: [{ taskId: 'owner' }] };
    const reviewReady = id => id !== 'owner';
    assert.equal(anchorOwnersReady(null, reviewReady), true);
    assert.equal(anchorOwnersReady(line, reviewReady), false, 'the full output is still loading');
    assert.equal(anchorOwnersReady({ ...line, lineExpanded: false }, reviewReady), true);
    item._fetchingFull = false;
    assert.equal(anchorOwnersReady(line, reviewReady), true);
    assert.equal(anchorOwnersReady({ reviewKey: 'reviewAttemptDetail', cardChain: [{ taskId: 'owner' }] }, reviewReady), false);
    assert.equal(anchorOwnersReady({ reviewKey: 'reviewAttemptDetail', cardChain: [{ taskId: 'other' }] }, reviewReady), true);
});

test('a living card straddling the top edge anchors the visible row below it and never shrinks offscreen', () => {
    const feed = layoutNode({ height: 750, scroll: 5000 });
    const anchors = createTimelineAnchors({ messagesDiv: feed, liveCardRecords: new Map() });
    const card = layoutNode({ classes: ['chat-bubble', 'chat-card-bubble'], data: { 'data-history-id': 'card:1' }, y: 1620, height: 520, parent: feed });
    const following = layoutNode({ classes: ['chat-bubble'], data: { 'data-history-id': 'row:2' }, y: 2154, height: 120, parent: feed });
    feed.scrollTop = 2000; // card is [-380..140], following is [154..274]
    const anchor = anchors.captureVisibleTimelineAnchor();
    assert.equal(anchor.node, following, 'straddling card bubble anchors on the visible message below');
    card.height = 120;
    following.y = 1754;
    assert.equal(anchors.restoreVisibleTimelineAnchor(anchor), true);
    assert.equal(feed.scrollTop, 1600);
    assert.equal(following.getBoundingClientRect().top - feed.getBoundingClientRect().top, 154, 'following row stays stationary');
    assert.equal(card.getBoundingClientRect().bottom - feed.getBoundingClientRect().top, 140, 'card bottom stays in view without oscillating');

    feed.children.length = 0;
    const loneCard = layoutNode({ classes: ['chat-bubble', 'chat-card-bubble'], data: { 'data-history-id': 'card:lone' }, y: 1620, height: 1200, parent: feed });
    feed.scrollTop = 2000;
    const loneAnchor = anchors.captureVisibleTimelineAnchor();
    loneCard.height = 120;
    assert.equal(anchors.restoreVisibleTimelineAnchor(loneAnchor), true);
    assert.equal(loneCard.getBoundingClientRect().top - feed.getBoundingClientRect().top, 0);
});

test('disappearing temporary cards at or straddling the top keep the reference row below stationary without ts collision', () => {
    const feed = layoutNode({ height: 750, scroll: 5000 });
    const records = new Map();
    const anchors = createTimelineAnchors({ messagesDiv: feed, liveCardRecords: records, band: () => ({ top: 88, bottom: 750 }) });
    const distantSameTs = layoutNode({ classes: ['chat-bubble'], data: { 'data-history-id': 'row:early', 'data-ts': '1700000000' }, y: 200, height: 100, parent: feed });
    const tempCard = layoutNode({ classes: ['chat-live-card'], data: { 'data-task-id': 'temp-1', 'data-ts': '1700000000' }, y: 2010, height: 98, parent: feed });
    const summary = layoutNode({ data: { 'data-live-summary-button': '1' }, y: 8, height: 40, parent: tempCard });
    const refRow = layoutNode({ classes: ['chat-bubble'], data: { 'data-history-id': 'row:ref', 'data-ts': '1700000030' }, y: 2122, height: 140, parent: feed });
    records.set('temp-1', { groupId: 'temp-1', root: tempCard, items: [] });
    feed.scrollTop = 2000;
    assert.equal(distantSameTs.getBoundingClientRect().top, -1800);
    assert.equal(summary.getBoundingClientRect().top, 18);

    const anchor = anchors.captureVisibleTimelineAnchor();
    tempCard.isConnected = false;
    feed.children.splice(feed.children.indexOf(tempCard), 1);
    records.delete('temp-1');
    refRow.y = 2010;
    assert.equal(anchors.restoreVisibleTimelineAnchor(anchor), true);
    assert.equal(refRow.getBoundingClientRect().top - feed.getBoundingClientRect().top, 122, 'reference row stays at +122 px');
});

test('collapsing an expanded card whose top is above the viewport preserves the visible row below it', () => {
    const feed = layoutNode({ height: 750, scroll: 5000 });
    const records = new Map();
    const anchors = createTimelineAnchors({ messagesDiv: feed, liveCardRecords: records, band: () => ({ top: 88, bottom: 750 }) });
    const card = layoutNode({ classes: ['chat-live-card'], data: { 'data-task-id': 'portal-1', 'data-expanded': '1' }, y: 1700, height: 466, parent: feed });
    const timeline = layoutNode({ data: { 'data-live-timeline': '1' }, y: 80, height: 370, parent: card });
    const line = layoutNode({ classes: ['chat-live-line'], data: { 'data-live-line-key': 'line-1' }, y: 230, height: 80, parent: timeline });
    const refRow = layoutNode({ classes: ['chat-bubble'], data: { 'data-history-id': 'row:after' }, y: 2180, height: 120, parent: feed });
    records.set('portal-1', { groupId: 'portal-1', root: card, items: [{ lineKey: 'line-1' }] });
    feed.scrollTop = 2000;

    const anchor = anchors.captureVisibleTimelineAnchor();
    assert.equal(anchor.node, line);
    card.attrs['data-expanded'] = '0';
    card.height = 162;
    line.height = 0;
    refRow.y = 1876;
    assert.equal(anchors.restoreVisibleTimelineAnchor(anchor), true);
    assert.equal(refRow.getBoundingClientRect().top - feed.getBoundingClientRect().top, 180, 'reference row moves 0 px');
});

test('an expanded live card collapsing under a floating bar keeps the row after it; with no bar its own top holds', () => {
    // The card's top is under the bar (+20 of a band from +88): the line read inside it goes when it collapses, and
    // the row below it holds the place. With no bar the same top is in the band, the card itself the place.
    for (const bar of [88, 0]) {
        const feed = layoutNode({ height: 750, scroll: 5000 });
        const records = new Map();
        const anchors = createTimelineAnchors({ messagesDiv: feed, liveCardRecords: records,
            band: () => (bar ? { top: bar, bottom: 750 } : null) });
        const card = layoutNode({ classes: ['chat-live-card'], data: { 'data-task-id': 'open-1', 'data-expanded': '1' }, y: 2020, height: 470, parent: feed });
        const line = layoutNode({ classes: ['chat-live-line'], data: { 'data-live-line-key': 'line-1' }, y: 90, height: 80, parent: card });
        const after = layoutNode({ classes: ['chat-bubble'], data: { 'data-history-id': 'row:after' }, y: 2500, height: 120, parent: feed });
        records.set('open-1', { groupId: 'open-1', root: card, items: [{ lineKey: 'line-1' }] });
        feed.scrollTop = 2000;
        const anchor = anchors.captureVisibleTimelineAnchor();
        assert.equal(anchor.node, bar ? line : card, bar ? 'under the bar the line inside is read' : 'the card heads the band');
        card.attrs['data-expanded'] = '0';
        card.height = 166;
        line.height = 0;
        after.y = 2196;
        assert.equal(anchors.restoreVisibleTimelineAnchor(anchor), true);
        if (bar) assert.equal(after.getBoundingClientRect().top, 500, 'under the bar the row after the card stays');
        else assert.equal(card.getBoundingClientRect().top, 20, 'with no bar the card keeps its own top');
    }
});

test('the row read is the first in the read band, not a row under the bar', () => {
    for (const bar of [88, 0]) {
        const feed = layoutNode({ height: 750, scroll: 5000 });
        const anchors = createTimelineAnchors({ messagesDiv: feed, liveCardRecords: new Map(),
            band: () => (bar ? { top: bar, bottom: 750 } : null) });
        const under = layoutNode({ classes: ['chat-bubble'], data: { 'data-history-id': 'row:under' }, y: 2000, height: 60, parent: feed });
        const read = layoutNode({ classes: ['chat-bubble'], data: { 'data-history-id': 'row:read' }, y: 2070, height: 100, parent: feed });
        feed.scrollTop = 2000;
        const anchor = anchors.captureVisibleTimelineAnchor();
        assert.equal(anchor.node, bar ? read : under, bar ? 'the bar covers the first row' : 'no bar: the first row is read');
        assert.equal(anchor.offset, bar ? 70 : 0, 'offsets stay feed-relative');
    }
});

test('a row whose picture still loads above the band is passed over; one whose picture starts in the band holds', () => {
    // A late picture grows when it lands: above the band's top it grows up out of view, so the row after it is read;
    // starting in the band it grows down, and its own row stays the place.
    for (const top of [40, 100]) {
        const feed = layoutNode({ height: 750, scroll: 5000 });
        const anchors = createTimelineAnchors({ messagesDiv: feed, liveCardRecords: new Map(), band: () => ({ top: 88, bottom: 750 }) });
        const photo = layoutNode({ classes: ['chat-bubble'], data: { 'data-history-id': 'row:photo' }, y: 2000 + top, height: 260, parent: feed });
        const picture = layoutNode({ tag: 'img', y: 0, height: 200, parent: photo });
        picture.complete = false;
        const next = layoutNode({ classes: ['chat-bubble'], data: { 'data-history-id': 'row:next' }, y: 2310 + top, height: 100, parent: feed });
        feed.scrollTop = 2000;
        assert.equal(anchors.captureVisibleTimelineAnchor().node, top < 88 ? next : photo,
            top < 88 ? 'a picture above the band grows out of it' : 'a picture in the band is the place read');
        picture.complete = true;
        assert.equal(anchors.captureVisibleTimelineAnchor().node, photo, 'a picture landed is no reason to pass its row');
    }
});

test('captureVisibleTimelineAnchor ignores collapsed stack rows with zero rects before and after a straddling card', () => {
    const feed = layoutNode({ y: 0, height: 844, scroll: 5000 });
    feed.offsetHeight = 844;
    const records = new Map();
    const anchors = createTimelineAnchors({ messagesDiv: feed, liveCardRecords: records });
    const makeHiddenStackRow = (id) => {
        const hidden = layoutNode({
            classes: ['chat-bubble', 'chat-relay-note'],
            data: { 'data-history-id': id, 'data-deck-tucked': '1' },
            y: 0, height: 0, parent: feed,
        });
        hidden.getBoundingClientRect = () => ({ top: 0, bottom: 0, left: 0, right: 0, width: 0, height: 0 });
        hidden.getClientRects = () => [];
        return hidden;
    };

    const earlyStackRow = makeHiddenStackRow('stack:early');
    layoutNode({
        classes: ['chat-bubble', 'chat-card-bubble'],
        data: { 'data-history-id': 'card:straddle' },
        y: 1800, height: 300, parent: feed,
    });
    const middleStackRow = makeHiddenStackRow('stack:middle');
    const visibleFollowing = layoutNode({
        classes: ['chat-bubble'],
        data: { 'data-history-id': 'row:visible' },
        y: 2120, height: 120, parent: feed,
    });
    feed.scrollTop = 2000;

    const anchor = anchors.captureVisibleTimelineAnchor();
    assert.notEqual(anchor.node, earlyStackRow);
    assert.notEqual(anchor.node, middleStackRow);
    assert.equal(anchor.node, visibleFollowing);
});

test('a shrink above that clamps scrollTop before reflow still restores the exact 1600 px place with or without CSS anchoring', async () => {
    const priorCss = globalThis.CSS;
    try {
        for (const cssSupports of [undefined, { supports: () => true }]) {
            globalThis.CSS = cssSupports;
            let maxScroll = 2200, top = 1950;
            const feed = layoutNode({ height: 400, scroll: 2200 });
            feed.offsetHeight = 400;
            feed.clientWidth = 600;
            const messages = Object.create(feed);
            Object.defineProperty(messages, 'scrollHeight', { get: () => 400 + maxScroll });
            Object.defineProperty(messages, 'scrollTop', {
                get: () => top,
                set: (v) => { top = Math.max(0, Math.min(maxScroll, v)); feed.scrollTop = top; },
            });
            messages.scrollTop = 1950;
            const row = layoutNode({ classes: ['chat-bubble'], data: { 'data-history-id': 'row:clamped' }, y: 1990, height: 120, parent: feed });
            const anchors = createTimelineAnchors({ messagesDiv: messages, liveCardRecords: new Map() });
            const anchor = anchors.captureVisibleTimelineAnchor();
            await Promise.resolve(); // settle microtasks as before a later ResizeObserver reflow
            // An element above shrinks by 350 px; the browser clamps scrollTop from 1950 to the new max 1850 before reflow.
            maxScroll = 1850;
            row.y = 1640;
            messages.scrollTop = 1850;
            assert.equal(anchors.restoreVisibleTimelineAnchor(anchor), true);
            assert.equal(messages.scrollTop, 1600, `expected 1600 when CSS=${Boolean(cssSupports)}`);
            assert.equal(row.getBoundingClientRect().top - messages.getBoundingClientRect().top, 40);
        }
    } finally {
        globalThis.CSS = priorCss;
    }
});

test('transient sending and routing notes step to the next message only when the note itself reaches the top edge', () => {
    const feed = layoutNode({ height: 750, scroll: 5000 });
    feed.offsetHeight = 750;
    const anchors = createTimelineAnchors({ messagesDiv: feed, liveCardRecords: new Map() });

    // Case 5a: long message straddling top; a 60 px sending note below the read line (+40 px) disappears.
    const longBubble = layoutNode({ classes: ['chat-bubble', 'user'], data: { 'data-history-id': 'row:long' }, y: 1800, height: 500, parent: feed });
    const body = layoutNode({ classes: ['message'], y: 0, height: 440, parent: longBubble });
    const readLine = layoutNode({ y: 240, height: 24, parent: body });
    const sendingNote = layoutNode({ classes: ['msg-pending'], data: { 'data-ingress-unconfirmed': '1' }, y: 440, height: 60, parent: longBubble });
    const afterLong = layoutNode({ classes: ['chat-bubble'], data: { 'data-history-id': 'row:after-long' }, y: 2314, height: 120, parent: feed });
    feed.scrollTop = 2000; // readLine is at +40 px; sendingNote is below it at [+240..+300]
    assert.equal(readLine.getBoundingClientRect().top, 40);

    const anchorBelow = anchors.captureVisibleTimelineAnchor();
    assert.equal(anchorBelow.node, longBubble, 'a note below the read passage does not step to the next message');
    longBubble.children.splice(longBubble.children.indexOf(sendingNote), 1);
    longBubble.height = 440;
    afterLong.y = 2254;
    assert.equal(anchors.restoreVisibleTimelineAnchor(anchorBelow), true);
    assert.equal(readLine.getBoundingClientRect().top - feed.getBoundingClientRect().top, 40, 'read line stays at +40 px');

    // Case 5b: reverse case where only .msg-routing-annotation (32 px) is visible at the top edge, next row at +34 px.
    feed.children.length = 0;
    const routedBubble = layoutNode({ classes: ['chat-bubble', 'user'], data: { 'data-history-id': 'row:routed' }, y: 1800, height: 220, parent: feed });
    layoutNode({ classes: ['message'], y: 0, height: 188, parent: routedBubble });
    const receipt = layoutNode({ classes: ['msg-routing-annotation'], y: 188, height: 32, parent: routedBubble });
    const nextRow = layoutNode({ classes: ['chat-bubble'], data: { 'data-history-id': 'row:next' }, y: 2034, height: 120, parent: feed });
    feed.scrollTop = 2000; // receipt crosses top at [-12..+20]; nextRow is at +34 px
    assert.equal(nextRow.getBoundingClientRect().top, 34);

    const anchorAtTop = anchors.captureVisibleTimelineAnchor();
    assert.equal(anchorAtTop.node, nextRow, '.msg-routing-annotation at the top edge steps to the following message');
    routedBubble.children.splice(routedBubble.children.indexOf(receipt), 1);
    routedBubble.height = 188;
    nextRow.y = 2002;
    assert.equal(anchors.restoreVisibleTimelineAnchor(anchorAtTop), true);
    assert.equal(nextRow.getBoundingClientRect().top - feed.getBoundingClientRect().top, 34, 'following row stays at +34 px');
});
