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

test('one history status stays in persistent chrome for gaps, failure and approximate restoration', () => {
    const doc = { byId: new Map(), createElement: tag => new ElementStub(tag, doc) };
    const messages = new ElementStub('div', doc), chrome = new ElementStub('div', doc);
    messages.isConnected = chrome.isConnected = true;
    const controls = createHistoryControls(messages, chrome);
    const snapshot = { initialized: true, canOlder: true };
    controls.render(snapshot, { gaps: true }, true);
    const note = chrome.querySelector('.chat-history-status');
    assert.match(note.textContent, /Shown messages may have gaps.*could not be restored exactly/);
    assert.equal(messages.querySelector('.chat-load-older').querySelector('.chat-load-older-note'), null);
    controls.render({ ...snapshot, error: new Error('read failed') }, {}, true);
    assert.equal(chrome.querySelector('.chat-history-status'), note);
    assert.match(note.textContent, /could not be loaded.*could not be restored exactly/);
    controls.render(snapshot, { complete: true }, false);
    assert.equal(chrome.children.length, 0);
    assert.equal(messages.querySelector('.chat-load-older').querySelector('.chat-load-older-note'), note);
    assert.equal(note.textContent, 'Beginning of saved history');
});

test('a status moved into persistent chrome leaves no empty history control padding', () => {
    const doc = { byId: new Map(), createElement: tag => new ElementStub(tag, doc) };
    const messages = new ElementStub('div', doc), chrome = new ElementStub('div', doc);
    messages.isConnected = chrome.isConnected = true;
    const controls = createHistoryControls(messages, chrome);
    controls.render({ initialized: true, canOlder: false, canNewer: false }, { gaps: true });
    assert.equal(controls.olderButton.hidden, true);
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

test('the empty-Main greeting and the reconnect notice are chrome: a read over them still shows its loading and failure', () => {
    const doc = { byId: new Map(), createElement: (tag) => new ElementStub(tag, doc) };
    const messages = new ElementStub('div', doc);
    messages.isConnected = true;
    const controls = createHistoryControls(messages);
    const node = (className) => { const element = doc.createElement('div'); element.className = className; return element; };
    messages.appendChild(node('chat-bubble assistant typing-bubble'));
    messages.appendChild(node('chat-empty-welcome'));
    const notice = node('chat-bubble system');
    notice.dataset.ephemeral = '1';
    messages.appendChild(notice);
    assert.equal(feedIsEmpty(messages), true);
    assert.equal(controls.beginRecent(), true);
    controls.render({ initialized: true });
    assert.equal(messages.querySelector('.chat-load-older').querySelector('.chat-load-older-note').textContent,
        'Loading saved history…');
    controls.endRecent(new Error('offline'));
    assert.equal(controls.recentFailed(), true);
    controls.endRecent();
    messages.appendChild(node('chat-bubble assistant'));
    assert.equal(feedIsEmpty(messages), false);
    assert.equal(controls.beginRecent(), false, 'a painted transcript gets no loading chrome');
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
    assert.equal((chatSource.match(/scheduleHistorySync\(\);/g) || []).length, 3);
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

// A small layout model: a child's top follows its parent's top, the parent's own
// scrollTop when the parent scrolls, and its offset in the parent's content.
function layoutNode({ classes = [], data = {}, y = 0, height = 0, scroll = null, parent = null } = {}) {
    const node = { classNames: new Set(classes), attrs: { ...data }, y, height, children: [], parentElement: null, isConnected: true };
    let top = 0;
    Object.defineProperty(node, 'scrollTop', { get: () => top,
        set: value => { top = scroll === null ? 0 : Math.max(0, Math.min(scroll, value)); } });
    Object.defineProperty(node, 'clientHeight', { get: () => node.height });
    Object.defineProperty(node, 'scrollHeight', { get: () => node.height + (scroll || 0) });
    node.dataset = new Proxy({}, { get: (_, key) => node.attrs[`data-${String(key).replace(/[A-Z]/g, c => `-${c.toLowerCase()}`)}`] });
    node.classList = { contains: value => node.classNames.has(value) };
    const one = selector => selector.startsWith('.') ? node.classNames.has(selector.slice(1))
        : selector.startsWith('[') ? selector.slice(1, -1).split('=')[0] in node.attrs : false;
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
