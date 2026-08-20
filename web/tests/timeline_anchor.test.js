// Behavioural characterization of the visible-timeline anchor owner, exercised
// where the code now lives. No jsdom: the anchor pair only reads geometry,
// classes, datasets and containment, so a hand-rolled element model reproduces
// every branch (viewport selection, live-card sub-anchoring, and each restore
// fallback) deterministically.

import assert from 'node:assert/strict';
import test from 'node:test';

import { createTimelineAnchors } from '../modules/chat_timeline_anchor.js';

function makeEl({ sel = [], ts = null, clientMessageId = '', liveLineKey = '', taskId = '', top = 0, height = 50, children = [] } = {}) {
    const el = {
        sel,
        children,
        parentElement: null,
        isConnected: true,
        top,
        height,
        dataset: {},
        classList: { contains: (name) => el.sel.includes(`.${name}`) },
        getBoundingClientRect: () => ({ top: el.top, bottom: el.top + el.height, width: 100, height: el.height }),
        getClientRects: () => (el.height > 0 ? [el.getBoundingClientRect()] : []),
        matches: (selector) => selector.split(',').some((one) => el.sel.includes(one.trim())),
        descendants() {
            return el.children.flatMap((child) => [child, ...child.descendants()]);
        },
        querySelectorAll: (selector) => el.descendants().filter((child) => child.matches(selector)),
        contains: (node) => node === el || el.descendants().includes(node),
        closest(selector) {
            let cursor = el;
            while (cursor) {
                if (cursor.matches(selector)) return cursor;
                cursor = cursor.parentElement;
            }
            return null;
        },
    };
    if (ts != null) el.dataset.ts = String(ts);
    if (clientMessageId) el.dataset.clientMessageId = clientMessageId;
    if (liveLineKey) el.dataset.liveLineKey = liveLineKey;
    if (taskId) el.dataset.taskId = taskId;
    for (const child of children) child.parentElement = el;
    return el;
}

function makeMessages({ children = [], scrollTop = 0, scrollHeight = 1000, clientHeight = 500, top = 0 } = {}) {
    const div = {
        children,
        scrollTop,
        scrollHeight,
        clientHeight,
        getBoundingClientRect: () => ({ top, bottom: top + clientHeight }),
        contains: (node) => div.children.some((child) => child === node || child.contains(node)),
    };
    return div;
}

function anchorsFor(messagesDiv, liveCardRecords = new Map()) {
    return createTimelineAnchors({ messagesDiv, liveCardRecords });
}

// ───────────────────────────── isNearBottom ─────────────────────────────

test('near-bottom uses the shared 160px default and honours an explicit threshold', () => {
    const messagesDiv = makeMessages({ scrollHeight: 1000, scrollTop: 900, clientHeight: 100 });
    const { isNearBottom } = anchorsFor(messagesDiv);
    assert.equal(isNearBottom(), true);           // remaining 0

    messagesDiv.scrollTop = 750;                  // remaining 150 <= 160
    assert.equal(isNearBottom(), true);
    messagesDiv.scrollTop = 700;                  // remaining 200 > 160
    assert.equal(isNearBottom(), false);
    assert.equal(isNearBottom(250), true);        // caller-supplied slack still applies
});

test('each instance reads its own transcript element', () => {
    const main = makeMessages({ scrollHeight: 1000, scrollTop: 900, clientHeight: 100 });
    const panel = makeMessages({ scrollHeight: 1000, scrollTop: 0, clientHeight: 100 });
    assert.equal(anchorsFor(main).isNearBottom(), true);
    assert.equal(anchorsFor(panel).isNearBottom(), false);
});

// ────────────────────────────── capture ──────────────────────────────

test('capture picks the first visible node and skips the typing bubble and load-older control', () => {
    const scrolledOff = makeEl({ top: -200, height: 100, ts: 1 });
    const loadOlder = makeEl({ sel: ['.chat-load-older'], top: 10, height: 30 });
    const typing = makeEl({ sel: ['.typing-bubble'], top: 10, height: 30 });
    const first = makeEl({ top: 10, height: 50, ts: 5, clientMessageId: 'c1' });
    const second = makeEl({ top: 70, height: 50, ts: 6 });
    const messagesDiv = makeMessages({ children: [scrolledOff, loadOlder, typing, first, second] });

    const anchor = anchorsFor(messagesDiv).captureVisibleTimelineAnchor();
    assert.equal(anchor.topNode, first);
    assert.equal(anchor.node, first);
    assert.equal(anchor.ts, '5');
    assert.equal(anchor.ordinal, 0);
    assert.equal(anchor.offset, 10);
    assert.equal(anchor.clientMessageId, 'c1');
    assert.deepEqual(anchor.cardChain, []);
});

test('capture excludes the named node and everything inside it', () => {
    const inner = makeEl({ top: 10, height: 20 });
    const excluded = makeEl({ top: 10, height: 50, ts: 5, children: [inner] });
    const next = makeEl({ top: 70, height: 50, ts: 6 });
    const messagesDiv = makeMessages({ children: [excluded, inner, next] });

    const anchor = anchorsFor(messagesDiv).captureVisibleTimelineAnchor(excluded);
    assert.equal(anchor.topNode, next);
});

test('capture returns null when nothing overlaps the viewport', () => {
    const above = makeEl({ top: -300, height: 100 });
    const below = makeEl({ top: 900, height: 100 });
    const messagesDiv = makeMessages({ children: [above, below], clientHeight: 500 });
    assert.equal(anchorsFor(messagesDiv).captureVisibleTimelineAnchor(), null);
});

test('a live card anchors on the visible child row, not the card root far above', () => {
    const line = makeEl({ sel: ['.chat-live-line'], liveLineKey: 'row-7', top: 40, height: 20 });
    const card = makeEl({ sel: ['.chat-live-card'], taskId: 't-1', ts: 9, top: -400, height: 600, children: [line] });
    const messagesDiv = makeMessages({ children: [card] });

    const anchor = anchorsFor(messagesDiv).captureVisibleTimelineAnchor();
    assert.equal(anchor.topNode, card);
    assert.equal(anchor.node, line, 'the reader sees the row, so the row is the anchor');
    assert.equal(anchor.lineKey, 'row-7');
    assert.equal(anchor.offset, 40);
    assert.equal(anchor.topOffset, -400, 'the card root offset is kept as the fallback');
    assert.deepEqual(anchor.cardChain.map((entry) => entry.taskId), ['t-1']);
});

test('a live-card role element is recorded so the restore can find it again', () => {
    const meta = makeEl({ sel: ['[data-live-meta]'], top: 30, height: 20 });
    const card = makeEl({ sel: ['.chat-live-card'], taskId: 't-2', top: -100, height: 300, children: [meta] });
    const messagesDiv = makeMessages({ children: [card] });

    const anchor = anchorsFor(messagesDiv).captureVisibleTimelineAnchor();
    assert.equal(anchor.node, meta);
    assert.equal(anchor.anchorRole, '[data-live-meta]');
    assert.equal(anchor.lineKey, '');
});

// ────────────────────────────── restore ──────────────────────────────

test('restore scrolls the captured node back to its captured offset', () => {
    const node = makeEl({ top: 100, height: 50, ts: 5 });
    const messagesDiv = makeMessages({ children: [node], scrollTop: 300 });
    const { captureVisibleTimelineAnchor, restoreVisibleTimelineAnchor } = anchorsFor(messagesDiv);

    const anchor = captureVisibleTimelineAnchor();
    assert.equal(anchor.offset, 100);
    node.top = 260;                                     // content inserted above pushed it down
    assert.equal(restoreVisibleTimelineAnchor(anchor), true);
    assert.equal(messagesDiv.scrollTop, 460);           // 300 + (260 - 100)
});

test('restore of a missing anchor is a no-op', () => {
    const messagesDiv = makeMessages({ children: [], scrollTop: 42 });
    assert.equal(anchorsFor(messagesDiv).restoreVisibleTimelineAnchor(null), false);
    assert.equal(messagesDiv.scrollTop, 42);
});

test('a recycled live card is re-found through its task record, not its stale node', () => {
    const staleLine = makeEl({ sel: ['.chat-live-line'], liveLineKey: 'row-3', top: 40, height: 20 });
    const staleCard = makeEl({ sel: ['.chat-live-card'], taskId: 't-9', top: -100, height: 300, children: [staleLine] });
    const messagesDiv = makeMessages({ children: [staleCard], scrollTop: 200 });
    const liveCardRecords = new Map();
    const { captureVisibleTimelineAnchor, restoreVisibleTimelineAnchor } = anchorsFor(messagesDiv, liveCardRecords);

    const anchor = captureVisibleTimelineAnchor();
    assert.equal(anchor.node, staleLine);

    // The card was rebuilt: the captured nodes are detached, a fresh root with
    // the same task id took their place, and only the record knows about it.
    const freshLine = makeEl({ sel: ['.chat-live-line'], liveLineKey: 'row-3', top: 90, height: 20 });
    const freshCard = makeEl({ sel: ['.chat-live-card'], taskId: 't-9', top: 50, height: 300, children: [freshLine] });
    staleCard.isConnected = false;
    staleLine.isConnected = false;
    messagesDiv.children = [freshCard];
    liveCardRecords.set('t-9', { root: freshCard });

    assert.equal(restoreVisibleTimelineAnchor(anchor), true);
    assert.equal(messagesDiv.scrollTop, 250);           // 200 + (90 - 40), the rebuilt row
});

test('a rebuilt plain bubble is re-found by client message id', () => {
    const original = makeEl({ top: 100, height: 50, ts: 5, clientMessageId: 'c-42' });
    const messagesDiv = makeMessages({ children: [original], scrollTop: 0 });
    const { captureVisibleTimelineAnchor, restoreVisibleTimelineAnchor } = anchorsFor(messagesDiv);

    const anchor = captureVisibleTimelineAnchor();
    const replacement = makeEl({ top: 140, height: 50, ts: 5, clientMessageId: 'c-42' });
    original.isConnected = false;
    messagesDiv.children = [replacement];

    assert.equal(restoreVisibleTimelineAnchor(anchor), true);
    assert.equal(messagesDiv.scrollTop, 40);            // 0 + (140 - 100)
});

test('without a client id the ordinal picks the right duplicate timestamp', () => {
    const first = makeEl({ top: -200, height: 50, ts: 7 });   // scrolled above the viewport
    const second = makeEl({ top: 160, height: 50, ts: 7 });
    const messagesDiv = makeMessages({ children: [first, second], scrollTop: 0 });
    const { captureVisibleTimelineAnchor, restoreVisibleTimelineAnchor } = anchorsFor(messagesDiv);

    const anchor = captureVisibleTimelineAnchor();
    assert.equal(anchor.topNode, second);
    assert.equal(anchor.ordinal, 1, 'the second row with ts=7');

    const rebuiltFirst = makeEl({ top: 100, height: 50, ts: 7 });
    const rebuiltSecond = makeEl({ top: 200, height: 50, ts: 7 });
    first.isConnected = false;
    second.isConnected = false;
    messagesDiv.children = [rebuiltFirst, rebuiltSecond];

    assert.equal(restoreVisibleTimelineAnchor(anchor), true);
    assert.equal(messagesDiv.scrollTop, 40);            // 0 + (200 - 160)
});

test('restore reports failure when the anchored content is gone entirely', () => {
    const node = makeEl({ top: 100, height: 50, ts: 5 });
    const messagesDiv = makeMessages({ children: [node], scrollTop: 12 });
    const { captureVisibleTimelineAnchor, restoreVisibleTimelineAnchor } = anchorsFor(messagesDiv);

    const anchor = captureVisibleTimelineAnchor();
    node.isConnected = false;
    messagesDiv.children = [];

    assert.equal(restoreVisibleTimelineAnchor(anchor), false);
    assert.equal(messagesDiv.scrollTop, 12, 'a failed restore must not move the transcript');
});
