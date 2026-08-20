import assert from 'node:assert/strict';
import test from 'node:test';

import { insertTimelineNode } from '../modules/chat_render_batch.js';
import { normalizeLogTs } from '../modules/log_events.js';
import { rawTimestampEpoch } from '../modules/utils.js';

function makeNode(id, ts = null, { height = 20, top = 100 } = {}) {
    return {
        id,
        dataset: ts == null ? {} : { ts: String(ts) },
        parentNode: null,
        height,
        getBoundingClientRect: () => ({ top }),
    };
}

function makeTimeline(nodes = []) {
    const timeline = {
        children: [],
        scrollTop: 0,
        clientHeight: 80,
        get scrollHeight() {
            return this.children.reduce((total, node) => total + (node.height || 0), 0);
        },
        getBoundingClientRect: () => ({ top: 0 }),
        insertBefore(node, before) {
            const oldIndex = this.children.indexOf(node);
            if (oldIndex !== -1) this.children.splice(oldIndex, 1);
            const index = this.children.indexOf(before);
            this.children.splice(index === -1 ? this.children.length : index, 0, node);
            node.parentNode = this;
        },
        appendChild(node) {
            const oldIndex = this.children.indexOf(node);
            if (oldIndex !== -1) this.children.splice(oldIndex, 1);
            this.children.push(node);
            node.parentNode = this;
        },
    };
    nodes.forEach((node) => timeline.appendChild(node));
    return timeline;
}

function ids(timeline) {
    return timeline.children.map((node) => node.id);
}

test('history sync inserts t1 and t2 before an existing t3', () => {
    const typing = makeNode('typing');
    const timeline = makeTimeline([makeNode('t3', 3), typing]);
    insertTimelineNode(timeline, makeNode('t1', 1), typing);
    insertTimelineNode(timeline, makeNode('t2', 2), typing);
    assert.deepEqual(ids(timeline), ['t1', 't2', 't3', 'typing']);
});

test('equal timestamps preserve arrival order', () => {
    const typing = makeNode('typing');
    const timeline = makeTimeline([typing]);
    insertTimelineNode(timeline, makeNode('first', 1), typing);
    insertTimelineNode(timeline, makeNode('second', 1), typing);
    assert.deepEqual(ids(timeline), ['first', 'second', 'typing']);
});

test('an existing t3 card whose anchor becomes t1 moves before t2', () => {
    const typing = makeNode('typing');
    const t2 = makeNode('t2', 2);
    const card = makeNode('card', 3);
    const timeline = makeTimeline([t2, card, typing]);
    card.dataset.ts = '1';
    insertTimelineNode(timeline, card, typing);
    assert.deepEqual(ids(timeline), ['card', 't2', 'typing']);
});

test('typing remains last and timestamp-free nodes keep append behavior', () => {
    const typing = makeNode('typing');
    const timeline = makeTimeline([makeNode('dated', 1), typing]);
    insertTimelineNode(timeline, makeNode('undated'), typing);
    assert.deepEqual(ids(timeline), ['dated', 'undated', 'typing']);
});

test('insertion above a scrolled-up viewport compensates scrollTop', () => {
    const typing = makeNode('typing', null, { height: 0 });
    const timeline = makeTimeline([makeNode('t3', 3, { height: 30, top: -10 }), typing]);
    timeline.scrollTop = 100;
    const beforeHeight = timeline.scrollHeight;
    const result = insertTimelineNode(
        timeline,
        makeNode('t2', 2, { height: 24 }),
        typing,
    );
    assert.equal(result.insertedAboveViewport, true);
    assert.equal(timeline.scrollHeight - beforeHeight, 24);
    assert.equal(timeline.scrollTop, 124);
});

test('near-bottom insertion keeps normal bottom stickiness', () => {
    const typing = makeNode('typing', null, { height: 0 });
    const timeline = makeTimeline([makeNode('t3', 3, { height: 30 }), typing]);
    insertTimelineNode(timeline, makeNode('t1', 1, { height: 20 }), typing, { stickToBottom: true });
    assert.equal(timeline.scrollTop, timeline.scrollHeight);
});

test('raw source timestamps become numeric epoch values', () => {
    assert.equal(rawTimestampEpoch('2026-07-18T12:00:00Z'), Date.parse('2026-07-18T12:00:00Z'));
    assert.equal(rawTimestampEpoch(1234), 1234);
    assert.equal(Number.isNaN(rawTimestampEpoch('not-a-timestamp')), true);
});

test('log timestamps include the date outside today', () => {
    const now = new Date('2026-07-18T12:00:00Z');
    const today = normalizeLogTs('2026-07-18T12:00:00Z', now);
    const older = normalizeLogTs('2025-07-18T10:00:00Z', now);
    assert.doesNotMatch(today, /2026/);
    assert.match(older, /2025/);
});
