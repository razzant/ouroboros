import assert from 'node:assert/strict';
import test from 'node:test';

import {
    chatMediaMessageKey,
    durableChatMediaUrl,
    isNonTerminalMediaHistoryRow,
    rawTimestampEpoch,
} from '../modules/chat_activity.js';
import { insertTimelineNode } from '../modules/chat.js';
import { normalizeLogTs } from '../modules/log_events.js';

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

test('chronological insertion leaves viewport ownership to the chat boundary', () => {
    const typing = makeNode('typing', null, { height: 0 });
    const timeline = makeTimeline([makeNode('t3', 3, { height: 30, top: -10 }), typing]);
    timeline.scrollTop = 100;
    const result = insertTimelineNode(
        timeline,
        makeNode('t2', 2, { height: 24 }),
        typing,
    );
    assert.equal(result.before.id, 't3');
    assert.equal(timeline.scrollTop, 100);
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

test('durable chat media accepts only the closed task-artifact URL shape', () => {
    const good = `/api/tasks/task-1/artifacts/chat-media-${'a'.repeat(64)}.png`;
    assert.equal(durableChatMediaUrl(good), good);
    assert.equal(durableChatMediaUrl('/api/files/content?path=secret'), '');
    assert.equal(durableChatMediaUrl(`/api/tasks/task-1/artifacts/${'a'.repeat(64)}.png`), '');
});

test('media replay dedup key is stable across live base64 and durable history URL', () => {
    const common = { type: 'photo', task_id: 'task-photo', ts: '2026-08-21T00:00:00Z', caption: 'shot', mime: 'image/png' };
    const live = { ...common, image_base64: 'aGVsbG8=' };
    const replay = { ...common, msg_type: 'photo', download_url: `/api/tasks/t/artifacts/chat-media-${'b'.repeat(64)}.png` };
    assert.equal(chatMediaMessageKey(live), chatMediaMessageKey(replay));
    assert.notEqual(chatMediaMessageKey(live), chatMediaMessageKey({ ...live, task_id: 'parallel-task' }));
});

test('URL-less media history rows cannot finalize a live task card', () => {
    assert.equal(isNonTerminalMediaHistoryRow({ system_type: 'photo', task_id: 'running' }), true);
    assert.equal(isNonTerminalMediaHistoryRow({ system_type: 'video', task_id: 'running' }), true);
    assert.equal(isNonTerminalMediaHistoryRow({ system_type: 'task_summary', task_id: 'done' }), false);
});

test('a timeline move keeps the caret of a focused text control instead of clearing document ranges', () => {
    // Chromium mirrors a focused input's caret into the document Selection: the
    // range save/restore around a node move collapsed a typed draft's selection.
    const calls = [];
    const input = {
        tagName: 'INPUT', selectionStart: 2, selectionEnd: 8, selectionDirection: 'forward',
        setSelectionRange(start, end, direction) { calls.push(['set', start, end, direction]); this.selectionStart = start; this.selectionEnd = end; },
        focus() { calls.push(['focus']); },
    };
    const selection = {
        rangeCount: 1,
        getRangeAt: () => ({ startContainer: {}, startOffset: 0, endContainer: {}, endOffset: 0 }),
        removeAllRanges() { calls.push(['removeAllRanges']); input.selectionStart = input.selectionEnd = 0; },
        addRange() { calls.push(['addRange']); },
    };
    const typing = makeNode('typing');
    const timeline = makeTimeline([makeNode('t3', 3), typing]);
    timeline.ownerDocument = { activeElement: input, getSelection: () => selection, createRange: () => ({ setStart() {}, setEnd() {} }) };
    const moving = makeNode('t1', 1);
    moving.contains = () => false;
    moving.isConnected = true;  // mounted elsewhere in the feed: a move, not a new row
    insertTimelineNode(timeline, moving, typing);
    assert.deepEqual(ids(timeline), ['t1', 't3', 'typing']);
    assert.ok(!calls.some(([name]) => name === 'removeAllRanges'), calls);
    assert.deepEqual([input.selectionStart, input.selectionEnd], [2, 8]);
    assert.deepEqual(calls, [['set', 2, 8, 'forward']]);
});

test('rows a page moves keep the selection the page read once; each mounted move still restores it', () => {
    let reads = 0;
    const start = { isConnected: true }, end = { isConnected: true }, restored = [];
    const selection = {
        get rangeCount() { reads += 1; return 1; },
        getRangeAt: () => ({ startContainer: start, startOffset: 1, endContainer: end, endOffset: 3 }),
        removeAllRanges() {},
        addRange(range) { restored.push([range.start, range.end]); },
    };
    const typing = makeNode('typing');
    const timeline = makeTimeline([makeNode('t9', 9), typing]);
    timeline.ownerDocument = { activeElement: null, getSelection: () => selection,
        createRange: () => ({ setStart(node, offset) { this.start = [node, offset]; }, setEnd(node, offset) { this.end = [node, offset]; } }) };
    const kept = { active: null, caret: null, selection, ranges: [[start, 1, end, 3]] };
    for (const [id, ts] of [['t1', 1], ['t2', 2], ['t3', 3]]) {
        const moving = makeNode(id, ts);
        moving.isConnected = true;  // mounted elsewhere in the feed: a move
        insertTimelineNode(timeline, moving, typing, kept);
    }
    assert.deepEqual(ids(timeline), ['t1', 't2', 't3', 't9', 'typing']);
    assert.equal(reads, 0, 'no selection read per move');
    assert.equal(restored.length, 3, 'every move gives the selection back');
    assert.deepEqual(restored[2], [[start, 1], [end, 3]]);
    const alone = makeNode('t4', 4);
    alone.isConnected = true;
    insertTimelineNode(timeline, alone, typing);
    assert.equal(reads, 1, 'a move outside a page reads the selection itself');
});
