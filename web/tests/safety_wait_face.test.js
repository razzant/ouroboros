// An open Safety check of a call is a quiet line under a closed live card's current action (Track09 #2863, owner A,
// 9 Nov 2026); the folded Details row keeps every check. Other host waits stay in Details only.
import assert from 'node:assert/strict';
import test from 'node:test';

import { summarizeChatLiveEvent } from '../modules/log_events.js';
import { updateLiveTimelineItem } from '../modules/chat_render_batch.js';
import { mergeHistoricalTimelineItem } from '../modules/chat_history_replay.js';
import { syncWaitFace } from '../modules/task_checkpoints.js';

function card() {
    const inserted = [];
    const doc = {
        createElement: () => {
            const node = { dataset: {}, attrs: {}, hidden: false, innerHTML: '', className: '',
                setAttribute(name, value) { this.attrs[name] = value; } };
            return node;
        },
    };
    const activityEl = { ownerDocument: doc, after: (node) => inserted.push(node) };
    return { record: { items: [], activityEl, finished: false }, inserted };
}

const frame = (episode_id, phase, elapsed_sec = 0, kind = 'safety_check_wait') => ({
    type: 'task_checkpoint', task_id: 't1', checkpoint_kind: kind, episode_id, phase, elapsed_sec,
    ts: '2026-11-09T22:00:00Z',
});

function deliver(record, event) {
    const summary = summarizeChatLiveEvent(event);
    updateLiveTimelineItem(record, summary, { ts: event.ts, rawTs: event.ts, syntheticKey: summary.dedupeKey,
        headline: summary.headline, inPlaceByKey: false });
}

const face = (record) => (record.waitFaceEl && !record.waitFaceEl.hidden ? record.waitFaceEl.innerHTML : '');

test('the face line shows while a check is open, concurrent checks are one line, and it leaves when all ended', () => {
    const { record, inserted } = card();
    deliver(record, frame('a', 'entered'));
    assert.equal(inserted.length, 1, 'one line, inserted once after the current action');
    assert.match(face(record), /Checking safety/);
    assert.match(face(record), /data-shape="still"/, 'the still mark of a wait: no animation or counter');
    assert.equal(record.waitFaceEl.attrs.role, 'status');
    deliver(record, frame('b', 'entered'));
    deliver(record, frame('a', 'ended', 1.5));
    assert.match(face(record), /Checking safety/, 'another check is still open');
    deliver(record, frame('b', 'ended', 0.4));
    assert.equal(face(record), '', 'every check ended: the face says nothing');
    assert.equal(inserted.length, 1, 'the line is reused, never a second node');
    assert.deepEqual(record.items.map((row) => row.headline), ['Safety checks ×2 · longest 1.5 s'],
        'the folded Details row keeps every check and the longest');
    // A redelivered entered row of an ended episode does not reopen it (one identity, ended once).
    deliver(record, frame('a', 'entered'));
    assert.equal(face(record), '');
});

test('a finished card never shows the line; other host waits never reach the face', () => {
    const { record, inserted } = card();
    record.finished = true;
    deliver(record, frame('a', 'entered'));
    assert.equal(inserted.length, 0);
    const other = card();
    deliver(other.record, frame('u', 'entered', 0, 'usage_lock_wait'));
    deliver(other.record, frame('c', 'entered', 0, 'control_store_wait'));
    assert.equal(other.inserted.length, 0);
    assert.equal(syncWaitFace({ items: [] }), false, 'a record without a card is a pure fold');
});

test('history replay of an open check paints the same line as live', () => {
    const { record } = card();
    const event = frame('a', 'entered');
    const row = { ...event, history_id: 'progress:1', history_position: { source: 'progress', offset: 1 } };
    mergeHistoricalTimelineItem(record, summarizeChatLiveEvent(event), row, event.ts);
    assert.match(face(record), /Checking safety/);
});
