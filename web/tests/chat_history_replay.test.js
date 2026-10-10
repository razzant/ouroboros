import assert from 'node:assert/strict';
import test from 'node:test';
import { mergeHistoricalTimelineItem, compareHistoryPosition } from '../modules/chat_history_replay.js';
import { updateLiveTimelineItem } from '../modules/chat_render_batch.js';
import { delegatedActivityView, reconcileDelegatedItems } from '../modules/delegated_activity.js';

test('revisioned merge rows coexist with distinct delegated speech through live-to-history replay', () => {
    const record = { items: [] }, ts = '2026-09-28T00:00:00Z';
    const key = 'cardrow|merge-receipt:receipt-1';
    const merge = (headline, revision) => ({ headline, phase: 'result', dedupeKey: key, cardRowRevision: revision });
    const speech = (seq) => delegatedActivityView({ is_progress: true, task_id: 'task', delegated_activity: {
        v: 1, task_id: 'task', run_id: 'run-1', after_seq: seq - 1, through_seq: seq,
        source: { kind: 'run_events', read_through: seq },
        parts: [{ kind: 'message', actor: 'executor', text: 'Same words 🙂', seq }],
    } });
    const row = (id, offset) => ({ history_id: id, ts,
        history_position: { source: 'progress', offset } });
    const live = (summary) => updateLiveTimelineItem(record, summary, {
        ts, rawTs: ts, syntheticKey: summary.dedupeKey, headline: summary.headline, inPlaceByKey: true,
    });
    live(merge('PR queued', 1));
    live(speech(1));
    live(merge('PR merged', 3));
    const receipt = record.items.find(item => item.dedupeKey === key);
    for (const seq of [2, 1, 2]) {
        mergeHistoricalTimelineItem(record, speech(seq), row(`speech-${seq}`, seq), ts);
    }
    assert.equal(mergeHistoricalTimelineItem(record, merge('PR queued', 2), row('receipt-old', 50), ts), true);
    assert.equal(receipt.sourceHistoryId, 'receipt-old', 'a rejected revision still supplies a missing durable locator');
    assert.equal(receipt.cardRowRevision, 3);
    assert.equal(receipt.headline, 'PR merged');
    assert.equal(record.items.filter(item => item.activity).length, 2);
    assert.deepEqual(record.items.filter(item => item.activity).map(item => item.activity.parts[0].seq), [1, 2]);

    // A cold page has no prior live revision: canonical source revision still
    // wins over page arrival/timestamp, without coalescing equal executor words.
    const cold = { items: [] };
    for (const [summary, source] of [[merge('PR merged', 3), row('receipt-new', 3)],
        [speech(2), row('speech-2', 2)], [merge('PR queued', 1), row('receipt-old', 50)],
        [speech(1), row('speech-1', 1)]]) {
        mergeHistoricalTimelineItem(cold, summary, source, ts);
    }
    reconcileDelegatedItems(cold);
    assert.equal(cold.items.length, 3);
    assert.equal(cold.items.find(item => item.dedupeKey === key).headline, 'PR merged');
    assert.deepEqual(cold.items.filter(item => item.activity).map(item => item.activity.parts[0].seq), [1, 2]);
});

test('one child lifecycle survives chronological replay and older pages without losing narration', () => {
    const narration = 'Searching evidence. '.repeat(60) + 'COMPLETE_NARRATION_END';
    const lifecycleKey = 'subagent-lifecycle:child';
    const frames = [
        { headline: 'Scheduled', phase: 'queued', dedupeKey: lifecycleKey },
        { headline: 'Searching evidence', body: narration, dedupeKey: 'subagent-progress:child' },
        { headline: 'Running', phase: 'working', dedupeKey: lifecycleKey },
        { headline: 'Searching evidence', body: narration, dedupeKey: 'subagent-progress:child' },
        { headline: 'Completed', phase: 'done', terminal: true, dedupeKey: lifecycleKey },
    ];
    const row = index => ({ history_id: `progress:${index}`,
        history_position: { source: 'progress', offset: index }, ts: `2026-09-12T12:00:0${index}Z` });
    for (const order of [[0, 1, 2, 3, 4], [4, 3, 2, 1, 0], [2, 1, 0, 4, 3]]) {
        const record = { items: [], finished: true };
        for (const index of order) mergeHistoricalTimelineItem(record, frames[index], row(index), String(index));
        const status = record.items.filter(item => item.dedupeKey === lifecycleKey);
        assert.equal(status.length, 1);
        assert.equal(status[0].headline, 'Completed');
        assert.equal(status[0].sourceHistoryId, 'progress:4');
        const voices = record.items.filter(item => item.historyId);
        assert.deepEqual(voices.map(item => item.historyId), ['progress:1', 'progress:3']);
        assert.deepEqual(voices.map(item => item.fullBody), [narration, narration]);
        assert.equal(record.items.length, 3);
        const before = JSON.stringify(record.items);
        for (const index of order) assert.equal(mergeHistoricalTimelineItem(record, frames[index], row(index), String(index)), false);
        assert.equal(JSON.stringify(record.items), before);
        assert.equal(record.finished, true);
    }
});

test('an older lifecycle page cannot regress a status already updated live', () => {
    const record = { items: [] };
    const key = 'subagent-lifecycle:child';
    const frame = (headline, second) => updateLiveTimelineItem(record,
        { headline, phase: 'working', dedupeKey: key },
        { headline, ts: `12:00:0${second}`, rawTs: `2026-09-12T12:00:0${second}Z`,
            syntheticKey: key, inPlaceByKey: true });
    frame('Scheduled', 0);
    const item = record.items[0];
    frame('Running', 3);
    const old = { history_id: 'progress:1', history_position: { source: 'progress', offset: 1 },
        ts: '2026-09-12T12:00:01Z' };
    assert.equal(mergeHistoricalTimelineItem(record,
        { headline: 'Scheduled', phase: 'queued', dedupeKey: key }, old, '12:00:01'), true);
    assert.equal(record.items[0], item);
    assert.equal(item.headline, 'Running');
    assert.equal(item.sourceTs, '2026-09-12T12:00:03Z');
    assert.equal(item.sourceHistoryId, 'progress:1');
    assert.equal(item.historyId, undefined);
    assert.equal(mergeHistoricalTimelineItem(record,
        { headline: 'Scheduled', phase: 'queued', dedupeKey: key },
        { ...old, history_id: 'progress:0', ts: '2026-09-12T12:00:00Z' }, '12:00:00'), false);
    assert.equal(item.sourceHistoryId, 'progress:1', 'a rejected older source cannot redirect an existing locator');
    assert.equal(record.items.length, 1);
});

test('equal-time child lifecycle rows use source order, not page arrival order', () => {
    const record = { items: [] };
    const summary = { headline: 'Running', phase: 'working', dedupeKey: 'subagent-lifecycle:child' };
    const row = offset => ({ history_id: `progress:${offset}`, ts: '2026-09-12T12:00:00Z',
        history_position: { source: 'progress', offset } });
    mergeHistoricalTimelineItem(record, summary, row(2), '12:00');
    assert.equal(mergeHistoricalTimelineItem(record, { ...summary, headline: 'Scheduled' }, row(1), '12:00'), false);
    assert.equal(record.items[0].headline, 'Running');
    mergeHistoricalTimelineItem(record, { ...summary, headline: 'Completed', terminal: true }, row(3), '12:00');
    assert.equal(record.items.length, 1);
    assert.equal(record.items[0].headline, 'Completed');
    assert.equal(mergeHistoricalTimelineItem(record, summary, row(2), '12:00'), false);
});

test('equal-time identical narration retains physical identities in source order', () => {
    const record = { items: [], finished: true };
    const summary = { visible: true, headline: 'Identical narration', phase: 'working', dedupeKey: 'same' };
    const row = offset => ({ history_id: `progress:${offset}`,
        history_position: { source: 'progress', offset }, ts: '2026-09-12T12:00:00Z' });
    for (const offset of [20, 10, 30, 10]) mergeHistoricalTimelineItem(record, summary, row(offset), '12:00');
    assert.deepEqual(record.items.map(item => item.historyId), ['progress:10', 'progress:20', 'progress:30']);
    assert.equal(record.finished, true);
    assert.equal(compareHistoryPosition(row(10).history_position, row(20).history_position), -10);
});

test('a richer terminal updates its existing line while older terminal content cannot replace it', () => {
    const record = { items: [] };
    const summary = { visible: true, terminal: true, phase: 'done', headline: 'Done', dedupeKey: 'task_done|task' };
    const row = (offset, second) => ({ history_id: `chat:${offset}`,
        history_position: { source: 'chat', offset }, ts: `2026-09-12T12:00:0${second}Z` });
    mergeHistoricalTimelineItem(record, summary, row(10, 1), '12:00:01');
    const item = record.items[0];
    mergeHistoricalTimelineItem(record, { ...summary, body: 'Retained result details' }, row(20, 2), '12:00:02');
    assert.equal(record.items[0], item);
    assert.equal(item.body, 'Retained result details');
    mergeHistoricalTimelineItem(record, { ...summary, body: 'Older incomplete details' }, row(5, 0), '12:00:00');
    assert.equal(item.body, 'Retained result details');
});

test('narration and current terminal projection of one source have distinct DOM keys', () => {
    const record = { items: [] };
    const row = { history_id: 'progress:100', ts: '2026-09-12T12:00:00Z' };
    mergeHistoricalTimelineItem(record, { visible: true, headline: 'Narration' }, row, '12:00');
    mergeHistoricalTimelineItem(record, { visible: true, terminal: true, headline: 'Done', dedupeKey: 'task_done|t' }, row, '12:00');
    assert.equal(new Set(record.items.map(item => item.lineKey)).size, 2);
    assert.equal(record.items.filter(item => item.historyId).length, 1);
});

test('canonical adoption adds source identity without replacing a live expanded line key', () => {
    const record = { items: [], expandedLineKeys: new Set() };
    const summary = { visible: true, phase: 'working', headline: 'Result',
        body: 'Preview', truncated: true, fullRef: 'child', dedupeKey: 'subagent-result:child' };
    updateLiveTimelineItem(record, summary, {
        syntheticKey: summary.dedupeKey, headline: summary.headline,
        ts: '12:00', rawTs: '2026-09-12T12:00:00Z', inPlaceByKey: true,
    });
    const live = record.items[0];
    const key = live.lineKey;
    record.expandedLineKeys.add(key);
    assert.equal(mergeHistoricalTimelineItem(record, summary, {
        history_id: 'progress:41', history_position: { source: 'progress', offset: 41 },
        ts: '2026-09-12T12:00:00Z',
    }, '12:00'), true);
    assert.equal(record.items[0], live);
    assert.equal(live.lineKey, key);
    assert.equal(live.historyId, 'progress:41');
    assert.equal(record.expandedLineKeys.has(key), true);
    assert.equal(live.fullRef, 'child');
});

test('receipt revision ordering preserves its live identity and disclosure across stale and newer replay', () => {
    const record = { items: [], expandedLineKeys: new Set() };
    const summary = (revision, headline) => ({ dedupeKey: 'cardrow|receipt',
        cardRowRevision: revision, headline, body: `${headline} evidence`, phase: 'done' });
    const row = (offset, minute) => ({ history_id: `chat:${offset}`,
        history_position: { source: 'chat', offset }, ts: `2026-09-12T12:0${minute}:00Z` });
    const live = (revision, headline) => updateLiveTimelineItem(record, summary(revision, headline), {
        syntheticKey: 'cardrow|receipt', headline, ts: '12:02', rawTs: row(2, 2).ts, inPlaceByKey: true,
    });
    live(3, 'Merged');
    const item = record.items[0], key = item.lineKey;
    record.expandedLineKeys.add(key);
    const { sourceHistoryId: _source, sourceHistoryRevision: _revision, historyPosition: _position, ...content } = item;
    assert.equal(mergeHistoricalTimelineItem(record, summary(2, 'Queued'), row(20, 9), '12:09'), true);
    assert.equal(item.sourceHistoryId, 'chat:20');
    assert.equal(item.cardRowRevision, 3);
    assert.equal(mergeHistoricalTimelineItem(record, summary(3, 'Conflicting equal revision'), row(2, 2), '12:02'), true);
    const { sourceHistoryId, sourceHistoryRevision, historyPosition, ...adoptedContent } = item;
    assert.deepEqual(adoptedContent, content, 'equal revision adopts only physical provenance');
    assert.equal(sourceHistoryId, 'chat:2', 'canonical current source supersedes the older fallback despite earlier delivery time');
    assert.equal(sourceHistoryRevision, 3);
    assert.deepEqual(historyPosition, row(2, 2).history_position);
    const original = JSON.stringify(item);
    for (const revision of [undefined, 2, 3]) {
        assert.equal(mergeHistoricalTimelineItem(record, summary(revision, 'Queued'), row(90, 9), '12:09'), false);
        assert.equal(JSON.stringify(item), original, 'a later delivery or physical row cannot override revision order');
        assert.equal(live(revision, 'Queued').timelineUpdate, 'duplicate-skip');
        assert.equal(JSON.stringify(item), original);
    }
    assert.equal(mergeHistoricalTimelineItem(record, summary(4, 'Merged with evidence'), row(1, 0), '12:00'), true);
    assert.equal(record.items[0], item);
    assert.equal(item.sourceHistoryId, 'chat:1');
    assert.equal(item.cardRowRevision, 4);
    assert.equal(item.headline, 'Merged with evidence', 'a newer revision wins despite an earlier delivery time/offset');
    assert.equal(item.lineKey, key);
    assert.equal(record.expandedLineKeys.has(key), true);
    const updated = JSON.stringify(item);
    assert.equal(mergeHistoricalTimelineItem(record, summary(3, 'Merged'), row(90, 9), '12:09'), false);
    assert.equal(JSON.stringify(item), updated);
    assert.equal(live(5, 'Verified merge').timelineUpdate, 'patch-at');
    assert.equal(item.cardRowRevision, 5);
    assert.equal(item.headline, 'Verified merge');
    assert.equal(item.sourceHistoryRevision, 4, 'live content does not pretend its archived source advanced');
    assert.equal(mergeHistoricalTimelineItem(record, summary(5, 'Conflicting canonical words'), row(3, 0), '12:00'), true);
    assert.equal(item.sourceHistoryId, 'chat:3');
    assert.equal(item.sourceHistoryRevision, 5);
    assert.equal(item.headline, 'Verified merge');
    assert.equal(item.lineKey, key);
    assert.equal(record.expandedLineKeys.has(key), true);
    assert.equal(record.items.length, 1);
});

test('adopting a missing locator preserves the timestamp tie ordering witness', () => {
    const summary = { dedupeKey: 'subagent-lifecycle:child', headline: 'Running' };
    const record = { items: [{ ...summary, sourceTs: '2026-09-12T12:00:00Z',
        historyPosition: { source: 'progress', offset: 9 } }] };
    assert.equal(mergeHistoricalTimelineItem(record, { ...summary, headline: 'Queued' }, {
        history_id: 'progress:1', history_position: { source: 'progress', offset: 1 },
        ts: record.items[0].sourceTs,
    }, '12:00'), true);
    assert.equal(record.items[0].headline, 'Running');
    assert.equal(record.items[0].sourceHistoryId, 'progress:1');
});
