import assert from 'node:assert/strict';
import test from 'node:test';
import { adoptsCardRowAddress, clearAnchoredCardRows, mergeHistoricalTimelineItem, compareHistoryPosition } from '../modules/chat_history_replay.js';
import { mergePlacedCardRow } from '../modules/chat_row_placement.js';
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

test('equal-time identical child speech keeps distinct source rows through live redelivery', () => {
    const record = { items: [] }, ts = '2026-09-12T12:00:00Z';
    const summary = { headline: 'Researcher', body: 'Same words', phase: 'working',
        dedupeKey: `subagent-line:child:${ts}:Same words` };
    const row = offset => ({ history_id: `chat:${offset}`, ts, history_position: { source: 'chat', offset } });
    const deliver = () => updateLiveTimelineItem(record, summary, {
        syntheticKey: summary.dedupeKey, headline: summary.headline, ts: '12:00', rawTs: ts, inPlaceByKey: false,
    });
    deliver();
    const originalLineKey = record.items[0].lineKey;
    for (const offset of [10, 20, 10, 20]) mergeHistoricalTimelineItem(record, summary, row(offset), '12:00');
    assert.deepEqual(record.items.map(item => item.historyId), ['chat:10', 'chat:20']);
    assert.equal(record.items[0].lineKey, originalLineKey, 'adoption keeps the live line');
    deliver();
    assert.deepEqual(record.items.map(item => item.historyId), ['chat:10', 'chat:20'],
        'matching live words cannot merge physical rows or invent a third row');
    assert.deepEqual(record.items.map(item => item.count), [1, 1]);
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

test('a finalizing projection is the outcome line Working, and a finished card takes nothing from it', () => {
    const row = (offset, second) => ({ history_id: `chat:${offset}`,
        history_position: { source: 'chat', offset }, ts: `2026-09-12T12:00:0${second}Z` });
    const working = { visible: true, terminal: false, phase: 'working', headline: 'Working', dedupeKey: 'task_done|t' };
    const done = { visible: true, terminal: true, phase: 'done', headline: 'Done', dedupeKey: 'task_done|t' };
    const live = (record, second) => updateLiveTimelineItem(record, done, { ts: `12:00:0${second}`,
        rawTs: `2026-09-12T12:00:0${second}Z`, syntheticKey: done.dedupeKey, headline: done.headline, inPlaceByKey: true });
    // Still running: both finalizing rows are the one outcome line, keyed as live, so the
    // live terminal settles that line in place instead of standing beside a line of work.
    const open = { items: [] };
    assert.equal(mergeHistoricalTimelineItem(open, working, row(924, 1), '12:00:01'), true);
    mergeHistoricalTimelineItem(open, working, row(1904, 2), '12:00:02');
    assert.deepEqual(open.items.map(item => [item.dedupeKey, item.phase, item.historyId]), [['task_done|t', 'working', undefined]]);
    live(open, 3);
    assert.deepEqual(open.items.map(item => [item.phase, item.headline]), [['done', 'Done']]);
    // Finished: a newer read still finalizing neither adds a line nor reopens Done, while a
    // settled row and the card's ordinary history rows still land.
    const finished = { items: [], finished: true };
    live(finished, 3);
    const outcome = finished.items[0];
    assert.equal(mergeHistoricalTimelineItem(finished, working, row(1904, 4), '12:00:04'), false);
    assert.deepEqual([finished.items.length, outcome.phase, outcome.headline], [1, 'done', 'Done']);
    assert.equal(mergeHistoricalTimelineItem(finished, { ...done, body: 'Settled' }, row(3830, 5), '12:00:05'), true);
    assert.equal(outcome.body, 'Settled');
    assert.equal(mergeHistoricalTimelineItem(finished, { visible: true, headline: 'Narration' }, row(500, 1), '12:00:01'), true);
    assert.equal(finished.items.length, 2);
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

test('a terminal report keeps its full-report reference when it replaces the scheduled line', () => {
    const record = { items: [] };
    const key = 'subagent-lifecycle:child';
    const row = (id, second) => ({ history_id: id, history_position: { source: 'progress', offset: second },
        ts: `2026-09-12T12:00:0${second}Z` });
    const fullBody = 'Subagent child completed.\n\n## Report to parent\nCapped report\n\n[TRACE]\nShort trace';
    const scheduled = { visible: true, headline: 'Researcher', phase: 'start', dedupeKey: key,
        body: 'Starting.', fullBody: 'Starting.', fullRef: '', truncated: false };
    const terminal = { visible: true, headline: 'Researcher', phase: 'done', terminal: true, dedupeKey: key,
        body: 'Capped report', fullBody, fullRef: 'child', truncated: true };
    assert.equal(mergeHistoricalTimelineItem(record, scheduled, row('p0', 0), '12:00:00'), true);
    assert.equal(mergeHistoricalTimelineItem(record, terminal, row('p2', 2), '12:00:02'), true);
    assert.equal(record.items[0].truncated, true);
    assert.equal(record.items[0].fullRef, 'child');
    const before = JSON.stringify(record.items);
    assert.equal(mergeHistoricalTimelineItem(record, terminal, row('p2', 2), '12:00:02'), false);
    assert.equal(mergeHistoricalTimelineItem(record, scheduled, row('p0', 0), '12:00:00'), false);
    assert.equal(JSON.stringify(record.items), before);
});

test('exact around anchor and focused live landing preserve earlier System rows in both orders while keeping ordinary feed coalesced', () => {
    const revRow = (id, off, ts, rev, addr, text) => ({
        history_id: id, history_position: { source: 'progress', offset: off }, ts, row_address: addr,
        role: 'system', system_type: 'host_progress', card_row: 'timeline', card_row_id: 'sys-1',
        ...(rev != null ? { card_row_revision: rev } : {}), text,
    });
    const v1 = revRow('progress:10', 10, '2026-10-10T12:00:01Z', 1, 'row:1@t1#aaaaaaaaaaaa', 'Paused for verification');
    const v2 = revRow('progress:20', 20, '2026-10-10T12:00:05Z', 2, 'row:1@t2#bbbbbbbbbbbb', 'Verification complete');
    const u1 = { ...revRow('progress:31', 31, '2026-10-10T12:01:00Z', null, 'row:1@u1#111111111111', 'Retired 1 block'), card_row_id: 'sys-unrev' };
    const u2 = { ...revRow('progress:32', 32, '2026-10-10T12:01:05Z', null, 'row:1@u2#222222222222', 'Retired 2 blocks'), card_row_id: 'sys-unrev' };

    // Ordinary replay (no anchor): both temporal orders coalesce onto v2 / u2 and track zero anchored cards.
    for (const pair of [[v1, v2], [v2, v1], [u1, u2], [u2, u1]]) {
        const rec = { items: [] }, anchored = new Set();
        mergePlacedCardRow(rec, pair[0], { phase: 'warn' }, pair[0].ts, { anchoredRecords: anchored });
        mergePlacedCardRow(rec, pair[1], { phase: 'warn' }, pair[1].ts, { anchoredRecords: anchored });
        assert.equal(rec.items.length, 1);
        assert.equal(rec.items[0].replyAddress, pair.includes(v2) ? v2.row_address : u2.row_address);
        assert.equal(anchored.size, 0);
        assert.deepEqual(clearAnchoredCardRows(anchored), []);
    }

    // Explicit around(v1): cold (v1 then v2) keeps v1's original lineKey; warm (ordinary v1->v2 already mounted so v2
    // inherited v1's lineKey, then around(v1)) gives anchored v1 a distinct lineKey so DOM reconciliation never collides.
    const anchorV1 = { state: 'found', row_address: v1.row_address, history_id: v1.history_id };
    for (const [label, rows] of [['cold', [v1, v2]], ['warm', [v1, v2]]]) {
        const rec = { items: [] }, anchored = new Set();
        if (label === 'warm') {
            mergePlacedCardRow(rec, v1, { phase: 'warn' }, v1.ts);
            mergePlacedCardRow(rec, v2, { phase: 'warn' }, v2.ts);
        }
        let v1Key = '';
        for (const row of rows) {
            mergePlacedCardRow(rec, row, { phase: 'warn' }, row.ts, { anchor: anchorV1, anchoredRecords: anchored });
            if (row === v1 && label === 'cold') v1Key = rec.items[0].lineKey;
        }
        assert.deepEqual(rec.items.map((it) => [it.replyAddress, it.headline]), [
            [v1.row_address, 'Paused for verification'],
            [v2.row_address, 'Verification complete'],
        ], `${label}: chronological [v1, v2]`);
        if (label === 'cold') assert.equal(rec.items[0].lineKey, v1Key, 'cold supersession preserves anchored lineKey');
        assert.equal(new Set(rec.items.map((it) => it.lineKey)).size, 2);
        // Idempotent re-application and unconditional clear on leaving / next jump.
        assert.equal(mergePlacedCardRow(rec, v1, { phase: 'warn' }, v1.ts, { anchor: anchorV1, anchoredRecords: anchored }), false);
        assert.equal(clearAnchoredCardRows(anchored).length, 1);
        assert.deepEqual(rec.items.map((it) => it.replyAddress), [v2.row_address]);
        assert.equal(anchored.size, 0);
    }

    // Non-found/mismatched anchors and anchoring the current revision never duplicate or resurrect earlier rows.
    for (const anchor of [
        { state: 'found', row_address: v2.row_address, history_id: v2.history_id },
        { state: 'absent', row_address: v1.row_address, history_id: v1.history_id },
        { state: 'found', row_address: v1.row_address, history_id: 'progress:999' },
    ]) {
        const rec = { items: [] }, anchored = new Set();
        mergePlacedCardRow(rec, v1, { phase: 'warn' }, v1.ts, { anchor, anchoredRecords: anchored });
        mergePlacedCardRow(rec, v2, { phase: 'warn' }, v2.ts, { anchor, anchoredRecords: anchored });
        assert.deepEqual(rec.items.map((it) => it.replyAddress), [v2.row_address]);
        assert.equal(anchored.size, 0);
    }

    // Ordinary focus (Tab to each newest line with no quote navigation / no anchor) MUST NOT pin or accumulate.
    const tabRec = { items: [] }, tabAnchored = new Set();
    for (let step = 1; step <= 4; step += 1) {
        if (tabRec.items.length) {
            const newestKey = tabRec.items.at(-1).lineKey;
            const focusedLine = { dataset: { liveLineKey: newestKey }, closest: (sel) => (sel === '.chat-live-line' ? focusedLine : null) };
            tabRec.timelineEl = { ownerDocument: { activeElement: focusedLine }, contains: (node) => node === focusedLine };
        }
        const liveStep = {
            ...v2, history_id: undefined, history_position: undefined, card_row_revision: step,
            ts: `2026-10-10T12:00:4${step}Z`, row_address: `row:1@tab${step}#00000000000${step}`, text: `Live status ${step}`,
        };
        assert.equal(mergePlacedCardRow(tabRec, liveStep, { phase: 'warn' }, liveStep.ts, { anchoredRecords: tabAnchored }), true);
    }
    assert.deepEqual(tabRec.items.map((it) => [it.replyAddress, it.headline]), [
        ['row:1@tab4#000000000004', 'Live status 4'],
    ], 'ordinary keyboard focus without explicit quote anchor keeps only the latest coalesced status');
    assert.equal(tabAnchored.size, 0);

    // Explicit mounted quote anchor on displayed v2 + live v3: v2 keeps its exact lineKey & words,
    // v3 appends chronologically with a new lineKey, live v4 coalesces onto v3, same-address keep preserves v2,
    // and a subsequent mounted jump to another address releases v2 cleanly.
    const liveRec = { items: [] }, liveAnchored = new Set();
    mergePlacedCardRow(liveRec, v2, { phase: 'warn' }, v2.ts);
    const v2Key = liveRec.items[0].lineKey;
    const anchorV2 = { state: 'found', row_address: v2.row_address, history_id: v2.history_id };
    const liveV3 = { ...v2, history_id: undefined, history_position: undefined, card_row_revision: 3,
        ts: '2026-10-10T12:00:50Z', row_address: 'row:1@t3#cccccccccccc', text: 'Live evolved v3' };
    const liveV4 = { ...liveV3, card_row_revision: 4, ts: '2026-10-10T12:00:55Z',
        row_address: 'row:1@t4#dddddddddddd', text: 'Live evolved v4' };
    assert.equal(mergePlacedCardRow(liveRec, liveV3, { phase: 'warn' }, liveV3.ts, { anchor: anchorV2, anchoredRecords: liveAnchored }), true);
    assert.equal(liveRec.items[0].lineKey, v2Key, 'pinned v2 keeps its keyed DOM lineKey');
    assert.notEqual(liveRec.items[1].lineKey, v2Key, 'live v3 gets its own lineKey');
    assert.deepEqual(liveRec.items.map((it) => [it.replyAddress, it.headline]), [
        [v2.row_address, 'Verification complete'],
        [liveV3.row_address, 'Live evolved v3'],
    ]);
    assert.equal(mergePlacedCardRow(liveRec, liveV4, { phase: 'warn' }, liveV4.ts, { anchor: anchorV2, anchoredRecords: liveAnchored }), true);
    assert.deepEqual(liveRec.items.map((it) => [it.replyAddress, it.headline]), [
        [v2.row_address, 'Verification complete'],
        [liveV4.row_address, 'Live evolved v4'],
    ]);
    assert.deepEqual(clearAnchoredCardRows(liveAnchored, v2.row_address), [], 're-jumping to the same anchored address keeps it');
    assert.equal(liveAnchored.size, 1);
    assert.equal(clearAnchoredCardRows(liveAnchored, liveV4.row_address).length, 1, 'mounted jump to another address drops prior pin');
    assert.deepEqual(liveRec.items.map((it) => it.replyAddress), [liveV4.row_address]);
    assert.equal(liveAnchored.size, 0);

    // updateLiveTimelineItem preserves its original { timelineUpdate, patchIndex } return shape.
    const shapeRec = { items: [] };
    const shapeRes = updateLiveTimelineItem(shapeRec, { dedupeKey: 'plain|1', headline: 'Hi', body: 'Body', phase: 'working' },
        { ts: '12:00', rawTs: '2026-10-10T12:00:00Z', syntheticKey: 'plain|1', headline: 'Hi', inPlaceByKey: true });
    assert.deepEqual(Object.keys(shapeRes).sort(), ['patchIndex', 'timelineUpdate']);
});

test('equal-revision receipt address adoption upgrades matching unaddressed projections on replay and live while rejecting mismatches, stale revisions and unauthored rows', () => {
    const baseText = 'PR #7 merge: refused\nBranch protection rejected the merge.';
    const receiptRow = (extra = {}) => ({
        role: 'system', system_type: 'host_progress', is_progress: true,
        card_row: 'reviews', card_row_id: 'merge-receipt:r7', card_row_revision: 2,
        ts: '2026-10-12T12:00:05Z', text: baseText, ...extra,
    });
    const addrV2 = 'row:1@2026-10-12T12:00:05Z#bbbbbbbbbbbb';
    const addrOther = 'row:1@2026-10-12T12:00:06Z#cccccccccccc';
    const projectedHistory = receiptRow({
        history_id: 'progress:1', history_position: { source: 'progress', offset: 1 },
        ts: '2026-10-12T12:00:01Z',
    }); // earlier carrier rewritten by server projection -> row_address stripped
    const genuineHistory = receiptRow({
        history_id: 'progress:2', history_position: { source: 'progress', offset: 2 },
        row_address: addrV2,
    });
    const genuineLive = receiptRow({ content: baseText, row_address: addrV2 });

    // 1. Replay adoption (ordinary + exact around anchor): upgrades in place, keeps lineKey & disclosure, no duplicate anchored row.
    const replayRec = { items: [], expandedLineKeys: new Set() }, replayAnchored = new Set();
    assert.equal(mergePlacedCardRow(replayRec, projectedHistory, { phase: 'result' }, projectedHistory.ts, { anchoredRecords: replayAnchored }), true);
    const replayItem = replayRec.items[0], replayLineKey = replayItem.lineKey;
    replayRec.expandedLineKeys.add(replayLineKey);
    assert.equal(replayItem.replyAddress, '');
    assert.equal(replayItem.sourceHistoryId, 'progress:1');
    const anchorV2 = { state: 'found', row_address: addrV2, history_id: 'progress:2' };
    assert.equal(mergePlacedCardRow(replayRec, genuineHistory, { phase: 'result' }, genuineHistory.ts, { anchor: anchorV2, anchoredRecords: replayAnchored }), true);
    assert.equal(replayRec.items.length, 1, 'equal-revision adoption upgrades the displayed row without mounting a duplicate anchored copy');
    assert.equal(replayRec.items[0], replayItem);
    assert.equal(replayItem.lineKey, replayLineKey);
    assert.equal(replayRec.expandedLineKeys.has(replayLineKey), true);
    assert.equal(replayItem.replyAddress, addrV2);
    assert.equal(replayItem.replyRole, 'system');
    assert.equal(replayItem.sourceHistoryId, 'progress:2');
    assert.deepEqual(replayItem.historyPosition, genuineHistory.history_position);
    assert.equal(replayAnchored.size, 0);
    // Idempotent re-delivery and refusal to overwrite an already-addressed row with a different equal-revision address.
    assert.equal(mergePlacedCardRow(replayRec, genuineHistory, { phase: 'result' }, genuineHistory.ts), false);
    assert.equal(mergePlacedCardRow(replayRec, { ...genuineHistory, history_id: 'progress:9', row_address: addrOther }, { phase: 'result' }, genuineHistory.ts), false);
    assert.equal(mergePlacedCardRow(replayRec, { ...genuineLive, row_address: addrOther }, { phase: 'result' }, genuineLive.ts), false);
    assert.equal(replayItem.replyAddress, addrV2, 'an already-addressed row never swaps its physical identity');
    assert.equal(replayItem.sourceHistoryId, 'progress:2');

    // 2. Live adoption followed by replay of the genuine stored row reconciles both physical address and durable history locator.
    const liveRec = { items: [], expandedLineKeys: new Set() };
    assert.equal(mergePlacedCardRow(liveRec, projectedHistory, { phase: 'result' }, projectedHistory.ts), true);
    const liveItem = liveRec.items[0], liveLineKey = liveItem.lineKey;
    liveRec.expandedLineKeys.add(liveLineKey);
    assert.equal(liveItem.replyAddress, '');
    assert.equal(mergePlacedCardRow(liveRec, genuineLive, { phase: 'result' }, genuineLive.ts), true);
    assert.equal(liveRec.items.length, 1);
    assert.equal(liveRec.items[0], liveItem);
    assert.equal(liveItem.lineKey, liveLineKey);
    assert.equal(liveRec.expandedLineKeys.has(liveLineKey), true);
    assert.equal(liveItem.replyAddress, addrV2);
    assert.equal(liveItem.replyRole, 'system');
    assert.equal(liveItem.sourceHistoryId, 'progress:1');
    assert.equal(mergePlacedCardRow(liveRec, genuineHistory, { phase: 'result' }, genuineHistory.ts), true);
    assert.equal(liveItem.sourceHistoryId, 'progress:2', 'canonical replay row reconciles sourceHistoryId after live adoption');
    assert.equal(liveItem.lineKey, liveLineKey);
    assert.equal(mergePlacedCardRow(liveRec, genuineHistory, { phase: 'result' }, genuineHistory.ts), false);

    // 3. Rejections: different wording (conflicting text, later-review FAIL note, unavailability note), stale/unversioned revision,
    // malformed address, and unauthored / non-system / narration / virtual / serviceStop projections must never acquire the address.
    const rejectedProjections = [
        ['different words', projectedHistory, { ...genuineHistory, text: 'PR #7 merge: refused\nDifferent detail.' }, { ...genuineLive, content: 'PR #7 merge: refused\nDifferent detail.', text: 'PR #7 merge: refused\nDifferent detail.' }],
        ['later-review FAIL projection', { ...projectedHistory, text: `${baseText}\nIts later review: FAIL.`, late_evidence: { kind: 'merge_review', record_id: 'rev', revision: 2, verdict: 'FAIL' } }, genuineHistory, genuineLive],
        ['unavailability note projection', { ...projectedHistory, text: `${baseText}\nCurrent merge receipt unavailable; showing recorded event.` }, genuineHistory, genuineLive],
        ['stale revision with identical words', { ...projectedHistory, card_row_revision: 3 }, genuineHistory, genuineLive],
        ['unversioned incoming row', projectedHistory, { ...genuineHistory, card_row_revision: undefined }, { ...genuineLive, card_row_revision: undefined }],
        ['non-host address', projectedHistory, { ...genuineHistory, row_address: 'task:merge-task' }, { ...genuineLive, row_address: 'row:' }],
        ['unauthored (missing role)', { ...projectedHistory, role: undefined }, genuineHistory, genuineLive],
        ['assistant-authored projection', { ...projectedHistory, role: 'assistant' }, genuineHistory, genuineLive],
        ['narration projection', { ...projectedHistory, narration: true }, genuineHistory, genuineLive],
        ['lifecycle_virtual projection', { ...projectedHistory, lifecycle_virtual: true }, genuineHistory, genuineLive],
        ['non-system incoming row', projectedHistory, { ...genuineHistory, role: 'assistant' }, { ...genuineLive, role: 'assistant' }],
    ];
    for (const [label, initial, incomingReplay, incomingLive] of rejectedProjections) {
        const rec = { items: [] };
        assert.equal(mergePlacedCardRow(rec, initial, { phase: 'result' }, initial.ts), true, `${label}: seed`);
        assert.equal(rec.items[0].replyAddress, '', `${label}: starts unaddressed`);
        assert.equal(mergePlacedCardRow(rec, incomingLive, { phase: 'result' }, incomingLive.ts), false, `${label}: live rejected`);
        assert.equal(rec.items[0].replyAddress, '', `${label}: stays unaddressed after live`);
        assert.equal(mergePlacedCardRow(rec, incomingReplay, { phase: 'result' }, incomingReplay.ts), false, `${label}: replay rejected`);
        assert.equal(rec.items[0].replyAddress, '', `${label}: stays unaddressed after replay`);
    }

    // 4. An exact around read for the genuine recorded row when the displayed projection has a FAIL note keeps both distinct:
    // the FAIL projection stays unaddressed while the exact recorded row mounts as an anchored copy with its own words & address.
    const failRec = { items: [] }, failAnchored = new Set();
    const failProjected = { ...projectedHistory, text: `${baseText}\nIts later review: FAIL.`,
        late_evidence: { kind: 'merge_review', record_id: 'rev', revision: 2, verdict: 'FAIL' } };
    assert.equal(mergePlacedCardRow(failRec, failProjected, { phase: 'result' }, failProjected.ts, { anchoredRecords: failAnchored }), true);
    assert.equal(mergePlacedCardRow(failRec, genuineHistory, { phase: 'result' }, genuineHistory.ts, { anchor: anchorV2, anchoredRecords: failAnchored }), true);
    assert.equal(failRec.items.length, 2);
    assert.equal(failRec.items.find((it) => it.dedupeKey === 'cardrow|merge-receipt:r7').replyAddress, '');
    assert.equal(failRec.items.find((it) => it.replyAddress === addrV2).body, 'Branch protection rejected the merge.');
    assert.equal(failAnchored.size, 1);

    // 5. A serviceStop item, unclassified prior/incoming record, or mismatched/missing key never adopts a receipt address.
    const stopRec = { items: [] };
    updateLiveTimelineItem(stopRec, {
        dedupeKey: 'cardrow|stop-1', cardRowRevision: 2, headline: 'Stopped', body: 'Done',
        phase: 'done', replyRole: 'system', serviceStop: { links: [] },
    }, { ts: '12:00', rawTs: '2026-10-12T12:00:00Z', syntheticKey: 'cardrow|stop-1', headline: 'Stopped', inPlaceByKey: true });
    assert.equal(updateLiveTimelineItem(stopRec, {
        dedupeKey: 'cardrow|stop-1', cardRowRevision: 2, headline: 'Stopped', body: 'Done',
        phase: 'done', replyRole: 'system', replyAddress: addrV2,
    }, { ts: '12:00', rawTs: '2026-10-12T12:00:00Z', syntheticKey: 'cardrow|stop-1', headline: 'Stopped', inPlaceByKey: true }).timelineUpdate, 'duplicate-skip');
    assert.equal(stopRec.items[0].replyAddress, undefined);
    const baseItem = { dedupeKey: 'cardrow|merge-receipt:r7', cardRowRevision: 2, headline: 'PR #7 merge: refused',
        body: 'Branch protection rejected the merge.', phase: 'result', replyAddress: '', replyRole: 'system' };
    const baseSum = { ...baseItem, replyAddress: addrV2 };
    assert.equal(adoptsCardRowAddress(baseItem, baseSum), true);
    assert.equal(adoptsCardRowAddress({ ...baseItem, replyRole: undefined }, baseSum), false, 'unclassified prior record must not be promoted');
    assert.equal(adoptsCardRowAddress(baseItem, { ...baseSum, replyRole: undefined }), false, 'unclassified summary rejected');
    assert.equal(adoptsCardRowAddress(baseItem, { ...baseSum, dedupeKey: undefined }), false, 'missing summary key rejected');
    assert.equal(adoptsCardRowAddress(baseItem, { ...baseSum, dedupeKey: 'cardrow|merge-receipt:other' }), false, 'different receipt key rejected');

    // 6. keepAddress preserves a host address ONLY for the same receipt/revision with positively System-authored
    // compatible words — never across revisions (e.g. observed_after advancing N -> N+1 with identical words), so a
    // later genuine N+1 row still adopts N+1's own host address on both replay and live.
    const addrV3 = 'row:1@2026-10-12T12:00:10Z#dddddddddddd';
    const passReview = (rev) => ({ kind: 'merge_review', record_id: 'rev', revision: rev, verdict: 'PASS' });
    const projectedRev3WithReview = receiptRow({
        card_row_revision: 3, history_id: 'progress:1', history_position: { source: 'progress', offset: 1 },
        ts: '2026-10-12T12:00:01Z', late_evidence: passReview(3),
    });
    const genuineRev3History = receiptRow({
        card_row_revision: 3, history_id: 'progress:3', history_position: { source: 'progress', offset: 3 },
        ts: '2026-10-12T12:00:10Z', row_address: addrV3,
    });
    const genuineRev3Live = receiptRow({
        card_row_revision: 3, ts: '2026-10-12T12:00:10Z', content: baseText, row_address: addrV3,
    });

    // Same revision + System + same words + newerReview preserves addrV2; non-system or changed words drop it.
    const sameRevKeep = { items: [] };
    assert.equal(mergePlacedCardRow(sameRevKeep, genuineHistory, { phase: 'result' }, genuineHistory.ts), true);
    assert.equal(mergePlacedCardRow(sameRevKeep, { ...projectedHistory, late_evidence: passReview(2) }, { phase: 'result' }, projectedHistory.ts), true);
    assert.equal(sameRevKeep.items[0].replyAddress, addrV2, 'same-revision System newerReview with identical words keeps addrV2');
    assert.equal(sameRevKeep.items[0].sourceHistoryId, 'progress:2');
    assert.equal(sameRevKeep.items[0].cardReview.revision, 2);
    assert.equal(mergePlacedCardRow(sameRevKeep, { ...projectedHistory, role: 'assistant', late_evidence: passReview(3) }, { phase: 'result' }, projectedHistory.ts), true);
    assert.equal(sameRevKeep.items[0].replyAddress, '', 'non-system newerReview must not preserve System address');

    // Advancing N -> N+1 with identical words + newerReview must drop rev N's address and adopt rev N+1's genuine address.
    for (const [mode, genuineNext] of [['replay', genuineRev3History], ['live', genuineRev3Live]]) {
        const rec = { items: [] };
        assert.equal(mergePlacedCardRow(rec, genuineHistory, { phase: 'result' }, genuineHistory.ts), true);
        const item = rec.items[0], lineKey = item.lineKey;
        assert.equal(mergePlacedCardRow(rec, projectedRev3WithReview, { phase: 'result' }, projectedRev3WithReview.ts), true);
        assert.equal(item.cardRowRevision, 3, `${mode}: revision advances to N+1`);
        assert.equal(item.replyAddress, '', `${mode}: rev N address must never carry across to rev N+1`);
        assert.equal(item.sourceHistoryId, 'progress:1');
        assert.equal(mergePlacedCardRow(rec, genuineNext, { phase: 'result' }, genuineNext.ts), true, `${mode}: genuine N+1 adopted`);
        assert.equal(rec.items.length, 1);
        assert.equal(item.lineKey, lineKey);
        assert.equal(item.replyAddress, addrV3, `${mode}: rev N+1 acquires its own genuine host address`);
        assert.equal(item.replyRole, 'system');
        if (mode === 'replay') assert.equal(item.sourceHistoryId, 'progress:3');
    }
});
