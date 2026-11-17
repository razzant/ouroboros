// A delegated run's typed observation (#1350) as a task-card row: readable
// attributed words, thinking behind a labelled disclosure, compact technical
// events, visible problems and unread ranges, the retained source one click
// away, and each (run_id, seq) shown once within the disclosed identity budgets,
// live, after a reload and reconnect, whatever ranges the observations covered.
import assert from 'node:assert/strict';
import test from 'node:test';
import { summarizeChatLiveEvent } from '../modules/log_events.js';
import {
    delegatedActivityBodyHtml, delegatedActivityView, delegatedHeadline, delegatedLineView, reconcileDelegatedItems,
} from '../modules/delegated_activity.js';
import { buildTimelineItemHtml } from '../modules/chat_activity.js';
import { mergeHistoricalTimelineItem } from '../modules/chat_history_replay.js';
import { updateLiveTimelineItem } from '../modules/chat_render_batch.js';

// The shared renderer escapes through a DOM text node; this file needs only that.
globalThis.document ??= {
    createElement: () => {
        let text = '';
        return {
            set textContent(value) { text = String(value); },
            get innerHTML() { return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); },
        };
    },
};

const SHA = 'ab'.repeat(32);
const refFor = (first, last, size = 10) => ({ kind: 'task_source', root: 'artifact_store',
    path: `source_handles/delegated_activity/run-1-${first}-${last}-${SHA}.jsonl`, size, sha256: SHA });

function activity(after, through, parts, extra = {}) {
    return { v: 1, task_id: 'child', run_id: 'run-1', after_seq: after, through_seq: through,
        source: { kind: 'run_events', read_through: through, events: through - after, ref: refFor(after + 1, through) },
        parts, ...extra };
}

const burst = activity(0, 8, [
    { kind: 'thinking', actor: 'claude/a01', text: 'maybe the key lives in chat.js', seq: 2, last_seq: 2 },
    { kind: 'message', actor: 'claude/a01', text: 'Checking the **Telegram** consumer next.', seq: 3, last_seq: 3 },
    { kind: 'problem', actor: 'claude/a01', label: 'Bash', text: 'exit 1: 2 failed', seq: 6 },
    { kind: 'thinking', actor: 'claude/a01', text: 'second thought', seq: 7, last_seq: 7 },
], { technical: { count: 4, labels: [['Read', 2], ['Bash', 2]], seqs: [[1, 1], [4, 5], [8, 8]],
    recent: [{ seq: 1, label: 'Read', detail: 'chat.js' }, { seq: 4, label: 'Bash', detail: 'pytest -q' },
        { seq: 5, label: 'Bash' }, { seq: 8, label: 'Read' }] } });

const frame = (record, extra = {}) => ({ type: 'send_message', is_progress: true, task_id: 'child',
    content: '💬 🛰 delegated run run-1 @seq 8\n[claude/a01] Bash · tool_result', narration: false,
    delegated_activity: record, ...extra });

test('root and child frames project the typed observation, never the joined label text', () => {
    for (const extra of [{}, { delegation_role: 'subagent', subagent_event: 'progress', subagent_task_id: 'child', parent_task_id: 'root' }]) {
        const view = summarizeChatLiveEvent(frame(burst, extra));
        assert.equal(view.headline, 'claude · a01 · run-1');
        assert.equal(view.body, 'Checking the **Telegram** consumer next.');
        assert.equal(view.dedupeKey, 'delegated:run-1:0-8');
        assert.equal(view.promote, false, 'executor words never claim the card title');
        assert.equal(view.human, false, 'nor the collapsed activity line');
        assert.equal(view.activity, burst);
        assert.equal(view.toolCall, undefined, 'technical events never enter the native tool fold or its counts');
        assert.ok(!view.fullBody.includes('tool_result'), 'the flattened frame text is not reinterpreted');
    }
    assert.equal(delegatedActivityView({ type: 'send_message', is_progress: true, content: 'plain' }), null);
    const silent = activity(8, 9, [], { technical: { count: 1, labels: [['Read', 1]], seqs: [[9, 9]],
        recent: [{ seq: 9, actor: 'codex/a02', label: 'Read' }] }, latest_message: { seq: 3, actor: 'claude/a01', text: 'x' } });
    assert.equal(summarizeChatLiveEvent(frame(silent)).headline, 'codex · a02 · technical activity · run-1',
        'a silent row is named by who did the technical work');
    assert.equal(summarizeChatLiveEvent(frame({ ...silent, technical: { count: 1, labels: [['Read', 1]], seqs: [[9, 9]] } })).headline,
        'claude · a01 · technical activity · run-1', 'else by the run latest speaker');
});

test('messages lead, thinking sits behind one labelled disclosure, problems, counts and the source stay reachable', () => {
    const view = delegatedLineView({ activity: burst });
    const html = delegatedActivityBodyHtml(view);
    const message = html.indexOf('chat-delegated-message');
    const problem = html.indexOf('chat-delegated-problem');
    const thinking = html.indexOf('<details class="chat-delegated-thinking"><summary>Thinking (2)</summary>');
    assert.ok(message !== -1 && problem > message && thinking > problem, html);
    assert.match(html, /<strong>Telegram<\/strong>/, 'the executor message uses the shared rich-text renderer');
    assert.match(html, /<div class="chat-delegated-problem">Problem: claude · a01 · Bash: exit 1: 2 failed/, 'words and warn tone identify the problem without a glyph');
    assert.match(html, /4 technical events since seq 1 · Read ×2 · Bash ×2/);
    assert.ok(!html.includes('chat-delegated-events'), 'the ordered technical detail waits for the row Expand');
    const expanded = delegatedActivityBodyHtml(view, { expanded: true });
    assert.match(expanded, /<ol class="chat-delegated-events"><li><span class="chat-delegated-seq">#1<\/span> Read — chat.js<\/li>/);
    const link = expanded.match(/<a class="chat-delegated-source" href="([^"]+)" download="([^"]+)">Full source · seq 1–8 · 8 events · 10 bytes \(JSONL\)<\/a>/);
    assert.ok(link, expanded);
    assert.equal(link[1], `/api/tasks/child/artifacts/run-1-1-8-${SHA}.jsonl?source=${encodeURIComponent(refFor(1, 8).path)}`,
        'the existing task-file route, published-source selector');
    const forged = activity(0, 8, [], { source: { kind: 'run_events', read_through: 8, events: 8,
        ref: { ...refFor(1, 8), path: 'source_handles/../x.jsonl' } } });
    assert.ok(!delegatedActivityBodyHtml(delegatedLineView({ activity: forged }), { expanded: true }).includes('<a '),
        'a ref outside the delegated source handles yields no link');
});

test('a bounded preview links its complete original; unread ranges are stated by seq, final ones as final', () => {
    const long = activity(10, 20, [{ kind: 'message', actor: 'claude/a01', text: 'x…', chars: 9000, truncated: true, seq: 11, last_seq: 11 }],
        { gaps: [{ after_seq: 15, through_seq: 20, reason: 'read_bound:time' }] });
    const html = delegatedActivityBodyHtml(delegatedLineView({ activity: long }));
    assert.match(html, /<div class="chat-delegated-note">Read it whole: <a class="chat-delegated-source" href="\/api\/tasks\/child\/artifacts\/run-1-11-20-/,
        'the complete text is one click from the collapsed row');
    assert.match(html, /Not shown yet: seq 16–20 \(read_bound:time\); read at the run&#39;s next advance or its end\.|Not shown yet: seq 16–20 \(read_bound:time\); read at the run's next advance or its end\./);
    const final = { ...long, gaps: [{ after_seq: 15, through_seq: 20, reason: 'read_bound:bytes', final: true, where: 'run_journal' }] };
    const finalHtml = delegatedActivityBodyHtml(delegatedLineView({ activity: final }));
    assert.match(finalHtml, /chat-delegated-gap chat-delegated-final/);
    assert.match(finalHtml, /Not shown: seq 16–20 \(read_bound:bytes\)\. The wait returned before reading them; they remain on run run-1/);
    const windowView = { ...long, source: { kind: 'timeline_window', reason: 'stream_not_listed', rows: 1, rows_omitted: 4 }, gaps: [] };
    const windowHtml = delegatedActivityBodyHtml(delegatedLineView({ activity: windowView }), { expanded: true });
    assert.match(windowHtml, /The run&#39;s journal holds it whole\.|The run's journal holds it whole\./);
    assert.match(windowHtml, /\+4 earlier timeline rows not shown/);
    assert.match(windowHtml, /Bounded timeline window of run run-1 \(stream_not_listed\); exact originals stay on the run/);
});

test('after the engine refused the journal a row promises no later read or journal copy, its code and range in Expand', () => {
    const APOS = "(?:'|&#39;)";
    const PROMISES = /reads them|holds it whole|stays on the run|on the run(?:'|&#39;)s journal|originals stay|remain on run/;
    const html = (activity, expanded) => delegatedActivityBodyHtml(delegatedLineView({ activity }), { expanded });
    const refusedGap = (reason, retryable, final = true) => [{ after_seq: 15, through_seq: 20, reason, retryable, final }];
    // A window the engine refused for good: the bounded window row is realistic (300 chars and its omission note,
    // delegate_progress), a second part was cut by this record's own preview.
    const windowText = `${'Reading the renderer. '.repeat(14)}\n⚠️ OMISSION NOTE: truncated at 300 chars; original length 7000`;
    const reclaimed = { ...activity(10, 20, [
        { kind: 'message', actor: 'claude/a01', text: windowText },
        { kind: 'message', actor: 'claude/a01', text: 'y…', chars: 9000, truncated: true }]),
        source: { kind: 'timeline_window', reason: 'stream_error:run_expired_by_retention', retryable: false, rows: 2 },
        gaps: refusedGap('stream_error:run_expired_by_retention', false), omitted: { message: 1 } };
    const quiet = html(reclaimed, false);
    assert.match(quiet, new RegExp(`chat-delegated-gap chat-delegated-final">Not all of this run${APOS}s events are shown: its journal can no longer be read, so waiting again will not show the rest\\. The run${APOS}s result is reported separately\\.`));
    assert.match(quiet, /Reading the renderer\./, 'the window words stay visible');
    assert.match(quiet, /The rest of it can no longer be read\./);
    assert.match(quiet, /Not in this row: 1 message \(they can no longer be read\)\./);
    assert.doesNotMatch(quiet, /stream_error|seq 16/, 'the code and the range stay quiet in the row');
    assert.doesNotMatch(quiet, PROMISES);
    const open = html(reclaimed, true);
    assert.match(open, /Not read: seq 16–20\./, 'the window source line below names the code once');
    assert.match(open, new RegExp(`Bounded timeline window of run run-1 \\(stream_error:run_expired_by_retention\\); the run${APOS}s journal can no longer be read\\.`));
    assert.doesNotMatch(open, PROMISES);
    // Exact events read before the refusal whose original failed to save, and a draft its complete text updated.
    const lost = activity(10, 20, [{ kind: 'message', actor: 'claude/a01', text: 'z…', chars: 9000, truncated: true, seq: 11, last_seq: 11 },
        { kind: 'message', actor: 'claude/a01', text: 'Final words.', speech: { key: 'k1', op: 'complete' }, updated: true, seq: 12, last_seq: 12 }],
    { source: { kind: 'run_events', read_through: 12, events: 2, retained: false },
        gaps: [{ after_seq: 12, through_seq: 20, reason: 'stream_error:run_expired_by_retention', retryable: false, final: true }] });
    const lostOpen = html(lost, true);
    assert.match(lostOpen, /Not read: seq 13–20 \(stream_error:run_expired_by_retention\)\./);
    assert.match(lostOpen, /The rest of it can no longer be read\./);
    assert.match(lostOpen, /Updated\. The streamed draft can no longer be read\./);
    assert.match(lostOpen, /Source: run run-1 seq 11–12 \(retention failed; originals can no longer be read\)/);
    assert.doesNotMatch(lostOpen, PROMISES);
    const noTail = { ...lost, through_seq: 12, gaps: [], source: { ...lost.source,
        reason: 'stream_error:run_expired_by_retention', retryable: false } };
    const noTailHtml = html(noTail, true);
    assert.match(noTailHtml, /originals can no longer be read/);
    assert.doesNotMatch(noTailHtml, PROMISES, 'a refusal is current even when no unread tail needs a gap');
    // Refused this time: the gap says another wait tries again, without a provisional line or the code.
    const retryable = { ...reclaimed, parts: [], omitted: {},
        source: { kind: 'timeline_window', reason: 'stream_error:run_event_not_found', provisional: true, retryable: true, rows: 0 },
        gaps: refusedGap('stream_error:run_event_not_found', true) };
    const retryQuiet = html(retryable, false);
    assert.match(retryQuiet, new RegExp(`Not all of this run${APOS}s events are shown yet: the engine could not read its journal\\. Another wait on the run tries again\\.`));
    assert.doesNotMatch(retryQuiet, /stream_error|seq 16|shown meanwhile|can no longer/);
    const retryOpen = html(retryable, true);
    assert.match(retryOpen, /Not read: seq 16–20\./);
    assert.match(retryOpen, new RegExp(`Bounded timeline window of run run-1 \\(stream_error:run_event_not_found\\); the engine could not read the run${APOS}s journal\\.`));
    assert.doesNotMatch(retryOpen, PROMISES);
    const live = html({ ...retryable, gaps: refusedGap('stream_error:run_event_not_found', true, false) }, false);
    assert.match(live, new RegExp(`The rest is read at the run${APOS}s next advance or its end\\.`));
});

test('the row renders its words without a second Expand and keeps Expand for the detail', () => {
    const view = summarizeChatLiveEvent(frame(burst));
    const record = { items: [], expandedLineKeys: new Set(), groupId: 'child' };
    updateLiveTimelineItem(record, view, { ts: '12:00', rawTs: '2026-09-27T12:00:00Z', syntheticKey: view.dedupeKey, headline: view.headline, inPlaceByKey: false });
    const [item] = record.items;
    assert.equal(item.activity, burst);
    const collapsed = buildTimelineItemHtml(item, record);
    assert.match(collapsed, /class="chat-live-line working expandable"/);
    assert.match(collapsed, /chat-live-line-body chat-delegated-activity/);
    assert.match(collapsed, /Checking the <strong>Telegram<\/strong> consumer next\./);
    record.expandedLineKeys.add(item.lineKey);
    assert.match(buildTimelineItemHtml(item, record), /chat-delegated-events/);
});

// -- identity -----------------------------------------------------------------------------

test('ordinary progress and thinking headlines still render Markdown through the timeline consumer', () => {
    for (const phase of ['working', 'thinking']) {
        const html = buildTimelineItemHtml({ phase, headline: '**Reading** `consumer.js`', lineKey: phase },
            { groupId: 'ordinary', expandedLineKeys: new Set() });
        assert.match(html, /data-chat-markdown-enhanced/);
        assert.match(html, /<strong>Reading<\/strong>/);
        assert.match(html, /<code class="inline-code">consumer.js<\/code>/);
        assert.ok(!html.includes('chat-delegated-activity'));
    }
});

const say = (seq, text, extra = {}) => ({ kind: 'message', actor: 'claude/a01', text, seq, last_seq: seq, ...extra });

test('overlapping bounded rows keep unseen omitted speech or problems and their full source', () => {
    for (const kind of ['message', 'problem']) {
        const first = activity(0, 40, Array.from({ length: 40 }, (_, i) => say(i + 1, `line ${i + 1}`)));
        const bounded = activity(0, 41, first.parts, { omitted: { [kind]: 1 } });
        const record = { items: [{ activity: first }, { activity: bounded }, { activity: bounded }] };
        reconcileDelegatedItems(record);
        const unseen = delegatedLineView(record.items[1]);
        assert.equal(unseen.hidden, false, 'seq 41 was read but its preview was omitted, not shown');
        const html = delegatedActivityBodyHtml(unseen);
        assert.match(html, new RegExp(`Not in this row: 1 ${kind}`));
        assert.match(html, /open the full source/);
        assert.ok(html.includes(encodeURIComponent(refFor(1, 41).path)));
        assert.equal(delegatedLineView(record.items[2]).hidden, true, 'the identical omission is already visible');
        const replay = { items: record.items.map(({ activity }) => ({ activity: structuredClone(activity) })) };
        reconcileDelegatedItems(replay);
        assert.equal(delegatedActivityBodyHtml(delegatedLineView(replay.items[1])), html);
    }
});
test('a conditional early stream end states unresolved coverage without claiming the missing source exists', () => {
    const row = activity(0, 3, [say(1, 'first')], { source: { kind: 'run_events', read_through: 1, ended: true },
        gaps: [{ after_seq: 1, through_seq: 3, reason: 'stream_end_before_fence', final: true }] });
    const record = { items: [{ activity: row }] };
    reconcileDelegatedItems(record);
    const html = delegatedActivityBodyHtml(delegatedLineView(record.items[0]));
    assert.match(html, /seq 2–3/);
    assert.match(html, /coverage unresolved on run run-1/);
    assert.ok(!html.includes('they remain'));
});
const tech = (seqs, label = 'Read') => {
    const flat = seqs.flatMap(([a, b]) => Array.from({ length: b - a + 1 }, (_, i) => a + i));
    return { count: flat.length, labels: [[label, flat.length]], seqs, recent: flat.map((seq) => ({ seq, label, actor: 'claude/a01' })) };
};

// The same journal observed twice: once by a process that polled (0,4] then (4,5],
// once by a process that lost its cursor and re-read (0,12] after a restart. The
// delta stream 3–6 ("Hel", "lo", " wor", "ld") crosses both boundaries, and the
// same words are said at seq 2 and seq 8.
const hello = (seq, last, text, cuts) => say(seq, text, { last_seq: last, delta: true, fragments: cuts.length + 1, cuts });
function journal() {
    return [
        activity(0, 4, [say(2, 'Reading the consumer.'), hello(3, 4, 'Hello', [[4, 3]])], { technical: tech([[1, 1]]) }),
        activity(4, 5, [hello(5, 5, ' wor', [])]),
        activity(0, 12, [say(2, 'Reading the consumer.'), hello(3, 6, 'Hello world', [[4, 3], [5, 5], [6, 9]]),
            say(8, 'Reading the consumer.'), say(11, 'Done.')], { technical: tech([[1, 1], [7, 7], [9, 10], [12, 12]]) }),
    ];
}

function visible(record) {
    reconcileDelegatedItems(record);
    return record.items.filter((item) => item.activity).map((item) => delegatedLineView(item))
        .filter((view) => !view.hidden)
        .map((view) => [view.parts.map((shown) => shown.text), view.stretch?.count ?? view.technical?.count ?? 0]);
}

const stamp = (index) => `2026-09-27T12:00:0${index}Z`;
const history = (index) => ({ history_id: `progress:${index}`, history_position: { source: 'progress', offset: index }, ts: stamp(index) });
const expected = [
    [['Reading the consumer.', 'Hello'], 1],
    [[' wor'], 0],
    [['ld', 'Reading the consumer.', 'Done.'], 4],
];

test('Unicode fragment cuts keep code point offsets across overlap, batching and history reload', () => {
    const records = [
        activity(0, 1, [hello(1, 1, '😀', [])]),
        activity(0, 3, [hello(1, 3, '😀ok🛰️', [[2, 1], [3, 3]])]),
        activity(1, 4, [hello(2, 3, 'ok🛰️', [[3, 2]]), say(4, '😀')]),
    ];
    const expectedUnicode = [[['😀'], 0], [['ok🛰️'], 0], [['😀'], 0]];
    const live = { items: [] };
    records.forEach((record, index) => {
        const view = summarizeChatLiveEvent(frame(record));
        updateLiveTimelineItem(live, view, { ts: stamp(index), rawTs: stamp(index), syntheticKey: view.dedupeKey, headline: view.headline });
    });
    assert.deepEqual(visible(live), expectedUnicode, 'no lone UTF-16 surrogate from a Python code point offset');
    for (const order of [[0, 1, 2], [2, 0, 1]]) {
        const replay = { items: [] };
        for (const index of order) {
            mergeHistoricalTimelineItem(replay, summarizeChatLiveEvent(frame(records[index])), history(index), stamp(index));
        }
        assert.deepEqual(visible(replay), expectedUnicode, `Unicode reload order ${order}`);
    }
    const otherBatching = { items: [
        { activity: activity(0, 2, [hello(1, 2, '😀ok', [[2, 1]])]) },
        { activity: activity(0, 4, [hello(1, 3, '😀ok🛰️', [[2, 1], [3, 3]]), say(4, '😀')]) },
    ] };
    assert.deepEqual(visible(otherBatching), [[['😀ok'], 0], [['🛰️', '😀'], 0]],
        'batching changes row boundaries, never words or durable event identity');
});

test('identity budgets disclose repeated previews and inexact technical overlap with their source', () => {
    const bounded = activity(0, 1003, [hello(1, 1002, 'abc', [[2, 1]])], {
        technical: { count: 501, labels: [['Read', 501]], seqs_truncated: true },
    });
    bounded.parts[0].cuts_truncated = true;
    const record = { items: [{ activity: activity(0, 1001, [say(1, 'ab')]) }, { activity: bounded }] };
    reconcileDelegatedItems(record);
    const view = delegatedLineView(record.items[1]);
    assert.equal(view.parts[0].repeats, true);
    assert.equal(view.technical.inexact, true);
    const html = delegatedActivityBodyHtml(view, { expanded: true });
    assert.match(html, /May repeat text shown above/);
    assert.match(html, /may include events counted above/);
    assert.match(html, /Full source · seq 1–1003/);
});

test('each (run, seq) is shown once across overlapping boundaries; equal words at distinct seqs stay distinct', () => {
    const live = { items: [] };
    journal().forEach((record, index) => {
        const view = summarizeChatLiveEvent(frame(record));
        updateLiveTimelineItem(live, view, { ts: stamp(index), rawTs: stamp(index), syntheticKey: view.dedupeKey, headline: view.headline });
    });
    assert.deepEqual(visible(live), expected,
        'the re-read shows only the unseen fragment of the crossing utterance, the new words and its 4 new technical events');
    const third = delegatedLineView(live.items[2]);
    assert.equal(third.parts[0].continued, true);
    assert.equal(third.technical.partial, true);
    assert.match(delegatedActivityBodyHtml(third), /4 technical events since seq 7 \(1 of this observation's shown above; kinds cover all 5\)/);
    assert.match(delegatedActivityBodyHtml(third), /Continues an utterance whose beginning is shown above\./);

    // A re-delivery of an already shown range shows nothing twice.
    const again = activity(4, 5, journal()[1].parts);
    const view = summarizeChatLiveEvent(frame(again));
    updateLiveTimelineItem(live, view, { ts: stamp(5), rawTs: stamp(5), syntheticKey: view.dedupeKey, headline: view.headline });
    assert.equal(live.items.length, 4, 'it is its own source record');
    assert.equal(delegatedLineView(live.items[3]).hidden, true, 'and every seq in it was shown above');
    assert.match(buildTimelineItemHtml(live.items[3], { expandedLineKeys: new Set() }), /data-delegated-folded hidden/);
    assert.deepEqual(visible(live), expected);
});

test('live, reload (any page order) and reconnect agree on what is shown', () => {
    const live = { items: [] };
    journal().forEach((record, index) => {
        const view = summarizeChatLiveEvent(frame(record));
        updateLiveTimelineItem(live, view, { ts: stamp(index), rawTs: stamp(index), syntheticKey: view.dedupeKey, headline: view.headline });
    });
    assert.deepEqual(visible(live), expected);
    for (const order of [[0, 1, 2], [2, 1, 0], [1, 2, 0]]) {
        const replay = { items: [] };
        for (const index of order) {
            mergeHistoricalTimelineItem(replay, summarizeChatLiveEvent(frame(journal()[index], { ts: stamp(index) })), history(index), stamp(index));
        }
        assert.deepEqual(visible(replay), expected, `history order ${order}`);
    }
    journal().forEach((record, index) => {
        mergeHistoricalTimelineItem(live, summarizeChatLiveEvent(frame(record, { ts: stamp(index) })), history(index), stamp(index));
    });
    assert.equal(live.items.length, 3, 'a reconnect adopts the live rows instead of drawing them again');
    assert.deepEqual(live.items.map((item) => item.historyId), ['progress:0', 'progress:1', 'progress:2']);
    assert.deepEqual(visible(live), expected);
    // A released older page leaves the later rows to show what they alone now carry,
    // saying where an utterance repeats what a remaining row shows.
    live.items = live.items.slice(1);
    assert.deepEqual(visible(live), [[[' wor'], 0], [['Reading the consumer.', 'Hello world', 'Reading the consumer.', 'Done.'], 5]]);
    assert.equal(delegatedLineView(live.items[1]).parts[1].repeats, true);
});

test('a silent stretch reads as one row; words, other rows or another run end it', () => {
    const records = [
        activity(0, 1, [say(1, 'Starting.')]),
        activity(1, 3, [], { technical: tech([[2, 3]]) }),
        activity(3, 6, [], { technical: tech([[4, 6]], 'Bash') }),
        activity(6, 7, [], { technical: tech([[7, 7]]) }),
        activity(7, 8, [say(8, 'Found it.')]),
        activity(8, 9, [], { technical: tech([[9, 9]], 'Grep') }),
    ];
    const live = { items: [] };
    const updates = records.map((record, index) => {
        const view = summarizeChatLiveEvent(frame(record));
        return updateLiveTimelineItem(live, view, { ts: stamp(index), rawTs: stamp(index), syntheticKey: view.dedupeKey, headline: view.headline }).timelineUpdate;
    });
    assert.deepEqual(updates, ['append', 'append', 'render', 'render', 'append', 'append'],
        'a folded row re-renders its stretch head instead of appending a line');
    assert.deepEqual(visible(live), [[['Starting.'], 0], [[], 6], [['Found it.'], 0], [[], 1]]);
    const head = delegatedLineView(live.items[1]);
    assert.equal(delegatedHeadline(head), 'claude · a01 · technical activity · run-1');
    assert.match(delegatedActivityBodyHtml(head), /6 technical events since seq 2 · Read ×3 · Bash ×3/);
    const detail = delegatedActivityBodyHtml(head, { expanded: true });
    assert.equal(detail.match(/class="chat-delegated-source"/g).length, 3, 'each folded observation keeps its own source download');
    assert.match(detail, /Full sources: <a class="chat-delegated-source" href="[^"]+" download="[^"]+" title="2 events · 10 bytes \(JSONL\)">seq 2–3<\/a> · /);
    assert.equal(live.items.length, 6, 'every source record stays an item');
    // Another row between two silent observations ends the stretch.
    const split = { items: [live.items[1], { headline: 'child note' }, live.items[2]] };
    assert.deepEqual(visible(split), [[[], 2], [[], 3]]);
    const other = { items: [live.items[1], { activity: { ...activity(1, 3, [], { technical: tech([[2, 3]]) }), run_id: 'run-2' } }] };
    assert.deepEqual(visible(other), [[[], 2], [[], 2]], 'another run is its own row');
    const long = { items: Array.from({ length: 45 }, (_, i) => ({ activity: activity(i, i + 1, [], { technical: tech([[i + 1, i + 1]]) }) })) };
    assert.deepEqual(visible(long), [[[], 40], [[], 5]], 'a very long silent stretch opens a new row after 40 observations');
});

test('a failed exact read is a provisional window that folds once the exact events arrive; legacy rows stay', () => {
    const provisional = { v: 1, task_id: 'child', run_id: 'run-1', after_seq: 4, through_seq: 9,
        source: { kind: 'timeline_window', reason: 'stream_unavailable:busy', provisional: true, rows: 1 },
        parts: [{ kind: 'message', actor: 'claude/a01', text: 'Reading the consumer.' }],
        gaps: [{ after_seq: 4, through_seq: 9, reason: 'stream_unavailable:busy' }] };
    const record = { items: [{ activity: activity(0, 4, [say(2, 'Reading the consumer.')]) }, { activity: provisional }] };
    reconcileDelegatedItems(record);
    const open = delegatedActivityBodyHtml(delegatedLineView(record.items[1]));
    assert.match(open, /Bounded timeline window shown meanwhile: the exact read failed \(stream_unavailable:busy\)\./);
    assert.match(open, /Not shown yet: seq 5–9 \(stream_unavailable:busy\)/);
    record.items.push({ activity: activity(4, 9, [say(8, 'Reading the consumer.')]) });
    reconcileDelegatedItems(record);
    const folded = delegatedLineView(record.items[1]);
    assert.equal(folded.superseded, true);
    assert.deepEqual(folded.parts, [], 'its window words are not shown beside the exact ones');
    assert.match(delegatedActivityBodyHtml(folded), /The exact read of seq 5–9 failed here \(stream_unavailable:busy\); a later row shows those exact events\./);
    assert.deepEqual(delegatedLineView(record.items[2]).parts.map((shown) => shown.text), ['Reading the consumer.']);
    // A gap that a later exact record read is no longer an open disclosure.
    const gapped = { items: [{ activity: activity(0, 9, [say(2, 'a')], { source: { kind: 'run_events', read_through: 4, events: 4 },
        gaps: [{ after_seq: 4, through_seq: 9, reason: 'read_bound:time' }] }) }, { activity: activity(4, 9, [say(6, 'b')]) }] };
    reconcileDelegatedItems(gapped);
    assert.deepEqual(delegatedLineView(gapped.items[0]).gaps, []);
    // Rows without identity (an engine without the stream) are never folded by text or time.
    const windowRow = { v: 1, task_id: 'child', run_id: 'run-1', after_seq: 0, through_seq: 3,
        source: { kind: 'timeline_window', reason: 'stream_not_listed', rows: 1 }, parts: [{ kind: 'message', actor: '', text: 'same' }] };
    const legacy = { items: [{ activity: windowRow }, { activity: { ...windowRow } }] };
    assert.deepEqual(visible(legacy), [[['same'], 0], [['same'], 0]]);
});

test('a later permanent journal refusal repaints earlier rows without changing their raw facts', () => {
    const earlier = { ...activity(0, 2, [say(1, 'first')]),
        source: { kind: 'run_events', read_through: 2, events: 2 },
        gaps: [] };
    const refused = { ...activity(2, 3, []),
        source: { kind: 'run_events', read_through: 3, events: 1 },
        gaps: [{ after_seq: 2, through_seq: 3, reason: 'stream_error:run_expired_by_retention',
            retryable: false, final: true }] };
    const record = { items: [{ activity: earlier }, { activity: refused }] };
    reconcileDelegatedItems({ items: [{ activity: structuredClone(earlier) }] });
    const before = { items: [{ activity: structuredClone(earlier) }] };
    reconcileDelegatedItems(before);
    const priorSignature = before.items[0].delegatedSignature;
    reconcileDelegatedItems(record);
    const first = delegatedLineView(record.items[0]);
    const html = delegatedActivityBodyHtml(first, { expanded: true });
    assert.notEqual(record.items[0].delegatedSignature, priorSignature,
        'the later permanent gap changes the current run projection');
    assert.match(html, /originals can no longer be read/);
    assert.equal(record.items[0].activity.source.ref, undefined,
        'the projection did not rewrite the earlier raw source facts');
});

test('an earlier gap that promised another read follows the journal refused since', () => {
    const tried = { ...activity(0, 2, [say(1, 'first')]), source: { kind: 'run_events', read_through: 1, events: 1 },
        gaps: [{ after_seq: 1, through_seq: 2, reason: 'stream_error:run_event_not_found', retryable: true, final: true }] };
    const refused = { ...activity(1, 3, []), source: { kind: 'timeline_window', reason: 'stream_error:run_expired_by_retention',
        retryable: false, rows: 0 }, gaps: [{ after_seq: 1, through_seq: 3, reason: 'stream_error:run_expired_by_retention',
        retryable: false, final: true }] };
    const record = { items: [{ activity: tried }, { activity: refused }] };
    reconcileDelegatedItems(record);
    const html = delegatedActivityBodyHtml(delegatedLineView(record.items[0]), { expanded: true });
    assert.match(html, /its journal can no longer be read, so waiting again will not show the rest/);
    assert.doesNotMatch(html, /tries again|reads them/, 'no row promises a read its journal can no longer give');
    assert.equal(record.items[0].activity.gaps[0].retryable, true, 'the row keeps its own historical fact');
});

test('the host problem label translates and escapes without rewriting executor evidence', async () => {
    const { applyPayload, currentPayload, CODE_PREFIX } = await import('../modules/i18n.js');
    const prior = currentPayload();
    try {
        applyPayload({ language: 'ru', english: false, entries: { [CODE_PREFIX + 'task.delegated.problem']: { text: '<Проблема>' } } });
        const html = delegatedActivityBodyHtml(delegatedLineView({ activity: burst }));
        assert.match(html, /chat-delegated-problem">&lt;Проблема&gt;: claude · a01 · Bash: exit 1: 2 failed/);
    } finally { applyPayload(prior); }
});
