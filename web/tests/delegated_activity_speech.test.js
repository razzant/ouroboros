// One engine-keyed utterance in a task card (Claudexor speech contract v2): its draft
// pieces are provisional, its complete text replaces them at the utterance's first
// visible row ("Updated" when the producer saw them differ), an abandoned draft is
// marked, equal words under other keys stay, and the typed final answer is its own
// disclosure, folded once the same run and actor spoke. Live arrival, history pages in
// any order and a reconnect run the same projection, so they agree.
import assert from 'node:assert/strict';
import test from 'node:test';
import { summarizeChatLiveEvent } from '../modules/log_events.js';
import {
    appendDelegatedItem, delegatedActivityBodyHtml, delegatedActivityView, delegatedLineView, reconcileDelegatedItems,
} from '../modules/delegated_activity.js';
import { buildTimelineItemHtml } from '../modules/chat_activity.js';
import { mergeHistoricalTimelineItem } from '../modules/chat_history_replay.js';
import { updateLiveTimelineItem } from '../modules/chat_render_batch.js';

globalThis.document ??= {
    createElement: () => {
        let text = '';
        return {
            set textContent(value) { text = String(value); },
            get innerHTML() { return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); },
        };
    },
};

const SHA = 'cd'.repeat(32);
const K1 = 'claude:ses-1:1:0';
const K2 = 'claude:ses-1:2:0';

function activity(after, through, parts, extra = {}, run = 'run-1') {
    return { v: 1, task_id: 'child', run_id: run, after_seq: after, through_seq: through,
        source: { kind: 'run_events', read_through: through, events: through - after, ref: { kind: 'task_source',
            root: 'artifact_store', path: `source_handles/delegated_activity/${run}-${after + 1}-${through}-${SHA}.jsonl`,
            size: 10, sha256: SHA } },
        parts, ...extra };
}
const draft = (seq, last, text, key = K1, extra = {}) => ({ kind: 'message', actor: 'claude/a01', text, seq, last_seq: last,
    speech: { key, op: 'fragment' }, delta: true, fragments: last - seq + 1,
    ...(last > seq ? { cuts: Array.from({ length: last - seq }, (_, i) => [seq + i + 1, i + 1]) } : {}), ...extra });
const complete = (seq, last, text, key = K1, extra = {}) => ({ kind: 'message', actor: 'claude/a01', text, seq, last_seq: last,
    speech: { key, op: 'complete' }, fragments: last - seq, ...extra });
const final = (seq, text, actor = 'claude/a01') => ({ kind: 'message', actor, text, seq, last_seq: seq, final: true });
const tool = (seq) => ({ count: 1, labels: [['Edit', 1]], seqs: [[seq, seq]], recent: [{ seq, label: 'Edit', actor: 'claude/a01' }] });
const frame = (record, extra = {}) => ({ type: 'send_message', is_progress: true, task_id: 'child', narration: false,
    content: `💬 🛰 delegated run ${record.run_id}`, delegated_activity: record, ...extra });
const stamp = (index) => `2026-10-27T14:05:1${index}Z`;
const history = (index) => ({ history_id: `progress:${index}`, history_position: { source: 'progress', offset: index }, ts: stamp(index) });

function live(records) {
    const card = { items: [] };
    const updates = records.map((record, index) => {
        const view = summarizeChatLiveEvent(frame(record));
        return updateLiveTimelineItem(card, view, { ts: stamp(index), rawTs: stamp(index), syntheticKey: view.dedupeKey, headline: view.headline });
    });
    return { card, updates };
}

function replay(records, order) {
    const card = { items: [] };
    for (const index of order) {
        mergeHistoricalTimelineItem(card, summarizeChatLiveEvent(frame(records[index], { ts: stamp(index) })), history(index), stamp(index));
    }
    return card;
}

/** What each visible row says: its words (with the answer state) and its technical count. */
function visible(card) {
    reconcileDelegatedItems(card);
    return card.items.filter((item) => item.activity).map((item) => delegatedLineView(item)).filter((view) => !view.hidden)
        .map((view) => [view.parts.map((shown) => (shown.answer ? `${shown.answer.folded ? 'folded' : 'open'} answer: ` : '')
            + shown.text + (shown.updated ? ' [updated]' : '') + (shown.abandoned ? ' [abandoned]' : '')),
        view.stretch?.count ?? view.technical?.count ?? 0]);
}

const PARAGRAPH = 'Both duplicates come from one feed item: I will key them by GUID.';

test('a keyed stream completed inside one record reads once; an updated text says where its draft is', () => {
    const same = activity(0, 4, [complete(1, 4, PARAGRAPH)]);
    const html = delegatedActivityBodyHtml(delegatedLineView({ activity: same }));
    assert.equal(html.match(/chat-delegated-message/g).length, 1);
    assert.ok(!html.includes('Updated'), html);
    const changed = activity(0, 4, [complete(1, 4, 'the cat', K1, { updated: true })]);
    const note = delegatedActivityBodyHtml(delegatedLineView({ activity: changed }));
    assert.match(note, /<div class="chat-delegated-message">(<p>)?the cat(<\/p>)?<div class="chat-delegated-note">Updated\. The streamed draft is in <a class="chat-delegated-source" href="\/api\/tasks\/child\/artifacts\/run-1-1-4-[^"]+" download="[^"]+">the full source<\/a>\.<\/div><\/div>/,
        'the approved note: the row\'s own note and source-link classes, inside the message');
    const windowed = { ...changed, source: { kind: 'timeline_window', reason: 'stream_not_listed', rows: 2 } };
    assert.match(delegatedActivityBodyHtml(delegatedLineView({ activity: windowed })), /Updated\. The streamed draft stays on the run(&#39;|')s journal\./);
});

// A split poll: the draft's first pieces, then the rest with the complete text and an Edit.
const split = (text = PARAGRAPH, extra = {}) => [
    activity(0, 3, [draft(2, 3, text.slice(0, 12))], { technical: tool(1) }),
    activity(3, 7, [complete(4, 6, text, K1, extra)], { technical: tool(7) }),
];

test('a split poll shows the complete text in place of the first row\'s draft and folds the completing row', () => {
    const { card, updates } = live(split());
    assert.deepEqual(updates.map((update) => update.timelineUpdate), ['append', 'render'],
        'the completing record re-renders the row that holds the draft');
    assert.deepEqual(visible(card), [[[PARAGRAPH], 1], [[], 1]]);
    const owner = delegatedLineView(card.items[0]);
    assert.equal(owner.parts[0].from, card.items[1].activity, 'the words come from the record that holds them');
});

test('a same-length correction still re-renders the row that shows the draft', () => {
    const records = [activity(0, 2, [draft(1, 2, 'teh cat')]), activity(2, 3, [complete(3, 3, 'the cat', K1, { updated: true })])];
    const card = { items: [] };
    const first = summarizeChatLiveEvent(frame(records[0]));
    appendDelegatedItem(card, first, { ts: stamp(0), rawTs: stamp(0), syntheticKey: first.dedupeKey, headline: first.headline });
    const before = card.items[0].delegatedSignature;
    const second = summarizeChatLiveEvent(frame(records[1]));
    const update = appendDelegatedItem(card, second, { ts: stamp(1), rawTs: stamp(1), syntheticKey: second.dedupeKey, headline: second.headline });
    assert.equal(update.timelineUpdate, 'render');
    assert.notEqual(card.items[0].delegatedSignature, before);
    assert.match(buildTimelineItemHtml(card.items[0], { expandedLineKeys: new Set(), groupId: 'child' }), /(<p>)?the cat(<\/p>)?<div class="chat-delegated-note">Updated\./);
    assert.match(buildTimelineItemHtml(card.items[1], { expandedLineKeys: new Set(), groupId: 'child' }), /data-delegated-folded hidden/,
        'a row whose only words were the completion folds');
});

test('live, every history page order and a reconnect agree; an older page moves the utterance to its first row', () => {
    const records = split(PARAGRAPH, { updated: true });
    const expected = [[[`${PARAGRAPH} [updated]`], 1], [[], 1]];
    assert.deepEqual(visible(live(records).card), expected);
    for (const order of [[0, 1], [1, 0]]) assert.deepEqual(visible(replay(records, order)), expected, `history order ${order}`);
    const newest = replay(records, [1]);
    assert.deepEqual(visible(newest), [[[`${PARAGRAPH} [updated]`], 1]], 'only the completing page: its row shows the words');
    mergeHistoricalTimelineItem(newest, summarizeChatLiveEvent(frame(records[0], { ts: stamp(0) })), history(0), stamp(0));
    assert.deepEqual(visible(newest), expected, 'the older page takes the utterance to where it began');
    const reconnect = live(records).card;
    records.forEach((record, index) => mergeHistoricalTimelineItem(reconnect, summarizeChatLiveEvent(frame(record, { ts: stamp(index) })), history(index), stamp(index)));
    assert.equal(reconnect.items.length, 2);
    assert.deepEqual(visible(reconnect), expected);
});

test('equal words under another key, actor or run stay their own utterance', () => {
    const words = 'Re-running the import tests.';
    const records = [
        activity(0, 3, [complete(1, 3, words, K1)]),
        activity(3, 6, [complete(4, 6, words, K2)]),
        activity(6, 9, [{ ...complete(7, 9, words, K1), actor: 'claude/a02' }]),
        activity(0, 3, [complete(1, 3, words, K1)], {}, 'run-2'),
        activity(9, 11, [{ kind: 'message', actor: 'claude/a01', text: words, seq: 10, last_seq: 10 }]),
    ];
    assert.deepEqual(visible(live(records).card), records.map(() => [[words], 0]));
});

test('an abandoned draft keeps its words, is marked once and never completes; the retry stays its own', () => {
    const records = [
        activity(0, 2, [draft(1, 2, 'Checking the')]),
        activity(2, 7, [complete(4, 7, 'Checking the file first.', K2)],
            { speech_abandoned: [{ actor: 'claude/a01', key: K1, seq: 3 }], technical: { count: 1, labels: [['status', 1]], seqs: [[3, 3]], recent: [] } }),
    ];
    const expected = [[['Checking the [abandoned]'], 0], [['Checking the file first.'], 1]];
    assert.deepEqual(visible(live(records).card), expected);
    assert.deepEqual(visible(replay(records, [1, 0])), expected);
    assert.match(delegatedActivityBodyHtml(delegatedLineView(live(records).card.items[0])),
        /(<p>)?Checking the(<\/p>)?<div class="chat-delegated-note">Abandoned: the executor never completed these words\.<\/div>/);
    const same = activity(0, 3, [draft(1, 2, 'Checking the')], { speech_abandoned: [{ actor: 'claude/a01', key: K1, seq: 3 }] });
    assert.deepEqual(visible({ items: [{ activity: same }] }), [[['Checking the [abandoned]'], 0]], 'an abandon in the same record');
});

test('the typed final answer is its own disclosure: folded after the same actor\'s words, open when sole', () => {
    const records = [
        activity(0, 4, [complete(1, 3, PARAGRAPH), final(4, PARAGRAPH)]),
        activity(0, 1, [final(1, 'The feed validates.')], {}, 'run-2'),
        activity(4, 6, [{ kind: 'message', actor: 'codex/a02', text: 'Reviewing.', seq: 5, last_seq: 5 }, final(6, 'Looks fine.')]),
    ];
    assert.deepEqual(visible(live(records).card), [
        [[PARAGRAPH, `folded answer: ${PARAGRAPH}`], 0], [['open answer: The feed validates.'], 0],
        [['Reviewing.', 'folded answer: Looks fine.'], 0]], 'run-1\'s claude spoke above before its answer');
    const html = delegatedActivityBodyHtml(delegatedLineView(live(records.slice(0, 1)).card.items[0]));
    assert.match(html, /<details class="chat-delegated-answer"><summary>Answer<\/summary><div class="chat-delegated-message">(<p>)?Both duplicates/,
        'the Thinking look, folded, the words whole inside');
    assert.match(delegatedActivityBodyHtml(delegatedLineView({ activity: records[1] })), /<details class="chat-delegated-answer" open><summary>Answer<\/summary>/);
    const another = activity(0, 2, [{ kind: 'message', actor: 'codex/a02', text: 'Starting.', seq: 1, last_seq: 1 }, final(2, 'Done.')]);
    assert.deepEqual(visible({ items: [{ activity: another }] }), [[['Starting.', 'open answer: Done.'], 0]], 'another actor\'s words do not fold it');
    // An answer read in a later record folds behind the words of an earlier one, live and from history in any order.
    const later = [activity(0, 3, [complete(1, 3, PARAGRAPH)]), activity(3, 4, [final(4, PARAGRAPH)])];
    const expected = [[[PARAGRAPH], 0], [[`folded answer: ${PARAGRAPH}`], 0]];
    assert.deepEqual(visible(live(later).card), expected);
    for (const order of [[0, 1], [1, 0]]) assert.deepEqual(visible(replay(later, order)), expected);
    assert.deepEqual(visible(replay(later, [1])), [[[`open answer: ${PARAGRAPH}`], 0]], 'alone in view it is the run\'s words');
    const summary = delegatedActivityView(frame(records[0]));
    assert.equal(summary.fullBody.split(PARAGRAPH).length - 1, 2, 'the logs text keeps the answer\'s own words beside the reply');
});

test('a provisional window naming the key yields the utterance to the exact record that covers it', () => {
    const window = { v: 1, task_id: 'child', run_id: 'run-1', after_seq: 3, through_seq: 7,
        source: { kind: 'timeline_window', reason: 'stream_unavailable:busy', provisional: true, rows: 1 },
        parts: [{ kind: 'message', actor: 'claude/a01', text: PARAGRAPH, speech: { key: K1, op: 'complete' } }],
        gaps: [{ after_seq: 3, through_seq: 7, reason: 'stream_unavailable:busy' }] };
    const records = [activity(0, 3, [draft(2, 3, 'Both duplic')]), window];
    assert.deepEqual(visible(live(records).card).map(([words]) => words), [[PARAGRAPH], []],
        'before the exact read the window\'s complete text is the best known words');
    const exact = [...records, activity(3, 7, [complete(4, 6, PARAGRAPH)], { technical: tool(7) })];
    const card = live(exact).card;
    assert.deepEqual(visible(card), [[[PARAGRAPH], 0], [[], 0], [[], 1]]);
    assert.equal(delegatedLineView(card.items[0]).parts[0].from, card.items[2].activity, 'the superseded window owns nothing');
    assert.equal(delegatedLineView(card.items[1]).superseded, true);
});

// A draft streamed across three polls before its complete text exists.
const WHOLE = 'Both duplicates come from one feed item.';
const across = [
    activity(0, 3, [draft(2, 3, 'Both duplic')], { technical: tool(1) }),
    activity(3, 7, [draft(4, 6, 'ates come from ')], { technical: tool(7) }),
    activity(7, 9, [draft(8, 9, 'one feed item.')]),
];

test('a keyed draft streamed across polls is one utterance where it began, live, from history and on reconnect', () => {
    const { card, updates } = live(across);
    assert.deepEqual(updates.map((update) => update.timelineUpdate), ['append', 'render', 'render'],
        'each new piece re-renders the row that holds the utterance');
    const expected = [[[WHOLE], 1], [[], 1]];
    assert.deepEqual(visible(card), expected, 'the last row held only words of the draft: it folds');
    assert.equal(delegatedActivityBodyHtml(delegatedLineView(card.items[0])).match(/chat-delegated-message/g).length, 1);
    for (const order of [[0, 1, 2], [2, 1, 0], [1, 2, 0]]) assert.deepEqual(visible(replay(across, order)), expected, `history order ${order}`);
    const reconnect = live(across).card;
    across.forEach((record, index) => mergeHistoricalTimelineItem(reconnect, summarizeChatLiveEvent(frame(record, { ts: stamp(index) })), history(index), stamp(index)));
    assert.equal(reconnect.items.length, 3);
    assert.deepEqual(visible(reconnect), expected);
    const done = [...across, activity(9, 10, [complete(10, 10, WHOLE)])];
    const finished = live(done).card;
    assert.deepEqual(visible(finished), expected, 'its complete text takes the same place');
    assert.match(delegatedActivityBodyHtml(delegatedLineView(finished.items[0]), { expanded: true }),
        /Full sources: <a [^>]*>seq 1–3<\/a> · <a [^>]*>seq 4–7<\/a> · <a [^>]*>seq 8–9<\/a> · <a [^>]*>seq 10–10<\/a>/,
        'the draft\'s and the complete text\'s records stay one click away though their rows folded');
});

test('an abandoned or never completed draft across polls is one utterance, marked once when abandoned', () => {
    const records = [
        activity(0, 2, [draft(1, 2, 'Checking')]),
        activity(2, 4, [draft(3, 3, ' the')], { speech_abandoned: [{ actor: 'claude/a01', key: K1, seq: 4 }],
            technical: { count: 1, labels: [['status', 1]], seqs: [[4, 4]], recent: [] } }),
    ];
    const expected = [[['Checking the [abandoned]'], 0], [[], 1]];
    assert.deepEqual(visible(live(records).card), expected);
    assert.deepEqual(visible(replay(records, [1, 0])), expected);
    const unfinished = [records[0], { ...records[1], speech_abandoned: undefined }];
    assert.deepEqual(visible(live(unfinished).card), [[['Checking the'], 0], [[], 1]], 'no statement: no mark, still one utterance');
});

test('a re-read before the complete adds only its unseen fragments to the utterance', () => {
    // One character per fragment: the re-read's unseen fragments start at seq 4, offset 2.
    const records = [activity(0, 3, [draft(2, 3, 'Ch')]), activity(0, 5, [draft(2, 5, 'Chec')])];
    assert.deepEqual(visible(live(records).card), [[['Chec'], 0]], 'seq identity: each fragment once, where the words began');
    assert.deepEqual(visible(replay(records, [1, 0])), [[['Chec'], 0]]);
});

test('pieces are never joined across events no record read, nor by a window row that has no seq', () => {
    const first = activity(0, 3, [draft(2, 3, 'Both duplic')]);
    const unread = { ...first, gaps: [{ after_seq: 3, through_seq: 5, reason: 'read_bound:time' }] };
    assert.deepEqual(visible(live([unread, activity(5, 8, [draft(6, 8, 'one feed')])]).card),
        [[['Both duplic'], 0], [['one feed'], 0]], 'seq 4–5 were never read: the words between are unknown');
    const windowed = { v: 1, task_id: 'child', run_id: 'run-1', after_seq: 3, through_seq: 7,
        source: { kind: 'timeline_window', reason: 'stream_not_listed', rows: 1 },
        parts: [{ kind: 'message', actor: 'claude/a01', text: 'ates come', speech: { key: K1, op: 'fragment' }, delta: true }] };
    assert.deepEqual(visible(live([first, windowed]).card), [[['Both duplic'], 0], [['ates come'], 0]]);
});

test('a draft is never joined across an event its record names as not read whole', () => {
    const hole = { gaps: [{ after_seq: 1, through_seq: 2, reason: 'not_served' }], technical: tool(4) };
    const across = [activity(0, 1, [draft(1, 1, 'the')]), activity(1, 4, [draft(3, 3, ' cat')], hole)];
    assert.deepEqual(visible(live(across).card), [[['the'], 0], [[' cat'], 1]], 'seq 2 was lost: the words stay apart');
    assert.match(delegatedActivityBodyHtml(delegatedLineView(live(across).card.items[1])),
        /Not read whole: seq 2–2 \(not_served\)\. An utterance around it may lack words/);
    for (const order of [[0, 1], [1, 0]]) assert.deepEqual(visible(replay(across, order)), [[['the'], 0], [[' cat'], 1]]);
    const one = [activity(0, 4, [draft(1, 1, 'the'), draft(3, 3, ' cat')], hole)];
    assert.deepEqual(visible(live(one).card), [[['the', ' cat'], 1]], 'in one record too: two pieces');
    // A later read that serves the lost event covers it: its gap line goes, and its words keep their own row
    // rather than joining behind words that came after them.
    const filled = live([...across, activity(1, 2, [draft(2, 2, ' big')])]).card;
    assert.deepEqual(visible(filled), [[['the'], 0], [[' cat'], 1], [[' big'], 0]]);
    assert.doesNotMatch(delegatedActivityBodyHtml(delegatedLineView(filled.items[1])), /Not read whole/);
});

test('a fragment holding only the space between words joins where it was read; alone it makes no row', () => {
    for (const space of [' ', '\n\n']) {
        const records = [activity(0, 1, [draft(1, 1, 'the')]), activity(1, 2, [draft(2, 2, space)]),
            activity(2, 4, [draft(3, 3, 'cat')], { technical: tool(4) })];
        const expected = [[[`the${space}cat`], 0], [[], 1]];
        assert.deepEqual(visible(live(records).card), expected, 'the read space keeps the words apart');
        for (const order of [[2, 1, 0], [1, 0, 2]]) assert.deepEqual(visible(replay(records, order)), expected, `history order ${order}`);
    }
    // The space alone so far: no empty paragraph, no summary words, and no voice that would fold the run's answer.
    const blank = activity(0, 1, [draft(1, 1, ' ')]);
    assert.deepEqual(visible(live([blank]).card), []);
    assert.equal(delegatedActivityView(frame(blank)).body, '');
    assert.deepEqual(visible(live([blank, activity(1, 2, [final(2, 'Done.')])]).card), [[['open answer: Done.'], 0]]);
    assert.deepEqual(visible(live([blank, activity(1, 2, [draft(2, 2, 'cat')])]).card), [[[' cat'], 0]],
        'words after it join it where the utterance began');
});

test('an exact re-read naming a hole still supersedes the window it replaces, whose bounded words yield', () => {
    const bounded = 'Both duplicates come from one feed item: I will…';
    const window = { v: 1, task_id: 'child', run_id: 'run-1', after_seq: 3, through_seq: 7,
        source: { kind: 'timeline_window', reason: 'stream_unavailable:busy', provisional: true, rows: 2 },
        parts: [{ kind: 'message', actor: 'claude/a01', text: bounded, speech: { key: K1, op: 'complete' } },
            { kind: 'message', actor: 'claude/a01', text: 'Unkeyed words.' }],
        gaps: [{ after_seq: 3, through_seq: 7, reason: 'stream_unavailable:busy' }] };
    const unkeyed = { kind: 'message', actor: 'claude/a01', text: 'Unkeyed words.', seq: 7, last_seq: 7 };
    const exact = activity(3, 7, [complete(4, 6, PARAGRAPH), unkeyed], { gaps: [{ after_seq: 4, through_seq: 5, reason: 'unreadable' }] });
    const records = [activity(0, 3, [draft(2, 3, 'Both duplic')]), window, exact];
    const expected = [[[PARAGRAPH], 0], [[], 0], [['Unkeyed words.'], 0]];
    const card = live(records).card;
    assert.deepEqual(visible(card), expected, 'the exact words once each; the window row says it was superseded');
    for (const order of [[2, 1, 0], [1, 2, 0]]) assert.deepEqual(visible(replay(records, order)), expected, `history order ${order}`);
    assert.equal(delegatedLineView(card.items[1]).superseded, true);
    assert.doesNotMatch(card.items.map((item) => delegatedActivityBodyHtml(delegatedLineView(item))).join(''), /Not shown yet: seq 4–7/,
        'the read range is no longer "not shown"');
    assert.match(delegatedActivityBodyHtml(delegatedLineView(card.items[2])), /Not read whole: seq 5–5 \(unreadable\)/,
        'the hole keeps its own line');
    assert.match(delegatedActivityBodyHtml(delegatedLineView(card.items[0]), { expanded: true }),
        /Full sources: <a [^>]*run-1-1-3-[^>]*>seq 1–3<\/a> · <a [^>]*run-1-4-7-[^>]*>seq 4–7<\/a>/, 'the retained sources stay reachable');
    // An exact read that reached only part of the window's range leaves the window up, but its complete text is the one shown.
    const partial = activity(3, 5, [complete(4, 5, PARAGRAPH)], { gaps: [{ after_seq: 5, through_seq: 7, reason: 'read_bound:time' }] });
    const early = live([records[0], window, partial]).card;
    assert.equal(delegatedLineView(early.items[1]).superseded, false, 'seq 6–7 are not read yet');
    assert.deepEqual(visible(early)[0], [[PARAGRAPH], 0], 'the exact complete text, not the window\'s bounded one');
});

// A poll of 40 other messages and the draft's ' big ' (seq 42): the preview's part count leaves the piece out.
const notes = (count, from = 2) => Array.from({ length: count }, (_, i) => ({ kind: 'message', actor: 'claude/a01',
    text: `Note ${i}.`, seq: from + i, last_seq: from + i }));
const named = (seq, key = K1) => ({ actor: 'claude/a01', key, seq, last_seq: seq });
const words = (card) => visible(card).map(([shown]) => shown.filter((text) => !text.startsWith('Note ')).map((text) => text.trim()));

test('words are never joined across a keyed piece a record\'s preview left out; a represented one still joins', () => {
    const first = activity(0, 1, [draft(1, 1, 'the')]);
    const last = activity(42, 43, [draft(43, 43, 'cat')]);
    const left = [first, activity(1, 42, notes(40), { omitted: { message: 1 }, speech_omitted: [named(42)] }), last];
    const apart = [['the'], [], ['cat']];
    assert.deepEqual(words(live(left).card), apart, 'never "thecat": the words between were read but not shown');
    for (const order of [[2, 1, 0], [1, 2, 0], [0, 2, 1]]) assert.deepEqual(words(replay(left, order)), apart, `history order ${order}`);
    assert.match(delegatedActivityBodyHtml(delegatedLineView(live(left).card.items[1])), /Not in this row: 1 message/);
    const shown = [first, activity(1, 42, [...notes(39), draft(42, 42, ' big ')]), last];
    assert.deepEqual(words(live(shown).card), [['the big cat'], []], 'the control: every piece is in a preview');
    const done = [...left, activity(43, 44, [complete(44, 44, 'the big cat')])];
    assert.deepEqual(words(live(done).card), [['the big cat'], []], 'the complete text is authoritative');
    // Another message left out of the same poll names no piece of this utterance: its pieces still join.
    const other = [first, activity(1, 42, [draft(2, 2, ' big '), ...notes(39, 3)], { omitted: { message: 1 }, speech_omitted: [] }), last];
    assert.deepEqual(words(live(other).card), [['the big cat'], []]);
});

test('a record that cannot say which speech it left out joins nothing across its range', () => {
    const first = activity(0, 1, [draft(1, 1, 'the')]);
    const last = activity(42, 43, [draft(43, 43, 'cat')]);
    const unnamed = activity(1, 42, notes(40), { omitted: { message: 1 } });                    // an older record
    const overflow = activity(1, 42, notes(40), { omitted: { message: 41 }, speech_omitted_truncated: true,
        speech_omitted: Array.from({ length: 40 }, (_, i) => named(2 + i, `claude:ses-1:9:${i}`)) });
    for (const middle of [unnamed, overflow]) assert.deepEqual(words(live([first, middle, last]).card), [['the'], [], ['cat']]);
    // A piece whose fragments lie past its record's preview ends where its own row says so, never joined behind 'the'.
    const reread = activity(0, 3, [draft(1, 3, `${'A'.repeat(20)}\n⚠️ OMISSION NOTE: truncated at 6000 chars; original length 9000`,
        K1, { truncated: true, chars: 9000, cuts: [] })]);
    const card = live([first, reread, activity(3, 4, [draft(4, 4, 'cat')])]).card;
    assert.deepEqual(words(card), [['the'], ['cat']]);
    assert.match(delegatedActivityBodyHtml(delegatedLineView(card.items[1])), /Continues beyond the preview shown above \(up to seq 3\)/);
});

test('a joined utterance keeps each piece\'s preview bound and every source reachable from its row', () => {
    const cut = `${'A'.repeat(20)}\n⚠️ OMISSION NOTE: truncated at 6000 chars; original length 9000`;
    const records = [activity(0, 2, [draft(1, 2, cut, K1, { truncated: true, chars: 9000 })]), activity(2, 4, [draft(3, 4, 'tail.')])];
    const card = live(records).card;
    assert.equal(card.items[1].delegatedView.hidden, true, 'the second row held only words of the draft');
    const html = delegatedActivityBodyHtml(delegatedLineView(card.items[0]));
    assert.ok(html.indexOf('OMISSION NOTE') < html.indexOf('tail.'), html);
    assert.match(html, /Read it whole: <a class="chat-delegated-source" href="[^"]*run-1-1-2-[^"]+"/, 'the cut piece names its own source');
    const expanded = delegatedActivityBodyHtml(delegatedLineView(card.items[0]), { expanded: true });
    assert.match(expanded, /Full sources: <a [^>]*run-1-1-2-[^>]*>seq 1–2<\/a> · <a [^>]*run-1-3-4-[^>]*>seq 3–4<\/a>/,
        'the folded row\'s source stays one click away');
});

test('the Answer label translates through the host strings', async () => {
    const { applyPayload, currentPayload, CODE_PREFIX } = await import('../modules/i18n.js');
    const prior = currentPayload();
    try {
        applyPayload({ language: 'ru', english: false, entries: { [CODE_PREFIX + 'task.delegated.answer']: { text: 'Ответ' } } });
        assert.match(delegatedActivityBodyHtml(delegatedLineView({ activity: activity(0, 1, [final(1, 'Готово.')]) })),
            /<summary>Ответ<\/summary>/);
    } finally { applyPayload(prior); }
});
