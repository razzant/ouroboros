import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

import { askEvidence } from '../modules/plan_record_presentation.js';
import {
    mergeReviewGroup,
    planReviewGroupFromTaskDetail,
    renderReviewsSection,
} from '../modules/review_presentation.js';

// The planning organ's record as the host stores it (plan_record, schema 3), and the host's
// objection fold as the task detail projects it (plan_objections): the panel renders the fold
// and never re-derives which objections stand.
const fold = (open = {}, withdrawn = [], extra = {}) => ({ open_objections: open, withdrawn, legacy_engaged: false, ...extra });
const standing = (id, text, askId, revision, over = {}) => ({ id, text, ask_id: askId, revision, holds: true, not_adjudicated: false, legacy: false, ...over });
const answered = (seat, model, reply, extra = {}) => ({
    seat_id: seat, model, route: 'api_chat', owner_marked: true, dispatched: true, status: 'answered',
    reply: { prose: '', objections: [], withdrawn: [], ...reply }, ...extra,
});

const RECORD = {
    schema_version: 3,
    revisions: [
        { n: 1, ts: '2026-10-15T01:00:00+00:00', goal: 'Ship the deck', plan: 'outline first', body: false, affected_paths: [],
          done_when: [{ id: 'c1', claim: 'five slides', support: 'count them' }] },
        { n: 2, ts: '2026-10-15T02:00:00+00:00', goal: 'Ship the deck', plan: 'outline, then a story', body: false, affected_paths: [],
          done_when: [{ id: 'c1', claim: 'a five-slide story', support: 'read it' }] },
    ],
    asks: [
        { ask_id: 'a1', ts: '2026-10-15T01:05:00+00:00', revision: 1, message: 'First look.', host_added: [], participants: [
            answered('triad_a', 'm/a', { prose: 'The bar is low.', objections: [{ id: 'o1', text: 'five slides is not the bar' }] }),
            answered('triad_b', 'm/b', { prose: 'Fine.' }),
            { seat_id: 'triad_c', model: 'm/c', route: 'api_chat', owner_marked: true, dispatched: false, status: 'not_dispatched',
              not_dispatched_reason: 'health_skip[window_exhausted]' },
        ] },
        { ask_id: 'a2', ts: '2026-10-15T02:05:00+00:00', revision: 2, message: 'Raised the bar; o1?', host_added: [], participants: [
            answered('triad_a', 'm/a', { prose: 'Better.', withdrawn: [{ id: 'o1', why: 'revision 2 raised the bar' }] }),
            { seat_id: 'critic_x', model: 'm/x', route: 'api_chat', owner_marked: false, dispatched: true, status: 'answered',
              reply: { prose: 'No.', objections: [{ id: 'k1', text: 'wrong shape' }], withdrawn: [] } },
        ] },
    ],
};

const RECORD_FOLD = fold({ critic_x: [standing('k1', 'wrong shape', 'a2', 2, { holds: false })] },
    [{ seat_id: 'triad_a', id: 'o1', withdrawn_in: 'a2', why: 'revision 2 raised the bar' }]);

test('the plan record renders its revisions, participants, contributions, open objections and done_when', () => {
    const group = planReviewGroupFromTaskDetail({ task_id: 'root', status: 'running', plan_record: RECORD, plan_objections: RECORD_FOLD }, 'root');
    assert.equal(group.id, 'plan:root');
    assert.equal(group.label, 'Plan');
    assert.equal(group.subject, 'revision 2');
    assert.equal(group.state, 'terminal');  // nobody is awaited
    // A1's o1 was withdrawn by its raiser in a2; the added critic's k1 stands but never holds.
    assert.equal(group.verdict, '1 open objection');
    assert.equal(group.tone, 'warn');
    assert.deepEqual(group.attempts.map((a) => a.id), ['revision:2', 'ask:a1', 'ask:a2']);
    const revision = group.attempts[0];
    assert.match(revision.detailText, /done_when:\n  c1: a five-slide story — shown by: read it/);
    assert.match(revision.detailText, /earlier revision 1 .*: c1: five slides/);
    const first = group.attempts[1];
    assert.match(first.detailText, /triad_a \(m\/a\): answered — 1 objection: The bar is low\./);
    assert.match(first.detailText, /triad_c \(m\/c\): not sent — health_skip\[window_exhausted\]/);
    assert.equal(first.verdict, 'no objection');  // o1 no longer stands
    const second = group.attempts[2];
    assert.match(second.detailText, /critic_x \(m\/x, added — never holds\): answered — 1 objection: No\./);
    assert.match(second.detailText, /objection critic_x\/k1 \(never holds\): wrong shape/);
    assert.match(second.detailText, /triad_a \(m\/a\): answered — 0 objections, 1 withdrawn: Better\./);
    const html = renderReviewsSection([group], { sectionExpanded: true, expandedGroups: new Set([group.id]),
        expandedAttempts: new Set(group.attempts.map((a) => `${group.id}:${a.id}`)) });
    assert.match(html, /a five-slide story/);
    assert.match(html, /wrong shape/);
});

test('an ask before any plan is shown as the reading of the request, never as "revision null"', () => {
    const reading = { schema_version: 3, revisions: [], asks: [{ ask_id: 'a1', ts: 't', revision: null, message: 'How do you read it?',
        host_added: [], participants: [answered('triad_a', 'm/a', { prose: 'A story, not slides.' })] }] };
    const group = planReviewGroupFromTaskDetail({ task_id: 'root', status: 'running', plan_record: reading, plan_objections: fold() }, 'root');
    assert.equal(group.subject, '');
    assert.deepEqual(group.attempts.map((a) => a.id), ['ask:a1']);
    const [ask] = group.attempts;
    assert.equal(ask.label, 'ask a1 · the reading of the request');
    assert.match(ask.detailText, /^ask a1 on the reading of the request — «How do you read it\?»/);
    assert.match(ask.detailText, /triad_a \(m\/a\): answered: A story, not slides\./);
    assert.doesNotMatch(ask.detailText, /objections?:|advice/);
    // A reading's points stay out of the fold, so the card shows them as advice with their text, never a bare count.
    const pointed = { ...reading, asks: [{ ...reading.asks[0], participants: [answered('triad_a', 'm/a', {
        prose: 'A story.', objections: [{ id: 'o1', text: 'The brief narrows the ask' }] })] }] };
    const pointedAsk = planReviewGroupFromTaskDetail({ task_id: 'root', status: 'running', plan_record: pointed, plan_objections: fold() }, 'root').attempts[0];
    assert.match(pointedAsk.detailText, /answered: A story\. — advice, never holds: The brief narrows the ask/);
    assert.doesNotMatch(pointedAsk.detailText, /1 objection/);
    assert.doesNotMatch(ask.detailText + ask.label, /revision (null|undefined|0)/);
    assert.equal(group.verdict, 'no open objection');
    const body = { ...reading, asks: [{ ...reading.asks[0], host_added: ['triad_b'] }] };
    const bodyAsk = planReviewGroupFromTaskDetail({ task_id: 'root', status: 'running', plan_record: body, plan_objections: fold() }, 'root').attempts[0];
    assert.match(bodyAsk.detailText, /a reading about Ouroboros's own body: the host added triad_b/);
    assert.doesNotMatch(bodyAsk.detailText, /first ask on a body plan/);
    // The other direction: an ask about a revision keeps its revision label.
    const planned = planReviewGroupFromTaskDetail({ task_id: 'root', status: 'running', plan_record: RECORD, plan_objections: RECORD_FOLD }, 'root');
    assert.equal(planned.attempts[1].label, 'ask a1 · revision 1');
    assert.match(planned.attempts[1].detailText, /^ask a1 on revision 1 — «First look\.»/);
});

test('an owner-marked objection that stands may hold, and awaited answers keep the group live only while the task runs', () => {
    const record = { ...RECORD, asks: [RECORD.asks[0], {
        ask_id: 'a2', ts: 't', revision: 2, message: '', host_added: [], participants: [
            { seat_id: 'triad_a', model: 'm/a', route: 'api_chat', owner_marked: true, dispatched: true, status: 'pending' },
        ] }] };
    const open = fold({ triad_a: [standing('o1', 'five slides is not the bar', 'a1', 1)] });
    const live = planReviewGroupFromTaskDetail({ task_id: 'root', status: 'running', plan_record: record, plan_objections: open }, 'root');
    assert.equal(live.verdict, '1 objection may hold');  // o1 stands: a1 raised it and nobody withdrew it
    assert.equal(live.state, 'running');
    assert.equal(live.activeCount, 1);
    assert.equal(live.attempts.at(-1).state, 'running');
    assert.equal(live.attempts.at(-1).progress, '0 of 1 answered');
    const done = planReviewGroupFromTaskDetail({ task_id: 'root', status: 'completed', plan_record: record, plan_objections: open }, 'root');
    assert.equal(done.state, 'terminal');
    assert.equal(done.attempts.at(-1).state, 'terminal');
});

test('a withdrawal by another seat and an omission are rendered as the host records them', () => {
    const record = { ...RECORD, asks: [RECORD.asks[0], {
        ask_id: 'a2', ts: 't', revision: 2, message: '', host_added: [], participants: [
            answered('triad_b', 'm/b', { withdrawn: [{ id: 'o1', why: 'I disagree with A' }] }),
            answered('triad_a', 'm/a', { prose: 'nothing about o1' }),
        ] }] };
    const flagged = fold({ triad_a: [standing('o1', 'five slides is not the bar', 'a1', 1, { not_adjudicated: true })] });
    const group = planReviewGroupFromTaskDetail({ task_id: 'root', status: 'running', plan_record: record, plan_objections: flagged }, 'root');
    assert.equal(group.verdict, '1 objection may hold');
    assert.match(group.attempts[1].detailText, /objection triad_a\/o1 \(not adjudicated\): five slides is not the bar/);
});

test('a body plan names the host-added reviewers and a record without asks says so', () => {
    const body = { schema_version: 3, revisions: [{ n: 1, ts: 't', goal: 'g', plan: 'p', body: true, body_note: 'constitutional: ouroboros/loop.py',
        affected_paths: ['ouroboros/loop.py'], done_when: [] }], asks: [] };
    const group = planReviewGroupFromTaskDetail({ task_id: 'root', status: 'running', plan_record: body, plan_objections: fold() }, 'root');
    assert.equal(group.verdict, 'not asked yet');
    assert.match(group.attempts[0].detailText, /body plan: constitutional: ouroboros\/loop\.py/);
    assert.match(group.attempts[0].detailText, /done_when: \(none stated yet\)/);
    const asked = { ...body, asks: [{ ask_id: 'a1', ts: 't', revision: 1, message: '', host_added: ['triad_b', 'triad_c'], participants: [
        answered('triad_a', 'm/a', {}), answered('triad_b', 'm/b', {}, { host_added: true }), answered('triad_c', 'm/c', {}, { host_added: true }),
    ] }] };
    const withAsk = planReviewGroupFromTaskDetail({ task_id: 'root', status: 'running', plan_record: asked, plan_objections: fold() }, 'root');
    assert.match(withAsk.attempts[1].detailText, /first ask on a body plan: the host added triad_b, triad_c/);
    assert.equal(withAsk.verdict, 'no open objection');
});

test('an empty plan record falls back to the retired organ\'s record and a detail with neither renders nothing', () => {
    const fingerprint = 'a'.repeat(64);
    const legacy = planReviewGroupFromTaskDetail({
        task_id: 'root', status: 'completed', plan_record: { schema_version: 3, revisions: [], asks: [] },
        plan_review_state: { schema_version: 2, current_attempt: { fingerprint, status: 'open', reason: '' },
            waves: [{ request_fingerprint: fingerprint, cycle_index: 1, aggregate: 'GREEN', closed: true }], waves_omitted: 0 },
    }, 'root');
    assert.equal(legacy.label, 'Plan review');
    assert.equal(legacy.attempts[0].label, 'wave 1');
    assert.equal(planReviewGroupFromTaskDetail({ task_id: 'root', status: 'completed' }, 'root'), null);
});

test('a refreshed record replaces the live ask in the store without inventing a verdict', () => {
    const store = new Map();
    const pending = { ...RECORD, asks: [{ ask_id: 'a1', ts: 't', revision: 2, message: '', host_added: [], participants: [
        { seat_id: 'triad_a', model: 'm/a', route: 'api_chat', owner_marked: true, dispatched: true, status: 'pending' },
    ] }] };
    mergeReviewGroup(store, planReviewGroupFromTaskDetail({ task_id: 'root', status: 'running', plan_record: pending, plan_objections: fold() }, 'root'));
    assert.equal(store.get('plan:root').attempts.at(-1).state, 'running');
    const settled = { ...pending, asks: [{ ...pending.asks[0], participants: [answered('triad_a', 'm/a', { prose: 'ok' })] }] };
    mergeReviewGroup(store, planReviewGroupFromTaskDetail({ task_id: 'root', status: 'running', plan_record: settled, plan_objections: fold() }, 'root'));
    const merged = store.get('plan:root');
    assert.equal(merged.attempts.at(-1).state, 'terminal');
    assert.equal(merged.attempts.at(-1).verdict, 'no objection');
    assert.equal(merged.verdict, 'no open objection');
});

test('an upgraded task shows the imported finding the host says holds, and its raiser\'s withdrawal clears it', () => {
    // Scope R1, C2/F3: the gate held on s1:f1 while the panel's own fold said "not asked yet".
    const upgraded = { schema_version: 3, revisions: [{ n: 1, ts: 't', goal: 'g', plan: 'p', body: false, affected_paths: [], done_when: [] }], asks: [] };
    const imported = fold({ s1: [standing('s1:f1', 'wrong shape', 'legacy', 0, { legacy: true, not_adjudicated: true })] }, [], { legacy_engaged: true });
    const held = planReviewGroupFromTaskDetail({ task_id: 'root', status: 'running', plan_record: upgraded, plan_objections: imported }, 'root');
    assert.equal(held.verdict, '1 objection may hold');
    assert.equal(held.tone, 'warn');
    assert.match(held.attempts[0].detailText, /imported objection s1\/s1:f1 from the previous plan review \(may hold\) \(not adjudicated\): wrong shape/);
    const asked = { ...upgraded, asks: [{ ask_id: 'a1', ts: 't', revision: 1, message: '', host_added: [], participants: [
        answered('s1', 'm/a', { withdrawn: [{ id: 's1:f1', why: 'fixed' }] }),
    ] }] };
    const released = fold({}, [{ seat_id: 's1', id: 's1:f1', withdrawn_in: 'a1', why: 'fixed' }], { legacy_engaged: true });
    const clear = planReviewGroupFromTaskDetail({ task_id: 'root', status: 'running', plan_record: asked, plan_objections: released }, 'root');
    assert.equal(clear.verdict, 'no open objection');
    assert.doesNotMatch(clear.attempts[0].detailText, /imported objection/);
    assert.match(clear.attempts[1].detailText, /answered — 0 objections, 1 withdrawn/);
});

test('a detail without the host fold never claims "no open objection", and an unreadable earlier state is named', () => {
    const record = { ...RECORD, asks: [RECORD.asks[0]] };
    const blind = planReviewGroupFromTaskDetail({ task_id: 'root', status: 'running', plan_record: record }, 'root');
    assert.equal(blind.verdict, 'objections not projected');
    assert.equal(blind.tone, 'neutral');
    assert.match(blind.attempts[0].detailText, /open objections: not projected by the host in this detail/);
    assert.doesNotMatch(blind.attempts[1].detailText, /objection triad_a\/o1/);  // nothing is re-derived from the raw record
    const named = planReviewGroupFromTaskDetail({ task_id: 'root', status: 'running', plan_record: record,
        plan_objections: fold({}, [], { legacy_unreadable: { source: 'plan_review_state', error: 'ValueError: PLAN_REVIEW_STATE_INVALID' } }) }, 'root');
    assert.match(named.attempts[0].detailText, /earlier plan-review state unreadable \(plan_review_state\): ValueError: PLAN_REVIEW_STATE_INVALID/);
});

test('pending room preparation stays qualified before a dispatch outcome, including a known pending seat', () => {
    const inline = { conversation_rows: 2, conversation_inline_rows: 2, other_rows: 1 };
    for (const unresolved of [false, true]) {
        const participant = { ...answered('pending', 'm/a', {}), status: 'pending', unresolved, room_delivery: inline };
        const record = { schema_version: 3, revisions: RECORD.revisions, asks: [
            { ask_id: 'a1', revision: 2, host_added: [], participants: [participant] },
        ] };
        const group = planReviewGroupFromTaskDetail({ task_id: 'root', status: 'running', plan_record: record, plan_objections: fold() }, 'root');
        assert.match(group.attempts.at(-1).detailText, /counted for a delivery that may not have been sent/);
    }
});

test('Record facts say what the room gave each seat and what its file reads covered, apart and without a judgement', () => {
    const inline = { conversation_rows: 19, conversation_inline_rows: 19, other_rows: 481 };
    const record = { schema_version: 3, revisions: RECORD.revisions, asks: [
        { ask_id: 'a1', ts: '2026-10-15T02:05:00+00:00', revision: 2, message: '', host_added: [], participants: [
            answered('unread', 'm/a', {}, { retrieves: true, room_delivery: inline, host_file_read_attestation: 'host_observed',
                room_read_coverage: { status: 'incomplete', covered_chars: 0, complete_chars: 52340, provenance: 'host_observed' } }),
            answered('part', 'm/b', {}, { retrieves: true, room_delivery: { ...inline, conversation_inline_rows: 7 },
                room_read_coverage: { status: 'incomplete', covered_chars: 12000, complete_chars: 52340, provenance: 'host_observed' } }),
            answered('session', 'm/c', {}, { retrieves: true, room_delivery: inline,
                room_read_coverage: { status: 'complete', covered_chars: 52340, complete_chars: 52340, provenance: 'harness_observed' } }),
            answered('unobserved', 'm/d', {}, { retrieves: true, room_delivery: inline, host_file_read_attestation: 'unobserved' }),
            answered('packet', 'm/e', {}, { retrieves: false, room_delivery: { conversation_rows: 2, conversation_inline_rows: 2, other_rows: 0 } }),
            answered('older', 'm/f', {}),
            // What an older projector could store: a blank 0 of N, and a paged brief's numbers with no provenance.
            answered('blank', 'm/g', {}, { room_read_coverage: { status: '', covered_chars: 0, complete_chars: 1234, provenance: '' } }),
            answered('brief', 'm/h', {}, { room_read_coverage: { status: 'incomplete', covered_chars: 10, complete_chars: 900000, provenance: '' } }),
        ] },
    ] };
    const group = planReviewGroupFromTaskDetail({ task_id: 'root', status: 'done', plan_record: record, plan_objections: fold() }, 'root');
    const facts = group.attempts.at(-1).detailText.split('\n');
    assert.ok(facts.includes('Participant unread room: 19 of 19 conversation rows inline, 481 other rows only in the snapshot file · file reads covered 0 of 52,340 characters · observed by the host'));
    assert.ok(facts.includes('Participant part room: 7 of 19 conversation rows inline, 481 other rows only in the snapshot file · file reads covered 12,000 of 52,340 characters · observed by the host'));
    assert.ok(facts.includes('Participant session room: 19 of 19 conversation rows inline, 481 other rows only in the snapshot file · file reads covered 52,340 of 52,340 characters · inferred from the harness journal'));
    // What nothing measured says nothing: never 0 of N, and an older record gains no line.
    assert.ok(facts.includes('Participant unobserved room: 19 of 19 conversation rows inline, 481 other rows only in the snapshot file'));
    assert.ok(facts.includes('Participant packet room: 2 of 2 conversation rows inline'));
    for (const seat of ['older', 'blank', 'brief']) assert.ok(!facts.some((line) => line.startsWith(`Participant ${seat} room`)), seat);
    assert.ok(!facts.join('\n').includes('%'));
    // These facts move no state, tone or verdict, and a read of nothing is no objection.
    assert.equal(group.verdict, 'no open objection');
    assert.equal(group.tone, 'done');
    assert.equal(group.attempts.at(-1).tone, 'done');
    assert.equal(group.attempts.at(-1).verdict, 'no objection');
    const html = renderReviewsSection([group], { sectionExpanded: true, expandedGroups: new Set([group.id]),
        expandedAttempts: new Set(group.attempts.map((a) => `${group.id}:${a.id}`)) });
    assert.match(html, /<dt>Participant unread room<\/dt><dd>19 of 19 conversation rows inline, 481 other rows only in the snapshot file · file reads covered 0 of 52,340 characters · observed by the host<\/dd>/);
});

// An ask whose evidence was mixed, as plan_task records it: an own artifact at its real absolute address (attached
// whole, and listed twice), a helper's saved result, the settings file (refused), a link (never fetched), a large file
// (its head) and a chat range whose history has a gap (attached, in part).
const LONG = '/Users/owner/Ouroboros/data/task_results/artifacts/example-task/source_handles/context_checkpoints/'
    + `plan-revision-0003-${'a'.repeat(64)}.json`;
const MIXED_ASK = {
    ask_id: 'a3', ts: '2026-10-15T03:05:00+00:00', revision: 2, message: 'Mixed evidence.', host_added: [],
    attach: [LONG, 'task:helper-1', LONG, '/Users/owner/Ouroboros/data/settings.json', 'https://example.com/spec',
        'docs/CHECKLISTS.md', 'chat:7::lines=1-40'],
    attached: [{ locator: LONG, sha256: 'a'.repeat(64), bytes: 11352 }, { locator: 'task:helper-1', sha256: 'b'.repeat(64), bytes: 812 },
        { locator: 'docs/CHECKLISTS.md', sha256: 'c'.repeat(64), bytes: 170512, ref: { root: 'artifact_store', path: 'x.txt' } },
        { locator: 'chat:7::lines=1-40', sha256: 'd'.repeat(64), bytes: 52340 }],
    omissions: [{ locator: LONG, reason: 'duplicate_locator' }, { locator: '/Users/owner/Ouroboros/data/settings.json', reason: 'denied_path' },
        { locator: 'https://example.com/spec', reason: 'url_not_fetched' }, { locator: 'docs/CHECKLISTS.md', reason: 'truncated_to_40000' },
        { locator: 'chat:7::lines=1-40', reason: 'chat_history_gap:mailbox_incomplete' }],
    participants: [answered('triad_a', 'm/a', { prose: 'Read what came.' })],
};

test('an ask names what it attached, whole or in part, apart from what it did not, and a cut never counts as a failure', () => {
    const record = { schema_version: 3, revisions: RECORD.revisions, asks: [MIXED_ASK] };
    const group = planReviewGroupFromTaskDetail({ task_id: 'root', status: 'done', plan_record: record, plan_objections: fold() }, 'root');
    const ask = group.attempts.at(-1);
    // The opened row's one sentence counts locators: the repeat is one item attached whole, the head and the gapped
    // range are attached in part, the refused file and the link are not attached.
    assert.equal(ask.evidenceNotice, 'Evidence for this ask: 2 of 6 attached whole, 2 in part, 2 not attached.');
    const facts = ask.detailText.split('\n');
    const at = facts.indexOf('Ask a3 evidence: prepared for its participants: 2 of 6 attached whole, 2 in part, 2 not attached');
    assert.ok(at > 0, ask.detailText);
    assert.deepEqual(facts.slice(at + 1, at + 7), [
        `Ask a3 attached: ${LONG} · source 11,352 bytes · listed more than once (duplicate_locator) · sha256 ${'a'.repeat(64)}`,
        `  task:helper-1 · source 812 bytes · sha256 ${'b'.repeat(64)}`,
        `  docs/CHECKLISTS.md · in part: the first 40,000 of 170,512 bytes (truncated_to_40000) · sha256 ${'c'.repeat(64)}`,
        `  chat:7::lines=1-40 · source 52,340 bytes · in part: the conversation history has a gap (chat_history_gap:mailbox_incomplete) · sha256 ${'d'.repeat(64)}`,
        'Ask a3 not attached: /Users/owner/Ouroboros/data/settings.json · settings or runtime data outside this task\'s own stores (denied_path)',
        '  https://example.com/spec · a link: links are listed, never fetched (url_not_fetched)',
    ]);
    // Evidence moves no state, tone or verdict.
    assert.equal(ask.verdict, 'no objection');
    assert.equal(ask.tone, 'done');
    const html = renderReviewsSection([group], { sectionExpanded: true, expandedGroups: new Set([group.id]) });
    const row = html.match(/<div class="review-attempt" data-review-attempt="plan:root:ask:a3">[\s\S]*?<div class="review-seat"/)[0];
    assert.match(row, /<p class="review-ask-line">[^<]*<\/p><p class="review-line review-line-whole"><span class="review-line-text">Evidence for this ask: 2 of 6 attached whole, 2 in part, 2 not attached\.<\/span><\/p>/);
    assert.doesNotMatch(row, /settings\.json|example\.com|truncated_to|denied_path/);  // which and why stay in Record facts
    const dl = html.match(/<details class="review-facts"[\s\S]*?<\/details>/)[0];
    assert.ok(dl.includes(`<dt>Ask a3 attached</dt><dd>${LONG} · source 11,352 bytes`), dl);
    assert.match(dl, /<dt>Ask a3 not attached<\/dt><dd>\/Users\/owner\/Ouroboros\/data\/settings\.json · [^<]*\nhttps:\/\/example\.com\/spec · /);
});

test('an older ask says what it did not record, never that nothing attached, and a complete ask adds no sentence', () => {
    const asks = [
        { ask_id: 'old', ts: '2026-10-15T01:05:00+00:00', revision: 1, message: '', host_added: [], participants: [answered('triad_a', 'm/a', {})] },
        { ask_id: 'cut', ts: '2026-10-15T01:06:00+00:00', revision: 1, message: '', host_added: [], attach: ['a.md', 'b.md'],
            attached: [{ locator: 'a.md', sha256: 'e'.repeat(64), bytes: 9 }], participants: [answered('triad_a', 'm/a', {})] },
        { ask_id: 'named', ts: '2026-10-15T01:07:00+00:00', revision: 1, message: '', host_added: [], attach: ['a.md'],
            participants: [answered('triad_a', 'm/a', {})] },
        { ask_id: 'whole', ts: '2026-10-15T01:08:00+00:00', revision: 1, message: '', host_added: [], attach: ['a.md'],
            attached: [{ locator: 'a.md', sha256: 'e'.repeat(64), bytes: 9 }], omissions: [], participants: [answered('triad_a', 'm/a', {})] },
        { ask_id: 'none', ts: '2026-10-15T01:09:00+00:00', revision: 1, message: '', host_added: [], attach: [], attached: [], omissions: [],
            participants: [answered('triad_a', 'm/a', {})] },
    ];
    const group = planReviewGroupFromTaskDetail({ task_id: 'root', status: 'done', plan_record: { schema_version: 3, revisions: RECORD.revisions, asks }, plan_objections: fold() }, 'root');
    const [old, cut, named, whole, none] = group.attempts.slice(1);
    assert.match(old.detailText, /^Ask old evidence: not recorded$/m);
    assert.doesNotMatch(old.detailText, /0 of|attached whole|nothing/);
    // What attached is known, why the other is not: it reads as not attached with its reasons not recorded.
    assert.equal(cut.evidenceNotice, 'Evidence for this ask: 0 of 2 attached whole, 1 attached, completeness not recorded, 1 not attached.');
    assert.match(cut.detailText, /Ask cut attached: a\.md · completeness not recorded/);
    assert.match(cut.detailText, /^Ask cut not attached: b\.md$/m);
    assert.match(cut.detailText, /^Ask cut evidence cuts and absences: not recorded$/m);
    // Only the declaration is known: the row discloses missing preparation facts.
    assert.equal(named.evidenceNotice, 'Evidence for this ask: 0 of 1 attached whole, 1 not recorded.');
    assert.match(named.detailText, /^Ask named declared, attachment not recorded: a\.md$/m);
    assert.match(named.detailText, /^Ask named evidence: prepared for its participants: 0 of 1 attached whole, 1 not recorded$/m);
    assert.equal(whole.evidenceNotice, '');
    assert.match(whole.detailText, /^Ask whole attached: a\.md · source 9 bytes · sha256 e{64}$/m);
    assert.equal(none.evidenceNotice, '');
    assert.doesNotMatch(none.detailText, /evidence|attached/);
    const html = renderReviewsSection([group], { sectionExpanded: true, expandedGroups: new Set([group.id]) });
    assert.equal((html.match(/Evidence for this ask:/g) || []).length, 2);
});


test('a declared locator missing both preparation rows stays unknown and visible', () => {
    const asks = [{ ask_id: 'gap', revision: 1, attach: ['unresolved.md'], attached: [], omissions: [], participants: [] }];
    const group = planReviewGroupFromTaskDetail({ task_id: 'root', status: 'done', plan_record: { schema_version: 3, revisions: RECORD.revisions, asks }, plan_objections: fold() }, 'root');
    const ask = group.attempts.at(-1);
    assert.equal(ask.evidenceNotice, 'Evidence for this ask: 0 of 1 attached whole, 1 not recorded.');
    assert.match(ask.detailText, /declared, attachment not recorded: unresolved\.md/);
});

test('the browser classifies every shared evidence case as the host does', () => {
    // One table with tests/test_plan_record.py (plan_record.ask_evidence): the two classifiers cannot drift apart.
    const { cases } = JSON.parse(readFileSync(new URL('./fixtures/plan_ask_evidence_parity.json', import.meta.url), 'utf8'));
    assert.ok(cases.length >= 9);
    for (const row of cases) {
        const { recorded, counts, items } = askEvidence(row.ask);
        assert.deepEqual({ recorded, counts, items: items.map((item) => [item.locator, item.state, item.reasons]) }, row.expected, row.name);
    }
});
