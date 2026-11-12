import assert from 'node:assert/strict';
import test from 'node:test';

import {
    isGroupedTaskEvent, lateOutcomeIsCurrent, summarizeChatLiveEvent, taskOutcomeSeverity, taskPresentation,
    taskReasonDetail, taskTerminalPhase, taskTerminalSummary,
} from '../modules/log_events.js';
import { TASK_CAUSE_PHRASES } from '../modules/task_reason_words.js';

// A review that was merely awaited when its task ended is a gap, never a warning:
// the chip stays Done and the card states the fact. Every real degradation keeps
// the chip, the reason and the sentence it had.

const AWAITED_SENTENCE = 'Not every plan reviewer had answered when the task ended.';
const ADVISORY_SENTENCE = 'The plan review was never closed; the work went on with what the reviewers said.';

const done = (axes, extra = {}) => ({
    status: 'completed', task_id: 't1', reason_code: 'final_message', outcome_axes: axes, ...extra,
});

test('a finish over an awaited plan review is Done and states the fact', () => {
    const record = done({
        execution: { status: 'ok', reason_code: 'final_message', plan_review: 'awaiting' },
        objective: { status: 'not_evaluated', source: 'none' },
        review: { status: 'skipped' },
    });
    assert.equal(taskOutcomeSeverity(record), 'done');
    assert.deepEqual(taskPresentation(taskTerminalPhase(record)), { phase: 'done', headline: 'Done' });
    assert.equal(taskReasonDetail(record), AWAITED_SENTENCE);
    const summary = taskTerminalSummary(record);
    assert.equal(summary.headline, 'Done');
    assert.equal(summary.body, AWAITED_SENTENCE);
    // A live frame may carry no reason code at all; the typed fact still speaks.
    assert.equal(taskReasonDetail({ ...record, reason_code: undefined }), AWAITED_SENTENCE);
});

test('the awaited fact is the cause sentence of a clean card only', () => {
    const execution = { status: 'ok', reason_code: 'final_message', plan_review: 'awaiting' };
    const warned = done({ execution, objective: { status: 'not_evaluated', warning: 'residual_tool_errors_without_review' } });
    assert.equal(taskOutcomeSeverity(warned), 'warn');
    assert.equal(taskReasonDetail(warned), 'Tool errors remain in the record.');
    const failed = done({ execution, artifacts: { status: 'missing' } });
    assert.equal(taskOutcomeSeverity(failed), 'error');
    assert.equal(taskReasonDetail(failed), 'Some result files are missing or could not be preserved.');
});

test('a plan review that really stayed open keeps its warning, code and sentence', () => {
    const record = {
        status: 'completed', reason_code: 'plan_review_advisory',
        outcome_axes: { execution: { status: 'degraded', reason_code: 'plan_review_advisory' } },
    };
    assert.equal(taskOutcomeSeverity(record), 'warn');
    assert.equal(taskPresentation(taskTerminalPhase(record)).headline, 'Done with warnings');
    assert.equal(taskReasonDetail(record), ADVISORY_SENTENCE);
});

test('an independent awaited fact coexists with review provenance but not a rail or owner stop', () => {
    const awaited = { status: 'ok', plan_review: 'awaiting' };
    const decided = done({
        execution: awaited,
        review: { status: 'pass', acceptance_decision: { status: 'finalized_unaccepted', reason: 'capsule_spent' } },
    });
    assert.equal(taskReasonDetail(decided), 'Not every plan reviewer had answered when the task ended · The one allowed improvement pass was already used.');
    const railed = { ...done({ execution: { ...awaited, status: 'best_effort' } }), reason_code: 'round_limit' };
    assert.equal(taskReasonDetail(railed), 'The task hit its round limit before it could finish cleanly');
    assert.equal(taskOutcomeSeverity(railed), 'warn');
    const stopped = { ...done({ execution: awaited }), reason_code: 'owner_requested_finalization' };
    assert.equal(taskReasonDetail(stopped), '');
    // Without the typed fact a clean finish still states nothing.
    assert.equal(taskReasonDetail(done({ execution: { status: 'ok' } })), '');
    assert.equal(taskReasonDetail(done({ execution: { status: 'ok', plan_review: 'open' } })), '');
});

test('an acceptance panel that was only awaited is not a warning; the author accepting beside it is', () => {
    assert.equal(taskOutcomeSeverity(done({ execution: { status: 'ok' }, review: { status: 'awaiting' } })), 'done');
    // The author's acceptance of a result whose review has no conclusion yet is always flagged (owner, 8 Nov 2026).
    const authored = done({
        execution: { status: 'ok' },
        objective: { status: 'pass', source: 'author_acceptance', review_status: 'awaiting' },
        review: {
            status: 'awaiting', eligibility: 'review_in_flight',
            acceptance_decision: { status: 'finalized_unaccepted', reason: 'author_finish' },
        },
    });
    assert.equal(taskOutcomeSeverity(authored), 'warn');
    assert.equal(taskReasonDetail(authored), TASK_CAUSE_PHRASES.author_finish);
    // The legacy top-level mirror of the review axis reads the same way.
    assert.equal(taskOutcomeSeverity({ status: 'completed', review_status: { status: 'awaiting' } }), 'done');
});

test('every real review outcome keeps the chip it had', () => {
    const chip = (axes, extra) => taskOutcomeSeverity(done(axes, extra));
    assert.equal(chip({ execution: { status: 'ok' }, review: { status: 'degraded' } }), 'warn');
    assert.equal(chip({ execution: { status: 'ok' }, review: { status: 'fail' } }), 'error');
    assert.equal(chip({ execution: { status: 'ok' }, review: { status: 'pass' } }), 'done');
    assert.equal(chip({ execution: { status: 'ok' }, review: { status: 'skipped' } }), 'done');
    // A rail, a degraded objective or an objective warning beside a wait still warns.
    assert.equal(chip({ execution: { status: 'best_effort' }, review: { status: 'awaiting' } }), 'warn');
    assert.equal(chip({ execution: { status: 'ok' }, objective: { status: 'degraded' }, review: { status: 'awaiting' } }), 'warn');
    assert.equal(chip({
        execution: { status: 'ok' }, review: { status: 'awaiting' },
        objective: { status: 'not_evaluated', warning: 'residual_tool_errors_without_review' },
    }), 'warn');
    assert.equal(chip({ execution: { status: 'ok' }, objective: { status: 'fail' }, review: { status: 'awaiting' } }), 'error');
});

// A late review that resolved its ended task (late_review_marks.announce_current_outcome) carries the task's CURRENT
// outcome on the existing review_late_result event: the open card repaints exactly what its reload reads.
// The warnings passed here are detail-only (receipt_absent): the host's phase stays Done (9 Nov 2026).
const lateOutcome = (warnings = []) => ({
    status: 'completed', reason_code: 'final_message', outcome_phase: 'done',
    outcome_axes: {
        lifecycle: { status: 'completed' }, execution: { status: 'ok', reason_code: 'final_message' },
        objective: { status: 'pass', source: 'result_review', review_status: 'pass',
            ...(warnings.length ? { warning: warnings[0], warnings } : {}) },
        review: { status: 'pass', source: 'result_review' }, artifacts: { status: 'not_applicable' },
    },
});
const lateEvent = (current) => ({ type: 'review_late_result', task_id: 't1', surface: 'result', record_id: 'rec-1',
    subject_sha: 's1', status: 'settled', aggregate: 'PASS', current_outcome: current });

test('a late PASS repaints the finished card with the current outcome its reload reads', () => {
    const event = lateEvent(lateOutcome());
    assert.equal(isGroupedTaskEvent(event), true);
    const live = summarizeChatLiveEvent(event);
    assert.deepEqual([live.terminal, live.phase, live.headline, live.body, live.dedupeKey, live.currentOutcome],
        [true, 'done', 'Done', '', 'task_done|t1', true]);
    const reloaded = taskTerminalSummary({ ...lateOutcome(), task_id: 't1', system_type: 'task_summary' });
    assert.deepEqual([live.phase, live.headline, live.body], [reloaded.phase, reloaded.headline, reloaded.body]);
    assert.equal(taskPresentation(taskTerminalPhase(lateOutcome())).phase, lateOutcome().outcome_phase, 'host phase twin');
    assert.deepEqual(summarizeChatLiveEvent(event), live, 'a replayed event paints the same outcome again');
});

test('an independent receipt_absent stays through a late PASS as a detail: Done and its own cause', () => {
    const outcome = lateOutcome(['receipt_absent']);
    const live = summarizeChatLiveEvent(lateEvent(outcome));
    assert.deepEqual([live.phase, live.headline, live.body, live.activityPreview],
        ['done', 'Done', TASK_CAUSE_PHRASES.receipt_absent, TASK_CAUSE_PHRASES.receipt_absent]);
    assert.equal(taskPresentation(taskTerminalPhase(outcome)).phase, outcome.outcome_phase, 'host phase twin');
});

test('a late seat result without a current outcome never reaches the card, so a late FAIL clears nothing', () => {
    const seat = { type: 'review_late_result', task_id: 't1', surface: 'result', slot_id: 's1', status: 'complete' };
    assert.equal(isGroupedTaskEvent(seat), false);
    const view = summarizeChatLiveEvent(seat);
    assert.deepEqual([view.terminal, view.visible, Boolean(view.currentOutcome)], [false, false, false]);
});

test('a late outcome repaints only a shown finished card, never over a newer frame of it (the card clock)', () => {
    const evt = { ts: '2026-11-09T00:10:00+00:00' };
    const card = (finished, latestSourceTs) => ({ finished, latestSourceTs });
    assert.equal(lateOutcomeIsCurrent(undefined, evt), false, 'no card in this page: none is created');
    assert.equal(lateOutcomeIsCurrent(card(false, ''), evt), false, 'a card still running waits for its own end');
    assert.equal(lateOutcomeIsCurrent(card(true, '2026-11-09T00:00:00+00:00'), evt), true);
    assert.equal(lateOutcomeIsCurrent(card(true, evt.ts), evt), true, 'the same frame again paints the same outcome');
    assert.equal(lateOutcomeIsCurrent(card(true, '2026-11-09T00:20:00+00:00'), evt), false,
        'a delayed late frame never paints over a newer state, its warnings included');
});

test('a late outcome is as old as the stored row it read, not the moment it was published (Astra, 10 Nov 2026)', () => {
    // The row was read at 00:10 and the snapshot published at 00:15; the card already shows a state of 00:12.
    const stale = { ts: '2026-11-09T00:15:00+00:00', current_outcome: { revision: '2026-11-09T00:10:00+00:00' } };
    const card = (latestSourceTs) => ({ finished: true, latestSourceTs });
    assert.equal(lateOutcomeIsCurrent(card('2026-11-09T00:12:00+00:00'), stale), false,
        'an older snapshot published later never repaints the newer state, its warning included');
    assert.equal(lateOutcomeIsCurrent(card('2026-11-09T00:05:00+00:00'), stale), true, 'over an older state it is current');
    assert.equal(lateOutcomeIsCurrent(card(stale.current_outcome.revision), stale), true,
        'chat.js sets the card clock to the revision it took, so the same frame again still paints');
    assert.equal(lateOutcomeIsCurrent(card('2026-11-09T00:12:00+00:00'), { ts: stale.ts }), true,
        'a frame without a revision keeps its own time');
});

test('a late outcome keeps its microseconds: two writes inside one millisecond stay ordered (Astra, 10 Nov 2026)', () => {
    // The writer's real isoformat() values from the reproduction: 688µs apart, one millisecond to Date.parse.
    const older = '2026-11-10T05:06:26.147169+00:00', newer = '2026-11-10T05:06:26.147857+00:00';
    const late = (revision) => ({ ts: '2026-11-10T05:06:40.000000+00:00', current_outcome: { revision } });
    const card = (latestSourceTs) => ({ finished: true, latestSourceTs });
    assert.equal(lateOutcomeIsCurrent(card(newer), late(older)), false, 'the older PASS never clears the newer warning');
    assert.equal(lateOutcomeIsCurrent(card(older), late(newer)), true, 'a truly later outcome inside the millisecond paints');
    assert.equal(lateOutcomeIsCurrent(card(newer), late(newer)), true, 'the same revision again is a replay');
    assert.equal(lateOutcomeIsCurrent(card(newer), { ts: older }), false, 'a frame without a revision orders by its own time');
    assert.equal(lateOutcomeIsCurrent(card('2026-11-10T05:06:26.146000+00:00'), late(older)), true, 'a distinct older millisecond still loses to it');
});

test('every instant spelling the host and the page write orders by the same microseconds', () => {
    const verdict = (clock, revision) => lateOutcomeIsCurrent({ finished: true, latestSourceTs: clock }, { ts: '', current_outcome: { revision } });
    for (const [clock, revision, current, why] of [
        ['2026-11-10T05:06:26.000001+00:00', '2026-11-10T05:06:26+00:00', false, 'isoformat() drops a zero fraction: the whole second is older than .000001'],
        ['2026-11-10T05:06:26+00:00', '2026-11-10T05:06:26.000001+00:00', true, 'and .000001 is later than the whole second'],
        ['2026-11-10T05:06:26+00:00', '2026-11-10T05:06:26.000000+00:00', true, 'a zero fraction is the same instant as none'],
        ['2026-11-10T05:06:26.147857Z', '2026-11-10T05:06:26.147169+00:00', false, 'Z and +00:00 name one zone'],
        ['2026-11-10T05:06:26.147169Z', '2026-11-10T05:06:26.147169+00:00', true, 'one instant in two spellings is a replay'],
        ['2026-11-10T05:06:26.147Z', '2026-11-10T05:06:26.147169+00:00', true, 'a page stamp has milliseconds: .147169 is not older than .147'],
        ['2026-11-10T05:06:26.147Z', '2026-11-10T05:06:26.146999+00:00', false, 'and .146999 is'],
        ['2026-11-10T05:06:26.5Z', '2026-11-10T05:06:26.499999Z', false, 'one fraction digit is tenths, never microseconds'],
        ['2026-11-10T05:06:26.148000+00:00', '2026-11-10T05:06:26.147999+00:00', false, 'no rounding across a millisecond'],
        ['2026-11-10T05:06:27+00:00', '2026-11-10T05:06:26.999999+00:00', false, 'no rounding across a second'],
        ['2026-11-10T05:06:26.147857+00:00', '2026-11-10T08:06:26.147169+03:00', false, 'another offset is the same clock'],
        ['2026-11-10T05:06:26.147169+00:00', '2026-11-10T08:06:26.147169+03:00', true, 'the same instant from another offset is a replay'],
    ]) assert.equal(verdict(clock, revision), current, why);
});

test('a moment the guard cannot read still never rejects a late outcome', () => {
    const time = '2026-11-10T05:06:26.147169+00:00';
    const card = (latestSourceTs) => ({ finished: true, latestSourceTs });
    for (const clock of ['', undefined, null, 'not a time']) {
        assert.equal(lateOutcomeIsCurrent(card(clock), { current_outcome: { revision: time } }), true, String(clock));
    }
    for (const evt of [{ current_outcome: {} }, { current_outcome: { revision: 'not a time' } }, { ts: '' }, {}]) {
        assert.equal(lateOutcomeIsCurrent(card(time), evt), true, JSON.stringify(evt));
    }
});
