import assert from 'node:assert/strict';
import test from 'node:test';

import {
    taskOutcomeSeverity, taskPresentation, taskReasonDetail, taskTerminalPhase, taskTerminalSummary,
} from '../modules/log_events.js';

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
    assert.equal(taskReasonDetail(warned), '');
    const failed = done({ execution, artifacts: { status: 'missing' } });
    assert.equal(taskOutcomeSeverity(failed), 'error');
    assert.equal(taskReasonDetail(failed), '');
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

test('the awaited fact never outranks a decision sentence, a rail reason or an owner stop', () => {
    const awaited = { status: 'ok', plan_review: 'awaiting' };
    const decided = done({
        execution: awaited,
        review: { status: 'pass', acceptance_decision: { status: 'finalized_unaccepted', reason: 'capsule_spent' } },
    });
    assert.equal(taskReasonDetail(decided), 'The one allowed improvement pass was already used.');
    const railed = { ...done({ execution: { ...awaited, status: 'best_effort' } }), reason_code: 'round_limit' };
    assert.equal(taskReasonDetail(railed), 'The task hit its round limit before it could finish cleanly');
    assert.equal(taskOutcomeSeverity(railed), 'warn');
    const stopped = { ...done({ execution: awaited }), reason_code: 'owner_requested_finalization' };
    assert.equal(taskReasonDetail(stopped), '');
    // Without the typed fact a clean finish still states nothing.
    assert.equal(taskReasonDetail(done({ execution: { status: 'ok' } })), '');
    assert.equal(taskReasonDetail(done({ execution: { status: 'ok', plan_review: 'open' } })), '');
});

test('an acceptance panel that was only awaited is not a warning', () => {
    assert.equal(taskOutcomeSeverity(done({ execution: { status: 'ok' }, review: { status: 'awaiting' } })), 'done');
    const authored = done({
        execution: { status: 'ok' },
        objective: { status: 'pass', source: 'author_acceptance', review_status: 'awaiting' },
        review: {
            status: 'awaiting', eligibility: 'review_in_flight',
            acceptance_decision: { status: 'finalized_unaccepted', reason: 'author_finish' },
        },
    });
    assert.equal(taskOutcomeSeverity(authored), 'done');
    assert.equal(
        taskReasonDetail(authored),
        'Ouroboros delivered this answer on its own judgement; the reviewers had not signed it off.',
    );
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
