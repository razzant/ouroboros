import assert from 'node:assert/strict';
import test from 'node:test';

import {
    summarizeLogEvent, taskReasonDetail, taskTerminalPhase, taskTerminalSummary, terminalDeliveryClause,
} from '../modules/log_events.js';

// The Done card states a delivery the host could not confirm (owner 20:48: unknown and
// exhausted deliveries are shown honestly): the projector writes the row-level
// objective.warning the pill reads plus the typed terminal_delivery the cause line renders.

const EXHAUSTED = 'Sending the final answer was not confirmed after the delivery attempts ran out';
const UNKNOWN = 'Delivery of the final answer could not be confirmed because the delivery registry could not be read; the result is in task details.';

function card(delivery, extra = {}) {
    const row = {
        type: 'task_summary', summary_kind: 'terminal_root_projection', outcome_final: true,
        task_id: 't1', status: 'completed', reason_code: 'final_message',
        outcome_axes: { lifecycle: { status: 'completed' }, execution: { status: 'ok' } },
        ...extra,
    };
    if (delivery) {
        row.terminal_delivery = delivery;
        row.outcome_axes = { ...row.outcome_axes, objective: { warning: `terminal_delivery_${delivery.state}` } };
    }
    return row;
}

test('an exhausted delivery renders as Done with warnings and names the saved copy', () => {
    const row = card({ state: 'exhausted', delivery_id: 'final:t1:abcdef0123456789', preserved_path: '/data/observability/salvaged/t1.undelivered.txt' });
    assert.equal(taskTerminalPhase(row), 'warn');
    const summary = taskTerminalSummary(row);
    assert.equal(summary.headline, 'Done with warnings');
    const sentence = `${EXHAUSTED}; a copy of the answer is saved at /data/observability/salvaged/t1.undelivered.txt.`;
    assert.equal(taskReasonDetail(row), sentence);
    assert.ok(summary.body.includes(sentence), 'the card body carries the sentence');
    assert.equal(summary.activityPreview, sentence, 'the collapsed line states it too');
    // The Logs/replay half keeps the machine fact beside the sentence.
    const replay = summarizeLogEvent({ ...row, type: 'task_done' });
    assert.deepEqual({ phase: replay.phase, headline: replay.headline }, { phase: 'warn', headline: 'Done with warnings' });
});

test('an exhausted delivery without a preserved copy never claims one', () => {
    const row = card({ state: 'exhausted', preserved_path: '' });
    assert.equal(taskReasonDetail(row), `${EXHAUSTED}; no copy of the answer could be saved.`);
    assert.doesNotMatch(taskReasonDetail(row), /saved at/);
});

test('an unreadable delivery registry is shown, not hidden', () => {
    const row = card({ state: 'unknown' });
    assert.equal(taskTerminalPhase(row), 'warn');
    assert.equal(taskReasonDetail(row), UNKNOWN);
    assert.equal(taskTerminalSummary(row).headline, 'Done with warnings');
});

test('a failed or cancelled task keeps its own headline and adds the delivery clause after its cause', () => {
    const failed = card({ state: 'unknown' }, {
        status: 'failed', reason_code: 'idle_timeout',
        outcome_axes: { lifecycle: { status: 'failed' }, execution: { status: 'infra_failed', reason_code: 'idle_timeout' } },
    });
    assert.equal(taskTerminalPhase(failed), 'error');
    assert.equal(taskReasonDetail(failed), `The task made no progress for too long · ${UNKNOWN}`);
    const cancelled = card({ state: 'exhausted', preserved_path: '/c.txt' }, {
        status: 'cancelled', outcome_axes: { lifecycle: { status: 'cancelled' } },
    });
    assert.equal(taskTerminalPhase(cancelled), 'cancelled');
    assert.ok(taskReasonDetail(cancelled).endsWith(`${EXHAUSTED}; a copy of the answer is saved at /c.txt.`));
});

test('a card without a delivery fact renders no delivery clause', () => {
    assert.equal(taskReasonDetail(card(null)), '');
    assert.equal(terminalDeliveryClause(undefined), '');
    assert.equal(terminalDeliveryClause({ state: 'delivered' }), '');
    assert.equal(taskTerminalPhase(card(null)), 'done');
});

test('a recovered attempt finalized from its registered answer states that cause', () => {
    const row = card(null, {
        status: 'failed', reason_code: 'finalization_incomplete',
        outcome_axes: { lifecycle: { status: 'failed' }, execution: { status: 'infra_failed', reason_code: 'finalization_incomplete' } },
    });
    assert.equal(taskReasonDetail(row),
        "The answer was authored, but the task's finalization did not complete, so its review evidence, artifacts and cost accounting are incomplete.");
    assert.equal(taskTerminalPhase(row), 'error');
    // The model's own final answer, delivered: the same record reads Done and says what was not stored.
    const delivered = { ...row, terminal_origin: 'model_final', finalization_recovery: { cause: 'app_stop', answer: 'delivered' } };
    assert.equal(taskTerminalPhase(delivered), 'done');
    assert.equal(taskReasonDetail(delivered),
        'The answer was delivered. The task stopped before its review evidence, artifacts and cost accounting were stored.');
    for (const kept of [{ terminal_origin: 'host_salvage' }, { finalization_recovery: { answer: 'owed' } }]) {
        assert.equal(taskTerminalPhase({ ...delivered, ...kept }), 'error');
    }
});
