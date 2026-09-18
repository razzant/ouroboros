import assert from 'node:assert/strict';
import test from 'node:test';
import { readdirSync, readFileSync } from 'node:fs';

import {
    taskDoneIsTerminal, taskPresentation, taskReasonDetail, taskReasonPhrase, taskTerminalPhase,
} from '../modules/log_events.js';

// A degraded delivery used to name one generic cause on every card. The record
// keeps the machine code; the card says what actually happened.

test('a typed cause is stated in the owner\'s words', () => {
    // No 'Reason:' / 'Acceptance:' prefix: a label naming an internal machine
    // concept in front of an owner sentence is the same leak in a politer font.
    assert.equal(
        taskReasonDetail({ reason_code: 'plan_review_advisory' }),
        'Plan review never closed; the work continued under advisory enforcement',
    );
    assert.equal(
        taskReasonDetail({ reason_code: 'delivery_control_degraded' }),
        'Delivery finished in a degraded control state',
    );
});

test('an unknown cause stays raw rather than becoming a wrong sentence', () => {
    assert.equal(taskReasonDetail({ reason_code: 'some_future_code' }), 'some_future_code');
    assert.equal(taskReasonPhrase('some_future_code'), 'some_future_code');
});

test('no cause and an owner-requested stop both render nothing', () => {
    assert.equal(taskReasonDetail({}), '');
    assert.equal(taskReasonDetail({ reason_code: '' }), '');
    // The soft stop is a SUCCESS and carries its own marker instead.
    assert.equal(taskReasonDetail({ reason_code: 'owner_requested_finalization' }), '');
});

test('every typed cause the loop can record has a sentence', () => {
    // Cross-language completeness: the reachable literals live in the Python
    // loop, the sentences live here, and a new cause must not silently fall
    // back to its machine code on the card.
    // v7 split ouroboros/loop.py into leaves — the reachable literals now sit in
    // loop_budget.py and loop_delivery.py — so scan the whole loop family
    // instead of the one file that used to hold them all.
    const pkg = new URL('../../ouroboros/', import.meta.url);
    const loop = readdirSync(pkg)
        .filter((name) => /^loop.*\.py$/.test(name))
        .sort()
        .map((name) => readFileSync(new URL(name, pkg), 'utf8'))
        .join('\n');
    const literals = new Set(
        [...loop.matchAll(/degraded_reason\s*=\s*"([a-z_]+)"/g)].map((m) => m[1]),
    );
    assert.ok(literals.size >= 3, `expected the known typed causes, saw ${[...literals]}`);
    for (const code of literals) {
        assert.notEqual(
            taskReasonPhrase(code), code,
            `no owner-facing sentence for degraded_reason "${code}" — add one to TASK_CAUSE_PHRASES`,
        );
    }
});

test('an accepted decision with a sentence still states its cause', () => {
    // Owner fork 1=B (2026-09-16): reviewers who approved the earlier revision
    // accept the task, and the row says which revision they approved. A clean
    // accepted decision keeps rendering nothing.
    const accepted = (reason) => taskReasonDetail({
        status: 'completed', reason_code: 'final_message',
        outcome_axes: { execution: { status: 'ok' },
            review: { status: 'pass', acceptance_decision: { status: 'accepted', reason } } },
    });
    assert.equal(accepted('previous_revision_accepted'),
        'The reviewers approved an earlier version of this answer; the current version was not re-reviewed.');
    assert.equal(accepted('clean_pass'), '');
    assert.equal(accepted(''), '');
});

test('every acceptance reason the host can record has a sentence', () => {
    // The second half of the same gate. Acceptance reasons are written as
    // `"reason": "<code>"` inside the four acceptance/finalization/settlement
    // leaves; the bypass family is a dict of literals in outcomes.py, four more
    // arrive through named constants there and the settlement leaf names its
    // own reason as a constant, so all three shapes are read explicitly.
    const pkg = new URL('../../ouroboros/', import.meta.url);
    const read = (name) => readFileSync(new URL(name, pkg), 'utf8');
    const decisions = [
        'loop_acceptance_review.py', 'loop_acceptance.py', 'loop_forced_finalization.py', 'acceptance_settlement.py',
    ].map(read).join('\n');
    const outcomes = read('outcomes.py');
    const acceptance = new Set([
        ...[...decisions.matchAll(/"reason":\s*(?:\n\s*)?"([a-z_]+)"/g)].map((m) => m[1]),
        ...[...outcomes.matchAll(/"(acceptance_bypassed_[a-z_]+)"/g)].map((m) => m[1]),
        ...[...outcomes.matchAll(
            /^REASON_(?:REVIEW_CYCLES_EXHAUSTED|IDENTICAL_ACCEPTANCE_REFUSED|ACCEPTANCE_REVIEW_SKIPPED_DEADLINE_RESERVE|ACCEPTANCE_SKIPPED_OWNER_HURRY) = "([a-z_]+)"$/gm,
        )].map((m) => m[1]),
        ...[...read('acceptance_settlement.py').matchAll(/^REASON_[A-Z_]+ = "([a-z_]+)"$/gm)].map((m) => m[1]),
    ]);
    assert.ok(acceptance.has('previous_revision_accepted'), 'the settlement leaf is scanned');
    // A CLEAN accepted decision renders no clause, the owner stop carries its
    // own marker instead, and queue_inspection_failed is a `{status, reason}`
    // probe shape rather than an acceptance reason.
    const exempt = new Set([
        'clean_pass', 'clean_pass_obligations_closed', 'queue_inspection_failed',
        'acceptance_bypassed_owner_requested_finalization',
    ]);
    assert.ok(acceptance.size >= 20, `expected the acceptance vocabulary, saw ${acceptance.size}`);
    for (const code of acceptance) {
        if (exempt.has(code)) continue;
        assert.notEqual(
            taskReasonPhrase(code), code,
            `no owner-facing sentence for acceptance reason "${code}" — add one to TASK_CAUSE_PHRASES`,
        );
    }
});

test('one status-word family: the card phase matches the host over the shared fixture', () => {
    // The same fixture is read by tests/test_project_plain_rows.py, so a
    // divergence between this severity fold and the host's durable label word
    // fails on both sides of the boundary.
    const fixture = JSON.parse(
        readFileSync(new URL('./fixtures/outcome_phase_parity.json', import.meta.url), 'utf8'),
    );
    assert.ok(fixture.cases.length >= 10);
    for (const { name, record, phase, headline, acceptance_clause: clause } of fixture.cases) {
        const resolved = taskDoneIsTerminal(record) ? taskTerminalPhase(record) : 'working';
        assert.deepEqual(taskPresentation(resolved), { phase, headline }, name);
        if (clause) {
            // The host composes the same sentence for its durable prose rows and
            // terminates it there; the card line adds no punctuation of its own.
            const detail = taskReasonDetail(record);
            assert.ok(clause === detail || clause === `${detail}.`, `${name}: ${detail} vs ${clause}`);
        }
    }
});

// A review-caused warning used to be explained by whatever execution reason sat
// beside it ('Reason: final_message'), which named the delivery step rather than
// the actual cause. The host's acceptance decision now speaks for itself.

const A4 = {
    status: 'completed',
    reason_code: 'final_message',
    outcome_axes: {
        execution: { status: 'ok' },
        review: {
            status: 'degraded',
            acceptance_decision: {
                status: 'finalized_unaccepted',
                reason: 'review_degraded',
                rationale: 'Acceptance reviewers did not reach a valid quorum.',
            },
        },
    },
};

test('an unaccepted decision explains the warning in its own words', () => {
    assert.equal(
        taskReasonDetail(A4),
        'No reviewer verdict was established for this answer.',
    );
    assert.doesNotMatch(taskReasonDetail(A4), /final_message/);
    // The stored reviewer rationale belongs to the card body, the task result
    // and Logs; the row states one sentence and never the machine words.
    assert.doesNotMatch(taskReasonDetail(A4), /quorum|finalized_unaccepted/);
});

test('an accepted decision omits neutral final_message and preserves substantive reasons', () => {
    const accepted = {
        ...A4,
        outcome_axes: {
            execution: { status: 'ok' },
            review: { status: 'pass', acceptance_decision: { status: 'accepted', rationale: 'Quorum reached.' } },
        },
    };
    assert.equal(taskReasonDetail(accepted), '');
    assert.equal(taskReasonDetail({ ...accepted, reason_code: 'custom_reason' }), 'custom_reason');
});

test('a decision carrying no typed reason states no cause at all', () => {
    // A status word is not a cause, and the collapsed status is already the
    // headline; a historical record without a reason therefore says nothing.
    const record = {
        outcome_axes: { review: { acceptance_decision: { status: 'revision_requested' } } },
        status: 'completed',
    };
    assert.equal(taskReasonDetail(record), '');
});

test('a decision with no reason code still reaches the acceptance branch', () => {
    // The old single early return swallowed this frame before the branch.
    const record = {
        status: 'completed',
        review_status: { acceptance_decision: { status: 'revision_requested', reason: 'owner_followup' } },
    };
    assert.equal(taskReasonDetail(record), 'A new message from you arrived, so the review was set aside for it.');
});

test('a hard failure keeps explaining itself by its execution reason', () => {
    // The row carries the debt the code names: the host renderer
    // (project_dialogue._completion_verdict) states this cause only while
    // delegated_runs_unreconciled is non-empty, because the debt heals from the
    // write side while the stored reason_code may not be rewritten. The fixture
    // keeps the two twins describing the same record.
    const failed = {
        ...A4, status: 'failed', reason_code: 'delegated_custody_unreconciled',
        delegated_runs_unreconciled: ['run-a1'],
    };
    assert.equal(taskReasonDetail(failed), 'Some delegated work was never reconciled.');
});

test('a stored rationale never reaches the row, however it is written', () => {
    // The rationale used to be flattened into the line; it is free reviewer
    // text up to 500 characters and belongs where the full copy lives.
    const noisy = {
        status: 'completed',
        outcome_axes: {
            review: {
                acceptance_decision: {
                    status: 'revision_requested', reason: 'evidence_refresh',
                    rationale: 'Two\n\nlines   here.',
                },
            },
        },
    };
    assert.equal(
        taskReasonDetail(noisy),
        'The work changed after the review was frozen, so it no longer covered the answer.',
    );
    assert.doesNotMatch(taskReasonDetail(noisy), /lines/);
});

// The custody overlay stamps `delegated_custody_unreconciled` as the row's
// reason_code while a delegated run is still unreconciled. The debt then heals
// from the WRITE side while the stored code may not be rewritten, so the code
// outlives the fact. The host renderer already selects on the row's own debt
// list (project_dialogue._custody_debt_reason); these cases mirror
// tests/test_terminal_truth_projection_p5.py so the card and the plain rows
// name the same cause on the same record.

test('a healed custody debt yields the current execution reason on the card', () => {
    assert.equal(taskReasonDetail({
        status: 'completed',
        reason_code: 'delegated_custody_unreconciled',
        delegated_runs_unreconciled: [],
        outcome_axes: { execution: { status: 'degraded', reason_code: 'tool_failure' } },
    }), 'tool_failure');
});

test('an open custody debt is still named on the card', () => {
    assert.equal(taskReasonDetail({
        status: 'completed',
        reason_code: 'delegated_custody_unreconciled',
        delegated_runs_unreconciled: ['run-a1'],
        outcome_axes: { execution: { status: 'ok' } },
    }), 'Some delegated work was never reconciled.');
});

test('a real debt beside a real execution cause is one card line', () => {
    assert.equal(taskReasonDetail({
        status: 'failed',
        reason_code: 'delegated_custody_unreconciled',
        delegated_runs_unreconciled: ['run-a1', 'run-b2'],
        outcome_axes: { execution: { status: 'failed', reason_code: 'provider_unavailable' } },
    }), 'provider_unavailable (Some delegated work was never reconciled.)');
});

// A LIVE task_done event carries the row's own debt list too
// (agent_task_pipeline._custody_debt_event_fields), so the card reads the SAME
// list the durable row holds and the stamped code is never a second source. A
// record that carries no readable list therefore states nothing about the debt,
// exactly as project_dialogue._custody_debt_reason reads the same record.

test('a live event carrying an open debt list names the debt through the list', () => {
    assert.equal(taskReasonDetail({
        status: 'completed',
        reason_code: 'delegated_custody_unreconciled',
        delegated_runs_unreconciled: ['run-a1'],
        outcome_axes: { execution: { status: 'ok', reason_code: 'tool_failure' } },
    }), 'tool_failure (Some delegated work was never reconciled.)');
});

test('a record carrying no debt list states nothing about the debt', () => {
    assert.equal(taskReasonDetail({
        status: 'failed',
        reason_code: 'delegated_custody_unreconciled',
        outcome_axes: { execution: { status: 'failed', reason_code: 'provider_unavailable' } },
    }), 'provider_unavailable');
    assert.equal(taskReasonDetail({
        status: 'completed',
        reason_code: 'delegated_custody_unreconciled',
        outcome_axes: { execution: { status: 'ok' } },
    }), '');
});

test('a debt list of an unreadable shape carries no readable debt', () => {
    // The host reads the same value the same way: only a list is a debt list.
    assert.equal(taskReasonDetail({
        status: 'completed',
        reason_code: 'delegated_custody_unreconciled',
        delegated_runs_unreconciled: 'run-a1',
        outcome_axes: { execution: { status: 'ok' } },
    }), '');
});

test('a healed debt with no execution cause states nothing on the card', () => {
    // The warn headline still comes from the frozen objective warning; inventing
    // a Reason line here is exactly the false statement this rule removes.
    assert.equal(taskReasonDetail({
        status: 'completed',
        reason_code: 'delegated_custody_unreconciled',
        delegated_runs_unreconciled: [],
        outcome_axes: { execution: { status: 'ok' } },
    }), '');
});
