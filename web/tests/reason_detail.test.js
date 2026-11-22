import assert from 'node:assert/strict';
import test from 'node:test';
import { readdirSync, readFileSync } from 'node:fs';
import { buildTimelineItemHtml, cardMetaKeys } from '../modules/chat_activity.js';
import { createChatInstance } from '../modules/chat.js';
import { ElementStub, installDom, restoreDom, walkCard } from './chat_dom_fixture.js';

import {
    childCauseFacts, childLineCause, reasonCodesNote, taskDoneIsTerminal, taskOutcomeSeverity, taskPresentation, taskReasonCodes, taskReasonDetail, taskReasonPhrase, taskTerminalPhase,
    summarizeChatLiveEvent,
} from '../modules/log_events.js';
import { timelineReadFields } from '../modules/chat_history_replay.js';
import { updateLiveTimelineItem } from '../modules/chat_render_batch.js';
import { TASK_CAUSE_PHRASES } from '../modules/task_reason_words.js';
import { noteHandoverAttempt, terminalChildFrame } from '../modules/settled_card.js';

const UNDESCRIBED = 'The recorded reason has no description yet.';
// The author's stop sentence as a clause (the table's words are review's; the place is pinned here).
const STOPPED = TASK_CAUSE_PHRASES.author_stop.replace(/\.$/, '');

// A degraded delivery used to name one generic cause on every card. The record
// keeps the machine code; the card says what actually happened.

test('a typed cause is stated in the owner\'s words', () => {
    // No 'Reason:' / 'Acceptance:' prefix: a label naming an internal machine
    // concept in front of an owner sentence is the same leak in a politer font.
    assert.equal(
        taskReasonDetail({ reason_code: 'plan_review_advisory' }),
        'The plan review was never closed; the work went on with what the reviewers said.',
    );
    assert.equal(
        taskReasonDetail({ reason_code: 'delivery_control_degraded' }),
        "Ouroboros's final delivery instruction could not be applied, so the answer stands as delivered.",
    );
});

test('a typed cause with no sentence yet says so; its exact code is the details\' (owner 21.10, 2A)', () => {
    assert.equal(taskReasonDetail({ reason_code: 'some_future_code' }), UNDESCRIBED);
    assert.deepEqual(taskReasonCodes({ reason_code: 'some_future_code' }), ['some_future_code']);
    assert.equal(reasonCodesNote(['some_future_code']), 'Recorded reason: some_future_code');
    assert.equal(reasonCodesNote([]), '');
    // The bare lookup (the Continue note, the host-text twin) still never borrows a sentence: it stays raw.
    assert.equal(taskReasonPhrase('some_future_code'), 'some_future_code');
    // A real producer's cause has its words and leaves nothing for the details.
    const crashed = { status: 'failed', reason_code: 'worker_crash_signal' };
    assert.equal(taskReasonDetail(crashed), 'The worker process running the task crashed');
    assert.deepEqual(taskReasonCodes(crashed), []);
    // Two undescribed codes are one neutral clause and both codes, in order.
    const two = { status: 'completed', reason_code: 'first_future_code', outcome_axes: {
        execution: { status: 'degraded', reason_code: 'first_future_code' },
        review: { acceptance_decision: { status: 'finalized_unaccepted', reason: 'second_future_code' } } } };
    assert.equal(taskReasonDetail(two), UNDESCRIBED);
    assert.deepEqual(taskReasonCodes(two), ['first_future_code', 'second_future_code']);
    assert.equal(reasonCodesNote(taskReasonCodes(two)), 'Recorded reasons: first_future_code, second_future_code');
    // Authored words are never censored: the author's own stop reason stays beside the neutral clause.
    const stop = { status: 'completed', reason_code: 'final_message', outcome_axes: {
        execution: { status: 'ok', task_completion: { action: 'stop', rationale: 'The export still fails on empty input.' } },
        objective: { status: 'fail', source: 'task_completion', reason: 'author_stop', warnings: ['partial coverage'] } } };
    assert.equal(taskReasonDetail(stop), `${STOPPED} · partial coverage · The export still fails on empty input.`);
    assert.deepEqual(taskReasonCodes(stop), [], 'a legacy warning may be authored prose, not a typed reason code');
    // The card's collapsed line says the sentence; its details name the code.
    const summary = summarizeChatLiveEvent({ type: 'task_done', task_id: 't', status: 'failed', reason_code: 'example_new_reason' });
    assert.equal(summary.activityPreview, UNDESCRIBED);
    assert.equal(summary.body, `${UNDESCRIBED}\nRecorded reason: example_new_reason`);
});

test('a foreign install keeps its preserved explanation despite the old terminal outcome', (t) => {
    const { prior } = installDom();
    t.after(() => restoreDom(prior));
    const detail = '/installs/<install_A>/task_drives/t belongs to another Ouroboros install at /installs/<install_A>; move it explicitly to work on it here';
    for (const prior of [
        { status: 'completed' },
        { status: 'failed' },
        { status: 'cancelled', cancel_origin: { actor: 'owner', reason: 'old stop' } },
        { status: 'failed', outcome_axes: { objective: { status: 'fail', source: 'plan_review_gate', reason: 'plan_review_advisory' } } },
    ]) {
        const record = { ...prior, reason_code: 'foreign_install_root', reason_detail: detail };
        assert.equal(taskReasonDetail(record), detail + (prior.status === 'cancelled' ? '.' : ''));
    }
    const summary = summarizeChatLiveEvent({ type: 'task_done', task_id: 't', status: 'failed',
        reason_code: 'foreign_install_root', reason_detail: detail });
    assert.equal(summary.body, detail);
    assert.equal(summary.activityPreview, detail);
    const html = buildTimelineItemHtml({ ...summary, lineKey: 'foreign' }, { expandedLineKeys: new Set() });
    assert.ok(html.includes('/installs/&lt;install_A&gt;/task_drives/t'));
    assert.ok(html.includes('move it explicitly to work on it here'));
    assert.ok(!html.includes('<install_A>'));
});

test('ordinary reasons ignore foreign detail and an absent detail stays explicit', () => {
    const reason_detail = 'This is not the reason for an ordinary task.';
    assert.equal(taskReasonDetail({ reason_code: 'final_message', reason_detail }), '');
    assert.equal(taskReasonDetail({ reason_code: 'some_future_code', reason_detail }), UNDESCRIBED);
    assert.equal(taskReasonDetail({ reason_code: 'foreign_install_root' }), UNDESCRIBED);
    assert.deepEqual(taskReasonCodes({ reason_code: 'foreign_install_root' }), ['foreign_install_root']);
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
    // A normal exit's own suffix marks are declared once (loop_delivery._SUFFIX_MARKS).
    for (const [, members] of loop.matchAll(/_SUFFIX_MARKS\s*=\s*frozenset\(\{([^}]*)\}\)/g)) {
        for (const [, code] of members.matchAll(/"([a-z_]+)"/g)) literals.add(code);
    }
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
    // Owner 2A (2026-09-21): a blocking install accepts a reviewer-approved answer whose
    // admission close the supervisor never confirmed, and the row says so.
    assert.equal(accepted('admission_close_unconfirmed'),
        'Reviewers approved this answer; the supervisor did not confirm that task admission was closed.');
    assert.equal(accepted('clean_pass'), '');
    assert.equal(accepted(''), '');
});

test('every acceptance reason a stored record can carry has a sentence', () => {
    // The second half of the same gate. The separate acceptance machine is gone (one
    // review mechanism, owner 2026-10-17 Q2-A), but its records stay on disk and still
    // render: this is the vocabulary it could write, frozen when its writers were
    // deleted, so no old task card falls back to a raw code.
    const historical = [
        'acceptance_bypassed_budget_exhausted', 'acceptance_bypassed_children_unabsorbed',
        'acceptance_bypassed_context_overflow', 'acceptance_bypassed_deadline',
        'acceptance_bypassed_provider_unavailable', 'acceptance_bypassed_round_limit',
        'admission_close_unconfirmed', 'delivery_binding_superseded', 'evidence_refresh',
        'fence_reopen_failed', 'identical_acceptance_refused', 'improvement_capsule', 'open_obligations',
        'owner_followup', 'owner_hurry', 'previous_revision_accepted', 'review_cycles_exhausted',
        'review_degraded', 'review_outcome_received', 'review_skipped_deadline_reserve',
        'reviewer_fail_no_capsule',
    ];
    // What the host still writes for a result review: an informed author finish, or a
    // blocking install that could buy no review of the result (outcomes._review_axis).
    const outcomes = readFileSync(new URL('../../ouroboros/outcomes.py', import.meta.url), 'utf8');
    const current = [...outcomes.matchAll(/"reason":\s*"(author_finish)"/g)].map((m) => m[1]);
    assert.ok(current.includes('author_finish'), 'the result review decision is scanned');
    assert.match(outcomes, /^REASON_REVIEW_CYCLES_EXHAUSTED = "review_cycles_exhausted"$/m);
    for (const code of new Set([...historical, ...current])) {
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
    for (const { name, record, phase, headline, acceptance_clause: clause, described_clause: described,
        undescribed = [], exact_clause = false } of fixture.cases) {
        const resolved = taskDoneIsTerminal(record) ? taskTerminalPhase(record) : 'working';
        assert.deepEqual(taskPresentation(resolved), { phase, headline }, name);
        if (clause !== undefined) {
            // The host composes the same sentence for its durable prose rows and terminates it there; the card line adds
            // no punctuation of its own. The card has details, so an undescribed code reads neutrally here (owner 2A).
            const expected = described ?? clause;
            const detail = taskReasonDetail(record);
            assert.ok(expected === detail || expected === `${detail}.`, `${name}: ${detail} vs ${expected}`);
            if (exact_clause) assert.equal(detail, expected, name);
            assert.deepEqual(taskReasonCodes(record), undescribed, name);
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
        'The reviewers did not reach a verdict on this answer.',
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
    assert.equal(taskReasonDetail({ ...accepted, reason_code: 'custom_reason' }), UNDESCRIBED);
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

// Rows written before #1583 carry `delegated_custody_unreconciled` as their
// reason_code whenever a delegated run was unreconciled at the terminal; new rows
// keep their own cause. The debt heals from the WRITE side while the stored code may
// not be rewritten, so the host renderer selects on the row's own debt list
// (project_dialogue._custody_debt_reason); these cases mirror
// tests/test_terminal_truth_projection_p5.py so the card and the plain rows
// name the same cause on the same record.

test('a healed custody debt yields the current execution reason on the card', () => {
    assert.equal(taskReasonDetail({
        status: 'completed',
        reason_code: 'delegated_custody_unreconciled',
        delegated_runs_unreconciled: [],
        outcome_axes: { execution: { status: 'degraded', reason_code: 'tool_failure' } },
    }), 'A tool this task used failed and nothing recovered it');
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
    }), 'The task could not get a usable model answer, so it could not finish · Some delegated work was never reconciled.');
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
    }), 'A tool this task used failed and nothing recovered it · Some delegated work was never reconciled.');
});

test('a record carrying no debt list states nothing about the debt', () => {
    assert.equal(taskReasonDetail({
        status: 'failed',
        reason_code: 'delegated_custody_unreconciled',
        outcome_axes: { execution: { status: 'failed', reason_code: 'provider_unavailable' } },
    }), 'The task could not get a usable model answer, so it could not finish');
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
    // Inventing a Reason line here is exactly the false statement this rule removes;
    // the headline reads the same known-empty list (outcome_phase_parity).
    assert.equal(taskReasonDetail({
        status: 'completed',
        reason_code: 'delegated_custody_unreconciled',
        delegated_runs_unreconciled: [],
        outcome_axes: { execution: { status: 'ok' } },
    }), '');
});

test('the census row and the final frame read the current debt list beside the cause it never replaces', async () => {
    // #1583: the live census (display keys) and the final frame (progress_meta) carry the row's list,
    // so an older stamp heals with it and a new row warns from the list alone.
    const { censusTaskFacts } = await import('../modules/task_phase_chip.js');
    const stamp = 'delegated_custody_unreconciled';
    const answered = {
        status: 'completed', reason_code: stamp, root_phase_checkpoint: { post_task_synthesis: 'running' },
        outcome_axes: { execution: { status: 'ok', reason_code: 'final_message' },
            objective: { status: 'not_evaluated', source: 'none', warning: stamp, warnings: [stamp] } },
    };
    assert.equal(censusTaskFacts({ ...answered, delegated_runs_unreconciled: [] }).outcome, 'done');
    assert.equal(censusTaskFacts({ ...answered, delegated_runs_unreconciled: ['run-1'] }).outcome, 'warn');
    assert.equal(censusTaskFacts(answered).outcome, 'warn', 'no list is no proof of reconciliation');
    const frame = {
        task_id: 'root', task_terminal_status: 'completed', reason_code: 'final_message',
        delegated_runs_unreconciled: ['run-1'], outcome_axes: { execution: { status: 'ok', reason_code: 'final_message' } },
    };
    assert.equal(taskTerminalPhase(frame), 'warn');
    assert.equal(taskReasonDetail(frame), 'Some delegated work was never reconciled.');
    assert.equal(taskTerminalPhase({ ...frame, delegated_runs_unreconciled: [] }), 'done');
});

test('a legacy failed stamp read back as its own execution reason names no current debt', () => {
    // R1 QA (synthetic-stamp-no-cause): the history row an earlier server normalized carries the stamp as
    // execution.reason_code; with the debt list known empty the card names no debt and invents no cause.
    const stamp = 'delegated_custody_unreconciled';
    const replayed = {
        task_id: 'synthetic-stamp-no-cause', task_terminal_status: 'failed', reason_code: stamp,
        delegated_runs_unreconciled: [], outcome_phase: 'error', outcome_final: true,
        continuation_offer: { cause: stamp, eligible: false, refusal: 'interruption_cause_unrecorded' },
        outcome_axes: {
            artifacts: { status: 'not_applicable' }, execution: { reason_code: stamp, status: 'failed' },
            lifecycle: { status: 'failed' }, review: { status: 'skipped', trigger: 'not_evaluated' }, schema_version: 1,
            objective: { source: 'none', status: 'not_evaluated', warning: stamp, warnings: [stamp] },
        },
    };
    assert.equal(taskTerminalPhase(replayed), 'error');
    assert.equal(taskReasonDetail(replayed), '');
    assert.equal(taskReasonDetail({ ...replayed, delegated_runs_unreconciled: ['run-1'] }),
        'Some delegated work was never reconciled.', 'a current debt still speaks');
    // synthetic-debt-unknown: an older finished row carrying no list keeps its warning headline.
    const { delegated_runs_unreconciled: _known, ...unlisted } = replayed;
    const finished = { ...unlisted, task_terminal_status: 'completed', outcome_phase: 'warn', outcome_axes: {
        ...replayed.outcome_axes, execution: { reason_code: 'final_message', status: 'ok' },
        lifecycle: { status: 'completed' } } };
    assert.equal(taskTerminalPhase(finished), 'warn', 'no list is no proof of reconciliation');
    assert.equal(taskTerminalPhase({ ...finished, delegated_runs_unreconciled: [] }), 'done');
});

// The plan review's outcome CLASS at delivery (outcome_axes.execution.plan_review) picks the
// sentence for an open plan review; the stamped code stays `plan_review_advisory` on the
// record. A record naming no class keeps the general sentence; an unknown class word is
// never turned into a wrong sentence.

const openPlan = (plan_review, extra = {}) => ({
    status: 'completed', reason_code: 'plan_review_advisory', terminal_plan_review_open: true,
    outcome_axes: { execution: { status: 'degraded', reason_code: 'plan_review_advisory', ...(plan_review ? { plan_review } : {}) } },
    ...extra,
});

test('the plan review class picks the owner sentence for an open plan review', () => {
    assert.equal(taskReasonDetail(openPlan('unanswered')),
        'Only some of the plan reviewers answered; the work went on with their notes.');
    assert.equal(taskReasonDetail(openPlan('none_answered')),
        'None of the plan reviewers answered; the work went on without their notes.');
    assert.equal(taskReasonDetail(openPlan('answered_open')),
        'The plan reviewers answered, but the review was never closed; the work went on with their notes.');
    // Legacy rows without a class, and a class nobody has a sentence for, keep the general sentence.
    assert.equal(taskReasonDetail(openPlan('')),
        'The plan review was never closed; the work went on with what the reviewers said.');
    assert.equal(taskReasonDetail(openPlan('some_future_class')),
        'The plan review was never closed; the work went on with what the reviewers said.');
    // The class is a plan-review fact: another execution reason is never reworded by it, and the
    // class states the standing limitation beside that reason (it rides the live event and the
    // replayed row where the result-only flag does not); a record with neither stays silent.
    assert.equal(taskReasonDetail({
        status: 'completed', reason_code: 'budget_exhausted',
        outcome_axes: { execution: { status: 'degraded', reason_code: 'budget_exhausted', plan_review: 'unanswered' } },
    }), 'The task ran out of budget before it could finish cleanly · Only some of the plan reviewers answered; the work went on with their notes.');
    assert.equal(taskReasonDetail({
        status: 'completed', reason_code: 'budget_exhausted',
        outcome_axes: { execution: { status: 'degraded', reason_code: 'budget_exhausted' } },
    }), 'The task ran out of budget before it could finish cleanly');
});

// Owner 04.11, 1A: a cause says what changes the reader's next step. An author's stop is unfinished work, whatever
// the reviewers said (that is the record's and the Reviews door's); an answer delivered as finished says that the
// reviewers did not sign it off, because that changes how far the owner can trust it.
test("an author's stop names the stop alone; an answer it delivered says the reviewers did not sign it off", () => {
    const stop = { status: 'completed', reason_code: 'final_message', outcome_axes: {
        execution: { status: 'ok', task_completion: { action: 'stop', rationale: 'Not ready.' } },
        objective: { status: 'fail', source: 'task_completion', reason: 'author_stop' }, review: { status: 'skipped' } } };
    const delivered = { status: 'completed', outcome_axes: { review: {
        acceptance_decision: { status: 'finalized_unaccepted', reason: 'author_finish' } } } };
    assert.ok(taskReasonDetail(stop).startsWith(STOPPED) && !/review/i.test(taskReasonDetail(stop)), taskReasonDetail(stop));
    assert.equal(taskReasonDetail(delivered), TASK_CAUSE_PHRASES.author_finish);
    assert.match(taskReasonDetail(delivered), /review/i);
});

test('a held task states why it was held and never that the work went on', () => {
    const held = (source, reason) => taskReasonDetail({
        status: 'completed', reason_code: 'final_message',
        outcome_axes: { execution: { status: 'ok' }, review: { status: 'skipped' },
            objective: { status: 'fail', source, reason, outcome_tier: 'blocked_with_evidence' } },
    });
    assert.equal(held('plan_review_quorum_unreachable', 'plan_review_quorum_unreachable'),
        'Too few plan reviewers could answer, so the work was held.');
    assert.equal(held('plan_review_cycles_exhausted', 'review_cycles_exhausted'),
        'The task used up its review rounds before the answer was signed off.');
    assert.equal(held('plan_review_author_stop', 'author_stop'), TASK_CAUSE_PHRASES.author_stop);
    for (const line of [held('plan_review_quorum_unreachable', 'plan_review_quorum_unreachable')]) {
        assert.doesNotMatch(line, /went on/);
    }
    // A Failed card held by anything else keeps stating nothing on a neutral final_message.
    assert.equal(held('task_acceptance_review', 'review_fail'), '');
});

// An exit as the loop records it (tests/test_terminal_provenance.py): the plan class, the reason its notice stamped,
// and the author's stop when it ended the work. A stop is the primary cause and no plan sentence says the work went
// on; an unreadable record stays unknown beside a stop, a rail or a clean finish.
const UNKNOWN = 'The plan record could not be read, so the state of the plan review is unknown.';
const planExit = (plan_review, { reason = 'plan_review_advisory', status = 'degraded', stop = '' } = {}) => ({
    status: 'completed', reason_code: reason,
    outcome_axes: {
        execution: { status, reason_code: reason, plan_review,
            ...(stop ? { task_completion: { action: 'stop', rationale: stop } } : {}) },
        objective: stop ? { status: 'fail', source: 'task_completion', reason: 'author_stop' } : { status: 'not_evaluated' },
        review: { status: 'skipped' },
    },
});

test("an author's stop never says the work went on, and an unreadable plan record stays unknown", () => {
    const held = 'The objection stands; stopping with the draft.';
    // Blocking held the answer over an open objection and the author stopped: the stop, its reason, the open review.
    assert.equal(taskReasonDetail(planExit('objections_open', { stop: held })),
        `${STOPPED} · The objection stands; stopping with the draft · The plan review was still open when this answer was delivered.`);
    // The same objection over an ordinary delivery: the work went on with it.
    assert.equal(taskReasonDetail(planExit('objections_open')),
        'Objections to the plan were still open when this answer was delivered; the work went on with them disclosed.');
    // An unreadable record ended by the author's stop: unknown, on the live frame and on the row with its flag.
    const unread = planExit('unreadable', { stop: 'The plan record cannot be read; stopping with the draft.' });
    for (const record of [unread, { ...unread, terminal_plan_review_open: true }]) {
        assert.equal(taskReasonDetail(record), `${STOPPED} · The plan record cannot be read; stopping with the draft · ${UNKNOWN}`);
    }
    // Cyber Pro stopped over it without a notice (no advisory reason): the class alone keeps it unknown.
    assert.equal(taskReasonDetail(planExit('unreadable', { reason: 'final_message', status: 'ok', stop: 'Stopping.' })),
        `${STOPPED} · Stopping · ${UNKNOWN}`);
    // A rail keeps its own cause, and a clean finish its silence, beside the unknown record.
    assert.equal(taskReasonDetail(planExit('unreadable', { reason: 'round_limit', status: 'best_effort' })),
        `The task hit its round limit before it could finish cleanly · ${UNKNOWN}`);
    assert.equal(taskReasonDetail(planExit('unreadable', { reason: 'final_message', status: 'ok' })), UNKNOWN);
    // An older stop over a class whose sentence says the work went on, or over none, reads the stop and the open review.
    for (const plan_review of ['unanswered', '']) {
        assert.equal(taskReasonDetail({ ...planExit(plan_review, { stop: held }), terminal_plan_review_open: true }),
            `${STOPPED} · The objection stands; stopping with the draft · The plan review was still open when this answer was delivered.`);
    }
    for (const line of [taskReasonDetail(planExit('objections_open', { stop: held })), taskReasonDetail(unread)]) {
        assert.doesNotMatch(line, /went on|proceeded/);
    }
});

test('standing limitations are stated beside the primary cause, each once', () => {
    const deferred = {
        status: 'completed', reason_code: 'child_results_deferred', terminal_plan_review_open: true,
        outcome_axes: { execution: { status: 'degraded', reason_code: 'child_results_deferred' },
            objective: { status: 'best_effort', source: 'child_result_disposition', deferred_count: 2 } },
    };
    assert.equal(taskReasonDetail(deferred),
        'Some sub-task results were deferred instead of being folded into this answer · The plan review was still open when this answer was delivered');
    // The open plan review is worded by its class when the record names one; the plan clause is second.
    assert.equal(taskReasonDetail({ ...deferred, outcome_axes: { ...deferred.outcome_axes,
        execution: { ...deferred.outcome_axes.execution, plan_review: 'unanswered' } } }),
    'Some sub-task results were deferred instead of being folded into this answer · Only some of the plan reviewers answered; the work went on with their notes.');
    // A clause that is not last drops its own full stop; equivalent clauses state themselves once.
    assert.equal(taskReasonDetail(openPlan('unanswered', { delegated_runs_unreconciled: ['run-a1'] })),
        'Only some of the plan reviewers answered; the work went on with their notes · Some delegated work was never reconciled.');
    assert.equal(taskReasonDetail({ ...openPlan('unanswered'), reason_code: 'delegated_custody_unreconciled',
        delegated_runs_unreconciled: ['run-a1'] }),
    'Only some of the plan reviewers answered; the work went on with their notes · Some delegated work was never reconciled.');
    // Without the flag or a deferred count the primary cause stands alone; an owner stop
    // contributes no primary cause but still carries its limitation.
    assert.equal(taskReasonDetail({ ...deferred, terminal_plan_review_open: false,
        outcome_axes: { execution: deferred.outcome_axes.execution } }),
    'Some sub-task results were deferred instead of being folded into this answer');
    assert.equal(taskReasonDetail({ ...deferred, reason_code: 'owner_requested_finalization' }),
        'Some sub-task results were deferred instead of being folded into this answer · The plan review was still open when this answer was delivered');
});


test('a child report keeps the typed plan state in its expanded detail without replacing its authored preview', () => {
    const result = 'Complete selected child answer.';
    for (const record of [planExit('unreadable'), planExit('unreadable', { reason: 'final_message', status: 'ok' }),
        planExit('unreadable', { reason: 'round_limit', status: 'best_effort' }), planExit('objections_open')]) {
        const view = summarizeChatLiveEvent({ ...record, type: 'send_message', is_progress: true,
            delegation_role: 'subagent', subagent_task_id: 'child', parent_task_id: 'root',
            subagent_event: 'completed', result });
        assert.equal(view.activityPreview, result);
        assert.ok(view.fullBody.endsWith(taskReasonDetail(record)), view.fullBody);
        if (record.outcome_axes.execution.plan_review === 'unreadable') {
            assert.ok(view.fullBody.includes(UNKNOWN));
            assert.doesNotMatch(view.fullBody, /went on|proceeded/);
        }
    }
});

test('an unfinished child keeps the complete authored result and its stop cause on the collapsed line', () => {
    const result = 'Complete selected answer, including what has already been done.';
    const outcome_axes = { execution: { status: 'ok', task_completion: {
        action: 'stop', rationale: 'The **export** still fails on empty input.',
    } }, objective: { status: 'fail', source: 'task_completion', reason: 'author_stop' } };
    const view = summarizeChatLiveEvent({ type: 'send_message', is_progress: true, delegation_role: 'subagent',
        subagent_task_id: 'child', parent_task_id: 'root', subagent_event: 'completed',
        status: 'completed', reason_code: 'final_message', outcome_axes, result });
    assert.equal(view.phase, 'cancelled');
    assert.equal(view.activityPreview, `${STOPPED} · The export still fails on empty input.`);
    assert.ok(view.fullBody.includes(result));
});

test('an owner stop over a held plan objective keeps the open-plan sentence beside its cause', () => {
    // Historical rows only (no live writer of plan_review_* objectives remains): the stop is phase cancelled, so the
    // plan hold that silences the sentence for an error does not apply; the sentence is the record's own fact.
    const record = { status: 'cancelled', cancel_origin: { actor: 'owner', reason: 'old stop' }, terminal_plan_review_open: true,
        outcome_axes: { objective: { status: 'fail', source: 'plan_review_cycles_exhausted', reason: 'plan_review_cycles_exhausted' },
            execution: { status: 'ok', plan_review: 'objections_open' } } };
    assert.ok(taskReasonDetail(record).includes('Objections to the plan were still open'), taskReasonDetail(record));
});

test('a settled child frame states its first coding attempt, its neighbours and no live debt the stored list denies', () => {
    const fact = { status: 'incomplete', reason: 'initial_execution_unresolved', initial_execution: { state: 'failed', run_id: 'run-1' } };
    const axes = { execution: { status: 'best_effort', reason_code: 'round_limit', actor_first_terminal: fact },
        objective: { status: 'degraded', warning: 'tool_failure' }, review: { status: 'skipped' } };
    // The adapter's own copy of the settled record (cardMetaKeys), as live task_done, replay and detail recovery hand it over.
    const frame = (subagent_event, debt) => summarizeChatLiveEvent({ type: 'send_message', is_progress: true,
        delegation_role: 'subagent', subagent_task_id: 'child', parent_task_id: 'root', subagent_event, status: 'completed',
        reason_code: 'round_limit', result: 'Answer.',
        ...cardMetaKeys({ outcome_axes: axes, ...(debt ? { delegated_runs_unreconciled: debt } : {}) }) });
    const cause = [TASK_CAUSE_PHRASES.round_limit.replace(/\.$/, ''), TASK_CAUSE_PHRASES.configured_actor_first_failed.replace(/\.$/, ''),
        TASK_CAUSE_PHRASES.tool_failure.replace(/\.$/, '')];
    for (const event of ['completed', 'completed_warn']) {  // the normal progress frame and the warn fallback read alike
        const live = frame(event, ['run-2']).fullBody, healed = frame(event, []).fullBody, unlisted = frame(event).fullBody;
        assert.ok(live.includes(`${cause.join(' · ')} · ${TASK_CAUSE_PHRASES.delegated_custody_unreconciled}`), `${event}: ${live}`);
        for (const body of [healed, unlisted]) {  // an explicit [] and an unlisted debt: neither is live debt
            assert.ok(body.endsWith(cause.join(' · ')), `${event}: ${body}`);
            assert.ok(!body.includes(TASK_CAUSE_PHRASES.delegated_custody_unreconciled), body);
        }
        assert.ok(live.includes('Answer.'));
    }
});

test('the legacy child custody stamp is not current debt when the stored list is empty, even without axes', () => {
    const legacy = { type: 'send_message', is_progress: true, delegation_role: 'subagent',
        subagent_task_id: 'child', parent_task_id: 'root', subagent_event: 'completed', status: 'completed',
        reason_code: 'delegated_custody_unreconciled', result: 'Answer.' };
    const warning = TASK_CAUSE_PHRASES.delegated_custody_unreconciled;
    assert.ok(summarizeChatLiveEvent(legacy).fullBody.includes(warning), 'a missing list has not reconciled the old stamp');
    const settled = summarizeChatLiveEvent({ ...legacy, delegated_runs_unreconciled: [] }).fullBody;
    assert.equal(settled, '## Report to parent\nAnswer.');
});

test('a settled child frame names the codes its cause leaves undescribed once in Details, and a described cause none', () => {
    const settled = (reason_code, execution = {}, extra = {}) => summarizeChatLiveEvent({ type: 'send_message', is_progress: true,
        delegation_role: 'subagent', subagent_task_id: 'child', parent_task_id: 'root', subagent_event: 'failed', status: 'failed',
        reason_code, result: 'Answer.', outcome_axes: { execution: { status: 'failed', reason_code, ...execution } }, ...extra }).fullBody;
    const occurrences = (text, part) => text.split(part).length - 1;
    const code = 'task_done_lifecycle_fault', note = reasonCodesNote([code]);
    const unknown = settled(code);  // a code with no sentence yet: the neutral sentence, then the exact code
    assert.ok(unknown.endsWith(`${UNDESCRIBED}\n\n${note}`), unknown);
    assert.equal(occurrences(unknown, note), 1, unknown);
    const described = settled('worker_crash_signal');  // a described cause reveals no code
    assert.ok(described.endsWith('The worker process running the task crashed'), described);
    assert.ok(!described.includes('Recorded reason'), described);
    // An author's stop states its cause through its own branch: the code is still named once, never twice.
    const stopped = settled(code, { status: 'ok', task_completion: { action: 'stop', rationale: 'The export still fails.' } },
        { subagent_event: 'completed', status: 'completed' });
    assert.ok(stopped.includes('The recorded reason has no description yet'), stopped);
    assert.equal(occurrences(stopped, note), 1, stopped);
});

// chat.js fetchFullLineOutput: past the wire's 4000 characters Show full reads the task and composes its line again, so the
// full result and trace come with what the capped Details stated (the first coding attempt, an undescribed code), once each.
const FULL_RESULT = 'REPORT_HEAD\n' + 'The retained answer has a complete paragraph.\n'.repeat(120) + 'REPORT_TAIL';
const FULL_TRACE = 'TRACE_HEAD\n' + 'Read the recorded evidence.\n'.repeat(160) + 'TRACE_TAIL';
const FIRST_ATTEMPT = { status: 'incomplete', reason: 'initial_execution_unresolved', initial_execution: { state: 'failed', run_id: 'run-1' } };
const DEBT = TASK_CAUSE_PHRASES.delegated_custody_unreconciled.replace(/\.$/, '');
const OPEN_PLAN = TASK_CAUSE_PHRASES.terminal_plan_review_open;
for (const [name, fields, stated, detailOnly = {}] of [
    ['a first coding attempt beside another primary cause', { status: 'completed', reason_code: 'round_limit', outcome_axes: {
        execution: { status: 'best_effort', reason_code: 'round_limit', actor_first_terminal: FIRST_ATTEMPT },
        objective: { status: 'degraded' }, review: { status: 'skipped' } }, delegated_runs_unreconciled: ['run-2'] },
    [TASK_CAUSE_PHRASES.configured_actor_first_failed.replace(/\.$/, '')], { terminal_plan_review_open: true }],
    ['a cause with no description yet', { status: 'failed', reason_code: 'task_done_lifecycle_fault', outcome_axes: {
        execution: { status: 'failed', reason_code: 'task_done_lifecycle_fault' } } },
    [UNDESCRIBED, 'Recorded reason: task_done_lifecycle_fault']],
]) {
    test(`Show full keeps what the capped Details stated, once, beside the full result and trace: ${name}`, async () => {
        const detail = { type: 'send_message', is_progress: true, chat_id: 2, task_id: 'child-1', delegation_role: 'subagent',
            parent_task_id: 'root-1', root_task_id: 'root-1', subagent_task_id: 'child-1', subagent_role: '',
            subagent_event: fields.status, model: 'x-ai/grok-4.7', ts: '2026-10-25T14:12:58Z', ...fields,
            result: FULL_RESULT, trace_summary: FULL_TRACE };
        const frame = { ...detail, history_id: 'child-terminal', role: 'system', content: '', text: '', result: FULL_RESULT.slice(0, 4000), result_truncated: true,
            trace_summary: FULL_TRACE.slice(0, 4000), trace_summary_truncated: true };
        const rows = [{ chat_id: 2, role: 'system', is_progress: true, content: 'Reading the brief.', narration: true,
            ts: '2026-10-25T13:35:40Z', task_id: 'root-1' }, frame];
        const reads = [];  // the line's full-output reads (chat.js fetchFullLineOutput), answered below
        const { prior, mount } = installDom(async (url, init = {}) => {
            if (String(url) === '/api/tasks/child-1' && init.cache === 'no-store') return new Promise((resolve) => reads.push(resolve));
            return { ok: true, json: async () => {
                if (String(url).startsWith('/api/chat/history')) return { messages: rows, page_cursor: 'recent', has_more: false,
                    next_cursor: null, window: { complete: true } };
                if (String(url) === '/api/tasks/child-1') return detail;  // the card's own detail read
                return { active_direct_turns: [] };
            } };
        });
        // This stub keeps parsed nodes flat: observe the real renderer's markup of the line.
        let rendered = '';
        const html = Object.getOwnPropertyDescriptor(ElementStub.prototype, 'innerHTML');
        Object.defineProperty(ElementStub.prototype, 'innerHTML', { ...html, set(value) {
            if (/^\s*<div\s+class="chat-live-line\s/.test(String(value)) && /REPORT_HEAD|Show full/.test(String(value))) rendered = String(value);
            html.set.call(this, value);
        } });
        const ws = { on() { return () => {}; }, isConnected: () => true, send() {} };
        let generation = 0;
        const stateSnapshots = { begin: () => ({ generation: ++generation, requestedAt: Date.now() }),
            gate() { return Promise.resolve(this.begin()); }, isCurrent: () => true, apply() {} };
        const instance = createChatInstance({ ws, state: { activePage: 'chat', projectChatIds: new Set(), unreadCount: 0 },
            updateUnreadBadge() {}, stateSnapshots, chatId: 2, idPrefix: 'chat', mountEl: mount, asPanel: true });
        try {
            await instance.refreshHistory({ revision: 1 });
            const card = walkCard(globalThis.document.byId.get('chat-messages'), 'child-1');
            assert.ok(card);
            card.querySelector('[data-live-disclosure]').click();
            await new Promise((resolve) => setTimeout(resolve, 0));
            const timeline = card.querySelector('[data-live-timeline]');
            const toggle = () => {
                const line = timeline.querySelector('.chat-live-line.expandable');
                // The fixture has no general closest/bubbling: dispatch the real listener.
                const target = { closest: (selector) => (selector === '.chat-live-line.expandable' ? line : null) };
                for (const fn of timeline.listeners.get('click') || []) fn({ target, stopPropagation() {} });
            };
            toggle();
            await new Promise((resolve) => setTimeout(resolve, 0));
            const capped = rendered;  // opened, the full read still pending
            assert.match(capped, /Loading full output/);
            assert.doesNotMatch(capped, /REPORT_TAIL/);
            assert.equal(reads.length, 1, 'Show full reads the task once');
            reads.shift()({ ok: true, json: async () => ({ ...detail, ...detailOnly }) });
            for (let tick = 0; tick < 30 && !rendered.includes('REPORT_TAIL'); tick++) {
                await new Promise((resolve) => setTimeout(resolve, 5));
            }
            const occurrences = (text, part) => text.split(part).length - 1;
            assert.match(rendered, /TRACE_TAIL/);
            assert.equal(occurrences(rendered, 'The retained answer has a complete paragraph.'), 120, 'the full result, undistorted');
            assert.equal(occurrences(rendered, 'Read the recorded evidence.'), 160, 'the full trace, undistorted');
            for (const sentence of stated) {
                assert.equal(occurrences(capped, sentence), 1, `the capped Details state: ${sentence}`);
                assert.equal(occurrences(rendered, sentence), 1, `Show full keeps it, once: ${sentence}`);
            }
            if (detailOnly.terminal_plan_review_open) {
                assert.equal(occurrences(capped, OPEN_PLAN), 0, 'the frame does not carry the detail-only flag');
                assert.equal(occurrences(rendered, OPEN_PLAN), 1);
                assert.equal(occurrences(capped, DEBT), 1);
                assert.equal(occurrences(rendered, DEBT), 1);
                rows[rows.length - 1] = { ...frame, delegated_runs_unreconciled: [] };
                await instance.refreshHistory({ revision: 2 });
                assert.equal(walkCard(globalThis.document.byId.get('chat-messages'), 'child-1'), card);
                assert.match(rendered, /data-expanded="1"/, 'the same disclosure stays open');
                assert.match(rendered, /REPORT_TAIL/);
                assert.match(rendered, /TRACE_TAIL/);
                assert.equal(occurrences(rendered, OPEN_PLAN), 1, 'a missing frame flag does not erase the detail fact');
                assert.equal(occurrences(rendered, stated[0]), 1);
                assert.equal(occurrences(rendered, DEBT), 0);
                const healed = rendered;
                toggle();
                toggle();
                assert.equal(rendered, healed);
                assert.equal(reads.length, 0, 'neither reconciliation nor reopening reads the full detail again');
            }
        } finally {
            instance.destroy();
            restoreDom(prior);
            Object.defineProperty(ElementStub.prototype, 'innerHTML', html);
        }
    });
}

// R6: the child's ending arrives in several frames. The cause line and the timeline fold are its two real
// writers; each is exercised as its own function, not through a stub DOM that cannot see the closed line.
test('an origin-less cancellation folds its explicit custody debt into its cause, and an emptied list clears it', () => {
    const debt = { subagent_event: 'cancelled', status: 'cancelled', reason_code: 'owner_requested_cancel',
        delegated_runs_unreconciled: ['run-1'] };
    const line = childLineCause(debt);
    assert.match(line.detail, /never reconciled/, 'the separate custody warning survives a cancellation recorded without an origin');
    assert.match(line.detail, /owner_requested_cancel/, 'beside its exact code');
    assert.doesNotMatch(line.detail, /Stopped (from|by|with)|Requested by/, 'and no actor is invented');
    assert.doesNotMatch(line.lead, /never reconciled/);
    assert.doesNotMatch(childLineCause({ ...debt, delegated_runs_unreconciled: [] }).detail, /never reconciled/,
        'an explicit empty list answers the debt without a stale warning');
    assert.doesNotMatch(childLineCause({ ...debt, reason_code: 'delegated_custody_unreconciled',
        delegated_runs_unreconciled: [] }).detail, /never reconciled/, 'the stamp whose known list is empty reads healed');
    assert.doesNotMatch(childLineCause({ subagent_event: 'cancelled', status: 'cancelled',
        reason_code: 'owner_requested_cancel' }).detail, /never reconciled/, 'a missing list proves nothing');
});

test('a frame of the outcome alone keeps the report and trace, retains an unstated debt, and an explicit [] clears it', () => {
    // R6: three real frames through the child summarizer and the live fold — the terminal notice with
    // report, trace and open debt, then a status-only outcome rewording the reason, then the healed list.
    const key = 'subagent-lifecycle:kid-a';
    const record = { items: [] };
    const frame = (extra) => summarizeChatLiveEvent({ type: 'send_message', is_progress: true,
        delegation_role: 'subagent', subagent_task_id: 'kid-a', parent_task_id: 'root', subagent_event: 'completed',
        status: 'completed', ...extra });
    const live = (summary, second) => updateLiveTimelineItem(record, summary, {
        ts: `12:00:0${second}`, rawTs: `2026-11-21T12:00:0${second}Z`, syntheticKey: key,
        headline: summary.headline, inPlaceByKey: true });
    live(frame({ reason_code: 'round_limit', delegated_runs_unreconciled: ['run-1'],
        result: 'Report head the answer Report tail', trace_summary: 'Trace head Trace tail' }), 1);
    assert.equal(record.items.length, 1);
    const item = record.items[0];
    assert.equal(item.fullRef, 'kid-a');
    assert.match(item.fullOutput, /Report head the answer Report tail/);
    assert.match(item.fullOutput, /Trace head Trace tail/);
    assert.equal(item.childCause._result_text_present, true);
    assert.deepEqual(item.childCause.delegated_runs_unreconciled, ['run-1']);
    assert.equal(item.childCause.reason_code, 'round_limit');
    assert.match(item.fullBody, /never reconciled/, 'the notice states the open debt');

    // The outcome alone (task_done, a late review's current outcome): no report, no debt stated, a new reason.
    const outcome = { reason_code: 'final_message',
        outcome_axes: { execution: { status: 'ok', reason_code: 'final_message' } } };
    assert.equal(live(frame(outcome), 2).timelineUpdate, 'patch-at');
    assert.equal(item.childCause.reason_code, 'final_message', 'the new authoritative reason replaces the old one');
    assert.deepEqual(item.childCause.delegated_runs_unreconciled, ['run-1'], 'an absent current debt keeps the known open warning');
    assert.equal(item.childCause._result_text_present, true, 'the answer\'s presence stays stated until a frame states it again');
    assert.match(item.fullOutput, /Report head the answer Report tail/, 'the report an outcome-only frame never carried stays');
    assert.match(item.fullOutput, /Trace head Trace tail/, 'and so does the trace');
    assert.equal(item.fullRef, 'kid-a', 'no further detail read is demanded');
    assert.match(item.fullBody, /never reconciled/, 'the reworded line keeps the warning beside the kept report');

    // The explicit empty list is a statement: it clears the debt with the same report and trace.
    assert.equal(live(frame({ ...outcome, delegated_runs_unreconciled: [] }), 3).timelineUpdate, 'patch-at');
    assert.deepEqual(item.childCause.delegated_runs_unreconciled, [], 'and the newest cause words the line');
    assert.match(item.fullOutput, /Report head the answer Report tail/);
    assert.match(item.fullOutput, /Trace head Trace tail/);
    assert.doesNotMatch(item.fullBody, /never reconciled/, 'an explicit empty list clears the warning');

    // An identical cause reuses the stored object.
    assert.equal(timelineReadFields({ fullRef: 'kid-a', truncated: false, childCause: item.childCause },
        { fullRef: 'kid-a', childCause: item.childCause }).childCause, item.childCause);
});

// R7 web-only: each cause keeps its own scope; explicit absence and [] stay distinct.
test('an origin-less cancel whose debt list is known empty still warns about missing result files', () => {
    const evt = { subagent_event: 'cancelled', status: 'cancelled', reason_code: 'owner_requested_cancel',
        delegated_runs_unreconciled: [], artifact_bundle: { status: 'missing' } };
    const line = childLineCause(evt);
    assert.ok(line.detail.includes(TASK_CAUSE_PHRASES.artifacts_unavailable), line.detail);
    assert.doesNotMatch(line.detail, /never reconciled/, 'the known empty list answers the custody stamp');
    assert.doesNotMatch(line.detail, /Stopped (from|by|with)|Requested by/, 'no actor is invented');
});

test('an author Stop still states the display-only first attempt beside its own rationale', () => {
    const fact = { status: 'incomplete', reason: 'initial_execution_display_only',
        initial_execution: { state: 'interrupted', run_id: 'run-9' } };
    const evt = { type: 'send_message', is_progress: true, delegation_role: 'subagent', subagent_task_id: 'kid-b',
        parent_task_id: 'root', subagent_event: 'completed', status: 'completed', reason_code: 'final_message',
        result: 'Answer.',
        outcome_axes: { execution: { status: 'ok', reason_code: 'final_message',
            task_completion: { action: 'stop', rationale: 'The export still fails.' }, actor_first_terminal: fact } } };
    const body = summarizeChatLiveEvent(evt).fullBody;
    assert.ok(body.includes(STOPPED), body);
    assert.ok(body.includes(TASK_CAUSE_PHRASES.configured_actor_first_interrupted.replace(/\.$/, '')), body);
    assert.ok(body.includes('The export still fails.'), body);
});

test('a typed terminal notice is recorded fact: a later status frame keeps the report and trace, never the old headline', () => {
    const notice = 'Done with warnings. Child (child kid-c of root). Some cause words.';
    const evt = { type: 'send_message', is_progress: true, delegation_role: 'subagent', subagent_task_id: 'kid-c',
        parent_task_id: 'root', subagent_event: 'completed', status: 'completed', reason_code: 'final_message',
        system_type: 'subagent_terminal_notice', text: notice, result: 'The answer.', trace_summary: 'The trace.',
        outcome_axes: { execution: { status: 'ok', reason_code: 'final_message' } } };
    const view = summarizeChatLiveEvent(evt);
    assert.doesNotMatch(view.fullBody, /Done with warnings/, 'the supervisor\'s generated headline is not a current warning');
    assert.match(view.fullOutput, /The answer\./, 'the report stays as it was');
    assert.match(view.fullOutput, /The trace\./, 'and so does the trace');
});

test('a canonical review PASS without a decision does not resurrect the legacy pending review_status', () => {
    const stale = { review_status: { status: 'pending', acceptance_decision: { status: 'pending', reason: 'plan_review_advisory' } } };
    const canonical = { status: 'completed', reason_code: 'final_message',
        outcome_axes: { execution: { status: 'ok', reason_code: 'final_message' }, review: { status: 'pass' } }, ...stale };
    assert.doesNotMatch(taskReasonDetail(canonical), /plan review/i);
    assert.match(taskReasonDetail({ status: 'completed', reason_code: 'final_message', ...stale }), /plan review/i,
        'a truly old record without the canonical axis keeps reading its legacy carrier');
});

// The live route, evaluated against the REAL frame builder (settled_card.js). Only the chat instance's own
// furniture is stubbed: the route's body, the frame it hands over and the attempt reconciliation it owns are the
// shipped ones, so a helper that drops a fact fails here rather than in a re-implementation of it.
function liveSubagentTerminalRoute({ records = new Map(), onFrame = () => true, onHydrate = () => {} } = {}) {
    const chatSource = readFileSync(new URL('../modules/chat.js', import.meta.url), 'utf8');
    // Slice by the function's own name: its phase default comes from the real selector, so the
    // harness injects `taskTerminalPhase` instead of pinning a signature that travels with it.
    const start = chatSource.indexOf('    function routeSubagentTerminalToCard(');
    const end = chatSource.indexOf('\n    function updateLiveCardFromLogEvent(evt) {', start);
    assert.ok(start > 0 && end > start, 'the live route moved — update this test');
    const parents = new Map([['kid-r', { parentId: 'root', role: 'scout' }], ['kid-s', { parentId: 'root', role: 'scout' }]]);
    return Function('subagentChildParents', 'liveCardRecords', 'terminalChildFrame', 'updateSubagentCardFromEvent',
        'noteHandoverAttempt', 'withStableViewport', 'hydrateOpenSettledChild', 'hydrateCardReviews', 'taskDoneIsTerminal', 'taskTerminalPhase',
        `${chatSource.slice(start, end)}\nreturn routeSubagentTerminalToCard;`)(
        parents, records, terminalChildFrame, onFrame, noteHandoverAttempt,
        () => {},  // the viewport writer: the stored read's own paint, not this witness's business
        onHydrate, () => {}, taskDoneIsTerminal, taskTerminalPhase);
}

test('the real routeSubagentTerminalToCard forwards cause facts and only the carried text', () => {
    let received = null;
    const route = liveSubagentTerminalRoute({ onFrame: (evt) => { received = evt; return true; } });
    assert.equal(route('kid-r', { type: 'task_done', task_id: 'kid-r', status: 'failed',
        reason_code: 'delegated_custody_unreconciled', reason_detail: 'the daemon stopped answering',
        artifact_status: 'missing', delegated_runs_unreconciled: ['run-9'],
        result: 'partial report', error: 'daemon unreachable', result_truncated: true,
        ts: '2026-11-21T10:00:00Z' }), true);
    assert.equal(received.subagent_event, 'failed');
    assert.equal(received.delegation_role, 'subagent');
    assert.equal(received.reason_code, 'delegated_custody_unreconciled');
    assert.deepEqual(received.delegated_runs_unreconciled, ['run-9']);
    // The cause facts the source stated and the card's own meta carry list does not: they travel only because the
    // frame builder spreads the shared cause facts, not because a key survived by accident.
    assert.equal(received.reason_detail, 'the daemon stopped answering');
    assert.equal(received.artifact_status, 'missing');
    assert.equal(received.result, 'partial report');
    assert.equal('trace_summary' in received, false, 'a text key the frame does not carry is not invented');
    assert.equal(received.error, 'daemon unreachable');
    assert.equal(received.result_truncated, true, 'the truncation flag travels beside the text it truncates');
    assert.equal(received.trace_summary_truncated, undefined, 'and no flag is invented for a text the frame does not carry');
    assert.equal(received._result_text_present, true, 'the source stated a report, and the frame says so');
});

test('the real route keeps the stored attempt and handover reconciliation of its own card', () => {
    const record = { finished: false };
    const route = liveSubagentTerminalRoute({ records: new Map([['kid-s', record]]) });
    const detail = { type: 'task_done', task_id: 'kid-s', status: 'completed', task_attempt: 2,
        authoring_handovers: [{ from_model: 'm-one', to_model: 'm-two', status: 'selected' }],
        result: 'the stored report', ts: '2026-11-21T10:00:00Z' };
    assert.equal(route('kid-s', detail, true), true);
    assert.equal(record.handoverAttempt, 2, 'the stored attempt is the card\'s own');
    assert.deepEqual(record.handoverRows, detail.authoring_handovers, 'the stored handover history is the card\'s own');
});
