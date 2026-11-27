import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

import {
    formatReviewProjection,
    modelExecutionView,
    summarizeChatLiveEvent,
    taskOutcomeSeverity,
} from '../modules/log_events.js';
import { apiRouteParts, displayApiModelTarget, compactModel, executorTitle, modelWords, recordedCoordinator, selectionLane } from '../modules/harness_presentation.js';
import { admissionRefusalRow, cardExecutorChip, executorChip, executorModelText, keepStickyExecutorChip, requestedExecutor, requestedFact } from '../modules/executor_card_facts.js';
import { terminalChildFrame } from '../modules/settled_card.js';
import { routeConnectionId } from '../modules/connection_labels.js';
import { composeModelSource, parseModelSource } from '../modules/route_editor_primitives.js';
import { applyPayload, currentPayload, setMissTransport } from '../modules/i18n.js';

test('shared task severity never paints degraded or best-effort outcomes green', () => {
    const clean = {
        outcome_axes: {
            lifecycle: { status: 'completed' }, execution: { status: 'ok' },
            objective: { status: 'pass' }, review: { status: 'pass' },
            artifacts: { status: 'ready' },
        },
    };
    assert.equal(taskOutcomeSeverity(clean), 'done');
    assert.equal(taskOutcomeSeverity({
        ...clean, outcome_axes: { ...clean.outcome_axes, objective: { status: 'best_effort' } },
    }), 'warn');
    assert.equal(taskOutcomeSeverity({
        ...clean, outcome_axes: { ...clean.outcome_axes, objective: { status: 'degraded' } },
    }), 'warn');
    assert.equal(taskOutcomeSeverity({
        ...clean, outcome_axes: { ...clean.outcome_axes, review: { status: 'degraded' } },
    }), 'warn');
    assert.equal(taskOutcomeSeverity({
        ...clean, outcome_axes: { ...clean.outcome_axes, review: { status: 'fail' } },
    }), 'error');
    assert.equal(taskOutcomeSeverity({
        ...clean,
        outcome_axes: {
            ...clean.outcome_axes,
            execution: { status: 'degraded', reason_code: 'review_skipped_deadline_reserve' },
            objective: { status: 'degraded', source: 'task_acceptance_deadline_reserve' },
            review: {
                status: 'skipped', eligibility: 'eligible', run_count: 0,
                // v6.78.0: canonical host status + typed reason (the reason literal
                // is what execution.reason_code / objective.source key off).
                acceptance_decision: {
                    status: 'finalized_unaccepted',
                    reason: 'review_skipped_deadline_reserve',
                },
            },
        },
    }), 'warn');
});

test('review projection exposes panel binding and every actor without raw output', () => {
    const text = formatReviewProjection({ panels: [{
        panel_id: 'panel_abc', surface: 'task_acceptance', authority: 'host_root',
        aggregate_signal: 'DEGRADED', transport_status: 'partial', parse_status: 'malformed',
        quorum: { required: 2, contributed: 1, configured: 3 },
        enforcement_impact: 'degrades_completion', reason: 'one actor malformed',
        candidate_hash: 'candidate', evidence_revision: 'evidence', fence_hash: 'fence-hash',
        actors: [
            {
                slot_id: 'slot_1', actor_role: 'task acceptance', provider: 'anthropic',
                model: 'anthropic/claude-fable-5', transport_status: 'success',
                parse_status: 'valid', semantic_verdict: 'DEGRADED',
                outcome_tier: 'best_effort',
                quorum_contribution: false, enforcement_impact: 'abstains', reason: 'uncertain',
            },
            {
                slot_id: 'slot_2', actor_role: 'task acceptance', provider: 'openai',
                model: 'openai/gpt-5.6-sol', transport_status: 'timeout',
                parse_status: 'malformed', semantic_verdict: '',
                quorum_contribution: false, enforcement_impact: 'abstains', reason: 'timeout',
            },
        ],
    }] });
    assert.match(text, /panel_abc/);
    assert.match(text, /candidate_hash=candidate/);
    assert.match(text, /fence_hash=fence-hash/);
    assert.match(text, /slot_1.*claude-fable-5.*verdict=DEGRADED/);
    assert.match(text, /slot_1.*outcome_tier=best_effort/);
    assert.match(text, /slot_2.*transport=timeout.*parse=malformed/);
    assert.doesNotMatch(text, /raw_text/);
});

test('mixed panel renders quorum participation separately from pass support and veto', () => {
    const text = formatReviewProjection({ panels: [{
        panel_id: 'panel_mixed', surface: 'task_acceptance', authority: 'host_root',
        aggregate_signal: 'FAIL', transport_status: 'success', parse_status: 'valid',
        quorum: { required: 2, contributed: 3, configured: 3 },
        enforcement_impact: 'requires_revision', reason: 'one valid reviewer vetoed',
        actors: [
            {
                slot_id: 'slot_1', actor_role: 'task acceptance', provider: 'anthropic',
                model: 'anthropic/claude-fable-5', transport_status: 'success',
                parse_status: 'valid', semantic_verdict: 'PASS',
                quorum_contribution: true, enforcement_impact: 'supports_pass', reason: 'ready',
            },
            {
                slot_id: 'slot_2', actor_role: 'task acceptance', provider: 'openai',
                model: 'openai/gpt-5.6-sol', transport_status: 'success',
                parse_status: 'valid', semantic_verdict: 'PASS',
                quorum_contribution: true, enforcement_impact: 'supports_pass', reason: 'ready',
            },
            {
                slot_id: 'slot_3', actor_role: 'task acceptance', provider: 'google',
                model: 'google/gemini-3.5-flash', transport_status: 'success',
                parse_status: 'valid', semantic_verdict: 'FAIL',
                quorum_contribution: true, enforcement_impact: 'veto', reason: 'gap',
            },
        ],
    }] });
    assert.match(text, /panel_mixed.*verdict=FAIL.*quorum=3\/3 \(required 2\)/);
    assert.match(text, /slot_1.*verdict=PASS.*quorum=contributes.*enforcement=supports_pass/);
    assert.match(text, /slot_3.*verdict=FAIL.*quorum=contributes.*enforcement=veto/);
});

test('subagent terminal projection leaves review rendering to the Reviews section', () => {
    const summary = summarizeChatLiveEvent({
        type: 'send_message',
        is_progress: true,
        delegation_role: 'subagent',
        parent_task_id: 'parent',
        subagent_task_id: 'child',
        subagent_role: 'reviewer',
        subagent_event: 'completed',
        result: 'Concrete child result',
        write_surface: 'none',
        status: 'completed',
        outcome_axes: {
            lifecycle: { status: 'completed' },
            execution: { status: 'ok' },
            objective: { status: 'best_effort' },
            review: { status: 'degraded' },
            artifacts: { status: 'ready' },
        },
        review_projection: { panels: [{
            panel_id: 'panel_child', surface: 'task_acceptance', authority: 'host_root',
            aggregate_signal: 'DEGRADED', transport_status: 'success', parse_status: 'valid',
            quorum: { required: 1, contributed: 1, configured: 1 },
            enforcement_impact: 'degrades_completion', reason: 'needs owner context',
            actors: [],
        }] },
    });
    assert.equal(summary.terminal, true);
    assert.equal(summary.phase, 'warn');
    assert.equal(summary.activityPreview, 'Concrete child result');
    assert.equal(summary.body, 'Concrete child result');
    assert.doesNotMatch(summary.fullBody, /panel_child|\[REVIEW\]/);
    assert.match(summary.fullBody, /^## Report to parent\nConcrete child result/);
    assert.equal(summary.activityLead, 'Report to parent');
    assert.deepEqual(summary.meta || [], []);
});

test('terminal subagent activity prefers the authoritative result over emitter boilerplate', () => {
    const summary = summarizeChatLiveEvent({
        type: 'send_message',
        is_progress: true,
        delegation_role: 'subagent',
        parent_task_id: 'parent',
        subagent_task_id: 'child',
        subagent_event: 'completed',
        content: 'Subagent child completed (researcher).',
        result: 'The evidence-backed conclusion and its concrete next step.',
        status: 'completed',
    });
    assert.equal(summary.activityPreview, 'The evidence-backed conclusion and its concrete next step.');
    assert.match(summary.fullBody, /Subagent child completed/);
    assert.match(summary.fullBody, /## Report to parent\nThe evidence-backed conclusion/);
});

test('a child warning retains its report label; failed and stopped results keep the neutral heading', () => {
    for (const [event, label] of [['completed_warn', 'Report to parent'], ['failed', ''], ['cancelled', '']]) {
        const summary = summarizeChatLiveEvent({ type: 'send_message', is_progress: true,
            delegation_role: 'subagent', parent_task_id: 'parent', subagent_task_id: 'child',
            subagent_event: event, result: '⚠️ Checked the implementation; one limitation remains.' });
        assert.equal(summary.activityLead || '', label, event);
        assert.match(summary.fullBody, label ? /^## Report to parent\n/ : /^\[RESULT\]\n/, event);
    }
    const failed = summarizeChatLiveEvent({ type: 'send_message', is_progress: true,
        delegation_role: 'subagent', parent_task_id: 'parent', subagent_task_id: 'child',
        subagent_event: 'completed', result: 'Retained output.',
        outcome_axes: { execution: { status: 'infra_failed' } } });
    assert.equal(failed.activityLead || '', '');
    assert.match(failed.fullBody, /^\[RESULT\]\n/);
});

test('review-only subagent projection has no activity and no auto-disclosure flag', () => {
    const summary = summarizeChatLiveEvent({
        type: 'send_message',
        is_progress: true,
        delegation_role: 'subagent',
        parent_task_id: 'parent',
        subagent_task_id: 'child',
        subagent_event: 'completed',
        review_projection: { panels: [{
            panel_id: 'panel_review_only', surface: 'task_acceptance', authority: 'host_root',
            aggregate_signal: 'PASS', transport_status: 'success', parse_status: 'valid',
            quorum: { required: 1, contributed: 1, configured: 1 },
            enforcement_impact: 'supports_pass', reason: 'review only', actors: [],
        }] },
    });
    assert.equal(summary.activityPreview, '');
    assert.equal(summary.body, '');
    assert.equal('expandByDefault' in summary, false);
});

// ---------------------------------------------------------------------------
// Phase 6, owner directive #1: the per-bubble / per-subagent executor chip.
// «бейдж точно нужен, но не рекламный … что ТУТ бабл \ субагент на codex»
// ---------------------------------------------------------------------------

test('a delegated frame yields a small harness chip; an ordinary frame yields none', () => {
    const chip = executorChip({ executor_route: 'codex' });
    assert.equal(chip.harness, 'codex');
    // Quiet label (owner choice #360 A): an assigned route is the bare name of
    // the selected helper — the card's own phase chip says Working, and the
    // name never claims the session computes now. The dispatch fact stays in
    // the detail sentence, which the expanded card prints as text (a
    // hover-only title is invisible on touch, to AT and in copies). A live
    // frame carries no evidence either way, so no evidence-grade negative.
    // Identity (branded name, mark geometry) comes from harness_presentation.
    assert.equal(chip.label, 'Codex');
    assert.equal(chip.hasEvidence, false);
    assert.match(chip.title, /Dispatched to Codex/);
    for (const status of ['completed', 'failed', 'cancelled', 'rejected_duplicate']) {
        const terminal = executorChip({ executor_route: 'codex', status });
        assert.equal(terminal.label, 'Codex · run outcome unknown');
        assert.equal(terminal.hasEvidence, false);
        assert.match(terminal.title, /terminal record has no delegated-run evidence/);
        assert.doesNotMatch(terminal.title, /arrives with the terminal receipt/);
    }
    assert.equal(executorChip({ executor_route: 'codex', subagent_event: 'completed' }).label,
        'Codex · run outcome unknown');
    assert.equal(executorChip({ executor_route: 'codex', task_terminal_status: 'failed', status: 'running' }).label,
        'Codex · run outcome unknown');
    assert.equal(executorChip({ executor_route: 'codex', status: 'interrupted' }).label, 'Codex');
    assert.equal('icon' in chip, false, 'mark geometry belongs to harness_presentation');
    assert.match(chip.title, /Codex/);
    // WHERE it ran is all the chip can see. It has no access to the run's spend, so
    // it must not describe the billing: "on your codex subscription" told the owner
    // the work was covered, which the ledger alone can say (and only when the harness
    // says it — a zero there may mean unknown pricing, never "free").
    assert.ok(!/subscription/i.test(chip.title), chip.title);
    // The opaque `harness=model` spelling prints the HARNESS part only.
    assert.equal(executorChip({ executor_route: 'codex=gpt-5.6-sol' }).label, 'Codex');
    // An unknown harness still gets a chip (a generic mark), never a crash.
    assert.equal(executorChip({ executor_route: 'opencode' }).label, 'OpenCode');
    // ABSENT FACT -> no chip at all: no placeholder, no "api" noise on every
    // ordinary bubble (the native path is the unremarkable case).
    assert.equal(executorChip({}), null);
    assert.equal(executorChip({ executor_route: '' }), null);
    assert.equal(executorChip(null), null);
});

test('a harness pin the resolution refused reads `{harness} · blocked`, never dispatched/no run yet (#363)', () => {
    // The typed $0 terminal (reason_code subagent_executor_unavailable) plus
    // the route the pin named: the chip says WHO refused and that the child
    // never ran. Checked before the evidence branches, so an empty custody
    // read cannot print "no run yet" over a child that was never dispatched.
    const blocked = executorChip({ executor_route: 'codex', reason_code: 'subagent_executor_unavailable' });
    assert.equal(blocked.label, 'Codex · blocked');
    assert.equal(blocked.harness, 'codex');
    assert.equal(blocked.hasEvidence, true, 'evidence-grade: a later dispatch-shaped frame cannot downgrade it');
    assert.match(blocked.title, /NOT run/);
    assert.doesNotMatch(blocked.title, /subscription/i);
    // A configured-session refusal after dispatch reaches the seam WITH an
    // (empty) evidence block: still blocked, not "no run yet".
    const sessionRefused = executorChip({
        executor_route: 'claude', reason_code: 'subagent_executor_unavailable',
        execution_evidence: { delegated_runs_started: 0, delegated_runs_settled: 0 },
        actual_substrate: 'native_only',
    });
    assert.equal(sessionRefused.label, 'Claude Code · blocked');
    assert.equal(keepStickyExecutorChip(blocked, executorChip({ executor_route: 'codex' })), true);
    // A blocked API-model actor names no route: no harness chip is invented.
    assert.equal(executorChip({ reason_code: 'subagent_executor_unavailable' }), null);
    // A helper refused for hidden or uncheckable delegate tools never started either: no "run outcome unknown".
    for (const code of ['delegate_tools_invisible', 'delegate_visibility_unverified']) {
        const refused = executorChip({ executor_route: 'claude', reason_code: code });
        assert.equal(refused.label, 'Claude Code · blocked', code);
        assert.match(refused.title, /could not start — the task was NOT run/, code);
    }
    // Any other reason code keeps the ordinary layered truth.
    assert.equal(executorChip({ executor_route: 'codex', reason_code: 'tool_failure' }).label, 'Codex');
});

test('the chip is layered truth: decision before evidence, receipt only from evidence', () => {
    // PRE-COMPLETION: the route is a DISPATCH decision, so the chip states only
    // that — never "Ran on your ... account", a receipt nothing has issued yet.
    // The claude harness prints its product name (owner decision: "Claude Code").
    const dispatched = executorChip({ executor_route: 'claude' });
    assert.equal(dispatched.label, 'Claude Code');
    assert.match(dispatched.title, /Dispatched to Claude Code/);
    assert.match(dispatched.title, /run evidence arrives with the terminal receipt/);
    // The coordinator is no part of the executor's sentence: the card's Coordinator row reads it from the chip.
    assert.deepEqual(dispatched.coordinator, { route: '', billing: 'unknown' });
    assert.doesNotMatch(dispatched.title, /runs on|bills as/);
    assert.doesNotMatch(dispatched.title, /Ran on your/);

    const dispatchedSub = executorChip({
        executor_route: 'claude',
        model: 'claudexor::claude=sonnet-4-6',
        coordinator_billing: 'subscription',
        parent_cognitive_route: { model: 'claudexor::claude=sonnet-4-6', chosen_by: 'inherited' },
    });
    assert.deepEqual(dispatchedSub.coordinator, { route: 'claude=sonnet-4-6', billing: 'subscription' });
    assert.equal(dispatchedSub.coordinatorProvenance, 'inherited');

    const dispatchedMeasured = executorChip({
        executor_route: 'codex',
        model: 'openai/gpt-5.6-sol',
        cost_usd: 0.02,
        coordinator_billing: 'measured',
    });
    assert.deepEqual(dispatchedMeasured.coordinator, { route: 'gpt-5.6-sol', billing: 'measured' });

    // EVIDENCE with settled runs: the completion-seam reconciliation proved the
    // delegation happened, so the chip may finally say so — with the count and
    // the ledger-derived subscription spend in the tooltip detail layer.
    const delegated = executorChip({
        executor_route: 'claude',
        execution_evidence: {
            delegated_runs_started: 1, delegated_runs_settled: 1,
            delegated_runs_succeeded: 1, delegated_runs_failed: 0,
            subscription_cost_usd: 0.0, harness_models: ['claude-sonnet'],
        },
    });
    // Every settled run ok is the quiet case: the bare name, still an
    // evidence-grade receipt for sticky precedence; the owner dictionary
    // (plan D9) keeps its counts in the detail.
    assert.equal(delegated.label, 'Claude Code');
    assert.equal(delegated.hasEvidence, true);
    assert.match(delegated.title, /Delegated to your Claude Code account — 1 run settled \(1 ok\), \$0\.00 subscription/);

    // ALL-FAILED runs must never read as success: the failure rides the LABEL,
    // not only the detail (delegated_runs_failed was previously unread and two
    // failed runs rendered as a clean "2 run(s)" receipt).
    const allFailed = executorChip({
        executor_route: 'codex',
        execution_evidence: {
            delegated_runs_started: 2, delegated_runs_settled: 2,
            delegated_runs_succeeded: 0, delegated_runs_failed: 2,
            subscription_cost_usd: null, harness_models: [],
        },
    });
    assert.equal(allFailed.label, 'Codex · 2 of 2 runs failed');
    assert.match(allFailed.title, /2 failed/);

    // STARTED but none settled: as far as the durable rows know, the run is
    // still out — the label says so instead of a bare harness name.
    const inFlight = executorChip({
        executor_route: 'codex',
        execution_evidence: {
            delegated_runs_started: 1, delegated_runs_settled: 0,
            delegated_runs_succeeded: 0, delegated_runs_failed: 0,
            subscription_cost_usd: null, harness_models: [],
        },
    });
    // Evidence is terminal-frame material: started-but-unsettled means the
    // run(s) never settled — a present-tense "running" on a finished card
    // would be a lie (adversarial wave B-ADV-4).
    assert.equal(inFlight.label, 'Codex · not settled');
    assert.match(inFlight.title, /1 run\(s\) started, none settled/);

    // ACCESS + UNVERIFIED (harness-health O13): the requested write surface and
    // the engine-disclosed applied access ride the detail; runs whose
    // work-order coverage never verified are discounted from ok AND named in
    // the label — never as a physical failure the counts do not prove.
    const withAccess = executorChip({
        executor_route: 'codex',
        write_surface: 'external_workspace',
        execution_evidence: {
            delegated_runs_started: 2, delegated_runs_settled: 2,
            delegated_runs_succeeded: 1, delegated_runs_failed: 1,
            delegated_runs_source_unresolved: 1,
            subscription_cost_usd: null, harness_models: [],
            applied_access_profiles: ['workspace_write'],
        },
    });
    assert.equal(withAccess.label, 'Codex · 1 of 2 runs unverified');
    assert.match(withAccess.title, /unverified work-order coverage/);
    assert.match(withAccess.title, /write surface external_workspace/);
    assert.match(withAccess.title, /access applied: workspace_write/);
    // Frames without the new facts render exactly as before (additive-only).
    assert.ok(!/unverified/.test(allFailed.label), allFailed.label);
    assert.ok(!/access applied/.test(allFailed.title), allFailed.title);

    // EVIDENCE with zero runs: the route was assigned and no delegated run left
    // a durable record — the chip points at the ledger instead of asserting
    // "ran natively" as a fact (a run started past a failed durable write is
    // still possible: started_uncustodied).
    const unused = executorChip({
        executor_route: 'claude',
        execution_evidence: {
            delegated_runs_started: 0, delegated_runs_settled: 0,
            subscription_cost_usd: null, harness_models: [],
        },
    });
    assert.equal(unused.label, 'Claude Code · no run yet');
    assert.match(unused.title, /no durable record of a delegated run/);

    // Unreadable custody evidence is UNKNOWN, never a "no run" receipt: the
    // zero counts ride evidence_read_failed and must not render as "recorded".
    const unreadable = executorChip({
        executor_route: 'claude',
        execution_evidence: {
            delegated_runs_started: 0, delegated_runs_settled: 0,
            evidence_read_failed: true,
            subscription_cost_usd: null, harness_models: [],
        },
    });
    assert.match(unreadable.label, /evidence unavailable/);
    assert.match(unreadable.title, /could not be read/);
    assert.match(unreadable.title, /unknown/);
    assert.doesNotMatch(unreadable.label, /no run yet/);
    // Zero custody rows prove neither non-execution nor the API path
    // (started_uncustodied exists) — the title must assert NEITHER.
    assert.doesNotMatch(unused.title, /natively|ran on the API/);

    // evidence_read_failed is honoured PAST recorded starts too: the partial
    // work-order replay can fail with started>0, and a confident settled/spend
    // receipt over admittedly incomplete evidence would be a lie.
    const unreadablePastStart = executorChip({
        executor_route: 'codex',
        execution_evidence: {
            delegated_runs_started: 2, delegated_runs_settled: 2,
            delegated_runs_succeeded: 2, delegated_runs_failed: 0,
            evidence_read_failed: true,
            subscription_cost_usd: 1.23, harness_models: [],
        },
    });
    assert.match(unreadablePastStart.label, /evidence unavailable/);
    assert.match(unreadablePastStart.title, /at least 2 delegated run\(s\) started/);
    assert.match(unreadablePastStart.title, /final counts are unknown/);
    assert.doesNotMatch(unreadablePastStart.title, /\$/);

    // Undisclosed spend never renders as a number.
    const undisclosed = executorChip({
        executor_route: 'codex',
        execution_evidence: {
            delegated_runs_started: 2, delegated_runs_settled: 2,
            subscription_cost_usd: null, harness_models: [],
        },
    });
    assert.match(undisclosed.title, /spend undisclosed/);
    assert.doesNotMatch(undisclosed.title, /\$/);

    // An estimated sum never renders as an exact receipt: the ~ prefix rides.
    const estimated = executorChip({
        executor_route: 'codex',
        execution_evidence: {
            delegated_runs_started: 1, delegated_runs_settled: 1,
            subscription_cost_usd: 0.42, subscription_cost_estimated: true,
            harness_models: [],
        },
    });
    assert.match(estimated.title, /~\$0\.42 subscription/);
});

test('the terminal substrate FACT rides the tooltip beside the counts', () => {
    // `actual_substrate` is the completion seam's typed claim (Q1A) and was a
    // dead wire field: on the frame, in the history allowlist, rendered nowhere.
    const nativeOnly = executorChip({
        executor_route: 'codex',
        actual_substrate: 'native_only',
        execution_evidence: {
            delegated_runs_started: 0, delegated_runs_settled: 0,
            subscription_cost_usd: null, harness_models: [],
        },
    });
    assert.equal(nativeOnly.label, 'Codex · no run yet');
    assert.match(nativeOnly.title, /no harness run recorded/);

    const used = executorChip({
        executor_route: 'codex',
        actual_substrate: 'harness_used',
        execution_evidence: {
            delegated_runs_started: 1, delegated_runs_settled: 1,
            delegated_runs_succeeded: 1, delegated_runs_failed: 0,
            subscription_cost_usd: null, harness_models: [],
        },
    });
    assert.match(used.title, /custody evidence confirms a harness run/);

    const attempted = executorChip({
        executor_route: 'codex',
        actual_substrate: 'harness_attempted',
        execution_evidence: {
            delegated_runs_started: 1, delegated_runs_settled: 1,
            delegated_runs_succeeded: 0, delegated_runs_failed: 1,
            subscription_cost_usd: null, harness_models: [],
        },
    });
    assert.match(attempted.title, /harness attempted; no successful run confirmed/);

    // An unknown/absent enum adds no clause — and never crashes the chip.
    const bare = executorChip({ executor_route: 'codex', actual_substrate: 'something_new' });
    assert.doesNotMatch(bare.title, /—\s*$/);
});

test('the chip rides both projections: an ordinary progress bubble and a subagent row', () => {
    const bubble = summarizeChatLiveEvent({
        is_progress: true, content: 'working on the thing', task_id: 't1',
        executor_route: 'codex',
    });
    assert.equal(bubble.executorChip.harness, 'codex');

    const subagentRow = summarizeChatLiveEvent({
        is_progress: true, content: 'child progress',
        delegation_role: 'subagent', subagent_task_id: 'child-1',
        subagent_event: 'running', subagent_role: 'implementer',
        executor_route: 'claude=sonnet-5',
    });
    assert.equal(subagentRow.executorChip.harness, 'claude');

    // Neither projection invents a chip when the fact is absent.
    assert.equal('executorChip' in summarizeChatLiveEvent({
        is_progress: true, content: 'plain', task_id: 't2',
    }), false);
    assert.equal('executorChip' in summarizeChatLiveEvent({
        is_progress: true, content: 'plain child', delegation_role: 'subagent',
        subagent_task_id: 'child-2', subagent_event: 'running',
    }), false);
});

test('the selected executor and model read as a request from Queued on; evidence outranks it', () => {
    const selection = { requested_executor_route: { kind: 'agent_session', target_id: 'codex=gpt-5.5' } };
    const line = (frame) => executorModelText(executorChip(frame), requestedExecutor(frame));
    // Queued: no dispatch route yet, so the chip names the SELECTED harness and the line its model, as a request. Nothing
    // was sent, so nothing promises a receipt: the work may still end before dispatch.
    const queued = { ...selection, subagent_event: 'scheduled', accepted: true };
    assert.equal(executorChip(queued).label, 'Codex');
    assert.equal(executorChip(queued).title, 'Selected Codex, model gpt-5.5 (requested)');
    assert.equal(line(queued), 'requested gpt-5.5');
    // Dispatched: the route's bare harness (a blocked or terminal route) keeps the selection's model.
    const running = { ...selection, executor_route: 'codex', subagent_event: 'running' };
    assert.match(executorChip(running).title, /^Dispatched to Codex, model gpt-5\.5 \(requested\)/);
    assert.equal(line(running), 'requested gpt-5.5');
    // An older frame: a full dispatch route still names its model (an `:effort` tail is no part of it); a bare one none.
    assert.deepEqual(requestedExecutor({ executor_route: 'codex=gpt-5.5:high' }), { kind: 'agent_session', harness: 'codex', model: 'gpt-5.5' });
    assert.equal(requestedExecutor({ executor_route: 'codex' }), null);
    assert.equal(line({ executor_route: 'codex' }), '');
    // A progress report without a model keeps its own harness's request, never another harness's.
    const report = (harness_id, extra = {}) => ({ ...selection, executor_route: 'codex', subagent_task_id: 'c', executor_observation: {
        task_id: 'c', task_attempt: '1', run_id: 'r', attempt_id: 'a', harness_id, phase: 'p', revision: 1, ...extra } });
    assert.match(executorChip(report('codex')).title, /Codex, model gpt-5\.5 \(requested\), run r/);
    assert.equal(line(report('codex')), 'requested gpt-5.5');
    assert.match(executorChip(report('cursor')).title, /Cursor, model unconfirmed/);
    assert.equal(line(report('cursor')), '');
    assert.equal(requestedFact(requestedExecutor(report('cursor')), executorChip(report('cursor')), ''), 'Codex · gpt-5.5');
    // Observed evidence takes the line; the request moves to Details only when it differs.
    const observed = report('codex', { model: 'gpt-5.5-2026-11', model_source: 'observed' });
    assert.equal(line(observed), 'gpt-5.5-2026-11');
    assert.equal(requestedFact(requestedExecutor(observed), executorChip(observed), line(observed)), 'Codex · gpt-5.5');
    assert.equal(requestedFact(requestedExecutor(queued), executorChip(queued), line(queued)), '');
    const settled = { ...selection, executor_route: 'codex', status: 'completed', execution_evidence: {
        delegated_runs_started: 1, delegated_runs_settled: 1, delegated_runs_failed: 0, harness_models: ['GPT-5.5 Codex'] } };
    assert.equal(line(settled), 'GPT-5.5 Codex');
    const vendorSelection = {kind: 'agent_session', harness: 'codex', model: 'vendor-a/sol'};
    assert.equal(requestedFact(vendorSelection, {harness: 'codex', settledRunModel: 'vendor-a/sol'}, 'vendor-a/sol'), '');
    assert.equal(requestedFact(vendorSelection, {harness: 'codex', settledRunModel: 'vendor-b/sol'}, 'vendor-b/sol'), 'Codex · vendor-a/sol');
    // A child that ended before any dispatch (cancelled while queued, refused at admission, with or without a
    // status) lost no run: its selection and request stay, with no missing-receipt warning and no promised receipt.
    for (const ended of [{ status: 'cancelled', subagent_event: 'cancelled' },
        { status: 'failed', subagent_event: 'rejected', accepted: false }, { subagent_event: 'rejected', accepted: false }]) {
        const frame = { ...selection, ...ended };
        assert.equal(executorChip(frame).label, 'Codex', JSON.stringify(ended));
        assert.equal(executorChip(frame).title, 'Selected Codex, model gpt-5.5 (requested); no dispatch was recorded before it ended');
        assert.equal(line(frame), 'requested gpt-5.5', JSON.stringify(ended));
    }
    // Only a dispatched route can have lost its receipt: there the warning stays.
    const dispatchedEnd = executorChip({ ...selection, executor_route: 'codex', status: 'cancelled' });
    assert.equal(dispatchedEnd.label, 'Codex · run outcome unknown');
    assert.match(dispatchedEnd.title, /^The Codex route was assigned, model gpt-5\.5 \(requested\), but this terminal record has no delegated-run evidence/);
    // An API-model selection runs on no harness: its target is never a chip, and the line requests its model.
    const api = { requested_executor_route: { kind: 'api_model', target_id: 'openai::gpt-5.5' }, executor_route: 'openai::gpt-5.5' };
    assert.equal(executorChip(api), null);
    assert.equal(executorModelText(null, requestedExecutor(api)), 'requested gpt-5.5');
    assert.equal(executorModelText(executorChip(running), requestedExecutor(api)), '', 'an API model never joins a harness');
    // Nor does an older frame's API target: no harness id heads it.
    for (const executor_route of ['openai::gpt-5.5', 'claudexor::codex=gpt-5', 'anthropic/claude-x']) {
        assert.equal(executorChip({ executor_route }), null, executor_route);
        assert.equal(requestedExecutor({ executor_route }), null, executor_route);
    }
    // Every frame that states a selection hands it to the card.
    assert.deepEqual(summarizeChatLiveEvent({ ...queued, is_progress: true, content: 'Scheduled', delegation_role: 'subagent',
        subagent_task_id: 'c', parent_task_id: 'p' }).requestedExecutor, { kind: 'agent_session', harness: 'codex', model: 'gpt-5.5' });
});

test('a local API selection is marked local once, queued and completed', () => {
    // The row spells a local target `… (local)` (configured_subagents / subagent_runtime): compacting it never adds a
    // second mark, while a bare local-named model keeps its single mark and a remote one gets none.
    assert.deepEqual(['qwen.gguf (local)', 'openai::gpt-5 (local)', 'local-model', 'local-model (local)', 'openai::gpt-5'].map(compactModel),
        ['qwen.gguf (local)', 'gpt-5 (local)', 'local-model (local)', 'local-model (local)', 'gpt-5']);
    for (const [target, asked, ran] of [['qwen.gguf (local)', 'qwen.gguf (local)', 'qwen.gguf'], ['openai::gpt-5 (local)', 'gpt-5 (local)', 'openai::gpt-5']]) {
        const selected = requestedExecutor({ requested_executor_route: { kind: 'api_model', target_id: target } });
        // Queued: the API child's line requests it, marked once; Details repeat nothing the line says.
        assert.equal(executorModelText(null, selected), `requested ${asked}`, target);
        assert.equal(requestedFact(selected, null, `requested ${asked}`), '', target);
        // Completed on that local model: its solve response names the bare model (the lane is `used_local`), which is
        // the same request, so Details add no false `Requested` row; a different model still shows the request.
        assert.equal(requestedFact(selected, null, compactModel(ran)), '', target);
        assert.equal(requestedFact(selected, null, 'gpt-4.1'), asked, target);
    }
});

test('local or remote follows the recorded lane, never a word in a provider, connection or model name', () => {
    // A connection named `local-team` is no lane; a recorded lane alone decides, even over a model named `local-…`.
    assert.equal(compactModel('openai@local-team::remote-a'), 'remote-a');
    assert.equal(compactModel('local-model', false), 'local-model');
    assert.equal(compactModel('qwen.gguf', true), 'qwen.gguf (local)');
    assert.equal(compactModel('qwen.gguf (local)', true), 'qwen.gguf (local)', 'marked once');
    const solve = (used_model, used_local, extra = {}) => ({ source: 'usable_solve_response', requested_model: used_model,
        requested_use_local: used_local, used_model, used_local, provider: used_local ? 'local' : 'openai', llm_call_id: 'c', ...extra });
    assert.equal(modelExecutionView(solve('openai@local-team::remote-a', false)).value, 'remote-a · openai');
    assert.equal(modelExecutionView(solve('local-model', false)).value, 'local-model · openai');
    assert.equal(modelExecutionView(solve('qwen.gguf', true)).value, 'qwen.gguf (local) · local');
    assert.equal(modelExecutionView(solve('qwen.gguf', false, { requested_use_local: true, provider: 'openrouter' })).value,
        'qwen.gguf (initial request: qwen.gguf (local)) · openrouter', 'an explicit lane switch stays visible');
    // The selection behind that remote solve is the same model on the same lane: Details add no `Requested` row.
    const selected = requestedExecutor({ requested_executor_route: { kind: 'api_model', target_id: 'openai@local-team::remote-a' } });
    const fact = solve('openai@local-team::remote-a', false);
    assert.equal(executorModelText(null, selected), 'requested remote-a');
    assert.equal(requestedFact(selected, null, modelExecutionView(fact).model, fact), '');
});

test('one provider and model spelled two ways is one identity; another provider, vendor or lane keeps the request', () => {
    const fact = (used_model, used_local = false) => ({ source: 'usable_solve_response', requested_model: used_model, requested_use_local: used_local,
        used_model, used_local, provider: 'openrouter', llm_call_id: 'c' });
    const details = (target, used, lane) => requestedFact(requestedExecutor({ requested_executor_route: { kind: 'api_model', target_id: target } }),
        null, modelExecutionView(fact(used, lane)).model, fact(used, lane));
    // A bare id is OpenRouter's (provider_models.parse_model_route), as is the built-in connection's.
    for (const [target, used] of [['openrouter::anthropic/claude', 'anthropic/claude'], ['anthropic/claude', 'openrouter::anthropic/claude'],
        ['openrouter@default::anthropic/claude', 'anthropic/claude']]) assert.equal(details(target, used), '', `${target} ran ${used}`);
    assert.equal(details('openai::gpt-5', 'openai/gpt-5'), 'openai::gpt-5', 'OpenAI\'s own API is not OpenRouter');
    assert.equal(details('vendor-a/sol', 'vendor-b/sol'), 'vendor-a/sol');
    assert.equal(details('anthropic/claude (local)', 'anthropic/claude'), 'claude (local)', 'local asked, remote ran');
    assert.equal(details('anthropic/claude', 'anthropic/claude', true), 'claude', 'remote asked, local ran');
});

test('an API child\'s own model names it: a harness it calls is its tool, never its chip', () => {
    const api = { kind: 'api_model', target_id: 'openai::gpt-5.5-pro' };
    const nested = { requested_executor_route: api, executor_route: 'openai::gpt-5.5-pro', subagent_task_id: 'api', executor_observation: {
        task_id: 'api', task_attempt: '0', run_id: 'nested', attempt_id: 'n01', harness_id: 'codex', phase: 'harness.event', revision: 2,
        model: 'gpt-5.5-codex', model_source: 'requested' } };
    assert.equal(executorChip(nested), null);
    // A frame that does not state the selection still reports a harness; the card that knows its selection keeps none.
    const sparse = executorChip({ ...nested, requested_executor_route: undefined, executor_route: undefined });
    assert.equal(sparse.label, 'Codex · last update');
    assert.equal(cardExecutorChip({ executorChip: sparse, requestedExecutor: requestedExecutor({ requested_executor_route: api }) }, sparse), null);
    assert.equal(cardExecutorChip({}, sparse), sparse, 'without a selection the report stays the executor fact');
});

test('a terminal record ends a progress pin: the line names the scheduled request, the report stays as history', () => {
    const selection = { requested_executor_route: { kind: 'agent_session', target_id: 'codex=gpt-5.5' }, subagent_task_id: 'c' };
    const selected = requestedExecutor(selection);
    const report = { ...selection, executor_route: 'codex=gpt-5.5', ts: '2026-11-22T10:00:02Z', executor_observation: { task_id: 'c',
        task_attempt: '0', run_id: 'r', attempt_id: 'a01', harness_id: 'codex', phase: 'harness.event', revision: 3,
        model: 'gpt-5.5-2026-11', model_source: 'observed' } };
    const progress = executorChip(report);
    assert.equal(progress.label, 'Codex · last update');
    assert.equal(executorModelText(progress, selected), 'gpt-5.5-2026-11', 'in flight, the run\'s own report names its model');
    // A terminal frame carrying the run's last report speaks for the run's end, not its progress.
    assert.equal(executorChip({ ...report, status: 'cancelled', subagent_event: 'cancelled' }).label, 'Codex · run outcome unknown');
    // Stop without settlement: the terminal record outranks the pin and names the request; the report is history.
    const stopped = executorChip({ ...selection, executor_route: 'codex=gpt-5.5', status: 'cancelled', ts: '2026-11-22T10:05:00Z' });
    assert.equal(keepStickyExecutorChip(progress, stopped), false);
    const card = cardExecutorChip({ executorChip: progress, requestedExecutor: selected }, stopped);
    assert.equal(card.label, 'Codex · run outcome unknown');
    assert.equal(executorModelText(card, selected), 'requested gpt-5.5');
    assert.equal(card.title, 'The Codex route was assigned, model gpt-5.5 (requested), but this terminal record has no delegated-run '
        + 'evidence; the run outcome is unknown. ' + progress.title);
    // No later progress or dispatch frame reopens it; a newer terminal record replaces it and keeps that history.
    assert.equal(keepStickyExecutorChip(card, executorChip({ ...report, ts: '' })), true);
    assert.equal(keepStickyExecutorChip(card, executorChip({ ...selection, executor_route: 'codex' })), true);
    const unrouted = executorChip({ ...selection, status: 'cancelled', subagent_event: 'cancelled' });
    assert.equal(keepStickyExecutorChip(card, unrouted), false);
    // That record found no dispatch, but this card saw a start: no false "no dispatch".
    const again = cardExecutorChip({ executorChip: card, requestedExecutor: selected }, unrouted);
    assert.equal(again.label, 'Codex · run outcome unknown');
    assert.ok(again.title.endsWith(progress.title), again.title);
    assert.equal(cardExecutorChip({ executorChip: progress, requestedExecutor: selected }, unrouted).label, 'Codex · run outcome unknown');
    // A card that saw no start keeps the cancelled-before-dispatch wording; a receipt is never displaced by a pin.
    assert.equal(cardExecutorChip({ executorChip: executorChip({ ...selection, subagent_event: 'scheduled' }), requestedExecutor: selected }, unrouted).title,
        'Selected Codex, model gpt-5.5 (requested); no dispatch was recorded before it ended');
    const receipt = executorChip({ ...selection, executor_route: 'codex', status: 'cancelled', execution_evidence: {
        delegated_runs_started: 1, delegated_runs_settled: 0 } });
    assert.equal(keepStickyExecutorChip(receipt, stopped), true);
    assert.equal(keepStickyExecutorChip(card, receipt), false);
});

test('a receipt ends a typed update: its line stays the receipt\'s, its Details keep the run\'s last report', () => {
    const selection = { requested_executor_route: { kind: 'agent_session', target_id: 'codex=gpt-5.5' }, subagent_task_id: 'c' };
    const selected = requestedExecutor(selection);
    const observation = { task_id: 'c', task_attempt: '0', run_id: 'r', attempt_id: 'a01', harness_id: 'codex', phase: 'harness.event',
        revision: 3, model: 'gpt-5.5-2026-11', model_source: 'observed' };
    const progress = executorChip({ ...selection, executor_route: 'codex=gpt-5.5', ts: '2026-11-22T10:00:02Z', executor_observation: observation });
    // Stop with one started and no settled run: the receipt is terminal and outranks the pin.
    const stop = { ...selection, executor_route: 'codex=gpt-5.5', status: 'cancelled', ts: '2026-11-22T10:05:00Z',
        execution_evidence: { delegated_runs_started: 1, delegated_runs_settled: 0 } };
    const receipt = executorChip(stop);
    assert.equal(receipt.ended, true);
    assert.equal(keepStickyExecutorChip(progress, receipt), false);
    const card = cardExecutorChip({ executorChip: progress, requestedExecutor: selected }, receipt);
    assert.equal(card.label, 'Codex · not settled', 'the receipt controls the compact line');
    assert.equal(executorModelText(card, selected, true), 'requested gpt-5.5');
    assert.equal(card.title, `Delegated to your Codex account — 1 run(s) started, none settled. ${progress.title}`);
    assert.match(card.lastUpdate, /run r, attempt a01, revision 3\./);
    // A settled receipt keeps the same history; its single successful run still owns the line.
    const settled = executorChip({ ...stop, status: 'completed', execution_evidence: { delegated_runs_started: 1,
        delegated_runs_settled: 1, delegated_runs_failed: 0, harness_models: ['gpt-5.5-codex'] } });
    const done = cardExecutorChip({ executorChip: progress, requestedExecutor: selected }, settled);
    assert.equal(executorModelText(done, selected, true), 'gpt-5.5-codex');
    assert.ok(done.title.endsWith(`successful run model: gpt-5.5-codex. ${progress.title}`), done.title);
    // A receipt whose own frame names the run's last report keeps that one, once.
    const own = executorChip({ ...stop, executor_observation: { ...observation, revision: 4 } });
    assert.match(own.lastUpdate, /revision 4\./);
    assert.equal(cardExecutorChip({ executorChip: progress, requestedExecutor: selected }, own).title, own.title);
    assert.equal(own.title.match(/Latest observed activity/g).length, 1);
    // A later replay of the same receipt adds no second copy.
    assert.equal(cardExecutorChip({ executorChip: card, requestedExecutor: selected }, receipt).title, card.title);
});

test('a run\'s last report stays with its own attempt: another known attempt, or a report unplaced across a retry, stays out', () => {
    const selection = { requested_executor_route: { kind: 'agent_session', target_id: 'codex=gpt-5.5' }, subagent_task_id: 'c' };
    const selected = requestedExecutor(selection);
    const observation = (task_attempt) => ({ task_id: 'c', task_attempt, run_id: 'old-run', attempt_id: 'a01', harness_id: 'codex',
        phase: 'harness.event', revision: 3 });
    const progress = (task_attempt) => executorChip({ ...selection, executor_route: 'codex=gpt-5.5', ts: '2026-11-22T10:00:02Z', task_attempt,
        executor_observation: observation(String(task_attempt)) });
    const stop = (task_attempt, extra = {}) => executorChip({ ...selection, executor_route: 'codex=gpt-5.5', status: 'cancelled',
        ts: '2026-11-22T10:05:00Z', task_attempt, execution_evidence: { delegated_runs_started: 1, delegated_runs_settled: 0 }, ...extra });
    const card = (prior, next, marks = {}) => cardExecutorChip({ executorChip: prior, requestedExecutor: selected, ...marks }, next);
    const ended = 'Delegated to your Codex account — 1 run(s) started, none settled';
    // The review's control: attempt 1 reported, attempt 2 stopped. The old run is no history of attempt 2.
    assert.equal(card(progress(1), stop(2)).title, ended);
    assert.equal(card(progress(1), stop(2)).label, 'Codex · not settled');
    // The same attempt keeps it, as does an end naming none on a card that crossed no attempt boundary.
    assert.match(card(progress(1), stop(1)).title, /\. Latest observed activity: Codex, model gpt-5\.5 \(requested\), run old-run, attempt a01, revision 3\./);
    assert.match(card(progress(1), stop(undefined)).title, /run old-run/);
    // A terminal frame's own observation of another known attempt is not its report and proves no start of this attempt.
    assert.equal(stop(2, { executor_observation: observation('1') }).title, ended);
    assert.equal(stop(2, { executor_observation: observation('1') }).lastUpdate, undefined);
    const bare = (task_attempt) => executorChip({ ...selection, status: 'cancelled', task_attempt, executor_observation: observation('1') });
    assert.equal(bare(2).title, 'Selected Codex, model gpt-5.5 (requested); no dispatch was recorded before it ended');
    // Its recorded request is worded at paint, so the Executor fact follows the reader's language; the record stays English.
    assert.equal(executorTitle(bare(2)), bare(2).title);
    setMissTransport(() => Promise.resolve(null));
    const priorPayload = currentPayload();
    try {
        applyPayload({ language: 'ru', english: false, entries: { 'Selected {name}, model {model} (requested); no dispatch was recorded before it ended': {
            text: 'Выбран {name}, модель {model} (запрошена); до конца запуск не записан' } } });
        assert.equal(executorTitle(bare(2)), 'Выбран Codex, модель gpt-5.5 (запрошена); до конца запуск не записан');
        assert.equal(bare(2).title, 'Selected Codex, model gpt-5.5 (requested); no dispatch was recorded before it ended');
    } finally { applyPayload(priorPayload || { language: '', english: true, entries: {} }); }
    assert.match(bare(1).title, /run outcome is unknown\. Latest observed activity: .*run old-run/);
    // History carried forward keeps its attempt: a later attempt's receipt does not inherit it, the same attempt's does.
    const first = card(progress(1), stop(1));
    assert.equal(first.lastUpdateAttempt, 1);
    assert.equal(card(first, stop(2)).title, ended);
    assert.equal(card(first, stop(1)).title, first.title);
    // An end naming no attempt dates to the card's latest attempt after a crossed boundary (task_phase_chip's marks), or to
    // none when that attempt was interrupted; a report it cannot place after the boundary stays out.
    assert.equal(card(progress(1), stop(undefined), { childRetried: true, childAttempt: 2 }).title, ended);
    assert.match(card(progress(2), stop(undefined), { childRetried: true, childAttempt: 2 }).title, /run old-run/);
    assert.equal(card(progress(1), stop(undefined), { childRetried: true, childAttempt: 1, interruptedAttempt: 1 }).title, ended);
    assert.match(card(progress(2), stop(undefined), { childRetried: true, childAttempt: 1, interruptedAttempt: 1 }).title, /run old-run/);
    // A newer numbered report the live line keeps past a lagging census (attempt 3 over 2) is no history of an end naming
    // no attempt across the known retry; a terminal of that same numbered attempt takes it.
    const ahead = { childRetried: true, childAttempt: 2 };
    assert.equal(cardExecutorChip({ executorChip: progress(3), requestedExecutor: selected, ...ahead }, null).observation?.task_attempt, '3');
    assert.equal(card(progress(3), stop(undefined), ahead).title, ended);
    assert.equal(card(progress(3), stop(undefined), ahead).lastUpdate, undefined);
    assert.match(card(progress(3), stop(3), ahead).title, /run old-run/);
    const unplaced = executorChip({ ...selection, executor_route: 'codex=gpt-5.5', executor_observation: observation('') });
    assert.equal(card(unplaced, stop(undefined), { childRetried: true, childAttempt: 2 }).title, ended);
    assert.match(card(unplaced, stop(undefined)).title, /run old-run/);
});

test('a run\'s live report speaks for its own attempt: a crossed boundary returns the line to the frozen request', () => {
    const selection = { requested_executor_route: { kind: 'agent_session', target_id: 'codex=gpt-5.5' }, subagent_task_id: 'c' };
    const selected = requestedExecutor(selection);
    const progress = (attempt, model) => executorChip({ ...selection, executor_route: 'codex=gpt-5.5', ts: '2026-11-22T10:00:02Z',
        task_attempt: attempt, executor_observation: { task_id: 'c', task_attempt: String(attempt), run_id: `run-${attempt}`,
            attempt_id: 'a01', harness_id: 'codex', phase: 'harness.event', revision: 3, model, model_source: 'observed' } });
    const dispatch = (attempt) => executorChip({ ...selection, executor_route: 'codex=gpt-5.5', subagent_event: 'running',
        ts: '2026-11-22T10:05:00Z', task_attempt: attempt });
    // A later attempt's dispatch is not held to the earlier attempt's report; the same attempt's is, and an older
    // attempt's report never replaces a newer one's.
    assert.equal(keepStickyExecutorChip(progress(1, 'first'), dispatch(2)), false);
    assert.equal(keepStickyExecutorChip(progress(1, 'first'), dispatch(1)), true);
    assert.equal(keepStickyExecutorChip(progress(1, 'first'), dispatch(undefined)), true);
    assert.equal(keepStickyExecutorChip(progress(2, 'second'), progress(1, 'first')), true);
    assert.equal(keepStickyExecutorChip(progress(1, 'first'), progress(2, 'second')), false);
    // On the card: a report it cannot place at its latest attempt is retired to history and the line names the request.
    const card = { executorChip: progress(1, 'first'), requestedExecutor: selected, childRetried: true, childAttempt: 2 };
    const placed = cardExecutorChip(card, null);
    assert.equal(placed.observation, undefined);
    assert.equal(placed.label, 'Codex');
    assert.equal(executorModelText(placed, selected), 'requested gpt-5.5');
    assert.equal(placed.lastUpdateAttempt, 1);
    assert.match(placed.lastUpdate, /model first \(observed\), run run-1/);
    // An interrupted attempt's report retires with it; the latest attempt's own report, and any report before a boundary, stay.
    assert.equal(cardExecutorChip({ ...card, childAttempt: 1, interruptedAttempt: 1 }, null).observation, undefined);
    const own = progress(2, 'second');
    assert.equal(cardExecutorChip({ ...card, executorChip: own }, null), own);
    const fresh = progress(1, 'first');
    assert.equal(cardExecutorChip({ executorChip: fresh, requestedExecutor: selected, childAttempt: 1 }, null), fresh);
    // Out of order: the first numbered census (attempt 2, no boundary seen yet) retires attempt 1's report; a newer numbered
    // report outruns a lagging census and survives it catching up; the latest attempt's own report stays (control).
    assert.equal(cardExecutorChip({ executorChip: fresh, requestedExecutor: selected, childAttempt: 2 }, null).observation, undefined);
    const ahead = progress(3, 'third');
    assert.equal(cardExecutorChip({ ...card, executorChip: ahead }, null), ahead);
    assert.equal(cardExecutorChip({ ...card, executorChip: ahead, childAttempt: 3 }, null), ahead);
    assert.equal(cardExecutorChip({ executorChip: own, requestedExecutor: selected, childAttempt: 2 }, null), own);
    // A terminal of the new attempt after the retirement keeps none of the old run.
    const stop = executorChip({ ...selection, executor_route: 'codex=gpt-5.5', status: 'cancelled', task_attempt: 2,
        ts: '2026-11-22T10:06:00Z', execution_evidence: { delegated_runs_started: 1, delegated_runs_settled: 0 } });
    assert.equal(cardExecutorChip({ ...card, executorChip: placed }, stop).title, 'Delegated to your Codex account — 1 run(s) started, none settled');
});

test('a line\'s requested qualifier is read in the reader\'s language; a refusal names its own request on its parent\'s row', () => {
    assert.equal(modelWords({ model: 'gpt-5.5', requested: true }), 'requested gpt-5.5');
    assert.equal(modelWords('gpt-5.5-2026-11'), 'gpt-5.5-2026-11');
    assert.equal(modelWords(''), '');
    // Only the typed request is translated: a model whose own identifier begins with `requested ` stays verbatim.
    setMissTransport(() => Promise.resolve(null));
    const prior = currentPayload();
    try {
        for (const [language, template] of [['de', 'angefordert {model}'], ['ru', 'запрошена {model}']]) {
            applyPayload({ language, english: false, entries: { 'requested {model}': { text: template } } });
            assert.equal(modelWords({ model: 'gpt-5.5', requested: true }), template.replace('{model}', 'gpt-5.5'), language);
            assert.equal(modelWords({ model: 'requested gpt-5.5' }), 'requested gpt-5.5', language);
            assert.equal(modelWords('requested gpt-5.5'), 'requested gpt-5.5', language);
        }
    } finally { applyPayload(prior || { language: '', english: true, entries: {} }); }
    const refusal = { chat_id: 1, task_id: 'parent', ts: 'T', role: 'system', system_type: 'subagent_admission_notice', is_progress: true,
        subagent_task_id: 'kid', parent_task_id: 'parent', delegation_role: 'subagent', subagent_event: 'rejected', status: 'failed',
        content: '⚠️ Refused.', requested_executor_route: { kind: 'api_model', target_id: 'openai@team::gpt-6' },
        parent_cognitive_route: { model: 'x' }, executor_route: 'codex' };
    const row = admissionRefusalRow(refusal);
    assert.equal(row.content, '⚠️ Refused. · requested openai::gpt-6');
    assert.equal(row.task_id, 'parent');
    for (const key of ['subagent_task_id', 'parent_task_id', 'delegation_role', 'subagent_event', 'status', 'requested_executor_route',
        'parent_cognitive_route', 'executor_route']) assert.equal(row[key], undefined, key);
    assert.equal(admissionRefusalRow({ ...refusal, requested_executor_route: undefined, executor_route: undefined }).content, '⚠️ Refused.');
});

test('two reported answers under the request\'s short name keep their own ids; the used route stays beside them', () => {
    const solve = (reported) => modelExecutionView({ source: 'usable_solve_response', requested_model: 'vendor-a/sol', used_model: 'vendor-b/sol',
        reported_model: reported, requested_use_local: false, used_local: false }, true);
    const [c, d] = [solve('vendor-c/sol'), solve('vendor-d/sol')];
    assert.equal(c.value, 'vendor-c/sol (initial request: openrouter::vendor-a/sol) · route: vendor-b/sol');
    assert.equal(d.value, 'vendor-d/sol (initial request: openrouter::vendor-a/sol) · route: vendor-b/sol');
    assert.equal(c.model, 'sol');
    assert.equal(d.model, 'sol', 'the compact line keeps the short name');
    // A helper's used route names no local connection; the provider's own report is shown as reported.
    assert.equal(modelExecutionView({ source: 'usable_solve_response', requested_model: 'vendor-a/sol', used_model: 'openai@team::sol',
        reported_model: 'vendor-c/sol' }, true).value, 'vendor-c/sol (initial request: openrouter::vendor-a/sol) · route: openai::sol');
    // Without a report the used route answers, said once.
    assert.equal(solve('').value, 'vendor-b/sol (initial request: openrouter::vendor-a/sol)');
    // The request was the route used, but the provider reports another model under the same short name: the report
    // answers by its own id, the used route beside it; an ordinary report on the route that served it stays short.
    const served = (reported, used = 'vendor-a/sol') => modelExecutionView({ source: 'usable_solve_response', requested_model: 'vendor-a/sol',
        used_model: used, reported_model: reported, requested_use_local: false, used_local: false }, true).value;
    assert.equal(served('vendor-b/sol'), 'vendor-b/sol · route: vendor-a/sol');
    assert.equal(served('vendor-c/sol'), 'vendor-c/sol · route: vendor-a/sol');
    assert.equal(served('sol-2026-11'), 'sol-2026-11 · route: vendor-a/sol', 'another short name is no collision');
    assert.equal(modelExecutionView({ source: 'usable_solve_response', requested_model: 'openai@team::sol', used_model: 'openai@team::sol',
        reported_model: 'sol', provider: 'openai' }, true).value, 'sol · openai · route: openai::sol', 'its own id on the route that served it');
});

test('a frame that repeats no selection takes its card\'s; a recorded start or report proves the work was sent', () => {
    const known = requestedExecutor({ requested_executor_route: { kind: 'agent_session', target_id: 'codex=gpt-5.5' } });
    const card = (executorChip) => ({ executorChip, requestedExecutor: known });
    // A sparse terminal record (a log or stored detail naming neither route nor selection) reads with the card's selection.
    const sparse = executorChip({ status: 'cancelled', subagent_event: 'cancelled' }, known);
    assert.equal(sparse.title, 'Selected Codex, model gpt-5.5 (requested); no dispatch was recorded before it ended');
    assert.equal(executorChip({ status: 'cancelled' }), null, 'without a selection it names no executor');
    // A running notice is a recorded start: the card that saw it is never told "no dispatch"; a queued one is.
    const started = executorChip({ subagent_event: 'running' }, known);
    assert.equal(started.title, 'Selected Codex, model gpt-5.5 (requested) — run evidence arrives with the terminal receipt');
    assert.equal(cardExecutorChip(card(started), sparse).title, 'The Codex route was assigned, model gpt-5.5 (requested), but this '
        + 'terminal record has no delegated-run evidence; the run outcome is unknown');
    assert.equal(cardExecutorChip(card(executorChip({ subagent_event: 'scheduled' }, known)), sparse), sparse);
    // A terminal record carrying its run's last report proves the start and keeps that report as history, once.
    const last = executorChip({ subagent_task_id: 'c', status: 'cancelled', executor_observation: { task_id: 'c', task_attempt: '0',
        run_id: 'r', attempt_id: 'a01', harness_id: 'codex', phase: 'p', revision: 4, model: 'gpt-5.5-2026-11', model_source: 'observed' } }, known);
    assert.equal(last.label, 'Codex · run outcome unknown');
    assert.match(last.lastUpdate, /^Latest observed activity: Codex, model gpt-5\.5-2026-11 \(observed\), run r, attempt a01, revision 4\./);
    assert.ok(last.title.endsWith(`unknown. ${last.lastUpdate}`), last.title);
    assert.equal(cardExecutorChip(card(executorChip({ subagent_event: 'running' }, known)), last).title, last.title);
    assert.equal(executorModelText(last, known), 'requested gpt-5.5');
});

test('a settled card never answers with a progress report\'s observed model; known paths carry their recorded lane', () => {
    const known = requestedExecutor({ requested_executor_route: { kind: 'agent_session', target_id: 'codex=gpt-5.5' } });
    const observed = { harness: 'codex', observation: { model: 'gpt-5.5-2026-11', model_source: 'observed' } };
    assert.equal(executorModelText(observed, known), 'gpt-5.5-2026-11', 'in flight, the run\'s own report');
    assert.equal(executorModelText(observed, known, true), 'requested gpt-5.5', 'settled: the scheduled request');
    assert.equal(executorModelText(observed, null, true), '', 'settled without a selection: no model rather than progress');
    const asked = { harness: 'cursor', observation: { model: 'cursor-model', model_source: 'requested' } };
    assert.equal(executorModelText(asked, null, true), 'requested cursor-model', 'an old record\'s explicit request stays a request');
    // A harness reports no lane and an API selection spells its own: a name saying `local` adds no mark on these paths.
    assert.equal(executorModelText({ harness: 'codex', observation: { model: 'local-coder', model_source: 'observed' } }), 'local-coder');
    const remote = requestedExecutor({ requested_executor_route: { kind: 'api_model', target_id: 'local-model' } });
    assert.equal(executorModelText(null, remote), 'requested local-model');
    assert.equal(compactModel('local-model', selectionLane(remote, 'local-model')), 'local-model');
    const local = requestedExecutor({ requested_executor_route: { kind: 'api_model', target_id: 'qwen.gguf (local)' } });
    assert.equal(compactModel('qwen.gguf', selectionLane(local, 'qwen.gguf')), 'qwen.gguf (local)');
    assert.equal(selectionLane(local, 'other-model'), false, 'another route (a fallback) records no lane: its words make none');
    assert.equal(compactModel('local-model'), 'local-model (local)', 'the standalone legacy reading is unchanged');
});

test('a solve\'s route is its provider and model however it is spelled; a reported short label proves no other route', () => {
    const fact = (requested_model, used_model, extra = {}) => ({ source: 'usable_solve_response', requested_model, used_model,
        requested_use_local: false, used_local: false, provider: 'openrouter', llm_call_id: 'c', ...extra });
    const details = (target, f) => requestedFact(requestedExecutor({ requested_executor_route: { kind: 'api_model', target_id: target } }),
        null, modelExecutionView(f).model, f);
    // A bare id is OpenRouter's: `openrouter::sol` and `sol` are one route; `openai::sol` is another provider's.
    assert.equal(details('openrouter::sol', fact('openrouter::sol', 'sol')), '');
    assert.equal(details('sol', fact('sol', 'openrouter::sol')), '');
    assert.equal(details('openai::sol', fact('openai::sol', 'sol')), 'openai::sol');
    // The provider's own short label names what answered; alone it claims no other route.
    assert.equal(details('openrouter::sol', fact('openrouter::sol', 'sol', { reported_model: 'sol-2026-11' })), 'sol');
    // Solve details: equivalent spellings on one lane are one request; another provider's is an initial request, said in
    // full where the short names agree; a lane switch stays one.
    assert.equal(modelExecutionView(fact('openrouter::sol', 'sol')).value, 'sol · openrouter');
    assert.equal(modelExecutionView(fact('openai::sol', 'sol')).value, 'sol (initial request: openai::sol) · openrouter');
    assert.equal(modelExecutionView(fact('sol', 'sol', { requested_use_local: true })).value, 'sol (initial request: sol (local)) · openrouter');
    assert.equal(modelExecutionView(fact('sol', 'sol', { requested_use_local: undefined })).value, 'sol · openrouter', 'an unstated lane is no switch');
    // A provider's own id is read in the provider that served it: an endpoint naming the selected model adds no request.
    const qwen = 'openai-compatible::Qwen/Qwen3.5-9B';
    const served = (used_model, reported_model) => fact(qwen, used_model, { provider: 'openai-compatible', reported_model });
    assert.equal(details(qwen, served(qwen, 'Qwen/Qwen3.5-9B')), '');
    // A genuine provider, model or lane change still keeps the request.
    assert.equal(details(qwen, served('openrouter::Qwen/Qwen3.5-9B', 'Qwen/Qwen3.5-9B')), qwen);
    assert.equal(details(qwen, served(qwen, 'Qwen/Qwen3.6-9B')), 'Qwen3.5-9B');
    assert.equal(details(qwen, { ...served(qwen, 'Qwen/Qwen3.5-9B'), used_local: true }), 'Qwen3.5-9B', 'remote asked, local ran');
    // The chosen route keeps the dispatch grammar even when its bytes equal the provider's reported id: a bare id is
    // OpenRouter's, so a solve another provider served keeps the request, said in full where the short names agree.
    assert.equal(requestedFact({ kind: 'api_model', model: 'Qwen/Qwen3.5-9B', harness: '' }, null, 'Qwen3.5-9B', { source: 'usable_solve_response',
        requested_model: 'Qwen/Qwen3.5-9B', used_model: qwen, reported_model: 'Qwen/Qwen3.5-9B', used_local: false }), 'Qwen/Qwen3.5-9B');
    assert.equal(details('Qwen/Qwen3.5-9B', served(qwen, 'Qwen/Qwen3.5-9B')), 'Qwen/Qwen3.5-9B');
});

test('a helper\'s solve names no local API connection; a root\'s keeps its route', () => {
    const solve = (used_model) => ({ source: 'usable_solve_response', requested_model: used_model, requested_use_local: false, used_model,
        reported_model: 'example-model-2026', used_local: false, provider: 'openai', llm_call_id: 'c' });
    assert.equal(modelExecutionView(solve('openai@team::example-model'), true).value, 'example-model-2026 · openai · route: openai::example-model');
    assert.equal(modelExecutionView(solve('openai@team::example-model')).value, 'example-model-2026 · openai · route: openai@team::example-model');
    // A name the resolver refuses is never shown, and no model it did not select is invented.
    assert.equal(modelExecutionView(solve('openai@Bad_Name::example-model'), true).value, 'example-model-2026 · openai');
});

test('a stored child detail uses the host selection, never a card reconstruction from raw snapshots', () => {
    const route = { kind: 'agent_session', target_id: 'codex=gpt-5.5' };
    const detail = (extra) => terminalChildFrame('kid', { parentId: 'root', role: 'reader' }, { status: 'cancelled', ...extra });
    const snapshot = { route: { kind: 'api_model', target_id: 'openai@work::stale', credential_profile_id: 'work-account' } };
    const stored = { configured_subagent: snapshot, metadata: { configured_subagent: snapshot } };
    const projected = detail({ ...stored, requested_executor_route: { ...route, credential_profile_id: 'work-account' } });
    assert.deepEqual(projected.requested_executor_route, route);
    assert.equal(projected.executor_route, undefined);
    assert.equal(projected.model, undefined);
    assert.equal(detail(stored).requested_executor_route, undefined, 'a missing host projection must remain missing');
    for (const bad of [{ kind: 'agent_session' }, { kind: 'shell', target_id: 'x' }, { kind: 'api_model', target_id: '  ' }, 'codex=gpt-5.5']) {
        assert.equal(detail({ ...stored, requested_executor_route: bad }).requested_executor_route, undefined, JSON.stringify(bad));
    }
});

test('a stored detail names a named API connection\'s model only: its carrier holds no local name and a refused name selects nothing', () => {
    const detail = (extra) => terminalChildFrame('kid', { parentId: 'root', role: 'reader' }, { status: 'completed', ...extra });
    // The table tests/test_subagent_card_facts.py reads too: one display semantics in Python and here.
    const cases = JSON.parse(readFileSync(new URL('../../tests/fixtures/api_route_display.json', import.meta.url), 'utf8'));
    for (const { raw, shown } of cases) {
        // Check even a legacy field before presentation: the short line hides much of a route.
        const frame = detail({ requested_executor_route: { kind: 'api_model', target_id: raw } });
        assert.deepEqual(frame.requested_executor_route, shown === null ? undefined : { kind: 'api_model', target_id: shown }, raw);
    }
    // Every reader of the route grammar agrees on it (apiRouteParts): the card's display, the key's label and the Models
    // editor's source, which keeps a refused name's spelling (it never rewrites a stored route) and says it is not valid.
    for (const { raw, shown, connection, valid } of cases) {
        const head = raw.replace(/ \(local\)$/, '');
        assert.equal(displayApiModelTarget(raw), shown ?? '', raw);
        assert.equal(routeConnectionId(raw), connection, raw);
        assert.equal(apiRouteParts(head).valid, valid, raw);
        const parsed = parseModelSource(head);
        assert.equal(parsed.valid !== false, valid, raw);
        // The editor writes the route back as it reads it: a refused head keeps its exact spelling, a built-in key's long
        // spelling (`@default`) its canonical short one, and a head that names no model writes no route.
        const written = !valid ? head : apiRouteParts(head).model ? head.replace('@default::', '::') : '';
        assert.equal(composeModelSource(parsed.source, parsed.model), written, raw);
        if (connection) assert.deepEqual(parsed, { source: connection, model: apiRouteParts(head).model }, raw);
    }
    assert.deepEqual(parseModelSource('openai@Bad_Name::m'), { source: 'openai@Bad_Name', model: 'm', valid: false });
    for (const refused of ['openai@::m', 'claudexor@x::codex=m', 'openai@default@x::m']) {
        const parsed = parseModelSource(refused);
        assert.equal(parsed.valid, false, refused);
        assert.equal(composeModelSource(parsed.source, parsed.model), refused, refused);
    }
});

test('a frame records only the coordinator facts it states', () => {
    // A sparse observation or receipt states none, so a card keeps what earlier frames recorded and invents no basis.
    assert.equal(recordedCoordinator({ model: 'm', executor_route: 'codex' }), null);
    assert.equal(recordedCoordinator({ executor_route: 'codex', execution_evidence: { delegated_runs_started: 1 } }), null);
    // A scheduling notice records the parent's snapshot: its route and provenance, and no billing it did not give.
    assert.deepEqual(recordedCoordinator({ parent_cognitive_route: { model: 'openai/gpt-6', chosen_by: 'inherited' } }),
        { route: 'gpt-6', provenance: 'inherited' });
    // A running or observation frame records the route and billing basis subagent_message_meta stamped.
    assert.deepEqual(recordedCoordinator({ model: 'openai/gpt-6', coordinator_route: 'recorded (local)', coordinator_billing: 'measured',
        use_local_model: true }), { route: 'recorded (local)', billing: 'measured' });
});

test('historical frames without the failed counter reconstruct it from succeeded', () => {
    // v6.94–v6.99 durable frames carry delegated_runs_succeeded but not
    // delegated_runs_failed (the counters merged at v6.100.0): the chip must
    // reconstruct the exact complement, never render an all-failed delegation
    // as a clean receipt (adversarial wave B-ADV-1).
    const historical = executorChip({
        executor_route: 'codex',
        execution_evidence: {
            delegated_runs_started: 2, delegated_runs_settled: 2,
            delegated_runs_succeeded: 0, subscription_cost_usd: null,
        },
    });
    assert.equal(historical.label, 'Codex · 2 of 2 runs failed');
    // A frame with NEITHER counter disclosed no outcome split: never quiet-clean.
    const bare = executorChip({
        executor_route: 'codex',
        execution_evidence: {
            delegated_runs_started: 1, delegated_runs_settled: 1,
            subscription_cost_usd: null,
        },
    });
    assert.equal(bare.label, 'Codex · run outcome unknown');
    assert.match(bare.title, /1 run settled, subscription spend undisclosed/);
});

test('#360 A: the quiet label names the helper; every exception stays one visible phrase', () => {
    const settledChip = (counts, extra = {}) => executorChip({
        executor_route: 'codex', ...extra,
        execution_evidence: { subscription_cost_usd: null, harness_models: [], ...counts },
    });
    // Clean: bare name, the receipt facts in the detail sentence.
    const clean = settledChip({ delegated_runs_started: 2, delegated_runs_settled: 2,
        delegated_runs_succeeded: 2, delegated_runs_failed: 0, subscription_cost_usd: 0.25 });
    assert.equal(clean.label, 'Codex');
    assert.match(clean.title, /2 runs settled \(2 ok\), \$0\.25 subscription/);
    // One success does not make a mixed delegation quiet-clean.
    const mixed = settledChip({ delegated_runs_started: 3, delegated_runs_settled: 3,
        delegated_runs_succeeded: 2, delegated_runs_failed: 1 });
    assert.equal(mixed.label, 'Codex · 1 of 3 runs failed');
    assert.match(mixed.title, /3 runs settled \(2 ok, 1 failed\)/);
    assert.equal(settledChip({ delegated_runs_started: 1, delegated_runs_settled: 1,
        delegated_runs_succeeded: 0, delegated_runs_failed: 1 }).label, 'Codex · run failed');
    // The aggregate failed count includes source-unverified runs: the phrase
    // never turns them into physical failures, and a mix names both.
    assert.equal(settledChip({ delegated_runs_started: 1, delegated_runs_settled: 1, delegated_runs_succeeded: 0,
        delegated_runs_failed: 1, delegated_runs_source_unresolved: 1 }).label, 'Codex · run unverified');
    // The same run may both fail physically and have incomplete source coverage.
    // Equal counts alone do not distinguish it from a coverage-only non-success.
    for (const states of [['failed', 'source_incomplete'], ['cancelled', 'source_incomplete']]) {
        const overlapping = settledChip({ delegated_runs_started: 1, delegated_runs_settled: 1,
            delegated_runs_succeeded: 0, delegated_runs_failed: 1, delegated_runs_source_unresolved: 1,
            delegated_run_failure_states: states });
        assert.equal(overlapping.label, 'Codex · run failed or unverified');
        assert.match(overlapping.title, /1 run settled \(0 ok, 1 failed or unverified\)/);
    }
    assert.equal(settledChip({ delegated_runs_started: 1, delegated_runs_settled: 1,
        delegated_runs_succeeded: 0, delegated_runs_failed: 1, delegated_runs_source_unresolved: 1,
        delegated_run_failure_states: ['source_incomplete'] }).label, 'Codex · run unverified');
    const both = settledChip({ delegated_runs_started: 3, delegated_runs_settled: 3, delegated_runs_succeeded: 1,
        delegated_runs_failed: 2, delegated_runs_source_unresolved: 1 });
    assert.equal(both.label, 'Codex · 2 of 3 runs failed or unverified');
    assert.doesNotMatch(both.label, /retry|recovered|then/);
    // The detail counts the same aggregate the phrase names, never a second failure.
    assert.match(both.title, /3 runs settled \(1 ok, 2 failed or unverified\)/);
    assert.match(both.title, /1 run\(s\) settled with unverified work-order coverage/);
    // A malformed frame naming more unverified runs than failures still reads unverified.
    assert.equal(settledChip({ delegated_runs_started: 1, delegated_runs_settled: 1, delegated_runs_succeeded: 1,
        delegated_runs_failed: 0, delegated_runs_source_unresolved: 1 }).label, 'Codex · run unverified');
    // Runs started but never settled beside settled successes stay visible.
    const partly = settledChip({ delegated_runs_started: 3, delegated_runs_settled: 2,
        delegated_runs_succeeded: 2, delegated_runs_failed: 0 });
    assert.equal(partly.label, 'Codex · 1 of 3 runs not settled');
    assert.match(partly.title, /1 more run\(s\) started, not settled/);
    // Every evidence branch stays evidence-grade; the counts never leave the detail.
    for (const chip of [clean, mixed, both, partly]) {
        assert.equal(chip.hasEvidence, true);
        assert.doesNotMatch(chip.label, /\bok\b|dispatched|subscription|\$/);
    }
    // The frozen refusal wording is untouched.
    assert.equal(executorChip({ executor_route: 'codex', reason_code: 'subagent_executor_unavailable' }).label,
        'Codex · blocked');
});

test('an evidence-bearing chip is marked so sticky rendering can refuse downgrades', () => {
    // The card keeps an evidence-bearing (receipt) chip when a later
    // evidence-less (dispatch) frame arrives — the history sync after
    // justFinished anchors on a mid-run row (adversarial wave B-ADV-2). The
    // marker is the contract between executorChip and the sticky card state.
    assert.equal(executorChip({ executor_route: 'codex' }).hasEvidence, false);
    assert.equal(executorChip({
        executor_route: 'codex',
        execution_evidence: { delegated_runs_started: 1, delegated_runs_settled: 1, delegated_runs_failed: 0, subscription_cost_usd: null },
    }).hasEvidence, true);
});

test('actor findings rows, omitted counts and the durable pointer ride the shared formatter', () => {
    const text = formatReviewProjection({ panels: [{
        panel_id: 'panel_f', surface: 'task_acceptance', aggregate_signal: 'FAIL',
        quorum: { required: 2, contributed: 2, configured: 3 },
        actors: [
            {
                slot_id: 'slot_1', model: 'm1', reason: 'summary text',
                coverage: { criteria_total: 3, findings: 10 },
                findings: [
                    {
                        id: 'f1', severity: 'critical', item: 'Preflop sizing is wrong',
                        summary: 'Sizing deviates from the baseline in early position',
                        evidence: 'raises 2bb from UTG', recommendation: 'raise 2.5bb',
                    },
                    { severity: 'low', verdict: 'FAIL', item: 'Style nit', reason: 'inconsistent spacing' },
                ],
                findings_omitted: 8,
                response_ref: { call_id: 'review_task_acceptance_slot_1_resp', sha256: 'a'.repeat(64) },
            },
            {
                slot_id: 'slot_2', model: 'm2',
                coverage: { criteria_total: 3, findings: 0 },
                findings: [], findings_omitted: 0,
                response_ref: { call_id: 'c2' },
            },
            {
                // A pre-findings-era projection: count says findings existed,
                // rows are absent — the pointer is the only resolvable source.
                slot_id: 'slot_3', model: 'm3',
                coverage: { criteria_total: 3, findings: 4 },
                response_ref: { call_id: 'c3' },
            },
        ],
    }] });

    assert.match(text, /Reviewer slot_1 finding: \[critical\] f1 Preflop sizing is wrong — summary: Sizing deviates from the baseline in early position — evidence: raises 2bb from UTG — fix: raise 2\.5bb/);
    assert.match(text, /Reviewer slot_1 finding: \[low FAIL\] Style nit — reason: inconsistent spacing/);
    assert.match(text, /Reviewer slot_1 findings omitted: 8/);
    assert.match(text, /Reviewer slot_1 full response: observability call review_task_acceptance_slot_1_resp/);
    assert.doesNotMatch(text, /Reviewer slot_2 finding:/);
    // The durable pointer is unconditional: bounded rows, per-string
    // truncation markers and pre-findings-era projections all resolve there.
    assert.match(text, /Reviewer slot_2 full response: observability call c2/);
    assert.match(text, /Reviewer slot_3 full response: observability call c3/);
    assert.doesNotMatch(text, /Reviewer slot_3 finding:/);
});


test('delegated cost preserves a known subtotal independently of estimation', () => {
    for (const estimated of [false, true]) {
        const chip = executorChip({executor_route: 'codex', execution_evidence: {
            delegated_runs_started: 1, delegated_runs_settled: 1,
            subscription_cost_usd: 2, subscription_cost_estimated: estimated,
            subscription_cost_partial: true, harness_models: [],
        }});
        assert.ok(chip.title.includes(`Tracked: ${estimated ? 'up to ' : ''}$2.00 (total unknown)`));
        assert.doesNotMatch(chip.title, /undisclosed|upper bound/);
    }
});
