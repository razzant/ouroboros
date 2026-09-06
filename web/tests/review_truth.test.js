import assert from 'node:assert/strict';
import test from 'node:test';

import {
    executorChip,
    formatReviewProjection,
    keepStickyExecutorChip,
    summarizeChatLiveEvent,
    taskOutcomeSeverity,
} from '../modules/log_events.js';

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
    assert.match(summary.fullBody, /\[RESULT\]\nConcrete child result/);
    assert.deepEqual(summary.meta, ['write=none', 'status=completed']);
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
    assert.match(summary.fullBody, /\[RESULT\]\nThe evidence-backed conclusion/);
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
    // ONE honest label — `{harness} · {state}` — never a bare product name: a
    // hover-only caveat is invisible on touch, to AT, and in copies, so the
    // run-state qualifier lives in the label itself. Identity (branded name,
    // mark geometry) comes from the harness_presentation SSOT.
    // A live frame carries no evidence either way — the label states the
    // dispatch-plan fact, never an evidence-grade negative (under the
    // pre-start charter the leaf usually IS running by round 1).
    assert.equal(chip.label, 'Codex · dispatched');
    assert.equal('icon' in chip, false, 'mark geometry belongs to harness_presentation');
    assert.match(chip.title, /Codex/);
    // WHERE it ran is all the chip can see. It has no access to the run's spend, so
    // it must not describe the billing: "on your codex subscription" told the owner
    // the work was covered, which the ledger alone can say (and only when the harness
    // says it — a zero there may mean unknown pricing, never "free").
    assert.ok(!/subscription/i.test(chip.title), chip.title);
    // The opaque `harness=model` spelling prints the HARNESS part only.
    assert.equal(executorChip({ executor_route: 'codex=gpt-5.6-sol' }).label, 'Codex · dispatched');
    // An unknown harness still gets a chip (a generic mark), never a crash.
    assert.equal(executorChip({ executor_route: 'opencode' }).label, 'OpenCode · dispatched');
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
    // Any other reason code keeps the ordinary layered truth.
    assert.equal(executorChip({ executor_route: 'codex', reason_code: 'tool_failure' }).label, 'Codex · dispatched');
});

test('the chip is layered truth: decision before evidence, receipt only from evidence', () => {
    // PRE-COMPLETION: the route is a DISPATCH decision, so the chip states only
    // that — never "Ran on your ... account", a receipt nothing has issued yet.
    // The claude harness prints its product name (owner decision: "Claude Code").
    const dispatched = executorChip({ executor_route: 'claude' });
    assert.equal(dispatched.label, 'Claude Code · dispatched');
    assert.match(dispatched.title, /Dispatched to Claude Code/);
    assert.match(dispatched.title, /run evidence arrives with the terminal receipt/);
    assert.match(dispatched.title, /itself runs on the API/);
    assert.doesNotMatch(dispatched.title, /Ran on your/);

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
    // Owner dictionary (plan D9): the label counts OK runs, not settled totals.
    assert.equal(delegated.label, 'Claude Code · 1 ok');
    assert.match(delegated.title, /Delegated to your Claude Code account — 1 run settled \(1 ok\), \$0\.00 subscription/);

    // ALL-FAILED runs must never read as success: the failure count rides the
    // LABEL, not only the tooltip (delegated_runs_failed was previously unread
    // and two failed runs rendered as a clean "2 run(s)" receipt).
    const allFailed = executorChip({
        executor_route: 'codex',
        execution_evidence: {
            delegated_runs_started: 2, delegated_runs_settled: 2,
            delegated_runs_succeeded: 0, delegated_runs_failed: 2,
            subscription_cost_usd: null, harness_models: [],
        },
    });
    assert.equal(allFailed.label, 'Codex · 0 ok, 2 failed');
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
    assert.equal(inFlight.label, 'Codex · 1 started, none settled');
    assert.match(inFlight.title, /1 run\(s\) started, none settled/);

    // ACCESS + UNVERIFIED (harness-health O13): the requested write surface and
    // the engine-disclosed applied access ride the tooltip; runs whose
    // work-order coverage never verified are discounted from ok AND named in
    // the label — a short ok-count with no visible cause reads as a bug.
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
    assert.match(withAccess.label, /1 unverified/);
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
    assert.match(attempted.title, /harness attempted, no delegated run succeeded/);

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
    assert.equal(historical.label, 'Codex · 0 ok, 2 failed');
    // A frame with NEITHER counter stays plain — exactly as wide as disclosed.
    const bare = executorChip({
        executor_route: 'codex',
        execution_evidence: {
            delegated_runs_started: 1, delegated_runs_settled: 1,
            subscription_cost_usd: null,
        },
    });
    assert.equal(bare.label, 'Codex · 1 run');
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
