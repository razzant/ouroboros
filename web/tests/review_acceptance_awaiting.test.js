import assert from 'node:assert/strict';
import test from 'node:test';

import {
    formatReviewProjection,
    mergeReviewGroup,
    planReviewGroupFromTaskDetail,
    renderReviewsSection,
    taskAcceptanceGroupFromTaskDetail,
} from '../modules/review_presentation.js';

// Projected actor rows as the host publishes them. An awaited slot is typed by
// `operation_state` alone; the projection carries neither `ok` nor `status`.
const actor = (slot, fields = {}) => ({
    slot_id: slot, model: 'codex=gpt-6-astra', provider: 'openrouter', actor_role: 'task acceptance',
    transport_status: 'success', parse_status: 'valid', semantic_verdict: 'PASS',
    coverage: { criteria_total: 1, findings: 0 }, quorum_contribution: true, reason: 'done',
    enforcement_impact: 'supports_pass', operation_id: `op-${slot}`, operation_state: 'settled',
    late_result_pending: false, executions: [], response_ref: {}, ...fields,
});
const SILENT = {
    parse_status: 'malformed', semantic_verdict: '', coverage: { criteria_total: 0, findings: 0 },
    quorum_contribution: false, enforcement_impact: 'abstains',
};
const AWAITING = {
    ...SILENT, transport_status: 'awaiting', parse_status: 'awaiting', operation_state: 'pending_dispatch',
    late_result_pending: true,
    reason: 'No answer recorded: the host returned at the dispatch barrier before this reviewer answered.',
};
const FAILED = {
    ...SILENT, transport_status: 'provider_transport_error', operation_state: 'late_settled',
    reason: 'delegated review session ended failed: harness_unavailable',
};
const TIMEOUT = {
    ...SILENT, transport_status: 'timeout', operation_state: 'in_flight', late_result_pending: true,
    reason: 'Timeout after 1800s; physical review operation remains in flight',
};
const CUSTODY_LOST = { ...FAILED, operation_state: 'custody_lost', late_result_pending: true };
const REFUSED = { ...SILENT, transport_status: 'not_dispatched', operation_state: 'not_dispatched' };
const SAID_FAIL = { semantic_verdict: 'FAIL', enforcement_impact: 'veto', reason: 'broken' };
const SAID_DEGRADED = { semantic_verdict: 'DEGRADED', quorum_contribution: false, enforcement_impact: 'abstains' };

function panel(actors, fields = {}) {
    const answered = actors.filter((row) => row.transport_status === 'success').length;
    return {
        panel_id: 'panel_a72b23783ba34908', surface: 'task_acceptance', authority: 'host_root',
        aggregate_signal: 'DEGRADED', transport_status: 'awaiting', parse_status: 'awaiting',
        coverage: { actors_configured: actors.length, transport_success: answered, parse_valid: answered, quorum_contributing: 0 },
        quorum: { required: 2, contributed: 0, configured: actors.length }, reason: 'recorded reason',
        enforcement_impact: 'pending_feedback', actors, superseded: false, applied_source_status: 'unavailable',
        task_attempt: 1, panel_index: 0, ...fields,
    };
}

const detail = (panels, status) => ({
    task_id: 'root', ...(status === undefined ? {} : { status }), review_projection: { panels },
});
const group = (panels, status) => taskAcceptanceGroupFromTaskDetail(detail(panels, status), 'root');
const facts = (value) => ({
    state: value.state, tone: value.tone, progress: value.progress, verdict: value.verdict,
    ...('activeCount' in value ? { activeCount: value.activeCount } : {}),
});
const html = (value) => renderReviewsSection([value], {
    sectionExpanded: true,
    expandedGroups: new Set([value.id]),
    expandedAttempts: new Set([`${value.id}:${value.attempts.at(-1).id}`]),
});
// The rows the owner reads, without the Record facts that keep the recorded tokens.
const rows = (markup) => markup.replace(/<details class="review-facts"[\s\S]*?<\/details>/g, '');
const attemptRow = (markup) => markup.match(/<div class="review-seat" data-review-opinion="[^"]+">[\s\S]*?<\/div><\/div>/)?.[0] || '';
const allAwaiting = () => panel([actor('s1', AWAITING), actor('s2', AWAITING), actor('s3', AWAITING)]);
const passWithHole = () => panel([actor('s1'), actor('s2'), actor('s3', AWAITING)], { aggregate_signal: 'PASS' });
const failWithHole = () => panel(
    [actor('s1', SAID_FAIL), actor('s2', AWAITING), actor('s3', AWAITING)], { aggregate_signal: 'FAIL' },
);

test('an awaited panel of a running task reads as work in progress', () => {
    const live = group([allAwaiting()], 'running');
    const expected = { state: 'running', tone: 'working', progress: 'in progress · 0 of 3 answered', verdict: 'DEGRADED' };
    assert.deepEqual(facts(live), { ...expected, activeCount: 1 });
    assert.deepEqual(facts(live.attempts[0]), expected);
    const text = live.attempts[0].detailText;
    assert.match(text, /· verdict=none \(3 awaiting; held as DEGRADED\) · transport=awaiting · parse=awaiting ·/);
    assert.match(text, /^Reviewer s1: .* · transport=awaiting · parse=awaiting · verdict=none · quorum=abstains/m);
    assert.doesNotMatch(text, /provider_transport_error|malformed|Pending dispatch/);
    const markup = html(live);
    // Live work is the moving status mark beside its words, on the group row and on the panel's row.
    assert.match(markup, /class="review-state" data-tone="working"><span class="status-mark" aria-hidden="true" data-shape="moving"[^>]*>(?:<i><\/i>){3}<\/span><span class="review-state-word">in progress · 0 of 3 answered</);
    assert.match(attemptRow(markup), /review-state-word">in progress · 0 of 3 answered</);
    assert.doesNotMatch(rows(markup), /DEGRADED/);
});

test('the same panel of a task that is not running is a recorded gap, never a running review', () => {
    const expected = {
        state: 'terminal', tone: 'neutral', progress: 'no verdict · 0 of 3 answered', verdict: 'DEGRADED', activeCount: 0,
    };
    // A detail without a top-level status (a child lifecycle frame, a terminal message) reads the same.
    for (const status of ['completed', 'failed', 'cancelled', 'interrupted', 'cancel_requested', '', undefined]) {
        assert.deepEqual(facts(group([allAwaiting()], status)), expected, String(status));
    }
    const markup = html(group([allAwaiting()], 'completed'));
    assert.match(markup, /class="review-state ui-status" data-tone="neutral">no verdict · 0 of 3 answered</);
    assert.doesNotMatch(markup, /in progress/);
});

test('a settled failure without a verdict reads in words and keeps its warning and its detail', () => {
    const settled = panel([actor('s1', FAILED), actor('s2', FAILED), actor('s3', FAILED)], {
        transport_status: 'provider_transport_error', parse_status: 'malformed', enforcement_impact: 'degrades_completion',
    });
    const reviewer = (slot) => [
        `Reviewer ${slot}: role=task acceptance · provider=openrouter · model=codex=gpt-6-astra · transport=provider_transport_error · parse=malformed · verdict=none · quorum=abstains · enforcement=abstains`,
        `Reviewer ${slot} coverage: criteria_total=0, findings=0`,
        `Reviewer ${slot} reason: delegated review session ended failed: harness_unavailable`,
    ];
    for (const status of ['completed', 'running']) {
        const value = group([settled], status);
        // #1369: the owner reads words; the stored DEGRADED stays in the attempt detail below.
        assert.deepEqual(facts(value), {
            state: 'terminal', tone: 'warn', progress: 'no verdict · 0 of 3 answered · 3 unavailable', verdict: 'DEGRADED', activeCount: 0,
        });
        assert.equal(value.attempts[0].detailText, [
            'Review panel panel_a72b23783ba34908: task_acceptance · authority=host_root · verdict=DEGRADED · transport=provider_transport_error · parse=malformed · quorum=0/3 (required 2) · enforcement=degrades_completion',
            'Panel reason: recorded reason',
            'Panel coverage: actors_configured=3, transport_success=0, parse_valid=0, quorum_contributing=0',
            ...reviewer('s1'), ...reviewer('s2'), ...reviewer('s3'),
            'Cost unavailable',
        ].join('\n'));
        assert.match(html(value), /class="review-state ui-status" data-tone="warn">no verdict · 0 of 3 answered · 3 unavailable</);
        assert.doesNotMatch(rows(html(value)), /DEGRADED/);
        assert.match(html(value), /<dd>task_acceptance · authority=host_root · verdict=DEGRADED · /);
    }
    // A reviewer's own DEGRADED answer is an answer: the panel still has no verdict, and nothing is unavailable.
    const judged = group([panel([actor('s1', SAID_DEGRADED), actor('s2', SAID_DEGRADED), actor('s3', FAILED)], {
        transport_status: 'partial', parse_status: 'valid',
    })], 'completed');
    assert.deepEqual(facts(judged), {
        state: 'terminal', tone: 'warn', progress: 'no verdict · 2 of 3 answered · 1 unavailable', verdict: 'DEGRADED', activeCount: 0,
    });
    // A settled verdict keeps its own word.
    const passed = group([panel([actor('s1'), actor('s2'), actor('s3')], { aggregate_signal: 'PASS' })], 'completed');
    assert.deepEqual([passed.progress, passed.verdict], ['', 'PASS']);
});

test('an expired window or lost custody stays a warning even while the task runs', () => {
    for (const lost of [TIMEOUT, CUSTODY_LOST]) {
        const value = group([panel([actor('s1', lost), actor('s2', lost), actor('s3', lost)], {
            transport_status: lost.transport_status, parse_status: 'malformed',
        })], 'running');
        assert.deepEqual(facts(value), {
            state: 'terminal', tone: 'warn', progress: 'no verdict · 0 of 3 answered · 3 unavailable', verdict: 'DEGRADED', activeCount: 0,
        });
        assert.match(value.attempts[0].detailText, new RegExp(`· verdict=DEGRADED · transport=${lost.transport_status} · parse=malformed ·`));
        assert.doesNotMatch(value.attempts[0].detailText, /awaiting/);
    }
});

test('a reviewer FAIL keeps its word and its tone while other slots are awaited', () => {
    const live = group([failWithHole()], 'running');
    assert.deepEqual(facts(live), {
        state: 'running', tone: 'error', progress: 'FAIL · 1 of 3 answered', verdict: 'FAIL', activeCount: 1,
    });
    assert.match(live.attempts[0].detailText, /· verdict=FAIL \(2 awaiting\) ·/);
    assert.match(html(live), /class="review-state ui-status" data-tone="error">rejected · 1 of 3 answered</);
    assert.doesNotMatch(rows(html(live)), /FAIL/);
    assert.deepEqual(facts(group([failWithHole()], 'completed')), {
        state: 'terminal', tone: 'error', progress: 'FAIL · 1 of 3 answered', verdict: 'FAIL', activeCount: 0,
    });
});

test('a quorum PASS with an awaited slot never reads as a bare PASS', () => {
    const live = group([passWithHole()], 'running');
    assert.deepEqual(facts(live), {
        state: 'running', tone: 'working', progress: 'PASS so far · 2 of 3 answered', verdict: 'PASS', activeCount: 1,
    });
    assert.match(live.attempts[0].detailText, /· verdict=PASS \(1 awaiting\) ·/);
    for (const status of ['completed', undefined]) {
        const ended = group([passWithHole()], status);
        assert.deepEqual(facts(ended), {
            state: 'terminal', tone: 'done', progress: 'PASS · 2 of 3 answered', verdict: 'PASS', activeCount: 0,
        });
        assert.match(html(ended), /class="review-state ui-status" data-tone="ok">passed · 2 of 3 answered</);
        assert.match(attemptRow(html(ended)), /data-tone="ok">passed · 2 of 3 answered</);
    }
    // The settled panel is the only one that speaks with the verdict alone.
    const settled = group([panel([actor('s1'), actor('s2'), actor('s3')], { aggregate_signal: 'PASS' })], 'running');
    assert.deepEqual(facts(settled), { state: 'terminal', tone: 'done', progress: '', verdict: 'PASS', activeCount: 0 });
    assert.match(html(settled), /data-tone="ok">passed<\/span><svg class="review-chevron"[\s\S]*?<\/svg><span class="review-check-meta">1 attempt<\/span>/);
});

test('a slot that is neither answered nor awaited keeps the panel loud beside a wait', () => {
    for (const broken of [FAILED, TIMEOUT, CUSTODY_LOST, REFUSED]) {
        const mixed = () => panel([actor('s1', broken), actor('s2', AWAITING), actor('s3', AWAITING)], {
            transport_status: broken.transport_status, parse_status: 'malformed',
        });
        assert.deepEqual(facts(group([mixed()], 'running')), {
            state: 'running', tone: 'warn', progress: 'in progress · 0 of 3 answered · 1 unavailable', verdict: 'DEGRADED', activeCount: 1,
        }, broken.operation_state);
        assert.deepEqual(facts(group([mixed()], 'completed')), {
            state: 'terminal', tone: 'warn', progress: 'no verdict · 0 of 3 answered · 1 unavailable', verdict: 'DEGRADED', activeCount: 0,
        }, broken.operation_state);
        assert.match(html(group([mixed()], 'running')), /class="review-state ui-status" data-tone="warn">in progress · 0 of 3 answered · 1 unavailable</);
    }
    // A FAIL outranks the warning; a quorum PASS beside a dead slot is still a warning.
    const failed = panel([actor('s1', SAID_FAIL), actor('s2', FAILED), actor('s3', AWAITING)], { aggregate_signal: 'FAIL' });
    assert.deepEqual([group([failed], 'running').tone, group([failed], 'running').progress],
        ['error', 'FAIL · 1 of 3 answered · 1 unavailable']);
    const passed = panel([actor('s1'), actor('s2'), actor('s3', FAILED), actor('s4', AWAITING)], { aggregate_signal: 'PASS' });
    assert.deepEqual([group([passed], 'completed').tone, group([passed], 'completed').progress],
        ['warn', 'PASS · 2 of 4 answered · 1 unavailable']);
    // A reviewer that answered DEGRADED did answer: nothing is unavailable.
    const judged = panel([actor('s1', SAID_DEGRADED), actor('s2', AWAITING), actor('s3', AWAITING)]);
    assert.deepEqual([group([judged], 'running').tone, group([judged], 'running').progress],
        ['working', 'in progress · 1 of 3 answered']);
});

test('a superseded awaited panel is history, and only the last panel can be live', () => {
    const old = { ...passWithHole(), superseded: true, panel_id: 'panel_old' };
    const current = panel([actor('s1'), actor('s2'), actor('s3')], { aggregate_signal: 'PASS', panel_index: 1 });
    const value = group([old, current], 'running');
    assert.deepEqual(facts(value), { state: 'terminal', tone: 'done', progress: '', verdict: 'PASS', activeCount: 0 });
    assert.deepEqual(facts(value.attempts[0]), {
        state: 'superseded', tone: 'done', progress: 'PASS · 2 of 3 answered', verdict: 'PASS',
    });
    assert.match(value.attempts[0].detailText, /· verdict=PASS \(1 awaiting\) · .* · superseded$/m);
    assert.doesNotMatch(value.attempts[0].detailText, /so far/);
    // An awaited panel that is not the last one is not live either.
    const earlier = group([{ ...allAwaiting(), panel_id: 'panel_old' }, { ...current }], 'running');
    assert.deepEqual([earlier.attempts[0].state, earlier.attempts[0].progress, earlier.activeCount],
        ['terminal', 'no verdict · 0 of 3 answered', 0]);
});

test('a pending-dispatch row that carries an answer is not counted as awaited', () => {
    const answered = { operation_state: 'pending_dispatch', late_result_pending: true };
    const value = group([panel([actor('s1'), actor('s2'), actor('s3', answered)], {
        aggregate_signal: 'PASS', transport_status: 'success', parse_status: 'valid',
    })], 'running');
    assert.deepEqual(facts(value), { state: 'terminal', tone: 'done', progress: '', verdict: 'PASS', activeCount: 0 });
    assert.match(value.attempts[0].detailText, /· verdict=PASS · transport=/);
    assert.match(value.attempts[0].detailText, /^Reviewer s3: .* · transport=success · parse=valid · verdict=PASS ·/m);
    assert.doesNotMatch(value.attempts[0].detailText, /awaiting/);
    // The same custody state without an answer is the awaited slot.
    const silent = group([panel([actor('s1'), actor('s2'), actor('s3', { ...AWAITING })], { aggregate_signal: 'PASS' })], 'running');
    assert.deepEqual([silent.state, silent.progress], ['running', 'PASS so far · 2 of 3 answered']);
});

test('a panel stored with failure words about a wait prints the typed state on its reviewer lines', () => {
    const stored = { ...AWAITING, transport_status: 'provider_transport_error', parse_status: 'malformed',
        reason: 'Pending dispatch; the physical review operation is in flight (window 21600s)' };
    const text = formatReviewProjection({ panels: [panel([actor('s1', stored), actor('s2')], {
        transport_status: 'partial', parse_status: 'malformed',
    })] });
    assert.match(text, /^Reviewer s1: .* · transport=awaiting · parse=awaiting · verdict=none ·/m);
    assert.match(text, /^Reviewer s2: .* · transport=success · parse=valid · verdict=PASS ·/m);
    // History is immutable: the stored panel line keeps the words it was written with.
    assert.match(text, /^Review panel .* · verdict=none \(1 awaiting; held as DEGRADED\) · transport=partial · parse=malformed ·/m);
});

test('merging keeps a terminal group terminal and its header free of a live progress phrase', () => {
    const store = new Map();
    const pending = () => group([allAwaiting()], 'running');
    const settled = () => group([panel([actor('s1'), actor('s2'), actor('s3')], { aggregate_signal: 'PASS' })], 'completed');
    mergeReviewGroup(store, pending());
    assert.deepEqual(facts(mergeReviewGroup(store, settled())), {
        state: 'terminal', tone: 'done', progress: '', verdict: 'PASS', activeCount: 0,
    });
    for (let round = 0; round < 2; round += 1) {
        const merged = mergeReviewGroup(store, pending());
        assert.deepEqual(facts(merged), { state: 'terminal', tone: 'done', progress: '', verdict: 'PASS', activeCount: 0 });
        assert.deepEqual(facts(merged.attempts[0]), { state: 'terminal', tone: 'done', progress: '', verdict: 'PASS' });
        assert.doesNotMatch(html(merged), /in progress/);
    }
    // The finished task's unanswered panel survives a stale running snapshot with its own words.
    const ended = new Map();
    mergeReviewGroup(ended, group([allAwaiting()], 'completed'));
    const merged = mergeReviewGroup(ended, pending());
    assert.deepEqual(facts(merged), {
        state: 'terminal', tone: 'neutral', progress: 'no verdict · 0 of 3 answered', verdict: 'DEGRADED', activeCount: 0,
    });
});

test('repeated merges never repaint a live FAIL or a live warning as plain work', () => {
    const mixed = () => panel([actor('s1', TIMEOUT), actor('s2', AWAITING), actor('s3', AWAITING)]);
    for (const [build, tone, progress] of [
        [failWithHole, 'error', 'FAIL · 1 of 3 answered'],
        [mixed, 'warn', 'in progress · 0 of 3 answered · 1 unavailable'],
        [allAwaiting, 'working', 'in progress · 0 of 3 answered'],
    ]) {
        const store = new Map();
        for (let round = 0; round < 3; round += 1) {
            const merged = mergeReviewGroup(store, group([build()], 'running'));
            assert.deepEqual([merged.state, merged.tone, merged.progress, merged.activeCount], ['running', tone, progress, 1]);
        }
    }
});

const FINGERPRINT = '1ef46f0328c4a52973d7cd9aa23e7ab5301a257eab72379a4637ac67e0855852';
const planActor = (slot, fields) => ({ slot_id: slot, model: 'codex=gpt-6-astra', failure_code: '', ...fields });
const PLAN_AWAITING = { ok: false, operation_state: 'pending_dispatch', late_result_pending: true };
const PLAN_ANSWERED = { ok: true, operation_state: 'settled', late_result_pending: false };

function planGroup(wave, current = { status: 'open' }) {
    return planReviewGroupFromTaskDetail({
        task_id: 'root',
        status: 'running', // custody is live work only while the owning task runs
        plan_review_state: {
            schema_version: 2,
            current_attempt: { fingerprint: FINGERPRINT, reason: '', ...current },
            waves: wave ? [{
                request_fingerprint: FINGERPRINT, cycle_index: 1, aggregate: 'DEGRADED', closed: false, paid: true,
                reviewed_at: '2026-09-20T10:47:07.300000+00:00', ...wave,
            }] : [],
            waves_omitted: 0,
        },
    }, 'root');
}

test('a plan group follows the same merge rules', () => {
    const collecting = () => planGroup({
        custody_pending: true,
        actors: [planActor('s1', PLAN_ANSWERED), planActor('s2', PLAN_AWAITING), planActor('s3', PLAN_AWAITING)],
    });
    const closed = () => planGroup({
        custody_pending: false, closed: true, aggregate: 'GREEN',
        actors: [planActor('s1', PLAN_ANSWERED), planActor('s2', PLAN_ANSWERED), planActor('s3', PLAN_ANSWERED)],
    }, { status: 'closed' });
    const store = new Map();
    for (let round = 0; round < 2; round += 1) {
        const merged = mergeReviewGroup(store, collecting());
        assert.deepEqual([merged.state, merged.tone, merged.progress], ['running', 'working', 'in progress · 1 of 3 answered']);
    }
    assert.deepEqual(facts(mergeReviewGroup(store, closed())), {
        state: 'terminal', tone: 'done', progress: '', verdict: 'GREEN', activeCount: 0,
    });
    const stale = mergeReviewGroup(store, collecting());
    assert.deepEqual(facts(stale), { state: 'terminal', tone: 'done', progress: '', verdict: 'GREEN', activeCount: 0 });
    assert.doesNotMatch(html(stale), /in progress/);

    // A wave with a dead slot stays a warning across merges; an ordinary open attempt stays plain work.
    const loud = new Map();
    const withDeadSlot = () => planGroup({
        custody_pending: true,
        actors: [planActor('s1', PLAN_ANSWERED), planActor('s2', { ok: false, operation_state: 'settled', failure_code: 'run_failed' }),
            planActor('s3', PLAN_AWAITING)],
    });
    for (let round = 0; round < 2; round += 1) {
        const merged = mergeReviewGroup(loud, withDeadSlot());
        assert.deepEqual([merged.tone, merged.progress], ['warn', 'in progress · 1 of 3 answered · 1 unavailable']);
    }
    // A collecting wave that a later projection replaces is retired without its live phrase.
    const replaced = new Map();
    mergeReviewGroup(replaced, collecting());
    const retired = mergeReviewGroup(replaced, planGroup(null, { fingerprint: 'f'.repeat(64), status: 'closed' }));
    assert.deepEqual(facts(retired.attempts[0]), { state: 'superseded', tone: 'neutral', progress: '', verdict: 'DEGRADED' });
    assert.deepEqual([retired.state, retired.progress, retired.activeCount], ['terminal', '', 0]);
    assert.doesNotMatch(html(retired), /in progress/);
    const plain = new Map();
    for (let round = 0; round < 2; round += 1) {
        const merged = mergeReviewGroup(plain, planGroup(null));
        assert.deepEqual([merged.state, merged.tone, merged.progress], ['running', 'working', '']);
    }
});

test('a panel that settled after its task ended keeps its label and its leading note', () => {
    const note = 'The reviewers answered after the task ended: PASS.';
    const value = group([panel([actor('s1'), actor('s2'), actor('s3')], {
        aggregate_signal: 'PASS', transport_status: 'success', parse_status: 'valid',
        enforcement_impact: 'allows_completion', late_settlement: { note },
    })], 'completed');
    assert.deepEqual(facts(value), { state: 'terminal', tone: 'done', progress: '', verdict: 'PASS', activeCount: 0 });
    assert.equal(value.attempts[0].label, 'panel panel_a72b23783ba34908 · settled after the task ended');
    assert.equal(value.summary, note);
    assert.ok(value.attempts[0].detailText.startsWith(`${note}\nReview panel panel_a72b23783ba34908: task_acceptance · authority=host_root · verdict=PASS · transport=success · parse=valid ·`));
});
