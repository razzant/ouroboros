import assert from 'node:assert/strict';
import test from 'node:test';

import {
    formatReviewProjection,
    planReviewGroupFromTaskDetail,
    renderReviewsSection,
} from '../modules/review_presentation.js';

const FINGERPRINT = '1ef46f0328c4a52973d7cd9aa23e7ab5301a257eab72379a4637ac67e0855852';

// Actor rows as the host stores them: a planned wait carries `pending_dispatch`
// and a prose window note, an answered slot carries `ok`, a real failure a code.
const AWAITING = {
    slot_id: 'triad_286lhb', model: 'codex=gpt-6-astra', ok: false,
    operation_state: 'pending_dispatch', late_result_pending: true, failure_code: '',
    error: 'Pending dispatch; the physical review operation is in flight (window 21600s)',
};
const ANSWERED = {
    slot_id: 'triad_w45a8z', model: 'Cursor Grok 4.6 Extra High Fast', ok: true,
    operation_state: 'late_settled', late_result_pending: false, failure_code: '', error: null,
};
const FAILED = {
    slot_id: 'triad_bkydwq', model: 'codex=gpt-6-astra', ok: false,
    operation_state: 'settled', late_result_pending: false, failure_code: 'run_failed',
    error: 'delegated review session run-e2336ca586e0 ended failed',
};

// Custody is activity only while the owning task runs: every live case states that fact.
function planGroup(wave, status = 'running') {
    return planReviewGroupFromTaskDetail({
        task_id: 'root',
        status,
        plan_review_state: {
            schema_version: 2,
            current_attempt: { fingerprint: FINGERPRINT, status: 'open', reason: '' },
            waves: [{
                request_fingerprint: FINGERPRINT,
                cycle_index: 1,
                aggregate: 'DEGRADED',
                closed: false,
                paid: true,
                reviewed_at: '2026-09-20T10:47:07.300000+00:00',
                counts: { configured: 3, parseable: 1, quorum: 2, blocking: 0, note: 3, need_evidence: 0 },
                ...wave,
            }],
            waves_omitted: 0,
        },
    }, 'root');
}

const expandedHtml = (group) => renderReviewsSection([group], {
    sectionExpanded: true,
    expandedGroups: new Set([group.id]),
    expandedAttempts: new Set([`${group.id}:${group.attempts[0].id}`]),
});

// A reviewer row is `<model> · <state>`; the marker is the state word, never a slot id.
const availabilityLines = (attempt, marker) => attempt.detailText
    .split('\n')
    .filter((line) => line.includes(marker));

test('a plan wave whose reviewers may still answer reads as work in progress', () => {
    const group = planGroup({
        custody_pending: true,
        actors: [AWAITING, ANSWERED, { ...AWAITING, slot_id: 'triad_bkydwq' }],
    });
    const attempt = group.attempts[0];
    assert.equal(attempt.tone, 'working');
    assert.equal(group.tone, 'working');
    // The stored wave is untouched; only the sentence about it changes.
    assert.equal(attempt.verdict, 'DEGRADED');
    assert.equal(group.verdict, 'DEGRADED');
    assert.match(attempt.detailText, /^Verdict: none \(wave held open\)$/m);
    assert.deepEqual(availabilityLines(attempt, ' · awaiting'), [
        'codex=gpt-6-astra · awaiting',
        'codex=gpt-6-astra · awaiting',
    ]);
    assert.deepEqual(availabilityLines(attempt, ' · unavailable'), []);
    assert.doesNotMatch(attempt.detailText, /Pending dispatch|triad_/);

    const html = expandedHtml(group);
    // Live work: the moving status mark beside the words, on the group row and on the wave's own row.
    assert.match(html, /class="review-check-head"[^>]*>[\s\S]*?class="review-state" data-tone="working"><span class="status-mark"[^>]*data-shape="moving"/);
    assert.equal(group.progress, 'in progress · 1 of 3 answered');
    assert.match(html, /review-state-word">in progress · 1 of 3 answered</);
    assert.match(html, /data-review-opinion="[^"]+"><div class="review-seat-head"><span class="review-seat-name">wave 1<\/span>[\s\S]*?class="review-state" data-tone="working">[\s\S]*?review-state-word">in progress · 1 of 3 answered</);
    assert.doesNotMatch(html, /DEGRADED|\d unavailable/);
});

test('a settled wave without quorum reads no verdict in the neutral tone and names its unavailable reviewers', () => {
    const group = planGroup({
        custody_pending: false,
        actors: [{ ...FAILED, slot_id: 'triad_286lhb' }, FAILED],
    });
    const attempt = group.attempts[0];
    // The stored DEGRADED word is the host's placeholder: it stays on the record and never paints.
    assert.equal(attempt.tone, 'neutral');
    assert.equal(group.tone, 'neutral');
    assert.equal(attempt.verdict, 'DEGRADED');
    assert.equal(attempt.progress, 'no verdict · 0 of 2 answered · 2 unavailable');
    assert.match(attempt.detailText, /^Verdict: none — fewer reviewers answered than needed$/m);
    assert.deepEqual(availabilityLines(attempt, ' · unavailable'), [
        'codex=gpt-6-astra · unavailable',
        'codex=gpt-6-astra · unavailable',
    ]);
    assert.deepEqual(availabilityLines(attempt, ' · awaiting'), []);

    const html = expandedHtml(group);
    assert.match(html, /class="review-state ui-status" data-tone="neutral">no verdict · 0 of 2 answered · 2 unavailable</);
    assert.doesNotMatch(html, /in progress|DEGRADED|run_failed/);

    // A slot that settled after the wave closed is a terminal answer too: only
    // the typed wait states change the wording.
    const late = planGroup({
        custody_pending: false,
        actors: [{ ...FAILED, operation_state: 'late_settled', late_result_pending: true }],
    });
    assert.deepEqual(availabilityLines(late.attempts[0], ' · unavailable'), [
        'codex=gpt-6-astra · unavailable',
    ]);
    // A settled wave with a real verdict word keeps it: only the placeholder is replaced.
    const revise = planGroup({ custody_pending: false, aggregate: 'REVISE_PLAN', actors: [ANSWERED] });
    assert.equal(revise.attempts[0].progress, '');
    assert.match(revise.attempts[0].detailText, /^Verdict: REVISE_PLAN$/m);
});

test('an earlier-schema plan verdict reads in the shared words; the token stays in its facts', () => {
    for (const [aggregate, word] of [['GREEN', 'passed'], ['REVISE_PLAN', 'plan revision asked'], ['REVIEW_REQUIRED', 'review required']]) {
        const group = planGroup({ custody_pending: false, closed: aggregate === 'GREEN', aggregate, actors: [ANSWERED] }, 'completed');
        const rows = expandedHtml(group).replace(/<details class="review-facts"[\s\S]*?<\/details>/g, '');
        // The subject row and the wave row both say it; neither prints the token.
        assert.equal(rows.split(`">${word}</span>`).length - 1, 2, `${aggregate}: ${rows}`);
        assert.doesNotMatch(rows, new RegExp(aggregate));
        assert.match(group.attempts[0].detailText, new RegExp(`^Verdict: ${aggregate}$`, 'm'));
    }
});

test('a dead reviewer row quotes the reported sentence and never the code', () => {
    const cause = 'Selected model is at capacity.\nPlease try a different model.';
    const dead = { ...FAILED, reported_cause: cause };
    const group = planGroup({ custody_pending: false, actors: [dead, ANSWERED] });
    const attempt = group.attempts[0];
    assert.deepEqual(availabilityLines(attempt, ' · unavailable'), [
        'codex=gpt-6-astra · unavailable — "Selected model is at capacity. Please try a different model."',
    ]);
    assert.doesNotMatch(attempt.detailText, /run_failed|triad_|ended failed/);
    assert.equal(attempt.progress, 'no verdict · 1 of 2 answered · 1 unavailable');
    // Without a reported sentence the row ends at the state word; the code stays in Logs.
    const silent = planGroup({ custody_pending: false, actors: [{ ...FAILED, reported_cause: '' }, ANSWERED] });
    assert.deepEqual(availabilityLines(silent.attempts[0], ' · unavailable'), ['codex=gpt-6-astra · unavailable']);
    // A $0 refusal was never sent the plan: it is not called unavailable.
    const refused = planGroup({
        custody_pending: false,
        actors: [{ ...FAILED, operation_state: 'not_dispatched', failure_code: 'subscription_window_exhausted' }, ANSWERED],
    });
    assert.deepEqual(availabilityLines(refused.attempts[0], ' · not sent'), ['codex=gpt-6-astra · not sent']);
    assert.deepEqual(availabilityLines(refused.attempts[0], ' · unavailable'), []);
    assert.equal(refused.attempts[0].progress, 'no verdict · 1 of 2 answered · 1 unavailable');
});

test('an in-flight wave names its failed slot and its awaited slot separately', () => {
    const group = planGroup({
        custody_pending: true,
        actors: [ANSWERED, FAILED, { ...AWAITING, slot_id: 'triad_qq41xk' }],
    });
    const attempt = group.attempts[0];
    assert.deepEqual(availabilityLines(attempt, ' · awaiting'), [
        'codex=gpt-6-astra · awaiting',
    ]);
    assert.deepEqual(availabilityLines(attempt, ' · unavailable'), [
        'codex=gpt-6-astra · unavailable',
    ]);
    assert.match(attempt.detailText, /^Verdict: none \(wave held open\)$/m);
    // A slot that is neither answered nor awaited keeps the wave's warning.
    assert.equal(attempt.progress, 'in progress · 1 of 3 answered · 1 unavailable');
    assert.equal(group.progress, attempt.progress);
    assert.deepEqual([attempt.tone, group.tone], ['warn', 'warn']);
    const html = expandedHtml(group);
    assert.match(html, /class="review-state ui-status" data-tone="warn">in progress · 1 of 3 answered · 1 unavailable</);
});

test('a reviewer whose window expired stays unresolved instead of awaited', () => {
    const lost = {
        ...AWAITING, slot_id: 'triad_qq41xk', operation_state: 'custody_lost',
        failure_code: 'review_custody_lost', error: 'Review custody was lost before the slot settled',
    };
    const group = planGroup({ custody_pending: true, actors: [ANSWERED, lost] });
    const attempt = group.attempts[0];
    assert.deepEqual(availabilityLines(attempt, ' · no answer'), [
        'codex=gpt-6-astra · no answer', // the raw custody state stays in the task detail and Logs, never on the row
    ]);
    const quoted = planGroup({ custody_pending: true, actors: [ANSWERED, { ...lost, reported_cause: 'Selected model is at capacity.' }] });
    assert.deepEqual(availabilityLines(quoted.attempts[0], ' · no answer'), ['codex=gpt-6-astra · no answer — "Selected model is at capacity."']);
    assert.doesNotMatch(availabilityLines(quoted.attempts[0], ' · no answer').join(' '), /custody_lost|review_custody_lost/);
    assert.deepEqual(availabilityLines(attempt, ' · awaiting'), []);
    assert.equal(group.progress, 'unresolved · 1 of 2 answered · 1 unavailable');
    assert.deepEqual([attempt.tone, group.tone], ['warn', 'warn']);
    assert.match(expandedHtml(group), /class="review-state ui-status" data-tone="warn">unresolved · 1 of 2 answered · 1 unavailable</);

    // One slot that is still merely awaited returns the wave to progress; the lost slot stays counted.
    const mixed = planGroup({ custody_pending: true, actors: [ANSWERED, lost, AWAITING] });
    assert.equal(mixed.progress, 'in progress · 1 of 3 answered · 1 unavailable');
    assert.equal(mixed.tone, 'warn');
    assert.match(expandedHtml(mixed), /class="review-state ui-status" data-tone="warn">in progress · 1 of 3 answered · 1 unavailable</);
});

test('a wave recorded without the typed custody fields reads the same plain family', () => {
    const group = planGroup({
        actors: [{ slot_id: 'slot_3', model: 'openai/gpt-5.6-sol', ok: false, failure_code: 'window_exhausted' }],
    });
    const attempt = group.attempts[0];
    assert.equal(attempt.tone, 'neutral');
    assert.equal(attempt.progress, 'no verdict · 0 of 1 answered · 1 unavailable');
    assert.equal(attempt.detailText, [
        'Verdict: none — fewer reviewers answered than needed',
        'Closed: no',
        'Reviewer panel dispatched: yes',
        'Findings: 0 blocking · 3 note · 0 need_evidence',
        'openai/gpt-5.6-sol · unavailable',
        'Cost unavailable',
    ].join('\n'));
    assert.match(expandedHtml(group), /data-tone="neutral">no verdict · 0 of 1 answered · 1 unavailable<\/span><svg class="review-chevron"[\s\S]*?<\/svg><span class="review-check-meta">[^<]*1 attempt<\/span>/);
});

test('a plan wave of a task that is not running is a recorded gap, never live work', () => {
    const wave = { custody_pending: true, actors: [AWAITING, ANSWERED, { ...AWAITING, slot_id: 'triad_bkydwq' }] };
    for (const status of ['completed', 'failed', 'cancelled', 'interrupted', '', null]) { // null: a frame without a status
        const group = planGroup(wave, status);
        const attempt = group.attempts[0];
        assert.equal(attempt.state, 'terminal', String(status));
        assert.equal(group.state === 'running' || group.activeCount > 0, false, String(status));
        assert.equal(attempt.progress, 'no verdict · 1 of 3 answered');
        assert.equal(group.progress, 'no verdict · 1 of 3 answered');
        assert.equal(attempt.tone, 'neutral');
        assert.equal(group.tone, 'neutral');
        assert.equal(attempt.verdict, 'DEGRADED'); // the stored wave is untouched
        assert.doesNotMatch(expandedHtml(group), /in progress/);
    }
    // The same wave under a running task IS live work (the guard fires in both directions).
    const live = planGroup(wave, 'running');
    assert.equal(live.attempts[0].state, 'running');
    assert.equal(live.progress, 'in progress · 1 of 3 answered');
    assert.equal(live.tone, 'working');
    // A real failure beside the wait stays loud after the task ended.
    const mixed = planGroup({ custody_pending: true, actors: [AWAITING, ANSWERED, FAILED] }, 'completed');
    assert.equal(mixed.progress, 'no verdict · 1 of 3 answered · 1 unavailable');
    assert.equal(mixed.tone, 'warn');
});

// --- "since HH:MM": the host's own record of when it sent the request ---------
// The clock is the VIEWER's local time, so every expectation is computed with the
// same platform API the renderer uses; the suite must pass in any timezone.
const localClock = (iso) => new Date(iso)
    .toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: false });
const localDay = (iso) => new Date(iso).toLocaleDateString([], { month: 'short', day: 'numeric' });
const todayAt = (hour, minute) => {
    const at = new Date();
    at.setHours(hour, minute, 0, 0);
    return at.toISOString();
};
const daysAgoAt = (days, hour, minute) => {
    const at = new Date(Date.now() - days * 86400000);
    at.setHours(hour, minute, 0, 0);
    return at.toISOString();
};

test('an awaited reviewer says since when, in the viewer local clock', () => {
    const sent = todayAt(9, 15);
    const group = planGroup({
        custody_pending: true,
        actors: [{ ...AWAITING, awaiting_since: sent }, ANSWERED],
    });
    const [line] = availabilityLines(group.attempts[0], ' · awaiting');
    assert.match(line, /^.* · awaiting · since \d{2}:\d{2}$/);
    assert.equal(line, `codex=gpt-6-astra · awaiting · since ${localClock(sent)}`);
});

test('a reviewer row the host never timed keeps exactly its untimed line', () => {
    // Both directions of the same guard: no field, an empty field and a value that
    // is not an instant all render byte-identically to the untimed line.
    const before = 'codex=gpt-6-astra · awaiting';
    for (const awaiting_since of [undefined, '', '   ', 'soon', 'since yesterday']) {
        const group = planGroup({
            custody_pending: true,
            actors: [{ ...AWAITING, ...(awaiting_since === undefined ? {} : { awaiting_since }) }],
        });
        assert.deepEqual(
            availabilityLines(group.attempts[0], ' · awaiting'), [before], String(awaiting_since),
        );
    }
});

test('a wait that began on an earlier day names that day too', () => {
    // "since 23:50" must never be misread as tonight when the wait is 30 hours old.
    const sent = daysAgoAt(1, 23, 50);
    const group = planGroup({ custody_pending: true, actors: [{ ...AWAITING, awaiting_since: sent }] });
    const [line] = availabilityLines(group.attempts[0], ' · awaiting');
    assert.equal(line, `codex=gpt-6-astra · awaiting · since ${localDay(sent)} ${localClock(sent)}`);
    assert.doesNotMatch(line, /^.* · awaiting · since \d{2}:\d{2}$/);
});

test('an unresolved reviewer says since when it was sent, under the same rule', () => {
    const sent = todayAt(7, 5);
    const lost = {
        ...AWAITING, operation_state: 'custody_lost', failure_code: 'review_custody_lost',
        error: 'Review custody was lost before the slot settled',
    };
    const timed = planGroup({ custody_pending: true, actors: [{ ...lost, awaiting_since: sent }] });
    assert.deepEqual(availabilityLines(timed.attempts[0], ' · no answer'), [
        `codex=gpt-6-astra · no answer · since ${localClock(sent)}`,
    ]);
    const untimed = planGroup({ custody_pending: true, actors: [lost] });
    assert.deepEqual(availabilityLines(untimed.attempts[0], ' · no answer'), [
        'codex=gpt-6-astra · no answer',
    ]);
});

test('a settled reviewer never grows a since suffix', () => {
    // The stored moment belongs to the wait; an answer that arrived is judged by itself.
    const group = planGroup({
        custody_pending: false,
        actors: [{ ...FAILED, awaiting_since: todayAt(6, 30) }],
    });
    assert.deepEqual(availabilityLines(group.attempts[0], ' · unavailable'), [
        'codex=gpt-6-astra · unavailable',
    ]);
});

test('the acceptance panel reviewer line says since when, under the same rule', () => {
    const sent = todayAt(8, 41);
    const row = (fields) => ({
        slot_id: 's1', model: 'codex=gpt-6-astra', provider: 'openrouter',
        actor_role: 'task acceptance', transport_status: 'awaiting', parse_status: 'awaiting',
        semantic_verdict: '', coverage: { criteria_total: 0, findings: 0 },
        quorum_contribution: false, enforcement_impact: 'abstains', operation_id: 'op-s1',
        operation_state: 'pending_dispatch', late_result_pending: true, executions: [],
        response_ref: {}, reason: 'no answer yet', ...fields,
    });
    const panelText = (fields) => formatReviewProjection({
        panels: [{
            panel_id: 'panel_a72b23783ba34908', surface: 'task_acceptance', authority: 'host_root',
            aggregate_signal: 'DEGRADED', transport_status: 'awaiting', parse_status: 'awaiting',
            quorum: { required: 1, contributed: 0, configured: 1 },
            enforcement_impact: 'pending_feedback', actors: [row(fields)],
        }],
    }).split('\n').filter((line) => line.startsWith('Reviewer s1:'));

    assert.deepEqual(panelText({ awaiting_since: sent }), [
        `Reviewer s1: role=task acceptance · provider=openrouter · model=codex=gpt-6-astra · transport=awaiting · parse=awaiting · verdict=none · quorum=abstains · enforcement=abstains · since ${localClock(sent)}`,
    ]);
    assert.deepEqual(panelText({}), [
        'Reviewer s1: role=task acceptance · provider=openrouter · model=codex=gpt-6-astra · transport=awaiting · parse=awaiting · verdict=none · quorum=abstains · enforcement=abstains',
    ]);
    assert.deepEqual(panelText({ awaiting_since: 'soon' }), panelText({}));
    // A settled reviewer line is untouched even if a moment rode along.
    assert.deepEqual(
        panelText({ operation_state: 'settled', transport_status: 'success', parse_status: 'valid',
                    semantic_verdict: 'PASS', awaiting_since: sent }),
        ['Reviewer s1: role=task acceptance · provider=openrouter · model=codex=gpt-6-astra · transport=success · parse=valid · verdict=PASS · quorum=abstains · enforcement=abstains'],
    );
});
