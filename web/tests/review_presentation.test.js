import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

import {
    classifyReviewLifecycle,
    classifyReviewLifecyclePointer,
    createReviewHydrator,
    createReviewPresentationController,
    mergeReviewGroup,
    planReviewGroupFromTaskDetail,
    renderReviewsSection,
    reviewExecutionEvidence,
    reviewExecutionEvidenceList,
    reviewGroupFromHistoryRow,
    reviewGroupFromLifecycle,
    reviewGroupsFromTaskDetail,
    reviewReferenceFromRow,
    taskAcceptanceGroupFromTaskDetail,
} from '../modules/review_presentation.js';
import { reconcileReviewElementTree } from '../modules/review_dom_patch.js';
import { getLogTaskGroupId, isGroupedTaskEvent } from '../modules/log_events.js';
import { loadSkillReviewDetail } from '../modules/skill_review_card.js';

const groupedSkillRow = (overrides = {}) => ({
    system_type: 'skill_review',
    skill: 'alpha',
    status: 'clean',
    job_id: 'job-2',
    task_id: 'initiator-child',
    review_group: {
        surface: 'skill',
        id: 'task:root:alpha',
        presentation_owner_task_id: 'root',
        projected_attempt_count: 2,
        count_is_authoritative: false,
        attempts: [
            { job_id: 'job-1', skill: 'alpha', status: 'blockers', review_round: 1, snapshot_attempt: 1 },
            { job_id: 'job-2', skill: 'alpha', status: 'clean', review_round: 2, snapshot_attempt: 1 },
        ],
        ...overrides,
    },
});

test('task-bound Skill projection uses only the explicit presentation owner', () => {
    const group = reviewGroupFromHistoryRow(groupedSkillRow());
    assert.equal(group.presentationOwnerTaskId, 'root');
    assert.equal(group.subjectTaskId, '');
    assert.equal(group.initiatorTaskId, 'initiator-child');
    assert.deepEqual(group.attempts.map((attempt) => attempt.id), ['job-1', 'job-2']);
    assert.equal(group.attemptCount, 2);
    assert.equal(group.countIsAuthoritative, false);

    assert.equal(reviewGroupFromHistoryRow(groupedSkillRow({ presentation_owner_task_id: '' })), null);
    assert.equal(reviewGroupFromHistoryRow({
        ...groupedSkillRow(),
        review_group: { ...groupedSkillRow().review_group, presentation_owner_task_id: undefined },
    }), null);
});

test('Skill subject stays absent unless the canonical source provides it', () => {
    assert.equal(reviewGroupFromHistoryRow(groupedSkillRow()).subjectTaskId, '');
    assert.equal(reviewGroupFromHistoryRow(groupedSkillRow({ subject_task_id: 'child' })).subjectTaskId, 'child');
    assert.equal(reviewGroupFromLifecycle({ lifecycle: {
        kind: 'review', status: 'running', target: 'alpha', job_id: 'job-1',
        group_id: 'task:root:alpha', presentation_owner_task_id: 'root',
    } }).subjectTaskId, '');
});

test('typed review lifecycle distinguishes source-incomplete rows from unrelated lifecycle', () => {
    assert.equal(classifyReviewLifecycle({ lifecycle: { kind: 'install' } }).classification, 'not_review');
    assert.equal(classifyReviewLifecycle({ lifecycle: {
        kind: 'review', status: 'running', target: 'alpha', job_id: 'manual-1',
    } }).classification, 'source_incomplete');
    assert.equal(classifyReviewLifecycle({ lifecycle: {
        kind: 'review', status: 'running', target: 'alpha', job_id: 'job-1',
        group_id: 'task:root:alpha', presentation_owner_task_id: 'root',
    } }).classification, 'source_complete');
});

test('lifecycle pointers are typed acknowledgements, never generic task lineage', () => {
    const incomplete = { progress_meta: { lifecycle_pointer: {
        kind: 'review', job_id: 'job-1', status: 'running', target: 'alpha',
    } } };
    assert.equal(classifyReviewLifecyclePointer(incomplete).classification, 'source_incomplete');
    assert.equal(getLogTaskGroupId({ ...incomplete, task_id: 'skill_lifecycle_review_alpha_job-1' }), '');

    const complete = { progress_meta: { lifecycle_pointer: {
        kind: 'review', job_id: 'job-1', status: 'running', target: 'alpha',
        group_id: 'task:root:alpha', presentation_owner_task_id: 'root',
    } } };
    const classified = classifyReviewLifecyclePointer(complete);
    assert.equal(classified.classification, 'source_complete');
    assert.equal(classified.group.presentationOwnerTaskId, 'root');
    assert.equal(getLogTaskGroupId({ ...complete, task_id: 'skill_lifecycle_review_alpha_job-1' }), '');
});

test('Plan review references normalize from both live and nested durable envelopes', () => {
    const revision = 'a'.repeat(64);
    assert.deepEqual(reviewReferenceFromRow({
        type: 'review_reference', surface: 'plan_review', task_id: 'root', state_revision: revision,
    }), {
        surface: 'plan_review', presentationOwnerTaskId: 'root',
        stateRevision: revision, reviewFingerprint: '',
    });
    assert.deepEqual(reviewReferenceFromRow({
        task_id: 'outer', progress_meta: { review_reference: {
            surface: 'plan_review', presentation_owner_task_id: 'root', state_revision: revision,
            review_fingerprint: 'fingerprint',
        } },
    }), {
        surface: 'plan_review', presentationOwnerTaskId: 'root',
        stateRevision: revision, reviewFingerprint: 'fingerprint',
    });
});

test('live lifecycle ignores its synthetic outer task id and updates the same group', () => {
    const live = reviewGroupFromLifecycle({
        task_id: 'skill_lifecycle_review_alpha_job-2',
        progress_meta: { lifecycle: {
            kind: 'review', status: 'running', target: 'alpha', id: 'job-2',
            group_id: 'task:root:alpha', presentation_owner_task_id: 'root',
            initiator_task_id: 'initiator-child', snapshot_revised: true,
            replayed_from_ts: '2026-08-24T00:00:00Z',
        } },
    });
    assert.equal(live.presentationOwnerTaskId, 'root');
    assert.equal(live.initiatorTaskId, 'initiator-child');
    assert.equal(live.activeCount, 1);
    assert.equal(live.attempts[0].id, 'job-2');
    assert.equal(live.attempts[0].revised, true);
    assert.equal(live.attempts[0].replayed, true);
});

test('lifecycle provenance ignores a synthetic outer task id but preserves an explicit origin', () => {
    const lifecycle = {
        kind: 'review', status: 'running', target: 'alpha', id: 'job-2',
        group_id: 'task:root:alpha', presentation_owner_task_id: 'root',
    };
    assert.equal(reviewGroupFromLifecycle({
        task_id: 'skill_lifecycle_review_alpha_job-2', lifecycle,
    }).initiatorTaskId, '');
    assert.equal(reviewGroupFromLifecycle({
        task_id: 'skill_lifecycle_review_alpha_job-2', origin_task_id: 'real-initiator', lifecycle,
    }).initiatorTaskId, 'real-initiator');
});

test('Logs groups review lifecycle only under its explicit presentation owner', () => {
    const incomplete = {
        type: 'task_progress', is_progress: true, task_id: 'skill_lifecycle_review_alpha_manual',
        lifecycle: { kind: 'review', status: 'running', target: 'alpha', job_id: 'manual' },
    };
    assert.equal(getLogTaskGroupId(incomplete), '');
    assert.equal(isGroupedTaskEvent(incomplete), false);

    const complete = {
        ...incomplete,
        lifecycle: {
            ...incomplete.lifecycle,
            group_id: 'task:root:alpha', presentation_owner_task_id: 'root',
        },
    };
    assert.equal(getLogTaskGroupId(complete), 'root');
    assert.equal(isGroupedTaskEvent(complete), true);

    const ordinary = { type: 'task_progress', is_progress: true, task_id: 'ordinary-task' };
    assert.equal(getLogTaskGroupId(ordinary), 'ordinary-task');
    assert.equal(isGroupedTaskEvent(ordinary), true);
});

test('group reducer keeps one row and ordered projected attempts across live to terminal', () => {
    const store = new Map();
    const live = reviewGroupFromLifecycle({ lifecycle: {
        kind: 'review', status: 'running', target: 'alpha', job_id: 'job-2',
        group_id: 'task:root:alpha', presentation_owner_task_id: 'root',
    } });
    mergeReviewGroup(store, live);
    mergeReviewGroup(store, reviewGroupFromHistoryRow(groupedSkillRow()));
    assert.equal(store.size, 1);
    const settled = store.get('task:root:alpha');
    assert.deepEqual(settled.attempts.map((attempt) => attempt.id), ['job-1', 'job-2']);
    assert.equal(settled.activeCount, 0);
    assert.equal(settled.state, 'terminal');
});

test('terminal attempt state is monotonic while a genuinely new attempt restores liveness', () => {
    const store = new Map();
    mergeReviewGroup(store, reviewGroupFromHistoryRow(groupedSkillRow()));

    mergeReviewGroup(store, reviewGroupFromLifecycle({ lifecycle: {
        kind: 'review', status: 'running', target: 'alpha', job_id: 'job-2',
        group_id: 'task:root:alpha', presentation_owner_task_id: 'root',
    } }));
    const stale = store.get('task:root:alpha');
    assert.equal(stale.state, 'terminal');
    assert.equal(stale.activeCount, 0);
    assert.equal(stale.verdict, 'clean');
    assert.equal(stale.tone, 'done');
    assert.equal(stale.attempts.find((attempt) => attempt.id === 'job-2')?.state, 'terminal');

    mergeReviewGroup(store, reviewGroupFromLifecycle({ lifecycle: {
        kind: 'review', status: 'running', target: 'alpha', job_id: 'job-3',
        group_id: 'task:root:alpha', presentation_owner_task_id: 'root',
    } }));
    const next = store.get('task:root:alpha');
    assert.equal(next.state, 'running');
    assert.equal(next.activeCount, 1);
    assert.equal(next.attempts.find((attempt) => attempt.id === 'job-3')?.state, 'running');
});

test('a stale partial projection cannot hide a newer active attempt it omits', () => {
    const store = new Map();
    mergeReviewGroup(store, reviewGroupFromHistoryRow(groupedSkillRow()));
    mergeReviewGroup(store, reviewGroupFromLifecycle({ lifecycle: {
        kind: 'review', status: 'running', target: 'alpha', job_id: 'job-3',
        group_id: 'task:root:alpha', presentation_owner_task_id: 'root',
    } }));
    mergeReviewGroup(store, reviewGroupFromHistoryRow(groupedSkillRow({
        status: 'clean',
        attempts: [{ job_id: 'job-1', skill: 'alpha', status: 'blockers' }],
    })));
    const merged = store.get('task:root:alpha');
    assert.equal(merged.state, 'running');
    assert.equal(merged.activeCount, 1);
    assert.deepEqual(merged.attempts.map((attempt) => attempt.id), ['job-1', 'job-2', 'job-3']);
    assert.equal(merged.attempts.find((attempt) => attempt.id === 'job-3')?.state, 'running');
});

test('an unmatched open Plan attempt restores liveness before its first wave lands', () => {
    const settledFingerprint = 'a'.repeat(64);
    const nextFingerprint = 'b'.repeat(64);
    const wave = {
        request_fingerprint: settledFingerprint,
        cycle_index: 1,
        aggregate: 'GREEN',
        closed: true,
    };
    const store = new Map();
    mergeReviewGroup(store, planReviewGroupFromTaskDetail({
        task_id: 'root',
        plan_review_state: {
            current_attempt: { fingerprint: settledFingerprint, status: 'closed' },
            waves: [wave],
        },
    }));

    mergeReviewGroup(store, planReviewGroupFromTaskDetail({
        task_id: 'root',
        plan_review_state: {
            current_attempt: { fingerprint: nextFingerprint, status: 'open' },
            waves: [wave],
        },
    }));

    const next = store.get('plan:root');
    assert.equal(next.state, 'running');
    assert.equal(next.activeCount, 1);
    assert.equal(next.verdict, 'open');
    assert.equal(next.tone, 'working');
    assert.equal(next.attempts.at(-1).id, nextFingerprint);
    assert.equal(next.attempts.at(-1).state, 'running');
    assert.equal(next.attempts.at(-1).label, 'current attempt');
});

test('Plan waves retain the canonical reviewed timestamp through full and compact projections', () => {
    const fingerprint = 'a'.repeat(64);
    const reviewedAt = '2026-08-29T01:02:03+00:00';
    const project = (compact) => planReviewGroupFromTaskDetail({
        task_id: 'root',
        plan_review_state: {
            schema_version: 2,
            current_attempt: compact ? {} : { fingerprint, status: 'closed' },
            waves: [{
                compact,
                request_fingerprint: fingerprint,
                cycle_index: 1,
                aggregate: 'GREEN',
                closed: true,
                reviewed_at: reviewedAt,
            }],
            waves_omitted: 0,
        },
    });

    assert.equal(project(false).attempts[0].timestamp, reviewedAt);
    assert.equal(project(true).attempts[0].timestamp, reviewedAt);
    assert.match(renderReviewsSection([project(false)], {
        sectionExpanded: true,
        expandedGroups: new Set(['plan:root']),
    }), new RegExp(reviewedAt.replaceAll('+', '\\+')));
});

test('a fresh Plan attempt replaces matching compact history until its full wave lands', () => {
    const firstFingerprint = 'a'.repeat(64);
    const secondFingerprint = 'b'.repeat(64);
    const compactFirst = {
        compact: true,
        request_fingerprint: firstFingerprint,
        cycle_index: 1,
        aggregate: 'REVISE_PLAN',
        closed: false,
    };
    const fullSecond = {
        request_fingerprint: secondFingerprint,
        cycle_index: 2,
        aggregate: 'GREEN',
        closed: true,
    };
    const store = new Map();
    mergeReviewGroup(store, planReviewGroupFromTaskDetail({
        task_id: 'root',
        plan_review_state: {
            current_attempt: { fingerprint: secondFingerprint, status: 'closed' },
            waves: [compactFirst, fullSecond],
        },
    }));

    const activeProjection = planReviewGroupFromTaskDetail({
        task_id: 'root',
        plan_review_state: {
            current_attempt: { fingerprint: firstFingerprint, status: 'open' },
            waves: [compactFirst, fullSecond],
        },
    });
    assert.equal(activeProjection.state, 'running');
    assert.equal(activeProjection.activeCount, 1);
    assert.equal(activeProjection.verdict, 'open');
    assert.deepEqual(activeProjection.attempts.map((attempt) => attempt.id), [
        secondFingerprint, firstFingerprint,
    ]);
    assert.equal(activeProjection.attempts.at(-1).label, 'current attempt');
    assert.equal(activeProjection.attempts.at(-1).compact, false);

    mergeReviewGroup(store, activeProjection);
    const active = store.get('plan:root');
    assert.equal(active.state, 'running');
    assert.equal(active.activeCount, 1);
    assert.equal(active.verdict, 'open');
    assert.deepEqual(active.attempts.map((attempt) => attempt.id), [
        secondFingerprint, firstFingerprint,
    ]);
    assert.equal(active.attempts.at(-1).state, 'running');

    mergeReviewGroup(store, planReviewGroupFromTaskDetail({
        task_id: 'root',
        plan_review_state: {
            current_attempt: { fingerprint: firstFingerprint, status: 'open' },
            waves: [fullSecond, {
                request_fingerprint: firstFingerprint,
                cycle_index: 3,
                aggregate: 'REVIEW_REQUIRED',
                closed: false,
            }],
        },
    }));
    const terminal = store.get('plan:root');
    assert.equal(terminal.state, 'terminal');
    assert.equal(terminal.activeCount, 0);
    assert.equal(terminal.verdict, 'REVIEW_REQUIRED');
    assert.deepEqual(terminal.attempts.map((attempt) => attempt.id), [
        secondFingerprint, firstFingerprint,
    ]);
    assert.equal(terminal.attempts.at(-1).state, 'terminal');
    assert.equal(terminal.attempts.at(-1).compact, false);
});

test('terminal Plan controls retain an open compact wave but never revive closed compact authority', () => {
    const fingerprint = 'a'.repeat(64);
    const compactOpen = {
        compact: true, request_fingerprint: fingerprint, cycle_index: 1,
        aggregate: 'REVIEW_REQUIRED', closed: false, cycles_exhausted: true,
    };
    for (const status of ['rail_degraded', 'cycles_exhausted', 'unavailable']) {
        const wave = status === 'unavailable'
            ? { ...compactOpen, cycles_exhausted: false }
            : compactOpen;
        const group = planReviewGroupFromTaskDetail({
            task_id: 'root',
            plan_review_state: {
                current_attempt: { fingerprint, status, reason: status },
                waves: [wave],
            },
        });
        assert.equal(group.attempts.length, 1, status);
        assert.equal(group.attempts[0].compact, true, status);
        assert.equal(group.attempts[0].verdict, 'REVIEW_REQUIRED', status);
        assert.match(group.attempts[0].detailText, /Verdict: REVIEW_REQUIRED/, status);
        assert.doesNotMatch(group.attempts[0].detailText, /Review result unavailable/, status);
        assert.equal(
            group.verdict,
            status === 'rail_degraded' ? 'rail_degraded'
                : (status === 'cycles_exhausted' ? 'cycles_exhausted' : 'REVIEW_REQUIRED'),
            status,
        );
    }
    const closed = planReviewGroupFromTaskDetail({
        task_id: 'root',
        plan_review_state: {
            current_attempt: { fingerprint, status: 'rail_degraded', reason: 'fresh rail' },
            waves: [{ ...compactOpen, aggregate: 'GREEN', closed: true }],
        },
    });
    assert.equal(closed.verdict, 'rail_degraded');
    assert.equal(closed.attempts[0].compact, false);
    assert.equal(closed.attempts[0].verdict, 'rail_degraded');
});

test('legacy Plan state uses the backend-derived compatibility projection', () => {
    const fingerprint = 'a'.repeat(64);
    const project = (legacy_v1_projection) => planReviewGroupFromTaskDetail({
        task_id: 'legacy-root',
        plan_review_state: { schema_version: 1, legacy_v1_projection },
    });
    const reviewedOpen = project({
        fingerprint, status: 'open', outcome: 'REVIEW_REQUIRED', closed: false, reason: '',
    });
    assert.equal(reviewedOpen.state, 'terminal');
    assert.equal(reviewedOpen.verdict, 'REVIEW_REQUIRED');
    assert.equal(reviewedOpen.attempts[0].verdict, 'REVIEW_REQUIRED');
    assert.equal(reviewedOpen.countIsAuthoritative, false);
    assert.deepEqual(reviewedOpen.attempts[0].executions, []);
    assert.match(reviewedOpen.attempts[0].detailText, /Cost unavailable/);

    const pending = project({ fingerprint: '', status: 'pending', outcome: '', closed: false });
    assert.equal(pending.state, 'queued');
    assert.equal(pending.activeCount, 1);
    assert.equal(pending.attempts.length, 0);

    const unavailable = project({
        fingerprint, status: 'open', outcome: '', closed: false, reason: 'reviewer unavailable',
    });
    assert.equal(unavailable.state, 'unavailable');
    assert.equal(unavailable.verdict, 'open');
    assert.equal(unavailable.tone, 'neutral');
    assert.equal(unavailable.summary, 'reviewer unavailable');

    const rail = project({
        fingerprint, status: 'rail_degraded', outcome: 'REVIEW_REQUIRED', closed: false,
        reason: 'deadline',
    });
    assert.equal(rail.verdict, 'rail_degraded');
    assert.equal(rail.attempts[0].verdict, 'REVIEW_REQUIRED');

    const closed = project({
        fingerprint, status: 'closed', outcome: 'GREEN', closed: true, reason: '',
    });
    assert.equal(closed.state, 'terminal');
    assert.equal(closed.verdict, 'GREEN');
    assert.equal(closed.attempts[0].verdict, 'GREEN');

    assert.equal(planReviewGroupFromTaskDetail({
        task_id: 'legacy-root',
        plan_review_state: {
            schema_version: 1,
            current_attempt: { fingerprint, status: 'open' },
            waves: [{ request_fingerprint: fingerprint, review: { aggregate_signal: 'GREEN' } }],
        },
    }), null, 'raw v1 never falls through the v2 parser');
});

test('typed Plan rail degradation controls the group without erasing open wave evidence', () => {
    const fingerprint = 'a'.repeat(64);
    const openWave = {
        request_fingerprint: fingerprint,
        cycle_index: 1,
        aggregate: 'REVIEW_REQUIRED',
        closed: false,
    };
    const openDetail = {
        task_id: 'root',
        plan_review_state: {
            current_attempt: { fingerprint, status: 'open' },
            waves: [openWave],
        },
    };
    const initial = planReviewGroupFromTaskDetail(openDetail);
    assert.equal(initial.verdict, 'REVIEW_REQUIRED');
    assert.equal(initial.attempts.length, 1);

    const degradedDetail = {
        task_id: 'root',
        plan_review_state: {
            current_attempt: {
                fingerprint,
                status: 'rail_degraded',
                reason: 'plan_task_deadline',
            },
            waves: [openWave],
        },
    };
    const degraded = planReviewGroupFromTaskDetail(degradedDetail);
    assert.equal(degraded.state, 'terminal');
    assert.equal(degraded.activeCount, 0);
    assert.equal(degraded.verdict, 'rail_degraded');
    assert.equal(degraded.summary, 'plan_task_deadline');
    assert.equal(degraded.attempts.length, 1);
    assert.equal(degraded.attempts[0].id, fingerprint);
    assert.equal(degraded.attempts[0].verdict, 'REVIEW_REQUIRED');

    const store = new Map();
    mergeReviewGroup(store, initial);
    mergeReviewGroup(store, degraded);
    const merged = store.get('plan:root');
    assert.equal(merged.state, 'terminal');
    assert.equal(merged.activeCount, 0);
    assert.equal(merged.verdict, 'rail_degraded');
    assert.equal(merged.summary, 'plan_task_deadline');
    assert.equal(merged.attempts.length, 1);
    assert.equal(merged.attempts[0].verdict, 'REVIEW_REQUIRED');

    const closed = planReviewGroupFromTaskDetail({
        task_id: 'root',
        plan_review_state: {
            current_attempt: { fingerprint, status: 'rail_degraded' },
            waves: [{ ...openWave, aggregate: 'GREEN', closed: true }],
        },
    });
    assert.equal(closed.verdict, 'GREEN');
});

test('a newer canonical Plan attempt retires prior unmatched liveness', () => {
    const settledFingerprint = 'a'.repeat(64);
    const firstOpen = 'b'.repeat(64);
    const secondOpen = 'c'.repeat(64);
    const settledWave = {
        request_fingerprint: settledFingerprint,
        cycle_index: 1,
        aggregate: 'GREEN',
        closed: true,
    };
    const store = new Map();
    mergeReviewGroup(store, planReviewGroupFromTaskDetail({
        task_id: 'root',
        plan_review_state: {
            current_attempt: { fingerprint: settledFingerprint, status: 'closed' },
            waves: [settledWave],
        },
    }));
    for (const fingerprint of [firstOpen, secondOpen]) {
        mergeReviewGroup(store, planReviewGroupFromTaskDetail({
            task_id: 'root',
            plan_review_state: {
                current_attempt: { fingerprint, status: 'open', reason: `attempt ${fingerprint[0]}` },
                waves: [settledWave],
            },
        }));
    }

    const running = store.get('plan:root');
    assert.equal(running.state, 'running');
    assert.equal(running.activeCount, 1);
    assert.deepEqual(running.attempts.map((attempt) => attempt.id), [
        settledFingerprint, firstOpen, secondOpen,
    ]);
    assert.equal(running.attempts.find((attempt) => attempt.id === firstOpen)?.state, 'superseded');
    assert.equal(running.attempts.find((attempt) => attempt.id === secondOpen)?.state, 'running');

    const terminalWave = {
        request_fingerprint: secondOpen,
        cycle_index: 2,
        aggregate: 'GREEN',
        closed: true,
    };
    mergeReviewGroup(store, planReviewGroupFromTaskDetail({
        task_id: 'root',
        plan_review_state: {
            current_attempt: { fingerprint: secondOpen, status: 'closed' },
            waves: [settledWave, terminalWave],
        },
    }));

    const terminal = store.get('plan:root');
    assert.equal(terminal.state, 'terminal');
    assert.equal(terminal.activeCount, 0);
    assert.equal(terminal.verdict, 'GREEN');
    assert.deepEqual(terminal.attempts.map((attempt) => attempt.id), [
        settledFingerprint, firstOpen, secondOpen,
    ]);
    assert.equal(terminal.attempts.find((attempt) => attempt.id === firstOpen)?.state, 'superseded');
    assert.equal(terminal.attempts.find((attempt) => attempt.id === secondOpen)?.state, 'terminal');
});

test('unmatched terminal Plan states remain inspectable attempts', () => {
    for (const status of ['unavailable', 'rail_degraded', 'cycles_exhausted']) {
        const fingerprint = status[0].repeat(64);
        const group = planReviewGroupFromTaskDetail({
            task_id: 'root',
            plan_review_state: {
                current_attempt: { fingerprint, status, reason: `typed ${status} reason` },
                waves: [],
            },
        });
        assert.equal(group.activeCount, 0, status);
        assert.equal(group.attempts.length, 1, status);
        assert.equal(group.attempts[0].id, fingerprint, status);
        assert.equal(group.attempts[0].state, status === 'unavailable' ? 'unavailable' : 'terminal', status);
        assert.match(group.attempts[0].detailText, new RegExp(`typed ${status} reason`));
        assert.match(group.attempts[0].detailText, /Cost unavailable/);
    }
});

test('plan review retains current and superseded waves without inventing authority', () => {
    const detail = {
        task_id: 'root',
        plan_review_state: {
            current_attempt: { fingerprint: 'new', status: 'open', reason: '' },
            waves_omitted: 0,
            waves: [
                { request_fingerprint: 'old', cycle_index: 1, aggregate: 'GREEN', closed: true },
                {
                    request_fingerprint: 'new', cycle_index: 2,
                    aggregate: 'REVIEW_REQUIRED', closed: false,
                    actors: [{ executions: [
                        { kind: 'harness', harness_id: 'cursor', model: 'cursor-grok-4.6-high' },
                    ] }],
                },
                { request_fingerprint: 'cached', cycle_index: 3, aggregate: 'GREEN', closed: true },
            ],
        },
    };
    const group = planReviewGroupFromTaskDetail(detail);
    assert.equal(group.presentationOwnerTaskId, 'root');
    assert.equal(group.state, 'terminal');
    assert.equal(group.activeCount, 0);
    assert.equal(group.attempts[0].superseded, true);
    assert.equal(group.attempts[1].superseded, false);
    assert.equal(group.attempts[1].verdict, 'REVIEW_REQUIRED');
    assert.deepEqual(group.attempts[1].executions, [
        { kind: 'harness', harness_id: 'cursor', model: 'cursor-grok-4.6-high' },
    ]);
    assert.equal(group.attempts[2].superseded, true);
    assert.equal(group.verdict, 'REVIEW_REQUIRED');
    assert.equal(group.countIsAuthoritative, true);

    delete detail.plan_review_state.waves_omitted;
    assert.equal(planReviewGroupFromTaskDetail(detail).countIsAuthoritative, false);
});

test('plan liveness comes only from an unmatched open current attempt', () => {
    const fingerprint = 'a'.repeat(64);
    const old = 'b'.repeat(64);
    const open = planReviewGroupFromTaskDetail({
        task_id: 'root',
        plan_review_state: {
            current_attempt: { fingerprint, status: 'open' },
            waves: [{ request_fingerprint: old, aggregate: 'GREEN', closed: true }],
        },
    });
    assert.equal(open.state, 'running');
    assert.equal(open.activeCount, 1);
    assert.equal(open.verdict, 'open');
    assert.equal(open.tone, 'working');
    assert.equal(open.attempts[0].superseded, true);

    for (const [status, expectedState] of [
        ['unavailable', 'unavailable'],
        ['rail_degraded', 'terminal'],
        ['cycles_exhausted', 'terminal'],
    ]) {
        const group = planReviewGroupFromTaskDetail({
            task_id: 'root',
            plan_review_state: { current_attempt: { fingerprint, status }, waves: [] },
        });
        assert.equal(group.state, expectedState);
        assert.equal(group.activeCount, 0);
        assert.equal(group.verdict, status);
    }
});

test('review tones use an explicit success allowlist', () => {
    const states = [
        ['PASS', 'done'], ['GREEN', 'done'],
        ['REVIEW_REQUIRED', 'warn'], ['REVISE_PLAN', 'warn'], ['DEGRADED', 'warn'],
        ['UNKNOWN', 'neutral'], ['transport_error', 'neutral'], ['timeout', 'error'],
    ];
    for (const [status, tone] of states) {
        const group = reviewGroupFromHistoryRow(groupedSkillRow({ status, verdict: status }));
        assert.equal(group.tone, tone, status || '(no verdict)');
    }
    const pending = reviewGroupFromLifecycle({ lifecycle: {
        kind: 'review', status: 'pending', target: 'alpha', job_id: 'job-pending',
        group_id: 'task:root:alpha', presentation_owner_task_id: 'root',
    } });
    assert.equal(pending.state, 'queued');
    assert.equal(pending.activeCount, 1);
});

test('terminal lifecycle wins over pending history after reload without inventing a verdict', () => {
    const group = reviewGroupFromHistoryRow({
        ...groupedSkillRow({
            attempts: [{
                job_id: 'pending-job',
                skill: 'alpha',
                status: 'pending',
                job_status: 'succeeded',
            }],
        }),
        status: 'pending',
        job_status: 'succeeded',
        job_id: 'pending-job',
    });
    assert.equal(group.state, 'terminal');
    assert.equal(group.tone, 'neutral');
    assert.equal(group.verdict, '');
    assert.equal(group.activeCount, 0);
    assert.equal(group.attempts[0].lifecycleOnly, true);
});

test('semantic history still wins when a successful lifecycle fact is present', () => {
    const group = reviewGroupFromHistoryRow({
        ...groupedSkillRow(),
        status: 'clean',
        job_status: 'succeeded',
    });
    assert.equal(group.state, 'terminal');
    assert.equal(group.tone, 'done');
    assert.equal(group.verdict, 'clean');
});

test('failed lifecycle stays visible beside a semantic clean verdict', () => {
    const group = reviewGroupFromHistoryRow({
        ...groupedSkillRow({
            attempts: [{
                job_id: 'deps-failed',
                skill: 'alpha',
                status: 'clean',
                review_status: 'clean',
                job_status: 'failed',
                terminal_reason: 'dependency install failed',
            }],
        }),
        status: 'clean',
        review_status: 'clean',
        job_status: 'failed',
        terminal_reason: 'dependency install failed',
        job_id: 'deps-failed',
    });
    assert.equal(group.verdict, 'clean');
    assert.equal(group.tone, 'error');
    assert.equal(group.lifecycleStatus, 'failed');
    assert.equal(group.attempts[0].verdict, 'clean');
    assert.equal(group.attempts[0].tone, 'error');
    assert.match(renderReviewsSection([group]), /lifecycle failed/);
});

test('lifecycle completion stays neutral until a semantic review verdict arrives', () => {
    const lifecycle = reviewGroupFromLifecycle({ lifecycle: {
        kind: 'review', status: 'succeeded', target: 'alpha', job_id: 'job-live',
        group_id: 'task:root:alpha', presentation_owner_task_id: 'root',
    } });
    assert.equal(lifecycle.state, 'terminal');
    assert.equal(lifecycle.tone, 'neutral');
    assert.equal(lifecycle.verdict, '');
    assert.equal(lifecycle.attempts[0].lifecycleOnly, true);
    assert.match(lifecycle.attempts[0].summary, /verdict unavailable/i);
    for (const [status, tone] of [
        ['failed', 'error'], ['cancelled', 'warn'], ['interrupted', 'warn'], ['timeout', 'error'],
    ]) {
        const terminal = reviewGroupFromLifecycle({ lifecycle: {
            kind: 'review', status, target: 'alpha', job_id: `job-${status}`,
            group_id: 'task:root:alpha', presentation_owner_task_id: 'root',
        } });
        assert.equal(terminal.verdict, '');
        assert.equal(terminal.tone, tone, status);
        assert.equal(terminal.attempts[0].tone, tone, status);
    }

    for (const status of ['failed', 'timeout']) {
        const pendingStore = new Map();
        mergeReviewGroup(pendingStore, reviewGroupFromHistoryRow({
            ...groupedSkillRow({
                status: 'pending',
                attempts: [{ job_id: 'pending-job', skill: 'alpha', status: 'pending' }],
            }),
            job_id: 'pending-job',
        }));
        const failed = mergeReviewGroup(pendingStore, reviewGroupFromLifecycle({ lifecycle: {
            kind: 'review', status, target: 'alpha', job_id: 'pending-job',
            group_id: 'task:root:alpha', presentation_owner_task_id: 'root',
        } }));
        assert.equal(failed.verdict, '');
        assert.equal(failed.tone, 'error', status);
        assert.equal(failed.attempts[0].state, 'terminal');
        assert.equal(failed.attempts[0].tone, 'error');
    }

    for (const [status, tone] of [
        ['failed', 'error'], ['error', 'error'], ['cancelled', 'warn'],
        ['interrupted', 'warn'], ['timeout', 'error'],
    ]) {
        const historical = reviewGroupFromHistoryRow({
            ...groupedSkillRow({
                attempts: [{
                    job_id: `history-${status}`,
                    skill: 'alpha',
                    status,
                    job_status: status,
                }],
            }),
            status,
            job_status: status,
            job_id: `history-${status}`,
        });
        assert.equal(historical.verdict, '', status);
        assert.equal(historical.lifecycleOnly, true, status);
        assert.equal(historical.tone, tone, status);
        assert.equal(historical.attempts[0].verdict, '', status);
        assert.equal(historical.attempts[0].lifecycleOnly, true, status);
        assert.equal(historical.attempts[0].tone, tone, status);
    }

    const store = new Map();
    const history = reviewGroupFromHistoryRow(groupedSkillRow());
    mergeReviewGroup(store, history);
    const late = reviewGroupFromLifecycle({ lifecycle: {
        kind: 'review', status: 'succeeded', target: 'alpha', job_id: 'job-2',
        group_id: 'task:root:alpha', presentation_owner_task_id: 'root',
    } });
    const merged = mergeReviewGroup(store, late);
    assert.equal(merged.verdict, 'clean');
    assert.equal(merged.tone, 'done');
    assert.equal(merged.attempts.at(-1).verdict, 'clean');
    assert.equal(merged.attempts.at(-1).tone, 'done');

    const newAttempt = reviewGroupFromLifecycle({ lifecycle: {
        kind: 'review', status: 'succeeded', target: 'alpha', job_id: 'job-3',
        group_id: 'task:root:alpha', presentation_owner_task_id: 'root',
    } });
    const newAttemptMerged = mergeReviewGroup(store, newAttempt);
    assert.equal(newAttemptMerged.verdict, '');
    assert.equal(newAttemptMerged.tone, 'neutral');
    assert.equal(newAttemptMerged.lifecycleOnly, true);
    assert.equal(newAttemptMerged.attempts.at(-1).id, 'job-3');
    assert.equal(newAttemptMerged.attempts.at(-1).verdict, '');

    const staleStore = new Map();
    mergeReviewGroup(staleStore, history);
    mergeReviewGroup(staleStore, newAttempt);
    const staleHistory = reviewGroupFromHistoryRow(groupedSkillRow({
        status: 'clean',
        attempts: [{ job_id: 'job-2', skill: 'alpha', status: 'clean' }],
    }));
    const staleMerged = mergeReviewGroup(staleStore, staleHistory);
    assert.equal(staleMerged.verdict, '');
    assert.equal(staleMerged.tone, 'neutral');
    assert.equal(staleMerged.lifecycleOnly, true);
    assert.equal(staleMerged.attempts.at(-1).id, 'job-3');
});

test('lifecycle timestamps fence delayed older history without inventing a verdict', () => {
    const lifecycle = reviewGroupFromLifecycle({
        ts: '2026-08-26T10:03:00Z',
        lifecycle: {
            kind: 'review', status: 'succeeded', target: 'alpha', job_id: 'job-3',
            group_id: 'task:root:alpha', presentation_owner_task_id: 'root',
        },
    });
    assert.equal(lifecycle.attempts[0].timestamp, '2026-08-26T10:03:00Z');

    const store = new Map();
    mergeReviewGroup(store, reviewGroupFromHistoryRow(groupedSkillRow({
        attempts: [{ job_id: 'job-1', skill: 'alpha', status: 'blockers', ts: '2026-08-26T10:00:00Z' }],
    })));
    mergeReviewGroup(store, lifecycle);
    const delayed = reviewGroupFromHistoryRow(groupedSkillRow({
        attempts: [
            { job_id: 'job-1', skill: 'alpha', status: 'blockers', ts: '2026-08-26T10:00:00Z' },
            { job_id: 'job-2', skill: 'alpha', status: 'clean', ts: '2026-08-26T10:02:00Z' },
        ],
    }));
    const merged = mergeReviewGroup(store, delayed);
    assert.deepEqual(merged.attempts.map((attempt) => attempt.id), ['job-1', 'job-2', 'job-3']);
    assert.equal(merged.lifecycleOnly, true);
    assert.equal(merged.verdict, '');
    assert.equal(merged.tone, 'neutral');

    const newer = reviewGroupFromHistoryRow(groupedSkillRow({
        attempts: [{ job_id: 'job-4', skill: 'alpha', status: 'clean', ts: '2026-08-26T10:04:00Z' }],
    }));
    const upgraded = mergeReviewGroup(store, newer);
    assert.equal(upgraded.attempts.at(-1).id, 'job-4');
    assert.equal(upgraded.verdict, 'clean');
    assert.equal(upgraded.tone, 'done');

    const mixed = reviewGroupFromHistoryRow(groupedSkillRow({
        attempts: [
            { job_id: 'job-2', skill: 'alpha', status: 'clean', ts: '2026-08-26T10:02:00Z' },
            { job_id: 'job-4', skill: 'alpha', status: 'clean', ts: '2026-08-26T10:04:00Z' },
        ],
    }));
    const mixedStore = new Map();
    mergeReviewGroup(mixedStore, reviewGroupFromHistoryRow(groupedSkillRow({
        attempts: [{ job_id: 'job-1', skill: 'alpha', status: 'blockers', ts: '2026-08-26T10:00:00Z' }],
    })));
    mergeReviewGroup(mixedStore, lifecycle);
    const mixedMerged = mergeReviewGroup(mixedStore, mixed);
    assert.deepEqual(mixedMerged.attempts.map((attempt) => attempt.id), ['job-1', 'job-2', 'job-3', 'job-4']);
    assert.equal(mixedMerged.verdict, 'clean');
    assert.equal(mixedMerged.tone, 'done');

    const semanticHistory = reviewGroupFromHistoryRow(groupedSkillRow({
        attempts: [
            { job_id: 'job-1', skill: 'alpha', status: 'clean', ts: '2026-08-26T10:00:00Z' },
            { job_id: 'job-2', skill: 'alpha', status: 'blockers', ts: '2026-08-26T10:02:00Z' },
        ],
    }));
    const lateLifecycle = reviewGroupFromLifecycle({
        lifecycle: {
            kind: 'review', status: 'succeeded', target: 'alpha', job_id: 'job-2',
            finished_at: '2026-08-26T10:05:00Z',
            group_id: 'task:root:alpha', presentation_owner_task_id: 'root',
        },
    });
    const sameAttemptStore = new Map();
    mergeReviewGroup(sameAttemptStore, semanticHistory);
    mergeReviewGroup(sameAttemptStore, lateLifecycle);
    const afterLateLifecycle = sameAttemptStore.get('task:root:alpha');
    assert.equal(afterLateLifecycle.attempts.find((attempt) => attempt.id === 'job-2').timestamp, '2026-08-26T10:02:00Z');
    const newerHistory = reviewGroupFromHistoryRow(groupedSkillRow({
        attempts: [{ job_id: 'job-3', skill: 'alpha', status: 'clean', ts: '2026-08-26T10:04:00Z' }],
    }));
    const afterNewerHistory = mergeReviewGroup(sameAttemptStore, newerHistory);
    assert.deepEqual(afterNewerHistory.attempts.map((attempt) => attempt.id), ['job-1', 'job-2', 'job-3']);

    const semanticStaleStore = new Map();
    mergeReviewGroup(semanticStaleStore, reviewGroupFromHistoryRow(groupedSkillRow({
        attempts: [{ job_id: 'job-3', skill: 'alpha', status: 'blockers', ts: '2026-08-26T10:03:00Z' }],
    })));
    const semanticStale = mergeReviewGroup(semanticStaleStore, reviewGroupFromHistoryRow(groupedSkillRow({
        status: 'clean',
        attempts: [{ job_id: 'job-2', skill: 'alpha', status: 'clean', ts: '2026-08-26T10:02:00Z' }],
    })));
    assert.deepEqual(semanticStale.attempts.map((attempt) => attempt.id), ['job-2', 'job-3']);
    assert.equal(semanticStale.verdict, 'blockers');
    assert.equal(semanticStale.tone, 'error');

    const pointer = classifyReviewLifecyclePointer({
        ts: '2026-08-26T11:00:00Z',
        lifecycle_pointer: {
            kind: 'review', status: 'running', target: 'alpha', job_id: 'pointer-1',
            group_id: 'task:root:alpha', presentation_owner_task_id: 'root',
        },
    });
    assert.equal(pointer.group.attempts[0].timestamp, '2026-08-26T11:00:00Z');
});

test('task acceptance adapts only task_acceptance panels; advisory and commit stay omitted', () => {
    const detail = {
        task_id: 'root',
        review_projection: { panels: [
            { panel_id: 'accept', surface: 'task_acceptance', aggregate_signal: 'PASS', actors: [{
                executions: [{ kind: 'harness', harness_id: 'claude', model: 'claude-fable-5' }],
            }] },
            { panel_id: 'commit', surface: 'commit', aggregate_signal: 'PASS', actors: [] },
            { panel_id: 'advisory', surface: 'advisory', aggregate_signal: 'PASS', actors: [] },
        ] },
    };
    const group = taskAcceptanceGroupFromTaskDetail(detail);
    assert.deepEqual(group.attempts.map((attempt) => attempt.id), ['accept']);
    assert.deepEqual(group.attempts[0].executions, [
        { kind: 'harness', harness_id: 'claude', model: 'claude-fable-5' },
    ]);
    assert.deepEqual(reviewGroupsFromTaskDetail(detail).map((item) => item.surface), ['task_acceptance']);
});

test('renderer is quiet, accessible and never invents review dollars', () => {
    const group = reviewGroupFromHistoryRow(groupedSkillRow());
    const html = renderReviewsSection([group], {
        sectionExpanded: true,
        expandedGroups: new Set([group.id]),
        expandedAttempts: new Set([`${group.id}:job-1`]),
    });
    assert.match(html, /<section class="chat-live-reviews"/);
    assert.match(html, /data-review-section-toggle aria-expanded="true"/);
    assert.match(html, /data-review-group-toggle="task:root:alpha" aria-expanded="true"/);
    assert.match(html, /data-skill-review-job="job-1"/);
    assert.match(html, /Initiated by task initiator-child/);
    assert.match(html, /data-review-attempt-detail="task:root:alpha:job-1"[^>]*aria-busy="false"/);
    assert.match(html, /2 shown/);
    assert.doesNotMatch(html, /\$\d|cost=/i);
});

test('attempt marks require explicit executed receipts, render every production execution, and keep intent separate', () => {
    const executed = {
        executed: { kind: 'agent_session', harness: 'claude', model: 'claude-fable-5' },
        requested: { kind: 'agent_session', harness: 'cursor' },
    };
    assert.deepEqual(reviewExecutionEvidence(executed), {
        harness: 'claude', channel: '', label: '', model: 'claude-fable-5',
    });
    assert.equal(reviewExecutionEvidence({ requested: { harness: 'claude' } }), null);
    assert.deepEqual(reviewExecutionEvidenceList([
        { kind: 'harness', harness_id: 'claude', model: 'claude-fable-5' },
        { kind: 'harness', harness_id: 'cursor', model: 'cursor-grok-4.6-high' },
        { kind: 'api', model: 'openai/gpt-5.6-sol' },
    ]).map((item) => [item.harness, item.model]), [
        ['claude', 'claude-fable-5'],
        ['cursor', 'cursor-grok-4.6-high'],
        ['api', 'openai/gpt-5.6-sol'],
    ]);

    const group = reviewGroupFromHistoryRow(groupedSkillRow({
        attempts: [{
            job_id: 'job-executed', skill: 'alpha', status: 'clean', executions: [
                { kind: 'harness', harness_id: 'claude', model: 'claude-fable-5' },
                { kind: 'harness', harness_id: 'cursor', model: 'cursor-grok-4.6-high' },
                { kind: 'api', model: 'openai/gpt-5.6-sol' },
            ],
        }, {
            job_id: 'job-requested', skill: 'alpha', status: 'clean',
            execution: { requested: { kind: 'agent_session', harness: 'cursor' } },
        }],
    }));
    const html = renderReviewsSection([group], {
        sectionExpanded: true,
        expandedGroups: new Set([group.id]),
    });
    assert.match(html, /data-harness-identity="claude"/);
    assert.match(html, /Claude Code/);
    assert.match(html, /claude-fable-5/);
    assert.match(html, /data-harness-identity="cursor"/);
    assert.match(html, /cursor-grok-4\.6-high/);
    assert.match(html, /data-harness-identity="api"/);
    const requestedAttempt = html.slice(html.indexOf('data-review-attempt="task:root:alpha:job-requested"'));
    assert.doesNotMatch(requestedAttempt, /data-harness-identity="cursor"/);

    const api = reviewExecutionEvidence({ executed: { kind: 'api_chat', model: 'openai\/gpt' } });
    assert.deepEqual(api, { harness: 'api', channel: 'api', label: '', model: 'openai\/gpt' });
    // A native tool-round episode renders as the API channel with its delivery named — never null.
    const native = reviewExecutionEvidence({ kind: 'native', model: 'openai/gpt' });
    assert.deepEqual(native, { harness: 'api', channel: 'api', label: 'API · native tool rounds', model: 'openai/gpt' });
    assert.equal(reviewExecutionEvidenceList([
        { kind: 'native', model: 'openai/gpt' }, { kind: 'api', model: 'openai/gpt' },
    ]).length, 2);
});

test('attempt provenance stays per attempt and is promoted to group only when uniform', () => {
    const mixed = reviewGroupFromHistoryRow(groupedSkillRow({ attempts: [
        { job_id: 'job-1', skill: 'alpha', status: 'blockers', initiator_task_id: 'child-a' },
        { job_id: 'job-2', skill: 'alpha', status: 'clean', initiator_task_id: 'child-b' },
    ] }));
    assert.equal(mixed.initiatorTaskId, '');
    const mixedHtml = renderReviewsSection([mixed], {
        sectionExpanded: true,
        expandedGroups: new Set([mixed.id]),
    });
    assert.match(mixedHtml, /Initiated by task child-a/);
    assert.match(mixedHtml, /Initiated by task child-b/);

    const uniform = reviewGroupFromHistoryRow(groupedSkillRow({ attempts: [
        { job_id: 'job-1', skill: 'alpha', status: 'blockers', initiator_task_id: 'child-a' },
        { job_id: 'job-2', skill: 'alpha', status: 'clean', initiator_task_id: 'child-a' },
    ] }));
    assert.equal(uniform.initiatorTaskId, 'child-a');
    const uniformHtml = renderReviewsSection([uniform], {
        sectionExpanded: true,
        expandedGroups: new Set([uniform.id]),
    });
    assert.equal((uniformHtml.match(/Initiated by task child-a/g) || []).length, 1);
});

test('expanded non-Skill attempts disclose unavailable cost while compact rows stay dollar-free', () => {
    const fingerprint = 'c'.repeat(64);
    const plan = planReviewGroupFromTaskDetail({
        task_id: 'root',
        plan_review_state: {
            current_attempt: { fingerprint, status: 'closed' },
            waves: [{ request_fingerprint: fingerprint, aggregate: 'GREEN', closed: true }],
        },
    });
    const attemptKey = `${plan.id}:${plan.attempts[0].id}`;
    const html = renderReviewsSection([plan], {
        sectionExpanded: true,
        expandedGroups: new Set([plan.id]),
        expandedAttempts: new Set([attemptKey]),
    });
    assert.match(html, /chat-review-group-cost">Cost unavailable/);
    assert.match(html, /Cost unavailable/);
    const compact = renderReviewsSection([plan], {});
    const compactRow = compact.slice(
        compact.indexOf('data-review-group-toggle'),
        compact.indexOf('<div class="chat-review-attempts"'),
    );
    assert.doesNotMatch(compactRow, /Cost unavailable|\$\d/);
});

test('initiator detail is omitted when it is the owner', () => {
    const row = groupedSkillRow({ initiator_task_id: 'root' });
    assert.doesNotMatch(renderReviewsSection([reviewGroupFromHistoryRow(row)], {
        sectionExpanded: true,
        expandedGroups: new Set(['task:root:alpha']),
    }), /Initiated by task/);
});

test('review updates never change owner disclosure state', () => {
    const host = { innerHTML: '', addEventListener() {} };
    const summary = { hidden: true, textContent: '' };
    const disclosure = { sectionExpanded: false, expandedGroups: new Set(), expandedAttempts: new Set() };
    let domWrites = 0;
    const controller = createReviewPresentationController({
        host, summary, disclosure,
        onDomWrite(mutate) { domWrites += 1; return mutate(); },
    });
    controller.update(reviewGroupFromLifecycle({ lifecycle: {
        kind: 'review', status: 'running', target: 'alpha', job_id: 'job-1',
        group_id: 'task:root:alpha', presentation_owner_task_id: 'root',
    } }));
    controller.update(reviewGroupFromHistoryRow(groupedSkillRow()));
    assert.equal(disclosure.sectionExpanded, false);
    assert.deepEqual([...disclosure.expandedGroups], []);
    assert.deepEqual([...disclosure.expandedAttempts], []);
    assert.equal(domWrites, 2, 'each actual review reconcile enters the pre-write boundary');
});

test('an open exact Skill detail survives a review re-render while its read is in flight', async () => {
    const detailStore = new Map();
    const details = [];
    const loads = [];
    let resolveFetch;
    let fetches = 0;
    const fetchGate = new Promise((resolve) => { resolveFetch = resolve; });
    const host = {
        ownerDocument: { activeElement: null },
        addEventListener() {},
        contains: () => false,
        querySelector: () => null,
        querySelectorAll(selector) {
            return selector === '[data-review-attempt-detail]' && details.length
                ? [details.at(-1)] : [];
        },
        set innerHTML(value) {
            this._html = value;
            if (!value.includes('data-skill-review-job="job-1"')) return;
            details.push({
                dataset: {
                    reviewAttemptDetail: 'task:root:alpha:job-1',
                    skillReviewSkill: 'alpha',
                    skillReviewJob: 'job-1',
                },
                hidden: false,
                innerHTML: '',
                attrs: {},
                setAttribute(key, next) { this.attrs[key] = next; },
            });
        },
        get innerHTML() { return this._html || ''; },
    };
    const disclosure = {
        sectionExpanded: true,
        expandedGroups: new Set(['task:root:alpha']),
        expandedAttempts: new Set(['task:root:alpha:job-1']),
    };
    const controller = createReviewPresentationController({
        host,
        summary: { hidden: true, textContent: '' },
        disclosure,
        onLoadSkillDetail(detail) {
            loads.push(loadSkillReviewDetail(detail, {
                skill: detail.dataset.skillReviewSkill,
                jobId: detail.dataset.skillReviewJob,
            }, {
                store: detailStore,
                fetchImpl: async () => {
                    fetches += 1;
                    await fetchGate;
                    return { ok: true, json: async () => ({ markdown: 'exact detail' }) };
                },
                render: (markdown) => markdown,
            }));
        },
    });
    controller.update(reviewGroupFromHistoryRow(groupedSkillRow()));
    await Promise.resolve();
    const firstDetail = details.at(-1);
    controller.update(reviewGroupFromHistoryRow(groupedSkillRow({ status: 'warnings' })));
    const rebuiltDetail = details.at(-1);
    assert.notEqual(rebuiltDetail, firstDetail);
    assert.equal(fetches, 1);
    assert.equal(rebuiltDetail.dataset.state, 'loading');
    resolveFetch();
    await Promise.all(loads);
    assert.equal(rebuiltDetail.dataset.state, 'loaded');
    assert.match(rebuiltDetail.innerHTML, /exact detail/);
});

test('typed invalidations dedupe revisions and guarantee one trailing refresh', async () => {
    const reads = [];
    const applied = [];
    const deferred = () => {
        let resolve;
        const promise = new Promise((done) => { resolve = done; });
        return { promise, resolve };
    };
    const hydrator = createReviewHydrator({
        fetchDetail(taskId) {
            const gate = deferred();
            reads.push({ taskId, gate });
            return gate.promise;
        },
        applyDetail(_taskId, detail) {
            applied.push(detail.revision);
            return true;
        },
    });

    const firstRevision = 'a'.repeat(64);
    const secondRevision = 'b'.repeat(64);
    const first = hydrator.hydrate('root', firstRevision);
    await Promise.resolve();
    const duplicate = hydrator.hydrate('root', firstRevision);
    const newer = hydrator.hydrate('root', secondRevision);
    const duplicatePending = hydrator.hydrate('root', secondRevision);
    assert.equal(reads.length, 1);
    reads[0].gate.resolve({ revision: firstRevision });
    await first;
    await Promise.resolve();
    assert.equal(reads.length, 2);
    reads[1].gate.resolve({ revision: secondRevision });
    await newer;
    await duplicatePending;
    await duplicate;
    assert.deepEqual(applied, [firstRevision, secondRevision]);
    assert.equal(await hydrator.hydrate('root', secondRevision), false);
    assert.equal(reads.length, 2);
});

test('review hydration keeps the originating viewport writer for a trailing revision', async () => {
    let activeWriter = '';
    const reads = [];
    const applied = [];
    const deferred = () => {
        let resolve;
        const promise = new Promise((done) => { resolve = done; });
        return { promise, resolve };
    };
    const writer = (label) => (mutate) => {
        const previous = activeWriter;
        activeWriter = label;
        try { return mutate(); } finally { activeWriter = previous; }
    };
    const hydrator = createReviewHydrator({
        fetchDetail() {
            const gate = deferred();
            reads.push(gate);
            return gate.promise;
        },
        applyDetail(_taskId, detail) {
            applied.push([detail.revision, activeWriter]);
            return true;
        },
        onState: () => false,
    });
    const firstRevision = 'd'.repeat(64);
    const secondRevision = 'e'.repeat(64);
    const first = hydrator.hydrate('root', firstRevision, { onDomWrite: writer('local') });
    await Promise.resolve();
    const trailing = hydrator.hydrate('root', secondRevision, { onDomWrite: writer('remote') });
    reads[0].resolve({ revision: firstRevision });
    await first;
    await Promise.resolve();
    reads[1].resolve({ revision: secondRevision });
    await trailing;
    assert.deepEqual(applied, [
        [firstRevision, 'local'],
        [secondRevision, 'remote'],
    ]);
    assert.equal(await hydrator.hydrate('root', secondRevision, {
        onDomWrite: writer('duplicate'),
    }), false);
});

test('applied-revision invalidation preserves and joins an in-flight physical read', async () => {
    let resolveRead;
    let reads = 0;
    const revision = 'c'.repeat(64);
    const hydrator = createReviewHydrator({
        fetchDetail() {
            reads += 1;
            return new Promise((resolve) => { resolveRead = resolve; });
        },
        applyDetail: () => true,
    });

    const first = hydrator.hydrate('root', revision);
    await Promise.resolve();
    hydrator.invalidateApplied();
    const joined = hydrator.hydrate('root', revision);
    assert.equal(reads, 1, 'presentation reset did not duplicate the in-flight GET');
    resolveRead({ revision });
    await Promise.all([first, joined]);
    assert.equal(await hydrator.hydrate('root', revision), false,
        'the joined read restored the applied revision receipt');
    assert.equal(reads, 1);
});

test('invalid review revisions stay opaque and do not become ordered counters', async () => {
    let reads = 0;
    const hydrator = createReviewHydrator({
        fetchDetail: async () => ({ ok: true }),
        applyDetail: () => { reads += 1; return true; },
    });
    await hydrator.hydrate('root', 42);
    await hydrator.hydrate('root', 42);
    assert.equal(reads, 2);
});

test('review re-render restores keyboard focus to the equivalent disclosure control', () => {
    const doc = { activeElement: null };
    const buttons = new Map();
    let clickHandler = null;
    const makeButton = (kind, key = '') => ({
        dataset: kind === 'group' ? { reviewGroupToggle: key } : {},
        matches: (selector) => kind === 'section' && selector === '[data-review-section-toggle]',
        closest(selector) {
            return kind === 'group' && selector === '[data-review-group-toggle]' ? this : null;
        },
        focus() { doc.activeElement = this; },
    });
    const host = {
        ownerDocument: doc,
        addEventListener(type, handler) { if (type === 'click') clickHandler = handler; },
        contains: (candidate) => [...buttons.values()].includes(candidate),
        querySelector: (selector) => selector === '[data-review-section-toggle]' ? buttons.get('section') : null,
        querySelectorAll: (selector) => selector === '[data-review-group-toggle]' ? [buttons.get('group')] : [],
        set innerHTML(value) {
            this._html = value;
            buttons.set('section', makeButton('section'));
            buttons.set('group', makeButton('group', 'task:root:alpha'));
        },
        get innerHTML() { return this._html || ''; },
    };
    const summary = { hidden: true, textContent: '' };
    const disclosure = { sectionExpanded: false, expandedGroups: new Set(), expandedAttempts: new Set() };
    const controller = createReviewPresentationController({ host, summary, disclosure });
    controller.update(reviewGroupFromHistoryRow(groupedSkillRow()));
    const oldGroupButton = buttons.get('group');
    doc.activeElement = oldGroupButton;
    clickHandler({ target: oldGroupButton });
    assert.equal(disclosure.expandedGroups.has('task:root:alpha'), true);
    assert.equal(doc.activeElement, buttons.get('group'));
    assert.notEqual(doc.activeElement, oldGroupButton);
});

test('keyed reconciliation preserves exact detail node, focused descendant and scrollTop', () => {
    class ReviewElement {
        constructor({ tag = 'div', dataset = {}, classes = [], attrs = {}, html = '', children = [] } = {}) {
            this.tagName = tag.toUpperCase();
            this.dataset = { ...dataset };
            this._attrs = new Map(Object.entries(attrs));
            for (const [key, value] of Object.entries(dataset)) {
                const attr = `data-${key.replace(/[A-Z]/g, (char) => `-${char.toLowerCase()}`)}`;
                this._attrs.set(attr, String(value));
            }
            this._classes = new Set(classes);
            if (classes.length) this._attrs.set('class', classes.join(' '));
            this.classList = { contains: (name) => this._classes.has(name) };
            this._innerHTML = html;
            this.children = [];
            this.scrollTop = 0;
            children.forEach((child) => this.insertBefore(child, null));
        }
        get attributes() { return [...this._attrs].map(([name, value]) => ({ name, value })); }
        get innerHTML() { return this._innerHTML; }
        set innerHTML(value) { this._innerHTML = String(value); this.children = []; }
        hasAttribute(name) { return this._attrs.has(name); }
        setAttribute(name, value) {
            const normalized = String(value);
            this._attrs.set(name, normalized);
            if (name.startsWith('data-')) {
                const key = name.slice(5).replace(/-([a-z])/g, (_all, char) => char.toUpperCase());
                this.dataset[key] = normalized;
            }
        }
        removeAttribute(name) { this._attrs.delete(name); }
        insertBefore(child, before) {
            child.remove();
            const index = before ? this.children.indexOf(before) : -1;
            if (index >= 0) this.children.splice(index, 0, child); else this.children.push(child);
            child.parentElement = this;
            return child;
        }
        remove() {
            if (!this.parentElement) return;
            const index = this.parentElement.children.indexOf(this);
            if (index >= 0) this.parentElement.children.splice(index, 1);
            this.parentElement = null;
        }
        cloneNode(deep) {
            return new ReviewElement({
                tag: this.tagName,
                dataset: this.dataset,
                classes: [...this._classes],
                attrs: Object.fromEntries(this._attrs),
                html: this._innerHTML,
                children: deep ? this.children.map((child) => child.cloneNode(true)) : [],
            });
        }
    }
    const node = (dataset, children = [], classes = []) => new ReviewElement({ dataset, children, classes });
    const focused = new ReviewElement({ tag: 'button', html: 'Retry' });
    const detail = node({
        reviewAttemptDetail: 'task:root:alpha:job-1',
        skillReviewSkill: 'alpha', skillReviewJob: 'job-1', state: 'loaded',
    }, [focused]);
    detail.setAttribute('aria-busy', 'false');
    detail.scrollTop = 37;
    const attempt = node({ reviewAttempt: 'task:root:alpha:job-1' }, [detail]);
    const attempts = node({}, [attempt], ['chat-review-attempts']);
    const group = node({ reviewGroup: 'task:root:alpha' }, [attempts]);
    const current = node({ reviewSection: '' }, [group]);

    const desiredDetail = node({
        reviewAttemptDetail: 'task:root:alpha:job-1',
        skillReviewSkill: 'alpha', skillReviewJob: 'job-1',
    });
    const desiredAttempt = node({ reviewAttempt: 'task:root:alpha:job-1' }, [desiredDetail]);
    const desiredNewAttempt = node({ reviewAttempt: 'task:root:alpha:job-2' }, [
        node({ reviewAttemptDetail: 'task:root:alpha:job-2', skillReviewSkill: 'alpha', skillReviewJob: 'job-2' }),
    ]);
    const desired = node({ reviewSection: '' }, [
        node({ reviewGroup: 'task:root:alpha' }, [
            node({}, [desiredAttempt, desiredNewAttempt], ['chat-review-attempts']),
        ]),
    ]);

    assert.equal(reconcileReviewElementTree(current, desired), true);
    assert.equal(current.children[0], group);
    assert.equal(group.children[0], attempts);
    assert.equal(attempts.children[0], attempt);
    assert.equal(attempt.children[0], detail);
    assert.equal(detail.children[0], focused);
    assert.equal(detail.scrollTop, 37);
    assert.equal(detail.dataset.state, 'loaded');
    assert.equal(attempts.children.length, 2);
});

test('Retry keeps keyboard focus on the live detail status while refetching', () => {
    const doc = { activeElement: null };
    let clickHandler = null;
    let retryOptions = null;
    const detail = {
        dataset: { skillReviewSkill: 'alpha', skillReviewJob: 'job-1' },
        setAttribute(key, value) { this[key] = value; },
        focus() { doc.activeElement = this; },
    };
    const retry = { closest: (selector) => selector === '[data-review-attempt-detail]' ? detail : null };
    const host = {
        ownerDocument: doc,
        addEventListener(type, handler) { if (type === 'click') clickHandler = handler; },
        contains: () => true,
        querySelectorAll: () => [],
    };
    createReviewPresentationController({
        host,
        summary: { hidden: true, textContent: '' },
        disclosure: {},
        onLoadSkillDetail(_detail, options) { retryOptions = options; },
    });
    clickHandler({ target: { closest: (selector) => selector === '[data-skill-review-retry]' ? retry : null } });
    assert.deepEqual(retryOptions, { retry: true });
    assert.equal(doc.activeElement, detail);
    assert.equal(detail.tabindex, '-1');
});

test('generic expandByDefault plumbing is gone from Chat and log projection', () => {
    const chat = readFileSync(new URL('../modules/chat.js', import.meta.url), 'utf8');
    const logs = readFileSync(new URL('../modules/log_events.js', import.meta.url), 'utf8');
    assert.doesNotMatch(chat, /expandByDefault/);
    assert.doesNotMatch(logs, /expandByDefault/);
    assert.doesNotMatch(chat, /stickyExpandedSlots/);
});

test('history and live chat intercept owner-bound lifecycle before generic progress', () => {
    const chat = readFileSync(new URL('../modules/chat.js', import.meta.url), 'utf8');
    const pass1 = chat.slice(chat.indexOf('// First pass builds'), chat.indexOf('// Pass 2 inserts cards'));
    assert.ok(pass1.indexOf('attachReviewFromRow(msg') >= 0);
    assert.ok(pass1.indexOf('attachReviewFromRow(msg') < pass1.indexOf('if (msg.is_progress)'));

    const liveStart = chat.indexOf("if (msg.role === 'assistant' || msg.role === 'system')");
    const live = chat.slice(liveStart, chat.indexOf("onWs('message_annotation'", liveStart));
    assert.ok(live.indexOf('attachReviewFromRow(msg') >= 0);
    assert.ok(live.indexOf('attachReviewFromRow(msg') < live.indexOf('if (msg.is_progress)'));
});
