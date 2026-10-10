import test from 'node:test';
import assert from 'node:assert/strict';
import {
    buildProjectActivityIndex,
    reconcileProjectActivityCensus,
    summarizeProjectActivities,
} from '../modules/project_activity.js';

const activity = (patch = {}) => ({
    activity_id: patch.activity_id || 'a-1',
    chat_id: patch.chat_id || 1,
    project_id: patch.project_id || '',
    kind: patch.kind || 'direct_chat',
    phase: patch.phase || 'thinking',
    ...patch,
});

test('working, thinking and finalizing are the only animated phases', () => {
    for (const phase of ['working', 'thinking', 'finalizing']) {
        const summary = summarizeProjectActivities([activity({ phase })]);
        assert.equal(summary.motion, true, phase);
        assert.equal(summary.state, 'working', phase);
    }
    for (const phase of ['queued', 'budget_paused', 'unknown']) {
        const summary = summarizeProjectActivities([activity({ phase })]);
        assert.equal(summary.motion, false, phase);
    }
});

test('waits are amber facts and resumed questions are not waits', () => {
    // Fixtures carry the producer's complete wait row: the reducer admits waits
    // exactly as the chat card does, so a bare {state, task_attempt} is dropped.
    const wait = (patch = {}) => ({ wait_id: 'w', revision: 1, task_attempt: 1, state: 'waiting', reason: 'quota', ...patch });
    const modelWait = activity({ project_id: 'p', phase: 'working', model_waits: { w: wait() }, task_attempt: 1 });
    const question = activity({ project_id: 'p', phase: 'queued', required_question: {
        wait_for_answer: true, owner_wait_state: 'waiting',
    } });
    const resumed = activity({ project_id: 'p', phase: 'queued', required_question: {
        wait_for_answer: true, owner_wait_state: 'resumed',
    } });
    const sameRowWait = summarizeProjectActivities([modelWait]);
    assert.equal(sameRowWait.motion, false);
    assert.equal(sameRowWait.state, 'waiting');
    assert.doesNotMatch(sameRowWait.label, /Working/);
    assert.match(sameRowWait.label, /Waiting for access/);
    const mixed = summarizeProjectActivities([
        modelWait,
        activity({ activity_id: 'independent', project_id: 'p', phase: 'working' }),
    ]);
    assert.equal(mixed.state, 'working');
    assert.equal(mixed.motion, true);
    assert.match(mixed.label, /Working/);
    assert.match(mixed.label, /Waiting for access/);
    assert.deepEqual(summarizeProjectActivities([question]), {
        state: 'waiting', motion: false, waiting: true,
        label: 'Waiting for your answer',
    });
    assert.equal(summarizeProjectActivities([resumed]).waiting, false);
});

test('project summaries include direct conversations and managed roots, excluding unbound work', () => {
    const rows = [
        activity({ activity_id: 'direct', project_id: 'p', phase: 'thinking' }),
        activity({ activity_id: 'managed', project_id: 'p', kind: 'managed_task', phase: 'queued' }),
        activity({ activity_id: 'main', phase: 'thinking' }),
        activity({ activity_id: 'child', project_id: 'p', phase: 'working', is_child: true }),
        activity({ activity_id: 'duplicate', project_id: 'p', phase: 'working' }),
        activity({ activity_id: 'duplicate', project_id: '', phase: 'thinking' }),
    ];
    const index = buildProjectActivityIndex(rows);
    assert.equal(index.byProject.get('p').motion, true);
    assert.match(index.byProject.get('p').label, /Thinking/);
    assert.match(index.byProject.get('p').label, /Working/);
    assert.equal(buildProjectActivityIndex([activity({ chat_id: 0 })]).aggregate.motion, false);
    assert.equal(index.aggregate.motion, true);
});

test('partial or disconnected census never clears a previous active row', () => {
    const first = reconcileProjectActivityCensus(new Map(), {
        active_chat_activities: [activity({ activity_id: 'live', project_id: 'p', phase: 'working' })],
        active_chat_activities_complete: true, supervisor_ready: true,
    });
    const partial = reconcileProjectActivityCensus(first.rows, {
        active_chat_activities: [], active_chat_activities_complete: false, supervisor_ready: true,
    });
    assert.equal(partial.rows.has('live'), true);
    const uncertain = buildProjectActivityIndex([...partial.rows.values()]);
    assert.equal(uncertain.byProject.get('p').state, 'unknown');
    assert.equal(uncertain.byProject.get('p').motion, false);
    assert.equal(uncertain.byProject.get('p').label, 'Activity unconfirmed');
    const disconnected = reconcileProjectActivityCensus(partial.rows, {});
    assert.equal(disconnected.rows.has('live'), true);
    const disconnectedIndex = buildProjectActivityIndex([...disconnected.rows.values()]);
    assert.equal(disconnectedIndex.byProject.get('p').state, 'unknown');
    assert.equal(disconnectedIndex.byProject.get('p').motion, false);
    const empty = reconcileProjectActivityCensus(disconnected.rows, {
        active_chat_activities: [], active_chat_activities_complete: true, supervisor_ready: true,
    });
    assert.equal(empty.rows.size, 0);
});

test('a partial census marks only omissions unknown and does not mutate producer rows', () => {
    const one = activity({ activity_id: 'one', project_id: 'p', phase: 'working' });
    const two = activity({ activity_id: 'two', project_id: 'q', phase: 'thinking' });
    const initial = reconcileProjectActivityCensus(new Map(), {
        active_chat_activities: [one], active_chat_activities_complete: true, supervisor_ready: true,
    });
    const partial = reconcileProjectActivityCensus(initial.rows, { active_chat_activities: [two] });
    const index = buildProjectActivityIndex([...partial.rows.values()]);
    assert.equal(index.byProject.get('p').state, 'unknown');
    assert.equal(index.byProject.get('p').motion, false);
    assert.equal(index.byProject.get('q').motion, true);
    assert.match(index.aggregate.label, /Thinking.*Activity unconfirmed/);
    assert.equal(Object.hasOwn(one, '_activityUnconfirmed'), false);
    assert.equal(Object.hasOwn(two, '_activityUnconfirmed'), false);
    const restored = reconcileProjectActivityCensus(partial.rows, { active_chat_activities: [one] });
    assert.equal(buildProjectActivityIndex([...restored.rows.values()]).byProject.get('p').motion, true);
});

test('complete needs an array and literal true readiness before it can clear absences', () => {
    const rows = new Map([['one', activity({ activity_id: 'one', project_id: 'p' })]]);
    for (const data of [
        {}, { active_chat_activities_complete: true, supervisor_ready: true },
        { active_chat_activities: [], active_chat_activities_complete: true, supervisor_ready: false },
        { active_chat_activities: [], active_chat_activities_complete: 1, supervisor_ready: true },
    ]) {
        const next = reconcileProjectActivityCensus(rows, data);
        assert.equal(next.rows.size, 1);
        assert.equal(buildProjectActivityIndex([...next.rows.values()]).byProject.get('p').state, 'unknown');
    }
});

test('model waits use the existing current-attempt rule and questions end on their own lifecycle', () => {
    const row = activity({ phase: 'finalizing', task_attempt: 2,
        model_waits: { previous: { wait_id: 'previous', revision: 1, task_attempt: 1, state: 'waiting', reason: 'quota' } } });
    assert.equal(summarizeProjectActivities([row]).motion, true);
    row.model_waits.current = { wait_id: 'current', revision: 1, task_attempt: 2, state: 'waiting', reason: 'auth' };
    assert.equal(summarizeProjectActivities([row]).state, 'waiting');
    row.model_waits.current.state = 'resolved';
    assert.equal(summarizeProjectActivities([row]).motion, true);
    for (const quiz_state of ['answered', 'superseded', 'expired_terminal']) {
        row.required_question = { quiz_state, wait_for_answer: true, owner_wait_state: 'waiting' };
        assert.equal(summarizeProjectActivities([row]).waiting, false);
    }
    row.required_question = { quiz_state: 'open', owner_wait_state: 'waiting' };
    assert.equal(summarizeProjectActivities([row]).motion, false);
    row.required_question.wait_ended_at = '2026-09-22T00:00:00Z';
    assert.equal(summarizeProjectActivities([row]).motion, true);
});

test('budget-paused work is a stationary wait, not a queue', () => { const s=summarizeProjectActivities([activity({phase:'budget_paused'})]); assert.equal(s.state,'waiting'); assert.equal(s.motion,false); assert.equal(s.label,'Paused'); });
test('budget-pausing work (#1196) is stationary and never a false terminal or motion', () => { const s=summarizeProjectActivities([activity({phase:'budget_pausing'})]); assert.equal(s.state,'waiting'); assert.equal(s.motion,false); assert.equal(s.label,'Pausing…'); });

test('model wait rows pass the same admission as the chat card: malformed waits cannot stop motion', () => {
    // Missing wait_id/revision/reason: the card would drop this row, so must the sidebar.
    const malformed = activity({ project_id: 'p', phase: 'working', task_attempt: 1,
        model_waits: { w: { state: 'waiting', task_attempt: 1 } } });
    const s = summarizeProjectActivities([malformed]);
    assert.equal(s.motion, true);
    assert.equal(s.state, 'working');
    assert.doesNotMatch(s.label, /Waiting for access/);
    // The complete row the producer writes (wait_id echoes its key, positive revision, typed reason).
    const wellFormed = activity({ project_id: 'p', phase: 'working', task_attempt: 1,
        model_waits: { w: { wait_id: 'w', revision: 1, task_attempt: 1, state: 'waiting', reason: 'quota' } } });
    const w = summarizeProjectActivities([wellFormed]);
    assert.equal(w.motion, false);
    assert.equal(w.state, 'waiting');
    assert.match(w.label, /Waiting for access/);
});

test('a row whose owner-question detail could not be read is static unknown, never moving', () => {
    const row = activity({ project_id: 'p', phase: 'working', required_question_unavailable: true });
    const s = summarizeProjectActivities([row]);
    assert.equal(s.motion, false);
    assert.equal(s.state, 'unknown');
    assert.match(s.label, /Activity unconfirmed/);
    // Beside an independently confirmed working row the project still moves, and the gap stays named.
    const mixed = summarizeProjectActivities([row, activity({ activity_id: 'b', project_id: 'p', phase: 'working' })]);
    assert.equal(mixed.motion, true);
    assert.match(mixed.label, /Activity unconfirmed/);
});
