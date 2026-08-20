// Behavioural characterization of the task-frame router, exercised where the
// code now lives. The router is driven with the same frame shapes the WS
// handlers and history replay feed it, with the live-card store, the task
// tracker and the subagent routing replaced by observable stubs.

import assert from 'node:assert/strict';
import test from 'node:test';

import { OWNER_STOP_DONE_HEADLINE } from '../modules/log_events.js';
import { createTaskFrames } from '../modules/chat_task_frames.js';

function taskFrames({ cardVisible = true, childParents = [] } = {}) {
    const liveCardRecords = new Map();
    const subagentChildParents = new Map(childParents);
    const subagentTerminalChildren = new Set();
    const activeDirectActivities = new Map();
    const taskUiStates = new Map();
    const calls = [];
    const spy = (name, result) => (...args) => { calls.push({ name, args }); return result; };

    const api = createTaskFrames({
        liveCardRecords,
        subagentChildParents,
        subagentTerminalChildren,
        activeDirectActivities,
        getActiveLiveGroupId: () => '',
        registerEphemeralDecisionFrame: spy('registerEphemeralDecisionFrame', false),
        revealBufferedCardIfNeeded: spy('revealBufferedCardIfNeeded'),
        queueTaskLiveUpdate: spy('queueTaskLiveUpdate'),
        getSubagentCardRecord: spy('getSubagentCardRecord'),
        applyLiveCardState: spy('applyLiveCardState'),
        finishLiveCard: spy('finishLiveCard'),
        applySuggestedName: spy('applySuggestedName'),
        getTaskUiState: (taskId, create) => {
            if (!taskUiStates.has(taskId) && !create) return null;
            if (!taskUiStates.has(taskId)) {
                taskUiStates.set(taskId, { taskId, cardVisible, completed: false, forceCard: false });
            }
            return taskUiStates.get(taskId);
        },
        scheduleTaskUiCleanup: spy('scheduleTaskUiCleanup'),
        markTaskToolCall: spy('markTaskToolCall'),
        forceTaskCard: spy('forceTaskCard'),
        markAssistantReply: spy('markAssistantReply'),
        markTaskCancelable: spy('markTaskCancelable'),
        updateSubagentCardFromEvent: spy('updateSubagentCardFromEvent', true),
        routeSubagentProgressToCard: spy('routeSubagentProgressToCard'),
        routeSubagentTerminalToCard: spy('routeSubagentTerminalToCard'),
        recordConcludedActivity: spy('recordConcludedActivity'),
        syncChatStatus: spy('syncChatStatus'),
    });

    return {
        ...api,
        liveCardRecords,
        activeDirectActivities,
        taskUiStates,
        calls,
        named: (name) => calls.filter((call) => call.name === name),
        seedState(taskId) {
            taskUiStates.set(taskId, { taskId, cardVisible, completed: false, forceCard: false });
        },
        restore() {},
    };
}

test('a cancelled task summary reads Cancelled, never a generic Done', () => {
    const f = taskFrames();
    // Without a review projection the summary only settles an EXISTING state.
    f.seedState('t1');
    f.appendTaskSummaryToLiveCard({ task_id: 't1', status: 'cancelled', ts: '2026-08-18T00:00:00Z' });
    const [applied] = f.named('applyLiveCardState');
    assert.equal(applied.args[0].headline, 'Cancelled');
    assert.equal(applied.args[0].terminal, true);
    assert.equal(f.named('finishLiveCard').length, 1);
    assert.equal(f.named('scheduleTaskUiCleanup').length, 1);
    f.restore();
});

test('an owner-requested soft stop presents its own success headline', () => {
    const f = taskFrames();
    f.seedState('t2');
    f.appendTaskSummaryToLiveCard({
        task_id: 't2', status: 'done',
        reason_code: 'owner_requested_finalization', result: 'summary text',
        ts: '2026-08-18T00:00:00Z',
    });
    const [applied] = f.named('applyLiveCardState');
    assert.equal(applied.args[0].headline, OWNER_STOP_DONE_HEADLINE);
    assert.equal(applied.args[0].visible, true, 'the owner-request marker renders in the details');
    f.restore();
});

test('a summary without a task id settles the active card as done', () => {
    const f = taskFrames();
    f.appendTaskSummaryToLiveCard({});
    assert.deepEqual(f.named('finishLiveCard')[0].args, ['', 'done']);
    assert.equal(f.named('applyLiveCardState').length, 0);
    f.restore();
});

test('an invisible card records the reply instead of forcing a card open', () => {
    const f = taskFrames({ cardVisible: false });
    f.seedState('t3');
    f.appendTaskSummaryToLiveCard({ task_id: 't3', status: 'done', ts: '2026-08-18T00:00:00Z' });
    assert.equal(f.named('markAssistantReply').length, 1);
    assert.equal(f.named('applyLiveCardState').length, 0);
    f.restore();
});

test('the host-attested cancelable marker is trusted from any progress frame', () => {
    const f = taskFrames();
    f.updateLiveCardFromProgressMessage({
        task_id: 'root-1', cancelable: true, content: 'working', ts: '2026-08-18T00:00:00Z',
    });
    assert.deepEqual(f.named('markTaskCancelable')[0].args, ['root-1']);
    assert.equal(f.named('queueTaskLiveUpdate').length, 1);
    const queued = f.named('queueTaskLiveUpdate')[0];
    assert.equal(queued.args[1], 'root-1');
    f.restore();
});

test('a known child routes its own progress to the child card, not the parent', () => {
    const f = taskFrames({ childParents: [['child-1', { parentId: 'root-1', role: 'worker' }]] });
    f.updateLiveCardFromProgressMessage({ task_id: 'child-1', content: 'child step' });
    assert.equal(f.named('routeSubagentProgressToCard').length, 1);
    assert.equal(f.named('queueTaskLiveUpdate').length, 0);
    f.restore();
});

test('a history progress row with terminal truth projects the summary too', () => {
    const f = taskFrames();
    f.updateLiveCardFromProgressMessage({
        task_id: 't4', content: 'step', suggested_name: 'Named task',
        task_terminal_status: 'done', reason_code: 'review_degraded',
        ts: '2026-08-18T00:00:00Z',
    });
    assert.deepEqual(f.named('applySuggestedName')[0].args, ['t4', 'Named task']);
    assert.equal(f.named('finishLiveCard').length, 1, 'the terminal truth lands through the summary path');
    f.restore();
});

test('owner_hurry marks the card silently: no timeline row, no bubble', () => {
    const f = taskFrames();
    const root = { attributes: {}, setAttribute(name, value) { this.attributes[name] = value; } };
    f.liveCardRecords.set('t5', { root });
    // The grouped-frame guard admits progress-stamped frames; live owner_hurry
    // events ride the progress channel.
    f.updateLiveCardFromLogEvent({ type: 'owner_hurry', task_id: 't5', phase: 'applied', is_progress: true });
    assert.equal(root.attributes['data-owner-hurry'], '1');
    assert.equal(f.named('queueTaskLiveUpdate').length, 0);
    assert.equal(f.named('applyLiveCardState').length, 0);
    f.restore();
});

test('tool-call and error log events drive the tracker before the summary', () => {
    const f = taskFrames();
    f.updateLiveCardFromLogEvent({ type: 'tool_call_started', task_id: 't6', ts: '2026-08-18T00:00:00Z' });
    assert.equal(f.named('markTaskToolCall').length, 1);
    f.updateLiveCardFromLogEvent({ type: 'llm_round_error', task_id: 't6', error: 'boom' });
    assert.ok(f.named('forceTaskCard').length >= 1);
    f.restore();
});

test('a task_done log event concludes the managed activity for the header', () => {
    const f = taskFrames();
    f.activeDirectActivities.set('t7', { activityId: 't7', kind: 'managed_task' });
    f.updateLiveCardFromLogEvent({ type: 'task_done', task_id: 't7', status: 'done', ts: '2026-08-18T00:00:00Z' });
    assert.equal(f.activeDirectActivities.has('t7'), false);
    assert.deepEqual(f.named('recordConcludedActivity')[0].args, ['t7']);
    assert.ok(f.named('syncChatStatus').length >= 1);
    f.restore();
});

test('a known child terminal log event routes to the child card', () => {
    const f = taskFrames({ childParents: [['child-2', { parentId: 'root-2', role: 'worker' }]] });
    f.updateLiveCardFromLogEvent({ type: 'task_done', task_id: 'child-2', status: 'done' });
    assert.equal(f.named('routeSubagentTerminalToCard').length, 1);
    assert.equal(f.named('finishLiveCard').length, 0, 'the parent card is untouched');
    f.restore();
});
