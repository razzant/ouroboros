// Behavioural characterization of the per-task UI bookkeeping owner, exercised
// where the code now lives. No DOM: the tracker only reads and writes the task
// state ledger and asks the instance to reveal a buffered card, so a recording
// stub is enough to pin every decision — when a card is earned, what it buffers
// until then, and which ids are retired from routine syncs.

import assert from 'node:assert/strict';
import test from 'node:test';

import { createTaskUiStateTracker } from '../modules/chat_task_ui_state.js';

function tracker() {
    const taskUiStates = new Map();
    const retiredTaskIds = new Set();
    const reveals = [];
    const api = createTaskUiStateTracker({
        taskUiStates,
        retiredTaskIds,
        revealBufferedCardIfNeeded: (taskState, options) => reveals.push({ taskState, options }),
    });
    return { ...api, taskUiStates, retiredTaskIds, reveals };
}

test('a fresh task state starts with no tool calls, no card and no buffer', () => {
    const t = tracker();
    const state = t.getTaskUiState('task-1');
    assert.deepEqual(state, {
        taskId: 'task-1',
        toolCalls: 0,
        forceCard: false,
        cardVisible: false,
        completed: false,
        completedPhase: '',
        bufferedLiveUpdates: [],
        cleanupTimer: null,
    });
    assert.equal(t.taskUiStates.get('task-1'), state);
});

test('getTaskUiState is a lookup without createIfMissing and never mints an empty id', () => {
    const t = tracker();
    assert.equal(t.getTaskUiState('task-1', false), null);
    assert.equal(t.taskUiStates.size, 0);
    assert.equal(t.getTaskUiState('', true), null);
    assert.equal(t.taskUiStates.size, 0);
    const state = t.getTaskUiState('task-1', true);
    assert.equal(t.getTaskUiState('task-1', false), state);
});

test('tool calls accumulate, but a metrics frame only raises the floor', () => {
    const t = tracker();
    t.markTaskToolCall('task-1', 1, false, 'ts-a');
    t.markTaskToolCall('task-1', 1, false, 'ts-b');
    assert.equal(t.getTaskUiState('task-1').toolCalls, 2);
    // minimumOnly: a late authoritative count never rewinds the live tally.
    t.markTaskToolCall('task-1', 1, true, 'ts-c');
    assert.equal(t.getTaskUiState('task-1').toolCalls, 2);
    t.markTaskToolCall('task-1', 5, true, 'ts-d');
    assert.equal(t.getTaskUiState('task-1').toolCalls, 5);
    // Every mark asks the instance to reveal, carrying the frame's raw timestamp.
    assert.deepEqual(t.reveals.map((entry) => entry.options.rawTs), ['ts-a', 'ts-b', 'ts-c', 'ts-d']);
});

test('forceTaskCard flags the state and asks for the reveal', () => {
    const t = tracker();
    const state = t.forceTaskCard('task-1', 'ts-a');
    assert.equal(state.forceCard, true);
    assert.equal(t.reveals.length, 1);
    assert.equal(t.reveals[0].taskState, state);
});

test('buffered updates keep their own dedupe key, falling back to the summary key', () => {
    const t = tracker();
    const state = t.getTaskUiState('task-1');
    t.bufferLiveUpdate(state, { dedupeKey: 'from-summary' }, 'ts', '', 'raw');
    t.bufferLiveUpdate(state, { dedupeKey: 'from-summary' }, 'ts', 'explicit', 'raw');
    t.bufferLiveUpdate(state, {}, 'ts', '', 'raw');
    assert.deepEqual(state.bufferedLiveUpdates.map((u) => u.dedupeKey), ['from-summary', 'explicit', '']);
    // A missing state or summary is a no-op, never a thrown frame.
    t.bufferLiveUpdate(null, { dedupeKey: 'x' }, 'ts');
    t.bufferLiveUpdate(state, null, 'ts');
    assert.equal(state.bufferedLiveUpdates.length, 3);
});

test('an assistant reply completes an existing state only, with a short wait when no card was shown', () => {
    const t = tracker();
    t.markAssistantReply('never-seen');
    assert.equal(t.taskUiStates.size, 0, 'a reply must not mint a ledger entry for an unknown task');
    const state = t.getTaskUiState('task-1');
    t.markAssistantReply('task-1');
    assert.equal(state.completed, true);
    assert.equal(state.completedPhase, 'done');
    assert.notEqual(state.cleanupTimer, null);
    clearTimeout(state.cleanupTimer);
});

test('markTaskComplete records completion without inventing a phase or a state', () => {
    const t = tracker();
    t.markTaskComplete('never-seen', 'error');
    assert.equal(t.taskUiStates.size, 0);
    const state = t.getTaskUiState('task-1');
    state.completedPhase = 'warn';
    t.markTaskComplete('task-1', '');
    assert.equal(state.completed, true);
    assert.equal(state.completedPhase, 'warn', 'an empty phase must not erase a recorded one');
    t.markTaskComplete('task-1', 'error');
    assert.equal(state.completedPhase, 'error');
});

test('cleanup drops the ledger entry and retires an ordinary id, never a reusable slot', async () => {
    const t = tracker();
    const ordinary = t.getTaskUiState('task-1');
    const reusable = t.getTaskUiState('bg-consciousness');
    t.scheduleTaskUiCleanup(ordinary, 1);
    t.scheduleTaskUiCleanup(reusable, 1);
    await new Promise((resolve) => setTimeout(resolve, 20));
    assert.equal(t.taskUiStates.has('task-1'), false);
    assert.equal(t.taskUiStates.has('bg-consciousness'), false);
    assert.deepEqual([...t.retiredTaskIds], ['task-1']);
});

test('rescheduling cleanup replaces the pending timer instead of stacking one', () => {
    const t = tracker();
    const state = t.getTaskUiState('task-1');
    t.scheduleTaskUiCleanup(state, 10000);
    const first = state.cleanupTimer;
    t.scheduleTaskUiCleanup(state, 10000);
    assert.notEqual(state.cleanupTimer, first);
    clearTimeout(state.cleanupTimer);
});

test('the background slot always shows its card; other tasks must earn one', () => {
    const t = tracker();
    assert.equal(t.isBackgroundTaskId('bg-consciousness'), true);
    assert.equal(t.isBackgroundTaskId('task-1'), false);
    assert.equal(t.shouldAlwaysShowTaskCard('bg-consciousness'), true);
    assert.equal(t.shouldAlwaysShowTaskCard('task-1'), false);
});

test('a foreground live card is connected, unfinished and not the background slot', () => {
    const t = tracker();
    const live = { groupId: 'task-1', finished: false, root: { isConnected: true } };
    assert.equal(t.isForegroundLiveCard(live), true);
    assert.equal(t.isForegroundLiveCard({ ...live, finished: true }), false);
    assert.equal(t.isForegroundLiveCard({ ...live, root: { isConnected: false } }), false);
    assert.equal(t.isForegroundLiveCard({ ...live, groupId: 'bg-consciousness' }), false);
    assert.equal(t.isForegroundLiveCard(null), false);
});
