import { REUSABLE_TASK_IDS } from './task_control_menu.js';

// Per-task UI bookkeeping for ONE chat instance: the ledger that decides whether
// a task ever earns a live card, what it buffered before it did, and when its
// entry is retired. The card DOM itself is never touched here — the tracker only
// records tool calls, forced reveals, assistant replies and completion, then asks
// the instance to reveal a buffered card. The state map, the retirement set and
// the reveal entry point are handed over explicitly, so a Main chat and a Project
// panel keep independent ledgers over their own transcripts.
export function createTaskUiStateTracker({
    taskUiStates,
    retiredTaskIds,
    revealBufferedCardIfNeeded,
}) {
    function isBackgroundTaskId(taskId = '') {
        return taskId === 'bg-consciousness';
    }

    function shouldAlwaysShowTaskCard(taskId = '') {
        return isBackgroundTaskId(taskId);
    }

    function isForegroundLiveCard(record) {
        return Boolean(record?.root?.isConnected && !record.finished && !isBackgroundTaskId(record.groupId));
    }

    function createTaskUiState(taskId) {
        if (!taskId) return null;
        const taskState = {
            taskId,
            toolCalls: 0,
            forceCard: false,
            cardVisible: false,
            completed: false,
            completedPhase: '',
            bufferedLiveUpdates: [],
            cleanupTimer: null,
        };
        taskUiStates.set(taskId, taskState);
        return taskState;
    }

    function getTaskUiState(taskId = '', createIfMissing = true) {
        if (!taskId) return null;
        if (taskUiStates.has(taskId)) return taskUiStates.get(taskId);
        return createIfMissing ? createTaskUiState(taskId) : null;
    }

    function scheduleTaskUiCleanup(taskState, delayMs = 120000) {
        if (!taskState) return;
        if (taskState.cleanupTimer) clearTimeout(taskState.cleanupTimer);
        taskState.cleanupTimer = setTimeout(() => {
            taskUiStates.delete(taskState.taskId);
            // Keep the finished card interactive, but mark it retired so routine
            // syncs do not rebuild duplicates. Reload/reconnect clears this set.
            if (!REUSABLE_TASK_IDS.has(taskState.taskId) && taskState.taskId !== '') {
                retiredTaskIds.add(taskState.taskId);
            }
        }, delayMs);
    }

    function bufferLiveUpdate(taskState, summary, ts, dedupeKey = '', rawTs = '') {
        if (!taskState || !summary) return;
        taskState.bufferedLiveUpdates.push({
            summary,
            ts,
            rawTs,
            dedupeKey: dedupeKey || summary.dedupeKey || '',
        });

    }

    function markTaskToolCall(taskId, count = 1, minimumOnly = false, rawTs = '') {
        const taskState = getTaskUiState(taskId, true);
        if (!taskState) return null;
        const safeCount = Math.max(0, Number(count) || 0);
        if (minimumOnly) {
            taskState.toolCalls = Math.max(taskState.toolCalls, safeCount);
        } else {
            taskState.toolCalls += safeCount;
        }
        revealBufferedCardIfNeeded(taskState, { rawTs });
        return taskState;
    }

    function forceTaskCard(taskId, rawTs = '') {
        const taskState = getTaskUiState(taskId, true);
        if (!taskState) return null;
        taskState.forceCard = true;
        revealBufferedCardIfNeeded(taskState, { rawTs });
        return taskState;
    }

    function markAssistantReply(taskId = '') {
        const resolvedTaskId = taskId || '';
        if (!resolvedTaskId) return;
        const taskState = getTaskUiState(resolvedTaskId, false);
        if (!taskState) return;
        taskState.completed = true;
        taskState.completedPhase = taskState.completedPhase || 'done';
        if (!taskState.cardVisible) {
            scheduleTaskUiCleanup(taskState, 30000);
            return;
        }
        scheduleTaskUiCleanup(taskState);
    }

    function markTaskComplete(taskId = '', phase = '') {
        const taskState = getTaskUiState(taskId, false);
        if (!taskState) return;
        taskState.completed = true;
        if (phase) taskState.completedPhase = phase;
    }

    return {
        isBackgroundTaskId,
        shouldAlwaysShowTaskCard,
        isForegroundLiveCard,
        createTaskUiState,
        getTaskUiState,
        scheduleTaskUiCleanup,
        bufferLiveUpdate,
        markTaskToolCall,
        forceTaskCard,
        markAssistantReply,
        markTaskComplete,
    };
}
