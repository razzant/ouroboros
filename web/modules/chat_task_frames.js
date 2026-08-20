// Task-frame routing for a chat instance: projecting task_summary rows, live
// progress messages and grouped log events onto the live cards they belong
// to. Ownership transfer from chat.js (v7 W3 wave D): the per-instance
// closure bodies move here with their captured collections and collaborator
// members lifted to explicit factory parameters of the same names; the
// active-group fallback reads through the live-card store's
// getActiveLiveGroupId accessor.

import {
    OWNER_STOP_DETAIL_MARKER,
    OWNER_STOP_DONE_HEADLINE,
    formatReviewProjection,
    getLogTaskGroupId,
    isGroupedTaskEvent,
    normalizeLogTs,
    ownerHurryProjection,
    summarizeChatLiveEvent,
    taskOutcomeSeverity,
    taskStoppedWithSummary,
    taskTerminalPhase,
} from './log_events.js';
import { showContextFitToast } from './chat_notices.js';
import { taskCostProjection, withTaskCostMeta } from './costs.js';

export function createTaskFrames({
    liveCardRecords,
    subagentChildParents,
    subagentTerminalChildren,
    activeDirectActivities,
    getActiveLiveGroupId,
    registerEphemeralDecisionFrame,
    revealBufferedCardIfNeeded,
    queueTaskLiveUpdate,
    getSubagentCardRecord,
    applyLiveCardState,
    finishLiveCard,
    applySuggestedName,
    getTaskUiState,
    scheduleTaskUiCleanup,
    markTaskToolCall,
    forceTaskCard,
    markAssistantReply,
    markTaskCancelable,
    updateSubagentCardFromEvent,
    routeSubagentProgressToCard,
    routeSubagentTerminalToCard,
    recordConcludedActivity,
    syncChatStatus,
}) {
    function appendTaskSummaryToLiveCard(msg, { suppressDomInsert = false } = {}) {
        const taskId = msg?.task_id || getActiveLiveGroupId() || '';
        const rawTs = msg?.ts || new Date().toISOString();
        if (registerEphemeralDecisionFrame(msg)) return;
        if (!taskId) {
            finishLiveCard(taskId, 'done');
            return;
        }
        // Cluster B: a card (re)built from a task_summary row also carries the coined name
        // on reload (history attaches suggested_name to summary rows too) — apply it so the
        // title survives even when no progress row was retained.
        if (msg?.suggested_name) applySuggestedName(taskId, msg.suggested_name);
        const reviewDetails = formatReviewProjection(msg?.review_projection);
        const taskState = getTaskUiState(taskId, Boolean(reviewDetails));
        if (!taskState) {
            finishLiveCard(taskId, 'done');
            return;
        }
        if (reviewDetails) taskState.forceCard = true;
        revealBufferedCardIfNeeded(taskState, { suppressDomInsert, rawTs });
        if (!taskState.cardVisible) {
            markAssistantReply(taskId);
            return;
        }
        const record = liveCardRecords.get(taskId);
        const reasonCode = msg?.reason_code ? String(msg.reason_code) : '';
        const severity = taskOutcomeSeverity(msg || {});
        const terminalPhase = taskTerminalPhase(msg || {});
        const failedResult = severity === 'error';
        // P5: a cancelled root says "Cancelled", never a generic "Done" headline.
        // №8/Q3: an owner-requested soft stop is a SUCCESS — its own headline,
        // never warn-styled, with the owner-request marker in the details.
        const softStopped = taskStoppedWithSummary(msg || {});
        const doneHeadline = severity === 'cancelled'
            ? 'Cancelled'
            : (failedResult && reasonCode
                ? `Done: ${reasonCode}`
                : (softStopped
                    ? OWNER_STOP_DONE_HEADLINE
                    : (severity === 'warn'
                        ? (reasonCode ? `Finished with warnings: ${reasonCode}` : 'Finished with warnings')
                        : ((record && record.lastHumanHeadline) || 'Done'))));
        const softStopDetail = softStopped ? OWNER_STOP_DETAIL_MARKER : '';
        applyLiveCardState(
            {
                phase: terminalPhase,
                headline: doneHeadline,
                body: [softStopDetail, reviewDetails].filter(Boolean).join('\n'),
                visible: Boolean(softStopDetail || reviewDetails),
                human: false,
                promote: true,
                terminal: true,
                expandByDefault: Boolean(reviewDetails),
                costProjection: taskCostProjection(msg, rawTs),
            },
            taskId,
            normalizeLogTs(rawTs),
            `task_done|${taskId}`,
            { suppressDomInsert, rawTs },
        );
        finishLiveCard(taskId, terminalPhase);
        scheduleTaskUiCleanup(taskState);
    }

    function updateLiveCardFromProgressMessage(msg) {
        const taskId = msg?.task_id || getActiveLiveGroupId() || '';
        const rawTs = msg?.ts || new Date().toISOString();
        if (registerEphemeralDecisionFrame(msg)) return;
        if (!taskId) return;
        // P5: host-attested cancelable marker (live WS frames AND history replay
        // via _PROGRESS_META_FIELDS). The supervisor stamps it ONLY on
        // lineage-resolved non-subagent ROOTS, so the marker is the truth —
        // re-deriving rootness from frame shape would wrongly reject a
        // timeout-retry root (root_task_id names the ORIGINAL task while the
        // endpoint cancels the current id). Direct-chat turns never carry it.
        if (msg?.cancelable === true && msg?.task_id) markTaskCancelable(String(msg.task_id));
        // Subagent lifecycle pings render as child cards linked to the parent;
        // they must not update the parent card's terminal state.
        const lifecycleParent = String(msg?.parent_task_id || '').trim();
        if (
            msg?.subagent_event
            && lifecycleParent
            && updateSubagentCardFromEvent(msg, rawTs)
        ) {
            return;
        }
        // A known subagent child's own (non-lifecycle) progress stays on the child
        // card so parallel work remains visible without expanding the parent.
        if (subagentChildParents.has(taskId)) {
            routeSubagentProgressToCard(taskId, msg);
            return;
        }
        // Progress messages are visible status; do not force-open completed replay.
        const taskState = getTaskUiState(taskId, true);
        if (taskState && !taskState.completed) taskState.forceCard = true;
        const summary = summarizeChatLiveEvent({
            type: 'send_message',
            is_progress: true,
            content: msg?.content || msg?.text || '',
            text: msg?.content || msg?.text || '',
            task_id: taskId,
            subagent_event: msg?.subagent_event || '',
            subagent_task_id: msg?.subagent_task_id || '',
            root_task_id: msg?.root_task_id || '',
            parent_task_id: msg?.parent_task_id || '',
            delegation_role: msg?.delegation_role || '',
            subagent_role: msg?.subagent_role || '',
            // The resolved delegated route; without it a LIVE progress bubble drops
            // the executor chip that the same bubble regains on reload.
            executor_route: msg?.executor_route || '',
            status: msg?.status || '',
            cost_usd: msg?.cost_usd,
            accounted_upper_bound_usd: msg?.accounted_upper_bound_usd,
            accounted_upper_bound_usd_with_children: msg?.accounted_upper_bound_usd_with_children,
            cost_accounting_status: msg?.cost_accounting_status,
            cost_accounting_error: msg?.cost_accounting_error,
            cost_final: msg?.cost_final,
            cost_usd_with_children: msg?.cost_usd_with_children,
            cost_with_children_partial: msg?.cost_with_children_partial,
            reserved_usd: msg?.reserved_usd,
            unresolved_upper_bound_usd: msg?.unresolved_upper_bound_usd,
            unknown_unmetered: msg?.unknown_unmetered,
            non_final_rows: msg?.non_final_rows,
            result: msg?.result || '',
            trace_summary: msg?.trace_summary || '',
            error: msg?.error || '',
            artifact_status: msg?.artifact_status || '',
            lifecycle: msg?.lifecycle || null,
        });
        if (!summary) return;
        const presented = withTaskCostMeta(summary, msg, { rawTs });
        queueTaskLiveUpdate(presented, taskId, normalizeLogTs(rawTs), presented.dedupeKey || '', rawTs);
        // Cluster B: history progress recs carry the coined name (live progress does
        // not — the live path uses the separate `task_named` event). Apply it after the
        // card exists so a reload shows the same title.
        if (msg?.suggested_name) applySuggestedName(taskId, msg.suggested_name);
        // History projects authoritative terminal truth onto the latest progress
        // anchor when the best-effort task_summary row is absent. Apply that truth
        // before a later ordinary assistant row closes the card, so replay cannot
        // freeze a degraded review as a green completion.
        if (
            msg?.task_terminal_status
            && (msg?.outcome_axes || msg?.review_projection || msg?.reason_code)
        ) {
            appendTaskSummaryToLiveCard(msg);
        }
    }

    function updateLiveCardFromLogEvent(evt) {
        if (!evt || !isGroupedTaskEvent(evt)) return;
        showContextFitToast(evt);
        if (registerEphemeralDecisionFrame(evt)) return;
        const taskId = getLogTaskGroupId(evt) || getActiveLiveGroupId() || '';
        if (!taskId) return;
        const eventType = evt.type || evt.event || '';
        const rawTs = evt.ts || evt.timestamp || new Date().toISOString();
        if (eventType === 'owner_hurry') {
            // HQ1: compact task-card status ONLY — never a timeline row or any
            // chat bubble (the summarizer also hides this family, visible=false).
            if (ownerHurryProjection(evt).applied) {
                liveCardRecords.get(taskId)?.root?.setAttribute('data-owner-hurry', '1');
            }
            return;
        }
        // A known subagent child's log events update its linked child card.
        if (subagentChildParents.has(taskId)) {
            if (eventType === 'task_done') {
                routeSubagentTerminalToCard(taskId, evt);
                return;
            }
            if (subagentTerminalChildren.has(taskId)) return;
            if (eventType === 'tool_call_started') {
                markTaskToolCall(taskId, 1, false, rawTs);
            } else if ((eventType === 'task_metrics_event' || eventType === 'task_eval') && Number.isFinite(Number(evt.tool_calls))) {
                markTaskToolCall(taskId, Number(evt.tool_calls), true, rawTs);
            } else if (
                eventType === 'tool_call_timeout'
                || eventType === 'tool_timeout'
                || eventType === 'llm_round_error'
                || eventType === 'llm_api_error'
                || (eventType === 'tool_call_finished' && evt.is_error)
            ) {
                forceTaskCard(taskId, rawTs);
            }
            const summary = summarizeChatLiveEvent(evt);
            if (!summary) return;
            const info = subagentChildParents.get(taskId);
            if (info) getSubagentCardRecord(taskId, info.parentId, info.role);
            const presented = withTaskCostMeta(summary, evt, {
                replace: eventType === 'task_done' || eventType === 'task_cost_finalized',
                rawTs,
            });
            queueTaskLiveUpdate(presented, taskId, normalizeLogTs(rawTs), presented.dedupeKey || '', rawTs);
            return;
        }
        if (eventType === 'tool_call_started') {
            markTaskToolCall(taskId, 1, false, rawTs);
        } else if ((eventType === 'task_metrics_event' || eventType === 'task_eval') && Number.isFinite(Number(evt.tool_calls))) {
            markTaskToolCall(taskId, Number(evt.tool_calls), true, rawTs);
        } else if (
            eventType === 'tool_call_timeout'
            || eventType === 'tool_timeout'
            || eventType === 'llm_round_error'
            || eventType === 'llm_api_error'
            || (eventType === 'tool_call_finished' && evt.is_error)
        ) {
            forceTaskCard(taskId, rawTs);
        }
        if (eventType === 'task_done' && formatReviewProjection(evt.review_projection)) {
            forceTaskCard(taskId, rawTs);
        }
        const summary = summarizeChatLiveEvent(evt);
        if (!summary) return;
        const presented = withTaskCostMeta(summary, evt, {
            replace: eventType === 'task_done' || eventType === 'task_cost_finalized',
            rawTs,
        });
        queueTaskLiveUpdate(presented, taskId, normalizeLogTs(rawTs), presented.dedupeKey || '', rawTs);
        updateSubagentCardFromEvent(evt, rawTs);
        if (eventType === 'task_done') {
            // The settled task_done concludes the managed activity too: panels
            // hydrate one-shot (no poll), so the header must not stay Working.
            if (activeDirectActivities.delete(taskId)) {
                recordConcludedActivity(taskId);
                syncChatStatus();
            }
            const taskState = getTaskUiState(taskId, false);
            revealBufferedCardIfNeeded(taskState, { rawTs });
        }
    }

    return {
        appendTaskSummaryToLiveCard,
        updateLiveCardFromProgressMessage,
        updateLiveCardFromLogEvent,
    };
}
