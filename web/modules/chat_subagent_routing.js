import { normalizeLogTs, summarizeChatLiveEvent, taskOutcomeSeverity } from './log_events.js';

// Subagent card routing for ONE chat instance: the child -> parent/role/model
// registry learned from lifecycle pings, and the four entry points that turn a
// subagent frame (lifecycle event, narration, final message, terminal log row)
// into an update on the child's own card. A child card is mounted under its
// parent but keeps independent phase state, so a finished child never marks the
// parent done and a late narration never revives a terminal child. The registry
// maps and the card/task helpers the routes call are handed over explicitly.
export function createSubagentRouting({
    subagentChildParents,
    subagentTerminalChildren,
    withTaskCostMeta,
    forceTaskCard,
    getTaskUiState,
    getSubagentCardRecord,
    queueTaskLiveUpdate,
}) {
    // E2 (v6.39 UI): merge a subagent's parent/role/model, PRESERVING a previously-seen model
    // when a later (model-less) event — e.g. a synthesized terminal — updates the entry, so the
    // "role · model" headline survives the child's lifecycle.
    function setSubagentParent(childId, { parentId = '', role = '', model = '' } = {}) {
        const prev = subagentChildParents.get(childId) || {};
        subagentChildParents.set(childId, {
            parentId: parentId || prev.parentId || '',
            role: role || prev.role || '',
            model: String(model || '').trim() || prev.model || '',
        });
    }

    function summarizeSubagentCardFrame(evt, overrides = {}, rawTs = '') {
        const summary = summarizeChatLiveEvent({
            ...evt,
            type: 'send_message',
            is_progress: true,
            delegation_role: 'subagent',
            ...overrides,
        });
        return summary ? withTaskCostMeta(summary, evt, { rawTs }) : null;
    }

    function updateSubagentCardFromEvent(evt, tsValue) {
        if (!evt || String(evt.delegation_role || '').toLowerCase() !== 'subagent') return false;
        const parentId = String(evt.parent_task_id || '').trim();
        const childId = String(evt.subagent_task_id || evt.task_id || '').trim();
        if (!parentId || !childId || parentId === childId) return false;
        const event = String(evt.subagent_event || '').toLowerCase();
        const role = String(evt.subagent_role || '').trim();
        setSubagentParent(childId, { parentId, role, model: evt.model });
        // Worker narration carries subagent_event="progress" too. It is activity,
        // not a lifecycle row: route it through the progress key so the later
        // terminal frame cannot overwrite the only full narration disclosure.
        if (![
            'scheduled', 'running', 'completed', 'completed_warn',
            'failed', 'cancelled', 'rejected', 'interrupted',
        ].includes(event)) {
            routeSubagentProgressToCard(childId, evt);
            return true;
        }
        const { model } = subagentChildParents.get(childId) || {};
        const rawTs = tsValue || new Date().toISOString();
        const summary = summarizeSubagentCardFrame(evt, {
            subagent_task_id: childId,
            parent_task_id: parentId,
            subagent_role: role,
            model,
        }, rawTs);
        if (!summary) return false;
        summary.dedupeKey = `subagent-lifecycle:${childId}`;
        // Interrupted is retryable and therefore non-terminal; the canonical
        // projector owns that distinction for both live and replay paths.
        if (summary.terminal) subagentTerminalChildren.add(childId);
        forceTaskCard(parentId, tsValue);
        const childState = getTaskUiState(childId, true);
        if (childState && !childState.completed) childState.forceCard = true;
        getSubagentCardRecord(childId, parentId, role);
        queueTaskLiveUpdate(
            summary,
            childId,
            normalizeLogTs(rawTs),
            summary.dedupeKey,
            rawTs,
        );
        return true;
    }

    // A known child's own (non-lifecycle) progress updates the linked child card.
    function routeSubagentProgressToCard(childId, msg) {
        const info = subagentChildParents.get(childId);
        if (!info) return;
        const { parentId, role, model } = info;
        const content = String(msg?.content || msg?.text || '').trim();
        if (!content) return;
        const rawTs = msg?.ts || new Date().toISOString();
        forceTaskCard(parentId, rawTs);
        const childState = getTaskUiState(childId, true);
        if (childState && !childState.completed) childState.forceCard = true;
        const record = getSubagentCardRecord(childId, parentId, role);
        const preserveTerminal = Boolean(record?.finished && subagentTerminalChildren.has(childId));
        const summary = summarizeSubagentCardFrame(msg, {
            content,
            text: content,
            subagent_event: 'running',
            subagent_task_id: childId,
            parent_task_id: parentId,
            subagent_role: role,
            model,
            // A replayed progress row may follow a terminal record because the
            // history pre-pass already knows the child's final state. Do not add
            // contradictory `status=running` metadata in that case.
            status: preserveTerminal ? '' : (msg?.status || ''),
        }, rawTs);
        if (!summary) return;
        summary.dedupeKey = `subagent-progress:${childId}`;
        if (preserveTerminal) {
            summary.phase = String(record.phaseEl?.dataset?.phase || 'done');
            summary.headline = String(record.titleEl?.textContent || summary.headline);
            summary.fullHeadline = summary.headline;
            summary.terminal = true;
        }
        queueTaskLiveUpdate(summary, childId, normalizeLogTs(rawTs), summary.dedupeKey, rawTs);
    }

    function routeSubagentFinalMessageToCard(taskId, msg) {
        const childId = String(taskId || '').trim();
        const info = subagentChildParents.get(childId);
        if (!childId || !info) return false;
        const { parentId, role, model } = info;
        const text = String(msg?.content || msg?.text || '').trim();
        const rawTs = msg?.ts || new Date().toISOString();
        forceTaskCard(parentId, rawTs);
        const record = getSubagentCardRecord(childId, parentId, role);
        const priorTerminalPhase = record?.finished ? String(record.phaseEl?.dataset?.phase || '') : '';
        const summary = summarizeSubagentCardFrame(msg, {
            content: '',
            text: '',
            result: text,
            subagent_event: 'completed',
            subagent_task_id: childId,
            parent_task_id: parentId,
            subagent_role: role,
            model,
        }, rawTs);
        if (!summary) return false;
        summary.dedupeKey = `subagent-result:${childId}`;
        if (priorTerminalPhase) {
            summary.phase = priorTerminalPhase;
            summary.headline = String(record.titleEl?.textContent || summary.headline);
            summary.fullHeadline = summary.headline;
            summary.terminal = true;
        }
        queueTaskLiveUpdate(summary, childId, normalizeLogTs(rawTs), summary.dedupeKey, rawTs);
        return true;
    }

    // Resolve a child's card from the child's terminal task_done
    // (which arrives on the log channel without subagent metadata).
    function routeSubagentTerminalToCard(childId, evt) {
        const info = subagentChildParents.get(childId);
        if (!info) return false;
        const status = String(evt.status || '').toLowerCase();
        const severity = taskOutcomeSeverity(evt);
        const failed = severity === 'error' || status === 'failed';
        const cancelled = status === 'cancelled' || status === 'cancel_requested';
        const rejected = status === 'rejected_duplicate';
        const event = failed ? 'failed' : cancelled ? 'cancelled' : rejected ? 'rejected' : (severity === 'warn' ? 'completed_warn' : 'completed');
        updateSubagentCardFromEvent({
            delegation_role: 'subagent',
            parent_task_id: info.parentId,
            subagent_task_id: childId,
            subagent_role: info.role,
            subagent_event: event,
            model: info.model || '',
            review_projection: evt.review_projection,
            result: evt.result || '',
            error: evt.error || '',
            cost_usd: evt.cost_usd,
            accounted_upper_bound_usd: evt.accounted_upper_bound_usd,
            accounted_upper_bound_usd_with_children: evt.accounted_upper_bound_usd_with_children,
            cost_accounting_status: evt.cost_accounting_status,
            cost_accounting_error: evt.cost_accounting_error,
            cost_final: evt.cost_final,
            cost_usd_with_children: evt.cost_usd_with_children,
            cost_with_children_partial: evt.cost_with_children_partial,
            reserved_usd: evt.reserved_usd,
            unresolved_upper_bound_usd: evt.unresolved_upper_bound_usd,
            unknown_unmetered: evt.unknown_unmetered,
            non_final_rows: evt.non_final_rows,
        }, evt.ts || evt.timestamp || new Date().toISOString());
        return true;
    }

    return {
        setSubagentParent,
        summarizeSubagentCardFrame,
        updateSubagentCardFromEvent,
        routeSubagentProgressToCard,
        routeSubagentFinalMessageToCard,
        routeSubagentTerminalToCard,
    };
}
