// remote_card_ui.js — RENDERING a remote (SSH-placed) task's state in a live card.
//
// The third piece of a deliberate three-way split. `remote_task_state.js` holds the
// DECISIONS — what a status means, which actions are offered, how one connection-wide
// frame fans out across the tasks riding that connection. `connections_ui.js` is the
// owner's connection admin. This is the VIEW: it turns those decisions into buttons,
// a headline and a phase colour on a chat card, and it decides nothing.
//
// It takes its host's capabilities explicitly rather than closing over a chat module.
// That is what keeps the boundary honest: everything this file is allowed to touch is
// named in one object, so "the card renderer reached into chat state" cannot happen by
// accident, and the same view can be driven by a test with a stub host.

import {
    reduceRemoteConnectionEvent,
    remoteActionErrorText,
    remoteDetailText,
    remoteReconnectNotice,
    remoteStateDetails,
    remoteStateLabel,
    remoteStateNote,
    remoteStateSummary,
    remoteTaskActions,
} from './remote_task_state.js';

/**
 * @param {object} host
 * @param {() => Map} host.getStates          current taskId -> remote state map
 * @param {(states: Map) => void} host.setStates  the reducer returns a NEW map
 * @param {() => string} host.getProjectId    this thread's project, for scope checks
 * @param {Map} host.liveCardRecords          taskId -> card record
 * @param {Function} host.forceTaskCard
 * @param {Function} host.queueTaskLiveUpdate
 * @param {Function} host.normalizeLogTs
 * @param {Function} host.setLiveCardTypingVisible
 * @param {Function} host.cancelTask
 * @param {Function} host.showToast
 * @param {Function} host.openConfirmDialog
 * @param {object} host.apiClient
 */
export function createRemoteCardUi(host) {
    function remoteButton(container, label, className, onActivate) {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = className;
        button.textContent = label;
        button.addEventListener('click', async (event) => {
            event.stopPropagation();
            button.disabled = true;
            try {
                await onActivate();
            } catch (error) {
                host.showToast(remoteActionErrorText(error), 'error');
                throw error;
            } finally {
                button.disabled = false;
            }
        });
        container.appendChild(button);
    }

    async function cancelRemoteTask(record, state) {
        // CASCADE, like every other cancel button on this surface: this file's own
        // "Cancel run" and the Activity tab's `task-cancel` both pass it, and the
        // endpoint's default is single-task (kept that way for headless callers). This
        // one omitted it — it could not even EXPRESS it, since it went through the
        // duplicate `apiClient.taskCancel` spelling that took no options — so
        // cancelling a remote ORCHESTRATOR from its own card left its subagents
        // running on the target. The prompt now says what will happen, in the same
        // words `activity.js` uses for the same action on the same task — and through
        // the same in-app dialog, since no surface here opens a native one (v6.90.3).
        const confirmedRemoteCancel = await host.openConfirmDialog({
            title: 'Cancel task',
            body: 'Cancel this task and all its subagents?',
            confirmLabel: 'Cancel task',
            cancelLabel: 'Keep running',
            danger: true,
        });
        if (!confirmedRemoteCancel) return;
        // Answered only once the subtree teardown has finished, so a resolved promise
        // means the run is really down. The optimistic status written here used to be
        // `cancel_requested`, which is in NEITHER `CANCELLABLE_TASK_STATES` nor
        // `TERMINAL_TASK_STATES` (`remote_task_state.js`) — so the card landed in a
        // limbo that offered no Cancel and never read as finished. After a settled
        // cascade the honest status is the terminal one.
        await host.cancelTask(state.taskId, { cascade: true });
        const states = host.getStates();
        states.set(state.taskId, {
            ...state,
            taskStatus: 'cancelled',
            completion: 'cancelled',
        });
        host.showToast('Task and subagents cancelled.', 'success');
        updateRemoteCardActions(record, states.get(state.taskId));
    }

    async function reconnectRemoteTask(state) {
        const frame = {
            connection_id: state.connectionId,
            task_id: state.taskId,
            project_id: state.projectId,
        };
        applyRemoteConnectionEvent({
            ...frame, status: 'connecting', phase: 'connect', completion: 'testing',
        }, { bypassScopeCheck: true });
        try {
            const result = await host.apiClient.connectionReconnect(state.connectionId);
            applyRemoteConnectionEvent({ ...result, ...frame }, { bypassScopeCheck: true });
            host.showToast(remoteReconnectNotice(host.getStates().get(state.taskId)), 'success');
        } catch (error) {
            // Repaint the task's own row from the failure, not just a toast.
            applyRemoteConnectionEvent(
                { ...(error?.body || {}), ...frame, status: 'degraded' },
                { bypassScopeCheck: true },
            );
            throw error;
        }
    }

    function updateRemoteCardActions(record, remoteState) {
        if (!record || record.isSubagent || !remoteState) return;
        let container = record.root.querySelector('[data-remote-task-actions]');
        if (!container) {
            container = document.createElement('div');
            container.className = 'chat-live-actions chat-live-remote-actions';
            container.dataset.remoteTaskActions = '1';
            record.timelineEl?.insertAdjacentElement('beforebegin', container);
        }
        container.replaceChildren();
        const actions = remoteTaskActions(remoteState);
        if (actions.canReconnect) {
            remoteButton(container, 'Reconnect', 'btn btn-xs btn-default',
                () => reconnectRemoteTask(remoteState));
        }
        if (actions.canCancel) {
            remoteButton(container, 'Cancel', 'btn btn-xs btn-danger',
                () => cancelRemoteTask(record, remoteState));
        }
        container.hidden = container.childElementCount === 0;
    }

    /**
     * Fold a `connection_state` frame into this chat's remote task states and
     * repaint the affected cards. A frame naming another Project's task is
     * ignored unless it is this instance's own optimistic echo
     * (`bypassScopeCheck`), so a Project thread never renders foreign work.
     */
    function applyRemoteConnectionEvent(event, { bypassScopeCheck = false } = {}) {
        const explicitTaskId = String(event?.task_id || '').trim();
        const eventProjectId = String(event?.project_id || '').trim();
        const projectId = host.getProjectId();
        const current = host.getStates();
        if (!bypassScopeCheck && explicitTaskId) {
            if (eventProjectId) {
                if (!projectId || eventProjectId !== projectId) return;
            } else if (
                !current.has(explicitTaskId)
                && !host.liveCardRecords.has(explicitTaskId)
            ) {
                return;
            }
        }
        const { states, taskIds } = reduceRemoteConnectionEvent(
            current,
            event,
            (taskId) => ({
                task_id: taskId,
                project_id: eventProjectId || current.get(taskId)?.projectId || projectId,
                status: current.get(taskId)?.taskStatus || event?.completion || '',
            }),
        );
        host.setStates(states);
        for (const taskId of taskIds) paintRemoteTaskState(taskId, states.get(taskId), event);
    }

    function paintRemoteTaskState(taskId, state, event = {}) {
        if (!state) return;
        host.forceTaskCard(taskId, event?.ts || new Date().toISOString());
        const details = remoteStateDetails(state);
        const terminalReadyNote = (
            remoteTaskActions(state).terminal && state.status === 'ready'
        )
            ? remoteReconnectNotice(state)
            : '';
        host.queueTaskLiveUpdate({
            phase: ['degraded', 'disconnected'].includes(state.status)
                ? 'error'
                : (state.status === 'unknown' ? 'warn' : 'working'),
            headline: `Remote connection: ${remoteStateLabel(state.status)}`,
            body: [remoteStateSummary(state), remoteStateNote(state), terminalReadyNote]
                .filter(Boolean).join('\n'),
            fullBody: details.map((item) => (
                `[${item.label}]\n${remoteDetailText(item.value)}`
            )).join('\n\n'),
            visible: true,
            promote: true,
            meta: [
                `SSH ${state.connectionId}`,
                state.phase ? `phase=${state.phase}` : '',
                state.completion ? `completion=${state.completion}` : '',
            ].filter(Boolean),
            dedupeKey: `remote-connection:${taskId}`,
        }, taskId, host.normalizeLogTs(event?.ts || new Date().toISOString()),
        `remote-connection:${taskId}`, event?.ts || '');
        const record = host.liveCardRecords.get(taskId);
        if (record) {
            record.root.dataset.remoteState = state.status;
            // The typing indicator is "work is happening": a degraded or
            // terminal remote task must not keep pulsing as if it were.
            host.setLiveCardTypingVisible(
                record,
                ['connecting', 'ready'].includes(state.status) && !remoteTaskActions(state).terminal,
            );
            updateRemoteCardActions(record, state);
        }
    }

    /**
     * A task reached a terminal status. The CONNECTION state is unchanged by that,
     * so only the task-side fields move — which is what stops a finished remote task
     * from still offering Cancel.
     */
    function settleTerminalTask(taskId, taskStatus) {
        const states = host.getStates();
        const remoteState = states.get(taskId);
        if (!remoteState) return;
        const settled = { ...remoteState, taskStatus, completion: taskStatus };
        states.set(taskId, settled);
        updateRemoteCardActions(host.liveCardRecords.get(taskId), settled);
    }

    /**
     * A rebuild destroys every card's DOM, so the remote state that DESCRIBED those
     * cards has to be reconciled with the new ones. It used to be in neither list —
     * not cleared, not repainted — so after a soft reconnect the map survived while
     * the elements it had painted were gone: a live-but-unrendered state that lost the
     * Reconnect and Cancel buttons, the "Remote connection:" row and the phase colour,
     * and left a degraded remote task pulsing its typing dots as if work were
     * happening. Repaint what still has a card; DROP what does not, since a
     * pre-disconnect status for a card that no longer exists is stale rather than
     * merely unrendered.
     */
    function reconcileAfterRebuild() {
        const states = host.getStates();
        if (!states.size) return;
        for (const [taskId, state] of Array.from(states)) {
            if (host.liveCardRecords.has(taskId)) paintRemoteTaskState(taskId, state);
            else states.delete(taskId);
        }
    }

    return {
        applyRemoteConnectionEvent,
        updateRemoteCardActions,
        paintRemoteTaskState,
        settleTerminalTask,
        reconcileAfterRebuild,
    };
}
