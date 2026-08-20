import { apiClient, apiFetch, fetchTaskDetail } from './api_client.js';
import {
    taskCancelPending,
    taskSoftStopPending,
    taskTerminalPhase,
} from './log_events.js';
import {
    ACTION_FINALIZE,
    ACTION_HURRY,
    TASK_CONTROL_TRIGGER_LABEL,
    cancelRunEligibility,
    hurryTaskAction,
    openTaskControlMenu,
    requestStop,
    taskControlBusy,
} from './task_control_menu.js';
import { showToast } from './toast.js';

function projectIdFromTask(taskId = '') {
    const seed = String(taskId || '')
        .toLowerCase()
        .replace(/[^a-z0-9_.-]+/g, '-')
        .replace(/^-+|-+$/g, '');
    return (seed ? `task-${seed}` : `task-${Date.now().toString(36)}`).slice(0, 64);
}

// Published for the owner's characterization test; chat.js reaches it only
// through turnTaskIntoProject.
export { projectIdFromTask };

// The two owner actions a live task card offers, for ONE chat instance: cancel
// the run (the host-attested marker decides whether the trigger is even
// mounted, and the dropdown's three actions settle through the durable record)
// and turn the task into a project (a one-click conversion that hands the task
// to the project panel and recolors the card into a project chip). Both write
// only through the card record and the instance's cancelable-id set, which —
// together with the record map, the viewport wrapper, the terminal seam and the
// composer cue — are handed over explicitly.
export function createCardActions({
    liveCardRecords,
    cancelableTaskIds,
    withStableViewport,
    finishLiveCard,
    signalChatFreed,
}) {
    async function turnTaskIntoProject(record) {
        if (!record || record.root?.dataset?.projectCreating === '1' || record.root?.dataset?.projectCreated === '1') return;
        const taskId = String(record.groupId || '').trim();
        const projectId = projectIdFromTask(taskId);
        record.root.dataset.projectCreating = '1';
        const actions = record.turnProjectBtn?.parentElement || record.root.querySelector('.chat-live-actions');
        if (actions) {
            withStableViewport(() => {
                actions.innerHTML = '<button type="button" class="chat-live-project-btn" disabled>Creating project…</button>';
                record.cancelRunBtn = null;
            });
        }
        try {
            // One-click convert (owner P1): no name prompt, no extra LLM call.
            // The SERVER derives the project name (gateway/projects.py
            // _derive_project_name: title -> objective -> queue snapshot). We also
            // hand it the owner's original request as a fallback hint so a still
            // in-progress DIRECT chat task — which has no server-side title/objective
            // yet — is named from what the owner asked, not "New project".
            const payload = await apiClient.projectFromTask(taskId, projectId, '', record.objectiveHint || '');
            const project = payload.project || { id: projectId, name: projectId };
            showToast(`Project created: ${project.name || project.id}`, 'ok');
            window.dispatchEvent(new CustomEvent('ouro:project-created', { detail: { project } }));
            markCardConverted(record, project);
        } catch (exc) {
            showToast(`Project creation failed: ${exc.message || exc}`, 'error');
            delete record.root.dataset.projectCreating;
            if (actions) {
                withStableViewport(() => {
                    actions.innerHTML = '<button type="button" class="chat-live-project-btn" data-turn-into-project>Turn into project</button>';
                    record.turnProjectBtn = actions.querySelector('[data-turn-into-project]');
                    // Re-wire the click handler — innerHTML replaced the original node,
                    // so without this the restored button would be dead after a
                    // transient failure (T5).
                    record.turnProjectBtn?.addEventListener('click', (event) => {
                        event.stopPropagation();
                        turnTaskIntoProject(record);
                    });
                    // P5: innerHTML also dropped a rendered "Cancel run" — restore it.
                    record.cancelRunBtn = null;
                    syncCancelRunButton(record);
                });
            }
        }
    }

    // v6.82 (P5): "Cancel run" on live pooled ROOT cards. Forced cancel of the
    // selected task AND its live subtree (explicit cascade — the endpoint's
    // default stays single-task for headless callers). Gated on the supervisor's
    // host-attested `cancelable` marker so a direct-chat turn (which mints a
    // card of the same shape but has no queue entry) never shows a dead button.
    function ensureLiveActionsEl(record) {
        if (!record?.root
            || record.root.dataset.projectCreated === '1'
            || record.root.dataset.projectCreating === '1') return null;
        let actions = record.root.querySelector('.chat-live-actions');
        if (!actions) {
            actions = document.createElement('div');
            actions.className = 'chat-live-actions';
            const timeline = record.timelineEl && record.timelineEl.parentElement === record.root
                ? record.timelineEl
                : null;
            record.root.insertBefore(actions, timeline);
        }
        return actions;
    }

    function syncCancelRunButton(record) {
        return withStableViewport(() => syncCancelRunButtonMutation(record));
    }

    function syncCancelRunButtonMutation(record) {
        if (!record?.root) return;
        const eligible = cancelRunEligibility({
            groupId: record.groupId,
            isSubagent: record.isSubagent,
            finished: record.finished,
            cancelable: cancelableTaskIds.has(record.groupId),
            converted: record.root.dataset.projectCreated === '1',
        });
        const existing = record.root.querySelector('[data-cancel-run]');
        if (!eligible) {
            existing?.remove();
            record.cancelRunBtn = null;
            return;
        }
        if (existing) {
            record.cancelRunBtn = existing;
            return;
        }
        const actions = ensureLiveActionsEl(record);
        if (!actions) return;
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'btn btn-xs btn-danger';
        btn.dataset.cancelRun = '1';
        btn.textContent = TASK_CONTROL_TRIGGER_LABEL;
        // S3 (Q2/HQ1): the trigger opens the three-action dropdown; dismissing
        // it continues the run. While a cancel intent is pending the menu
        // offers ONLY the hard escalation ("Stop now").
        btn.addEventListener('click', (event) => {
            event.stopPropagation();
            openTaskControlMenu(btn, {
                cancelPending: Boolean(record.cancelPendingPolicy),
                busy: taskControlBusy(record.groupId),
                onAction: (action) => (action === ACTION_HURRY
                    ? hurryTaskAction(record.groupId)
                    : cancelRunFromCard(record, action)),
            });
        });
        actions.appendChild(btn);
        record.cancelRunBtn = btn;
    }

    // Interim "Cancelling…" phase (phase A cancel redesign): the durable cancel
    // intent is recorded and the supervisor is confirming the teardown — the
    // card stays honestly LIVE (never an instant "Cancelled" lie) and resolves
    // on the settled task_done: Cancelled, or Completed when the run finished
    // first (completion wins). S3 (Q1): a pending SOFT stop shows "Finalizing…"
    // instead — a bounded final turn is running before the same intent settles.
    function markLiveCardCancelPending(taskId = '', soft = false) {
        const record = liveCardRecords.get(String(taskId || '').trim());
        if (!record || record.finished || !record.phaseEl) return;
        record.cancelPendingPolicy = soft ? 'finalize' : 'immediate';
        record.finalizingHold = false;  // owner cancel outranks the hold
        record.phaseEl.dataset.phase = 'working';
        record.phaseEl.textContent = soft ? 'Finalizing…' : 'Cancelling…';
        record.phaseEl.className = 'chat-live-phase working cancelling';
    }

    // Early final on a managed root: hold the card on a sticky "Finalizing…"
    // until the settled task_done (post-task synthesis still runs).
    function markLiveCardFinalizing(taskId = '') {
        const record = liveCardRecords.get(String(taskId || '').trim());
        if (!record || record.finished || !record.phaseEl) return;
        if (record.cancelPendingPolicy) return;
        record.finalizingHold = true;
        record.phaseEl.dataset.phase = 'working';
        record.phaseEl.textContent = 'Finalizing…';
        record.phaseEl.className = 'chat-live-phase working finalizing';
    }

    // Snapshot / restore of the live phase element around the optimistic
    // "Cancelling…" mark (GR2-8a): a cancel request that FAILS must not leave
    // the optimistic phase lying on a card whose cancellation is not pending.
    function captureLiveCardPhase(record) {
        if (!record?.phaseEl) return null;
        return {
            phase: record.phaseEl.dataset.phase,
            text: record.phaseEl.textContent,
            className: record.phaseEl.className,
        };
    }

    function restoreLiveCardPhase(record, snapshot) {
        if (!record?.phaseEl || !snapshot || record.finished) return;
        record.phaseEl.dataset.phase = snapshot.phase;
        record.phaseEl.textContent = snapshot.text;
        record.phaseEl.className = snapshot.className;
    }

    // Task-detail reconciliation for the cancel flow (GR2-8b): the typed
    // cancel_state projection is consulted FIRST — a live task wedged in the
    // legacy `cancel_requested` STATUS latch (intent, not outcome) must show
    // as cancel-pending, not resolve as a terminal "Cancelled" while the
    // supervisor is still tearing it down. Only genuinely settled statuses
    // (or an intent-free legacy latch, which is history awaiting boot
    // migration) fall through to the terminal seam.
    function reconcileCancelCardFromDetail(record, taskId, stored) {
        if (!stored || record.finished) return;
        if (taskCancelPending(stored)) {
            markLiveCardCancelPending(taskId, taskSoftStopPending(stored));
            return;
        }
        const status = String(stored?.status || '');
        if (['completed', 'failed', 'cancelled', 'cancel_requested', 'rejected_duplicate'].includes(status)) {
            finishLiveCard(taskId, taskTerminalPhase(stored));
        }
    }

    async function cancelRunFromCard(record, action = '') {
        const taskId = String(record?.groupId || '').trim();
        if (!taskId || record.finished) return;
        // Q2: the dropdown itself is the confirmation surface — dismissing it
        // continued the run, so a selected action executes immediately.
        const soft = action === ACTION_FINALIZE;
        const btn = record.cancelRunBtn;
        if (btn) btn.disabled = true;
        const priorPhase = captureLiveCardPhase(record);
        markLiveCardCancelPending(taskId, soft);
        try {
            // Immediate: answered only after the teardown finished, so a resolved
            // promise means the run is really down. Soft (Q1): a 202 arrives with
            // the durable intent open while the bounded finalization runs — the
            // card stays "Finalizing…". A refusal throws and is toasted below.
            await requestStop(taskId, action);
            // Backend publication is fail-soft past the durable boundary, so a 200
            // can arrive with the task_done event lost. Reconcile from the durable
            // record through the same terminal seam replay uses — idempotent with
            // a later event, so double resolution is harmless.
            try {
                reconcileCancelCardFromDetail(record, taskId, await fetchTaskDetail(taskId));
            } catch {
                // The card still resolves on its own frame if one arrives.
            }
            // Immediate: the card resolves via the existing task_done frames and
            // the button stays disabled until then. Soft: the hard escalation
            // must stay REACHABLE during the wait (Q1), so re-enable the trigger
            // (the pending menu offers only "Stop now").
            if (btn && !record.finished && record.cancelPendingPolicy === 'finalize') {
                btn.disabled = false;
            }
        } catch (exc) {
            // 404 = nothing live anymore (natural completion beat the cancel):
            // graceful no-op, the card resolves on its own terminal frame.
            if (exc?.status === 404 || record.finished) {
                // Completion-wins race: the run finished while the request was in
                // flight, so there is nothing to cancel. RESYNC rather than leave a
                // dead disabled button — the card resolves on its own terminal
                // frame, and until then the action is simply no longer offered.
                // `cancelableTaskIds` is the eligibility AUTHORITY — clearing the
                // record flag alone left the button mounted and merely re-enabled.
                cancelableTaskIds.delete(taskId);
                record.cancelable = false;
                syncCancelRunButton(record);
                // REAL resync, not just button removal: 404 says the task is no
                // longer live, but if its terminal frame was lost this card would
                // sit "Working" forever. Ask the durable record and resolve the
                // card through the same terminal seam replay uses.
                try {
                    reconcileCancelCardFromDetail(record, taskId, await fetchTaskDetail(taskId));
                } catch {
                    // The card still resolves on its own frame if one arrives;
                    // nothing worse than the pre-resync behavior.
                }
                return;
            }
            showToast(`Cancel failed: ${exc?.message || exc}`, 'error');
            // GR3-10: reconcile the durable detail BEFORE touching the button —
            // a non-404 failure can sit over a task whose durable record is
            // already terminal (finish the card, button stays gone) or whose
            // durable intent really is pending (keep the button disabled and
            // the honest "Cancelling…"). Only a genuinely-live, non-pending
            // task gets its prior phase restored and the button re-enabled.
            let stored = null;
            try {
                stored = await fetchTaskDetail(taskId);
            } catch {
                // Typed state unreachable — handled by the null guard below.
            }
            if (stored === null) {
                // GR4-5: the detail fetch itself failed, so NOTHING was proven —
                // restoring the prior phase and re-enabling Cancel would assert
                // "not pending" without evidence. Keep the pending presentation
                // and the disabled button; the next reconcile/poll (or the
                // task_done frame) resolves the card either way.
                return;
            }
            // The shared seam: pending keeps the interim, a terminal record
            // finishes the card (same path replay uses).
            reconcileCancelCardFromDetail(record, taskId, stored);
            const stillPending = Boolean(taskCancelPending(stored));
            if (record.finished || stillPending) return;
            // Only a fetched, live, non-pending detail restores the button.
            if (btn) btn.disabled = false;
            restoreLiveCardPhase(record, priorPhase);
            record.cancelPendingPolicy = '';
        }
    }

    function markTaskCancelable(taskId = '') {
        const id = String(taskId || '').trim();
        if (!id || cancelableTaskIds.has(id)) return;
        cancelableTaskIds.add(id);
        const record = liveCardRecords.get(id);
        if (record) syncCancelRunButton(record);
    }

    // One-way conversion (P3): the WHOLE card becomes a calm "project identity"
    // chip. The live task is now owned by the project panel (it's bound there),
    // so the main chat is freed — the card stops being a busy red task and
    // recolors to the project fuchsia. Plain wording (no "ack"); click opens the panel.
    function markCardConverted(record, project) {
        return withStableViewport(() => markCardConvertedMutation(record, project));
    }

    function markCardConvertedMutation(record, project) {
        delete record.root.dataset.projectCreating;
        record.root.dataset.projectCreated = '1';
        record.root.dataset.projectId = project.id || '';
        const name = String(project.name || project.id || 'Project').trim();
        const chip = document.createElement('button');
        chip.type = 'button';
        chip.className = 'chat-live-project-card-btn';
        const icon = document.createElement('span');
        icon.className = 'chat-live-project-icon';
        icon.setAttribute('aria-hidden', 'true');
        icon.textContent = '📁';
        const nameEl = document.createElement('span');
        nameEl.className = 'chat-live-project-name';
        nameEl.textContent = name;  // textContent — no HTML injection from a project name
        const status = document.createElement('span');
        status.className = 'chat-live-project-status';
        status.textContent = 'running in background ↗';
        chip.append(icon, nameEl, status);
        chip.addEventListener('click', () => {
            window.dispatchEvent(new CustomEvent('ouro:open-project', { detail: { project } }));
        });
        // Atomic detach-and-reparent (C4.5): replaceChildren swaps the whole live
        // timeline (subagent cards, working bubble) for the chip in one paint.
        record.root.replaceChildren(chip);
        record.turnProjectBtn = null;
        record.cancelRunBtn = null;
        record.finished = true;
        // Recolor on the next frame so the 250ms fuchsia fade actually animates.
        requestAnimationFrame(() => record.root.classList.add('is-project'));
        signalChatFreed();  // subtle "this chat is free again" composer cue
    }

    return {
        turnTaskIntoProject,
        ensureLiveActionsEl,
        syncCancelRunButton,
        markLiveCardCancelPending,
        markLiveCardFinalizing,
        captureLiveCardPhase,
        restoreLiveCardPhase,
        reconcileCancelCardFromDetail,
        cancelRunFromCard,
        markTaskCancelable,
        markCardConverted,
    };
}
