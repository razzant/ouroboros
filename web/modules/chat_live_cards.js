// The live-card store for a chat instance: minting and reusing card records,
// child-card adoption, buffered reveal, chronology re-anchoring, the live
// update application pipeline and the terminal transitions. Ownership transfer
// from chat.js (v7 W3 wave D): the per-instance closure bodies move here with
// their captured collections and helpers lifted to explicit factory parameters
// of the same names. The per-instance mutable flags this cluster wrote
// (activeLiveGroupId, the pending project objective, pass-1 suppression, the
// terminal-attention latch, the nested-subagent default and the card-domain
// destroyed latch) become factory state reachable through the returned
// accessors; the rebuildAll replay-batch handle stays with chat.js's
// syncHistory and is consulted here through the getRebuildBatch parameter.
// Collaborators from the mutually-recursive factories (task tracker, card
// view, card actions) arrive late through bindLiveCardCollaborators, exactly
// once, before any event can fire.

import { escapeHtmlAttr } from './utils.js';
import { apiFetch } from './api_client.js';
import { REUSABLE_TASK_IDS } from './task_control_menu.js';
import {
    boundActivityPreview,
    clearStickyCardState,
    isTerminalTaskPhase,
    liveLineRowToggleKey,
    projectCollapsedActivity,
} from './chat_card_state.js';
import { mergeStickyCostMeta } from './costs.js';

export function createChatLiveCards({
    liveCardRecords,
    taskUiStates,
    retiredTaskIds,
    stickyExpandedSlots,
    pendingSuggestedNames,
    ephemeralDecisionTaskIds,
    cancelableTaskIds,
    subagentChildParents,
    isMain,
    withStableViewport,
    insertMessageNode,
    stampNodeTimestamp,
    hideTypingIndicatorOnly,
    syncChatStatus,
    scheduleHistorySync,
    hasActiveLiveCard,
    getRebuildBatch,
}) {
    // Late-bound collaborators: these factories and this one need each other,
    // so chat.js constructs them all and then binds this set exactly once.
    let isBackgroundTaskId, shouldAlwaysShowTaskCard, getTaskUiState,
        bufferLiveUpdate, markTaskComplete,
        turnTaskIntoProject, syncCancelRunButton,
        renderCollapsedActivity, ensureSubagentContainer, setLiveCardTypingVisible,
        formatLiveCardPhaseLabel, setLiveCardExpanded, syncLiveCardToggle,
        directSubagentCount, renderLiveCardTimeline, appendTimelineItem,
        patchLastTimelineItem, patchTimelineItemAt, renderLiveCardMeta;
    function bindLiveCardCollaborators(deps) {
        ({
            isBackgroundTaskId, shouldAlwaysShowTaskCard, getTaskUiState,
            bufferLiveUpdate, markTaskComplete,
            turnTaskIntoProject, syncCancelRunButton,
            renderCollapsedActivity, ensureSubagentContainer, setLiveCardTypingVisible,
            formatLiveCardPhaseLabel, setLiveCardExpanded, syncLiveCardToggle,
            directSubagentCount, renderLiveCardTimeline, appendTimelineItem,
            patchLastTimelineItem, patchTimelineItemAt, renderLiveCardMeta,
        } = deps);
    }

    // The owner's last main-chat request, handed to the next live card it spawns so a
    // "turn into project" conversion can name the project from it (P1).
    let _pendingCardObjective = '';
    let activeLiveGroupId = '';
    let nestedSubagentsExpanded = false;
    let lastTerminalAttention = false;
    // Pass 1 builds live cards in memory; pass 2 inserts them in transcript order.
    let _syncPass1Active = false;
    // Mirrors the instance's destroyed latch so late async continuations no-op.
    let destroyed = false;

    function registerEphemeralDecisionFrame(frame) {
        return withStableViewport(() => registerEphemeralDecisionFrameMutation(frame));
    }

    function registerEphemeralDecisionFrameMutation(frame) {
        const taskId = String(frame?.task_id || '').trim();
        if (!taskId) return false;
        if (frame?.ephemeral_decision) {
            ephemeralDecisionTaskIds.add(taskId);
            const taskState = taskUiStates.get(taskId);
            if (taskState?.cleanupTimer) clearTimeout(taskState.cleanupTimer);
            taskUiStates.delete(taskId);
            const record = liveCardRecords.get(taskId);
            if (record) {
                record.root?.remove();
                liveCardRecords.delete(taskId);
            }
            pendingSuggestedNames.delete(taskId);
            if (activeLiveGroupId === taskId) activeLiveGroupId = '';
        }
        return ephemeralDecisionTaskIds.has(taskId);
    }

    function reanchorTaskCard(
        record,
        rawTs,
        { suppressDomInsert = false } = {},
        seen = new Set(),
    ) {
        if (!record || seen.has(record.groupId)) return false;
        seen.add(record.groupId);
        const movedEarlier = stampNodeTimestamp(record.root, rawTs, { anchor: true });
        if (record.isSubagent) {
            const parent = liveCardRecords.get(record.parentGroupId);
            const parentMoved = reanchorTaskCard(parent, rawTs, { suppressDomInsert }, seen);
            return movedEarlier || parentMoved;
        }
        if (!movedEarlier) return false;
        if (suppressDomInsert || _syncPass1Active) {
            record._anchorOrderDirty = true;
            return true;
        }
        insertMessageNode(record.root, { reorderExisting: true });
        record._anchorOrderDirty = false;
        return true;
    }

    function reanchorVisibleTaskCard(taskState, rawTs, options = {}) {
        if (!taskState?.cardVisible) return false;
        return reanchorTaskCard(liveCardRecords.get(taskState.taskId), rawTs, options);
    }

    function revealBufferedCardIfNeeded(taskState, { suppressDomInsert = false, rawTs = '' } = {}) {
        return withStableViewport(() => revealBufferedCardMutation(
            taskState, { suppressDomInsert, rawTs },
        ));
    }

    function revealBufferedCardMutation(taskState, { suppressDomInsert = false, rawTs = '' } = {}) {
        if (!taskState) return;
        if (taskState.cardVisible) {
            reanchorVisibleTaskCard(taskState, rawTs, { suppressDomInsert });
            return;
        }
        if (!(taskState.forceCard || taskState.toolCalls > 0 || shouldAlwaysShowTaskCard(taskState.taskId))) {
            return;
        }
        taskState.cardVisible = true;
        activeLiveGroupId = taskState.taskId;
        const subagentInfo = subagentChildParents.get(taskState.taskId);
        const record = subagentInfo
            ? getSubagentCardRecord(
                taskState.taskId,
                subagentInfo.parentId,
                subagentInfo.role,
            )
            : getLiveCardRecord(taskState.taskId);
        let anchorMovedEarlier = false;
        if (!record.isSubagent) {
            anchorMovedEarlier = stampNodeTimestamp(record.root, rawTs, { anchor: true });
            for (const update of taskState.bufferedLiveUpdates) {
                anchorMovedEarlier = stampNodeTimestamp(
                    record.root, update.rawTs, { anchor: true }
                ) || anchorMovedEarlier;
            }
        }
        ensureLiveCardVisible(record, { suppressDomInsert, reorderExisting: anchorMovedEarlier });
        const bufferedUpdates = [...taskState.bufferedLiveUpdates];
        taskState.bufferedLiveUpdates = [];
        for (const update of bufferedUpdates) {
            applyLiveCardState(update.summary, taskState.taskId, update.ts, update.dedupeKey, {
                suppressDomInsert,
                rawTs: update.rawTs,
            });
        }
        if (taskState.completed) {
            finishLiveCard(taskState.taskId, taskState.completedPhase || 'done');
        }
    }

    function queueTaskLiveUpdate(summary, taskId, ts, dedupeKey = '', rawTs = '') {
        return withStableViewport(() => queueTaskLiveUpdateMutation(
            summary, taskId, ts, dedupeKey, rawTs,
        ));
    }

    function queueTaskLiveUpdateMutation(summary, taskId, ts, dedupeKey = '', rawTs = '') {
        const resolvedTaskId = taskId || activeLiveGroupId || '';
        if (!resolvedTaskId) return;
        const taskState = getTaskUiState(resolvedTaskId, true);
        if (!taskState) return;
        // Even an already-completed card must absorb an earlier historical/nested
        // event into its chronology anchor before lifecycle policy ignores the
        // event's content.
        reanchorVisibleTaskCard(taskState, rawTs);
        if (taskState.completed && !isTerminalTaskPhase(summary.phase || '', summary.terminal)) {
            // A non-terminal event on a reusable id starts a fresh visible cycle.
            if (REUSABLE_TASK_IDS.has(resolvedTaskId)) {
                if (taskState.cleanupTimer) clearTimeout(taskState.cleanupTimer);
                taskState.completed = false;
                taskState.completedPhase = '';
                taskState.cardVisible = false;
                taskState.bufferedLiveUpdates = [];
                taskState.toolCalls = 0;
                taskState.forceCard = false;
                const oldRec = liveCardRecords.get(resolvedTaskId);
                if (oldRec) {
                    oldRec.root?.remove();
                    liveCardRecords.delete(resolvedTaskId);
                }
                retiredTaskIds.delete(resolvedTaskId);
            } else {
                return;
            }
        }
        if (summary.phase === 'error' || summary.phase === 'timeout' || (summary.terminal && summary.phase === 'warn')) {
            taskState.forceCard = true;
        }
        if (!taskState.cardVisible) {
            bufferLiveUpdate(taskState, summary, ts, dedupeKey, rawTs);
            revealBufferedCardIfNeeded(taskState, { rawTs });
            return;
        }
        applyLiveCardState(summary, resolvedTaskId, ts, dedupeKey, { rawTs });
    }

    function createLiveCardRecord(groupId = '', options = {}) {
        const normalizedGroupId = groupId || `task-${Date.now()}-${Math.random().toString(16).slice(2)}`;
        const timelineId = `chat-live-timeline-${normalizedGroupId.replace(/[^A-Za-z0-9_-]/g, '-')}`;
        const root = document.createElement('div');
        root.className = 'chat-live-card';
        root.dataset.taskId = normalizedGroupId;
        if (options.isSubagent) {
            root.classList.add('subagent');
            root.dataset.subagent = '1';
            root.dataset.parentTaskId = String(options.parentGroupId || '');
            root.dataset.subagentRole = String(options.role || '');
        }
        root.dataset.finished = '0';
        root.dataset.expanded = ((options.isSubagent && nestedSubagentsExpanded) || stickyExpandedSlots.has(normalizedGroupId)) ? '1' : '0';
        // No "Turn into project" for: subagent cards, non-main panels, or a task that
        // is ALREADY bound to a project (a project-chat follow-up) — see task_bindings
        // from /api/state, surfaced on window.__ouroTaskBindings (P2).
        const alreadyBound = !!(window.__ouroTaskBindings || {})[normalizedGroupId];
        const projectActionHtml = (
            isMain
            && !options.isSubagent
            && !alreadyBound
            && !ephemeralDecisionTaskIds.has(normalizedGroupId)
        )
            ? `<div class="chat-live-actions"><button type="button" class="chat-live-project-btn" data-turn-into-project>Turn into project</button></div>`
            : '';
        root.innerHTML = `
            <button type="button" class="chat-live-summary-button" data-live-summary-button aria-expanded="false" aria-controls="${escapeHtmlAttr(timelineId)}">
                <div class="chat-live-summary">
                    <div class="chat-live-summary-main">
                        <span class="chat-live-phase working" data-live-phase>Working</span>
                        <div class="chat-live-typing" data-live-typing aria-hidden="true">
                            <span></span><span></span><span></span>
                        </div>
                        <span class="chat-live-title" data-live-title>Waiting for work</span>
                    </div>
                    <div class="chat-live-summary-side">
                        <span class="chat-live-count" data-live-count hidden>2 notes</span>
                        <span class="chat-live-toggle" data-live-toggle>Show details</span>
                        <svg class="chat-live-chevron" width="14" height="14" viewBox="0 0 20 20" fill="none" aria-hidden="true">
                            <path d="M5 7.5 10 12.5 15 7.5" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"></path>
                        </svg>
                    </div>
                </div>
                <div class="chat-live-activity" data-live-activity></div>
                <div class="chat-live-meta" data-live-meta></div>
            </button>
            ${projectActionHtml}
            <div class="chat-live-timeline" data-live-timeline id="${escapeHtmlAttr(timelineId)}"></div>
        `;
        const record = {
            groupId: normalizedGroupId,
            root,
            summaryButtonEl: root.querySelector('[data-live-summary-button]'),
            phaseEl: root.querySelector('[data-live-phase]'),
            inlineTypingEl: root.querySelector('[data-live-typing]'),
            titleEl: root.querySelector('[data-live-title]'),
            activityEl: root.querySelector('[data-live-activity]'),
            countEl: root.querySelector('[data-live-count]'),
            metaEl: root.querySelector('[data-live-meta]'),
            toggleEl: root.querySelector('[data-live-toggle]'),
            turnProjectBtn: root.querySelector('[data-turn-into-project]'),
            // P5: "Cancel run" button element (rendered lazily by syncCancelRunButton
            // once the host-attested cancelable marker is known for this task).
            cancelRunBtn: null,
            timelineEl: root.querySelector('[data-live-timeline]'),
            updates: 0,
            finished: false,
            items: [],
            lastHumanHeadline: '',
            expandedLineKeys: new Set(),
            isSubagent: Boolean(options.isSubagent),
            parentGroupId: String(options.parentGroupId || ''),
            subagentRole: String(options.role || ''),
            subagentsEl: null,
            _anchorOrderDirty: false,
            // perf2 P4.4: collapsed timelines defer DOM building; the flag says
            // the rendered timeline DOM is stale relative to record.items.
            _timelineDirty: false,
            // perf2 P4.3: last frame's summary meta strings — meta renders from
            // record state (renderLiveCardMeta), once per card in a batch.
            _lastFrameMeta: [],
            // Hidden-page layout sync is deferred until page/visibility returns.
            _needsLayoutSync: false,
            // The owner's request that spawned this card (main, non-subagent only),
            // used to name a project on "turn into project" when the server has no
            // title/objective yet (P1, direct-chat conversion). One-shot handoff.
            objectiveHint: (isMain && !options.isSubagent) ? _pendingCardObjective : '',
            // Cluster B: the proactively-coined LLM project name; when set it becomes
            // the card title (the activity headline keeps rendering in the lines below).
            suggestedName: '',
            // P1 (v6.82): last bounded activity projection (remembered even while
            // the collapsed line is suppressed on unnamed root cards) + sticky cost.
            collapsedActivity: '',
            costMeta: null,
        };
        if (isMain && !options.isSubagent) _pendingCardObjective = '';
        record.summaryButtonEl?.addEventListener('click', () => {
            const nowExpanded = record.root.dataset.expanded !== '1';
            setLiveCardExpanded(record, nowExpanded);
            if (REUSABLE_TASK_IDS.has(record.groupId)) {
                if (nowExpanded) stickyExpandedSlots.add(record.groupId);
                else stickyExpandedSlots.delete(record.groupId);
            }
        });
        record.turnProjectBtn?.addEventListener('click', (event) => {
            event.stopPropagation();
            turnTaskIntoProject(record);
        });
        record.timelineEl?.addEventListener('click', (event) => {
            const button = event.target.closest('[data-live-line-toggle]');
            // Row-surface disclosure (v6.71.0): any click on the line's
            // NON-interactive surface toggles it (guards live in the pure
            // helper: nested interactive elements, active text selection).
            const lineKey = button
                ? (button.dataset.liveLineToggle || '')
                : liveLineRowToggleKey(event.target, window.getSelection?.());
            if (!lineKey) return;
            const nowExpanded = !record.expandedLineKeys.has(lineKey);
            if (nowExpanded) record.expandedLineKeys.add(lineKey);
            else record.expandedLineKeys.delete(lineKey);
            renderLiveCardTimeline(record);
            syncLiveCardLayout(record);
            // Keyboard/AT continuity: focus the rebuilt line's REAL toggle button.
            record.timelineEl
                ?.querySelector(`[data-live-line-toggle="${(window.CSS && CSS.escape) ? CSS.escape(lineKey) : lineKey}"]`)
                ?.focus?.({ preventScroll: true });
            // P3: on expand, lazily fetch the genuinely-full output for a server-truncated
            // line (the WS preview was capped at 4000); cached on the item so a re-render
            // keeps it. Best-effort — the capped preview stays on failure.
            if (nowExpanded) {
                const item = record.items.find((it) => it.lineKey === lineKey);
                if (item && item.truncated && item.fullRef && !item.fetchedFull && !item._fetchingFull) {
                    fetchFullLineOutput(item, record);
                }
            }
        });
        liveCardRecords.set(normalizedGroupId, record);
        // Cluster B: apply a name that arrived (task_named) before this card existed.
        const _pendingName = pendingSuggestedNames.get(normalizedGroupId);
        if (_pendingName && !record.isSubagent) {
            pendingSuggestedNames.delete(normalizedGroupId);
            record.suggestedName = _pendingName;
            if (record.titleEl) record.titleEl.textContent = _pendingName;
        }
        resetLiveCardRecord(record);
        // P5: the cancelable marker may have arrived (scheduled progress frame /
        // history replay) before this card was minted.
        syncCancelRunButton(record);
        return record;
    }

    function getLiveCardRecord(groupId = '') {
        const normalizedGroupId = groupId || activeLiveGroupId || 'chat';
        return liveCardRecords.get(normalizedGroupId) || createLiveCardRecord(normalizedGroupId);
    }

    function getSubagentCardRecord(childId = '', parentId = '', role = '') {
        return withStableViewport(() => getSubagentCardRecordMutation(childId, parentId, role));
    }

    function getSubagentCardRecordMutation(childId = '', parentId = '', role = '') {
        if (!childId || !parentId) return null;
        const existing = liveCardRecords.get(childId);
        const wasSubagent = existing?.isSubagent === true || existing?.root?.classList.contains('subagent');
        const record = existing || createLiveCardRecord(childId, {
            isSubagent: true,
            parentGroupId: parentId,
            role,
        });
        record.isSubagent = true;
        record.parentGroupId = parentId;
        record.subagentRole = role || record.subagentRole || '';
        record.root.classList.add('subagent');
        record.root.dataset.subagent = '1';
        record.root.dataset.parentTaskId = parentId;
        record.root.dataset.subagentRole = record.subagentRole;
        const container = ensureSubagentContainer(parentId);
        if (container && record.root.parentNode !== container) {
            container.appendChild(record.root);
        }
        const parentRecord = liveCardRecords.get(parentId);
        if (parentRecord) updateLiveCardCount(parentRecord);
        if (!wasSubagent) setLiveCardExpanded(record, nestedSubagentsExpanded);
        return record;
    }

    function resetLiveCardRecord(record) {
        record.updates = 0;
        record.finished = false;
        record.items = [];
        record.lastHumanHeadline = '';
        record.expandedLineKeys.clear();
        record._anchorOrderDirty = false;
        record._timelineDirty = false;
        record._lastFrameMeta = [];
        clearStickyCardState(record);
        record.titleEl.textContent = 'Working...';
        record.phaseEl.dataset.phase = 'working';
        record.phaseEl.textContent = 'Working';
        record.phaseEl.className = 'chat-live-phase working';
        record.countEl.hidden = true;
        record.countEl.textContent = '0 notes';
        record.metaEl.innerHTML = '';
        record.timelineEl.innerHTML = '';
        record.root.dataset.finished = '0';
        setLiveCardTypingVisible(record, true);
        setLiveCardExpanded(record, (record.isSubagent && nestedSubagentsExpanded) || stickyExpandedSlots.has(record.groupId));
    }

    function ensureLiveCardVisible(
        record,
        { suppressDomInsert = false, reorderExisting = false } = {},
    ) {
        if (record?.isSubagent && record.parentGroupId) {
            if (!suppressDomInsert && !_syncPass1Active) {
                const parentRecord = getLiveCardRecord(record.parentGroupId);
                if (parentRecord.isSubagent && parentRecord.parentGroupId) {
                    ensureLiveCardVisible(parentRecord);
                } else {
                    insertMessageNode(parentRecord.root);
                }
                const container = ensureSubagentContainer(record.parentGroupId);
                if (container && record.root.parentNode !== container) {
                    container.appendChild(record.root);
                }
                updateLiveCardCount(parentRecord);
            }
            return;
        }
        if (!record.isSubagent && !suppressDomInsert && !_syncPass1Active) {
            insertMessageNode(record.root, { reorderExisting });
        }
    }

    function updateLiveCardCount(record) {
        // perf2 P4.3: one count render per card at the end of a replay batch.
        const batch = getRebuildBatch();
        if (batch) {
            batch.touch(record);
            return;
        }
        if (!record?.countEl) return;
        const bits = [];
        if (record.items.length >= 2) bits.push(`${record.items.length} notes`);
        const children = directSubagentCount(record);
        if (children) bits.push(`${children} ${children === 1 ? 'child' : 'children'}`);
        record.countEl.hidden = bits.length === 0;
        record.countEl.textContent = bits.join(' · ');
    }

    function syncLiveCardLayout(record) {
        // perf2 P4.3: one layout sync per card after the batch mount.
        const batch = getRebuildBatch();
        if (batch) {
            batch.touch(record);
            return;
        }
        if (!record?.root) return;
        // Hidden SPA/browser tabs report zero geometry; defer to avoid collapsed
        // cards. Generalized to panel instances: any visible host counts.
        const activePage = record.root.closest('.page.active');
        const panelHost = record.root.closest('.chat-instance-panel');
        // A panel counts as visible only when it is actually shown (not a
        // hidden/display:none secondary instance) — zero geometry otherwise.
        const visibleHost = activePage || (panelHost && panelHost.offsetParent !== null);
        if (!visibleHost || document.hidden) {
            record._needsLayoutSync = true;
            return;
        }
        record._needsLayoutSync = false;
        if (record.isSubagent && record.parentGroupId) {
            const parentRecord = liveCardRecords.get(record.parentGroupId);
            if (parentRecord?.root?.isConnected) {
                requestAnimationFrame(() => syncLiveCardLayout(parentRecord));
            }
        }
    }

    // P3: fetch the genuinely-full output for a server-truncated timeline line (the WS
    // preview was capped at 4000 chars), cache it on the item, then re-render if the line
    // is still expanded. The full text is fetched on demand (not pushed over the socket)
    // and shown in a bounded-scroll box. Best-effort — the capped preview stays on failure.
    async function fetchFullLineOutput(item, record) {
        item._fetchingFull = true;
        try {
            const resp = await apiFetch(`/api/tasks/${encodeURIComponent(item.fullRef)}`, { cache: 'no-store' });
            const data = resp && typeof resp.json === 'function' ? await resp.json() : resp;
            // Compose ALL available full fields — a subagent line can carry both a result AND a
            // (separately truncated) trace_summary, so `result || trace_summary` would hide the
            // full trace. Label each section when both are present.
            const result = String((data && data.result) || '').trim();
            const trace = String((data && data.trace_summary) || '').trim();
            let full = '';
            if (result && trace) full = `[RESULT]\n${result}\n\n[TRACE]\n${trace}`;
            else full = result || trace;
            if (full) item.fetchedFull = full;
        } catch {
            // best-effort: leave the capped preview on failure
        } finally {
            item._fetchingFull = false;
            if (!destroyed && record.expandedLineKeys.has(item.lineKey)) {
                const hadFocus = Boolean(
                    document.activeElement?.closest?.(`[data-live-line-key="${(window.CSS && CSS.escape) ? CSS.escape(item.lineKey) : item.lineKey}"]`),
                );
                renderLiveCardTimeline(record);
                syncLiveCardLayout(record);
                if (hadFocus) {
                    record.timelineEl
                        ?.querySelector(`[data-live-line-toggle="${(window.CSS && CSS.escape) ? CSS.escape(item.lineKey) : item.lineKey}"]`)
                        ?.focus?.({ preventScroll: true });
                }
            }
        }
    }

    function applyLiveCardState(summary, groupId, ts, dedupeKey = '', options = {}) {
        return withStableViewport(() => applyLiveCardStateMutation(
            summary, groupId, ts, dedupeKey, options,
        ));
    }

    function applyLiveCardStateMutation(summary, groupId, ts, dedupeKey = '', { suppressDomInsert = false, rawTs = '' } = {}) {
        const nextGroupId = groupId || activeLiveGroupId || 'active';
        const record = getLiveCardRecord(nextGroupId);
        // A converted card is now a terminal project chip — its task is owned by the
        // project panel. Ignore ALL further frames (incl. terminal) so they neither
        // overwrite the chip nor dereference the nulled element refs (P3).
        if (record.root?.dataset?.projectCreated === '1') return;
        const nextPhase = summary.phase || '';
        if (record.finished && !isTerminalTaskPhase(nextPhase, summary.terminal)) {
            // A late cost frame still settles the finished card's cost meta.
            if (summary.costProjection) {
                record.costMeta = mergeStickyCostMeta(record.costMeta, summary.costProjection);
                const batch = getRebuildBatch();
                if (batch) batch.touch(record);
                else renderLiveCardMeta(record);
            }
            return;
        }

        if (!record.isSubagent) {
            activeLiveGroupId = nextGroupId;
            reanchorTaskCard(record, rawTs, { suppressDomInsert });
        }
        ensureLiveCardVisible(record, { suppressDomInsert });
        record.updates += 1;
        const wasFinished = record.finished;
        // Prefer the last meaningful headline when an update carries none (e.g. a
        // structured terminal marker), so finishing a card doesn't blank its title.
        const headline = summary.headline || record.lastHumanHeadline || 'Working...';
        const syntheticKey = summary.dedupeKey || dedupeKey || `${summary.phase || 'working'}|${headline}|${summary.body || ''}`;
        const isLegacyParentSubagentKey = syntheticKey.startsWith('parent-subagent:');
        const inPlaceByKey = isLegacyParentSubagentKey
            || syntheticKey.startsWith('subagent-lifecycle:')
            || syntheticKey.startsWith('subagent-progress:')
            || syntheticKey.startsWith('subagent-result:')
            || syntheticKey.startsWith('task_done|');
        if (!isLegacyParentSubagentKey) {
            record.finished = isTerminalTaskPhase(nextPhase, summary.terminal);
        }
        record.root.dataset.finished = record.finished ? '1' : '0';
        if (summary.human && headline) {
            record.lastHumanHeadline = headline;
        }

        const shouldPromote =
            Boolean(summary.promote)
            || !record.lastHumanHeadline
            || record.finished;
        const activeHeadline = shouldPromote
            ? headline
            : (record.lastHumanHeadline || headline);
        const activePhase = record.finished
            ? (summary.phase || 'done')
            : (shouldPromote ? (summary.phase || 'working') : (record.phaseEl.dataset.phase || 'working'));

        record.phaseEl.dataset.phase = activePhase;
        record.phaseEl.textContent = formatLiveCardPhaseLabel(activePhase);
        record.phaseEl.className = `chat-live-phase ${activePhase}`;
        // Sticky hold: post-task frames must not repaint "Working".
        if (record.finalizingHold && !record.finished && !record.cancelPendingPolicy) {
            record.phaseEl.dataset.phase = 'working';
            record.phaseEl.textContent = 'Finalizing…';
            record.phaseEl.className = 'chat-live-phase working finalizing';
        }
        // Cluster B: a coined project name takes the title slot; the live activity
        // headline still renders in the timeline lines below. Falls back to the
        // activity headline until the proactive namer has produced a name.
        record.titleEl.textContent = record.suggestedName || activeHeadline;
        // The collapsed line is a compact presentation projection, while the
        // complete latest activity remains independently reachable through the
        // expanded timeline. Root cards accept activity only from human frames;
        // terminal "Done" markers must not overwrite the last real action.
        const previewSource = record.isSubagent
            ? String(summary.activityPreview ?? summary.body ?? '')
            : (summary.human ? String(summary.activityPreview ?? activeHeadline ?? '') : '');
        const activityCandidate = previewSource.trim();
        if (activityCandidate) record.collapsedActivity = boundActivityPreview(activityCandidate);
        const activityText = projectCollapsedActivity({
            isSubagent: record.isSubagent,
            suggestedName: record.suggestedName,
            headline: record.isSubagent ? '' : record.collapsedActivity,
            body: record.isSubagent ? record.collapsedActivity : '',
            previous: record.collapsedActivity,
        });
        renderCollapsedActivity(record, activityText);

        const shouldRenderLine = summary.visible !== false && Boolean(headline || summary.body);
        // Legacy parent-subagent rows update in place if replayed from old
        // history. Child-card lifecycle/progress rows also evolve in place.
        let timelineUpdate = 'none';
        let patchIndex = -1;
        if (shouldRenderLine) {
            const lastIdx = record.items.length - 1;
            // Full-array dedup (Variant A): match the incoming line's key ANYWHERE in
            // the card, not only against the last item. Otherwise a background
            // syncHistory(rebuildAll=false) re-feeds historical progress lines whose
            // key != the last item, and each gets re-appended → the "Notes" count
            // grows without bound on every sync/reconnect.
            const existingIdx = record.items.findIndex((it) => it.dedupeKey === syntheticKey);
            if (existingIdx !== -1 && inPlaceByKey) {
                const it = record.items[existingIdx];
                it.phase = summary.phase || it.phase;
                it.headline = headline || it.headline;
                it.fullHeadline = summary.fullHeadline || headline || it.fullHeadline;
                it.body = summary.body || '';
                it.fullBody = summary.fullBody || summary.body || it.fullBody || '';
                it.fullRef = summary.fullRef || it.fullRef || '';
                it.truncated = summary.truncated || it.truncated || false;
                it.ts = ts || it.ts;
                patchIndex = existingIdx;
                timelineUpdate = 'patch-at';
            } else if (existingIdx === lastIdx && existingIdx !== -1) {
                // Consecutive live duplicate of the most recent line → coalesce count.
                const it = record.items[existingIdx];
                it.count += 1;
                it.ts = ts || it.ts;
                it.fullHeadline = summary.fullHeadline || it.fullHeadline || it.headline;
                it.fullBody = summary.fullBody || it.fullBody || it.body;
                it.fullRef = summary.fullRef || it.fullRef || '';
                it.truncated = summary.truncated || it.truncated || false;
                timelineUpdate = 'patch-last';
            } else if (existingIdx !== -1) {
                // Already rendered earlier in this card (e.g. a historical progress line
                // re-fed by a background sync). Do NOT re-append (the unbounded "Notes"
                // growth) and do NOT bump its count — just keep its timestamp fresh.
                const it = record.items[existingIdx];
                it.ts = ts || it.ts;
                timelineUpdate = 'duplicate-skip';
            } else {
                const lineKey = `line-${Date.now()}-${Math.random().toString(16).slice(2)}`;
                record.items.push({
                    phase: summary.phase || 'working',
                    headline: headline || 'Update',
                    fullHeadline: summary.fullHeadline || headline || 'Update',
                    body: summary.body || '',
                    fullBody: summary.fullBody || summary.body || '',
                    fullRef: summary.fullRef || '',
                    truncated: summary.truncated || false,
                    ts: ts || '',
                    count: 1,
                    dedupeKey: syntheticKey,
                    lineKey,
                });
                timelineUpdate = 'append';
            }
        }
        updateLiveCardCount(record);
        // "Latest" is an ACTIVITY clock, not a bookkeeping clock: a cost-only frame
        // (task_cost_finalized and friends carry no human narration) must not make a
        // silent card look freshly active. Only frames that actually said something
        // move it.
        if (ts && (summary.human || activityCandidate)) record.latestActivityTs = ts;
        // P1 (v6.82): sticky cost — only frames carrying task-scope accounting
        // evidence attach a costProjection; a costless frame re-renders the
        // previous projection instead of erasing it.
        if (summary.costProjection) {
            record.costMeta = mergeStickyCostMeta(record.costMeta, summary.costProjection);
        }
        // Phase 6 (owner directive #1): the executor chip is STICKY on the card —
        // a later costless/quiet frame must not erase the fact that this bubble
        // ran on a harness. Absent fact leaves it absent; no placeholder chip.
        if (summary.executorChip) record.executorChip = summary.executorChip;
        // perf2 P4.3: meta renders from record state — immediately on the live
        // path, once per card at the end of a rebuildAll replay batch.
        record._lastFrameMeta = Array.isArray(summary.meta) ? summary.meta : [];
        const batch = getRebuildBatch();
        if (batch) batch.touch(record);
        else renderLiveCardMeta(record);
        // Incremental updates; full rebuilds stay limited to toggles.
        const lastItem = record.items[record.items.length - 1];
        if (timelineUpdate === 'append' && lastItem) {
            appendTimelineItem(lastItem, record);
        } else if (timelineUpdate === 'patch-last' && lastItem) {
            patchLastTimelineItem(lastItem, record);
        } else if (timelineUpdate === 'patch-at' && patchIndex !== -1) {
            patchTimelineItemAt(record.items[patchIndex], record);
        }
        ensureLiveCardVisible(record, { suppressDomInsert });
        syncLiveCardLayout(record);
        hideTypingIndicatorOnly();
        const justFinished = record.finished && !wasFinished;
        const drivesComposerStatus = !isBackgroundTaskId(nextGroupId);
        // P5: a finished card must not keep offering "Cancel run". A log-channel
        // task_done terminates the card HERE without passing finishLiveCard, so
        // the cancelable marker must be dropped on this path too (P3 growth cap).
        if (justFinished) {
            cancelableTaskIds.delete(record.groupId);
            syncCancelRunButton(record);
        }
        if (record.finished) {
            setLiveCardTypingVisible(record, false);
            markTaskComplete(nextGroupId, summary.phase || 'done');
            if (justFinished) {
                if (!stickyExpandedSlots.has(record.groupId)) {
                    setLiveCardExpanded(record, record.isSubagent && nestedSubagentsExpanded);
                }
                scheduleHistorySync();
            }
            syncLiveCardToggle(record);
            if (drivesComposerStatus) {
                lastTerminalAttention = (summary.phase === 'error' || summary.phase === 'timeout');
                syncChatStatus();
            }
        } else {
            setLiveCardTypingVisible(record, true);
            if (drivesComposerStatus) {
                lastTerminalAttention = false;
                syncChatStatus();
            } else if (!hasActiveLiveCard()) {
                syncChatStatus();
            }
        }
        if (summary.expandByDefault) {
            setLiveCardExpanded(record, true);
        }
    }

    function finishLiveCard(groupId = '', phase = '') {
        return withStableViewport(() => finishLiveCardMutation(groupId, phase));
    }

    function finishLiveCardMutation(groupId = '', phase = '') {
        const record = groupId
            ? liveCardRecords.get(groupId)
            : (activeLiveGroupId ? liveCardRecords.get(activeLiveGroupId) : null);
        if (!record) return;
        // A converted card is a terminal project chip now — ignore late terminal
        // frames so they neither overwrite the chip nor touch its element refs (T4).
        if (record.root?.dataset?.projectCreated === '1') return;
        const wasFinished = record.finished;
        record.finished = true;
        record.finalizingHold = false;
        record.root.dataset.finished = '1';
        // A finished task can never be cancelled again; dropping the marker here
        // keeps the set from accumulating every task id of a long session (P3).
        cancelableTaskIds.delete(record.groupId);
        syncCancelRunButton(record);
        const activePhase = ['error', 'timeout', 'warn', 'cancelled'].includes(phase) ? phase : 'done';
        record.phaseEl.dataset.phase = activePhase;
        record.phaseEl.textContent = formatLiveCardPhaseLabel(activePhase);
        record.phaseEl.className = `chat-live-phase ${activePhase}`;
        setLiveCardTypingVisible(record, false);
        markTaskComplete(record.groupId, activePhase);
        if (!wasFinished) {
            if (!stickyExpandedSlots.has(record.groupId)) {
                setLiveCardExpanded(record, record.isSubagent && nestedSubagentsExpanded);
            }
            scheduleHistorySync();
        }
        syncLiveCardToggle(record);
        if (activeLiveGroupId === record.groupId) activeLiveGroupId = '';
        lastTerminalAttention = (activePhase === 'error' || activePhase === 'timeout');
        syncChatStatus();
    }

    return {
        registerEphemeralDecisionFrame,
        revealBufferedCardIfNeeded,
        queueTaskLiveUpdate,
        getLiveCardRecord,
        getSubagentCardRecord,
        ensureLiveCardVisible,
        updateLiveCardCount,
        syncLiveCardLayout,
        applyLiveCardState,
        finishLiveCard,
        bindLiveCardCollaborators,
        getActiveLiveGroupId: () => activeLiveGroupId,
        setActiveLiveGroupId: (groupId) => { activeLiveGroupId = groupId; },
        setPendingCardObjective: (text) => { _pendingCardObjective = text; },
        setNestedSubagentsExpanded: (expanded) => { nestedSubagentsExpanded = expanded; },
        getLastTerminalAttention: () => lastTerminalAttention,
        setLastTerminalAttention: (attention) => { lastTerminalAttention = attention; },
        setSyncPass1Active: (active) => { _syncPass1Active = active; },
        markLiveCardsDestroyed: () => { destroyed = true; },
    };
}
