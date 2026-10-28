import { escapeHtmlAttr, escapeHtmlText as escapeHtml } from './utils.js';
import { bindMarkdownTables, destroyChatMarkdown, enhanceChatMarkdown, mountChatMarkdown, renderChatMarkdown } from './chat_markdown.js';
import { renderPageHeader } from './page_header.js';
import { strokeIcon } from './page_icons.js';
import { createRoomHeader, mainHeaderActionsHtml } from './chat_header.js';
import { pageLabel } from './pages.js';
import { showToast } from './toast.js';
import { tx } from './i18n.js';
import { hostRowDisplayText } from './host_notice_text.js';
import { ORIGIN_ADDRESSED_NOTICE_TYPES, childRowPlacement, replayConcludesTask } from './chat_row_placement.js';
import { appendProjectRow, decorateProjectRow, PROJECT_ROW_TYPES, refreshProjectAnswerFold, syncSavedProjectContext } from './project_answer.js';
import { createWorkPortals, isPortal, reconcilePortalNotes } from './work_portal.js';
import { createTaskEpisodes } from './task_episodes.js';
import { markSwarmLabel, syncSwarmReceipt } from './swarm_presentation.js';
import { createChatMedia, decorateContactSource, showTaskIncidentToast } from './chat_media.js';
import { attachmentCaption, attachmentViews, createUnconfirmedSends } from './chat_attachments.js';
import { createChatDecision } from './chat_decision.js';
import { createRoomWork } from './room_work.js';
import { applyCardWorkViews, cardControls, cardMenuActions, changedOffers, emptyWorkState, openCardControls, ownStopPress, reconcileWorkState, roomStatusView } from './work_status.js';
import { statusMarkHtml } from './status_mark.js';
import { createMainWaiting } from './main_waiting.js';
import { createHostStart } from './host_starting_status.js';
import { createModelWaitController, isModelWaitReference } from './model_wait.js';
import { syncSettledItems } from './settled_card.js';
import { createChatReadingPosition } from './chat_reading_position.js';
import { createRowSleep } from './chat_row_sleep.js';
import { onRefit } from './text_fit.js';
import { createChatHistoryPager, historyCoverage, historyIslandAtEdge } from './chat_history.js';
import { createProjectReadReceipt, isAtNewestMessage, newestMessageOffset } from './project_read_state.js';
import { mergeHistoricalTimelineItem, historyNodeIsProtected, historyRowIds, historyStamps, stampHistoryNode, compareHistoryPosition } from './chat_history_replay.js';
import { apiClient, apiFetch, fetchTaskDetail, fetchTaskDetailStrict, isStoreMigrating } from './api_client.js';
import { syncHistoryRetentionItem } from './history_retention.js';
import {
    childReportLabel,
    getLogTaskGroupId,
    isGroupedTaskEvent,
    isTerminalTaskDetail,
    normalizeLogTs,
    ownerHurryProjection,
    summarizeChatLiveEvent,
    taskCancelPending,
    taskOutcomeSeverity,
    taskPresentation,
    taskSoftStopPending,
    taskDoneIsTerminal,
    keepStickyExecutorChip,
    taskTerminalSummary,
    taskTerminalPhase,
} from './log_events.js';
import {
    ACTION_FINALIZE,
    REUSABLE_TASK_IDS,
    TASK_CONTROL_TRIGGER_LABEL,
    cancelRunEligibility,
    createTaskResumeButton,
    requestStop,
} from './task_control_menu.js';
import { composerMarkup, createComposer } from './composer.js';
import { effortFact } from './effort_chip.js';
import { mountEmptyChatWelcome } from './welcome_preference.js';
import { createChatReplies } from './chat_reply.js';
import { createHistoryReveal } from './chat_history_reveal.js';
import { createChatRowMenu } from './chat_row_menu.js';
import { clearTransientRoutingAnnotations } from './routing_receipt.js';
import {
    captureLiveCardPhaseState,
    censusTaskFacts,
    desiredLiveCardPhase,
    liveCardCountBits,
    markOwnerHurry,
    replayTerminalPhase,
    restoreLiveCardPhaseState,
    setLiveCardPhase,
    setHistoricalUnavailable,
    setHistoricalUnconfirmed,
    syncParkedPhase,
} from './task_phase_chip.js';
import {
    renderSkillReviewDisclosure,
    wireSkillReviewDisclosure,
} from './skill_review_card.js';
import {
    classifyReviewLifecycle,
    classifyReviewLifecyclePointer,
    createReviewPresentationController,
    createReviewHydrator,
    reviewReferenceFromRow,
    reviewGroupFromHistoryRow,
    reviewGroupsFromTaskDetail,
    setReviewAnchor,
} from './review_presentation.js';
import {
    captureLiveCardProjection,
    createHistoryResyncScheduler,
    createHistoryControls,
    createLiveCardBound,
    createLiveCardTimelineRenderer,
    createTimelineAnchors,
    insertTimelineNode,
    liveCardProjectionChanged,
    keepSelection,
    replayedEvidenceTs,
    syncLiveCardToggle,
    timelineScroll,
    updateLiveTimelineItem,
    upsertToolFoldRow,
} from './chat_render_batch.js';
import { measure } from './layout_batch.js';
import { placeCheckRow, placeReplayedEvidence } from './verification_presentation.js';
import {
    COLLAPSED_ACTIVITY_MAX,
    boundActivityPreview,
    buildTimelineItemHtml,
    buildMessageKey,
    chatLogThreadAccepts,
    chatMediaMessageKey,
    chatThreadAccepts,
    clearStickyCardState,
    sendPanic,
    computeHydratedDirectActivities,
    documentMessageKey,
    durableChatMediaUrl,
    formatMsgTime,
    getOrCreateChatSessionId,
    headerBudgetPresentation,
    isForegroundLiveCard,
    isNonTerminalMediaHistoryRow,
    isReplayEvidenceRow,
    isTerminalTaskPhase,
    markIngressSaved,
    liveLineRowToggleKey,
    bindLiveCardTimeline,
    subagentIdentityTitle,
    subagentTwin,
    mergeStickyCostMeta,
    applyToolObservation,
    noteToolHostMetrics,
    toolEvidenceIncomplete,
    partitionLocalEchoJournal,
    projectCollapsedActivity,
    positiveTaskTerminalFact,
    unkeyedFrameEndsTurn,
    rawTimestampEpoch,
    stampNodeTimestamp,
    reconcileHydratedDirectActivities,
    restartNoticeText,
    senderLabel,
    taskCostMeta,
    taskCostProjection,
    unconfirmedForegroundCardIds,
    withTaskCostMeta,
    applyHistoricalModelExecution,
    cardMetaKeys,
    renderCollapsedActivity,
    syncCandidateRemainderItem,
    readCandidateRemainder,
    renderLiveCardMeta as renderCardMeta,
    ensureLiveActionsEl,
    cardRowEvidenceRef,
    cardRowSummary,
    evidenceLinkHtml,
    refitCollapsedActivity,
} from './chat_activity.js';

export {
    COLLAPSED_ACTIVITY_MAX,
    boundActivityPreview,
    chatMediaMessageKey,
    clearStickyCardState,
    clearTransientRoutingAnnotations,
    sendPanic,
    computeHydratedDirectActivities,
    headerBudgetPresentation,
    insertTimelineNode,
    isTerminalTaskDetail,
    isTerminalTaskPhase,
    durableChatMediaUrl,
    isNonTerminalMediaHistoryRow,
    liveLineRowToggleKey,
    mergeStickyCostMeta,
    projectCollapsedActivity,
    rawTimestampEpoch,
    reconcileHydratedDirectActivities,
    taskCostMeta,
    taskCostProjection,
};

// Host placement: custody warns; settled reviews show results.
const CARD_ROW_PHASES = new Map([['timeline', 'warn'], ['reviews', 'result']]);
const SUBAGENT_LIFECYCLE_EVENTS = new Set(['scheduled', 'running', 'completed', 'completed_warn', 'failed',
    'cancelled', 'rejected', 'interrupted']);  // a helper's narration is `progress`
const CHAT_STORAGE_KEY = 'ouro_chat';

export function initChat(ctx) {
    // Back-compat main-chat entry: one full-page instance bound to chat 1.
    return createChatInstance(ctx);
}

const taskKey = (value) => String(value || '').trim();

export function createChatInstance({
    ws, state, openSettingsTab, openDashboardTab,
    stateSnapshots,
    chatId = 1, projectId = '', idPrefix = 'chat', mountEl = null,
    asPanel = false, title = pageLabel('chat'),
    // app.js signal "a project panel is opening right now" — Main
    // defers its first hydration to it (bounded by an unconditional deadline).
    isProjectOpening = null,
    onHistoryRetry = null,
    onReadingLatest, revealQuotedSource,
}) {
    const container = mountEl || document.getElementById('content');
    const chatSessionId = getOrCreateChatSessionId(sessionStorage, globalThis.crypto);
    const isMain = chatId === 1;
    // Per-thread storage so a project thread never bleeds into the main chat.
    const storeKey = (base) => (isMain ? base : `${base}:${chatId}`);

    const page = document.createElement('div');
    page.id = asPanel ? `panel-${idPrefix}` : 'page-chat';
    page.className = asPanel ? 'chat-instance-panel' : 'page active';
    // A Project panel's bar comes from app.js (page_header.js); the instance seeds only its status badge.
    const headerHtml = asPanel
        ? `<div class="chat-panel-statusbar"><span id="chat-status" class="status-badge offline">Connecting…</span></div>`
        : renderPageHeader({
            title: title,
            variant: 'overlay',
            className: 'chat-page-header',
            toolbarClass: 'ui-chrome-group',
            actionsHtml: mainHeaderActionsHtml(),
            showMobileNav: false,  // a room's ☰ is its composer row's first control
        });
    // Main: a failed read or approximate restoration floats under the bar; Project keeps its status bar.
    page.innerHTML = `
        ${headerHtml}${asPanel ? '' : '<div class="chat-page-note"></div>'}
        <div id="chat-messages"></div>
        <div id="chat-input-area">
            <div class="chat-input-wrap">
                <button class="chat-scroll-bottom-btn" id="chat-scroll-bottom" type="button" aria-label="Scroll to latest message" title="Scroll to latest message">
                    ${strokeIcon('<path d="M12 5v14"/><path d="M19 12l-7 7-7-7"/>', 16)}
                    <span class="chat-scroll-activity-dot" aria-hidden="true" hidden></span>
                </button>
                ${composerMarkup()}
            </div>
        </div>
    `;
    if (idPrefix !== 'chat') {
        // Namespace ids; mirror classes retain the shared composer styling.
        page.querySelectorAll('[id]').forEach((el) => {
            if (el.id.startsWith('chat-')) {
                el.classList.add(el.id);
                el.id = idPrefix + '-' + el.id.slice(5);
            }
        });
    }
    container.appendChild(page);

    const byId = (suffix) => page.querySelector(`[id="${idPrefix}-${suffix}"]`);
    const messagesDiv = byId('messages');
    const input = byId('input');
    const inputArea = byId('input-area');
    const statusBadge = byId('status');
    const pageHeader = page.querySelector('.chat-page-header') || (asPanel ? container?.closest?.('.project-panel')?.querySelector?.('.project-panel-bar') : null);
    const pageNote = page.querySelector('.chat-page-note');
    const scrollBottomBtn = byId('scroll-bottom');
    const scrollActivityDot = scrollBottomBtn?.querySelector('.chat-scroll-activity-dot');
    let nestedSubagentsExpanded = false;
    let _remoteActivityDepth = 0;
    // The composer (composer.js), wired once the room's feed owners exist; Main's bar (chat_header.js) once its status is.
    let composer = null, header = null;

    // Destroyed instances ignore late async work and animation frames.
    let destroyed = false;
    // Every ws.on subscription's disposer, released together in destroy().
    const wsDisposers = [];
    const onWs = (event, fn) => wsDisposers.push(ws.on(event, fn));
    const replies = createChatReplies({ chatId, feed: messagesDiv, input, composerArea: inputArea, key: storeKey('ouro_chat_reply'),
        reading: () => reading, onDomWrite: withStableViewport, history: () => historyReveal,
        navigate: revealQuotedSource, present: () => !destroyed && isInstanceVisible() });
    const chatMedia = createChatMedia({
        chatSessionId,
        durableChatMediaUrl,
        formatMsgTime,
        insertMessageNode,
        senderLabel,
        stampNodeTimestamp,
        onDomWrite: withStableViewport,
        replies, composer: () => composer,
    });
    const mainWaiting = isMain ? createMainWaiting({ page, messagesDiv, insertMessageNode, stampNodeTimestamp,
        openSettings: openSettingsTab, refresh: () => refreshHeaderControlState(true), mutate: withStableViewport,
        decision: () => chatDecision, revealQuestion: (...a) => revealQuestion(...a), reading: () => reading }) : null;
    const chatDecision = createChatDecision({
        apiFetch,
        frameNode: chatMedia.bubbleFrameNode,
        renderMaterial: chatMedia.buildDocumentBubble,
        releaseMaterials: chatMedia.release,
        mountMarkdown: mountChatMarkdown,
        enhanceMarkdown: enhanceMountedMarkdown,
        showToast,
        fetchDetail: fetchTaskDetailStrict,
        onDomWrite: withStableViewport,
        isMain, chatId, projectId,
        insertMessageNode,
        onQuestionsChanged: mainWaiting?.refresh,
        // A settled Main question mirror leaves through the ordinary retirement path.
        removeMessageNode: (node) => withStableViewport(() => { releaseMessageNode(node); return true; },
            { excludeAnchorNode: node }),
        focusAfterRemoval: () => input?.focus?.({ preventScroll: true }),
    });

    async function loadUiPreferences() {
        try {
            const prefs = await apiClient.uiPreferences();
            if (destroyed) return;
            nestedSubagentsExpanded = prefs?.nested_subagents_expanded === true;
            emptyWelcome?.setPreference(prefs?.welcome);
        } catch {
            nestedSubagentsExpanded = false;
        }
    }

    // Sent attachment messages not yet confirmed saved: "Send again" with the same id, kept by this tab across a reload.
    const unconfirmed = createUnconfirmedSends({ send: (frame, options) => ws.send(frame, options),
        root: () => messagesDiv, onDomWrite: withStableViewport, showToast, compose: replies.compose,
        storage: sessionStorage, storageKey: storeKey('ouro_chat_unconfirmed') });
    const ownProjectId = projectId;

    let _syncPass1Active = false;
    let _historyReplayActive = false;
    let _historyRow = null;
    let _historyAppending = false;
    // While a page is replayed, every identity key a mounted bubble carries (a superset: keys are added, never taken
    // back): a row of the page looks for its prior bubble only when one could be mounted, instead of scanning the feed
    // once per row (O(rows²) per page).
    let replayKeys = null;
    // ...and the selection as the reader left it: every mounted row the page moves keeps it (one read for the page, not
    // one per move: in Chromium a selection read lays the page out).
    let replaySelection = null;
    function noteBubbleKeys(node) {
        if (!replayKeys || !node) return;
        const nodes = node.classList?.contains('chat-bubble') ? [node] : [];
        for (const bubble of [...nodes, ...(node.querySelectorAll?.('.chat-bubble') || [])]) {
            const { historyId, clientMessageId, originId, messageKey } = bubble.dataset || {};
            if (historyId) replayKeys.history.add(historyId);
            if (clientMessageId) replayKeys.client.add(clientMessageId);
            if (originId) replayKeys.origin.add(originId);
            if (messageKey) replayKeys.legacy.add(messageKey);
        }
    }

    const persistedHistory = [];
    const seenMessageKeys = new Set();
    const messageKeyOrder = [];
    const pendingUserBubbles = new Map();
    let historyLoaded = false;
    // A source read has painted the room. The sessionStorage preview a failed first
    // read leaves is not one: the first source read after it still opens the room.
    let sourceHydrated = false;
    let historySyncPromise = null;
    let lastHistorySyncSucceeded = false;
    // STICKY single-flight hydration promise.
    // Unlike historySyncPromise it survives success, so hydration triggers
    // (bootstrap IIFE, first non-reconnect socket open, refreshHistory without
    // a new revision) short-circuit instead of refetching. Any FAILED sync
    // resets it; scheduleHistorySync and the reconnect path never consult it.
    let initialHydrationPromise = null;
    // one-shot idle gate for Main's deferred first hydration.
    let hydrationGatePromise = null;
    // One derived physical coverage/status for the mounted reading window.
    let historyWindow = null;
    let _hasNewActivity = false;
    // A Project room app.js retired is leaving the screen (it fades out inside the room card): not shown, not read.
    let leaving = false;
    const isInstanceVisible = () =>
        !leaving && Boolean(messagesDiv) && messagesDiv.offsetParent !== null && !document.hidden;
    const LIVE_CARD_CAP = 200;
    const liveCardBound = createLiveCardBound(LIVE_CARD_CAP);
    const liveCardRecords = new Map();
    // The server's work projection (work_status.js), re-read on facts only.
    let roomWork = null, workState = emptyWorkState();
    const reread = () => void refreshHeaderControlState(true);
    // A wait creates its task record; its open/closed state is one
    // input of the block predicate, so the block appears with the controls and
    // leaves with them (the record and its no-reopen wait ledger stay).
    const modelWaits = createModelWaitController({
        getRecord: (id, create = true) => (create ? getLiveCardRecord(id) : liveCardRecords.get(id)),
        openSettings: openSettingsTab, onDomWrite: withStableViewport, notify: showToast,
        onChange: (id) => { ensureLiveCardVisible(liveCardRecords.get(id)); syncChatStatus(); },
    });
    function disposeLiveCard(id, preserveWaits = false) {
        if (!preserveWaits) modelWaits.forget(id);
        const record = liveCardRecords.get(id);
        record?.timelineDispose?.();
        record?.episodeDisclosureDispose?.();
        if (record?.episodeDisclosure) releaseMessageNode(record.episodeDisclosure);
        if (record?.root) releaseMessageNode(record.root);
        liveCardRecords.delete(id);
    }
    const markReviewAnchor = (r, on = false) => setReviewAnchor(r, on, setLiveCardPhase);
    const explicitCardExpansion = new Map();
    const reviewDisclosureByTask = new Map();
    const reviewHydrator = createReviewHydrator({
        fetchDetail: fetchTaskDetailStrict,
        onSettled: () => !destroyed && reading.position(),
        applyDetail: (id, detail) => !destroyed && attachTaskDetailReviews(id, detail),
        onState: (id, status) => !destroyed
            && (liveCardRecords.get(id)?.reviewController?.setHydrateStatus?.(status) ?? false),
    });
    // Buffer task_named frames that precede their card.
    const pendingSuggestedNames = new Map();
    // Server-confirmed direct/managed activities.
    const activeDirectActivities = new Map();
    // Local user submissions awaiting server confirmation (clientMessageId
    // -> { clientMessageId, timestamp }).
    const pendingSubmissions = new Map();
    const hostStart = createHostStart(); // В9 readiness and boot phases; a disconnect forgets both
    // Bounded conclusions retain known outcomes for later cards; reusable slots clear on settlement.
    const concludedDirectActivities = new Map();
    const CONCLUDED_ACTIVITY_LEDGER_MAX = 200;
    // Queue-loss candidates and single-flight detail reads.
    const missingManagedTaskIds = new Set();
    const managedTaskDetailReads = new Set();
    // Owner rows kept until history confirms client_message_id.
    const localEchoJournal = new Map();
    const LOCAL_ECHO_JOURNAL_MAX = 50;

    function recordLocalEcho(clientMessageId, text, ts) {
        if (!clientMessageId) return;
        localEchoJournal.set(clientMessageId, { clientMessageId, text, ts, annotation: null });
        while (localEchoJournal.size > LOCAL_ECHO_JOURNAL_MAX) {
            localEchoJournal.delete(localEchoJournal.keys().next().value);
        }
    }

    function recordConcludedActivity(activityId, phase = '') {
        const aid = taskKey(activityId);
        if (!aid) return;
        missingManagedTaskIds.delete(aid);
        phase ||= concludedDirectActivities.get(aid) || '';
        concludedDirectActivities.delete(aid);
        concludedDirectActivities.set(aid, phase);
        while (concludedDirectActivities.size > CONCLUDED_ACTIVITY_LEDGER_MAX) {
            concludedDirectActivities.delete(concludedDirectActivities.keys().next().value);
        }
    }
    function recordTerminalActivity(taskId, phase = '') {
        const id = taskKey(taskId);
        if (!id) return;
        activeDirectActivities.delete(id);
        missingManagedTaskIds.delete(id);
        if (REUSABLE_TASK_IDS.has(id)) concludedDirectActivities.delete(id);
        else recordConcludedActivity(id, phase);
        settleTerminalRootChildren(id);
    }
    // A terminal root reconciles open descendants through single-flight detail
    // reads; only proven child terminality changes their state (#300).
    function settleTerminalRootChildren(rootId) {
        for (const [childId, info] of subagentChildParents) {
            if (info.parentId !== rootId) continue;
            settleTerminalRootChildren(childId);
            const record = liveCardRecords.get(childId);
            if (!record || record.finished || managedTaskDetailReads.has(childId)) continue;
            managedTaskDetailReads.add(childId);
            fetchTaskDetail(childId).then((detail) => {
                if (destroyed || !isTerminalTaskDetail(detail)) return;
                withStableViewport(() => routeSubagentTerminalToCard(childId, { ...detail, task_id: childId }));
            }).catch(() => {}).finally(() => managedTaskDetailReads.delete(childId));
        }
    }
    // Root census rows omit children; complete work membership can admit their own
    // result read before the root ends. Unknown/nonterminal replies retry on the
    // existing refresh; a confirmed 404 waits for new card evidence or reconnect.
    function settleAbsentChildren(snapshotRequestedAt) {
        for (const [childId, record] of liveCardRecords) {
            if (!record.isSubagent || record.finished || !record.root?.isConnected) continue;
            if (workState.members.has(childId)) {
                record.childDetailRead = undefined;
                if (record.historicalUnavailable) withStableViewport(() => {
                    setHistoricalUnavailable(record, false);
                    renderLiveCardMeta(record);
                    return true;
                });
                continue;
            }
            const observedAt = record.lastLiveObservedAt || 0;
            const updates = record.updates;
            if (!workState.complete || observedAt >= snapshotRequestedAt
                    || (record.childDetailRead?.missing && record.childDetailRead.updates === updates)
                    || managedTaskDetailReads.has(childId)
                    || !subagentChildParents.has(childId)) continue;
            // Queue departure and membership share this read; keep the visible hold
            // until actual recovery or terminal evidence, not just absence.
            record.projectHoldQueued = false;
            const observation = record.childDetailRead = { updates };
            managedTaskDetailReads.add(childId);
            fetchTaskDetailStrict(childId).then((detail) => {
                if (destroyed || ws.isConnected?.() === false || !workState.complete
                        || liveCardRecords.get(childId) !== record || record.finished
                        || workState.members.has(childId) || record.updates !== updates
                        || record.childDetailRead !== observation) return;
                if (detail === null) {
                    observation.missing = true;
                    withStableViewport(() => settleMissingTaskRecord(record));
                } else if (isTerminalTaskDetail(detail)) {
                    withStableViewport(() => routeSubagentTerminalToCard(childId, { ...detail, task_id: childId }));
                }
            }).catch(() => {}).finally(() => managedTaskDetailReads.delete(childId));
        }
    }
    let activeLiveGroupId = '';
    let pendingReconnectSync = false;  // Set when a fromReconnect sync arrives while one is already in-flight.
    let pendingRestartNotice = '';
    try {
        pendingRestartNotice = restartNoticeText(new URL(window.location.href).searchParams.get('_ouro_reason') || '');
    } catch {}
    function clearReloadParams() {
        try {
            const url = new URL(window.location.href);
            if (!url.searchParams.has('_ouro_reason') && !url.searchParams.has('_ouro_refresh')) return;
            url.searchParams.delete('_ouro_reason');
            url.searchParams.delete('_ouro_refresh');
            window.history.replaceState({}, '', url);
        } catch {}
    }

    function rememberMessageKey(key) {
        if (!key || seenMessageKeys.has(key)) return;
        seenMessageKeys.add(key);
        messageKeyOrder.push(key);
        if (messageKeyOrder.length > 2000) {
            const oldest = messageKeyOrder.shift();
            if (oldest) seenMessageKeys.delete(oldest);
        }
    }

    function setStatus({ kind, text, step = '', tone, mark, owner, amber, title = '' }) {
        // Replay frames never touch the badge; syncChatStatus writes it once.
        if (!statusBadge) return;
        statusBadge.className = `status-badge ${kind}`;
        statusBadge.textContent = text;
        statusBadge.title = title; // a failed boot step's cause; empty otherwise
        Object.assign(statusBadge.dataset, { tone, mark, owner: owner ? '1' : '', amber, step });
        roomWork?.update();
    }

    function syncHeaderControlState(data) {
        composer?.syncState(data);
        header?.update(data);
    }

    // Main: a moved work's card is its portal (work_portal.js); Continue episodes share one card (task_episodes.js).
    const portals = createWorkPortals({ isMain, feed: messagesDiv, records: liveCardRecords, onDomWrite: withStableViewport,
        isConnected: () => ws.isConnected?.() !== false, onConverted: () => composer?.signalFreed(), remember: rememberMessageKey,
        append: (msg, text, opts) => appendProjectRow(addMessage, msg, text, opts), actionsOf: ensureLiveActionsEl,
        card: { get: getLiveCardRecord, show: ensureLiveCardVisible, reanchor: reanchorTaskCard, restore: restoreCardActivity,
            meta: renderLiveCardMeta, controls: syncCancelRunButton, finish: finishLiveCardMutation, note: noteTaskDetails,
            missing: observeMissingManagedTask, summary: appendTaskSummaryToLiveCard,
            collapse: (record) => (explicitCardExpansion.delete(record.groupId), record.root.dataset.expanded === '1'
                && setLiveCardExpanded(record, false)) } });
    const episodes = createTaskEpisodes({ records: liveCardRecords, reanchor: reanchorTaskCard, renamed: portals.relabel,
        activity: (record) => currentActivity(record), follow: portals.follow,
        expand: (record) => { setLiveCardExpanded(record, true); hydrateCardReviews(record.groupId); } });
    const rowMenu = createChatRowMenu({ feed: messagesDiv, replies, cardOf: (id) => liveCardRecords.get(id),
        actionsOf: (record) => cardMenuActions(workState, record, currentActivity(record)), afterAction: reread });

    let childHoldRead = false;
    async function refreshChildProjectHolds() {
        const children = [...liveCardRecords.values()].filter(r => r.isSubagent && !r.finished && r.root?.isConnected);
        if (destroyed || childHoldRead || !children.length) return;
        childHoldRead = true;
        const started = Date.now();
        try {
            const response = await apiFetch('/api/tasks?queue_only=1');
            if (!response.ok) return;
            const { queue } = await response.json();
            if (destroyed || !Array.isArray(queue?.pending) || !Array.isArray(queue?.running)) return;
            const tasks = new Map([[queue.pending, 'queued'], [queue.running, 'working']]
                .flatMap(([rows, phase]) => rows.map(row => [row.id, { ...row.task, phase }])));
            withStableViewport(() => {
                for (const record of children) {
                    if (record.finished || !record.root?.isConnected || record.lastLiveObservedAt > started) continue;
                    const task = tasks.get(record.groupId);
                    if (task && 'project_admission_hold' in task) {
                        missingManagedTaskIds.delete(record.groupId);
                        restoreCardActivity(record, task.project_admission_hold);
                        syncParkedPhase(record, ws.isConnected?.() === false ? 'unknown' : task.phase, task);
                        record.projectHoldQueued = Boolean(record.projectHold);
                    } else if (!task && record.projectHoldQueued) {
                        record.projectHoldQueued = false;
                        missingManagedTaskIds.add(record.groupId);
                        void reconcileMissingManagedTask(record.groupId);
                    }
                }
            });
        } catch { /* Keep the last fact; absence or a failed read proves no recovery. */ }
        finally { childHoldRead = false; }
    }

    function hydrateStateSnapshot(data, snapshotRequestedAt = Infinity) {
        if (data) void refreshChildProjectHolds();
        const before = workState;
        workState = reconcileWorkState(workState, data, chatId, ws.isConnected?.() !== false);
        if (Array.isArray(data?.works)) settleAbsentChildren(snapshotRequestedAt);
        applyCardWorkViews(liveCardRecords, workState);
        changedOffers(liveCardRecords, before, workState).forEach(syncCancelRunButton);
        syncHeaderControlState(data ?? { accounting: { available: false } });
        portals.sync(data);
        mainWaiting?.snapshot(data, snapshotRequestedAt);
        hostStart.read(data);
        const activities = Array.isArray(data?.active_chat_activities)
            ? data.active_chat_activities
            : data?.active_direct_turns;
        if (Array.isArray(activities)) {
            withStableViewport(() => {
                hydrateDirectActivities(activities, snapshotRequestedAt,
                    data.active_chat_activities_complete === true && data.supervisor_ready === true);
                return true;
            });
            for (const activity of activities) {
                if (activity.required_question) chatDecision.appendActivityQuestion(activity.required_question, snapshotRequestedAt);
                if (Number(activity.chat_id ?? 1) === chatId) modelWaits.observe(activity.activity_id, activity);
            }
        }
        syncChatStatus();
    }

    async function refreshHeaderControlState(force = false) {
        if (!force && state.activePage !== 'chat') return;
        const request = await stateSnapshots.gate(force);
        if (!request) return;
        try {
            const resp = await apiFetch(request.path || '/api/state', { cache: 'no-store' });
            if (!resp.ok) throw new Error(`State read failed: HTTP ${resp.status}`);
            stateSnapshots.apply(request, await resp.json());
        } catch {
            stateSnapshots.fail?.(request);
        }
    }

    function persistVisibleHistory() {
        try {
            sessionStorage.setItem(storeKey(CHAT_STORAGE_KEY), JSON.stringify(persistedHistory.slice(-200)));
        } catch {}
    }

    const ACTUAL_BOTTOM_TOLERANCE_PX = 6;

    const isNearBottom = threshold => reading.nearBottom(threshold);

    const { captureVisibleTimelineAnchor, restoreVisibleTimelineAnchor, serializeTimelineAnchor, anchorOwnersReady } =
        createTimelineAnchors({ messagesDiv, liveCardRecords });
    // Far rows skipped once laid out (style.css); after a restyle every row's size may change at once, and the reading
    // owner places the passage again.
    const rowSleep = createRowSleep({ feed: messagesDiv, onRestyle: () => reading.reflow() });
    // A fitted line re-fits on the frame after its width changed (text_fit.js): a reader at the newest message is placed
    // there again once the room's lines have their size (Main shown again laid out its newest message before).
    const offRefit = onRefit((els) => { if (reading.stick && els.some((el) => messagesDiv.contains(el))) reading.reflow(); });
    const reading = createChatReadingPosition({
        feed: messagesDiv, alive: () => !destroyed,
        // A row asleep that the write changed is laid out before the place is restored (chat_row_sleep.js).
        afterWrite: () => { rowSleep.flush(); roomWork?.update(); }, activity: () => { _hasNewActivity = true; },
        updateButton: updateScrollButton,
        visible: () => !destroyed && isInstanceVisible(),
        ready: () => historyLoaded && anchorOwnersReady(reading.target?.historyAnchor, reviewHydrator.ready),
        anchors: { serialize: serializeTimelineAnchor, restore: restoreVisibleTimelineAnchor, capture: captureVisibleTimelineAnchor },
        // A settled place is where the reader is (the read receipt).
        changed: () => { syncLoadOlderControl(); updateScrollButton(); readReceipt.note({ discrete: true }); },
    });

    function withStableViewport(mutate, options) { return reading.mutate(mutate, options); }
    // Turned back into history, or on the way to a place: not following the newest message.
    const readingHistory = () => !reading.stick || reading.pending;

    function withRemoteActivity(mutate) {
        _remoteActivityDepth += 1;
        try {
            return withStableViewport(mutate, { remoteContent: true });
        } finally {
            _remoteActivityDepth -= 1;
        }
    }

    function enhanceMountedMarkdown(root) {
        // Late diagrams/fallbacks can add keyboard stops after the fold was measured.
        // Refresh only their own bubble, inside the existing viewport transaction.
        const foldedWrite = writer => (mutate, options) => writer(() => {
            const changed = mutate();
            refreshProjectAnswerFold(root.closest?.('.project-answer'));
            return changed;
        }, options);
        return enhanceChatMarkdown(root, {
            onDomWrite: foldedWrite(_remoteActivityDepth > 0 ? withRemoteActivity : withStableViewport),
            onThemeDomWrite: foldedWrite(withStableViewport),
        });
    }
    const {
        renderLiveCardTimeline,
        appendTimelineItem,
        patchLastTimelineItem,
        patchTimelineItemAt,
    } = createLiveCardTimelineRenderer({
        withStableViewport, buildTimelineItemHtml, isReplayActive: () => _historyReplayActive,
    });

    function insertMessageNode(node, options = {}) {
        if (!node) return false;
        const isMounted = node.parentNode === messagesDiv;
        if (isMounted && !options.reorderExisting) {
            return false;
        }
        return withStableViewport(() => {
            // Scope to THIS instance's column — a global id lookup would resolve to
            // the first panel's typing node and misplace project-thread messages. It is
            // the feed's last child (rows go before it): no walk of the feed per row.
            const last = messagesDiv.lastElementChild;
            const typing = last?.classList?.contains('typing-bubble') ? last : messagesDiv.querySelector('.typing-bubble');
            insertTimelineNode(messagesDiv, node, typing, replaySelection);
            noteBubbleKeys(node);
            replies.syncDerivedQuotes(node);  // a landing row may repeat, or stop repeating, the quote below it
            return true;
        }, {
            forceFollow: Boolean(options.forceStick),
            excludeAnchorNode: isMounted ? node : null,
        });
    }

    function reanchorTaskCard(
        record,
        rawTs,
        { suppressDomInsert = false } = {},
        seen = new Set(),
    ) {
        if (!record || seen.has(record.groupId)) return false;
        seen.add(record.groupId);
        if (record.isSubagent) {
            const parent = liveCardRecords.get(record.parentGroupId);
            return reanchorTaskCard(parent, rawTs, { suppressDomInsert }, seen);
        }
        const movedEarlier = stampNodeTimestamp(record.root, rawTs, { anchor: true });
        const position = _historyRow?.history_position;
        const positionChanged = position && (!record.historyPosition
            || compareHistoryPosition(position, record.historyPosition) < 0);
        if (positionChanged) {
            record.historyPosition = position;
            record.root.dataset.historySource = position.source;
            record.root.dataset.historyOffset = String(position.offset);
        }
        if (!movedEarlier && !positionChanged) return false;
        if (suppressDomInsert || _syncPass1Active) {
            record._anchorOrderDirty = true;
            return true;
        }
        ensureLiveCardVisible(record, { reorderExisting: true });
        record._anchorOrderDirty = false;
        return true;
    }

    // DESIGN's activity-block predicate rechecks facts on each mutation;
    // completion notes and receipts are not content, and visibility is not sticky.
    function blockVisible(record) {
        if (!record || record.isSubagent) return true;
        // A moved work's portal never leaves Main (one work, one card).
        return isPortal(record) || blockHasWork(record)
            || !!(record.modelWaiting || record.cancelPendingPolicy || record.reviewAnchor)
            || stopEligible(record)
            || (record.finished && record.phaseEl?.dataset?.phase !== 'done');
    }

    // Work chooses card chrome (owner 16.09): title and Main conversion, regardless
    // of lane. Attention-only or non-Done endings have neither title nor conversion.
    // Bare outcome/receipts are not work; _is_direct_chat retains its host duties.
    function blockHasWork(record) {
        const id = record.groupId;
        // Visibility depends on work, never the lane.
        return record.reviewController?.groups.size > 0
            || [...subagentChildParents.values()].some((info) => info.parentId === id)
            || record.items.some((item) => !item.receipt && !String(item.dedupeKey || '').startsWith('task_done|'))
            || record.toolErrors > 0;
    }

    // Child, else project, identity names the card; authored activity stays in its timeline. An unnamed
    // root with work takes its newest narration, `Task activity` once ended (#1369).
    function cardTitle(record, work = blockHasWork(record)) {
        return record.isSubagent ? childTitle(record) : record.suggestedName
            || (work ? record.lastHumanHeadline || (record.finished ? 'Task activity' : '') : '');
    }

    // Write only on change: a rewrite would destroy a selection being copied. A finished root's own sentence is its
    // line under an authored title; a wake's sentence is its title, so its last narration stays the line; a host cause
    // holding the line stays.
    function syncCardHeadline(record) {
        const title = cardTitle(record);
        const own = record.finished && !record.isSubagent && !record.causeHeld && record.wakeSummary;
        const line = own && own !== title ? own : record.collapsedActivity;
        if (record.titleEl.textContent !== title) { record.titleEl.textContent = title; portals.relabel(record); }
        renderCollapsedActivity(record, projectCollapsedActivity({ isSubagent: record.isSubagent, suggestedName: title, headline: line, body: line }));
    }

    // Fold host facts; absent counters stay absent.
    function noteToolMetrics(taskId, metrics, rawTs, { suppressDomInsert = false } = {}) {
        const rounds = Number.isInteger(metrics?.rounds) && metrics.rounds >= 0 ? metrics.rounds : null;
        const [calls, errors, routing, completion] = ['tool_calls', 'tool_errors', 'routing_tool_calls', 'completion_tool_calls']
            .map(key => Number.isInteger(metrics?.[key]) && metrics[key] >= 0 ? metrics[key] : null);
        const hasToolEvidence = calls > 0 || errors > 0 || metrics.tool_evidence?.observations?.length
            || metrics.tool_evidence?.legacy?.calls || toolEvidenceIncomplete(metrics.tool_evidence?.coverage);
        if (rounds === null && calls === null && !hasToolEvidence) return false;
        return withStableViewport(() => {
            const record = getLiveCardRecord(taskId);
            const before = captureLiveCardProjection(record);
            const duration = Number(metrics.duration_sec);
            if (rounds !== null) record.rounds = Math.max(record.rounds ?? 0, rounds);
            if (metrics.duration_sec != null && Number.isFinite(duration)) record.durationSec = duration;
            if (calls !== null) record.toolCalls = Math.max(record.toolCalls ?? 0, calls);
            let changed = false;
            // Rounds and a known zero are facts, not a tool invocation or a row of work.
            if (hasToolEvidence) {
                const summary = noteToolHostMetrics(record, { calls, errors, routing, completion, counts: metrics.tool_call_counts, evidence: metrics.tool_evidence });
                if (summary.calls > 0 || record.toolCalls != null) record.toolCalls = summary.calls;
                record.toolErrors = summary.errors;
                changed = placeReplayedEvidence(record, summary, metrics.tool_evidence, rawTs, normalizeLogTs)
                    && renderLiveCardTimeline(record);
            }
            updateLiveCardCount(record);
            renderLiveCardMeta(record);
            // A timestamp-free current-work carrier intentionally belongs at the page end.
            // Dated carriers may carry earlier evidence past an intervening owner message.
            const anchor = Number.isFinite(Date.parse(rawTs)) ? replayedEvidenceTs(metrics.tool_evidence, rawTs) : rawTs;
            reanchorTaskCard(record, anchor, { suppressDomInsert });
            ensureLiveCardVisible(record, { suppressDomInsert });
            return Boolean(changed || liveCardProjectionChanged(before, record));
        });
    }

    // P5: task ids whose progress carried the supervisor's host-attested
    // `cancelable` marker (queue tasks the cancel endpoint can genuinely reach).
    // Learned from live WS frames and history replay alike, possibly before the
    // card exists, so it lives beside the card records rather than on them.
    const cancelableTaskIds = new Set();

    function queueTaskLiveUpdate(summary, taskId, ts, dedupeKey = '', rawTs = '') {
        return withStableViewport(() => {
            // Explicit task/review identity owns a card; viewport position never does.
            if (!taskId) return false;
            let changed = false;
            const record = liveCardRecords.get(taskId);
            // Only reusable slots replace finished cards; other late frames update
            // cost/model facts in applyLiveCardState without reviving the phase.
            if (!_historyRow && record?.finished && REUSABLE_TASK_IDS.has(taskId)
                    && !isTerminalTaskPhase(summary.phase || '', summary.terminal)) {
                changed = Boolean(record.root?.isConnected);
                disposeLiveCard(taskId);
            }
            return Boolean(applyLiveCardState(summary, taskId, ts, dedupeKey, { rawTs }) || changed);
        });
    }

    // Main blocks with work offer conversion unless their origin is bound (the same /api/state fact app.js
    // reads); titles use the same work predicate. Every root work card offers Reply in its row menu.
    function syncBlockChrome(record) {
        replies.syncCard(record, !record.isSubagent && (isPortal(record) || blockHasWork(record)));
        // A child is never a convertible unit (it inherits its root's Project by
        // lineage) and its title is its role: both root writers skip it.
        const work = !record.isSubagent && blockHasWork(record);
        // The first row of work lands after the title writers ran for its frame: an empty title
        // takes a name here (Working is the chip's word, #1369); the writers own it from then on.
        if ((work || isPortal(record)) && !record.titleEl.textContent) record.titleEl.textContent = cardTitle(record, true);
        portals.syncConvert(record, isMain && work && !isPortal(record) && !(window.__ouroTaskBindings || {})[record.groupId]);
    }

    function syncCancelRunButton(record) {
        return withStableViewport(() => syncCancelRunButtonMutation(record));
    }

    // The live activity a card's controls act on: this chat's census row, or — for a Main portal —
    // its work's row in the room's chat (the portal carries the same Stop/Resume on every surface).
    const currentActivity = (record) => activeDirectActivities.get(record?.groupId) || portals.activity(record);

    // Shared Stop predicate: a hidden button cannot justify a visible block.
    const stopEligible = (record) => cancelRunEligibility({
        groupId: record.groupId, isSubagent: record.isSubagent, finished: record.finished,
        cancelable: !record.historicalUnavailable && !record.historicalUnconfirmed
            && (cancelableTaskIds.has(record.groupId) || activeDirectActivities.has(record.groupId) && blockHasWork(record)
                || Boolean(portals.activity(record))),
    });

    function syncCancelRunButtonMutation(record) {
        if (!record?.root) return false;
        // Resume where the card's work offers it.
        const canResume = () => cardControls(workState, record.groupId).resume;
        const resumable = canResume();
        let changed = false;
        if (resumable && !record.resumeRunBtn) {
            record.resumeRunBtn = createTaskResumeButton(record.groupId, { canResume,
                onSettled: () => refreshHeaderControlState(true) });
            // One order everywhere: [Turn into project] [Resume] [Stop…].
            const actions = ensureLiveActionsEl(record);
            actions?.insertBefore(record.resumeRunBtn, actions.querySelector('[data-cancel-run]'));
            changed = true;
        } else if (!resumable && record.resumeRunBtn) {
            record.resumeRunBtn.remove(); record.resumeRunBtn = null; changed = true;
        }
        const eligible = stopEligible(record) || resumable || cardControls(workState, record.groupId).menu;
        const existing = record.root.querySelector('[data-cancel-run]');
        if (!eligible) {
            if (!existing) return changed;
            existing.remove();
            record.cancelRunBtn = null;
            return true;
        }
        if (existing) {
            record.cancelRunBtn = existing;
            if (record.finished) existing.disabled = false;
            return changed;
        }
        const actions = ensureLiveActionsEl(record);
        if (!actions) return false;
        const btn = document.createElement('button');
        btn.type = 'button';
        // Neutral trigger: Resume or Continue stays the card's one primary.
        btn.className = 'btn btn-sm btn-default';
        btn.dataset.cancelRun = '1';
        btn.textContent = TASK_CONTROL_TRIGGER_LABEL;
        // Pending cancel offers only hard escalation; dismiss keeps the run.
        btn.addEventListener('click', (event) => {
            event.stopPropagation();
            openCardControls(btn, record, cardControls(workState, record.groupId).work, { activity: currentActivity(record),
                stop: (action) => cancelRunFromCard(record, action), afterAction: reread });
        });
        actions.appendChild(btn);
        record.cancelRunBtn = btn;
        return true;
    }

    // Pending intent stays live until settled; soft stop shows Finalizing….
    function markLiveCardCancelPending(taskId = '', soft = false) {
        const record = liveCardRecords.get(taskKey(taskId));
        if (!record || record.finished || !record.phaseEl) return false;
        record.cancelPendingPolicy = soft ? 'finalize' : 'immediate';
        record.finalizingHold = false;  // owner cancel outranks the hold
        return setLiveCardPhase(
            record, 'working', soft ? 'Finalizing…' : 'Cancelling…',
            'chat-live-phase working cancelling',
        );
    }

    // Early outcome is not completion.
    function markLiveCardFinalizing(taskId = '', fact = {}) {
        return withStableViewport(() => {
            const record = liveCardRecords.get(taskKey(taskId));
            if (!record || record.finished || !record.phaseEl) return false;
            const anchored = markReviewAnchor(record);
            if (record.cancelPendingPolicy) return anchored;
            record.observedOutcome = taskTerminalSummary({ ...fact, task_phase: 'finalizing' }).observedOutcome || record.observedOutcome;
            record.finalizingHold = true;
            const desired = desiredLiveCardPhase(record);
            const phased = setLiveCardPhase(
                record, desired.phase, desired.text, desired.className, desired.secondary,
            );
            return Boolean(anchored || phased);
        });
    }

    // Durable cancel state wins over legacy status; only settled truth closes the card.
    function reconcileCancelCardFromDetail(record, taskId, stored) {
        return withStableViewport(() => {
            if (!stored || !record || record.finished) return false;
            if (taskCancelPending(stored)) {
                return markLiveCardCancelPending(taskId, taskSoftStopPending(stored));
            }
            if (!taskDoneIsTerminal(stored)) return false;
            return appendTaskSummaryToLiveCard({ ...stored, task_id: taskId });
        });
    }

    async function cancelRunFromCard(record, action = '') {
        const taskId = taskKey(record?.groupId);
        if (!taskId || record.finished) return;
        // Q2: the dropdown itself is the confirmation surface — dismissing it
        // continued the run, so a selected action executes immediately.
        const soft = action === ACTION_FINALIZE;
        const btn = record.cancelRunBtn;
        const priorPhase = captureLiveCardPhaseState(record);
        withStableViewport(() => {
            if (btn) btn.disabled = true;
            return markLiveCardCancelPending(taskId, soft);
        });
        try {
            await requestStop(taskId, action);
            // Durable detail heals lost best-effort task_done publication
            try {
                reconcileCancelCardFromDetail(record, taskId, await fetchTaskDetail(taskId));
            } catch {
                // The card still resolves on its own frame if one arrives.
            }
            // Soft stop keeps hard escalation reachable while finalizing.
            if (btn && !record.finished && record.cancelPendingPolicy === 'finalize') {
                btn.disabled = false;
            }
        } catch (exc) {
            // Said once, by the request; 404 is the task already gone.
            if (exc?.status !== 404) showToast(`Cancel failed: ${exc?.message || exc}`, 'error');
            if (exc?.status === 404 || record.finished) {
                // Completion won: remove the dead action, then reconcile detail.
                revokeManagedTaskCancelAuthority(taskId);
                try {
                    reconcileCancelCardFromDetail(record, taskId, await fetchTaskDetail(taskId));
                } catch {
                    // A later terminal frame can still resolve the card.
                }
                return;
            }
            // Reconcile durable truth before restoring any optimistic UI.
            let stored = null;
            try {
                stored = await fetchTaskDetail(taskId);
            } catch {}
            if (stored === null) {
                // Only a fetched, live, non-pending detail restores the button.
                return;
            }
            withStableViewport(() => {
                let changed = reconcileCancelCardFromDetail(record, taskId, stored);
                if (record.finished || taskCancelPending(stored)) return changed;
                if (btn) btn.disabled = false;
                const restoredPhase = restoreLiveCardPhaseState(record, priorPhase);
                if (restoredPhase) {
                    changed = setLiveCardPhase(
                        record, restoredPhase.phase, restoredPhase.text, restoredPhase.className,
                        restoredPhase.secondary,
                    ) || changed;
                }
                return changed;
            });
        }
    }

    function restoreCardActivity(record, held = {}) {
        if (!setHistoricalUnavailable(record, false, held)) return;
        renderLiveCardMeta(record);
        syncCancelRunButton(record);
    }

    function settleMissingTaskRecord(record) {
        const historical = record.historicalTerminal;
        if (historical) {
            const changed = applyHistoricalModelExecution(record, historical) && renderLiveCardMeta(record);
            return finishLiveCard(record.groupId, historical.phase) || changed;
        }
        const changed = setHistoricalUnavailable(record, true);
        renderLiveCardMeta(record);
        syncChatStatus();
        return changed;
    }

    function markTaskCancelable(taskId = '') {
        const id = taskKey(taskId);
        if (!id) return false;
        cancelableTaskIds.add(id);
        return syncCancelRunButton(liveCardRecords.get(id));
    }

    const reviewAnchorEligible = (id) => !liveCardRecords.has(id) && !activeDirectActivities.has(id);

    function attachReviewGroup(group, rawTs = '') {
        const ownerTaskId = taskKey(group?.presentationOwnerTaskId);
        if (!ownerTaskId) return false;
        const reviewAnchor = reviewAnchorEligible(ownerTaskId);
        const record = getLiveCardRecord(ownerTaskId);
        if (reviewAnchor) markReviewAnchor(record, true);
        const merged = record.reviewController?.update(group);
        const wasMounted = Boolean(record.root?.isConnected);
        if (rawTs) reanchorTaskCard(record, rawTs);
        ensureLiveCardVisible(record);
        const mounted = !wasMounted && Boolean(record.root?.isConnected);
        if (merged && !record.reviewOwnerDetailObserved) {
            record.reviewOwnerDetailObserved = true;
            observeMissingManagedTask(
                ownerTaskId,
                _remoteActivityDepth > 0 ? withRemoteActivity : withStableViewport,
            );
        }
        return Boolean(merged || mounted);
    }

    function attachTaskDetailReviews(taskId, detail) {
        return withStableViewport(() => {
            const id = taskKey(taskId);
            modelWaits.observe(id, detail);
            const filed = noteTaskDetails(liveCardRecords.get(id), detail);
            const groups = reviewGroupsFromTaskDetail(detail, id);
            if (!id || groups.length === 0) return filed;
            const fresh = !liveCardRecords.has(id);
            const record = getLiveCardRecord(id);
            if (fresh) reanchorTaskCard(record, detail?.ts || detail?.timestamp || '');
            const changed = record.reviewController.updateMany(groups);
            ensureLiveCardVisible(record);
            const reconciled = reconcileCancelCardFromDetail(record, id, detail);
            return Boolean(changed || reconciled || filed);
        });
    }

    // Refresh files/history and a Swarm root's receipt line; retain lifecycle.
    function noteTaskDetails(record, detail) {
        const files = syncSettledItems(record, detail, { write: withStableViewport });
        const words = record && detail && 'wake_summary' in detail && record.wakeSummary !== detail.wake_summary;
        if (words) { record.wakeSummary = detail.wake_summary; syncCardHeadline(record); }  // live and replay meet here: the sentence is also the finished card's line
        const origin = record && detail?.schedule_name && record.scheduleName !== detail.schedule_name;
        if (origin) record.scheduleName = detail.schedule_name;
        const history = syncHistoryRetentionItem(record, detail);
        const candidate = syncCandidateRemainderItem(record, detail);
        const chain = episodes.note(record, detail);
        const receipt = syncSwarmReceipt(record, detail);
        return Boolean((words || origin || files || history || candidate) && (renderLiveCardMeta(record), updateLiveCardCount(record), renderLiveCardTimeline(record), true))
            || chain || receipt;
    }

    function hydrateCardReviews(taskId, revision = null) {
        return destroyed ? Promise.resolve(false) : reviewHydrator.hydrate(taskId, revision, {
            onDomWrite: _remoteActivityDepth > 0 ? withRemoteActivity : withStableViewport,
        });
    }

    function attachReviewFromRow(row, rawTs = '', showPointerAck = false) {
        const pointer = classifyReviewLifecyclePointer(row);
        if (pointer.classification !== 'not_pointer') {
            return withStableViewport(() => {
                const record = pointer.group && liveCardRecords.get(pointer.group.presentationOwnerTaskId);
                let changed = false;
                if (record?.reviewController) {
                    changed = Boolean(record.reviewController.update(pointer.group));
                } else if (showPointerAck) {
                    const ack = String(row?.text || row?.content || '').trim();
                    if (ack) changed = Boolean(addMessage(
                        ack, 'assistant', !!row?.markdown, rawTs || row?.ts || null,
                        true, { systemType: 'lifecycle_pointer' },
                    ));
                }
                return changed;
            });
        }
        const historyGroup = reviewGroupFromHistoryRow(row);
        if (historyGroup) {
            return withStableViewport(() => attachReviewGroup(
                historyGroup, rawTs || row?.ts || row?.timestamp || '',
            ));
        }
        const lifecycle = classifyReviewLifecycle(row);
        if (lifecycle.classification === 'source_complete') {
            return withStableViewport(() => {
                const attached = attachReviewGroup(lifecycle.group, rawTs || row?.ts || row?.timestamp || '');
                if (lifecycle.group.activeCount === 0 && lifecycle.group.lifecycleStatus) scheduleHistorySync();
                return attached;
            });
        }
        return lifecycle.classification === 'source_incomplete' ? false : undefined;
    }

    // Host keys place content without changing phase/expansion.
    // Reviews rehydrate; failures retain rows, unkeyed rows stay bubbles.
    // Pass 2 uses pass-1 cards without a type allowlist.
    function isPlacedCardRow(msg) {
        return CARD_ROW_PHASES.has(taskKey(msg?.card_row)) && !!taskKey(msg?.task_id);
    }

    function attachCardRow(msg, rawTs = '', { suppressDomInsert = false } = {}) {
        const placement = taskKey(msg?.card_row);
        const phase = CARD_ROW_PHASES.get(placement);
        const taskId = taskKey(msg?.task_id);
        const record = phase && taskId ? liveCardRecords.get(taskId) : null;
        if (!record) return undefined;
        const summary = cardRowSummary(msg, phase, rawTs);
        return withStableViewport(() => {
            const before = captureLiveCardProjection(record);
            let fresh;
            if (msg.history_id) {
                // Replay keeps source/order/page identity.
                fresh = mergeHistoricalTimelineItem(record, summary, msg, normalizeLogTs(rawTs));
            } else {
                const { timelineUpdate } = updateLiveTimelineItem(record, summary, {
                    ts: normalizeLogTs(rawTs), rawTs, syntheticKey: summary.dedupeKey, headline: summary.headline, inPlaceByKey: true,
                });
                fresh = !['none', 'duplicate-skip'].includes(timelineUpdate);
            }
            const changed = fresh ? renderLiveCardTimeline(record) : false;
            updateLiveCardCount(record);
            reanchorTaskCard(record, rawTs, { suppressDomInsert });
            ensureLiveCardVisible(record, { suppressDomInsert });
            // Only new review rows trigger a read.
            if (fresh && placement === 'reviews') hydrateCardReviews(taskId);
            return Boolean(changed || liveCardProjectionChanged(before, record));
        });
    }

    function admitCardMetadata(row) {
        // Carrier facts precede presentation returns.
        if ((row.tool_evidence || Number.isInteger(row.rounds)) && row.task_id) noteToolMetrics(row.task_id, row, row.ts || row.timestamp || '');
        if (row.system_type === 'task_evidence') {
            // The task's own evidence: mints its card at the carrier's row (zero tools included).
            const record = getLiveCardRecord(row.task_id);
            const activity = activeDirectActivities.get(row.task_id);
            if (row.ts) reanchorTaskCard(record, row.ts);
            if (row.suggested_name) applySuggestedName(row.task_id, row.suggested_name);
            if (row.task_phase === 'finalizing') markLiveCardFinalizing(row.task_id, row);
            else if (!row.task_terminal_status && !activity) setHistoricalUnconfirmed(record);
            if (activity) syncParkedPhase(record, activity.phase, activity);
            syncCancelRunButton(record);
            return true;
        }
        if (isModelWaitReference(row)) {
            const changed = modelWaits.observe(row.task_id, row);
            return row.outcome_axes ? appendTaskSummaryToLiveCard(row) || changed : changed;
        }
        const reference = reviewReferenceFromRow(row);
        if (!reference) return undefined;
        return withStableViewport(() => {
            const owner = reference.presentationOwnerTaskId;
            const anchor = reviewAnchorEligible(owner);
            const record = getLiveCardRecord(owner);
            const wasVisible = record.root?.isConnected;
            if (row?.ts) reanchorTaskCard(record, row.ts);
            const anchored = anchor && markReviewAnchor(record, true);
            ensureLiveCardVisible(record);
            hydrateCardReviews(owner, reference.stateRevision);
            // Task money follows its carrier, not the review owner.
            const costChanged = renderLiveCardMeta(liveCardRecords.get(taskKey(row.task_id)),
                taskCostProjection(row, row.ts || row.timestamp || ''));
            return Boolean((!wasVisible && record.root?.isConnected) || anchored || costChanged);
        });
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
        const initialExpanded = explicitCardExpansion.has(normalizedGroupId)
            ? explicitCardExpansion.get(normalizedGroupId)
            : Boolean(options.isSubagent && nestedSubagentsExpanded);
        root.dataset.expanded = initialExpanded ? '1' : '0';
        // Selectable summary; Reviews and Details have independent native controls.
        root.innerHTML = `
            <div class="chat-live-summary-button" data-live-summary-button>
                <div class="chat-live-summary">
                    <div class="chat-live-summary-main">
                        <span class="chat-live-phase working" data-live-phase role="status" aria-live="polite" aria-atomic="true" aria-label="${options.isSubagent ? 'Subagent' : 'Task'} status: Working">Working</span>
                        <span class="chat-live-phase-secondary" data-live-phase-secondary hidden></span>
                        <span class="chat-live-hurry" data-live-hurry role="img" hidden></span>
                    </div>
                    <span class="chat-live-title" data-live-title>Waiting for work</span>
                    <button type="button" class="chat-live-summary-side ui-hit-area" data-live-disclosure aria-expanded="false" aria-controls="${escapeHtmlAttr(timelineId)}">
                        <span class="chat-live-count" data-live-count hidden></span>
                        <span class="chat-live-toggle" data-live-toggle>Details</span>
                        <svg class="chat-live-chevron" width="14" height="14" viewBox="0 0 20 20" fill="none" aria-hidden="true">
                            <path d="M5 7.5 10 12.5 15 7.5" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"></path>
                        </svg>
                    </button>
                </div>
                <div class="chat-live-episodes" data-live-episodes hidden></div>
                <div class="chat-live-activity" data-live-activity></div>
            </div>
            <div class="chat-live-foot" data-live-foot>
                <div class="chat-live-meta" data-live-meta></div>
                <div class="chat-live-foot-end" data-live-foot-end>
                    <button type="button" class="chat-live-review-summary ui-hit-area" data-live-review-summary aria-expanded="false" hidden></button>
                </div>
            </div>
            <div data-live-reviews-host></div>
            <div class="chat-live-details">
                <div data-live-phase-detail hidden></div>
                <dl class="chat-live-facts" data-live-facts hidden></dl>
                <div data-live-cost-detail hidden></div>
            </div>
            <div class="chat-live-timeline" data-live-timeline id="${escapeHtmlAttr(timelineId)}"></div>
            <div class="chat-live-episode-history" data-live-episode-history hidden></div>
        `;
        const record = {
            groupId: normalizedGroupId,
            root,
            summaryButtonEl: root.querySelector('[data-live-summary-button]'),
            phaseEl: root.querySelector('[data-live-phase]'),
            titleEl: root.querySelector('[data-live-title]'),
            episodesEl: root.querySelector('[data-live-episodes]'),
            episodeHistoryEl: root.querySelector('[data-live-episode-history]'),
            activityEl: root.querySelector('[data-live-activity]'),
            countEl: root.querySelector('[data-live-count]'),
            disclosureEl: root.querySelector('[data-live-disclosure]'),
            footEl: root.querySelector('[data-live-foot]'),
            footEndEl: root.querySelector('[data-live-foot-end]'),
            metaEl: root.querySelector('[data-live-meta]'),
            reviewSummaryEl: root.querySelector('[data-live-review-summary]'),
            toggleEl: root.querySelector('[data-live-toggle]'),
            // Both actions render lazily from the record's facts (syncBlockChrome,
            // syncCancelRunButton).
            turnProjectBtn: null,
            cancelRunBtn: null,
            timelineEl: root.querySelector('[data-live-timeline]'),
            reviewsHostEl: root.querySelector('[data-live-reviews-host]'),
            expandedLineKeys: new Set(),
            isSubagent: Boolean(options.isSubagent),
            parentGroupId: String(options.parentGroupId || ''),
            subagentRole: String(options.role || ''),
            subagentsEl: null,
            // The proactively-coined LLM name; becomes the card title when set.
            suggestedName: '',
            reviewOwnerDetailObserved: false,
        };
        const reviewDisclosure = reviewDisclosureByTask.get(normalizedGroupId) || {
            sectionExpanded: false,
            expandedGroups: new Set(),
        };
        reviewDisclosureByTask.set(normalizedGroupId, reviewDisclosure);
        record.reviewController = createReviewPresentationController({
            host: record.reviewsHostEl,
            summary: record.reviewSummaryEl,
            disclosure: reviewDisclosure,
            onHydrate: () => hydrateCardReviews(normalizedGroupId),
            onRead: (file, trigger, event) => chatMedia.openReview(file, record.root, trigger, event),
            onDomWrite: withStableViewport,
        });
        // A native button: Enter and Space press it. A portal hides it (its room holds what it would open).
        record.disclosureEl?.addEventListener('click', () => {
            if (isPortal(record)) return;
            const nowExpanded = record.root.dataset.expanded !== '1';
            explicitCardExpansion.set(record.groupId, nowExpanded);
            setLiveCardExpanded(record, nowExpanded);
            if (nowExpanded) hydrateCardReviews(record.groupId);
        });
        record.timelineDispose = bindLiveCardTimeline(record.timelineEl, (lineKey) => {
            const nowExpanded = !record.expandedLineKeys.has(lineKey);
            if (nowExpanded) record.expandedLineKeys.add(lineKey);
            else record.expandedLineKeys.delete(lineKey);
            renderLiveCardTimeline(record);
            // Expand fetches full output once; failures retain the preview.
            if (nowExpanded) {
                const item = record.items.find((it) => it.lineKey === lineKey);
                if (item && item.truncated && item.fullRef && !item.fetchedFull && !item._fetchingFull) {
                    fetchFullLineOutput(item, record);
                }
            }
        });
        liveCardRecords.set(normalizedGroupId, record);
        liveCardBound.observe(liveCardRecords.size);
        // apply a name that arrived (task_named) before this card existed.
        const _pendingName = pendingSuggestedNames.get(normalizedGroupId);
        if (_pendingName && !record.isSubagent) {
            pendingSuggestedNames.delete(normalizedGroupId);
            record.suggestedName = _pendingName;
        }
        resetLiveCardRecord(record);
        // A card the last complete body does not name asks for a read.
        if (!_historyReplayActive && workState.complete && !workState.members.has(normalizedGroupId)) reread();
        applyCardWorkViews(new Map([[normalizedGroupId, record]]), workState);
        const activity = activeDirectActivities.get(normalizedGroupId);
        if (activity) syncParkedPhase(record, activity.phase, activity);
        const ended = concludedDirectActivities.get(normalizedGroupId);
        if (ended) finishLiveCardMutation(normalizedGroupId, ended);
        // P5: the cancelable marker may have arrived (scheduled progress frame /
        // history replay) before this card was minted.
        syncCancelRunButton(record);
        // Children that arrived before their parent's evidence nest under it now.
        for (const [childId, info] of subagentChildParents) {
            if (info.parentId === normalizedGroupId) ensureLiveCardVisible(liveCardRecords.get(childId));
        }
        return record;
    }

    // A lineage-known id is minted as the nested card its parent owns, never
    // as a root-shaped shell (#636), whichever path reaches it first.
    function getLiveCardRecord(groupId = '') {
        const normalizedGroupId = groupId || activeLiveGroupId || 'chat';
        const lineage = subagentChildParents.get(normalizedGroupId);
        return liveCardRecords.get(normalizedGroupId) || (lineage
            ? getSubagentCardRecordMutation(normalizedGroupId, lineage.parentId, lineage.role)
            : createLiveCardRecord(normalizedGroupId));
    }

    // Apply a coined name to a visible main card; buffer an early task_named frame.
    function applySuggestedName(taskId, name) {
        return withStableViewport(() => applySuggestedNameMutation(taskId, name));
    }

    function applySuggestedNameMutation(taskId, name) {
        const tid = taskKey(taskId);
        const nm = taskKey(name);
        if (!tid || !nm) return false;
        const record = liveCardRecords.get(tid);
        if (!record) {
            // task_named reaches every instance, so bound the early-name buffer.
            pendingSuggestedNames.set(tid, nm);
            if (pendingSuggestedNames.size > 100) {
                pendingSuggestedNames.delete(pendingSuggestedNames.keys().next().value);
            }
            return false;
        }
        return episodes.rename(record, nm);
    }

    // A parent card comes only from the parent's OWN evidence here (its record, a census
    // activity, a replayed row of its own), never a child's bare parent_task_id (A5).
    function ensureSubagentContainer(parentId = '', depth = 0) {
        if (!parentId || !(liveCardRecords.has(parentId) || activeDirectActivities.has(parentId)
                || evidencedTaskIds.has(parentId))) return null;
        // A lineage-known parent is minted as the nested card ITS parent owns,
        // never as a root-shaped shell (#636); depth bounds a lineage cycle.
        const lineage = depth < 32 ? subagentChildParents.get(parentId) : null;
        const parentRecord = lineage
            ? getSubagentCardRecordMutation(parentId, lineage.parentId, lineage.role, depth + 1)
            : getLiveCardRecord(parentId);
        let container = parentRecord.subagentsEl;
        if (!container) {
            container = document.createElement('div');
            container.className = 'chat-subagents';
            container.dataset.subagentsFor = parentId;
            parentRecord.subagentsEl = container;
        }
        const anchor = parentRecord.reviewsHostEl || parentRecord.timelineEl;
        if (container.parentNode !== parentRecord.root || container.previousElementSibling !== anchor) {
            anchor?.insertAdjacentElement('afterend', container);
        }
        return container;
    }

    // A child's frame reanchors its parent's card (one this room has or just minted) at the frame's time.
    function getSubagentCardRecord(childId = '', parentId = '', role = '', rawTs = '') {
        const existing = liveCardRecords.get(childId);
        const childBefore = captureLiveCardProjection(existing);
        const parentBefore = captureLiveCardProjection(liveCardRecords.get(parentId));
        let record = null;
        withStableViewport(() => {
            record = getSubagentCardRecordMutation(childId, parentId, role);
            if (rawTs) reanchorTaskCard(liveCardRecords.get(parentId), rawTs);
            return liveCardProjectionChanged(childBefore, record)
                || liveCardProjectionChanged(parentBefore, liveCardRecords.get(parentId));
        });
        return record;
    }

    function getSubagentCardRecordMutation(childId = '', parentId = '', role = '', depth = 0) {
        if (!childId || !parentId) return null;
        const existing = liveCardRecords.get(childId);
        const record = existing || createLiveCardRecord(childId, {
            isSubagent: true,
            parentGroupId: parentId,
            role,
        });
        const promoted = !record.isSubagent;
        record.isSubagent = true;
        record.parentGroupId = parentId;
        record.subagentRole = role || record.subagentRole || '';
        if (promoted) {
            record.root.classList.add('subagent');
            record.root.dataset.subagent = '1';
            // A frame that outran its lineage minted this shell root-shaped:
            // the conversion a root earned is re-derived from the child's facts.
            syncBlockChrome(record);
            refitCollapsedActivity(record);  // a child shows one line
        }
        if (record.root.dataset.parentTaskId !== parentId) record.root.dataset.parentTaskId = parentId;
        if (record.root.dataset.subagentRole !== record.subagentRole) {
            record.root.dataset.subagentRole = record.subagentRole;
        }
        if (existing && !explicitCardExpansion.has(childId)) {
            setLiveCardExpanded(record, nestedSubagentsExpanded);
        }
        const container = ensureSubagentContainer(parentId, depth);
        if (container && record.root.parentNode !== container) {
            container.appendChild(record.root);
        }
        const parentRecord = liveCardRecords.get(parentId);
        if (parentRecord) updateLiveCardCount(parentRecord);
        return record;
    }

    function resetLiveCardRecord(record) {
        record.updates = 0;
        record.finished = false;
        record.items = [];
        record.lastHumanHeadline = '';
        record.lastHumanHeadlineSource = null;
        record.expandedLineKeys.clear();
        record._anchorOrderDirty = false;
        // collapsed timelines defer DOM building; the flag says
        // the rendered timeline DOM is stale relative to record.items.
        record._timelineDirty = false;
        // Render metadata once per card in a batch.
        record._lastFrameMeta = [];
        // P1: last bounded activity projection (remembered even while
        // the collapsed line is suppressed on unnamed root cards) + sticky cost.
        clearStickyCardState(record);
        record.titleEl.textContent = record.suggestedName || '';
        setLiveCardPhase(record, 'working');
        record.countEl.hidden = true;
        record.countEl.textContent = '0 notes';
        record.metaEl.innerHTML = '';
        record.timelineEl.innerHTML = '';
        record.root.dataset.finished = '0';
        const expanded = explicitCardExpansion.has(record.groupId)
            ? explicitCardExpansion.get(record.groupId)
            : Boolean(record.isSubagent && nestedSubagentsExpanded);
        setLiveCardExpanded(record, expanded);
        syncLiveCardToggle(record);  // a card born expanded says so from its first paint
    }

    function ensureLiveCardVisible(
        record,
        { suppressDomInsert = false, reorderExisting = false } = {},
    ) {
        if (!record || suppressDomInsert || _syncPass1Active) return;
        const container = record.isSubagent ? ensureSubagentContainer(record.parentGroupId) : null;
        if (container) {
            // The container nests a lineage-known parent (#636); the recursion
            // carries the chain up to the inserted root.
            if (record.root.parentNode !== container) container.appendChild(record.root);
            const parentRecord = liveCardRecords.get(record.parentGroupId);
            ensureLiveCardVisible(parentRecord);
            updateLiveCardCount(parentRecord);
            return;
        }
        syncBlockChrome(record);
        episodes.sync(record);
        if (blockVisible(record)) insertMessageNode(record.root, { reorderExisting });
        else if (record.root.parentNode === messagesDiv) record.root.remove();
    }

    function setLiveCardExpanded(record, expanded) {
        const mutate = () => {
            if (!record?.root) return false;
            const value = expanded ? '1' : '0';
            if (record.root.dataset.expanded === value) return false;
            record.root.dataset.expanded = value;
            // First expansion materializes its deferred timeline.
            if (expanded && record._timelineDirty) renderLiveCardTimeline(record);
            syncLiveCardToggle(record);
            refitCollapsedActivity(record);
            return Boolean(record.root.isConnected);
        };
        return record?.root?.isConnected ? withStableViewport(mutate) : mutate();
    }

    function directSubagentCount(record) {
        return record?.subagentsEl?.querySelectorAll(':scope > .chat-live-card.subagent').length || 0;
    }

    function updateLiveCardCount(record) {
        if (!record?.countEl) return;
        const bits = liveCardCountBits(record.items.length, directSubagentCount(record));
        const hidden = bits.length === 0;
        const text = bits.join(' · ');
        if (record.countEl.hidden !== hidden) record.countEl.hidden = hidden;
        if (record.countEl.textContent !== text) record.countEl.textContent = text;
    }

    // Returning to the Chat page reopens the room at its newest message; a window
    // shown again keeps the same reading intent (data readiness positions it);
    // hidden, it ends an unread opening still owed.
    const handlePageShown = (event) => {
        if (event?.detail?.page === 'chat') void showLatest();
        else if (event?.type !== 'visibilitychange') return;
        else if (document.hidden) reading.land(() => 0, reading.arriving);
        else { reading.request(); historyControls.showNotice(); }
    };
    window.addEventListener('ouro:page-shown', handlePageShown);
    document.addEventListener('visibilitychange', handlePageShown);

    // Beyond the 4000-char preview: fetch/cache on expansion, bounded scroll, render while expanded.
    async function fetchFullLineOutput(item, record) {
        item._fetchingFull = true;
        item.fullReadError = '';
        let changed = false;
        try {
            const resp = await apiFetch(`/api/tasks/${encodeURIComponent(item.fullRef)}`, { cache: 'no-store' });
            const data = resp && typeof resp.json === 'function' ? await resp.json() : resp;
            // Keep result and trace: the preview can truncate either independently.
            const result = String((data && data.result) || '').trim();
            const trace = String((data && data.trace_summary) || '').trim();
            const report = result && ['done', 'warn'].includes(taskOutcomeSeverity(data))
                ? childReportLabel(data.status) : '';  // the label the capped body led with
            let full = '';
            if (result && trace) full = `${report ? `## ${report}` : '[RESULT]'}\n${result}\n\n[TRACE]\n${trace}`;
            else full = report ? `## ${report}\n${result}` : result || trace;
            if (item.fullField === 'candidate_remainder') {
                full = readCandidateRemainder(item, data, resp?.ok !== false);
                changed = true;
            }
            if (full && item.fetchedFull !== full) {
                item.fetchedFull = full;
                changed = true;
            }
        } catch {
            // best-effort: leave the capped preview on failure
            if (item.fullField === 'candidate_remainder') {
                readCandidateRemainder(item, null, false);
                changed = true;
            }
        } finally {
            item._fetchingFull = false;
            if (changed && !destroyed && record.expandedLineKeys.has(item.lineKey)) {
                renderLiveCardTimeline(record);
            }
            reading.position();
        }
    }

    function scheduleHistorySync() {
        historyResyncScheduler.schedule(liveCardBound.isArmed());
    }

    const historyResyncScheduler = createHistoryResyncScheduler({
        isReplayActive: () => _historyReplayActive,
        // Re-arm if the joined sync fetched its window before the arm.
        run: () => historyPager.read().catch(() => {}).then(() => {
            if (!destroyed && lastHistorySyncSucceeded && liveCardBound.isArmed()) scheduleHistorySync();
        }),
    });

    function applyLiveCardState(summary, groupId, ts, dedupeKey = '', options = {}) {
        return withStableViewport(() => applyLiveCardStateMutation(
            summary, groupId, ts, dedupeKey, options,
        ));
    }

    // Authored naming has its own clock: newer execution or a terminal outcome
    // may already own the card when its saved narration reaches this instance.
    // A root's collapsed line keeps it while showing narration (a cause holds).
    function reconcileAuthoredHeadline(record, summary, rawTs) {
        if (!summary.human || !summary.headline) return;
        const previous = record.lastHumanHeadlineSource, shown = previous?.preview || '';
        const incoming = { time: Date.parse(rawTs), position: _historyRow?.history_position,
            preview: boundActivityPreview(summary.activityPreview ?? summary.headline) || shown };
        if (Number.isFinite(previous?.time) && (!Number.isFinite(incoming.time) || incoming.time < previous.time)) return;
        if (incoming.time === previous?.time && (compareHistoryPosition(incoming.position, previous.position) < 0
                || previous.position && !incoming.position)) return;
        if (!record.isSubagent && !record.causeHeld && record.collapsedActivity === shown) record.collapsedActivity = incoming.preview;
        record.lastHumanHeadline = summary.headline;
        record.lastHumanHeadlineSource = incoming;
        return true;
    }

    function applyLiveCardStateMutation(summary, groupId, ts, dedupeKey = '', { suppressDomInsert = false, rawTs = '' } = {}) {
        const nextGroupId = groupId || activeLiveGroupId || 'active';
        const record = getLiveCardRecord(nextGroupId);
        const before = captureLiveCardProjection(record);
        const typingBefore = typingEl.style.display;
        let timelineChanged = false;
        const nextPhase = summary.phase || '';
        const closedPhase = record.finished && !isTerminalTaskPhase(nextPhase, summary.terminal) ? replayTerminalPhase(record) : '';
        const initialContent = record.finished && (!record.updates || (summary.human && !record.lastHumanHeadline));
        // A closed card drops a late live row (below); saved narration joins its timeline and may name it.
        const newest = (!closedPhase || initialContent || _historyRow?.history_id) && reconcileAuthoredHeadline(record, summary, rawTs);
        if (_historyRow?.history_id && record.updates > 0 && !initialContent && !summary.terminal
                && (record.finished || _historyAppending
                    || Date.parse(rawTs) < Date.parse(record.latestSourceTs || ''))) {
            const changed = mergeHistoricalTimelineItem(record, summary, _historyRow, ts);
            reanchorTaskCard(record, rawTs, { suppressDomInsert });
            if (changed) { renderLiveCardTimeline(record); updateLiveCardCount(record); }
            if (newest) syncCardHeadline(record);
            return changed || liveCardProjectionChanged(before, record);
        }
        // One tool evidence row per block.
        const foldView = summary.toolCall ? applyToolObservation(record, summary.toolCall) : null;
        if (summary.toolCall?.fact === 'wait_ended' && record.toolFold.calls.get(summary.toolCall.key)?.settlement) {
            summary = { ...summary, visible: false }; // wait history remains in the fold
        }
        if (closedPhase && !initialContent) {
            if (foldView) {
                upsertToolFoldRow(record, foldView, ts, rawTs);
                if (summary.dedupeKey?.endsWith(':check')) placeCheckRow(record, summary, ts, rawTs);
                renderLiveCardTimeline(record);
                updateLiveCardCount(record);
            }
            if (summary.modelExecution) record.modelExecution = summary.modelExecution;
            renderLiveCardMeta(record, summary.costProjection);
            return liveCardProjectionChanged(before, record);
        }
        if (!closedPhase && (summary.terminal || (!_historyReplayActive && !_syncPass1Active
                && ['working', 'thinking'].includes(summary.phase)))) {
            restoreCardActivity(record);
            if (!summary.terminal) {
                if (record.isSubagent && record.parkedPhase === 'unknown' && ws.isConnected?.() !== false) syncParkedPhase(record, summary.phase);
                record.lastLiveObservedAt = Date.now();
                missingManagedTaskIds.delete(nextGroupId);
            }
        }
        markReviewAnchor(record);
        if (foldView?.clearedNotice) { renderLiveCardTimeline(record); timelineChanged = true; }

        if (!record.isSubagent) {
            if (!closedPhase) activeLiveGroupId = nextGroupId;
            reanchorTaskCard(record, rawTs, { suppressDomInsert });
        }
        record.updates += 1;
        const wasFinished = record.finished;
        if (summary.terminal && typeof summary.quietEffective === 'boolean') record.quietEffective = summary.quietEffective;
        // Empty frames name nothing (#1369).
        const headline = summary.headline || record.lastHumanHeadline || '';
        const syntheticKey = summary.dedupeKey || dedupeKey || `${summary.phase || 'working'}|${headline}|${summary.body || ''}`;
        const isLegacyParentSubagentKey = syntheticKey.startsWith('parent-subagent:');
        // Failure/timeout update one row; success feeds the fold.
        const inPlaceByKey = isLegacyParentSubagentKey
            || ['subagent-lifecycle:', 'subagent-progress:', 'subagent-result:', 'task_done|', 'tool:']
                .some((prefix) => syntheticKey.startsWith(prefix));
        if (!isLegacyParentSubagentKey) {
            record.finished = Boolean(closedPhase) || isTerminalTaskPhase(nextPhase, summary.terminal);
        }
        if (summary.model) record.agentModel = summary.model;
        // Origin (including consciousness) stays sticky once observed.
        if (summary.initiator) record.initiator = summary.initiator;

        // Only task facts own the outcome chip under a hold; failed tools are diagnostics.
        if (summary.observedOutcome && !record.finished) record.observedOutcome = summary.observedOutcome;
        const desiredPhase = desiredLiveCardPhase(record, closedPhase || (record.finished ? summary.phase || 'done' : ''));
        setLiveCardPhase(record, desiredPhase.phase, desiredPhase.text, desiredPhase.className,
            desiredPhase.secondary);
        // Collapsed activity comes only from the turn's voice, including child cards;
        // host notes and terminal Done cannot overwrite it. Expand keeps full activity.
        const activityCandidate = (record.isSubagent && summary.human !== false
            ? String(summary.activityPreview ?? summary.body ?? '')
            : String(!summary.human && summary.terminal && summary.activityPreview || '')).trim();
        if (activityCandidate) {
            record.collapsedActivity = boundActivityPreview(activityCandidate);
            record.collapsedActivityLead = summary.activityLead || '';  // a child's "Report to parent"
            record.causeHeld = !record.isSubagent;  // a root's line is its outcome's cause
        }
        syncCardHeadline(record);

        const shouldRenderLine = summary.visible !== false && Boolean(headline || summary.body);
        let timelineUpdate = 'none';
        let patchIndex = -1;
        if (_historyRow?.history_id) {
            if (mergeHistoricalTimelineItem(record, summary, _historyRow, ts)) timelineUpdate = 'render';
        } else if (shouldRenderLine && !syntheticKey.startsWith('tools|')) {
            ({ timelineUpdate, patchIndex } = updateLiveTimelineItem(record, summary,
                { ts, rawTs, syntheticKey, headline, inPlaceByKey }));
        }
        // Failures keep their row and feed the fold.
        if (foldView && !_historyRow?.history_id) {
            const fold = upsertToolFoldRow(record, foldView, ts, rawTs);
            if (timelineUpdate === 'none') ({ timelineUpdate, patchIndex } = fold);
            else if (!['none', 'duplicate-skip'].includes(fold.timelineUpdate)) timelineUpdate = 'render';
        }
        updateLiveCardCount(record);
        // Cost does not move the activity clock.
        if (ts && (summary.human || activityCandidate)) record.latestActivityTs = ts, record.latestActivityRaw = rawTs || '';
        if (summary.executorChip
                && !keepStickyExecutorChip(record.executorChip, summary.executorChip)) {
            record.executorChip = summary.executorChip;
        }
        if (summary.modelExecution) record.modelExecution = summary.modelExecution;
        if (summary.effort) record.effort = summary.effort;
        if (Number.isInteger(summary.toolCalls)) record.toolCalls = Math.max(record.toolCalls ?? 0, summary.toolCalls);
        record._lastFrameMeta = Array.isArray(summary.meta) ? summary.meta : [];
        if (rawTs) record.latestSourceTs = rawTs;
        renderLiveCardMeta(record, summary.costProjection);
        const lastItem = record.items[record.items.length - 1];
        if (timelineUpdate === 'render') {
            timelineChanged = renderLiveCardTimeline(record);
        } else if (timelineUpdate === 'append' && lastItem) {
            timelineChanged = appendTimelineItem(lastItem, record);
        } else if (timelineUpdate === 'patch-last' && lastItem) {
            timelineChanged = patchLastTimelineItem(lastItem, record);
        } else if (timelineUpdate === 'patch-at' && patchIndex !== -1) {
            timelineChanged = patchTimelineItemAt(record.items[patchIndex], record);
        }
        ensureLiveCardVisible(record, { suppressDomInsert });
        hideTypingIndicatorOnly();
        // Log task_done bypasses finishLiveCard; settle Cancel and its marker.
        if (record.finished) settleLiveCard(record, wasFinished);
        syncChatStatus();
        return Boolean(timelineChanged
            || typingBefore !== typingEl.style.display
            || liveCardProjectionChanged(before, record));
    }

    function finishLiveCard(groupId = '', phase = '') {
        return withStableViewport(() => finishLiveCardMutation(groupId, phase));
    }

    // Author controls end; paid-review waits retain their own lifetime.
    function settleLiveCard(record, wasFinished) {
        record.root.dataset.finished = '1';
        if (record.toolFold) {
            upsertToolFoldRow(record, noteToolHostMetrics(record, {}), '', '');
            renderLiveCardTimeline(record);
        }
        cancelableTaskIds.delete(record.groupId);
        missingManagedTaskIds.delete(record.groupId);
        syncCancelRunButton(record);
        modelWaits.finish(record.groupId);
        // A lost task_done is healed only by refetching; a block nobody sees
        // owes no refetch.
        if (!wasFinished && blockVisible(record)) scheduleHistorySync();
        syncLiveCardToggle(record);
    }

    function finishLiveCardMutation(groupId = '', phase = '') {
        const record = groupId ? liveCardRecords.get(groupId) : null;
        if (!record) return false;
        const before = captureLiveCardProjection(record);
        const typingBefore = typingEl.style.display;
        if (setHistoricalUnavailable(record, false)) renderLiveCardMeta(record);
        markReviewAnchor(record);
        const wasFinished = record.finished;
        record.finished = true;
        record.finalizingHold = false;
        const presentation = taskPresentation(phase || 'done', record.quietEffective ?? record.historicalTerminal?.quiet_effective);
        setLiveCardPhase(record, presentation.phase, presentation.headline);
        syncCardHeadline(record);
        settleLiveCard(record, wasFinished);
        ensureLiveCardVisible(record);
        if (activeLiveGroupId === record.groupId) activeLiveGroupId = '';
        syncChatStatus();
        return Boolean(typingBefore !== typingEl.style.display
            || liveCardProjectionChanged(before, record));
    }

    function appendTaskSummaryToLiveCard(msg, { suppressDomInsert = false } = {}) {
        const taskId = msg?.task_id || '';
        const rawTs = msg?.ts || new Date().toISOString();
        if (!taskId) {
            // Unkeyed summaries keep durable/log evidence but close no card.
            return false;
        }
        let changed = false;
        // Restore historical task name.
        if (msg?.suggested_name) changed = applySuggestedName(taskId, msg.suggested_name) || changed;
        const finalizing = msg?.task_phase === 'finalizing' || msg?.outcome_final === false;
        changed = noteToolMetrics(taskId, msg, rawTs, { suppressDomInsert }) || changed;
        const record = getLiveCardRecord(taskId);
        msg = { quiet_effective: record.quietEffective ?? record.historicalTerminal?.quiet_effective, ...msg };
        const summary = taskTerminalSummary({ ...msg, task_id: taskId });
        changed = noteTaskDetails(record, msg) || changed;
        syncSettledItems(record, msg, { write: withStableViewport });
        changed = Boolean(record.reviewController?.updateMany(reviewGroupsFromTaskDetail(msg, taskId))) || changed;
        if (finalizing && !record.finished) record.finalizingHold = true;
        changed = applyLiveCardState(
            { ...summary, costProjection: taskCostProjection(msg, rawTs) },
            taskId, normalizeLogTs(rawTs), summary.dedupeKey, { suppressDomInsert, rawTs },
        ) || changed;
        if (finalizing) return changed;
        changed = finishLiveCard(taskId, summary.phase) || changed;
        return changed;
    }

    // Child lineage retains model-less updates; child finality is independent.
    const subagentChildParents = new Map();
    // Tasks whose own rows this room replayed: a parent card's evidence (A5).
    const evidencedTaskIds = new Set();
    // Late progress must not revive a terminal child.
    const subagentTerminalChildren = new Set();

    function renderLiveCardMeta(record, costProjection = null) {
        if (record && costProjection) record.costMeta = mergeStickyCostMeta(record.costMeta, costProjection);
        return renderCardMeta(record, { agentModel: record?.isSubagent
            ? subagentChildParents.get(record.groupId)?.model : record?.agentModel });
    }

    function setSubagentParent(childId, { parentId = '', role = '', model = '' } = {}) {
        const prev = subagentChildParents.get(childId) || {};
        const next = {
            parentId: parentId || prev.parentId || '',
            role: role || prev.role || '',
            model: taskKey(model) || prev.model || '',
        };
        if (['parentId', 'role', 'model'].every((k) => next[k] === prev[k])) return;
        subagentChildParents.set(childId, next);
        // Lineage reclassifies a root-shaped shell the moment it is learned,
        // whatever the frame that carries it goes on to render.
        if (next.parentId && liveCardRecords.get(childId)?.isSubagent === false) {
            ensureLiveCardVisible(getSubagentCardRecord(childId, next.parentId, next.role));
        }
        for (const sid of subagentChildParents.keys()) {
            const rec = liveCardRecords.get(sid);
            if (rec?.isSubagent) { syncCardHeadline(rec); renderLiveCardMeta(rec); }
        }
    }

    function learnSubagentLineage(msg) {
        if (String(msg?.delegation_role || '').toLowerCase() !== 'subagent') return '';
        const parentId = taskKey(msg.parent_task_id);
        const childId = String(msg.subagent_task_id || msg.task_id || '').trim();
        if (!parentId || !childId || parentId === childId) return '';
        setSubagentParent(childId, {
            parentId, role: taskKey(msg.subagent_role), model: msg.model,
        });
        const event = String(msg.subagent_event || '').toLowerCase();
        const replayTerminal = msg.task_terminal_status && taskDoneIsTerminal(msg);
        if (replayTerminal || ['completed', 'completed_warn', 'failed', 'cancelled', 'rejected'].includes(event)) {
            subagentTerminalChildren.add(childId);
        }
        return childId;
    }

    function summarizeSubagentCardFrame(evt, childId, overrides = {}, rawTs = '') {
        const { parentId = '', role = '', model = '' } = subagentChildParents.get(childId) || {};
        const summary = summarizeChatLiveEvent({
            ...evt,
            type: 'send_message',
            is_progress: true,
            delegation_role: 'subagent',
            subagent_task_id: childId,
            parent_task_id: parentId,
            subagent_role: role,
            model,
            ...overrides,
        });
        // The child's effort fact rides every frame that states a level (live, terminal, replay).
        return summary ? withTaskCostMeta({ ...summary, ...(evt.effort_level ? { effort: effortFact(evt) } : {}) }, evt, { rawTs }) : null;
    }

    // A child's title is its lineage identity plus, for twins (same displayed identity
    // under one parent), the short id; re-projected on every title write and lineage
    // change (terminal children included).
    function childTitle(record) {
        const twin = subagentTwin(subagentChildParents, record.groupId);
        return subagentIdentityTitle(subagentChildParents.get(record.groupId))
            + (twin ? ` (${record.groupId.slice(0, 8)})` : '');
    }

    function updateLiveCardFromProgressMessage(msg, { grantCancelAuthority = true } = {}) {
        if (msg?.system_type === 'task_checkpoint') return updateLiveCardFromLogEvent({ ...msg, type: 'task_checkpoint', is_progress: false });
        const taskId = msg?.task_id || '';
        const rawTs = msg?.ts || new Date().toISOString();
        if (!taskId) return false;
        modelWaits.observe(taskId, msg);
        let changed = msg.tool_evidence || Number.isInteger(msg.rounds) ? noteToolMetrics(taskId, msg, rawTs) : false;
        // Only host-attested progress grants Stop authority.
        if (grantCancelAuthority && msg.cancelable === true) {
            changed = markTaskCancelable(String(taskId)) || changed;
        }
        const lifecycleParent = taskKey(msg.parent_task_id);
        if (msg.subagent_event && lifecycleParent) {
            const updated = updateSubagentCardFromEvent(msg, rawTs);
            if (updated !== undefined) return Boolean(changed || updated);
        }
        if (subagentChildParents.has(taskId)) {
            return Boolean(routeSubagentProgressToCard(taskId, msg) || changed);
        }
        const summary = summarizeChatLiveEvent({
            type: 'send_message',
            is_progress: true,
            content: msg.content || msg.text || '',
            task_id: taskId,
            ...Object.fromEntries(['subagent_event', 'subagent_task_id', 'root_task_id',
                'parent_task_id', 'delegation_role', 'subagent_role', 'status', 'result',
                'trace_summary', 'error', 'artifact_status'].map((key) => [key, msg[key] || ''])),
            ...cardMetaKeys(msg),
            lifecycle: msg.lifecycle || null,
            // Preserve source authorship and kind, live and on replay.
            role: msg.role, system_type: msg.system_type, narration: msg.narration,
        });
        if (!summary) return changed;
        const presented = withTaskCostMeta(summary, msg, { rawTs });
        changed = queueTaskLiveUpdate(
            presented, taskId, normalizeLogTs(rawTs), presented.dedupeKey || '', rawTs,
        ) || changed;
        // History carries task_named's title.
        if (msg.suggested_name) changed = applySuggestedName(taskId, msg.suggested_name) || changed;
        // Outcome survives a lost summary even while post-work keeps controls live.
        if (
            (msg.task_phase === 'finalizing' || taskDoneIsTerminal(msg))
            && (msg.outcome_axes || msg.review_projection || msg.reason_code)
        ) {
            changed = appendTaskSummaryToLiveCard(msg) || changed;
        }
        return changed;
    }

    function updateSubagentCardFromEvent(evt, tsValue) {
        // undefined = not a subagent frame: callers fall through.
        if (!evt || String(evt.delegation_role || '').toLowerCase() !== 'subagent') return undefined;
        const parentId = taskKey(evt.parent_task_id);
        const childId = String(evt.subagent_task_id || evt.task_id || '').trim();
        if (!parentId || !childId || parentId === childId) return undefined;
        const event = String(evt.subagent_event || '').toLowerCase();
        const role = taskKey(evt.subagent_role);
        setSubagentParent(childId, { parentId, role, model: evt.model });
        // Worker narration carries subagent_event="progress" too. It is activity,
        // not a lifecycle row: route it through the progress key so the later
        // terminal frame cannot overwrite the only full narration disclosure.
        if (!SUBAGENT_LIFECYCLE_EVENTS.has(event)) {
            return routeSubagentProgressToCard(childId, evt);
        }
        const rawTs = tsValue || new Date().toISOString();
        const summary = summarizeSubagentCardFrame(evt, childId, {}, rawTs);
        if (!summary) return false;
        summary.dedupeKey = `subagent-lifecycle:${childId}`;
        // Interrupted is retryable and therefore non-terminal; the canonical
        // projector owns that distinction for both live and replay paths.
        if (summary.terminal) subagentTerminalChildren.add(childId);
        stampNodeTimestamp(getSubagentCardRecord(childId, parentId, role, rawTs)?.root, rawTs, { anchor: true });
        const reviewsChanged = attachTaskDetailReviews(childId, evt);
        const updated = queueTaskLiveUpdate(
            summary,
            childId,
            normalizeLogTs(rawTs),
            summary.dedupeKey,
            rawTs,
        );
        return Boolean(reviewsChanged || updated);
    }

    // A known child's own (non-lifecycle) progress updates the linked child card.
    function routeSubagentProgressToCard(childId, msg, ownLine = false) {  // ownLine: the child's own words
        const info = subagentChildParents.get(childId);
        if (!info) return false;
        const { parentId, role } = info;
        const content = String(msg?.content || msg?.text || '').trim();
        if (!content) return false;
        const rawTs = msg?.ts || new Date().toISOString();
        const record = getSubagentCardRecord(childId, parentId, role, rawTs);
        const preserveTerminal = Boolean(record?.finished && (ownLine || subagentTerminalChildren.has(childId)));
        const summary = summarizeSubagentCardFrame(msg, childId, {
            content,
            text: content,
            subagent_event: 'running',
            // A replayed progress row may follow a terminal record because the
            // history pre-pass already knows the child's final state. Do not add
            // contradictory `status=running` metadata in that case.
            status: preserveTerminal ? '' : (msg?.status || ''),
        }, rawTs);
        if (!summary) return false;
        // Same frame adopts one speech occurrence; progress replaces a line.
        if (!summary.activity) summary.dedupeKey = ownLine ? `subagent-line:${childId}:${rawTs}:${content}` : `subagent-progress:${childId}`;
        if (preserveTerminal && !_historyReplayActive) {
            summary.phase = String(record.phaseEl?.dataset?.phase || 'done');
            summary.terminal = true;
        }
        return queueTaskLiveUpdate(summary, childId, normalizeLogTs(rawTs), summary.dedupeKey, rawTs);
    }

    function routeSubagentFinalMessageToCard(taskId, msg) {
        const childId = taskKey(taskId);
        const info = subagentChildParents.get(childId);
        if (!childId || !info) return false;
        const { parentId, role } = info;
        const text = String(msg?.content || msg?.text || '').trim();
        const rawTs = msg?.ts || new Date().toISOString();
        const record = getSubagentCardRecord(childId, parentId, role, rawTs);
        const priorTerminalPhase = record?.finished ? String(record.phaseEl?.dataset?.phase || '') : '';
        const summary = summarizeSubagentCardFrame(msg, childId, {
            content: '',
            text: '',
            result: text,
            subagent_event: 'completed',
        }, rawTs);
        if (!summary) return false;
        summary.dedupeKey = `subagent-result:${childId}`;
        if (priorTerminalPhase) {
            summary.phase = priorTerminalPhase;
            summary.terminal = true;
        }
        return queueTaskLiveUpdate(
            summary, childId, normalizeLogTs(rawTs), summary.dedupeKey, rawTs,
        );
    }

    // Resolve a child's card from the child's terminal task_done
    // (which arrives on the log channel without subagent metadata).
    function routeSubagentTerminalToCard(childId, evt) {
        const info = subagentChildParents.get(childId);
        if (!info) return false;
        const status = String(evt.status || '').toLowerCase();
        const severity = taskOutcomeSeverity(evt);
        const interrupted = status === 'interrupted';
        const failed = severity === 'error' || status === 'failed';
        const cancelled = status === 'cancelled' || status === 'cancel_requested';
        const rejected = status === 'rejected_duplicate';
        const event = interrupted ? 'interrupted'
            : failed ? 'failed'
                : cancelled ? 'cancelled'
                    : rejected ? 'rejected'
                        : (severity === 'warn' ? 'completed_warn' : 'completed');
        return Boolean(updateSubagentCardFromEvent({
            delegation_role: 'subagent',
            parent_task_id: info.parentId,
            subagent_task_id: childId,
            subagent_role: info.role,
            subagent_event: event,
            review_projection: evt.review_projection,
            model_execution: evt.model_execution,
            result: evt.result || '',
            error: evt.error || '',
            reason_code: evt.reason_code || '',
            ...cardMetaKeys(evt),
        }, evt.ts || evt.timestamp || new Date().toISOString()));
    }

    function updateLiveCardFromLogEvent(evt) {
        if (!evt) return false;
        const eventType = evt.type || evt.event || '';
        if (eventType === 'history_retention') {
            return withStableViewport(() => noteTaskDetails(liveCardRecords.get(taskKey(evt.task_id)), evt));
        }
        const reference = admitCardMetadata(evt);
        if (reference !== undefined) return reference;
        // Typed and never progress (owner_hurry.py): it only marks an existing card.
        if (eventType === 'owner_hurry') {
            return markOwnerHurry(ownerHurryProjection(evt).applied ? liveCardRecords.get(getLogTaskGroupId(evt)) : null);
        }
        if (!isGroupedTaskEvent(evt)) return false;
        const taskId = getLogTaskGroupId(evt) || '';
        if (!taskId) return false;
        const rawTs = evt.ts || evt.timestamp || new Date().toISOString();
        // Skill lifecycle belongs to its explicit owner.
        const review = attachReviewFromRow(evt, rawTs);
        if (review !== undefined) return review;
        const childInfo = subagentChildParents.get(taskId);
        if (childInfo && eventType === 'task_done') return routeSubagentTerminalToCard(taskId, evt);
        const summary = summarizeChatLiveEvent(evt);
        if (childInfo && subagentTerminalChildren.has(taskId) && !summary?.toolCall) return false;
        // Root/child fold: metrics, terminal and replay.
        let changed = evt.tool_evidence || Number.isInteger(evt.rounds) || ['task_metrics_event', 'task_eval', 'task_done'].includes(eventType)
            ? noteToolMetrics(taskId, evt, rawTs) : false;
        if (!childInfo) changed = attachTaskDetailReviews(taskId, evt) || changed;
        if (!summary) return changed;
        if (childInfo) {
            getSubagentCardRecord(taskId, childInfo.parentId, childInfo.role);
            changed = attachTaskDetailReviews(taskId, evt) || changed;
        }
        const presented = withTaskCostMeta(summary, evt, {
            replace: eventType === 'task_done' || eventType === 'task_cost_finalized',
            rawTs,
        });
        const queued = queueTaskLiveUpdate(
            presented, taskId, normalizeLogTs(rawTs), presented.dedupeKey || '', rawTs,
        );
        if (childInfo) return Boolean(changed || queued);
        const subagentChanged = updateSubagentCardFromEvent(evt, rawTs);
        // Host-attested Stop facts also travel on tool-only turns.
        if (evt.cancelable === true) markTaskCancelable(taskId);
        if (eventType === 'task_done' && summary.terminal) {
            recordTerminalActivity(taskId, summary.phase);
            syncChatStatus();
        }
        return Boolean(changed || queued || subagentChanged);
    }

    function addMessage(text, role, markdown = false, timestamp = null, isProgress = false, opts = {}) {
        const pending = !!opts.pending;
        const ephemeral = !!opts.ephemeral;
        // Normalize the persisted identity once; the same fields label and deduplicate a row.
        const meta = Object.fromEntries([
            'clientMessageId', 'senderLabel', 'senderSessionId', 'source', 'initiator',
            'systemType', 'taskId', 'projectId', 'projectName', 'taskName', 'handoffId',
            'originId', 'completionLimitation', 'deliveryId',
        ].map(key => [key, opts[key] || '']));
        const { clientMessageId, source, systemType, taskId, projectId, projectName } = meta;
        const attachments = role === 'user' ? attachmentViews(opts.attachments) : [];
        const ts = timestamp || new Date().toISOString();
        const legacyKey = buildMessageKey(role, text, ts, { ...meta, isProgress });
        const bubbles = () => [...messagesDiv.querySelectorAll('.chat-bubble')];
        const messageKey = opts.historyId ? `history:${opts.historyId}` : legacyKey;
        const deliveryKey = opts.deliveryId ? `delivery:${opts.deliveryId}` : '';
        // A delivery is one bubble while that bubble is mounted; a released page mounts it again.
        const delivered = deliveryKey && seenMessageKeys.has(deliveryKey)
            && bubbles().find(node => node.dataset.deliveryId === opts.deliveryId);
        if (delivered) {
            stampHistoryNode(delivered, opts.historyId, opts.historyPosition);
            noteBubbleKeys(delivered);
            rememberMessageKey(messageKey);
            return false;
        }
        const prior = opts.historyId
            ? (replayKeys && !replayKeys.history.has(opts.historyId) ? null
                : historyNodes(opts.historyId).find(node => node.classList.contains('chat-bubble')))
            : (replayKeys && !replayKeys.legacy.has(legacyKey) ? null
                : Array.from(messagesDiv.querySelectorAll('.chat-bubble')).find(node => node.dataset.messageKey === legacyKey));
        if (prior && (opts.historyId || _historyReplayActive)) {
            rememberMessageKey(messageKey);
            const originChanged = syncSavedProjectContext(prior, opts.originProjected, opts.originId);
            noteBubbleKeys(prior);
            return chatDecision.renderRoutingDecision(prior, opts.chatAnnotation) || originChanged;
        }
        // History may land first; a live copy adds no new facts.
        if (prior || (messageKey && seenMessageKeys.has(messageKey))) return false;
        if (opts.historyId) {
            const mounted = !replayKeys || (clientMessageId && replayKeys.client.has(clientMessageId))
                || (opts.originId && replayKeys.origin.has(opts.originId)) || replayKeys.legacy.has(legacyKey);
            const prior = mounted && bubbles().find(node =>
                (clientMessageId && node.dataset.clientMessageId === clientMessageId)
                || (opts.originId && node.dataset.originId === opts.originId)
                || (!node.dataset.historyId && node.dataset.messageKey === legacyKey));
            if (prior) {
                stampHistoryNode(prior, opts.historyId, opts.historyPosition);
                noteBubbleKeys(prior);
                rememberMessageKey(messageKey);
                chatDecision.renderRoutingDecision(prior, opts.chatAnnotation);
                const changed = syncSavedProjectContext(prior, opts.originProjected);
                for (const row of persistedHistory) if (opts.originId && row.originId === opts.originId
                    || clientMessageId && row.clientMessageId === clientMessageId) row.originProjected = false;
                return changed;
            }
        }

        if (!isProgress && !ephemeral && !_historyAppending) {
            persistedHistory.push({
                text, role, ts, markdown: !!markdown, ...meta, narration: opts.narration,
                originProjected: !!opts.originProjected, terminalTime: opts.terminalTime || null,
                completionNotice: opts.completionNotice || null,
                skillReview: opts.skillReview || null, evidenceRef: opts.evidenceRef || null,
                presenceContact: opts.presenceContact || null,
                attachments: attachments.length ? attachments : null,
                textPlaceholder: opts.textPlaceholder === true,
            });
            // Match the persisted-history cap.
            if (persistedHistory.length > 200) persistedHistory.splice(0, persistedHistory.length - 200);
            if (!_historyReplayActive) persistVisibleHistory();
        }

        const bubble = document.createElement('div');
        bubble.className = `chat-bubble ${role}` + (isProgress ? ' progress' : '');
        if (pending) bubble.classList.add('pending');
        if (ephemeral) bubble.dataset.ephemeral = '1';
        for (const key of ['clientMessageId', 'systemType', 'senderSessionId', 'taskId', 'deliveryId', 'projectId']) {
            if (meta[key]) bubble.dataset[key] = meta[key];
        }
        if (legacyKey) bubble.dataset.messageKey = legacyKey;
        stampHistoryNode(bubble, opts.historyId, opts.historyPosition);

        const stored = text;
        if (role === 'system' && ['task_pause_notice', 'legacy_memory_notice'].includes(systemType)) text = tx(text);
        // Render-only: Copy below keeps `shown`, the pre-cleanup sentence.
        const painted = hostRowDisplayText({ text: stored, role, system_type: systemType, narration: opts.narration }, text);
        const skillReview = role === 'system' && systemType === 'skill_review' && opts.narration !== true;
        if (skillReview) bubble.classList.add('chat-skill-review');
        const richMarkdown = role !== 'user' && !skillReview && (role !== 'system' || markdown === true);
        // Only the web composer writes the file-name tail; a Telegram/skill caption is kept word for word.
        const shown = attachments.length ? attachmentCaption(text, attachments,
            { placeholder: opts.textPlaceholder === true, composed: source === 'web' }) : text;
        const rendered = role === 'user'
            ? escapeHtml(shown)
            : skillReview
                ? renderSkillReviewDisclosure(text, opts.skillReview || null)
                : role === 'system' && markdown !== true
                    ? escapeHtml(painted)
                    : renderChatMarkdown(painted);
        const timeFmt = formatMsgTime(ts);
        const timeHtml = timeFmt ? `<div class="msg-time" title="${escapeHtmlAttr(timeFmt.full)}">${escapeHtml(timeFmt.short)}</div>` : '';
        const pendingHtml = pending ? `<div class="msg-pending">Queued until reconnect</div>` : '';
        // Preserve the card's source link in a standalone row.
        bubble.innerHTML = `
            <div class="sender">${escapeHtml(senderLabel(role, isProgress, systemType, meta, chatSessionId))}</div>
            <div class="message${richMarkdown ? ' ui-rich-content' : ''}">${rendered}</div>
            ${evidenceLinkHtml(opts.evidenceRef)}
            ${pendingHtml}
            ${timeHtml}
        `;
        decorateContactSource(bubble, opts.presenceContact);
        if (attachments.length) chatMedia.mountAttachments(bubble, attachments, shown);
        if (!isProgress && shown) chatMedia.attachCopyControl(bubble, String(shown));
        if (!isProgress) { replies.decorate(bubble, opts.reply); markSwarmLabel(messagesDiv, opts.reply, bubble); }
        // Legacy start/transfer rows without task or Project stay plain text with Copy; an answer tied to
        // another room's work ends with that Project reference, unfolded.
        const projectRow = PROJECT_ROW_TYPES.has(systemType);
        if (projectRow ? !(isMain && ['project_handoff', 'project_started'].includes(systemType))
            : role === 'assistant' && projectId !== ownProjectId) decorateProjectRow(bubble, { ...meta, ...opts, role,
            fold: projectRow && role === 'assistant', addedAt: ts });
        syncSavedProjectContext(bubble, opts.originProjected, opts.originId);
        wireSkillReviewDisclosure(bubble, { onDomWrite: withStableViewport });
        stampNodeTimestamp(bubble, ts);
        insertMessageNode(bubble, { forceStick: !!opts.forceStick });
        if (richMarkdown) enhanceMountedMarkdown(bubble);
        chatDecision.renderRoutingDecision(bubble, opts.chatAnnotation);
        if (isMain) reconcilePortalNotes(messagesDiv, bubble);
        rememberMessageKey(messageKey);
        if (deliveryKey) rememberMessageKey(deliveryKey);
        if (pending && clientMessageId) pendingUserBubbles.set(clientMessageId, bubble);
        return bubble;
    }

    function updateMessageAnnotation(clientMessageId, annotation) {
        const messageId = String(clientMessageId || '');
        if (!messageId) return false;
        // The journal copy carries the ack, so a re-render restores it too.
        const journalEntry = localEchoJournal.get(messageId);
        if (journalEntry) journalEntry.annotation = annotation || null;
        const bubble = Array.from(messagesDiv.querySelectorAll('.chat-bubble.user[data-client-message-id]'))
            .find((candidate) => candidate.dataset.clientMessageId === messageId);
        const changed = chatDecision.renderRoutingDecision(bubble, annotation);
        if (bubble && isMain) reconcilePortalNotes(messagesDiv, bubble);
        return changed;
    }

    function markPendingDelivered(clientMessageId, dropped = false) {
        const bubble = pendingUserBubbles.get(clientMessageId || '');
        if (!bubble) return false;
        return withStableViewport(() => {
            const note = bubble.querySelector('.msg-pending');
            if (dropped) {
                if (note) note.textContent = 'Not delivered — send again';
            } else {
                bubble.classList.remove('pending');
                note?.remove();
            }
            pendingUserBubbles.delete(clientMessageId);
            return true;
        });
    }

    const markPendingDropped = (clientMessageId) => markPendingDelivered(clientMessageId, true);

    // Host-owned empty state: never a bubble, a history row or a model reply.
    const emptyWelcome = isMain ? mountEmptyChatWelcome(messagesDiv) : null;

    // Hydration triggers share one sticky request; reconnect/resync read again.
    const awaitInitialHydration = () => (initialHydrationPromise ||= syncHistory());

    // Main briefly yields hydration to an opening Project, with a hard bound.
    const MAIN_HYDRATION_MAX_DEFER_MS = 3500;
    function waitForHydrationWindow() {
        if (!isMain) return Promise.resolve();
        if (hydrationGatePromise) return hydrationGatePromise;
        hydrationGatePromise = new Promise((resolve) => {
            const deadline = Date.now() + MAIN_HYDRATION_MAX_DEFER_MS;
            const scheduleIdle = (callback) => (typeof requestIdleCallback === 'function'
                ? requestIdleCallback(callback, { timeout: 1000 })
                : setTimeout(callback, 50));
            const attempt = () => {
                if (destroyed) {
                    resolve();
                    return;
                }
                if (Date.now() < deadline
                    && typeof isProjectOpening === 'function'
                    && isProjectOpening()) {
                    setTimeout(attempt, 200);
                    return;
                }
                resolve();
            };
            scheduleIdle(attempt);
        });
        return hydrationGatePromise;
    }

    const historicalTerminals = new Map();
    const pendingHistoryUpserts = new Map();
    const pendingLiveEvictions = new Set();
    function releaseLiveOverflow() {
        // Every card is judged as the reader sees the page, before the first one leaves.
        const due = [...pendingLiveEvictions].filter((id) => {
            const record = liveCardRecords.get(id);
            if (!record) { pendingLiveEvictions.delete(id); return false; }
            return !(record.historyIds?.size || activeDirectActivities.has(id)
                    // A retained successor still owns this physical episode's content/controller.
                    || record.episodeDisclosure?.isConnected
                    || (!record.finished && !record.historicalUnavailable)
                    || historyNodeIsProtected(record.root, messagesDiv));
        });
        for (const id of due) {
            disposeLiveCard(id);
            subagentChildParents.delete(id); subagentTerminalChildren.delete(id);
            explicitCardExpansion.delete(id); reviewDisclosureByTask.delete(id);
            pendingLiveEvictions.delete(id);
        }
    }
    function applyHistoryMessages(messages, { fromReconnect = false, archived = false } = {}) {
        const knownCards = new Set(liveCardRecords.keys());
        _historyReplayActive = true;
        _historyAppending = archived;
        liveCardBound.beginReplay();
        try {
            withStableViewport(() => {
                replayKeys = { history: new Set(), client: new Set(), origin: new Set(), legacy: new Set() };
                noteBubbleKeys(messagesDiv);
                replaySelection = keepSelection(document);
                // A formerly unbound final can gain its real child lineage on a
                // deeper page. Keep a selected old bubble until it can move: every old bubble is judged before one leaves.
                const olds = new Map(messages.filter(row => row.history_id && row.delegation_role === 'subagent' && !row.is_progress
                    && replayKeys.history.has(row.history_id))
                    .map(row => [row, historyNodes(row.history_id).find(node => node.classList.contains('chat-bubble'))])
                    .filter(([, old]) => old));
                const held = new Set([...olds].filter(([, old]) => historyNodeIsProtected(old, messagesDiv)).map(([row]) => row));
                messages = messages.filter(row => {
                    const old = olds.get(row);
                    if (!old?.isConnected) return true;
                    if (held.has(row)) {
                        pendingHistoryUpserts.set(row.history_id, row); return false;
                    }
                    pendingHistoryUpserts.delete(row.history_id);
                    releaseMessageNode(old); seenMessageKeys.delete(`history:${row.history_id}`);
                    return true;
                });
                // Retire only local echoes this source snapshot confirms.
                const localEcho = partitionLocalEchoJournal(localEchoJournal, new Set(messages
                    .filter((m) => m.role === 'user' && m.client_message_id)
                    .map((m) => String(m.client_message_id))));
                for (const entry of localEcho.confirmed) {
                    localEchoJournal.delete(entry.clientMessageId);
                }
                for (const msg of messages) {
                    learnSubagentLineage(msg);
                    // A notice names its target, not a speaker: no evidence of that task here.
                    if (msg.task_id && !ORIGIN_ADDRESSED_NOTICE_TYPES.has(msg.system_type)
                            && msg.system_type !== 'project_question_pointer') evidencedTaskIds.add(msg.task_id);
                }

                const historicalTerminalProjections = new Set();
                for (const msg of messages) if (msg.historical_terminal && msg.task_id) {
                    historicalTerminals.set(msg.task_id, { ...msg.historical_terminal, wake_summary: msg.wake_summary, suggested_name: msg.suggested_name, schedule_name: msg.schedule_name });
                    if (msg.summary_kind === 'terminal_root_projection' || msg.outcome_final === true) {
                        historicalTerminalProjections.add(msg.task_id);
                    }
                }
                // Rows pass 1 attached to a card; pass 2 mounts the card with them.
                const cardRowsAttached = new Set();
                // First pass builds card state without DOM insertion.
                _syncPass1Active = true;
                try { for (const msg of messages) {
                    _historyRow = msg;
                    if (['quiz_answer', 'quiz_closed'].includes(msg.system_type)) chatDecision.applyQuizStateFrame(messagesDiv, { ...msg.quiz, task_id: msg.task_id });
                    if (msg.system_type === 'task_evidence') { admitCardMetadata(msg); continue; }
                    if (isReplayEvidenceRow(msg) || msg.system_type === 'project_question_pointer') continue;
                    if (admitCardMetadata(msg) !== undefined) continue;
                    if (attachReviewFromRow(msg, msg.ts || '') !== undefined) continue;
                    if (attachCardRow(msg, msg.ts || '', { suppressDomInsert: true }) !== undefined) {
                        cardRowsAttached.add(msg);
                        continue;
                    }
                    // Placement precedes progress; later real rows may create the card.
                    if (isPlacedCardRow(msg)) continue;
                    const taskId = msg.task_id || '';
                    if (!taskId) continue;
                    if (msg.is_progress) {
                        if (msg.suggested_name) applySuggestedName(taskId, msg.suggested_name);
                        updateLiveCardFromProgressMessage(msg, { grantCancelAuthority: msg.project_mirror !== true });
                        const historyCard = liveCardRecords.get(taskId);
                        if (historyCard && !historyCard.isSubagent && !activeDirectActivities.has(taskId)
                                && !msg.task_terminal_status && msg.task_phase !== 'finalizing'
                                && setHistoricalUnconfirmed(historyCard)) {
                            syncCancelRunButton(historyCard);
                            renderLiveCardMeta(historyCard);
                        }
                        if (historyCard && historicalTerminals.has(taskId)) historyCard.historicalTerminal = historicalTerminals.get(taskId);
                        continue;
                    }
                    if (msg.historical_terminal) getLiveCardRecord(taskId).quietEffective = msg.historical_terminal.quiet_effective;
                    if (msg.system_type === 'task_summary') appendTaskSummaryToLiveCard(msg, { suppressDomInsert: true });
                } } finally { _syncPass1Active = false; _historyRow = null; }

                // Pass 2 inserts cards at the first visible task message, then finishes them.
                const insertedCardTaskIds = new Set();
                function mountReplayedCard(rec) {
                    if (rec?._anchorOrderDirty && !rec.isSubagent && rec.root?.isConnected) {
                        insertMessageNode(rec.root, { reorderExisting: true });
                        rec._anchorOrderDirty = false;
                    }
                    ensureLiveCardVisible(rec);
                }
                function insertCardIfNeeded(taskId) {
                    if (!taskId || insertedCardTaskIds.has(taskId)) return;
                    insertedCardTaskIds.add(taskId);
                    mountReplayedCard(liveCardRecords.get(taskId));
                }
                // A stored row as its bubble: its own identity and labels, plus what its kind adds.
                const addStoredRow = (msg, taskId, extra) => addMessage(msg.text, msg.role, !!msg.markdown, msg.ts || null, false, {
                    presenceContact: msg.presence_contact,
                    historyId: msg.history_id, historyPosition: msg.history_position,
                    systemType: msg.system_type || '', narration: msg.narration, taskId,
                    source: msg.source || '', initiator: msg.initiator || '',
                    senderLabel: msg.sender_label || '', senderSessionId: msg.sender_session_id || '', reply: msg,
                    ...extra,
                });
                for (const msg of messages) {
                    _historyRow = msg;
                    const taskId = msg.task_id || '';
                    if (msg.system_type === 'project_question_pointer') { chatDecision.appendQuestionPointer(msg); continue; }
                    if (isReplayEvidenceRow(msg)) continue;
                    // Owner-bound reviews attached in pass 1 are not terminal bubbles.
                    if (
                        admitCardMetadata(msg) !== undefined
                        || attachReviewFromRow(msg, msg.ts || '', true) !== undefined
                        // A record minted after its row in pass 1 takes the row here.
                        || cardRowsAttached.has(msg) || attachCardRow(msg, msg.ts || '') !== undefined
                    ) continue;
                    if (isPlacedCardRow(msg)) {
                        addStoredRow(msg, taskId, { evidenceRef: cardRowEvidenceRef(msg) });
                        continue;
                    }
                    // History clears Sending; census re-links live turns.
                    if (fromReconnect && msg.role === 'user' && msg.client_message_id) {
                        pendingSubmissions.delete(String(msg.client_message_id));
                    }
                    if (msg.is_progress) {
                        // Progress-only/failed tasks still anchor at their first event.
                        insertCardIfNeeded(taskId);
                        // Open post-task checkpoint replays as "Finalizing…".
                        if (msg.task_phase === 'finalizing') markLiveCardFinalizing(taskId, msg);
                        continue;
                    }
                    if (msg.system_type === 'task_summary') continue;
                    if (PROJECT_ROW_TYPES.has(msg.system_type)) {
                        portals.row(msg, msg.text, { historyId: msg.history_id, historyPosition: msg.history_position, taskId });
                        continue;
                    }
                    if (appendDelivery(msg)) continue;
                    if (subagentChildParents.has(taskId) && childRowPlacement(msg) === 'line') {
                        insertCardIfNeeded(taskId); routeSubagentProgressToCard(taskId, msg, true); continue;
                    }
                    if (taskId && replayConcludesTask(msg)) {
                        insertCardIfNeeded(taskId);
                        const child = subagentChildParents.has(taskId);
                        if (child) routeSubagentFinalMessageToCard(taskId, msg);
                        // A replayed early final must not finalize a root's card.
                        if (!child && msg.task_phase === 'finalizing') markLiveCardFinalizing(taskId, msg);
                        else if (child || !msg.historical_terminal || positiveTaskTerminalFact(msg)) {
                            finishLiveCard(taskId, msg.task_terminal_status ? taskTerminalPhase(msg) : replayTerminalPhase(liveCardRecords.get(taskId)));
                        }
                        if (child) continue;
                    }
                    // A durable routing receipt retires Sending like its live frame.
                    if (msg.chat_annotation && msg.client_message_id) {
                        pendingSubmissions.delete(String(msg.client_message_id));
                    }
                    addStoredRow(msg, taskId, {
                        clientMessageId: msg.client_message_id || '', attachments: msg.attachments,
                        projectId: msg.role === 'assistant' ? msg.project_id || '' : '', projectName: msg.project_name || '',
                        textPlaceholder: msg.text_placeholder === true,
                        originProjected: msg.origin_projected === true,
                        originId: msg.origin_id || '',
                        chatAnnotation: msg.chat_annotation || null,
                        skillReview: msg.system_type === 'skill_review' && msg.skill && msg.job_id
                            ? { skill: msg.skill, jobId: msg.job_id, ...(msg.record_id ? { recordId: msg.record_id } : {}) }
                            : null,
                    });
                    if (msg.role === 'user') { markIngressSaved(messagesDiv, msg); markSwarmLabel(messagesDiv, msg); replies.adopt(msg); unconfirmed.settle(msg); }
                }
                _historyRow = null;
                // Durable terminal facts settle the named card, including progress-only replays.
                const settleReplayedTerminal = (tid, terminalRecord, phase, summary = false) => {
                    if (!taskDoneIsTerminal(terminalRecord)) return;
                    if (subagentChildParents.has(tid)) return routeSubagentTerminalToCard(tid, terminalRecord);
                    const record = liveCardRecords.get(tid);
                    if (!record) return;
                    if (terminalRecord.suggested_name) applySuggestedName(tid, terminalRecord.suggested_name);  // evidence names a card whose own rows lie elsewhere
                    record.quietEffective = terminalRecord.quiet_effective ?? record.quietEffective;
                    noteTaskDetails(record, terminalRecord);
                    insertCardIfNeeded(tid);
                    if (summary) appendTaskSummaryToLiveCard(terminalRecord); else finishLiveCard(tid, phase);
                };
                const terminalTaskRecords = new Map(messages.filter((msg) => msg.task_id && msg.task_terminal_status)
                    .map((msg) => [msg.task_id, { ...msg, status: String(msg.task_terminal_status) }]));
                for (const [tid, terminalRecord] of terminalTaskRecords) {
                    _historyRow = terminalRecord;
                    settleReplayedTerminal(tid, terminalRecord, taskTerminalPhase(terminalRecord),
                        Boolean(terminalRecord.outcome_axes || terminalRecord.review_projection || terminalRecord.reason_code));
                }
                _historyRow = null;

                // Durable phase outranks narrative rows that may still say finalizing.
                for (const [tid, historicalTerminal] of historicalTerminals) {
                    if (!historicalTerminalProjections.has(tid)) continue;
                    settleReplayedTerminal(tid, { ...historicalTerminal, task_id: tid }, historicalTerminal.phase);
                }

                // Cards without terminal or census evidence remain unconfirmed.
                for (const [id, record] of liveCardRecords) {
                    mountReplayedCard(record);
                    if (!knownCards.has(id) && !record.finished && !currentActivity(record)) {
                        setHistoricalUnconfirmed(record);
                        syncCancelRunButton(record);
                        renderLiveCardMeta(record);
                    }
                }
                for (const row of messages) {
                    const record = liveCardRecords.get(row.presentation_owner_task_id || row.task_id);
                    if (!record) continue;
                    record.historyIds ||= new Set();
                    for (const id of historyRowIds(row)) record.historyIds.add(id);
                }
                _historyReplayActive = false;
                // The cards this page changed are drawn with the scroll and selection they had, read for all at once.
                const dirty = [...liveCardRecords.values()].filter(record => record._timelineDirty);
                measure(() => { const kept = keepSelection(document); return dirty.map(record => ({ scroll: timelineScroll(record), kept })); },
                    (before) => dirty.forEach((record, index) => renderLiveCardTimeline(record, before[index])));
                persistVisibleHistory();
                return true;
            });
        } finally {
            _historyRow = null;
            _historyReplayActive = false;
            _historyAppending = false;
            replayKeys = null; replaySelection = null;
        }
        syncChatStatus();
    }

    async function syncHistory({ fromReconnect = false } = {}) {
        if (destroyed) return false;
        if (historySyncPromise) {
            // A reconnect joining an older read still owes a rebuild.
            if (fromReconnect) {
                pendingReconnectSync = true;
                return historySyncPromise.then(() => {
                    // One waiter consumes the queued rebuild; peers await it.
                    if (pendingReconnectSync) {
                        pendingReconnectSync = false;
                        return syncHistory({ fromReconnect: true });
                    }
                    return historySyncPromise || lastHistorySyncSucceeded;
                });
            }
            return historySyncPromise;
        }
        historySyncPromise = (async () => {
            const armedAtStart = liveCardBound.begin();
            const cardsAtStart = new Set(liveCardRecords.keys());
            // Replay can refill the feed without moving the reader's intent.
            const followingAtStart = reading.stick, generationAtStart = reading.generation;
            const landedAtStart = historyPager.arrivalsLanded();
            try {
                // No welcome until this read lands.
                emptyWelcome?.historyPending();
                // An empty feed shows the read in flight (#1102); a painted one stays.
                if (historyControls.beginRecent()) syncLoadOlderControl();
                const data = await fetchHistory(null);
                // Closed rooms do not consume late responses.
                if (destroyed) {
                    lastHistorySyncSucceeded = false;
                    initialHydrationPromise = null;
                    return false;
                }
                if (data.recentVersion < recentApplied) return lastHistorySyncSucceeded;
                const messages = Array.isArray(data.messages) ? data.messages : [];
                const oldRecentIds = recentHistoryIds;
                const admitted = acceptRecentWindow(data, messages);
                const pagerBeforeRecent = historyPager.getState();
                // A reader in older history keeps the chain `Load more history` goes on
                // from, and the rows the newest read let go stay below them; a shifted
                // window re-anchors only a reader following the newest message.
                const reader = readingHistory();
                const rechainRecent = admitted && !reader && !data.reason_code && pagerBeforeRecent.initialized
                    && !pagerBeforeRecent.canNewer && [...oldRecentIds].some(id => !recentHistoryIds.has(id));
                const result = historyPager.acceptRecent(data, rechainRecent && { fromReconnect });
                if (result.status !== 'applied') applyHistoryMessages(messages, { fromReconnect });
                withStableViewport(() => releaseHistoryIds(!admitted || rechainRecent || reader
                    || historyPager.getState().canNewer ? [] : oldRecentIds));
                // Arrivals that landed meanwhile may lie past this page: read them again.
                if (historyPager.arrivalsLanded() !== landedAtStart) void historyPager.read();
                if (armedAtStart) {
                    const represented = new Set(messages.map(row => row.presentation_owner_task_id || row.task_id));
                    for (const id of cardsAtStart) if (!represented.has(id)) pendingLiveEvictions.add(id);
                    withStableViewport(releaseLiveOverflow);
                }
                syncChatStatus();

                composer.seedInputHistory(messages);

                const wasFirstLoad = !historyLoaded;
                // Recovery from a failed first read still opens at the newest message.
                const opensAfterPreview = !sourceHydrated && followingAtStart && reading.generation === generationAtStart;
                historyLoaded = sourceHydrated = true;
                lastHistorySyncSucceeded = true;
                messagesDiv.dataset.historyHydrated = 'true';
                emptyWelcome?.historyRead(data.window?.complete === true);
                liveCardBound.settle({ rebuilt: wasFirstLoad || armedAtStart, size: liveCardRecords.size });
                modelWaits.retainCards(liveCardRecords);
                // Later hydration joins this successful source read.
                initialHydrationPromise = historySyncPromise;
                syncLoadOlderControl();
                // A room opens at its newest message (owner decisions 2026-07-10,
                // 2026-10-05); an in-room intent still awaiting data lands first.
                if (reading.pending) {
                    updateMessagesPadding(false);
                    reading.position();
                } else if (wasFirstLoad && reading.stick) {
                    updateMessagesPadding();
                    reading.followAfterLayout();
                } else if (opensAfterPreview) {
                    updateMessagesPadding();
                    reading.followAfterLayout();
                }
                readReceipt.settle();
                return messages.length > 0;
            } catch (err) {
                if (destroyed) return false;
                lastHistorySyncSucceeded = false;
                emptyWelcome?.historyRead(false);
                initialHydrationPromise = null;
                // Migration keeps a calm note and retries through the existing 700 ms scheduler.
                historyControls.endRecent(err); syncLoadOlderControl();
                if (isStoreMigrating(err)) { historyResyncScheduler.schedule(true); return false; }
                const socketState = ws?.ws?.readyState;
                const expectedDisconnect = socketState !== WebSocket.OPEN;
                if (expectedDisconnect && err instanceof TypeError) {
                    return false;
                }
                console.error('Failed to load chat history:', err);
                return false;
            } finally {
                historySyncPromise = null;
                // A reconnect caller waiting on the active promise owns replay of
                // pendingReconnectSync above, so its own promise resolves only after
                // the authoritative rebuild.
            }
        })();
        return historySyncPromise;
    }

    // A new revision needs a read begun after it: one in flight is awaited, never
    // joined; the sticky hydration answers a covered one (project_read_state.js).
    const newestFacts = { delivered: id => retainedHistoryIds().has(id), nodes: id => historyNodes(id),
        viewport: messagesDiv, header: pageHeader, composer: inputArea };
    const readReceipt = createProjectReadReceipt({
        read: async (fresh, whole) => {
            await (!fresh && initialHydrationPromise || historyPager.read({ whole }));
            return lastHistorySyncSucceeded;
        },
        // A destroyed page reports hidden===false, hence the lifecycle flag.
        isShown: () => !destroyed && !leaving && !page.hidden,
        // A place still being restored is not where the reader is.
        isReadingLatest: latest => !destroyed && !reading.pending && !replies.revealing && isInstanceVisible() && isAtNewestMessage(latest, {
            ...newestFacts, atBottom: () => isNearBottom() && !historyPager.getState().canNewer,
        }),
        onReadingLatest, arriving: () => reading.arriving, delivered: newestFacts.delivered,
        older: async current => {
            await historyPager.whenIdle();
            if (!current()) return {};
            const direction = readReceipt.continuation(pagerRecent);
            // A failed present keeps the existing horizon repair.
            if (direction === 'latest' || (direction && !historyPager.getState().error)) return historyPager[direction]();
            return loadHistoryAtEdge('older', current);
        },
        present: current => historyPager.whenIdle().then(() => current() && historyPager.getState().canNewer && historyPager.latest()),
        land: (latest, token) => reading.land(() => newestMessageOffset(latest, newestFacts), token),
    });

    (async () => {
        await loadUiPreferences();
        // Main waits for hydration; Projects pass through. History reads share one flight.
        await waitForHydrationWindow();
        if (destroyed) return;
        // A message this tab kept across a reload that the first read (or its preview) did not show saved is in doubt.
        const reconcileUnsaved = () => !destroyed && unconfirmed.reconcile(({ frame, views, ts }) => addMessage(frame.content,
            'user', false, ts || null, false, { source: 'web', senderSessionId: chatSessionId, clientMessageId: frame.client_message_id, attachments: views }));
        if (await awaitInitialHydration()) return reconcileUnsaved();
        try {
            const saved = JSON.parse(sessionStorage.getItem(storeKey(CHAT_STORAGE_KEY)) || '[]');
            // Replay the persisted addMessage options through the same normalizer.
            for (const { text, role, markdown, ts, ...restored } of saved) {
                addMessage(text, role, !!markdown, ts || null, false, restored);
            }
        } catch {}
        historyLoaded = true;
        // The next successful source read reconciles this offline preview.
        if (!lastHistorySyncSucceeded) liveCardBound.arm();
        reconcileUnsaved();
    })();

    // The composer (composer.js): its field, the room's capsules, attachments and the send path.
    composer = createComposer({ page, byId, ws, replies, chatSessionId, isMain, chatId, projectId, storeKey,
        refreshState: refreshHeaderControlState, openSettings: openSettingsTab, onLayout: () => updateMessagesPadding(),
        onSent: showSent, readingBack: () => !reading.stick && reading.ownMove });

    // A send the composer made, shown in the feed: the bubble, its doubt until the host's saved echo, `Sending...`.
    function showSent({ text, frame, result, views }) {
        const sentTs = new Date().toISOString();
        reading.cancel(); // an accepted Send supersedes any saved place
        const row = addMessage(text, 'user', false, sentTs, false, {
            pending: result?.status === 'queued',
            source: 'web',
            senderSessionId: chatSessionId,
            clientMessageId: result?.clientMessageId || '',
            forceStick: true, reply: replies.echoRow(),
            attachments: views,
        });
        replies.clear();
        // Only the host's saved echo proves acceptance: every message's frame stays (a bare slash command is a
        // command); a queued one is the socket queue's until it is written.
        if (views.length || !text.startsWith('/')) unconfirmed.track({ ...frame, client_message_id: result.clientMessageId },
            { views, ts: sentTs, queued: result?.status === 'queued' });
        // Chat sends carry their minted client_message_id.
        const pendingId = result?.clientMessageId || '';
        if (pendingId) {
            pendingSubmissions.set(pendingId, {
                clientMessageId: pendingId,
                timestamp: Date.now(),
            });
            recordLocalEcho(pendingId, text, sentTs);
        }
        syncChatStatus();
        composer.resize();
        reading.followAfterLayout();
        return row;  // the sent draft rises into it (send_flight.js)
    }
    // Dynamic CSS reserve keeps the absolute composer from covering messages.

    // Scroll events observe position; only positive navigation changes follow intent.
    messagesDiv?.addEventListener('scroll', () => { reading.scroll(); settleHistoryViewport(); readReceipt.note(); }, { passive: true });
    // An image's load reflows in the next frame, after that frame's scroll events: a load
    // landing between a move (a wheel step, a scrollIntoView) and the scroll event the move
    // still owes would otherwise restore the anchor of the place just left.
    let loadReflowQueued = false;
    messagesDiv?.addEventListener('load', () => {
        if (loadReflowQueued) return;
        loadReflowQueued = true;
        requestAnimationFrame(() => { loadReflowQueued = false; if (!destroyed) reading.reflow(); });
    }, true);

    // Navigation plus one coalesced, non-live-region remote-activity bit.
    function updateScrollButton() {
        if (!scrollBottomBtn) return;
        if (_hasNewActivity && isNearBottom(ACTUAL_BOTTOM_TOLERANCE_PX)) {
            _hasNewActivity = false;
        }
        const label = _hasNewActivity
            ? 'New activity — scroll to latest message'
            : 'Scroll to latest message';
        scrollBottomBtn.setAttribute('aria-label', label);
        scrollBottomBtn.title = label;
        if (scrollActivityDot) scrollActivityDot.hidden = !_hasNewActivity;
        scrollBottomBtn.classList.toggle('visible', isInstanceVisible() && (!isNearBottom() || (!reading.stick && historyWindow?.gaps)));
    }
    // The one way to the present: ↓, and returning to the room (owner decision
    // 2026-10-05: a room is always reopened at its newest message), unread at what arrived last.
    async function showLatest({ arrive = false } = {}) {
        const current = reading.claim(arrive === true);
        if (historySyncPromise) await historySyncPromise;
        await historyPager.whenIdle();
        if (destroyed || !current()) return;
        if (historyPager.getState().away || historyWindow?.horizonGap) {
            if ((await historyPager.latest()).status !== 'applied') return;
        }
        if (destroyed || !current()) return;
        reading.stick = true;
        for (const record of liveCardRecords.values()) record.childDetailRead = undefined;
        reading.followAfterLayout();
        updateScrollButton();
        historyControls.showNotice();
    }
    scrollBottomBtn?.addEventListener('click', showLatest);

    // The composer's height once its reply strip has finished moving (chat_reply.js `data-motion`): opening, the strip
    // counts at its full height (`--reply-h`) and the wrap's gap; leaving, not at all. The reserve then moves once per
    // motion, not on every frame of it: each write restyles the whole room (~90 ms a write on a 34k-element Main in
    // WebKit, where four in a row made the strip's close stutter).
    function composerHeightAtRest() {
        const strips = inputArea.querySelectorAll('.chat-quote--composer[data-motion]');
        if (!strips.length) return inputArea.offsetHeight || 0;
        // Mid-motion heights are fractional: whole pixels would differ by one from frame to frame and write again.
        let height = inputArea.getBoundingClientRect().height;
        for (const strip of strips) {
            // The room a strip takes is its box and its margins (style.css): opening it ends at --reply-room, shut at none.
            const { marginTop, marginBottom } = getComputedStyle(strip);
            const now = strip.getBoundingClientRect().height + (parseFloat(marginTop) || 0) + (parseFloat(marginBottom) || 0);
            const end = strip.dataset.motion === 'open' ? (parseFloat(strip.style.getPropertyValue('--reply-room')) || 0) : 0;
            height += end - now;
        }
        return height;
    }

    function updateMessagesPadding(preserveStickiness = true) {
        // A room not displayed (Main behind a page) measures its bars as 0: it keeps the reserves it was laid out
        // with, or it would come back with its newest message under the composer.
        if (!page.getClientRects().length) return false;
        const mutate = () => {
            let changed = false;
            if (pageHeader && messagesDiv) {
                // The bar's own height places the note under it; the transcript clears both.
                const bar = Math.ceil(pageHeader.offsetHeight || 0);
                const note = pageNote?.offsetParent ? Math.ceil(pageNote.offsetHeight || 0) : 0;
                const barValue = `${bar}px`;
                if (page.style.getPropertyValue('--chat-header-bar') !== barValue) {
                    page.style.setProperty('--chat-header-bar', barValue);
                    changed = true;
                }
                const value = `${Math.max(56, bar + note)}px`;
                if (page.style.getPropertyValue('--chat-header-reserve') !== value) {
                    page.style.setProperty('--chat-header-reserve', value);
                    changed = true;
                }
            }
            if (inputArea && messagesDiv) {
                const reserve = Math.max(92, Math.ceil(composerHeightAtRest()) + 16);
                const value = `${reserve}px`;
                if (page.style.getPropertyValue('--chat-input-reserve') !== value) {
                    page.style.setProperty('--chat-input-reserve', value);
                    changed = true;
                }
            }
            return changed;
        };
        const changed = preserveStickiness ? withStableViewport(mutate) : mutate();
        updateScrollButton();
        return changed;
    }

    // Kept on the instance so destroy() can disconnect it (the observer was
    // previously an unreachable closure — the P3 lifecycle leak).
    let chatResizeObserver = null;

    function installChatResizeObservers() {
        if (typeof ResizeObserver !== 'function') return;
        let queued = false;
        const schedule = () => {
            if (queued) return;
            queued = true;
            requestAnimationFrame(() => {
                queued = false;
                if (destroyed) return;
                reading.reflow();
                updateMessagesPadding();
            });
        };
        chatResizeObserver = new ResizeObserver(schedule);
        chatResizeObserver.observe(messagesDiv);
        if (pageHeader) chatResizeObserver.observe(pageHeader);
        if (pageNote) chatResizeObserver.observe(pageNote);
        if (inputArea) chatResizeObserver.observe(inputArea);
    }

    installChatResizeObservers();

    let headerControlInterval = null;
    if (asPanel) {
        // Late panels missed socket open; seed from the socket and latest snapshot, its works included: a working
        // Project opened from its row said Online until its own first read (owner 20.10).
        const latest = stateSnapshots.latest?.();
        hostStart.read(latest);
        workState = reconcileWorkState(workState, latest, chatId, ws.isConnected?.() !== false);
        const seed = deriveChatStatus();
        if (ws.isConnected?.()) setStatus(seed);
    } else headerControlInterval = setInterval(refreshHeaderControlState, 3000);
    // Both Main and late panels hydrate once; each instance filters its chat.
    refreshHeaderControlState(true);

    const typingEl = document.createElement('div');
    // A per-instance id (Main's stays 'typing-indicator'): open chat columns never share one.
    typingEl.id = idPrefix === 'chat' ? 'typing-indicator' : `${idPrefix}-typing-indicator`;
    typingEl.className = 'chat-bubble assistant typing-bubble';
    typingEl.style.display = 'none';
    typingEl.innerHTML = statusMarkHtml('typing-mark', { shape: 'moving', hue: 'ink' });
    messagesDiv.appendChild(typingEl);

    const pageHistoryIds = new Map();
    let recentCoverage = null;
    let pagerRecent = {};
    let recentHasOrigins = false;
    let recentHistoryIds = new Set();
    let recentRequest = 0, recentApplied = 0;
    async function fetchHistory(cursor, options = {}) {
        const recentVersion = cursor == null ? ++recentRequest : 0;
        return { ...await apiClient.chatHistory({ chatId, cursor, ...options }), recentVersion };
    }
    function acceptRecentWindow(data, messages) {
        if (data.recentVersion <= recentApplied) return false;
        recentApplied = data.recentVersion;
        // The newest admitted window (↓ too) owns the failure note.
        historyControls.endRecent(data.reason_code ? new Error('Some saved history could not be loaded.') : null);
        recentCoverage = data.coverage ?? null;
        recentHasOrigins = messages.some(row => row.origin_projected);
        readReceipt.recent(data);
        const ids = new Set(messages.flatMap(historyRowIds));
        for (const id of recentHistoryIds) if (data.window?.truncated_by?.includes(`${id.split(':')[0]}_source_unavailable`)) ids.add(id);
        recentHistoryIds = ids;
        return true;
    }
    const pendingHistoryEvictions = new Set();
    const historyNodes = (id) => Array.from(messagesDiv.querySelectorAll('[data-history-id]'))
        .filter(node => node.dataset.historyId === id);
    const retainedHistoryIds = () => new Set([...recentHistoryIds,
        ...[...pageHistoryIds.values()].flatMap(ids => [...ids])]);
    function isHistoryPageProtected(descriptor) {
        const ids = pageHistoryIds.get(descriptor.id) || new Set();
        if (historyStamps(messagesDiv).some(([id, node]) => ids.has(id) && historyNodeIsProtected(node, messagesDiv))) return true;
        return [...liveCardRecords.values()].some(record =>
            [...(record.historyIds || [])].some(id => ids.has(id))
            && [record.summaryButtonEl, record.footEl, record.reviewsHostEl].some(node => historyNodeIsProtected(node, messagesDiv)));
    }

    // One retirement path per message node: media, decision views and markdown go with it.
    function releaseMessageNode(node) {
        chatMedia.release(node); chatDecision.releaseViews(node); destroyChatMarkdown(node); node.remove();
    }

    function releaseHistoryIds(ids) {
        if (!(ids.size || ids.length)) return;
        const retained = retainedHistoryIds();
        const released = new Set();
        const byId = new Map();
        for (const [key, node] of historyStamps(messagesDiv)) {
            if (!byId.has(key)) byId.set(key, []);
            byId.get(key).push(node);
        }
        // The page as the reader sees it decides, before any row leaves: a row or card read, focused or selected stays,
        // and a card keeps its timeline's selection. Read for all of them first, one layout for all.
        const due = new Set([...ids].filter(id => !retained.has(id)));
        const held = new Set([...due].filter(id => (byId.get(id) || []).some(node => historyNodeIsProtected(node, messagesDiv))));
        const kept = keepSelection(document);
        const heldCards = new Set([...liveCardRecords].filter(([, record]) => ([...(record.historyIds || [])].some(key => due.has(key))
            || record.items.some(item => due.has(item.historyId || item.sourceHistoryId)))
            && historyNodeIsProtected(record.root, messagesDiv)).map(([id]) => id));
        for (const id of ids) {
            if (retained.has(id)) { pendingHistoryEvictions.delete(id); continue; }
            const nodes = byId.get(id) || [];
            if (held.has(id)) {
                pendingHistoryEvictions.add(id); continue;
            }
            pendingHistoryEvictions.delete(id);
            for (const node of nodes) {
                const wrapper = node.closest('.chat-bubble');
                releaseMessageNode(node);
                if (wrapper && wrapper !== node && !wrapper.querySelector('.chat-gallery-item, .chat-file-item, .chat-quiz-card')) {
                    chatMedia.release(wrapper); wrapper.remove();
                }
            }
            seenMessageKeys.delete(`history:${id}`);
            released.add(id);
        }
        if (!released.size) return;
        // A timeline's scroll is read once its released lines have left (where the engine anchors scrolling, the line
        // being read has stayed in place), for every card they left at once: one layout.
        const at = new Map([...liveCardRecords].filter(([, record]) => record.items.some(item => released.has(item.historyId || item.sourceHistoryId)))
            .map(([id, record]) => [id, { scroll: timelineScroll(record), kept }]));
        for (const [id, record] of [...liveCardRecords].reverse()) {
            let affected = false;
            for (const key of record.historyIds || []) {
                if (released.has(key)) affected = record.historyIds.delete(key) || affected;
            }
            const items = record.items.filter(item => !released.has(item.historyId || item.sourceHistoryId));
            if (items.length !== record.items.length) {
                record.items = items;
                renderLiveCardTimeline(record, at.get(id));
                updateLiveCardCount(record);
            }
            if (!affected || record.historyIds?.size || activeDirectActivities.has(id)
                    || record.episodeDisclosure?.isConnected
                    || record.subagentsEl?.querySelector('.chat-live-card')
                    || heldCards.has(id)) continue;
            if (!record.finished && !record.historicalUnavailable && !record.historicalUnconfirmed) continue;
            disposeLiveCard(id);
            subagentChildParents.delete(id); subagentTerminalChildren.delete(id);
            reviewDisclosureByTask.delete(id); explicitCardExpansion.delete(id); historicalTerminals.delete(id);
        }
    }

    const historyControls = createHistoryControls(messagesDiv, pageNote || page.querySelector('.chat-panel-statusbar'), {
        isVisible: () => isInstanceVisible() && !messagesDiv.closest('[inert]')
            && !document.body.classList.contains('nav-drawer-open') && (asPanel || (!isProjectOpening?.()
            && !document.body.classList.contains('project-panel-open'))),
    });
    const { olderButton: loadOlderBtn } = historyControls;

    const historyPager = createChatHistoryPager({
        fetchPage: fetchHistory,
        isAlive: () => !destroyed,
        applyPage: (messages, descriptor) => {
            pageHistoryIds.set(descriptor.id, new Set(messages.flatMap(historyRowIds)));
            const oldRecentIds = recentHistoryIds;
            const admitted = descriptor.direction === 'latest' && (acceptRecentWindow(descriptor, messages) || descriptor.rechained);
            const archived = descriptor.direction !== 'recent' && descriptor.direction !== 'latest';
            if (!archived) pagerRecent = descriptor;
            applyHistoryMessages(messages, { ...descriptor, archived });
            // The present re-anchors the chain: rows no page or window owns any more
            // (windows kept while the reader was in older history) go with it.
            if (admitted) withStableViewport(() => releaseHistoryIds(new Set([...oldRecentIds,
                ...historyStamps(messagesDiv).map(([id]) => id)])));
            // An older page drawn can show, or name, the newest arrival without a scroll.
            if (admitted) readReceipt.settle();
            else if (archived) readReceipt.page(descriptor);
        },
        releasePage: descriptor => {
            const ids = pageHistoryIds.get(descriptor.id) || [];
            pageHistoryIds.delete(descriptor.id);
            withStableViewport(() => releaseHistoryIds(ids));
        },
        // Rows stored since the newest read join its window.
        applyArrivals: (messages, { window, fromReconnect }) => {
            for (const id of messages.flatMap(historyRowIds)) recentHistoryIds.add(id);
            applyHistoryMessages(messages, { fromReconnect });
            readReceipt.arrived(window);
        },
        readNewest: syncHistory, newestInFlight: () => historySyncPromise,
        canArrive: () => sourceHydrated && lastHistorySyncSucceeded && !liveCardBound.isArmed(),
        isPageProtected: isHistoryPageProtected,
        onState: snapshot => syncLoadOlderControl(snapshot),
    });

    function syncLoadOlderControl(snapshot = historyPager.getState()) {
        if (destroyed) return;
        withStableViewport(() => {
            const coverage = historyCoverage(recentCoverage, snapshot.coverage);
            if (recentHasOrigins && !coverage.complete) coverage.gaps = true;
            historyWindow = historyControls.render(snapshot, coverage, reading.approximate);
            return true;
        });
    }
    const handleHistoryLanguage = () => syncLoadOlderControl();
    window.addEventListener('ouro:language-changed', handleHistoryLanguage);
    async function loadOlderHistory() {
        if (historyControls.recentFailed()) {
            const owned = onHistoryRetry?.();
            return historySyncPromise ? owned : syncHistory();
        }
        const snapshot = historyPager.getState();
        if (snapshot.error?.body?.reason_code === 'history_view_changed') {
            reading.cancel();
            return historyPager.latest();
        }
        if (snapshot.error) return historyPager.retry();
        return loadHistoryAtEdge('older');
    }
    // `Load more history` reads older messages; ↓ returns to the present (owner
    // decisions 2026-09-14, 2026-10-05). Empty pages cross the server read ceiling:
    // read on until rows land or the chat ends. A grown room re-anchors unless the
    // reader is still walking its older chain. Openings guard each request with `current`.
    async function loadHistoryAtEdge(direction, current = () => true) {
        const { canNewer, canOlder } = historyPager.getState();
        if (direction === 'older' && current() && historyWindow?.horizonGap && !canNewer && !(canOlder && readingHistory())) {
            const rebased = await historyPager.latest();
            if (destroyed || rebased.status !== 'applied') return rebased;
        }
        let result;
        do {
            result = current() ? await historyPager[direction]() : {};
        } while (!destroyed && result.status === 'applied' && result.messageCount === 0
            && historyPager.getState()[direction === 'older' ? 'canOlder' : 'canNewer']);
        return result;
    }
    loadOlderBtn.addEventListener('click', loadOlderHistory);

    function settleHistoryViewport() {
        if (destroyed || reading.pending || !historyLoaded || !isInstanceVisible()) return;
        retryHistoricalUpserts();
        historyPager.trim();
        if (pendingLiveEvictions.size) withStableViewport(releaseLiveOverflow);
        if (pendingHistoryEvictions.size) withStableViewport(() => releaseHistoryIds([...pendingHistoryEvictions]));
    }
    function navigateHistoryAtEdge(direction) {
        if (destroyed || reading.pending || !historyLoaded || !isInstanceVisible()) return;
        settleHistoryViewport();
        const snapshot = historyPager.getState();
        if (snapshot.loading || snapshot.error || !snapshot.initialized) return;
        if (direction < 0 && messagesDiv.scrollTop < 80 && snapshot.canOlder) void loadHistoryAtEdge('older');
        else if (direction > 0 && snapshot.canNewer) {
            // Only an unambiguous page-owned island has a physical edge. A
            // mixed card has no chronological boundary: use the same button.
            const ids = pageHistoryIds.get(snapshot.firstPage?.id) || new Set();
            if (historyIslandAtEdge(messagesDiv, ids, historyNodes)) void loadHistoryAtEdge('newer');
        }
    }
    const disposeReadingGestures = reading.bindGestures(navigateHistoryAtEdge);
    // Compact tables share rich-answer keyboard/overflow affordances; destroy releases them.
    bindMarkdownTables(messagesDiv);
    function retryHistoricalUpserts() {
        const ready = [...pendingHistoryUpserts.values()].filter(row =>
            !historyNodes(row.history_id).some(node => historyNodeIsProtected(node, messagesDiv)));
        if (ready.length) applyHistoryMessages(ready, { archived: true });
    }
    document.addEventListener('selectionchange', retryHistoricalUpserts);

    // Active blocks host their own running indicator.
    const foregroundCards = () => Array.from(liveCardRecords.values()).filter((r) => isForegroundLiveCard(r) && !r.modelWaiting);

    // The server's room under this client's own facts.
    function deriveChatStatus() {
        return roomStatusView(workState, { connected: ws.isConnected ? ws.isConnected() : true, ready: hostStart.ready(),
            boot: hostStart.status().bootPhase,
            sending: pendingSubmissions.size > 0, stopping: ownStopPress(workState, liveCardRecords) });
    }

    function syncChatStatus() {
        const derived = deriveChatStatus();
        setStatus(derived);
        return setTypingIndicatorVisible(derived.mark === 'moving' && foregroundCards().length === 0);
    }

    function setTypingIndicatorVisible(visible) {
        const display = visible ? '' : 'none';
        if (typingEl.style.display === display) return false;
        return withStableViewport(() => {
            typingEl.style.display = display;
            return true;
        });
    }

    function hideTypingIndicatorOnly() {
        return setTypingIndicatorVisible(false);
    }

    function revokeManagedTaskCancelAuthority(taskId) {
        cancelableTaskIds.delete(taskId);
        syncCancelRunButton(liveCardRecords.get(taskId));
    }

    async function reconcileMissingManagedTask(taskId, onDomWrite = withStableViewport) {
        if (
            destroyed
            || managedTaskDetailReads.has(taskId)
            || concludedDirectActivities.has(taskId)
            || !missingManagedTaskIds.has(taskId)
        ) return;
        managedTaskDetailReads.add(taskId);
        try {
            const detail = await fetchTaskDetailStrict(taskId);
            if (destroyed || concludedDirectActivities.has(taskId)) return;
            const currentRecord = liveCardRecords.get(taskId);
            if (!currentRecord || currentRecord.finished) return;
            onDomWrite(() => {
                let changed = Boolean(attachTaskDetailReviews(taskId, detail));
                const cancelPending = taskCancelPending(detail);
                if (cancelPending || taskKey(detail?.status)) {
                    changed = setHistoricalUnavailable(currentRecord, false) || changed;
                    changed = markReviewAnchor(currentRecord) || changed;
                }
                const vouched = !missingManagedTaskIds.has(taskId) || activeDirectActivities.has(taskId);
                if (!vouched && detail === null) {
                    revokeManagedTaskCancelAuthority(taskId);
                    return settleMissingTaskRecord(currentRecord) || changed;
                }
                if (cancelPending || (!vouched && !isTerminalTaskDetail(detail))) {
                    if (currentRecord.isSubagent && detail?.status) missingManagedTaskIds.delete(taskId);
                    return Boolean(reconcileCancelCardFromDetail(currentRecord, taskId, detail) || changed);
                }
                if (vouched) return changed;
                recordTerminalActivity(taskId, taskTerminalPhase(detail));
                return Boolean(appendTaskSummaryToLiveCard({ ...detail, task_id: taskId }) || changed);
            });
        } catch {
            // No terminal fact was proved. A later existing snapshot retries.
        } finally {
            managedTaskDetailReads.delete(taskId);
        }
    }

    function observeMissingManagedTask(taskId, onDomWrite = withStableViewport) {
        const id = taskKey(taskId);
        if (!id || concludedDirectActivities.has(id) || subagentChildParents.has(id) || liveCardRecords.get(id)?.isSubagent) return;
        missingManagedTaskIds.add(id);
        void reconcileMissingManagedTask(id, onDomWrite);
    }

    function hydrateDirectActivities(turnsList, snapshotBarrierMs = Infinity, complete = false) {
        if (!Array.isArray(turnsList)) return;
        const connected = ws.isConnected?.() !== false;
        complete &&= connected;
        const {
            activities: nextMap,
            departedManagedTaskIds,
            disappearedManagedTaskIds,
            concludedDirectActivities: settledDirectRows,
            globallyActiveActivityIds,
        } = reconcileHydratedDirectActivities(
            activeDirectActivities, turnsList, chatId,
            concludedDirectActivities, complete,
        );
        activeDirectActivities.clear();
        for (const [k, v] of nextMap.entries()) {
            if (v.clientMessageId) pendingSubmissions.delete(v.clientMessageId);
            const record = liveCardRecords.get(k);
            if (!connected) v._activityUnconfirmed = true;
            const facts = censusTaskFacts(v);
            if (facts.ended) {
                recordTerminalActivity(k, facts.outcome);
                if (record) finishLiveCardMutation(k, facts.outcome);
                continue;
            }
            activeDirectActivities.set(k, v);
            restoreCardActivity(liveCardRecords.get(k), v.project_admission_hold);
            syncParkedPhase(record, v.phase, v);
            syncCancelRunButton(record);
            markReviewAnchor(record);
            if (v.kind === 'managed_task') missingManagedTaskIds.delete(k);
        }
        for (const taskId of globallyActiveActivityIds) missingManagedTaskIds.delete(taskId);
        for (const row of settledDirectRows) {
            // Visible task cards settle from durable detail in the scan below.
            if (!REUSABLE_TASK_IDS.has(row.activityId)
                    && !isForegroundLiveCard(liveCardRecords.get(row.activityId))) recordConcludedActivity(row.activityId);
            if (row.clientMessageId) pendingSubmissions.delete(row.clientMessageId);
        }
        for (const taskId of departedManagedTaskIds) revokeManagedTaskCancelAuthority(taskId);
        for (const taskId of disappearedManagedTaskIds) observeMissingManagedTask(taskId);
        if (complete) for (const taskId of unconfirmedForegroundCardIds(
            Array.from(liveCardRecords, ([id, r]) => ({
                id, finished: r.finished, isSubagent: r.isSubagent, connected: r.root?.isConnected,
            })),
            globallyActiveActivityIds,
        )) {
            syncCancelRunButton(liveCardRecords.get(taskId));
            const observedAt = liveCardRecords.get(taskId)?.lastLiveObservedAt || 0;
            if (observedAt < snapshotBarrierMs) {
                // Census absence revokes a live card's Stop; history cards await detail.
                if (observedAt) revokeManagedTaskCancelAuthority(taskId);
                observeMissingManagedTask(taskId);
            }
        }
        if (complete) for (const taskId of missingManagedTaskIds) {
            if (!activeDirectActivities.has(taskId)) void reconcileMissingManagedTask(taskId);
        }
    }

    onWs('typing', (msg) => {
        if (!isMyThread(msg)) return;  // each column shows typing only for its own thread
        // A typing frame is a submission receipt, never liveness: it re-reads the
        // state at once and retires the linked `Sending...` only once that read
        // answers, so the room steps from Sending... to the server's word.
        const cmid = String(msg.client_message_id || '');
        if (!cmid) return;
        void refreshHeaderControlState(true).then(() => {
            if (pendingSubmissions.delete(cmid)) syncChatStatus();
        });
    });

    const isMyThread = msg => chatThreadAccepts(msg, isMain, chatId, state.projectChatIds);
    const isMyLogThread = msg => chatLogThreadAccepts(msg, isMain, chatId, state.projectChatIds);

    onWs('chat', (msg) => {
        if (!isMyThread(msg)) return;
        if (msg.system_type === 'initialization_notice') {
            unconfirmed.unsettle(msg.client_message_id ? [msg.client_message_id] : undefined);
        }
        if (msg.system_type === 'project_question_pointer') { chatDecision.appendQuestionPointer(msg); return; }
        if (msg.role === 'user') {
            const clientMessageId = msg.client_message_id || '';
            const senderSessionId = msg.sender_session_id || '';
            // 2A: the user echo is receipt of the user ROW, not turn start —
            // it settles the bubble but must NOT retire the `Sending...`
            // submission; that takes a linked typing frame / snapshot turn /
            // routing receipt or the turn's conclusion.
            const own = senderSessionId === chatSessionId && clientMessageId;
            if (own) markPendingDelivered(clientMessageId);
            const added = !own && withRemoteActivity(() => addMessage(
                msg.content, 'user', false, msg.ts || null, false, {
                source: msg.source || '',
                senderLabel: msg.sender_label || '',
                senderSessionId,
                clientMessageId,
                taskId: msg.task_id || '',
                attachments: msg.attachments,
                textPlaceholder: msg.text_placeholder === true, reply: msg,
                },
            ));
            withStableViewport(() => markIngressSaved(messagesDiv, msg) | replies.adopt(msg) | markSwarmLabel(messagesDiv, msg));
            unconfirmed.settle(msg);
            syncChatStatus();
            return;
        }

        if (msg.role === 'assistant' || msg.role === 'system') {
            return withRemoteActivity(() => {
            const explicitTaskId = msg.task_id || '';
            // Every path ends by syncing status. Main's number is its room revision, as a Project's (DESIGN "Unread").
            const settled = (result) => {
                syncChatStatus();
                return result;
            };
            if (isModelWaitReference(msg)) reread();
            const reference = admitCardMetadata(msg);
            if (reference !== undefined) return settled(reference);
            const review = attachReviewFromRow(msg, msg.ts || '', true);
            if (review !== undefined) return settled(review);
            const cardRow = attachCardRow(msg, msg.ts || '');
            if (cardRow !== undefined) return settled(cardRow);
            if (PROJECT_ROW_TYPES.has(msg.system_type)) {
                return settled(Boolean(portals.row(msg, msg.content, { taskId: explicitTaskId })));
            }
            learnSubagentLineage(msg);
            if (msg.is_progress) {
                showTaskIncidentToast(msg);
                if (SUBAGENT_LIFECYCLE_EVENTS.has(String(msg.subagent_event || '').toLowerCase())) reread();  // members
                const changed = updateLiveCardFromProgressMessage(msg, { grantCancelAuthority: true });
                return settled(changed);
            }

            // An early final (post-task still running) is NOT the turn's
            // conclusion; task_done or the queue snapshot concludes it.
            const finalizing = Boolean(explicitTaskId) && msg.task_phase === 'finalizing';
            const typedTerminal = positiveTaskTerminalFact(msg);
            if (typedTerminal) reread();
            if (!finalizing && (!explicitTaskId || typedTerminal)) {
                if (explicitTaskId) {
                    // 4A (active set): a keyed final concludes ITS OWN turn —
                    // the finished activity + its linked pending — never a
                    // concurrent turn's state (2A keeps later `Sending...`).
                    const finished = activeDirectActivities.get(explicitTaskId);
                    activeDirectActivities.delete(explicitTaskId);
                    if (!REUSABLE_TASK_IDS.has(explicitTaskId)) recordConcludedActivity(explicitTaskId, taskTerminalPhase(msg));
                    if (finished?.clientMessageId) {
                        pendingSubmissions.delete(finished.clientMessageId);
                    }
                } else if (unkeyedFrameEndsTurn(msg)) {  // a legacy model final
                    activeDirectActivities.clear();
                    pendingSubmissions.clear();
                } else pendingSubmissions.delete(String(msg.client_message_id || ''));
            }

            if (msg.system_type === 'task_summary') {
                const changed = appendTaskSummaryToLiveCard(msg);
                return settled(Boolean(changed));
            }
            // A placed host row is never a child's answer; with no card it stays a System row.
            const placed = explicitTaskId && !msg.card_row && subagentChildParents.has(explicitTaskId) && childRowPlacement(msg);
            if (placed === 'line') return settled(Boolean(routeSubagentProgressToCard(explicitTaskId, msg, true)));
            if (placed) return settled(Boolean(routeSubagentFinalMessageToCard(explicitTaskId, msg)));
            let changed = false;
            if (finalizing) changed = markLiveCardFinalizing(explicitTaskId, msg) || changed;
            else if (explicitTaskId && typedTerminal) {
                changed = appendTaskSummaryToLiveCard(msg) || changed;
            }
            const routingCleared = clearTransientRoutingAnnotations(messagesDiv);
            const added = addMessage(msg.content, msg.role, msg.markdown, msg.ts || null, false, {
                presenceContact: msg.presence_contact,
                systemType: msg.system_type || '', narration: msg.narration,
                source: msg.source || '',
                initiator: msg.initiator || '',
                taskId: explicitTaskId, reply: msg,
                // The saved row's own name, as history keys it; this frame's client_message_id is the owner's.
                clientMessageId: msg.row_client_message_id || '',
                // A placed row whose card is not in this chat keeps its record link.
                evidenceRef: isPlacedCardRow(msg) ? cardRowEvidenceRef(msg) : null,
            });
            return settled(Boolean(added || changed || routingCleared));
            });
        }
    });

    onWs('message_annotation', (msg) => {
        if (!isMyThread(msg)) return;
        if (msg.annotation_type !== 'routing_ack') return;
        const apply = () => updateMessageAnnotation(msg.client_message_id || '', msg);
        (msg.status === 'needs_manual_target' ? withRemoteActivity : withStableViewport)(apply);
        // A routing receipt ends this submission's `Sending...` phase.
        const receiptCid = String(msg.client_message_id || '');
        if (receiptCid && pendingSubmissions.delete(receiptCid)) {
            syncChatStatus();
        }
    });

    onWs('outbound_dropped', (msg) => {
        // Evicted from the offline queue: the submission will never reach
        // the server, so it can never earn a receipt, a turn, or a journal row.
        const cid = String(msg?.clientMessageId || '');
        if (!cid) return;
        unconfirmed.flushed(cid);  // the kept frame offers Send again from now on
        markPendingDropped(cid);
        localEchoJournal.delete(cid);
        if (pendingSubmissions.delete(cid)) syncChatStatus();
    });

    onWs('log', (msg) => {
        if (!msg?.data) return;
        // Log frames carry the task's canonical Project chat_id, so the
        // Project panel alone builds/animates/finalizes that card. Legacy
        // frames without chat_id default to the main chat.
        if (!isMyLogThread(msg)) return;
        withRemoteActivity(() => updateLiveCardFromLogEvent({ ...msg.data, _live_tool_frame: !msg.data._historical }));
        if (!msg.data._historical && ['task_done', 'task_model_wait'].includes(msg.data.type)) reread();
    });

    // Admission names the card and its future Project. The broadcast has only
    // task_id: applySuggestedName ignores ids this thread does not hold.
    onWs('task_named', (msg) => {
        withRemoteActivity(
            () => applySuggestedName(msg?.task_id || '', msg?.suggested_name || ''),
        );
    });

    onWs('outbound_sent', (evt) => {
        const cid = evt?.clientMessageId || '';
        if (cid) {
            unconfirmed.flushed(cid);
            // A socket write is not durable acceptance (2A): settle the
            // bubble only; `Sending...` retires on authoritative evidence.
            markPendingDelivered(cid);
            syncChatStatus();
        }
    });

    const { appendDelivery, appendQuizMessage } = chatMedia.wireDeliveries({
        onWs,
        isMyThread,
        hideTypingIndicatorOnly,
        syncChatStatus,
        seenMessageKeys,
        rememberMessageKey,
        chatMediaMessageKey,
        documentMessageKey,
        buildQuizCard: chatDecision.buildQuizCard,
        applyQuizStateFrame: chatDecision.applyQuizStateFrame,
        messagesRoot: () => messagesDiv,
        deliverContentMutation: withRemoteActivity,
        onQuestionFrame: reread,
    });

    let wsHasConnectedOnce = false;

    onWs('open', (msg) => {
        mainWaiting?.setConnected(true);
        refreshHeaderControlState(true);
        syncChatStatus();
        // The socket's previouslyConnected covers newly mounted Projects too;
        // per-instance state is only the fallback for legacy open events.
        const isReconnect = typeof msg?.previouslyConnected === 'boolean'
            ? msg.previouslyConnected
            : wsHasConnectedOnce;
        // Room status carries the outage; only a reload onto changed code adds a restart notice.
        const restartNotice = isReconnect ? '' : pendingRestartNotice;
        pendingRestartNotice = '';
        wsHasConnectedOnce = true;
        updateMessagesPadding();
        const settled = () => {
            if (restartNotice) addMessage(restartNotice, 'system', false, null, false, { ephemeral: true, systemType: 'reconnect' });
            clearReloadParams();
        };
        loadUiPreferences()
            // Reconnect refetches lost task_done; first open shares idle-gated hydration.
            .then(() => (isReconnect ? syncHistory({ fromReconnect: true })
                : waitForHydrationWindow().then(awaitInitialHydration)))
            .then(settled, settled);
    });

    onWs('close', () => {
        unconfirmed.unsettle();
        mainWaiting?.setConnected(false);
        withStableViewport(() => {
            for (const [id, activity] of activeDirectActivities) {
                activeDirectActivities.set(id, { ...activity, _activityUnconfirmed: true });
            }
            let changed = false;
            for (const [id, record] of liveCardRecords) {
                record.childDetailRead = undefined;
                changed = syncParkedPhase(record, 'unknown', activeDirectActivities.get(id)) || changed;
            }
            return changed;
        });
        hostStart.forget();
        hideTypingIndicatorOnly();
        syncChatStatus();
        syncHeaderControlState({ accounting: { available: false } });
    });

    const revealQuestion = (taskId, quizId, options) => chatDecision.revealQuestion(
        taskId, quizId, projectId, chatId, appendQuizMessage, isInstanceVisible,
        () => { const current = reading.claim(); reading.stick = false; return current; }, reading.scroll, options);
    const historyReveal = createHistoryReveal({ pager: historyPager, reading, feed: messagesDiv, records: liveCardRecords,
        present: isInstanceVisible, expand: record =>
            (explicitCardExpansion.set(record.groupId, true), setLiveCardExpanded(record, true)) });
    roomWork = createRoomWork({ page, asPanel, ws, statusBadge, messagesDiv, reading, stateSnapshots, revealQuestion,
        works: () => workState, liveCardRecords, pendingSubmissions, pendingSuggestedNames,
        cancelRunFromCard, refreshHeaderControlState, captureVisibleTimelineAnchor, restoreVisibleTimelineAnchor,
        historyPager, showLatest, hostReady: hostStart.ready });
    // Main's bar: Restart, Panic, the budget and the consciousness chip; a Project's bar is app.js's.
    header = asPanel ? null : createRoomHeader({ bar: pageHeader, ws, shade: roomWork, openDashboardTab,
        refreshState: refreshHeaderControlState });

    return {
        page,
        chatId,
        projectId,
        // The room's composer (composer.js): its draft, its "+" menu and its voice and chips slots.
        composer,
        // Called by app.js when a kept panel is shown again: a room reopens at its
        // newest message (owner decision 2026-10-05).
        showLatest,
        // Shell exposure after its motion: present retained notice state, without fetching or scrolling.
        showHistoryNotice: () => { if (!destroyed) historyControls.showNotice(); },
        refreshHistory: readReceipt.refresh,
        revealQuestion,
        revealSource: replies.reveal,
        cancelHistoryPaint: readReceipt.cancel,
        // app.js fans its already-existing /api/state refresh to every open
        // thread; panels gain convergence without acquiring their own poll.
        hydrateStateSnapshot,
        // True once a history snapshot has actually been fetched and painted;
        // app.js uses it to decide whether a reopen needs a forced repaint.
        hasPaintedHistory: () => historyLoaded && lastHistorySyncSucceeded,
        // Unsendable client-side state (staged File objects / an in-flight
        // upload / a sent message not yet saved, whose Send again
        // lives here). app.js must hide, not destroy, an instance holding it.
        hasPendingWork: () => composer.hasPendingWork() || unconfirmed.count > 0,
        // app.js, when this room leaves the screen without being destroyed: the
        // modal a file card opened (reader, file dialog) closes and its read
        // stops; staged attachments and uploads are untouched.
        closeTransient: () => { reading.cancel(); roomWork?.leave(); chatMedia.closeTransient(); },
        // app.js retires a room that is leaving the screen and keeps it whole for its exit: its paint receipt, its
        // reading and its transients stop at once, it is neither shown nor read; `stay` is a reopen meanwhile.
        leave: () => { leaving = true; readReceipt.cancel(); reading.cancel(); roomWork?.leave(); chatMedia.closeTransient(); },
        stay: () => { leaving = false; },
        // Full teardown (P3): release every resource this instance acquired —
        // ws subscriptions, window/document listeners, the ResizeObserver, all
        // timers — then drop the buffered collections and remove the DOM last.
        // Idempotent; late rAF/async continuations no-op on `destroyed`.
        destroy() {
            if (destroyed) return;
            destroyed = true;
            composer.destroy();
            replies.destroy();
            rowMenu.destroy();
            rowSleep.destroy();
            offRefit();
            unconfirmed.release();
            emptyWelcome?.dispose();
            readReceipt.cancel();
            for (const dispose of wsDisposers) {
                try { dispose(); } catch {}
            }
            wsDisposers.length = 0;
            historyControls.endRecent();
            historyControls.destroy();
            window.removeEventListener('ouro:language-changed', handleHistoryLanguage);
            historyPager.destroy();
            disposeReadingGestures();
            document.removeEventListener('selectionchange', retryHistoricalUpserts);
            chatMedia.destroy();
            roomWork?.destroy();
            chatDecision.destroy();
            modelWaits.destroy();
            mainWaiting?.destroy();
            window.removeEventListener('ouro:page-shown', handlePageShown);
            document.removeEventListener('visibilitychange', handlePageShown);
            header?.destroy();
            chatResizeObserver?.disconnect();
            chatResizeObserver = null;
            historyResyncScheduler.cancel();
            if (headerControlInterval) { clearInterval(headerControlInterval); headerControlInterval = null; }
            for (const id of liveCardRecords.keys()) disposeLiveCard(id);
            explicitCardExpansion.clear();
            reviewDisclosureByTask.clear();
            reviewHydrator.clear();
            pendingSuggestedNames.clear();
            subagentChildParents.clear();
            subagentTerminalChildren.clear();
            cancelableTaskIds.clear();
            missingManagedTaskIds.clear();
            managedTaskDetailReads.clear();
            pendingUserBubbles.clear();
            localEchoJournal.clear();
            seenMessageKeys.clear();
            messageKeyOrder.length = 0;
            persistedHistory.length = 0;
            pageHistoryIds.clear(); recentCoverage = null; historicalTerminals.clear(); pendingHistoryEvictions.clear();
            pendingHistoryUpserts.clear();
            pendingLiveEvictions.clear();
            try { destroyChatMarkdown(page); page.remove(); } catch {}
        },
    };
}
