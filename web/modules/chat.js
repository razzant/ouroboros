import { escapeHtmlAttr, escapeHtmlText as escapeHtml } from './utils.js';
import { bindMarkdownTables, destroyChatMarkdown, enhanceChatMarkdown, mountChatMarkdown, renderChatMarkdown } from './chat_markdown.js';
import { renderPageHeader } from './page_header.js';
import { PAGE_ICONS } from './page_icons.js';
import { showToast } from './toast.js';
import { decorateProjectRow, syncSavedProjectContext } from './project_answer.js';
import { createProjectHandoffs, receiptNotice } from './project_handoff.js';
import { bindComposerFileTargets, cleanupUploadedAttachments, createChatMedia, showTaskIncidentToast } from './chat_media.js';
import { createChatDecision } from './chat_decision.js';
import { bindProjectWorkPointer } from './project_work_pointer.js';
import { createModelWaitController, isModelWaitReference } from './model_wait.js';
import { clientSurfaceField } from './client_surface.js';
import { syncResultFilesItem } from './result_files.js';
import { createChatReadingPosition } from './chat_reading_position.js';
import { createChatHistoryPager, historyCoverage, historyIslandAtEdge } from './chat_history.js';
import { createProjectReadReceipt, isAtNewestMessage } from './project_read_state.js';
import { mergeHistoricalTimelineItem, historyNodeIsProtected, historyRowIds, historyStamps, stampHistoryNode, compareHistoryPosition } from './chat_history_replay.js';
import { apiClient, apiFetch, fetchTaskDetail, fetchTaskDetailStrict } from './api_client.js';
import { syncHistoryRetentionItem } from './history_retention.js';
import {
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
    ACTION_HURRY,
    REUSABLE_TASK_IDS,
    ACTION_RESUME,
    TASK_CONTROL_TRIGGER_LABEL,
    cancelRunEligibility,
    hurryTaskAction,
    openTaskControlMenu,
    requestStop,
    resumeTaskAction,
    taskControlBusy,
} from './task_control_menu.js';
import { openConfirmDialog } from './confirm_dialog.js';
import { bindEnterSubmit } from './ui_interactions.js';
import { mountEmptyChatWelcome } from './welcome_preference.js';
import {
    captureLiveCardPhaseState,
    desiredLiveCardPhase,
    replayTerminalPhase,
    restoreLiveCardPhaseState,
    setLiveCardPhase,
    setLiveCardTypingVisible,
    setHistoricalUnavailable,
    setHistoricalUnconfirmed,
} from './task_phase_chip.js';
import {
    loadSkillReviewDetail,
    nestedSkillReviewRef,
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
    syncLiveCardToggle,
    updateLiveTimelineItem,
    upsertToolFoldRow,
} from './chat_render_batch.js';
import {
    COLLAPSED_ACTIVITY_MAX,
    boundActivityPreview,
    buildTimelineItemHtml,
    buildMessageKey,
    chatStatusCounts,
    chatLogThreadAccepts,
    chatMediaMessageKey,
    chatThreadAccepts,
    clearStickyCardState,
    clearTransientRoutingAnnotations,
    confirmAndSendPanic,
    computeDerivedChatStatus,
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
    loadChatInputHistory,
    markIngressSaved,
    liveLineRowToggleKey,
    bindContentButton,
    bindLiveCardTimeline,
    subagentIdentityTitle,
    subagentTwin,
    mergeStickyCostMeta,
    applyToolObservation,
    noteToolHostMetrics,
    partitionLocalEchoJournal,
    projectCollapsedActivity,
    positiveTaskTerminalFact,
    projectIdFromTask,
    rawTimestampEpoch,
    stampNodeTimestamp,
    reconcileHydratedDirectActivities,
    reconnectBannerText,
    saveChatInputHistory,
    senderLabel,
    shouldFirePanic,
    supervisorReady,
    taskCostMeta,
    taskCostProjection,
    unconfirmedForegroundCardIds,
    withTaskCostMeta,
    applyHistoricalModelExecution,
    cardMetaKeys,
    renderCollapsedActivity,
    renderLiveCardMeta as renderCardMeta,
    ensureLiveActionsEl,
    cardRowEvidenceRef,
    cardRowSummary,
    evidenceLinkHtml,
} from './chat_activity.js';

export {
    COLLAPSED_ACTIVITY_MAX,
    boundActivityPreview,
    chatMediaMessageKey,
    clearStickyCardState,
    clearTransientRoutingAnnotations,
    confirmAndSendPanic,
    computeDerivedChatStatus,
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
    shouldFirePanic,
};

const PROJECT_ROW_TYPES = new Set(['project_started', 'project_handoff', 'project_completion_summary']);
// HOST placement: custody warns; settled reviews read as results.
const CARD_ROW_PHASES = new Map([['timeline', 'warn'], ['reviews', 'result']]);
const CHAT_STORAGE_KEY = 'ouro_chat';
const CHAT_DRAFT_KEY = 'ouro_chat_draft';
const CHAT_INPUT_HISTORY_KEY = 'ouro_chat_input_history';
const ATTACHMENT_PREVIEW_COUNT = 25;

export function initChat(ctx) {
    // Back-compat main-chat entry: one full-page instance bound to chat 1.
    return createChatInstance(ctx);
}

const taskKey = (value) => String(value || '').trim();

export function createChatInstance({
    ws, state, updateUnreadBadge, openSettingsTab, openDashboardTab,
    stateSnapshots,
    chatId = 1, projectId = '', idPrefix = 'chat', mountEl = null,
    asPanel = false, title = 'Chat', initialScrollState = null,
    // app.js signal "a project panel is opening right now" — Main
    // defers its first hydration to it (bounded by an unconditional deadline).
    isProjectOpening = null,
    onHistoryRetry = null,
    onReadingLatest,
}) {
    const container = mountEl || document.getElementById('content');
    const chatSessionId = getOrCreateChatSessionId(sessionStorage, globalThis.crypto);
    const isMain = chatId === 1;
    // Per-thread storage so a project thread never bleeds into the main chat.
    const storeKey = (base) => (isMain ? base : `${base}:${chatId}`);

    const page = document.createElement('div');
    page.id = asPanel ? `panel-${idPrefix}` : 'page-chat';
    page.className = asPanel ? 'chat-instance-panel' : 'page active';
    // A project panel reuses the lean `.project-panel-bar` (title + close) from
    // index.html, so it renders a minimal status-only header — NOT the overlay
    // page header (that would duplicate the title and drag in the GLOBAL
    // Evolve/Restart/Panic/budget chrome, which belongs to the one agent, not a
    // single project thread). The main chat keeps the full overlay header.
    const headerHtml = asPanel
        ? `<div class="chat-panel-statusbar"><span id="chat-status" class="status-badge offline">Connecting...</span></div>`
        : renderPageHeader({
            title: title,
            icon: PAGE_ICONS.chat,
            variant: 'overlay',
            className: 'chat-page-header',
            actionsHtml: `
                <div class="chat-header-actions" id="chat-header-actions">
                    <button class="chat-header-btn" type="button" data-chat-command="restart" title="Restart agent">Restart</button>
                    <button class="chat-header-btn danger" type="button" data-chat-command="panic" title="Stop all workers">Panic</button>
                    <details class="chat-header-more">
                        <summary class="chat-header-btn" title="More agent controls">More</summary>
                        <div class="chat-header-menu">
                            <button class="chat-header-menu-item" type="button" data-chat-command="bg" title="Toggle background consciousness">Consciousness</button>
                            <button class="chat-header-menu-item" type="button" data-chat-command="evolve" title="Toggle evolution mode">Evolve</button>
                            <button class="chat-header-menu-item" type="button" data-chat-command="review" title="Run review now">Review</button>
                        </div>
                    </details>
                </div>
                <button class="chat-budget-pill" id="chat-budget-pill" type="button" title="Open budget controls" aria-label="Open budget controls">
                    <span class="chat-budget-text" id="chat-budget-text">Loading…</span>
                    <div class="chat-budget-bar">
                        <div class="chat-budget-bar-fill" id="chat-budget-bar-fill"></div>
                    </div>
                </button>
                <span id="chat-status" class="status-badge offline">Connecting...</span>
            `,
        });
    page.innerHTML = `
        ${headerHtml}
        <div id="chat-messages"></div>
        <div id="chat-input-area">
            <div class="chat-input-wrap">
                <button class="chat-scroll-bottom-btn" id="chat-scroll-bottom" type="button" aria-label="Scroll to latest message" title="Scroll to latest message">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 5v14"/><path d="M19 12l-7 7-7-7"/></svg>
                    <span class="chat-scroll-activity-dot" aria-hidden="true" hidden></span>
                </button>
                <div id="chat-attachment-preview" class="chat-attachment-preview"></div>
                <div class="chat-toolbar-row">
                    <div class="chat-composer-pills" id="chat-composer-pills">
                        <button class="chat-swarm" id="chat-swarm" type="button" data-armed="false" title="Swarm: route your next message into a new managed task, run a deep plan review with plan_task, then delegate when parallel work helps. Auto-disarms after sending.">Swarm</button>
                        <div class="chat-context-mode" id="chat-context-mode" data-context-mode="max" role="group" aria-label="Context size mode" title="Context mode (owner setting). Nano targets 85K tokens, Low 250K, Max the model window; reply room is included. Saves immediately; lowering requires Ouroboros to be idle."><button class="chat-seg" type="button" data-mode="nano">Nano</button><button class="chat-seg" type="button" data-mode="low">Low</button><button class="chat-seg" type="button" data-mode="max">Max</button></div>
                    </div>
                </div>
                <div class="chat-text-row">
                    <button class="chat-attach-btn" id="chat-attach" type="button" title="Attach file">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66l-9.2 9.19a2 2 0 0 1-2.83-2.83l8.49-8.48"/></svg>
                    </button>
                    <input type="file" id="chat-file-input" class="chat-file-input-hidden" accept="*/*" multiple>
                    <textarea id="chat-input" placeholder="Message Ouroboros..." rows="1" autocorrect="off" autocapitalize="off" spellcheck="false"></textarea>
                    <div class="chat-send-group">
                        <button class="chat-send-inline" id="chat-send" title="Send message">Send</button>
                    </div>
                </div>
            </div>
        </div>
    `;
    if (idPrefix !== 'chat') {
        // Instance-namespaced ids + mirror classes so the shared #chat-* CSS
        // (extended with .chat-* twins) keeps styling secondary instances.
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
    const sendBtn = byId('send');
    const statusBadge = byId('status');
    const headerActions = byId('header-actions');
    const pageHeader = page.querySelector('.chat-page-header');
    const budgetPill = byId('budget-pill');
    const attachBtn = byId('attach');
    const fileInput = byId('file-input');
    const attachmentPreview = byId('attachment-preview');
    const scrollBottomBtn = byId('scroll-bottom');
    const scrollActivityDot = scrollBottomBtn?.querySelector('.chat-scroll-activity-dot');
    let pendingAttachments = [];
    let attachmentsUploading = false;
    let nestedSubagentsExpanded = false;
    let _remoteActivityDepth = 0;

    // Instance lifecycle (P3): destroy() flips this so rAF loops and late async
    // continuations become no-ops instead of touching a removed DOM subtree.
    let destroyed = false;
    // Every ws.on subscription's disposer, released together in destroy().
    const wsDisposers = [];
    const onWs = (event, fn) => wsDisposers.push(ws.on(event, fn));
    const chatMedia = createChatMedia({
        chatSessionId,
        durableChatMediaUrl,
        formatMsgTime,
        insertMessageNode,
        senderLabel,
        stampNodeTimestamp,
    });
    const chatDecision = createChatDecision({
        apiFetch,
        frameNode: chatMedia.bubbleFrameNode,
        mountMarkdown: mountChatMarkdown,
        enhanceMarkdown: enhanceMountedMarkdown,
        showToast,
        fetchDetail: fetchTaskDetailStrict,
        onDomWrite: withStableViewport,
        isMain, chatId,
        insertMessageNode,
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

    function updateAttachmentPreview() {
        if (!pendingAttachments.length) {
            attachmentPreview.classList.remove('visible');
            attachmentPreview.innerHTML = '';
            requestAnimationFrame(() => updateMessagesPadding());
            return;
        }
        attachmentPreview.classList.add('visible');
        attachmentPreview.innerHTML = pendingAttachments.map((item) => `
            <span class="attach-badge" data-attachment-id="${escapeHtmlAttr(item.id)}">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/><polyline points="13 2 13 9 20 9"/></svg>
                <span class="attach-name" title="${escapeHtmlAttr(item.display_name)}">${escapeHtml(item.display_name)}</span>
                <button class="attach-remove" type="button" title="Remove" aria-label="Remove attachment ${escapeHtmlAttr(item.display_name)}" data-attachment-remove="${escapeHtmlAttr(item.id)}" ${attachmentsUploading ? 'disabled aria-disabled="true"' : ''}>×</button>
            </span>
        `).join('');
        requestAnimationFrame(() => updateMessagesPadding());
        attachmentPreview.querySelectorAll('[data-attachment-remove]').forEach((button) => {
            button.addEventListener('click', () => {
                if (attachmentsUploading) return;
                const removeId = button.getAttribute('data-attachment-remove') || '';
                pendingAttachments = pendingAttachments.filter((item) => item.id !== removeId);
                updateAttachmentPreview();
            });
        });
    }

    // Shared paperclip/paste stager; upload still happens only on Send.
    function stagePendingFiles(files) {
        const incoming = Array.from(files || []).filter(Boolean);
        if (!incoming.length) return;
        if (attachmentsUploading) {
            showToast('Wait for the current upload to finish before changing attachments.', 'error');
            return;
        }
        pendingAttachments = pendingAttachments.concat(incoming.map((file) => ({
            id: (globalThis.crypto && typeof crypto.randomUUID === 'function')
                ? crypto.randomUUID()
                : `attachment-${Date.now()}-${Math.random().toString(16).slice(2)}`,
            file,
            display_name: file.name || 'upload',
        })));
        updateAttachmentPreview();
    }

    function setAttachmentUploadState(uploading) {
        attachmentsUploading = uploading;
        attachBtn.disabled = uploading;
        attachBtn.classList.toggle('uploading', uploading);
        fileInput.disabled = uploading;
        input.disabled = uploading;
        updateAttachmentPreview();
    }

    attachBtn.addEventListener('click', () => fileInput.click());

    // Local-only staging avoids orphan uploads and fast-send races.
    fileInput.addEventListener('change', () => {
        const files = Array.from(fileInput.files || []);
        fileInput.value = '';
        stagePendingFiles(files);
    });

    bindComposerFileTargets({ page, inputArea, input, stagePendingFiles });

    let _syncPass1Active = false;
    let _historyReplayActive = false;
    let _historyRow = null;
    let _historyAppending = false;

    const persistedHistory = [];
    const seenMessageKeys = new Set();
    const messageKeyOrder = [];
    const pendingUserBubbles = new Map();
    const inputHistory = loadChatInputHistory(sessionStorage, CHAT_INPUT_HISTORY_KEY);
    let inputHistoryIndex = inputHistory.length;
    let inputDraft = '';
    let historyLoaded = false;
    let inputHistorySeededFromServer = false; // set true only after a successful server-side recall seed
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
    // Saved page and whole recent read gate only the cross-instance place; reshow
    // targets and visible mutations use live geometry.
    let restoredPageReady = !initialScrollState?.history;
    let recentReady = false;
    let _hasNewActivity = false;
    const isInstanceVisible = () =>
        Boolean(messagesDiv) && messagesDiv.offsetParent !== null && !document.hidden;
    const LIVE_CARD_CAP = 200;
    const liveCardBound = createLiveCardBound(LIVE_CARD_CAP);
    const liveCardRecords = new Map();
    const workPointer = asPanel ? bindProjectWorkPointer(page.querySelector('.chat-panel-statusbar'), {
        records: liveCardRecords,
        onNavigate: (root) => {
            reading.cancel();
            messagesDiv.scrollTop += root.getBoundingClientRect().top - messagesDiv.getBoundingClientRect().top;
            reading.stick = false;
            reading.scroll();
        },
    }) : null;
    // A wait materializes the real task record; its open/closed state is one
    // input of the block predicate, so the block appears with the controls and
    // leaves with them (the record and its no-reopen wait ledger stay).
    const modelWaits = createModelWaitController({
        getRecord: (id, create = true) => (create ? getLiveCardRecord(id) : liveCardRecords.get(id)),
        openSettings: openSettingsTab, onDomWrite: withStableViewport,
        onChange: (id) => { ensureLiveCardVisible(liveCardRecords.get(id)); syncChatStatus(); },
    });
    function disposeLiveCard(id, preserveWaits = false) {
        if (!preserveWaits) modelWaits.forget(id);
        liveCardRecords.get(id)?.timelineDispose?.();
        liveCardRecords.get(id)?.root?.remove();
        liveCardRecords.delete(id);
    }
    const markReviewAnchor = (r, on = false) => setReviewAnchor(r, on, setLiveCardPhase);
    const explicitCardExpansion = new Map(initialScrollState?.disclosures?.cards || []);
    const reviewDisclosureByTask = new Map((initialScrollState?.disclosures?.reviews || []).map(([id, value]) =>
        [id, { ...value, expandedGroups: new Set(value.expandedGroups), expandedAttempts: new Set(value.expandedAttempts) }]));
    const skillReviewDetailStore = new Map();
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
    // В9: Starting… until a snapshot says supervisor_ready; a disconnect forgets it.
    let hostReady = false;
    // Bounded conclusions block stale state snapshots; reusable logical task
    // slots are cleared whenever their cycle settles.
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

    function recordConcludedActivity(activityId) {
        const aid = taskKey(activityId);
        if (!aid) return;
        missingManagedTaskIds.delete(aid);
        concludedDirectActivities.delete(aid);
        concludedDirectActivities.set(aid, Date.now());
        while (concludedDirectActivities.size > CONCLUDED_ACTIVITY_LEDGER_MAX) {
            concludedDirectActivities.delete(concludedDirectActivities.keys().next().value);
        }
    }
    function recordTerminalActivity(taskId) {
        const id = taskKey(taskId);
        if (!id) return;
        activeDirectActivities.delete(id);
        missingManagedTaskIds.delete(id);
        if (REUSABLE_TASK_IDS.has(id)) concludedDirectActivities.delete(id);
        else recordConcludedActivity(id);
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
    let activeLiveGroupId = '';
    let pendingReconnectSync = false;  // Set when a fromReconnect sync arrives while one is already in-flight.
    let pendingReconnectBannerText = '';
    try {
        pendingReconnectBannerText = reconnectBannerText(new URL(window.location.href).searchParams.get('_ouro_reason') || '');
    } catch {}
    function clearPendingReconnectBanner() {
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

    function setStatus(kind, text) {
        // replay frames never touch the badge; the reducer
        // (syncChatStatus) writes it once after the batch.
        if (!statusBadge) return;
        statusBadge.className = `status-badge ${kind}`;
        statusBadge.textContent = text;
    }

    function syncHeaderControlState(data) {
        headerActions?.querySelectorAll('[data-chat-command]').forEach((button) => {
            const cmd = button.dataset.chatCommand;
            const state = cmd === 'evolve' ? [data?.evolution_enabled, data?.evolution_state?.detail]
                : cmd === 'bg' ? [data?.bg_consciousness_enabled, data?.bg_consciousness_state?.detail] : null;
            if (state) {
                const unknown = state[0] == null;
                button.classList.toggle('on', state[0] === true);
                button.dataset.tone = unknown ? 'warn' : '';
                button.textContent = `${cmd === 'bg' ? 'Consciousness' : 'Evolve'}${unknown ? ' · unknown' : ''}`;
                button.title = state[1] || (unknown ? 'State unknown' : 'Toggle mode');
            }
        });
        // More reflects active background mode.
        const moreSummary = headerActions?.querySelector('.chat-header-more > summary');
        if (moreSummary) {
            const anyActive = !!data?.evolution_enabled || !!data?.bg_consciousness_enabled;
            moreSummary.classList.toggle('has-active', anyActive);
        }
        const ctxBtn = byId('context-mode');
        if (ctxBtn && typeof data?.context_mode === 'string') {
            ctxBtn.dataset.contextMode = ['nano', 'low', 'max'].includes(data.context_mode) ? data.context_mode : 'max';
        }
        const budget = headerBudgetPresentation(data);
        const budgetText = byId('budget-text');
        const budgetFill = byId('budget-bar-fill');
        if (budgetText) budgetText.textContent = budget.label;
        if (budgetFill) budgetFill.style.width = `${budget.fillPct}%`;
    }

    // Main-only: transfer receipts are Main history rows.
    const handoffs = isMain ? createProjectHandoffs({ feed: messagesDiv, fetchDetail: fetchTaskDetailStrict, mutate: withStableViewport }) : null;

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
            const tasks = new Map([...queue.pending, ...queue.running].map(row => [row.id, row.task]));
            withStableViewport(() => {
                for (const record of children) {
                    if (record.finished || !record.root?.isConnected || record.lastLiveObservedAt > started) continue;
                    const task = tasks.get(record.groupId);
                    if (task && 'project_admission_hold' in task) {
                        missingManagedTaskIds.delete(record.groupId);
                        restoreCardActivity(record, task.project_admission_hold);
                        record.projectHoldQueued = Boolean(record.projectHold);
                    } else if (!task && record.projectHoldQueued) {
                        record.projectHoldQueued = false;
                        missingManagedTaskIds.add(record.groupId);
                        void reconcileMissingManagedTask(record.groupId);
                    }
                }
            });
            syncChatStatus();
        } catch { /* Keep the last fact; absence or a failed read proves no recovery. */ }
        finally { childHoldRead = false; }
    }

    function hydrateStateSnapshot(data, snapshotRequestedAt = Infinity) {
        void refreshChildProjectHolds();
        syncHeaderControlState(data);
        handoffs?.snapshot(data);
        hostReady = supervisorReady(data) ?? hostReady;
        const activities = Array.isArray(data?.active_chat_activities)
            ? data.active_chat_activities
            : data?.active_direct_turns;
        if (Array.isArray(activities)) {
            hydrateDirectActivities(activities, snapshotRequestedAt,
                data.active_chat_activities_complete === true && data.supervisor_ready === true);
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
            const resp = await apiFetch('/api/state', { cache: 'no-store' });
            if (!resp.ok) throw new Error(`State read failed: HTTP ${resp.status}`);
            const data = await resp.json();
            stateSnapshots.apply(request, data);
        } catch {
            if (stateSnapshots.isCurrent(request)) syncHeaderControlState({ accounting: { available: false } });
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
    const reading = createChatReadingPosition({
        initial: initialScrollState, feed: messagesDiv, alive: () => !destroyed,
        afterWrite: () => workPointer?.update(), activity: () => { _hasNewActivity = true; },
        updateButton: updateScrollButton,
        visible: () => !destroyed && isInstanceVisible(),
        ready: () => historyLoaded && (!reading.restoring || restoredPageReady && recentReady)
            && anchorOwnersReady(reading.target?.historyAnchor, reviewHydrator.ready),
        anchors: { serialize: serializeTimelineAnchor, restore: restoreVisibleTimelineAnchor, capture: captureVisibleTimelineAnchor },
        fallback: target => {
            const ids = pageHistoryIds.get(target.history?.pages?.[target.history?.focus]?.id);
            const node = ids && [...ids].flatMap(historyNodes).find(node => node.getClientRects().length);
            return node ? restoreVisibleTimelineAnchor({ node, offset: target.historyAnchor?.offset || 0 }) : false;
        },
        // A settled place is where the reader is (the read receipt).
        changed: () => { syncLoadOlderControl(); updateScrollButton(); readReceipt.note({ discrete: true }); },
    });

    function withStableViewport(mutate, options) { return reading.mutate(mutate, options); }

    function withRemoteActivity(mutate) {
        _remoteActivityDepth += 1;
        try {
            return withStableViewport(mutate, { remoteContent: true });
        } finally {
            _remoteActivityDepth -= 1;
        }
    }

    function enhanceMountedMarkdown(root) {
        return enhanceChatMarkdown(root, {
            onDomWrite: _remoteActivityDepth > 0 ? withRemoteActivity : withStableViewport,
            onThemeDomWrite: withStableViewport,
        });
    }
    const {
        renderLiveCardTimeline,
        appendTimelineItem,
        patchLastTimelineItem,
        patchTimelineItemAt,
    } = createLiveCardTimelineRenderer({
        withStableViewport, buildTimelineItemHtml, isReplayActive: () => _historyReplayActive,
        initialAnchor: initialScrollState?.historyAnchor, hydrate: fetchFullLineOutput,
    });

    function insertMessageNode(node, options = {}) {
        if (!node) return false;
        const isMounted = node.parentNode === messagesDiv;
        if (isMounted && !options.reorderExisting) {
            return false;
        }
        return withStableViewport(() => {
            // Scope to THIS instance's column — a global id lookup would resolve to
            // the first panel's typing node and misplace project-thread messages.
            const typing = messagesDiv.querySelector('.typing-bubble');
            insertTimelineNode(messagesDiv, node, typing);
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

    // The ONE rule for a task's block being in the transcript (docs/DESIGN.md
    // "Conversation activity block"): facts the record holds, re-read at every
    // mutation, no sticky flag; the completion note and receipt rows are not content.
    function blockVisible(record) {
        if (!record || record.isSubagent) return true;
        return blockHasWork(record)
            || !!(record.modelWaiting || record.cancelPendingPolicy || record.reviewAnchor)
            || stopEligible(record)
            || (record.finished && record.phaseEl?.dataset?.phase !== 'done');
    }

    // The host's lane fact (census kind, rebuilt task_done, history rows) is
    // kept on the record for the header pill only: a direct turn keeps the
    // census verdict (Thinking…) beside its block. It never chooses chrome.
    function noteDirectTurn(record, direct) {
        if (record && typeof direct === 'boolean') record.direct = direct;
    }

    // The work the block stands on — the presence facts minus open attention and
    // minus a bare terminal outcome. It selects the chrome (owner decision 16.09):
    // a block with work is the task card whatever lane produced it (a title, the
    // conversion control in Main); a block that exists only for open attention or
    // a non-Done ending keeps no title placeholder and offers no conversion. The
    // host's lane fact (`_is_direct_chat`) keeps its host jobs and never chooses chrome.
    function blockHasWork(record) {
        const id = record.groupId;
        // Visibility depends on work, never the lane.
        return record.reviewController?.groups.size > 0
            || [...subagentChildParents.values()].some((info) => info.parentId === id)
            || record.items.some((item) => !item.receipt && !String(item.dedupeKey || '').startsWith('task_done|'))
            || record.toolErrors > 0;
    }

    // Fold host facts; absent counters stay absent.
    function noteToolMetrics(taskId, metrics, rawTs, { suppressDomInsert = false } = {}) {
        const [calls, errors, routing, completion] = ['tool_calls', 'tool_errors', 'routing_tool_calls', 'completion_tool_calls']
            .map(key => Number.isInteger(metrics?.[key]) ? metrics[key] : null);
        if (!calls && !errors && !metrics.tool_evidence?.observations?.length && !metrics.tool_evidence?.legacy?.calls) return false;
        return withStableViewport(() => {
            const record = getLiveCardRecord(taskId);
            const before = captureLiveCardProjection(record);
            const duration = Number(metrics.duration_sec);
            if (Number.isFinite(duration)) record.durationSec = duration;
            const summary = noteToolHostMetrics(record, { calls, errors, routing, completion, counts: metrics.tool_call_counts, evidence: metrics.tool_evidence });
            record.toolCalls = summary.calls;
            record.toolErrors = summary.errors;
            const { timelineUpdate } = upsertToolFoldRow(record, summary, normalizeLogTs(rawTs), rawTs);
            const changed = !['none', 'duplicate-skip'].includes(timelineUpdate) && renderLiveCardTimeline(record);
            updateLiveCardCount(record);
            renderLiveCardMeta(record);
            reanchorTaskCard(record, rawTs, { suppressDomInsert });
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
        return withStableViewport(() => queueTaskLiveUpdateMutation(
            summary, taskId, ts, dedupeKey, rawTs,
        ));
    }

    function queueTaskLiveUpdateMutation(summary, taskId, ts, dedupeKey = '', rawTs = '') {
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
    }

    async function turnTaskIntoProject(record) {
        if (!record || record.root?.dataset?.projectCreating === '1' || record.root?.dataset?.projectCreated === '1') return;
        const taskId = taskKey(record.groupId);
        const projectId = projectIdFromTask(taskId);
        record.root.dataset.projectCreating = '1';
        // The control itself reports the in-flight conversion; Stop stays where
        // it was, because the task keeps running whatever the bind answers.
        const btn = record.turnProjectBtn;
        if (btn) { btn.disabled = true; btn.textContent = 'Creating project…'; }
        try {
            // One-click convert (owner P1): no name prompt, no extra LLM call.
            // The SERVER names the project (gateway/projects.py: explicit
            // title, coined name, then the task's own origin text) and adopts
            // the project the task's owner message already has.
            const payload = await apiClient.projectFromTask(taskId, projectId, '');
            const project = payload?.project;
            if (!project?.id || payload?.binding?.project_id !== project.id) {
                throw new Error('Project binding response is unconfirmed; check the Project list before retrying.');
            }
            showToast(`${payload.adopted ? 'Project opened' : 'Project created'}: ${project.name || project.id}`, 'ok');
            window.dispatchEvent(new CustomEvent('ouro:project-created', { detail: { project } }));
            markCardConverted(record, payload);
            const notice = receiptNotice(payload.handoff_receipt);
            if (notice) showToast(notice, 'warn');
        } catch (exc) {
            showToast(`Project conversion not confirmed: ${exc.message || exc}`, 'error');
            delete record.root.dataset.projectCreating;
            if (btn) { btn.disabled = false; btn.textContent = 'Turn into project'; }
        }
    }

    // Main blocks with work offer conversion unless their origin is bound
    // (the same /api/state fact app.js reads); titles use the same work predicate.
    function syncBlockChrome(record) {
        if (record.root.dataset.projectCreated === '1') return;
        // A child is never a convertible unit (it inherits its root's Project by
        // lineage) and its title is its role: both root writers skip it.
        const work = !record.isSubagent && blockHasWork(record);
        // The first row of work lands after the title writers ran for its frame: an empty title
        // takes a name here (Working is the chip's word, #1369); the writers own it from then on.
        if (work && !record.titleEl.textContent) {
            record.titleEl.textContent = record.suggestedName || record.lastHumanHeadline
                || (record.finished ? 'Task activity' : '');
        }
        const wanted = isMain && work && record.root.dataset.projectBound !== '1'
            && !(window.__ouroTaskBindings || {})[record.groupId];
        if (wanted === Boolean(record.turnProjectBtn)) return;
        if (!wanted) {
            record.turnProjectBtn.remove();
            record.turnProjectBtn = null;
            return;
        }
        const actions = ensureLiveActionsEl(record);
        if (!actions) return;
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'btn btn-xs btn-default';
        btn.dataset.turnIntoProject = '1';
        btn.textContent = 'Turn into project';
        btn.addEventListener('click', (event) => {
            event.stopPropagation();
            turnTaskIntoProject(record);
        });
        actions.prepend(btn);
        record.turnProjectBtn = btn;
    }

    function syncCancelRunButton(record) {
        return withStableViewport(() => syncCancelRunButtonMutation(record));
    }

    // The one reading of "this record offers Stop": the block predicate and the
    // control share it, so a block never stands on a Stop it hides.
    const stopEligible = (record) => cancelRunEligibility({
        groupId: record.groupId, isSubagent: record.isSubagent, finished: record.finished,
        cancelable: !record.historicalUnavailable && !record.historicalUnconfirmed && cancelableTaskIds.has(record.groupId),
        converted: record.root.dataset.projectCreated === '1',
    });

    function syncCancelRunButtonMutation(record) {
        if (!record?.root) return false;
        const eligible = stopEligible(record);
        const existing = record.root.querySelector('[data-cancel-run]');
        if (!eligible) {
            if (!existing) return false;
            existing.remove();
            record.cancelRunBtn = null;
            return true;
        }
        if (existing) {
            record.cancelRunBtn = existing;
            return false;
        }
        const actions = ensureLiveActionsEl(record);
        if (!actions) return false;
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'btn btn-xs btn-danger';
        btn.dataset.cancelRun = '1';
        btn.textContent = TASK_CONTROL_TRIGGER_LABEL;
        // Pending cancel offers only hard escalation; dismiss keeps the run.
        btn.addEventListener('click', (event) => {
            event.stopPropagation();
            openTaskControlMenu(btn, {
                cancelPending: Boolean(record.cancelPendingPolicy),
                budgetPaused: activeDirectActivities.get(record.groupId)?.phase === 'budget_paused',
                busy: taskControlBusy(record.groupId),
                onAction: (action) => {
                    if (action === ACTION_HURRY) return hurryTaskAction(record.groupId);
                    if (action === ACTION_RESUME) return resumeTaskAction(record.groupId);
                    return cancelRunFromCard(record, action);
                },
            });
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
            showToast(`Cancel failed: ${exc?.message || exc}`, 'error');
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

    function markTaskCancelable(taskId = '') {
        const id = taskKey(taskId);
        if (!id) return false;
        cancelableTaskIds.add(id);
        return syncCancelRunButton(liveCardRecords.get(id));
    }

    // Conversion replaces the whole Main card with a Project pointer; the
    // bound task lives in that panel. The pointer opens it and uses Project ink.
    function markCardConverted(record, { project, handoff_id, handoff_receipt }) {
        return withStableViewport(() => {
            modelWaits.forget(record.groupId);
            delete record.root.dataset.projectCreating;
            record.root.dataset.projectCreated = '1';
            record.root.dataset.projectId = project.id || '';
            // Atomic detach-and-reparent (C4.5): replaceChildren swaps the whole live
            // timeline (subagent cards, working bubble) for the chip in one paint.
            handoffs?.mount(record.root, { taskId: record.groupId, projectId: project.id, projectName: project.name,
                title: record.titleEl?.textContent, handoffId: handoff_id, receipt: handoff_receipt, kind: 'card' });
            record.turnProjectBtn = null;
            record.cancelRunBtn = null;
            record.finished = true;
            // Recolor on the next frame so the 250ms fuchsia fade actually animates.
            requestAnimationFrame(() => record.root.classList.add('is-project'));
            signalChatFreed();  // subtle "this chat is free again" composer cue
        });
    }

    // A brief composer brighten when a task leaves the main chat for a project —
    // a calm "you're free to start something else" signal (P3). The CSS
    // animation owns the duration; the class leaves with it, no timer to dispose.
    function signalChatFreed() {
        const row = page.querySelector('.chat-text-row');
        row?.classList.add('chat-freed');
        row?.addEventListener('animationend', function done(event) {
            if (event.target !== row) return;  // a child's animation is not this cue
            row.classList.remove('chat-freed');
            row.removeEventListener('animationend', done);
        });
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

    // Detail reads refresh files/history, preserving lifecycle.
    function noteTaskDetails(record, detail) {
        const files = syncResultFilesItem(record, detail);
        const history = syncHistoryRetentionItem(record, detail);
        return Boolean((files || history) && (renderLiveCardMeta(record), updateLiveCardCount(record), renderLiveCardTimeline(record), true));
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

    // Host placement/task/card keys own content-only rows; phase/expansion stay.
    // Reviews rehydrate; read failure keeps rows. Unkeyed rows stay bubbles.
    // Pass 2 uses pass-1 cards, with no type allowlist.
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
        if (row.tool_evidence && row.task_id) noteToolMetrics(row.task_id, row, row.ts || row.timestamp || '');
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
        root.innerHTML = `
            <div class="chat-live-summary-button" role="button" tabindex="0" data-live-summary-button aria-expanded="false" aria-controls="${escapeHtmlAttr(timelineId)}">
                <div class="chat-live-summary">
                    <div class="chat-live-summary-main">
                        <span class="chat-live-phase working" data-live-phase role="status" aria-live="polite" aria-atomic="true" aria-label="${options.isSubagent ? 'Subagent' : 'Task'} status: Working">Working</span>
                        <span class="chat-live-phase-secondary" data-live-phase-secondary hidden></span>
                        <div class="chat-live-typing" data-live-typing aria-hidden="true">
                            <span></span><span></span><span></span>
                        </div>
                    </div>
                    <span class="chat-live-title" data-live-title>Waiting for work</span>
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
                <div class="chat-live-review-summary" data-live-review-summary hidden></div>
            </div>
            <div class="chat-live-timeline" data-live-timeline id="${escapeHtmlAttr(timelineId)}"></div>
            <div data-live-reviews-host></div>
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
            // Host fact for the header pill: a direct conversation turn (census
            // kind, rebuilt task_done, history rows) vs a managed/Swarm root.
            direct: String(activeDirectActivities.get(normalizedGroupId)?.kind || 'managed_task') !== 'managed_task',
        };
        const reviewDisclosure = reviewDisclosureByTask.get(normalizedGroupId) || {
            sectionExpanded: false,
            expandedGroups: new Set(),
            expandedAttempts: new Set(),
        };
        reviewDisclosureByTask.set(normalizedGroupId, reviewDisclosure);
        record.reviewController = createReviewPresentationController({
            host: record.reviewsHostEl,
            summary: record.reviewSummaryEl,
            disclosure: reviewDisclosure,
            onHydrate: () => hydrateCardReviews(normalizedGroupId),
            onLoadSkillDetail: (detail, detailOptions = {}) => loadSkillReviewDetail(
                detail,
                nestedSkillReviewRef(detail),
                {
                    store: skillReviewDetailStore,
                    retry: detailOptions.retry === true,
                    onDomWrite: withStableViewport,
                },
            ),
            onDomWrite: withStableViewport,
        });
        bindContentButton(record.summaryButtonEl, () => {
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
            if (record.titleEl) record.titleEl.textContent = _pendingName;
        }
        resetLiveCardRecord(record);
        // P5: the cancelable marker may have arrived (scheduled progress frame /
        // history replay) before this card was minted.
        syncCancelRunButton(record);
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
            // task_named is broadcast to every instance, so bound the early-name buffer.
            pendingSuggestedNames.set(tid, nm);
            if (pendingSuggestedNames.size > 100) {
                pendingSuggestedNames.delete(pendingSuggestedNames.keys().next().value);
            }
            return false;
        }
        if (record.isSubagent) return false;
        const titleChanged = Boolean(record.titleEl && record.titleEl.textContent !== nm);
        let changed = record.suggestedName !== nm || titleChanged;
        record.suggestedName = nm;
        if (titleChanged) record.titleEl.textContent = nm;
        // Restore the deferred collapsed activity after naming.
        changed = renderCollapsedActivity(record, projectCollapsedActivity({
            suggestedName: nm,
            headline: record.collapsedActivity,
            previous: record.collapsedActivity,
        })) || changed;
        return Boolean(changed && record.root?.isConnected);
    }

    function ensureSubagentContainer(parentId = '', depth = 0) {
        if (!parentId) return null;
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

    function getSubagentCardRecord(childId = '', parentId = '', role = '') {
        const existing = liveCardRecords.get(childId);
        const childBefore = captureLiveCardProjection(existing);
        const parentBefore = captureLiveCardProjection(liveCardRecords.get(parentId));
        let record = null;
        withStableViewport(() => {
            record = getSubagentCardRecordMutation(childId, parentId, role);
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
        record.expandedLineKeys.clear();
        record._anchorOrderDirty = false;
        // collapsed timelines defer DOM building; the flag says
        // the rendered timeline DOM is stale relative to record.items.
        record._timelineDirty = false;
        // last frame's summary meta strings — meta renders from
        // record state (renderLiveCardMeta), once per card in a batch.
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
        setLiveCardTypingVisible(record, true);
        const expanded = explicitCardExpansion.has(record.groupId)
            ? explicitCardExpansion.get(record.groupId)
            : Boolean(record.isSubagent && nestedSubagentsExpanded);
        setLiveCardExpanded(record, expanded);
    }

    function ensureLiveCardVisible(
        record,
        { suppressDomInsert = false, reorderExisting = false } = {},
    ) {
        if (!record || suppressDomInsert || _syncPass1Active) return;
        if (record.isSubagent && record.parentGroupId) {
            // The container mints a lineage-known parent nested (#636); the
            // recursion carries the chain up to the inserted root.
            const container = ensureSubagentContainer(record.parentGroupId);
            if (container && record.root.parentNode !== container) {
                container.appendChild(record.root);
            }
            const parentRecord = liveCardRecords.get(record.parentGroupId);
            ensureLiveCardVisible(parentRecord);
            updateLiveCardCount(parentRecord);
            return;
        }
        syncBlockChrome(record);
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
            return Boolean(record.root.isConnected);
        };
        return record?.root?.isConnected ? withStableViewport(mutate) : mutate();
    }

    function directSubagentCount(record) {
        return record?.subagentsEl?.querySelectorAll(':scope > .chat-live-card.subagent').length || 0;
    }

    function updateLiveCardCount(record) {
        if (!record?.countEl) return;
        const bits = [];
        if (record.items.length >= 2) bits.push(`${record.items.length} notes`);
        const children = directSubagentCount(record);
        if (children) bits.push(`${children} ${children === 1 ? 'child' : 'children'}`);
        const hidden = bits.length === 0;
        const text = bits.join(' · ');
        if (record.countEl.hidden !== hidden) record.countEl.hidden = hidden;
        if (record.countEl.textContent !== text) record.countEl.textContent = text;
    }

    // Re-show the same reading intent; data readiness controls positioning.
    const handlePageShown = (event) => {
        if (
            event?.detail?.page === 'chat'
            || (event?.type === 'visibilitychange' && !document.hidden)
        ) reading.request();
    };
    window.addEventListener('ouro:page-shown', handlePageShown);
    document.addEventListener('visibilitychange', handlePageShown);

    // Beyond the 4000-char WS preview: fetch/cache on expansion, bound scrolling,
    // render only while expanded; failure keeps the preview.
    async function fetchFullLineOutput(item, record) {
        item._fetchingFull = true;
        let changed = false;
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
            if (full && item.fetchedFull !== full) {
                item.fetchedFull = full;
                changed = true;
            }
        } catch {
            // best-effort: leave the capped preview on failure
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
        run: () => syncHistory({ includeUser: false }).catch(() => {}).then(() => {
            if (!destroyed && lastHistorySyncSucceeded && liveCardBound.isArmed()) scheduleHistorySync();
        }),
    });

    function applyLiveCardState(summary, groupId, ts, dedupeKey = '', options = {}) {
        return withStableViewport(() => applyLiveCardStateMutation(
            summary, groupId, ts, dedupeKey, options,
        ));
    }

    function applyLiveCardStateMutation(summary, groupId, ts, dedupeKey = '', { suppressDomInsert = false, rawTs = '' } = {}) {
        const nextGroupId = groupId || activeLiveGroupId || 'active';
        const record = getLiveCardRecord(nextGroupId);
        if (record.root?.dataset?.projectCreated === '1') return false;
        const before = captureLiveCardProjection(record);
        const typingBefore = typingEl.style.display;
        let timelineChanged = false;
        const nextPhase = summary.phase || '';
        if (_historyRow?.history_id && record.updates > 0 && !summary.terminal
                && (record.finished || _historyAppending
                    || Date.parse(rawTs) < Date.parse(record.latestSourceTs || ''))) {
            const changed = mergeHistoricalTimelineItem(record, summary, _historyRow, ts);
            reanchorTaskCard(record, rawTs, { suppressDomInsert });
            if (changed) { renderLiveCardTimeline(record); updateLiveCardCount(record); }
            return changed;
        }
        // One tool evidence row per block.
        const foldView = summary.toolCall ? applyToolObservation(record, summary.toolCall) : null;
        if (summary.toolCall?.fact === 'wait_ended' && record.toolFold.calls.get(summary.toolCall.key)?.settlement) {
            summary = { ...summary, visible: false }; // wait history remains in the fold
        }
        if (record.finished && !isTerminalTaskPhase(nextPhase, summary.terminal)) {
            if (foldView) {
                upsertToolFoldRow(record, foldView, ts, rawTs);
                renderLiveCardTimeline(record);
                updateLiveCardCount(record);
            }
            if (summary.modelExecution) record.modelExecution = summary.modelExecution;
            renderLiveCardMeta(record, summary.costProjection);
            return liveCardProjectionChanged(before, record);
        }
        if (summary.terminal || (!_historyReplayActive && !_syncPass1Active
                && ['working', 'thinking'].includes(summary.phase))) {
            restoreCardActivity(record);
            if (!summary.terminal) {
                record.lastLiveObservedAt = Date.now();
                missingManagedTaskIds.delete(nextGroupId);
            }
        }
        markReviewAnchor(record);
        if (foldView?.clearedNotice) { renderLiveCardTimeline(record); timelineChanged = true; }

        if (!record.isSubagent) {
            activeLiveGroupId = nextGroupId;
            reanchorTaskCard(record, rawTs, { suppressDomInsert });
        }
        record.updates += 1;
        const wasFinished = record.finished;
        // No placeholder: a frame naming nothing is no narration; the chip says Working (#1369).
        const headline = summary.headline || record.lastHumanHeadline || '';
        const syntheticKey = summary.dedupeKey || dedupeKey || `${summary.phase || 'working'}|${headline}|${summary.body || ''}`;
        const isLegacyParentSubagentKey = syntheticKey.startsWith('parent-subagent:');
        // Failure/timeout update one row; success feeds the fold.
        const inPlaceByKey = isLegacyParentSubagentKey
            || ['subagent-lifecycle:', 'subagent-progress:', 'subagent-result:', 'task_done|', 'tool:']
                .some((prefix) => syntheticKey.startsWith(prefix));
        if (!isLegacyParentSubagentKey) {
            record.finished = isTerminalTaskPhase(nextPhase, summary.terminal);
        }
        if (summary.human && headline) {
            record.lastHumanHeadline = headline;
        }
        if (summary.model) record.agentModel = summary.model;
        // Origin (including consciousness) stays sticky once observed.
        if (summary.initiator) record.initiator = summary.initiator;

        const shouldPromote = Boolean(summary.promote) || record.finished;
        const activeHeadline = shouldPromote
            ? headline
            : (record.lastHumanHeadline
                || (record.updates > 1 ? record.titleEl.textContent : ''));
        // Only task facts own the outcome chip under a hold; failed tools are diagnostics.
        if (summary.observedOutcome && !record.finished) record.observedOutcome = summary.observedOutcome;
        const desiredPhase = desiredLiveCardPhase(record, record.finished ? summary.phase || 'done' : '');
        setLiveCardPhase(record, desiredPhase.phase, desiredPhase.text, desiredPhase.className,
            desiredPhase.secondary);
        // Title: project name, child's lineage, or activity; an empty block has none.
        // Project naming leaves the activity headline in the timeline.
        const title = record.suggestedName || (record.isSubagent ? childTitle(record)
            : !blockHasWork(record) ? ''
                : (record.finished ? record.lastHumanHeadline || 'Task activity'
                    : record.lastHumanHeadline || activeHeadline));
        if (record.titleEl.textContent !== title) record.titleEl.textContent = title;
        // Collapsed activity comes only from the turn's voice, including child cards;
        // host notes and terminal Done cannot overwrite it. Expand keeps full activity.
        const previewSource = record.isSubagent && summary.human !== false
            ? String(summary.activityPreview ?? summary.body ?? '')
            : (summary.human ? String(summary.activityPreview ?? activeHeadline ?? '')
                : (summary.terminal && summary.activityPreview ? String(summary.activityPreview) : ''));
        const activityCandidate = previewSource.trim();
        if (activityCandidate) record.collapsedActivity = boundActivityPreview(activityCandidate);
        const activityText = projectCollapsedActivity({
            isSubagent: record.isSubagent,
            suggestedName: title,
            headline: record.isSubagent ? '' : record.collapsedActivity,
            body: record.isSubagent ? record.collapsedActivity : '',
            previous: record.collapsedActivity,
        });
        renderCollapsedActivity(record, activityText);

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
        if (ts && (summary.human || activityCandidate)) record.latestActivityTs = ts;
        if (summary.executorChip
                && !keepStickyExecutorChip(record.executorChip, summary.executorChip)) {
            record.executorChip = summary.executorChip;
        }
        if (summary.modelExecution) record.modelExecution = summary.modelExecution;
        if (Number.isInteger(summary.toolCalls)) record.toolCalls = summary.toolCalls;
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
        if (record.finished) {
            settleLiveCard(record, summary.phase || 'done', wasFinished);
        } else {
            setLiveCardTypingVisible(record, true);
        }
        syncChatStatus();
        return Boolean(timelineChanged
            || typingBefore !== typingEl.style.display
            || liveCardProjectionChanged(before, record));
    }

    function finishLiveCard(groupId = '', phase = '') {
        return withStableViewport(() => finishLiveCardMutation(groupId, phase));
    }

    // Author controls end; paid-review waits retain their own lifetime.
    function settleLiveCard(record, phase, wasFinished) {
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
        // Converted project chips ignore later task terminals (T4).
        if (record.root?.dataset?.projectCreated === '1') return false;
        const before = captureLiveCardProjection(record);
        const typingBefore = typingEl.style.display;
        if (setHistoricalUnavailable(record, false)) renderLiveCardMeta(record);
        markReviewAnchor(record);
        const wasFinished = record.finished;
        record.finished = true;
        record.finalizingHold = false;
        const presentation = taskPresentation(phase || 'done');
        const activePhase = presentation.phase;
        setLiveCardPhase(record, activePhase, presentation.headline);
        if (record.isSubagent) record.titleEl.textContent = childTitle(record);
        else if (!record.suggestedName && !record.lastHumanHeadline
                && record.titleEl.textContent !== presentation.headline) {
            record.titleEl.textContent = blockHasWork(record) ? 'Task activity' : '';
        }
        settleLiveCard(record, activePhase, wasFinished);
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
        if (msg?.suggested_name) {
            changed = applySuggestedName(taskId, msg.suggested_name) || changed;
        }
        const finalizing = msg?.task_phase === 'finalizing' || msg?.outcome_final === false;
        changed = noteToolMetrics(taskId, msg, rawTs, { suppressDomInsert }) || changed;
        const summary = taskTerminalSummary({ ...msg, task_id: taskId });
        const record = getLiveCardRecord(taskId);
        changed = noteTaskDetails(record, msg) || changed;
        noteDirectTurn(record, msg?._is_direct_chat);
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
            // Write only on change: a rewrite would destroy a selection being copied.
            const next = rec?.isSubagent ? childTitle(rec) : '';
            if (next && rec.titleEl.textContent !== next) rec.titleEl.textContent = next;
            if (rec?.isSubagent) renderLiveCardMeta(rec);
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
        return summary ? withTaskCostMeta(summary, evt, { rawTs }) : null;
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
        let changed = msg.tool_evidence ? noteToolMetrics(taskId, msg, rawTs) : false;
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
            text: msg.content || msg.text || '',
            task_id: taskId,
            ...Object.fromEntries(['subagent_event', 'subagent_task_id', 'root_task_id',
                'parent_task_id', 'delegation_role', 'subagent_role', 'status', 'result',
                'trace_summary', 'error', 'artifact_status'].map((key) => [key, msg[key] || ''])),
            ...cardMetaKeys(msg),
            lifecycle: msg.lifecycle || null,
            // The frame's voice, live and on replay; absent stays absent.
            narration: msg.narration,
        });
        if (!summary) return changed;
        const presented = withTaskCostMeta(summary, msg, { rawTs });
        changed = queueTaskLiveUpdate(
            presented, taskId, normalizeLogTs(rawTs), presented.dedupeKey || '', rawTs,
        ) || changed;
        noteDirectTurn(liveCardRecords.get(taskId), msg._is_direct_chat);
        // History may carry the coined name that live frames deliver separately.
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
        if (![
            'scheduled', 'running', 'completed', 'completed_warn',
            'failed', 'cancelled', 'rejected', 'interrupted',
        ].includes(event)) {
            return routeSubagentProgressToCard(childId, evt);
        }
        const rawTs = tsValue || new Date().toISOString();
        const summary = summarizeSubagentCardFrame(evt, childId, {}, rawTs);
        if (!summary) return false;
        summary.dedupeKey = `subagent-lifecycle:${childId}`;
        // Interrupted is retryable and therefore non-terminal; the canonical
        // projector owns that distinction for both live and replay paths.
        if (summary.terminal) subagentTerminalChildren.add(childId);
        reanchorTaskCard(getLiveCardRecord(parentId), rawTs);
        stampNodeTimestamp(getSubagentCardRecord(childId, parentId, role)?.root, rawTs, { anchor: true });
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
    function routeSubagentProgressToCard(childId, msg) {
        const info = subagentChildParents.get(childId);
        if (!info) return false;
        const { parentId, role } = info;
        const content = String(msg?.content || msg?.text || '').trim();
        if (!content) return false;
        const rawTs = msg?.ts || new Date().toISOString();
        reanchorTaskCard(getLiveCardRecord(parentId), rawTs);
        const record = getSubagentCardRecord(childId, parentId, role);
        const preserveTerminal = Boolean(record?.finished && subagentTerminalChildren.has(childId));
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
        if (!summary.activity) summary.dedupeKey = `subagent-progress:${childId}`;
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
        reanchorTaskCard(getLiveCardRecord(parentId), rawTs);
        const record = getSubagentCardRecord(childId, parentId, role);
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
        if (!isGroupedTaskEvent(evt)) return false;
        const taskId = getLogTaskGroupId(evt) || '';
        if (!taskId) return false;
        const rawTs = evt.ts || evt.timestamp || new Date().toISOString();
        // Task-bound Skill lifecycle is presentation on its explicit owner,
        // never a synthetic lifecycle task card.
        const review = attachReviewFromRow(evt, rawTs);
        if (review !== undefined) return review;
        if (eventType === 'owner_hurry') {
            const root = ownerHurryProjection(evt).applied
                ? liveCardRecords.get(taskId)?.root : null;
            if (!root || root.getAttribute('data-owner-hurry') === '1') return false;
            root.setAttribute('data-owner-hurry', '1');
            return true;
        }
        const childInfo = subagentChildParents.get(taskId);
        if (childInfo && eventType === 'task_done') return routeSubagentTerminalToCard(taskId, evt);
        const summary = summarizeChatLiveEvent(evt);
        if (childInfo && subagentTerminalChildren.has(taskId) && !summary?.toolCall) return false;
        // Root/child fold: metrics, terminal and replay.
        let changed = evt.tool_evidence || ['task_metrics_event', 'task_eval', 'task_done'].includes(eventType)
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
        // Host-attested lane and Stop facts also travel on tool-only turns.
        if (typeof evt._is_direct_chat === 'boolean') noteDirectTurn(liveCardRecords.get(taskId), evt._is_direct_chat);
        if (evt.cancelable === true) markTaskCancelable(taskId);
        if (eventType === 'task_done' && summary.terminal) {
            recordTerminalActivity(taskId);
            syncChatStatus();
        }
        return Boolean(changed || queued || subagentChanged);
    }

    function addMessage(text, role, markdown = false, timestamp = null, isProgress = false, opts = {}) {
        const pending = !!opts.pending;
        const ephemeral = !!opts.ephemeral;
        const clientMessageId = opts.clientMessageId || '';
        const senderLabelOverride = opts.senderLabel || '';
        const senderSessionId = opts.senderSessionId || '';
        const source = opts.source || '';
        const initiator = opts.initiator || '';
        const systemType = opts.systemType || '';
        const taskId = opts.taskId || '';
        const projectId = opts.projectId || '';
        const projectName = opts.projectName || '';
        const ts = timestamp || new Date().toISOString();
        const legacyKey = buildMessageKey(role, text, ts, {
            clientMessageId,
            systemType,
            isProgress,
            source,
            senderLabel: senderLabelOverride,
            senderSessionId,
            taskId,
        });
        const messageKey = opts.historyId ? `history:${opts.historyId}` : legacyKey;
        if (opts.historyId || _historyReplayActive) {
            const prior = opts.historyId
                ? historyNodes(opts.historyId).find(node => node.classList.contains('chat-bubble'))
                : Array.from(messagesDiv.querySelectorAll('.chat-bubble')).find(node => node.dataset.messageKey === legacyKey);
            if (prior) {
                rememberMessageKey(messageKey);
                const originChanged = syncSavedProjectContext(prior, opts.originProjected, opts.originId);
                return chatDecision.renderRoutingDecision(prior, opts.chatAnnotation) || originChanged;
            }
        }
        if (messageKey && seenMessageKeys.has(messageKey)) {
            return false;
        }
        if (opts.historyId) {
            const prior = Array.from(messagesDiv.querySelectorAll('.chat-bubble')).find(node =>
                (clientMessageId && node.dataset.clientMessageId === clientMessageId)
                || (opts.originId && node.dataset.originId === opts.originId)
                || (!node.dataset.historyId && node.dataset.messageKey === legacyKey));
            if (prior) {
                stampHistoryNode(prior, opts.historyId, opts.historyPosition);
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
                text,
                role,
                ts,
                markdown: !!markdown,
                systemType,
                source,
                initiator,
                senderLabel: senderLabelOverride,
                senderSessionId,
                clientMessageId,
                taskId,
                projectId,
                projectName,
                handoffId: opts.handoffId || '',
                originProjected: Boolean(opts.originProjected),
                originId: opts.originId || '',
                terminalTime: opts.terminalTime || null,
                skillReview: opts.skillReview || null,
                evidenceRef: opts.evidenceRef || null,
            });
            // Mirror the sessionStorage slice(-200): the in-memory copy exists
            // only to feed that snapshot, so it obeys the same cap (P3).
            if (persistedHistory.length > 200) {
                persistedHistory.splice(0, persistedHistory.length - 200);
            }
            if (!_historyReplayActive) persistVisibleHistory();
        }

        const bubble = document.createElement('div');
        bubble.className = `chat-bubble ${role}` + (isProgress ? ' progress' : '');
        if (pending) bubble.classList.add('pending');
        if (ephemeral) bubble.dataset.ephemeral = '1';
        if (clientMessageId) bubble.dataset.clientMessageId = clientMessageId;
        if (systemType) bubble.dataset.systemType = systemType;
        if (senderSessionId) bubble.dataset.senderSessionId = senderSessionId;
        if (taskId) bubble.dataset.taskId = taskId;
        if (projectId) bubble.dataset.projectId = projectId;
        if (legacyKey) bubble.dataset.messageKey = legacyKey;
        stampHistoryNode(bubble, opts.historyId, opts.historyPosition);

        const sender = senderLabel(role, isProgress, systemType, {
            source, senderLabel: senderLabelOverride, senderSessionId, initiator,
        }, chatSessionId);
        const richMarkdown = role !== 'user' && systemType !== 'skill_review' && (role !== 'system' || markdown === true);
        const rendered = role === 'user'
            ? escapeHtml(text)
            : role === 'system' && systemType === 'skill_review'
                ? renderSkillReviewDisclosure(text, opts.skillReview || null)
                : role === 'system' && systemType !== 'skill_review' && markdown !== true
                    ? escapeHtml(text)
                    : renderChatMarkdown(text);
        const timeFmt = formatMsgTime(ts);
        const timeHtml = timeFmt ? `<div class="msg-time" title="${escapeHtmlAttr(timeFmt.full)}">${escapeHtml(timeFmt.short)}</div>` : '';
        const pendingHtml = pending ? `<div class="msg-pending">Queued until reconnect</div>` : '';
        // A placed row with no card keeps the record link its card row would offer.
        bubble.innerHTML = `
            <div class="sender">${escapeHtml(sender)}</div>
            <div class="message${richMarkdown ? ' ui-rich-content' : ''}">${rendered}</div>
            ${evidenceLinkHtml(opts.evidenceRef)}
            ${pendingHtml}
            ${timeHtml}
        `;
        if (!isProgress && text) chatMedia.attachCopyControl(bubble, String(text));
        if (systemType === 'project_handoff' && handoffs) handoffs.mount(bubble, { taskId, projectId, projectName, title: text, handoffId: opts.handoffId, kind: 'receipt' });
        else if (PROJECT_ROW_TYPES.has(systemType)) decorateProjectRow(bubble, { role, projectId, projectName,
            terminalTime: opts.terminalTime, addedAt: ts, completion: systemType === 'project_completion_summary' });
        syncSavedProjectContext(bubble, opts.originProjected, opts.originId);
        wireSkillReviewDisclosure(bubble, { onDomWrite: withStableViewport });
        stampNodeTimestamp(bubble, ts);
        insertMessageNode(bubble, { forceStick: !!opts.forceStick });
        if (richMarkdown) enhanceMountedMarkdown(bubble);
        chatDecision.renderRoutingDecision(bubble, opts.chatAnnotation);
        handoffs?.reconcile(bubble);
        rememberMessageKey(messageKey);
        if (pending && clientMessageId) pendingUserBubbles.set(clientMessageId, bubble);
        return bubble;
    }

    // Host-stamped Project lifecycle rows (§3 Main rows), live and replayed alike:
    // a mirrored final answer is an ordinary assistant message, the rest System text.
    function addProjectRow(msg, text, opts) {
        const a = msg.completion_answer;
        return addMessage(a || text, a ? 'assistant' : 'system', !!(a || msg.markdown), msg.ts || null, false, {
            ...opts, systemType: msg.system_type, projectId: msg.project_id || '',
            projectName: msg.project_name || '', handoffId: msg.handoff_id || '', terminalTime: msg.terminal_time || null,
        });
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
        if (bubble) handoffs?.reconcile(bubble);
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

    // Hydration triggers share one sticky request; reconnect/resync still refetch.
    function awaitInitialHydration({ includeUser = false } = {}) {
        if (initialHydrationPromise) return initialHydrationPromise;
        initialHydrationPromise = syncHistory({ includeUser });
        return initialHydrationPromise;
    }

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
        for (const id of pendingLiveEvictions) {
            const record = liveCardRecords.get(id);
            if (!record) { pendingLiveEvictions.delete(id); continue; }
            if (record.historyIds?.size || activeDirectActivities.has(id)
                    || (!record.finished && !record.historicalUnavailable)
                    || historyNodeIsProtected(record.root, messagesDiv)) continue;
            disposeLiveCard(id);
            subagentChildParents.delete(id); subagentTerminalChildren.delete(id);
            explicitCardExpansion.delete(id); reviewDisclosureByTask.delete(id);
            pendingLiveEvictions.delete(id);
        }
    }
    function applyHistoryMessages(messages, { fromReconnect = false, includeUser = true, archived = false } = {}) {
        const knownCards = new Set(liveCardRecords.keys());
        _historyReplayActive = true;
        _historyAppending = archived;
        liveCardBound.beginReplay();
        try {
            withStableViewport(() => {
                // A formerly unbound final can gain its real child lineage on a
                // deeper page. Keep a selected old bubble until it can move.
                messages = messages.filter(row => {
                    if (!row.history_id || row.delegation_role !== 'subagent' || row.is_progress) return true;
                    const old = historyNodes(row.history_id).find(node => node.classList.contains('chat-bubble'));
                    if (!old) return true;
                    if (historyNodeIsProtected(old, messagesDiv)) {
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
                for (const msg of messages) learnSubagentLineage(msg);

                const historicalTerminalProjections = new Set();
                for (const msg of messages) if (msg.historical_terminal && msg.task_id) {
                    historicalTerminals.set(msg.task_id, msg.historical_terminal);
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
                    if (msg.system_type === 'quiz_answer') chatDecision.applyQuizStateFrame(messagesDiv, { ...msg.quiz, task_id: msg.task_id });
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
                        updateLiveCardFromProgressMessage(msg, { grantCancelAuthority: msg.project_mirror !== true });
                        const historyCard = liveCardRecords.get(taskId);
                        if (historyCard && !historyCard.isSubagent && !activeDirectActivities.has(taskId)
                                && !msg.task_terminal_status && msg.task_phase !== 'finalizing') {
                            if (setHistoricalUnconfirmed(historyCard)) {
                                syncCancelRunButton(historyCard);
                                renderLiveCardMeta(historyCard);
                            }
                        }
                        if (historyCard && historicalTerminals.has(taskId)) historyCard.historicalTerminal = historicalTerminals.get(taskId);
                        continue;
                    }
                    // Pass 2 mounts the block in its transcript position.
                    if (msg.system_type === 'task_summary') appendTaskSummaryToLiveCard(msg, { suppressDomInsert: true });
                } } finally { _syncPass1Active = false; _historyRow = null; }

                // Pass 2 inserts cards at the first visible task message, then finishes them.
                const insertedCardTaskIds = new Set();
                function reorderDirtyCardIfNeeded(rec) {
                    if (!rec?._anchorOrderDirty || rec.isSubagent || !rec.root?.isConnected) return;
                    insertMessageNode(rec.root, { reorderExisting: true });
                    rec._anchorOrderDirty = false;
                }
                function insertCardIfNeeded(taskId) {
                    if (!taskId || insertedCardTaskIds.has(taskId)) return;
                    insertedCardTaskIds.add(taskId);
                    const rec = liveCardRecords.get(taskId);
                    reorderDirtyCardIfNeeded(rec);
                    ensureLiveCardVisible(rec);
                }
                // A stored row as its bubble: its own identity and labels, plus what its kind adds.
                const addStoredRow = (msg, taskId, extra) => addMessage(msg.text, msg.role, !!msg.markdown, msg.ts || null, false, {
                    historyId: msg.history_id, historyPosition: msg.history_position,
                    systemType: msg.system_type || '', taskId,
                    source: msg.source || '', initiator: msg.initiator || '',
                    senderLabel: msg.sender_label || '', senderSessionId: msg.sender_session_id || '',
                    ...extra,
                });
                for (const msg of messages) {
                    _historyRow = msg;
                    const taskId = msg.task_id || '';
                    if (msg.system_type === 'project_question_pointer') { chatDecision.appendQuestionPointer(msg); continue; }
                    if (isReplayEvidenceRow(msg)) continue;
                    // Owner-bound reviews attached in pass 1 are not terminal chat bubbles.
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
                    // Reconnect: a durably recorded submission must not stay
                    // `Sending...` — history + snapshot are the authorities
                    // (a live turn re-links via census hydration).
                    if (fromReconnect && msg.role === 'user' && msg.client_message_id) {
                        pendingSubmissions.delete(String(msg.client_message_id));
                    }
                    if (!includeUser && msg.role === 'user') continue;
                    if (msg.is_progress) {
                        // Progress-only/failed tasks still anchor at their first event.
                        insertCardIfNeeded(taskId);
                        // Open post-task checkpoint replays as "Finalizing…".
                        if (msg.task_phase === 'finalizing') markLiveCardFinalizing(taskId, msg);
                        continue;
                    }
                    if (msg.system_type === 'task_summary') continue;
                    if (PROJECT_ROW_TYPES.has(msg.system_type)) {
                        addProjectRow(msg, msg.text, { historyId: msg.history_id, historyPosition: msg.history_position, taskId });
                        continue;
                    }
                    // Delivered media is a bubble, not a task-final
                    // message — render it BEFORE the taskId/finishLiveCard block so
                    // a mid-task delivery replayed while its task is still
                    // running does not falsely finalize that task's live card.
                    if (['document', 'photo', 'video', 'links', 'quiz'].includes(msg.msg_type)) {
                        if (msg.msg_type === 'document') appendDocumentBubble(msg);
                        else if (msg.msg_type === 'links') appendLinksMessage(msg);
                        else if (msg.msg_type === 'quiz') appendQuizMessage(msg);
                        else appendMediaBubble(msg);
                        continue;
                    }
                    // Replay conclusion: a typed terminal fact OR a plain
                    // untyped final (replay has no later task_done frame, so
                    // the bare final is the task's last word; marked rows —
                    // system_type/msg_type — still never conclude).
                    const plainUntypedFinal = !msg.system_type && !msg.msg_type;
                    if (
                        taskId
                        && (msg.role === 'assistant' || msg.role === 'system')
                        && (positiveTaskTerminalFact(msg) || plainUntypedFinal)
                        && !isNonTerminalMediaHistoryRow(msg)
                    ) {
                        if (subagentChildParents.has(taskId)) {
                            insertCardIfNeeded(taskId);
                            routeSubagentFinalMessageToCard(taskId, msg);
                            const record = liveCardRecords.get(taskId);
                            finishLiveCard(taskId, msg.task_terminal_status ? taskTerminalPhase(msg) : replayTerminalPhase(record));
                            continue;
                        }
                        insertCardIfNeeded(taskId);
                        // A replayed early final must not finalize the card.
                        if (msg.task_phase === 'finalizing') {
                            markLiveCardFinalizing(taskId, msg);
                        } else if (!msg.historical_terminal || positiveTaskTerminalFact(msg)) {
                            const record = liveCardRecords.get(taskId);
                            finishLiveCard(taskId, msg.task_terminal_status ? taskTerminalPhase(msg) : replayTerminalPhase(record));
                        }
                    }
                    // A replayed durable routing receipt carries the same
                    // authority as its live WS frame: a receipt that landed
                    // while the socket was down still retires `Sending...`.
                    if (msg.chat_annotation && msg.client_message_id) {
                        pendingSubmissions.delete(String(msg.client_message_id));
                    }
                    addStoredRow(msg, taskId, {
                        clientMessageId: msg.client_message_id || '',
                        originProjected: msg.origin_projected === true,
                        originId: msg.origin_id || '',
                        chatAnnotation: msg.chat_annotation || null,
                        skillReview: msg.system_type === 'skill_review' && msg.skill && msg.job_id
                            ? { skill: msg.skill, jobId: msg.job_id }
                            : null,
                    });
                    if (msg.role === 'user') markIngressSaved(messagesDiv, msg);
                }
                _historyRow = null;
                // Resolve cards whose task is already terminal on the server
                // (crash storm / hard timeout / cancellation write a terminal
                // status but no task_summary). Without this their progress-only
                // cards re-inflate as "Working" forever on reload/reconnect.
                const terminalTaskRecords = new Map();
                for (const msg of messages) {
                    const tid = msg.task_id || '';
                    if (tid && msg.task_terminal_status) {
                        terminalTaskRecords.set(tid, {
                            ...msg,
                            status: String(msg.task_terminal_status),
                        });
                    }
                }
                for (const [tid, terminalRecord] of terminalTaskRecords) {
                    _historyRow = terminalRecord;
                    if (!taskDoneIsTerminal(terminalRecord)) continue;
                    // Subagent terminal status resolves the child card, not the
                    // parent. Otherwise reload can revive a crashed/cancelled child.
                    if (subagentChildParents.has(tid)) {
                        routeSubagentTerminalToCard(tid, terminalRecord);
                        continue;
                    }
                    const rec = liveCardRecords.get(tid);
                    if (rec) {
                        insertCardIfNeeded(tid);
                        if (terminalRecord.outcome_axes || terminalRecord.review_projection || terminalRecord.reason_code) {
                            appendTaskSummaryToLiveCard(terminalRecord);
                        } else {
                            // P5: shared terminal mapping — a cancelled root replays
                            // as "Cancelled", never as a generic "Done".
                            finishLiveCard(tid, taskTerminalPhase(terminalRecord));
                        }
                    }
                }
                _historyRow = null;

                // The hidden terminal-root projection is the durable outcome
                // authority. Apply its projected phase after narrative rows,
                // which may still say finalizing; status alone loses failures.
                for (const [tid, historicalTerminal] of historicalTerminals) {
                    if (!historicalTerminalProjections.has(tid)) continue;
                    const terminalRecord = { ...historicalTerminal, task_id: tid };
                    if (!taskDoneIsTerminal(terminalRecord)) continue;
                    if (subagentChildParents.has(tid)) {
                        routeSubagentTerminalToCard(tid, terminalRecord);
                        continue;
                    }
                    const rec = liveCardRecords.get(tid);
                    if (!rec) continue;
                    insertCardIfNeeded(tid);
                    finishLiveCard(tid, historicalTerminal.phase);
                }

                // Every block the predicate admits is in the transcript after a rebuild.
                for (const rec of liveCardRecords.values()) {
                    reorderDirtyCardIfNeeded(rec);
                    ensureLiveCardVisible(rec);
                }

                for (const [id, record] of liveCardRecords) {
                    if (!knownCards.has(id) && !record.finished && !activeDirectActivities.has(id)) {
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
                for (const record of liveCardRecords.values()) {
                    if (record._timelineDirty) renderLiveCardTimeline(record);
                }
                persistVisibleHistory();
                return true;
            });
        } finally {
            _historyRow = null;
            _historyReplayActive = false;
            _historyAppending = false;
        }
        syncChatStatus();
    }

    async function syncHistory({ includeUser = false, fromReconnect = false } = {}) {
        if (historySyncPromise) {
            // Preserve reconnect intent across an in-flight ordinary sync.
            if (fromReconnect) {
                pendingReconnectSync = true;
                return historySyncPromise.then(() => {
                    // One waiter consumes the queued rebuild; peers await it.
                    if (pendingReconnectSync) {
                        pendingReconnectSync = false;
                        return syncHistory({ includeUser: false, fromReconnect: true });
                    }
                    return historySyncPromise || lastHistorySyncSucceeded;
                });
            }
            return historySyncPromise;
        }
        historySyncPromise = (async () => {
            const armedAtStart = liveCardBound.begin();
            const cardsAtStart = new Set(liveCardRecords.keys());
            try {
                // No welcome until this read lands.
                emptyWelcome?.historyPending();
                // An empty feed shows the read in flight (#1102); a painted one is left alone.
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
                const restoring = !restoredPageReady && !historyPager.getState().initialized
                    ? historyPager.restore(initialScrollState.history) : null;
                const oldRecentIds = recentHistoryIds;
                const admitted = acceptRecentWindow(data, messages);
                const pagerBeforeRecent = historyPager.getState();
                const rechainRecent = admitted && !data.reason_code && pagerBeforeRecent.initialized && !pagerBeforeRecent.canNewer
                    && [...oldRecentIds].some(id => !recentHistoryIds.has(id));
                const result = historyPager.acceptRecent(data);
                if (result.status !== 'applied') applyHistoryMessages(messages, { fromReconnect });
                // Recent owners exist before the saved page attaches content-only rows.
                if (restoring) {
                    restoredPageReady = (await restoring).status === 'applied';
                    if (destroyed) return false;
                }
                const recentState = historyPager.getState();
                const releasableRecentIds = !admitted || rechainRecent || recentState.canNewer ? [] : oldRecentIds;
                withStableViewport(() => releaseHistoryIds(releasableRecentIds));
                if (rechainRecent && !destroyed) void historyPager.latest();
                if (armedAtStart) {
                    const represented = new Set(messages.map(row => row.presentation_owner_task_id || row.task_id));
                    for (const id of cardsAtStart) if (!represented.has(id)) pendingLiveEvictions.add(id);
                    withStableViewport(releaseLiveOverflow);
                }
                syncChatStatus();

                if (!inputHistorySeededFromServer) {
                    const serverTexts = [];
                    for (const msg of messages) {
                        if (msg.role !== 'user') continue;
                        let text = (msg.text || '').trim();
                        if (text) serverTexts.push(text);
                    }
                    const combined = [...serverTexts, ...inputHistory];
                    const deduped = [];
                    const seen = new Set();
                    for (let i = combined.length - 1; i >= 0; i--) {
                        if (!seen.has(combined[i])) {
                            deduped.unshift(combined[i]);
                            seen.add(combined[i]);
                        }
                    }
                    inputHistory.length = 0;
                    inputHistory.push(...deduped.slice(-50));
                    saveChatInputHistory(sessionStorage, CHAT_INPUT_HISTORY_KEY, inputHistory);
                    inputHistoryIndex = inputHistory.length;
                    inputHistorySeededFromServer = true;
                }

                const wasFirstLoad = !historyLoaded;
                historyLoaded = true;
                lastHistorySyncSucceeded = true;
                messagesDiv.dataset.historyHydrated = 'true';
                emptyWelcome?.historyRead(data.window?.complete === true);
                liveCardBound.settle({ rebuilt: wasFirstLoad || armedAtStart, size: liveCardRecords.size });
                modelWaits.retainCards(liveCardRecords);
                // ANY successful sync leaves the instance hydrated
                // — later hydration triggers ride this sticky promise.
                initialHydrationPromise = historySyncPromise;
                syncLoadOlderControl();
                // A recreated project instance restores its predecessor's stashed
                // mid-history position on first paint instead of pinning to newest.
                if (reading.pending) {
                    updateMessagesPadding(false);
                    reading.position();
                } else if (wasFirstLoad && reading.stick) {
                    updateMessagesPadding();
                    reading.followAfterLayout();
                }
                readReceipt.settle();
                return messages.length > 0;
            } catch (err) {
                lastHistorySyncSucceeded = false;
                emptyWelcome?.historyRead(false);
                initialHydrationPromise = null;
                // Never leave an empty feed blank: the failure and its Retry replace the loading state.
                historyControls.endRecent(err); syncLoadOlderControl();
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
    const readReceipt = createProjectReadReceipt({
        read: async (fresh) => {
            if (!fresh && initialHydrationPromise) await initialHydrationPromise;
            else {
                if (historySyncPromise) await historySyncPromise;
                await syncHistory({ includeUser: true });
            }
            return lastHistorySyncSucceeded;
        },
        // A destroyed page reports hidden===false, hence the lifecycle flag.
        isShown: () => !destroyed && !page.hidden,
        // A place still being restored is not where the reader is.
        isReadingLatest: latest => !destroyed && !reading.pending && isInstanceVisible() && isAtNewestMessage(latest, {
            delivered: id => retainedHistoryIds().has(id), nodes: historyNodes, viewport: messagesDiv, header: pageHeader, composer: inputArea,
            atBottom: () => isNearBottom() && !historyPager.getState().canNewer,
        }),
        onReadingLatest,
    });

    (async () => {
        await loadUiPreferences();
        // Main waits for the (bounded) idle hydration window;
        // project instances pass straight through. The sticky single-flight
        // below folds this trigger with the first socket open / refreshHistory.
        await waitForHydrationWindow();
        if (destroyed) return;
        if (await awaitInitialHydration({ includeUser: true })) return;
        try {
            const saved = JSON.parse(sessionStorage.getItem(storeKey(CHAT_STORAGE_KEY)) || '[]');
            // A snapshot row is the addMessage option bag persistedHistory wrote;
            // addMessage normalizes every option, so no field list is kept in sync.
            for (const { text, role, markdown, ts, ...restored } of saved) {
                addMessage(text, role, !!markdown, ts || null, false, restored);
            }
        } catch {}
        historyLoaded = true;
        // The next successful source read reconciles this offline preview.
        if (!lastHistorySyncSucceeded) liveCardBound.arm();
    })();

    function rememberInput(text) {
        if (!text) return;
        if (inputHistory[inputHistory.length - 1] !== text) inputHistory.push(text);
        saveChatInputHistory(sessionStorage, CHAT_INPUT_HISTORY_KEY, inputHistory);
        inputHistoryIndex = inputHistory.length;
        inputDraft = '';
    }

    function resizeChatInput() {
        const caretAtEnd = input.selectionEnd >= input.value.length - 1;
        const previousScrollTop = input.scrollTop;
        input.style.height = 'auto';
        input.style.height = Math.min(input.scrollHeight, 120) + 'px';
        input.scrollTop = caretAtEnd ? input.scrollHeight : previousScrollTop;
        updateMessagesPadding();
    }

    function restoreInputHistory(step) {
        if (!inputHistory.length) return;
        if (step < 0) {
            if (input.selectionStart !== 0 || input.selectionEnd !== 0) return;
            if (inputHistoryIndex === inputHistory.length) inputDraft = input.value;
            inputHistoryIndex = Math.max(0, inputHistoryIndex - 1);
            input.value = inputHistory[inputHistoryIndex] || '';
        } else {
            if (input.selectionStart !== input.value.length || input.selectionEnd !== input.value.length) return;
            inputHistoryIndex = Math.min(inputHistory.length, inputHistoryIndex + 1);
            input.value = inputHistoryIndex === inputHistory.length ? inputDraft : (inputHistory[inputHistoryIndex] || '');
        }
        resizeChatInput();
        const cursor = input.value.length;
        input.setSelectionRange(cursor, cursor);
    }

    async function sendMessage(planMode = false) {
        if (sendBtn.disabled) return;  // guard against Enter re-entry during async upload
        let text = input.value.trim();
        const hasAttachments = pendingAttachments.length > 0;
        let uploadedAttachments = [];
        let attachmentMeta = [];
        if (!text && !pendingAttachments.length) return;
        if (pendingAttachments.length) {
            // Upload immediately before send; offline queueing would orphan files.
            if (ws.ws?.readyState !== WebSocket.OPEN) {
                showToast('Cannot attach file while offline. Reconnect and try again.', 'error');
                return;
            }
            const staged = [...pendingAttachments];
            const uploaded = [];
            setAttachmentUploadState(true);
            setSendBusy(true, staged.length > 1 ? 'Uploading files' : 'Uploading');
            try {
                for (const stagedItem of staged) {
                    if (ws.ws?.readyState !== WebSocket.OPEN) throw new Error('Connection closed during upload. Reconnect and try again.');
                    const formData = new FormData();
                    formData.append('file', stagedItem.file);
                    const resp = await apiFetch('/api/chat/upload', { method: 'POST', body: formData });
                    const data = await resp.json().catch(() => ({}));
                    if (!resp.ok || !data.ok) {
                        throw new Error(data.error || resp.statusText);
                    }
                    uploaded.push({
                        filename: data.filename || '',
                        path: data.path || '',
                        display_name: data.display_name || stagedItem.display_name,
                        mime: data.mime || stagedItem.file?.type || '',
                    });
                }
                if (ws.ws?.readyState !== WebSocket.OPEN) throw new Error('Connection closed after upload. Reconnect and try again.');
                uploadedAttachments = uploaded;
                const attachmentLines = uploaded.slice(0, ATTACHMENT_PREVIEW_COUNT)
                    .map((item) => `[Attached file: ${item.display_name}]`)
                    .concat(uploaded.length > ATTACHMENT_PREVIEW_COUNT ? [`[${uploaded.length - ATTACHMENT_PREVIEW_COUNT} more attached files]`] : [])
                    .join('\n');
                text += (text ? '\n\n' : '') + attachmentLines;
                // Structured attachment metadata rides the WS frame so the
                // gateway can hand image uploads to the model as NATIVE image
                // blocks (vision models) instead of only a path label.
                attachmentMeta = uploaded.map((item) => ({
                    filename: item.filename,
                    display_name: item.display_name,
                    mime: item.mime || '',
                }));
            } catch (e) {
                await cleanupUploadedAttachments(uploaded);
                showToast('Upload error: ' + e.message, 'error');
                return;  // pending attachments and preview remain so the user can retry
            } finally {
                setAttachmentUploadState(false);
                setSendBusy(false);
            }
        }
        if (!text) return;
        const forcePlan = !!planMode && !text.startsWith('/');
        const result = ws.send({
            type: 'chat',
            content: text,
            sender_session_id: chatSessionId,
            force_plan: forcePlan,
            ...(isMain ? {} : { chat_id: chatId }),
            ...(projectId ? { project_id: projectId } : {}),
            ...(attachmentMeta.length ? { attachments: attachmentMeta } : {}),
            ...clientSurfaceField(),
        }, hasAttachments ? { queue: false } : undefined);
        if (hasAttachments && result?.status !== 'sent') {
            await cleanupUploadedAttachments(uploadedAttachments);
            showToast('Connection lost before send. Reconnect and try again.', 'error');
            return;
        }
        // One-shot: disarm Swarm now that the message is sent.
        if (planMode) setSwarm(false);
        if (hasAttachments) {
            pendingAttachments = [];
            updateAttachmentPreview();
        }
        rememberInput(text);
        input.value = '';
        clearInputDraft();
        const sentTs = new Date().toISOString();
        reading.cancel(); // an accepted Send supersedes any saved place
        addMessage(text, 'user', false, sentTs, false, {
            pending: result?.status === 'queued',
            source: 'web',
            senderSessionId: chatSessionId,
            clientMessageId: result?.clientMessageId || '',
            forceStick: true,
        });
        // ws.send always coins a client_message_id for chat frames; guard
        // only against a non-chat result shape.
        const pendingId = result?.clientMessageId || '';
        if (pendingId) {
            pendingSubmissions.set(pendingId, {
                clientMessageId: pendingId,
                timestamp: Date.now(),
            });
            recordLocalEcho(pendingId, text, sentTs);
        }
        syncChatStatus();
        resizeChatInput();
        reading.followAfterLayout();
    }

    // Send mode lives on DOM so CSS and click/Enter share one source.
    const sendGroup = page.querySelector('.chat-send-group');

    // Swarm is a one-shot arm: the next send goes through plan_task multi-model
    // brainstorm/planning, then the pill auto-disarms so it never sticks.
    const swarmBtn = byId('swarm');
    function swarmArmed() {
        return swarmBtn?.dataset.armed === 'true';
    }
    function setSwarm(armed) {
        if (swarmBtn) swarmBtn.dataset.armed = armed ? 'true' : 'false';
    }

    function setSendBusy(busy, label = '') {
        sendGroup.dataset.busy = busy ? '1' : '0';
        sendBtn.disabled = busy;
        if (busy) {
            sendBtn.textContent = label || 'Sending';
            sendBtn.title = label || 'Sending';
        } else {
            sendBtn.textContent = 'Send';
            sendBtn.title = 'Send message';
        }
    }

    swarmBtn?.addEventListener('click', () => setSwarm(!swarmArmed()));

    // Context-mode quick toggle: the owner endpoint hot-applies the setting
    // without a restart; Max -> Low is accepted only while Ouroboros is idle.
    const contextModeBtn = byId('context-mode');
    contextModeBtn?.addEventListener('click', async (event) => {
        const seg = event.target.closest('.chat-seg');
        if (!seg || contextModeBtn.dataset.disabled === 'true') return;
        const next = ['nano', 'low', 'max'].includes(seg.dataset.mode) ? seg.dataset.mode : 'max';
        const current = ['nano', 'low', 'max'].includes(contextModeBtn.dataset.contextMode) ? contextModeBtn.dataset.contextMode : 'max';
        if (next === current) return;
        contextModeBtn.dataset.disabled = 'true';
        const postMode = (mode) => apiFetch('/api/owner/context-mode', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mode }),
        });
        try {
            const resp = await postMode(next);
            if (resp.ok) {
                contextModeBtn.dataset.contextMode = next;
            } else {
                let message = 'Could not change context mode.';
                try { const p = await resp.json(); if (p?.error) message = p.error; } catch {}
                showToast(message, 'error');
            }
        } catch (e) {
            showToast(`Could not change context mode: ${e.message || e}`, 'error');
            /* leave the current value; /api/state refresh will resync */
        } finally {
            contextModeBtn.dataset.disabled = 'false';
            refreshHeaderControlState(true);
        }
    });

    // Arrow wrappers avoid MouseEvent leaking into sendMessage(planMode).
    sendBtn.addEventListener('click', () => sendMessage(swarmArmed()));
    bindEnterSubmit(input, sendBtn);
    input.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowUp' && !e.shiftKey) {
            restoreInputHistory(-1);
        } else if (e.key === 'ArrowDown' && !e.shiftKey) {
            restoreInputHistory(1);
        }
    });
    // Dynamic CSS reserve keeps the absolute composer from covering messages.
    function scrollToBottom() {
        messagesDiv.scrollTop = messagesDiv.scrollHeight;
        updateScrollButton();
    }

    // Scroll events observe position; only positive navigation changes follow intent.
    messagesDiv?.addEventListener('scroll', () => { reading.scroll(); settleHistoryViewport(); readReceipt.note(); }, { passive: true });
    messagesDiv?.addEventListener('load', reading.reflow, true);

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
    scrollBottomBtn?.addEventListener('click', async () => {
        const current = reading.claim();
        if (historySyncPromise) await historySyncPromise;
        await historyPager.whenIdle();
        if (destroyed || !current()) return;
        if (historyPager.getState().canNewer || historyWindow?.horizonGap) {
            if ((await historyPager.latest()).status !== 'applied') return;
        }
        if (destroyed || !current()) return;
        reading.stick = true;
        reading.followAfterLayout();
        updateScrollButton();
    });

    function updateMessagesPadding(preserveStickiness = true) {
        const mutate = () => {
            let changed = false;
            if (pageHeader && messagesDiv) {
                const headerReserve = Math.max(56, Math.ceil(pageHeader.offsetHeight || 0));
                const value = `${headerReserve}px`;
                if (page.style.getPropertyValue('--chat-header-reserve') !== value) {
                    page.style.setProperty('--chat-header-reserve', value);
                    changed = true;
                }
            }
            if (inputArea && messagesDiv) {
                const reserve = Math.max(92, Math.ceil(inputArea.offsetHeight || 0) + 16);
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
        if (inputArea) chatResizeObserver.observe(inputArea);
    }

    installChatResizeObservers();

    // Per-thread drafts survive destroy-on-close: save on input, restore on
    // creation, clear on send.
    function saveInputDraft() {
        try {
            if (input.value) sessionStorage.setItem(storeKey(CHAT_DRAFT_KEY), input.value);
            else sessionStorage.removeItem(storeKey(CHAT_DRAFT_KEY));
        } catch {}
    }

    function clearInputDraft() {
        try { sessionStorage.removeItem(storeKey(CHAT_DRAFT_KEY)); } catch {}
    }

    try {
        const savedDraft = sessionStorage.getItem(storeKey(CHAT_DRAFT_KEY)) || '';
        if (savedDraft && !input.value) {
            input.value = savedDraft;
            inputDraft = savedDraft;
            resizeChatInput();
        }
    } catch {}

    input.addEventListener('input', () => {
        if (inputHistoryIndex === inputHistory.length) inputDraft = input.value;
        resizeChatInput();
        saveInputDraft();
    });

    headerActions?.addEventListener('click', async (event) => {
        const button = event.target.closest('[data-chat-command]');
        if (!button) return;
        button.closest('details')?.removeAttribute('open');
        const command = button.dataset.chatCommand;
        if (command === 'evolve') {
            const next = !button.classList.contains('on');
            button.classList.toggle('on', next);
            ws.send({ type: 'command', cmd: `/evolve ${next ? 'start' : 'stop'}` });
            return;
        }
        if (command === 'bg') {
            const next = !button.classList.contains('on');
            button.classList.toggle('on', next);
            ws.send({ type: 'command', cmd: `/bg ${next ? 'start' : 'stop'}` });
            return;
        }
        if (command === 'review') {
            ws.send({ type: 'command', cmd: '/review' });
            return;
        }
        if (command === 'restart') {
            ws.send({ type: 'command', cmd: '/restart' });
            return;
        }
        if (command === 'panic') {
            // CRITICAL CONTROL: the whole confirm-and-send flow lives in the
            // node-tested confirmAndSendPanic (dialog options + strict
            // shouldFirePanic gate + the exact /panic command); this handler
            // only injects the real deps. Manual check on release: click
            // Panic → dialog → "Kill all workers" sends /panic;
            // Cancel/Escape/backdrop send nothing.
            await confirmAndSendPanic({ openConfirmDialog, ws });
        }
    });

    // Native <details> needs outside-click/Escape dismissal; destroy removes
    // these retained handler refs.
    let documentClickHandler = null;
    let documentKeydownHandler = null;
    if (!asPanel) {
        const collapseHeaderMenus = (predicate) => {
            page.querySelectorAll('details.chat-header-more[open]').forEach((details) => {
                if (predicate(details)) details.removeAttribute('open');
            });
        };
        documentClickHandler = (event) => {
            collapseHeaderMenus((details) => !details.contains(event.target));
        };
        document.addEventListener('click', documentClickHandler);
        documentKeydownHandler = (event) => {
            if (event.key === 'Escape') collapseHeaderMenus(() => true);
        };
        document.addEventListener('keydown', documentKeydownHandler);
    }

    budgetPill?.addEventListener('click', () => {
        if (typeof openDashboardTab === 'function') openDashboardTab('costs');
        else if (typeof openSettingsTab === 'function') openSettingsTab('costs');
    });

    let headerControlInterval = null;
    if (asPanel) {
        // Late panels missed socket open; seed from the socket and latest snapshot.
        hostReady = supervisorReady(stateSnapshots.latest?.()) ?? hostReady;
        const seed = computeDerivedChatStatus({ supervisorStarting: !hostReady });
        if (ws.isConnected?.()) setStatus(seed.kind, seed.text);
    } else headerControlInterval = setInterval(refreshHeaderControlState, 3000);
    // Both Main and late panels hydrate once; each instance filters its chat.
    refreshHeaderControlState(true);

    const typingEl = document.createElement('div');
    // Per-instance id (main stays 'typing-indicator'; panels get a unique id) so
    // multiple open chat columns never collide on a duplicate DOM id.
    typingEl.id = idPrefix === 'chat' ? 'typing-indicator' : `${idPrefix}-typing-indicator`;
    typingEl.className = 'chat-bubble assistant typing-bubble';
    typingEl.style.display = 'none';
    typingEl.innerHTML = `<div class="typing-dots"><span></span><span></span><span></span></div>`;
    messagesDiv.appendChild(typingEl);

    const pageHistoryIds = new Map();
    let recentCoverage = null;
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
        // The newest admitted window (↓ too) owns readiness and the failure note.
        recentReady = !data.reason_code;
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
        if (reading.pending && reading.target?.history?.pages?.[reading.target.history.focus]?.id === descriptor.id) return true;
        const ids = pageHistoryIds.get(descriptor.id) || new Set();
        if (historyStamps(messagesDiv).some(([id, node]) => ids.has(id) && historyNodeIsProtected(node, messagesDiv))) return true;
        return [...liveCardRecords.values()].some(record =>
            [...(record.historyIds || [])].some(id => ids.has(id))
            && [record.summaryButtonEl, record.reviewsHostEl].some(node => historyNodeIsProtected(node, messagesDiv)));
    }

    // One retirement path for a message node: its media, decision views and markdown go with it.
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
        for (const id of ids) {
            if (retained.has(id)) { pendingHistoryEvictions.delete(id); continue; }
            const nodes = byId.get(id) || [];
            if (nodes.some(node => historyNodeIsProtected(node, messagesDiv))) {
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
        for (const [id, record] of [...liveCardRecords].reverse()) {
            let affected = false;
            for (const key of record.historyIds || []) {
                if (released.has(key)) affected = record.historyIds.delete(key) || affected;
            }
            const items = record.items.filter(item => !released.has(item.historyId || item.sourceHistoryId));
            if (items.length !== record.items.length) {
                record.items = items;
                renderLiveCardTimeline(record);
                updateLiveCardCount(record);
            }
            if (!affected || record.historyIds?.size || activeDirectActivities.has(id)
                    || record.subagentsEl?.querySelector('.chat-live-card')
                    || historyNodeIsProtected(record.root, messagesDiv)) continue;
            if (!record.finished && !record.historicalUnavailable && !record.historicalUnconfirmed) continue;
            disposeLiveCard(id);
            subagentChildParents.delete(id); subagentTerminalChildren.delete(id);
            reviewDisclosureByTask.delete(id); explicitCardExpansion.delete(id); historicalTerminals.delete(id);
        }
    }

    const historyControls = createHistoryControls(messagesDiv, pageHeader || page.querySelector('.chat-panel-statusbar'));
    const { olderButton: loadOlderBtn } = historyControls;

    const historyPager = createChatHistoryPager({
        fetchPage: fetchHistory,
        isAlive: () => !destroyed,
        applyPage: (messages, descriptor) => {
            pageHistoryIds.set(descriptor.id, new Set(messages.flatMap(historyRowIds)));
            const oldRecentIds = recentHistoryIds;
            const admitted = descriptor.direction === 'latest' && acceptRecentWindow(descriptor, messages);
            const archived = descriptor.direction !== 'recent' && descriptor.direction !== 'latest';
            applyHistoryMessages(messages, { archived });
            if (admitted) withStableViewport(() => releaseHistoryIds(oldRecentIds));
            // An older page drawn can show, or name, the newest arrival without a scroll.
            if (admitted) readReceipt.settle();
            else if (archived) readReceipt.page(descriptor);
        },
        releasePage: descriptor => {
            const ids = pageHistoryIds.get(descriptor.id) || [];
            pageHistoryIds.delete(descriptor.id);
            withStableViewport(() => releaseHistoryIds(ids));
        },
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
    async function loadOlderHistory() {
        if (historyControls.recentFailed()) {
            const owned = onHistoryRetry?.();
            return historySyncPromise ? owned : syncHistory({ includeUser: true });
        }
        const snapshot = historyPager.getState();
        if (snapshot.error?.body?.reason_code === 'history_view_changed') {
            reading.cancel(); restoredPageReady = true;
            return historyPager.latest();
        }
        if (snapshot.error) {
            const result = await historyPager.retry();
            if (snapshot.retryDirection === 'restore' && result.status === 'applied') {
                restoredPageReady = true;
                reading.position();
                const owned = onHistoryRetry?.();
                if (historySyncPromise) await owned;
                else await syncHistory({ includeUser: true });
            }
            return result;
        }
        return loadHistoryAtEdge(snapshot.canNewer ? 'newer' : 'older');
    }
    async function loadHistoryAtEdge(direction) {
        let result;
        // Two bounded physical reads per gesture cross sparse pages without
        // turning a short island or reflow into an automatic archive chase.
        for (let reads = 0; reads < 2; reads++) {
            if (direction === 'older' && historyWindow?.horizonGap && !historyPager.getState().canNewer) {
                result = await historyPager.latest(); // rebase bytes, preserve the reading destination
            } else result = await historyPager[direction]();
            if (destroyed || result.status !== 'applied' || result.messageCount !== 0) break;
        }
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

    function deriveChatStatus() {
        return computeDerivedChatStatus({
            ...chatStatusCounts(activeDirectActivities, liveCardRecords.values(), (id) => modelWaits.waiting(id)),
            isConnected: ws.isConnected ? ws.isConnected() : true,
            pendingSubmissionsCount: pendingSubmissions.size,
            supervisorStarting: !hostReady,
        });
    }

    function syncChatStatus() {
        const derived = deriveChatStatus();
        setStatus(derived.kind, derived.text);
        return setTypingIndicatorVisible(derived.showDots && foregroundCards().length === 0);
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
                    const historical = currentRecord.historicalTerminal;
                    if (historical) {
                        if (applyHistoricalModelExecution(currentRecord, historical)) {
                            changed = renderLiveCardMeta(currentRecord) || changed;
                        }
                        return finishLiveCard(taskId, historical.phase) || changed;
                    }
                    changed = setHistoricalUnavailable(currentRecord, true) || changed;
                    renderLiveCardMeta(currentRecord);
                    syncChatStatus();
                    return changed;
                }
                if (cancelPending || (!vouched && !isTerminalTaskDetail(detail))) {
                    if (currentRecord.isSubagent && detail?.status) missingManagedTaskIds.delete(taskId);
                    return Boolean(reconcileCancelCardFromDetail(currentRecord, taskId, detail) || changed);
                }
                if (vouched) return changed;
                recordTerminalActivity(taskId);
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
            activeDirectActivities.set(k, v);
            restoreCardActivity(liveCardRecords.get(k), v.project_admission_hold);
            markReviewAnchor(liveCardRecords.get(k));
            noteDirectTurn(liveCardRecords.get(k), v.kind !== 'managed_task');
            if (v.kind === 'managed_task') missingManagedTaskIds.delete(k);
            if (v.clientMessageId) pendingSubmissions.delete(v.clientMessageId);
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

    const isKnownProjectFrame = (msg) => {
        const cid = Number(msg?.chat_id ?? 1);
        return state.projectChatIds instanceof Set && state.projectChatIds.has(cid);
    };

    function incrementUnreadIfNeeded(msg) {
        if (!isMain) return;  // the global unread badge tracks the main chat
        // Project visible_revision is the sole unread authority for a Project.
        // Project-owned frames never create a second Main unread.
        if (isKnownProjectFrame(msg)) return;
        if (state.activePage === 'chat') return;
        state.unreadCount++;
        updateUnreadBadge();
    }

    onWs('typing', (msg) => {
        if (!isMyThread(msg)) return;  // each column shows typing only for its own thread
        // A typing frame is a submission receipt, never liveness: it pulls the
        // authoritative census at once and retires the linked `Sending...`
        // only once that census has answered, so the header steps from
        // Sending... straight to Thinking... when the census lists the turn
        // (hydration retires the cmid itself) and to Online otherwise, never
        // through a blank in between. The header derives from that census,
        // pending sends and live cards only.
        const cmid = String(msg.client_message_id || '');
        if (!cmid) return;
        void refreshHeaderControlState(true).then(() => {
            if (pendingSubmissions.delete(cmid)) syncChatStatus();
        });
    });

    const isMyThread = (msg) => {
        return chatThreadAccepts(msg, isMain, chatId, state.projectChatIds);
    };

    const isMyLogThread = (msg) => {
        return chatLogThreadAccepts(msg, isMain, chatId, state.projectChatIds);
    };

    onWs('chat', (msg) => {
        if (!isMyThread(msg)) return;
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
                },
            ));
            if (added) incrementUnreadIfNeeded(msg);
            withStableViewport(() => markIngressSaved(messagesDiv, msg));
            syncChatStatus();
            return;
        }

        if (msg.role === 'assistant' || msg.role === 'system') {
            return withRemoteActivity(() => {
            const explicitTaskId = msg.task_id || '';
            // Every path ends by syncing status; `unread` rows also count toward the badge.
            const settled = (result, unread = result) => {
                if (unread) incrementUnreadIfNeeded(msg);
                syncChatStatus();
                return result;
            };
            const reference = admitCardMetadata(msg);
            if (reference !== undefined) return settled(reference, false);
            const review = attachReviewFromRow(msg, msg.ts || '', true);
            if (review !== undefined) return settled(review, false);
            const cardRow = attachCardRow(msg, msg.ts || '');
            if (cardRow !== undefined) return settled(cardRow);
            if (PROJECT_ROW_TYPES.has(msg.system_type)) {
                return settled(Boolean(addProjectRow(msg, msg.content, { taskId: explicitTaskId })));
            }
            learnSubagentLineage(msg);
            if (msg.is_progress) {
                showTaskIncidentToast(msg);
                const changed = updateLiveCardFromProgressMessage(msg, { grantCancelAuthority: true });
                return settled(changed, false);
            }

            // An early final (post-task still running) is NOT the turn's
            // conclusion; task_done or the queue snapshot concludes it.
            const finalizing = Boolean(explicitTaskId) && msg.task_phase === 'finalizing';
            const typedTerminal = positiveTaskTerminalFact(msg);
            if (!finalizing && (!explicitTaskId || typedTerminal)) {
                if (explicitTaskId) {
                    // 4A (active set): a keyed final concludes ITS OWN turn —
                    // the finished activity + its linked pending — never a
                    // concurrent turn's state (2A keeps later `Sending...`).
                    const finished = activeDirectActivities.get(explicitTaskId);
                    activeDirectActivities.delete(explicitTaskId);
                    if (!REUSABLE_TASK_IDS.has(explicitTaskId)) recordConcludedActivity(explicitTaskId);
                    if (finished?.clientMessageId) {
                        pendingSubmissions.delete(finished.clientMessageId);
                    }
                } else if (msg.system_type !== 'terminal_incident') {
                    // Unkeyed finals clear unscoped state; incidents are informational.
                    activeDirectActivities.clear();
                    pendingSubmissions.clear();
                }
            }

            if (msg.system_type === 'task_summary') {
                const changed = appendTaskSummaryToLiveCard(msg);
                return settled(Boolean(changed));
            }
            // A placed host row is never a child's answer; with no card it stays a System row.
            if (explicitTaskId && !msg.card_row && subagentChildParents.has(explicitTaskId)) {
                return settled(Boolean(routeSubagentFinalMessageToCard(explicitTaskId, msg)));
            }
            let changed = false;
            if (finalizing) changed = markLiveCardFinalizing(explicitTaskId, msg) || changed;
            else if (explicitTaskId && typedTerminal) {
                changed = appendTaskSummaryToLiveCard(msg) || changed;
            }
            const routingCleared = clearTransientRoutingAnnotations(messagesDiv);
            const added = addMessage(msg.content, msg.role, msg.markdown, msg.ts || null, false, {
                systemType: msg.system_type || '',
                source: msg.source || '',
                initiator: msg.initiator || '',
                taskId: explicitTaskId,
                // A placed row whose card is not in this chat keeps its record link.
                evidenceRef: isPlacedCardRow(msg) ? cardRowEvidenceRef(msg) : null,
            });
            return settled(Boolean(added || changed || routingCleared), added || changed);
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
            // A socket write is not durable acceptance (2A): settle the
            // bubble only; `Sending...` retires on authoritative evidence.
            markPendingDelivered(cid);
            syncChatStatus();
        }
    });

    const { appendMediaBubble, appendDocumentBubble, appendLinksMessage, appendQuizMessage } =
        chatMedia.wireDeliveries({
            onWs,
            isMyThread,
            hideTypingIndicatorOnly,
            syncChatStatus,
            incrementUnreadIfNeeded,
            seenMessageKeys,
            rememberMessageKey,
            chatMediaMessageKey,
            documentMessageKey,
            buildQuizCard: chatDecision.buildQuizCard,
            applyQuizStateFrame: chatDecision.applyQuizStateFrame,
            messagesRoot: () => messagesDiv,
            deliverContentMutation: withRemoteActivity,
        });

    let wsHasConnectedOnce = false;

    onWs('open', (msg) => {
        handoffs?.setConnected(true);
        refreshHeaderControlState(true);
        syncChatStatus();
        // The socket's previouslyConnected covers newly mounted Projects too;
        // per-instance state is only the fallback for legacy open events.
        const isReconnect = typeof msg?.previouslyConnected === 'boolean'
            ? msg.previouslyConnected
            : wsHasConnectedOnce;
        const reconnectBanner =
            pendingReconnectBannerText
            || (isReconnect ? '♻️ Reconnected' : '');
        const shouldClearReconnectParams = Boolean(pendingReconnectBannerText);
        pendingReconnectBannerText = '';
        wsHasConnectedOnce = true;
        updateMessagesPadding();
        loadUiPreferences()
            // Reconnect refetches lost task_done; first open shares idle-gated hydration.
            .then(() => (isReconnect
                ? syncHistory({ includeUser: !historyLoaded, fromReconnect: isReconnect })
                : waitForHydrationWindow().then(
                    () => awaitInitialHydration({ includeUser: !historyLoaded }),
                )))
            .then(() => {
                if (reconnectBanner) {
                    addMessage(reconnectBanner, 'system', false, null, false, { ephemeral: true, systemType: 'reconnect' });
                    if (shouldClearReconnectParams) clearPendingReconnectBanner();
                }
            })
            .catch(() => {
                if (reconnectBanner) {
                    addMessage(reconnectBanner, 'system', false, null, false, { ephemeral: true, systemType: 'reconnect' });
                    if (shouldClearReconnectParams) clearPendingReconnectBanner();
                }
            });
    });

    onWs('close', () => {
        handoffs?.setConnected(false);
        hostReady = false;
        hideTypingIndicatorOnly();
        syncChatStatus();
        syncHeaderControlState({ accounting: { available: false } });
    });

    return {
        page,
        chatId,
        projectId,
        // Called by app.js when this instance's panel is (re)shown so a project
        // thread restores its scroll position instead of jumping to the top (P7).
        restoreScrollPosition: reading.request,
        refreshHistory: readReceipt.refresh,
        revealQuestion: (taskId, quizId) => chatDecision.revealQuestion(
            taskId, quizId, projectId, chatId, appendQuizMessage, isInstanceVisible,
            () => { const current = reading.claim(); reading.stick = false; return current; }, reading.scroll),
        cancelHistoryPaint: readReceipt.cancel,
        // app.js fans its already-existing /api/state refresh to every open
        // thread; panels gain convergence without acquiring their own poll.
        hydrateStateSnapshot,
        // True once a history snapshot has actually been fetched and painted;
        // app.js uses it to decide whether a reopen needs a forced repaint.
        hasPaintedHistory: () => historyLoaded && lastHistorySyncSucceeded,
        // Unsendable client-side state (staged File objects / an in-flight
        // upload). app.js must hide, not destroy, an instance holding it.
        hasPendingWork: () => pendingAttachments.length > 0 || attachmentsUploading,
        // Viewport intent stash source for the single-live-panel policy.
        getScrollState: () => {
            if (reading.pending) return reading.export();
            const anchor = serializeTimelineAnchor();
            const tasks = new Set(anchor?.cardChain?.map(entry => entry.taskId) || []);
            const sources = [anchor?.historyId, ...[...tasks].flatMap(id => [...(liveCardRecords.get(id)?.historyIds || [])])];
            const pageId = sources.map(id => [...pageHistoryIds].find(([, ids]) => ids.has(id))?.[0]).find(Boolean);
            return { scrollTop: reading.top, stick: reading.stick,
                history: historyPager.exportResume(pageId), historyAnchor: anchor,
                disclosures: {
                    cards: [...tasks].map(id => [id, liveCardRecords.get(id)?.root.dataset.expanded === '1']),
                    reviews: [...reviewDisclosureByTask].filter(([id]) => tasks.has(id)).map(([id, value]) =>
                        [id, { ...value, expandedGroups: [...value.expandedGroups], expandedAttempts: [...value.expandedAttempts] }]),
                } };
        },
        // Full teardown (P3): release every resource this instance acquired —
        // ws subscriptions, window/document listeners, the ResizeObserver, all
        // timers — then drop the buffered collections and remove the DOM last.
        // Idempotent; late rAF/async continuations no-op on `destroyed`.
        destroy() {
            if (destroyed) return;
            destroyed = true;
            emptyWelcome?.dispose();
            readReceipt.cancel();
            for (const dispose of wsDisposers) {
                try { dispose(); } catch {}
            }
            wsDisposers.length = 0;
            historyControls.endRecent();
            historyPager.destroy();
            disposeReadingGestures();
            document.removeEventListener('selectionchange', retryHistoricalUpserts);
            chatMedia.destroy();
            workPointer?.destroy();
            chatDecision.destroy();
            modelWaits.destroy();
            handoffs?.destroy();
            window.removeEventListener('ouro:page-shown', handlePageShown);
            document.removeEventListener('visibilitychange', handlePageShown);
            if (documentClickHandler) document.removeEventListener('click', documentClickHandler);
            if (documentKeydownHandler) document.removeEventListener('keydown', documentKeydownHandler);
            chatResizeObserver?.disconnect();
            chatResizeObserver = null;
            historyResyncScheduler.cancel();
            if (headerControlInterval) { clearInterval(headerControlInterval); headerControlInterval = null; }
            for (const id of liveCardRecords.keys()) disposeLiveCard(id);
            explicitCardExpansion.clear();
            reviewDisclosureByTask.clear();
            skillReviewDetailStore.clear();
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
