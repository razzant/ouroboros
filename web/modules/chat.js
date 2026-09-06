import { escapeHtmlAttr, escapeHtmlText as escapeHtml } from './utils.js';
import { destroyChatMarkdown, enhanceChatMarkdown, renderChatMarkdown } from './chat_markdown.js';
import { renderPageHeader } from './page_header.js';
import { PAGE_ICONS } from './page_icons.js';
import { showToast } from './toast.js';
import { createSystemMessageAction, renderProjectChip } from './ui_helpers.js';
import { cleanupUploadedAttachments, createChatMedia, showTaskIncidentToast } from './chat_media.js';
import { createChatDecision } from './chat_decision.js';
import { clientSurfaceField } from './client_surface.js';
import { apiClient, apiFetch, fetchTaskDetail, fetchTaskDetailStrict } from './api_client.js';
import {
    OWNER_STOP_DETAIL_MARKER,
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
    taskStoppedWithSummary,
    taskDoneIsTerminal,
    keepStickyExecutorChip,
    taskReasonDetail,
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
import {
    captureLiveCardPhaseState,
    desiredLiveCardPhase,
    replayTerminalPhase,
    restoreLiveCardPhaseState,
    setLiveCardPhase,
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
import { harnessIdentityMarkup } from './harness_presentation.js';
import {
    captureLiveCardProjection,
    createHistoryResyncScheduler,
    createLiveCardBound,
    createLiveCardTimelineRenderer,
    createRebuildBatch,
    createTimelineAnchors,
    insertTimelineNode,
    liveCardProjectionChanged,
    loadOlderControlState,
    nextQuotaEscalation,
    syncLiveCardToggle,
} from './chat_render_batch.js';
import {
    COLLAPSED_ACTIVITY_MAX,
    boundActivityPreview,
    buildTimelineItemHtml,
    buildMessageKey,
    chatLogThreadAccepts,
    chatMediaMessageKey,
    chatThreadAccepts,
    clearStickyCardState,
    confirmAndSendPanic,
    computeDerivedChatStatus,
    computeHydratedDirectActivities,
    documentMessageKey,
    durableChatMediaUrl,
    formatMsgTime,
    getOrCreateChatSessionId,
    headerBudgetPresentation,
    isBackgroundTaskId,
    isFileDrag,
    isForegroundLiveCard,
    isNonTerminalMediaHistoryRow,
    isTerminalTaskPhase,
    loadChatInputHistory,
    liveLineRowToggleKey,
    bindContentButton,
    subagentIdentityTitle,
    subagentTwin,
    mergeStickyCostMeta,
    partitionLocalEchoJournal,
    pendingAttachmentBytes,
    projectCollapsedActivity,
    positiveTaskTerminalFact,
    projectIdFromTask,
    rawTimestampEpoch,
    stampNodeTimestamp,
    reconcileHydratedDirectActivities,
    reconnectBannerText,
    saveChatInputHistory,
    senderLabel,
    shouldAlwaysShowTaskCard,
    shouldFirePanic,
    taskCostMeta,
    taskCostProjection,
    unconfirmedForegroundCardIds,
    withTaskCostMeta,
} from './chat_activity.js';

export {
    COLLAPSED_ACTIVITY_MAX,
    boundActivityPreview,
    chatMediaMessageKey,
    clearStickyCardState,
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

const PROJECT_ROW_TYPES = new Set(['project_started', 'project_completion_summary']);
const CHAT_STORAGE_KEY = 'ouro_chat';
const CHAT_DRAFT_KEY = 'ouro_chat_draft';
const CHAT_INPUT_HISTORY_KEY = 'ouro_chat_input_history';
const MAX_PENDING_ATTACHMENTS = 10;
const MAX_ATTACHMENT_FILE_BYTES = 50 * 1024 * 1024;
const MAX_PENDING_ATTACHMENT_BYTES = 100 * 1024 * 1024;

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
                        <div class="chat-context-mode" id="chat-context-mode" data-context-mode="max" role="group" aria-label="Context size mode" title="Context mode (owner setting). Low fits ~200K / local models; Max is full. Saves immediately; lowering to Low requires Ouroboros to be idle.">
                            <button class="chat-seg" type="button" data-mode="low">Low</button>
                            <button class="chat-seg" type="button" data-mode="max">Max</button>
                        </div>
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
        renderMarkdown: renderChatMarkdown,
        enhanceMarkdown: enhanceMountedMarkdown,
        showToast,
        onDomWrite: withStableViewport,
    });

    async function loadUiPreferences() {
        try {
            const prefs = await apiClient.uiPreferences();
            if (destroyed) return;
            nestedSubagentsExpanded = prefs?.nested_subagents_expanded === true;
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
        if (pendingAttachments.length + incoming.length > MAX_PENDING_ATTACHMENTS) {
            showToast(`Attach up to ${MAX_PENDING_ATTACHMENTS} files per message.`, 'error');
            return;
        }
        const oversized = incoming.find((file) => Number(file.size || 0) > MAX_ATTACHMENT_FILE_BYTES);
        if (oversized) {
            showToast(`Each attachment must be ${Math.round(MAX_ATTACHMENT_FILE_BYTES / (1024 * 1024))} MB or smaller.`, 'error');
            return;
        }
        const incomingBytes = incoming.reduce((total, file) => total + Number(file.size || 0), 0);
        if (pendingAttachmentBytes(pendingAttachments) + incomingBytes > MAX_PENDING_ATTACHMENT_BYTES) {
            const limitMb = Math.round(MAX_PENDING_ATTACHMENT_BYTES / (1024 * 1024));
            showToast(`Attachments are limited to ${limitMb} MB total per message.`, 'error');
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

    // Image paste uses the same stager; only image matches call preventDefault().
    // Timestamped names keep repeated clipboard images distinct.
    input.addEventListener('paste', (e) => {
        const items = e.clipboardData && e.clipboardData.items;
        if (!items) return;
        const pastedImages = [];
        for (let i = 0; i < items.length; i += 1) {
            const item = items[i];
            if (item && item.kind === 'file' && typeof item.type === 'string' && item.type.startsWith('image/')) {
                const blob = item.getAsFile();
                if (!blob) continue;
                const ext = (item.type.split('/')[1] || 'png').split(';')[0].trim() || 'png';
                const ts = Date.now() + i;
                const safeBlob = blob instanceof File
                    ? new File([blob], `clipboard-${ts}.${ext}`, { type: blob.type })
                    : new File([blob], `clipboard-${ts}.${ext}`, { type: item.type });
                pastedImages.push(safeBlob);
            }
        }
        if (!pastedImages.length) return;
        e.preventDefault();
        stagePendingFiles(pastedImages);
    });

    let fileDragDepth = 0;
    function setFileDragActive(active) {
        inputArea.classList.toggle('drag-active', Boolean(active));
    }
    page.addEventListener('dragenter', (event) => {
        if (!isFileDrag(event)) return;
        event.preventDefault();
        fileDragDepth += 1;
        setFileDragActive(true);
    });
    page.addEventListener('dragover', (event) => {
        if (!isFileDrag(event)) return;
        event.preventDefault();
        if (event.dataTransfer) event.dataTransfer.dropEffect = 'copy';
        setFileDragActive(true);
    });
    page.addEventListener('dragleave', (event) => {
        if (!isFileDrag(event)) return;
        fileDragDepth = Math.max(0, fileDragDepth - 1);
        if (fileDragDepth === 0) setFileDragActive(false);
    });
    page.addEventListener('drop', (event) => {
        if (!isFileDrag(event)) return;
        event.preventDefault();
        fileDragDepth = 0;
        setFileDragActive(false);
        stagePendingFiles(event.dataTransfer?.files || []);
    });

    // Pass 1 builds live cards in memory; pass 2 inserts them in transcript order.
    let _syncPass1Active = false;
    // Double-fetch fix: true while syncHistory replays the
    // fetched rows into cards — pass 1, pass 2 AND the terminal-resolution
    // sweep (both the rebuildAll and the routine branch). Finished transitions
    // raised inside that replay must not schedule the 700ms post-completion
    // resync: the data just came from the canonical source. The replay block
    // is fully synchronous, so no live WS frame can ever observe the flag.
    let _historyReplayActive = false;

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
    let historyPaintGeneration = 0;
    // STICKY single-flight hydration promise.
    // Unlike historySyncPromise it survives success, so hydration triggers
    // (bootstrap IIFE, first non-reconnect socket open, refreshHistory without
    // a new revision) short-circuit instead of refetching. Any FAILED sync
    // resets it; scheduleHistorySync and the reconnect path never consult it.
    let initialHydrationPromise = null;
    // highest project revision whose history has been fetched;
    // refreshHistory only bypasses the sticky promise for a NEWER revision.
    let lastLoadedHistoryRevision = 0;
    // one-shot idle gate for Main's deferred first hydration.
    let hydrationGatePromise = null;
    // Detached rebuild batch; live/routine syncs leave it null.
    let _rebuildBatch = null;
    // server window verdict + the explicit Load-older quotas.
    let historyWindow = null;
    let historyQuotaOverride = null;
    let loadingOlderHistory = false;
    let welcomeShown = false;
    // Cross-instance hide/show position; visible mutations use live geometry.
    let _savedScrollTop = Math.max(0, Number(initialScrollState?.scrollTop) || 0);
    let _savedStick = initialScrollState ? initialScrollState.stick !== false : true;
    let _initialScrollPending = Boolean(initialScrollState) && !_savedStick;
    let _hasNewActivity = false;
    let _restoring = false;
    let _viewportMutationDepth = 0;
    const isInstanceVisible = () =>
        Boolean(messagesDiv) && messagesDiv.offsetParent !== null && !document.hidden;
    const LIVE_CARD_CAP = 200;
    const liveCardBound = createLiveCardBound(LIVE_CARD_CAP);
    const liveCardRecords = new Map();
    const markReviewAnchor = (r, on = false) => setReviewAnchor(r, on, setLiveCardPhase);
    const explicitCardExpansion = new Map();
    const reviewDisclosureByTask = new Map();
    const skillReviewDetailStore = new Map();
    const reviewHydrator = createReviewHydrator({
        fetchDetail: fetchTaskDetailStrict,
        applyDetail: (id, detail) => !destroyed && attachTaskDetailReviews(id, detail),
        onState: (id, status) => !destroyed
            && (liveCardRecords.get(id)?.reviewController?.setHydrateStatus?.(status) ?? false),
    });
    // A task_named frame can arrive before the card's record exists; buffer it.
    const pendingSuggestedNames = new Map();
    const taskUiStates = new Map();
    // Decision turns keep activity ordering but never render task cards.
    const ephemeralDecisionTaskIds = new Set();
    // Server-confirmed in-flight direct/ephemeral/managed activities.
    const activeDirectActivities = new Map();
    // Local user submissions awaiting server confirmation (clientMessageId
    // -> { clientMessageId, timestamp }).
    const pendingSubmissions = new Map();
    // Bounded conclusions block late root typing and stale state snapshots; reusable
    // logical task slots are cleared whenever their cycle settles.
    const concludedDirectActivities = new Map();
    const CONCLUDED_ACTIVITY_LEDGER_MAX = 200;
    // Retryable queue-loss candidates plus process-local single-flight reads.
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
            const oldest = concludedDirectActivities.keys().next().value;
            concludedDirectActivities.delete(oldest);
        }
    }
    function recordTerminalActivity(taskId) {
        const id = taskKey(taskId);
        if (!id) return;
        activeDirectActivities.delete(id);
        missingManagedTaskIds.delete(id);
        if (REUSABLE_TASK_IDS.has(id)) concludedDirectActivities.delete(id);
        else recordConcludedActivity(id);
    }
    // Finished task ids hidden from routine syncs until reload/reconnect rebuilds history.
    const retiredTaskIds = new Set();
    // The owner's last main-chat request, handed to the next live card it spawns so a
    // "turn into project" conversion can name the project from it (P1).
    let _pendingCardObjective = '';
    let activeLiveGroupId = '';
    let pendingReconnectSync = false;  // Set when a fromReconnect sync arrives while one is already in-flight.
    let pendingReconnectBannerText = readPendingReconnectBanner();

    function registerEphemeralDecisionFrame(frame) {
        const taskId = taskKey(frame?.task_id);
        if (!taskId) return undefined;
        if (frame?.ephemeral_decision) return withStableViewport(
            () => registerEphemeralDecisionFrameMutation(taskId),
        );
        return ephemeralDecisionTaskIds.has(taskId) ? false : undefined;
    }

    function registerEphemeralDecisionFrameMutation(taskId) {
        ephemeralDecisionTaskIds.add(taskId);
        const taskState = taskUiStates.get(taskId);
        if (taskState?.cleanupTimer) clearTimeout(taskState.cleanupTimer);
        taskUiStates.delete(taskId);
        const record = liveCardRecords.get(taskId);
        const changed = Boolean(record?.root?.isConnected);
        if (record) {
            record.root?.remove();
            liveCardRecords.delete(taskId);
        }
        pendingSuggestedNames.delete(taskId);
        if (activeLiveGroupId === taskId) activeLiveGroupId = '';
        return changed;
    }

    function readPendingReconnectBanner() {
        try {
            const url = new URL(window.location.href);
            return reconnectBannerText(url.searchParams.get('_ouro_reason') || '');
        } catch {
            return '';
        }
    }

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
        if (_rebuildBatch || !statusBadge) return;
        statusBadge.className = `status-badge ${kind}`;
        statusBadge.textContent = text;
    }

    function syncHeaderControlState(data) {
        headerActions?.querySelectorAll('[data-chat-command]').forEach((button) => {
            const cmd = button.dataset.chatCommand;
            if (cmd === 'evolve') {
                button.classList.toggle('on', !!data?.evolution_enabled);
                if (data?.evolution_state?.detail) button.title = data.evolution_state.detail;
            } else if (cmd === 'bg') {
                button.classList.toggle('on', !!data?.bg_consciousness_enabled);
                if (data?.bg_consciousness_state?.detail) button.title = data.bg_consciousness_state.detail;
            }
        });
        // Mark More while background mode is active in the menu.
        const moreSummary = headerActions?.querySelector('.chat-header-more > summary');
        if (moreSummary) {
            const anyActive = !!data?.evolution_enabled || !!data?.bg_consciousness_enabled;
            moreSummary.classList.toggle('has-active', anyActive);
        }
        const ctxBtn = byId('context-mode');
        if (ctxBtn && typeof data?.context_mode === 'string') {
            ctxBtn.dataset.contextMode = data.context_mode === 'low' ? 'low' : 'max';
        }
        const budget = headerBudgetPresentation(data);
        const budgetText = byId('budget-text');
        const budgetFill = byId('budget-bar-fill');
        if (budgetText) budgetText.textContent = budget.label;
        if (budgetFill) budgetFill.style.width = `${budget.fillPct}%`;
    }

    function hydrateStateSnapshot(data, snapshotRequestedAt = Infinity, snapshotGeneration = 0) {
        syncHeaderControlState(data);
        const activities = Array.isArray(data?.active_chat_activities)
            ? data.active_chat_activities
            : data?.active_direct_turns;
        if (Array.isArray(activities)) {
            hydrateDirectActivities(activities, snapshotRequestedAt, snapshotGeneration);
        }
    }

    async function refreshHeaderControlState(force = false) {
        if (!force && state.activePage !== 'chat') return;
        const request = stateSnapshots.begin();
        try {
            const resp = await apiFetch('/api/state', { cache: 'no-store' });
            if (!resp.ok) {
                if (stateSnapshots.isCurrent(request)) {
                    syncHeaderControlState({ accounting: { available: false } });
                }
                return;
            }
            const data = await resp.json();
            stateSnapshots.apply(request, data);
        } catch {
            if (stateSnapshots.isCurrent(request)) {
                syncHeaderControlState({ accounting: { available: false } });
            }
        }
    }

    function persistVisibleHistory() {
        try {
            sessionStorage.setItem(storeKey(CHAT_STORAGE_KEY), JSON.stringify(persistedHistory.slice(-200)));
        } catch {}
    }

    const NEAR_BOTTOM_THRESHOLD_PX = 48;
    const ACTUAL_BOTTOM_TOLERANCE_PX = 6;

    function isNearBottom(threshold = NEAR_BOTTOM_THRESHOLD_PX) {
        const remaining = messagesDiv.scrollHeight - messagesDiv.scrollTop - messagesDiv.clientHeight;
        return remaining <= threshold;
    }

    const { captureVisibleTimelineAnchor, restoreVisibleTimelineAnchor } =
        createTimelineAnchors({ messagesDiv, liveCardRecords });

    function withStableViewport(mutate, {
        forceFollow = false,
        remoteContent = false,
        excludeAnchorNode = null,
    } = {}) {
        if (typeof mutate !== 'function') return undefined;
        if (destroyed) return false;
        if (_viewportMutationDepth > 0) return mutate();
        if (_restoring || !isInstanceVisible()) {
            const result = mutate();
            if (remoteContent && result && !_savedStick) _hasNewActivity = true;
            return result;
        }

        const followBottom = forceFollow || isNearBottom();
        const anchor = followBottom ? null : captureVisibleTimelineAnchor(excludeAnchorNode);
        // Pre-mutation geometry, not the mutate() return, decides restore/
        // follow (survives throws and lying change-flags); booleans keep only
        // the activity-marker and write-idempotence duties.
        const preScrollHeight = messagesDiv.scrollHeight;
        const preScrollTop = messagesDiv.scrollTop;
        let result;
        _viewportMutationDepth = 1;
        try {
            result = mutate();
            return result;
        } finally {
            _viewportMutationDepth = 0;
            if (isInstanceVisible()) {
                if (followBottom) {
                    if (forceFollow || messagesDiv.scrollHeight !== preScrollHeight) {
                        messagesDiv.scrollTop = messagesDiv.scrollHeight;
                    } else if (messagesDiv.scrollTop !== preScrollTop) {
                        // Engine drift on a no-op frame: put the reader back.
                        messagesDiv.scrollTop = preScrollTop;
                    }
                } else {
                    // Restores the captured offset; a zero-delta frame re-lands
                    // on the same position.
                    restoreVisibleTimelineAnchor(anchor);
                }
                if (remoteContent && result && !followBottom) _hasNewActivity = true;
                _savedScrollTop = messagesDiv.scrollTop;
                _savedStick = isNearBottom();
                updateScrollButton();
            }
        }
    }

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
        });
    }

    const {
        renderLiveCardTimeline,
        appendTimelineItem,
        patchLastTimelineItem,
        patchTimelineItemAt,
    } = createLiveCardTimelineRenderer({ withStableViewport, buildTimelineItemHtml });

    function insertMessageNode(node, options = {}) {
        if (!node) return false;
        // rebuildAll only: collect into the detached batch. One
        // stable sort + one fragment mount replace per-row chronological
        // insertion; the end-of-sync anchor restore replaces the per-row
        // insertedAboveViewport compensation. Routine syncs and live frames
        // (batch inactive) keep the chronological insertTimelineNode path.
        if (_rebuildBatch) {
            _rebuildBatch.collect(node);
            return true;
        }
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
        if (!taskState) return false;
        if (taskState.cardVisible) {
            return reanchorVisibleTaskCard(taskState, rawTs, { suppressDomInsert });
        }
        if (!(taskState.forceCard || taskState.toolCalls > 0 || shouldAlwaysShowTaskCard(taskState.taskId))) {
            return false;
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
        const before = captureLiveCardProjection(record);
        ensureLiveCardVisible(record, { suppressDomInsert, reorderExisting: anchorMovedEarlier });
        let changed = liveCardProjectionChanged(before, record);
        const bufferedUpdates = [...taskState.bufferedLiveUpdates];
        taskState.bufferedLiveUpdates = [];
        for (const update of bufferedUpdates) {
            changed = applyLiveCardState(update.summary, taskState.taskId, update.ts, update.dedupeKey, {
                suppressDomInsert,
                rawTs: update.rawTs,
            }) || changed;
        }
        if (taskState.completed) {
            changed = finishLiveCard(taskState.taskId, taskState.completedPhase || 'done') || changed;
        }
        return changed;
    }

    function markTaskToolCall(taskId, count = 1, minimumOnly = false, rawTs = '') {
        const taskState = getTaskUiState(taskId, true);
        if (!taskState) return false;
        const safeCount = Math.max(0, Number(count) || 0);
        if (minimumOnly) {
            taskState.toolCalls = Math.max(taskState.toolCalls, safeCount);
        } else {
            taskState.toolCalls += safeCount;
        }
        return revealBufferedCardIfNeeded(taskState, { rawTs });
    }

    function forceTaskCard(taskId, rawTs = '') {
        const taskState = getTaskUiState(taskId, true);
        if (!taskState) return null;
        taskState.forceCard = true;
        revealBufferedCardIfNeeded(taskState, { rawTs });
        return taskState;
    }

    function forceTaskCardVisibleChange(taskId, rawTs = '') {
        const before = captureLiveCardProjection(liveCardRecords.get(taskId));
        forceTaskCard(taskId, rawTs);
        return liveCardProjectionChanged(before, liveCardRecords.get(taskId));
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
        const resolvedTaskId = taskId || activeLiveGroupId || '';
        if (!resolvedTaskId) return false;
        const taskState = getTaskUiState(resolvedTaskId, true);
        if (!taskState) return false;
        let changed = reanchorVisibleTaskCard(taskState, rawTs);
        if (taskState.completed && !isTerminalTaskPhase(summary.phase || '', summary.terminal)) {
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
                    changed = Boolean(oldRec.root?.isConnected) || changed;
                    oldRec.root?.remove();
                    liveCardRecords.delete(resolvedTaskId);
                }
                retiredTaskIds.delete(resolvedTaskId);
            } else {
                return changed;
            }
        }
        if (summary.phase === 'error' || summary.phase === 'timeout' || (summary.terminal && summary.phase === 'warn')) {
            taskState.forceCard = true;
        }
        if (!taskState.cardVisible) {
            bufferLiveUpdate(taskState, summary, ts, dedupeKey, rawTs);
            const revealed = revealBufferedCardIfNeeded(taskState, { rawTs });
            return Boolean(changed || revealed);
        }
        const applied = applyLiveCardState(summary, resolvedTaskId, ts, dedupeKey, { rawTs });
        return Boolean(changed || applied);
    }

    async function turnTaskIntoProject(record) {
        if (!record || record.root?.dataset?.projectCreating === '1' || record.root?.dataset?.projectCreated === '1') return;
        const taskId = taskKey(record.groupId);
        const projectId = projectIdFromTask(taskId);
        record.root.dataset.projectCreating = '1';
        const actions = record.turnProjectBtn?.parentElement || record.root.querySelector('.chat-live-actions');
        if (actions) {
            withStableViewport(() => {
                actions.innerHTML = '<button type="button" class="btn btn-xs btn-default" disabled>Creating project…</button>';
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
                    actions.innerHTML = '<button type="button" class="btn btn-xs btn-default" data-turn-into-project>Turn into project</button>';
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

    // Only host-attested cancelable queue roots receive this control.
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
        if (!record?.root) return false;
        const eligible = cancelRunEligibility({
            groupId: record.groupId,
            isSubagent: record.isSubagent,
            finished: record.finished,
            cancelable: cancelableTaskIds.has(record.groupId),
            converted: record.root.dataset.projectCreated === '1',
        });
        const existing = record.root.querySelector('[data-cancel-run]');
        if (!eligible) {
            if (!existing) return false;
            existing?.remove();
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

    // Early final stays live while post-task synthesis runs.
    function markLiveCardFinalizing(taskId = '') {
        return withStableViewport(() => {
            const record = liveCardRecords.get(taskKey(taskId));
            if (!record || record.finished || !record.phaseEl) return false;
            const anchored = markReviewAnchor(record);
            if (record.cancelPendingPolicy) return anchored;
            record.finalizingHold = true;
            const phased = setLiveCardPhase(
                record, 'working', 'Finalizing…', 'chat-live-phase working finalizing',
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
                cancelableTaskIds.delete(taskId);
                record.cancelable = false;
                syncCancelRunButton(record);
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
                    ) || changed;
                }
                return changed;
            });
        }
    }

    function markTaskCancelable(taskId = '') {
        const id = taskKey(taskId);
        if (!id || cancelableTaskIds.has(id)) return false;
        cancelableTaskIds.add(id);
        const record = liveCardRecords.get(id);
        return record ? syncCancelRunButton(record) : false;
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
        const chip = renderProjectChip({
            name: String(project.name || project.id || 'Project').trim(),
            status: 'running in background ↗',
            onClick: () => window.dispatchEvent(new CustomEvent('ouro:open-project', { detail: { project } })),
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

    // A brief composer brighten when a task leaves the main chat for a project —
    // a calm "you're free to start something else" signal (P3). Self-clearing.
    let _chatFreedTimer = null;
    function signalChatFreed() {
        const row = page.querySelector('.chat-text-row');
        if (!row) return;
        row.classList.add('chat-freed');
        if (_chatFreedTimer) clearTimeout(_chatFreedTimer);
        _chatFreedTimer = setTimeout(() => row.classList.remove('chat-freed'), 900);
    }

    const reviewAnchorEligible = (id) => !liveCardRecords.has(id)
        && !taskUiStates.has(id) && !activeDirectActivities.has(id);

    function attachReviewGroup(group, rawTs = '') {
        const ownerTaskId = taskKey(group?.presentationOwnerTaskId);
        if (!ownerTaskId) return false;
        if (retiredTaskIds.has(ownerTaskId) && !liveCardRecords.has(ownerTaskId)) return false;
        const reviewAnchor = reviewAnchorEligible(ownerTaskId);
        const ownerState = forceTaskCard(ownerTaskId, rawTs);
        if (!ownerState?.cardVisible) return false;
        const record = liveCardRecords.get(ownerTaskId);
        if (reviewAnchor) markReviewAnchor(record, true);
        const merged = record?.reviewController?.update(group);
        const wasMounted = Boolean(record?.root?.isConnected);
        if (!_syncPass1Active) ensureLiveCardVisible(record);
        const mounted = !wasMounted && Boolean(record?.root?.isConnected);
        if (merged) {
            if (rawTs) reanchorVisibleTaskCard(ownerState, rawTs);
            if (!record.reviewOwnerDetailObserved) {
                record.reviewOwnerDetailObserved = true;
                observeMissingManagedTask(
                    ownerTaskId,
                    _remoteActivityDepth > 0 ? withRemoteActivity : withStableViewport,
                );
            }
        }
        return Boolean(merged || mounted);
    }

    function attachTaskDetailReviews(taskId, detail) {
        return withStableViewport(() => {
            const id = taskKey(taskId);
            const groups = reviewGroupsFromTaskDetail(detail, id);
            if (!id || groups.length === 0) return false;
            if (!liveCardRecords.has(id)) forceTaskCard(id, detail?.ts || detail?.timestamp || '');
            const record = liveCardRecords.get(id);
            if (!record?.reviewController) return false;
            const changed = record.reviewController.updateMany(groups);
            const reconciled = reconcileCancelCardFromDetail(record, id, detail);
            return Boolean(changed || reconciled);
        });
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

    function handleReviewReference(row) {
        const reference = reviewReferenceFromRow(row);
        if (!reference) return undefined;
        return withStableViewport(() => {
            const owner = reference.presentationOwnerTaskId;
            const anchor = reviewAnchorEligible(owner);
            const wasVisible = Boolean(liveCardRecords.get(owner)?.root?.isConnected);
            forceTaskCard(owner, row?.ts);
            const record = liveCardRecords.get(owner);
            const mounted = !wasVisible && Boolean(record?.root?.isConnected);
            const anchored = anchor && markReviewAnchor(record, true);
            hydrateCardReviews(owner, reference.stateRevision);
            return Boolean(mounted || anchored);
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
            ? `<div class="chat-live-actions"><button type="button" class="btn btn-xs btn-default" data-turn-into-project>Turn into project</button></div>`
            : '';
        root.innerHTML = `
            <div class="chat-live-summary-button" role="button" tabindex="0" data-live-summary-button aria-expanded="false" aria-controls="${escapeHtmlAttr(timelineId)}">
                <div class="chat-live-summary">
                    <div class="chat-live-summary-main">
                        <span class="chat-live-phase working" data-live-phase role="status" aria-live="polite" aria-atomic="true" aria-label="${options.isSubagent ? 'Subagent' : 'Task'} status: Working">Working</span>
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
            ${projectActionHtml}
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
            turnProjectBtn: root.querySelector('[data-turn-into-project]'),
            // P5: "Cancel run" button element (rendered lazily by syncCancelRunButton
            // once the host-attested cancelable marker is known for this task).
            cancelRunBtn: null,
            timelineEl: root.querySelector('[data-live-timeline]'),
            reviewsHostEl: root.querySelector('[data-live-reviews-host]'),
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
            // collapsed timelines defer DOM building; the flag says
            // the rendered timeline DOM is stale relative to record.items.
            _timelineDirty: false,
            // last frame's summary meta strings — meta renders from
            // record state (renderLiveCardMeta), once per card in a batch.
            _lastFrameMeta: [],
            // The owner's request that spawned this card (main, non-subagent only),
            // used to name a project on "turn into project" when the server has no
            // title/objective yet (P1, direct-chat conversion). One-shot handoff.
            objectiveHint: (isMain && !options.isSubagent) ? _pendingCardObjective : '',
            // The proactively-coined LLM name; becomes the card title when set.
            suggestedName: '',
            // P1: last bounded activity projection (remembered even while
            // the collapsed line is suppressed on unnamed root cards) + sticky cost.
            collapsedActivity: '',
            costMeta: null,
            reviewOwnerDetailObserved: false,
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
        if (isMain && !options.isSubagent) _pendingCardObjective = '';
        bindContentButton(record.summaryButtonEl, () => {
            const nowExpanded = record.root.dataset.expanded !== '1';
            explicitCardExpansion.set(record.groupId, nowExpanded);
            setLiveCardExpanded(record, nowExpanded);
            if (nowExpanded) hydrateCardReviews(record.groupId);
        });
        record.turnProjectBtn?.addEventListener('click', (event) => {
            event.stopPropagation();
            turnTaskIntoProject(record);
        });
        record.timelineEl?.addEventListener('click', (event) => {
            const button = event.target.closest('[data-live-line-toggle]');
            // Row-surface disclosure: any click on the line's
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

    function getLiveCardRecord(groupId = '') {
        const normalizedGroupId = groupId || activeLiveGroupId || 'chat';
        return liveCardRecords.get(normalizedGroupId) || createLiveCardRecord(normalizedGroupId);
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
                const oldest = pendingSuggestedNames.keys().next().value;
                pendingSuggestedNames.delete(oldest);
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

    // Full narration stays in the timeline, not a mouse-only title.
    function renderCollapsedActivity(record, text) {
        if (!record?.activityEl) return false;
        const changed = record.activityEl.textContent !== text
            || Boolean(record.activityEl.hasAttribute?.('title'));
        if (record.activityEl.textContent !== text) record.activityEl.textContent = text;
        if (record.activityEl.hasAttribute?.('title')) record.activityEl.removeAttribute('title');
        return Boolean(changed && record.activityEl.isConnected);
    }

    function ensureSubagentContainer(parentId = '') {
        if (!parentId) return null;
        const parentRecord = getLiveCardRecord(parentId);
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

    function getSubagentCardRecordMutation(childId = '', parentId = '', role = '') {
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
        }
        if (record.root.dataset.parentTaskId !== parentId) record.root.dataset.parentTaskId = parentId;
        if (record.root.dataset.subagentRole !== record.subagentRole) {
            record.root.dataset.subagentRole = record.subagentRole;
        }
        if (existing && !explicitCardExpansion.has(childId)) {
            setLiveCardExpanded(record, nestedSubagentsExpanded);
        }
        const container = ensureSubagentContainer(parentId);
        if (container && record.root.parentNode !== container) {
            container.appendChild(record.root);
        }
        const parentRecord = liveCardRecords.get(parentId);
        if (parentRecord) updateLiveCardCount(parentRecord);
        return record;
    }

    function setLiveCardTypingVisible(record, visible) {
        if (!record?.inlineTypingEl) return false;
        const display = visible ? '' : 'none';
        if (record.inlineTypingEl.style.display === display) return false;
        record.inlineTypingEl.style.display = display;
        return Boolean(record.inlineTypingEl.isConnected);
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
        // one count render per card at the end of a replay batch.
        if (_rebuildBatch) {
            _rebuildBatch.touch(record);
            return;
        }
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

    // Re-showing a hidden instance still restores its saved viewport. Live-card
    // geometry itself has no deferred layout work; every real DOM mutation is
    // stabilized at its write boundary.
    const handlePageShown = (event) => {
        if (
            event?.detail?.page === 'chat'
            || (event?.type === 'visibilitychange' && !document.hidden)
        ) restoreScrollPosition();
    };
    window.addEventListener('ouro:page-shown', handlePageShown);
    document.addEventListener('visibilitychange', handlePageShown);

    // P3: fetch the genuinely-full text of a server-truncated timeline line (the WS
    // preview is capped at 4000 chars) on demand, not over the socket; cache it on
    // the item, re-render if the line is still expanded, and show it in a
    // bounded-scroll box. Best-effort: the capped preview stays on failure.
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
                const hadFocus = Boolean(
                    document.activeElement?.closest?.(`[data-live-line-key="${(window.CSS && CSS.escape) ? CSS.escape(item.lineKey) : item.lineKey}"]`),
                );
                renderLiveCardTimeline(record);
                if (hadFocus) {
                    record.timelineEl
                        ?.querySelector(`[data-live-line-toggle="${(window.CSS && CSS.escape) ? CSS.escape(item.lineKey) : item.lineKey}"]`)
                        ?.focus?.({ preventScroll: true });
                }
            }
        }
    }

    function scheduleHistorySync() {
        historyResyncScheduler.schedule(liveCardBound.isArmed());
    }

    const historyResyncScheduler = createHistoryResyncScheduler({
        isReplayActive: () => _historyReplayActive,
        // A joined run's timer was spent on a window fetched before the arm: re-arm.
        run: () => syncHistory({ includeUser: false }).catch(() => {}).then(() => {
            if (!destroyed && lastHistorySyncSucceeded && liveCardBound.isArmed()) scheduleHistorySync();
        }),
    });

    // The 12 cost-meta keys shared by both subagent whitelists (the delegation
    // trio stays inline in each literal — the wire test scans those literals).
    function costMetaKeys(src) {
        return {
            cost_usd: src?.cost_usd,
            accounted_upper_bound_usd: src?.accounted_upper_bound_usd,
            accounted_upper_bound_usd_with_children: src?.accounted_upper_bound_usd_with_children,
            cost_accounting_status: src?.cost_accounting_status,
            cost_accounting_error: src?.cost_accounting_error,
            cost_final: src?.cost_final,
            cost_usd_with_children: src?.cost_usd_with_children,
            cost_with_children_partial: src?.cost_with_children_partial,
            reserved_usd: src?.reserved_usd,
            unresolved_upper_bound_usd: src?.unresolved_upper_bound_usd,
            unknown_unmetered: src?.unknown_unmetered,
            non_final_rows: src?.non_final_rows,
        };
    }

    // the ONE meta-line renderer, fed entirely from record state,
    // so a replay batch renders it exactly once per card.
    function renderLiveCardMeta(record) {
        if (!record?.metaEl) return false;
        const executorChipHtml = record.executorChip
            ? `<span class="harness-chip chat-live-executor-chip" title="${escapeHtmlAttr(record.executorChip.title || '')}">`
              + harnessIdentityMarkup(record.executorChip.harness, {
                  label: record.executorChip.label || '',
                  className: 'chat-live-executor-identity',
              })
              + '</span>'
            : '';
        const html = executorChipHtml + [
            record.groupId === 'bg-consciousness' ? 'Background thinking' : '',
            ...(Array.isArray(record._lastFrameMeta) ? record._lastFrameMeta : []),
            ...((record.costMeta && Array.isArray(record.costMeta.meta)) ? record.costMeta.meta : []),
            record.latestActivityTs ? `updated ${record.latestActivityTs}` : '',
        ].filter(Boolean).map((item) => `<span class="chat-live-meta-text">${escapeHtml(item)}</span>`).join(' · ');
        if (record.metaEl.innerHTML === html) return false;
        record.metaEl.innerHTML = html;
        return Boolean(record.metaEl.isConnected);
    }

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
        if (record.finished && !isTerminalTaskPhase(nextPhase, summary.terminal)) {
            if (summary.costProjection) {
                record.costMeta = mergeStickyCostMeta(record.costMeta, summary.costProjection);
                if (_rebuildBatch) _rebuildBatch.touch(record);
                else renderLiveCardMeta(record);
                return liveCardProjectionChanged(before, record);
            }
            return false;
        }
        markReviewAnchor(record);

        if (!record.isSubagent) {
            activeLiveGroupId = nextGroupId;
            reanchorTaskCard(record, rawTs, { suppressDomInsert });
        }
        ensureLiveCardVisible(record, { suppressDomInsert });
        record.updates += 1;
        const wasFinished = record.finished;
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
        if (summary.human && headline) {
            record.lastHumanHeadline = headline;
        }

        const shouldPromote = Boolean(summary.promote) || record.finished;
        const activeHeadline = shouldPromote
            ? headline
            : (record.lastHumanHeadline
                || (record.updates > 1 ? record.titleEl.textContent : '')
                || 'Working...');
        const activePhase = record.finished
            ? (summary.phase || 'done')
            : (shouldPromote ? (summary.phase || 'working') : (record.phaseEl.dataset.phase || 'working'));

        const desiredPhase = desiredLiveCardPhase(record, activePhase);
        setLiveCardPhase(record, desiredPhase.phase, desiredPhase.text, desiredPhase.className);
        // A coined project name takes the title slot (the activity headline stays in the
        // timeline); a child's title is its lineage identity; otherwise the activity headline.
        const title = record.suggestedName || (record.isSubagent ? childTitle(record) : activeHeadline);
        if (record.titleEl.textContent !== title) record.titleEl.textContent = title;
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
            // Full-array dedup keeps routine history syncs from growing Notes.
            const existingIdx = record.items.findIndex((it) => it.dedupeKey === syntheticKey);
            if (existingIdx !== -1 && inPlaceByKey) {
                const it = record.items[existingIdx];
                const patch = {
                    phase: summary.phase || it.phase,
                    headline: headline || it.headline,
                    fullHeadline: summary.fullHeadline || headline || it.fullHeadline,
                    body: summary.body || '',
                    fullBody: summary.fullBody || summary.body || it.fullBody || '',
                    fullRef: summary.fullRef || it.fullRef || '',
                    truncated: summary.truncated || it.truncated || false,
                    ts: ts || it.ts,
                };
                if (Object.entries(patch).some(([key, value]) => it[key] !== value)) {
                    Object.assign(it, patch);
                    patchIndex = existingIdx;
                    timelineUpdate = 'patch-at';
                } else {
                    timelineUpdate = 'duplicate-skip';
                }
            } else if (existingIdx === lastIdx && existingIdx !== -1) {
                const it = record.items[existingIdx];
                const patch = {
                    ts: ts || it.ts,
                    fullHeadline: summary.fullHeadline || it.fullHeadline || it.headline,
                    fullBody: summary.fullBody || it.fullBody || it.body,
                    fullRef: summary.fullRef || it.fullRef || '',
                    truncated: summary.truncated || it.truncated || false,
                };
                if (Object.entries(patch).every(([key, value]) => it[key] === value)) {
                    timelineUpdate = 'duplicate-skip';
                } else {
                    Object.assign(it, patch);
                    it.count += 1;
                    timelineUpdate = 'patch-last';
                }
            } else if (existingIdx !== -1) {
                // An older duplicate only refreshes its timestamp.
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
        // Cost-only bookkeeping does not move the activity clock.
        if (ts && (summary.human || activityCandidate)) record.latestActivityTs = ts;
        if (summary.costProjection) {
            record.costMeta = mergeStickyCostMeta(record.costMeta, summary.costProjection);
        }
        if (summary.executorChip
                && !keepStickyExecutorChip(record.executorChip, summary.executorChip)) {
            record.executorChip = summary.executorChip;
        }
        record._lastFrameMeta = Array.isArray(summary.meta) ? summary.meta : [];
        if (_rebuildBatch) _rebuildBatch.touch(record);
        else renderLiveCardMeta(record);
        const lastItem = record.items[record.items.length - 1];
        if (timelineUpdate === 'append' && lastItem) {
            timelineChanged = appendTimelineItem(lastItem, record);
        } else if (timelineUpdate === 'patch-last' && lastItem) {
            timelineChanged = patchLastTimelineItem(lastItem, record);
        } else if (timelineUpdate === 'patch-at' && patchIndex !== -1) {
            timelineChanged = patchTimelineItemAt(record.items[patchIndex], record);
        }
        ensureLiveCardVisible(record, { suppressDomInsert });
        hideTypingIndicatorOnly();
        const justFinished = record.finished && !wasFinished;
        const drivesComposerStatus = !isBackgroundTaskId(nextGroupId);
        // P5: a finished card must not keep offering "Cancel run". A log-channel
        // task_done terminates the card HERE without passing finishLiveCard, so
        // the cancelable marker must be dropped on this path too (P3 growth cap).
        if (justFinished) {
            record.root.dataset.finished = '1';
            cancelableTaskIds.delete(record.groupId);
            syncCancelRunButton(record);
        }
        if (record.finished) {
            setLiveCardTypingVisible(record, false);
            markTaskComplete(nextGroupId, summary.phase || 'done');
            if (justFinished) {
                scheduleHistorySync();
            }
            syncLiveCardToggle(record);
            if (drivesComposerStatus) syncChatStatus();
        } else {
            setLiveCardTypingVisible(record, true);
            if (drivesComposerStatus || !hasActiveLiveCard()) {
                syncChatStatus();
            }
        }
        return Boolean(timelineChanged
            || typingBefore !== typingEl.style.display
            || liveCardProjectionChanged(before, record));
    }

    function finishLiveCard(groupId = '', phase = '') {
        return withStableViewport(() => finishLiveCardMutation(groupId, phase));
    }

    function finishLiveCardMutation(groupId = '', phase = '') {
        const record = groupId
            ? liveCardRecords.get(groupId)
            : (activeLiveGroupId ? liveCardRecords.get(activeLiveGroupId) : null);
        if (!record) return false;
        // A converted card is a terminal project chip now — ignore late terminal
        // frames so they neither overwrite the chip nor touch its element refs (T4).
        if (record.root?.dataset?.projectCreated === '1') return false;
        const before = captureLiveCardProjection(record);
        const typingBefore = typingEl.style.display;
        markReviewAnchor(record);
        const wasFinished = record.finished;
        record.finished = true;
        record.finalizingHold = false;
        // A finished task can never be cancelled again; dropping the marker here
        // keeps the set from accumulating every task id of a long session (P3).
        cancelableTaskIds.delete(record.groupId);
        syncCancelRunButton(record);
        const presentation = taskPresentation(phase || 'done');
        const activePhase = presentation.phase;
        setLiveCardPhase(record, activePhase, presentation.headline);
        if (record.isSubagent) record.titleEl.textContent = childTitle(record);
        else if (!record.suggestedName && !record.lastHumanHeadline
                && record.titleEl.textContent !== presentation.headline) {
            record.titleEl.textContent = presentation.headline;
        }
        setLiveCardTypingVisible(record, false);
        markTaskComplete(record.groupId, activePhase);
        if (!wasFinished) {
            record.root.dataset.finished = '1';
            scheduleHistorySync();
        }
        syncLiveCardToggle(record);
        if (activeLiveGroupId === record.groupId) activeLiveGroupId = '';
        syncChatStatus();
        return Boolean(typingBefore !== typingEl.style.display
            || liveCardProjectionChanged(before, record));
    }

    function appendTaskSummaryToLiveCard(msg, { suppressDomInsert = false } = {}) {
        const taskId = msg?.task_id || activeLiveGroupId || '';
        const rawTs = msg?.ts || new Date().toISOString();
        const ephemeral = registerEphemeralDecisionFrame(msg);
        if (ephemeral !== undefined) return ephemeral;
        if (!taskId) {
            return finishLiveCard(taskId, 'done');
        }
        let changed = false;
        // Restore task name from history.
        if (msg?.suggested_name) {
            changed = applySuggestedName(taskId, msg.suggested_name) || changed;
        }
        const finalizing = msg?.task_phase === 'finalizing' || msg?.outcome_final === false;
        const projectedReviews = reviewGroupsFromTaskDetail(msg, taskId);
        const hasAcceptanceReview = projectedReviews.length > 0;
        const taskState = getTaskUiState(taskId, hasAcceptanceReview || finalizing);
        if (!taskState) {
            return finishLiveCard(taskId, 'done') || changed;
        }
        if (hasAcceptanceReview) taskState.forceCard = true;
        changed = revealBufferedCardIfNeeded(taskState, { suppressDomInsert, rawTs }) || changed;
        if (!taskState.cardVisible) {
            if (!finalizing) markAssistantReply(taskId);
            return changed;
        }
        const presentation = taskPresentation(finalizing ? 'working' : taskTerminalPhase(msg || {}));
        // P5: a cancelled root says "Cancelled", never a generic "Done" headline.
        // №8/Q3: an owner-requested soft stop is a SUCCESS — its own headline,
        // never warn-styled, with the owner-request marker in the details.
        const softStopped = taskStoppedWithSummary(msg || {});
        const softStopDetail = softStopped ? OWNER_STOP_DETAIL_MARKER : '';
        const reasonDetail = taskReasonDetail(msg || {});
        const record = liveCardRecords.get(taskId);
        changed = Boolean(record?.reviewController?.updateMany(projectedReviews)) || changed;
        if (finalizing && record && !record.finished) record.finalizingHold = true;
        changed = applyLiveCardState(
            {
                phase: presentation.phase,
                headline: presentation.headline,
                body: [softStopDetail, reasonDetail].filter(Boolean).join('\n'),
                visible: Boolean(softStopDetail || reasonDetail),
                human: false,
                promote: true,
                terminal: !finalizing,
                costProjection: taskCostProjection(msg, rawTs),
            },
            taskId,
            normalizeLogTs(rawTs),
            `task_done|${taskId}`,
            { suppressDomInsert, rawTs },
        ) || changed;
        if (finalizing) return changed;
        changed = finishLiveCard(taskId, presentation.phase) || changed;
        scheduleTaskUiCleanup(taskState);
        return changed;
    }

    // child task_id -> { parentId, role, model } from subagent lifecycle pings. Child
    // cards mount under the parent, but their phase/terminal state is independent
    // (a finished child never marks the parent done); a later model-less event keeps
    // the previously seen model so the "role · model" headline survives.
    const subagentChildParents = new Map();
    // Children whose card reached a terminal phase: late non-lifecycle progress
    // must NOT revive it back to "working".
    const subagentTerminalChildren = new Set();

    function setSubagentParent(childId, { parentId = '', role = '', model = '' } = {}) {
        const prev = subagentChildParents.get(childId) || {};
        const next = {
            parentId: parentId || prev.parentId || '',
            role: role || prev.role || '',
            model: taskKey(model) || prev.model || '',
        };
        if (['parentId', 'role', 'model'].every((k) => next[k] === prev[k])) return;
        subagentChildParents.set(childId, next);
        for (const sid of subagentChildParents.keys()) {
            const rec = liveCardRecords.get(sid);
            // Write only on change: a rewrite would destroy a selection being copied.
            const next = rec?.isSubagent ? childTitle(rec) : '';
            if (next && rec.titleEl.textContent !== next) rec.titleEl.textContent = next;
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
        const replayTerminal = msg.task_terminal_status
            ? taskDoneIsTerminal({ ...msg, status: String(msg.task_terminal_status) })
            : false;
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
        const taskId = msg?.task_id || activeLiveGroupId || '';
        const rawTs = msg?.ts || new Date().toISOString();
        const ephemeral = registerEphemeralDecisionFrame(msg);
        if (ephemeral !== undefined) return ephemeral;
        const review = attachReviewFromRow(msg, rawTs);
        if (review !== undefined) return review;
        if (!taskId) return false;
        let changed = false;
        // Only host-attested local progress grants Stop authority.
        if (grantCancelAuthority && msg?.cancelable === true && msg?.task_id) {
            changed = markTaskCancelable(String(msg.task_id));
        }
        const lifecycleParent = taskKey(msg?.parent_task_id);
        if (msg?.subagent_event && lifecycleParent) {
            const updated = updateSubagentCardFromEvent(msg, rawTs);
            if (updated !== undefined) return Boolean(changed || updated);
        }
        if (subagentChildParents.has(taskId)) {
            const updated = routeSubagentProgressToCard(taskId, msg);
            return Boolean(changed || updated);
        }
        const taskState = getTaskUiState(taskId, true);
        const restartReusable = taskState?.completed && REUSABLE_TASK_IDS.has(taskId);
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
            // Delegation trio: a forgotten key freezes the chip
            // (wire_contract.test.js pins all three).
            executor_route: msg?.executor_route || '',
            execution_evidence: msg?.execution_evidence,
            actual_substrate: msg?.actual_substrate || '',
            status: msg?.status || '',
            ...costMetaKeys(msg),
            result: msg?.result || '',
            trace_summary: msg?.trace_summary || '',
            error: msg?.error || '',
            artifact_status: msg?.artifact_status || '',
            lifecycle: msg?.lifecycle || null,
        });
        if (!summary) return changed;
        const presented = withTaskCostMeta(summary, msg, { rawTs });
        changed = Boolean(queueTaskLiveUpdate(
            presented, taskId, normalizeLogTs(rawTs), presented.dedupeKey || '', rawTs,
        )) || changed;
        if (restartReusable) forceTaskCard(taskId, rawTs);
        // History may carry the coined name that live frames deliver separately.
        if (msg?.suggested_name) changed = applySuggestedName(taskId, msg.suggested_name) || changed;
        // Replay terminal truth when the best-effort summary row was lost.
        if (
            msg?.task_terminal_status
            && taskDoneIsTerminal({ ...msg, status: String(msg.task_terminal_status) })
            && (msg?.outcome_axes || msg?.review_projection || msg?.reason_code)
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
        forceTaskCard(parentId, tsValue);
        const childState = getTaskUiState(childId, true);
        if (childState && !childState.completed) childState.forceCard = true;
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
        forceTaskCard(parentId, rawTs);
        const childState = getTaskUiState(childId, true);
        if (childState && !childState.completed) childState.forceCard = true;
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
        summary.dedupeKey = `subagent-progress:${childId}`;
        if (preserveTerminal) {
            summary.phase = String(record.phaseEl?.dataset?.phase || 'done');
            summary.terminal = true;
        }
        return queueTaskLiveUpdate(
            summary, childId, normalizeLogTs(rawTs), summary.dedupeKey, rawTs,
        );
    }

    function routeSubagentFinalMessageToCard(taskId, msg) {
        const childId = taskKey(taskId);
        const info = subagentChildParents.get(childId);
        if (!childId || !info) return false;
        const { parentId, role } = info;
        const text = String(msg?.content || msg?.text || '').trim();
        const rawTs = msg?.ts || new Date().toISOString();
        forceTaskCard(parentId, rawTs);
        forceTaskCard(childId, rawTs);
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
            model: info.model || '',
            // Second whitelist: a log-channel-only terminal still upgrades the chip.
            executor_route: evt.executor_route || '',
            execution_evidence: evt.execution_evidence,
            actual_substrate: evt.actual_substrate || '',
            review_projection: evt.review_projection,
            result: evt.result || '',
            error: evt.error || '',
            reason_code: evt.reason_code || '',
            ...costMetaKeys(evt),
        }, evt.ts || evt.timestamp || new Date().toISOString()));
    }

    function updateLiveCardFromLogEvent(evt) {
        if (!evt) return false;
        const eventType = evt.type || evt.event || '';
        const reference = handleReviewReference(evt);
        if (reference !== undefined) return reference;
        if (!isGroupedTaskEvent(evt)) return false;
        const ephemeral = registerEphemeralDecisionFrame(evt);
        if (ephemeral !== undefined) return ephemeral;
        const taskId = getLogTaskGroupId(evt) || activeLiveGroupId || '';
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
        // Tool counts and the error shapes that force a visible card are
        // classified the same way for a subagent child and for its owner.
        const applyEventTelemetry = () => {
            if (eventType === 'tool_call_started') return markTaskToolCall(taskId, 1, false, rawTs);
            if ((eventType === 'task_metrics_event' || eventType === 'task_eval') && Number.isFinite(Number(evt.tool_calls))) {
                return markTaskToolCall(taskId, Number(evt.tool_calls), true, rawTs);
            }
            if (
                eventType === 'tool_call_timeout'
                || eventType === 'tool_timeout'
                || eventType === 'llm_round_error'
                || eventType === 'llm_api_error'
                || (eventType === 'tool_call_finished' && evt.is_error)
            ) return forceTaskCardVisibleChange(taskId, rawTs);
            return false;
        };
        // A known subagent child's log events update its linked child card.
        if (subagentChildParents.has(taskId)) {
            if (eventType === 'task_done') {
                return routeSubagentTerminalToCard(taskId, evt);
            }
            if (subagentTerminalChildren.has(taskId)) return false;
            let changed = applyEventTelemetry();
            const summary = summarizeChatLiveEvent(evt);
            if (!summary) return changed;
            const info = subagentChildParents.get(taskId);
            if (info) getSubagentCardRecord(taskId, info.parentId, info.role);
            changed = attachTaskDetailReviews(taskId, evt) || changed;
            const presented = withTaskCostMeta(summary, evt, {
                replace: eventType === 'task_done' || eventType === 'task_cost_finalized',
                rawTs,
            });
            const queued = queueTaskLiveUpdate(
                presented, taskId, normalizeLogTs(rawTs), presented.dedupeKey || '', rawTs,
            );
            return Boolean(changed || queued);
        }
        let changed = applyEventTelemetry();
        changed = attachTaskDetailReviews(taskId, evt) || changed;
        const summary = summarizeChatLiveEvent(evt);
        if (!summary) return changed;
        const presented = withTaskCostMeta(summary, evt, {
            replace: eventType === 'task_done' || eventType === 'task_cost_finalized',
            rawTs,
        });
        const queued = queueTaskLiveUpdate(
            presented, taskId, normalizeLogTs(rawTs), presented.dedupeKey || '', rawTs,
        );
        const subagentChanged = updateSubagentCardFromEvent(evt, rawTs);
        if (eventType === 'task_done' && summary.terminal) {
            recordTerminalActivity(taskId);
            syncChatStatus();
            const taskState = getTaskUiState(taskId, false);
            changed = revealBufferedCardIfNeeded(taskState, { rawTs }) || changed;
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
        const systemType = opts.systemType || '';
        const taskId = opts.taskId || '';
        const projectId = opts.projectId || '';
        const projectName = opts.projectName || '';
        const ts = timestamp || new Date().toISOString();
        const messageKey = buildMessageKey(role, text, ts, {
            clientMessageId,
            systemType,
            isProgress,
            source,
            senderLabel: senderLabelOverride,
            senderSessionId,
            taskId,
        });
        if (messageKey && seenMessageKeys.has(messageKey)) return false;

        if (!isProgress && !ephemeral) {
            persistedHistory.push({
                text,
                role,
                ts,
                markdown: !!markdown,
                systemType,
                source,
                senderLabel: senderLabelOverride,
                senderSessionId,
                clientMessageId,
                taskId,
                projectId,
                projectName,
                skillReview: opts.skillReview || null,
            });
            // Mirror the sessionStorage slice(-200): the in-memory copy exists
            // only to feed that snapshot, so it obeys the same cap (P3).
            if (persistedHistory.length > 200) {
                persistedHistory.splice(0, persistedHistory.length - 200);
            }
            // a rebuildAll replay serializes the sessionStorage
            // snapshot ONCE at the end of the batch, not per historical row.
            if (!_rebuildBatch) persistVisibleHistory();
        }

        const bubble = document.createElement('div');
        bubble.className = `chat-bubble ${role}` + (isProgress ? ' progress' : '');
        if (pending) bubble.classList.add('pending');
        if (ephemeral) bubble.dataset.ephemeral = '1';
        if (clientMessageId) bubble.dataset.clientMessageId = clientMessageId;
        if (systemType) bubble.dataset.systemType = systemType;
        if (senderSessionId) bubble.dataset.senderSessionId = senderSessionId;
        if (taskId) bubble.dataset.taskId = taskId;

        const sender = senderLabel(role, isProgress, systemType, {
            source, senderLabel: senderLabelOverride, senderSessionId,
        }, chatSessionId);
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
        bubble.innerHTML = `
            <div class="sender">${escapeHtml(sender)}</div>
            <div class="message">${rendered}</div>
            ${pendingHtml}
            ${timeHtml}
        `;
        if (!isProgress && text) chatMedia.attachCopyControl(bubble, String(text));
        if (PROJECT_ROW_TYPES.has(systemType) && projectId) {
            const actions = document.createElement('div');
            actions.className = 'system-message-actions';
            actions.append(createSystemMessageAction({
                label: 'Open Project ↗',
                onClick: () => window.dispatchEvent(new CustomEvent('ouro:open-project', {
                    detail: { project: { id: projectId, name: projectName || 'Project' } },
                })),
            }));
            bubble.querySelector('.message')?.append(actions);
        }
        wireSkillReviewDisclosure(bubble, { onDomWrite: withStableViewport });
        stampNodeTimestamp(bubble, ts);
        insertMessageNode(bubble, { forceStick: !!opts.forceStick });
        if (role !== 'user' && systemType !== 'skill_review' && (role !== 'system' || markdown === true)) enhanceMountedMarkdown(bubble);
        chatDecision.renderRoutingDecision(bubble, opts.chatAnnotation);
        rememberMessageKey(messageKey);
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
        return chatDecision.renderRoutingDecision(bubble, annotation);
    }

    function clearTransientRoutingAnnotations() {
        let changed = false;
        for (const note of messagesDiv.querySelectorAll(
            '.msg-routing-annotation[data-annotation-status="pending"]',
        )) {
            const bubble = note.closest('.chat-bubble');
            if (bubble) delete bubble.dataset.chatAnnotationStatus;
            note.remove();
            changed = true;
        }
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

    function ensureWelcomeMessage() {
        if (!isMain) return;
        if (welcomeShown) return;
        const hasRealBubbles = Array.from(messagesDiv.querySelectorAll('.chat-bubble')).some(
            bubble => !bubble.classList.contains('typing-bubble')
        );
        if (hasRealBubbles) return;
        welcomeShown = true;
        addMessage('Ouroboros has awakened', 'assistant', false, null, false, { ephemeral: true });
    }

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

    // Apply deferred card/typing/storage work once after the replay batch mounts.
    function finalizeRebuildBatch(batch) {
        for (const record of batch.touched) {
            renderLiveCardMeta(record);
            updateLiveCardCount(record);
        }
        if (batch.typingHidden) hideTypingIndicatorOnly();
        persistVisibleHistory();
    }

    async function syncHistory({ includeUser = false, fromReconnect = false, forceRebuild = false } = {}) {
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
            try {
                // Server defaults own first-load quotas; Load older overrides them.
                let historyUrl = `/api/chat/history${isMain ? '' : `?chat_id=${chatId}`}`;
                if (historyQuotaOverride) {
                    const sep = historyUrl.includes('?') ? '&' : '?';
                    historyUrl += `${sep}n_human=${historyQuotaOverride.n_human}`
                        + `&n_progress=${historyQuotaOverride.n_progress}`;
                }
                const resp = await apiFetch(historyUrl, { cache: 'no-store' });
                if (!resp.ok) {
                    lastHistorySyncSucceeded = false;
                    initialHydrationPromise = null;
                    return false;
                }
                const data = await resp.json();
                // A late continuation on a destroyed instance must not rebuild a
                // detached DOM subtree or repopulate the cleared collections.
                if (destroyed) {
                    lastHistorySyncSucceeded = false;
                    initialHydrationPromise = null;
                    return false;
                }
                const messages = Array.isArray(data.messages) ? data.messages : [];
                // the server's window verdict (P3.2 additive field)
                // drives the Load-older button/notice after this sync lands.
                historyWindow = (data && typeof data.window === 'object' && data.window)
                    ? data.window
                    : null;
                const scrollBeforeSync = {
                    top: messagesDiv.scrollTop,
                    nearBottom: isNearBottom(),
                    anchor: captureVisibleTimelineAnchor(),
                };

                // First load/reconnect trusts server history and fully rebuilds the
                // feed; routine post-completion syncs only fold in new task cards.
                // Load-older (forceRebuild) and an arm this sync inherited rebuild fully too.
                const rebuildAll = !historyLoaded || fromReconnect || forceRebuild || armedAtStart;
                // On a soft reconnect the module (and its dedupe set)
                // survives: a plain re-sync would dedupe-drop every bubble.
                // Restore user text and rebuild from durable history on every
                // rebuild — incl. includeUser=false triggers (clean open /
                // 700ms resync), since the rebuild clears those too.
                const renderUser = includeUser || fromReconnect || armedAtStart;
                // Every rebuild replays everything, so retirement resets with it.
                if (rebuildAll) retiredTaskIds.clear();

                // the ENTIRE mutation below (clear -> pass 1 ->
                // pass 2 -> terminal resolution -> sweep) is one synchronous
                // closure. On rebuildAll it runs inside ONE outer
                // withStableViewport with a detached batch collecting the
                // top-level nodes; NO awaits may occur between the feed
                // clearing and the batch mount. The routine path
                // (rebuildAll=false) calls it directly — unchanged behavior.
                const applySyncedMessages = () => {
                // Server-confirmed rows retire their journal copy; the rest
                // survive the rebuild below.
                const localEcho = partitionLocalEchoJournal(localEchoJournal, new Set(messages
                    .filter((m) => m.role === 'user' && m.client_message_id)
                    .map((m) => String(m.client_message_id))));
                for (const entry of localEcho.confirmed) {
                    localEchoJournal.delete(entry.clientMessageId);
                }
                if (rebuildAll) {
                    chatMedia.reset();
                    // The cards below are a new presentation generation. Keep
                    // hydrator single-flight/pending state, but let an already
                    // applied durable Plan revision attach to the rebuilt card.
                    reviewHydrator.invalidateApplied();
                    for (const record of liveCardRecords.values()) record.root?.remove();
                    liveCardRecords.clear();
                    for (const taskState of taskUiStates.values()) {
                        if (taskState?.cleanupTimer) clearTimeout(taskState.cleanupTimer);
                    }
                    taskUiStates.clear();
                    ephemeralDecisionTaskIds.clear();
                    // Rebuild replays the durable truth: stale name buffers and
                    // cancelable markers from the previous connection are dropped
                    // and re-learned from history rows (P3 growth caps).
                    pendingSuggestedNames.clear();
                    cancelableTaskIds.clear();
                    activeLiveGroupId = '';
                    // Atomically drop the standalone message bubbles and the dedupe
                    // state so the rebuild below cannot produce duplicates even if
                    // stale bubbles lingered in the DOM. Keep the typing indicator.
                    for (const bubble of Array.from(messagesDiv.querySelectorAll('.chat-bubble'))) {
                        if (!bubble.classList.contains('typing-bubble')) {
                            destroyChatMarkdown(bubble);
                            bubble.remove();
                        }
                    }
                    seenMessageKeys.clear();
                    messageKeyOrder.length = 0;
                    subagentChildParents.clear();
                    subagentTerminalChildren.clear();
                    for (const msg of messages) learnSubagentLineage(msg);
                }

                // First pass builds card state without DOM insertion.
                _syncPass1Active = true;
                try { for (const msg of messages) {
                    if (handleReviewReference(msg) !== undefined) continue;
                    if (attachReviewFromRow(msg, msg.ts || '') !== undefined) continue;
                    const taskId = msg.task_id || '';
                    if (!taskId) continue;
                    if (retiredTaskIds.has(taskId)) continue;
                    if (msg.is_progress) {
                        updateLiveCardFromProgressMessage(msg, { grantCancelAuthority: msg.project_mirror !== true });
                        continue;
                    }
                    if (msg.system_type === 'task_summary') {
                        // Historical cards only for non-trivial tasks.
                        const hadToolCalls = (msg.tool_calls || 0) > 0;
                        const hadMultipleRounds = (msg.rounds || 0) > 1;
                        const severity = taskOutcomeSeverity(msg);
                        const needsVisibleTerminal = severity === 'error' || severity === 'warn' || severity === 'cancelled';
                        if (hadToolCalls || hadMultipleRounds || needsVisibleTerminal) {
                            const taskState = getTaskUiState(taskId, true);
                            if (taskState) taskState.forceCard = true;
                        }
                        // Pass 2 inserts this in the right transcript position.
                        appendTaskSummaryToLiveCard(msg, { suppressDomInsert: true });
                    }
                } } finally { _syncPass1Active = false; }

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
                    if (rec && rec.root && !rec.root.isConnected) {
                        if (rec.isSubagent) ensureLiveCardVisible(rec);
                        else insertMessageNode(rec.root);
                    }
                }
                for (const msg of messages) {
                    const taskId = msg.task_id || '';
                    // Owner-bound reviews attached in pass 1 are not terminal chat bubbles.
                    if (
                        handleReviewReference(msg) !== undefined
                        || attachReviewFromRow(msg, msg.ts || '', true) !== undefined
                    ) continue;
                    // Reconnect: a durably recorded submission must not stay
                    // `Sending...` — history + snapshot are the authorities
                    // (a live turn re-links via hydration / next typing frame).
                    if (fromReconnect && msg.role === 'user' && msg.client_message_id) {
                        pendingSubmissions.delete(String(msg.client_message_id));
                    }
                    if (!renderUser && msg.role === 'user') continue;
                    if (msg.is_progress) {
                        // Progress-only/failed tasks still anchor at their first event.
                        insertCardIfNeeded(taskId);
                        // Open post-task checkpoint replays as "Finalizing…".
                        if (msg.task_phase === 'finalizing') markLiveCardFinalizing(taskId);
                        continue;
                    }
                    if (msg.system_type === 'task_summary') continue;
                    if (PROJECT_ROW_TYPES.has(msg.system_type)) {
                        addMessage(msg.text, 'system', !!msg.markdown, msg.ts || null, false, {
                            systemType: msg.system_type,
                            taskId,
                            projectId: msg.project_id || '',
                            projectName: msg.project_name || '',
                        });
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
                            const taskState = getTaskUiState(taskId, false);
                            const record = liveCardRecords.get(taskId);
                            finishLiveCard(taskId, msg.task_terminal_status ? taskTerminalPhase(msg) : replayTerminalPhase(taskState, record));
                            continue;
                        }
                        insertCardIfNeeded(taskId);
                        // A replayed early final must not finalize the card.
                        if (msg.task_phase === 'finalizing') {
                            markLiveCardFinalizing(taskId);
                        } else {
                            const taskState = getTaskUiState(taskId, false);
                            const record = liveCardRecords.get(taskId);
                            finishLiveCard(taskId, msg.task_terminal_status ? taskTerminalPhase(msg) : replayTerminalPhase(taskState, record));
                        }
                    }
                    // A replayed durable routing receipt carries the same
                    // authority as its live WS frame: a receipt that landed
                    // while the socket was down still retires `Sending...`.
                    if (msg.chat_annotation && msg.client_message_id) {
                        pendingSubmissions.delete(String(msg.client_message_id));
                    }
                    addMessage(msg.text, msg.role, !!msg.markdown, msg.ts || null, false, {
                        systemType: msg.system_type || '',
                        source: msg.source || '',
                        senderLabel: msg.sender_label || '',
                        senderSessionId: msg.sender_session_id || '',
                        clientMessageId: msg.client_message_id || '',
                        taskId,
                        chatAnnotation: msg.chat_annotation || null,
                        skillReview: msg.system_type === 'skill_review' && msg.skill && msg.job_id
                            ? { skill: msg.skill, jobId: msg.job_id }
                            : null,
                    });
                }
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
                    if (!taskDoneIsTerminal(terminalRecord)) continue;
                    // Subagent terminal status resolves the child card, not the
                    // parent. Otherwise reload can revive a crashed/cancelled child.
                    if (subagentChildParents.has(tid)) {
                        routeSubagentTerminalToCard(tid, terminalRecord);
                        continue;
                    }
                    const rec = liveCardRecords.get(tid);
                    if (rec && !rec.finished) {
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

                // Append disconnected visible cards after mid-task reload; skip trivial placeholders.
                for (const [tid, rec] of liveCardRecords) {
                    reorderDirtyCardIfNeeded(rec);
                    if (rec && rec.root && !rec.root.isConnected && !retiredTaskIds.has(tid)) {
                        const ts = taskUiStates.get(tid);
                        if (ts && !ts.cardVisible && ts.completed) continue;
                        if (rec.isSubagent) ensureLiveCardVisible(rec);
                        else insertMessageNode(rec.root);
                    }
                }
                // A rebuild replays only what the server returned; re-render
                // owner rows a stale snapshot has not confirmed yet.
                if (rebuildAll) {
                    for (const entry of localEcho.unconfirmed) {
                        addMessage(entry.text, 'user', false, entry.ts, false, {
                            // A still-queued offline row keeps its pending mark.
                            pending: pendingUserBubbles.has(entry.clientMessageId),
                            source: 'web',
                            senderSessionId: chatSessionId,
                            clientMessageId: entry.clientMessageId,
                            chatAnnotation: entry.annotation,
                        });
                    }
                }
                };  // end applySyncedMessages

                // Double-fetch fix: the replay below marks
                // historical cards finished; those transitions must not
                // schedule the post-completion resync (the rows just arrived
                // from this very fetch). The flag spans BOTH branches and is
                // dropped synchronously, so a real live completion frame can
                // never land while it is up.
                _historyReplayActive = true;
                liveCardBound.beginReplay();
                try {
                    if (rebuildAll) {
                        // one outer withStableViewport for
                        // the whole rebuild (inner wrappers collapse on the
                        // _viewportMutationDepth gate — no per-frame layout
                        // storm); one stable sort, one fragment mount, ONE
                        // persist, all synchronous: live frames never observe
                        // "records cleared, fragment not yet mounted".
                        _rebuildBatch = createRebuildBatch();
                        try {
                            withStableViewport(() => {
                                applySyncedMessages();
                                const batch = _rebuildBatch;
                                _rebuildBatch = null;
                                batch.mount(messagesDiv, messagesDiv.querySelector('.typing-bubble'));
                                finalizeRebuildBatch(batch);
                            });
                        } finally {
                            _rebuildBatch = null;
                        }
                    } else {
                        // Routine replay is one transaction: repeated rows may
                        // touch intermediate projections, but only a net height
                        // or top-level order change owns the viewport.
                        const beforeHeight = messagesDiv.scrollHeight;
                        const beforeNodes = Array.from(messagesDiv.children);
                        withStableViewport(() => {
                            applySyncedMessages();
                            const afterNodes = messagesDiv.children;
                            return beforeHeight !== messagesDiv.scrollHeight
                                || beforeNodes.length !== afterNodes.length
                                || beforeNodes.some((node, index) => node !== afterNodes[index]);
                        });
                    }
                } finally {
                    _historyReplayActive = false;
                }

                // After first load, sync status from live cards/active turns.
                syncChatStatus();

                // One-shot server recall seed includes other clients without resetting
                // ArrowUp during reconnect. Merge [server..., local...], newest wins.
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
                liveCardBound.settle({ rebuilt: rebuildAll, size: liveCardRecords.size });
                // ANY successful sync leaves the instance hydrated
                // — later hydration triggers ride this sticky promise.
                initialHydrationPromise = historySyncPromise;
                // reflect the server's window verdict in the
                // Load-older control now that the feed matches this response.
                syncLoadOlderControl();
                // A recreated project instance restores its predecessor's stashed
                // mid-history position on first paint instead of pinning to newest.
                if (wasFirstLoad && _initialScrollPending) {
                    _initialScrollPending = false;
                    updateMessagesPadding(false);
                    restoreScrollPosition();
                } else
                // First load jumps to latest; reconnect preserves older-message reading.
                if (wasFirstLoad || (fromReconnect ? scrollBeforeSync.nearBottom : isNearBottom())) {
                    updateMessagesPadding();
                    // Bootstrap pin: a fresh feed starts at scrollTop 0, where
                    // the ordinary boundary would not land on the newest row.
                    if (wasFirstLoad) scrollToBottomAfterLayout();
                } else if (fromReconnect) {
                    // Rebuild may add rows both ABOVE and BELOW the viewport; a
                    // scrollHeight delta cannot tell them apart and over-scrolls
                    // readers. Restore the first visible timestamped node to its
                    // prior visual offset instead (equal-ts ordinals keep
                    // arrival-order identity), after two RAF frames so async
                    // card heights above the anchor cannot move the reader.
                    await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
                    const restoredFromAnchor = restoreVisibleTimelineAnchor(scrollBeforeSync.anchor);
                    if (!restoredFromAnchor) messagesDiv.scrollTop = scrollBeforeSync.top;
                    updateScrollButton();
                }
                return messages.length > 0;
            } catch (err) {
                lastHistorySyncSucceeded = false;
                initialHydrationPromise = null;
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

    function cancelHistoryPaint() {
        historyPaintGeneration += 1;
    }

    async function refreshHistory({ revision = 0 } = {}) {
        const generation = ++historyPaintGeneration;
        const targetRevision = Math.max(0, Number(revision) || 0);
        // only a NEW revision (or a never-hydrated instance)
        // forces a real fetch; otherwise the sticky hydration promise answers
        // and the paint receipt below still runs.
        if (targetRevision > lastLoadedHistoryRevision || !initialHydrationPromise) {
            await syncHistory({ includeUser: true });
        } else {
            await awaitInitialHydration({ includeUser: true });
        }
        if (lastHistorySyncSucceeded && targetRevision > lastLoadedHistoryRevision) {
            lastLoadedHistoryRevision = targetRevision;
        }
        if (destroyed || !lastHistorySyncSucceeded || generation !== historyPaintGeneration || page.hidden) {
            return { painted: false, revision: targetRevision };
        }
        // A successful fetch is not a read acknowledgement until the rebuilt
        // DOM has crossed an actual browser paint while this Project remains
        // visible. Two frames cover layout followed by paint/composite. A
        // destroyed page reports hidden===false, so the paint receipt must also
        // consult the lifecycle flag — a late paint on a torn-down instance
        // would otherwise acknowledge a revision that was never shown.
        await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
        return {
            painted: !destroyed && generation === historyPaintGeneration && !page.hidden,
            revision: targetRevision,
        };
    }

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
            for (const msg of saved) {
                addMessage(msg.text, msg.role, !!msg.markdown, msg.ts || null, false, {
                    systemType: msg.systemType || '',
                    source: msg.source || '',
                    senderLabel: msg.senderLabel || '',
                    senderSessionId: msg.senderSessionId || '',
                    clientMessageId: msg.clientMessageId || '',
                    taskId: msg.taskId || '',
                    projectId: msg.projectId || '',
                    projectName: msg.projectName || '',
                    skillReview: msg.skillReview || null,
                });
            }
        } catch {}
        historyLoaded = true;
        // this offline fallback sets historyLoaded=true, which would
        // make the first successful post-outage sync a NON-rebuilding routine
        // fold over stale sessionStorage bubbles. Flag it so that sync
        // rebuilds from durable history instead.
        if (!lastHistorySyncSucceeded) liveCardBound.arm();
        ensureWelcomeMessage();
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
        // The owner's pure typed request (before attachment lines) — captured so a
        // live card spawned by this message can name a project from it on a "turn
        // into project" conversion even before the task records its objective (P1,
        // direct-chat case: the server has no title/objective/queue source yet).
        const objectiveText = text;
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
                const attachmentLines = uploaded
                    .map((item) => `[Attached file: ${item.display_name} saved to ${item.path}]`)
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
        // Hand the objective to the NEXT main-chat live card this message spawns.
        if (isMain && objectiveText) _pendingCardObjective = objectiveText;
        if (hasAttachments) {
            pendingAttachments = [];
            updateAttachmentPreview();
        }
        rememberInput(text);
        input.value = '';
        clearInputDraft();
        const sentTs = new Date().toISOString();
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
        scrollToBottomAfterLayout();
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
        const next = seg.dataset.mode === 'low' ? 'low' : 'max';
        const current = contextModeBtn.dataset.contextMode === 'low' ? 'low' : 'max';
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
    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage(swarmArmed());
            return;
        }
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

    function scrollToBottomAfterLayout() {
        requestAnimationFrame(() => {
            if (destroyed) return;
            scrollToBottom();
            requestAnimationFrame(() => {
                if (destroyed) return;
                scrollToBottom();
            });
        });
    }

    // Ignore hidden/restoring scroll events so browser resets cannot corrupt saved intent.
    messagesDiv?.addEventListener('scroll', () => {
        if (!isInstanceVisible()) return;
        if (_restoring) { updateScrollButton(); return; }
        _savedScrollTop = messagesDiv.scrollTop;
        _savedStick = isNearBottom();
        updateScrollButton();
    }, { passive: true });

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
        scrollBottomBtn.classList.toggle('visible', isInstanceVisible() && !isNearBottom());
    }
    scrollBottomBtn?.addEventListener('click', () => {
        _savedStick = true;
        scrollToBottomAfterLayout();
        updateScrollButton();
    });

    function restoreScrollPosition() {
        if (!isInstanceVisible()) return;  // hidden column has no geometry yet
        // Reapply across relayout frames for pre-scroll-anchoring WKWebView.
        _restoring = true;
        const targetStick = _savedStick;
        const targetTop = _savedScrollTop;
        let frames = 0;
        const apply = () => {
            if (destroyed || !isInstanceVisible()) { _restoring = false; return; }
            messagesDiv.scrollTop = targetStick ? messagesDiv.scrollHeight : targetTop;
            updateScrollButton();
            if (++frames < 12) requestAnimationFrame(apply);
            else _restoring = false;
        };
        requestAnimationFrame(apply);
    }

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
                updateMessagesPadding();
            });
        };
        chatResizeObserver = new ResizeObserver(schedule);
        if (pageHeader) chatResizeObserver.observe(pageHeader);
        if (inputArea) chatResizeObserver.observe(inputArea);
    }

    installChatResizeObservers();

    // Per-thread input draft (P3): destroy-on-close would otherwise lose typed
    // but unsent text. Saved on every input (cheap), restored at instance
    // creation, cleared on send.
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

    // The More menu is a native <details> (no auto-dismiss): collapse it when a
    // click/tap lands outside it, or on Escape, so it never stays stuck open.
    // Handler refs are kept so destroy() can remove them (P3 lifecycle).
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
        // A panel has no global controls/budget to poll; seed the status from
        // the live socket so a late-created panel never gets stuck on
        // "Connecting…" (the one-shot WS `open` already fired before it existed;
        // future reconnects still update it via the shared `open` handler).
        if (ws.isConnected?.()) setStatus('online', 'Online');
        // 1A a panel created AFTER the socket opened missed the typing frame
        // and the `open`-driven refresh — hydrate in-flight turns once from
        // the snapshot (per-instance closure filters to this panel's chat_id).
        refreshHeaderControlState(true);
    } else {
        refreshHeaderControlState(true);
        headerControlInterval = setInterval(refreshHeaderControlState, 3000);
    }

    const typingEl = document.createElement('div');
    // Per-instance id (main stays 'typing-indicator'; panels get a unique id) so
    // multiple open chat columns never collide on a duplicate DOM id.
    typingEl.id = idPrefix === 'chat' ? 'typing-indicator' : `${idPrefix}-typing-indicator`;
    typingEl.className = 'chat-bubble assistant typing-bubble';
    typingEl.style.display = 'none';
    typingEl.innerHTML = `<div class="typing-dots"><span></span><span></span><span></span></div>`;
    messagesDiv.appendChild(typingEl);

    // "Load older" atop the feed. Server truth (window.complete /
    // truncated_by, P3.2) picks refetch button vs honest boundary notice;
    // excluded from viewport anchoring; mounted ONLY with content — a hidden
    // permanent node would break child-order consumers (ui-smoke chronology).
    const loadOlderEl = document.createElement('div');
    loadOlderEl.className = 'chat-load-older';
    const loadOlderBtn = document.createElement('button');
    loadOlderBtn.type = 'button';
    loadOlderBtn.className = 'chat-load-older-btn';
    loadOlderBtn.textContent = 'Load older messages';
    const loadOlderNote = document.createElement('span');
    loadOlderNote.className = 'chat-load-older-note';
    loadOlderNote.hidden = true;
    loadOlderEl.append(loadOlderBtn, loadOlderNote);
    loadOlderBtn.addEventListener('click', () => { loadOlderHistory(); });

    function syncLoadOlderControl() {
        const control = loadOlderControlState(historyWindow, historyQuotaOverride);
        return withStableViewport(() => {
            if (control.mode === 'hidden') {
                if (!loadOlderEl.isConnected) return false;
                loadOlderEl.remove();
                return true;
            }
            const buttonHidden = control.mode !== 'button';
            const buttonText = loadingOlderHistory
                ? 'Loading…'
                : (control.mode === 'button' ? control.label : 'Load older messages');
            const noteHidden = control.mode !== 'notice';
            const changed = !loadOlderEl.isConnected
                || loadOlderBtn.hidden !== buttonHidden
                || loadOlderBtn.disabled !== loadingOlderHistory
                || loadOlderBtn.textContent !== buttonText
                || loadOlderNote.hidden !== noteHidden
                || (!noteHidden && loadOlderNote.textContent !== control.label);
            if (!changed) return false;
            if (!loadOlderEl.isConnected) messagesDiv.prepend(loadOlderEl);
            loadOlderBtn.hidden = buttonHidden;
            loadOlderBtn.disabled = loadingOlderHistory;
            loadOlderBtn.textContent = buttonText;
            loadOlderNote.hidden = noteHidden;
            if (!noteHidden) loadOlderNote.textContent = control.label;
            return true;
        });
    }

    async function loadOlderHistory() {
        if (loadingOlderHistory) return;
        const next = nextQuotaEscalation(historyQuotaOverride);
        if (!next) return;
        loadingOlderHistory = true;
        syncLoadOlderControl();
        // Anchor the current first visible timestamped node (the control
        // itself is excluded from capture, like .typing-bubble) so the reader
        // does not drift when older rows land above the viewport.
        const anchor = isNearBottom() ? null : captureVisibleTimelineAnchor();
        const previousQuota = historyQuotaOverride;
        historyQuotaOverride = next;
        try {
            // Drain EVERY in-flight sync first: coalescing into one would
            // silently drop forceRebuild and the escalated window, and another
            // waiter can install a NEW promise right as the previous one
            // settles — so re-check until the slot is genuinely free. Only
            // then does syncHistory below start as OUR fetch (its head sees
            // historySyncPromise === null synchronously).
            while (historySyncPromise) {
                try { await historySyncPromise; } catch {}
                if (destroyed) return;
            }
            await syncHistory({ includeUser: true, forceRebuild: true });
            if (destroyed) return;
            if (!lastHistorySyncSucceeded) {
                historyQuotaOverride = previousQuota;
                return;
            }
            // Like the reconnect restore: wait two frames so late card layout
            // above the anchor cannot move the reader right after this call.
            await new Promise((resolve) => requestAnimationFrame(
                () => requestAnimationFrame(resolve)
            ));
            if (destroyed) return;
            if (anchor) restoreVisibleTimelineAnchor(anchor);
            updateScrollButton();
        } finally {
            loadingOlderHistory = false;
            if (!destroyed) syncLoadOlderControl();
        }
    }

    function hasActiveLiveCard() {
        return Array.from(liveCardRecords.values()).some(isForegroundLiveCard);
    }

    function deriveChatStatus() {
        let directCount = 0, managedActive = 0, managedQueued = 0, managedPaused = 0;
        for (const entry of activeDirectActivities.values()) {
            if (String(entry?.kind || '') !== 'managed_task') directCount += 1;
            else if (String(entry?.phase || '') === 'queued') managedQueued += 1;
            else if (String(entry?.phase || '') === 'budget_paused') managedPaused += 1;
            else managedActive += 1;
        }
        return computeDerivedChatStatus({
            isConnected: ws.isConnected ? ws.isConnected() : true,
            hasActiveLiveCard: hasActiveLiveCard(),
            activeDirectCount: directCount,
            activeManagedCount: managedActive,
            queuedManagedCount: managedQueued,
            pausedManagedCount: managedPaused,
            pendingSubmissionsCount: pendingSubmissions.size,
        });
    }

    function syncChatStatus() {
        const derived = deriveChatStatus();
        setStatus(derived.kind, derived.text);
        return setTypingIndicatorVisible(derived.showDots && !hasActiveLiveCard());
    }

    function setTypingIndicatorVisible(visible) {
        const display = visible ? '' : 'none';
        if (typingEl.style.display === display) return false;
        return withStableViewport(() => {
            typingEl.style.display = display;
            return true;
        });
    }

    function showTyping(activityId = '', meta = {}) {
        const actId = taskKey(activityId) || ('direct-' + chatId);
        // A typing frame after its turn's keyed final must not resurrect the
        // concluded turn — but it still carries the activity<->cmid link, so
        // it settles the linked submission (broadcasts are not ordered).
        if (concludedDirectActivities.has(actId)) {
            if (meta.clientMessageId && pendingSubmissions.delete(meta.clientMessageId)) {
                syncChatStatus();
            }
            return;
        }
        // Fresh live evidence outranks a task-detail read woken by an older
        // queue-loss snapshot. The in-flight read may finish, but cannot apply.
        missingManagedTaskIds.delete(actId);
        activeDirectActivities.set(actId, {
            activityId: actId,
            // '' = not registry-tracked (queued managed task): visible in the
            // active set but exempt from /api/state snapshot deletion.
            kind: meta.kind || '',
            phase: meta.phase || 'thinking',
            clientMessageId: meta.clientMessageId || '',
            startedAt: Date.now(),
        });
        markReviewAnchor(liveCardRecords.get(actId));
        if (meta.clientMessageId) {
            pendingSubmissions.delete(meta.clientMessageId);
        }
        syncChatStatus();
    }

    function hideTypingIndicatorOnly() {
        // one typing-indicator write per replay batch.
        if (_rebuildBatch) {
            _rebuildBatch.typingHidden = true;
            return true;
        }
        return setTypingIndicatorVisible(false);
    }

    function revokeManagedTaskCancelAuthority(taskId) {
        cancelableTaskIds.delete(taskId);
        const record = liveCardRecords.get(taskId);
        if (!record) return;
        record.cancelable = false;
        syncCancelRunButton(record);
    }

    async function reconcileMissingManagedTask(taskId, onDomWrite = withStableViewport) {
        if (
            destroyed
            || managedTaskDetailReads.has(taskId)
            || concludedDirectActivities.has(taskId)
            || !missingManagedTaskIds.has(taskId)
        ) return;
        const record = liveCardRecords.get(taskId);
        if (subagentChildParents.has(taskId) || record?.isSubagent) {
            missingManagedTaskIds.delete(taskId);
            return;
        }
        managedTaskDetailReads.add(taskId);
        try {
            const detail = await fetchTaskDetail(taskId);
            if (destroyed || concludedDirectActivities.has(taskId)) return;
            const currentRecord = liveCardRecords.get(taskId);
            if (!currentRecord || currentRecord.isSubagent || subagentChildParents.has(taskId)) return;
            onDomWrite(() => {
                let changed = Boolean(attachTaskDetailReviews(taskId, detail));
                const cancelPending = taskCancelPending(detail);
                if (cancelPending || taskKey(detail?.status)) {
                    changed = markReviewAnchor(currentRecord) || changed;
                }
                if (cancelPending) {
                    return Boolean(
                        reconcileCancelCardFromDetail(currentRecord, taskId, detail) || changed
                    );
                }
                if (
                    !missingManagedTaskIds.has(taskId)
                    || activeDirectActivities.has(taskId)
                ) return changed;
                if (!isTerminalTaskDetail(detail)) {
                    return Boolean(
                        reconcileCancelCardFromDetail(currentRecord, taskId, detail) || changed
                    );
                }
                recordTerminalActivity(taskId);
                return Boolean(
                    appendTaskSummaryToLiveCard({ ...detail, task_id: taskId }) || changed
                );
            });
        } catch {
            // No terminal fact was proved. A later existing snapshot retries.
        } finally {
            managedTaskDetailReads.delete(taskId);
        }
    }

    function observeMissingManagedTask(taskId, onDomWrite = withStableViewport) {
        const id = taskKey(taskId);
        if (!id || concludedDirectActivities.has(id)) return;
        const record = liveCardRecords.get(id);
        if (subagentChildParents.has(id) || record?.isSubagent) return;
        missingManagedTaskIds.add(id);
        void reconcileMissingManagedTask(id, onDomWrite);
    }

    function hydrateDirectActivities(turnsList, snapshotBarrierMs = Infinity, snapshotGeneration = 0) {
        if (!Array.isArray(turnsList)) return;
        const {
            activities: nextMap,
            departedManagedTaskIds,
            disappearedManagedTaskIds,
            concludedDirectActivities: settledDirectRows,
            globallyActiveActivityIds,
        } = reconcileHydratedDirectActivities(
            activeDirectActivities, turnsList, chatId, snapshotBarrierMs,
            concludedDirectActivities, snapshotGeneration,
        );
        activeDirectActivities.clear();
        for (const [k, v] of nextMap.entries()) {
            activeDirectActivities.set(k, v);
            markReviewAnchor(liveCardRecords.get(k));
            if (v.kind === 'managed_task') missingManagedTaskIds.delete(k);
            if (v.clientMessageId) {
                pendingSubmissions.delete(v.clientMessageId);
            }
        }
        for (const taskId of globallyActiveActivityIds) missingManagedTaskIds.delete(taskId);
        for (const row of settledDirectRows) {
            if (!REUSABLE_TASK_IDS.has(row.activityId)) recordConcludedActivity(row.activityId);
            if (row.clientMessageId) pendingSubmissions.delete(row.clientMessageId);
        }
        for (const taskId of departedManagedTaskIds) revokeManagedTaskCancelAuthority(taskId);
        for (const taskId of disappearedManagedTaskIds) observeMissingManagedTask(taskId);
        for (const taskId of unconfirmedForegroundCardIds(
            Array.from(liveCardRecords, ([id, r]) => ({
                id, finished: r.finished, isSubagent: r.isSubagent, connected: r.root?.isConnected,
            })),
            new Set([...globallyActiveActivityIds, ...activeDirectActivities.keys()]),
        )) observeMissingManagedTask(taskId);
        for (const taskId of missingManagedTaskIds) {
            if (!activeDirectActivities.has(taskId)) void reconcileMissingManagedTask(taskId);
        }
        syncChatStatus();
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
        const actId = msg.activity_id || msg.task_id || ('direct-' + (msg.chat_id || chatId));
        const clientMsgId = msg.client_message_id || '';
        showTyping(actId, {
            clientMessageId: clientMsgId,
            phase: msg.phase || 'thinking',
            // Server-stamped for registry turns and RUNNING queue roots;
            // kind-less frames stay outside snapshot deletion authority.
            kind: msg.kind || '',
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
        if (msg.role === 'user') {
            const clientMessageId = msg.client_message_id || '';
            const senderSessionId = msg.sender_session_id || '';
            // 2A: the user echo is receipt of the user ROW, not turn start —
            // it settles the bubble but must NOT retire the `Sending...`
            // submission; that takes a linked typing frame / snapshot turn /
            // routing receipt or the turn's conclusion.
            if (senderSessionId === chatSessionId && clientMessageId) {
                markPendingDelivered(clientMessageId);
                syncChatStatus();
                return;
            }
            const added = withRemoteActivity(() => addMessage(
                msg.content, 'user', false, msg.ts || null, false, {
                source: msg.source || '',
                senderLabel: msg.sender_label || '',
                senderSessionId,
                clientMessageId,
                taskId: msg.task_id || '',
                },
            ));
            if (added) incrementUnreadIfNeeded(msg);
            syncChatStatus();
            return;
        }

        if (msg.role === 'assistant' || msg.role === 'system') {
            return withRemoteActivity(() => {
            const explicitTaskId = msg.task_id || '';
            const reference = handleReviewReference(msg);
            if (reference !== undefined) {
                syncChatStatus();
                return reference;
            }
            const review = attachReviewFromRow(msg, msg.ts || '', true);
            if (review !== undefined) {
                syncChatStatus();
                return review;
            }
            if (PROJECT_ROW_TYPES.has(msg.system_type)) {
                const added = addMessage(msg.content, 'system', msg.markdown, msg.ts || null, false, {
                    systemType: msg.system_type,
                    taskId: explicitTaskId,
                    projectId: msg.project_id || '',
                    projectName: msg.project_name || '',
                });
                if (added) incrementUnreadIfNeeded(msg);
                syncChatStatus();
                return Boolean(added);
            }
            learnSubagentLineage(msg);
            const ephemeral = registerEphemeralDecisionFrame(msg);
            const isEphemeral = ephemeral !== undefined;
            // Late duplicate progress cannot resurrect a concluded activity.
            if (isEphemeral && explicitTaskId && !concludedDirectActivities.has(explicitTaskId)) {
                const existing = activeDirectActivities.get(explicitTaskId) || {};
                activeDirectActivities.set(explicitTaskId, {
                    activityId: explicitTaskId,
                    kind: 'ephemeral_decision',
                    phase: 'thinking',
                    startedAt: existing.startedAt || Date.now(),
                    clientMessageId: existing.clientMessageId || '',
                });
            }
            if (msg.is_progress) {
                showTaskIncidentToast(msg);
                if (isEphemeral) return ephemeral;
                if (
                    msg.cancelable === true
                    && explicitTaskId
                    && !msg.subagent_event
                    && !subagentChildParents.has(explicitTaskId)
                ) {
                    showTyping(explicitTaskId, {
                        kind: 'managed_task', phase: msg.phase || 'working',
                    });
                }
                const changed = updateLiveCardFromProgressMessage(msg, { grantCancelAuthority: true });
                syncChatStatus();
                return changed;
            }

            // An early final (post-task still running) is NOT the turn's
            // conclusion; task_done or the queue snapshot concludes it.
            const finalizing = Boolean(explicitTaskId) && msg.task_phase === 'finalizing';
            const typedTerminal = positiveTaskTerminalFact(msg);
            const concludesTurn = !explicitTaskId || typedTerminal;
            if (!finalizing && concludesTurn) {
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
                if (!finalizing) markAssistantReply(explicitTaskId);
                if (changed) incrementUnreadIfNeeded(msg);
                syncChatStatus();
                return Boolean(ephemeral || changed);
            }
            if (explicitTaskId && subagentChildParents.has(explicitTaskId)) {
                const changed = routeSubagentFinalMessageToCard(explicitTaskId, msg);
                if (typedTerminal) markAssistantReply(explicitTaskId);
                if (changed) incrementUnreadIfNeeded(msg);
                syncChatStatus();
                return Boolean(ephemeral || changed);
            }
            let changed = Boolean(ephemeral);
            if (finalizing) changed = markLiveCardFinalizing(explicitTaskId) || changed;
            else if (explicitTaskId && typedTerminal) {
                changed = finishLiveCard(explicitTaskId, taskTerminalPhase(msg)) || changed;
            }
            if (!finalizing && typedTerminal) markAssistantReply(explicitTaskId);
            const routingCleared = clearTransientRoutingAnnotations();
            const added = addMessage(msg.content, msg.role, msg.markdown, msg.ts || null, false, {
                systemType: msg.system_type || '',
                source: msg.source || '',
                taskId: explicitTaskId,
            });
            if (added || changed) incrementUnreadIfNeeded(msg);
            syncChatStatus();
            return Boolean(added || changed || routingCleared);
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
        withRemoteActivity(() => updateLiveCardFromLogEvent(msg.data));
    });

    // the proactive namer coined a project name for a fresh card — show it
    // as the card title up front (turn-into-project then reuses the same name). Not
    // thread-gated on chat_id: the broadcast carries only task_id, and applySuggestedName
    // no-ops unless THIS thread already holds that card.
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
        // Reconnect drops kind-less entries (no snapshot source tracks them);
        // kind-stamped ones reconcile against the refreshed snapshot below.
        for (const [aid, entry] of activeDirectActivities) {
            if (!entry.kind) activeDirectActivities.delete(aid);
        }
        refreshHeaderControlState(true);
        syncChatStatus();
        // Reconnect truth comes from the ws CLIENT
        // (previouslyConnected rides the open event) — a project instance
        // created while the socket was already open must still treat the next
        // open as a reconnect. The per-instance flag stays only as a fallback
        // for open events without a payload.
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
            // Reconnect ALWAYS does a real fetch (a lost task_done is healed
            // only by refetching); the first clean open is a hydration trigger
            // and rides the sticky single-flight behind Main's idle gate.
            .then(() => (isReconnect
                ? syncHistory({ includeUser: !historyLoaded, fromReconnect: isReconnect })
                : waitForHydrationWindow().then(
                    () => awaitInitialHydration({ includeUser: !historyLoaded }),
                )))
            .then((hasMessages) => {
                if (!hasMessages) ensureWelcomeMessage();
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
        restoreScrollPosition,
        refreshHistory,
        cancelHistoryPaint,
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
        getScrollState: () => ({ scrollTop: _savedScrollTop, stick: _savedStick }),
        // Full teardown (P3): release every resource this instance acquired —
        // ws subscriptions, window/document listeners, the ResizeObserver, all
        // timers — then drop the buffered collections and remove the DOM last.
        // Idempotent; late rAF/async continuations no-op on `destroyed`.
        destroy() {
            if (destroyed) return;
            destroyed = true;
            cancelHistoryPaint();
            for (const dispose of wsDisposers) {
                try { dispose(); } catch {}
            }
            wsDisposers.length = 0;
            chatMedia.destroy();
            window.removeEventListener('ouro:page-shown', handlePageShown);
            document.removeEventListener('visibilitychange', handlePageShown);
            if (documentClickHandler) document.removeEventListener('click', documentClickHandler);
            if (documentKeydownHandler) document.removeEventListener('keydown', documentKeydownHandler);
            chatResizeObserver?.disconnect();
            chatResizeObserver = null;
            historyResyncScheduler.cancel();
            if (_chatFreedTimer) { clearTimeout(_chatFreedTimer); _chatFreedTimer = null; }
            for (const taskState of taskUiStates.values()) {
                if (taskState?.cleanupTimer) clearTimeout(taskState.cleanupTimer);
            }
            if (headerControlInterval) { clearInterval(headerControlInterval); headerControlInterval = null; }
            liveCardRecords.clear();
            explicitCardExpansion.clear();
            reviewDisclosureByTask.clear();
            skillReviewDetailStore.clear();
            reviewHydrator.clear();
            taskUiStates.clear();
            pendingSuggestedNames.clear();
            subagentChildParents.clear();
            subagentTerminalChildren.clear();
            cancelableTaskIds.clear();
            missingManagedTaskIds.clear();
            managedTaskDetailReads.clear();
            ephemeralDecisionTaskIds.clear();
            retiredTaskIds.clear();
            pendingUserBubbles.clear();
            localEchoJournal.clear();
            seenMessageKeys.clear();
            messageKeyOrder.length = 0;
            persistedHistory.length = 0;
            try { destroyChatMarkdown(page); page.remove(); } catch {}
        },
    };
}
