import {
    escapeHtmlAttr,
    escapeHtmlText as escapeHtml,
    rawTimestampEpoch,
    renderMarkdown,
} from './utils.js';
import { renderPageHeader } from './page_header.js';
import { PAGE_ICONS } from './page_icons.js';
import { showToast } from './toast.js';
import { clientSurfaceField } from './client_surface.js';
import { apiClient, apiFetch } from './api_client.js';
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
import { REUSABLE_TASK_IDS } from './task_control_menu.js';
import { openConfirmDialog } from './confirm_dialog.js';
import { renderSkillReviewDisclosure, wireSkillReviewDisclosure } from './skill_review_card.js';
import {
    createHistoryResyncScheduler,
    createRebuildBatch,
    insertTimelineNode,
    loadOlderControlState,
    nextQuotaEscalation,
} from './chat_render_batch.js';
import {
    COLLAPSED_ACTIVITY_MAX,
    boundActivityPreview,
    clearStickyCardState,
    isTerminalTaskPhase,
    liveLineRowToggleKey,
    projectCollapsedActivity,
} from './chat_card_state.js';
import {
    computeDerivedChatStatus,
    computeHydratedDirectActivities,
    partitionLocalEchoJournal,
    reconnectBannerText,
} from './chat_activity.js';
import { createCardActions } from './chat_card_actions.js';
import { confirmAndSendPanic, shouldFirePanic } from './chat_controls.js';
import { createChatAttachments } from './chat_attachments.js';
import { createChatLiveCards } from './chat_live_cards.js';
import { createComposer } from './chat_composer.js';
import { createFrameRouting } from './chat_frame_routing.js';
import { createHeaderControls } from './chat_header_controls.js';
import { createChatHistorySync } from './chat_history_sync.js';
import { createDocumentBubbles } from './chat_document_bubble.js';
import { createMessageIdentity } from './chat_message_identity.js';
import { createLiveCardView } from './chat_live_card_view.js';
import { createMediaBubbles } from './chat_media_bubbles.js';
import { createMessageAnnotations } from './chat_message_annotations.js';
import { showContextFitToast, showTaskIncidentToast } from './chat_notices.js';
import { createSubagentRouting } from './chat_subagent_routing.js';
import { createTaskFrames } from './chat_task_frames.js';
import { createTaskUiStateTracker } from './chat_task_ui_state.js';
import { createTimelineAnchors } from './chat_timeline_anchor.js';
import {
    headerBudgetPresentation,
    mergeStickyCostMeta,
    taskCostMeta,
    taskCostProjection,
    withTaskCostMeta,
} from './costs.js';

// Compatibility facade: chat.js keeps publishing every helper it used to own,
// bound to the owner's exact value, so external importers and identity tests
// see no change.
export {
    COLLAPSED_ACTIVITY_MAX,
    boundActivityPreview,
    clearStickyCardState,
    computeDerivedChatStatus,
    computeHydratedDirectActivities,
    confirmAndSendPanic,
    headerBudgetPresentation,
    insertTimelineNode,
    isTerminalTaskPhase,
    liveLineRowToggleKey,
    mergeStickyCostMeta,
    projectCollapsedActivity,
    rawTimestampEpoch,
    shouldFirePanic,
    taskCostMeta,
    taskCostProjection,
};

const CHAT_DRAFT_KEY = 'ouro_chat_draft';
const CHAT_INPUT_HISTORY_KEY = 'ouro_chat_input_history';
const CHAT_SESSION_ID_KEY = 'ouro_chat_session_id';
function getOrCreateChatSessionId() {
    try {
        const existing = sessionStorage.getItem(CHAT_SESSION_ID_KEY);
        if (existing) return existing;
        const created = (globalThis.crypto && typeof crypto.randomUUID === 'function')
            ? crypto.randomUUID()
            : `chat-${Date.now()}-${Math.random().toString(16).slice(2)}`;
        sessionStorage.setItem(CHAT_SESSION_ID_KEY, created);
        return created;
    } catch {
        return `chat-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    }
}

function loadInputHistory() {
    try {
        const raw = JSON.parse(sessionStorage.getItem(CHAT_INPUT_HISTORY_KEY) || '[]');
        return Array.isArray(raw) ? raw.filter(Boolean).slice(-50) : [];
    } catch {
        return [];
    }
}

function saveInputHistory(entries) {
    try {
        sessionStorage.setItem(CHAT_INPUT_HISTORY_KEY, JSON.stringify(entries.slice(-50)));
    } catch {}
}

export function initChat(ctx) {
    // Back-compat main-chat entry: one full-page instance bound to chat 1.
    return createChatInstance(ctx);
}

export function createChatInstance({
    ws, state, updateUnreadBadge, openSettingsTab, openDashboardTab,
    chatId = 1, projectId = '', idPrefix = 'chat', mountEl = null,
    asPanel = false, title = 'Chat', initialScrollState = null,
    // perf2 P4.2: app.js signal "a project panel is opening right now" — Main
    // defers its first hydration to it (bounded by an unconditional deadline).
    isProjectOpening = null,
}) {
    const container = mountEl || document.getElementById('content');
    const chatSessionId = getOrCreateChatSessionId();
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
            <div id="chat-attachment-preview" class="chat-attachment-preview"></div>
            <div class="chat-input-wrap">
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
                        <button class="chat-scroll-bottom-btn" id="chat-scroll-bottom" type="button" aria-label="Scroll to latest message" title="Scroll to latest message">
                            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 5v14"/><path d="M19 12l-7 7-7-7"/></svg>
                        </button>
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

    // Instance lifecycle (P3): destroy() flips this so rAF loops and late async
    // continuations become no-ops instead of touching a removed DOM subtree.
    let destroyed = false;
    // Every ws.on subscription's disposer, released together in destroy().
    const wsDisposers = [];
    const onWs = (event, fn) => wsDisposers.push(ws.on(event, fn));

    async function loadUiPreferences() {
        try {
            const prefs = await apiClient.uiPreferences();
            if (destroyed) return;
            setNestedSubagentsExpanded(prefs?.nested_subagents_expanded === true);
        } catch {
            setNestedSubagentsExpanded(false);
        }
    }


    const persistedHistory = [];
    const seenMessageKeys = new Set();
    const messageKeyOrder = [];
    const pendingUserBubbles = new Map();
    const inputHistory = loadInputHistory();
    let inputHistoryIndex = inputHistory.length;
    let inputDraft = '';
    // Per-instance viewport intent. Content growth does not emit a user scroll,
    // so `_savedStick` survives a large live-card mutation that would make a
    // post-mutation `isNearBottom()` check lose the owner's prior intent.
    // A recreated project instance seeds these from the scroll state stashed by
    // app.js when its predecessor was destroyed (single-live-panel policy);
    // `_initialScrollPending` defers the actual restore until first paint.
    let _savedScrollTop = Math.max(0, Number(initialScrollState?.scrollTop) || 0);
    let _savedStick = initialScrollState ? initialScrollState.stick !== false : true;
    let _restoring = false;
    let _viewportMutationDepth = 0;
    const isInstanceVisible = () =>
        Boolean(messagesDiv) && messagesDiv.offsetParent !== null && !document.hidden;
    const liveCardRecords = new Map();
    // Reusable slots (bg-consciousness, active) destroy+recreate their card on every
    // new cycle and auto-collapse on each cycle finish. Remember the owner's explicit
    // expand per slot so cycle churn restores it instead of snapping the card shut.
    const stickyExpandedSlots = new Set();
    // Cluster B: a proactively-coined name (task_named) can arrive BEFORE the card's
    // liveCardRecords entry exists (the namer broadcasts as the task starts). Buffer it
    // here so createLiveCardRecord can apply it when the card appears (no lost title).
    const pendingSuggestedNames = new Map();
    const taskUiStates = new Map();
    // Busy-chat decision turns reuse the normal agent/event path for ordering and
    // observability, but they are not user tasks. Their structural backend marker
    // suppresses the transient card while preserving the typed routing annotation
    // and any non-empty final conversational answer as separate UI roles.
    const ephemeralDecisionTaskIds = new Set();
    // Server-confirmed in-flight direct/ephemeral turns
    // (activityId -> { activityId, kind, phase, clientMessageId, startedAt }).
    const activeDirectActivities = new Map();
    // Local user submissions awaiting server confirmation (clientMessageId
    // -> { clientMessageId, timestamp }).
    const pendingSubmissions = new Map();
    // Conclusion ledger: ids concluded by a keyed final. Task ids never
    // restart, so late typing frames / stale snapshots must not resurrect a
    // concluded turn (project panels hydrate one-shot, no poll). Bounded FIFO.
    const concludedDirectActivities = new Map();
    const CONCLUDED_ACTIVITY_LEDGER_MAX = 200;
    // Local-echo journal: owner rows kept until server history confirms
    // their client_message_id (partitionLocalEchoJournal).
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
        const aid = String(activityId || '').trim();
        if (!aid) return;
        concludedDirectActivities.delete(aid);
        concludedDirectActivities.set(aid, Date.now());
        while (concludedDirectActivities.size > CONCLUDED_ACTIVITY_LEDGER_MAX) {
            const oldest = concludedDirectActivities.keys().next().value;
            concludedDirectActivities.delete(oldest);
        }
    }
    // Finished task ids hidden from routine syncs until reload/reconnect rebuilds history.
    const retiredTaskIds = new Set();

    const {
        buildMessageKey,
        rememberMessageKey,
        formatMsgTime,
        stampNodeTimestamp,
        getSenderLabel,
    } = createMessageIdentity({ chatSessionId, seenMessageKeys, messageKeyOrder });

    function setStatus(kind, text) {
        // perf2 P4.3: replay frames write the composer status once per batch
        // (last write wins), not once per historical frame.
        const batch = getRebuildBatch();
        if (batch) {
            batch.status = { kind, text };
            return;
        }
        if (!statusBadge) return;
        statusBadge.className = `status-badge ${kind}`;
        statusBadge.textContent = text;
    }

    const { syncHeaderControlState, refreshHeaderControlState } = createHeaderControls({
        byId, headerActions, state, hydrateDirectActivities,
    });
    const {
        isNearBottom,
        captureVisibleTimelineAnchor,
        restoreVisibleTimelineAnchor,
    } = createTimelineAnchors({ messagesDiv, liveCardRecords });

    function withStableViewport(mutate) {
        if (typeof mutate !== 'function') return undefined;
        if (_viewportMutationDepth > 0 || _restoring || !isInstanceVisible()) return mutate();

        const followBottom = _savedStick || isNearBottom();
        const anchor = followBottom ? null : captureVisibleTimelineAnchor();
        _viewportMutationDepth = 1;
        try {
            return mutate();
        } finally {
            _viewportMutationDepth = 0;
            if (isInstanceVisible()) {
                if (followBottom) messagesDiv.scrollTop = messagesDiv.scrollHeight;
                else restoreVisibleTimelineAnchor(anchor);
                _savedScrollTop = messagesDiv.scrollTop;
                _savedStick = followBottom || isNearBottom();
                updateScrollButton();
            }
        }
    }

    // v6.82 (P5): task ids whose progress carried the supervisor's host-attested
    // `cancelable` marker (queue tasks the cancel endpoint can genuinely reach).
    // Learned from live WS frames and history replay alike, possibly before the
    // card exists, so it lives beside the card records rather than on them.
    const cancelableTaskIds = new Set();

    // child task_id -> { parentId, role }, learned from subagent lifecycle pings.
    // Child cards are mounted under the parent card, but their phase/terminal
    // state is independent so a finished child cannot mark the parent done.
    const subagentChildParents = new Map();
    // Children whose card has reached a terminal phase — late non-lifecycle
    // progress for these must NOT revive it back to "working".
    const subagentTerminalChildren = new Set();

    // Live-card store owner (W3 wave D): records, reveal, re-anchoring, the
    // update pipeline and the terminal transitions live in the leaf; the
    // instance passes its collections and helpers explicitly and reads the
    // shared card-domain flags back through accessors.
    const {
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
        getActiveLiveGroupId,
        setActiveLiveGroupId,
        setPendingCardObjective,
        setNestedSubagentsExpanded,
        getLastTerminalAttention,
        setLastTerminalAttention,
        setSyncPass1Active,
        markLiveCardsDestroyed,
    } = createChatLiveCards({
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
        // The feed/history owner is constructed later in this body (it needs
        // the composer); these forwarders resolve at call time.
        insertMessageNode: (node, options) => insertMessageNode(node, options),
        stampNodeTimestamp,
        hideTypingIndicatorOnly,
        syncChatStatus,
        scheduleHistorySync: () => scheduleHistorySync(),
        hasActiveLiveCard,
        getRebuildBatch: () => getRebuildBatch(),
    });


    const {
        isBackgroundTaskId,
        shouldAlwaysShowTaskCard,
        isForegroundLiveCard,
        getTaskUiState,
        scheduleTaskUiCleanup,
        bufferLiveUpdate,
        markTaskToolCall,
        forceTaskCard,
        markAssistantReply,
        markTaskComplete,
    } = createTaskUiStateTracker({ taskUiStates, retiredTaskIds, revealBufferedCardIfNeeded });

    const {
        turnTaskIntoProject,
        syncCancelRunButton,
        markTaskCancelable,
        markLiveCardFinalizing,
    } = createCardActions({
        liveCardRecords,
        cancelableTaskIds,
        withStableViewport,
        finishLiveCard,
        signalChatFreed,
    });
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

    const {
        applySuggestedName,
        renderCollapsedActivity,
        ensureSubagentContainer,
        setLiveCardTypingVisible,
        formatLiveCardPhaseLabel,
        setLiveCardExpanded,
        syncLiveCardToggle,
        directSubagentCount,
        renderLiveCardTimeline,
        appendTimelineItem,
        patchLastTimelineItem,
        patchTimelineItemAt,
        renderLiveCardMeta,
    } = createLiveCardView({
        liveCardRecords,
        pendingSuggestedNames,
        withStableViewport,
        getLiveCardRecord,
        syncLiveCardLayout,
    });

    bindLiveCardCollaborators({
        isBackgroundTaskId,
        shouldAlwaysShowTaskCard,
        getTaskUiState,
        bufferLiveUpdate,
        markTaskComplete,
        turnTaskIntoProject,
        syncCancelRunButton,
        renderCollapsedActivity,
        ensureSubagentContainer,
        setLiveCardTypingVisible,
        formatLiveCardPhaseLabel,
        setLiveCardExpanded,
        syncLiveCardToggle,
        directSubagentCount,
        renderLiveCardTimeline,
        appendTimelineItem,
        patchLastTimelineItem,
        patchTimelineItemAt,
        renderLiveCardMeta,
    });

    // Re-sync cards after SPA return or browser tab visibility restore, then put
    // the thread back where the user left it (P7) instead of at the very top.
    // Named handlers so destroy() can remove them (P3 lifecycle).
    const handlePageShown = (event) => {
        if (event?.detail?.page !== 'chat') return;
        for (const record of liveCardRecords.values()) {
            if (record?.root?.isConnected) syncLiveCardLayout(record);
        }
        restoreScrollPosition();  // no-op for hidden panel instances
    };
    window.addEventListener('ouro:page-shown', handlePageShown);
    const handleVisibilityChange = () => {
        if (document.hidden) return;
        if (state.activePage !== 'chat') return;
        for (const record of liveCardRecords.values()) {
            if (record?.root?.isConnected && record._needsLayoutSync) syncLiveCardLayout(record);
        }
    };
    document.addEventListener('visibilitychange', handleVisibilityChange);


    const {
        setSubagentParent,
        summarizeSubagentCardFrame,
        updateSubagentCardFromEvent,
        routeSubagentProgressToCard,
        routeSubagentFinalMessageToCard,
        routeSubagentTerminalToCard,
    } = createSubagentRouting({
        subagentChildParents,
        subagentTerminalChildren,
        withTaskCostMeta,
        forceTaskCard,
        getTaskUiState,
        getSubagentCardRecord,
        queueTaskLiveUpdate,
    });

    // Task-frame router (W3 wave D): task_summary rows, live progress and
    // grouped log events project onto the live cards through the leaf.
    const {
        appendTaskSummaryToLiveCard,
        updateLiveCardFromProgressMessage,
        updateLiveCardFromLogEvent,
    } = createTaskFrames({
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
    });

    const {
        renderRoutingAnnotation,
        updateMessageAnnotation,
        clearTransientRoutingAnnotations,
        markPendingDelivered,
    } = createMessageAnnotations({ messagesDiv, pendingUserBubbles, localEchoJournal });

    function markPendingDropped(clientMessageId) {
        const bubble = pendingUserBubbles.get(clientMessageId || '');
        if (!bubble) return;
        const note = bubble.querySelector('.msg-pending');
        if (note) note.textContent = 'Not delivered — send again';
        pendingUserBubbles.delete(clientMessageId);
    }

    function rememberInput(text) {
        if (!text) return;
        if (inputHistory[inputHistory.length - 1] !== text) inputHistory.push(text);
        saveInputHistory(inputHistory);
        inputHistoryIndex = inputHistory.length;
        inputDraft = '';
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
        resizeChatInput({ preserveStickiness: false });
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
        const hasAttachments = hasPendingAttachments();
        let uploadedAttachments = [];
        let attachmentMeta = [];
        if (!text && !hasAttachments) return;
        if (hasAttachments) {
            // Upload immediately before send; offline queueing would orphan files.
            if (ws.ws?.readyState !== WebSocket.OPEN) {
                showToast('Cannot attach file while offline. Reconnect and try again.', 'error');
                return;
            }
            const staged = stagedAttachmentItems();
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
        if (isMain && objectiveText) setPendingCardObjective(objectiveText);
        if (hasAttachments) {
            clearPendingAttachments();
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
        resizeChatInput({ preserveStickiness: false });
        scrollToBottomAfterLayout();
    }

    // Send mode lives on DOM so CSS and click/Enter share one source.
    const sendGroup = page.querySelector('.chat-send-group');

    // Swarm is a one-shot arm: the next send goes through plan_task multi-model
    // brainstorm/planning, then the pill auto-disarms so it never sticks.
    const swarmBtn = byId('swarm');
    const {
        resizeChatInput,
        swarmArmed,
        setSwarm,
        setSendBusy,
        scrollToBottom,
        updateScrollButton,
        updateMessagesPadding,
    } = createComposer({
        page,
        input,
        inputArea,
        pageHeader,
        messagesDiv,
        sendBtn,
        sendGroup,
        swarmBtn,
        scrollBottomBtn,
        isInstanceVisible,
        isNearBottom,
        scrollToBottomAfterLayout,
    });
    swarmBtn?.addEventListener('click', () => setSwarm(!swarmArmed()));

    // Attachment staging owner (W3 wave D): the leaf wires the paperclip,
    // paste and drag/drop listeners itself; send-time consumers reach the
    // staged files only through these accessors.
    const {
        updateAttachmentPreview,
        cleanupUploadedAttachments,
        setAttachmentUploadState,
        hasPendingAttachments,
        stagedAttachmentItems,
        clearPendingAttachments,
        isAttachmentUploadBusy,
    } = createChatAttachments({
        page,
        input,
        inputArea,
        attachBtn,
        fileInput,
        attachmentPreview,
        updateMessagesPadding,
    });

    const { buildDocumentBubble, documentMessageKey, appendDocumentBubble } = createDocumentBubbles({
        seenMessageKeys,
        getSenderLabel,
        formatMsgTime,
        stampNodeTimestamp,
        rememberMessageKey,
        insertMessageNode: (node, options) => insertMessageNode(node, options),
    });

    // History/feed owner (W3 wave D): hydration and syncHistory replay, the
    // feed mount primitives, the sessionStorage bootstrap, the Load-older
    // control and the socket-open resync live in the leaf; the instance
    // passes its collections and helpers explicitly and reads the replay
    // batch handle back through getRebuildBatch.
    const {
        insertMessageNode,
        addMessage,
        scheduleHistorySync,
        cancelHistoryPaint,
        refreshHistory,
        hasPaintedHistory,
        handleSocketOpen,
        getRebuildBatch,
        cancelPendingHistoryResync,
        markHistoryDestroyed,
    } = createChatHistorySync({
        ws,
        isMain,
        chatId,
        page,
        messagesDiv,
        storeKey,
        chatSessionId,
        initialScrollPending: Boolean(initialScrollState) && !_savedStick,
        isProjectOpening,
        persistedHistory,
        seenMessageKeys,
        messageKeyOrder,
        pendingUserBubbles,
        inputHistory,
        localEchoJournal,
        pendingSubmissions,
        retiredTaskIds,
        liveCardRecords,
        taskUiStates,
        ephemeralDecisionTaskIds,
        pendingSuggestedNames,
        cancelableTaskIds,
        subagentChildParents,
        subagentTerminalChildren,
        activeDirectActivities,
        buildMessageKey,
        rememberMessageKey,
        formatMsgTime,
        getSenderLabel,
        stampNodeTimestamp,
        renderRoutingAnnotation,
        appendDocumentBubble,
        isNearBottom,
        captureVisibleTimelineAnchor,
        restoreVisibleTimelineAnchor,
        withStableViewport,
        updateMessagesPadding,
        updateScrollButton,
        scrollToBottomAfterLayout,
        restoreScrollPosition,
        isViewportSticky: () => _savedStick,
        setStatus,
        syncChatStatus,
        hideTypingIndicatorOnly,
        hasActiveLiveCard,
        loadUiPreferences,
        refreshHeaderControlState,
        setActiveLiveGroupId,
        setSyncPass1Active,
        finishLiveCard,
        ensureLiveCardVisible,
        getTaskUiState,
        markLiveCardFinalizing,
        updateLiveCardFromProgressMessage,
        appendTaskSummaryToLiveCard,
        setSubagentParent,
        routeSubagentFinalMessageToCard,
        routeSubagentTerminalToCard,
        renderLiveCardMeta,
        updateLiveCardCount,
        syncLiveCardLayout,
        saveInputHistory,
        setInputHistoryIndex: (index) => { inputHistoryIndex = index; },
    });

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
    function scrollToBottomAfterLayout() {
        requestAnimationFrame(() => {
            if (destroyed) return;
            scrollToBottom();
            requestAnimationFrame(() => { if (!destroyed) scrollToBottom(); });
        });
    }

    // P7 — per-instance scroll memory. Switching tabs/opening a project panel used
    // to drop this thread back to the very top (the browser zeroes a hidden
    // column's scrollTop, and toggling .page display can reset it too). We
    // remember where the user was and restore it on show: pinned to the latest
    // message in the common case, or the exact spot they'd scrolled back to.
    messagesDiv?.addEventListener('scroll', () => {
        // Ignore the spurious scrollTop=0 a browser emits while the column is
        // hidden — that would erase the real position we want to restore.
        if (!isInstanceVisible()) return;
        // WebKit fires a scrollTop=0 event when a hidden column is re-shown, and
        // our own re-pin loop writes scrollTop too — neither is a real user
        // scroll, so during a restore pass we only refresh the button, never the
        // saved position (which would corrupt a mid-history restore to the top).
        if (_restoring) { updateScrollButton(); return; }
        _savedScrollTop = messagesDiv.scrollTop;
        _savedStick = isNearBottom();
        updateScrollButton();
    }, { passive: true });

    scrollBottomBtn?.addEventListener('click', () => {
        _savedStick = true;
        scrollToBottomAfterLayout();
        updateScrollButton();
    });

    function restoreScrollPosition() {
        if (!isInstanceVisible()) return;  // hidden column has no geometry yet
        // WebKit (the desktop WKWebView) leaves a freshly un-hidden flex column's
        // scrollTop pinned at 0 for a frame or two after the page is shown, so a
        // single/double rAF re-pin (which is enough in Chromium) lands the user at
        // the very top. Re-apply the target position across several frames until
        // the late relayout settles, then keep the button state in sync.
        _restoring = true;
        const targetStick = _savedStick;
        const targetTop = _savedScrollTop;
        let frames = 0;
        const apply = () => {
            if (destroyed || !isInstanceVisible()) { _restoring = false; return; }
            // scrollHeight is re-read each frame so a sticky thread tracks late
            // card-layout growth; a restored mid-history spot re-pins to the exact
            // saved offset (idempotent, so it isn't overridden).
            messagesDiv.scrollTop = targetStick ? messagesDiv.scrollHeight : targetTop;
            updateScrollButton();
            if (++frames < 12) requestAnimationFrame(apply);
            else _restoring = false;
        };
        requestAnimationFrame(apply);
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
                updateMessagesPadding({ preserveStickiness: true });
            });
        };
        chatResizeObserver = new ResizeObserver(schedule);
        if (pageHeader) chatResizeObserver.observe(pageHeader);
        if (inputArea) chatResizeObserver.observe(inputArea);
        if (messagesDiv) chatResizeObserver.observe(messagesDiv);
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
            resizeChatInput({ preserveStickiness: false });
        }
    } catch {}

    input.addEventListener('input', () => {
        if (inputHistoryIndex === inputHistory.length) inputDraft = input.value;
        resizeChatInput({ preserveStickiness: false });
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
        // The panel has no global controls/budget to poll; seed the status from
        // the live socket so a late-created panel never gets stuck on
        // "Connecting…" (the one-shot WS `open` already fired before it existed;
        // future reconnects still update it via the shared `open` handler).
        if (ws.isConnected?.()) setStatus('online', 'Online');
        // 1A: a panel created AFTER the socket opened missed the typing frame
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


    function hasActiveLiveCard() {
        return Array.from(liveCardRecords.values()).some(isForegroundLiveCard);
    }

    function deriveChatStatus() {
        let directCount = 0, managedActive = 0, managedQueued = 0;
        for (const entry of activeDirectActivities.values()) {
            if (String(entry?.kind || '') !== 'managed_task') directCount += 1;
            else if (String(entry?.phase || '') === 'queued') managedQueued += 1;
            else managedActive += 1;
        }
        return computeDerivedChatStatus({
            isConnected: ws.isConnected ? ws.isConnected() : true,
            hasActiveLiveCard: hasActiveLiveCard(),
            activeDirectCount: directCount,
            activeManagedCount: managedActive,
            queuedManagedCount: managedQueued,
            pendingSubmissionsCount: pendingSubmissions.size,
            lastTerminalAttention: getLastTerminalAttention(),
        });
    }

    function syncChatStatus() {
        const derived = deriveChatStatus();
        setStatus(derived.kind, derived.text);
        if (derived.showDots && !hasActiveLiveCard()) {
            typingEl.style.display = '';
            if (isNearBottom()) messagesDiv.scrollTop = messagesDiv.scrollHeight;
        } else {
            typingEl.style.display = 'none';
        }
    }

    function showTyping(activityId = '', meta = {}) {
        const actId = String(activityId || '').trim() || ('direct-' + chatId);
        // A typing frame after its turn's keyed final must not resurrect the
        // concluded turn — but it still carries the activity<->cmid link, so
        // it settles the linked submission (broadcasts are not ordered).
        if (concludedDirectActivities.has(actId)) {
            if (meta.clientMessageId && pendingSubmissions.delete(meta.clientMessageId)) {
                syncChatStatus();
            }
            return;
        }
        activeDirectActivities.set(actId, {
            activityId: actId,
            // '' = not registry-tracked (queued managed task): visible in the
            // active set but exempt from /api/state snapshot deletion.
            kind: meta.kind || '',
            phase: meta.phase || 'thinking',
            clientMessageId: meta.clientMessageId || '',
            startedAt: Date.now(),
        });
        if (meta.clientMessageId) {
            pendingSubmissions.delete(meta.clientMessageId);
        }
        setLastTerminalAttention(false);
        syncChatStatus();
    }

    function hideTypingIndicatorOnly() {
        // perf2 P4.3: one typing-indicator write per replay batch.
        const batch = getRebuildBatch();
        if (batch) {
            batch.typingHidden = true;
            return;
        }
        typingEl.style.display = 'none';
    }

    function hydrateDirectActivities(turnsList, snapshotBarrierMs = Infinity) {
        if (!Array.isArray(turnsList)) return;
        const nextMap = computeHydratedDirectActivities(
            activeDirectActivities, turnsList, chatId, snapshotBarrierMs, concludedDirectActivities);
        activeDirectActivities.clear();
        for (const [k, v] of nextMap.entries()) {
            activeDirectActivities.set(k, v);
            if (v.clientMessageId) {
                pendingSubmissions.delete(v.clientMessageId);
            }
        }
        syncChatStatus();
    }

    const {
        isKnownProjectFrame,
        incrementUnreadIfNeeded,
        isProjectMirrorFrame,
        isMyThread,
    } = createFrameRouting({ state, isMain, chatId, updateUnreadBadge });
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

    onWs('chat', (msg) => {
        if (!isMyThread(msg, { mirrorProject: true })) return;
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
            addMessage(msg.content, 'user', false, msg.ts || null, false, {
                source: msg.source || '',
                senderLabel: msg.sender_label || '',
                senderSessionId,
                clientMessageId,
                taskId: msg.task_id || '',
            });
            incrementUnreadIfNeeded(msg);
            syncChatStatus();
            return;
        }

        if (msg.role === 'assistant' || msg.role === 'system') {
            const explicitTaskId = msg.task_id || '';
            const ephemeralDecision = registerEphemeralDecisionFrame(msg);
            // 3A: Main mirrors Project frames as штаб presentation only — a
            // mirrored ephemeral turn never enters THIS instance's active set.
            const isMirror = isMain && isKnownProjectFrame(msg);
            if (ephemeralDecision && explicitTaskId && !isMirror) {
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
                if (ephemeralDecision) return;
                updateLiveCardFromProgressMessage(msg);
                syncChatStatus();
                return;
            }

            // An early final (post-task still running) is NOT the turn's
            // conclusion; task_done or the queue snapshot concludes it.
            const finalizing = Boolean(explicitTaskId) && msg.task_phase === 'finalizing';
            if (!isMirror && !finalizing) {
                if (explicitTaskId) {
                    // 4A (active set): a keyed final concludes ITS OWN turn —
                    // the finished activity + its linked pending — never a
                    // concurrent turn's state (2A keeps later `Sending...`).
                    const finished = activeDirectActivities.get(explicitTaskId);
                    activeDirectActivities.delete(explicitTaskId);
                    recordConcludedActivity(explicitTaskId);
                    if (finished?.clientMessageId) {
                        pendingSubmissions.delete(finished.clientMessageId);
                    }
                } else {
                    // A bare (unkeyed) final cannot be scoped: clear the set
                    // but NEVER ledger — no proof any specific turn ended; a
                    // live turn stays restorable by typing frame or snapshot.
                    activeDirectActivities.clear();
                    pendingSubmissions.clear();
                }
            }

            if (msg.system_type === 'task_summary') {
                appendTaskSummaryToLiveCard(msg);
                markAssistantReply(explicitTaskId);
                incrementUnreadIfNeeded(msg);
                syncChatStatus();
                return;
            }
            if (explicitTaskId && subagentChildParents.has(explicitTaskId)) {
                routeSubagentFinalMessageToCard(explicitTaskId, msg);
                markAssistantReply(explicitTaskId);
                incrementUnreadIfNeeded(msg);
                syncChatStatus();
                return;
            }
            if (finalizing) markLiveCardFinalizing(explicitTaskId);
            else if (explicitTaskId) finishLiveCard(explicitTaskId);
            if (!finalizing) markAssistantReply(explicitTaskId);
            clearTransientRoutingAnnotations();
            addMessage(msg.content, msg.role, msg.markdown, msg.ts || null, false, {
                systemType: msg.system_type || '',
                source: msg.source || '',
                taskId: explicitTaskId,
            });
            incrementUnreadIfNeeded(msg);
            syncChatStatus();
        }
    });

    onWs('message_annotation', (msg) => {
        if (!isMyThread(msg)) return;
        if (msg.annotation_type !== 'routing_ack') return;
        updateMessageAnnotation(msg.client_message_id || '', msg);
        // Any routing receipt is the durable disposition of the submission and
        // ends its `Sending...` phase; further activity announces itself via
        // its own typing frame or task card.
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
        // Log frames now carry the task's chat_id (backend stamps it), so the
        // per-thread fan-out routes the full live card to its own column: a
        // project panel builds/animates/finalizes ITS card, while the main
        // chat mirrors project progress as штаб. Legacy frames without chat_id
        // default to the main chat.
        if (!isMyThread(msg, { mirrorProject: true })) return;
        updateLiveCardFromLogEvent(msg.data);
    });

    // Cluster B: the proactive namer coined a project name for a fresh card — show it
    // as the card title up front (turn-into-project then reuses the same name). Not
    // thread-gated on chat_id: the broadcast carries only task_id, and applySuggestedName
    // no-ops unless THIS thread already holds that card.
    onWs('task_named', (msg) => {
        applySuggestedName(msg?.task_id || '', msg?.suggested_name || '');
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

    // Media bubble owner (W3 wave D): photo/video frames render in the leaf.
    const { handlePhotoFrame, handleVideoFrame } = createMediaBubbles({
        isMyThread,
        hideTypingIndicatorOnly,
        syncChatStatus,
        getSenderLabel,
        formatMsgTime,
        stampNodeTimestamp,
        insertMessageNode,
        incrementUnreadIfNeeded,
    });
    onWs('photo', handlePhotoFrame);
    onWs('video', handleVideoFrame);

    onWs('document', (msg) => {
        if (!isMyThread(msg)) return;
        hideTypingIndicatorOnly();
        syncChatStatus();
        if (appendDocumentBubble(msg)) incrementUnreadIfNeeded(msg);
    });

    onWs('open', handleSocketOpen);

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
        // True once a history snapshot has actually been fetched and painted;
        // app.js uses it to decide whether a reopen needs a forced repaint.
        hasPaintedHistory,
        // Unsendable client-side state (staged File objects / an in-flight
        // upload). app.js must hide, not destroy, an instance holding it.
        hasPendingWork: () => hasPendingAttachments() || isAttachmentUploadBusy(),
        // Viewport intent stash source for the single-live-panel policy.
        getScrollState: () => ({ scrollTop: _savedScrollTop, stick: _savedStick }),
        // Full teardown (P3): release every resource this instance acquired —
        // ws subscriptions, window/document listeners, the ResizeObserver, all
        // timers — then drop the buffered collections and remove the DOM last.
        // Idempotent; late rAF/async continuations no-op on `destroyed`.
        destroy() {
            if (destroyed) return;
            destroyed = true;
            markLiveCardsDestroyed();
            markHistoryDestroyed();
            cancelHistoryPaint();
            for (const dispose of wsDisposers) {
                try { dispose(); } catch {}
            }
            wsDisposers.length = 0;
            window.removeEventListener('ouro:page-shown', handlePageShown);
            document.removeEventListener('visibilitychange', handleVisibilityChange);
            if (documentClickHandler) document.removeEventListener('click', documentClickHandler);
            if (documentKeydownHandler) document.removeEventListener('keydown', documentKeydownHandler);
            chatResizeObserver?.disconnect();
            chatResizeObserver = null;
            cancelPendingHistoryResync();
            if (_chatFreedTimer) { clearTimeout(_chatFreedTimer); _chatFreedTimer = null; }
            for (const taskState of taskUiStates.values()) {
                if (taskState?.cleanupTimer) clearTimeout(taskState.cleanupTimer);
            }
            if (headerControlInterval) { clearInterval(headerControlInterval); headerControlInterval = null; }
            liveCardRecords.clear();
            taskUiStates.clear();
            pendingSuggestedNames.clear();
            subagentChildParents.clear();
            subagentTerminalChildren.clear();
            cancelableTaskIds.clear();
            ephemeralDecisionTaskIds.clear();
            retiredTaskIds.clear();
            pendingUserBubbles.clear();
            localEchoJournal.clear();
            seenMessageKeys.clear();
            messageKeyOrder.length = 0;
            persistedHistory.length = 0;
            try { page.remove(); } catch {}
        },
    };
}
