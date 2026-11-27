import { escapeHtmlAttr, isoMicros } from './utils.js';
import { tr } from './i18n.js';
import { bindCompactMarkdown, destroyChatMarkdown, enhanceChatMarkdown, mountChatMarkdown } from './chat_markdown.js';
import { clearReloadParams } from './layers.js';
import { fluentIcon, renderSkillIcon } from './page_icons.js';
import { bindScrollFade } from './scroll_fade.js';
import { createRoomHeader, roomBarHtml } from './chat_header.js';
import { placeRoomEdges, ROOM_EDGE_HTML, watchRoomEdges } from './page_header.js';
import { pageLabel } from './pages.js';
import { showToast } from './toast.js';
import { roomCovered, roomNotice, watchRoomCover } from './room_notice.js';
import { ORIGIN_ADDRESSED_NOTICE_TYPES, lineReplyFields, childRowPlacement, placedCardRow, createPlacedCardAnchors, replayConcludesTask } from './chat_row_placement.js';
import { appendProjectRow, PROJECT_ROW_TYPES, refreshProjectAnswerFold, syncSavedProjectContext } from './project_answer.js';
import { createWorkPortals, isPortal, reconcilePortalNotes } from './work_portal.js';
import { createTaskEpisodes } from './task_episodes.js';
import { markSwarmLabel, syncSwarmReceipt } from './swarm_presentation.js';
import { syncCostBreakdown } from './cost_breakdown_presentation.js';
import { createChatMedia, showTaskIncidentToast } from './chat_media.js';
import { createEarlyTextBubbles, createTextBubbles } from './chat_text_bubble.js';
import { endLetters, letterRow } from './peer_letter_card.js';
import { noticeLine } from './host_notice_text.js';
import { createCatchUp, createWindowReturn } from './room_retire.js';
import { createChatPreview, createRoomProfileLifecycle } from './chat_preview.js';
import { createUnconfirmedSends } from './chat_attachments.js';
import { createChatDecision } from './chat_decision.js';
import { createRoomWork } from './room_work.js';
import { SERVER_ANSWERS_EVENT } from './ws.js';
import { disposeCardModel, followCardCatalogue } from './card_model.js';
import { applyCardWorkViews, cardControls, changedOffers, emptyWorkState, memberActivityEvidenced, memberState, noteControlQueue, ownStopPress, promoteChildCard, reconcileWorkState, roomStatusView, syncCardControls } from './work_status.js';
import { statusMarkHtml } from './status_mark.js';
import { createMainWaiting } from './main_waiting.js';
import { createHostStart } from './host_starting_status.js';
import { createModelWaitController, isModelWaitReference } from './model_wait.js';
import { acceptLiveFrame, applyCardEnding, attemptOf, cardDetailReader, cardMetricFacts, closingLine, foldCardMetrics, hydrateOpenSettledChild, hydrateRevealedCard, hydrateRevealedChildren, noteHandoverAttempt, readCardDetail, readMissingCardDetail, refreshSettledCardDetail, replayedTerminalRecords, syncClosingSentence, syncSettledItems, terminalChildFrame } from './settled_card.js';
import { noteOrigin, paintOrigin } from './task_origin.js';
import { createChatReadingPosition } from './chat_reading_position.js';
import { createRowSleep } from './chat_row_sleep.js';
import { onRefit } from './text_fit.js';
import { createChatHistoryPager, historyCoverage, historyIslandAtEdge } from './chat_history.js';
import { createProjectReadReceipt, isAtNewestMessage, newestMessageOffset, readBand } from './project_read_state.js';
import { createSharedReadTracker } from './read_ahead.js';
import { adoptsRetainedCopy, createReplayKeys, livePriorBubble, messageIdentity, priorHistoryBubble, ownerMessageBubble, ownerRowRoom, mergeHistoricalTimelineItem, historyNodeIsProtected, historyRowIds, historyStamps, stampHistoryNode, compareHistoryPosition } from './chat_history_replay.js';
import { apiClient, apiFetch, fetchTaskDetailStrict, isStoreMigrating } from './api_client.js';
import { readStateSnapshot } from './state_snapshots.js';
import { syncHistoryRetentionItem } from './history_retention.js';
import {
    childCauseFacts,
    getLogTaskGroupId,
    isGroupedTaskEvent,
    isTerminalTaskDetail,
    lateOutcomeIsCurrent,
    normalizeLogTs,
    ownerHurryProjection,
    summarizeChatLiveEvent,
    taskCancelPending,
    taskOutcomeSeverity,
    taskPresentation,
    taskSoftStopPending,
    taskDoneIsTerminal,
    taskTerminalSummary,
    taskTerminalPhase,
} from './log_events.js';
import { admissionRefusalRow, cardExecutorChip, keepStickyExecutorChip } from './executor_card_facts.js';
import {
    ACTION_FINALIZE,
    REUSABLE_TASK_IDS,
    cancelRunEligibility,
    requestStop,
} from './task_control_menu.js';
import { composerMarkup, createComposer } from './composer.js';
import { composerHeightAtRest, syncShortRoom } from './composer_fold.js';
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
    frameObservesPhase,
    markOwnerHurry,
    observeChildPhase,
    observeChildStart,
    replayTerminalPhase,
    restoreLiveCardPhaseState,
    setLiveCardPhase,
    setHistoricalUnavailable,
    setHistoricalUnconfirmed,
    syncParkedPhase,
    updateLiveCardCount,
} from './task_phase_chip.js';
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
    childProgressKey,
    upsertToolFoldRow,
} from './chat_render_batch.js';
import { measure } from './layout_batch.js';
import { archiveDoor } from './room_archive.js';
import { placeToolDiagnostic, placeReplayedEvidence } from './verification_presentation.js';
import {
    boundActivityPreview,
    authoredLine,
    authoredTitle,
    buildTimelineItemHtml,
    chatLogThreadAccepts,
    chatMediaMessageKey,
    chatThreadAccepts,
    clearStickyCardState,
    documentMessageKey,
    durableChatMediaUrl,
    formatMsgTime,
    getOrCreateChatSessionId,
    isForegroundLiveCard,
    isReplayEvidenceRow,
    isTerminalTaskPhase,
    markIngressSaved,
    bindLiveCardTimeline,
    subagentIdentityTitle,
    subagentTwin,
    mergeStickyCostMeta,
    applyToolObservation,
    noteToolHostMetrics,
    partitionLocalEchoJournal,
    projectCollapsedActivity,
    positiveTaskTerminalFact,
    TERMINAL_SUBAGENT_EVENTS,
    unkeyedFrameEndsTurn,
    stampNodeTimestamp,
    reconcileHydratedDirectActivities,
    restartNoticeText,
    senderLabel,
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
    dockChildReviewDoor,
    cardRowEvidenceRef,
    refitCollapsedActivity,
} from './chat_activity.js';


const SUBAGENT_LIFECYCLE_EVENTS = new Set(['scheduled', 'running', 'interrupted', ...TERMINAL_SUBAGENT_EVENTS]);
// Details and Reviews use the shared control glyph; ::after remains the touch hit area.
const CHEVRON = renderSkillIcon('chevron', { size: 14, className: 'chat-live-chevron' });


export function initChat(ctx) {
    // Back-compat main-chat entry: one full-page instance bound to chat 1.
    return createChatInstance(ctx);
}

const taskKey = (value) => String(value || '').trim();

export function createChatInstance({
    ws, state, openSettings, openDashboardTab,
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
    const roomProfile = createRoomProfileLifecycle({
        ws, doc: document, chatId, isMain, isProjectOpening,
        isDestroyed: () => destroyed, isLeaving: () => leaving,
        readHistory: args => apiClient.chatHistory(args), nextRecentVersion: () => ++recentRequest,
        syncHistory, loadUiPreferences, getPreview: () => preview, reconcileSavedSends, kept: () => unconfirmed,
        onPreviewShown: () => { updateMessagesPadding(); reading.followAfterLayout(true); },
    });
    const { storeKey, foreignProfile, fetchHistory, awaitInitialHydration, waitForHydrationWindow } = roomProfile;

    const page = document.createElement('div');
    page.id = asPanel ? `panel-${idPrefix}` : 'page-chat';
    page.className = asPanel ? 'chat-instance-panel' : 'page active';
    page.innerHTML = `
        ${roomBarHtml({ asPanel, title })}
        <div id="chat-messages"></div>
        <div id="chat-input-area">${ROOM_EDGE_HTML}
            <div class="chat-input-wrap">
                <button class="chat-scroll-bottom-btn ui-hit-area" id="chat-scroll-bottom" type="button" aria-label="Scroll to latest message" title="Scroll to latest message">
                    ${fluentIcon('arrow-down', 16, 'regular')}
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
    // ↓ latest stands on the capsule line, at its right end (style.css .chat-scroll-bottom-btn).
    page.querySelector('.chat-toolbar-row')?.append(page.querySelector('.chat-scroll-bottom-btn'));

    const byId = (suffix) => page.querySelector(`[id="${idPrefix}-${suffix}"]`);
    const messagesDiv = byId('messages');
    const input = byId('input');
    const inputArea = byId('input-area');
    const statusBadge = byId('status');
    const pageHeader = page.querySelector('.chat-page-header') || (asPanel ? container?.closest?.('.project-panel')?.querySelector?.('.project-panel-bar') : null);
    const scrollBottomBtn = byId('scroll-bottom');
    const scrollActivityDot = scrollBottomBtn?.querySelector('.chat-scroll-activity-dot');
    let nestedSubagentsExpanded = false;
    let _remoteActivityDepth = 0;
    // The composer (composer.js), wired once the room's feed owners exist; Main's bar (chat_header.js), once its status.
    let composer = null, header = null;

    // Destroyed instances ignore late async work and animation frames.
    let destroyed = false;
    // Every ws.on subscription's disposer, released together in destroy().
    const wsDisposers = [];
    const onWs = (event, fn) => wsDisposers.push(ws.on(event, fn));
    const replies = createChatReplies({ chatId, feed: messagesDiv, input, composerArea: inputArea, key: storeKey('ouro_chat_reply'),
        reading: () => reading, onDomWrite: withStableViewport, history: () => historyReveal,
        navigate: revealQuotedSource, present: () => !destroyed && isInstanceVisible(), onDraftChange: () => composer?.invalidateDraft() });
    const chatMedia = createChatMedia({
        durableChatMediaUrl,
        formatMsgTime,
        insertMessageNode,
        senderLabel,
        stampNodeTimestamp,
        onDomWrite: withStableViewport,
        replies, composer: () => composer,
    });
    // A text row as its bubble and doors (chat_text_bubble.js); addMessage keeps its identity and its place.
    const drawTextBubble = createTextBubbles({ chatMedia, replies, feed: messagesDiv, isMain, ownProjectId: projectId,
        onDomWrite: withStableViewport });
    const mainWaiting = isMain ? createMainWaiting({ page, messagesDiv, insertMessageNode, stampNodeTimestamp,
        openSettings, refresh: refreshHeaderControlState, mutate: withStableViewport,
        decision: () => chatDecision, revealQuestion: (...a) => revealQuestion(...a), reading: () => reading, noticeVisible: () => historyVisible() }) : null;
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

    // Sent attachment messages not yet confirmed saved: "Send again" with one id, kept by this tab across a reload.
    const unconfirmed = createUnconfirmedSends({ send: (frame, options) => ws.send(frame, options),
        root: () => messagesDiv, onDomWrite: withStableViewport, showToast, compose: (...args) => composer?.prepareNewMessage(...args),
        // A shown copy whose word lapsed leaves the feed; its key goes with it, so a later word draws it again.
        release: node => withStableViewport(() => (seenMessageKeys.delete(node.dataset.messageKey), releaseMessageNode(node), true),
            { excludeAnchorNode: node }),
        storage: sessionStorage, storageKey: storeKey('ouro_chat_unconfirmed') });

    let _syncPass1Active = false;
    let _historyReplayActive = false;
    let _historyRow = null;
    let _historyAppending = false;
    // During a page replay, the keys its mounted bubbles carry (createReplayKeys)...
    let replayKeys = null;
    // ...and the selection as the reader left it: every mounted row the page moves keeps it (one read per page, not
    // per move: in Chromium a selection read lays the page out).
    let replaySelection = null;
    function noteBubbleKeys(node) { replayKeys?.note(node); }

    const seenMessageKeys = new Set();
    const messageKeyOrder = [];
    const pendingUserBubbles = new Map();
    // A source read has painted the room (the saved preview never counts).
    let historyLoaded = false;
    let historySyncPromise = null;
    let lastHistorySyncSucceeded = false;
    // One derived physical coverage/status for the mounted reading window.
    let historyWindow = null;
    let _hasNewActivity = false;
    // A room leaving the screen or away (room_retire.js): not shown, not read; `outageAway`: the sends a disconnect doubted.
    let leaving = false;
    let visit = 0, historyReadAbort = null, outageAway = null;
    const isInstanceVisible = () =>
        !leaving && Boolean(messagesDiv) && messagesDiv.offsetParent !== null && !document.hidden;
    const LIVE_CARD_CAP = 200;
    const liveCardBound = createLiveCardBound(LIVE_CARD_CAP);
    const liveCardRecords = new Map();
    // The server's work projection (work_status.js), re-read on facts only.
    let roomWork = null, workState = emptyWorkState();
    const reread = () => void refreshHeaderControlState();
    // A wait creates its task record; its open/closed state feeds the block predicate, so the block appears and leaves
    // with the controls (the record and its no-reopen wait ledger stay).
    const modelWaits = createModelWaitController({
        getRecord: (id, create = true) => (create ? getLiveCardRecord(id) : liveCardRecords.get(id)),
        openSettings, onDomWrite: withStableViewport, notify: showToast,
        onChange: (id) => { ensureLiveCardVisible(liveCardRecords.get(id)); syncChatStatus(); },
    });
    const cardRowAnchors = createPlacedCardAnchors({ repaint: record => renderLiveCardTimeline(record),
        updateCount: updateLiveCardCount, withStableViewport });
    // A card let go for good: its node and per-card facts beside it (a reusable slot keeps them: disposeLiveCard).
    function forgetLiveCard(id) {
        disposeLiveCard(id);
        for (const facts of [subagentChildParents, subagentTerminalChildren, explicitCardExpansion, reviewDisclosureByTask]) facts.delete(id);
    }
    function disposeLiveCard(id, preserveWaits = false) {
        if (!preserveWaits) modelWaits.forget(id);
        const record = liveCardRecords.get(id);
        record?.timelineDispose?.();
        if (record) releaseTimeline(record);
        record?.reviewController?.destroy();
        record?.regions?.forEach((dispose) => dispose());
        record?.episodeDisclosureDispose?.();
        if (record) disposeCardModel(record);
        if (record?.episodeDisclosure) releaseMessageNode(record.episodeDisclosure);
        if (record?.root) releaseMessageNode(record.root);
        cardRowAnchors.forget(record);
        liveCardRecords.delete(id);
    }
    // A subscription fact's mark follows the window's catalog (card_model.js): one listener for the room's life,
    // over the room's own live cards, each pass inside the stable-viewport seam. Disposed by destroy().
    const unfollowCatalogue = followCardCatalogue({ records: liveCardRecords, viewport: withStableViewport });
    const markReviewAnchor = (r, on = false) => setReviewAnchor(r, on, setLiveCardPhase);
    const explicitCardExpansion = new Map();
    const reviewDisclosureByTask = new Map();
    const reviewHydrator = createReviewHydrator({
        fetchDetail: cardDetailReader(liveCardRecords, fetchTaskDetailStrict),
        onSettled: () => !destroyed && reading.position(),
        applyDetail: (id, detail) => !destroyed && attachTaskDetailReviews(id, detail, true),
        onState: (id, status) => !destroyed
            && (liveCardRecords.get(id)?.reviewController?.setHydrateStatus?.(status) ?? false),
    });
    // Buffer task_named frames that precede their card.
    const pendingSuggestedNames = new Map();
    // Server-confirmed direct/managed activities.
    const activeDirectActivities = new Map();
    // Local user submissions awaiting server confirmation.
    const pendingSubmissions = new Map();
    const hostStart = createHostStart(); // В9 readiness and boot phases; a disconnect and a reconnect forget both
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
    // A stored row's ts is its creation, not its receipt.
    const settleChildFrom = (id, d) => withStableViewport(() => routeSubagentTerminalToCard(id, { ...d, task_id: id, ts: d.updated_at || d.ts }, true));
    // A terminal root reads each open descendant once; only proven child terminality changes its state (#300).
    function settleTerminalRootChildren(rootId) {
        for (const [childId, info] of subagentChildParents) {
            if (info.parentId !== rootId) continue;
            settleTerminalRootChildren(childId);
            const record = liveCardRecords.get(childId);
            if (!record || record.finished || record.managedDetail?.current() || managedTaskDetailReads.has(childId)) continue;
            managedTaskDetailReads.add(childId);
            readCardDetail(record).then((detail) => {
                if (!destroyed && !leaving && isTerminalTaskDetail(detail)) settleChildFrom(childId, detail);
            }).catch(() => {}).finally(() => managedTaskDetailReads.delete(childId));
        }
    }
    // Root census rows omit children; complete work membership can admit their own result read before the root ends.
    // Unknown/nonterminal replies retry on the next refresh; a confirmed 404 waits for new card evidence or reconnect.
    function settleAbsentChildren(snapshotRequestedAt) {
        for (const [childId, record] of liveCardRecords) {
            if (!record.isSubagent || record.finished || !record.root?.isConnected) continue;
            if (workState.members.has(childId)) {
                record.childDetailRead = undefined;
                if (record.historicalUnavailable
                        || (record.historicalUnconfirmed && memberActivityEvidenced(workState.memberStates?.get(childId)))) withStableViewport(() => {
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
                    || record.managedDetail?.current() || managedTaskDetailReads.has(childId)
                    || !subagentChildParents.has(childId)) continue;
            // Queue departure and membership share this read; keep the visible hold
            // until actual recovery or terminal evidence, not just absence.
            record.projectHoldQueued = false;
            const observation = record.childDetailRead = { updates };
            managedTaskDetailReads.add(childId);
            readCardDetail(record).then((detail) => {
                if (destroyed || ws.isConnected?.() === false || !workState.complete
                        || liveCardRecords.get(childId) !== record || record.finished
                        || workState.members.has(childId) || record.updates !== updates
                        || record.childDetailRead !== observation) return;
                if (detail === null) {
                    observation.missing = true;
                    withStableViewport(() => settleMissingTaskRecord(record));
                } else if (isTerminalTaskDetail(detail)) settleChildFrom(childId, detail);
            }).catch(() => {}).finally(() => managedTaskDetailReads.delete(childId));
        }
    }
    let activeLiveGroupId = '';
    // A return's or reconnect's read: status once out past its bound, ↓ Retry once it ends unconfirmed (shown only,
    // as a hidden feed reads as at its end).
    const catchUp = createCatchUp(() => { syncChatStatus(); if (isInstanceVisible()) updateScrollButton(); });
    let pendingRestartNotice = '';
    try {
        pendingRestartNotice = restartNoticeText(new URL(window.location.href).searchParams.get('_ouro_reason') || '');
    } catch {}
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
        statusBadge.title = title; // a failed boot step's cause, or the server-answers sentence; empty otherwise
        Object.assign(statusBadge.dataset, { tone, mark, owner: owner ? '1' : '', amber, step });
        roomWork?.update();
    }

    function syncHeaderControlState(data) {
        composer?.syncState(data);
        header?.update(data);
    }

    // Main: a moved work's card is its portal (work_portal.js); Continue episodes share a card (task_episodes.js).
    const portals = createWorkPortals({ isMain, feed: messagesDiv, records: liveCardRecords, onDomWrite: withStableViewport,
        isConnected: () => ws.isConnected?.() !== false, onConverted: () => composer?.signalFreed(), remember: rememberMessageKey,
        append: (msg, text, opts) => appendProjectRow(addMessage, msg, text, opts), actionsOf: ensureLiveActionsEl,
        card: { get: getLiveCardRecord, show: ensureLiveCardVisible, reanchor: reanchorTaskCard,
            restore: (r, hold) => { const changed = restoreCardActivity(r, hold); return markReviewAnchor(r) || changed; },
            meta: renderLiveCardMeta, controls: syncCancelRunButton, finish: finishLiveCardMutation, note: noteTaskDetails,
            missing: observeMissingManagedTask, summary: appendTaskSummaryToLiveCard,
            collapse: (record) => (explicitCardExpansion.delete(record.groupId), record.root.dataset.expanded === '1'
                && setLiveCardExpanded(record, false)) } });
    const episodes = createTaskEpisodes({ records: liveCardRecords, reanchor: reanchorTaskCard, renamed: (record) => (portals.relabel(record), syncCardHeadline(record)),
        activity: (record) => currentActivity(record), follow: portals.follow,
        expand: (record) => { setLiveCardExpanded(record, true); hydrateShownDetails(record); revealChildDetails(record); }, meta: renderLiveCardMeta });
    const rowMenu = createChatRowMenu({ feed: messagesDiv, replies, cardOf: (id) => liveCardRecords.get(id),
        readControls: cardOffer, afterAction: reread });

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
            if (destroyed || leaving || !Array.isArray(queue?.pending) || !Array.isArray(queue?.running)) return;
            const tasks = new Map([...queue.pending, ...queue.running].map(row => [row.id, row.task]));
            withStableViewport(() => {
                for (const record of children) {
                    if (record.finished || !record.root?.isConnected || record.lastLiveObservedAt > started) continue;
                    const task = tasks.get(record.groupId);
                    if (task && 'project_admission_hold' in task) {
                        missingManagedTaskIds.delete(record.groupId);
                        restoreCardActivity(record, task.project_admission_hold);
                        if (ws.isConnected?.() !== false) {
                            // A fresh connected read releases the park this client set when it lost the connection.
                            const offline = record.parkedOffline;
                            record.parkedOffline = false;
                            if (record.projectHold && record.censusPhase && (!workState.memberStates || offline)) syncParkedPhase(record, '');
                        }
                        record.projectHoldQueued = Boolean(record.projectHold);
                    } else if (!tasks.has(record.groupId) && record.projectHoldQueued) {
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
        if (destroyed || leaving) return;
        if (data) void refreshChildProjectHolds();
        const before = workState;
        workState = reconcileWorkState(workState, data, chatId, ws.isConnected?.() !== false, snapshotRequestedAt);
        if (Array.isArray(data?.works)) settleAbsentChildren(snapshotRequestedAt);
        withStableViewport(() => applyCardWorkViews(liveCardRecords, workState));
        chatDecision.syncWorks(workState);
        changedOffers(liveCardRecords, before, workState).filter(r => !isPortal(r)).forEach(syncCancelRunButton);
        syncHeaderControlState(data ?? { accounting: { available: false } });
        portals.sync(data, Array.isArray(data?.works) ? workState.episodes : null);
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
                noteHandoverAttempt(liveCardRecords.get(activity.activity_id), activity, withStableViewport);
            }
        }
        syncChatStatus();
    }

    // A page-wide state read taken after this room's own change, coalesced behind one in flight (app.js polls).
    async function refreshHeaderControlState() {
        if (destroyed || leaving) return;
        const request = await stateSnapshots.gate(true);
        if (request) await readStateSnapshot(stateSnapshots, request);
    }

    const ACTUAL_BOTTOM_TOLERANCE_PX = 6;

    const band = () => readBand(messagesDiv, pageHeader, inputArea);
    const { captureVisibleTimelineAnchor, restoreVisibleTimelineAnchor, serializeTimelineAnchor, anchorOwnersReady } =
        createTimelineAnchors({ messagesDiv, liveCardRecords, band });
    // Far rows skipped once laid out (style.css); a restyle may change every row's size at once, and the reading owner
    // places the passage again.
    let rowSleep = createRowSleep({ feed: messagesDiv, onRestyle: () => reading.reflow() });
    // Text fitting changes layout a frame later; a follower lands again.
    const offRefit = onRefit((els) => { if (reading.stick && els.some((el) => messagesDiv.contains(el))) reading.reflow(); });
    const reading = createChatReadingPosition({
        feed: messagesDiv, alive: () => !destroyed,
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
        // Late rich content refreshes its own fold inside the viewport write.
        const foldedWrite = writer => (mutate, options) => writer(() => {
            const changed = mutate();
            refreshProjectAnswerFold(root.closest?.('.project-answer'));
            return changed;
        }, options);
        return enhanceChatMarkdown(root, {
            onDomWrite: foldedWrite(_remoteActivityDepth > 0 ? withRemoteActivity : withStableViewport),
            onThemeDomWrite: foldedWrite(withStableViewport),
            keptSelection: () => self?.keptRanges?.length > 0,
        });
    }
    const {
        renderLiveCardTimeline,
        appendTimelineItem,
        patchLastTimelineItem,
        patchTimelineItemAt,
        oweTimeline, drawOwedTimelines, releaseTimeline,
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
            // Typing stays last in this room's own column.
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

    // DESIGN / 16.09: visibility and chrome follow work.
    function blockVisible(record) {
        if (!record || record.isSubagent) return true;
        // A moved work's portal never leaves Main (one work, one card).
        return isPortal(record) || blockHasWork(record)
            || !!(record.modelWaiting || record.cancelPendingPolicy || record.reviewAnchor)
            || stopEligible(record)
            || (record.finished && record.phaseEl?.dataset?.phase !== 'done');
    }

    function blockHasWork(record) {
        const id = record.groupId;
        // Visibility depends on work, never the lane.
        return record.reviewController?.groups.size > 0
            || [...subagentChildParents.values()].some((info) => info.parentId === id)
            || record.items.some((item) => !item.receipt && (item.fullField === 'schedule_report'
                || !String(item.dedupeKey || '').startsWith('task_done|')))
            || record.toolErrors > 0;
    }

    // Child or project names the card; an unnamed root uses narration, `Task activity` once ended.
    function cardTitle(record, work = blockHasWork(record)) {
        return record.isSubagent ? childTitle(record) : record.suggestedName
            || (work ? authoredTitle(record) || (record.finished ? 'Task activity' : '') : '');
    }

    // Change only (a rewrite loses a copy selection). A finished root's own sentence, not narration, is its line
    // or its sole head (owner 01.11, A); a held cause stays.
    function syncCardHeadline(record) {
        const title = cardTitle(record);
        const line = record.finished && !record.isSubagent && record.wakeSummary ? closingLine(record, title) : record.collapsedActivity;
        if (record.titleEl.textContent !== title) { record.titleEl.textContent = title; portals.relabel(record); }
        paintOrigin(record);  // the name may already say the place
        return renderCollapsedActivity(record, projectCollapsedActivity({ record, suggestedName: title, headline: line, body: line }));
    }

    // Fold host facts; absent counters stay absent.
    function noteToolMetrics(taskId, metrics, rawTs, { suppressDomInsert = false, attempt, recorded = false } = {}) {
        const facts = taskId && cardMetricFacts(metrics);
        if (!facts) return false;
        return withStableViewport(() => {
            const record = getLiveCardRecord(taskId);
            const before = captureLiveCardProjection(record);
            const projection = foldCardMetrics(record, metrics, facts, attempt, recorded);
            if (!projection) return false;
            const { owned, summary } = projection;
            // A dateless current-work carrier belongs last; other attempts can anchor a real call, not today's chip.
            const anchor = owned && !Number.isFinite(Date.parse(rawTs)) ? rawTs : replayedEvidenceTs(facts.evidence, owned ? rawTs : '');
            const changed = summary && placeReplayedEvidence(record, summary, facts.evidence, anchor, normalizeLogTs, summarizeChatLiveEvent)
                && renderLiveCardTimeline(record);
            updateLiveCardCount(record);
            renderLiveCardMeta(record);
            reanchorTaskCard(record, anchor, { suppressDomInsert });
            ensureLiveCardVisible(record, { suppressDomInsert });
            return Boolean(changed || liveCardProjectionChanged(before, record));
        });
    }

    // P5: task ids whose progress carried supervisor's host-attested `cancelable` marker (queue
    // tasks cancel endpoint reaches), from live frames, replay, maybe before card: kept with records.
    const cancelableTaskIds = new Set();

    function queueTaskLiveUpdate(summary, taskId, rawTs) {
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
            return Boolean(applyLiveCardState(summary, taskId, rawTs) || changed);
        });
    }

    // Main blocks with work offer conversion unless origin bound (same /api/state fact app.js reads);
    // titles use that predicate. A work card offers Reply to that exact work.
    function syncBlockChrome(record) {
        replies.syncCard(record, record.isSubagent || isPortal(record) || blockHasWork(record));
        // A child is never a convertible unit (it inherits its root's Project by
        // lineage) and its title is its role: both root writers skip it.
        const work = !record.isSubagent && blockHasWork(record);
        // The first row of work lands after the title writers ran for its frame: an empty title
        // takes a name here (Working is the chip's word, #1369); the writers own it from then on.
        if ((work || isPortal(record)) && !record.titleEl.textContent) record.titleEl.textContent = cardTitle(record, true);
        portals.syncConvert(record, isMain && work && !isPortal(record) && !(window.__ouroTaskBindings || {})[record.groupId]);
    }

    function syncCancelRunButton(record) {
        return withStableViewport(() => syncCardControls(record, { read: cardOffer, actionsOf: ensureLiveActionsEl,
            stop: cancelRunFromCard, afterAction: refreshHeaderControlState }));
    }

    const currentActivity = (r) => isPortal(r) ? portals.activity(r) : activeDirectActivities.get(r?.groupId);

    // The old Main marker never speaks for a portal. A room row can offer legacy Stop only without a work offer.
    const stopEligible = (record, activity = currentActivity(record), work = undefined) => cancelRunEligibility({
        groupId: record.groupId, isSubagent: record.isSubagent, finished: record.finished,
        cancelable: !record.historicalUnavailable && !record.historicalUnconfirmed && (isPortal(record)
            ? work === null && activity?.cancelable === true
            : activity?.cancelable !== false && (cancelableTaskIds.has(record.groupId) || activity?.cancelable === true
                || activity && activity.kind !== 'direct_chat' && blockHasWork(record))),
    });

    function cardOffer(record) {
        if (destroyed || leaving || liveCardRecords.get(record.groupId) !== record) return {};
        const portal = isPortal(record);
        const attempt = portal ? undefined : workState.attempts.get(record.groupId);
        noteControlQueue(record, attempt);
        const stopOnly = attempt < Math.max(record.controlQueueAttempt ?? 0, record.cardAttempt ?? 0);
        const work = portal ? portals.workOf(record) : stopOnly ? null : undefined;
        const activity = stopOnly ? null : currentActivity(record);
        const offered = cardControls(workState, record.groupId, work, portal);
        const memberClose = record.isSubagent && !record.finished && !stopOnly && !workState.memberReadFailed
            ? memberState(workState, record.groupId)?.close_expired : null;
        const close = offered.close || (memberClose?.target === record.groupId ? memberClose.close_expired : '');
        return { ...offered, close, activity, stopOnly,
            legacy: !offered.work && stopEligible(record, activity, work), episode: record.controlEpisode,
            scope: portal ? record.root.dataset.projectId : '' };
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
            const view = taskTerminalSummary({ ...fact, task_phase: 'finalizing' });
            record.observedOutcome = portals.summary(record, fact, view).observedOutcome || record.observedOutcome;
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
            if (!stored || !record || record.finished && !isPortal(record)) return false;
            if (taskCancelPending(stored)) {
                return markLiveCardCancelPending(taskId, taskSoftStopPending(stored));
            }
            return taskDoneIsTerminal(stored) ? appendTaskSummaryToLiveCard({ ...stored, task_id: taskId })
                : portals.observeDetail(record, stored);
        });
    }

    async function cancelRunFromCard(record, action = '') {
        const taskId = taskKey(record?.groupId);
        if (!taskId || record.finished) return;
        const soft = action === ACTION_FINALIZE;
        const btn = record.cancelRunBtn;
        const { portalProof, controlEpisode } = record;
        const priorPhase = captureLiveCardPhaseState(record);
        withStableViewport(() => {
            if (btn) btn.disabled = true;
            return markLiveCardCancelPending(taskId, soft);
        });
        let restore = false;
        try {
            await requestStop(taskId, action);
        } catch (exc) {
            if (exc?.status !== 404) showToast(`Cancel failed: ${exc?.message || exc}`, 'error');
            restore = exc?.status !== 404 && !record.finished;
            if (!restore && record.controlEpisode === controlEpisode) revokeManagedTaskCancelAuthority(taskId);
        }
        if (liveCardRecords.get(taskId) !== record || record.controlEpisode !== controlEpisode) return;
        const stored = await readCardDetail(record, { fresh: true }).catch(() => undefined);
        withStableViewport(() => {
            if (liveCardRecords.get(taskId) !== record || record.controlEpisode !== controlEpisode) return false;
            if (btn && !record.finished && soft && record.cancelPendingPolicy === 'finalize') btn.disabled = false;
            if (!stored || isPortal(record) && !portals.acceptsDetail(record, stored)) return false;
            const changed = reconcileCancelCardFromDetail(record, taskId, stored);
            if (!restore || record.finished || taskCancelPending(stored)
                    || record.cancelPendingPolicy !== (soft ? 'finalize' : 'immediate')) return changed;
            if (btn) btn.disabled = false;
            const phase = restoreLiveCardPhaseState(record, record.portalProof === portalProof ? priorPhase : captureLiveCardPhaseState(record));
            return setLiveCardPhase(record, phase.phase, phase.text, phase.className, phase.secondary) || changed;
        });
    }

    function restoreCardActivity(record, held = {}) {
        if (!setHistoricalUnavailable(record, false, held)) return;
        renderLiveCardMeta(record);
        syncCancelRunButton(record);
        return true;
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

    function markTaskCancelable(taskId = '', attempt) {
        const id = taskKey(taskId);
        const record = liveCardRecords.get(id);
        if (!id || attempt < record?.cardAttempt) return false;
        cancelableTaskIds.add(id);
        return syncCancelRunButton(record);
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

    function attachTaskDetailReviews(taskId, detail, stored = false) {
        return withStableViewport(() => {
            const id = taskKey(taskId), prior = liveCardRecords.get(id);
            if (!id || stored && !prior) return false;
            const portal = stored && isPortal(prior);
            const stale = portal && !portals.acceptsDetail(prior, detail);
            if (!stale) modelWaits.observe(id, detail);
            const groups = reviewGroupsFromTaskDetail(detail, id);
            const record = liveCardRecords.get(id) || (groups.length ? getLiveCardRecord(id) : null);
            const filed = !stale && noteTaskDetails(record, detail, stored);
            const reconciled = !stale && (portal || groups.length) && reconcileCancelCardFromDetail(record, id, detail);
            if (!groups.length) return Boolean(filed || reconciled);
            if (!prior) reanchorTaskCard(record, detail?.ts || detail?.timestamp || '');
            const changed = record.reviewController.updateMany(groups);
            ensureLiveCardVisible(record);
            return Boolean(changed || reconciled || filed);
        });
    }

    // Detail facts, independent of the card's lifecycle.
    function noteTaskDetails(record, detail, stored) {
        const files = syncSettledItems(record, detail, { write: withStableViewport, stored });
        if (!stored && !record?.isSubagent) refreshSettledCardDetail(record, detail, hydrateCardReviews);
        const words = syncClosingSentence(record, detail);
        if (words) syncCardHeadline(record);
        const origin = noteOrigin(record, detail);
        const history = syncHistoryRetentionItem(record, detail);
        const candidate = syncCandidateRemainderItem(record, detail);
        const chain = episodes.note(record, detail);
        const receipt = syncSwarmReceipt(record, detail);
        const money = syncCostBreakdown(record, detail);  // a root's "Spent on" / "Ran on" rows in its Details
        return Boolean((words || origin || files || history || candidate || money) && (renderLiveCardMeta(record), updateLiveCardCount(record), renderLiveCardTimeline(record), true))
            || chain || receipt;
    }

    function hydrateCardReviews(taskId, revision = null, fresh = false) {
        return destroyed ? Promise.resolve(false) : reviewHydrator.hydrate(taskId, revision, { fresh,
            onDomWrite: _remoteActivityDepth > 0 ? withRemoteActivity : withStableViewport,
        });
    }

    function revealChildDetails(record) {
        hydrateRevealedChildren(record, liveCardRecords, hydrateCardReviews);
    }

    function hydrateShownDetails(record, wasOpen = false) {
        hydrateRevealedCard(record, wasOpen, hydrateCardReviews, reviewHydrator.ready);
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
                    // A notice is a System row (its line; the original job behind Details); an older row, a progress note.
                    const said = Boolean(noticeLine(row));
                    if (ack) changed = Boolean(addMessage(
                        ack, said ? 'system' : 'assistant', !said && !!row?.markdown, rawTs || row?.ts || null,
                        !said, { systemType: 'lifecycle_pointer', ...(said ? { reply: row } : {}) },
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
                if (lifecycle.group.activeCount === 0 && lifecycle.group.lifecycleStatus) scheduleHistorySync('settled');
                return attached;
            });
        }
        return lifecycle.classification === 'source_incomplete' ? false : undefined;
    }

    // Host keys place content without changing phase/expansion.
    // Reviews rehydrate; failures retain rows, unkeyed rows stay bubbles.
    // Pass 2 uses pass-1 cards without a type allowlist.
    function attachCardRow(msg, rawTs = '', { suppressDomInsert = false } = {}) {
        const placed = placedCardRow(msg);
        const record = placed && liveCardRecords.get(placed.taskId);
        if (!record) return undefined;
        const { taskId, placement } = placed;
        return withStableViewport(() => {
            const before = captureLiveCardProjection(record);
            const fresh = cardRowAnchors.merge(record, msg, placed, rawTs);
            const changed = fresh ? renderLiveCardTimeline(record) : false;
            updateLiveCardCount(record);
            reanchorTaskCard(record, rawTs, { suppressDomInsert });
            ensureLiveCardVisible(record, { suppressDomInsert });
            // Only new review rows trigger a read; a live receipt on a settled card re-reads its stored line.
            if (fresh && placement === 'reviews') hydrateCardReviews(taskId);
            if (fresh && !msg.history_id && record.finished && msg.terminal_receipt) scheduleHistorySync('settled');
            return Boolean(changed || liveCardProjectionChanged(before, record));
        });
    }

    function admitCardMetadata(row) {
        if (row.system_type === 'task_evidence') {
            if (!_historyReplayActive) noteToolMetrics(row.task_id, row, row.ts);
            // The task's own evidence: mints its card at the carrier's row (zero tools included).
            const record = getLiveCardRecord(row.task_id);
            const activity = currentActivity(record);
            if (row.ts) reanchorTaskCard(record, row.ts);
            if (row.suggested_name) applySuggestedName(row.task_id, row.suggested_name);
            if (row.task_phase === 'finalizing') markLiveCardFinalizing(row.task_id, row);
            else if (!row.task_terminal_status && !activity && !isPortal(record)) setHistoricalUnconfirmed(record);
            if (isPortal(record)) portals.present(record);
            else if (activity) syncParkedPhase(record, activity.phase, activity);
            syncCancelRunButton(record);
            return true;
        }
        if (isModelWaitReference(row)) {
            const changed = modelWaits.observe(row.task_id, row);
            return row.outcome_axes && (!_historyReplayActive || row.task_phase === 'finalizing') ? appendTaskSummaryToLiveCard(row) || changed : changed;
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
                        <span class="chat-live-when" data-live-when hidden></span>
                    </div>
                    <span class="chat-live-title" data-live-title></span>
                    <div class="chat-live-doors">
                    <button type="button" class="chat-live-review-summary ui-hit-area" data-live-review-summary aria-expanded="false" hidden>${CHEVRON}</button>
                    <button type="button" class="chat-live-summary-side ui-hit-area" data-live-disclosure aria-expanded="false" aria-controls="${escapeHtmlAttr(timelineId)}">
                        <span class="chat-live-count" data-live-count hidden>0 notes</span>
                        <span class="chat-live-toggle" data-live-toggle>Details</span>
                        ${CHEVRON}
                    </button></div>
                </div>
                <div class="chat-live-episodes" data-live-episodes hidden></div>
                <div class="chat-live-activity" data-live-activity></div>
            </div>
            <div class="chat-live-foot" data-live-foot>
                <div class="chat-live-meta" data-live-meta></div>
                <div class="chat-live-foot-end" data-live-foot-end></div>
            </div>
            <div data-live-reviews-host></div>
            <div class="chat-live-details scroll-fade-y">
                <div data-live-phase-detail hidden></div>
                <dl class="chat-live-facts" data-live-facts hidden></dl>
                <div data-live-cost-detail hidden></div>
            </div>
            <div class="chat-live-timeline scroll-fade-y" data-live-timeline id="${escapeHtmlAttr(timelineId)}"></div>
            <div class="chat-live-episode-history" data-live-episode-history hidden></div>
        `;
        const part = (name) => root.querySelector(`[data-live-${name}]`);  // the card's own named parts
        const record = {
            groupId: normalizedGroupId,
            root,
            summaryButtonEl: part('summary-button'),
            phaseEl: part('phase'),
            titleEl: part('title'),
            episodesEl: part('episodes'),
            episodeHistoryEl: part('episode-history'),
            activityEl: part('activity'),
            countEl: part('count'),
            disclosureEl: part('disclosure'),
            footEl: part('foot'),
            footEndEl: part('foot-end'),
            metaEl: part('meta'),
            reviewSummaryEl: part('review-summary'),
            toggleEl: part('toggle'),
            // Both actions render lazily from the record's facts (syncBlockChrome,
            // syncCancelRunButton).
            turnProjectBtn: null,
            cancelRunBtn: null,
            timelineEl: part('timeline'),
            detailsEl: root.querySelector('.chat-live-details'),
            reviewsHostEl: part('reviews-host'),
            updates: 0,
            wakeSummary: '',
            finished: false,
            items: [],
            lastHumanHeadline: '',
            lastHumanHeadlineSource: null,
            expandedLineKeys: new Set(),
            _anchorOrderDirty: false,
            _timelineDirty: false,
            _lastFrameMeta: [],
            isSubagent: Boolean(options.isSubagent),
            parentGroupId: String(options.parentGroupId || ''),
            subagentRole: String(options.role || ''),
            subagentsEl: null,
            // The proactively-coined LLM name; becomes the card title when set.
            suggestedName: '',
            reviewOwnerDetailObserved: false,
            // Its meta line for a writer outside a frame: the member state that moves a child's chip (work_status.js).
            paintMeta: () => renderLiveCardMeta(record),
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
            discloseLiveCard(record, nowExpanded);
            if (nowExpanded) hydrateShownDetails(record);
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
        // While a part Details opens scrolls inside (style.css) it is a named region the keyboard reaches.
        record.regions = [[record.detailsEl, 'Scrollable task details'], [record.timelineEl, 'Scrollable task notes']]
            .map(([node, region]) => bindScrollFade(node, { region }));
        liveCardRecords.set(normalizedGroupId, record);
        liveCardBound.observe(liveCardRecords.size);
        // apply a name that arrived (task_named) before this card existed.
        const _pendingName = pendingSuggestedNames.get(normalizedGroupId);
        if (_pendingName && !record.isSubagent) {
            pendingSuggestedNames.delete(normalizedGroupId);
            record.suggestedName = _pendingName;
            record.titleEl.textContent = _pendingName;
        }
        clearStickyCardState(record);
        setLiveCardPhase(record, 'working');
        syncLiveCardToggle(record);  // a card born expanded says so from its first paint
        // A card the last complete body does not name asks for a read.
        if (!_historyReplayActive && workState.complete && !workState.members.has(normalizedGroupId)) reread();
        applyCardWorkViews([record], workState);
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
        // Children follow what the card's Details opens (its facts and notes), under the card's own lines, never behind
        // them; earlier episodes follow the children, an episode's disclosure keeping them after its Reviews.
        const own = parentRecord.timelineEl?.parentNode === parentRecord.root;
        const anchor = own ? parentRecord.timelineEl : parentRecord.reviewsHostEl || parentRecord.timelineEl;
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
        dockChildReviewDoor(record);
        if (promoted) {
            record.root.classList.add('subagent');
            record.root.dataset.subagent = '1';
            promoteChildCard(record, workState);
            if (record.finished) setLiveCardPhase(record, replayTerminalPhase(record), record.phaseEl.textContent, record.phaseEl.className);
            // This shell received frames before its lineage.
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

    function ensureLiveCardVisible(
        record,
        { suppressDomInsert = false, reorderExisting = false } = {},
    ) {
        if (!record || suppressDomInsert || _syncPass1Active) return;
        const container = record.isSubagent ? ensureSubagentContainer(record.parentGroupId) : null;
        if (container) {
            replies.syncCard(record, true);
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

    function discloseLiveCard(record, expanded = true) {
        explicitCardExpansion.set(record.groupId, expanded);
        setLiveCardExpanded(record, expanded);
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
            if (expanded && !record.episodeDisclosure) revealChildDetails(record);
            return Boolean(record.root.isConnected);
        };
        return record?.root?.isConnected ? withStableViewport(mutate) : mutate();
    }

    // A window shown again keeps the reading intent (data readiness positions it); hidden, it ends an unread opening
    // still owed; a room away stays put (its return: `resume`).
    const handleVisibility = () => {
        if (leaving) return;
        if (document.hidden) reading.land(() => 0, reading.arriving);
        else { reading.request(); historyControls.showNotice(); }
    };
    document.addEventListener('visibilitychange', handleVisibility);

    async function fetchFullLineOutput(item, record) {
        const basis = item.childCause;
        item._fetchingFull = true;
        item.fullReadError = '';
        let changed = false;
        try {
            const resp = await apiFetch(`/api/tasks/${encodeURIComponent(item.fullRef)}`, { cache: 'no-store' });
            const data = resp && typeof resp.json === 'function' ? await resp.json() : resp;
            let full = '';
            if (item.fullField === 'candidate_remainder') {
                full = readCandidateRemainder(item, data, resp?.ok !== false);
                changed = true;
            } else if (item.fullField === 'schedule_report') {
                full = resp?.ok !== false && typeof data?.result === 'string' ? data.result : '';
                if (!full) throw new Error('saved report unavailable');
            } else if (data?.result || data?.trace_summary) {
                const view = summarizeSubagentCardFrame({ ...childCauseFacts(data), subagent_event: data.status,
                    result: data.result, trace_summary: data.trace_summary, error: data.error }, item.fullRef);
                full = view?.fullOutput || '';
                if (full) {
                    item.fetchedCause = { ...basis, ...view.childCause };
                    item.fetchedCauseBasis = basis;
                }
            }
            if (full && item.fetchedFull !== full) {
                item.fetchedFull = full;
                changed = true;
            }
        } catch {
            if (item.fullField === 'candidate_remainder') {
                readCandidateRemainder(item, null, false);
                changed = true;
            } else if (item.fullField === 'schedule_report') {
                item.fullReadError = tr('schedule.report_unavailable', 'The saved report is unavailable.');
                changed = true;
            }
        } finally {
            item._fetchingFull = false;
            if (changed && !destroyed && liveCardRecords.get(record.groupId) === record && record.expandedLineKeys.has(item.lineKey)) {
                (leaving ? oweTimeline : renderLiveCardTimeline)(record);
            }
            reading.position();
        }
    }

    function scheduleHistorySync(reason) {
        if (destroyed || leaving) return;
        historyResyncScheduler.schedule(liveCardBound.isArmed(), { reason });
    }

    const historyResyncScheduler = createHistoryResyncScheduler({
        isReplayActive: () => _historyReplayActive,
        // Re-arm if the joined sync fetched its window before the arm.
        run: tags => historyPager.read({ ...tags }).catch(() => {}).then(() => {
            if (!destroyed && !leaving && lastHistorySyncSucceeded && liveCardBound.isArmed()) scheduleHistorySync('bound_recheck');
        }),
    });

    function applyLiveCardState(summary, groupId, rawTs, options) {
        return withStableViewport(() => applyLiveCardStateMutation(summary, groupId, rawTs, options));
    }

    // Authored naming has own clock: newer execution or terminal outcome may own card before saved
    // narration arrives. Root's collapsed line shows narration unless cause holds.
    function reconcileAuthoredHeadline(record, summary, rawTs) {
        if (!summary.human || !summary.headline) return;
        const previous = record.lastHumanHeadlineSource, shown = previous?.preview || '';
        const incoming = { time: Date.parse(rawTs), position: _historyRow?.history_position,
            preview: authoredLine(record, summary) || shown };
        if (Number.isFinite(previous?.time) && (!Number.isFinite(incoming.time) || incoming.time < previous.time)) return;
        if (incoming.time === previous?.time && (compareHistoryPosition(incoming.position, previous.position) < 0
                || previous.position && !incoming.position)) return;
        if (!record.isSubagent && !record.causeHeld && record.collapsedActivity === shown) record.collapsedActivity = incoming.preview;
        record.lastHumanHeadline = summary.headline;
        record.lastHumanHeadlineSource = incoming;
        return true;
    }

    function applyLiveCardStateMutation(summary, groupId, rawTs, { suppressDomInsert = false } = {}) {
        const ts = normalizeLogTs(rawTs);
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
        const call = summary.toolCall;
        const settlement = call && record.toolFold.calls.get(call.key)?.settlement;
        if (settlement && (call.fact === 'wait_ended' || call.fact === 'settled'
                && (call.status !== settlement.status || Boolean(call.hostError) !== settlement.hostError))) {
            summary = { ...summary, visible: false }; // the accepted settlement owns the diagnostic
        }
        if (closedPhase && !initialContent) {
            if (foldView) {
                upsertToolFoldRow(record, foldView, ts, rawTs);
                // A plain error arriving after the ending stays in the fold, just as on replay.
                const plainError = call.status === 'error' && summary.phase === 'error' && !call.hostError
                    && !summary.hostRefusedStep && !['run_command', 'run_script'].includes(call.tool);
                if (!plainError && call.fact === 'settled' && summary.visible && summary.headline) placeToolDiagnostic(record, summary, ts, rawTs);
                renderLiveCardTimeline(record);
                updateLiveCardCount(record);
                if (!record.isSubagent) reanchorTaskCard(record, rawTs, { suppressDomInsert });
            }
            if (summary.modelExecution) record.modelExecution = summary.modelExecution;
            renderLiveCardMeta(record, summary.costProjection);
            return liveCardProjectionChanged(before, record);
        }
        if (!closedPhase && (summary.terminal
                || frameObservesPhase(record, summary, !_historyReplayActive && !_syncPass1Active))) {
            restoreCardActivity(record);
            if (!summary.terminal && (!record.isSubagent || ws.isConnected?.() !== false)) {
                if (!record.isSubagent || observeChildPhase(record, summary.phase)) record.livePhaseAt = Date.now();
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
        const syntheticKey = summary.dedupeKey || `${summary.phase || 'working'}|${headline}|${summary.body || ''}`;
        const isLegacyParentSubagentKey = syntheticKey.startsWith('parent-subagent:');
        // Failure/timeout update one row; success feeds the fold.
        const inPlaceByKey = isLegacyParentSubagentKey
            || ['subagent-lifecycle:', 'subagent-progress:', 'subagent-result:', 'task_done|', 'tool:']
                .some((prefix) => syntheticKey.startsWith(prefix));
        if (!isLegacyParentSubagentKey) {
            record.finished = Boolean(closedPhase) || isTerminalTaskPhase(nextPhase, summary.terminal);
        }
        if (summary.model) record.agentModel = summary.model;
        // Keep the observed initiator.
        if (summary.initiator) record.initiator = summary.initiator;
        noteOrigin(record, null, summary.placeOrigin);
        if (summary.continueRequester) record.continueRequester = summary.continueRequester;

        // Only task facts own the outcome chip under a hold; failed tools are diagnostics.
        if (summary.observedOutcome && !record.finished) record.observedOutcome = summary.observedOutcome;
        const desiredPhase = desiredLiveCardPhase(record, closedPhase || (record.finished ? summary.phase || 'done' : ''));
        setLiveCardPhase(record, desiredPhase.phase, desiredPhase.text, desiredPhase.className,
            desiredPhase.secondary);
        // Collapsed: voice or outcome cause, never host notes; a cleared late cause restores voice. Expand keeps all activity.
        const activityCandidate = (record.isSubagent && summary.human !== false
            ? String(summary.activityPreview ?? summary.body ?? '')
            : String(!summary.human && summary.terminal && summary.activityPreview || '')).trim();
        if (activityCandidate) {
            record.collapsedActivity = boundActivityPreview(activityCandidate);
            record.collapsedActivityLead = summary.activityLead || '';  // a child's "Report to parent"
            record.causeHeld = !record.isSubagent;  // a root's line is its outcome's cause
        } else if (summary.currentOutcome && record.causeHeld) {
            record.collapsedActivity = record.lastHumanHeadlineSource?.preview || '', record.causeHeld = false;
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
        if (ts && (summary.human || summary.reasoning || activityCandidate)) record.latestActivityTs = ts, record.latestActivityRaw = rawTs || '';
        if (summary.requestedExecutor) record.requestedExecutor = summary.requestedExecutor;  // the child's selection
        const chip = summary.executorChip, adopt = chip && !keepStickyExecutorChip(record.executorChip, chip);
        record.executorChip = cardExecutorChip(record, adopt ? chip : null);
        // Coordinator facts outlive the chip: recorded ones over a pre-evidence chip's.
        const drafted = chip?.coordinator;
        if (summary.coordinator || drafted) record.coordinator = { ...drafted, ...record.coordinator, ...summary.coordinator };
        if (summary.modelExecution) record.modelExecution = summary.modelExecution;
        if (summary.effort) record.effort = summary.effort;
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
        endLetters(messagesDiv, record.groupId);  // its open letters are read no more (peer_letter_card.js)
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
        if (!wasFinished && blockVisible(record)) scheduleHistorySync('settled');
        if (!wasFinished && record.isSubagent) hydrateOpenSettledChild(record, hydrateCardReviews);
        syncLiveCardToggle(record);
    }

    function finishLiveCardMutation(groupId = '', phase = '') {
        const record = groupId ? liveCardRecords.get(groupId) : null;
        if (!record) return false;
        const before = captureLiveCardProjection(record);
        const typingBefore = typingEl.style.display;
        setHistoricalUnavailable(record, false);
        markReviewAnchor(record);
        const wasFinished = record.finished;
        record.finished = true;
        record.finalizingHold = false;
        const presentation = taskPresentation(phase || 'done', record.quietEffective ?? record.historicalTerminal?.quiet_effective);
        setLiveCardPhase(record, presentation.phase, presentation.headline);
        syncCardHeadline(record); renderLiveCardMeta(record);
        settleLiveCard(record, wasFinished);
        ensureLiveCardVisible(record);
        if (activeLiveGroupId === record.groupId) activeLiveGroupId = '';
        syncChatStatus();
        return Boolean(typingBefore !== typingEl.style.display
            || liveCardProjectionChanged(before, record));
    }

    function appendTaskSummaryToLiveCard(msg, { suppressDomInsert = false, currentOutcome = false, drawn = true } = {}) {
        const taskId = msg?.task_id || '';
        const rawTs = msg?.ts || msg?.revision || '';
        if (!taskId) {
            // Unkeyed summaries keep durable/log evidence but close no card.
            return false;
        }
        const record = getLiveCardRecord(taskId);
        let changed = Boolean(record.reviewController?.updateMany(reviewGroupsFromTaskDetail(msg, taskId)));
        if (!acceptLiveFrame(msg, _historyReplayActive, () => record, (card, words) => { changed = noteTaskDetails(card, words) || changed; })) {
            return renderLiveCardMeta(record, taskCostProjection(msg, rawTs)) || changed;
        }
        // Restore historical task name.
        if (msg?.suggested_name) changed = applySuggestedName(taskId, msg.suggested_name) || changed;
        const finalizing = msg?.task_phase === 'finalizing' || msg?.outcome_final === false;
        // `currentOutcome` marks the host's composed selected ending: the count it carries is a record.
        changed = noteToolMetrics(taskId, msg, rawTs, { suppressDomInsert, recorded: currentOutcome }) || changed;
        msg = { quiet_effective: record.quietEffective ?? record.historicalTerminal?.quiet_effective, ...msg };
        const summary = taskTerminalSummary({ ...msg, task_id: taskId });
        const view = portals.summary(record, msg, summary);
        changed = noteTaskDetails(record, msg) || changed;
        if (finalizing && !record.finished) record.finalizingHold = true;
        changed = applyLiveCardState(
            { ...view, visible: drawn && summary.visible, currentOutcome, costProjection: taskCostProjection(msg, rawTs) },
            taskId, rawTs, { suppressDomInsert },
        ) || changed;
        if (finalizing) return changed;
        changed = finishLiveCard(taskId, view.phase) || changed;
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

    function registerChild(msg) {  // a refusal names a child id it never admitted (admissionRefusalRow)
        if (String(msg?.delegation_role || '').toLowerCase() !== 'subagent' || msg.system_type === 'subagent_admission_notice') return null;
        const parentId = taskKey(msg.parent_task_id);
        const childId = taskKey(msg.subagent_task_id || msg.task_id);
        if (!parentId || !childId || parentId === childId) return null;
        const role = taskKey(msg.subagent_role);
        setSubagentParent(childId, { parentId, role, model: msg.model });
        return { childId, parentId, role };
    }

    function learnSubagentLineage(msg) {
        const child = registerChild(msg);
        if (!child) return '';
        const event = String(msg.subagent_event || '').toLowerCase();
        const replayTerminal = msg.task_terminal_status && taskDoneIsTerminal(msg);
        if (replayTerminal || TERMINAL_SUBAGENT_EVENTS.has(event)) subagentTerminalChildren.add(child.childId);
        return child.childId;
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
        }, liveCardRecords.get(childId)?.requestedExecutor);  // its kept selection, for a frame that repeats none
        // The child's effort fact rides every frame that states a level (live, terminal, replay).
        return summary ? withTaskCostMeta({ ...summary, ...(evt.effort_level ? { effort: effortFact(evt) } : {}) }, evt, { rawTs }) : null;
    }

    // A child's title is its lineage identity plus, for twins (one displayed identity under
    // one parent), the short id; re-projected on every title write and lineage change
    // (terminal children too).
    function childTitle(record) {
        const twin = subagentTwin(subagentChildParents, record.groupId);
        return subagentIdentityTitle(subagentChildParents.get(record.groupId))
            + (twin ? ` (${record.groupId.slice(0, 8)})` : '');
    }

    function updateLiveCardFromProgressMessage(msg, { grantCancelAuthority = true } = {}) {
        if (msg?.system_type === 'task_checkpoint') return updateLiveCardFromLogEvent({ ...msg, type: 'task_checkpoint', is_progress: false });
        if (msg?.system_type === 'subagent_admission_notice') msg = admissionRefusalRow(msg);  // its parent's row, never the child's
        const taskId = msg?.task_id || '';
        const rawTs = msg?.ts || new Date().toISOString();
        if (!taskId) return false;
        modelWaits.observe(taskId, msg);
        let changed = !_historyReplayActive && noteToolMetrics(taskId, msg, rawTs);
        // Only host-attested progress grants Stop authority.
        if (grantCancelAuthority && msg.cancelable === true) {
            changed = markTaskCancelable(String(taskId), attemptOf(msg)) || changed;
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
        if (!summary.activity) Object.assign(summary, lineReplyFields(msg));
        changed = queueTaskLiveUpdate(withTaskCostMeta(summary, msg, { rawTs }), taskId, rawTs) || changed;
        // History carries task_named's title.
        if (msg.suggested_name) changed = applySuggestedName(taskId, msg.suggested_name) || changed;
        // Outcome survives a lost summary even while post-work keeps controls live.
        if (
            (msg.task_phase === 'finalizing' || (!_historyReplayActive && taskDoneIsTerminal(msg)))
            && (msg.outcome_axes || msg.review_projection || msg.reason_code)
        ) {
            changed = appendTaskSummaryToLiveCard(msg) || changed;
        }
        return changed;
    }

    function updateSubagentCardFromEvent(evt, tsValue) {
        const child = registerChild(evt);
        if (!child) return undefined;  // not a subagent frame: callers fall through
        const { childId, parentId, role } = child;
        const event = String(evt.subagent_event || '').toLowerCase();
        // Worker narration also carries subagent_event="progress": activity, not a lifecycle row. Its key keeps a later
        // terminal frame from overwriting the only full narration disclosure.
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
        const record = getSubagentCardRecord(childId, parentId, role, rawTs);
        stampNodeTimestamp(record?.root, rawTs, { anchor: true });
        observeChildStart(record, evt);
        const reviewsChanged = attachTaskDetailReviews(childId, evt);
        const updated = queueTaskLiveUpdate(summary, childId, rawTs);
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
            // Replay can show progress after terminal; omit `status=running` when history knows the child's final state.
            status: preserveTerminal ? '' : (msg?.status || ''),
        }, rawTs);
        if (!summary) return false;
        // Live notes replace in place; every saved child line keeps its address.
        if (!summary.activity) Object.assign(summary, lineReplyFields(msg));
        const occurrence = `${childId}:${rawTs}:${content}`;
        if (summary.reasoning) summary.dedupeKey = `reasoning:${occurrence}`;
        else if (!summary.activity) {
            summary.occurrenceKey = `subagent-line:${occurrence}`;
            summary.dedupeKey = ownLine || _historyRow?.history_id ? summary.occurrenceKey : childProgressKey(record, summary.occurrenceKey);
        }
        if (preserveTerminal && !_historyReplayActive) {
            summary.phase = String(record.phaseEl?.dataset?.phase || 'done');
            summary.terminal = true;
        }
        return queueTaskLiveUpdate(summary, childId, rawTs);
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
        if (text) Object.assign(summary, lineReplyFields(msg));
        summary.dedupeKey = `subagent-result:${childId}`;
        if (priorTerminalPhase) {
            summary.phase = priorTerminalPhase;
            summary.terminal = true;
        }
        return queueTaskLiveUpdate(summary, childId, rawTs);
    }

    // A child's card from its task_done or a late review's current outcome (log frames without subagent metadata).
    function routeSubagentTerminalToCard(childId, evt, stored = false, phase = taskTerminalPhase(evt)) {
        const info = subagentChildParents.get(childId);
        if (!info) return false;
        const record = liveCardRecords.get(childId);
        if (stored) noteHandoverAttempt(record, evt, withStableViewport, true);
        const alreadyFinished = record?.finished;
        const updated = Boolean(updateSubagentCardFromEvent(terminalChildFrame(childId, info, evt, phase), evt.ts || evt.timestamp));
        if (!stored && alreadyFinished && taskDoneIsTerminal(evt)) hydrateOpenSettledChild(record, hydrateCardReviews);
        return updated;
    }

    function updateLiveCardFromLogEvent(evt) {
        if (!evt) return false;
        const eventType = evt.type || evt.event || '';
        if (eventType === 'history_retention' || evt.review_overrides) {
            return withStableViewport(() => noteTaskDetails(liveCardRecords.get(taskKey(evt.work_task_id || evt.task_id)), evt));
        }
        const reference = admitCardMetadata(evt);
        if (reference !== undefined) return reference;
        // owner_hurry.py marks an existing card, never progress.
        if (eventType === 'owner_hurry') return markOwnerHurry(ownerHurryProjection(evt).applied ? liveCardRecords.get(getLogTaskGroupId(evt)) : null);
        if (!isGroupedTaskEvent(evt)) return false;
        const taskId = getLogTaskGroupId(evt) || '', late = evt.current_outcome;
        if (!taskId) return false;
        const rawTs = late?.revision || evt.ts || evt.timestamp || new Date().toISOString();
        // Skill lifecycle belongs to its explicit owner.
        const review = attachReviewFromRow(evt, rawTs);
        if (review !== undefined) return review;
        let changed = false;
        const prior = liveCardRecords.get(taskId), wasFinished = prior?.finished, attempt = prior?.cardAttempt;
        if ((late && !lateOutcomeIsCurrent(liveCardRecords.get(taskId), evt)) || ((late || ['task_started', 'task_done'].includes(eventType))
                && !acceptLiveFrame(evt, _historyReplayActive, () => getLiveCardRecord(taskId), (card, words) => { changed = noteTaskDetails(card, words); }))) {
            const journal = evt.tool_evidence && noteToolMetrics(taskId, { tool_evidence: evt.tool_evidence }, '');
            return renderLiveCardMeta(liveCardRecords.get(taskId), taskCostProjection(evt, rawTs)) || Boolean(journal) || changed;
        }
        if (wasFinished && !liveCardRecords.get(taskId)?.finished) concludedDirectActivities.delete(taskId);
        if (eventType === 'task_started' && attemptOf(evt) > (attempt ?? 0)) cancelableTaskIds.delete(taskId);
        const metrics = late || evt;
        changed = ((metrics.tool_evidence || Number.isInteger(metrics.rounds)
            || ['task_metrics_event', 'task_eval', 'task_done'].includes(eventType)) && noteToolMetrics(taskId, metrics, rawTs)) || changed;
        const childInfo = subagentChildParents.get(taskId);
        if (childInfo && (late || eventType === 'task_done')) return routeSubagentTerminalToCard(taskId, { ...evt, ...late, ts: rawTs }) || changed;
        const summary = summarizeChatLiveEvent(evt);
        // Finished children still receive settled costs.
        if (childInfo && subagentTerminalChildren.has(taskId) && !summary?.toolCall && eventType !== 'task_cost_finalized') return false;
        if (!childInfo) changed = attachTaskDetailReviews(taskId, late ? { ...late, task_id: taskId } : evt) || changed;
        if (!summary) return changed;
        if (childInfo) {
            getSubagentCardRecord(taskId, childInfo.parentId, childInfo.role);
            changed = attachTaskDetailReviews(taskId, evt) || changed;
        }
        const replace = eventType === 'task_done' || eventType === 'task_cost_finalized';
        const queued = queueTaskLiveUpdate(withTaskCostMeta(summary, evt, { replace, rawTs }), taskId, rawTs);
        if (childInfo) return Boolean(changed || queued);
        const subagentChanged = updateSubagentCardFromEvent(evt, rawTs);
        // Host-attested Stop facts also travel on tool-only turns.
        if (evt.cancelable === true) markTaskCancelable(taskId, attemptOf(evt));
        if (eventType === 'task_done' && summary.terminal) {
            recordTerminalActivity(taskId, summary.phase);
            syncChatStatus();
        }
        return Boolean(changed || queued || subagentChanged);
    }

    const earlyRows = createEarlyTextBubbles(messagesDiv, drawTextBubble, insertMessageNode, releaseMessageNode,
        withStableViewport, address => rememberMessageKey(`window|${address}`), () => self?.keptRanges,
        address => windowReturn.release(address));
    function addMessage(text, role, markdown = false, timestamp = null, isProgress = false, opts = {}) {
        const { meta, ts, legacyKey, messageKey, deliveryKey } = messageIdentity(text, role, timestamp, isProgress, opts, chatId);
        const bubbles = () => [...messagesDiv.querySelectorAll('.chat-bubble')];
        const delivered = deliveryKey && seenMessageKeys.has(deliveryKey) && bubbles().find(node => node.dataset.deliveryId === opts.deliveryId);
        if (delivered) return recordBubble(delivered);
        const prior = opts.historyId
            ? (replayKeys && !replayKeys.history.has(opts.historyId) ? null
                : historyNodes(opts.historyId).find(node => node.classList.contains('chat-bubble')))
            : livePriorBubble(bubbles, opts, replayKeys, legacyKey, chatId);
        if (prior && (opts.historyId || _historyReplayActive)) {
            rememberMessageKey(messageKey);
            const originChanged = syncSavedProjectContext(prior, opts.originProjected, opts.originId);
            const quoteChanged = replies.syncDerivedQuestion(prior, opts.reply, { authoritative: true });
            noteBubbleKeys(prior);
            return chatDecision.renderRoutingDecision(prior, opts.chatAnnotation) || originChanged || quoteChanged;
        }
        if (prior || (messageKey && seenMessageKeys.has(messageKey))) return false;
        const address = role === 'assistant' && !opts.originProjected && opts.reply?.row_address;
        const existing = opts.historyId && earlyRows.get(address);
        const matched = opts.historyId && !existing && priorHistoryBubble(bubbles, opts, replayKeys, legacyKey, chatId);
        if (matched) {
            const placed = recordBubble(matched);
            chatDecision.renderRoutingDecision(matched, opts.chatAnnotation);
            const adopted = adoptsRetainedCopy(matched, opts) && replies.adopt(opts.reply, matched);
            const quoteChanged = replies.syncDerivedQuestion(matched, opts.reply, { authoritative: true });
            const restored = syncSavedProjectContext(matched, opts.originProjected, opts.originId);
            return placed || restored || adopted || quoteChanged;
        }
        if (existing) earlyRows.delete(address);
        const { bubble, rich } = drawTextBubble(text, role, { markdown, ts, isProgress, meta, legacyKey, opts, existing });
        insertMessageNode(bubble, { forceStick: !!opts.forceStick, reorderExisting: Boolean(existing) });
        if (rich) enhanceMountedMarkdown(bubble);
        chatDecision.renderRoutingDecision(bubble, opts.chatAnnotation);
        if (isMain) reconcilePortalNotes(messagesDiv, bubble);
        rememberMessageKey(messageKey);
        if (deliveryKey) rememberMessageKey(deliveryKey);
        if (opts.pending && meta.clientMessageId) pendingUserBubbles.set(meta.clientMessageId, bubble);
        return bubble;

        function recordBubble(node) {
            const positioned = stampHistoryNode(node, opts.historyId, opts.historyPosition);
            noteBubbleKeys(node);
            rememberMessageKey(messageKey);
            return positioned && insertMessageNode(node, { reorderExisting: true });
        }
    }

    const localOwnerRow = row => !row.origin_projected && ownerRowRoom(row, chatId) === chatId;

    function confirmOwnerRow(row) {
        const bubble = ownerMessageBubble(messagesDiv, row, chatId);
        const changed = markIngressSaved(messagesDiv, row, bubble) | markSwarmLabel(messagesDiv, row, bubble) | replies.adopt(row);
        if (localOwnerRow(row)) unconfirmed.settle(row);
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
            forgetLiveCard(id);
            pendingLiveEvictions.delete(id);
        }
    }
    // `requestedAt` dates the question records in these rows.
    function applyHistoryMessages(messages, { fromReconnect = false, archived = false, direction = '', anchor = null,
        requestedAt, window, hasMore, newest = direction === 'recent' || direction === 'latest' } = {}) {
        const sourceRows = messages;  // even rows withheld from replay carry their exact identity
        // ↓ hears only new words, not an older page or an adopted identity.
        const reportActivity = historyLoaded && !archived;
        const knownRows = reportActivity ? new Set(messagesDiv.children) : null;
        const knownCards = new Map([...liveCardRecords].map(([id, record]) => [id, reportActivity
            ? { projection: captureLiveCardProjection(record), items: record.items.length, source: record.latestSourceTs } : null]));
        _historyReplayActive = true;
        _historyAppending = archived;
        liveCardBound.beginReplay();
        try {
            withStableViewport(() => {
                preview.retire();
                cardRowAnchors.beginPage(direction, anchor);
                replayKeys = createReplayKeys();
                noteBubbleKeys(messagesDiv);
                replaySelection = keepSelection(document);
                // A formerly unbound final can gain its real child lineage on a
                // deeper page. A selected old bubble stays until it can move: every old one is judged before one leaves.
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
                    .filter((m) => m.role === 'user' && m.client_message_id && localOwnerRow(m))
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

                const endings = replayedTerminalRecords(messages, liveCardRecords, historicalTerminals);
                // Rows pass 1 attached to a card; pass 2 mounts the card with them.
                const cardRowsAttached = new Set();
                // First pass builds card state without DOM insertion.
                _syncPass1Active = true;
                try { for (const msg of messages) {
                    _historyRow = msg;
                    if (['quiz_answer', 'quiz_closed'].includes(msg.system_type)) chatDecision.applyQuizStateFrame(messagesDiv, { ...msg.quiz, task_id: msg.task_id }, { requestedAt });
                    if (msg.system_type !== 'task_evidence' && (isReplayEvidenceRow(msg) || msg.system_type === 'project_question_pointer')) continue;
                    if (msg.role !== 'user' && (msg.is_progress
                            || ['task_summary', 'task_evidence'].includes(msg.system_type)
                            || ['task_metrics_event', 'task_eval', 'task_done'].includes(msg.type || msg.event))) {
                        noteToolMetrics(msg.task_id, msg, msg.ts, { attempt: endings.get(msg.task_id)?.stamp.attempt });
                    }
                    if (admitCardMetadata(msg) !== undefined) continue;
                    if (attachReviewFromRow(msg, msg.ts || '') !== undefined) continue;
                    if (attachCardRow(msg, msg.ts || '', { suppressDomInsert: true }) !== undefined) {
                        cardRowsAttached.add(msg);
                        continue;
                    }
                    // Placement precedes progress; later rows may create the card.
                    if (placedCardRow(msg)) continue;
                    const taskId = msg.task_id || '';
                    if (!taskId || msg.peer_letter_card) continue;
                    if (msg.is_progress) {
                        if (msg.suggested_name) applySuggestedName(taskId, msg.suggested_name);
                        updateLiveCardFromProgressMessage(msg, { grantCancelAuthority: msg.project_mirror !== true });
                        const historyCard = liveCardRecords.get(taskId);
                        if (msg.review_overrides) noteTaskDetails(historyCard, msg);
                        if (historyCard && !historyCard.isSubagent && !isPortal(historyCard) && !activeDirectActivities.has(taskId)
                                && !msg.task_terminal_status && msg.task_phase !== 'finalizing'
                                && setHistoricalUnconfirmed(historyCard)) {
                            syncCancelRunButton(historyCard);
                            renderLiveCardMeta(historyCard);
                        }
                        if (historyCard && historicalTerminals.has(taskId)) historyCard.historicalTerminal = historicalTerminals.get(taskId);
                        continue;
                    }
                    if (msg.system_type === 'task_summary') {
                        if (endings.has(taskId) || msg.task_phase === 'unfinished') getLiveCardRecord(taskId);
                        else appendTaskSummaryToLiveCard(msg, { suppressDomInsert: true });
                    }
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
                    if (msg.system_type === 'project_question_pointer') { chatDecision.appendQuestionPointer(msg, { requestedAt }); continue; }
                    if (isReplayEvidenceRow(msg)) continue;
                    // Owner-bound reviews attached in pass 1 are not terminal bubbles.
                    if (
                        admitCardMetadata(msg) !== undefined
                        || attachReviewFromRow(msg, msg.ts || '', true) !== undefined
                        // A record minted after its row in pass 1 takes the row here.
                        || cardRowsAttached.has(msg) || attachCardRow(msg, msg.ts || '') !== undefined
                    ) continue;
                    if (placedCardRow(msg)) {
                        addStoredRow(msg, taskId, { evidenceRef: cardRowEvidenceRef(msg) });
                        continue;
                    }
                    // History clears Sending; census re-links live turns.
                    if (fromReconnect && msg.role === 'user' && localOwnerRow(msg) && msg.client_message_id) {
                        pendingSubmissions.delete(String(msg.client_message_id));
                    }
                    if (letterRow(msg, messagesDiv, addMessage)) continue;
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
                    if (appendDelivery(msg, requestedAt)) continue;
                    if (subagentChildParents.has(taskId) && childRowPlacement(msg) === 'line') {
                        insertCardIfNeeded(taskId); routeSubagentProgressToCard(taskId, msg, true); continue;
                    }
                    if (taskId && replayConcludesTask(msg)) {
                        const child = subagentChildParents.has(taskId);
                        insertCardIfNeeded(taskId);
                        if (child) routeSubagentFinalMessageToCard(taskId, msg);
                        // A replayed early final must not finalize a root's card.
                        if (!child && msg.task_phase === 'finalizing') markLiveCardFinalizing(taskId, msg);
                        else if (!endings.has(taskId) && (child || !msg.historical_terminal || positiveTaskTerminalFact(msg))) {
                            finishLiveCard(taskId, msg.task_terminal_status ? taskTerminalPhase(msg) : replayTerminalPhase(liveCardRecords.get(taskId)));
                        }
                        if (child) continue;
                    }
                    // A durable routing receipt retires Sending like its live frame.
                    if (msg.chat_annotation && localOwnerRow(msg) && msg.client_message_id &&
                            (!msg.chat_annotation.source_row_address || msg.sender_session_id === chatSessionId)) {
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
                    if (msg.role === 'user') confirmOwnerRow(msg);
                }
                const settled = newest && !catchUp.owed;
                earlyRows.clear(sourceRows, settled && (window?.complete === true && hasMore === false ? 'complete' : 'bounded'));
                if (settled) windowReturn.settle(earlyRows.get);
                _historyRow = null;
                for (const [tid, ending] of endings) {
                    if (!ending) continue;
                    const record = liveCardRecords.get(tid);
                    if (!record) continue;
                    const { detail, phase, rich, stamp, drawn } = ending;
                    _historyRow = detail;
                    applyCardEnding(record, stamp);
                    if (subagentChildParents.has(tid)) {
                        routeSubagentTerminalToCard(tid, detail, false, phase);
                        continue;
                    }
                    record.quietEffective = detail.quiet_effective ?? record.quietEffective;
                    insertCardIfNeeded(tid);
                    if (rich) appendTaskSummaryToLiveCard(detail, { currentOutcome: true, drawn });
                    else { if (detail.suggested_name) applySuggestedName(tid, detail.suggested_name);
                        noteTaskDetails(record, detail); finishLiveCard(tid, phase); }
                }
                _historyRow = null;

                // Cards without terminal or census evidence remain unconfirmed.
                for (const [id, record] of liveCardRecords) {
                    mountReplayedCard(record);
                    if (!knownCards.has(id) && !record.finished && !isPortal(record) && !currentActivity(record)) {
                        setHistoricalUnconfirmed(record);
                        syncCancelRunButton(record);
                        renderLiveCardMeta(record);
                    }
                }
                for (const row of messages) {
                    const record = liveCardRecords.get(row.presentation_owner_task_id || row.task_id);
                    if (!record) continue;
                    const revision = row.current_outcome?.revision;
                    if (revision && !(isoMicros(revision) < isoMicros(record.latestSourceTs))) record.latestSourceTs = revision;
                    record.historyIds ||= new Set();
                    for (const id of historyRowIds(row)) record.historyIds.add(id);
                }
                _historyReplayActive = false;
                // The cards this page changed are drawn with the scroll and selection they had, read for all at once.
                const dirty = [...liveCardRecords.values()].filter(record => record._timelineDirty);
                measure(() => { const kept = keepSelection(document); return dirty.map(record => ({ scroll: timelineScroll(record), kept })); },
                    (before) => dirty.forEach((record, index) => renderLiveCardTimeline(record, before[index])));
                return reportActivity && ([...messagesDiv.children].some(node => node.classList.contains('chat-bubble') && !knownRows.has(node))
                    || [...liveCardRecords].some(([id, record]) => {
                        const before = knownCards.get(id);
                        // Reconciliation can change liveness or placement of words already seen; that is no arrival.
                        return record.root?.isConnected && (record.items.length > (before?.items || 0)
                            || (!before?.projection || isoMicros(record.latestSourceTs) > isoMicros(before.source))
                                && liveCardProjectionChanged(before?.projection || null, record));
                    }));
            }, { remoteContent: reportActivity });
        } finally {
            _historyRow = null;
            _historyReplayActive = false;
            _historyAppending = false;
            replayKeys = null; replaySelection = null;
        }
        portals.refresh();
        syncChatStatus();
    }

    async function syncHistory({ fromReconnect = false, ...tags } = {}) {
        if (destroyed || leaving) return false;
        if (foreignProfile()) return false;  // the socket serves another profile: read nothing into this window (#3294)
        if (historySyncPromise) return historySyncPromise;
        historySyncPromise = (async () => {
            const startedVisit = visit;
            const abort = historyReadAbort = new AbortController();
            const armedAtStart = liveCardBound.begin();
            const cardsAtStart = new Set(liveCardRecords.keys());
            // Replay can refill the feed without moving the reader's intent.
            const followingAtStart = reading.stick, generationAtStart = reading.generation;
            const landedAtStart = historyPager.arrivalsLanded(), recovering = !lastHistorySyncSucceeded || catchUp.owed;
            try {
                // No welcome until this read lands.
                emptyWelcome?.historyPending();
                // An empty feed shows the read in flight (#1102); a painted one stays.
                if (historyControls.beginRecent()) syncLoadOlderControl();
                const data = await fetchHistory(null, { signal: abort.signal, tags: { ...tags, mode: 'newest' } });
                // A closed, left or superseded room takes no late response.
                if (destroyed || startedVisit !== visit || foreignProfile()) {
                    lastHistorySyncSucceeded = false;
                    roomProfile.resetInitialHydration();
                    return false;
                }
                if (data.recentVersion < recentApplied) return lastHistorySyncSucceeded;
                const messages = Array.isArray(data.messages) ? data.messages : [];
                const oldRecentIds = retainedRecentIds();
                const admitted = acceptRecentWindow(data, messages);
                const pagerBeforeRecent = historyPager.getState();
                const reader = readingHistory(), returning = reading.returning;
                const rechainRecent = admitted && !catchUp.owed && (!reader || returning) && pagerBeforeRecent.initialized
                    && (returning || pagerBeforeRecent.away && (fromReconnect || recovering || pagerBeforeRecent.retryReturn)
                        || !pagerBeforeRecent.canNewer && [...oldRecentIds].some(id => !recentHistoryIds.has(id)));
                const result = historyPager.acceptRecent(data, rechainRecent && { fromReconnect });
                if (result.status !== 'applied') applyHistoryMessages(messages, { fromReconnect, newest: admitted,
                    requestedAt: data.requestedAt, window: data.window, hasMore: data.has_more });
                withStableViewport(() => releaseHistoryIds(!admitted || rechainRecent || reader
                    || historyPager.getState().canNewer ? [] : oldRecentIds));
                // Arrivals that landed meanwhile may lie past this page: read them again.
                if (historyPager.arrivalsLanded() !== landedAtStart) void historyPager.read({ reason: tags.reason, decision: 'base_superseded' });
                if (armedAtStart) {
                    const represented = new Set(messages.map(row => row.presentation_owner_task_id || row.task_id));
                    for (const id of cardsAtStart) if (!represented.has(id)) pendingLiveEvictions.add(id);
                    withStableViewport(releaseLiveOverflow);
                }
                syncChatStatus();

                composer.seedInputHistory(messages);

                const wasFirstLoad = !historyLoaded;
                // The first source read opens at the newest message unless the reader moved.
                const opensAfterPreview = wasFirstLoad && followingAtStart && reading.generation === generationAtStart;
                historyLoaded = true;
                lastHistorySyncSucceeded = true;
                messagesDiv.dataset.historyHydrated = 'true';
                emptyWelcome?.historyRead(data.window?.complete === true);
                liveCardBound.settle({ rebuilt: wasFirstLoad || armedAtStart, size: liveCardRecords.size });
                modelWaits.retainCards(liveCardRecords);
                // Later hydration joins this successful source read.
                roomProfile.settleInitialHydration(historySyncPromise);
                syncLoadOlderControl();
                // A first opening follows the newest message (owner 2026-07-10,
                // 2026-10-05); an in-room intent still awaiting data lands first.
                if (reading.pending) {
                    updateMessagesPadding(false);
                    reading.position();
                } else if ((wasFirstLoad && reading.stick) || opensAfterPreview) {
                    updateMessagesPadding();
                    reading.followAfterLayout();
                }
                readReceipt.settle();
                roomProfile.noteCanonicalHistory(wasFirstLoad);
                return messages.length > 0;
            } catch (err) {
                if (destroyed || startedVisit !== visit) { roomProfile.resetInitialHydration(); return false; }
                lastHistorySyncSucceeded = false;
                preview.failed(err);
                emptyWelcome?.historyRead(false);
                roomProfile.resetInitialHydration();
                // Migration keeps a calm note and retries through the existing 700 ms scheduler.
                historyControls.endRecent(err); syncLoadOlderControl();
                roomProfile.noteHistoryFailure(err);  // kept sends may still show, on current proof (chat_preview.js)
                if (isStoreMigrating(err)) { historyResyncScheduler.schedule(true, { reason: 'retry', detail: 'migrating' }); return false; }
                // A fetch that failed while the socket is down is that outage, not an error to report.
                const expectedDisconnect = ws?.ws?.readyState !== WebSocket.OPEN && err instanceof TypeError;
                if (!expectedDisconnect) console.error('Failed to load chat history:', err);
                return false;
            } finally {
                if (historyReadAbort === abort) historyReadAbort = null;
                historySyncPromise = null;
            }
        })();
        return historySyncPromise;
    }

    // A new revision needs a read begun after it: one in flight is awaited, never
    // joined; the sticky hydration answers a covered one (project_read_state.js).
    const newestFacts = { delivered: id => retainedHistoryIds().has(id), nodes: id => historyNodes(id),
        viewport: messagesDiv, header: pageHeader, composer: inputArea };
    // A message this room shares with another is read once wherever it was shown whole (read_ahead.js).
    const sharedRead = createSharedReadTracker({ chatId, feed: messagesDiv, band,
        shown: () => !destroyed && !leaving && !page.hidden && isInstanceVisible() });
    const RECEIPT = { reason: 'read_receipt' };  // its reads and the pages its opening walks
    const readReceipt = createProjectReadReceipt({ onNote: sharedRead.sample,
        read: async (fresh, whole, why) => {
            await (!fresh && (catchUp.reading || !catchUp.owed && roomProfile.initialHydration)
                || historyPager.read({ whole, ...RECEIPT, ...why, decision: whole && 'latest_unknown' }));
            return lastHistorySyncSucceeded && !catchUp.owed;
        },
        // A destroyed page reports hidden===false, hence the lifecycle flag.
        isShown: () => !destroyed && !leaving && !page.hidden && !archive?.active,
        // A place still being restored is not where the reader is.
        isReadingLatest: latest => !archive?.active && !destroyed && !catchUp.owed && !reading.pending && !replies.revealing && isInstanceVisible() && isAtNewestMessage(latest, {
            ...newestFacts, atBottom: () => reading.nearBottom() && !historyPager.getState().canNewer,
        }),
        onReadingLatest, arriving: () => reading.arriving, delivered: newestFacts.delivered,
        older: async current => {
            await historyPager.whenIdle();
            if (!current()) return {};
            const direction = readReceipt.continuation(pagerRecent);
            // A failed present keeps the existing horizon repair.
            if (direction === 'latest') return historyPager.latest(null, RECEIPT);
            if (direction && !historyPager.getState().error) return historyPager.older(RECEIPT);
            return historyPager.edge('older', { current, tags: RECEIPT });
        },
        present: current => historyPager.whenIdle().then(() => current() && historyPager.getState().canNewer && historyPager.latest(null, RECEIPT)),
        land: (latest, token) => reading.land(() => newestMessageOffset(latest, newestFacts), token),
    });

    function reconcileSavedSends(word) {
        unconfirmed.reconcile(({ frame, reply, views, ts }) => addMessage(frame.content, 'user', false, ts || null,
            false, { source: 'web', senderSessionId: chatSessionId, clientMessageId: frame.client_message_id, attachments: views, reply }), word);
    }

    const opening = roomProfile.startOpening();

    // The composer (composer.js): its field, the room's capsules, attachments and the send path.
    composer = createComposer({ page, byId, ws, replies, chatSessionId, isMain, chatId, projectId, storeKey,
        durableDraft: Boolean(roomProfile.profileScope), refreshState: refreshHeaderControlState,
        openSettings, onLayout: () => updateMessagesPadding(),
        onSent: showSent, readingBack: () => !reading.stick && reading.ownMove });

    // A send the composer made, shown in the feed: the bubble, its doubt until the host's saved echo, `Sending...`.
    function showSent({ text, frame, reply, result, views }) {
        const sentTs = new Date().toISOString();
        reading.cancel(); // an accepted Send supersedes any saved place
        const row = addMessage(text, 'user', false, sentTs, false, {
            pending: result?.status === 'queued',
            source: 'web',
            senderSessionId: chatSessionId,
            clientMessageId: result?.clientMessageId || '',
            forceStick: true, reply,
            attachments: views,
        });
        // Only the host's saved echo proves acceptance: every message's frame stays (a bare slash command is a
        // command); a queued one is the socket queue's until it is written.
        if (views.length || frame.reply_excerpt || !text.startsWith('/')) unconfirmed.track({ ...frame, client_message_id: result.clientMessageId },
            { views, reply, ts: sentTs, queued: result?.status === 'queued' });
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
    // An image's load reflows in the next frame, after its scroll events: a load between a move (a wheel step, a
    // scrollIntoView) and the scroll event it still owes would else restore the anchor of the place just left.
    let loadReflowQueued = false;
    messagesDiv?.addEventListener('load', () => {
        if (loadReflowQueued) return;
        loadReflowQueued = true;
        requestAnimationFrame(() => { loadReflowQueued = false; if (!destroyed) reading.reflow(); });
    }, true);

    // ↓: navigation, coalesced remote activity (not a live region), and busy/Retry for the present read (style.css).
    function updateScrollButton() {
        if (!scrollBottomBtn) return;
        _hasNewActivity &&= !reading.nearBottom(ACTUAL_BOTTOM_TOLERANCE_PX);
        const busy = reading.returning;
        const retry = historyWindow?.failed || catchUp.retry && !historyWindow?.waiting;
        const label = busy ? 'Loading latest messages…' : retry ? 'Retry loading latest messages'
            : _hasNewActivity ? 'New activity — scroll to latest message'
            : 'Scroll to latest message';
        scrollBottomBtn.setAttribute('aria-label', label);
        scrollBottomBtn.title = label;
        scrollBottomBtn.setAttribute('aria-busy', String(busy));
        if (scrollActivityDot) scrollActivityDot.hidden = !_hasNewActivity;
        scrollBottomBtn.classList.toggle('visible', isInstanceVisible() && (busy || retry || !reading.nearBottom() || (!reading.stick && historyWindow?.gaps)));
    }
    // Freshness gates reads, not retained movement.
    function presentHeld() {
        return historyLoaded && !catchUp.owed && !historyWindow?.failed && !historyPager.getState().away && !historyWindow?.horizonGap;
    }
    let resumeGeneration;  // Main's automatic return isn't a new navigation away from the live edge.
    async function showLatest({ arrive = false, reason = 'navigate' } = {}) {
        if (reading.returning && reading.arriving === null && !arrive) { reading.followAfterLayout(); return; }
        const at = visit, claim = reading.claim(arrive === true);
        if (reason === 'resume') resumeGeneration = reading.generation;
        const current = () => !destroyed && at === visit && claim();
        historyPager.toPresent();
        cardRowAnchors.toPresent();
        const needed = !presentHeld();
        const follow = () => {
            reading.stick = true;
            for (const record of liveCardRecords.values()) record.childDetailRead = undefined;
            reading.followAfterLayout();
            historyControls.showNotice();
        };
        reading.wait(current, needed);
        if (!arrive || !needed) follow();
        if (!needed) return;
        try {
            const shared = catchUp.reading || historySyncPromise || !historyLoaded && opening.then(awaitInitialHydration);
            await shared;
            await catchUp.settled(current);
            if (!current() || shared && !lastHistorySyncSucceeded) return;
            if (!presentHeld() && (await historyPager.latest(current, { reason })).status !== 'applied') return;
            await catchUp.settled(current);  // a reconnect during that read still owes feedback
            if (current() && !catchUp.owed && arrive) follow();
        } finally { reading.wait(current, false); }
    }
    scrollBottomBtn?.addEventListener('click', showLatest);
    // Return as left: older reading or following.
    function resume(arrive = false) {
        if ((reading.target || reading).stick === false) {
            _hasNewActivity ||= arrive; reading.position(); updateScrollButton(); return;
        }
        const returning = showLatest({ arrive, reason: 'resume' });
        reading.request(); // Restore following even on failure, without fresh read credit.
        return returning;
    }
    const keepPlace = () => { if (isInstanceVisible()) { reading.cancel(); reading.request(); } };

    function updateMessagesPadding(preserveStickiness = true) {
        // Off screen, keep the last nonzero reserves.
        if (leaving || !page.getClientRects().length) return false;
        // Stage only changed CSS properties; then apply the measurements in one timeline transaction.
        const changes = [];
        const setProperty = (name, value) => page.style.getPropertyValue(name) !== value && (changes.push([name, value]), true);
        const shortMoved = syncShortRoom(page);
        let chromeMoved = false;
        if (pageHeader && messagesDiv) {
            // The bar's own height: what floats under it hangs there, and the transcript clears it.
            const bar = Math.ceil(pageHeader.offsetHeight || 0);
            setProperty('--chat-header-bar', `${bar}px`);
            setProperty('--chat-header-reserve', `${Math.max(56, bar)}px`);
        }
        if (inputArea && messagesDiv) {
            setProperty('--chat-composer-height', `${composerHeightAtRest(inputArea)}px`);
            // Keyboard up: all but feed and field text is chrome (style.css), read at rest.
            const box = el => el.getBoundingClientRect();
            if (document.body.classList.contains('keyboard-open') && !inputArea.querySelector('[data-motion]')) {
                const top = readBand(messagesDiv, pageHeader, null).top;
                chromeMoved = setProperty('--chat-keyboard-chrome', `${Math.round(top - box(pageHeader || page).top
                    + box(inputArea).height - box(input).height)}px`);
            }
        }
        placeRoomEdges(pageHeader, inputArea);  // a room's edges end at its capsules (page_header.js)
        if (changes.length || shortMoved) {
            const write = () => { for (const [name, value] of changes) page.style.setProperty(name, value); };
            const apply = () => preserveStickiness ? withStableViewport(write) : write();
            if ((chromeMoved || shortMoved) && composer) composer.remeasureInput(apply);
            else if (changes.length) apply();
        }
        // The scrollbar this feed draws (none where scrollbars overlay the page): a Project's focus corners keep off it.
        if (asPanel) {
            const lane = `${messagesDiv.offsetWidth - messagesDiv.clientWidth}px`;
            if (lane !== page.style.getPropertyValue('--chat-feed-lane')) page.style.setProperty('--chat-feed-lane', lane);
        }
        updateScrollButton();
    }

    // Kept on the instance so destroy() can disconnect it (the observer was
    // previously an unreachable closure — the P3 lifecycle leak).
    let chatResizeObserver = null;

    function reflowViewport() {
        if (destroyed || leaving) return;
        reading.reflow(updateMessagesPadding);
    }

    function installChatResizeObservers() {
        if (typeof ResizeObserver !== 'function') return;
        let queued = false;
        const schedule = () => {
            if (queued) return;
            queued = true;
            requestAnimationFrame(() => {
                queued = false;
                reflowViewport();
            });
        };
        chatResizeObserver = new ResizeObserver(schedule);
        chatResizeObserver.observe(messagesDiv);
        if (pageHeader) chatResizeObserver.observe(pageHeader);
        if (inputArea) chatResizeObserver.observe(inputArea);
    }

    const syncComposerReserve = () => updateMessagesPadding();
    inputArea?.addEventListener('ouro:composer-reserve', syncComposerReserve);
    installChatResizeObservers();
    const unwatchEdges = watchRoomEdges(globalThis.window, () => placeRoomEdges(pageHeader, inputArea));

    // Open Project room's feed holds the keys (app.js focusProjectRoom), focus drawn as two corners
    // (style.css): a pointer in panel quiets them; any key restores them (Back to reading from the
    // composer too) but lone modifiers and non-scrolling shortcuts (Cmd+C, Cmd+Shift+4).
    const focusPanel = asPanel ? container.closest?.('.project-panel') : null;
    const quietFocus = () => messagesDiv.classList.add('is-quiet-focus');
    const showFocus = ({ key, metaKey, ctrlKey, altKey }) => {
        if (metaKey || ctrlKey || altKey ? /^(Arrow|Page|Home$|End$| $)/.test(key)
            : !/^(Shift|CapsLock|Fn)$/.test(key)) messagesDiv.classList.remove('is-quiet-focus');
    };
    focusPanel?.addEventListener('pointerdown', quietFocus);
    focusPanel?.addEventListener('keydown', showFocus);

    if (asPanel) {
        // Late panels missed socket open; seed from socket and latest snapshot, its works: a working
        // Project opened from row said Online until first read (owner 20.10).
        const latest = stateSnapshots.latest?.();
        hostStart.read(latest);
        workState = reconcileWorkState(workState, latest, chatId, ws.isConnected?.() !== false);
        const seed = deriveChatStatus();
        if (ws.isConnected?.()) setStatus(seed);
    }
    // Both Main and late panels hydrate once; each instance filters its chat. The page's poll is app.js's.
    refreshHeaderControlState();

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
    let beforePartialIds = null;
    const retainedRecentIds = () => new Set([...recentHistoryIds, ...(beforePartialIds || [])]);
    let recentRequest = 0, recentApplied = 0;
    // Admit each newest response once, including ↓; the preview follows it.
    function acceptRecentWindow(data, messages) {
        if (data.recentVersion <= recentApplied) return false;
        recentApplied = data.recentVersion;
        const whole = catchUp.confirmWindow(data);
        preview.save(messages, data);
        // The newest admitted window (↓ too) owns the failure note.
        historyControls.endRecent(whole ? null : new Error('Some saved history could not be loaded.'), Boolean(data.page_cursor), data);
        recentCoverage = data.coverage ?? null;
        recentHasOrigins = messages.some(row => row.origin_projected);
        readReceipt.recent(data);
        // Keep the rows held before the fault across a rechain; repeated partial windows replace one another.
        if (whole) beforePartialIds = null;
        else beforePartialIds ??= retainedHistoryIds();
        recentHistoryIds = new Set(messages.flatMap(historyRowIds));
        return true;
    }
    const pendingHistoryEvictions = new Set();
    const historyNodes = (id) => Array.from(messagesDiv.querySelectorAll('[data-history-id]'))
        .filter(node => node.dataset.historyId === id);
    const retainedHistoryIds = () => new Set([...retainedRecentIds(),
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
        sharedRead.release(node);
        replies.release(node);
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
        // a card keeps its timeline's selection. All are read first, in one layout.
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
                if (node.dataset.letterPart) seenMessageKeys.delete(node.dataset.messageKey);
                releaseMessageNode(node);
                if (wrapper && wrapper !== node && !wrapper.querySelector('.chat-gallery-item, .chat-file-item, .chat-quiz-card')) {
                    releaseMessageNode(wrapper);
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
            forgetLiveCard(id);
            historicalTerminals.delete(id);
        }
    }

    // A room's news waits until it is the room shown and leaves when it is covered or left (room_notice.js).
    const historyCovered = roomCovered(asPanel ? container?.closest?.('.project-panel') : pageHeader, pageHeader);
    const historyVisible = () => isInstanceVisible() && !messagesDiv.closest('[inert]') && !historyCovered()
        && !document.body?.classList.contains('nav-drawer-open') && (asPanel || (!isProjectOpening?.()
        && !document.body?.classList.contains('project-panel-open')));
    const archive = isMain && !asPanel ? archiveDoor({ returnFocus: messagesDiv,
        readReceipt, syncHistory, isVisible: historyVisible }) : null;
    const historyControls = createHistoryControls(messagesDiv, { notify: roomNotice, archive, others: [mainWaiting?.notice], isVisible: historyVisible });
    const coverWatch = watchRoomCover(asPanel ? container?.closest?.('.project-panel') : pageHeader, pageHeader,
        () => (historyVisible() ? historyControls.showNotice() : historyControls.hideNotice()));
    const { olderButton: loadOlderBtn } = historyControls;

    // The first gate closed to arrivals ('' none: a hydrated, healthy window that needs no rebuild).
    const arrivalGate = () => (!historyLoaded ? 'not_loaded' : !lastHistorySyncSucceeded ? 'prior_failed'
        : catchUp.owed ? 'catch_up_owed' : liveCardBound.isArmed() ? 'live_card_bound' : '');
    const historyPager = createChatHistoryPager({
        fetchPage: fetchHistory,
        isAlive: () => !destroyed && !leaving,
        applyPage: (messages, descriptor) => {
            if (foreignProfile()) return;  // another profile's history rows never paint here (#3294)
            const oldRecentIds = retainedRecentIds();
            const admitted = descriptor.direction === 'latest' && (acceptRecentWindow(descriptor, messages) || descriptor.rechained);
            pageHistoryIds.set(descriptor.id, new Set(messages.flatMap(historyRowIds)));
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
        // Rows stored since the newest read join its window (and the saved preview).
        applyArrivals: (messages, { window, fromReconnect, requestedAt }) => {
            if (foreignProfile()) return;
            preview.append(messages);
            for (const id of messages.flatMap(historyRowIds)) recentHistoryIds.add(id);
            applyHistoryMessages(messages, { fromReconnect, requestedAt });
            readReceipt.arrived(window);
        },
        readNewest: syncHistory, newestInFlight: () => historySyncPromise,
        canArrive: () => !arrivalGate(), arrivalGate,
        rebaseOlderEdge: ({ canNewer, canOlder }) => historyWindow?.horizonGap && !canNewer && !(canOlder && readingHistory()),
        isPageProtected: isHistoryPageProtected,
        onState: snapshot => syncLoadOlderControl(snapshot),
    });
    // The saved text shown until the first canonical read (chat_preview.js); `self` carries room_retire.js's kept selection.
    let self = null;
    const preview = createChatPreview({ feed: messagesDiv, room: [chatId, projectId], label: senderLabel,
        mutate: withStableViewport, kept: () => self?.keptRanges, canShow: () => ws?.canShowSavedProfile?.() === true });

    function syncLoadOlderControl(snapshot = historyPager.getState()) {
        if (destroyed || leaving) return;
        withStableViewport(() => {
            const coverage = historyCoverage(recentCoverage, snapshot.coverage);
            if (recentHasOrigins && !coverage.complete) coverage.gaps = true;
            historyWindow = historyControls.render(snapshot, coverage, reading.approximate);
            return true;
        });
    }
    const handleHistoryLanguage = () => withStableViewport(() => {
        syncLoadOlderControl();
        for (const record of liveCardRecords.values()) renderLiveCardMeta(record);  // its line's own words (modelWords, clockLabel)
        replies.retranslateDerivedQuotes();  // a mounted derived question quote's lead, room label and aria follow the new dictionary
    });
    window.addEventListener('ouro:language-changed', handleHistoryLanguage);
    async function loadOlderHistory() {
        if (historyControls.recentFailed()) {
            const owned = onHistoryRetry?.();
            return historySyncPromise ? owned : syncHistory({ reason: 'retry' });
        }
        const snapshot = historyPager.getState();
        // ↓'s own failed read is a fresh ↓, later reading supersedes it; a changed view reads the present in place.
        if (snapshot.retryReturn) return showLatest({ reason: 'retry' });
        if (snapshot.error?.body?.reason_code === 'history_view_changed') {
            reading.cancel();
            return historyPager.latest(null, { reason: 'retry', decision: 'view_changed' });
        }
        return snapshot.error ? historyPager.retry() : historyPager.edge('older');
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
        if (direction < 0 && messagesDiv.scrollTop < 80 && snapshot.canOlder) void historyPager.edge('older');
        else if (direction > 0 && snapshot.canNewer) {
            // Only an unambiguous page-owned island has a physical edge. A
            // mixed card has no chronological boundary: use the same button.
            const ids = pageHistoryIds.get(snapshot.firstPage?.id) || new Set();
            if (historyIslandAtEdge(messagesDiv, ids, historyNodes)) void historyPager.edge('newer');
        }
    }
    // Only the owner's own move toward older messages puts a phone keyboard away.
    const disposeReadingGestures = reading.bindGestures(direction => {
        if (direction < 0 && document.activeElement === input && document.body.classList.contains('keyboard-open')) input.blur();
        navigateHistoryAtEdge(direction);
    });
    // One compact Markdown binding, released with the column.
    bindCompactMarkdown(messagesDiv);
    function retryHistoricalUpserts() {
        if (destroyed || leaving) return;
        const ready = [...pendingHistoryUpserts.values()].filter(row =>
            !historyNodes(row.history_id).some(node => historyNodeIsProtected(node, messagesDiv)));
        if (ready.length) applyHistoryMessages(ready, { archived: true });
    }
    document.addEventListener('selectionchange', retryHistoricalUpserts);

    // Active blocks host their own running indicator.
    const foregroundCards = () => Array.from(liveCardRecords.values()).filter((r) => isForegroundLiveCard(r) && !r.modelWaiting);

    // The server's room under this client's own facts.
    function deriveChatStatus() {
        return roomStatusView(workState, { connected: ws.isConnected ? ws.isConnected() : true,
            serverAnswers: ws.serverAnswersWhileDown?.() === true, catchingUp: catchUp.active,
            ready: hostStart.ready(),
            boot: hostStart.status().bootPhase,
            sending: pendingSubmissions.size > 0, stopping: ownStopPress(workState, liveCardRecords) });
    }

    // A room away draws no status: its paused socket view is no outage; it derives the status again on its return.
    function syncChatStatus() {
        if (leaving) return false;
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

    function reconcileMissingManagedTask(taskId, onDomWrite = withStableViewport) {
        const record = liveCardRecords.get(taskId), started = visit, portal = isPortal(record);
        if (!record || destroyed || leaving || managedTaskDetailReads.has(taskId)
                || concludedDirectActivities.has(taskId) || !missingManagedTaskIds.has(taskId)) return;
        const shown = () => !destroyed && !leaving && visit === started && liveCardRecords.get(taskId) === record
            && isPortal(record) === portal;
        readMissingCardDetail(record, shown, (detail, epoch) => {
            onDomWrite(() => {
                if (portal) {
                    if (detail !== null) return attachTaskDetailReviews(taskId, detail, true);
                    return !record.finished && !record.observedOutcome && portals.acceptsDetail(record, null, epoch)
                        && settleMissingTaskRecord(record);
                }
                if (record.finished || concludedDirectActivities.has(taskId)) {
                    return isTerminalTaskDetail(detail) && attachTaskDetailReviews(taskId, detail, true);
                }
                const vouched = !missingManagedTaskIds.has(taskId) || activeDirectActivities.has(taskId);
                let changed = attachTaskDetailReviews(taskId, detail, true);
                const cancelPending = taskCancelPending(detail);
                if (cancelPending || taskKey(detail?.status)) {
                    changed = setHistoricalUnavailable(record, false) || changed;
                    changed = markReviewAnchor(record) || changed;
                }
                if (!vouched && detail === null) {
                    revokeManagedTaskCancelAuthority(taskId);
                    return settleMissingTaskRecord(record) || changed;
                }
                if (cancelPending || (!vouched && !isTerminalTaskDetail(detail))) {
                    if (record.isSubagent && detail?.status) missingManagedTaskIds.delete(taskId);
                    return Boolean(reconcileCancelCardFromDetail(record, taskId, detail) || changed);
                }
                if (vouched) return changed;
                recordTerminalActivity(taskId, taskTerminalPhase(detail));
                return Boolean(appendTaskSummaryToLiveCard({ ...detail, task_id: taskId }) || changed);
            });
        }, s => reviewHydrator.note(taskId, s, onDomWrite));
    }

    function observeMissingManagedTask(taskId, onDomWrite = withStableViewport) {
        const id = taskKey(taskId);
        if (!id || concludedDirectActivities.has(id) || subagentChildParents.has(id) || liveCardRecords.get(id)?.isSubagent) return;
        missingManagedTaskIds.add(id);
        reconcileMissingManagedTask(id, onDomWrite);
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
            if (isPortal(record) || attemptOf(v) < record?.cardAttempt) continue;
            if (!connected) v._activityUnconfirmed = true;
            const facts = censusTaskFacts(v);
            if (facts.ended) {
                recordTerminalActivity(k, facts.outcome);
                if (record) finishLiveCardMutation(k, facts.outcome);
                continue;
            }
            activeDirectActivities.set(k, v);
            restoreCardActivity(record, v.project_admission_hold);
            syncParkedPhase(record, v.phase, v);
            syncCancelRunButton(record);
            markReviewAnchor(record);
            if (v.kind === 'managed_task') missingManagedTaskIds.delete(k);
        }
        for (const taskId of globallyActiveActivityIds) {
            if (!isPortal(liveCardRecords.get(taskId))) missingManagedTaskIds.delete(taskId);
        }
        for (const row of settledDirectRows) {
            // Visible task cards settle from durable detail in the scan below.
            const record = liveCardRecords.get(row.activityId);
            if (!REUSABLE_TASK_IDS.has(row.activityId) && !isPortal(record) && !isForegroundLiveCard(record)) {
                recordConcludedActivity(row.activityId);
            }
            if (row.clientMessageId) pendingSubmissions.delete(row.clientMessageId);
        }
        for (const taskId of departedManagedTaskIds) revokeManagedTaskCancelAuthority(taskId);
        for (const taskId of disappearedManagedTaskIds) {
            if (!isPortal(liveCardRecords.get(taskId))) observeMissingManagedTask(taskId);
        }
        if (complete) for (const taskId of unconfirmedForegroundCardIds(
            Array.from(liveCardRecords, ([id, r]) => ({
                id, finished: r.finished, isSubagent: r.isSubagent, connected: r.root?.isConnected && !isPortal(r),
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
            const record = liveCardRecords.get(taskId);
            if (!isPortal(record) && !currentActivity(record)) void reconcileMissingManagedTask(taskId);
        }
    }

    onWs('typing', (msg) => {
        if (!isMyThread(msg)) return;  // each column shows typing only for its own thread
        // Typing frame is submission receipt, never liveness: re-reads state at once, retires linked
        // `Sending...` only when that read answers: room steps from Sending... to server's word.
        const cmid = String(msg.client_message_id || '');
        if (!cmid) return;
        void refreshHeaderControlState().then(() => {
            if (pendingSubmissions.delete(cmid)) syncChatStatus();
        });
    });

    const isMyThread = msg => chatThreadAccepts(msg, isMain, chatId, state.projectChatIds);
    const isMyLogThread = msg => chatLogThreadAccepts(msg, isMain, chatId, state.projectChatIds);
    const windowReturn = createWindowReturn({ socket: ws.socket, doc: document, chatId, projectId,
        accepts: isMyThread, knownChild: id => subagentChildParents.has(id), onInvalidate: () => earlyRows.forget() });
    const windowFollower = () => reading.stick && (resumeGeneration === reading.generation || !reading.pending && !reading.returning)
        && !archive?.active && !replies.revealing
        && !page.hidden && messagesDiv.offsetParent !== null && !messagesDiv.closest('[inert]');
    function forgetWindowReturn() {
        windowReturn.drop();
        earlyRows.forget();
    }

    onWs('chat', (msg) => {
        if (foreignProfile()) return;  // live frames of another profile never render (#3294)
        if (!isMyThread(msg)) return;
        const address = msg.role === 'assistant' && !msg.is_progress && msg.row_address;
        if (earlyRows.get(address)) return;  // its words are already shown, not confirmed by history
        if (msg.system_type === 'initialization_notice') {
            unconfirmed.refused(msg);  // its typed cause decides a copy's word (chat_attachments.js)
            if (leaving) return;
        }
        if (msg.system_type === 'project_question_pointer') { chatDecision.appendQuestionPointer(msg); return; }
        if (msg.role === 'user') {
            const clientMessageId = msg.client_message_id || '';
            const senderSessionId = msg.sender_session_id || '';
            // 2A: user echo is receipt of user ROW, not turn start: settles bubble but must NOT retire the
            // `Sending...` submission (linked typing frame, snapshot turn, routing receipt or conclusion does).
            const local = localOwnerRow(msg);
            const own = local && senderSessionId === chatSessionId && clientMessageId;
            if (own) {
                markPendingDelivered(clientMessageId);
                const pending = pendingSubmissions.get(clientMessageId);
                if (pending && msg.row_address) pending.rowAddress = msg.row_address;
            }
            // The custody listener can settle this room's send while away; another client's echo waits for history.
            if (leaving) {
                if (own) markIngressSaved(messagesDiv, msg, ownerMessageBubble(messagesDiv, msg, chatId));
                if (local) unconfirmed.settle(msg);
                return;
            }
            if (!own) withRemoteActivity(() => addMessage(
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
            withStableViewport(() => confirmOwnerRow(msg));
            syncChatStatus();
            return;
        }

        if (leaving) return void showTaskIncidentToast(msg);  // away, a task's incident toasts; nothing is drawn
        if (msg.role === 'assistant' || msg.role === 'system') {
            return withRemoteActivity(() => {
            const explicitTaskId = msg.task_id || '';
            // Every path syncs status last. Main's number is its room revision, as a Project's (DESIGN "Unread").
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
            if (letterRow(msg, messagesDiv, addMessage)) return settled(true);
            if (msg.is_progress) {
                showTaskIncidentToast(msg);
                if (SUBAGENT_LIFECYCLE_EVENTS.has(String(msg.subagent_event || '').toLowerCase())) reread();  // members
                const changed = updateLiveCardFromProgressMessage(msg, { grantCancelAuthority: true });
                return settled(changed);
            }

            // Finalizing awaits task_done/census. A repeat never concludes a later legacy turn.
            const finalizing = Boolean(explicitTaskId) && msg.task_phase === 'finalizing';
            const typedTerminal = positiveTaskTerminalFact(msg);
            if (typedTerminal) reread();
            if (!finalizing && (explicitTaskId ? typedTerminal : !seenMessageKeys.has(`window|${address}`))) {
                if (explicitTaskId) {
                    // 4A (active set): a keyed final concludes ITS OWN turn (the finished activity and its linked
                    // pending), never a concurrent turn's state (2A keeps later `Sending...`).
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
                taskId: explicitTaskId, reply: msg, projectId: msg.project_id, projectName: msg.project_name,
                // The saved row's own name, as history keys it; this frame's client_message_id is the owner's.
                clientMessageId: msg.row_client_message_id || '',
                // A placed row whose card is not in this chat keeps its record link.
                evidenceRef: placedCardRow(msg) ? cardRowEvidenceRef(msg) : null,
            });
            return settled(Boolean(added || changed || routingCleared));
            });
        }
    });

    onWs('message_annotation', (msg) => {
        if (msg.annotation_type !== 'routing_ack') return;
        // Only the host places relay reads, including resolved older quotes.
        if (msg.action === 'steer_task' && msg.status === 'read' && !msg.source_row_address) return;
        const own = isMyThread(msg);
        const apply = () => chatDecision.applyRoutingAck(messagesDiv, msg, { own, journal: localEchoJournal });
        (own && msg.status === 'needs_manual_target' ? withRemoteActivity : withStableViewport)(apply);

        const receiptCid = String(msg.client_message_id || '');
        if (own && receiptCid && Number(msg.chat_id ?? 1) === chatId &&
                (msg.source_row_address ? msg.source_row_address === pendingSubmissions.get(receiptCid)?.rowAddress : msg.source_row_address !== '') &&
                pendingSubmissions.delete(receiptCid)) {
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
        // A log frame carries its task's canonical Project chat_id: only that panel builds/animates/finalizes its
        // card; a legacy frame without one goes to Main.
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

    let wsHasConnectedOnce = false, closedSinceOpen = false;

    onWs('open', (msg) => {
        const openedVisit = visit, readOnOpen = recentRequest;
        // A return across a close it heard (shown or away) is a reconnect.
        const reason = msg?.resumed && !closedSinceOpen ? 'resume' : 'reconnect';
        closedSinceOpen = false;
        // The socket's previouslyConnected covers newly mounted Projects too;
        // per-instance state is only the fallback for legacy open events.
        const isReconnect = typeof msg?.previouslyConnected === 'boolean'
            ? msg.previouslyConnected
            : wsHasConnectedOnce;
        // A body polled during a real outage may predate the restart.
        if (isReconnect && !msg?.resumed) hostStart.forget();
        mainWaiting?.setConnected(true);
        refreshHeaderControlState();
        syncChatStatus();
        // Main owes a changed-code notice even if away at first open; a Project mounted later owes none.
        if (isReconnect && !wsHasConnectedOnce && !isMain) pendingRestartNotice = '';
        const restartNotice = pendingRestartNotice;
        wsHasConnectedOnce = true;
        updateMessagesPadding();
        const settled = () => {
            if (destroyed || openedVisit !== visit) return;  // left meanwhile: a return opens again
            catchUp.end(read);
            if (restartNotice && pendingRestartNotice === restartNotice) {
                pendingRestartNotice = '';
                addMessage(restartNotice, 'system', false, null, false, { ephemeral: true, systemType: 'reconnect' });
            }
            clearReloadParams();
        };
        const read = loadUiPreferences()
            // A return shares its paint receipt's newest read during preferences, even a refusal; a real reconnect owes
            // a read newer than any it interrupted.
            .then(() => (destroyed || openedVisit !== visit ? false
                : msg?.resumed && recentRequest > readOnOpen ? historySyncPromise || historyPager.whenIdle()
                : isReconnect ? historyPager.read({ fromReconnect: true, whole: true, reason, current: () => catchUp.reading === read })
                    : waitForHydrationWindow().then(awaitInitialHydration)))
            .then(settled, settled);
        if (isReconnect) catchUp.begin(() => !destroyed && openedVisit === visit, read, readOnOpen);
    });

    // A disconnect: written sends not yet saved say so and the room shows the outage; away, which sends it doubts is
    // decided now (custody) and both are drawn on its return (`stay`).
    onWs('close', () => {
        forgetWindowReturn();
        closedSinceOpen = true;
        catchUp.cancel(recentRequest);
        if (leaving) { outageAway = [...outageAway || [], ...unconfirmed.written()]; return; }
        showOutage();
    });
    onWs(SERVER_ANSWERS_EVENT, () => syncChatStatus());
    roomProfile.listen(onWs);  // a socket's proof (an owed saved text, kept sends), the registry's word on the room
    function showOutage(doubted) {
        unconfirmed.unsettle(doubted);
        workState = reconcileWorkState(workState, null, chatId);
        mainWaiting?.setConnected(false);
        withStableViewport(() => {
            for (const [id, activity] of activeDirectActivities) {
                activeDirectActivities.set(id, { ...activity, _activityUnconfirmed: true });
            }
            let changed = false;
            for (const [id, record] of liveCardRecords) {
                if (isPortal(record)) continue;
                record.childDetailRead = undefined;
                if (record.isSubagent) record.parkedOffline = true;  // this client's own park, not the server's view
                changed = syncParkedPhase(record, 'unknown', activeDirectActivities.get(id)) || changed;
            }
            return changed;
        });
        hostStart.forget();
        portals.retire();
        hideTypingIndicatorOnly();
        syncChatStatus();
        syncHeaderControlState({ accounting: { available: false } });
    }

    const revealQuestion = (taskId, quizId, options) => chatDecision.revealQuestion(
        taskId, quizId, projectId, chatId, appendQuizMessage, isInstanceVisible,
        () => { const current = reading.claim(); reading.stick = false; return current; }, reading.scroll, options);
    const historyReveal = createHistoryReveal({ pager: historyPager, reading, feed: messagesDiv, records: liveCardRecords,
        present: isInstanceVisible, expand: discloseLiveCard,
        hydrate: hydrateShownDetails, pin: cardRowAnchors.pin });
    roomWork = createRoomWork({ page, asPanel, ws, statusBadge, messagesDiv, reading, stateSnapshots, revealQuestion,
        works: () => workState, liveCardRecords, pendingSubmissions, pendingSuggestedNames,
        cancelRunFromCard, refreshHeaderControlState, captureVisibleTimelineAnchor, restoreVisibleTimelineAnchor,
        historyPager, showLatest, hostReady: hostStart.ready, hostBoot: () => hostStart.status().bootPhase });
    // Main's bar: Restart, Panic, the budget and the consciousness chip; a Project's bar is app.js's.
    header = asPanel ? null : createRoomHeader({ bar: pageHeader, ws, shade: roomWork, openDashboardTab,
        refreshState: refreshHeaderControlState });

    return self = {
        page,
        chatId,
        projectId,
        // The room's composer (composer.js): its draft, its "+" menu and its voice and chips slots.
        composer,
        showLatest,
        resume,
        ws,
        reflowViewport,
        captureReading: () => reading.capture(),
        showHistoryNotice: () => { if (!destroyed) historyControls.showNotice(); },
        refreshHistory: readReceipt.refresh,
        revealQuestion,
        revealSource: replies.reveal,
        drawsSource: replies.draws,
        search: replies.search,
        cancelHistoryPaint: readReceipt.cancel,
        // app.js offers each room its /api/state snapshots (it polls); a room away takes none, catching up on return.
        hydrateStateSnapshot,
        // True once a history snapshot has actually been fetched and painted;
        // app.js uses it to decide whether a reopen needs a forced repaint.
        hasPaintedHistory: () => historyLoaded && lastHistorySyncSucceeded,
        // That, or this browser's saved text of the room (chat_preview.js): the docked panel comes in with either.
        hasPaintedWords: () => (historyLoaded && lastHistorySyncSucceeded) || preview.shown(),
        archiveOpen: () => archive?.active,
        // Unsendable state (staged Files, an upload in flight, an unsaved send and its Send again): room_retire.js keeps
        // such a room even when its project goes away.
        hasPendingWork: () => composer.hasPendingWork() || unconfirmed.count > 0,
        // Another place covers the room: a reader or file dialog its card opened closes, its place is kept; `pause`
        // (another place replaces it) also pauses what plays. Staged files stay.
        closeTransient: (pause) => { archive?.close(); roomWork?.leave(); keepPlace(); chatMedia.closeTransient(pause);
            historyControls.hideNotice(); },
        // Away keeps the place, pausing paint and reads (room_retire.js).
        forgetWindowReturn,
        leave: (windowHidden = false) => {
            if (leaving) return;
            if (windowHidden) windowReturn.hold(windowFollower() && reading.nearBottom(), presentHeld());
            roomProfile.left();  // copies shown on the core's word leave the feed, kept (chat_preview.js)
            catchUp.cancel(recentRequest);  // its return reads, and says so, anew
            // Left in an outage (down after the page's socket opened), it will not hear the reconnect that forgets a
            // readiness polled meanwhile. Before the first open there is none: that open keeps the boot read.
            if (ws.isConnected?.() === false && ws.previouslyConnected?.()) hostStart.forget();
            roomWork?.leave(); keepPlace(); leaving = true; visit += 1; historyControls.hideNotice();
            readReceipt.cancel(); historyReadAbort?.abort(); historyPager.suspend(); historyResyncScheduler.cancel();
            rowSleep.destroy(); chatResizeObserver?.disconnect(); reviewHydrator.suspend(); replies.search.close();
        },
        stay: () => {
            if (!leaving || destroyed) return;
            leaving = false;
            rowSleep = createRowSleep({ feed: messagesDiv, onRestyle: () => reading.reflow() });
            installChatResizeObservers();
            reviewHydrator.resume(); drawOwedTimelines();
            if (outageAway) { const doubted = outageAway; outageAway = null; showOutage(doubted); }
            roomProfile.stayed();  // an opening's saved text owed since paints at once (chat_preview.js)
            if (isMain || state.projectChatIds.has(chatId)) earlyRows.show(windowReturn.take(windowFollower()));
            else forgetWindowReturn();
        },
        // How the room was opened (app.js): pressed open by a pointer, its focus corners wait for a key.
        quietFocus: (quiet) => messagesDiv.classList.toggle('is-quiet-focus', Boolean(quiet)),
        // Release resources before removing the DOM; late callbacks see `destroyed`.
        destroy() {
            if (destroyed) return;
            windowReturn.drop(); earlyRows.destroy();
            destroyed = true;
            catchUp.cancel();
            historyReadAbort?.abort();
            composer.destroy();
            replies.destroy();
            rowMenu.destroy();
            rowSleep.destroy();
            offRefit();
            unconfirmed.release();
            emptyWelcome?.dispose();
            readReceipt.cancel();
            sharedRead.dispose();
            for (const dispose of wsDisposers) {
                try { dispose(); } catch {}
            }
            wsDisposers.length = 0;
            historyControls.endRecent();
            historyControls.destroy(); coverWatch?.disconnect();
            window.removeEventListener('ouro:language-changed', handleHistoryLanguage);
            historyPager.destroy();
            disposeReadingGestures();
            document.removeEventListener('selectionchange', retryHistoricalUpserts);
            chatMedia.destroy();
            roomWork?.destroy();
            chatDecision.destroy();
            modelWaits.destroy();
            mainWaiting?.destroy();
            document.removeEventListener('visibilitychange', handleVisibility);
            header?.destroy();
            focusPanel?.removeEventListener('pointerdown', quietFocus);
            focusPanel?.removeEventListener('keydown', showFocus);
            inputArea?.removeEventListener('ouro:composer-reserve', syncComposerReserve);
            chatResizeObserver?.disconnect();
            chatResizeObserver = null;
            unwatchEdges();
            historyResyncScheduler.cancel();
            preview.destroy();
            for (const id of liveCardRecords.keys()) disposeLiveCard(id);
            unfollowCatalogue();
            reviewHydrator.clear();
            // The buffered collections go last; a page replay never clears the subagent lineage, only this does.
            for (const held of [explicitCardExpansion, reviewDisclosureByTask, pendingSuggestedNames, subagentChildParents,
                subagentTerminalChildren, cancelableTaskIds, missingManagedTaskIds, managedTaskDetailReads, pendingUserBubbles,
                localEchoJournal, seenMessageKeys, pageHistoryIds, historicalTerminals, pendingHistoryEvictions,
                pendingHistoryUpserts, pendingLiveEvictions]) held.clear();
            messageKeyOrder.length = 0;
            recentCoverage = null;
            try { destroyChatMarkdown(page); page.remove(); } catch {}
        },
    };
}
