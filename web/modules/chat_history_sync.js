// The history/feed owner for a chat instance: durable-history hydration and
// replay (syncHistory with its rebuildAll batch), the feed mount primitives
// (insertMessageNode, addMessage), the sessionStorage bootstrap and snapshot,
// the Load-older control and the socket-open resync. Ownership transfer from
// chat.js (v7 W3 wave D): the per-instance closure bodies move here with
// their captured collections and collaborator members lifted to explicit
// factory parameters of the same names; the hydration/replay flags
// (historyLoaded, the sticky single-flight promise, the replay-batch handle,
// the reconnect intents and the Load-older quotas) become factory state, and
// chat.js reads the replay-batch handle back through getRebuildBatch.

import {
    escapeHtmlAttr,
    escapeHtmlText as escapeHtml,
    renderMarkdown,
} from './utils.js';
import { apiFetch } from './api_client.js';
import {
    createHistoryResyncScheduler,
    createRebuildBatch,
    insertTimelineNode,
    loadOlderControlState,
    nextQuotaEscalation,
} from './chat_render_batch.js';
import { partitionLocalEchoJournal, reconnectBannerText } from './chat_activity.js';
import { taskOutcomeSeverity, taskTerminalPhase } from './log_events.js';
import { renderSkillReviewDisclosure, wireSkillReviewDisclosure } from './skill_review_card.js';

const CHAT_STORAGE_KEY = 'ouro_chat';

export function createChatHistorySync({
    ws,
    isMain,
    chatId,
    page,
    messagesDiv,
    storeKey,
    chatSessionId,
    initialScrollPending,
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
    isViewportSticky,
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
    setInputHistoryIndex,
}) {

    let pendingReconnectSync = false;  // Set when a fromReconnect sync arrives while one is already in-flight.
    let pendingReconnectBannerText = readPendingReconnectBanner();
    // perf2 P4 follow-up (double-fetch fix): true while syncHistory replays the
    // fetched rows into cards — pass 1, pass 2 AND the terminal-resolution
    // sweep (both the rebuildAll and the routine branch). Finished transitions
    // raised inside that replay must not schedule the 700ms post-completion
    // resync: the data just came from the canonical source. The replay block
    // is fully synchronous, so no live WS frame can ever observe the flag.
    let _historyReplayActive = false;
    let historyLoaded = false;
    let inputHistorySeededFromServer = false; // set true only after a successful server-side recall seed
    let historySyncPromise = null;
    let lastHistorySyncSucceeded = false;
    let historyPaintGeneration = 0;
    // perf2 P4.1 [GPT#12 + Fable#1]: STICKY single-flight hydration promise.
    // Unlike historySyncPromise it survives success, so hydration triggers
    // (bootstrap IIFE, first non-reconnect socket open, refreshHistory without
    // a new revision) short-circuit instead of refetching. Any FAILED sync
    // resets it; scheduleHistorySync and the reconnect path never consult it.
    let initialHydrationPromise = null;
    // perf2 P4.1 [GPT#17]: the offline bootstrap painted the sessionStorage
    // fallback and set historyLoaded=true — the first successful sync after
    // the server comes back must still rebuild the feed from durable history.
    let offlineBootstrapPainted = false;
    // perf2 P4.1: highest project revision whose history has been fetched;
    // refreshHistory only bypasses the sticky promise for a NEWER revision.
    let lastLoadedHistoryRevision = 0;
    // perf2 P4.2: one-shot idle gate for Main's deferred first hydration.
    let hydrationGatePromise = null;
    // perf2 P4.3: non-null only inside a rebuildAll replay — routes per-row
    // feed insertion / meta / count / layout / typing / status / persist
    // through one end-of-batch application. Routine syncs never set it.
    let _rebuildBatch = null;
    // perf2 P4.5: server window verdict + the explicit Load-older quotas.
    let historyWindow = null;
    let historyQuotaOverride = null;
    let loadingOlderHistory = false;
    let welcomeShown = false;
    // Seeded by chat.js from the stashed scroll state (single-live-panel).
    let _initialScrollPending = initialScrollPending;
    // Mirrors the instance's destroyed latch so late async continuations no-op.
    let destroyed = false;


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

    function persistVisibleHistory() {
        try {
            sessionStorage.setItem(storeKey(CHAT_STORAGE_KEY), JSON.stringify(persistedHistory.slice(-200)));
        } catch {}
    }

    function insertMessageNode(node, options = {}) {
        if (!node) return;
        // perf2 P4.3 (rebuildAll only): collect into the detached batch. One
        // stable sort + one fragment mount replace per-row chronological
        // insertion; the end-of-sync anchor restore replaces the per-row
        // insertedAboveViewport compensation. Routine syncs and live frames
        // (batch inactive) keep the chronological insertTimelineNode path.
        if (_rebuildBatch) {
            _rebuildBatch.collect(node);
            return;
        }
        const shouldStick = Boolean(options.forceStick) || isNearBottom();
        const isMounted = node.parentNode === messagesDiv;
        if (isMounted && !options.reorderExisting) {
            if (shouldStick) messagesDiv.scrollTop = messagesDiv.scrollHeight;
            updateScrollButton();
            return;
        }
        const reorderAnchor = isMounted && !shouldStick
            ? captureVisibleTimelineAnchor(node)
            : null;
        // Scope to THIS instance's column — a global id lookup would resolve to
        // the first panel's typing node and misplace project-thread messages.
        const typing = messagesDiv.querySelector('.typing-bubble');
        insertTimelineNode(messagesDiv, node, typing, { stickToBottom: shouldStick });
        if (reorderAnchor) restoreVisibleTimelineAnchor(reorderAnchor);
        // A new message arriving while the user is scrolled up reveals the
        // jump-to-newest button instead of silently piling up off-screen.
        updateScrollButton();
    }

    function addMessage(text, role, markdown = false, timestamp = null, isProgress = false, opts = {}) {
        const pending = !!opts.pending;
        const ephemeral = !!opts.ephemeral;
        const clientMessageId = opts.clientMessageId || '';
        const senderLabel = opts.senderLabel || '';
        const senderSessionId = opts.senderSessionId || '';
        const source = opts.source || '';
        const systemType = opts.systemType || '';
        const taskId = opts.taskId || '';
        const ts = timestamp || new Date().toISOString();
        const messageKey = buildMessageKey(role, text, ts, {
            clientMessageId,
            systemType,
            isProgress,
            source,
            senderLabel,
            senderSessionId,
            taskId,
        });
        if (messageKey && seenMessageKeys.has(messageKey)) return null;

        if (!isProgress && !ephemeral) {
            persistedHistory.push({
                text,
                role,
                ts,
                markdown: !!markdown,
                systemType,
                source,
                senderLabel,
                senderSessionId,
                clientMessageId,
                taskId,
                skillReview: opts.skillReview || null,
            });
            // Mirror the sessionStorage slice(-200): the in-memory copy exists
            // only to feed that snapshot, so it obeys the same cap (P3).
            if (persistedHistory.length > 200) {
                persistedHistory.splice(0, persistedHistory.length - 200);
            }
            // perf2 P4.3: a rebuildAll replay serializes the sessionStorage
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

        const sender = getSenderLabel(role, isProgress, systemType, { source, senderLabel, senderSessionId });
        const rendered = role === 'user'
            ? escapeHtml(text)
            : (role === 'system' && systemType === 'skill_review'
                ? renderSkillReviewDisclosure(text, opts.skillReview || null)
                : renderMarkdown(text));
        const timeFmt = formatMsgTime(ts);
        const timeHtml = timeFmt ? `<div class="msg-time" title="${escapeHtmlAttr(timeFmt.full)}">${escapeHtml(timeFmt.short)}</div>` : '';
        const pendingHtml = pending ? `<div class="msg-pending">Queued until reconnect</div>` : '';
        bubble.innerHTML = `
            <div class="sender">${escapeHtml(sender)}</div>
            <div class="message">${rendered}</div>
            ${pendingHtml}
            ${timeHtml}
        `;
        wireSkillReviewDisclosure(bubble, () => requestAnimationFrame(() => !destroyed && updateMessagesPadding({ preserveStickiness: true })));
        stampNodeTimestamp(bubble, ts);
        insertMessageNode(bubble, { forceStick: !!opts.forceStick });
        renderRoutingAnnotation(bubble, opts.chatAnnotation);
        rememberMessageKey(messageKey);
        if (pending && clientMessageId) pendingUserBubbles.set(clientMessageId, bubble);
        return bubble;
    }

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

    // perf2 P4.1 [GPT#12 + Fable#1]: sticky single-flight for HYDRATION
    // triggers ONLY — bootstrap IIFE, the first non-reconnect socket open, and
    // refreshHistory without a new revision. scheduleHistorySync (the 700ms
    // post-completion resync) and the reconnect path NEVER short-circuit here:
    // a lost task_done is healed only by a real refetch (their coalescence is
    // historySyncPromise). Any failed sync resets the sticky promise so the
    // next trigger fetches for real.
    function awaitInitialHydration({ includeUser = false } = {}) {
        if (initialHydrationPromise) return initialHydrationPromise;
        initialHydrationPromise = syncHistory({ includeUser });
        return initialHydrationPromise;
    }

    // perf2 P4.2: Main's first hydration waits for an idle slot and yields to
    // an opening project panel, but only within an UNCONDITIONAL upper bound
    // [GPT#16] — an hour-open panel must not defer hydration forever. Project
    // instances hydrate immediately. One-shot: live frames rendered before
    // hydration are rebuilt by the first rebuildAll replay.
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

    // perf2 P4.3: the deferred per-card finals, applied exactly once after the
    // batch mount — meta/count/layout per touched card, typing and composer
    // status once per batch, ONE sessionStorage persist for the whole replay.
    function finalizeRebuildBatch(batch) {
        for (const record of batch.touched) {
            renderLiveCardMeta(record);
            updateLiveCardCount(record);
            syncLiveCardLayout(record);
        }
        if (batch.typingHidden) hideTypingIndicatorOnly();
        if (batch.status) {
            // Post-mount truth: an unfinished mounted foreground card keeps
            // the composer on "Working..." exactly like the live path, where
            // hasActiveLiveCard() sees connected roots during replay.
            if (hasActiveLiveCard()) setStatus('thinking', 'Working...');
            else setStatus(batch.status.kind, batch.status.text);
        }
        persistVisibleHistory();
    }

    async function syncHistory({ includeUser = false, fromReconnect = false, forceRebuild = false } = {}) {
        if (historySyncPromise) {
            // Preserve reconnect intent so retiredTaskIds is cleared after this sync.
            if (fromReconnect) {
                pendingReconnectSync = true;
                return historySyncPromise.then(() => {
                    // The first reconnect waiter consumes the queued rebuild; any
                    // concurrent waiter sees and awaits the newly installed global
                    // promise. No caller may render its Reconnected banner against
                    // the intermediate (non-rebuilt) DOM.
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
            try {
                // Default request sends NO quota params — the server's window
                // constants govern the first-load window (perf2 P3). A Load-
                // older escalation adds explicit n_human/n_progress (perf2 P4).
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
                // perf2 P4.5: the server's window verdict (P3.2 additive field)
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
                // perf2 P4: a Load-older refetch (forceRebuild) and the first
                // successful sync after an offline sessionStorage bootstrap
                // [GPT#17] rebuild fully too.
                const rebuildAll = !historyLoaded || fromReconnect || forceRebuild
                    || offlineBootstrapPainted;
                // On a soft reconnect the module (and its dedupe set) survives, so a
                // plain re-sync would skip user messages and dedupe-drop every
                // assistant bubble — the conversation would vanish. Restore user text
                // and rebuild from durable history whenever we rebuild. The
                // offline-bootstrap rebuild clears the fallback-painted bubbles
                // too, so it must restore user rows even when the trigger came
                // with includeUser=false (first clean open / 700ms resync).
                const renderUser = includeUser || fromReconnect || offlineBootstrapPainted;
                if (!historyLoaded || fromReconnect) retiredTaskIds.clear();
                // The extra rebuild causes (Load-older / offline bootstrap)
                // replay everything too, so retirement resets with them.
                if (rebuildAll) retiredTaskIds.clear();

                // perf2 P4.3: the ENTIRE mutation below (clear -> pass 1 ->
                // pass 2 -> terminal resolution -> sweep) is one synchronous
                // closure. On rebuildAll it runs inside ONE outer
                // withStableViewport with a detached batch collecting the
                // top-level nodes; NO awaits may occur between the feed
                // clearing and the batch mount [GPT#14]. The routine path
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
                    for (const record of liveCardRecords.values()) record.root?.remove();
                    liveCardRecords.clear();
                    taskUiStates.clear();
                    ephemeralDecisionTaskIds.clear();
                    // Rebuild replays the durable truth: stale name buffers and
                    // cancelable markers from the previous connection are dropped
                    // and re-learned from history rows (P3 growth caps).
                    pendingSuggestedNames.clear();
                    cancelableTaskIds.clear();
                    setActiveLiveGroupId('');
                    // Atomically drop the standalone message bubbles and the dedupe
                    // state so the rebuild below cannot produce duplicates even if
                    // stale bubbles lingered in the DOM. Keep the typing indicator.
                    for (const bubble of Array.from(messagesDiv.querySelectorAll('.chat-bubble'))) {
                        if (!bubble.classList.contains('typing-bubble')) bubble.remove();
                    }
                    seenMessageKeys.clear();
                    messageKeyOrder.length = 0;
                    // Subagent lineage + terminal state live only in memory. Clear and
                    // rebuild them from durable history BEFORE the card passes, so a
                    // finished child card finalizes regardless of replay order or which
                    // event carried the terminal signal (a subagent 'completed' event OR
                    // a server task_terminal_status). Otherwise finished children stick
                    // on "working" and get revived by parent heartbeats on reload.
                    subagentChildParents.clear();
                    subagentTerminalChildren.clear();
                    for (const msg of messages) {
                        if (String(msg.delegation_role || '').toLowerCase() !== 'subagent') continue;
                        const parentId = String(msg.parent_task_id || '').trim();
                        const childId = String(msg.subagent_task_id || msg.task_id || '').trim();
                        if (!parentId || !childId || parentId === childId) continue;
                        if (!subagentChildParents.has(childId)) {
                            setSubagentParent(childId, { parentId, role: String(msg.subagent_role || '').trim(), model: msg.model });
                        }
                        const ev = String(msg.subagent_event || '').toLowerCase();
                        if (msg.task_terminal_status || ['completed', 'completed_warn', 'failed', 'cancelled', 'rejected'].includes(ev)) {
                            subagentTerminalChildren.add(childId);
                        }
                    }
                }

                // Two passes ensure cards exist before finishLiveCard() marks them done.

                // Pass 1 builds timelines with DOM insertion suppressed.
                setSyncPass1Active(true);
                try { for (const msg of messages) {
                    const taskId = msg.task_id || '';
                    if (!taskId) continue;
                    if (retiredTaskIds.has(taskId)) continue;
                    if (msg.is_progress) {
                        updateLiveCardFromProgressMessage(msg);
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
                } } finally { setSyncPass1Active(false); }

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
                    // A delivered document is a media bubble, not a task-final
                    // message — render it BEFORE the taskId/finishLiveCard block so
                    // a mid-task file delivery replayed while its task is still
                    // running does not falsely finalize that task's live card.
                    if (msg.msg_type === 'document') {
                        appendDocumentBubble(msg);
                        continue;
                    }
                    if (taskId && (msg.role === 'assistant' || msg.role === 'system')) {
                        if (subagentChildParents.has(taskId)) {
                            insertCardIfNeeded(taskId);
                            routeSubagentFinalMessageToCard(taskId, msg);
                            const taskState = getTaskUiState(taskId, false);
                            const record = liveCardRecords.get(taskId);
                            const preservedPhase = taskState?.completedPhase || record?.phaseEl?.dataset?.phase || 'done';
                            finishLiveCard(taskId, preservedPhase);
                            continue;
                        }
                        insertCardIfNeeded(taskId);
                        // A replayed early final must not finalize the card.
                        if (msg.task_phase === 'finalizing') {
                            markLiveCardFinalizing(taskId);
                        } else {
                            const taskState = getTaskUiState(taskId, false);
                            const record = liveCardRecords.get(taskId);
                            const preservedPhase = taskState?.completedPhase || record?.phaseEl?.dataset?.phase || 'done';
                            finishLiveCard(taskId, preservedPhase);
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
                    const status = String(terminalRecord.status || '');
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

                // perf2 P4 follow-up (double-fetch fix): the replay below marks
                // historical cards finished; those transitions must not
                // schedule the post-completion resync (the rows just arrived
                // from this very fetch). The flag spans BOTH branches and is
                // dropped synchronously, so a real live completion frame can
                // never land while it is up.
                _historyReplayActive = true;
                try {
                    if (rebuildAll) {
                        // perf2 P4.3 [GPT#14]: one outer withStableViewport for the
                        // whole rebuild — inner per-row wrappers collapse on the
                        // existing _viewportMutationDepth gate, killing the
                        // per-frame isInstanceVisible/anchor layout storm. One
                        // stable sort, one fragment mount before typing, then the
                        // per-card finals and ONE persist. The whole section is
                        // synchronous: live frames can never observe "records
                        // cleared, fragment not yet mounted".
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
                        // Routine sync: the old per-row live-DOM path, untouched.
                        applySyncedMessages();
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
                    saveInputHistory(inputHistory);
                    setInputHistoryIndex(inputHistory.length);
                    inputHistorySeededFromServer = true;
                }

                const wasFirstLoad = !historyLoaded;
                historyLoaded = true;
                lastHistorySyncSucceeded = true;
                // The durable rebuild superseded the offline fallback paint.
                offlineBootstrapPainted = false;
                // perf2 P4.1: ANY successful sync leaves the instance hydrated
                // — later hydration triggers ride this sticky promise.
                initialHydrationPromise = historySyncPromise;
                // perf2 P4.5: reflect the server's window verdict in the
                // Load-older control now that the feed matches this response.
                syncLoadOlderControl();
                // A recreated project instance restores its predecessor's stashed
                // mid-history position on first paint instead of pinning to newest.
                if (wasFirstLoad && _initialScrollPending) {
                    _initialScrollPending = false;
                    updateMessagesPadding({ preserveStickiness: false });
                    restoreScrollPosition();
                } else
                // First load jumps to latest; reconnect preserves older-message reading.
                if (wasFirstLoad || (fromReconnect ? scrollBeforeSync.nearBottom : isNearBottom())) {
                    updateMessagesPadding({ preserveStickiness: false });
                    scrollToBottomAfterLayout();
                } else if (fromReconnect) {
                    // Rebuild may add rows both ABOVE and BELOW the viewport; a
                    // scrollHeight delta cannot tell them apart and over-scrolls
                    // readers. Restore the first visible timestamped node to its
                    // prior visual offset instead (equal-ts ordinals keep
                    // arrival-order identity), after two RAF frames so async
                    // card heights above the anchor cannot move the reader.
                    await new Promise((resolve) => requestAnimationFrame(
                        () => requestAnimationFrame(resolve)
                    ));
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
        // perf2 P4.1: only a NEW revision (or a never-hydrated instance)
        // forces a real fetch; otherwise the sticky hydration promise answers
        // and the paint receipt below still runs [GPT#12].
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
        // would otherwise acknowledge a revision that was never shown (GPT#15).
        await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
        return {
            painted: !destroyed && generation === historyPaintGeneration && !page.hidden,
            revision: targetRevision,
        };
    }

    function scheduleHistorySync() {
        historyResyncScheduler.schedule();
    }

    // perf2 P4 follow-up (double-fetch fix): finished transitions replayed by
    // syncHistory itself (_historyReplayActive) are dropped by the scheduler —
    // the rows just arrived from the canonical source, so the 700ms resync was
    // refetching the whole window after every history load. A LIVE completion
    // (WS frame outside a replay) still always schedules a REAL fetch [GPT#12].
    const historyResyncScheduler = createHistoryResyncScheduler({
        isReplayActive: () => _historyReplayActive,
        run: () => syncHistory({ includeUser: false }).catch(() => {}),
    });
    // perf2 P4.5: "Load older" control at the very top of the feed. Server
    // truth (window.complete / truncated_by from P3.2) decides between a
    // quota-escalating refetch button and the honest boundary notice; the
    // container class is excluded from viewport anchoring like .typing-bubble.
    // The control is mounted ONLY while it has something to show: a
    // permanently-present (even hidden) node would be an extra top-level feed
    // child, breaking child-order consumers (ui-smoke chronology pattern) and
    // diverging from the pre-P4 feed layout on complete windows.
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
        if (control.mode === 'hidden') {
            loadOlderEl.remove();
            return;
        }
        if (!loadOlderEl.isConnected) messagesDiv.prepend(loadOlderEl);
        loadOlderBtn.hidden = control.mode !== 'button';
        loadOlderBtn.disabled = loadingOlderHistory;
        loadOlderBtn.textContent = loadingOlderHistory
            ? 'Loading…'
            : (control.mode === 'button' ? control.label : 'Load older messages');
        loadOlderNote.hidden = control.mode !== 'notice';
        if (control.mode === 'notice') loadOlderNote.textContent = control.label;
    }

    async function loadOlderHistory() {
        if (loadingOlderHistory) return;
        const next = nextQuotaEscalation(historyQuotaOverride);
        if (!next) return;
        loadingOlderHistory = true;
        syncLoadOlderControl();
        // Anchor the current first visible timestamped node (the control
        // itself is excluded from capture, like .typing-bubble) so the reader
        // does not drift when older rows land above the viewport [GPT#13].
        const anchor = isViewportSticky() || isNearBottom() ? null : captureVisibleTimelineAnchor();
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

    let wsHasConnectedOnce = false;

    function handleSocketOpen(msg) {
        // Reconnect drops kind-less entries (no snapshot source tracks them);
        // kind-stamped ones reconcile against the refreshed snapshot below.
        for (const [aid, entry] of activeDirectActivities) {
            if (!entry.kind) activeDirectActivities.delete(aid);
        }
        refreshHeaderControlState(true);
        syncChatStatus();
        // perf2 P4.1 [Gemini#3]: reconnect truth comes from the ws CLIENT
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
    }

    // The bootstrap paint: durable history first, sessionStorage fallback
    // second — exactly the IIFE chat.js used to run at instance construction.
    (async () => {
        await loadUiPreferences();
        // perf2 P4.2: Main waits for the (bounded) idle hydration window;
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
                    skillReview: msg.skillReview || null,
                });
            }
        } catch {}
        historyLoaded = true;
        // GPT#17: this offline fallback sets historyLoaded=true, which would
        // make the first successful post-outage sync a NON-rebuilding routine
        // fold over stale sessionStorage bubbles. Flag it so that sync
        // rebuilds from durable history instead.
        if (!lastHistorySyncSucceeded) offlineBootstrapPainted = true;
        ensureWelcomeMessage();
    })();

    return {
        insertMessageNode,
        addMessage,
        scheduleHistorySync,
        cancelHistoryPaint,
        refreshHistory,
        hasPaintedHistory: () => historyLoaded && lastHistorySyncSucceeded,
        handleSocketOpen,
        getRebuildBatch: () => _rebuildBatch,
        cancelPendingHistoryResync: () => historyResyncScheduler.cancel(),
        markHistoryDestroyed: () => { destroyed = true; },
    };
}
